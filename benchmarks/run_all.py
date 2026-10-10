"""Reproducible Vyom benchmark suite. Writes raw JSON to benchmarks/results/.

Usage (from repo root):
    PYTHONPATH="src;." python benchmarks/run_all.py                 # everything
    PYTHONPATH="src;." python benchmarks/run_all.py --only reasoning planning
    PYTHONPATH="src;." python benchmarks/run_all.py --quick         # fewer steps/seeds (smoke)

Sections
--------
world_model  Train WorldStatePredictor on real GridWorld transitions (3 seeds); held-out maps.
planning     Learned-model planner vs baselines on 200 held-out maps.
reasoning    Multi-hop pointer-chasing ablation: A, A-deep (param-matched), B, C, D (3 seeds).
adaptive     Accuracy / steps / latency for fixed steps vs adaptive halting (trained C, T=6).
memory       Retrieval top-1 accuracy: MiniLM vs hashed-BoW, plus user-isolation check.
lm           DialoGPT-small: perplexity A vs gated (gate 0) vs legacy prefix; chat latency;
             KV-cache vs recompute; peak RSS.

No number in the docs is produced any other way. Failures are recorded, not hidden.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from neurograph.config import NeuroGraphConfig  # noqa: E402
from neurograph.model import NeuroGraphModel  # noqa: E402


def env_info() -> Dict[str, Any]:
    import psutil
    import transformers

    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "threads": torch.get_num_threads(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1), "cuda": torch.cuda.is_available(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def save(name: str, payload: Dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"benchmark": name, "env": env_info(), **payload}
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2))
    print(f"[saved] benchmarks/results/{name}.json")


def mean_std(xs: List[float]) -> Dict[str, float]:
    return {"mean": round(statistics.mean(xs), 4), "std": round(statistics.pstdev(xs), 4) if len(xs) > 1 else 0.0,
            "values": [round(x, 4) for x in xs]}


# =============================================================== world model
def bench_world_model(quick: bool) -> None:
    from neurograph.world_model_training import save_world_model, train_world_model

    runs = []
    for seed in ([0] if quick else [0, 1, 2]):
        model, rep = train_world_model(epochs=8 if quick else 30, seed=seed)
        runs.append(rep)
        if seed == 0:
            save_world_model(model, ROOT / "models" / "world_model_gridworld.pt", rep)
    save("world_model", {
        "task": "GridWorld 6x6, density 0.2, random-walk transitions; train maps 0-199, test maps 1000-1099",
        "runs": runs,
        "summary": {
            "test_position_accuracy": mean_std([r["test"]["position_accuracy"] for r in runs]),
            "test_mse": mean_std([r["test"]["mse"] for r in runs]),
            "copy_baseline_position_accuracy": runs[0]["test"]["copy_baseline_position_accuracy"],
            "untrained_position_accuracy": runs[0]["untrained_test"]["position_accuracy"],
        },
    })


# ================================================================== planning
def bench_planning(quick: bool) -> None:
    from environments.gridworld import GridWorld
    from neurograph.config import PlannerConfig, WorldModelConfig
    from neurograph.planner import ModelBasedRolloutPlanner, SearchBasedPlanner, WorldModelSearchPlanner
    from neurograph.world_model import WorldStatePredictor
    from neurograph.world_model_training import load_world_model

    wm, _ = load_world_model(ROOT / "models" / "world_model_gridworld.pt")
    torch.manual_seed(0)
    untrained = WorldStatePredictor(WorldModelConfig(state_dim=32)).eval()
    seeds = range(1000, 1050 if quick else 1200)
    max_steps = 30

    def greedy(env):
        rng = random.Random(0)
        obs, info = env.reset()
        for t in range(max_steps):
            r, c = env.current_pos
            gr, gc = env.goal_pos
            valid = [a for a in env.get_valid_actions() if a != 4]
            best = min(abs(r + GridWorld.ACTION_MAP[a][0] - gr) + abs(c + GridWorld.ACTION_MAP[a][1] - gc) for a in valid)
            a = rng.choice([a for a in valid if abs(r + GridWorld.ACTION_MAP[a][0] - gr) + abs(c + GridWorld.ACTION_MAP[a][1] - gc) == best])
            _, _, done, info = env.step(a)
            if info["reached_goal"]:
                return {"success": True, "steps": t + 1}
            if done:
                break
        return {"success": False, "steps": max_steps}

    def bfs(env):
        p = SearchBasedPlanner(GridWorld.ACTION_MAP).plan_bfs(env.start_pos, env.goal_pos, env.is_valid_coord, max_steps)
        return {"success": bool(p), "steps": len(p)}

    planners = {
        "bfs_oracle_full_map": bfs,
        "greedy_manhattan_local_sensors": greedy,
        "legacy_rollout_planner_trained_wm": lambda env: ModelBasedRolloutPlanner(
            PlannerConfig(), world_model=wm, policy_model=None, action_dim=5).execute_plan(env, max_steps),
        "world_model_search_untrained_wm": lambda env: WorldModelSearchPlanner(untrained, 6, 6).execute_plan(env, max_steps),
        "world_model_search_trained_wm": lambda env: WorldModelSearchPlanner(wm, 6, 6).execute_plan(env, max_steps),
    }
    results: Dict[str, Any] = {}
    for density in (0.2, 0.35):
        envs = [GridWorld.generate_random_map(6, 6, density, seed=s, state_dim=32) for s in seeds]
        empty = sum(1 for e in envs if not e.obstacles)
        res_d: Dict[str, Any] = {"maps_without_obstacles_fallback": empty}
        for name, fn in planners.items():
            succ, steps, lat, pred_acc = 0, [], [], []
            for s in seeds:
                env = GridWorld.generate_random_map(6, 6, density, seed=s, state_dim=32)
                env.max_steps = max_steps
                t0 = time.perf_counter()
                res = fn(env)
                lat.append((time.perf_counter() - t0) * 1000)
                succ += int(res["success"])
                if res["success"]:
                    steps.append(res["steps"])
                if "prediction_accuracy" in res:
                    pred_acc.append(res["prediction_accuracy"])
            res_d[name] = {
                "success_rate": round(succ / len(seeds), 4),
                "mean_steps_when_successful": round(statistics.mean(steps), 2) if steps else None,
                "latency_ms_mean": round(statistics.mean(lat), 3),
                "latency_ms_p95": round(sorted(lat)[int(0.95 * len(lat)) - 1], 3),
                "online_prediction_accuracy": round(statistics.mean(pred_acc), 4) if pred_acc else None,
            }
            print(density, name, res_d[name])
        results[f"density_{density}"] = res_d
    save("planning", {"maps": [seeds.start, seeds.stop - 1], "max_steps": max_steps,
                      "world_model_trained_on_density": 0.2, "results": results,
                      "notes": "BFS reads the obstacle map (oracle). All other planners observe only the state "
                               "vector (position, goal, 4 local wall sensors)."})


# ================================================================= reasoning
ARROW, SEP, QUERY, HOP1, HOP2 = 1, 2, 3, 4, 5
ENT0, NENT = 10, 50
VOCAB = ENT0 + NENT
GATE_INIT = 1.0  # from-scratch LM: no pretrained behaviour to protect, so the gate starts open


def make_batch(n: int, k: int, hops, g: torch.Generator):
    """Random k-cycle over fresh entities given as shuffled 'a -> b' facts.

    Sequence: facts..., QUERY, HOP<h>, x  -> target: h-th successor of x.
    ``hops`` is an int or a tuple to sample from (mixed-hop training).
    """
    hop_choices = (hops,) if isinstance(hops, int) else tuple(hops)
    xs, ys = [], []
    for _ in range(n):
        ents = (torch.randperm(NENT, generator=g)[:k] + ENT0).tolist()
        succ = {ents[i]: ents[(i + 1) % k] for i in range(k)}  # a cycle so every hop is defined
        order = torch.randperm(k, generator=g).tolist()
        seq = []
        for i in order:
            seq += [ents[i], ARROW, succ[ents[i]], SEP]
        q = ents[int(torch.randint(k, (1,), generator=g))]
        h = hop_choices[int(torch.randint(len(hop_choices), (1,), generator=g))]
        t = q
        for _ in range(h):
            t = succ[t]
        xs.append(seq + [QUERY, HOP1 if h == 1 else HOP2, q])
        ys.append(t)
    return torch.tensor(xs), torch.tensor(ys)


def reasoning_model(cond: str, layers: int = 2, steps: int = 3, seed: int = 0) -> NeuroGraphModel:
    torch.manual_seed(seed)
    cfg = NeuroGraphConfig()
    cfg.lm.d_model, cfg.lm.num_layers, cfg.lm.num_heads = 64, layers, 4
    cfg.lm.vocab_size, cfg.lm.max_seq_len, cfg.lm.freeze_base_model = VOCAB, 64, False
    cfg.workspace.node_dim, cfg.workspace.num_nodes, cfg.workspace.recurrent_steps = 64, 8, steps
    cfg.workspace.dynamic_connectivity = cond != "D"
    cfg.workspace.dropout = 0.0
    cfg.workspace.adapter_gate_init = GATE_INIT
    cfg.condition = "C" if cond in ("A", "A-deep") else cond
    return NeuroGraphModel(cfg.validate())


def used_params(m: NeuroGraphModel, cond: str) -> int:
    mods = [m.base_lm] + ([] if cond.startswith("A") else [m.workspace, m.adapter])
    return sum(p.numel() for mod in mods for p in mod.parameters())


def answer_logits(m: NeuroGraphModel, ids: torch.Tensor, cond: str, steps=None, halt=None):
    emb = m.encode_context(ids)
    ws = None
    if not cond.startswith("A"):
        ws = m.run_workspace(ids, condition=cond, steps=steps, halt_threshold=halt)
        emb = m.inject(emb, ws)
    return m.lm_call(emb)["logits"][:, -1], ws


def train_reasoning(cond: str, seed: int, train_steps: int, layers: int = 2, steps: int = 3, k: int = 6, hops=(1, 2)):
    m = reasoning_model(cond, layers, steps, seed)
    params = [p for mod in ([m.base_lm] if cond.startswith("A") else [m.base_lm, m.workspace, m.adapter])
              for p in mod.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    m.train()
    t0 = time.perf_counter()
    losses = []
    for it in range(train_steps):
        x, y = make_batch(64, k, hops, g)
        logits, ws = answer_logits(m, x, cond)
        loss = F.cross_entropy(logits, y) + (0.01 * ws["reg_loss"] if ws is not None else 0.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if it % 100 == 0:
            losses.append(round(float(loss.item()), 4))
    return m, {"train_seconds": round(time.perf_counter() - t0, 1), "loss_curve_every_100": losses}


@torch.no_grad()
def eval_reasoning(m, cond, k, hops, seed=999, n=2000, steps=None, halt=None):
    m.eval()
    g = torch.Generator().manual_seed(seed)
    x, y = make_batch(n, k, hops, g)
    t0 = time.perf_counter()
    correct, step_counts = 0, []
    for i in range(0, n, 250):
        logits, ws = answer_logits(m, x[i:i + 250], cond, steps=steps, halt=halt)
        correct += int((logits.argmax(-1) == y[i:i + 250]).sum())
        if ws is not None:
            step_counts += ws["steps_executed"].tolist()
    ms_per_example = (time.perf_counter() - t0) * 1000 / n
    return {"accuracy": round(correct / n, 4), "ms_per_example": round(ms_per_example, 4),
            "mean_workspace_steps": round(statistics.mean(step_counts), 3) if step_counts else 0}


REASONING_STEPS_FULL = 8000


def _reasoning_job(job):
    cond, seed, train_steps, kw = job
    torch.set_num_threads(4)
    try:
        m, info = train_reasoning(cond, seed, train_steps, **kw)
        info.update(
            seed=seed,
            params=used_params(m, cond),
            two_hop_k6=eval_reasoning(m, cond, 6, 2),
            one_hop_k6=eval_reasoning(m, cond, 6, 1),
            two_hop_k8_length_gen=eval_reasoning(m, cond, 8, 2),
        )
        if not cond.startswith("A"):
            info["final_adapter_gate"] = m.adapter_gate()
    except Exception as exc:
        info = {"seed": seed, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-800:]}
    print(cond, seed, {k: v for k, v in info.items() if k not in ("loss_curve_every_100", "trace")}, flush=True)
    return cond, info


def bench_reasoning(quick: bool) -> None:
    from concurrent.futures import ProcessPoolExecutor

    train_steps = 300 if quick else REASONING_STEPS_FULL
    seeds = [0] if quick else [0, 1, 2]
    # Parameter-matched deep baseline: add LM layers until >= Condition C's used parameters.
    c_params = used_params(reasoning_model("C"), "C")
    deep_layers = 2
    while used_params(reasoning_model("A", layers=deep_layers), "A") < c_params:
        deep_layers += 1
    conds = {"A": dict(layers=2), "A-deep": dict(layers=deep_layers), "B": dict(layers=2),
             "C": dict(layers=2, steps=3), "D": dict(layers=2, steps=3)}
    jobs = [(c, s, train_steps, kw) for c, kw in conds.items() for s in seeds]
    with ProcessPoolExecutor(max_workers=4) as ex:
        finished = list(ex.map(_reasoning_job, jobs))
    out: Dict[str, Any] = {}
    for cond, kw in conds.items():
        runs = [info for c, info in finished if c == cond]
        ok = [r for r in runs if "error" not in r]

        def agg(key, field="accuracy"):
            return mean_std([r[key][field] for r in ok]) if ok else None

        out[cond] = {
            "config": kw, "runs": runs, "params": ok[0]["params"] if ok else None,
            "two_hop_acc": agg("two_hop_k6"), "one_hop_acc": agg("one_hop_k6"),
            "two_hop_length_gen_acc": agg("two_hop_k8_length_gen"),
            "ms_per_example": agg("two_hop_k6", "ms_per_example"),
        }
    save("reasoning", {
        "task": "Pointer chasing over a random 6-cycle of entity tokens (50-token entity vocabulary) given as shuffled "
                "'a -> b' facts, then QUERY, HOP1|HOP2, x; predict the 1st or 2nd successor of x. Training mixes 1- and "
                "2-hop queries 50/50. Chance = 1/6 among in-context entities. Held-out eval: 2000 fresh examples "
                "(seed 999) per metric; length generalisation uses 8 facts (never seen in training).",
        "train_steps": train_steps, "batch_size": 64, "optimizer": "AdamW lr 1e-3 wd 0.01, grad clip 1.0",
        "seeds": seeds, "lm": "MockCausalLM d=64 heads=4, trained from scratch (LM unfrozen)",
        "workspace": "8 nodes x 64d, context_source=hidden, gated cross-attention injection, gate_init=1.0",
        "deep_baseline_layers": deep_layers, "conditions": out,
    })


# ================================================================== adaptive
def bench_adaptive(quick: bool) -> None:
    train_steps = 300 if quick else REASONING_STEPS_FULL
    m, info = train_reasoning("C", 0, train_steps, layers=2, steps=6)
    info["final_adapter_gate"] = m.adapter_gate()
    rows = []
    policies = [(f"fixed_{s}", s, None) for s in [1, 2, 3, 4, 6]] + \
               [(f"halt_{t}", 6, t) for t in [0.3, 0.1, 0.05, 0.02, 0.01]]
    for name, steps, thr in policies:
        two = eval_reasoning(m, "C", 6, 2, steps=steps, halt=thr)
        one = eval_reasoning(m, "C", 6, 1, steps=steps, halt=thr)
        rows.append({"policy": name, "two_hop": two, "one_hop": one})
        print(rows[-1])
    save("adaptive", {"model": f"Condition C trained with T=6 for {train_steps} steps on the mixed-hop reasoning task (seed 0)",
                      "train": info, "results": rows,
                      "note": "halt_x = stop updating a sample once ||M(t+1)-M(t)||/||M(t)|| < x (max 6 steps). "
                              "ms_per_example includes the full LM pass used as workspace context."})


# ==================================================================== memory
MEMORY_PAIRS = [
    ("My dog is called Biscuit", "what is my dog's name"),
    ("I am allergic to peanuts", "do I have any food allergies"),
    ("My sister lives in Toronto", "where does my sister live"),
    ("I work as a data engineer", "what is my job"),
    ("My favourite programming language is Rust", "which language do I like coding in most"),
    ("I was born in 1994", "what year was I born"),
    ("My car is a blue Honda Civic", "what kind of car do I drive"),
    ("I prefer tea over coffee", "what hot drink do I prefer"),
    ("My thesis is about graph neural networks", "what is my research topic"),
    ("I go running every Tuesday and Thursday", "on which days do I exercise"),
    ("My manager's name is Priya", "who is my boss"),
    ("The wifi password at home is stored in the drawer", "where do I keep the wifi password"),
    ("I am learning Japanese", "which foreign language am I studying"),
    ("My flight to Berlin leaves on 3 March", "when is my trip to Germany"),
    ("I use a standing desk", "what kind of desk do I have"),
    ("My son plays the violin", "what instrument does my child play"),
    ("I am vegetarian", "do I eat meat"),
    ("My laptop is a ThinkPad X1", "which computer do I use"),
    ("I support FC Barcelona", "which football team do I follow"),
    ("My mother's birthday is in July", "when is mom's birthday"),
    ("I live in a flat on the fourth floor", "which floor is my apartment on"),
    ("I usually wake up at 6 am", "what time do I get up"),
    ("My bank is HDFC", "which bank do I use"),
    ("I am afraid of heights", "what am I scared of"),
    ("My favourite film is Interstellar", "what movie do I love most"),
    ("I have two cats", "how many pets do I have"),
    ("My gym membership expires in December", "when does my fitness club subscription end"),
    ("I drink oat milk", "what milk do I use"),
    ("My office is in Bengaluru", "which city do I work in"),
    ("I am saving money for a motorbike", "what am I saving up for"),
]


def bench_memory(quick: bool) -> None:
    from neurograph.memory_store import HashingEmbedder, MemoryStore, load_embedder

    out = {}
    for name, emb in [("hashed_bow", HashingEmbedder()), ("minilm", None)]:
        try:
            emb = emb or load_embedder("minilm")
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        st = MemoryStore(":memory:", embedder=emb)
        ids = [st.add("alice", fact)["id"] for fact, _ in MEMORY_PAIRS]
        st.add("bob", "Bob's dog is called Rex")  # isolation probe
        top1 = top3 = 0
        t0 = time.perf_counter()
        for i, (_, q) in enumerate(MEMORY_PAIRS):
            res = st.search("alice", q, top_k=3)
            got = [r["id"] for r in res]
            top1 += int(got[:1] == [ids[i]])
            top3 += int(ids[i] in got)
        lat = (time.perf_counter() - t0) * 1000 / len(MEMORY_PAIRS)
        leak = any("Rex" in r["content"] for r in st.search("alice", "what is the dog called", top_k=50))
        out[name] = {"embedder": st.embedder_name, "top1_accuracy": round(top1 / len(MEMORY_PAIRS), 4),
                     "top3_accuracy": round(top3 / len(MEMORY_PAIRS), 4), "ms_per_query": round(lat, 2),
                     "cross_user_leak": leak}
        print(name, out[name])
    save("memory", {"n_facts": len(MEMORY_PAIRS), "queries": "paraphrased questions (hand-written, see run_all.py)",
                    "chance_top1": round(1 / len(MEMORY_PAIRS), 4), "results": out})


# ======================================================================== LM
def bench_lm(quick: bool) -> None:
    import psutil

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from neurograph.engine import ChatEngine, ChatRequestData

    out: Dict[str, Any] = {}
    texts: List[str] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("imdb", split="test")
        idx = random.Random(0).sample(range(len(ds)), 50 if quick else 200)
        texts = [ds[i]["text"] for i in idx]
        out["ppl_data"] = f"imdb test split, {len(texts)} random reviews (seed 0), first 128 tokens"
    except Exception as exc:
        out["ppl_data_error"] = f"{type(exc).__name__}: {exc}"

    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    eng = ChatEngine("microsoft/DialoGPT-small")
    out["model_load_seconds"] = eng.load_seconds
    out["rss_after_load_mb"] = round(proc.memory_info().rss / 1e6, 1)
    out["rss_model_delta_mb"] = round((proc.memory_info().rss - rss0) / 1e6, 1)
    tok = eng.model.tokenizer

    if texts:
        def ppl(model: NeuroGraphModel, cond: str) -> float:
            nll, n = 0.0, 0
            with torch.no_grad():
                for t in texts:
                    ids = tok(t, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
                    o = model(ids, labels=ids, condition=cond)
                    k = ids.shape[1] - 1
                    nll += float(o["lm_loss"]) * k
                    n += k
            return round(float(torch.exp(torch.tensor(nll / n))), 2)

        out["perplexity"] = {"A_base": ppl(eng.model, "A"), "C_gated_gate0": ppl(eng.model, "C")}
        cfg = NeuroGraphConfig()
        cfg.lm.model_name_or_path = "microsoft/DialoGPT-small"
        cfg.workspace.node_dim = 64
        cfg.workspace.adapter_mode = "prefix"
        cfg.workspace.context_source = "embeddings"
        torch.manual_seed(0)
        legacy = NeuroGraphModel(cfg).eval()
        out["perplexity"]["C_legacy_prefix_untrained"] = ppl(legacy, "C")
        del legacy
        print(out["perplexity"])

    prompts = ["Hello, how are you?", "What do you do for fun?", "Do you like music?", "Tell me about your day.",
               "What's your favourite food?", "Are you a robot?", "Where are you from?", "Any plans for the weekend?",
               "What do you think about AI?", "Recommend me a book."]
    if quick:
        prompts = prompts[:4]
    lat: Dict[str, Any] = {}
    for mode in ("fast", "cognitive", "adaptive"):
        totals, firsts, tps, toks, stops, ws_steps = [], [], [], [], [], []
        for p in prompts:
            r = eng.chat(ChatRequestData(message=p, history=[], mode=mode, do_sample=False, max_new_tokens=64))
            s = r["stats"]
            totals.append(s["total_ms"])
            firsts.append(s["first_token_ms"])
            tps.append(s["tokens_per_s"] or 0)
            toks.append(s["new_tokens"])
            stops.append(s["stop_reason"])
            ws_steps.append((r["telemetry"] or {}).get("steps_executed") or 0)
        lat[mode] = {
            "total_ms_median": round(statistics.median(totals), 1), "total_ms_max": round(max(totals), 1),
            "first_token_ms_median": round(statistics.median(firsts), 1),
            "tokens_per_s_median": round(statistics.median(tps), 2), "new_tokens_mean": round(statistics.mean(toks), 1),
            "stop_reasons": {k: stops.count(k) for k in set(stops)}, "workspace_steps_mean": statistics.mean(ws_steps),
            "within_2s": sum(t <= 2000 for t in totals) / len(totals),
        }
        print(mode, lat[mode])
    out["chat_latency"] = {"prompts": len(prompts), "max_new_tokens": 64, "greedy": True,
                           "time_budget_s": eng.max_time_s, "modes": lat}

    ids = eng.build_input_ids("Do you like music?", [], None)
    g = eng.generator
    t0 = time.perf_counter()
    kv = g.generate(ids, max_new_tokens=64, condition="A", stop_on_eos=False)
    t_kv = time.perf_counter() - t0
    seq = ids.clone()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(64):
            seq = torch.cat([seq, eng.model.base_lm(input_ids=seq).logits[:, -1].argmax(-1, keepdim=True)], 1)
    t_full = time.perf_counter() - t0
    out["kv_cache"] = {"tokens": 64, "kv_cache_s": round(t_kv, 3), "full_recompute_s": round(t_full, 3),
                       "speedup": round(t_full / t_kv, 2), "identical_tokens": bool(torch.equal(kv, seq))}
    out["peak_rss_mb"] = round(proc.memory_info().rss / 1e6, 1)
    print(out["kv_cache"])
    save("lm", out)


SECTIONS = {"world_model": bench_world_model, "planning": bench_planning, "reasoning": bench_reasoning,
            "adaptive": bench_adaptive, "memory": bench_memory, "lm": bench_lm}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(SECTIONS))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    for name in args.only or list(SECTIONS):
        print(f"\n===== {name} =====")
        t0 = time.perf_counter()
        try:
            SECTIONS[name](args.quick)
        except Exception as exc:
            save(name, {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()})
        print(f"[{name}] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
