"""Unit tests for MemoryStore, planners, generation streaming and checkpoint checks."""

import threading

import torch

from neurograph.checkpoint import check_compatibility, save_checkpoint
from neurograph.config import NeuroGraphConfig
from neurograph.generation import CognitiveGenerator
from neurograph.memory_store import HashingEmbedder, MemoryStore
from neurograph.model import NeuroGraphModel


def _store():
    return MemoryStore(":memory:", embedder=HashingEmbedder())


def test_memory_store_isolation_and_retrieval():
    s = _store()
    s.add("u1", "The project deadline is Friday")
    s.add("u1", "I prefer Python over Java")
    s.add("u2", "The project deadline is Monday")
    res = s.search("u1", "when is the deadline", top_k=1)
    assert res and "Friday" in res[0]["content"]
    assert all("Monday" not in r["content"] for r in s.search("u1", "deadline", top_k=5))


def test_memory_conflict_flagging_and_disable():
    s = _store()
    a = s.add("u", "My cat is named Luna")
    b = s.add("u", "My cat is named Luna!")
    assert a["id"] in b["possible_conflict_with"]
    s.set_enabled("u", False)
    assert s.search("u", "cat name") == []
    assert s.search("u", "cat name", respect_disabled=False)


def test_memory_rejects_empty_and_oversize():
    s = _store()
    for bad in ["", "   ", "x" * 5000]:
        try:
            s.add("u", bad)
            raise AssertionError("should have raised")
        except ValueError:
            pass


def _tiny_model():
    cfg = NeuroGraphConfig()
    cfg.lm.d_model = 32
    cfg.lm.vocab_size = 100
    cfg.workspace.node_dim = 32
    cfg.workspace.num_nodes = 4
    return NeuroGraphModel(cfg)


def test_stream_cancellation_stops_early():
    m = _tiny_model()
    g = CognitiveGenerator(m)
    ev = threading.Event()
    out = []
    for i, t in enumerate(g.stream(torch.tensor([[1, 2, 3]]), max_new_tokens=50, condition="C", cancel_event=ev)):
        out.append(t)
        if i == 2:
            ev.set()
    assert len(out) == 3
    assert g.last_stats.stop_reason == "cancelled"


def test_stream_time_budget():
    g = CognitiveGenerator(_tiny_model())
    list(g.stream(torch.tensor([[1, 2, 3]]), max_new_tokens=10_000, condition="A", max_time_s=0.05))
    assert g.last_stats.stop_reason in {"max_time", "context_limit"}


def test_adaptive_halting_reduces_steps():
    m = _tiny_model().eval()
    ids = torch.randint(1, 100, (2, 6))
    with torch.no_grad():
        full = m.run_workspace(ids, condition="C", steps=8)
        halted = m.run_workspace(ids, condition="C", steps=8, halt_threshold=10.0)  # halts after min_steps
    assert int(full["steps_executed"].min()) == 8
    assert int(halted["steps_executed"].max()) < 8


def test_condition_d_requires_static_model():
    m = _tiny_model()
    try:
        m.run_workspace(torch.tensor([[1, 2]]), condition="D")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_checkpoint_compatibility_report(tmp_path):
    m = _tiny_model()
    p = save_checkpoint(tmp_path / "ok.pt", m)
    assert check_compatibility(p, _tiny_model())["compatible"]
    cfg = NeuroGraphConfig()
    cfg.lm.d_model = 48
    cfg.workspace.node_dim = 32
    other = NeuroGraphModel(cfg)
    rep = check_compatibility(p, other)
    assert not rep["compatible"] and rep["shape_mismatches"] > 0
    (tmp_path / "bad.pt").write_bytes(b"garbage")
    assert "unreadable" in check_compatibility(tmp_path / "bad.pt", m)["reason"]


def test_world_model_search_planner_on_trivial_map():
    from environments.gridworld import GridWorld
    from neurograph.planner import WorldModelSearchPlanner
    from neurograph.world_model_training import train_world_model

    wm, rep = train_world_model(train_seeds=range(0, 20), test_seeds=range(500, 505), epochs=8)
    assert rep["test"]["position_accuracy"] > rep["untrained_test"]["position_accuracy"]
    env = GridWorld(6, 6, obstacles=set(), state_dim=32)
    res = WorldModelSearchPlanner(wm, 6, 6).execute_plan(env, 30)
    assert res["success"]
