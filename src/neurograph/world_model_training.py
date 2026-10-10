"""Collect GridWorld transitions and train / evaluate the WorldStatePredictor.

The world model is trained on real environment transitions (s, a, s') from random
obstacle maps. Evaluation uses held-out map seeds that never appear in training, and
reports both state MSE and *decoded position accuracy* (predicted (row, col) rounded to
the grid equals the true next position).
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from environments.gridworld import GridWorld
from neurograph.config import WorldModelConfig
from neurograph.world_model import WorldStatePredictor

GRID = 6


def decode_pos(state: torch.Tensor, height: int = GRID, width: int = GRID) -> Tuple[int, int]:
    r = round(float((state[0].item() + 1.0) / 2.0 * (height - 1)))
    c = round(float((state[1].item() + 1.0) / 2.0 * (width - 1)))
    return max(0, min(height - 1, r)), max(0, min(width - 1, c))


def collect_transitions(
    seeds: range, state_dim: int, episodes_per_map: int = 4, steps_per_episode: int = 25,
    obstacle_density: float = 0.2, rng_seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Random-walk transitions with random start/goal on each map seed."""
    rng = random.Random(rng_seed)
    S, A, S2 = [], [], []
    for seed in seeds:
        env = GridWorld.generate_random_map(GRID, GRID, obstacle_density, seed=seed, state_dim=state_dim)
        free = [(r, c) for r in range(GRID) for c in range(GRID) if env.is_valid_coord(r, c)]
        for _ in range(episodes_per_map):
            start, goal = rng.sample(free, 2)
            env.max_steps = steps_per_episode + 1
            s, _ = env.reset(start_pos=start, goal_pos=goal)
            for _ in range(steps_per_episode):
                a = rng.randrange(5)
                s2, _, _, _ = env.step(a)
                S.append(s)
                A.append(a)
                S2.append(s2)
                s = s2
    return {"s": torch.stack(S), "a": torch.tensor(A, dtype=torch.long), "s2": torch.stack(S2)}


@torch.no_grad()
def evaluate(model: WorldStatePredictor, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
    model.eval()
    pred, unc = model(data["s"], data["a"])
    mse = float(((pred - data["s2"]) ** 2).mean().item())
    hits = sum(decode_pos(p) == decode_pos(t) for p, t in zip(pred, data["s2"]))
    # Copy baseline: predict s' = s (correct whenever the move is blocked or STAY).
    copy_hits = sum(decode_pos(s) == decode_pos(t) for s, t in zip(data["s"], data["s2"]))
    return {
        "mse": round(mse, 6),
        "position_accuracy": round(hits / len(pred), 4),
        "copy_baseline_position_accuracy": round(copy_hits / len(pred), 4),
        "mean_predicted_std": round(float(unc.mean().item()), 5),
        "n": len(pred),
    }


def train_world_model(
    state_dim: int = 32, hidden_dim: int = 64, train_seeds: range = range(0, 200),
    test_seeds: range = range(1000, 1100), epochs: int = 30, batch_size: int = 256,
    lr: float = 3e-3, seed: int = 0,
) -> Tuple[WorldStatePredictor, Dict[str, object]]:
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    train = collect_transitions(train_seeds, state_dim, rng_seed=seed)
    test = collect_transitions(test_seeds, state_dim, rng_seed=seed + 12345)
    model = WorldStatePredictor(WorldModelConfig(state_dim=state_dim, hidden_dim=hidden_dim))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = len(train["a"])
    untrained = evaluate(model, test)
    history: List[float] = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            out = model.compute_loss(train["s"][idx], train["a"][idx], train["s2"][idx])
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(out["mse"].item()) * len(idx)
        history.append(round(tot / n, 6))
    report = {
        "train_transitions": n,
        "test_transitions": len(test["a"]),
        "train_map_seeds": [train_seeds.start, train_seeds.stop - 1],
        "test_map_seeds": [test_seeds.start, test_seeds.stop - 1],
        "epochs": epochs,
        "seed": seed,
        "train_mse_history": history,
        "untrained_test": untrained,
        "train": evaluate(model, train),
        "test": evaluate(model, test),
        "train_seconds": round(time.perf_counter() - t0, 2),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    return model, report


def save_world_model(model: WorldStatePredictor, path: str | Path, report: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": vars(model.config), "report": report}, path)


def load_world_model(path: str | Path) -> Tuple[WorldStatePredictor, Dict[str, object]]:
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = WorldStatePredictor(WorldModelConfig(**blob["config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob.get("report", {})
