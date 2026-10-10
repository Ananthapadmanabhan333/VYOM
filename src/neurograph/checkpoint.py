"""Checkpoint saving, loading, and state management for NeuroGraph-LLM."""

from typing import Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from neurograph.config import NeuroGraphConfig


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    step: int = 0,
    config: Optional[NeuroGraphConfig] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> Path:
    """Save full model and training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": config.to_dict() if config is not None else None,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(state, path)
    return path


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Load checkpoint and restore model/optimizer states."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "step": checkpoint.get("step", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", None),
    }
