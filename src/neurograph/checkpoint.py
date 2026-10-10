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


def check_compatibility(path: Path | str, model: nn.Module) -> Dict[str, Any]:
    """Compare a checkpoint's tensors with ``model`` without modifying the model.

    Returns a report with ``compatible`` (bool), counts of missing/unexpected keys and
    shape mismatches, and a human-readable ``reason``. A checkpoint is compatible only
    when every key matches in name and shape.
    """
    path = Path(path)
    report: Dict[str, Any] = {"path": str(path), "exists": path.exists(), "compatible": False}
    if not path.exists():
        report["reason"] = "checkpoint file not found"
        return report
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # corrupt / unreadable file
        report["reason"] = f"unreadable checkpoint: {type(exc).__name__}: {exc}"
        return report
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else {}
    own = model.state_dict()
    missing = [k for k in own if k not in sd]
    unexpected = [k for k in sd if k not in own]
    mismatched = [k for k in own if k in sd and tuple(sd[k].shape) != tuple(own[k].shape)]
    report.update(
        missing_keys=len(missing),
        unexpected_keys=len(unexpected),
        shape_mismatches=len(mismatched),
        example_mismatch=(
            f"{mismatched[0]}: ckpt {tuple(sd[mismatched[0]].shape)} vs model {tuple(own[mismatched[0]].shape)}"
            if mismatched else None
        ),
        step=ckpt.get("step") if isinstance(ckpt, dict) else None,
    )
    report["compatible"] = not (missing or unexpected or mismatched)
    report["reason"] = "all tensors match" if report["compatible"] else (
        f"{len(missing)} missing, {len(unexpected)} unexpected, {len(mismatched)} shape-mismatched tensors"
    )
    return report
