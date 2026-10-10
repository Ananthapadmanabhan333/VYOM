from typing import Dict, Optional
import torch

from neurograph.model import NeuroGraphModel
from neurograph.planner import ModelBasedRolloutPlanner

try:
    from environments.gridworld import GridWorld
except ImportError:
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from environments.gridworld import GridWorld


@torch.no_grad()
def evaluate_model_step(
    model: NeuroGraphModel,
    batch: Dict[str, torch.Tensor],
    condition: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate model on a single synthetic multi-task batch."""
    model.eval()
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch.get("labels"),
        active_goal=batch.get("active_goal"),
        uncertainty=batch.get("uncertainty"),
        current_state=batch.get("current_state"),
        target_action=batch.get("target_action"),
        target_next_state=batch.get("target_next_state"),
        condition=condition,
    )

    metrics = {
        "total_loss": float(outputs["total_loss"].item()),
        "lm_loss": float(outputs["lm_loss"].item()) if outputs["lm_loss"] is not None else 0.0,
        "world_loss": float(outputs["world_loss"].item()),
        "action_loss": float(outputs["action_loss"].item()),
        "reg_loss": float(outputs["reg_loss"].item()),
    }

    if outputs["action_logits"] is not None and "target_action" in batch:
        preds = torch.argmax(outputs["action_logits"], dim=-1)
        acc = (preds == batch["target_action"]).float().mean().item()
        metrics["action_acc"] = float(acc)

    return metrics


def evaluate_planning_environment(
    model: NeuroGraphModel,
    num_episodes: int = 10,
    seed: int = 100,
    max_steps: int = 25,
) -> Dict[str, float]:
    """Evaluate model rollout planner on a set of GridWorld test environments."""
    planner = ModelBasedRolloutPlanner(
        config=model.config.planner,
        world_model=model.world_model,
        policy_model=model,
        action_dim=model.config.world_model.action_dim,
    )

    successes = 0
    total_steps = 0
    total_invalids = 0

    for ep in range(num_episodes):
        env = GridWorld.generate_random_map(
            height=6,
            width=6,
            obstacle_density=0.15,
            seed=seed + ep,
            state_dim=model.config.world_model.state_dim,
        )
        res = planner.execute_plan(env, max_steps=max_steps)

        if res["success"]:
            successes += 1
        total_steps += res["steps"]
        total_invalids += res["invalid_actions"]

    completion_rate = successes / max(1, num_episodes)
    avg_steps = total_steps / max(1, num_episodes)
    invalid_rate = total_invalids / max(1, total_steps)

    return {
        "goal_completion_rate": completion_rate,
        "avg_steps_to_goal": avg_steps,
        "invalid_action_rate": invalid_rate,
    }
