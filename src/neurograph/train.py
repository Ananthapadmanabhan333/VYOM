"""Multi-Objective Training Pipeline for NeuroGraph-LLM."""

from typing import Dict, Any, Optional
import argparse
import random
from pathlib import Path
import torch
import torch.optim as optim

from neurograph.config import NeuroGraphConfig
from neurograph.model import NeuroGraphModel
from neurograph.checkpoint import save_checkpoint

try:
    from environments.gridworld import GridWorld
except ImportError:
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from environments.gridworld import GridWorld


def generate_synthetic_batch(
    config: NeuroGraphConfig,
    batch_size: int,
    device: torch.device,
    seq_len: int = 16,
) -> Dict[str, torch.Tensor]:
    """Generate synthetic multi-task batch of language, world states, and actions."""
    vocab_size = config.lm.vocab_size
    state_dim = config.world_model.state_dim
    goal_dim = config.workspace.goal_dim
    unc_dim = config.workspace.uncertainty_dim
    action_dim = config.world_model.action_dim

    # 1. Language inputs and targets
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    # 2. Synthetic environment states and transitions (using GridWorld logic)
    current_states = []
    target_actions = []
    target_next_states = []

    for _ in range(batch_size):
        r = random.randint(0, 5)
        c = random.randint(0, 5)
        gr = 5
        gc = 5
        action = random.randint(0, action_dim - 1)

        # Simplified state tensor
        s_curr = torch.zeros(state_dim, device=device)
        s_curr[0] = (r / 5.0) * 2.0 - 1.0
        s_curr[1] = (c / 5.0) * 2.0 - 1.0
        s_curr[2] = (gr / 5.0) * 2.0 - 1.0
        s_curr[3] = (gc / 5.0) * 2.0 - 1.0

        dr, dc = GridWorld.ACTION_MAP.get(action, (0, 0))
        nr = max(0, min(5, r + dr))
        nc = max(0, min(5, c + dc))

        s_next = torch.zeros(state_dim, device=device)
        s_next[0] = (nr / 5.0) * 2.0 - 1.0
        s_next[1] = (nc / 5.0) * 2.0 - 1.0
        s_next[2] = (gr / 5.0) * 2.0 - 1.0
        s_next[3] = (gc / 5.0) * 2.0 - 1.0

        current_states.append(s_curr)
        target_actions.append(action)
        target_next_states.append(s_next)

    current_state = torch.stack(current_states)
    target_action = torch.tensor(target_actions, dtype=torch.long, device=device)
    target_next_state = torch.stack(target_next_states)

    # 3. Active goal and uncertainty signals
    active_goal = torch.randn(batch_size, goal_dim, device=device)
    uncertainty = torch.abs(torch.randn(batch_size, unc_dim, device=device) * 0.1)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "current_state": current_state,
        "target_action": target_action,
        "target_next_state": target_next_state,
        "active_goal": active_goal,
        "uncertainty": uncertainty,
    }


def train(
    config: NeuroGraphConfig,
    max_steps: Optional[int] = None,
    device_name: Optional[str] = None,
    output_dir: str = "checkpoints",
) -> Dict[str, Any]:
    """Execute training loop with multi-objective tracking."""
    device = torch.device(device_name or config.training.device)
    # Check CUDA availability if requested
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    total_steps = max_steps or config.training.max_steps
    batch_size = config.training.batch_size
    grad_accum = config.training.gradient_accumulation_steps

    # Set seeds for reproducibility
    torch.manual_seed(config.training.seed)
    random.seed(config.training.seed)

    print(f"Initializing NeuroGraph-LLM (Condition: {config.condition}) on {device}...")
    model = NeuroGraphModel(config).to(device)
    model.train()

    # Trainable parameters (filter out frozen base_lm if applicable)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Starting training for {total_steps} steps (Batch size: {batch_size}, Grad accum: {grad_accum})...")

    history = {
        "total_loss": [],
        "lm_loss": [],
        "world_loss": [],
        "action_loss": [],
        "reg_loss": [],
    }

    optimizer.zero_grad()
    for step in range(1, total_steps + 1):
        batch = generate_synthetic_batch(config, batch_size=batch_size, device=device)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            active_goal=batch["active_goal"],
            uncertainty=batch["uncertainty"],
            current_state=batch["current_state"],
            target_action=batch["target_action"],
            target_next_state=batch["target_next_state"],
            condition=config.condition,
        )

        loss = outputs["total_loss"] / grad_accum
        loss.backward()

        if step % grad_accum == 0 or step == total_steps:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Record metrics
        t_loss = float(outputs["total_loss"].item())
        lm_l = float(outputs["lm_loss"].item()) if outputs["lm_loss"] is not None else 0.0
        w_l = float(outputs["world_loss"].item())
        a_l = float(outputs["action_loss"].item())
        r_l = float(outputs["reg_loss"].item())

        history["total_loss"].append(t_loss)
        history["lm_loss"].append(lm_l)
        history["world_loss"].append(w_l)
        history["action_loss"].append(a_l)
        history["reg_loss"].append(r_l)

        if step % max(1, total_steps // 5) == 0 or step == total_steps:
            print(
                f"Step {step:4d}/{total_steps} | "
                f"Total Loss: {t_loss:.4f} | "
                f"LM Loss: {lm_l:.4f} | "
                f"World Loss: {w_l:.4f} | "
                f"Action Loss: {a_l:.4f} | "
                f"Reg Loss: {r_l:.4f}"
            )

    # Save final checkpoint
    out_path = Path(output_dir) / f"{config.experiment_name}_final.pt"
    save_checkpoint(
        path=out_path,
        model=model,
        optimizer=optimizer,
        step=total_steps,
        config=config,
        metrics={"final_total_loss": history["total_loss"][-1]},
    )
    print(f"Training complete. Checkpoint saved to: {out_path}")

    return {
        "model": model,
        "optimizer": optimizer,
        "history": history,
        "checkpoint_path": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train NeuroGraph-LLM.")
    parser.add_argument("--config", type=str, default="configs/tiny.yaml", help="Path to config YAML")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    cfg = NeuroGraphConfig.from_yaml(args.config)
    train(config=cfg, max_steps=args.max_steps, device_name=args.device, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
