"""Fine-tuning pipeline for NeuroGraph-LLM on expert navigation demonstrations."""

from typing import Dict, Any
import argparse
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from neurograph.config import NeuroGraphConfig
from neurograph.model import NeuroGraphModel
from neurograph.checkpoint import save_checkpoint
from neurograph.dataset import NavigationDemonstrationDataset, collate_demonstrations
from neurograph.evaluate import evaluate_planning_environment


def finetune(
    config: NeuroGraphConfig,
    epochs: int = 8,
    train_episodes: int = 80,
    val_episodes: int = 20,
    output_path: str = "checkpoints/neurograph_perfected.pt",
) -> Dict[str, Any]:
    """Fine-tune NeuroGraph-LLM on expert demonstrations."""
    device = torch.device(config.training.device)
    torch.manual_seed(config.training.seed)

    print(f"=== Starting Cognitive Fine-Tuning (Condition: {config.condition}) on {device} ===")

    # 1. Prepare Datasets
    print(f"Generating {train_episodes} expert demonstration episodes for training...")
    train_ds = NavigationDemonstrationDataset(
        num_episodes=train_episodes,
        seed=config.training.seed,
        state_dim=config.world_model.state_dim,
        goal_dim=config.workspace.goal_dim,
        uncertainty_dim=config.workspace.uncertainty_dim,
        vocab_size=config.lm.vocab_size,
    )
    val_ds = NavigationDemonstrationDataset(
        num_episodes=val_episodes,
        seed=config.training.seed + 999,
        state_dim=config.world_model.state_dim,
        goal_dim=config.workspace.goal_dim,
        uncertainty_dim=config.workspace.uncertainty_dim,
        vocab_size=config.lm.vocab_size,
    )

    print(f"Dataset compiled: {len(train_ds)} train transitions, {len(val_ds)} validation transitions.")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_demonstrations,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_demonstrations,
    )

    # 2. Instantiate Model
    model = NeuroGraphModel(config).to(device)

    # Evaluate pre-finetuning planning baseline
    print("Evaluating pre-finetuning planning on 10 held-out maps...")
    pre_eval = evaluate_planning_environment(model, num_episodes=10, seed=700)
    print(f"Pre-FT Goal Completion: {pre_eval['goal_completion_rate']*100:.1f}%, Invalid Rate: {pre_eval['invalid_action_rate']*100:.1f}%")

    # Trainable parameters (workspace + world model + action head + LoRA adapter)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)

    # 3. Fine-tuning Loop
    history = {"epoch": [], "loss": [], "action_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct_actions = 0
        total_actions = 0

        for batch in train_loader:
            optimizer.zero_grad()

            b_input_ids = batch["input_ids"].to(device)
            b_labels = batch["labels"].to(device)
            b_curr_s = batch["current_state"].to(device)
            b_target_a = batch["target_action"].to(device)
            b_next_s = batch["target_next_state"].to(device)
            b_goal = batch["active_goal"].to(device)
            b_unc = batch["uncertainty"].to(device)

            outputs = model(
                input_ids=b_input_ids,
                labels=b_labels,
                active_goal=b_goal,
                uncertainty=b_unc,
                current_state=b_curr_s,
                target_action=b_target_a,
                target_next_state=b_next_s,
                condition=config.condition,
            )

            loss = outputs["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(b_target_a)

            if outputs["action_logits"] is not None:
                preds = torch.argmax(outputs["action_logits"], dim=-1)
                correct_actions += (preds == b_target_a).sum().item()
                total_actions += len(b_target_a)

        train_loss = epoch_loss / max(1, total_actions)
        train_acc = correct_actions / max(1, total_actions)

        # Validation pass
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for v_batch in val_loader:
                v_out = model(
                    input_ids=v_batch["input_ids"].to(device),
                    active_goal=v_batch["active_goal"].to(device),
                    uncertainty=v_batch["uncertainty"].to(device),
                    current_state=v_batch["current_state"].to(device),
                    condition=config.condition,
                )
                if v_out["action_logits"] is not None:
                    v_preds = torch.argmax(v_out["action_logits"], dim=-1)
                    val_correct += (v_preds == v_batch["target_action"].to(device)).sum().item()
                    val_total += len(v_batch["target_action"])

        val_acc = val_correct / max(1, val_total)

        history["epoch"].append(epoch)
        history["loss"].append(train_loss)
        history["action_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Action Acc: {train_acc*100:5.1f}% | "
            f"Val Action Acc: {val_acc*100:5.1f}%"
        )

    # 4. Post-finetuning evaluation
    print("\nEvaluating post-finetuning planning on 10 held-out maps...")
    post_eval = evaluate_planning_environment(model, num_episodes=10, seed=700)
    print(
        f"Post-FT Goal Completion: {post_eval['goal_completion_rate']*100:.1f}%, "
        f"Avg Steps: {post_eval['avg_steps_to_goal']:.2f}, "
        f"Invalid Rate: {post_eval['invalid_action_rate']*100:.1f}%"
    )

    # 5. Save perfected checkpoint
    save_path = Path(output_path)
    save_checkpoint(
        path=save_path,
        model=model,
        optimizer=optimizer,
        step=epochs,
        config=config,
        metrics={
            "final_train_acc": history["action_acc"][-1],
            "final_val_acc": history["val_acc"][-1],
            "post_goal_completion": post_eval["goal_completion_rate"],
        },
    )
    print(f"Fine-tuned model successfully saved to: {save_path}")

    return {
        "model": model,
        "history": history,
        "pre_eval": pre_eval,
        "post_eval": post_eval,
        "checkpoint_path": str(save_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune NeuroGraph-LLM.")
    parser.add_argument("--config", type=str, default="configs/tiny.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=8, help="Number of fine-tuning epochs")
    parser.add_argument("--train-episodes", type=int, default=80, help="Number of demonstration episodes")
    parser.add_argument("--output", type=str, default="checkpoints/neurograph_perfected.pt", help="Checkpoint output path")
    args = parser.parse_args()

    cfg = NeuroGraphConfig.from_yaml(args.config)
    finetune(config=cfg, epochs=args.epochs, train_episodes=args.train_episodes, output_path=args.output)


if __name__ == "__main__":
    main()
