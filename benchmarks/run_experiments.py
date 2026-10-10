"""Run controlled experiments and benchmarks for NeuroGraph-LLM."""

from typing import Dict, Any
import argparse
import json
from pathlib import Path
import torch

from neurograph.config import NeuroGraphConfig
from neurograph.train import train
from neurograph.evaluate import evaluate_planning_environment
from benchmarks.metrics import measure_model_efficiency


def run_experiment_suite(config_path: str = "configs/tiny.yaml", output_dir: str = "experiments_output") -> Dict[str, Any]:
    """Execute end-to-end training and evaluation experiment."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = NeuroGraphConfig.from_yaml(config_path)
    print(f"=== Starting Experiment: {cfg.experiment_name} (Condition: {cfg.condition}) ===")

    # 1. Train model
    train_res = train(config=cfg, output_dir=str(out_dir))
    model = train_res["model"]

    # 2. Evaluate efficiency (params, latency)
    sample_input = torch.randint(1, cfg.lm.vocab_size, (1, 16))
    efficiency = measure_model_efficiency(model, sample_input, condition=cfg.condition)

    # 3. Evaluate planning on held-out test maps
    print("Evaluating planning performance on 10 held-out maps...")
    planning_metrics = evaluate_planning_environment(model, num_episodes=10, seed=500)

    # 4. Evaluate world model accuracy on synthetic transitions
    print("Evaluating world model transitions...")
    s0 = torch.randn(20, cfg.world_model.state_dim)
    actions = torch.randint(0, cfg.world_model.action_dim, (20,))
    s1_target = s0 + torch.randn_like(s0) * 0.1
    wm_metrics = model.world_model.evaluate_transitions(s0, actions, s1_target)

    # 5. Compile full results
    results = {
        "experiment_name": cfg.experiment_name,
        "condition": cfg.condition,
        "final_total_loss": train_res["history"]["total_loss"][-1],
        "final_lm_loss": train_res["history"]["lm_loss"][-1],
        "final_world_loss": train_res["history"]["world_loss"][-1],
        "final_action_loss": train_res["history"]["action_loss"][-1],
        **efficiency,
        **planning_metrics,
        **wm_metrics,
    }

    results_file = out_dir / f"{cfg.experiment_name}_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults successfully saved to: {results_file}")
    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description="Run NeuroGraph-LLM Experiment Suite.")
    parser.add_argument("--config", type=str, default="configs/tiny.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default="experiments_output", help="Output directory")
    args = parser.parse_args()

    run_experiment_suite(config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
