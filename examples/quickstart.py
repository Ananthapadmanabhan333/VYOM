import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
from neurograph.config import NeuroGraphConfig
from neurograph.model import NeuroGraphModel
from neurograph.planner import ModelBasedRolloutPlanner
from environments.gridworld import GridWorld


def run_quickstart():
    print("=== NeuroGraph-LLM Quickstart ===")

    # 1. Initialize configuration (Condition C: Recurrent Dynamic Graph Workspace)
    config = NeuroGraphConfig.from_yaml("configs/tiny.yaml")
    config.condition = "C"
    print(f"Loaded configuration: {config.experiment_name} | Condition: {config.condition}")

    # 2. Instantiate Model
    model = NeuroGraphModel(config)
    model.eval()
    print("NeuroGraph-LLM model initialized successfully.")

    # 3. Simulate forward pass with contextual language tokens and goal
    batch_size = 1
    seq_len = 8
    input_ids = torch.randint(1, config.lm.vocab_size, (batch_size, seq_len))
    active_goal = torch.randn(batch_size, config.workspace.goal_dim)
    uncertainty = torch.abs(torch.randn(batch_size, config.workspace.uncertainty_dim) * 0.1)

    outputs = model(
        input_ids=input_ids,
        active_goal=active_goal,
        uncertainty=uncertainty,
        condition="C",
    )

    print("\nForward pass outputs:")
    print(f"- Logits shape: {outputs['logits'].shape}")
    print(f"- Action logits shape: {outputs['action_logits'].shape}")
    print(f"- Workspace node states shape: {outputs['workspace_outputs']['final_nodes'].shape}")
    print(f"- Workspace pooled representation shape: {outputs['workspace_outputs']['pooled_state'].shape}")
    print(f"- Recurrent steps executed: {len(outputs['workspace_outputs']['adj_history'])}")

    # 4. Demonstrate Persistent Episodic Memory
    print("\nTesting Episodic Memory:")
    mem_vec = torch.randn(config.memory.memory_dim)
    rec = model.memory.insert(
        key="navigation_landmark_1",
        content=mem_vec,
        task_context="gridworld_easy",
        confidence=0.95,
        metadata={"note": "Initial safe zone"},
    )
    print(f"- Inserted record ID: {rec.id}, Version: {rec.version}")

    retrieved = model.memory.retrieve(query=mem_vec, top_k=1)
    print(f"- Retrieved record: {retrieved[0][0].key} with relevance score: {retrieved[0][1]:.4f}")

    # 5. Demonstrate Rollout Planning in GridWorld
    print("\nTesting Closed-loop Planning in GridWorld:")
    env = GridWorld(height=5, width=5, start_pos=(0, 0), goal_pos=(4, 4), state_dim=config.world_model.state_dim)
    planner = ModelBasedRolloutPlanner(config.planner, world_model=model.world_model)
    res = planner.execute_plan(env, max_steps=10)
    print(f"- Planning completed. Steps: {res['steps']}, Invalid actions: {res['invalid_actions']}, Trajectory: {res['trajectory']}")


if __name__ == "__main__":
    run_quickstart()
