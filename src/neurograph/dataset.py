"""Expert demonstration dataset generator for cognitive fine-tuning."""

from typing import List, Dict
import torch
from torch.utils.data import Dataset

from environments.gridworld import GridWorld
from neurograph.planner import SearchBasedPlanner


class NavigationDemonstrationDataset(Dataset):
    """Generates expert demonstration trajectories using optimal search (BFS/A*)

    for multi-objective fine-tuning of NeuroGraph-LLM.
    """

    def __init__(
        self,
        num_episodes: int = 100,
        height: int = 6,
        width: int = 6,
        obstacle_density: float = 0.15,
        state_dim: int = 16,
        goal_dim: int = 32,
        uncertainty_dim: int = 8,
        seed: int = 42,
        vocab_size: int = 256,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.samples: List[Dict[str, torch.Tensor]] = []
        planner = SearchBasedPlanner(GridWorld.ACTION_MAP)

        for ep in range(num_episodes):
            env_seed = seed + ep * 7
            env = GridWorld.generate_random_map(
                height=height,
                width=width,
                obstacle_density=obstacle_density,
                seed=env_seed,
                state_dim=state_dim,
            )

            start = env.start_pos
            goal = env.goal_pos
            path = planner.plan_bfs(start, goal, is_valid_fn=env.is_valid_coord, max_depth=30)
            if not path:
                continue

            curr_pos = start
            for a in path:
                s_curr = env.get_state_tensor(curr_pos, goal)
                dr, dc = GridWorld.ACTION_MAP[a]
                next_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
                s_next = env.get_state_tensor(next_pos, goal)

                # Goal representation vector
                g_vec = torch.zeros(goal_dim)
                g_vec[0] = (goal[0] / max(1, height - 1)) * 2.0 - 1.0
                g_vec[1] = (goal[1] / max(1, width - 1)) * 2.0 - 1.0
                g_vec[2] = (next_pos[0] - curr_pos[0])
                g_vec[3] = (next_pos[1] - curr_pos[1])

                # Tokenized prompt representation
                action_name = GridWorld.ACTION_NAMES[a]
                prompt_text = f"Pos: {curr_pos}. Goal: {goal}. Move: {action_name}"
                words = prompt_text.split()
                token_ids = [abs(hash(w)) % (vocab_size - 1) + 1 for w in words]
                if len(token_ids) < max_seq_len:
                    token_ids += [0] * (max_seq_len - len(token_ids))
                else:
                    token_ids = token_ids[:max_seq_len]

                tok_tensor = torch.tensor(token_ids, dtype=torch.long)

                self.samples.append({
                    "input_ids": tok_tensor,
                    "labels": tok_tensor.clone(),
                    "current_state": s_curr,
                    "target_action": torch.tensor(a, dtype=torch.long),
                    "target_next_state": s_next,
                    "active_goal": g_vec,
                    "uncertainty": torch.zeros(uncertainty_dim),
                })
                curr_pos = next_pos

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_demonstrations(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate individual samples into batched tensors."""
    return {
        "input_ids": torch.stack([s["input_ids"] for s in batch]),
        "labels": torch.stack([s["labels"] for s in batch]),
        "current_state": torch.stack([s["current_state"] for s in batch]),
        "target_action": torch.stack([s["target_action"] for s in batch]),
        "target_next_state": torch.stack([s["target_next_state"] for s in batch]),
        "active_goal": torch.stack([s["active_goal"] for s in batch]),
        "uncertainty": torch.stack([s["uncertainty"] for s in batch]),
    }
