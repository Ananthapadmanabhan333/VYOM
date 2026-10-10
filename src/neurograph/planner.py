"""Goal-Directed Planner Module for NeuroGraph-LLM."""

from typing import List, Tuple, Dict, Any, Optional
import heapq
from collections import deque
import torch

from neurograph.config import PlannerConfig
from neurograph.world_model import WorldStatePredictor


class SearchBasedPlanner:
    """Conventional baseline planner using Breadth-First Search (BFS) or A*."""

    def __init__(self, action_map: Dict[int, Tuple[int, int]]):
        self.action_map = action_map

    def plan_bfs(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        is_valid_fn,
        max_depth: int = 50,
    ) -> List[int]:
        """Breadth-first search path planner returning list of action IDs."""
        if start == goal:
            return []

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            curr, path = queue.popleft()
            if len(path) >= max_depth:
                continue

            r, c = curr
            for a_idx, (dr, dc) in self.action_map.items():
                nr, nc = r + dr, c + dc
                if (nr, nc) == goal:
                    return path + [a_idx]

                if is_valid_fn(nr, nc) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [a_idx]))

        return []  # No path found within max_depth

    def plan_astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        is_valid_fn,
        max_depth: int = 50,
    ) -> List[int]:
        """A* search with Manhattan distance heuristic."""
        if start == goal:
            return []

        def heuristic(p):
            return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

        # Priority queue entries: (f_score, g_score, current_pos, path)
        pq = [(heuristic(start), 0, start, [])]
        best_g = {start: 0}

        while pq:
            _, g, curr, path = heapq.heappop(pq)
            if curr == goal:
                return path
            if len(path) >= max_depth:
                continue

            r, c = curr
            for a_idx, (dr, dc) in self.action_map.items():
                nr, nc = r + dr, c + dc
                if is_valid_fn(nr, nc):
                    new_g = g + 1
                    if (nr, nc) not in best_g or new_g < best_g[(nr, nc)]:
                        best_g[(nr, nc)] = new_g
                        f_score = new_g + heuristic((nr, nc))
                        heapq.heappush(pq, (f_score, new_g, (nr, nc), path + [a_idx]))

        return []


class ModelBasedRolloutPlanner:
    """Bounded model-based rollout planner utilizing the learned WorldStatePredictor.

    Simulates future trajectories using state predictions and scores actions based on
    heuristic distance to goal and epistemic uncertainty penalties.
    """

    def __init__(
        self,
        config: PlannerConfig,
        world_model: Optional[WorldStatePredictor] = None,
        policy_model: Optional[Any] = None,
        action_dim: int = 5,
    ):
        self.config = config
        self.world_model = world_model
        self.policy_model = policy_model
        self.action_dim = action_dim
        self.max_depth = config.max_depth
        self.discount = config.discount
        self.heuristic_weight = config.heuristic_weight

    def evaluate_state_heuristic(self, state_tensor: torch.Tensor) -> float:
        """Estimate distance from state tensor: uses index 6 (normalized Manhattan distance)."""
        if state_tensor.shape[-1] > 6:
            dist = float(state_tensor[6].item())
            return -dist * self.heuristic_weight
        return 0.0

    def select_action(
        self,
        current_state: torch.Tensor,
        valid_actions: Optional[List[int]] = None,
        active_goal: Optional[torch.Tensor] = None,
        depth: Optional[int] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Select best candidate action combining policy network prior, world model

        rollouts, and goal distance heuristics.
        """
        device = current_state.device
        candidates = valid_actions if (valid_actions is not None and len(valid_actions) > 0) else list(range(self.action_dim))
        lookahead = depth if depth is not None else min(2, self.max_depth)

        best_score = -float("inf")
        best_action = candidates[0]
        action_scores = {}

        # 1. Policy network prior from cognitive workspace
        policy_logits = None
        if self.policy_model is not None:
            self.policy_model.eval()
            with torch.no_grad():
                goal_in = active_goal
                if goal_in is None and current_state.shape[-1] >= 4:
                    goal_dim = getattr(self.policy_model.config.workspace, "goal_dim", 32)
                    goal_in = torch.zeros(1, goal_dim, device=device)
                    goal_in[0, :min(4, goal_dim)] = current_state[2:6]

                dummy_ids = torch.tensor([[1]], dtype=torch.long, device=device)
                p_out = self.policy_model(
                    input_ids=dummy_ids,
                    active_goal=goal_in,
                    current_state=current_state.unsqueeze(0),
                    condition=self.policy_model.config.condition,
                )
                if p_out["action_logits"] is not None:
                    policy_logits = p_out["action_logits"][0]

        # 2. Score candidate actions
        for a in candidates:
            score = 0.0
            if policy_logits is not None:
                score += float(policy_logits[a].item()) * 2.0

            if self.world_model is not None:
                self.world_model.eval()
                with torch.no_grad():
                    s0 = current_state.unsqueeze(0).to(device)
                    a_tensor = torch.tensor([a], dtype=torch.long, device=device)
                    pred_s1, unc1 = self.world_model(s0, a_tensor)

                    score += self.evaluate_state_heuristic(pred_s1[0])
                    score -= float(unc1[0].mean().item()) * 0.1

                    if lookahead > 1:
                        future_scores = []
                        for a2 in range(self.action_dim):
                            a2_t = torch.tensor([a2], dtype=torch.long, device=device)
                            pred_s2, _ = self.world_model(pred_s1, a2_t)
                            future_scores.append(self.evaluate_state_heuristic(pred_s2[0]))
                        if future_scores:
                            score += self.discount * max(future_scores)

            action_scores[a] = score
            if score > best_score:
                best_score = score
                best_action = a

        return best_action, {"scores": action_scores, "best_score": best_score}

    def execute_plan(
        self,
        env,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute closed-loop planning with environment feedback and replanning."""
        steps_limit = max_steps or self.max_depth
        obs, info = env.reset()

        trajectory = [info["pos"]]
        actions_taken = []
        invalid_count = 0
        success = False

        for step in range(steps_limit):
            valid_actions = env.get_valid_actions()
            action, plan_info = self.select_action(obs, valid_actions=valid_actions)
            actions_taken.append(action)

            obs, reward, done, step_info = env.step(action)
            trajectory.append(step_info["pos"])

            if not step_info["valid_action"]:
                invalid_count += 1

            if step_info["reached_goal"]:
                success = True
                break

            if done:
                break

        return {
            "success": success,
            "steps": len(actions_taken),
            "invalid_actions": invalid_count,
            "trajectory": trajectory,
            "actions": actions_taken,
        }
