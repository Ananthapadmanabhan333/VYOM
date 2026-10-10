"""Synthetic GridWorld Environment for Navigation and Planning Benchmarks."""

from typing import Tuple, List, Set, Dict, Any, Optional
import random
import torch


class GridWorld:
    """Deterministic GridWorld environment with obstacle maps and goal navigation.

    Action Space:
        0: UP    (-1, 0)
        1: DOWN  (+1, 0)
        2: LEFT  (0, -1)
        3: RIGHT (0, +1)
        4: STAY  (0, 0)
    """

    ACTION_MAP = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),   # RIGHT
        4: (0, 0),   # STAY
    }
    ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

    def __init__(
        self,
        height: int = 6,
        width: int = 6,
        obstacles: Optional[Set[Tuple[int, int]]] = None,
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        max_steps: int = 30,
        state_dim: int = 16,
    ):
        self.height = height
        self.width = width
        self.max_steps = max_steps
        self.state_dim = state_dim

        self.obstacles: Set[Tuple[int, int]] = obstacles or set()
        self.start_pos: Tuple[int, int] = start_pos or (0, 0)
        self.goal_pos: Tuple[int, int] = goal_pos or (height - 1, width - 1)

        self.current_pos = self.start_pos
        self.steps_taken = 0

    def reset(
        self,
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset environment to initial or specified state."""
        self.current_pos = start_pos if start_pos is not None else self.start_pos
        if goal_pos is not None:
            self.goal_pos = goal_pos
        self.steps_taken = 0

        obs_tensor = self.get_state_tensor(self.current_pos, self.goal_pos)
        info = {
            "pos": self.current_pos,
            "goal": self.goal_pos,
            "text": self.get_textual_description(),
            "valid_actions": self.get_valid_actions(self.current_pos),
        }
        return obs_tensor, info

    def is_valid_coord(self, r: int, c: int) -> bool:
        """Check if coordinate is inside grid boundaries and not an obstacle."""
        return 0 <= r < self.height and 0 <= c < self.width and (r, c) not in self.obstacles

    def get_valid_actions(self, pos: Optional[Tuple[int, int]] = None) -> List[int]:
        """Return list of valid action IDs from given or current position."""
        r, c = pos if pos is not None else self.current_pos
        valid = []
        for a_idx, (dr, dc) in self.ACTION_MAP.items():
            nr, nc = r + dr, c + dc
            if self.is_valid_coord(nr, nc):
                valid.append(a_idx)
        return valid

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """Execute action, return (next_state_tensor, reward, done, info)."""
        self.steps_taken += 1
        r, c = self.current_pos
        dr, dc = self.ACTION_MAP.get(action, (0, 0))
        nr, nc = r + dr, c + dc

        is_valid = self.is_valid_coord(nr, nc)
        if is_valid:
            self.current_pos = (nr, nc)
        # If invalid, agent remains in place

        reached_goal = (self.current_pos == self.goal_pos)
        timeout = (self.steps_taken >= self.max_steps)
        done = reached_goal or timeout

        # Reward: +10.0 for goal, -0.1 step cost, -0.5 for invalid action
        reward = -0.1
        if not is_valid:
            reward -= 0.5
        if reached_goal:
            reward += 10.0

        obs_tensor = self.get_state_tensor(self.current_pos, self.goal_pos)
        info = {
            "pos": self.current_pos,
            "goal": self.goal_pos,
            "valid_action": is_valid,
            "reached_goal": reached_goal,
            "steps_taken": self.steps_taken,
            "text": self.get_textual_description(),
        }
        return obs_tensor, reward, done, info

    def get_state_tensor(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> torch.Tensor:
        """Create dense state feature representation."""
        vec = torch.zeros(self.state_dim, dtype=torch.float32)
        # Normalized coordinates: [agent_r, agent_c, goal_r, goal_c, delta_r, delta_c, dist]
        r, c = pos
        gr, gc = goal
        vec[0] = (r / max(1, self.height - 1)) * 2.0 - 1.0
        vec[1] = (c / max(1, self.width - 1)) * 2.0 - 1.0
        vec[2] = (gr / max(1, self.height - 1)) * 2.0 - 1.0
        vec[3] = (gc / max(1, self.width - 1)) * 2.0 - 1.0
        vec[4] = (gr - r) / max(1, self.height)
        vec[5] = (gc - c) / max(1, self.width)
        manhattan = abs(gr - r) + abs(gc - c)
        vec[6] = manhattan / max(1, self.height + self.width)
        # Remaining dimensions can represent local wall sensors: 4 directions (up, down, left, right blocked)
        vec[7] = 1.0 if not self.is_valid_coord(r - 1, c) else 0.0
        vec[8] = 1.0 if not self.is_valid_coord(r + 1, c) else 0.0
        vec[9] = 1.0 if not self.is_valid_coord(r, c - 1) else 0.0
        vec[10] = 1.0 if not self.is_valid_coord(r, c + 1) else 0.0
        return vec

    def get_textual_description(self) -> str:
        """Render observation as natural language prompt for the LLM."""
        r, c = self.current_pos
        gr, gc = self.goal_pos
        valid_moves = [self.ACTION_NAMES[a] for a in self.get_valid_actions()]
        return (
            f"You are navigating a {self.height}x{self.width} grid. "
            f"Current position: ({r}, {c}). Goal position: ({gr}, {gc}). "
            f"Valid actions from here: {', '.join(valid_moves)}. "
            f"What is your next action?"
        )

    @classmethod
    def generate_random_map(
        cls,
        height: int = 6,
        width: int = 6,
        obstacle_density: float = 0.2,
        seed: int = 42,
        state_dim: int = 16,
    ) -> "GridWorld":
        """Deterministically generate a solvable grid map with obstacles."""
        rng = random.Random(seed)
        all_coords = [(r, c) for r in range(height) for c in range(width)]

        start = (0, 0)
        goal = (height - 1, width - 1)
        available = [c for c in all_coords if c != start and c != goal]

        num_obs = int(len(available) * obstacle_density)
        # Select obstacles while ensuring a valid path exists (using BFS check)
        for _ in range(50):  # attempts
            obs_sample = set(rng.sample(available, num_obs))
            env = cls(height=height, width=width, obstacles=obs_sample, start_pos=start, goal_pos=goal, state_dim=state_dim)
            if env.has_path(start, goal):
                return env

        # Fallback to no obstacles if random sampling fails to preserve path
        return cls(height=height, width=width, obstacles=set(), start_pos=start, goal_pos=goal, state_dim=state_dim)

    def has_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """Check reachability from start to goal via BFS."""
        from collections import deque
        queue = deque([start])
        visited = {start}
        while queue:
            curr = queue.popleft()
            if curr == goal:
                return True
            r, c = curr
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if self.is_valid_coord(nr, nc) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False
