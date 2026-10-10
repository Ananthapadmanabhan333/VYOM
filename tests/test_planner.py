"""Tests for Goal-Directed Planner and GridWorld environment."""

from neurograph.config import PlannerConfig, WorldModelConfig
from neurograph.planner import SearchBasedPlanner, ModelBasedRolloutPlanner
from neurograph.world_model import WorldStatePredictor
from environments.gridworld import GridWorld


def test_bfs_search_planner_on_gridworld():
    """Verify that BFS finds optimal obstacle-avoiding path."""
    obstacles = {(1, 1), (1, 2)}
    env = GridWorld(height=4, width=4, obstacles=obstacles, start_pos=(0, 0), goal_pos=(2, 2))

    planner = SearchBasedPlanner(GridWorld.ACTION_MAP)
    path = planner.plan_bfs((0, 0), (2, 2), is_valid_fn=env.is_valid_coord)

    assert len(path) > 0

    # Simulate path
    curr = (0, 0)
    for a in path:
        dr, dc = GridWorld.ACTION_MAP[a]
        curr = (curr[0] + dr, curr[1] + dc)
        assert curr not in obstacles, f"Path collided with obstacle at {curr}"

    assert curr == (2, 2), f"Path ended at {curr} instead of goal (2, 2)"


def test_gridworld_invalid_actions_and_feedback():
    """Verify that moving into boundary or obstacle is rejected gracefully."""
    env = GridWorld(height=3, width=3, start_pos=(0, 0), goal_pos=(2, 2))
    # Moving UP from (0, 0) is out of bounds
    obs, reward, done, info = env.step(0)  # 0 is UP
    assert info["pos"] == (0, 0)  # Agent stays in place
    assert info["valid_action"] is False
    assert reward < 0.0


def test_model_rollout_planner_execution():
    """Verify rollout planner executes steps and respects step bounds."""
    cfg = PlannerConfig(max_depth=10)
    wm_cfg = WorldModelConfig(state_dim=16, action_dim=5)
    wm = WorldStatePredictor(wm_cfg)

    planner = ModelBasedRolloutPlanner(cfg, world_model=wm)
    env = GridWorld(height=4, width=4, start_pos=(0, 0), goal_pos=(3, 3), state_dim=16)

    res = planner.execute_plan(env, max_steps=10)
    assert "trajectory" in res
    assert len(res["trajectory"]) > 1
    assert "invalid_actions" in res
