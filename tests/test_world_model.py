"""Tests for WorldStatePredictor module."""

import torch
import torch.optim as optim
from neurograph.config import WorldModelConfig
from neurograph.world_model import WorldStatePredictor


def test_world_model_training_and_uncertainty():
    """Verify that the world model trains on transitions and reduces MSE loss."""
    cfg = WorldModelConfig(state_dim=8, action_dim=4, hidden_dim=32, predict_uncertainty=True)
    model = WorldStatePredictor(cfg)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Synthetic transition rule: delta = +0.2 * action
    torch.manual_seed(42)
    s0 = torch.randn(100, 8)
    actions = torch.randint(0, 4, (100,))
    s1 = s0.clone()
    for i in range(100):
        s1[i] += 0.2 * (actions[i].item() + 1)

    # Train for 40 steps
    initial_loss = None
    final_loss = None
    for step in range(40):
        optimizer.zero_grad()
        out = model.compute_loss(s0, actions, s1)
        loss = out["loss"]
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if step == 39:
            final_loss = loss.item()

    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"

    # Evaluate on held-out transitions
    test_s0 = torch.randn(20, 8)
    test_actions = torch.randint(0, 4, (20,))
    test_s1 = test_s0.clone()
    for i in range(20):
        test_s1[i] += 0.2 * (test_actions[i].item() + 1)

    metrics = model.evaluate_transitions(test_s0, test_actions, test_s1)
    assert "eval_mse" in metrics
    assert "avg_uncertainty" in metrics
    assert metrics["eval_mse"] < 0.5
    assert metrics["avg_uncertainty"] > 0.0
