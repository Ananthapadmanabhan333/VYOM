"""Tests for tensor shapes and data types across NeuroGraph-LLM modules."""

import torch
from neurograph.config import NeuroGraphConfig
from neurograph.model import NeuroGraphModel
from neurograph.cognitive_workspace import DynamicCognitiveWorkspace
from neurograph.world_model import WorldStatePredictor


def test_workspace_shapes():
    cfg = NeuroGraphConfig()
    cfg.workspace.num_nodes = 6
    cfg.workspace.node_dim = 32
    cfg.workspace.recurrent_steps = 3

    ws = DynamicCognitiveWorkspace(cfg.workspace, context_dim=64)

    B, L = 4, 12
    context = torch.randn(B, L, 64)
    active_goal = torch.randn(B, cfg.workspace.goal_dim)
    uncertainty = torch.randn(B, cfg.workspace.uncertainty_dim)

    out = ws(context, active_goal=active_goal, uncertainty=uncertainty)

    assert out["final_nodes"].shape == (B, 6, 32)
    assert out["pooled_state"].shape == (B, 32)
    assert len(out["adj_history"]) == 3
    assert out["adj_history"][0].shape == (B, 6, 6)
    assert len(out["node_history"]) == 4  # t=0, 1, 2, 3


def test_world_model_shapes():
    cfg = NeuroGraphConfig()
    cfg.world_model.state_dim = 16
    cfg.world_model.action_dim = 5
    cfg.world_model.hidden_dim = 32

    wm = WorldStatePredictor(cfg.world_model)

    B = 8
    states = torch.randn(B, 16)
    actions = torch.randint(0, 5, (B,))

    mean, unc = wm(states, actions)
    assert mean.shape == (B, 16)
    assert unc.shape == (B, 16)
    assert mean.dtype == torch.float32


def test_model_end_to_end_shapes():
    cfg = NeuroGraphConfig()
    cfg.lm.d_model = 32
    cfg.lm.vocab_size = 100
    cfg.workspace.num_nodes = 4
    cfg.workspace.node_dim = 32

    model = NeuroGraphModel(cfg)

    B, S = 2, 10
    input_ids = torch.randint(1, 100, (B, S))

    # Condition C
    out_c = model(input_ids, condition="C")
    assert "logits" in out_c
    assert out_c["logits"].shape[-1] == 100
    assert out_c["action_logits"].shape == (B, cfg.world_model.action_dim)

    # Condition A
    out_a = model(input_ids, condition="A")
    assert out_a["logits"].shape == (B, S, 100)
    assert out_a["workspace_outputs"] is None
