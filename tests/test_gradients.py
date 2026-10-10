"""Tests for gradient propagation and numerical stability."""

import torch
from neurograph.config import NeuroGraphConfig
from neurograph.model import NeuroGraphModel


def test_finite_gradients_and_graph_to_lm_flow():
    """Verify that language generation loss propagates gradients through the adapter

    and into the recurrent cognitive workspace.
    """
    cfg = NeuroGraphConfig()
    cfg.lm.d_model = 32
    cfg.lm.vocab_size = 100
    cfg.workspace.num_nodes = 4
    cfg.workspace.node_dim = 32
    cfg.workspace.recurrent_steps = 2
    cfg.condition = "C"

    model = NeuroGraphModel(cfg)
    model.train()
    # The gated adapter starts as an exact identity (gate=0). Open it so gradients reach
    # adapter internals and the workspace (the init behaviour is tested separately).
    with torch.no_grad():
        model.adapter.gate.fill_(0.5)

    B, S = 2, 8
    input_ids = torch.randint(1, 100, (B, S))
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels, condition="C")
    lm_loss = outputs["lm_loss"]

    assert lm_loss is not None
    assert torch.isfinite(lm_loss).item()

    lm_loss.backward()

    # 1. Check adapter gradients
    for name, p in model.adapter.named_parameters():
        assert p.grad is not None, f"Adapter parameter {name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"Adapter parameter {name} has non-finite gradient"
        assert torch.norm(p.grad) > 0.0, f"Adapter parameter {name} has zero gradient"

    # 2. Check recurrent workspace gradients
    assert model.workspace.initial_slot_queries.grad is not None
    assert torch.isfinite(model.workspace.initial_slot_queries.grad).all()
    assert torch.norm(model.workspace.initial_slot_queries.grad) > 0.0

    # 3. Check connectivity gradients
    if cfg.workspace.dynamic_connectivity:
        assert model.workspace.connectivity.q_proj.weight.grad is not None
        assert torch.isfinite(model.workspace.connectivity.q_proj.weight.grad).all()

    # 4. Check recurrent update gate gradients
    assert model.workspace.recurrent_update.w_z.weight.grad is not None
    assert torch.isfinite(model.workspace.recurrent_update.w_z.weight.grad).all()


def test_gated_adapter_is_identity_at_init_and_gate_learns():
    """At init the workspace must not change LM outputs, but the gate must receive gradient."""
    torch.manual_seed(0)
    cfg = NeuroGraphConfig()
    cfg.lm.d_model = 32
    cfg.lm.vocab_size = 100
    cfg.workspace.num_nodes = 4
    cfg.workspace.node_dim = 32
    model = NeuroGraphModel(cfg)
    model.eval()
    ids = torch.randint(1, 100, (2, 8))
    with torch.no_grad():
        logits_a = model(ids, condition="A")["logits"]
        logits_c = model(ids, condition="C")["logits"]
    assert torch.allclose(logits_a, logits_c, atol=1e-6)
    assert model.adapter_gate() == 0.0

    model.train()
    out = model(ids, labels=ids, condition="C")
    out["lm_loss"].backward()
    assert model.adapter.gate.grad is not None
    assert torch.norm(model.adapter.gate.grad) > 0.0
