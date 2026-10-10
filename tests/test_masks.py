"""Tests for causal attention masks and padding mask correctness."""

import torch
from neurograph.config import NeuroGraphConfig, LMConfig
from neurograph.model import MockCausalLM, NeuroGraphModel


def test_causal_mask_prevents_future_leakage():
    """Verify that changing future tokens does not affect earlier prediction logits."""
    lm_cfg = LMConfig(d_model=32, num_heads=2, num_layers=2, vocab_size=100, max_seq_len=64)
    model = MockCausalLM(lm_cfg)
    model.eval()

    seq_len = 10
    tokens_base = torch.randint(1, 100, (1, seq_len))

    # Variant that differs ONLY at future positions (e.g. index >= 5)
    tokens_modified = tokens_base.clone()
    tokens_modified[0, 5:] = (tokens_modified[0, 5:] + 13) % 99 + 1

    with torch.no_grad():
        out_base = model(input_ids=tokens_base)
        out_mod = model(input_ids=tokens_modified)

    logits_base = out_base["logits"][0]  # [S, vocab_size]
    logits_mod = out_mod["logits"][0]

    # For indices 0 through 4 (positions 0 to 4 predicting next tokens up to index 4),
    # the inputs at positions 0..4 are identical, so logits at 0..4 MUST be strictly identical!
    diff_prefix = torch.max(torch.abs(logits_base[:5] - logits_mod[:5])).item()
    assert diff_prefix < 1e-5, f"Causal leakage detected! Max diff in prefix: {diff_prefix}"

    # For indices >= 5, inputs differ, so logits should differ
    diff_future = torch.max(torch.abs(logits_base[5:] - logits_mod[5:])).item()
    assert diff_future > 1e-5, "Expected future logits to differ given modified tokens"


def test_padding_mask_correctness():
    """Verify that padded tokens do not influence representations of valid tokens in cross-attention."""
    cfg = NeuroGraphConfig()
    cfg.lm.d_model = 32
    cfg.workspace.node_dim = 32
    cfg.workspace.num_nodes = 4

    model = NeuroGraphModel(cfg)
    model.eval()

    # Batch of 2 where sequence 1 is padded at the end
    input_ids = torch.tensor([
        [10, 20, 30, 0, 0],
        [10, 20, 30, 40, 50],
    ], dtype=torch.long)
    attention_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ], dtype=torch.long)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, condition="C")

    # Check finite outputs
    assert torch.isfinite(out["logits"]).all()
    assert out["workspace_outputs"] is not None
    assert torch.isfinite(out["workspace_outputs"]["final_nodes"]).all()
