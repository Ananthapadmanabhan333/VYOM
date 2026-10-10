"""Tests for checkpoint saving, loading, and consistency."""

import tempfile
from pathlib import Path
import torch
from neurograph.config import NeuroGraphConfig
from neurograph.model import NeuroGraphModel
from neurograph.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_save_and_load_consistency():
    cfg = NeuroGraphConfig()
    cfg.lm.d_model = 32
    cfg.workspace.node_dim = 32
    cfg.workspace.num_nodes = 4

    model1 = NeuroGraphModel(cfg)
    model1.eval()
    optimizer = torch.optim.Adam(model1.parameters(), lr=1e-3)

    sample_input = torch.randint(1, cfg.lm.vocab_size, (2, 8))
    with torch.no_grad():
        out1 = model1(sample_input, condition="C")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "model_ckpt.pt"
        save_checkpoint(ckpt_path, model1, optimizer=optimizer, step=15, config=cfg)

        model2 = NeuroGraphModel(cfg)
        meta = load_checkpoint(ckpt_path, model2)
        model2.eval()

        assert meta["step"] == 15

        # Verify weights are identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Model parameters differ after checkpoint load"

        # Verify outputs on identical input match exactly
        with torch.no_grad():
            out2 = model2(sample_input, condition="C")
        assert torch.allclose(out1["logits"], out2["logits"], atol=1e-5)
