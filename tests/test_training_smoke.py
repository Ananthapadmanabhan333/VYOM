"""Smoke test for end-to-end multi-objective training on CPU."""

import tempfile
from neurograph.config import NeuroGraphConfig
from neurograph.train import train


def test_training_smoke_reduces_loss():
    cfg = NeuroGraphConfig()
    cfg.experiment_name = "test_smoke"
    cfg.condition = "C"
    cfg.lm.d_model = 32
    cfg.lm.vocab_size = 64
    cfg.workspace.num_nodes = 4
    cfg.workspace.node_dim = 32
    cfg.workspace.recurrent_steps = 2
    cfg.training.learning_rate = 0.005
    cfg.training.batch_size = 4
    cfg.training.max_steps = 30
    cfg.training.device = "cpu"

    with tempfile.TemporaryDirectory() as tmpdir:
        res = train(config=cfg, max_steps=30, output_dir=tmpdir)

        history = res["history"]
        assert len(history["total_loss"]) == 30

        initial_loss = sum(history["total_loss"][:5]) / 5.0
        final_loss = sum(history["total_loss"][-5:]) / 5.0

        assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss}, final={final_loss}"
