"""Tests for expert dataset generation and cognitive fine-tuning."""

import tempfile
from pathlib import Path
from neurograph.config import NeuroGraphConfig
from neurograph.dataset import NavigationDemonstrationDataset
from neurograph.finetune import finetune


def test_demonstration_dataset_generation():
    ds = NavigationDemonstrationDataset(num_episodes=5, seed=123)
    assert len(ds) > 0

    sample = ds[0]
    assert "input_ids" in sample
    assert "current_state" in sample
    assert "target_action" in sample
    assert 0 <= sample["target_action"].item() <= 4


def test_finetuning_converges():
    cfg = NeuroGraphConfig()
    cfg.condition = "C"
    cfg.lm.d_model = 32
    cfg.workspace.node_dim = 32
    cfg.workspace.num_nodes = 4
    cfg.workspace.recurrent_steps = 2
    cfg.training.batch_size = 4
    cfg.training.device = "cpu"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_ckpt = Path(tmpdir) / "test_ft.pt"
        res = finetune(
            config=cfg,
            epochs=2,
            train_episodes=10,
            val_episodes=4,
            output_path=str(out_ckpt),
        )

        assert out_ckpt.exists()
        assert len(res["history"]["action_acc"]) == 2
        # Accuracy should be positive and non-zero
        assert res["history"]["action_acc"][-1] > 0.0
