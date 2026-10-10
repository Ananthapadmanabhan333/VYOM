"""Tests for autoregressive language generation interface."""

import torch
from neurograph.config import NeuroGraphConfig
from neurograph.model import NeuroGraphModel
from neurograph.generation import CognitiveGenerator


def test_autoregressive_generation_greedy_and_sample():
    cfg = NeuroGraphConfig()
    cfg.lm.d_model = 32
    cfg.lm.vocab_size = 100
    cfg.workspace.node_dim = 32
    cfg.workspace.num_nodes = 4

    model = NeuroGraphModel(cfg)
    generator = CognitiveGenerator(model)

    prompt = torch.tensor([[10, 25, 40]], dtype=torch.long)

    # 1. Greedy generation (Condition C)
    gen_greedy = generator.generate(
        input_ids=prompt,
        max_new_tokens=6,
        do_sample=False,
        condition="C",
    )
    assert gen_greedy.shape == (1, 3 + 6)
    assert (gen_greedy[:, :3] == prompt).all()

    # Determinism of greedy generation
    gen_greedy_2 = generator.generate(
        input_ids=prompt,
        max_new_tokens=6,
        do_sample=False,
        condition="C",
    )
    assert (gen_greedy == gen_greedy_2).all(), "Greedy generation should be strictly deterministic"

    # 2. Sampling generation
    gen_sample = generator.generate(
        input_ids=prompt,
        max_new_tokens=6,
        do_sample=True,
        temperature=0.8,
        condition="C",
    )
    assert gen_sample.shape == (1, 3 + 6)

    # 3. Base model without graph (Condition A)
    gen_base = generator.generate(
        input_ids=prompt,
        max_new_tokens=4,
        condition="A",
    )
    assert gen_base.shape == (1, 3 + 4)
