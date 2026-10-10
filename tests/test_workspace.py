"""Tests for Cognitive Workspace and Graph Layers."""

import torch
from neurograph.config import WorkspaceConfig
from neurograph.cognitive_workspace import DynamicCognitiveWorkspace
from neurograph.graph_layer import DynamicGraphConnectivity


def test_graph_connectivity_normalization():
    """Verify that graph adjacency matrix rows sum to 1.0 (valid probability distribution)."""
    conn = DynamicGraphConnectivity(node_dim=32, num_nodes=6, dynamic=True)
    nodes = torch.randn(4, 6, 32)
    adj, logits = conn(nodes)

    assert adj.shape == (4, 6, 6)
    row_sums = torch.sum(adj, dim=-1)
    ones = torch.ones_like(row_sums)
    assert torch.allclose(row_sums, ones, atol=1e-5), f"Adjacency rows do not sum to 1: {row_sums}"


def test_sparse_top_k_connectivity():
    """Verify that top-k connectivity only retains k non-zero edges per row."""
    top_k = 2
    conn = DynamicGraphConnectivity(node_dim=32, num_nodes=6, dynamic=True, top_k=top_k)
    nodes = torch.randn(2, 6, 32)
    adj, _ = conn(nodes)

    # In each row, exactly top_k entries should be non-zero (> 1e-6)
    non_zero_count = (adj > 1e-6).sum(dim=-1)
    assert (non_zero_count == top_k).all(), f"Expected {top_k} non-zero edges, got {non_zero_count}"


def test_recurrent_steps_difference():
    """Verify that multiple recurrent steps produce progressive state refinements compared to 1 step."""
    cfg = WorkspaceConfig(num_nodes=4, node_dim=32, recurrent_steps=3)
    ws = DynamicCognitiveWorkspace(cfg, context_dim=32)

    context = torch.randn(2, 5, 32)
    out = ws(context, steps=3)

    nodes_t0 = out["node_history"][0]
    nodes_t1 = out["node_history"][1]
    nodes_t2 = out["node_history"][2]
    nodes_t3 = out["node_history"][3]

    diff_0_1 = torch.norm(nodes_t1 - nodes_t0).item()
    diff_1_2 = torch.norm(nodes_t2 - nodes_t1).item()
    diff_2_3 = torch.norm(nodes_t3 - nodes_t2).item()

    assert diff_0_1 > 0.01, "Step 1 did not update state"
    assert diff_1_2 > 0.01, "Step 2 did not update state"
    assert diff_2_3 > 0.01, "Step 3 did not update state"
