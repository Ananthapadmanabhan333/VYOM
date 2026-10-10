"""Dynamic Cognitive Workspace Module for NeuroGraph-LLM."""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurograph.config import WorkspaceConfig
from neurograph.graph_layer import (
    DynamicGraphConnectivity,
    GraphMessagePassingLayer,
    ContextCrossAttention,
    GatedRecurrentStateUpdate,
)


class DynamicCognitiveWorkspace(nn.Module):
    """Dynamic cognitive workspace that iteratively integrates language representations

    via recurrent graph message passing and goal/uncertainty modulation.

    Tensor Dimensions:
        context_features H: [Batch, SeqLen, ContextDim]
        workspace_nodes M:  [Batch, NumNodes, NodeDim]
        adjacency A:        [Batch, NumNodes, NumNodes]
        active_goal:        [Batch, GoalDim]
        uncertainty:        [Batch, UncertaintyDim]
    """

    def __init__(self, config: WorkspaceConfig, context_dim: int):
        super().__init__()
        self.config = config
        self.num_nodes = config.num_nodes
        self.node_dim = config.node_dim
        self.recurrent_steps = config.recurrent_steps
        self.context_dim = context_dim

        # Initial learned workspace slots (slot queries)
        self.initial_slot_queries = nn.Parameter(
            torch.randn(self.num_nodes, self.node_dim) * 0.02
        )

        # Optional projection from context dimension to node dimension
        if context_dim != self.node_dim:
            self.context_proj = nn.Linear(context_dim, self.node_dim)
        else:
            self.context_proj = nn.Identity()

        # Goal and uncertainty projection layers
        self.goal_proj = nn.Linear(config.goal_dim, self.node_dim)
        self.unc_proj = nn.Linear(config.uncertainty_dim, self.node_dim)

        # Graph connectivity computer
        self.connectivity = DynamicGraphConnectivity(
            node_dim=self.node_dim,
            num_nodes=self.num_nodes,
            dynamic=config.dynamic_connectivity,
            top_k=config.top_k_edges,
            dropout=config.dropout,
        )

        # Graph message passing
        self.message_passing = GraphMessagePassingLayer(
            node_dim=self.node_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        # Context cross-attention (nodes querying language tokens H)
        self.cross_attention = ContextCrossAttention(
            node_dim=self.node_dim,
            context_dim=self.node_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        # Gated Recurrent Update: input includes aggregated msg (node_dim) + context msg (node_dim) + goal (node_dim) + unc (node_dim)
        recurrent_input_dim = self.node_dim * 4
        self.recurrent_update = GatedRecurrentStateUpdate(
            node_dim=self.node_dim,
            input_dim=recurrent_input_dim,
            dropout=config.dropout,
        )

        # Output projection / pooling
        self.pool_attention = nn.Linear(self.node_dim, 1)
        self.norm = nn.LayerNorm(self.node_dim) if config.use_layer_norm else nn.Identity()

    def get_initial_states(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create initial slot states replicated across batch."""
        return self.initial_slot_queries.unsqueeze(0).expand(batch_size, -1, -1).to(device)

    def forward(
        self,
        context_features: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        active_goal: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        initial_nodes: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute recurrent cognitive workspace updates.

        Args:
            context_features (torch.Tensor): Language representations [B, L, D_ctx].
            context_mask (Optional[torch.Tensor]): Attention mask [B, L] (1=valid, 0=pad).
            active_goal (Optional[torch.Tensor]): Active goal representation [B, D_goal].
            uncertainty (Optional[torch.Tensor]): Epistemic uncertainty signal [B, D_unc].
            initial_nodes (Optional[torch.Tensor]): Pre-existing workspace states [B, N, D_node].
            steps (Optional[int]): Override default recurrent steps.

        Returns:
            Dict containing:
                'final_nodes': [B, N, D_node]
                'pooled_state': [B, D_node]
                'node_history': List of [B, N, D_node] for t=0..T
                'adj_history': List of [B, N, N] for t=0..T-1
                'reg_loss': Graph regularization scalar
        """
        B, L, _ = context_features.shape
        device = context_features.device
        num_steps = steps if steps is not None else self.recurrent_steps

        # Project context to workspace dimension
        H = self.context_proj(context_features)  # [B, L, D_node]

        # Initialize workspace slots
        if initial_nodes is not None:
            assert initial_nodes.shape == (B, self.num_nodes, self.node_dim)
            m_t = initial_nodes
        else:
            m_t = self.get_initial_states(B, device)

        # Prepare goal features
        if active_goal is not None:
            g = self.goal_proj(active_goal).unsqueeze(1).expand(-1, self.num_nodes, -1)  # [B, N, D_node]
        else:
            g = torch.zeros(B, self.num_nodes, self.node_dim, device=device)

        # Prepare uncertainty features
        if uncertainty is not None:
            u = self.unc_proj(uncertainty).unsqueeze(1).expand(-1, self.num_nodes, -1)  # [B, N, D_node]
        else:
            u = torch.zeros(B, self.num_nodes, self.node_dim, device=device)

        node_history: List[torch.Tensor] = [m_t]
        adj_history: List[torch.Tensor] = []
        logits_history: List[torch.Tensor] = []

        # Recurrent graph updates: M(t+1) = GraphUpdate(M(t), H, G(t))
        for step in range(num_steps):
            # 1. Compute dynamic graph connectivity G(t)
            adj_t, logits_t = self.connectivity(m_t)
            adj_history.append(adj_t)
            logits_history.append(logits_t)

            # 2. Relational message passing across workspace slots
            msg_t = self.message_passing(m_t, adj_t)  # [B, N, D_node]

            # 3. Cross-attention: nodes query language context H
            ctx_t = self.cross_attention(m_t, H, context_mask=context_mask)  # [B, N, D_node]

            # 4. Integrate incoming signals: [GraphMessages, LanguageContext, Goal, Uncertainty]
            incoming = torch.cat([msg_t, ctx_t, g, u], dim=-1)  # [B, N, D_node * 4]

            # 5. Gated recurrent transition: M(t+1) = GRU(M(t), incoming)
            m_t = self.recurrent_update(m_t, incoming)
            node_history.append(m_t)

        m_final = self.norm(m_t)

        # Attention pooling over slots to obtain global workspace representation
        attn_scores = F.softmax(self.pool_attention(m_final), dim=1)  # [B, N, 1]
        pooled_state = torch.sum(attn_scores * m_final, dim=1)  # [B, D_node]

        # Regularization loss: penalize high entropy or encourage sparsity on adjacency
        reg_losses = []
        for adj in adj_history:
            # Entropy: - sum(A * log(A + eps))
            eps = 1e-8
            entropy = -torch.sum(adj * torch.log(adj + eps), dim=-1).mean()
            reg_losses.append(entropy)
        reg_loss = torch.stack(reg_losses).mean() if reg_losses else torch.tensor(0.0, device=device)

        return {
            "final_nodes": m_final,
            "pooled_state": pooled_state,
            "node_history": node_history,
            "adj_history": adj_history,
            "reg_loss": reg_loss,
        }
