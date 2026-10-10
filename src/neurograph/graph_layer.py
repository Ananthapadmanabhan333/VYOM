"""Graph layers and recurrent message-passing primitives for the cognitive workspace.

Mathematical Formulation & Distinctions from Standard Attention:
1. Standard Self-Attention:
   - Stateless pairwise mapping: Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V.
   - Computes static token-to-token projections in a single pass without recurrence.
   - All tokens participate equally without topology constraints or recurrent memory retention.

2. NeuroGraph Dynamic Cognitive Workspace:
   - Maintains an explicit set of persistent latent workspace slots M(t) in R^{B x N x D}.
   - Dynamic Connectivity G(t) is computed iteratively: G(t) = Sparsify(Softmax(Q_M K_M^T / sqrt(d))).
   - Recurrent Message Passing: M(t+1) = RecurrentUpdate(M(t), MessagePass(M(t), G(t)), ContextCrossAttend(M(t), H)).
   - Updates occur via a gated recurrent transition (e.g., GRU gate), preserving temporal continuity across
     discrete reasoning steps t = 1, ..., T.
   - Explicitly modulated by goal representations and uncertainty states.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGraphConnectivity(nn.Module):
    """Computes dynamic or static directed edge weights between workspace nodes.

    Attributes:
        node_dim (int): Dimensionality of each workspace node state.
        num_nodes (int): Number of workspace nodes N.
        dynamic (bool): If True, edges depend on current node states. If False, uses learned static adjacency.
        top_k (Optional[int]): If provided, only retains top-k outgoing/incoming edges per node.
    """

    def __init__(
        self,
        node_dim: int,
        num_nodes: int,
        dynamic: bool = True,
        top_k: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.num_nodes = num_nodes
        self.dynamic = dynamic
        self.top_k = top_k
        self.scale = 1.0 / math.sqrt(node_dim)

        if self.dynamic:
            self.q_proj = nn.Linear(node_dim, node_dim, bias=False)
            self.k_proj = nn.Linear(node_dim, node_dim, bias=False)
        else:
            # Condition D: Learned static adjacency logits
            self.static_adj_logits = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute normalized adjacency matrix A and raw similarity logits S.

        Args:
            nodes (torch.Tensor): Shape [B, N, D]

        Returns:
            A (torch.Tensor): Row-normalized adjacency matrix [B, N, N], where A[b, i, j] is edge j -> i.
            logits (torch.Tensor): Raw pre-softmax edge logits [B, N, N].
        """
        B, N, D = nodes.shape
        assert N == self.num_nodes, f"Expected {self.num_nodes} nodes, got {N}"
        assert D == self.node_dim, f"Expected node dim {self.node_dim}, got {D}"

        if self.dynamic:
            q = self.q_proj(nodes)  # [B, N, D]
            k = self.k_proj(nodes)  # [B, N, D]
            logits = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B, N, N]
        else:
            # Broadcast static logits across batch
            logits = self.static_adj_logits.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]

        if self.top_k is not None and self.top_k < N:
            # Sparse top-k message passing: keep top-k incoming edges per node
            top_vals, _ = torch.topk(logits, k=self.top_k, dim=-1)
            min_top = top_vals[..., -1, None]  # [B, N, 1]
            mask = logits < min_top
            masked_logits = logits.masked_fill(mask, -1e9)
            adj = F.softmax(masked_logits, dim=-1)
        else:
            adj = F.softmax(logits, dim=-1)

        adj = self.dropout(adj)
        return adj, logits


class GraphMessagePassingLayer(nn.Module):
    """Multi-head relational message passing over workspace nodes."""

    def __init__(
        self,
        node_dim: int,
        num_heads: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert node_dim % num_heads == 0, f"node_dim ({node_dim}) must be divisible by num_heads ({num_heads})"
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads

        self.v_proj = nn.Linear(node_dim, node_dim, bias=False)
        self.out_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Aggregate node features based on graph adjacency A.

        Args:
            nodes (torch.Tensor): Node states [B, N, D].
            adj (torch.Tensor): Row-normalized adjacency [B, N, N].

        Returns:
            aggregated (torch.Tensor): Messages [B, N, D].
        """
        B, N, D = nodes.shape
        v = self.v_proj(nodes)  # [B, N, D]

        # Multi-head aggregation:
        # Reshape to [B, num_heads, N, head_dim]
        v_heads = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        adj_heads = adj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [B, H, N, N]

        # Messages: A * V
        msgs = torch.matmul(adj_heads, v_heads)  # [B, H, N, head_dim]
        msgs = msgs.permute(0, 2, 1, 3).contiguous().view(B, N, D)  # [B, N, D]

        out = self.out_proj(msgs)
        return self.dropout(out)


class ContextCrossAttention(nn.Module):
    """Allows workspace nodes to cross-attend to contextual language representations H."""

    def __init__(
        self,
        node_dim: int,
        context_dim: int,
        num_heads: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(node_dim, node_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, node_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, node_dim, bias=False)
        self.out_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        nodes: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-attend nodes to language context tokens.

        Args:
            nodes (torch.Tensor): [B, N, D_node]
            context (torch.Tensor): [B, L, D_ctx]
            context_mask (Optional[torch.Tensor]): [B, L] boolean mask (True for valid tokens, False for padding).

        Returns:
            attended (torch.Tensor): Contextual updates for nodes [B, N, D_node].
        """
        B, N, _ = nodes.shape
        L = context.shape[1]

        q = self.q_proj(nodes).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, head_dim]
        k = self.k_proj(context).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, head_dim]
        v = self.v_proj(context).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, head_dim]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, L]

        if context_mask is not None:
            # Expand mask [B, 1, 1, L] and cast to bool
            bool_mask = context_mask.bool().unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~bool_mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_msgs = torch.matmul(attn_weights, v)  # [B, H, N, head_dim]
        context_msgs = context_msgs.transpose(1, 2).contiguous().view(B, N, self.node_dim)

        return self.out_proj(context_msgs)


class GatedRecurrentStateUpdate(nn.Module):
    """Gated recurrent state updater (GRU-style) for workspace node states M(t)."""

    def __init__(self, node_dim: int, input_dim: int, dropout: float = 0.0):
        super().__init__()
        self.node_dim = node_dim
        self.input_dim = input_dim

        # GRU gating gates: update gate z, reset gate r, candidate state h_tilde
        self.w_z = nn.Linear(node_dim + input_dim, node_dim)
        self.w_r = nn.Linear(node_dim + input_dim, node_dim)
        self.w_h = nn.Linear(node_dim + input_dim, node_dim)

        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, m_prev: torch.Tensor, incoming_inputs: torch.Tensor) -> torch.Tensor:
        """Perform gated recurrent update: M(t+1) = GRU(M(t), incoming_inputs).

        Args:
            m_prev (torch.Tensor): Current node states [B, N, D].
            incoming_inputs (torch.Tensor): Integrated input features [B, N, input_dim].

        Returns:
            m_next (torch.Tensor): Updated node states [B, N, D].
        """
        combined = torch.cat([m_prev, incoming_inputs], dim=-1)
        z = torch.sigmoid(self.w_z(combined))
        r = torch.sigmoid(self.w_r(combined))

        reset_combined = torch.cat([r * m_prev, incoming_inputs], dim=-1)
        h_tilde = torch.tanh(self.w_h(reset_combined))

        # Gated combination
        m_cand = (1.0 - z) * m_prev + z * h_tilde

        # Residual connection + LayerNorm
        m_next = self.norm(m_prev + self.dropout(m_cand))
        return m_next
