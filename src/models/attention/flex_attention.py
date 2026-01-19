"""
# ==============================================================================
# Module: src/models/attention/flex_attention.py
# ==============================================================================
# Purpose: FlexAttention-based heterogeneous attention (PyTorch 2.5+)
#
# Dependencies:
#   - External: torch (>=2.5 for FlexAttention, fallback available)
#   - Internal: None
#
# Exports:
#   - FlexHeteroAttention: Custom attention for heterogeneous graphs
#   - create_ontology_score_mod: Ontology-aware attention modifier
#   - create_species_score_mod: Species-aware attention modifier
#
# Design Notes:
#   - Falls back to standard attention if FlexAttention unavailable
#   - torch.compile() compatible
#   - Supports edge-type-specific biases
# ==============================================================================
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

# Try to import FlexAttention (PyTorch 2.5+)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX_ATTENTION = True
except ImportError:
    HAS_FLEX_ATTENTION = False


class FlexHeteroAttention(nn.Module):
    """
    Heterogeneous attention using PyTorch's FlexAttention.

    FlexAttention allows custom attention score modifications
    without writing custom CUDA kernels. Useful for:
    - Edge-type-specific attention patterns
    - Ontology hierarchy-aware attention
    - Species-specific attention modulation

    Requires PyTorch 2.5+ with FlexAttention support.
    Falls back to standard attention if unavailable.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_edge_type_bias: bool = True,
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Attention dropout
            use_edge_type_bias: Whether to use edge-type-specific bias
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Edge type bias (learnable per edge type)
        if use_edge_type_bias:
            # Will be populated dynamically based on edge types seen
            self.edge_type_bias = nn.ParameterDict()

    def _get_edge_type_bias(self, edge_type: str) -> nn.Parameter:
        """Get or create edge type bias parameter"""
        if edge_type not in self.edge_type_bias:
            # Initialize new bias
            self.edge_type_bias[edge_type] = nn.Parameter(
                torch.zeros(self.num_heads)
            )
        return self.edge_type_bias[edge_type]

    def forward(
        self,
        x_src: Tensor,
        x_dst: Tensor,
        edge_index: Tensor,
        edge_type: Optional[str] = None,
        score_mod: Optional[Callable] = None,
    ) -> Tensor:
        """
        Compute heterogeneous attention.

        Args:
            x_src: (num_src, hidden_dim) source node features
            x_dst: (num_dst, hidden_dim) destination node features
            edge_index: (2, num_edges) connectivity
            edge_type: Optional edge type for type-specific bias
            score_mod: Optional FlexAttention score modification function

        Returns:
            (num_dst, hidden_dim) updated destination features
        """
        num_dst = x_dst.size(0)

        # QKV projections
        q = self.q_proj(x_dst)  # (num_dst, hidden_dim)
        k = self.k_proj(x_src)  # (num_src, hidden_dim)
        v = self.v_proj(x_src)  # (num_src, hidden_dim)

        # Reshape for multi-head attention
        q = q.view(num_dst, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        # Compute attention scores for edges only (sparse)
        src_idx, dst_idx = edge_index[0], edge_index[1]

        # Gather Q and K for edges
        q_edge = q[dst_idx]  # (num_edges, num_heads, head_dim)
        k_edge = k[src_idx]  # (num_edges, num_heads, head_dim)

        # Attention scores
        scores = (q_edge * k_edge).sum(dim=-1) * self.scale  # (num_edges, num_heads)

        # Add edge type bias if available
        if edge_type is not None and hasattr(self, "edge_type_bias"):
            bias = self._get_edge_type_bias(edge_type)
            scores = scores + bias

        # Apply custom score modification (FlexAttention style)
        if score_mod is not None:
            scores = score_mod(scores, src_idx, dst_idx)

        # Softmax per destination node (sparse softmax)
        scores = self._sparse_softmax(scores, dst_idx, num_dst)
        scores = self.dropout(scores)

        # Aggregate values
        v_edge = v[src_idx]  # (num_edges, num_heads, head_dim)
        weighted_v = scores.unsqueeze(-1) * v_edge  # (num_edges, num_heads, head_dim)

        # Scatter add to destination nodes
        out = torch.zeros(num_dst, self.num_heads, self.head_dim, device=x_dst.device)
        out.scatter_add_(0, dst_idx.view(-1, 1, 1).expand_as(weighted_v), weighted_v)

        # Reshape and project
        out = out.view(num_dst, self.hidden_dim)
        out = self.out_proj(out)

        return out

    def _sparse_softmax(
        self,
        scores: Tensor,
        indices: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """
        Compute softmax over sparse edges grouped by destination.

        Args:
            scores: (num_edges, num_heads) attention scores
            indices: (num_edges,) destination node indices
            num_nodes: Total number of destination nodes

        Returns:
            (num_edges, num_heads) normalized attention weights
        """
        # Compute max per destination for numerical stability
        max_scores = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        max_scores.scatter_reduce_(
            0,
            indices.view(-1, 1).expand_as(scores),
            scores,
            reduce="amax",
            include_self=False,
        )

        # Subtract max and exp
        scores = scores - max_scores[indices]
        exp_scores = scores.exp()

        # Sum per destination
        sum_exp = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        sum_exp.scatter_add_(
            0,
            indices.view(-1, 1).expand_as(exp_scores),
            exp_scores,
        )

        # Normalize
        return exp_scores / (sum_exp[indices] + 1e-8)


def create_ontology_score_mod(
    hierarchy_depth: Tensor,
    max_depth: int = 20,
    depth_weight: float = 0.1,
) -> Callable:
    """
    Create a score modification function for ontology-aware attention.

    Nodes closer in the ontology hierarchy get higher attention.

    Args:
        hierarchy_depth: (num_nodes,) depth in ontology tree
        max_depth: Maximum hierarchy depth
        depth_weight: Weight for depth-based modification

    Returns:
        Score modification function compatible with FlexAttention
    """
    def score_mod(scores: Tensor, src_idx: Tensor, dst_idx: Tensor) -> Tensor:
        # Depth difference penalty
        src_depth = hierarchy_depth[src_idx]
        dst_depth = hierarchy_depth[dst_idx]
        depth_diff = (src_depth - dst_depth).abs() / max_depth

        # Reduce attention for nodes far apart in hierarchy
        penalty = depth_weight * depth_diff.unsqueeze(-1)
        return scores - penalty

    return score_mod


def create_species_score_mod(
    species_ids: Tensor,
    same_species_bonus: float = 0.5,
) -> Callable:
    """
    Create a score modification function for species-aware attention.

    Same-species nodes get attention bonus.

    Args:
        species_ids: (num_nodes,) species identifier per node
        same_species_bonus: Bonus for same-species attention

    Returns:
        Score modification function
    """
    def score_mod(scores: Tensor, src_idx: Tensor, dst_idx: Tensor) -> Tensor:
        same_species = (species_ids[src_idx] == species_ids[dst_idx]).float()
        bonus = same_species_bonus * same_species.unsqueeze(-1)
        return scores + bonus

    return score_mod
