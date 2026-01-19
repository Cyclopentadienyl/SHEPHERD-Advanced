"""
# ==============================================================================
# Module: src/models/encoders/type_encoder.py
# ==============================================================================
# Purpose: Type embeddings for heterogeneous graph nodes and edges
#
# Dependencies:
#   - External: torch (>=2.9)
#   - Internal: None
#
# Exports:
#   - NodeTypeEncoder: Learnable node type embeddings
#   - EdgeTypeEncoder: Learnable edge/relation type embeddings
#
# Design Notes:
#   - torch.compile() compatible
#   - Supports fusion modes: add, concat, gate
# ==============================================================================
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class NodeTypeEncoder(nn.Module):
    """
    Encodes node types into learnable embeddings.

    Supports both:
    1. Type-only encoding (no input features)
    2. Type + feature fusion (concatenate or add)

    torch.compile() compatible.
    """

    def __init__(
        self,
        num_types: int,
        hidden_dim: int,
        fusion_mode: str = "add",  # "add", "concat", "gate"
    ):
        """
        Args:
            num_types: Number of distinct node types
            hidden_dim: Output embedding dimension
            fusion_mode: How to combine type embedding with features
        """
        super().__init__()
        self.num_types = num_types
        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode

        # Learnable type embeddings
        self.type_embedding = nn.Embedding(num_types, hidden_dim)

        # For gated fusion
        if fusion_mode == "gate":
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with small values"""
        nn.init.normal_(self.type_embedding.weight, std=0.02)

    def forward(
        self,
        type_indices: Tensor,
        features: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            type_indices: (num_nodes,) node type indices
            features: Optional (num_nodes, hidden_dim) input features

        Returns:
            (num_nodes, hidden_dim) encoded representations
        """
        type_emb = self.type_embedding(type_indices)

        if features is None:
            return type_emb

        if self.fusion_mode == "add":
            return features + type_emb
        elif self.fusion_mode == "concat":
            # Caller should handle dimension change
            return torch.cat([features, type_emb], dim=-1)
        elif self.fusion_mode == "gate":
            combined = torch.cat([features, type_emb], dim=-1)
            gate_value = self.gate(combined)
            return gate_value * features + (1 - gate_value) * type_emb
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")


class EdgeTypeEncoder(nn.Module):
    """
    Encodes edge types into learnable embeddings.

    Used for:
    1. Edge-type-specific message passing weights
    2. Attention score modification based on relation type

    torch.compile() compatible.
    """

    def __init__(
        self,
        num_types: int,
        hidden_dim: int,
    ):
        """
        Args:
            num_types: Number of distinct edge/relation types
            hidden_dim: Output embedding dimension
        """
        super().__init__()
        self.num_types = num_types
        self.hidden_dim = hidden_dim

        self.type_embedding = nn.Embedding(num_types, hidden_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.type_embedding.weight, std=0.02)

    def forward(self, type_indices: Tensor) -> Tensor:
        """
        Args:
            type_indices: (num_edges,) edge type indices

        Returns:
            (num_edges, hidden_dim) edge type embeddings
        """
        return self.type_embedding(type_indices)
