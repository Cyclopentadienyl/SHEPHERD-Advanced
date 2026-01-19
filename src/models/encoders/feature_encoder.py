"""
# ==============================================================================
# Module: src/models/encoders/feature_encoder.py
# ==============================================================================
# Purpose: Feature projection for heterogeneous graph nodes
#
# Dependencies:
#   - External: torch (>=2.9)
#   - Internal: None
#
# Exports:
#   - HeteroFeatureEncoder: Projects heterogeneous features to unified space
#
# Design Notes:
#   - torch.compile() compatible via dict-based operations
#   - Supports optional type embeddings per node type
# ==============================================================================
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class HeteroFeatureEncoder(nn.Module):
    """
    Projects heterogeneous node features to unified hidden space.

    Each node type can have different input feature dimensions,
    but all are projected to the same hidden_dim for message passing.

    torch.compile() compatible via dict-based operations.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_dim: int,
        use_type_embedding: bool = True,
    ):
        """
        Args:
            in_channels_dict: {node_type: input_dim} for each type
            hidden_dim: Unified output dimension
            use_type_embedding: Whether to add type embedding
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_type_embedding = use_type_embedding

        # Type-specific projections
        self.projections = nn.ModuleDict({
            node_type: nn.Linear(in_dim, hidden_dim)
            for node_type, in_dim in in_channels_dict.items()
        })

        # Type embeddings
        if use_type_embedding:
            self.type_embeddings = nn.ParameterDict({
                node_type: nn.Parameter(torch.randn(hidden_dim) * 0.02)
                for node_type in in_channels_dict.keys()
            })

    def forward(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            x_dict: {node_type: (num_nodes, in_dim)} input features

        Returns:
            {node_type: (num_nodes, hidden_dim)} projected features
        """
        out_dict = {}

        for node_type, x in x_dict.items():
            if node_type in self.projections:
                h = self.projections[node_type](x)

                if self.use_type_embedding and node_type in self.type_embeddings:
                    h = h + self.type_embeddings[node_type]

                out_dict[node_type] = h

        return out_dict
