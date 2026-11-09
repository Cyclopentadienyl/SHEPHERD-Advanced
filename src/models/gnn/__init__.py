"""
# ==============================================================================
# Module: src/models/gnn/__init__.py
# ==============================================================================
# Purpose: the heterogeneous GNN layer kept for the multi-architecture port
#
# Exports:
#   - HeteroGNNLayer: Heterogeneous GNN layer
# ==============================================================================
"""
from src.models.gnn.layers import HeteroGNNLayer

__all__ = [
    "HeteroGNNLayer",
]
