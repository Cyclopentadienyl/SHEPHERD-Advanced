"""
# ==============================================================================
# Module: src/models/gnn/__init__.py
# ==============================================================================
# Purpose: Graph Neural Network components for heterogeneous medical KGs
#
# Exports:
#   - ShepherdGNN: Main heterogeneous GNN model
#   - ShepherdGNNConfig: Model configuration
#   - HeteroGNNLayer: Heterogeneous GNN layer
#   - OrthologGate: Cross-species gating module
# ==============================================================================
"""
from src.models.gnn.shepherd_gnn import (
    ShepherdGNN,
    ShepherdGNNConfig,
)
from src.models.gnn.layers import HeteroGNNLayer, OrthologGate

__all__ = [
    # Main model
    "ShepherdGNN",
    "ShepherdGNNConfig",
    # Layers
    "HeteroGNNLayer",
    "OrthologGate",
]
