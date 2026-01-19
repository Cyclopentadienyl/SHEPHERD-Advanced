"""
# ==============================================================================
# Module: src/models/gnn/__init__.py
# ==============================================================================
# Purpose: Graph Neural Network components for heterogeneous medical KGs
#
# Exports:
#   - ShepherdGNN: Main heterogeneous GNN model
#   - ShepherdGNNConfig: Model configuration
#   - PhenotypeDiseaseMatcher: Phenotype-to-disease matching
#   - create_model: Factory function
#   - HeteroGNNLayer: Heterogeneous GNN layer
#   - OrthologGate: Cross-species gating module
# ==============================================================================
"""
from src.models.gnn.shepherd_gnn import (
    ShepherdGNN,
    ShepherdGNNConfig,
    PhenotypeDiseaseMatcher,
    create_model,
)
from src.models.gnn.layers import HeteroGNNLayer, OrthologGate

__all__ = [
    # Main model
    "ShepherdGNN",
    "ShepherdGNNConfig",
    "PhenotypeDiseaseMatcher",
    "create_model",
    # Layers
    "HeteroGNNLayer",
    "OrthologGate",
]
