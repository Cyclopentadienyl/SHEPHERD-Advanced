"""
# ==============================================================================
# Module: src/models/__init__.py
# ==============================================================================
# Purpose: Neural network models for rare disease diagnosis
#
# Dependencies:
#   - External: torch (>=2.9), torch_geometric (>=2.7), pyg-lib
#   - Internal: src.core.types, src.core.schema
#
# Structure:
#   src/models/
#   ├── __init__.py          # This file - unified exports
#   ├── encoders/            # Feature and type encoders
#   │   ├── type_encoder.py
#   │   ├── position_encoder.py
#   │   └── feature_encoder.py
#   ├── gnn/                 # GNN layers and main model
#   │   ├── layers.py
#   │   └── shepherd_gnn.py
#   ├── attention/           # Attention backend selection
#   │   └── adaptive_backend.py
#   ├── bundle/              # Model spec, builder and graph-side modules
#   └── tasks/               # Patient encoder, task heads and task losses
#
# Exports:
#   - ShepherdGNN: Main heterogeneous GNN model
#   - NodeTypeEncoder: Node type embedding module
#   - PositionalEncoder: Structural position encoders (LapPE, RWSE)
#   - OrthologGate: Gated module for cross-species attention
#
# Design Principles:
#   - torch.compile() compatible (no graph breaks)
#   - Dynamic shape support (variable graph sizes)
#   - Mixed precision (bfloat16) ready
#
# Hardware Target:
#   - CUDA 13.0+ with Tensor Core optimization
#   - cuDNN 9.17+ for attention acceleration
#
# Usage:
#   from src.models import ShepherdGNN, ShepherdGNNConfig
#
#   model = ShepherdGNN(
#       metadata=(node_types, edge_types),
#       config=ShepherdGNNConfig(hidden_dim=256, num_layers=4),
#   )
#   model = torch.compile(model, dynamic=True)
# ==============================================================================
"""

# Lazy imports to handle missing torch/torch_geometric
def __getattr__(name):
    """Lazy import to avoid ImportError when torch is not installed"""
    _public_api = {
        # === GNN (Main model) ===
        "ShepherdGNN": ("src.models.gnn", "ShepherdGNN"),
        "ShepherdGNNConfig": ("src.models.gnn", "ShepherdGNNConfig"),
        # === GNN Layers ===
        "HeteroGNNLayer": ("src.models.gnn.layers", "HeteroGNNLayer"),
        "OrthologGate": ("src.models.gnn.layers", "OrthologGate"),
        # === Encoders ===
        "NodeTypeEncoder": ("src.models.encoders", "NodeTypeEncoder"),
        "PositionalEncoder": ("src.models.encoders", "PositionalEncoder"),
        "LaplacianPE": ("src.models.encoders", "LaplacianPE"),
        "RandomWalkSE": ("src.models.encoders", "RandomWalkSE"),
        "HeteroFeatureEncoder": ("src.models.encoders", "HeteroFeatureEncoder"),
        # === Attention ===
        "AdaptiveAttentionBackend": ("src.models.attention", "AdaptiveAttentionBackend"),
    }

    if name in _public_api:
        module_path, attr_name = _public_api[name]
        import importlib
        try:
            module = importlib.import_module(module_path)
            return getattr(module, attr_name)
        except ImportError as e:
            raise ImportError(
                f"Cannot import {name}: torch and torch_geometric are required. "
                f"Install with: pip install torch torch_geometric"
            ) from e

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # === Main model ===
    "ShepherdGNN",
    "ShepherdGNNConfig",
    # === GNN Layers ===
    "HeteroGNNLayer",
    "OrthologGate",
    # === Encoders ===
    "NodeTypeEncoder",
    "PositionalEncoder",
    "LaplacianPE",
    "RandomWalkSE",
    "HeteroFeatureEncoder",
    # === Attention ===
    "AdaptiveAttentionBackend",
]
