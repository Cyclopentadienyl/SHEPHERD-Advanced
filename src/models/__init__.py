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
#   ├── gnn/                 # The heterogeneous layer kept for the port
#   │   └── layers.py
#   ├── attention/           # Attention backend selection
#   │   └── adaptive_backend.py
#   ├── bundle/              # Model spec, builder and graph-side modules
#   └── tasks/               # Patient encoder, task heads and task losses
#
# Exports:
#   - HeteroGNNLayer: the heterogeneous GNN layer
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
# Usage: a model is built from its spec by src.models.bundle.builder.build_bundle.
# ==============================================================================
"""

# Lazy imports to handle missing torch/torch_geometric
def __getattr__(name):
    """Lazy import to avoid ImportError when torch is not installed"""
    _public_api = {
        # === GNN Layers ===
        "HeteroGNNLayer": ("src.models.gnn.layers", "HeteroGNNLayer"),
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
    # === GNN Layers ===
    "HeteroGNNLayer",
    # === Attention ===
    "AdaptiveAttentionBackend",
]
