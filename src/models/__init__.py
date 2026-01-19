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
# Exports:
#   - ShepherdGNN: Main heterogeneous GNN model
#   - NodeTypeEncoder: Node type embedding module
#   - PositionalEncoder: Structural position encoders (LapPE, RWSE)
#   - DiagnosisHead: Prediction head for disease ranking
#   - OrthologGate: Gated module for cross-species attention
#
# Design Principles:
#   - torch.compile() compatible (no graph breaks)
#   - Dynamic shape support (variable graph sizes)
#   - FlexAttention for custom attention patterns
#   - Mixed precision (bfloat16) ready
#
# Hardware Target:
#   - CUDA 13.0+ with Tensor Core optimization
#   - cuDNN 9.17+ for attention acceleration
#
# Usage:
#   from src.models import ShepherdGNN, create_model
#
#   model = create_model(
#       metadata=kg.metadata(),
#       hidden_dim=256,
#       num_layers=4,
#       use_ortholog_gate=True,
#   )
#   model = torch.compile(model, dynamic=True)
# ==============================================================================
"""

# Lazy imports to handle missing torch/torch_geometric
def __getattr__(name):
    """Lazy import to avoid ImportError when torch is not installed"""
    _public_api = {
        # Main model
        "ShepherdGNN": ("src.models.gnn", "ShepherdGNN"),
        "create_model": ("src.models.gnn", "create_model"),
        # Encoders
        "NodeTypeEncoder": ("src.models.encoders", "NodeTypeEncoder"),
        "EdgeTypeEncoder": ("src.models.encoders", "EdgeTypeEncoder"),
        "PositionalEncoder": ("src.models.encoders", "PositionalEncoder"),
        "LaplacianPE": ("src.models.encoders", "LaplacianPE"),
        "RandomWalkSE": ("src.models.encoders", "RandomWalkSE"),
        # Layers
        "HeteroGNNLayer": ("src.models.layers", "HeteroGNNLayer"),
        "OrthologGate": ("src.models.layers", "OrthologGate"),
        "FlexHeteroAttention": ("src.models.layers", "FlexHeteroAttention"),
        # Heads
        "DiagnosisHead": ("src.models.heads", "DiagnosisHead"),
        "LinkPredictionHead": ("src.models.heads", "LinkPredictionHead"),
        "NodeClassificationHead": ("src.models.heads", "NodeClassificationHead"),
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
    # Main model
    "ShepherdGNN",
    "create_model",
    # Encoders
    "NodeTypeEncoder",
    "EdgeTypeEncoder",
    "PositionalEncoder",
    "LaplacianPE",
    "RandomWalkSE",
    # Layers
    "HeteroGNNLayer",
    "OrthologGate",
    "FlexHeteroAttention",
    # Heads
    "DiagnosisHead",
    "LinkPredictionHead",
    "NodeClassificationHead",
]
