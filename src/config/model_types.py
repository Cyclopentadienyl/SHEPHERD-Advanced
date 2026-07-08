"""
Model-type constants — dependency-free.
=======================================
The supported GNN conv types, kept in a torch-free module so training scripts,
API services, and path helpers can import them without pulling the model stack
(``src/models/gnn`` eagerly imports torch via its package __init__).

``src/models/gnn/layers.py`` (the GNN factory) imports these.

Module: src/config/model_types.py
"""
from __future__ import annotations

# Conv types the GNN factory (src/models/gnn/layers.py) can build. Adding a new
# architecture = implement its branch in HeteroGNNLayer AND add its name here.
SUPPORTED_CONV_TYPES = ("hgt", "gat", "sage")

# Default when a conv type is missing or unrecognised.
DEFAULT_CONV_TYPE = "gat"

__all__ = ["SUPPORTED_CONV_TYPES", "DEFAULT_CONV_TYPE"]
