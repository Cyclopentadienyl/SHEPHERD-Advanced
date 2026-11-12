"""
# ==============================================================================
# Module: src/models/gnn/__init__.py
# ==============================================================================
# Purpose: the heterogeneous GNN layer, kept and reached only by tests: only
#   gatv2 is declared and built (src.models.bundle.spec.ENCODER_DECLARATIONS),
#   and a second architecture is chosen after the ops-wiring round's fit
#   assessment (docs/working/ops-wiring-and-ui-ux/README.md §10)
#
# Exports:
#   - HeteroGNNLayer: Heterogeneous GNN layer
# ==============================================================================
"""
from src.models.gnn.layers import HeteroGNNLayer

__all__ = [
    "HeteroGNNLayer",
]
