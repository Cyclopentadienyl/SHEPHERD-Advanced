"""
# ==============================================================================
# Module: src/models/gnn/layers.py
# ==============================================================================
# Purpose: GNN layers for heterogeneous medical knowledge graphs
#
# Dependencies:
#   - External: torch (>=2.9), torch_geometric (>=2.7)
#   - Internal: None
#
# Exports:
#   - HeteroGNNLayer: Unified heterogeneous GNN layer
#
# Design Notes:
#   - All layers are torch.compile() compatible
#
# References:
#   - HGT (WWW'20): Heterogeneous Graph Transformer
#   - PyG HeteroConv: Native heterogeneous support
# ==============================================================================
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# Try to import PyG components
try:
    from torch_geometric.nn import HeteroConv, Linear, HGTConv, GATConv, SAGEConv
    from torch_geometric.typing import EdgeType, NodeType
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    EdgeType = Tuple[str, str, str]
    NodeType = str

logger = logging.getLogger(__name__)

# Single source of truth for the conv types the factory below can build lives in
# a dependency-free module (so training scripts / API services / path helpers can
# import it without pulling torch — src/models/gnn eagerly imports torch).
# Adding a new architecture = implement its branch here AND add its name there.
from src.config.model_types import SUPPORTED_CONV_TYPES  # noqa: E402,F401


class HeteroGNNLayer(nn.Module):
    """
    Unified heterogeneous GNN layer supporting multiple convolution types.

    Wraps PyG's HeteroConv with:
    - Automatic handling of missing edge types
    - Residual connections
    - Layer normalization
    - Dropout

    torch.compile() compatible.
    """

    def __init__(
        self,
        hidden_dim: int,
        metadata: Tuple[List[str], List[EdgeType]],
        conv_type: str = "hgt",  # "hgt", "gat", "sage"
        num_heads: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True,
    ):
        """
        Args:
            hidden_dim: Hidden dimension (must be divisible by num_heads for HGT/GAT)
            metadata: (node_types, edge_types) from HeteroData.metadata()
            conv_type: Type of convolution to use
            num_heads: Number of attention heads (for HGT/GAT)
            dropout: Dropout rate
            use_residual: Whether to use residual connections
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        if not HAS_PYG:
            raise ImportError("torch_geometric required for HeteroGNNLayer")

        self.hidden_dim = hidden_dim
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.conv_type = conv_type
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        # Build convolution
        if conv_type == "hgt":
            # HGTConv's internal segment_matmul comes from pyg_lib. Without the
            # native extension the layer still builds, but that op drops to a
            # much slower fallback — and HGT is the most pyg_lib-sensitive conv
            # type we offer, so warn loudly rather than let it run silently slow.
            try:
                from src.utils.pyg_native_check import check_pyg_native_extensions

                if not check_pyg_native_extensions()["pyg_lib"].available:
                    logger.warning(
                        "HGTConv selected but pyg_lib is not importable: "
                        "segment_matmul will use a slow fallback. Install the PyG "
                        "native extensions (see scripts/validate_pyg_ext.py) for "
                        "usable HGT performance."
                    )
            except Exception as exc:  # never block model construction
                logger.debug("pyg_lib availability check skipped: %s", exc)

            # True HGT: type-aware heterogeneous graph transformer
            self.conv = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=num_heads,
            )
            # **The conv's input is passed in the conv's own node-type order**
            # (BACKLOG item 20 (a); docs/working/PLAN_HGT_LINEAR_PATHS.md §1).
            # PyG's grouped branch of `HeteroDictLinear` builds its inputs in its
            # layers' order and stores the outputs under the input dict's keys,
            # so a caller whose order differs gets one node type's projection
            # under another's name, silently more often than not. Orders do
            # differ: the ShepherdGNN training loader put its seed types first
            # until K8-7, and from the second layer on the types that received
            # no messages come last.
            # Both linears share the order the conv was built in; checked here
            # so a PyG change that split them fails at construction instead.
            self._hgt_node_order: Tuple[str, ...] = tuple(self.conv.kqv_lin.lins)
            if self._hgt_node_order != tuple(self.conv.out_lin.lins):
                raise RuntimeError(
                    "HGTConv's input and output linears are built in different "
                    f"node-type orders ({list(self._hgt_node_order)} vs "
                    f"{list(self.conv.out_lin.lins)}); the input could not be "
                    "aligned with both"
                )
            # The edges likewise go in the conv's edge-type order (§1.7 of the
            # plan): its relation linears are built `is_sorted=True` and cut
            # their input by edge-type index in the order the edges arrive, so
            # another order gives wrong outputs silently on both kernel paths.
            # No formal caller builds this layer, which tests alone reach: only
            # gatv2 is declared and built (`spec.ENCODER_DECLARATIONS`,
            # `builder._ENCODERS`), and a second architecture is chosen after the
            # ops-wiring round's fit assessment
            # (docs/working/ops-wiring-and-ui-ux/README.md §10).
            self._hgt_edge_order: Tuple[EdgeType, ...] = tuple(self.conv.edge_types_map)
        else:
            # GAT or SAGE: per-edge-type convolutions wrapped in HeteroConv
            conv_dict = {}
            for edge_type in self.edge_types:
                if conv_type == "gat":
                    conv_dict[edge_type] = GATConv(
                        hidden_dim,
                        hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        add_self_loops=False,
                        concat=True,
                    )
                elif conv_type == "sage":
                    conv_dict[edge_type] = SAGEConv(
                        hidden_dim,
                        hidden_dim,
                    )
                else:
                    raise ValueError(
                        f"Unknown conv_type: {conv_type!r} "
                        f"(supported: {SUPPORTED_CONV_TYPES})"
                    )

            self.conv = HeteroConv(conv_dict, aggr="sum")

        # Layer normalization per node type
        if use_layer_norm:
            self.layer_norms = nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in self.node_types
            })

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Args:
            x_dict: {node_type: (num_nodes, hidden_dim)} input features
            edge_index_dict: {edge_type: (2, num_edges)} connectivity

        Returns:
            Updated x_dict
        """
        # Store residual
        residual = x_dict if self.use_residual else None

        # Apply heterogeneous convolution
        # HGTConv's internal segment_matmul (pyg_lib) only supports float32.
        # Disable AMP autocast and cast inputs to float32 for HGT.
        # This increases VRAM usage — use smaller batch sizes with HGT.
        if self.conv_type == "hgt":
            with torch.amp.autocast(device_type="cuda", enabled=False):
                # In the conv's node-type order (see __init__). A type the conv
                # does not know is left out, as its per-type path ignores it, and
                # is carried through by the loop below.
                x_float = {
                    nt: x_dict[nt].float() for nt in self._hgt_node_order if nt in x_dict
                }
                # In the conv's edge-type order (see __init__). An edge type the
                # conv does not know stays in, last, so the conv still refuses it.
                edges = {
                    et: edge_index_dict[et]
                    for et in self._hgt_edge_order if et in edge_index_dict
                }
                edges.update(edge_index_dict)
                out_dict = self.conv(x_float, edges)
        else:
            out_dict = self.conv(x_dict, edge_index_dict)

        # Preserve features for node types that didn't receive messages
        # (they won't appear in out_dict from HeteroConv)
        for node_type in x_dict:
            if node_type not in out_dict:
                out_dict[node_type] = x_dict[node_type]

        # Post-processing per node type
        for node_type in out_dict:
            h = out_dict[node_type]

            # Residual connection
            if residual is not None and node_type in residual:
                h = h + residual[node_type]

            # Layer norm
            if self.use_layer_norm and node_type in self.layer_norms:
                h = self.layer_norms[node_type](h)

            # Dropout
            h = self.dropout(h)

            out_dict[node_type] = h

        return out_dict
