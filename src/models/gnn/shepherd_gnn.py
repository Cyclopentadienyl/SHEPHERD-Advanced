"""
# ==============================================================================
# Module: src/models/gnn/shepherd_gnn.py
# ==============================================================================
# Purpose: ShepherdGNN - Main heterogeneous GNN model for rare disease diagnosis
#
# Dependencies:
#   - External: torch (>=2.9), torch_geometric (>=2.7)
#   - Internal: src.models.encoders, src.models.gnn.layers
#
# Exports:
#   - ShepherdGNNConfig: Model configuration dataclass
#   - ShepherdGNN: Main heterogeneous GNN model
#   - build_shepherd_model: Rebuild a trained model from its checkpoint
#
# Design Principles:
#   - Flexible input: works with/without ortholog data
#   - torch.compile() compatible (dynamic=True for variable graphs)
#   - Mixed precision (bfloat16) ready
#   - Interpretable: ortholog gate values accessible
# ==============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.models.encoders import (
    HeteroFeatureEncoder,
    NodeTypeEncoder,
    PositionalEncoder,
)
from src.models.gnn.layers import HeteroGNNLayer, OrthologGate

# Type aliases
NodeType = str
EdgeType = Tuple[str, str, str]
Metadata = Tuple[List[NodeType], List[EdgeType]]


@dataclass
class ShepherdGNNConfig:
    """Configuration for ShepherdGNN model."""

    # Dimensions
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8

    # Convolution settings
    conv_type: str = "gat"  # "hgt", "gat", "sage"
    dropout: float = 0.1

    # Positional encoding
    use_positional_encoding: bool = True
    use_lap_pe: bool = True
    use_rwse: bool = True
    lap_pe_dim: int = 16
    rwse_walk_length: int = 20

    # Ortholog gating
    use_ortholog_gate: bool = True
    ortholog_gate_type: str = "learned"  # "learned", "attention", "fixed"

    # Residual and normalization
    use_residual: bool = True
    use_layer_norm: bool = True

    # Output
    pooling: str = "mean"  # "mean", "max", "attention"


class ShepherdGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for rare disease diagnosis.

    Features:
    - Flexible input handling (works with any subset of node/edge types)
    - Ortholog-aware gating for cross-species reasoning
    - torch.compile() compatible

    The model processes a heterogeneous knowledge graph containing:
    - Phenotypes (HPO terms)
    - Genes
    - Diseases (MONDO terms)
    - Optionally: Mouse genes, Mouse phenotypes, Pathways, etc.
    """

    def __init__(
        self,
        metadata: Metadata,
        in_channels_dict: Optional[Dict[str, int]] = None,
        config: Optional[ShepherdGNNConfig] = None,
    ):
        """
        Args:
            metadata: (node_types, edge_types) from HeteroData.metadata()
            in_channels_dict: {node_type: input_dim} for feature projection
                             If None, assumes all features are already hidden_dim
            config: Model configuration
        """
        super().__init__()

        self.config = config or ShepherdGNNConfig()
        self.metadata = metadata
        self.node_types = metadata[0]
        self.edge_types = metadata[1]

        hidden_dim = self.config.hidden_dim

        # === Feature Encoder ===
        if in_channels_dict is not None:
            self.feature_encoder = HeteroFeatureEncoder(
                in_channels_dict=in_channels_dict,
                hidden_dim=hidden_dim,
                use_type_embedding=True,
            )
        else:
            self.feature_encoder = None

        # === Type Embeddings (for nodes without features) ===
        self.node_type_encoder = NodeTypeEncoder(
            num_types=len(self.node_types),
            hidden_dim=hidden_dim,
        )
        self._node_type_to_idx = {nt: i for i, nt in enumerate(self.node_types)}

        # === Positional Encoder ===
        if self.config.use_positional_encoding:
            self.pos_encoder = PositionalEncoder(
                hidden_dim=hidden_dim,
                use_lap_pe=self.config.use_lap_pe,
                use_rwse=self.config.use_rwse,
                lap_pe_dim=self.config.lap_pe_dim,
                rwse_walk_length=self.config.rwse_walk_length,
            )
        else:
            self.pos_encoder = None

        # === GNN Layers ===
        self.gnn_layers = nn.ModuleList([
            HeteroGNNLayer(
                hidden_dim=hidden_dim,
                metadata=metadata,
                conv_type=self.config.conv_type,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
                use_residual=self.config.use_residual,
                use_layer_norm=self.config.use_layer_norm,
            )
            for _ in range(self.config.num_layers)
        ])

        # === Ortholog Gate ===
        if self.config.use_ortholog_gate:
            self.ortholog_gate = OrthologGate(
                hidden_dim=hidden_dim,
                gate_type=self.config.ortholog_gate_type,
            )
        else:
            self.ortholog_gate = None

        # === Final Layer Norm ===
        self.final_norms = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim)
            for node_type in self.node_types
        })

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        pos_encoding_dict: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass through the heterogeneous GNN.

        Args:
            x_dict: {node_type: (num_nodes, in_dim)} node features
            edge_index_dict: {(src, rel, dst): (2, num_edges)} connectivity
            pos_encoding_dict: Optional {node_type: {"lap_pe": ..., "rwse": ...}}

        Returns:
            {node_type: (num_nodes, hidden_dim)} final node embeddings
        """
        # === 1. Feature Encoding ===
        if self.feature_encoder is not None:
            h_dict = self.feature_encoder(x_dict)
        else:
            h_dict = x_dict

        # Add type embeddings for missing node types
        h_dict = self._add_type_embeddings(h_dict, x_dict)

        # === 2. Positional Encoding ===
        if self.pos_encoder is not None and pos_encoding_dict is not None:
            h_dict = self._add_positional_encoding(h_dict, pos_encoding_dict)

        # === 3. GNN Message Passing ===
        # Separate core and ortholog edges
        core_edges, ortholog_edges = self._split_edge_types(edge_index_dict)

        for i, layer in enumerate(self.gnn_layers):
            # Core message passing (always happens)
            h_dict = layer(h_dict, core_edges)

            # Ortholog message passing (if data available and gate enabled)
            if self.ortholog_gate is not None and len(ortholog_edges) > 0:
                # Get ortholog-influenced representations
                h_ortholog = layer(h_dict, ortholog_edges)

                # Gate combination
                h_dict = self.ortholog_gate(h_dict, h_ortholog)

        # === 4. Final Normalization ===
        out_dict = {}
        for node_type, h in h_dict.items():
            if node_type in self.final_norms:
                out_dict[node_type] = self.final_norms[node_type](h)
            else:
                out_dict[node_type] = h

        return out_dict

    def _add_type_embeddings(
        self,
        h_dict: Dict[str, Tensor],
        x_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Add type embeddings for node types not in h_dict"""
        out_dict = dict(h_dict)

        for node_type in self.node_types:
            if node_type not in out_dict and node_type in x_dict:
                # Use type embedding as features
                num_nodes = x_dict[node_type].size(0)
                type_idx = self._node_type_to_idx[node_type]
                type_indices = torch.full(
                    (num_nodes,), type_idx,
                    device=x_dict[node_type].device,
                    dtype=torch.long,
                )
                out_dict[node_type] = self.node_type_encoder(type_indices)

        return out_dict

    def _add_positional_encoding(
        self,
        h_dict: Dict[str, Tensor],
        pos_encoding_dict: Dict[str, Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Add positional encodings to node features"""
        out_dict = {}

        for node_type, h in h_dict.items():
            if node_type in pos_encoding_dict:
                pe_data = pos_encoding_dict[node_type]
                pos_enc = self.pos_encoder(
                    lap_pe=pe_data.get("lap_pe"),
                    rwse=pe_data.get("rwse"),
                    degree=pe_data.get("degree"),
                )
                out_dict[node_type] = h + pos_enc
            else:
                out_dict[node_type] = h

        return out_dict

    def _split_edge_types(
        self,
        edge_index_dict: Dict[EdgeType, Tensor],
    ) -> Tuple[Dict[EdgeType, Tensor], Dict[EdgeType, Tensor]]:
        """Split edges into core and ortholog categories"""
        core_edges = {}
        ortholog_edges = {}

        for edge_type, edge_index in edge_index_dict.items():
            if OrthologGate.is_ortholog_edge(edge_type):
                ortholog_edges[edge_type] = edge_index
            else:
                core_edges[edge_type] = edge_index

        return core_edges, ortholog_edges

    def get_ortholog_contribution(self) -> Optional[Dict[str, float]]:
        """Get ortholog contribution scores (for interpretability)"""
        if self.ortholog_gate is not None:
            return self.ortholog_gate.get_ortholog_contribution()
        return None


# ==============================================================================
# Construction from a training checkpoint — production semantics, one copy
# ==============================================================================
def build_shepherd_model(
    checkpoint: Dict[str, Any],
    graph_data: Dict[str, Any],
    device: Optional[Any] = None,
) -> ShepherdGNN:
    """Rebuild a trained model the way the deployed pipeline does.

    **One implementation, two callers.** `src/inference/pipeline.py` uses this to
    load the served model, and `src/evaluation` uses it so measurement's Mode C
    encodes with production semantics. Written twice, the copies would drift, and
    a measurement that quietly stopped describing the deployed scorer is the one
    failure this work item cannot detect from its own output.

    **Metadata comes from `graph_data`, never from the knowledge-graph object.**
    `scripts/train_model.py` builds the model from `graph_data` keys, which include
    the `rev_*` reverse edges added for bidirectional message passing; the
    knowledge graph's own edges are only the forward ones. Using those produces a model
    with fewer conv layers than the checkpoint expects, and `load_state_dict`
    fails — which is the good case, because the alternative is a silently
    different model.

    Architecture is recovered by the documented precedence in
    `src.config.model_types.resolve_arch_params` — self-describing `model_config`
    first, then legacy flat fields, then inference from the weight names — so an
    HGT or SAGE checkpoint is never rebuilt as the GAT default.

    Args:
        checkpoint: a loaded checkpoint dict. Either trainer format
            (``model_state_dict``) or callback format (``state_dict``).
        graph_data: ``x_dict`` and ``edge_index_dict`` of the graph the model will
            run on. Its keys define the architecture, so this is not optional.
        device: where to place the model. ``None`` leaves it where it was built.

    Returns:
        The model in eval mode, weights loaded.

    Raises:
        KeyError: no recognised state dict in the checkpoint.
        ValueError: `graph_data` lacks the dicts the architecture is derived from.
        RuntimeError: from `load_state_dict`, re-raised with the metadata detail
            an operator needs to tell a compatibility problem from a bug.
    """
    import dataclasses as _dc

    from src.config.model_types import SUPPORTED_CONV_TYPES, resolve_arch_params

    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError(
            "checkpoint has neither 'model_state_dict' (trainer format) nor "
            f"'state_dict' (callback format); keys are {sorted(checkpoint)[:10]}"
        )

    x_dict = graph_data.get("x_dict")
    edge_index_dict = graph_data.get("edge_index_dict")
    if not x_dict or not edge_index_dict:
        raise ValueError(
            "graph_data must carry both 'x_dict' and 'edge_index_dict'; the model's "
            "architecture is derived from their keys, so an absent one would "
            "silently build a different model"
        )

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
    in_channels_dict = {
        node_type: features.size(-1)
        for node_type, features in x_dict.items()
        if features.dim() >= 2
    }

    state_keys = set(state_dict.keys())
    arch_params = resolve_arch_params(
        checkpoint.get("config", {}) or {},
        state_keys,
        valid_fields={f.name for f in _dc.fields(ShepherdGNNConfig)},
        supported_conv=SUPPORTED_CONV_TYPES,
        has_pos_encoder=any(k.startswith("pos_encoder.") for k in state_keys),
        has_ortholog_gate=any(k.startswith("ortholog_gate.") for k in state_keys),
    )
    model_config = ShepherdGNNConfig(**arch_params)

    # A node type with no usable feature tensor still needs an input width.
    for node_type in metadata[0]:
        in_channels_dict.setdefault(node_type, model_config.hidden_dim)

    model = ShepherdGNN(
        metadata=metadata,
        in_channels_dict=in_channels_dict,
        config=model_config,
    )

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}\nModel expects {len(metadata[1])} edge types: "
            f"{[f'{s}--{r}--{t}' for s, r, t in metadata[1]][:5]}...\n"
            f"Checkpoint has {len(state_dict)} parameter tensors. Common cause: "
            "trainer and inference derived metadata from different sources — the "
            "trainer uses graph_data keys, and so must this."
        ) from exc

    model.eval()
    if device is not None:
        model = model.to(device)
    return model
