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
#   - OrthologGate: Gated cross-species information integration
#
# Design Notes:
#   - All layers are torch.compile() compatible
#   - OrthologGate preserves interpretability (get_ortholog_contribution)
#
# References:
#   - HGT (WWW'20): Heterogeneous Graph Transformer
#   - PyG HeteroConv: Native heterogeneous support
# ==============================================================================
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

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

        # Build convolution dictionary
        conv_dict = {}
        for edge_type in self.edge_types:
            src_type, rel_type, dst_type = edge_type

            if conv_type == "hgt":
                # HGT requires specific setup
                conv_dict[edge_type] = self._create_hgt_conv(
                    hidden_dim, num_heads
                )
            elif conv_type == "gat":
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
                raise ValueError(f"Unknown conv_type: {conv_type}")

        self.conv = HeteroConv(conv_dict, aggr="sum")

        # Layer normalization per node type
        if use_layer_norm:
            self.layer_norms = nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in self.node_types
            })

        self.dropout = nn.Dropout(dropout)

    def _create_hgt_conv(self, hidden_dim: int, num_heads: int):
        """Create a simplified HGT-style attention layer"""
        # Using GAT as HGT approximation for simplicity
        # Full HGT requires metadata at init time
        return GATConv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            add_self_loops=False,
            concat=True,
        )

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
        # HeteroConv automatically skips missing edge types
        out_dict = self.conv(x_dict, edge_index_dict)

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


class OrthologGate(nn.Module):
    """
    Gated module for ortholog (cross-species) information integration.

    Features:
    - Learns when to trust ortholog evidence
    - Provides interpretable contribution scores
    - Graceful degradation when ortholog data is missing

    The gate output can be inspected for explainability:
    "This diagnosis is X% influenced by mouse ortholog evidence"

    Preserved interfaces for cross-species reasoning:
    - ORTHOLOG_EDGE_TYPES: Set of edge types considered ortholog-related
    - is_ortholog_edge(): Static method to check edge types
    - get_ortholog_contribution(): Get interpretable gate values
    """

    # Edge types considered as ortholog edges
    # These are preserved for PubMed data compatibility and cross-species reasoning
    ORTHOLOG_EDGE_TYPES = {
        # Human-Mouse orthologs
        ("gene", "ortholog_of", "mouse_gene"),
        ("gene", "human_mouse_ortholog", "mouse_gene"),
        ("mouse_gene", "ortholog_of", "gene"),
        ("mouse_gene", "mouse_human_ortholog", "gene"),
        # Human-Zebrafish orthologs
        ("gene", "human_zebrafish_ortholog", "zebrafish_gene"),
        ("zebrafish_gene", "zebrafish_human_ortholog", "gene"),
        # Mouse phenotype associations (for cross-species inference)
        ("mouse_gene", "has_phenotype", "mouse_phenotype"),
        ("mouse_phenotype", "associated_with", "mouse_gene"),
    }

    def __init__(
        self,
        hidden_dim: int,
        num_species: int = 3,  # human, mouse, zebrafish
        gate_type: str = "learned",  # "learned", "fixed", "attention"
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_species: Number of species for species embedding
            gate_type: Type of gating mechanism
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_type = gate_type

        # Species embedding (for species-aware gating)
        self.species_embedding = nn.Embedding(num_species, hidden_dim)

        if gate_type == "learned":
            # Simple learned gate based on node features
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
        elif gate_type == "attention":
            # Attention-based gating
            self.query = nn.Linear(hidden_dim, hidden_dim)
            self.key = nn.Linear(hidden_dim, hidden_dim)
            self.gate_proj = nn.Linear(hidden_dim, 1)

        # Store last gate values for interpretability
        self._last_gate_values: Optional[Dict[str, Tensor]] = None

    def forward(
        self,
        h_core: Dict[str, Tensor],
        h_ortholog: Optional[Dict[str, Tensor]] = None,
        species_indices: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        Combine core and ortholog representations with gating.

        Args:
            h_core: Core pathway representations (always present)
            h_ortholog: Ortholog pathway representations (may be None)
            species_indices: Species index per node type

        Returns:
            Gated combination of core and ortholog features
        """
        # If no ortholog information, return core directly
        if h_ortholog is None or len(h_ortholog) == 0:
            self._last_gate_values = None
            return h_core

        out_dict = {}
        gate_values = {}

        for node_type, h_c in h_core.items():
            if node_type not in h_ortholog:
                # No ortholog info for this type
                out_dict[node_type] = h_c
                continue

            h_o = h_ortholog[node_type]

            # Compute gate value
            if self.gate_type == "learned":
                # Gate based on core features
                gate = self.gate_net(h_c)  # (num_nodes, 1)

            elif self.gate_type == "attention":
                # Cross-attention gating
                q = self.query(h_c)
                k = self.key(h_o)
                attn = torch.sum(q * k, dim=-1, keepdim=True) / math.sqrt(self.hidden_dim)
                gate = torch.sigmoid(self.gate_proj(torch.tanh(attn)))

            else:  # fixed
                gate = torch.full((h_c.size(0), 1), 0.5, device=h_c.device)

            # Apply species-specific modulation if available
            if species_indices is not None and node_type in species_indices:
                species_emb = self.species_embedding(species_indices[node_type])
                species_mod = torch.sigmoid(
                    torch.sum(h_c * species_emb, dim=-1, keepdim=True)
                )
                gate = gate * species_mod

            # Gated combination
            out_dict[node_type] = (1 - gate) * h_c + gate * h_o

            # Store for interpretability
            gate_values[node_type] = gate.detach()

        self._last_gate_values = gate_values
        return out_dict

    def get_ortholog_contribution(self) -> Optional[Dict[str, float]]:
        """
        Get average ortholog contribution per node type.

        Returns:
            {node_type: avg_contribution} or None if no ortholog data
        """
        if self._last_gate_values is None:
            return None

        return {
            node_type: gate.mean().item()
            for node_type, gate in self._last_gate_values.items()
        }

    @staticmethod
    def is_ortholog_edge(edge_type: EdgeType) -> bool:
        """
        Check if an edge type is ortholog-related.

        Args:
            edge_type: (src_type, rel_type, dst_type) tuple

        Returns:
            True if the edge type is related to ortholog/cross-species data
        """
        # Check by edge type tuple
        if edge_type in OrthologGate.ORTHOLOG_EDGE_TYPES:
            return True

        # Check by relation name
        _, rel, _ = edge_type
        ortholog_keywords = ["ortholog", "mouse", "zebrafish", "cross_species"]
        return any(kw in rel.lower() for kw in ortholog_keywords)
