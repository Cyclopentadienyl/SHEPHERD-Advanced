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
#   - PhenotypeDiseaseMatcher: Phenotype to disease matching
#   - create_model: Factory function
#
# Design Principles:
#   - Flexible input: works with/without ortholog data
#   - torch.compile() compatible (dynamic=True for variable graphs)
#   - Mixed precision (bfloat16) ready
#   - Interpretable: attention weights and gate values accessible
# ==============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    - Interpretable attention weights
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

        # Store intermediate representations for interpretability
        self._layer_outputs: List[Dict[str, Tensor]] = []
        self._attention_weights: Dict[EdgeType, Tensor] = {}

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        pos_encoding_dict: Optional[Dict[str, Dict[str, Tensor]]] = None,
        return_all_layers: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Forward pass through the heterogeneous GNN.

        Args:
            x_dict: {node_type: (num_nodes, in_dim)} node features
            edge_index_dict: {(src, rel, dst): (2, num_edges)} connectivity
            pos_encoding_dict: Optional {node_type: {"lap_pe": ..., "rwse": ...}}
            return_all_layers: Whether to return intermediate layer outputs

        Returns:
            {node_type: (num_nodes, hidden_dim)} final node embeddings
        """
        self._layer_outputs = []

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

            if return_all_layers:
                self._layer_outputs.append({k: v.clone() for k, v in h_dict.items()})

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

    def get_layer_outputs(self) -> List[Dict[str, Tensor]]:
        """Get intermediate layer outputs (must call forward with return_all_layers=True)"""
        return self._layer_outputs


class PhenotypeDiseaseMatcher(nn.Module):
    """
    Matches patient phenotypes to candidate diseases.

    Given:
    - Patient phenotype embeddings (from ShepherdGNN)
    - Candidate disease embeddings (from ShepherdGNN)

    Computes:
    - Similarity scores between patient profile and diseases
    - Ranked disease predictions
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        similarity_type: str = "bilinear",  # "bilinear", "mlp", "cosine"
    ):
        """
        Args:
            hidden_dim: Embedding dimension
            num_layers: Number of MLP layers for matching
            dropout: Dropout rate
            similarity_type: Type of similarity computation
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.similarity_type = similarity_type

        # Patient profile aggregator
        self.phenotype_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if similarity_type == "bilinear":
            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)
        elif similarity_type == "mlp":
            self.match_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        phenotype_embeddings: Tensor,
        disease_embeddings: Tensor,
        phenotype_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute matching scores.

        Args:
            phenotype_embeddings: (batch, max_phenotypes, hidden_dim) or (num_phenotypes, hidden_dim)
            disease_embeddings: (num_diseases, hidden_dim)
            phenotype_mask: Optional (batch, max_phenotypes) mask for padding

        Returns:
            (batch, num_diseases) or (num_diseases,) similarity scores
        """
        # Handle batched vs unbatched input
        if phenotype_embeddings.dim() == 2:
            # Unbatched: (num_phenotypes, hidden_dim)
            return self._compute_single(phenotype_embeddings, disease_embeddings)
        else:
            # Batched: (batch, max_phenotypes, hidden_dim)
            return self._compute_batched(
                phenotype_embeddings, disease_embeddings, phenotype_mask
            )

    def _compute_single(
        self,
        phenotype_embeddings: Tensor,
        disease_embeddings: Tensor,
    ) -> Tensor:
        """Compute scores for single patient"""
        # Aggregate phenotypes to patient profile
        patient_profile = self.phenotype_aggregator(
            phenotype_embeddings.mean(dim=0, keepdim=True)
        )  # (1, hidden_dim)

        return self._compute_similarity(patient_profile, disease_embeddings).squeeze(0)

    def _compute_batched(
        self,
        phenotype_embeddings: Tensor,
        disease_embeddings: Tensor,
        phenotype_mask: Optional[Tensor],
    ) -> Tensor:
        """Compute scores for batch of patients"""
        batch_size = phenotype_embeddings.size(0)

        # Masked mean for phenotype aggregation
        if phenotype_mask is not None:
            mask = phenotype_mask.unsqueeze(-1).float()
            summed = (phenotype_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            patient_profiles = summed / counts
        else:
            patient_profiles = phenotype_embeddings.mean(dim=1)

        # Aggregate
        patient_profiles = self.phenotype_aggregator(patient_profiles)  # (batch, hidden_dim)

        return self._compute_similarity(patient_profiles, disease_embeddings)

    def _compute_similarity(
        self,
        patient_profiles: Tensor,
        disease_embeddings: Tensor,
    ) -> Tensor:
        """Compute similarity between profiles and diseases"""
        # patient_profiles: (batch, hidden_dim) or (1, hidden_dim)
        # disease_embeddings: (num_diseases, hidden_dim)

        if self.similarity_type == "cosine":
            # Normalize and dot product
            p_norm = F.normalize(patient_profiles, p=2, dim=-1)
            d_norm = F.normalize(disease_embeddings, p=2, dim=-1)
            return torch.mm(p_norm, d_norm.t())

        elif self.similarity_type == "bilinear":
            # Bilinear similarity
            batch_size = patient_profiles.size(0)
            num_diseases = disease_embeddings.size(0)

            # Expand for pairwise computation
            p_exp = patient_profiles.unsqueeze(1).expand(-1, num_diseases, -1)
            d_exp = disease_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

            return self.bilinear(
                p_exp.reshape(-1, self.hidden_dim),
                d_exp.reshape(-1, self.hidden_dim),
            ).view(batch_size, num_diseases)

        else:  # mlp
            batch_size = patient_profiles.size(0)
            num_diseases = disease_embeddings.size(0)

            # Expand and concatenate
            p_exp = patient_profiles.unsqueeze(1).expand(-1, num_diseases, -1)
            d_exp = disease_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            combined = torch.cat([p_exp, d_exp], dim=-1)

            return self.match_mlp(combined).squeeze(-1)


def create_model(
    metadata: Metadata,
    hidden_dim: int = 256,
    num_layers: int = 4,
    in_channels_dict: Optional[Dict[str, int]] = None,
    use_ortholog_gate: bool = True,
    **kwargs,
) -> ShepherdGNN:
    """
    Factory function to create ShepherdGNN model.

    Args:
        metadata: (node_types, edge_types) from HeteroData.metadata()
        hidden_dim: Hidden dimension
        num_layers: Number of GNN layers
        in_channels_dict: {node_type: input_dim} for feature projection
        use_ortholog_gate: Whether to use ortholog gating
        **kwargs: Additional config parameters

    Returns:
        Configured ShepherdGNN model
    """
    config = ShepherdGNNConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_ortholog_gate=use_ortholog_gate,
        **kwargs,
    )

    return ShepherdGNN(
        metadata=metadata,
        in_channels_dict=in_channels_dict,
        config=config,
    )
