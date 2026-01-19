"""
# ==============================================================================
# Module: src/models/encoders.py
# ==============================================================================
# Purpose: Feature encoders for heterogeneous graph nodes and edges
#
# Dependencies:
#   - External: torch (>=2.9), torch_geometric (>=2.7), scipy
#   - Internal: None
#
# Input:
#   - Node features: Raw features or one-hot type indices
#   - Graph structure: edge_index for positional encoding computation
#
# Output:
#   - Encoded node representations: (num_nodes, hidden_dim)
#   - Type embeddings: (num_nodes, hidden_dim)
#   - Positional encodings: (num_nodes, pe_dim)
#
# Design Notes:
#   - All encoders are torch.compile() compatible
#   - Positional encodings computed once, cached for reuse
#   - Supports both dense and sparse computation modes
#
# References:
#   - GraphGPS (NeurIPS'22): Laplacian PE, Random Walk SE
#   - HGT (WWW'20): Type-specific projection
# ==============================================================================
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ==============================================================================
# Node/Edge Type Encoders
# ==============================================================================
class NodeTypeEncoder(nn.Module):
    """
    Encodes node types into learnable embeddings.

    Supports both:
    1. Type-only encoding (no input features)
    2. Type + feature fusion (concatenate or add)

    torch.compile() compatible.
    """

    def __init__(
        self,
        num_types: int,
        hidden_dim: int,
        fusion_mode: str = "add",  # "add", "concat", "gate"
    ):
        """
        Args:
            num_types: Number of distinct node types
            hidden_dim: Output embedding dimension
            fusion_mode: How to combine type embedding with features
        """
        super().__init__()
        self.num_types = num_types
        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode

        # Learnable type embeddings
        self.type_embedding = nn.Embedding(num_types, hidden_dim)

        # For gated fusion
        if fusion_mode == "gate":
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with small values"""
        nn.init.normal_(self.type_embedding.weight, std=0.02)

    def forward(
        self,
        type_indices: Tensor,
        features: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            type_indices: (num_nodes,) node type indices
            features: Optional (num_nodes, hidden_dim) input features

        Returns:
            (num_nodes, hidden_dim) encoded representations
        """
        type_emb = self.type_embedding(type_indices)

        if features is None:
            return type_emb

        if self.fusion_mode == "add":
            return features + type_emb
        elif self.fusion_mode == "concat":
            # Caller should handle dimension change
            return torch.cat([features, type_emb], dim=-1)
        elif self.fusion_mode == "gate":
            combined = torch.cat([features, type_emb], dim=-1)
            gate_value = self.gate(combined)
            return gate_value * features + (1 - gate_value) * type_emb
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")


class EdgeTypeEncoder(nn.Module):
    """
    Encodes edge types into learnable embeddings.

    Used for:
    1. Edge-type-specific message passing weights
    2. Attention score modification based on relation type

    torch.compile() compatible.
    """

    def __init__(
        self,
        num_types: int,
        hidden_dim: int,
    ):
        """
        Args:
            num_types: Number of distinct edge/relation types
            hidden_dim: Output embedding dimension
        """
        super().__init__()
        self.num_types = num_types
        self.hidden_dim = hidden_dim

        self.type_embedding = nn.Embedding(num_types, hidden_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.type_embedding.weight, std=0.02)

    def forward(self, type_indices: Tensor) -> Tensor:
        """
        Args:
            type_indices: (num_edges,) edge type indices

        Returns:
            (num_edges, hidden_dim) edge type embeddings
        """
        return self.type_embedding(type_indices)


# ==============================================================================
# Positional Encoders (GraphGPS-style)
# ==============================================================================
class LaplacianPE(nn.Module):
    """
    Laplacian Positional Encoding (LapPE).

    Computes eigenvectors of the graph Laplacian as positional features.
    Sign-invariant through learned sign flip or absolute value.

    Reference: GraphGPS (NeurIPS'22)

    Note: Eigendecomposition is computed once during preprocessing,
    not during forward pass (for torch.compile compatibility).
    """

    def __init__(
        self,
        pe_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        sign_inv_method: str = "abs",  # "abs", "learned"
    ):
        """
        Args:
            pe_dim: Dimension of raw positional encoding (num eigenvectors)
            hidden_dim: Output dimension after MLP projection
            num_layers: Number of MLP layers
            sign_inv_method: Method for sign invariance
        """
        super().__init__()
        self.pe_dim = pe_dim
        self.hidden_dim = hidden_dim
        self.sign_inv_method = sign_inv_method

        # MLP to project PE to hidden dimension
        layers = []
        in_dim = pe_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))

        self.mlp = nn.Sequential(*layers)

        # For learned sign invariance
        if sign_inv_method == "learned":
            self.sign_mlp = nn.Sequential(
                nn.Linear(pe_dim, pe_dim),
                nn.Tanh(),
            )

    def forward(self, pe: Tensor) -> Tensor:
        """
        Args:
            pe: (num_nodes, pe_dim) precomputed Laplacian eigenvectors

        Returns:
            (num_nodes, hidden_dim) positional encoding
        """
        if self.sign_inv_method == "abs":
            pe = pe.abs()
        elif self.sign_inv_method == "learned":
            pe = pe * self.sign_mlp(pe)

        return self.mlp(pe)

    @staticmethod
    def compute_pe(
        edge_index: Tensor,
        num_nodes: int,
        pe_dim: int,
        padding: bool = True,
    ) -> Tensor:
        """
        Compute Laplacian eigenvectors (preprocessing step).

        Args:
            edge_index: (2, num_edges) graph connectivity
            num_nodes: Number of nodes
            pe_dim: Number of eigenvectors to compute
            padding: Pad with zeros if graph is too small

        Returns:
            (num_nodes, pe_dim) eigenvectors
        """
        try:
            from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
            import scipy.sparse.linalg as sla
            import numpy as np
        except ImportError:
            raise ImportError("scipy required for Laplacian PE computation")

        # Build normalized Laplacian
        edge_index_lap, edge_weight = get_laplacian(
            edge_index,
            normalization="sym",
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index_lap, edge_weight, num_nodes)

        # Compute smallest eigenvectors (skip first trivial one)
        k = min(pe_dim + 1, num_nodes)
        try:
            eigenvalues, eigenvectors = sla.eigsh(
                L, k=k, which="SM", return_eigenvectors=True
            )
            # Skip first eigenvector (constant)
            pe = torch.from_numpy(eigenvectors[:, 1:k]).float()
        except Exception:
            # Fallback for disconnected graphs
            pe = torch.zeros(num_nodes, pe_dim)

        # Pad if needed
        if padding and pe.size(1) < pe_dim:
            pe = F.pad(pe, (0, pe_dim - pe.size(1)))

        return pe


class RandomWalkSE(nn.Module):
    """
    Random Walk Structural Encoding (RWSE).

    Computes diagonal of random walk matrix powers as structural features.
    Captures local neighborhood structure.

    Reference: GraphGPS (NeurIPS'22)

    Note: RWSE is computed once during preprocessing.
    """

    def __init__(
        self,
        walk_length: int,
        hidden_dim: int,
        num_layers: int = 2,
    ):
        """
        Args:
            walk_length: Maximum random walk length (K)
            hidden_dim: Output dimension after MLP
            num_layers: Number of MLP layers
        """
        super().__init__()
        self.walk_length = walk_length
        self.hidden_dim = hidden_dim

        # MLP to project RWSE to hidden dimension
        layers = []
        in_dim = walk_length
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, rwse: Tensor) -> Tensor:
        """
        Args:
            rwse: (num_nodes, walk_length) precomputed RWSE

        Returns:
            (num_nodes, hidden_dim) structural encoding
        """
        return self.mlp(rwse)

    @staticmethod
    def compute_rwse(
        edge_index: Tensor,
        num_nodes: int,
        walk_length: int,
    ) -> Tensor:
        """
        Compute Random Walk Structural Encoding (preprocessing).

        Args:
            edge_index: (2, num_edges) graph connectivity
            num_nodes: Number of nodes
            walk_length: Maximum walk length K

        Returns:
            (num_nodes, walk_length) RWSE features
        """
        try:
            from torch_geometric.utils import to_scipy_sparse_matrix, degree
        except ImportError:
            raise ImportError("torch_geometric required for RWSE computation")

        # Build adjacency matrix
        row, col = edge_index[0], edge_index[1]
        deg = degree(row, num_nodes=num_nodes)
        deg_inv = 1.0 / deg.clamp(min=1)

        # Normalized adjacency (random walk matrix)
        edge_weight = deg_inv[row]

        import scipy.sparse as sp
        import numpy as np

        A = sp.coo_matrix(
            (edge_weight.numpy(), (row.numpy(), col.numpy())),
            shape=(num_nodes, num_nodes),
        ).tocsr()

        # Compute powers of A and extract diagonals
        rwse = torch.zeros(num_nodes, walk_length)
        Ak = sp.eye(num_nodes, format="csr")

        for k in range(walk_length):
            Ak = Ak @ A
            rwse[:, k] = torch.from_numpy(Ak.diagonal()).float()

        return rwse


# ==============================================================================
# Combined Positional Encoder
# ==============================================================================
class PositionalEncoder(nn.Module):
    """
    Combined positional/structural encoder.

    Supports multiple encoding types:
    - Laplacian PE (global structure)
    - Random Walk SE (local structure)
    - Degree encoding (simple baseline)

    All encodings are summed or concatenated.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_lap_pe: bool = True,
        use_rwse: bool = True,
        use_degree: bool = True,
        lap_pe_dim: int = 16,
        rwse_walk_length: int = 20,
        max_degree: int = 512,
        combination: str = "add",  # "add" or "concat"
    ):
        """
        Args:
            hidden_dim: Output dimension
            use_lap_pe: Whether to use Laplacian PE
            use_rwse: Whether to use RWSE
            use_degree: Whether to use degree encoding
            lap_pe_dim: Dimension of Laplacian PE
            rwse_walk_length: Length for RWSE
            max_degree: Maximum degree for degree encoding
            combination: How to combine encodings
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_lap_pe = use_lap_pe
        self.use_rwse = use_rwse
        self.use_degree = use_degree
        self.combination = combination

        # Initialize enabled encoders
        if use_lap_pe:
            self.lap_pe = LaplacianPE(lap_pe_dim, hidden_dim)

        if use_rwse:
            self.rwse = RandomWalkSE(rwse_walk_length, hidden_dim)

        if use_degree:
            self.degree_encoder = nn.Embedding(max_degree + 1, hidden_dim)
            nn.init.normal_(self.degree_encoder.weight, std=0.02)

        # For concat mode, need projection
        if combination == "concat":
            num_encodings = sum([use_lap_pe, use_rwse, use_degree])
            self.projection = nn.Linear(hidden_dim * num_encodings, hidden_dim)

    def forward(
        self,
        lap_pe: Optional[Tensor] = None,
        rwse: Optional[Tensor] = None,
        degree: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            lap_pe: (num_nodes, lap_pe_dim) Laplacian eigenvectors
            rwse: (num_nodes, walk_length) RWSE features
            degree: (num_nodes,) node degrees

        Returns:
            (num_nodes, hidden_dim) combined positional encoding
        """
        encodings = []

        if self.use_lap_pe and lap_pe is not None:
            encodings.append(self.lap_pe(lap_pe))

        if self.use_rwse and rwse is not None:
            encodings.append(self.rwse(rwse))

        if self.use_degree and degree is not None:
            # Clamp degree to max
            degree = degree.clamp(max=self.degree_encoder.num_embeddings - 1)
            encodings.append(self.degree_encoder(degree))

        if not encodings:
            raise ValueError("At least one positional encoding must be provided")

        if self.combination == "add":
            return sum(encodings)
        else:  # concat
            return self.projection(torch.cat(encodings, dim=-1))


# ==============================================================================
# Feature Projection for Heterogeneous Nodes
# ==============================================================================
class HeteroFeatureEncoder(nn.Module):
    """
    Projects heterogeneous node features to unified hidden space.

    Each node type can have different input feature dimensions,
    but all are projected to the same hidden_dim for message passing.

    torch.compile() compatible via dict-based operations.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_dim: int,
        use_type_embedding: bool = True,
    ):
        """
        Args:
            in_channels_dict: {node_type: input_dim} for each type
            hidden_dim: Unified output dimension
            use_type_embedding: Whether to add type embedding
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_type_embedding = use_type_embedding

        # Type-specific projections
        self.projections = nn.ModuleDict({
            node_type: nn.Linear(in_dim, hidden_dim)
            for node_type, in_dim in in_channels_dict.items()
        })

        # Type embeddings
        if use_type_embedding:
            self.type_embeddings = nn.ParameterDict({
                node_type: nn.Parameter(torch.randn(hidden_dim) * 0.02)
                for node_type in in_channels_dict.keys()
            })

    def forward(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            x_dict: {node_type: (num_nodes, in_dim)} input features

        Returns:
            {node_type: (num_nodes, hidden_dim)} projected features
        """
        out_dict = {}

        for node_type, x in x_dict.items():
            if node_type in self.projections:
                h = self.projections[node_type](x)

                if self.use_type_embedding and node_type in self.type_embeddings:
                    h = h + self.type_embeddings[node_type]

                out_dict[node_type] = h

        return out_dict
