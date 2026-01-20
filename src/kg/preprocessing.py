"""
# ==============================================================================
# Module: src/kg/preprocessing.py
# ==============================================================================
# Purpose: Graph preprocessing utilities for GNN input preparation
#
# Dependencies:
#   - External: torch, torch_geometric, scipy, numpy
#   - Internal: src.kg.graph (KnowledgeGraph)
#
# Input:
#   - KnowledgeGraph or PyG HeteroData
#
# Output:
#   - Preprocessed HeteroData with positional encodings and node features
#
# Exports:
#   - compute_laplacian_pe: Compute Laplacian positional encodings
#   - compute_rwse: Compute random walk structural encodings
#   - compute_degree_features: Compute node degree features
#   - preprocess_for_gnn: Full preprocessing pipeline
#
# Design Notes:
#   - All preprocessing is done BEFORE model training (not in forward pass)
#   - Supports heterogeneous graphs with per-type preprocessing
#   - Caches computed features for efficiency
# ==============================================================================
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.kg.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def compute_laplacian_pe(
    edge_index: "np.ndarray",
    num_nodes: int,
    pe_dim: int = 16,
) -> "np.ndarray":
    """
    Compute Laplacian Positional Encoding.

    Eigendecomposition of the normalized graph Laplacian.
    Returns the k smallest non-trivial eigenvectors.

    Args:
        edge_index: (2, num_edges) edge connectivity
        num_nodes: Number of nodes
        pe_dim: Number of eigenvectors to compute

    Returns:
        (num_nodes, pe_dim) positional encoding matrix

    Reference:
        GraphGPS (NeurIPS'22) - Recipe for a General, Powerful, Scalable Graph Transformer
    """
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import eigsh
    except ImportError:
        logger.warning("scipy not available, returning zero PE")
        return np.zeros((num_nodes, pe_dim), dtype=np.float32)

    if num_nodes == 0:
        return np.zeros((0, pe_dim), dtype=np.float32)

    # Build adjacency matrix
    row = edge_index[0]
    col = edge_index[1]
    data = np.ones(len(row), dtype=np.float32)

    # Make symmetric (for undirected Laplacian)
    row_sym = np.concatenate([row, col])
    col_sym = np.concatenate([col, row])
    data_sym = np.concatenate([data, data])

    A = coo_matrix((data_sym, (row_sym, col_sym)), shape=(num_nodes, num_nodes))
    A = A.tocsr()

    # Compute degree matrix
    deg = np.array(A.sum(axis=1)).flatten()
    deg = np.maximum(deg, 1)  # Avoid division by zero

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = coo_matrix((deg_inv_sqrt, (np.arange(num_nodes), np.arange(num_nodes))),
                            shape=(num_nodes, num_nodes)).tocsr()

    L = coo_matrix((np.ones(num_nodes), (np.arange(num_nodes), np.arange(num_nodes))),
                   shape=(num_nodes, num_nodes)).tocsr() - D_inv_sqrt @ A @ D_inv_sqrt

    # Compute eigenvectors
    k = min(pe_dim + 1, num_nodes - 1)  # +1 because we skip the trivial eigenvector
    if k <= 0:
        return np.zeros((num_nodes, pe_dim), dtype=np.float32)

    try:
        # Compute smallest eigenvalues/eigenvectors
        eigenvalues, eigenvectors = eigsh(L, k=k, which='SM', tol=1e-3)

        # Sort by eigenvalue (ascending)
        idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, idx]

        # Skip the first (trivial) eigenvector, take next pe_dim
        pe = eigenvectors[:, 1:pe_dim + 1]

        # Pad if not enough eigenvectors
        if pe.shape[1] < pe_dim:
            padding = np.zeros((num_nodes, pe_dim - pe.shape[1]), dtype=np.float32)
            pe = np.concatenate([pe, padding], axis=1)

        return pe.astype(np.float32)

    except Exception as e:
        logger.warning(f"Eigendecomposition failed: {e}, returning zero PE")
        return np.zeros((num_nodes, pe_dim), dtype=np.float32)


def compute_rwse(
    edge_index: "np.ndarray",
    num_nodes: int,
    walk_length: int = 20,
) -> "np.ndarray":
    """
    Compute Random Walk Structural Encoding.

    Diagonal entries of random walk matrices at different steps.
    Captures local structural information.

    Args:
        edge_index: (2, num_edges) edge connectivity
        num_nodes: Number of nodes
        walk_length: Maximum walk length (number of steps)

    Returns:
        (num_nodes, walk_length) RWSE matrix

    Reference:
        GraphGPS (NeurIPS'22)
    """
    try:
        from scipy.sparse import coo_matrix
    except ImportError:
        logger.warning("scipy not available, returning zero RWSE")
        return np.zeros((num_nodes, walk_length), dtype=np.float32)

    if num_nodes == 0:
        return np.zeros((0, walk_length), dtype=np.float32)

    # Build adjacency matrix
    row = edge_index[0]
    col = edge_index[1]
    data = np.ones(len(row), dtype=np.float32)

    # Make symmetric
    row_sym = np.concatenate([row, col])
    col_sym = np.concatenate([col, row])
    data_sym = np.concatenate([data, data])

    A = coo_matrix((data_sym, (row_sym, col_sym)), shape=(num_nodes, num_nodes))
    A = A.tocsr()

    # Compute transition matrix: P = D^{-1} A
    deg = np.array(A.sum(axis=1)).flatten()
    deg = np.maximum(deg, 1)  # Avoid division by zero
    deg_inv = 1.0 / deg

    D_inv = coo_matrix((deg_inv, (np.arange(num_nodes), np.arange(num_nodes))),
                       shape=(num_nodes, num_nodes)).tocsr()

    P = D_inv @ A

    # Compute diagonal of P^k for k = 1, ..., walk_length
    rwse = np.zeros((num_nodes, walk_length), dtype=np.float32)
    Pk = P.copy()

    for k in range(walk_length):
        rwse[:, k] = Pk.diagonal()
        Pk = Pk @ P

    return rwse


def compute_degree_features(
    edge_index: "np.ndarray",
    num_nodes: int,
    max_degree: int = 128,
) -> "np.ndarray":
    """
    Compute node degree features.

    Args:
        edge_index: (2, num_edges) edge connectivity
        num_nodes: Number of nodes
        max_degree: Maximum degree to encode (higher degrees clipped)

    Returns:
        (num_nodes,) degree array
    """
    if num_nodes == 0:
        return np.zeros(0, dtype=np.int64)

    # Count degrees (both directions for undirected)
    row = edge_index[0]
    col = edge_index[1]

    degree = np.zeros(num_nodes, dtype=np.int64)
    np.add.at(degree, row, 1)
    np.add.at(degree, col, 1)

    # Clip to max_degree
    degree = np.minimum(degree, max_degree)

    return degree


def preprocess_for_gnn(
    kg: "KnowledgeGraph",
    compute_pe: bool = True,
    compute_rwse_flag: bool = True,
    pe_dim: int = 16,
    rwse_walk_length: int = 20,
    initial_feature_dim: int = 64,
) -> Tuple[Any, Dict[str, Dict[str, Any]]]:
    """
    Full preprocessing pipeline for GNN input.

    Converts KnowledgeGraph to PyG HeteroData with:
    - Node initial features (learnable embeddings or from attributes)
    - Positional encodings (Laplacian PE)
    - Structural encodings (RWSE)
    - Degree features

    Args:
        kg: KnowledgeGraph instance
        compute_pe: Whether to compute Laplacian PE
        compute_rwse_flag: Whether to compute RWSE
        pe_dim: Dimension of positional encoding
        rwse_walk_length: Length of random walks for RWSE
        initial_feature_dim: Dimension of initial node features

    Returns:
        (HeteroData, pos_encoding_dict) where:
        - HeteroData: PyG graph with node features and edge indices
        - pos_encoding_dict: {node_type: {"lap_pe": ..., "rwse": ..., "degree": ...}}

    Usage:
        data, pe_dict = preprocess_for_gnn(kg)
        out = model(data.x_dict, data.edge_index_dict, pe_dict)
    """
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError:
        raise ImportError(
            "torch and torch_geometric required for preprocessing. "
            "Install via: pip install torch torch_geometric"
        )

    # Get basic HeteroData from KG
    data = kg.to_pyg_hetero_data()

    # Get node type info
    node_types, edge_types = kg.metadata()

    # Prepare positional encoding dict
    pos_encoding_dict: Dict[str, Dict[str, Any]] = {}

    # Process each node type
    for node_type in node_types:
        if node_type not in data.node_types:
            continue

        num_nodes = data[node_type].num_nodes
        if num_nodes == 0:
            continue

        # Initialize positional encoding storage
        pos_encoding_dict[node_type] = {}

        # Replace placeholder features with learnable-size embeddings
        # (actual values will be learned, this is just shape initialization)
        data[node_type].x = torch.randn(num_nodes, initial_feature_dim) * 0.01

        # Collect edges involving this node type (as destination)
        combined_edge_index = []
        for et in edge_types:
            src_type, rel_type, dst_type = et
            if dst_type == node_type and et in data.edge_types:
                edge_idx = data[et].edge_index
                # Map source indices to global space for this computation
                combined_edge_index.append(edge_idx.numpy())

        # If we have edges, compute structural features
        if combined_edge_index:
            # For positional encodings, we need edges within the same type
            # or we compute on the homogeneous projection
            # Here we compute on incoming edges to this node type
            all_edges = np.concatenate(combined_edge_index, axis=1) if combined_edge_index else np.zeros((2, 0), dtype=np.int64)

            # Compute degree
            degree = compute_degree_features(all_edges, num_nodes)
            pos_encoding_dict[node_type]["degree"] = torch.from_numpy(degree).long()

        else:
            # No edges, use zeros
            pos_encoding_dict[node_type]["degree"] = torch.zeros(num_nodes, dtype=torch.long)

        # For Laplacian PE and RWSE, we need the full graph structure
        # We'll compute these on the homogeneous projection of the subgraph
        if compute_pe or compute_rwse_flag:
            # Get all edges where this node type is involved
            homo_edges = _get_homogeneous_edges_for_type(data, node_type, edge_types)

            if compute_pe:
                lap_pe = compute_laplacian_pe(homo_edges, num_nodes, pe_dim)
                pos_encoding_dict[node_type]["lap_pe"] = torch.from_numpy(lap_pe).float()

            if compute_rwse_flag:
                rwse = compute_rwse(homo_edges, num_nodes, rwse_walk_length)
                pos_encoding_dict[node_type]["rwse"] = torch.from_numpy(rwse).float()

    logger.info(f"Preprocessing complete: {len(node_types)} node types, {len(edge_types)} edge types")

    return data, pos_encoding_dict


def _get_homogeneous_edges_for_type(
    data: Any,
    node_type: str,
    edge_types: List[Tuple[str, str, str]],
) -> "np.ndarray":
    """
    Get edges involving a specific node type as a homogeneous graph.

    For Laplacian PE and RWSE computation.
    """
    edges = []

    for et in edge_types:
        src_type, rel_type, dst_type = et

        # Only consider edges within the same node type
        if src_type == node_type and dst_type == node_type:
            if et in data.edge_types:
                edge_idx = data[et].edge_index.numpy()
                edges.append(edge_idx)

    if edges:
        return np.concatenate(edges, axis=1)
    else:
        return np.zeros((2, 0), dtype=np.int64)
