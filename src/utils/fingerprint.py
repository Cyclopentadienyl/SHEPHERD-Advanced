"""
Data Fingerprint Utilities
===========================
Compute and verify compatibility fingerprints for graph data,
ensuring that model checkpoints are loaded with the same KG
structure they were trained on.

Module: src/utils/fingerprint.py

A fingerprint captures the structural identity of a graph dataset:
  - Node types and their counts
  - Edge types (including reverse edges)
  - Feature dimensions per node type
  - Total KG node/edge counts

When a checkpoint is saved during training, the fingerprint of the
graph data used is embedded in the checkpoint. At inference load time,
the current graph data's fingerprint is compared against the saved one.
Mismatches produce a WARNING (not a hard error) so operators can decide.

Usage:
    from src.utils.fingerprint import compute_fingerprint, verify_fingerprint

    # During training:
    fp = compute_fingerprint(graph_data)
    checkpoint["data_fingerprint"] = fp

    # During inference:
    warnings = verify_fingerprint(checkpoint, current_graph_data)
    if warnings:
        for w in warnings:
            logger.warning(w)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_fingerprint(
    graph_data: Dict[str, Any],
    kg_total_nodes: Optional[int] = None,
    kg_total_edges: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute a structural fingerprint from graph data tensors.

    Args:
        graph_data: dict with keys "x_dict", "edge_index_dict", "num_nodes_dict"
        kg_total_nodes: total KG node count (optional, for extra validation)
        kg_total_edges: total KG edge count (optional, for extra validation)

    Returns:
        Fingerprint dict suitable for embedding in a checkpoint.
    """
    x_dict = graph_data.get("x_dict", {})
    edge_index_dict = graph_data.get("edge_index_dict", {})
    num_nodes_dict = graph_data.get("num_nodes_dict", {})

    node_types = sorted(x_dict.keys())

    feature_dims = {}
    for nt, tensor in x_dict.items():
        if hasattr(tensor, "size") and tensor.dim() >= 2:
            feature_dims[nt] = tensor.size(-1)

    edge_types = sorted(
        [list(k) if isinstance(k, tuple) else k for k in edge_index_dict.keys()],
        key=str,
    )

    node_counts = {k: int(v) for k, v in num_nodes_dict.items()}

    fp: Dict[str, Any] = {
        "node_types": node_types,
        "node_counts": node_counts,
        "feature_dims": feature_dims,
        "edge_types": edge_types,
        "num_edge_types": len(edge_types),
    }

    if kg_total_nodes is not None:
        fp["kg_total_nodes"] = kg_total_nodes
    if kg_total_edges is not None:
        fp["kg_total_edges"] = kg_total_edges

    return fp


def verify_fingerprint(
    checkpoint: Dict[str, Any],
    current_graph_data: Dict[str, Any],
    kg_total_nodes: Optional[int] = None,
    kg_total_edges: Optional[int] = None,
) -> List[str]:
    """
    Verify that a checkpoint's data fingerprint matches current graph data.

    Returns a list of warning strings. Empty list = compatible.
    Non-empty = mismatches detected (but not necessarily fatal).

    Args:
        checkpoint: loaded checkpoint dict (may or may not have "data_fingerprint")
        current_graph_data: current graph data dict
        kg_total_nodes: current KG total nodes (optional)
        kg_total_edges: current KG total edges (optional)

    Returns:
        List of human-readable warning strings (English).
    """
    saved_fp = checkpoint.get("data_fingerprint")
    if saved_fp is None:
        return [
            "Checkpoint does not contain a data fingerprint "
            "(legacy format). Cannot verify KG compatibility. "
            "Proceed with caution."
        ]

    current_fp = compute_fingerprint(
        current_graph_data,
        kg_total_nodes=kg_total_nodes,
        kg_total_edges=kg_total_edges,
    )

    warnings: List[str] = []

    # Check node types
    saved_nt = set(saved_fp.get("node_types", []))
    current_nt = set(current_fp.get("node_types", []))
    if saved_nt != current_nt:
        added = current_nt - saved_nt
        removed = saved_nt - current_nt
        parts = ["Node type mismatch:"]
        if added:
            parts.append(f"added {added}")
        if removed:
            parts.append(f"removed {removed}")
        warnings.append(" ".join(parts))

    # Check node counts
    saved_nc = saved_fp.get("node_counts", {})
    current_nc = current_fp.get("node_counts", {})
    for nt in set(saved_nc) | set(current_nc):
        s = saved_nc.get(nt, 0)
        c = current_nc.get(nt, 0)
        if s != c:
            warnings.append(
                f"Node count mismatch for '{nt}': "
                f"checkpoint expects {s}, current data has {c}"
            )

    # Check edge types count
    saved_net = saved_fp.get("num_edge_types", 0)
    current_net = current_fp.get("num_edge_types", 0)
    if saved_net != current_net:
        warnings.append(
            f"Edge type count mismatch: "
            f"checkpoint expects {saved_net}, current data has {current_net}"
        )

    # Check feature dimensions
    saved_fd = saved_fp.get("feature_dims", {})
    current_fd = current_fp.get("feature_dims", {})
    for nt in set(saved_fd) | set(current_fd):
        s = saved_fd.get(nt)
        c = current_fd.get(nt)
        if s is not None and c is not None and s != c:
            warnings.append(
                f"Feature dimension mismatch for '{nt}': "
                f"checkpoint expects {s}, current data has {c}. "
                f"Model weights will not load correctly."
            )

    if warnings:
        warnings.insert(
            0,
            "KG/data version mismatch detected between checkpoint and "
            "current data. Inference results may be incorrect."
        )

    return warnings
