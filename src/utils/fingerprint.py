"""
Data Fingerprint Utilities
===========================
Compute and verify compatibility fingerprints for graph data,
ensuring that model checkpoints are loaded with the same KG
structure they were trained on.

Module: src/utils/fingerprint.py

**Two identities live here, and they answer different questions.**

A *fingerprint* captures the structural identity of a graph dataset:
  - Node types and their counts
  - Edge types (including reverse edges)
  - Feature dimensions per node type
  - Total KG node/edge counts

It answers "is this checkpoint structurally compatible with the graph in front of
me?". It cannot answer "which exact inputs produced this checkpoint" — two runs
over different sample files share it, and sample files do not enter it at all.

*Content digests* (`file_sha256`, `compute_input_digests`) answer the second
question. They are kept separate rather than folded into the fingerprint, because
a structural identity that also changed whenever a byte moved would stop being a
compatibility check.

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

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

logger = logging.getLogger(__name__)


def file_sha256(path: Union[str, Path]) -> Optional[str]:
    """Raw content digest, or ``None`` if the file is not there.

    `hashlib.file_digest` (stdlib, 3.11+) reads in chunks, so a multi-gigabyte
    checkpoint is not loaded into memory to be identified. This hashes **bytes**
    and nothing else: no canonical form, no key ordering, no serialisation policy.
    Two runs quoting the same digest consumed the same file, which is the entire
    claim being made.

    **Lives here rather than in a script.** It was defined in
    `scripts/measure_scorer.py` and imported from there by
    `scripts/calibrate_mode_a.py`, `scripts/benchmark_sp_lookup.py` and a test —
    scripts importing a script, which makes an entry point into a library. The SP
    benchmark in particular has nothing to do with scorer measurement and had no
    business depending on it. `src.utils` is the bottom layer, so every caller can
    reach this without reaching through anything else.

    **Absent is `None`, not an error.** A role a run did not consume, and a file
    that is genuinely missing, are both recorded rather than raised on; the caller
    decides what an absent input means. Note that the digest is taken at a
    different instant from any load, so a file changed in between would be recorded
    incorrectly — a limitation, not something locking is added for.
    """
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compute_input_digests(roles: Mapping[str, Union[str, Path]]) -> Dict[str, Optional[str]]:
    """Content digests keyed by the **semantic role** each file played.

    Role-keyed and nothing more. This function knows nothing about splits,
    checkpoints, training or measurement — each caller names the roles *its own
    run* consumed, and two callers with different roles do not have to agree on a
    vocabulary or share a schema. That is what keeps a shared digest contract from
    turning into a shared domain model.

    What it does own is the one decision both callers must not disagree about:
    a missing file is `None`, uniformly. Two dict comprehensions could drift on
    that; this cannot.

    Callers should pass **only the roles the run actually consumed**. Hashing every
    file that happens to sit in a directory records the directory, not the run —
    an unrelated split appearing beside the inputs would change a record of work
    that never read it.
    """
    return {role: file_sha256(path) for role, path in roles.items()}


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

    **This checks structure only, and does not look at
    `training_input_digests`.** An empty warning list therefore means "the graph is
    structurally compatible", never "this checkpoint was trained on these files".
    The digests are captured so the question *can* be answered later; comparing
    them against a current workspace needs decisions this function does not yet
    have — which path stands for each semantic role, and how a relocated
    workspace, an absent input or a checkpoint predating the field should read.
    That comparison is deliberately deferred rather than half-built here, and the
    absence of a digest map means only "not recorded".

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
