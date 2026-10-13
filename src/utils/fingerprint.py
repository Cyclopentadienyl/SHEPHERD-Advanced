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

*Content digests* (`read_once`, `file_sha256`) answer the second question. They are
kept separate rather than folded into the fingerprint, because
a structural identity that also changed whenever a byte moved would stop being a
compatibility check.

`read_once` is the content digest of an input that is about to be parsed: it
returns the bytes and the digest of those same bytes, so what a reader parses and
what a record names cannot come from two different reads (contract M2.1).
`file_sha256` identifies a file nobody is about to parse. Within the static
pin's scope (`src/**`, the entry points and the scripts M2.1's S9 moved), every
remaining caller is a named exception in `tests/unit/test_provenance_pins.py`;
BACKLOG item 19's scripts outside that scope still hash by path, each with its
home there.

When a checkpoint is saved during training, the fingerprint of the
graph data used is embedded in the checkpoint. At inference load time,
the current graph data's fingerprint is compared against the saved one.
Mismatches produce a WARNING (not a hard error) so operators can decide: the
fingerprint is a diagnostic, never identity.

**Identity is `verify_training_record`'s** (contract M2.4): the one comparator
between a checkpoint's `training_input_digests` and the digests a consumer holds
for the same roles. It raises; it does not warn.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadIdentity:
    """Which file was read, and the SHA-256 of the bytes that read returned.

    It carries no bytes, so a reader can keep it beside what it parsed and let the
    raw buffer go.
    """

    path: Path
    sha256: str


@dataclass(frozen=True)
class FileRead:
    """One read of one file: its bytes, and the SHA-256 of exactly those bytes.

    A reader parses `data`, keeps `identity`, and drops this object, so the buffer
    is released once the parse is done.
    """

    path: Path
    data: bytes
    sha256: str

    @property
    def identity(self) -> ReadIdentity:
        return ReadIdentity(path=self.path, sha256=self.sha256)


def read_once(path: Union[str, Path]) -> FileRead:
    """Read a file once and return its bytes with their SHA-256.

    **One open, one full read, and the digest is taken over that same bytes
    object.** Hashing a path and then loading it reads the file twice, and a file
    replaced in between is parsed as one thing and recorded as another. Hashing an
    open handle and then parsing from it survives an atomic rename but not an
    in-place rewrite, which is how `torch.save` writes; hashing the buffer that is
    parsed survives both.

    **A missing file raises** (`FileNotFoundError`), unlike `file_sha256`. A file
    deleted between two reads is then refused instead of recorded as absent.

    No retry, no fallback. The read is alone on its line: the readers' release
    check (contract M2.1, S2) finds the raw buffer by the allocation that line
    makes, so nothing a reader keeps may be made on it.
    """
    path = Path(path)
    with path.open("rb") as handle:
        data = handle.read()
    sha256 = hashlib.sha256(data).hexdigest()
    return FileRead(path=path, data=data, sha256=sha256)


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



class TrainingRecordMismatch(ValueError):
    """A checkpoint's training record does not name the inputs it is compared
    with, or records none."""


def verify_training_record(
    checkpoint: Mapping[str, Any],
    current: Mapping[str, str],
    roles: Iterable[str],
    *,
    recorded_by: str,
    compared_with: str,
) -> None:
    """Refuse unless `checkpoint` records, for every role in `roles`, the digest
    `current` holds for it (contract M2.4).

    **The one model-to-graph comparator.** Serving and measurement compare the
    four graph roles with the workspace they verified, and the deployment probe
    asserts through this function; resume compares its parent with the run over
    the graph and data roles (M2.3). The set of roles is the argument and the
    rule is the same for every caller.

    **Present on both sides and equal, or refused.** A checkpoint with no
    `training_input_digests` (one from before the contract), a role it does not
    record, a role `current` does not hold and a different digest are each
    refused by name, so an empty record cannot pass vacuously. `current` comes
    from the caller's reads; nothing here opens a file.

    Args:
        checkpoint: the checkpoint as read.
        current: role -> SHA-256 the consumer holds for the inputs in front of it.
        roles: the roles compared; must not be empty.
        recorded_by: names the checkpoint in a refusal ("checkpoint <path>").
        compared_with: names `current` in a refusal ("the workspace <path>").

    Raises:
        TrainingRecordMismatch: any role is missing on either side or differs.
    """
    roles = tuple(roles)
    if not roles:
        raise ValueError("no roles to compare; an empty comparison passes anything")
    record = (
        checkpoint.get("training_input_digests")
        if isinstance(checkpoint, Mapping) else None
    )
    if not isinstance(record, Mapping) or not record:
        raise TrainingRecordMismatch(
            f"{recorded_by} records no training_input_digests, so the inputs it was "
            f"trained on cannot be compared with {compared_with}. A checkpoint from "
            "before the provenance contract is not used; retrain it."
        )
    problems: List[str] = []
    for role in roles:
        recorded, held = record.get(role), current.get(role)
        if not recorded:
            problems.append(f"{role} is not recorded")
        elif not held:
            problems.append(f"{role} is recorded and {compared_with} has none")
        elif recorded != held:
            problems.append(
                f"{role} differs (recorded {str(recorded)[:12]}..., "
                f"{compared_with} has {str(held)[:12]}...)"
            )
    if problems:
        raise TrainingRecordMismatch(
            f"{recorded_by} does not match {compared_with}: "
            + "; ".join(problems)
            + ". Its record names the bytes it was trained on, and these are not "
            "they."
        )


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
    That question is `verify_training_record`'s, which compares the digests by
    role and refuses; this stays a diagnostic beside it.

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
