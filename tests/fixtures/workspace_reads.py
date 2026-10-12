"""
A workspace read and checked as a graph consumer reads and checks it.
=====================================================================
Since contract M2.1's S9 there is no path-form verifier to call on a directory:
every consumer reads each file once through the shared readers and checks those
reads against its one manifest reading. Tests that ask "does a consumer accept
the workspace this writer produced?" call this, which composes the same readers
and read-form verifiers serving and training use — `kg.json` parsed and its
identity compared, as serving does, and both generated cohorts, as training
does. It adds no rule of its own.

Module: tests/fixtures/workspace_reads.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

GENERATED = ("train", "val")


def verify_workspace_reads(
    root: Path, splits: Sequence[str] = GENERATED
) -> Tuple[Dict[str, str], Any]:
    """Read `root`'s manifest, `kg.json`, tensors and cohorts once each, and
    verify them from those reads. Returns the manifest-bound graph digests and
    the verified cohorts (`GeneratedCohorts`)."""
    from src.evaluation.cohort import verify_cohort_reads
    from src.kg.artifacts import GRAPH_ARTIFACTS, read_split_manifest, verify_graph_source_read
    from src.kg.graph import KnowledgeGraph
    from src.kg.storage.file_storage import read_graph_artifacts, read_samples

    root = Path(root)
    manifest = read_split_manifest(root)
    _, graph = KnowledgeGraph.read_json(root / GRAPH_ARTIFACTS["kg"])
    tensors = read_graph_artifacts(root)
    bound = verify_graph_source_read(manifest, graph, tensors.reads)
    cohorts = verify_cohort_reads(manifest, {split: read_samples(root, split) for split in splits})
    return bound, cohorts
