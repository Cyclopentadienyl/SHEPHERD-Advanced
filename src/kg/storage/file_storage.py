"""
Reading the knowledge graph's on-disk layout — one implementation.
==================================================================
The KG is written by `src/kg/builder.py` as files under
`data/workspaces/<kg>/`, and read back independently by every consumer that
needs it. This is the shared reader those copies are meant to collapse into.

**This is the P1's first callers, not the P1.** The package docstring records the
whole job: become the single reader of this layout, replacing all six copies.
Two of them are migrated here — the measurement harness's Mode C path and the
legacy Mode A loader that delegates to it — because B-0.3 needed a reader that
does **not** retire with the frozen evaluator, and adding another copy to get
one would have been the opposite of the point. `src/inference/pipeline.py`,
`scripts/train_model.py`, `scripts/build_index.py` and `scripts/setup_demo.py`
still have their own; migrating them belongs to P1 and is not smuggled in here.

**No abstraction beyond the two readers below.** No `Storage` Protocol, no
backend registry, no adapter hierarchy, no migration framework. The package name
says "backends" plural and that plural is aspirational; the shape a second
backend needs will be known when there is one.

**Each file is parsed from one read, and its digest comes from that read**
(contract M2.1). A reader returns what it parsed and a `ReadIdentity`, never the
bytes; each buffer is released before the next file is read, which the release
check in `tests/fixtures/release.py` shows. Torch files are parsed through this
module's own `BytesIO`, which that check replaces to watch the wrapper.

Module: src/kg/storage/file_storage.py
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple

from src.kg.artifacts import GRAPH_ARTIFACTS
from src.utils.fingerprint import ReadIdentity

__all__ = ["GraphArtifactsRead", "SamplesRead", "read_graph_artifacts", "read_samples"]


class GraphArtifactsRead(NamedTuple):
    """The graph tensors a consumer loads, and the identity of each file read.

    `reads` is keyed by manifest role (`GRAPH_ARTIFACTS`) and holds exactly the
    files that were read, so an absent file has neither a key in `graph_data`
    nor an identity.
    """

    graph_data: Dict[str, Any]
    reads: Dict[str, ReadIdentity]


class SamplesRead(NamedTuple):
    """One split's samples, and the identity of the file they were parsed from."""

    samples: List[Any]
    identity: ReadIdentity


def _read_json(path: Path) -> Tuple[Any, ReadIdentity]:
    """Decoded as UTF-8, as the workspace writers write it; the bytes are released
    before the text is parsed."""
    from src.utils.fingerprint import read_once

    read = read_once(path)
    identity = read.identity
    text = read.data.decode("utf-8")
    del read
    return json.loads(text), identity


def _read_tensors(path: Path, map_location: Any) -> Tuple[Any, ReadIdentity]:
    import torch

    from src.utils.fingerprint import read_once

    read = read_once(path)
    return (
        torch.load(BytesIO(read.data), weights_only=True, map_location=map_location),
        read.identity,
    )


def read_graph_artifacts(data_dir: Path, *, map_location: Any = "cpu") -> GraphArtifactsRead:
    """Load `node_features.pt`, `edge_indices.pt` and `num_nodes.json`.

    Absent files are omitted rather than defaulted: a caller that needs
    `x_dict` should fail on its absence with its own message, which says what
    the caller was trying to do, rather than receive an empty dict that behaves
    like a graph with no nodes.

    `weights_only=True` — these are tensor files, and loading them must not be
    able to execute code. They re-enter a clinical tool. `map_location="cpu"`,
    as serving already passes: an export is saved from CPU, so this matches
    loading with no `map_location` for every export the writer produces.
    """
    graph_data: Dict[str, Any] = {}
    reads: Dict[str, ReadIdentity] = {}
    for role, key in (("node_features", "x_dict"), ("edge_indices", "edge_index_dict")):
        path = data_dir / GRAPH_ARTIFACTS[role]
        if path.exists():
            graph_data[key], reads[role] = _read_tensors(path, map_location)
    num_nodes = data_dir / GRAPH_ARTIFACTS["num_nodes"]
    if num_nodes.exists():
        graph_data["num_nodes_dict"], reads["num_nodes"] = _read_json(num_nodes)
    return GraphArtifactsRead(graph_data, reads)


def read_samples(data_dir: Path, split: str, *, training_fields: bool = False) -> SamplesRead:
    """Load `<split>_samples.json` as `DiagnosisSample` objects.

    Missing is an error, not an empty cohort: an empty list would flow into a
    measurement and produce metrics over nobody.

    **The error names the splits that do exist, and never falls back to one.**
    `src/kg/sample_generator.py` writes `train_samples.json` and
    `val_samples.json` only, so a caller asking for `test` on an ordinary
    workspace gets a file that was never generated. Listing what is present
    turns that from "which path did I mistype" into "this workspace has no test
    split" — and choosing a substitute here would silently answer a question
    the caller has to answer, since `val` is the checkpoint-selection split
    rather than held-out data.

    **Three fields by default; `training_fields=True` adds the two only
    training reads,** `candidate_disease_ids` and `gene_ids`. Measurement reads
    three fields, as the frozen evaluator did, and `gene_ids` become subgraph
    seeds (`src/kg/data_loader.py:675-676`), so carrying them by default would
    change what Modes A and B measure.
    """
    from src.kg.data_loader import DiagnosisSample

    path = data_dir / f"{split}_samples.json"
    if not path.exists():
        available = sorted(
            candidate.name[: -len("_samples.json")]
            # `is_file()`: a directory named `foo_samples.json` is not a split,
            # and reporting one as available would send the caller after a name
            # that can never load.
            for candidate in data_dir.glob("*_samples.json")
            if candidate.is_file()
        )
        detail = (
            f"this workspace has: {', '.join(available)}"
            if available
            else "this workspace has no *_samples.json files at all"
        )
        raise FileNotFoundError(f"Samples file not found: {path} — {detail}")

    entries, identity = _read_json(path)
    if training_fields:
        samples = [
            DiagnosisSample(
                patient_id=entry["patient_id"],
                phenotype_ids=entry["phenotype_ids"],
                disease_id=entry["disease_id"],
                candidate_disease_ids=entry.get("candidate_disease_ids"),
                gene_ids=entry.get("gene_ids"),
            )
            for entry in entries
        ]
    else:
        samples = [
            DiagnosisSample(
                patient_id=entry["patient_id"],
                phenotype_ids=entry["phenotype_ids"],
                disease_id=entry["disease_id"],
            )
            for entry in entries
        ]
    return SamplesRead(samples, identity)
