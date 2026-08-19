"""
Reading the knowledge graph's on-disk layout — one implementation.
==================================================================
The KG is written by `src/kg/builder.py` as files under
`data/workspaces/<kg>/`, and read back independently by every consumer that
needs it. This is the shared reader those copies are meant to collapse into.

**This is the P1's first callers, not the P1.** The package docstring records the
whole job: become the single reader of this layout, replacing all seven copies.
Two of them are migrated here — the measurement harness's Mode C path and the
legacy Mode A loader that delegates to it — because B-0.3 needed a reader that
does **not** retire with the frozen evaluator, and adding an eighth copy to get
one would have been the opposite of the point. `src/inference/pipeline.py`,
`scripts/train_model.py`, `scripts/evaluate_model.py`, `scripts/build_index.py`
and `scripts/setup_demo.py` still have their own; migrating them belongs to P1
and is not smuggled in here.

**No abstraction beyond the two functions below.** No `Storage` Protocol, no
backend registry, no adapter hierarchy, no migration framework. The package name
says "backends" plural and that plural is aspirational; the shape a second
backend needs will be known when there is one.

Module: src/kg/storage/file_storage.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

__all__ = ["read_graph_artifacts", "read_samples"]


def read_graph_artifacts(data_dir: Path) -> Dict[str, Any]:
    """Load `node_features.pt`, `edge_indices.pt` and `num_nodes.json`.

    Absent files are omitted rather than defaulted: a caller that needs
    `x_dict` should fail on its absence with its own message, which says what
    the caller was trying to do, rather than receive an empty dict that behaves
    like a graph with no nodes.

    `weights_only=True` — these are tensor files, and loading them must not be
    able to execute code. They re-enter a clinical tool.
    """
    import torch

    artifacts: Dict[str, Any] = {}
    node_features = data_dir / "node_features.pt"
    if node_features.exists():
        artifacts["x_dict"] = torch.load(node_features, weights_only=True)
    edge_indices = data_dir / "edge_indices.pt"
    if edge_indices.exists():
        artifacts["edge_index_dict"] = torch.load(edge_indices, weights_only=True)
    num_nodes = data_dir / "num_nodes.json"
    if num_nodes.exists():
        artifacts["num_nodes_dict"] = json.loads(num_nodes.read_text())
    return artifacts


def read_samples(data_dir: Path, split: str) -> List[Any]:
    """Load `<split>_samples.json` as `DiagnosisSample` objects.

    Missing is an error, not an empty cohort: an empty list would flow into a
    measurement and produce metrics over nobody.
    """
    from src.kg.data_loader import DiagnosisSample

    path = data_dir / f"{split}_samples.json"
    if not path.exists():
        raise FileNotFoundError(f"Samples file not found: {path}")

    return [
        DiagnosisSample(
            patient_id=entry["patient_id"],
            phenotype_ids=entry["phenotype_ids"],
            disease_id=entry["disease_id"],
        )
        for entry in json.loads(path.read_text())
    ]
