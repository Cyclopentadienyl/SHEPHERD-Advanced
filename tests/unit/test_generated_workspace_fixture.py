"""
The generated-workspace fixture writes files a parsing reader accepts.
======================================================================
Contract M2.1's readers parse every file whose digest a record names, so the
shared fixture's graph files are real (S2): written by the production writers,
loadable by the production readers, and different in every workspace, so that a
file copied from another workspace is caught as that workspace's file. Training's
rows carry both optional fields it reads.

Module: tests/unit/test_generated_workspace_fixture.py
"""
from __future__ import annotations

import json

import pytest

from tests.fixtures.generated_workspace import default_graph_export, write_generated_workspace

torch = pytest.importorskip("torch")

ROLES = ("kg", "node_features", "edge_indices", "num_nodes")


def _workspace(root, **kwargs):
    return write_generated_workspace(root, train_ids=[0, 1, 2], val_ids=[3], **kwargs)


def test_the_graph_files_load_through_the_production_readers(tmp_path):
    from src.kg import KnowledgeGraph
    from src.kg.storage.file_storage import read_graph_artifacts

    root, _ = _workspace(tmp_path / "ws")

    kg = KnowledgeGraph.load_json(str(root / "kg.json"))
    graph = read_graph_artifacts(root)

    assert kg.total_nodes == sum(graph["num_nodes_dict"].values())
    assert set(graph["x_dict"]) == set(graph["num_nodes_dict"])
    for node_type, features in graph["x_dict"].items():
        assert features.shape == (graph["num_nodes_dict"][node_type],
                                  default_graph_export()["feature_dim"])
    assert graph["edge_index_dict"], "the export has edges"


def test_every_graph_file_differs_between_two_workspaces(tmp_path):
    from src.kg.artifacts import GRAPH_ARTIFACTS

    one, manifest_one = _workspace(tmp_path / "one")
    two, manifest_two = _workspace(tmp_path / "two")

    for role in ROLES:
        filename = GRAPH_ARTIFACTS[role]
        assert (one / filename).read_bytes() != (two / filename).read_bytes(), role
        assert manifest_one["artifacts"][role] != manifest_two["artifacts"][role], role


def test_the_same_name_writes_the_same_files(tmp_path):
    """Drawn from the name, not at random, so a failure reproduces."""
    from src.kg.artifacts import GRAPH_ARTIFACTS

    one, _ = _workspace(tmp_path / "a" / "ws")
    two, _ = _workspace(tmp_path / "b" / "ws")

    for role in ROLES:
        filename = GRAPH_ARTIFACTS[role]
        assert (one / filename).read_bytes() == (two / filename).read_bytes(), role


def test_files_a_test_wrote_or_supplied_are_kept(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "kg.json").write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")

    _, manifest = _workspace(root, graph_bytes={"num_nodes": b'{"disease": 1}'})

    assert json.loads((root / "kg.json").read_text(encoding="utf-8")) == {"nodes": [], "edges": []}
    assert (root / "num_nodes.json").read_bytes() == b'{"disease": 1}'
    assert torch.load(root / "node_features.pt", weights_only=True)


def test_training_rows_carry_both_optional_fields(tmp_path):
    root, _ = _workspace(tmp_path / "ws", training_fields=True)

    for split in ("train", "val"):
        rows = json.loads((root / f"{split}_samples.json").read_text(encoding="utf-8"))
        assert rows
        for row in rows:
            assert row["gene_ids"]
            assert row["disease_id"] in row["candidate_disease_ids"]


def test_default_rows_carry_neither(tmp_path):
    root, _ = _workspace(tmp_path / "ws")

    rows = json.loads((root / "train_samples.json").read_text(encoding="utf-8"))
    assert all(set(row) == {"patient_id", "phenotype_ids", "disease_id"} for row in rows)
