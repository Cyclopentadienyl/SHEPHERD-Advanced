"""
A checkpoint is used only over the graph it was trained on (contract M2.4).
==========================================================================
`verify_training_record` is the one model-to-graph comparator: serving and
measurement compare a checkpoint's recorded `kg`, `node_features`,
`edge_indices` and `num_nodes` with the digests the workspace check verified,
and resume compares a parent with its run (M2.3). Shown here:

- the rule: every role present on both sides and equal, or refused by name; a
  checkpoint with no record, or an empty one, is refused, and so is an empty
  role set, which would pass anything;
- serving and measurement refuse a checkpoint from a same-shaped other
  workspace, and one from before the contract, before any model is built; the
  control, the checkpoint over its own workspace, is accepted by both.

Module: tests/unit/test_training_record.py
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from src.inference.pipeline import DiagnosisPipeline, PipelineBuildError  # noqa: E402
from src.kg.graph import KnowledgeGraph  # noqa: E402
from src.utils.fingerprint import (  # noqa: E402
    TrainingRecordMismatch,
    verify_training_record,
)
from tests.fixtures.generated_workspace import write_generated_workspace  # noqa: E402
from tests.fixtures.synthetic_workspace import N_DISEASES, build_workspace  # noqa: E402

GRAPH_ROLES = ("kg", "node_features", "edge_indices", "num_nodes")
CURRENT = {role: f"{role}-digest" for role in GRAPH_ROLES}


def _verify(record, current=CURRENT, roles=GRAPH_ROLES):
    checkpoint = {} if record is None else {"training_input_digests": record}
    verify_training_record(
        checkpoint, current, roles, recorded_by="checkpoint X", compared_with="workspace Y"
    )


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------
def test_the_same_digests_for_every_role_pass():
    _verify(dict(CURRENT, split_manifest="not compared"))


@pytest.mark.parametrize("record", [None, {}, "not a map"])
def test_a_checkpoint_with_no_record_is_refused(record):
    with pytest.raises(TrainingRecordMismatch, match="records no training_input_digests"):
        _verify(record)


def test_a_role_not_recorded_is_refused_by_name():
    record = {role: digest for role, digest in CURRENT.items() if role != "kg"}
    with pytest.raises(TrainingRecordMismatch, match="kg is not recorded"):
        _verify(record)


def test_a_role_the_current_side_lacks_is_refused_by_name():
    current = {role: digest for role, digest in CURRENT.items() if role != "num_nodes"}
    with pytest.raises(TrainingRecordMismatch, match="num_nodes is recorded and workspace Y has none"):
        _verify(dict(CURRENT), current=current)


def test_a_different_digest_is_refused_and_every_difference_named():
    record = dict(CURRENT, kg="another graph", edge_indices="other edges")
    with pytest.raises(TrainingRecordMismatch) as refused:
        _verify(record)
    assert "kg differs" in str(refused.value) and "edge_indices differs" in str(refused.value)
    assert "node_features" not in str(refused.value)


def test_an_empty_role_set_is_a_caller_error_not_a_pass():
    with pytest.raises(ValueError, match="no roles"):
        _verify(dict(CURRENT), roles=())


def test_a_checkpoint_that_is_not_a_mapping_is_refused():
    with pytest.raises(TrainingRecordMismatch, match="records no training_input_digests"):
        verify_training_record(
            torch.zeros(1), CURRENT, GRAPH_ROLES, recorded_by="x", compared_with="y"
        )


# ---------------------------------------------------------------------------
# Serving and measurement
# ---------------------------------------------------------------------------
def _same_shaped_other(root):
    """Workspace B: the shapes of A's graph, other bytes in `kg.json` and
    `node_features.pt`, bound by its own manifest."""
    data_dir, _ = build_workspace(root)
    features = torch.load(data_dir / "node_features.pt", weights_only=True)
    torch.save({k: v + 1.0 for k, v in features.items()}, data_dir / "node_features.pt")
    (data_dir / "kg.json").write_text(json.dumps({"synthetic": True, "workspace": "b"}))
    write_generated_workspace(data_dir, train_ids=[0, 1], val_ids=[N_DISEASES + 1])
    return data_dir


def _pre_contract(checkpoint, path):
    contents = torch.load(checkpoint, map_location="cpu", weights_only=False)
    del contents["training_input_digests"]
    torch.save(contents, path)
    return path


def _serve(data_dir, checkpoint):
    return DiagnosisPipeline(
        graph=KnowledgeGraph.read_json(data_dir / "kg.json"),
        checkpoint_path=str(checkpoint),
        data_dir=str(data_dir),
        device="cpu",
    )


@pytest.fixture
def a(tmp_path):
    return build_workspace(tmp_path / "a")


def test_serving_accepts_a_checkpoint_over_its_own_workspace(a):
    data_dir, checkpoint = a
    assert _serve(data_dir, checkpoint).get_pipeline_config()["gnn_ready"] is True


def test_serving_refuses_a_checkpoint_from_a_same_shaped_other_workspace(a, tmp_path, monkeypatch):
    _, checkpoint = a
    other = _same_shaped_other(tmp_path / "b")
    import src.models.gnn.shepherd_gnn as builder

    monkeypatch.setattr(builder, "build_shepherd_model",
                        lambda *a, **k: pytest.fail("a model was built"))
    with pytest.raises(PipelineBuildError, match="does not match this workspace's graph") as refused:
        _serve(other, checkpoint)
    assert "kg differs" in str(refused.value)
    assert "node_features differs" in str(refused.value)
    assert isinstance(refused.value.__cause__, TrainingRecordMismatch)


def test_serving_refuses_a_checkpoint_from_before_the_contract(a, tmp_path):
    data_dir, checkpoint = a
    old = _pre_contract(checkpoint, tmp_path / "old.pt")
    with pytest.raises(PipelineBuildError, match="records no training_input_digests"):
        _serve(data_dir, old)


def _measure(data_dir, checkpoint, output):
    import scripts.measure_scorer as cli

    return cli.main([
        "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
        "--split", "test", "--cohort-kind", "supplied", "--device", "cpu",
        "--output", str(output),
    ])


def test_measurement_accepts_a_checkpoint_over_its_own_workspace(a, tmp_path):
    data_dir, checkpoint = a
    output = tmp_path / "out" / "measurement.json"
    assert _measure(data_dir, checkpoint, output) == 0
    assert output.exists()


@pytest.mark.parametrize("case", ["other_workspace", "pre_contract"])
def test_measurement_refuses_before_building_or_writing(a, tmp_path, monkeypatch, case):
    data_dir, checkpoint = a
    if case == "other_workspace":
        data_dir, message = _same_shaped_other(tmp_path / "b"), "kg differs"
    else:
        checkpoint = _pre_contract(checkpoint, tmp_path / "old.pt")
        message = "records no training_input_digests"
    import src.models.gnn.shepherd_gnn as builder

    monkeypatch.setattr(builder, "build_shepherd_model",
                        lambda *a, **k: pytest.fail("a model was built"))
    output = tmp_path / "out" / "measurement.json"
    with pytest.raises(TrainingRecordMismatch, match=message):
        _measure(data_dir, checkpoint, output)
    assert not output.exists() and not output.parent.exists()
