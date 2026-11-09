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

Serving (K8-2a) and measurement (`--mode disease`, K8-2b) are shown with a
trained task model: the session's disease child (`trained_disease_child`) over
its own workspace, a same-shaped other demo workspace, and the child with its
record removed.

Module: tests/unit/test_training_record.py
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.inference.pipeline import DiagnosisPipeline, PipelineBuildError  # noqa: E402
from src.kg.graph import KnowledgeGraph  # noqa: E402
from src.utils.fingerprint import (  # noqa: E402
    TrainingRecordMismatch,
    verify_training_record,
)
from tests.fixtures.synthetic_workspace import (  # noqa: E402,F401 (test_entry_documents reads it)
    build_workspace,
)
from tests.unit import test_serving_reads_once as reads_once  # noqa: E402

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
def _pre_contract(checkpoint, path):
    contents = torch.load(checkpoint, map_location="cpu", weights_only=False)
    del contents["training_input_digests"]
    torch.save(contents, path)
    return path


def _same_shaped_demo(root):
    """Workspace B for the disease child: the demo graph its workspace holds,
    with other bytes in `kg.json` (one node described) and `node_features.pt`
    (another feature seed), the same shapes, bound by its own manifest."""
    from scripts.setup_demo import build_demo_kg
    from src.kg.graph import DEFAULT_FEATURE_SEED
    from src.kg.workspace import SampleBudget, write_workspace

    kg = build_demo_kg()
    next(kg.iter_nodes()).description = "workspace b"
    write_workspace(kg, root, feature_dim=8, feature_seed=DEFAULT_FEATURE_SEED + 1,
                    samples=SampleBudget(num_train=20, num_val=5, val_disease_fraction=0.2))
    return root


def _serve(data_dir, checkpoint):
    return DiagnosisPipeline(
        graph=KnowledgeGraph.read_json(data_dir / "kg.json"),
        checkpoint_path=str(checkpoint),
        data_dir=str(data_dir),
        device="cpu",
    )


#: The disease child and its workspace, ``(data_dir, checkpoint)``. Read-only:
#: what a test writes goes to `tmp_path`. Shared with `test_serving_reads_once.py`.
child = reads_once.child


def test_serving_accepts_a_checkpoint_over_its_own_workspace(child):
    data_dir, checkpoint = child
    assert _serve(data_dir, checkpoint).get_pipeline_config()["gnn_ready"] is True


def test_serving_refuses_a_checkpoint_from_a_same_shaped_other_workspace(child, tmp_path,
                                                                        monkeypatch):
    _, checkpoint = child
    other = _same_shaped_demo(tmp_path / "b")
    import src.models.bundle.record as builder

    monkeypatch.setattr(builder, "load_bundle",
                        lambda *a, **k: pytest.fail("a model was built"))
    with pytest.raises(PipelineBuildError, match="does not match this workspace's graph") as refused:
        _serve(other, checkpoint)
    assert "kg differs" in str(refused.value)
    assert "node_features differs" in str(refused.value)
    assert isinstance(refused.value.__cause__, TrainingRecordMismatch)


def test_serving_refuses_a_checkpoint_from_before_the_contract(child, tmp_path):
    data_dir, checkpoint = child
    old = _pre_contract(checkpoint, tmp_path / "old.pt")
    with pytest.raises(PipelineBuildError, match="records no training_input_digests"):
        _serve(data_dir, old)


def _measure(data_dir, checkpoint, output):
    """The disease task on the generated `val` split: R3 is the same check for a
    registered version, whose own refusals are in
    `test_measure_scorer_registered.py`."""
    import scripts.measure_scorer as cli

    return cli.main([
        "--mode", "disease", "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
        "--split", "val", "--device", "cpu", "--output", str(output),
    ])


def test_measurement_accepts_a_checkpoint_over_its_own_workspace(child, tmp_path):
    data_dir, checkpoint = child
    output = tmp_path / "out" / "measurement.json"
    assert _measure(data_dir, checkpoint, output) == 0
    assert output.exists()


@pytest.mark.parametrize("case", ["other_workspace", "pre_contract"])
def test_measurement_refuses_before_building_or_writing(child, tmp_path, monkeypatch, case):
    """The pre-contract file is the child with its record removed: it keeps its
    model spec, and the record is compared before the spec is read."""
    data_dir, checkpoint = child
    if case == "other_workspace":
        data_dir, message = _same_shaped_demo(tmp_path / "b"), "kg differs"
    else:
        checkpoint = _pre_contract(checkpoint, tmp_path / "old.pt")
        message = "records no training_input_digests"
    import src.models.bundle.record as builder

    monkeypatch.setattr(builder, "load_bundle",
                        lambda *a, **k: pytest.fail("a model was built"))
    output = tmp_path / "out" / "measurement.json"
    with pytest.raises(TrainingRecordMismatch, match=message):
        _measure(data_dir, checkpoint, output)
    assert not output.exists() and not output.parent.exists()
