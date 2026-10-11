"""
Training reads each recorded input once, and records those reads.
=================================================================
Contract M2.1, S8. Before the run directories, `config.yaml` and `runtime.json`,
`train` reads the manifest, the graph tensors and both sample files once each
(`read_training_inputs`, `training_fields=True`) and verifies from those reads,
with the `kg.json` identification check (decision 1). The resume parent is read
once with `read_checkpoint` and the trainer restores from the dict that read
parsed. `training_input_digests` is built only from the reads. The acceptance
shown here, at the real entry point (`train`, with `Trainer.train` stubbed):

- each input is opened once, and the record equals the digests of the files;
- a graph tensor, a sample file or the parent replaced after its read, by
  atomic rename or in-place rewrite, is the one trained on and recorded; a file
  replaced after the manifest read is refused, the refusal names it, and no run
  artifact and no later stage is reached;
- a republished manifest is refused before the file it covers is read, and is
  irrelevant after both reads;
- a file removed after its read is still recorded by the digest of what was
  read, never as `None`;
- with every path form raising and every path hash poisoned, the run completes
  and no poison reaches the record;
- a workspace with no validation samples records none; an unrelated split is
  never opened;
- a parent that does not parse leaves no checkpoint and no metrics;
- the path forms are gone, not kept beside the read forms.

Module: tests/unit/test_training_reads.py
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.generated_workspace import write_generated_workspace  # noqa: E402
from tests.fixtures.opens import count_opens  # noqa: E402
from tests.fixtures.path_forms import POISON, poison_path_forms  # noqa: E402
from tests.fixtures.replacement import HOWS, hook_read_once, replace_after_read  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDED = {
    "train_samples": "train_samples.json",
    "val_samples": "val_samples.json",
    "node_features": "node_features.pt",
    "edge_indices": "edge_indices.pt",
    "num_nodes": "num_nodes.json",
    "split_manifest": "split_manifest.json",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def train_model():
    import scripts.train_model as module

    return module


@pytest.fixture
def captured(monkeypatch):
    """`Trainer.train` replaced by a recorder: everything up to training runs."""
    from src.training import trainer as trainer_module

    seen = {}

    def train(self):
        seen["trainer"] = self
        return {"val_loss": 1.0}

    monkeypatch.setattr(trainer_module.Trainer, "train", train)
    return seen


def _workspace(root, **overrides):
    arguments = {"train_ids": [0, 1, 2], "val_ids": [3], "training_fields": True}
    arguments.update(overrides)
    return write_generated_workspace(root, **arguments)[0]


def _config(train_model, root, run, **overrides):
    return train_model.TrainConfig(
        data_dir=str(root), output_dir=str(run / "outputs"), log_dir=str(run / "logs"),
        checkpoint_dir=str(run / "checkpoints"), device="cpu", hidden_dim=16,
        num_layers=1, num_heads=2, num_workers=0, use_amp=False, **overrides,
    )


def _run_artifacts(run):
    return [name for name in ("outputs", "logs", "checkpoints") if (run / name).exists()]


@pytest.fixture
def workspace(tmp_path):
    return _workspace(tmp_path / "ws")


@pytest.fixture
def other(tmp_path):
    """Another workspace: the same layout, different bytes in every file."""
    return _workspace(tmp_path / "other", train_ids=[5, 6], val_ids=[7])


@pytest.fixture
def parents(train_model, captured, workspace, tmp_path):
    """Two checkpoints of this workspace's model: A at epoch 7, B at epoch 9 with
    every weight shifted."""
    train_model.train(_config(train_model, workspace, tmp_path / "seed"))
    trainer = captured.pop("trainer")
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    trainer.state.epoch = 7
    trainer.save_checkpoint(a)
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(1.0)
    trainer.state.epoch = 9
    trainer.save_checkpoint(b)
    return a, b


def _forbid_stages(monkeypatch, train_model):
    """Each stage after the preflight fails if it is reached."""
    for name in ("create_dataloaders", "create_model_from_config", "Trainer",
                 "resolve_resume_checkpoint"):
        def sentinel(*args, _name=name, **kwargs):
            raise AssertionError(f"{_name} ran after a refusal")

        monkeypatch.setattr(train_model, name, sentinel)


# ---------------------------------------------------------------------------
# One read each, and the record is those reads
# ---------------------------------------------------------------------------
def test_each_input_is_opened_once(train_model, captured, workspace, parents, tmp_path):
    parent, _ = parents

    with count_opens() as opens:
        train_model.train(_config(train_model, workspace, tmp_path / "run",
                                  resume_from=str(parent)))

    for name in (*RECORDED.values(), "kg.json"):
        assert opens.opens(workspace / name) == 1, name
    assert opens.opens(parent) == 1


def test_the_record_is_the_digests_of_the_bytes_read(
    train_model, captured, workspace, parents, tmp_path
):
    parent, _ = parents

    train_model.train(_config(train_model, workspace, tmp_path / "run",
                              resume_from=str(parent)))

    record = captured["trainer"].training_input_digests
    expected = {role: _sha(workspace / name) for role, name in RECORDED.items()}
    expected["resume_checkpoint"] = _sha(parent)
    assert record == expected


# ---------------------------------------------------------------------------
# Replacements: used and recorded, or refused
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("how", HOWS)
def test_a_graph_tensor_replaced_after_its_read_is_trained_on_and_recorded(
    train_model, captured, workspace, other, monkeypatch, tmp_path, how
):
    target = workspace / "node_features.pt"
    read = _sha(target)
    tensors = torch.load(target, weights_only=True)
    record = replace_after_read(monkeypatch, after=target, target=target,
                                data=(other / "node_features.pt").read_bytes(), how=how)

    train_model.train(_config(train_model, workspace, tmp_path / "run"))

    assert record.fired == 1
    trainer = captured["trainer"]
    assert trainer.training_input_digests["node_features"] == read != _sha(target)
    trained_on = trainer.train_dataloader.graph_data["x_dict"]
    for node_type, features in tensors.items():
        assert torch.equal(trained_on[node_type], features)


@pytest.mark.parametrize("how", HOWS)
def test_a_sample_file_replaced_after_its_read_is_trained_on_and_recorded(
    train_model, captured, workspace, other, monkeypatch, tmp_path, how
):
    target = workspace / "train_samples.json"
    read = _sha(target)
    patients = {row["patient_id"] for row in json.loads(target.read_text())}
    record = replace_after_read(monkeypatch, after=target, target=target,
                                data=(other / "train_samples.json").read_bytes(), how=how)

    train_model.train(_config(train_model, workspace, tmp_path / "run"))

    assert record.fired == 1
    trainer = captured["trainer"]
    assert trainer.training_input_digests["train_samples"] == read != _sha(target)
    trained_on = trainer.train_dataloader.dataset.samples
    assert {sample.patient_id for sample in trained_on} == patients
    assert {sample.disease_id for sample in trained_on} == {0, 1, 2}


@pytest.mark.parametrize("how", HOWS)
def test_the_parent_replaced_after_its_read_is_restored_and_recorded(
    train_model, captured, workspace, parents, monkeypatch, tmp_path, how
):
    parent, other_parent = parents
    read = _sha(parent)
    record = replace_after_read(monkeypatch, after=parent, target=parent,
                                data=other_parent.read_bytes(), how=how)

    train_model.train(_config(train_model, workspace, tmp_path / "run",
                              resume_from=str(parent)))

    assert record.fired == 1
    trainer = captured["trainer"]
    assert trainer.state.epoch == 7, "the state restored is A's, not B's"
    assert trainer.training_input_digests["resume_checkpoint"] == read != _sha(parent)


@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("name", ["node_features.pt", "edge_indices.pt", "train_samples.json",
                                  "val_samples.json", "kg.json"])
def test_a_file_replaced_after_the_manifest_read_is_refused_naming_it(
    train_model, workspace, other, monkeypatch, tmp_path, name, how
):
    _forbid_stages(monkeypatch, train_model)
    target = workspace / name
    run = tmp_path / "run"
    record = replace_after_read(monkeypatch, after=workspace / "split_manifest.json",
                                target=target, data=(other / name).read_bytes(), how=how)

    with pytest.raises(ValueError) as refused:
        train_model.train(_config(train_model, workspace, run))

    assert record.fired == 1
    assert str(target) in str(refused.value)
    assert "split_manifest.json" in str(refused.value)
    assert _run_artifacts(run) == [], "a run artifact was written before the refusal"


@pytest.mark.parametrize("how", HOWS)
def test_a_manifest_republished_before_the_tensor_read_is_refused(
    train_model, workspace, other, monkeypatch, tmp_path, how
):
    """The run's one reading of the manifest is A; the tensor read after the
    republication is B; the comparison is between those reads."""
    _forbid_stages(monkeypatch, train_model)
    target = workspace / "node_features.pt"
    record = replace_after_read(monkeypatch, after=workspace / "split_manifest.json",
                                target=target, data=(other / "node_features.pt").read_bytes(),
                                how=how, republish="node_features")

    with pytest.raises(ValueError, match="is not the node_features artifact"):
        train_model.train(_config(train_model, workspace, tmp_path / "run"))

    assert record.fired == 1


@pytest.mark.parametrize("how", HOWS)
def test_a_manifest_republished_after_both_reads_changes_nothing_recorded(
    train_model, captured, workspace, other, monkeypatch, tmp_path, how
):
    manifest = workspace / "split_manifest.json"
    target = workspace / "node_features.pt"
    read_manifest, read_tensor = _sha(manifest), _sha(target)
    record = replace_after_read(monkeypatch, after=target, target=target,
                                data=(other / "node_features.pt").read_bytes(), how=how,
                                republish="node_features")

    train_model.train(_config(train_model, workspace, tmp_path / "run"))

    assert record.fired == 1
    digests = captured["trainer"].training_input_digests
    assert digests["node_features"] == read_tensor
    assert digests["split_manifest"] == read_manifest != _sha(manifest)


@pytest.mark.parametrize("name", ["val_samples.json", "split_manifest.json", "parent"])
def test_a_file_removed_after_its_read_is_still_recorded_never_as_none(
    train_model, captured, workspace, parents, monkeypatch, tmp_path, name
):
    parent, _ = parents
    target = parent if name == "parent" else workspace / name
    read = _sha(target)

    def remove(path):
        if Path(path).resolve() == target.resolve():
            target.unlink()

    hook_read_once(monkeypatch, on_return=remove)

    train_model.train(_config(train_model, workspace, tmp_path / "run",
                              resume_from=str(parent)))

    assert not target.exists(), "the hook did not fire"
    digests = captured["trainer"].training_input_digests
    assert None not in digests.values()
    role = {"parent": "resume_checkpoint", "val_samples.json": "val_samples",
            "split_manifest.json": "split_manifest"}[name]
    assert digests[role] == read


# ---------------------------------------------------------------------------
# Only from the readers
# ---------------------------------------------------------------------------
def test_no_path_form_is_reached_and_no_path_digest_is_recorded(
    train_model, captured, workspace, parents, monkeypatch, tmp_path
):
    parent, _ = parents
    poison_path_forms(monkeypatch, true_digest_for=[workspace / "kg.json"])

    train_model.train(_config(train_model, workspace, tmp_path / "run",
                              resume_from=str(parent)))

    digests = captured["trainer"].training_input_digests
    assert POISON not in digests.values()
    assert digests["resume_checkpoint"] == _sha(parent)
    for role, name in RECORDED.items():
        assert digests[role] == _sha(workspace / name), role


def test_a_workspace_without_validation_samples_records_none(
    train_model, captured, tmp_path
):
    root = _workspace(tmp_path / "ws", val_ids=[])

    with count_opens() as opens:
        train_model.train(_config(train_model, root, tmp_path / "run"))

    trainer = captured["trainer"]
    assert trainer.val_dataloader is None
    assert "val_samples" not in trainer.training_input_digests
    assert opens.opens(root / "val_samples.json") == 1, "read and verified, once"


def test_an_unrelated_split_is_never_opened(train_model, captured, workspace, tmp_path):
    (workspace / "test_samples.json").write_text("[]")

    with count_opens() as opens:
        train_model.train(_config(train_model, workspace, tmp_path / "run"))

    assert opens.opens(workspace / "test_samples.json") == 0
    assert "test_samples" not in captured["trainer"].training_input_digests


def test_a_parent_that_does_not_parse_leaves_no_checkpoint_and_no_metrics(
    train_model, captured, workspace, tmp_path
):
    """The parent is read where it is restored (its move to the top is M2.3's), so
    the run directories and `config.yaml` exist by then; what a failed read must
    not leave is anything that looks like a result."""
    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not a checkpoint")
    run = tmp_path / "run"

    with pytest.raises(Exception):  # noqa: B017 — torch's own unpickling error
        train_model.train(_config(train_model, workspace, run, resume_from=str(junk)))

    assert "trainer" not in captured, "training started after a failed parent read"
    assert not (run / "outputs" / "final_metrics.json").exists()
    assert not list((run / "checkpoints").glob("*.pt"))


# ---------------------------------------------------------------------------
# The path forms are gone, not kept beside the read forms
# ---------------------------------------------------------------------------
def test_the_path_forms_are_gone_from_training():
    from src.core.protocols import TrainerProtocol
    from src.training.trainer import Trainer

    assert not hasattr(Trainer, "load_checkpoint")
    assert hasattr(TrainerProtocol, "restore_checkpoint")
    assert not hasattr(TrainerProtocol, "load_checkpoint")

    source = (REPO_ROOT / "scripts" / "train_model.py").read_text(encoding="utf-8")
    names = {node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(ast.parse(source))
              if isinstance(node, ast.Attribute)}
    names |= {alias.name for node in ast.walk(ast.parse(source))
              if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    gone = {"verify_graph_artifacts", "verify_generated_cohorts", "verify_graph_source",
            "compute_input_digests", "file_sha256", "load_graph_data", "load_samples",
            "training_input_roles", "load_checkpoint"}
    assert not names & gone, f"train_model.py still names {sorted(names & gone)}"
    assert "torch.load" not in source


def test_restore_checkpoint_reads_nothing_and_accepts_both_layouts(
    train_model, captured, workspace, tmp_path
):
    from src.training.trainer import Trainer

    train_model.train(_config(train_model, workspace, tmp_path / "seed"))
    trainer = captured.pop("trainer")
    path = tmp_path / "parent.pt"
    trainer.state.epoch = 4
    trainer.save_checkpoint(path)
    saved = torch.load(path, weights_only=False)
    weights = {name: value.clone() for name, value in saved["model_state_dict"].items()}

    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(1.0)
    trainer.state.epoch = 0

    with count_opens() as opens:
        trainer.restore_checkpoint(saved)

    assert sum(opens.counts.values()) == 0, "restore_checkpoint opened a file"
    assert trainer.state.epoch == 4
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value, weights[name])

    callback_layout = dict(saved)
    callback_layout["state_dict"] = callback_layout.pop("model_state_dict")
    trainer.restore_checkpoint(callback_layout)
    with pytest.raises(KeyError, match="neither 'model_state_dict' nor 'state_dict'"):
        trainer.restore_checkpoint({"optimizer_state_dict": saved["optimizer_state_dict"]})
    assert isinstance(trainer, Trainer)
