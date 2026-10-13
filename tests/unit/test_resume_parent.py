"""
The resume parent is checked before anything is restored or written (contract M2.3).
====================================================================================
At the real entry point (`train`, with `Trainer.train` stubbed). The parent is
read once at the top of the run, beside the workspace checks, and compared with
this run's inputs through `verify_training_record` over every workspace role.
Each of these is refused before any state is restored, and leaves no run
directory, `config.yaml`, `runtime.json` or checkpoint behind:

- a parent from before the contract;
- a parent from a same-shaped other graph;
- a parent that read other sample files;
- a parent that ran validation, where this run's parsed `val_samples` are empty;
- a requested parent that does not exist; no run trains from scratch instead.

And the control: a parent trained on these inputs resumes, and the child's
record names the bytes of that parent.

Module: tests/unit/test_resume_parent.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.utils.fingerprint import TrainingRecordMismatch  # noqa: E402
from tests.fixtures.generated_workspace import write_generated_workspace  # noqa: E402

IDS = {"train_ids": [0, 1, 2], "val_ids": [3]}


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


def _config(train_model, root, run, **overrides):
    return train_model.TrainConfig(
        data_dir=str(root), output_dir=str(run / "outputs"), log_dir=str(run / "logs"),
        checkpoint_dir=str(run / "checkpoints"), device="cpu", hidden_dim=16,
        num_layers=1, num_heads=2, num_workers=0, use_amp=False, **overrides,
    )


def _rebind(root, **ids):
    """Rewrite the cohorts and the manifest over the graph files already there."""
    write_generated_workspace(root, training_fields=True, **{**IDS, **ids})
    return root


def _copy(source, root, **ids):
    """`source`'s graph files, its cohorts cut again (`ids`), a new manifest."""
    shutil.copytree(source, root)
    (root / "split_manifest.json").unlink()
    return _rebind(root, **ids)


def _parent_of(train_model, captured, root, path, run):
    """A checkpoint written by a run over `root`, recording that run's inputs."""
    train_model.train(_config(train_model, root, run))
    trainer = captured.pop("trainer")
    trainer.state.epoch = 7
    trainer.save_checkpoint(path)
    return path


@pytest.fixture
def ws(tmp_path):
    return write_generated_workspace(tmp_path / "ws", training_fields=True, **IDS)[0]


def _no_run_artifacts(run):
    return [name for name in ("outputs", "logs", "checkpoints") if (run / name).exists()]


def _forbid_later_stages(monkeypatch, train_model):
    for name in ("create_dataloaders", "create_model_from_config", "Trainer"):
        def sentinel(*args, _name=name, **kwargs):
            raise AssertionError(f"{_name} ran after a refusal")

        monkeypatch.setattr(train_model, name, sentinel)


# ---------------------------------------------------------------------------
# The control
# ---------------------------------------------------------------------------
def test_a_parent_trained_on_these_inputs_resumes_and_is_recorded(
    train_model, captured, ws, tmp_path
):
    parent = _parent_of(train_model, captured, ws, tmp_path / "parent.pt", tmp_path / "seed")

    train_model.train(_config(train_model, ws, tmp_path / "run", resume_from=str(parent)))

    trainer = captured["trainer"]
    assert trainer.state.epoch == 7, "the parent's state was restored"
    assert trainer.training_input_digests["resume_checkpoint"] == _sha(parent)


def test_the_parent_read_is_released_before_training(
    train_model, captured, monkeypatch, ws, tmp_path
):
    """The review's P2-2. The read made at the top of the run used to stay bound
    until training returned, keeping every source weight it parsed alive beside
    the model's own copies. By the time `Trainer.train` runs, none of the
    parent's model weights is reachable; what was restored is unchanged."""
    import weakref

    import src.utils.checkpoint_io as checkpoint_io
    from src.training import trainer as trainer_module

    parent = _parent_of(train_model, captured, ws, tmp_path / "parent.pt", tmp_path / "seed")
    real, refs, seen = checkpoint_io.read_checkpoint, [], {}

    def recording(path, **kwargs):
        read = real(path, **kwargs)
        if Path(path) == parent:
            refs.extend(weakref.ref(t) for t in read.checkpoint["model_state_dict"].values())
        return read

    def train(self):
        seen["alive"] = sum(ref() is not None for ref in refs)
        seen["trainer"] = self
        return {"val_loss": 1.0}

    monkeypatch.setattr(checkpoint_io, "read_checkpoint", recording)
    monkeypatch.setattr(trainer_module.Trainer, "train", train)

    train_model.train(_config(train_model, ws, tmp_path / "run", resume_from=str(parent)))

    assert refs, "the parent's weights were not observed"
    assert seen["alive"] == 0, f"{seen['alive']} of {len(refs)} parent weights still alive"
    trainer = seen["trainer"]
    assert trainer.state.epoch == 7, "the parent's state was restored"
    assert trainer.training_input_digests["resume_checkpoint"] == _sha(parent)


def test_a_grandchild_resumes_too(train_model, captured, ws, tmp_path):
    """A parent that was itself resumed records `resume_checkpoint`; that is its
    lineage, not a workspace input, and does not stop its own child."""
    first = _parent_of(train_model, captured, ws, tmp_path / "first.pt", tmp_path / "seed")
    train_model.train(_config(train_model, ws, tmp_path / "second", resume_from=str(first)))
    second = tmp_path / "second.pt"
    captured.pop("trainer").save_checkpoint(second)

    train_model.train(_config(train_model, ws, tmp_path / "run", resume_from=str(second)))

    assert captured["trainer"].training_input_digests["resume_checkpoint"] == _sha(second)


# ---------------------------------------------------------------------------
# Refused before anything is restored or written
# ---------------------------------------------------------------------------
def _pre_contract(train_model, captured, ws, tmp_path):
    parent = _parent_of(train_model, captured, ws, tmp_path / "p.pt", tmp_path / "seed")
    contents = torch.load(parent, map_location="cpu", weights_only=False)
    del contents["training_input_digests"]
    torch.save(contents, parent)
    return ws, parent, "records no training_input_digests", ()


def _other_graph(train_model, captured, ws, tmp_path):
    """Workspace B: A's shapes and samples, other `kg.json` and feature values."""
    other = tmp_path / "b"
    shutil.copytree(ws, other)
    (other / "split_manifest.json").unlink()
    features = torch.load(other / "node_features.pt", weights_only=True)
    torch.save({k: v + 1.0 for k, v in features.items()}, other / "node_features.pt")
    kg = json.loads((other / "kg.json").read_text())
    kg["metadata"] = {**kg.get("metadata", {}), "workspace": "b"}
    (other / "kg.json").write_text(json.dumps(kg))
    _rebind(other)
    assert (other / "train_samples.json").read_bytes() == (ws / "train_samples.json").read_bytes()
    parent = _parent_of(train_model, captured, other, tmp_path / "p.pt", tmp_path / "seed")
    return ws, parent, "kg differs", ("node_features differs",)


def _other_samples(train_model, captured, ws, tmp_path):
    """The same graph files, another cut of the cohorts."""
    other = _copy(ws, tmp_path / "c", train_ids=[0, 1], val_ids=[3])
    assert (other / "node_features.pt").read_bytes() == (ws / "node_features.pt").read_bytes()
    parent = _parent_of(train_model, captured, other, tmp_path / "p.pt", tmp_path / "seed")
    return ws, parent, "train_samples differs", ()


def _validated_parent(train_model, captured, ws, tmp_path):
    """The parent validated; this run's parsed `val_samples` are empty."""
    parent = _parent_of(train_model, captured, ws, tmp_path / "p.pt", tmp_path / "seed")
    child = _copy(ws, tmp_path / "d", val_ids=[])
    return child, parent, "val_samples is recorded and this run has none", ()


def _missing(train_model, captured, ws, tmp_path):
    return ws, tmp_path / "absent.pt", "does not train from scratch", ()


@pytest.mark.parametrize("make", [_pre_contract, _other_graph, _other_samples,
                                  _validated_parent, _missing],
                         ids=["pre-contract", "other-graph", "other-samples",
                              "validated-parent", "missing"])
def test_a_parent_that_does_not_qualify_is_refused_before_anything_exists(
    train_model, captured, monkeypatch, ws, tmp_path, make
):
    root, parent, message, also = make(train_model, captured, ws, tmp_path)
    restored = []
    from src.training import trainer as trainer_module

    monkeypatch.setattr(trainer_module.Trainer, "restore_checkpoint",
                        lambda self, checkpoint: restored.append(checkpoint))
    _forbid_later_stages(monkeypatch, train_model)
    run = tmp_path / "run"

    expected = FileNotFoundError if make is _missing else TrainingRecordMismatch
    with pytest.raises(expected, match=message) as refused:
        train_model.train(_config(train_model, root, run, resume_from=str(parent)))

    for phrase in also:
        assert phrase in str(refused.value)
    assert restored == [] and "trainer" not in captured
    assert _no_run_artifacts(run) == []
