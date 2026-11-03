"""
The resume parent is checked before anything is restored or written (contract M2.3).
====================================================================================
At the real entry point (`train`, with `Trainer.train` stubbed), on disease
finetunes of a base trained on the demo workspace (`train_children`): the
parents are finetune checkpoints, which record every workspace role, the
samples among them (a base records the graph roles alone). The parent is
read once at the top of the run, beside the workspace checks, and compared with
this run's inputs through `verify_training_record` over every workspace role.
Each of these is refused before any state is restored, and leaves no run
directory, `config.yaml`, `runtime.json` or checkpoint behind:

- a parent from before the contract;
- a parent from a same-shaped other graph (a finetune of a base trained on it);
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
from tests.fixtures.generated_workspace import recut_cohorts  # noqa: E402
from tests.fixtures.trained_children import (  # noqa: E402,F401 - the demo and workspace fixtures
    demo_fixture,
    finetune_demo_base,
    train_children,
    workspace_fixture,
)


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


def _copy(source, root, **rows):
    """`source`'s graph files and base, its cohorts cut again (`rows`), a new
    manifest."""
    shutil.copytree(source, root)
    return recut_cohorts(root, **rows)


def _parent_of(train_model, captured, root, path, run):
    """A checkpoint written by a run over `root`, recording that run's inputs."""
    finetune_demo_base(train_model, root, run)
    trainer = captured.pop("trainer")
    trainer.state.epoch = 7
    trainer.save_checkpoint(path)
    return path


@pytest.fixture(scope="module")
def other_graph(demo, tmp_path_factory):
    """Workspace B: A's shapes and samples, other `kg.json` and feature values,
    and a base trained on it, since a finetune starts only from a base of its
    own knowledge graph. Shared, and only read."""
    other = tmp_path_factory.mktemp("b") / "data"
    shutil.copytree(demo, other, ignore=shutil.ignore_patterns("runs"))
    features = torch.load(other / "node_features.pt", weights_only=True)
    torch.save({k: v + 1.0 for k, v in features.items()}, other / "node_features.pt")
    kg = json.loads((other / "kg.json").read_text())
    kg["metadata"] = {**kg.get("metadata", {}), "workspace": "b"}
    (other / "kg.json").write_text(json.dumps(kg))
    recut_cohorts(other)
    assert (other / "train_samples.json").read_bytes() == (demo / "train_samples.json").read_bytes()
    train_children(other, tasks=())
    return other


def _no_run_artifacts(run):
    """What a run left in its run directory (contract M3a-2); a refused run
    creates none."""
    return sorted(p.name for p in run.iterdir()) if run.exists() else []


def _forbid_later_stages(monkeypatch, train_model):
    """The stages after the parent check: its run plan, the task data's binding
    and the Trainer."""
    for name in ("plan_from_parent", "bind_task_data", "Trainer"):
        def sentinel(*args, _name=name, **kwargs):
            raise AssertionError(f"{_name} ran after a refusal")

        monkeypatch.setattr(train_model, name, sentinel)


# ---------------------------------------------------------------------------
# The control
# ---------------------------------------------------------------------------
def test_a_parent_trained_on_these_inputs_resumes_and_is_recorded(
    train_model, captured, workspace, tmp_path
):
    parent = _parent_of(train_model, captured, workspace, tmp_path / "parent.pt", tmp_path / "seed")

    finetune_demo_base(train_model, workspace, tmp_path / "run", resume_from=str(parent))

    trainer = captured["trainer"]
    assert trainer.state.epoch == 7, "the parent's state was restored"
    assert trainer.training_input_digests["resume_checkpoint"] == _sha(parent)


def test_the_parent_read_is_released_before_training(
    train_model, captured, monkeypatch, workspace, tmp_path
):
    """The review's P2-2. The read made at the top of the run used to stay bound
    until training returned, keeping every source weight it parsed alive beside
    the model's own copies. By the time `Trainer.train` runs, none of the
    parent's model weights is reachable; what was restored is unchanged."""
    import weakref

    import src.utils.checkpoint_io as checkpoint_io
    from src.training import trainer as trainer_module

    parent = _parent_of(train_model, captured, workspace, tmp_path / "parent.pt", tmp_path / "seed")
    real, refs, seen = checkpoint_io.read_checkpoint, [], {}

    def recording(path, **kwargs):
        read = real(path, **kwargs)
        if Path(path) == parent:
            refs.extend(weakref.ref(t) for t in read.checkpoint["state_dict"].values())
        return read

    def train(self):
        seen["alive"] = sum(ref() is not None for ref in refs)
        seen["trainer"] = self
        return {"val_loss": 1.0}

    monkeypatch.setattr(checkpoint_io, "read_checkpoint", recording)
    monkeypatch.setattr(trainer_module.Trainer, "train", train)

    finetune_demo_base(train_model, workspace, tmp_path / "run", resume_from=str(parent))

    assert refs, "the parent's weights were not observed"
    assert seen["alive"] == 0, f"{seen['alive']} of {len(refs)} parent weights still alive"
    trainer = seen["trainer"]
    assert trainer.state.epoch == 7, "the parent's state was restored"
    assert trainer.training_input_digests["resume_checkpoint"] == _sha(parent)


def test_a_grandchild_resumes_too(train_model, captured, workspace, tmp_path):
    """A parent that was itself resumed records `resume_checkpoint`; that is its
    lineage, not a workspace input, and does not stop its own child."""
    first = _parent_of(train_model, captured, workspace, tmp_path / "first.pt", tmp_path / "seed")
    finetune_demo_base(train_model, workspace, tmp_path / "second", resume_from=str(first))
    second = tmp_path / "second.pt"
    captured.pop("trainer").save_checkpoint(second)

    finetune_demo_base(train_model, workspace, tmp_path / "run", resume_from=str(second))

    assert captured["trainer"].training_input_digests["resume_checkpoint"] == _sha(second)


# ---------------------------------------------------------------------------
# Refused before anything is restored or written
# ---------------------------------------------------------------------------
def _pre_contract(train_model, captured, ws, tmp_path, other_graph):
    parent = _parent_of(train_model, captured, ws, tmp_path / "p.pt", tmp_path / "seed")
    contents = torch.load(parent, map_location="cpu", weights_only=False)
    del contents["training_input_digests"]
    torch.save(contents, parent)
    return ws, parent, "records no training_input_digests", ()


def _other_graph(train_model, captured, ws, tmp_path, other_graph):
    """A finetune over workspace B (`other_graph`), of B's own base; its run
    directory is outside B, which it only reads."""
    parent = _parent_of(train_model, captured, other_graph, tmp_path / "p.pt", tmp_path / "seed")
    return ws, parent, "kg differs", ("node_features differs",)


def _other_samples(train_model, captured, ws, tmp_path, other_graph):
    """The same graph files, another cut of the cohorts: a training case less."""
    rows = json.loads((ws / "train_samples.json").read_text(encoding="utf-8"))
    other = _copy(ws, tmp_path / "c", train=rows[1:])
    assert (other / "node_features.pt").read_bytes() == (ws / "node_features.pt").read_bytes()
    parent = _parent_of(train_model, captured, other, tmp_path / "p.pt", tmp_path / "seed")
    return ws, parent, "train_samples differs", ()


def _validated_parent(train_model, captured, ws, tmp_path, other_graph):
    """The parent validated; this run's parsed `val_samples` are empty. (A
    disease finetune without validation would be refused later, by its task;
    the parent check comes first.)"""
    parent = _parent_of(train_model, captured, ws, tmp_path / "p.pt", tmp_path / "seed")
    child = _copy(ws, tmp_path / "d", val=[])
    return child, parent, "val_samples is recorded and this run has none", ()


def _missing(train_model, captured, ws, tmp_path, other_graph):
    return ws, tmp_path / "absent.pt", "does not train from scratch", ()


@pytest.mark.parametrize("make", [_pre_contract, _other_graph, _other_samples,
                                  _validated_parent, _missing],
                         ids=["pre-contract", "other-graph", "other-samples",
                              "validated-parent", "missing"])
def test_a_parent_that_does_not_qualify_is_refused_before_anything_exists(
    train_model, captured, monkeypatch, workspace, other_graph, tmp_path, make
):
    root, parent, message, also = make(train_model, captured, workspace, tmp_path, other_graph)
    restored = []
    from src.training import trainer as trainer_module

    monkeypatch.setattr(trainer_module.Trainer, "continue_from",
                        lambda self, checkpoint, *a, **k: restored.append(checkpoint))
    _forbid_later_stages(monkeypatch, train_model)
    run = tmp_path / "run"

    expected = FileNotFoundError if make is _missing else TrainingRecordMismatch
    with pytest.raises(expected, match=message) as refused:
        finetune_demo_base(train_model, root, run, resume_from=str(parent))

    for phrase in also:
        assert phrase in str(refused.value)
    assert restored == [] and "trainer" not in captured
    assert _no_run_artifacts(run) == []
