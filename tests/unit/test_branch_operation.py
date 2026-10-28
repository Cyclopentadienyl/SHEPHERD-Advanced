"""The branch operation (README §1; contract §9.2; BACKLOG 23), and the
settings record it shares with resume.

A branch is a new run of its parent's stage and task from its parent's
weights, loaded strictly, with a new optimizer and stage. It keeps every
setting of its parent but those the request changes on the stage's closed
list (`src.config.recipe.branch_settings`), and records each change beside
the exact parent file (`parent.changes`). It is refused before anything is
written when it names two parents, another stage or task, a setting off the
list, an extension, no change, a weights-only parent, other bytes than
expected or other data than its parent's. A paused branch resumes as its own
run.

With it, three shared defects are fixed at their root: early stopping follows
the patience in effect, so a resumed or branched run keeps it on or off as
its parent had it; a continued run's recipe record names its settings as
inherited, never as the recipe's; and the base's validation pairs are the
split's, whatever a run's seed or batch size. The configuration file is read
strictly: an unknown key, a missing or unreadable path or a document that is
not a mapping is refused; a pipe is read like a file.

Module: tests/unit/test_branch_operation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from src.models.bundle.record import load_bundle  # noqa: E402
from src.utils.run_plan import RunPlanRefused  # noqa: E402
from tests.fixtures.base_workspace import write_base_workspace  # noqa: E402
from tests.unit.test_train_base_stage import _config, _small_spec  # noqa: E402


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return write_base_workspace(tmp_path_factory.mktemp("branch"))


@pytest.fixture(scope="module")
def parent(workspace, tmp_path_factory):
    """A finished two-epoch base: the recipe's rate (1e-4) and seed (1), batch 16."""
    import train_model

    root = tmp_path_factory.mktemp("parent")
    train_model.train(*_config(workspace, root / "base"))
    return root / "base" / "checkpoints" / "last.pt"


def _branch(workspace, run_dir, parent, **changes):
    import train_model

    fields = {"data_dir": str(workspace), "run_dir": str(run_dir), "learning_stage": "base",
              "branch_from": str(parent), "device": "cpu", **changes}
    explicit = {name for name, value in fields.items() if value is not None}
    return train_model.TrainConfig(**fields), explicit - {"data_dir", "run_dir"}


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def test_a_branch_trains_from_its_parents_weights_with_its_changes(workspace, parent, tmp_path,
                                                                    monkeypatch):
    import src.training.lp_task as lp_task
    import train_model
    from src.training.trainer import Trainer
    from src.utils.fingerprint import file_sha256

    before = _load(parent)
    seen, validation = {}, []
    real_train, real_task = Trainer.train, lp_task.BaseLPTask

    def train_checked(self):
        seen["weights"] = all(torch.equal(value, before["state_dict"][name])
                              for name, value in self.model.state_dict().items())
        seen["optimizer_state"] = dict(self.optimizer.state_dict()["state"])
        return real_train(self)

    def task_seen(**kwargs):
        validation.append((kwargs["seed"], kwargs["val_batch_size"]))
        return real_task(**kwargs)

    monkeypatch.setattr(Trainer, "train", train_checked)
    monkeypatch.setattr(lp_task, "BaseLPTask", task_seen)
    train_model.train(*_branch(workspace, tmp_path / "branch", parent, learning_rate=1e-3,
                               seed=5, num_epochs=1, batch_size=8))

    last = _load(tmp_path / "branch" / "checkpoints" / "last.pt")
    _, record = load_bundle(last, learning_stage="base")
    assert seen == {"weights": True, "optimizer_state": {}}, "the parent's weights, a new optimizer"
    assert (record.operation, record.family_id) == ("branch", "base")
    assert record.parent["sha256"] == file_sha256(parent) and record.parent["run"] == "base"
    assert record.parent["changes"] == {
        "learning_rate": {"parent": 1e-4, "new": 1e-3}, "seed": {"parent": 1, "new": 5},
        "num_epochs": {"parent": 2, "new": 1}, "batch_size": {"parent": 16, "new": 8}}
    assert record.weight_transfer["initialised_new"] == [] and record.weight_transfer[
        "optimizer"] == "new"
    assert last["progress"]["completed_epochs"] == 1
    assert last["progress"]["stage"]["kind"] == "original"
    requested, parents = last["config"]["requested"], before["config"]["requested"]
    assert requested["learning_rate"] == 1e-3 and requested["model_spec"] == parents["model_spec"]
    assert requested["num_neighbors"] == parents["num_neighbors"]
    recipe = last["config"]["effective"]["loaders"]["train"]["recipe"]
    assert recipe["reference_filled"] == {}
    assert set(recipe["operator_set"]) == {"learning_rate", "seed", "num_epochs", "batch_size"}
    assert "weight_decay" in recipe["inherited_from_parent"]
    # Validated on the parent's pairs, whatever its seed and batch size.
    assert validation == [(0, 512)]
    assert last["config"]["effective"]["loaders"]["val"] == (
        before["config"]["effective"]["loaders"]["val"])
    assert "resume_checkpoint" not in last["training_input_digests"]


def _other_spec():
    spec = dict(_small_spec())
    spec["encoder"] = {**spec["encoder"], "hidden": 8}
    return spec


#: Each refused request: (changes, error, what the refusal says).
REFUSALS = {
    "two parents": (dict(resume_from="PARENT", learning_rate=1e-3), ValueError,
                    "resume_from and branch_from name two parents"),
    "another stage": (dict(learning_stage="finetune", task="disease", learning_rate=1e-3),
                      RunPlanRefused, "a branch keeps its parent's stage and task"),
    "the split": (dict(split_seed=3, learning_rate=1e-3), RunPlanRefused,
                  "keeps every setting of its parent but those on its list"),
    "the spec": (dict(model_spec=_other_spec(), learning_rate=1e-3), RunPlanRefused,
                 "keeps every setting of its parent but those on its list"),
    "early stopping": (dict(early_stopping_patience=3), RunPlanRefused,
                       "keeps every setting of its parent but those on its list"),
    "an extension": (dict(extend_epochs=1, learning_rate=1e-3), RunPlanRefused,
                     "a branch starts a new run: remove them to branch"),
    "no change": (dict(learning_rate=1e-4), RunPlanRefused, "changes at least one setting"),
    "the ShepherdGNN path": (dict(learning_stage=None, learning_rate=1e-3), ValueError,
                             "belong to the model bundle path"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_a_refused_branch_writes_nothing(workspace, parent, tmp_path, case):
    import train_model

    changes, error, fragment = REFUSALS[case]
    changes = {k: (str(parent) if v == "PARENT" else v) for k, v in changes.items()}
    with pytest.raises(error, match=fragment):
        train_model.train(*_branch(workspace, tmp_path / "x", parent, **changes))
    assert not (tmp_path / "x").exists()


def test_a_parent_without_its_settings_or_its_bytes_is_refused(workspace, parent, tmp_path):
    import train_model
    from src.utils.checkpoint_io import CheckpointChanged

    weights_only = {k: v for k, v in _load(parent).items()
                    if k not in ("config", "optimizer_state_dict", "scheduler_state_dict")}
    weights_only["checkpoint_kind"] = "weights"
    torch.save(weights_only, tmp_path / "weights.pt")
    old = {**_load(parent), "record_version": 1}
    torch.save(old, tmp_path / "old.pt")
    for path, error, fragment in (
            (tmp_path / "weights.pt", RunPlanRefused, "holds weights only"),
            (tmp_path / "old.pt", RunPlanRefused, "record version")):
        with pytest.raises(error, match=fragment):
            train_model.train(*_branch(workspace, tmp_path / "x", path, learning_rate=1e-3))
    with pytest.raises(CheckpointChanged):
        train_model.train(*_branch(workspace, tmp_path / "x", parent, learning_rate=1e-3),
                          expected_parent_sha256="0" * 64)
    assert not (tmp_path / "x").exists()


def test_a_branch_over_other_data_is_refused(parent, tmp_path):
    """The branch's data is its parent's (BACKLOG 23, M2.3's rule): another
    workspace's graph is refused before anything is written."""
    import train_model
    from src.utils.fingerprint import TrainingRecordMismatch

    other = write_base_workspace(tmp_path / "other", seed=8)
    with pytest.raises(TrainingRecordMismatch,
                       match="The branch parent .* does not match this run: .*kg differs"):
        train_model.train(*_branch(other, tmp_path / "x", parent, learning_rate=1e-3))
    assert not (tmp_path / "x").exists()


def test_a_paused_branch_resumes_as_its_own_run(workspace, parent, tmp_path):
    """And a resume names its settings as inherited, never as the recipe's."""
    import train_model

    train_model.train(*_branch(workspace, tmp_path / "b", parent, learning_rate=1e-3,
                               num_epochs=2, pause_after_epoch=1))
    branched = tmp_path / "b" / "checkpoints" / "last.pt"
    assert _load(branched)["progress"]["end"] == "paused"

    def resume(run, **kw):
        fields = dict(data_dir=str(workspace), run_dir=str(tmp_path / run),
                      learning_stage="base", resume_from=str(branched), device="cpu", **kw)
        return train_model.TrainConfig(**fields), set(fields) - {"data_dir", "run_dir"}

    train_model.train(*resume("r"))
    child = _load(tmp_path / "r" / "checkpoints" / "last.pt")
    _, record = load_bundle(child, learning_stage="base")
    assert (record.operation, record.family_id, record.parent["run"]) == ("resume", "base", "b")
    assert "changes" not in record.parent
    assert child["config"]["requested"]["learning_rate"] == 1e-3
    assert child["progress"]["completed_epochs"] == 2
    recipe = child["config"]["effective"]["loaders"]["train"]["recipe"]
    assert recipe["reference_filled"] == {} and "learning_rate" in recipe["inherited_from_parent"]
    with pytest.raises(RunPlanRefused, match="branch the parent"):
        train_model.train(*resume("s", learning_rate=1e-4))
    assert not (tmp_path / "s").exists()


def test_early_stopping_follows_the_patience_in_effect(workspace, tmp_path):
    """A resumed and a branched run keep their parent's early stopping."""
    import train_model

    train_model.train(*_config(workspace, tmp_path / "es", num_epochs=3, pause_after_epoch=1,
                               early_stopping_patience=5))
    paused = tmp_path / "es" / "checkpoints" / "last.pt"
    assert _load(paused)["config"]["early_stopping"] is True
    resumed = train_model.TrainConfig(data_dir=str(workspace), run_dir=str(tmp_path / "r"),
                                      learning_stage="base", resume_from=str(paused),
                                      device="cpu")
    train_model.train(resumed, {"learning_stage", "resume_from", "device"})
    train_model.train(*_branch(workspace, tmp_path / "b", paused, learning_rate=1e-3,
                               num_epochs=1))
    for run in ("r", "b"):
        child = _load(tmp_path / run / "checkpoints" / "last.pt")
        assert child["config"]["early_stopping"] is True, run
        assert child["config"]["requested"]["early_stopping_patience"] == 5, run
    # The recipe's default: no patience, early stopping off.
    train_model.train(*_config(workspace, tmp_path / "off", num_epochs=1))
    off = _load(tmp_path / "off" / "checkpoints" / "last.pt")
    assert off["config"]["early_stopping"] is False
    assert off["config"]["requested"]["early_stopping_patience"] is None


def test_a_disease_child_branches_within_its_task(tmp_path):
    import train_model
    from tests.unit.test_disease_finetune import _base_run, _finetune_config

    workspace = write_base_workspace(tmp_path / "ws")
    base = _base_run(workspace, tmp_path / "base")
    train_model.train(*_finetune_config(workspace, tmp_path / "child", base, num_epochs=1))
    child = tmp_path / "child" / "checkpoints" / "last.pt"
    fields = dict(data_dir=str(workspace), run_dir=str(tmp_path / "branch"),
                  learning_stage="finetune", task="disease", branch_from=str(child),
                  task_loss_weight=0.5, device="cpu")
    train_model.train(train_model.TrainConfig(**fields), set(fields) - {"data_dir", "run_dir"})
    last = _load(tmp_path / "branch" / "checkpoints" / "last.pt")
    _, record = load_bundle(last, task="disease")
    assert (record.operation, record.family_id) == ("branch", "base")
    assert record.parent["changes"] == {"task_loss_weight": {"parent": 0.1, "new": 0.5}}
    assert record.loss_identity["task_loss_weight"] == 0.5
    assert record.weight_transfer["initialised_new"] == []


# ---------------------------------------------------------------------------
# The configuration file
# ---------------------------------------------------------------------------
def test_the_configuration_file_is_read_strictly(tmp_path, monkeypatch):
    import train_model

    def args(*argv):
        monkeypatch.setattr(sys, "argv", ["train_model.py", *argv])
        return train_model.parse_args()

    files = {"typo.yaml": "learning_stage: base\nbranch_frm: parent.pt\n", "list.yaml": "- a\n",
             "good.yaml": "learning_stage: base\nnum_epochs: 3\n", "empty.yaml": ""}
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    for name, fragment in (("typo.yaml", "does not have: branch_frm"),
                           ("list.yaml", "is not a mapping"), ("none.yaml", "does not exist")):
        with pytest.raises(ValueError, match=fragment):
            train_model.load_config_with_explicit(args("--config", str(tmp_path / name)))
    assert train_model._train_main(args("--config", str(tmp_path / "typo.yaml"))) == 1
    config, explicit = train_model.load_config_with_explicit(
        args("--config", str(tmp_path / "good.yaml"), "--branch-from", "parent.pt"))
    assert (config.num_epochs, config.branch_from) == (3, "parent.pt")
    assert {"learning_stage", "num_epochs", "branch_from"} <= explicit
    config, explicit = train_model.load_config_with_explicit(
        args("--config", str(tmp_path / "empty.yaml")))
    assert explicit == set()
    # A pipe is read like a file; a path that cannot be read says so.
    import os
    import threading

    pipe = tmp_path / "config.fifo"
    os.mkfifo(pipe)
    writer = threading.Thread(target=pipe.write_text, args=("num_epochs: 4\n",))
    writer.start()
    config, explicit = train_model.load_config_with_explicit(args("--config", str(pipe)))
    writer.join(timeout=10)
    assert (config.num_epochs, explicit) == (4, {"num_epochs"})
    with pytest.raises(ValueError, match="cannot be read .IsADirectoryError"):
        train_model.load_config_with_explicit(args("--config", str(tmp_path)))


def test_the_validation_pairs_are_the_splits(workspace):
    """Two tasks over one split draw the same pairs; another split seed other ones."""
    from src.kg.graph_view import build_graph_view, split_facts
    from src.kg.storage.file_storage import read_graph_artifacts
    from src.training.lp_task import BaseLPTask

    view = build_graph_view(read_graph_artifacts(workspace, map_location="cpu").graph_data)
    split = split_facts(view, 0)
    train, val = split.mask("train"), split.mask("val")

    def pairs(seed):
        task = BaseLPTask(num_nodes=view.num_nodes, train_edges=view.edge_index(train),
                          train_relations=view.rel[train], val_edges=view.edge_index(val),
                          val_relations=view.rel[val], seed=seed, val_batch_size=512)
        return task.val_head, task.val_tail, task.val_neg_tail

    assert all(torch.equal(a, b) for a, b in zip(pairs(0), pairs(0)))
    assert not all(torch.equal(a, b) for a, b in zip(pairs(0), pairs(1)))
