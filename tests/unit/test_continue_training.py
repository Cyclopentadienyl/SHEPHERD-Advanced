"""A run continued from a parent checkpoint (contract M3a-3).

On a small controlled input, trained for real on the CPU through the production
entry point (`scripts/train_model.py::train`):

- a run paused after epoch 2 and resumed takes the same learning rate at every
  optimizer update as a run never interrupted, through warmup and into decay,
  with an accumulation that does not divide the batches, and ends with the same
  progress and optimizer step count;
- the pause writes a resumable snapshot even in an epoch without validation;
- a run stopped between epochs resumes from its last completed epoch;
- the child writes only into its own run directory and never deletes a
  parent's best-k file;
- what the parent's state does not allow is refused with the reason, and a
  schedule the parent's record cannot rebuild is refused before the run
  directory exists.

Not claimed: identical weights with and without a pause. Random state is not
restored (the plan, "Not claimed").

Module: tests/unit/test_continue_training.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.trainable_workspace import write_trainable_workspace  # noqa: E402

#: 12 training samples in batches of 5 give 3 batches an epoch; an
#: accumulation of 2 makes that 2 updates (one of them the tail), 8 over four
#: epochs. Three warmup steps, then cosine decay.
SETTINGS = dict(
    device="cpu", hidden_dim=16, num_layers=1, num_heads=2, num_workers=0,
    use_amp=False, num_epochs=4, batch_size=5, gradient_accumulation_steps=2,
    scheduler_type="cosine", warmup_steps=3, save_top_k=2, early_stopping_patience=10,
    seed=42,
)


@pytest.fixture(scope="module")
def train_model():
    import scripts.train_model as module

    return module


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return write_trainable_workspace(tmp_path_factory.mktemp("ws"))


@pytest.fixture
def rates(monkeypatch):
    """The learning rate each optimizer update used, per parameter group."""
    from src.training.trainer import Trainer

    seen = []
    original = Trainer._optimizer_update

    def recording(self):
        seen.append([float(g["lr"]) for g in self.optimizer.param_groups])
        return original(self)

    monkeypatch.setattr(Trainer, "_optimizer_update", recording)
    return seen


def _run(train_model, workspace, run, **overrides):
    config = train_model.TrainConfig(data_dir=str(workspace), run_dir=str(run),
                                     **{**SETTINGS, **overrides})
    explicit = {"data_dir", "run_dir", *overrides}
    return train_model.train(config, explicit)


def _read(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree(directory):
    return {str(p.relative_to(directory)): _sha(p)
            for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def _adam_step(checkpoint):
    return max(float(s["step"]) for s in checkpoint["optimizer_state_dict"]["state"].values())


# ---------------------------------------------------------------------------
# Pause and resume follow the uninterrupted schedule
# ---------------------------------------------------------------------------
def test_a_paused_and_resumed_run_takes_the_uninterrupted_rate_at_every_update(
    train_model, workspace, tmp_path, rates
):
    _run(train_model, workspace, tmp_path / "whole")
    whole_rates = list(rates)
    whole = _read(tmp_path / "whole" / "checkpoints" / "last.pt")
    rates.clear()

    _run(train_model, workspace, tmp_path / "first", pause_after_epoch=2)
    paused = _read(tmp_path / "first" / "checkpoints" / "last.pt")
    assert paused["progress"]["end"] == "paused"
    assert paused["progress"]["completed_epochs"] == 2
    assert paused["progress"]["stop_reason"] == "paused after epoch 2 (requested)"
    parent_files = _tree(tmp_path / "first")

    _run(train_model, workspace, tmp_path / "second",
         resume_from=str(tmp_path / "first" / "checkpoints" / "last.pt"))
    resumed = _read(tmp_path / "second" / "checkpoints" / "last.pt")

    # Warmup is 3 updates and the pause falls after update 4, so the resumed
    # half starts inside the decay. The schedule's length is 8 updates.
    assert len(whole_rates) == 8
    assert rates == whole_rates, "the resumed run left the uninterrupted schedule"
    assert whole_rates[0] < whole_rates[2] < whole_rates[3] and whole_rates[4] < whole_rates[3]

    for key in ("completed_epochs", "next_epoch", "optimizer_attempts", "optimizer_updates",
                "scheduler_steps", "end", "stage"):
        assert resumed["progress"][key] == whole["progress"][key], key
    assert _adam_step(resumed) == _adam_step(whole) == 8.0
    assert resumed["scheduler_state_dict"]["last_epoch"] == whole["scheduler_state_dict"]["last_epoch"]

    # Lineage, and the parent's files left exactly as they were.
    assert resumed["training_input_digests"]["resume_checkpoint"] == _sha(
        tmp_path / "first" / "checkpoints" / "last.pt")
    assert _tree(tmp_path / "first") == parent_files
    # The child inherited every learning setting: its record says what it ran with.
    inherited = {k: v for k, v in resumed["config"]["requested"].items()
                 if k not in ("pause_after_epoch", "run_name")}
    assert inherited == {k: v for k, v in whole["config"]["requested"].items()
                         if k not in ("pause_after_epoch", "run_name")}


def test_early_stopping_and_best_k_continue_without_touching_the_parent(
    train_model, workspace, tmp_path
):
    _run(train_model, workspace, tmp_path / "first", pause_after_epoch=2)
    parent = _read(tmp_path / "first" / "checkpoints" / "last.pt")
    parent_files = _tree(tmp_path / "first")

    # A top-k of one: the child's better epochs rank the parent's file out.
    _run(train_model, workspace, tmp_path / "second",
         resume_from=str(tmp_path / "first" / "checkpoints" / "last.pt"))
    child = _read(tmp_path / "second" / "checkpoints" / "last.pt")

    assert _tree(tmp_path / "first") == parent_files, "the child changed its parent's run"
    early_parent = parent["callbacks"]["early_stopping"]
    early_child = child["callbacks"]["early_stopping"]
    assert early_child["best_epoch"] >= early_parent["best_epoch"]
    # Every best-k entry the child lists resolves to bytes with its recorded digest,
    # its own files and its parent's alike.
    child_dir = tmp_path / "second" / "checkpoints"
    for entry in child["callbacks"]["model_checkpoint"]["best_k"]:
        assert _sha((child_dir / entry["file"]).resolve()) == entry["sha256"]
    source = child["callbacks"]["best_weights_source"]
    if source is not None and source["kind"] == "file":
        assert _sha((child_dir / source["file"]).resolve()) == source["sha256"]


def test_a_pause_in_an_epoch_without_validation_still_writes_the_snapshot(
    train_model, workspace, tmp_path
):
    _run(train_model, workspace, tmp_path / "first", eval_every_n_epochs=2,
         pause_after_epoch=1)
    last = _read(tmp_path / "first" / "checkpoints" / "last.pt")
    assert last["progress"]["completed_epochs"] == 1 and last["progress"]["end"] == "paused"
    assert not list((tmp_path / "first" / "checkpoints").glob("model-*.pt")), (
        "no validation ran, so no best-k file")

    _run(train_model, workspace, tmp_path / "second",
         resume_from=str(tmp_path / "first" / "checkpoints" / "last.pt"))
    assert _read(tmp_path / "second" / "checkpoints" / "last.pt")["progress"]["end"] == "finished"


def test_a_run_stopped_between_epochs_resumes_from_its_last_completed_epoch(
    train_model, workspace, tmp_path, monkeypatch
):
    """Stop (SIGINT) lands as KeyboardInterrupt inside an epoch; that epoch's
    progress is lost and the previous epoch's `last.pt` is the state to resume."""
    from src.training.trainer import Trainer

    original = Trainer._train_epoch

    def interrupted(self, epoch):
        if epoch == 2:
            raise KeyboardInterrupt
        return original(self, epoch)

    monkeypatch.setattr(Trainer, "_train_epoch", interrupted)
    _run(train_model, workspace, tmp_path / "first")
    monkeypatch.setattr(Trainer, "_train_epoch", original)

    last = _read(tmp_path / "first" / "checkpoints" / "last.pt")
    assert last["checkpoint_kind"] == "training_state"
    assert last["progress"]["completed_epochs"] == 2 and last["progress"]["end"] is None

    _run(train_model, workspace, tmp_path / "second",
         resume_from=str(tmp_path / "first" / "checkpoints" / "last.pt"))
    assert _read(tmp_path / "second" / "checkpoints" / "last.pt")["progress"][
        "completed_epochs"] == 4


def test_a_best_k_file_of_an_unfinished_run_is_resumed_from_its_own_epoch(
    train_model, workspace, tmp_path
):
    _run(train_model, workspace, tmp_path / "first", pause_after_epoch=3)
    best = sorted((tmp_path / "first" / "checkpoints").glob("model-*.pt"))[0]
    epoch = _read(best)["progress"]["completed_epochs"]

    _run(train_model, workspace, tmp_path / "second", resume_from=str(best))
    child = _read(tmp_path / "second" / "checkpoints" / "last.pt")
    assert child["progress"]["completed_epochs"] == 4
    assert child["progress"]["optimizer_updates"] == 8, (
        f"resumed from epoch {epoch}, the run took the remaining updates only")


# ---------------------------------------------------------------------------
# Refusals, from the parent's record
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def parent(train_model, workspace, tmp_path_factory):
    """A real paused checkpoint, read once; the refusal cases edit a copy."""
    run = tmp_path_factory.mktemp("parent") / "run"
    _run(train_model, workspace, run, pause_after_epoch=1)
    path = run / "checkpoints" / "last.pt"
    return path, _read(path)


def _plan(record, path, request, explicit=(), device="cpu"):
    from src.training.run_plan import plan_continuation

    return plan_continuation(record, Path(path), request, set(explicit), device=device)


def _request(record, **changes):
    return {**record["config"]["requested"], "run_dir": "", "resume_from": "x",
            "extend_epochs": None, "extend_lr": None, "pause_after_epoch": None, **changes}


def test_an_unfinished_parent_is_resumed_with_its_settings_inherited(parent):
    path, record = parent
    plan = _plan(record, path, _request(record, learning_rate=0.5, batch_size=99))

    assert plan.mode == "resume" and plan.start_epoch == 1 and plan.end_epoch == 4
    assert plan.action == "Resume the original training (1/4 done, 3 left)"
    assert plan.settings["learning_rate"] == record["config"]["requested"]["learning_rate"]
    assert plan.inherited["batch_size"] == 5
    assert "device" not in plan.inherited, "an operational field is never inherited"


def test_an_explicit_learning_setting_that_differs_is_refused_not_ignored(parent):
    from src.training.run_plan import RunPlanRefused

    path, record = parent
    with pytest.raises(RunPlanRefused, match=r"learning_rate: the parent's 0\.0001, requested 0\.5"):
        _plan(record, path, _request(record, learning_rate=0.5), explicit={"learning_rate"})
    # The same value, given explicitly, is not a conflict.
    _plan(record, path, _request(record), explicit={"learning_rate", "batch_size", "seed"})


@pytest.mark.parametrize("edit,message", [
    (lambda r: r["progress"].update(end="finished"), "nothing left to resume"),
    (lambda r: r["progress"].update(end="stopped_early", stop_reason="early stopping: x"),
     "stopped early"),
    (lambda r: r.pop("record_version"), "before contract M3a"),
    (lambda r: r.update(checkpoint_kind="weights", weights_from={"kind": "best", "epoch": 0}),
     "holds weights only"),
    (lambda r: r["config"].update(requested=None), "does not record the settings"),
    (lambda r: r["config"]["requested"].update(no_such_setting=1), "no_such_setting"),
    (lambda r: r["config"]["effective"]["amp"].update(enabled=True, dtype="float16"),
     "mixed precision on"),
], ids=["finished", "stopped-early", "pre-M3a", "weights-only", "no-settings",
        "unknown-setting", "amp-differs"])
def test_what_the_parents_state_does_not_allow_is_refused_with_the_reason(parent, edit, message):
    import copy

    from src.training.run_plan import RunPlanRefused

    path, record = parent
    record = copy.deepcopy(record)
    request = _request(record)
    edit(record)
    with pytest.raises(RunPlanRefused, match=message):
        _plan(record, path, request)


def test_extend_epochs_on_an_unfinished_parent_is_refused(parent):
    from src.training.run_plan import RunPlanRefused

    path, record = parent
    with pytest.raises(RunPlanRefused, match="applies only to a finished stage"):
        _plan(record, path, _request(record, extend_epochs=2))


def test_a_pause_at_or_before_the_start_is_refused(parent):
    from src.training.run_plan import RunPlanRefused

    path, record = parent
    with pytest.raises(RunPlanRefused, match="not after this run's start"):
        _plan(record, path, _request(record, pause_after_epoch=1))


def test_a_missing_or_replaced_best_weights_file_is_refused(parent, tmp_path, train_model):
    import copy
    import shutil

    from src.training.run_plan import RunPlanRefused
    from src.utils.checkpoint_io import read_checkpoint

    path, record = parent
    moved = tmp_path / "checkpoints"
    shutil.copytree(path.parent, moved)
    record = copy.deepcopy(record)
    record["callbacks"]["best_weights_source"] = {
        "kind": "file", "file": "model-best.pt", "sha256": "0" * 64}
    with pytest.raises(RunPlanRefused, match="model-best.pt.*not at"):
        _plan(record, moved / "last.pt", _request(record))

    shutil.copyfile(moved / "last.pt", moved / "model-best.pt")
    torch.save(record, moved / "edited.pt")
    with pytest.raises(RunPlanRefused, match="the file there now has"):
        train_model.plan_from_parent(
            read_checkpoint(moved / "edited.pt", map_location="cpu", weights_only=True),
            train_model.TrainConfig(**_request(record)), set(), "cpu")


def test_a_schedule_the_record_cannot_rebuild_is_refused_before_the_run_exists(
    train_model, workspace, parent, tmp_path
):
    import copy

    from src.training.run_plan import RunPlanRefused

    path, record = parent
    record = copy.deepcopy(record)
    record["progress"]["stage"]["schedule"]["total_steps"] = 999
    edited = tmp_path / "edited.pt"
    torch.save(record, edited)

    with pytest.raises(RunPlanRefused, match="total_steps differs"):
        _run(train_model, workspace, tmp_path / "child", resume_from=str(edited))
    assert not (tmp_path / "child").exists(), "the refused run created its directory"


def test_a_conflicting_cli_flag_is_refused_and_an_omitted_one_inherited(
    train_model, workspace, parent, tmp_path
):
    from src.training.run_plan import RunPlanRefused

    path, _ = parent
    with pytest.raises(RunPlanRefused, match="learning_rate"):
        _run(train_model, workspace, tmp_path / "child", resume_from=str(path),
             learning_rate=0.5)
    assert not (tmp_path / "child").exists()


def test_a_parents_best_k_file_ranked_out_leaves_the_list_and_stays_on_disk(tmp_path, monkeypatch):
    """Deterministic: the child's better epochs rank both of the parent's files
    out of a top-2. They leave the list; the bytes stay where the parent wrote
    them (contract M3a-2), and the child's own ranked-out file is deleted."""
    from src.training.callbacks import ModelCheckpoint, ModelCheckpointConfig

    parent_dir = tmp_path / "parent" / "checkpoints"
    parent_dir.mkdir(parents=True)
    entries = []
    for epoch, value in ((0, 0.10), (1, 0.20)):
        path = parent_dir / f"model-{epoch:02d}.pt"
        path.write_bytes(f"parent {epoch}".encode())
        entries.append({"path": path, "file": path.name, "value": value, "epoch": epoch,
                        "sha256": _sha(path)})

    checkpoint = ModelCheckpoint(ModelCheckpointConfig(
        dirpath=str(tmp_path / "child" / "checkpoints"), filename="model-{epoch:02d}",
        monitor="val_mrr", mode="max", save_top_k=2, save_last=False))
    checkpoint.on_train_begin(None)
    monkeypatch.setattr(checkpoint, "_save_checkpoint",
                        lambda trainer, path, epoch, logs: (path.write_bytes(b"child"), "d")[1])
    checkpoint.restore_foreign(entries)

    for epoch, value in ((2, 0.30), (3, 0.40), (4, 0.50)):
        checkpoint.on_epoch_end(None, epoch, {"val_mrr": value})

    assert [e["epoch"] for e in checkpoint.best_k_models] == [4, 3]
    assert sorted(p.name for p in parent_dir.iterdir()) == ["model-00.pt", "model-01.pt"]
    assert all(_sha(e["path"]) == e["sha256"] for e in entries)
    assert sorted(p.name for p in (tmp_path / "child" / "checkpoints").iterdir()) == [
        "model-03.pt", "model-04.pt"], "the child's own ranked-out file is pruned as before"
