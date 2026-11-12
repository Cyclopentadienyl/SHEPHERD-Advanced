"""`scripts/train_model.py --stage base` end to end on a small workspace.

The base stage reads the graph alone, forms the view and its split before any
output, trains the bundle through the Trainer with early stopping off, writes
bundle checkpoints with the model record and the split artifact, pauses and
resumes strictly (regression 8), and refuses before writing: ShepherdGNN-only
settings, a parent written before the model spec, a different split.

Module: tests/unit/test_train_base_stage.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from src.config.model_types import WRITTEN_BEFORE_SPEC  # noqa: E402
from src.models.bundle import reference_spec  # noqa: E402
from src.models.bundle.record import MODEL_RECORD_VERSION, load_bundle  # noqa: E402
from src.models.bundle.spec import InputSpec  # noqa: E402
from tests.fixtures.base_workspace import write_base_workspace  # noqa: E402


def _small_spec():
    spec = reference_spec("base")
    return dataclasses.replace(
        spec, input=InputSpec(dim=8),
        encoder=dataclasses.replace(spec.encoder, layers=2, hidden=4, output=4, heads=2)
    ).validate().to_dict()


def _config(data_dir, run_dir, **kw):
    import train_model

    fields = dict(data_dir=str(data_dir), run_dir=str(run_dir), learning_stage="base",
                  model_spec=_small_spec(), num_epochs=2, batch_size=16, num_neighbors=[4, 4],
                  device="cpu", save_top_k=1)
    fields.update(kw)
    explicit = set(fields) - {"data_dir", "run_dir"}
    return train_model.TrainConfig(**fields), explicit


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return write_base_workspace(tmp_path_factory.mktemp("base"))


def test_base_run_writes_bundle_checkpoints(workspace, tmp_path):
    import train_model

    config, explicit = _config(workspace, tmp_path / "run")
    claimed = []
    metrics = train_model.train(config, explicit, on_claim=claimed.append)
    assert "val_lp_accuracy" in metrics
    run = tmp_path / "run"
    # The run directory, its first artifact, is reported once claimed: the
    # boundary the command line tells a refusal from a failed run by.
    assert claimed == [run]
    for name in ("config.yaml", "graph_view.json", "graph_split.pt", "final_metrics.json"):
        assert (run / name).is_file(), name
    last = torch.load(run / "checkpoints" / "last.pt", map_location="cpu", weights_only=True)
    assert last["record_version"] == MODEL_RECORD_VERSION
    bundle, record = load_bundle(last, learning_stage="base")
    split = torch.load(run / "graph_split.pt", weights_only=True)
    assert record.data_roles["split"]["digest"] == split["record"]["digest"]
    assert record.operation == "new" and record.family_id == "run"
    digests = last["training_input_digests"]
    assert "train_samples" not in digests and "kg" in digests
    import json

    view_record = json.loads((run / "graph_view.json").read_text())
    assert view_record["source"]["kg"] == digests["kg"]
    assert view_record["digest"] == record.graph_view.digest
    recipe = last["config"]["effective"]["loaders"]["train"]["recipe"]
    assert "learning_rate" in recipe["reference_filled"]
    assert last["config"]["requested"]["learning_rate"] == 1e-4
    assert last["config"]["early_stopping"] is False
    assert not last["config"]["use_amp"]


def test_pause_and_resume_strictly(workspace, tmp_path):
    import train_model

    first, explicit = _config(workspace, tmp_path / "a", num_epochs=3, pause_after_epoch=1)
    train_model.train(first, explicit)
    parent = tmp_path / "a" / "checkpoints" / "last.pt"
    payload = torch.load(parent, map_location="cpu", weights_only=True)
    assert payload["progress"]["end"] == "paused"

    resumed = train_model.TrainConfig(data_dir=str(workspace), run_dir=str(tmp_path / "b"),
                                      learning_stage="base", resume_from=str(parent),
                                      device="cpu")
    train_model.train(resumed, {"learning_stage", "resume_from", "device"})
    child = torch.load(tmp_path / "b" / "checkpoints" / "last.pt", map_location="cpu",
                       weights_only=True)
    _, record = load_bundle(child, learning_stage="base")
    assert record.operation == "resume" and record.family_id == "a"
    assert record.parent["run"] == "a"
    assert child["progress"]["completed_epochs"] == 3
    assert child["model_spec"] == payload["model_spec"]


def test_refusals_before_anything_is_written(workspace, tmp_path):
    import train_model

    # A setting this build does not have is no TrainConfig field (a YAML that
    # names one is refused by `load_config_with_explicit`, "names settings this
    # version does not have").
    with pytest.raises(TypeError, match="hidden_dim"):
        _config(workspace, tmp_path / "x", hidden_dim=64)
    config, explicit = _config(workspace, tmp_path / "y", num_neighbors=[4, 4, 4])
    with pytest.raises(ValueError, match="one hop per encoder layer"):
        train_model.train(config, explicit)
    assert not (tmp_path / "x").exists() and not (tmp_path / "y").exists()


def _no_read(monkeypatch, train_model):
    """Every read `train` makes starts with the workspace's manifest
    (`read_training_inputs`); failing it shows a refusal came before any read."""
    monkeypatch.setattr(train_model, "read_training_inputs",
                        lambda *a, **k: pytest.fail("read before the refusal"))
    monkeypatch.setattr(train_model, "read_resume_checkpoint",
                        lambda *a, **k: pytest.fail("read before the refusal"))


def test_the_operators_set_is_required_never_inferred(workspace, tmp_path, monkeypatch):
    """P2-1: a setting stated at its default value is the operator's, so the
    set of stated fields is required. A missing set, or `None`, is a TypeError
    before anything is read or written, never "whatever differs from the
    defaults"; so is `plan_from_parent`'s."""
    from types import SimpleNamespace

    import train_model

    _no_read(monkeypatch, train_model)
    config, _ = _config(workspace, tmp_path / "run")
    with pytest.raises(TypeError):
        train_model.train(config)
    with pytest.raises(TypeError, match="never inferred"):
        train_model.train(config, None)
    monkeypatch.setattr("src.utils.run_plan.plan_continuation",
                        lambda *a, **k: pytest.fail("planned without the set"))
    parent = SimpleNamespace(checkpoint={}, identity=SimpleNamespace(path="p.pt", sha256="0" * 64))
    with pytest.raises(TypeError, match="never inferred"):
        train_model.plan_from_parent(parent, config, None, "cpu",
                                     record_version=MODEL_RECORD_VERSION)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("stage", [None, "pretrain"])
def test_a_run_without_a_learning_stage_is_refused_before_any_read(workspace, tmp_path,
                                                                   monkeypatch, stage):
    """Every run is a model bundle stage: one without a stage (the retired
    stage-less path), a resume included, is refused naming the stage and
    `--stage`, before anything is read or written."""
    import train_model

    _no_read(monkeypatch, train_model)
    for extra in ({}, {"resume_from": str(tmp_path / "parent.pt")}):
        config = train_model.TrainConfig(data_dir=str(workspace), run_dir=str(tmp_path / "run"),
                                         learning_stage=stage, device="cpu", **extra)
        with pytest.raises(ValueError, match=r"learning stage \(learning_stage, --stage\)"):
            train_model.train(config, {"device", *extra})
    assert not (tmp_path / "run").exists()


def test_the_command_line_says_a_refusal_in_one_line_without_a_traceback(workspace, tmp_path):
    """Checkpoint A's wording item: the command line run without --stage (a
    resume among them) is refused by `train()` before anything is written; it
    says so in one line, ``train_model: refused: … Nothing was written.``,
    exit 1, never "Training failed:" with a traceback."""
    import subprocess

    script = Path(__file__).resolve().parents[2] / "scripts" / "train_model.py"
    for extra in ([], ["--resume", str(tmp_path / "parent.pt")]):
        done = subprocess.run(
            [sys.executable, str(script), "--data-dir", str(workspace), "--device", "cpu",
             "--run-dir", str(tmp_path / "run"), *extra],
            cwd=script.parents[1], capture_output=True, text=True, timeout=300)
        assert done.returncode == 1, done.stderr[-2000:]
        lines = [line for line in done.stderr.splitlines() if "train_model: refused:" in line]
        assert len(lines) == 1, done.stderr[-2000:]
        assert lines[0].endswith(
            "train_model: refused: A run names its learning stage (learning_stage, --stage): "
            "one of ['base', 'finetune'], not None. Nothing was written."), lines[0]
        assert "Traceback" not in done.stderr and "Training failed" not in done.stderr
        assert not (tmp_path / "run").exists()


def test_an_error_after_the_run_directory_is_claimed_is_a_failure_not_a_refusal(
        tmp_path, monkeypatch, caplog):
    """The claim is the boundary (`train(on_claim=)`): every refusal comes
    before it, so the command line says a ValueError raised before it as a
    refusal, and one raised after it as a failed run, with its traceback."""
    import logging

    import train_model

    claimed = tmp_path / "run"
    monkeypatch.setattr(train_model, "load_config_with_explicit",
                        lambda args: (train_model.TrainConfig(data_dir=str(tmp_path)), set()))

    def after_the_claim(config, explicit, *, on_claim, **kwargs):
        on_claim(claimed)
        raise ValueError("a shape the model does not take")

    def before_the_claim(config, explicit, *, on_claim, **kwargs):
        raise ValueError("refused for a reason")

    monkeypatch.setattr(sys, "argv", ["train_model.py", "--data-dir", str(tmp_path),
                                      "--stage", "base"])
    args = train_model.parse_args()
    caplog.set_level(logging.INFO, logger=train_model.logger.name)
    monkeypatch.setattr(train_model, "train", after_the_claim)
    assert train_model._train_main(args) == 1
    failed = [r for r in caplog.records if r.getMessage().startswith("Training failed:")]
    assert failed and failed[0].exc_info is not None
    assert not any("refused" in r.getMessage() for r in caplog.records)
    caplog.clear()
    monkeypatch.setattr(train_model, "train", before_the_claim)
    assert train_model._train_main(args) == 1
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == [
        "train_model: refused: refused for a reason. Nothing was written."]
    assert all(r.exc_info is None for r in caplog.records if r.levelno >= logging.ERROR)


def test_a_run_directory_that_is_not_free_is_a_refusal_in_one_line(tmp_path, monkeypatch,
                                                                    caplog):
    """The review of C3 (listing 2): a run directory that holds files is
    refused by `train()` before its claim (`RunDirectoryTaken`, a
    FileExistsError), so the command line says it in one line, never
    "Training failed:" with a traceback, and the file stays as it was."""
    import logging

    import train_model

    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / "notes.txt").write_text("the operator's")
    monkeypatch.setattr(sys, "argv", ["train_model.py", "--data-dir", str(tmp_path),
                                      "--stage", "base", "--device", "cpu",
                                      "--run-dir", str(taken)])
    caplog.set_level(logging.INFO, logger=train_model.logger.name)
    assert train_model._train_main(train_model.parse_args()) == 1
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1 and errors[0].exc_info is None, [r.getMessage() for r in errors]
    said = errors[0].getMessage()
    assert said.startswith("train_model: refused: ") and said.endswith(" Nothing was written.")
    assert "already holds files" in said and "Training failed" not in said
    assert [p.name for p in taken.iterdir()] == ["notes.txt"]


@pytest.mark.parametrize("parent", ["list", "specless", "other version", "incomplete"])
def test_the_preview_and_the_process_say_the_same_of_a_file_that_is_no_record(
        workspace, tmp_path, parent):
    """Checkpoint A's wording item, at the shared root: a resume parent that is
    no checkpoint record (a list), one written before the model spec, one
    with a spec at another record version, or a record of this version the
    reader refuses is refused by the training service's preview and by the
    training process in the same words, the record reader's
    (`read_model_record`, which both read the parent with first), and nothing
    is written. The review of C3 (listing 3, display 2): the preview said
    "written before the model spec" of a file with a spec, where the process
    named its record version."""
    import train_model

    from src.api.services.training_manager import TrainingManager
    from src.models.bundle.record import ModelRecordError, read_model_record
    from src.utils.checkpoint_io import TRAINING_RULES_VERSION
    from src.utils.run_plan import RunPlanRefused, plan_continuation

    old = tmp_path / "old" / "checkpoints" / "last.pt"
    old.parent.mkdir(parents=True)
    spec = {"encoder": {"kind": "gatv2"}}
    payload = {"list": [1, 2, 3], "specless": {"record_version": 1, "epoch": 2},
               "other version": {"model_spec": spec, "record_version": 3, "epoch": 2},
               "incomplete": {"model_spec": spec, "record_version": 2, "epoch": 2,
                              "training_rules": TRAINING_RULES_VERSION}}[parent]
    torch.save(payload, old)
    said = f"The parent last.pt: {WRITTEN_BEFORE_SPEC}"
    if parent == "incomplete":
        with pytest.raises(ModelRecordError) as reader:
            read_model_record(payload)
        said = f"The parent last.pt: {reader.value}"
    if parent == "other version":
        said = ("The parent last.pt: checkpoint record version 3 is not 2; this build reads "
                "only its own")
        # The run plan's own check, for a caller that plans without the
        # reader, names the version too, never the phrase.
        with pytest.raises(RunPlanRefused) as planned:
            plan_continuation(payload, old, {}, set(), device="cpu", record_version=2)
        assert str(planned.value) == said
    preview = TrainingManager().preview_continuation(str(old), device="cpu")
    assert (preview["ok"], preview["reason"]) == (False, said)
    config = train_model.TrainConfig(data_dir=str(workspace), run_dir=str(tmp_path / "q"),
                                     learning_stage="base", resume_from=str(old), device="cpu")
    with pytest.raises(RunPlanRefused) as refused:
        train_model.train(config, {"learning_stage", "resume_from", "device"},
                          on_claim=lambda run: pytest.fail(f"a refusal claimed {run}"))
    assert str(refused.value) == said
    assert not (tmp_path / "q").exists()


def test_parent_without_a_model_spec_is_refused(workspace, tmp_path):
    import train_model
    from src.utils.run_plan import RunPlanRefused

    config, explicit = _config(workspace, tmp_path / "p")
    train_model.train(config, explicit)
    parent = tmp_path / "p" / "checkpoints" / "last.pt"
    payload = torch.load(parent, map_location="cpu", weights_only=True)
    for key in ("model_spec",):
        payload.pop(key)
    payload["record_version"] = 1
    old = tmp_path / "old.pt"
    torch.save(payload, old)
    resumed = train_model.TrainConfig(data_dir=str(workspace), run_dir=str(tmp_path / "q"),
                                      learning_stage="base", resume_from=str(old), device="cpu")
    with pytest.raises(RunPlanRefused, match="written before the model spec"):
        train_model.train(resumed, {"learning_stage", "resume_from", "device"})
    assert not (tmp_path / "q").exists()


def test_a_record_version_1_file_is_never_continued(workspace, tmp_path):
    """TM13: record version 1 (contract M3a's ShepherdGNN checkpoint, no model
    spec) retired with its writer. Resume, extension and branch of such a file
    are refused before anything is written, and the training service's preview
    says the same, by the one record-version check."""
    import train_model

    from src.api.services.training_manager import TrainingManager
    from src.config.model_types import MODEL_RECORD_KEYS
    from src.utils.run_plan import RunPlanRefused

    config, explicit = _config(workspace, tmp_path / "p", num_epochs=3, pause_after_epoch=1)
    train_model.train(config, explicit)
    payload = torch.load(tmp_path / "p" / "checkpoints" / "last.pt", map_location="cpu",
                         weights_only=True)
    old_file = {k: v for k, v in payload.items()
                if k not in MODEL_RECORD_KEYS and k != "training_rules"}
    old_file["record_version"] = 1
    old = tmp_path / "old" / "checkpoints" / "last.pt"
    old.parent.mkdir(parents=True)
    torch.save(old_file, old)

    base = {"data_dir": str(workspace), "learning_stage": "base", "device": "cpu"}
    attempts = {
        "resume": {"resume_from": str(old)},
        "extend": {"resume_from": str(old), "extend_epochs": 1},
        "branch": {"branch_from": str(old), "learning_rate": 5e-4},
    }
    for name, fields in attempts.items():
        run = tmp_path / name
        refused = train_model.TrainConfig(run_dir=str(run), **base, **fields)
        with pytest.raises(RunPlanRefused, match="written before the model spec"):
            train_model.train(refused, set(base) | set(fields))
        assert not run.exists(), f"a refused {name} wrote its run directory"
    preview = TrainingManager().preview_continuation(str(old), device="cpu")
    assert preview["ok"] is False
    # The preview says what the process says of the same file, in the record
    # reader's one phrase (checkpoint A's wording item).
    assert preview["reason"] == f"The parent last.pt: {WRITTEN_BEFORE_SPEC}"


def _with_earlier_rules(payload):
    """A parent as K4 wrote it (the reviewer's K5 review, P2-1): no training
    rules recorded, early stopping off in effect while the requested settings
    hold TrainConfig's default patience, the base validated with the run's own
    seed and batch size. Only the record differs from a current parent."""
    old = dict(payload)
    old.pop("training_rules")
    config = dict(old["config"])
    config["early_stopping"] = False
    config["requested"] = {**config["requested"], "early_stopping_patience": 10}
    effective = dict(config["effective"])
    loaders = dict(effective["loaders"])
    loaders["val"] = {k: v for k, v in loaders["val"].items() if k not in ("seed", "batch_size")}
    effective["loaders"] = loaders
    config["effective"] = effective
    old["config"] = config
    return old


def test_a_parent_trained_under_other_rules_is_never_continued(workspace, tmp_path):
    """Resume, extend and branch keep their parent's rules, so a bundle parent
    written under other training rules is refused before anything is written,
    by the one continuation check the manager's preview uses too; K5's rules
    would otherwise turn early stopping on and draw other validation pairs
    beside the parent's restored best value and history."""
    import train_model
    from src.api.services.training_manager import TrainingManager
    from src.utils.checkpoint_io import TRAINING_RULES_VERSION
    from src.utils.run_plan import RunPlanRefused

    config, explicit = _config(workspace, tmp_path / "p", num_epochs=3, pause_after_epoch=1)
    train_model.train(config, explicit)
    payload = torch.load(tmp_path / "p" / "checkpoints" / "last.pt", map_location="cpu",
                         weights_only=True)
    assert payload["training_rules"] == TRAINING_RULES_VERSION
    old = tmp_path / "old" / "checkpoints" / "last.pt"
    old.parent.mkdir(parents=True)
    torch.save(_with_earlier_rules(payload), old)

    base = dict(data_dir=str(workspace), learning_stage="base", device="cpu")
    attempts = {
        "resume": dict(resume_from=str(old)),
        "extend": dict(resume_from=str(old), extend_epochs=1),
        "branch": dict(branch_from=str(old), learning_rate=5e-4),
    }
    for name, fields in attempts.items():
        run = tmp_path / name
        refused = train_model.TrainConfig(run_dir=str(run), **base, **fields)
        with pytest.raises(RunPlanRefused, match="other training rules") as why:
            train_model.train(refused, set(base) | set(fields))
        assert "Start a new run, or finetune from it" in str(why.value), name
        assert not run.exists(), f"a refused {name} wrote its run directory"
    preview = TrainingManager().preview_continuation(str(old), device="cpu")
    assert preview["ok"] is False and "other training rules" in preview["reason"]


@pytest.mark.parametrize("patience", [None, 3])
def test_a_parent_of_these_rules_resumes_with_its_early_stopping_and_validation(
        workspace, tmp_path, patience):
    """A parent written under the current rules resumes as before, keeping
    early stopping on or off as it was and validating on the same pairs."""
    import train_model

    from src.utils.checkpoint_io import TRAINING_RULES_VERSION

    extra = {} if patience is None else {"early_stopping_patience": patience}
    config, explicit = _config(workspace, tmp_path / "a", num_epochs=3, pause_after_epoch=1,
                               **extra)
    train_model.train(config, explicit)
    parent = torch.load(tmp_path / "a" / "checkpoints" / "last.pt", map_location="cpu",
                        weights_only=True)
    resumed = train_model.TrainConfig(data_dir=str(workspace), run_dir=str(tmp_path / "b"),
                                      learning_stage="base", resume_from=str(
                                          tmp_path / "a" / "checkpoints" / "last.pt"),
                                      device="cpu")
    train_model.train(resumed, {"learning_stage", "resume_from", "device"})
    child = torch.load(tmp_path / "b" / "checkpoints" / "last.pt", map_location="cpu",
                       weights_only=True)
    assert child["training_rules"] == parent["training_rules"] == TRAINING_RULES_VERSION
    assert child["config"]["early_stopping"] is parent["config"]["early_stopping"] is (
        patience is not None)
    assert child["config"]["effective"]["loaders"]["val"] == (
        parent["config"]["effective"]["loaders"]["val"])


def test_bundle_runs_are_listed(tmp_path):
    """Every run-layout key's runs are listed. Which one serves is the record's
    call (task-first selection), pinned where selection lives
    (test_checkpoint_paths.py, test_task_first_selection.py)."""
    from src.utils.checkpoint_paths import list_runs

    for arch, run in (("gat", "20261001-000000_old_aaaaaa"), ("gatv2", "20261027-000000_base_bbbbbb")):
        ckpt = tmp_path / "runs" / arch / run / "checkpoints"
        ckpt.mkdir(parents=True)
        (ckpt / "last.pt").write_bytes(b"x")
    names = {r.name for r in list_runs(tmp_path)}
    assert names == {"20261001-000000_old_aaaaaa", "20261027-000000_base_bbbbbb"}
