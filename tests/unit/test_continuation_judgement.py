"""
A continuation gets one judgement: the CLI, the job runner and HTTP alike
=========================================================================
Checkpoint A's independent verification (ops-wiring round, decisions S1-S4,
S6) refuted "the CLI and the service give the same judgement of the same
request and parent" for continuations, by real executions: a resume stating a
setting its parent's stage does not read was refused by the CLI before
anything was written and spawned by the runner (null) or refused there as a
settings conflict with a branch remedy (8), and answered 'Extra inputs are not
permitted' over HTTP; an extension of a finetune child restating its own
setting was refused over HTTP only; a stated stage or task got the CLI's
identity sentence but the settings-conflict sentence from the runner; an
out-of-range value got the parent-conflict sentence from the CLI and the
runner, the field spec's from HTTP; and device auto on a host without CUDA
was refused by the CLI and spawned by the runner.

Each case here sends one request to the three entries: the training process
(`train_model.train`, the CLI and every spawned job), the job runner's start
(`TrainingManager.start_training` / `start_finetune`, the process stubbed) and
POST /api/v1/training/start (or /finetune, /branch) over the real routes, and
asserts the same accept or refuse in the same sentence. One rule each, in the
shared functions: `run_plan.parent_identity_problem` and `stated_problem`
(through `plan_continuation`) and the start request carrying every setting
and the task. The device, resolved before any spawn by the process's
resolver, is tested beside it (`test_training_device.py`).

Module: tests/unit/test_continuation_judgement.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")


def _train_model():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    if _SCRIPTS not in sys.path:
        sys.path.insert(0, _SCRIPTS)
    import train_model

    return train_model


@pytest.fixture(scope="module")
def paused_base(tmp_path_factory):
    """A real base paused after its first epoch, with an early stopping
    patience (3), as the verifier's base_pat."""
    train_model = _train_model()
    from tests.fixtures.base_workspace import write_base_workspace
    from tests.unit.test_train_base_stage import _config

    ws = write_base_workspace(tmp_path_factory.mktemp("paused-base"))
    run = ws / "runs" / "gatv2" / "20261001-000000_base_pat"
    train_model.train(*_config(ws, run, pause_after_epoch=1, early_stopping_patience=3))
    return ws, run / "checkpoints" / "last.pt"


@pytest.fixture
def entries(monkeypatch):
    """The runner and the real training routes over it; the job's process is
    stubbed where a test starts them (`_spawn_stubbed`)."""
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routes.training import router
    from src.api.services.training_manager import TrainingManager
    from tests.fixtures.operator_access import OPERATOR_BASE_URL, operator_mode

    manager = TrainingManager()
    monkeypatch.setattr("src.api.routes.training.training_manager", manager)
    monkeypatch.delenv("SHEPHERD_DEVICE", raising=False)
    operator_mode(monkeypatch)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return TestClient(app, base_url=OPERATOR_BASE_URL), manager


def _spawn_stubbed():
    """The job's process stubbed (`subprocess.Popen`), around the runner's and
    the route's starts only: the training process itself runs for real."""
    from tests.fixtures.job_processes import FakeProcess

    return patch("subprocess.Popen", side_effect=lambda *args, **kwargs: FakeProcess(pid=11))


def _cli(train_model, fields):
    """The training process on `fields`, the operator's: its refusal's words,
    or None when it trained."""
    config = train_model.TrainConfig(**fields)
    try:
        train_model.train(config, set(fields))
    except (ValueError, train_model.TrainingDeviceUnavailable) as refused:
        return str(refused)
    return None


def _refused_everywhere(train_model, entries, cli_fields, body, said, *, code, path="start"):
    """`said` from the CLI, the runner and HTTP, with HTTP `code` (422 for
    invalid settings, else 409) and the runner marking invalid settings
    exactly then; nothing spawned."""
    http, manager = entries
    assert _cli(train_model, cli_fields) == said
    with _spawn_stubbed() as popen:
        runner = manager.start_training(dict(body))
        resp = http.post(f"/api/v1/training/{path}", json=body)
    assert runner["success"] is False and runner["error"] == said, runner
    assert bool(runner.get("invalid_settings")) is (code == 422), runner
    assert resp.status_code == code, resp.text
    detail = resp.json()["detail"]
    assert detail == said or (
        isinstance(detail, list) and [e["msg"] for e in detail] == [f"Value error, {said}"]), detail
    popen.assert_not_called()
    assert manager.active_job() is None and manager.run_dir is None


# ---------------------------------------------------------------------------
# A setting the parent's stage and task do not read (decisions S1, S3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [None, 8], ids=["null", "eight"])
def test_a_resume_stating_a_setting_its_stage_does_not_read_is_refused_alike(
        paused_base, entries, value):
    """candidate_sample_size, a finetune's, on a resume of a base: the one
    check of a stated setting (`run_plan.stated_problem`, by the parent's
    stage and task), invalid settings everywhere (422), never spawned, never a
    settings conflict with a branch remedy, never pydantic's words."""
    from src.utils.run_plan import unread_problem

    train_model = _train_model()
    ws, parent = paused_base
    said = unread_problem("base", (None,), {"candidate_sample_size"})
    cli = {"data_dir": str(ws), "learning_stage": "base", "device": "cpu",
           "resume_from": str(parent), "candidate_sample_size": value}
    body = {"data_dir": str(ws), "device": "cpu", "resume_from": str(parent),
            "candidate_sample_size": value}
    _refused_everywhere(train_model, entries, cli, body, said, code=422)


def test_a_resume_stating_an_out_of_range_setting_its_stage_does_not_read_hears_unread_first(
        paused_base, entries):
    """candidate_sample_size 0 on a resume of a base breaks two rules: a base
    does not read it, and its field spec takes 1 or more. The training
    process says the first (`run_plan.stated_problem`: what is read, then
    each value), and so do the runner and HTTP, which carries a
    continuation's settings to the runner as sent (review of AF1: HTTP
    answered the field spec's sentence first)."""
    from src.utils.run_plan import unread_problem

    train_model = _train_model()
    ws, parent = paused_base
    said = unread_problem("base", (None,), {"candidate_sample_size"})
    cli = {"data_dir": str(ws), "learning_stage": "base", "device": "cpu",
           "resume_from": str(parent), "candidate_sample_size": 0}
    body = {"data_dir": str(ws), "device": "cpu", "resume_from": str(parent),
            "candidate_sample_size": 0}
    _refused_everywhere(train_model, entries, cli, body, said, code=422)


def test_an_out_of_range_value_on_a_resume_is_its_field_specs_refusal_alike(
        paused_base, entries):
    """learning_rate 5.0 on a resume: the field spec's sentence on every entry
    (422), before the parent's settings are compared."""
    from src.config.training_fields import value_problem

    train_model = _train_model()
    ws, parent = paused_base
    said = value_problem("learning_rate", 5.0)
    assert said and said.startswith("learning_rate must be")
    cli = {"data_dir": str(ws), "learning_stage": "base", "device": "cpu",
           "resume_from": str(parent), "learning_rate": 5.0}
    body = {"data_dir": str(ws), "device": "cpu", "resume_from": str(parent),
            "learning_rate": 5.0}
    _refused_everywhere(train_model, entries, cli, body, said, code=422)


def test_a_setting_other_than_the_parents_is_still_a_conflict_of_the_parent(
        paused_base, entries):
    """A setting the run reads, stated at another value than the parent's,
    stays the parent's refusal (409), in one sentence."""
    train_model = _train_model()
    ws, parent = paused_base
    cli = {"data_dir": str(ws), "learning_stage": "base", "device": "cpu",
           "resume_from": str(parent), "early_stopping_patience": 5}
    said = _cli(train_model, cli)
    assert said.startswith("The parent last.pt was trained with other settings") and (
        "early_stopping_patience: the parent's 3, requested 5" in said), said
    body = {"data_dir": str(ws), "device": "cpu", "resume_from": str(parent),
            "early_stopping_patience": 5}
    _refused_everywhere(train_model, entries, cli, body, said, code=409)


# ---------------------------------------------------------------------------
# A stage or task other than the parent's: one identity sentence (S6)
# ---------------------------------------------------------------------------
BASE_IDENTITY = ("The parent last.pt is a base checkpoint; a run continues only its own stage "
                 "and task.")


@pytest.mark.parametrize("stated", [{"learning_stage": "finetune"}, {"task": "disease"},
                                    {"task": "foo"}],
                         ids=["stage", "task", "unknown-task"])
def test_a_resume_stating_another_stage_or_task_gets_the_identity_sentence(
        paused_base, entries, stated):
    """A stage or task names the run: another than the parent's is refused by
    the one check (`run_plan.parent_identity_problem`), a refusal of the
    parent (409), with no branch remedy, which cannot change a stage. The
    CLI states a stage always (a finetune with its task). A task that is no
    task of this build is still another than the parent's: HTTP carries it
    to the runner as sent (review of AF1: it answered the field spec's
    choices first, 422)."""
    from src.utils.run_plan import parent_identity_problem

    train_model = _train_model()
    ws, parent = paused_base
    assert parent_identity_problem("base", None, "finetune", "disease",
                                   "The parent last.pt") == BASE_IDENTITY
    cli_stage = ({"learning_stage": "finetune", "task": "disease"}
                 if "learning_stage" in stated else {"learning_stage": "base", **stated})
    cli = {"data_dir": str(ws), "device": "cpu", "resume_from": str(parent), **cli_stage}
    body = {"data_dir": str(ws), "device": "cpu", "resume_from": str(parent), **stated}
    _refused_everywhere(train_model, entries, cli, body, BASE_IDENTITY, code=409)


def test_a_branch_stating_another_task_gets_the_branch_identity_sentence(paused_base, entries):
    """The branch's identity check is the same function, in a branch's words;
    the CLI and the runner agree. POST /training/branch cannot state a task
    (its request names the run by data_dir, branch_from, device and
    run_name, and takes settings to change): a task among its settings is
    refused as no training setting (422), and a finetune's setting on a
    base's branch by the one check of a stated setting."""
    train_model = _train_model()
    ws, parent = paused_base
    said = ("The branch parent last.pt is a base checkpoint; a branch keeps its parent's "
            "stage and task.")
    http, manager = entries
    assert _cli(train_model, {"data_dir": str(ws), "device": "cpu", "learning_stage": "base",
                              "task": "disease", "branch_from": str(parent)}) == said
    branch = {"data_dir": str(ws), "device": "cpu", "branch_from": str(parent)}
    with _spawn_stubbed() as popen:
        runner = manager.start_training({**branch, "task": "disease"})
        task = http.post("/api/v1/training/branch",
                         json={**branch, "settings": {"task": "disease"}})
        setting = http.post("/api/v1/training/branch",
                            json={**branch, "settings": {"task_loss_weight": 0.5}})
    assert runner["success"] is False and runner["error"] == said, runner
    assert task.status_code == 422, task.text
    assert [e["msg"] for e in task.json()["detail"]] == [
        "Value error, settings holds 'task', which is not a training setting; a branch's run "
        "is named by data_dir, branch_from, device and run_name."]
    assert setting.status_code == 422 and setting.json()["detail"].startswith(
        "task_loss_weight is not a setting a base run reads"), setting.text
    popen.assert_not_called()


# ---------------------------------------------------------------------------
# A finetune child's own setting, restated at its value: inherited (S3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["candidate_sample_size", "frozen_parts"])
def test_an_extension_of_a_child_restating_its_own_setting_is_accepted_alike(
        trained_disease_child, entries, tmp_path, name):
    """Extending a finished disease child, restating a setting of its own at
    the child's value: the CLI trains it (on a copy of the workspace), the
    runner starts it and POST /training/start takes it (never 'Extra
    inputs'), each with the setting kept."""
    import torch

    train_model = _train_model()
    http, manager = entries
    child = trained_disease_child["disease"]
    recorded = torch.load(child, map_location="cpu", weights_only=True)["config"]["requested"]
    value = recorded[name]
    stated = {name: value}

    copy = tmp_path / "ws"
    shutil.copytree(trained_disease_child["data_dir"], copy,
                    ignore=shutil.ignore_patterns("runs"))
    assert _cli(train_model, {"data_dir": str(copy), "learning_stage": "finetune",
                              "task": "disease", "device": "cpu", "resume_from": str(child),
                              "extend_epochs": 1, **stated}) is None

    body = {"data_dir": str(trained_disease_child["data_dir"]), "device": "cpu",
            "resume_from": str(child), "extend_epochs": 1, **stated}
    with _spawn_stubbed():
        runner = manager.start_training(dict(body))
        assert runner["success"] is True, runner
        written = yaml.safe_load(Path(runner["config_path"]).read_text())
        assert written[name] == value and written["learning_stage"] == "finetune"
        manager.stop_training()
        resp = http.post("/api/v1/training/start", json=body)
    assert resp.status_code == 200, resp.text
    assert yaml.safe_load(Path(resp.json()["config_path"]).read_text())[name] == value


def test_a_new_base_is_still_refused_a_finetune_setting_in_the_shared_sentence(
        entries, tmp_path):
    """The start carries every setting and the task, and a new base keeps
    refusing what a base does not read in the one check's sentence
    (`run_plan.new_run_problem`), on the CLI, the runner and HTTP alike,
    before anything is read; an unknown name stays pydantic's
    extra_forbidden."""
    from src.utils.run_plan import new_run_problem

    train_model = _train_model()
    http, manager = entries
    body = {"data_dir": str(tmp_path / "ws"), "learning_stage": "base", "device": "cpu"}
    with _spawn_stubbed() as popen:
        for stated in ({"candidate_sample_size": 8}, {"frozen_parts": "none"}):
            said = new_run_problem("base", (None,), stated)
            assert said
            assert _cli(train_model, {**body, **stated}) == said
            runner = manager.start_training({**body, **stated})
            assert runner["error"] == f"{said} Nothing was started."
            assert runner["invalid_settings"] is True
            resp = http.post("/api/v1/training/start", json={**body, **stated})
            assert resp.status_code == 422, resp.text
            assert [e["msg"] for e in resp.json()["detail"]] == [f"Value error, {said}"]
        resp = http.post("/api/v1/training/start", json={**body, "hidden_dim": 64})
    assert resp.status_code == 422 and resp.json()["detail"][0]["type"] == "extra_forbidden"
    popen.assert_not_called()


# ---------------------------------------------------------------------------
# A new base's stage and task: one identity sentence (review of AF1)
# ---------------------------------------------------------------------------
BASE_HAS_NO_TASK = "A base run has no task and no finetune parent."


def test_a_new_base_stating_a_task_gets_one_identity_sentence(entries, tmp_path):
    """A task names a run, and a base has none: a task stated on a new base
    is refused by the one check of a new run's stage and task
    (`run_plan.new_run_identity_problem`), never as a setting a base does
    not read, on the CLI, the runner (invalid settings) and HTTP (422)
    alike, before anything is read; a finetune parent on a base is the same
    refusal on the CLI and the runner (the start request has no such
    field)."""
    from src.utils.run_plan import new_run_identity_problem

    train_model = _train_model()
    http, manager = entries
    assert new_run_identity_problem("base", "disease") == BASE_HAS_NO_TASK
    assert new_run_identity_problem("base", None, "base.pt") == BASE_HAS_NO_TASK
    body = {"data_dir": str(tmp_path / "ws"), "learning_stage": "base", "device": "cpu"}
    with _spawn_stubbed() as popen:
        for stated in ({"task": "disease"}, {"task": "foo"}, {"finetune_from": "base.pt"}):
            assert _cli(train_model, {**body, **stated}) == BASE_HAS_NO_TASK, stated
            runner = manager.start_training({**body, **stated})
            assert runner["error"] == f"{BASE_HAS_NO_TASK} Nothing was started.", runner
            assert runner["invalid_settings"] is True
            if "task" in stated:
                resp = http.post("/api/v1/training/start", json={**body, **stated})
                assert resp.status_code == 422, resp.text
                assert [e["msg"] for e in resp.json()["detail"]] == [
                    f"Value error, {BASE_HAS_NO_TASK}"]
    popen.assert_not_called()


def test_a_new_base_stating_a_null_task_is_accepted_alike(entries, tmp_path):
    """An explicit null task is no task, decided alike on every entry
    (decision S2: it is the operator's, and a base has none): the CLI trains
    the base, the runner starts it and POST /training/start takes it."""
    train_model = _train_model()
    from tests.fixtures.base_workspace import write_base_workspace
    from tests.unit.test_train_base_stage import _config

    http, manager = entries
    ws = write_base_workspace(tmp_path / "ws")
    config, explicit = _config(ws, tmp_path / "cli-run", task=None, num_epochs=1)
    assert "task" in explicit
    assert "val_lp_accuracy" in train_model.train(config, explicit)
    body = {"data_dir": str(ws), "learning_stage": "base", "device": "cpu", "task": None}
    with _spawn_stubbed():
        runner = manager.start_training(dict(body))
        assert runner["success"] is True, runner
        manager.stop_training()
        resp = http.post("/api/v1/training/start", json=body)
    assert resp.status_code == 200, resp.text
