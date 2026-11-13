"""The settings projection (ops-wiring round A1a; decisions S1-S6).

What a new run takes is stated once (`run_plan.new_run_settings`), and the
training process (`train_model.train`), the job runner's starts and the API's
start (`TrainingStartRequest`) read it. The rules across settings are data in
the field spec (`training_fields.CROSS_FIELD_RULES`) with one evaluator, so
the CLI, every spawned job and the service give one judgement in one
sentence: an explicit setting the run would not read is refused, never
ignored, and an inherited one never is. A Fine-tune start checks what each
child's process checks against its base before any child starts. The start
request is a thin projection: an explicit null reaches the service and is
judged, a -1 hop is legal, and a finetune child continues over HTTP. Each
setting's source is recorded, and the continuation preview states the
optimizer groups whose rate ended at 0.

Module: tests/unit/test_settings_projection.py
"""
from __future__ import annotations

import copy
import dataclasses
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.config import recipe
from src.config import training_fields as tf
from src.utils.run_plan import (
    OPERATIONAL_FIELDS,
    new_run_problem,
    new_run_settings,
    new_run_values,
    unread_problem,
)

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")

#: Every stage and task a new run is.
RUNS = [("base", None)] + [("finetune", task) for task in recipe.FINETUNE_TASKS]


def _train_model():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    if _SCRIPTS not in sys.path:
        sys.path.insert(0, _SCRIPTS)
    import train_model

    return train_model


# ---------------------------------------------------------------------------
# S1: one definition of what a new run takes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stage, task", RUNS)
def test_a_new_run_takes_the_learning_settings_but_those_its_stage_does_not_read(stage, task):
    """The definition: every learning setting (a TrainConfig field outside
    the operational ones), but the run's stage and task and the recipe's
    settings this stage and task do not read; and each has a field spec."""
    train_model = _train_model()
    fields = {f.name for f in dataclasses.fields(train_model.TrainConfig)}
    learning = fields - OPERATIONAL_FIELDS - train_model.LOCATION_FIELDS
    expected = learning - {"learning_stage", "task"} - recipe.unread_settings(stage, task)
    assert new_run_settings(stage, task) == expected
    assert new_run_settings(stage, task) <= set(tf.names())
    assert set(recipe.reference_settings(stage, task)[0]) <= new_run_settings(stage, task)
    assert {"model_spec", "split_seed", "min_lr_ratio", "amp_dtype",
            "gradient_accumulation_steps", "eval_every_n_epochs",
            "save_top_k"} <= new_run_settings(stage, task)


@pytest.mark.parametrize("stage, task", RUNS)
def test_the_training_process_refuses_exactly_what_a_new_run_does_not_take(stage, task, tmp_path):
    """`train()` asks `new_run_settings`: a learning setting outside it is
    refused before anything is read (the workspace here does not exist), and
    one inside it passes that check (it is refused later, by another rule or
    by the absent workspace, never as a setting the run does not read)."""
    train_model = _train_model()
    parent = {"finetune_from": str(tmp_path / "base.pt")} if stage == "finetune" else {}
    learning = (set(tf.names()) - OPERATIONAL_FIELDS) - {"learning_stage", "task"}
    for name in sorted(learning):
        config = train_model.TrainConfig(data_dir=str(tmp_path / "absent"), device="cpu",
                                         learning_stage=stage, task=task, **parent)
        with pytest.raises(Exception) as refused:
            train_model.train(config, {"learning_stage", "task", "device", name})
        said = str(refused.value)
        unread = said.startswith(f"{name} is not a setting a {stage} ")
        assert unread is (name not in new_run_settings(stage, task)), (name, said)
        if unread:
            assert said == unread_problem(stage, (task,), {name})


#: Requests every entry refuses by one check of a new run's settings, before
#: anything is read: a bound, a null, a kind, a setting a base does not read,
#: and a null where a rule needs a number.
ONE_SENTENCE = {
    "bound": {"early_stopping_patience": 0},
    "null": {"split_seed": None},
    "kind": {"seed": "5"},
    "bool": {"use_amp": 1},
    "unread": {"task_loss_weight": 0.5},
    "null under a rule": {"scheduler_type": "onecycle", "min_lr_ratio": None},
}


@pytest.mark.parametrize("case", sorted(ONE_SENTENCE))
def test_one_request_has_one_sentence_on_the_cli_the_runner_and_http(case, tmp_path,
                                                                    monkeypatch):
    """The one check of a new run's settings (`run_plan.new_run_problem`):
    the training process (the CLI and every spawned job), the job runner's
    start and POST /training/start refuse the same request in the same
    sentence, before anything is read, written or spawned. A value is never
    coerced (a string is not a number, 1 is not true), and a null under a
    rule is refused as a null, never left for the Trainer to fail on."""
    train_model = _train_model()
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routes.training import router
    from src.api.services.training_manager import TrainingManager
    from tests.fixtures.operator_access import OPERATOR_BASE_URL, operator_mode

    settings = ONE_SENTENCE[case]
    said = new_run_problem("base", (None,), settings)
    assert said and said.startswith("min_lr_ratio must be set" if case == "null under a rule"
                                    else next(iter(settings))), said
    config = train_model.TrainConfig(data_dir=str(tmp_path / "absent"), device="cpu",
                                     learning_stage="base", **settings)
    with pytest.raises(ValueError) as cli:
        train_model.train(config, {"learning_stage", "device", *settings})
    assert str(cli.value) == said

    manager = TrainingManager()
    monkeypatch.setattr("src.api.routes.training.training_manager", manager)
    monkeypatch.delenv("SHEPHERD_DEVICE", raising=False)
    operator_mode(monkeypatch)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    body = {"data_dir": str(tmp_path / "ws"), "learning_stage": "base", "device": "cpu",
            **settings}
    with patch("subprocess.Popen") as popen:
        runner = manager.start_training(dict(body))
        resp = TestClient(app, base_url=OPERATOR_BASE_URL).post("/api/v1/training/start",
                                                                json=body)
    assert runner["success"] is False and runner["invalid_settings"] is True
    assert runner["error"] == f"{said} Nothing was started."
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    if isinstance(detail, str):          # passed on: the runner's own words
        assert detail == f"{said} Nothing was started."
    else:                                # the request's check, by the same function
        assert [error["msg"] for error in detail] == [f"Value error, {said}"], detail
    popen.assert_not_called()
    assert manager.active_job() is None and manager.run_dir is None


def test_the_fill_of_a_new_run_is_the_operators_then_the_recipes_then_the_builds():
    values = new_run_values("base", None, {"learning_rate": 0.5, "split_seed": 3})
    assert set(values) == new_run_settings("base")
    assert (values["learning_rate"], values["split_seed"]) == (0.5, 3)
    assert values["num_epochs"] == recipe.BASE["num_epochs"]
    assert values["min_lr_ratio"] == tf.by_name("min_lr_ratio").default
    assert values["model_spec"] is None


# ---------------------------------------------------------------------------
# S3: the rules across settings, one evaluator, one sentence
# ---------------------------------------------------------------------------
#: Each rule: settings it refuses on a new base (the recipe's scheduler none,
#: AMP off) and settings it takes.
RULES = {
    "warmup_steps_read_only_by_cosine": (
        {"warmup_steps": 500}, {"scheduler_type": "cosine", "warmup_steps": 500}),
    "min_lr_ratio_read_only_by_a_scheduler": (
        {"min_lr_ratio": 0.05}, {"scheduler_type": "linear", "min_lr_ratio": 0.05}),
    "amp_dtype_read_only_with_amp": (
        {"amp_dtype": "bfloat16"}, {"use_amp": True, "amp_dtype": "bfloat16"}),
    "onecycle_requires_positive_min_lr_ratio": (
        {"scheduler_type": "onecycle", "min_lr_ratio": 0.0},
        {"scheduler_type": "onecycle", "min_lr_ratio": 0.01}),
}


def _sentence(rule, settings):
    return tf.rule_problem(rule, new_run_values("base", None, settings))


def test_every_rule_is_data_with_its_cases_here():
    assert set(RULES) == {rule.name for rule in tf.CROSS_FIELD_RULES}
    said = _sentence("warmup_steps_read_only_by_cosine", {"warmup_steps": 500})
    assert said.startswith("warmup_steps is read only when scheduler_type is 'cosine', and "
                           "this run's scheduler_type is 'none', so it would not take effect.")
    assert "0 warms up over the Trainer's warmup_ratio" in said
    assert _sentence("onecycle_requires_positive_min_lr_ratio",
                     RULES["onecycle_requires_positive_min_lr_ratio"][0]).startswith(
        "min_lr_ratio must be more than 0 when scheduler_type is 'onecycle', not 0.0")


@pytest.mark.parametrize("rule", sorted(RULES))
def test_a_rule_refuses_only_what_the_operator_set(rule):
    """Judged on the settings the run trains with, for what the operator set:
    the same values filled or inherited are never refused."""
    refused, taken = RULES[rule]
    values = new_run_values("base", None, refused)
    assert tf.cross_field_problems(values, set(refused)) == [tf.rule_problem(rule, values)]
    assert tf.cross_field_problems(values, set()) == []
    assert tf.cross_field_problems(new_run_values("base", None, taken), set(taken)) == []


@pytest.mark.parametrize("rule", sorted(rule.name for rule in tf.CROSS_FIELD_RULES
                                         if rule.kind == "read_only_when"))
def test_what_a_form_leaves_out_is_what_a_start_refuses(rule):
    """`not_read_by_rules` reads each rule's own guard: a setting it says the
    run would not read is the one `cross_field_problems` refuses once the
    operator sets it, and one it says the run reads is taken (the Training
    page's base form hides the first and sends the second). A guard the
    values do not hold is not judged."""
    refused, taken = RULES[rule]
    subject = next(r.subject for r in tf.CROSS_FIELD_RULES if r.name == rule)
    values = new_run_values("base", None, refused)
    assert subject in tf.not_read_by_rules(values)
    assert tf.cross_field_problems(values, {subject})
    values = new_run_values("base", None, taken)
    assert subject not in tf.not_read_by_rules(values)
    assert tf.cross_field_problems(values, {subject}) == []
    assert tf.not_read_by_rules({subject: refused[subject]}) == frozenset()


@pytest.mark.parametrize("rule", sorted(RULES))
def test_the_cli_refuses_a_rule_in_its_sentence_before_anything_is_read(rule, tmp_path):
    """`train()`: the illegal case is refused by the rule before the
    workspace is read (it does not exist here); the legal case passes the
    rule and reaches the workspace's own refusal."""
    train_model = _train_model()
    refused, taken = RULES[rule]
    for settings, legal in ((refused, False), (taken, True)):
        config = train_model.TrainConfig(data_dir=str(tmp_path / "absent"), device="cpu",
                                         learning_stage="base", **settings)
        with pytest.raises(Exception) as said:
            train_model.train(config, {"learning_stage", "device", *settings})
        if legal:
            assert not any(str(said.value).startswith(r.subject) for r in tf.CROSS_FIELD_RULES)
        else:
            assert str(said.value) == _sentence(rule, settings)


def test_the_trainer_refuses_onecycle_in_the_rules_sentence():
    pytest.importorskip("torch")
    from src.training.trainer import _validate_min_lr_ratio

    with pytest.raises(ValueError) as refused:
        _validate_min_lr_ratio(0.0, "onecycle")
    assert str(refused.value) == tf.rule_problem(
        "onecycle_requires_positive_min_lr_ratio",
        {"scheduler_type": "onecycle", "min_lr_ratio": 0.0})


def test_amp_asked_on_a_cpu_device_stays_accepted_for_a_new_run():
    """Decision S3: the record states the precision requested and in effect
    apart, so a request for AMP on a device without it is no rule's problem."""
    values = new_run_values("base", None, {"use_amp": True, "amp_dtype": "float16"})
    assert tf.cross_field_problems(values, {"use_amp", "amp_dtype"}) == []


def _base_parent(**recorded):
    """A paused base's checkpoint record, its settings the recipe's filled
    (scheduler none, AMP off) with `recorded` over them."""
    from src.utils.checkpoint_io import TRAINING_RULES_VERSION

    requested = {**new_run_values("base", None, recorded), "learning_stage": "base",
                 "task": None}
    return {"record_version": 2, "training_rules": TRAINING_RULES_VERSION,
            "checkpoint_kind": "training_state", "learning_stage": "base", "task": None,
            "config": {"requested": requested,
                       "effective": {"amp": {"enabled": False, "dtype": "float16"}}},
            "progress": {"end": "paused", "completed_epochs": 1,
                         "stage": {"kind": "original", "start_epoch": 0, "target_epochs": 2,
                                   "schedule": {"type": "constant"}}}}, requested


def test_a_continuation_meets_the_rules_only_in_what_it_changes():
    """The plans judge the rules across settings on what a request changes
    (a branch's changes): a setting kept from the parent, inherited or stated
    at the parent's value, is never refused by them, so a resume and an
    extension, which change nothing, never meet them; a branch's change the
    run would not read is refused, and so is a change of the guard that makes
    an inherited value break a rule."""
    from src.utils.run_plan import RunPlanRefused, plan_branch, plan_continuation

    parent, recorded = _base_parent()
    request = {**recorded, "device": "cpu", "extend_epochs": None, "extend_lr": None}
    where = Path("runs/gatv2/base/checkpoints/last.pt")
    plan = plan_continuation(parent, where, request, set(), device="cpu", record_version=2)
    assert plan.settings["warmup_steps"] == 0 and plan.inherited["warmup_steps"] == 0
    stated = plan_continuation(parent, where, request, {"warmup_steps", "min_lr_ratio",
                                                        "amp_dtype"},
                               device="cpu", record_version=2)
    assert stated.settings == plan.settings and "warmup_steps" not in stated.inherited
    branch = {**request, "branch_from": "p.pt"}
    with pytest.raises(RunPlanRefused, match="^min_lr_ratio is read only when"):
        plan_branch(parent, where, {**branch, "min_lr_ratio": 0.2}, {"min_lr_ratio"},
                    device="cpu", record_version=2)
    kept = plan_branch(parent, where, {**branch, "learning_rate": 0.5},
                       {"learning_rate", "warmup_steps"}, device="cpu", record_version=2)
    assert set(kept.changes) == {"learning_rate"} and kept.settings["warmup_steps"] == 0
    assert plan_branch(parent, where, {**branch, "scheduler_type": "linear",
                                       "min_lr_ratio": 0.2},
                       {"scheduler_type", "min_lr_ratio"}, device="cpu",
                       record_version=2).changes["min_lr_ratio"] == {"parent": 0.01, "new": 0.2}
    floor, recorded = _base_parent(scheduler_type="cosine", min_lr_ratio=0.0)
    with pytest.raises(RunPlanRefused, match="^min_lr_ratio must be more than 0 when "
                                             "scheduler_type is 'onecycle', not 0.0"):
        plan_branch(floor, where, {**recorded, "device": "cpu", "branch_from": "p.pt",
                                   "scheduler_type": "onecycle"},
                    {"scheduler_type"}, device="cpu", record_version=2)


def test_the_warmup_help_says_what_the_trainer_does():
    """The words, not the trainer, are corrected (decision S3): warmup is read
    only by the cosine schedule, and 0 warms up over the Trainer's
    warmup_ratio, a tenth of the steps."""
    pytest.importorskip("gradio")
    pytest.importorskip("torch")
    from src.training.trainer import TrainerConfig
    from src.webui.components import training_console

    # The help's lead is the field spec's description (one text of it).
    lead = training_console.FIELD_HELP["warmup_steps"][1].split("\n", 1)[0]
    assert lead == tf.by_name("warmup_steps").description
    assert TrainerConfig().warmup_ratio == 0.1
    assert "read only with cosine" in lead and "0 warms up over a tenth of the steps" in lead
    assert "from zero" not in lead and len(lead) <= 90
    # The format under the field says it too, and the cap the Trainer applies
    # (`Trainer._create_scheduler`: at most half the run's steps).
    shown = training_console.FIELD_FORMATS["warmup_steps"]
    assert "0 takes a tenth" in shown and "at most half" in shown


@pytest.mark.parametrize("requested, effective", [
    (10_000, 20),   # a number far above the run's steps: at most half of them (40 // 2)
    (0, 4),         # 0: a tenth of them (int(40 * 0.1))
    (7, 7),         # a number under the cap: as asked
])
def test_the_warmup_words_are_what_a_real_trainer_does(tmp_path, requested, effective):
    """The words under Warmup steps ("at most half the run's; 0 takes a
    tenth of them") held to a real cosine Trainer's schedule: 10 epochs of 4
    batches, 40 steps, as its record states them (`schedule_record`)."""
    torch = pytest.importorskip("torch")
    from src.training.trainer import Trainer, TrainerConfig
    from tests.unit.test_model_record import _stored_task

    trainer = Trainer(model=torch.nn.Linear(2, 1), train_dataloader=[{}] * 4,
                      config=TrainerConfig(device="cpu", scheduler_type="cosine",
                                           warmup_steps=requested, use_amp=False, num_epochs=10,
                                           checkpoint_dir=str(tmp_path / "ckpt"),
                                           log_dir=str(tmp_path / "logs"),
                                           early_stopping_monitor="val_pooled_mrr"),
                      task=_stored_task())
    schedule = trainer.schedule_record
    assert schedule["total_steps"] == 40
    assert schedule["warmup_steps_requested"] == requested
    assert schedule["warmup_steps_effective"] == effective


# ---------------------------------------------------------------------------
# S6: the start request over HTTP
# ---------------------------------------------------------------------------
@pytest.fixture
def client(monkeypatch):
    """The real route over a real TrainingManager; only the process is stubbed."""
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routes.training import router
    from src.api.services.training_manager import TrainingManager
    from tests.fixtures.job_processes import FakeProcess
    from tests.fixtures.operator_access import OPERATOR_BASE_URL, operator_mode

    manager = TrainingManager()
    monkeypatch.setattr("src.api.routes.training.training_manager", manager)
    monkeypatch.delenv("SHEPHERD_DEVICE", raising=False)
    operator_mode(monkeypatch)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    process = FakeProcess(pid=11)
    with patch("subprocess.Popen", return_value=process) as popen:
        yield TestClient(app, base_url=OPERATOR_BASE_URL), popen, manager
    process.finish(0)


def _written(resp):
    return yaml.safe_load(Path(resp.json()["config_path"]).read_text())


@pytest.mark.parametrize("rule", sorted(RULES))
def test_the_api_refuses_a_rule_with_422_before_any_spawn_in_the_clis_sentence(client, rule,
                                                                              tmp_path):
    http, popen, manager = client
    refused, taken = RULES[rule]
    body = {"data_dir": str(tmp_path / "ws"), "learning_stage": "base", "device": "cpu"}
    resp = http.post("/api/v1/training/start", json={**body, **refused})
    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"] == f"{_sentence(rule, refused)} Nothing was started."
    popen.assert_not_called()
    assert manager.active_job() is None and manager.run_dir is None
    resp = http.post("/api/v1/training/start", json={**body, **taken})
    assert resp.status_code == 200, resp.text
    popen.assert_called_once()
    assert {name: _written(resp)[name] for name in taken} == taken


def test_the_api_takes_every_setting_a_new_base_reads_and_a_minus_one_hop(client, tmp_path):
    http, popen, _ = client
    settings = {"split_seed": 7, "gradient_accumulation_steps": 2, "eval_every_n_epochs": 2,
                "save_top_k": 1, "num_neighbors": [-1, 10, 5], "compile": False}
    resp = http.post("/api/v1/training/start", json={
        "data_dir": str(tmp_path / "ws"), "learning_stage": "base", "device": "cpu", **settings})
    assert resp.status_code == 200, resp.text
    assert {name: _written(resp)[name] for name in settings} == settings


def test_an_explicit_null_reaches_the_service_and_is_judged(client, tmp_path):
    """Decision S2: a null the caller sent is the caller's: refused where the
    setting takes none, kept where it does (no patience), never dropped."""
    http, popen, _ = client
    body = {"data_dir": str(tmp_path / "ws"), "learning_stage": "base", "device": "cpu"}
    resp = http.post("/api/v1/training/start", json={**body, "num_epochs": None})
    assert resp.status_code == 422 and "num_epochs must be set, not null." in resp.text
    popen.assert_not_called()
    resp = http.post("/api/v1/training/start", json={**body, "early_stopping_patience": None,
                                                     "resume_from": None})
    assert resp.status_code == 200, resp.text
    written = _written(resp)
    assert "early_stopping_patience" in written and written["early_stopping_patience"] is None
    assert "resume_from" not in written, "a null parent is no parent"


@pytest.fixture(scope="module")
def patient_base(tmp_path_factory):
    """A real paused base whose run has an early stopping patience (5)."""
    train_model = _train_model()
    from tests.fixtures.base_workspace import write_base_workspace
    from tests.unit.test_train_base_stage import _config

    ws = write_base_workspace(tmp_path_factory.mktemp("patient-base"))
    run = ws / "runs" / "gatv2" / "20261001-000000_base_cccccc"
    train_model.train(*_config(ws, run, pause_after_epoch=1, early_stopping_patience=5))
    return ws, run / "checkpoints" / "last.pt"


def test_an_explicit_null_patience_on_a_resume_is_a_change_refused(client, patient_base):
    http, popen, manager = client
    ws, parent = patient_base
    body = {"data_dir": str(ws), "resume_from": str(parent), "device": "cpu"}
    resp = http.post("/api/v1/training/start", json={**body, "early_stopping_patience": None})
    assert resp.status_code == 409, resp.text
    assert "early_stopping_patience: the parent's 5, requested None" in resp.text
    popen.assert_not_called()
    resp = http.post("/api/v1/training/start", json=body)
    assert resp.status_code == 200, resp.text
    assert "early_stopping_patience" not in _written(resp), "inherited, not written"


def test_a_resume_restating_a_setting_its_run_never_read_is_accepted(client, patient_base):
    """The parent's scheduler is none, so it never read its warmup_steps (0);
    restated at that value, the resume changes nothing, and no rule across
    settings refuses what a continuation keeps."""
    http, popen, manager = client
    ws, parent = patient_base
    resp = http.post("/api/v1/training/start", json={
        "data_dir": str(ws), "resume_from": str(parent), "device": "cpu", "warmup_steps": 0})
    assert resp.status_code == 200, resp.text
    assert _written(resp)["warmup_steps"] == 0
    popen.assert_called_once()


def test_a_finetune_child_is_extended_over_http_without_its_stage(client, trained_disease_child):
    """Decision S6: with resume_from the stage may be left out (the parent's
    record gives it), and a stage given is compared with the parent's, by the
    one check of a run's stage and task against its parent's
    (`run_plan.parent_identity_problem`, checkpoint A's verification)."""
    http, popen, manager = client
    body = {"data_dir": str(trained_disease_child["data_dir"]),
            "resume_from": str(trained_disease_child["disease"]), "device": "cpu",
            "extend_epochs": 1}
    resp = http.post("/api/v1/training/start", json={**body, "learning_stage": "base"})
    assert resp.status_code == 409 and resp.json()["detail"] == (
        "The parent last.pt is a finetune checkpoint for disease; a run continues only its "
        "own stage and task.")
    stated = manager.preview_continuation(body["resume_from"], device="cpu", extend_epochs=1,
                                          requested={"learning_stage": "finetune"})
    assert stated["ok"] is True and stated["mode"] == "extend", stated
    resp = http.post("/api/v1/training/start", json=body)
    assert resp.status_code == 200, resp.text
    written = _written(resp)
    assert (written["learning_stage"], written["task"]) == ("finetune", "disease")


def test_the_preview_states_the_groups_whose_rate_ended_at_zero(client, trained_disease_child,
                                                                 tmp_path):
    """A parent whose rate decayed to 0 is extended only with extend_lr: the
    preview names those groups from its one read, and the API takes the rate."""
    import torch

    http, popen, manager = client
    child = trained_disease_child["disease"]
    assert manager.preview_continuation(str(child), device="cpu", extend_epochs=1,
                                        )["zero_rate_groups"] == []
    payload = torch.load(child, map_location="cpu", weights_only=True)
    for group in payload["optimizer_state_dict"]["param_groups"]:
        group["lr"] = 0.0
    zero = tmp_path / "runs" / "gatv2" / "20261003-000000_disease_zzzzzz" / "checkpoints"
    zero.mkdir(parents=True)
    torch.save(payload, zero / "last.pt")
    preview = manager.preview_continuation(str(zero / "last.pt"), device="cpu",
                                           extend_epochs=1)
    groups = list(range(len(payload["optimizer_state_dict"]["param_groups"])))
    assert preview["ok"] is False and preview["zero_rate_groups"] == groups
    assert "Give extend_lr" in preview["reason"]
    resp = http.post("/api/v1/training/start", json={
        "data_dir": str(trained_disease_child["data_dir"]), "resume_from": str(zero / "last.pt"),
        "device": "cpu", "extend_epochs": 1, "extend_lr": 1e-5})
    assert resp.status_code == 200, resp.text
    assert _written(resp)["extend_lr"] == 1e-5
    del payload["optimizer_state_dict"]
    torch.save(payload, zero / "weights.pt")
    unknown = manager.preview_continuation(str(zero / "weights.pt"), device="cpu",
                                           extend_epochs=1)
    assert unknown["ok"] is False and unknown["zero_rate_groups"] is None, unknown


# ---------------------------------------------------------------------------
# S4: a Fine-tune start checks what each child's process checks against its base
# ---------------------------------------------------------------------------
@pytest.fixture
def base(tmp_path):
    """A one-layer base trained on a generated workspace's graph, its split
    seed recorded (`test_finetune_children`'s, with a split record)."""
    pytest.importorskip("torch")
    from tests.fixtures.generated_workspace import write_generated_workspace
    from tests.unit.test_finetune_children import _derive_fixture, _trained_on
    from tests.unit.test_task_first_selection import BASE_RUN, _put, _record_payload

    workspace, _ = write_generated_workspace(tmp_path / "ws", train_ids=[1, 2], val_ids=[3])
    _derive_fixture(workspace)
    payload = _trained_on(workspace, _record_payload())
    payload["data_roles"] = {**payload["data_roles"], "split": {"seed": 4}}
    return workspace, _put(workspace, "gatv2", BASE_RUN, payload, 100)


def _finetune(base, *tasks, **settings):
    from src.api.services.training_manager import FinetuneInputs, TrainingManager
    from tests.fixtures.job_processes import FakeProcess
    from tests.unit.test_job_runner import Popens

    workspace, path = base
    calls = Popens(FakeProcess(pid=1), FakeProcess(pid=2))
    with patch("subprocess.Popen", calls):
        result = TrainingManager().start_finetune(FinetuneInputs(
            data_dir=str(workspace), finetune_from=str(path), tasks=tasks, device="cpu",
            settings=settings))
    for process in calls.queue:
        process.finish(0)
    return result, calls.calls


def test_the_hop_list_a_run_samples_is_said_once():
    """Review of B2: which hop list a new run samples (the one stated, else
    the one fill's: the stage's and task's reference) is
    `run_plan.new_run_hops`, the fill the training process makes
    (`new_run_values`), which the runner's base start, its Fine-tune check of
    each child against its base and the pages ask, none restating it."""
    from src.utils.run_plan import new_run_hops, new_run_values

    for stage, task in (("base", None), *(("finetune", t) for t in recipe.FINETUNE_TASKS)):
        reference = recipe.reference_settings(stage, task)[0]["num_neighbors"]
        assert new_run_hops(stage, task, {}) == new_run_values(stage, task, {})[
            "num_neighbors"] == reference, (stage, task)
        assert new_run_hops(stage, task, {"num_neighbors": [4, 2], "seed": 3}) == [4, 2]


def test_a_fine_tune_whose_hops_contradict_its_base_is_refused_before_any_child(base):
    result, calls = _finetune(base, "disease")
    assert result["success"] is False and result["invalid_settings"] is True and calls == []
    assert result["error"] == (
        "The Disease child's hop list [15, 10, 5] (its task's reference) does not fit its base's "
        "encoder: num_neighbors needs 1 numbers, one hop per encoder layer, not 3. Nothing was "
        "started.")
    result, calls = _finetune(base, "disease", "gene", num_neighbors=[3])
    assert result["success"] is True and len(calls) == 1


def test_a_fine_tune_whose_split_seed_contradicts_its_base_is_refused(base):
    result, calls = _finetune(base, "disease", num_neighbors=[3], split_seed=9)
    assert result["success"] is False and calls == []
    assert "split_seed 9 differs from the base's 4; a finetune trains on its base's split." \
        in result["error"]
    result, calls = _finetune(base, "disease", num_neighbors=[3], split_seed=4)
    assert result["success"] is True, result


def test_a_fine_tune_spec_is_one_tasks_and_keeps_its_bases_graph_side(base):
    from src.models.bundle.spec import finetune_spec_from_parent, reference_spec
    from tests.unit.test_task_first_selection import _base_spec

    own = finetune_spec_from_parent(_base_spec(), "disease").to_dict()
    result, calls = _finetune(base, "disease", "gene", num_neighbors=[3], model_spec=own)
    assert result["success"] is False and calls == []
    assert "model_spec is one task's spec" in result["error"]
    result, calls = _finetune(base, "disease", "gene", num_neighbors=[3], model_spec=None)
    assert result["success"] is True and len(calls) == 1, "a null spec is no spec"
    assert yaml.safe_load(Path(calls[0][calls[0].index("--config") + 1]).read_text())[
        "model_spec"] is None
    other = reference_spec("finetune", "disease").to_dict()
    result, calls = _finetune(base, "disease", num_neighbors=[3], model_spec=other)
    assert result["success"] is False and calls == []
    assert "a finetune keeps its base's graph view, input, encoder and LP head" in result["error"]
    result, calls = _finetune(base, "disease", num_neighbors=[3],
                              model_spec=finetune_spec_from_parent(_base_spec(),
                                                                   "gene").to_dict())
    assert "model_spec is a finetune spec for gene, not a finetune spec for disease" \
        in result["error"]
    result, calls = _finetune(base, "disease", num_neighbors=[3], model_spec=own)
    assert result["success"] is True and len(calls) == 1
    assert yaml.safe_load(Path(calls[0][calls[0].index("--config") + 1]).read_text())[
        "model_spec"] == own


# ---------------------------------------------------------------------------
# S2: each setting's source, recorded
# ---------------------------------------------------------------------------
def _recipe_record(path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["config"]["effective"]["loaders"]["train"]["recipe"], payload


def test_a_new_run_records_the_operators_the_recipes_and_the_builds_settings(
        trained_disease_child):
    """Decision S2: every setting a new run reads has exactly one source.
    `operator_set` lists every setting the operator set (not only recipe
    keys); `reference_filled` each one the stage's reference filled, with its
    source: the recipe's, and the model spec and a finetune's split seed by
    their stage's rule (the model record holds the spec, the data roles the
    split); `build_defaults` every other one, with this build's default."""
    from scripts.setup_demo import demo_task_training

    for name, stage, task in (("base", "base", None), ("disease", "finetune", "disease")):
        record, payload = _recipe_record(trained_disease_child[name])
        given = set(demo_task_training()[name]) | {"num_epochs", "save_top_k"}
        assert set(record["operator_set"]) == given & new_run_settings(stage, task)
        references = set(recipe.reference_settings(stage, task)[0])
        by_rule = {"model_spec"} | ({"split_seed"} if stage == "finetune" else set())
        assert set(record["reference_filled"]) == (references | by_rule) - given
        if "model_spec" not in given:
            assert "reference" in record["reference_filled"]["model_spec"]
        if stage == "finetune":
            assert "its base's split" in record["reference_filled"]["split_seed"]
        assert set(record["build_defaults"]) == (
            new_run_settings(stage, task) - references - by_rule - given)
        for setting, value in record["build_defaults"].items():
            assert value == payload["config"]["requested"][setting] == tf.by_name(
                setting).default, setting
        sources = (set(record["operator_set"]), set(record["reference_filled"]),
                   set(record["build_defaults"]))
        assert sum(len(s) for s in sources) == len(set().union(*sources)), "one source each"
        assert set().union(*sources) == new_run_settings(stage, task), "a source for each"


def test_a_continued_run_records_inherited_and_stated_over_every_setting(tmp_path,
                                                                        patient_base):
    """The inherited record lists every setting the run reads as the parent's
    or the operator's; nothing is filled from the recipe or a build default."""
    train_model = _train_model()

    ws, parent = patient_base
    run = tmp_path / "resumed"
    config = train_model.TrainConfig(data_dir=str(ws), run_dir=str(run), device="cpu",
                                     learning_stage="base", resume_from=str(parent),
                                     split_seed=0)
    train_model.train(config, {"learning_stage", "device", "resume_from", "split_seed"})
    record, _ = _recipe_record(run / "checkpoints" / "last.pt")
    assert record["reference_filled"] == {} and record["build_defaults"] == {}
    assert record["operator_set"] == ["split_seed"]
    assert set(record["inherited_from_parent"]) == new_run_settings("base") - {"split_seed"}


def test_existing_records_read_unchanged(trained_disease_child, tmp_path):
    """Additive and informational (decision S2): a checkpoint whose recipe
    record has the shape it had before (no `build_defaults`, `operator_set`
    over the recipe's keys only) is read as before. The continuation preview
    decides the same, and the listing (`read_run_checkpoints`, the one read
    the run lists, the results service and the probe read) reads the same
    settings and model record, the recipe record shown as it is recorded:
    no reader decides by it."""
    import torch

    from src.api.services.training_manager import TrainingManager
    from src.models.bundle.record import listed_record
    from src.utils.checkpoint_paths import read_run_checkpoints
    from src.utils.run_plan import recorded_training_settings

    now = torch.load(trained_disease_child["disease"], map_location="cpu", weights_only=True)
    before = copy.deepcopy(now)
    old = before["config"]["effective"]["loaders"]["train"]["recipe"]
    del old["build_defaults"]
    keys = set(recipe.reference_settings("finetune", "disease")[0])
    old["operator_set"] = [name for name in old["operator_set"] if name in keys]
    assert old["operator_set"] != _recipe_record(trained_disease_child["disease"])[0][
        "operator_set"], "the shape before lists fewer settings"
    paths = {}
    for i, (name, payload) in enumerate((("now", now), ("before", before))):
        directory = tmp_path / "runs" / "gatv2" / f"2026100{i}-000000_{name}_aaaaaa" / "checkpoints"
        directory.mkdir(parents=True)
        torch.save(payload, directory / "last.pt")
        paths[name] = directory / "last.pt"

    manager = TrainingManager()
    previews = {name: {key: value for key, value in manager.preview_continuation(
        str(path), device="cpu", extend_epochs=1).items() if key not in ("parent", "sha256")}
        for name, path in paths.items()}
    assert previews["now"]["ok"] is True and previews["before"] == previews["now"]

    rows = {}
    for entry in read_run_checkpoints(tmp_path):
        settings = recorded_training_settings(entry.training)
        shown = settings["effective"]["loaders"]["train"].pop("recipe")
        rows[entry.names[1].split("_")[1]] = (settings, listed_record(entry.model_record), shown)
    assert rows["before"][:2] == rows["now"][:2]
    assert rows["before"][0]["unshown"] == {} and rows["before"][0]["unrecorded"] == []
    assert rows["before"][2] == old
    assert rows["now"][2] == now["config"]["effective"]["loaders"]["train"]["recipe"]
