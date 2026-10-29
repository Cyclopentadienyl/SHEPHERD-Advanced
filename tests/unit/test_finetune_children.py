"""Several finetune children from one base, in sequence, through the one job
runner (README §1, "Children never chain"; contract §9.3 hazard 6).

`start_finetune` checks the parent once (a base with weights, the digest the
caller showed) and every task, setting name and setting value before anything
starts, then runs one training job per task, in order, each its own run with
its own record, status, failure and cancellation; the runner holds no second
queue or lock. Between children the slot's rules decide: when the GPU lock is
held elsewhere or the backend is stopping, the remaining children are refused
and say why. A failed child does not stop its siblings, and what fails in
starting one never stops the monitor; a stop ends the running child (shown
"stopping" meanwhile) and cancels the queued ones, also when the child had
just ended on its own; either way the stop names the cancelled children. Each
child is bound to the parent's digest and given only the settings asked for,
so the task's recipe fills the rest. A parent trained on another graph than
the workspace's, and a gene or patients-like-me task whose functional fixture
is absent or derived from another build, are refused before any child starts
(decision 3). A new base run through `start_training` is checked by the same
settings helper and filed under its spec's encoder kind, never the ShepherdGNN
default (G1, G6). A bundle parent is previewed for resume by its own record
version and filed under its encoder, and the job started for it trains.

Module: tests/unit/test_finetune_children.py
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")

import src.api.services.training_manager as tm  # noqa: E402
from src.api.services.training_manager import FinetuneInputs, TrainingManager  # noqa: E402
from src.utils.fingerprint import file_sha256  # noqa: E402
from tests.fixtures.job_processes import FakeProcess  # noqa: E402
from tests.unit.test_job_runner import Popens  # noqa: E402
from tests.unit.test_task_first_selection import BASE_RUN, _put, _record_payload  # noqa: E402


@pytest.fixture(autouse=True)
def _no_device_setting(monkeypatch):
    monkeypatch.delenv("SHEPHERD_DEVICE", raising=False)


@pytest.fixture
def popen(monkeypatch):
    holder = {}

    def install(*processes):
        holder["popen"] = Popens(*processes)
        monkeypatch.setattr("subprocess.Popen", holder["popen"])
        return holder["popen"]

    yield install
    for process in holder.get("popen", Popens()).queue:
        process.finish(0)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One generated workspace, its manifest binding its graph, built once."""
    from tests.fixtures.generated_workspace import write_generated_workspace

    root, _manifest = write_generated_workspace(tmp_path_factory.mktemp("built") / "ws",
                                                train_ids=[1, 2], val_ids=[3])
    return root


def _trained_on(workspace, payload):
    """`payload` recording the workspace's graph as the one it trained on."""
    from src.kg.artifacts import read_split_manifest

    kg = read_split_manifest(workspace).manifest["artifacts"]["kg"]
    return {**payload, "data_roles": {**payload["data_roles"], "kg": kg}}


def _derive_fixture(workspace, manifest_sha256=None):
    """A functional fixture record bound to this build of the workspace, or to
    the build `manifest_sha256` names, through the fixture's own publisher."""
    from src.kg.artifacts import read_split_manifest
    from src.kg.task_fixture import publish_gene_fixture

    sha = manifest_sha256 or read_split_manifest(workspace).identity.sha256
    publish_gene_fixture(workspace, {"train": [], "val": []},
                         {"workspace": {"split_manifest": sha}}, replace=True)


@pytest.fixture
def base(tmp_path, built):
    """A workspace with a functional fixture derived for it, holding one base
    checkpoint trained on its graph and one disease child."""
    workspace = tmp_path / "ws"
    shutil.copytree(built, workspace)
    _derive_fixture(workspace)
    path = _put(workspace, "gatv2", BASE_RUN, _trained_on(workspace, _record_payload()), 100)
    child = _put(workspace, "gatv2", "20261002-000000_disease_bbbbbb",
                 _trained_on(workspace, _record_payload(
                     "disease", parent={"sha256": file_sha256(path), "run": BASE_RUN})), 200)
    return workspace, path, child


def _inputs(base, *tasks, **kw):
    workspace, path, _ = base
    fields = dict(data_dir=str(workspace), finetune_from=str(path), tasks=tasks, device="cpu")
    fields.update(kw)
    return FinetuneInputs(**fields)


def _wait(predicate, seconds=5.0):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.01)


def _children(manager):
    return [(c["task"], c["status"]) for c in manager.get_status()["finetune"]["children"]]


def _config(argv):
    return yaml.safe_load(Path(argv[argv.index("--config") + 1]).read_text())


def test_children_run_in_order_and_a_failure_does_not_stop_its_siblings(base, popen):
    first, second, third = FakeProcess(pid=1), FakeProcess(pid=2), FakeProcess(pid=3)
    calls = popen(first, second, third)
    manager = TrainingManager()
    started = manager.start_finetune(_inputs(base, "disease", "gene", "patients_like_me",
                                             settings={"num_epochs": 2}))
    assert started["success"], started
    assert _children(manager) == [("disease", "running"), ("gene", "queued"),
                                  ("patients_like_me", "queued")]
    sha = file_sha256(base[1])
    argv = calls.calls[0]
    assert argv[argv.index("--expect-parent-sha256") + 1] == sha
    config = _config(argv)
    assert config == {"data_dir": str(base[0]), "learning_stage": "finetune", "task": "disease",
                      "finetune_from": str(base[1]), "device": "cpu", "num_epochs": 2,
                      "run_dir": config["run_dir"]}, "only what was asked: the recipe fills the rest"
    assert "/runs/gatv2/" in config["run_dir"], "filed under the parent's encoder"

    first.finish(1)  # the disease child fails ...
    _wait(lambda: len(calls.calls) == 2)
    assert _children(manager)[:2] == [("disease", "failed"), ("gene", "running")]
    assert _config(calls.calls[1])["task"] == "gene"
    second.finish(0)  # ... and its siblings still run
    _wait(lambda: len(calls.calls) == 3)
    third.finish(0)
    _wait(lambda: manager.active_job() is None)
    status = manager.get_status()["finetune"]
    assert [(c["task"], c["status"]) for c in status["children"]] == [
        ("disease", "failed"), ("gene", "completed"), ("patients_like_me", "completed")]
    assert status["parent_sha256"] == sha and status["cancelled"] is False
    runs = {c["run_dir"] for c in status["children"]}
    assert len(runs) == 3, "each child is its own run"


def test_a_stop_ends_the_running_child_and_cancels_the_queued_ones(base, popen):
    first = FakeProcess(pid=1)
    calls = popen(first, FakeProcess(pid=2))
    manager = TrainingManager()
    assert manager.start_finetune(_inputs(base, "disease", "gene"))["success"]
    stopped = manager.stop_training()
    assert stopped["success"] and stopped["status"] == "stopped"
    assert stopped["cancelled_children"] == ["gene"], "a normal stop names them too (G7)"
    _wait(lambda: manager.active_job() is None)
    assert _children(manager) == [("disease", "stopped"), ("gene", "cancelled")]
    assert manager.get_status()["finetune"]["cancelled"] is True
    assert len(calls.calls) == 1, "no queued child starts after a stop"


def test_the_gpu_lock_taken_between_children_refuses_the_rest(base, popen, monkeypatch):
    first = FakeProcess(pid=1)
    calls = popen(first, FakeProcess(pid=2))
    manager = TrainingManager()
    assert manager.start_finetune(_inputs(base, "disease", "gene"))["success"]
    monkeypatch.setattr(tm, "lock_state", lambda: {"problem": None, "path": "/x/gpu.lock",
                                                   "held_by": "pid 999 (a CLI run)"})
    first.finish(0)
    _wait(lambda: manager.active_job() is None)
    children = manager.get_status()["finetune"]["children"]
    assert [(c["task"], c["status"]) for c in children] == [("disease", "completed"),
                                                           ("gene", "refused")]
    assert "pid 999 (a CLI run)" in children[1]["error"]
    assert len(calls.calls) == 1


def test_the_backend_stopping_between_children_refuses_the_rest(base, popen):
    first = FakeProcess(pid=1)
    calls = popen(first, FakeProcess(pid=2))
    manager = TrainingManager()
    assert manager.start_finetune(_inputs(base, "disease", "gene"))["success"]
    manager.close()
    first.finish(0)
    _wait(lambda: manager.active_job() is None)
    children = manager.get_status()["finetune"]["children"]
    assert [(c["task"], c["status"]) for c in children] == [("disease", "completed"),
                                                           ("gene", "refused")]
    assert children[1]["error"] == TrainingManager.STOPPING and len(calls.calls) == 1


def test_a_stop_after_the_child_ended_on_its_own_still_cancels_the_queue(base, popen):
    """The stop arrives while the monitor is still reading the ended child: it
    stopped nothing, but cancelled the queued children, and says both."""
    import threading

    first = FakeProcess(pid=1)
    calls = popen(first, FakeProcess(pid=2), FakeProcess(pid=3))
    manager = TrainingManager()
    assert manager.start_finetune(_inputs(base, "disease", "gene", "patients_like_me"))["success"]
    gate = threading.Event()
    manager._end_group = lambda job: gate.wait(10)
    first.finish(0)
    answer = {}
    stopper = threading.Thread(target=lambda: answer.update(manager.stop_training()))
    stopper.start()
    _wait(lambda: manager.get_status()["finetune"]["cancelled"])
    gate.set()
    stopper.join(10)
    assert answer["success"] is True and answer["status"] == "completed"
    assert answer["cancelled_children"] == ["gene", "patients_like_me"]
    assert "had already ended on its own" in answer["message"]
    assert answer["message"].endswith(
        "The queued children (gene, patients_like_me) were cancelled.")
    _wait(lambda: manager.active_job() is None)
    assert _children(manager) == [("disease", "completed"), ("gene", "cancelled"),
                                  ("patients_like_me", "cancelled")]
    assert len(calls.calls) == 1


def test_a_child_being_stopped_shows_stopping(base, popen):
    first = FakeProcess(pid=1)
    popen(first, FakeProcess(pid=2))
    manager = TrainingManager()
    assert manager.start_finetune(_inputs(base, "disease", "gene"))["success"]
    seen = []

    def end_process(job, process):
        seen.append(_children(manager))
        process.finish(-2)

    manager._end_process = end_process
    assert manager.stop_training()["success"]
    assert seen == [[("disease", "stopping"), ("gene", "cancelled")]]
    assert _children(manager) == [("disease", "stopped"), ("gene", "cancelled")]


def test_what_fails_in_starting_a_child_never_stops_the_monitor(base, popen, monkeypatch):
    """A configuration that cannot be written fails that child, and the next
    still starts; an error raised while starting one refuses the rest. Either
    way the job ends and frees the slot."""
    first, third = FakeProcess(pid=1), FakeProcess(pid=3)
    calls = popen(first, third)
    manager = TrainingManager()
    assert manager.start_finetune(_inputs(base, "disease", "gene", "patients_like_me"))["success"]
    real, made = tm.tempfile.mkstemp, []

    def mkstemp(*args, **kwargs):
        made.append(1)
        if len(made) == 1:
            raise OSError(28, "No space left on device")
        return real(*args, **kwargs)

    monkeypatch.setattr(tm.tempfile, "mkstemp", mkstemp)
    first.finish(0)
    _wait(lambda: len(calls.calls) == 2)
    children = manager.get_status()["finetune"]["children"]
    assert [(c["task"], c["status"]) for c in children] == [
        ("disease", "completed"), ("gene", "failed"), ("patients_like_me", "running")]
    assert "Its configuration could not be written" in children[1]["error"]
    third.finish(0)
    _wait(lambda: manager.active_job() is None)

    broken = FakeProcess(pid=4)
    calls = popen(broken)
    manager = TrainingManager()
    assert manager.start_finetune(_inputs(base, "disease", "gene"))["success"]

    def launch(sequence, child):
        raise RuntimeError("the run directory vanished")

    manager._launch_child_locked = launch
    broken.finish(0)
    _wait(lambda: manager.active_job() is None)
    children = manager.get_status()["finetune"]["children"]
    assert [(c["task"], c["status"]) for c in children] == [("disease", "completed"),
                                                           ("gene", "refused")]
    assert "could not be started (the run directory vanished)" in children[1]["error"]
    assert manager.get_status()["status"] == "completed"


@pytest.mark.parametrize("change, fragment", [
    (dict(tasks=()), "one to three tasks"),
    (dict(tasks=("disease", "disease")), "more than once"),
    (dict(tasks=("disease", "genes")), "'genes' is not a task"),
    (dict(settings={"candidate_sample_size": 8}, tasks=("disease", "gene")),
     "candidate_sample_size is not a setting a Fine-tune start of Disease and Gene takes"),
    (dict(settings={"hidden_dim": 8}),
     "hidden_dim is not a setting a Fine-tune start of Disease takes"),
    (dict(settings={"save_top_k": 2}), "save_top_k is not a setting"),
    (dict(settings={"num_epochs": 0}), "num_epochs must be between 1 and 10000, not 0."),
    (dict(settings={"learning_rate": "fast"}), "learning_rate must be a number, not 'fast'"),
    (dict(settings={"use_amp": "false"}), "use_amp must be true or false, not 'false'"),
    (dict(settings={"num_neighbors": [3, 0]}), "num_neighbors items must be 1 or more"),
    (dict(expected_parent_sha256="0" * 64), "changed since it was chosen"),
])
def test_a_finetune_is_refused_before_anything_starts(base, popen, change, fragment):
    calls = popen(FakeProcess(pid=1))
    manager = TrainingManager()
    fields = {"tasks": ("disease",), **change}
    refused = manager.start_finetune(_inputs(base, *fields.pop("tasks"), **fields))
    assert refused["success"] is False and fragment in refused["error"]
    assert "Nothing was started" in refused["error"]
    assert refused.get("invalid_settings", False) == ("settings" in change)
    assert calls.calls == [] and manager.get_status()["finetune"] is None


@pytest.mark.parametrize("fixture, fragment", [
    ("absent", "No gene task fixture"),
    ("another build", "derived from another build of this workspace"),
])
def test_a_fixture_task_is_refused_without_its_fixture_and_disease_alone_starts(
        base, popen, fixture, fragment):
    """Decision 3 (G5): a gene or patients-like-me child would fail in its own
    process while its siblings ran; the start refuses it before any child."""
    from src.kg.task_fixture import fixture_dir

    workspace = base[0]
    if fixture == "absent":
        shutil.rmtree(fixture_dir(workspace))
    else:
        _derive_fixture(workspace, "0" * 64)
    job = FakeProcess(pid=1)
    calls = popen(job)
    manager = TrainingManager()
    # The tasks named as every page names them (`caveats.TASK_NAMES`), never
    # by their ids (K6 closing review: "gene and patients_like_me train on …").
    for tasks, lead in ((("gene",), "Gene trains"),
                        (("disease", "patients_like_me"), "Patients-like-me trains"),
                        (("gene", "patients_like_me"), "Gene and Patients-like-me train")):
        refused = manager.start_finetune(_inputs(base, *tasks))
        assert refused["success"] is False and fragment in refused["error"]
        assert refused["error"].startswith(f"{lead} on the gene task's functional fixture: ")
        assert "patients_like_me" not in refused["error"] and "gene train" not in refused["error"]
        assert refused["error"].endswith("Nothing was started.")
    assert calls.calls == [] and manager.get_status()["finetune"] is None
    started = manager.start_finetune(_inputs(base, "disease"))
    assert started["success"], started
    assert _children(manager) == [("disease", "running")]
    job.finish(0)


def test_a_parent_trained_on_another_graph_is_refused_before_any_child_starts(base, popen):
    """Critic (a)1, README §8 refusal 7: the graph is compared at the start,
    not by each child's process one after another."""
    other = _put(base[0], "gatv2", "20261003-000000_base_cccccc", _record_payload(), 300)
    calls = popen(FakeProcess(pid=1))
    manager = TrainingManager()
    refused = manager.start_finetune(_inputs(base, "disease", "gene", finetune_from=str(other)))
    assert refused["success"] is False
    assert "was trained on another knowledge graph" in refused["error"]
    assert "Nothing was started" in refused["error"]
    assert calls.calls == [] and manager.get_status()["finetune"] is None


def test_a_new_base_run_is_filed_under_its_spec_encoder_with_only_what_was_sent(base, popen):
    """G1: a base run named no conv type, so it was filed under the ShepherdGNN
    default ``runs/gat`` while its record said gatv2. Its configuration holds
    what was sent and nothing more, so the recipe fills the rest."""
    import dataclasses

    from src.models.bundle import reference_spec

    jobs = [FakeProcess(pid=1), FakeProcess(pid=2), FakeProcess(pid=3)]
    calls = popen(*jobs)
    manager = TrainingManager()
    sent = {"data_dir": str(base[0]), "learning_stage": "base", "device": "cpu",
            "num_epochs": 5, "run_name": "short"}
    started = manager.start_training(sent)
    assert started["success"], started
    assert "/runs/gatv2/" in started["run_dir"]
    config = _config(calls.calls[0])
    assert config == {**sent, "run_dir": started["run_dir"]}
    jobs[0].finish(0)
    _wait(lambda: manager.active_job() is None)

    spec = reference_spec("base")
    sage = dataclasses.replace(spec, encoder=dataclasses.replace(spec.encoder, kind="sage"))
    started = manager.start_training({"data_dir": str(base[0]), "learning_stage": "base",
                                      "device": "cpu", "model_spec": sage.to_dict()})
    assert started["success"], started
    assert "/runs/sage/" in started["run_dir"], "the spec's own encoder kind"
    jobs[1].finish(0)
    _wait(lambda: manager.active_job() is None)

    # A ShepherdGNN start is decided as before: its conv type, its settings.
    started = manager.start_training({"data_dir": str(base[0]), "conv_type": "gat",
                                      "hidden_dim": 8})
    assert started["success"], started
    assert "/runs/gat/" in started["run_dir"]
    jobs[2].finish(0)


@pytest.mark.parametrize("change, fragment", [
    (dict(hidden_dim=8), "hidden_dim is not a setting a base run takes"),
    (dict(task_loss_weight=0.5), "task_loss_weight is not a setting a base run takes"),
    (dict(conv_type="gat"), "conv_type is not a setting a base run takes"),
    (dict(task="disease"), "task is not a setting a base run takes"),
    (dict(num_epochs=0), "num_epochs must be between 1 and 10000, not 0."),
    (dict(learning_rate="fast"), "learning_rate must be a number, not 'fast'"),
    (dict(early_stopping_patience=None, seed=None), "seed must be set, not null"),
    # One hop per layer of the spec's encoder (the reference recipe's three):
    # refused by the start, not by the process after a run directory exists.
    (dict(num_neighbors=[15, 10]), "num_neighbors needs 3 numbers, one hop per encoder layer, "
                                   "not 2"),
    (dict(learning_stage="finetune"), "A finetune is started as a Fine-tune start"),
    (dict(learning_stage="pretrain"), "'pretrain' is not a learning stage a new run starts"),
    (dict(model_spec={"spec_version": 0}), "The model_spec cannot be read"),
])
def test_a_new_base_run_is_refused_whole_before_anything_starts(base, popen, change, fragment):
    """G6, BACKLOG 54: every YAML key counts as the operator's, so a key outside
    the recipe and the run keys, or a value its field spec refuses, is refused
    by the start (the helper `start_finetune` shares), not passed on."""
    calls = popen(FakeProcess(pid=1))
    manager = TrainingManager()
    refused = manager.start_training({"data_dir": str(base[0]), "learning_stage": "base",
                                      "device": "cpu", **change})
    assert refused["success"] is False and refused["invalid_settings"] is True
    assert fragment in refused["error"] and refused["error"].endswith("Nothing was started.")
    assert calls.calls == [] and manager.run_dir is None
    assert not (base[0] / "runs" / "gat").exists()


def test_a_parent_that_is_not_a_base_is_refused(base, popen, tmp_path):
    calls = popen(FakeProcess(pid=1))
    manager = TrainingManager()
    refused = manager.start_finetune(_inputs(base, "gene", finetune_from=str(base[2])))
    assert refused["success"] is False and "children never chain" in refused["error"]
    weightless = tmp_path / "weightless.pt"
    torch.save({**torch.load(base[1], map_location="cpu", weights_only=True),
                "state_dict": None}, weightless)
    refused = manager.start_finetune(_inputs(base, "gene", finetune_from=str(weightless)))
    assert refused["success"] is False
    assert "The parent weightless.pt holds no model weights." in refused["error"]
    assert calls.calls == []


def test_a_finetune_waits_for_no_one_while_the_slot_is_busy(base, popen, tmp_path):
    running = FakeProcess(pid=1)
    calls = popen(running)
    manager = TrainingManager()
    assert manager.start_training({"data_dir": str(tmp_path / "ws")})["success"]
    refused = manager.start_finetune(_inputs(base, "disease"))
    assert refused["success"] is False and "already running" in refused["error"]
    assert len(calls.calls) == 1
    running.finish(0)


def test_a_bundle_parent_is_previewed_by_its_own_record(base, tmp_path):
    """A paused bundle run resumes through the runner's preview, filed under
    its encoder kind, never under the ShepherdGNN default."""
    workspace, path, _ = base
    paused = {**torch.load(path, map_location="cpu", weights_only=True),
              "checkpoint_kind": "training_state",
              "progress": {"end": "paused", "completed_epochs": 1,
                           "stage": {"kind": "original", "start_epoch": 0, "target_epochs": 2,
                                     "schedule": {"type": "constant"}}},
              "config": {"requested": {"learning_rate": 1e-4}}, "training_rules": 1}
    target = tmp_path / "paused.pt"
    torch.save(paused, target)
    preview = TrainingManager().preview_continuation(str(target), device="cpu",
                                                     requested={})
    assert preview["ok"] is True, preview.get("reason")
    assert (preview["mode"], preview["architecture"]) == ("resume", "gatv2")


def test_a_paused_bundle_run_resumed_through_the_runner_trains(tmp_path, popen, monkeypatch):
    """The runner's job for a bundle resume names the parent's stage, so the
    training process it starts runs the bundle path and resumes the run (its
    configuration and arguments, run here in process)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import train_model
    from src.models.bundle.record import load_bundle
    from tests.fixtures.base_workspace import write_base_workspace
    from tests.unit.test_train_base_stage import _config as base_config

    workspace = write_base_workspace(tmp_path / "ws")
    train_model.train(*base_config(workspace, tmp_path / "paused", pause_after_epoch=1))
    paused = tmp_path / "paused" / "checkpoints" / "last.pt"
    import subprocess

    real_popen = subprocess.Popen
    job = FakeProcess(pid=1)
    calls = popen(job)
    manager = TrainingManager()
    started = manager.start_training({"data_dir": str(workspace), "resume_from": str(paused),
                                      "device": "cpu"})
    assert started["success"], started
    argv = calls.calls[0]
    config = _config(argv)
    assert (config["learning_stage"], "task" in config) == ("base", False)
    monkeypatch.setattr(subprocess, "Popen", real_popen)  # the run's own tools
    monkeypatch.setattr(sys, "argv", ["train_model.py", *argv[2:]])
    assert train_model._train_main(train_model.parse_args()) == 0
    resumed = torch.load(Path(config["run_dir"]) / "checkpoints" / "last.pt",
                         map_location="cpu", weights_only=True)
    _, record = load_bundle(resumed, learning_stage="base")
    assert record.operation == "resume" and resumed["progress"]["completed_epochs"] == 2
    job.finish(0)


def test_the_api_is_strict_and_gated(base, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app
    from src.api.services import training_manager as service
    from tests.fixtures.operator_access import OPERATOR_BASE_URL, operator_mode

    seen = []
    monkeypatch.setattr(service.training_manager, "start_finetune",
                        lambda inputs, **kw: seen.append(inputs) or {"success": True})
    body = {"data_dir": str(base[0]), "finetune_from": str(base[1]), "tasks": ["disease"]}
    assert TestClient(app, base_url="http://192.0.2.10").post(
        "/api/v1/training/finetune", json=body).status_code == 403
    operator_mode(monkeypatch)
    client = TestClient(app, base_url=OPERATOR_BASE_URL)
    assert client.post("/api/v1/training/finetune", json=body).status_code == 200
    assert seen[0].tasks == ("disease",) and seen[0].finetune_from == str(base[1])
    for extra in ({"task": "disease"}, {"learning_stage": "finetune"}):
        assert client.post("/api/v1/training/finetune",
                           json={**body, **extra}).status_code == 422
    assert client.post("/api/v1/training/start",
                       json={"data_dir": "x", "learning_stage": "base"}).status_code == 422
    assert len(seen) == 1
    # It reads the whole base: FastAPI runs a plain def in its threadpool.
    import inspect

    from src.api.routes import training

    assert not inspect.iscoroutinefunction(training.start_finetune)
