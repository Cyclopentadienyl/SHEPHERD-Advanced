"""Several finetune children from one base, in sequence, through the one job
runner (README §1, "Children never chain"; contract §9.3 hazard 6).

`start_finetune` checks the parent once (a base with weights, the digest the
caller showed) and every task and setting before anything starts, then runs
one training job per task, in order, each its own run with its own record,
status, failure and cancellation; the runner holds no second queue or lock.
Between children the slot's rules decide: when the GPU lock is held
elsewhere, the remaining children are refused and say by whom. A failed child
does not stop its siblings; a stop ends the running child and cancels the
queued ones. Each child is bound to the parent's digest and given only the
settings asked for, so the task's recipe fills the rest. A bundle parent is
previewed for resume by its own record version and filed under its encoder.

Module: tests/unit/test_finetune_children.py
"""
from __future__ import annotations

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


@pytest.fixture
def base(tmp_path):
    """A workspace holding one base checkpoint and one disease child."""
    workspace = tmp_path / "ws"
    path = _put(workspace, "gatv2", BASE_RUN, _record_payload(), 100)
    child = _put(workspace, "gatv2", "20261002-000000_disease_bbbbbb",
                 _record_payload("disease", parent={"sha256": file_sha256(path),
                                                    "run": BASE_RUN}), 200)
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


@pytest.mark.parametrize("change, fragment", [
    (dict(tasks=()), "one to three tasks"),
    (dict(tasks=("disease", "disease")), "more than once"),
    (dict(tasks=("disease", "genes")), "'genes' is not a task"),
    (dict(settings={"candidate_sample_size": 8}, tasks=("disease", "gene")),
     "candidate_sample_size is not a setting every chosen task reads"),
    (dict(settings={"hidden_dim": 8}), "hidden_dim is not a setting every chosen task reads"),
    (dict(expected_parent_sha256="0" * 64), "changed since it was chosen"),
])
def test_a_finetune_is_refused_before_anything_starts(base, popen, change, fragment):
    calls = popen(FakeProcess(pid=1))
    manager = TrainingManager()
    fields = {"tasks": ("disease",), **change}
    refused = manager.start_finetune(_inputs(base, *fields.pop("tasks"), **fields))
    assert refused["success"] is False and fragment in refused["error"]
    assert "Nothing was started" in refused["error"]
    assert calls.calls == [] and manager.get_status()["finetune"] is None


def test_a_parent_that_is_not_a_base_is_refused(base, popen):
    calls = popen(FakeProcess(pid=1))
    manager = TrainingManager()
    refused = manager.start_finetune(_inputs(base, "gene", finetune_from=str(base[2])))
    assert refused["success"] is False and "children never chain" in refused["error"]
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
              "config": {"requested": {"learning_rate": 1e-4}}}
    target = tmp_path / "paused.pt"
    torch.save(paused, target)
    preview = TrainingManager().preview_continuation(str(target), device="cpu",
                                                     requested={})
    assert preview["ok"] is True, preview.get("reason")
    assert (preview["mode"], preview["architecture"]) == ("resume", "gatv2")


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
