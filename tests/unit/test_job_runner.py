"""
The job runner (M5): training and test runs, one GPU job at a time.
===================================================================
`training_manager` reshaped in place: one GPU slot that a running **or
stopping** job of either type holds; a stop that never crosses job types; a
test run's command line built from the server's lists and settings only; and,
once a test run has ended, what it left read from the disk, never from its
exit status alone. Processes are `FakeProcess` stand-ins, except where a real
child must hold the GPU lock.

Module: tests/unit/test_job_runner.py
"""
from __future__ import annotations

import inspect
import os
import signal
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

import src.api.services.training_manager as tm
from src.api.services.training_manager import TestRunInputs, TrainingManager, build_test_argv
from src.utils.gpu_lock import (
    LOCK_FILE_ENV,
    LOCK_FILE_NAME,
    WORKSPACES_ROOT_ENV,
    GpuLock,
    probe,
)
from tests.fixtures.job_processes import FakeProcess

SHA_CKPT = "c" * 64
SHA_VERSION = "e" * 64


@pytest.fixture(autouse=True)
def _no_lock_settings(monkeypatch):
    """Each test names the lock it uses; the developer's environment does not."""
    monkeypatch.delenv(LOCK_FILE_ENV, raising=False)
    monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)
    monkeypatch.delenv("SHEPHERD_DEVICE", raising=False)


class Popens:
    """`subprocess.Popen` replaced: each call takes the next fake process."""

    def __init__(self, *processes: FakeProcess) -> None:
        self.queue = list(processes)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        return self.queue.pop(0)


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
def inputs(tmp_path):
    registry = tmp_path / "registry"
    registry.mkdir()
    workspace = tmp_path / "workspaces" / "ws"
    return TestRunInputs(
        registry=registry, version_label="mygene2-2026-10", version_sha256=SHA_VERSION,
        workspace=workspace, checkpoint=workspace / "runs" / "hgt" / "r1" / "checkpoints" / "best.pt",
        checkpoint_sha256=SHA_CKPT)


def reports(inputs: TestRunInputs) -> Path:
    return inputs.registry / "versions" / inputs.version_label / "reports"


def wait_for(predicate, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.01)


def ended(manager: TrainingManager, job_type: str) -> None:
    assert manager._last[job_type].done.wait(5), f"the {job_type} job did not end"


# --------------------------------------------------------------------------- one slot
def test_a_start_while_a_stop_is_still_waiting_is_refused(tmp_path, popen, monkeypatch, inputs):
    """The race the map reproduced: a start accepted during `stopping` launched
    a second process while the first was still ending."""
    monkeypatch.setattr(tm, "STOP_GRACE_SECONDS", 5.0)
    first = FakeProcess(pid=1, ignore_sigint=True)
    calls = popen(first, FakeProcess(pid=2))
    manager = TrainingManager()
    assert manager.start_training({"data_dir": str(tmp_path / "ws")})["success"]

    stopper = threading.Thread(target=manager.stop_training)
    stopper.start()
    wait_for(lambda: manager.get_status()["status"] == "stopping")

    again = manager.start_training({"data_dir": str(tmp_path / "ws")})
    assert again["success"] is False and "Training is still stopping" in again["error"]
    test = manager.start_test(inputs)
    assert test["success"] is False and "Training is still stopping" in test["error"]
    assert len(calls.calls) == 1, "no second process while the first is ending"

    first.finish(-2)
    stopper.join(10)
    assert manager.get_status()["status"] == "stopped"
    assert manager.active_job() is None


def test_one_gpu_job_at_a_time_across_types(tmp_path, popen, inputs):
    training, test = FakeProcess(pid=1), FakeProcess(pid=2)
    calls = popen(training, test)
    manager = TrainingManager()
    assert manager.start_training({"data_dir": str(tmp_path / "ws")})["success"]

    refused = manager.start_test(inputs)
    assert refused["success"] is False
    assert refused["error"].startswith("Training is already running")
    assert "a test run cannot start until it has ended" in refused["error"]

    training.finish(0)
    ended(manager, "training")
    assert manager.start_test(inputs)["success"]
    refused = manager.start_training({"data_dir": str(tmp_path / "ws")})
    assert refused["success"] is False
    assert refused["error"].startswith("A test run is already running")
    assert len(calls.calls) == 2
    test.finish(0)


def test_a_stop_never_crosses_job_types(tmp_path, popen, inputs):
    training, test = FakeProcess(pid=1), FakeProcess(pid=2)
    popen(training, test)
    manager = TrainingManager()
    manager.start_training({"data_dir": str(tmp_path / "ws")})

    refused = manager.cancel_test()
    assert refused["success"] is False
    assert refused["error"] == ("No test run is running: the GPU job running is training, "
                                "which this stop does not reach")
    assert training.signals == [] and manager.get_status()["status"] == "running"

    training.finish(0)
    ended(manager, "training")
    manager.start_test(inputs)
    refused = manager.stop_training()
    assert refused["success"] is False and refused["error"].startswith("No training is running")
    assert test.signals == [] and manager.get_test_status()["status"] == "running"
    test.finish(0)


def test_the_training_status_shows_a_test_run_in_the_gpu_slot_never_as_training(
        tmp_path, popen, inputs):
    test = FakeProcess(pid=9)
    popen(test)
    manager = TrainingManager()
    manager.start_test(inputs)
    status = manager.get_status()
    assert status["status"] == "idle" and status["pid"] is None
    assert status["gpu"]["busy"] is True and status["gpu"]["job"]["job_type"] == "test"
    test.finish(0)
    ended(manager, "test")
    assert manager.get_status()["gpu"]["busy"] is False


# --------------------------------------------------------------------------- the GPU lock
def test_a_start_is_refused_while_another_process_holds_the_gpu_lock(tmp_path, popen,
                                                                     monkeypatch, inputs):
    root = tmp_path / "workspaces"
    root.mkdir(exist_ok=True)
    monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(root))
    calls = popen()
    manager = TrainingManager()
    with GpuLock(root / LOCK_FILE_NAME, job_type="training", command="train_model.py").acquire():
        for result in (manager.start_training({"data_dir": str(root / "ws")}),
                       manager.start_test(inputs)):
            assert result["success"] is False
            assert "is held by a training job (train_model.py" in result["error"]
            assert result["error"].endswith("Nothing was started.")
        assert manager.get_status()["gpu"]["lock"]["held_by"]
    assert calls.calls == []


def test_an_unusable_lock_setting_refuses_every_start(tmp_path, popen, monkeypatch):
    monkeypatch.setenv(LOCK_FILE_ENV, "relative.lock")
    calls = popen()
    result = TrainingManager().start_training({"data_dir": str(tmp_path / "ws")})
    assert result["success"] is False and "The GPU lock cannot be used" in result["error"]
    assert calls.calls == []


# --------------------------------------------------------------------------- stop
def test_a_stop_escalates_sigint_sigterm_sigkill(tmp_path, popen, monkeypatch):
    for name in ("STOP_GRACE_SECONDS", "TERMINATE_GRACE_SECONDS", "KILL_GRACE_SECONDS"):
        monkeypatch.setattr(tm, name, 0.1)
    stubborn = FakeProcess(pid=3, ignore_sigint=True, ignore_terminate=True)
    popen(stubborn)
    manager = TrainingManager()
    manager.start_training({"data_dir": str(tmp_path / "ws")})
    result = manager.stop_training()
    assert stubborn.signals == ["SIGINT", "SIGTERM", "SIGKILL"]
    assert result["success"] is True and result["status"] == "stopped"
    assert manager.get_status()["exit_code"] == -9


def test_a_cancelled_test_run_is_neither_failed_nor_completed(popen, inputs):
    test = FakeProcess(pid=4)
    popen(test)
    manager = TrainingManager()
    manager.start_test(inputs)
    result = manager.cancel_test()
    assert result["success"] is True and result["status"] == "cancelled"
    assert result["outcome"]["kept"] is False, "no report directory was created"
    status = manager.get_test_status()
    assert status["status"] == "cancelled" and status["exit_code"] == -2
    assert status["exit_meaning"] == "ended by SIGINT"


def test_a_cancel_after_the_run_finished_cleanly_is_not_called_cancelled(popen, inputs):
    test = FakeProcess(pid=5, on_sigint=0)
    popen(test)
    manager = TrainingManager()
    manager.start_test(inputs)
    assert manager.cancel_test()["status"] == "completed"


def test_the_stop_route_does_not_run_on_the_event_loop():
    from src.api.routes import training

    assert not inspect.iscoroutinefunction(training.stop_training)


# --------------------------------------------------------------------------- the test job's inputs
def test_the_test_command_line_comes_from_the_inputs_only(inputs):
    argv = build_test_argv(inputs, None)
    assert argv[:2] == [sys.executable, str(tm.PROJECT_ROOT / "scripts" / "measure_scorer.py")]
    assert argv[2:] == [
        "--checkpoint", str(inputs.checkpoint), "--data-dir", str(inputs.workspace),
        "--cohort", "mygene2-2026-10", "--registry", str(inputs.registry),
        "--expect-checkpoint-sha256", SHA_CKPT, "--expect-version-sha256", SHA_VERSION]
    assert "--batch-size" not in argv, "the CLI's default stays its one definition"


def test_the_operators_choices_and_the_evidence_setting_reach_the_command_line(inputs):
    from dataclasses import replace

    chosen = replace(inputs, record=True, deterministic=True,
                     evidence_dirs=(Path("/srv/evidence/a"), Path("/srv/evidence/b")))
    argv = build_test_argv(chosen, "cpu")
    assert argv[-8:] == ["--training-evidence", "/srv/evidence/a", "--training-evidence",
                         "/srv/evidence/b", "--device", "cpu", "--record", "--deterministic"]


@pytest.mark.parametrize("setting, device", [(None, None), ("auto", None), ("cpu", "cpu"),
                                             ("cuda", "cuda")])
def test_the_device_is_the_deployments(monkeypatch, setting, device):
    if setting is not None:
        monkeypatch.setenv("SHEPHERD_DEVICE", setting)
    assert tm.deployment_device() == device


def test_a_deployment_device_the_cli_cannot_take_refuses_the_start(monkeypatch, popen, inputs):
    monkeypatch.setenv("SHEPHERD_DEVICE", "cuda:1")
    calls = popen()
    result = TrainingManager().start_test(inputs)
    assert result["success"] is False and "'cuda:1'" in result["error"]
    assert result["error"].endswith("Nothing was started.") and calls.calls == []


@pytest.mark.parametrize("change, says", [
    ({"checkpoint_sha256": "c" * 12}, "not a full SHA-256"),
    ({"version_sha256": "E" * 64}, "not a full SHA-256"),
    ({"workspace": Path("relative/ws")}, "not an absolute path"),
    ({"evidence_dirs": (Path("evidence"),)}, "not an absolute path"),
    ({"version_label": "../x"}, "cohort label is refused"),
    ({"registry": Path("registry")}, "not an absolute path"),
])
def test_inputs_of_the_wrong_shape_are_refused_before_launch(popen, inputs, change, says):
    from dataclasses import replace

    calls = popen()
    result = TrainingManager().start_test(replace(inputs, **change))
    assert result["success"] is False and says in result["error"]
    assert calls.calls == []


# --------------------------------------------------------------------------- what a test run left
class Reader:
    """A report-state reader that records what it was asked."""

    def __init__(self, kept, registered, reason=""):
        self.answer = {"kept": kept, "registered": registered, "reason": reason}
        self.asked = []

    def __call__(self, report, checkpoint):
        self.asked.append((report, checkpoint))
        return self.answer


def _run(manager, popen, inputs, *, code=0, make=(), existing=()):
    for name in existing:
        (reports(inputs) / name).mkdir(parents=True)
    process = FakeProcess(pid=6)
    popen(process)
    assert manager.start_test(inputs)["success"]
    for name in make:
        (reports(inputs) / name).mkdir(parents=True)
    process.finish(code)
    ended(manager, "test")
    return manager.get_test_status()


def test_the_report_this_run_created_is_read_from_the_disk(popen, inputs):
    reader = Reader(True, False, "refused: a contradiction")
    run_id = f"20261020T100000Z-{SHA_CKPT[:12]}-0a1b2c3d"
    status = _run(TrainingManager(report_state=reader), popen, inputs, code=3, make=[run_id],
                  existing=[f"20261019T100000Z-{SHA_CKPT[:12]}-ffffffff"])
    run_dir = reports(inputs) / run_id
    assert reader.asked == [(run_dir / "report.json", inputs.checkpoint)]
    assert status["status"] == "failed" and status["exit_code"] == 3
    assert status["exit_meaning"] == "measured and kept, but not registered"
    outcome = status["outcome"]
    assert outcome["report_dir"] == str(run_dir) and outcome["kept"] is True
    assert outcome["registered"] is False
    assert outcome["summary"] == (f"A report was kept at {run_dir}; it is not registered. "
                                  "(refused: a contradiction)")


def test_kept_and_registered_come_from_the_disk_not_the_exit_status(popen, inputs):
    """Exit 0, but the run directory holds no report.json: not called kept."""
    reader = Reader(False, False)
    status = _run(TrainingManager(report_state=reader), popen, inputs, code=0,
                  make=[f"20261020T100000Z-{SHA_CKPT[:12]}-0a1b2c3d"])
    assert status["status"] == "completed"
    assert status["outcome"]["kept"] is False and "an interrupted write" in \
        status["outcome"]["summary"]


def test_no_new_report_directory_means_nothing_was_kept(popen, inputs):
    reader = Reader(True, True)
    status = _run(TrainingManager(report_state=reader), popen, inputs, code=1,
                  make=[f"20261020T100000Z-{'d' * 12}-0a1b2c3d"])  # another checkpoint's
    assert reader.asked == []
    assert status["outcome"]["kept"] is False and status["outcome"]["registered"] is False
    assert status["status"] == "failed"


def test_two_candidate_reports_are_never_guessed_between(popen, inputs):
    reader = Reader(True, True)
    names = [f"20261020T10000{i}Z-{SHA_CKPT[:12]}-0a1b2c3{i}" for i in (1, 2)]
    status = _run(TrainingManager(report_state=reader), popen, inputs, make=names)
    assert reader.asked == []
    assert status["outcome"]["kept"] is None and status["outcome"]["registered"] is None
    assert "cannot be told, so neither is claimed" in status["outcome"]["summary"]


def test_without_a_report_state_reader_nothing_is_claimed(popen, inputs):
    status = _run(TrainingManager(), popen, inputs,
                  make=[f"20261020T100000Z-{SHA_CKPT[:12]}-0a1b2c3d"])
    assert status["outcome"]["kept"] is None and status["outcome"]["registered"] is None
    assert "cannot be checked" in status["outcome"]["summary"]


def test_a_run_ended_by_sigkill_says_staged_files_may_be_left(popen, inputs, monkeypatch):
    for name in ("STOP_GRACE_SECONDS", "TERMINATE_GRACE_SECONDS", "KILL_GRACE_SECONDS"):
        monkeypatch.setattr(tm, name, 0.1)
    process = FakeProcess(pid=7, ignore_sigint=True, ignore_terminate=True)
    popen(process)
    manager = TrainingManager(report_state=Reader(False, False))
    manager.start_test(inputs)
    (reports(inputs) / f"20261020T100000Z-{SHA_CKPT[:12]}-0a1b2c3d").mkdir(parents=True)
    result = manager.cancel_test()
    assert result["status"] == "cancelled"
    assert "ended by SIGKILL, so staged files (*.tmp) may be left" in result["outcome"]["summary"]


# --------------------------------------------------------------------------- per-job logs
def test_each_job_keeps_its_own_output_tail(tmp_path, popen, inputs):
    training = FakeProcess(pid=1, output="epoch 1\nepoch 2\n")
    test = FakeProcess(pid=2, output="INFO Device: cpu\nReport: x\n")
    popen(training, test)
    manager = TrainingManager()
    manager.start_training({"data_dir": str(tmp_path / "ws")})
    training.finish(0)
    ended(manager, "training")
    manager.start_test(inputs)
    test.finish(0)
    ended(manager, "test")
    assert manager.get_log("training")["lines"] == ["epoch 1", "epoch 2"]
    assert manager.get_log("test")["lines"] == ["INFO Device: cpu", "Report: x"]


def test_a_failure_names_the_signal_that_ended_it(tmp_path, popen):
    process = FakeProcess(pid=1, output="CUDA out of memory\n")
    popen(process)
    manager = TrainingManager()
    manager.start_training({"data_dir": str(tmp_path / "ws")})
    process.finish(-9)
    ended(manager, "training")
    status = manager.get_status()
    assert status["status"] == "failed"
    assert status["error_message"].startswith("Process ended by SIGKILL")
    assert "CUDA out of memory" in status["error_message"]


# --------------------------------------------------------------------------- a real child
CHILD = r'''
import signal, sys, time
sys.path.insert(0, sys.argv[1])
from src.utils.gpu_lock import take_gpu_lock
with take_gpu_lock("training", "child.py"):
    print("HELD", flush=True)
    if sys.argv[2] == "ignore":
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(600)
'''


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("behaviour, expected_exit", [("plain", -signal.SIGINT),
                                                      ("ignore", -signal.SIGKILL)])
def test_a_real_child_holds_the_lock_and_releases_it_when_stopped(
        tmp_path, monkeypatch, behaviour, expected_exit):
    """The child takes the lock itself; the runner's stop (SIGINT, or SIGKILL
    for a child that ignores the gentler signals) ends it, and the kernel frees
    the lock with it."""
    root = tmp_path / "workspaces"
    root.mkdir()
    monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(root))
    for name in ("STOP_GRACE_SECONDS", "TERMINATE_GRACE_SECONDS"):
        monkeypatch.setattr(tm, name, 1.0)
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    real_popen = tm.subprocess.Popen

    def launch(argv, **kwargs):
        return real_popen([sys.executable, str(script), str(tm.PROJECT_ROOT), behaviour],
                          **kwargs)

    manager = TrainingManager()
    with patch("subprocess.Popen", launch):
        assert manager.start_training({"data_dir": str(root / "ws")})["success"]
    wait_for(lambda: "HELD" in manager.get_log("training")["lines"], 30)
    pid = manager.get_status()["pid"]
    assert f"pid {pid}" in probe(root / LOCK_FILE_NAME)

    result = manager.stop_training()
    assert result["status"] == "stopped" and manager.get_status()["exit_code"] == expected_exit
    assert probe(root / LOCK_FILE_NAME) is None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_a_child_killed_outside_the_runner_frees_the_slot_and_the_lock(tmp_path, monkeypatch):
    root = tmp_path / "workspaces"
    root.mkdir()
    monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(root))
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    real_popen = tm.subprocess.Popen

    def launch(argv, **kwargs):
        return real_popen([sys.executable, str(script), str(tm.PROJECT_ROOT), "plain"],
                          **kwargs)

    manager = TrainingManager()
    with patch("subprocess.Popen", launch):
        manager.start_training({"data_dir": str(root / "ws")})
    wait_for(lambda: "HELD" in manager.get_log("training")["lines"], 30)
    os.kill(manager.get_status()["pid"], signal.SIGKILL)  # the OOM killer, say
    ended(manager, "training")
    status = manager.get_status()
    assert status["status"] == "failed" and status["error_message"].startswith(
        "Process ended by SIGKILL")
    assert manager.active_job() is None and probe(root / LOCK_FILE_NAME) is None
