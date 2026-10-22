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
import json
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
    """Each test uses the lock tests/conftest.py gives it, or names its own;
    the developer's environment never decides."""
    monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)
    monkeypatch.delenv("SHEPHERD_DEVICE", raising=False)


def _lock_under(monkeypatch, root: Path) -> None:
    """The lock under the workspaces root `root`, the deployment's default:
    the conftest's own file is unset, since beside the root it names another
    file and is refused (root-cause review, P24-S1)."""
    monkeypatch.delenv(LOCK_FILE_ENV, raising=False)
    monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(root))


def _end_real_job(manager: TrainingManager, job_type: str = "training") -> None:
    """A test's own real child, stopped by its pid if it still runs (a test
    that failed), so no process outlives the test."""
    job = manager._last.get(job_type)
    if job is None or job.process is None:
        return
    if job.process.poll() is None:
        job.process.kill()
        job.process.wait(timeout=10)
    job.done.wait(30)


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


#: The registration's states, by the runner's True / False / None.
KEPT = {True: "kept", False: "not_kept", None: "cannot_be_checked"}
REGISTERED = {True: "registered", False: "not_registered", None: "cannot_be_checked"}


class Reader:
    """A report-state reader with the registration helper's signature
    (`registration.report_state`) that records what it was asked and answers
    a `ReportState`."""

    def __init__(self, kept, registered, why_kept="the reader's kept reason",
                 why_registered="the reader's registered reason"):
        self.answer = (kept, registered, why_kept, why_registered)
        self.asked = []

    def __call__(self, report, *, checkpoint, expected_sha256=None):
        from src.evaluation.registration import ReportState

        self.asked.append((report, checkpoint, expected_sha256))
        kept, registered, why_kept, why_registered = self.answer
        return ReportState(KEPT[kept], why_kept, REGISTERED[registered], why_registered,
                           expected_sha256, None, None)


REPORT_SHA = "a" * 64
RUN_ID = f"20261020T100000Z-{SHA_CKPT[:12]}-0a1b2c3d"


def outcome_path(popens) -> Path:
    """The `--outcome` file the runner passed to the latest test run."""
    argv = popens.calls[-1]
    return Path(argv[argv.index("--outcome") + 1])


def write_outcome(path: Path, inputs: TestRunInputs, *, complete=True,
                  registration=None, report=None) -> Path:
    """What measure_scorer writes there once its report is published."""
    report = report or reports(inputs) / RUN_ID / "report.json"
    path.write_text(json.dumps({
        "schema_version": 1, "complete": complete, "run_id": report.parent.name,
        "report": str(report), "report_sha256": REPORT_SHA,
        "checkpoint": str(inputs.checkpoint), "checkpoint_sha256": SHA_CKPT,
        "registration": registration or {"requested": False, "registered": False,
                                         "outcome": "not_requested", "why": "no --record"},
    }))
    return report


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
    _lock_under(monkeypatch, root)
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


def test_a_lock_not_configured_refuses_every_start_before_anything(tmp_path, popen,
                                                                   monkeypatch, inputs):
    """Review of 8242b47, P2-4: with neither lock setting set, the runner
    refuses a start of either type, saying how to configure the lock; nothing
    is started, named or written (it used to keep only its own one-job rule
    while a CLI elsewhere ran unlocked)."""
    monkeypatch.delenv(LOCK_FILE_ENV, raising=False)
    monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)
    calls = popen()
    manager = TrainingManager()
    for result in (manager.start_training({"data_dir": str(tmp_path / "ws"),
                                           "run_dir": str(tmp_path / "run")}),
                   manager.start_test(inputs)):
        assert result["success"] is False
        assert result["error"].startswith("The GPU lock cannot be used: neither "
                                          "SHEPHERD_GPU_LOCK_FILE nor SHEPHERD_WORKSPACES_ROOT "
                                          "is set")
        assert "§4.11" in result["error"] and result["error"].endswith("Nothing was started.")
    assert calls.calls == [] and manager.run_dir is None and not (tmp_path / "run").exists()
    lock = manager.get_status()["gpu"]["lock"]
    assert lock["unset"] is True and lock["configured"] is False and lock["problem"]


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


class HeldOutput:
    """A job's output that stays open until released: its monitor waits for it
    after the process has ended (`OUTPUT_GRACE_SECONDS`), as for a job whose
    forked worker keeps the pipe."""

    def __init__(self) -> None:
        self.released = threading.Event()

    def readline(self) -> str:
        self.released.wait(10)
        return ""


class HeldReader(Reader):
    """A report-state reader that takes its time (WP4's reads the ledger and
    hashes files), until released."""

    def __init__(self, kept, registered, why_registered="the reader's registered reason"):
        super().__init__(kept, registered, why_registered=why_registered)
        self.reading = threading.Event()
        self.released = threading.Event()

    def __call__(self, report, **kwargs):
        self.reading.set()
        self.released.wait(10)
        return super().__call__(report, **kwargs)


def test_a_cancel_after_the_run_ended_on_its_own_is_refused_and_keeps_its_ending(
        popen, inputs):
    """Exit 3 (not registered) on its own, its monitor still reading what it
    left: the cancel sends nothing, calls nothing cancelled, and hides no
    failure; the refusal says how the run ended and what it left."""
    process = FakeProcess(pid=8)
    popens = popen(process)
    reader = HeldReader(True, False, "refused: runs disagree")
    manager = TrainingManager(report_state=reader)
    manager.start_test(inputs)
    write_outcome(outcome_path(popens), inputs)
    process.finish(3)
    assert reader.reading.wait(5)
    assert manager.get_test_status()["status"] == "running", "the window: still reading"
    threading.Timer(0.2, reader.released.set).start()

    result = manager.cancel_test()
    assert process.signals == [], "no signal for a process that had ended"
    assert result["success"] is False and result["status"] == "failed"
    assert result["error"].startswith("A test run (job test-")
    assert result["error"].endswith("had already ended on its own (failed, exit code 3) before "
                                    "this stop arrived; nothing was stopped.")
    assert result["outcome"]["kept"] is True and result["outcome"]["registered"] is False
    status = manager.get_test_status()
    assert status["status"] == "failed" and status["error_message"].startswith(
        "Process exited with code 3")


def test_a_training_stop_after_the_process_ended_on_its_own_is_refused(tmp_path, popen):
    """The same window for training: its failure is not turned into "stopped"
    and its failure text is kept."""
    process = FakeProcess(pid=9, output="")
    process.stdout = HeldOutput()
    popen(process)
    manager = TrainingManager()
    manager.start_training({"data_dir": str(tmp_path / "ws")})
    process.finish(1)
    threading.Timer(0.2, process.stdout.released.set).start()

    result = manager.stop_training()
    assert process.signals == []
    assert result["success"] is False and result["status"] == "failed"
    assert result["error"].endswith("had already ended on its own (failed, exit code 1) before "
                                    "this stop arrived; nothing was stopped.")
    status = manager.get_status()
    assert status["status"] == "failed"
    assert status["error_message"].startswith("Process exited with code 1")


@pytest.mark.parametrize("own_exit, status", [(0, "completed"), (1, "failed"), (3, "failed")])
def test_a_cancel_that_lands_as_the_run_ends_on_its_own_keeps_that_ending(
        popen, inputs, own_exit, status):
    """The cancel's SIGINT reached a process that then exited with its own
    status: measure_scorer dies by SIGINT when interrupted, so a status of 0
    or more is the run's own ending, never "cancelled"."""
    popen(FakeProcess(pid=5, on_sigint=own_exit))
    manager = TrainingManager()
    manager.start_test(inputs)
    result = manager.cancel_test()
    assert result["status"] == status
    error = manager.get_test_status()["error_message"]
    assert (error is None) if own_exit == 0 else error.startswith(
        f"Process exited with code {own_exit}")


@pytest.mark.parametrize("signal_exit", [-signal.SIGINT, -signal.SIGTERM, -signal.SIGKILL])
def test_a_cancel_that_ends_the_run_by_a_signal_is_cancelled(popen, inputs, signal_exit):
    popen(FakeProcess(pid=5, on_sigint=signal_exit))
    manager = TrainingManager()
    manager.start_test(inputs)
    assert manager.cancel_test()["status"] == "cancelled"
    assert manager.get_test_status()["error_message"] is None


def test_a_cancel_reports_the_run_it_cancelled_not_one_started_since(popen, inputs):
    """A second session's start lands between the stop's end and the reply."""
    first, second = FakeProcess(pid=1), FakeProcess(pid=2)
    popen(first, second)

    class Interleaved(TrainingManager):
        def _stop(self, job_type):
            result = super()._stop(job_type)
            assert self.start_test(inputs)["success"]
            return result

    manager = Interleaved()
    manager.start_test(inputs)
    result = manager.cancel_test()
    assert result["status"] == "cancelled"
    assert result["outcome"] is not None and result["outcome"]["kept"] is False
    assert manager.get_test_status()["status"] == "running", "the second run"
    second.finish(0)


def test_a_training_stop_reads_the_snapshot_of_the_run_it_stopped(tmp_path, popen):
    from src.utils.checkpoint_paths import run_checkpoint_dir

    first, second = FakeProcess(pid=1), FakeProcess(pid=2)
    popen(first, second)
    workspace = tmp_path / "ws"

    class Interleaved(TrainingManager):
        def _stop(self, job_type):
            result = super()._stop(job_type)
            assert self.start_training({"data_dir": str(workspace)})["success"]
            return result

    manager = Interleaved()
    run_dir = Path(manager.start_training({"data_dir": str(workspace)})["run_dir"])
    snapshot = run_checkpoint_dir(run_dir) / "last.pt"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"not a checkpoint")
    result = manager.stop_training()
    assert manager.run_dir != run_dir, "a second run was started meanwhile"
    assert result["outcome"]["checkpoint"] == str(snapshot)
    second.finish(0)


def test_the_runner_does_not_import_the_composition_root():
    """The deployment's device is read from src.config, not from src.api.main:
    a service never builds the application to read a setting."""
    import ast

    tree = ast.parse(Path(tm.__file__).read_text())
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                 for alias in node.names}
    assert "src.api.main" not in imported and "src.api" not in imported


def test_the_stop_route_does_not_run_on_the_event_loop():
    from src.api.routes import training

    assert not inspect.iscoroutinefunction(training.stop_training)


# --------------------------------------------------------------------------- the test job's inputs
def test_the_test_command_line_comes_from_the_inputs_only(inputs):
    outcome = Path("/tmp/job/outcome.json")
    argv = build_test_argv(inputs, None, outcome)
    assert argv[:2] == [sys.executable, str(tm.PROJECT_ROOT / "scripts" / "measure_scorer.py")]
    assert argv[2:] == [
        "--checkpoint", str(inputs.checkpoint), "--data-dir", str(inputs.workspace),
        "--cohort", "mygene2-2026-10", "--registry", str(inputs.registry),
        "--expect-checkpoint-sha256", SHA_CKPT, "--expect-version-sha256", SHA_VERSION,
        "--outcome", str(outcome)]
    assert "--batch-size" not in argv, "the CLI's default stays its one definition"


def test_every_test_run_writes_its_outcome_to_a_new_file_in_its_own_directory(popen, inputs):
    """`--outcome` names a file that does not exist yet (measure_scorer refuses
    an existing one), in a directory made for this job alone, which is removed
    once the run has ended and been read."""
    first, second = FakeProcess(pid=1), FakeProcess(pid=2)
    popens = popen(first, second)
    manager = TrainingManager(report_state=Reader(True, False))
    paths = []
    for process in (first, second):
        assert manager.start_test(inputs)["success"]
        path = outcome_path(popens)
        assert path.parent.is_dir() and not path.exists()
        assert sorted(path.parent.iterdir()) == []
        paths.append(path)
        process.finish(1)
        ended(manager, "test")
        assert not path.parent.exists(), "the job's own directory is removed at its end"
    assert paths[0].parent != paths[1].parent


def test_the_operators_choices_and_the_evidence_setting_reach_the_command_line(inputs):
    from dataclasses import replace

    chosen = replace(inputs, record=True, deterministic=True,
                     evidence_dirs=(Path("/srv/evidence/a"), Path("/srv/evidence/b")))
    argv = build_test_argv(chosen, "cpu", Path("/tmp/job/outcome.json"))
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
def _run(manager, popen, inputs, *, code=0, make=(), existing=(), outcome=None):
    """One test run: `make` run directories appear in the store during it, and
    `outcome` (keyword arguments for `write_outcome`, or True) is written to the
    file the runner named, before the process ends with `code`."""
    for name in existing:
        (reports(inputs) / name).mkdir(parents=True)
    process = FakeProcess(pid=6)
    popens = popen(process)
    assert manager.start_test(inputs)["success"]
    for name in make:
        (reports(inputs) / name).mkdir(parents=True)
    if outcome is not None:
        write_outcome(outcome_path(popens), inputs, **({} if outcome is True else outcome))
    process.finish(code)
    ended(manager, "test")
    return manager.get_test_status()


def test_what_the_run_published_is_read_from_its_outcome_and_the_disk(popen, inputs):
    """The outcome file names the report and the digest the run published; the
    registration's helper reads kept and registered with that digest."""
    reader = Reader(True, False, why_registered="there is no evaluations.json")
    status = _run(TrainingManager(report_state=reader), popen, inputs, code=3, outcome={
        "registration": {"requested": True, "registered": False, "outcome": "refused",
                         "why": "runs disagree — investigate"}},
        existing=[f"20261019T100000Z-{SHA_CKPT[:12]}-ffffffff"])
    run_dir = reports(inputs) / RUN_ID
    assert reader.asked == [(run_dir / "report.json", inputs.checkpoint, REPORT_SHA)]
    assert status["status"] == "failed" and status["exit_code"] == 3
    # Exit 3 is also a report not as published, which may be gone: "kept" is
    # the outcome's, read from the disk, never the exit status's.
    assert status["exit_meaning"].startswith("measured, but not registered")
    assert "kept: see the outcome" in status["exit_meaning"]
    assert "measured and kept" not in status["exit_meaning"]
    outcome = status["outcome"]
    assert outcome["report_dir"] == str(run_dir) and outcome["kept"] is True
    assert outcome["registered"] is False and outcome["run_outcome"]["report_sha256"] == REPORT_SHA
    assert outcome["summary"] == (
        f"A report was kept at {run_dir}; it is not registered (there is no "
        "evaluations.json). The run said of its registration: runs disagree — investigate.")


def test_a_refused_registration_is_said_even_when_an_earlier_record_names_the_same_bytes(
        popen, inputs):
    """M5-3 review EVAL-2: repeated runs of one measurement on one machine can
    publish byte-identical reports, so the ledger may hold an earlier run's
    record naming this report's bytes while this run's own --record was
    refused (runs disagree). The card showed "Registered ✓" with no word of
    the refusal; the summary now says both."""
    reader = Reader(True, True, why_registered="record 2 in evaluations.json names this report")
    status = _run(TrainingManager(report_state=reader), popen, inputs, code=3, outcome={
        "registration": {"requested": True, "registered": False, "outcome": "refused",
                         "why": "runs disagree — investigate"}})
    outcome = status["outcome"]
    assert outcome["registered"] is True
    run_dir = reports(inputs) / RUN_ID
    assert outcome["summary"] == (
        f"A report was kept at {run_dir}; it is registered (record 2 in evaluations.json names "
        "this report). This run's own registration was refused: runs disagree — investigate. "
        "The record found names the same report bytes, which an earlier run of this "
        "measurement registered; it says nothing of this run's refusal.")
    # Not asked to register: the record found is the whole answer.
    status = _run(TrainingManager(report_state=reader), popen, inputs, code=0, outcome={
        "registration": {"requested": False, "registered": False, "outcome": "not_requested",
                         "why": "--record was not given"}})
    assert "own registration was refused" not in status["outcome"]["summary"]


@pytest.mark.parametrize("kept, registered, says", [
    (False, False, "The run published a report at {report}, but it is not kept: "
                   "the reader's kept reason."),
    (None, None, "Whether the report at {report} is kept cannot be checked "
                 "(the reader's kept reason)."),
    (True, True, "A report was kept at {run_dir}; it is registered (the reader's registered "
                 "reason)."),
])
def test_kept_and_registered_come_from_the_disk_not_the_exit_status(popen, inputs, kept,
                                                                    registered, says):
    """Exit 0 whatever the reader says: kept and registered are the reader's,
    with its reasons, never a cause the runner supposes."""
    status = _run(TrainingManager(report_state=Reader(kept, registered)), popen, inputs,
                  code=0, outcome=True)
    report = reports(inputs) / RUN_ID / "report.json"
    assert status["status"] == "completed"
    assert (status["outcome"]["kept"], status["outcome"]["registered"]) == (kept, registered)
    assert status["outcome"]["summary"].startswith(
        says.format(report=report, run_dir=report.parent))


def test_an_outcome_written_only_at_publication_says_the_run_did_not_finish(popen, inputs):
    status = _run(TrainingManager(report_state=Reader(True, False)), popen, inputs, code=1,
                  outcome={"complete": False})
    assert status["outcome"]["kept"] is True
    assert status["outcome"]["summary"].endswith(
        "Its outcome file is the one written when the report was published: the run did not "
        "get as far as writing it again at its end.")


@pytest.mark.parametrize("content", [b"{not json", b'{"report": 3}', b"[]",
                                     b'{"report": "/r/report.json", "report_sha256": "x"}'])
def test_an_outcome_file_that_cannot_be_read_claims_nothing(popen, inputs, content):
    reader = Reader(True, True)
    process = FakeProcess(pid=6)
    popens = popen(process)
    manager = TrainingManager(report_state=reader)
    manager.start_test(inputs)
    outcome_path(popens).write_bytes(content)
    process.finish(0)
    ended(manager, "test")
    outcome = manager.get_test_status()["outcome"]
    assert reader.asked == []
    assert outcome["kept"] is None and outcome["registered"] is None
    assert "is not one this runner reads" in outcome["summary"]
    assert outcome["summary"].endswith("so what it left cannot be checked.")


def test_no_outcome_file_and_no_new_report_directory_means_nothing_was_published(popen, inputs):
    reader = Reader(True, True)
    status = _run(TrainingManager(report_state=reader), popen, inputs, code=1,
                  make=[f"20261020T100000Z-{'d' * 12}-0a1b2c3d"])  # another checkpoint's
    assert reader.asked == []
    assert status["outcome"]["kept"] is False and status["outcome"]["registered"] is False
    assert status["outcome"]["summary"] == (
        f"No report was published: the run wrote no outcome file and created no report "
        f"directory under {reports(inputs)}, so it kept and registered nothing.")
    assert status["status"] == "failed"


def test_a_report_left_without_an_outcome_file_is_still_found(popen, inputs):
    """A run stopped between publishing its report and writing its outcome
    (the window `--outcome` cannot close) left a report: saying "no report
    was published" would not be true, so the run directory it created is read,
    with no published digest to compare."""
    reader = Reader(True, False)
    status = _run(TrainingManager(report_state=reader), popen, inputs, code=1, make=[RUN_ID])
    run_dir = reports(inputs) / RUN_ID
    assert reader.asked == [(run_dir / "report.json", inputs.checkpoint, None)]
    assert status["outcome"]["kept"] is True
    assert status["outcome"]["summary"].startswith(
        f"The run wrote no outcome file, but a report directory for this checkpoint appeared "
        f"during it: {run_dir}. A report was kept at {run_dir}; it is not registered")


def test_two_candidate_reports_are_never_guessed_between(popen, inputs):
    reader = Reader(True, True)
    names = [f"20261020T10000{i}Z-{SHA_CKPT[:12]}-0a1b2c3{i}" for i in (1, 2)]
    status = _run(TrainingManager(report_state=reader), popen, inputs, make=names)
    assert reader.asked == []
    assert status["outcome"]["kept"] is None and status["outcome"]["registered"] is None
    assert "cannot be told, so neither is claimed" in status["outcome"]["summary"]


def test_the_runners_reader_is_the_registrations_one_helper(popen, inputs, tmp_path):
    """No reader given: the registration's `report_state` reads the disk. The
    report the outcome names is not there, so it is not kept (never inferred
    from the exit status 0), and with no ledger it is not registered."""
    status = _run(TrainingManager(), popen, inputs, code=0, outcome=True)
    outcome = status["outcome"]
    assert outcome["kept"] is False and outcome["registered"] is False
    assert "but it is not kept: there is no" in outcome["summary"]
    assert tm.TrainingManager()._report_state is tm._registration_report_state


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
    run_dir = reports(inputs) / f"20261020T100000Z-{SHA_CKPT[:12]}-0a1b2c3d"
    assert result["outcome"]["summary"].endswith(
        f"The run was ended by SIGKILL, so staged files (*.tmp) may be left in {run_dir}.")
    assert "ledger" not in result["outcome"]["summary"], "it was not registering"


def test_a_registering_run_ended_by_sigterm_names_the_ledgers_directory_too(
        popen, inputs, monkeypatch):
    """`--record` stages the ledger beside the checkpoint, so that directory is
    named as well as the run directory."""
    from dataclasses import replace

    for name in ("STOP_GRACE_SECONDS", "TERMINATE_GRACE_SECONDS"):
        monkeypatch.setattr(tm, name, 0.1)
    popen(FakeProcess(pid=7, ignore_sigint=True))
    manager = TrainingManager(report_state=Reader(True, False))
    manager.start_test(replace(inputs, record=True))
    run_dir = reports(inputs) / f"20261020T100000Z-{SHA_CKPT[:12]}-0a1b2c3d"
    run_dir.mkdir(parents=True)
    summary = manager.cancel_test()["outcome"]["summary"]
    assert summary.endswith(
        f"The run was ended by SIGTERM, so staged files (*.tmp) may be left in {run_dir}, and, "
        f"as it was registering the report, in {inputs.checkpoint.resolve().parent} beside the "
        "checkpoint's ledger.")


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
    _lock_under(monkeypatch, root)
    for name in ("STOP_GRACE_SECONDS", "TERMINATE_GRACE_SECONDS"):
        monkeypatch.setattr(tm, name, 1.0)
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    real_popen = tm.subprocess.Popen

    def launch(argv, **kwargs):
        return real_popen([sys.executable, str(script), str(tm.PROJECT_ROOT), behaviour],
                          **kwargs)

    manager = TrainingManager()
    try:
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
    finally:
        _end_real_job(manager)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_a_child_killed_outside_the_runner_frees_the_slot_and_the_lock(tmp_path, monkeypatch):
    root = tmp_path / "workspaces"
    root.mkdir()
    _lock_under(monkeypatch, root)
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    real_popen = tm.subprocess.Popen

    def launch(argv, **kwargs):
        return real_popen([sys.executable, str(script), str(tm.PROJECT_ROOT), "plain"],
                          **kwargs)

    manager = TrainingManager()
    try:
        with patch("subprocess.Popen", launch):
            manager.start_training({"data_dir": str(root / "ws")})
        wait_for(lambda: "HELD" in manager.get_log("training")["lines"], 30)
        os.kill(manager.get_status()["pid"], signal.SIGKILL)  # the OOM killer, say
        ended(manager, "training")
        status = manager.get_status()
        assert status["status"] == "failed" and status["error_message"].startswith(
            "Process ended by SIGKILL")
        assert manager.active_job() is None and probe(root / LOCK_FILE_NAME) is None
    finally:
        _end_real_job(manager)
