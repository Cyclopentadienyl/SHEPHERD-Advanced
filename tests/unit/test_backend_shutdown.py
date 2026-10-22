"""
The backend's shutdown stops its job (the owner's decision of 2026-10-22)
========================================================================
The parts, each through its own entry point; the whole, a real backend
stopped from its terminal, is tests/integration/test_backend_stops_its_job.py.

- `TrainingManager.end_active_job`: the job holding the slot stopped, one an
  operator is already stopping waited for, nothing when no job runs.
- The lifespan's shutdown calls it: a real child job ends when an in-process
  application shuts down.
- A hangup (SIGHUP) is routed to the server's own SIGTERM handler while the
  application runs, only from the main thread and only where SIGHUP has its
  default action, and put back at shutdown.
- No auto-reload (`src.api.main:main()`, setup_demo's printed commands): a
  reload restarts the server, and so would stop a training on every save.
- The launcher in its browser mode waits for the server through a Ctrl-C
  instead of killing it after 0.25 s, so the server's shutdown can end.

Module: tests/unit/test_backend_shutdown.py
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

import src.api.services.training_manager as tm  # noqa: E402
from tests.fixtures.job_processes import FakeProcess  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
POSIX = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")

#: A real job that ends on SIGINT, as training does.
SLEEPER = "import time\ntry:\n    time.sleep(600)\nexcept KeyboardInterrupt:\n    pass\n"


def _real_job(manager: tm.TrainingManager, job_type: str = "training"):
    with manager._lock:
        job, error = manager._spawn_locked(tm.JOB_TYPES[job_type],
                                           [sys.executable, "-c", SLEEPER],
                                           inputs={"checkpoint_sha256": "c" * 64})
    assert job is not None, error
    return job


def _cleanup(job) -> None:
    if job is not None and job.process.poll() is None:
        job.process.kill()
        job.process.wait(timeout=10)
    if job is not None:
        job.done.wait(30)


# --------------------------------------------------------------------------- end_active_job
def test_nothing_to_end_when_no_job_holds_the_slot():
    assert tm.TrainingManager().end_active_job() is None


@POSIX
@pytest.mark.parametrize("job_type, status", [("training", "stopped"), ("test", "cancelled")])
def test_the_job_holding_the_slot_is_stopped_whatever_its_type(job_type, status):
    manager = tm.TrainingManager()
    job = _real_job(manager, job_type)
    try:
        ended = manager.end_active_job()
        assert ended["job_type"] == job_type and ended["job_id"] == job.job_id
        assert ended["success"] is True and ended["status"] == status, ended
        assert job.process.poll() is not None and manager.active_job() is None
    finally:
        _cleanup(job)


def test_a_job_an_operator_is_already_stopping_is_waited_for(monkeypatch, tmp_path):
    """The operator's stop is escalating in its own thread; shutdown does not
    start another, it waits for that one to end the job."""
    monkeypatch.setattr(tm, "STOP_GRACE_SECONDS", 0.3)
    monkeypatch.setattr(tm, "TERMINATE_GRACE_SECONDS", 5.0)
    stubborn = FakeProcess(pid=4321, ignore_sigint=True, ignore_terminate=True)
    manager = tm.TrainingManager()
    with patch("subprocess.Popen", return_value=stubborn):
        assert manager.start_training({"data_dir": str(tmp_path)})["success"]
    stopper = threading.Thread(target=manager.stop_training, daemon=True)
    stopper.start()
    deadline = time.monotonic() + 5
    while manager.get_job_status_word("training") != "stopping" and time.monotonic() < deadline:
        time.sleep(0.01)
    threading.Timer(0.5, lambda: stubborn.finish(-15)).start()
    ended = manager.end_active_job()
    stopper.join(10)
    assert ended == {"job_type": "training", "job_id": manager._last["training"].job_id,
                     "success": True, "status": "stopped"}
    assert stubborn.signals.count("SIGINT") == 1, "shutdown started a second stop"


# --------------------------------------------------------------------------- the lifespan
@POSIX
def test_the_application_s_shutdown_stops_the_job_its_runner_holds(monkeypatch):
    from fastapi.testclient import TestClient

    import src.api.main as api

    manager = tm.TrainingManager()
    monkeypatch.setattr(tm, "training_manager", manager)
    job = None
    try:
        with TestClient(api.app) as client:
            assert client.get("/health").status_code == 200
            job = _real_job(manager)
            assert job.process.poll() is None
        assert job.done.is_set() and job.status == "stopped" and job.stop_signal == "SIGINT"
        assert job.process.poll() is not None and manager.active_job() is None
    finally:
        _cleanup(job)


@POSIX
def test_a_hangup_is_routed_to_the_server_s_termination_handler_and_put_back():
    import src.api.main as api

    def server_handler(number, frame):  # uvicorn's handle_exit, say
        pass

    previous_term = signal.signal(signal.SIGTERM, server_handler)
    previous_hup = signal.signal(signal.SIGHUP, signal.SIG_DFL)
    try:
        replaced = api._hangup_as_termination()
        assert replaced is signal.SIG_DFL
        assert signal.getsignal(signal.SIGHUP) is server_handler
        signal.signal(signal.SIGHUP, replaced)
        # Ignored (nohup) or handled by someone else: left as it is.
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        assert api._hangup_as_termination() is None
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
        # No handler of the server's to route it to: left as it is.
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        assert api._hangup_as_termination() is None
        # Not the main thread (an in-process test client's lifespan): untouched.
        signal.signal(signal.SIGTERM, server_handler)
        seen = []
        worker = threading.Thread(target=lambda: seen.append(api._hangup_as_termination()))
        worker.start()
        worker.join()
        assert seen == [None] and signal.getsignal(signal.SIGHUP) is signal.SIG_DFL
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGHUP, previous_hup)


# --------------------------------------------------------------------------- no auto-reload
def test_the_api_entry_point_does_not_reload(monkeypatch):
    pytest.importorskip("uvicorn")
    import uvicorn

    from src.api import main

    runs = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: runs.append(kwargs))
    main.main()
    assert not runs[0].get("reload"), "a reload would stop a training on every saved file"


# --------------------------------------------------------------------------- the launcher
#: A stand-in server that, like uvicorn with a job to stop, takes a while to
#: shut down after Ctrl-C and then says so.
SLOW_SERVER = r'''import signal, sys, time
from pathlib import Path
def shut_down(number, frame):
    time.sleep(1.5)
    Path(sys.argv[1]).write_text("shut down")
    sys.exit(7)
signal.signal(signal.SIGINT, shut_down)
print("READY", flush=True)
time.sleep(600)
'''
#: The launcher's `serve`, as its browser mode runs it, in a session of its own.
LAUNCHER = r'''import importlib.util, sys
spec = importlib.util.spec_from_file_location("shep_launch", sys.argv[1])
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
sys.exit(launcher.serve([sys.executable, sys.argv[2], sys.argv[3]]))
'''


@POSIX
def test_the_launcher_waits_for_the_server_through_a_ctrl_c(tmp_path):
    """Ctrl-C reaches the terminal's foreground group, the launcher and the
    server alike. The launcher keeps waiting while the server shuts down (and
    stops its job), and exits with the server's status; `subprocess.run`
    killed the server 0.25 s after the first Ctrl-C."""
    server, launcher = tmp_path / "server.py", tmp_path / "launcher.py"
    server.write_text(SLOW_SERVER)
    launcher.write_text(LAUNCHER)
    done_marker = tmp_path / "done"
    process = subprocess.Popen(
        [sys.executable, str(launcher), str(REPO / "scripts" / "launch" / "shep_launch.py"),
         str(server), str(done_marker)],
        stdout=subprocess.PIPE, text=True, start_new_session=True, cwd=str(REPO))
    try:
        assert "READY" in process.stdout.readline() + process.stdout.readline()
        assert process.pid > 1 and os.getpgid(process.pid) == process.pid != os.getpgrp()
        os.killpg(process.pid, signal.SIGINT)
        assert process.wait(timeout=60) == 7
        assert done_marker.read_text() == "shut down", "the server was killed mid-shutdown"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        process.stdout.close()
