"""
The backend's shutdown stops its job (the owner's decision of 2026-10-22)
========================================================================
The parts, each through its own entry point; the whole, a real backend
stopped from its terminal, is tests/integration/test_backend_stops_its_job.py.

- `TrainingManager.end_active_job`: the job holding the slot stopped, one an
  operator is already stopping waited for, nothing when no job runs.
- The lifespan's shutdown calls it: a real child job ends when an in-process
  application shuts down.
- The stop starts at the signal (`_ShutdownSignals`; verification of the
  owner's decision): uvicorn's shutdown waits for every open connection, and
  an open dashboard tab keeps one, before the lifespan's end. The server's
  SIGINT and SIGTERM handlers are wrapped and SIGHUP is routed to its SIGTERM
  handler: each still runs the server's handler, and the first starts the
  job's stop in a thread of its own; every one ends the dashboard's heartbeat
  streams (`_end_dashboard_streams`). Only from the main thread, only where a
  server set SIGTERM, SIGHUP only where it has its default action, and each
  put back at shutdown unless someone else replaced it since.
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


class _Server:
    """uvicorn's handlers as far as these tests need: what each signal did."""

    def __init__(self):
        self.seen = []

    def handle_exit(self, number, frame):
        self.seen.append(signal.Signals(number).name)


@pytest.fixture
def server_signals(monkeypatch):
    """A server's SIGINT and SIGTERM handlers in place, SIGHUP at its default,
    the job's stop and the end of the dashboard's streams recorded (the stop
    blocks until released, to show it runs in a thread of its own); every
    handler put back after."""
    import src.api.main as api

    server = _Server()
    names = ("SIGINT", "SIGTERM", "SIGHUP")
    previous = {name: signal.getsignal(getattr(signal, name)) for name in names}
    signal.signal(signal.SIGINT, server.handle_exit)
    signal.signal(signal.SIGTERM, server.handle_exit)
    signal.signal(signal.SIGHUP, signal.SIG_DFL)
    calls = {"stops": [], "streams": 0, "release": threading.Event()}

    def stop_the_active_job():
        calls["stops"].append(threading.current_thread().name)
        calls["release"].wait(10)

    def end_dashboard_streams(application):
        assert application is api.app
        calls["streams"] += 1

    monkeypatch.setattr(api, "_stop_the_active_job", stop_the_active_job)
    monkeypatch.setattr(api, "_end_dashboard_streams", end_dashboard_streams)
    try:
        yield api, server, calls
    finally:
        calls["release"].set()
        for name, handler in previous.items():
            signal.signal(getattr(signal, name), handler)


def _in_the_event_loop(scenario):
    """`scenario()` run as the lifespan runs: in an event loop, in the main thread."""
    import asyncio

    async def run():
        return await scenario()

    return asyncio.run(run())


async def _delivered(number):
    """`number` sent to this process, and the callback it handed the loop run."""
    import asyncio

    os.kill(os.getpid(), number)
    for _ in range(50):
        await asyncio.sleep(0.01)


@POSIX
def test_the_first_shutdown_signal_starts_the_job_s_stop_in_a_thread(server_signals):
    """SIGTERM: the server's own handler runs (its shutdown begins as before),
    the job's stop starts in a thread of its own at once, not at the
    lifespan's end, and the dashboard's heartbeat streams end. A hangup next
    goes to the server's SIGTERM handler and starts no second stop; a second
    Ctrl-C is still the server's (uvicorn's forced quit)."""
    api, server, calls = server_signals

    async def scenario():
        signals = api._ShutdownSignals(api.app)
        signals.install()
        try:
            await _delivered(signal.SIGTERM)
            assert server.seen == ["SIGTERM"]
            assert calls["stops"] == ["backend-stops-its-job"], "the stop did not start"
            assert calls["streams"] == 1
            assert signals._stopping.is_alive() and not signals._stopping.daemon
            await _delivered(signal.SIGHUP)
            await _delivered(signal.SIGINT)
            await _delivered(signal.SIGINT)
            assert server.seen == ["SIGTERM", "SIGHUP", "SIGINT", "SIGINT"]
            assert calls["stops"] == ["backend-stops-its-job"], "a second stop was started"
            assert calls["streams"] == 4
        finally:
            calls["release"].set()
            signals.end_jobs()          # waits for the stop, then stops what holds the slot
            signals.restore()
        assert not signals._stopping.is_alive()
        assert calls["stops"] == ["backend-stops-its-job", "MainThread"]

    _in_the_event_loop(scenario)
    assert signal.getsignal(signal.SIGINT) == server.handle_exit
    assert signal.getsignal(signal.SIGTERM) == server.handle_exit
    assert signal.getsignal(signal.SIGHUP) is signal.SIG_DFL


@POSIX
def test_a_hangup_alone_shuts_down_as_sigterm_does(server_signals):
    api, server, calls = server_signals

    async def scenario():
        signals = api._ShutdownSignals(api.app)
        signals.install()
        try:
            await _delivered(signal.SIGHUP)
            assert server.seen == ["SIGHUP"], "the hangup did not reach the server's handler"
            assert calls["stops"] == ["backend-stops-its-job"] and calls["streams"] == 1
        finally:
            calls["release"].set()
            signals.end_jobs()
            signals.restore()

    _in_the_event_loop(scenario)
    assert signal.getsignal(signal.SIGHUP) is signal.SIG_DFL


@POSIX
def test_without_a_signal_the_lifespan_s_end_stops_the_job_itself(server_signals):
    """No signal began the shutdown (an in-process client, a server told to
    exit): `end_jobs` stops the job, and no thread was started."""
    api, server, calls = server_signals
    calls["release"].set()

    async def scenario():
        signals = api._ShutdownSignals(api.app)
        signals.install()
        signals.end_jobs()
        signals.restore()
        assert signals._stopping is None

    _in_the_event_loop(scenario)
    assert calls["stops"] == ["MainThread"] and calls["streams"] == 0 and server.seen == []


@POSIX
def test_a_handler_replaced_since_is_not_put_back(server_signals):
    """After uvicorn's forced quit it has put back its own handlers before the
    lifespan's end runs (in the event loop's teardown): those stay."""
    api, server, calls = server_signals

    async def scenario():
        signals = api._ShutdownSignals(api.app)
        signals.install()
        signal.signal(signal.SIGINT, signal.default_int_handler)   # uvicorn's own put back
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signals.restore()

    _in_the_event_loop(scenario)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    assert signal.getsignal(signal.SIGHUP) is signal.SIG_DFL


@POSIX
def test_signals_no_server_set_are_left_as_they_are(server_signals):
    """SIGHUP ignored (nohup) or handled by someone else, SIGINT at Python's
    default handler, or no handler of a server's for SIGTERM: left as they are."""
    api, server, calls = server_signals

    async def ignored_hangup_and_default_ctrl_c():
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signals = api._ShutdownSignals(api.app)
        signals.install()
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
        assert signal.getsignal(signal.SIGTERM) != server.handle_exit, "SIGTERM was not wrapped"
        signals.restore()
        assert signal.getsignal(signal.SIGTERM) == server.handle_exit

    async def no_server():
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
        signals = api._ShutdownSignals(api.app)
        signals.install()
        assert signals._installed == {} and signal.getsignal(signal.SIGHUP) is signal.SIG_DFL

    _in_the_event_loop(ignored_hangup_and_default_ctrl_c)
    _in_the_event_loop(no_server)


@POSIX
def test_not_the_main_thread_nothing_is_installed(server_signals):
    """An in-process test client runs the lifespan in another thread, where no
    server sets handlers (and none can be set)."""
    import asyncio

    api, server, calls = server_signals
    seen = []

    def in_a_thread():
        async def scenario():
            signals = api._ShutdownSignals(api.app)
            signals.install()
            seen.append(dict(signals._installed))

        asyncio.run(scenario())

    worker = threading.Thread(target=in_a_thread)
    worker.start()
    worker.join(30)
    assert seen == [{}]
    assert signal.getsignal(signal.SIGTERM) == server.handle_exit
    assert signal.getsignal(signal.SIGHUP) is signal.SIG_DFL


def test_the_dashboard_s_heartbeat_streams_are_ended_as_gradio_s_own_server_ends_them():
    """The served dashboard's `stop_event` (Gradio's own server sets it when
    it stops, `Blocks._end_streaming_responses`; mounted, nothing did) is set,
    which ends each open tab's heartbeat stream; with no dashboard mounted
    there is nothing to end."""
    gr = pytest.importorskip("gradio")
    from fastapi import FastAPI
    from starlette.routing import Mount

    import src.api.main as api

    with gr.Blocks() as blocks:
        gr.State(0)                     # session state: the page keeps a heartbeat open
    parent = gr.mount_gradio_app(FastAPI(), blocks, path=api.DASHBOARD_PATH)
    served = next(route.app for route in parent.routes
                  if isinstance(route, Mount) and route.path == api.DASHBOARD_PATH)
    assert not served.stop_event.is_set()
    api._end_dashboard_streams(parent)
    assert served.stop_event.is_set()
    api._end_dashboard_streams(FastAPI())   # nothing mounted: nothing to end


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
