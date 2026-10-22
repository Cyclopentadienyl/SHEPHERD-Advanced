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


# --------------------------------------------------------------------------- the closed slot
class _Popens:
    """`subprocess.Popen` replaced: every call recorded, each given a stand-in."""

    def __init__(self):
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        return FakeProcess(pid=4000 + len(self.calls))


@pytest.fixture
def test_inputs(tmp_path):
    registry = tmp_path / "registry"
    registry.mkdir()
    workspace = tmp_path / "workspaces" / "ws"
    return tm.TestRunInputs(
        registry=registry, version_label="mygene2-2026-10", version_sha256="e" * 64,
        workspace=workspace, checkpoint=workspace / "runs" / "gat" / "r" / "checkpoints" / "b.pt",
        checkpoint_sha256="c" * 64)


def test_once_the_backend_stops_no_job_starts_by_any_route(monkeypatch, tmp_path, test_inputs):
    """Root-cause review, P23-1: "closed" is a state of the one GPU slot. The
    shutdown's stop closes it before anything else (`_stop_the_active_job`),
    so a start still under way when the backend began to stop (a request in
    flight on a kept-alive connection, a queued dashboard event) is refused
    there, training and test runs alike, and nothing is spawned; it used to be
    accepted, and the job trained, holding the GPU lock, while uvicorn waited
    for the other connections."""
    import src.api.main as api

    manager = tm.TrainingManager()
    monkeypatch.setattr(tm, "training_manager", manager)
    spawned = _Popens()
    monkeypatch.setattr("subprocess.Popen", spawned)
    api._stop_the_active_job()                       # what the signal's thread runs first
    for started in (manager.start_training({"data_dir": str(tmp_path / "ws")}),
                    manager.start_test(test_inputs)):
        assert started["success"] is False
        assert started["error"] == "The backend is stopping; nothing was started."
    assert spawned.calls == [] and manager.active_job() is None
    manager.open()                                   # the next startup's
    assert manager.start_training({"data_dir": str(tmp_path / "ws")})["success"]
    assert len(spawned.calls) == 1
    manager.end_active_job()


def test_closing_reads_the_job_holding_the_slot_in_the_same_step(monkeypatch, tmp_path):
    """A start that took the slot before the close is the job the close
    returns, which the shutdown stops; there is no moment in between where a
    job could start unseen."""
    manager = tm.TrainingManager()
    monkeypatch.setattr("subprocess.Popen", _Popens())
    started = manager.start_training({"data_dir": str(tmp_path / "ws")})
    closed = manager.close()
    assert closed["job_id"] == started["job_id"] and closed["status"] == "running"
    assert manager.close_if_idle() is False
    assert manager.end_active_job()["status"] == "stopped"
    assert manager.close() is None
    assert manager.close_if_idle() is False, "a restart closed a slot the stop owns"
    manager.open()
    assert manager.close_if_idle() is True


@POSIX
def test_the_application_s_shutdown_leaves_the_slot_closed_and_its_startup_opens_it(
        monkeypatch, tmp_path):
    """No signal began this shutdown (an in-process client): the lifespan's
    end closes the slot itself; the next startup opens it again."""
    from fastapi.testclient import TestClient

    import src.api.main as api

    manager = tm.TrainingManager()
    monkeypatch.setattr(tm, "training_manager", manager)
    monkeypatch.setattr("subprocess.Popen", _Popens())
    with TestClient(api.app):
        pass
    refused = manager.start_training({"data_dir": str(tmp_path / "ws")})
    assert refused["error"] == "The backend is stopping; nothing was started."
    with TestClient(api.app):
        assert manager.start_training({"data_dir": str(tmp_path / "ws")})["success"]


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
    names = ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT")
    previous = {name: signal.getsignal(getattr(signal, name)) for name in names}
    signal.signal(signal.SIGINT, server.handle_exit)
    signal.signal(signal.SIGTERM, server.handle_exit)
    signal.signal(signal.SIGHUP, signal.SIG_DFL)
    signal.signal(signal.SIGQUIT, signal.SIG_DFL)
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
            signals.end_jobs()          # waits for the stop and does not repeat it (P3-4)
            signals.restore()
        assert not signals._stopping.is_alive()
        assert calls["stops"] == ["backend-stops-its-job"]

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
def test_a_ctrl_backslash_shuts_down_as_sigterm_does(server_signals):
    """Root-cause review, P23-2: SIGQUIT, the terminal's other terminating
    signal, reached the job while it shared the backend's group; it no longer
    does, and it was routed nowhere, so the backend dumped core and the job
    trained on with the lock. It goes to the server's SIGTERM handler, which
    records it (uvicorn raises it again once it has shut down), and starts
    the job's stop; it is put back at its default afterwards."""
    api, server, calls = server_signals

    async def scenario():
        signals = api._ShutdownSignals(api.app)
        signals.install()
        try:
            # Checked first: delivered unrouted, it would end this process.
            assert signal.getsignal(signal.SIGQUIT) is not signal.SIG_DFL, "Ctrl-\\ not routed"
            await _delivered(signal.SIGQUIT)
            assert server.seen == ["SIGQUIT"], "Ctrl-\\ did not reach the server's handler"
            assert calls["stops"] == ["backend-stops-its-job"] and calls["streams"] == 1
        finally:
            calls["release"].set()
            signals.end_jobs()
            signals.restore()

    _in_the_event_loop(scenario)
    assert signal.getsignal(signal.SIGQUIT) is signal.SIG_DFL


@POSIX
def test_every_signal_a_job_no_longer_gets_is_routed_to_the_shutdown(server_signals):
    """One definition, beside the spawn (`JOB_DETACHED_SIGNALS`): the backend
    takes every one of them present here, never a list of its own."""
    api, server, calls = server_signals
    present = {getattr(signal, name) for name in tm.JOB_DETACHED_SIGNALS
               if hasattr(signal, name)}

    async def scenario():
        signals = api._ShutdownSignals(api.app)
        signals.install()
        try:
            assert set(signals._installed) == present
        finally:
            signals.restore()

    _in_the_event_loop(scenario)


@POSIX
def test_a_signal_the_server_handles_itself_is_wrapped_whatever_its_name(server_signals):
    """Verification of P23-2, T2-V4: the server's handlers are told by the
    handler, never by a copy of the server's list of signal names. A server
    that handles SIGHUP itself (a uvicorn whose set grew) has its handler
    wrapped like SIGTERM's, so the hangup starts the job's stop; with the
    copied list it was left unwrapped and the job trained on."""
    api, server, calls = server_signals
    signal.signal(signal.SIGHUP, server.handle_exit)

    async def scenario():
        signals = api._ShutdownSignals(api.app)
        signals.install()
        try:
            await _delivered(signal.SIGHUP)
            assert server.seen == ["SIGHUP"]
            assert calls["stops"] == ["backend-stops-its-job"], "the server's own SIGHUP " \
                "handler was not wrapped"
        finally:
            calls["release"].set()
            signals.end_jobs()
            signals.restore()

    _in_the_event_loop(scenario)
    assert signal.getsignal(signal.SIGHUP) == server.handle_exit


@POSIX
def test_uvicorn_s_own_handlers_are_the_ones_wrapped():
    """uvicorn's real ``capture_signals``: every signal it handles itself
    (``uvicorn.server.HANDLED_SIGNALS``) is wrapped, its handler kept as the
    one called first, with nothing in the backend that lists those signals."""
    import uvicorn
    import uvicorn.server

    import src.api.main as api

    server = uvicorn.Server(uvicorn.Config(api.app))

    async def scenario():
        with server.capture_signals():
            signals = api._ShutdownSignals(api.app)
            signals.install()
            try:
                for number in uvicorn.server.HANDLED_SIGNALS:
                    assert signals._installed[number][0] == server.handle_exit, number
            finally:
                signals.restore()

    _in_the_event_loop(scenario)


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


class _Unkillable(FakeProcess):
    """A job SIGKILL cannot end (a process stuck in the driver, D state)."""

    def kill(self) -> None:
        self.signals.append("SIGKILL")


@POSIX
def test_one_stop_per_shutdown_a_job_sigkill_cannot_end_is_not_waited_for_twice(
        monkeypatch, tmp_path):
    """Round 3 of the review fixes, P3-4: the signal's stop gives up on a job
    SIGKILL cannot end ("did not end after SIGKILL") with the job still
    holding the slot. The lifespan's end then waits for that stop and
    returns; it used to stop again, which waited `END_ACTIVE_JOB_SECONDS`
    more, so the backend spent two bounds on its job against the one the
    systemd unit's TimeoutStopSec covers. Through the real signal path:
    the server's SIGTERM handler wrapped, the signal delivered, the real
    `_stop_the_active_job` in its thread, then `end_jobs`."""
    import src.api.main as api

    for name, seconds in (("STOP_GRACE_SECONDS", 0.2), ("TERMINATE_GRACE_SECONDS", 0.2),
                          ("KILL_GRACE_SECONDS", 0.2), ("RESULT_WAIT_SECONDS", 0.3)):
        monkeypatch.setattr(tm, name, seconds)
    monkeypatch.setattr(tm, "END_ACTIVE_JOB_SECONDS", 0.9)
    manager = tm.TrainingManager()
    monkeypatch.setattr(tm, "training_manager", manager)
    stuck = _Unkillable(pid=4321, ignore_sigint=True, ignore_terminate=True)
    with patch("subprocess.Popen", return_value=stuck):
        assert manager.start_training({"data_dir": str(tmp_path)})["success"]
    calls = []
    end_active_job = manager.end_active_job

    def counted():
        calls.append(threading.current_thread().name)
        return end_active_job()

    monkeypatch.setattr(manager, "end_active_job", counted)
    server = _Server()
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, server.handle_exit)

    async def scenario():
        signals = api._ShutdownSignals(api.app)
        signals.install()
        try:
            started = time.monotonic()
            await _delivered(signal.SIGTERM)
            signals.end_jobs()
            return time.monotonic() - started
        finally:
            signals.restore()

    try:
        took = _in_the_event_loop(scenario)
    finally:
        signal.signal(signal.SIGTERM, previous)
        stuck.finish(-9)
        manager._last["training"].done.wait(10)
    assert server.seen == ["SIGTERM"]
    assert calls == ["backend-stops-its-job"], "the lifespan's end stopped the job again"
    assert stuck.signals == ["SIGINT", "SIGTERM", "SIGKILL"], stuck.signals
    assert took < 0.9 + 0.5, f"{took:.2f} s: more than one END_ACTIVE_JOB_SECONDS"


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
    default handler or someone else's, or no handler of a server's for
    SIGTERM: left as they are."""
    api, server, calls = server_signals

    def someone_else_s(number, frame):
        pass

    async def ctrl_c_handled_by_someone_else():
        signal.signal(signal.SIGINT, someone_else_s)
        signals = api._ShutdownSignals(api.app)
        signals.install()
        assert signal.getsignal(signal.SIGINT) is someone_else_s
        assert signal.SIGINT not in signals._installed
        signals.restore()

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
    _in_the_event_loop(ctrl_c_handled_by_someone_else)
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


def _lifespan_in_the_main_thread(api) -> None:
    """The application's real lifespan, run as uvicorn runs it: in an event
    loop in the main thread, started and ended at once."""
    async def scenario():
        async with api.lifespan(api.app):
            pass

    _in_the_event_loop(scenario)


@POSIX
def test_a_backend_that_cannot_attach_to_the_server_s_signals_says_so(monkeypatch, caplog):
    """Round 3 of the review fixes, F4: in the main thread with no server's
    SIGTERM handler to wrap, the stop at the signal and the closed slot fall
    back to the lifespan's end, after every open connection has closed. That
    is a warning in the log; it used to be silent. With the server's handler
    in place nothing is said (the control)."""
    import src.api.main as api

    monkeypatch.setattr(tm, "training_manager", tm.TrainingManager())
    for name in ("SHEPHERD_KG_PATH", "SHEPHERD_CHECKPOINT_PATH", "SHEPHERD_BIND_HOST"):
        monkeypatch.delenv(name, raising=False)
    previous = signal.getsignal(signal.SIGTERM)
    server = _Server()
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        with caplog.at_level("WARNING", logger=api.logger.name):
            _lifespan_in_the_main_thread(api)
        said = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("cannot attach to the server's shutdown signals" in m
                   and "only at the lifespan's end" in m for m in said), said
        caplog.clear()
        signal.signal(signal.SIGTERM, server.handle_exit)
        with caplog.at_level("WARNING", logger=api.logger.name):
            _lifespan_in_the_main_thread(api)
        assert not any("cannot attach" in r.getMessage() for r in caplog.records)
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_off_the_main_thread_the_backend_says_why_it_takes_no_signal(monkeypatch, caplog):
    """An in-process client runs the lifespan off the main thread, where no
    signal can be taken: a debug line, not a warning (the tests' logs stay
    quiet), and never nothing."""
    from fastapi.testclient import TestClient

    import src.api.main as api

    monkeypatch.setattr(tm, "training_manager", tm.TrainingManager())
    with caplog.at_level("DEBUG", logger=api.logger.name):
        with TestClient(api.app):
            pass
    said = [(r.levelname, r.getMessage()) for r in caplog.records
            if "off the main thread" in r.getMessage()]
    assert [level for level, _ in said] == ["DEBUG"], said
    assert not any("cannot attach" in r.getMessage() for r in caplog.records)


def test_a_dashboard_with_no_stop_event_is_a_warning_not_a_silent_no_op(caplog):
    """F4: the dashboard's `stop_event` is the served Gradio app's attribute.
    A mounted app without it (a Gradio that renamed it) leaves every open tab
    holding the shutdown, and the log says so."""
    from fastapi import FastAPI
    from starlette.applications import Starlette

    import src.api.main as api

    parent = FastAPI()
    parent.mount(api.DASHBOARD_PATH, Starlette())
    with caplog.at_level("WARNING", logger=api.logger.name):
        api._end_dashboard_streams(parent)
        api._end_dashboard_streams(FastAPI())       # nothing mounted: nothing said
    said = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(said) == 1 and "has no stop_event" in said[0], said
    assert "holds the backend's shutdown until it is closed" in said[0]


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
