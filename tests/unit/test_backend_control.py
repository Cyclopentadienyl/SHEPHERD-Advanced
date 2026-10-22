"""
Unit tests for the backend self-restart service.

These avoid actually re-exec'ing the process: ``restart_backend`` exposes
``_scheduler`` / ``_execv`` hooks so the side effect can be captured instead of
performed. The pure helpers (env re-resolution, argv reconstruction, the
training lock) are exercised directly.
"""
import os
import signal
import sys
import threading

import pytest

# backend_control lives under src.api, whose package __init__ imports FastAPI.
# Skip cleanly where the API stack isn't installed (matches test_webui.py).
pytest.importorskip("fastapi")

import src.api.services.backend_control as bc
from src.config.runtime_presets import ALLOCATOR_PRESETS, DEFAULT_ALLOCATOR


# --------------------------------------------------------------------------- lock
@pytest.mark.parametrize("job_type", ["training", "test"])
@pytest.mark.parametrize("status", ["running", "stopping"])
def test_is_training_active_true_while_any_job_holds_the_gpu_slot(monkeypatch, job_type, status):
    """The restart lock reads the job runner's one GPU slot (M5): a test run
    is orphaned by a re-exec as surely as training is."""
    job = {"job_type": job_type, "status": status}
    monkeypatch.setattr(bc.training_manager, "active_job", lambda: job)
    assert bc.is_training_active() is True


def test_is_training_active_false_when_no_job_holds_the_slot(monkeypatch):
    monkeypatch.setattr(bc.training_manager, "active_job", lambda: None)
    assert bc.is_training_active() is False


def test_is_training_active_follows_a_real_test_job(monkeypatch, tmp_path):
    """Through the real runner: a running test run locks the restart, and its
    end unlocks it."""
    from src.api.services.training_manager import TrainingManager
    from tests.fixtures.job_processes import FakeProcess

    manager = TrainingManager()
    monkeypatch.setattr(bc, "training_manager", manager)
    process = FakeProcess()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("subprocess.Popen", lambda *a, **k: process)
        assert manager.start_training({"data_dir": str(tmp_path)})["success"]
    assert bc.is_training_active() is True
    process.finish(0)
    manager._last["training"].done.wait(5)
    assert bc.is_training_active() is False


def test_is_training_active_swallows_errors(monkeypatch):
    def boom():
        raise RuntimeError("status backend down")

    monkeypatch.setattr(bc.training_manager, "active_job", boom)
    assert bc.is_training_active() is False  # UI default: fail open


def test_is_training_active_fail_closed_on_error(monkeypatch):
    def boom():
        raise RuntimeError("status backend down")

    monkeypatch.setattr(bc.training_manager, "active_job", boom)
    # Destructive-restart path passes default_on_error=True -> treat as busy.
    assert bc.is_training_active(default_on_error=True) is True
    assert bc.is_training_active(default_on_error=False) is False


# --------------------------------------------------------------------------- env re-resolve
def test_resolve_restart_env_reresolves_preset(monkeypatch):
    monkeypatch.setattr(
        bc, "load_runtime_settings", lambda: {"allocator_preset": "expandable"}
    )
    out = bc.resolve_restart_env(
        {bc.ALLOC_SOURCE_ENV: "preset", "PYTORCH_ALLOC_CONF": "stale-value"}
    )
    assert out["PYTORCH_ALLOC_CONF"] == ALLOCATOR_PRESETS["expandable"]
    assert out[bc.ALLOC_SOURCE_ENV] == "preset"


def test_resolve_restart_env_unknown_preset_falls_back(monkeypatch):
    monkeypatch.setattr(
        bc, "load_runtime_settings", lambda: {"allocator_preset": "bogus"}
    )
    out = bc.resolve_restart_env({bc.ALLOC_SOURCE_ENV: "preset"})
    assert out["PYTORCH_ALLOC_CONF"] == ALLOCATOR_PRESETS[DEFAULT_ALLOCATOR]


def test_resolve_restart_env_preserves_explicit_override():
    # Marker says the value was an explicit user override -> leave it untouched.
    src = {bc.ALLOC_SOURCE_ENV: "env", "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"}
    out = bc.resolve_restart_env(src)
    assert out["PYTORCH_ALLOC_CONF"] == "backend:cudaMallocAsync"


def test_resolve_restart_env_no_marker_left_alone():
    # No marker but an explicit allocator env var is present -> preserve it.
    out = bc.resolve_restart_env({"PYTORCH_ALLOC_CONF": "backend:native"})
    assert out["PYTORCH_ALLOC_CONF"] == "backend:native"


def test_resolve_restart_env_no_marker_with_cuda_env_preserved():
    # The legacy PYTORCH_CUDA_ALLOC_CONF override (no marker) is also preserved.
    out = bc.resolve_restart_env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    assert out["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert "PYTORCH_ALLOC_CONF" not in out  # nothing injected over an override


def test_resolve_restart_env_direct_launch_resolves(monkeypatch):
    # Direct `uvicorn ...` launch: no marker AND no allocator env -> the saved
    # preset must be applied so "Apply + Restart" actually takes effect.
    monkeypatch.setattr(
        bc, "load_runtime_settings", lambda: {"allocator_preset": "expandable"}
    )
    out = bc.resolve_restart_env({})
    assert out["PYTORCH_ALLOC_CONF"] == ALLOCATOR_PRESETS["expandable"]
    assert out[bc.ALLOC_SOURCE_ENV] == "preset"


def test_restart_applies_the_start_rule(monkeypatch):
    # Restart Backend and the service entry point share one rule, so for any
    # environment a restart resolves what a start would.
    from src.config.runtime_presets import allocator_env

    settings = {"allocator_preset": "native"}
    monkeypatch.setattr(bc, "load_runtime_settings", lambda: settings)
    for env in (
        {},
        {bc.ALLOC_SOURCE_ENV: "preset", "PYTORCH_ALLOC_CONF": "stale-value"},
        {bc.ALLOC_SOURCE_ENV: "env", "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"},
        {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    ):
        assert bc.resolve_restart_env(env) == allocator_env(env, settings), env


def test_resolve_restart_env_returns_copy():
    src = {bc.ALLOC_SOURCE_ENV: "env"}
    out = bc.resolve_restart_env(src)
    assert out is not src


# --------------------------------------------------------------------------- argv
def test_build_restart_argv_uvicorn_module():
    # argv[0] is uvicorn's __main__ when launched via `python -m uvicorn`.
    argv = [
        os.path.join("site-packages", "uvicorn", "__main__.py"),
        "src.api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    out = bc.build_restart_argv("/usr/bin/python", argv)
    assert out == [
        "/usr/bin/python",
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]


def test_build_restart_argv_non_uvicorn_verbatim():
    argv = ["/some/script.py", "--flag"]
    out = bc.build_restart_argv("/usr/bin/python", argv)
    assert out == ["/usr/bin/python", "/some/script.py", "--flag"]


# --------------------------------------------------------------------------- restart
def test_restart_backend_refused_while_training(monkeypatch):
    monkeypatch.setattr(bc, "is_training_active", lambda **_: True)
    calls = []
    res = bc.restart_backend(_scheduler=lambda fn: calls.append(fn))
    assert res["success"] is False
    assert "A GPU job (training or a test run) is in progress" in res["error"]
    assert calls == []  # nothing scheduled


@pytest.fixture
def runner(monkeypatch):
    """A job runner of the test's own in place of the singleton, its processes
    stand-ins, so a start in the restart's window can be made for real."""
    from unittest.mock import patch

    from src.api.services.training_manager import TrainingManager
    from tests.fixtures.job_processes import FakeProcess

    manager = TrainingManager()
    monkeypatch.setattr(bc, "training_manager", manager)
    with patch("subprocess.Popen", side_effect=lambda *a, **k: FakeProcess(pid=4321)):
        yield manager
    if manager.active_job() is not None:
        manager.end_active_job()


def _start(manager, tmp_path):
    return manager.start_training({"data_dir": str(tmp_path / "ws")})


def test_restart_backend_aborts_if_a_job_starts_during_delay(runner, tmp_path):
    # The guard sees idle (schedule); a job starts during the delay; the
    # worker sees it when it closes the slot, so it must NOT exec, and the
    # slot stays open for that job's runner.
    executed = []
    worker_box = []
    res = bc.restart_backend(
        delay=0,
        _scheduler=lambda fn: worker_box.append(fn),
        _execv=lambda *a: executed.append(a),
    )
    assert res["success"] is True  # scheduled at guard time
    assert _start(runner, tmp_path)["success"]
    worker_box[0]()  # run the worker — it should abort at the slot
    assert executed == []  # execve never called
    runner.end_active_job()
    assert _start(runner, tmp_path)["success"], "the aborted restart left the slot closed"


def test_a_start_racing_the_restart_is_refused_while_its_exec_runs(runner, tmp_path):
    """Root-cause review, P23-1: the worker closes the slot in the same step
    as it finds it idle and keeps it closed through execve, so a start asked
    for in that window (through the REST API or another tab) is refused, not
    orphaned by the exec. It used to re-check, then exec with nothing held.
    The exec runs under the runner's lock (round 3 of the review fixes,
    P3-3), so the start waits on it while the exec runs and is refused once
    the stand-in exec has returned."""
    racing = []

    def execv(executable, argv, env):
        assert runner._lock.locked(), "the exec ran without the runner's lock"
        start = threading.Thread(target=lambda: racing.append(_start(runner, tmp_path)))
        start.start()
        start.join(0.3)
        assert racing == [], "a start went through while the exec ran"
        racing.append(start)

    worker_box = []
    assert bc.restart_backend(delay=0, _scheduler=worker_box.append, _execv=execv)["success"]
    worker_box[0]()
    racing[0].join(10)
    assert racing[1]["success"] is False
    assert racing[1]["error"] == "The backend is stopping; nothing was started."
    assert runner.active_job() is None


def test_a_failed_exec_opens_the_slot_again(runner, tmp_path):
    def execv(executable, argv, env):
        raise OSError("exec format error")

    worker_box = []
    assert bc.restart_backend(delay=0, _scheduler=worker_box.append, _execv=execv)["success"]
    worker_box[0]()
    assert _start(runner, tmp_path)["success"], "the process serves on, so jobs start"


def _the_backend_stops(monkeypatch, runner):
    """The backend's own stop, as its shutdown signal runs it
    (`src.api.main._stop_the_active_job`), on the test's runner."""
    import src.api.main as api
    import src.api.services.training_manager as tm

    monkeypatch.setattr(tm, "training_manager", runner)
    api._stop_the_active_job()


def test_a_stop_during_the_restart_delay_is_never_overridden_by_the_exec(
        monkeypatch, runner, tmp_path):
    """Verification of P23-1, T2-V2: Restart Backend is clicked, and the
    backend begins to stop (Ctrl-C, ``systemctl stop``) before its delay has
    passed. The stop owns the closed slot, so the restart is aborted: no exec
    replaces the stopping process (a new image would serve on and ignore the
    stop), and the slot stays closed. It used to exec, and a failed exec
    opened the slot the stop had closed."""
    executed = []
    worker_box = []
    assert bc.restart_backend(delay=0, _scheduler=worker_box.append,
                              _execv=lambda *a: executed.append(a))["success"]
    _the_backend_stops(monkeypatch, runner)
    worker_box[0]()
    assert executed == [], "execve was called while the backend stops"
    assert _start(runner, tmp_path)["error"] == "The backend is stopping; nothing was started."


def test_a_failed_exec_never_opens_a_slot_the_stop_closed_meanwhile(
        monkeypatch, runner, tmp_path):
    """The other order: the restart closed the slot, then the backend began
    to stop while the exec ran, and the exec failed. The stop's close waits
    for the runner's lock, which the exec holds (P3-3); the failed exec opens
    only its own close, so whichever runs first, the stop's stays and
    nothing starts."""
    import src.api.main as api
    import src.api.services.training_manager as tm

    monkeypatch.setattr(tm, "training_manager", runner)
    stopping = []

    def execv(executable, argv, env):
        stop = threading.Thread(target=api._stop_the_active_job)
        stop.start()
        stopping.append(stop)
        raise OSError("exec format error")

    worker_box = []
    assert bc.restart_backend(delay=0, _scheduler=worker_box.append, _execv=execv)["success"]
    worker_box[0]()
    stopping[0].join(10)
    assert _start(runner, tmp_path)["error"] == "The backend is stopping; nothing was started."


def test_a_stop_between_the_restart_s_check_and_its_exec_aborts_the_exec(
        monkeypatch, runner, tmp_path):
    """Round 3 of the review fixes, P3-3: the restart has closed the slot
    (`close_if_idle`), and the backend's stop closes it while the restart
    resolves its environment, before the exec. The final check and the exec
    are one step under the runner's lock, so the exec is refused and the
    slot stays the stop's. It used to exec all the same: the new image
    opened the slot at startup and served on, and the stop's signal was
    lost with the old image."""
    resolve = bc.resolve_restart_env

    def resolve_while_the_backend_stops(env):
        _the_backend_stops(monkeypatch, runner)
        return resolve(env)

    monkeypatch.setattr(bc, "resolve_restart_env", resolve_while_the_backend_stops)
    executed = []
    worker_box = []
    assert bc.restart_backend(delay=0, _scheduler=worker_box.append,
                              _execv=lambda *a: executed.append(a))["success"]
    worker_box[0]()
    assert executed == [], "execve was called after the stop closed the slot"
    assert runner._closed_by == runner.CLOSED_BY_STOP
    assert _start(runner, tmp_path)["error"] == "The backend is stopping; nothing was started."


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_a_stop_signal_aborts_the_restart_before_its_stop_has_closed_the_slot(
        monkeypatch, runner, tmp_path):
    """P3-3: the signal's handler records the stop before anything else, so
    a restart whose delay ends after the handler ran, but before the stop it
    started has closed the slot (that stop runs in a thread the event loop
    starts), is refused at the exec. Through the backend's real handler: the
    server's SIGTERM handler wrapped, SIGTERM delivered, the stop held back.
    The next startup forgets the signal, and a restart execs again."""
    import asyncio

    import src.api.main as api
    import src.api.services.training_manager as tm

    monkeypatch.setattr(tm, "training_manager", runner)
    held = threading.Event()
    monkeypatch.setattr(api, "_stop_the_active_job", lambda: held.wait(10))
    monkeypatch.setattr(api, "_end_dashboard_streams", lambda application: None)
    seen = []
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda number, frame: seen.append(number))
    executed = []

    async def scenario():
        signals = api._ShutdownSignals(api.app)
        signals.install()
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.sleep(0.2)
            worker_box = []
            assert bc.restart_backend(delay=0, _scheduler=worker_box.append,
                                      _execv=lambda *a: executed.append(a))["success"]
            worker_box[0]()
            return runner._closed_by
        finally:
            held.set()
            signals.end_jobs()
            signals.restore()

    try:
        closed_by = asyncio.run(scenario())
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert seen == [signal.SIGTERM]
    assert closed_by == runner.CLOSED_BY_RESTART, "the stop closed the slot: not this case"
    assert executed == [], "execve was called after a stop signal"

    runner.open()                                   # the next startup's
    worker_box = []
    assert bc.restart_backend(delay=0, _scheduler=worker_box.append,
                              _execv=lambda *a: executed.append(a))["success"]
    worker_box[0]()
    assert len(executed) == 1


def test_restart_backend_schedules_and_execs(monkeypatch, runner):
    monkeypatch.setattr(bc, "is_training_active", lambda **_: False)
    monkeypatch.setattr(bc.sys, "executable", "/usr/bin/python")
    monkeypatch.setattr(
        bc.sys, "argv", ["/x/uvicorn/__main__.py", "src.api.main:app", "--port", "8000"]
    )

    captured = {}

    def fake_execv(executable, argv, env):
        captured["executable"] = executable
        captured["argv"] = argv
        captured["env"] = env

    worker_box = []
    res = bc.restart_backend(
        delay=0,
        _scheduler=lambda fn: worker_box.append(fn),
        _execv=fake_execv,
    )
    assert res["success"] is True
    assert len(worker_box) == 1  # exactly one job scheduled

    # Run the scheduled worker manually (it would normally run in a thread).
    worker_box[0]()
    assert captured["executable"] == "/usr/bin/python"
    assert captured["argv"] == [
        "/usr/bin/python",
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--port",
        "8000",
    ]
    assert isinstance(captured["env"], dict)
