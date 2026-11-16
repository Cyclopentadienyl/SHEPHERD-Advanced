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
from pathlib import Path

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
        assert manager.start_training({"data_dir": str(tmp_path), "learning_stage": "base",
                                       "device": "cpu"})["success"]
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
    """A new run's start, a base (K8, TM12), as the runner takes one, on the
    CPU (auto is refused on a host without CUDA before anything starts)."""
    return manager.start_training({"data_dir": str(tmp_path / "ws"), "learning_stage": "base",
                                   "device": "cpu"})


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


# --------------------------------------------------------------------------- serving (I7)
@pytest.fixture
def serving(monkeypatch, runner, tmp_path):
    """A model served on (faked) CUDA through the real lifecycle, the GPU lock
    held under the test's own root; builds stubbed and counted, the runner
    the test's own for the load's slot check too."""
    import src.api.main as api
    import src.api.services.training_manager as tm
    from tests.fixtures.serving_lifecycle import (
        StubBuilds,
        checkpoint,
        fake_cuda,
        reload,
        served_workspace,
    )

    for name in ("pipeline", "kg", "_current_data_dir", "_current_checkpoint_path"):
        monkeypatch.setattr(api.app_state, name, None)
    monkeypatch.setattr(tm, "training_manager", runner)
    fake_cuda(monkeypatch)
    builds = StubBuilds().install(monkeypatch)
    workspace = served_workspace(tmp_path / "served")
    loaded = reload(workspace, checkpoint(workspace, "a"), "cuda")
    assert loaded.success and loaded.status.holds_gpu_lock, loaded.message
    return api, builds


def test_restart_stops_serving_before_the_exec(serving, runner):
    """The restart drains and releases before the image is replaced: at the
    exec, nothing is served and the GPU lock is free (its description gone),
    the stop recorded as the restart's."""
    from src.utils.gpu_lock import lock_state

    api, _ = serving
    seen = {}

    def execv(executable, argv, env):
        seen.update(pipeline=api.app_state.pipeline, lock=lock_state()["held_by"],
                    held=api.app_state.gpu_lock)

    worker_box = []
    assert bc.restart_backend(delay=0, _scheduler=worker_box.append, _execv=execv)["success"]
    worker_box[0]()
    assert seen == {"pipeline": None, "lock": None, "held": None}
    assert api.app_state.not_serving.kind == api.STOPPED
    assert api.app_state.not_serving_reason.startswith("Inference was stopped for Restart Backend")


def test_restart_is_refused_while_a_load_or_stop_runs(serving, runner):
    """One load or stop at a time is the lifecycle's: Restart Backend is
    refused while one runs, nothing scheduled and nothing stopped; a load
    begun during the delay aborts the restart and opens the slot again."""
    api, _ = serving
    executed, scheduled = [], []
    assert api.app_state._lifecycle.acquire(blocking=False)
    try:
        refused = bc.restart_backend(_scheduler=scheduled.append,
                                     _execv=lambda *a: executed.append(a))
    finally:
        api.app_state._lifecycle.release()
    assert refused == {"success": False, "error": "A model is being loaded or inference is "
                                                  "stopping — restart is locked."}
    assert scheduled == [] and api.app_state.pipeline is not None

    worker_box = []
    assert bc.restart_backend(delay=0, _scheduler=worker_box.append,
                              _execv=lambda *a: executed.append(a))["success"]
    assert api.app_state._lifecycle.acquire(blocking=False)    # a load begins in the delay
    try:
        worker_box[0]()
    finally:
        api.app_state._lifecycle.release()
    assert executed == [] and api.app_state.pipeline is not None
    assert runner._closed_by is None, "the aborted restart left the slot closed"


def test_a_failed_exec_does_not_load_a_model_again(serving, runner, tmp_path):
    """The stop came before the exec; a failed exec opens the slot again and
    loads nothing: no build, no model, the lock free."""
    from src.utils.gpu_lock import lock_state

    api, builds = serving
    built = len(builds.calls)

    def execv(executable, argv, env):
        raise OSError("exec format error")

    worker_box = []
    assert bc.restart_backend(delay=0, _scheduler=worker_box.append, _execv=execv)["success"]
    worker_box[0]()
    assert len(builds.calls) == built, "a failed exec loaded a model"
    assert api.app_state.pipeline is None and lock_state()["held_by"] is None
    assert _start(runner, tmp_path)["success"], "the process serves on, so jobs start"


def test_a_restart_whose_stop_fails_opens_the_slot_again(serving, runner, monkeypatch):
    """A stop of serving that fails aborts the restart as a failed exec does:
    nothing is executed and the slot the restart closed is opened again, so
    the process serves on and no job is refused for a restart that is not
    coming."""
    api, _ = serving
    executed = []

    def failing_stop(**kwargs):
        raise RuntimeError("the stop failed")

    monkeypatch.setattr(api.app_state, "stop_serving", failing_stop)
    worker_box = []
    assert bc.restart_backend(delay=0, _scheduler=worker_box.append,
                              _execv=lambda *a: executed.append(a))["success"]
    worker_box[0]()

    assert executed == []
    assert runner._closed_by is None, "the slot stayed closed after the restart's stop failed"


def test_the_restart_words_say_a_loaded_model_is_stopped_and_not_loaded_again(
        monkeypatch, tmp_path):
    """The ops round's A3b (I7): Settings' Restart Backend says, in the
    question it asks before the restart (R-12) and when the restart starts,
    what the backend does: a loaded model is stopped first and is not loaded
    again unless the start-up configures one; no page Load is restored. The
    words name no internal variable (NEW-settings-datasets-footer-2)."""
    pytest.importorskip("gradio")
    import src.config.runtime_presets as presets
    from src.webui.components import runtime_settings as rs
    from tests.fixtures.operator_access import operator_request

    assert "is stopped first" in rs.RESTART_STOPS_SERVING
    assert "not loaded again unless the service's start-up configures one" in (
        rs.RESTART_STOPS_SERVING)
    assert "SHEPHERD_" not in rs.RESTART_QUESTION
    assert rs.RESTART_QUESTION.startswith("Restart the backend? Every open page reconnects")
    monkeypatch.setattr(presets, "RUNTIME_SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    restarts = []
    monkeypatch.setattr(rs, "restart_backend", lambda: restarts.append(1) or {"success": True})
    _row, question, *_buttons, _said = rs._on_restart_request(operator_request(monkeypatch))
    assert rs.RESTART_STOPS_SERVING.replace("'", "&#x27;") in question["value"]
    assert restarts == []
    said = rs._on_restart(operator_request(monkeypatch))[-1]
    assert rs.RESTART_STOPS_SERVING in said and restarts == [1]


# --------------------------------------------------------------------------- data locations (UI-06)
def _pending_root_change(tmp_path, monkeypatch):
    """A saved settings file of the test's own, read by this process, whose
    saved workspaces folder waits to be applied; the environment names none."""
    import json

    from src.config.deployment_settings import ENV_SETTINGS_FILE, FORMAT

    old, new = tmp_path / "old-root", tmp_path / "new-root"
    old.mkdir()
    new.mkdir()
    path = tmp_path / "settings" / "deployment.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"format": FORMAT, "applied": {"workspaces_root": str(old)},
                                "saved": {"workspaces_root": str(new)}}))
    monkeypatch.setenv(ENV_SETTINGS_FILE, str(path))
    monkeypatch.delenv("SHEPHERD_WORKSPACES_ROOT", raising=False)
    return old.resolve(), new.resolve()


@pytest.fixture
def locations(monkeypatch, runner):
    """The data locations service with the test's own runner."""
    import src.api.services.data_locations as dl

    monkeypatch.setattr(dl, "training_manager", runner)
    return dl


def test_an_apply_waits_while_a_job_holds_the_runner_s_slot(locations, runner, tmp_path,
                                                             monkeypatch):
    """No job in the runner's slot, checked under its lock as a restart's
    `close_if_idle` checks it: the job read the old locations, so the change
    stays pending and says when it applies."""
    from src.config import server_settings

    old, new = _pending_root_change(tmp_path, monkeypatch)
    assert _start(runner, tmp_path)["success"]
    outcome = locations.apply_saved()
    assert not outcome.applied and outcome.pending
    assert "training is running in this service's job runner" in outcome.message
    assert outcome.message.endswith("press Apply then, or it applies at the next start.")
    # The view says the reason Apply gave, from the same check (T06): the
    # runner's own words, read without acting.
    busy = runner.busy_reason()
    assert busy.startswith("training is running in this service's job runner (job ")
    assert busy in outcome.message
    assert locations.view().pending_reason == (
        f"{busy[:1].upper()}{busy[1:]}, so the change applies when it ends: press Apply then, "
        "or it applies at the next start.")
    assert server_settings.workspaces_root().value == old
    runner.end_active_job()
    assert runner.busy_reason() is None
    assert locations.view().pending_reason == "Press Apply, or it applies at the next start."
    assert locations.apply_saved().applied
    assert server_settings.workspaces_root().value == new


def test_a_save_tries_the_apply_at_once(locations, tmp_path, monkeypatch):
    from src.config import server_settings

    old, new = _pending_root_change(tmp_path, monkeypatch)
    other = tmp_path / "third"
    other.mkdir()
    done = locations.save_and_apply({"workspaces_root": str(other)})
    assert done.save.saved and done.apply.applied
    assert server_settings.workspaces_root().value == other.resolve()


def test_a_save_refused_tries_no_apply(locations, tmp_path, monkeypatch):
    _pending_root_change(tmp_path, monkeypatch)
    done = locations.save_and_apply({"registry": "relative"})
    assert not done.save.saved and done.apply is None


def test_the_backend_start_applies_a_pending_change(locations, tmp_path, monkeypatch):
    """The startup's apply (`src.api.main._apply_saved_settings`), before the
    configured load; a hand command holding the old lock leaves it pending
    and the start goes on."""
    import src.api.main as api
    from src.config import server_settings
    from src.utils.gpu_lock import LOCK_FILE_NAME, GpuLock

    old, new = _pending_root_change(tmp_path, monkeypatch)
    with GpuLock(old / LOCK_FILE_NAME, job_type="training", command="by hand").acquire():
        api._apply_saved_settings()
        assert server_settings.workspaces_root().value == old
    api._apply_saved_settings()
    assert server_settings.workspaces_root().value == new
    source = __import__("inspect").getsource(api.lifespan)
    assert source.index("_open_the_gpu_slot()") < source.index("_apply_saved_settings()") < (
        source.index("initialize_pipeline()"))


def test_a_restart_reads_the_saved_file_again_never_a_derived_environment(
        tmp_path, monkeypatch):
    """Nothing writes a resolved data location into the environment, so the
    restart's environment names none and the new process reads the file."""
    _pending_root_change(tmp_path, monkeypatch)
    from src.config import server_settings

    assert server_settings.workspaces_root().source == server_settings.SOURCE_SAVED
    env = bc.resolve_restart_env(dict(os.environ))
    for variable in ("SHEPHERD_WORKSPACES_ROOT", "SHEPHERD_REGISTRY", "SHEPHERD_INTAKE_DIR",
                     "SHEPHERD_TRAINING_EVIDENCE_DIRS", "SHEPHERD_DEPLOYMENT_MODE"):
        assert variable not in env
    assert env["SHEPHERD_SETTINGS_FILE"] == os.environ["SHEPHERD_SETTINGS_FILE"]


# --------------------------------------------------------------------------- Stop service (UI-06e)
def test_stop_backend_is_refused_while_a_job_runs(monkeypatch):
    """A service started from the desktop entry has no terminal to stop it
    from, so Settings offers Stop service; like a restart it is refused while
    a GPU job runs (an unreadable status counts as one), with nothing
    scheduled."""
    monkeypatch.setattr(bc, "is_training_active", lambda **kwargs: kwargs["default_on_error"])
    scheduled = []
    result = bc.stop_backend(_scheduler=scheduled.append, _signal=lambda: None)
    assert result == {"success": False, "error": bc.STOP_BUSY_JOB} and scheduled == []


def test_stop_backend_is_refused_while_a_load_or_stop_of_inference_runs(monkeypatch, runner):
    import src.api.main as api

    monkeypatch.setattr(api.app_state, "lifecycle_busy", lambda: True)
    scheduled = []
    result = bc.stop_backend(_scheduler=scheduled.append, _signal=lambda: None)
    assert result == {"success": False, "error": bc.STOP_BUSY_LIFECYCLE} and scheduled == []


def test_stop_backend_closes_the_idle_slot_then_sends_the_stop(runner, tmp_path, caplog):
    """After its delay the stop closes the idle slot in one step, so no job
    starts in between, names who asked in the log, and sends the orderly
    stop's signal once."""
    signals, worker_box = [], []
    assert bc.stop_backend(delay=0, _scheduler=worker_box.append,
                           _signal=lambda: signals.append("SIGTERM")) == {"success": True}
    assert signals == []
    with caplog.at_level("WARNING", logger=bc.logger.name):
        worker_box[0]()
    assert signals == ["SIGTERM"]
    assert "asked for by Stop service in Settings" in caplog.text
    assert _start(runner, tmp_path)["error"] == "The backend is stopping; nothing was started."


def test_a_job_started_during_the_stop_s_delay_aborts_it(runner, tmp_path):
    signals, worker_box = [], []
    assert bc.stop_backend(delay=0, _scheduler=worker_box.append,
                           _signal=lambda: signals.append(1))["success"]
    assert _start(runner, tmp_path)["success"]
    worker_box[0]()
    assert signals == [], "the stop was sent while a job ran"
    runner.end_active_job()
    assert _start(runner, tmp_path)["success"], "the aborted stop left the slot closed"


def test_a_stop_signal_that_cannot_be_sent_leaves_the_service_serving(runner, tmp_path):
    def fails():
        raise OSError("no such process")

    worker_box = []
    assert bc.stop_backend(delay=0, _scheduler=worker_box.append, _signal=fails)["success"]
    worker_box[0]()
    assert _start(runner, tmp_path)["success"], "the slot stayed closed after a failed stop"


#: A real server (uvicorn in its own main thread, so its signal handlers and
#: the backend's are installed) whose Stop service is asked for once it serves.
_STOPPED_SERVER = r'''
import sys, threading, time, urllib.request
import uvicorn
from src.api import main
from src.api.services import backend_control

port = int(sys.argv[1])

def ask_for_the_stop():
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            break
        except OSError:
            time.sleep(0.2)
    print("STOP", backend_control.stop_backend(delay=0.2), flush=True)

threading.Thread(target=ask_for_the_stop, daemon=True).start()
uvicorn.run(main.app, host="127.0.0.1", port=port, log_level="info")
print("EXITED", flush=True)
'''


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_stop_service_stops_a_real_server_through_its_orderly_stop(tmp_path):
    """End to end: Stop service sends the server the SIGTERM a service
    manager sends, so uvicorn shuts down and the application's shutdown runs
    (the backend's stop of its job and of serving), with the reason in its
    log. uvicorn then ends by the signal it caught, as after a service
    manager's stop (systemd counts that as a clean stop)."""
    import socket
    import subprocess

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {k: v for k, v in os.environ.items()
           if k not in ("SHEPHERD_KG_PATH", "SHEPHERD_CHECKPOINT_PATH")}
    env.update(GRADIO_ANALYTICS_ENABLED="False", PYTHONUNBUFFERED="1")
    done = subprocess.run([sys.executable, "-c", _STOPPED_SERVER, str(port)],
                          cwd=str(Path(bc.__file__).resolve().parents[3]), env=env,
                          capture_output=True, text=True, timeout=300)
    out = done.stdout + done.stderr
    assert done.returncode == -signal.SIGTERM, out[-4000:]
    assert "STOP {'success': True}" in out
    assert "Stopping the service (pid" in out and "asked for by Stop service in Settings" in out
    assert out.index("Stopping the service") < out.index("Shutting down SHEPHERD-Advanced API")
    # The signal's own line (FL-SVC) says this one was the service's own.
    assert ("SIGTERM received, sent by this service itself, asked for by Stop service in "
            "Settings.") in out
    assert "not known to the service" not in out
    assert "Application shutdown complete" in out


#: A real server stopped from outside: uvicorn in its own main thread, so its
#: signal handlers and the backend's are installed; the test sends the signal.
_SERVED = r'''
import sys
import uvicorn
from src.api import main

uvicorn.run(main.app, host="127.0.0.1", port=int(sys.argv[1]), log_level="info")
'''


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("sent, source", [
    (signal.SIGTERM, "a stop from outside the service (a service manager, kill, a logout)"),
    (signal.SIGHUP, "a hangup (the terminal it ran in was closed)"),
])
def test_a_stop_signal_from_outside_is_named_in_the_log(tmp_path, sent, source):
    """FL-SVC: a stop by a signal from outside was logged only as uvicorn's
    "Shutting down", without the signal, so the manual acceptance's 16:09:14
    stop read the same as any other. The service now logs the signal and what
    usually sends it, and says it cannot know the sender (a Python handler
    gets the signal's number only), before the application's shutdown."""
    import socket
    import subprocess
    import time
    import urllib.request

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {k: v for k, v in os.environ.items()
           if k not in ("SHEPHERD_KG_PATH", "SHEPHERD_CHECKPOINT_PATH")}
    env.update(GRADIO_ANALYTICS_ENABLED="False", PYTHONUNBUFFERED="1")
    server = subprocess.Popen([sys.executable, "-c", _SERVED, str(port)],
                              cwd=str(Path(bc.__file__).resolve().parents[3]), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 240
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                break
            except OSError:
                assert server.poll() is None and time.monotonic() < deadline, \
                    "the server did not answer /health"
                time.sleep(0.2)
        server.send_signal(sent)
        out, _ = server.communicate(timeout=120)
    finally:
        if server.poll() is None:
            server.kill()
    said = (f"Stopping the service (pid {server.pid}): {signal.Signals(sent).name} received, "
            f"{source}; who sent it is not known to the service.")
    assert said in out, out[-4000:]
    assert out.index(said) < out.index("Shutting down SHEPHERD-Advanced API"), out[-4000:]
    assert "asked for by" not in out
