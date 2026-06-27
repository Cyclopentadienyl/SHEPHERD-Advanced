"""
Unit tests for the backend self-restart service.

These avoid actually re-exec'ing the process: ``restart_backend`` exposes
``_scheduler`` / ``_execv`` hooks so the side effect can be captured instead of
performed. The pure helpers (env re-resolution, argv reconstruction, the
training lock) are exercised directly.
"""
import os

import pytest

# backend_control lives under src.api, whose package __init__ imports FastAPI.
# Skip cleanly where the API stack isn't installed (matches test_webui.py).
pytest.importorskip("fastapi")

import src.api.services.backend_control as bc
from src.runtime_presets import ALLOCATOR_PRESETS, DEFAULT_ALLOCATOR


# --------------------------------------------------------------------------- lock
def test_is_training_active_true_when_running(monkeypatch):
    monkeypatch.setattr(bc.training_manager, "get_status", lambda: {"status": "running"})
    assert bc.is_training_active() is True


def test_is_training_active_true_when_stopping(monkeypatch):
    monkeypatch.setattr(bc.training_manager, "get_status", lambda: {"status": "stopping"})
    assert bc.is_training_active() is True


def test_is_training_active_false_when_idle(monkeypatch):
    for s in ("idle", "completed", "failed"):
        monkeypatch.setattr(bc.training_manager, "get_status", lambda s=s: {"status": s})
        assert bc.is_training_active() is False, s


def test_is_training_active_swallows_errors(monkeypatch):
    def boom():
        raise RuntimeError("status backend down")

    monkeypatch.setattr(bc.training_manager, "get_status", boom)
    assert bc.is_training_active() is False  # must never raise into the UI


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
    out = bc.resolve_restart_env({"PYTORCH_ALLOC_CONF": "backend:native"})
    assert out["PYTORCH_ALLOC_CONF"] == "backend:native"


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
    monkeypatch.setattr(bc, "is_training_active", lambda: True)
    calls = []
    res = bc.restart_backend(_scheduler=lambda fn: calls.append(fn))
    assert res["success"] is False
    assert "Training in progress" in res["error"]
    assert calls == []  # nothing scheduled


def test_restart_backend_schedules_and_execs(monkeypatch):
    monkeypatch.setattr(bc, "is_training_active", lambda: False)
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
