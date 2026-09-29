"""
`scripts/launch/serve_backend.py`: the backend started under the saved preset.

The allocator is fixed when CUDA first allocates, so what matters is the
environment the entry point hands to uvicorn and that nothing imports torch
before it does. The restart path is `backend_control.resolve_restart_env`;
both call `allocator_env`, and the scenarios below check that a start, a
systemd restart (a fresh start from the unit's environment) and a UI restart
agree for the same saved settings.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.config.runtime_presets import (
    ALLOC_SOURCE_ENV,
    ALLOCATOR_PRESETS,
    DEFAULT_ALLOCATOR,
    allocator_env,
    effective_allocator,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRY = REPO_ROOT / "scripts" / "launch" / "serve_backend.py"


def _entry():
    spec = importlib.util.spec_from_file_location("serve_backend", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_it_execs_bare_uvicorn_with_the_arguments_it_was_given():
    entry = _entry()
    program, command, _env = entry.exec_plan(
        {}, {}, ["--host", "127.0.0.1", "--port", "8264"], executable="/venv/python"
    )
    assert program == "/venv/python"
    # Exactly the bare command backend_control rebuilds on a restart.
    assert command == [
        "/venv/python", "-m", "uvicorn", "src.api.main:app",
        "--host", "127.0.0.1", "--port", "8264",
    ]


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({"allocator_preset": "native"}, ALLOCATOR_PRESETS["native"]),
        ({"allocator_preset": "expandable"}, ALLOCATOR_PRESETS["expandable"]),
        ({}, ALLOCATOR_PRESETS[DEFAULT_ALLOCATOR]),
        ({"allocator_preset": "bogus"}, ALLOCATOR_PRESETS[DEFAULT_ALLOCATOR]),
    ],
)
def test_start_and_both_restarts_resolve_the_saved_preset(settings, expected):
    entry = _entry()
    unit_env = {"PYTHONUNBUFFERED": "1"}

    started = entry.exec_plan(unit_env, settings, [])[2]
    systemd_restart = entry.exec_plan(unit_env, settings, [])[2]
    ui_restart = allocator_env(started, settings)

    assert started["PYTORCH_ALLOC_CONF"] == expected
    assert started[ALLOC_SOURCE_ENV] == "preset"
    assert systemd_restart["PYTORCH_ALLOC_CONF"] == expected
    assert ui_restart["PYTORCH_ALLOC_CONF"] == expected
    # And the launcher, reading the same settings, would pick the same one.
    assert effective_allocator({}, settings)[1] == expected


def test_an_explicit_override_survives_start_and_both_restarts():
    entry = _entry()
    unit_env = {"PYTORCH_ALLOC_CONF": "backend:native"}
    settings = {"allocator_preset": "expandable"}

    started = entry.exec_plan(unit_env, settings, [])[2]
    assert started["PYTORCH_ALLOC_CONF"] == "backend:native"
    assert ALLOC_SOURCE_ENV not in started
    assert entry.exec_plan(unit_env, settings, [])[2] == started
    assert allocator_env(started, settings) == started


def test_loading_it_imports_neither_torch_nor_the_application():
    # The allocator must be in the environment before torch is imported, and
    # importing src.api builds the application, so the entry point may load
    # neither before it execs.
    probe = (
        "import importlib.util, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('serve_backend', {str(ENTRY)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "print(json.dumps({'torch': 'torch' in sys.modules,"
        " 'api': any(m == 'src.api' or m.startswith('src.api.') for m in sys.modules)}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True,
        cwd=REPO_ROOT, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"torch": False, "api": False}
