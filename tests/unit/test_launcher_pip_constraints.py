"""
The launcher's optional accelerators never replace a locked package (BACKLOG 50).
================================================================================
`pip_install` is the launcher's one install function. pip is not in a
deploy-created .venv, but the ARM source build (`scripts/build_pyg_arm.sh`)
puts it there, and an accelerator installed with its dependencies can ask for
another torch. The accelerators stay out of uv.lock (optional third-party
plugins); what is checked here is that, with pip present, every install goes
through constraints exported from uv.lock, and that nothing is installed when
those constraints cannot be had.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "launch" / "shep_launch.py"


def _launcher():
    spec = importlib.util.spec_from_file_location("shep_launch", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recording(launcher, monkeypatch, *, uv="/usr/bin/uv", export_fails=False, pip_fails=False):
    """Replace the launcher's command runner: `uv export` writes a constraints
    file (or fails), pip is recorded with the constraints it was given."""
    calls = []

    def run(cmd, check=False):
        calls.append(list(cmd))
        if cmd[0] == uv:
            if export_fails:
                raise subprocess.CalledProcessError(2, cmd)
            output = Path(cmd[cmd.index("--output-file") + 1])
            output.write_text("torch==2.10.0+cu130\ntorch-geometric==2.8.1\n")
            return subprocess.CompletedProcess(cmd, 0)
        if "-c" in cmd:
            calls[-1].append(Path(cmd[cmd.index("-c") + 1]).read_text())
        if pip_fails:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    import shutil

    monkeypatch.setattr(launcher, "run", run)
    monkeypatch.setattr(shutil, "which", lambda name: uv if name == "uv" else None)
    return calls


def test_with_pip_present_an_install_is_constrained_to_the_lock(monkeypatch):
    launcher = _launcher()
    calls = _recording(launcher, monkeypatch)

    assert launcher.pip_install("xformers", "xFormers") is True

    export, pip = calls
    assert export[:2] == ["/usr/bin/uv", "export"] and "--frozen" in export
    assert export[export.index("--project") + 1] == str(REPO_ROOT)
    assert pip[:5] == [sys.executable, "-m", "pip", "install", "-U"]
    assert pip[5] == "-c" and pip[-2] == "xformers"
    assert "torch==2.10.0+cu130" in pip[-1]          # the constraints pip was given
    assert not Path(pip[6]).exists()                # a scratch file, removed after


def test_the_flags_still_reach_pip_beside_the_constraints(monkeypatch):
    launcher = _launcher()
    calls = _recording(launcher, monkeypatch)

    assert launcher.pip_install("flash-attn --no-build-isolation", "FlashAttention",
                                reinstall=True, no_deps=True) is True

    pip = calls[-1]
    assert "-c" in pip and "--force-reinstall" in pip and "--no-deps" in pip


def test_without_the_constraints_nothing_is_installed(monkeypatch, capsys):
    """No uv to export the lock, or an export that fails: refused before pip
    runs, saying why."""
    launcher = _launcher()
    calls = _recording(launcher, monkeypatch, uv=None)
    assert launcher.pip_install("xformers", "xFormers") is False
    assert calls == []
    assert "Not installing xFormers: uv is not on PATH" in capsys.readouterr().out

    calls = _recording(launcher, monkeypatch, export_fails=True)
    assert launcher.pip_install("xformers", "xFormers") is False
    assert [call[1] for call in calls] == ["export"]          # pip never ran
    assert "uv could not export uv.lock" in capsys.readouterr().out


def test_a_conflict_with_the_lock_is_a_refused_install(monkeypatch, capsys):
    """pip refuses an accelerator that needs another torch; the launcher says
    the locked set was kept and goes on without it."""
    launcher = _launcher()
    _recording(launcher, monkeypatch, pip_fails=True)

    assert launcher.pip_install("xformers", "xFormers") is False
    assert "the locked set was kept" in capsys.readouterr().out
