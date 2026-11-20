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


def _configured_specs(launcher):
    """Every install spec `configs/accelerators.json` gives an accelerator
    the launcher installs, as `choose_spec` returns them."""
    table = launcher.read_json(launcher.ACCEL_TABLE)
    specs = set()

    def leaves(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if not key.startswith("_"):
                    leaves(value)
        elif isinstance(node, str):
            specs.add(node)

    for accelerator in launcher.ACCELERATORS:
        leaves(table[accelerator.key])
    return sorted(specs)


def test_a_configured_spec_reaches_pip_as_separate_arguments(monkeypatch):
    """The recheck of d4209c4 (OPS-RECHECK-I1): an explicit --flash-attn on
    linux/x86_64, Python 3.12, torch 2.10.0, CUDA 13.0 gets the spec
    "flash-attn --no-build-isolation" from the table, and `pip_install`
    appended it as one argv element, which pip refuses ("Invalid
    requirement: 'flash-attn --no-build-isolation'"). The one installer now
    splits a spec into its requirement and its options; every spec the table
    holds reaches pip as separate arguments, after the launcher's own."""
    launcher = _launcher()
    spec = launcher.choose_spec(launcher.read_json(launcher.ACCEL_TABLE), "flash_attn",
                                torch_v="2.10.0", cuda_v="13.0", py_v="3.12",
                                os_name="linux", arch="x86_64")
    assert spec == "flash-attn --no-build-isolation"
    calls = _recording(launcher, monkeypatch)
    assert launcher.pip_install(spec, "FlashAttention", no_deps=True) is True
    pip = calls[-1][:-1]                              # the constraints text recorded last
    assert pip[-2:] == ["flash-attn", "--no-build-isolation"]
    assert not any(" " in part for part in pip)

    for spec in _configured_specs(launcher):
        calls = _recording(launcher, monkeypatch)
        assert launcher.pip_install(spec, "an accelerator") is True
        pip = calls[-1][:-1]
        assert pip[-len(spec.split()):] == spec.split(), spec
        assert not any(" " in part for part in pip), spec
