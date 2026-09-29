"""
The shipped systemd unit starts the application the launcher starts.

It named `app.main:app`, a module that has never existed here, with the
system interpreter, which does not have the project's dependencies. Nothing
failed, because nothing read the file. SPEC_4 §4 item A4.

Read as text on purpose: importing `src.api` builds the whole application,
Gradio dashboard included. Every read names UTF-8, because the files carry
non-ASCII text and the default encoding is the locale's -- cp950 on a
Traditional Chinese Windows, where an unnamed read fails before any assertion.
"""
import ast
import importlib.util
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UNIT = REPO_ROOT / "scripts" / "service" / "systemd" / "shepherd.service"
LAUNCHER = REPO_ROOT / "scripts" / "launch" / "shep_launch.py"
PRESETS = REPO_ROOT / "src" / "config" / "runtime_presets.py"


def _unit_text() -> str:
    return UNIT.read_text(encoding="utf-8")


def _directive(name: str) -> str:
    matches = re.findall(rf"^{name}=(.*)$", _unit_text(), flags=re.MULTILINE)
    assert len(matches) == 1, f"expected one {name}= in {UNIT.name}, found {matches}"
    return matches[0].strip()


def _environment() -> dict:
    pairs = re.findall(r"^Environment=(\S+?)=(.*)$", _unit_text(), flags=re.MULTILINE)
    return {key: value.strip() for key, value in pairs}


def _constant(source: Path, name: str) -> str:
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{source.name} no longer defines {name}")


def _launcher():
    spec = importlib.util.spec_from_file_location("shep_launch", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_unit_starts_through_the_launcher_with_service_flags():
    argv = shlex.split(_directive("ExecStart"))
    working_directory = _directive("WorkingDirectory")
    # The launcher, not a second entry point: one start path, so a systemd
    # start gets what launch_shepherd.sh gets.
    assert argv[1] == f"{working_directory}/scripts/launch/shep_launch.py"
    own, app_args = argv[2:argv.index("--")], argv[argv.index("--") + 1:]
    assert "--no-auto-install" in own
    assert "--no-browser" in own
    assert app_args == ["--port", "8264"]

    app = _constant(LAUNCHER, "UVICORN_APP")
    module, _, attr = app.partition(":")
    source = REPO_ROOT.joinpath(*module.split(".")).with_suffix(".py")
    assert source.is_file(), f"{module} does not exist"
    assigned = {
        t.id
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    assert attr in assigned, f"{module} assigns no top-level {attr!r}"


def _run_launcher(monkeypatch, launcher_args):
    """Run the launcher's own main with the server and the browser stubbed."""
    launcher = _launcher()
    commands, browsers = [], []

    class _Thread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            browsers.append(True)

    monkeypatch.setattr(launcher, "run", lambda cmd, check=False: (
        commands.append(cmd) or subprocess.CompletedProcess(cmd, 0)))
    monkeypatch.setattr(launcher.threading, "Thread", _Thread)
    monkeypatch.setattr(launcher, "load_runtime_settings", lambda: {})
    monkeypatch.setattr(launcher.os, "environ", {})
    monkeypatch.setattr(sys, "argv", ["shep_launch.py", *launcher_args])
    assert launcher.main() == 0
    return commands, browsers


def test_the_launcher_runs_the_units_command_and_opens_no_browser(monkeypatch):
    unit_args = shlex.split(_directive("ExecStart"))[2:]
    commands, browsers = _run_launcher(monkeypatch, unit_args)
    assert browsers == []
    # uvicorn takes the last of a repeated option, so the unit's port is the
    # one served, after the launcher's defaults.
    assert commands[-1][-2:] == ["--port", "8264"]
    assert commands[-1][1:4] == ["-m", "uvicorn", _constant(LAUNCHER, "UVICORN_APP")]


def test_without_no_browser_the_launcher_still_opens_one(monkeypatch):
    # The control for the test above: the stub does see the browser thread.
    _commands, browsers = _run_launcher(monkeypatch, ["--no-auto-install"])
    assert browsers == [True]


def test_the_unit_runs_the_project_interpreter_in_the_user_manager():
    interpreter = shlex.split(_directive("ExecStart"))[0]
    working_directory = _directive("WorkingDirectory")
    assert interpreter == f"{working_directory}/.venv/bin/python"
    # %h is the home of the service manager's user; with the repository under a
    # user's home this is installed as a user unit, and the user manager has
    # default.target, not multi-user.target.
    assert working_directory.startswith("%h/")
    assert _directive("WantedBy") == "default.target"


def test_the_unit_leaves_the_allocator_to_the_saved_preset():
    # A value here would be an explicit override, kept on every start and
    # restart, and the preset saved in Runtime Settings would never apply.
    # Which preset the launcher resolves is test_runtime_presets.py's subject.
    env = _environment()
    assert "PYTORCH_ALLOC_CONF" not in env
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    assert _constant(PRESETS, "ALLOC_SOURCE_ENV") not in env
