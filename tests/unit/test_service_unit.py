"""
The shipped systemd unit starts the application the launcher starts.

It named `app.main:app`, a module that has never existed here, with the
system interpreter, which does not have the project's dependencies. Nothing
failed, because nothing read the file. SPEC_4 §4 item A4.

Read as text on purpose: importing `src.api` builds the whole application,
Gradio dashboard included.
"""
import ast
import re
import shlex
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UNIT = REPO_ROOT / "scripts" / "service" / "systemd" / "shepherd.service"
LAUNCHER = REPO_ROOT / "scripts" / "launch" / "shep_launch.py"
BACKEND_CONTROL = REPO_ROOT / "src" / "api" / "services" / "backend_control.py"


def _directive(name: str) -> str:
    matches = re.findall(rf"^{name}=(.*)$", UNIT.read_text(), flags=re.MULTILINE)
    assert len(matches) == 1, f"expected one {name}= in {UNIT.name}, found {matches}"
    return matches[0].strip()


def _environment() -> dict:
    pairs = re.findall(r"^Environment=(\S+?)=(.*)$", UNIT.read_text(), flags=re.MULTILINE)
    return {key: value.strip() for key, value in pairs}


def _constant(source: Path, name: str) -> str:
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{source.name} no longer defines {name}")


def test_the_unit_serves_the_app_the_launcher_serves():
    argv = shlex.split(_directive("ExecStart"))
    assert argv[1:3] == ["-m", "uvicorn"]
    app = argv[3]
    assert app == _constant(LAUNCHER, "UVICORN_APP")

    module, _, attr = app.partition(":")
    source = REPO_ROOT.joinpath(*module.split(".")).with_suffix(".py")
    assert source.is_file(), f"{module} does not exist"
    assigned = {
        t.id
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    assert attr in assigned, f"{module} assigns no top-level {attr!r}"


def test_the_unit_runs_the_project_interpreter_in_the_user_manager():
    interpreter = shlex.split(_directive("ExecStart"))[0]
    working_directory = _directive("WorkingDirectory")
    assert interpreter == f"{working_directory}/.venv/bin/python"
    # %h is the home of the service manager's user, so this is a user unit,
    # and the user manager has default.target, not multi-user.target.
    assert working_directory.startswith("%h/")
    assert _directive("WantedBy") == "default.target"


def test_the_unit_starts_with_the_allocator_the_launcher_applies():
    # A bare uvicorn start applies no preset, and the native allocator is the
    # one this project measured fragmenting under HGT training.
    from src.config.runtime_presets import ALLOCATOR_PRESETS, DEFAULT_ALLOCATOR

    env = _environment()
    assert env.get("PYTORCH_ALLOC_CONF") == ALLOCATOR_PRESETS[DEFAULT_ALLOCATOR]
    # "preset", not absent: Restart Backend then re-resolves the saved preset
    # instead of preserving this value as an explicit override.
    assert env.get(_constant(BACKEND_CONTROL, "ALLOC_SOURCE_ENV")) == "preset"
