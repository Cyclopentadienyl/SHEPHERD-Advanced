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

import pytest

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


class _ExecReachedError(Exception):
    """Raised by the execve stub: a real exec does not return either."""


def _run_launcher(monkeypatch, launcher_args, env=None, hand_over=True):
    """Run the launcher's own main with the server, installs and browser stubbed.

    Returns what it did: the server command, whether it exec'd or waited,
    browser threads started, pip installs attempted, and main's status.
    """
    launcher = _launcher()
    record = {"exec": [], "run": [], "browsers": 0, "installs": []}

    class _Thread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            record["browsers"] += 1

    def _execve(program, argv, environ):
        record["exec"].append(list(argv))
        raise _ExecReachedError

    monkeypatch.setattr(launcher, "run", lambda cmd, check=False: (
        record["run"].append(list(cmd)) or subprocess.CompletedProcess(cmd, 0)))
    monkeypatch.setattr(launcher, "pip_install", lambda spec, name, **kwargs: (
        record["installs"].append(name) or False))
    # Every accelerator absent, so a requested one reaches the install decision.
    monkeypatch.setattr(launcher, "have_module", lambda module: False)
    monkeypatch.setattr(launcher.threading, "Thread", _Thread)
    monkeypatch.setattr(launcher.os, "execve", _execve)
    monkeypatch.setattr(launcher, "load_runtime_settings", lambda: {})
    # Set either way, so every host tests both hand-overs; which one the host
    # really gets is test_can_hand_over_follows_the_platform's subject.
    monkeypatch.setattr(launcher, "can_hand_over", lambda: hand_over)
    monkeypatch.setattr(launcher.os, "environ", dict(env or {}))
    monkeypatch.setattr(sys, "argv", ["shep_launch.py", *launcher_args])
    try:
        record["status"] = launcher.main()
    except _ExecReachedError:
        record["status"] = None
    record["server"] = (record["exec"] or record["run"])[-1]
    record["app_args"] = record["server"][4:]  # after python -m uvicorn APP
    return record


LAUNCHER_FLAGS = ("--no-auto-install", "--no-browser", "--xformers", "--flash-attn")


def test_the_launcher_runs_the_units_command_by_becoming_the_server(monkeypatch):
    done = _run_launcher(monkeypatch, shlex.split(_directive("ExecStart"))[2:])
    # Headless on POSIX: exec, so nothing is left waiting beside the server.
    assert done["exec"] and not done["run"]
    assert done["browsers"] == 0
    assert done["server"][1:4] == ["-m", "uvicorn", _constant(LAUNCHER, "UVICORN_APP")]
    # uvicorn takes the last of a repeated option, so the unit's port is the
    # one served, after the launcher's defaults.
    assert done["app_args"][-2:] == ["--port", "8264"]


def test_the_launcher_logs_the_port_the_unit_serves_on(monkeypatch, capsys):
    # The access points used to be logged as port 8000 whatever --port said, so
    # the unit's journal told an operator to bookmark a port nothing listened on.
    _run_launcher(monkeypatch, shlex.split(_directive("ExecStart"))[2:])
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8264/docs" in out
    assert "http://127.0.0.1:8264/ui" in out
    assert ":8000/" not in out


@pytest.mark.parametrize(
    "uvicorn_args, expected",
    [
        (["--host", "0.0.0.0", "--port", "8000"], "http://127.0.0.1:8000"),
        (["--host", "0.0.0.0", "--port", "8000", "--port", "8264"], "http://127.0.0.1:8264"),
        (["--host", "0.0.0.0", "--port", "8000", "--port=8264"], "http://127.0.0.1:8264"),
        (["--host", "0.0.0.0", "--host=10.0.0.5", "--port", "8000"], "http://10.0.0.5:8000"),
        (["--host", "10.0.0.5", "--host", "0.0.0.0", "--port", "8000"], "http://127.0.0.1:8000"),
        (["--host", "0.0.0.0", "--port", "8000", "--host", "::"], "http://[::1]:8000"),
        (["--host", "0.0.0.0", "--port", "8000", "--host", "fd00::5"], "http://[fd00::5]:8000"),
    ],
    ids=["defaults", "last-port-wins", "port-equals-form", "host-equals-form",
         "last-host-wins-wildcard", "ipv6-wildcard", "ipv6-literal"],
)
def test_the_displayed_address_follows_uvicorns_own_rules(uvicorn_args, expected):
    # uvicorn takes the last of a repeated option in either spelling. A display
    # that missed --port=N logged 8000 again -- the F1 mismatch -- and an IPv6
    # host needs brackets to be a URL at all.
    assert _launcher().display_base_url(uvicorn_args) == expected


def test_without_no_browser_the_launcher_waits_and_opens_one(monkeypatch):
    # The control for the test above: the stubs do see the other path.
    done = _run_launcher(monkeypatch, ["--no-auto-install"])
    assert done["run"] and not done["exec"]
    assert done["browsers"] == 1
    assert done["status"] == 0


def test_where_exec_would_change_the_pid_the_launcher_waits(monkeypatch):
    # Windows: os.exec* would end this process and start another, so a
    # supervisor would lose the PID. Headless there still waits.
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser"], hand_over=False)
    assert done["run"] and not done["exec"]
    assert done["browsers"] == 0


@pytest.mark.parametrize("os_name, expected", [("posix", True), ("nt", False)])
def test_can_hand_over_follows_the_platform(monkeypatch, os_name, expected):
    launcher = _launcher()
    monkeypatch.setattr(launcher.os, "name", os_name)
    assert launcher.can_hand_over() is expected


def test_a_failed_exec_is_a_non_zero_exit(monkeypatch):
    launcher = _launcher()

    def _fails(*args):
        raise OSError("no such file")

    monkeypatch.setattr(launcher.os, "execve", _fails)
    assert launcher.exec_server(["/nonexistent/python", "-m", "uvicorn"]) == 1


@pytest.mark.parametrize("name", ["SHEP_COMMANDLINE_ARGS", "COMMANDLINE_ARGS"])
def test_app_arguments_in_the_environment_do_not_swallow_the_command_line(monkeypatch, name):
    # Both sources carry app arguments. Each is split at its own "--", so the
    # command line's launcher flags stay launcher flags.
    done = _run_launcher(
        monkeypatch,
        ["--no-auto-install", "--no-browser", "--", "--port", "8264"],
        env={name: "--xformers -- --port 9000"},
    )
    assert done["installs"] == []          # --no-auto-install honoured
    assert done["browsers"] == 0           # --no-browser honoured
    assert not set(LAUNCHER_FLAGS) & set(done["app_args"])
    # Environment first, command line last: the command line's port wins.
    assert done["app_args"][-4:] == ["--port", "9000", "--port", "8264"]


def test_launcher_flags_in_the_environment_still_apply(monkeypatch):
    done = _run_launcher(
        monkeypatch, ["--", "--port", "8264"],
        env={"SHEP_COMMANDLINE_ARGS": "--no-auto-install --no-browser --xformers"},
    )
    assert done["installs"] == []
    assert done["browsers"] == 0
    assert done["app_args"][-2:] == ["--port", "8264"]


def test_without_no_auto_install_a_missing_accelerator_is_installed(monkeypatch):
    # The control for the two tests above: the install spy does fire.
    done = _run_launcher(monkeypatch, ["--no-browser"], env={"SHEP_COMMANDLINE_ARGS": "--xformers"})
    assert done["installs"] == ["xFormers"]


def test_only_one_environment_source_is_read(monkeypatch):
    both = {"SHEP_COMMANDLINE_ARGS": "-- --port 9000", "COMMANDLINE_ARGS": "-- --port 7000"}
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser"], env=both)
    assert "9000" in done["app_args"] and "7000" not in done["app_args"]

    fallback = {"COMMANDLINE_ARGS": "-- --port 7000"}
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser"], env=fallback)
    assert done["app_args"][-2:] == ["--port", "7000"]


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
