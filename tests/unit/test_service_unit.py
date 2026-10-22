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
    # Loopback, stated where an operator would change it (SPEC_4 B1, M5-1).
    assert app_args == ["--host", "127.0.0.1", "--port", "8264"]

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
    browser threads started, browser openers started beside an exec (their
    URLs), pip installs attempted, and main's status.
    """
    launcher = _launcher()
    record = {"exec": [], "run": [], "browsers": 0, "threads": [], "installs": [],
              "openers": []}

    class _Thread:
        def __init__(self, *args, **kwargs):
            record["threads"].append(kwargs)

        def start(self):
            record["browsers"] += 1

    def _execve(program, argv, environ):
        record["exec"].append(list(argv))
        record["exec_env"] = dict(environ)
        raise _ExecReachedError

    monkeypatch.setattr(launcher, "run", lambda cmd, check=False: (
        record["run"].append(list(cmd)) or subprocess.CompletedProcess(cmd, 0)))
    monkeypatch.setattr(launcher, "serve", lambda cmd: record["run"].append(list(cmd)) or 0)
    monkeypatch.setattr(launcher, "start_browser_opener",
                        lambda url, health: record["openers"].append((url, health)))
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
    record["environ"] = dict(launcher.os.environ)
    started = record["exec"] or record["run"]
    record["server"] = started[-1] if started else None
    if record["server"] is not None:
        # After python -m uvicorn APP, and before what the launcher always
        # appends last (`UVICORN_FORCED_ARGS`), which the tests below check.
        forced = launcher.UVICORN_FORCED_ARGS
        assert record["server"][-len(forced):] == forced
        record["app_args"] = record["server"][4:-len(forced)]
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


def test_the_launcher_binds_loopback_by_default_and_says_so(monkeypatch, capsys):
    """SPEC_4 B1 (M5-1): no --host given, uvicorn is given 127.0.0.1, and the
    launcher prints the address it binds beside the URLs it shows."""
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser"])
    assert done["app_args"][:2] == ["--host", "127.0.0.1"]
    out = capsys.readouterr().out
    assert "Listening on          : 127.0.0.1:8000 (loopback: this machine only)" in out


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.10", "", "myhost.lan"])
def test_a_network_bind_without_risk_acceptance_starts_nothing(monkeypatch, capsys, host):
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser", "--",
                                       f"--host={host}", "--port", "8264"])
    assert done["status"] == 2 and done["server"] is None
    out = capsys.readouterr().out
    assert "ERROR: refusing to bind" in out and "Nothing was started." in out
    assert "SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND=1" in out


def _seconds(value: str) -> float:
    """A systemd time span as TimeoutStopSec= takes it (``150``, ``150s``,
    ``2min 30s``), in seconds."""
    units = {"": 1, "s": 1, "sec": 1, "m": 60, "min": 60, "h": 3600}
    total = 0.0
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]*)", value):
        assert unit in units, f"unexpected time unit {unit!r} in {value!r}"
        total += float(number) * units[unit]
    return total


def test_stopping_the_unit_hands_the_job_s_stop_to_the_backend():
    """Root-cause review, P23-3: the default KillMode=control-group SIGTERMs
    every process of the service at once, the GPU job's main process and its
    workers too, ahead of the backend's own stop (SIGINT first, which a test
    run holds across publishing its report); and the stop timeout was the
    manager's default, which can be shorter than that stop. KillMode=mixed
    sends SIGTERM to the main process only, the server (the launcher execs
    into it, test_the_launcher_runs_the_units_command_by_becoming_the_server),
    and SIGKILLs the rest once it has exited; TimeoutStopSec is explicit and
    covers the backend's longest stop of its job, computed from the job
    runner's own constants, so a change there fails here."""
    from src.api.services import training_manager as tm  # builds no application

    assert _directive("KillMode") == "mixed"
    bound = tm.END_ACTIVE_JOB_SECONDS
    assert bound == (tm.STOP_GRACE_SECONDS + tm.TERMINATE_GRACE_SECONDS
                     + tm.KILL_GRACE_SECONDS + tm.RESULT_WAIT_SECONDS)
    # Room beyond the job's stop for the server's own shutdown and exit.
    assert _seconds(_directive("TimeoutStopSec")) >= bound + 20


def test_the_unit_does_not_restart_a_refused_bind(monkeypatch, capsys):
    """The launcher refuses its own configuration with exit status 2, which a
    restart cannot fix, so the unit must not restart on it."""
    assert _directive("Restart") == "on-failure"
    assert _directive("RestartPreventExitStatus").split() == ["2"]
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser", "--",
                                       "--host", "0.0.0.0"])
    assert done["status"] == 2 and done["server"] is None


def test_an_ipv6_bind_is_shown_with_brackets(monkeypatch, capsys):
    """`::1:8000` cannot be read as an address and a port; `[::1]:8000` can."""
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser", "--",
                                       "--host", "::1"])
    assert done["exec"] and done["app_args"][-2:] == ["--host", "::1"]
    assert "Listening on          : [::1]:8000 (loopback: this machine only)" in \
        capsys.readouterr().out


def test_risk_acceptance_starts_a_network_bind_and_says_it_is_no_control(monkeypatch, capsys):
    done = _run_launcher(
        monkeypatch, ["--no-auto-install", "--no-browser", "--", "--host", "0.0.0.0"],
        env={"SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND": "1"})
    assert done["exec"] and done["app_args"][-2:] == ["--host", "0.0.0.0"]
    out = capsys.readouterr().out
    assert "WARNING: binding all interfaces" in out and "not a control" in out
    assert "Listening on          : 0.0.0.0:8000 (all interfaces)" in out


def test_the_bind_guard_refuses_before_anything_is_installed(monkeypatch, capsys):
    """A refused bind starts nothing, a pip install of an accelerator
    included: the guard runs right after the arguments are read."""
    done = _run_launcher(monkeypatch, ["--no-browser", "--xformers", "--", "--host", "0.0.0.0"])
    assert done["status"] == 2 and done["server"] is None
    assert done["installs"] == []
    assert "ERROR: refusing to bind all interfaces" in capsys.readouterr().out


def test_the_host_bound_is_passed_to_application_startup(monkeypatch):
    """SPEC_4 §3.1's third enforcement point reads SHEPHERD_BIND_HOST, which
    the launcher sets to the host uvicorn binds, over any value inherited."""
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser"],
                         env={"SHEPHERD_BIND_HOST": "0.0.0.0"})
    assert done["exec_env"]["SHEPHERD_BIND_HOST"] == "127.0.0.1"
    done = _run_launcher(
        monkeypatch, ["--no-auto-install", "--no-browser", "--", "--host=0.0.0.0"],
        env={"SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND": "1"})
    assert done["exec_env"]["SHEPHERD_BIND_HOST"] == "0.0.0.0"


@pytest.mark.parametrize("app_args, env", [
    (["--fd", "3"], {}), (["--fd=3"], {}), ([], {"UVICORN_FD": "3"}),
    (["--uds", "/run/shepherd.sock", "--fd", "3"], {}),
], ids=["fd", "fd-equals", "fd-from-environment", "fd-beside-uds"])
def test_a_socket_handed_to_uvicorn_counts_as_a_network_bind(monkeypatch, capsys, app_args,
                                                             env):
    """uvicorn listens on an inherited socket (--fd, or UVICORN_FD) before
    --host, and a socket unit's ListenStream binds every interface: its
    address is not read here, so it needs the risk acceptance, and the
    launcher never calls it loopback."""
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser", "--", *app_args],
                         env=env)
    assert done["status"] == 2 and done["server"] is None
    out = capsys.readouterr().out
    assert "ERROR: refusing to serve on the inherited socket (file descriptor 3)" in out
    assert "loopback: this machine only" not in out

    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser", "--", *app_args],
                         env={**env, "SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND": "1"})
    assert done["exec"]
    out = capsys.readouterr().out
    assert "WARNING: serving on the inherited socket (file descriptor 3)" in out
    assert ("Listening on          : the inherited socket (file descriptor 3), whose address "
            "is not read here") in out
    assert "http://127.0.0.1:8000/ui" not in out and "no URL is shown" in out
    assert "SHEPHERD_BIND_HOST" not in done["exec_env"]


@pytest.mark.parametrize("app_args, env", [
    (["--uds", "/run/shepherd.sock"], {}), (["--uds=/run/shepherd.sock"], {}),
    ([], {"UVICORN_UDS": "/run/shepherd.sock"}),
], ids=["uds", "uds-equals", "uds-from-environment"])
def test_a_unix_socket_is_shown_as_its_path_not_as_a_url(monkeypatch, capsys, app_args, env):
    """A unix socket is reachable on this machine only (no acceptance needed),
    has no URL to show or open, and passes no host to startup."""
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--", *app_args],
                         env={**env, "SHEPHERD_BIND_HOST": "0.0.0.0"})
    assert done["exec"] and not done["run"]
    assert done["browsers"] == 0 and done["openers"] == []
    out = capsys.readouterr().out
    assert "Listening on          : unix socket /run/shepherd.sock (this machine only" in out
    assert "http://" not in out and "No browser is opened" in out
    assert "SHEPHERD_BIND_HOST" not in done["environ"]


def test_an_empty_socket_variable_is_unset_as_uvicorn_reads_it(monkeypatch, capsys):
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser"],
                         env={"UVICORN_FD": "", "UVICORN_UDS": ""})
    assert done["exec"]
    assert "Listening on          : 127.0.0.1:8000 (loopback" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["0", "true", "yes", " 1"])
def test_only_the_exact_value_accepts_the_risk(monkeypatch, value):
    done = _run_launcher(
        monkeypatch, ["--no-auto-install", "--no-browser", "--", "--host", "0.0.0.0"],
        env={"SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND": value})
    assert done["status"] == 2 and done["server"] is None


def test_uvicorn_never_trusts_proxy_headers(monkeypatch):
    """`--no-proxy-headers` is last, so a `--proxy-headers` passed through
    cannot turn it back on: uvicorn would rewrite the client address from
    X-Forwarded-For for a peer on 127.0.0.1."""
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser", "--",
                                       "--proxy-headers", "--port", "8264"])
    assert done["server"][-1] == "--no-proxy-headers"
    assert done["server"].index("--proxy-headers") < done["server"].index("--no-proxy-headers")


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
        (["--host", "0.0.0.0", "--port", "8000", "--host="], "http://127.0.0.1:8000"),
        (["--host", "0.0.0.0", "--port", "8000", "--host", "[::1]"], "http://[::1]:8000"),
    ],
    ids=["defaults", "last-port-wins", "port-equals-form", "host-equals-form",
         "last-host-wins-wildcard", "ipv6-wildcard", "ipv6-literal", "empty-host-is-wildcard",
         "already-bracketed"],
)
def test_the_displayed_address_follows_uvicorns_own_rules(uvicorn_args, expected):
    # uvicorn takes the last of a repeated option in either spelling. A display
    # that missed --port=N logged 8000 again -- the F1 mismatch -- and an IPv6
    # host needs brackets to be a URL at all.
    assert _launcher().display_base_url(uvicorn_args) == expected


def test_with_a_browser_to_open_the_launcher_still_becomes_the_server(monkeypatch):
    """Round 3 of the review fixes, P3-2: the browser mode used to wait beside
    the server, so a SIGTERM sent to the PID that was started ended the
    launcher alone and the server served on with its job. On POSIX it execs
    as the headless mode does, and the browser opener is a helper started
    first, given the URLs."""
    done = _run_launcher(monkeypatch, ["--no-auto-install"])
    assert done["exec"] and not done["run"]
    assert done["threads"] == [] and done["browsers"] == 0
    assert done["openers"] == [("http://127.0.0.1:8000/ui", "http://127.0.0.1:8000/health")]


def test_where_exec_would_change_the_pid_the_browser_opens_from_a_thread(monkeypatch):
    # The control for the tests around it: the stubs do see the other path.
    done = _run_launcher(monkeypatch, ["--no-auto-install"], hand_over=False)
    assert done["run"] and not done["exec"] and done["openers"] == []
    assert done["browsers"] == 1
    assert done["status"] == 0
    # A daemon: the opener waits for /health with no deadline, so only the
    # process ending when the server exits ends it.
    (opener,) = done["threads"]
    assert opener["daemon"] is True
    assert opener["target"].__name__ == "_open_browser_when_ready"
    assert opener["args"] == ("http://127.0.0.1:8000/ui", "http://127.0.0.1:8000/health")


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


def test_the_unit_names_each_setting_the_operator_sets_for_the_service():
    """Round 3 of the review fixes, P3-7: the unit offered only the deployment
    mode, whose comment read as if M1 alone enabled training and the lists,
    while the workspaces root (the GPU lock's one name) and the registry,
    which the deployment guide tells the CLI shell to copy from the unit,
    had no line. Each is there to set, commented, named by the settings'
    own constants, and the M1 comment says it is not enough."""
    from src.config import server_settings  # imports nothing of the application

    text = _unit_text()
    for name in (server_settings.ENV_DEPLOYMENT_MODE, server_settings.ENV_WORKSPACES_ROOT,
                 server_settings.ENV_REGISTRY):
        assert re.search(rf"^#?Environment={name}=\S", text, flags=re.MULTILINE), \
            f"the unit has no Environment={name}= line"
    assert "M1 is necessary, not sufficient" in text
    assert server_settings.GPU_LOCK_FILE_NAME in text


def test_the_unit_leaves_the_allocator_to_the_saved_preset():
    # A value here would be an explicit override, kept on every start and
    # restart, and the preset saved in Runtime Settings would never apply.
    # Which preset the launcher resolves is test_runtime_presets.py's subject.
    env = _environment()
    assert "PYTORCH_ALLOC_CONF" not in env
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    assert _constant(PRESETS, "ALLOC_SOURCE_ENV") not in env
