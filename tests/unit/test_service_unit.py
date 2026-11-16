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
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import types
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


def _run_launcher(monkeypatch, launcher_args, env=None, hand_over=True, busy_ports=(),
                  terminal=True, output_kept=False, launcher=None, keep_lock=False,
                  browser=True):
    """Run the launcher's own main with the server, installs and browser stubbed.

    Returns what it did: the server command, whether it exec'd or waited,
    browser threads started, browser openers started beside an exec (their
    URLs and options), pip installs attempted, pages opened, whether it
    changed to the repository, and main's status. `busy_ports` are the ports
    another program holds (the port check is stubbed, so no socket is bound
    and the host's own ports never decide a test); `terminal` is whether the
    start has one, `output_kept` whether its output already goes to a file or
    a pipe. The instance lock a start hands to its exec (`handed`) is
    released afterwards, unless `keep_lock`. `browser` is what the stubbed
    ``webbrowser.open`` answers: False, as on a host where it finds none.
    """
    launcher = launcher or _launcher()
    record = {"exec": [], "run": [], "browsers": 0, "threads": [], "installs": [],
              "openers": [], "opened": [], "chdir": 0, "handed": []}
    real_hand = launcher.instances.hand_to_exec
    monkeypatch.setattr(launcher.instances, "hand_to_exec",
                        lambda fd: (record["handed"].append(fd), real_hand(fd)))

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
                        lambda url, health, **options: record["openers"].append(
                            (url, health, *([options] if any(options.values()) else []))))
    monkeypatch.setattr(launcher, "port_problem", lambda host, port: (
        launcher.IN_USE if port in busy_ports else None))
    monkeypatch.setattr(launcher, "enter_repository",
                        lambda: record.__setitem__("chdir", record["chdir"] + 1))
    monkeypatch.setattr(launcher, "has_terminal", lambda: terminal)
    monkeypatch.setattr(launcher, "output_is_kept", lambda: output_kept)
    monkeypatch.setattr(launcher, "webbrowser", types.SimpleNamespace(
        open=lambda url: record["opened"].append(url) or browser))
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
    if not keep_lock:
        for fd in record["handed"]:
            os.close(fd)
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


POSIX_SCRIPT = pytest.mark.skipif(os.name != "posix", reason="launch_shepherd.sh is the POSIX "
                                   "start; Windows starts through launch_shepherd.cmd")


def _script_checkout(root: Path, venv: bool = True) -> Path:
    """A copy of launch_shepherd.sh in `root`, beside a stub virtual
    environment (unless `venv` is False) and a stub launcher that says its
    PID, working directory and arguments, so the script is tested without the
    checkout's .venv, which the end-to-end case in
    tests/integration/test_backend_stops_its_job.py needs."""
    script = REPO_ROOT / "launch_shepherd.sh"
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy(script, root / script.name)
    if venv:
        venv_bin = root / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").symlink_to(sys.executable)
        (venv_bin / "activate").write_text(f'PATH={shlex.quote(str(venv_bin))}:"$PATH"\n',
                                           encoding="utf-8")
    launcher = root / "scripts" / "launch" / "shep_launch.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("import json, os, sys\n"
                        "print(json.dumps({'pid': os.getpid(), 'cwd': os.getcwd(), "
                        "'launcher': sys.argv[0], 'python': sys.executable, "
                        "'argv': sys.argv[1:]}))\n", encoding="utf-8")
    return root / script.name


@POSIX_SCRIPT
def test_launch_shepherd_sh_puts_the_launcher_in_the_pid_that_was_started(tmp_path):
    """Round 3 of the review fixes, P3-2: the script ran the launcher as its
    child, so a SIGTERM sent to the PID a supervisor started reached bash
    alone, and the server served on with its job. It execs the launcher now,
    which execs into uvicorn (the tests above), as the local start
    (``--local-entry`` first, UI-06)."""
    script = _script_checkout(tmp_path)
    args = ["--no-auto-install", "--", "--port", "8264"]
    process = subprocess.Popen(["bash", script.name, *args], cwd=tmp_path,
                               stdout=subprocess.PIPE, text=True)
    out, _ = process.communicate(timeout=60)
    said = json.loads(out.strip().splitlines()[-1])
    assert said["pid"] == process.pid, (
        f"the launcher ran as pid {said['pid']}, not as the pid that was started "
        f"({process.pid}): a SIGTERM sent to that pid would not reach it")
    assert said["argv"] == ["--local-entry", *args]
    assert process.returncode == 0


@POSIX_SCRIPT
@pytest.mark.parametrize("through", ["path", "relative-link", "absolute-link"])
def test_launch_shepherd_sh_starts_from_any_directory(tmp_path, through):
    """UI-06a: started from another directory (a desktop entry's, a file
    manager's, a link on the desktop), the script looked for .venv in that
    directory, said "Virtual environment not found" and started nothing.
    It finds its own directory, links followed, and runs the checkout's
    launcher with the checkout's virtual environment; the working directory
    is left as it was started, so a relative --settings-file keeps its
    meaning until the launcher has read it (the launcher then changes to
    the repository: test_the_launcher_changes_to_the_repository_...)."""
    script = _script_checkout(tmp_path / "checkout")
    elsewhere = tmp_path / "desktop"
    elsewhere.mkdir()
    if through == "path":
        argv = ["bash", str(script)]
    else:
        link = elsewhere / "SHEPHERD"
        link.symlink_to(os.path.relpath(script, elsewhere) if through == "relative-link"
                        else script)
        argv = ["bash", str(link)]
    done = subprocess.run([*argv, "--print-plan"], cwd=elsewhere, capture_output=True,
                          text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    said = json.loads(done.stdout.strip().splitlines()[-1])
    checkout = script.parent.resolve()
    assert Path(said["launcher"]) == checkout / "scripts" / "launch" / "shep_launch.py"
    assert Path(said["python"]).parent == checkout / ".venv" / "bin"
    assert Path(said["cwd"]) == elsewhere.resolve()
    assert said["argv"] == ["--local-entry", "--print-plan"]


@POSIX_SCRIPT
def test_without_a_virtual_environment_or_a_terminal_a_page_says_why(tmp_path):
    """UI-06e: a start from the desktop entry has no terminal to show "run
    deploy.sh first" in. The script appends it to the launch log and opens a
    local page saying what failed, what to do and where the log is; every
    value escaped. With a terminal it only says so there."""
    script = _script_checkout(tmp_path / "check<out>", venv=False)
    state = tmp_path / "state"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    opened = tmp_path / "opened"
    (bin_dir / "xdg-open").write_text(f'#!/bin/sh\necho "$1" >> {shlex.quote(str(opened))}\n',
                                      encoding="utf-8")
    (bin_dir / "xdg-open").chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path / "home"),
           "SHEPHERD_STATE_DIR": str(state)}
    done = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 1
    assert "was not found" in done.stdout
    page = state / "logs" / "launch-error.html"
    for _ in range(50):  # xdg-open is started in the background
        if opened.exists():
            break
        time.sleep(0.1)
    assert opened.read_text(encoding="utf-8").strip() == f"file://{page}"
    text = page.read_text(encoding="utf-8")
    assert "SHEPHERD-Advanced did not start" in text
    assert "check&lt;out&gt;/.venv was not found" in text and "<out>" not in text
    assert "Run deploy.sh in" in text and str(state / "logs" / "launch.log") in text
    assert "was not found" in (state / "logs" / "launch.log").read_text(encoding="utf-8")
    assert (page.stat().st_mode & 0o777) == 0o600


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


# ---------------------------------------------------------------------------
# The local start (UI-06): any directory, its mode, its settings file, the
# port, one instance per configuration, a start without a terminal, and the
# desktop entry
# ---------------------------------------------------------------------------
def test_the_launcher_changes_to_the_repository_before_it_starts_the_server(monkeypatch,
                                                                            tmp_path):
    """UI-06a: uvicorn imports `src` from its working directory, so a launcher
    started elsewhere ended with "No module named 'src'". It changes to the
    repository first, also for the systemd unit's start."""
    done = _run_launcher(monkeypatch, shlex.split(_directive("ExecStart"))[2:])
    assert done["chdir"] == 1 and done["exec"]
    launcher = _launcher()
    monkeypatch.chdir(tmp_path)
    launcher.enter_repository()
    assert Path.cwd() == launcher.REPO_ROOT


@pytest.mark.parametrize("app_args, env, marked", [
    ([], {}, True),
    (["--host", "localhost"], {}, True),
    (["--host", "0.0.0.0"], {"SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND": "1"}, False),
    (["--uds", "/run/shepherd.sock"], {}, False),
], ids=["loopback", "localhost", "network-bind", "unix-socket"])
def test_the_local_start_marks_only_a_loopback_bind(monkeypatch, app_args, env, marked):
    """D06: ``--local-entry`` gives the server `SHEPHERD_LOCAL_ENTRY=1` (an
    unset mode is then M1 for this computer) only on a loopback bind; D8
    still checks every request."""
    done = _run_launcher(monkeypatch, ["--local-entry", "--no-auto-install", "--no-browser",
                                       "--", *app_args], env=env)
    assert done["exec"]
    assert (done["exec_env"].get("SHEPHERD_LOCAL_ENTRY") == "1") is marked


def test_the_local_start_mark_is_never_inherited(monkeypatch):
    """The mark is the launcher's: one in its environment (a terminal of a
    local start, say) never reaches a server started without --local-entry,
    as the systemd unit starts it."""
    done = _run_launcher(monkeypatch, shlex.split(_directive("ExecStart"))[2:],
                         env={"SHEPHERD_LOCAL_ENTRY": "1", "SHEPHERD_INSTANCE_ID": "inherited",
                              "SHEPHERD_INSTANCE_FILE": "/tmp/inherited.json"})
    for name in ("SHEPHERD_LOCAL_ENTRY", "SHEPHERD_INSTANCE_ID", "SHEPHERD_INSTANCE_FILE"):
        assert name not in done["exec_env"]


def test_an_explicit_read_only_mode_is_said_and_kept_at_a_local_start(monkeypatch, capsys):
    done = _run_launcher(monkeypatch, ["--local-entry", "--no-auto-install", "--no-browser"],
                         env={"SHEPHERD_DEPLOYMENT_MODE": "read-only"})
    assert done["exec_env"]["SHEPHERD_DEPLOYMENT_MODE"] == "read-only"
    out = capsys.readouterr().out
    assert ("Operator controls     : off. this deployment is read-only: the deployment mode set "
            "at start is not the operator mode") in out and "never overrides it" in out
    done = _run_launcher(monkeypatch, ["--local-entry", "--no-auto-install", "--no-browser"])
    assert "Operator controls     : on for requests made on this computer (the local start)" \
        in capsys.readouterr().out


def test_settings_file_names_the_service_s_saved_settings(monkeypatch, tmp_path, capsys):
    """``--settings-file`` (an A/B slot) is the service's
    `SHEPHERD_SETTINGS_FILE`, over one the environment names; a relative
    path is read against the directory the launcher was started in, not the
    repository it changes to."""
    monkeypatch.chdir(tmp_path)
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser", "--settings-file",
                                       "slot-b/deployment.json"],
                         env={"SHEPHERD_SETTINGS_FILE": "/elsewhere/deployment.json"})
    assert done["exec_env"]["SHEPHERD_SETTINGS_FILE"] == str(tmp_path / "slot-b"
                                                             / "deployment.json")
    assert (f"Saved settings file   : {tmp_path / 'slot-b' / 'deployment.json'} "
            "(--settings-file)") in capsys.readouterr().out
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser"],
                         env={"SHEPHERD_SETTINGS_FILE": "/elsewhere/deployment.json"})
    assert done["exec_env"]["SHEPHERD_SETTINGS_FILE"] == "/elsewhere/deployment.json"


def test_a_default_port_in_use_moves_to_the_next_free_one_and_says_so(monkeypatch, capsys):
    """Plan §4.5: with no --port, a port 8000 another program holds made
    uvicorn exit after the whole start ("address already in use"). The
    launcher takes the next free one and says which, in the URLs too."""
    done = _run_launcher(monkeypatch, ["--no-auto-install"], busy_ports={8000, 8001})
    assert done["app_args"][:4] == ["--host", "127.0.0.1", "--port", "8002"]
    assert done["openers"][0][:2] == ("http://127.0.0.1:8002/ui", "http://127.0.0.1:8002/health")
    out = capsys.readouterr().out
    assert "Ports 8000 to 8001 are in use by other programs, so the service uses port 8002." in out
    assert "http://127.0.0.1:8002/ui" in out and ":8000/" not in out


def test_every_fallback_port_in_use_starts_nothing(monkeypatch, capsys):
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser"],
                         busy_ports=set(range(8000, 8020)))
    assert done["status"] == 1 and done["server"] is None
    assert "Ports 8000 to 8019 are all in use by other programs" in capsys.readouterr().out


def test_a_port_passed_in_use_is_refused_with_the_reason(monkeypatch, capsys):
    """A port the operator chose is that port or nothing (the unit's 8264):
    never moved to another one an operator did not ask for."""
    argv = shlex.split(_directive("ExecStart"))[2:]
    done = _run_launcher(monkeypatch, argv, busy_ports={8264})
    assert done["status"] == 1 and done["server"] is None
    out = capsys.readouterr().out
    assert "Port 8264, passed with --port, is in use by another program, so nothing was " \
           "started." in out
    done = _run_launcher(monkeypatch, argv, busy_ports={8000})
    assert done["exec"] and done["app_args"][-2:] == ["--port", "8264"]


def test_port_problem_tells_a_port_in_use_from_a_free_one():
    """The real check, on loopback: a port a listening socket holds is in
    use, the same port once closed is free (SO_REUSEADDR, as uvicorn binds)."""
    import socket as sockets

    launcher = _launcher()
    holder = sockets.socket(sockets.AF_INET, sockets.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        assert launcher.port_problem("127.0.0.1", port) == launcher.IN_USE
    finally:
        holder.close()
    assert launcher.port_problem("127.0.0.1", port) is None


def _state_env(tmp_path, **extra):
    return {"SHEPHERD_STATE_DIR": str(tmp_path / "state"),
            "SHEPHERD_SETTINGS_FILE": str(tmp_path / "settings" / "deployment.json"), **extra}


def _instance_files(launcher, tmp_path, settings=None):
    settings = settings or tmp_path / "settings" / "deployment.json"
    return launcher.instances.instance_files(
        tmp_path / "state", launcher.instances.instance_key(launcher.REPO_ROOT, settings))


def test_a_start_holds_its_configuration_s_lock_through_the_exec(monkeypatch, tmp_path):
    """Plan §4.5: the start takes its configuration's instance lock and
    hands it to the server it becomes (an inheritable descriptor at the
    exec), writes the info file (0600) with what a second start needs, and
    gives the server its id and file."""
    launcher = _launcher()
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser"],
                         env=_state_env(tmp_path), busy_ports={8000}, launcher=launcher,
                         keep_lock=True)
    handed = done["handed"]
    files = _instance_files(launcher, tmp_path)
    try:
        assert done["exec"] and len(handed) == 1 and os.get_inheritable(handed[0])
        info = json.loads(files.info.read_text(encoding="utf-8"))
        assert info["format"] == "shepherd-instance/1"
        assert info["pid"] == os.getpid() and info["port"] == 8001
        assert info["url"] == "http://127.0.0.1:8001"
        assert info["repo_root"] == str(launcher.REPO_ROOT)
        assert info["settings_file"] == str(tmp_path / "settings" / "deployment.json")
        assert done["exec_env"]["SHEPHERD_INSTANCE_ID"] == info["instance_id"]
        assert done["exec_env"]["SHEPHERD_INSTANCE_FILE"] == str(files.info)
        assert (files.info.stat().st_mode & 0o777) == 0o600
        assert launcher.instances.claim(files.lock) is None, "the lock is not held"
    finally:
        for fd in handed:
            os.close(fd)
    assert launcher.instances.claim(files.lock) is not None


class _Health:
    """A loopback server whose /health answers with `instance_id`."""

    def __init__(self, instance_id):
        import http.server

        answer = json.dumps({"status": "healthy", "instance_id": instance_id}).encode()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(answer)

            def log_message(self, *args):
                pass

        import threading

        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _running_instance(launcher, tmp_path, instance_id, answer_id, settings=None):
    """Another start of the configuration: its lock held, its info file
    written, its /health answering with `answer_id`."""
    files = _instance_files(launcher, tmp_path, settings)
    fd = launcher.instances.claim(files.lock)
    health = _Health(answer_id)
    launcher.instances.write_info(files.info, instance_id=instance_id, pid=4242, port=1,
                                  url=health.url, repo_root=launcher.REPO_ROOT,
                                  settings_file=settings, log=None)
    return files, fd, health


@pytest.mark.parametrize("browser, found", [(True, True), (True, False), (False, None)],
                         ids=["opened", "no-browser-found", "--no-browser"])
def test_a_second_start_of_a_running_configuration_opens_it(monkeypatch, tmp_path, capsys,
                                                            browser, found):
    """Plan §4.5: a second double click started a second backend on the same
    data (or failed late on the port). It finds the instance whose lock is
    held and whose /health answers with its info file's id, opens it, and
    starts nothing. T08: it said "Opening it." whatever `webbrowser.open`
    answered, so a host with no browser was told a page opened that did not;
    it now says which happened, and gives the address when none opened.

    The second start is given SHEPHERD_DEVICE=cpu, which the running one was
    not: a start-up setting is not part of the configuration
    (`instance_key`), so it still starts nothing, and the running service
    keeps its device. setup_demo.py's CPU steps say so and send the operator
    to stop it first (`test_setup_demo.py::
    test_a_cpu_demo_says_to_stop_the_running_dashboard_before_its_cpu_start`)."""
    launcher = _launcher()
    files, fd, health = _running_instance(launcher, tmp_path, "abc123", "abc123")
    try:
        done = _run_launcher(monkeypatch, ["--no-auto-install"] + ([] if browser
                                                                   else ["--no-browser"]),
                             env=_state_env(tmp_path, SHEPHERD_DEVICE="cpu"),
                             launcher=launcher, browser=found)
    finally:
        health.close()
        os.close(fd)
    assert done["status"] == 0 and done["server"] is None and done["installs"] == []
    ui = f"{health.url}/ui"
    assert done["opened"] == ([ui] if browser else [])
    out = capsys.readouterr().out
    assert f"SHEPHERD-Advanced already runs for this configuration at {ui}" in out
    assert "Opening it." not in out
    opened = f"Opened {ui} in the browser."
    missing = f"No browser could be opened on this computer: open {ui} in a browser."
    assert (opened in out, missing in out) == (found is True, found is False)


def test_another_configuration_is_another_instance(monkeypatch, tmp_path):
    """An A/B slot with a settings file of its own has its own lock: a start
    of A never opens B."""
    launcher = _launcher()
    other = tmp_path / "slot-b" / "deployment.json"
    files, fd, health = _running_instance(launcher, tmp_path, "abc123", "abc123", settings=other)
    try:
        done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser"],
                             env=_state_env(tmp_path), launcher=launcher)
    finally:
        health.close()
        os.close(fd)
    assert done["exec"] and done["opened"] == []


def test_an_answer_without_the_instance_s_id_is_never_taken_for_it(monkeypatch, tmp_path,
                                                                    capsys):
    """/health 200 alone is not identity (another program may hold the port):
    while the lock is held and the answer names another id, the start waits;
    once the other start ends and the lock is free, it starts itself."""
    launcher = _launcher()
    files, fd, health = _running_instance(launcher, tmp_path, "abc123", "someone-else")
    released = []

    def sleep(seconds):
        if not released:
            os.close(fd)
            released.append(seconds)

    monkeypatch.setattr(launcher, "time", types.SimpleNamespace(
        monotonic=time.monotonic, sleep=sleep, strftime=time.strftime))
    try:
        done = _run_launcher(monkeypatch, ["--no-auto-install"], env=_state_env(tmp_path),
                             launcher=launcher)
    finally:
        health.close()
    assert released, "the start did not wait for the instance that holds the lock"
    assert done["exec"] and done["opened"] == []
    assert "Another start of this configuration is under way" in capsys.readouterr().out
    info = json.loads(files.info.read_text(encoding="utf-8"))
    assert info["instance_id"] != "abc123" and info["pid"] == os.getpid()


def test_a_stale_info_file_with_a_free_lock_is_replaced(monkeypatch, tmp_path, capsys):
    """A file left with the lock free is what a crash leaves (its server
    ended without the orderly stop that removes it, and so without a stop
    line): FL-SVC, the next start says so once, naming the earlier start,
    then replaces it. A start that finds no such file says nothing of it."""
    launcher = _launcher()
    files = _instance_files(launcher, tmp_path)
    left = launcher.instances.write_info(files.info, instance_id="crashed", pid=1, port=8000,
                                         url="http://127.0.0.1:9",
                                         repo_root=launcher.REPO_ROOT, settings_file=None,
                                         log=str(tmp_path / "launch.log"))
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser"],
                         env=_state_env(tmp_path), launcher=launcher)
    assert done["exec"]
    assert json.loads(files.info.read_text(encoding="utf-8"))["instance_id"] != "crashed"
    out = capsys.readouterr().out
    assert (f"WARNING: The previous start of this configuration (pid 1, started "
            f"{left['started_at']}) ended without its orderly stop: it was killed (SIGKILL or "
            "the out-of-memory killer), crashed, was forced to quit, or the machine stopped; "
            f"its last output is in {tmp_path / 'launch.log'}.") in out, out
    files.info.unlink()
    _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser"], env=_state_env(tmp_path),
                  launcher=_launcher())
    assert "ended without its orderly stop" not in capsys.readouterr().out


def test_a_refused_port_releases_the_lock_and_writes_no_info(monkeypatch, tmp_path):
    launcher = _launcher()
    done = _run_launcher(monkeypatch, ["--no-auto-install", "--no-browser", "--", "--port",
                                       "8264"], env=_state_env(tmp_path), busy_ports={8264},
                         launcher=launcher)
    files = _instance_files(launcher, tmp_path)
    assert done["status"] == 1 and not files.info.exists()
    fd = launcher.instances.claim(files.lock)
    assert fd is not None
    os.close(fd)


def test_without_a_terminal_the_output_goes_to_the_launch_log(monkeypatch, tmp_path):
    """UI-06e: a start from the desktop entry has no terminal, so its output
    and the server's go to the launch log in the state directory, and the
    browser opener is given the error page to show if the server ends before
    it answers. A start with a terminal keeps its output there."""
    launcher = _launcher()
    logged = []
    monkeypatch.setattr(launcher, "send_output_to_log", logged.append)
    done = _run_launcher(monkeypatch, ["--local-entry", "--no-auto-install"],
                         env=_state_env(tmp_path), terminal=False, launcher=launcher)
    state = tmp_path / "state"
    assert logged == [state / "logs" / "launch.log"]
    (url, health, options), = done["openers"]
    assert options["failure_page"] == state / "logs" / "launch-error.html"
    assert options["log_path"] == state / "logs" / "launch.log"
    assert options["instance_id"] == done["exec_env"]["SHEPHERD_INSTANCE_ID"]
    assert done["exec_env"]["PYTHONUNBUFFERED"] == "1"
    logged.clear()
    done = _run_launcher(monkeypatch, ["--local-entry", "--no-auto-install"],
                         env=_state_env(tmp_path, SHEPHERD_STATE_DIR=str(tmp_path / "other")),
                         terminal=True, launcher=launcher)
    (url, health, options), = done["openers"]
    assert logged == [] and options["failure_page"] is None and options["log_path"] is None


def test_without_a_terminal_output_already_redirected_stays_where_it_was(monkeypatch, tmp_path):
    """``nohup launch_shepherd.sh > my.log`` (or a script that reads the
    output) keeps its output there: only output nobody keeps (/dev/null, the
    desktop session's journal) goes to the launch log. A failure still opens
    the error page, which then names no log of its own."""
    launcher = _launcher()
    logged = []
    monkeypatch.setattr(launcher, "send_output_to_log", logged.append)
    done = _run_launcher(monkeypatch, ["--local-entry", "--no-auto-install"],
                         env=_state_env(tmp_path), terminal=False, output_kept=True,
                         launcher=launcher)
    assert logged == [] and done["exec"]
    (url, health, options), = done["openers"]
    assert options["failure_page"] == tmp_path / "state" / "logs" / "launch-error.html"
    assert options["log_path"] is None


def test_output_is_kept_tells_a_file_or_pipe_from_dev_null(tmp_path):
    program = ("import importlib.util, sys\n"
               "spec = importlib.util.spec_from_file_location('shep_launch', sys.argv[1])\n"
               "launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)\n"
               "sys.stderr.write(str(launcher.output_is_kept()))\n")
    said = {}
    for name, stdout in (("file", open(tmp_path / "out.log", "w")), ("pipe", subprocess.PIPE),
                         ("devnull", subprocess.DEVNULL)):
        done = subprocess.run([sys.executable, "-c", program, str(LAUNCHER)], stdout=stdout,
                              stderr=subprocess.PIPE, text=True, timeout=60)
        said[name] = done.stderr.strip().splitlines()[-1]
    assert said == {"file": "True", "pipe": "True", "devnull": "False"}


def test_without_a_terminal_a_refused_start_opens_a_page_saying_why(monkeypatch, tmp_path,
                                                                   capsys):
    """A failure before the server is up (here a network bind without the
    risk accepted) opens a local page with what failed, what to do and where
    the log is, escaped; it says it in the log too."""
    launcher = _launcher()
    monkeypatch.setattr(launcher, "send_output_to_log", lambda path: None)
    done = _run_launcher(monkeypatch, ["--local-entry", "--no-auto-install", "--", "--host",
                                       "0.0.0.0"], env=_state_env(tmp_path), terminal=False,
                         launcher=launcher)
    page = tmp_path / "state" / "logs" / "launch-error.html"
    assert done["status"] == 2 and done["server"] is None
    assert done["opened"] == [page.as_uri()]
    assert f"Opened {page.as_uri()} in the browser." in capsys.readouterr().out
    text = page.read_text(encoding="utf-8")
    assert "refusing to bind all interfaces" in text and "section 4.13" in text
    assert str(tmp_path / "state" / "logs" / "launch.log") in text
    assert (page.stat().st_mode & 0o777) == 0o600


def test_the_launch_log_receives_the_output_and_is_bounded(tmp_path):
    """The redirect itself, in a child process (it replaces fds 1 and 2):
    the launcher's lines and a later process's land in the log, and a log
    past its bound is moved aside once."""
    log = tmp_path / "logs" / "launch.log"
    log.parent.mkdir()
    log.write_bytes(b"x" * 64)
    program = (
        "import importlib.util, os, sys\n"
        "spec = importlib.util.spec_from_file_location('shep_launch', sys.argv[1])\n"
        "launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)\n"
        "launcher.instances.LOG_ROTATE_BYTES = 32\n"
        "launcher.instances.rotate_log.__defaults__ = (32,)\n"
        "launcher.send_output_to_log(__import__('pathlib').Path(sys.argv[2]))\n"
        "launcher.log('from the launcher')\n"
        "os.execv(sys.executable, [sys.executable, '-c', "
        "'import sys; print(\"from the server\"); print(\"to stderr\", file=sys.stderr)'])\n")
    done = subprocess.run([sys.executable, "-c", program, str(LAUNCHER), str(log)],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and done.stdout == "" and done.stderr == ""
    text = log.read_text(encoding="utf-8")
    assert "from the launcher" in text and "from the server" in text and "to stderr" in text
    assert (log.with_name("launch.log.1")).read_bytes() == b"x" * 64
    assert (log.stat().st_mode & 0o777) == 0o600


def _desktop_values(text):
    """A desktop entry's keys as the specification reads them: the string
    escapes, then (for Exec) the quoting of each argument."""
    values = {}
    for line in text.splitlines()[1:]:
        key, _, value = line.partition("=")
        values[key] = value
    return values


def _exec_argv(value):
    """An Exec value's arguments by the desktop entry specification: the
    string escapes (only \\s, \\n, \\t, \\r and \\\\), then each argument,
    unquoted or in double quotes where a backslash escapes ", `, $ and
    itself; %% is a literal %."""
    string = re.sub(r"\\([sntr\\])", lambda m: {"s": " ", "n": "\n", "t": "\t", "r": "\r",
                                                 "\\": "\\"}[m.group(1)], value)
    argv, current, quoted, i = [], None, False, 0
    while i < len(string):
        ch = string[i]
        if quoted:
            if ch == "\\" and string[i + 1] in '"`$\\':
                current += string[i + 1]
                i += 1
            elif ch == '"':
                quoted = False
            else:
                current += ch
        elif ch == " ":
            if current is not None:
                argv.append(current)
            current = None
        elif ch == '"':
            current, quoted = (current or ""), True
        else:
            current = (current or "") + ch
        i += 1
    if current is not None:
        argv.append(current)
    return [arg.replace("%%", "%") for arg in argv]


@pytest.mark.parametrize("settings", [None, "/srv/slot b/$HOME's \"100%\" `x`\\y.json"],
                         ids=["default", "awkward-settings-path"])
def test_the_desktop_entry_starts_this_checkout_without_a_terminal(monkeypatch, tmp_path,
                                                                   settings):
    """UI-06e: ``launch_shepherd.sh --install-desktop-entry`` writes an entry
    whose Exec is this checkout's absolute launch_shepherd.sh (by bash), with
    ``--settings-file`` for an A/B slot, quoted and escaped as the desktop
    entry specification asks, no terminal, this repository's icon."""
    launcher = _launcher()
    home = tmp_path / "home"
    args = ["--install-desktop-entry"] + (["--settings-file", settings] if settings else [])
    done = _run_launcher(monkeypatch, args, env={"HOME": str(home)}, launcher=launcher)
    assert done["status"] == 0 and done["server"] is None
    entries = list((home / ".local" / "share" / "applications").iterdir())
    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == ("shepherd-advanced.desktop" if settings is None
                          else f"shepherd-advanced-{launcher.instances.instance_key(launcher.REPO_ROOT, settings)[:8]}.desktop")
    text = entry.read_text(encoding="utf-8")
    assert text.startswith("[Desktop Entry]\n")
    values = _desktop_values(text)
    assert values["Type"] == "Application" and values["Terminal"] == "false"
    expected = ["bash", str(launcher.REPO_ROOT / "launch_shepherd.sh")]
    if settings:
        expected += ["--settings-file", settings]
    assert _exec_argv(values["Exec"]) == expected
    assert Path(values["Icon"]).is_file() and Path(values["Path"]) == launcher.REPO_ROOT
    assert (entry.stat().st_mode & 0o777) == 0o644


def test_the_desktop_entry_follows_xdg_data_home(monkeypatch, tmp_path):
    data = tmp_path / "data"
    done = _run_launcher(monkeypatch, ["--install-desktop-entry"],
                         env={"HOME": str(tmp_path / "home"), "XDG_DATA_HOME": str(data)})
    assert done["status"] == 0
    assert (data / "applications" / "shepherd-advanced.desktop").is_file()
    assert not (tmp_path / "home").exists()


def test_the_desktop_entry_is_refused_without_a_home(monkeypatch, capsys):
    done = _run_launcher(monkeypatch, ["--install-desktop-entry"], env={})
    assert done["status"] == 2
    assert "HOME is not set" in capsys.readouterr().out
