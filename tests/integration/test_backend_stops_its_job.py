"""
Stopping the backend stops its jobs (the owner's decision of 2026-10-22)
=======================================================================
Each GPU job now leads its own process group (review of 8242b47, P2-3), so a
Ctrl-C, a Ctrl-\\ or a hangup of the terminal the backend runs in, or a SIGTERM
sent to the backend alone, no longer reaches the job. The backend stops the
job its runner holds (`end_active_job`) from the signal on (`src.api.main`'s
`_ShutdownSignals`), and a hangup and a Ctrl-\\ (SIGQUIT), which uvicorn does
not handle, are routed to its orderly shutdown (root-cause review, P23-2).
From the signal on the GPU slot is closed, so a start still under way is
refused (P23-1).

End to end: the real backend (`python -m uvicorn src.api.main:app`, as
setup_demo prints it; the launcher, in its browser mode and as the systemd unit
starts it; and `launch_shepherd.sh`) in a session of its own, a real training
job started through the API (`scripts/train_model.py`, held by its epochs),
then the terminal's signal sent to the backend's process group, which the job
is not in, or a SIGTERM sent to the PID that was started alone, as a service
manager or `kill` sends it. The backend must end, and with it every process of
the job and the GPU lock. **Every POSIX start path leaves the server as the
process that was started** (round 3 of the review fixes, P3-2): the script
execs the launcher, which execs into uvicorn, its browser opener a helper that
is not the server's child.

**With a dashboard tab open** (verification of the owner's decision): the tab
keeps Gradio's session heartbeat stream open, and uvicorn waits for every open
connection before the lifespan's end, so a stop begun there left the job
training until the tab was closed. The tab's stream is held open here, as the
browser holds it; the backend must end all the same. **With a connection that
does not close** (a client that never finishes its request), the job must stop
while the backend still waits for it, and a second Ctrl-C, uvicorn's forced
quit, must leave no job behind. Every process a test starts is stopped by its
pid in `finally`.

Module: tests/integration/test_backend_stops_its_job.py
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("uvicorn")

from src.utils.gpu_lock import LOCK_FILE_NAME, WORKSPACES_ROOT_ENV, probe  # noqa: E402
from tests.fixtures.job_processes import group_members, processes  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="process groups and /proc: Linux")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _request(port: int, method: str, path: str, body=None, timeout: float = 30.0):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as answer:
        return answer.status, json.loads(answer.read() or b"null")


def _open_dashboard_tab(port: int) -> socket.socket:
    """A dashboard tab's heartbeat stream (`/ui/gradio_api/heartbeat/<session>`),
    held open as the browser holds it, once its first beat has come."""
    tab = socket.create_connection(("127.0.0.1", port), timeout=30)
    tab.sendall(f"GET /ui/gradio_api/heartbeat/tab0session HTTP/1.1\r\nHost: 127.0.0.1:{port}"
                "\r\nAccept: text/event-stream\r\n\r\n".encode())
    received = b""
    while b"data: ALIVE" not in received:
        chunk = tab.recv(4096)
        assert chunk, f"the heartbeat stream closed at once: {received!r}"
        received += chunk
    assert received.startswith(b"HTTP/1.1 200"), received[:200]
    return tab


def _unfinished_request(port: int) -> socket.socket:
    """A connection the backend must wait for: a request whose body never
    finishes arriving (it never parses, so nothing is started)."""
    client = socket.create_connection(("127.0.0.1", port), timeout=30)
    client.sendall(f"POST /api/v1/training/start HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                   "Content-Type: application/json\r\nContent-Length: 1000\r\n\r\n{".encode())
    return client


def _wait(condition, seconds: float, what: str):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            value = condition()
        except OSError:
            value = None
        if value:
            return value
        time.sleep(0.3)
    raise AssertionError(f"timed out waiting for {what}")


def _children(pid: int) -> list:
    """`(pid, state, cmdline)` of each process whose parent is `pid`."""
    found = []
    for child, fields in processes():
        if int(fields[1]) != pid:
            continue
        try:
            command = Path(f"/proc/{child}/cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, ProcessLookupError):  # gone since: no longer a child
            continue
        found.append((child, fields[0], command.decode(errors="replace")))
    return found


def _no_core_file() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class _Backend:
    """The real backend, started as `started_by` starts it, in a session of its
    own, with a real training job of its runner holding the GPU lock."""

    def __init__(self, tmp_path: Path, started_by: str):
        self.tmp_path, self.started_by = tmp_path, started_by
        self.process = None
        self.job_pid = None
        #: The service's workspaces root, which names its GPU lock.
        self.root = tmp_path / "workspaces"
        self.lock = self.root / LOCK_FILE_NAME
        self.port = _free_port()
        self._log = None

    def start(self, *, with_job: bool = True) -> "_Backend":
        from tests.fixtures.base_workspace import write_base_workspace

        self.data = write_base_workspace(self.tmp_path / "case")
        self.root.mkdir(exist_ok=True)
        env = {k: v for k, v in os.environ.items()
               if k not in ("SHEPHERD_KG_PATH", "SHEPHERD_CHECKPOINT_PATH", "SHEPHERD_BIND_HOST")}
        env.update({WORKSPACES_ROOT_ENV: str(self.root), "SHEPHERD_DEPLOYMENT_MODE": "M1",
                    "GRADIO_ANALYTICS_ENABLED": "False", "BROWSER": "/bin/true",
                    "PYTHONUNBUFFERED": "1"})
        launcher = [sys.executable, str(REPO / "scripts" / "launch" / "shep_launch.py"),
                    "--no-auto-install"]
        if self.started_by == "uvicorn":
            argv = [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1",
                    "--port", str(self.port), "--no-access-log"]
        elif self.started_by == "service":     # the systemd unit's ExecStart flags
            argv = [*launcher, "--no-browser", "--", "--host", "127.0.0.1",
                    "--port", str(self.port)]
        elif self.started_by == "script":      # launch_shepherd.sh, in its browser mode
            argv = ["bash", str(REPO / "launch_shepherd.sh"), "--no-auto-install", "--",
                    "--port", str(self.port)]
        else:
            argv = [*launcher, "--", "--port", str(self.port)]
        self._log = (self.tmp_path / "backend.log").open("w")
        # No core file: a backend that ends by SIGQUIT (Ctrl-\) would write one
        # into the repository, its working directory.
        self.process = subprocess.Popen(argv, cwd=str(REPO), env=env, stdout=self._log,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        preexec_fn=_no_core_file)
        pid = self.process.pid
        assert pid > 1 and os.getpgid(pid) == pid != os.getpgrp()
        port = self.port
        _wait(lambda: _request(port, "GET", "/health", timeout=5)[0] == 200, 240,
              "the backend to answer")
        if not with_job:
            return self
        status, started = _request(port, "POST", "/api/v1/training/start",
                                   body=self.training_request())
        assert status == 200 and started["success"], started
        self.job_pid = job_pid = started["pid"]
        assert os.getpgid(job_pid) == job_pid != pid, "the job leads its own group"
        _wait(lambda: f"pid {job_pid}" in (probe(self.lock) or ""), 180,
              "the job to take the lock")
        return self

    def training_request(self) -> dict:
        """A base training job held by its epochs (the API's start is a base
        start, K7; the recipe has no early stopping), on the test's
        workspace."""
        return dict(data_dir=str(self.data), learning_stage="base", run_name="held",
                    device="cpu", num_epochs=10000, batch_size=4, num_neighbors=[2, 2, 2],
                    use_amp=False)

    def send(self, number: int) -> None:
        """The terminal's signal: its foreground group only, which the job is
        not in. A SIGTERM to a backend started through the launcher or the
        script, as systemd's KillMode=mixed stop or `kill <pid>` sends it: the
        PID that was started only, which must be the server itself (the
        script and the launcher exec'd into it), never the job's processes,
        with no child of its own left from the start (the browser opener is
        not its child, nor a zombie of it)."""
        if self.started_by != "uvicorn" and number == signal.SIGTERM:
            pid = self.process.pid
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            assert b"uvicorn" in command, f"the main process is not the server: {command}"
            left = [child for child in _children(pid)
                    if child[0] != self.job_pid or child[1] in ("Z", "X")]
            assert left == [], f"the server has children of the start: {left}"
            os.kill(pid, number)
            return
        os.killpg(self.process.pid, number)

    def said(self) -> str:
        return (self.tmp_path / "backend.log").read_text()

    def job_ended(self) -> bool:
        return group_members(self.job_pid) == [] and probe(self.lock) is None

    def assert_the_job_ended_with_it(self) -> None:
        said = self.said()
        assert "The backend is stopping, so it stops the training job" in said, said[-3000:]
        assert group_members(self.job_pid) == [], "a process of the job outlived the backend"
        assert probe(self.lock) is None, "the GPU lock outlived the backend"

    def stop(self) -> None:
        """Whatever a failed test left: the backend and the job, by their pids."""
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=30)
        for pid in group_members(self.job_pid) if self.job_pid else []:
            os.kill(pid, signal.SIGKILL)
        if self._log is not None:
            self._log.close()


@pytest.mark.parametrize("started_by, sig, tab_open", [
    pytest.param("uvicorn", signal.SIGINT, False, id="uvicorn-ctrl-c"),
    pytest.param("uvicorn", signal.SIGHUP, False, id="uvicorn-terminal-closed"),
    pytest.param("uvicorn", signal.SIGTERM, False, id="uvicorn-sigterm"),
    # Ctrl-\ (root-cause review, P23-2): it used to orphan the job
    pytest.param("uvicorn", signal.SIGQUIT, False, id="uvicorn-ctrl-backslash"),
    # Ctrl-C with the launcher (its browser mode) started from the terminal
    pytest.param("launcher", signal.SIGINT, False, id="launcher-ctrl-c"),
    # SIGTERM to the PID that was started alone (round 3 of the review fixes,
    # P3-2): the launcher in its browser mode, and launch_shepherd.sh, which
    # used to end alone while the server served on with its job
    pytest.param("launcher", signal.SIGTERM, False, id="launcher-sigterm"),
    pytest.param("script", signal.SIGTERM, False, id="script-sigterm"),
    # The systemd unit's start and stop (KillMode=mixed: SIGTERM to the main
    # process only; root-cause review, P23-3)
    pytest.param("service", signal.SIGTERM, False, id="service-stop"),
    pytest.param("service", signal.SIGTERM, True, id="service-stop-tab-open"),
    # A dashboard tab open: its heartbeat stream held open, as the browser holds it
    pytest.param("uvicorn", signal.SIGINT, True, id="uvicorn-ctrl-c-tab-open"),
    pytest.param("uvicorn", signal.SIGHUP, True, id="uvicorn-terminal-closed-tab-open"),
    pytest.param("uvicorn", signal.SIGTERM, True, id="uvicorn-sigterm-tab-open"),
    pytest.param("uvicorn", signal.SIGQUIT, True, id="uvicorn-ctrl-backslash-tab-open"),
    pytest.param("launcher", signal.SIGINT, True, id="launcher-ctrl-c-tab-open"),
])
def test_the_backend_stopped_from_its_terminal_stops_the_job_it_runs(tmp_path, started_by, sig,
                                                                     tab_open):
    if started_by == "script" and not (REPO / ".venv" / "bin" / "activate").is_file():
        # The script's exec itself is checked without it, by
        # tests/unit/test_service_unit.py (a stub environment and launcher).
        pytest.skip("launch_shepherd.sh sources .venv/bin/activate in the repository")
    backend = _Backend(tmp_path, started_by)
    tab = None
    try:
        backend.start()
        if tab_open:
            tab = _open_dashboard_tab(backend.port)
        backend.send(sig)
        backend.process.wait(timeout=120)
        backend.assert_the_job_ended_with_it()
        if sig == signal.SIGQUIT:       # its meaning kept: the backend still ends by it
            assert backend.process.returncode == -signal.SIGQUIT
    finally:
        if tab is not None:
            tab.close()
        backend.stop()


@pytest.mark.parametrize("second_ctrl_c", [
    pytest.param(False, id="sigterm"),
    pytest.param(True, id="ctrl-c-twice"),
])
def test_the_job_stops_at_the_signal_while_a_connection_holds_the_backend(tmp_path,
                                                                         second_ctrl_c):
    """A connection that does not close holds uvicorn's shutdown before the
    lifespan's end (a dashboard tab's stream is ended at the signal; this one
    cannot be). SIGTERM: the job stops all the same, while the backend still
    waits, and the backend ends once the connection closes. A second Ctrl-C
    (uvicorn's forced quit, which skips the lifespan's end): the backend ends
    without waiting for the connection, and no process of the job outlives it."""
    backend = _Backend(tmp_path, "uvicorn")
    client = None
    try:
        backend.start()
        client = _unfinished_request(backend.port)
        time.sleep(0.5)                 # its headers read: the request is under way
        if second_ctrl_c:
            backend.send(signal.SIGINT)
            time.sleep(0.3)
            backend.send(signal.SIGINT)
            backend.process.wait(timeout=120)
        else:
            backend.send(signal.SIGTERM)
            _wait(backend.job_ended, 120, "the job to end while the backend waits")
            assert backend.process.poll() is None, "the backend did not wait for the connection"
            assert "Waiting for connections to close" in backend.said()
            client.close()
            client = None
            backend.process.wait(timeout=120)
        backend.assert_the_job_ended_with_it()
    finally:
        if client is not None:
            client.close()
        backend.stop()


def _training_start_in_flight(port: int, body: dict) -> tuple:
    """A training start whose headers and first byte of body have arrived (the
    request is under way) and whose rest is sent later: the connection and the
    rest of its body."""
    payload = json.dumps(body).encode()
    client = socket.create_connection(("127.0.0.1", port), timeout=60)
    client.sendall(f"POST /api/v1/training/start HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                   f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
                   "\r\n".encode() + payload[:1])
    return client, payload[1:]


def _response(client: socket.socket) -> tuple:
    """The status and body of the one response on `client` (the server closes
    a connection once it is shutting down)."""
    received = b""
    while True:
        chunk = client.recv(65536)
        if not chunk:
            break
        received += chunk
    head, _, body = received.partition(b"\r\n\r\n")
    return int(head.split()[1]), body


def test_a_start_under_way_when_the_backend_begins_to_stop_is_refused(tmp_path):
    """Root-cause review, P23-1: once the backend begins to stop, no job
    starts. A training start under way at the SIGTERM (its request partly
    received), completed while uvicorn waits for another connection that does
    not close, used to be answered 200 and its job trained, holding the GPU
    lock, until that connection closed; a second SIGTERM did not help. It is
    refused (409, "The backend is stopping; nothing was started."), no process
    is started and the lock stays free; the backend ends once the other
    connection closes."""
    backend = _Backend(tmp_path, "uvicorn")
    hanging = in_flight = None
    try:
        backend.start(with_job=False)
        hanging = _unfinished_request(backend.port)
        in_flight, rest = _training_start_in_flight(backend.port, backend.training_request())
        time.sleep(0.5)                 # both requests' headers read: under way
        backend.send(signal.SIGTERM)
        _wait(lambda: "Waiting for connections to close" in backend.said(), 60,
              "the backend to wait for its connections")
        in_flight.sendall(rest)
        status, body = _response(in_flight)
        backend.job_pid = (json.loads(body) or {}).get("pid")   # to be stopped if started
        assert status == 409, body
        assert json.loads(body)["error"] == "The backend is stopping; nothing was started."
        said = backend.said()
        assert "Starting training job" not in said, said[-3000:]
        assert probe(backend.lock) is None, "a job took the GPU lock after the signal"
        assert backend.process.poll() is None, "the backend did not wait for the connection"
        hanging.close()
        hanging = None
        backend.process.wait(timeout=120)
        assert probe(backend.lock) is None
    finally:
        for client in (hanging, in_flight):
            if client is not None:
                client.close()
        backend.stop()
