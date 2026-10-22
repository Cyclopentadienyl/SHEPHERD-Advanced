"""
Stopping the backend stops its jobs (the owner's decision of 2026-10-22)
=======================================================================
Each GPU job now leads its own process group (review of 8242b47, P2-3), so a
Ctrl-C or a hangup of the terminal the backend runs in, or a SIGTERM sent to
the backend alone, no longer reaches the job. The backend stops the job its
runner holds (`end_active_job`) from the signal on (`src.api.main`'s
`_ShutdownSignals`), and a hangup, which uvicorn does not handle, is routed to
its orderly shutdown.

End to end: the real backend (`python -m uvicorn src.api.main:app`, as
setup_demo prints it, and the launcher in its browser mode, which waits beside
the server) in a session of its own, a real training job started through the
API (`scripts/train_model.py`, held by its epochs), then the terminal's signal
sent to the backend's process group, which the job is not in. The backend must
end, and with it every process of the job and the GPU lock.

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

from src.utils.gpu_lock import LOCK_FILE_ENV, WORKSPACES_ROOT_ENV, probe  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="process groups and /proc: Linux")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _members(group: int) -> list:
    """The processes of `group` that are not zombies (a zombie holds nothing)."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if int(fields[2]) == group and fields[0] not in ("Z", "X"):
            found.append(int(entry.name))
    return found


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


class _Backend:
    """The real backend, started as `started_by` starts it, in a session of its
    own, with a real training job of its runner holding the GPU lock."""

    def __init__(self, tmp_path: Path, started_by: str):
        self.tmp_path, self.started_by = tmp_path, started_by
        self.process = None
        self.job_pid = None
        self.lock = tmp_path / "gpu.lock"
        self.port = _free_port()
        self._log = None

    def start(self) -> "_Backend":
        from tests.fixtures.trainable_workspace import write_trainable_workspace

        data = write_trainable_workspace(self.tmp_path / "case")
        env = {k: v for k, v in os.environ.items()
               if k not in (WORKSPACES_ROOT_ENV, "SHEPHERD_KG_PATH", "SHEPHERD_CHECKPOINT_PATH",
                            "SHEPHERD_BIND_HOST")}
        env.update({LOCK_FILE_ENV: str(self.lock), "SHEPHERD_DEPLOYMENT_MODE": "M1",
                    "GRADIO_ANALYTICS_ENABLED": "False", "BROWSER": "/bin/true",
                    "PYTHONUNBUFFERED": "1"})
        if self.started_by == "uvicorn":
            argv = [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1",
                    "--port", str(self.port), "--no-access-log"]
        else:
            argv = [sys.executable, str(REPO / "scripts" / "launch" / "shep_launch.py"),
                    "--no-auto-install", "--", "--port", str(self.port)]
        self._log = (self.tmp_path / "backend.log").open("w")
        self.process = subprocess.Popen(argv, cwd=str(REPO), env=env, stdout=self._log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        pid = self.process.pid
        assert pid > 1 and os.getpgid(pid) == pid != os.getpgrp()
        port = self.port
        _wait(lambda: _request(port, "GET", "/health", timeout=5)[0] == 200, 240,
              "the backend to answer")
        status, started = _request(port, "POST", "/api/v1/training/start", body=dict(
            data_dir=str(data), run_name="held", device="cpu", num_epochs=10000,
            early_stopping_patience=10000, hidden_dim=32, num_heads=2, num_layers=1,
            conv_type="gat", batch_size=2, num_workers=0, use_amp=False))
        assert status == 200 and started["success"], started
        self.job_pid = job_pid = started["pid"]
        assert os.getpgid(job_pid) == job_pid != pid, "the job leads its own group"
        _wait(lambda: f"pid {job_pid}" in (probe(self.lock) or ""), 180,
              "the job to take the lock")
        return self

    def send(self, number: int) -> None:
        """The terminal's signal: its foreground group only, which the job is not in."""
        os.killpg(self.process.pid, number)

    def said(self) -> str:
        return (self.tmp_path / "backend.log").read_text()

    def job_ended(self) -> bool:
        return _members(self.job_pid) == [] and probe(self.lock) is None

    def assert_the_job_ended_with_it(self) -> None:
        said = self.said()
        assert "The backend is stopping, so it stops the training job" in said, said[-3000:]
        assert _members(self.job_pid) == [], "a process of the job outlived the backend"
        assert probe(self.lock) is None, "the GPU lock outlived the backend"

    def stop(self) -> None:
        """Whatever a failed test left: the backend and the job, by their pids."""
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=30)
        for pid in _members(self.job_pid) if self.job_pid else []:
            os.kill(pid, signal.SIGKILL)
        if self._log is not None:
            self._log.close()


@pytest.mark.parametrize("started_by, sig, tab_open", [
    pytest.param("uvicorn", signal.SIGINT, False, id="uvicorn-ctrl-c"),
    pytest.param("uvicorn", signal.SIGHUP, False, id="uvicorn-terminal-closed"),
    pytest.param("uvicorn", signal.SIGTERM, False, id="uvicorn-sigterm"),
    # Ctrl-C with the launcher (its browser mode) waiting beside the server
    pytest.param("launcher", signal.SIGINT, False, id="launcher-ctrl-c"),
    # A dashboard tab open: its heartbeat stream held open, as the browser holds it
    pytest.param("uvicorn", signal.SIGINT, True, id="uvicorn-ctrl-c-tab-open"),
    pytest.param("uvicorn", signal.SIGHUP, True, id="uvicorn-terminal-closed-tab-open"),
    pytest.param("uvicorn", signal.SIGTERM, True, id="uvicorn-sigterm-tab-open"),
    pytest.param("launcher", signal.SIGINT, True, id="launcher-ctrl-c-tab-open"),
])
def test_the_backend_stopped_from_its_terminal_stops_the_job_it_runs(tmp_path, started_by, sig,
                                                                     tab_open):
    backend = _Backend(tmp_path, started_by)
    tab = None
    try:
        backend.start()
        if tab_open:
            tab = _open_dashboard_tab(backend.port)
        backend.send(sig)
        backend.process.wait(timeout=120)
        backend.assert_the_job_ended_with_it()
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
