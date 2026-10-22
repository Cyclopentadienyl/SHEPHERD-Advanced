"""
Stopping the backend stops its jobs (the owner's decision of 2026-10-22)
=======================================================================
Each GPU job now leads its own process group (review of 8242b47, P2-3), so a
Ctrl-C or a hangup of the terminal the backend runs in, or a SIGTERM sent to
the backend alone, no longer reaches the job. The backend's shutdown stops the
job its runner holds (`src.api.main`'s lifespan, `end_active_job`), and a
hangup, which uvicorn does not handle, is routed to its orderly shutdown.

End to end: the real backend (`python -m uvicorn src.api.main:app`, as
setup_demo prints it, and the launcher in its browser mode, which waits beside
the server) in a session of its own, a real training job started through the
API (`scripts/train_model.py`, held by its epochs), then the terminal's signal
sent to the backend's process group, which the job is not in. The backend must
end, and with it every process of the job and the GPU lock. Every process a
test starts is stopped by its pid in `finally`.

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


@pytest.mark.parametrize("started_by, sig", [
    pytest.param("uvicorn", signal.SIGINT, id="uvicorn-ctrl-c"),
    pytest.param("uvicorn", signal.SIGHUP, id="uvicorn-terminal-closed"),
    pytest.param("uvicorn", signal.SIGTERM, id="uvicorn-sigterm"),
    # Ctrl-C with the launcher (its browser mode) waiting beside the server
    pytest.param("launcher", signal.SIGINT, id="launcher-ctrl-c"),
])
def test_the_backend_stopped_from_its_terminal_stops_the_job_it_runs(tmp_path, started_by, sig):
    from tests.fixtures.trainable_workspace import write_trainable_workspace

    data = write_trainable_workspace(tmp_path / "case")
    lock = tmp_path / "gpu.lock"
    port = _free_port()
    env = {k: v for k, v in os.environ.items()
           if k not in (WORKSPACES_ROOT_ENV, "SHEPHERD_KG_PATH", "SHEPHERD_CHECKPOINT_PATH",
                        "SHEPHERD_BIND_HOST")}
    env.update({LOCK_FILE_ENV: str(lock), "SHEPHERD_DEPLOYMENT_MODE": "M1",
                "GRADIO_ANALYTICS_ENABLED": "False", "BROWSER": "/bin/true",
                "PYTHONUNBUFFERED": "1"})
    if started_by == "uvicorn":
        argv = [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1",
                "--port", str(port), "--no-access-log"]
    else:
        argv = [sys.executable, str(REPO / "scripts" / "launch" / "shep_launch.py"),
                "--no-auto-install", "--", "--port", str(port)]
    log = (tmp_path / "backend.log").open("w")
    backend = subprocess.Popen(argv, cwd=str(REPO), env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    job_pid = None
    try:
        assert backend.pid > 1 and os.getpgid(backend.pid) == backend.pid != os.getpgrp()
        _wait(lambda: _request(port, "GET", "/health", timeout=5)[0] == 200, 240,
              "the backend to answer")
        status, started = _request(port, "POST", "/api/v1/training/start", body=dict(
            data_dir=str(data), run_name="held", device="cpu", num_epochs=10000,
            early_stopping_patience=10000, hidden_dim=32, num_heads=2, num_layers=1,
            conv_type="gat", batch_size=2, num_workers=0, use_amp=False))
        assert status == 200 and started["success"], started
        job_pid = started["pid"]
        assert os.getpgid(job_pid) == job_pid != backend.pid, "the job leads its own group"
        _wait(lambda: f"pid {job_pid}" in (probe(lock) or ""), 180, "the job to take the lock")

        os.killpg(backend.pid, sig)      # the terminal's signal: its foreground group only
        backend.wait(timeout=180)

        said = (tmp_path / "backend.log").read_text()
        assert "The backend is stopping, so it stops the training job" in said, said[-3000:]
        assert _members(job_pid) == [], "a process of the job outlived the backend"
        assert probe(lock) is None, "the GPU lock outlived the backend"
    finally:
        if backend.poll() is None:
            os.killpg(backend.pid, signal.SIGKILL)
            backend.wait(timeout=30)
        for pid in _members(job_pid) if job_pid else []:
            os.kill(pid, signal.SIGKILL)
        log.close()
