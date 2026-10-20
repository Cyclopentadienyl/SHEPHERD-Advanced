"""
Processes for the job runner's tests (M5)
=========================================
`FakeProcess` stands in for `subprocess.Popen` where a test drives the runner's
state without running a program: it runs until it is signalled or finished, so
a job stays "running" for as long as the test needs, and a stop's escalation
can be observed signal by signal.

`LOCK_HOLDER` is a real program for the GPU lock's tests: it takes the lock
through the same function the CLIs use (`take_gpu_lock`), says so on stdout,
and then does what its first argument asks (sleep, fork a child that keeps the
descriptor, or start a program that must not), so a test can kill it, let it
exit, or watch what its children keep.

Module: tests/fixtures/job_processes.py
"""
from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class FakeProcess:
    """A `Popen` with no output that runs until signalled or `finish`ed.

    SIGINT ends it with `on_sigint` unless `ignore_sigint`; `terminate` ends it
    with -15 unless `ignore_terminate`; `kill` always ends it with -9.
    """

    def __init__(self, pid: int = 12345, *, on_sigint: int = -2, ignore_sigint: bool = False,
                 ignore_terminate: bool = False, output: str = "") -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.stdout = io.StringIO(output)
        self.signals: list[str] = []
        self.on_sigint = on_sigint
        self.ignore_sigint = ignore_sigint
        self.ignore_terminate = ignore_terminate
        self._ended = threading.Event()

    def finish(self, code: int = 0) -> None:
        if not self._ended.is_set():
            self.returncode = code
            self._ended.set()

    def wait(self, timeout: float | None = None) -> int:
        if not self._ended.wait(timeout):
            raise subprocess.TimeoutExpired("fake job", timeout)
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode if self._ended.is_set() else None

    def send_signal(self, sig: int) -> None:
        self.signals.append(signal.Signals(sig).name)
        if not self.ignore_sigint:
            self.finish(self.on_sigint)

    def terminate(self) -> None:
        self.signals.append("SIGTERM")
        if not self.ignore_terminate:
            self.finish(-15)

    def kill(self) -> None:
        self.signals.append("SIGKILL")
        self.finish(-9)


#: A program that holds the GPU lock as the CLIs do. argv[1] is what it does
#: once it holds it: "sleep" (until killed), "exit" (return at once), "fork"
#: (fork a child that keeps the inherited descriptor and sleeps, then exit),
#: "spawn" (start a sleeping program with close_fds=False, print its pid, then
#: exit). It prints "HELD" once the lock is held, and a child's pid as
#: "CHILD <pid>".
LOCK_HOLDER = r'''
import os, subprocess, sys, time
sys.path.insert(0, sys.argv[2])
from src.utils.gpu_lock import take_gpu_lock

action = sys.argv[1]
with take_gpu_lock("training", "lock_holder.py"):
    print("HELD", flush=True)
    if action == "sleep":
        time.sleep(600)
    elif action == "fork":
        pid = os.fork()
        if pid == 0:
            time.sleep(600)
            os._exit(0)
        print(f"CHILD {pid}", flush=True)
        os._exit(0)
    elif action == "spawn":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"],
                                 close_fds=False)
        print(f"CHILD {child.pid}", flush=True)
        os._exit(0)
'''


def start_lock_holder(action: str, env: dict, tmp: Path) -> subprocess.Popen:
    """Start `LOCK_HOLDER` with `action` and wait until it holds the lock."""
    script = tmp / "lock_holder.py"
    script.write_text(LOCK_HOLDER)
    process = subprocess.Popen([sys.executable, str(script), action, str(REPO)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               env={**os.environ, **env})
    first = process.stdout.readline().strip()
    if first != "HELD":
        process.kill()
        raise AssertionError(f"the lock holder did not take the lock: {first!r} "
                             f"{process.stderr.read()}")
    return process
