"""
Processes for the job runner's tests (M5)
=========================================
`FakeProcess` stands in for `subprocess.Popen` where a test drives the runner's
state without running a program: it runs until it is signalled or finished, so
a job stays "running" for as long as the test needs, and a stop's escalation
can be observed signal by signal.

`LOCK_HOLDER` is a real program for the GPU lock's tests: it takes the lock
through the same function the CLIs use (`take_gpu_lock`), says so on stdout,
and then does what its first argument asks (sleep, fork a child that shares
the descriptor and exit with or without releasing, or start a program that
must not keep it), so a test can kill it, let it exit, or watch what its
children keep.

`process_stat` is the tests' one reader of a process's state (Linux,
`/proc/<pid>/stat`), with what is read from it: `process_state`, `live`,
`processes` and `group_members`.

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
#: (fork a child that shares the descriptor and sleeps, then exit without
#: releasing, `os._exit`), "fork-release" (the same child, then leave the
#: `with` block normally, which releases, and exit), "spawn" (start a sleeping
#: program with close_fds=False, print its pid, then exit). It prints "HELD"
#: once the lock is held, and a child's pid as "CHILD <pid>".
LOCK_HOLDER = r'''
import os, subprocess, sys, time
sys.path.insert(0, sys.argv[2])
from src.utils.gpu_lock import take_gpu_lock

action = sys.argv[1]
with take_gpu_lock("training", "lock_holder.py"):
    print("HELD", flush=True)
    if action == "sleep":
        time.sleep(600)
    elif action in ("fork", "fork-release"):
        pid = os.fork()
        if pid == 0:
            time.sleep(600)
            os._exit(0)
        print(f"CHILD {pid}", flush=True)
        if action == "fork":
            os._exit(0)
    elif action == "spawn":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"],
                                 close_fds=False)
        print(f"CHILD {child.pid}", flush=True)
        os._exit(0)
'''


#: The states of a process that has ended and not yet been reaped (a zombie,
#: or one being torn down): it holds no descriptor, so not the GPU lock.
ENDED_STATES = ("Z", "X")


def process_stat(pid: int) -> list | None:
    """The fields of `/proc/<pid>/stat` after the command name (the state,
    the parent, the process group, ...), or None once the process is gone,
    either way the system says so: its entry absent (FileNotFoundError), or
    the process reaped while its entry is read (`read` fails with ESRCH,
    ProcessLookupError; the review of 2887506 caught a copy of this reader
    that knew only the first). Any other failure is raised."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            data = handle.read()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return data.rsplit(b")", 1)[1].decode("ascii").split()


def process_state(pid: int) -> str:
    """The state letter of process `pid` (`R`, `S`, `T`, `Z`, ...), or "gone"."""
    fields = process_stat(pid)
    return "gone" if fields is None else fields[0]


def live(pid: int) -> bool:
    """Whether process `pid` is there and has not ended (a zombie has)."""
    return process_state(pid) not in ("gone", *ENDED_STATES)


def processes():
    """`(pid, fields)` of each process there is, its fields `process_stat`'s."""
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            fields = process_stat(int(name))
        except PermissionError:  # not this account's process, so not one of its jobs'
            continue
        if fields is not None:
            yield int(name), fields


def group_members(group: int) -> list:
    """The processes of process group `group` (the fifth field of the stat)
    that have not ended."""
    return [pid for pid, fields in processes()
            if int(fields[2]) == group and fields[0] not in ENDED_STATES]


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
