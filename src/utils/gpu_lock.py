"""
One GPU lock per deployment (M5)
================================
Managed training and evaluation jobs exclude each other through one lock file
per deployment, whatever the workspace (`docs/working/PLAN_TEST_RESULTS.md` §6,
"M5's boundaries"). The jobs the WebUI or the API start through the job runner
(`src/api/services/training_manager.py`) and the formal CLIs
(`scripts/train_model.py`, `scripts/measure_scorer.py`) take the same lock. A
busy lock is refused, naming its holder; nothing is queued.

**Where it is.** `SHEPHERD_GPU_LOCK_FILE`, an absolute path whose directory
exists, when it is set; otherwise `.shepherd-gpu.lock` directly under
`SHEPHERD_WORKSPACES_ROOT`, the one workspace root, so two workspaces share one
lock. With neither set the lock is **not configured**: a CLI says so in its log
and runs without it, and the runner keeps only its own one-job rule. A setting
that is set but unusable (relative, or its directory missing) is refused, never
replaced by a default.

**What the lock is.** An exclusive, non-blocking `fcntl.flock` on the lock file
on POSIX; `msvcrt.locking` on the file's first byte on Windows (untested there,
like the rest of the backend; nothing here imports a POSIX-only module on
Windows). The holder keeps it for its whole working life: the kernel releases
it when the last descriptor of that open file closes, so a normal exit, an
exception, SIGTERM or SIGKILL all release it, and nothing has to be cleaned up
for the next job to start.

**The holder's description** (`<lock file>.json`: job type, command, pid, host,
start time) is written after the lock is taken and removed before it is
released, for the "held by" message. It is descriptive only: the lock is the
flock, never the file's content, and a description left behind by a killed
holder is replaced by the next one.

**The limits, stated rather than hidden:**
- **Cooperative.** A GPU process that does not take the lock is not stopped by
  it; the resident inference service does not take it and may hold GPU memory.
  There is no cross-process inference scheduling.
- **Inherited descriptors.** The descriptor is close-on-exec, so a program the
  holder starts (`subprocess`, any `exec`) does not keep the lock. A process
  forked without exec (DataLoader workers under the `fork` start method, say)
  shares the open file and keeps the lock until it exits too, even after the
  holder has gone; the message then says the recorded holder has exited.
- **Local machine.** The lock file belongs on a local filesystem; the liveness
  hint in the message is only given for a holder on this host.

Module: src/utils/gpu_lock.py
"""
from __future__ import annotations

import errno
import json
import logging
import os
import socket
import sys
import time
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from src.utils.file_modes import publish_bytes

if sys.platform == "win32":  # pragma: no cover - the backend is untested on Windows
    import msvcrt

    fcntl = None
else:
    import fcntl

    msvcrt = None

logger = logging.getLogger(__name__)

#: An explicit lock file (absolute; its directory must exist).
LOCK_FILE_ENV = "SHEPHERD_GPU_LOCK_FILE"
#: The one workspace root; the lock lives directly under it when no explicit
#: file is named.
WORKSPACES_ROOT_ENV = "SHEPHERD_WORKSPACES_ROOT"
LOCK_FILE_NAME = ".shepherd-gpu.lock"
HOLDER_SUFFIX = ".json"

#: How long a taker keeps trying before it refuses: long enough to step over
#: the runner's instant check, a holder that has just let go, or the forked
#: workers of a job that has just ended; short enough that nothing waits for a
#: running job (a busy lock is refused, never queued). The description read
#: then is the one the refusal names, whatever it says, since only the flock
#: decides.
GRACE_SECONDS = 1.0
_RETRY_SECONDS = 0.05


class GpuLockBusy(OSError):
    """Another process holds the GPU lock. An `OSError`, so a CLI's program
    entry prints it as a refusal before anything was written."""

    def __init__(self, path: Path, holder: str) -> None:
        super().__init__(
            f"the GPU lock {path} is held by {holder}. One managed GPU job runs at a "
            "time: wait for it to end, or stop it where it was started."
        )
        self.path = path
        self.holder = holder


class GpuLockMisconfigured(ValueError):
    """A lock setting is set but cannot name a usable lock file. Refused rather
    than replaced by a default, so two processes never lock different files."""


class LockLocation(NamedTuple):
    """Where the deployment's lock file is, and which setting named it; `path`
    is None when the lock is not configured, `source` then saying why."""

    path: Path | None
    source: str


# ---------------------------------------------------------------------------
# Where the lock is
# ---------------------------------------------------------------------------
def lock_location(environ: Mapping[str, str] | None = None) -> LockLocation:
    """The lock file the settings name, read from the environment at each call.

    Raises:
        GpuLockMisconfigured: a setting is set but is not an absolute path, or
            names a directory that does not exist.
    """
    # Integration note (M5-1): `SHEPHERD_WORKSPACES_ROOT` is WP1's setting; once
    # its settings module exists, these two reads go through it (one reader).
    env = os.environ if environ is None else environ
    explicit = (env.get(LOCK_FILE_ENV) or "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            raise GpuLockMisconfigured(
                f"{LOCK_FILE_ENV}={explicit!r} is not an absolute path. Every process that "
                "takes the GPU lock must name the same file, whatever its working directory.")
        if not path.parent.is_dir():
            raise GpuLockMisconfigured(
                f"{LOCK_FILE_ENV}={explicit!r}: the directory {path.parent} does not exist. "
                "Nothing creates it; create it, or point the setting elsewhere.")
        if path.is_dir():
            raise GpuLockMisconfigured(
                f"{LOCK_FILE_ENV}={explicit!r} is a directory; name a file inside it.")
        return LockLocation(path, LOCK_FILE_ENV)
    root = (env.get(WORKSPACES_ROOT_ENV) or "").strip()
    if root:
        root_path = Path(root)
        if not root_path.is_absolute():
            raise GpuLockMisconfigured(
                f"{WORKSPACES_ROOT_ENV}={root!r} is not an absolute path, so the GPU lock "
                "under it would depend on each process's working directory.")
        if not root_path.is_dir():
            raise GpuLockMisconfigured(
                f"{WORKSPACES_ROOT_ENV}={root!r} does not exist or is not a directory, so "
                "the GPU lock cannot be kept under it.")
        return LockLocation(root_path / LOCK_FILE_NAME, f"under {WORKSPACES_ROOT_ENV}")
    return LockLocation(
        None, f"not configured (neither {LOCK_FILE_ENV} nor {WORKSPACES_ROOT_ENV} is set)")


def holder_path(lock_path: Path) -> Path:
    """The holder's description, next to the lock file."""
    return lock_path.with_name(lock_path.name + HOLDER_SUFFIX)


# ---------------------------------------------------------------------------
# Who holds it (descriptive only)
# ---------------------------------------------------------------------------
def read_holder(lock_path: Path) -> dict[str, Any] | None:
    """The holder's own description, or None when there is none or it cannot be
    read. Never a reason to take or refuse the lock: only the flock decides."""
    try:
        data = json.loads(holder_path(lock_path).read_bytes())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _holder_alive(holder: Mapping[str, Any]) -> bool | None:
    """Whether the recorded pid still runs, where that can be told: on POSIX, for
    a holder on this host. None otherwise. On Windows `os.kill(pid, 0)` would
    terminate the process, so it is never called there."""
    if sys.platform == "win32" or holder.get("host") != socket.gethostname():
        return None
    pid = holder.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def describe_holder(holder: Mapping[str, Any] | None) -> str:
    """Who holds the lock, in words, from its description when there is one."""
    if holder is None:
        return ("a process that has not described itself (a holder that is just starting, "
                "or a process that inherited the lock's descriptor from one that has ended)")
    job = holder.get("job_type") or "unnamed"
    parts = [str(holder.get("command") or "unknown command")]
    if holder.get("pid") is not None:
        parts.append(f"pid {holder.get('pid')}"
                     + (f" on {holder.get('host')}" if holder.get("host") else ""))
    if holder.get("started_at"):
        parts.append(f"since {holder.get('started_at')}")
    article = "an" if str(job)[:1].lower() in "aeiou" else "a"
    text = f"{article} {job} job ({', '.join(parts)}, as it described itself)"
    if _holder_alive(holder) is False:
        text += ("; that process has exited, so the lock is held by a process that "
                 "inherited its descriptor")
    return text


# ---------------------------------------------------------------------------
# The lock itself
# ---------------------------------------------------------------------------
def _busy_errnos() -> tuple:
    return (errno.EACCES, getattr(errno, "EDEADLOCK", errno.EDEADLK), errno.EDEADLK)


def _try_lock(fd: int) -> bool:
    """Take the lock on `fd` without waiting: True when taken, False when another
    open file holds it. Any other failure is raised."""
    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True
    os.lseek(fd, 0, os.SEEK_SET)  # pragma: no cover - Windows
    try:  # pragma: no cover - Windows
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    except OSError as exc:  # pragma: no cover - Windows
        if exc.errno in _busy_errnos():
            return False
        raise
    return True  # pragma: no cover - Windows


def _unlock(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    else:  # pragma: no cover - Windows
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


def _open_lock_file(path: Path) -> int:
    """The lock file opened (created if absent, with the mode `open()` would
    give), its descriptor close-on-exec. Read-only when this account may not
    write it: a lock needs only an open file."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    except PermissionError:
        fd = os.open(path, os.O_RDONLY)
    os.set_inheritable(fd, False)
    return fd


class GpuLock(AbstractContextManager):
    """The deployment's GPU lock as one process holds it.

    `acquire` takes it or raises `GpuLockBusy`; `release` (or leaving the
    `with` block) gives it back. The process ending releases it too.
    """

    def __init__(self, path: Path, *, job_type: str, command: str) -> None:
        self.path = Path(path)
        self.job_type = job_type
        self.command = command
        self._fd: int | None = None
        self._pid: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self, grace: float = GRACE_SECONDS) -> GpuLock:
        """Take the lock, trying for `grace` seconds, or refuse naming the
        holder as its description has it."""
        if self._fd is not None:
            return self
        try:
            fd = _open_lock_file(self.path)
        except OSError as exc:
            raise OSError(f"the GPU lock {self.path} cannot be opened ({exc}).") from exc
        try:
            deadline = time.monotonic() + grace
            while not self._try(fd):
                if time.monotonic() >= deadline:
                    raise GpuLockBusy(self.path, describe_holder(read_holder(self.path)))
                time.sleep(_RETRY_SECONDS)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        self._pid = os.getpid()
        self._describe()
        return self

    def _try(self, fd: int) -> bool:
        try:
            return _try_lock(fd)
        except OSError as exc:
            raise OSError(f"the GPU lock {self.path} cannot be taken ({exc}).") from exc

    def _describe(self) -> None:
        record = {
            "job_type": self.job_type,
            "command": self.command,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "note": f"Descriptive only: the lock is the flock on {self.path.name}, "
                    "not this file's content.",
        }
        try:
            publish_bytes(holder_path(self.path),
                          (json.dumps(record, indent=2) + "\n").encode("utf-8"))
        except OSError as exc:
            logger.warning("GPU lock %s is held, but its description could not be written "
                           "(%s); a refused job will not see who holds it.", self.path, exc)

    def release(self) -> None:
        """Give the lock back: the description first (only this process's own),
        then the lock. Safe to call twice.

        **Only in the process that took it.** A process forked from the holder
        shares its open file, and an unlock there would release the holder's
        lock; it only closes its own descriptor."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        if self._pid != os.getpid():
            os.close(fd)
            return
        try:
            holder = read_holder(self.path)
            if holder is not None and holder.get("pid") == os.getpid():
                holder_path(self.path).unlink(missing_ok=True)
        except OSError:
            pass
        try:
            _unlock(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def __exit__(self, *exc_info: Any) -> None:
        self.release()


def probe(path: Path) -> str | None:
    """None when nobody holds the lock at this instant, else who holds it.

    Opens the file read-only and creates nothing; takes the lock for an instant
    and gives it back when it is free. A taker that meets that instant tries
    again (`GRACE_SECONDS`) instead of being refused.

    Raises:
        OSError: the lock file exists but cannot be opened or locked.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return None
    os.set_inheritable(fd, False)
    try:
        if _try_lock(fd):
            _unlock(fd)
            return None
        return describe_holder(read_holder(path))
    finally:
        os.close(fd)


def take_gpu_lock(job_type: str, command: str, *, log: logging.Logger | None = None,
                  environ: Mapping[str, str] | None = None) -> AbstractContextManager:
    """The lock for a CLI's run, taken now: a context manager that releases it.

    With the lock not configured, a no-op context manager, and the log says that
    this run does not exclude other managed GPU jobs.

    Raises:
        GpuLockBusy: another process holds it.
        GpuLockMisconfigured: a lock setting is set but unusable.
        OSError: the lock file cannot be opened or locked.
    """
    log = log or logger
    location = lock_location(environ)
    if location.path is None:
        log.warning("GPU lock %s, so this run does not exclude other managed GPU jobs.",
                    location.source)
        return nullcontext()
    lock = GpuLock(location.path, job_type=job_type, command=command).acquire()
    log.info("GPU lock %s held (%s) until this run ends.", location.path, location.source)
    return lock


def lock_state(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The lock as a status shows it: `configured`, `path`, `source` (the
    setting that names it, or why there is none), who holds it at this instant
    (`held_by`, None when free), and `problem` when it cannot be used or
    checked, in which case a job is refused rather than started unlocked."""
    state: dict[str, Any] = {"configured": False, "path": None, "source": None,
                             "held_by": None, "problem": None}
    try:
        location = lock_location(environ)
    except GpuLockMisconfigured as exc:
        return {**state, "problem": str(exc)}
    state.update(source=location.source)
    if location.path is None:
        return state
    state.update(configured=True, path=str(location.path))
    try:
        state["held_by"] = probe(location.path)
    except OSError as exc:
        state["problem"] = f"the GPU lock {location.path} cannot be checked ({exc})"
    return state


__all__ = [
    "GpuLock",
    "GpuLockBusy",
    "GpuLockMisconfigured",
    "HOLDER_SUFFIX",
    "LOCK_FILE_ENV",
    "LOCK_FILE_NAME",
    "LockLocation",
    "GRACE_SECONDS",
    "WORKSPACES_ROOT_ENV",
    "describe_holder",
    "holder_path",
    "lock_location",
    "lock_state",
    "probe",
    "read_holder",
    "take_gpu_lock",
]
