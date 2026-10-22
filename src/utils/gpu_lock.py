"""
One GPU lock per deployment (M5)
================================
Managed training and evaluation jobs exclude each other through one lock file
per deployment, whatever the workspace (`docs/working/PLAN_TEST_RESULTS.md` §6,
"M5's boundaries"). The jobs the WebUI or the API start through the job runner
(`src/api/services/training_manager.py`) and the formal CLIs
(`scripts/train_model.py`, `scripts/measure_scorer.py`) take the same lock. A
busy lock is refused, naming its holder; nothing is queued.

**Where it is.** **The workspaces root is the lock's identity**: with
`SHEPHERD_WORKSPACES_ROOT` set, `.shepherd-gpu.lock` directly under it, so two
workspaces share one lock and every process that names the deployment's root
takes the same file; `SHEPHERD_GPU_LOCK_FILE` beside it is accepted only when
it names that file, and another file is refused naming both. With no root
set, `SHEPHERD_GPU_LOCK_FILE`, an absolute path whose directory exists. The
rule is the server settings' (`server_settings.gpu_lock_location`), the one
reader of both variables. **With neither set the lock is not configured and
every managed job is refused** (review of 8242b47, P2-4): a formal CLI exits
before it reads anything and the runner refuses a start, each saying how to
configure it; nothing runs without excluding the others. A setting that is set
but unusable (relative, or its directory missing) is refused too, never
replaced by the other one or by a default. What the environment cannot
settle: a process that names another root, or a lock file and no root, takes
another file.

**What the lock is.** An exclusive, non-blocking `fcntl.flock` on the lock file
on POSIX; `msvcrt.locking` on the file's first byte on Windows (untested there,
like the rest of the backend; nothing here imports a POSIX-only module on
Windows). The holder keeps it for its whole working life and gives it back
when it ends normally or by an exception (`release`); when it cannot (SIGTERM,
SIGKILL, `os._exit`), the kernel releases it once the last descriptor of that
open file closes. Nothing has to be cleaned up for the next job to start.

**The holder's description** (`<lock file>.json`: job type, command, pid, host,
start time) is written after the lock is taken and removed before it is
released, for the "held by" message. It is descriptive only: the lock is the
flock, never the file's content. A description left behind by a killed holder
is removed by the next one as soon as it holds the lock, before it describes
itself; a description whose process has exited is never named as the holder.

**The limits, stated rather than hidden:**
- **Cooperative.** A GPU process that does not take the lock is not stopped by
  it; the resident inference service does not take it and may hold GPU memory.
  There is no cross-process inference scheduling.
- **Inherited descriptors.** The descriptor is close-on-exec, so a program the
  holder starts (`subprocess`, any `exec`) does not keep the lock. A process
  forked without exec (DataLoader workers under the `fork` start method, say)
  shares the open file: a holder that ends normally releases the lock for it
  too, but a holder that is killed or exits without releasing (`os._exit`)
  leaves the lock with that child until it exits; the message then says the
  recorded holder has exited. A job the runner starts leads its own process
  group, which a stop's SIGTERM and SIGKILL reach whole and which the runner
  ends before it frees the slot (`training_manager`); a CLI killed by hand
  with `kill -9 <pid>` leaves its forked workers to end by themselves.
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
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from src.config import server_settings
from src.utils.file_modes import publish_bytes

if sys.platform == "win32":  # pragma: no cover - the backend is untested on Windows
    import msvcrt

    fcntl = None
else:
    import fcntl

    msvcrt = None

logger = logging.getLogger(__name__)

#: An explicit lock file (absolute; its directory must exist): the lock where no
#: workspace root is set, and beside one only when it names the root's own.
LOCK_FILE_ENV = server_settings.ENV_GPU_LOCK_FILE
#: The one workspace root; when it is set, the lock is directly under it.
WORKSPACES_ROOT_ENV = server_settings.ENV_WORKSPACES_ROOT
LOCK_FILE_NAME = server_settings.GPU_LOCK_FILE_NAME
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
    """No usable lock file: a setting is set but cannot name one, or neither is
    set (`GpuLockNotConfigured`). Refused rather than replaced by a default or
    run unlocked, so two processes never lock different files that way and no
    managed job runs without excluding the others."""


class GpuLockNotConfigured(GpuLockMisconfigured):
    """Neither lock setting is set (review of 8242b47, P2-4): every managed job
    is refused until one is."""


class LockLocation(NamedTuple):
    """Where the deployment's lock file is, and which setting named it."""

    path: Path
    source: str


# ---------------------------------------------------------------------------
# Where the lock is
# ---------------------------------------------------------------------------
def lock_location(environ: Mapping[str, str] | None = None) -> LockLocation:
    """The lock file the settings name, read from the environment at each call
    through the server settings' one rule (`server_settings.gpu_lock_location`).

    Raises:
        GpuLockNotConfigured: neither setting is set (the setting's `unset`).
        GpuLockMisconfigured: a setting is set but is not an absolute path,
            names a directory that does not exist, or names a path this account
            cannot check; or the lock file named beside the workspaces root is
            not the one under it.
    """
    setting = server_settings.gpu_lock_location(environ)
    if setting.value is None:
        if setting.unset:
            raise GpuLockNotConfigured(setting.problem)
        raise GpuLockMisconfigured(setting.problem)
    source = LOCK_FILE_ENV if setting.variable == LOCK_FILE_ENV else f"under {WORKSPACES_ROOT_ENV}"
    return LockLocation(setting.value, source)


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
    """Who holds the lock, in words, from its description when there is one.

    **A description whose process has exited never names the holder.** It was
    left by a holder that ended without removing it (one that was killed, say);
    the lock is then held by a process that inherited that holder's descriptor,
    or by a new holder that has not removed the description yet (it does as
    soon as it holds the lock) or could not (an account that may not write the
    directory). Which one cannot be told from here, so the words say both."""
    if holder is None:
        return ("a process that has not described itself (a holder that is just starting or "
                "could not write its description, or a process that inherited the lock's "
                "descriptor from one that has ended)")
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
        return ("a process other than the one its description names. The description "
                f"names {text}, which has exited; the lock is held by a process that "
                "inherited its descriptor, or by a new holder that has not described "
                "itself (yet, or because it could not)")
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
        """Describe this holder in `<lock file>.json`. Any description already
        there is a previous holder's (this process holds the lock now), left by
        one that ended without removing it (a killed one): it is removed first,
        so it is never read as this holder's, even when this one cannot write
        its own."""
        try:
            holder_path(self.path).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("GPU lock %s: a previous holder's description %s could not be "
                           "removed (%s).", self.path, holder_path(self.path), exc)
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
        then the lock. Safe to call twice. The unlock is on the open file, which
        a child forked without exec shares, so after it that child does not keep
        the lock either.

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


def _held(path: Path) -> bool:
    """Whether someone holds the lock at this instant.

    Opens the file read-only and creates nothing; takes the lock for an instant
    and gives it back when it is free. A taker that meets that instant tries
    again (`GRACE_SECONDS`) instead of being refused.

    Raises:
        OSError: the lock file exists but cannot be opened or locked.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    os.set_inheritable(fd, False)
    try:
        if _try_lock(fd):
            _unlock(fd)
            return False
        return True
    finally:
        os.close(fd)


def probe(path: Path) -> str | None:
    """None when nobody holds the lock at this instant, else who holds it
    (`describe_holder`). Raises `OSError` as `_held` does."""
    return describe_holder(read_holder(path)) if _held(path) else None


def take_gpu_lock(job_type: str, command: str, *, log: logging.Logger | None = None,
                  environ: Mapping[str, str] | None = None) -> AbstractContextManager:
    """The lock for a CLI's run, taken now: a context manager that releases it.

    **Never a run without it** (review of 8242b47, P2-4): with the lock not
    configured, or a setting unusable, this raises before anything is read.

    Raises:
        GpuLockBusy: another process holds it.
        GpuLockNotConfigured: neither lock setting is set.
        GpuLockMisconfigured: a lock setting is set but unusable.
        OSError: the lock file cannot be opened or locked.
    """
    log = log or logger
    location = lock_location(environ)
    lock = GpuLock(location.path, job_type=job_type, command=command).acquire()
    log.info("GPU lock %s held (%s) until this run ends.", location.path, location.source)
    return lock


def lock_state(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The lock as a status shows it, read now: **the one place where the
    location and the probe become a status** (root-cause review, P24-S3); the
    job runner's start check and status, the dashboard's GPU chip and the
    probe read this.

    `configured`, `path`, `source` (the setting that names it), who holds it at
    this instant (`held_by`, `describe_holder`'s words, None when free) with
    the holder's own description record (`holder`, None when free or not
    described) and whether its process still runs (`holder_alive`: True,
    False, or None when that cannot be told), and `problem` when it is not
    configured or cannot be used or checked, in which case every job is
    refused rather than started unlocked; `unset` says the problem is that
    neither setting is set (the setting's own `unset`, never its words)."""
    state: dict[str, Any] = {"configured": False, "path": None, "source": None,
                             "held_by": None, "holder": None, "holder_alive": None,
                             "problem": None, "unset": False}
    try:
        location = lock_location(environ)
    except GpuLockNotConfigured as exc:
        return {**state, "problem": str(exc), "unset": True}
    except GpuLockMisconfigured as exc:
        return {**state, "problem": str(exc)}
    except OSError as exc:  # a check `lock_location` does not wrap: still a refusal
        return {**state, "problem": f"the GPU lock's location cannot be checked ({exc})"}
    state.update(source=location.source, configured=True, path=str(location.path))
    try:
        held = _held(location.path)
    except OSError as exc:
        return {**state, "problem": f"the GPU lock {location.path} cannot be checked ({exc})"}
    if held:
        holder = read_holder(location.path)
        state.update(held_by=describe_holder(holder), holder=holder,
                     holder_alive=_holder_alive(holder) if holder is not None else None)
    return state


__all__ = [
    "GpuLock",
    "GpuLockBusy",
    "GpuLockMisconfigured",
    "GpuLockNotConfigured",
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
