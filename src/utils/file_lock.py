"""
One cooperative file lock, shared by every lock in the repository (K8-6r.1)
===========================================================================
The platform primitives the GPU lock was built on, moved here as they were so
one implementation serves each lock: the deployment's GPU lock
(`src/utils/gpu_lock.py`), the ledger beside a checkpoint
(`src/evaluation/sidecar.ledger_lock`) and each file of a cohort version's
report store (`src/evaluation/report_store._append_entry`).

**What a lock is.** An exclusive, non-blocking `fcntl.flock` on the lock file
on POSIX; `msvcrt.locking` on the file's first byte on Windows (untested there,
like the rest of the backend; nothing here imports a POSIX-only module on
Windows). The lock belongs to the open file, so two opens of one lock file
exclude each other, within one process too. The kernel releases it when the
last descriptor of that open file closes, so a holder that is killed leaves
nothing to clean up. The descriptor is close-on-exec.

**One lock per file, never a global lock.** Each lock file guards one thing
(one ledger, one store file), and a caller names it. Where one caller holds
several at once, they are taken in the order GPU, ledger, usage history, never
the other way (`docs/DISEASE_SCORER_POLICY.md` rev 16).

**The limits, stated rather than hidden.** Cooperative: a writer that takes no
lock is not stopped by it. Local: the lock file belongs on a local filesystem.
The Windows path is untested. A lock file is never removed: removing it while
another process holds it, or waits for it, would let a third take a new file
under the same name.

Module: src/utils/file_lock.py
"""
from __future__ import annotations

import errno
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

if sys.platform == "win32":  # pragma: no cover - the backend is untested on Windows
    import msvcrt

    fcntl = None
else:
    import fcntl

    msvcrt = None

#: How often a taker tries again while it waits for a busy lock.
RETRY_SECONDS = 0.05


class FileLockBusy(OSError):
    """Another open file held the lock for the whole grace, so nothing was done
    under it. An `OSError`, so a caller's existing failure path for its file
    reports it with its reason; the caller says what that means for its own
    write."""

    def __init__(self, path: Path, grace: float) -> None:
        super().__init__(
            f"{path} is busy: another writer held this lock for the {grace:g} s this one waited"
        )
        self.path = Path(path)
        self.grace = grace


def _busy_errnos() -> tuple:
    return (errno.EACCES, getattr(errno, "EDEADLOCK", errno.EDEADLK), errno.EDEADLK)


def try_lock(fd: int) -> bool:
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


def unlock(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    else:  # pragma: no cover - Windows
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


def open_lock_file(path: Path, nofollow: bool = False) -> int:
    """The lock file opened (created if absent, with the mode `open()` would
    give), its descriptor close-on-exec. Read-only when this account may not
    write it: a lock needs only an open file. When it may not create it either
    (a directory it cannot write, with no lock file there yet), the
    `PermissionError` of the create is raised, not the read-only open's "no
    such file", so the refusal names the real cause.

    With ``nofollow``, a link in the lock file's place is not followed
    (``O_NOFOLLOW``, where the platform has it): the open fails instead of
    creating or locking a file the link names."""
    extra = getattr(os, "O_NOFOLLOW", 0) if nofollow else 0
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | extra, 0o666)
    except PermissionError as denied:
        try:
            fd = os.open(path, os.O_RDONLY | extra)
        except FileNotFoundError:
            raise denied from None
    os.set_inheritable(fd, False)
    return fd


@contextmanager
def exclusive_lock(path: Path, *, grace: float, nofollow: bool = False) -> Iterator[Path]:
    """Hold the lock file at `path` for the body of a `with` block.

    Opens the file (`open_lock_file`, passing ``nofollow`` on) and tries the
    lock every `RETRY_SECONDS` until `grace` seconds have passed; then the
    lock is released and the file closed when the block ends, normally or by
    an exception. The lock file is kept.

    **Only the process that took it releases it**: a process forked inside the
    block shares the open file, and an unlock there would release the
    holder's lock; it only closes its own descriptor.

    Raises:
        FileLockBusy: another open file held the lock for the whole grace.
        OSError: the lock file cannot be opened or locked.
    """
    path = Path(path)
    fd = open_lock_file(path, nofollow=nofollow)
    owner = os.getpid()
    try:
        deadline = time.monotonic() + grace
        while not try_lock(fd):
            if time.monotonic() >= deadline:
                raise FileLockBusy(path, grace)
            time.sleep(RETRY_SECONDS)
        try:
            yield path
        finally:
            if os.getpid() == owner:
                try:
                    unlock(fd)
                except OSError:
                    pass
    finally:
        os.close(fd)


__all__ = [
    "FileLockBusy",
    "RETRY_SECONDS",
    "exclusive_lock",
    "open_lock_file",
    "try_lock",
    "unlock",
]
