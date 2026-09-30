"""
Staging files that publish with a plain open()'s mode
=====================================================
A file written to a temporary name and renamed into place carries the
temporary file's mode, and ``tempfile`` creates every file 0600. Before the
atomic writers arrived (September 2026), the UI config was written with
``open(path, "w")`` and the shortest-path table with ``torch.save(path)``: a
new file got the directory's policy and a rewritten one kept its mode. The
atomic writers turned both into 0600 -- new files, and replacements of files
that had been 0644 -- so a service or a second account reading the workspace
could no longer open them.

``create_staging_file`` restores what a plain ``open()`` does, without reading
the umask:

- **A new target** is staged with ``os.open(..., 0o666)``, so the kernel applies
  the umask, or the directory's default ACL where one exists, exactly as it
  would for ``open()``.
- **An existing target** is staged with that file's permission bits, so an
  operator's 0600 stays 0600 and a group-shared 0640 stays 0640.

What renaming cannot reproduce, and this does not claim to: the replacement
is a new inode, owned by the writing account and its group (or the
directory's group where the directory is setgid), and extended ACL entries on
the old file are not copied. Whether another account can read the published
file also depends on the directories above it being searchable by that account.

What this does not defend against: an account that can rename files in the
directory can still substitute the staged file between the write and the
caller's rename, as it could with ``tempfile``. The mode is set through the
open descriptor, so such an account cannot redirect that; the directories
themselves must not be writable by accounts that are not trusted.

Module: src/utils/file_modes.py
"""
from __future__ import annotations

import errno
import os
import secrets
import stat
from pathlib import Path
from typing import IO

#: Only permission bits are carried over: a setuid, setgid or sticky bit on a
#: data file is not something a rewrite should reproduce.
_PERMISSION_BITS = 0o777


def create_staging_file(
    target: str | os.PathLike[str],
    *,
    binary: bool = True,
    encoding: str | None = None,
    prefix: str | None = None,
    suffix: str = ".tmp",
) -> tuple[IO, str]:
    """Open a new file beside ``target`` for writing; return it and its path.

    The file is created with the mode ``open(target, "w")`` would have left
    ``target`` with (see the module docstring), so renaming it over ``target``
    publishes that mode. The caller writes, closes, and renames it, and removes
    it on failure, as it did with ``tempfile``.

    The name is ``prefix`` (``target``'s name by default), a random part, and
    ``suffix``, created with ``O_EXCL`` so it is never an existing file or a
    symlink planted in its place.
    """
    target = Path(target)
    try:
        existing: int | None = stat.S_IMODE(os.stat(target).st_mode) & _PERMISSION_BITS
    except FileNotFoundError:
        existing = None

    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_BINARY", 0)      # Windows: no newline translation
        | getattr(os, "O_NOINHERIT", 0)   # Windows: not inherited by children
        | getattr(os, "O_CLOEXEC", 0)
    )
    stem = target.name if prefix is None else prefix
    for _ in range(100):
        path = target.parent / f"{stem}{secrets.token_hex(8)}{suffix}"
        try:
            fd = os.open(path, flags, 0o666 if existing is None else existing)
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(
            errno.EEXIST, "no unused staging name after 100 tries", str(target.parent)
        )

    try:
        if existing is not None and hasattr(os, "fchmod"):
            # The umask applies to os.open's mode too, so a 0664 target staged
            # under umask 022 comes out 0644 until this sets the old mode
            # exactly. **Through the descriptor, not the path:** between the
            # O_EXCL create and a chmod by name, an account that can rename
            # files in this directory could swap the name for a symlink, and
            # chmod would follow it to another file. Without fchmod (Windows
            # before Python 3.13) os.open's mode has already set the one bit
            # Windows keeps, the read-only flag.
            os.fchmod(fd, existing)
    except BaseException:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    try:
        # fdopen owns the descriptor from here, and closes it if it fails;
        # closing it again could close a number another thread has reused.
        handle = os.fdopen(fd, "wb" if binary else "w", encoding=None if binary else encoding)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return handle, str(path)


__all__ = ["create_staging_file"]
