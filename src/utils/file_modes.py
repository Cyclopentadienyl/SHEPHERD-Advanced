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

``publish_bytes`` is the whole publish sequence for bytes already in memory --
stage, write, flush, fsync, rename, and remove the staged file on any failure --
returning the digest of exactly the bytes written. ``publish_chunks`` is the
same sequence for bytes produced chunk by chunk (a file too large to hold as
one bytes object), and ``publish_bytes`` is that sequence with one chunk. Each
writer that predates them repeats that sequence itself; they are not migrated
here.

**Publishing without replacing** (``publish_bytes(..., replace=False)``, since
K8-6r.4): the staged file is hard-linked to the target, so the check that
nothing is there and the placement are one step, and the target is never
overwritten. Where the filesystem has no hard links, the target is first
claimed as a new empty file and the staged file then renamed onto that claim;
the limit of that fallback is stated at ``publish_bytes``.

Module: src/utils/file_modes.py
"""
from __future__ import annotations

import errno
import hashlib
import os
import secrets
import stat
from pathlib import Path
from typing import IO, Iterable, Tuple

#: Only permission bits are carried over: a setuid, setgid or sticky bit on a
#: data file is not something a rewrite should reproduce.
_PERMISSION_BITS = 0o777


def _new_file_flags() -> int:
    """The flags of a new file that must not exist yet (``O_EXCL``: not even
    as a dangling link): a staging name (`create_staging_file`) and the claim
    of a target published without replacing (`publish_bytes`,
    ``replace=False``). Read at each call, as the platform states them."""
    return (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_BINARY", 0)      # Windows: no newline translation
        | getattr(os, "O_NOINHERIT", 0)   # Windows: not inherited by children
        | getattr(os, "O_CLOEXEC", 0)
    )


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

    flags = _new_file_flags()
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


def publish_bytes(
    target: str | os.PathLike[str], data: bytes, *, replace: bool = True
) -> str:
    """Write `data` to `target` whole, and return the SHA-256 of exactly those bytes.

    The bytes go to a staging file beside `target` (`create_staging_file`: an
    unused name, the mode ``open()`` would give, an existing target's mode
    kept), are flushed to disk and renamed over `target`. A failure before the
    file is placed (the rename, or with ``replace=False`` the link or the
    replace of the claim) removes the staged file and leaves `target` as it
    was, but for the empty file the ``replace=False`` fallback claims there
    (below), so a reader sees the old file or the new one and never a part of
    either. The placement is the commit point: an `OSError` after it (only
    the removal of the staging name, with ``replace=False``) does not report
    a whole `target` as failed.

    **The digest is of the bytes written, not of the file read back** (the
    writers' rule since contract M3b): a manifest that binds this file can take
    it without a second read, and a file replaced after the rename is not the
    one recorded. `data` must be `bytes`, which cannot change between the hash
    and the write; anything else is refused before a staging file exists.

    **Creates no directories.** A missing parent raises `FileNotFoundError`:
    where a file is published is the caller's decision, and a directory made
    here would be one nobody chose.

    **``replace=True``** (the default, every writer before K8-6r.4):
    ``os.replace`` overwrites an existing target.

    **``replace=False``: an existing target is never overwritten.** The staged
    file is hard-linked to `target` (``os.link``), which fails when anything is
    there, a dangling link included, so the check and the placement are one
    step; the staging name is then removed, and an `OSError` doing so leaves
    that name, a second name for the same bytes, beside the published
    `target` rather than report it as failed. A target that is taken raises
    `FileExistsError` naming `target` only, with the staged file removed and
    `target` as it was. Only that error is a collision: any other error from
    ``os.link`` (on Linux, EPERM where the filesystem has no hard links,
    ENOTSUP or EXDEV) falls back to claiming `target` as a new empty file
    (``O_CREAT | O_EXCL``, which also fails with `FileExistsError` when
    anything, a dangling link included, is there) and then ``os.replace``-ing
    the staged file onto that empty claim, so the bytes still appear whole.
    A failure after the claim leaves the empty claimed file at `target`, never
    removed by name: the caller reports it as a failed publication, and a later
    writer that must not overwrite is refused by it.

    **The limit of that fallback, stated rather than hidden.** Its
    ``os.replace`` is itself an overwriting call. It does not overwrite another
    writer's file only because `target` is still the empty file this same call
    claimed with ``O_CREAT | O_EXCL`` a moment before, and because every
    writer that publishes there follows the same rule (claim first, never
    replace or remove another's claim). It is not safe against a writer that
    does not, or a person, replacing or removing the claimed file between the
    claim and the replace, and no check before the replace is claimed to close
    that window. The error numbers are Linux's; what Windows gives on FAT or
    exFAT is not checked.
    """
    if not isinstance(data, bytes):
        raise TypeError(
            f"publish_bytes takes bytes, got {type(data).__name__}; nothing was written to "
            f"{target}"
        )
    return publish_chunks(target, (data,), replace=replace)[0]


def publish_chunks(
    target: str | os.PathLike[str], chunks: Iterable[bytes], *, replace: bool = True
) -> Tuple[str, int]:
    """Write the bytes `chunks` yields, in order, to `target` whole, and return
    the SHA-256 of exactly those bytes and their length.

    `publish_bytes`' sequence (its docstring states it, ``replace`` included)
    for bytes made one chunk at a time, so a large file is never held as one
    bytes object: each chunk is hashed and written to the staging file as it
    comes, then the whole file is flushed to disk and placed at `target` in one
    step. A failure at any point, a chunk that is not `bytes` or an interrupt
    while the chunks are made among them, removes the staged file and leaves
    `target` as it was (but for the empty claim of the ``replace=False``
    fallback, as there), so a reader never sees part of the file.
    """
    target = Path(target)
    digest = hashlib.sha256()
    size = 0
    handle, staged = create_staging_file(target, binary=True)
    try:
        with handle:
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise TypeError(f"publish_chunks takes chunks of bytes, got "
                                    f"{type(chunk).__name__}; nothing was written to {target}")
                digest.update(chunk)
                handle.write(chunk)
                size += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(staged, target)
        else:
            _place_new(staged, target)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise
    return digest.hexdigest(), size


def _taken(target: Path) -> FileExistsError:
    """The collision `publish_bytes(replace=False)` raises: `target` only, never
    the staging name beside it."""
    return FileExistsError(errno.EEXIST, "a file is already there and is never replaced",
                           str(target))


def _place_new(staged: str, target: Path) -> None:
    """Put the written `staged` file at `target`, which must not exist (see
    `publish_bytes`, ``replace=False``). The caller removes `staged` on failure."""
    try:
        os.link(staged, target)
    except FileExistsError:
        raise _taken(target) from None
    except OSError:
        # No hard link here (a filesystem without them, or another device):
        # claim the target as a new empty file, then replace that claim.
        try:
            fd = os.open(target, _new_file_flags(), 0o666)
        except FileExistsError:
            raise _taken(target) from None
        os.close(fd)
        os.replace(staged, target)
        return
    # The link is the commit point: `target` holds the whole bytes. Removing the
    # staging name is tidying, and its failure must not report that file as failed.
    try:
        Path(staged).unlink(missing_ok=True)
    except OSError:
        pass


__all__ = ["create_staging_file", "publish_bytes", "publish_chunks"]
