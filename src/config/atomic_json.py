"""
One atomic writer for the small JSON settings files
===================================================
The saved deployment settings (`src/config/deployment_settings.py`, written by
`src/utils/settings_store.py`) and the saved runtime settings
(`src/config/runtime_presets.save_runtime_settings`) are written by this one
function, so a reader sees the old file or the new one and never a part of
either (UI-06, 2026-11-16).

**The sequence** (as Syncthing's `CreateAtomic`, Home Assistant's
`write_utf8_file_atomic` and Jupyter's `secure_write` do it): the bytes go to a
new file beside the target (a random name, created with ``O_EXCL``, so never an
existing file or a link planted in its place), are flushed to disk with
``fsync``, and the file is renamed over the target with ``os.replace``; then
the directory is flushed, so the rename itself survives a power cut. A failure
before the rename removes the new file and leaves the target as it was; a
process killed between the write and the rename leaves the target as it was
and a stray ``.<name>.<random>.tmp`` beside it, which no reader opens.

**Modes.** The file is 0600 (the account that runs the service reads it; the
paths it names are the deployment's, and a registry path is a patient-level
location). A directory this creates is 0700, as the XDG base directory
specification asks for a missing configuration directory; a directory that
exists keeps its mode.

**Not here.** Locking: two writers of one file take its lock first
(`src/utils/settings_store.py`, `src.utils.file_lock`). This module is in
`src.config`, below `src.utils`, so the settings readers in `src.config` and
the launcher can import it before anything else starts; it imports nothing
but the standard library.

Module: src/config/atomic_json.py
"""
from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any

#: The mode of a file this writes.
FILE_MODE = 0o600
#: The mode of a directory this creates (an existing one keeps its own).
DIRECTORY_MODE = 0o700

__all__ = ["DIRECTORY_MODE", "FILE_MODE", "ensure_directory", "write_json_atomic"]


def ensure_directory(directory: Any, *, mode: int = DIRECTORY_MODE) -> Path:
    """`directory`, created with its missing parents when absent, each one this
    creates with exactly `mode` (the umask does not apply). An existing
    directory is left as it is. Returns it as a `Path`.

    Raises:
        OSError: a directory cannot be created, or a parent is a file."""
    directory = Path(directory)
    missing = []
    probe = directory
    while not probe.exists():
        missing.append(probe)
        if probe.parent == probe:
            break
        probe = probe.parent
    for path in reversed(missing):
        try:
            os.mkdir(path, mode)
        except FileExistsError:
            continue
        try:
            os.chmod(path, mode)
        except OSError:  # pragma: no cover - a filesystem without modes
            pass
    return directory


def _fsync_directory(directory: Path) -> None:
    """Flush `directory`'s entries to disk, where the platform allows opening a
    directory (POSIX); a no-op elsewhere."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(directory, flags)
    except OSError:  # Windows cannot open a directory; nothing to flush there
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - a filesystem that refuses a directory fsync
        pass
    finally:
        os.close(fd)


def write_json_atomic(path: Any, data: Any, *, file_mode: int = FILE_MODE,
                      directory_mode: int = DIRECTORY_MODE) -> None:
    """Write `data` as JSON (indented, UTF-8, a final newline) to `path` whole.

    Serialised first (``allow_nan=False``), so a value JSON cannot hold raises
    before anything is created. The parent directory is created with
    `directory_mode` when absent (`ensure_directory`); the file is written
    with exactly `file_mode` to a staging name beside `path`, flushed, renamed
    over `path`, and the directory flushed (module docstring).

    Raises:
        TypeError, ValueError: `data` cannot be written as JSON; nothing was created.
        OSError: the directory or the file cannot be written; `path` is as it was.
    """
    path = Path(path)
    payload = (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
               ).encode("utf-8")
    directory = ensure_directory(path.parent, mode=directory_mode)
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
             | getattr(os, "O_NOINHERIT", 0) | getattr(os, "O_CLOEXEC", 0))
    for _ in range(100):
        staged = directory / f".{path.name}.{secrets.token_hex(8)}.tmp"
        try:
            fd = os.open(staged, flags, file_mode)
        except FileExistsError:
            continue
        break
    else:  # pragma: no cover - 100 random names all taken
        raise FileExistsError(f"no unused staging name beside {path} after 100 tries")
    try:
        try:
            if hasattr(os, "fchmod"):
                # Exactly `file_mode`: the umask may have removed bits from it.
                os.fchmod(fd, file_mode)
            view = memoryview(payload)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(staged, path)
    except BaseException:
        try:
            staged.unlink()
        except OSError:
            pass
        raise
    _fsync_directory(directory)
