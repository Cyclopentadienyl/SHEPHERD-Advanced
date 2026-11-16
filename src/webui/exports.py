"""
The dashboard's download files: one directory, one way to name them, a bounded life.
====================================================================================
Every file a page offers to download (the Diagnosis result's CSV and report,
Models › Results' export) is written into one directory of this process,
created on first use under the system's temporary directory, where Gradio
serves a file a handler returns. Files are named by what they hold
(`file_slug`), never by a server path; Models › Results writes each as a
new file (`write_new`), never replacing one. Moved here from the Diagnosis
panel in research comparison Stage 8b, when Models › Results became its
second page.

**How long a file is kept** (B56): a download file holds a result's rows, a
patient's terms among them, so it is not left behind for good. Each lives
`LIFETIME_SECONDS` (a day) after it was written, so a result on screen can be
downloaded for that long; after it, Download offers nothing and the result is
run or exported again.

- Each time a page asks for the directory (`export_dir`, before every write),
  its files older than that are deleted (`prune`).
- The directory is removed when the process ends normally (an exit handler).
  A process that ends without it (a crash, Restart Backend's exec) leaves its
  directory: the next directory this account creates deletes the expired files
  of those left behind, and each one once empty (`_sweep_left_behind`).
- Gradio serves a copy it keeps in its own cache; the dashboard's
  ``delete_cache`` (`src/webui/app.py`, `CACHE_SWEEP`) deletes those copies on
  the same lifetime, every `SWEEP_SECONDS`, and all of them at shutdown.

Module: src/webui/exports.py
"""
from __future__ import annotations

import atexit
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

#: The one directory, created on first use (and again if it was removed).
_EXPORT_DIR: Optional[Path] = None
#: Its name's prefix, under the system's temporary directory.
PREFIX = "shepherd_export_"
#: How long a download file is kept after it was written (module docstring).
LIFETIME_SECONDS = 24 * 3600
#: How often Gradio's copies are swept (`CACHE_SWEEP`).
SWEEP_SECONDS = 3600
#: The dashboard's ``gr.Blocks(delete_cache=...)``: Gradio's copies of the
#: download files, swept every `SWEEP_SECONDS`, kept `LIFETIME_SECONDS`.
CACHE_SWEEP = (SWEEP_SECONDS, LIFETIME_SECONDS)


def prune(directory: Path, now: Optional[float] = None) -> int:
    """Delete the files of `directory` written more than `LIFETIME_SECONDS`
    before `now` (the time when None); return how many. A file that cannot be
    read or deleted is left, and a link is never followed."""
    now = time.time() if now is None else now
    removed = 0
    try:
        entries = list(directory.iterdir())
    except OSError:
        return 0
    for entry in entries:
        try:
            if (entry.is_file() and not entry.is_symlink()
                    and now - entry.stat().st_mtime > LIFETIME_SECONDS):
                entry.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def _sweep_left_behind(now: float) -> None:
    """The download directories this account's earlier processes left (a
    crash, an exec): their expired files deleted (`prune`), each directory
    removed once it is empty. Another live process's directory keeps its
    files that are not expired, and is made again if it was emptied."""
    mine = getattr(os, "getuid", None)
    for left in Path(tempfile.gettempdir()).glob(PREFIX + "*"):
        try:
            if (left == _EXPORT_DIR or left.is_symlink() or not left.is_dir()
                    or (mine is not None and left.stat().st_uid != mine())):
                continue
        except OSError:
            continue
        prune(left, now)
        try:
            left.rmdir()  # only when nothing is left in it
        except OSError:
            pass


def export_dir() -> Path:
    """The directory this process writes download files into, its expired
    files deleted first (module docstring)."""
    global _EXPORT_DIR
    now = time.time()
    if _EXPORT_DIR is None or not _EXPORT_DIR.exists():
        _EXPORT_DIR = Path(tempfile.mkdtemp(prefix=PREFIX))
        atexit.register(shutil.rmtree, _EXPORT_DIR, ignore_errors=True)
        _sweep_left_behind(now)
    prune(_EXPORT_DIR, now)
    return _EXPORT_DIR


def file_slug(value: Any, default: str) -> str:
    """`value` as a part of a file name: letters, digits, ``_``, ``.`` and
    ``-``, every other run of characters one ``_``; `default` when nothing is
    left."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_") or default


def write_new(name: str, extension: str, text: str) -> Path:
    """`text` written as a new file of the download directory, named `name`
    (already a `file_slug`) and `extension`, never replacing a file: a name
    already taken becomes ``<name>-2``, ``<name>-3``, ... The whole text is
    encoded as UTF-8 first, with no newline translation, so its bytes are the
    text's, and published whole (`publish_bytes`, never replacing a file), as
    the command line's export writes its file: two writes at once never share
    one, and a text that cannot be encoded leaves no file (review fix RC-R1).
    Returns the file's path."""
    from src.utils.file_modes import publish_bytes

    data = text.encode("utf-8")
    directory = export_dir()
    path, counter = directory / f"{name}.{extension}", 2
    while True:
        try:
            publish_bytes(path, data, replace=False)
            return path
        except FileExistsError:
            path, counter = directory / f"{name}-{counter}.{extension}", counter + 1


__all__ = ["CACHE_SWEEP", "LIFETIME_SECONDS", "PREFIX", "SWEEP_SECONDS", "export_dir",
           "file_slug", "prune", "write_new"]
