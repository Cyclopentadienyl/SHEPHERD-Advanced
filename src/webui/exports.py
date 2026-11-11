"""
The dashboard's download files: one directory, one way to name them.
====================================================================
Every file a page offers to download (the Diagnosis result's CSV and report,
Models › Results' export) is written into one directory of this process,
created on first use under the system's temporary directory, where Gradio
serves a file a handler returns. Files are named by what they hold
(`file_slug`), never by a server path; Models › Results writes each as a
new file (`write_new`), never replacing one. Moved here from the Diagnosis
panel in research comparison Stage 8b, when Models › Results became its
second page; periodic cleanup is future work, as it was.

Module: src/webui/exports.py
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Optional

#: The one directory, created on first use (and again if it was removed).
_EXPORT_DIR: Optional[Path] = None


def export_dir() -> Path:
    """The directory this process writes download files into."""
    global _EXPORT_DIR
    if _EXPORT_DIR is None or not _EXPORT_DIR.exists():
        _EXPORT_DIR = Path(tempfile.mkdtemp(prefix="shepherd_export_"))
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


__all__ = ["export_dir", "file_slug", "write_new"]
