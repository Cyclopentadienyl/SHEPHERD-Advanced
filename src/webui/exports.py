"""
The dashboard's download files: one directory, one way to name them.
====================================================================
Every file a page offers to download (the Diagnosis result's CSV and report,
Models › Results' export) is written into one directory of this process,
created on first use under the system's temporary directory, where Gradio
serves a file a handler returns. Files are named by what they hold
(`file_slug`), never by a server path. Moved here from the Diagnosis panel in
research comparison Stage 8b, when Models › Results became its second page;
periodic cleanup is future work, as it was.

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


__all__ = ["export_dir", "file_slug"]
