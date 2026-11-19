"""
The dashboard's download files: one directory, one way to name them, a bounded life, one link.
==============================================================================================
Every file a page offers to download (the Diagnosis result's CSV and report,
Models › Results' export) is written into one directory of this process,
created on first use under the system's temporary directory. Files are named
by what they hold (`file_slug`), never by a server path; Models › Results
writes each as a new file (`write_new`), never replacing one. Moved here from
the Diagnosis panel in research comparison Stage 8b, when Models › Results
became its second page.

**How a file reaches the browser** (OPS-C-R4, the WebUI ledger's B72-E1): by
an opaque id, never by its path. A page gives the path its writer returned
to `offer`, and its download control (a ``gr.Button`` with a ``link``, the
dashboard's only kind) the link `offer` answers: ``download/<id>``. The link
is relative to the page, whose ``<base href="./">`` (Gradio's) is the mount,
so it reaches ``<mount>/download/<id>``: the dashboard's one download route
(`ROUTES`, given to the mount by `src.webui.app.mount_options`), which serves
the file an id names (`offered`) as an attachment under its own name, and
answers 404 for anything else. An id is 32 random bytes
(``secrets.token_urlsafe``), new at each offer and kept in this process's
memory only. Only a file this directory holds can be offered, and the route
takes an id, never a path, so no submitted path becomes a download target.
Gradio's file controls, its file route and its cache are not used for a
download: before, ``gr.DownloadButton`` and ``gr.File`` had Gradio copy each
file into its cache, and the event's answer and the link named that copy's
absolute path (``/ui/gradio_api/file=/tmp/…``). (A subclass of either would
have Gradio write a type stub beside this module at import, an untracked file
that marks every recorded run's checkout dirty: `run_environment`.)

Who may download is who was given the link: each entry keeps its own
permission, decided by its handler before it writes (Diagnosis' exports for
whoever may run a diagnosis; Models › Results' Export for an operator, D8),
and the route asks nothing more of the request, as Gradio's file route did
not.

**How long a file is kept** (B56): a download file holds a result's rows, a
patient's terms among them, so it is not left behind for good. Each lives
`LIFETIME_SECONDS` (a day) after it was written, so a result on screen can be
downloaded for that long; after it, its link answers 404 and the result is
run or exported again.

- Each time a page asks for the directory (`export_dir`, before every write),
  its files older than that are deleted (`prune`); an id lives as long as its
  file (`offered`), and the ids of expired or deleted files are forgotten.
- The directory is removed when the process ends normally (an exit handler).
  A process that ends without it (a crash, Restart Backend's exec) leaves its
  directory: the next directory this account creates deletes the expired files
  of those left behind, and each one once empty (`_sweep_left_behind`). The
  ids end with their process: a link given before a restart answers 404.

Module: src/webui/exports.py
"""
from __future__ import annotations

import atexit
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from starlette.requests import Request
from starlette.responses import FileResponse, PlainTextResponse, Response
from starlette.routing import Route

#: The one directory, created on first use (and again if it was removed).
_EXPORT_DIR: Optional[Path] = None
#: Its name's prefix, under the system's temporary directory.
PREFIX = "shepherd_export_"
#: How long a download file is kept after it was written (module docstring).
LIFETIME_SECONDS = 24 * 3600

#: The ids offered, each to the file it names (`offer`, `offered`): this
#: process's memory only, never written anywhere.
_OFFERED: Dict[str, Path] = {}
_OFFERED_LOCK = threading.Lock()
#: An id as `offer` makes it: 32 random bytes, URL-safe base64 without padding.
_ID = re.compile(r"[A-Za-z0-9_-]{43}")
#: The dashboard's download route, under its mount (`ROUTES`).
ROUTE = "/download/{file_id}"
#: A link as the browser gets it: relative to the page, which is the mount.
LINK = "download/"
#: What the route answers for an id it does not serve, naming nothing.
NOT_OFFERED = ("Not found: this download is not offered here, or it expired a day after it "
               "was written. Run the diagnosis or the export again for a new one.")
#: Every answer of the route: never kept by the browser's cache (a result's
#: rows, a patient's terms among them), never read as another type.
_HEADERS = {"cache-control": "no-store", "x-content-type-options": "nosniff"}


def _expired(mtime: float, now: float) -> bool:
    """Whether a file written at `mtime` is past its lifetime at `now`."""
    return now - mtime > LIFETIME_SECONDS


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
                    and _expired(entry.stat().st_mtime, now)):
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


# ---------------------------------------------------------------------------
# The link: an id for a file, the route that serves it
# ---------------------------------------------------------------------------
def _held(path: Path, now: float) -> bool:
    """Whether `path` is a file the download directory holds, within its
    lifetime: a regular file directly in it, never a link."""
    try:
        return (_EXPORT_DIR is not None and path.parent == _EXPORT_DIR
                and not path.is_symlink() and path.is_file()
                and not _expired(path.stat().st_mtime, now))
    except OSError:
        return False


def offer(path: Any) -> str:
    """The link the browser gets for the download file at `path` (a
    control's ``link``): ``download/<id>`` (`LINK`), a new, unguessable id
    that names the file; nothing that says where it is. Only a file the
    download directory holds, within its lifetime, is offered: any other
    path (another directory, a link, a directory, a file not there) is
    refused with ValueError, so no other path becomes a download target. The
    ids of files that expired or went are forgotten here."""
    now = time.time()
    candidate = Path(path)
    if not _held(candidate, now):
        raise ValueError("Only a file of the dashboard's download directory can be offered.")
    file_id = secrets.token_urlsafe(32)
    with _OFFERED_LOCK:
        for known, held in list(_OFFERED.items()):
            if not _held(held, now):
                del _OFFERED[known]
        _OFFERED[file_id] = candidate
    return LINK + file_id


def offered(file_id: Any) -> Optional[Path]:
    """The file `file_id` names (`offer`) while it lives; None for anything
    else: an id never offered, a malformed or path-like one, a file expired
    or gone (its id is then forgotten)."""
    if not isinstance(file_id, str) or not _ID.fullmatch(file_id):
        return None
    with _OFFERED_LOCK:
        path = _OFFERED.get(file_id)
        if path is None:
            return None
        if not _held(path, time.time()):
            del _OFFERED[file_id]
            return None
        return path


async def download(request: Request) -> Response:
    """The dashboard's download route: the file `offered` names, as an
    attachment under its own name (HEAD and a byte range answered too); for
    anything else 404 and `NOT_OFFERED`, which names nothing."""
    path = offered(request.path_params.get("file_id"))
    if path is None:
        return PlainTextResponse(NOT_OFFERED, status_code=404, headers=_HEADERS)
    return FileResponse(path, filename=path.name, headers=_HEADERS)


#: The dashboard's one download route, added to the mounted app
#: (`src.webui.app.mount_options`): ``<mount>/download/<id>``.
ROUTES: List[Route] = [Route(ROUTE, download, methods=["GET", "HEAD"])]


__all__ = ["LIFETIME_SECONDS", "LINK", "NOT_OFFERED", "PREFIX", "ROUTE", "ROUTES", "download",
           "export_dir", "file_slug", "offer", "offered", "prune", "write_new"]
