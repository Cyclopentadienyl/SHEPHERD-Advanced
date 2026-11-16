"""
The saved deployment settings file: where it is, and what it holds
==================================================================
UI-06 (the owner's decision of 2026-11-16): the data locations a deployment
uses -- the workspaces folder, the results registry, the import folder and the
training evidence folders -- can be chosen in Settings › Data locations and are
kept in one small file outside the repository, read again at the next start.
This module is that file's location, its format and its reading; the settings
themselves are resolved, checked and worded by `src/config/server_settings.py`
(the one reader every caller goes through), and every write is
`src/utils/settings_store.py`'s, under the file's lock.

**Where it is** (`settings_location`): `SHEPHERD_SETTINGS_FILE` when it is set
(an explicit file: an A/B slot, a test, the deployment probe), else the
account's user configuration directory, as platformdirs names it (not
installed here, so these few lines are ours):

| Platform | File |
|---|---|
| Linux and others | ``${XDG_CONFIG_HOME:-~/.config}/shepherd-advanced/deployment.json`` |
| macOS | ``~/Library/Application Support/shepherd-advanced/deployment.json`` |
| Windows | ``%LOCALAPPDATA%\\shepherd-advanced\\deployment.json`` |

Outside the repository, so a checkout can be replaced without losing the
choice. **A mapping given as `environ` is the only thing read**: its
`SHEPHERD_SETTINGS_FILE`, `XDG_CONFIG_HOME`, `HOME` and `LOCALAPPDATA`. A
mapping that names none of them has no saved file, so a caller that passes
``{"SHEPHERD_REGISTRY": ...}`` reads exactly that, as before; `None` is the
process's environment.

**What it holds**::

    {"format": "shepherd-deployment-settings/1",
     "applied": {"workspaces_root": "/…", "registry": "/…", "intake_dir": "/…",
                 "training_evidence_dirs": ["/…", "/…"]},
     "saved":   {…the same keys…},
     "applied_at": "2026-11-16T09:00:00Z", "saved_at": "2026-11-16T09:00:00Z"}

`applied` is what every process uses: the service, a command run by hand and a
job the service starts. `saved` is the latest choice made in Settings. They
differ only while a change waits to be applied (`settings_store.apply_saved`:
not while a GPU job could be using the old locations), so saving never splits
the GPU lock between processes. Every key is optional; a missing one is not
set. Unknown keys, at the top and inside either object, are kept when the file
is rewritten (`compose`). The deployment mode is not stored here: it is an
explicit environment value or the local start's default
(`server_settings.deployment_mode`).

**Reading** (`read_saved`): an absent file is nothing saved. A file that
cannot be read -- an I/O or permission error, text that is not JSON, JSON that
is not an object of this shape, another format -- is a problem that names the
file and why, for every setting that would come from it; it is never read as
empty and no other file is read instead. Reads are cached by the file's
identity (device, inode, size, modification and change times), so a reader
that asks at each use costs one ``stat``.

Gradio-free and torch-free, standard library only: the launcher reads it
before anything starts.

Module: src/config/deployment_settings.py
"""
from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple

#: The variable naming the file explicitly.
ENV_SETTINGS_FILE = "SHEPHERD_SETTINGS_FILE"
#: The file's format, compared exactly.
FORMAT = "shepherd-deployment-settings/1"
#: The directory under the user configuration directory, and the file in it.
APP_DIRECTORY = "shepherd-advanced"
FILE_NAME = "deployment.json"
#: The lock beside the file every write takes (`settings_store`).
LOCK_SUFFIX = ".lock"
#: What an unreadable file is renamed to before it is replaced
#: (`settings_store.replace_unreadable`), followed by a UTC stamp.
UNREADABLE_SUFFIX = ".unreadable-"

#: The data locations the file holds, one key each.
KEY_WORKSPACES_ROOT = "workspaces_root"
KEY_REGISTRY = "registry"
KEY_INTAKE_DIR = "intake_dir"
KEY_TRAINING_EVIDENCE_DIRS = "training_evidence_dirs"
#: The keys holding one path (a string), and the one holding a list of paths.
PATH_KEYS: Tuple[str, ...] = (KEY_WORKSPACES_ROOT, KEY_REGISTRY, KEY_INTAKE_DIR)
LIST_KEYS: Tuple[str, ...] = (KEY_TRAINING_EVIDENCE_DIRS,)
KEYS: Tuple[str, ...] = PATH_KEYS + LIST_KEYS

__all__ = [
    "APP_DIRECTORY",
    "ENV_SETTINGS_FILE",
    "FILE_NAME",
    "FORMAT",
    "KEYS",
    "KEY_INTAKE_DIR",
    "KEY_REGISTRY",
    "KEY_TRAINING_EVIDENCE_DIRS",
    "KEY_WORKSPACES_ROOT",
    "LIST_KEYS",
    "LOCK_SUFFIX",
    "Location",
    "PATH_KEYS",
    "SavedFile",
    "UNREADABLE_SUFFIX",
    "compose",
    "lock_path",
    "normalise_value",
    "parse",
    "read_file",
    "read_saved",
    "settings_location",
    "utc_stamp",
]


class Location(NamedTuple):
    """Where the saved settings file is for one environment. `path` is None
    when there is none: nothing names a place (`problem` None: nothing is
    saved, and nothing can be), or `SHEPHERD_SETTINGS_FILE` names one that is
    refused (`problem` says why). `explicit`: named by that variable."""

    path: Optional[Path]
    problem: Optional[str] = None
    explicit: bool = False


def _absolute(value: Optional[str]) -> Optional[Path]:
    """`value` as a path when it is a non-empty absolute one, else None."""
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else None


def settings_location(environ: Optional[Mapping[str, str]] = None, *,
                      platform: Optional[str] = None) -> Location:
    """The saved settings file for `environ` (the process's when None), by the
    table in the module docstring; `platform` is `sys.platform` unless given.

    A relative `SHEPHERD_SETTINGS_FILE` is refused rather than read against the
    working directory; a relative `XDG_CONFIG_HOME` is ignored, as the XDG
    specification asks."""
    env = os.environ if environ is None else environ
    named = env.get(ENV_SETTINGS_FILE)
    if named:
        path = Path(named)
        if not path.is_absolute():
            return Location(None, (
                f"{ENV_SETTINGS_FILE} {named!r} is not an absolute path, so which settings file "
                "is meant cannot be told; give its absolute path"), explicit=True)
        return Location(path, explicit=True)
    platform = sys.platform if platform is None else platform
    if platform.startswith("win"):
        base = _absolute(env.get("LOCALAPPDATA"))
    elif platform == "darwin":
        home = _absolute(env.get("HOME"))
        base = home / "Library" / "Application Support" if home else None
    else:
        home = _absolute(env.get("HOME"))
        base = _absolute(env.get("XDG_CONFIG_HOME")) or (home / ".config" if home else None)
    if base is None:
        return Location(None)
    return Location(base / APP_DIRECTORY / FILE_NAME)


def lock_path(settings_file: Path) -> Path:
    """The lock file beside `settings_file` that every write takes."""
    return settings_file.with_name(settings_file.name + LOCK_SUFFIX)


def utc_stamp(moment: Optional[datetime] = None) -> str:
    """A UTC time as the file records it (``2026-11-16T09:00:00Z``)."""
    moment = moment or datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SavedFile(NamedTuple):
    """The saved settings file as read now.

    `path`: the file (None when there is no place for one). `exists`: it is
    there. `problem`: why it cannot be read, naming the file (None when it was
    read or is absent); `repairable` says the problem is its content (not
    JSON, the wrong shape, another format), which
    `settings_store.replace_unreadable` may rename away, not an I/O or
    permission error, which is fixed where it is. `applied` and `saved`: the
    known keys each holds, normalised (`normalise_value`; a key not set is
    absent). `document`: the whole object as read, for a rewrite that keeps
    unknown keys."""

    path: Optional[Path]
    exists: bool = False
    problem: Optional[str] = None
    repairable: bool = False
    applied: Mapping[str, Any] = {}
    saved: Mapping[str, Any] = {}
    applied_at: Optional[str] = None
    saved_at: Optional[str] = None
    document: Mapping[str, Any] = {}

    @property
    def pending(self) -> Tuple[str, ...]:
        """The keys whose saved value is not the applied one, in `KEYS` order."""
        return tuple(key for key in KEYS if self.saved.get(key) != self.applied.get(key))


def normalise_value(key: str, value: Any) -> Any:
    """A value of `key` as the file holds it, or None when it is not set: a
    path key's non-empty string; the list key's tuple of non-empty strings
    (None when empty).

    Raises:
        ValueError: `key` is not a data location, or `value` is not of its shape."""
    if key in PATH_KEYS:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a path (a string), not {type(value).__name__}")
        return value or None
    if key in LIST_KEYS:
        if value is None:
            return None
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise ValueError(f"{key} must be a list of paths, not {type(value).__name__}")
        if not all(isinstance(entry, str) for entry in value):
            raise ValueError(f"{key} must be a list of paths (strings)")
        entries = tuple(entry for entry in value if entry)
        return entries or None
    raise ValueError(f"{key!r} is not a data location the settings file holds")


def _values(document: Mapping[str, Any], part: str) -> Dict[str, Any]:
    """The known keys of `document[part]`, normalised; ValueError for a part
    or a value of the wrong shape."""
    held = document.get(part, {})
    if held is None:
        held = {}
    if not isinstance(held, dict):
        raise ValueError(f"its {part!r} part is not a JSON object")
    values: Dict[str, Any] = {}
    for key in KEYS:
        try:
            value = normalise_value(key, held.get(key))
        except ValueError as exc:
            raise ValueError(f"its {part!r} part is of the wrong shape: {exc}") from None
        if value is not None:
            values[key] = value
    return values


def _unreadable(path: Path, reason: str, *, repairable: bool) -> SavedFile:
    return SavedFile(path, exists=True, repairable=repairable,
                     problem=f"The saved settings file {path} cannot be read ({reason})")


def parse(path: Path, data: bytes) -> SavedFile:
    """`data`, the bytes of the file at `path`, as a `SavedFile`."""
    try:
        document = json.loads(data.decode("utf-8"))
    except UnicodeDecodeError:
        return _unreadable(path, "it is not UTF-8 text", repairable=True)
    except json.JSONDecodeError as exc:
        return _unreadable(path, f"it is not valid JSON: {exc.msg} at line {exc.lineno}",
                           repairable=True)
    if not isinstance(document, dict):
        return _unreadable(path, "it does not hold a JSON object", repairable=True)
    if "format" not in document:
        return _unreadable(path, f"it names no format; this version reads {FORMAT}",
                           repairable=True)
    if document.get("format") != FORMAT:
        return _unreadable(path, f"its format is not {FORMAT}, the one this version reads",
                           repairable=True)
    try:
        applied = _values(document, "applied")
        saved = _values(document, "saved")
    except ValueError as exc:
        return _unreadable(path, str(exc), repairable=True)
    stamps = {name: document.get(name) if isinstance(document.get(name), str) else None
              for name in ("applied_at", "saved_at")}
    return SavedFile(path, exists=True, applied=applied, saved=saved,
                     applied_at=stamps["applied_at"], saved_at=stamps["saved_at"],
                     document=document)


def read_file(path: Path) -> SavedFile:
    """The file at `path` read now, never from the cache (a writer reads it so,
    under the lock). Absent: nothing saved."""
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError:
        return SavedFile(Path(path))
    except IsADirectoryError:
        return _unreadable(Path(path), "it is a directory", repairable=False)
    except OSError as exc:
        return _unreadable(Path(path), f"{type(exc).__name__}: {exc.strerror or exc}",
                           repairable=False)
    return parse(Path(path), data)


_cache_lock = threading.Lock()
_cache: Dict[str, Tuple[Tuple[int, ...], SavedFile]] = {}


def read_saved(environ: Optional[Mapping[str, str]] = None) -> SavedFile:
    """The saved settings file for `environ` (`settings_location`), as read now.

    No place for one: an empty `SavedFile` (`path` None), with the location's
    problem when `SHEPHERD_SETTINGS_FILE` is refused. Cached by the file's
    identity, so a read at each use costs one ``stat``; a rewrite (a new
    inode, by `os.replace`) is read again."""
    location = settings_location(environ)
    if location.path is None:
        return SavedFile(None, problem=location.problem)
    path = location.path
    try:
        info = os.stat(path)
    except FileNotFoundError:
        return SavedFile(path)
    except OSError:
        return read_file(path)  # says why it cannot be read
    identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    key = str(path)
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None and cached[0] == identity:
        return cached[1]
    result = read_file(path)
    with _cache_lock:
        _cache[key] = (identity, result)
    return result


def _merged(previous: Any, values: Mapping[str, Any]) -> Dict[str, Any]:
    """`previous` (an object of the file, or anything else read as empty) with
    each known key replaced by `values` (removed when absent there); unknown
    keys of `previous` are kept."""
    out = {key: value for key, value in (previous.items() if isinstance(previous, dict) else ())
           if key not in KEYS}
    for key in KEYS:
        value = values.get(key)
        if value is not None:
            out[key] = list(value) if key in LIST_KEYS else value
    return out


def compose(document: Mapping[str, Any], *, saved: Optional[Mapping[str, Any]] = None,
            applied: Optional[Mapping[str, Any]] = None,
            moment: Optional[datetime] = None) -> Dict[str, Any]:
    """The file's next content: `document` (as read; unknown keys kept) with
    `saved` and/or `applied` replacing the known keys of that part (all of
    them: a key absent from the mapping is no longer set), each part's stamp
    renewed. A part not given is kept as it was."""
    out: Dict[str, Any] = dict(document) if isinstance(document, Mapping) else {}
    out["format"] = FORMAT
    stamp = utc_stamp(moment)
    for part, values in (("applied", applied), ("saved", saved)):
        if values is None:
            out.setdefault(part, {})
            continue
        out[part] = _merged(out.get(part), values)
        out[f"{part}_at"] = stamp
    return out
