"""
One local start per configuration: the instance lock, its info file, the launch log
====================================================================================
UI-06 (plan §4.5; the owner's decision of 2026-11-16). A clinician starts the
dashboard by double click, and may do it again while it runs. Two backends
serving the same data locations must never both start, and a second start
should open the one that runs; a start without a terminal needs a log and a
way to say what failed. This module is the gradio-free part of that, shared by
the launcher (`scripts/launch/shep_launch.py`), which takes the lock and
writes the info file, and the service (`src/api/main.py`), which answers its
id on ``/health`` and removes the file at shutdown.

**Where** (`state_directory`): `SHEPHERD_STATE_DIR` when it is set (an absolute
path: tests, the A/B slots), else the account's user state directory, as
platformdirs names it (not installed here, so these few lines are ours):

| Platform | Directory |
|---|---|
| Linux and others | ``${XDG_STATE_HOME:-~/.local/state}/shepherd-advanced`` |
| macOS | ``~/Library/Application Support/shepherd-advanced`` (``$XDG_STATE_HOME/shepherd-advanced`` when that is set) |
| Windows | ``%LOCALAPPDATA%\\shepherd-advanced`` |

**A mapping given as `environ` is the only thing read** (its
`SHEPHERD_STATE_DIR`, `XDG_STATE_HOME`, `HOME` and `LOCALAPPDATA`), as for the
saved settings file (`src.config.deployment_settings`); none of them, no state
directory, and the launcher says it starts without the single-instance check.

**One configuration** is the repository's root and the saved settings file
the service will read (`instance_key`): a hash of both names its lock,
``instances/<key>.lock``, and its info file, ``instances/<key>.json``. An A/B
slot started with its own ``--settings-file`` is another configuration.

**The lock** is an exclusive lock taken without waiting (`claim`;
`src.utils.file_lock`). The launcher takes it before it starts anything and
the server it becomes keeps it: the descriptor is made inheritable just
before the exec (`hand_to_exec`), so the server process holds it until it
exits, also across Restart Backend's re-exec; a job the server starts does not
inherit it (`subprocess` closes every other descriptor). Where the launcher
waits beside the server (Windows) it holds the lock itself. The kernel
releases it when the process ends, so a crash leaves nothing held; the lock
file is never removed.

**The info file** (`write_info`, the one atomic JSON writer, 0600) is written
by the launcher once it holds the lock: the format, a random instance id, the
pid that becomes the server's, the port, the URL, the repository root, the
settings file, the start time, the launch log, and the settings variables the
server is started with (``start_settings``, `SETTINGS_VARIABLES`). The service
is given the id and the file as `SHEPHERD_INSTANCE_ID` and
`SHEPHERD_INSTANCE_FILE` (the launcher's, like `SHEPHERD_BIND_HOST`: it removes
them from its own environment first), answers the id on ``/health``
(`served_instance_id`) and removes the file at shutdown when it still names
that id (`remove_own_info`). A file found while the lock is free is stale and
replaced: **what a crash leaves** (FL-SVC). Its server ended without its
orderly stop (killed by SIGKILL or the out-of-memory killer, crashed, forced
to quit, or the machine stopped), which writes no line of its own; the next
start logs that it found it (`ended_without_stop_words`).

**Identity** is the lock held and ``/health`` answering with the info file's
id: an answer of 200 alone never counts, since another program may hold the
port (`instance_answers`, in the launcher, which reads ``/health``).

**A second start opens the running instance only when it gets what it asks
for** (`asks_not_met`; review of 6a2d52e, OPS-C-R1). A start asks for a
setting explicitly when its environment sets the deployment mode or a data
location (`ASKED_VARIABLES`, set and not empty, as the resolution reads them).
Each is compared with what the running service has in effect now: the one
resolution (`src.config.server_settings`) read for the service's environment,
which is this start's with the recorded ``start_settings`` in place of its own
(`running_environment`), so the saved file is read now and a Save and Apply
made in the running service counts. Nothing asked, or all of it in effect: the
launcher opens the running service, as before. Anything else: it opens
nothing, says what runs and what was asked, and exits non-zero; the settings
are never part of the key, so one configuration still has one backend. An info
file without ``start_settings`` (an earlier launcher's) says nothing about
them, so every ask counts as not met.

**The launch log** (`log_file`): a start without a terminal appends its output
to ``logs/launch.log`` here, the server's included, and an error page beside
it says what failed (`error_page`). The log is moved to ``launch.log.1`` once it
passes `LOG_ROTATE_BYTES`, so it stays bounded.

Standard library only (and `src.utils.file_lock`, `src.config.atomic_json`,
`src.config.server_settings`): the launcher imports it before anything starts.

Module: src/utils/launch_instance.py
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple

from src.config import server_settings
from src.config.atomic_json import ensure_directory, write_json_atomic
from src.config.deployment_settings import APP_DIRECTORY, utc_stamp
from src.config.server_settings import (
    ENV_BIND_HOST,
    ENV_DEPLOYMENT_MODE,
    ENV_LOCAL_ENTRY,
    FIELD_NAMES,
    SAVED_VARIABLES,
)
from src.utils.file_lock import open_lock_file, try_lock

#: The variable naming the state directory explicitly.
ENV_STATE_DIR = "SHEPHERD_STATE_DIR"
#: The launcher's, passed to the server it starts: the instance's id, which
#: ``/health`` answers, and its info file, which the server removes at shutdown.
ENV_INSTANCE_ID = "SHEPHERD_INSTANCE_ID"
ENV_INSTANCE_FILE = "SHEPHERD_INSTANCE_FILE"
#: The info file's format, compared exactly.
INFO_FORMAT = "shepherd-instance/1"
#: The directories under the state directory.
INSTANCES = "instances"
LOGS = "logs"
#: The launch log, the error page beside it, and the size past which the log
#: is moved aside once at the next start.
LOG_NAME = "launch.log"
ERROR_PAGE_NAME = "launch-error.html"
LOG_ROTATE_BYTES = 10 * 1024 * 1024
#: The variables a server reads its settings from (`server_settings`): the
#: deployment mode, the launcher's local-start mark and bind host, which decide
#: the mode with it, and the data locations. The saved settings file is not
#: one: it is the configuration itself (`instance_key`).
SETTINGS_VARIABLES = (ENV_DEPLOYMENT_MODE, ENV_LOCAL_ENTRY, ENV_BIND_HOST,
                      *SAVED_VARIABLES.values())
#: Those a start sets to ask for a setting: the mode and the data locations
#: (the launcher's own two never are; it sets them).
ASKED_VARIABLES = (ENV_DEPLOYMENT_MODE, *SAVED_VARIABLES.values())
#: What the running service has when its info file does not record it.
NOT_RECORDED = "not recorded (it was started by an earlier version of the launcher)"

__all__ = [
    "ASKED_VARIABLES",
    "Ask",
    "ENV_INSTANCE_FILE",
    "ENV_INSTANCE_ID",
    "ENV_STATE_DIR",
    "INFO_FORMAT",
    "Instance",
    "LOG_ROTATE_BYTES",
    "NOT_RECORDED",
    "SETTINGS_VARIABLES",
    "asks_not_met",
    "claim",
    "ended_without_stop_words",
    "error_page",
    "hand_to_exec",
    "instance_files",
    "instance_key",
    "log_file",
    "new_instance_id",
    "read_info",
    "remove_own_info",
    "rotate_log",
    "running_environment",
    "served_instance_id",
    "state_directory",
    "write_info",
]


def _absolute(value: Optional[str]) -> Optional[Path]:
    """`value` as a path when it is a non-empty absolute one, else None."""
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else None


def state_directory(environ: Optional[Mapping[str, str]] = None, *,
                    platform: Optional[str] = None) -> Optional[Path]:
    """The state directory for `environ` (the process's when None), by the
    table in the module docstring; None when nothing names one. A relative
    value is ignored, as the XDG specification asks."""
    env = os.environ if environ is None else environ
    named = _absolute(env.get(ENV_STATE_DIR))
    if named is not None:
        return named
    platform = sys.platform if platform is None else platform
    if platform.startswith("win"):
        base = _absolute(env.get("LOCALAPPDATA"))
        return base / APP_DIRECTORY if base else None
    xdg = _absolute(env.get("XDG_STATE_HOME"))
    if xdg is not None:
        return xdg / APP_DIRECTORY
    home = _absolute(env.get("HOME"))
    if home is None:
        return None
    if platform == "darwin":
        return home / "Library" / "Application Support" / APP_DIRECTORY
    return home / ".local" / "state" / APP_DIRECTORY


def instance_key(repo_root: Any, settings_file: Optional[Any]) -> str:
    """The configuration's key: a hash of the repository root and the saved
    settings file (None when there is none), each as an absolute path."""
    def name(path: Any) -> str:
        return os.path.normcase(os.path.abspath(str(path))) if path is not None else ""

    digest = hashlib.sha256(f"{name(repo_root)}\0{name(settings_file)}".encode("utf-8"))
    return digest.hexdigest()[:16]


class Instance(NamedTuple):
    """One configuration's files under the state directory."""

    key: str
    lock: Path
    info: Path


def instance_files(state_dir: Path, key: str) -> Instance:
    directory = Path(state_dir) / INSTANCES
    return Instance(key, directory / f"{key}.lock", directory / f"{key}.json")


def claim(lock: Path) -> Optional[int]:
    """Take the instance lock at `lock` without waiting: its descriptor
    (close-on-exec until `hand_to_exec`), or None when another process holds
    it. The directory is created 0700 when absent.

    Raises:
        OSError: the directory or the lock file cannot be created or opened."""
    ensure_directory(Path(lock).parent)
    fd = open_lock_file(Path(lock))
    try:
        if try_lock(fd):
            return fd
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    return None


def hand_to_exec(fd: int) -> None:
    """Keep the lock through the coming exec: the server it becomes holds it
    until it exits (module docstring)."""
    os.set_inheritable(fd, True)


def new_instance_id() -> str:
    return secrets.token_hex(16)


def read_info(path: Path) -> Optional[Dict[str, Any]]:
    """The info file at `path`, or None when it is absent or is not one
    (unreadable, not JSON, another format): such a file says nothing about a
    running instance."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("format") != INFO_FORMAT:
        return None
    if not isinstance(document.get("instance_id"), str) or not isinstance(document.get("url"), str):
        return None
    return document


def write_info(path: Path, *, instance_id: str, pid: int, port: int, url: str,
               repo_root: Any, settings_file: Optional[Any], log: Optional[Any],
               environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Write the info file at `path` whole (0600, its directory 0700) and
    return what it holds. `environ` is the environment the server is started
    with: its `SETTINGS_VARIABLES` that are set and not empty are recorded as
    ``start_settings`` (none recorded when it is None)."""
    info = {
        "format": INFO_FORMAT,
        "instance_id": instance_id,
        "pid": pid,
        "port": port,
        "url": url,
        "repo_root": str(repo_root),
        "settings_file": None if settings_file is None else str(settings_file),
        "log": None if log is None else str(log),
        "started_at": utc_stamp(),
    }
    if environ is not None:
        info["start_settings"] = {name: environ[name] for name in SETTINGS_VARIABLES
                                  if environ.get(name)}
    write_json_atomic(path, info)
    return info


def running_environment(info: Mapping[str, Any],
                        environ: Mapping[str, str]) -> Optional[Dict[str, str]]:
    """The environment the server of `info` reads its settings from: `environ`
    (this start's, which names the same saved settings file, `instance_key`)
    with each of `SETTINGS_VARIABLES` as the server was started with it
    (``start_settings``); None when the info file does not record them."""
    recorded = info.get("start_settings")
    if not isinstance(recorded, dict) or not all(
            isinstance(value, str) for value in recorded.values()):
        return None
    running = {name: value for name, value in environ.items() if name not in SETTINGS_VARIABLES}
    running.update({name: recorded[name] for name in SETTINGS_VARIABLES if recorded.get(name)})
    return running


class Ask(NamedTuple):
    """A setting a start asks for that the running service does not have in
    effect, in words: its plain name, what the running service has (None
    when its info file does not record it), what this start asks for, and the
    variable that asks."""

    name: str
    running: Optional[str]
    asked: str
    variable: str


#: Each asked variable's plain name in a sentence.
_ASKED_NAMES = {ENV_DEPLOYMENT_MODE: "operator controls",
                **{variable: FIELD_NAMES[key] for key, variable in SAVED_VARIABLES.items()}}


def _state(settings: Mapping[str, Any], environ: Mapping[str, str],
           variable: str) -> Tuple[Any, str]:
    """A setting as `environ` reads it by the one resolution (`settings`, its
    readers' answers by variable), and in words: the operator controls on or
    off; a data location's value in effect, else what is set and refused,
    else nothing."""
    setting = settings[variable]
    if variable == ENV_DEPLOYMENT_MODE:
        return setting.available, "on" if setting.available else "off"
    if setting.value is not None and setting.available:
        value = setting.value
        words = (", ".join(str(entry) for entry in value) or "none"
                 if isinstance(value, tuple) else str(value))
        return ("in effect", value), words
    raw = server_settings.configured(variable, environ).raw
    if raw is None:
        return None, "not set"
    if isinstance(raw, str) and variable == server_settings.ENV_TRAINING_EVIDENCE_DIRS:
        raw = [entry for entry in raw.split(os.pathsep) if entry]
    if isinstance(raw, (list, tuple)):
        raw = tuple(str(entry) for entry in raw)
        return ("refused", raw), f"{', '.join(raw)} (refused)"
    return ("refused", raw), f"{raw} (refused)"


def _read(environ: Mapping[str, str]) -> Dict[str, Any]:
    return {setting.variable: setting for setting in server_settings.all_settings(environ)}


def asks_not_met(info: Mapping[str, Any], environ: Mapping[str, str]) -> Tuple[Ask, ...]:
    """What this start asks for (`environ`'s `ASKED_VARIABLES`, set and not
    empty) that the running server of `info` does not have in effect now, in
    `ASKED_VARIABLES` order; empty when it asks for nothing or has it all
    (module docstring). With no ``start_settings`` recorded, every ask."""
    asked = [variable for variable in ASKED_VARIABLES if environ.get(variable)]
    if not asked:
        return ()
    running = running_environment(info, environ)
    here, there = _read(environ), (_read(running) if running is not None else None)
    unmet = []
    for variable in asked:
        want, want_words = _state(here, environ, variable)
        if there is None:
            unmet.append(Ask(_ASKED_NAMES[variable], None, want_words, variable))
            continue
        have, have_words = _state(there, running, variable)
        if have != want:
            unmet.append(Ask(_ASKED_NAMES[variable], have_words, want_words, variable))
    return tuple(unmet)


def ended_without_stop_words(info: Mapping[str, Any]) -> str:
    """What a start says when it finds a previous start's info file with the
    lock free (module docstring): that start's server ended without its
    orderly stop, which removes the file, and so without a stop line in its
    log. Every value it names is the file's, as plain words."""
    pid = info.get("pid")
    when = info.get("started_at")
    said = ", ".join(part for part in (
        f"pid {pid}" if isinstance(pid, int) and not isinstance(pid, bool) else "",
        f"started {when}" if isinstance(when, str) and when else "") if part)
    log = info.get("log")
    where = (f"; its last output is in {log}" if isinstance(log, str) and log else
             "; its last output, if it was kept, says how far it got")
    return ("The previous start of this configuration"
            + (f" ({said})" if said else "")
            + " ended without its orderly stop: it was killed (SIGKILL or the out-of-memory "
              "killer), crashed, was forced to quit, or the machine stopped" + where + ".")


def served_instance_id(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The instance id the launcher gave this server, or None (a server not
    started by the launcher)."""
    env = os.environ if environ is None else environ
    return env.get(ENV_INSTANCE_ID) or None


def remove_own_info(environ: Optional[Mapping[str, str]] = None) -> bool:
    """At the server's shutdown: its info file removed when it still names
    this server's id (a later start's file is never removed). True when it was."""
    env = os.environ if environ is None else environ
    own = served_instance_id(env)
    named = _absolute(env.get(ENV_INSTANCE_FILE))
    if own is None or named is None:
        return False
    info = read_info(named)
    if info is None or info.get("instance_id") != own:
        return False
    try:
        named.unlink()
    except FileNotFoundError:
        return False
    return True


def log_file(state_dir: Path) -> Path:
    return Path(state_dir) / LOGS / LOG_NAME


def error_page(state_dir: Path) -> Path:
    return Path(state_dir) / LOGS / ERROR_PAGE_NAME


def rotate_log(path: Path, limit: int = LOG_ROTATE_BYTES) -> bool:
    """Move the log at `path` to ``<name>.1`` (replacing an older one) when it
    has passed `limit` bytes; True when it was moved."""
    try:
        size = Path(path).stat().st_size
    except OSError:
        return False
    if size <= limit:
        return False
    os.replace(path, Path(path).with_name(Path(path).name + ".1"))
    return True
