"""
Writing the saved deployment settings: save, apply, replace an unreadable file
==============================================================================
UI-06 (the owner's decision of 2026-11-16). The file and its reading are
`src/config/deployment_settings.py`'s; which value is in effect is
`src/config/server_settings.py`'s (an explicit environment variable wins over
the file). Every write is here, and each one:

- takes **the file's lock** first (`deployment.json.lock` beside it, an
  exclusive `src.utils.file_lock` lock), so two pages saving at once cannot
  interleave, and reads the file again under it, never from the cache;
- writes the whole file with **the one atomic writer**
  (`src.config.atomic_json.write_json_atomic`: a staged file, fsync,
  ``os.replace``, the directory flushed; the file 0600, a directory it
  creates 0700);
- **never overwrites a file that cannot be read**: a save or an apply is
  refused while it is unreadable, and `replace_unreadable` first renames it to
  ``deployment.json.unreadable-<UTC stamp>``, kept, and says so. A file that
  cannot be opened (a permission or I/O error) is never renamed: it is fixed
  where it is.

**Save** (`save`) writes the Settings page's choice as the file's `saved`
values, after checking each value by the rule its setting is read by
(`server_settings.check_data_location`); one refused value refuses the whole
save, with a reason per field. It changes nothing in use.

**Apply** (`apply_saved`) is the one step that changes what the deployment
uses: `applied = saved`. It is safe only when nothing could be using the old
locations, so it takes, without waiting, **every GPU lock the old values name**
(the lock in effect for this process, and the one under the old applied
workspaces folder) and writes while it holds them. A job of a command started
by hand holds one; so does this service's CUDA inference (a second open file
in the same process is refused by the lock too). Busy: nothing is written and
the change stays pending, with the reason. A taker that read the old location
just before re-checks it once it holds the lock (`gpu_lock.take_gpu_lock`),
so it never runs under an old lock after an apply. The job runner's own slot
is checked by the caller, which holds the runner's lock around this
(`src/api/services/data_locations.py`); this module does not import the API.

**A location the applying process's own start-up setting sets to another
value is held back** (`held_back`; T03 review, 2026-11-16): its applied value
stays as it was and the change stays pending, applied by a start without that
setting. The service does not use the file's value for it, so applying it
would change only what every other process reads -- a workspaces folder
applied so would give a command run by hand another GPU lock than the
service's job, both held at once. A saved value equal to the start-up
setting is applied as usual: every process then reads the same folder.

Module: src/utils/settings_store.py
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from src.config import deployment_settings, server_settings
from src.config.atomic_json import ensure_directory, write_json_atomic
from src.config.deployment_settings import KEYS, KEY_WORKSPACES_ROOT
from src.utils.file_lock import FileLockBusy, exclusive_lock
from src.utils.gpu_lock import GpuLock, GpuLockBusy

#: How long a write waits for another writer of the file before it refuses.
WRITE_GRACE_SECONDS = 5.0
#: How the GPU lock describes an apply while it holds it, for an instant.
APPLY_JOB_TYPE = "settings"
APPLY_COMMAND = "Settings › Data locations applying a saved change"
#: The words a pending change ends with, wherever it is said.
APPLIES_LATER = "the change applies when it ends: press Apply then, or it applies at the next start"
#: The words a change held back by a start-up setting ends with (`held_back`).
APPLIES_WITHOUT_OVERRIDE = "the saved value applies at a start without that setting"

__all__ = [
    "APPLIES_LATER",
    "APPLIES_WITHOUT_OVERRIDE",
    "ApplyOutcome",
    "RepairOutcome",
    "SaveOutcome",
    "WRITE_GRACE_SECONDS",
    "apply_saved",
    "held_back",
    "held_back_words",
    "replace_unreadable",
    "save",
]


class SaveOutcome(NamedTuple):
    """What a save did. `saved`: the file now holds the values as `saved`.
    `refusals`: the reason per key when a value was refused (nothing saved).
    `message`: one sentence for the page."""

    saved: bool
    message: str
    refusals: Mapping[str, str] = {}


class ApplyOutcome(NamedTuple):
    """What an apply did. `applied`: the saved values are now the applied
    ones, but for those a start-up setting holds back (`held_back`).
    `pending`: a saved change still waits (the reason is `message`); both
    are true when some were applied and others held back."""

    applied: bool
    pending: bool
    message: str


class RepairOutcome(NamedTuple):
    """What `replace_unreadable` did: `replaced`, and the copy `kept`."""

    replaced: bool
    message: str
    kept: Optional[Path] = None


def _place(environ: Optional[Mapping[str, str]]) -> "Path | str":
    """The settings file, or why there is no place for one."""
    location = deployment_settings.settings_location(environ)
    if location.path is not None:
        return location.path
    if location.problem:
        return location.problem
    return ("there is no place for the saved settings: none of SHEPHERD_SETTINGS_FILE, "
            "XDG_CONFIG_HOME, HOME or LOCALAPPDATA names one in this environment")


def _locked(path: Path):
    """The file's lock, its directory created first (0700) when absent."""
    ensure_directory(path.parent)
    return exclusive_lock(deployment_settings.lock_path(path), grace=WRITE_GRACE_SECONDS)


def save(values: Mapping[str, Any], environ: Optional[Mapping[str, str]] = None
         ) -> SaveOutcome:
    """Save `values` (key → value; None or empty clears a key) as the file's
    `saved` values; keys not given keep their saved value. Nothing in use
    changes: `apply_saved` does that.

    Each value is checked first by the rule its setting is read by
    (`server_settings.check_data_location`), and one refusal refuses the whole
    save, with a reason for each refused key. A file that cannot be read is
    never overwritten: the save is refused, naming it."""
    refusals: Dict[str, str] = {}
    normalised: Dict[str, Any] = {}
    for key, value in values.items():
        if key not in KEYS:
            refusals[str(key)] = f"{key!r} is not a data location"
            continue
        try:
            server_settings.check_data_location(key, value)
            normalised[key] = deployment_settings.normalise_value(key, value)
        except ValueError as exc:  # DirectoryRefused is a ValueError
            refusals[key] = str(exc)
    if refusals:
        return SaveOutcome(False, "Nothing was saved: " + "; ".join(
            f"{key}: {reason}" for key, reason in refusals.items()), refusals)
    place = _place(environ)
    if not isinstance(place, Path):
        return SaveOutcome(False, f"Nothing was saved: {place}.")
    try:
        with _locked(place):
            current = deployment_settings.read_file(place)
            if current.problem is not None:
                return SaveOutcome(False, (
                    f"Nothing was saved: {current.problem}. Replace the unreadable file first "
                    "(a copy is kept)."))
            saved = {**current.saved, **normalised}
            write_json_atomic(place, deployment_settings.compose(current.document, saved=saved))
    except FileLockBusy as exc:
        return SaveOutcome(False, f"Nothing was saved: {exc}; try again.")
    except OSError as exc:
        return SaveOutcome(False, f"Nothing was saved: {place} cannot be written ({exc}).")
    return SaveOutcome(True, f"Saved in {place}.")


def _old_lock_files(current: deployment_settings.SavedFile,
                    environ: Optional[Mapping[str, str]]) -> List[Path]:
    """Every GPU lock the values in use name: the one in effect for this
    process (`server_settings.gpu_lock_location`), and the one under the old
    applied workspaces folder when it is usable. Sorted, so two takers take
    them in one order."""
    found = set()
    effective = server_settings.gpu_lock_location(environ)
    if effective.value is not None:
        found.add(Path(effective.value))
    old_root = current.applied.get(KEY_WORKSPACES_ROOT)
    if old_root is not None:
        try:
            root, _note = server_settings.check_data_location(KEY_WORKSPACES_ROOT, old_root)
        except ValueError:
            root = None
        if root is not None:
            found.add(root / server_settings.GPU_LOCK_FILE_NAME)
    return sorted(found, key=str)


def _same_value(key: str, one: Any, other: Any) -> bool:
    """Whether two values of `key` name the same folders, compared as paths
    (``/srv/a/`` is ``/srv/a``), with no I/O; None and no folders are one."""
    if key in deployment_settings.LIST_KEYS:
        return [Path(entry) for entry in one or ()] == [Path(entry) for entry in other or ()]
    if one is None or other is None:
        return one is None and other is None
    return Path(one) == Path(other)


def held_back(current: deployment_settings.SavedFile,
              environ: Optional[Mapping[str, str]] = None) -> Tuple[str, ...]:
    """The pending keys of `current` that a start-up setting of `environ` (the
    process's when None) sets to a value other than the saved one
    (`server_settings.environment_value`): an apply here keeps their applied
    value (module docstring), in `KEYS` order."""
    held = []
    for key in current.pending:
        here = server_settings.environment_value(key, environ)
        if here is not None and not _same_value(key, here, current.saved.get(key)):
            held.append(key)
    return tuple(held)


def held_back_words(keys: Any) -> str:
    """Why `keys` (`held_back`) are not applied here, as a clause."""
    names = [server_settings.FIELD_NAMES[key] for key in keys]
    named = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return (f"a start-up setting sets {named} for this service, so "
            f"{APPLIES_WITHOUT_OVERRIDE}")


def apply_saved(environ: Optional[Mapping[str, str]] = None) -> ApplyOutcome:
    """Make the saved values the applied ones, when nothing could be using
    the old locations, but for those a start-up setting of `environ` holds
    back (`held_back`; module docstring); else leave the change pending and
    say why. The caller holds the job runner's lock with its slot idle
    (`data_locations.apply_saved`). Never raises for a busy lock or an
    unreadable file: the outcome says it."""
    place = _place(environ)
    if not isinstance(place, Path):
        return ApplyOutcome(False, False, f"Nothing is saved: {place}.")
    if not place.exists():
        return ApplyOutcome(False, False, "Nothing is saved, so nothing was applied.")
    locks: List[GpuLock] = []
    held: Tuple[str, ...] = ()
    try:
        with _locked(place):
            current = deployment_settings.read_file(place)
            if current.problem is not None:
                return ApplyOutcome(False, False, f"Nothing was applied: {current.problem}.")
            if not current.pending:
                return ApplyOutcome(False, False, "The saved settings are the ones in use.")
            held = held_back(current, environ)
            if len(held) == len(current.pending):
                return ApplyOutcome(False, True,
                                    f"Saved, not applied yet: {held_back_words(held)}.")
            applied = {key: current.applied.get(key) if key in held else current.saved.get(key)
                       for key in KEYS}
            for lock_file in _old_lock_files(current, environ):
                try:
                    locks.append(GpuLock(lock_file, job_type=APPLY_JOB_TYPE,
                                         command=APPLY_COMMAND).acquire(grace=0))
                except GpuLockBusy as exc:
                    return ApplyOutcome(False, True, (
                        f"Saved, not applied yet: the GPU lock {exc.path} is held by "
                        f"{exc.holder}, so {APPLIES_LATER}."))
                except OSError as exc:
                    return ApplyOutcome(False, True, (
                        f"Saved, not applied yet: the GPU lock {lock_file} cannot be checked "
                        f"({exc}), so {APPLIES_LATER}."))
            write_json_atomic(place, deployment_settings.compose(
                current.document, applied=applied))
    except FileLockBusy as exc:
        return ApplyOutcome(False, True, f"Saved, not applied yet: {exc}; press Apply again.")
    except OSError as exc:
        return ApplyOutcome(False, True,
                            f"Saved, not applied yet: {place} cannot be written ({exc}).")
    finally:
        for lock in reversed(locks):
            lock.release()
    if held:
        return ApplyOutcome(True, True, ("Applied: the saved data locations are now in use, "
                                         f"but not all: {held_back_words(held)}."))
    return ApplyOutcome(True, False, "Applied: the saved data locations are now in use.")


def _kept_name(path: Path, moment: datetime) -> Path:
    """An unused name for the unreadable file's copy beside it."""
    stamp = moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = path.with_name(f"{path.name}{deployment_settings.UNREADABLE_SUFFIX}{stamp}")
    candidate, number = base, 1
    while candidate.exists() or candidate.is_symlink():
        number += 1
        candidate = base.with_name(f"{base.name}-{number}")
    return candidate


def replace_unreadable(environ: Optional[Mapping[str, str]] = None) -> RepairOutcome:
    """Move a saved settings file whose content cannot be read out of the way,
    keeping it: renamed to ``deployment.json.unreadable-<UTC stamp>`` beside
    it, so nothing is saved until Settings saves again. Refused for a file
    that can be read, and for one that cannot be opened (a permission or I/O
    error), which is fixed where it is, never renamed."""
    place = _place(environ)
    if not isinstance(place, Path):
        return RepairOutcome(False, f"Nothing was replaced: {place}.")
    try:
        with _locked(place):
            current = deployment_settings.read_file(place)
            if current.problem is None:
                return RepairOutcome(False, "The saved settings file can be read; nothing was "
                                            "replaced.")
            if not current.repairable:
                return RepairOutcome(False, (
                    f"Nothing was replaced: {current.problem}. A file that cannot be opened is "
                    "fixed where it is (its permissions, or the disk), never renamed."))
            kept = _kept_name(place, datetime.now(timezone.utc))
            os.rename(place, kept)
    except FileLockBusy as exc:
        return RepairOutcome(False, f"Nothing was replaced: {exc}; try again.")
    except OSError as exc:
        return RepairOutcome(False, f"Nothing was replaced: {place} cannot be renamed ({exc}).")
    return RepairOutcome(True, (f"The unreadable settings file was renamed to {kept} and kept. "
                                "Nothing is saved now: choose the data locations again."), kept)
