"""
The data locations as one view: in effect, from where, saved, pending, why
==========================================================================
UI-06 (the owner's decision of 2026-11-16). Settings › Data locations, the
read-only route ``GET /api/v1/settings/deployment`` and
``scripts/deployment_settings.py show`` say the same thing about each data
location, so the dashboard, a shell and a job the service starts can be
compared. This module is that one view; it reads nothing of its own:

- what is **in effect**, and its **source**, is `src.config.server_settings`'
  (its readers, through their one read point: an explicit environment
  variable, else the saved file's applied value, else nothing), checked by the
  same rules a job's read checks it by;
- what is **configured** (the value set, before its rule checks it: a refused
  value is shown with its reason, never as "not set") is the same read point's
  (`server_settings.configured`);
- what is **saved** and **pending** (saved in Settings, not applied yet) is the
  saved file's (`src.config.deployment_settings.read_saved`);
- whether a pending change can be applied now is read from the GPU lock
  (`src.utils.gpu_lock.lock_state`); Apply itself decides, with the job
  runner's slot too (`src/api/services/data_locations.py`).

**Overridden**: a location the environment sets while the saved file holds a
value of its own. The environment's value is in effect; the saved one applies
once the start-time setting is removed (Syncthing's ``guiAddressOverridden``).
Nothing ever writes an environment value into the file, and a saved value
that differs from this process's start-up setting is not applied here either
(`held`, `settings_store.held_back`): it would change only what every other
process reads.

**Paths shown or hidden** (`as_dict`): every path, and every text that may
name one, is shown for a request D8 allows, and hidden for any other: a
read-only viewer learns what is set and from where, never where. A value
that is set and refused, or a settings file named at start and refused, is
said in fixed words for that viewer (`problem_words`, `file_problem_words`),
never in its refusal's, which quote the value as it was given: the one
redaction (`src.utils.path_redaction.redact_paths`, ``a server path``) finds
an absolute path only, so a relative or ``~`` value would be shown (T03
review, 2026-11-16). Every other text goes through that redaction
(`hide_paths`).

Gradio-free and torch-free.

Module: src/utils/deployment_view.py
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from src.config import deployment_settings, server_settings
from src.config.deployment_settings import (
    KEY_INTAKE_DIR,
    KEY_REGISTRY,
    KEY_TRAINING_EVIDENCE_DIRS,
    KEY_WORKSPACES_ROOT,
    KEYS,
)
from src.utils.path_redaction import redact_paths

#: What a hidden path reads as (the dashboard's words for one).
HIDDEN_PATH = "a server path"

#: Each data location's plain name and what it is for (who needs it), in the
#: order Settings shows them. The variable names appear only beside them.
FIELD_WORDS: Dict[str, Tuple[str, str]] = {
    KEY_WORKSPACES_ROOT: (
        "Workspaces folder",
        "Datasets, runs and saved models. Needed by every Research page, model loading and "
        "GPU jobs."),
    KEY_REGISTRY: (
        "Results registry",
        "Cohort versions and registered reports. Needed by Test to measure a cohort and "
        "register, and by Diagnosis' reference cohorts."),
    KEY_INTAKE_DIR: (
        "Import folder",
        "Case files you place here to import as a cohort version (Test › Import)."),
    KEY_TRAINING_EVIDENCE_DIRS: (
        "Training evidence folders",
        "Optional. More folders the training/test overlap check (R10) reads."),
}

#: Each data location's reader (`server_settings`), by its key in the file.
_READERS = {
    KEY_WORKSPACES_ROOT: server_settings.workspaces_root,
    KEY_REGISTRY: server_settings.cohort_registry,
    KEY_INTAKE_DIR: server_settings.intake_directory,
    KEY_TRAINING_EVIDENCE_DIRS: server_settings.training_evidence_dirs,
}

#: The one optional location: unset is usable (an empty list).
OPTIONAL_KEYS = (KEY_TRAINING_EVIDENCE_DIRS,)

__all__ = [
    "DeploymentView",
    "FIELD_WORDS",
    "FieldView",
    "FileView",
    "HIDDEN_PATH",
    "OPTIONAL_KEYS",
    "as_dict",
    "deployment_view",
    "file_problem_words",
    "hide_paths",
    "problem_words",
    "text_lines",
]


class FieldView(NamedTuple):
    """One data location as read now.

    `value`: what is in effect (a resolved path as text; a tuple of them for
    the evidence folders; None when nothing usable is in effect).
    `source`: where the value read came from (`server_settings.SOURCE_*`;
    None when nothing is set or the saved file cannot be read).
    `configured`: the value that source sets, before its rule checked it.
    `applied` / `saved`: the saved file's values (None when it holds none).
    `pending`: saved in Settings and not applied yet. `overridden`: the
    environment sets it while the file holds a value. `problem`: why its
    feature is off; `note`: a remark that turns nothing off; `unset`:
    nothing is set at all."""

    key: str
    name: str
    purpose: str
    variable: str
    value: Any
    source: Optional[str]
    configured: Any
    applied: Any
    saved: Any
    pending: bool
    overridden: bool
    problem: Optional[str]
    note: Optional[str]
    unset: bool
    optional: bool


class FileView(NamedTuple):
    """The saved settings file: where it is (`path`, None with no place for
    one), named by `SHEPHERD_SETTINGS_FILE` (`explicit`), whether it is there,
    why it cannot be read (`problem`) and whether that is its content
    (`repairable`: Settings may rename it away, keeping it)."""

    path: Optional[str]
    explicit: bool
    exists: bool
    problem: Optional[str]
    repairable: bool
    applied_at: Optional[str]
    saved_at: Optional[str]


class DeploymentView(NamedTuple):
    """Every data location, the saved file, the deployment mode and the GPU
    lock, as read now. `pending`: the keys saved and not applied yet, and
    `pending_reason` why (None when nothing is pending). `held`: those of
    them a start-up setting of this process holds back
    (`settings_store.held_back`): an Apply here does not apply them, a start
    without that setting does."""

    fields: Tuple[FieldView, ...]
    file: FileView
    mode: Any
    gpu_lock: Any
    pending: Tuple[str, ...]
    pending_reason: Optional[str]
    held: Tuple[str, ...] = ()

    @property
    def applicable(self) -> Tuple[str, ...]:
        """The pending keys an Apply here would apply (`pending` less `held`)."""
        return tuple(key for key in self.pending if key not in self.held)

    def field(self, key: str) -> FieldView:
        return next(item for item in self.fields if item.key == key)

    @property
    def nothing_set(self) -> bool:
        """No required location is set anywhere (a first start)."""
        return all(item.unset for item in self.fields if not item.optional)


def _text(value: Any) -> Any:
    """A path, or a tuple of them, as text."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(str(entry) for entry in value) or None
    return str(value)


def _pending_reason(environ: Optional[Mapping[str, str]], pending: Tuple[str, ...],
                    held: Tuple[str, ...]) -> str:
    """Why a saved change is not applied yet: for one a start-up setting
    holds back, that setting (`settings_store.held_back_words`); for any
    other, as far as the GPU lock tells, a holder of the lock in use (a job,
    a command started by hand, this service's CUDA inference), else nothing
    but Apply. Apply checks the job runner's slot too, and says so."""
    from src.utils.gpu_lock import lock_state
    from src.utils.settings_store import APPLIES_LATER, held_back_words

    reasons = []
    if any(key not in held for key in pending):
        state = lock_state(environ)
        if state.get("held_by"):
            reasons.append(f"The GPU lock is held by {state['held_by']}, so {APPLIES_LATER}.")
        else:
            reasons.append("Saved, not applied yet: press Apply, or it applies at the next start.")
    if held:
        words = held_back_words(held)
        reasons.append(f"Not applied here: {words}." if reasons else
                       f"Saved, not applied yet: {words}.")
    return " ".join(reasons)


def deployment_view(environ: Optional[Mapping[str, str]] = None) -> DeploymentView:
    """Every data location as `environ` (the process's when None) reads it
    now (module docstring)."""
    from src.utils.settings_store import held_back

    saved_file = deployment_settings.read_saved(environ)
    location = deployment_settings.settings_location(environ)
    fields = []
    for key in KEYS:
        variable = server_settings.SAVED_VARIABLES[key]
        setting = _READERS[key](environ)
        found = server_settings.configured(variable, environ)
        applied = saved_file.applied.get(key)
        saved = saved_file.saved.get(key)
        name, purpose = FIELD_WORDS[key]
        fields.append(FieldView(
            key=key, name=name, purpose=purpose, variable=variable,
            value=_text(setting.value) if setting.available else None,
            source=setting.source if setting.source else found.source,
            configured=_text(found.raw.split(os.pathsep)
                             if key == KEY_TRAINING_EVIDENCE_DIRS and isinstance(found.raw, str)
                             else found.raw),
            applied=_text(applied), saved=_text(saved),
            pending=key in saved_file.pending,
            overridden=(found.source == server_settings.SOURCE_ENVIRONMENT
                        and (applied is not None or saved is not None)),
            problem=setting.problem, note=setting.note, unset=setting.unset,
            optional=key in OPTIONAL_KEYS))
    pending = tuple(saved_file.pending) if saved_file.problem is None else ()
    held = held_back(saved_file, environ) if pending else ()
    return DeploymentView(
        fields=tuple(fields),
        file=FileView(
            path=str(saved_file.path) if saved_file.path is not None else None,
            explicit=location.explicit, exists=saved_file.exists,
            problem=saved_file.problem, repairable=saved_file.repairable,
            applied_at=saved_file.applied_at, saved_at=saved_file.saved_at),
        mode=server_settings.deployment_mode(environ),
        gpu_lock=server_settings.gpu_lock_location(environ),
        pending=pending,
        pending_reason=_pending_reason(environ, pending, held) if pending else None,
        held=held,
    )


def hide_paths(text: Any) -> Optional[str]:
    """`text` with every absolute path replaced (`HIDDEN_PATH`), for a viewer
    D8 does not allow; None stays None. Only an absolute path is found, so a
    text that may quote a value as it was given (a refusal) is never passed
    here: `problem_words` and `file_problem_words` say it in fixed words."""
    if text is None:
        return None
    return redact_paths(text, [], other=HIDDEN_PATH)


#: Where an operator reads why, in the fixed words a hidden viewer is given.
OPERATOR_SEES_WHY = f"an operator sees why in {server_settings.DATA_LOCATIONS}"
#: Where a refused value was set, by its source, in those words.
_SET_WHERE = {
    server_settings.SOURCE_ENVIRONMENT: "set when the service started",
    server_settings.SOURCE_SAVED: "saved in Settings",
}


def problem_words(item: FieldView, *, paths_shown: bool) -> Optional[str]:
    """A location's problem as a viewer may read it. Shown: its own words.
    Hidden: a value that is set and refused, or one that would come from a
    saved file that cannot be used, in fixed words naming only the location
    and its source, never the refusal's (which quote the value as given, a
    relative or ``~`` path included); the unset words, which name no value,
    with every path hidden (`hide_paths`)."""
    if paths_shown or item.problem is None:
        return item.problem
    if item.unset:
        return hide_paths(item.problem)
    verb = "are" if item.key == KEY_TRAINING_EVIDENCE_DIRS else "is"
    where = _SET_WHERE.get(item.source)
    if where is not None:
        return f"{item.name} {where} {verb} refused: {OPERATOR_SEES_WHY}."
    return (f"{item.name} {verb} not known: the saved settings file cannot be used, and "
            f"{OPERATOR_SEES_WHY}.")


def file_problem_words(file: FileView, *, paths_shown: bool) -> Optional[str]:
    """The saved settings file's problem as a viewer may read it. Hidden: a
    file named at start and refused (`SHEPHERD_SETTINGS_FILE` relative, whose
    words quote it) in fixed words; a file that cannot be read with its path
    hidden (`hide_paths`; its reason names no value)."""
    if paths_shown or file.problem is None:
        return file.problem
    if file.path is None:
        return f"The settings file named when the service started is refused: {OPERATOR_SEES_WHY}."
    return hide_paths(file.problem)


def _lock_problem_words(lock: Any, *, paths_shown: bool) -> Optional[str]:
    """The GPU lock's problem as a viewer may read it: hidden, a refused or
    unknown workspaces folder in fixed words (its words are the folder's
    refusal), as the GPU chip says it (`src.webui.components.chips`)."""
    if paths_shown or lock.problem is None:
        return lock.problem
    if lock.unset:
        return hide_paths(lock.problem)
    return ("The GPU lock cannot be used, so every managed GPU job is refused until the "
            f"workspaces folder is corrected: {OPERATOR_SEES_WHY}.")


def _shown(value: Any, paths_shown: bool) -> Any:
    if paths_shown or value is None:
        return list(value) if isinstance(value, tuple) else value
    return None


def as_dict(view: DeploymentView, *, paths_shown: bool) -> Dict[str, Any]:
    """The view as plain data (the route's answer, the CLI's ``--json``).
    With `paths_shown` False every path is None, a refused value is said in
    fixed words (`problem_words`, `file_problem_words`) and every other text
    that may name a path is redacted (`hide_paths`); what is set, from where,
    and whether it is pending stay."""
    def words(text: Any) -> Optional[str]:
        return text if paths_shown else hide_paths(text)

    fields: List[Dict[str, Any]] = []
    for item in view.fields:
        fields.append({
            "key": item.key, "name": item.name, "variable": item.variable,
            "set": item.value is not None or item.configured is not None,
            "in_effect": _shown(item.value, paths_shown),
            "source": item.source,
            "configured": _shown(item.configured, paths_shown),
            "saved": _shown(item.saved, paths_shown),
            "applied": _shown(item.applied, paths_shown),
            "pending": item.pending, "overridden": item.overridden,
            "problem": problem_words(item, paths_shown=paths_shown),
            "note": words(item.note),
            "unset": item.unset, "optional": item.optional,
        })
    return {
        "paths_shown": paths_shown,
        "fields": fields,
        "settings_file": {
            "path": view.file.path if paths_shown else None,
            "explicit": view.file.explicit, "exists": view.file.exists,
            "problem": file_problem_words(view.file, paths_shown=paths_shown),
            "repairable": view.file.repairable,
            "applied_at": view.file.applied_at, "saved_at": view.file.saved_at,
        },
        "deployment_mode": {"value": view.mode.value, "source": view.mode.source,
                            "problem": view.mode.problem},
        "gpu_lock": {"path": str(view.gpu_lock.value) if paths_shown and view.gpu_lock.value
                     else None,
                     "configured": view.gpu_lock.value is not None,
                     "problem": _lock_problem_words(view.gpu_lock, paths_shown=paths_shown)},
        "pending": list(view.pending),
        "held_back": list(view.held),
        "pending_reason": words(view.pending_reason),
    }


#: How a source reads in a shell (`text_lines`): the variable may be named.
_SOURCE_TEXT = {
    server_settings.SOURCE_ENVIRONMENT: "environment (a start-time setting; it wins)",
    server_settings.SOURCE_SAVED: "saved settings file (applied)",
    server_settings.SOURCE_LOCAL_DEFAULT: "the local start's default",
    None: "not set",
}


def _value_text(value: Any) -> str:
    if value is None:
        return "not set"
    if isinstance(value, (tuple, list)):
        return ", ".join(value) if value else "none"
    return str(value)


def text_lines(view: DeploymentView) -> List[str]:
    """The view for a shell (`scripts/deployment_settings.py show`): paths
    shown, the variable of each location named, one fact per line."""
    lines = []
    file = view.file
    lines.append(f"Settings file: {file.path or 'none (no place for one in this environment)'}"
                 + (" (named by SHEPHERD_SETTINGS_FILE)" if file.explicit else ""))
    if file.problem:
        lines.append(f"  problem: {file.problem}")
    elif not file.exists:
        lines.append("  not there: nothing is saved")
    mode = view.mode
    lines.append(f"Deployment mode: {mode.value or 'read-only'} "
                 f"(source: {_SOURCE_TEXT.get(mode.source, mode.source)})")
    for item in view.fields:
        lines.append("")
        lines.append(f"{item.name} ({item.variable}, saved as {item.key})")
        lines.append("  in effect: " + ("none" if item.optional and item.value is None
                                            and item.problem is None
                                            else _value_text(item.value)))
        lines.append(f"  source:    {_SOURCE_TEXT.get(item.source, item.source)}")
        if item.configured is not None and item.value is None:
            lines.append(f"  set to:    {_value_text(item.configured)} (refused)")
        if item.saved is not None or item.applied is not None:
            lines.append(f"  saved:     {_value_text(item.saved)}"
                         + (" (pending: not applied yet)" if item.pending else ""))
        if item.overridden:
            lines.append("  the environment overrides the saved value; it applies once "
                         f"{item.variable} is removed")
        if item.problem:
            lines.append(f"  problem:   {item.problem}")
        if item.note:
            lines.append(f"  note:      {item.note}")
    lines.append("")
    lock = view.gpu_lock
    lines.append(f"GPU lock: {lock.value if lock.value is not None else 'not configured'}")
    if view.pending_reason:
        lines.append(f"Pending: {', '.join(view.pending)}. {view.pending_reason}")
    return lines
