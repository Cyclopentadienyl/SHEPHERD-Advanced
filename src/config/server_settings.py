"""
Server settings: what an operator configures for M5's controls and lists.
=========================================================================
`docs/working/PLAN_TEST_RESULTS.md` §6 ("M5's boundaries"), D7, D8 and §4.9
question 3; SPEC_4 §3 (deployment mode M1); UI-06 (the owner's decision of
2026-11-16). Each reader returns a `Setting`: the value in effect and where it
came from (`source`), or why the feature that needs it is off.

**Where a setting comes from** (since 2026-11-16; one resolution for every
reader, `_resolve`, read at each use and never from a request):

1. an environment variable of the process that is set and not empty: source
   ``environment``, a start-time override, which always wins;
2. else the **applied** value of the saved settings file
   (`src/config/deployment_settings.py`: chosen in Settings › Data locations,
   kept outside the repository and read again at the next start): source
   ``saved``;
3. else, for the deployment mode only, the local start's default (below):
   source ``local-default``;
4. else nothing: the feature is off with a stated reason, never a default path.

So the service, a command run by hand by the same account and a job the
service starts read the same values, from the same file, with no environment
prepared first; nothing writes a resolved value into an environment, so a
restart reads the file again. *History: from M5-1 until 2026-11-16 these
settings came only from the environment ("no route, form or saved UI file can
set one"). The owner withdrew that rule on 2026-11-16 because it made a start
by double click unusable; an environment-configured deployment keeps working
unchanged, since an explicit variable still wins.*

Every data location is shown in one view -- in effect, its source, the saved
value, pending or not, its problem -- by Settings › Data locations,
``GET /api/v1/settings/deployment`` and ``scripts/deployment_settings.py
show`` (`src/utils/deployment_view.py`), which read through the readers here
and `configured` (this read point's raw answer), never a reader of their own.

| Variable | Saved as | What it is | Unset |
|---|---|---|---|
| `SHEPHERD_DEPLOYMENT_MODE` | not saved | SPEC_4's deployment mode; only the exact value `M1` enables operator controls, which D8's check then allows request by request (below). Unset, the local start's default applies | read-only |
| `SHEPHERD_WORKSPACES_ROOT` | `workspaces_root` | the one workspace root: its subdirectories are the workspaces the server lists (SPEC_4 M1's "one explicit workspace namespace"), and the GPU lock is `.shepherd-gpu.lock` directly under it | no workspace listed; no GPU lock, so every managed GPU job is refused |
| `SHEPHERD_REGISTRY` | `registry` | the cohort registry (the `--registry` of the CLI), the only one reports are registered from and cohort versions measured in (`deployment_registry_refusal`) | no cohort feature; every registration and `--cohort` measurement refused |
| `SHEPHERD_INTAKE_DIR` | `intake_dir` | where an operator places import sources | no import source |
| `SHEPHERD_TRAINING_EVIDENCE_DIRS` | `training_evidence_dirs` (a list) | R10 evidence directories, `os.pathsep`-separated in the environment (optional); one that cannot be checked is kept and noted | none |

**A saved value is checked like an environment value**, by the same
functions at each read (`check_operator_directory`, `check_registry_root`,
`is_directory`), and a value to be saved is checked by them before it is
(`check_data_location`). A saved settings file that cannot be read is a
problem for every setting that would come from it, naming the file; it is
never read as empty.

**The GPU lock's location is read here too** (`gpu_lock_location`, the one
rule `src.utils.gpu_lock` applies). **The workspaces root is its one name**:
the lock is `.shepherd-gpu.lock` under the workspaces root in effect, and no
other setting names one (root-cause round, P24-S1 and its verification). With
no root, every managed GPU job is refused (review of 8242b47, P2-4). A saved
root changes only when nothing could be using the old lock, and never while
the applying service's own start-up setting names another root
(`src/utils/settings_store.py`, `apply_saved`).

**A missing setting turns its feature off with a stated reason; nothing falls
back to a default path**, inside the repository or anywhere else. The
repository's `data/workspaces/` is a workspace root only when the operator names
it. The reasons point to Settings › Data locations in plain words; the
variable follows as "Ref:" for a start from the command line (UI-06, item 9).

**One rule for a directory an operator names** (`check_operator_directory`):
absolute, resolved, an existing directory, and for the two that hold
patient-level data (the registry and the intake directory) outside the
repository after `resolve()`, so a symlink into the tree is caught. The registry
rule is `check_registry_root`, and `src.evaluation.cohort_registry.
validate_registry` applies the same function, so `--registry` on the command line
and the service's registry are one rule. It lives here, below
`src.evaluation`, because the D8 check and the GPU lock in `src.utils` read these
settings too.

**Whether a named path is a directory is one explicit stat** (`is_directory`):
it is one; it is not and cannot be one (absent, not a directory, a link loop, a
name too long to exist, a path that cannot be encoded); or its state cannot be
learned (a parent the account may not enter, an I/O error), which is neither.
`check_operator_directory` refuses the last as `DirectoryUncheckable`, a
`DirectoryRefused`, so a required directory is refused for it as before. Only
the optional evidence setting keeps such an entry (`training_evidence_dirs`),
and R10 applies the same `is_directory` to `--training-evidence`, so the Test
page and the command line read a directory alike.

**The deployment mode** is never saved. An explicit `SHEPHERD_DEPLOYMENT_MODE`
is read as before: exactly `M1` enables, anything else is read-only and never
upgraded. Unset, **the local start's default** gives `M1` (source
``local-default``) only when `SHEPHERD_LOCAL_ENTRY` is exactly ``1`` and the
host passed to startup (`SHEPHERD_BIND_HOST`) is a loopback bind: the desktop
entry and `launch_shepherd.sh` start the launcher with ``--local-entry``, and
the launcher sets that variable only for a loopback bind (it is the
launcher's, like `SHEPHERD_BIND_HOST`). A uvicorn started by hand passes no
bind host and is not the local start.

**What the deployment mode does not do.** `M1` is one of D8's conditions, not an
authorisation: the check on each operator control (loopback arrival, no
forwarding header, a loopback Host) is D8's, applied where the request is
handled, for the local default as for an explicit `M1`; no route skips it.
Setting `M1` behind a reverse proxy or on a shared machine does not make
those controls safe (SPEC_4 §3: loopback is not authorisation).

**Where it is applied.** `deployment_mode()` is read by D8's check
(`src/utils/operator_access.py`) on every operator control, at the moment the
request is handled: without `M1`, every control is refused and the deployment
is read-only, whatever arrives on loopback.

**The network bind** (SPEC_4 §3.1, B1): the service binds `127.0.0.1` by
default, and a bind to any other address is refused unless
`SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND=1` (`network_bind`). The flag
records **risk acceptance, not a control**: it authenticates nobody, and it
never lifts D8's checks, so an operator control stays refused for every request
that does not arrive on loopback. It is read from the environment only, never
from the saved file. §3.1's three enforcement points apply that one
rule: the launcher and `src.api.main:main()` before they start anything, and
application startup (`startup_bind`) on the host those two pass in as
`SHEPHERD_BIND_HOST`, which they set to the address they bind every time (it is
theirs, not an operator's setting). A server handed a listening socket (uvicorn
``--fd``) has an address nothing here reads, so it counts as a network bind
(`inherited_socket_bind`).

Platform: nothing here is POSIX-only; `os.pathsep` is `;` on Windows.

Module: src/config/server_settings.py
"""
from __future__ import annotations

import errno
import ipaddress
import os
import stat
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Optional, Tuple

from src.config import deployment_settings
from src.config.deployment_settings import (
    KEY_INTAKE_DIR,
    KEY_REGISTRY,
    KEY_TRAINING_EVIDENCE_DIRS,
    KEY_WORKSPACES_ROOT,
)

#: The repository this module is part of. Computed from this file, as
#: `src/config/runtime_presets.py` does, and compared after `resolve()`.
REPO_ROOT = Path(__file__).resolve().parents[2]

ENV_DEPLOYMENT_MODE = "SHEPHERD_DEPLOYMENT_MODE"
ENV_WORKSPACES_ROOT = "SHEPHERD_WORKSPACES_ROOT"
ENV_REGISTRY = "SHEPHERD_REGISTRY"
ENV_INTAKE_DIR = "SHEPHERD_INTAKE_DIR"
ENV_TRAINING_EVIDENCE_DIRS = "SHEPHERD_TRAINING_EVIDENCE_DIRS"
#: The launcher's mark of the local start (`deployment_mode`): set by it, to
#: `LOCAL_ENTRY`, only for ``--local-entry`` and a loopback bind; an operator
#: does not set it.
ENV_LOCAL_ENTRY = "SHEPHERD_LOCAL_ENTRY"
LOCAL_ENTRY = "1"

#: Where a setting in effect came from (`Setting.source`): an environment
#: variable of the process (a start-time override), the saved settings file's
#: applied values, or the local start's default mode. None: nothing is set.
SOURCE_ENVIRONMENT = "environment"
SOURCE_SAVED = "saved"
SOURCE_LOCAL_DEFAULT = "local-default"
#: Where a clinician chooses a data location, as the reasons name it.
DATA_LOCATIONS = "Settings › Data locations"
#: The GPU lock's file name directly under the workspaces root, its one place
#: (`gpu_lock_location`).
GPU_LOCK_FILE_NAME = ".shepherd-gpu.lock"
ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND = "SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND"
#: The bind address the launcher and `src.api.main:main()` pass to application
#: startup (`startup_bind`). They set it to the host they bind, every time, and
#: remove it when they bind no host; an operator does not set it.
ENV_BIND_HOST = "SHEPHERD_BIND_HOST"

#: The address the service binds unless another is configured (SPEC_4 B1).
DEFAULT_BIND_HOST = "127.0.0.1"

#: The one value of `ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND` that accepts the
#: risk; compared exactly, like the deployment mode.
ACCEPTS_NETWORK_BIND = "1"

#: The one deployment mode that enables operator controls (§4.9 question 3).
#: Compared exactly: `m1`, ` M1` and `M1 ` are other values, never inferred.
OPERATOR_DEPLOYMENT_MODE = "M1"

#: What the two patient-level directories hold, in their refusals.
PATIENT_LEVEL_DATA = "patient-level data"

__all__ = [
    "ACCEPTS_NETWORK_BIND",
    "BindDecision",
    "CHOOSE_IN_DATA_LOCATIONS",
    "CHOOSE_THEM_IN_DATA_LOCATIONS",
    "CREATE_SUGGESTED",
    "DATA_LOCATIONS_TARGET",
    "DEFAULT_BIND_HOST",
    "DirectoryRefused",
    "DirectoryUncheckable",
    "ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND",
    "ENV_BIND_HOST",
    "ENV_DEPLOYMENT_MODE",
    "DATA_LOCATIONS",
    "ENV_INTAKE_DIR",
    "ENV_LOCAL_ENTRY",
    "ENV_REGISTRY",
    "ENV_TRAINING_EVIDENCE_DIRS",
    "ENV_WORKSPACES_ROOT",
    "FIELD_NAMES",
    "GPU_LOCK_FILE_NAME",
    "GPU_LOCK_GUIDE",
    "GPU_LOCK_NOT_CONFIGURED",
    "GPU_LOCK_NOTICE",
    "IMPORT_FOLDER",
    "INTAKE_DIR_NOTICE",
    "INTAKE_DIR_NOT_SET",
    "INTAKE_FORMAT",
    "LOCAL_ENTRY",
    "NO_IMPORT_FOLDER",
    "NO_RESULTS_REGISTRY",
    "NO_WORKSPACES_FOLDER",
    "Notice",
    "OPEN_DATA_LOCATIONS",
    "OUTSIDE_REPOSITORY",
    "PLACE_ACTIONS",
    "MODE_OFF_STATE",
    "OPERATOR_DEPLOYMENT_MODE",
    "REGISTRY_NOTICE",
    "RESULTS_REGISTRY",
    "SEVERAL_ACTIONS",
    "REGISTRY_NOT_SET",
    "REGISTRY_OPTION",
    "SAVED_VARIABLES",
    "SOURCE_ENVIRONMENT",
    "SOURCE_LOCAL_DEFAULT",
    "SOURCE_SAVED",
    "SUGGESTED_WORKSPACES",
    "Setting",
    "WORKSPACES_FOLDER",
    "WORKSPACES_ROOT_NOTICE",
    "WORKSPACES_ROOT_NOT_SET",
    "all_settings",
    "check_data_location",
    "check_operator_directory",
    "bind_is_loopback",
    "check_registry_root",
    "cohort_registry",
    "configured",
    "deployment_mode",
    "deployment_registry_refusal",
    "environment_value",
    "gpu_lock_location",
    "inherited_socket_bind",
    "intake_directory",
    "is_directory",
    "is_loopback_ip",
    "mode_off_why",
    "network_bind",
    "startup_bind",
    "training_evidence_dirs",
    "workspaces_root",
]


class DirectoryRefused(ValueError):
    """A directory an operator named is not one this rule accepts."""


class DirectoryUncheckable(DirectoryRefused):
    """A directory an operator named whose state cannot be learned: its stat or
    its resolve failed for a reason that does not say it is absent or cannot
    exist (a parent the account may not enter, an I/O error). It is neither
    shown to exist nor shown absent.

    A `DirectoryRefused`, so every required directory is refused for it as
    before. `training_evidence_dirs` alone keeps such an entry, as a place R10
    could not look in. `path` is the path checked: resolved when its resolve
    worked, else as given. `reason` is "<ErrorType>: <strerror>"."""

    def __init__(self, message: str, *, path: Path, reason: str) -> None:
        super().__init__(message)
        self.path = path
        self.reason = reason


#: The stat errors that say a path is not a directory and cannot be one: it is
#: not there, or not reachable as one (ENOENT, ENOTDIR, EBADF, ELOOP, as Python
#: 3.12's `Path.is_dir` reads them), or its name is too long to exist. Any
#: other error leaves its state unknown.
_NOT_A_DIRECTORY_ERRNOS = frozenset({
    errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP, errno.ENAMETOOLONG,
})
#: The Windows errors Python 3.12's `Path.is_dir` reads the same way: not
#: ready, an invalid name, a link that cannot be resolved.
_NOT_A_DIRECTORY_WINERRORS = frozenset({21, 123, 1921})


def is_directory(path: Any) -> bool:
    """Whether `path` is a directory, links followed, from one explicit stat.

    True: it is one. False: it is not and cannot be one -- absent, not a
    directory, a link loop, a name too long to exist, or a path the platform
    cannot encode.

    Explicit, because `Path.is_dir` does not keep one rule: from Python 3.14
    it answers False for every `OSError`, so a directory whose state is unknown
    would read as one that is not there.

    Raises:
        OSError: the state cannot be learned (any other stat error: a parent
            the account may not enter, an I/O error). The caller decides what
            that means for its directory; it is neither True nor False.
    """
    try:
        mode = Path(path).stat().st_mode
    except OSError as exc:
        if _cannot_be_a_directory(exc):
            return False
        raise
    except ValueError:  # an embedded null byte, or a path the platform cannot encode
        return False
    return stat.S_ISDIR(mode)


def _cannot_be_a_directory(exc: OSError) -> bool:
    """Whether a stat or resolve error says the path is not a directory and
    cannot be one, rather than leaving its state unknown."""
    return (exc.errno in _NOT_A_DIRECTORY_ERRNOS
            or getattr(exc, "winerror", None) in _NOT_A_DIRECTORY_WINERRORS)


def _error_reason(exc: OSError) -> str:
    """"<ErrorType>: <strerror>", as a refusal names the error."""
    return f"{type(exc).__name__}: {exc.strerror or exc}"


class Setting(NamedTuple):
    """One setting as read now.

    `value` is what is in effect: a string, a resolved `Path`, or a tuple; `None`
    when the setting is unusable. `problem` is why the feature that needs it is
    off, in words an operator can act on; `None` when it is usable. `note` is a
    remark that turns nothing off (an evidence directory kept although it
    cannot be checked, say). `unset`
    says the problem is that nothing is set at all, not a value that is set
    and refused; a caller tells the two apart by it, never by the words.
    `source` is where the value read came from (`SOURCE_ENVIRONMENT`,
    `SOURCE_SAVED`, `SOURCE_LOCAL_DEFAULT`), for a refused value too; None
    when nothing is set or the saved file that would hold it cannot be read.
    `variable` is the setting's environment variable whatever the source.
    """

    variable: str
    value: Any
    problem: Optional[str] = None
    note: Optional[str] = None
    unset: bool = False
    source: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.problem is None


def check_operator_directory(
    path: Any, *, what: str, holds: str, outside_repository: bool
) -> Path:
    """The directory `path` names, resolved, or a refusal.

    **Absolute,** because a relative path means whatever the working directory
    makes of it. **An existing directory,** because the operator creates it; a
    mistyped path must not become a directory nobody chose. With
    `outside_repository`, **outside the repository after `resolve()`** (D7), so a
    symlink into the tree is caught.

    Returns the resolved path, which every later step joins onto. `what` names
    the directory in the refusal ("the registry"); `holds` says what it holds
    ("patient-level data"), which is why it must be absolute.

    Whether it is a directory is `is_directory`'s one explicit stat. A path
    whose state cannot be learned, there or in its resolve, is
    `DirectoryUncheckable`, never read as absent.

    Raises:
        DirectoryUncheckable: its state cannot be learned (a parent the account
            cannot enter, an I/O error): "cannot be checked", or "cannot be
            resolved" when its resolve failed so. A `DirectoryRefused`.
        DirectoryRefused: naming the directory and which rule it fails.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise DirectoryRefused(f"{what} must be a path, got {path!r}")
    given = Path(path)
    if not given.is_absolute():
        raise DirectoryRefused(
            f"{what} {str(given)!r} is not an absolute path. It holds {holds}, so where it "
            "is must not depend on the working directory; give its absolute path."
        )
    try:
        resolved = given.resolve()
    except (OSError, RuntimeError, ValueError) as exc:  # a link loop: RuntimeError before 3.13
        message = f"{what} {given} cannot be resolved ({type(exc).__name__}: {exc})."
        if isinstance(exc, OSError) and not _cannot_be_a_directory(exc):
            raise DirectoryUncheckable(message, path=given, reason=_error_reason(exc)) from None
        raise DirectoryRefused(message) from None
    if outside_repository:
        repo = REPO_ROOT.resolve()
        if resolved == repo or repo in resolved.parents:
            raise DirectoryRefused(
                f"{what} {given} resolves to {resolved}, inside the repository ({repo}). "
                f"{holds[:1].upper()}{holds[1:]} is never kept in the repository tree (D7); "
                "give a directory outside it."
            )
    try:
        directory = is_directory(resolved)
    except OSError as exc:
        raise DirectoryUncheckable(
            f"{what} {given} cannot be checked ({_error_reason(exc)}). The account that runs "
            "this must be able to reach it and every directory above it.",
            path=resolved, reason=_error_reason(exc),
        ) from None
    if not directory:
        raise DirectoryRefused(
            f"{what} {given} does not exist or is not a directory. The operator creates it; "
            "nothing creates one, so a mistyped path cannot become a directory nobody chose."
        )
    return resolved


def check_registry_root(path: Any) -> Path:
    """The cohort registry's root, resolved, or a refusal: the one registry rule,
    applied to `--registry` (`validate_registry`) and to `SHEPHERD_REGISTRY`."""
    return check_operator_directory(
        path, what="the registry", holds=PATIENT_LEVEL_DATA, outside_repository=True
    )


def _environment(environ: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _raw(environ: Optional[Mapping[str, str]], variable: str) -> Optional[str]:
    """The environment variable's value, or `None` when it is unset or empty.
    The environment only: the deployment mode, the local start's mark and the
    network-bind flag are read so, never from the saved file."""
    value = _environment(environ).get(variable)
    return value if value else None


# ---------------------------------------------------------------------------
# The one resolution of a data location (UI-06, 2026-11-16)
# ---------------------------------------------------------------------------
class _Field(NamedTuple):
    """A data location: its key in the saved settings file and its plain name
    in the reasons (`name`, `verb` agreeing with it)."""

    key: str
    name: str
    verb: str = "is"


_FIELDS = {
    ENV_WORKSPACES_ROOT: _Field(KEY_WORKSPACES_ROOT, "the workspaces folder"),
    ENV_REGISTRY: _Field(KEY_REGISTRY, "the results registry"),
    ENV_INTAKE_DIR: _Field(KEY_INTAKE_DIR, "the import folder"),
    ENV_TRAINING_EVIDENCE_DIRS: _Field(KEY_TRAINING_EVIDENCE_DIRS,
                                       "the training evidence folders", "are"),
}
#: The environment variable of each key the saved settings file holds.
SAVED_VARIABLES = {field.key: variable for variable, field in _FIELDS.items()}
#: Each key's plain name in a sentence ("the workspaces folder").
FIELD_NAMES = {field.key: field.name for field in _FIELDS.values()}


class _Resolved(NamedTuple):
    """What `_resolve` found: the raw value (an environment string, or the
    saved file's string or tuple) and its source; or, with `raw` None, the
    saved file's problem when it cannot be read, else nothing set."""

    raw: Any
    source: Optional[str]
    problem: Optional[str] = None


def _resolve(environ: Optional[Mapping[str, str]], variable: str) -> _Resolved:
    """**The one read point of a data location**, for every reader here (the
    lists, the D8 check's settings, the GPU lock, the registry rule, the CLIs
    and the jobs the service starts): the environment variable when it is set
    and not empty (a start-time override, which always wins), else the saved
    settings file's **applied** value (`deployment_settings.read_saved`, from
    `environ`'s location only), else nothing.

    A saved file that cannot be read is a problem for this setting, naming the
    file and why, unless the environment sets it; it is never read as empty.
    Nothing here writes a value into any environment."""
    raw = _raw(environ, variable)
    if raw is not None:
        return _Resolved(raw, SOURCE_ENVIRONMENT)
    field = _FIELDS[variable]
    saved = deployment_settings.read_saved(environ)
    if saved.problem is not None:
        remedy = (f"Fix it in {DATA_LOCATIONS}." if saved.path is not None else
                  "Correct it where the service is started.")
        return _Resolved(None, None, f"{saved.problem}: {field.name} is not known. {remedy}")
    value = saved.applied.get(field.key)
    if value is None:
        return _Resolved(None, None)
    return _Resolved(value, SOURCE_SAVED)


def environment_value(key: str, environ: Optional[Mapping[str, str]] = None) -> Any:
    """What an explicit environment variable of `environ` (the process's when
    None) sets the data location `key` to, in the saved file's shape (a
    string; for the evidence folders a tuple, split by `os.pathsep`, empty
    entries skipped), or None when it sets nothing: `_resolve`'s first layer
    alone, for an apply that must not change a value this process does not
    read from the file (`settings_store.held_back`)."""
    raw = _raw(environ, SAVED_VARIABLES[key])
    if raw is None or key not in deployment_settings.LIST_KEYS:
        return raw
    return tuple(entry for entry in raw.split(os.pathsep) if entry)


def configured(variable: str, environ: Optional[Mapping[str, str]] = None) -> _Resolved:
    """What a data location is set to and from where, before its rule checks
    it: `_resolve`'s answer (the one read point), for a view that shows a
    refused value with what was set (`src/utils/deployment_view.py`). Its
    `raw` is the environment's string or the saved file's value."""
    return _resolve(environ, variable)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _where(variable: str, source: Optional[str]) -> str:
    """The setting named with where its value came from, for a refusal:
    "the workspaces folder saved in Settings", or "... set at start (Ref: VAR)"."""
    field = _FIELDS[variable]
    if source == SOURCE_SAVED:
        return f"{field.name} saved in Settings"
    return f"{field.name} set at start (Ref: {variable})"


def _refusal(variable: str, source: Optional[str], reason: str, consequence: str) -> str:
    """A refused value's problem: which setting, from where, why, and what is
    off; a saved one says where to correct it."""
    text = f"{_cap(_where(variable, source))} {_FIELDS[variable].verb} refused: {reason} {consequence}"
    if source == SOURCE_SAVED:
        text += f" Correct it in {DATA_LOCATIONS}."
    return text


def _local_entry(environ: Optional[Mapping[str, str]]) -> bool:
    """Whether this is the local start: the launcher's mark is exactly
    `LOCAL_ENTRY` and the host passed to startup is a loopback bind."""
    return (_raw(environ, ENV_LOCAL_ENTRY) == LOCAL_ENTRY
            and bind_is_loopback(_environment(environ).get(ENV_BIND_HOST)))


#: Why the operator controls are off (`deployment_mode`): the state, then
#: why, in whole sentences, kept as parts so a line that names the controls
#: itself says the same (the launcher's "Operator controls : off, so …",
#: `mode_off_why`; T08 flow: it cut this lead off a sentence and printed the
#: rest in lower case after its own "off."). The value set is never
#: repeated: a read-only page may show it. Plain words first, the codes in
#: one trailing "Ref:" (the WebUI skill's §5; D8's reasons read the same
#: way, `operator_access.REF_MARK`).
MODE_OFF_STATE = "off, so this deployment is read-only"
#: Why, with no mode set.
MODE_NOT_SET_WHY = (
    "They turn on when the service is started from its desktop entry or launch_shepherd.sh "
    f"on this computer. Ref: {ENV_DEPLOYMENT_MODE}={OPERATOR_DEPLOYMENT_MODE} also turns them "
    "on, for a verified single-workspace deployment reached on the machine or through SSH "
    "forwarding"
)
#: Why, with a mode set at start that is not the operator mode.
MODE_NOT_M1_WHY = (
    "The deployment mode set at start is not the operator mode. A local start never "
    f"overrides it. Ref: {ENV_DEPLOYMENT_MODE} is not {OPERATOR_DEPLOYMENT_MODE} (the value "
    "is compared exactly)"
)
MODE_NOT_SET = f"Operator controls are {MODE_OFF_STATE}. {MODE_NOT_SET_WHY}"
MODE_NOT_M1 = f"Operator controls are {MODE_OFF_STATE}. {MODE_NOT_M1_WHY}"


def mode_off_why(mode: Setting) -> Optional[str]:
    """Why the operator controls are off, the whole sentences that
    `mode.problem` says after its state (`MODE_OFF_STATE`), for a line that
    names the controls itself; None while they are on. Told apart by
    `unset`, as a mode not set is told from one set and refused, never by
    the words."""
    if mode.available:
        return None
    return MODE_NOT_SET_WHY if mode.unset else MODE_NOT_M1_WHY


def deployment_mode(environ: Optional[Mapping[str, str]] = None) -> Setting:
    """The deployment mode: `M1`, or `None` and the reason the operator
    controls are off. Never from the saved file.

    An explicit `SHEPHERD_DEPLOYMENT_MODE` decides: exactly `M1` enables (source
    ``environment``), anything else is read-only and **never upgraded** by the
    local start. Unset, the local start's default gives `M1` (source
    ``local-default``) when `SHEPHERD_LOCAL_ENTRY` is exactly ``1`` and the bind
    host passed to startup is loopback (`_local_entry`); else it is unset and
    read-only. Either way D8 still checks every request. The value set is not
    repeated in the reason, which a read-only page may show."""
    raw = _raw(environ, ENV_DEPLOYMENT_MODE)
    if raw is None:
        if _local_entry(environ):
            return Setting(ENV_DEPLOYMENT_MODE, OPERATOR_DEPLOYMENT_MODE,
                           source=SOURCE_LOCAL_DEFAULT)
        return Setting(ENV_DEPLOYMENT_MODE, None, MODE_NOT_SET, unset=True)
    if raw != OPERATOR_DEPLOYMENT_MODE:
        return Setting(ENV_DEPLOYMENT_MODE, None, MODE_NOT_M1, source=SOURCE_ENVIRONMENT)
    return Setting(ENV_DEPLOYMENT_MODE, OPERATOR_DEPLOYMENT_MODE, source=SOURCE_ENVIRONMENT)


def _directory_setting(
    environ: Optional[Mapping[str, str]], variable: str, *, unset: str,
    check: Callable[[str], Path], consequence: str,
) -> Setting:
    """A setting naming one directory, resolved by `_resolve`: unset, its
    saved file unreadable, refused by `check`, or resolved."""
    found = _resolve(environ, variable)
    if found.problem is not None:
        return Setting(variable, None, found.problem)
    if found.raw is None:
        return Setting(variable, None, unset, unset=True)
    try:
        resolved = check(found.raw)
    except DirectoryRefused as exc:
        return Setting(variable, None, _refusal(variable, found.source, str(exc), consequence),
                       source=found.source)
    return Setting(variable, resolved, source=found.source)


#: The rules of the two directories other than the registry, by name, for the
#: readers and for a value to be saved (`check_data_location`).
_check_workspaces_root = partial(check_operator_directory, what="the workspaces root",
                                 holds="the workspaces", outside_repository=False)
_check_intake_directory = partial(check_operator_directory, what="the intake directory",
                                  holds=PATIENT_LEVEL_DATA, outside_repository=True)

# ---------------------------------------------------------------------------
# A setting's state in parts (the WebUI's notices) and as one sentence
# ---------------------------------------------------------------------------
class Notice(NamedTuple):
    """**A setting's state said in parts** (the WebUI's first-use notices).
    The WebUI shows `summary` and `action` and keeps the rest one click away
    (`renderers.notice_block`); the CLI, the API and the logs keep the whole
    sentence (the ``*_NOT_SET`` constants), which is built from the same part
    strings, so no fact can drift between the two. A notice is chosen by a
    reader's ``unset`` flag (`workspaces.setting_notice`), never by its words.

    - `key`: which notice (``workspaces_root``, ``gpu_lock``, ``registry``):
      tests, de-duplication on one screen and the page's ``data-notice``;
    - `summary`: the state, one front-loaded sentence ("No workspaces folder
      is set.");
    - `short`: the state as a relation says it, lower case and without a
      full stop ("no workspaces folder is set": "Start disabled: …");
    - `action`: the next step, by its place in the GUI ("Choose one in
      Settings › Data locations");
    - `target`: where that step is, ``"<page>#<element id>"`` (the WebUI's
      link opens the page and focuses the element's heading);
    - `details`: term rows, ``"Term: text"`` each, the why and how behind
      "Details ›", no server path in them;
    - `ref`: the trailing reference line's items (the variable, a guide
      section);
    - `cause`: the key of the notice this one follows from (the GPU lock's
      is the workspaces folder's): one screen says them as one;
    - `name`: the setting's plain name, as Settings › Data locations names
      its row ("Results registry"); where several notices are said as one
      block, it heads this notice's rows under "Details ›", so the state is
      said once on the screen ("" for a notice that has no row there).
    """

    key: str
    summary: str
    short: str
    action: str
    target: str
    details: Tuple[str, ...]
    ref: Tuple[str, ...]
    cause: Optional[str] = None
    name: str = ""


#: Each data location's plain name (a notice's `name`; the Settings row's,
#: `deployment_view.FIELD_WORDS`).
WORKSPACES_FOLDER = "Workspaces folder"
RESULTS_REGISTRY = "Results registry"
IMPORT_FOLDER = "Import folder"
#: The parts the notices and the whole sentences share (`Notice`).
NO_WORKSPACES_FOLDER = "No workspaces folder is set"
NO_RESULTS_REGISTRY = "No results registry is set"
NO_IMPORT_FOLDER = "No import folder is set"
#: The format of an import source the import folder holds.
INTAKE_FORMAT = "shepherd-cases/1"
#: The Data locations button that makes the suggested folder
#: (`data_locations.CREATE_LABEL`).
CREATE_SUGGESTED = "Create suggested folder"
#: Where the GPU lock's rule is written down for an operator.
GPU_LOCK_GUIDE = "deployment guide §4.11"
#: The commands' option that names the registry.
REGISTRY_OPTION = "--registry"
#: Where the registry must be.
OUTSIDE_REPOSITORY = "a folder outside the repository"
#: The suggested workspaces folder in words (`deployment_settings.suggested_words`).
SUGGESTED_WORKSPACES = deployment_settings.suggested_words(KEY_WORKSPACES_ROOT)
#: The next step of every unset data location, and where it is in the GUI.
CHOOSE_IN_DATA_LOCATIONS = f"Choose one in {DATA_LOCATIONS}"
DATA_LOCATIONS_TARGET = "settings#shep-dl"
#: **One next step for several notices** (the WebUI's `notices_block`: a screen that says two
#: unset locations draws one link to their one place): their one action as several notices say
#: it (`SEVERAL_ACTIONS`; an action not listed reads the same for several), or, where their
#: actions differ, the place itself (`PLACE_ACTIONS`, by target). Chosen by the parts, never by
#: reading a sentence's words.
CHOOSE_THEM_IN_DATA_LOCATIONS = f"Choose them in {DATA_LOCATIONS}"
OPEN_DATA_LOCATIONS = f"Open {DATA_LOCATIONS}"
SEVERAL_ACTIONS = {CHOOSE_IN_DATA_LOCATIONS: CHOOSE_THEM_IN_DATA_LOCATIONS}
PLACE_ACTIONS = {DATA_LOCATIONS_TARGET: OPEN_DATA_LOCATIONS}


def _relation(state: str) -> str:
    """A state's words as a relation says them: its first letter lower case."""
    return state[:1].lower() + state[1:]


#: Why nothing is listed with no workspaces folder set (`workspaces_root`).
#: It names the one suggested folder Settings offers
#: (`deployment_settings.suggested_words`), never another.
WORKSPACES_ROOT_NOT_SET = (
    f"{NO_WORKSPACES_FOLDER}, so the server lists no workspace. Choose the folder whose "
    f"subfolders are the workspaces in {DATA_LOCATIONS}; its {CREATE_SUGGESTED} makes "
    f"{SUGGESTED_WORKSPACES}. No default is assumed. "
    f"Ref: {ENV_WORKSPACES_ROOT}"
)
#: `WORKSPACES_ROOT_NOT_SET` in parts.
WORKSPACES_ROOT_NOTICE = Notice(
    key="workspaces_root",
    summary=f"{NO_WORKSPACES_FOLDER}.",
    short=_relation(NO_WORKSPACES_FOLDER),
    action=CHOOSE_IN_DATA_LOCATIONS,
    target=DATA_LOCATIONS_TARGET,
    details=(
        "Workspaces: one per subfolder",
        "Unset: the server lists no workspace",
        f"Suggested: {SUGGESTED_WORKSPACES} ({CREATE_SUGGESTED})",
        "Default: none is assumed",
    ),
    ref=(ENV_WORKSPACES_ROOT,),
    name=WORKSPACES_FOLDER,
)


def workspaces_root(environ: Optional[Mapping[str, str]] = None) -> Setting:
    """`SHEPHERD_WORKSPACES_ROOT`, or the saved workspaces folder: the one
    directory whose subdirectories are the workspaces the server lists. It may
    be inside the repository (the repository's `data/workspaces/` is the usual
    one), but only when named."""
    return _directory_setting(
        environ, ENV_WORKSPACES_ROOT, unset=WORKSPACES_ROOT_NOT_SET,
        check=_check_workspaces_root,
        consequence="The server lists no workspace until it is corrected.",
    )


#: Why every managed GPU job is refused when no workspaces root is set
#: (`gpu_lock_location`; review of 8242b47, P2-4).
GPU_LOCK_NOT_CONFIGURED = (
    f"{NO_WORKSPACES_FOLDER}, so the GPU lock is not configured and every managed GPU "
    "job is refused rather than run without excluding the others. Choose the folder in "
    f"{DATA_LOCATIONS}: the lock is {GPU_LOCK_FILE_NAME} in it, the one file every process "
    f"that uses the folder takes ({GPU_LOCK_GUIDE}). Ref: {ENV_WORKSPACES_ROOT}"
)
#: `GPU_LOCK_NOT_CONFIGURED` in parts; it follows from the workspaces folder's
#: notice, so a screen that says that one says this one with it.
GPU_LOCK_NOTICE = Notice(
    key="gpu_lock",
    summary="GPU jobs are refused until a workspaces folder is set.",
    short="GPU jobs need a workspaces folder",
    action=CHOOSE_IN_DATA_LOCATIONS,
    target=DATA_LOCATIONS_TARGET,
    details=(
        f"GPU lock: {GPU_LOCK_FILE_NAME} in the workspaces folder",
        "Taken by: every process that uses the folder",
        "GPU jobs: refused while it is unset, never run without the lock",
    ),
    ref=(ENV_WORKSPACES_ROOT, GPU_LOCK_GUIDE),
    cause=WORKSPACES_ROOT_NOTICE.key,
)


def gpu_lock_location(environ: Optional[Mapping[str, str]] = None) -> Setting:
    """The deployment's GPU lock file (§6, "M5's boundaries"): **the one rule**
    the job runner, the formal CLIs and the dashboard's chip apply, through
    `src.utils.gpu_lock.lock_location`.

    **The workspaces root is the lock's one name** (root-cause review,
    P24-S1, and its verification): the lock is `.shepherd-gpu.lock` directly
    under the workspaces root in effect, as `workspaces_root` reads it (the
    environment's, else the saved one), so every process that names the
    deployment's root -- the service, a shell given the same root, and since
    2026-11-16 any process of the account that reads the saved settings --
    takes the one file, and the server's list and the lock never read that
    setting differently. No other setting names a lock:
    `SHEPHERD_GPU_LOCK_FILE`, which did, is no longer read, since a process
    configured with it (a service with no root, say) and one given only the
    root counted as configured, locked two files and ran side by side. The
    tests and the deployment probe take a lock of their own by naming a
    workspaces root of their own.

    **No root is a problem, not a default** (review of 8242b47, P2-4):
    `value` is None, `unset` True and `problem` says how to configure it, so
    every managed GPU job is refused rather than run without excluding the
    others. A root that is set but refused, or a saved file that cannot be
    read, is a problem too, never replaced by a default.

    **The limit one rule keeps**: each process names the lock from its own
    settings, so a process whose environment names another root takes another
    file and is not excluded. A saved root is changed only while nothing
    could be using the old lock (`settings_store.apply_saved`), and never by
    a service whose own start-up setting names another root, which would
    move every other process's lock away from the one it takes
    (`settings_store.held_back`); a taker checks after taking it that the
    location still holds (`gpu_lock.take_gpu_lock`).
    """
    root = workspaces_root(environ)
    if root.value is None:
        if root.unset:
            return Setting(ENV_WORKSPACES_ROOT, None, GPU_LOCK_NOT_CONFIGURED, unset=True)
        return Setting(ENV_WORKSPACES_ROOT, None, (
            f"{root.problem} The GPU lock under it cannot be used either, so every managed "
            "GPU job is refused until it is corrected."), source=root.source)
    return Setting(ENV_WORKSPACES_ROOT, root.value / GPU_LOCK_FILE_NAME, source=root.source)


#: Why the cohort features are off with no registry set (`cohort_registry`).
REGISTRY_NOT_SET = (
    f"{NO_RESULTS_REGISTRY}, so the features that read the cohort registry are off. "
    f"Choose it in {DATA_LOCATIONS}: {OUTSIDE_REPOSITORY}, the one the commands "
    f"take as {REGISTRY_OPTION}. Ref: {ENV_REGISTRY}"
)
#: `REGISTRY_NOT_SET` in parts.
REGISTRY_NOTICE = Notice(
    key="registry",
    summary=f"{NO_RESULTS_REGISTRY}.",
    short=_relation(NO_RESULTS_REGISTRY),
    action=CHOOSE_IN_DATA_LOCATIONS,
    target=DATA_LOCATIONS_TARGET,
    details=(
        "Holds: the cohort versions and registered reports",
        "Off while unset: every feature that reads the registry",
        f"Where: {OUTSIDE_REPOSITORY}, the commands' {REGISTRY_OPTION}",
    ),
    ref=(ENV_REGISTRY,),
    name=RESULTS_REGISTRY,
)


def cohort_registry(environ: Optional[Mapping[str, str]] = None) -> Setting:
    """`SHEPHERD_REGISTRY`, or the saved results registry: the cohort registry,
    checked by the registry rule (`check_registry_root`: absolute, existing,
    outside the repository)."""
    return _directory_setting(
        environ, ENV_REGISTRY, unset=REGISTRY_NOT_SET,
        check=check_registry_root,
        consequence="The features that read the cohort registry are off until it is corrected.",
    )


def deployment_registry_refusal(registry: Path,
                                environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """`None` when `registry`, a registry root already resolved, is the
    deployment's registry -- the one `cohort_registry` resolves (the
    environment's `SHEPHERD_REGISTRY`, else the saved results registry), by
    the registry rule (`check_registry_root`) -- or why it is not, as a
    clause with no closing full stop, for the caller to put in its own sentence.

    **The one rule for which registry a report is registered from and a cohort
    version is measured in** (D5, D7, BACKLOG 37): `locate_store_report` asks it
    for every registration, the UI's included, and `measure_scorer.py` for every
    ``--cohort`` run, before either reads anything in the registry. Another
    directory keeps its own report store and usage history, which the service
    never reads.

    **Compared as resolved paths** (`Path.resolve()` on both sides): a link to
    the configured directory is that directory. Never by device and inode
    (`os.path.samefile`), which can be 0 for every file on some Windows volumes
    (WebDAV), where every directory would then compare equal and a copy would
    pass (cpython issue 78116). The cost falls on the safe side: the same
    directory reached through a bind mount or another mount resolves to another
    path and is refused, and the refusal names the path to use.

    **Unset, every such use is refused**: a process that names no registry
    cannot tell which registry is the deployment's, and accepting any would be
    the bypass this rule closes. Since 2026-11-16 a shell of the account the
    service runs as reads the saved registry too (BACKLOG 45), so a fresh SSH
    session is no longer that process. It reads this process's settings (or
    `environ`): a command run with the variable set to another directory on
    purpose uses that one. Like the store's other rules, it catches a registry
    reached by mistake; it does not make the store tamper-proof (D2).
    """
    found = _resolve(environ, ENV_REGISTRY)
    if found.problem is not None:
        return found.problem.rstrip().rstrip(".")
    if found.raw is None:
        return ("no results registry is set here, so which registry is the deployment's cannot "
                f"be told. Choose it in {DATA_LOCATIONS}, which the service and the commands "
                f"of the same account read, or set {ENV_REGISTRY} for this command to the "
                "service's value")
    raw = found.raw
    try:
        configured = check_registry_root(raw)
    except DirectoryRefused as exc:
        return f"{_where(ENV_REGISTRY, found.source)} is refused: {str(exc).rstrip().rstrip('.')}"
    if Path(registry) != configured:
        change = (f"choose it in {DATA_LOCATIONS}" if found.source == SOURCE_SAVED else
                  f"set {ENV_REGISTRY} to it for the service and here")
        return (f"the registry {registry} is not the deployment's: "
                f"{_where(ENV_REGISTRY, found.source)} is {raw}, which resolves to {configured}. "
                "Registries are compared by resolved path, so "
                "another directory is another registry whatever it holds, a copy of this one or "
                "the same directory reached through a bind mount or another mount. Use the path "
                f"under {configured}, or, if {registry} is now the deployment's registry, {change}")
    return None


#: Why no import source is offered with no import folder set.
INTAKE_DIR_NOT_SET = (
    f"{NO_IMPORT_FOLDER}, so no import source is offered. Choose it in "
    f"{DATA_LOCATIONS}: {OUTSIDE_REPOSITORY} where you place {INTAKE_FORMAT} "
    f"files. Ref: {ENV_INTAKE_DIR}"
)
#: `INTAKE_DIR_NOT_SET` in parts (the Test page's import drawer).
INTAKE_DIR_NOTICE = Notice(
    key="intake_dir",
    summary=f"{NO_IMPORT_FOLDER}.",
    short=_relation(NO_IMPORT_FOLDER),
    action=CHOOSE_IN_DATA_LOCATIONS,
    target=DATA_LOCATIONS_TARGET,
    details=(
        f"Offers: the {INTAKE_FORMAT} files you place in it",
        "Until set: no import source is offered",
        f"Location: {OUTSIDE_REPOSITORY}",
    ),
    ref=(ENV_INTAKE_DIR,),
    name=IMPORT_FOLDER,
)


def intake_directory(environ: Optional[Mapping[str, str]] = None) -> Setting:
    """`SHEPHERD_INTAKE_DIR`, or the saved import folder: where the operator
    places `shepherd-cases/1` sources to import (D7), checked as the registry
    is: absolute, existing, outside the repository."""
    return _directory_setting(
        environ, ENV_INTAKE_DIR, unset=INTAKE_DIR_NOT_SET,
        check=_check_intake_directory,
        consequence="No import source is offered until it is corrected.",
    )


def _check_evidence_dirs(entries: Iterable[str]) -> Tuple[Tuple[Path, ...], Optional[str]]:
    """The evidence directories `entries` name, resolved, in order, repeats
    kept once, and the note naming those whose state cannot be learned (kept
    in their place); `training_evidence_dirs` states the rule.

    Raises:
        DirectoryRefused: an entry is known not to be a usable directory."""
    resolved: list[Path] = []
    unchecked: list[str] = []
    for index, entry in enumerate((e for e in entries if e), start=1):
        try:
            directory = check_operator_directory(
                entry, what=f"evidence directory {index}", holds="training evidence",
                outside_repository=False,
            )
        except DirectoryUncheckable as exc:
            directory = exc.path
            if directory not in resolved:
                unchecked.append(f"{directory} ({exc.reason})")
        if directory not in resolved:
            resolved.append(directory)
    note = None
    if unchecked:
        # What a run does with it, no more: R10 names it only for inputs it
        # finds nowhere else (`r10.compute_r10`).
        them = "it" if len(unchecked) == 1 else "them"
        note = (f"Cannot be checked: {'; '.join(unchecked)}. R10 cannot look in {them}: "
                f"inputs found nowhere else read unverifiable, naming {them}.")
    return tuple(resolved), note


def training_evidence_dirs(environ: Optional[Mapping[str, str]] = None) -> Setting:
    """`SHEPHERD_TRAINING_EVIDENCE_DIRS` (separated by `os.pathsep`), or the
    saved training evidence folders (a list): R10 evidence directories (the
    measurement's `--training-evidence`).

    Optional: unset or empty is an empty tuple, usable. Empty entries (a
    doubled or trailing separator) are skipped, never read as the working
    directory. Each entry must be an absolute, existing directory; one known
    not to be (relative, absent, not a directory, a link loop, a name too long
    to exist, a path that cannot be encoded) refuses the whole setting (value
    `None`), so a run never goes ahead on part of what the operator listed. A
    saved settings file that cannot be read refuses it too.

    **An entry whose state cannot be learned** (`DirectoryUncheckable`: a
    parent this account may not enter, an I/O error) is neither refused nor
    dropped (K8-6 completion review, 2026-11-08): it is optional evidence, so it
    stays in the value, in its place (resolved when its resolve worked), and
    `note` names it with its reason; the setting stays available. A run is
    given it as `--training-evidence`, where R10 cannot look in it: a scope
    found elsewhere stays found, and only one found nowhere else names it, as
    `search_incomplete`. Repeats are kept once, by resolved path, in the order
    given.
    """
    found = _resolve(environ, ENV_TRAINING_EVIDENCE_DIRS)
    if found.problem is not None:
        return Setting(ENV_TRAINING_EVIDENCE_DIRS, None, found.problem)
    if found.raw is None:
        return Setting(ENV_TRAINING_EVIDENCE_DIRS, ())
    entries = found.raw.split(os.pathsep) if isinstance(found.raw, str) else found.raw
    try:
        resolved, note = _check_evidence_dirs(entries)
    except DirectoryRefused as exc:
        return Setting(
            ENV_TRAINING_EVIDENCE_DIRS, None,
            _refusal(ENV_TRAINING_EVIDENCE_DIRS, found.source, str(exc),
                     "No evidence directory is offered until it is corrected."),
            source=found.source,
        )
    return Setting(ENV_TRAINING_EVIDENCE_DIRS, resolved, note=note, source=found.source)


def check_data_location(key: str, value: Any) -> Tuple[Any, Optional[str]]:
    """A value to be saved for the data location `key` (a key of the saved
    settings file), checked now by the rule its setting is read by:
    ``(resolved, note)``, the resolved directory (a tuple of them for the
    evidence folders; None when `value` sets nothing) and the note a reader
    would give. The value saved is the one given; each read checks it again,
    as an environment value is checked.

    Raises:
        ValueError: `key` is not a data location or `value` is not of its
            shape (`deployment_settings.normalise_value`).
        DirectoryRefused: the rule refuses it, saying why.
    """
    value = deployment_settings.normalise_value(key, value)
    if value is None:
        return None, None
    if key == KEY_TRAINING_EVIDENCE_DIRS:
        return _check_evidence_dirs(value)
    check = {KEY_WORKSPACES_ROOT: _check_workspaces_root, KEY_REGISTRY: check_registry_root,
             KEY_INTAKE_DIR: _check_intake_directory}[key]
    return check(value), None


def all_settings(environ: Optional[Mapping[str, str]] = None) -> Tuple[Setting, ...]:
    """Every setting as read now, in the table's order, for a status view."""
    return (
        deployment_mode(environ),
        workspaces_root(environ),
        cohort_registry(environ),
        intake_directory(environ),
        training_evidence_dirs(environ),
    )


# ---------------------------------------------------------------------------
# Loopback, and the network bind (SPEC_4 §3.1, B1)
# ---------------------------------------------------------------------------
def is_loopback_ip(text: Any) -> bool:
    """Whether `text` is an IP address literal on the loopback network:
    127.0.0.0/8, ``::1``, or an IPv4-mapped IPv6 form of one
    (``::ffff:127.0.0.1``, which a dual-stack socket reports and which
    `ipaddress` itself does not call loopback). A name, ``localhost``
    included, is not an address and answers False; so does anything that
    is not a string.
    """
    if not isinstance(text, str):
        return False
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return mapped is not None and mapped.is_loopback


def bind_is_loopback(host: Any) -> bool:
    """Whether a bind address keeps the service on this machine: a loopback
    address literal (`is_loopback_ip`, brackets around an IPv6 literal
    allowed) or the name ``localhost``. The wildcards (``0.0.0.0``, ``::``,
    an empty host) and every other name or address are not, since what they
    reach cannot be told from here."""
    if not isinstance(host, str):
        return False
    text = host.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return text.lower() == "localhost" or is_loopback_ip(text)


class BindDecision(NamedTuple):
    """Whether the service may bind `host`: `allowed`, and `message`, the
    refusal or, for an accepted network bind, the warning to show (`None`
    for a loopback bind)."""

    host: str
    allowed: bool
    message: Optional[str]


def network_bind(host: Any, environ: Optional[Mapping[str, str]] = None) -> BindDecision:
    """SPEC_4 §3.1's bind guard for one bind address.

    A loopback bind (`bind_is_loopback`) is allowed. Any other is refused
    unless `SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND` is exactly ``1``:
    the service has no authentication, so a network bind is reachable by
    whoever reaches the address. The flag is **risk acceptance, not a
    control**, and the warning says so; it never lifts D8's checks, so
    operator controls stay refused for requests that do not arrive on
    loopback. Read at each call, never from a request.

    Not covered, and not claimed (SPEC_4 §3.1): a uvicorn started by hand
    with ``--host``, which nothing in the application can read.
    """
    shown = host if isinstance(host, str) else repr(host)
    if bind_is_loopback(host):
        return BindDecision(shown, True, None)
    where = "all interfaces" if shown.strip() in ("", "0.0.0.0", "::", "[::]") else shown
    if _accepts_network_bind(environ):
        return BindDecision(shown, True, (
            f"binding {where}, which is not loopback: the service is reachable from the "
            f"network with no authentication, accepted by "
            f"{ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND}={ACCEPTS_NETWORK_BIND}. That is risk "
            "acceptance, not a control: it authenticates nobody. Operator controls stay "
            "refused for every request that does not arrive on loopback (D8)."))
    return BindDecision(shown, False, (
        f"refusing to bind {where}: an address other than loopback makes the service "
        "reachable from the network, and it has no authentication. Bind "
        f"{DEFAULT_BIND_HOST} (the default) and reach it on the machine or through SSH "
        "forwarding to localhost, or, to accept that risk, set "
        f"{ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND}={ACCEPTS_NETWORK_BIND} (risk acceptance, "
        "not a control; operator controls stay refused off loopback). Nothing was started."))


def _accepts_network_bind(environ: Optional[Mapping[str, str]]) -> bool:
    """Whether the risk of a network bind is accepted: the flag is exactly ``1``."""
    return _raw(environ, ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND) == ACCEPTS_NETWORK_BIND


def inherited_socket_bind(fd: Any, environ: Optional[Mapping[str, str]] = None) -> BindDecision:
    """§3.1's bind guard for a server handed a listening socket (uvicorn
    ``--fd``, as systemd socket activation passes one). Its address is not
    read here, and a socket unit's ``ListenStream=8264`` listens on every
    interface, so it counts as a network bind: refused unless
    `SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND` is exactly ``1``, and
    then allowed with `network_bind`'s warning."""
    shown = f"the inherited socket (file descriptor {fd})"
    if _accepts_network_bind(environ):
        return BindDecision(shown, True, (
            f"serving on {shown}, whose address is not read here, so it counts as a network "
            "bind: the service may be reachable from the network with no authentication, "
            f"accepted by {ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND}={ACCEPTS_NETWORK_BIND}. That "
            "is risk acceptance, not a control: it authenticates nobody. Operator controls stay "
            "refused for every request that does not arrive on loopback (D8)."))
    return BindDecision(shown, False, (
        f"refusing to serve on {shown}: its address is not read here, so it counts as a "
        "network bind, which would make the service reachable from the network with no "
        f"authentication. Bind {DEFAULT_BIND_HOST} (the default, with --host) and reach it on "
        "the machine or through SSH forwarding to localhost, or, to accept that risk, set "
        f"{ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND}={ACCEPTS_NETWORK_BIND} (risk acceptance, "
        "not a control; operator controls stay refused off loopback). Nothing was started."))


def startup_bind(environ: Optional[Mapping[str, str]] = None) -> Optional[BindDecision]:
    """§3.1's third enforcement point: `network_bind` for the host the
    launcher or `src.api.main:main()` passed to application startup
    (`ENV_BIND_HOST`; an empty value is the empty ``--host=``, which binds
    every interface), or `None` when none was passed. Both set it to the host
    they bind after applying the same rule, so startup refuses only a host
    passed in without that check. A uvicorn started by hand passes none and
    is not checked (SPEC_4 §3.1: not covered, and not claimed)."""
    host = _environment(environ).get(ENV_BIND_HOST)
    if host is None:
        return None
    return network_bind(host, environ)
