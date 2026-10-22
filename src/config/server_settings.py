"""
Server settings: what an operator configures for M5's controls and lists.
=========================================================================
`docs/working/PLAN_TEST_RESULTS.md` §6 ("M5's boundaries"), D7, D8 and §4.9
questions 1 and 3; SPEC_4 §3 (deployment mode M1). Every setting here is an
environment variable of the service, **read at each use and never from a
request**: no route, form or saved UI file can set one, and a change takes effect
the next time a setting is read (a restart is the reliable way to apply one to
everything). Each reader returns a `Setting`: the value in effect, or why the
feature that needs it is off.

| Variable | What it is | Unset |
|---|---|---|
| `SHEPHERD_DEPLOYMENT_MODE` | SPEC_4's deployment mode; only the exact value `M1` enables operator controls, which D8's check then allows request by request (below) | read-only |
| `SHEPHERD_WORKSPACES_ROOT` | the one workspace root: its subdirectories are the workspaces the server lists (SPEC_4 M1's "one explicit workspace namespace") | no workspace listed |
| `SHEPHERD_REGISTRY` | the cohort registry (the `--registry` of the CLI), the only one reports are registered from and cohort versions measured in (`deployment_registry_refusal`) | no cohort feature; every registration and `--cohort` measurement refused |
| `SHEPHERD_INTAKE_DIR` | where an operator places import sources | no import source |
| `SHEPHERD_TRAINING_EVIDENCE_DIRS` | R10 evidence directories, `os.pathsep`-separated (optional) | none |
| `SHEPHERD_ORDERING_ROLES` | cohort roles that offer ordering by test score, comma-separated | none |

**The GPU lock's location is read here too** (`gpu_lock_location`, the one
rule `src.utils.gpu_lock` applies). **The workspaces root is its identity**:
with `SHEPHERD_WORKSPACES_ROOT` set the lock is `.shepherd-gpu.lock` under it,
and `SHEPHERD_GPU_LOCK_FILE` is accepted only when it names that same file;
with no root set, `SHEPHERD_GPU_LOCK_FILE` names it. With neither set, every
managed GPU job is refused (review of 8242b47, P2-4).

**A missing setting turns its feature off with a stated reason; nothing falls
back to a default path**, inside the repository or anywhere else. The
repository's `data/workspaces/` is a workspace root only when the operator names
it.

**One rule for a directory an operator names** (`check_operator_directory`):
absolute, resolved, an existing directory, and for the two that hold
patient-level data (the registry and the intake directory) outside the
repository after `resolve()`, so a symlink into the tree is caught. The registry
rule is `check_registry_root`, and `src.evaluation.cohort_registry.
validate_registry` applies the same function, so `--registry` on the command line
and `SHEPHERD_REGISTRY` in the service are one rule. It lives here, below
`src.evaluation`, because the D8 check and the GPU lock in `src.utils` read these
settings too.

**What the deployment mode does not do.** `M1` is one of D8's conditions, not an
authorisation: the check on each operator control (loopback arrival, no
forwarding header, a loopback Host) is D8's, applied where the request is
handled. Setting `M1` behind a reverse proxy or on a shared machine does not make
those controls safe (SPEC_4 §3: loopback is not authorisation).

**Where it is applied.** `deployment_mode()` is read by D8's check
(`src/utils/operator_access.py`) on every operator control, at the moment the
request is handled: unset or any value but `M1`, every control is refused and
the deployment is read-only, whatever arrives on loopback.

**The network bind** (SPEC_4 §3.1, B1): the service binds `127.0.0.1` by
default, and a bind to any other address is refused unless
`SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND=1` (`network_bind`). The flag
records **risk acceptance, not a control**: it authenticates nobody, and it
never lifts D8's checks, so an operator control stays refused for every request
that does not arrive on loopback. §3.1's three enforcement points apply that one
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

import ipaddress
import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Optional, Tuple

#: The repository this module is part of. Computed from this file, as
#: `src/config/runtime_presets.py` does, and compared after `resolve()`.
REPO_ROOT = Path(__file__).resolve().parents[2]

ENV_DEPLOYMENT_MODE = "SHEPHERD_DEPLOYMENT_MODE"
ENV_WORKSPACES_ROOT = "SHEPHERD_WORKSPACES_ROOT"
ENV_REGISTRY = "SHEPHERD_REGISTRY"
ENV_INTAKE_DIR = "SHEPHERD_INTAKE_DIR"
ENV_TRAINING_EVIDENCE_DIRS = "SHEPHERD_TRAINING_EVIDENCE_DIRS"
ENV_ORDERING_ROLES = "SHEPHERD_ORDERING_ROLES"
#: The deployment's GPU lock file, when it is not the one under the workspaces
#: root (`gpu_lock_location`).
ENV_GPU_LOCK_FILE = "SHEPHERD_GPU_LOCK_FILE"
#: The GPU lock's file name directly under the workspaces root.
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

#: Roles that never offer ordering by test score, even when listed (§4.9
#: question 1): until the institution's formal evaluation rules exist.
NEVER_ORDERED_ROLES: Tuple[str, ...] = ("institutional_acceptance",)

#: What the two patient-level directories hold, in their refusals.
PATIENT_LEVEL_DATA = "patient-level data"

__all__ = [
    "ACCEPTS_NETWORK_BIND",
    "BindDecision",
    "DEFAULT_BIND_HOST",
    "DirectoryRefused",
    "ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND",
    "ENV_BIND_HOST",
    "ENV_DEPLOYMENT_MODE",
    "ENV_GPU_LOCK_FILE",
    "ENV_INTAKE_DIR",
    "ENV_ORDERING_ROLES",
    "ENV_REGISTRY",
    "ENV_TRAINING_EVIDENCE_DIRS",
    "ENV_WORKSPACES_ROOT",
    "GPU_LOCK_FILE_NAME",
    "GPU_LOCK_NOT_CONFIGURED",
    "NEVER_ORDERED_ROLES",
    "OPERATOR_DEPLOYMENT_MODE",
    "Setting",
    "all_settings",
    "check_operator_directory",
    "bind_is_loopback",
    "check_registry_root",
    "cohort_registry",
    "deployment_mode",
    "deployment_registry_refusal",
    "gpu_lock_location",
    "inherited_socket_bind",
    "intake_directory",
    "is_loopback_ip",
    "network_bind",
    "offers_ordering",
    "ordering_roles",
    "role_offers_ordering",
    "startup_bind",
    "training_evidence_dirs",
    "workspaces_root",
]


class DirectoryRefused(ValueError):
    """A directory an operator named is not one this rule accepts."""


class Setting(NamedTuple):
    """One setting as read now.

    `value` is what is in effect: a string, a resolved `Path`, or a tuple; `None`
    when the setting is unusable. `problem` is why the feature that needs it is
    off, in words an operator can act on; `None` when it is usable. `note` is a
    remark that turns nothing off (an entry that was ignored, say). `unset`
    says the problem is that nothing is set at all, not a value that is set
    and refused; a caller tells the two apart by it, never by the words.
    """

    variable: str
    value: Any
    problem: Optional[str] = None
    note: Optional[str] = None
    unset: bool = False

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

    Raises:
        DirectoryRefused: naming the directory and which rule it fails, or that
            it cannot be checked (a parent the account cannot enter).
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
        raise DirectoryRefused(
            f"{what} {given} cannot be resolved ({type(exc).__name__}: {exc})."
        ) from None
    if outside_repository:
        repo = REPO_ROOT.resolve()
        if resolved == repo or repo in resolved.parents:
            raise DirectoryRefused(
                f"{what} {given} resolves to {resolved}, inside the repository ({repo}). "
                f"{holds[:1].upper()}{holds[1:]} is never kept in the repository tree (D7); "
                "give a directory outside it."
            )
    try:
        is_directory = resolved.is_dir()
    except OSError as exc:  # `is_dir` answers False for a missing path, raises EACCES
        raise DirectoryRefused(
            f"{what} {given} cannot be checked ({type(exc).__name__}: "
            f"{exc.strerror or exc}). The account that runs this must be able to reach it "
            "and every directory above it."
        ) from None
    if not is_directory:
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
    """The variable's value, or `None` when it is unset or empty."""
    value = _environment(environ).get(variable)
    return value if value else None


def deployment_mode(environ: Optional[Mapping[str, str]] = None) -> Setting:
    """`SHEPHERD_DEPLOYMENT_MODE`: `M1` when it is exactly that, else `None` and
    the reason the operator controls are off. The value set is not repeated in
    the reason, which a read-only page may show."""
    raw = _raw(environ, ENV_DEPLOYMENT_MODE)
    if raw is None:
        return Setting(
            ENV_DEPLOYMENT_MODE, None,
            f"{ENV_DEPLOYMENT_MODE} is not set, so the operator controls are off and this "
            f"deployment is read-only. Only {OPERATOR_DEPLOYMENT_MODE}, a verified "
            "single-workspace deployment reached on the machine or through SSH forwarding, "
            "enables them.",
            unset=True,
        )
    if raw != OPERATOR_DEPLOYMENT_MODE:
        return Setting(
            ENV_DEPLOYMENT_MODE, None,
            f"{ENV_DEPLOYMENT_MODE} is set, but not to {OPERATOR_DEPLOYMENT_MODE} (the value "
            "is compared exactly), so the operator controls are off and this deployment is "
            "read-only.",
        )
    return Setting(ENV_DEPLOYMENT_MODE, OPERATOR_DEPLOYMENT_MODE)


def _directory_setting(
    environ: Optional[Mapping[str, str]], variable: str, *, unset: str,
    check: Callable[[str], Path], consequence: str,
) -> Setting:
    """A setting naming one directory: unset, refused by `check`, or resolved."""
    raw = _raw(environ, variable)
    if raw is None:
        return Setting(variable, None, unset, unset=True)
    try:
        resolved = check(raw)
    except DirectoryRefused as exc:
        return Setting(variable, None, f"{variable} is refused: {exc} {consequence}")
    return Setting(variable, resolved)


def workspaces_root(environ: Optional[Mapping[str, str]] = None) -> Setting:
    """`SHEPHERD_WORKSPACES_ROOT`: the one directory whose subdirectories are the
    workspaces the server lists. It may be inside the repository (the
    repository's `data/workspaces/` is the usual one), but only when named."""
    return _directory_setting(
        environ, ENV_WORKSPACES_ROOT,
        unset=(f"{ENV_WORKSPACES_ROOT} is not set, so the server lists no workspace. Set it "
               "to the absolute path of the directory whose subdirectories are the "
               "workspaces (for example this repository's data/workspaces); no default is "
               "assumed."),
        check=partial(check_operator_directory, what="the workspaces root",
                      holds="the workspaces", outside_repository=False),
        consequence="The server lists no workspace until it is corrected.",
    )


#: Why every managed GPU job is refused when neither lock setting is set
#: (`gpu_lock_location`; review of 8242b47, P2-4).
GPU_LOCK_NOT_CONFIGURED = (
    f"neither {ENV_GPU_LOCK_FILE} nor {ENV_WORKSPACES_ROOT} is set, so the GPU lock is not "
    "configured and every managed GPU job is refused rather than run without excluding the "
    f"others. Set {ENV_WORKSPACES_ROOT} in this environment to the value the service uses: the "
    f"lock is then {GPU_LOCK_FILE_NAME} under that root, the one file every process that names "
    f"the root takes (deployment guide §4.11). {ENV_GPU_LOCK_FILE} names the lock only where no "
    "workspaces root is set; for a service set up that way, name its lock file, and no root, "
    "here."
)


def _lock_file(raw: str) -> Setting:
    """The file `SHEPHERD_GPU_LOCK_FILE`'s value `raw` names, its directory
    resolved (`check_operator_directory`), or why it names none: not absolute,
    a directory that does not exist or cannot be checked, or a directory
    itself."""
    given = Path(raw)
    if not given.is_absolute():
        return Setting(ENV_GPU_LOCK_FILE, None, (
            f"{ENV_GPU_LOCK_FILE} is refused: {raw!r} is not an absolute path. Every "
            "process that takes the GPU lock must name the same file, whatever its "
            "working directory."))
    try:
        directory = check_operator_directory(
            given.parent, what="the GPU lock file's directory", holds="the GPU lock",
            outside_repository=False)
    except DirectoryRefused as exc:
        return Setting(ENV_GPU_LOCK_FILE, None, f"{ENV_GPU_LOCK_FILE} is refused: {exc}")
    path = directory / given.name
    try:
        is_directory = path.is_dir()
    except OSError as exc:  # `is_dir` raises EACCES on 3.12 rather than answering
        return Setting(ENV_GPU_LOCK_FILE, None, (
            f"{ENV_GPU_LOCK_FILE} is refused: {path} cannot be checked "
            f"({type(exc).__name__}: {exc.strerror or exc})."))
    if is_directory:
        return Setting(ENV_GPU_LOCK_FILE, None, (
            f"{ENV_GPU_LOCK_FILE} is refused: {path} is a directory; name a file "
            "inside it."))
    return Setting(ENV_GPU_LOCK_FILE, path)


def gpu_lock_location(environ: Optional[Mapping[str, str]] = None) -> Setting:
    """The deployment's GPU lock file (§6, "M5's boundaries"): **the one rule**
    the job runner, the formal CLIs and the dashboard's chip apply, through
    `src.utils.gpu_lock.lock_location`.

    **The workspaces root is the lock's identity** (root-cause review,
    P24-S1). With `SHEPHERD_WORKSPACES_ROOT` set, as `workspaces_root` reads
    it, the lock is `.shepherd-gpu.lock` directly under it: every process that
    names the deployment's root (the service, and a shell given the same root)
    takes the one file, whatever else it sets, and the server's list and the
    lock never read that variable differently. `SHEPHERD_GPU_LOCK_FILE` beside
    it is accepted only when it names that same file; another file is a
    problem that names both, never settled by picking one (a precedence
    between the two let a service with a lock file and a shell with only the
    root lock different files and run side by side). With no root set,
    `SHEPHERD_GPU_LOCK_FILE` names the lock (`_lock_file`): the tests' and the
    probe's own lock, or a deployment that names no workspaces root.
    `variable` names the setting that decided.

    **Neither set is a problem, not a default** (review of 8242b47, P2-4):
    `value` is None, `unset` True and `problem` says how to configure it, so
    every managed GPU job is refused rather than run without excluding the
    others. A setting that is set but unusable is refused too, never replaced
    by the other one or by a default.

    **The limit one rule keeps**: each process names the lock from its own
    environment, so a process that names another root, or a lock file and no
    root, takes another file and is not excluded.
    """
    raw_file = _raw(environ, ENV_GPU_LOCK_FILE)
    if _raw(environ, ENV_WORKSPACES_ROOT) is None:
        if raw_file is None:
            return Setting(ENV_GPU_LOCK_FILE, None, GPU_LOCK_NOT_CONFIGURED, unset=True)
        return _lock_file(raw_file)
    root = workspaces_root(environ)
    if root.value is None:  # set, and refused: whatever the file setting says
        return Setting(ENV_WORKSPACES_ROOT, None, (
            f"{root.problem} The GPU lock under it cannot be used either, so every managed "
            "GPU job is refused until it is corrected."))
    lock = root.value / GPU_LOCK_FILE_NAME
    if raw_file is not None:
        named = _lock_file(raw_file)
        if named.value is None:
            return named
        if named.value != lock:
            return Setting(ENV_GPU_LOCK_FILE, None, (
                f"{ENV_GPU_LOCK_FILE} names {named.value}, and {ENV_WORKSPACES_ROOT} names "
                f"the lock {lock}: two different GPU lock files. With the workspaces root "
                f"set, the lock is {lock}, the file every process that names this root takes, "
                "so a job on the other file would not exclude theirs. Remove "
                f"{ENV_GPU_LOCK_FILE} from this environment (or set it to {lock}); every "
                "managed GPU job is refused until then."))
    return Setting(ENV_WORKSPACES_ROOT, lock)


def cohort_registry(environ: Optional[Mapping[str, str]] = None) -> Setting:
    """`SHEPHERD_REGISTRY`: the cohort registry, checked by the registry rule
    (`check_registry_root`: absolute, existing, outside the repository)."""
    return _directory_setting(
        environ, ENV_REGISTRY,
        unset=(f"{ENV_REGISTRY} is not set, so the features that read the cohort registry are "
               "off. Set it to the registry's absolute path, outside the repository: the "
               "directory the commands take as --registry."),
        check=check_registry_root,
        consequence="The features that read the cohort registry are off until it is corrected.",
    )


def deployment_registry_refusal(registry: Path,
                                environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """`None` when `registry`, a registry root already resolved, is the
    deployment's registry -- the directory `SHEPHERD_REGISTRY` names, resolved
    by the registry rule (`check_registry_root`) -- or why it is not, as a
    clause with no closing full stop, for the caller to put in its own sentence.

    **The one rule for which registry a report is registered from and a cohort
    version is measured in** (D5, D7, BACKLOG 37): `locate_store_report` asks it
    for every registration, the UI's included, and `measure_scorer.py` for every
    ``--cohort`` run, before either reads anything in the registry. Another
    directory keeps its own report store, disagreement block and usage history,
    which the service never reads.

    **Compared as resolved paths** (`Path.resolve()` on both sides): a link to
    the configured directory is that directory. Never by device and inode
    (`os.path.samefile`), which can be 0 for every file on some Windows volumes
    (WebDAV), where every directory would then compare equal and a copy would
    pass (cpython issue 78116). The cost falls on the safe side: the same
    directory reached through a bind mount or another mount resolves to another
    path and is refused, and the refusal names the path to use.

    **Unset, every such use is refused**: a shell without the service's settings
    (a fresh SSH session) cannot tell which registry is the deployment's, and
    accepting any would be the bypass this rule closes. It reads this process's
    environment (or `environ`): a command run with the variable set to another
    directory on purpose uses that one. Like the store's other rules, it
    catches a registry reached by mistake; it does not make the store
    tamper-proof (D2).
    """
    raw = _raw(environ, ENV_REGISTRY)
    if raw is None:
        return (f"{ENV_REGISTRY} is not set here, so which registry is the deployment's cannot "
                f"be told. Set {ENV_REGISTRY} in this environment to the value the service's "
                "environment gives it (deployment guide §4.11: one file that both the service "
                "and this shell read)")
    try:
        configured = check_registry_root(raw)
    except DirectoryRefused as exc:
        return f"{ENV_REGISTRY} is refused: {str(exc).rstrip().rstrip('.')}"
    if Path(registry) != configured:
        return (f"the registry {registry} is not the deployment's: {ENV_REGISTRY} ({raw}) resolves "
                f"to {configured}. Registries are compared by resolved path, so another "
                "directory is another registry whatever it holds, a copy of this one or the "
                "same directory reached through a bind mount or another mount. Use the path "
                f"under {configured}, or, if {registry} is now the deployment's registry, set "
                f"{ENV_REGISTRY} to it for the service and here")
    return None


def intake_directory(environ: Optional[Mapping[str, str]] = None) -> Setting:
    """`SHEPHERD_INTAKE_DIR`: where the operator places `shepherd-cases/1` sources
    to import (D7), checked as the registry is: absolute, existing, outside the
    repository."""
    return _directory_setting(
        environ, ENV_INTAKE_DIR,
        unset=(f"{ENV_INTAKE_DIR} is not set, so no import source is offered. Set it to "
               "the absolute path of a directory outside the repository where you place "
               "shepherd-cases/1 files."),
        check=partial(check_operator_directory, what="the intake directory",
                      holds=PATIENT_LEVEL_DATA, outside_repository=True),
        consequence="No import source is offered until it is corrected.",
    )


def training_evidence_dirs(environ: Optional[Mapping[str, str]] = None) -> Setting:
    """`SHEPHERD_TRAINING_EVIDENCE_DIRS`: R10 evidence directories (the
    measurement's `--training-evidence`), separated by `os.pathsep`.

    Optional: unset or empty is an empty tuple, usable. Empty entries (a
    doubled or trailing separator) are skipped, never read as the working
    directory. Each entry must be an absolute, existing directory; one that is
    not refuses the whole setting (value `None`), so a run never goes ahead on
    part of what the operator listed. Repeats are kept once, by resolved path,
    in the order given.
    """
    raw = _raw(environ, ENV_TRAINING_EVIDENCE_DIRS)
    if raw is None:
        return Setting(ENV_TRAINING_EVIDENCE_DIRS, ())
    resolved = []
    for index, entry in enumerate((e for e in raw.split(os.pathsep) if e), start=1):
        try:
            directory = check_operator_directory(
                entry, what=f"evidence directory {index}", holds="training evidence",
                outside_repository=False,
            )
        except DirectoryRefused as exc:
            return Setting(
                ENV_TRAINING_EVIDENCE_DIRS, None,
                f"{ENV_TRAINING_EVIDENCE_DIRS} is refused: {exc} No evidence directory is "
                "offered until it is corrected.",
            )
        if directory not in resolved:
            resolved.append(directory)
    return Setting(ENV_TRAINING_EVIDENCE_DIRS, tuple(resolved))


def ordering_roles(environ: Optional[Mapping[str, str]] = None) -> Setting:
    """`SHEPHERD_ORDERING_ROLES`: the cohort roles that offer ordering by test
    score (D4; §4.9 question 1), comma-separated, surrounding spaces ignored.

    Default empty: no role offers it. `institutional_acceptance` is refused even
    when listed, and the note says so. Roles are compared exactly, and a role is
    a lowercase identifier (`validate_role`), so an entry that is not one simply
    matches no version.
    """
    raw = _raw(environ, ENV_ORDERING_ROLES) or ""
    listed = [entry.strip() for entry in raw.split(",") if entry.strip()]
    refused = sorted({role for role in listed if role in NEVER_ORDERED_ROLES})
    kept: list = []
    for role in listed:
        if role not in NEVER_ORDERED_ROLES and role not in kept:
            kept.append(role)
    note = None
    if refused:
        note = (
            f"{ENV_ORDERING_ROLES} lists {', '.join(refused)}, which never offers ordering by "
            "test score until the institution's formal evaluation rules exist (§4.9 "
            "question 1); it is ignored."
        )
    return Setting(ENV_ORDERING_ROLES, tuple(kept), None, note)


def offers_ordering(role: Any, listed: Any) -> bool:
    """**The one rule** for whether a cohort version of this role offers
    ordering by test score, given the roles the deployment lists: a role that
    is listed and is never one of `NEVER_ORDERED_ROLES` (§4.9 question 1).
    Pure; D4's ordering (`src.evaluation.ordering.ordering_offered`) and every
    page ask it, so a role added to the constant is refused everywhere at once
    (M5-3 review ARCH-2)."""
    if not isinstance(role, str) or role in NEVER_ORDERED_ROLES:
        return False
    try:
        return role in set(listed or ())
    except TypeError:  # an unhashable entry is no role
        return False


def role_offers_ordering(role: Any, environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether a cohort version with this role offers ordering by test score
    under this deployment's `SHEPHERD_ORDERING_ROLES` (`offers_ordering`)."""
    return offers_ordering(role, ordering_roles(environ).value)


def all_settings(environ: Optional[Mapping[str, str]] = None) -> Tuple[Setting, ...]:
    """Every setting as read now, in the table's order, for a status view."""
    return (
        deployment_mode(environ),
        workspaces_root(environ),
        cohort_registry(environ),
        intake_directory(environ),
        training_evidence_dirs(environ),
        ordering_roles(environ),
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
