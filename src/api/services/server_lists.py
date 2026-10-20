"""
The server's lists: workspaces, their checkpoints, cohort versions and their reports.
=====================================================================================
`docs/working/PLAN_TEST_RESULTS.md` §6 ("M5's boundaries"), D7 and D8:
"checkpoints and cohorts are chosen from server-side lists, never from a path in a
request". Functions over the server's settings (`src.config.server_settings`,
read at each call, never from a request) and the disk, keeping no state between
calls; the API's routes and the WebUI's handlers (from M5-2) both call them, so
the two cannot list different things.

**Items are identified by name and digest, never by a path.** A workspace is its
directory name under ``SHEPHERD_WORKSPACES_ROOT`` and its dataset fingerprint
(the full SHA-256 of its ``split_manifest.json``; a short form is for display
only and never compared); a checkpoint is its workspace, architecture, run and
file name and the SHA-256 of its bytes; a cohort version is its label and the
digest of its ``cohort.json``. No item carries a server path for a client to
send back: `resolve_workspace` and `resolve_checkpoint` turn a name into a path
only by finding it in the server's list again. (A refusal's text may name a
server path, for the operator.)

**One reader per file, each file read once per call:** ``split_manifest.json``
through `read_split_manifest`, the dataset record through
`verify_dataset_record` (its origin and binding state kept apart), the build
record through `read_workspace_build` (never a modification time), each
checkpoint through `read_run_checkpoints` (`read_checkpoint`, weights only),
each version through `list_versions` (`resolve_version`).

**What "trainable" claims, and no more:** the manifest is readable at the
current schema, the dataset record it binds is verified, and the files it binds
are present. The lists hash no graph file; a training run or a measurement
checks every file on the bytes it reads, and may still refuse.

**What the lists open.** The workspace and cohort lists read small JSON files.
The checkpoint list opens every checkpoint inside the workspace, which SPEC_4
classes as checkpoint inspection (C3): a surface offering it over the network is
gated by D8. A file that resolves outside the workspace is listed with that
reason and never opened. No access check is made here; the gate belongs where a
request is handled.

**One unreadable entry never hides the rest.** A workspace, a version, a run or
a checkpoint that cannot be read or checked (a malformed file, a directory
another account made unreadable to this one) is listed with why; only a list
that cannot be listed at all (its setting missing or refused, its directory
unreadable) offers no item, and says why.

**What a list does not say.** Whether a report was kept and whether it is
registered is read by the registration's one helper, passed in as
`report_state` (`stored_report_state` is that helper for a run directory);
nothing here infers either from which files exist.

Module: src/api/services/server_lists.py
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from src.config import server_settings

__all__ = [
    "CheckpointItem",
    "CohortVersionItem",
    "Listing",
    "ReportItem",
    "ServerListRefused",
    "WorkspaceItem",
    "list_checkpoints",
    "list_cohort_versions",
    "list_workspaces",
    "resolve_checkpoint",
    "resolve_workspace",
    "resolve_workspaces",
    "stored_report_state",
]

#: The sample files a generated workspace's manifest binds beside the graph.
_SAMPLE_FILES: Tuple[str, ...] = ("train_samples.json", "val_samples.json")

#: How much of a name from a request a refusal repeats.
_ECHO_LIMIT = 80

#: Reads what a run kept and whether its report is registered, from disk: the
#: registration's one helper, given the run directory.
ReportStateReader = Callable[[Path], Mapping[str, Any]]


class ServerListRefused(ValueError):
    """A name that is not an item of the server's list, or a list the server
    cannot offer (its setting is missing or refused)."""


@dataclass(frozen=True)
class Listing:
    """A list as the server offers it now: its items, or why it offers none
    (`problem`, in words an operator can act on)."""

    items: Tuple[Any, ...]
    problem: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.problem is None

    def as_dict(self, key: str) -> Dict[str, Any]:
        """The list as JSON-ready data: `{key: [item, ...], "problem": ...}`."""
        return {key: [asdict(item) for item in self.items], "problem": self.problem}


@dataclass(frozen=True)
class WorkspaceItem:
    """One workspace under the workspaces root.

    `dataset_record` holds `binding` (verified, mismatch, unrecorded or
    unverifiable; `None` when not read), `origin` (`files` or `synthetic`),
    `sha256` and `problem`: the binding state and the origin are two facts,
    never merged. `build` is `read_workspace_build`'s answer: `state`
    (`recorded` or `not_recorded`), `built_at`, `code_revision` and `reason`.
    `declared_releases` are the dataset record's sources as each declares its
    release (`declared_version`, `None` where a source declares none); empty for
    a synthetic workspace. The counts are the sample counts the manifest records.
    """

    name: str
    dataset_fingerprint: Optional[str]
    trainable: bool
    not_trainable: Optional[str]
    dataset_record: Dict[str, Any]
    build: Dict[str, Any]
    declared_releases: Tuple[Dict[str, Any], ...] = ()
    train_samples: Optional[int] = None
    val_samples: Optional[int] = None


@dataclass(frozen=True)
class CheckpointItem:
    """One checkpoint of a workspace, from one read of its file.

    `validation` is the Trainer's validation figure (metric and value, the
    trainer's top-20 selection metric), never a test result.
    `dataset_fingerprint` is the dataset the checkpoint records it was trained
    on; `trained_on_this_dataset` compares it in full with the workspace's
    fingerprint now (`None` when either is unknown). A file that cannot be read
    is listed with `problem`, never dropped; one that resolves outside the
    workspace is listed with `problem` and every field read from a file
    (digest, size, record) `None`, since it was never opened. `workspace` is the
    name the server lists, which `resolve_checkpoint` takes back.
    """

    workspace: str
    architecture: str
    run: str
    filename: str
    sha256: Optional[str]
    size_bytes: Optional[int]
    checkpoint_kind: Optional[str]
    epoch: Optional[int]
    validation: Optional[Dict[str, Any]]
    dataset_fingerprint: Optional[str]
    trained_on_this_dataset: Optional[bool]
    problem: Optional[str]


@dataclass(frozen=True)
class ReportItem:
    """One run directory of a version's report store. `state` is what the
    report-state reader says of it (kept, registered); `None`, with `problem`,
    when it was not read."""

    run_id: str
    state: Optional[Dict[str, Any]]
    problem: Optional[str]


@dataclass(frozen=True)
class CohortVersionItem:
    """One cohort version in the registry: its label and the digest of its
    `cohort.json` (the version's identity), its role and whether that role
    offers ordering by test score (`SHEPHERD_ORDERING_ROLES`), and the
    aggregates its manifest records. Never its exclusions by case: D7 keeps
    case identifiers on the machine. A version that cannot be read is listed
    with `problem`."""

    label: str
    manifest_sha256: Optional[str]
    role: Optional[str]
    offers_ordering: bool
    source_format: Optional[str]
    source_sha256: Optional[str]
    counts: Optional[Dict[str, Any]]
    graph: Optional[Dict[str, Any]]
    dataset_record: Optional[Dict[str, Any]]
    imported_at: Optional[str]
    problem: Optional[str]
    reports: Tuple[ReportItem, ...] = ()
    reports_problem: Optional[str] = None


def _echo(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= _ECHO_LIMIT else text[:_ECHO_LIMIT - 3] + "..."


def _inside(path: Path, root: Path) -> bool:
    return path != root and root in path.parents


def _as_dict(value: Any) -> Optional[Dict[str, Any]]:
    return dict(value) if isinstance(value, dict) else None


def _count(value: Any) -> Optional[int]:
    """A whole number of at least 0, else `None` (`True` is an `int`)."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


# ---------------------------------------------------------------------------
# Workspaces
# ---------------------------------------------------------------------------
def _workspace_root(environ: Optional[Mapping[str, str]]) -> Path:
    setting = server_settings.workspaces_root(environ)
    if not setting.available:
        raise ServerListRefused(setting.problem)
    return setting.value


def _is_workspace_entry(entry: Path) -> bool:
    """A subdirectory not named with a leading dot (a hidden directory, an
    editor's or a tool's). An entry whose kind cannot be checked (a link into a
    directory this account may not enter) is kept, so its item says why rather
    than the whole list failing or the entry vanishing."""
    if entry.name.startswith("."):
        return False
    try:
        return entry.is_dir()
    except OSError:
        return True


def _workspace_entries(root: Path) -> List[Path]:
    """The root's subdirectories, by name."""
    try:
        entries = [p for p in root.iterdir() if _is_workspace_entry(p)]
    except OSError as exc:
        raise ServerListRefused(
            f"the workspaces root {root} cannot be listed ({type(exc).__name__}: {exc})"
        ) from None
    return sorted(entries, key=lambda p: p.name)


def _not_read(problem: str) -> Dict[str, Any]:
    return {"binding": None, "origin": None, "sha256": None, "problem": problem}


def _workspace_item(root: Path, entry: Path) -> WorkspaceItem:
    from src.kg.artifacts import (
        GRAPH_ARTIFACTS,
        DatasetRecordRefused,
        manifest_section,
        read_split_manifest,
        read_workspace_build,
        verify_dataset_record,
    )

    name = entry.name
    try:
        resolved = entry.resolve()
    except (OSError, RuntimeError) as exc:
        problem = f"{name} cannot be resolved ({type(exc).__name__}: {exc})"
        return WorkspaceItem(name, None, False, problem, _not_read(problem),
                             {"state": "not_recorded", "built_at": None,
                              "code_revision": None, "reason": problem})
    if not _inside(resolved, root):
        problem = (f"{name} resolves to {resolved}, outside the workspaces root {root}, so it "
                   "is not in the server's one workspace namespace")
        return WorkspaceItem(name, None, False, problem, _not_read(problem),
                             {"state": "not_recorded", "built_at": None,
                              "code_revision": None, "reason": problem})

    try:
        manifest = read_split_manifest(resolved)
    except (ValueError, OSError) as exc:
        return WorkspaceItem(
            name, None, False, str(exc),
            _not_read(f"not read: the workspace has no readable split_manifest.json ({exc})"),
            read_workspace_build(resolved, None)._asdict(),
        )
    fingerprint = manifest.identity.sha256

    releases: Tuple[Dict[str, Any], ...] = ()
    try:
        artifacts = manifest_section(manifest.manifest, manifest.identity.path, "artifacts")
        record = verify_dataset_record(manifest, artifacts.get("kg"))
    except DatasetRecordRefused as exc:
        dataset = {"binding": exc.binding, "origin": exc.origin, "sha256": None,
                   "problem": str(exc)}
    except ValueError as exc:
        dataset = _not_read(str(exc))
    else:
        dataset = {"binding": record.binding, "origin": record.origin,
                   "sha256": record.identity.sha256, "problem": None}
        releases = tuple({"role": source["role"], "declared_version": source["declared_version"]}
                         for source in record.record["sources"])

    missing, unchecked = [], []
    for filename in (*GRAPH_ARTIFACTS.values(), *_SAMPLE_FILES):
        try:
            if not (resolved / filename).is_file():
                missing.append(filename)
        except OSError as exc:  # `is_file` raises EACCES rather than answer
            unchecked.append(f"{filename}: {type(exc).__name__}")
    if dataset["problem"] is not None:
        not_trainable: Optional[str] = dataset["problem"]
    elif missing or unchecked:
        not_trainable = "; ".join(
            ([f"files its split_manifest.json binds are missing ({', '.join(missing)})"]
             if missing else [])
            + ([f"files its split_manifest.json binds cannot be checked "
                f"({', '.join(unchecked)})"] if unchecked else []))
    else:
        not_trainable = None

    generation = manifest.manifest.get("generation")
    generation = generation if isinstance(generation, dict) else {}
    return WorkspaceItem(
        name=name,
        dataset_fingerprint=fingerprint,
        trainable=not_trainable is None,
        not_trainable=not_trainable,
        dataset_record=dataset,
        build=read_workspace_build(resolved, fingerprint)._asdict(),
        declared_releases=releases,
        train_samples=_count(generation.get("num_train")),
        val_samples=_count(generation.get("num_val")),
    )


def list_workspaces(*, environ: Optional[Mapping[str, str]] = None) -> Listing:
    """Every workspace under `SHEPHERD_WORKSPACES_ROOT`, by name, with its
    dataset identity (fingerprint, build time, dataset record), its sources'
    declared releases, its sample counts and whether it is trainable.

    No root configured, or one that cannot be listed: no item, and why.
    """
    try:
        root = _workspace_root(environ)
        entries = _workspace_entries(root)
    except ServerListRefused as exc:
        return Listing((), str(exc))
    return Listing(tuple(_workspace_item(root, entry) for entry in entries))


def _find_workspace(name: Any, environ: Optional[Mapping[str, str]]) -> Tuple[str, Path]:
    """The listed name and the resolved directory of the workspace the server
    lists as `name`. The name is the list's own, which an item carries: a
    workspace that is a link inside the root (``current -> builds/...``) is
    named as listed, never by its target's directory name, so the item's name
    resolves again."""
    root = _workspace_root(environ)
    for entry in _workspace_entries(root):
        if entry.name == name:
            try:
                resolved = entry.resolve()
            except (OSError, RuntimeError) as exc:
                raise ServerListRefused(
                    f"the workspace {entry.name!r} cannot be resolved "
                    f"({type(exc).__name__}: {exc})") from None
            if not _inside(resolved, root):
                raise ServerListRefused(
                    f"the workspace {entry.name!r} resolves outside the workspaces root "
                    f"{root}, so it is not in the server's one workspace namespace")
            return entry.name, resolved
    raise ServerListRefused(
        f"{_echo(name)} is not a workspace the server lists under {root}")


def resolve_workspace(name: Any, *, environ: Optional[Mapping[str, str]] = None) -> Path:
    """The resolved directory of the workspace the server lists as `name`.

    Found by listing the root again, never by joining a request's text onto a
    path, so only a listed name resolves; and the directory must resolve inside
    the root.

    Raises:
        ServerListRefused: no workspace root, or `name` is not a listed workspace.
    """
    return _find_workspace(name, environ)[1]


def resolve_workspaces(*, environ: Optional[Mapping[str, str]] = None) -> Dict[str, Path]:
    """Every listed workspace's resolved directory, by its listed name, from one
    listing of the root (M5-2): what `resolve_workspace` answers for each name,
    for a caller that needs many names at once (one listing, not one per name).
    A name that cannot be resolved, or resolves outside the root, is left out,
    as `resolve_workspace` refuses it.

    Raises:
        ServerListRefused: no workspace root, or the root cannot be listed.
    """
    root = _workspace_root(environ)
    resolved: Dict[str, Path] = {}
    for entry in _workspace_entries(root):
        try:
            path = entry.resolve()
        except (OSError, RuntimeError):
            continue
        if _inside(path, root):
            resolved[entry.name] = path
    return resolved


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def list_checkpoints(
    workspace: Any, architecture: Optional[str] = None, *,
    environ: Optional[Mapping[str, str]] = None,
) -> Listing:
    """The checkpoints of a listed workspace's runs, every architecture's or
    one's, in the listing's order (latest run first, newest file first), each
    from one read of its file (`read_run_checkpoints`).

    Opens every checkpoint inside the workspace (SPEC_4 C3). One that
    resolves outside it is listed with that reason and never opened
    (`read_run_checkpoints`' `within`). Each item names the workspace as the
    list does, so `resolve_checkpoint` finds it again by that name.
    """
    from src.kg.artifacts import read_split_manifest
    from src.utils.checkpoint_paths import (
        normalize_conv_type,
        read_run_checkpoints,
        run_architecture,
    )

    try:
        name, path = _find_workspace(workspace, environ)
        conv_type = normalize_conv_type(architecture) if architecture else None
    except ValueError as exc:  # ServerListRefused, or an unsupported architecture
        return Listing((), str(exc))
    try:
        fingerprint: Optional[str] = read_split_manifest(path).identity.sha256
    except (ValueError, OSError):
        fingerprint = None
    try:
        entries = read_run_checkpoints(path, conv_type, within=path)
    except OSError as exc:  # a runs directory this account may not enter
        return Listing((), (f"the runs of the workspace {name!r} cannot be listed "
                            f"({type(exc).__name__}: {exc.strerror or exc})"))

    items = []
    for entry in entries:
        recorded = entry.dataset_fingerprint
        items.append(CheckpointItem(
            workspace=name,
            architecture=run_architecture(entry.run_dir),
            run=entry.run_dir.name,
            filename=entry.path.name,
            sha256=entry.sha256,
            size_bytes=entry.size_bytes,
            checkpoint_kind=(entry.checkpoint_kind
                             if isinstance(entry.checkpoint_kind, str) else None),
            epoch=_count(entry.epoch),
            validation=({"metric": entry.validation[0], "value": entry.validation[1]}
                        if entry.validation else None),
            dataset_fingerprint=recorded,
            trained_on_this_dataset=(recorded == fingerprint
                                     if recorded is not None and fingerprint is not None
                                     else None),
            problem=entry.problem,
        ))
    return Listing(tuple(items))


def resolve_checkpoint(
    workspace: Any, architecture: Any, run: Any, filename: Any, *,
    environ: Optional[Mapping[str, str]] = None,
) -> Path:
    """The path of the checkpoint the server lists as `run`/`filename` of
    `architecture` in `workspace`, found in the listing again; reads no file.

    The caller checks the bytes it then reads against the digest it was shown
    (a measurement's expected checkpoint digest, a load's served identity).

    Raises:
        ServerListRefused: the workspace or the checkpoint is not listed, or the
            file resolves outside the workspace.
    """
    from src.utils.checkpoint_paths import list_run_checkpoints, normalize_conv_type

    name, path = _find_workspace(workspace, environ)
    # An empty name would be read as the default architecture; a request names one.
    if not (isinstance(architecture, str) and architecture.strip()):
        raise ServerListRefused(f"{_echo(architecture)} is not an architecture")
    try:
        conv_type = normalize_conv_type(architecture)
    except ValueError as exc:
        raise ServerListRefused(str(exc)) from None
    try:
        listed = list_run_checkpoints(path, conv_type)
    except OSError as exc:  # a runs directory this account may not enter
        raise ServerListRefused(
            f"the runs of the workspace {name!r} cannot be listed "
            f"({type(exc).__name__}: {exc.strerror or exc})") from None
    for run_dir, checkpoint in listed:
        if run_dir.name == run and checkpoint.name == filename:
            try:
                resolved = checkpoint.resolve()
            except (OSError, RuntimeError) as exc:  # a link loop: RuntimeError before 3.13
                raise ServerListRefused(
                    f"{run_dir.name}/{checkpoint.name} cannot be resolved "
                    f"({type(exc).__name__}), so it is not offered") from None
            if not _inside(resolved, path):
                raise ServerListRefused(
                    f"{run_dir.name}/{checkpoint.name} resolves outside the workspace "
                    f"{name!r}, so it is not offered")
            return checkpoint
    raise ServerListRefused(
        f"no checkpoint {_echo(filename)} in run {_echo(run)} of {conv_type} is listed in "
        f"the workspace {name!r}")


# ---------------------------------------------------------------------------
# Cohort versions and their reports
# ---------------------------------------------------------------------------
def _report_items(
    version: Any, report_state: Optional[ReportStateReader]
) -> Tuple[Tuple[ReportItem, ...], Optional[str]]:
    from src.evaluation.cohort_registry import RegistryRefused, list_report_runs

    try:
        runs = list_report_runs(version)
    except RegistryRefused as exc:
        return (), str(exc)
    items = []
    for run in runs:
        if run.problem is not None:
            items.append(ReportItem(run.run_id, None, run.problem))
        elif report_state is None:
            items.append(ReportItem(run.run_id, None, (
                "kept and registered are not read here: no report-state reader was given")))
        else:
            try:
                items.append(ReportItem(run.run_id, dict(report_state(run.directory)), None))
            except Exception as exc:  # noqa: BLE001 — one unreadable run must not hide the rest
                items.append(ReportItem(run.run_id, None, (
                    f"its state could not be read ({type(exc).__name__}: {exc})")))
    return tuple(items), None


def _version_item(listed: Any, report_state: Optional[ReportStateReader],
                  environ: Optional[Mapping[str, str]]) -> CohortVersionItem:
    if listed.version is None:
        return CohortVersionItem(listed.label, None, None, False, None, None, None, None,
                                 None, None, listed.problem)
    manifest = listed.version.manifest
    source = _as_dict(manifest.get("source")) or {}
    imported = _as_dict(manifest.get("imported")) or {}
    role = manifest.get("role")
    reports, reports_problem = _report_items(listed.version, report_state)
    return CohortVersionItem(
        label=listed.label,
        manifest_sha256=listed.version.identity.sha256,
        role=role,
        offers_ordering=server_settings.role_offers_ordering(role, environ),
        source_format=source.get("format"),
        source_sha256=source.get("sha256"),
        counts=_as_dict(manifest.get("counts")),
        graph=_as_dict(manifest.get("graph")),
        dataset_record=_as_dict(manifest.get("dataset_record")),
        imported_at=imported.get("at"),
        problem=None,
        reports=reports,
        reports_problem=reports_problem,
    )


def stored_report_state(run_directory: Path) -> Dict[str, Any]:
    """Whether the report in this run directory of a report store is kept and
    whether it is registered, as the registration's one helper reads them
    (`src.evaluation.registration.report_state`): kept from the store,
    registered from the ledger beside the checkpoint the report records it
    measured, since a list knows no checkpoint file. So ``registered`` is
    ``registered`` when that ledger names the report, and otherwise
    ``cannot_be_checked``, never ``not_registered``: a registration filed
    beside another copy of the same weights is not looked for, and the reason
    says which ledger was. JSON-ready: ``kept`` and
    ``registered`` (one of the registration's states each), their reasons,
    the report's digest and the ledger record's sequence. No field is a server
    path (a reason may name one, for the operator)."""
    from src.evaluation.cohort_registry import REPORT_FILE
    from src.evaluation.registration import report_state

    state = report_state(Path(run_directory) / REPORT_FILE)
    return {"kept": state.kept, "why_kept": state.why_kept,
            "registered": state.registered, "why_registered": state.why_registered,
            "report_sha256": state.report_sha256, "sequence": state.sequence}


def list_cohort_versions(
    *, environ: Optional[Mapping[str, str]] = None,
    report_state: Optional[ReportStateReader] = None,
) -> Listing:
    """Every cohort version in `SHEPHERD_REGISTRY`, by label, with its
    manifest's digest, role, aggregates, and its report store's runs.

    `report_state`, given a run directory, says what that run kept and whether
    its report is registered: the registration's one helper
    (`stored_report_state`, which the API route passes). Without it, each run
    is listed with no state and says so.

    No registry configured, or one refused: no item, and why.
    """
    from src.evaluation.cohort_registry import RegistryRefused, list_versions

    setting = server_settings.cohort_registry(environ)
    if not setting.available:
        return Listing((), setting.problem)
    try:
        listed = list_versions(setting.value)
    except RegistryRefused as exc:
        return Listing((), str(exc))
    return Listing(tuple(_version_item(entry, report_state, environ) for entry in listed))
