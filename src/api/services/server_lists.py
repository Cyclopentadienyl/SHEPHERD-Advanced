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

**D5's state of each run's key** (`ReportItem.contest`) comes with the run's
state, from one scan of the version's store (`record_states.run_contest`, the
one layer that turns a scan into a shown D5 state, for records too): the one
run listing that `GET /api/v1/cohorts` and the Test page share, so the two
cannot disagree about a run, and neither judges D5 itself. It is bound to the
bytes the run's state read, so one item never describes two versions of a
report. The open route's listing leaves out the problems' words, which may name
a server path; the Test page's keeps them for the operator. A version whose
runs' states are read is scanned once per call: each kept report read once
more, on an open route (the trade BACKLOG 36 accepted for the loaded-model
view).

**Import sources** (M5-2, the Test page; D7): the ``*.json`` files the operator
placed directly in ``SHEPHERD_INTAKE_DIR`` (`list_intake_files`), by file name;
a request names one and `resolve_intake_file` finds it in that listing again,
refusing any other name and a link that leaves the directory. Nothing is read
from a file to list it (its size and modification time come from the
directory); the importer reads it once when the operator previews or commits.
**One cohort version by its label** (`resolve_cohort_version`,
`cohort_version_item`) and **one run's report** (`resolve_report`) are found the
same way: through the registry's one reader and the version's listed run
directories, never by joining a request's text onto a path.

Module: src/api/services/server_lists.py
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from src.config import server_settings

__all__ = [
    "FAMILY_GROUPS",
    "CheckpointItem",
    "CohortVersionItem",
    "FamilyRow",
    "IntakeFileItem",
    "Listing",
    "ParentLookup",
    "ReportItem",
    "ServerListRefused",
    "WorkspaceItem",
    "cohort_version_item",
    "family_rows",
    "list_checkpoints",
    "list_cohort_versions",
    "list_family",
    "list_intake_files",
    "list_workspaces",
    "locate_checkpoint",
    "record_matches",
    "resolve_checkpoint",
    "resolve_cohort_version",
    "resolve_intake_file",
    "resolve_report",
    "resolve_workspace",
    "resolve_workspaces",
    "stored_report_state",
    "task_readiness",
    "version_item",
]

#: The files an intake listing offers: shepherd-cases/1 sources are JSON.
_INTAKE_SUFFIX = ".json"

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
    `completed_epochs` and `end` are the file's progress record (epochs
    completed; how its stage ended), `None` where the file records none (a
    weights-only file, an older format): `epoch` is the record's epoch index,
    never adjusted into a count.

    The model record's summary comes from the same read, through the one
    reader of records (`src.models.bundle.record.listed_record`): `record` is
    "model", "history" (no model spec: listed, never run) or "refused" (with
    `record_problem`); a model adds its learning stage, task, operation,
    family, parent (`parent_sha256`, the exact parent file, and `parent_run`),
    encoder kind, the names of its spec's fields that differ from the
    reference recipe (`differences`, sorted; empty for the reference) and
    whether there are none (`reference`). All `None` for a file that was not
    read.
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
    completed_epochs: Optional[int] = None
    end: Optional[str] = None
    record: Optional[str] = None
    record_problem: Optional[str] = None
    learning_stage: Optional[str] = None
    task: Optional[str] = None
    operation: Optional[str] = None
    family_id: Optional[str] = None
    parent_sha256: Optional[str] = None
    parent_run: Optional[str] = None
    encoder: Optional[str] = None
    reference: Optional[bool] = None
    differences: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class ReportItem:
    """One run directory of a version's report store. `state` is what the
    report-state reader says of it (kept, registered); `None`, with `problem`,
    when it was not read. `contest` is D5's state of the key its report is
    under, for the bytes `state` read (`record_states.run_contest`): ``state``
    (uncontested, contested, cannot_be_checked), ``differing``,
    ``disagreeing_runs`` and ``problems`` (empty but on an operator's listing);
    `None` when the run's state was not read or it kept no report."""

    run_id: str
    state: Optional[Dict[str, Any]]
    problem: Optional[str]
    contest: Optional[Dict[str, Any]] = None


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


@dataclass(frozen=True)
class IntakeFileItem:
    """One import source in the intake directory: its file name (what a request
    names, and what the version manifest records as the source's name), its
    size and its modification time (UTC, ISO 8601; descriptive only, nothing
    compares it). One that is not offered says why (`problem`: a link that
    leaves the directory, an entry that is not a file or cannot be checked),
    with its size and time `None`."""

    name: str
    size_bytes: Optional[int]
    modified_at: Optional[str]
    problem: Optional[str]


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


#: Why a task cannot be started on a workspace (`task_readiness`), in words
#: that name no server path: the list is open, where a refusal is not.
READINESS_WORDS = {
    "not_listed": "not a listed workspace",
    "not_trainable": "dataset not trainable",
    "absent": "no functional fixture",
    "another_build": "fixture from another build",
    "unreadable": "fixture not readable",
}


def task_readiness(workspace: Any, *, environ: Optional[Mapping[str, str]] = None
                   ) -> Dict[str, Optional[str]]:
    """Whether each task can be finetuned on the listed workspace `workspace`
    now, as ``{task: None | why}`` (`READINESS_WORDS`), from small JSON files
    only (the workspace's manifest and dataset record, then the fixture
    record), so a page offers it without D8.

    Every task needs a trainable workspace (`WorkspaceItem.trainable`); the
    disease task needs nothing more. A task that trains on the gene task's
    functional fixture (`FIXTURE_TASKS`) also needs its record derived from this
    build of the workspace (`read_fixture_record` against the fingerprint the
    item read): "ready" says no more than that. The sample files are hashed
    by the training process, which may still refuse them, and a Fine-tune start
    checks the record again (`TrainingManager.start_finetune`).
    """
    from src.kg.task_fixture import FIXTURE_TASKS, TaskFixtureError, read_fixture_record
    from src.models.bundle.spec import TASKS

    try:
        root = _workspace_root(environ)
        entry = next((e for e in _workspace_entries(root) if e.name == workspace), None)
    except ServerListRefused:
        entry = None
    if entry is None:
        return dict.fromkeys(TASKS, READINESS_WORDS["not_listed"])
    item = _workspace_item(root, entry)
    if not item.trainable:
        return dict.fromkeys(TASKS, READINESS_WORDS["not_trainable"])
    try:
        read_fixture_record(entry.resolve(), manifest_sha256=item.dataset_fingerprint)
    except TaskFixtureError as exc:
        fixture: Optional[str] = READINESS_WORDS[exc.state]
    except (OSError, RuntimeError):  # resolved by the item a moment ago
        fixture = READINESS_WORDS["unreadable"]
    else:
        fixture = None
    return {task: fixture if task in FIXTURE_TASKS else None for task in TASKS}


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def list_checkpoints(
    workspace: Any, architecture: Optional[str] = None, *,
    task: Optional[str] = None, learning_stage: Optional[str] = None,
    run: Optional[str] = None, environ: Optional[Mapping[str, str]] = None,
) -> Listing:
    """The checkpoints of a listed workspace's runs, every architecture's or
    one's, in the listing's order (latest run first, newest file first), each
    from one read of its file (`read_run_checkpoints`).

    Opens every checkpoint inside the workspace (SPEC_4 C3). One that
    resolves outside it is listed with that reason and never opened
    (`read_run_checkpoints`' `within`). Each item names the workspace as the
    list does, so `resolve_checkpoint` finds it again by that name.

    `task` and `learning_stage` keep the files whose model record names them
    (`listed_record`, `record_matches`); a file without a readable record
    names neither, so either filter leaves it out. `run` lists that run
    directory's files alone (`read_run_checkpoints`' `run`): only they are
    opened.
    """
    from src.kg.artifacts import read_split_manifest
    from src.models.bundle.record import listed_record
    from src.utils.checkpoint_paths import (
        normalize_run_layout_key,
        read_run_checkpoints,
        run_architecture,
    )

    try:
        name, path = _find_workspace(workspace, environ)
        conv_type = normalize_run_layout_key(architecture) if architecture else None
        _check_record_filter(task, learning_stage)
    except ValueError as exc:  # ServerListRefused, or an unsupported architecture
        return Listing((), str(exc))
    try:
        fingerprint: Optional[str] = read_split_manifest(path).identity.sha256
    except (ValueError, OSError):
        fingerprint = None
    try:
        entries = read_run_checkpoints(path, conv_type, within=path, run=run)
    except OSError as exc:  # a runs directory this account may not enter
        return Listing((), (f"the runs of the workspace {name!r} cannot be listed "
                            f"({type(exc).__name__}: {exc.strerror or exc})"))

    items = []
    for entry in entries:
        recorded = entry.dataset_fingerprint
        read = entry.sha256 is not None and entry.problem is None
        summary = listed_record(entry.model_record) if read else {}
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
            completed_epochs=_count(entry.completed_epochs),
            end=entry.end,
            **summary,
        ))
    return Listing(tuple(item for item in items if record_matches(item, task, learning_stage)))


def _check_record_filter(task: Optional[str], learning_stage: Optional[str]) -> None:
    from src.models.bundle.spec import LEARNING_STAGES, TASKS

    if task is not None and task not in TASKS:
        raise ServerListRefused(f"{_echo(task)} is not a task; one of {list(TASKS)}")
    if learning_stage is not None and learning_stage not in LEARNING_STAGES:
        raise ServerListRefused(
            f"{_echo(learning_stage)} is not a learning stage; one of {list(LEARNING_STAGES)}")


def record_matches(item: Any, task: Optional[str] = None,
                   learning_stage: Optional[str] = None) -> bool:
    """Whether a listed checkpoint's model record names `task` and
    `learning_stage` (each None: any): **the one predicate** the server's lists
    filter by and the WebUI's lists show by, so the two never keep different
    files. `item` is a `CheckpointItem` or its JSON form (`asdict`)."""
    def field(name: str) -> Any:
        return item.get(name) if isinstance(item, Mapping) else getattr(item, name, None)

    return ((task is None or field("task") == task)
            and (learning_stage is None or field("learning_stage") == learning_stage))


#: A parent looked up outside the listed workspace (`family_rows`): given the
#: run a child records and the SHA-256 of the exact parent file, ``("found",
#: workspace)`` for another listed workspace whose run of that name holds those
#: bytes, ``("unread", workspace)`` for one whose run holds files that could not
#: be read (so whether they are the parent cannot be said), ``("changed",
#: workspace)`` for one whose run was read and does not hold them, or
#: ``("absent", None)``.
ParentLookup = Callable[[str, str], Tuple[str, Optional[str]]]

#: The groups of a family view, in order: the model families, the files
#: without a model spec (history, never run) and the files whose record or
#: read was refused (with why).
FAMILY_GROUPS = ("family", "history", "refused")


@dataclass(frozen=True)
class FamilyRow:
    """One file of a workspace's family view (`family_rows`).

    `group` is one of `FAMILY_GROUPS`. In a family, `depth` counts the listed
    ancestors above the file. `parent_state` says where the exact parent file
    its record names is: "listed" (`parent` is that file's key,
    ``architecture/run/file``), "elsewhere" (`parent_workspace`, another listed
    workspace, holds those bytes) or "unavailable" (`parent_problem` says why);
    `None` for a file with no parent. `context` marks a row a filter kept only
    because a kept file descends from it (a task's base)."""

    item: CheckpointItem
    group: str
    family_id: Optional[str] = None
    depth: int = 0
    parent: Optional[str] = None
    parent_state: Optional[str] = None
    parent_workspace: Optional[str] = None
    parent_problem: Optional[str] = None
    context: bool = False


def _row_key(item: CheckpointItem) -> str:
    return f"{item.architecture}/{item.run}/{item.filename}"


def _not_held(run: str, sha256: str, workspace: Optional[str] = None) -> str:
    where = f" in the workspace {workspace!r}" if workspace is not None else ""
    return (f"the run {run}{where} no longer holds the parent's bytes ({sha256[:12]}): "
            "rewritten or removed")


def _parent_not_read(run: str, sha256: str, workspace: Optional[str] = None,
                     problem: Optional[str] = None) -> str:
    where = f" in the workspace {workspace!r}" if workspace is not None else ""
    why = f" ({problem})" if problem else ""
    return (f"the run {run}{where} holds files that could not be read{why}, so whether "
            f"one is the parent ({sha256[:12]}) cannot be said")


def family_rows(
    items: Tuple[CheckpointItem, ...], *, lookup: Optional[ParentLookup] = None,
    task: Optional[str] = None, learning_stage: Optional[str] = None,
) -> Tuple[FamilyRow, ...]:
    """A workspace's checkpoints as families (README §1, "Parents and the
    family tree"), from the items of one listing (`list_checkpoints`), so the
    tree applies no reading or record rule of its own.

    **Each child sits under the exact file its record names**, matched by
    SHA-256, never by run or name: a run's other files are other bytes. A
    child whose parent is not listed here is looked up by `lookup` (another
    listed workspace, `list_family`), and otherwise reads "parent
    unavailable" with why. Families are grouped by the family id of their
    topmost listed file, the family with the newest file first; within one,
    each file is followed by its children, in listing order. Files without a
    model spec are the history group, refused ones the last group.

    `task` and `learning_stage` keep the matching model files and, as context
    rows, the listed files they descend from (a task's base); the history and
    refused groups name neither, so a filter leaves them out.
    """
    lookup = lookup or (lambda run, sha256: ("absent", None))
    models = [item for item in items if item.record == "model"]
    by_digest: Dict[str, List[CheckpointItem]] = {}
    for item in models:
        by_digest.setdefault(item.sha256, []).append(item)
    runs_listed = {item.run for item in items}
    #: Per run, the first problem of a file its listing did not read.
    unread_in: Dict[str, str] = {}
    for item in items:
        if item.sha256 is None or item.problem is not None:
            unread_in.setdefault(item.run, item.problem or "not read")

    parent_of: Dict[str, CheckpointItem] = {}
    placed: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = {}
    for item in models:
        key, sha256, run = _row_key(item), item.parent_sha256, item.parent_run
        if sha256 is None:
            placed[key] = (None, None, None)
            continue
        listed = by_digest.get(sha256, [])
        listed = [p for p in listed if p.run == run] or listed
        if listed:
            parent_of[key] = listed[0]
            placed[key] = ("listed", None, None)
        elif not isinstance(run, str) or not run:
            # The record reader requires the run; nothing is looked up without it.
            placed[key] = ("unavailable", None,
                           f"the parent ({sha256[:12]}) is recorded without its run")
        elif run in unread_in:
            placed[key] = ("unavailable", None,
                           _parent_not_read(run, sha256, problem=unread_in[run]))
        elif run in runs_listed:
            placed[key] = ("unavailable", None, _not_held(run, sha256))
        else:
            state, where = lookup(run, sha256)
            placed[key] = (
                ("elsewhere", where, None) if state == "found" else
                ("unavailable", where, _parent_not_read(run, sha256, where))
                if state == "unread" else
                ("unavailable", where, _not_held(run, sha256, where)) if state == "changed" else
                ("unavailable", None,
                 f"the parent run {run} ({sha256[:12]}) is not in a listed workspace"))

    children: Dict[str, List[CheckpointItem]] = {}
    for item in models:
        if _row_key(item) in parent_of:
            children.setdefault(_row_key(parent_of[_row_key(item)]), []).append(item)

    def top(item: CheckpointItem) -> CheckpointItem:
        seen = {_row_key(item)}
        while _row_key(item) in parent_of and _row_key(parent_of[_row_key(item)]) not in seen:
            item = parent_of[_row_key(item)]
            seen.add(_row_key(item))
        return item

    families: Dict[str, List[FamilyRow]] = {}
    shown: set = set()

    def emit(start: CheckpointItem, family: str) -> None:
        stack = [(start, 0)]
        while stack:
            item, depth = stack.pop()
            key = _row_key(item)
            if key in shown:
                continue
            shown.add(key)
            state, where, problem = placed[key]
            if depth:  # under the listed parent
                parent, state = _row_key(parent_of[key]), "listed"
            else:  # topmost: no parent listed (a record naming its descendant shows none)
                parent, state = None, (state if state != "listed" else None)
            families[family].append(FamilyRow(item, "family", family, depth, parent, state,
                                              where, problem))
            stack.extend((child, depth + 1) for child in reversed(children.get(key, [])))

    def family_of(item: CheckpointItem) -> str:
        root = top(item)
        return root.family_id or root.run

    for item in models:  # the family with the newest file first
        families.setdefault(family_of(item), [])
    for item in models:  # within one, the topmost files in listing order
        if _row_key(item) not in parent_of:
            emit(item, family_of(item))
    for item in models:  # a record naming its own descendant: from its topmost file
        if _row_key(item) not in shown:
            emit(top(item), family_of(item))

    rows = [row for family in families.values() for row in family]
    if task is not None or learning_stage is not None:
        by_key = {_row_key(row.item): row for row in rows}
        keep = {key for key, row in by_key.items()
                if record_matches(row.item, task, learning_stage)}
        needed = set(keep)
        for key in keep:
            parent = by_key[key].parent
            while parent is not None and parent not in needed:
                needed.add(parent)
                parent = by_key[parent].parent
        return tuple(replace(row, context=_row_key(row.item) not in keep)
                     for row in rows if _row_key(row.item) in needed)
    return tuple(rows) + tuple(
        FamilyRow(item, "history") for item in items if item.record == "history") + tuple(
        FamilyRow(item, "refused") for item in items if item.record not in ("model", "history"))


def list_family(
    workspace: Any, *, task: Optional[str] = None, learning_stage: Optional[str] = None,
    listing: Optional[Listing] = None, environ: Optional[Mapping[str, str]] = None,
) -> Listing:
    """A listed workspace's checkpoints as families (`family_rows`), from one
    listing (`list_checkpoints`). A parent outside the workspace is looked for
    in the other listed workspaces only, by the run its child records, and
    matched by its SHA-256 (provenance §9.3, "Parents in another workspace"):
    only that run of each is read, and no other root is searched.

    `listing` is that workspace's unfiltered `list_checkpoints` listing when
    the caller already holds it: the tree is drawn from it and nothing is
    listed again, so a page's list and its tree come from one read (README
    §1)."""
    try:
        _check_record_filter(task, learning_stage)
    except ValueError as exc:
        return Listing((), str(exc))
    if listing is None:
        listing = list_checkpoints(workspace, environ=environ)
    if not listing.available or not listing.items:
        return listing
    return Listing(family_rows(listing.items,
                               lookup=_parent_lookup(listing.items[0].workspace, environ),
                               task=task, learning_stage=learning_stage))


def _parent_lookup(home: str, environ: Optional[Mapping[str, str]]) -> ParentLookup:
    """`ParentLookup` over the listed workspaces other than `home`, by name:
    each one's run of the recorded name, read through the one reader within
    that workspace; nothing else is opened."""
    from src.utils.checkpoint_paths import read_run_checkpoints

    listed: Dict[str, Path] = {}

    def lookup(run: str, sha256: str) -> Tuple[str, Optional[str]]:
        if not listed:  # one listing of the root per view
            try:
                listed.update(resolve_workspaces(environ=environ))
            except ServerListRefused:
                return "absent", None
        unread: Optional[str] = None
        changed: Optional[str] = None
        for name, path in sorted(listed.items()):
            if name == home:
                continue
            try:
                entries = read_run_checkpoints(path, within=path, run=run)
            except OSError:
                continue
            if any(e.sha256 == sha256 and e.problem is None for e in entries):
                return "found", name
            if any(e.sha256 is None or e.problem is not None for e in entries):
                unread = unread or name
            elif entries:
                changed = changed or name
        if unread is not None:
            return "unread", unread
        return ("changed", changed) if changed is not None else ("absent", None)

    return lookup


def resolve_checkpoint(
    workspace: Any, architecture: Any, run: Any, filename: Any, *,
    environ: Optional[Mapping[str, str]] = None,
) -> Path:
    """The path of the checkpoint the server lists as `run`/`filename` of
    `architecture` in `workspace`, found in the listing again; reads no file.

    The caller's reader compares the bytes it reads with the digest it was
    shown (a measurement's `--expect-checkpoint-sha256`, a load's
    `expected_checkpoint_sha256`, a continuation's parent, the Diagnosis card's
    served digest), on that read, before the bytes are parsed
    (`read_checkpoint(expected_sha256=)`).

    Raises:
        ServerListRefused: the workspace or the checkpoint is not listed, or the
            file resolves outside the workspace.
    """
    from src.utils.checkpoint_paths import list_run_checkpoints, normalize_run_layout_key

    name, path = _find_workspace(workspace, environ)
    # An empty name would be read as the default architecture; a request names one.
    if not (isinstance(architecture, str) and architecture.strip()):
        raise ServerListRefused(f"{_echo(architecture)} is not an architecture")
    try:
        # The run layout's key: a ShepherdGNN conv type or a model bundle's
        # encoder kind (``gatv2``), as the listing files and names them.
        conv_type = normalize_run_layout_key(architecture)
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


def locate_checkpoint(
    workspace: Any, typed: Any, *, environ: Optional[Mapping[str, str]] = None,
) -> Tuple[str, str, str]:
    """``(architecture, run, filename)`` of the checkpoint the server lists in
    `workspace` whose file is the one the operator `typed` (a path, relative to
    this process's directory or absolute, compared after resolving both): a
    typed path only *locates* a listed item, which `resolve_checkpoint` then
    finds again by those names (`PLAN_TEST_RESULTS.md` D8, 2026-10-19). Reads
    no file: the listing and the paths' resolution only.

    Raises:
        ServerListRefused: the workspace is not listed, or no checkpoint it
            lists is that file (an unlisted file is never opened). The reason
            does not repeat the typed path.
    """
    from src.utils.checkpoint_paths import list_run_checkpoints, run_architecture

    name, path = _find_workspace(workspace, environ)
    text = typed.strip() if isinstance(typed, str) else ""
    if not text:
        raise ServerListRefused("no checkpoint path was typed")
    try:
        wanted = Path(text).resolve()
        listed = list_run_checkpoints(path)
    except (OSError, RuntimeError, ValueError) as exc:  # RuntimeError: a link loop before 3.13
        raise ServerListRefused(
            f"the typed path or the runs of the workspace {name!r} cannot be resolved "
            f"({type(exc).__name__})") from None
    for run_dir, checkpoint in listed:
        try:
            found = checkpoint.resolve()
        except (OSError, RuntimeError):
            continue
        if found == wanted and _inside(found, path):
            return run_architecture(run_dir), run_dir.name, checkpoint.name
    raise ServerListRefused(
        f"the typed path is not a checkpoint the server lists in the workspace {name!r}: a "
        "typed path only locates one of the run bar's workspace's listed checkpoints")


# ---------------------------------------------------------------------------
# Cohort versions and their reports
# ---------------------------------------------------------------------------
def _report_items(
    version: Any, report_state: Optional[ReportStateReader], *,
    store: Optional[Any] = None, operator: bool = False,
) -> Tuple[Tuple[ReportItem, ...], Optional[str]]:
    """The version's runs, newest first. Given `report_state`, each with what
    it kept, whether its report is registered, and D5's state of its key for
    those bytes (`run_contest`) from one scan of the store (`store`, a
    `record_states.read_report_store`, made here when not given)."""
    from src.api.services.record_states import read_report_store, run_contest
    from src.evaluation.cohort_registry import RegistryRefused, list_report_runs

    try:
        runs = list_report_runs(version)
    except RegistryRefused as exc:
        return (), str(exc)
    if report_state is not None and store is None:
        store = read_report_store(version)
    items = []
    for run in runs:
        if run.problem is not None:
            items.append(ReportItem(run.run_id, None, run.problem))
        elif report_state is None:
            items.append(ReportItem(run.run_id, None, (
                "kept and registered are not read here: no report-state reader was given")))
        else:
            try:
                state = dict(report_state(run.directory))
            except Exception as exc:  # noqa: BLE001 — one unreadable run must not hide the rest
                items.append(ReportItem(run.run_id, None, (
                    f"its state could not be read ({type(exc).__name__}: {exc})")))
                continue
            items.append(ReportItem(run.run_id, state, None,
                                    run_contest(store, run.run_id, state, operator=operator)))
    return tuple(items), None


def _version_item(listed: Any, report_state: Optional[ReportStateReader],
                  environ: Optional[Mapping[str, str]], *,
                  store: Optional[Any] = None,
                  operator: bool = False) -> CohortVersionItem:
    if listed.version is None:
        return CohortVersionItem(listed.label, None, None, False, None, None, None, None,
                                 None, None, listed.problem)
    manifest = listed.version.manifest
    source = _as_dict(manifest.get("source")) or {}
    imported = _as_dict(manifest.get("imported")) or {}
    role = manifest.get("role")
    reports, reports_problem = _report_items(listed.version, report_state, store=store,
                                             operator=operator)
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
    (`stored_report_state`, which the API route passes). With it, each run
    also carries D5's state of its key, without the problems' words (one scan
    of each version's store). Without it, each run is listed with no state and
    says so, and no store is scanned.

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


def _registry_root(environ: Optional[Mapping[str, str]]) -> Path:
    setting = server_settings.cohort_registry(environ)
    if not setting.available:
        raise ServerListRefused(setting.problem)
    return setting.value


def resolve_cohort_version(label: Any, *, environ: Optional[Mapping[str, str]] = None) -> Any:
    """The cohort version the registry holds as `label`, read by the registry's
    one reader (`resolve_version`: one read of its ``cohort.json``, checked,
    its directory resolved inside the registry). The label is checked by the
    registry's label rule first, so a request's text never becomes a path.

    Raises:
        ServerListRefused: no registry configured or it is refused, a label the
            rule refuses, or a version that cannot be read (an interrupted
            import, a manifest this revision does not read).
    """
    from src.evaluation.cohort_registry import (
        CohortVersionRefused,
        RegistryRefused,
        resolve_version,
        validate_label,
    )

    root = _registry_root(environ)
    try:
        return resolve_version(root, validate_label(label))
    except (RegistryRefused, CohortVersionRefused) as exc:
        raise ServerListRefused(str(exc)) from None
    except OSError as exc:
        raise ServerListRefused(
            f"the cohort version {_echo(label)} cannot be read "
            f"({type(exc).__name__}: {exc.strerror or exc})") from None


def cohort_version_item(
    label: Any, *, environ: Optional[Mapping[str, str]] = None,
    report_state: Optional[ReportStateReader] = None,
) -> CohortVersionItem:
    """One cohort version as `list_cohort_versions` lists it, with its report
    store's runs and, given `report_state`, what each kept, whether its report
    is registered and D5's state of its key: for a page that shows one
    version's reports without reading every other version's.

    Raises:
        ServerListRefused: as `resolve_cohort_version`.
    """
    return version_item(resolve_cohort_version(label, environ=environ),
                        report_state=report_state, environ=environ)


def version_item(version: Any, *, report_state: Optional[ReportStateReader] = None,
                 store: Optional[Any] = None, operator: bool = False,
                 environ: Optional[Mapping[str, str]] = None) -> CohortVersionItem:
    """A cohort version already read (`resolve_cohort_version`), listed as
    `list_cohort_versions` lists it. `store` is that version's one scan
    (`record_states.read_report_store`), for a caller that shows the scan's
    records too, so the store is scanned once; made here when not given. `operator` keeps the
    problems' words in each run's D5 state, for an operator's page (the Test
    page); the open route's listing leaves them out."""
    from src.evaluation.cohort_registry import ListedVersion

    return _version_item(ListedVersion(version.manifest.get("label"), version, None),
                         report_state, environ, store=store, operator=operator)


def resolve_report(label: Any, run_id: Any, *,
                   environ: Optional[Mapping[str, str]] = None) -> Path:
    """The ``report.json`` path of the run the version's report store lists as
    `run_id`, found among its listed run directories (`list_report_runs`: each
    resolved inside ``reports/``), never by joining a request's text onto a
    path. Reads no file: whether a report is there, and what it says, is the
    reader's to find (the registration reads it once).

    Raises:
        ServerListRefused: the version is not readable, its report store cannot
            be listed, or no run of that name is listed (or it is listed with
            a problem).
    """
    from src.evaluation.cohort_registry import REPORT_FILE, RegistryRefused, list_report_runs

    version = resolve_cohort_version(label, environ=environ)
    try:
        runs = list_report_runs(version)
    except RegistryRefused as exc:
        raise ServerListRefused(str(exc)) from None
    for run in runs:
        if run.run_id == run_id:
            if run.problem is not None or run.directory is None:
                raise ServerListRefused(run.problem or f"{run.run_id} is not a run directory")
            return run.directory / REPORT_FILE
    raise ServerListRefused(
        f"no run {_echo(run_id)} is listed in the report store of the cohort version "
        f"{version.manifest.get('label')!r}")


# ---------------------------------------------------------------------------
# Import sources: the intake directory (D7)
# ---------------------------------------------------------------------------
def _intake_directory(environ: Optional[Mapping[str, str]]) -> Path:
    setting = server_settings.intake_directory(environ)
    if not setting.available:
        raise ServerListRefused(setting.problem)
    return setting.value


def _intake_entries(directory: Path) -> List[Path]:
    """The intake directory's ``*.json`` entries, by name; hidden ones (an
    editor's or a copy tool's, named with a leading dot) are left out."""
    try:
        entries = [p for p in directory.iterdir()
                   if not p.name.startswith(".") and p.name.lower().endswith(_INTAKE_SUFFIX)]
    except OSError as exc:
        raise ServerListRefused(
            f"the intake directory {directory} cannot be listed "
            f"({type(exc).__name__}: {exc.strerror or exc})") from None
    return sorted(entries, key=lambda p: p.name)


def _intake_item(directory: Path, entry: Path) -> IntakeFileItem:
    """One entry, checked: it must resolve inside the intake directory (a link
    that leaves it is not offered) and be a regular file there. Its size and
    modification time come from that check's `stat`; nothing is read."""
    from datetime import UTC, datetime

    name = entry.name
    try:
        resolved = entry.resolve()
    except (OSError, RuntimeError) as exc:  # a link loop: RuntimeError before 3.13
        return IntakeFileItem(name, None, None,
                              f"{name} cannot be resolved ({type(exc).__name__}), so it is "
                              "not offered")
    if not _inside(resolved, directory):
        return IntakeFileItem(name, None, None,
                              f"{name} resolves to {resolved}, outside the intake directory "
                              f"{directory}, so it is not offered")
    try:
        if not resolved.is_file():
            return IntakeFileItem(name, None, None, f"{name} is not a file, so it is not offered")
        info = resolved.stat()
    except OSError as exc:
        return IntakeFileItem(name, None, None,
                              f"{name} cannot be checked ({type(exc).__name__}: "
                              f"{exc.strerror or exc}), so it is not offered")
    modified = datetime.fromtimestamp(info.st_mtime, tz=UTC)
    return IntakeFileItem(name, info.st_size, modified.strftime("%Y-%m-%dT%H:%M:%SZ"), None)


def list_intake_files(*, environ: Optional[Mapping[str, str]] = None) -> Listing:
    """The import sources the operator placed in `SHEPHERD_INTAKE_DIR`: its
    ``*.json`` files, by name, each with its size and modification time, or why
    it is not offered (`IntakeFileItem`). Reads no file's content.

    No intake directory configured, or one that cannot be listed: no item, and
    why.
    """
    try:
        directory = _intake_directory(environ)
        entries = _intake_entries(directory)
    except ServerListRefused as exc:
        return Listing((), str(exc))
    return Listing(tuple(_intake_item(directory, entry) for entry in entries))


def resolve_intake_file(name: Any, *, environ: Optional[Mapping[str, str]] = None) -> Path:
    """The path of the import source the server lists as `name`, found in the
    intake directory's listing again and checked again (inside the directory,
    a regular file); any other name, a path, or a link that leaves the
    directory is refused. Returns the listed entry (``<intake>/<name>``), so the
    version manifest records the name the operator chose. Reads nothing; the
    check is not atomic with the importer's read (one writer: the operator's
    machine).

    Raises:
        ServerListRefused: no intake directory, or `name` is not an offered
            source.
    """
    directory = _intake_directory(environ)
    for entry in _intake_entries(directory):
        if entry.name == name:
            item = _intake_item(directory, entry)
            if item.problem is not None:
                raise ServerListRefused(item.problem)
            return entry
    raise ServerListRefused(
        f"{_echo(name)} is not an import source the server lists in the intake directory "
        f"{directory}")
