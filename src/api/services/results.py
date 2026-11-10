"""
The one read-only results service: every kept and every recorded result, as rows.
=================================================================================
Research comparison §5 Stage 8 (owner's O6; plan review F2, F3, E3, E5). One
query over every result the server can read, for every client: the command
line's export (`scripts/export_results.py`) and Models › Results
(`src/webui/components/results_segment.py`, Stage 8b), which lists, compares
and exports these rows. It reads through the server's own lists and readers
and writes nothing.

**What it reads, and through what.** Each approved source context
(`SourceContext`: a name and the settings it is read with) is read by the
server's own readers, with the same checks, never by a path a request or a
report gives:

- a registry (``SHEPHERD_REGISTRY``'s rule, `server_settings.cohort_registry`):
  every version it lists (`list_versions`) and, for each, one scan of its report
  store (`record_states.read_report_store`, `scan_report_store`): every kept
  report of every kind and every mode, built into the record a ledger would
  hold (`build_record`), a kept report of a retired mode among them, each read
  by the one reader as its kind's result view (`record_views.result_view`);
- a workspaces root (``SHEPHERD_WORKSPACES_ROOT``'s rule,
  `server_settings.workspaces_root`): every entry it lists
  (`server_lists.resolve_workspaces`; an entry that is not a workspace the
  server can use, one resolving outside the root or not at all, is a row with
  its reason), every checkpoint of each workspace
  (`server_lists.list_checkpoints`, one read of each file), and every ledger of
  each workspace (`record_views.workspace_ledgers`: each run's, whether its
  weights are listed, listed with a problem or gone), every record of each read
  by the readers' one listing (`record_views.ledger_results`, `read_records`).

The server's own context is its settings (`deployment_sources`); the command
line passes explicit roots, each a context of its own (`explicit_sources`),
read with the same checks and never by rewriting the process's settings.

**One row per result, joined only on a reliable identity.** A run is its run
directory in a version's report store: a kept report and a ledger record are
one row exactly when the record names that run (``run_id``) of that version
(``cohort_digest``) and holds that report's bytes (``source_artifact_digest``);
copies of one run in several listed stores or ledgers are one row with each
location. A ledger record whose report no listed store keeps is its own row,
with the fields it has and the state "report not available"; one whose weights
no listed workspace holds readably keeps its row, its model unknown with why. A
report no listed ledger registered is its own row too, and says so only when no
listed ledger holds a record naming its run that cannot be read. A record merged before each run
had its own record (historical, ``run_id`` null) is a row of its own, never
joined to a report and never counted as a run. Two runs of one condition group,
with equal or different numbers, are two rows of one group (``condition_group``),
with the group's description in each listed store of its version
(`record_states.group_state`, the one turning of a scan into a group's state:
the fields they differ on). A
run whose report bytes differ between two listed places (a store and a ledger,
or two stores) keeps a row for each, each saying so (``conflict``); nothing is
chosen between them.

**One unreadable item never hides another.** A registry, a version, a store, a
kept report, a workspace entry or its listing, a checkpoint file, a ledger or a
ledger record (one naming no weights among them) that cannot be read is a row
of its own with its reason (built from
fixed words, field names and the item's own names; any problem text passes the
one redaction first, `path_redaction.redact_paths`), and every other row is
read as before.

**Each row lists what its records say, and nothing else** (F3): its model's
identity (the checkpoint's SHA-256, and where the server's lists find it:
source, workspace, run and file, never a path), lineage, spec summary and
structure, every value of its spec (`listed_record`), its training settings as
the checkpoint records them
(`run_plan.recorded_training_settings`: rules, learning settings and what took
effect, whether or not a freeze setting exists), its data identity and role,
its measurement settings and protocol (`record_views.recorded_settings`), its
metrics as recorded (each value as its record holds it, an integer as an
integer) and a KG edge set's run counts, its environment (the class, the
regime and the native build identity), its relation to the model's training
where it was recorded (a supplied cohort's R10 states and counts, a KG edge
set's exposure; a generated split's is its allocation's, not compared), its
kept / registered / group states and its content references
(`cohort_registry.version_references` and their states).
Anything without a source is null and named in ``unknown`` with why; nothing is
filled from this build's recipe. A report's ``checkpoint_path`` and
``data_dir`` are never opened: the record leaves them in the report.

**Measurement aggregates only** (D7). No Diagnosis reference patient, case id,
rank, ranks file or server path; R10 evidence is not read. The long form
(`long_rows`, one row per result and metric, protocol and metric name in their
own columns so no two protocols share a metric column) is the one serializer of
CSV and JSON (`to_csv`, `to_json`), its cells by the project's one set of CSV
rules (`src.utils.csv_cells`). No row is capped, ordered by a metric, or
compared by a statistic; results are newest first.

Module: src/api/services/results.py
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.utils import csv_cells

#: The word a field without a source reads as in a CSV cell; its reason is in
#: the row's ``unknown`` (JSON keeps such a field null).
UNKNOWN = "unknown"
#: The word a value recorded as null reads as in a CSV cell.
NULL_WORD = "null"
#: The word an empty recorded text reads as in a CSV cell.
EMPTY_WORD = "(empty)"

#: What an export says it is (`to_json`).
EXPORT_FORMAT = "shepherd-results-export/1"

#: A row's kind: a measurement run (its run directory in a store); a ledger
#: record merged before each run had its own record (never a run); an item that
#: could not be read.
RUN, HISTORICAL, UNREADABLE = "run", "historical_record", "unreadable"

#: A row's report state.
REPORT_KEPT = "kept"
REPORT_NOT_AVAILABLE = "report not available"
#: A row's registration state.
REGISTERED = "registered"
NOT_REGISTERED = "not registered in a listed ledger"
#: A kept report no listed ledger names as its run, whose bytes a record merged
#: before each run had its own record holds (K8-6r.2): never counted twice.
POSSIBLY_MERGED = ("not registered as its own run; a record merged before each run had its "
                   "own record holds its bytes (a possible overlap, not confirmed)")
MERGED = "registered, merged before each run had its own record"
#: A kept report no listed ledger registered readably, while a listed ledger
#: holds a record naming its run that cannot be read (a hand edit): never
#: "not registered", which would say no record exists.
RECORD_UNREADABLE = ("a listed ledger holds a record naming this run that cannot be read "
                     "(see its unreadable row)")

#: How a result's data stands to the model's training, by its kind: a KG edge
#: set's run records its exposure (its own state); a supplied cohort's records
#: R10 (``r10``: its disease and phenotype-set overlap with the training and
#: validation inputs the checkpoint records); a generated split's is its
#: allocation's, which is never compared with the model's record
#: (`caveats.GENERATED_EXPOSURE_NOTE`).
EXPOSURE_AS_R10 = "recorded as R10 (see r10)"
EXPOSURE_BY_ALLOCATION = ("not compared: a generated split's relation to a model's training is "
                          "its allocation's (see data_identity)")


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceContext:
    """One approved source: the name rows show for it (never a path), and the
    settings it is read with (`environ`: ``SHEPHERD_WORKSPACES_ROOT`` and/or
    ``SHEPHERD_REGISTRY``), each checked by the server's own rule when read."""

    name: str
    environ: Mapping[str, str]


def deployment_sources(environ: Optional[Mapping[str, str]] = None
                       ) -> Tuple[SourceContext, ...]:
    """The server's own source: its workspaces root and its registry, as its
    settings name them (`os.environ` when `environ` is None)."""
    return (SourceContext("deployment", os.environ if environ is None else environ),)


def explicit_sources(workspaces_roots: Sequence[Tuple[str, Any]] = (),
                     registries: Sequence[Tuple[str, Any]] = ()) -> Tuple[SourceContext, ...]:
    """Sources an operator names explicitly, each ``(name, directory)``: a
    workspaces root or a registry, each a context of its own read with the
    setting's rule (`server_settings.workspaces_root`, `cohort_registry`), as
    the service reads its own; the process's settings are not changed or read
    for them. Names must be distinct, so every row's location is one source.

    Raises:
        ValueError: a name is empty or given twice."""
    from src.config.server_settings import ENV_REGISTRY, ENV_WORKSPACES_ROOT

    named: List[SourceContext] = []
    for variable, given in ((ENV_WORKSPACES_ROOT, workspaces_roots), (ENV_REGISTRY, registries)):
        for name, directory in given:
            named.append(SourceContext(str(name), {variable: str(directory)}))
    seen: Dict[str, int] = {}
    for source in named:
        if not source.name.strip():
            raise ValueError("a source needs a name, which rows show in its place")
        seen[source.name] = seen.get(source.name, 0) + 1
    twice = sorted(name for name, count in seen.items() if count > 1)
    if twice:
        raise ValueError(f"source names must be distinct; given twice: {', '.join(twice)}")
    return tuple(named)


# ---------------------------------------------------------------------------
# What a query read
# ---------------------------------------------------------------------------
@dataclass
class _Kept:
    """A kept report as one store's scan built it."""

    source: str
    label: str
    run_id: str
    record: Dict[str, Any]
    view: Any


@dataclass
class _Registered:
    """A ledger record the readers' one listing read as a result."""

    location: Dict[str, Any]
    record: Dict[str, Any]
    view: Any


@dataclass
class _Store:
    """A listed version's report store, as one scan read it (`StoreRead`),
    with the source and label it was listed under and the roots its problems'
    words are redacted by."""

    source: str
    label: str
    version: Any
    read: Any
    roots: Tuple[Tuple[str, str], ...]


@dataclass
class _Run:
    """One result: a run's kept reports and ledger records with one report's
    bytes, or one historical record."""

    kind: str
    kept: List[_Kept] = field(default_factory=list)
    registered: List[_Registered] = field(default_factory=list)
    conflict: Optional[str] = None


@dataclass
class _Read:
    """Everything one query read, before rows are made."""

    runs: Dict[Tuple[str, str], Dict[str, _Run]] = field(default_factory=dict)
    historical: Dict[str, _Run] = field(default_factory=dict)
    unreadable: List[Dict[str, Any]] = field(default_factory=list)
    #: By version manifest digest: each listed store of it (`_Store`).
    stores: Dict[str, List[_Store]] = field(default_factory=dict)
    #: By checkpoint digest: every listed location of it, and its item (one
    #: listed with a problem whose digest is known among them).
    models: Dict[str, List[Tuple[str, Mapping[str, Any]]]] = field(default_factory=dict)
    #: The run of each ledger record that names one and cannot be read.
    unreadable_runs: set = field(default_factory=set)
    #: Each listed data version's content references and their states, by
    #: its manifest digest, read once per query (`_references`).
    references: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def merged_digests(self) -> frozenset:
        """Every report digest a listed historical record holds (its source
        and its merged reports): a kept report among them is a possible
        overlap, never counted as a second run (K8-6r.2)."""
        held = set()
        for run in self.historical.values():
            for registered in run.registered:
                listed = registered.record.get("corroborating_reports")
                held.update(d for d in [registered.record.get("source_artifact_digest"),
                                        *(listed if isinstance(listed, list) else [])]
                            if isinstance(d, str))
        return frozenset(held)


def _content(value: Any) -> str:
    """A record's content as one comparable text: two ledgers holding the same
    record (a copy) hold the same text."""
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _unreadable(read: _Read, source: str, what: str, reason: str, **names: Any) -> None:
    """One item that could not be read, as a row of its own."""
    read.unreadable.append({"source": source, "item": what, "reason": reason,
                            **{key: value for key, value in names.items() if value is not None}})


def _redacted(text: Any, roots: Sequence[Tuple[str, str]]) -> str:
    """A problem's words with no server path (`redact_paths`), on one line."""
    from src.utils.checkpoint_paths import problem_line
    from src.utils.path_redaction import redact_paths

    return problem_line(redact_paths(text, roots, other="a server path")) or "no reason given"


def _root_word(source: SourceContext) -> str:
    """What a problem's words show in place of a source's directory: its
    name, marked as a name (``<lab>``), so a message that already says "the
    workspaces root" or "the registry" before the directory reads once, and a
    path under it reads ``<lab>/...``."""
    return f"<{source.name}>"


def _read_registry(read: _Read, source: SourceContext) -> None:
    """Every version a source's registry lists and every kept report of each,
    each read by the one reader as its kind's result view."""
    from src.api.services.record_states import read_report_store
    from src.api.services.record_views import result_view, unreadable_reason
    from src.config import server_settings
    from src.evaluation.cohort_registry import RegistryRefused, list_versions

    setting = server_settings.cohort_registry(source.environ)
    if setting.unset:
        return
    if not setting.available:
        _unreadable(read, source.name, "registry", _redacted(setting.problem, ()))
        return
    roots = ((str(setting.value), _root_word(source)),)
    try:
        listed = list_versions(setting.value)
    except (RegistryRefused, OSError) as exc:
        _unreadable(read, source.name, "registry", _redacted(exc, roots))
        return
    for entry in listed:
        if entry.version is None:
            _unreadable(read, source.name, "data version", _redacted(entry.problem, roots),
                        label=entry.label)
            continue
        version = entry.version
        store = read_report_store(version)
        read.stores.setdefault(version.identity.sha256, []).append(
            _Store(source.name, entry.label, version, store, roots))
        if store.scan is None:
            _unreadable(read, source.name, "report store", _redacted(store.problem, roots),
                        label=entry.label)
            continue
        scan = store.scan
        for runs in scan.reports.values():
            for run_id, (_run, record) in runs.items():
                try:
                    view = result_view(record)
                except ValueError as exc:
                    _unreadable(read, source.name, "kept report",
                                "its record cannot be read as a result: "
                                + unreadable_reason(exc), label=entry.label, run_id=run_id,
                                report_sha256=scan.digests.get(run_id))
                    continue
                key = (record["cohort_digest"], run_id)
                run = read.runs.setdefault(key, {}).setdefault(
                    record["source_artifact_digest"], _Run(RUN))
                run.kept.append(_Kept(source.name, entry.label, run_id, record, view))
        for entries in scan.unbuilt.values():
            for run_id, problem in entries:
                _unreadable(read, source.name, "kept report", _redacted(problem, roots),
                            label=entry.label, run_id=run_id,
                            report_sha256=scan.digests.get(run_id))
        for run_id, problem in scan.unplaced:
            _unreadable(read, source.name, "kept report" if run_id else "report store",
                        _redacted(problem, roots), label=entry.label, run_id=run_id,
                        report_sha256=scan.digests.get(run_id) if run_id else None)


def _read_workspaces(read: _Read, source: SourceContext) -> None:
    """Every entry a source's workspaces root lists, every checkpoint of each
    workspace, and every ledger of each (`workspace_ledgers`), its records
    read by the readers' one listing (`ledger_results`)."""
    from src.api.services import server_lists
    from src.api.services.record_views import item_was_read, workspace_ledgers
    from src.config import server_settings

    setting = server_settings.workspaces_root(source.environ)
    if setting.unset:
        return
    if not setting.available:
        _unreadable(read, source.name, "workspaces root", _redacted(setting.problem, ()))
        return
    roots = ((str(setting.value), _root_word(source)),)
    refused: Dict[str, str] = {}
    try:
        workspaces = server_lists.resolve_workspaces(environ=source.environ, refused=refused)
    except server_lists.ServerListRefused as exc:
        _unreadable(read, source.name, "workspaces root", _redacted(exc, roots))
        return
    for name, problem in sorted(refused.items()):
        _unreadable(read, source.name, "workspace", _redacted(problem, roots), workspace=name)
    for name, directory in sorted(workspaces.items()):
        listing = server_lists.list_checkpoints(name, environ=source.environ)
        if not listing.available:
            _unreadable(read, source.name, "workspace", _redacted(listing.problem, roots),
                        workspace=name)
            continue
        items = [asdict(checkpoint) for checkpoint in listing.items]
        for item in items:
            if not item_was_read(item):
                _unreadable(read, source.name, "checkpoint",
                            "the checkpoint is listed with a problem: "
                            + _redacted(item["problem"], roots),
                            workspace=name, checkpoint=item["key"],
                            checkpoint_sha256=item["sha256"])
            if item["sha256"] is not None:
                read.models.setdefault(item["sha256"], []).append((source.name, item))
        for where, found in workspace_ledgers(directory, items):
            if found.problem is not None:
                _unreadable(read, source.name, "ledger", _redacted(found.problem, roots),
                            workspace=name, ledger=where)
            elif found.ledger is not None:
                _read_ledger_records(read, source.name,
                                     {"source": source.name, "workspace": name, "ledger": where},
                                     found.ledger)


def _read_ledger_records(read: _Read, source: str, where: Dict[str, Any],
                         ledger: Dict[str, Any]) -> None:
    """Every record of one ledger (`ledger_results`): each read as a result
    joins its run, or is a historical record; each other is a row with its
    reason, and its run, where it names one, is remembered."""
    from src.api.services.record_views import ledger_results

    viewed, unreadable = ledger_results(ledger)
    for _checkpoint, record, view in viewed:
        location = {**where, "sequence": view.registration.sequence,
                    "registered_at": view.registration.registered_at}
        if view.run_id is None:
            # Merged before each run had its own record: a record, never a
            # run, never joined to a report. One record held by two ledgers
            # (a copy) is one row with both locations.
            run = read.historical.setdefault(_content(record), _Run(HISTORICAL))
            run.registered.append(_Registered(location, record, view))
            continue
        key = (record["cohort_digest"], view.run_id)
        run = read.runs.setdefault(key, {}).setdefault(record["source_artifact_digest"],
                                                        _Run(RUN))
        run.registered.append(_Registered(location, record, view))
    for checkpoint, item in unreadable:
        if item.run_id is not None:
            read.unreadable_runs.add(item.run_id)
        _unreadable(read, source, "ledger record", item.reason,
                    workspace=where["workspace"], ledger=where["ledger"],
                    checkpoint_sha256=checkpoint, ledger_index=item.index,
                    sequence=item.sequence, run_id=item.run_id,
                    report_sha256=item.source_artifact_digest)


def read_all(sources: Iterable[SourceContext]) -> _Read:
    """Read every source: each registry's stores, then each workspaces root's
    checkpoints and ledgers. Runs that hold different bytes for one run are
    marked, never chosen between."""
    from src.config.server_settings import ENV_REGISTRY, ENV_WORKSPACES_ROOT

    read = _Read()
    sources = tuple(sources)
    for source in sources:
        if not any(source.environ.get(name) for name in (ENV_WORKSPACES_ROOT, ENV_REGISTRY)):
            _unreadable(read, source.name, "source",
                        f"it names no workspaces root ({ENV_WORKSPACES_ROOT}) and no registry "
                        f"({ENV_REGISTRY}), so nothing was read from it")
        _read_registry(read, source)
    for source in sources:
        _read_workspaces(read, source)
    for (_cohort, run_id), by_digest in read.runs.items():
        if len(by_digest) > 1:
            for digest, run in by_digest.items():
                others = sorted(f"{d[:12]}..." for d in by_digest if d != digest)
                run.conflict = (f"run {run_id} holds other report bytes elsewhere "
                                f"({', '.join(others)}): each is kept as found, none chosen")
    return read


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
#: A model's fields read from its listed checkpoint's record (`listed_record`).
_MODEL_FIELDS = ("learning_stage", "task", "operation", "family_id", "parent_sha256",
                 "parent_run", "encoder", "reference", "differences", "spec", "capabilities")
_SETTINGS_FIELDS = ("training_rules", "learning_settings", "effective_settings")


def _model(read: _Read, checkpoint: Optional[str], unknown: Dict[str, str]) -> Dict[str, Any]:
    """A result's model as the server's lists find its checkpoint: its
    locations (source, workspace and listing key; never a path), its lineage,
    its spec summary and structure and its recorded training settings, from
    the first location whose file was read; a checkpoint listed only with a
    problem, or not listed, has them unknown with why."""
    from src.api.services.record_views import item_was_read

    listed = read.models.get(checkpoint or "", [])
    model: Dict[str, Any] = {
        "checkpoint_sha256": checkpoint,
        "locations": [{"source": source, "workspace": item["workspace"], "checkpoint": item["key"]}
                      for source, item in listed],
    }
    item = next((item for _source, item in listed if item_was_read(item)), None)
    if item is None:
        model.update(dict.fromkeys((*_MODEL_FIELDS, *_SETTINGS_FIELDS)))
        why = ("its checkpoint is listed only with a problem, so its record is not read (see its "
               "unreadable row)" if listed else
               "its checkpoint is not in a listed workspace")
        for name in (*_MODEL_FIELDS, *_SETTINGS_FIELDS):
            unknown[f"model.{name}"] = why
        return model
    for name in _MODEL_FIELDS:
        value = item.get(name)
        model[name] = list(value) if isinstance(value, tuple) else value
    if item.get("record") != "model":
        why = ("its checkpoint's model record is not read by this build"
               if item.get("record") == "refused" else "its checkpoint has no model record read")
        for name in _MODEL_FIELDS:
            unknown[f"model.{name}"] = why
    settings = item.get("training_settings") or {}
    model["training_rules"] = settings.get("rules")
    model["learning_settings"] = settings.get("learning")
    model["effective_settings"] = settings.get("effective")
    for part, name in (("rules", "training_rules"), ("learning", "learning_settings"),
                       ("effective", "effective_settings")):
        if not settings or part in (settings.get("unrecorded") or []):
            unknown[f"model.{name}"] = (
                "a weights-only file records no training settings"
                if item.get("checkpoint_kind") == "weights" and part != "rules"
                else "its checkpoint does not record it")
    return model


def _group(read: _Read, record: Mapping[str, Any]) -> Dict[str, Any]:
    """A record's condition group (its key) and how its runs compare in each
    listed store of its version (`record_states.group_state`, the one turning
    of a scan into a group's state), which decides nothing. Listed stores that
    describe it differently leave it ``cannot_be_checked``, saying so."""
    from src.api.services.record_states import group_state, key_text
    from src.evaluation.sidecar import record_key

    key = record_key(dict(record))
    group = {"condition_group": key_text(key), "state": "cannot_be_checked", "differing": [],
             "why": None}
    stores = read.stores.get(key[2], [])
    if not stores:
        group["why"] = "its data version is not in a listed registry"
        return group
    described = [(store, group_state(store.read, key, operator=True)) for store in stores]
    found = {(state["state"], tuple(state["differing"])) for _store, state in described}
    if len(found) > 1:
        group["why"] = "the listed stores of its version describe its group differently"
        return group
    group["state"], differing = next(iter(found))
    group["differing"] = list(differing)
    if group["state"] == "cannot_be_checked":
        problems = [_redacted(problem, store.roots) for store, state in described
                    for problem in state["problems"]]
        group["why"] = ("; ".join(dict.fromkeys(problems))
                        or "the comparison of its group is incomplete")
    return group


def _version(read: _Read, digest: str) -> Optional[Any]:
    listed = read.stores.get(digest) or []
    return listed[0].version if listed else None


def _references(read: _Read, digest: str, unknown: Dict[str, str]) -> Optional[List[Any]]:
    """The data version's content references and their states
    (`check_references`, the version's own files and the registry's only), by
    name, digest and where each is kept; never a path."""
    from src.evaluation.cohort_registry import check_references

    version = _version(read, digest)
    if version is None:
        unknown["references"] = "its data version is not in a listed registry"
        return None
    if digest not in read.references:
        read.references[digest] = [
            {"name": state.reference.name, "sha256": state.reference.sha256,
             "kept": state.reference.kept, "state": state.state, "reason": state.reason}
            for state in check_references(version)]
    return read.references[digest]


def _data(view: Any, record: Mapping[str, Any]) -> Dict[str, Any]:
    """A result's data identity, as its view reads it: kind, label, role, the
    version's manifest digest and its kind's own digests and counts."""
    cohort = view.cohort.model_dump(mode="json")
    own = {name: value for name, value in cohort.items()
           if name not in ("label", "role", "kind", "version_manifest_sha256")}
    counts = view.counts.model_dump(mode="json") if hasattr(view.counts, "model_dump") \
        else dict(view.counts)
    return {"kind": view.cohort.kind, "label": view.cohort.label, "role": view.cohort.role,
            "version_sha256": view.cohort.version_manifest_sha256,
            "dataset_origin": view.dataset_origin, "identity": own, "counts": counts,
            "count_units": getattr(view, "count_units", None)}


def _metrics(view: Any, record: Mapping[str, Any], unknown: Dict[str, str]
             ) -> List[Dict[str, Any]]:
    """Each metric as recorded, with its status, unit and denominator where its
    record holds them (a KG edge set's run: `measured.metrics`); a metric that
    could not be computed is its status, never a number. Each value is the
    record's own (``record["metrics"]``, which the view validated): an integer
    stays an integer, never the view's float."""
    measured = getattr(view, "measured", None)
    statuses = measured.metrics if measured is not None else {}
    out = []
    for name in view.metrics:
        status = statuses.get(name)
        out.append({"name": name, "value": record["metrics"][name],
                    "status": status.status if status is not None else "recorded",
                    "unit": status.unit if status is not None else None,
                    "denominator": (f"{status.denominator.count} {status.denominator.unit}"
                                    if status is not None else None)})
    for name, status in statuses.items():
        if name not in view.metrics:
            out.append({"name": name, "value": status.status, "status": status.status,
                        "unit": status.unit,
                        "denominator": f"{status.denominator.count} {status.denominator.unit}"})
    if measured is None:
        unknown["metric.unit"] = ("its record holds no unit per metric (only a KG edge set's run "
                                  "records each metric's unit and denominator)")
    return out


def _run_counts(view: Any, unknown: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """A KG edge set's run's own counts, each with its unit (its pairs scored
    and the pairs whose score was not finite among them), as recorded
    (`measured`); another kind's run records none beyond ``n_ranked``."""
    measured = getattr(view, "measured", None)
    if measured is None:
        unknown["run_counts"] = ("its record holds no run counts beyond n_ranked (only a KG "
                                 "edge set's run records its own counts, each with its unit)")
        return None
    return {"counts": dict(measured.counts), "units": dict(measured.count_units)}


def _exposure(view: Any) -> Dict[str, Any]:
    """How the result's data stands to the model's training, as its record
    says it, by kind: a KG edge set's exposure (its state, reading, definition
    and counts, or why its block does not read); a supplied cohort's R10
    (``r10``: each scope's evidence state and disease and phenotype-set
    overlap, states and counts only, as the one reader's view holds them); a
    generated split's is its allocation's, not compared."""
    from src.evaluation.caveats import exposure_text

    out = {"state": None, "reading": None, "definition_version": None,
           "measured_edges": None, "in_recorded_graph_use": None, "problem": None, "r10": None}
    kind = view.cohort.kind
    if kind == "supplied":
        return {**out, "state": EXPOSURE_AS_R10, "r10": view.r10.model_dump(mode="json")}
    if kind != "kg_edges":
        return {**out, "state": EXPOSURE_BY_ALLOCATION}
    if view.exposure is None:
        return {**out, "state": "not recorded", "problem": view.exposure_problem}
    exposure = view.exposure
    return {**out, "state": exposure.state,
            "reading": exposure_text(exposure.model_dump(mode="json")),
            "definition_version": exposure.definition_version,
            "measured_edges": exposure.measured_edges.n,
            "in_recorded_graph_use": exposure.measured_edges.in_recorded_graph_use}


def _result(read: _Read, run: _Run, merged_digests: frozenset = frozenset()) -> Dict[str, Any]:
    """One result row (before the long form): every field its records hold,
    and ``unknown`` naming each without a source, with why."""
    from src.api.services.record_views import recorded_settings

    first = run.kept[0] if run.kept else run.registered[0]
    view, record = first.view, first.record
    unknown: Dict[str, str] = {}
    settings = recorded_settings(record)
    for path in settings["unread"]:
        unknown[f"measurement.{path}"] = "its record does not hold it as a value"
    registered = sorted(run.registered, key=lambda r: (r.location["source"],
                                                       r.location["workspace"],
                                                       r.location["ledger"]))
    merged = view.source_artifact_digest in merged_digests
    run_id = getattr(first, "run_id", None) or getattr(view, "run_id", None)
    result: Dict[str, Any] = {
        "result": run.kind,
        "run_id": run_id,
        "report_sha256": view.source_artifact_digest,
        "report": REPORT_KEPT if run.kept else REPORT_NOT_AVAILABLE,
        "registered": (MERGED if run.kind == HISTORICAL
                       else REGISTERED if run.registered
                       else RECORD_UNREADABLE if run_id in read.unreadable_runs
                       else POSSIBLY_MERGED if merged else NOT_REGISTERED),
        "stores": [{"source": k.source, "label": k.label} for k in run.kept],
        "ledgers": [r.location for r in registered],
        "n_corroborating": (registered[0].view.n_corroborating if registered else None),
        "historical_overlap": (sorted({s for r in registered for s in r.view.historical_overlap})
                               if registered else []),
        "conflict": run.conflict,
        "group": _group(read, record),
        "protocol": view.mode,
        "semantics": view.score_semantics,
        "measurement": {**settings["scoring"], **settings["regime"],
                        "n_ranked": view.n_ranked,
                        "scoring_semantics_digest": view.scoring_semantics_digest,
                        "measurement_semantics_digest": view.measurement_semantics_digest},
        "data": _data(view, record),
        "environment": {
            "class": view.environment_class,
            "deterministic_algorithms": view.deterministic_algorithms,
            "gpu": view.gpu.model_dump(mode="json") if view.gpu is not None else None,
            "native_identity": ({name: component.model_dump(mode="json")
                                 for name, component in view.native_identity.items()}
                                if view.native_identity is not None else None),
        },
        "exposure": _exposure(view),
        "reason": None,
    }
    if view.native_identity is None:
        unknown["environment.native_identity"] = "its record names no environment components"
    if not run.kept:
        unknown["report"] = ("no listed store keeps this run's report"
                             if run.kind == RUN else
                             "a record merged before each run had its own record names no run")
    result["model"] = _model(read, record.get("checkpoint_digest"), unknown)
    result["references"] = _references(read, record["cohort_digest"], unknown)
    result["run_counts"] = _run_counts(view, unknown)
    result["metrics"] = _metrics(view, record, unknown)
    result["unknown"] = unknown
    return result


def _unreadable_result(item: Mapping[str, Any]) -> Dict[str, Any]:
    names = {key: value for key, value in item.items() if key not in ("item", "reason")}
    return {"result": UNREADABLE, "item": item["item"], "reason": item["reason"],
            "where": names, "run_id": item.get("run_id"),
            "report_sha256": item.get("report_sha256"), "metrics": [], "unknown": {}}


def _newest_first(result: Mapping[str, Any]) -> Tuple[int, str]:
    if result["result"] == UNREADABLE:
        return (0, "")
    if result["run_id"]:
        return (2, result["run_id"])
    times = [location.get("registered_at") or "" for location in result.get("ledgers") or []]
    return (1, max(times, default=""))


def query(sources: Iterable[SourceContext]) -> List[Dict[str, Any]]:
    """Every result the sources hold, one row each (module docstring): the
    runs newest first (by their run directory's time), then the historical
    records newest first (by their registration time), then each item that
    could not be read. Never capped, never ordered by a metric."""
    read = read_all(sources)
    merged = read.merged_digests()
    results = [_result(read, run, merged) for by_digest in read.runs.values()
               for run in by_digest.values()]
    results += [_result(read, run) for run in read.historical.values()]
    results.sort(key=_newest_first, reverse=True)
    return results + [_unreadable_result(item) for item in read.unreadable]


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
#: What a query can be filtered by, each one or more paths into a result row
#: (`ANY` steps into every item of a list): a row matches a filter when a value
#: found along one of its paths is accepted; a row whose field is unknown or
#: absent (null) has no value there (`filter_values`), so it does not match a
#: filter on it. A workspace is where the row was found: a listed checkpoint of
#: its model, or a ledger holding its record (Models › Results scopes by it);
#: it is compared by the workspace's name, so over several sources one name
#: keeps that name's workspace in every source (each location names its
#: ``source``). A row found in no listed workspace has no workspace. Items
#: that could not be read are kept whatever the filters, so a filter never
#: hides one.
ANY = "*"
FILTERS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "family": (("model", "family_id"),),
    "workspace": (("model", "locations", ANY, "workspace"), ("ledgers", ANY, "workspace")),
    "protocol": (("protocol",),),
    "data_kind": (("data", "kind"),),
    "role": (("data", "role"),),
    "environment_class": (("environment", "class"),),
    "device": (("measurement", "device"),),
    "torch_version": (("measurement", "torch_version"),),
    "cuda_version": (("measurement", "cuda_version"),),
    "software_revision": (("measurement", "software_revision"),),
}


def _at(row: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = row
    for step in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(step)
    return value


def _found(value: Any, path: Sequence[str]) -> List[Any]:
    """Every value along `path` from `value`, `ANY` stepping into each item of
    a list; none where a step does not read."""
    if not path:
        return [value]
    step, rest = path[0], path[1:]
    if step == ANY:
        items = value if isinstance(value, (list, tuple)) else []
        return [found for item in items for found in _found(item, rest)]
    if not isinstance(value, Mapping):
        return []
    return _found(value.get(step), rest)


def filter_values(row: Mapping[str, Any], name: str) -> List[str]:
    """A result's values for one filter (`FILTERS`), as text, in the order
    found and each once: what a filter on `name` compares, and what a page
    offers to filter by. A field that is unknown or absent (null) gives no
    value, so no filter value matches it (never the text "None")."""
    values = [str(value) for path in FILTERS[name] for value in _found(row, path)
              if value is not None]
    return list(dict.fromkeys(values))


def filtered(results: Iterable[Dict[str, Any]],
             criteria: Mapping[str, Iterable[str]]) -> List[Dict[str, Any]]:
    """The results matching every given filter (`FILTERS`: each a set of
    accepted values, compared as text); items that could not be read are kept.

    Raises:
        ValueError: a filter this service does not know."""
    unknown = sorted(set(criteria) - set(FILTERS))
    if unknown:
        raise ValueError(f"unknown filter(s): {', '.join(unknown)}; known: "
                         f"{', '.join(FILTERS)}")
    wanted = {name: {str(v) for v in values} for name, values in criteria.items() if values}
    out = []
    for row in results:
        if row["result"] == UNREADABLE or all(
                any(value in accepted for value in filter_values(row, name))
                for name, accepted in wanted.items()):
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# The long form: one serializer for CSV and JSON
# ---------------------------------------------------------------------------
#: The long form's columns, in order, each with where its value is in a
#: result row; the metric columns come from the metric itself.
COLUMNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("result", ("result",)),
    ("protocol", ("protocol",)),
    ("metric_name", ("metric", "name")),
    ("value", ("metric", "value")),
    ("value_status", ("metric", "status")),
    ("unit", ("metric", "unit")),
    ("denominator", ("metric", "denominator")),
    ("role", ("data", "role")),
    ("semantics", ("semantics",)),
    ("data_kind", ("data", "kind")),
    ("data_label", ("data", "label")),
    ("data_version_sha256", ("data", "version_sha256")),
    ("dataset_origin", ("data", "dataset_origin")),
    ("data_identity", ("data", "identity")),
    ("data_counts", ("data", "counts")),
    ("data_count_units", ("data", "count_units")),
    ("checkpoint_sha256", ("model", "checkpoint_sha256")),
    ("model_locations", ("model", "locations")),
    ("learning_stage", ("model", "learning_stage")),
    ("task", ("model", "task")),
    ("operation", ("model", "operation")),
    ("family_id", ("model", "family_id")),
    ("parent_sha256", ("model", "parent_sha256")),
    ("parent_run", ("model", "parent_run")),
    ("encoder", ("model", "encoder")),
    ("spec_reference", ("model", "reference")),
    ("spec_differences", ("model", "differences")),
    ("spec", ("model", "spec")),
    ("capabilities", ("model", "capabilities")),
    ("training_rules", ("model", "training_rules")),
    ("learning_settings", ("model", "learning_settings")),
    ("effective_settings", ("model", "effective_settings")),
    ("run_id", ("run_id",)),
    ("report_sha256", ("report_sha256",)),
    ("report", ("report",)),
    ("registered", ("registered",)),
    ("stores", ("stores",)),
    ("ledgers", ("ledgers",)),
    ("n_corroborating", ("n_corroborating",)),
    ("historical_overlap", ("historical_overlap",)),
    ("conflict", ("conflict",)),
    ("condition_group", ("group", "condition_group")),
    ("group_state", ("group", "state")),
    ("group_differing", ("group", "differing")),
    ("group_why", ("group", "why")),
    ("candidate_construction", ("measurement", "candidate_construction")),
    ("model_construction", ("measurement", "model_construction")),
    ("tie_policy", ("measurement", "canonical_tie_policy_version")),
    ("metric_schema_version", ("measurement", "metric_schema_version")),
    ("batch_size", ("measurement", "batch_size")),
    ("n_ranked", ("measurement", "n_ranked")),
    ("run_counts", ("run_counts",)),
    ("scoring_semantics_sha256", ("measurement", "scoring_semantics_digest")),
    ("measurement_semantics_sha256", ("measurement", "measurement_semantics_digest")),
    ("environment_class", ("environment", "class")),
    ("device", ("measurement", "device")),
    ("dtype", ("measurement", "dtype")),
    ("amp_enabled", ("measurement", "amp_enabled")),
    ("amp_dtype", ("measurement", "amp_dtype")),
    ("torch_compile_wrapped", ("measurement", "torch_compile_wrapped")),
    ("deterministic_algorithms", ("measurement", "deterministic_algorithms")),
    ("cudnn_deterministic", ("measurement", "cudnn_deterministic")),
    ("cudnn_benchmark", ("measurement", "cudnn_benchmark")),
    ("torch_version", ("measurement", "torch_version")),
    ("cuda_version", ("measurement", "cuda_version")),
    ("software_revision", ("measurement", "software_revision")),
    ("gpu", ("environment", "gpu")),
    ("native_identity", ("environment", "native_identity")),
    ("exposure_state", ("exposure", "state")),
    ("exposure_reading", ("exposure", "reading")),
    ("exposure_definition", ("exposure", "definition_version")),
    ("exposure_measured_edges", ("exposure", "measured_edges")),
    ("exposure_in_recorded_graph_use", ("exposure", "in_recorded_graph_use")),
    ("exposure_problem", ("exposure", "problem")),
    ("r10", ("exposure", "r10")),
    ("references", ("references",)),
    ("unreadable_item", ("item",)),
    ("unreadable_where", ("where",)),
    ("reason", ("reason",)),
    ("unknown", ("unknown",)),
)
COLUMN_NAMES: Tuple[str, ...] = tuple(name for name, _path in COLUMNS)

#: Which ``unknown`` names make a column's empty value read as `UNKNOWN` in a
#: CSV cell (JSON keeps it null; ``unknown`` says why).
_UNKNOWN_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "unit": ("metric.unit",),
    "denominator": ("metric.unit",),
    "native_identity": ("environment.native_identity",),
    "references": ("references",),
    "run_counts": ("run_counts",),
}
for _name, _path in COLUMNS:
    if _path[0] == "model" and _path[1] not in ("checkpoint_sha256", "locations"):
        _UNKNOWN_COLUMNS[_name] = (f"model.{_path[1]}",)
    elif _path[0] == "measurement":
        _UNKNOWN_COLUMNS[_name] = (f"measurement.scoring.{_path[1]}",
                                   f"measurement.regime.{_path[1]}")


def long_rows(results: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The long form: one row per result and metric (one row for a result with
    no metric, and for an item that could not be read), every row with the
    same columns (`COLUMN_NAMES`). Values are as recorded (numbers unrounded,
    objects as objects); ``unknown`` names each field without a source."""
    rows = []
    for result in results:
        for metric in result.get("metrics") or [None]:
            scoped = {**result, "metric": metric or {}}
            row = {name: _at(scoped, path) for name, path in COLUMNS}
            rows.append(row)
    return rows


def _cell(name: str, value: Any, unknown: Mapping[str, str]) -> str:
    """One long-form value as a CSV cell, by the project's one set of rules
    (`csv_cells`): a digest ``sha256:<hex>``, a text defused, a number
    unrounded, an object or list as compact JSON text, null as `NULL_WORD`;
    a field without a source as `UNKNOWN`."""
    if value is None and any(path in unknown for path in _UNKNOWN_COLUMNS.get(name, ())):
        return UNKNOWN
    if isinstance(value, (dict, list, tuple)):
        return csv_cells.text_cell(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                              separators=(",", ":")))
    digest = csv_cells.digest_cell(value) if name.endswith("sha256") else None
    if digest is not None:
        return digest
    return csv_cells.value_cell(value, null_word=NULL_WORD, empty_word=EMPTY_WORD)


def to_csv(results: Iterable[Mapping[str, Any]]) -> str:
    """The long form as CSV text: one header row, then one row per result and
    metric, each cell by `_cell`."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMN_NAMES)
    for row in long_rows(results):
        unknown = row.get("unknown") or {}
        writer.writerow([_cell(name, row[name], unknown) for name in COLUMN_NAMES])
    return buffer.getvalue()


def _json_value(value: Any) -> Any:
    """A value as JSON holds it: a number that is not finite (a training
    setting may record one) as its text, which JSON has no number for."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def to_json(results: Iterable[Mapping[str, Any]]) -> str:
    """The long form as JSON text: the format's name, the columns and the
    rows, each with every column (null where a field holds nothing; ``unknown``
    says why for a field without a source)."""
    return json.dumps({"format": EXPORT_FORMAT, "columns": list(COLUMN_NAMES),
                       "rows": _json_value(long_rows(results))},
                      indent=1, sort_keys=False, ensure_ascii=False, allow_nan=False) + "\n"


__all__ = [
    "COLUMN_NAMES",
    "EXPORT_FORMAT",
    "FILTERS",
    "HISTORICAL",
    "RUN",
    "ANY",
    "SourceContext",
    "UNREADABLE",
    "deployment_sources",
    "explicit_sources",
    "filter_values",
    "filtered",
    "long_rows",
    "query",
    "read_all",
    "to_csv",
    "to_json",
]
