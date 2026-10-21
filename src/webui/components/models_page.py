"""
Research › Evaluate › Models: the research preset of the shared model list (M5-2, UI-D).
========================================================================================
`docs/working/PLAN_TEST_RESULTS.md` §6, "M5's boundaries" (Model Management on
the shared model list, Trainer validation and each cohort's Mode C results kept
apart), D4 (ordering by test score), D5 (contested records), D8 (the list is an
operator control) and §4.9 question 1 (ordering per role, answered 2026-10-19).

**One list, one Load.** The page places the research preset of the shared list
(`ctx.create_model_list("research")`, `src/webui/components/model_list.py`): the
workspace select, the table ("Fingerprints: model · dataset trained on", the
Trainer's validation figure, the recorded tests' state), "In use (Diagnosis)" on
the served row, and the one gated Load with its confirmation. This module never
lists a checkpoint, never reloads and never writes a usage history entry itself:
it shapes the list through the handle's ``view`` (an architecture, its groups,
one extra column, the note beside Load) and ``ordering`` (so that a Load from a
list ordered by a cohort version's score is written to that version's usage
history by the shared Load, `model_list._record_usage`).

**Ordering by test score (D4).** Offered only for a cohort version whose role is
in ``SHEPHERD_ORDERING_ROLES`` (WP1's version list says so per version,
`CohortVersionItem.offers_ordering`); ``institutional_acceptance`` is listed but
off, with its reason. The order itself is WP4's one pure function,
`src.evaluation.ordering.order_by_test_score`, over the records each listed
checkpoint's ledger holds (read through the one ledger reader, `read_ledger`),
one metric, one cohort version (its manifest digest) and one scoring semantics:
one record per checkpoint by D4's fixed rule, a contested record (its key's state
in the version's report store, `scan_report_store`) left out and never replaced
by the next one. The page shows:

- the independence note, wherever ordering is offered: a cohort version whose
  scores were used to choose a model is no longer an independent final test
  for that model;
- the groups "Ordered by … · highest first", "Contested on this version · not
  ordered", "Not compared on this version · not ordered" (records that cannot be
  compared: another scoring semantics, two ledgers, a record or ledger that
  cannot be read, a checkpoint that cannot be read) and "Not tested on this
  cohort version · newest first";
- on each ordered row its value, the record's registration number, its regime
  (device · deterministic or not) and "regime differs" where it is not the one
  most ordered rows share.

**The detail** of the selected checkpoint: its full model fingerprint (Copy), what
it was trained on, the Trainer's validation figure in its own card (a selection
metric on val, never beside a Mode C number), Model Facts' warnings, and two
in-page segments: "Recorded test results (Mode C)" (every record its ledger
holds, as the API's record view shows a record: D3's label, coverage with every
exclusion, relations, R10 as states with counts only where the state has one,
so *unverifiable* is a word and never 0, environment, registration; contested or
"contested status cannot be checked" per record, never uncontested by default;
when ordered, which record D4's rule picks and how the others differ) and
"Usage history" (the entries of every listed cohort version's usage history
that name this checkpoint, newest first).

**Read-only (D8).** Listing opens checkpoint files (SPEC_4 C3), so a request D8
refuses gets no list: the page says why (the three conditions of an operator
request) and shows "Recorded test results of the model in use", read by the
API's own read-only route function (`get_loaded_model_evaluations`): the ledger
beside the served checkpoint, filtered by the served digest, **opening no
checkpoint**, with each record's contested state from the report store. Its
reasons name no server path.

**Per browser session:** the page's data, its controls and the selection are
`gr.State` or component values; nothing here is module state. Nothing ticks: the
page is read when it is opened (`ctx.on_page_open`), when the workspace changes
and after a Load (the shared list's own refresh).

Module: src/webui/components/models_page.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gradio as gr

# D3's label, the pre-B-1 note, R10's and the relations' words: rendered, never reworded.
from src.evaluation.caveats import (
    MODE_C_RESULT_LABEL,
    PRE_B1_NOTE,
    R10_LABEL_TEXT,
    R10_STATE_TEXT,
    R10_UNVERIFIABLE_REASON,
    RECORD_ABSENCE,
    RELATION_DISPLAY_NAMES,
)
from src.webui.components import model_list as ml
from src.webui.components import renderers as r
from src.webui.components.chips import D8_SHORT
from src.webui.components.help import copy_button, help_button
from src.webui.components.served_status import read_status, served_checkpoint_sha
from src.webui.operator_access import operator_access
from src.webui.shell import page_elem_id, plain_html

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Words
# ---------------------------------------------------------------------------
#: Each metric a list can be ordered by (`ordering.ORDERABLE_METRICS`), as a
#: recorded result names it: always "Mode C", never the Trainer's "Val MRR".
METRIC_LABELS: Dict[str, str] = {
    "untruncated_mrr": "Mode C MRR (untruncated)",
    "hits@1": "Mode C Hits@1",
    "hits@5": "Mode C Hits@5",
    "hits@10": "Mode C Hits@10",
    "hits@20": "Mode C Hits@20",
    "hits@50": "Mode C Hits@50",
    "hits@100": "Mode C Hits@100",
    "mean_rank": "Mode C mean rank",
}
DEFAULT_METRIC = "untruncated_mrr"

NEWEST, BY_SCORE = "newest", "score"
ORDER_CHOICES = [("Newest first", NEWEST), ("By test score", BY_SCORE)]
ALL_ARCHITECTURES = "all"
RECORDS, USAGE = "records", "usage"

#: §4.9 question 1: said wherever ordering is offered.
INDEPENDENCE_NOTE = ("A cohort version whose scores were used to choose a model is no longer an "
                     "independent final test for that model.")
INDEPENDENCE_DETAIL = ("Only records of one cohort version and one scoring semantics are ordered "
                       "together; contested records are not.")

ROLE_ON = "ordering on: the role is in the server's model-selection roles"
ROLE_NEVER = "off: awaits the institution's formal evaluation rules"
#: Added when the deployment lists the role anyway (the setting's own note says it is ignored).
ROLE_NEVER_LISTED = " (listed in SHEPHERD_ORDERING_ROLES, and ignored)"
ROLE_NOT_LISTED = ("off: the role is not in the server's model-selection roles "
                   "(SHEPHERD_ORDERING_ROLES)")

GROUP_ORDERED = "Ordered by {metric} on {version} · {direction} first"
GROUP_CONTESTED = "Contested on this version · not ordered"
GROUP_NOT_COMPARED = "Not compared on this version · not ordered"
GROUP_NOT_TESTED = "Not tested on this cohort version · newest first"
#: Every shown checkpoint is placed in a group; one that is not was listed
#: after the page last read its records (another workspace just chosen).
GROUP_UNREAD = "Records not read yet · newest first"

LOAD_NOTE = ("Reloads the pipeline and asks once, from → to, as Change model does. Results "
             "already produced keep their own identity.")
LOAD_NOTE_ORDERED = ("Reloads the pipeline and asks once, from → to, as Change model does. "
                     "Loaded from this ordered list, so the cohort's usage history records "
                     "“ordered by {metric} on {version}”. Results already produced keep their "
                     "own identity.")

READONLY_TITLE = "Recorded test results of the model in use"
READONLY_WHERE = "from the ledger beside the loaded checkpoint · opens no checkpoint"
#: D8's three conditions, named together for a read-only request.
D8_CONDITIONS = ("Operator controls need this deployment set to M1, a request on this "
                 "machine's loopback (here or through SSH forwarding), and no forwarding "
                 "header.")
LIST_IS_C3 = ("Listing opens checkpoint files, which SPEC_4 classes as checkpoint inspection "
              "(C3), so the list is served only where the operator controls are (D8).")
LIST_C3_LINE = "🔒 Checkpoint list: operator controls only (D8 · SPEC_4 C3)"
NO_LOAD_HERE = ("No validation (Trainer) figures and no load here: both need the checkpoint "
                "files.")
RECORDS_MODEL_CHANGED = ("The model in use changed while its recorded results were read; open "
                         "the page again.")

#: What a dataset record's origin means (the Diagnosis panel's words).
ORIGIN_MEANING = {
    "synthetic": "a generated stand-in, not a clinical knowledge graph",
    "files": "built from identified source files",
}
_R10_EVIDENCE = {"found": "found", "not_found": "not found", "unreadable": "found, not readable",
                 "not_recorded": "not recorded by the checkpoint"}
_RELATION_SHORT = {"cohort_graph": "cohort↔graph"}

#: The help texts this page's ⓘ name (registered through `ctx.help`).
PAGE_HELP = {
    "models-ordering": (
        "Ordering by test score",
        "Newest first lists every checkpoint of the workspace, latest run first. By test score "
        "orders them by one Mode C metric of their recorded results on one cohort version under "
        "one scoring semantics (D4). It is offered only for a cohort version whose role is in "
        "the server's model-selection roles (SHEPHERD_ORDERING_ROLES), never for "
        "institutional_acceptance. One record per checkpoint, by a fixed rule: its first "
        "registered record from a deterministic regime, else its first registered record. A "
        "contested record is not ordered, and the next record is never put in its place. No "
        "model is ever chosen automatically by a test score.",
    ),
    "models-regime": (
        "Regime",
        "The numerical regime of the record the fixed rule picked: the device, and whether "
        "deterministic algorithms were on (batch size, versions and the code revision are "
        "compared too). A row whose regime is not the one most ordered rows share is marked "
        "“regime differs”: its number is in the same scoring group, measured another way.",
    ),
    "models-contested": (
        "Contested",
        "Two or more reports kept under this record's key in the cohort version's report store "
        "disagree (D5). No report under the key can be registered while they do, and a contested "
        "record is not ordered. Investigation notes are kept with the reports (Test page); they "
        "record what was found and never lift the block.",
    ),
    "models-usage": (
        "Usage history",
        "Each cohort version keeps an append-only usage history (reports/usage.jsonl in its "
        "report store): every test run, every registration and every model load from a list "
        "ordered by its score, whether started here or from the command line. It makes the use "
        "of a cohort for model selection visible; it does not prevent it.",
    ),
    "models-r10": (
        "R10 labels",
        " · ".join(f"{text['name']}: {text['definition']}"
                   for labels in R10_LABEL_TEXT.values() for text in labels.values()) + ".",
    ),
}

#: How many usage history entries of one checkpoint the detail shows.
USAGE_SHOWN = 200

_RUN_ID_TIME = re.compile(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z-")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _get(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def fmt_metric(name: str, value: Any) -> str:
    """A Mode C value as shown: four decimals, two for the mean rank."""
    if not _number(value):
        return "not reported"
    return f"{value:.2f}" if name == "mean_rank" else f"{value:.4f}"


def metric_label(name: Any) -> str:
    text = str(name)
    if text in METRIC_LABELS:
        return METRIC_LABELS[text]
    return f"Mode C {text}"


def _utc_text(raw: Any) -> str:
    """"2026-10-17 15:31 UTC" from a recorded ISO time, or the text as given."""
    if not isinstance(raw, str):
        return "not reported"
    try:
        moment = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw
    if moment.tzinfo is None:
        return raw
    return moment.strftime("%Y-%m-%d %H:%M UTC")


def run_time(run_id: Any) -> Optional[str]:
    """"2026-10-17 13:40 UTC" from a report store run id, else None."""
    match = _RUN_ID_TIME.match(str(run_id or ""))
    if not match:
        return None
    y, mo, d, h, mi, _s = match.groups()
    return f"{y}-{mo}-{d} {h}:{mi} UTC"


def _key_text(key: Sequence[Any]) -> str:
    return json.dumps(list(key))


def listing_key(listing: Mapping[str, Any]) -> str:
    """What one listing is, as one comparable value (the page re-reads its own
    data only for a listing it has not read yet): the listing itself, compared
    whole. It already carries each checkpoint's digest from the one listing
    (WP1's), so nothing here hashes again (`tests/unit/test_provenance_pins.py`)."""
    return json.dumps(listing, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Reading (server side): versions, ledgers, the report store, usage histories
# ---------------------------------------------------------------------------
def read_versions(environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """WP1's cohort version list as the page needs it: each version's label,
    manifest digest, role and whether that role offers ordering (the server's
    one predicate, `role_offers_ordering`), or why the registry offers none."""
    from src.api.services.server_lists import list_cohort_versions
    from src.config import server_settings

    listing = list_cohort_versions(environ=environ)
    roles = server_settings.ordering_roles(environ)
    versions = []
    for item in listing.items:
        versions.append({
            "label": item.label, "sha256": item.manifest_sha256, "role": item.role,
            "offers": bool(item.offers_ordering and item.problem is None),
            "problem": item.problem,
        })
    return {"versions": versions, "problem": listing.problem,
            "roles": list(roles.value or ()), "roles_note": roles.note}


class _Versions:
    """The registry's versions resolved once per read, by label, each checked
    against the digest the server's list gave it."""

    def __init__(self, versions: Sequence[Mapping[str, Any]],
                 environ: Optional[Mapping[str, str]]) -> None:
        from src.config import server_settings

        self.listed = {v["sha256"]: v for v in versions if r.is_digest(v.get("sha256"))}
        setting = server_settings.cohort_registry(environ)
        self.registry = setting.value if setting.available else None
        self.problem = None if setting.available else (setting.problem or "no cohort registry")
        self._resolved: Dict[str, Any] = {}

    def by_digest(self, digest: Any) -> Tuple[Any, Optional[str]]:
        """``(version, None)`` or ``(None, why)`` for a cohort version digest."""
        from src.evaluation.cohort_registry import resolve_version

        if self.registry is None:
            return None, f"no cohort registry: {self.problem}"
        listed = self.listed.get(digest)
        if listed is None:
            return None, (f"its cohort version ({r.short_hex(digest)}) is not one the server's "
                          "registry lists")
        label = listed["label"]
        if label not in self._resolved:
            try:
                version = resolve_version(self.registry, label)
            except Exception as exc:  # noqa: BLE001 - a state, never a crash of the page
                self._resolved[label] = (None, f"{label} cannot be read ({type(exc).__name__})")
            else:
                if version.identity.sha256 != digest:
                    self._resolved[label] = (None, f"{label} changed since it was listed")
                else:
                    self._resolved[label] = (version, None)
        return self._resolved[label]


def ledger_records(workspace_dir: Path, items: Sequence[Mapping[str, Any]]
                   ) -> Dict[str, Dict[str, Any]]:
    """For each listed checkpoint with a digest, what the ledger beside it holds
    for its weights, read through the one ledger reader (`read_ledger`, once per
    directory) and `records_for`: ``state`` (``records``, ``none``,
    ``no_ledger`` or ``not_read``), the records as the ledger holds them (kept
    on the server, in the session's state; never sent to the browser as they
    are), ``problem`` (naming the file, never its directory) and ``ledger``
    (the ledger's resolved path, which only tells two ledgers apart).

    The checkpoint file is never opened: its path is the run layout's, resolved
    to find the directory its ledger is in, as registration writes it."""
    from src.api.routes.evaluations import _ledger_problem
    from src.evaluation.sidecar import LEDGER_FILENAME, read_ledger, records_for
    from src.utils.checkpoint_paths import RUN_CHECKPOINTS_DIRNAME, RUNS_DIRNAME

    workspace_dir = Path(workspace_dir).resolve()
    read: Dict[Path, Tuple[Optional[dict], Optional[str], bool]] = {}
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not ml.loadable(item):
            continue
        key = ml.item_key(item)
        file = (workspace_dir / RUNS_DIRNAME / str(item.get("architecture")) / str(item.get("run"))
                / RUN_CHECKPOINTS_DIRNAME / str(item.get("filename")))
        try:
            directory = file.resolve().parent
        except (OSError, RuntimeError):
            out[key] = {"state": "not_read", "records": [], "ledger": None,
                        "problem": "the checkpoint's directory cannot be resolved"}
            continue
        if directory != workspace_dir and workspace_dir not in directory.parents:
            out[key] = {"state": "not_read", "records": [], "ledger": None,
                        "problem": "the checkpoint resolves outside the workspace"}
            continue
        ledger_path = directory / LEDGER_FILENAME
        if directory not in read:
            try:
                ledger, digest = read_ledger(ledger_path)
                read[directory] = (ledger, None, digest is not None)
            except (OSError, ValueError) as exc:
                read[directory] = (None, _ledger_problem(exc, ledger_path), True)
        ledger, problem, present = read[directory]
        if problem is not None:
            out[key] = {"state": "not_read", "records": [], "ledger": str(ledger_path),
                        "problem": problem}
        elif not present:
            out[key] = {"state": "no_ledger", "records": [], "ledger": str(ledger_path),
                        "problem": None}
        else:
            records = records_for(ledger, item["sha256"])
            out[key] = {"state": "records" if records else "none", "records": records,
                        "ledger": str(ledger_path), "problem": None}
    return out


def contest_states(records: Iterable[Mapping[str, Any]], versions: _Versions, *,
                   operator: bool = True) -> Dict[str, Dict[str, Any]]:
    """Each record key's state in its cohort version's report store (D5), by
    the key's text: one scan of each store (`scan_report_store`), then
    `StoreScan.status`. A key whose state cannot be read is ``cannot_be_checked``,
    never uncontested. For a contested key, the disagreeing runs' values of the
    metrics that differ. `operator` False leaves out the problems' own words,
    which may name a server path."""
    from src.evaluation.report_store import scan_report_store
    from src.evaluation.sidecar import record_key

    keys: Dict[str, Tuple[Any, ...]] = {}
    for record in records:
        try:
            key = record_key(dict(record))
        except ValueError:
            continue
        keys[_key_text(key)] = key
    scans: Dict[str, Tuple[Any, Optional[str]]] = {}
    out: Dict[str, Dict[str, Any]] = {}
    for text, key in keys.items():
        digest = key[2]
        if digest not in scans:
            version, why = versions.by_digest(digest)
            if version is None:
                scans[digest] = (None, why)
            else:
                try:
                    scans[digest] = (scan_report_store(version), None)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("A report store could not be scanned")
                    scans[digest] = (None, f"its report store could not be read "
                                           f"({type(exc).__name__})")
        scan, why = scans[digest]
        if scan is None:
            out[text] = {"state": "cannot_be_checked", "differing": [], "runs": [],
                         "problems": [why] if operator else []}
            continue
        try:
            status = scan.status(key)
        except Exception as exc:  # noqa: BLE001
            out[text] = {"state": "cannot_be_checked", "differing": [], "runs": [],
                         "problems": [f"({type(exc).__name__})"] if operator else []}
            continue
        runs = []
        if status.state == "contested":
            for run_id, record in (scan.reports.get(tuple(key)) or {}).values():
                if run_id in status.disagreeing_runs:
                    metrics = record.get("metrics") if isinstance(record, Mapping) else None
                    runs.append({"run": run_id, "values": {
                        name: metrics.get(name) for name in status.differing
                        if isinstance(metrics, Mapping) and name in metrics}})
        out[text] = {"state": status.state, "differing": list(status.differing),
                     "runs": sorted(runs, key=lambda run: run["run"]),
                     "problems": list(status.problems) if operator else []}
    return out


def usage_entries(versions: Sequence[Mapping[str, Any]], resolver: _Versions,
                  shas: Iterable[str]) -> Dict[str, Any]:
    """The entries of every listed cohort version's usage history that name one
    of these checkpoints, by checkpoint digest, newest first, each with its
    version's label; and what could not be read (a line that is not an entry is
    named, never skipped silently)."""
    from src.evaluation.report_store import read_usage_history

    wanted = {s for s in shas if r.is_digest(s)}
    by_sha: Dict[str, List[Dict[str, Any]]] = {}
    problems: List[str] = []
    read_from: List[str] = []
    if not wanted:
        return {"entries": {}, "problems": [], "read_from": []}
    for listed in versions:
        if listed.get("problem") or not r.is_digest(listed.get("sha256")):
            continue
        version, why = resolver.by_digest(listed["sha256"])
        if version is None:
            problems.append(f"{listed['label']}: {why}")
            continue
        try:
            entries, found = read_usage_history(version)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"usage.jsonl of {listed['label']} could not be read "
                            f"({type(exc).__name__})")
            continue
        read_from.append(listed["label"])
        for problem in found:
            problems.append(re.sub(r"^\S*usage\.jsonl", f"usage.jsonl of {listed['label']}",
                                   problem))
        for entry in entries:
            sha = entry.get("checkpoint_sha256")
            if sha in wanted:
                by_sha.setdefault(sha, []).append({**entry, "version": listed["label"]})
    for entries in by_sha.values():
        entries.sort(key=lambda e: str(e.get("at") or ""), reverse=True)
    return {"entries": by_sha, "problems": problems, "read_from": read_from}


def read_page(listing: Mapping[str, Any], *, environ: Optional[Mapping[str, str]] = None
              ) -> Dict[str, Any]:
    """The page's data for one operator listing, read once per listing: the
    cohort versions (WP1's list), each listed checkpoint's ledger records, every
    record key's contested state, and the usage history entries that name a
    listed checkpoint. Plain data for a `gr.State`."""
    from src.api.services.server_lists import ServerListRefused, resolve_workspace

    data: Dict[str, Any] = {"listing_key": listing_key(listing), "readonly": False}
    try:
        data.update(read_versions(environ))
    except Exception:  # noqa: BLE001 - the list still shows; ordering says why it is off
        logger.exception("The cohort versions could not be listed")
        data.update({"versions": [], "problem": "the cohort versions could not be listed",
                     "roles": [], "roles_note": None})
    items = list(listing.get("items") or [])
    ledgers: Dict[str, Dict[str, Any]] = {}
    workspace = listing.get("workspace")
    if workspace and items:
        try:
            # The listed name found again by one listing of the root (never a
            # path from the listing): the ledgers are beside its checkpoints.
            ledgers = ledger_records(resolve_workspace(workspace, environ=environ), items)
        except ServerListRefused:
            ledgers = {}
        except Exception:  # noqa: BLE001 - the list still shows; each row says "not read"
            logger.exception("The ledgers could not be read")
    data["ledgers"] = ledgers
    resolver = _Versions(data["versions"], environ)
    records = [rec for entry in ledgers.values() for rec in entry["records"]]
    data["contest"] = contest_states(records, resolver, operator=True)
    data["usage"] = usage_entries(data["versions"], resolver,
                                  (i.get("sha256") for i in items if ml.loadable(i)))
    return data


def _route_records() -> Dict[str, Any]:
    """The API's read-only route for the served model's recorded results
    (`get_loaded_model_evaluations`), called in this process: the ledger beside
    the served checkpoint, filtered by the served digest, opening no checkpoint.
    Its handler has no ``await``, so it runs to completion here."""
    from fastapi.encoders import jsonable_encoder

    from src.api.routes import evaluations

    return jsonable_encoder(asyncio.run(evaluations.get_loaded_model_evaluations()))


def read_readonly(*, environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The read-only view's data: the status (for the served digest and the
    time it was read), the route's answer for the served model, and each of its
    records' contested states (with no problem text: no server path reaches a
    read-only request)."""
    from src.evaluation.sidecar import KEY_FIELDS

    data: Dict[str, Any] = {"readonly": True, "listing_key": None}
    status, problem = read_status()
    data["status_at"] = datetime.now().strftime("%H:%M:%S")
    data["status_problem"] = problem
    data["served_sha"] = served_checkpoint_sha(status) if status is not None else None
    try:
        answer = _route_records()
    except Exception:  # noqa: BLE001 - a state, never a traceback
        logger.exception("The recorded results of the model in use could not be read")
        answer = {"state": "unavailable"}
    data["answer"] = answer
    records = answer.get("records") if isinstance(answer.get("records"), list) else []
    keyed = []
    for record in records:
        key = route_record_key(record, answer.get("checkpoint_sha256"))
        if key is not None:
            keyed.append(dict(zip(KEY_FIELDS, key, strict=True)))
    contest: Dict[str, Dict[str, Any]] = {}
    if keyed:
        try:
            versions = read_versions(environ)["versions"]
        except Exception:  # noqa: BLE001
            versions = []
        contest = contest_states(keyed, _Versions(versions, environ), operator=False)
    data["contest"] = contest
    return data


def route_record_key(record: Mapping[str, Any], checkpoint_sha: Any
                     ) -> Optional[Tuple[str, ...]]:
    """The ledger key (`KEY_FIELDS`' order) of a record as the API's record
    view shows it: the digest the route filtered by, the cohort's role and
    version digest, the measurement semantics digest."""
    key = (checkpoint_sha, _get(record, "cohort", "role"),
           _get(record, "cohort", "version_manifest_sha256"),
           record.get("measurement_semantics_digest"))
    return key if all(isinstance(part, str) and part for part in key) else None


# ---------------------------------------------------------------------------
# One record, as a person reads it
# ---------------------------------------------------------------------------
def record_view(record: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """A ledger record as the API's record view (`RecordView`: aggregates and
    digests only, every field checked), plus what D4 and D5 need beside it: the
    key, the device and whether deterministic algorithms were on. None when the
    record does not have the shape registration writes (it is counted, never
    shown half-read)."""
    from src.api.routes.evaluations import _record_view
    from src.evaluation.sidecar import record_key

    try:
        view = _record_view(dict(record)).model_dump(mode="json")
        key = record_key(dict(record))
    except ValueError:
        return None
    regime = record.get("regime") if isinstance(record.get("regime"), Mapping) else {}
    flag = regime.get("deterministic_algorithms")
    view["key"] = list(key)
    view["deterministic"] = flag if type(flag) is bool else None
    view["device"] = regime.get("device") if isinstance(regime.get("device"), str) else None
    view["source"] = record.get("source_artifact_digest")
    return view


def view_from_route(record: Mapping[str, Any], checkpoint_sha: Any) -> Dict[str, Any]:
    """A record view the read-only route answered, with the same extras."""
    from src.evaluation.sidecar import ENVIRONMENT_CLASSES

    view = dict(record)
    key = route_record_key(record, checkpoint_sha)
    view["key"] = list(key) if key else None
    klass = record.get("environment_class")
    view["deterministic"] = (True if klass == ENVIRONMENT_CLASSES[0] else
                             False if klass == ENVIRONMENT_CLASSES[1] else None)
    view["device"] = ("cuda" if klass in ENVIRONMENT_CLASSES[:2] else
                      "cpu" if klass == ENVIRONMENT_CLASSES[2] else None)
    view["source"] = record.get("source_artifact_digest")
    return view


def regime_text(device: Any, deterministic: Any) -> str:
    """"CUDA · deterministic" from a record's regime."""
    kind = (device.split(":", 1)[0].upper() if isinstance(device, str) and device
            else "device not read")
    determinism = ("deterministic" if deterministic is True else "non-deterministic"
                   if deterministic is False else "determinism not read")
    return f"{kind} · {determinism}"


def coverage_html(view: Mapping[str, Any]) -> str:
    counts = view.get("counts") if isinstance(view.get("counts"), Mapping) else {}
    excluded = counts.get("excluded") if isinstance(counts.get("excluded"), Mapping) else None
    if excluded is None:
        reasons = "not reported"
    else:
        reasons = " · ".join(f"{str(name).replace('_', ' ')} {count}"
                             for name, count in excluded.items()) or "none"
    return (f'<div>{r.e(view.get("n_ranked"))} of {r.e(counts.get("source_cases"))} source cases '
            f'ranked ({r.e(counts.get("ranked_eligible"))} eligible)</div>'
            f'<div class="shep-sub">{r.e(counts.get("n_excluded"))} excluded: {r.e(reasons)}</div>')


def relations_html(relations: Any) -> str:
    """Each relation's state: ✓ verified, "not applicable"; the full names in
    the tooltip."""
    if not isinstance(relations, Mapping):
        return "Relations: not reported"
    names = [*(n for n in RELATION_DISPLAY_NAMES if n in relations),
             *(n for n in relations if n not in RELATION_DISPLAY_NAMES)]
    short, full = [], []
    for name in names:
        label = _RELATION_SHORT.get(name, str(name))
        state = relations[name]
        mark = "✓" if state == "verified" else ("not applicable" if state == "not_applicable"
                                                else str(state))
        short.append(f"{label} {mark}")
        full.append(f"{label} {RELATION_DISPLAY_NAMES.get(name, name)}: {mark}")
    return (f'<span title="{r.e(" · ".join(full))}">Relations: '
            f'{r.e(" · ".join(short))}</span>')


def r10_label_text(label: str, name: str, part: Any, evidence: Any) -> str:
    """One R10 label in the caveats' words: a count with its denominator only
    where the state has one; *unverifiable* is a reason, never a number."""
    state = _get(part, "state")
    words = R10_STATE_TEXT.get(state) if isinstance(state, str) else None
    if state == "unverifiable" and isinstance(evidence, str):
        words = R10_UNVERIFIABLE_REASON.get(evidence, words)
    if words is None:
        return f"{name}: not reported"
    if state not in ("overlap", "none"):
        return f"{name}: {words}"
    unit = "distinct test diseases" if label == "disease_overlap" else "test cases"
    text = f"{name}: {words} — {_get(part, 'k')} of {_get(part, 'n')} {unit}"
    if label == "disease_overlap":
        text += f"; {_get(part, 'cases')} test cases with one of them"
    return text


def r10_html(r10: Any) -> str:
    lines = []
    for scope, labels in R10_LABEL_TEXT.items():
        part = _get(r10, scope)
        evidence = _get(part, "evidence", "state")
        found = _R10_EVIDENCE.get(evidence) if isinstance(evidence, str) else None
        text = " · ".join(r10_label_text(label, words["name"], _get(part, label), evidence)
                          for label, words in labels.items())
        lines.append(f'<span class="shep-mp-ln"><b>R10, {r.e(scope)} inputs (evidence '
                     f'{r.e(found or "not reported")}):</b> {r.e(text)}</span>')
    return "".join(lines)


def contest_html(state: Optional[Mapping[str, Any]], *, ordered_out: bool = False) -> str:
    """A record's D5 state beside it: contested (with what the runs disagree on)
    or "cannot be checked"; nothing for an uncontested key."""
    if state is None:
        return ('<div class="shep-mp-delta shep-muted">contested status cannot be checked: it was '
                'not read</div>')
    kind = state.get("state")
    if kind == "uncontested":
        return ""
    if kind == "contested":
        values = []
        names = state.get("differing") or []
        for name in names:
            per_run = [f"{fmt_metric(name, run['values'][name])} (run "
                       f"{run_time(run['run']) or run['run']})"
                       for run in state.get("runs") or [] if name in (run.get("values") or {})]
            if per_run:
                values.append(f"{metric_label(name)} " + " vs ".join(per_run))
        what = "; ".join(values) or (" · ".join(str(n) for n in names) + " differ")
        tail = " · contested — not ordered" if ordered_out else ""
        return (f'<div class="shep-mp-delta shep-amber-t">runs disagree — investigate: '
                f'{r.e(what)}{r.e(tail)}</div>')
    problems = "; ".join(str(p) for p in state.get("problems") or [])
    return ('<div class="shep-mp-delta shep-muted">contested status cannot be checked'
            + (f": {r.e(r.short_reason(problems, 160))}" if problems else "") + "</div>")


def _record_cell(view: Mapping[str, Any], state: Optional[Mapping[str, Any]],
                 pick: Optional[Mapping[str, Any]]) -> str:
    sequence = _get(view, "registration", "sequence")
    kind = (state or {}).get("state")
    if kind == "contested":
        chip = (r.chip(f"contested #{sequence}", "amber")
                + help_button("models-contested", "Contested"))
    elif kind == "uncontested":
        chip = r.chip(f"registered #{sequence}", "ok")
    else:
        chip = r.chip(f"registered #{sequence}", "grey")
    determinism = ("deterministic on" if view.get("deterministic") is True else
                   "deterministic off" if view.get("deterministic") is False else
                   "determinism not read")
    n = view.get("n_corroborating")
    corroborating = (f'<span class="shep-sub">{r.e(n)} corroborating report'
                     f'{"s" if n != 1 else ""}</span>' if _number(n) and n else "")
    parts = [f'{chip} <span class="shep-sub">{r.e(determinism)}</span> {corroborating}']
    if pick is not None:
        if pick.get("chosen"):
            parts.append('<div class="shep-mp-pick">' + r.chip(
                "D4 rule picks this record", "accent",
                title="D4's fixed rule: the first registered record from a deterministic regime "
                      "represents this checkpoint on this cohort version") + "</div>")
        elif pick.get("delta") is not None:
            parts.append(f'<div class="shep-mp-delta shep-amber-t">{r.e(pick["delta"])}</div>')
    parts.append(contest_html(state, ordered_out=bool(pick and pick.get("chosen")
                                                      and kind != "uncontested")))
    return "".join(parts)


def records_table_html(views: Sequence[Mapping[str, Any]], contest: Mapping[str, Any], *,
                       picks: Optional[Mapping[str, Mapping[str, Any]]] = None,
                       roles: Optional[Mapping[str, str]] = None) -> str:
    """The records, one main row and one row of detail each: the cohort version
    (label, version digest, synthetic, role), Mode C MRR and Hits@10 in their
    own columns, coverage, the record's state; then relations, the other Mode C
    metrics, R10 per scope, environment, dataset and registration."""
    picks = picks or {}
    body = []
    for view in views:
        cohort = view.get("cohort") if isinstance(view.get("cohort"), Mapping) else {}
        metrics = view.get("metrics") if isinstance(view.get("metrics"), Mapping) else {}
        key = view.get("key")
        state = contest.get(_key_text(key)) if key else None
        synthetic = (r.chip("synthetic", "amber") if view.get("dataset_origin") == "synthetic"
                     else "")
        role = cohort.get("role")
        role_note = (roles or {}).get(str(cohort.get("version_manifest_sha256")))
        cohort_cell = (f'<div class="shep-w600">{r.e(cohort.get("label"))}</div>'
                       f'<div class="shep-sub">(version <code>'
                       f'{r.e(r.short_hex(cohort.get("version_manifest_sha256")))}</code>) '
                       f'{synthetic}</div><div class="shep-sub">role <code>{r.e(role)}</code>'
                       + (f" · {r.e(role_note)}" if role_note else "") + "</div>")
        mrr = fmt_metric("untruncated_mrr", metrics.get("untruncated_mrr"))
        main = (f'<tr class="shep-mp-main"><td>{cohort_cell}</td>'
                f'<td class="shep-num">{r.e(mrr)}</td>'
                f'<td class="shep-num">{r.e(fmt_metric("hits@10", metrics.get("hits@10")))}</td>'
                f'<td>{coverage_html(view)}</td>'
                f'<td>{_record_cell(view, state, picks.get(str(view.get("source"))))}</td></tr>')
        others = [f"{metric_label(name)} {fmt_metric(name, value)}"
                  for name, value in sorted(metrics.items(), key=lambda kv: _metric_order(kv[0]))
                  if name not in ("untruncated_mrr", "hits@10")]
        gpu = view.get("gpu") if isinstance(view.get("gpu"), Mapping) else None
        environment = str(view.get("environment_class") or "not reported")
        if gpu is not None:
            environment += (f" · GPU {gpu.get('name') or 'name not read'}, driver "
                            f"{gpu.get('driver') or 'not read'}")
        origin = view.get("dataset_origin")
        n = view.get("n_corroborating")
        corroborating = ("no corroborating report" if n == 0 else
                         f"{n} corroborating report{'s' if n != 1 else ''}" if _number(n)
                         else "corroborating reports not reported")
        registration = (f"Registered #{_get(view, 'registration', 'sequence')} at "
                        f"{_utc_text(_get(view, 'registration', 'registered_at'))} · "
                        f"{corroborating} · from report {r.short_hex(view.get('source'))}")
        more = (f'<tr class="shep-mp-more"><td colspan="5">'
                f'<span class="shep-mp-ln">{relations_html(view.get("relations"))}'
                f'{" · " + r.e(" · ".join(others)) if others else ""}, over '
                f'{r.e(view.get("n_ranked"))} ranked cases</span>{r10_html(view.get("r10"))}'
                f'<span class="shep-mp-ln">Environment: {r.e(environment)} · Dataset: '
                f'<b>{r.e(origin)}</b> · {r.e(registration)}</span></td></tr>')
        body.append(main + more)
    return ('<div class="shep-tbl-wrap shep-mp-recwrap"><table class="shep-tbl shep-mp-rec" '
            'aria-label="Recorded test results"><colgroup><col class="shep-mp-c1"><col '
            'class="shep-mp-c2"><col class="shep-mp-c3"><col class="shep-mp-c4"><col></colgroup>'
            '<thead><tr><th scope="col">Cohort version</th>'
            '<th scope="col" class="shep-num">Mode C MRR <span class="shep-nowrap">(untruncated)'
            '</span></th>'
            '<th scope="col" class="shep-num">Mode C Hits@10</th><th scope="col">Coverage</th>'
            f'<th scope="col">Record</th></tr></thead><tbody>{"".join(body)}</tbody></table>'
            '</div>')


def _metric_order(name: str) -> Tuple[int, int, str]:
    if name == "untruncated_mrr":
        return (0, 0, name)
    if name == "mean_rank":
        return (1, 0, name)
    k = name[len("hits@"):]
    if name.startswith("hits@") and k.isdigit():
        return (2, int(k), name)
    return (3, 0, name)


def records_block_html(views: Sequence[Mapping[str, Any]], contest: Mapping[str, Any], *,
                       not_shown: int = 0,
                       picks: Optional[Mapping[str, Mapping[str, Any]]] = None,
                       roles: Optional[Mapping[str, str]] = None) -> str:
    """D3's label and the pre-B-1 note once, the dataset line, the table, the
    count of records not shown, and the R10 definitions behind a ⓘ."""
    parts = []
    if views:
        parts.append(f'<p class="shep-mp-p"><b>{r.e(MODE_C_RESULT_LABEL)}.</b> '
                     f'{r.e(PRE_B1_NOTE)}</p>')
        origins = {v.get("dataset_origin") for v in views}
        if len(origins) == 1 and next(iter(origins)) in ORIGIN_MEANING:
            origin = next(iter(origins))
            word = (f'<b class="shep-amber-t">{r.e(origin)}</b>' if origin == "synthetic"
                    else f"<b>{r.e(origin)}</b>")
            parts.append(f'<p class="shep-mp-p">Dataset of every record below: {word} — '
                         f'{r.e(ORIGIN_MEANING[origin])}.</p>')
        parts.append(records_table_html(views, contest, picks=picks, roles=roles))
        parts.append(f'<p class="shep-note">R10 labels '
                     f'{help_button("models-r10", "R10 labels")} · Mode C metrics are over the '
                     'ranked cases; the Trainer validation figure is not among them.</p>')
    if not_shown:
        many = not_shown != 1
        parts.append(f'<p class="shep-mp-p shep-amber-t">{not_shown} record{"s" if many else ""} '
                     f'for these weights {"are" if many else "is"} not shown: '
                     f'{"they do" if many else "it does"} not have the shape registration writes '
                     "(the server's log says why).</p>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# The controls, the order and the view (pure)
# ---------------------------------------------------------------------------
def architectures(listing: Mapping[str, Any]) -> List[str]:
    return sorted({str(i.get("architecture")) for i in listing.get("items") or []
                   if i.get("architecture")})


def offered_versions(data: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return [v for v in data.get("versions") or []
            if v.get("offers") and r.is_digest(v.get("sha256"))]


def scoring_groups(listing: Mapping[str, Any], data: Mapping[str, Any], version_sha: Any,
                   arch: Optional[str]) -> List[Tuple[str, int]]:
    """The scoring semantics the shown checkpoints' records on this version are
    under, with how many checkpoints each, most first."""
    counts: Dict[str, set] = {}
    for item in ml.shown_items(listing, {"architecture": arch}):
        entry = (data.get("ledgers") or {}).get(ml.item_key(item))
        for record in (entry or {}).get("records") or []:
            if record.get("cohort_digest") == version_sha:
                digest = record.get("scoring_semantics_digest")
                if isinstance(digest, str):
                    counts.setdefault(digest, set()).add(item.get("sha256"))
    return sorted(((d, len(s)) for d, s in counts.items()), key=lambda ds: (-ds[1], ds[0]))


def controls(listing: Mapping[str, Any], data: Mapping[str, Any], order: Any, version: Any,
             metric: Any, scoring: Any, arch: Any) -> Dict[str, Any]:
    """The page's controls as they can apply now: an architecture the listing
    has (else all), "By test score" only when a version offers ordering (else
    newest first), an offered version (else the first offered), an orderable
    metric, and a scoring semantics the records are under (else the one most
    checkpoints share)."""
    from src.evaluation.ordering import ORDERABLE_METRICS

    archs = architectures(listing)
    chosen_arch = arch if arch in archs else None
    offered = offered_versions(data)
    labels = [v["label"] for v in offered]
    chosen_version = version if version in labels else (labels[0] if labels else None)
    mode = BY_SCORE if (order == BY_SCORE and offered and not data.get("readonly")) else NEWEST
    chosen_metric = metric if metric in ORDERABLE_METRICS else DEFAULT_METRIC
    entry = next((v for v in offered if v["label"] == chosen_version), None)
    groups = scoring_groups(listing, data, entry["sha256"], chosen_arch) if entry else []
    digests = [d for d, _n in groups]
    chosen_scoring = scoring if scoring in digests else (digests[0] if digests else None)
    return {"arch": chosen_arch, "order": mode, "version": chosen_version, "metric": chosen_metric,
            "scoring": chosen_scoring, "scoring_groups": groups, "version_entry": entry,
            "architectures": archs}


def rank(listing: Mapping[str, Any], data: Mapping[str, Any], ctl: Mapping[str, Any]
         ) -> Optional[Dict[str, Any]]:
    """D4's order of the shown checkpoints (WP4's pure function), or None when
    the list is newest first. ``rows`` are the ordering's rows; ``keys`` maps
    each row's checkpoint digest to the listed keys holding it; ``outside``
    lists the shown keys that are not candidates (a checkpoint or a ledger that
    cannot be read), with why."""
    from src.evaluation.ordering import Candidate, order_by_test_score

    if ctl.get("order") != BY_SCORE or ctl.get("version_entry") is None:
        return None
    entry = ctl["version_entry"]
    items = ml.shown_items(listing, {"architecture": ctl.get("arch")})
    ledgers = data.get("ledgers") or {}
    candidates, outside, keys = [], {}, {}
    for item in items:
        key = ml.item_key(item)
        ledger = ledgers.get(key)
        if not ml.loadable(item):
            outside[key] = "checkpoint not readable"
            continue
        if ledger is None or ledger["state"] == "not_read":
            outside[key] = "ledger not readable — not ordered"
            continue
        keys.setdefault(item["sha256"], []).append(key)
        candidates.append(Candidate(item["sha256"], tuple(ledger["records"]),
                                    str(ledger["ledger"])))
    contest = data.get("contest") or {}
    try:
        ordering = order_by_test_score(
            candidates, cohort_digest=entry["sha256"],
            scoring_semantics_digest=ctl.get("scoring") or "", metric=ctl["metric"],
            role=entry["role"], ordering_roles=data.get("roles") or (),
            contested_state=lambda key: (contest.get(_key_text(key)) or {}).get(
                "state", "cannot_be_checked"))
    except ValueError as exc:
        return {"offered": False, "why": str(exc), "rows": [], "keys": keys, "outside": outside,
                "items": items}
    return {"offered": ordering.offered, "why": ordering.why, "rows": list(ordering.rows),
            "keys": keys, "outside": outside, "items": items}


def _regime_cell(row: Any) -> str:
    regime = row.chosen.get("regime") if isinstance(row.chosen.get("regime"), Mapping) else {}
    text = regime_text(regime.get("device"), regime.get("deterministic_algorithms"))
    mark = ('<div><span class="shep-mp-mark" title="The numerical regime differs from the one '
            'most ordered rows share">regime differs</span></div>' if row.regime_differs else "")
    return f'<div class="shep-sub shep-nowrap">{r.e(text)}</div>{mark}'


def shape(listing: Mapping[str, Any], data: Mapping[str, Any], ctl: Mapping[str, Any]
          ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """The shared list's ``view`` and ``ordering`` for these controls.

    Newest first: the architecture and the Load note only. By test score: the
    groups in D4's order (the ordered rows, each checkpoint's listed keys in the
    listing's order; then contested; then not compared; then not tested, newest
    first), the Mode C column with each row's value, record number and regime,
    and the note saying the usage history records the Load."""
    if data.get("readonly"):
        return {"note": ""}, None
    ranked = rank(listing, data, ctl)
    if ranked is None:
        return {"architecture": ctl.get("arch"), "note": LOAD_NOTE}, None
    if not ranked["offered"]:
        # Asked for, and refused by the ordering itself (a record naming
        # another role than the version's, say): newest first, and why.
        return {"architecture": ctl.get("arch"), "note": LOAD_NOTE,
                "refused": ranked["why"]}, None
    entry = ctl["version_entry"]
    metric = ctl["metric"]
    label = metric_label(metric)
    from src.evaluation.ordering import LOWER_IS_BETTER

    order_of = {ml.item_key(i): n for n, i in enumerate(ranked["items"])}
    ordered, contested, compared, untested = [], [], [], []
    cells: Dict[str, str] = {}
    for row in ranked["rows"]:
        keys = sorted(ranked["keys"].get(row.checkpoint_sha256, []), key=order_of.get)
        if row.state == "ordered":
            sequence = row.chosen.get("registration", {}).get("sequence")
            rule = " · D4 rule" if row.others else ""
            cell = (f'<div class="shep-w600">{r.e(fmt_metric(metric, row.value))}</div>'
                    f'<div class="shep-sub">#{r.e(sequence)}{r.e(rule)}</div>{_regime_cell(row)}')
            ordered.extend(keys)
        elif row.state in ("contested", "contested_unchecked"):
            kind = "shep-amber-t" if row.state == "contested" else "shep-muted"
            cell = f'<span class="{kind}">{r.e(row.label)}</span>'
            contested.extend(keys)
        elif row.state == "not_tested":
            cell = '<span class="shep-muted">—</span>'
            untested.extend(keys)
        else:
            cell = f'<span class="shep-muted">{r.e(row.label)}</span>'
            compared.extend(keys)
        for key in keys:
            cells[key] = cell
    for key, why in ranked["outside"].items():
        compared.append(key)
        cells[key] = f'<span class="shep-muted">{r.e(why)}</span>'
    groups = []
    direction = "lowest" if metric in LOWER_IS_BETTER else "highest"
    for title, keys in ((GROUP_ORDERED.format(metric=label, version=entry["label"],
                                              direction=direction), ordered),
                        (GROUP_CONTESTED, sorted(contested, key=order_of.get)),
                        (GROUP_NOT_COMPARED, sorted(compared, key=order_of.get)),
                        (GROUP_NOT_TESTED, sorted(untested, key=order_of.get))):
        if keys:
            groups.append({"title": title, "keys": keys})
    # The column says the metric briefly (the group's title names it in full).
    short = "Mode C MRR" if metric == "untruncated_mrr" else label
    view = {"architecture": ctl.get("arch"), "groups": groups, "rest_title": GROUP_UNREAD,
            "column": f"{short}, {entry['label']} · regime", "cells": cells,
            "note": LOAD_NOTE_ORDERED.format(metric=label, version=entry["label"])}
    ordering = {"version": entry["label"], "version_sha256": entry["sha256"], "metric": metric,
                "scoring_semantics_digest": ctl.get("scoring") or ""}
    return view, ordering


def picks_for(item: Mapping[str, Any], ranked: Optional[Mapping[str, Any]],
              metric: str) -> Dict[str, Dict[str, Any]]:
    """Which of a checkpoint's records D4's rule picked on the ordered version,
    and how each other record of its group differs from it, by report digest."""
    if not ranked:
        return {}
    row = next((x for x in ranked.get("rows") or []
                if x.checkpoint_sha256 == item.get("sha256") and x.chosen is not None), None)
    if row is None:
        return {}
    chosen = row.chosen
    out = {str(chosen.get("source_artifact_digest")): {"chosen": True}}
    base = _get(chosen, "metrics", metric)
    sequence = _get(chosen, "registration", "sequence")
    for other in row.others:
        value = _get(other, "metrics", metric)
        parts = []
        if _number(value) and _number(base):
            parts.append(f"Δ {metric_label(metric)} {value - base:+.4f} vs #{sequence}")
        flag = _get(other, "regime", "deterministic_algorithms")
        if flag != _get(chosen, "regime", "deterministic_algorithms") and type(flag) is bool:
            parts.append("deterministic " + ("on" if flag else "off"))
        out[str(other.get("source_artifact_digest"))] = {"chosen": False,
                                                         "delta": " · ".join(parts) or None}
    return out


# ---------------------------------------------------------------------------
# The page's HTML (pure)
# ---------------------------------------------------------------------------
def access_short(listing: Mapping[str, Any], condition: Optional[str]) -> str:
    return D8_SHORT.get(condition or "", "request not readable")


def left_header_html(listing: Mapping[str, Any], data: Mapping[str, Any],
                     condition: Optional[str] = None) -> str:
    """The list panel's header; for a read-only request, why there is no list
    (D8's three conditions named together) instead."""
    if listing.get("refused"):
        short = access_short(listing, condition)
        return (f'<div class="shep-panel-hd"><h2>Models</h2></div>'
                f'<div class="shep-mp-ro" role="status"><div class="shep-mp-refusal">'
                f'<b>Model list unavailable: read-only, {r.e(short)}.</b>'
                f'<span>{r.e(LIST_IS_C3)}</span><span>{r.e(LIST_C3_LINE)}</span>'
                f'<span class="shep-sub">{r.e(listing.get("problem"))}</span></div>'
                f'<div class="shep-reason">{r.e(D8_CONDITIONS)}</div></div>')
    return ('<div class="shep-panel-hd"><h2>Models</h2><span class="shep-panel-right">🔒 listed '
            'by the server (operator controls only)</span></div>')


def options_html(data: Mapping[str, Any], ctl: Mapping[str, Any],
                 refused: Optional[str] = None) -> str:
    """Under the order options: the versions that do not offer ordering, with
    why; why ordering is not offered at all; and, when the ordering itself
    refused what was asked (`refused`, the view's), that the list stays newest
    first and why."""
    if data.get("readonly"):
        return ""
    versions = data.get("versions") or []
    parts = []
    off = []
    for version in versions:
        if version.get("offers"):
            continue
        if version.get("problem"):
            why = f"not read: {r.short_reason(version['problem'], 120)}"
        elif version.get("role") == "institutional_acceptance":
            why = ROLE_NEVER + (ROLE_NEVER_LISTED if data.get("roles_note") else "")
        else:
            why = ROLE_NOT_LISTED
        off.append(f'<div class="shep-mp-off" title="{r.e(version.get("problem") or why)}">'
                   f'<span class="shep-mp-radio" aria-hidden="true"></span><span><b>'
                   f'{r.e(version.get("label"))}</b> (role <code>{r.e(version.get("role"))}'
                   f'</code>) — {r.e(why)}</span></div>')
    if refused:
        parts.append(f'<div class="shep-reason shep-amber-t">Not ordered, so the list stays '
                     f'newest first: {r.e(r.short_reason(refused, 240))}</div>')
    if not offered_versions(data):
        reason = (data.get("problem") or "no cohort version's role is in the server's "
                  "model-selection roles (SHEPHERD_ORDERING_ROLES)")
        parts.append(f'<div class="shep-reason">Ordering by test score is not offered: '
                     f'{r.e(r.short_reason(reason, 220))}</div>')
    if ctl.get("order") == BY_SCORE and off:
        parts.append('<div class="shep-mp-offs">' + "".join(off) + "</div>")
    return "".join(parts)


def independence_html(data: Mapping[str, Any]) -> str:
    """§4.9 question 1's note, wherever ordering is offered (and nowhere else)."""
    if data.get("readonly") or not offered_versions(data):
        return ""
    return (f'<div class="shep-notice shep-notice-amber shep-mp-indep" role="note"><b>'
            f'{r.e(INDEPENDENCE_NOTE)}</b> {r.e(INDEPENDENCE_DETAIL)}</div>')


def version_choices(data: Mapping[str, Any]) -> List[Tuple[str, str]]:
    return [(f"{v['label']} (role {v['role']}) — {ROLE_ON}", v["label"])
            for v in offered_versions(data)]


def scoring_choices(groups: Sequence[Tuple[str, int]]) -> List[Tuple[str, str]]:
    return [(f"scoring semantics {d[:12]} · {n} checkpoint{'s' if n != 1 else ''}", d)
            for d, n in groups]


def _served_line(listing: Mapping[str, Any]) -> str:
    served = listing.get("served_sha")
    if served:
        return (f'in use for diagnosis: model <code class="shep-fpv" title="{r.e(served)}">'
                f'{r.e(served[:12])}</code>')
    if listing.get("status_problem"):
        return "in use for diagnosis: not read"
    return "nothing is in use for diagnosis"


def detail_parts(listing: Mapping[str, Any], data: Mapping[str, Any], key: Any,
                 ctl: Mapping[str, Any]) -> Dict[str, Any]:
    """The detail of the selected checkpoint (operator), or the read-only view.
    Returns ``header``, ``top``, ``records``, ``usage`` (HTML) and ``seg``
    (the segment labels, or None when the segments are not shown)."""
    if data.get("readonly"):
        return readonly_parts(data)
    item = ml.find_item(listing, key)
    if item is None or item not in ml.shown_items(listing, {"architecture": ctl.get("arch")}):
        return {"header": ('<div class="shep-panel-hd"><h2>Details</h2>'
                           f'<span class="shep-panel-right">{_served_line(listing)}</span></div>'),
                "top": ('<p class="shep-muted shep-mp-p">Select a model to see its fingerprints, '
                        'its validation figure and its recorded test results.</p>'),
                "records": "", "usage": "", "seg": None}
    in_use = bool(listing.get("served_sha")) and item.get("sha256") == listing.get("served_sha")
    tag = r.chip("In use (Diagnosis)", "accent") if in_use else r.chip("not in use", "grey")
    header = (f'<div class="shep-panel-hd"><h2>{r.e(r.model_name_line(item))}</h2>{tag}'
              f'<span class="shep-panel-right">{_served_line(listing)}</span></div>')
    workspaces = listing.get("workspaces") or []
    if not ml.loadable(item):
        why = item.get("problem") or "its digest was not read"
        top = (f'<div class="shep-reason">{r.e(item.get("filename"))} · metadata not readable · '
               f'cannot be loaded: {r.e(why)}</div>')
        return {"header": header, "top": top, "records": "", "usage": "", "seg": None}
    sha = item["sha256"]
    ident = (f'<div class="shep-digest"><span class="shep-dl">Model fingerprint (checkpoint '
             f'sha256) · <code>{r.e(item.get("architecture"))}/{r.e(item.get("run"))}</code> · '
             f'{r.e(item.get("filename"))}{help_button("model-fingerprint", "Model fingerprint")}'
             f'</span><code class="shep-full">{r.e(sha)}</code>'
             f'{copy_button(sha, "the full model fingerprint")}</div>')
    trained = r.trained_on_html(item.get("dataset_fingerprint"), workspaces)
    marks = []
    if item.get("trained_on_this_dataset") is True:
        marks.append(r.chip("✓ this workspace's current dataset", "ok"))
    elif item.get("trained_on_this_dataset") is False:
        marks.append(r.chip("not this workspace's current dataset", "amber"))
    if r.is_synthetic_training(item.get("dataset_fingerprint"), workspaces):
        marks.append(r.chip("synthetic", "amber"))
    facts = r.facts_from_checkpoint(item, workspaces)
    validation = (f'<div class="shep-card shep-mp-val"><div class="shep-mp-valbd"><b>Validation '
                  f'(Trainer)</b>{help_button("validation-trainer", "Validation (Trainer)")} '
                  f'<span>{r.e(r.validation_text(facts.validation))}</span></div></div>')
    # Model Facts' warnings, but its synthetic banner: that one speaks of "the
    # served workspace", which a listed checkpoint need not be; the amber chip
    # above and the records' dataset line say synthetic for this checkpoint.
    warnings = "".join(f'<p class="shep-warn"><b>⚠ {r.e(h)}</b> <span>{r.e(d)}</span></p>'
                       for h, d in facts.warnings if (h, d) != r.IDENTITY_BANNERS["synthetic"])
    top = (ident + f'<div class="shep-mp-trained">{trained} {" ".join(marks)}</div>' + validation
           + warnings)
    ledger = (data.get("ledgers") or {}).get(ml.item_key(item))
    ranked = rank(listing, data, ctl)
    records, count, contested = _records_for_item(item, ledger, data, ranked, ctl)
    usage, n_usage = usage_html(item, data)
    seg_records = f"Recorded test results (Mode C) · {count}" + (
        f" · {contested} contested" if contested else "")
    return {"header": header, "top": top, "records": records, "usage": usage,
            "seg": [(seg_records, RECORDS), (f"Usage history · {n_usage}", USAGE)]}


def _records_for_item(item: Mapping[str, Any], ledger: Optional[Mapping[str, Any]],
                      data: Mapping[str, Any], ranked: Optional[Mapping[str, Any]],
                      ctl: Mapping[str, Any]) -> Tuple[str, int, int]:
    from src.evaluation.sidecar import LEDGER_FILENAME

    if ledger is None:
        return ('<p class="shep-mp-p shep-muted">The ledger beside this checkpoint was not '
                'read.</p>', 0, 0)
    if ledger["state"] == "not_read":
        return (f'<p class="shep-mp-p shep-amber-t">The ledger (<code>{LEDGER_FILENAME}</code>) '
                f'beside this checkpoint was not read, so none of its records are shown: '
                f'{r.e(ledger.get("problem"))}</p>', 0, 0)
    if ledger["state"] == "no_ledger":
        return (f'<p class="shep-mp-p">No ledger (<code>{LEDGER_FILENAME}</code>) is beside this '
                'checkpoint, so no recorded result can be shown; that does not mean these '
                'weights were never evaluated.</p>', 0, 0)
    if ledger["state"] == "none":
        return f'<p class="shep-mp-p">{r.e(RECORD_ABSENCE)}</p>', 0, 0
    views, not_shown = [], 0
    for record in ledger["records"]:
        view = record_view(record)
        if view is None:
            not_shown += 1
        else:
            views.append(view)
    picks = picks_for(item, ranked, ctl.get("metric") or DEFAULT_METRIC)
    entry = ctl.get("version_entry") if ranked else None
    first = entry["sha256"] if entry else None
    views.sort(key=lambda v: (0 if picks.get(str(v.get("source")), {}).get("chosen") else
                              1 if _get(v, "cohort", "version_manifest_sha256") == first else 2,
                              -(_get(v, "registration", "sequence") or 0)))
    contest = data.get("contest") or {}
    roles = {}
    for version in data.get("versions") or []:
        if not version.get("offers") and version.get("role") == "institutional_acceptance":
            roles[str(version.get("sha256"))] = "ordering off: awaits the formal evaluation rules"
    contested = sum(1 for v in views
                    if (contest.get(_key_text(v["key"])) or {}).get("state") == "contested")
    return (records_block_html(views, contest, not_shown=not_shown, picks=picks, roles=roles),
            len(views) + not_shown, contested)


def usage_entry_html(entry: Mapping[str, Any]) -> str:
    kind = entry.get("kind")
    when = _utc_text(entry.get("at"))
    version = r.e(entry.get("version"))
    if kind == "test_run":
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), Mapping) else {}
        mrr = metrics.get("untruncated_mrr")
        what = "Test run"
        result = (f"{metric_label('untruncated_mrr')} {fmt_metric('untruncated_mrr', mrr)} over "
                  f"{entry.get('n_ranked')} ranked cases · "
                  f"{regime_text(entry.get('device'), entry.get('deterministic'))}"
                  + (" · registration asked" if entry.get("record_requested") else ""))
        detail = f"run {entry.get('run_id')}"
    elif kind == "registered":
        what = "Registered"
        result = f"#{entry.get('sequence')} · {entry.get('outcome')}"
        detail = f"report of run {entry.get('run_id')}"
    elif kind == "model_load":
        what = "Model load from an ordered list"
        loaded = entry.get("loaded")
        result = ("loaded" if loaded is True else
                  f"not loaded: {entry.get('why') or 'no reason recorded'}")
        detail = (f"ordered by {metric_label(entry.get('metric'))} · scoring semantics "
                  f"{r.short_hex(entry.get('scoring_semantics_digest'))}")
    else:
        what, result, detail = str(kind or "entry"), "", ""
    return (f'<tr><td class="shep-nowrap">{r.e(when)}</td><td>{version}</td>'
            f'<td><div class="shep-w600">{r.e(what)}</div><div class="shep-sub">{r.e(detail)}'
            f'</div></td><td>{r.e(result)}</td></tr>')


def usage_html(item: Mapping[str, Any], data: Mapping[str, Any]) -> Tuple[str, int]:
    """The usage history entries naming this checkpoint, newest first."""
    usage = data.get("usage") or {}
    entries = (usage.get("entries") or {}).get(item.get("sha256")) or []
    read_from = usage.get("read_from") or []
    parts = [f'<p class="shep-mp-p">Entries naming this checkpoint in the usage history of '
             f'every listed cohort version{help_button("models-usage", "Usage history")}: '
             f'{r.e(", ".join(read_from)) if read_from else "no cohort version was read"}.</p>']
    if entries:
        shown = entries[:USAGE_SHOWN]
        more = (f'<p class="shep-note">The latest {USAGE_SHOWN} of {len(entries)} entries are '
                'shown.</p>' if len(entries) > USAGE_SHOWN else "")
        parts.append('<div class="shep-tbl-wrap shep-mp-recwrap"><table class="shep-tbl '
                     'shep-mp-usage" aria-label="Usage history"><thead><tr><th scope="col">When'
                     '</th><th scope="col">Cohort version</th><th scope="col">What</th>'
                     '<th scope="col">Result</th></tr></thead><tbody>'
                     + "".join(usage_entry_html(e) for e in shown) + "</tbody></table></div>"
                     + more)
    else:
        parts.append('<p class="shep-mp-p shep-muted">No usage history entry names this '
                     'checkpoint.</p>')
    for problem in usage.get("problems") or []:
        parts.append(f'<p class="shep-mp-p shep-amber-t">{r.e(problem)}</p>')
    return "".join(parts), len(entries)


def readonly_parts(data: Mapping[str, Any]) -> Dict[str, Any]:
    """"Recorded test results of the model in use": the served digest (from
    the status), then the route's records for it, each with its contested
    state. Nothing here opened a checkpoint."""
    header = (f'<div class="shep-panel-hd"><h2>{r.e(READONLY_TITLE)}</h2>'
              f'<span class="shep-panel-right">{r.e(READONLY_WHERE)}</span></div>')
    answer = data.get("answer") or {}
    served = data.get("served_sha")
    state = answer.get("state")
    note = f'<p class="shep-note">{r.e(NO_LOAD_HERE)}</p>'
    if data.get("status_problem"):
        top = (f'<p class="shep-mp-p shep-muted">What is served cannot be read now '
               f'({r.e(data["status_problem"])}), so no recorded result is shown.</p>')
        return {"header": header, "top": top, "records": note, "usage": "", "seg": None}
    if served:
        top = (f'<div class="shep-digest"><span class="shep-dl">Model in use: model fingerprint '
               f'(checkpoint sha256, the status read {r.e(data.get("status_at"))}) · records are '
               f'found by this digest</span><code class="shep-full">{r.e(served)}</code>'
               f'{copy_button(served, "the full model fingerprint of the model in use")}</div>')
    else:
        top = ""
    if state == "unavailable" or state not in ("no_model", "not_identified", "no_ledger",
                                               "ledger_not_read", "no_records", "records"):
        body = '<p class="shep-mp-p shep-amber-t">The recorded results could not be read.</p>'
    elif answer.get("checkpoint_sha256") != served:
        body = f'<p class="shep-mp-p shep-amber-t">{r.e(RECORDS_MODEL_CHANGED)}</p>'
    elif state == "no_model":
        body = ('<p class="shep-mp-p">No trained model is loaded, so there are no recorded '
                'results to look up.</p>')
    elif state == "not_identified":
        body = ('<p class="shep-mp-p">The loaded model was supplied in memory, not read from a '
                'checkpoint file, so it has no digest to look recorded results up by.</p>')
    elif state == "no_ledger":
        body = ('<p class="shep-mp-p">No ledger (<code>evaluations.json</code>) is beside the '
                'loaded checkpoint, so no recorded result can be shown; that does not mean these '
                'weights were never evaluated.</p>')
    elif state == "ledger_not_read":
        body = (f'<p class="shep-mp-p shep-amber-t">The ledger beside the loaded checkpoint was '
                f'not read, so none of its records are shown: '
                f'{r.e(_get(answer, "ledger", "problem"))}</p>')
    elif state == "no_records":
        body = f'<p class="shep-mp-p">{r.e(RECORD_ABSENCE)}</p>'
    else:
        views = [view_from_route(rec, answer.get("checkpoint_sha256"))
                 for rec in answer.get("records") or [] if isinstance(rec, Mapping)]
        views.sort(key=lambda v: -(_get(v, "registration", "sequence") or 0))
        body = records_block_html(views, data.get("contest") or {},
                                  not_shown=int(answer.get("records_not_shown") or 0))
    return {"header": header, "top": top, "records": body + note, "usage": "", "seg": None}


# ---------------------------------------------------------------------------
# The component
# ---------------------------------------------------------------------------
def create_models_page(ctx: Any) -> None:
    """Build Research › Evaluate › Models (the page contract: `src/webui/shell.py`)."""
    ctx.help.topics(PAGE_HELP)
    data_state = gr.State({"readonly": False, "listing_key": None, "versions": [], "ledgers": {},
                           "contest": {}, "usage": {}})
    parts: Dict[str, Any] = {}

    def head() -> Dict[str, Any]:
        parts["lhd"] = plain_html(left_header_html({}, {}), elem_id="shep-mp-hd")
        parts["arch"] = gr.Dropdown(label="Architecture",
                                    choices=[("All architectures", ALL_ARCHITECTURES)],
                                    value=ALL_ARCHITECTURES, interactive=True,
                                    elem_id="shep-mp-arch")
        with gr.Column(elem_id="shep-mp-controls", elem_classes=["shep-mp-controls"]):
            with gr.Row(elem_classes=["shep-mp-order-row"]):
                plain_html('<span class="shep-mp-lbl">Order</span>', elem_id="shep-mp-order-lbl")
                parts["order"] = gr.Radio(choices=ORDER_CHOICES, value=NEWEST, label="Order",
                                          show_label=False, interactive=False,
                                          elem_id="shep-mp-order", elem_classes=["shep-seg"],
                                          container=False)
                plain_html(help_button("models-ordering", "Ordering by test score"),
                           elem_id="shep-mp-order-i")
            # First under the switch, so it stays in view however short the window.
            parts["independence"] = plain_html("", elem_id="shep-mp-independence")
            with gr.Column(elem_classes=["shep-mp-opts"], visible=False) as opts:
                parts["opts"] = opts
                with gr.Row(elem_classes=["shep-mp-opts-row"]):
                    plain_html('<span class="shep-mp-lbl">By</span>', elem_id="shep-mp-by")
                    parts["metric"] = gr.Dropdown(label="By", show_label=False, container=False,
                                                  choices=[(METRIC_LABELS[m], m)
                                                           for m in METRIC_LABELS],
                                                  value=DEFAULT_METRIC, interactive=True,
                                                  elem_id="shep-mp-metric")
                    plain_html('<span class="shep-mp-lbl">on cohort version:</span>',
                               elem_id="shep-mp-on")
                parts["version"] = gr.Radio(label="On cohort version", show_label=False,
                                            container=False, choices=[], value=None,
                                            interactive=True, elem_id="shep-mp-version")
                parts["scoring"] = gr.Dropdown(label="Scoring semantics", choices=[], value=None,
                                               interactive=True, visible=False,
                                               elem_id="shep-mp-scoring")
            parts["options"] = plain_html("", elem_id="shep-mp-options")
        return parts

    def foot() -> Any:
        with gr.Column(elem_id="shep-mp-detail", elem_classes=["shep-mp-detail"]) as column:
            parts["detail_hd"] = plain_html(detail_parts({}, {}, None, {})["header"],
                                            elem_id="shep-mp-detail-hd")
            with gr.Column(elem_id="shep-mp-detail-bd", elem_classes=["shep-mp-detail-bd"]):
                parts["detail_top"] = plain_html(detail_parts({}, {}, None, {})["top"],
                                                 elem_id="shep-mp-detail-top")
                parts["seg"] = gr.Radio(choices=[("Recorded test results (Mode C)", RECORDS),
                                                 ("Usage history", USAGE)], value=RECORDS,
                                        show_label=False, label="Shown", visible=False,
                                        elem_id="shep-mp-seg", elem_classes=["shep-seg"],
                                        container=False)
                parts["records"] = plain_html("", elem_id="shep-mp-records")
                parts["usage"] = plain_html("", elem_id="shep-mp-usage", visible=False)
        return column

    with gr.Row(elem_classes=["shep-page", "shep-models"], equal_height=False):
        handle = ctx.create_model_list("research", head=head, foot=foot)

    control_inputs = [parts["order"], parts["version"], parts["metric"], parts["scoring"],
                      parts["arch"]]
    page_outputs = [data_state, handle.view, handle.ordering, parts["arch"], parts["order"],
                    parts["opts"], parts["version"], parts["scoring"], parts["options"],
                    parts["independence"], parts["lhd"], parts["detail_hd"],
                    parts["detail_top"], parts["seg"], parts["records"], parts["usage"]]

    def detail_updates(listing, data, selected, ctl, seg_value) -> Tuple[Any, ...]:
        """The detail's header, top, segments, records and usage history; the
        segment shown is kept across selections."""
        detail = detail_parts(listing, data, selected, ctl)
        seg_shown = detail["seg"] is not None
        seg = seg_value if seg_value in (RECORDS, USAGE) else RECORDS
        return (detail["header"], detail["top"],
                (gr.update(choices=detail["seg"], value=seg, visible=True) if seg_shown
                 else gr.update(visible=False)),
                gr.update(value=detail["records"], visible=(not seg_shown) or seg == RECORDS),
                gr.update(value=detail["usage"], visible=seg_shown and seg == USAGE))

    def outputs_for(listing, data, ctl_values, selected, seg_value, condition, *,
                    new_data: bool) -> Tuple[Any, ...]:
        ctl = controls(listing, data, *ctl_values)
        view, ordering = shape(listing, data, ctl)
        offered = bool(offered_versions(data)) and not data.get("readonly")
        arch_choices = [("All architectures", ALL_ARCHITECTURES)] + [
            (a.upper(), a) for a in ctl["architectures"]]
        return (
            data if new_data else gr.update(),
            view, ordering,
            gr.update(choices=arch_choices, value=ctl["arch"] or ALL_ARCHITECTURES),
            gr.update(value=ctl["order"], interactive=offered),
            gr.update(visible=ctl["order"] == BY_SCORE),
            gr.update(choices=version_choices(data), value=ctl["version"]),
            gr.update(choices=scoring_choices(ctl["scoring_groups"]), value=ctl["scoring"],
                      visible=len(ctl["scoring_groups"]) > 1),
            options_html(data, ctl, view.get("refused")),
            independence_html(data),
            left_header_html(listing, data, condition),
            *detail_updates(listing, data, selected, ctl, seg_value),
        )

    def read_for(listing) -> Dict[str, Any]:
        if listing.get("refused"):
            return read_readonly()
        return read_page(listing)

    shared_refresh = handle.refresh["fn"]

    def _open(name, older, current_view, order, version, metric, scoring, arch, seg_value,
              request: gr.Request):
        shared = shared_refresh(name, older, current_view, request)
        listing = shared[0]
        data = read_for(listing)
        condition = operator_access(request).condition
        return (*shared, *outputs_for(listing, data, (order, version, metric, scoring, arch),
                                      None, seg_value, condition, new_data=True))

    ctx.on_page_open("models", {
        "fn": _open,
        "inputs": [*handle.refresh["inputs"], *control_inputs, parts["seg"]],
        "outputs": [*handle.refresh["outputs"], *page_outputs],
        "show_progress": "minimal"})

    def _recompute(listing, data, order, version, metric, scoring, arch, selected, seg_value,
                   request: gr.Request):
        # The shared list was read again (a workspace chosen, after a Load):
        # this page reads its own data again only for a listing it has not read.
        if isinstance(data, Mapping) and data.get("listing_key") == listing_key(listing) \
                and not listing.get("refused"):
            return tuple(gr.update() for _ in page_outputs)
        fresh = read_for(listing)
        condition = operator_access(request).condition
        return outputs_for(listing, fresh, (order, version, metric, scoring, arch), selected,
                           seg_value, condition, new_data=True)

    handle.listing.change(_recompute, inputs=[handle.listing, data_state, *control_inputs,
                                              handle.selected, parts["seg"]],
                          outputs=page_outputs, show_progress="hidden")

    def _reorder(listing, data, order, version, metric, scoring, arch, selected, seg_value,
                 request: gr.Request):
        condition = operator_access(request).condition
        return outputs_for(listing, data, (order, version, metric, scoring, arch), selected,
                           seg_value, condition, new_data=False)

    for control in control_inputs:
        control.input(_reorder, inputs=[handle.listing, data_state, *control_inputs,
                                        handle.selected, parts["seg"]],
                      outputs=page_outputs, show_progress="hidden")

    def _detail(listing, data, selected, order, version, metric, scoring, arch, seg_value):
        ctl = controls(listing, data, order, version, metric, scoring, arch)
        return detail_updates(listing, data, selected, ctl, seg_value)

    detail_outputs = [parts["detail_hd"], parts["detail_top"], parts["seg"], parts["records"],
                      parts["usage"]]
    handle.selected.change(_detail, inputs=[handle.listing, data_state, handle.selected,
                                            *control_inputs, parts["seg"]],
                           outputs=detail_outputs, show_progress="hidden")

    def _segment(value):
        return (gr.update(visible=value != USAGE), gr.update(visible=value == USAGE))

    parts["seg"].input(_segment, inputs=[parts["seg"]],
                       outputs=[parts["records"], parts["usage"]], queue=False,
                       show_progress="hidden")


_PAGE = "#" + page_elem_id("models")
_LIST = "#shep-models-research"

#: This page's style sheet (the page contract: listed in `src/webui/app.py`'s
#: ``PAGE_CSS``). The shared list's column is laid out as the mockup draws the
#: page: the list panel (its header, the workspace and architecture selects, the
#: order controls, the count and the table) on the left; the selected model's
#: detail with the Load footer on the right, the confirmation drawn over them
#: while it is open. The research preset's Model Facts are not shown here: the
#: detail shows the same identity and warnings with the recorded results.
PAGE_CSS = f"""
{_PAGE} .shep-models {{ gap: 0 !important; }}
{_PAGE} {_LIST} {{
  display: grid !important; gap: 0 16px !important; width: 100%;
  grid-template-columns: minmax(0, 46fr) minmax(0, 54fr);
  grid-template-rows: auto minmax(0, 1fr);
  grid-template-areas: "head foot" "body foot";
  padding: 0 !important; border: 0 !important; background: none !important;
}}
{_PAGE} {_LIST} > .shep-model-head {{
  grid-area: head; display: grid !important; gap: 10px 10px !important; align-items: start;
  align-content: start;
  grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr);
  grid-template-rows: repeat(4, max-content);
  grid-template-areas: "lhd lhd" "ws arch" "ctl ctl" "count count";
  background: var(--shep-panel); border: 1px solid var(--shep-border); border-bottom: 0;
  border-radius: 10px 10px 0 0; padding: 0 16px 8px !important; margin: 0 !important;
}}
{_PAGE} {_LIST} > .shep-model-head > * {{ min-width: 0 !important; }}
{_PAGE} #shep-mp-hd {{ grid-area: lhd; margin: 0 -16px; width: calc(100% + 32px) !important; max-width: none; }}
{_PAGE} {_LIST} > .shep-model-head > .form:has(#shep-models-research-workspace) {{ grid-area: ws; }}
{_PAGE} {_LIST} > .shep-model-head > .form:has(#shep-mp-arch) {{ grid-area: arch; }}
{_PAGE} #shep-mp-controls {{ grid-area: ctl; gap: 8px !important; }}
{_PAGE} #shep-models-research-count {{ grid-area: count; font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} {_LIST} > .shep-model-body {{
  grid-area: body; min-height: 0; background: var(--shep-panel); border: 1px solid var(--shep-border);
  border-top: 0; border-radius: 0 0 10px 10px; padding: 0 16px 12px !important; margin: 0 !important;
  flex-wrap: nowrap !important;
}}
{_PAGE} #shep-models-research-rows {{ flex: 1 1 auto !important; min-width: 0; }}
{_PAGE} #shep-models-research-facts {{ display: none !important; }}
{_PAGE} {_LIST} > .shep-model-foot {{
  grid-area: foot; min-height: 0; display: grid !important; margin: 0 !important;
  grid-template-columns: auto minmax(0, auto) minmax(0, 1fr); grid-template-rows: minmax(0, 1fr) auto;
  grid-template-areas: "detail detail detail" "load reason note";
  align-items: center; gap: 0 12px !important; padding: 0 !important;
  background: var(--shep-panel); border: 1px solid var(--shep-border); border-radius: 10px;
}}
{_PAGE} {_LIST} > .shep-model-foot > #shep-mp-detail {{ grid-area: detail; align-self: stretch; min-height: 0; gap: 0 !important; }}
{_PAGE} {_LIST} > .shep-model-foot > #shep-models-research-load {{ grid-area: load; margin: 12px 0 12px 16px; }}
{_PAGE} {_LIST} > .shep-model-foot > #shep-models-research-reason {{ grid-area: reason; }}
{_PAGE} {_LIST} > .shep-model-foot > #shep-models-research-note {{ grid-area: note; padding-right: 16px; }}
{_PAGE} {_LIST} > .shep-model-foot::after {{
  content: ""; grid-row: 2; grid-column: 1 / -1; border-top: 1px solid var(--shep-border);
  align-self: start; pointer-events: none;
}}
{_PAGE} {_LIST} > .shep-model-confirm-box {{
  grid-area: foot; z-index: 3; min-height: 0; overflow: auto; margin: 0 !important;
  background: var(--shep-panel); border: 1px solid var(--shep-accent-100); border-radius: 10px;
  padding: 14px 16px !important;
}}
{_PAGE} #shep-mp-detail-bd {{ padding: 12px 16px !important; gap: 12px !important; min-height: 0; flex-wrap: nowrap !important; }}
{_PAGE} #shep-mp-detail-bd > * {{ flex-shrink: 0 !important; }}
{_PAGE} #shep-mp-seg {{ overflow: visible !important; }}
{_PAGE} #shep-mp-detail-hd .shep-panel-hd h2, {_PAGE} #shep-mp-hd .shep-panel-hd h2 {{ overflow-wrap: anywhere; }}
{_PAGE} .shep-mp-order-row {{ align-items: center !important; gap: 8px !important; flex-wrap: nowrap !important; }}
{_PAGE} .shep-mp-order-row > * {{ flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }}
{_PAGE} .shep-mp-lbl {{ font-size: 15px; color: var(--shep-text-2); white-space: nowrap; }}
{_PAGE} .shep-mp-opts {{
  border: 1px solid var(--shep-border); border-radius: 8px; background: var(--shep-panel-2);
  padding: 8px 10px !important; gap: 6px !important;
}}
{_PAGE} .shep-mp-opts-row {{ gap: 8px !important; align-items: center !important; flex-wrap: wrap !important; }}
{_PAGE} .shep-mp-opts-row > * {{ flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; margin: 0 !important; }}
{_PAGE} .shep-mp-opts-row > .form {{ flex: 0 1 300px !important; border: 0 !important; background: none !important; }}
{_PAGE} #shep-mp-version .wrap {{ flex-direction: column; align-items: flex-start; gap: 4px; }}
{_PAGE} #shep-mp-version .wrap label {{
  background: none !important; border: 0 !important; box-shadow: none !important;
  padding: 2px 0 !important; color: var(--shep-text) !important; font-size: 15px;
}}
{_PAGE} #shep-mp-version .wrap label span {{ color: inherit !important; }}
{_PAGE} #shep-mp-version .wrap label.selected {{ font-weight: 600; }}
{_PAGE} .shep-mp-offs {{ display: flex; flex-direction: column; gap: 4px; padding: 0 10px; }}
{_PAGE} .shep-mp-off {{ display: flex; gap: 8px; align-items: flex-start; font-size: 14px; color: var(--shep-muted); }}
{_PAGE} .shep-mp-radio {{ flex: none; width: 14px; height: 14px; margin-top: 3px; border-radius: 50%;
  border: 1px solid var(--shep-grey-200); background: var(--shep-grey-50); }}
{_PAGE} .shep-mp-indep {{ font-size: 14px; }}
{_PAGE} .shep-mp-mark {{ display: inline-block; margin-top: 2px; padding: 0 6px; border-radius: 6px;
  background: var(--shep-amber-50); border: 1px solid var(--shep-amber-300); color: var(--shep-amber-800);
  font-size: 14px; font-weight: 600; white-space: nowrap; }}
{_PAGE} .shep-mp-p {{ margin: 0; font-size: 15px; color: var(--shep-text-2); line-height: 1.45; }}
{_PAGE} .shep-mp-trained {{ display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: center; font-size: 15px; }}
{_PAGE} .shep-mp-val {{ padding: 10px 12px; }}
{_PAGE} .shep-mp-valbd {{ display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: center; font-size: 15px; }}
{_PAGE} .shep-mp-recwrap {{ margin: 0 !important; }}
{_PAGE} .shep-mp-rec td, {_PAGE} .shep-mp-usage td {{ overflow-wrap: anywhere; }}
{_PAGE} .shep-mp-rec {{ table-layout: fixed; }}
{_PAGE} .shep-mp-rec .shep-mp-c1 {{ width: 24%; }}
{_PAGE} .shep-mp-rec .shep-mp-c2 {{ width: 14%; }}
{_PAGE} .shep-mp-rec .shep-mp-c3 {{ width: 11%; }}
{_PAGE} .shep-mp-rec .shep-mp-c4 {{ width: 25%; }}
{_PAGE} .shep-mp-rec th {{ overflow-wrap: anywhere; }}
{_PAGE} #shep-mp-metric, {_PAGE} #shep-mp-scoring {{ max-width: 360px; }}
{_PAGE} .shep-mp-opts-row > #shep-mp-metric {{ min-width: 290px !important; }}
{_PAGE} .shep-mp-rec tr.shep-mp-main td {{ border-bottom: 0; }}
{_PAGE} .shep-mp-rec tr.shep-mp-more td {{ background: var(--shep-panel-2); font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} .shep-mp-ln {{ display: block; line-height: 1.45; }}
{_PAGE} .shep-mp-delta {{ font-size: 14px; margin-top: 4px; }}
{_PAGE} .shep-mp-pick {{ margin-top: 4px; }}
{_PAGE} .shep-mp-refusal {{ display: flex; flex-direction: column; gap: 6px; margin: 12px 0;
  border: 1px solid var(--shep-grey-200); background: var(--shep-grey-50); border-radius: 8px;
  padding: 12px 14px; font-size: 15px; color: var(--shep-grey-800); }}
{_PAGE} .shep-mp-ro {{ padding: 0 16px 8px; }}
{_PAGE} .shep-mtbl td.shep-num .shep-sub {{ text-align: right; }}
/* Read-only: no list, no workspace, no order; the reason is the header's. */
{_PAGE} {_LIST}:has(.shep-mp-ro) > .shep-model-head > .form,
{_PAGE} {_LIST}:has(.shep-mp-ro) #shep-mp-controls,
{_PAGE} {_LIST}:has(.shep-mp-ro) #shep-models-research-count,
{_PAGE} {_LIST}:has(.shep-mp-ro) > .shep-model-body {{ display: none !important; }}
{_PAGE} {_LIST}:has(.shep-mp-ro) > .shep-model-head {{ grid-row: 1 / -1; border-bottom: 1px solid var(--shep-border); border-radius: 10px; align-content: start; }}
{_PAGE} {_LIST}:has(.shep-mp-ro) {{ grid-template-columns: minmax(0, 30fr) minmax(0, 70fr); }}
@media (min-width: 1400px) and (min-height: 760px) {{
  {_PAGE} > .column {{ height: 100%; }}
  {_PAGE} .shep-models {{ height: 100%; min-height: 0; }}
  {_PAGE} {_LIST} {{ height: 100%; min-height: 0; }}
  {_PAGE} #shep-models-research-rows {{ overflow: auto !important; min-height: 0; max-height: 100%; }}
  {_PAGE} #shep-mp-detail {{ overflow: hidden; display: flex !important; flex-direction: column; flex-wrap: nowrap !important; }}
  {_PAGE} #shep-mp-detail-bd {{ flex: 1 1 auto !important; overflow: auto !important; }}
  {_PAGE} {_LIST} > .shep-model-body {{ overflow: hidden; display: flex !important; }}
  /* The table's header sticks to the list region's top. */
  {_PAGE} #shep-models-research-rows .shep-tbl-wrap {{ overflow: visible; }}
  /* Words break only where they must (a column is never narrower than its
     longest word, so "contested" stays one word); the table scrolls in its
     region rather than squeezing a word. */
  {_PAGE} .shep-mtbl td {{ overflow-wrap: break-word; }}
  /* A short window: the list's controls take at most 60% of the panel (50%
     below 900 px) and scroll in their own region, so the table always keeps
     room of its own. */
  {_PAGE} {_LIST} {{ grid-template-rows: fit-content(60%) minmax(0, 1fr); }}
  {_PAGE} {_LIST} > .shep-model-head {{ min-height: 0; overflow: auto; }}
}}
@media (min-width: 1400px) and (min-height: 760px) and (max-height: 899px) {{
  {_PAGE} {_LIST} {{ grid-template-rows: fit-content(50%) minmax(0, 1fr); }}
}}
@media (max-width: 1399px), (max-height: 759px) {{
  {_PAGE} {_LIST} {{
    grid-template-columns: minmax(0, 1fr); grid-template-rows: auto;
    grid-template-areas: "head" "body" "foot" "confirm"; row-gap: 0 !important;
  }}
  {_PAGE} {_LIST}:has(.shep-mp-ro) {{ grid-template-columns: minmax(0, 1fr); }}
  {_PAGE} {_LIST}:has(.shep-mp-ro) > .shep-model-head {{ grid-row: auto; margin-bottom: 16px !important; }}
  {_PAGE} {_LIST} > .shep-model-body {{ margin-bottom: 16px !important; }}
  {_PAGE} {_LIST} > .shep-model-confirm-box {{ grid-area: confirm; margin-top: 16px !important; overflow: visible; }}
  {_PAGE} {_LIST} > .shep-model-head {{ grid-template-columns: minmax(0, 1fr); grid-template-rows: repeat(5, max-content); grid-template-areas: "lhd" "ws" "arch" "ctl" "count"; }}
}}
"""


__all__ = [
    "BY_SCORE",
    "D8_CONDITIONS",
    "GROUP_CONTESTED",
    "GROUP_NOT_COMPARED",
    "GROUP_NOT_TESTED",
    "GROUP_UNREAD",
    "INDEPENDENCE_NOTE",
    "LOAD_NOTE",
    "LOAD_NOTE_ORDERED",
    "METRIC_LABELS",
    "NEWEST",
    "PAGE_CSS",
    "PAGE_HELP",
    "READONLY_TITLE",
    "contest_states",
    "controls",
    "create_models_page",
    "detail_parts",
    "independence_html",
    "ledger_records",
    "left_header_html",
    "listing_key",
    "options_html",
    "picks_for",
    "rank",
    "read_page",
    "read_readonly",
    "read_versions",
    "readonly_parts",
    "record_view",
    "records_block_html",
    "shape",
    "usage_entries",
]
