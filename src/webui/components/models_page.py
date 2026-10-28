"""
Research › Evaluate › Models: the research preset of the shared model list (M5-2, UI-D).
========================================================================================
`docs/working/PLAN_TEST_RESULTS.md` §6, "M5's boundaries" (Model Management on
the shared model list, Trainer validation and each cohort's Mode C results kept
apart), D4 (ordering by test score), D5 (contested records), D8 (the list is an
operator control) and §4.9 question 1 (ordering per role, answered 2026-10-19).

**One list, one Load.** The page places the research preset of the shared list
(`ctx.create_model_list("research")`, `src/webui/components/model_list.py`): the
workspace select (the chosen workspace's dataset identity said again under it, by
the shared renderer, `workspace_line_html`), the table ("Fingerprints: model ·
dataset trained on", the Trainer's validation figure, the recorded tests' state),
"In use (Diagnosis)" on the served row, and the one gated Load with its
confirmation. This module never
lists a checkpoint, never reloads and never writes a usage history entry itself:
it shapes the list through the handle's ``view`` (a task filter, its rows and
groups, one extra column, the note beside Load) and ``ordering`` (so that a
Load from a list ordered by a cohort version's score is written to that
version's usage history by the shared Load, `model_list._record_usage`).

**Task and order (README §8: "Models with the family and a task filter").**
The Task select (All, Base, Disease, Gene, Patients-like-me, History) is the
shared list's one filter vocabulary (`model_list.TASK_FILTERS`); a file whose
record was refused or not read is listed under All only. Newest first under
All adds a Kind column (`renderers.record_kind_text`). By family draws the
families of the listing's family view (`model_list.family_views`, from the one
listing), each parent before its children, with a Lineage column (the kind,
the operation and the exact parent file, indented by depth; a task's base kept
as a context row; a parent that is not available said in amber, "▲ parent
unavailable: …"), and under All the history files and the records not
readable as two groups after them. By test score keeps its policy (Mode C
records only, below); task records are shown in the detail with their own
metrics (`RECORD_COLUMNS`), and ordering by them is the owner's open item.
Load is refused for a base, a history file or a record not readable before
anything is read (`model_list.load_refusal`), its reason beside the button.

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
  ordered" (the reports under the picked record's key disagree), "Contested
  status cannot be checked · not ordered" (whether they disagree cannot be read:
  the version's report store, a version the registry does not list, a key the
  page holds no state for), "Not compared on this version · not ordered"
  (records that cannot be compared: another scoring semantics, two ledgers, a
  record or ledger that cannot be read, a checkpoint that cannot be read) and
  "Not tested on this cohort version · newest first";
- on each ordered row its value, the record's registration number, its regime
  (device · deterministic or not) and "regime differs" (with its ⓘ) where it is
  not the one most ordered rows share.

**The detail** of the selected checkpoint: its full model fingerprint (Copy), what
it was trained on, what it is (kind, operation, parent, family, encoder and its
relation to the paper, README §5 axis 2: "matches the reference recipe" or the
fields that differ), the Trainer's validation figure in its own card (a selection
metric on val, never beside a Mode C number), Model Facts' warnings, and two
in-page segments: "Recorded test results" (every record its ledger holds, one
table per mode, as the API's record view shows a record: its mode's label, coverage with every
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
checkpoint**, with each record's contested state as that route answers it (the
one implementation, `record_states.contest_states`; never computed again here,
so this view and an API client cannot disagree). Its reasons name no server
path.

**Per browser session:** the page's data, its controls and the selection are
`gr.State` or component values; nothing here is module state. Nothing ticks: the
page is read when it is opened (`ctx.on_page_open`), when the workspace changes
and after every Load. A Load marks the page's data stale ("Reload with …"
clicked) and the page reads it again once the Load has finished, even when the
listing the shared refresh reads afterwards is unchanged (a refused reload, the
model in use reloaded): the usage history it wrote is then shown. The shared
refresh keeps a selection still listed in the same workspace (a row picked after
the Load stays picked, M5-3 review UI-3); one it did not keep is selected again
after a Load while its row is shown, so Close returns to its detail. One listener
(`_sync`) answers a new listing, a selection and the controls, so its answers
for one session never overlap.

**Narrow** (below 1400 × 760), choosing a row scrolls to its detail, which has a
"Back to the list" button (`SCROLL_JS`).

Module: src/webui/components/models_page.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gradio as gr

from src.api.services.record_states import CohortVersions, key_text

# Each mode's label and note, R10's and the relations' words: rendered, never reworded.
from src.evaluation.caveats import (
    R10_LABEL_TEXT,
    R10_STATE_TEXT,
    R10_UNVERIFIABLE_REASON,
    RECORD_ABSENCE,
    RELATION_DISPLAY_NAMES,
    TASK_NAMES,
    metric_text,
    mode_label,
    mode_note,
    served_scoring,
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
#: Each metric a list can be ordered by (`ordering.ORDERABLE_METRICS`, Mode C's),
#: as a recorded result names it (`caveats.metric_text`): always "Mode C", never
#: the Trainer's "Val MRR".
METRIC_LABELS: Dict[str, str] = {
    name: metric_text("C", name)
    for name in ("untruncated_mrr", "hits@1", "hits@5", "hits@10", "hits@20", "hits@50",
                 "hits@100", "mean_rank")
}
DEFAULT_METRIC = "untruncated_mrr"
#: The two metrics a record shows in their own columns (an MRR and a top-10
#: figure), by its mode; Mode D's are over the import-eligible cases (PLAN_B05
#: §5). A task's: its pooled MRR (README §2.6 E1) and, for genes and
#: patients-like-me, the macro recall@10 reported beside it (E2); the disease
#: task's pooled top-10 (one truth per case).
RECORD_COLUMNS: Dict[str, Tuple[str, str]] = {
    "C": ("untruncated_mrr", "hits@10"),
    "D": ("post_import_mrr", "post_import_hits@10"),
    "disease": ("pooled_mrr", "pooled_top10"),
    "gene": ("pooled_mrr", "macro_recall@10"),
    "patients_like_me": ("pooled_mrr", "macro_recall@10"),
}

NEWEST, BY_FAMILY, BY_SCORE = "newest", "family", "score"
ORDER_CHOICES = [("Newest first", NEWEST), ("By family", BY_FAMILY), ("By test score", BY_SCORE)]
#: The Task select's options: the shared list's filters a page offers, in the
#: list's own words (`model_list.FILTER_WORDS`); "all" first, short, so the
#: workspace select beside it keeps the room its value needs.
TASK_CHOICES = [(ml.FILTER_WORDS[name][0], name)
                for name in ("all", "base", *TASK_NAMES, "history")]
ALL_TASKS = "all"
RECORDS, USAGE = "records", "usage"

#: By family's groups and extra column; Newest first's column under All.
FAMILY_TITLE = "Family from run {run} · {count} file{s}"
GROUP_HISTORY = "History · listed, never run"
GROUP_REFUSED = "Record not readable"
LINEAGE_COLUMN = "Lineage"
KIND_COLUMN = "Kind"
FAMILY_NOT_READ = "the family view was not read with this list"
#: A model record's operation (`record.OPERATIONS`) as the lineage says it.
OPERATION_WORDS = {"new": "New run", "resume": "Resumed run", "extend": "Extended run",
                   "branch": "Branch", "finetune": "Finetune"}
#: The reference recipe as README §5 axis 2 has it read, never "identical to upstream".
REFERENCE_ANCHOR = ("The reference recipe is anchored on e95433a, with corrections "
                    "(Ref: README §2.4).")

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
GROUP_UNCHECKED = "Contested status cannot be checked · not ordered"
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
NO_LOAD_HERE = ("No validation figures here: they need the checkpoint files; Load is an "
                "operator control.")
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
        "Order",
        "How the list is ordered; the Task select chooses which rows.\n"
        "• Newest first: latest run first.\n"
        "• By family: each base, then the models finetuned from it.\n"
        "• By test score: one Mode C metric of the recorded results on one cohort version "
        "and one scoring semantics (D4); Mode D and task records are shown, never ordered. "
        "Offered only for a role in SHEPHERD_ORDERING_ROLES, never institutional_acceptance.\n"
        "One record per checkpoint by a fixed rule (its first registered deterministic record, "
        "else its first); a contested record is not ordered and never replaced. No model is "
        "ever chosen automatically by a test score.",
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

#: In the detail's header, shown only narrow (`SCROLL_JS`): back to the selected row.
BACK_BUTTON = ('<button type="button" class="shep-btn shep-mp-back">↑ Back to the list'
               '</button>')

#: Narrow (the layout rule's stacked sizes), the detail is below the whole list:
#: choosing a row (click, Enter or Space, as the shared list's own script takes
#: them) scrolls to the detail and moves the focus to its title; "Back to the
#: list" scrolls to the selected row and focuses it. It waits until the detail's
#: title names the chosen row (its ``data-key``, `detail_parts`; the title changes
#: more than once before the server's answer is drawn), and places the detail
#: again while the rest of it is drawn (the page grows under it), unless the
#: person has scrolled since. Document-level delegation, attached once per page
#: load (the page's HTML may mount again).
SCROLL_JS = r"""
if (!window.__shepModelsScroll) {
  window.__shepModelsScroll = true;
  var narrow = window.matchMedia('(max-width: 1399px), (max-height: 759px)');
  var pick = function (e) {
    var rows = document.getElementById('shep-models-research-rows');
    if (!narrow.matches || !rows || !e.target.closest || !rows.contains(e.target)) { return; }
    if (e.target.closest('button, a')) { return; }
    var row = e.target.closest('tr[data-key]');
    if (!row || row.getAttribute('aria-disabled') === 'true') { return; }
    if (e.type === 'keydown' && (e.target !== row || (e.key !== 'Enter' && e.key !== ' '))) { return; }
    var key = row.getAttribute('data-key'), detail = document.getElementById('shep-mp-detail');
    if (!detail) { return; }
    var done = false, placedAt = null;
    var title = function () {
      var h = detail.querySelector('#shep-mp-detail-hd h2[data-key]');
      return h && h.getAttribute('data-key') === key ? h : null;
    };
    var place = function () { detail.scrollIntoView({block: 'start'}); placedAt = window.scrollY; };
    var step = function () {
      var h = title();
      if (!h) { return; }
      if (!done) {
        done = true;
        place();
        h.focus({preventScroll: true});
        return;
      }
      /* The rest of the detail drawn after its title: place it again while the
         person has not scrolled since; give the focus back if it was lost. */
      if (window.scrollY === placedAt && detail.getBoundingClientRect().top > 1) { place(); }
      if (!document.activeElement || document.activeElement === document.body) {
        h.focus({preventScroll: true});
      }
    };
    var observer = new MutationObserver(step);
    observer.observe(detail, {childList: true, subtree: true, characterData: true,
                              attributes: true, attributeFilter: ['class', 'style', 'hidden']});
    setTimeout(function () { observer.disconnect(); }, 5000);
    step();   /* the row already shown: at once */
  };
  document.addEventListener('click', pick);
  document.addEventListener('keydown', pick);
  document.addEventListener('click', function (e) {
    var back = e.target.closest ? e.target.closest('#shep-mp-detail-hd button.shep-mp-back') : null;
    if (!back) { return; }
    var rows = document.getElementById('shep-models-research-rows');
    var row = rows ? (rows.querySelector('tr.shep-on') || rows) : null;
    if (!row) { return; }
    row.scrollIntoView({block: 'center'});
    if (row.tagName === 'TR') { row.focus({preventScroll: true}); }
  });
}
"""

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
    """A recorded value as shown: four decimals, two for a mean rank."""
    if not _number(value):
        return "not reported"
    return f"{value:.2f}" if str(name).endswith("mean_rank") else f"{value:.4f}"


def metric_label(name: Any, mode: str = "C") -> str:
    """A metric of a record of `mode` as named on this page (`caveats.metric_text`)."""
    return metric_text(mode, str(name))


def record_mode(view: Mapping[str, Any]) -> Optional[str]:
    """A record view's measurement mode (`RecordView.mode`), or None when the
    view carries none it can read (never taken for Mode C)."""
    mode = view.get("mode")
    return mode if isinstance(mode, str) and mode.strip() else None


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


def usage_entries(versions: Sequence[Mapping[str, Any]], resolver: CohortVersions,
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
    cohort versions (WP1's list) and the usage history entries that name a
    listed checkpoint; each listed checkpoint's ledger records and every record
    key's contested state come from the listing, which read them for its rows.
    Plain data for a `gr.State`."""
    data: Dict[str, Any] = {"listing_key": listing_key(listing), "readonly": False}
    try:
        data.update(read_versions(environ))
    except Exception:  # noqa: BLE001 - the list still shows; ordering says why it is off
        logger.exception("The cohort versions could not be listed")
        data.update({"versions": [], "problem": "the cohort versions could not be listed",
                     "roles": [], "roles_note": None})
    items = list(listing.get("items") or [])
    # The ledgers beside the listed checkpoints and their records' contested
    # states, as the shared listing read them once for its own rows
    # (`model_list.list_models`): never read a second time here.
    data["ledgers"] = dict(listing.get("ledgers") or {})
    data["contest"] = dict(listing.get("contest") or {})
    resolver = CohortVersions(data["versions"], environ)
    data["usage"] = usage_entries(data["versions"], resolver,
                                  (i.get("sha256") for i in items if ml.loadable(i)))
    return data


def _route_records() -> Dict[str, Any]:
    """The API's read-only route for the served model's recorded results
    (`get_loaded_model_evaluations`), called in this process on an event loop
    of its own: the ledger beside the served checkpoint, filtered by the served
    digest, opening no checkpoint, each record with its D5 state. The handler
    takes the served identity before its one ``await``, which reads the files
    in the threadpool, and runs to completion here."""
    from fastapi.encoders import jsonable_encoder

    from src.api.routes import evaluations

    return jsonable_encoder(asyncio.run(evaluations.get_loaded_model_evaluations()))


def read_readonly() -> Dict[str, Any]:
    """The read-only view's data: the status (for the served digest and the
    time it was read), the route's answer for the served model, and each of its
    records' D5 states **as the route answered them** (with no problem text: no
    server path reaches a read-only request), read with the service's own
    settings."""
    data: Dict[str, Any] = {"readonly": True, "listing_key": None}
    status, problem = read_status()
    data["status_at"] = r.clock_utc(datetime.now())
    data["status_problem"] = problem
    data["served_sha"] = served_checkpoint_sha(status) if status is not None else None
    try:
        answer = _route_records()
    except Exception:  # noqa: BLE001 - a state, never a traceback
        logger.exception("The recorded results of the model in use could not be read")
        answer = {"state": "unavailable"}
    data["answer"] = answer
    records = answer.get("records") if isinstance(answer.get("records"), list) else []
    # **Each record's D5 state as the route answered it** (`record_states.
    # contest_states`, the one implementation), never computed a second time
    # here, so this view and an API client cannot disagree (BACKLOG 36). A
    # record without one reads "cannot be checked" where it is shown
    # (`contest_html`), never uncontested. The route gives null for every value
    # that is not a finite number, so it is passed on as NaN and shown "not a
    # finite number", never "not reported": the report may hold NaN or text.
    contest: Dict[str, Dict[str, Any]] = {}
    for record in records:
        key = route_record_key(record, answer.get("checkpoint_sha256"))
        state = record.get("contest") if isinstance(record, Mapping) else None
        if key is None or not isinstance(state, Mapping):
            continue
        runs = [{**run, "values": {name: (math.nan if value is None else value)
                                   for name, value in (run.get("values") or {}).items()}}
                for run in state.get("runs") or [] if isinstance(run, Mapping)]
        contest[key_text(key)] = {**state, "runs": runs, "problems": []}
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
    # The record's own setting, which the route reads from the ledger (no
    # checkpoint opened), so read-only shows what an operator sees; the CUDA
    # classes carry it too, a CPU run's class does not (M5-3).
    flag = record.get("deterministic_algorithms")
    view["deterministic"] = (flag if type(flag) is bool else
                             True if klass == ENVIRONMENT_CLASSES[0] else
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
    """The record's case sets: a Mode C record's ranked cases of the source cases;
    a Mode D record's reached cases of the import-eligible ones (PLAN_B05 §4)."""
    counts = view.get("counts") if isinstance(view.get("counts"), Mapping) else {}
    excluded = counts.get("excluded") if isinstance(counts.get("excluded"), Mapping) else None
    if excluded is None:
        reasons = "not reported"
    else:
        reasons = " · ".join(f"{str(name).replace('_', ' ')} {count}"
                             for name, count in excluded.items()) or "none"
    mode = record_mode(view)
    if mode == "D":
        return (f'<div>{r.e(view.get("n_ranked"))} of {r.e(counts.get("ranked_eligible"))} '
                'import-eligible cases reached</div>'
                f'<div class="shep-sub">{r.e(counts.get("source_cases"))} source cases; '
                f'{r.e(counts.get("n_excluded"))} excluded at import: {r.e(reasons)}</div>')
    if mode == "C":
        return (f'<div>{r.e(view.get("n_ranked"))} of {r.e(counts.get("source_cases"))} source '
                f'cases ranked ({r.e(counts.get("ranked_eligible"))} eligible)</div>'
                f'<div class="shep-sub">{r.e(counts.get("n_excluded"))} excluded: '
                f'{r.e(reasons)}</div>')
    if mode in TASK_NAMES:
        # A task record's n_ranked counts the cases (query patients) with a
        # ranked positive; its coverage metric says the share (README §2.6 E5).
        return (f'<div>{r.e(view.get("n_ranked"))} of {r.e(counts.get("source_cases"))} source '
                'cases ranked by the model</div>'
                f'<div class="shep-sub">{r.e(counts.get("n_excluded"))} excluded at import: '
                f'{r.e(reasons)}</div>')
    # A mode this version does not describe: its counts, and no claim of what
    # its n_ranked counts.
    return (f'<div>{r.e(counts.get("source_cases"))} source cases; '
            f'{r.e(counts.get("n_excluded"))} excluded: {r.e(reasons)}</div>'
            f'<div class="shep-sub">{r.e(view.get("n_ranked"))} cases counted by the record; what '
            'they are is not described for this mode</div>')


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


def _contested_value(name: str, value: Any) -> str:
    """A disagreeing run's value as `contest_html` shows it: null is "not
    reported"; a value that is not a finite number (NaN, infinity, text, an
    object, an integer too large for a float: a hand edit) is "not a finite
    number", never formatted as one."""
    if value is None:
        return "not reported"
    try:
        finite = _number(value) and math.isfinite(value)
    except OverflowError:  # an integer too large for a float
        finite = False
    if not finite:
        return "not a finite number"
    return fmt_metric(name, value)


def contest_html(state: Optional[Mapping[str, Any]], *, ordered_out: bool = False,
                 mode: str = "C") -> str:
    """A record's D5 state beside it: contested (with what the runs disagree on,
    each metric named in the record's `mode`) or "cannot be checked"; nothing for
    an uncontested key."""
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
            per_run = [f"{_contested_value(name, run['values'][name])} (run "
                       f"{run_time(run['run']) or run['run']})"
                       for run in state.get("runs") or [] if name in (run.get("values") or {})]
            if per_run:
                values.append(f"{metric_label(name, mode)} " + " vs ".join(per_run))
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
                                                      and kind != "uncontested"),
                              mode=record_mode(view)))
    return "".join(parts)


def records_table_html(views: Sequence[Mapping[str, Any]], contest: Mapping[str, Any], *,
                       picks: Optional[Mapping[str, Mapping[str, Any]]] = None,
                       roles: Optional[Mapping[str, str]] = None) -> str:
    """The records, **one table per mode, never mixed**, each row a main row and a
    row of detail: the cohort version (label, version digest, synthetic, role),
    the mode's MRR and Hits@10 in their own columns, named in full in the header
    (`RECORD_COLUMNS`, `caveats.metric_text`: the mode and the case set), coverage,
    the record's state; then relations, the other metrics, R10 per scope,
    environment, dataset and registration. A Mode D record's served scoring is in
    its mode's label when every Mode D record shown shares it
    (`_uniform_semantics`), else on its own detail row."""
    return "".join(_mode_table(mode, group, contest, picks or {}, roles)
                   for mode, group in _by_mode(views))


def _mode_table(mode: Optional[str], views: Sequence[Mapping[str, Any]],
                contest: Mapping[str, Any], picks: Mapping[str, Mapping[str, Any]],
                roles: Optional[Mapping[str, str]]) -> str:
    """One mode's records as one table (`records_table_html`)."""
    columns = RECORD_COLUMNS.get(mode) if mode else None
    served_per_row = mode == "D" and _uniform_semantics(views) is None
    body = []
    for view in views:
        cohort = view.get("cohort") if isinstance(view.get("cohort"), Mapping) else {}
        metrics = view.get("metrics") if isinstance(view.get("metrics"), Mapping) else {}
        key = view.get("key")
        state = contest.get(key_text(key)) if key else None
        synthetic = (r.chip("synthetic", "amber") if view.get("dataset_origin") == "synthetic"
                     else "")
        role = cohort.get("role")
        role_note = (roles or {}).get(str(cohort.get("version_manifest_sha256")))
        cohort_cell = (f'<div class="shep-w600">{r.e(cohort.get("label"))}</div>'
                       f'<div class="shep-sub">(version <code>'
                       f'{r.e(r.short_hex(cohort.get("version_manifest_sha256")))}</code>) '
                       f'{synthetic}</div><div class="shep-sub">role '
                       f'{r.ident_html(role)}'
                       + (f" · {r.e(role_note)}" if role_note else "") + "</div>")
        mrr, hits = (fmt_metric(name, metrics.get(name)) for name in columns) if columns else (
            "not described", "not described")
        main = (f'<tr class="shep-mp-main"><td>{cohort_cell}</td>'
                f'<td class="shep-num">{r.e(mrr)}</td>'
                f'<td class="shep-num">{r.e(hits)}</td>'
                f'<td>{coverage_html(view)}</td>'
                f'<td>{_record_cell(view, state, picks.get(str(view.get("source"))))}</td></tr>')
        others = [f"{metric_label(name, mode)} {fmt_metric(name, value)}"
                  for name, value in sorted(metrics.items(), key=lambda kv: _metric_order(kv[0]))
                  if name not in (columns or ())]
        served = served_scoring(view.get("score_semantics")) if served_per_row else None
        if served:
            others.insert(0, f"served scoring {served}")
        # A Mode D record's case sets are its coverage cell and each metric's name.
        over = f", over {r.e(view.get('n_ranked'))} ranked cases" if mode == "C" else ""
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
                f'{" · " + r.e(" · ".join(others)) if others else ""}{over}</span>'
                f'{r10_html(view.get("r10"))}'
                f'<span class="shep-mp-ln">Environment: {r.e(environment)} · Dataset: '
                f'<b>{r.e(origin)}</b> · {r.e(registration)}</span></td></tr>')
        body.append(main + more)
    # Each table named for its mode, so a screen reader tells them apart.
    return ('<div class="shep-tbl-wrap shep-mp-recwrap"><table class="shep-tbl shep-mp-rec" '
            f'aria-label="{r.e(_TABLE_NAMES.get(mode, "Recorded test results, " + mode_label(mode)))}">'
            '<colgroup><col class="shep-mp-c1"><col '
            'class="shep-mp-c2"><col class="shep-mp-c3"><col class="shep-mp-c4"><col></colgroup>'
            '<thead><tr><th scope="col">Cohort version</th>'
            f'<th scope="col" class="shep-num">{_column_head(mode, 0)}</th>'
            f'<th scope="col" class="shep-num">{_column_head(mode, 1)}</th>'
            '<th scope="col">Coverage</th>'
            f'<th scope="col">Record</th></tr></thead><tbody>{"".join(body)}</tbody></table>'
            '</div>')


#: Each mode's records table as assistive technology names it.
_TABLE_NAMES = {"C": "Recorded test results, model ranking (Mode C)",
                "D": "Recorded test results, served ranking (Mode D)"}


def _by_mode(views: Sequence[Mapping[str, Any]]
             ) -> List[Tuple[Optional[str], List[Mapping[str, Any]]]]:
    """The records shown, grouped by mode: Mode C first, then Mode D, then any
    other in the order first seen; each group in the order given."""
    groups: Dict[Optional[str], List[Mapping[str, Any]]] = {}
    for view in views:
        groups.setdefault(record_mode(view), []).append(view)
    order = list(groups)
    return sorted(groups.items(), key=lambda kv: (kv[0] != "C", kv[0] != "D",
                                                 order.index(kv[0])))


def _uniform_semantics(views: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """The score semantics every Mode D record of `views` shares, when they share
    one served scoring mode and mixing weight (`caveats.served_scoring`); else
    None."""
    served = {served_scoring(view.get("score_semantics")) for view in views}
    if len(served) != 1 or None in served:
        return None
    return str(views[0].get("score_semantics"))


def _column_head(mode: Optional[str], index: int) -> str:
    """A metric column's header: the metric in full, its mode and case set
    ("Mode C MRR (untruncated)", "Mode D MRR (import-eligible)"); the bare metric
    for a mode this version does not describe, whose cells say so."""
    columns = RECORD_COLUMNS.get(mode) if mode else None
    if columns is None:
        return ("MRR", "Hits@10")[index]
    head, _sep, tail = metric_text(mode, columns[index]).partition(" (")
    return r.e(head) + (f' <span class="shep-nowrap">({r.e(tail)}</span>' if tail else "")


#: A task record's single figures, then its cut-off families, in the order a
#: record shows them (README §2.6: E1's pooled reading, E5's conservative one,
#: the macro and best-positive MRRs; pooled top-k, then E2's macro recall and
#: precision by k).
_TASK_FIGURES = ("pooled_mrr", "conservative_pooled_mrr", "macro_mrr", "best_positive_mrr")
_TASK_FAMILIES = ("pooled_top", "macro_recall@", "macro_precision@")


def _metric_order(name: str) -> Tuple[int, int, int, str]:
    """MRR, mean rank, Hits@K by K, the rest; Mode D's import-eligible family
    first, then coverage, then its reached family. A task record's after
    coverage: its MRRs, then each cut-off family by k."""
    if name in _TASK_FIGURES:
        return (3, _TASK_FIGURES.index(name), 0, name)
    for index, prefix in enumerate(_TASK_FAMILIES):
        k = name[len(prefix):]
        if name.startswith(prefix) and k.isdigit():
            return (4 + index, 0, int(k), name)
    family, base = 0, name
    if name.startswith("post_import_"):
        base = name[len("post_import_"):]
    elif name == "coverage":
        family = 1
    elif name.startswith("reached_"):
        family, base = 2, name[len("reached_"):]
    if base in ("untruncated_mrr", "mrr"):
        return (family, 0, 0, name)
    if base == "mean_rank":
        return (family, 1, 0, name)
    k = base[len("hits@"):]
    if base.startswith("hits@") and k.isdigit():
        return (family, 2, int(k), name)
    return (family, 3, 0, name)


def records_block_html(views: Sequence[Mapping[str, Any]], contest: Mapping[str, Any], *,
                       not_shown: int = 0,
                       picks: Optional[Mapping[str, Mapping[str, Any]]] = None,
                       roles: Optional[Mapping[str, str]] = None) -> str:
    """Each shown mode's label and note once (the pre-B-1 note beside Mode C
    records only), the dataset line, the table, the count of records not shown,
    and the R10 definitions behind a ⓘ."""
    parts = []
    if views:
        groups = _by_mode(views)
        modes = [mode for mode, _group in groups]
        origins = {v.get("dataset_origin") for v in views}
        if len(origins) == 1 and next(iter(origins)) in ORIGIN_MEANING:
            origin = next(iter(origins))
            word = (f'<b class="shep-amber-t">{r.e(origin)}</b>' if origin == "synthetic"
                    else f"<b>{r.e(origin)}</b>")
            parts.append(f'<p class="shep-mp-p">Dataset of every record below: {word} — '
                         f'{r.e(ORIGIN_MEANING[origin])}.</p>')
        # Each mode's label and note directly above its own table.
        for mode, group in groups:
            note = mode_note(mode)
            label = mode_label(mode, _uniform_semantics(group) if mode == "D" else None)
            parts.append(f'<p class="shep-mp-p"><b>{r.e(label)}.</b>'
                         + (f" {r.e(note)}" if note else "") + "</p>")
            parts.append(records_table_html(group, contest, picks=picks, roles=roles))
        over = {"C": "Mode C metrics are over the ranked cases",
                "D": "Mode D metrics over the import-eligible or the reached cases, as each "
                     "name says"}
        sets = "; ".join(over[m] for m in modes if m in over)
        figure = ("; the Trainer validation figure is not among them." if sets else
                  "The Trainer validation figure is not among these records' metrics.")
        parts.append(f'<p class="shep-note">R10 labels '
                     f'{help_button("models-r10", "R10 labels")} · {sets}{figure}</p>')
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
def offered_versions(data: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return [v for v in data.get("versions") or []
            if v.get("offers") and r.is_digest(v.get("sha256"))]


def scoring_groups(listing: Mapping[str, Any], data: Mapping[str, Any], version_sha: Any,
                   task: Optional[str]) -> List[Tuple[str, int]]:
    """The scoring semantics the shown checkpoints' Mode C records on this version
    are under, with how many checkpoints each, most first. Only Mode C's: the
    metrics a list is ordered by are Mode C's, and a Mode D or task record is
    shown, never ordered (PLAN_B05 §6; K6 decision 14)."""
    counts: Dict[str, set] = {}
    for item in ml.shown_items(listing, {"task": task}):
        entry = (data.get("ledgers") or {}).get(ml.item_key(item))
        for record in (entry or {}).get("records") or []:
            scoring = record.get("scoring")
            if not isinstance(scoring, Mapping) or scoring.get("mode") != "C":
                continue
            if record.get("cohort_digest") == version_sha:
                digest = record.get("scoring_semantics_digest")
                if isinstance(digest, str):
                    counts.setdefault(digest, set()).add(item.get("sha256"))
    return sorted(((d, len(s)) for d, s in counts.items()), key=lambda ds: (-ds[1], ds[0]))


def controls(listing: Mapping[str, Any], data: Mapping[str, Any], order: Any, version: Any,
             metric: Any, scoring: Any, task: Any) -> Dict[str, Any]:
    """The page's controls as they can apply now: a task filter of the Task
    select (else all), "By family" for an operator's list, "By test score"
    only when a version offers ordering (else newest first), an offered
    version (else the first offered), an orderable metric, and a scoring
    semantics the records are under (else the one most checkpoints share)."""
    from src.evaluation.ordering import ORDERABLE_METRICS

    chosen_task = task if task in {value for _label, value in TASK_CHOICES} else ALL_TASKS
    offered = offered_versions(data)
    labels = [v["label"] for v in offered]
    chosen_version = version if version in labels else (labels[0] if labels else None)
    readonly = bool(data.get("readonly"))
    mode = (BY_SCORE if order == BY_SCORE and offered and not readonly else
            BY_FAMILY if order == BY_FAMILY and not readonly else NEWEST)
    chosen_metric = metric if metric in ORDERABLE_METRICS else DEFAULT_METRIC
    entry = next((v for v in offered if v["label"] == chosen_version), None)
    groups = scoring_groups(listing, data, entry["sha256"], chosen_task) if entry else []
    digests = [d for d, _n in groups]
    chosen_scoring = scoring if scoring in digests else (digests[0] if digests else None)
    return {"task": chosen_task, "order": mode, "version": chosen_version,
            "metric": chosen_metric, "scoring": chosen_scoring, "scoring_groups": groups,
            "version_entry": entry}


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
    items = ml.shown_items(listing, {"task": ctl.get("task")})
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
            contested_state=lambda key: (contest.get(key_text(key)) or {}).get(
                "state", "cannot_be_checked"))
    except ValueError as exc:
        return {"offered": False, "why": str(exc), "rows": [], "keys": keys, "outside": outside,
                "items": items}
    return {"offered": ordering.offered, "why": ordering.why, "rows": list(ordering.rows),
            "keys": keys, "outside": outside, "items": items}


def regime_html(device: Any, deterministic: Any) -> str:
    """`regime_text` for a narrow cell: it wraps only after "·", never inside
    "non-deterministic" or between the device and its dot."""
    kind, _sep, determinism = regime_text(device, deterministic).partition(" · ")
    return (f'<span class="shep-nowrap">{r.e(kind)}&nbsp;·</span> '
            f'<span class="shep-nowrap">{r.e(determinism)}</span>')


def _regime_cell(row: Any) -> str:
    regime = row.chosen.get("regime") if isinstance(row.chosen.get("regime"), Mapping) else {}
    html = regime_html(regime.get("device"), regime.get("deterministic_algorithms"))
    # The mark's ⓘ says what a regime is and what "differs" means, for the
    # keyboard and touch too (the mark's `title` reaches neither).
    mark = ('<div class="shep-mp-markln"><span class="shep-mp-mark" title="The numerical regime '
            'differs from the one most ordered rows share">regime differs</span> '
            f'{help_button("models-regime", "Regime")}</div>' if row.regime_differs else "")
    return f'<div class="shep-sub shep-mp-regime">{html}</div>{mark}'


def _run_words(run: Any) -> str:
    """"run 2026-10-01 00:00" for a run named by its start time, else its name."""
    started = r.run_started(run)
    return f"run {started}" if started else f"run {run}"


def warn_mark_html(text: str) -> str:
    """A condition to watch (skill §6): amber, ▲ and its words, escaped."""
    return (f'<span class="shep-mp-warn"><span aria-hidden="true">▲</span> '
            f'{r.e(text)}</span>')


def parent_html(item: Mapping[str, Any], row: Optional[Mapping[str, Any]]) -> str:
    """Where the exact parent file a model's record names is (its 12-hex and
    run; listed here, in another listed workspace, or not available and why,
    from the family view's row); "none: a new run" for a file with no parent."""
    sha = item.get("parent_sha256")
    if not r.is_digest(sha):
        return "none: a new run"
    where = (f'<code class="shep-fpv">{r.e(r.short_hex(sha))}</code> of '
             f'{r.e(_run_words(item.get("parent_run")))}')
    state = (row or {}).get("parent_state")
    if state == "listed":
        return f"{where} · listed here"
    if state == "elsewhere":
        return f'{where} · in workspace <b>{r.e(row.get("parent_workspace"))}</b>'
    if state == "unavailable":
        return (f'{where}<div>'
                + warn_mark_html("parent unavailable: "
                                 + r.short_reason(row.get("parent_problem") or "not found", 160))
                + "</div>")
    return where


def lineage_cell_html(item: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    """A row's Lineage cell in By family: its kind, then its operation and the
    exact parent file (or that the parent is not available, in amber, or that
    it is only a context row), indented by its depth in the family."""
    depth = min(int(row.get("depth") or 0), 3)
    lines = [f'<div class="shep-w600">{r.e(r.record_kind_text(item))}</div>']
    operation = item.get("operation")
    sha = item.get("parent_sha256")
    if operation and r.is_digest(sha):
        where = (f' in workspace <b>{r.e(row.get("parent_workspace"))}</b>'
                 if row.get("parent_state") == "elsewhere" else "")
        lines.append(f'<div class="shep-sub">{r.e(OPERATION_WORDS.get(operation, operation))} of '
                     f'<code class="shep-fpv">{r.e(r.short_hex(sha))}</code>{where}</div>')
    elif operation:
        lines.append(f'<div class="shep-sub">{r.e(OPERATION_WORDS.get(operation, operation))}'
                     '</div>')
    if row.get("parent_state") == "unavailable":
        lines.append('<div class="shep-sub">' + warn_mark_html(
            "parent unavailable: " + r.short_reason(row.get("parent_problem") or "not found"))
            + "</div>")
    if row.get("context"):
        lines.append('<div class="shep-sub shep-mp-ctx"><span aria-hidden="true">○</span> '
                     'Context: parent of a shown file</div>')
    return f'<div class="shep-mp-lin shep-mp-d{depth}">{"".join(lines)}</div>'


def family_view(listing: Mapping[str, Any], task: str) -> Optional[Dict[str, Any]]:
    """By family's rows for the Task filter, from the listing's family view
    (`model_list.family_views`, one listing): ``keys`` (context rows
    included), ``groups`` (one per family, parent first, in the view's order;
    under All the history files and the records not readable after them;
    History is its group alone) and the Lineage ``cells``. None when the
    listing carries no family view."""
    views = listing.get("family")
    if not isinstance(views, Mapping) or "all" not in views:
        return None
    rows = views.get("all" if task in (ALL_TASKS, "history") else task) or []
    if task == "history":
        rows = [row for row in rows if row.get("group") == "history"]
    items = {ml.item_key(i): i for i in listing.get("items") or []}
    groups: List[Dict[str, Any]] = []
    cells: Dict[str, str] = {}
    current: Optional[Tuple[str, Any]] = None
    for row in rows:
        key = row.get("key")
        if key not in items:
            continue
        group = (row.get("group"), row.get("family_id") if row.get("group") == "family" else None)
        if group != current:
            current = group
            groups.append({"group": group, "keys": []})
        groups[-1]["keys"].append(key)
        cells[key] = lineage_cell_html(items[key], row)
    for group in groups:
        kind, family = group.pop("group")
        count = len(group["keys"])
        group["title"] = (FAMILY_TITLE.format(run=(r.run_started(family) or family), count=count,
                                              s="s" if count != 1 else "")
                          if kind == "family" else GROUP_HISTORY if kind == "history"
                          else GROUP_REFUSED)
    return {"keys": [key for group in groups for key in group["keys"]], "groups": groups,
            "cells": cells}


def family_row(listing: Mapping[str, Any], key: str) -> Optional[Mapping[str, Any]]:
    """A checkpoint's row in the listing's whole family view, or None."""
    views = listing.get("family")
    rows = views.get("all") if isinstance(views, Mapping) else None
    return next((row for row in rows or [] if row.get("key") == key), None)


def shown_view(listing: Mapping[str, Any], ctl: Mapping[str, Any]) -> Dict[str, Any]:
    """Which rows the controls show (the shared list's ``view`` without its
    order): By family's rows, context rows included, else the Task filter's."""
    if ctl.get("order") == BY_FAMILY:
        family = family_view(listing, ctl.get("task") or ALL_TASKS)
        if family is not None:
            return {"keys": family["keys"]}
    return {"task": ctl.get("task")}


def shape(listing: Mapping[str, Any], data: Mapping[str, Any], ctl: Mapping[str, Any]
          ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """The shared list's ``view`` and ``ordering`` for these controls.

    Newest first: the Task filter and the Load note, and under All a Kind
    column. By family: the families' groups and rows (`family_view`) with
    their Lineage column. By test score: the groups in D4's order (the ordered
    rows, each checkpoint's listed keys in the listing's order; then contested;
    then those whose contested status cannot be checked; then not compared;
    then not tested, newest first), the Mode C column with each row's value,
    record number and regime, and the note saying the usage history records
    the Load."""
    if data.get("readonly"):
        return {"note": ""}, None
    task = ctl.get("task") or ALL_TASKS
    newest: Dict[str, Any] = {"task": task, "note": LOAD_NOTE}
    if task == ALL_TASKS:
        newest.update(column=KIND_COLUMN, cells={
            ml.item_key(i): r.e(r.record_kind_text(i)) for i in listing.get("items") or []})
    if ctl.get("order") == BY_FAMILY:
        family = family_view(listing, task)
        if family is None:
            return {**newest, "refused": FAMILY_NOT_READ}, None
        return {"task": task, "keys": family["keys"], "groups": family["groups"],
                "column": LINEAGE_COLUMN, "cells": family["cells"], "note": LOAD_NOTE}, None
    ranked = rank(listing, data, ctl)
    if ranked is None:
        return newest, None
    if not ranked["offered"]:
        # Asked for, and refused by the ordering itself (a record naming
        # another role than the version's, say): newest first, and why.
        return {**newest, "refused": ranked["why"]}, None
    entry = ctl["version_entry"]
    metric = ctl["metric"]
    label = metric_label(metric)
    from src.evaluation.ordering import LOWER_IS_BETTER

    order_of = {ml.item_key(i): n for n, i in enumerate(ranked["items"])}
    ordered, contested, unchecked, compared, untested = [], [], [], [], []
    cells: Dict[str, str] = {}
    for row in ranked["rows"]:
        keys = sorted(ranked["keys"].get(row.checkpoint_sha256, []), key=order_of.get)
        if row.state == "ordered":
            sequence = row.chosen.get("registration", {}).get("sequence")
            rule = " · D4 rule" if row.others else ""
            cell = (f'<div class="shep-w600">{r.e(fmt_metric(metric, row.value))}</div>'
                    f'<div class="shep-sub">record #{r.e(sequence)}{r.e(rule)}</div>'
                    f'{_regime_cell(row)}')
            ordered.extend(keys)
        elif row.state == "contested":
            cell = f'<span class="shep-amber-t">{r.e(row.label)}</span>'
            contested.extend(keys)
        elif row.state == "contested_unchecked":
            # Never under "Contested" (that heading would claim the reports
            # disagree), never ordered: its own group says what is known.
            cell = f'<span class="shep-muted">{r.e(row.label)}</span>'
            unchecked.extend(keys)
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
                        (GROUP_UNCHECKED, sorted(unchecked, key=order_of.get)),
                        (GROUP_NOT_COMPARED, sorted(compared, key=order_of.get)),
                        (GROUP_NOT_TESTED, sorted(untested, key=order_of.get))):
        if keys:
            groups.append({"title": title, "keys": keys})
    # The column says the metric briefly (the group's title names it in full).
    short = "Mode C MRR" if metric == "untruncated_mrr" else label
    view = {"task": task, "groups": groups, "rest_title": GROUP_UNREAD,
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
                f'<span>{r.e(LIST_IS_C3)}</span><span>{r.e(LIST_C3_LINE)}</span></div>'
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
                   f'{r.e(version.get("label"))}</b> (role {r.ident_html(version.get("role"))}) '
                   f'— {r.e(why)}</span></div>')
    if refused:
        parts.append(f'<div class="shep-reason shep-amber-t">Not ordered, so the list stays '
                     f'newest first: {r.e(r.short_reason(refused, 240))}</div>')
    if not offered_versions(data):
        reason = (data.get("problem") or "no cohort version's role is in the server's "
                  "model-selection roles (SHEPHERD_ORDERING_ROLES)")
        parts.append(f'<div class="shep-reason">Ordering by test score is not offered: '
                     f'{r.e(r.short_reason(reason, 220))}</div>')
    if ctl.get("order") == BY_SCORE and offered_versions(data):
        parts.insert(0, f'<div class="shep-mp-roleon">Each version offered: {r.e(ROLE_ON)}.</div>')
    if ctl.get("order") == BY_SCORE and off:
        # Closed by default: a long registry's versions never push the order
        # controls and the table out of view (M5-3 review UI-1).
        count = len(off)
        parts.append(f'<details class="shep-mp-offd"><summary>{count} cohort version'
                     f'{"s" if count != 1 else ""} without ordering</summary>'
                     '<div class="shep-mp-offs">' + "".join(off) + "</div></details>")
    return "".join(parts)


def workspace_line_html(listing: Mapping[str, Any]) -> str:
    """Under the workspace select: the chosen workspace's dataset identity
    without its name (the shared renderer, `dataset_identity_html`: the dataset
    fingerprint with the digits the select's options use, "synthetic", the
    record's state), so a long workspace name that the select cuts never hides
    them. Nothing for a read-only request (no workspace is chosen there)."""
    if listing.get("refused"):
        return ""
    rows = listing.get("workspaces") or []
    row = next((w for w in rows if w.get("name") == listing.get("workspace")), None)
    if row is None:
        return ""
    digits = r.distinguishing_digits(w.get("dataset_fingerprint") for w in rows)
    return (f'<div class="shep-mp-wsline">'
            f'{r.dataset_identity_html(row, digits=digits, name=False)}</div>')


def independence_html(data: Mapping[str, Any]) -> str:
    """§4.9 question 1's note, wherever ordering is offered (and nowhere else)."""
    if data.get("readonly") or not offered_versions(data):
        return ""
    return (f'<div class="shep-notice shep-notice-amber shep-mp-indep" role="note"><b>'
            f'{r.e(INDEPENDENCE_NOTE)}</b> {r.e(INDEPENDENCE_DETAIL)}</div>')


def version_choices(data: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """The cohort version select's options, one line each: what offers ordering
    is said once beside the select (`options_html`), not in every option."""
    return [(f"{v['label']} (role {v['role']})", v["label"]) for v in offered_versions(data)]


def scoring_choices(groups: Sequence[Tuple[str, int]]) -> List[Tuple[str, str]]:
    return [(f"scoring semantics {d[:12]} · {n} checkpoint{'s' if n != 1 else ''}", d)
            for d, n in groups]


def _served_line(listing: Mapping[str, Any]) -> str:
    """What is served for diagnosis, as the listing's status read says it; "not
    read yet" before any read (the page as built, before its first listing),
    never "nothing is in use" (M5-3 review UI-4)."""
    if not listing.get("status_read"):
        return "in use for diagnosis: not read yet"
    served = listing.get("served_sha")
    if served:
        return (f'in use for diagnosis: model <code class="shep-fpv" title="{r.e(served)}">'
                f'{r.e(served[:12])}</code>')
    if listing.get("status_problem"):
        return "in use for diagnosis: not read"
    return "nothing is in use for diagnosis"


def identity_html(item: Mapping[str, Any], row: Optional[Mapping[str, Any]]) -> str:
    """What the selected checkpoint is, as one key-value list (README §5):
    its kind; for a model record its operation, the exact parent file and
    where it is (`parent_html`), its family, its encoder and its relation to
    the paper (axis 2: "matches the reference recipe" with what that recipe
    is, or the fields that differ). Model Facts' Task and Configuration rows
    say the kind and axis 1 elsewhere; neither is repeated here."""
    rows = [("Kind", r.e(r.record_kind_text(item)))]
    if item.get("record") == "model":
        operation = item.get("operation")
        differences = item.get("differences")
        if differences:
            relation = f'differs in: {r.e(", ".join(str(d) for d in differences))}'
        elif differences is not None:
            relation = (f'matches the reference recipe<div class="shep-sub">'
                        f'{r.e(REFERENCE_ANCHOR)}</div>')
        else:
            relation = '<span class="shep-muted">not read</span>'
        rows += [("Operation", r.e(OPERATION_WORDS.get(operation, operation or "not read"))),
                 ("Parent", parent_html(item, row)),
                 ("Family", r.e(_run_words(item.get("family_id")))),
                 ("Encoder", r.e(r.architecture_label(item.get("encoder")))),
                 ("Relation to the paper", relation)]
    return f'<div class="shep-kv shep-mp-ident">{r.kv_cells(rows)}</div>'


def detail_parts(listing: Mapping[str, Any], data: Mapping[str, Any], key: Any,
                 ctl: Mapping[str, Any]) -> Dict[str, Any]:
    """The detail of the selected checkpoint (operator), or the read-only view.
    Returns ``header``, ``top``, ``records``, ``usage`` (HTML) and ``seg``
    (the segment labels, or None when the segments are not shown)."""
    if data.get("readonly"):
        return readonly_parts(data)
    item = ml.find_item(listing, key)
    if item is None or item not in ml.shown_items(listing, shown_view(listing, ctl)):
        return {"header": ('<div class="shep-panel-hd"><h2>Details</h2>'
                           f'<span class="shep-panel-right">{_served_line(listing)}</span></div>'),
                "top": ('<p class="shep-muted shep-mp-p">Select a model to see its fingerprints, '
                        'its validation figure and its recorded test results.</p>'),
                "records": "", "usage": "", "seg": None}
    in_use = bool(listing.get("served_sha")) and item.get("sha256") == listing.get("served_sha")
    tag = r.chip("In use (Diagnosis)", "accent") if in_use else r.chip("not in use", "grey")
    # The title names the row (`SCROLL_JS` waits for it) and takes the focus.
    header = (f'<div class="shep-panel-hd"><h2 tabindex="-1" data-key="{r.e(ml.item_key(item))}">'
              f'{r.e(r.model_name_line(item))}</h2>{tag}<span class="shep-panel-right">'
              f'{_served_line(listing)}</span>{BACK_BUTTON}</div>')
    workspaces = listing.get("workspaces") or []
    if not ml.loadable(item):
        why = item.get("problem") or "its digest was not read"
        top = (f'<div class="shep-reason">{r.e(item.get("filename"))} · metadata not readable · '
               f'cannot be loaded: {r.e(why)}</div>')
        return {"header": header, "top": top, "records": "", "usage": "", "seg": None}
    sha = item["sha256"]
    run_ident = r.ident_html(f'{item.get("architecture")}/{item.get("run")}')
    ident = (f'<div class="shep-digest"><span class="shep-dl">Model fingerprint (checkpoint '
             f'sha256) · {run_ident} · {r.ident_html(item.get("filename"))}'
             f'{help_button("model-fingerprint", "Model fingerprint")}'
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
    top = (ident + f'<div class="shep-mp-trained">{trained} {" ".join(marks)}</div>'
           + identity_html(item, family_row(listing, ml.item_key(item))) + validation + warnings)
    ledger = (data.get("ledgers") or {}).get(ml.item_key(item))
    ranked = rank(listing, data, ctl)
    records, count, contested = _records_for_item(item, ledger, data, ranked, ctl)
    usage, n_usage = usage_html(item, data)
    seg_records = f"Recorded test results · {count}" + (
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
                    if (contest.get(key_text(v["key"])) or {}).get("state") == "contested")
    return (records_block_html(views, contest, not_shown=not_shown, picks=picks, roles=roles),
            len(views) + not_shown, contested)


def usage_entry_html(entry: Mapping[str, Any]) -> str:
    kind = entry.get("kind")
    when = _utc_text(entry.get("at"))
    version = r.e(entry.get("version"))
    if kind == "test_run":
        from src.evaluation.report_store import entry_mode

        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), Mapping) else {}
        # Named in the entry's own mode, which the figure's label carries.
        mode = entry_mode(entry)
        what = "Test run"
        columns = RECORD_COLUMNS.get(mode)
        if columns is None:
            figure = f"{mode_label(mode)}, {entry.get('n_ranked')} cases"
        else:
            value = f"{metric_label(columns[0], mode)} {fmt_metric(columns[0], metrics.get(columns[0]))}"
            figure = (f"{value} · {entry.get('n_ranked')} cases reached" if mode == "D"
                      else f"{value} over {entry.get('n_ranked')} ranked cases")
        result = (f"{figure} · {regime_text(entry.get('device'), entry.get('deterministic'))}"
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
    #: True from "Reload with …" until the page reads its data after that Load.
    stale_state = gr.State(False)
    #: The last checkpoint selected, ``{"workspace", "key"}``: selected again after a Load.
    last_state = gr.State(None)
    parts: Dict[str, Any] = {}

    def head() -> Dict[str, Any]:
        parts["lhd"] = plain_html(left_header_html({}, {}), elem_id="shep-mp-hd")
        parts["wsline"] = plain_html("", elem_id="shep-mp-wsline")
        parts["task"] = gr.Dropdown(label="Task", choices=TASK_CHOICES, value=ALL_TASKS,
                                    interactive=True, elem_id="shep-mp-task")
        with gr.Column(elem_id="shep-mp-controls", elem_classes=["shep-mp-controls"]):
            with gr.Row(elem_classes=["shep-mp-order-row"]):
                plain_html('<span class="shep-mp-lbl">Order</span>', elem_id="shep-mp-order-lbl")
                parts["order"] = gr.Radio(choices=ORDER_CHOICES, value=NEWEST, label="Order",
                                          show_label=False, interactive=False,
                                          elem_id="shep-mp-order", elem_classes=["shep-seg"],
                                          container=False)
                plain_html(help_button("models-ordering", "Order"), elem_id="shep-mp-order-i")
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
                    parts["version"] = gr.Dropdown(label="On cohort version",
                                                   show_label=False, container=False,
                                                   choices=[], value=None, interactive=True,
                                                   elem_id="shep-mp-version")
                parts["scoring"] = gr.Dropdown(label="Scoring semantics", choices=[], value=None,
                                               interactive=True, visible=False,
                                               elem_id="shep-mp-scoring")
            parts["options"] = plain_html("", elem_id="shep-mp-options")
        return parts

    def foot() -> Any:
        with gr.Column(elem_id="shep-mp-detail", elem_classes=["shep-mp-detail"]) as column:
            parts["detail_hd"] = plain_html(detail_parts({}, {}, None, {})["header"],
                                            elem_id="shep-mp-detail-hd", js_on_load=SCROLL_JS)
            with gr.Column(elem_id="shep-mp-detail-bd", elem_classes=["shep-mp-detail-bd"]):
                parts["detail_top"] = plain_html(detail_parts({}, {}, None, {})["top"],
                                                 elem_id="shep-mp-detail-top")
                parts["seg"] = gr.Radio(choices=[("Recorded test results", RECORDS),
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
                      parts["task"]]
    #: By element id: once the page has updated a control, Gradio keeps a new
    #: instance of it in the session (another ``_id``), which an event names.
    control_ids = {c.elem_id for c in control_inputs}
    page_outputs = [data_state, handle.view, handle.ordering, parts["task"], parts["order"],
                    parts["opts"], parts["version"], parts["scoring"], parts["options"],
                    parts["independence"], parts["lhd"], parts["wsline"], parts["detail_hd"],
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
        return (
            data if new_data else gr.update(),
            view, ordering,
            gr.update(value=ctl["task"]),
            # Always offered to an operator: By test score asked where no
            # version offers it falls back to newest first, and says why.
            gr.update(value=ctl["order"], interactive=not data.get("readonly")),
            gr.update(visible=ctl["order"] == BY_SCORE),
            gr.update(choices=version_choices(data), value=ctl["version"]),
            gr.update(choices=scoring_choices(ctl["scoring_groups"]), value=ctl["scoring"],
                      visible=len(ctl["scoring_groups"]) > 1),
            options_html(data, ctl, view.get("refused")),
            independence_html(data),
            left_header_html(listing, data, condition),
            workspace_line_html(listing),
            *detail_updates(listing, data, selected, ctl, seg_value),
        )

    def read_for(listing) -> Dict[str, Any]:
        if listing.get("refused"):
            return read_readonly()
        return read_page(listing)

    shared_refresh = handle.refresh["fn"]

    def _open(name, older, current_view, previous, key, order, version, metric, scoring, task,
              seg_value, phase, request: gr.Request):
        shared = shared_refresh(name, older, current_view, previous, key, request)
        listing = shared[0]
        data = read_for(listing)
        condition = operator_access(request).condition
        # Read now; a Load still running reads again once it has finished.
        stale = False if phase != "loading" else gr.update()
        return (*shared, stale, *outputs_for(listing, data, (order, version, metric, scoring, task),
                                             None, seg_value, condition, new_data=True))

    ctx.on_page_open("models", {
        "fn": _open,
        "inputs": [*handle.refresh["inputs"], *control_inputs, parts["seg"], handle.phase],
        "outputs": [*handle.refresh["outputs"], stale_state, *page_outputs],
        "show_progress": "minimal"})

    def _load_started():
        return True

    # Beside the shared Load's own chain (its reload, then its refresh): the
    # page's data is stale from here until it is read after the Load.
    handle.reload_button.click(_load_started, inputs=None, outputs=[stale_state], queue=False,
                               show_progress="hidden")

    def _sync(listing, data, stale, last, phase, order, version, metric, scoring, task, selected,
              seg_value, request: gr.Request, evt: gr.EventData):
        """A new listing (a workspace chosen, the refresh after a Load), a
        selection, or a control changed. The page reads its own data again for
        a listing it has not read, and once after a Load has finished (the
        listing may be unchanged); after a Load it selects the last checkpoint
        again while its row is shown, when the shared refresh did not keep it."""
        ctl_values = (order, version, metric, scoring, task)
        settled = phase != "loading"
        after_load = bool(stale) and settled
        fresh = (not isinstance(data, Mapping) or bool(listing.get("refused"))
                 or data.get("listing_key") != listing_key(listing) or after_load)
        if fresh:
            data = read_for(listing)
        key, again = selected, None
        if key is None and after_load and isinstance(last, Mapping) \
                and last.get("workspace") == listing.get("workspace"):
            item = ml.find_item(listing, last.get("key"))
            shown = shown_view(listing, controls(listing, data, *ctl_values))
            if item is not None and item in ml.shown_items(listing, shown):
                key = again = last["key"]
        states = (False if fresh and settled else gr.update(),
                  ({"workspace": listing.get("workspace"), "key": key} if key is not None
                   else gr.update()),
                  again if again is not None else gr.update())
        by_control = getattr(getattr(evt, "target", None), "elem_id", None) in control_ids
        if not (fresh or again is not None or by_control):
            ctl = controls(listing, data, *ctl_values)
            return (*states, *(gr.update() for _ in page_outputs[:-5]),
                    *detail_updates(listing, data, key, ctl, seg_value))
        condition = operator_access(request).condition
        out = list(outputs_for(listing, data, ctl_values, key, seg_value, condition,
                               new_data=fresh))
        if again is not None:
            # A new view value, so the shared list draws its rows again with
            # the selection (and enables Load); the token means nothing else.
            out[1] = {**out[1], "redraw": time.monotonic_ns()}
        return (*states, *out)

    gr.on([handle.listing.change, handle.selected.change, *(c.input for c in control_inputs)],
          _sync, inputs=[handle.listing, data_state, stale_state, last_state, handle.phase,
                         *control_inputs, handle.selected, parts["seg"]],
          outputs=[stale_state, last_state, handle.selected, *page_outputs],
          show_progress="hidden", trigger_mode="always_last")

    def _segment(value):
        return (gr.update(visible=value != USAGE), gr.update(visible=value == USAGE))

    parts["seg"].input(_segment, inputs=[parts["seg"]],
                       outputs=[parts["records"], parts["usage"]], queue=False,
                       show_progress="hidden")


_PAGE = "#" + page_elem_id("models")
_LIST = "#shep-models-research"

#: This page's style sheet (the page contract: listed in `src/webui/app.py`'s
#: ``PAGE_CSS``). The shared list's column is laid out as the mockup draws the
#: page: the list panel (its header, the workspace and task selects, the
#: order controls, the count and the table) on the left; the selected model's
#: detail with the Load footer on the right, the confirmation drawn over them
#: while it is open. The research preset's Model Facts are not shown here: the
#: detail shows the same identity and warnings with the recorded results.
PAGE_CSS = f"""
{_PAGE} .shep-models {{ gap: 0 !important; }}
{_PAGE} {_LIST} {{
  display: grid !important; gap: 0 16px !important; width: 100%;
  grid-template-columns: minmax(0, 46fr) minmax(0, 54fr);
  grid-template-rows: minmax(0, 1fr);
  grid-template-areas: "main foot";
  padding: 0 !important; border: 0 !important; background: none !important;
}}
/* The list panel: its header (the workspace and task selects, the
   order controls, the count) and the table, in one column (M5-3 review UI-1:
   a header capped in its own scroll region cut off the independence note and
   hid the order controls under a second scrollbar). */
{_PAGE} {_LIST} > .shep-model-main {{
  grid-area: main; min-height: 0; min-width: 0; display: flex !important;
  flex-direction: column; flex-wrap: nowrap !important; gap: 0 !important;
  background: var(--shep-panel); border: 1px solid var(--shep-border); border-radius: 10px;
  padding: 0 !important; margin: 0 !important;
}}
{_PAGE} {_LIST} > .shep-model-main > .shep-model-head {{
  flex: none; display: grid !important; gap: 10px 10px !important; align-items: start;
  align-content: start;
  /* The workspace select takes the row but what the task select needs for its
     longest option, "Patients-like-me" (the workspace value carries the name,
     the fingerprint and "synthetic"; the identity line under it says them again). */
  grid-template-columns: minmax(0, 1fr) 170px;
  grid-template-rows: repeat(5, max-content);
  grid-template-areas: "lhd lhd" "ws task" "wsline wsline" "ctl ctl" "count count";
  background: none; border: 0; padding: 0 16px 8px !important; margin: 0 !important;
}}
{_PAGE} {_LIST} > .shep-model-main > .shep-model-head > * {{ min-width: 0 !important; }}
{_PAGE} #shep-mp-hd {{ grid-area: lhd; margin: 0 -16px; width: calc(100% + 32px) !important; max-width: none; }}
{_PAGE} {_LIST} .shep-model-head > .form:has(#shep-models-research-workspace) {{ grid-area: ws; }}
{_PAGE} {_LIST} .shep-model-head > .form:has(#shep-mp-task) {{ grid-area: task; }}
{_PAGE} #shep-mp-wsline {{ grid-area: wsline; margin-top: -4px; }}
/* A value too long for its select ends in "…" before the chevron, never under it (the
   titled ellipsis is the shell's rule for every select). */
{_PAGE} #shep-models-research-workspace input, {_PAGE} #shep-mp-task input,
{_PAGE} #shep-mp-version input {{
  padding-right: 24px !important;
}}
{_PAGE} .shep-mp-wsline {{ display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: center; font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} #shep-mp-controls {{ grid-area: ctl; gap: 8px !important; }}
{_PAGE} #shep-models-research-count {{ grid-area: count; font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} {_LIST} > .shep-model-main > .shep-model-body {{
  min-height: 0; background: none; border: 0; border-radius: 0 0 10px 10px;
  padding: 0 16px 12px !important; margin: 0 !important; flex-wrap: nowrap !important;
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
{_PAGE} .shep-mp-opts-row > #shep-mp-version {{ flex: 1 1 220px !important; min-width: 200px !important; max-width: 360px; }}
{_PAGE} .shep-mp-roleon {{ font-size: 14px; color: var(--shep-text-2); padding: 0 10px; }}
{_PAGE} .shep-mp-offd {{ font-size: 14px; color: var(--shep-text-2); padding: 0 10px; }}
{_PAGE} .shep-mp-offd > summary {{ cursor: pointer; }}
{_PAGE} .shep-mp-offd > .shep-mp-offs {{ padding: 6px 0 0; }}
{_PAGE} .shep-mp-offs {{ display: flex; flex-direction: column; gap: 4px; padding: 0 10px; }}
{_PAGE} .shep-mp-off {{ display: flex; gap: 8px; align-items: flex-start; font-size: 14px; color: var(--shep-muted); }}
{_PAGE} .shep-mp-radio {{ flex: none; width: 14px; height: 14px; margin-top: 3px; border-radius: 50%;
  border: 1px solid var(--shep-grey-200); background: var(--shep-grey-50); }}
{_PAGE} .shep-mp-indep {{ font-size: 14px; }}
{_PAGE} .shep-mp-markln {{ margin-top: 2px; }}
{_PAGE} .shep-mp-markln button.shep-i {{ vertical-align: middle; }}
{_PAGE} .shep-mp-back {{ display: none; }}
{_PAGE} .shep-mp-mark {{ display: inline-block; padding: 0 6px; border-radius: 6px;
  background: var(--shep-amber-50); border: 1px solid var(--shep-amber-300); color: var(--shep-amber-800);
  font-size: 14px; font-weight: 600; white-space: nowrap; }}
{_PAGE} .shep-mp-p {{ margin: 0; font-size: 15px; color: var(--shep-text-2); line-height: 1.45; }}
{_PAGE} .shep-mp-trained {{ display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: center; font-size: 15px; }}
{_PAGE} .shep-mp-val {{ padding: 10px 12px; }}
{_PAGE} .shep-mp-valbd {{ display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: center; font-size: 15px; }}
{_PAGE} .shep-mp-recwrap {{ margin: 0 !important; }}
{_PAGE} .shep-mp-recwrap + .shep-mp-p {{ margin-top: 16px; }}
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
/* By family's Lineage cell: a child sits one step in from its parent; a
   parent that is not available is a condition to watch (amber, ▲, its words),
   a context row a neutral state (grey, ○, its words): skill §6. */
{_PAGE} .shep-mp-lin.shep-mp-d1 {{ padding-left: 12px; }}
{_PAGE} .shep-mp-lin.shep-mp-d2 {{ padding-left: 24px; }}
{_PAGE} .shep-mp-lin.shep-mp-d3 {{ padding-left: 36px; }}
{_PAGE} .shep-mp-warn {{ color: var(--shep-status-warn); }}
{_PAGE} .shep-mp-ctx {{ color: var(--shep-status-neutral); }}
{_PAGE} .shep-mp-ident {{ font-size: 15px; }}
/* Read-only: no list, no workspace, no order; the reason is the header's. */
{_PAGE} {_LIST}:has(.shep-mp-ro) .shep-model-head > .form,
{_PAGE} {_LIST}:has(.shep-mp-ro) #shep-mp-wsline,
{_PAGE} {_LIST}:has(.shep-mp-ro) #shep-mp-controls,
{_PAGE} {_LIST}:has(.shep-mp-ro) #shep-models-research-count,
{_PAGE} {_LIST}:has(.shep-mp-ro) > .shep-model-main > .shep-model-body {{ display: none !important; }}
{_PAGE} {_LIST}:has(.shep-mp-ro) {{ grid-template-columns: minmax(0, 30fr) minmax(0, 70fr); }}
@media (min-width: 1400px) and (min-height: 760px) {{
  {_PAGE} > .column {{ height: 100%; }}
  {_PAGE} .shep-models {{ height: 100%; min-height: 0; }}
  {_PAGE} {_LIST} {{ height: 100%; min-height: 0; }}
  {_PAGE} {_LIST} > .shep-model-main {{ height: 100%; overflow-y: auto; }}
  {_PAGE} #shep-mp-detail {{ overflow: hidden; display: flex !important; flex-direction: column; flex-wrap: nowrap !important; }}
  {_PAGE} #shep-mp-detail-bd {{ flex: 1 1 auto !important; overflow: auto !important; }}
  /* The table's header sticks to the top of the region the rows scroll in. */
  {_PAGE} #shep-models-research-rows .shep-tbl-wrap {{ overflow: visible; }}
  /* Words break only where they must (a column is never narrower than its
     longest word, so "contested" stays one word); the table scrolls in its
     region rather than squeezing a word. */
  {_PAGE} .shep-mtbl td {{ overflow-wrap: break-word; }}
  /* The table fits its frame at 1440 x 900 with a classic 15 px scrollbar beside it (M5-3:
     at 10 px a side its columns' longest words, "run 2026-10-21 18:03", a 12-hex digest,
     "Validation", "non-deterministic", "contested", made it 602 px in a 589 px frame). */
  {_PAGE} .shep-mtbl th, {_PAGE} .shep-mtbl td {{ padding-left: 7px; padding-right: 7px; }}
  /* Below 1000 px the header and the table do not both fit: the list panel
     scrolls as one region (the header first, then the rows under the table's
     sticky header), never a header region above a table region. */
  /* The confirmation: its table scrolls inside, its buttons stay in view
     under it however short the window. */
  {_PAGE} {_LIST} > .shep-model-confirm-box {{
    display: flex !important; flex-direction: column; flex-wrap: nowrap !important; overflow: hidden;
  }}
  {_PAGE} {_LIST} > .shep-model-confirm-box > #shep-models-research-confirm {{ flex: 1 1 auto; min-height: 0; overflow: auto !important; }}
  {_PAGE} {_LIST} > .shep-model-confirm-box > .row {{ flex: none; }}
  /* The panel's title stays at its top however far the panel scrolls (M5-3:
     at 1440 x 900 it scrolled away with the controls and the rows). The header
     row's items are laid out in the panel's own grid (the row itself draws no
     box), so the title sticks to the panel, not to the header row. */
  {_PAGE} {_LIST} > .shep-model-main {{
    display: grid !important; align-items: start; align-content: start; gap: 10px 10px !important;
    grid-template-columns: minmax(0, 1fr) 170px;
    grid-template-rows: repeat(5, max-content) auto;
    grid-template-areas: "lhd lhd" "ws task" "wsline wsline" "ctl ctl" "count count" "body body";
    padding: 0 16px !important;
  }}
  {_PAGE} {_LIST} > .shep-model-main > .shep-model-head {{ display: contents !important; }}
  {_PAGE} {_LIST} > .shep-model-main > .shep-model-body {{ grid-area: body; padding: 0 0 12px !important; }}
  /* Above the field labels it passes over (Gradio draws them at z-index 40). */
  {_PAGE} #shep-mp-hd {{ position: sticky; top: 0; z-index: 45; background: var(--shep-panel); border-radius: 10px 10px 0 0; }}
  {_PAGE} {_LIST}:has(.shep-mp-ro) > .shep-model-main {{ row-gap: 0 !important; }}
}}
@media (min-width: 1400px) and (min-height: 760px) and (max-height: 999px) {{
  /* The panel scrolls as one region: the table's header sticks under the title. */
  {_PAGE} #shep-models-research-rows .shep-tbl th {{ top: 48px; }}
}}
@media (min-width: 1400px) and (min-height: 1000px) {{
  /* Room for both: the header stays whole and the rows scroll in the region
     under it (at least 240 px of it; a taller header scrolls the panel). */
  {_PAGE} {_LIST} > .shep-model-main {{ grid-template-rows: repeat(5, max-content) minmax(240px, 1fr); }}
  {_PAGE} {_LIST} > .shep-model-main > .shep-model-body {{
    align-self: stretch; min-height: 0; overflow: hidden; display: flex !important;
  }}
  {_PAGE} #shep-models-research-rows {{ overflow: auto !important; min-height: 0; max-height: 100%; }}
}}
@media (max-width: 1399px), (max-height: 759px) {{
  {_PAGE} {_LIST} {{
    grid-template-columns: minmax(0, 1fr); grid-template-rows: auto;
    grid-template-areas: "main" "foot" "confirm"; row-gap: 0 !important;
  }}
  {_PAGE} {_LIST}:has(.shep-mp-ro) {{ grid-template-columns: minmax(0, 1fr); }}
  {_PAGE} {_LIST} > .shep-model-main {{ margin-bottom: 16px !important; }}
  {_PAGE} {_LIST} > .shep-model-confirm-box {{ grid-area: confirm; margin-top: 16px !important; overflow: visible; }}
  {_PAGE} {_LIST} > .shep-model-main > .shep-model-head {{ grid-template-columns: minmax(0, 1fr); grid-template-rows: repeat(6, max-content); grid-template-areas: "lhd" "ws" "wsline" "task" "ctl" "count"; }}
  {_PAGE} .shep-mp-back {{ display: inline-flex; align-items: center; margin-left: auto; }}
}}
"""


__all__ = [
    "ALL_TASKS",
    "BY_FAMILY",
    "BY_SCORE",
    "D8_CONDITIONS",
    "FAMILY_TITLE",
    "GROUP_CONTESTED",
    "GROUP_HISTORY",
    "GROUP_NOT_COMPARED",
    "GROUP_NOT_TESTED",
    "GROUP_REFUSED",
    "GROUP_UNCHECKED",
    "GROUP_UNREAD",
    "INDEPENDENCE_NOTE",
    "KIND_COLUMN",
    "LINEAGE_COLUMN",
    "LOAD_NOTE",
    "LOAD_NOTE_ORDERED",
    "METRIC_LABELS",
    "NEWEST",
    "ORDER_CHOICES",
    "PAGE_CSS",
    "PAGE_HELP",
    "READONLY_TITLE",
    "RECORD_COLUMNS",
    "TASK_CHOICES",
    "controls",
    "create_models_page",
    "detail_parts",
    "family_view",
    "identity_html",
    "independence_html",
    "left_header_html",
    "lineage_cell_html",
    "listing_key",
    "options_html",
    "parent_html",
    "picks_for",
    "rank",
    "read_page",
    "read_readonly",
    "read_versions",
    "readonly_parts",
    "record_view",
    "records_block_html",
    "shape",
    "shown_view",
    "usage_entries",
]
