"""
Research › Evaluate › Models: the research preset of the shared model list (M5-2, UI-D).
========================================================================================
`docs/working/PLAN_TEST_RESULTS.md` §6, "M5's boundaries" (Model Management on
the shared model list, Trainer validation and each version's recorded results
kept apart; a result is named by its version's kind and role, never "test" for
another role), D5 (every run's record, each with its run; runs under one condition
group that differ are described) and D8 (the list is an operator control). D4's
ordering by test score retired in K8-6d, with its role list, its Mode C rule and
its one record per checkpoint: no order ranks a model by a test score.

**One list, one Load.** The page places the research preset of the shared list
(`ctx.create_model_list("research")`, `src/webui/components/model_list.py`): the
workspace select (its options are the names; the chosen workspace's dataset
identity is said once, under it, by the shared renderer, `workspace_line_html`),
the table (each row its run by the one run words and how far it trained,
its model fingerprint's 12 hex digits only, the dataset only as an exception
mark where it is not the workspace's current one, `renderers.dataset_exception`,
the Trainer's validation figure, the recorded tests' state),
"In use (Diagnosis)" on the served row, and the one gated Load with its
confirmation. This module never
lists a checkpoint, never reloads and never writes a usage history entry itself:
it shapes the list through the handle's ``view`` (a task filter, its rows and
groups, one extra column, the note beside Load).

**Task and order (README §8: "Models with the family and a task filter").**
The Task select (All, Base, Disease, Gene, Patients-like-me) is the
shared list's one filter vocabulary (`model_list.TASK_FILTERS`); a file whose
record was refused or not read is listed under All only. Every row says what
its file is (the shared list's first line, `renderers.model_kind_text`: the
frozen-backbone control reads "Disease · frozen graph backbone"), under every
order. The Order switch offers Newest first and By family. By
family draws the families of the listing's family view
(`model_list.family_views`, from the one listing), each parent before its
children, with a Lineage column (the operation and the exact parent file,
indented by depth; a task's base kept as a context row; a parent that is not
available said in amber, "▲ parent unavailable: …"), and under All the
records not read and the records not readable (a file without a model spec
among them) as groups after them. Every record is shown in the detail with its own metrics, as
recorded (`RECORD_COLUMNS` names the two shown in columns for a measurement this
build describes).
Load is refused for a base or a record not readable before
anything is read (`model_list.load_refusal`), its reason beside the button.

**The detail** of the selected checkpoint, one summary list (`identity_html`):
its model fingerprint (12 hex, Copy), its dataset by its relation to the
workspace's ("the selected workspace's current dataset", or which one it is),
what it is (kind, operation, parent, family, encoder and its relation to the
paper, README §5 axis 2: "matches the reference recipe" or the fields that
differ), the Trainer's validation figure as a row (a selection metric on val,
never beside a test result, never a box of its own), its run directory, file
and whole fingerprint one "Details ›" away; Model Facts' warnings by the
shared mark, and three
in-page segments: "Recorded results" (every record its ledger holds, one
table per mode in the order first shown, as the API's record view shows a record
by **one display for every mode** (R2, K8-6e.2): each relation a row of its own,
its name visible whole (`renderers.relation_rows`); its mode's label, a described
measurement's two columns in words with every metric as recorded one click away,
any other mode's metrics as recorded in one column, its score semantics as
recorded, coverage with every
exclusion, relations, R10 as states with counts only where the state has one,
so *unverifiable* is a word and never 0, environment, registration and its
run, or "not recorded" for a record merged before each run had its own record;
a possible overlap between such a record and a named run; "runs under these
conditions differ" (neutral; amber with "— investigate" only for a group that
records deterministic algorithms on) or "comparison incomplete" per record, never
agreeing by default, and never hiding or ordering out a record; a record for
these weights that cannot be read as a result is listed apart, by its index,
registration and report with its reason, and never counted as one, K8-6e.1)
"Usage history" (the entries of every listed cohort version's usage history
that name this checkpoint, newest first, each shown as it was recorded), and
"Results" (research comparison Stage 8b, `results_segment`: every result of
this model's family or this workspace from the one results service, one table
per protocol newest first, up to five compared side by side with differences
only, and the chosen or shown rows exported as CSV or JSON; its own handlers
each ask D8 first).

**Read-only (D8).** Listing opens checkpoint files (SPEC_4 C3), so a request D8
refuses gets no list: the page says why (the three conditions of an operator
request) and shows "Recorded results of the model in use", read by the
API's own read-only route function (`get_loaded_model_evaluations`): the ledger
beside the served checkpoint, filtered by the served digest, **opening no
checkpoint**, with each record's run and group description as that route
answers them (the one implementation, `record_states.contest_states`; never
computed again here, so this view and an API client cannot disagree). Its
reasons name no server path.

**Per browser session:** the page's data, its controls and the selection are
`gr.State` or component values; nothing here is module state. Nothing ticks: the
page is read when it is opened (`ctx.on_page_open`), when the workspace changes
and after every Load. A Load marks the page's data stale ("Reload with …"
clicked) and the page reads it again once the Load has finished, even when the
listing the shared refresh reads afterwards is unchanged (a refused reload, the
model in use reloaded): what changed meanwhile is then shown. The shared
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
    RECORD_ABSENCE,
    TASK_NAMES,
    metric_text,
    mode_label,
    mode_note,
    recorded_metric_lines,
    recorded_value_text,
    version_words,
)
from src.utils.run_environment import native_identity_text
from src.webui.components import model_list as ml
from src.webui.components import renderers as r
from src.webui.components.chips import D8_SHORT, LOCK
from src.webui.components.help import copy_button, help_button
from src.webui.components.served_status import read_status, served_checkpoint_sha
from src.webui.operator_access import operator_access
from src.webui.shell import page_elem_id, plain_html
from src.webui.style import sticky_room_js

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Words
# ---------------------------------------------------------------------------
#: The two metrics a record of a measurement this build describes shows in their
#: own columns (an MRR and a top-10 figure), by its mode: its pooled MRR (README
#: §2.6 E1) and, for genes and patients-like-me, the macro recall@10 reported
#: beside it (E2); the disease task's pooled top-10 (one truth per case). A record
#: of any other mode shows every metric as recorded in one column instead.
RECORD_COLUMNS: Dict[str, Tuple[str, str]] = {
    "disease": ("pooled_mrr", "pooled_top10"),
    "gene": ("pooled_mrr", "macro_recall@10"),
    "patients_like_me": ("pooled_mrr", "macro_recall@10"),
}

NEWEST, BY_FAMILY = "newest", "family"
ORDER_CHOICES = [("Newest first", NEWEST), ("By family", BY_FAMILY)]
#: The Task select's options: the shared list's filters a page offers, in the
#: list's own words (`model_list.FILTER_WORDS`); "all" first, short, so the
#: workspace select beside it keeps the room its value needs.
TASK_CHOICES = [(ml.FILTER_WORDS[name][0], name)
                for name in ("all", "base", *TASK_NAMES)]
ALL_TASKS = "all"
#: The detail's segments: this checkpoint's recorded results, its usage
#: history, and the results of its family or workspace (Stage 8b,
#: `results_segment`).
RECORDS, USAGE, RESULTS = "records", "usage", "results"

#: By family's groups and extra column (the files whose record was refused or
#: never read are each grouped under their own `record_kind_text` word, the
#: one their rows lead with).
#: A family's group title on Models' By family order: its root run in the
#: one run words (`renderers.run_words`, which hold the title's one " · ").
FAMILY_TITLE = "Family from run {run} ({count} file{s})"
LINEAGE_COLUMN = "Lineage"
FAMILY_NOT_READ = "the family view was not read with this list"
#: A model record's operation (`record.OPERATIONS`) as the lineage says it.
OPERATION_WORDS = {"new": "New run", "resume": "Resumed run", "extend": "Extended run",
                   "branch": "Branch", "finetune": "Finetune"}
#: A branch that changed no setting of its parent (`parent_changes` `{}`;
#: ops-wiring round, decisions B1, B4): a new run from its parent's weights.
SAME_SETTINGS = "Same settings as its parent"
#: A model whose family is its own run (the family's root).
OWN_FAMILY = "its own run (the family's root)"
#: A model whose family is its parent's run, which the Parent row names
#: (rule 1.2: the run is said once; the browser check of B3 found it twice).
PARENT_FAMILY = "its parent's run (the family's root)"
#: The reference recipe as README §5 axis 2 has it read, never "identical to upstream".
REFERENCE_ANCHOR = ("The reference recipe is anchored on e95433a, with corrections "
                    "(Ref: README §2.4).")

LOAD_NOTE = ("Reloads the pipeline and asks once, from → to, as Change model does. Results "
             "already produced keep their own identity.")

READONLY_TITLE = "Recorded results of the model in use"
READONLY_WHERE = "from the ledger beside the loaded checkpoint · opens no checkpoint"
#: D8's three conditions, named together for a read-only request.
D8_CONDITIONS = ("Operator controls need this deployment set to M1, a request on this "
                 "machine's loopback (here or through SSH forwarding), and no forwarding "
                 "header.")
LIST_IS_C3 = ("Listing opens checkpoint files, which SPEC_4 classes as checkpoint inspection "
              "(C3), so the list is served only where the operator controls are (D8).")
#: Shown only for a listing D8 refused, so it leads with the lock (a refusal).
LIST_C3_LINE = f"{LOCK} Checkpoint list: operator controls only (D8 · SPEC_4 C3)"
NO_LOAD_HERE = ("No validation figures here: they need the checkpoint files; Load is an "
                "operator control.")
RECORDS_MODEL_CHANGED = ("The model in use changed while its recorded results were read; open "
                         "the page again.")

#: What a dataset record's origin means (the Diagnosis panel's words).
ORIGIN_MEANING = {
    "synthetic": "a generated stand-in, not a clinical knowledge graph",
    "files": "built from identified source files",
}
#: The help texts this page's ⓘ name (registered through `ctx.help`).
PAGE_HELP = {
    "models-ordering": (
        "Order",
        "How the list is ordered; the Task select chooses the rows.\n"
        "• Newest first: latest run first.\n"
        "• By family: each base, then its finetuned models.",
    ),
    "models-contested": (
        "Runs differ",
        "Runs kept under this record's conditions in the cohort version's report store have "
        "results that differ. Every run is kept as recorded, and each can be registered. "
        "Without deterministic algorithms, runs can differ. With them on, the runs were "
        "expected to agree, so the difference is flagged: investigate it, comparing "
        "environment facts first. Investigation notes are kept with the reports (Test page); "
        "they record what was found and change nothing.",
    ),
    "models-usage": (
        "Usage history",
        "Each cohort version keeps an append-only usage history (reports/usage.jsonl in its "
        "report store): every test run and every registration, whether started here or from "
        "the command line. Model loads from a list once ordered by test score stay shown as "
        "they were recorded.",
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

#: The list panel's sticky header (``#shep-mp-hd``, wide layout) as
#: ``--shep-mp-hd-h`` on the panel's scroll region (`style.sticky_room_js`):
#: its scroll padding, so a field focused in the panel (the workspace, the
#: task, the order) scrolls clear of the header, and the table's sticky
#: header's place under it.
HEADER_ROOM_JS = sticky_room_js("#shep-mp-hd", ".shep-model-main", "--shep-mp-hd-h")

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
    """A described measurement's value in its column or its words: four
    decimals. Every value is also shown unrounded, as recorded
    (`recorded_metric_html`), and a differing run's value only so
    (`contested_value_text`)."""
    if not _number(value):
        return "not reported"
    return f"{value:.4f}"


def metric_label(name: Any, mode: str) -> str:
    """A metric of a record of `mode` as named on this page (`caveats.metric_text`):
    the mode is always named, never assumed."""
    return metric_text(mode, str(name))


def record_mode(view: Mapping[str, Any]) -> Optional[str]:
    """A record view's measurement mode (`RecordView.mode`), or None when the
    view carries none it can read (never taken for another mode)."""
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
    """WP1's cohort version list as the page needs it (the usage history reads
    each version it lists): each version's label, manifest digest and role, or
    why the registry offers none."""
    from src.api.services.server_lists import list_cohort_versions

    listing = list_cohort_versions(environ=environ)
    versions = []
    for item in listing.items:
        versions.append({
            "label": item.label, "sha256": item.manifest_sha256, "role": item.role,
            "problem": item.problem,
        })
    return {"versions": versions, "problem": listing.problem}


def usage_entries(versions: Sequence[Mapping[str, Any]], resolver: CohortVersions,
                  shas: Iterable[str]) -> Dict[str, Any]:
    """The entries of every listed cohort version's usage history that name one
    of these checkpoints, by checkpoint digest, newest first, each with its
    version's label; and what could not be read (a listed version that cannot
    be read, or a line that is not an entry, is named, never skipped silently)."""
    from src.evaluation.report_store import read_usage_history

    wanted = {s for s in shas if r.is_digest(s)}
    by_sha: Dict[str, List[Dict[str, Any]]] = {}
    problems: List[str] = []
    read_from: List[str] = []
    if not wanted:
        return {"entries": {}, "problems": [], "read_from": []}
    for listed in versions:
        if listed.get("problem"):
            problems.append(f"{listed.get('label')}: not read: "
                            f"{r.short_reason(str(listed['problem']), 220)}")
            continue
        if not r.is_digest(listed.get("sha256")):
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
    listed checkpoint; each listed checkpoint's ledger records (with each
    record's run) and every record group's description come from the listing,
    which read them for its rows.
    Plain data for a `gr.State`."""
    data: Dict[str, Any] = {"listing_key": listing_key(listing), "readonly": False}
    try:
        data.update(read_versions(environ))
    except Exception:  # noqa: BLE001 - the list still shows; the usage history says why
        logger.exception("The cohort versions could not be listed")
        data.update({"versions": [], "problem": "the cohort versions could not be listed"})
    items = list(listing.get("items") or [])
    # The ledgers beside the listed checkpoints and their records' group
    # descriptions, as the shared listing read them once for its own rows
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
    digest, opening no checkpoint, each record with its run and group description. The handler
    takes the served identity before its one ``await``, which reads the files
    in the threadpool, and runs to completion here."""
    from fastapi.encoders import jsonable_encoder

    from src.api.routes import evaluations

    return jsonable_encoder(asyncio.run(evaluations.get_loaded_model_evaluations()))


def read_readonly() -> Dict[str, Any]:
    """The read-only view's data: the status (for the served digest and the
    time it was read), the route's answer for the served model, and each of its
    records' group descriptions **as the route answered them** (with no problem text: no
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
    # **Each record's group description as the route answered it**
    # (`record_states.contest_states`, the one implementation), never computed
    # a second time here, so this view and an API client cannot disagree
    # (BACKLOG 36). A record without one reads "comparison incomplete" where it
    # is shown (`contest_html`), never agreeing. The route gives null for every value
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
def record_view(record: Mapping[str, Any], reading: Optional[Sequence[Any]] = None, *,
                ledger: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """A ledger record as the API's record view (`RecordView`: aggregates and
    digests only, every field checked; its run and possible overlaps), plus what
    the page needs beside it: the condition group, the device and whether
    deterministic algorithms were on. `reading` is the record's
    ``[run_id, historical, problem, overlaps]`` as its ledger's boundary reads it
    (`record_views.ledger_records`). The listing gives this only the records the
    shared check already read as results (`record_views.read_records`, which
    lists every other one with its reason), so None, for a record that does
    not have the shape registration writes or whose run identity cannot be
    read (never shown half-read, never shown as historical), is a defensive
    case here; the server's log then says why in the listing's one bounded
    line (`unreadable_reason`, never an input value), naming `ledger` (the
    file it was read from) when given."""
    from src.api.services.record_views import _record_view, recorded_settings, unreadable_reason
    from src.evaluation.sidecar import RecordIdentity, record_key

    identity, overlaps = None, ()
    if reading is not None:
        run_id, historical, problem, found = reading
        identity, overlaps = RecordIdentity(run_id, bool(historical), problem), tuple(found)
    try:
        view = _record_view(dict(record), identity, overlaps).model_dump(mode="json")
        key = record_key(dict(record))
    except ValueError as exc:
        logger.warning("A ledger record%s is not shown: %s",
                       f" in {ledger}" if ledger else "", unreadable_reason(exc))
        return None
    # The record's settings as the one reader reads them (`recorded_settings`:
    # a value no answer can carry is not read, review fix RC-R1).
    regime = recorded_settings(record)["regime"]
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


def data_kind(view: Mapping[str, Any]) -> str:
    """The kind of data a record view measured (its ``cohort.kind``, by which
    the one reader read it): supplied, kg_edges or generated."""
    kind = _get(view, "cohort", "kind")
    return kind if isinstance(kind, str) else "supplied"


def role_text(view: Mapping[str, Any]) -> Optional[str]:
    """A version of another kind than a supplied cohort as the record's line
    names it, by its kind and role ("A KG edge set (test)", "A generated split
    (val)", `caveats.version_words`); None for a supplied cohort, whose line
    says "role <role>" as it always did. Never "cohort" for the others."""
    kind = data_kind(view)
    if kind == "supplied":
        return None
    words = version_words(kind, _get(view, "cohort", "role"))
    return words[:1].upper() + words[1:]


def coverage_html(view: Mapping[str, Any]) -> str:
    """The record's case sets: a task record's ranked cases of the
    import-eligible ones (PLAN_B05 §4); for any other mode, its source cases and
    exclusions and the cases its record counts, with no claim of what they are.
    A KG edge set's record: the pairs scored of those stored, and its positive
    edges; a generated split's: the cases ranked, and its generated patients
    and diseases (each count's unit is in the record's counts, below)."""
    kind = data_kind(view)
    counts = view.get("counts") if isinstance(view.get("counts"), Mapping) else {}
    count = r.recorded_count_text
    if kind == "kg_edges":
        run = _get(view, "measured", "counts")
        run = run if isinstance(run, Mapping) else {}
        return (f'<div>{r.e(count(run.get("pairs_scored")))} of '
                f'{r.e(count(counts.get("pairs")))} stored pairs scored</div>'
                f'<div class="shep-sub">{r.e(count(counts.get("positive_edges")))} '
                f'positive edges; {r.e(count(counts.get("edges_without_negative")))} '
                'without a negative to pair</div>')
    if kind == "generated":
        ranked = (f"{count(view.get('n_ranked'))} cases ranked by the model"
                  if record_mode(view) in TASK_NAMES else
                  f"{count(view.get('n_ranked'))} counted by the record; what they are is "
                  "not described for this mode")
        diseases = counts.get("diseases")
        return (f'<div>{r.e(ranked)}</div><div class="shep-sub">'
                f'{r.e(count(counts.get("samples")))} generated patients of '
                f'{r.e(count(diseases))} disease{"" if diseases == 1 else "s"}</div>')
    excluded = counts.get("excluded") if isinstance(counts.get("excluded"), Mapping) else None
    if excluded is None:
        reasons = "not reported"
    else:
        reasons = " · ".join(f"{str(name).replace('_', ' ')} {count}"
                             for name, count in excluded.items()) or "none"
    mode = record_mode(view)
    if mode in TASK_NAMES:
        # A task record's n_ranked counts the version's import-eligible cases
        # (query patients) with a ranked positive; its coverage metric is that
        # share (README §2.6 E5), as the Test page says it (`_record_html`).
        return (f'<div>{r.e(view.get("n_ranked"))} of {r.e(counts.get("ranked_eligible"))} '
                'import-eligible cases ranked by the model</div>'
                f'<div class="shep-sub">{r.e(counts.get("source_cases"))} source cases; '
                f'{r.e(counts.get("n_excluded"))} excluded at import: {r.e(reasons)}</div>')
    # A mode this version does not describe: its counts, and no claim of what
    # its n_ranked counts.
    return (f'<div>{r.e(counts.get("source_cases"))} source cases; '
            f'{r.e(counts.get("n_excluded"))} excluded: {r.e(reasons)}</div>'
            f'<div class="shep-sub">{r.e(view.get("n_ranked"))} cases counted by the record; what '
            'they are is not described for this mode</div>')


def _counts_fold(title: str, counts: Any, units: Any) -> str:
    """A version's or a run's counts, each with its unit as recorded, one
    click away: a closed disclosure, "<title> (N)"."""
    counts = counts if isinstance(counts, Mapping) else {}
    units = units if isinstance(units, Mapping) else {}
    lines = "".join(f'<span class="shep-mp-ln"><code>{r.e(name)}</code> '
                    f'{r.e(r.recorded_count_text(value))}: '
                    f'{r.e(units.get(name) or "unit not recorded")}'
                    '</span>' for name, value in counts.items())
    return (f'<details class="shep-mp-fold"><summary>{r.e(title)} ({len(counts)})</summary>'
            f'{lines}</details>')


def kind_html(view: Mapping[str, Any]) -> str:
    """The detail lines a record's data kind decides (research comparison
    Stage 6): a supplied cohort's R10 per scope (`r10_html`); a KG edge set's
    and a generated split's lines in the one rendering every page shares
    (`renderers.data_kind_lines`: the exposure, its source, the graph and pairs
    and the metrics not computable; the generation identity and how it stands
    to a model's training), then the counts with their units one click away.
    The KG edges' "held out" is never said for a patient cohort, and R10 never
    for a KG edge set or a generated split."""
    kind = data_kind(view)
    if kind == "supplied":
        return r10_html(view.get("r10"))
    lines = "".join(f'<span class="shep-mp-ln"><b>{r.e(label)}:</b> {value}</span>'
                    for label, value in r.data_kind_lines(view))
    counts = _counts_fold("Data version counts", view.get("counts"), view.get("count_units"))
    if kind == "generated":
        return lines + counts
    run_counts = _counts_fold("Run counts", _get(view, "measured", "counts"),
                              _get(view, "measured", "count_units"))
    return f"{lines}{counts}{run_counts}"


def r10_html(r10: Any) -> str:
    """A supplied cohort's R10, one line per scope, "R10, training inputs
    (evidence found): …": the one rendering (`renderers.r10_rows`), its
    scope's label in bold leading its row, as this page lays out a kind's
    lines."""
    return "".join(f'<span class="shep-mp-ln"><b>{r.e(label)}</b> {value}</span>'
                   for label, value in r.r10_rows(r10))


def contested_value_text(value: Any) -> str:
    """A differing run's value as `contest_html` shows it, whatever the mode:
    a finite number as recorded, unrounded (`caveats.recorded_value_text`), so
    two runs that differ never read as equal; null is "not reported"; a value
    that is not a finite number (NaN, infinity, text, an object, an integer
    too large for a float: a hand edit) is "not a finite number", never
    formatted as one."""
    if value is None:
        return "not reported"
    try:
        finite = _number(value) and math.isfinite(value)
    except OverflowError:  # an integer too large for a float
        finite = False
    if not finite:
        return "not a finite number"
    return recorded_value_text(value)


def contest_html(state: Optional[Mapping[str, Any]], *, mode: Optional[str],
                 deterministic: Optional[bool]) -> str:
    """How a record's condition group compares, beside it: "runs under these
    conditions differ" (with what differs, each metric named in the record's
    `mode` and each run's value as recorded, `contested_value_text`) or
    "comparison incomplete"; nothing when the runs agree. Every
    caller gives the mode and whether the group records deterministic
    algorithms on (`deterministic`, from the record's regime: every run of the
    group shares it); neither is assumed. Only a deterministic group's runs
    were expected to agree, so only there is the difference amber, with ▲ and
    "— investigate"; otherwise, and when it is not known (None), it is neutral.
    A description: it hides and refuses nothing."""
    if state is None:
        return ('<div class="shep-mp-delta shep-muted">comparison incomplete: it was '
                'not read</div>')
    kind = state.get("state")
    if kind == "uncontested":
        return ""
    if kind == "contested":
        values = []
        names = state.get("differing") or []
        for name in names:
            per_run = [f"{contested_value_text(run['values'][name])} (run "
                       f"{r.run_time(run['run']) or run['run']})"
                       for run in state.get("runs") or [] if name in (run.get("values") or {})]
            if per_run:
                values.append(f"{metric_label(name, mode)} " + " vs ".join(per_run))
        # The runs' values when they are given, else the fields' names alone
        # (an API answer that could not show them), else nothing more.
        shown, fields = "; ".join(values), ", ".join(str(n) for n in names)
        if deterministic is True:
            what = shown or fields
            said = f": {what}" if what else ""
            return ('<div class="shep-mp-delta">' + r.status_mark(
                "warn", f"runs under these conditions differ — investigate{said}. "
                        "Deterministic algorithms were on, so they were expected to agree.")
                    + "</div>")
        detail = f": {shown}" if shown else f" on: {fields}" if fields else ""
        return (f'<div class="shep-mp-delta shep-muted">runs under these conditions differ'
                f'{r.e(detail)}</div>')
    problems = "; ".join(str(p) for p in state.get("problems") or [])
    return ('<div class="shep-mp-delta shep-muted">comparison incomplete'
            + (f": {r.e(r.short_reason(problems, 160))}" if problems else "") + "</div>")


def merged_text(view: Mapping[str, Any]) -> Optional[str]:
    """A record's merged reports, shown only when there are any: "N reports
    merged into this record before each run had its own record" (K8-6r.2)."""
    n = view.get("n_corroborating")
    if not _number(n) or not n:
        return None
    return (f"{n} report{'s' if n != 1 else ''} merged into this record before each run had "
            "its own record")


def overlap_text(view: Mapping[str, Any]) -> Optional[str]:
    """"possible overlap with record N (not recorded)": the records this one
    may be one execution with (`RecordView.historical_overlap`), never
    confirmed; None when there are none."""
    found = [n for n in view.get("historical_overlap") or [] if _number(n)]
    if not found:
        return None
    many = len(found) != 1
    return (f"possible overlap with record{'s' if many else ''} "
            f"{', '.join(str(n) for n in found)} (not recorded)")


def run_text(view: Mapping[str, Any]) -> str:
    """The run a record was registered from, or "not recorded" for a record
    merged before each run had its own record."""
    run_id = view.get("run_id")
    return f"run {run_id}" if isinstance(run_id, str) and run_id else "run not recorded"


def _record_cell(view: Mapping[str, Any], state: Optional[Mapping[str, Any]]) -> str:
    """The record's sequence and its group's description: a "runs differ" chip,
    neutral, or amber with ▲ for a group that records deterministic algorithms
    on (its runs were expected to agree), which asks for attention and blocks
    nothing; the merged reports and a possible overlap with another record,
    when there are any."""
    sequence = _get(view, "registration", "sequence")
    kind = (state or {}).get("state")
    deterministic = view.get("deterministic") if type(view.get("deterministic")) is bool else None
    if kind == "contested":
        chip = (r.warn_chip(f"runs differ #{sequence}") if deterministic is True
                else r.chip(f"runs differ #{sequence}", "grey"))
        chip += help_button("models-contested", "Runs differ")
    elif kind == "uncontested":
        chip = r.chip(f"registered #{sequence}", "ok")
    else:
        chip = r.chip(f"registered #{sequence}", "grey")
    determinism = ("deterministic on" if deterministic is True else
                   "deterministic off" if deterministic is False else
                   "determinism not read")
    parts = [f'{chip} <span class="shep-sub">{r.e(determinism)}</span>']
    for text in (merged_text(view), overlap_text(view)):
        if text:
            parts.append(f'<div class="shep-sub">{r.e(text)}</div>')
    parts.append(contest_html(state, mode=record_mode(view), deterministic=deterministic))
    return "".join(parts)


def records_table_html(views: Sequence[Mapping[str, Any]], contest: Mapping[str, Any]) -> str:
    """The records, **one table per mode, never mixed**, the tables in the order
    their modes are first shown (`_by_mode`), each row a main row and a row of
    detail. **One display for every mode** (R2): the main row holds the cohort
    version (label, version digest, synthetic, role), the metrics, coverage and
    the record's state. A measurement this build describes shows its two
    columns (`RECORD_COLUMNS`) in words (`caveats.metric_text`); any other
    mode, a kept one included, shows every metric as recorded in one "Recorded
    metrics" column (`recorded_metric_html`). The detail row: relations, a
    described measurement's other metrics in words, R10 per scope, the score
    semantics as recorded, every metric as recorded behind a closed disclosure
    (a described measurement only), environment, dataset and registration."""
    return "".join(_mode_table(mode, group, contest) for mode, group in _by_mode(views))


def recorded_metric_html(metrics: Mapping[str, Any]) -> str:
    """Every metric of a record or usage entry as recorded, one per line
    (`caveats.recorded_metric_lines`: the name as written, a code, and the
    value's JSON text, unrounded), every value escaped; "no metric recorded"
    when it holds none. A record's metrics come through its record view
    (`RecordView`), which holds every metric as a number with a fraction, so a
    whole number recorded as 1 reads "1.0" here and "1" on the Test page,
    which shows the report's own value; a usage entry's are shown as its
    history holds them."""
    lines = recorded_metric_lines(metrics)
    if not lines:
        return '<span class="shep-mp-ln shep-muted">no metric recorded</span>'
    return "".join(f'<span class="shep-mp-ln"><code>{r.e(name)}</code> {r.e(value)}</span>'
                   for name, value in lines)


def recorded_metric_fold(metrics: Mapping[str, Any]) -> str:
    """A described measurement's metrics as recorded, one click away: a closed
    native disclosure, "Recorded metrics (N)" (skill §1.5, §10), or "no metric
    recorded" when it holds none."""
    if not metrics:
        return '<span class="shep-mp-ln shep-muted">no metric recorded</span>'
    return (f'<details class="shep-mp-fold"><summary>Recorded metrics ({len(metrics)})'
            f'</summary>{recorded_metric_html(metrics)}</details>')


def semantics_html(view: Mapping[str, Any]) -> str:
    """"Score semantics: <the record's `score_semantics` exactly as recorded>",
    or "not recorded": never split, interpreted or rebuilt."""
    raw = view.get("score_semantics")
    value = (f"<code>{r.e(raw)}</code>" if isinstance(raw, str) and raw.strip()
             else '<span class="shep-muted">not recorded</span>')
    return f'<span class="shep-mp-ln">Score semantics: {value}</span>'


def _mode_table(mode: Optional[str], views: Sequence[Mapping[str, Any]],
                contest: Mapping[str, Any]) -> str:
    """One mode's records as one table (`records_table_html`)."""
    columns = RECORD_COLUMNS.get(mode) if mode else None
    body = []
    for view in views:
        cohort = view.get("cohort") if isinstance(view.get("cohort"), Mapping) else {}
        metrics = view.get("metrics") if isinstance(view.get("metrics"), Mapping) else {}
        key = view.get("key")
        state = contest.get(key_text(key)) if key else None
        synthetic = (r.chip("synthetic", "amber") if view.get("dataset_origin") == "synthetic"
                     else "")
        role = cohort.get("role")
        # A version of another kind is named by its kind and role, never as
        # a cohort (`role_text`).
        role_line = role_text(view)
        version = r.fingerprint_value(cohort.get("version_manifest_sha256"),
                                      what="version fingerprint", copy=False)
        cohort_cell = (f'<div class="shep-w600">{r.e(cohort.get("label"))}</div>'
                       f'<div class="shep-sub">(version {version}) '
                       f'{synthetic}</div><div class="shep-sub">'
                       + (r.e(role_line) if role_line else f"role {r.ident_html(role)}")
                       + '</div>')
        if columns:
            mrr, hits = (fmt_metric(name, metrics.get(name)) for name in columns)
            metric_cells = (f'<td class="shep-num">{r.e(mrr)}</td>'
                            f'<td class="shep-num">{r.e(hits)}</td>')
            others = [f"{metric_label(name, mode)} {fmt_metric(name, value)}"
                      for name, value in sorted(metrics.items(),
                                                key=lambda kv: _metric_order(kv[0]))
                      if name not in columns]
            fold = f'<div class="shep-mp-ln">{recorded_metric_fold(metrics)}</div>'
        else:
            # A mode this build has no words for: every metric, as recorded, is
            # the main result; the detail row repeats none of them.
            metric_cells = (f'<td colspan="2" class="shep-mp-recm">'
                            f'{recorded_metric_html(metrics)}</td>')
            others, fold = [], ""
        main = (f'<tr class="shep-mp-main"><td>{cohort_cell}</td>{metric_cells}'
                f'<td>{coverage_html(view)}</td>'
                f'<td>{_record_cell(view, state)}</td></tr>')
        gpu = view.get("gpu") if isinstance(view.get("gpu"), Mapping) else None
        environment = str(view.get("environment_class") or "not reported")
        if gpu is not None:
            environment += (f" · GPU {gpu.get('name') or 'name not read'}, driver "
                            f"{gpu.get('driver') or 'not read'}")
        # The run environment's one wording of the native build identity, on
        # one line (the Test page's narrower column puts a component per line).
        native = native_identity_text(view.get("native_identity"))
        origin = view.get("dataset_origin")
        # The record's run once, here; its merged reports and overlaps are in
        # the Record cell (the skill's rule 1.2).
        registration = (f"Registered #{_get(view, 'registration', 'sequence')} at "
                        f"{_utc_text(_get(view, 'registration', 'registered_at'))} from "
                        f"{run_text(view)}, report {r.short_hex(view.get('source'))}")
        # Each relation a row of its own, its name visible whole (the one
        # rendering, `renderers.relation_rows`), never a chain in a title.
        relations = r.summary_list(r.relation_rows(view.get("relations")), classes="shep-mp-rel")
        more = (f'<tr class="shep-mp-more"><td colspan="5">{relations}'
                + (f'<span class="shep-mp-ln">{r.e(" · ".join(others))}</span>' if others else "")
                + f'{kind_html(view)}{semantics_html(view)}{fold}'
                f'<span class="shep-mp-ln">Environment: {r.e(environment)} · Dataset: '
                f'<b>{r.e(origin)}</b> · {r.e(registration)}</span>'
                f'<span class="shep-mp-ln">Native build: {r.e(native)}</span></td></tr>')
        body.append(main + more)
    heads = (f'<th scope="col" class="shep-num">{_column_head(mode, 0)}</th>'
             f'<th scope="col" class="shep-num">{_column_head(mode, 1)}</th>' if columns else
             '<th scope="col" colspan="2">Recorded metrics</th>')
    # Each table named for its mode's label, so a screen reader tells them apart.
    return ('<div class="shep-tbl-wrap shep-mp-recwrap"><table class="shep-tbl shep-mp-rec" '
            f'aria-label="{r.e("Recorded results, " + mode_label(mode))}">'
            '<colgroup><col class="shep-mp-c1"><col '
            'class="shep-mp-c2"><col class="shep-mp-c3"><col class="shep-mp-c4"><col></colgroup>'
            f'<thead><tr><th scope="col">{_version_head(views)}</th>{heads}'
            '<th scope="col">Coverage</th>'
            f'<th scope="col">Record</th></tr></thead><tbody>{"".join(body)}</tbody></table>'
            '</div>')


def _version_head(views: Sequence[Mapping[str, Any]]) -> str:
    """The first column's header: "Cohort version" for a table of supplied
    cohorts' records, as it always was; "Data version" once a KG edge set's or
    a generated split's record is in it (never "cohort" for those)."""
    return ("Cohort version" if all(data_kind(view) == "supplied" for view in views)
            else "Data version")


def _by_mode(views: Sequence[Mapping[str, Any]]
             ) -> List[Tuple[Optional[str], List[Mapping[str, Any]]]]:
    """The records shown, grouped by mode, the groups in the order their modes
    are first shown and each group in the order given: no mode is put first or
    last by its name."""
    groups: Dict[Optional[str], List[Mapping[str, Any]]] = {}
    for view in views:
        groups.setdefault(record_mode(view), []).append(view)
    return list(groups.items())


def _column_head(mode: Optional[str], index: int) -> str:
    """A described measurement's metric column header: the metric in full, with
    its task ("Disease pooled MRR"); a part in parentheses kept on one line."""
    head, _sep, tail = metric_text(mode, RECORD_COLUMNS[mode][index]).partition(" (")
    return r.e(head) + (f' <span class="shep-nowrap">({r.e(tail)}</span>' if tail else "")


#: A task record's single figures, then its cut-off families, in the order a
#: record shows them (README §2.6: E1's pooled reading, E5's conservative one,
#: the macro and best-positive MRRs; pooled top-k, then E2's macro recall and
#: precision by k).
_TASK_FIGURES = ("pooled_mrr", "conservative_pooled_mrr", "macro_mrr", "best_positive_mrr")
_TASK_FAMILIES = ("pooled_top", "macro_recall@", "macro_precision@")


def _metric_order(name: str) -> Tuple[int, int, int, str]:
    """A described measurement's other metrics, in words: coverage, its MRRs,
    then each cut-off family by k; any other name after them, by name."""
    if name == "coverage":
        return (0, 0, 0, name)
    if name in _TASK_FIGURES:
        return (1, _TASK_FIGURES.index(name), 0, name)
    for index, prefix in enumerate(_TASK_FAMILIES):
        k = name[len(prefix):]
        if name.startswith(prefix) and k.isdigit():
            return (2 + index, 0, int(k), name)
    return (2 + len(_TASK_FAMILIES), 0, 0, name)


def records_block_html(views: Sequence[Mapping[str, Any]], contest: Mapping[str, Any], *,
                       unreadable: Sequence[Mapping[str, Any]] = ()) -> str:
    """Each shown mode's label (and a described measurement's note) once,
    directly above its own table, the tables in the order their modes are first
    shown; the dataset line; the R10 definitions behind a ⓘ; and the records
    that cannot be read as results (`unreadable_html`). The same for the
    operator's detail and the read-only view."""
    parts = []
    if views:
        origins = {v.get("dataset_origin") for v in views}
        if len(origins) == 1 and next(iter(origins)) in ORIGIN_MEANING:
            origin = next(iter(origins))
            word = (f'<b class="shep-amber-t">{r.e(origin)}</b>' if origin == "synthetic"
                    else f"<b>{r.e(origin)}</b>")
            parts.append(f'<p class="shep-mp-p">Dataset of every record below: {word} — '
                         f'{r.e(ORIGIN_MEANING[origin])}.</p>')
        for mode, group in _by_mode(views):
            note = mode_note(mode)
            parts.append(f'<p class="shep-mp-p"><b>{r.e(mode_label(mode))}.</b>'
                         + (f" {r.e(note)}" if note else "") + "</p>")
            parts.append(records_table_html(group, contest))
        # R10's definitions where a supplied cohort's record is shown; a KG
        # edge set's and a generated split's records carry none.
        r10 = (f'R10 labels {help_button("models-r10", "R10 labels")} · '
               if any(data_kind(view) == "supplied" for view in views) else "")
        parts.append(f'<p class="shep-note">{r10}The Trainer validation '
                     "figure is not among these records' metrics.</p>")
    parts.append(unreadable_html(unreadable))
    return "".join(parts)


def unreadable_html(items: Sequence[Mapping[str, Any]]) -> str:
    """The records for these weights that cannot be read as results, as the
    readers' one listing gives them (`record_views.read_records`; the API's
    ``unreadable_records``, K8-6e.1): one sentence in amber with ▲ (needs
    attention, blocks nothing; skill §6), then a small table, Index ·
    Registration · Report · Reason, every value escaped. A sequence or report
    digest the record does not carry readably is "not readable", never a
    dash. The reason is the listing's one line (`unreadable_reason`), which
    names each refused field by its path in the ledger record and never its
    value; a key the record itself carries inside an object (a metric's
    name, an exclusion reason) is named as written, as a readable record's
    metric names are shown. Nothing when there is none."""
    if not items:
        return ""
    many = len(items) != 1
    rows = []
    for item in items:
        sequence = item.get("sequence")
        digest = item.get("source_artifact_digest")
        registration = (f"#{sequence}" if _number(sequence) else
                        '<span class="shep-muted">not readable</span>')
        report = r.fingerprint_value(digest, what="report fingerprint", copy=False,
                                     missing="not readable")
        rows.append(f'<tr><td class="shep-num">{r.e(item.get("index"))}</td>'
                    f'<td class="shep-nowrap">{registration}</td><td>{report}</td>'
                    f'<td>{r.e(item.get("reason"))}</td></tr>')
    return ('<p class="shep-mp-p shep-mp-unread-p">' + r.status_mark(
                "warn", f'{len(items)} record{"s" if many else ""} for these weights cannot be read '
                        f'as {"results" if many else "a result"}. The ledger keeps '
                        f'{"them" if many else "it"} as written.') + '</p>'
            '<div class="shep-tbl-wrap shep-mp-recwrap shep-mp-unreadwrap"><table '
            'class="shep-tbl shep-mp-unread" '
            'aria-label="Records that cannot be read"><thead><tr>'
            '<th scope="col" class="shep-num">Index</th><th scope="col">Registration</th>'
            '<th scope="col">Report</th><th scope="col">Reason</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


# ---------------------------------------------------------------------------
# The controls, the order and the view (pure)
# ---------------------------------------------------------------------------
def controls(data: Mapping[str, Any], order: Any, task: Any) -> Dict[str, Any]:
    """The page's controls as they can apply now: a task filter of the Task
    select (else all) and the order, "By family" for an operator's list (else
    newest first)."""
    chosen_task = task if task in {value for _label, value in TASK_CHOICES} else ALL_TASKS
    readonly = bool(data.get("readonly"))
    mode = BY_FAMILY if order == BY_FAMILY and not readonly else NEWEST
    return {"task": chosen_task, "order": mode}


def parent_html(item: Mapping[str, Any], row: Optional[Mapping[str, Any]]) -> str:
    """Where the exact parent file a model's record names is (its 12-hex and
    run; listed here, in another listed workspace, or not available and why,
    from the family view's row); "none: a new run" for a file with no parent.
    The parent's fingerprint is the one digest value
    (`renderers.fingerprint_value`, its whole value its title; its Copy is in
    the parent's own details), its run named whole by the one run words
    (`renderers.run_words`, `fit` False: the record names its directory
    alone, so no layout), where it is, after a semicolon, so the row keeps
    the words' one " · "."""
    sha = item.get("parent_sha256")
    if not r.is_digest(sha):
        return "none: a new run"
    where = (f'{r.fingerprint_value(sha, what="parent fingerprint", copy=False)} of run '
             f'{r.e(r.run_words(None, item.get("parent_run"), fit=False))}')
    state = (row or {}).get("parent_state")
    if state == "listed":
        return f"{where}; listed here"
    if state == "elsewhere":
        return f'{where}; in workspace <b>{r.e(row.get("parent_workspace"))}</b>'
    if state == "unavailable":
        return (f'{where}<div>'
                + r.status_mark("warn", "parent unavailable: "
                                 + r.short_reason(row.get("parent_problem") or "not found", 160))
                + "</div>")
    return where


def changes_html(changes: Any, *, values: bool = True) -> str:
    """What a branch changed from its parent, from its listed record
    (`parent_changes`, decision B4): `SAME_SETTINGS` for ``{}``; else each
    setting changed, by its words (`renderers.field_words`), with its
    parent's value and its own (`renderers.change_value`: the one
    value-words rule, never Python's repr) (`values`), or by name alone;
    "not read" when the list holds none."""
    if not isinstance(changes, Mapping):
        return '<span class="shep-muted">not read</span>'
    if not changes:
        return r.e(SAME_SETTINGS)
    names = sorted(str(name) for name in changes)
    if not values:
        return "Changed: " + r.e(", ".join(r.field_words(name) for name in names))
    return "; ".join(f"{r.e(r.field_words(name))} {r.change_value(name, changes[name])}"
                     for name in names)


def lineage_cell_html(item: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    """A row's Lineage cell in By family: its operation and the exact parent
    file (or that the parent is not available, in amber, or that it is only a
    context row), and for a branch what it changed, by name, or that it kept
    its parent's settings (`changes_html`), indented by its depth in the
    family; a state's own word for a file with no lineage (a record not readable or not
    read: Unknown; a model record naming no operation: Not recorded). Its
    kind is the row's own line (the shared list's Model cell)."""
    depth = min(int(row.get("depth") or 0), 3)
    lines = []
    operation = item.get("operation")
    sha = item.get("parent_sha256")
    if operation and r.is_digest(sha):
        where = (f' in workspace <b>{r.e(row.get("parent_workspace"))}</b>'
                 if row.get("parent_state") == "elsewhere" else "")
        lines.append(f'<div class="shep-sub">{r.e(OPERATION_WORDS.get(operation, operation))} of '
                     f'{r.fingerprint_value(sha, what="parent fingerprint", copy=False)}'
                     f'{where}</div>')
    elif operation:
        lines.append(f'<div class="shep-sub">{r.e(OPERATION_WORDS.get(operation, operation))}'
                     '</div>')
    if operation == "branch":
        lines.append(f'<div class="shep-sub">'
                     f'{changes_html(item.get("parent_changes"), values=False)}</div>')
    if row.get("parent_state") == "unavailable":
        lines.append('<div class="shep-sub">' + r.status_mark("warn",
            "parent unavailable: " + r.short_reason(row.get("parent_problem") or "not found"))
            + "</div>")
    if row.get("context"):
        lines.append('<div class="shep-sub">'
                     + r.status_mark("neutral", "Context: parent of a shown file") + '</div>')
    # No lineage to show: a record not readable or not read holds none this
    # build reads (Unknown); a model record that names no operation, none
    # recorded (the skill's §6, never a bare dash).
    body = "".join(lines) or r.status_mark(
        "neutral", r.NOT_RECORDED if item.get("record") == "model" else r.UNKNOWN)
    return f'<div class="shep-mp-lin shep-mp-d{depth}">{body}</div>'


def family_view(listing: Mapping[str, Any], task: str) -> Optional[Dict[str, Any]]:
    """By family's rows for the Task filter, from the listing's family view
    (`model_list.family_views`, one listing): ``keys`` (context rows
    included), ``groups`` (one per family, parent first, in the view's order;
    under All the files whose record was not read and those whose record is
    not readable, each group under its own `record_kind_text` word) and the
    Lineage ``cells``. None when the listing carries no family view."""
    views = listing.get("family")
    if not isinstance(views, Mapping) or "all" not in views:
        return None
    rows = views.get(task) or []
    items = {ml.item_key(i): i for i in listing.get("items") or []}
    groups: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    cells: Dict[str, str] = {}
    for row in rows:
        key = row.get("key")
        if key not in items:
            continue
        kind = row.get("group")
        if kind == "family":
            name = row.get("family_id")
        else:
            # The server groups every file without a usable record as
            # "refused", and a record never read and a record refused say
            # different things (skill §6), each under its own word.
            name = r.record_kind_text(items[key])
        group = (kind, name)
        groups.setdefault(group, {"group": group, "keys": []})["keys"].append(key)
        cells[key] = lineage_cell_html(items[key], row)
    for group in groups.values():
        kind, name = group.pop("group")
        count = len(group["keys"])
        group["title"] = (FAMILY_TITLE.format(run=r.run_words(None, name), count=count,
                                              s="s" if count != 1 else "")
                          if kind == "family" else name)
    ordered = list(groups.values())
    return {"keys": [key for group in ordered for key in group["keys"]], "groups": ordered,
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
          ) -> Dict[str, Any]:
    """The shared list's ``view`` for these controls.

    Newest first: the Task filter and the Load note (each row's kind is the
    shared list's own line). By family: the families' groups and rows
    (`family_view`) with their Lineage column; without a family view, newest
    first and why (``refused``)."""
    if data.get("readonly"):
        return {"note": ""}
    task = ctl.get("task") or ALL_TASKS
    newest: Dict[str, Any] = {"task": task, "note": LOAD_NOTE}
    if ctl.get("order") == BY_FAMILY:
        family = family_view(listing, task)
        if family is None:
            return {**newest, "refused": FAMILY_NOT_READ}
        return {"task": task, "keys": family["keys"], "groups": family["groups"],
                "column": LINEAGE_COLUMN, "cells": family["cells"], "note": LOAD_NOTE}
    return newest


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
    return ('<div class="shep-panel-hd"><h2>Models</h2><span class="shep-panel-right">listed '
            'by the server (operator controls only)</span></div>')


def options_html(data: Mapping[str, Any], refused: Optional[str] = None) -> str:
    """Under the order options: when the order asked could not be applied
    (`refused`, the view's: By family without a family view), that the list
    stays newest first and why."""
    if data.get("readonly") or not refused:
        return ""
    return (f'<div class="shep-reason shep-amber-t">Not ordered, so the list stays newest '
            f'first: {r.e(r.short_reason(refused, 240))}</div>')


def workspace_line_html(listing: Mapping[str, Any]) -> str:
    """Under the workspace select: the chosen workspace's dataset identity
    without its name (the shared renderer, `dataset_identity_html`: the dataset
    fingerprint with the digits that tell the listed fingerprints apart,
    "synthetic", the record's state), said once here: the select's option is
    the name (`renderers.workspace_choices`). Nothing for a read-only request
    (no workspace is chosen there)."""
    if listing.get("refused"):
        return ""
    rows = listing.get("workspaces") or []
    row = next((w for w in rows if w.get("name") == listing.get("workspace")), None)
    if row is None:
        return ""
    digits = r.distinguishing_digits(w.get("dataset_fingerprint") for w in rows)
    return (f'<div class="shep-mp-wsline">'
            f'{r.dataset_identity_html(row, digits=digits, name=False)}</div>')


def _served_line(listing: Mapping[str, Any]) -> str:
    """What is served for diagnosis, as the listing's status read says it; "not
    read yet" before any read (the page as built, before its first listing),
    never "nothing is in use" (M5-3 review UI-4)."""
    if not listing.get("status_read"):
        return "in use for diagnosis: not read yet"
    served = listing.get("served_sha")
    if served:
        return ("in use for diagnosis: model "
                + r.fingerprint_value(served, what="model fingerprint", copy=False))
    if listing.get("status_problem"):
        return "in use for diagnosis: not read"
    return "nothing is in use for diagnosis"


def identity_html(item: Mapping[str, Any], row: Optional[Mapping[str, Any]], *,
                  workspaces: Sequence[Mapping[str, Any]] = ()) -> str:
    """**The selected checkpoint's details, as one summary list** (the
    skill's §4, README §5): its model fingerprint (12 hex with its Copy: the
    table shows each row's, this is the chosen one's to copy); its dataset by
    its relation to the workspace chosen above, whose identity that line says
    once (`renderers.dataset_relation_html`: "the selected workspace's
    current dataset", or which dataset, its fingerprint and "synthetic" where
    it is another); its kind
    (`renderers.model_kind_text`: a task model with a frozen graph backbone
    says so); for a model record its operation, the exact parent file and
    where it is (`parent_html`), for a branch the settings it changed from its
    parent, each with both values, or that it kept them all (`changes_html`),
    its family (its own run where it is the family's root, its parent's
    where the Parent row names that run, else the run by name), its encoder and
    its relation to the paper (axis 2: "matches the reference recipe" with
    what that recipe is, or the fields that differ); and the Trainer's
    validation figure, a row like every details list's (`renderers.validation_value`),
    never a box of its own. Its run directory, file and whole fingerprint are
    one "Details ›" away (the skill's §5: the full provenance one click
    away). Model Facts' Task and Configuration rows say the kind and axis 1
    elsewhere; neither is repeated here."""
    sha = item.get("sha256")
    rows = [("Fingerprint", r.fingerprint_value(sha, what="model fingerprint")),
            ("Dataset", r.dataset_relation_html(item, workspaces)),
            ("Kind", r.e(r.model_kind_text(item)))]
    if item.get("record") == "model":
        operation = item.get("operation")
        differences = item.get("differences")
        if differences:
            relation = "differs in: " + ", ".join(
                f'<span title="{r.e(d)}">{r.e(r.field_path_words(d))}</span>'
                for d in differences)
        elif differences is not None:
            relation = (f'matches the reference recipe<div class="shep-sub">'
                        f'{r.e(REFERENCE_ANCHOR)}</div>')
        else:
            relation = r.status_mark("neutral", r.UNKNOWN)
        family = item.get("family_id")
        rows += [("Operation", r.e(OPERATION_WORDS.get(operation, operation or "not read"))),
                 ("Parent", parent_html(item, row)),
                 *([("Settings changed", changes_html(item.get("parent_changes")))]
                   if operation == "branch" else []),
                 ("Family", r.e(OWN_FAMILY if family == item.get("run")
                                else PARENT_FAMILY if family == item.get("parent_run")
                                and r.is_digest(item.get("parent_sha256"))
                                else f'run {r.run_words(None, family, fit=False)}')),
                 ("Encoder", r.e(r.architecture_label(item.get("encoder")))),
                 ("Relation to the paper", relation)]
    rows.append(("Validation", r.e(r.validation_value(r.validation_figure(item),
                                                      read=r.record_read(item)))))
    architecture, run, filename = (item.get("architecture"), item.get("run"),
                                   item.get("filename"))
    unknown = r.status_mark("neutral", r.UNKNOWN)
    provenance = [("Run directory", r.ident_html(f"{architecture}/{run}") if architecture and run
                   else unknown),
                  ("File", r.ident_html(filename) if filename else unknown),
                  ("Whole fingerprint", r.fingerprint_value(sha, full=True, copy=False))]
    return (r.summary_list(rows, classes="shep-mp-ident",
                           helps={"Fingerprint": ("model-fingerprint", "Model fingerprint"),
                                  "Validation": ("validation-trainer", "Validation (Trainer)")})
            + r.details_list(provenance))


def detail_parts(listing: Mapping[str, Any], data: Mapping[str, Any], key: Any,
                 ctl: Mapping[str, Any]) -> Dict[str, Any]:
    """The detail of the selected checkpoint (operator), or the read-only view.
    Returns ``header``, ``top``, ``records``, ``usage`` (HTML) and ``seg``
    (the segment labels, or None when the segments are not shown; Results'
    content is its own handlers', `results_segment`)."""
    if data.get("readonly"):
        return readonly_parts(data)
    item = ml.find_item(listing, key)
    if item is None or item not in ml.shown_items(listing, shown_view(listing, ctl)):
        return {"header": ('<div class="shep-panel-hd"><h2>Details</h2>'
                           f'<span class="shep-panel-right">{_served_line(listing)}</span></div>'),
                "top": ('<p class="shep-muted shep-mp-p">Select a model to see its fingerprints, '
                        'its validation figure and its recorded results.</p>'),
                "records": "", "usage": "", "seg": None}
    in_use = bool(listing.get("served_sha")) and item.get("sha256") == listing.get("served_sha")
    # In use: its tag says so, and what is served is this model, said once;
    # not in use: the model that is, by its own fingerprint (another fact).
    tag = r.chip("In use (Diagnosis)", "accent") if in_use else r.chip("not in use", "grey")
    served = "" if in_use else f'<span class="shep-panel-right">{_served_line(listing)}</span>'
    # The title names the row (`SCROLL_JS` waits for it) and takes the focus.
    header = (f'<div class="shep-panel-hd"><h2 tabindex="-1" data-key="{r.e(ml.item_key(item))}">'
              f'{r.e(r.model_name_line(item))}</h2>{tag}{served}{BACK_BUTTON}</div>')
    workspaces = listing.get("workspaces") or []
    if not ml.loadable(item):
        why = item.get("problem") or "its digest was not read"
        top = (f'<div class="shep-reason">{r.e(item.get("filename"))} · metadata not readable · '
               f'cannot be loaded: {r.e(why)}</div>')
        return {"header": header, "top": top, "records": "", "usage": "", "seg": None}
    facts = r.facts_from_checkpoint(item, workspaces)
    # Model Facts' warnings, but its synthetic banner: that one speaks of "the
    # served workspace", which a listed checkpoint need not be; the workspace
    # line and the Dataset row say synthetic for this checkpoint.
    warnings = "".join(r.warning_html(h, d) for h, d in facts.warnings
                       if (h, d) != r.IDENTITY_BANNERS["synthetic"])
    top = (identity_html(item, family_row(listing, ml.item_key(item)), workspaces=workspaces)
           + warnings)
    ledger = (data.get("ledgers") or {}).get(ml.item_key(item))
    records, count, contested, unreadable = _records_for_item(ledger, data)
    usage, n_usage = usage_html(item, data)
    # The results, then the rest named apart: one " · " (skill §5).
    notes = [f"{unreadable} not readable"] if unreadable else []
    if contested:
        notes.append(f"{contested} where runs differ")
    seg_records = f"Recorded results · {count}" + (
        f" ({'; '.join(notes)})" if notes else "")
    return {"header": header, "top": top, "records": records, "usage": usage,
            "seg": [(seg_records, RECORDS), (f"Usage history · {n_usage}", USAGE),
                    ("Results", RESULTS)]}


def _records_for_item(ledger: Optional[Mapping[str, Any]], data: Mapping[str, Any]
                      ) -> Tuple[str, int, int, int]:
    """The ledger entry's block, how many records read as results, how many of
    them are in a group whose runs differ, and how many records cannot be read
    as results: the listing classified them once (`record_views.ledger_records`, the
    readers' one listing), so a record is never counted as both."""
    from src.evaluation.sidecar import LEDGER_FILENAME

    if ledger is None:
        return ('<p class="shep-mp-p shep-muted">The ledger beside this checkpoint was not '
                'read.</p>', 0, 0, 0)
    if ledger["state"] == "not_read":
        return (f'<p class="shep-mp-p shep-amber-t">The ledger (<code>{LEDGER_FILENAME}</code>) '
                f'beside this checkpoint was not read, so none of its records are shown: '
                f'{r.e(ledger.get("problem"))}</p>', 0, 0, 0)
    if ledger["state"] == "no_ledger":
        return (f'<p class="shep-mp-p">No ledger (<code>{LEDGER_FILENAME}</code>) is beside this '
                'checkpoint, so no recorded result can be shown; that does not mean these '
                'weights were never evaluated.</p>', 0, 0, 0)
    if ledger["state"] == "none":
        return f'<p class="shep-mp-p">{r.e(RECORD_ABSENCE)}</p>', 0, 0, 0
    views = []
    readings = ledger.get("readings") or [None] * len(ledger["records"])
    for record, reading in zip(ledger["records"], readings):
        view = record_view(record, reading, ledger=ledger.get("ledger"))
        if view is not None:  # read as a result by the listing; the view adds the page's extras
            views.append(view)
    views.sort(key=lambda v: -(_get(v, "registration", "sequence") or 0))
    unreadable = [item for item in ledger.get("unreadable") or [] if isinstance(item, Mapping)]
    contest = data.get("contest") or {}
    contested = sum(1 for v in views
                    if (contest.get(key_text(v["key"])) or {}).get("state") == "contested")
    return (records_block_html(views, contest, unreadable=unreadable),
            len(views), contested, len(unreadable))


def usage_entry_html(entry: Mapping[str, Any]) -> str:
    kind = entry.get("kind")
    when = _utc_text(entry.get("at"))
    version = r.e(entry.get("version"))
    if kind == "test_run":
        from src.evaluation.report_store import entry_mode

        recorded = entry.get("metrics")
        metrics = recorded if isinstance(recorded, Mapping) else {}
        # Named in the entry's own mode, which the figure's label carries; an
        # entry that records none reads "Measurement mode not recorded".
        mode = entry_mode(entry)
        what = "Test run"
        columns = RECORD_COLUMNS.get(mode)
        n_ranked = entry.get("n_ranked")
        counted = type(n_ranked) is int
        if columns is None:
            # What n_ranked counts: a KG link-prediction run's pairs, any
            # other mode's cases.
            unit = "pairs" if mode == "kg_link_prediction" else "cases"
            figure = (f"{mode_label(mode)}, "
                      + (f"{n_ranked} {unit}" if counted else f"{unit[:-1]} count not recorded"))
        else:
            value = f"{metric_label(columns[0], mode)} {fmt_metric(columns[0], metrics.get(columns[0]))}"
            figure = (f"{value} over {n_ranked} ranked cases" if counted
                      else f"{value}, case count not recorded")
        # One " · " per line (the skill's §5): the figure, then how it ran,
        # then every metric the entry records, as recorded, one click away.
        how = (regime_text(entry.get("device"), entry.get("deterministic"))
               + ("; registration asked" if entry.get("record_requested") else ""))
        listed = (recorded_metric_fold(metrics)
                  if recorded is None or isinstance(recorded, Mapping) else
                  '<span class="shep-mp-ln shep-muted">metrics not readable as recorded'
                  '</span>')
        detail = f"run {entry.get('run_id')}"
        return (f'<tr><td class="shep-nowrap">{r.e(when)}</td><td>{version}</td>'
                f'<td><div class="shep-w600">{r.e(what)}</div><div class="shep-sub">'
                f'{r.e(detail)}</div></td><td><div>{r.e(figure)}</div>'
                f'<div class="shep-sub">{r.e(how)}</div>{listed}</td></tr>')
    elif kind == "registered":
        what = "Registered"
        result = f"#{entry.get('sequence')} · {entry.get('outcome')}"
        detail = f"report of run {entry.get('run_id')}"
    elif kind == "model_load":
        # Kept as written: no Load writes one since By test score retired
        # (K8-6d). Its metric is said as recorded, never given a mode it does
        # not record.
        what = "Model load from an ordered list"
        loaded = entry.get("loaded")
        result = ("loaded" if loaded is True else
                  f"not loaded: {entry.get('why') or 'no reason recorded'}")
        detail = (f"ordered by {entry.get('metric') or 'a metric not recorded'} · scoring "
                  f"semantics {r.short_hex(entry.get('scoring_semantics_digest'))}")
    else:
        what, result, detail = str(kind or "entry"), "", ""
    return (f'<tr><td class="shep-nowrap">{r.e(when)}</td><td>{version}</td>'
            f'<td><div class="shep-w600">{r.e(what)}</div><div class="shep-sub">{r.e(detail)}'
            f'</div></td><td>{r.e(result)}</td></tr>')


def usage_html(item: Mapping[str, Any], data: Mapping[str, Any]) -> Tuple[str, int]:
    """The usage history entries naming this checkpoint, newest first; with no
    cohort version read, why (the registry's own reason, when it gave one)."""
    usage = data.get("usage") or {}
    entries = (usage.get("entries") or {}).get(item.get("sha256")) or []
    read_from = usage.get("read_from") or []
    unread = "no cohort version was read" + (
        f" ({r.short_reason(str(data['problem']), 220)})" if data.get("problem") else "")
    parts = [f'<p class="shep-mp-p">Entries naming this checkpoint in the usage history of '
             f'every listed cohort version{help_button("models-usage", "Usage history")}: '
             f'{r.e(", ".join(read_from) if read_from else unread)}.</p>']
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
    """"Recorded results of the model in use": the served digest (from
    the status), then the route's records for it, each with its run and its
    group's description. Nothing here opened a checkpoint."""
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
    if state == "unavailable" or state not in ("no_model", "no_ledger", "ledger_not_read",
                                               "no_records", "records"):
        body = '<p class="shep-mp-p shep-amber-t">The recorded results could not be read.</p>'
    elif answer.get("checkpoint_sha256") != served:
        body = f'<p class="shep-mp-p shep-amber-t">{r.e(RECORDS_MODEL_CHANGED)}</p>'
    elif state == "no_model":
        body = ('<p class="shep-mp-p">No model is loaded, so there are no recorded results to '
                'look up.</p>')
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
                                  unreadable=[item for item in
                                              answer.get("unreadable_records") or []
                                              if isinstance(item, Mapping)])
    return {"header": header, "top": top, "records": body + note, "usage": "", "seg": None}


# ---------------------------------------------------------------------------
# The component
# ---------------------------------------------------------------------------
def create_models_page(ctx: Any) -> None:
    """Build Research › Evaluate › Models (the page contract: `src/webui/shell.py`)."""
    # The Results segment's module reads this one's words and renderers, so it
    # is imported here, never at the top (no import cycle).
    from src.webui.components import results_segment as rs

    ctx.help.topics(PAGE_HELP)
    ctx.help.topics(rs.SEGMENT_HELP)
    data_state = gr.State({"readonly": False, "listing_key": None, "versions": [], "ledgers": {},
                           "contest": {}, "usage": {}})
    #: True from "Reload with …" until the page reads its data after that Load.
    stale_state = gr.State(False)
    #: The last checkpoint selected, ``{"workspace", "key"}``: selected again after a Load.
    last_state = gr.State(None)
    parts: Dict[str, Any] = {}

    def head() -> Dict[str, Any]:
        # The header's height on the list panel's scroll region, its scroll
        # padding (`PAGE_CSS`): a field focused there scrolls clear of the
        # sticky header (the one script for a sticky block's room).
        parts["lhd"] = plain_html(left_header_html({}, {}), elem_id="shep-mp-hd",
                                  js_on_load=HEADER_ROOM_JS)
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
            parts["options"] = plain_html("", elem_id="shep-mp-options")
        return parts

    def foot() -> Any:
        with gr.Column(elem_id="shep-mp-detail", elem_classes=["shep-mp-detail"]) as column:
            parts["detail_hd"] = plain_html(detail_parts({}, {}, None, {})["header"],
                                            elem_id="shep-mp-detail-hd", js_on_load=SCROLL_JS)
            with gr.Column(elem_id="shep-mp-detail-bd", elem_classes=["shep-mp-detail-bd"]):
                parts["detail_top"] = plain_html(detail_parts({}, {}, None, {})["top"],
                                                 elem_id="shep-mp-detail-top")
                parts["seg"] = gr.Radio(choices=[("Recorded results", RECORDS),
                                                 ("Usage history", USAGE),
                                                 ("Results", RESULTS)], value=RECORDS,
                                        show_label=False, label="Shown", visible=False,
                                        elem_id="shep-mp-seg", elem_classes=["shep-seg"],
                                        container=False)
                parts["records"] = plain_html("", elem_id="shep-mp-records")
                parts["usage"] = plain_html("", elem_id="shep-mp-usage", visible=False)
                parts["results"] = rs.create_results_segment()
        return column

    with gr.Row(elem_classes=["shep-page", "shep-models"], equal_height=False):
        handle = ctx.create_model_list("research", head=head, foot=foot)

    control_inputs = [parts["order"], parts["task"]]
    #: By element id: once the page has updated a control, Gradio keeps a new
    #: instance of it in the session (another ``_id``), which an event names.
    control_ids = {c.elem_id for c in control_inputs}
    page_outputs = [data_state, handle.view, parts["task"], parts["order"], parts["options"],
                    parts["lhd"], parts["wsline"], parts["detail_hd"], parts["detail_top"],
                    parts["seg"], parts["records"], parts["usage"],
                    parts["results"]["column"]]

    def detail_updates(listing, data, selected, ctl, seg_value) -> Tuple[Any, ...]:
        """The detail's header, top, segments, records, usage history and the
        Results segment's column (its content is its own handlers'); the
        segment shown is kept across selections."""
        detail = detail_parts(listing, data, selected, ctl)
        seg_shown = detail["seg"] is not None
        seg = seg_value if seg_value in (RECORDS, USAGE, RESULTS) else RECORDS
        return (detail["header"], detail["top"],
                (gr.update(choices=detail["seg"], value=seg, visible=True) if seg_shown
                 else gr.update(visible=False)),
                gr.update(value=detail["records"], visible=(not seg_shown) or seg == RECORDS),
                gr.update(value=detail["usage"], visible=seg_shown and seg == USAGE),
                gr.update(visible=seg_shown and seg == RESULTS))

    def outputs_for(listing, data, ctl_values, selected, seg_value, condition, *,
                    new_data: bool) -> Tuple[Any, ...]:
        ctl = controls(data, *ctl_values)
        view = shape(listing, data, ctl)
        return (
            data if new_data else gr.update(),
            view,
            gr.update(value=ctl["task"]),
            # Always offered to an operator; off for a read-only request.
            gr.update(value=ctl["order"], interactive=not data.get("readonly")),
            options_html(data, view.get("refused")),
            left_header_html(listing, data, condition),
            workspace_line_html(listing),
            *detail_updates(listing, data, selected, ctl, seg_value),
        )

    def read_for(listing) -> Dict[str, Any]:
        if listing.get("refused"):
            return read_readonly()
        return read_page(listing)

    shared_refresh = handle.refresh["fn"]

    def _open(name, older, current_view, previous, key, order, task, seg_value, phase,
              request: gr.Request):
        shared = shared_refresh(name, older, current_view, previous, key, request)
        listing = shared[0]
        data = read_for(listing)
        condition = operator_access(request).condition
        # Read now; a Load still running reads again once it has finished.
        stale = False if phase != "loading" else gr.update()
        return (*shared, stale, *outputs_for(listing, data, (order, task), None, seg_value,
                                             condition, new_data=True))

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

    def _sync(listing, data, stale, last, phase, order, task, selected, seg_value,
              request: gr.Request, evt: gr.EventData):
        """A new listing (a workspace chosen, the refresh after a Load), a
        selection, or a control changed. The page reads its own data again for
        a listing it has not read, and once after a Load has finished (the
        listing may be unchanged); after a Load it selects the last checkpoint
        again while its row is shown, when the shared refresh did not keep it."""
        ctl_values = (order, task)
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
            shown = shown_view(listing, controls(data, *ctl_values))
            if item is not None and item in ml.shown_items(listing, shown):
                key = again = last["key"]
        states = (False if fresh and settled else gr.update(),
                  ({"workspace": listing.get("workspace"), "key": key} if key is not None
                   else gr.update()),
                  again if again is not None else gr.update())
        by_control = getattr(getattr(evt, "target", None), "elem_id", None) in control_ids
        if not (fresh or again is not None or by_control):
            ctl = controls(data, *ctl_values)
            return (*states, *(gr.update() for _ in page_outputs[:-6]),
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
        return (gr.update(visible=value == RECORDS), gr.update(visible=value == USAGE),
                gr.update(visible=value == RESULTS))

    parts["seg"].input(_segment, inputs=[parts["seg"]],
                       outputs=[parts["records"], parts["usage"], parts["results"]["column"]],
                       queue=False, show_progress="hidden")
    # Models › Results (research comparison Stage 8b): its own gated handlers.
    rs.wire_results_segment(parts["results"], seg=parts["seg"], listing=handle.listing,
                            selected=handle.selected)


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
     longest option, "Patients-like-me" (the workspace option is its name; the
     identity line under it says its fingerprint and "synthetic", once). */
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
{_PAGE} #shep-models-research-workspace input, {_PAGE} #shep-mp-task input {{
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
  grid-template-columns: auto minmax(0, auto) minmax(min(16em, 33%), 1fr);
  grid-template-rows: minmax(0, 1fr) auto;
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
{_PAGE} .shep-mp-back {{ display: none; }}
{_PAGE} .shep-mp-p {{ margin: 0; font-size: 15px; color: var(--shep-text-2); line-height: 1.45; }}
{_PAGE} .shep-mp-p.shep-amber-t {{ color: var(--shep-amber-800); }}
{_PAGE} .shep-mp-recwrap {{ margin: 0 !important; }}
{_PAGE} .shep-mp-recwrap + .shep-mp-p {{ margin-top: 16px; }}
{_PAGE} .shep-note + .shep-mp-unread-p {{ margin-top: 16px; }}
{_PAGE} .shep-mp-unread-p + .shep-mp-recwrap {{ margin-top: 8px !important; }}
{_PAGE} .shep-mp-unreadwrap + .shep-note {{ margin-top: 16px; }}
{_PAGE} .shep-mp-rec td, {_PAGE} .shep-mp-usage td, {_PAGE} .shep-mp-unread td {{ overflow-wrap: anywhere; }}
{_PAGE} .shep-mp-rec {{ table-layout: fixed; }}
{_PAGE} .shep-mp-rec .shep-mp-c1 {{ width: 24%; }}
{_PAGE} .shep-mp-rec .shep-mp-c2 {{ width: 14%; }}
{_PAGE} .shep-mp-rec .shep-mp-c3 {{ width: 11%; }}
{_PAGE} .shep-mp-rec .shep-mp-c4 {{ width: 25%; }}
{_PAGE} .shep-mp-rec th {{ overflow-wrap: anywhere; }}
{_PAGE} .shep-mp-rec tr.shep-mp-main td {{ border-bottom: 0; }}
{_PAGE} .shep-mp-rec tr.shep-mp-more td {{ background: var(--shep-panel-2); font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} .shep-mp-ln {{ display: block; line-height: 1.45; }}
/* Models › Results (Stage 8b): its controls in rows that wrap, its tables
   scrolling sideways inside their own wrapper, Compare's groups as row
   headers. A wrapper not scrolled to its end shades its right edge, so a
   column cut there reads as more to the right: a shade fixed at the edge
   (`scroll`) under a cover that moves with the table (`local`) and covers it
   at the table's end (a table that fits is at its end); it shows through
   every cell without a background of its own. A protocol table's words
   break only where they must: with
   `break-word` a column is never sized narrower than its longest word, so a
   word, a 12-hex fingerprint or a raw value stays whole (with `anywhere`
   every character was a break point when the columns were sized, and the
   other columns were squeezed around the one-line result name); a result's
   name wraps only after its " · " (`results_segment._name_html`).
   Compare's prose, JSON and reasons may break anywhere, so its long values
   never set its columns' widths; a number never breaks (a raw value split
   mid-digit read as two numbers), nor does a run id between its hyphens
   (`results_segment._run_id_html`). Five columns of raw values can be wider
   than the panel: the table then scrolls sideways inside its own wrapper,
   never the page, its Field column pinned at the left with the words of its
   group and sub-heading rows (`shep-rs-pin`). A nested entry's row, named by
   its own words, sits one step in under the sub-heading that says its path
   (a plain cell styled as a heading, `results_segment.compare_html`). */
{_PAGE} .shep-rs {{ gap: 12px !important; }}
{_PAGE} .shep-rs-row {{ align-items: flex-end !important; gap: 8px !important; flex-wrap: wrap !important; }}
{_PAGE} .shep-rs-wrap {{ margin: 0 !important; overflow-x: auto;
  background: linear-gradient(to left, var(--shep-panel) 14px, transparent) right / 28px 100% no-repeat local,
    linear-gradient(to left, rgba(20, 24, 40, .12), transparent) right / 14px 100% no-repeat scroll,
    var(--shep-panel); }}
{_PAGE} .shep-rs-wrap + .shep-mp-p {{ margin-top: 16px; }}
{_PAGE} .shep-rs-tbl td {{ overflow-wrap: break-word; vertical-align: top; }}
{_PAGE} .shep-rs-cmp td, {_PAGE} .shep-rs-unread td {{ overflow-wrap: anywhere; vertical-align: top; }}
{_PAGE} .shep-rs-cmp td.shep-num {{ white-space: nowrap; overflow-wrap: normal; }}
{_PAGE} .shep-rs-cmp tbody th {{ position: static; }}
{_PAGE} .shep-rs-cmp tbody th[scope="row"], {_PAGE} .shep-rs-cmp thead th:first-child {{ position: sticky; left: 0; box-shadow: inset -1px 0 0 var(--shep-border); }}
{_PAGE} .shep-rs-cmp tbody th[scope="row"] {{ top: auto; z-index: 1; }}
{_PAGE} .shep-rs-cmp thead th:first-child {{ z-index: 2; }}
{_PAGE} .shep-rs-cmp th[scope="row"] {{ text-align: left; font-weight: 400; color: var(--shep-text-2); background: var(--shep-panel); }}
{_PAGE} .shep-rs-cmp tr.shep-rs-entry th[scope="row"] {{ padding-left: 22px; }}
{_PAGE} .shep-rs-cmp tr.shep-rs-grp th {{ text-align: left; font-weight: 600; background: var(--shep-panel-2); }}
{_PAGE} .shep-rs-cmp tr.shep-rs-sub td {{ text-align: left; font-size: 14px; font-weight: 600; color: var(--shep-text-2); line-height: inherit; padding: 12px 10px 4px; border-bottom: 0; }}
{_PAGE} .shep-rs-cmp .shep-rs-pin {{ position: sticky; left: 10px; display: inline-block; }}
/* Compare's sideways scroll cue at its top (`results_segment.SCROLL_CUE`): a line
   and a bar of the table's width that scrolls with it, as the foot's scrollbar
   does, shown only while the table is wider than the panel. */
{_PAGE} .shep-rs-cue {{ display: flex; flex-direction: column; gap: 2px; margin: 0 0 2px; }}
{_PAGE} .shep-rs-cue[hidden] {{ display: none; }}
{_PAGE} .shep-rs-top {{ overflow-x: auto; overflow-y: hidden; }}
{_PAGE} .shep-rs-top-w {{ height: 1px; }}
{_PAGE} .shep-rs-cmp col.shep-rs-cfield {{ width: 14em; }}
/* Every metric as recorded (R2): a closed native disclosure, its summary a 24 px
   target that shows the keyboard focus (skill §10). */
{_PAGE} .shep-mp-fold > summary {{ cursor: pointer; min-height: 24px; line-height: 24px; width: max-content; max-width: 100%; }}
{_PAGE} .shep-mp-fold > summary:focus-visible {{ outline: 2px solid var(--shep-accent); outline-offset: 2px; }}
{_PAGE} .shep-mp-fold[open] > summary {{ margin-bottom: 2px; }}
{_PAGE} .shep-mp-delta {{ font-size: 14px; margin-top: 4px; }}
{_PAGE} .shep-mp-refusal {{ display: flex; flex-direction: column; gap: 6px; margin: 12px 0;
  border: 1px solid var(--shep-grey-200); background: var(--shep-grey-50); border-radius: 8px;
  padding: 12px 14px; font-size: 15px; color: var(--shep-grey-800); }}
{_PAGE} .shep-mp-ro {{ padding: 0 16px 8px; }}
{_PAGE} .shep-mtbl td.shep-num .shep-sub {{ text-align: right; }}
/* By family's Lineage cell: a child sits one step in from its parent; a
   parent that is not available is a condition to watch, a context row a
   neutral state (`renderers.status_mark`, skill §6). */
{_PAGE} .shep-mp-lin.shep-mp-d1 {{ padding-left: 12px; }}
{_PAGE} .shep-mp-lin.shep-mp-d2 {{ padding-left: 24px; }}
{_PAGE} .shep-mp-lin.shep-mp-d3 {{ padding-left: 36px; }}
{_PAGE} .shep-mp-ident {{ font-size: 15px; }}
/* A record's relations, a row each under its detail row's words (`renderers.relation_rows`). */
{_PAGE} .shep-mp-rel {{ margin: 0 0 6px; font-size: 14px; }}
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
  /* The sticky header's room (`HEADER_ROOM_JS`): a field focused in the panel scrolls clear of it. */
  {_PAGE} {_LIST} > .shep-model-main {{ height: 100%; overflow-y: auto; scroll-padding-top: var(--shep-mp-hd-h, 48px); }}
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
  {_PAGE} #shep-models-research-rows .shep-tbl th {{ top: var(--shep-mp-hd-h, 48px); }}
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
    "D8_CONDITIONS",
    "FAMILY_TITLE",
    "LINEAGE_COLUMN",
    "LOAD_NOTE",
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
    "left_header_html",
    "lineage_cell_html",
    "listing_key",
    "options_html",
    "parent_html",
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
