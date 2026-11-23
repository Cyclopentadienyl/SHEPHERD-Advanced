"""
Research › Evaluate › Test: import a cohort version, run a test, keep and register its report (M5-2, UI-D).
==========================================================================================================
Every data kind of the registry (research comparison Stage 7): a supplied
cohort, a KG edge set and a generated split are listed, measured and
registered here.
`docs/working/PLAN_TEST_RESULTS.md` §6 ("M5's boundaries": the Test tab on
M4's flow), D5, D7 and D8; the page contract is `src/webui/shell.py`'s
docstring. Three panels, as the mockup draws them (``test-wide.png``,
``test-import.png``):

- **Data**: the workspace (the server's list, the shared dataset renderer),
  the data version (the registry's list by kind and role: a supplied cohort
  by its role and case counts, "KG edge set (test)", "Generated split (val)",
  each with its counts in its own units, `version_choices`, `version_html`;
  with whether it is mapped against this workspace's graph, a shared mark:
  ▲ with the refusal beside the version when it was mapped against another
  graph, which Start's reason then points to, `ANOTHER_GRAPH`; when the
  registry holds no KG edge set, the command that imports one on the
  machine, `kg_edges_empty_html`), and "Import a new version…" (a supplied
  cohort);
- **Run**: the model as **Run → Checkpoint**, the shared cascading pair
  (`model_list.parents_update` over `model_parents`, nothing chosen until
  the operator picks a run: `model_list.run_choices` and
  `checkpoint_choices`, each option at most 40 characters, the run's name or
  start time first, a file by what it is and how far it trained; each value
  the listing's key; WP1's
  checkpoint list of the workspace, every checkpoint: listing opens every
  checkpoint, so it is an operator control and a read-only request is not
  shown it), the chosen model's details under them (`model_html`: what its
  record says it is, `renderers.model_kind_text`, "Disease · frozen graph
  backbone" for the control, its file, the Trainer's figure, its
  fingerprint and its dataset by its relation to the workspace's), the
  measurement
  (a radio of the protocols this build implements that the chosen model's
  listed capabilities admit and that measure the chosen version's kind,
  `protocols.available`: the only one is chosen, of several none is until the
  operator picks one, and with none Start says why; one Start runs one
  measurement), the R10 evidence by the runner's one choice by the version's
  kind (`run_evidence`, `training_manager.evidence_choice`: the server's
  setting for a supplied cohort, a refused setting a red ✕ with its words;
  "Not applicable" with why for a KG edge set or a generated split, which is
  given none and does not need the setting), "Register when the run ends"
  and "Deterministic execution", the batch size and device the server fixes
  (D5), each with its own ⓘ,
  "Start test" through the job runner (`TrainingManager.start_test`, given
  the measurement chosen, the model's learning stage, task and capability
  description and the version's kind from the session's listings) with the
  **full digests shown** as the expected ones, so
  the run refuses before measuring when the file or the version it reads is
  not the one chosen. **A run starts only for a measurement this build
  implements that takes the model's record and the version's inputs**
  (`measure_refusal`, one rule for Start's reason and for the start itself,
  which asks `protocols.refusal` after the page's own record states): a
  record refused (a file without a model spec among them) or not read, a measurement
  that does not measure the version's data kind, a measurement whose needs the model's
  capabilities do not meet (its task head among them), and a measurement
  needing genes the version does not list are refused with their reason; then the job:
  its state and elapsed time in the panel's foot, "Cancel" beside them while
  a run is busy (the runner's stop for test runs only), and its card: once it
  has ended **what it left, first** (a kept report is Result's, which says
  its registration; or no report kept, from the runner's outcome read from
  disk by WP4's helper), its inputs (one summary list, a row each, the run's
  own whatever is chosen now: `job_input_rows`), and one "Details ›" with the
  rest: its settings, what the runner read at its end, its exit status and
  its output's tail (B52-E). While no run was started, one muted line in the
  foot and no card (R-81);
- **Result**: at its top, the version's report store as a **compact
  single-select table** (`reports_html`), newest first (the newest
  `ROWS_SHOWN` and the selected one shown, "Show all N reports" for the
  rest): each run's time, its model by its relation to the chosen one ("the
  selected model", or another listed file by what it is, its run and its
  file in its Run → Checkpoint options' words, `_other_model_html`, or its
  fingerprint), its measurement (`measurement_words`) and its state (kept,
  registered by the registration's one helper,
  `server_lists.stored_report_state`, asked again for an operator
  beside the listed checkpoint Register would use: `refine_rows`,
  `ledger_checkpoint`; how its condition group reads, "runs differ",
  "comparison incomplete"), all from the server's one run listing that
  ``GET /api/v1/cohorts`` shares (`server_lists.version_item`); a running
  test, or the last one that ended with nothing in the store, is a row no
  click selects. Under it, in the panel's one scroll region, the selected report as the record the ledger would hold
  (`build_record`, through the store's scan), read by the one reader as its
  data kind's result view (`record_views.result_view`, the results service's
  reading of a kept report, registered or not; a record it cannot read is
  shown with its reason, not as a result), **what it is of said by its
  relation to what is chosen now** (`result_identity_rows`: "the selected
  workspace", "the selected model", "the selected version" when they are
  the same, each in full when it is not, so a past result keeps its own
  identity after the selection changes; its run id once), its kind's lines in the rendering
  Models shares (`renderers.data_kind_lines`), by **one display for every
  mode** (R2, K8-6e.2): a measurement this build describes in its tiles and
  words (`caveats.metric_text`) with every metric as recorded one click away;
  any other mode's metrics, a kept report's included, as recorded
  (`caveats.recorded_metric_lines`); the score semantics as recorded,
  coverage with every exclusion, each relation a row of its own, its meaning
  first (`renderers.relations_html`), R10 as its table in the
  caveats' words (`renderers.r10_table_html`: *unverifiable* is a word,
  never a 0), environment, dataset
  origin, whether it is kept and registered; "Register this report" through the one
  registration function (`register_report`, the ledger beside the listed
  checkpoint holding the weights it measured, found by digest in the server's
  checkpoint list, never the report's recorded path). **Register is enabled
  for a kept report of this run that the page can confirm** (policy rev 16,
  K8-6r.2; `register_state`): it is off only for a request that is not an
  operator's, a run that kept no report, this run already registered, or this
  run's own report not built (its own reason) or not read by the listing (a
  store that could not be listed says so, and the run is registered from the
  command line by naming its report). **Another run never disables it**: how
  the runs of its condition group compare is a description ("runs under these
  conditions differ", in neutral words; amber with "— investigate" only when
  the group records deterministic algorithms on, so its runs were expected to
  agree; "comparison incomplete"), and investigation notes (who, when, what)
  are listed and added (gated); they record and change nothing. On a wide
  screen the note form stays at the panel's foot above Register, outside the
  scrolling report, so "Add note" is in view without scrolling.

**Import** (a drawer over Run and Result; D7): the files the operator placed
in the server's intake directory (`server_lists.list_intake_files`; a request
names a file and the server finds it in that listing again, refusing any other
name or a link that leaves the directory); version label, role (the version's
declared purpose) and the lost-terms policy; "Preview" runs the
importer (`prepare_import`, in this process on the CPU; it reads the
workspace's graph) and **shows only the exclusion reasons and their counts**
(no case id reaches the browser: the CLI preview on the machine and the
committed version keep them; a refused source is listed by case index,
`import_refusal`, and the job's output without the ids a refusal names,
`case_free`); "Commit" prepares again from the same inputs
and commits under the preview id shown, so changed inputs are refused
(`commit_import`). There is no upload control: Gradio's upload route is refused
in front of the mount.

**D8.** Every state-changing handler (import preview and commit, start,
cancel, register, add a note) and every read that opens checkpoints (the model
list, the registration's search for the checkpoint) asks D8's Gradio helper on
its own triggering request first, then calls the shared service in process;
the buttons a read-only request cannot use are disabled, with the lock and
D8's reason as visible text beside them, and show no lock for an operator (the
one gated-button rule, `chips.gated`, `GATED_LABELS`); the handler's check is
the protection. A read-only request
sees the cohort versions, their reports and states, the job's state and the
results; no checkpoint list.

**No absolute server path is shown** to any request: a path under the
registry, the workspaces root, the intake directory, an evidence directory or
the repository is shown under its name (``<registry>/versions/…``), any other
reads "a server path" (`redact`).

**Per browser session**: every listing, selection, preview and the job last
seen are `gr.State`; the job is polled on the shell's slow timer (outside the
tabs, ``queue=False``, every 5 s), doing work only while this page is shown.
The poll writes the job (its title row, Cancel and Cancel's reason in the Run
panel's foot, its details in the panel's body) and what the session saw of
the job (``job_state``, which nothing else writes), never a select, the
selection or what is drawn from it (Start and its reason included). Nothing
here is module state.
**Every write of the selection** (the checkpoint listing, the version's
reports, the report shown, and the table, the result, Start and its reason
drawn from them) runs in one queued group of one (`SELECTION_CONCURRENCY`):
page open, the workspace, the version, the run, the checkpoint, the
measurement, a report's row, ↻, the read after an act or an import, and the read after the poll saw
the job or the GPU line change (the poll hands that over through a hidden
component's ``.change``, naming what it saw before and what it sees now). Every choice
joins it, also one made while an earlier one waits. So each starts from the
selection the one before it wrote, in the order they were asked, and a late
answer never writes back a selection it started from over one chosen
meanwhile. The report of a run that has just ended is chosen once for that
run: only the poll's hand-off of that end chooses it (a run the poll saw busy,
or one that started and ended between two of its ticks), and the poll, the
one writer of what the session saw of the job, hands each change over once.

Module: src/webui/components/test_page.py
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import gradio as gr

from src.evaluation import protocols
from src.evaluation.caveats import (
    MEASUREMENT_NAMES,
    TASK_NAMES,
    label_words,
    mode_label,
    mode_note,
    version_words,
)
from src.webui.components import renderers as r
from src.webui.components.chips import LOCK, chips_on_change, gated
from src.webui.components.chips import operator_only as _operator_only
from src.webui.components.help import help_button, sentence_lines
from src.webui.components import model_list as ml
from src.webui.components.model_list import item_key
from src.webui.components.workspaces import (
    SERVER_PATH,
    binding_state,
    contents_reason,
    path_roots,
    redact_server as redact,
    setting_notice,
    version_row,
    workspace_binding,
)
from src.webui.operator_access import operator_access
from src.webui.shell import page_elem_id, picked, plain_html, unread_choice
from src.webui.style import SIDE, STACK

logger = logging.getLogger(__name__)

PAGE = "test"

#: The batch size a test run is given: the measurement's own default, which the
#: runner leaves to the CLI (D5: fixed by the server). A test pins it to
#: ``scripts/measure_scorer.py``'s default.
BATCH_SIZE_SHOWN = 32

#: Lines of a job's output shown (the runner keeps more).
LOG_LINES_SHOWN = 12
#: Unmappable terms the preview lists (the CLI preview's ten).
TERMS_SHOWN = 10
#: Problems of a refused import source the page lists (the CLI preview lists all).
PROBLEMS_SHOWN = 10
#: Report rows the list shows before "Show all N reports" (the running row and
#: the selected report are always shown).
ROWS_SHOWN = 8
#: The one queued group of every listener that writes the session's selection
#: (``models_state``, ``reports_state``, ``selected_report`` and the list, the
#: result, Start and its reason drawn from them): one event at a time, first
#: asked, first run (`create_test_page`).
SELECTION_CONCURRENCY = "shep-test-selection"

#: The investigation note's fields are never narrower (px): "Who" holds a
#: normal name whole.
NOTE_FIELD_MIN_WIDTH = 240

#: The lost-terms policy, as the select names it.
POLICY_CHOICES = [("keep case", "keep"),
                  ("exclude case", "exclude")]

#: Each measurement this build describes, as this page names its result: from
#: the one vocabulary (`caveats.MEASUREMENT_NAMES`), a task's measurement by
#: its task's name and "task" ("Disease task"), the KG link-prediction
#: measurement by its own words ("KG link prediction").
_RESULT_WORDS = {mode: f"{name} task" if mode in TASK_NAMES else name
                 for mode, name in MEASUREMENT_NAMES.items()}
#: The ⓘ beside a described measurement's label, holding its note
#: (`caveats.mode_note`).
_MODE_HELP = {mode: f"test-mode-{mode}" for mode in _RESULT_WORDS}
#: The job's statuses, as its pill shows them: never green for failed,
#: stopped or cancelled.
_PILLS = {
    "idle": ("Idle", "grey"), "running": ("Running", "accent"),
    "stopping": ("Cancelling…", "accent"), "completed": ("Completed", "ok"),
    "failed": ("Failed", "red"), "cancelled": ("Cancelled", "grey"),
}
_BUSY = ("running", "stopping")

#: Field help, behind the shared ⓘ (each title is its component's label).
#: The import source select's label: the folder by the name Settings › Data
#: locations gives it (`deployment_view.FIELD_WORDS`), where it is chosen.
INTAKE_LABEL = "File in the import folder (server)"
#: The run option that registers the report as the run ends (R-79): the act
#: is "Register" on the page, here and on the Result's "Register this report"
#: (`GATED_LABELS`); the option says when, the button which report.
RECORD_LABEL = "Register when the run ends"

FIELD_HELP = {
    "shep-test-workspace": (
        "Workspace",
        "One of the server's workspaces, shown by its dataset fingerprint.\n\n"
        "- Graph: what a data version is imported against and a run measures on\n"
        "- Models: its checkpoints are the ones offered\n"
        "- Folder: the workspaces folder, chosen in Settings › Data locations",
    ),
    "shep-test-version": (
        "Data version",
        "A version in the server's results registry, listed by kind and role.\n\n"
        "- A supplied cohort: source cases mapped to one graph's nodes (Import).\n"
        "- A KG edge set: one part of a model's edge split, with fixed pairs.\n"
        "- A generated split: the workspace's own patients.\n"
        "- Immutable: named by its cohort.json digest, measured on its own graph only.",
    ),
    "shep-test-model-run": (
        "Run",
        "The workspace's training runs with checkpoints the server lists, latest first.\n\n"
        "- Shown: by its name or its start time\n"
        "- Choosing a run: chooses its latest save\n"
        "- One testable checkpoint: chosen already\n"
        "- Operators only: listing opens every checkpoint",
    ),
    "shep-test-model": (
        "Checkpoint",
        "The chosen run's checkpoints, each by what the file is and how far it trained.\n\n"
        "- Under it: what its record says it is\n"
        "- No model spec: the file is listed, never measured\n"
        "- Measured: what its capabilities meet, its task or its LP head\n"
        "- Other bytes than its fingerprint: the run refuses before measuring",
    ),
    "shep-test-measurement": (
        "Measurement",
        "The measurement one Start runs and records, of those this build implements.\n\n"
        "- Offered: what the model's capabilities meet for the version's kind\n"
        "- Kinds: a task's on a cohort or split, KG link prediction on an edge set\n"
        "- A base: offered KG link prediction only\n"
        "- Choice: a lone option is chosen already, else choose one",
    ),
    "shep-test-record": (
        RECORD_LABEL,
        "Checked, the run's report is registered in the ledger beside its checkpoint.\n\n"
        "- When: as the run ends\n"
        "- Unchecked: kept unregistered in the registry's report store\n"
        "- Later: Register this report takes it whole, whatever the version's kind\n"
        "- Either way: every run keeps its report\n"
        "Ref: D5",
    ),
    "shep-test-deterministic": (
        "Deterministic execution",
        "Deterministic PyTorch and cuDNN algorithms, cuDNN benchmark off.\n\n"
        "- No deterministic version: the operation refuses the run, never falls back\n"
        "- Recorded: deterministic runs are a condition group of their own\n"
        "- Expected: its runs agree, and a difference asks for an investigation\n"
        "- Limit: guards known non-deterministic operations, certifies no number",
    ),
    "shep-test-intake": (
        INTAKE_LABEL,
        "The case files placed in the server's import folder, one chosen to import.\n\n"
        "- Folder: outside the repository, set in Settings › Data locations\n"
        "- Checked: the server finds the chosen name there again\n"
        "- Refused: any other name, a link leaving the folder included\n"
        "- Upload: none, the operator places each file on the server\n"
        "Ref: shepherd-cases/1 files (*.json)",
    ),
    "shep-test-label": (
        "Version label",
        "The version's name in the registry: lowercase letters, digits, '-' and '_'.\n\n"
        "Versions are immutable, so a label is used once.",
    ),
    "shep-test-role": (
        "Role",
        "The cohort version's declared purpose.\n\n"
        "For example mygene2, research_selection or institutional_acceptance.",
    ),
    "shep-test-partial": (
        "Lost terms",
        "What a case becomes when some, not all, of its terms fail to map.\n\n"
        "- Ranked on the terms that map, or excluded.\n"
        "- Recorded in the version.",
    ),
    "shep-test-note-who": (
        "Who",
        "Your name as you want it recorded.\n\n"
        "Nothing verifies it under this deployment mode.\n"
        "Ref: deployment mode M1",
    ),
    "shep-test-note-what": (
        "What was found",
        "What the investigation found.\n\n"
        "The note is kept in the report store with this condition group's reports.\n"
        "It deletes no report and changes nothing.",
    ),
}
#: Help a rendered ⓘ names. A described measurement's result note is the
#: governed caveat's own text (`caveats.mode_note`, shown on Models and
#: Results too), each sentence a line (`sentence_lines`), never reworded here.
TOPIC_HELP = {
    "test-preview-id": (
        "Preview id",
        "The first 16 hex digits of the SHA-256 of the would-be version manifest.\n\n"
        "- Covers: the source's bytes and file name, the graph's digests\n"
        "- Also: the dataset record, label, role, policy and every count\n"
        "- Commit: prepares again from the same inputs, refused unless the id matches\n"
        "- So: what is committed is what was previewed",
    ),
    **{_MODE_HELP[mode]: (f"{words} result", sentence_lines(mode_note(mode)))
       for mode, words in _RESULT_WORDS.items()},
    "test-batch": (
        "Batch size",
        "Cases scored at once, fixed by the job runner.\n"
        "- It is the measurement's default.\n"
        "- A test run has no reason to vary it.\n"
        "- It is recorded with the run.\n"
        "Ref: D5",
    ),
    "test-device": (
        "Device",
        "Where a test run measures: the deployment's device, with no fallback.\n"
        "- Set: on the server, not on this page\n"
        "- Recorded: with the run, so a CPU and a CUDA run are different measurements\n"
        "Ref: D5, SHEPHERD_DEVICE",
    ),
    "test-evidence": (
        "R10 evidence",
        "Where a supplied cohort's run looks for the inputs the checkpoint records.\n"
        "- Counts: their overlap with the cohort's cases\n"
        "- Looks in: the workspace, always, and Settings' training evidence folders\n"
        "- Digests: only bytes it records count, else a label is unverifiable, never 0\n"
        "- KG edge set, generated split: no overlap labels, so given no folder\n"
        "Ref: R10",
    ),
}


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------
e = r.e
#: A report row's time from its run id: the one `renderers.run_time` every
#: page names a run by.
run_time = r.run_time

#: A report store run id, read here only for its checkpoint's 12 hex digits
#: (`run_sha12`).
_RUN_ID = re.compile(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z-([0-9a-f]{12})-[0-9a-f]{8}$")


#: A string literal as Python's ``repr``/``ascii`` writes one: quoted, its
#: escapes kept, never a raw newline.
_PY_STR = r"""(?:'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*")"""
#: Where the importer's refusals name a case: ``cases[3] (case_id 'x'): …``
#: and ``case_id 'x' is not printable`` (`cohort_registry.parse_source`).
_CASE_IDS = (
    (re.compile(r" \(case_id " + _PY_STR + r"\)"), ""),
    (re.compile(r"\bcase_id " + _PY_STR + r"(?= is not printable)"), "case_id"),
)


def case_free(text: Any) -> str:
    """`text` with the case ids the importer's refusals name taken out (D7:
    the per-case detail stays on the machine): ``cases[3] (case_id 'x'):
    missing …`` reads ``cases[3]: missing …``. Nothing else of an importer
    refusal names a case: it describes a term or a disease by its type, never
    its value."""
    flat = str(text if text is not None else "")
    for pattern, replacement in _CASE_IDS:
        flat = pattern.sub(replacement, flat)
    return flat


def import_refusal(exc: BaseException) -> str:
    """What the page says of a refused import, with no case id: a source that
    does not fit the format lists its problems by case index only (at most
    `PROBLEMS_SHOWN`), and says where the case ids are listed."""
    from src.evaluation.cohort_registry import SOURCE_FORMAT, MalformedSource

    if not isinstance(exc, MalformedSource):
        return "Refused: " + said(case_free(exc), "Nothing was written.")
    problems = []
    for problem in exc.problems:
        text = case_free(problem)
        if re.search(r"\bcase_id ['\"]", text):  # a wording this filter does not know
            where = re.match(r"cases\[\d+\]", text)
            text = f"{where.group(0) if where else 'a case'}: a problem that names its case id"
        problems.append(text)
    count = len(problems)
    more = count - PROBLEMS_SHOWN
    listed = "; ".join(problems[:PROBLEMS_SHOWN]) + (f"; … {more} more" if more > 0 else "")
    return "Refused: " + said(
        f"The file is not a {SOURCE_FORMAT} source ({count} problem{'s' if count != 1 else ''}); "
        f"the import is refused as a whole: {listed}. The problems are listed by case index: "
        "case ids are not sent to the browser, and the command-line preview on the machine "
        "(scripts/import_cohort.py) lists each one with its case id",
        "Nothing was written.")


def _access(request: Any) -> Dict[str, Any]:
    """D8's answer for this request, as the page keeps it."""
    decision = operator_access(request)
    return {"allowed": decision.allowed, "condition": decision.condition,
            "reason": decision.reason}


def operator_only(access: Mapping[str, Any]) -> str:
    """The visible reason a control a read-only request cannot use is off
    (`chips.operator_only`, D8's short words; the header chip's ⓘ gives D8's
    whole reason), as a sentence."""
    return f"{_operator_only(access.get('condition'))}."


#: The page's gated buttons and their words, shown with no lock while the
#: operator may use them (`chips.gated`; the WebUI skill's §3). Preview and
#: Commit are in the import drawer, which opens only for a request D8 allowed,
#: so they show none; each handler checks its own request again.
GATED_LABELS = {"import": "Import a new version…", "start": "▶ Start test", "cancel": "Cancel",
                "reload": "↻", "note": "Add note", "register": "Register this report"}


def gated_update(name: str, page: Mapping[str, Any], *, interactive: bool = True) -> Any:
    """A gated button's update by the one rule (`chips.gated`), from D8's
    answer the page keeps for this session (`_access`)."""
    allowed = bool((page.get("access") or {}).get("allowed"))
    return gr.update(**gated(GATED_LABELS[name], allowed, interactive=interactive))


def _reason_html(text: Optional[str], *, kind: str = "") -> str:
    """A reason line: with `kind` "lock", led by the lock (a request D8
    refuses); "off", by the mark of a disabled control's reason
    (`renderers.DISABLED_MARK`, as every page leads it)."""
    if not text:
        return ""
    mark = (f'<span aria-hidden="true">{LOCK}</span>' if kind == "lock" else
            r.DISABLED_MARK if kind == "off" else "")
    return f'<div class="shep-reason shep-test-reason">{mark}<span>{e(text)}</span></div>'


def run_sha12(run_id: Any) -> Optional[str]:
    match = _RUN_ID.match(str(run_id or ""))
    return match.group(7) if match else None


def _clock(seconds: Any) -> str:
    try:
        total = int(seconds)
    except (TypeError, ValueError):
        return "—"
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def utc_time(iso: Any) -> Optional[str]:
    """"02:06:07 UTC" from the runner's start or end time (its clock's local
    time, with no zone), as the report rows name runs by their UTC time."""
    try:
        moment = datetime.fromisoformat(str(iso))
    except ValueError:
        return None
    # A time with no zone is the server's local time (`datetime.now()`).
    return r.clock_utc(moment)


def _pill(status: str) -> str:
    text, kind = _PILLS.get(status, (status, "grey"))
    return f'<span class="shep-tag shep-tag-{kind} shep-test-pill" data-status="{e(status)}">{e(text)}</span>'


def said(text: Any, tail: str) -> str:
    """`text` as a sentence, then `tail` ("Nothing was started.")."""
    body = " ".join(str(text or "").split()).rstrip()
    if body and not body.endswith((".", "!", "?")):
        body += "."
    return f"{body} {tail}".strip()


def _count(value: Any) -> str:
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else "—"


# ---------------------------------------------------------------------------
# Reading what the page shows (one session's data, plain JSON for gr.State)
# ---------------------------------------------------------------------------
def read_page(request: Any, *, environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The page's server lists for one session: D8's answer for this request,
    the workspaces (the dashboard's one reader), the serving workspace, the
    registry's cohort versions (their manifests only), the evidence setting and
    the device, and the workspaces folder's and the registry's notices
    (`workspaces.setting_notice`: None while each can be used). Opens no
    checkpoint."""
    from src.api.services.server_lists import list_cohort_versions
    from src.config import server_settings
    from src.webui.components.served_status import read_status, serving_workspace
    from src.webui.components.workspaces import read_workspaces, setting_notice

    roots = path_roots(environ)
    listed = read_workspaces(environ)
    status, _problem = read_status()
    serving = (serving_workspace(status, listed["rows"], directories=listed["directories"])
               if status is not None else None)
    registry = server_settings.cohort_registry(environ)
    # Each version's kept files checked against its record (remembered by
    # their state, so a page opened again hashes nothing unchanged): a version
    # whose files changed after its import is never shown as bound.
    versions = (list_cohort_versions(environ=environ, contents=True)
                if registry.available else None)
    evidence = server_settings.training_evidence_dirs(environ)
    try:
        from src.api.services.training_manager import deployment_device

        device: Dict[str, Any] = {"value": deployment_device() or "auto", "problem": None}
    except ValueError as exc:
        device = {"value": None, "problem": str(exc)}
    return {
        "access": _access(request),
        "roots": roots,
        "workspaces": listed["rows"],
        "directories": {name: str(path) for name, path in listed["directories"].items()},
        "workspaces_problem": listed["problem"],
        # Each setting's state in parts (`workspaces.setting_notice`), from the
        # same reads as the problems beside them (kept for the API-shaped
        # state): the Data panel says each once (`data_notice_html`), every
        # control it turns off a relation to it (`renderers.see_above`). None
        # while the setting can be used.
        "workspaces_notice": listed["notice"],
        "registry_notice": setting_notice(registry),
        "serving": serving,
        "registry_problem": None if registry.available else registry.problem,
        "versions": [version_row(item) for item in versions.items] if versions else [],
        "versions_problem": versions.problem if versions is not None else None,
        # The setting as read, which the runner's one choice by the chosen
        # version's kind is drawn from (`run_evidence`); its directories stay
        # in the session's server-side state and are shown by name only.
        "evidence": {"dirs": [str(d) for d in (evidence.value or ())],
                     "names": [Path(d).name for d in (evidence.value or ())],
                     "problem": evidence.problem, "note": evidence.note},
        "device": device,
    }


def read_models(workspace: Any, request: Any, *,
                environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The workspace's checkpoints for one session, D8 first: listing opens
    every checkpoint (SPEC_4 C3), so a refused request lists and opens nothing
    and says why."""
    from src.api.services.server_lists import list_checkpoints

    access = operator_access(request)
    if not access.allowed:
        return {"workspace": workspace, "items": [], "problem": access.refusal, "refused": True}
    if not workspace:
        return {"workspace": None, "items": [], "problem": None, "refused": False}
    listing = list_checkpoints(workspace, environ=environ)
    return {"workspace": workspace, "items": [asdict(i) for i in listing.items],
            "problem": listing.problem, "refused": False}


def read_reports(label: Any, *, environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """One cohort version's report store for one session: every run directory,
    newest first, with what it kept, whether its report is registered and how
    its condition group reads, as the server's one run listing gives them
    (`server_lists.version_item`, the listing ``GET /api/v1/cohorts`` shares,
    here with the problems' words for the operator); each run's group and
    record (the record the ledger would hold) from the same one scan of the
    store, found by the run's id for the bytes its state read (byte-identical
    runs are each their own); its own problem when the scan could not build
    it (``unbuilt``); the store's own listing problem, when the scan could not
    list it (``store_problem``); and the investigation notes kept with the
    group. The version is read once. Opens no checkpoint."""
    from src.api.services.record_states import read_report_store
    from src.api.services.server_lists import (
        ServerListRefused,
        resolve_cohort_version,
        stored_report_state,
        version_item,
    )
    from src.evaluation.report_store import read_investigation_notes

    empty = {"label": label, "runs": [], "problem": None, "notes": [], "notes_problems": [],
             "version_sha256": None, "store_problem": None}
    if not label:
        return empty
    try:
        version = resolve_cohort_version(label, environ=environ)
    except ServerListRefused as exc:
        return {**empty, "problem": str(exc)}
    store = read_report_store(version)
    item = version_item(version, report_state=stored_report_state, store=store, operator=True)
    notes, notes_problems = [], []
    try:
        notes, notes_problems = read_investigation_notes(version)
    except Exception as exc:  # noqa: BLE001
        notes_problems = [f"the investigation notes could not be read ({type(exc).__name__})"]
    # **The store's own listing problem** (Decision 10): with it, no run's
    # report can be confirmed here, and a run is registered from the command
    # line by naming its report. Another run's problem is never this.
    store_problem = store.problem or ("; ".join(
        problem for run, problem in store.scan.unplaced if run is None) or None)
    runs = []
    for report in item.reports:
        # By the run's id, for the digest of the bytes its state read: no group
        # (and so no note) for a report replaced since the scan read it; a
        # byte-identical report of another run is that run's own record.
        digest = (report.state or {}).get("report_sha256")
        placed = store.scan.run_key(report.run_id, digest) if store.scan is not None else None
        key, unbuilt = placed if placed is not None else (None, None)
        built = store.scan.reports.get(key, {}).get(report.run_id) if key is not None else None
        if built is not None and built[1].get("source_artifact_digest") != digest:
            built = None
        runs.append({
            "run_id": report.run_id, "when": run_time(report.run_id),
            "sha12": run_sha12(report.run_id), "state": report.state,
            "problem": report.problem, "key": list(key) if key is not None else None,
            "record": built[1] if built is not None else None,
            "contest": report.contest, "unbuilt": unbuilt,
        })
    return {"label": item.label, "version_sha256": item.manifest_sha256, "runs": runs,
            "problem": item.reports_problem, "notes": notes, "notes_problems": notes_problems,
            "store_problem": store_problem}


class LedgerPlace(NamedTuple):
    """Where this page files a report's record: the listed checkpoint file
    holding the weights it measured (`path`, its listing entry `item`), the
    listed workspaces looked in (`workspaces`, in order), and whether a
    workspace that had to be listed was left unread (`unread`)."""

    path: Optional[Path]
    item: Optional[Dict[str, Any]]
    workspaces: Tuple[str, ...]
    unread: bool = False


def _resolved(text: Any) -> Optional[str]:
    """An absolute path a report records, resolved; None for anything else."""
    if not isinstance(text, str) or not text or not Path(text).is_absolute():
        return None
    try:
        return str(Path(text).resolve())
    except (OSError, RuntimeError):  # a link loop: RuntimeError before 3.13
        return None


def ledger_checkpoint(manifest: Mapping[str, Any], page: Mapping[str, Any],
                      models: Mapping[str, Any], workspace: Any, *, read_others: bool = True,
                      cache: Optional[Dict[Any, Any]] = None,
                      environ: Optional[Mapping[str, str]] = None) -> LedgerPlace:
    """The listed checkpoint a report is registered beside from this page: a
    file the server lists holding the weights the report measured (its
    manifest's checkpoint digest, compared in full), in this order —

    1. **the file the run read** (the manifest's ``checkpoint_path``), when it
       is listed with those weights: the ledger the report store's list reads
       (`stored_report_state`), so the list and this page answer for one ledger;
    2. another listed copy in **the workspace the run measured** (the
       manifest's ``data_dir``);
    3. a listed copy in **the workspace this page shows** (`workspace`).

    Never the recorded path itself: the file is found in the server's list
    again (`resolve_checkpoint`). The session's listing (`models`) stands for
    its own workspace only; another workspace is listed now (listing opens its
    checkpoints: the caller is an operator's handler), or, with `read_others`
    False, not at all, and the place is `unread`. `cache` (one dict for one
    pass over many reports) keeps each workspace's files holding a digest."""
    from src.api.services.server_lists import (
        ServerListRefused,
        list_checkpoints,
        resolve_checkpoint,
    )

    manifest = manifest if isinstance(manifest, Mapping) else {}
    digests = manifest.get("artifact_digests")
    digest = digests.get("checkpoint") if isinstance(digests, Mapping) else None
    directories = page.get("directories") or {}
    recorded_dir = _resolved(manifest.get("data_dir"))
    recorded_file = _resolved(manifest.get("checkpoint_path"))
    order = [name for name, d in directories.items() if recorded_dir and d == recorded_dir]
    if workspace in directories and workspace not in order:
        order.append(workspace)
    if not r.is_digest(digest):
        return LedgerPlace(None, None, tuple(order))
    held = models.get("workspace") if not models.get("refused") else None
    found: Dict[Any, List[Tuple[Path, Dict[str, Any]]]] = cache if cache is not None else {}

    def hits(name: str) -> Optional[List[Tuple[Path, Dict[str, Any]]]]:
        """The listed files of `name` holding the weights; None when unread."""
        key = (name, digest)
        if key not in found:
            if name == held:
                items = models.get("items") or []
            elif read_others:
                items = [asdict(i) for i in list_checkpoints(name, environ=environ).items]
            else:
                return None
            found[key] = []
            for item in items:
                if item.get("sha256") != digest or item.get("problem"):
                    continue
                try:
                    path = resolve_checkpoint(name, item.get("architecture"), item.get("run"),
                                              item.get("filename"), environ=environ)
                except ServerListRefused:
                    continue
                found[key].append((path, dict(item)))
        return found[key]

    if recorded_file:
        for name in order:
            exact = next((hit for hit in hits(name) or ()
                          if _resolved(str(hit[0])) == recorded_file), None)
            if exact is not None:
                return LedgerPlace(exact[0], exact[1], tuple(order))
    for name in order:
        listed = hits(name)
        if listed is None:
            return LedgerPlace(None, None, tuple(order), unread=True)
        if listed:
            return LedgerPlace(listed[0][0], listed[0][1], tuple(order))
    return LedgerPlace(None, None, tuple(order))


def _needs_refining(run: Mapping[str, Any]) -> bool:
    state = run.get("state") or {}
    return state.get("kept") == "kept" and state.get("registered") == "cannot_be_checked"


def _refined_state(page: Mapping[str, Any], label: Any, run: Mapping[str, Any],
                   models: Mapping[str, Any], *, read_others: bool, path: Optional[Path] = None,
                   cache: Optional[Dict[Any, Any]] = None,
                   environ: Optional[Mapping[str, str]] = None) -> Optional[Dict[str, Any]]:
    """A kept report's registration asked again of the registration's one
    helper (`report_state`) with the checkpoint this page would register it
    beside (`ledger_checkpoint`); None when the list's answer stands (it is not
    "cannot be checked", or no listed file holds the weights, or finding it
    would list another workspace and `read_others` is False). `path`: the
    report as the store's listing names it (`resolve_report` finds it when it
    is not given). **One read, the helper's** (round 3 of the review fixes,
    its verification's A-3): the place is found from the report the helper
    reads bound to the digest the list showed (its ``place``), so the page
    reads no report itself, and a report changed since the list was read is
    never decoded to place a ledger: no place is asked for, and the list's
    answer stands."""
    from src.api.services.server_lists import resolve_report
    from src.evaluation.registration import report_state

    if not _needs_refining(run):
        return None
    state = run.get("state") or {}
    if path is None:
        path = resolve_report(label, run.get("run_id"), environ=environ)
    placed: List[LedgerPlace] = []

    def beside_listed(manifest: Mapping[str, Any]) -> Optional[Path]:
        place = ledger_checkpoint(manifest, page, models, models.get("workspace"),
                                  read_others=read_others, cache=cache, environ=environ)
        placed.append(place)
        return place.path if place.item is not None else None

    fresh = report_state(path, place=beside_listed, expected_sha256=state.get("report_sha256"))
    place = placed[0] if placed else None
    if place is None or place.path is None or place.item is None:
        return None
    beside = item_key(place.item)
    if place.item.get("workspace") != models.get("workspace"):
        beside += f" in workspace {place.item.get('workspace')}"
    return {**state, "kept": fresh.kept, "why_kept": fresh.why_kept,
            "registered": fresh.registered, "why_registered": fresh.why_registered,
            "sequence": fresh.sequence, "checked_beside": beside,
            "historical_overlap": list(fresh.historical_overlap),
            "replaced_after_registration": fresh.replaced_after_registration}


def _operator_listing(page: Mapping[str, Any], models: Mapping[str, Any]) -> bool:
    return bool((page.get("access") or {}).get("allowed")) and not models.get("refused") \
        and bool(models.get("workspace"))


def refine_registration(page: Mapping[str, Any], reports: Mapping[str, Any], run_id: Any,
                        models: Mapping[str, Any], *, request: Any = None,
                        environ: Optional[Mapping[str, str]] = None) -> Mapping[str, Any]:
    """The selected report's registration asked again of the registration's
    one helper, with the checkpoint file this page would register it beside:
    the list knows no checkpoint file, so for a report it finds no record for
    it can only say "cannot be checked" (a registration beside another copy of
    the weights is not ruled out). Given the listed checkpoint holding the
    weights the report measured (`ledger_checkpoint`, the order Register uses),
    the helper answers for that file's ledger: registered, or not registered
    there. Operators only (the listing opens checkpoints); otherwise, or when
    no listed file holds the weights, the list's answer stands. When the order
    needs another workspace's listing, it is read only if D8 allows `request`
    (the triggering request, asked now, as the model list asks it); with no
    request it is not read and the list's answer stands."""
    run = find(reports.get("runs") or [], "run_id", run_id)
    if run is None or not _operator_listing(page, models) or not _needs_refining(run):
        return reports
    read_others = request is not None and operator_access(request).allowed
    try:
        refined = _refined_state(page, reports.get("label"), run, models,
                                 read_others=read_others, environ=environ)
    except Exception:  # noqa: BLE001 - the list's own answer then stands
        logger.exception("The registration of %s could not be read again", run_id)
        return reports
    if refined is None:
        return reports
    runs = [dict(x, state=refined) if x.get("run_id") == run_id else x
            for x in reports.get("runs") or []]
    return {**reports, "runs": runs}


def refine_rows(page: Mapping[str, Any], reports: Mapping[str, Any], models: Mapping[str, Any],
                *, environ: Optional[Mapping[str, str]] = None) -> Mapping[str, Any]:
    """Every row's registration asked again, as `refine_registration` asks the
    selected one, from the session's checkpoint listing only (no other
    workspace is listed for a row: a row whose place needs one keeps the
    list's answer until it is selected). So an operator's list says "not
    registered" for each report Register would file beside a listed checkpoint
    and none is filed, and the answer does not change with the selection.
    Reads each such report and its ledger once; opens no checkpoint.
    Operators only; for anyone else the list's answer stands."""
    from src.api.services.server_lists import ServerListRefused, resolve_cohort_version
    from src.evaluation.cohort_registry import REPORT_FILE, RegistryRefused, list_report_runs

    if not _operator_listing(page, models) or not any(
            _needs_refining(run) for run in reports.get("runs") or []):
        return reports
    # The store listed once for every row (as `resolve_report` lists it for one).
    try:
        version = resolve_cohort_version(reports.get("label"), environ=environ)
        paths = {run.run_id: run.directory / REPORT_FILE for run in list_report_runs(version)
                 if run.problem is None and run.directory is not None}
    except (ServerListRefused, RegistryRefused):
        return reports
    runs, failed, cache = [], 0, {}
    for run in reports.get("runs") or []:
        refined = None
        if _needs_refining(run) and run.get("run_id") in paths:
            try:
                refined = _refined_state(page, reports.get("label"), run, models,
                                         read_others=False, path=paths[run["run_id"]],
                                         cache=cache, environ=environ)
            except Exception:  # noqa: BLE001 - that row keeps the list's answer
                failed += 1
        runs.append(dict(run, state=refined) if refined is not None else run)
    if failed:
        logger.warning("The registration of %d report(s) of %s could not be read again",
                       failed, reports.get("label"))
    return {**reports, "runs": runs}


def read_job(*, manager: Any = None) -> Dict[str, Any]:
    """The latest test run as the runner says it (status, inputs, outcome) and
    its output tail."""
    if manager is None:
        from src.api.services.training_manager import training_manager as manager
    status = manager.get_test_status()
    log = manager.get_log("test")
    status["log"] = list(log.get("lines") or [])[-LOG_LINES_SHOWN:]
    return status


def gpu_line(*, manager: Any = None, operator: bool = False) -> Tuple[str, str]:
    """``(kind, words)`` of the GPU chip, read now (the header's wording, with
    where to stop this service's inference when that holds it: `busy_words`)."""
    from src.webui.components.chips import busy_words, read_gpu_chip

    if manager is None:
        from src.api.services.training_manager import training_manager as manager
    chip = read_gpu_chip(manager, operator=operator)
    return chip.kind, busy_words(chip)


# ---------------------------------------------------------------------------
# Choices
# ---------------------------------------------------------------------------
def model_parents(models: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    """The session's checkpoint listing by its keys (`item_key`, the list's
    identity), as the shared Run → Checkpoint selects take it."""
    return {item_key(item): item for item in models.get("items") or []}


def _kind_words(version: Mapping[str, Any]) -> Optional[str]:
    """A version of another kind than a supplied cohort by its kind and role,
    as its option names it: "KG edge set (test)", "Generated split (val)"
    (`caveats.version_words` as a label, `caveats.label_words`), or None for
    a supplied cohort. Such a version has no source cases or ranked-eligible
    count, so it is never described in a supplied cohort's words."""
    kind = version.get("kind")
    if kind in (None, protocols.SUPPLIED):
        return None
    return label_words(version_words(kind, version.get("role")))


#: The order the version select lists the kinds in: supplied cohorts, then KG
#: edge sets, then generated splits, then any other kind; each kind's versions
#: in the registry's order.
_KIND_ORDER = (protocols.SUPPLIED, protocols.KG_EDGES, protocols.GENERATED)


def _by_kind(versions: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """`versions` in `_KIND_ORDER`, each kind's in the order given (a stable
    sort); a version whose kind was not read stays with the supplied ones,
    as it always listed."""
    order = {kind: n for n, kind in enumerate(_KIND_ORDER)}
    return sorted(versions, key=lambda version: order.get(
        version.get("kind") if version.get("kind") is not None else protocols.SUPPLIED,
        len(order)))


def version_choices(page: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """The version select's options, by kind (`_KIND_ORDER`) and role, each
    its label first and at most 40 characters with one separator (skill §8:
    `renderers.fit_options`): a supplied cohort by its role ("mygene2-2026-09
    · role test"), a KG edge set or a generated split by its kind and role
    ("kg-test · KG edge set (test)", "gen-val · Generated split (val)"); one
    that cannot be read says so. Its counts, in its own units, are the chosen
    version's details (`version_html`), as is whether it was mapped against
    this workspace's graph (its tag there, and Start's reason), never its
    option's text. The values are the labels, whatever the words."""
    entries = []
    for version in _by_kind(page.get("versions") or []):
        label = str(version.get("label"))
        if version.get("problem"):
            entries.append((label, "cannot be read", label))
            continue
        entries.append((label, _kind_words(version) or f"role {version.get('role')}", label))
    return r.fit_options(entries)


def find(rows: Sequence[Mapping[str, Any]], field: str, value: Any) -> Optional[Mapping[str, Any]]:
    return next((row for row in rows or [] if row.get(field) == value), None)


def find_model(models: Mapping[str, Any], key: Any) -> Optional[Mapping[str, Any]]:
    return next((i for i in models.get("items") or [] if item_key(i) == key), None)


def default_workspace(page: Mapping[str, Any], current: Any) -> Optional[str]:
    names = [w["name"] for w in page.get("workspaces") or []]
    if current in names:
        return current
    if page.get("serving") in names:
        return page["serving"]
    trainable = [w["name"] for w in page.get("workspaces") or [] if w.get("trainable")]
    return (trainable or names or [None])[0]


def default_version(page: Mapping[str, Any], binding: Optional[Mapping[str, Any]],
                    current: Any) -> Optional[str]:
    """The version chosen: `current` while it is listed; otherwise the first
    one mapped against this workspace's graph, else the first listed, in the
    order the select shows them (`_by_kind`), so the version chosen is the
    first option of its kind."""
    versions = _by_kind(page.get("versions") or [])
    labels = [v["label"] for v in versions]
    if current in labels:
        return current
    bound = [v["label"] for v in versions if binding_state(v, binding)[0] == "bound"]
    return (bound or labels or [None])[0]


# ---------------------------------------------------------------------------
# Rendering: Cohort
# ---------------------------------------------------------------------------
#: What the settings the Data panel's notice names mean on this page, said
#: once with them (`data_notice_html`; the text brief's §1: the state, its
#: consequence, then the next step): by how many it names.
TESTING_WAITS = {1: "Testing waits for it.", 2: "Testing waits for both."}
#: The controls a disabled reason names (`renderers.disabled_words`): a
#: relation to the notice (`renderers.see_above`: "Import disabled: no results
#: registry is set."), never the notice again (the skill's §9), or the page's
#: own reason ("Start disabled: choose a data version.").
IMPORT = "Import"
START = "Start"
REGISTER = "Register"
#: Where Start's relation points: the panel that shows the notice.
NOTICE_PANEL = "Data"
#: Under the Workspace select when the workspaces folder can be used and
#: holds none (the empty-folder state; the skill's §9).
NO_WORKSPACE_YET = "The workspaces folder holds no workspace yet."


def _notice(page: Mapping[str, Any], key: str) -> Optional[Mapping[str, Any]]:
    found = page.get(key)
    return found if isinstance(found, Mapping) else None


def data_notices(page: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """**The settings this page waits for**, as their notices
    (`read_page`: `workspaces.setting_notice`), the results registry first
    (the versions, an import, a run and its reports read it), then the
    workspaces folder (every workspace, its checkpoints and the GPU lock are
    in it): each said once on the screen, by the Data panel
    (`data_notice_html`); none while both can be used."""
    return [n for n in (_notice(page, "registry_notice"), _notice(page, "workspaces_notice"))
            if n is not None]


def data_notice_html(page: Mapping[str, Any]) -> str:
    """**The Data panel's one notice** (the skill's §9: each cause once per
    screen, where it empties a region): the registry's and the workspaces
    folder's states, one line each with its mark, what they mean here
    (`TESTING_WAITS`) and one next step when their places agree, the why and
    how under "Details ›" (`renderers.notices_block`). "" while both can be
    used."""
    notices = data_notices(page)
    if not notices:
        return ""
    return r.notices_block(notices, consequence=TESTING_WAITS[min(len(notices), 2)])


def relation_html(control: str, notice: Mapping[str, Any], *,
                  where: Optional[str] = None) -> str:
    """A control's one line on the notice it waits for (`renderers.see_above`)."""
    return r.see_above(control, notice, where=where)


def cohort_header_html(page: Mapping[str, Any]) -> str:
    """The Data panel's title and, under it, the notice of the settings the
    page waits for (`data_notice_html`), said once: the title's right side
    names where the registry comes from only while no notice shows."""
    notice = data_notice_html(page)
    if notice:
        return (f'<div class="shep-panel-hd"><h2>Data</h2></div>'
                f'<div class="shep-test-pad">{notice}</div>')
    if page.get("registry_problem"):
        # A page read without its notices: the registry's own words (`read_page`
        # always gives the notice with the problem).
        return ('<div class="shep-panel-hd"><h2>Data</h2></div><div class="shep-test-pad">'
                f'{_reason_html(redact(page["registry_problem"], page.get("roots") or []))}</div>')
    return ('<div class="shep-panel-hd"><h2>Data</h2><span class="shep-panel-right">'
            'registry: server setting</span></div>')


def workspace_html(page: Mapping[str, Any], workspace: Optional[str]) -> str:
    """The chosen workspace's identity under its select; nothing while the
    workspaces folder cannot be used (the panel's notice says it, once);
    otherwise why none is shown: the listing's problem, an empty folder
    (`NO_WORKSPACE_YET`) or none chosen."""
    row = find(page.get("workspaces") or [], "name", workspace)
    if row is None:
        if _notice(page, "workspaces_notice") is not None:
            return ""
        why = page.get("workspaces_problem") or (
            "No workspace is chosen." if page.get("workspaces") else NO_WORKSPACE_YET)
        return _reason_html(why)
    digits = r.distinguishing_digits(w.get("dataset_fingerprint")
                                     for w in page.get("workspaces") or [])
    note = ('<span class="shep-mark shep-mark-serve">serving reads this workspace</span>'
            if page.get("serving") == workspace else "")
    return (f'<div class="shep-test-ident">'
            f'{r.dataset_identity_html(row, digits=digits, name=False, marks=note)}</div>')


#: The import command a page names when the registry holds no KG edge set:
#: a KG edge set is made on the machine, from a checkpoint's recorded split
#: (`scripts/import_cohort.py --kind kg_edges`; the deployment guide §4.9),
#: never from the browser. Its role is the part asked, train, val or test.
KG_EDGES_IMPORT = "scripts/import_cohort.py --kind kg_edges --from-checkpoint <checkpoint>"


def kg_edges_empty_html(page: Mapping[str, Any]) -> str:
    """The empty state of the KG edge sets (the skill's §9): when the registry
    lists none, one line saying what a KG edge set is for, and under "Details ›"
    the command that makes one on the machine (`KG_EDGES_IMPORT`, a row of its
    own: never a command in a sentence, the text brief's §8) and the guide's
    section on the Ref line; "" when it lists one, or the registry is not
    available."""
    if page.get("registry_problem") or any(
            v.get("kind") == protocols.KG_EDGES for v in page.get("versions") or []):
        return ""
    command = r.summary_list([("Import one on the machine", f"<code>{e(KG_EDGES_IMPORT)}</code>")],
                             wide=("Import one on the machine",))
    return ('<div class="shep-test-kgempty"><p class="shep-sub">No KG edge set in the registry '
            'yet. KG link prediction measures one.</p>'
            f'<details class="shep-more"><summary>{e(r.DETAILS_WORD)}</summary>{command}'
            '<p class="shep-test-ref">Ref: deployment guide §4.9</p></details></div>')


def _version_counts(version: Mapping[str, Any]) -> str:
    """A KG edge set's or a generated split's counts, each in its own unit:
    the counts that say what it holds first, in words, then the unit its
    manifest records for every count, one click away ("Counts and units
    (N)"), with the value of each count the words above do not give, so no
    count is shown twice (skill §1.2)."""
    counts = version.get("counts") if isinstance(version.get("counts"), Mapping) else {}
    units = version.get("count_units") if isinstance(version.get("count_units"), Mapping) else {}
    if version.get("kind") == protocols.KG_EDGES:
        said: Tuple[str, ...] = ("pairs", "positive_edges", "message_passing_edges")
        lead = (f"{_count(counts.get('pairs'))} pairs from {_count(counts.get('positive_edges'))} "
                f"positive edges; message passing over "
                f"{_count(counts.get('message_passing_edges'))} edges")
    elif version.get("kind") == protocols.GENERATED:
        said = ("samples", "diseases", protocols.GENERATED_FIXTURE_COUNT)
        diseases = counts.get("diseases")
        fixture = counts.get(protocols.GENERATED_FIXTURE_COUNT)
        lead = (f"{_count(counts.get('samples'))} generated patients of {_count(diseases)} "
                f"disease{'' if diseases == 1 else 's'}; "
                + (f"{_count(fixture)} gene fixture rows" if protocols.GENERATED_FIXTURE_COUNT
                   in counts else "no gene fixture kept"))
    else:
        said = tuple(counts)
        lead = ", ".join(f"{name} {_count(value)}" for name, value in counts.items())
    rows = "".join(f"<div><code>{e(name)}</code>"
                   f"{'' if name in said else ' ' + e(_count(value))}: "
                   f"{e(units.get(name) or 'unit not recorded')}</div>"
                   for name, value in counts.items())
    return (f"<div>{e(lead)}</div>"
            f'<details class="shep-test-units"><summary>Counts and units ({len(counts)})'
            f"</summary>{rows}</details>")


def version_html(page: Mapping[str, Any], label: Optional[str],
                 binding: Optional[Mapping[str, Any]]) -> str:
    """The chosen version, in the one summary list
    (`renderers.summary_list`): its counts, its digest, and whether it was
    mapped against this workspace's graph, when it was imported and from
    what format. A supplied cohort by its case counts; a KG edge set or a
    generated split by its counts in its own units (`_version_counts`),
    never in a supplied cohort's words: its kind and role are its option's
    (`version_choices`), not repeated under it (skill §1.2; R-70), unless
    the option had to cut a long role, which then has a row. Under it, the
    KG edge sets' empty state when the registry holds none
    (`kg_edges_empty_html`)."""
    if page.get("registry_problem"):
        return ""
    empty = kg_edges_empty_html(page)
    version = find(page.get("versions") or [], "label", label)
    if version is None:
        why = page.get("versions_problem") or (
            "No data version is in the registry yet: import a cohort with “Import a new "
            "version…”." if not page.get("versions") else "No data version is chosen.")
        return _reason_html(redact(why, page.get("roots") or [])) + empty
    roots = page.get("roots") or []
    if version.get("problem"):
        return _reason_html("This version cannot be read: "
                            + redact(version["problem"], roots)) + empty
    counts = version.get("counts") or {}
    state, words = binding_state(version, binding)
    # The binding's state by the shared mark (the skill's §6): bound a filled
    # mark, with when its files were read; mapped against another graph a
    # condition with ▲ and, under the list, the refusal it brings
    # (`ANOTHER_GRAPH`); files changed after the import a refusal with ✕ and,
    # under the list, which file (`CHANGED_FILES`); not checked a hollow one.
    mark = r.status_mark({"bound": "ok", "another": "warn", "changed": "error",
                          "unknown": "neutral"}[state], words[:1].upper() + words[1:])
    checked = (version.get("contents") or {}).get("checked_at") if state == "bound" else None
    if isinstance(checked, str) and checked:
        mark += (' <span class="shep-sub">' + r.word_groups(
            "files checked", checked.replace("T", " ").replace("Z", " UTC")) + '</span>')
    excluded = counts.get("excluded") if isinstance(counts.get("excluded"), Mapping) else {}
    reasons = "; ".join(f"{str(k).replace('_', ' ')} {_count(v)}" for k, v in excluded.items())
    imported = str(version.get("imported_at") or "").replace("T", " ").replace("Z", " UTC")
    if _kind_words(version):
        rows = [("Holds", _version_counts(version))]
    else:
        # Its role is its option's (R-70, BACKLOG 51 (b)); a row of its own
        # only where the option had to cut it.
        option = {value: words for words, value in version_choices(page)}.get(label, "")
        role = version.get("role")
        rows = ([] if f"role {role}" in option else [("Role", r.ident_html(role))])
        rows.append(("Cases", e(f'{_count(counts.get("source_cases"))} source cases, '
                                f'{_count(counts.get("ranked_eligible"))} ranked-eligible'
                                + (f" ({reasons} excluded)" if reasons else ""))))
    fingerprint = r.fingerprint_value(version.get("manifest_sha256"), what="version fingerprint")
    rows.append(("Version fingerprint", fingerprint))
    rows.append(("Graph", mark))
    rows.append(("Imported", e(imported) if imported
                 else f'<span class="shep-muted">{e(r.NOT_RECORDED.lower())}</span>'))
    source_format = version.get("source_format")
    if source_format:
        rows.append(("Format", e(source_format)))
    refusal = (r.status_mark("error", ANOTHER_GRAPH) if state == "another" else
               r.status_mark("error", CHANGED_FILES.format(
                   reason=contents_reason(version) or "a kept file is not the bytes recorded"))
               if state == "changed" else "")
    reason = f'<div class="shep-reason shep-test-vd-why">{refusal}</div>' if refusal else ""
    return f'<div class="shep-test-vd">{r.summary_list(rows)}{reason}</div>{empty}'


#: **The words of a result's or a model's relation to what is chosen now**
#: (the skill's §1.2: a fact said once, elsewhere by its relation): the
#: chosen model, workspace and version are said by their selects and
#: details; a past result that measured the same says so, and one that did
#: not says its own in full (`result_identity_rows`), so it keeps its identity
#: after the selection changes.
SELECTED_MODEL = "the selected model"
SELECTED_WORKSPACE = "the selected workspace"
SELECTED_VERSION = "the selected version"

#: **Why a run on a version mapped against another graph is refused**, said
#: beside the version (`version_html`) and named by Start's reason
#: (`start_state`), which points here: the run would refuse it.
ANOTHER_GRAPH = ("Not measured on this workspace: the version was mapped against another "
                 "graph, so a run would refuse it. Choose a workspace with its graph, or "
                 "import the source again against this one.")

#: **Why a run on a version whose kept files changed is refused**, said beside
#: the version with the file it names (`workspaces.contents_reason`) and
#: named by Start's reason, which points here: a committed version's files
#: are never rewritten, so other bytes there are an edit after the import.
CHANGED_FILES = ("Not measured: {reason}. A version's files are never rewritten after its "
                 "import, so a run would refuse it. Import it again under a new label.")


def _row_chip(run: Mapping[str, Any]) -> str:
    """The row's one chip. In a group that records deterministic algorithms on,
    "▲ runs differ" (amber) asks for attention, since its runs were expected
    to agree; elsewhere "runs differ" and "comparison incomplete" (grey)
    describe the group after the run's own state. None blocks anything."""
    contest = run.get("contest") or {}
    state = run.get("state") or {}
    if contest.get("state") == "contested" and _deterministic_group(run):
        return r.warn_chip("runs differ")
    if run.get("problem"):
        return r.chip("not read", "grey")
    if state.get("kept") != "kept":
        return r.chip("no report kept" if state.get("kept") == "not_kept" else "kept: not checked",
                      "grey")
    if state.get("registered") == "registered":
        seq = state.get("sequence")
        return r.status_mark("ok", "Registered" + (f" #{seq}" if isinstance(seq, int) else ""),
                             as_chip=True)
    if contest.get("state") == "contested":
        return r.chip("runs differ", "grey")
    if contest.get("state") == "cannot_be_checked":
        return r.chip("comparison incomplete", "grey")
    return r.chip("not registered" if state.get("registered") == "not_registered"
                  else "registration not checked", "neutral")


def _why_not_kept(state: Mapping[str, Any]) -> str:
    """Why a run kept no report, in the registration helper's words."""
    return str(state.get("why_kept") or "the registration's helper gave no reason")


def measurement_words(mode: Optional[str]) -> str:
    """**A measurement as a run or its result names it** (the reports table's
    report and job rows, the job card): its mode in this page's words
    (`_RESULT_WORDS`: "Disease task", "KG link prediction"), a mode this
    version does not describe by its recorded code, "Mode not recorded" for
    none. The Measurement radio's options are the short `measurement_label`
    ("Disease"), a choice's words."""
    return _RESULT_WORDS.get(mode) or (
        f"Mode {mode}, not described by this version" if mode else "Mode not recorded")


def _row_mode(run: Mapping[str, Any]) -> str:
    """A report row's Measurement cell: its record's mode
    (`measurement_words`); Unknown for a run whose record was not built (no
    report kept, or one the scan could not read: the Result says why)."""
    record = run.get("record") if isinstance(run.get("record"), Mapping) else None
    if record is None:
        return r.status_mark("neutral", r.UNKNOWN)
    scoring = record.get("scoring") if isinstance(record.get("scoring"), Mapping) else {}
    mode = scoring.get("mode")
    return e(measurement_words(mode if isinstance(mode, str) and mode.strip() else None))


def _job_measurement(protocol: Any) -> str:
    """The measurement a test run was asked for (`inputs["protocol"]`, which
    can differ from the model's task), in the words its report's row will
    carry (`measurement_words`), or Not recorded."""
    if isinstance(protocol, str) and protocol.strip():
        return e(measurement_words(protocol))
    return r.status_mark("neutral", r.NOT_RECORDED)


def _listed_model(digest: Any, models: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The listed checkpoint holding the weights `digest` names (its full
    SHA-256, or the first 12 hex digits a report's run id gives), or None."""
    if not isinstance(digest, str) or not digest:
        return None
    return next((i for i in models.get("items") or []
                 if r.is_digest(i.get("sha256")) and i["sha256"].startswith(digest)), None)


def _is_selected(digest: Any, selection: Optional["Selection"]) -> bool:
    """Whether `digest` (whole, or its first 12 digits) names the chosen
    checkpoint's weights."""
    chosen = (selection.model or {}).get("sha256") if selection is not None else None
    return (isinstance(digest, str) and bool(digest) and r.is_digest(chosen)
            and chosen.startswith(digest))


def _other_model_html(item: Mapping[str, Any], words: Mapping[str, Tuple[str, str]], *,
                      whole: bool = False) -> str:
    """**A listed model other than the chosen one, named in full** (the
    skill's §1.2: a result's own identity, which can differ from the
    selection): what it is first (`renderers.model_kind_text`, decision F1:
    "Disease · frozen graph backbone" for the frozen-backbone control, never
    read as the full fine-tune), then its run and its file by the words of
    its Run → Checkpoint pair (`model_list.pair_words`: no two listed files
    share both, so two files of one run never read alike), the run whole
    in details (`whole`; `renderers.run_words`)."""
    run, file = words.get(ml.item_key(item)) or (r.model_name_line(item), "")
    if whole:
        run = r.run_words(item.get("architecture"), item.get("run"), fit=False)
    return (f'<div class="shep-sub">{e(r.model_kind_text(item))}</div><div>{e(run)}</div>'
            + (f'<div class="shep-sub">{e(file)}</div>' if file else ""))


def _model_cell(digest: Any, models: Mapping[str, Any], selection: Optional["Selection"],
                words: Mapping[str, Tuple[str, str]], *, prefix: bool = False) -> str:
    """A report row's Model cell: "the selected model" for the chosen
    checkpoint's weights (`SELECTED_MODEL`); another listed file by what it
    is, its run and its file (`_other_model_html`); weights the list does not
    hold by their fingerprint (`renderers.fingerprint_value`, 12 hex digits;
    with `prefix`, a digest the report's run id names by its first 12 digits
    alone, for a run whose record was not built)."""
    if _is_selected(digest, selection):
        return e(SELECTED_MODEL)
    item = _listed_model(digest, models)
    if item is not None:
        return _other_model_html(item, words)
    return r.fingerprint_value(digest, what="model fingerprint", copy=False, prefix=prefix)


def _row_names(runs: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Each report row's Report cell, by run id: its run's time. Rows that
    would read the same (two runs in a minute) get the seconds, and if those
    match too, the run id's own last part (`renderers.distinct_names`, as
    Models › Results names its rows). A row still selects and registers by
    its full run id (``data-key``)."""
    def name(run: Mapping[str, Any], level: int) -> str:
        run_id = str(run["run_id"])
        when = (run_time(run_id, seconds=True) if level else None) or run.get("when") or run_id
        return f"{when} · run …{run_id[-8:]}" if level > 1 else str(when)

    return r.distinct_names({str(run["run_id"]): run for run in runs}, name)


def _job_row(job: Mapping[str, Any], label: Any, run_ids: Sequence[str],
             models: Mapping[str, Any], selection: Optional["Selection"],
             words: Mapping[str, Tuple[str, str]]) -> str:
    """The latest test run of this version as a row no click selects: while it
    runs ("started …", Running), or when it ended with no report the store
    lists (cancelled before its run directory, failed, or kept nothing; the
    runner remembers it until the next start or a restart). "" otherwise."""
    status = job.get("status")
    inputs = job.get("inputs") or {}
    if inputs.get("version_label") != label:
        return ""
    outcome = job.get("outcome") if isinstance(job.get("outcome"), Mapping) else {}
    if status in _BUSY:
        started = utc_time(job.get("start_time"))
        what = "started " + started if started else "running"
        state = ("registers when it completes" if inputs.get("record")
                 else "kept, not registered, when it completes")
        css = "shep-test-rep-running"
    elif status in ("cancelled", "failed", "completed"):
        report_dir = outcome.get("report_dir")
        if outcome.get("kept") is True or (report_dir and Path(str(report_dir)).name in run_ids):
            return ""
        how = {"cancelled": "interrupted: cancelled", "failed": "failed"}.get(status, "ended")
        if status == "failed" and job.get("exit_meaning"):
            how += f" ({job['exit_meaning']})"
        kept = ("no report kept" if outcome.get("kept") is False
                else "whether a report was kept cannot be checked (see Run)")
        when = utc_time(job.get("end_time")) or utc_time(job.get("start_time"))
        what = "ended " + when if when else "ended"
        state = f"{how}; {kept}; the last test run on this server, not in the report store"
        css = "shep-test-rep-ended"
    else:
        return ""
    measured = _job_measurement(inputs.get("protocol"))
    model = _model_cell(str(inputs.get("checkpoint_sha256") or "") or None, models, selection,
                        words)
    return (f'<tr class="{css}" aria-disabled="true"><td>{e(what)}</td><td>{model}</td>'
            f'<td>{measured}</td><td>{_pill(str(status))}'
            f'<div class="shep-sub">{e(state)}</div></td></tr>')


#: The reports table's columns (the skill's §8: one or two words each).
REPORT_COLUMNS = ("Report", "Model", "Measurement", "State")


def reports_header_html(reports: Mapping[str, Any]) -> str:
    """The Result panel's title: the reports of the version chosen in Data
    (named there, so here by its relation, `SELECTED_VERSION`), with their
    count (the skill's §8: a table's title with its count) while it lists
    any; with none, the panel's empty state says why (`reports_empty_html`),
    once."""
    count = len(reports.get("runs") or [])
    right = (f'<span class="shep-panel-right">{count} of {e(SELECTED_VERSION)}, kept in the '
             "registry's report store</span>" if reports.get("label") and count else "")
    return f'<div class="shep-panel-hd"><h2>Reports</h2>{right}</div>'


#: **The Reports panel's empty state**, in its own words by cause (the
#: skill's §9: what will appear and the next step; in the flow, `.shep-empty`):
#: the registry not set, or set and not usable (the Data panel's notice says
#: which and where to choose it); no data version in it; none chosen; a
#: version no run has kept a report on.
REPORTS_NO_REGISTRY = "No report yet. Reports appear here once a results registry is set."
REPORTS_REGISTRY_REFUSED = ("No report yet. Reports appear here once the results registry can "
                            "be used.")
REPORTS_NO_VERSION = "No report yet. Reports appear here once a data version is imported."
REPORTS_CHOOSE_VERSION = "Choose a data version to see its reports."
REPORTS_NONE_YET = "No report on this version yet. A test run's report appears here once it ends."


def _empty(words: str) -> str:
    return f'<p class="shep-empty">{e(words)}</p>'


def reports_empty_html(page: Mapping[str, Any]) -> str:
    """The Reports panel's empty state while no data version is chosen, by its
    cause (`REPORTS_NO_REGISTRY` and the rest)."""
    registry = _notice(page, "registry_notice")
    if registry is not None or page.get("registry_problem"):
        words = (REPORTS_REGISTRY_REFUSED if registry is not None and registry.get("kind") == "error"
                 else REPORTS_NO_REGISTRY)
    elif not page.get("versions") and not page.get("versions_problem"):
        words = REPORTS_NO_VERSION
    else:
        words = REPORTS_CHOOSE_VERSION
    return f'<div class="shep-test-replist">{_empty(words)}</div>'


def reports_html(reports: Mapping[str, Any], selected: Optional[str], *,
                 models: Optional[Mapping[str, Any]] = None, job: Optional[Mapping[str, Any]] = None,
                 roots: Sequence[Tuple[str, str]] = (),
                 selection: Optional["Selection"] = None,
                 page: Optional[Mapping[str, Any]] = None) -> str:
    """**The version's report store as a compact single-select table** at the
    top of the Result panel (the skill's §4: many things, the same fields;
    §8: a row selects the report, shown under the table), newest first: each
    row its run's time (`_row_names`), its model by its relation to the
    chosen one (`_model_cell`), its measurement (`_row_mode`) and its state
    (`_row_chip`: kept, registered, how its condition group compares); what
    the state means for the chosen report, and why, is the Result's. A
    running test, or the last one that ended with nothing in the store, is a
    row of its own that no click selects. The newest `ROWS_SHOWN` rows are
    shown, with the running row, the selected report and "Show all N reports"
    (a version with hundreds of runs would otherwise push the result a page
    long down): the older rows are sent marked and shown by that control in
    the browser, so the choice survives the list's updates. A plain table's
    selected row is ``aria-current`` (the skill's §8). With no version chosen,
    the panel's empty state by its cause, from the `page` read
    (`reports_empty_html`); with a version that holds no report and no run row,
    its own (`REPORTS_NONE_YET`)."""
    label = reports.get("label")
    if not label:
        return reports_empty_html(page) if page is not None else ""
    models = models or {}
    runs = reports.get("runs") or []
    words = ml.pair_words(model_parents(models))
    rows = [_job_row(job or {}, label, [x["run_id"] for x in runs], models, selection, words)]
    hidden = 0
    names = _row_names(runs)
    for index, run in enumerate(runs):
        run_id = run["run_id"]
        on = run_id == selected
        old = index >= ROWS_SHOWN and not on
        hidden += old
        classes = " ".join(c for c, yes in (("shep-on", on), ("shep-test-rep-old", old)) if yes)
        # The record's model, else the first 12 digits its run id names.
        recorded = ((run.get("record") or {}).get("checkpoint_digest")
                    if isinstance(run.get("record"), Mapping) else None)
        digest = recorded or run.get("sha12")
        current = ' aria-current="true"' if on else ""
        rows.append(
            f'<tr class="{classes}" data-key="{e(run_id)}" tabindex="0"{current}>'
            f'<td class="shep-test-rep-name">{e(names[str(run_id)])}</td>'
            f'<td>{_model_cell(digest, models, selection, words, prefix=not recorded)}</td>'
            f'<td>{_row_mode(run)}</td><td>{_row_chip(run)}</td></tr>')
    entries = [row for row in rows if row]
    more = ('<div class="shep-test-more">'
            f'<button type="button" class="shep-test-more-all" data-more="all">Show all '
            f'{len(runs)} reports ›</button>'
            f'<span class="shep-sub shep-test-more-all">{hidden} older not shown</span>'
            f'<button type="button" class="shep-test-more-fewer" data-more="fewer">Show the newest '
            f'{ROWS_SHOWN} only</button></div>') if hidden else ""
    heads = "".join(f'<th scope="col">{e(head)}</th>' for head in REPORT_COLUMNS)
    table = ("" if not entries else
             '<div class="shep-tbl-wrap shep-test-repwrap"><table class="shep-tbl shep-test-reps" '
             f'aria-label="Reports of {e(label)}"><thead><tr>{heads}</tr></thead>'
             f'<tbody>{"".join(entries)}</tbody></table></div>')
    problem = reports.get("problem")
    # The store's problem is the list's reason when it could not be read: no
    # empty state claims there is no report then.
    notes = "" if entries or problem else _empty(REPORTS_NONE_YET)
    if problem:
        notes += f'<div class="shep-list-note">{e(redact(problem, roots))}</div>'
    return f'<div class="shep-test-replist">{table}{more}{notes}</div>'


# ---------------------------------------------------------------------------
# Rendering: Run
# ---------------------------------------------------------------------------
def run_header_html() -> str:
    """The Run panel's title. The workspace and the version are said once,
    by their selects in Data (the skill's §1.2), never again here."""
    return '<div class="shep-panel-hd"><h2>Run</h2></div>'


#: The empty state under the Run and Checkpoint selects when the workspace
#: lists no checkpoint (the skill's §9).
NO_CHECKPOINT = "This workspace has no checkpoint to test."
#: Start's reason while the workspace lists checkpoints and none is chosen:
#: the one place the page asks for the choice, naming its two selects
#: (NEW-test-1: it said "choose a model", a control the page no longer has,
#: under the selects' own "Choose the run, then the checkpoint to test.").
CHOOSE_A_CHECKPOINT = r.disabled_words(START, "choose the run, then the checkpoint to test.")


def testable(item: Mapping[str, Any]) -> bool:
    """Whether a listed checkpoint is one this page can be asked to test at
    all: its file was read (a digest, no problem) and its record was read
    (`measure_refusal` refuses a record refused or not read). Whether a
    measurement takes it on the chosen version is Start's to say."""
    return (not item.get("problem") and r.is_digest(item.get("sha256"))
            and item.get("record") not in ("refused", None))


def chosen_model(models: Mapping[str, Any], key: Any) -> Optional[str]:
    """**The checkpoint chosen** for the listing `models`: `key` while it is
    listed; otherwise, when the workspace lists exactly one checkpoint and it
    can be tested (`testable`), that one, by the shared list's one rule
    (`model_list.lone_key`, R-27: a lone candidate is no dead end; RJ-61b);
    otherwise none, until the operator picks a run. Of two or more, the page
    never picks one itself."""
    if find_model(models, key) is not None:
        return key
    return ml.lone_key(models.get("items") or [], testable)


def listing_why_html(page: Mapping[str, Any], models: Mapping[str, Any]) -> str:
    """**Why the Run and Checkpoint lists are empty** for a request D8
    refuses (listing the checkpoints opens them: an operator control), said
    once above the fields it turns off, the Run row with its locked ↻ and
    then Checkpoint: the gated-button rule's visible reason beside the ↻
    (`chips.gated`), placed as Settings places its panel's above the fields
    it turns off (T06-D). It was the Checkpoint field's note, 123 px under
    the ↻ and off screen with it at 390 × 844 (UI ledger V-15). Empty for
    an operator."""
    if not models.get("refused"):
        return ""
    return _reason_html("Not listed: listing the checkpoints opens them. "
                        + operator_only(page.get("access") or {}), kind="lock")


def model_html(page: Mapping[str, Any], models: Mapping[str, Any], key: Optional[str]) -> str:
    """The chosen model's details, by the shared renderers in the one summary
    list (`renderers.summary_list`), each a fact its options do not say: what
    it is (`renderers.model_kind_text`: "Disease · frozen graph backbone"
    for a task model whose graph backbone kept its base's weights), its
    file, the Trainer's validation figure, its model fingerprint (12 hex,
    copyable) and how its dataset stands to the chosen workspace's
    (`renderers.dataset_relation_html`: "the selected workspace's current
    dataset", which Data says once; which one and its fingerprint where it is
    another); or why there is no list. A listing D8 refused is said above the
    fields instead (`listing_why_html`), so nothing here."""
    roots = page.get("roots") or []
    if models.get("refused"):
        return ""
    if models.get("problem"):
        return _reason_html(redact(models["problem"], roots))
    item = find_model(models, key)
    if item is None:
        if not models.get("items"):
            return _reason_html(NO_CHECKPOINT if models.get("workspace") else "")
        # The instruction is Start's reason, said once (NEW-test-1, RJ-61b).
        return ""
    if item.get("problem") or not r.is_digest(item.get("sha256")):
        return _reason_html(f"{item.get('filename')}: {r.unreadable_words(item)} · cannot be "
                            f"tested ({redact(item.get('problem') or 'no digest was read', roots)})")
    rows = [("Model", e(r.model_kind_text(item))),
            ("File", r.ident_html(item.get("filename"))),
            ("Validation", e(r.validation_value(r.validation_figure(item),
                                                read=r.record_read(item)))),
            ("Model fingerprint", r.fingerprint_value(item["sha256"], what="model fingerprint")),
            ("Dataset", r.dataset_relation_html(item, page.get("workspaces") or []))]
    return ('<div class="shep-test-model">'
            + r.summary_list(rows, helps={"Model fingerprint": ("model-fingerprint",
                                                                "Model fingerprint")})
            + "</div>")


def run_evidence(page: Mapping[str, Any], version: Optional[Mapping[str, Any]]) -> Any:
    """The R10 evidence a run on the listed version `version` is given, by the
    job runner's one choice by its data kind
    (`training_manager.evidence_choice`), over the evidence setting this page
    read (`read_page`): a supplied cohort version takes the setting's
    directories and a refused setting refuses its Start; a KG edge set or a
    generated split takes none, the setting not required, and says why. Start's
    check (`start_state`) and the R10 evidence row (`evidence_html`) read it;
    the start itself asks the same choice of the setting read again
    (`start_test_run`). With no version chosen, or one whose kind was not
    read (a version that cannot be read, which Start refuses first), the
    supplied case: the row shows the setting and claims no kind."""
    from src.api.services.training_manager import evidence_choice
    from src.config.server_settings import ENV_TRAINING_EVIDENCE_DIRS, Setting

    read = page.get("evidence") or {}
    setting = Setting(ENV_TRAINING_EVIDENCE_DIRS,
                      None if read.get("problem") else tuple(
                          Path(d) for d in read.get("dirs") or read.get("names") or ()),
                      read.get("problem"), read.get("note"))
    kind = (version or {}).get("kind")
    return evidence_choice(protocols.SUPPLIED if kind is None else kind, setting)


def evidence_html(page: Mapping[str, Any], workspace: Optional[str],
                  version: Optional[Mapping[str, Any]] = None) -> str:
    """The R10 evidence row for a run on the chosen version (`run_evidence`,
    the runner's one choice by its data kind; research comparison Stage 7).

    For a KG edge set or a generated split: "Not applicable" in grey with a
    hollow ○, then why (R10 is a supplied cohort's; how that kind stands to
    a model's training), whatever the setting says, since a run on it is
    given no evidence directory. For a supplied cohort version (and before a
    version is chosen): the measured workspace (by its relation, "the
    selected workspace": its name is its select's), then the setting's
    directories, or, when the setting is refused, which turns Start off, a red
    ✕ with "Refused:" and why (BACKLOG 68's R10 row; the skill's §6). A
    directory the server cannot check stays in the list and does not block
    Start: the setting's note says which and why on a line of its own, in
    amber with ▲ (needs attention, does not block), redacted as the problem
    is. Each fact is a line of its own, in sentence case (L7 SW-LK7: one
    lowercase " · " chain, its second fact broken over two lines)."""
    choice = run_evidence(page, version)
    helps = {"R10 evidence": ("test-evidence", "R10 evidence")}

    def lines(*facts: str) -> str:
        return "".join(f"<div>{fact}</div>" for fact in facts if fact)

    if choice.not_applicable:
        return r.summary_list([("R10 evidence", lines(r.status_mark("neutral", r.NOT_APPLICABLE),
                                                      e(_sentence(choice.not_applicable))))],
                              helps=helps, classes=r.FIELD_ROWS)
    evidence = page.get("evidence") or {}
    names = evidence.get("names") or []
    roots = page.get("roots") or []
    # The workspace by its relation: its name is its select's, said once.
    measured = (f"{e(_sentence(SELECTED_WORKSPACE))} (measured, always)" if workspace
                else "The measured workspace (always)")
    note = ""
    if choice.problem:
        extra = r.status_mark("error", "Refused: " + redact(choice.problem, roots))
    elif names:
        extra = ", ".join(f"<code>{e(n)}</code>" for n in names) + " (the server's setting)"
        if choice.note:
            note = r.status_mark("warn", redact(choice.note, roots))
    else:
        extra = "No other evidence directory is set on the server"
    return r.summary_list([("R10 evidence", lines(measured, extra, note))], helps=helps,
                          classes=r.FIELD_ROWS)


def _sentence(words: str) -> str:
    """`words` as a line of its own starts: its first letter a capital."""
    return words[:1].upper() + words[1:]


def fixed_html(page: Mapping[str, Any]) -> str:
    """What the server fixes for a run (D5): the batch size and the device,
    each with its own ⓘ (R-79: one ⓘ beside Batch size explained both). The
    measurement is chosen beside them (the Measurement radio,
    `measurement_choices`)."""
    device = page.get("device") or {}
    if device.get("problem"):
        shown = f'<span class="shep-amber-t">{e(device["problem"])}</span>'
    else:
        value = device.get("value") or "auto"
        shown = (f"<code>{e(value)}</code>" + (" (requires CUDA)" if value == "auto" else ""))
    return r.summary_list([("Batch size", f"{BATCH_SIZE_SHOWN} · fixed by the server"),
                           ("Device", f"{shown} · fixed by the server")],
                          helps={"Batch size": ("test-batch", "Batch size"),
                                 "Device": ("test-device", "Device")},
                          classes=r.FIELD_ROWS)


def measurement_label(name: Any) -> str:
    """A protocol's words on this page (`caveats.MEASUREMENT_NAMES`: a task's
    measurement by its task's name, "KG link prediction"), or the name as it
    is for one with no words there."""
    return MEASUREMENT_NAMES.get(name, str(name)) if isinstance(name, str) else str(name)


def _kind(version: Optional[Mapping[str, Any]]) -> Any:
    """The listed version's data kind, which the measurements offered for it
    are judged by; None for no version chosen (the model's offer alone)."""
    return version.get("kind") if version is not None else None


def measurement_choices(item: Optional[Mapping[str, Any]],
                        version: Optional[Mapping[str, Any]] = None) -> List[Tuple[str, str]]:
    """The Measurement radio's options for the chosen listing entry `item`
    (None: no model chosen) on the chosen listed version `version` (None: none
    chosen): the protocols this build implements whose needs its listed
    capability description meets and that measure the version's data kind
    (`protocols.available`, in their order; research comparison Stage 7),
    each labelled by `measurement_label` (plain words, never the protocol's
    code), its value the protocol's name. A finetune on a supplied cohort is
    offered its task's measurement alone, so it is chosen; on a KG edge set,
    KG link prediction alone. Empty when none does; Start then says why
    (`measure_refusal`)."""
    if not item:
        return []
    names, _reason = protocols.available(item.get("capabilities"), _kind(version))
    return [(measurement_label(name), name) for name in names]


def chosen_measurement(choices: Sequence[Tuple[str, str]], value: Any) -> Optional[str]:
    """The Measurement radio's value for its `choices`: `value` while it is one
    of them, the only choice when there is one, else None (of several, none is
    chosen until the operator picks one)."""
    values = [v for _label, v in choices]
    if value in values:
        return value
    return values[0] if len(values) == 1 else None


#: This page's own reason beside a record's state (`measure_refusal`): a
#: record that is refused or not read offers no measurement.
NOT_MEASURED_HERE = "not measured here. Choose a model with a readable record"
#: Start's reason while the chosen model offers several measurements and none is chosen.
CHOOSE_A_MEASUREMENT = "choose a measurement"


def measure_refusal(item: Mapping[str, Any], version: Optional[Mapping[str, Any]],
                    protocol: Optional[str]) -> Optional[str]:
    """Why this page does not measure the listed checkpoint `item` on the
    listed cohort version `version` by the protocol `protocol` (None: none
    chosen), or None. **The one rule** behind Start's reason (`start_state`)
    and the start itself (`start_test_run`); the run judges again on the bytes
    it reads.

    A record refused (with the record reader's reason; a file without a model
    spec among them) or not read is said in the word the Model option leads
    with (`renderers.record_kind_text`), then this page's reason
    (`NOT_MEASURED_HERE`). Otherwise, with no protocol chosen, the reason no
    protocol takes the model and measures the version's kind
    (`protocols.available`, as the options are drawn) or `CHOOSE_A_MEASUREMENT`;
    with one, the protocol definition's reason (`protocols.refusal`: the listed
    capability description, then the version's listed kind and counts, which
    the importer writes for genes only for a source that lists them). A
    version whose causal genes are all
    unmapped passes here and the run refuses it."""
    record = item.get("record")
    if record in ("refused", None):
        why = (f" ({item.get('record_problem') or 'no reason was given'})"
               if record == "refused" else "")
        return f"{r.record_kind_text(item)}{why}: {NOT_MEASURED_HERE}"
    model = item.get("capabilities")
    if protocol is None:
        names, reason = protocols.available(model, _kind(version))
        return reason if not names else CHOOSE_A_MEASUREMENT
    return protocols.refusal(protocol, model, (version or {}).get("kind"),
                             (version or {}).get("counts"))


def import_reason_html(page: Mapping[str, Any], workspace: Optional[str]) -> str:
    """Why "Import a new version…" is off, under it: a request D8 refuses,
    the lock and D8's words; a setting the Data panel's notice names, the
    button's relation to it (`IMPORT_DISABLED`: the registry first, a version
    is written there; then the workspaces folder, a version is mapped against
    one of its workspaces); else no workspace chosen; "" while it can be
    used."""
    access = page.get("access") or {}
    if not access.get("allowed"):
        return _reason_html(operator_only(access), kind="lock")
    registry, folder = _notice(page, "registry_notice"), _notice(page, "workspaces_notice")
    if registry is not None:
        return relation_html(IMPORT, registry)
    if page.get("registry_problem"):
        return _reason_html(r.disabled_words(IMPORT, "no cohort registry is available."),
                            kind="off")
    if workspace:
        return ""
    if folder is not None:
        return relation_html(IMPORT, folder)
    return _reason_html(r.disabled_words(IMPORT, "choose a workspace to import against."
                                         if page.get("workspaces")
                                         else "no workspace to import against."), kind="off")


def start_notice(page: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """**The setting Start waits for**, as the notice the Data panel shows
    (`data_notices`, its first: the results registry, a run's reports are kept
    in it; then the workspaces folder, which holds every workspace and the GPU
    lock), or None when both can be used. Start's reason is its relation
    (`start_state`, `start_reason_html`)."""
    found = data_notices(page)
    return found[0] if found else None


def start_reason_html(page: Mapping[str, Any], allowed: bool, can_start: bool,
                      why: Optional[str]) -> str:
    """Start's reason as drawn under it: a request D8 refuses, its lock and
    D8's words; a setting the Data panel's notice names, Start's relation to
    it (`renderers.see_above`, the words `start_state` gives); else
    `start_state`'s reason."""
    notice = start_notice(page) if allowed and not can_start else None
    if notice is not None:
        return relation_html(START, notice, where=NOTICE_PANEL)
    return _reason_html(why, kind="lock" if not allowed else "off")


def start_state(page: Mapping[str, Any], models: Mapping[str, Any], workspace: Optional[str],
                label: Optional[str], key: Optional[str], binding: Optional[Mapping[str, Any]],
                gpu: Tuple[str, str], *, protocol: Optional[str] = None
                ) -> Tuple[bool, Optional[str]]:
    """Whether "Start test" can be used now for the measurement `protocol`
    (the Measurement radio's value; None: none chosen), and why not, in the
    order an operator fixes things: a setting the Data panel's notice names
    first, as Start's relation to it (`start_notice`, `relation_words`: its
    short words, never the notice again), then a busy or unread GPU lock
    before the choices, since no choice would enable Start while it lasts.
    Only a display: the handler checks D8 and the runner refuses again at
    start."""
    access = page.get("access") or {}
    roots = page.get("roots") or []
    if not access.get("allowed"):
        return False, operator_only(access)
    notice = start_notice(page)
    if notice is not None:
        return False, r.relation_words(START, notice, where=NOTICE_PANEL)
    off = partial(r.disabled_words, START)
    if page.get("registry_problem"):
        return False, off("no registry is available (see Data).")
    kind, words = gpu
    if kind == "busy":
        # The holder in the GPU chip's words, with where to stop this
        # service's inference when that holds it (`gpu_line`), as Training's
        # and Fine-tune's starts say it: whatever holds the lock, a job or
        # inference, Start waits for it.
        return False, off(f"{words}.")
    if kind == "unread":
        return False, off("the GPU lock cannot be read or used just now (see the GPU chip).")
    workspace_row = find(page.get("workspaces") or [], "name", workspace)
    if workspace_row is None:
        # An empty folder says so under the Workspace select (`NO_WORKSPACE_YET`).
        return False, off("choose a workspace." if page.get("workspaces")
                          else "no workspace is listed (see Data).")
    version = find(page.get("versions") or [], "label", label)
    if version is None:
        return False, off("choose a data version.")
    if version.get("problem"):
        return False, off("this data version cannot be read.")
    bound = binding_state(version, binding)[0]
    if bound == "changed":
        # Its refusal, naming the file, is said beside the version (`CHANGED_FILES`).
        return False, off("the data version's files changed after its import (see Data).")
    if bound == "another":
        # Its refusal is said beside the version (`ANOTHER_GRAPH`), once.
        return False, off("the data version is mapped against another graph (see Data).")
    item = find_model(models, key)
    if item is None:
        # Its empty state or its problem is said under the selects (`model_html`).
        return False, (CHOOSE_A_CHECKPOINT if models.get("items")
                       else off("no checkpoint to test (see Run)."))
    if item.get("problem") or not r.is_digest(item.get("sha256")):
        return False, off("this checkpoint's metadata is not readable.")
    refusal = measure_refusal(item, version, protocol)
    if refusal:
        return False, off(said(redact(refusal, roots), ""))
    # The runner's one evidence choice by the version's kind: the setting is
    # required only where it is given (a supplied cohort version).
    evidence = run_evidence(page, version)
    if evidence.problem:
        return False, off("the evidence directories setting is refused: "
                          + redact(evidence.problem, roots))
    if (page.get("device") or {}).get("problem"):
        return False, off(str(page["device"]["problem"]))
    return True, None


#: The job's one line while no test run was started since the server was
#: (R-81): in the Run panel's foot, where the job's state is said, muted; no
#: card, no pill, no Cancel.
NO_JOB_YET = "No test run since the server started."


def job_head_html(job: Mapping[str, Any]) -> str:
    """The job's title row, in the Run panel's foot under Start: the state and
    its time (Cancel sits beside it while a run is busy); while no run was
    started, one muted line (`NO_JOB_YET`)."""
    status = job.get("status") or "idle"
    if status == "idle":
        return f'<div class="shep-test-job-hd"><span class="shep-sub">{e(NO_JOB_YET)}</span></div>'
    head = f'<div class="shep-test-job-hd"><h3>Test run</h3>{_pill(status)}'
    if status in _BUSY:
        head += f'<span class="shep-sub">elapsed <b>{e(_clock(job.get("elapsed_seconds")))}</b></span>'
    else:
        ended = utc_time(job.get("end_time"))
        head += f'<span class="shep-sub">ended {e(ended or "—")}</span>'
    return head + "</div>"


def job_input_rows(job: Mapping[str, Any]) -> Tuple[List[Tuple[str, str]],
                                                    List[Tuple[str, str]]]:
    """**What a test run was given, as its inputs recorded them**, as rows
    of the one summary list (the skill's §4, §5: a fact a row, never a chain
    of " · "): what it measures, its version (label and fingerprint), its
    model (its fingerprint, the one digest value) and its measurement
    (`_job_measurement`), and its start; and the rows its card draws one
    click away (`job_body_html`'s one "Details ›"): its workspace, its
    register, determinism and device settings (the Result's keys for them)
    and its R10 evidence (by its own kind, the runner's choice: none for a
    KG edge set or a generated split). Drawn from the job alone, never from
    the selection (the card, `_job_outputs`, drawn again on each change the
    poll finds), so a run keeps its own identity while another model or
    version is chosen: these rows are the run's, which can differ from the
    selection's (the skill's §1.2)."""
    inputs = job.get("inputs") or {}
    started = utc_time(job.get("start_time"))
    evidence = inputs.get("evidence_dirs") or []
    label = inputs.get("version_label")
    version = (e(label) + " " + r.fingerprint_value(inputs.get("version_sha256"),
                                                     what="version fingerprint", copy=False)
               if label else r.status_mark("neutral", r.NOT_RECORDED))
    workspace = Path(str(inputs.get("workspace") or "")).name
    takes = protocols.takes_training_evidence(inputs.get("version_kind", protocols.SUPPLIED))
    rows = [("Version", version),
            ("Model", r.fingerprint_value(inputs.get("checkpoint_sha256"),
                                          what="model fingerprint", copy=False)),
            ("Measurement", _job_measurement(inputs.get("protocol")))]
    if started:
        rows.append(("Started", e(started)))
    more = [("Workspace", e(workspace) if workspace
             else r.status_mark("neutral", r.NOT_RECORDED)),
            ("Register", e(r.setting_words("record", bool(inputs.get("record"))))),
            ("Deterministic algorithms",
             e(r.setting_words("deterministic_algorithms", bool(inputs.get("deterministic"))))),
            ("Device", f"<code>{e(inputs.get('device') or 'auto')}</code>"),
            ("R10 evidence", e("the workspace" + (
                f" + {len(evidence)} director{'ies' if len(evidence) != 1 else 'y'}"
                if evidence else "")) if takes else r.status_mark("neutral", r.NOT_APPLICABLE))]
    return rows, more


#: What an ended run left, as the job card leads with it (B52-E: the result
#: first). A kept report is the Result's, which says its registration: the
#: card never says one of its own (NEW-test-3: one source for the
#: registration state, which can change after the run).
OUTCOME_KEPT = "Its report was kept: Result shows it and its registration."


def outcome_lead_html(outcome: Mapping[str, Any]) -> str:
    """What an ended run left, as the runner read it from disk when the run
    ended, said first on its card (B52-E): a kept report points to Result
    (`OUTCOME_KEPT`), which reads its registration from the ledger now; no
    report kept, or whether one was kept cannot be checked, in its state's
    own words (the skill's §6). Never its registration (NEW-test-3: the card
    said "Not registered" beside Result's "Registered" once a report was
    registered from the page); that moment's reading, with its paths, is the
    card's Details (`job_body_html`)."""
    kept = outcome.get("kept")
    if kept is True:
        return f'<div class="shep-test-outcome">{e(OUTCOME_KEPT)}</div>'
    return ('<div class="shep-test-outcome">'
            + r.status_mark("neutral", "No report kept" if kept is False
                            else "Kept: cannot be checked") + "</div>")


def job_body_html(job: Mapping[str, Any], roots: Sequence[Tuple[str, str]],
                  gpu: Tuple[str, str]) -> str:
    """The job's card, in the Run panel's scroll region after the choices,
    titled "Latest test run": **the result first** (B52-E), then what the
    run was given (`job_input_rows`), and **one disclosure** ("Details ›",
    `renderers.details_list`; the skill's §1.5) holding the rest: the inputs'
    other rows, what the runner read when the run ended (its summary, paths
    redacted), its exit status and its output's tail (redacted, case-free,
    D7). The result is what the run left once it ended (`outcome_lead_html`)
    and, for a failed run, its first error line; while it runs, the GPU line
    under the card. Its state is the title row's (`job_head_html`), in the
    panel's foot. While no run was started there is no card (R-81: the foot's
    one muted line says so)."""
    status = job.get("status") or "idle"
    if status == "idle":
        return ""
    lead = []
    outcome = job.get("outcome")
    if status not in _BUSY and isinstance(outcome, Mapping):
        lead.append(outcome_lead_html(outcome))
    if status == "failed" and job.get("error_message"):
        first = str(job["error_message"]).split("\n", 1)[0]
        lead.append(f'<div class="shep-test-fail">{e(redact(case_free(first), roots))}</div>')
    rows, more = job_input_rows(job)
    if status not in _BUSY and isinstance(outcome, Mapping) and outcome.get("summary"):
        more.append(("At its end", e(redact(outcome["summary"], roots))))
    if job.get("exit_meaning") and status not in _BUSY:
        more.append(("Exit status", e(f'{job.get("exit_code")}: {job["exit_meaning"]}')))
    # The output is the run's own; a refusal in it may name a case (D7).
    lines = [redact(case_free(line), roots) for line in job.get("log") or []]
    if lines:
        more.append((f"Output, last {len(lines)} lines",
                     f'<pre class="shep-test-log shep-mono">{e(chr(10).join(lines))}</pre>'))
    foot = ""
    _kind, words = gpu
    if status in _BUSY:
        foot = (f'<div class="shep-test-job-ft"><span class="shep-chip shep-chip-busy">'
                f'<span class="shep-dot" aria-hidden="true"></span>{e(words)}</span>'
                '<span class="shep-sub">Training start is refused while this runs.</span></div>')
    # Its Details stays as the reader left it while the card shows this job.
    return (f'<div class="shep-test-job"{r.keep_as(job.get("job_id"))}>'
            '<h3 class="shep-test-job-title">Latest test run</h3>'
            f'<div class="shep-test-job-bd">{"".join(lead)}'
            + r.summary_list(rows, classes="shep-test-ident-rows")
            + r.details_list(more, classes="shep-test-ident-rows")
            + f"</div>{foot}</div>")


def job_html(job: Mapping[str, Any], roots: Sequence[Tuple[str, str]],
             gpu: Tuple[str, str]) -> str:
    """The latest test run's title row and card together; the page draws
    them apart (`job_head_html` in the Run panel's foot, `job_body_html` in
    its scroll region)."""
    return job_head_html(job) + job_body_html(job, roots, gpu)


# ---------------------------------------------------------------------------
# Rendering: Result
# ---------------------------------------------------------------------------
class Selection(NamedTuple):
    """**What is chosen now**, which a report's identity is said against
    (`result_identity_rows`, `reports_html`): the chosen workspace's name and
    its verified dataset record's digest (`workspace_binding`), the chosen
    checkpoint's listing entry (None while none is chosen, or for a request
    that is not shown the list) and the chosen version's manifest digest."""

    workspace: Optional[str]
    record_sha256: Optional[str]
    model: Optional[Mapping[str, Any]]
    version_sha256: Optional[str]


def selection_of(page: Mapping[str, Any], models: Mapping[str, Any],
                 binding: Optional[Mapping[str, Any]], workspace: Any, label: Any,
                 key: Any) -> Selection:
    """The page's selection as `Selection` holds it, from the session's
    listings (nothing is read again)."""
    version = find(page.get("versions") or [], "label", label) or {}
    return Selection(workspace if isinstance(workspace, str) else None,
                     (binding or {}).get("record_sha256"), find_model(models, key),
                     version.get("manifest_sha256"))


#: **A result's dataset by the one relation its record supports**
#: (NEW-test-5): a report records the dataset record it measured
#: (``dataset_record.sha256``), never the workspace it was read from, so a
#: result of the selected workspace's dataset record is said to be that, not
#: "the selected workspace's"; two workspaces built on one dataset record
#: hold the same record, and which of them was read is not recorded.
SELECTED_DATASET = "the selected workspace's dataset record"


def result_dataset_html(record: Mapping[str, Any], view: Mapping[str, Any],
                        selection: Selection, workspaces: Sequence[Mapping[str, Any]]) -> str:
    """The Result's Dataset row: **the relation exactly as the record holds
    it** (NEW-test-5). Its dataset record is the selected workspace's:
    `SELECTED_DATASET`, and every other listed workspace that holds the same
    record named under it, since the report does not say which one it read;
    another: the listed workspaces holding it by name ("demo-copy's dataset
    record") or that none does, then its origin as the one reader reads it
    (the view's ``dataset_origin``, `renderers.dataset_origin_html`; the
    selected workspace's origin is its line in Data); none recorded: so."""
    held = record.get("dataset_record") if isinstance(record.get("dataset_record"),
                                                      Mapping) else {}
    dataset = held.get("sha256")
    if not r.is_digest(dataset):
        return r.status_mark("neutral", r.NOT_RECORDED)
    names = [str(w.get("name")) for w in workspaces
             if (w.get("dataset_record") or {}).get("sha256") == dataset]
    if dataset == selection.record_sha256:
        others = [name for name in names if name != selection.workspace]
        said = e(SELECTED_DATASET)
        if others:
            said += ('<div class="shep-sub">' + e(
                f"{', '.join(others)} hold{'s' if len(others) == 1 else ''} the same record; "
                "the report does not record which workspace was read") + "</div>")
        return said
    who = (e(f"{names[0]}'s dataset record") if len(names) == 1
           else e(f"the dataset record of {', '.join(names[:-1])} and {names[-1]}") if names
           else r.status_mark("neutral", "No listed workspace holds its dataset record"))
    return f'{who}<div class="shep-sub">{r.dataset_origin_html(view.get("dataset_origin"))}</div>'


def result_identity_rows(record: Mapping[str, Any], view: Mapping[str, Any],
                         selection: Optional[Selection], models: Mapping[str, Any],
                         workspaces: Sequence[Mapping[str, Any]]) -> List[Tuple[str, str]]:
    """**What a result is of, by its relation to what is chosen now** (the
    skill's §1.2: a fact once, elsewhere by its relation; repeated only where
    the two can differ): its dataset (the dataset record its report measured,
    `result_dataset_html`), its model (`checkpoint_digest`) and its version
    (the version manifest's digest). Each that is what is chosen reads so,
    "the selected workspace's dataset record", "the selected model", "the
    selected version"; one that is not reads in full: the listed workspaces
    holding its dataset record, its model by what it is, its run (whole) and
    its file (`_other_model_html`) and its fingerprint, its version's label,
    role and fingerprint. So a past result keeps its own identity when the
    selection changes: it is drawn again from the record each time, never
    from the selects. Its run id, which names its model's fingerprint again,
    is one click away (`_record_html`; R-70, BACKLOG 51 (d))."""
    selection = selection or Selection(None, None, None, None)
    digest = record.get("checkpoint_digest")
    if _is_selected(digest, selection):
        model = e(SELECTED_MODEL)
    else:
        item = _listed_model(digest, models) if r.is_digest(digest) else None
        fingerprint = r.fingerprint_value(digest, what="model fingerprint")
        model = (_other_model_html(item, ml.pair_words(model_parents(models)), whole=True)
                 + f"<div>{fingerprint}</div>" if item is not None else fingerprint)
    cohort = view["cohort"]
    if cohort["version_manifest_sha256"] == selection.version_sha256:
        version = e(SELECTED_VERSION)
    else:
        kind = cohort["kind"]
        role = (f'role {r.ident_html(cohort["role"])}' if kind == "supplied"
                else e(version_words(kind, cohort["role"])))
        version = (f'{e(cohort["label"])} ({role})<div>'
                   + r.fingerprint_value(cohort["version_manifest_sha256"],
                                         what="version fingerprint") + "</div>")
    return [("Dataset", result_dataset_html(record, view, selection, workspaces)),
            ("Model", model), ("Version", version)]


def result_html(reports: Mapping[str, Any], run_id: Optional[str],
                roots: Sequence[Tuple[str, str]], *, job: Optional[Mapping[str, Any]] = None,
                notes: Optional[List[Mapping[str, Any]]] = None,
                notes_problems: Sequence[str] = (), selection: Optional[Selection] = None,
                models: Optional[Mapping[str, Any]] = None,
                workspaces: Sequence[Mapping[str, Any]] = ()) -> str:
    """The report selected in the table, under it, as the record the ledger
    would hold: aggregates and states only, never a case or a path. What it
    is of is said by its relation to what is chosen now (`selection`;
    `result_identity_rows`), its own run id once."""
    run = find(reports.get("runs") or [], "run_id", run_id)
    if run is None:
        return ('<div class="shep-test-pad"><p class="shep-muted">Select a report in the table '
                'to see its result.</p></div>' if reports.get("runs") else "")
    head = '<div class="shep-test-res-hd"><h3>Result</h3></div>'
    state = run.get("state") or {}
    record = run.get("record")
    chips = []
    job_outcome = (job or {}).get("outcome") or {}
    if job_outcome.get("report_dir") and Path(str(job_outcome["report_dir"])).name == run_id:
        chips.append(_pill(str((job or {}).get("status"))))
    # A kept report is the normal state, which the record under says by being
    # there: no chip (R-81). Only a report not kept, or not checked, is one.
    if state.get("kept") != "kept":
        chips.append(r.chip("No report kept" if state.get("kept") == "not_kept"
                            else "Kept: cannot be checked", "grey"))
    elif record is None:
        # The registration is the record's Registration row (`_registration_html`),
        # said once; with no record shown, here.
        seq = state.get("sequence")
        chips.append(r.status_mark("ok", "Registered" + (f" record {seq}" if isinstance(seq, int)
                                                         else ""), as_chip=True)
                     if state.get("registered") == "registered" else
                     r.chip("Not registered" if state.get("registered") == "not_registered"
                            else "Registration: cannot be checked from the list", "neutral"))
    contest = run.get("contest") or {}
    if contest.get("state") == "contested":
        chips.append(r.warn_chip("Runs differ — investigate") if _deterministic_group(run)
                     else r.chip("Runs differ", "grey"))
    parts = [f'<div class="shep-test-chips">{"".join(chips)}</div>'] if chips else []
    if run.get("problem"):
        parts.append(_reason_html(redact(run["problem"], roots)))
    if record is None:
        parts.append(r.summary_list([("Report", r.ident_html(run_id))],
                                    classes="shep-test-ident-rows"))
        if state.get("kept") == "kept":
            why = run.get("unbuilt") or reports.get("store_problem") or (
                "the report store's listing did not read it")
            parts.append(_reason_html("This report cannot be shown or compared: "
                                      + redact(why, roots)))
        else:
            # The helper's own reason (K8-6e.1): a report.json that is there
            # but is not a report, or holds other bytes, is never "not there".
            parts.append(_reason_html(
                "No report was kept in this run directory: "
                + redact(_why_not_kept(state), roots) + ". report.json is the run's commit "
                "point, which an interrupted, cancelled or refused run never writes; nothing "
                "here can be registered."))
    else:
        parts.append(_record_html(record, run, state, selection=selection, models=models or {},
                                  workspaces=workspaces))
    parts.append(_registration_reason_html(state, roots))
    parts.append(block_html(run, roots, notes or [], notes_problems))
    # Its disclosures stay as the reader left them while it shows this report.
    return (f'<div class="shep-test-pad shep-test-result"{r.keep_as(run_id)}>{head}'
            f'{"".join(parts)}</div>')


def _registration_reason_html(state: Mapping[str, Any], roots: Sequence[Tuple[str, str]]) -> str:
    """Why this run's registration reads as it does, where "not registered"
    or "cannot be checked" alone would mislead (K8-6r.2): a record names this
    run with other bytes (its report was replaced after it was registered, so
    a registration is refused), or the ledger beside the listed checkpoint was
    read and cannot tell (this run's identity record damaged, the run-identity
    boundary lost, a record whose sequence cannot be read). The registration's
    own words (`report_state`), redacted."""
    why = state.get("why_registered")
    if state.get("kept") != "kept" or not why:
        return ""
    if state.get("registered") == "not_registered" and state.get("replaced_after_registration"):
        return _reason_html("Not registered: " + redact(why, roots))
    if state.get("registered") == "cannot_be_checked" and state.get("checked_beside"):
        return _reason_html("Registration cannot be checked: " + redact(why, roots))
    return ""


def _registration_html(state: Mapping[str, Any]) -> str:
    """A kept report's registration as the store's state says it (the
    record's Registration row, `renderers.record_rows`): registered as which
    record, not registered, or cannot be checked, and which ledger was read;
    a possible overlap with a record merged before each run had its own
    record (K8-6r.2), never said as registered."""
    registered = state.get("registered")
    seq = state.get("sequence")
    beside = state.get("checked_beside")
    where = f" (the ledger beside {e(beside)} was read)" if beside else ""
    merged = [n for n in state.get("historical_overlap") or [] if isinstance(n, int)]
    overlap = (f"; possible overlap with record{'s' if len(merged) != 1 else ''} "
               f"{', '.join(str(n) for n in merged)}, merged before each run had its own record "
               "(not recorded whether this run is one of them); registering adds this run's "
               "own record" if merged and registered != "registered" else "")
    return ((f"registered as record {seq}{where}" if registered == "registered"
             and isinstance(seq, int)
             else f"registered{where}" if registered == "registered"
             else f"not registered{where}" if registered == "not_registered"
             else f"cannot be checked{where}" if beside
             else "none found beside the checkpoint the run read, or its ledger cannot be "
                  "read (a registration beside another copy of the weights is looked for only "
                  "with the checkpoint list, an operator control)") + e(overlap))


def _record_html(record: Mapping[str, Any], run: Mapping[str, Any],
                 state: Mapping[str, Any], *, selection: Optional[Selection] = None,
                 models: Optional[Mapping[str, Any]] = None,
                 workspaces: Sequence[Mapping[str, Any]] = ()) -> str:
    """The selected report's record (`build_record`'s, through the store's
    scan) as the one reader reads it (research comparison Stage 8b): its data
    kind's result view (`record_views.result_view`, the one the results
    service reads every kept report by, registered or not), **drawn by the
    one record renderer Models draws its records by** (`renderers.record_html`,
    R-82: the same rows in the same order). Around it, what only this page
    knows: what the result is of, by its relation to what is chosen now
    (`result_identity_rows`, its run id once), its measurement's label with
    its ⓘ, and its registration as the store's state says it. The
    measurement's device is the record's recorded setting
    (`record_views.recorded_settings`, as the service reads it). Only the
    metric values are the record's own, by the names the view read (an
    integer stays an integer, as the service keeps them, `results._metrics`).
    The dataset origin is said only where the report's dataset record is not
    the selected workspace's, whose origin the workspace line says (the
    skill's §1.2). A record the reader cannot read is not shown as a result:
    its reason, naming fields and never values, as Models and the results
    service list it."""
    from src.api.services.record_views import (
        recorded_settings,
        result_view,
        unreadable_reason,
    )

    try:
        view = result_view(dict(record)).model_dump(mode="json")
    except ValueError as exc:
        return _reason_html("This report's record cannot be read as a result, so it is not "
                            "shown as one (Models › Results lists it with this reason): "
                            + unreadable_reason(exc))
    metrics = {name: record["metrics"][name] for name in view["metrics"]}
    view["device"] = recorded_settings(record)["regime"].get("device")
    # Labelled by the record's own mode, as recorded, never by another's.
    mode = view["mode"]
    identity_rows = result_identity_rows(record, view, selection, models or {}, workspaces)
    identity = r.summary_list(identity_rows, classes="shep-test-ident-rows")
    topic = _MODE_HELP.get(mode)
    mode_help = help_button(topic, TOPIC_HELP[topic][0]) if topic else ""
    # The dataset is the identity's row (its origin there when it is not the
    # selected workspace's record); the version's counts are Data's when it
    # is the selected version (R-70, BACKLOG 51 (c)); the run id, which names
    # the model's fingerprint again, is one click away (BACKLOG 51 (d)).
    selected_version = dict(identity_rows).get("Version") == e(SELECTED_VERSION)
    body = r.record_html(view, metrics, registration=_registration_html(state),
                         version_counts=not selected_version,
                         details=[("Run id", r.ident_html(run.get("run_id")))])
    return (f'{identity}<div class="shep-rec-label">{e(mode_label(mode))}{mode_help}</div>'
            f'{body}')


def _deterministic_group(run: Mapping[str, Any]) -> Optional[bool]:
    """Whether the run's condition group records deterministic algorithms on
    (True) or off (False): the setting is in the group's semantics, so every
    run of it shares it. None when the run's record does not say (no record
    read, or no boolean there): nothing is assumed, and None reads as off."""
    record = run.get("record") if isinstance(run.get("record"), Mapping) else {}
    regime = record.get("regime") if isinstance(record.get("regime"), Mapping) else {}
    value = regime.get("deterministic_algorithms")
    return value if type(value) is bool else None


def block_html(run: Mapping[str, Any], roots: Sequence[Tuple[str, str]],
               notes: Sequence[Mapping[str, Any]], notes_problems: Sequence[str] = ()) -> str:
    """How the report's condition group compares in the store, and the
    investigation notes kept with it: who, when, what. **A description, never
    a block** (policy rev 16): every run is kept as recorded, and nothing here
    disables Register. Runs that differ are described in neutral words; only a
    group that records deterministic algorithms on, whose runs were expected
    to agree, is flagged for investigation (amber, ▲). Notes that cannot be
    read are said, so "no note is kept" is never shown when one may be."""
    contest = run.get("contest") or {}
    key = run.get("key")
    parts = []
    if contest.get("state") == "contested":
        others = ", ".join(run_time(x) or x for x in contest.get("disagreeing_runs") or ())
        differing = e(", ".join(contest.get("differing") or ()))
        deterministic = _deterministic_group(run)
        if deterministic:
            parts.append(
                '<div class="shep-notice shep-notice-amber shep-test-block"><b>'
                + r.status_mark("warn", "Runs under these conditions differ — investigate.")
                + '</b> They differ '
                f'on: {differing} (runs {e(others)}). Deterministic algorithms were on, so these '
                "runs were expected to agree. Every run is kept as recorded and each can be "
                "registered; compare their environment facts first. An investigation note "
                "records what was found.</div>")
        else:
            # "Without deterministic algorithms" only when the record says so:
            # a determinism not recorded is not asserted either way.
            parts.append(
                '<div class="shep-reason shep-reason-grey shep-test-block"><span><b>Runs under '
                f'these conditions differ</b> on: {differing} (runs {e(others)}). '
                + ("Without deterministic algorithms, runs can differ; every"
                   if deterministic is False else "Every")
                + " run is kept as recorded and each can be registered. An investigation note "
                "records what was looked at.</span></div>")
    elif contest.get("state") == "cannot_be_checked":
        problems = "; ".join(redact(p, roots) for p in contest.get("problems") or ())
        if key:
            parts.append(_reason_html(f"Comparison incomplete: {problems}. This describes the "
                                      "group only; it does not stop a registration."))
        else:
            # This run's own report could not be placed in a group: its own
            # reason, which marks no other run (K8-6r.2).
            parts.append(_reason_html(f"This run's report cannot be compared: {problems}. It "
                                      "marks no other run."))
    if key:
        from src.evaluation.sidecar import KEY_FIELDS

        wanted = dict(zip(KEY_FIELDS, key, strict=False))
        mine = [n for n in notes if n.get("key") == wanted]
        shown = bool(mine) or contest.get("state") in ("contested", "cannot_be_checked")
        # The one rendering of a group's notes (Models draws them so too, B40).
        parts.append(r.investigation_notes_html(
            mine, problems=[redact(p, roots) for p in notes_problems] if shown else (),
            expected=shown))
    return "".join(parts)


def _store_not_listed(problem: str, roots: Sequence[Tuple[str, str]], what: str) -> str:
    """Why Register is off when the store could not be listed (Decision 10):
    no report can be confirmed here, and the command line registers a named
    report it verifies. Who acts: the operator, on the server."""
    return r.disabled_words(REGISTER, f"the report store could not be listed, so {what}'s "
                            f"report cannot be confirmed here: "
                            f"{redact(problem, roots).rstrip('.')}. Register it from the "
                            "command line by naming its report (scripts/record_evaluation.py "
                            "--report …).")


def register_state(page: Mapping[str, Any], reports: Mapping[str, Any],
                   run_id: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Whether "Register this report" can be used, and why not.

    **Enabled for a kept report of this run the page can confirm** (policy rev
    16, K8-6r.2; Decision 10). Off only when the request is not an
    operator's, the run kept no report, **this run** is already registered
    (`report_state` finds the record naming this run with these bytes), or
    this run's own report was not built (its own reason) or not read by the
    listing; a store that could not be listed says so and points to the
    command line, which registers a named report it verifies. How the runs of
    the condition group compare, and any other run's problem, never disables
    it."""
    access = page.get("access") or {}
    roots = page.get("roots") or []
    # D8 first, whatever is chosen: a refused request's button shows the lock,
    # and the lock is never without its reason (the one gated-button rule).
    if not access.get("allowed"):
        return False, operator_only(access)
    run = find(reports.get("runs") or [], "run_id", run_id)
    if run is None:
        # The version was read but its store's runs could not be listed.
        listing = reports.get("problem") if reports.get("version_sha256") else None
        if listing and not reports.get("runs"):
            return False, _store_not_listed(listing, roots, "a run")
        return False, None
    state = run.get("state") or {}
    if state.get("kept") != "kept":
        return False, r.disabled_words(REGISTER, "no report was kept in this run.")
    if state.get("registered") == "registered":
        return False, "Already registered."
    if run.get("record") is None:
        if run.get("unbuilt"):
            return False, r.disabled_words(REGISTER, "this report cannot be registered: "
                                                     f"{redact(run['unbuilt'], roots)}")
        if reports.get("store_problem"):
            return False, _store_not_listed(reports["store_problem"], roots, "this run")
        return False, r.disabled_words(REGISTER, "the report store's listing did not read this "
                                                 "report (it may have been written or replaced "
                                                 "since); press ↻ to read it again.")
    return True, None


def notes_visible(reports: Mapping[str, Any], run_id: Optional[str]) -> bool:
    run = find(reports.get("runs") or [], "run_id", run_id)
    return bool(run and run.get("key")
                and (run.get("contest") or {}).get("state") in ("contested", "cannot_be_checked"))


# ---------------------------------------------------------------------------
# Rendering: Import
# ---------------------------------------------------------------------------
def intake_choices(listing: Mapping[str, Any]) -> List[Tuple[str, str]]:
    choices = []
    for item in listing.get("items") or []:
        if item.get("problem"):
            continue
        size = item.get("size_bytes")
        kb = f"{max(1, round(size / 1024))} KB" if isinstance(size, int) else ""
        when = str(item.get("modified_at") or "").replace("T", " ").replace("Z", " UTC")
        choices.append((" · ".join(x for x in (item["name"], kb, f"modified {when}" if when else "")
                                   if x), item["name"]))
    return choices


def intake_note_html(listing: Mapping[str, Any], roots: Sequence[Tuple[str, str]]) -> str:
    """What the import drawer says under its select: the import folder's
    notice when it is unset or refused (one line, its link to Settings › Data
    locations, the rest under "Details ›"), else the listing's own problem,
    else what the folder offers and why any file in it is not offered."""
    if listing.get("notice"):
        return r.notice_block(listing["notice"])
    if listing.get("problem"):
        return _reason_html(redact(listing["problem"], roots))
    items = listing.get("items") or []
    offered = [i for i in items if not i.get("problem")]
    notes = [f'<span class="shep-sub">server list · {len(offered)} file{"s" if len(offered) != 1 else ""} '
             "· shepherd-cases/1 (*.json)</span>"]
    for item in items:
        if item.get("problem"):
            notes.append(f'<div class="shep-sub shep-amber-t">{e(redact(item["problem"], roots))}</div>')
    if not offered:
        notes.append(_reason_html("No import file is in the import folder: place a "
                                  "shepherd-cases/1 file there on the server, then open this "
                                  "again."))
    return "".join(notes)


def role_line(role: Any) -> str:
    """The import form's note under Role: with no role chosen, what to choose;
    for a chosen role, no note."""
    if not isinstance(role, str) or not role.strip():
        return ("Choose the version's role: its declared purpose (for example mygene2, "
                "research_selection or institutional_acceptance).")
    return ""


def role_choices(page: Mapping[str, Any]) -> List[str]:
    """The import form's Role choices: the listed supplied cohorts' own roles,
    then mygene2 and institutional_acceptance, each once (a role may also be
    typed). A KG edge set's role (``train``, ``val``, ``test``) is a part of an
    edge split, and a generated split's the split it is; a supplied import
    refuses ``train`` and ``val``, so neither is offered."""
    roles: List[str] = []
    supplied = (v.get("role") for v in page.get("versions") or []
                if v.get("kind") == protocols.SUPPLIED)
    for role in [*supplied, "mygene2", "institutional_acceptance"]:
        if isinstance(role, str) and role and role not in roles:
            roles.append(role)
    return roles


def preview_summary(plan: Any, inputs: Mapping[str, Any]) -> Dict[str, Any]:
    """What the page keeps of a preview: the inputs it was made from, its id
    and the manifest's aggregates. Never its exclusions by case (D7: case ids
    stay on the machine)."""
    manifest = plan.manifest
    return {
        "preview_id": plan.preview_id, "inputs": dict(inputs),
        "label": manifest.get("label"), "role": manifest.get("role"),
        "source": {k: manifest["source"].get(k) for k in ("filename", "format", "sha256", "n_cases")},
        "counts": dict(manifest.get("counts") or {}),
        "unmappable_terms": dict(manifest.get("unmappable_terms") or {}),
        "mapping": dict(manifest.get("mapping") or {}),
        "graph": dict(manifest.get("graph") or {}),
        "dataset_record": dict(manifest.get("dataset_record") or {}),
        "samples_n": (manifest.get("samples") or {}).get("n"),
    }


def _source_entry(entry: Any) -> str:
    if not isinstance(entry, Mapping):
        return "not recorded"
    return ("digest " + r.fingerprint_value(entry.get("digest"), what="source digest",
                                            copy=False)
            + f", declared version {e(entry.get('declared_version') or 'none')}")


def _gene_rows(genes: Any) -> List[Tuple[str, str]]:
    """The preview's gene counts, when the source lists genes (README §8): the
    operator's check, before commit, that the identifiers map."""
    if not isinstance(genes, Mapping):
        return []
    return [
        ("Gene identifiers", f"<b>{e(_count(genes.get('identifiers_listed')))}</b> listed: "
                             f"{e(_count(genes.get('identifiers_unmappable')))} name no gene "
                             f"of the graph, {e(_count(genes.get('identifiers_repeated')))} "
                             "repeat an earlier one"),
        ("Cases with genes", f"{e(_count(genes.get('cases_with_causal_genes')))} with causal "
                             f"genes ({e(_count(genes.get('cases_with_unmapped_causal_genes')))} "
                             f"with some unmapped), "
                             f"{e(_count(genes.get('cases_with_candidate_genes')))} with "
                             "candidate genes"),
    ]


def preview_html(preview: Optional[Mapping[str, Any]]) -> str:
    """The preview card: counts with their denominators, exclusion reasons and
    their counts, the unmappable terms with how many cases list each, what the
    version binds. No case id."""
    if not preview:
        return ""
    counts = preview.get("counts") or {}
    source = preview.get("source") or {}
    record = preview.get("dataset_record") or {}
    mapping = preview.get("mapping") or {}
    n = counts.get("source_cases")
    excluded = counts.get("excluded") if isinstance(counts.get("excluded"), Mapping) else {}
    reasons = " · ".join(f"{str(k).replace('_', ' ')} {_count(v)}" for k, v in excluded.items())
    terms = preview.get("unmappable_terms") or {}
    top = sorted(terms.items(), key=lambda item: (-item[1], item[0]))[:TERMS_SHOWN]
    term_list = "".join(f'<span class="shep-nowrap"><code>{e(t)}</code> listed by {e(_count(c))} '
                        f'case{"s" if c != 1 else ""}</span>' for t, c in top)
    if len(terms) > len(top):
        term_list += f'<span class="shep-muted">… {len(terms) - len(top)} more, every one recorded in the version</span>'
    graph = preview.get("graph") or {}
    rows = [
        ("Source", f'{e(source.get("filename"))} ({e(source.get("format"))}) · sha256 '
                   + r.fingerprint_value(source.get("sha256"), what="source fingerprint",
                                         full=True, copy=False)),
        ("Source cases", f"<b>{e(_count(n))}</b>"),
        ("Ranked-eligible", f"<b>{e(_count(counts.get('ranked_eligible')))}</b> of {e(_count(n))}"),
        ("Excluded", f"<b>{e(_count(counts.get('n_excluded')))}</b> of {e(_count(n))}"
                     f"{': ' + e(reasons) if reasons else ''} "
                     '<span class="shep-muted">(by category; case ids are not sent to the browser: '
                     "the command-line preview on the machine lists them, and the committed "
                     "version records them)</span>"),
        ("Terms unmappable", f"<b>{e(_count(counts.get('terms_unmappable')))}</b> distinct"
                             + (f'<div class="shep-test-terms">{term_list}</div>' if term_list else "")),
        ("Cases with lost terms", f"{e(_count(counts.get('cases_with_lost_terms')))} of "
                                  f"{e(_count(counts.get('ranked_eligible')))} ranked (scored on fewer "
                                  "terms than listed)"),
        ("Repeats removed", f"{e(_count(counts.get('repeats_removed')))} in "
                            f"{e(_count(counts.get('cases_with_repeats')))} case(s) (phenotype "
                            f"normalisation v{e(mapping.get('phenotype_normalisation_version'))})"),
        ("Partial phenotypes", e(mapping.get("partial_phenotypes"))),
        *_gene_rows(counts.get("genes")),
        ("Graph", "the workspace manifest’s digests: " + ", ".join(
            f"{e(role)} " + r.fingerprint_value(d, what=f"{role} digest", copy=False)
            for role, d in graph.items())),
        ("Dataset record", ('origin <b class="shep-amber-t">synthetic</b>' if record.get("origin") == "synthetic"
                            else f"origin {e(record.get('origin'))}")
                           + ", sha256 " + r.fingerprint_value(record.get("sha256"),
                                                               what="dataset record digest",
                                                               copy=False)),
        ("HPO · MONDO", f"HPO {_source_entry(record.get('hpo'))} · MONDO {_source_entry(record.get('mondo'))}"),
    ]
    warn = ""
    if record.get("origin") == "synthetic":
        warn += ('<div class="shep-notice shep-notice-amber"><b>SYNTHETIC WORKSPACE:</b> this '
                 "graph was not built from real files.</div>")
    if not preview.get("samples_n"):
        warn += _reason_html("No case of the source is eligible for ranking against this graph, "
                             "so there is nothing to commit.")
    return (f'<div class="shep-card shep-test-preview"><div class="shep-card-hd">Preview '
            f'<code>{e(preview.get("preview_id"))}</code> {r.chip("nothing written", "grey")}'
            f'{help_button("test-preview-id", "Preview id")}'
            f'<span class="shep-muted"> label {e(preview.get("label"))} · role {e(preview.get("role"))}'
            f'</span></div>{r.summary_list(rows)}{warn}</div>')


# ---------------------------------------------------------------------------
# The operator controls (each: D8 on its own request first)
# ---------------------------------------------------------------------------
def _outcome_box(kind: str, text: str, roots: Sequence[Tuple[str, str]] = ()) -> str:
    return f'<div class="shep-outcome shep-outcome-{kind}">{e(redact(text, roots))}</div>'


def start_test_run(page: Mapping[str, Any], models: Mapping[str, Any], workspace: Any,
                   label: Any, key: Any, record: Any, deterministic: Any, request: Any, *,
                   protocol: Any = None, manager: Any = None,
                   environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """"Start test": D8 on this request; the workspace, the version and the
    checkpoint found in the server's lists again (never a client path); a
    measurement `protocol` (the Measurement radio's value) this page does not
    start for that checkpoint and version refused (`measure_refusal`, the rule
    Start's reason shows); the **digests shown to the operator** passed as the
    expected ones, with the protocol apart from the model's learning stage,
    task and capability description from the session's listing and the listed
    version's kind and counts, so the runner judges again before it starts and
    the run refuses before measuring when the checkpoint or the version it
    reads is not the one chosen, or its model is not one the protocol takes;
    the runner's one start for test runs."""
    from src.api.services.server_lists import (
        ServerListRefused,
        resolve_checkpoint,
        resolve_workspace,
    )
    from src.api.services.training_manager import TestRunInputs, evidence_choice
    from src.config import server_settings

    access = operator_access(request)
    if not access.allowed:
        return {"success": False, "error": access.refusal}
    if manager is None:
        from src.api.services.training_manager import training_manager as manager
    version = find(page.get("versions") or [], "label", label)
    item = find_model(models, key)
    if version is None or version.get("problem") or not r.is_digest(version.get("manifest_sha256")):
        return {"success": False,
                "error": "Choose a data version the server lists. Nothing was started."}
    if (item is None or models.get("workspace") != workspace or item.get("problem")
            or not r.is_digest(item.get("sha256"))):
        return {"success": False, "error": "Choose a model the server lists in this workspace. "
                                           "Nothing was started."}
    refusal = measure_refusal(item, version, protocol)
    if refusal:
        return {"success": False, "error": said(refusal[:1].upper() + refusal[1:],
                                                "Nothing was started.")}
    registry = server_settings.cohort_registry(environ)
    # The runner's one evidence choice by the version's kind, of the setting
    # read now: a supplied version's directories (a refused setting refuses
    # the start), none for a KG edge set or a generated split.
    evidence = evidence_choice(version.get("kind"), environ=environ)
    if not registry.available:
        return {"success": False, "error": said(registry.problem, "Nothing was started.")}
    if evidence.problem:
        return {"success": False, "error": said(evidence.problem, "Nothing was started.")}
    try:
        data_dir = resolve_workspace(workspace, environ=environ)
        checkpoint = resolve_checkpoint(workspace, item.get("architecture"), item.get("run"),
                                        item.get("filename"), environ=environ)
    except ServerListRefused as exc:
        return {"success": False, "error": said(exc, "Nothing was started.")}
    inputs = TestRunInputs(
        registry=registry.value, version_label=str(label),
        version_sha256=version["manifest_sha256"], workspace=data_dir, checkpoint=checkpoint,
        checkpoint_sha256=item["sha256"], protocol=protocol,
        learning_stage=item["learning_stage"], task=item["task"],
        capabilities=item["capabilities"], version_kind=version["kind"],
        version_counts=version["counts"], evidence_dirs=evidence.dirs,
        record=bool(record), deterministic=bool(deterministic))
    return manager.start_test(inputs)


def cancel_test_run(request: Any, *, manager: Any = None) -> Dict[str, Any]:
    """"Cancel": D8 on this request; the runner's stop for test runs only (a
    training job is never stopped by it). Returns once the run has ended, with
    what it left."""
    access = operator_access(request)
    if not access.allowed:
        return {"success": False, "error": access.refusal}
    if manager is None:
        from src.api.services.training_manager import training_manager as manager
    return manager.cancel_test()


def register_selected(page: Mapping[str, Any], reports: Mapping[str, Any],
                      models: Mapping[str, Any], workspace: Any, run_id: Any, request: Any, *,
                      environ: Optional[Mapping[str, str]] = None) -> Tuple[bool, str]:
    """"Register this report": D8 on this request; the run's report found in
    the version's listed report store; then the one registration function,
    given the digest shown, which compares it on the one read it registers,
    before parsing it (a report changed since it was shown is refused there,
    in its words, and never parsed; one changed after that read is registered
    as the bytes the read holds, the ones shown), and which finds the
    ledger's place from that same read through this page's placement
    (`ledger_checkpoint`: the listed file holding the weights the report
    measured, by full digest, the file the run read first; never the report's
    recorded path), then checks the deployment's registry, the store, the
    version and the checkpoint's bytes again. The page reads and judges no
    report itself (round 3 of the review fixes, F2). A registered run's message
    carries the two notes that decide nothing (`group_note`,
    `historical_note`)."""
    from src.api.services.server_lists import ServerListRefused, resolve_report
    from src.evaluation.registration import (
        RegistrationRefused,
        group_note,
        historical_note,
        register_report,
    )

    access = operator_access(request)
    if not access.allowed:
        return False, access.refusal
    label = reports.get("label")
    run = find(reports.get("runs") or [], "run_id", run_id)
    state = (run or {}).get("state") or {}
    shown = state.get("report_sha256")
    # A kept report is listed with its digest; without one there is nothing to
    # bind the registration to, and it is never registered unbound.
    if run is None or state.get("kept") != "kept" or not r.is_digest(shown):
        return False, "Select a kept report of the version first. Nothing was registered."
    try:
        path = resolve_report(label, run_id, environ=environ)
    except ServerListRefused as exc:
        return False, said(exc, "Nothing was registered.")

    def beside(manifest: Mapping[str, Any]) -> Path:
        """The ledger's place, from the manifest the registration parsed from
        the bytes it compared with the digest shown."""
        place = ledger_checkpoint(manifest, page, models, workspace, environ=environ)
        if place.path is None:
            where = ", ".join(place.workspaces) or "no listed workspace"
            measured = manifest["artifact_digests"]["checkpoint"]
            raise ValueError(
                f"No checkpoint the server lists ({where}) holds the weights this report "
                f"measured (model {measured[:12]}…), so where its ledger goes cannot be found "
                "from the server's lists. A checkpoint moved since the run can be registered "
                "on the machine with scripts/record_evaluation.py --report … --checkpoint ….")
        return place.path

    try:
        # **Bound to the report shown** (the M5 review's P2-1): the digest shown
        # is compared on the one read the registration parses and registers,
        # and the ledger is placed from that read; the settings this report was
        # found by say which registry is the deployment's.
        result = register_report(path, place=beside, expected_report_sha256=shown,
                                 environ=environ)
    except RegistrationRefused as exc:
        return False, str(exc)
    sequence = result.record.get("registration", {}).get("sequence")
    # Not named `said`: that would make the module's `said` local to this
    # function, and the refusals above would fail before it is assigned.
    message = {"registered": f"Registered as record {sequence}.",
               "already_registered": f"Already registered as record {sequence}; nothing was "
                                     "written."}.get(result.outcome, result.outcome)
    for note in (group_note(result), historical_note(result)):
        if note:
            message += f" {note}"
    if result.usage_history_problem:
        message += f" {result.usage_history_problem[:1].upper()}{result.usage_history_problem[1:]}."
    return True, message


def add_note(reports: Mapping[str, Any], run_id: Any, who: Any, what: Any, request: Any, *,
             environ: Optional[Mapping[str, str]] = None) -> Tuple[bool, str]:
    """"Add note": D8 on this request; one investigation note kept with the
    report's condition group in the store (`add_investigation_note`). It
    records what was found and changes nothing."""
    from src.api.services.server_lists import ServerListRefused, resolve_cohort_version
    from src.evaluation.report_store import add_investigation_note
    from src.utils.file_lock import FileLockBusy

    access = operator_access(request)
    if not access.allowed:
        return False, access.refusal
    run = find(reports.get("runs") or [], "run_id", run_id)
    if run is None or not run.get("key"):
        return False, "Select a report whose condition group is known first. Nothing was written."
    try:
        version = resolve_cohort_version(reports.get("label"), environ=environ)
        add_investigation_note(version, tuple(run["key"]), who=str(who or ""), what=str(what or ""))
    except ServerListRefused as exc:
        return False, said(exc, "Nothing was written.")
    except FileLockBusy as exc:
        # Another writer held the notes file (K8-6r.1): busy, not refused for
        # good. The note stays in its box, so the operator can add it again.
        return False, (f"The note was not kept: another writer held the notes file for the "
                       f"{exc.grace:g} s this one waited. Nothing was written this time; add it "
                       "again.")
    except (OSError, ValueError) as exc:
        return False, str(exc)
    return True, "The note was kept with this group's reports. It records what was found and changes nothing."


#: The import drawer's title and what the import is, under it (two short
#: sentences, never a chain beside Close; the workspace by its relation, its
#: name being its select's in Data): Close stands on the title's line.
IMPORT_TITLE_HTML = ('<h2>Import a new cohort version</h2><p class="shep-sub">Mapped against the '
                     'selected workspace and read on the CPU by this service. Nothing is written '
                     'before Commit.</p>')


def open_import(request: Any, *, environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """"Import a new version…": D8 on this request (the intake directory holds
    patient-level data), then the server's intake listing; when nothing is
    listed because the import folder is unset or refused, its notice
    (`workspaces.setting_notice`, chosen by the setting's flags, never by the
    listing's words), which the drawer says in place of the sentence."""
    from src.api.services.server_lists import list_intake_files
    from src.config import server_settings

    access = operator_access(request)
    if not access.allowed:
        return {"refused": access.refusal, "items": [], "problem": None, "notice": None}
    listing = list_intake_files(environ=environ)
    notice = (setting_notice(server_settings.intake_directory(environ)) if listing.problem
              else None)
    return {"refused": None, "items": [asdict(i) for i in listing.items], "problem": listing.problem,
            "notice": notice}


def _import_inputs(workspace: Any, file: Any, label: Any, role: Any, policy: Any) -> Dict[str, Any]:
    return {"workspace": workspace, "file": file,
            "label": (label or "").strip() if isinstance(label, str) else label,
            "role": (role or "").strip() if isinstance(role, str) else role, "policy": policy}


def _prepare(inputs: Mapping[str, Any], environ: Optional[Mapping[str, str]]) -> Any:
    """The importer's preview from server-resolved inputs: the registry from
    its setting, the workspace and the source found in the server's lists."""
    from src.api.services.server_lists import resolve_intake_file, resolve_workspace
    from src.config import server_settings
    from src.evaluation.cohort_registry import prepare_import

    registry = server_settings.cohort_registry(environ)
    if not registry.available:
        raise ValueError(registry.problem)
    data_dir = resolve_workspace(inputs["workspace"], environ=environ)
    source = resolve_intake_file(inputs["file"], environ=environ)
    return prepare_import(registry.value, data_dir, source, inputs["label"], inputs["role"],
                          inputs["policy"])


def preview_import(workspace: Any, file: Any, label: Any, role: Any, policy: Any,
                   request: Any, *, environ: Optional[Mapping[str, str]] = None
                   ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """"Preview": D8 on this request; the importer's `prepare_import` on the
    server-resolved inputs (it writes nothing); the page keeps the aggregates
    only (`preview_summary`)."""
    access = operator_access(request)
    if not access.allowed:
        return None, access.refusal
    inputs = _import_inputs(workspace, file, label, role, policy)
    if not inputs["file"]:
        return None, "Choose a file from the import folder. Nothing was written."
    try:
        plan = _prepare(inputs, environ)
    except (OSError, ValueError) as exc:  # ServerListRefused, RegistryRefused, MalformedSource, …
        return None, import_refusal(exc)
    return preview_summary(plan, inputs), None


def commit_import_ui(preview: Optional[Mapping[str, Any]], workspace: Any, file: Any, label: Any,
                     role: Any, policy: Any, request: Any, *,
                     environ: Optional[Mapping[str, str]] = None) -> Tuple[bool, str, Optional[str]]:
    """"Commit": D8 on this request; the form must be the one previewed; the
    importer prepares again from fresh reads of the same server-resolved inputs
    and commits only under the preview id shown (`commit_import` recomputes it
    and refuses changed inputs). Returns (done, message, committed label)."""
    from src.config import server_settings
    from src.evaluation.cohort_registry import commit_import

    access = operator_access(request)
    if not access.allowed:
        return False, access.refusal, None
    if not preview:
        return False, "Preview first: Commit publishes only what a preview showed. Nothing was written.", None
    inputs = _import_inputs(workspace, file, label, role, policy)
    if inputs != preview.get("inputs"):
        return False, ("The form changed since the preview (file, label, role, policy or "
                       "workspace): preview again. Nothing was written."), None
    try:
        plan = _prepare(inputs, environ)
    except (OSError, ValueError) as exc:
        return False, import_refusal(exc), None
    registry = server_settings.cohort_registry(environ)
    try:
        version = commit_import(plan, registry.value, preview["preview_id"])
    except OSError as exc:
        return False, (f"Failed while writing: {exc}. The version was not committed (cohort.json "
                       "is written last); a version directory left without it is an interrupted "
                       "import, refused by every reader. Remove it before reusing the label."), None
    except ValueError as exc:
        # As the importer says it: a refusal after the source copy was kept
        # says so, so "Nothing was written." is not added here.
        return False, f"Refused: {exc}", None
    manifest = version.manifest
    return True, (f"Committed cohort version {manifest['label']} (role {manifest['role']}): "
                  f"{manifest['samples']['n']} ranked of {manifest['counts']['source_cases']} "
                  f"source cases; version {version.identity.sha256[:12]}."), manifest["label"]


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
#: Report row clicks become the list's click event (Enter and Space click a
#: button). "Show all N reports" and "Show the newest N only" switch a class on
#: the component's own element, which stays while the list's value is
#: replaced, so the choice survives the list's updates (this browser tab only).
_REPORTS_JS = r"""
element.addEventListener('click', function (e) {
  if (e.target.closest && e.target.closest('.shep-copy, .shep-i')) { return; }
  var more = e.target.closest ? e.target.closest('button[data-more]') : null;
  if (more) {
    var all = more.getAttribute('data-more') === 'all';
    element.classList.toggle('shep-test-all', all);
    var other = element.querySelector('button[data-more="' + (all ? 'fewer' : 'all') + '"]');
    if (other) { other.focus(); }
    return;
  }
  var row = e.target.closest ? e.target.closest('tr[data-key]') : null;
  if (row) { trigger('click', {key: row.getAttribute('data-key')}); }
});
element.addEventListener('keydown', function (e) {
  var row = e.target.closest ? e.target.closest('tr[data-key]') : null;
  if (!row || e.target !== row) { return; }
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); trigger('click', {key: row.getAttribute('data-key')}); }
});
"""

#: When the import drawer opens, its title takes the focus: at narrow sizes the
#: drawer is an overlay, and a keyboard or screen-reader user starts in it.
_IMPORT_FOCUS_JS = r"""
() => {
  setTimeout(function () {
    var title = document.querySelector('#shep-test-import-title h2');
    if (title && title.offsetParent !== null) {
      title.setAttribute('tabindex', '-1');
      title.focus({preventScroll: true});
    }
  }, 60);
}
"""

_EMPTY_MODELS = {"workspace": None, "items": [], "problem": None, "refused": False}
_EMPTY_REPORTS = {"label": None, "runs": [], "problem": None, "notes": [], "notes_problems": []}


def _dd(choices: List[Tuple[str, str]], value: Any, *, interactive: bool = True) -> Any:
    values = [v for _l, v in choices]
    return gr.update(choices=choices, value=value if value in values else None,
                     interactive=interactive)


def create_test_page(ctx: Any) -> None:
    """Build Research › Evaluate › Test (the page contract: `src/webui/shell.py`)."""
    ctx.help.fields(FIELD_HELP)
    ctx.help.topics(TOPIC_HELP)
    page_state = gr.State({})
    models_state = gr.State(dict(_EMPTY_MODELS))
    binding_state_ = gr.State(None)
    reports_state = gr.State(dict(_EMPTY_REPORTS))
    selected_report = gr.State(None)
    # The Measurement radio's value as it was last written (`_measurement_outputs`),
    # which the selection's listeners read in place of the radio itself: the
    # radio's value an event was asked with can be one the event before it in
    # the group took away from the radio's choices (a Model pick sends its
    # input twice, and the first event can empty the radio), and Gradio would
    # refuse that event with an error before the page's own handling ran.
    measurement_state = gr.State(None)
    # The chosen checkpoint's key as it was last written (`_run_outputs`), which
    # the selection's listeners read in place of the Checkpoint select, for the
    # radio's reason: a run's choice sets the select's choices to that run's
    # checkpoints, so a value an event was asked with can be one the event
    # before it took away from them, and Gradio would refuse that event with an
    # error before the page's own handling ran (the run's own pick is asked
    # twice). The select's own pick, and Start, read the select itself.
    model_state = gr.State(None)
    # What this session saw of the job: the poll's alone (`_poll`).
    job_state = gr.State({})
    preview_state = gr.State(None)

    with gr.Row(elem_classes=["shep-page", "shep-test"], elem_id="shep-test-row", equal_height=False):
        # ------------------------------------------------------------ Cohort
        # Each panel sized by what it holds (`.shep-sized`: its basis, growth
        # and floor in rem, `PAGE_CSS`), never a share fitted to one window.
        with gr.Column(elem_classes=["shep-panel", "shep-sized", "shep-test-col"],
                       elem_id="shep-test-cohort"):
            cohort_hd = plain_html(cohort_header_html({}), elem_id="shep-test-cohort-hd")
            # Its one scroll region says when there is more below (R-73).
            with gr.Column(elem_classes=["shep-test-bd"]):
                workspace = gr.Dropdown(label="Workspace", choices=[], value=None,
                                        elem_id="shep-test-workspace", interactive=True,
                                        elem_classes=["shep-test-select"])
                workspace_info = plain_html("", elem_id="shep-test-workspace-info")
                version = gr.Dropdown(label="Data version", choices=[], value=None,
                                      elem_id="shep-test-version", interactive=True,
                                      elem_classes=["shep-test-select"])
                version_info = plain_html("", elem_id="shep-test-version-info")
                with gr.Row(elem_classes=["shep-test-actions"]):
                    import_open = gr.Button(GATED_LABELS["import"], variant="secondary",
                                            interactive=False, elem_id="shep-test-import-open",
                                            scale=0)
                import_reason = plain_html("", elem_id="shep-test-import-reason")
                import_msg = plain_html("", elem_id="shep-test-import-msg")
        # ------------------------------------------------------------ Import (a drawer)
        with gr.Column(visible=False, elem_classes=["shep-panel", "shep-test-drawer"],
                       elem_id="shep-test-import") as drawer:
            with gr.Row(elem_classes=["shep-panel-hd", "shep-test-drawer-hd"]):
                drawer_title = plain_html(IMPORT_TITLE_HTML, elem_id="shep-test-import-title")
                import_close = gr.Button("✕ Close", variant="secondary", size="sm", scale=0,
                                         elem_id="shep-test-import-close")
            with gr.Column(elem_classes=["shep-test-bd"]):
                intake = gr.Dropdown(label=INTAKE_LABEL, choices=[],
                                     value=None, elem_id="shep-test-intake", interactive=True,
                                     elem_classes=["shep-test-select"])
                intake_note = plain_html("", elem_id="shep-test-intake-note")
                with gr.Row(elem_classes=["shep-test-form"]):
                    new_label = gr.Textbox(label="Version label", elem_id="shep-test-label",
                                           placeholder="for example mygene2-2026-10", max_lines=1)
                    new_role = gr.Dropdown(label="Role", choices=[], value=None,
                                           allow_custom_value=True, elem_id="shep-test-role",
                                           elem_classes=["shep-test-select"])
                    new_policy = gr.Dropdown(label="Lost terms", choices=POLICY_CHOICES,
                                             value="keep", elem_id="shep-test-partial",
                                             elem_classes=["shep-test-select"])
                role_note = plain_html(_reason_html(role_line(None)), elem_id="shep-test-role-note")
                with gr.Row(elem_classes=["shep-test-actions"]):
                    preview_btn = gr.Button("Preview", variant="secondary",
                                            elem_id="shep-test-preview", scale=0)
                preview_out = plain_html("", elem_id="shep-test-preview-card")
            with gr.Row(elem_classes=["shep-panel-ft"]):
                commit_btn = gr.Button("Commit", variant="primary", interactive=False,
                                       elem_id="shep-test-commit", scale=0)
                plain_html(
                    '<span class="shep-note">Commit prepares again and refuses if the preview id '
                    "changed.</span>", elem_id="shep-test-commit-note")
                commit_out = plain_html("", elem_id="shep-test-commit-out")
        # ------------------------------------------------------------ Run
        # Three rows at wide sizes: the title; the body, the panel's one
        # scroll region, with the choices and then the job's details; and the
        # foot, in flow and never scrolled, with Start, its reason, the action
        # line and the job's title row (its state, Cancel beside it), so Start,
        # the job's state and Cancel are on screen whatever the model's
        # details or the output's length. Narrow, the three follow one another
        # in flow.
        with gr.Column(elem_classes=["shep-panel", "shep-sized", "shep-test-col"],
                       elem_id="shep-test-run"):
            plain_html(run_header_html(), elem_id="shep-test-run-hd")
            with gr.Column(elem_classes=["shep-test-bd"]):
                # The model as Run → Checkpoint, the shared cascading pair
                # (`model_list.parents_update`), one under the other so each
                # 40-character option shows whole in the column; ↻ lists the
                # workspace's checkpoints again. Above them, for a request D8
                # refuses, why both lists are empty and ↻ is locked
                # (`listing_why_html`), once.
                model_why = plain_html("", elem_id="shep-test-model-why")
                # A bar of one group (the select and its ↻ never part), its
                # button on the box's line (`shep-field-row`).
                with gr.Row(elem_classes=["shep-test-model-row", "shep-bar"]):
                    with gr.Row(elem_classes=["shep-grp", "shep-grp-grow", "shep-field-row"]):
                        run_select = gr.Dropdown(label="Run", choices=[], value=None,
                                                 elem_id="shep-test-model-run", interactive=False,
                                                 elem_classes=["shep-test-select"])
                        model_reload = gr.Button(GATED_LABELS["reload"], variant="secondary",
                                                 size="sm", scale=0,
                                                 elem_id="shep-test-model-reload",
                                                 elem_classes=["shep-icon-btn"], interactive=False)
                model = gr.Dropdown(label="Checkpoint", choices=[], value=None,
                                    elem_id="shep-test-model", interactive=False,
                                    elem_classes=["shep-test-select"])
                model_info = plain_html("", elem_id="shep-test-model-info")
                # The protocols the chosen model's listed capabilities admit (`measurement_choices`).
                measurement = gr.Radio(label="Measurement", choices=[], value=None,
                                       elem_id="shep-test-measurement", interactive=False,
                                       visible=False)
                evidence = plain_html("", elem_id="shep-test-evidence")
                with gr.Row(elem_classes=["shep-test-checks"]):
                    record = gr.Checkbox(label=RECORD_LABEL, value=True,
                                         elem_id="shep-test-record")
                    deterministic = gr.Checkbox(label="Deterministic execution", value=False,
                                                elem_id="shep-test-deterministic")
                fixed = plain_html("", elem_id="shep-test-fixed")
                # The job's card at its natural height, after the choices:
                # what it left, what it was given, the rest one Details away
                # (`job_body_html`); none while no run was started. Its state
                # is the foot's. The poll draws it again as a run prints, so
                # its Details stays as the reader left it (`plain_html`'s
                # keeper), while it shows the same job (`job_body_html`'s
                # `r.keep_as`).
                job_out = plain_html(job_body_html({}, [], ("none", "")), elem_id="shep-test-job")
            with gr.Column(elem_classes=["shep-test-runft"], elem_id="shep-test-runft"):
                start = gr.Button(GATED_LABELS["start"], variant="primary",
                                  interactive=False, elem_id="shep-test-start")
                start_reason = plain_html("", elem_id="shep-test-start-reason")
                action_out = plain_html("", elem_id="shep-test-action")
                # A bar: the job's state, its words free to wrap, then Cancel
                # at the line's end, on a line of its own when they leave it
                # no room (never a squeezed state).
                with gr.Row(elem_classes=["shep-test-job-top", "shep-bar"]):
                    job_head = plain_html(job_head_html({}), elem_id="shep-test-job-head",
                                          elem_classes=["shep-grp-grow"])
                    # Drawn only while a run is busy (R-81: an idle Cancel
                    # nothing can use took the foot's room).
                    cancel = gr.Button(GATED_LABELS["cancel"], variant="stop",
                                       interactive=False, visible=False,
                                       elem_id="shep-test-cancel", scale=0, size="sm",
                                       elem_classes=["shep-grp-end"])
                cancel_reason = plain_html("", elem_id="shep-test-cancel-reason")
        # ------------------------------------------------------------ Result
        # The reports as a compact single-select table at the panel's top, the
        # chosen report's result under it, both in the panel's one scroll
        # region; the title, the note form and Register are in flow.
        with gr.Column(elem_classes=["shep-panel", "shep-sized", "shep-test-col"],
                       elem_id="shep-test-result"):
            reports_hd = plain_html(reports_header_html({}), elem_id="shep-test-reports-hd")
            # Its one scroll region says when there is more below (R-73).
            with gr.Column(elem_classes=["shep-test-bd", "shep-test-bd-flush"]):
                reports_list = plain_html("", elem_id="shep-test-reports", js_on_load=_REPORTS_JS)
                # Drawn again when the poll sees a job start or end: its
                # disclosures stay as the reader left them for the same report
                # (`plain_html`'s keeper).
                result_out = plain_html(result_html({}, None, []), elem_id="shep-test-result-body")
            # The investigation note form (shown while the runs of the report's
            # condition group differ, or their comparison is incomplete) is
            # outside the scrolling body: on a wide screen it
            # stays at the panel's foot, above Register, so Add note is in view
            # however long the report (in a short window it takes at most half
            # the panel and scrolls in its own region); narrow, it follows the
            # report. "What was found" grows to four lines, then scrolls inside
            # itself, so a long note cannot push Add note out of that region
            # either (unbounded, eight lines did at every wide size).
            with gr.Column(visible=False, elem_classes=["shep-test-noteform"],
                           elem_id="shep-test-noteform") as note_form:
                # "Who" holds a normal name whole (M5-3: at 166 px "research
                # screenshots v2" lost its "v2"): two fifths of the row, never
                # under 240 px; the row wraps before either gets narrower.
                with gr.Row(elem_classes=["shep-test-noterow"]):
                    note_who = gr.Textbox(label="Who", elem_id="shep-test-note-who", max_lines=1,
                                          scale=2, min_width=NOTE_FIELD_MIN_WIDTH)
                    note_what = gr.Textbox(label="What was found", elem_id="shep-test-note-what",
                                           lines=2, max_lines=4, scale=3,
                                           min_width=NOTE_FIELD_MIN_WIDTH)
                with gr.Row(elem_classes=["shep-test-actions"]):
                    note_btn = gr.Button(GATED_LABELS["note"], variant="secondary",
                                         elem_id="shep-test-note-add", scale=0)
                    note_reason = plain_html("", elem_id="shep-test-note-reason")
            note_out = plain_html("", elem_id="shep-test-note-out")
            with gr.Row(elem_classes=["shep-panel-ft"]):
                register = gr.Button(GATED_LABELS["register"], variant="primary",
                                     interactive=False, elem_id="shep-test-register", scale=0)
                register_reason = plain_html("", elem_id="shep-test-register-reason",
                                             elem_classes=["shep-beside"])
                register_out = plain_html("", elem_id="shep-test-register-out")
    # The poll's hand-off to the selection's group (`_poll`): a hidden
    # component, not a State, because the answer to a queue=False request
    # dispatches a component's .change and never a State's.
    job_changed = gr.JSON(value=None, visible=False, elem_id="shep-test-job-changed")
    # The poll's hand-off of a job card that changed (its elapsed time, its
    # state), drawn again in the selection's group (B46-races).
    card_changed = gr.JSON(value=None, visible=False, elem_id="shep-test-job-card-changed")

    # ------------------------------------------------------------------ helpers
    def _models_for(page: Mapping[str, Any], current: Mapping[str, Any], name: Optional[str],
                    request: Any, *, again: bool = False) -> Dict[str, Any]:
        """The session's checkpoint listing for `name`: kept while it is the
        same workspace (listing opens every checkpoint, so a page open does not
        repeat it), read again on a change or on ↻."""
        if not (page.get("access") or {}).get("allowed"):
            return {"workspace": name, "items": [], "problem": None, "refused": True}
        if not again and current.get("workspace") == name and not current.get("refused") \
                and name is not None and (current.get("items") or current.get("problem")):
            return dict(current)
        return read_models(name, request)

    def _binding_for(page: Mapping[str, Any], name: Optional[str]) -> Dict[str, Any]:
        directory = (page.get("directories") or {}).get(name)
        row = find(page.get("workspaces") or [], "name", name)
        return workspace_binding(Path(directory) if directory else None, row)

    def _gpu(page: Mapping[str, Any]) -> Tuple[str, str]:
        return gpu_line(operator=bool((page.get("access") or {}).get("allowed")))

    def _start_outputs(page, models, binding, ws, label, key, protocol, gpu):
        """Start and its reason, from the selection (`start_outputs`); `protocol`
        is the Measurement radio's value, as it shows it."""
        allowed = bool((page.get("access") or {}).get("allowed"))
        can_start, why = start_state(page, models, ws, label, key, binding, gpu,
                                     protocol=protocol)
        return [gated_update("start", page, interactive=can_start),
                start_reason_html(page, allowed, can_start, why)]

    def _measurement_outputs(page, models, label, key, protocol):
        """The Measurement radio for the chosen model on the chosen version
        (`measurement_choices`: what its capabilities meet that measures the
        version's kind), and the value it shows (`chosen_measurement`), which
        Start's state is then drawn from. Like the Model select, it is
        interactive for an operator only. With nothing to offer (no model
        chosen) it is not drawn: a label over an empty choice group says
        nothing (the skill's §9; Start's reason says what to choose)."""
        allowed = bool((page.get("access") or {}).get("allowed"))
        choices = measurement_choices(find_model(models, key),
                                      find(page.get("versions") or [], "label", label))
        protocol = chosen_measurement(choices, protocol)
        return gr.update(choices=choices, value=protocol, visible=bool(choices),
                         interactive=allowed and bool(choices)), protocol

    def _job_outputs(page, job, gpu):
        """The job's title row, Cancel and Cancel's reason (the Run panel's
        foot) and its card (in the panel's body), from the job alone
        (`job_outputs`). Cancel is drawn only while a run is busy (R-81);
        refused, it shows the lock then, and its reason beside it
        (`chips.gated`)."""
        roots = page.get("roots") or []
        allowed = bool((page.get("access") or {}).get("allowed"))
        status = job.get("status") or "idle"
        busy = status in _BUSY
        can_cancel = allowed and status == "running"
        cancel_why = None if allowed or not busy else operator_only(page.get("access") or {})
        return [job_head_html(job), job_body_html(job, roots, gpu),
                gr.update(**gated(GATED_LABELS["cancel"], allowed, interactive=can_cancel),
                          visible=busy),
                _reason_html(cancel_why, kind="lock")]

    def _run_outputs(page, models, binding, ws, label, key, protocol, job, gpu):
        allowed = bool((page.get("access") or {}).get("allowed"))
        radio, protocol = _measurement_outputs(page, models, label, key, protocol)
        return [
            # The shared pair; nothing chosen until the operator picks a run
            # (`chosen_in_run` then takes its latest save), but a lone
            # checkpoint that can be tested (`chosen_model`): of two or more,
            # the page never picks the model it measures itself.
            *ml.parents_update(model_parents(models), key, first=False, interactive=allowed),
            gated_update("reload", page, interactive=bool(ws)),
            listing_why_html(page, models),
            model_html(page, models, key),
            radio,
            evidence_html(page, ws, find(page.get("versions") or [], "label", label)),
            fixed_html(page),
            *_start_outputs(page, models, binding, ws, label, key, protocol, gpu),
            *_job_outputs(page, job, gpu),
            gr.update(interactive=allowed),
            gr.update(interactive=allowed),
            protocol,
            key if find_model(models, key) is not None else None,
        ]

    # Start and its reason depend on the selection, the job's title row, its
    # details and Cancel on the job alone: only the selection's group writes
    # any of them, the poll's hand-offs included (B46-races: when the poll
    # wrote the card too, a tick that read the job before an act's read and
    # answered after it brought back the older card).
    start_outputs = [start, start_reason]
    job_outputs = [job_head, job_out, cancel, cancel_reason]
    run_outputs = [run_select, model, model_reload, model_why, model_info, measurement, evidence,
                   fixed, *start_outputs, *job_outputs, record, deterministic, measurement_state,
                   model_state]

    def _result_outputs(page, reports, models, run_id, job, binding, ws, label, key):
        """The table, the result and its controls for `reports` as this
        session keeps them (`_reports` and a row's click ask the shown report's
        registration first), each report said against the selection
        (`selection_of`: the workspace, version and checkpoint chosen now)."""
        roots = page.get("roots") or []
        can, why = register_state(page, reports, run_id)
        show_notes = notes_visible(reports, run_id)
        allowed = bool((page.get("access") or {}).get("allowed"))
        chosen = selection_of(page, models, binding, ws, label, key)
        return [
            reports_header_html(reports),
            reports_html(reports, run_id, models=models, job=job, roots=roots, selection=chosen,
                         page=page),
            result_html(reports, run_id, roots, job=job, notes=reports.get("notes"),
                        notes_problems=reports.get("notes_problems") or (), selection=chosen,
                        models=models, workspaces=page.get("workspaces") or []),
            gated_update("register", page, interactive=can),
            _reason_html(why, kind="lock" if not allowed else "off"),
            gr.update(visible=show_notes),
            gated_update("note", page),
            gr.update(interactive=allowed),
            gr.update(interactive=allowed),
            _reason_html(None if allowed else operator_only(page.get("access") or {}),
                         kind="lock"),
        ]

    result_outputs = [reports_hd, reports_list, result_out, register, register_reason, note_form,
                      note_btn, note_who, note_what, note_reason]

    def _reports(page, models, label, current_report, job, request, *, prefer_outcome=False):
        """The version's reports as this session shows them: read again, every
        row's registration refined from the session's checkpoint listing
        (operators), the report to show chosen, and its own registration asked
        with the order Register uses (D8 asked on `request`, the triggering
        one, before another workspace is listed); kept in the session's state,
        so a row's answer does not change with the selection."""
        reports = refine_rows(page, read_reports(label), models)
        run_id = _select_report(reports, current_report, job, prefer_outcome=prefer_outcome)
        return refine_registration(page, reports, run_id, models, request=request), run_id

    def _cohort_outputs(page, binding, ws, label):
        registry_ok = not page.get("registry_problem")
        return [
            cohort_header_html(page),
            _dd(r.workspace_choices(page.get("workspaces") or []), ws),
            workspace_html(page, ws),
            _dd(version_choices(page), label, interactive=registry_ok),
            version_html(page, label, binding),
            gated_update("import", page, interactive=registry_ok and bool(ws)),
            import_reason_html(page, ws),
        ]

    cohort_outputs = [cohort_hd, workspace, workspace_info, version, version_info, import_open,
                      import_reason]

    def _select_report(reports: Mapping[str, Any], current: Any, job: Mapping[str, Any],
                       *, prefer_outcome: bool = False) -> Any:
        """The report to show: the one chosen, unless the poll's hand-off of
        a run's end prefers that run's report (`_after_job_change`), or the
        chosen one is gone (then the newest)."""
        runs = [x["run_id"] for x in reports.get("runs") or []]
        outcome = (job or {}).get("outcome") or {}
        if outcome.get("report_dir"):
            name = Path(str(outcome["report_dir"])).name
            if name in runs and (prefer_outcome or current not in runs):
                return name
        if current in runs:
            return current
        return runs[0] if runs else None

    # ------------------------------------------------------------------ the selection's order
    # Gradio reads a State when an event starts and writes it when the event
    # ends, and orders no two events of one session (each listener is its own
    # concurrency group by default, and a queue=False request runs beside
    # them), so an event that ended late would write back the selection it
    # started from over one chosen meanwhile. Every listener that writes the
    # selection, or Start and its reason drawn from it, is therefore in one
    # group of one: the page open, the workspace, the version, the model, the
    # measurement, a report's row, ↻, the read after an act or an import, and
    # the read after the poll saw the job or the GPU line change. They run first
    # asked, first run, each from what the one before it wrote, and the
    # browser takes a select's value when the event is asked. The group is
    # the server's, as the import's is: two sessions' events wait for each
    # other too.
    one_at_a_time = {"concurrency_id": SELECTION_CONCURRENCY, "concurrency_limit": 1}
    # In the group an event can wait as long as the one before it runs, and
    # with Gradio's defaults the browser does not send a listener's trigger
    # while its event waits: a click's or an input's ("once") is dropped; a
    # change's ("always_last") is sent once the wait ends, only the last, with
    # the values shown then (a select set back to the first pick by its
    # answer) and the first trigger's data. A second row or version chosen,
    # or a hand-off followed by another, would be lost; with "multiple" each
    # joins, and the last one asked is the one that stays.
    every_one = {"trigger_mode": "multiple"}

    # ------------------------------------------------------------------ page open
    def _read(ws, label, key, protocol, current_models, current_report, request):
        """Everything read again for the selection: the states
        (`state_outputs`) and the panels (`cohort_outputs`, `run_outputs`,
        `result_outputs`), apart, for the page's opening and the read after a
        Commit (`_after_commit`), which write the same outputs but for the
        hand-off between them."""
        page = read_page(request)
        ws = default_workspace(page, ws)
        binding = _binding_for(page, ws)
        label = default_version(page, binding, label)
        models = _models_for(page, current_models, ws, request)
        key = chosen_model(models, key)
        job = read_job()
        gpu = _gpu(page)
        reports, run_id = _reports(page, models, label, current_report, job, request)
        return ([page, models, binding, reports, run_id],
                [*_cohort_outputs(page, binding, ws, label),
                 *_run_outputs(page, models, binding, ws, label, key, protocol, job, gpu),
                 *_result_outputs(page, reports, models, run_id, job, binding, ws, label, key)])

    def _refresh(ws, label, key, protocol, current_models, current_report, handed,
                 request: gr.Request):
        """The page shown: everything read again for the selection (`_read`).
        A model another page handed over as it opened this one
        (`ctx.test_model`: Models' "Test this model ›", R-68) is the
        selection's workspace and checkpoint, chosen as a pick would (while it
        is listed), and the hand-off is cleared, so a later opening keeps the
        reader's choice."""
        if isinstance(handed, Mapping) and handed.get("workspace"):
            ws, key = handed["workspace"], handed.get("key")
        states, panels = _read(ws, label, key, protocol, current_models, current_report, request)
        return [*states, None, *panels]

    state_outputs = [page_state, models_state, binding_state_, reports_state, selected_report]
    ctx.on_page_open(PAGE, {
        "fn": _refresh,
        "inputs": [workspace, version, model_state, measurement_state, models_state,
                   selected_report, ctx.test_model],
        "outputs": [*state_outputs, ctx.test_model, *cohort_outputs, *run_outputs,
                    *result_outputs],
        "show_progress": "hidden",
        **one_at_a_time,
    })

    # ------------------------------------------------------------------ choices
    def _on_workspace(page, ws, label, key, protocol, current_models, current_report,
                      request: gr.Request):
        if not page:
            return [gr.update()] * (6 + len(cohort_outputs) + len(run_outputs) + len(result_outputs))
        binding = _binding_for(page, ws)
        label = default_version(page, binding, label)
        models = _models_for(page, current_models, ws, request)
        key = chosen_model(models, key)
        job = read_job()
        reports, run_id = _reports(page, models, label, current_report, job, request)
        # An import is made against one workspace: its drawer closes, its preview goes.
        return [models, binding, reports, run_id, None, gr.update(visible=False),
                *_cohort_outputs(page, binding, ws, label),
                *_run_outputs(page, models, binding, ws, label, key, protocol, job, _gpu(page)),
                *_result_outputs(page, reports, models, run_id, job, binding, ws, label, key)]

    picked(workspace)(_on_workspace,
                      inputs=[page_state, workspace, version, model_state, measurement_state,
                              models_state, selected_report],
                      outputs=[models_state, binding_state_, reports_state, selected_report,
                               preview_state, drawer, *cohort_outputs, *run_outputs,
                               *result_outputs],
                      show_progress="minimal", **one_at_a_time, **every_one) \
        .then(**ctx.chip_refresh)

    def _on_version(page, binding, models, ws, label, key, protocol, current_report,
                    request: gr.Request):
        if not page:
            return [gr.update()] * (4 + len(run_outputs) + len(result_outputs))
        job = read_job()
        reports, run_id = _reports(page, models, label, current_report, job, request)
        # The select too: an event asked before this one (a workspace's, a
        # page open) may have set it to the version it was asked with.
        return [reports, run_id,
                _dd(version_choices(page), label,
                    interactive=not page.get("registry_problem")),
                version_html(page, label, binding),
                *_run_outputs(page, models, binding, ws, label, key, protocol, job, _gpu(page)),
                *_result_outputs(page, reports, models, run_id, job, binding, ws, label, key)]

    picked(version)(_on_version,
                    inputs=[page_state, binding_state_, models_state, workspace, version,
                            model_state,
                            measurement_state, selected_report],
                    outputs=[reports_state, selected_report, version, version_info,
                             *run_outputs, *result_outputs],
                    show_progress="hidden", **one_at_a_time, **every_one)

    def _on_model(page, models, binding, ws, label, key, protocol, reports, run_id):
        """A checkpoint chosen: the Run panel for it, and the reports table and
        result said again against it (a report of the model chosen now reads
        "the selected model", any other its own model in full)."""
        if not page:
            return [gr.update()] * (len(run_outputs) + len(result_outputs))
        job = read_job()
        return [*_run_outputs(page, models, binding, ws, label, key, protocol, job, _gpu(page)),
                *_result_outputs(page, reports, models, run_id, job, binding, ws, label, key)]

    def _on_run(page, models, binding, ws, label, run, key, protocol, reports, run_id):
        """A run chosen: its checkpoint the shared pair chooses
        (`model_list.chosen_in_run`: the one chosen while it is this run's,
        else its latest save), then as a checkpoint chosen."""
        key = ml.chosen_in_run(model_parents(models or {}), run, key)
        return _on_model(page, models, binding, ws, label, key, protocol, reports, run_id)

    # In the selection's group: Start and its reason are written there only,
    # so a read handed over before this choice cannot write them back after it.
    model_inputs = [page_state, models_state, binding_state_, workspace, version, model,
                    measurement_state, reports_state, selected_report]
    picked(model)(_on_model, inputs=model_inputs, outputs=[*run_outputs, *result_outputs],
                  show_progress="hidden", **one_at_a_time, **every_one)
    picked(run_select)(_on_run, inputs=[*model_inputs[:5], run_select, model_state,
                                        *model_inputs[6:]],
                       outputs=[*run_outputs, *result_outputs], show_progress="hidden",
                       **one_at_a_time, **every_one)

    def _on_measurement(page, models, binding, ws, label, key, protocol):
        """A measurement picked: the radio (written back, as the version select
        is, over a value an event asked before this one wrote), Start and its
        reason for it, and the value the other listeners read
        (``measurement_state``). Of the selection's listeners, the one given
        the radio itself; Start takes it too, as the operator saw it."""
        if not page:
            return [gr.update()] * (2 + len(start_outputs))
        radio, protocol = _measurement_outputs(page, models, label, key, protocol)
        return [radio, *_start_outputs(page, models, binding, ws, label, key, protocol,
                                       _gpu(page)), protocol]

    # In the selection's group, for the same reason as the model's.
    measurement.input(_on_measurement,
                      inputs=[page_state, models_state, binding_state_, workspace, version,
                              model_state,
                              measurement],
                      outputs=[measurement, *start_outputs, measurement_state],
                      show_progress="hidden", **one_at_a_time, **every_one)

    def _on_model_reload(page, binding, ws, label, key, protocol, current_models, reports,
                         run_id, request: gr.Request):
        if not page:
            return [gr.update()] * (1 + len(run_outputs) + len(result_outputs))
        models = _models_for(page, current_models, ws, request, again=True)
        key = chosen_model(models, key)
        job = read_job()
        return [models, *_run_outputs(page, models, binding, ws, label, key, protocol, job,
                                      _gpu(page)),
                *_result_outputs(page, reports, models, run_id, job, binding, ws, label, key)]

    # In the selection's group: the listing it writes is the one the rows'
    # registration and Start are asked from. A second ↻ while one waits is
    # dropped (a click's "once"), which loses nothing: the one waiting lists
    # when it runs.
    model_reload.click(_on_model_reload,
                       inputs=[page_state, binding_state_, workspace, version, model_state,
                               measurement_state, models_state, reports_state, selected_report],
                       outputs=[models_state, *run_outputs, *result_outputs],
                       show_progress="minimal",
                       **one_at_a_time) \
        .then(**ctx.chip_refresh)

    def _on_report(page, reports, models, binding, ws, label, model_key, evt: gr.EventData,
                   request: gr.Request):
        raw = getattr(evt, "_data", None)
        key = raw.get("key") if isinstance(raw, Mapping) else None
        if not page or find(reports.get("runs") or [], "run_id", key) is None:
            return [gr.update()] * (2 + len(result_outputs))
        # The answer asked for this report stays with it when another is chosen.
        reports = refine_registration(page, reports, key, models, request=request)
        return [key, reports, *_result_outputs(page, reports, models, key, read_job(), binding,
                                               ws, label, model_key)]

    reports_list.click(_on_report, inputs=[page_state, reports_state, models_state,
                                           binding_state_, workspace, version, model_state],
                       outputs=[selected_report, reports_state, *result_outputs],
                       show_progress="hidden", **one_at_a_time, **every_one)

    # ------------------------------------------------------------------ the job
    def _job_read_again(page, models, current_report, reports, request, *, ended=None):
        """The selection read again in its group, after an act or the poll's
        hand-off: the job, then the reports of the version the session's
        reports were read for (the version shown: an event asked before this
        one may have changed it since this one was asked), and the report to
        show chosen again. `ended` is the run whose end the poll handed over
        (`_after_job_change` decides it, and nothing else does): its report is
        chosen, over the one chosen, while the job read now is still that run."""
        job = read_job()
        label = (reports or {}).get("label")
        prefer = ended is not None and job.get("job_id") == ended
        reports, run_id = _reports(page, models, label, current_report, job, request,
                                   prefer_outcome=prefer)
        return job, label, reports, run_id

    def _after_act(page, models, binding, ws, key, protocol, current_report, reports,
                   request: gr.Request):
        """The read after Start, Cancel, Register or a note: the job's title
        row and details show the job at once. It decides no end: an ended
        run's report is chosen by the poll's hand-off of that end, on its next
        tick."""
        job, label, reports, run_id = _job_read_again(page, models, current_report, reports,
                                                      request)
        gpu = _gpu(page)
        return [reports, run_id,
                *_start_outputs(page, models, binding, ws, label, key, protocol, gpu),
                *_job_outputs(page, job, gpu),
                *_result_outputs(page, reports, models, run_id, job, binding, ws, label, key)]

    after_outputs = [reports_state, selected_report, *start_outputs, *job_outputs,
                     *result_outputs]
    after_inputs = [page_state, models_state, binding_state_, workspace, model_state,
                    measurement_state,
                    selected_report, reports_state]

    def _on_start(page, models, ws, label, run, key, protocol, rec, det, request: gr.Request):
        """Start: for an operator's request (D8 is `start_test_run`'s first
        check), the Run select as this click's own request carries it
        (`run`) against the run of the checkpoint chosen in the session's
        listing (`shell.unread_choice`: a Run whose checkpoints have not
        landed is refused, never tested on the checkpoint of the run before,
        OPS-C-R2); then `start_test_run`."""
        roots = (page or {}).get("roots") or []
        item = find_model(models or {}, key)
        unread = operator_access(request).allowed and unread_choice(
            *([("Run", run, ml.item_run_key(item))] if item else []))
        if unread:
            return _outcome_box("amber", said(unread[:1].upper() + unread[1:],
                                              "Nothing was started."), roots)
        result = start_test_run(page, models, ws, label, key, rec, det, request,
                                protocol=protocol)
        if result.get("success"):
            return _outcome_box("ok", "The test run started. It checks the model and the cohort "
                                      "again on the bytes it reads before measuring.")
        return _outcome_box("amber", str(result.get("error") or "Nothing was started."), roots)

    # Start carries the Run select as the reader sees it (`_on_start`).
    start.click(_on_start, inputs=[page_state, models_state, workspace, version, run_select,
                                   model, measurement, record, deterministic],
                outputs=[action_out], show_progress="minimal") \
        .then(_after_act, inputs=after_inputs, outputs=after_outputs, show_progress="hidden",
              **one_at_a_time) \
        .then(**ctx.chip_refresh)

    def _on_cancel(page, request: gr.Request):
        result = cancel_test_run(request)
        roots = (page or {}).get("roots") or []
        if result.get("success"):
            summary = ((result.get("outcome") or {}).get("summary") or "")
            return _outcome_box("grey", f"The test run ended: {result.get('status')}. {summary}", roots)
        return _outcome_box("amber", str(result.get("error") or "Nothing was cancelled."), roots)

    cancel.click(_on_cancel, inputs=[page_state], outputs=[action_out], show_progress="minimal") \
        .then(_after_act, inputs=after_inputs, outputs=after_outputs, show_progress="hidden",
              **one_at_a_time) \
        .then(**ctx.chip_refresh)

    def _poll(location, page, seen, request: gr.Request):
        """The job, read again while this page is shown: what this session
        saw of the job (``job_state``, which only this poll writes, one tick
        after the other: a timer's tick is not sent while the one before is
        pending) and the hand-offs. It writes no piece: when the job or the
        GPU line (Start's first reason) is not what it saw on its last tick,
        it hands the read over to `_after_job_change`, naming what it saw then
        and what it sees now (so each change is handed over once); when the
        job's card (its title row, details and Cancel) is not the one it saw,
        to `_job_card_now` (a running job's elapsed time, every tick). Both
        run in the selection's group, so no card read before an act's is
        applied after it."""
        if location != PAGE or not page:
            return [gr.update()] * (len(poll_outputs) + 1)
        job = read_job()
        gpu = _gpu(page)
        seen = seen or {}
        card = json.dumps(_job_outputs(page, job, gpu), sort_keys=True, default=str)
        new_seen = {"job_id": job.get("job_id"), "status": job.get("status"), "gpu": list(gpu),
                    "card": card}
        moved = {k: new_seen[k] for k in ("job_id", "status", "gpu")} != {
            k: seen.get(k) for k in ("job_id", "status", "gpu")}
        # Each hand-off is a new value (its own time), so its .change always fires.
        handoff = ({"seen": {"job_id": seen.get("job_id"), "status": seen.get("status")},
                    "now": {"job_id": job.get("job_id"), "status": job.get("status")},
                    "at": time.monotonic_ns()} if moved else gr.update())
        redraw = {"at": time.monotonic_ns()} if card != seen.get("card") else gr.update()
        # The GPU line is the header's GPU chip read now: when it is not
        # what this poll saw last, the chips are read again on this tick.
        return [new_seen, redraw, handoff, chips_on_change(seen.get("gpu"), new_seen["gpu"],
                                                           request)]

    poll_outputs = [job_state, card_changed, job_changed]
    poll_inputs = [ctx.location, page_state, job_state]

    # The slow timer only: a listener on a shell timer sends a request on every
    # tick whatever it then does, so one on the fast timer would cost every open
    # page 40 requests a minute even with no job. A test run's state and elapsed
    # time are read every 5 s; Start, Cancel and Register read it at once.
    ctx.timer_slow.tick(_poll, inputs=poll_inputs, outputs=[*poll_outputs, ctx.chips],
                        queue=False, show_progress="hidden")

    def _job_card_now(page, _change, request: gr.Request):
        """The poll's card hand-off, in the selection's group: the job's
        title row, details and Cancel read again now (`_job_outputs`)."""
        if not page:
            return [gr.update()] * len(job_outputs)
        return _job_outputs(page, read_job(), _gpu(page))

    card_changed.change(_job_card_now, inputs=[page_state, card_changed], outputs=job_outputs,
                        show_progress="hidden", **one_at_a_time)

    def _after_job_change(page, models, binding, ws, key, protocol, current_report, reports,
                          change, request: gr.Request):
        """The poll's hand-off, in the selection's group: Start and its reason
        from the selection as it is now, and the reports read again. The one
        place an end is decided, from what the poll saw on its last tick
        (``seen``) and sees now (``now``): the run read now is no longer busy,
        and the poll saw it busy, or saw another job or none (``idle``) — a
        run that started and ended between two ticks, such as a short one
        started here (the act's read after Start writes nothing the poll
        reads). That run's report is chosen. A session's first tick (nothing
        seen yet) decides no end."""
        if not page or not isinstance(change, Mapping):
            return [gr.update()] * (2 + len(start_outputs) + len(result_outputs))
        seen, now = change.get("seen") or {}, change.get("now") or {}
        saw_any = seen.get("job_id") is not None or seen.get("status") is not None
        started_since = saw_any and now.get("job_id") not in (None, seen.get("job_id"))
        ended = (now.get("job_id") if (seen.get("status") in _BUSY or started_since)
                 and now.get("status") not in _BUSY else None)
        job, label, reports, run_id = _job_read_again(page, models, current_report, reports,
                                                      request, ended=ended)
        return [reports, run_id,
                *_start_outputs(page, models, binding, ws, label, key, protocol, _gpu(page)),
                *_result_outputs(page, reports, models, run_id, job, binding, ws, label, key)]

    job_changed.change(_after_job_change,
                       inputs=[page_state, models_state, binding_state_, workspace, model_state,
                               measurement_state, selected_report, reports_state, job_changed],
                       outputs=[reports_state, selected_report, *start_outputs, *result_outputs],
                       show_progress="hidden", **one_at_a_time, **every_one)

    # ------------------------------------------------------------------ register and notes
    def _on_register(page, reports, models, ws, run_id, request: gr.Request):
        done, said = register_selected(page, reports, models, ws, run_id, request)
        return _outcome_box("ok" if done else "amber", said, (page or {}).get("roots") or [])

    register.click(_on_register, inputs=[page_state, reports_state, models_state, workspace,
                                         selected_report],
                   outputs=[register_out], show_progress="minimal") \
        .then(_after_act, inputs=after_inputs, outputs=after_outputs, show_progress="hidden",
              **one_at_a_time) \
        .then(**ctx.chip_refresh)
    # A message about one report goes when another is chosen.
    selected_report.change(lambda _run: ("", ""), inputs=[selected_report],
                           outputs=[register_out, note_out], show_progress="hidden")

    def _on_note(page, reports, run_id, who, what, request: gr.Request):
        done, said = add_note(reports, run_id, who, what, request)
        box = _outcome_box("ok" if done else "amber", said, (page or {}).get("roots") or [])
        return box, (gr.update(value="") if done else gr.update())

    note_btn.click(_on_note, inputs=[page_state, reports_state, selected_report, note_who,
                                     note_what],
                   outputs=[note_out, note_what], show_progress="minimal") \
        .then(_after_act, inputs=after_inputs, outputs=after_outputs, show_progress="hidden",
              **one_at_a_time) \
        .then(**ctx.chip_refresh)

    # ------------------------------------------------------------------ import
    def _on_open_import(page, ws, request: gr.Request):
        listing = open_import(request)
        roots = (page or {}).get("roots") or []
        if listing["refused"]:
            return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), _outcome_box("amber", listing["refused"]), gr.update(),
                    gr.update())
        return (gr.update(visible=True), _dd(intake_choices(listing), None),
                intake_note_html(listing, roots), IMPORT_TITLE_HTML,
                gr.update(choices=role_choices(page or {})), None,
                gr.update(interactive=False, value="Commit"), "", "", "")

    import_open.click(_on_open_import, inputs=[page_state, workspace],
                      outputs=[drawer, intake, intake_note, drawer_title, new_role, preview_state,
                               commit_btn, import_msg, preview_out, commit_out],
                      show_progress="minimal") \
        .then(**ctx.chip_refresh) \
        .then(fn=None, inputs=None, outputs=None, js=_IMPORT_FOCUS_JS)

    import_close.click(lambda: (gr.update(visible=False), None, "", gr.update(interactive=False),
                                ""),
                       outputs=[drawer, preview_state, preview_out, commit_btn, commit_out],
                       show_progress="hidden")

    new_role.change(lambda role: _reason_html(role_line(role)), inputs=[new_role],
                    outputs=[role_note], show_progress="hidden")

    def _on_preview(page, ws, file, label, role, policy, request: gr.Request):
        preview, refusal = preview_import(ws, file, label, role, policy, request)
        roots = (page or {}).get("roots") or []
        if preview is None:
            return None, _outcome_box("amber", refusal or "Nothing was written.", roots), \
                gr.update(interactive=False, value="Commit"), ""
        can = bool(preview.get("samples_n"))
        return preview, preview_html(preview), gr.update(
            interactive=can, value=f"Commit as version {preview.get('label')}"), ""

    preview_btn.click(_on_preview, inputs=[page_state, workspace, intake, new_label, new_role,
                                           new_policy],
                      outputs=[preview_state, preview_out, commit_btn, commit_out],
                      show_progress="minimal",
                      concurrency_id="shep-test-import", concurrency_limit=1) \
        .then(**ctx.chip_refresh)

    def _on_commit(page, preview, ws, file, label, role, policy, request: gr.Request):
        done, said, committed = commit_import_ui(preview, ws, file, label, role, policy, request)
        roots = (page or {}).get("roots") or []
        box = _outcome_box("ok" if done else "amber", said, roots)
        if not done:
            return box, gr.update(), gr.update(), None, gr.update(), gr.update()
        # Done: the drawer closes and the new version is chosen; the message
        # stays under "Import a new version…".
        return ("", None, gr.update(interactive=False, value="Commit"), committed,
                gr.update(visible=False), box)

    def _after_commit(committed, ws, label, key, protocol, current_models, current_report,
                      request: gr.Request):
        """The page read again, as a page open reads it (`_read`), with the
        version just committed chosen. A commit hands no model over, so it
        neither reads nor clears the opening's hand-off (`ctx.test_model`)."""
        states, panels = _read(ws, committed or label, key, protocol, current_models,
                               current_report, request)
        return [*states, *panels]

    committed_label = gr.State(None)
    commit_btn.click(_on_commit, inputs=[page_state, preview_state, workspace, intake, new_label,
                                         new_role, new_policy],
                     outputs=[commit_out, preview_state, commit_btn, committed_label, drawer,
                              import_msg],
                     show_progress="minimal", concurrency_id="shep-test-import",
                     concurrency_limit=1) \
        .then(_after_commit, inputs=[committed_label, workspace, version, model_state,
                                     measurement_state, models_state, selected_report],
              outputs=[*state_outputs, *cohort_outputs, *run_outputs, *result_outputs],
              show_progress="hidden", **one_at_a_time) \
        .then(**ctx.chip_refresh)


_PAGE = "#" + page_elem_id(PAGE)

#: This page's style sheet (the page contract: listed in `src/webui/app.py`'s
#: ``PAGE_CSS``), scoped to its own ids. A report's record is the one record
#: rendering's (`renderers.record_html`), styled once in `src/webui/style.py`.
#: ``.shep-test-replist``: the reports, a compact single-select table at the
#: Result panel's top (32-40 px rows, the skill's §8), in the panel's one scroll
#: region with the result under it; its header does not stick (the region
#: scrolls the table away under the result, and nothing sticky covers a row).
#: **The three panels are sized by what they hold** (``.shep-sized``, the
#: sweep's M4): a basis and a growth in rem, never a share fitted to one
#: window, and a floor (``--shep-panel-min``, the layout rule's measure,
#: `src/webui/shell.py`): the version's identity line, the two checkboxes on
#: one line, the reports' table. While the Reports panel draws no table (an
#: empty state: first use, no version, no report yet) its floor is its empty
#: state's and it grows less, so Data and Run take the room it would leave
#: blank (the sweep's B-04: a panel scrolled while its neighbour stood blank).
PAGE_CSS = f"""
{_PAGE} #shep-test-row {{ position: relative; gap: 12px; }}
{_PAGE} .shep-test-bd {{ padding: var(--shep-space-3) var(--shep-inset) !important; gap: 10px !important; }}
{_PAGE} .shep-test-bd > *, {_PAGE} .shep-test-col > *, {_PAGE} .shep-test-drawer > * {{ flex-shrink: 0 !important; }}
{_PAGE} .shep-test-bd-flush {{ padding: 0 !important; }}
{_PAGE} .shep-test-pad {{ padding: var(--shep-space-3) var(--shep-inset); display: flex; flex-direction: column; gap: 10px; }}
{_PAGE} .shep-test-reason {{ margin: 2px 0; }}
{_PAGE} .shep-test-actions {{ gap: 10px !important; align-items: center !important; flex-wrap: wrap !important; }}
{_PAGE} .shep-test-actions > button {{ flex: none !important; }}
{_PAGE} .shep-test-ident {{ font-size: 15px; line-height: 1.5; overflow-wrap: anywhere; }}
{_PAGE} .shep-test-ident > div {{ margin: 1px 0; }}
{_PAGE} #shep-test-model-why:not(:has(.shep-reason)) {{ display: none !important; }}
{_PAGE} .shep-test-checks, {_PAGE} .shep-test-checks > .form {{ gap: 4px 18px !important; flex-wrap: wrap !important; }}
{_PAGE} .shep-test-checks > .form {{ background: none !important; border: 0 !important; box-shadow: none !important; }}
{_PAGE} .shep-test-checks > .form > .block {{ flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; margin: 0 !important; }}
{_PAGE} .shep-test-checks label > span {{ white-space: nowrap; }}
{_PAGE} .shep-test-replist {{ padding: var(--shep-space-3) var(--shep-inset) 0; }}
{_PAGE} .shep-test-repwrap {{ margin: 0 !important; overflow: visible; }}
{_PAGE} .shep-test-reps th {{ position: static; }}
{_PAGE} .shep-test-reps td {{ padding: 7px 10px; overflow-wrap: anywhere; }}
{_PAGE} .shep-test-reps td.shep-test-rep-name {{ white-space: nowrap; }}
{_PAGE} .shep-test-reps tr[aria-disabled="true"] td {{ cursor: default; background: var(--shep-panel-2); }}
{_PAGE} .shep-test-res-hd {{ display: flex; align-items: baseline; gap: 10px; }}
{_PAGE} .shep-test-res-hd h3 {{ margin: 0; font-size: 15px; font-weight: 600; }}
{_PAGE} .shep-test-vd-why {{ margin-top: 6px; }}
{_PAGE} .shep-test-job {{ border: 1px solid var(--shep-border); border-radius: 10px; background: var(--shep-panel-2); padding: 0 0 8px; display: flex; flex-direction: column; gap: 4px; }}
{_PAGE} .shep-test-job-top {{ align-items: center !important; }}
{_PAGE} #shep-test-job-head {{ --shep-grp-basis: 12rem; min-width: 0; }}
{_PAGE} #shep-test-cancel {{ flex: none !important; white-space: nowrap; }}
{_PAGE} .shep-test-job-hd {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
{_PAGE} .shep-test-job-hd h3, {_PAGE} .shep-test-job-title {{ margin: 0; font-size: 15px; font-weight: 600; }}
{_PAGE} .shep-test-job-title {{ padding: 10px 14px 4px; }}
{_PAGE} .shep-test-job-bd {{ padding: 0 14px; display: flex; flex-direction: column; gap: 6px; overflow-wrap: anywhere; }}
{_PAGE} .shep-test-job-ft {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; padding: 8px 14px 0; }}
{_PAGE} .shep-test-log {{ margin: 0; padding: 8px 10px; background: #fff; border: 1px solid var(--shep-border); border-radius: 6px; font-size: 14px; white-space: pre-wrap; overflow-wrap: anywhere; }}
{_PAGE} .shep-test-fail {{ color: var(--shep-red-700); font-size: 15px; }}
{_PAGE} .shep-test-units {{ font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} .shep-test-kgempty {{ display: flex; flex-direction: column; gap: var(--shep-space-1); margin-top: var(--shep-space-2); }}
{_PAGE} .shep-test-kgempty .shep-kv {{ font-size: 14px; }}
{_PAGE} .shep-test-ref {{ margin: var(--shep-space-1) 0 0; font-size: 14px; color: var(--shep-muted); }}
{_PAGE} .shep-test-units > summary {{ cursor: pointer; min-height: 24px; line-height: 24px; width: max-content; max-width: 100%; }}
{_PAGE} .shep-test-outcome {{ display: flex; flex-direction: column; gap: 6px; font-size: 15px; }}
{_PAGE} .shep-test-chips {{ display: flex; flex-wrap: wrap; gap: 6px 8px; }}
{_PAGE} .shep-test-noteform {{ padding: 10px 16px 8px !important; gap: 6px !important; border-top: 1px solid var(--shep-border); }}
{_PAGE} .shep-test-noterow {{ gap: 10px !important; flex-wrap: wrap !important; align-items: flex-start !important; }}
{_PAGE} .shep-test-noteform .shep-test-actions > .block {{ flex: 1 1 200px !important; min-width: 0 !important; }}
{_PAGE} .shep-test-terms {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(13.75rem, 100%), 1fr)); gap: 1px 12px; font-size: 14px; margin-top: 2px; }}
{_PAGE} .shep-test-preview .shep-kv {{ padding: 10px 12px; overflow-wrap: anywhere; }}
{_PAGE} .shep-test-preview .shep-notice, {_PAGE} .shep-test-preview .shep-reason {{ margin: 0 12px 10px; }}
{_PAGE} .shep-test-drawer {{ box-shadow: -8px 0 24px rgba(20, 24, 40, .14); }}
{_PAGE} .shep-test-drawer-hd {{ flex-wrap: nowrap !important; align-items: flex-start !important; }}
{_PAGE} .shep-test-drawer-hd > .block {{ flex: 1 1 auto !important; min-width: 0; }}
{_PAGE} #shep-test-import-title h2 {{ margin: 0; font-size: var(--shep-title-size); font-weight: 600; }}
{_PAGE} #shep-test-import-title .shep-sub {{ margin: var(--shep-space-head-below) 0 0; line-height: var(--shep-lh-prose); }}
{_PAGE} #shep-test-import-close {{ flex: none !important; width: auto !important; min-width: max-content !important; white-space: nowrap; }}
{_PAGE} .shep-test-form {{ gap: 10px !important; flex-wrap: wrap !important; }}
{_PAGE} .shep-test-form > * {{ min-width: min(11.25rem, 100%) !important; }}
{_PAGE} #shep-test-start {{ width: 100%; }}
{_PAGE} .shep-test-runft {{ padding: 10px 16px 12px !important; gap: 10px !important; border-top: 1px solid var(--shep-border); }}
{_PAGE} .shep-test-select input {{ padding-right: 26px !important; }}
{_PAGE} .shep-test-rep-old, {_PAGE} .shep-test-more-fewer {{ display: none !important; }}
{_PAGE} .shep-test-all .shep-test-rep-old {{ display: table-row !important; }}
{_PAGE} .shep-test-all .shep-test-more-fewer {{ display: inline-flex !important; }}
{_PAGE} .shep-test-all .shep-test-more-all {{ display: none !important; }}
{_PAGE} .shep-test-more {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 8px 0 0; }}
{_PAGE} .shep-test-more button {{ font: inherit; font-size: 15px; color: var(--shep-accent-text); background: none; border: 0; padding: 2px 0; cursor: pointer; text-decoration: underline; text-underline-offset: 2px; }}
{_PAGE} .shep-panel-ft > button, {_PAGE} .shep-test-actions > button {{ white-space: nowrap; flex: none !important; }}
{_PAGE} .shep-panel-ft > .block:not(.shep-beside) {{ flex: 1 1 200px !important; min-width: 0 !important; }}
{_PAGE} #shep-test-cohort {{ --shep-panel-basis: 22rem; --shep-panel-grow: 1; --shep-panel-min: 19.7rem; }}
{_PAGE} #shep-test-run {{ --shep-panel-basis: 24rem; --shep-panel-grow: 1; --shep-panel-min: 22.5rem; }}
{_PAGE} #shep-test-result {{ --shep-panel-basis: 32.2rem; --shep-panel-grow: 1.5; --shep-panel-min: 32.2rem; }}
{_PAGE} #shep-test-result:not(:has(.shep-test-reps)) {{ --shep-panel-basis: 20rem; --shep-panel-grow: 0.5; --shep-panel-min: 20rem; }}
{SIDE} {{
  {_PAGE} .shep-test-col > .shep-test-bd {{ flex: 1 1 auto !important; min-height: 0; overflow: auto !important; flex-wrap: nowrap !important; }}
  {_PAGE} #shep-test-noteform {{ max-height: 50%; overflow: auto !important; flex-wrap: nowrap !important; }}
  {_PAGE} #shep-test-runft {{ flex: none !important; }}
  {_PAGE} .shep-test-drawer {{ position: absolute !important; top: 0; right: 0; bottom: 0; width: min(48rem, 62%) !important; z-index: 50; display: flex; flex-direction: column; max-height: 100%; }}
  {_PAGE} .shep-test-drawer > .shep-test-bd {{ flex: 1 1 auto !important; min-height: 0; overflow: auto !important; flex-wrap: nowrap !important; }}
}}
{STACK} {{
  {_PAGE} .shep-test-drawer {{ position: fixed !important; top: 0; right: 0; bottom: 0; width: min(40rem, 100%) !important; max-width: 100% !important; z-index: 60; display: flex; flex-direction: column; flex-wrap: nowrap !important; border-radius: 0 !important; }}
  {_PAGE} .shep-test-drawer > .shep-test-bd {{ flex: 1 1 auto !important; min-height: 0; overflow: auto !important; flex-wrap: nowrap !important; }}
}}
"""


__all__ = [
    "BATCH_SIZE_SHOWN",
    "FIELD_HELP",
    "PAGE_CSS",
    "TOPIC_HELP",
    "add_note",
    "binding_state",
    "cancel_test_run",
    "commit_import_ui",
    "create_test_page",
    "job_body_html",
    "job_head_html",
    "job_html",
    "open_import",
    "preview_html",
    "preview_import",
    "read_models",
    "read_page",
    "read_reports",
    "redact",
    "register_selected",
    "register_state",
    "reports_html",
    "result_html",
    "role_line",
    "start_state",
    "start_test_run",
]
