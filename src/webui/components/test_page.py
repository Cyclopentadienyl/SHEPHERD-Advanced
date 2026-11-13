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
  given none and does not need the setting), "Register the result" and
  "Deterministic execution", the batch size and device the server fixes (D5),
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
  needing genes the version does not list are refused with their reason; then the job: its state, elapsed time,
  inputs, output tail, "Cancel" (the runner's stop for test runs only), and
  once it has ended, **what it left, from the runner** (its outcome file, kept
  and registered read from disk by WP4's helper; "no report was published"
  when there is no outcome file; a cancel says what was left);
- **Result**: at its top, the version's report store as a **compact
  single-select table** (`reports_html`), newest first (the newest
  `ROWS_SHOWN` and the selected one shown, "Show all N reports" for the
  rest): each run's time, its model by its relation to the chosen one ("the
  selected model", or the other file's title or fingerprint), its
  measurement and its state (kept, registered by the registration's one
  helper, `server_lists.stored_report_state`, asked again for an operator
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
  coverage with every exclusion, each relation a row of its own
  (`renderers.relation_rows`), R10 in the
  caveats' words (*unverifiable* is a word, never a 0), environment, dataset
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
The poll writes the job card and what the session saw of the job
(``job_state``, which nothing else writes), never a select, the selection or
what is drawn from it (Start and its reason included). Nothing here is module
state.
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

import logging
import re
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import gradio as gr

from src.evaluation import protocols
from src.evaluation.caveats import (
    MEASUREMENT_NAMES,
    R10_LABEL_TEXT,
    TASK_NAMES,
    label_words,
    metric_text,
    mode_label,
    mode_note,
    recorded_metric_lines,
    version_words,
)
from src.utils.run_environment import native_identity_lines
from src.webui.components import renderers as r
from src.webui.components.chips import LOCK, chips_on_change, gated
from src.webui.components.chips import operator_only as _operator_only
from src.webui.components.help import help_button
from src.webui.components import model_list as ml
from src.webui.components.model_list import item_key
from src.utils.path_redaction import redact_paths
from src.webui.components.workspaces import (
    binding_state,
    version_row,
    workspace_binding,
)
from src.webui.operator_access import operator_access
from src.webui.shell import page_elem_id, plain_html

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

#: What replaces an absolute path that is under no named server directory.
SERVER_PATH = "a server path"
#: The investigation note's fields are never narrower (px): "Who" holds a
#: normal name whole.
NOTE_FIELD_MIN_WIDTH = 240

#: The lost-terms policy, as the select names it.
POLICY_CHOICES = [("keep case", "keep"),
                  ("exclude case", "exclude")]

#: The words the dataset origin is shown with.
_ORIGIN_MEANING = {
    "synthetic": "a generated stand-in, not a clinical knowledge graph",
    "files": "built from identified source files",
}
#: The metrics shown as tiles, in order, by the record's mode (`scoring.mode`),
#: for a measurement this build describes; each is named by
#: `caveats.metric_text`. A task's (`src.evaluation.task_measurement`) are
#: pooled over its positive pairs or a mean over its ranked cases, and its
#: coverage. A record of any other mode has no tiles: its metrics as recorded
#: are its main result (`_record_html`).
_TILES = {
    "disease": ("pooled_mrr", "pooled_top1", "pooled_top10", "coverage"),
    "gene": ("pooled_mrr", "pooled_top1", "macro_recall@5", "coverage"),
    "patients_like_me": ("pooled_mrr", "pooled_top10", "macro_recall@10", "coverage"),
}
#: A task record's metric families that take a cut-off k, each shown as one
#: row (its k values in order), and its single figures shown as a row each.
_TASK_FAMILIES = ("pooled_top", "macro_recall@", "macro_precision@")
_TASK_ROWS = ("macro_mrr", "best_positive_mrr")
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
FIELD_HELP = {
    "shep-test-workspace": (
        "Workspace",
        "The workspace the server lists under SHEPHERD_WORKSPACES_ROOT: its graph is what a "
        "data version is imported against and what a test run measures on; its checkpoints "
        "are the models offered. Shown by its dataset fingerprint, as on Datasets.",
    ),
    "shep-test-version": (
        "Data version",
        "A version in the server's registry (SHEPHERD_REGISTRY), listed by kind and role.\n"
        "• A supplied cohort: source cases mapped to one graph's nodes (Import).\n"
        "• A KG edge set: one part of a model's edge split, with fixed pairs.\n"
        "• A generated split: the workspace's own patients.\n"
        "Immutable, identified by its cohort.json's digest. Measured only on the workspace "
        "whose graph it was mapped against; the run checks the bytes again.",
    ),
    "shep-test-model-run": (
        "Run",
        "The workspace's training runs, latest first, each by its name or its start time: "
        "the runs whose checkpoints the server lists.\n"
        "• Choosing a run offers its checkpoints and chooses its latest save.\n"
        "• Listing opens every checkpoint, so it is an operator control.",
    ),
    "shep-test-model": (
        "Checkpoint",
        "The chosen run's checkpoints, each by what the file is and how far it trained.\n"
        "• What its record says it is shows under it.\n"
        "• A model is measured by what its capabilities meet: its task head, or its LP head "
        "on a KG edge set (Measurement); a file without a model spec is listed only.\n"
        "• The run is given the model fingerprint shown here and refuses before measuring if "
        "the file it reads holds other bytes.",
    ),
    "shep-test-measurement": (
        "Measurement",
        "What the run measures: one of the measurements this build implements for the chosen "
        "model.\n"
        "• The options are those the model's capabilities meet that measure the chosen "
        "version's kind: a task's on a cohort or generated split, KG link prediction on a KG "
        "edge set.\n"
        "• A base is offered KG link prediction only.\n"
        "• A single option is chosen already; of several, choose one. Each Start runs one.\n"
        "• The report records the measurement as its mode.",
    ),
    "shep-test-record": (
        "Register the result",
        "Checked: when the run completes, its report is registered in the ledger beside the "
        "checkpoint automatically. Unchecked: the report is kept in the registry's report "
        "store and not registered; it can be registered later from this page, as a whole "
        "report, whatever the version's kind. Every run keeps its report either way (D5).",
    ),
    "shep-test-deterministic": (
        "Deterministic execution",
        "Runs with PyTorch's deterministic algorithms, cuDNN deterministic and its benchmark "
        "off. An operation with no deterministic implementation refuses the run; it never "
        "falls back. It is recorded, so deterministic runs are their own condition group: "
        "they are expected to agree, and a difference between them asks for an "
        "investigation. It guards against known non-deterministic operations; it does not "
        "certify a number.",
    ),
    "shep-test-intake": (
        "File in the intake directory (server)",
        "The shepherd-cases/1 files (*.json) the operator placed in the server's intake "
        "directory (SHEPHERD_INTAKE_DIR), outside the repository. The server finds the chosen "
        "name in that directory again and refuses anything else, a link leaving it included. "
        "Nothing is uploaded from the browser.",
    ),
    "shep-test-label": (
        "Version label",
        "The new version's name in the registry: lowercase letters, digits, '-' and '_'. "
        "Versions are immutable, so a label is used once.",
    ),
    "shep-test-role": (
        "Role",
        "The cohort version's declared purpose, for example mygene2, research_selection or "
        "institutional_acceptance.",
    ),
    "shep-test-partial": (
        "Lost terms",
        "What a case that loses some, but not all, of its phenotype terms in mapping "
        "becomes: ranked on the terms that map, or excluded. Recorded in the version.",
    ),
    "shep-test-note-who": (
        "Who",
        "Your name as you want it recorded. Under deployment mode M1 nothing verifies it.",
    ),
    "shep-test-note-what": (
        "What was found",
        "What the investigation found. The note is kept in the report store with this "
        "condition group's reports; it deletes no report and changes nothing.",
    ),
}
#: Help a rendered ⓘ names.
TOPIC_HELP = {
    "test-preview-id": (
        "Preview id",
        "The first 16 hex digits of the SHA-256 of the would-be version manifest: the source "
        "bytes and file name, the graph's digests, the dataset record, the label, the role, "
        "the policy and every count. Commit prepares again from the same inputs and refuses "
        "unless it gives the same id, so what is committed is what was previewed.",
    ),
    **{_MODE_HELP[mode]: (f"{words} result", mode_note(mode))
       for mode, words in _RESULT_WORDS.items()},
    "test-r10": (
        "R10 labels",
        " · ".join(f"{text['name']}: {text['definition']}"
                   for labels in R10_LABEL_TEXT.values() for text in labels.values()),
    ),
    "test-fixed": (
        "Fixed by the server",
        "A test run has no reason to vary these, so the job runner fixes them (D5).\n"
        "• Batch size: the measurement's default.\n"
        "• Device: the deployment's (SHEPHERD_DEVICE), with no fallback.\n"
        "The measurement and deterministic execution are chosen here; both are recorded.",
    ),
    "test-evidence": (
        "R10 evidence",
        "Where a run on a supplied cohort looks for the inputs the checkpoint records (R10).\n"
        "• It counts their overlap with the cohort's cases: the measured workspace always, and "
        "the directories SHEPHERD_TRAINING_EVIDENCE_DIRS names.\n"
        "• Only bytes whose digest the checkpoint records count; otherwise a label is "
        "unverifiable, never 0.\n"
        "• A KG edge set or a generated split has no R10: its run is given no directory, and "
        "the setting is not needed.",
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


def path_roots(environ: Optional[Mapping[str, str]] = None) -> List[Tuple[str, str]]:
    """The server directories a shown text may name, with the name each is
    shown under, longest first: the registry, the workspaces root, the intake
    directory, each evidence directory and the repository."""
    from src.config import server_settings

    roots: List[Tuple[str, str]] = []
    for reader, token in ((server_settings.cohort_registry, "<registry>"),
                          (server_settings.workspaces_root, "<workspaces>"),
                          (server_settings.intake_directory, "<intake>")):
        try:
            setting = reader(environ)
        except Exception:  # noqa: BLE001 - a root not read is a path shown as "a server path"
            continue
        if setting.available:
            roots.append((str(setting.value), token))
    try:
        evidence = server_settings.training_evidence_dirs(environ)
        for index, directory in enumerate(evidence.value or (), start=1):
            roots.append((str(directory), f"<evidence {index}>"))
    except Exception:  # noqa: BLE001
        pass
    roots.append((str(server_settings.REPO_ROOT), "<repository>"))
    return sorted(roots, key=lambda pair: -len(pair[0]))


def redact(text: Any, roots: Sequence[Tuple[str, str]]) -> str:
    """`text` with no absolute server path left: one under a named server
    directory is shown under its name (``<registry>/versions/…``), any other
    as "a server path" (the dashboard's one redaction, `redact_paths`)."""
    return redact_paths(text, roots, other=SERVER_PATH)


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
    if not text:
        return ""
    lock = f'<span aria-hidden="true">{LOCK}</span>' if kind == "lock" else ""
    return f'<div class="shep-reason shep-test-reason">{lock}<span>{e(text)}</span></div>'


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


#: A check mark that stays on its line (Gradio's base style makes an svg a block).


def said(text: Any, tail: str) -> str:
    """`text` as a sentence, then `tail` ("Nothing was started.")."""
    body = " ".join(str(text or "").split()).rstrip()
    if body and not body.endswith((".", "!", "?")):
        body += "."
    return f"{body} {tail}".strip()


def _count(value: Any) -> str:
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else "—"


def _number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{value:.4f}"


# ---------------------------------------------------------------------------
# Reading what the page shows (one session's data, plain JSON for gr.State)
# ---------------------------------------------------------------------------
def read_page(request: Any, *, environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The page's server lists for one session: D8's answer for this request,
    the workspaces (the dashboard's one reader), the serving workspace, the
    registry's cohort versions (their manifests only), the evidence setting and
    the device. Opens no checkpoint."""
    from src.api.services.server_lists import list_cohort_versions
    from src.config import server_settings
    from src.webui.components.served_status import read_status, serving_workspace
    from src.webui.components.workspaces import read_workspaces

    roots = path_roots(environ)
    listed = read_workspaces(environ)
    status, _problem = read_status()
    serving = (serving_workspace(status, listed["rows"], directories=listed["directories"])
               if status is not None else None)
    registry = server_settings.cohort_registry(environ)
    versions = list_cohort_versions(environ=environ) if registry.available else None
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
def cohort_header_html(page: Mapping[str, Any]) -> str:
    right = ("registry: server setting" if not page.get("registry_problem")
             else "registry: not available")
    head = (f'<div class="shep-panel-hd"><h2>Data</h2><span class="shep-panel-right">'
            f'{e(right)}</span></div>')
    problem = page.get("registry_problem")
    if problem:
        head += (f'<div class="shep-test-pad">{_reason_html(redact(problem, page.get("roots") or []))}'
                 '</div>')
    return head


def workspace_html(page: Mapping[str, Any], workspace: Optional[str]) -> str:
    row = find(page.get("workspaces") or [], "name", workspace)
    if row is None:
        why = page.get("workspaces_problem") or "No workspace is chosen."
        return _reason_html(why)
    digits = r.distinguishing_digits(w.get("dataset_fingerprint")
                                     for w in page.get("workspaces") or [])
    note = (' <span class="shep-mark shep-mark-serve">serving reads this workspace</span>'
            if page.get("serving") == workspace else "")
    return (f'<div class="shep-test-ident">{r.dataset_identity_html(row, digits=digits, name=False)}'
            f'{note}</div>')


#: The import command a page names when the registry holds no KG edge set:
#: a KG edge set is made on the machine, from a checkpoint's recorded split
#: (`scripts/import_cohort.py --kind kg_edges`; the deployment guide §4.9),
#: never from the browser. Its role is the part asked, train, val or test.
KG_EDGES_IMPORT = "scripts/import_cohort.py --kind kg_edges --from-checkpoint <checkpoint>"


def kg_edges_empty_html(page: Mapping[str, Any]) -> str:
    """The empty state of the KG edge sets (the skill's §9): when the registry
    lists none, one line saying what a KG edge set is for and the command
    that makes one on the machine (`KG_EDGES_IMPORT`); "" when it lists one,
    or the registry is not available."""
    if page.get("registry_problem") or any(
            v.get("kind") == protocols.KG_EDGES for v in page.get("versions") or []):
        return ""
    return ('<div class="shep-sub">No KG edge set in the registry yet: KG link prediction '
            f"measures one. Import one on the machine with <code>{e(KG_EDGES_IMPORT)}</code> "
            "(deployment guide §4.9).</div>")


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
    what format. A supplied cohort by its role and its case counts; a KG
    edge set or a generated split by its counts in its own units
    (`_version_counts`), never in a supplied cohort's words: its kind and
    role are its option's (`version_choices`), not repeated under it (skill
    §1.2). Under it, the KG edge sets' empty state when the registry holds
    none (`kg_edges_empty_html`)."""
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
    # mark; mapped against another graph a condition with ▲ and, under the
    # list, the refusal it brings (`ANOTHER_GRAPH`), beside the version;
    # not checked a hollow one.
    mark = r.status_mark({"bound": "ok", "another": "warn", "unknown": "neutral"}[state],
                         words[:1].upper() + words[1:])
    excluded = counts.get("excluded") if isinstance(counts.get("excluded"), Mapping) else {}
    reasons = "; ".join(f"{str(k).replace('_', ' ')} {_count(v)}" for k, v in excluded.items())
    imported = str(version.get("imported_at") or "").replace("T", " ").replace("Z", " UTC")
    if _kind_words(version):
        rows = [("Holds", _version_counts(version))]
    else:
        rows = [("Role", r.ident_html(version.get("role"))),
                ("Cases", e(f'{_count(counts.get("source_cases"))} source cases, '
                            f'{_count(counts.get("ranked_eligible"))} ranked-eligible'
                            + (f" ({reasons} excluded)" if reasons else "")))]
    fingerprint = r.fingerprint_value(version.get("manifest_sha256"), what="version fingerprint")
    rows.append(("Version fingerprint", fingerprint))
    rows.append(("Graph", mark))
    rows.append(("Imported", e(imported) if imported
                 else f'<span class="shep-muted">{e(r.NOT_RECORDED.lower())}</span>'))
    source_format = version.get("source_format")
    if source_format:
        rows.append(("Format", e(source_format)))
    refusal = (r.status_mark("error", ANOTHER_GRAPH) if state == "another" else "")
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
    count (the skill's §8: a table's title with its count)."""
    label = reports.get("label")
    count = len(reports.get("runs") or [])
    right = (f"{count} of {SELECTED_VERSION}, kept in the registry's report store"
             if label else "choose a data version")
    return (f'<div class="shep-panel-hd"><h2>Reports</h2><span class="shep-panel-right">'
            f'{e(right)}</span></div>')


def reports_html(reports: Mapping[str, Any], selected: Optional[str], *,
                 models: Optional[Mapping[str, Any]] = None, job: Optional[Mapping[str, Any]] = None,
                 roots: Sequence[Tuple[str, str]] = (),
                 selection: Optional["Selection"] = None) -> str:
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
    selected row is ``aria-current`` (the skill's §8)."""
    label = reports.get("label")
    if not label:
        return ""
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
    notes = ("" if entries else
             '<div class="shep-list-note">No run has kept a report on this version yet.</div>')
    problem = reports.get("problem")
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


def model_html(page: Mapping[str, Any], models: Mapping[str, Any], key: Optional[str]) -> str:
    """The chosen model's details, by the shared renderers in the one summary
    list (`renderers.summary_list`), each a fact its options do not say: what
    it is (`renderers.model_kind_text`: "Disease · frozen graph backbone"
    for a task model whose graph backbone kept its base's weights), its
    file, the Trainer's validation figure, its model fingerprint (12 hex,
    copyable) and how its dataset stands to the chosen workspace's
    (`renderers.dataset_relation_html`: "the selected workspace's current
    dataset", which Data says once; which one and its fingerprint where it is
    another); or why there is no list."""
    roots = page.get("roots") or []
    if models.get("refused"):
        return _reason_html("Not shown · listing checkpoints opens them, which is an operator "
                            "control. " + operator_only(page.get("access") or {}), kind="lock")
    if models.get("problem"):
        return _reason_html(redact(models["problem"], roots))
    item = find_model(models, key)
    if item is None:
        if not models.get("items"):
            return _reason_html("This workspace has no checkpoint to test." if models.get("workspace")
                                else "")
        return '<div class="shep-sub">Choose the run, then the checkpoint to test.</div>'
    if item.get("problem") or not r.is_digest(item.get("sha256")):
        return _reason_html(f"{item.get('filename')}: metadata not readable · cannot be tested "
                            f"({redact(item.get('problem') or 'no digest was read', roots)})")
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
    is."""
    choice = run_evidence(page, version)
    helps = {"R10 evidence": ("test-evidence", "R10 evidence")}
    if choice.not_applicable:
        return r.summary_list([("R10 evidence", f'{r.status_mark("neutral", r.NOT_APPLICABLE)} · '
                                                f'{e(choice.not_applicable)}')], helps=helps)
    evidence = page.get("evidence") or {}
    names = evidence.get("names") or []
    roots = page.get("roots") or []
    # The workspace by its relation: its name is its select's, said once.
    measured = (f"{e(SELECTED_WORKSPACE)} (measured, always)" if workspace
                else "the measured workspace (always)")
    note = ""
    if choice.problem:
        extra = " · " + r.status_mark("error", "Refused: " + redact(choice.problem, roots))
    elif names:
        extra = " · " + ", ".join(f"<code>{e(n)}</code>" for n in names) + " (the server's setting)"
        if choice.note:
            note = f'<div>{r.status_mark("warn", redact(choice.note, roots))}</div>'
    else:
        extra = " · no other evidence directory is set on the server"
    return r.summary_list([("R10 evidence", f"{measured}{extra}{note}")], helps=helps)


def fixed_html(page: Mapping[str, Any]) -> str:
    """What the server fixes for a run (D5): the batch size and the device.
    The measurement is chosen beside them (the Measurement radio,
    `measurement_choices`)."""
    device = page.get("device") or {}
    if device.get("problem"):
        shown = f'<span class="shep-amber-t">{e(device["problem"])}</span>'
    else:
        value = device.get("value") or "auto"
        shown = (f"<code>{e(value)}</code>" + (" (requires CUDA)" if value == "auto" else ""))
    return r.summary_list([("Batch size", f"{BATCH_SIZE_SHOWN} · fixed by the server"),
                           ("Device", f"{shown} · fixed by the server")],
                          helps={"Batch size": ("test-fixed", "Fixed by the server")})


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


def start_state(page: Mapping[str, Any], models: Mapping[str, Any], workspace: Optional[str],
                label: Optional[str], key: Optional[str], binding: Optional[Mapping[str, Any]],
                gpu: Tuple[str, str], *, protocol: Optional[str] = None
                ) -> Tuple[bool, Optional[str]]:
    """Whether "Start test" can be used now for the measurement `protocol`
    (the Measurement radio's value; None: none chosen), and why not, in the
    order an operator fixes things: a busy or unread GPU lock comes before the
    choices, since no choice would enable Start while it lasts. Only a
    display: the handler checks D8 and the runner refuses again at start."""
    access = page.get("access") or {}
    roots = page.get("roots") or []
    if not access.get("allowed"):
        return False, operator_only(access)
    if page.get("registry_problem"):
        return False, "Disabled: no registry is available (see Data)."
    kind, words = gpu
    if kind == "busy":
        # The holder in the GPU chip's words, with where to stop this
        # service's inference when that holds it (`gpu_line`), as Training's
        # and Fine-tune's starts say it: whatever holds the lock, a job or
        # inference, Start waits for it.
        return False, f"Disabled: {words}."
    if kind == "unread":
        return False, "Disabled: the GPU lock cannot be read or used just now (see the GPU chip)."
    workspace_row = find(page.get("workspaces") or [], "name", workspace)
    if workspace_row is None:
        return False, "Disabled: choose a workspace."
    version = find(page.get("versions") or [], "label", label)
    if version is None:
        return False, "Disabled: choose a data version."
    if version.get("problem"):
        return False, "Disabled: this data version cannot be read."
    if binding_state(version, binding)[0] == "another":
        # Its refusal is said beside the version (`ANOTHER_GRAPH`), once.
        return False, "Disabled: the data version is mapped against another graph (see Data)."
    item = find_model(models, key)
    if item is None:
        return False, "Disabled: choose a model."
    if item.get("problem") or not r.is_digest(item.get("sha256")):
        return False, "Disabled: this checkpoint's metadata is not readable."
    refusal = measure_refusal(item, version, protocol)
    if refusal:
        return False, "Disabled: " + said(redact(refusal, roots), "")
    # The runner's one evidence choice by the version's kind: the setting is
    # required only where it is given (a supplied cohort version).
    evidence = run_evidence(page, version)
    if evidence.problem:
        return False, ("Disabled: the evidence directories setting is refused: "
                       + redact(evidence.problem, roots))
    if (page.get("device") or {}).get("problem"):
        return False, f"Disabled: {page['device']['problem']}"
    return True, None


def job_head_html(job: Mapping[str, Any]) -> str:
    """The job card's title row: the state and its time (Cancel sits beside it)."""
    status = job.get("status") or "idle"
    head = f'<div class="shep-test-job-hd"><h3>Test run</h3>{_pill(status)}'
    if status in _BUSY:
        head += f'<span class="shep-sub">elapsed <b>{e(_clock(job.get("elapsed_seconds")))}</b></span>'
    elif status != "idle":
        ended = utc_time(job.get("end_time"))
        head += f'<span class="shep-sub">ended {e(ended or "—")}</span>'
    return head + "</div>"


def job_inputs_html(job: Mapping[str, Any]) -> str:
    """**What a test run was given, as its inputs recorded them**, in the one
    summary list (the skill's §4, §5: a fact a row, never a chain of " · "):
    its version (label and fingerprint), its model (its fingerprint, the one
    digest value), its measurement (`_job_measurement`), its workspace, and
    its register, determinism and device settings, its R10 evidence (by its
    own kind, the runner's choice: none for a KG edge set or a generated
    split) and its start. Drawn from the job alone, never from the selection
    (the poll writes the card, `_card_outputs`), so a run keeps its own
    identity while another model or version is chosen: these rows are the
    run's, which can differ from the selection's (the skill's §1.2)."""
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
            ("Measurement", _job_measurement(inputs.get("protocol"))),
            ("Workspace", e(workspace) if workspace
             else r.status_mark("neutral", r.NOT_RECORDED)),
            ("Register", e(r.setting_words("record", bool(inputs.get("record"))))),
            ("Deterministic algorithms",
             e(r.setting_words("deterministic_algorithms", bool(inputs.get("deterministic"))))),
            ("Device", f"<code>{e(inputs.get('device') or 'auto')}</code>"),
            ("R10 evidence", e("the workspace" + (
                f" + {len(evidence)} director{'ies' if len(evidence) != 1 else 'y'}"
                if evidence else "")) if takes else r.status_mark("neutral", r.NOT_APPLICABLE))]
    if started:
        rows.append(("Started", e(started)))
    return r.summary_list(rows, classes="shep-test-ident-rows")


def job_body_html(job: Mapping[str, Any], roots: Sequence[Tuple[str, str]],
                  gpu: Tuple[str, str]) -> str:
    """The job card's body: what the run was given (`job_inputs_html`), once
    it has ended what it left (the runner's outcome), its output tail, and
    while it runs the GPU line."""
    status = job.get("status") or "idle"
    if status == "idle":
        return ('<div class="shep-test-job-bd"><span class="shep-sub">No test run since the '
                "server started.</span></div>")
    body = [job_inputs_html(job)]
    outcome = job.get("outcome")
    if status not in _BUSY and isinstance(outcome, Mapping):
        body.append(outcome_html(outcome, roots))
    if status == "failed" and job.get("error_message"):
        first = str(job["error_message"]).split("\n", 1)[0]
        body.append(f'<div class="shep-test-fail">{e(redact(case_free(first), roots))}</div>')
    if job.get("exit_meaning") and status not in _BUSY:
        body.append(f'<div class="shep-sub">Exit status {e(job.get("exit_code"))}: '
                    f'{e(job["exit_meaning"])}</div>')
    # The output is the run's own; a refusal in it may name a case (D7).
    lines = [redact(case_free(line), roots) for line in job.get("log") or []]
    if lines and status == "cancelled":
        # A cancel interrupts the run mid-step; what it printed up to then, and
        # its last line saying it was cancelled, are not news, so they are
        # folded (open it to read it).
        body.append(f'<details class="shep-test-fold"><summary class="shep-sub">Output, last '
                    f'{len(lines)} lines (the interrupted run\'s)</summary>'
                    f'<pre class="shep-test-log">{e(chr(10).join(lines))}</pre></details>')
    elif lines:
        body.append(f'<div class="shep-sub">Output, last {len(lines)} lines</div>'
                    f'<pre class="shep-test-log">{e(chr(10).join(lines))}</pre>')
    foot = ""
    _kind, words = gpu
    if status in _BUSY:
        foot = (f'<div class="shep-test-job-ft"><span class="shep-chip shep-chip-busy">'
                f'<span class="shep-dot" aria-hidden="true"></span>{e(words)}</span>'
                '<span class="shep-sub">Training start is refused while this runs.</span></div>')
    return f'<div class="shep-test-job-bd">{"".join(body)}</div>' + foot


def job_html(job: Mapping[str, Any], roots: Sequence[Tuple[str, str]],
             gpu: Tuple[str, str]) -> str:
    """The latest test run's whole card (title row and body)."""
    return job_head_html(job) + job_body_html(job, roots, gpu)


def outcome_html(outcome: Mapping[str, Any], roots: Sequence[Tuple[str, str]]) -> str:
    """What an ended run left, as the runner read it from disk when the run
    ended: kept and registered (True, False or None for "cannot be checked")
    and its summary. It is that moment's reading and says so; a report
    registered afterwards (Register this report) reads "Registered" in Result,
    which is where its registration now is shown (the M5 close-out,
    2026-10-25)."""
    kept, registered = outcome.get("kept"), outcome.get("registered")
    chips = ['<span class="shep-sub">When the run ended:</span>']
    chips.append(r.status_mark("ok", "Report kept", as_chip=True) if kept is True
                 else r.chip("No report kept", "grey") if kept is False
                 else r.chip("Kept: cannot be checked", "grey"))
    if kept is not False:
        chips.append(r.status_mark("ok", "Registered", as_chip=True)
                     if registered is True
                     else r.chip("Not registered", "neutral") if registered is False
                     else r.chip("Registered: cannot be checked", "grey"))
    summary = outcome.get("summary") or ""
    now = ('<div class="shep-sub">The report\'s registration now is shown in Result.</div>'
           if kept is True else "")
    return (f'<div class="shep-test-outcome"><div class="shep-test-chips">{"".join(chips)}</div>'
            f'<div>{e(redact(summary, roots))}</div>{now}</div>')


# ---------------------------------------------------------------------------
# Rendering: Result
# ---------------------------------------------------------------------------
def _environment(view: Mapping[str, Any]) -> str:
    """The view's environment class (`sidecar.environment_class`, as the one
    reader classified it) and, for a run on CUDA, its GPU and driver."""
    text = str(view["environment_class"])
    gpu = view["gpu"]
    if isinstance(gpu, Mapping):
        text += (f" · GPU {gpu.get('name') or 'name not read'}, driver "
                 f"{gpu.get('driver') or 'not read'}")
    return e(text)


def _native_build(view: Mapping[str, Any]) -> str:
    """The view's native build identity (the run environment's one
    projection, `native_identity`, as the API's record view and Models show
    it) in its one wording: a lead with the route and the count, then one line
    per component (`native_identity_lines`), since the one line runs to four in
    this column (the skill's §4: past two lines, structure); "not recorded"
    for a run recorded before its environment named components."""
    return "".join(f"<div>{e(line)}</div>"
                   for line in native_identity_lines(view["native_identity"]))


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


def result_identity_rows(record: Mapping[str, Any], view: Mapping[str, Any],
                         selection: Optional[Selection], models: Mapping[str, Any],
                         workspaces: Sequence[Mapping[str, Any]]) -> List[Tuple[str, str]]:
    """**What a result is of, by its relation to what is chosen now** (the
    skill's §1.2: a fact once, elsewhere by its relation; repeated only where
    the two can differ): its workspace (the dataset record its report
    measured, `dataset_record.sha256`), its model (`checkpoint_digest`) and
    its version (the version manifest's digest). Each that is what is chosen
    reads so, "the selected workspace", "the selected model", "the selected
    version"; one that is not reads in full: the listed workspaces holding
    its dataset record, its model by what it is, its run (whole) and its file
    (`_other_model_html`) and its fingerprint, its version's label, role and
    fingerprint. So a past result keeps its own identity when the selection
    changes: it is drawn again from the record each time, never from the
    selects."""
    selection = selection or Selection(None, None, None, None)
    dataset = (record.get("dataset_record") or {}).get("sha256") if isinstance(
        record.get("dataset_record"), Mapping) else None
    if r.is_digest(dataset) and dataset == selection.record_sha256:
        workspace = e(SELECTED_WORKSPACE)
    else:
        names = [str(w.get("name")) for w in workspaces if r.is_digest(dataset)
                 and ((w.get("dataset_record") or {}).get("sha256") == dataset)]
        workspace = (", ".join(f"<b>{e(name)}</b>" for name in names) if names
                     else r.status_mark("neutral", "No listed workspace holds its dataset record")
                     if r.is_digest(dataset) else r.status_mark("neutral", r.NOT_RECORDED))
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
    return [("Workspace", workspace), ("Model", model), ("Version", version)]


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
    if state.get("kept") == "kept":
        chips.append(r.status_mark("ok", "Report kept", as_chip=True))
    else:
        chips.append(r.chip("No report kept" if state.get("kept") == "not_kept"
                            else "Kept: cannot be checked", "grey"))
    if state.get("registered") == "registered":
        seq = state.get("sequence")
        chips.append(r.status_mark("ok", "Registered" + (f" record {seq}" if isinstance(seq, int)
                                                         else ""), as_chip=True))
    elif state.get("kept") == "kept":
        chips.append(r.chip("Not registered" if state.get("registered") == "not_registered"
                            else "Registration: cannot be checked from the list", "neutral"))
    contest = run.get("contest") or {}
    if contest.get("state") == "contested":
        chips.append(r.warn_chip("Runs differ — investigate") if _deterministic_group(run)
                     else r.chip("Runs differ", "grey"))
    parts = [f'<div class="shep-test-chips">{"".join(chips)}</div>']
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
    return f'<div class="shep-test-pad shep-test-result">{head}{"".join(parts)}</div>'


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


def _tile_note(name: str, kind: str = "supplied") -> str:
    """What a described measurement's tile figure is over: its positive pairs
    (pooled), its ranked cases (macro) or the share of the import-eligible cases
    ranked (coverage; a generated split's cases are not imported, so its share
    is of the cases measured), with no count, since the record's Coverage row
    says "n of M" once (skill rule 1.2)."""
    if name == "coverage":
        return ("share of import-eligible cases ranked" if kind == "supplied"
                else "share of the cases measured that were ranked")
    if name.startswith("pooled_"):
        return "pooled over positive pairs"
    if name.startswith("macro_"):
        return "mean over the ranked cases"
    return ""


def _figure_words(mode: str, name: str) -> str:
    """A task metric's words as a row under the record's label, which names
    the task once: `caveats.metric_text`'s "Disease pooled top-k" less the
    task's name, "Pooled top-k"."""
    words = metric_text(mode, name)[len(TASK_NAMES[mode]) + 1:]
    return words[:1].upper() + words[1:]


def _task_rows(mode: str, metrics: Mapping[str, Any], tiles: Sequence[str]
               ) -> List[Tuple[str, str]]:
    """A task record's figures past its tiles (README E1, E2, E5): the
    conservative pooled MRR with what it counts, one row for each family that
    takes a cut-off k (its k values in order, a tile's left out), the macro
    and best-positive MRR, and any other figure the record holds. No mean
    rank: a task reports none."""
    shown = set(tiles) | {"conservative_pooled_mrr"}
    conservative = _number(metrics.get("conservative_pooled_mrr"))
    rows = [(_figure_words(mode, "conservative_pooled_mrr"),
             (conservative if conservative != "—" else "not reported")
             + " · an unranked case whose truth can be assessed counts 0")]
    for prefix in _TASK_FAMILIES:
        family = sorted((int(name[len(prefix):]), name) for name in metrics
                        if str(name).startswith(prefix) and str(name)[len(prefix):].isdigit())
        shown.update(name for _k, name in family)
        values = [f"k={k} {_number(metrics[name])}" for k, name in family if name not in tiles]
        if values:
            rows.append((_figure_words(mode, f"{prefix}k"), e(", ".join(values))))
    for name in _TASK_ROWS:
        if name in metrics:
            shown.add(name)
            rows.append((_figure_words(mode, name), e(_number(metrics[name]))))
    other = [f"{metric_text(mode, str(k))} {_number(v)}" for k, v in sorted(metrics.items())
             if k not in shown]
    if other:
        rows.append(("Other metrics", e(" · ".join(other))))
    return rows


def recorded_metrics_html(metrics: Mapping[str, Any]) -> str:
    """Every metric of a record as recorded, one per line
    (`caveats.recorded_metric_lines`: the name as written, a code, and the
    value's JSON text, unrounded), every value escaped; "no metric recorded"
    when it holds none."""
    lines = recorded_metric_lines(metrics)
    if not lines:
        return '<span class="shep-muted">no metric recorded</span>'
    return "".join(f"<div><code>{e(name)}</code> {e(value)}</div>" for name, value in lines)


def _semantics_text(raw: Any) -> str:
    """The record's `scoring.score_semantics` exactly as recorded (the view's
    ``score_semantics``), a code, or "not recorded": never split, interpreted
    or rebuilt."""
    if isinstance(raw, str) and raw.strip():
        return f"<code>{e(raw)}</code>"
    return '<span class="shep-muted">not recorded</span>'


def _record_html(record: Mapping[str, Any], run: Mapping[str, Any],
                 state: Mapping[str, Any], *, selection: Optional[Selection] = None,
                 models: Optional[Mapping[str, Any]] = None,
                 workspaces: Sequence[Mapping[str, Any]] = ()) -> str:
    """The selected report's record (`build_record`'s, through the store's
    scan) as the one reader reads it (research comparison Stage 8b): its data
    kind's result view (`record_views.result_view`, the one the results
    service reads every kept report by, registered or not). Every field shown
    is the view's: its mode, data (label, role, version, kind and counts),
    the run's counts, score semantics, relations, R10 or the kind's lines (the
    shared renderings, `renderers.r10_rows` and `renderers.data_kind_lines`,
    as Models shows a registered record's), environment, native build and
    dataset origin; the measurement's device is the record's recorded
    setting (`record_views.recorded_settings`, as the service reads it), and
    the model is the record's key field. Only the metric values are the
    record's own, by the names the view read (an integer stays an integer,
    as the service keeps them, `results._metrics`). A record the reader
    cannot read is not shown as a result: its reason, naming fields and
    never values, as Models and the results service list it."""
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
    cohort, counts, n_ranked = view["cohort"], view["counts"], view["n_ranked"]
    metrics = {name: record["metrics"][name] for name in view["metrics"]}
    # Labelled by the record's own mode, as recorded, never by another's.
    mode = view["mode"]
    label = mode_label(mode)
    tile_names = _TILES.get(mode, ())
    task = mode in TASK_NAMES
    eligible = _count(counts.get("ranked_eligible"))
    deterministic = view["deterministic_algorithms"]
    device = recorded_settings(record)["regime"].get("device")
    # The record's data kind, as the one reader read it (research comparison
    # Stage 6): a version of another kind than a supplied cohort is named by
    # its kind and role, and read by its own counts and exposure, never a
    # cohort's or R10's words.
    kind = cohort["kind"]
    # The one value-words rule (on / off); a record without the flag holds
    # none: "not recorded", never "not set".
    determinism = (r.setting_words("deterministic_algorithms", deterministic)
                   if deterministic is not None else r.NOT_RECORDED.lower())
    # What the result is of, as the record holds it, by its relation to what
    # is chosen now: one row per fact in the one summary list (the skill's
    # §4, §5: never a chain of " · "), the report's run id once.
    identity = r.summary_list([
        ("Report", r.ident_html(run.get("run_id"))),
        *result_identity_rows(record, view, selection, models or {}, workspaces),
        ("Deterministic algorithms", e(determinism)),
        ("Device", f"<code>{e(device)}</code>" if device
         else f'<span class="shep-muted">{e(r.NOT_RECORDED.lower())}</span>'),
    ], classes="shep-test-ident-rows")
    tiles = "".join(
        f'<div class="shep-test-tile"><div class="shep-test-tile-l">{e(metric_text(mode, name))}</div>'
        f'<div class="shep-test-tile-v">{e(_number(metrics.get(name)))}</div>'
        f'<div class="shep-test-tile-n">{e(_tile_note(name, kind))}</div>'
        '</div>'
        for name in tile_names)
    excluded = counts.get("excluded") if isinstance(counts.get("excluded"), Mapping) else {}
    reasons = " · ".join(f"{str(k).replace('_', ' ')} {_count(v)}" for k, v in excluded.items())
    excluded_text = (f"{e(_count(counts.get('n_excluded')))} excluded"
                     f"{' at import' if task else ''}"
                     f"{': ' + e(reasons) if reasons else ''}")
    if kind == "kg_edges":
        run = view["measured"]["counts"]
        coverage = (f"<b>{e(_count(run.get('pairs_scored')))} of {e(_count(counts.get('pairs')))}"
                    f"</b> stored pairs scored · {e(_count(counts.get('positive_edges')))} "
                    f"positive edges; {e(_count(counts.get('edges_without_negative')))} without a "
                    "negative to pair")
    elif kind == "generated":
        diseases = counts.get("diseases")
        coverage = (f"<b>{e(_count(n_ranked))}</b> "
                    + ("cases ranked by the model" if task else "counted by the record")
                    + f" · {e(_count(counts.get('samples')))} generated patients of "
                    f"{e(_count(diseases))} disease{'' if diseases == 1 else 's'}")
    elif task:
        coverage = (f"<b>{e(_count(n_ranked))} of {e(eligible)}</b> import-eligible cases ranked "
                    f"by the model · {e(_count(counts.get('source_cases')))} source cases; "
                    f"{excluded_text}")
    else:
        # No claim of what this mode's n_ranked counts.
        coverage = (f"{e(_count(counts.get('source_cases')))} source cases · {excluded_text}; "
                    f"{e(_count(n_ranked))} cases counted by the record; what they are is not "
                    "described for this mode")
    # One display for every mode (R2): a described measurement's figures in
    # words, every metric as recorded one click away (below); any other mode's
    # metrics as recorded are its main result, one per line.
    figures = (_task_rows(mode, metrics, tile_names) if task
               else [("Recorded metrics", recorded_metrics_html(metrics))])
    lines = [
        ("Coverage", coverage),
        *figures,
        ("Score semantics", _semantics_text(view["score_semantics"])),
        *r.relation_rows(view["relations"]),
        *(r.r10_rows(view["r10"]) if kind == "supplied" else r.data_kind_lines(view)),
        ("Environment", _environment(view)),
        ("Native build", _native_build(view)),
    ]
    origin = view["dataset_origin"]
    meaning = _ORIGIN_MEANING.get(origin)
    lines.append(("Dataset", (f'<b class="shep-amber-t">{e(origin)}</b>' if origin == "synthetic"
                              else e(origin or "not reported"))
                  + (f" — {e(meaning)}" if meaning else "")))
    registered = state.get("registered")
    seq = state.get("sequence")
    beside = state.get("checked_beside")
    where = f" (the ledger beside {e(beside)} was read)" if beside else ""
    # A record merged before each run had its own record may hold these bytes
    # (K8-6r.2): said as a possible overlap, never as registered.
    merged = [n for n in state.get("historical_overlap") or [] if isinstance(n, int)]
    overlap = (f"; possible overlap with record{'s' if len(merged) != 1 else ''} "
               f"{', '.join(str(n) for n in merged)}, merged before each run had its own record "
               "(not recorded whether this run is one of them); registering adds this run's "
               "own record" if merged and registered != "registered" else "")
    lines.append(("Registration",
                  (f"registered as record {seq}{where}" if registered == "registered" and isinstance(seq, int)
                   else f"registered{where}" if registered == "registered"
                   else f"not registered{where}" if registered == "not_registered"
                   else f"cannot be checked{where}" if beside
                   else "none found beside the checkpoint the run read, or its ledger cannot be "
                        "read (a registration beside another copy of the weights is looked for only "
                        "with the checkpoint list, an operator control)") + e(overlap)))
    # The R10 labels' ⓘ sits beside the first R10 line, which it explains.
    kv = r.summary_list(lines, helps={"R10, training": ("test-r10", "R10 labels")},
                        classes="shep-test-lines")
    topic = _MODE_HELP.get(mode)
    mode_help = help_button(topic, TOPIC_HELP[topic][0]) if topic else ""
    fold = (f'<details class="shep-test-fold shep-test-recm"><summary>Recorded metrics '
            f'({len(metrics)})</summary>{recorded_metrics_html(metrics)}</details>'
            if task and metrics else
            '<div class="shep-sub shep-test-recm">no metric recorded</div>' if task else "")
    return (f'{identity}'
            f'<div class="shep-test-label">{e(label)}{mode_help}</div>'
            f'<div class="shep-test-tiles">{tiles}</div>{kv}'
            f'{fold}')


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
        if mine:
            items = "".join(
                f'<li><b>{e(n.get("who"))}</b> · {e(str(n.get("at") or "").replace("T", " ").replace("Z", " UTC"))}'
                f'<div>{e(n.get("what"))}</div></li>' for n in mine)
            parts.append(f'<div class="shep-test-notes"><div class="shep-w600">Investigation '
                         f'notes ({len(mine)})</div><ul>{items}</ul></div>')
        if shown and notes_problems:
            parts.append(_reason_html(
                "The investigation notes cannot all be read, so a note kept for this group may "
                "be missing here: " + "; ".join(redact(p, roots) for p in notes_problems)))
        elif shown and not mine:
            parts.append('<div class="shep-sub">No investigation note is kept for this '
                         'group.</div>')
    return "".join(parts)


def _store_not_listed(problem: str, roots: Sequence[Tuple[str, str]], what: str) -> str:
    """Why Register is off when the store could not be listed (Decision 10):
    no report can be confirmed here, and the command line registers a named
    report it verifies. Who acts: the operator, on the server."""
    return (f"Disabled: the report store could not be listed, so {what}'s report cannot be "
            f"confirmed here: {redact(problem, roots).rstrip('.')}. Register it from the "
            "command line by naming its report (scripts/record_evaluation.py --report …).")


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
        return False, "Disabled: no report was kept in this run."
    if state.get("registered") == "registered":
        return False, "Already registered."
    if run.get("record") is None:
        if run.get("unbuilt"):
            return False, ("Disabled: this report cannot be registered: "
                           f"{redact(run['unbuilt'], roots)}")
        if reports.get("store_problem"):
            return False, _store_not_listed(reports["store_problem"], roots, "this run")
        return False, ("Disabled: the report store's listing did not read this report (it may "
                       "have been written or replaced since); press ↻ to read it again.")
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
        notes.append(_reason_html("No import source is in the intake directory: place a "
                                  "shepherd-cases/1 file there on the machine, then open this again."))
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
    return (f"digest <code>{e(str(entry.get('digest') or '')[:12])}</code>…, declared version "
            f"{e(entry.get('declared_version') or 'none')}")


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
                   f'<code class="shep-full">{e(source.get("sha256"))}</code>'),
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
        ("Graph", "the workspace manifest’s digests: " + " · ".join(
            f"{e(role)} <code>{e(str(d or '')[:12])}</code>" for role, d in graph.items())),
        ("Dataset record", ('origin <b class="shep-amber-t">synthetic</b>' if record.get("origin") == "synthetic"
                            else f"origin {e(record.get('origin'))}")
                           + f", sha256 <code>{e(str(record.get('sha256') or '')[:12])}</code>…"),
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


def open_import(request: Any, *, environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """"Import a new version…": D8 on this request (the intake directory holds
    patient-level data), then the server's intake listing."""
    from src.api.services.server_lists import list_intake_files

    access = operator_access(request)
    if not access.allowed:
        return {"refused": access.refusal, "items": [], "problem": None}
    listing = list_intake_files(environ=environ)
    return {"refused": None, "items": [asdict(i) for i in listing.items], "problem": listing.problem}


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
        return None, "Choose a file from the intake directory. Nothing was written."
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
        with gr.Column(elem_classes=["shep-panel", "shep-test-col"], elem_id="shep-test-cohort",
                       scale=9, min_width=300):
            cohort_hd = plain_html(cohort_header_html({}), elem_id="shep-test-cohort-hd")
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
                drawer_title = plain_html(
                    '<h2>Import a new cohort version</h2>', elem_id="shep-test-import-title")
                import_close = gr.Button("✕ Close", variant="secondary", size="sm", scale=0,
                                         elem_id="shep-test-import-close")
            with gr.Column(elem_classes=["shep-test-bd"]):
                intake = gr.Dropdown(label="File in the intake directory (server)", choices=[],
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
        with gr.Column(elem_classes=["shep-panel", "shep-test-col"], elem_id="shep-test-run",
                       scale=10, min_width=320):
            plain_html(run_header_html(), elem_id="shep-test-run-hd")
            with gr.Column(elem_classes=["shep-test-bd"]):
                # The model as Run → Checkpoint, the shared cascading pair
                # (`model_list.parents_update`), one under the other so each
                # 40-character option shows whole in the column; ↻ lists the
                # workspace's checkpoints again.
                with gr.Row(elem_classes=["shep-test-model-row"]):
                    run_select = gr.Dropdown(label="Run", choices=[], value=None,
                                             elem_id="shep-test-model-run", interactive=False,
                                             elem_classes=["shep-test-select"])
                    model_reload = gr.Button(GATED_LABELS["reload"], variant="secondary",
                                             size="sm", scale=0,
                                             elem_id="shep-test-model-reload",
                                             interactive=False)
                model = gr.Dropdown(label="Checkpoint", choices=[], value=None,
                                    elem_id="shep-test-model", interactive=False,
                                    elem_classes=["shep-test-select"])
                model_info = plain_html("", elem_id="shep-test-model-info")
                # The protocols the chosen model's listed capabilities admit (`measurement_choices`).
                measurement = gr.Radio(label="Measurement", choices=[], value=None,
                                       elem_id="shep-test-measurement", interactive=False)
                evidence = plain_html("", elem_id="shep-test-evidence")
                with gr.Row(elem_classes=["shep-test-checks"]):
                    record = gr.Checkbox(label="Register the result", value=True,
                                         elem_id="shep-test-record")
                    deterministic = gr.Checkbox(label="Deterministic execution", value=False,
                                                elem_id="shep-test-deterministic")
                fixed = plain_html("", elem_id="shep-test-fixed")
            # Start and the job are the panel's main operations: at wide sizes
            # this region stays at the panel's foot (the choices above scroll),
            # and the job's output scrolls inside its card under its title row,
            # so Start and Cancel are on screen whatever the model's details or
            # the output's length. Narrow, it follows the choices.
            with gr.Column(elem_classes=["shep-test-runft"], elem_id="shep-test-runft"):
                start = gr.Button(GATED_LABELS["start"], variant="primary", interactive=False,
                                  elem_id="shep-test-start")
                start_reason = plain_html("", elem_id="shep-test-start-reason")
                action_out = plain_html("", elem_id="shep-test-action")
                with gr.Column(elem_classes=["shep-test-job"], elem_id="shep-test-jobcard"):
                    # Cancel beside the state, so it stays in view however long the output.
                    with gr.Row(elem_classes=["shep-test-job-top"]):
                        job_head = plain_html(job_head_html({}), elem_id="shep-test-job-head")
                        cancel = gr.Button(GATED_LABELS["cancel"], variant="stop",
                                           interactive=False, elem_id="shep-test-cancel",
                                           scale=0, size="sm")
                    cancel_reason = plain_html("", elem_id="shep-test-cancel-reason")
                    job_out = plain_html(job_body_html({}, [], ("none", "")),
                                         elem_id="shep-test-job")
        # ------------------------------------------------------------ Result
        # The reports as a compact single-select table at the panel's top, the
        # chosen report's result under it, both in the panel's one scroll
        # region; the title, the note form and Register are in flow.
        with gr.Column(elem_classes=["shep-panel", "shep-test-col"], elem_id="shep-test-result",
                       scale=16, min_width=360):
            reports_hd = plain_html(reports_header_html({}), elem_id="shep-test-reports-hd")
            with gr.Column(elem_classes=["shep-test-bd", "shep-test-bd-flush"]):
                reports_list = plain_html("", elem_id="shep-test-reports", js_on_load=_REPORTS_JS)
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
                register_reason = plain_html("", elem_id="shep-test-register-reason")
                register_out = plain_html("", elem_id="shep-test-register-out")
    # The poll's hand-off to the selection's group (`_poll`): a hidden
    # component, not a State, because the answer to a queue=False request
    # dispatches a component's .change and never a State's.
    job_changed = gr.JSON(value=None, visible=False, elem_id="shep-test-job-changed")

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
                _reason_html(why, kind="lock" if not allowed else "")]

    def _measurement_outputs(page, models, label, key, protocol):
        """The Measurement radio for the chosen model on the chosen version
        (`measurement_choices`: what its capabilities meet that measures the
        version's kind), and the value it shows (`chosen_measurement`), which
        Start's state is then drawn from. Like the Model select, it is
        interactive for an operator only."""
        allowed = bool((page.get("access") or {}).get("allowed"))
        choices = measurement_choices(find_model(models, key),
                                      find(page.get("versions") or [], "label", label))
        protocol = chosen_measurement(choices, protocol)
        return gr.update(choices=choices, value=protocol,
                         interactive=allowed and bool(choices)), protocol

    def _card_outputs(page, job, gpu):
        """The job card and Cancel, from the job alone (`card_outputs`)."""
        roots = page.get("roots") or []
        allowed = bool((page.get("access") or {}).get("allowed"))
        status = job.get("status") or "idle"
        can_cancel = allowed and status == "running"
        # Refused, Cancel shows the lock whatever the job's state, so its
        # reason is beside it whatever the job's state (`chips.gated`).
        cancel_why = None if allowed else operator_only(page.get("access") or {})
        return [job_head_html(job), job_body_html(job, roots, gpu),
                gated_update("cancel", page, interactive=can_cancel),
                _reason_html(cancel_why, kind="lock")]

    def _run_outputs(page, models, binding, ws, label, key, protocol, job, gpu):
        allowed = bool((page.get("access") or {}).get("allowed"))
        radio, protocol = _measurement_outputs(page, models, label, key, protocol)
        return [
            # The shared pair; nothing chosen until the operator picks a run
            # (`chosen_in_run` then takes its latest save): the page never
            # measures a model it picked itself.
            *ml.parents_update(model_parents(models), key, first=False, interactive=allowed),
            gated_update("reload", page, interactive=bool(ws)),
            model_html(page, models, key),
            radio,
            evidence_html(page, ws, find(page.get("versions") or [], "label", label)),
            fixed_html(page),
            *_start_outputs(page, models, binding, ws, label, key, protocol, gpu),
            *_card_outputs(page, job, gpu),
            gr.update(interactive=allowed),
            gr.update(interactive=allowed),
            protocol,
            key if find_model(models, key) is not None else None,
        ]

    # Start and its reason depend on the selection: only the selection's group
    # writes them. The job card and Cancel depend on the job alone: the poll
    # writes them too.
    start_outputs = [start, start_reason]
    card_outputs = [job_head, job_out, cancel, cancel_reason]
    run_outputs = [run_select, model, model_reload, model_info, measurement, evidence, fixed,
                   *start_outputs, *card_outputs, record, deterministic, measurement_state,
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
            reports_html(reports, run_id, models=models, job=job, roots=roots, selection=chosen),
            result_html(reports, run_id, roots, job=job, notes=reports.get("notes"),
                        notes_problems=reports.get("notes_problems") or (), selection=chosen,
                        models=models, workspaces=page.get("workspaces") or []),
            gated_update("register", page, interactive=can),
            _reason_html(why, kind="lock" if not allowed else ""),
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
        allowed = bool((page.get("access") or {}).get("allowed"))
        registry_ok = not page.get("registry_problem")
        import_why = (operator_only(page.get("access") or {}) if not allowed
                      else None if registry_ok and ws else
                      "Disabled: no cohort registry is available." if not registry_ok
                      else "Disabled: choose a workspace to import against.")
        return [
            cohort_header_html(page),
            _dd(r.workspace_choices(page.get("workspaces") or []), ws),
            workspace_html(page, ws),
            _dd(version_choices(page), label, interactive=registry_ok),
            version_html(page, label, binding),
            gated_update("import", page, interactive=registry_ok and bool(ws)),
            _reason_html(import_why, kind="lock" if not allowed else ""),
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
    def _refresh(ws, label, key, protocol, current_models, current_report, request: gr.Request):
        page = read_page(request)
        ws = default_workspace(page, ws)
        binding = _binding_for(page, ws)
        label = default_version(page, binding, label)
        models = _models_for(page, current_models, ws, request)
        key = key if find_model(models, key) is not None else None
        job = read_job()
        gpu = _gpu(page)
        reports, run_id = _reports(page, models, label, current_report, job, request)
        return [page, models, binding, reports, run_id,
                *_cohort_outputs(page, binding, ws, label),
                *_run_outputs(page, models, binding, ws, label, key, protocol, job, gpu),
                *_result_outputs(page, reports, models, run_id, job, binding, ws, label, key)]

    state_outputs = [page_state, models_state, binding_state_, reports_state, selected_report]
    ctx.on_page_open(PAGE, {
        "fn": _refresh,
        "inputs": [workspace, version, model_state, measurement_state, models_state,
                   selected_report],
        "outputs": [*state_outputs, *cohort_outputs, *run_outputs, *result_outputs],
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
        key = key if find_model(models, key) is not None else None
        job = read_job()
        reports, run_id = _reports(page, models, label, current_report, job, request)
        # An import is made against one workspace: its drawer closes, its preview goes.
        return [models, binding, reports, run_id, None, gr.update(visible=False),
                *_cohort_outputs(page, binding, ws, label),
                *_run_outputs(page, models, binding, ws, label, key, protocol, job, _gpu(page)),
                *_result_outputs(page, reports, models, run_id, job, binding, ws, label, key)]

    workspace.input(_on_workspace,
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

    version.input(_on_version,
                  inputs=[page_state, binding_state_, models_state, workspace, version,
                          model_state,
                          measurement_state, selected_report],
                  outputs=[reports_state, selected_report, version, version_info, *run_outputs,
                           *result_outputs],
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
    model.input(_on_model, inputs=model_inputs, outputs=[*run_outputs, *result_outputs],
                show_progress="hidden", **one_at_a_time, **every_one)
    run_select.input(_on_run, inputs=[*model_inputs[:5], run_select, model_state,
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
        key = key if find_model(models, key) is not None else None
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
        """The read after Start, Cancel, Register or a note: the job card shows
        the job at once. It decides no end: an ended run's report is chosen by
        the poll's hand-off of that end, on its next tick."""
        job, label, reports, run_id = _job_read_again(page, models, current_report, reports,
                                                      request)
        gpu = _gpu(page)
        return [reports, run_id,
                *_start_outputs(page, models, binding, ws, label, key, protocol, gpu),
                *_card_outputs(page, job, gpu),
                *_result_outputs(page, reports, models, run_id, job, binding, ws, label, key)]

    after_outputs = [reports_state, selected_report, *start_outputs, *card_outputs,
                     *result_outputs]
    after_inputs = [page_state, models_state, binding_state_, workspace, model_state,
                    measurement_state,
                    selected_report, reports_state]

    def _on_start(page, models, ws, label, key, protocol, rec, det, request: gr.Request):
        result = start_test_run(page, models, ws, label, key, rec, det, request,
                                protocol=protocol)
        roots = (page or {}).get("roots") or []
        if result.get("success"):
            return _outcome_box("ok", "The test run started. It checks the model and the cohort "
                                      "again on the bytes it reads before measuring.")
        return _outcome_box("amber", str(result.get("error") or "Nothing was started."), roots)

    start.click(_on_start, inputs=[page_state, models_state, workspace, version, model,
                                   measurement, record, deterministic],
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
        """The job, read again while this page is shown: the job card, Cancel
        and what this session saw of the job (``job_state``, which only this
        poll writes, one tick after the other: a timer's tick is not sent while
        the one before is pending). Nothing it writes is drawn from the
        selection: when the job or the GPU line (Start's first reason) is not
        what it saw on its last tick, it hands the read over to
        `_after_job_change`, in the selection's group, naming what it saw then
        and what it sees now. So each change is handed over once."""
        if location != PAGE or not page:
            return [gr.update()] * (len(poll_outputs) + 1)
        job = read_job()
        gpu = _gpu(page)
        seen = seen or {}
        new_seen = {"job_id": job.get("job_id"), "status": job.get("status"), "gpu": list(gpu)}
        # Each hand-off is a new value (its own time), so its .change always fires.
        handoff = ({"seen": {"job_id": seen.get("job_id"), "status": seen.get("status")},
                    "now": {"job_id": job.get("job_id"), "status": job.get("status")},
                    "at": time.monotonic_ns()} if new_seen != seen else gr.update())
        # The GPU line is the header's GPU chip read now: when it is not
        # what this poll saw last, the chips are read again on this tick.
        return [new_seen, *_card_outputs(page, job, gpu), handoff,
                chips_on_change(seen.get("gpu"), new_seen["gpu"], request)]

    poll_outputs = [job_state, *card_outputs, job_changed]
    poll_inputs = [ctx.location, page_state, job_state]

    # The slow timer only: a listener on a shell timer sends a request on every
    # tick whatever it then does, so one on the fast timer would cost every open
    # page 40 requests a minute even with no job. A test run's state and elapsed
    # time are read every 5 s; Start, Cancel and Register read it at once.
    ctx.timer_slow.tick(_poll, inputs=poll_inputs, outputs=[*poll_outputs, ctx.chips],
                        queue=False, show_progress="hidden")

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
        sub = (f'<h2>Import a new cohort version</h2><span class="shep-muted">workspace '
               f'{e(ws)} · read on the CPU by this service · nothing is written before Commit'
               '</span>')
        return (gr.update(visible=True), _dd(intake_choices(listing), None),
                intake_note_html(listing, roots), sub,
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
        """The page read again, as a page open reads it, with the version just
        committed chosen."""
        return _refresh(ws, committed or label, key, protocol, current_models, current_report,
                        request)

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
#: ``PAGE_CSS``), scoped to its own ids. ``.shep-test-recm``: a described
#: measurement's metrics as recorded (R2), one click away under its figures, its
#: summary a 24 px target that shows the keyboard focus (skill §10).
#: ``.shep-test-replist``: the reports, a compact single-select table at the
#: Result panel's top (32-40 px rows, the skill's §8), in the panel's one scroll
#: region with the result under it; its header does not stick (the region
#: scrolls the table away under the result, and nothing sticky covers a row).
PAGE_CSS = f"""
{_PAGE} #shep-test-row {{ position: relative; gap: 12px; }}
{_PAGE} .shep-test-bd {{ padding: 12px 16px !important; gap: 10px !important; }}
{_PAGE} .shep-test-bd > *, {_PAGE} .shep-test-col > *, {_PAGE} .shep-test-drawer > * {{ flex-shrink: 0 !important; }}
{_PAGE} .shep-test-bd-flush {{ padding: 0 !important; }}
{_PAGE} .shep-test-pad {{ padding: 12px 16px; display: flex; flex-direction: column; gap: 10px; }}
{_PAGE} .shep-test-reason {{ margin: 2px 0; }}
{_PAGE} .shep-test-actions {{ gap: 10px !important; align-items: center !important; flex-wrap: wrap !important; }}
{_PAGE} .shep-test-actions > button {{ flex: none !important; }}
{_PAGE} .shep-test-ident {{ font-size: 15px; line-height: 1.5; overflow-wrap: anywhere; }}
{_PAGE} .shep-test-ident > div {{ margin: 1px 0; }}
{_PAGE} .shep-test-model-row {{ align-items: flex-end !important; gap: 8px !important; }}
{_PAGE} .shep-test-model-row > button {{ flex: none !important; min-width: 40px; margin-bottom: 4px; }}
{_PAGE} .shep-test-checks {{ gap: 18px !important; flex-wrap: wrap !important; }}
{_PAGE} .shep-test-checks > .block {{ flex: 0 1 auto !important; min-width: 0 !important; width: auto !important; }}
{_PAGE} .shep-test-replist {{ padding: 12px 16px 0; }}
{_PAGE} .shep-test-repwrap {{ margin: 0 !important; overflow: visible; }}
{_PAGE} .shep-test-reps th {{ position: static; }}
{_PAGE} .shep-test-reps td {{ padding: 7px 10px; overflow-wrap: anywhere; }}
{_PAGE} .shep-test-reps td.shep-test-rep-name {{ white-space: nowrap; }}
{_PAGE} .shep-test-reps tr[aria-disabled="true"] td {{ cursor: default; background: var(--shep-panel-2); }}
{_PAGE} .shep-test-res-hd {{ display: flex; align-items: baseline; gap: 10px; }}
{_PAGE} .shep-test-res-hd h3 {{ margin: 0; font-size: 15px; font-weight: 600; }}
{_PAGE} .shep-test-vd-why {{ margin-top: 6px; }}
{_PAGE} .shep-test-job {{ border: 1px solid var(--shep-border) !important; border-radius: 10px !important; background: var(--shep-panel-2) !important; padding: 0 0 8px !important; gap: 4px !important; }}
{_PAGE} .shep-test-job-top {{ align-items: center !important; flex-wrap: nowrap !important; gap: 8px !important; padding-right: 14px; }}
{_PAGE} .shep-test-job-top > .block {{ flex: 1 1 auto !important; min-width: 0 !important; }}
{_PAGE} .shep-test-job-top > button {{ flex: none !important; white-space: nowrap; }}
{_PAGE} #shep-test-cancel-reason {{ padding: 0 14px; }}
{_PAGE} .shep-test-job-hd {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 10px 14px 4px; }}
{_PAGE} .shep-test-job-hd h3 {{ margin: 0; font-size: 15px; font-weight: 600; }}
{_PAGE} .shep-test-job-bd {{ padding: 0 14px; display: flex; flex-direction: column; gap: 6px; overflow-wrap: anywhere; }}
{_PAGE} .shep-test-job-ft {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; padding: 8px 14px 0; }}
{_PAGE} .shep-test-log {{ margin: 0; padding: 8px 10px; background: #fff; border: 1px solid var(--shep-border); border-radius: 6px; font-size: 14px; white-space: pre-wrap; overflow-wrap: anywhere; font-family: var(--shep-mono); }}
{_PAGE} .shep-test-fail {{ color: var(--shep-red-700); font-size: 15px; }}
{_PAGE} .shep-test-units {{ font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} .shep-test-units > summary {{ cursor: pointer; min-height: 24px; line-height: 24px; width: max-content; max-width: 100%; }}
{_PAGE} .shep-test-units > summary:focus-visible {{ outline: 2px solid var(--shep-accent); outline-offset: 2px; }}
{_PAGE} .shep-test-fold summary {{ cursor: pointer; }}
{_PAGE} .shep-test-fold[open] summary {{ margin-bottom: 6px; }}
{_PAGE} .shep-test-recm {{ margin-top: 12px; font-size: 15px; overflow-wrap: anywhere; }}
{_PAGE} .shep-test-recm > summary {{ min-height: 24px; line-height: 24px; width: max-content; max-width: 100%; color: var(--shep-text-2); }}
{_PAGE} .shep-test-recm > summary:focus-visible {{ outline: 2px solid var(--shep-accent); outline-offset: 2px; }}
{_PAGE} .shep-test-outcome {{ display: flex; flex-direction: column; gap: 6px; font-size: 15px; }}
{_PAGE} .shep-test-chips {{ display: flex; flex-wrap: wrap; gap: 6px 8px; }}
{_PAGE} .shep-test-label {{ font-weight: 600; font-size: 15px; display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }}
{_PAGE} .shep-test-tiles {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }}
{_PAGE} .shep-test-tile {{ border: 1px solid var(--shep-border); border-radius: 8px; padding: 8px 10px; background: #fff; min-width: 0; }}
{_PAGE} .shep-test-tile-l {{ font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} .shep-test-tile-v {{ font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }}
{_PAGE} .shep-test-tile-n {{ font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} .shep-test-notes ul {{ margin: 4px 0 0; padding-left: 18px; font-size: 15px; }}
{_PAGE} .shep-test-noteform {{ padding: 10px 16px 8px !important; gap: 6px !important; border-top: 1px solid var(--shep-border); }}
{_PAGE} .shep-test-noterow {{ gap: 10px !important; flex-wrap: wrap !important; align-items: flex-start !important; }}
{_PAGE} .shep-test-noteform .shep-test-actions > .block {{ flex: 1 1 200px !important; min-width: 0 !important; }}
{_PAGE} .shep-test-terms {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1px 12px; font-size: 14px; margin-top: 2px; }}
{_PAGE} .shep-test-preview .shep-kv {{ padding: 10px 12px; overflow-wrap: anywhere; }}
{_PAGE} .shep-test-preview .shep-notice, {_PAGE} .shep-test-preview .shep-reason {{ margin: 0 12px 10px; }}
{_PAGE} .shep-test-drawer {{ box-shadow: -8px 0 24px rgba(20, 24, 40, .14); }}
{_PAGE} .shep-test-drawer-hd {{ flex-wrap: nowrap !important; align-items: center !important; }}
{_PAGE} .shep-test-drawer-hd > .block {{ flex: 1 1 auto !important; }}
{_PAGE} #shep-test-import-title h2 {{ display: inline; margin-right: 10px; font-size: 16px; font-weight: 600; }}
{_PAGE} .shep-test-form {{ gap: 10px !important; flex-wrap: wrap !important; }}
{_PAGE} .shep-test-form > * {{ min-width: 180px !important; }}
{_PAGE} #shep-test-start {{ width: 100%; }}
{_PAGE} .shep-test-runft {{ padding: 0 16px 12px !important; gap: 10px !important; }}
{_PAGE} .shep-test-select input {{ padding-right: 26px !important; }}
{_PAGE} .shep-test-rep-old, {_PAGE} .shep-test-more-fewer {{ display: none !important; }}
{_PAGE} .shep-test-all .shep-test-rep-old {{ display: table-row !important; }}
{_PAGE} .shep-test-all .shep-test-more-fewer {{ display: inline-flex !important; }}
{_PAGE} .shep-test-all .shep-test-more-all {{ display: none !important; }}
{_PAGE} .shep-test-more {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 8px 0 0; }}
{_PAGE} .shep-test-more button {{ font: inherit; font-size: 15px; color: var(--shep-accent-text); background: none; border: 0; padding: 2px 0; cursor: pointer; text-decoration: underline; text-underline-offset: 2px; }}
{_PAGE} .shep-panel-ft > button, {_PAGE} .shep-test-actions > button {{ white-space: nowrap; flex: none !important; }}
{_PAGE} .shep-panel-ft > .block {{ flex: 1 1 200px !important; min-width: 0 !important; }}
@media (min-width: 1400px) and (min-height: 760px) {{
  {_PAGE} .shep-test-col > .shep-test-bd {{ flex: 1 1 auto !important; min-height: 0; overflow: auto !important; flex-wrap: nowrap !important; }}
  {_PAGE} #shep-test-noteform {{ max-height: 50%; overflow: auto !important; flex-wrap: nowrap !important; }}
  {_PAGE} #shep-test-run > .shep-test-bd {{ flex: 0 1 auto !important; }}
  {_PAGE} #shep-test-runft {{ flex: 0 0 auto !important; min-height: 0; max-height: 64%; flex-wrap: nowrap !important; padding-top: 10px !important; border-top: 1px solid var(--shep-border); }}
  {_PAGE} #shep-test-runft > * {{ flex-shrink: 0 !important; }}
  {_PAGE} #shep-test-runft > #shep-test-jobcard {{ flex: 0 1 auto !important; min-height: 96px; overflow: auto !important; flex-wrap: nowrap !important; }}
  {_PAGE} #shep-test-jobcard > .shep-test-job-top {{ position: sticky; top: 0; z-index: 2; background: var(--shep-panel-2); flex-shrink: 0 !important; }}
  {_PAGE} #shep-test-jobcard > * {{ flex-shrink: 0 !important; }}
  {_PAGE} .shep-test-drawer {{ position: absolute !important; top: 0; right: 0; bottom: 0; width: min(780px, 62%) !important; z-index: 50; display: flex; flex-direction: column; max-height: 100%; }}
  {_PAGE} .shep-test-drawer > .shep-test-bd {{ flex: 1 1 auto !important; min-height: 0; overflow: auto !important; flex-wrap: nowrap !important; }}
}}
@media (max-width: 1399px), (max-height: 759px) {{
  {_PAGE} .shep-test-tiles {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  {_PAGE} .shep-test-drawer {{ position: fixed !important; top: 0; right: 0; bottom: 0; width: min(640px, 100%) !important; max-width: 100% !important; z-index: 60; display: flex; flex-direction: column; flex-wrap: nowrap !important; border-radius: 0 !important; }}
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
