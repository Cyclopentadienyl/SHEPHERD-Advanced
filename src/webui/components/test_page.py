"""
Research › Evaluate › Test: import a cohort version, run a test, keep and register its report (M5-2, UI-D).
==========================================================================================================
`docs/working/PLAN_TEST_RESULTS.md` §6 ("M5's boundaries": the Test tab on
M4's flow), D5, D7 and D8; the page contract is `src/webui/shell.py`'s
docstring. Three panels, as the mockup draws them (``test-wide.png``,
``test-import.png``):

- **Cohort**: the workspace (the server's list, the shared dataset renderer),
  the cohort version (the registry's list, with whether it is mapped against
  this workspace's graph), "🔒 Import a new version…", and the version's report
  store: every run directory, newest first (the newest `ROWS_SHOWN` and the
  selected one shown, "Show all N reports" for the rest), with what it kept
  and whether its report is registered (the registration's one helper,
  `server_lists.stored_report_state`, asked again for an operator beside the
  listed checkpoint Register would use: `refine_rows`, `ledger_checkpoint`)
  and D5's state of its key ("runs disagree — investigate"), both from the
  server's one run listing that ``GET /api/v1/cohorts`` shares
  (`server_lists.version_item`); a running test, or the last one that ended
  with nothing in the store, is the top row;
- **Run**: the model (WP1's checkpoint list of the workspace, by name and model
  fingerprint: listing opens every checkpoint, so it is an operator control and
  a read-only request is not shown it), the R10 evidence the server's setting
  names, "Register the result" and "Deterministic execution", the batch size and
  device the server fixes (D5), "🔒 Start test" through the job runner
  (`TrainingManager.start_test`) with the **full digests shown** as the expected
  ones, so the run refuses before measuring when the file or the version it
  reads is not the one chosen; then the job: its state, elapsed time, inputs,
  output tail, "🔒 Cancel" (the runner's stop for test runs only), and once it
  has ended, **what it left, from the runner** (its outcome file, kept and
  registered read from disk by WP4's helper; "no report was published" when
  there is no outcome file; a cancel says what was left);
- **Result**: the selected report as the record the ledger would hold
  (`build_record`, through the store's scan): the Mode C metrics over the cases
  ranked, coverage with every exclusion, relations, R10 in the caveats' words
  (*unverifiable* is a word, never a 0), environment, dataset origin, whether
  it is kept and registered; "🔒 Register this report" through the one
  registration function (`register_report`, the ledger beside the listed
  checkpoint holding the weights it measured, found by digest in the server's
  checkpoint list, never the report's recorded path); while runs under its key
  disagree, the block is shown and the button is off, and investigation notes
  (who, when, what) are listed and added (gated) **without lifting the block**;
  on a wide screen the note form stays at the panel's foot above Register,
  outside the scrolling report, so "🔒 Add note" is in view without scrolling.

**Import** (a drawer over Run and Result; D7): the files the operator placed
in the server's intake directory (`server_lists.list_intake_files`; a request
names a file and the server finds it in that listing again, refusing any other
name or a link that leaves the directory); version label, role (with whether it
offers ordering by test score in Models: `SHEPHERD_ORDERING_ROLES`, and never
``institutional_acceptance``) and the lost-terms policy; "🔒 Preview" runs the
importer (`prepare_import`, in this process on the CPU; it reads the
workspace's graph) and **shows only the exclusion reasons and their counts**
(no case id reaches the browser: the CLI preview on the machine and the
committed version keep them; a refused source is listed by case index,
`import_refusal`, and the job's output without the ids a refusal names,
`case_free`); "🔒 Commit" prepares again from the same inputs
and commits under the preview id shown, so changed inputs are refused
(`commit_import`). There is no upload control: Gradio's upload route is refused
in front of the mount.

**D8.** Every state-changing handler (import preview and commit, start,
cancel, register, add a note) and every read that opens checkpoints (the model
list, the registration's search for the checkpoint) asks D8's Gradio helper on
its own triggering request first, then calls the shared service in process;
the buttons a read-only request cannot use are disabled with D8's reason as
visible text, but the handler's check is the protection. A read-only request
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
reports, the report shown, and the list, the result, Start and its reason
drawn from them) runs in one queued group of one (`SELECTION_CONCURRENCY`):
page open, the workspace, the version, the model, a report's row, ↻, the read
after an act or an import, and the read after the poll saw the job or the GPU
line change (the poll hands that over through a hidden component's
``.change``, naming what it saw before and what it sees now). Every choice
joins it, also one made while an earlier one waits. So each starts from the
selection the one before it wrote, in the order they were asked, and a late
answer never writes back a selection it started from over one chosen
meanwhile. The report of a run that has just ended is chosen once for that
run: only the poll's hand-off of that end chooses it, and the poll, the one
writer of what the session saw of the job, hands each change over once.

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

from src.evaluation.caveats import (
    MODE_C_RESULT_LABEL,
    PRE_B1_NOTE,
    R10_LABEL_TEXT,
    R10_STATE_TEXT,
    R10_UNVERIFIABLE_REASON,
    RELATION_DISPLAY_NAMES,
)
from src.webui.components import renderers as r
from src.webui.components.chips import D8_SHORT, chips_on_change
from src.webui.components.help import help_button
from src.webui.components.model_list import item_key
from src.webui.components.workspaces import redact_paths
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

#: The roles a version may be given that never offer ordering (§4.9 question 1).
_NEVER_ORDERED = ("institutional_acceptance",)

#: The lost-terms policy, as the select names it.
POLICY_CHOICES = [("keep case", "keep"),
                  ("exclude case", "exclude")]

#: The words the dataset origin is shown with.
_ORIGIN_MEANING = {
    "synthetic": "a generated stand-in, not a clinical knowledge graph",
    "files": "built from identified source files",
}
_R10_EVIDENCE_WORDS = {
    "found": "evidence found",
    "not_found": "evidence not found",
    "unreadable": "evidence found, not readable",
    "not_recorded": "evidence not recorded by the checkpoint",
}
#: The Mode C metrics shown as tiles, in order.
_TILES = (("untruncated_mrr", "Mode C MRR (untruncated)"), ("hits@1", "Mode C Hits@1"),
          ("hits@5", "Mode C Hits@5"), ("hits@10", "Mode C Hits@10"))
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
        "cohort version is imported against and what a test run measures on; its checkpoints "
        "are the models offered. Shown by its dataset fingerprint, as on Datasets.",
    ),
    "shep-test-version": (
        "Cohort version",
        "A cohort version in the server's registry (SHEPHERD_REGISTRY): a source case set "
        "mapped to one graph's node indices, immutable, identified by the digest of its "
        "cohort.json. It can be measured only on a workspace whose graph it was mapped "
        "against; the run checks that again on the bytes it reads.",
    ),
    "shep-test-model": (
        "Model",
        "The workspace's checkpoints, read from the server's list (each file is read once "
        "and named by its model fingerprint). Listing them opens every checkpoint, so it is "
        "an operator control. The run is given the model fingerprint shown here and refuses "
        "before measuring if the file it reads holds other bytes.",
    ),
    "shep-test-record": (
        "Register the result",
        "Checked: when the run completes, its report is registered in the ledger beside the "
        "checkpoint automatically. Unchecked: the report is kept in the registry's report "
        "store and not registered; it can be registered later from this page, as a whole "
        "report. Every run keeps its report either way (D5).",
    ),
    "shep-test-deterministic": (
        "Deterministic execution",
        "Runs with PyTorch's deterministic algorithms, cuDNN deterministic and its benchmark "
        "off. An operation with no deterministic implementation refuses the run; it never "
        "falls back. It is recorded, so a deterministic run is a different measurement key, "
        "the way forward when runs under one key disagree. It guards against known "
        "non-deterministic operations; it does not certify a number.",
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
        "institutional_acceptance. Whether Models offers ordering by test score is decided by "
        "the role (the server's SHEPHERD_ORDERING_ROLES), never by the source's name; "
        "institutional_acceptance never offers it.",
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
        "What the investigation found. The note is kept in the report store with this key's "
        "reports; it deletes no report and does not lift the block.",
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
    "test-mode-c": ("Mode C result", PRE_B1_NOTE),
    "test-r10": (
        "R10 labels",
        " · ".join(f"{text['name']}: {text['definition']}"
                   for labels in R10_LABEL_TEXT.values() for text in labels.values()),
    ),
    "test-fixed": (
        "Fixed by the server",
        "A test run has no reason to vary these, so the job runner fixes them (D5): the "
        "batch size is the measurement's default and the device the deployment's "
        "(SHEPHERD_DEVICE), with no fallback. Only deterministic execution is chosen here, "
        "and it is recorded.",
    ),
    "test-evidence": (
        "R10 evidence",
        "Where the run looks for the training and validation inputs the checkpoint records, "
        "to count their overlap with the cohort (R10): the measured workspace always, and the "
        "directories the server's SHEPHERD_TRAINING_EVIDENCE_DIRS names. Only bytes whose "
        "digest the checkpoint records count; otherwise a label is unverifiable, never 0.",
    ),
}


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------
e = r.e

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
#: Where the importer's and the measurement's refusals name a case: the
#: importer's ``cases[3] (case_id 'x'): …`` and ``case_id 'x' is not printable``
#: (`cohort_registry.parse_source`), the measurement's ``sample 'x' has …``
#: (`measurement._assert_ids_in_range`).
_CASE_IDS = (
    (re.compile(r" \(case_id " + _PY_STR + r"\)"), ""),
    (re.compile(r"\bcase_id " + _PY_STR + r"(?= is not printable)"), "case_id"),
    (re.compile(r"\bsample " + _PY_STR), "a sample"),
)


def case_free(text: Any) -> str:
    """`text` with the case ids the importer's and the measurement's refusals
    name taken out (D7: the per-case detail stays on the machine): ``cases[3]
    (case_id 'x'): missing …`` reads ``cases[3]: missing …``, ``sample 'x' has
    …`` reads ``a sample has …``. Nothing else of a refusal names a case: the
    importer describes a term or a disease by its type, never its value."""
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
    """The visible reason a control a read-only request cannot use is off, in
    D8's short words (the header chip's ⓘ gives D8's whole reason)."""
    short = D8_SHORT.get(access.get("condition"), "request not readable")
    return f"Operator controls only: {short}."


def _reason_html(text: Optional[str], *, kind: str = "") -> str:
    if not text:
        return ""
    lock = '<span aria-hidden="true">🔒</span>' if kind == "lock" else ""
    return f'<div class="shep-reason shep-test-reason">{lock}<span>{e(text)}</span></div>'


def run_time(run_id: Any) -> Optional[str]:
    """"2026-10-17 15:24 UTC" from a run id (`create_report_dir` names it by
    its UTC time), else None."""
    match = _RUN_ID.match(str(run_id or ""))
    if not match:
        return None
    y, mo, d, h, mi, _s, _sha = match.groups()
    return f"{y}-{mo}-{d} {h}:{mi} UTC"


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
OK_MARK = f'<span class="shep-ok-t">{r.CHECK}</span>'


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
def _version_row(item: Any) -> Dict[str, Any]:
    row = asdict(item)
    row.pop("reports", None)
    row.pop("reports_problem", None)
    return row


def workspace_binding(directory: Optional[Path], row: Optional[Mapping[str, Any]]
                      ) -> Dict[str, Any]:
    """What a cohort version is compared with to say whether it was mapped
    against this workspace's graph: the four graph digests its manifest binds
    (one read of ``split_manifest.json``) and its verified dataset record's
    digest. A display hint only: the run compares the bytes it reads."""
    from src.kg.artifacts import GRAPH_ARTIFACTS, read_split_manifest

    if directory is None or row is None:
        return {"graph": None, "record_sha256": None, "problem": "no workspace is chosen"}
    try:
        artifacts = read_split_manifest(directory).manifest.get("artifacts")
    except (ValueError, OSError):
        return {"graph": None, "record_sha256": None,
                "problem": "its split_manifest.json cannot be read"}
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    record = row.get("dataset_record") or {}
    sha = record.get("sha256") if record.get("binding") == "verified" else None
    return {"graph": {role: artifacts.get(role) for role in GRAPH_ARTIFACTS},
            "record_sha256": sha,
            "problem": None if sha else "its dataset record is not verified"}


def binding_state(version: Mapping[str, Any], binding: Optional[Mapping[str, Any]]
                  ) -> Tuple[str, str]:
    """``("bound" | "another" | "unknown", words)``: whether `version` was
    mapped against the workspace `binding` describes, by the registry's one
    comparison (`verify_cohort_binding`, R3's)."""
    from src.evaluation.cohort_registry import CohortBindingMismatch, verify_cohort_binding

    if version.get("problem"):
        return "unknown", "the version cannot be read"
    if not binding or binding.get("problem") or not binding.get("graph"):
        why = (binding or {}).get("problem") or "no workspace is chosen"
        return "unknown", f"not checked against this workspace: {why}"
    try:
        verify_cohort_binding({"graph": version.get("graph"),
                               "dataset_record": version.get("dataset_record"),
                               "label": version.get("label")},
                              binding["graph"], binding["record_sha256"])
    except CohortBindingMismatch as exc:
        return "another", f"mapped against another graph ({', '.join(exc.roles)} differ)"
    return "bound", "bound to this graph"


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
        "versions": [_version_row(item) for item in versions.items] if versions else [],
        "versions_problem": versions.problem if versions is not None else None,
        "evidence": {"names": [Path(d).name for d in (evidence.value or ())],
                     "problem": evidence.problem},
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
    newest first, with what it kept, whether its report is registered and the
    D5 state of its key, as the server's one run listing gives them
    (`server_lists.version_item`, the listing ``GET /api/v1/cohorts`` shares,
    here with the problems' words for the operator); each run's key and record
    (the record the ledger would hold) from the same one scan of the store,
    found by the run's id for the bytes its state read; and the investigation
    notes kept with the key. The version is read once. Opens no checkpoint."""
    from src.api.services.record_states import read_report_store
    from src.api.services.server_lists import (
        ServerListRefused,
        resolve_cohort_version,
        stored_report_state,
        version_item,
    )
    from src.evaluation.report_store import read_investigation_notes

    empty = {"label": label, "runs": [], "problem": None, "notes": [], "notes_problems": [],
             "version_sha256": None}
    if not label:
        return empty
    try:
        version = resolve_cohort_version(label, environ=environ)
    except ServerListRefused as exc:
        return {**empty, "problem": str(exc)}
    store = read_report_store(version)
    item = version_item(version, report_state=stored_report_state, store=store, operator=True,
                        environ=environ)
    notes, notes_problems = [], []
    try:
        notes, notes_problems = read_investigation_notes(version)
    except Exception as exc:  # noqa: BLE001
        notes_problems = [f"the investigation notes could not be read ({type(exc).__name__})"]
    runs = []
    for report in item.reports:
        # By the digest of the bytes the run's state read: no key (and so no
        # note) for a report replaced since the scan read it; and byte-identical
        # copies under other runs are one report in the scan, so a run's own
        # digest names its record.
        digest = (report.state or {}).get("report_sha256")
        placed = store.scan.run_key(report.run_id, digest) if store.scan is not None else None
        key, unbuilt = placed if placed is not None else (None, None)
        built = store.scan.reports.get(key, {}).get(digest) if key is not None else None
        runs.append({
            "run_id": report.run_id, "when": run_time(report.run_id),
            "sha12": run_sha12(report.run_id), "state": report.state,
            "problem": report.problem, "key": list(key) if key is not None else None,
            "record": built[1] if built is not None else None,
            "contest": report.contest, "unbuilt": unbuilt,
        })
    return {"label": item.label, "version_sha256": item.manifest_sha256, "runs": runs,
            "problem": item.reports_problem, "notes": notes, "notes_problems": notes_problems}


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
            "sequence": fresh.sequence, "checked_beside": beside}


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
    """``(kind, words)`` of the GPU chip, read now (the header's wording)."""
    from src.webui.components.chips import read_gpu_chip

    if manager is None:
        from src.api.services.training_manager import training_manager as manager
    chip = read_gpu_chip(manager, operator=operator)
    return chip.kind, chip.text


# ---------------------------------------------------------------------------
# Choices
# ---------------------------------------------------------------------------
def model_choices(models: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """The model select's options: the epoch first (what tells a run's
    checkpoints apart when the select is narrow and its text ends in an
    ellipsis), the run, the model fingerprint or why a checkpoint cannot be
    tested, then the file name."""
    choices = []
    for item in models.get("items") or []:
        epoch = item.get("epoch")
        epoch_text = str(epoch) if isinstance(epoch, int) and not isinstance(epoch, bool) else "—"
        label = f"epoch {epoch_text} · {r.run_label(item.get('architecture'), item.get('run'))}"
        if item.get("problem") or not r.is_digest(item.get("sha256")):
            label += " · metadata not readable"
        else:
            label += f" · model fingerprint {item['sha256'][:12]}"
        choices.append((f"{label} · {item.get('filename')}", item_key(item)))
    return choices


def version_choices(page: Mapping[str, Any], binding: Optional[Mapping[str, Any]]
                    ) -> List[Tuple[str, str]]:
    choices = []
    for version in page.get("versions") or []:
        label = str(version.get("label"))
        if version.get("problem"):
            choices.append((f"{label} · cannot be read", label))
            continue
        counts = version.get("counts") or {}
        text = (f"{label} · role {version.get('role')} · {_count(counts.get('ranked_eligible'))} "
                f"of {_count(counts.get('source_cases'))} ranked-eligible")
        state, _words = binding_state(version, binding)
        if state == "another":
            text += " · another graph"
        choices.append((text, label))
    return choices


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
    labels = [v["label"] for v in page.get("versions") or []]
    if current in labels:
        return current
    bound = [v["label"] for v in page.get("versions") or []
             if binding_state(v, binding)[0] == "bound"]
    return (bound or labels or [None])[0]


# ---------------------------------------------------------------------------
# Rendering: Cohort
# ---------------------------------------------------------------------------
def cohort_header_html(page: Mapping[str, Any]) -> str:
    right = ("registry: server setting" if not page.get("registry_problem")
             else "registry: not available")
    head = (f'<div class="shep-panel-hd"><h2>Cohort</h2><span class="shep-panel-right">'
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


def version_html(page: Mapping[str, Any], label: Optional[str],
                 binding: Optional[Mapping[str, Any]]) -> str:
    """The chosen version: role, counts, its digest, and whether it was mapped
    against this workspace's graph."""
    if page.get("registry_problem"):
        return ""
    version = find(page.get("versions") or [], "label", label)
    if version is None:
        why = page.get("versions_problem") or (
            "No cohort version is in the registry yet: import one with “Import a new version…”."
            if not page.get("versions") else "No cohort version is chosen.")
        return _reason_html(redact(why, page.get("roots") or []))
    roots = page.get("roots") or []
    if version.get("problem"):
        return _reason_html(f"This version cannot be read: {redact(version['problem'], roots)}")
    counts = version.get("counts") or {}
    state, words = binding_state(version, binding)
    kind = {"bound": "ok", "another": "amber", "unknown": "grey"}[state]
    chip = r.chip(words, kind) if state != "bound" else (
        f'<span class="shep-tag shep-tag-ok">{e(words)} {OK_MARK}</span>')
    ordering = ("offers ordering by test score in Models" if version.get("offers_ordering")
                else "does not offer ordering by test score in Models")
    excluded = counts.get("excluded") if isinstance(counts.get("excluded"), Mapping) else {}
    reasons = " · ".join(f"{str(k).replace('_', ' ')} {_count(v)}" for k, v in excluded.items())
    imported = str(version.get("imported_at") or "").replace("T", " ").replace("Z", " UTC")
    return (
        '<div class="shep-test-vd">'
        f'<div>role {r.ident_html(version.get("role"))} · {_count(counts.get("source_cases"))} '
        f'source cases · {_count(counts.get("ranked_eligible"))} ranked-eligible'
        f'{" (" + e(reasons) + " excluded)" if reasons else ""}</div>'
        f'<div>{r.fingerprint_html("version", version.get("manifest_sha256"))} {chip}</div>'
        f'<div class="shep-sub">The role {e(ordering)}. Imported {e(imported or "—")} · '
        f'{e(version.get("source_format") or "")}</div></div>')


def _row_chip(run: Mapping[str, Any]) -> str:
    contest = run.get("contest") or {}
    state = run.get("state") or {}
    if contest.get("state") == "contested":
        return r.chip("runs disagree", "amber")
    if run.get("problem"):
        return r.chip("not read", "grey")
    if state.get("kept") != "kept":
        return r.chip("no report kept" if state.get("kept") == "not_kept" else "kept: not checked",
                      "grey")
    if state.get("registered") == "registered":
        seq = state.get("sequence")
        return (f'<span class="shep-tag shep-tag-ok">Registered {OK_MARK}'
                f'{" #" + str(seq) if isinstance(seq, int) else ""}</span>')
    if contest.get("state") == "cannot_be_checked":
        return r.chip("disagreement not checked", "grey")
    return r.chip("not registered" if state.get("registered") == "not_registered"
                  else "registration not checked", "neutral")


def _row_state(run: Mapping[str, Any], roots: Sequence[Tuple[str, str]]) -> str:
    """The row's state line, true to what the disk says."""
    if run.get("problem"):
        return e(redact(run["problem"], roots))
    state = run.get("state") or {}
    contest = run.get("contest") or {}
    kept = state.get("kept")
    if kept == "not_kept":
        return ("no report kept: report.json, the run's commit point, is not there (an "
                "interrupted or refused run); nothing in it is registered")
    if kept != "kept":
        return f"whether a report was kept cannot be checked: {e(redact(state.get('why_kept'), roots))}"
    registered = state.get("registered")
    parts = ["report kept"]
    if registered == "registered":
        seq = state.get("sequence")
        parts.append(f"registered{' #' + str(seq) if isinstance(seq, int) else ''}")
    elif registered == "not_registered":
        parts.append("not registered")
    else:
        parts.append("registration cannot be checked from the list")
    line = " · ".join(parts)
    if contest.get("state") == "contested":
        others = [x for x in contest.get("disagreeing_runs") or [] if x != run.get("run_id")]
        line += (f' · <span class="shep-amber-t">runs disagree — investigate: they differ on '
                 f'{e(", ".join(contest.get("differing") or ()))}'
                 f'{" with " + e(", ".join(run_time(x) or x for x in others)) if others else ""}'
                 '</span>')
    elif contest.get("state") == "cannot_be_checked":
        line += " · whether runs under its key disagree cannot be checked"
    return line


def _model_words(sha12: Optional[str], models: Mapping[str, Any]) -> str:
    item = next((i for i in models.get("items") or []
                 if r.is_digest(i.get("sha256")) and sha12 and i["sha256"].startswith(sha12)),
                None)
    if item is not None:
        return f"{e(r.run_label(item.get('architecture'), item.get('run')))} · model <code>{e(sha12)}</code>"
    return f"model <code>{e(sha12 or '—')}</code>"


def _ended_row(job: Mapping[str, Any], label: Any, run_ids: Sequence[str],
               models: Mapping[str, Any]) -> str:
    """The latest test run of this version as a row when it ended with no report
    the store lists (cancelled before its run directory, failed, or kept
    nothing): the runner remembers it until the next start or a restart."""
    status = job.get("status")
    inputs = job.get("inputs") or {}
    outcome = job.get("outcome") if isinstance(job.get("outcome"), Mapping) else {}
    if status not in ("cancelled", "failed", "completed") or inputs.get("version_label") != label:
        return ""
    report_dir = outcome.get("report_dir")
    if outcome.get("kept") is True or (report_dir and Path(str(report_dir)).name in run_ids):
        return ""
    how = {"cancelled": "interrupted: cancelled", "failed": "failed"}.get(status, "ended")
    if status == "failed" and job.get("exit_meaning"):
        how += f" ({job['exit_meaning']})"
    kept = ("no report kept" if outcome.get("kept") is False
            else "whether a report was kept cannot be checked (see Run)")
    when = utc_time(job.get("end_time")) or utc_time(job.get("start_time"))
    sha = str(inputs.get("checkpoint_sha256") or "")[:12]
    return ('<div class="shep-test-rep shep-test-rep-ended" aria-disabled="true">'
            f'<span class="shep-test-rep-main">{e("ended " + when) if when else "ended"} · '
            f'{_model_words(sha, models)}</span>'
            f'<span class="shep-test-rep-chip">{_pill(str(status))}</span>'
            f'<span class="shep-test-rep-sub"><span class="shep-amber-t">{e(how)}</span> · '
            f'{e(kept)} · the last test run on this server, not in the report store</span></div>')


def reports_html(reports: Mapping[str, Any], selected: Optional[str], *,
                 models: Optional[Mapping[str, Any]] = None, job: Optional[Mapping[str, Any]] = None,
                 roots: Sequence[Tuple[str, str]] = ()) -> str:
    """The version's report store as a list (newest first); each row selects
    the report in the Result panel. The newest `ROWS_SHOWN` rows are shown,
    with the running row, the selected report and "Show all N reports" (a
    version with hundreds of runs would otherwise push Run and Result a page
    long down at narrow sizes): the older rows are sent marked and shown by
    that control in the browser, so the choice survives the list's updates."""
    label = reports.get("label")
    if not label:
        return ""
    models = models or {}
    job = job or {}
    runs = reports.get("runs") or []
    rows = []
    inputs = job.get("inputs") or {}
    if job.get("status") in _BUSY and inputs.get("version_label") == label:
        sha = str(inputs.get("checkpoint_sha256") or "")[:12]
        started = utc_time(job.get("start_time"))
        rows.append(
            '<div class="shep-test-rep shep-test-rep-running" aria-disabled="true">'
            f'<span class="shep-test-rep-main">{e("started " + started) if started else "running"} · '
            f'{_model_words(sha, models)}</span>'
            f'<span class="shep-test-rep-chip">{_pill(job["status"])}</span>'
            f'<span class="shep-test-rep-sub">running · '
            f'{"registers when it completes" if inputs.get("record") else "kept, not registered, when it completes"}'
            '</span></div>')
    else:
        rows.append(_ended_row(job, label, [x["run_id"] for x in runs], models))
    hidden = 0
    for index, run in enumerate(runs):
        run_id = run["run_id"]
        on = run_id == selected
        old = index >= ROWS_SHOWN and not on
        hidden += old
        rows.append(
            f'<button type="button" class="shep-test-rep{" shep-on" if on else ""}'
            f'{" shep-test-rep-old" if old else ""}" '
            f'data-key="{e(run_id)}" aria-pressed="{"true" if on else "false"}">'
            f'<span class="shep-test-rep-main">{e(run.get("when") or run_id)} · '
            f'{_model_words(run.get("sha12"), models)}</span>'
            f'<span class="shep-test-rep-chip">{_row_chip(run)}</span>'
            f'<span class="shep-test-rep-sub">{_row_state(run, roots)}</span></button>')
    if hidden:
        rows.append(
            '<div class="shep-test-more">'
            f'<button type="button" class="shep-test-more-all" data-more="all">Show all '
            f'{len(runs)} reports ›</button>'
            f'<span class="shep-sub shep-test-more-all">{hidden} older not shown</span>'
            f'<button type="button" class="shep-test-more-fewer" data-more="fewer">Show the newest '
            f'{ROWS_SHOWN} only</button></div>')
    problem = reports.get("problem")
    count = len(runs)
    head = (f'<div class="shep-list-hd"><b>Reports of {e(label)}</b> · {count} · kept in the '
            "registry's report store</div>")
    body = "".join(rows) if any(rows) else (
        '<div class="shep-list-note">No run has kept a report on this version yet.</div>')
    if problem:
        body += f'<div class="shep-list-note">{e(redact(problem, roots))}</div>'
    return f'<div class="shep-list shep-test-reps" role="list" aria-label="Reports of {e(label)}">{head}{body}</div>'


# ---------------------------------------------------------------------------
# Rendering: Run
# ---------------------------------------------------------------------------
def run_header_html(workspace: Optional[str], label: Optional[str]) -> str:
    parts = []
    if workspace:
        parts.append(f"workspace <b>{e(workspace)}</b>")
    if label:
        parts.append(f"version <b>{e(label)}</b>")
    return (f'<div class="shep-panel-hd"><h2>Run</h2><span class="shep-panel-right">'
            f'{" · ".join(parts)}</span></div>')


def model_html(page: Mapping[str, Any], models: Mapping[str, Any], key: Optional[str]) -> str:
    """The chosen model, by the shared renderers; or why there is no list."""
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
        return '<div class="shep-sub">Choose the model to test.</div>'
    if item.get("problem") or not r.is_digest(item.get("sha256")):
        return _reason_html(f"{item.get('filename')}: metadata not readable · cannot be tested "
                            f"({redact(item.get('problem') or 'no digest was read', roots)})")
    workspaces = page.get("workspaces") or []
    validation = item.get("validation")
    figure = ((str(validation.get("metric")), float(validation["value"]))
              if isinstance(validation, Mapping) and isinstance(validation.get("value"), (int, float))
              else None)
    return (
        '<div class="shep-test-model">'
        f'<div class="shep-w600">{e(r.model_name_line(item))} '
        f'<span class="shep-muted">· {e(item.get("filename"))}</span></div>'
        f'<div class="shep-sub">{e(r.validation_text(figure))}</div>'
        f'<div>{r.fingerprint_html("model fingerprint", item["sha256"], help_key="model-fingerprint")}'
        f' · {r.trained_on_html(item.get("dataset_fingerprint"), workspaces)}</div></div>')


def evidence_html(page: Mapping[str, Any], workspace: Optional[str]) -> str:
    evidence = page.get("evidence") or {}
    names = evidence.get("names") or []
    measured = f"{e(workspace)} (measured, always)" if workspace else "the measured workspace (always)"
    if evidence.get("problem"):
        extra = f' · <span class="shep-amber-t">{e(redact(evidence["problem"], page.get("roots") or []))}</span>'
    elif names:
        extra = " · " + ", ".join(f"<code>{e(n)}</code>" for n in names) + " (the server's setting)"
    else:
        extra = " · no other evidence directory is set on the server"
    return (f'<div class="shep-test-kv"><span class="shep-k">R10 evidence'
            f'{help_button("test-evidence", "R10 evidence")}</span><span>{measured}{extra}</span></div>')


def fixed_html(page: Mapping[str, Any]) -> str:
    device = page.get("device") or {}
    if device.get("problem"):
        shown = f'<span class="shep-amber-t">{e(device["problem"])}</span>'
    else:
        value = device.get("value") or "auto"
        shown = (f"<code>{e(value)}</code>" + (" (requires CUDA)" if value == "auto" else ""))
    return (f'<div class="shep-test-kv"><span class="shep-k">Batch size'
            f'{help_button("test-fixed", "Fixed by the server")}</span>'
            f'<span>{BATCH_SIZE_SHOWN} · fixed by the server</span>'
            f'<span class="shep-k">Device</span><span>{shown} · fixed by the server</span></div>')


def start_state(page: Mapping[str, Any], models: Mapping[str, Any], workspace: Optional[str],
                label: Optional[str], key: Optional[str], binding: Optional[Mapping[str, Any]],
                gpu: Tuple[str, str]) -> Tuple[bool, Optional[str]]:
    """Whether "Start test" can be used now, and why not, in the order an
    operator fixes things: a busy or unread GPU lock comes before the choices,
    since no choice would enable Start while it lasts. Only a display: the
    handler checks D8 and the runner refuses again at start."""
    access = page.get("access") or {}
    roots = page.get("roots") or []
    if not access.get("allowed"):
        return False, operator_only(access)
    if page.get("registry_problem"):
        return False, "Disabled: no cohort registry is available (see Cohort)."
    kind, words = gpu
    if kind == "busy":
        return False, (f"Disabled: {words}. One managed GPU job at a time (training or "
                       "evaluation).")
    if kind == "unread":
        return False, "Disabled: the GPU lock cannot be read or used just now (see the GPU chip)."
    workspace_row = find(page.get("workspaces") or [], "name", workspace)
    if workspace_row is None:
        return False, "Disabled: choose a workspace."
    version = find(page.get("versions") or [], "label", label)
    if version is None:
        return False, "Disabled: choose a cohort version."
    if version.get("problem"):
        return False, "Disabled: this cohort version cannot be read."
    if binding_state(version, binding)[0] == "another":
        return False, ("Disabled: this cohort version was mapped against another graph than "
                       "this workspace's; the run would refuse it. Choose a workspace with its "
                       "graph, or import the source again against this workspace.")
    item = find_model(models, key)
    if item is None:
        return False, "Disabled: choose a model."
    if item.get("problem") or not r.is_digest(item.get("sha256")):
        return False, "Disabled: this checkpoint's metadata is not readable."
    if (page.get("evidence") or {}).get("problem"):
        return False, ("Disabled: the evidence directories setting is refused: "
                       + redact(page["evidence"]["problem"], roots))
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


def job_body_html(job: Mapping[str, Any], roots: Sequence[Tuple[str, str]],
                  gpu: Tuple[str, str]) -> str:
    """The job card's body: what the run was given, once it has ended what it
    left (the runner's outcome), its output tail, and while it runs the GPU
    line."""
    status = job.get("status") or "idle"
    if status == "idle":
        return ('<div class="shep-test-job-bd"><span class="shep-sub">No test run since the '
                "server started.</span></div>")
    inputs = job.get("inputs") or {}
    started = utc_time(job.get("start_time"))
    evidence = inputs.get("evidence_dirs") or []
    parts = [e(inputs.get("version_label") or "—"),
             f"model <code>{e(str(inputs.get('checkpoint_sha256') or '')[:12])}</code>",
             f"workspace {e(Path(str(inputs.get('workspace') or '')).name or '—')}",
             f"register {'on' if inputs.get('record') else 'off'}",
             f"deterministic {'on' if inputs.get('deterministic') else 'off'}",
             f"device {e(inputs.get('device') or 'auto')}",
             "R10 evidence: the workspace"
             + (f" + {len(evidence)} director{'ies' if len(evidence) != 1 else 'y'}" if evidence
                else "")]
    if started:
        parts.append(f"started {e(started)}")
    body = [f'<div class="shep-sub">{" · ".join(parts)}</div>']
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
    """What an ended run left, as the runner read it from disk: kept and
    registered (True, False or None for "cannot be checked") and its summary."""
    kept, registered = outcome.get("kept"), outcome.get("registered")
    chips = []
    chips.append(f'<span class="shep-tag shep-tag-ok">Report kept {OK_MARK}</span>' if kept is True
                 else r.chip("No report kept", "grey") if kept is False
                 else r.chip("Kept: cannot be checked", "grey"))
    if kept is not False:
        chips.append(f'<span class="shep-tag shep-tag-ok">Registered {OK_MARK}</span>'
                     if registered is True
                     else r.chip("Not registered", "neutral") if registered is False
                     else r.chip("Registered: cannot be checked", "grey"))
    summary = outcome.get("summary") or ""
    return (f'<div class="shep-test-outcome"><div class="shep-test-chips">{"".join(chips)}</div>'
            f'<div>{e(redact(summary, roots))}</div></div>')


# ---------------------------------------------------------------------------
# Rendering: Result
# ---------------------------------------------------------------------------
def _metric_text(name: str) -> str:
    if name == "untruncated_mrr":
        return "Mode C MRR (untruncated)"
    if name == "mean_rank":
        return "Mode C mean rank"
    if name.startswith("hits@"):
        return f"Mode C Hits@{name[5:]}"
    return f"Mode C {name}"


def _r10_label(label: str, name: str, part: Any, evidence: Any) -> str:
    """One R10 label in the caveats' words: a count with its denominator only
    where the state has one; *unverifiable* is a reason, never a 0."""
    state = part.get("state") if isinstance(part, Mapping) else None
    words = R10_STATE_TEXT.get(state) if isinstance(state, str) else None
    if state == "unverifiable" and isinstance(evidence, str):
        words = R10_UNVERIFIABLE_REASON.get(evidence, words)
    if words is None:
        return f"{e(name)}: not reported"
    if state not in ("overlap", "none"):
        if state == "unverifiable":
            reason = words.split(":", 1)
            return f"{e(name)}: <b>{e(reason[0])}</b>{e(':' + reason[1]) if len(reason) > 1 else ''}"
        return f"{e(name)}: {e(words)}"
    unit = "distinct test diseases" if label == "disease_overlap" else "test cases"
    count = f"{_count(part.get('k'))} of {_count(part.get('n'))} {unit}"
    if label == "disease_overlap":
        count += f"; {_count(part.get('cases'))} test cases with one of them"
    return f"{e(name)}: {e(words)} — {e(count)}"


def _r10_lines(r10: Any) -> List[Tuple[str, str]]:
    lines = []
    for scope, labels in R10_LABEL_TEXT.items():
        part = r10.get(scope) if isinstance(r10, Mapping) else None
        part = part if isinstance(part, Mapping) else {}
        evidence = (part.get("evidence") or {}).get("state") if isinstance(part.get("evidence"), Mapping) else None
        found = _R10_EVIDENCE_WORDS.get(evidence, "evidence not reported")
        text = (f'<span class="shep-muted">inputs ({e(found)}):</span> '
                + " · ".join(_r10_label(label, t["name"], part.get(label), evidence)
                             for label, t in labels.items()))
        lines.append((f"R10, {scope}", text))
    return lines


def _relations(relations: Any) -> str:
    if not isinstance(relations, Mapping):
        return "not reported"
    parts = []
    for name, state in relations.items():
        display = RELATION_DISPLAY_NAMES.get(name, name)
        label = f"{name} {display}" if re.fullmatch(r"R\d+", str(name)) else display
        if state == "verified":
            parts.append(f"{e(label)} {OK_MARK}")
        elif state == "not_applicable":
            parts.append(f"{e(label)}: not applicable")
        else:
            parts.append(f"{e(label)}: {e(state)}")
    return " · ".join(parts)


def _environment(record: Mapping[str, Any]) -> str:
    from src.evaluation.sidecar import environment_class

    try:
        text = environment_class(record)
    except ValueError:
        text = "not classified"
    facts = (record.get("environment") or {}).get("facts") if isinstance(record.get("environment"), Mapping) else None
    gpu = facts.get("gpu") if isinstance(facts, Mapping) else None
    if isinstance(gpu, Mapping):
        text += (f" · GPU {gpu.get('name') or 'name not read'}, driver "
                 f"{gpu.get('driver') or 'not read'}")
    return e(text)


def result_html(reports: Mapping[str, Any], run_id: Optional[str],
                roots: Sequence[Tuple[str, str]], *, job: Optional[Mapping[str, Any]] = None,
                notes: Optional[List[Mapping[str, Any]]] = None,
                notes_problems: Sequence[str] = ()) -> str:
    """The selected report as the record the ledger would hold: aggregates and
    states only, never a case or a path."""
    run = find(reports.get("runs") or [], "run_id", run_id)
    if run is None:
        return ('<div class="shep-panel-hd"><h2>Result</h2></div><div class="shep-test-pad">'
                '<p class="shep-muted">Select a report of the version to see its result.</p></div>')
    when = run.get("when") or run_id
    head = (f'<div class="shep-panel-hd"><h2>Result</h2><span class="shep-muted">report of '
            f'{e(when)}</span><span class="shep-panel-right"><code>{e(run_id)}</code></span></div>')
    state = run.get("state") or {}
    record = run.get("record")
    chips = []
    job_outcome = (job or {}).get("outcome") or {}
    if job_outcome.get("report_dir") and Path(str(job_outcome["report_dir"])).name == run_id:
        chips.append(_pill(str((job or {}).get("status"))))
    if state.get("kept") == "kept":
        chips.append(f'<span class="shep-tag shep-tag-ok">Report kept {OK_MARK}</span>')
    else:
        chips.append(r.chip("No report kept" if state.get("kept") == "not_kept"
                            else "Kept: cannot be checked", "grey"))
    if state.get("registered") == "registered":
        seq = state.get("sequence")
        chips.append(f'<span class="shep-tag shep-tag-ok">Registered {OK_MARK}'
                     f'{" record " + str(seq) if isinstance(seq, int) else ""}</span>')
    elif state.get("kept") == "kept":
        chips.append(r.chip("Not registered" if state.get("registered") == "not_registered"
                            else "Registration: cannot be checked from the list", "neutral"))
    contest = run.get("contest") or {}
    if contest.get("state") == "contested":
        chips.append(r.chip("Runs disagree — investigate", "amber"))
    parts = [f'<div class="shep-test-chips">{"".join(chips)}</div>']
    if run.get("problem"):
        parts.append(_reason_html(redact(run["problem"], roots)))
    if record is None:
        if state.get("kept") == "kept":
            why = run.get("unbuilt") or "; ".join(
                (run.get("contest") or {}).get("problems") or []) or (
                "the report store's comparison did not read it")
            parts.append(_reason_html("This report cannot be shown or compared: "
                                      + redact(why, roots)))
        else:
            parts.append(_reason_html(
                "No report was kept in this run directory: report.json, the run's commit "
                "point, is not there (an interrupted, cancelled or refused run). Nothing in it "
                "is registered, and nothing here can be."))
    else:
        parts.append(_record_html(record, run, state))
    parts.append(block_html(run, roots, notes or [], notes_problems))
    return head + f'<div class="shep-test-pad shep-test-result">{"".join(parts)}</div>'


def _record_html(record: Mapping[str, Any], run: Mapping[str, Any],
                 state: Mapping[str, Any]) -> str:
    cohort = record.get("cohort") if isinstance(record.get("cohort"), Mapping) else {}
    regime = record.get("regime") if isinstance(record.get("regime"), Mapping) else {}
    metrics = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
    n_ranked = record.get("n_ranked")
    counts = cohort.get("counts") if isinstance(cohort.get("counts"), Mapping) else {}
    deterministic = regime.get("deterministic_algorithms")
    line = (f'{e(cohort.get("label"))} (version <code>{e(str(record.get("cohort_digest") or "")[:12])}</code>, '
            f'role {r.ident_html(record.get("cohort_role"))}) · model '
            f'<code>{e(str(record.get("checkpoint_digest") or "")[:12])}</code> · '
            f'deterministic {"on" if deterministic is True else "off" if deterministic is False else "not recorded"}'
            f' · device <code>{e(regime.get("device"))}</code>')
    tiles = "".join(
        f'<div class="shep-test-tile"><div class="shep-test-tile-l">{e(label)}</div>'
        f'<div class="shep-test-tile-v">{e(_number(metrics.get(name)))}</div>'
        f'<div class="shep-test-tile-n">over {e(_count(n_ranked))} ranked</div></div>'
        for name, label in _TILES)
    excluded = counts.get("excluded") if isinstance(counts.get("excluded"), Mapping) else {}
    reasons = " · ".join(f"{str(k).replace('_', ' ')} {_count(v)}" for k, v in excluded.items())
    lines = [
        ("Coverage", f"<b>{e(_count(n_ranked))} of {e(_count(counts.get('source_cases')))}</b> source "
                     f"cases ranked ({e(_count(counts.get('ranked_eligible')))} eligible) · "
                     f"{e(_count(counts.get('n_excluded')))} excluded"
                     f"{': ' + e(reasons) if reasons else ''}"),
        ("Mean rank", f"{e(_metric_text('mean_rank'))} "
                      f"{e(_number(metrics.get('mean_rank')).rstrip('0').rstrip('.') if isinstance(metrics.get('mean_rank'), (int, float)) else '—')}"
                      f", over {e(_count(n_ranked))} ranked cases"),
        ("Other metrics", " · ".join(f"{e(_metric_text(k))} {e(_number(v))}"
                                     for k, v in sorted(metrics.items())
                                     if k not in {n for n, _ in _TILES} | {"mean_rank"}) or "—"),
        ("Relations", _relations(record.get("relations"))),
        *_r10_lines(record.get("r10")),
        ("Environment", _environment(record)),
    ]
    origin = (record.get("dataset_record") or {}).get("origin") if isinstance(record.get("dataset_record"), Mapping) else None
    meaning = _ORIGIN_MEANING.get(origin)
    lines.append(("Dataset", (f'<b class="shep-amber-t">{e(origin)}</b>' if origin == "synthetic"
                              else e(origin or "not reported"))
                  + (f" — {e(meaning)}" if meaning else "")))
    registered = state.get("registered")
    seq = state.get("sequence")
    beside = state.get("checked_beside")
    where = f" (the ledger beside {e(beside)} was read)" if beside else ""
    lines.append(("Registration",
                  f"registered as record {seq}{where}" if registered == "registered" and isinstance(seq, int)
                  else f"registered{where}" if registered == "registered"
                  else f"not registered{where}" if registered == "not_registered"
                  else "none found beside the checkpoint the run read, or its ledger cannot be "
                       "read (a registration beside another copy of the weights is looked for only "
                       "with the checkpoint list, an operator control)"))
    # The R10 labels' ⓘ sits beside the first R10 line, which it explains.
    kv = "".join(f'<span class="shep-k">{e(k)}'
                 f'{help_button("test-r10", "R10 labels") if k == "R10, training" else ""}</span>'
                 f'<span class="shep-v">{v}</span>' for k, v in lines)
    return (f'<div class="shep-sub">{line}</div>'
            f'<div class="shep-test-label">{e(MODE_C_RESULT_LABEL)}'
            f'{help_button("test-mode-c", "Mode C result")}</div>'
            f'<div class="shep-test-tiles">{tiles}</div><div class="shep-kv shep-test-lines">{kv}</div>')


def block_html(run: Mapping[str, Any], roots: Sequence[Tuple[str, str]],
               notes: Sequence[Mapping[str, Any]], notes_problems: Sequence[str] = ()) -> str:
    """D5's block on the report's key, and the investigation notes kept with
    it: who, when, what. A note never lifts the block. Notes that cannot be
    read are said, so "no note is kept" is never shown when one may be."""
    contest = run.get("contest") or {}
    key = run.get("key")
    parts = []
    if contest.get("state") == "contested":
        others = ", ".join(run_time(x) or x for x in contest.get("disagreeing_runs") or ())
        parts.append(
            '<div class="shep-notice shep-notice-amber shep-test-block"><b>Runs disagree — '
            f'investigate.</b> Reports under this key differ on: '
            f'{e(", ".join(contest.get("differing") or ()))} (runs {e(others)}). While they do, '
            "no report under this key is registered, checked or unchecked: registering one "
            "would be choosing between results. Every report stays in the store; compare their "
            "environment facts first. A deterministic re-run is another key. An investigation "
            "note records what was found; it does not lift the block.</div>")
    elif contest.get("state") == "cannot_be_checked":
        problems = "; ".join(redact(p, roots) for p in contest.get("problems") or ())
        parts.append(_reason_html("Whether runs under this key disagree cannot be checked, so no "
                                  f"report under it is registered: {problems}"))
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
                "The investigation notes cannot all be read, so a note kept for this key may be "
                "missing here: " + "; ".join(redact(p, roots) for p in notes_problems)))
        elif shown and not mine:
            parts.append('<div class="shep-sub">No investigation note is kept for this key.</div>')
    return "".join(parts)


def register_state(page: Mapping[str, Any], reports: Mapping[str, Any],
                   run_id: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Whether "Register this report" can be used, and why not."""
    access = page.get("access") or {}
    run = find(reports.get("runs") or [], "run_id", run_id)
    if run is None:
        return False, None
    if not access.get("allowed"):
        return False, operator_only(access)
    state = run.get("state") or {}
    if state.get("kept") != "kept":
        return False, "Disabled: no report was kept in this run."
    if state.get("registered") == "registered":
        return False, "Already registered."
    contest = (run.get("contest") or {}).get("state")
    if contest == "contested":
        return False, ("Disabled: runs disagree — investigate. Registering one would be choosing "
                       "between results; an investigation note does not lift this.")
    if contest == "cannot_be_checked" or run.get("record") is None:
        return False, ("Disabled: whether runs under this key disagree cannot be checked, so no "
                       "report under it is registered.")
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


def role_line(role: Any, *, environ: Optional[Mapping[str, str]] = None) -> str:
    """Whether a version with this role offers ordering by test score in Models
    (`role_offers_ordering`: listed in SHEPHERD_ORDERING_ROLES, never
    institutional_acceptance)."""
    from src.config import server_settings

    if not isinstance(role, str) or not role.strip():
        return ("Choose the version's role: its declared purpose (for example mygene2, "
                "research_selection or institutional_acceptance).")
    role = role.strip()
    if role in _NEVER_ORDERED:
        return (f"Role {role} never offers ordering by test score in Models, until the "
                "institution's formal evaluation rules exist.")
    if server_settings.role_offers_ordering(role, environ):
        return (f"Role {role} offers ordering by test score in Models: it is in the server's "
                "model-selection roles setting (SHEPHERD_ORDERING_ROLES). "
                "institutional_acceptance does not.")
    return (f"Role {role} does not offer ordering by test score in Models: it is not in the "
            "server's model-selection roles setting (SHEPHERD_ORDERING_ROLES). "
            "institutional_acceptance never does.")


def role_choices(page: Mapping[str, Any], *, environ: Optional[Mapping[str, str]] = None
                 ) -> List[str]:
    from src.config import server_settings

    roles: List[str] = []
    for role in [*(server_settings.ordering_roles(environ).value or ()),
                 *(v.get("role") for v in page.get("versions") or []), "mygene2",
                 "institutional_acceptance"]:
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
        ("Graph", "the workspace manifest’s digests: " + " · ".join(
            f"{e(role)} <code>{e(str(d or '')[:12])}</code>" for role, d in graph.items())),
        ("Dataset record", ('origin <b class="shep-amber-t">synthetic</b>' if record.get("origin") == "synthetic"
                            else f"origin {e(record.get('origin'))}")
                           + f", sha256 <code>{e(str(record.get('sha256') or '')[:12])}</code>…"),
        ("HPO · MONDO", f"HPO {_source_entry(record.get('hpo'))} · MONDO {_source_entry(record.get('mondo'))}"),
    ]
    kv = "".join(f'<span class="shep-k">{e(k)}</span><span class="shep-v">{v}</span>' for k, v in rows)
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
            f'</span></div><div class="shep-kv">{kv}</div>{warn}</div>')


# ---------------------------------------------------------------------------
# The operator controls (each: D8 on its own request first)
# ---------------------------------------------------------------------------
def _outcome_box(kind: str, text: str, roots: Sequence[Tuple[str, str]] = ()) -> str:
    return f'<div class="shep-outcome shep-outcome-{kind}">{e(redact(text, roots))}</div>'


def start_test_run(page: Mapping[str, Any], models: Mapping[str, Any], workspace: Any,
                   label: Any, key: Any, record: Any, deterministic: Any, request: Any, *,
                   manager: Any = None, environ: Optional[Mapping[str, str]] = None
                   ) -> Dict[str, Any]:
    """"Start test": D8 on this request; the workspace, the version and the
    checkpoint found in the server's lists again (never a client path); the
    **digests shown to the operator** passed as the expected ones, so the run
    refuses before measuring when the checkpoint or the version it reads is
    not the one chosen; the runner's one start for test runs."""
    from src.api.services.server_lists import (
        ServerListRefused,
        resolve_checkpoint,
        resolve_workspace,
    )
    from src.api.services.training_manager import TestRunInputs
    from src.config import server_settings

    access = operator_access(request)
    if not access.allowed:
        return {"success": False, "error": access.refusal}
    if manager is None:
        from src.api.services.training_manager import training_manager as manager
    version = find(page.get("versions") or [], "label", label)
    item = find_model(models, key)
    if version is None or version.get("problem") or not r.is_digest(version.get("manifest_sha256")):
        return {"success": False, "error": "Choose a cohort version the server lists. Nothing was started."}
    if (item is None or models.get("workspace") != workspace or item.get("problem")
            or not r.is_digest(item.get("sha256"))):
        return {"success": False, "error": "Choose a model the server lists in this workspace. "
                                           "Nothing was started."}
    registry = server_settings.cohort_registry(environ)
    evidence = server_settings.training_evidence_dirs(environ)
    if not registry.available:
        return {"success": False, "error": said(registry.problem, "Nothing was started.")}
    if not evidence.available:
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
        checkpoint_sha256=item["sha256"], evidence_dirs=tuple(evidence.value),
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
    version, D5's block and the checkpoint's bytes again. The page reads and
    judges no report itself (round 3 of the review fixes, F2)."""
    from src.api.services.server_lists import ServerListRefused, resolve_report
    from src.evaluation.registration import RegistrationRefused, corroboration_note, register_report

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
               "corroborated": f"Added to record {sequence} as a corroborating report: it agrees "
                               "on every compared field.",
               "already_registered": f"Already registered as record {sequence}; nothing was "
                                     "written."}.get(result.outcome, result.outcome)
    note = corroboration_note(result, path)
    if note:
        message += f" {note}"
    if result.usage_history_problem:
        message += f" {result.usage_history_problem[:1].upper()}{result.usage_history_problem[1:]}."
    return True, message


def add_note(reports: Mapping[str, Any], run_id: Any, who: Any, what: Any, request: Any, *,
             environ: Optional[Mapping[str, str]] = None) -> Tuple[bool, str]:
    """"Add note": D8 on this request; one investigation note kept with the
    report's key in the store (`add_investigation_note`). It lifts nothing."""
    from src.api.services.server_lists import ServerListRefused, resolve_cohort_version
    from src.evaluation.report_store import add_investigation_note

    access = operator_access(request)
    if not access.allowed:
        return False, access.refusal
    run = find(reports.get("runs") or [], "run_id", run_id)
    if run is None or not run.get("key"):
        return False, "Select a report whose key is known first. Nothing was written."
    try:
        version = resolve_cohort_version(reports.get("label"), environ=environ)
        add_investigation_note(version, tuple(run["key"]), who=str(who or ""), what=str(what or ""))
    except ServerListRefused as exc:
        return False, said(exc, "Nothing was written.")
    except (OSError, ValueError) as exc:
        return False, str(exc)
    return True, "The note was kept with this key's reports. The block stays until the reports agree."


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
  var row = e.target.closest ? e.target.closest('button[data-key]') : null;
  if (row) { trigger('click', {key: row.getAttribute('data-key')}); }
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
    # What this session saw of the job: the poll's alone (`_poll`).
    job_state = gr.State({})
    preview_state = gr.State(None)

    with gr.Row(elem_classes=["shep-page", "shep-test"], elem_id="shep-test-row", equal_height=False):
        # ------------------------------------------------------------ Cohort
        with gr.Column(elem_classes=["shep-panel", "shep-test-col"], elem_id="shep-test-cohort",
                       scale=11, min_width=320):
            cohort_hd = plain_html(cohort_header_html({}), elem_id="shep-test-cohort-hd")
            with gr.Column(elem_classes=["shep-test-bd"]):
                workspace = gr.Dropdown(label="Workspace", choices=[], value=None,
                                        elem_id="shep-test-workspace", interactive=True,
                                        elem_classes=["shep-test-select"])
                workspace_info = plain_html("", elem_id="shep-test-workspace-info")
                version = gr.Dropdown(label="Cohort version", choices=[], value=None,
                                      elem_id="shep-test-version", interactive=True,
                                      elem_classes=["shep-test-select"])
                version_info = plain_html("", elem_id="shep-test-version-info")
                with gr.Row(elem_classes=["shep-test-actions"]):
                    import_open = gr.Button("🔒 Import a new version…", variant="secondary",
                                            interactive=False, elem_id="shep-test-import-open",
                                            scale=0)
                import_reason = plain_html("", elem_id="shep-test-import-reason")
                import_msg = plain_html("", elem_id="shep-test-import-msg")
                reports_list = plain_html("", elem_id="shep-test-reports", js_on_load=_REPORTS_JS)
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
                    preview_btn = gr.Button("🔒 Preview", variant="secondary",
                                            elem_id="shep-test-preview", scale=0)
                preview_out = plain_html("", elem_id="shep-test-preview-card")
            with gr.Row(elem_classes=["shep-panel-ft"]):
                commit_btn = gr.Button("🔒 Commit", variant="primary", interactive=False,
                                       elem_id="shep-test-commit", scale=0)
                plain_html(
                    '<span class="shep-note">Commit prepares again and refuses if the preview id '
                    "changed.</span>", elem_id="shep-test-commit-note")
                commit_out = plain_html("", elem_id="shep-test-commit-out")
        # ------------------------------------------------------------ Run
        with gr.Column(elem_classes=["shep-panel", "shep-test-col"], elem_id="shep-test-run",
                       scale=10, min_width=320):
            run_hd = plain_html(run_header_html(None, None), elem_id="shep-test-run-hd")
            with gr.Column(elem_classes=["shep-test-bd"]):
                with gr.Row(elem_classes=["shep-test-model-row"]):
                    model = gr.Dropdown(label="Model", choices=[], value=None,
                                        elem_id="shep-test-model", interactive=False,
                                        elem_classes=["shep-test-select"])
                    model_reload = gr.Button("↻", variant="secondary", size="sm", scale=0,
                                             elem_id="shep-test-model-reload",
                                             interactive=False)
                model_info = plain_html("", elem_id="shep-test-model-info")
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
                start = gr.Button("🔒 ▶ Start test", variant="primary", interactive=False,
                                  elem_id="shep-test-start")
                start_reason = plain_html("", elem_id="shep-test-start-reason")
                action_out = plain_html("", elem_id="shep-test-action")
                with gr.Column(elem_classes=["shep-test-job"], elem_id="shep-test-jobcard"):
                    # Cancel beside the state, so it stays in view however long the output.
                    with gr.Row(elem_classes=["shep-test-job-top"]):
                        job_head = plain_html(job_head_html({}), elem_id="shep-test-job-head")
                        cancel = gr.Button("🔒 Cancel", variant="stop", interactive=False,
                                           elem_id="shep-test-cancel", scale=0, size="sm")
                    cancel_reason = plain_html("", elem_id="shep-test-cancel-reason")
                    job_out = plain_html(job_body_html({}, [], ("none", "")),
                                         elem_id="shep-test-job")
        # ------------------------------------------------------------ Result
        with gr.Column(elem_classes=["shep-panel", "shep-test-col"], elem_id="shep-test-result",
                       scale=14, min_width=360):
            with gr.Column(elem_classes=["shep-test-bd", "shep-test-bd-flush"]):
                result_out = plain_html(result_html({}, None, []), elem_id="shep-test-result-body")
            # The investigation note form (shown while runs under the report's
            # key disagree) is outside the scrolling body: on a wide screen it
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
                    note_btn = gr.Button("🔒 Add note", variant="secondary",
                                         elem_id="shep-test-note-add", scale=0)
                    note_reason = plain_html("", elem_id="shep-test-note-reason")
            note_out = plain_html("", elem_id="shep-test-note-out")
            with gr.Row(elem_classes=["shep-panel-ft"]):
                register = gr.Button("🔒 Register this report", variant="primary",
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

    def _start_outputs(page, models, binding, ws, label, key, gpu):
        """Start and its reason, from the selection (`start_outputs`)."""
        allowed = bool((page.get("access") or {}).get("allowed"))
        can_start, why = start_state(page, models, ws, label, key, binding, gpu)
        return [gr.update(interactive=can_start),
                _reason_html(why, kind="lock" if not allowed else "")]

    def _card_outputs(page, job, gpu):
        """The job card and Cancel, from the job alone (`card_outputs`)."""
        roots = page.get("roots") or []
        allowed = bool((page.get("access") or {}).get("allowed"))
        status = job.get("status") or "idle"
        can_cancel = allowed and status == "running"
        cancel_why = (operator_only(page.get("access") or {})
                      if not allowed and status == "running" else None)
        return [job_head_html(job), job_body_html(job, roots, gpu),
                gr.update(interactive=can_cancel), _reason_html(cancel_why, kind="lock")]

    def _run_outputs(page, models, binding, ws, label, key, job, gpu):
        allowed = bool((page.get("access") or {}).get("allowed"))
        return [
            run_header_html(ws, label),
            _dd(model_choices(models), key, interactive=allowed and bool(models.get("items"))),
            gr.update(interactive=allowed and bool(ws)),
            model_html(page, models, key),
            evidence_html(page, ws),
            fixed_html(page),
            *_start_outputs(page, models, binding, ws, label, key, gpu),
            *_card_outputs(page, job, gpu),
            gr.update(interactive=allowed),
            gr.update(interactive=allowed),
        ]

    # Start and its reason depend on the selection: only the selection's group
    # writes them. The job card and Cancel depend on the job alone: the poll
    # writes them too.
    start_outputs = [start, start_reason]
    card_outputs = [job_head, job_out, cancel, cancel_reason]
    run_outputs = [run_hd, model, model_reload, model_info, evidence, fixed, *start_outputs,
                   *card_outputs, record, deterministic]

    def _result_outputs(page, reports, models, run_id, job):
        """The list, the result and its controls for `reports` as this session
        keeps them (`_reports` and a row's click ask the shown report's
        registration first)."""
        roots = page.get("roots") or []
        can, why = register_state(page, reports, run_id)
        show_notes = notes_visible(reports, run_id)
        allowed = bool((page.get("access") or {}).get("allowed"))
        return [
            reports_html(reports, run_id, models=models, job=job, roots=roots),
            result_html(reports, run_id, roots, job=job, notes=reports.get("notes"),
                        notes_problems=reports.get("notes_problems") or ()),
            gr.update(interactive=can),
            _reason_html(why, kind="lock" if not allowed and why else ""),
            gr.update(visible=show_notes),
            gr.update(interactive=allowed),
            gr.update(interactive=allowed),
            gr.update(interactive=allowed),
            _reason_html(None if allowed else operator_only(page.get("access") or {}),
                         kind="lock"),
        ]

    result_outputs = [reports_list, result_out, register, register_reason, note_form, note_btn,
                      note_who, note_what, note_reason]

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
            _dd(version_choices(page, binding), label, interactive=registry_ok),
            version_html(page, label, binding),
            gr.update(interactive=allowed and registry_ok and bool(ws)),
            _reason_html(import_why, kind="lock" if not allowed else ""),
        ]

    cohort_outputs = [cohort_hd, workspace, workspace_info, version, version_info, import_open,
                      import_reason]

    def _select_report(reports: Mapping[str, Any], current: Any, job: Mapping[str, Any],
                       *, prefer_outcome: bool = False) -> Any:
        """The report to show: the one chosen, unless a run this session saw
        running has just ended with a report (then that one), or the chosen
        one is gone (then the newest)."""
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
    # group of one: the page open, the workspace, the version, the model, a
    # report's row, ↻, the read after an act or an import, and the read after
    # the poll saw the job or the GPU line change. They run first
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
    def _refresh(ws, label, key, current_models, current_report, request: gr.Request):
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
                *_run_outputs(page, models, binding, ws, label, key, job, gpu),
                *_result_outputs(page, reports, models, run_id, job)]

    state_outputs = [page_state, models_state, binding_state_, reports_state, selected_report]
    ctx.on_page_open(PAGE, {
        "fn": _refresh,
        "inputs": [workspace, version, model, models_state, selected_report],
        "outputs": [*state_outputs, *cohort_outputs, *run_outputs, *result_outputs],
        "show_progress": "hidden",
        **one_at_a_time,
    })

    # ------------------------------------------------------------------ choices
    def _on_workspace(page, ws, label, key, current_models, current_report, request: gr.Request):
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
                *_run_outputs(page, models, binding, ws, label, key, job, _gpu(page)),
                *_result_outputs(page, reports, models, run_id, job)]

    workspace.input(_on_workspace,
                    inputs=[page_state, workspace, version, model, models_state, selected_report],
                    outputs=[models_state, binding_state_, reports_state, selected_report,
                             preview_state, drawer, *cohort_outputs, *run_outputs,
                             *result_outputs],
                    show_progress="minimal", **one_at_a_time, **every_one) \
        .then(**ctx.chip_refresh)

    def _on_version(page, binding, models, ws, label, key, current_report, request: gr.Request):
        if not page:
            return [gr.update()] * (4 + len(run_outputs) + len(result_outputs))
        job = read_job()
        reports, run_id = _reports(page, models, label, current_report, job, request)
        # The select too: an event asked before this one (a workspace's, a
        # page open) may have set it to the version it was asked with.
        return [reports, run_id,
                _dd(version_choices(page, binding), label,
                    interactive=not page.get("registry_problem")),
                version_html(page, label, binding),
                *_run_outputs(page, models, binding, ws, label, key, job, _gpu(page)),
                *_result_outputs(page, reports, models, run_id, job)]

    version.input(_on_version,
                  inputs=[page_state, binding_state_, models_state, workspace, version, model,
                          selected_report],
                  outputs=[reports_state, selected_report, version, version_info, *run_outputs,
                           *result_outputs],
                  show_progress="hidden", **one_at_a_time, **every_one)

    def _on_model(page, models, binding, ws, label, key):
        if not page:
            return [gr.update()] * len(run_outputs)
        return _run_outputs(page, models, binding, ws, label, key, read_job(), _gpu(page))

    # In the selection's group: Start and its reason are written there only,
    # so a read handed over before this choice cannot write them back after it.
    model.input(_on_model, inputs=[page_state, models_state, binding_state_, workspace, version,
                                   model],
                outputs=run_outputs, show_progress="hidden", **one_at_a_time, **every_one)

    def _on_model_reload(page, binding, ws, label, key, current_models, request: gr.Request):
        if not page:
            return [gr.update()] * (1 + len(run_outputs))
        models = _models_for(page, current_models, ws, request, again=True)
        key = key if find_model(models, key) is not None else None
        return [models, *_run_outputs(page, models, binding, ws, label, key, read_job(), _gpu(page))]

    # In the selection's group: the listing it writes is the one the rows'
    # registration and Start are asked from. A second ↻ while one waits is
    # dropped (a click's "once"), which loses nothing: the one waiting lists
    # when it runs.
    model_reload.click(_on_model_reload,
                       inputs=[page_state, binding_state_, workspace, version, model, models_state],
                       outputs=[models_state, *run_outputs], show_progress="minimal",
                       **one_at_a_time) \
        .then(**ctx.chip_refresh)

    def _on_report(page, reports, models, evt: gr.EventData, request: gr.Request):
        raw = getattr(evt, "_data", None)
        key = raw.get("key") if isinstance(raw, Mapping) else None
        if not page or find(reports.get("runs") or [], "run_id", key) is None:
            return [gr.update()] * (2 + len(result_outputs))
        # The answer asked for this report stays with it when another is chosen.
        reports = refine_registration(page, reports, key, models, request=request)
        return [key, reports, *_result_outputs(page, reports, models, key, read_job())]

    reports_list.click(_on_report, inputs=[page_state, reports_state, models_state],
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

    def _after_act(page, models, binding, ws, key, current_report, reports,
                   request: gr.Request):
        """The read after Start, Cancel, Register or a note: the job card shows
        the job at once. It decides no end: an ended run's report is chosen by
        the poll's hand-off of that end, on its next tick."""
        job, label, reports, run_id = _job_read_again(page, models, current_report, reports,
                                                      request)
        gpu = _gpu(page)
        return [reports, run_id, *_start_outputs(page, models, binding, ws, label, key, gpu),
                *_card_outputs(page, job, gpu),
                *_result_outputs(page, reports, models, run_id, job)]

    after_outputs = [reports_state, selected_report, *start_outputs, *card_outputs,
                     *result_outputs]
    after_inputs = [page_state, models_state, binding_state_, workspace, model,
                    selected_report, reports_state]

    def _on_start(page, models, ws, label, key, rec, det, request: gr.Request):
        result = start_test_run(page, models, ws, label, key, rec, det, request)
        roots = (page or {}).get("roots") or []
        if result.get("success"):
            return _outcome_box("ok", "The test run started. It checks the model and the cohort "
                                      "again on the bytes it reads before measuring.")
        return _outcome_box("amber", str(result.get("error") or "Nothing was started."), roots)

    start.click(_on_start, inputs=[page_state, models_state, workspace, version, model, record,
                                   deterministic],
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

    def _after_job_change(page, models, binding, ws, key, current_report, reports, change,
                          request: gr.Request):
        """The poll's hand-off, in the selection's group: Start and its reason
        from the selection as it is now, and the reports read again. The one
        place an end is decided: when the poll saw a run busy (``seen``) and
        now sees it no longer busy (``now``), that run's report is chosen."""
        if not page or not isinstance(change, Mapping):
            return [gr.update()] * (2 + len(start_outputs) + len(result_outputs))
        seen, now = change.get("seen") or {}, change.get("now") or {}
        ended = (now.get("job_id") if seen.get("status") in _BUSY
                 and now.get("status") not in _BUSY else None)
        job, label, reports, run_id = _job_read_again(page, models, current_report, reports,
                                                      request, ended=ended)
        return [reports, run_id,
                *_start_outputs(page, models, binding, ws, label, key, _gpu(page)),
                *_result_outputs(page, reports, models, run_id, job)]

    job_changed.change(_after_job_change,
                       inputs=[page_state, models_state, binding_state_, workspace, model,
                               selected_report, reports_state, job_changed],
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
                gr.update(interactive=False, value="🔒 Commit"), "", "", "")

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
                gr.update(interactive=False, value="🔒 Commit"), ""
        can = bool(preview.get("samples_n"))
        return preview, preview_html(preview), gr.update(
            interactive=can, value=f"🔒 Commit as version {preview.get('label')}"), ""

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
        return ("", None, gr.update(interactive=False, value="🔒 Commit"), committed,
                gr.update(visible=False), box)

    def _after_commit(committed, ws, label, key, current_models, current_report,
                      request: gr.Request):
        """The page read again, as a page open reads it, with the version just
        committed chosen."""
        return _refresh(ws, committed or label, key, current_models, current_report, request)

    committed_label = gr.State(None)
    commit_btn.click(_on_commit, inputs=[page_state, preview_state, workspace, intake, new_label,
                                         new_role, new_policy],
                     outputs=[commit_out, preview_state, commit_btn, committed_label, drawer,
                              import_msg],
                     show_progress="minimal", concurrency_id="shep-test-import",
                     concurrency_limit=1) \
        .then(_after_commit, inputs=[committed_label, workspace, version, model, models_state,
                                     selected_report],
              outputs=[*state_outputs, *cohort_outputs, *run_outputs, *result_outputs],
              show_progress="hidden", **one_at_a_time) \
        .then(**ctx.chip_refresh)


_PAGE = "#" + page_elem_id(PAGE)

#: This page's style sheet (the page contract: listed in `src/webui/app.py`'s
#: ``PAGE_CSS``), scoped to its own ids.
PAGE_CSS = f"""
{_PAGE} #shep-test-row {{ position: relative; gap: 12px; }}
{_PAGE} .shep-test-bd {{ padding: 12px 16px !important; gap: 10px !important; }}
{_PAGE} .shep-test-bd > *, {_PAGE} .shep-test-col > *, {_PAGE} .shep-test-drawer > * {{ flex-shrink: 0 !important; }}
{_PAGE} .shep-test-bd-flush {{ padding: 0 !important; }}
{_PAGE} .shep-test-pad {{ padding: 12px 16px; display: flex; flex-direction: column; gap: 10px; }}
{_PAGE} .shep-test-reason {{ margin: 2px 0; }}
{_PAGE} .shep-test-actions {{ gap: 10px !important; align-items: center !important; flex-wrap: wrap !important; }}
{_PAGE} .shep-test-actions > button {{ flex: none !important; }}
{_PAGE} .shep-test-ident, {_PAGE} .shep-test-vd, {_PAGE} .shep-test-model {{ font-size: 15px; line-height: 1.5; overflow-wrap: anywhere; }}
{_PAGE} .shep-test-vd > div, {_PAGE} .shep-test-model > div {{ margin: 1px 0; }}
{_PAGE} .shep-test-kv {{ display: grid; grid-template-columns: max-content 1fr; gap: 4px 12px; font-size: 15px; align-items: baseline; overflow-wrap: anywhere; }}
{_PAGE} .shep-test-model-row {{ align-items: flex-end !important; gap: 8px !important; }}
{_PAGE} .shep-test-model-row > button {{ flex: none !important; min-width: 40px; margin-bottom: 4px; }}
{_PAGE} .shep-test-checks {{ gap: 18px !important; flex-wrap: wrap !important; }}
{_PAGE} .shep-test-checks > .block {{ flex: 0 1 auto !important; min-width: 0 !important; width: auto !important; }}
{_PAGE} .shep-test-reps {{ margin-top: 4px; }}
{_PAGE} .shep-test-rep {{
  display: grid; grid-template-columns: 1fr auto; gap: 2px 10px; align-items: center; width: 100%;
  text-align: left; padding: 9px 12px; border: 0; border-bottom: 1px solid #edf0f3; background: #fff;
  font: inherit; font-size: 15px; color: inherit; cursor: pointer;
}}
{_PAGE} .shep-test-rep:last-child {{ border-bottom: 0; }}
{_PAGE} .shep-test-rep:hover {{ background: var(--shep-panel-2); }}
{_PAGE} .shep-test-rep.shep-on {{ background: var(--shep-accent-50); box-shadow: inset 4px 0 0 var(--shep-accent); }}
{_PAGE} .shep-test-rep-running {{ cursor: default; }}
{_PAGE} .shep-test-rep-main {{ font-weight: 500; overflow-wrap: anywhere; min-width: 0; }}
{_PAGE} .shep-test-rep-chip {{ grid-row: 1 / span 2; grid-column: 2; }}
{_PAGE} .shep-test-rep-sub {{ font-size: 14px; color: var(--shep-text-2); overflow-wrap: anywhere; min-width: 0; }}
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
{_PAGE} .shep-test-fold summary {{ cursor: pointer; }}
{_PAGE} .shep-test-fold[open] summary {{ margin-bottom: 6px; }}
{_PAGE} .shep-test-outcome {{ display: flex; flex-direction: column; gap: 6px; font-size: 15px; }}
{_PAGE} .shep-test-chips {{ display: flex; flex-wrap: wrap; gap: 6px 8px; }}
{_PAGE} .shep-test-label {{ font-weight: 600; font-size: 15px; display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }}
{_PAGE} .shep-test-tiles {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }}
{_PAGE} .shep-test-tile {{ border: 1px solid var(--shep-border); border-radius: 8px; padding: 8px 10px; background: #fff; min-width: 0; }}
{_PAGE} .shep-test-tile-l {{ font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} .shep-test-tile-v {{ font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }}
{_PAGE} .shep-test-tile-n {{ font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} .shep-test-lines {{ overflow-wrap: anywhere; }}
{_PAGE} .shep-test-lines .shep-v {{ min-width: 0; }}
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
{_PAGE} .shep-test-rep-ended {{ cursor: default; }}
{_PAGE} .shep-test-rep-old, {_PAGE} .shep-test-more-fewer {{ display: none !important; }}
{_PAGE} .shep-test-all .shep-test-rep-old {{ display: grid !important; }}
{_PAGE} .shep-test-all .shep-test-more-fewer {{ display: inline-flex !important; }}
{_PAGE} .shep-test-all .shep-test-more-all {{ display: none !important; }}
{_PAGE} .shep-test-more {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 8px 12px; border-top: 1px solid #edf0f3; }}
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
