"""
Diagnosis: the "Model in use" card and its Details (M5-2, UI-B).
================================================================
`docs/working/PLAN_TEST_RESULTS.md` §6, "M5's boundaries": the served identity
appears once, on the Diagnosis page, as the "Model in use" card in the input
panel, directly above Run Diagnosis (the mockup's round 2; the header's
"Serving now" strip is gone).

**One reading of what is served.** The card reads the status through the
dashboard's one reader (`served_status.read_status`: the status route's own
function, in process), **with a 5 s limit** (`read_status_within`), per browser
session: on page load, when Diagnosis is shown, on the shell's slow timer while
it is shown, after each diagnosis, after a Load ("served model changed") and on
↻. What it knows lives in this session's `gr.State` (a plain dict,
`new_state`): no module state, nothing assumes one browser (BACKLOG 28).

**The states** (always words plus colour; `card_mode`):

- *current*: the last read succeeded and is not older than the result shown;
- *cannot check*: the last read failed (timed out after 5 s, or the status
  could not be read): grey, "last known HH:MM:SS · not current", Retry; the
  identity lines lose their ✓;
- *stale*: the last good read is older than the result shown: grey, "read
  HH:MM:SS · older than this result";
- *changed*: a read found something else served than this session's last read
  did: the dismissable "Model changed at HH:MM UTC — was …, now …";
- what is served: *no model loaded* (no pipeline: the service refuses every
  diagnosis, Run Diagnosis is disabled with its reason and a bubble, and the
  card offers "Load a model…"), or a model read from its checkpoint.

**What the card names, and where each fact comes from:**

- the name line ("GAT · run 2026-10-17 09:15 · epoch 5") from the checkpoint's
  place in the serving workspace's runs (the status's served paths, found
  again in the server's list by `resolve_checkpoint`, which reads no file) and
  the status's ``checkpoint_meta``. **A task model is named by its task**
  (K6 decision 4, the one place the card names it): "Disease task model",
  with "GATv2 · run … · epoch 5" on the line under it; the task is read from
  the served identity's scoring mode through the one task table
  (`served_task`, `src.core.types.SCORING_MODE_TASKS`);
- "model fingerprint <12 hex> ✓ verified" from the served identity;
- "trained on <name> · dataset fingerprint <12 hex>" from **the checkpoint's own
  record** (``training_input_digests.split_manifest``: the served file, found
  in the server's list by `resolve_checkpoint` and read once by the one
  checkpoint reader bound to the served digest, so other bytes are compared
  away on that read before they are parsed), **never the served workspace's
  manifest**. Reading a checkpoint is an operator control (SPEC_4 C3, D8), so
  a read-only request sees "not shown · reading the checkpoint is an operator
  control". Only the served file is opened, on a queued refresh, once per
  served model per session (what it records is kept while the served
  identity is the same); the timer's read never opens a checkpoint;
- the mode chips, only when true: amber "synthetic data", neutral "no SP
  table" (a ShepherdGNN file served without one, ``gnn_only``; a task model
  that uses no table is not missing one), red "GNN not loaded";
- "Recorded tests: <state>" from the ledger beside the checkpoint and the
  cohort versions' report stores, by the shared model list's own reader
  (`model_list.recorded_tests_for`): a state, never a number of a result, the
  same words as the checkpoint's row in "Change model…". The recorded tests and
  the listed workspaces (which name the datasets) are read again on every
  queued refresh (Diagnosis shown, a diagnosis, a Load, ↻), so a test recorded
  in Research › Test shows when Diagnosis is shown again; the timer's read
  keeps them (it reads the status alone);
- for a patients-like-me model, **the reference cohort versions** the page's
  Reference cohort select offers (`reference_cohorts`): every version the
  registry lists (`list_cohort_versions`, the list ``GET /cohorts`` reads; no
  report store is scanned), each with whether it was mapped against the
  serving workspace's graph (`workspaces.binding_state`, the registry's one
  comparison), read with the rest on a queued refresh and kept on the timer.
  A hint only: the server checks the version it is sent.

**Details** (the identity drawer's "Model in use") is the shared Model Facts
summary (`renderers.model_facts_html`) of the served checkpoint, with what only
a served model has added in this module (`served_facts_html`): the served
graph's and dataset record's digests with Copy, what serving reads, the
structural fingerprint warnings and the GNN state, and the link to the full
records in Research › Models.

Module: src/webui/components/diagnosis_card.py
"""
from __future__ import annotations

import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from src.core.types import SCORING_MODE_TASKS
from src.evaluation.caveats import TASK_NAMES
from src.webui.components import renderers as r
from src.webui.components.chips import D8_SHORT
from src.webui.components.help import copy_button
from src.webui.components.served_status import (
    STATUS_NOT_READ,
    read_status,
    served_checkpoint_sha,
    served_key,
)
from src.webui.components.workspaces import redact_paths

logger = logging.getLogger(__name__)

#: How long a status read may take before the card says it timed out.
STATUS_TIMEOUT_SECONDS = 5.0

#: Said where the trained-on dataset would be, to a request D8 refuses.
TRAINED_NOT_SHOWN = "not shown · reading the checkpoint is an operator control"
#: Said where it would be before a queued refresh has read it.
TRAINED_NOT_READ_YET = "not read yet"
#: The short words the card's Details use for the dataset fingerprint it has
#: not read, by the trained-on state (the long words are on the Trained on line).
_FINGERPRINT_NOT_READ = {"not_shown": "not shown", "pending": "not read yet"}

#: The card's own short words for a failed read (the full reason is in the
#: badge's line and in Details); a reason not listed is shown as it is.
_CARD_PROBLEM = {STATUS_NOT_READ: "status not read"}
_TIMED_OUT = re.compile(r"timed out \((?P<seconds>[0-9.]+) s\)")

#: The card's ⓘ (registered by the page as a topic).
CARD_HELP = (
    "Model in use",
    "What every session is served now, read from the server's status for this page: the model "
    "(its task, architecture, run, epoch), its model fingerprint, the dataset it records it was "
    "trained on, and the states that change results (synthetic data, a model file served "
    "without its shortest-path table). With no model loaded, diagnoses are refused until one "
    "is loaded. It is read when the page opens, every 5 s while "
    "Diagnosis is shown, after each diagnosis, after a model change and with ↻; the recorded "
    "tests and the dataset names are read again each time Diagnosis is shown, after each "
    "diagnosis, after a model change and with ↻. A read that fails or is older than the result "
    "shown turns the card grey: what it shows is then the last known state, not the current "
    "one.",
)


# ---------------------------------------------------------------------------
# Reading the status, with a limit
# ---------------------------------------------------------------------------
def card_problem(problem: Any) -> str:
    """A failed read's reason in the card's few words ("status not read",
    "timed out (5 s)"), so its state row fits the input column."""
    text = str(problem or STATUS_NOT_READ)
    return _CARD_PROBLEM.get(text, text)


def problem_sentence(problem: Any) -> str:
    """A failed read's reason for a sentence in parentheses: "timed out after
    5 s" (no parentheses inside), else the reason as it is."""
    text = str(problem or STATUS_NOT_READ)
    timed_out = _TIMED_OUT.fullmatch(text)
    return f"timed out after {timed_out.group('seconds')} s" if timed_out else text


def read_status_within(seconds: float = STATUS_TIMEOUT_SECONDS,
                       reader: Optional[Callable[[], Tuple[Optional[Dict[str, Any]],
                                                           Optional[str]]]] = None
                       ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``(status, None)`` from the dashboard's one status reader, or ``(None,
    reason)``: "timed out (5 s)" when it has not answered within `seconds`
    (the read then finishes on its own, and is dropped), or its own reason."""
    read = reader or read_status
    box: Dict[str, Any] = {}

    def run() -> None:
        try:
            box["answer"] = read()
        except Exception:  # noqa: BLE001 - read_status never raises; a test reader may
            logger.exception("The status read failed")
            box["answer"] = (None, STATUS_NOT_READ)

    worker = threading.Thread(target=run, name="shep-dx-status-read", daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive() or "answer" not in box:
        return None, f"timed out ({seconds:g} s)"
    status, problem = box["answer"]
    if status is None:
        return None, problem or STATUS_NOT_READ
    return status, None


# ---------------------------------------------------------------------------
# What is served, as the card names it
# ---------------------------------------------------------------------------
def _part(mapping: Any, *keys: str) -> Any:
    value = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def served_kind(status: Optional[Mapping[str, Any]]) -> str:
    """``no_model`` (no pipeline: diagnoses are refused), ``model`` (a
    checkpoint read; every served pipeline is one), or ``unknown`` (a status
    without an identity)."""
    if not isinstance(status, Mapping):
        return "unknown"
    if not status.get("initialized"):
        return "no_model"
    state = _part(status, "served_identity", "checkpoint", "state")
    if state == "verified" and served_checkpoint_sha(status):
        return "model"
    return "unknown"


def served_task(status: Optional[Mapping[str, Any]]) -> Optional[str]:
    """**The one reader of the served task** (K6 decision 7): ``"disease"``,
    ``"gene"`` or ``"patients_like_me"``, from the served identity's scoring
    mode through the core table (`SCORING_MODE_TASKS`); None for no model or
    a ShepherdGNN file's mode. The card's name line, its chips and serving
    note, and the page's task inputs, Run gating and request all read it."""
    if not isinstance(status, Mapping) or not status.get("initialized"):
        return None
    return SCORING_MODE_TASKS.get(_part(status, "served_identity", "scoring_mode"))


def task_model_name(task: str) -> str:
    """"Disease task model", "Gene task model", "Patients-like-me task model"
    (`caveats.TASK_NAMES`, the one vocabulary)."""
    return f"{TASK_NAMES[task]} task model"


#: What an absolute path the cohort registry does not hold reads as.
SERVER_PATH = "a server path"


def redacted(text: Any, environ: Optional[Mapping[str, str]] = None) -> str:
    """`text` with no absolute server path left (`workspaces.redact_paths`, the
    dashboard's one redaction): a path in the cohort registry reads
    ``<registry>/…`` (the Test page's name for it), any other "a server path".
    For the registry's and the reference cohort's refusals, which name the
    registry's files."""
    from src.config import server_settings

    roots: List[Tuple[str, str]] = []
    try:
        registry = server_settings.cohort_registry(environ)
        if registry.available:
            roots.append((str(registry.value), "<registry>"))
    except Exception:  # noqa: BLE001 - a registry not read is a path shown as "a server path"
        logger.debug("The cohort registry setting could not be read", exc_info=True)
    return redact_paths(text, roots, other=SERVER_PATH)


def reference_cohorts(serving: Optional[str], rows: List[Mapping[str, Any]],
                      directories: Mapping[str, Path], *,
                      environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The reference cohort versions a patients-like-me model may rank, as
    plain data: ``{"versions": [...], "problem": ...}``. Every version the
    registry lists (`list_cohort_versions`, without the report stores), each a
    `workspaces.version_row` with ``binding`` and ``binding_words``: whether
    it was mapped against the serving workspace's graph (`binding_state`, the
    registry's one comparison; a hint, the server checks the version it is
    sent). ``problem``: why no version is listed (no registry, or one
    refused). Every text is without a server path (`redacted`)."""
    from src.api.services.server_lists import list_cohort_versions
    from src.webui.components.workspaces import binding_state, version_row, workspace_binding

    try:
        listing = list_cohort_versions(environ=environ)
    except Exception:  # noqa: BLE001 - said as a problem, never raised
        logger.exception("The cohort registry could not be listed for the card")
        return {"versions": [], "problem": "the cohort registry could not be listed"}
    row = next((w for w in rows if w.get("name") == serving), None)
    binding = (workspace_binding(directories.get(serving), row)
               if serving is not None and row is not None
               else {"graph": None, "record_sha256": None,
                     "problem": "the serving workspace is not a listed workspace"})
    versions = []
    for item in listing.items:
        version = version_row(item)
        state, words = binding_state(version, binding)
        if version.get("problem"):
            version["problem"] = redacted(version["problem"], environ)
        versions.append({**version, "binding": state, "binding_words": words})
    return {"versions": versions,
            "problem": redacted(listing.problem, environ) if listing.problem else None}


def _validation(meta: Any) -> Optional[Dict[str, Any]]:
    from src.utils.checkpoint_paths import ranking_score_detail

    detail = ranking_score_detail(meta if isinstance(meta, dict) else None)
    return {"metric": detail[0], "value": detail[1]} if detail else None


def served_item(status: Mapping[str, Any], serving: Optional[str], *,
                environ: Optional[Mapping[str, str]] = None
                ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The served checkpoint as an item of the serving workspace's runs
    (architecture, run, file, the served digest, the status's epoch and the
    Trainer's validation figure), or ``(None, why)``. Its place comes from the
    status's served paths and is found again in the server's list
    (`resolve_checkpoint`, which reads no file); no checkpoint is opened."""
    from src.api.services.server_lists import ServerListRefused, resolve_checkpoint
    from src.utils.checkpoint_paths import RUN_CHECKPOINTS_DIRNAME, RUNS_DIRNAME

    sha = served_checkpoint_sha(status)
    if not sha:
        return None, None
    if serving is None:
        return None, "the serving workspace is not a listed workspace"
    checkpoint, data_dir = status.get("current_checkpoint_path"), status.get("current_data_dir")
    if not isinstance(checkpoint, str) or not isinstance(data_dir, str):
        return None, "the status names no checkpoint file"
    try:
        parts = Path(os.path.abspath(checkpoint)).relative_to(os.path.abspath(data_dir)).parts
    except ValueError:
        parts = ()
    if len(parts) != 5 or parts[0] != RUNS_DIRNAME or parts[3] != RUN_CHECKPOINTS_DIRNAME:
        return None, "the checkpoint in use is not in the serving workspace's runs"
    _, architecture, run, _, filename = parts
    try:
        resolve_checkpoint(serving, architecture, run, filename, environ=environ)
    except ServerListRefused:
        return None, "the checkpoint in use is not in the serving workspace's list"
    meta = status.get("checkpoint_meta") if isinstance(status.get("checkpoint_meta"), dict) else {}
    epoch = meta.get("epoch")
    return ({"workspace": serving, "architecture": architecture, "run": run,
             "filename": filename, "sha256": sha,
             "epoch": epoch if isinstance(epoch, int) and not isinstance(epoch, bool) else None,
             "validation": _validation(meta), "dataset_fingerprint": None, "problem": None},
            None)


def trained_on_listed(item: Mapping[str, Any], *, environ: Optional[Mapping[str, str]] = None
                      ) -> Dict[str, Any]:
    """The dataset the served checkpoint records it was trained on
    (``training_input_digests.split_manifest``): the file the server lists for
    the item (`resolve_checkpoint`, which reads no file) is read once by the one
    checkpoint reader (`read_checkpoint`, ``weights_only``; opening a
    checkpoint is an operator control), bound to the served digest (the full
    digest, ``expected_sha256``), as `resolve_checkpoint` asks of its callers:
    other bytes, a checkpoint or not, and a file removed are "not read" for
    what they are (another model than the one loaded, or gone), compared on
    that read before torch is handed them (round 3 of the review fixes,
    INT-R1's class). Only the served file is opened, never the other
    checkpoints of its runs, so a workspace with hundreds of checkpoints costs
    the card one read."""
    from src.api.services.server_lists import ServerListRefused, resolve_checkpoint
    from src.utils.checkpoint_io import CheckpointChanged, read_checkpoint

    try:
        path = resolve_checkpoint(item["workspace"], item["architecture"], item["run"],
                                  item["filename"], environ=environ)
    except ServerListRefused:
        return {"state": "not_read", "fingerprint": None,
                "problem": "the checkpoint in use is no longer listed"}
    try:
        read = read_checkpoint(path, map_location="cpu", weights_only=True,
                               expected_sha256=item["sha256"])
    except CheckpointChanged as exc:
        return {"state": "not_read", "fingerprint": None,
                "problem": ("the file the model was loaded from is gone" if exc.found is None
                            else f"the file now holds model {r.short_hex(exc.found)}, not the "
                                 "one loaded")}
    record = read.checkpoint if isinstance(read.checkpoint, dict) else {}
    digests = record.get("training_input_digests")
    recorded = digests.get("split_manifest") if isinstance(digests, dict) else None
    return {"state": "read", "fingerprint": recorded if r.is_digest(recorded) else None,
            "problem": None}


def recorded_for(item: Mapping[str, Any], directory: Path, *,
                 environ: Optional[Mapping[str, str]] = None) -> Optional[Dict[str, Any]]:
    """The served checkpoint's recorded tests, as the shared list reads them
    (`model_list.recorded_tests_for`: the ledger beside it, the report stores
    for contested; no checkpoint is opened)."""
    from src.webui.components.model_list import item_key, recorded_tests_for

    try:
        found = recorded_tests_for(directory, [item], environ=environ).get(item_key(item))
    except Exception:  # noqa: BLE001 - the card then says "not read"
        logger.exception("The recorded tests of the model in use could not be read")
        return None
    return found._asdict() if found is not None else None


def served_facts(status: Mapping[str, Any], *, operator: bool, open_checkpoints: bool,
                 trained: Optional[Mapping[str, Any]] = None,
                 environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """What the card adds to the status, as plain data: the listed workspaces
    (to name datasets), the serving workspace, the served checkpoint as an item
    of its runs, its recorded tests and the dataset it records it was trained
    on. `open_checkpoints` lets the served checkpoint be read (a queued
    refresh of a request D8 allows); otherwise the trained-on dataset is "not read yet"
    (or "not shown" to a request D8 refuses). `trained` is what an earlier
    read of the same served checkpoint found: it is used as it is, and the
    checkpoint is not opened again. A patients-like-me model adds the
    reference cohort versions (``cohorts``, `reference_cohorts`; None for
    any other)."""
    from src.webui.components.served_status import serving_workspace
    from src.webui.components.workspaces import read_workspaces

    facts: Dict[str, Any] = {"key": served_key(status), "operator": operator, "workspaces": [],
                             "serving": None, "item": None, "item_problem": None,
                             "recorded": None, "trained": None, "cohorts": None}
    try:
        listed = read_workspaces(environ)
        facts["workspaces"] = listed["rows"]
        facts["serving"] = serving_workspace(status, listed["rows"],
                                             directories=listed["directories"])
        directories = listed["directories"]
    except Exception:  # noqa: BLE001 - the card names no workspace then
        logger.exception("The workspaces could not be listed for the card")
        directories = {}
    if served_task(status) == "patients_like_me":
        facts["cohorts"] = reference_cohorts(facts["serving"], facts["workspaces"], directories,
                                             environ=environ)
    if served_kind(status) != "model":
        return facts
    item, why = served_item(status, facts["serving"], environ=environ)
    facts["item"], facts["item_problem"] = item, why
    if item is None:
        facts["trained"] = {"state": "not_shown" if not operator else "not_read",
                            "fingerprint": None, "problem": why}
        return facts
    if facts["serving"] in directories:
        facts["recorded"] = recorded_for(item, directories[facts["serving"]], environ=environ)
    if not operator:
        facts["trained"] = {"state": "not_shown", "fingerprint": None, "problem": None}
    elif isinstance(trained, Mapping) and trained.get("state") in ("read", "not_read"):
        facts["trained"] = dict(trained)
    elif open_checkpoints:
        try:
            facts["trained"] = trained_on_listed(item, environ=environ)
        except Exception:  # noqa: BLE001 - said as not read, never raised
            logger.exception("The checkpoint in use could not be read")
            facts["trained"] = {"state": "not_read", "fingerprint": None,
                                "problem": "the checkpoint in use could not be read"}
    else:
        facts["trained"] = {"state": "pending", "fingerprint": None, "problem": None}
    if facts["trained"]["state"] == "read":
        # The same bytes (the full digest matched): the status's own figures
        # (the pipeline's read) stay; the file adds what only it records.
        facts["item"] = {**item, "dataset_fingerprint": facts["trained"]["fingerprint"]}
    return facts


# ---------------------------------------------------------------------------
# The card's state (one browser session's)
# ---------------------------------------------------------------------------
def new_state() -> Dict[str, Any]:
    """A session's card before its first read."""
    return {"attempt_at": None, "ok": None, "problem": None, "good_at": None, "status": None,
            "key": None, "facts": None, "change": None, "operator": False, "d8_short": None}


def stamp(moment: datetime) -> str:
    """Kept to the microsecond: a read in the same second as the result it
    follows must not compare as older than it (shown to the second)."""
    return moment.isoformat(timespec="microseconds")


def parse_local(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment.replace(tzinfo=None) if moment.tzinfo is None else None


def hhmmss(value: Any) -> str:
    """A local stamp as the dashboard shows a clock time ("15:50:35 UTC",
    `renderers.clock_utc`), or "—"."""
    return r.clock_utc(parse_local(value)) or "—"


def advance(state: Optional[Mapping[str, Any]], status: Optional[Mapping[str, Any]],
            problem: Optional[str], *, now: datetime, operator: bool,
            d8_short: Optional[str], open_checkpoints: bool, reread: bool = False,
            facts: Optional[Callable[..., Dict[str, Any]]] = None) -> Dict[str, Any]:
    """The session's card state after one read (`status`, or `problem`).

    A failed read keeps the last known status (shown grey, never as current).
    A read that finds another served identity than this session's last read
    (`served_key`: every part, and the served workspace) sets the change
    notice. What the card adds to the status (`served_facts`) is read again
    on every queued refresh (`open_checkpoints`: the listed workspaces, the
    recorded tests), keeping what the served checkpoint records while the
    served identity and the request's D8 answer are the same (the file is
    opened once); ↻ (`reread`) reads that again too. The timer's read
    (`open_checkpoints` False) keeps them all, as long as the served identity
    and the D8 answer are the same."""
    facts = facts or served_facts
    new = dict(state) if isinstance(state, Mapping) else new_state()
    new.update(attempt_at=stamp(now), operator=operator, d8_short=d8_short)
    if status is None:
        new.update(ok=False, problem=problem or STATUS_NOT_READ)
        return new
    key = served_key(status)
    previous = new.get("status")
    if previous is not None and new.get("key") != key:
        was, was_sha = served_kind(previous), served_checkpoint_sha(previous)
        is_sha = served_checkpoint_sha(status)
        new["change"] = {"at": r.clock_utc(now, seconds=False),
                         "was": r.short_hex(was_sha) if was_sha else _kind_word(was),
                         "now": r.short_hex(is_sha) if is_sha else _kind_word(served_kind(status)),
                         "same_model": bool(was_sha) and was_sha == is_sha}
    kept = new.get("facts")
    same = (not reread and isinstance(kept, Mapping) and kept.get("key") == key
            and kept.get("operator") == operator)
    if same and not open_checkpoints:
        found = dict(kept)
    else:
        found = facts(status, operator=operator, open_checkpoints=open_checkpoints,
                      trained=kept.get("trained") if same else None)
    new.update(ok=True, problem=None, good_at=stamp(now), status=dict(status), key=key,
               facts=found)
    return new


def _kind_word(kind: str) -> str:
    return {"no_model": "no model loaded"}.get(kind, "not known")


def dismiss_change(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {**state, "change": None}


def card_mode(state: Mapping[str, Any], produced_at: Any = None) -> str:
    """``checking`` (no read yet), ``unknown`` (no read succeeded yet),
    ``cannot_check`` (the last read failed; a last known status is shown),
    ``stale`` (the last good read is older than the result shown), or
    ``current``."""
    if state.get("ok") is None:
        return "checking"
    if state.get("status") is None:
        return "unknown"
    if not state.get("ok"):
        return "cannot_check"
    produced, good = parse_local(produced_at), parse_local(state.get("good_at"))
    if produced is not None and good is not None and good < produced:
        return "stale"
    return "current"


def run_blocked(state: Mapping[str, Any]) -> bool:
    """Run Diagnosis is off when the last read succeeded and found no model
    loaded (no pipeline: the service refuses every diagnosis). An unreadable
    status leaves it on (the service answers, or refuses with its reason)."""
    return bool(state.get("ok")) and served_kind(state.get("status")) == "no_model"


# ---------------------------------------------------------------------------
# Rendering the card
# ---------------------------------------------------------------------------
_CHECK = ('<svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true" focusable="false">'
          '<path d="M3.2 8.6l3 3 6.6-7.4" fill="none" stroke="currentColor" stroke-width="2.2" '
          'stroke-linecap="round" stroke-linejoin="round"/></svg>')


def time_html(state: Mapping[str, Any], produced_at: Any = None) -> str:
    """The title row's right side: "checked HH:MM:SS UTC" while the card is current."""
    if card_mode(state, produced_at) != "current":
        return ""
    return f'<span class="shep-dx-checked">checked {r.e(hhmmss(state.get("good_at")))}</span>'


def _trained_html(facts: Mapping[str, Any]) -> str:
    trained = facts.get("trained") or {}
    state = trained.get("state")
    if state == "read":
        return r.trained_on_html(trained.get("fingerprint"), facts.get("workspaces") or [])
    if state == "not_shown":
        return (f'<span class="shep-trained">trained on: <span class="shep-muted">🔒 '
                f'{r.e(TRAINED_NOT_SHOWN)}</span></span>')
    if state == "pending":
        return (f'<span class="shep-trained">trained on: <span class="shep-muted">'
                f'{r.e(TRAINED_NOT_READ_YET)} · </span><button type="button" '
                'class="shep-dx-link" data-key="refresh">read it</button></span>')
    why = trained.get("problem") or "not read"
    return (f'<span class="shep-trained">trained on: <span class="shep-muted">not read · '
            f'{r.e(why)}</span></span>')


def _chips(status: Mapping[str, Any], *, grey: bool) -> List[str]:
    identity = status.get("served_identity") or {}
    kind = served_kind(status)
    chips: List[Tuple[str, str]] = []
    if kind == "model" and not status.get("gnn_ready", True):
        chips.append(("GNN not loaded", "red"))
    if _part(identity, "dataset_record", "origin") == "synthetic":
        chips.append(("synthetic data", "amber"))
    # Only a file served without the table it would mix (the test
    # `served_warnings` uses): a task model that uses none is not missing one.
    if kind == "model" and _part(identity, "scoring_mode") == "gnn_only":
        chips.append(("no SP table", "neutral"))
    return [r.chip(text, "grey shep-tag-dashed" if grey else kind_) for text, kind_ in chips]


def _recorded_html(facts: Mapping[str, Any], *, grey: bool) -> str:
    if grey:
        return f'<span class="shep-dx-tests">{r.e("Recorded tests: not read")}</span>'
    raw = facts.get("recorded")
    recorded = r.RecordedTests(**raw) if isinstance(raw, Mapping) else None
    text = r.recorded_tests_text(recorded)
    if recorded is None and facts.get("item_problem"):
        text += f" · {facts['item_problem']}"
    return (f'<span class="shep-dx-tests">{r.e(text)} <button type="button" class="shep-dx-link" '
            'data-key="evidence" aria-label="Recorded tests: open Details at Evidence">›</button>'
            '</span>')


def body_html(state: Mapping[str, Any], produced_at: Any = None) -> str:
    """The card's body: the change notice, a grey state row when the card is
    not current, then what is served."""
    mode = card_mode(state, produced_at)
    if mode == "checking":
        return '<div class="shep-dx-card-bd"><p class="shep-muted">Checking the model in use…</p></div>'
    parts: List[str] = []
    change = state.get("change")
    if isinstance(change, Mapping):
        if change.get("same_model"):
            head = f"What is served changed at {r.e(change.get('at'))}"
            rest = f"same model {r.e(change.get('now'))}, another part changed"
        else:
            head = f"Model changed at {r.e(change.get('at'))}"
            rest = f"was {r.e(change.get('was'))}, now {r.e(change.get('now'))}"
        parts.append(f'<div class="shep-notice shep-notice-amber shep-dx-change" role="status">'
                     f'<span><b>{head}</b> — {rest}</span>'
                     '<button type="button" class="shep-dx-x" data-key="dismiss" '
                     'aria-label="Dismiss the change notice">×</button></div>')
    retry = ('<button type="button" class="shep-btn shep-dx-retry" data-key="refresh">'
             '↻ Retry</button>')
    # The failure in the card's few words (the full reason in its title, the
    # badge's line and Details), Retry beside it: the row stays on one line
    # in the 300-370 px column, and "last known" fits the next one.
    problem = state.get("problem")
    failed = (f'<div class="shep-dx-state"><span class="shep-tag shep-tag-grey">cannot check'
              f'</span> <span title="{r.e(problem_sentence(problem))}">'
              f'{r.e(card_problem(problem))}</span> {retry}</div>')
    if mode == "unknown":
        parts.append(failed + '<p class="shep-muted">Nothing is known yet about what is served.</p>')
        return '<div class="shep-dx-card-bd shep-dx-card-grey">' + "".join(parts) + "</div>"
    grey = mode in ("cannot_check", "stale")
    if mode == "cannot_check":
        parts.append(failed + '<div class="shep-dx-state"><span class="shep-tag shep-tag-grey">'
                     f'last known {r.e(hhmmss(state.get("good_at")))} · not current</span></div>')
    elif mode == "stale":
        parts.append(f'<div class="shep-dx-state"><span class="shep-tag shep-tag-grey">read '
                     f'{r.e(hhmmss(state.get("good_at")))}</span> <span>older than this result'
                     '</span></div><div class="shep-dx-state"><span class="shep-tag shep-tag-grey">'
                     f'last known · not current</span> {retry}</div>')
    status = state.get("status") or {}
    facts = state.get("facts") or {}
    kind = served_kind(status)
    lines: List[str] = []
    if kind == "no_model":
        lines.append('<p class="shep-dx-name">No model loaded</p>'
                     '<p class="shep-dx-line">Nothing is served on this server: diagnoses are '
                     'refused until a model is loaded.</p>')
    elif kind == "model":
        item = facts.get("item")
        sha = served_checkpoint_sha(status)
        task = served_task(status)
        why = ""
        if item:
            run = r.e(r.model_name_line(item))
        else:
            meta = status.get("checkpoint_meta") or {}
            epoch = meta.get("epoch") if isinstance(meta, Mapping) else None
            run = f"Checkpoint · epoch {r.e(epoch if epoch is not None else '—')}"
            why = r.e(facts.get("item_problem") or "")
        if task is not None:
            # The task, named once on the card (K6 decision 4); the run under it.
            name = (f'<p class="shep-dx-name">{r.e(task_model_name(task))}</p>'
                    f'<p class="shep-dx-line shep-dx-sub">{run}{f" · {why}" if why else ""}</p>')
        elif item:
            name = f'<p class="shep-dx-name">{run}</p>'
        else:
            name = f'<p class="shep-dx-name">{run}<span class="shep-dx-sub"> · {why}</span></p>'
        mark = "" if grey else f' <span class="shep-ok-t">{_CHECK} verified</span>'
        lines.append(f'{name}'
                     f'<p class="shep-dx-line">{r.fingerprint_html("model fingerprint", sha, copy=False)}'
                     f'{mark}</p>'
                     f'<p class="shep-dx-line">{_trained_html(facts)}</p>')
    else:
        lines.append('<p class="shep-dx-line shep-muted">The status did not say what is served.</p>')
    chips = _chips(status, grey=grey)
    tail = []
    if chips:
        tail.append('<span class="shep-dx-chips">' + "".join(chips) + "</span>")
    if kind == "model":
        tail.append(_recorded_html(facts, grey=grey))
    if tail:
        lines.append('<p class="shep-dx-line shep-dx-tail">' + " ".join(tail) + "</p>")
    classes = "shep-dx-card-bd shep-dx-card-grey" if grey else "shep-dx-card-bd"
    return f'<div class="{classes}">' + "".join(parts + lines) + "</div>"


def change_button(state: Mapping[str, Any]) -> Dict[str, Any]:
    """The card's action: "🔒 Load a model…" (primary) when no model is
    loaded, else "🔒 Change model…"; off for a request D8 refuses."""
    no_model = served_kind(state.get("status")) == "no_model" and bool(state.get("ok"))
    return {"value": "🔒 Load a model…" if no_model else "🔒 Change model…",
            "variant": "primary" if no_model else "secondary",
            "interactive": bool(state.get("operator"))}


def change_reason_html(state: Mapping[str, Any]) -> str:
    """Why Change model… is off, as visible text (D8's short words)."""
    if state.get("operator") or state.get("ok") is None:
        return ""
    short = state.get("d8_short") or D8_SHORT["unreadable_request"]
    # Two word groups, so a line breaks between them and never leaves "to M1"
    # alone on the second line (M5-3).
    return f'<p class="shep-dx-why">{r.word_groups("🔒 Operator controls only:", short)}</p>'


def details_visible(state: Mapping[str, Any]) -> bool:
    """Details › is offered whenever something is served (or was last known)."""
    return served_kind(state.get("status")) in ("model", "unknown") \
        and state.get("status") is not None


# ---------------------------------------------------------------------------
# Details: the identity drawer's "Model in use"
# ---------------------------------------------------------------------------
_ID_ANCHOR = "</div></div></details>"


def _insert_after_section(html_: str, section: str, extra: str, closing: str) -> str:
    """Put `extra` at the end of one Model Facts section's body (before its
    closing markup), found by its ``data-sec``. Model Facts has no slot for
    what only a served model has (`renderers.model_facts_html`); the smallest
    extension is here, and the tests pin the markup it relies on."""
    start = html_.find(f'data-sec="{section}"')
    if start < 0:
        return html_
    end = html_.find(closing, start)
    if end < 0:
        return html_
    return html_[:end] + extra + html_[end:]


#: Model Facts' words for a dataset fingerprint the checkpoint does not record.
_NOT_RECORDED = '<span class="shep-muted">not recorded</span>'


def _replace_in_section(html_: str, section: str, old: str, new: str) -> str:
    """Replace every `old` inside one Model Facts disclosure (found by its
    ``data-sec``, up to its ``</details>``), and nowhere else."""
    start = html_.find(f'data-sec="{section}"')
    if start < 0:
        return html_
    end = html_.find("</details>", start)
    end = len(html_) if end < 0 else end
    return html_[:start] + html_[start:end].replace(old, new) + html_[end:]


def _kv_rows(rows: List[Tuple[str, str]], *, wide: bool = False) -> str:
    """Model Facts' label/value cells (`renderers.kv_cells`); `wide`, for the
    digest rows: each label on its own line over its full value."""
    return r.kv_cells(rows, wide=[k for k, _v in rows] if wide else ())


def _full(digest: Any, what: str) -> str:
    """A full digest with an icon-only Copy, so it keeps one line in the
    identity drawer (M5-3: Model Facts fits at 1920 x 1080)."""
    if not r.is_digest(digest):
        return '<span class="shep-muted">none</span>'
    return f'<code class="shep-full">{r.e(digest)}</code> ' + copy_button(digest, what,
                                                                         icon_only=True)


def served_warnings(status: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """The served state's warnings in the banners' words: synthetic, no
    shortest-path table."""
    identity = status.get("served_identity") or {}
    found = []
    if _part(identity, "dataset_record", "origin") == "synthetic":
        found.append(r.IDENTITY_BANNERS["synthetic"])
    if served_kind(status) == "model" and _part(identity, "scoring_mode") == "gnn_only":
        found.append(r.IDENTITY_BANNERS["no_sp"])
    return found


#: The structural comparator's own words (`src.utils.fingerprint.verify_fingerprint`):
#: its headline before the differences, and its answer for a checkpoint that
#: records no structural fingerprint (nothing was compared). The tests run the
#: comparator itself, so a change of its words is seen.
_STRUCTURE_HEADLINE = "Graph structure differs from the checkpoint's structural fingerprint"
_STRUCTURE_LEGACY = "Checkpoint does not contain a data fingerprint"


def _structure_line(status: Mapping[str, Any]) -> str:
    """The structural fingerprint line: no warnings; not compared (the
    checkpoint records no structural fingerprint); or what differs, counted
    without the comparator's own headline."""
    warnings = [str(w) for w in status.get("fingerprint_warnings") or []]
    gnn = "GNN loaded" if status.get("gnn_ready") else "GNN not loaded"
    if not warnings:
        return f'<p class="shep-muted">Graph structural fingerprint: no warnings · {r.e(gnn)}</p>'
    if len(warnings) == 1 and warnings[0].startswith(_STRUCTURE_LEGACY):
        return (f'<p class="shep-warn"><b>⚠ Graph structure not compared</b> (the checkpoint records '
                f'no structural fingerprint: legacy format) · {r.e(gnn)}</p>')
    differences = [w for w in warnings if not w.startswith(_STRUCTURE_HEADLINE)]
    items = "".join(f"<li>{r.e(w)}</li>" for w in differences[:3])
    more = (f'<li class="shep-muted">and {len(differences) - 3} more (the server log lists them)'
            '</li>' if len(differences) > 3 else "")
    return (f'<p class="shep-warn"><b>⚠ Graph structure differs from the checkpoint\'s structural '
            f'fingerprint</b> ({len(differences)}; node types and counts, edge-type count and '
            f'feature sizes compared, not the graph\'s identity) · {r.e(gnn)}</p>'
            f'<ul class="shep-dx-list">{items}{more}</ul>')


def _serving_note(status: Mapping[str, Any], facts: Mapping[str, Any],
                  trained: Optional[str]) -> str:
    identity = status.get("served_identity") or {}
    serving = facts.get("serving")
    manifest = _part(identity, "graph", "split_manifest_sha256")
    parts = []
    if serving:
        same = ("the same dataset ✓" if trained and trained == manifest
                else "another dataset" if trained else "dataset not compared")
        parts.append(f"serving reads workspace {serving}: {same}")
    graph_state = _part(identity, "graph", "state")
    parts.append("graph ✓" if graph_state == "verified" else f"graph {graph_state or 'not reported'}")
    sp = _part(identity, "shortest_paths", "state")
    parts.append(f"SP table ✓ build {_part(identity, 'shortest_paths', 'build_id')}"
                 if sp == "verified" else "SP from the graph view" if sp == "computed"
                 else "SP not used by this task" if served_task(status) is not None
                 else "no SP table")
    origin = _part(identity, "dataset_record", "origin")
    if _part(identity, "dataset_record", "state") == "verified":
        parts.append(f"dataset record ✓ {origin}")
    return " · ".join(parts)


def served_facts_html(state: Mapping[str, Any], *, open_section: str = "identity") -> str:
    """The identity drawer's "Model in use": Model Facts of the served model
    (the last known, said so when the card is not current)."""
    status = state.get("status")
    if status is None:
        why = state.get("problem") or "it has not been read yet"
        return f'<div class="shep-reason shep-reason-grey">The model in use could not be read: {r.e(why)}.</div>'
    facts = state.get("facts") or {}
    kind = served_kind(status)
    identity = status.get("served_identity") or {}
    note = ""
    if not state.get("ok"):
        note = (f'<div class="shep-reason shep-reason-grey">Last known at '
                f'{r.e(hhmmss(state.get("good_at")))}, not current: {r.e(state.get("problem"))}.'
                '</div>')
    if kind == "no_model":
        return note + ('<div class="shep-reason">No model loaded: nothing is served, and '
                       'diagnoses are refused until a model is loaded.</div>')
    if kind == "unknown":
        return note + ('<div class="shep-reason shep-reason-grey">The status did not say what is '
                       'served.</div>')
    served_rows = [
        ("Knowledge graph served (kg.json)", _full(_part(identity, "graph", "kg_sha256"),
                                                   "the full knowledge graph digest")),
        ("Dataset record served (kg.provenance.json) · origin "
         f"{_part(identity, 'dataset_record', 'origin') or 'none'}",
         _full(_part(identity, "dataset_record", "sha256"), "the full dataset record digest")),
    ]
    warnings = served_warnings(status)
    item = dict(facts.get("item") or {})
    if not item:
        sha = served_checkpoint_sha(status)
        meta = status.get("checkpoint_meta") if isinstance(status.get("checkpoint_meta"), dict) else {}
        item = {"architecture": None, "run": None, "filename": None, "sha256": sha,
                "epoch": meta.get("epoch"), "validation": _validation(meta),
                "dataset_fingerprint": None, "problem": None}
    trained = facts.get("trained") or {}
    shown = trained.get("state") == "read"
    if shown:
        item["dataset_fingerprint"] = trained.get("fingerprint")
    serving_row = next((w for w in facts.get("workspaces") or []
                        if w.get("name") == facts.get("serving")), None)
    model_facts = r.facts_from_checkpoint(
        item, facts.get("workspaces") or [],
        recorded=r.RecordedTests(**facts["recorded"]) if isinstance(facts.get("recorded"), Mapping)
        else None,
        serving_workspace=serving_row)
    item_warnings = [w for w in model_facts.warnings
                     if w[0] != r.IDENTITY_BANNERS["synthetic"][0]
                     and (shown or not w[0].startswith("DATASET NOT RECORDED"))]
    meta = status.get("checkpoint_meta") if isinstance(status.get("checkpoint_meta"), dict) else {}
    params = meta.get("params")
    name_line = model_facts.name_line if facts.get("item") else \
        f"Checkpoint · epoch {item.get('epoch') if item.get('epoch') is not None else '—'}"
    if isinstance(params, int) and not isinstance(params, bool):
        name_line += f" · {params:,} params"
    model_facts = model_facts._replace(
        name_line=name_line, warnings=(*warnings, *item_warnings),
        serving_note=_serving_note(status, facts,
                                   trained.get("fingerprint") if shown else None))
    html_ = r.model_facts_html(model_facts, open_section=open_section)
    if not shown:
        # Model Facts reads an unknown trained-on dataset as "not recorded"
        # (in Identity, and twice in Data); here it was not read (D8, or not
        # yet), which is said instead, everywhere in both sections.
        words = TRAINED_NOT_SHOWN if trained.get("state") == "not_shown" else \
            (TRAINED_NOT_READ_YET if trained.get("state") == "pending"
             else f"not read · {trained.get('problem') or 'not read'}")
        short = _FINGERPRINT_NOT_READ.get(trained.get("state"), "not read")
        html_ = _replace_in_section(html_, "identity", _NOT_RECORDED,
                                    f'<span class="shep-muted">{r.e(words)}</span>')
        html_ = _replace_in_section(html_, "data", "Trained on not recorded",
                                    f"Trained on: {r.e(words)}")
        html_ = _replace_in_section(html_, "data", _NOT_RECORDED,
                                    f'<span class="shep-muted">{r.e(short)}</span>')
    html_ = _insert_after_section(html_, "identity", _kv_rows(served_rows, wide=True),
                                  _ID_ANCHOR)
    html_ = _insert_after_section(
        html_, "evidence",
        ' · <button type="button" class="shep-dx-link" data-key="models">Full records: '
        'Research › Models ›</button>', "</p></div></details>")
    html_ = _insert_after_section(html_, "warnings", _structure_line(status),
                                  "</div></section>")
    return note + html_


__all__ = [
    "CARD_HELP",
    "SERVER_PATH",
    "STATUS_TIMEOUT_SECONDS",
    "TRAINED_NOT_READ_YET",
    "TRAINED_NOT_SHOWN",
    "advance",
    "body_html",
    "card_mode",
    "card_problem",
    "change_button",
    "change_reason_html",
    "details_visible",
    "dismiss_change",
    "hhmmss",
    "parse_local",
    "new_state",
    "problem_sentence",
    "read_status_within",
    "recorded_for",
    "redacted",
    "reference_cohorts",
    "run_blocked",
    "served_facts",
    "served_facts_html",
    "served_item",
    "served_kind",
    "served_task",
    "served_warnings",
    "stamp",
    "task_model_name",
    "time_html",
    "trained_on_listed",
]
