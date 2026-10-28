"""
Research › Prepare › Training (M5-2, UI-C)
==========================================
`docs/working/PLAN_TEST_RESULTS.md` §6, "M5's boundaries" (Training: the run
bar, the parameters for the next run, Continue, the run view and its charts).
Built by `create_training_tab` under the page contract (`src/webui/shell.py`).

Module: src/webui/components/training_console.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/webui/components/training_console.py

**The run bar** (full width, never scrolls): the Dataset select over the
server's workspace list (`workspaces.read_workspaces`, the shared option label
`renderers.workspace_choices`); choosing one reads its manifest and record and
the dataset chip shows them (`renderers.dataset_identity_html`) with a
"Details ›" popover (the full fingerprint with Copy, what the manifest binds,
the split, the sources, the path relative to the workspace root, "All
datasets ›"). While a training job runs the select is disabled and shows that
job's dataset ("in use by the running job"). Then the status pill (idle,
starting, running with epoch, batch and time, stopping, completed, stopped,
failed in red; never green for a failed or stopped run), the run name, one
reason line saying why an action is disabled (D8's short words for a read-only
request; the GPU job holding the slot, from the job runner and the GPU lock's
holder record, with "Open Test ›" for a test run), and 🔒 Start, 🔒 Continue…
and 🔒 Stop. Stop asks first, in the code's words ("Stop training" / "Keep
training"). A failed run's error banner spans the page: under its headline the
last error line the run printed, beside it where the run's log is (only when it
has one), then its output tail.

**Start sends the full dataset fingerprint it showed** (the session's
``gr.State``, never the client's); the job runner reads the manifest again and
refuses a different one before anything is written or started
(`TrainingManager.start_training`'s ``expected_dataset_fingerprint``, the one
start path the API route uses too).

**Parameters "for the next base run"** (K6 decision 12): exactly the
reference recipe's base settings (`src.config.recipe.BASE`), each field
starting at the recipe's value (Epochs, Learning rate, Batch size and Seed
shown; the rest under "More reference settings"), their help behind the
shared ⓘ (`FIELD_HELP`: what it is, the reference value and its source), and
one visible line saying which differ from the reference settings (decision
5). Start sends ``learning_stage="base"`` and only the settings that differ;
each value is checked by its field spec first, a problem named by its label.
A base's task models are trained on Research › Fine-tune
(`src/webui/components/finetune_page.py`).

**Continue…** opens a drawer over the parameters column: the parent list is
the run bar's workspace's model records only (bases and task models; a file
without a model spec is never continued from the dashboard, decision 1),
from `server_lists.list_checkpoints` (an operator control since it opens
every checkpoint), or a typed path that locates one of them; Check checkpoint
shows what continuing does (the shared run plan) and compares the parent's
recorded dataset fingerprint with the workspace's current one. While it is
open the run bar's actions are Start (disabled, with the reason), 🔒 Continue
Training (disabled while the datasets differ, with the reason) and Cancel. A
listed parent is resolved by the server from its list again (never a client
path) and its bytes compared with the digest listed; the runner and the
training process still decide at start.

**The run view**: the run the job runner started last (a Fine-tune child
included), or, before any since the server started, the latest run of the run
bar's workspace; its header says which dataset fingerprint the run used (from
its checkpoint: operators only; a read-only request sees "not shown"), the
snapshot sentence (operators: where it can be continued from), the shared
chart component (`src/webui/components/charts.py`: Loss, the Trainer's
validation figure and its companions, chosen by the keys the run's log holds
(a base's LP accuracy, ROC-AUC and AP; a task model's pooled MRR and pooled
top-k; a file without a model spec's MRR and Hits@K), and the learning rate on
a log axis, one 2×2 figure sharing the epoch axis; the data from the run's
metrics log through the component's server function) and the resources line.

**Per browser session, no module state:** the selection, the drawer, the last
values sent and the banners a session dismissed are ``gr.State``. The page is
refreshed when it is shown (``ctx.on_page_open``) and by the shell's timers
(outside ``gr.Tabs``: ``ctx.timer_fast`` for the run bar and the run view,
``ctx.timer_slow`` for the resources line), both ``queue=False``, doing no work
while another page is shown (``ctx.location``).
**The run bar and the run view are one projection of the session's
selection, written only by a queued render** (`render`, the pieces of
`VIEW_KEYS`: the run bar's workspace, ``ws`` and ``ctx.training_workspace``;
the select's value, the chip, the run shown and its chart token, Start,
Continue…, Stop and their reason, the drawer's line): every listener that
writes them is in one queued group of one (``ctx.training_workspace_writes``),
first asked first run, each from what the one before it wrote: the page open,
then its render; the select; Datasets' "Use for training ›" and the change of
``ctx.training_workspace`` it makes; the render after every action, and the
render the fast poll hands a change over to. So a workspace chosen while a
render waits is applied after it, and no piece drawn from an older selection
is applied after one drawn from a newer. **The fast poll writes only the
status pill** (`POLL_KEYS`, the job runner's state alone, on every tick, as
the renders do), the header's chips and a hand-off: it renders the page as it
is now and compares it with
what the session was last sent; when a piece differs (new epochs, a job that
started or ended, the GPU slot, a running job's workspace to follow), it gives
a hidden component a new value (`CHANGED_ID`), whose ``.change`` runs the
queued render, which reads the session's selection and the job again at its
own turn and sends what changed (everything again every `FULL_SEND_TICKS`
hand-offs). It writes no cache: what the session was sent, and the reads a
render keeps, have the renders as their one writer (the poll keeps its own
record, its reads throttled by time, `POLL_READS`, and the GPU slot it saw,
in a State only it writes).

**D8:** Start, Continue…, Continue Training, Check checkpoint, the parent list
and Stop (its confirmation reads the run's checkpoint) call the Gradio helper
first, on their own request, then the shared service in process; each is
followed by the chips' read. A read-only request sees every action disabled
with D8's reason as text, and no checkpoint path, snapshot or recorded
fingerprint. The metrics log and the workspaces' JSON records are not
operator-only. Stop's question also names the Fine-tune children queued after
the running one (`stop_question`); the Fine-tune page's Stop runs the same
stop logic (`ask_stop`, `confirm_stop`) behind its own D8-checked handlers.

Architecture Note:
    The training service is the job runner (`src.api.services.training_manager`,
    allowed: webui -> api); nothing here imports `src.training`.

Version: 2.0.0
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import gradio as gr

from src.api.services.training_manager import training_manager
from src.config.recipe import BASE, BASE_SOURCES
from src.config.runtime_presets import load_runtime_settings
from src.config.training_fields import by_name, value_problem
from src.evaluation.caveats import TASK_NAMES
from src.utils.gpu_lock import lock_state
from src.webui.components import charts
from src.webui.components import renderers as r
from src.webui.components.chips import D8_SHORT, chips_on_change, gpu_chip_from
from src.webui.components.help import copy_button, help_button
from src.webui.components.workspaces import (
    manifest_facts,
    read_workspaces,
    redact,
    workspaces_root,
)
from src.webui.operator_access import operator_access

logger = logging.getLogger(__name__)

#: This page's id in the shell (`src.webui.shell.TRAINING`, which a test
#: checks it equals). Not imported from the shell at the top:
#: ``src/webui/components/__init__.py`` imports this module while the shell
#: is still importing its own components, so the shell is imported where it
#: is used (`_shell`).
TRAINING = "training"


def _shell():
    from src.webui import shell

    return shell


def plain_html(value: Any = "", **kwargs: Any) -> Any:
    """The shell's ``plain_html`` (no prose CSS, no default click script)."""
    return _shell().plain_html(value, **kwargs)


TRAINING_CONFIG_FILE = Path(".shepherd_training_config.json")
DISPLAY_PREFIX = "SHEPHERD-Advanced/"

#: Every this many renders the fast poll hands over, the run bar and the run
#: view are sent whole, whatever the session's record of what it was last sent
#: says (each other one sends what changed; the render after an action, whole).
FULL_SEND_TICKS = 8
#: The GPU lock is read at most this often per session (seconds).
LOCK_READ_SECONDS = 5.0
#: The run bar's workspace's runs are listed again at most this often per
#: session (seconds), for the run view's "latest" run.
LATEST_READ_SECONDS = 10.0
#: Lines of a failed run's output the banner shows.
BANNER_TAIL_LINES = 12
#: The dataset chip's "Details ›" popover is this wide (px, at most the window
#: less 48): "What the manifest binds" lays four columns of 12-hex digests and
#: their labels in it (M5-3: at 460 the second column's digests ended 2 px
#: past the card's inner edge).
DETAILS_POPOVER_WIDTH = 480
#: The dataset chip's stacking level: above the parameter fields' labels
#: (Gradio's 40), which its Details popover opens over in a narrow window
#: (M5-3: at 1280 x 720 "Learning Rate" and "Batch Size" were drawn on it).
CHIP_Z_INDEX = 60


def _save_training_config(data_dir: str, request: gr.Request) -> str:
    """Save training path config: an operator control (D8), checked on this
    event's own request before anything is written.

    **Unused since M5-2**: its only caller, the Data Paths section's Save Path,
    is gone (the workspace is the run bar's select). Kept, gated, for the
    owner's question "Save as the default dataset", which is not built."""
    access = operator_access(request)
    if not access.allowed:
        return access.refusal
    stripped = data_dir.strip()
    if stripped.startswith(DISPLAY_PREFIX):
        stripped = stripped[len(DISPLAY_PREFIX):]
    try:
        with open(TRAINING_CONFIG_FILE, "w") as f:
            json.dump({"data_dir": stripped}, f, indent=2)
        return "Training path config saved."
    except Exception as e:
        return f"Failed to save: {e}"


# =============================================================================
# The recipe's settings in a form (K6 decisions 5, 11, 12)
# =============================================================================
#: **The label of each recipe setting a form offers**, one name per setting:
#: the base form (`BASE_FIELDS`) and the Fine-tune page's Epochs and Seed.
SETTING_LABELS: Dict[str, str] = {
    "num_epochs": "Epochs",
    "learning_rate": "Learning rate",
    "batch_size": "Batch size",
    "seed": "Seed",
    "num_neighbors": "Neighbours per hop",
    "max_grad_norm": "Gradient clip",
    "scheduler_type": "Scheduler",
    "warmup_steps": "Warmup steps",
    "weight_decay": "Weight decay",
    "use_amp": "Mixed precision",
    "early_stopping_patience": "Early stopping patience",
}
#: The base form's fields, in its order: **exactly the reference recipe's base
#: settings** (`recipe.BASE`), the first `BASE_SHOWN` of them shown, the rest
#: under "More reference settings".
BASE_FIELDS: Tuple[str, ...] = ("num_epochs", "learning_rate", "batch_size", "seed",
                                "num_neighbors", "max_grad_norm", "scheduler_type",
                                "warmup_steps", "weight_decay", "use_amp",
                                "early_stopping_patience")
BASE_SHOWN = 4

#: Each setting's help lead (what it is, under 90 characters), before its
#: reference value and source (`setting_help`).
_SETTING_LEADS = {
    "num_epochs": "How many passes over the training data the run makes.",
    "learning_rate": "The optimizer's step size; with the scheduler none it stays constant.",
    "batch_size": "Training examples per step: larger is faster and needs more memory.",
    "seed": "The random seed of the weights' start and the sampling (the split has its own).",
    "num_neighbors": "Neighbours sampled at each hop, one number per encoder layer; -1 takes all.",
    "max_grad_norm": "The largest gradient norm a step takes; a larger one is scaled down.",
    "scheduler_type": "How the learning rate changes over the run; none keeps it constant.",
    "warmup_steps": "Steps over which the learning rate ramps up from zero; 0 is none.",
    "weight_decay": "The optimizer's penalty on large weights; 0 is none.",
    "use_amp": "16-bit arithmetic on CUDA to save memory and time; off keeps float32.",
    "early_stopping_patience": "Epochs without a better Trainer validation figure before the "
                               "run stops; empty is off.",
}


def setting_shown(value: Any) -> str:
    """A setting's value as a form and its notes say it: a list as its
    comma-separated items, on/off for a switch, "off" for no patience."""
    if value is None:
        return "off"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def setting_help(name: str, references: Mapping[str, Any], sources: Mapping[str, str]
                 ) -> Tuple[str, str]:
    """A setting's help, ``(title, text)``: its label, then what it is, its
    reference value and the source of that value (`src.config.recipe`)."""
    return (SETTING_LABELS[name],
            f"{_SETTING_LEADS[name]}\n\n"
            f"Reference: {setting_shown(references[name])}.\n"
            f"Source: {sources[name]}.\n\n"
            "A provisional default from the source, not tuned on this data.")


def form_value(name: str, value: Any) -> Any:
    """A recipe value as its field holds it (the hop list as text)."""
    return setting_shown(value) if name == "num_neighbors" else value


def setting_field(name: str, value: Any, *, elem_id: str) -> Any:
    """**The one field of a recipe setting** (the base form's and the
    Fine-tune page's), by its field spec: the hop list as text, a switch, the
    spec's few choices as a radio, else a number box (whole numbers for an
    integer setting, which `value_problem` refuses as a float). Labelled by
    `SETTING_LABELS`, its help behind the ⓘ (never ``info=``)."""
    label, spec = SETTING_LABELS[name], by_name(name)
    if name == "num_neighbors":
        return gr.Textbox(label=label, value=form_value(name, value), max_lines=1,
                          elem_id=elem_id)
    if spec.kind == "bool":
        return gr.Checkbox(label=label, value=bool(value), elem_id=elem_id)
    if spec.choices:
        return gr.Radio(label=label, choices=list(spec.choices), value=value, elem_id=elem_id)
    return gr.Number(label=label, value=value, precision=0 if spec.kind == "int" else None,
                     elem_id=elem_id)


def base_form_defaults() -> Tuple[Any, ...]:
    """The base form's starting values, in `BASE_FIELDS` order: the recipe's."""
    return tuple(form_value(name, BASE[name]) for name in BASE_FIELDS)


def _parsed(name: str, value: Any) -> Tuple[Any, Optional[str]]:
    """A field's value as its setting takes it: the hop text as a list of
    whole numbers; a whole float of an integer field (a number box's value) as
    an int; an int of a float field as a float. ``(value, why not)``."""
    if name == "num_neighbors":
        try:
            return [int(part) for part in str(value or "").split(",") if part.strip()], None
        except ValueError:
            return None, "must be whole numbers separated by commas, such as 15, 10, 5."
    kind = by_name(name).kind
    if kind == "int" and isinstance(value, float) and value.is_integer():
        return int(value), None
    if kind == "float" and isinstance(value, int) and not isinstance(value, bool):
        return float(value), None
    return value, None


def setting_problem(name: str, value: Any, reference: Any) -> Tuple[Any, Optional[str]]:
    """A field's value parsed and checked by its field spec (`value_problem`,
    the one check the start makes again): ``(value, problem)``, the problem
    named by the field's label (`SETTING_LABELS`). Empty is taken only where the
    reference value is (early stopping's patience)."""
    label = SETTING_LABELS[name]
    parsed, why = _parsed(name, value)
    if why:
        return None, f"{label} {why}"
    problem = value_problem(name, parsed, allow_none=reference is None)
    if problem and problem.startswith(name):
        problem = label + problem[len(name):]
    elif problem:
        problem = f"{label}: {problem}"
    return parsed, problem


def _same(value: Any, reference: Any) -> bool:
    return value == reference and isinstance(value, bool) == isinstance(reference, bool)


def recipe_differences(values: Mapping[str, Any], references: Mapping[str, Any]
                       ) -> Tuple[Dict[str, Any], List[str]]:
    """**What a start sends of a form** (K6 decisions 11, 12): ``(settings,
    problems)``, the settings whose value differs from `references` (the
    recipe's), each parsed and checked (`setting_problem`), and every problem,
    named by its label. A setting left at its reference value is not sent, so
    the run records it as the recipe's, never as the operator's."""
    settings: Dict[str, Any] = {}
    problems: List[str] = []
    for name, value in values.items():
        parsed, problem = setting_problem(name, value, references[name])
        if problem:
            problems.append(problem)
        elif not _same(parsed, references[name]):
            settings[name] = parsed
    return settings, problems


def differs_html(values: Mapping[str, Any], references: Mapping[str, Any]) -> str:
    """The form's one line on the reference settings (decision 5): "Differs
    from the reference settings: Epochs 5 (200)", each setting that differs
    with its reference value in brackets, or "Uses the reference settings.";
    then any value the start would refuse, by its label. Escaped."""
    settings, problems = recipe_differences(values, references)
    if settings:
        text = "Differs from the reference settings: " + ", ".join(
            f"{SETTING_LABELS[name]} {setting_shown(value)} ({setting_shown(references[name])})"
            for name, value in settings.items())
    else:
        text = "Uses the reference settings."
    refused = (f'<p class="shep-amber-t">Not valid: {r.e(" ".join(problems))}</p>'
               if problems else "")
    return f'<div class="shep-differs"><p>{r.e(text)}</p>{refused}</div>'


def _on_base_changed(*values: Any) -> str:
    """Any base field changed: the line on the reference settings again."""
    return differs_html(dict(zip(BASE_FIELDS, values)), BASE)


# =============================================================================
# The run bar's workspace (the session's selection)
# =============================================================================
def empty_workspace() -> Dict[str, Any]:
    return {"name": None, "row": None, "facts": None, "choices": [], "listed": [],
            "digits": r.SHORT_HEX, "problem": None, "latest_run": None}


def _latest_run(directory: Optional[Path]) -> Optional[str]:
    """The newest run directory of a workspace, as "<arch>/<run>", from the
    run layout's one listing (most recently trained first); None for none."""
    if directory is None:
        return None
    from src.utils.checkpoint_paths import list_runs, run_architecture

    try:
        runs = list_runs(directory)
    except OSError:
        return None
    return f"{run_architecture(runs[0])}/{runs[0].name}" if runs else None


def workspace_state(name: Optional[str], listed: Optional[Mapping[str, Any]] = None
                     ) -> Dict[str, Any]:
    """The session's run-bar workspace, read now (or from `listed`, a
    `read_workspaces` answer just read): the server's list (the dashboard's one
    reader, every reason without an absolute path) with the options' labels,
    the selected item, what its manifest binds, the listed fingerprints (to say
    what a parent trained on) and its latest run. The full fingerprint kept
    here is the one Start sends."""
    listed = listed if listed is not None else read_workspaces()
    rows = listed["rows"]
    state = empty_workspace()
    # The options name the workspace; its identity (fingerprint, synthetic) is
    # the chip's, beside the select (said once on the page). A workspace that
    # cannot be trained keeps its reason in its option, read before choosing.
    trainable = {x["name"] for x in rows if r.is_digest(x.get("dataset_fingerprint"))}
    state["choices"] = [[name if name in trainable else label, name]
                        for label, name in r.workspace_choices(rows)]
    state["listed"] = [{"name": x["name"], "dataset_fingerprint": x.get("dataset_fingerprint"),
                        "dataset_record": x.get("dataset_record")} for x in rows]
    state["digits"] = r.distinguishing_digits(x.get("dataset_fingerprint") for x in rows)
    state["problem"] = listed["problem"]
    row = next((x for x in rows if x["name"] == name), None)
    if row is not None:
        state["name"], state["row"] = row["name"], row
        directory = listed["directories"].get(row["name"])
        if row.get("manifest") == "read" and directory is not None:
            state["facts"], _problem = manifest_facts(directory, listed["root"])
        state["latest_run"] = _latest_run(directory)
    return state


def workspace_fingerprint(ws: Optional[Mapping[str, Any]]) -> Optional[str]:
    row = (ws or {}).get("row") or {}
    value = row.get("dataset_fingerprint")
    return value if r.is_digest(value) else None


def dataset_choices_update(ws: Mapping[str, Any], *, interactive: bool = True) -> Any:
    choices = [tuple(c) for c in ws.get("choices") or []]
    return gr.update(choices=choices, value=ws.get("name"), interactive=interactive)


# --------------------------------------------------------------------------- the chip
LOCK_ICON = ('<svg class="shep-lock" width="12" height="12" viewBox="0 0 16 16" aria-hidden="true" '
         'focusable="false"><rect x="3" y="7" width="10" height="7.5" rx="1.6" '
         'fill="currentColor"/><path d="M5.2 7V5.2a2.8 2.8 0 0 1 5.6 0V7" fill="none" '
         'stroke="currentColor" stroke-width="1.6"/></svg>')
NO_ICON = ('<svg class="shep-no" width="14" height="14" viewBox="0 0 16 16" aria-hidden="true" '
       'focusable="false"><circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" '
       'stroke-width="1.5"/><path d="M4.6 8h6.8" stroke="currentColor" stroke-width="1.5"/></svg>')


def details_html(ws: Mapping[str, Any]) -> str:
    """The chip's "Details ›" popover: the workspace's name and record, the
    shared details body (`renderers.dataset_details_html`, as Research ›
    Datasets shows it), and "All datasets ›"."""
    row = ws.get("row") or {}
    name = str(row.get("name") or "")
    head = (f'<div class="shep-pop-hd"><b>{r.e(name)}</b>{r.record_chip(row)}'
            '<button type="button" class="shep-pop-x" data-act="close" '
            'aria-label="Close the dataset details">×</button></div>')
    return (head + r.dataset_details_html(row, ws.get("facts"))
            + '<button type="button" class="shep-link" data-act="datasets">All datasets ›</button>')


def chip_html(ws: Optional[Mapping[str, Any]], *, in_use: bool = False) -> str:
    """The dataset chip beside the select: "in use by the running job" while a
    training job runs, the shared dataset identity (built or sources, the
    dataset fingerprint in bold monospace, synthetic, record verified, not
    trainable), and "Details ›" with its popover."""
    ws = ws or {}
    row = ws.get("row")
    lead = '<span class="shep-rb-inuse">in use by the running job</span>' if in_use else ""
    if not row:
        why = ws.get("problem") or ("No dataset selected: choose one from the server's list."
                                    if ws.get("choices") else "The server lists no workspace.")
        return f'<div class="shep-rb-chip">{lead}<span class="shep-muted">{r.e(why)}</span></div>'
    identity = r.dataset_identity_html(row, name=False, digits=ws.get("digits") or r.SHORT_HEX)
    return (f'<div class="shep-rb-chip">{lead}{identity}'
            '<button type="button" class="shep-link shep-rb-details" data-act="details" '
            'aria-expanded="false" aria-controls="shep-train-details">Details ›</button>'
            '<div class="shep-pop" id="shep-train-details" role="dialog" '
            f'aria-label="Dataset details" hidden>{details_html(ws)}</div></div>')


#: A "Details ›" popover's behaviour, element-scoped (the element is never
#: re-mounted, and a replaced value keeps the listeners, being delegated): the
#: dataset chip's, with its "All datasets ›" cross-link, and the run view
#: header's. Escape, a click outside or focus leaving it closes it.
_POPOVER_JS = r"""
function pop() { return element.querySelector('.shep-pop'); }
function toggle(open) {
  var p = pop(), b = element.querySelector('[data-act="details"]');
  if (!p || !b) { return; }
  p.hidden = !open;
  b.setAttribute('aria-expanded', open ? 'true' : 'false');
  if (open) { var c = p.querySelector('.shep-copy, .shep-pop-x'); if (c) { c.focus(); } }
}
element.addEventListener('click', function (e) {
  var a = e.target.closest ? e.target.closest('[data-act]') : null;
  if (!a) { return; }
  var act = a.getAttribute('data-act');
  if (act === 'details') { var p = pop(); toggle(p && p.hidden); }
  else if (act === 'close') { toggle(false); var b = element.querySelector('[data-act="details"]'); if (b) { b.focus(); } }
  else if (act === 'datasets') { toggle(false); trigger('click', {act: 'datasets'}); }
});
element.addEventListener('keydown', function (e) {
  if (e.key === 'Escape' && pop() && !pop().hidden) { e.stopPropagation(); toggle(false);
    var b = element.querySelector('[data-act="details"]'); if (b) { b.focus(); } }
});
element.addEventListener('focusout', function (e) {
  var p = pop();
  if (p && !p.hidden && e.relatedTarget && !element.contains(e.relatedTarget)) { toggle(false); }
});
element.addEventListener('pointerdown', function (e) { e.stopPropagation(); });
document.addEventListener('pointerdown', function (e) {
  var p = pop();
  if (p && !p.hidden && element.isConnected && !element.contains(e.target)) { toggle(false); }
}, {passive: true});
"""

#: Cross-links in the reason line ("Open Test ›") and the banner's Dismiss.
TRIGGER_JS = r"""
element.addEventListener('click', function (e) {
  var a = e.target.closest ? e.target.closest('[data-act]') : null;
  if (a) { trigger('click', {act: a.getAttribute('data-act'), key: a.getAttribute('data-key') || ''}); }
});
"""

#: A wide window at least this tall shows a failed run's output tail open;
#: a shorter one shows its summary line, so the charts keep their room.
BANNER_OPEN_HEIGHT = 1000

#: The banner: Dismiss (`TRIGGER_JS`) and its output tail's disclosure, sent
#: open and closed here once per banner on a wide window under
#: `BANNER_OPEN_HEIGHT` px; once the user opens or closes it, their choice
#: holds for the banners that follow on this page.
_BANNER_JS = TRIGGER_JS + r"""
function fitTail() {
  var d = element.querySelector('details.shep-banner-more:not([data-fit])');
  if (!d) { return; }
  d.setAttribute('data-fit', '');
  var chosen = element.getAttribute('data-tail');
  var wide = false;
  try { wide = window.matchMedia('(min-width: 1400px) and (min-height: 760px)').matches; } catch (err) { /* keep it open */ }
  d.open = chosen ? chosen === 'open' : !(wide && window.innerHeight < __OPEN__);
}
element.addEventListener('click', function (e) {
  var s = e.target.closest ? e.target.closest('details.shep-banner-more > summary') : null;
  if (s) { element.setAttribute('data-tail', s.parentElement.open ? 'closed' : 'open'); }
});
new MutationObserver(fitTail).observe(element, {childList: true, subtree: true});
fitTail();
""".replace("__OPEN__", str(BANNER_OPEN_HEIGHT))


# =============================================================================
# Status, GPU jobs and the reason line
# =============================================================================
def _clock(seconds: Any) -> Optional[str]:
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds < 0:
        return None
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _elapsed(status: Mapping[str, Any]) -> Optional[str]:
    """How long the job has run (or ran, from its start and end)."""
    if status.get("elapsed_seconds") is not None:
        return _clock(status["elapsed_seconds"])
    try:
        start = datetime.fromisoformat(str(status["start_time"]))
        end = datetime.fromisoformat(str(status["end_time"]))
    except (KeyError, TypeError, ValueError):
        return None
    return _clock((end - start).total_seconds())


def _epochs(status: Mapping[str, Any], sep: str = " of ") -> Optional[str]:
    epoch, total = status.get("current_epoch"), status.get("total_epochs")
    if epoch is None:
        return None
    return f"epoch {epoch}{sep}{total}" if total else f"epoch {epoch}"


#: **The status table of the training jobs** (K6 decision 9): each state's
#: kind, never green for a failed, stopped or cancelled run. The run bar's
#: pill reads it, and so does the Fine-tune job card for each child, whose
#: own states are a Fine-tune start's: queued, cancelled (by a stop) and
#: refused (the slot would not take it).
PILL_KIND = {"idle": "idle", "starting": "accent", "running": "running", "stopping": "accent",
             "completed": "ok", "stopped": "grey", "failed": "red",
             "queued": "grey", "cancelled": "grey", "refused": "red"}


def pill_state(status: Mapping[str, Any]) -> Tuple[str, str]:
    """(kind, words) for the status pill."""
    s = status.get("status") or "idle"
    elapsed = _elapsed(status)
    if s == "idle":
        return "idle", "Idle · no training job since the server started"
    if s == "running":
        if status.get("current_epoch") is None:
            phase = ("loading data and the model" if status.get("phase") == "initializing"
                     else "the process is starting")
            return "starting", " · ".join(p for p in ("Starting", phase, elapsed) if p)
        parts = ["RUNNING", _epochs(status, "/")]
        batch, batches = status.get("batch"), status.get("total_batches")
        if batch is not None and batches:
            parts.append(f"batch {batch}/{batches}")
        parts.append(elapsed)
        return "running", " · ".join(p for p in parts if p)
    if s == "stopping":
        return "stopping", " · ".join(p for p in ("Stopping", _epochs(status), elapsed) if p)
    if s == "completed":
        return "completed", " · ".join(p for p in ("Completed", _epochs(status), elapsed) if p)
    if s == "stopped":
        return "stopped", " · ".join(p for p in ("Stopped", _epochs(status), elapsed) if p)
    if s == "failed":
        return "failed", " · ".join(p for p in ("FAILED", _epochs(status), elapsed) if p)
    return "stopped", str(s)


def pill_html(status: Mapping[str, Any]) -> str:
    state, words = pill_state(status)
    kind = PILL_KIND.get(state, "grey")
    mark = {"ok": r.CHECK, "red": '<span class="shep-dot-r" aria-hidden="true"></span>',
            "running": '<span class="shep-dot-w" aria-hidden="true"></span>'}.get(kind, "")
    return (f'<div class="shep-pill shep-pill-{kind}" data-state="{r.e(state)}" role="status">'
            f'{mark}<span>{r.e(words)}</span></div>')


def _gpu_line(active: Optional[Mapping[str, Any]], lock: Optional[Mapping[str, Any]],
              test: Optional[Mapping[str, Any]], *, operator: bool) -> Optional[Dict[str, Any]]:
    """The GPU job holding the one slot when it is not this page's training
    job, from the job runner (a test run) or the GPU lock's holder record (a
    process this server did not start: a formal CLI, or another server's
    runner on the same lock; or a lock that cannot be used): ``{"text",
    "test"}`` or None."""
    if active and active.get("job_type") == "test":
        inputs = (test or {}).get("inputs") or {}
        label = inputs.get("version_label")
        elapsed = None
        try:
            started = datetime.fromisoformat(str(active.get("start_time")))
            elapsed = _clock((datetime.now() - started).total_seconds())
        except (TypeError, ValueError):
            pass
        detail = " · ".join(str(p) for p in (label, elapsed) if p)
        text = "GPU job: test run" + (f" ({detail})" if detail else " (job runner)")
        if active.get("status") == "stopping":
            text += " · stopping"
        return {"text": text, "test": True}
    if active or lock is None:
        return None
    chip = gpu_chip_from(None, lock, operator=operator)
    if chip.kind == "busy" or lock.get("problem"):
        text = chip.text if chip.kind == "busy" else "the GPU lock cannot be used"
        if lock.get("problem") and operator:
            text += f" ({lock['problem']})"
        return {"text": text, "test": False}
    return None


def run_bar_view(status: Mapping[str, Any], ws: Mapping[str, Any], mode: Mapping[str, Any],
                 *, access: Any, active: Optional[Mapping[str, Any]],
                 gpu: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """What the run bar's actions are and why, as one decision for the tick
    and every action: ``start``, ``cont``, ``stop`` and ``resume`` (interactive
    or not), ``reason`` (escaped HTML: one line), ``drawer_reason`` and
    ``progress`` (a fraction while running, else None)."""
    s = status.get("status") or "idle"
    training_busy = s in ("running", "stopping")
    drawer = bool(mode.get("drawer"))
    row = ws.get("row") or {}
    view: Dict[str, Any] = {"start": False, "cont": False, "stop": False, "resume": False,
                            "progress": None, "drawer_reason": ""}
    epoch, total = status.get("current_epoch"), status.get("total_epochs")
    if s == "running" and isinstance(epoch, int) and isinstance(total, int) and total > 0:
        view["progress"] = max(0.0, min(1.0, epoch / total))

    def line(icon: str, text: str, extra: str = "") -> str:
        return f'<span class="shep-rb-why">{icon}<span>{r.e(text)}</span>{extra}</span>'

    if not access.allowed:
        short = D8_SHORT.get(access.condition, "request not readable")
        view["reason"] = line(LOCK_ICON, f"Operator controls only: {short}")
        view["drawer_reason"] = view["reason"]
        return view
    if training_busy:
        view["stop"] = s == "running" and not confirming(status, mode)
        view["reason"] = line(NO_ICON, "Start and Continue… disabled: a training job is running."
                              if s == "running" else
                              "Stopping: the job is ending; Start and Continue… wait for it.")
        view["drawer_reason"] = view["reason"]
        return view
    if gpu:
        open_test = ('<button type="button" class="shep-link" data-act="open-test">Open Test ›</button>'
                     if gpu.get("test") else "")
        # The sentence ends with its period, or with "· Open Test ›" instead.
        view["reason"] = (f'<span class="shep-rb-why"><span class="shep-tag shep-tag-accent">GPU</span>'
                          f'<span>Start and Continue… disabled — {r.e(gpu["text"])}'
                          f'{" · " + open_test if open_test else "."}</span></span>')
        view["drawer_reason"] = view["reason"]
        return view
    if not row:
        view["reason"] = line(NO_ICON, "Start and Continue… disabled: select a dataset.")
        view["drawer_reason"] = view["reason"]
        return view
    if not row.get("trainable"):
        view["reason"] = line(NO_ICON, f"Start and Continue… disabled: {row.get('name')} is not "
                                   f"trainable: {r.not_trainable_short(row)}.")
        view["drawer_reason"] = view["reason"]
        return view
    if not drawer:
        view["start"] = view["cont"] = True
        view["reason"] = line(NO_ICON, "Stop disabled: no training is running.")
        return view
    parent = mode.get("parent") or {}
    why = continue_refusal(parent, ws)
    view["resume"] = why is None
    text = "Start disabled: close Continue… to start a new run."
    if why:
        # One word group per sentence (`renderers.word_groups`): a line breaks
        # between the two sentences, never leaving the last word of the second
        # alone on a line under the first ("…cannot be" / "continued.").
        view["reason"] = (f'<span class="shep-rb-why">{NO_ICON}<span>'
                          f'{r.word_groups(text, f"Continue Training disabled: {why}")}</span></span>')
        view["drawer_reason"] = line(NO_ICON, f"Continue Training, in the run bar above, is disabled: {why}")
    else:
        view["reason"] = line(NO_ICON, text)
    return view


def _other_workspace(item: Mapping[str, Any], ws: Mapping[str, Any]) -> Optional[str]:
    """Why a listed parent is not the run bar's workspace's (the list was read
    for another workspace than the one selected now), or None."""
    listed = item.get("workspace")
    if listed is None or listed == ws.get("name"):
        return None
    return (f"the run list shows {listed}’s runs, not {ws.get('name')}’s; refresh the list "
            "(Refresh beside it).")


def continue_refusal(parent: Mapping[str, Any], ws: Mapping[str, Any]) -> Optional[str]:
    """Why Continue Training is disabled for the chosen parent, or None."""
    if not parent.get("chosen"):
        return "choose a checkpoint first."
    other = _other_workspace(parent, ws) if parent.get("source") == CONTINUE_FROM_LIST else None
    if other:
        return other
    if parent.get("refused"):
        return "the check says this checkpoint cannot be continued."
    recorded, current = parent.get("fingerprint"), workspace_fingerprint(ws)
    if r.is_digest(recorded) and current and recorded != current:
        return "the parent’s dataset is not this workspace’s current one."
    return None


# =============================================================================
# The error banner (red: a failed run; amber: PyG's slow fallback)
# =============================================================================
def _get_pyg_warning() -> Optional[str]:
    """PyG native-extension fallback warning, probed in this process (cheap
    after its first, logged, probe: `pyg_native_check` caches it)."""
    try:
        from src.utils.pyg_native_check import get_fallback_warning

        return get_fallback_warning()
    except Exception as exc:  # torch stack may be unavailable in this process
        logger.debug("PyG native check unavailable in UI process: %s", exc)
        return None


#: A logging line at ERROR or CRITICAL ("<date> <time> - <logger> - ERROR -
#: <message>", train_model.py's format, or logging's default "ERROR:<logger>:
#: <message>"): its message.
_LOG_ERROR = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[,.]\d+)? - \S+ - (?:ERROR|CRITICAL) - (?P<m>\S.*)$"
    r"|^(?:ERROR|CRITICAL):[\w.]*:(?P<d>\S.*)$")
#: The last line of a Python traceback: an exception's (dotted) class name,
#: then ": message" or nothing, not indented, under the traceback's indented
#: lines.
_EXCEPTION_LINE = re.compile(r"^(?:[A-Za-z_]\w*\.)*[A-Z]\w*(?::\s+\S.*)?$")


def last_error_line(lines: Sequence[str]) -> Optional[str]:
    """The last error line a run printed in `lines` (its output tail), or None:
    a logging line at ERROR or CRITICAL (its message), or the exception line
    that ends a Python traceback (as printed: the class and the message),
    whichever came last. Nothing is guessed from other lines."""
    found = None
    previous = ""
    for line in lines:
        text = line.rstrip()
        logged = _LOG_ERROR.match(text)
        if logged:
            found = (logged.group("m") or logged.group("d")).strip()
        elif (text and previous[:1].isspace() and _EXCEPTION_LINE.match(text)):
            found = text
        previous = line
    return found


def _log_where(run: Optional[str], run_dir: Optional[Path]) -> str:
    """Where a failed run's log is, said only when it exists: its training log
    (``logs/train_*.json``) named; else that none is there (no log in its
    ``logs/``; or no run directory, as a run refused before it made its run
    directory leaves it). Without `run_dir` (not known here) nothing is said of the log."""
    if not run:
        return ""
    head = f'run <code>{r.e(run)}</code>'
    if run_dir is None:
        return f'<span class="shep-banner-where">{head}</span>'
    try:
        exists = Path(run_dir).is_dir()
    except OSError:
        exists = False
    if not exists:
        said = "no run directory, so no log"
    else:
        log = training_manager.metrics_log_file(Path(run_dir))
        said = (f'its log: <code>logs/{r.e(log.name)}</code>' if log is not None
                else 'no log in its <code>logs/</code>')
    return f'<span class="shep-banner-where">{head} · {said}</span>'


def _as_is(text: str) -> str:
    return text


def read_only_scrub(root: Optional[Path]) -> Callable[[str], str]:
    """A read-only request's redaction of a printed line (`redact`: no
    absolute path), its leading indentation kept: `redact` joins whitespace,
    and a traceback's lines and its ``^^^`` markers line up by their indent."""
    def scrub(text: str) -> str:
        text = str(text)
        return text[:len(text) - len(text.lstrip())] + redact(text, root)
    return scrub


def _render_error_block(message: str, *, headline: str = "Training failed",
                        run: Optional[str] = None, run_dir: Optional[Path] = None,
                        scrub: Callable[[str], str] = _as_is) -> str:
    """The red block: the headline, the run and where its log is (only when
    there is one), Dismiss; under the headline the last error line the run
    printed (`last_error_line`), so the cause is read without scrolling; and
    the output tail (monospace, escaped) in a disclosure, open as sent; the
    banner's script closes it on a wide window under `BANNER_OPEN_HEIGHT` px
    tall, where it would squeeze the charts (`_BANNER_JS`). Every text the run
    printed goes through `scrub` (a read-only request's redaction), the cause
    as the tail's lines, and then is escaped: the redaction reads text, never
    the markup around it."""
    first, _sep, tail = str(message).partition("\n\n--- Last output ---\n")
    every = tail.splitlines() if tail else []
    lines = [scrub(line) for line in every[-BANNER_TAIL_LINES:]]
    cause = last_error_line(every)
    cause_html = (f'<div class="shep-banner-cause"><span class="shep-banner-k">Last error '
                  f'printed:</span> <span class="shep-banner-v">{r.e(scrub(cause))}</span></div>'
                  if cause else "")
    count = f'{len(lines)} line{"" if len(lines) == 1 else "s"}'
    body = ('<details class="shep-banner-more" open><summary>Last output '
            f'<span class="shep-banner-n">({count})</span></summary>'
            f'<pre class="shep-banner-tail">{r.e(chr(10).join(lines))}</pre></details>'
            if lines else "")
    return ('<div class="shep-banner shep-banner-red" role="alert">'
            f'<div class="shep-banner-hd"><span class="shep-banner-h">⚠ {r.e(headline)}: '
            f'{r.e(scrub(first.strip()))}</span>{_log_where(run, run_dir)}'
            '<button type="button" class="shep-btn shep-banner-x" data-act="dismiss">× Dismiss</button>'
            f'</div>{cause_html}{body}</div>')


def _render_warning_block(message: str, scrub: Callable[[str], str] = _as_is) -> str:
    """The amber block for a non-fatal advisory."""
    return ('<div class="shep-banner shep-banner-amber" role="status">'
            f'<div class="shep-banner-hd"><span class="shep-banner-h">⚠ Warning: '
            f'{r.e(scrub(str(message)))}</span>'
            '<button type="button" class="shep-btn shep-banner-x" data-act="dismiss">× Dismiss</button>'
            '</div></div>')


def _build_error_banner_payload(
    status_info: Mapping[str, Any], pyg_warning: Optional[str], *, run: Optional[str] = None,
    run_dir: Optional[Path] = None, scrub: Callable[[str], str] = _as_is,
) -> Tuple[Optional[str], Optional[str]]:
    """``(html, key)`` for the banner, or ``(None, None)`` when nothing must
    show. ``key`` names the content, so a session's dismissal holds until it
    changes (a new or different error shows again). `scrub` redacts each text
    before it is escaped (a read-only request: no absolute path)."""
    blocks: List[str] = []
    key_parts: List[str] = []
    error_msg = status_info.get("error_message")
    if error_msg:
        epoch = _epochs(status_info)
        headline = f"Training failed at {epoch}" if epoch else "Training failed"
        blocks.append(_render_error_block(error_msg, headline=headline, run=run, run_dir=run_dir,
                                          scrub=scrub))
        key_parts.append("ERR:" + str(error_msg))
    if pyg_warning:
        blocks.append(_render_warning_block(pyg_warning, scrub))
        key_parts.append("PYG:" + str(pyg_warning))
    if not blocks:
        return None, None
    return "".join(blocks), "||".join(key_parts)


# =============================================================================
# The run view
# =============================================================================
def _shown_path(path: Any) -> str:
    """A server path as an operator sees it here: relative to the workspaces
    root when it is inside it (``<workspace>/runs/…``), else as it is."""
    root, _problem = workspaces_root()
    text = str(path)
    if root is not None:
        try:
            return Path(text).resolve().relative_to(Path(root).resolve()).as_posix()
        except (ValueError, OSError):
            pass
    return text


def _snapshot_sentence(snapshot: Optional[Dict[str, Any]]) -> str:
    """Where a run can be continued from, in one sentence (contract M3a); the
    checkpoint named relative to the workspaces root."""
    if not snapshot:
        return "No checkpoint was saved, so nothing can be continued."
    shown = _shown_path(snapshot.get("checkpoint", ""))
    if not snapshot.get("readable", True):
        return f"The latest checkpoint {shown} cannot be read ({snapshot.get('reason')})."
    epoch, target = snapshot.get("completed_epochs"), snapshot.get("target_epochs")
    where = f"epoch {epoch} of {target}" if target else f"epoch {epoch}"
    end = snapshot.get("end")
    if end == "finished":
        return f"Finished ({where}): {shown}. To train further, use Continue…"
    if end == "stopped_early":
        return f"Stopped early at {where} ({snapshot.get('stop_reason')}); it cannot be continued."
    label = "Paused at" if end == "paused" else "Latest resumable checkpoint:"
    return f"{label} {where}, {shown}."


class RunRef(Dict[str, Any]):
    """The run the view shows: ``id`` (what the chart's token and server
    function name it by), ``label`` ("gat/<run>"), ``path``, ``workspace``
    and ``latest`` (True for a workspace's latest run, before any job since
    the server started)."""


def workspace_of_run(run_dir: Any) -> Optional[str]:
    """The listed workspace a run directory is inside, by the name the server
    lists it under, or None (one listing of the root: `resolve_workspaces`)."""
    from src.api.services.server_lists import ServerListRefused, resolve_workspaces

    try:
        directories = resolve_workspaces()
    except ServerListRefused:
        return None
    try:
        resolved = Path(run_dir).resolve()
    except OSError:
        resolved = Path(run_dir)
    return next((listed for listed, directory in directories.items()
                 if directory in resolved.parents), None)


def _runner_run(status: Mapping[str, Any], cache: Dict[str, Any]) -> Optional[RunRef]:
    """The run the job runner started last, named by its listed workspace
    (``<workspace>/<arch>/<run>``) or, when the workspace is not listed, by
    its job (``job:<id>``). The workspace's name is looked up once per run
    and session (`workspace_of_run`)."""
    run_dir = training_manager.run_dir
    if run_dir is None or (status.get("status") or "idle") == "idle":
        return None
    key = str(run_dir)
    named = cache.get("runner_ws")
    if not named or named[0] != key:
        named = (key, workspace_of_run(run_dir))
        cache["runner_ws"] = named
    arch = Path(run_dir).parent.name
    label = f"{arch}/{Path(run_dir).name}"
    run_id = f"{named[1]}/{label}" if named[1] else f"job:{status.get('job_id')}"
    return RunRef(id=run_id, label=label, path=Path(run_dir), workspace=named[1], latest=False)


def _latest_of(ws: Mapping[str, Any], cache: Dict[str, Any]) -> Optional[str]:
    """The run bar's workspace's latest run ("<arch>/<run>"): first as its
    selection read it, then listed again at most every `LATEST_READ_SECONDS`
    per session, so a run started elsewhere (the CLI, the API, another
    browser) is shown once it writes a checkpoint, while the page stays
    open."""
    name = ws.get("name")
    if not name:
        return None
    now = datetime.now().timestamp()
    held = cache.get("latest")
    if held and held[0] == name and now - held[1] < LATEST_READ_SECONDS:
        return held[2]
    if held and held[0] == name:
        from src.api.services.server_lists import ServerListRefused

        try:
            latest = _latest_run(_workspace_path(name))
        except ServerListRefused:
            latest = None
    else:
        latest = ws.get("latest_run")
    cache["latest"] = [name, now, latest]
    return latest


def _current_run(status: Mapping[str, Any], ws: Mapping[str, Any],
                 cache: Dict[str, Any]) -> Optional[RunRef]:
    runner = _runner_run(status, cache)
    if runner is not None:
        return runner
    latest = _latest_of(ws, cache)
    if latest and ws.get("name"):
        path = resolve_run(f"{ws['name']}/{latest}")
        if path is not None:
            return RunRef(id=f"{ws['name']}/{latest}", label=latest, path=path,
                          workspace=ws["name"], latest=True)
    return None


def resolve_run(run_id: Any) -> Optional[Path]:
    """The directory of the run the server lists as `run_id`, or None: a
    listed workspace's run, found by listing ``runs/<arch>/`` again (never by
    joining the id onto a path), inside that workspace, for a run layout key
    (`RUN_LAYOUT_KEYS`: a model bundle's encoder kind, or a ShepherdGNN conv
    type for a run kept as history); or ``job:<id>``, the job runner's latest
    training job."""
    if not isinstance(run_id, str) or len(run_id) > 400:
        return None
    if run_id.startswith("job:"):
        status = training_manager.get_job_status_word("training")
        job_id = training_manager.get_status().get("job_id") if status != "idle" else None
        return training_manager.run_dir if job_id and run_id == f"job:{job_id}" else None
    parts = run_id.split("/")
    if len(parts) != 3 or not all(parts):
        return None
    workspace, arch, run = parts
    from src.api.services.server_lists import ServerListRefused, resolve_workspace
    from src.utils.checkpoint_paths import RUN_LAYOUT_KEYS, RUNS_DIRNAME

    if arch not in RUN_LAYOUT_KEYS:
        return None
    try:
        directory = resolve_workspace(workspace)
        entries = list((directory / RUNS_DIRNAME / arch).iterdir())
    except (ServerListRefused, OSError):
        return None
    for entry in entries:
        if entry.name != run:
            continue
        try:
            resolved = entry.resolve()
            inside = directory in resolved.parents and resolved.is_dir()
        except OSError:
            inside = False
        return entry if inside else None
    return None


def _snapshot_for(run: RunRef, cache: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The run's latest saved state (`TrainingManager.latest_snapshot`, which
    reads its ``last.pt``), read again only when the file changed since this
    session last read it."""
    from src.utils.checkpoint_paths import run_checkpoint_dir

    last = run_checkpoint_dir(run["path"]) / "last.pt"
    try:
        stat = last.stat()
        key = [str(last), stat.st_mtime_ns, stat.st_size]
    except OSError:
        key = [str(last), None, None]
    held = cache.get("snapshot")
    if held and held[0] == key:
        return held[1]
    snapshot = training_manager.latest_snapshot(run["path"]) if key[1] is not None else None
    cache["snapshot"] = [key, snapshot]
    return snapshot


def _used_while_live(run: RunRef, cache: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """While the job runner's run is live: only what the run view's header
    needs, the dataset fingerprint the run records. It is fixed for a run, so
    it is read from the run's first saved state once per run and session and
    then kept (``last.pt`` is rewritten every epoch and read whole); the live
    snapshot sentence names the newest file without reading it."""
    held = cache.get("used")
    if held and held[0] == run["id"]:
        return {"readable": True, "dataset_fingerprint": held[1]}
    snapshot = _snapshot_for(run, cache)
    if snapshot and snapshot.get("readable", True):
        cache["used"] = [run["id"], snapshot.get("dataset_fingerprint")]
    return snapshot


def _newest_checkpoint(run_dir: Path) -> Optional[Path]:
    """The newest checkpoint file of a run, by its time (nothing is read)."""
    from src.utils.checkpoint_paths import run_checkpoints

    best, when = None, -1.0
    for path in run_checkpoints(run_dir):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > when:
            best, when = path, mtime
    return best


def digest_html(digest: Any, digits: int = r.SHORT_HEX, what: str = "fingerprint") -> str:
    """A digest's short form in bold monospace, the full value in its title
    and on an icon Copy; "not recorded" when there is none."""
    if not r.is_digest(digest):
        return '<span class="shep-muted">not recorded</span>'
    return (f'<code class="shep-fpv" title="{r.e(digest)}">{r.e(str(digest)[:digits])}</code> '
            + copy_button(str(digest), f"the full {what}", icon_only=True))


def _dataset_relation(ws: Mapping[str, Any], *, operator: bool,
                      snapshot: Optional[Mapping[str, Any]], live: bool) -> Tuple[str, str]:
    """How the run's recorded dataset stands to the run bar's: ``(tag, row)``,
    the header's short tag (always shown: a mismatch, a missing record or a
    refusal changes what the curves mean) and the Details row with the
    fingerprint itself. Reading it opens the run's checkpoint: an operator's
    only."""
    if not operator:
        return (f'<span class="shep-tag shep-tag-grey">{LOCK_ICON} dataset not shown · operator '
                'control</span>',
                '<span class="shep-muted">not shown: reading the checkpoint is an operator '
                'control</span>')
    if snapshot is None:
        words = "recorded at the first checkpoint" if live else "not recorded (no checkpoint)"
        return f'<span class="shep-muted">dataset {words}</span>', f'<span class="shep-muted">{words}</span>'
    if not snapshot.get("readable", True):
        return ('<span class="shep-tag shep-tag-amber">checkpoint cannot be read</span>',
                '<span class="shep-muted">not read (the checkpoint cannot be read)</span>')
    recorded, current = snapshot.get("dataset_fingerprint"), workspace_fingerprint(ws)
    row = digest_html(recorded, ws.get("digits") or r.SHORT_HEX, "dataset fingerprint")
    if not (r.is_digest(recorded) and current):
        return '<span class="shep-muted">dataset not recorded</span>', row
    if recorded == current:
        return f'<span class="shep-tag shep-tag-ok">{r.CHECK} same dataset as the run bar</span>', row
    return '<span class="shep-tag shep-tag-amber">≠ not the selected dataset</span>', row


def run_head_html(run: Optional[RunRef], ws: Mapping[str, Any], *, operator: bool,
                  snapshot: Optional[Mapping[str, Any]], live: bool = False,
                  continuing: Optional[str] = None) -> str:
    """The run view's header, one line: "Run shown" and the run as a person
    names it (`run_words`), whether it is the workspace's latest, how its
    recorded dataset stands to the run bar's (`_dataset_relation`), and, while
    Continue… is open with a listed checkpoint chosen, whether this is the run
    continued from (`continuing`, its "<architecture>/<run>"): the drawer's
    parent and the run shown are different runs unless chosen so. The rest,
    read less often (the run's directory, its checkpoint and logs, the dataset
    fingerprint it used; its workspace when it is not the run bar's), is in
    Details."""
    if run is None:
        return ('<div class="shep-panel-hd shep-rv-hd"><h2>Run shown</h2><span class="shep-muted">'
                'none yet</span></div>')
    label = str(run["label"])
    arch, _, directory = label.partition("/")
    tag, used = _dataset_relation(ws, operator=operator, snapshot=snapshot, live=live)
    latest = '<span class="shep-muted">latest</span>' if run.get("latest") else ""
    relation = ""
    if continuing:
        relation = ('<span class="shep-tag shep-tag-accent">the run continued from</span>'
                    if continuing == label else
                    '<span class="shep-tag shep-tag-grey">not the run continued from</span>')
    rows = [("Run", f"<code>{r.e(label)}</code>")]
    if run.get("workspace") and run.get("workspace") != ws.get("name"):
        # Said here only when it is not the run bar's (which names its own).
        rows.append(("Workspace", r.e(run["workspace"])))
    if operator:
        if live:
            newest = _newest_checkpoint(run["path"])
            rows.append(("Newest checkpoint", f"<code>{r.e(_shown_path(newest))}</code>" if newest
                         else '<span class="shep-muted">none written yet</span>'))
        elif snapshot and snapshot.get("checkpoint"):
            rows.append(("Checkpoint", f"<code>{r.e(_shown_path(snapshot['checkpoint']))}</code>"))
    rows.append(("Logs", "the run directory’s <code>logs/</code>"))
    rows.append(("Dataset used", used + help_button("training-used-dataset",
                                                    "Used dataset fingerprint")))
    cells = "".join(f'<span class="shep-k">{k}</span><span class="shep-v">{v}</span>' for k, v in rows)
    details = ('<button type="button" class="shep-link shep-rv-details" data-act="details" '
               'aria-expanded="false" aria-controls="shep-train-run-details">Details ›</button>'
               '<div class="shep-pop shep-rv-pop" id="shep-train-run-details" role="dialog" '
               'aria-label="Run details" hidden><div class="shep-pop-hd"><b>Run details</b>'
               '<button type="button" class="shep-pop-x" data-act="close" '
               f'aria-label="Close the run details">×</button></div><div class="shep-kv">{cells}'
               '</div></div>')
    return (f'<div class="shep-panel-hd shep-rv-hd"><h2><span class="shep-rv-k">Run shown</span> '
            f'<span class="shep-rv-name" title="{r.e(label)}">'
            f'{r.e(run_words(arch, directory))}</span></h2>{latest}{tag}{relation}'
            f'<span class="shep-rv-tools">{details}</span></div>')


def _epochs_done(snapshot: Mapping[str, Any]) -> str:
    """"195 of 1000 epochs completed" from a snapshot's progress record."""
    done, target = snapshot.get("completed_epochs"), snapshot.get("target_epochs")
    return f"{done} of {target} epochs completed" if target else f"{done} epochs completed"


def run_snapshot_html(run: Optional[RunRef], status: Mapping[str, Any], *, operator: bool,
                      snapshot: Optional[Mapping[str, Any]]) -> str:
    """The run's state in one short line under the header: how far it got and
    whether it can be continued, for an operator from its checkpoint's record;
    for a read-only request its state and that the rest is an operator's. The
    checkpoint's and the logs' paths are in the header's Details."""
    if run is None:
        return ('<p class="shep-rv-snap">No training job since the server started, and the run '
                'bar’s dataset has no runs yet: start one with the parameters on the left.</p>')
    s = (status.get("status") or "idle") if not run.get("latest") else "latest"
    live = s in ("running", "stopping")
    try:
        exists = Path(run["path"]).is_dir()
    except OSError:
        exists = False
    ended = pill_state(status)[1].split(" · ")[0].capitalize()
    if not exists and not live:
        # A run refused before it made its run directory (e.g. "--device
        # auto requires CUDA" on a host without it): there is no
        # checkpoint and no log to point to (for any request: a directory's
        # existence is not checkpoint inspection).
        state = (f"{ended} ({_epochs(status)}). " if _epochs(status) else f"{ended}. ") \
            if s in ("completed", "stopped", "failed") else ""
        return (f'<p class="shep-rv-snap">{r.e(state)}No run directory exists for this run, so '
                'it has no checkpoint and no log.</p>')
    if live:
        epochs = _epochs(status)
        return f'<p class="shep-rv-snap">Running{" · " + r.e(epochs) if epochs else ""}</p>'
    if not operator:
        state = (f"{ended} · {_epochs(status)}" if s in ("completed", "stopped", "failed")
                 and _epochs(status) else "")
        return (f'<p class="shep-rv-snap">{r.e(state)}{" · " if state else ""}'
                'checkpoint details are an operator control</p>')
    if not snapshot:
        return '<p class="shep-rv-snap">No checkpoint saved, so nothing can be continued</p>'
    if not snapshot.get("readable", True):
        return (f'<p class="shep-rv-snap">The latest checkpoint cannot be read '
                f'({r.e(snapshot.get("reason"))})</p>')
    end = snapshot.get("end")
    if end == "finished":
        line = f"Finished · {_epochs_done(snapshot)} · to train further, use Continue…"
    elif end == "stopped_early":
        reason = snapshot.get("stop_reason")
        line = (f"Stopped early{f' ({reason})' if reason else ''} · {_epochs_done(snapshot)} · "
                "cannot be continued")
    else:
        line = _epochs_done(snapshot) + (" · can be continued" if snapshot.get("resumable", True)
                                         else "")
    return f'<p class="shep-rv-snap">{r.e(line)}</p>'


# =============================================================================
# The charts: data from the run's metrics log (the server function)
# =============================================================================
#: The four charts, in figure order (2×2): Loss, the Trainer's validation
#: figure ("mrr") and its companions ("hits"), the learning rate. The ids stay
#: whatever the run's kind; the middle two take their titles and series from
#: `VALIDATION_CHARTS`.
CHART_TITLES = {
    "loss": "Loss",
    "lr": "Learning rate (log)",
}
SHARED_AXIS = "epoch (shared axis)"
_HITS = re.compile(r"^val_hits@(\d+)$")
_POOLED_TOP = re.compile(r"^val_pooled_top(\d+)$")
#: The middle two charts by the run's kind, read from the keys its validation
#: log holds (`validation_kind`): ``(the figure's title, its series, the
#: companions' title, theirs)``, each series ``(label, key)``; the companions
#: by cut-off are filled from the log (`_k_series`). A run that has logged no
#: validation yet is "none": its two charts wait, titled by what they will
#: show for any run.
VALIDATION_CHARTS = {
    "base": ("LP accuracy (validation, Trainer)", (("LP accuracy", "val_lp_accuracy"),),
             "ROC-AUC and AP (validation, Trainer)",
             (("ROC-AUC", "val_lp_roc_auc"), ("AP", "val_lp_average_precision"))),
    "finetune": ("Pooled MRR (validation, Trainer)", (("pooled MRR", "val_pooled_mrr"),),
                 "Pooled top-k (validation, Trainer)", _POOLED_TOP),
    "history": ("MRR (validation, Trainer top-20)", (("MRR", "val_mrr"),),
                "Hits@K (validation, Trainer)", _HITS),
    "none": ("Validation figure (Trainer)", (), "More validation figures (Trainer)", ()),
}


def validation_kind(validation: Sequence[Mapping[str, Any]]) -> str:
    """Which run a validation log is, by its keys: "history" for a file
    without a model spec (MRR, Hits@K), "finetune" (pooled MRR), "base" (LP
    accuracy), or "none" before the first validation."""
    keys = {str(key) for entry in validation for key in entry}
    if "val_mrr" in keys or any(_HITS.match(key) for key in keys):
        return "history"
    if "val_pooled_mrr" in keys:
        return "finetune"
    if "val_lp_accuracy" in keys:
        return "base"
    return "none"


def _k_series(validation: Sequence[Mapping[str, Any]], pattern: Any, words: str
              ) -> List[Tuple[str, str]]:
    """Every cut-off k the log names by `pattern`, in order, as ``(label,
    key)``: "Hits@1", "top-10"."""
    found = {}
    for entry in validation:
        for key in entry:
            match = pattern.match(str(key))
            if match:
                found[int(match.group(1))] = str(key)
    return [(f"{words}{k}", found[k]) for k in sorted(found)]


def _by_epoch(entries: Sequence[Mapping[str, Any]], *keys: str) -> Dict[int, Any]:
    """{1-indexed epoch: the first of `keys` an entry has} (epochs in the log
    are 0-indexed; the status and the charts count from 1)."""
    out: Dict[int, Any] = {}
    for entry in entries:
        epoch = entry.get("epoch") if isinstance(entry, Mapping) else None
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            continue
        for key in keys:
            if entry.get(key) is not None:
                out[epoch + 1] = entry.get(key)
                break
    return out


def _has_value(values: Sequence[Any]) -> bool:
    return any(charts.rounded(v) is not None for v in values)


def training_charts(history: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The four chart specs from a metrics history (`get_metrics_history`):
    Loss (train, and validation where the run logs one: ``val_loss``, a
    base's ``val_lp_loss``), the Trainer's validation figure and its
    companions as the run's kind logs them (`VALIDATION_CHARTS`: a base's LP
    accuracy, then its ROC-AUC and AP; a task model's pooled MRR, then its
    pooled top-k; a file without a model spec's MRR, then its Hits@K), and
    the learning rate on a log axis. A series with no value yet is left out,
    so no chart draws a figure the run does not log."""
    train = [e for e in history.get("train") or [] if isinstance(e, Mapping)]
    val = [e for e in history.get("validation") or [] if isinstance(e, Mapping)]
    specs = []

    def spec(chart_id, title, columns, *, y_scale="linear", x_label=None):
        epochs = sorted(set().union(*[set(c) for _label, c in columns])) if columns else []
        series = [(label, [col.get(x) for x in epochs]) for label, col in columns]
        series = [(label, values) for label, values in series if _has_value(values)]
        return charts.chart_spec(chart_id, title, epochs if series else [],
                                 series, y_scale=y_scale, x_label=x_label)

    def columns(pairs):
        return [(label, _by_epoch(val, key)) for label, key in pairs]

    figure_title, figure, others_title, others = VALIDATION_CHARTS[validation_kind(val)]
    if not isinstance(others, tuple):
        others = _k_series(val, others, "Hits@" if others is _HITS else "top-")
    specs.append(spec("loss", CHART_TITLES["loss"],
                      [("train", _by_epoch(train, "total", "train_loss")),
                       ("validation", _by_epoch(val, "val_loss", "val_lp_loss"))]))
    specs.append(spec("mrr", figure_title, columns(figure)))
    specs.append(spec("hits", others_title, columns(others), x_label=SHARED_AXIS))
    specs.append(spec("lr", CHART_TITLES["lr"],
                      [("learning rate", _by_epoch(train, "learning_rate"))],
                      y_scale="log", x_label=SHARED_AXIS))
    return specs


def _rows(history: Mapping[str, Any]) -> int:
    return len(history.get("train") or []) + len(history.get("validation") or [])


def training_chart_series(data: Any) -> Dict[str, Any]:
    """The chart component's server function: the figure for the run its
    token names, if the server lists that run (`resolve_run`; never a path),
    read from the run's metrics log. Not an operator control: the metrics log
    is not checkpoint inspection."""
    t = charts.token_from(data)
    run_id = t.get("run")
    path = resolve_run(run_id)
    if path is None:
        return charts.figure_payload(None, 0, [], empty="This run is no longer listed.")
    history = training_manager.get_metrics_history(path)
    return charts.figure_payload(run_id, _rows(history), training_charts(history),
                                 empty="No epochs logged yet.")


def _metrics_csv_text(history: Mapping[str, Any]) -> str:
    """Every metric the log holds, one row per entry and split (epochs from 1)."""
    train = [e for e in history.get("train") or [] if isinstance(e, Mapping)]
    val = [e for e in history.get("validation") or [] if isinstance(e, Mapping)]
    keys = sorted({k for entry in train + val for k in entry} - {"epoch"})
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=["epoch", "split", *keys], extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for split, entries in (("train", train), ("validation", val)):
        for entry in entries:
            row = {k: entry.get(k, "") for k in keys}
            epoch = entry.get("epoch")
            row["epoch"] = epoch + 1 if isinstance(epoch, int) and not isinstance(epoch, bool) else ""
            row["split"] = split
            writer.writerow(row)
    return out.getvalue()


def training_chart_csv(data: Any) -> Dict[str, Any]:
    """Export CSV, the component's second server function: the run's whole
    metrics log as CSV text, which the browser saves (no file is written on
    the server)."""
    t = charts.token_from(data)
    run_id = t.get("run")
    path = resolve_run(run_id)
    if path is None:
        return {"filename": None, "csv": ""}
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id).split("/", 1)[-1])
    return {"filename": f"training_metrics_{name}.csv",
            "csv": _metrics_csv_text(training_manager.get_metrics_history(path))}


#: The chart's line for a run with no run directory: a live one has not made
#: it yet; an ended one has none (a run refused before it made its run directory).
NO_RUN_DIR_LIVE = "No epochs logged yet."
NO_RUN_DIR_ENDED = "This run has no run directory, so it logged no epochs to chart."


def _chart_token(run: Optional[RunRef], cache: Dict[str, Any], *,
                 live: bool = False) -> Dict[str, Any]:
    """The component's token for the run shown: its id and how many rows its
    metrics log holds, counted again only when the log file changed (a stat).
    A run whose directory does not exist is not asked for (the component
    would say "no longer listed" of a run that was never listed): its line
    says why there is nothing, `live` or ended."""
    if run is None:
        return charts.token(None, 0, empty="No run to show yet.")
    try:
        exists = Path(run["path"]).is_dir()
    except OSError:
        exists = False
    if not exists:
        return charts.token(None, 0, empty=NO_RUN_DIR_LIVE if live else NO_RUN_DIR_ENDED)
    log = training_manager.metrics_log_file(run["path"])
    try:
        stat = log.stat() if log is not None else None
        sig = [run["id"], str(log), stat.st_mtime_ns, stat.st_size] if stat else [run["id"], None]
    except OSError:
        sig = [run["id"], None]
    held = cache.get("rows")
    if held and held[0] == sig:
        rows = held[1]
    else:
        rows = _rows(training_manager.get_metrics_history(run["path"])) if sig[1] else 0
        cache["rows"] = [sig, rows]
    return charts.token(run["id"], rows)


# =============================================================================
# The resources line
# =============================================================================
def _format_resources(resources: Mapping[str, Any]) -> str:
    """The resources as one line of 14 px text: each GPU's use, memory and
    temperature (one shared "unified memory" figure on a unified-memory
    machine, never the same bytes twice), and the system memory."""
    parts: List[str] = []
    unified = bool(resources.get("unified_memory"))
    gpu = resources.get("gpu") or {}
    if gpu.get("available"):
        for dev in gpu.get("devices") or []:
            idx = dev.get("index", 0)
            util = dev.get("utilization_percent")
            parts.append(f"GPU {idx} {'N/A' if util is None else f'{util:.0f} %'}")
            if not unified:
                used, total = dev.get("memory_used_mb"), dev.get("memory_total_mb")
                parts.append(f"VRAM {used / 1024:.1f} / {total / 1024:.1f} GB"
                             if used is not None and total else "VRAM N/A")
            temp = dev.get("temperature_c")
            parts.append("N/A °C" if temp is None else f"{temp:.0f} °C")
            if dev.get("name"):
                parts.append(str(dev["name"]))
    else:
        parts.append("GPU: not available")
    ram = resources.get("ram") or {}
    label = "unified memory" if unified else "memory"
    if "error" in ram:
        parts.append(f"{label}: {ram.get('error')}")
    elif ram:
        total = ram.get("total_gb", 0) or 0
        used = (total - ram.get("available_gb", 0)) if unified and ram.get(
            "available_gb") is not None else ram.get("used_gb", 0)
        parts.append(f"{label} {used:.1f} / {total:.1f} GB"
                     + (" (shared CPU + GPU pool)" if unified else ""))
    items = '<span class="shep-sep" aria-hidden="true"> · </span>'.join(
        f'<span>{r.e(p)}</span>' for p in parts)
    return f'<div class="shep-rv-res"><b>Resources</b> {items}</div>'


_IDLE_RESOURCE_HTML = '<div class="shep-rv-res"><b>Resources</b> <span class="shep-muted">reading…</span></div>'


def _poll_resources(location: Any = TRAINING) -> Any:
    """The resources line, on the shell's slow timer while Training is shown."""
    if location != TRAINING:
        return gr.update()
    try:
        return _format_resources(training_manager.get_system_resources())
    except Exception:  # noqa: BLE001 - the line must not break the page
        logger.exception("The resources could not be read")
        return '<div class="shep-rv-res"><b>Resources</b> <span class="shep-muted">not read</span></div>'


# =============================================================================
# The view: one render for the tick and after every action
# =============================================================================
#: The view's pieces, in output order (`create_training_tab` lists the same).
VIEW_KEYS = ("ws", "dataset", "chip", "pill", "reason", "start", "cont", "stop", "resume",
             "runbar", "page", "run_name", "banner", "head", "snap", "charts", "drawer_reason",
             "training_workspace")
#: The pieces that are the session's selection (the run bar's workspace):
#: written by a render only when it changed them (a running job's dataset:
#: `render`'s ``ws_changed``).
SELECTION_KEYS = ("ws", "training_workspace")
#: The one piece the fast poll writes (`_poll`), on every tick, besides the
#: renders: the status pill, drawn from the job runner alone, never from the
#: session's selection. A tick that crosses a render can show an older pill
#: until the next tick.
POLL_KEYS = ("pill",)
#: Every other piece but the selection's own, which a queued render alone
#: writes (they are drawn from the session's selection, ``ws``, and ``mode``
#: where it shapes them: the drawer, Stop's confirmation; the banner and the
#: run name with them): when the poll finds one of them not what the session
#: was last sent, it hands the change over (`CHANGED_ID`).
HANDED_KEYS = tuple(key for key in VIEW_KEYS if key not in SELECTION_KEYS + POLL_KEYS)
#: The reads a render keeps per session that are throttled by time, not by a
#: file's change (the GPU lock's probe, `LOCK_READ_SECONDS`; the workspace's
#: runs, `LATEST_READ_SECONDS`): the fast poll keeps its own (``polled``, which
#: only it writes), so it reads them no more often than a render does; the
#: poll and the render it hands a change over to take each from there when it
#: is newer than the last render's (`_with_polled`).
POLL_READS = ("lock", "latest")
#: The hidden component the fast poll hands a change over through (`_poll`),
#: whose ``.change`` runs the queued render (`_render_changed`): not a
#: ``gr.State``, because the answer to a ``queue=False`` request dispatches a
#: component's ``.change``, never a State's.
CHANGED_ID = "shep-train-changed"


def gpu_view(active, cache: Dict[str, Any], operator: bool) -> Optional[Dict[str, Any]]:
    """The GPU line: the job runner each time; the lock (it probes the flock)
    at most every `LOCK_READ_SECONDS` per session, when the runner holds no job."""
    if active:
        test = training_manager.get_test_status() if active.get("job_type") == "test" else None
        return _gpu_line(active, None, test, operator=operator)
    now = datetime.now().timestamp()
    held = cache.get("lock")
    if not held or now - held[0] >= LOCK_READ_SECONDS:
        try:
            lock = lock_state()
        except Exception:  # noqa: BLE001 - a lock that cannot be read is said, not raised
            lock = {"configured": True, "problem": "it could not be read just now"}
        held = [now, lock]
        cache["lock"] = held
    return _gpu_line(None, held[1], None, operator=operator)


def render(ws: Optional[Mapping[str, Any]], mode: Optional[Mapping[str, Any]],
           cache: Optional[Mapping[str, Any]], dismissed: Optional[Sequence[str]],
           request: Optional[gr.Request]) -> Dict[str, Any]:
    """Every piece of the run bar and the run view, computed now for this
    request, as plain values (`VIEW_KEYS`) plus the session's new cache."""
    ws = dict(ws or empty_workspace())
    mode = dict(mode or {})
    cache = dict(cache or {})
    access = operator_access(request)
    status = training_manager.get_status()
    active = status.get("gpu", {}).get("job") if isinstance(status.get("gpu"), Mapping) else None
    s = status.get("status") or "idle"

    # While a training job runs, the select shows that job's dataset.
    run = _current_run(status, ws, cache)
    in_use = s in ("running", "stopping")
    ws_changed = False
    if in_use and run is not None and run.get("workspace") and run["workspace"] != ws.get("name"):
        ws = workspace_state(run["workspace"])
        ws_changed = True
    gpu = None if in_use else gpu_view(active, cache, access.allowed)
    view = run_bar_view(status, ws, mode, access=access, active=active, gpu=gpu)
    # What this render read of the GPU slot, for the header's chips
    # (`chips_on_change`): the training state, the runner's job, the lock's
    # line. Not the test run's line, whose elapsed time changes every tick.
    job = active or {}
    cache["gpu_seen"] = [s, job.get("job_type"), job.get("status"), job.get("start_time"),
                         None if active or gpu is None else gpu.get("text")]

    # The run's checkpoint record (the used fingerprint, the snapshot): read
    # for an operator only, and again only when its last.pt changed; while
    # the job runner's run is live, its recorded fingerprint only, once.
    if run is None or not access.allowed:
        snapshot = None
    elif in_use and not run.get("latest"):
        snapshot = _used_while_live(run, cache)
    else:
        snapshot = _snapshot_for(run, cache)
    # A read-only request sees no absolute path: each printed text is redacted
    # before it is escaped (redacting the built markup took its closing tags
    # for paths: "</span>" became "<a path outside the workspaces root").
    banner_html, banner_key = _build_error_banner_payload(
        status if s == "failed" else {}, _get_pyg_warning(), run=run["label"] if run else None,
        run_dir=run["path"] if run else None,
        scrub=_as_is if access.allowed else read_only_scrub(workspaces_root()[0]))
    if banner_key is None or banner_key in (dismissed or ()):
        banner_html = ""
    cache["banner_key"] = banner_key

    progress = view["progress"]
    bar = (f'<span class="shep-rb-progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" '
           f'aria-valuenow="{round(progress * 100)}" aria-label="Training progress">'
           f'<i style="width:{progress * 100:.1f}%"></i></span>' if progress is not None else "")
    classes = ["shep-runbar"]
    if mode.get("drawer"):
        classes.append("shep-rb-drawer")
    if confirming(status, mode):
        classes.append("shep-rb-confirming")
    page = ["shep-page", "shep-train-page"] + (["shep-train-drawer-open"] if mode.get("drawer") else [])
    pieces = {
        "ws": ws,
        "dataset": (tuple(tuple(c) for c in ws.get("choices") or []), ws.get("name"), not in_use),
        "chip": chip_html(ws, in_use=in_use),
        "pill": pill_html(status),
        "reason": f'<div class="shep-rb-reason">{bar}{view["reason"]}</div>',
        "start": view["start"], "cont": view["cont"], "stop": view["stop"],
        "resume": view["resume"],
        "runbar": tuple(classes),
        "page": tuple(page),
        "run_name": bool(access.allowed),
        "banner": banner_html or "",
        "head": run_head_html(run, ws, operator=access.allowed, snapshot=snapshot,
                              live=in_use and run is not None and not run.get("latest"),
                              continuing=((mode.get("parent") or {}).get("run_label")
                                          if mode.get("drawer") else None)),
        "snap": run_snapshot_html(run, status, operator=access.allowed, snapshot=snapshot),
        "charts": _chart_token(run, cache, live=in_use),
        "drawer_reason": (f'<div class="shep-drawer-why">{view["drawer_reason"]}</div>'
                          if view["drawer_reason"] else ""),
        "training_workspace": ws.get("name"),
    }
    return {"pieces": pieces, "cache": cache, "ws_changed": ws_changed}


def _update_for(key: str, value: Any) -> Any:
    """A piece as the component's update."""
    if key == "dataset":
        choices, name, interactive = value
        return gr.update(choices=[tuple(c) for c in choices], value=name, interactive=interactive)
    if key in ("start", "cont", "stop", "resume"):
        return gr.update(interactive=bool(value))
    if key in ("runbar", "page"):
        return gr.update(elem_classes=list(value))
    if key == "run_name":
        return gr.update(interactive=bool(value))
    return value


def _comparable(value: Any) -> str:
    """A piece as the session's record of what it was sent keeps it."""
    return json.dumps(value, sort_keys=True, default=str)


def view_updates(rendered: Mapping[str, Any], cache_before: Optional[Mapping[str, Any]], *,
                 full: bool = False) -> Tuple[Any, ...]:
    """A queued render's outputs (`VIEW_KEYS`) and the cache: a piece that
    equals what this session was last sent is ``gr.update()`` (nothing sent),
    unless `full`."""
    sent = dict((cache_before or {}).get("sent") or {})
    pieces = rendered["pieces"]
    cache = dict(rendered["cache"])
    out = []
    for key in VIEW_KEYS:
        value = pieces[key]
        if key in SELECTION_KEYS:
            # The session's selection is written only when this render changed
            # it (a running job's dataset); a queued render's turn in the run
            # bar's workspace group keeps it from writing back an older one.
            out.append(value if rendered.get("ws_changed") else gr.update())
            continue
        comparable = _comparable(value)
        if not full and sent.get(key) == comparable:
            out.append(gr.update())
        else:
            out.append(_update_for(key, value))
            sent[key] = comparable
    cache["sent"] = sent
    return (*out, cache)


def _render_view(ws, mode, cache, dismissed, request: gr.Request):
    """Every action's last step: the run bar and the run view as they are now,
    whole (an action changed what they show)."""
    rendered = render(ws, mode, cache, dismissed, request)
    rendered["cache"]["handed"] = 0
    return view_updates(rendered, cache, full=True)


def _with_polled(cache: Optional[Mapping[str, Any]], polled: Optional[Mapping[str, Any]]
                 ) -> Dict[str, Any]:
    """The session's cache with each of the fast poll's own reads kept by
    time (`POLL_READS`) in place of the last render's only when it is newer:
    the lock's (``[when, lock]``); the latest run's (``[workspace, when,
    run]``) when it is of the same workspace too (another workspace's is not
    comparable, and the render's stays: `_latest_of` reads a workspace the
    renders have not drawn yet from its selection); and either when the
    render kept none. So neither a tick nor the render it hands over to
    draws an older read than the last render drew. The render after an
    action reads the lock or the runs again when its own read is too old,
    while the tick's own may still be young enough to keep and be from
    before a CLI took the GPU or a run was written."""
    merged = dict(cache or {})
    own = polled or {}
    lock, held = own.get("lock"), merged.get("lock")
    if lock and (not held or lock[0] > held[0]):
        merged["lock"] = lock
    latest, held = own.get("latest"), merged.get("latest")
    if latest and (not held or (latest[0] == held[0] and latest[1] > held[1])):
        merged["latest"] = latest
    return merged


def _render_changed(ws, mode, cache, dismissed, polled, request: gr.Request):
    """The render the fast poll hands a change over to (`CHANGED_ID`), at its
    turn in the run bar's workspace group: the run bar and the run view from
    the session's selection and the job as they are now, what changed since
    the session was last sent them, and everything every `FULL_SEND_TICKS`
    hand-offs. The reads kept by time (`POLL_READS`) are the newer of the
    poll's own and the last render's (`_with_polled`), so it draws what the
    poll found, never an older read than the last render drew."""
    rendered = render(ws, mode, _with_polled(cache, polled), dismissed, request)
    handed = int((cache or {}).get("handed") or 0) + 1
    rendered["cache"]["handed"] = handed
    return view_updates(rendered, cache, full=handed % FULL_SEND_TICKS == 0)


def _poll(location, ws, mode, cache, dismissed, polled, request: gr.Request):
    """The shell's fast timer (``queue=False``), while Training is shown: the
    status pill (`POLL_KEYS`) on every tick, the header's chips on the tick
    that sees the GPU slot change (`chips_on_change`), the hand-off, and its
    own record (``polled``: its reads kept by time and the GPU slot it saw);
    nothing while another page is shown.

    It writes no piece drawn from the session's selection, and no cache (a
    tick's answer is applied whenever it arrives, also after a render that
    read a newer selection): it renders the page as it is now and compares
    it with what the session was last sent, the cache's record, which only
    the renders write. When a piece of `HANDED_KEYS` is not what was sent,
    the GPU slot is not what the last render saw, or a training job runs in
    another workspace than the session's (``ws_changed``), it gives the
    hand-off token (`CHANGED_ID`) a new value, naming what it found, and the
    queued render that token's change runs reads the session's selection
    and the job again at its own turn.

    The reads a render keeps by a file's change (a run's log rows and
    record, the runner's workspace) are the last render's; those it keeps by
    time (`POLL_READS`) are the poll's own, which only it writes, so it
    reads them no more often than a render does; it and the render it hands
    over to take each where it is newer than the last render's
    (`_with_polled`)."""
    idle = (gr.update(),) * 4
    if location != TRAINING:
        return idle
    before = cache or {}
    try:
        rendered = render(ws, mode, _with_polled(before, polled), dismissed, request)
    except Exception:  # noqa: BLE001 - a tick that raises would stop the page's updates
        logger.exception("The Training page could not be read")
        return idle
    sent = before.get("sent") or {}
    pieces = rendered["pieces"]
    changed = [key for key in HANDED_KEYS if sent.get(key) != _comparable(pieces[key])]
    seen = rendered["cache"].get("gpu_seen")
    if seen != before.get("gpu_seen"):
        changed.append("gpu")
    if rendered["ws_changed"]:
        changed.append("ws")
    # The chips: read again on the tick that sees the GPU slot change.
    chips = chips_on_change((polled or {}).get("gpu_seen"), seen, request)
    # Each hand-off is a new value (its own time), so its .change always fires.
    handoff = {"changed": changed, "at": time.monotonic_ns()} if changed else gr.update()
    own = {key: rendered["cache"][key] for key in POLL_READS if key in rendered["cache"]}
    own["gpu_seen"] = seen
    return pieces["pill"], handoff, chips, own


# =============================================================================
# The handlers
# =============================================================================
def _serving_name(rows: Sequence[Mapping[str, Any]], directories: Mapping[str, Path]
                  ) -> Optional[str]:
    """The listed workspace the served pipeline reads, or None (the status
    read in process, through the shared reader)."""
    from src.webui.components.served_status import read_status, serving_workspace

    try:
        status, _problem = read_status()
        return serving_workspace(status, rows, directories=directories) if status else None
    except Exception:  # noqa: BLE001 - a default, never a reason to fail the page
        logger.debug("The serving workspace could not be read", exc_info=True)
        return None


def _open(ws, preferred):
    """The page shown: the server's list read again (once), the run bar's
    workspace chosen: Datasets' "Use for training", else this session's own,
    else the running job's, else the one serving reads, else the first
    trainable; the cache emptied so the next tick sends everything; the
    resources read."""
    status = training_manager.get_status()
    cache: Dict[str, Any] = {}
    runner = _runner_run(status, cache)
    listed = read_workspaces()
    rows = listed["rows"]
    names = [x["name"] for x in rows]
    candidates = (preferred, (ws or {}).get("name"), runner and runner.get("workspace"))
    choice = next((n for n in candidates if n and n in names), None)
    if choice is None:
        serving = _serving_name(rows, listed["directories"])
        trainable = [x["name"] for x in rows if x.get("trainable")]
        choice = (serving if serving in trainable else
                  trainable[0] if trainable else names[0] if names else None)
    state = workspace_state(choice, listed)
    return (state, dataset_choices_update(state), chip_html(state), choice, {},
            _poll_resources(TRAINING))


def _on_dataset(name, mode, request: gr.Request):
    """A dataset chosen in the run bar: its manifest and record read now (the
    chip shows them), the session's run-bar workspace set (Datasets marks
    it), any check of a parent cleared (the drawer lists the new workspace's
    runs on its next refresh)."""
    state = workspace_state(name)
    mode = dict(mode or {})
    if mode.get("parent"):
        mode["parent"] = {}
    return state, chip_html(state), state.get("name"), mode, ""


def _on_training_workspace(name, ws, mode):
    """Datasets' "Use for training ›" (``ctx.training_workspace``): the run
    bar selects that workspace (nothing changes on the server)."""
    if not name or name == (ws or {}).get("name"):
        return gr.update(), gr.update(), gr.update(), gr.update()
    state = workspace_state(name)
    if state.get("name") is None:
        return gr.update(), gr.update(), gr.update(), gr.update()
    mode = dict(mode or {})
    mode["parent"] = {}
    return state, dataset_choices_update(state), chip_html(state), mode


def event_act(evt: Any) -> Optional[str]:
    """The ``act`` a page element's trigger sent (`TRIGGER_JS`, `_POPOVER_JS`)."""
    data = getattr(evt, "_data", None)
    return data.get("act") if isinstance(data, Mapping) else None


def notice_html(text: Any) -> str:
    """The run bar's notice (an HTML component): the one place a handler's
    sentence becomes markup. The sentence carries server text (a workspace or
    run name, a path, the job runner's error), so it is escaped whole; ``""``
    for no notice."""
    said = "" if text is None else str(text)
    return f'<span class="shep-notice-t">{r.e(said)}</span>' if said else ""


def _on_banner_click(dismissed, cache, evt: gr.EventData):
    """Dismiss: this session hides this banner until its content changes."""
    key = (cache or {}).get("banner_key")
    if event_act(evt) != "dismiss" or not key:
        return gr.update(), gr.update()
    return [*(dismissed or []), key][-20:], ""


def _workspace_path(name: Any) -> Path:
    """The directory of the workspace the server lists as `name` (found in its
    list again: `server_lists.resolve_workspace`)."""
    from src.api.services.server_lists import resolve_workspace

    return resolve_workspace(name)


def _on_start(request: gr.Request, ws, run_name, device, *values):
    """Start: an operator control (D8), checked first on this event's own
    request; then a new base run (`_start`). ``values`` are the base form's,
    in `BASE_FIELDS` order.

    Returns the notice (`notice_html`): the refusal, inline, or nothing once
    started.
    """
    access = operator_access(request)
    if not access.allowed:
        return notice_html(access.refusal)
    return notice_html(_start(ws, run_name, device, values))


def _start(ws, run_name, device, values) -> str:
    """Start's work after D8, as a plain sentence ("" once started): **a new
    base run** (K6 decision 12) on the workspace the server resolves from its
    list, through the job runner's one start (`start_training`) with
    ``learning_stage="base"``, the run keys, and only the settings that differ
    from the recipe's (`recipe_differences`), with the full dataset
    fingerprint this session showed. The runner checks the settings again
    (`_new_bundle_run`) and refuses what the recipe does not take."""
    row = (ws or {}).get("row") or {}
    fingerprint = workspace_fingerprint(ws)
    if not row or not fingerprint:
        return "Not started: select a dataset from the server's list first."
    if not row.get("trainable"):
        return f"Not started: {row.get('name')} is not trainable: {r.not_trainable_short(row)}."
    settings, problems = recipe_differences(dict(zip(BASE_FIELDS, values)), BASE)
    if problems:
        return "Not started: " + " ".join(problems)
    from src.api.services.server_lists import ServerListRefused

    try:
        directory = _workspace_path(row["name"])
    except ServerListRefused:
        return (f"Not started: {row['name']} is no longer listed by the server; select a dataset "
                "again.")
    config = {"data_dir": str(directory), "learning_stage": "base",
              "device": device or "auto", "run_name": (run_name or "").strip(),
              # torch.compile is a runtime setting (Settings), read once here.
              "compile": bool(load_runtime_settings().get("torch_compile", False)),
              **settings}
    result = training_manager.start_training(config, expected_dataset_fingerprint=fingerprint)
    if result.get("success"):
        return ""
    return f"Not started: {result.get('error', 'unknown error')}"


def running_job(status: Mapping[str, Any]) -> Optional[str]:
    """The id of the training job running now (the one Stop would stop), or
    None when none is running."""
    if (status.get("status") or "idle") != "running":
        return None
    return str(status.get("job_id") or "running")


def confirming(status: Mapping[str, Any], mode: Mapping[str, Any]) -> bool:
    """Whether the run bar shows Stop's confirmation: only for the job it was
    asked for, while that job runs. A confirmation left unanswered when its
    job ended (completed, failed, or stopped elsewhere) is never shown for, or
    applied to, the next job."""
    asked = mode.get("confirm")
    return isinstance(asked, str) and asked == running_job(status)


#: Stop's question. It names no epoch: the job goes on training while the
#: question is open, so an epoch read when it was asked would be out of date
#: by the answer (the GB10 acceptance: asked at epoch 183, at 186 on screen,
#: 195 saved). What can be continued is said once the job has stopped, from
#: the run's own checkpoint (the snapshot sentence).
STOP_QUESTION = ("<b>Stop training now?</b> The job stops. The checkpoints it has saved are "
                 "kept and progress not yet saved is lost; if it has saved none yet, this run "
                 "cannot be continued. Once it has stopped, the page says where it can be "
                 "continued from.")


def stop_question(status: Mapping[str, Any]) -> str:
    """Stop's question (`STOP_QUESTION`, HTML), and, when the job is a
    Fine-tune child with children queued after it, that those will not start
    (a stop cancels them: `TrainingManager._stop`). Read from the status the
    question is asked with; nothing else is read."""
    finetune = status.get("finetune") or {}
    queued = [TASK_NAMES.get(child.get("task"), str(child.get("task")))
              for child in finetune.get("children") or [] if child.get("status") == "queued"]
    if not queued:
        return STOP_QUESTION
    return (f"{STOP_QUESTION} The queued Fine-tune children ({r.e(', '.join(queued))}) will "
            "not start.")


def ask_stop(access: Any, mode: Optional[Mapping[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """**Stop's first step, the one for every page** (the Training run bar,
    the Fine-tune job card): for a request D8 allowed (`access`, read by the
    page's own handler on its own request), ask before anything happens
    (`stop_question`), reading nothing but the job's status. The
    confirmation is recorded for the job running now (its id).

    Returns the confirmation's text and the session's mode.
    """
    mode = dict(mode or {})
    mode["confirm"] = None
    if not access.allowed:
        return notice_html(access.refusal), mode
    status = training_manager.get_status()
    job = running_job(status)
    if job is None:
        return notice_html("Nothing to stop: no training job is running."), mode
    mode["confirm"] = job
    return f'<div class="shep-rb-confirm-q">{stop_question(status)}</div>', mode


def confirm_stop(access: Any, mode: Optional[Mapping[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """**The confirmed Stop, the one for every page**: for a request D8
    allowed (`access`; refused, nothing is stopped). A confirmation asked for
    another job than the one running now stops nothing (it described that
    job). The notice names the queued Fine-tune children the stop cancelled.

    Returns the notice and the session's mode (the confirmation closed).
    """
    mode = dict(mode or {})
    asked = mode.get("confirm")
    mode["confirm"] = None
    if not access.allowed:
        return notice_html(access.refusal), mode
    if isinstance(asked, str) and asked != running_job(training_manager.get_status()):
        return notice_html("Not stopped: the training job you asked to stop is no longer the "
                           "one running; click Stop again if this one should stop."), mode
    result = training_manager.stop_training()
    if result.get("success"):
        cancelled = [TASK_NAMES.get(task, str(task)) for task in result.get("cancelled_children")
                     or ()]
        tail = (f" The queued Fine-tune children ({', '.join(cancelled)}) were cancelled."
                if cancelled else "")
        return notice_html("Stopped. " + _snapshot_sentence(result.get("outcome")) + tail), mode
    return notice_html(f"Not stopped: {result.get('error', 'unknown error')}"), mode


def _on_stop_request(request: gr.Request, mode):
    """Stop, on the run bar: D8 checked first, on this event's own request,
    as for the Stop itself; then the question (`ask_stop`)."""
    return ask_stop(operator_access(request), mode)


def close_stop_question(mode):
    """Keep training (and any Start: a confirmation left from an earlier job
    is closed)."""
    mode = dict(mode or {})
    mode["confirm"] = None
    return mode


def _on_stop(request: gr.Request, mode):
    """The confirmed Stop, on the run bar: an operator control (D8), checked
    on this event's own request (`confirm_stop`)."""
    return confirm_stop(operator_access(request), mode)


# --------------------------------------------------------------------------- Continue…
#: The two sources of a parent checkpoint. Both go through the same check.
CONTINUE_FROM_LIST = "From run list"
CONTINUE_FROM_PATH = "Manual path"

#: The continue block's prompt before a check, and after the selection changes.
_CONTINUE_PROMPT = "Select a checkpoint, then click <b>Check checkpoint</b>."


#: What a checkpoint file holds, as its record says (`checkpoint_kind`).
CHECKPOINT_KINDS = {"training_state": "training state", "weights": "weights only"}
#: A file the trainer keeps as one of its top-k by the validation metric
#: (`ModelCheckpoint`'s ``model-{epoch:02d}-{metric:.4f}.pt``).
_TOP_K_FILE = re.compile(r"^model-\d+-\d+(?:\.\d+)?\.pt$")
#: The longest option a select offers (the WebUI skill §8).
OPTION_CHARS = 40


def run_words(architecture: Any, run: Any) -> str:
    """A run as a person names it: its name (when it was given one), the
    architecture and its start time, "ui-live-stop-02 · GAT · 2026-10-24 17:39"
    from ``20261024-173909_ui-live-stop-02_c1d9cf``; an unnamed run ("run") is
    its architecture and time; a directory not named by its start time is its
    whole name."""
    arch = r.architecture_label(architecture)
    started = r.run_started(run)
    if not started:
        return f"{arch} · {run}"
    rest = r.run_suffix(run) or ""
    name = re.sub(r"_[0-9a-f]{6}$", "", rest)
    return f"{name} · {arch} · {started}" if name and name != "run" else f"{arch} · {started}"


def _run_key(item: Mapping[str, Any]) -> str:
    return f"{item.get('architecture')}/{item.get('run')}"


def runs_of(parents: Mapping[str, Any]) -> Dict[str, List[str]]:
    """The listed checkpoints by run, in the list's order (latest run first,
    newest file first): ``{"<architecture>/<run>": [key, ...]}``."""
    runs: Dict[str, List[str]] = {}
    for key, item in parents.items():
        runs.setdefault(_run_key(item), []).append(key)
    return runs


def _progress_words(item: Mapping[str, Any]) -> Optional[str]:
    """"195 epochs completed" from the file's progress record; a file with no
    progress record says the epoch index it records, never a count made from
    it; None when it records neither."""
    completed, epoch = item.get("completed_epochs"), item.get("epoch")
    if isinstance(completed, int) and not isinstance(completed, bool):
        return f"{completed} epoch{'s' if completed != 1 else ''} completed"
    if isinstance(epoch, int) and not isinstance(epoch, bool):
        return f"recorded epoch {epoch}"
    return None


def validation_words(item: Mapping[str, Any]) -> Optional[str]:
    """"Validation MRR (Trainer, top-20) 0.2867": the Trainer's validation
    figure the file records, in the dashboard's one map of their names
    (`renderers.VALIDATION_LABELS`), never a test result."""
    validation = item.get("validation") or {}
    metric, value = validation.get("metric"), validation.get("value")
    if metric not in r.VALIDATION_LABELS or not isinstance(value, (int, float)):
        return None
    return f"{r.VALIDATION_LABELS[metric]} {value:.4f}"


def _fit(what: str, rest: Optional[str]) -> str:
    """"<what> · <rest>" in at most `OPTION_CHARS` characters with one
    separator: `what` (a file's name) shortened with an ellipsis to fit."""
    room = OPTION_CHARS - (len(rest) + 3 if rest else 0)
    if len(what) > room:
        what = what[:max(room - 1, 1)] + "…"
    return f"{what} · {rest}" if rest else what


def checkpoint_words(item: Mapping[str, Any]) -> str:
    """A checkpoint's option in its run, in a few words (the WebUI skill §8:
    at most `OPTION_CHARS` characters, one separator): what it is and how far
    it trained, "Latest save · 195 epochs completed", "Top-k save · 21 epochs
    completed"; a file that holds weights only, or cannot be read, says so.
    How its stage ended, its stage, its Trainer validation figure, what it
    holds and its model fingerprint are the chosen file's details
    (`parent_details_html`)."""
    filename = str(item.get("filename") or "")
    if item.get("problem") or not r.is_digest(item.get("sha256")):
        return _fit(filename, "cannot be read")
    what = ("Latest save" if filename == "last.pt" else
            "Top-k save" if _TOP_K_FILE.match(filename) else filename)
    if item.get("checkpoint_kind") == "weights":
        return _fit(what, "weights only")
    return _fit(what, _progress_words(item))


def _choices(labels: Sequence[Tuple[str, str]], fallback: Mapping[str, str]) -> List[Tuple[str, str]]:
    """Options whose label another option also has get their own distinct
    words (`fallback`), so no two options read the same."""
    seen: Dict[str, int] = {}
    for label, _key in labels:
        seen[label] = seen.get(label, 0) + 1
    return [(f"{label} · {fallback[key]}" if seen[label] > 1 else label, key)
            for label, key in labels]


def run_choices(parents: Mapping[str, Any]) -> List[Tuple[str, str]]:
    runs = runs_of(parents)
    first = {run: parents[keys[0]] for run, keys in runs.items()}
    return _choices([(run_words(item.get("architecture"), item.get("run")), run)
                     for run, item in first.items()],
                    {run: str(item.get("run")) for run, item in first.items()})


def checkpoint_choices(parents: Mapping[str, Any], keys: Sequence[str]) -> List[Tuple[str, str]]:
    """A run's checkpoints as options: their words (`checkpoint_words`); two
    that would read alike are named by their file instead (unique in a run),
    still one separator and `OPTION_CHARS` at most."""
    labels = [(checkpoint_words(parents[key]), key) for key in keys]
    seen = Counter(label for label, _key in labels)
    return [(label if seen[label] == 1 else
             _fit(str(parents[key].get("filename")), _progress_words(parents[key])), key)
            for label, key in labels]


def checkpoint_items(listing: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """A checkpoint listing (`server_lists.list_checkpoints`) as the session
    keeps it: ``({"<architecture>/<run>/<file>": item}, problem)``, the
    problem redacted (no absolute path)."""
    from dataclasses import asdict

    if not listing.available:
        return {}, redact(listing.problem, workspaces_root()[0])
    return {f"{item.architecture}/{item.run}/{item.filename}": asdict(item)
            for item in listing.items}, None


def _list_parents(ws: Mapping[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """The run bar's workspace's model records from the server's list (it
    opens every checkpoint: the caller checked D8): ``({key: item},
    problem)``. **Only a model record is offered** (K6 decision 1): a file
    without a model spec, or one whose record cannot be read, is never
    continued from the dashboard; Research › Models lists it."""
    from src.api.services.server_lists import list_checkpoints

    name = ws.get("name")
    if not name:
        return {}, "Select a dataset in the run bar first."
    listed, problem = checkpoint_items(list_checkpoints(name))
    if problem:
        return {}, problem
    parents = {key: item for key, item in listed.items() if item.get("record") == "model"}
    if listed and not parents:
        return {}, (f"None of the {len(listed)} checkpoint files of this dataset's runs holds a "
                    "model record; a file without a model spec is never continued here "
                    "(Research › Models lists it).")
    return parents, None


def parents_update(parents: Mapping[str, Any], keep: Optional[str] = None) -> Tuple[Any, Any]:
    """The two selects for `parents`: the runs, and the chosen run's
    checkpoints. `keep` (a checkpoint's key) stays chosen while it is listed;
    otherwise the first run and its first checkpoint (its latest save)."""
    runs = runs_of(parents)
    run = _run_key(parents[keep]) if keep in parents else next(iter(runs), None)
    keys = runs.get(run, []) if run else []
    value = keep if keep in keys else (keys[0] if keys else None)
    return (gr.update(choices=run_choices(parents), value=run),
            gr.update(choices=checkpoint_choices(parents, keys), value=value))


def on_run_chosen(run, parents, current):
    """A run chosen: its checkpoints, the latest save first. The checkpoint
    chosen stays when it is this run's (opening the drawer sets both)."""
    keys = runs_of(parents or {}).get(run or "", [])
    value = current if current in keys else (keys[0] if keys else None)
    return gr.update(choices=checkpoint_choices(parents or {}, keys), value=value)


def _parents_note(ws: Mapping[str, Any], parents: Mapping[str, Any], problem: Optional[str]) -> str:
    if problem:
        return f'<span class="shep-muted">{r.e(problem)}</span>'
    runs = len(runs_of(parents))
    return (f'<span class="shep-muted">{LOCK_ICON} server list · {runs} run{"s" if runs != 1 else ""}, '
            f'{len(parents)} checkpoint{"s" if len(parents) != 1 else ""} · latest first</span>')


def stage_words(item: Mapping[str, Any]) -> str:
    """A listed model's stage, in its record's words: "Base", or "Fine-tune ·
    Gene" (`renderers.record_kind_text`, built on `caveats.TASK_NAMES`)."""
    kind = r.record_kind_text(item)
    return f"Fine-tune · {kind}" if item.get("learning_stage") == "finetune" else kind


def parent_details_html(item: Optional[Mapping[str, Any]]) -> str:
    """The chosen checkpoint's details, as a two-column table, saying what its
    option does not (`checkpoint_words` gives what it is and its progress):
    its file, its stage, what it holds, how its stage ended where that limits
    continuing, its Trainer validation figure (never a test result), its model
    fingerprint (copyable) and its run's directory. A file that cannot be read
    says why, in amber. Empty when none is chosen from the list."""
    if not item:
        return ""
    filename = str(item.get("filename") or "")
    rows = [("File", f"<code>{r.e(filename)}</code>")]
    if item.get("problem") or not r.is_digest(item.get("sha256")):
        why = item.get("problem") or "it records no metadata"
        return (f'<div class="shep-ckpt shep-ckpt-amber"><div class="shep-kv">'
                f'<span class="shep-k">File</span><span class="shep-v"><code>{r.e(filename)}</code></span>'
                f'<span class="shep-k">Cannot be read</span><span class="shep-v">{r.e(why)}</span>'
                '</div></div>')
    rows.append(("Stage", r.e(stage_words(item))))
    kind = item.get("checkpoint_kind")
    rows.append(("Holds", "weights only (cannot be continued)" if kind == "weights" else
                 r.e(CHECKPOINT_KINDS.get(kind or "", "not recorded"))))
    going_on = {"finished": "extends it by the Extra epochs you enter",
                "stopped_early": "not possible: it stopped early"}.get(item.get("end") or "")
    if going_on and kind != "weights":
        rows.append(("Continuing", r.e(going_on)))
    figure = validation_words(item)
    rows.append(("Validation", r.e(figure.removeprefix("Validation ") if figure
                                   else "not recorded in this file")))
    rows.append(("Model fingerprint", digest_html(item["sha256"], what="model fingerprint")))
    rows.append(("Run", f"<code>{r.e(_run_key(item))}</code>"))
    cells = "".join(f'<span class="shep-k">{k}</span><span class="shep-v">{v}</span>' for k, v in rows)
    return f'<div class="shep-ckpt"><div class="shep-kv">{cells}</div></div>'


def _on_continue_open(request: gr.Request, ws, mode):
    """Continue…: the drawer over the parameters, listing the run bar's
    workspace's checkpoints (an operator control: D8 first; refused, the
    drawer stays closed and nothing is opened).

    Returns the notice, the session's mode, the parents (state), the run and
    checkpoint selects, their note, and the check's text."""
    access = operator_access(request)
    mode = dict(mode or {})
    if not access.allowed:
        return (notice_html(access.refusal), mode, gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update())
    parents, problem = _list_parents(ws or {})
    mode.update(drawer=True, confirm=None, parent={})
    runs, checkpoints = parents_update(parents)
    first = checkpoints.get("value") if isinstance(checkpoints, dict) else None
    mode["parent"] = _chosen(CONTINUE_FROM_LIST, first, "", parents, ws)
    return ("", mode, parents, runs, checkpoints, _parents_note(ws or {}, parents, problem),
            _CONTINUE_PROMPT)


def _refresh_checkpoints(request: gr.Request, ws, mode, current):
    """Refresh the parent list: the run bar's workspace's checkpoints, again
    from the server's list. It opens every checkpoint (SPEC_4 C3), so D8 is
    checked first, on this event's own request, as for the checkpoint
    routes; refused, the list is emptied, nothing is opened, and the note says
    why."""
    access = operator_access(request)
    return _parents_listed(access, ws, current)


def _parents_listed(access: Any, ws: Optional[Mapping[str, Any]], current: Any):
    """The parent list for `ws` as D8 allows it: ``(parents, run select,
    checkpoint select, note)``."""
    if not access.allowed:
        return ({}, gr.update(choices=[], value=None), gr.update(choices=[], value=None),
                f'<span class="shep-muted">{r.e(access.refusal)}</span>')
    parents, problem = _list_parents(ws or {})
    return (parents, *parents_update(parents, current),
            _parents_note(ws or {}, parents, problem))


def _relist_parents(request: gr.Request, ws, mode):
    """The run bar's workspace changed (chosen by hand, by Datasets' "Use for
    training ›", or followed to a job running in another workspace) while
    Continue… is open: the parent list becomes the new workspace's runs,
    through the Refresh's path (it opens every checkpoint, so D8 first, on
    this event's own request; refused, the list is emptied and the note says
    why), and the last check, the comparison and the Extra epochs, which were
    about the old list, are cleared. With the drawer closed nothing is listed
    (it lists when opened).

    Returns the parents (state), the run and checkpoint selects, their note,
    the comparison, the check's text and the Extra epochs field."""
    access = operator_access(request)
    if not (mode or {}).get("drawer"):
        return (gr.update(),) * 7
    parents, runs, checkpoints, note = _parents_listed(access, ws, None)
    return (parents, runs, checkpoints, note, "",
            f'<p class="shep-note">{_CONTINUE_PROMPT}</p>', gr.update(visible=False, value=None))


def _on_continue_close(mode):
    mode = dict(mode or {})
    mode.update(drawer=False, parent={})
    return mode


def _chosen(source: str, listed: Any, manual: Any, parents: Mapping[str, Any],
            ws: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The chosen parent as the session records it: whether one is chosen,
    its recorded dataset fingerprint when the list says it, the model
    fingerprint shown for it (``sha256``: the list's; none yet for a typed
    path, whose Check shows one, `_checked`), and the run it continues (for
    the run bar's line)."""
    if source == CONTINUE_FROM_PATH:
        path = _strip_display_prefix(manual or "")
        return {"chosen": bool(path), "source": source, "fingerprint": None,
                "run_label": None, "sha256": None}
    item = (parents or {}).get(listed) if listed else None
    if not item:
        return {"chosen": False, "source": source}
    return {"chosen": True, "source": source, "fingerprint": item.get("dataset_fingerprint"),
            "run_label": _run_key(item), "key": listed,
            "workspace": item.get("workspace"), "sha256": item.get("sha256")}


def compare_html(parent: Mapping[str, Any], ws: Mapping[str, Any]) -> str:
    """The parent's recorded dataset fingerprint against the workspace's
    current one: ✓ the same, amber when it differs ("matches no listed
    workspace", or the listed workspace it matches), or what is not known yet."""
    if not parent.get("chosen"):
        return ""
    recorded, current = parent.get("fingerprint"), workspace_fingerprint(ws)
    digits = ws.get("digits") or r.SHORT_HEX
    if not r.is_digest(recorded):
        why = ("The parent’s recorded dataset is read when you check it."
               if parent.get("source") == CONTINUE_FROM_PATH and not parent.get("checked")
               else "The parent records no dataset fingerprint; the start checks it against "
                    "this workspace’s data.")
        return f'<div class="shep-cmp shep-cmp-grey">{r.e(why)}</div>'
    now = (f'this workspace now: <b>{r.e(ws.get("name"))}</b> · '
           + r.fingerprint_html("dataset fingerprint", current, digits=digits, copy=False))
    if current and recorded == current:
        # Said once, as a relation: the fingerprint itself is the run bar's.
        return (f'<div class="shep-cmp shep-cmp-ok">{r.CHECK} Trained on the dataset selected '
                'in the run bar</div>')
    listed = ws.get("listed") or []
    trained = r.trained_on_html(recorded, listed, digits=digits)
    built = r.built_text((ws.get("row") or {}).get("build"))
    rebuilt = (f"<p>It was {r.e(built)}. The start refuses a different dataset.</p>"
               if built else "<p>The start refuses a different dataset.</p>")
    return ('<div class="shep-cmp shep-cmp-amber"><b>⚠ The parent’s dataset is not this '
            f'workspace’s current one</b><p>parent {trained}</p><p>{now}</p>{rebuilt}</div>')


def _check_html(preview: Mapping[str, Any], source: str, *, differs: bool = False,
                listed: Optional[str] = None) -> str:
    """The check result, what continuing does, and the settings taken from the
    checkpoint (a closed disclosure), as a read-only summary (contract M3a-3;
    not disabled inputs, which screen readers skip)."""
    # The file read: named in full for a typed path; for a listed one by its
    # key in the server's list (`listed`, "<architecture>/<run>/<file>"), which
    # the server resolved to the file it read, its details shown above.
    if source == CONTINUE_FROM_PATH or not listed:
        lines = [f'<div><b>Checkpoint</b> (manual path): <code>{r.e(_shown_path(preview.get("parent", "")))}</code></div>'
                 if source == CONTINUE_FROM_PATH else
                 f'<div><b>Checkpoint</b> (run list): <code>{r.e(_shown_path(preview.get("parent", "")))}</code></div>']
    else:
        lines = [f'<div><b>Checked</b> <code>{r.e(listed)}</code> from the run list</div>']
    if preview.get("sha256") and source == CONTINUE_FROM_PATH:
        arch = preview.get("architecture")
        lines.append('<div>' + r.fingerprint_html("model fingerprint", preview["sha256"], copy=False)
                     + (f", architecture <b>{r.e(r.architecture_label(arch))}</b>" if arch else "") + "</div>")
    if not preview.get("ok"):
        return ('<div class="shep-check shep-check-amber"><div class="shep-check-h">⚠ Cannot '
                f'continue.</div>{"".join(lines)}<p>{r.e(preview.get("reason"))}</p></div>')
    # The plan is said once, in the line under the headline (M5-3: the
    # headline repeated it, two lines more in a drawer that must fit).
    head = ('<div class="shep-check-h">⚠ Checkpoint read; cannot continue on the current '
            'dataset.</div>' if differs else
            f'<div class="shep-check-h shep-ok-t">{r.CHECK} Checkpoint read; it can continue '
            'on this workspace.</div>')
    # A rate such as 3e-05 is never split at its hyphen.
    action = re.sub(r"(\d(?:\.\d+)?e-\d+)", r'<span class="shep-nowrap">\1</span>',
                    r.e(preview["action"]))
    lines.append(f"<p>Plan read from the checkpoint: {action}.</p>")
    inherited = preview.get("inherited") or {}
    if inherited:
        rows = "".join(f"<tr><td>{r.e(k)}</td><td><code>{r.e(repr(inherited[k]))}</code></td></tr>"
                       for k in sorted(inherited))
        lines.append(
            f'<details class="shep-inherit"><summary>Settings taken from this checkpoint '
            f'<span class="shep-muted">({len(inherited)}; fixed when continuing)</span></summary>'
            '<p class="shep-note">They cannot be changed when continuing; to train with other '
            'settings, start a new run.</p>'
            f'<table class="shep-ftbl"><thead><tr><th>Setting</th><th>Value</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></details>')
    if not differs:
        # When the datasets differ, the comparison above already says the
        # start refuses (M5-3: one line less, so the drawer fits).
        lines.append('<p class="shep-note">The start checks again against this workspace’s '
                     'data and graph.</p>')
    kind = "amber" if differs else "ok"
    return f'<div class="shep-check shep-check-{kind}">{head}{"".join(lines)}</div>'


def _extra_epochs_update(preview: Mapping[str, Any]):
    """The Extra epochs field belongs to a finished checkpoint: shown for one,
    and hidden and cleared for any other, so no value is left behind for a
    checkpoint it does not apply to (contract M3a review of `094b90d`, P2-2)."""
    if preview.get("end") == "finished":
        return gr.update(visible=True)
    return gr.update(visible=False, value=None)


def _on_continue_selection_changed():
    """The chosen checkpoint, or where it comes from, changed: the last check
    and the Extra epochs entered for it no longer apply, and are cleared."""
    return gr.update(visible=False, value=None), _CONTINUE_PROMPT


def _on_listed_changed(source):
    """The run list changed. It is the chosen checkpoint only when the source
    is the run list: a Refresh repopulates it even while a typed path is the
    source, and must not clear that path's check."""
    if source == CONTINUE_FROM_PATH:
        return gr.update(), gr.update()
    return _on_continue_selection_changed()


def _on_manual_changed(source):
    """The typed path changed; it is the chosen checkpoint only when the
    source is the typed path."""
    if source != CONTINUE_FROM_PATH:
        return gr.update(), gr.update()
    return _on_continue_selection_changed()


def _continue_device(device):
    """Continue…'s Device starts from the Parameters' each time it opens."""
    return device if device in ("auto", "cuda", "cpu") else "auto"


def _on_continue_device_changed():
    """The device is part of the check (mixed precision takes effect only on
    CUDA): the last check no longer describes what Continue will do. The Extra
    epochs entered stay; they belong to the checkpoint, not the device."""
    return _CONTINUE_PROMPT


def _on_parent_chosen(source, listed, manual, parents, ws, mode):
    """The session records the chosen parent (for the run bar's actions and
    the dataset comparison), and the chosen listed checkpoint's details are
    shown (none for a typed path, whose Check shows what it read)."""
    mode = dict(mode or {})
    if not mode.get("drawer"):
        return gr.update(), gr.update(), gr.update()
    mode["parent"] = _chosen(source, listed, manual, parents, ws)
    item = (parents or {}).get(listed) if source != CONTINUE_FROM_PATH and listed else None
    return mode, compare_html(mode["parent"], ws or {}), parent_details_html(item)


def _strip_display_prefix(path: str) -> str:
    """Strip the UI display prefix (``SHEPHERD-Advanced/...`` -> ``...``)."""
    p = (path or "").strip()
    return p[len("SHEPHERD-Advanced/"):] if p.startswith("SHEPHERD-Advanced/") else p


def _parent_path(source: str, listed: Any, manual: Any, ws: Mapping[str, Any],
                 parents: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str],
                                                      Optional[Mapping[str, Any]]]:
    """``(path, why_not, listed_item)`` for the chosen parent, always one the
    server lists in the run bar's workspace and resolved from that list again
    (never a client path, `PLAN_TEST_RESULTS.md` D8, 2026-10-19): a listed one
    by its names; a typed one only locates a listed checkpoint
    (`server_lists.locate_checkpoint`), and any other file is refused unopened."""
    from src.api.services.server_lists import (
        ServerListRefused,
        locate_checkpoint,
        resolve_checkpoint,
    )

    if source == CONTINUE_FROM_PATH:
        typed = _strip_display_prefix(manual or "")
        if not typed:
            return None, "enter a checkpoint path", None
        if not ws.get("name"):
            return None, "select a dataset in the run bar first", None
        try:
            architecture, run, filename = locate_checkpoint(ws.get("name"), typed)
        except ServerListRefused as exc:
            return None, redact(exc, workspaces_root()[0]), None
        item = (parents or {}).get(f"{architecture}/{run}/{filename}")
        if not item:
            # Only a model record the run list offers is continued (K6
            # decision 1), so a file without a model spec is refused unopened.
            return None, ("the typed path is not a model record the run list offers (a file "
                          "without a model spec is never continued here); refresh the list if "
                          "its run is new"), None
    else:
        item = (parents or {}).get(listed) if listed else None
    if not item:
        return None, "choose a checkpoint from the run list", None
    other = _other_workspace(item, ws)
    if other:
        return None, other.rstrip("."), None

    try:
        path = resolve_checkpoint(ws.get("name"), item["architecture"], item["run"],
                                  item["filename"])
    except ServerListRefused as exc:
        return None, f"it is no longer listed ({redact(exc, workspaces_root()[0])}); refresh the list", None
    return str(path), None, item


def _preview_for(parent: str, device: Optional[str], extend_epochs,
                 expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    """The service's preview, with Extra epochs only for a finished
    checkpoint: the field is the console's, and for any other checkpoint a
    value in it is one left from an earlier selection, not a request. The API
    and the CLI send `extend_epochs` as given and are refused as before.
    `expected_sha256` is the model fingerprint shown, which the service
    compares on its read."""
    epochs = int(extend_epochs) if extend_epochs else None
    return training_manager.preview_continuation(
        parent, device=device or "auto", extend_epochs=epochs, extend_if_finished=True,
        expected_sha256=expected_sha256)


def _changed_parent_action(source: str) -> str:
    """What to do when the service refuses the parent as changed (its
    ``changed``), for where its fingerprint was shown: a listed parent's is
    the run list's, which choosing the same option sends again until the list
    is read again; a typed path's is its last Check's."""
    if source == CONTINUE_FROM_PATH:
        return "Click Check checkpoint to read the file there now."
    return "Refresh the list, then choose it again."


def _checked(source, listed, manual, device, extend_epochs, ws, parents, mode, *,
             starting: bool = False):
    """The check every Check and Continue Training runs: the parent resolved,
    the preview of the parent shown, the dataset compared. Returns (preview or
    None, why_not, the session's parent record).

    **The model fingerprint shown goes to the service's preview, which
    compares it on its read** (root-cause round, P22-S1; no comparison here):
    the run list's, which its option names, for a listed parent; for a typed
    path, the one its last Check read and showed for that same file, when
    Continue Training (`starting`) follows it (a Check is what shows it). The
    record keeps the digest shown, which Continue Training sends to the start.
    A refusal as changed gets the action for that source
    (`_changed_parent_action`)."""
    path, why, item = _parent_path(source, listed, manual, ws or {}, parents or {})
    record = dict((mode or {}).get("parent") or _chosen(source, listed, manual, parents, ws))
    if path is None:
        record.update(refused=True, checked=True)
        return None, why, record
    if source == CONTINUE_FROM_PATH:
        shown = record.get("sha256") if starting and record.get("path") == path else None
    else:
        shown = (item or {}).get("sha256")
    preview = _preview_for(path, device, extend_epochs, expected_sha256=shown)
    if preview.get("changed"):
        preview = {**preview, "reason": f"{preview['reason']} {_changed_parent_action(source)}"}
    recorded = preview.get("dataset_fingerprint") or (item or {}).get("dataset_fingerprint")
    record.update(chosen=True, checked=True, refused=not preview.get("ok"), fingerprint=recorded,
                  sha256=shown or preview.get("sha256"), path=path)
    return preview, None, record


def _on_check_continue(source, listed, manual, device, extend_epochs, ws, parents, mode,
                       request: gr.Request):
    """Check checkpoint: preview what continuing from the chosen checkpoint
    does. It opens the checkpoint (SPEC_4 C3), so D8 is checked first, on
    this event's own request; refused, nothing is opened.

    Returns the check's text, the extend-epochs input's update (shown only
    for a finished checkpoint, where extending is the only way on), the
    session's mode and the dataset comparison.
    """
    access = operator_access(request)
    if not access.allowed:
        return f'<div class="shep-reason">{r.e(access.refusal)}</div>', gr.update(), gr.update(), gr.update()
    mode = dict(mode or {})
    preview, why, record = _checked(source, listed, manual, device, extend_epochs, ws, parents, mode)
    mode["parent"] = record
    if preview is None:
        return (f'<div class="shep-reason">Nothing to check: {r.e(why)}.</div>',
                gr.update(visible=False, value=None), mode, compare_html(record, ws or {}))
    differs = continue_refusal(record, ws or {}) is not None and preview.get("ok")
    text = _check_html(preview, source, differs=bool(differs), listed=listed)
    if preview.get("end") == "finished" and not preview.get("ok") and not extend_epochs:
        text += '<p class="shep-note">Enter <b>Extra epochs</b> above, then click <b>Check checkpoint</b> again.</p>'
    return text, _extra_epochs_update(preview), mode, compare_html(record, ws or {})


def _on_resume(ws, device, source, listed, manual, extend_epochs, parents, mode,
               request: gr.Request):
    """Continue Training: an operator control (D8, checked first on this
    event's own request; refused, nothing is opened or started), then the same
    check as the preview, the dataset comparison, then the job runner's start
    with the full dataset fingerprint and the parent's model fingerprint this
    session showed.

    Only operational fields are sent (contract M3a-3): every learning setting
    comes from the checkpoint, so the training form's values are not used.
    The job runner and the training process each check again, the process on
    the parent the runner decided on (P22-S1).

    Returns the notice (`notice_html`), the check's text, the extend-epochs
    update and the session's mode (the drawer closed once started).
    """
    access = operator_access(request)
    if not access.allowed:
        return notice_html(access.refusal), gr.update(), gr.update(), gr.update()
    said, text, extend, mode = _resume(ws, device, source, listed, manual, extend_epochs,
                                       parents, mode)
    return notice_html(said), text, extend, mode


def _resume(ws, device, source, listed, manual, extend_epochs, parents, mode):
    """Continue Training's work after D8: ``(sentence, check text, Extra
    epochs update, mode)``, the sentence plain text ("" once started)."""
    mode = dict(mode or {})
    row = (ws or {}).get("row") or {}
    fingerprint = workspace_fingerprint(ws)
    if not row or not fingerprint:
        return "Not started: select a dataset first.", gr.update(), gr.update(), mode
    preview, why, record = _checked(source, listed, manual, device, extend_epochs, ws, parents,
                                    mode, starting=True)
    mode["parent"] = record
    if preview is None:
        return f"Not started: {why}.", gr.update(), gr.update(visible=False, value=None), mode
    refusal = continue_refusal(record, ws)
    text = _check_html(preview, source, differs=preview.get("ok") and refusal is not None,
                       listed=listed)
    if not preview.get("ok"):
        if preview.get("end") == "finished" and not extend_epochs:
            text += ('<p class="shep-note">Enter <b>Extra epochs</b> above, then click '
                     '<b>Continue Training</b> again.</p>')
        return ("Not started: this checkpoint cannot be continued; the reason is in the check.",
                text, _extra_epochs_update(preview), mode)
    if refusal:
        return f"Not started: {refusal}", text, _extra_epochs_update(preview), mode
    from src.api.services.server_lists import ServerListRefused

    try:
        directory = _workspace_path(row["name"])
    except ServerListRefused:
        return (f"Not started: {row['name']} is no longer listed by the server.", text,
                _extra_epochs_update(preview), mode)
    config: Dict[str, Any] = {
        "data_dir": str(directory),
        "run_name": "",
        "device": device or "auto",
        "resume_from": preview["parent"],
        "compile": bool(load_runtime_settings().get("torch_compile", False)),
    }
    if preview["mode"] == "extend":
        config["extend_epochs"] = preview["stage"]["target_epochs"]
    result = training_manager.start_training(config, expected_dataset_fingerprint=fingerprint,
                                             expected_parent_sha256=record["sha256"])
    if result.get("success"):
        mode.update(drawer=False, parent={}, confirm=None)
        return "", text, _extra_epochs_update(preview), mode
    error = result.get("error", "unknown error")
    if result.get("parent_changed"):  # changed after this check's own read
        error = f"{error} {_changed_parent_action(source)}"
    return f"Not started: {error}", text, _extra_epochs_update(preview), mode


def _toggle_continue_source(source):
    """The run list's two selects, its Refresh and the chosen checkpoint's
    details, or the typed path."""
    from_list = source != CONTINUE_FROM_PATH
    return (gr.update(visible=from_list), gr.update(visible=from_list),
            gr.update(visible=from_list), gr.update(visible=from_list),
            gr.update(visible=not from_list))


# =============================================================================
# Help
# =============================================================================
#: The fields' help, behind the shared ⓘ (`src/webui/components/help.py`):
#: ``{elem_id: (title, text)}``, registered by the tab builder. Only a
#: state-bearing text stays in ``info=``.
FIELD_HELP = {
    "shep-train-dataset": (
        "Dataset",
        "The server's workspace list, as on Research › Datasets. Choosing one only reads it; "
        "nothing changes on the server.\n\n"
        "Start sends the dataset fingerprint shown, and the server refuses the start if the "
        "workspace changed since.",
    ),
    "run_name": (
        "Run name",
        "Optional. Each run is saved in a new folder of its own:\n"
        "<workspace>/runs/<architecture>/<date-time>_<name>_<id>/\n\n"
        "Letters, digits and hyphens are kept; blank uses \"run\". The run view shows the "
        "name once training starts.",
    ),
    # The base form: each recipe setting's label, what it is, its reference
    # value and that value's source (`setting_help`).
    **{name: setting_help(name, BASE, BASE_SOURCES) for name in BASE_FIELDS},
    "device": (
        "Device",
        "auto: the GPU (CUDA). With no CUDA device visible, Start is refused: training never "
        "falls back to the CPU on its own.\n\n"
        "- cpu: for development, recorded as asked.\n"
        "- Continue… has its own Device, which starts from this one.",
    ),
    "continue_source": (
        "Checkpoint source",
        "Where the checkpoint to continue from comes from.\n\n"
        "• From run list: the run bar's dataset's model records (bases and task models), as "
        "the server lists them.\n"
        "• Manual path: a typed path. It only finds a checkpoint that list has; any other "
        "file is refused, unopened.\n\n"
        "A file without a model spec is never continued here. Both are checked the same way. "
        "Listing opens each checkpoint, so it is an operator control.",
    ),
    "continue_run": (
        "Run",
        "The run to continue from, latest first: its name, architecture and start time.\n\n"
        "Continuing trains a new run of its own; the run chosen here is not changed.",
    ),
    "checkpoint_dropdown": (
        "Checkpoint",
        "The chosen run's checkpoints, newest first.\n\n"
        "• Latest save: last.pt, the run's last training state.\n"
        "• Top-k save: kept for its Trainer validation figure (LP accuracy for a base, pooled "
        "MRR for a task model; not a test result).\n\n"
        "The table under it shows the file, its stage, what it holds, its Trainer validation "
        "figure and its model fingerprint. The server finds the file in its list again and "
        "compares its bytes with that fingerprint. Refresh lists them again.",
    ),
    "manual_checkpoint": (
        "Checkpoint path",
        "The path of a model record the run list has, typed or pasted. It is checked the same "
        "way as a listed one.",
    ),
    "continue_device": (
        "Device",
        "Where the continued run trains: auto is the GPU (CUDA), and with no CUDA device "
        "visible the start is refused; choose cpu for development.\n\n"
        "Check checkpoint and Continue Training both use it. Mixed precision works only on "
        "CUDA; the check says when the checkpoint's setting would not hold here.\n\n"
        "It starts from the Parameters' Device each time Continue… opens.",
    ),
    "extend_epochs": (
        "Extra epochs",
        "Extend a finished run by this many epochs, trained at the fixed rate shown in the "
        "check result.",
    ),
}

#: The ⓘ inside the page's rendered HTML.
TOPIC_HELP = {
    "training-charts": (
        "Chart gestures",
        charts.GESTURES + "\n\n"
        "The wheel zooms around the pointer:\n"
        "• over the epoch axis, the epochs (the four charts share them);\n"
        "• over a value axis, that chart's values;\n"
        "• inside a plot, both.\n\n"
        "Keys, with a chart focused: + and − zoom, x and y zoom one axis (Shift zooms out), "
        "the arrows pan, End jumps to the latest epochs, 0 resets.\n\n"
        "The view is kept per run in this browser tab, and the epoch axis never starts below "
        "0. While zoomed in a live run, a button counts the new epochs beyond the view. Away "
        "from the charts the wheel scrolls the page.",
    ),
    "training-used-dataset": (
        "Used dataset fingerprint",
        "The dataset the run trained on, as its checkpoint records it (the split manifest's "
        "fingerprint), compared in full with the run bar's dataset.\n\n"
        "• Shown to the operator only: reading it opens the checkpoint.\n"
        "• Before the first checkpoint nothing records it yet; a run that ended without one "
        "never recorded it.",
    ),
}


# =============================================================================
# The page
# =============================================================================
def create_training_tab(ctx: Any) -> None:
    """Build Research › Prepare › Training (the page contract:
    `src/webui/shell.py`). `ctx` gives the help registry, the shared timers,
    the cross-links, the chips' read after every gated act and this session's
    run-bar workspace (``ctx.training_workspace``)."""
    ctx.help.fields(FIELD_HELP)
    ctx.help.topics(TOPIC_HELP)

    ws = gr.State(empty_workspace())
    mode = gr.State({})
    cache = gr.State({})
    # The fast poll's own record (its reads kept by time, `POLL_READS`, and the
    # GPU slot it saw last, for the chips): only it writes it.
    polled = gr.State({})
    dismissed = gr.State([])
    parents = gr.State({})

    with gr.Column(elem_id="shep-train", elem_classes=["shep-train"]):
        # ------------------------------------------------------------- run bar
        with gr.Column(elem_id="shep-train-runbar", elem_classes=["shep-runbar"]) as runbar:
            with gr.Row(elem_classes=["shep-rb-row", "shep-rb-row1"]):
                dataset = gr.Dropdown(label="Dataset", choices=[], value=None,
                                      elem_id="shep-train-dataset", scale=0, min_width=240,
                                      container=False)
                chip = plain_html(chip_html(None), elem_id="shep-train-chip", js_on_load=_POPOVER_JS)
                pill = plain_html(pill_html({"status": "idle"}), elem_id="shep-train-pill")
            with gr.Row(elem_classes=["shep-rb-row", "shep-rb-row2"]):
                run_name = gr.Textbox(label="Run name", value="", max_lines=1, elem_id="run_name",
                                      scale=0, min_width=240, container=False)
                continues = plain_html("", elem_id="shep-train-continues")
                reason = plain_html('<div class="shep-rb-reason"></div>', elem_id="shep-train-reason",
                                    js_on_load=TRIGGER_JS)
                start_btn = gr.Button("▸ Start", variant="primary", interactive=False, scale=0,
                                      elem_id="shep-train-start", elem_classes=["shep-gated"])
                cont_btn = gr.Button("Continue…", interactive=False, scale=0,
                                     elem_id="shep-train-continue", elem_classes=["shep-gated"])
                stop_btn = gr.Button("■ Stop", variant="secondary", interactive=False, scale=0,
                                     elem_id="shep-train-stop", elem_classes=["shep-gated", "shep-stop"])
                resume_btn = gr.Button("▸ Continue Training", variant="primary", interactive=False,
                                       scale=0, elem_id="shep-train-resume",
                                       elem_classes=["shep-gated", "shep-drawer-only"])
                cancel_btn = gr.Button("Cancel", scale=0, elem_id="shep-train-cancel",
                                       elem_classes=["shep-drawer-only"])
            # Stop asks first, in the code's words (contract M3a).
            with gr.Row(elem_classes=["shep-rb-row", "shep-rb-confirm"]):
                stop_text = plain_html("", elem_id="shep-train-stop-text")
                stop_confirm_btn = gr.Button("■ Stop training", variant="stop", scale=0,
                                             elem_id="shep-train-stop-confirm",
                                             elem_classes=["shep-gated"])
                stop_keep_btn = gr.Button("Keep training", scale=0, elem_id="shep-train-stop-keep")
            notice = plain_html("", elem_id="shep-train-notice")
        banner = plain_html("", elem_id="shep-train-banner", js_on_load=_BANNER_JS)

        with gr.Row(elem_id="shep-train-page", elem_classes=["shep-page", "shep-train-page"],
                    equal_height=False) as page:
            # ------------------------------------------------------- parameters
            with gr.Column(elem_id="shep-train-params", elem_classes=["shep-panel"], scale=10,
                           min_width=360):
                plain_html('<div class="shep-panel-hd"><h2>Parameters</h2>'
                           '<span class="shep-panel-right">for the next base run</span></div>',
                           elem_id="shep-train-params-hd")
                with gr.Column(elem_id="shep-train-params-bd", elem_classes=["shep-train-bd"]):
                    # Exactly the reference recipe's base settings (decision 12),
                    # each starting at the recipe's value.
                    with gr.Group(elem_classes=["shep-train-grid"]):
                        shown_fields = [setting_field(name, BASE[name], elem_id=name)
                                        for name in BASE_FIELDS[:BASE_SHOWN]]
                        device = gr.Radio(label="Device", choices=["auto", "cuda", "cpu"],
                                          value="auto", elem_id="device")
                    with gr.Accordion("More reference settings", open=False,
                                      elem_id="shep-train-more"):
                        with gr.Group():
                            more_fields = [setting_field(name, BASE[name], elem_id=name)
                                           for name in BASE_FIELDS[BASE_SHOWN:]]
                    differs = plain_html(
                        differs_html(dict(zip(BASE_FIELDS, base_form_defaults())), BASE),
                        elem_id="shep-train-differs")
                    # torch.compile is configured in the Runtime Settings tab (it is an
                    # execution/runtime knob, applied per training run).
                base_fields = [*shown_fields, *more_fields]

            # ------------------------------------------------- Continue… drawer
            with gr.Column(elem_id="shep-train-drawer", elem_classes=["shep-panel"], scale=10,
                           min_width=360):
                with gr.Row(elem_classes=["shep-panel-hd", "shep-drawer-hd"]):
                    plain_html('<h2>Continue a run</h2><span class="shep-muted">replaces the '
                               'parameters until closed</span>', elem_id="shep-train-drawer-title")
                    close_btn = gr.Button("Close", scale=0, elem_id="shep-train-drawer-close",
                                          elem_classes=["shep-x-btn"])
                with gr.Column(elem_id="shep-train-drawer-bd", elem_classes=["shep-train-bd"]):
                    continue_source = gr.Radio(choices=[CONTINUE_FROM_LIST, CONTINUE_FROM_PATH],
                                               value=CONTINUE_FROM_LIST, label="Checkpoint source",
                                               elem_id="continue_source")
                    parents_note = plain_html("", elem_id="shep-train-parents-note")
                    # Two steps (the GB10 WebUI recheck: one option per file, each the
                    # run, epoch, file, kind and fingerprint, read as a wall of text):
                    # the run, then its checkpoint, each in a few words; the chosen
                    # file's details in a table under them.
                    with gr.Row(elem_classes=["shep-train-parent-row"]):
                        continue_run = gr.Dropdown(label="Run", choices=[], value=None,
                                                   elem_id="continue_run")
                        refresh_ckpt_btn = gr.Button("Refresh", size="sm", scale=0,
                                                     elem_id="shep-train-refresh")
                    checkpoint_dropdown = gr.Dropdown(label="Checkpoint", choices=[], value=None,
                                                      elem_id="checkpoint_dropdown")
                    parent_details = plain_html("", elem_id="shep-train-parent-details")
                    manual_checkpoint = gr.Textbox(label="Checkpoint path", value="", max_lines=1,
                                                   visible=False, elem_id="manual_checkpoint")
                    # The device is part of the check, so it is here, not only
                    # in the parameters the drawer covers.
                    continue_device = gr.Radio(label="Device", choices=["auto", "cuda", "cpu"],
                                               value="auto", elem_id="continue_device")
                    # Extra epochs (shown for a finished checkpoint) beside Check
                    # checkpoint, above what the check says (M5-3: under a long
                    # result it fell below the fold at 1920 x 1080).
                    with gr.Row(elem_classes=["shep-train-check-row"]):
                        # No widget minimum: hidden and empty, a Number is sent as
                        # 0, which a minimum would reject before any handler runs.
                        # Empty or 0 means "not extending"; the run plan refuses
                        # below 1.
                        extend_epochs = gr.Number(label="Extra epochs", value=None, precision=0,
                                                  visible=False, elem_id="extend_epochs",
                                                  scale=0, min_width=0)
                        check_continue_btn = gr.Button("Check checkpoint", size="sm", scale=0,
                                                       elem_id="shep-train-check",
                                                       elem_classes=["shep-gated"])
                    compare = plain_html("", elem_id="shep-train-compare")
                    continue_preview = plain_html(f'<p class="shep-note">{_CONTINUE_PROMPT}</p>',
                                                  elem_id="continue_preview")
                drawer_reason = plain_html("", elem_id="shep-train-drawer-why",
                                           elem_classes=["shep-panel-ft"])

            # ------------------------------------------------------- run view
            with gr.Column(elem_id="shep-train-runview", elem_classes=["shep-panel"], scale=22):
                # The header and the state line: the run view's first row; the
                # charts and the resources line are the others (`PAGE_CSS`).
                with gr.Column(elem_id="shep-train-rv-top"):
                    run_head = plain_html(run_head_html(None, {}, operator=False, snapshot=None),
                                          elem_id="shep-train-run-head", js_on_load=_POPOVER_JS)
                    run_snap = plain_html("", elem_id="shep-train-run-snap")
                figure = charts.chart_figure(
                    key="training", elem_id="shep-train-charts", series_fn=training_chart_series,
                    csv_fn=training_chart_csv, help_key="training-charts",
                    empty="No run to show yet.")
                resources = plain_html(_IDLE_RESOURCE_HTML, elem_id="resource_display")

    # =========================================================================
    # Wiring
    # =========================================================================
    # The line on the reference settings follows every base field (one listener).
    gr.on(triggers=[field.change for field in base_fields], fn=_on_base_changed,
          inputs=base_fields, outputs=[differs], queue=False, show_progress="hidden")
    view_outputs = [ws, dataset, chip, pill, reason, start_btn, cont_btn, stop_btn, resume_btn,
                    runbar, page, run_name, banner, run_head, run_snap, figure, drawer_reason,
                    ctx.training_workspace, cache]
    view_inputs = [ws, mode, cache, dismissed]
    # Every render writes the run bar and the run view drawn from the session's
    # selection, and may write the workspace itself (a running job's dataset),
    # so every one is in the group of the workspace's writers
    # (`shell.TRAINING_SELECTION_CONCURRENCY`): it runs after those asked
    # before it, from what they wrote, and its answer is never applied after
    # the answer of one asked after it.
    in_order = ctx.training_workspace_writes
    view_step = {"fn": _render_view, "inputs": view_inputs, "outputs": view_outputs,
                 "show_progress": "hidden", **in_order}
    # In the group an event can wait as long as the one before it runs, and with
    # Gradio's defaults the browser does not send a listener's trigger while its
    # event waits (an input's "once" is dropped; a change's "always_last" is sent
    # once, with the first trigger's data): with "multiple" each choice of the
    # workspace joins, and the last one asked is the one that stays.
    every_one = {"trigger_mode": "multiple"}
    # The fast poll's hand-off of a change it found (`_poll`).
    changed = gr.JSON(value=None, visible=False, elem_id=CHANGED_ID)

    # The page shown: the list read again, then the run bar whole, in that order.
    ctx.on_page_open(TRAINING, {
        "fn": _open, "inputs": [ws, ctx.training_workspace],
        "outputs": [ws, dataset, chip, ctx.training_workspace, cache, resources],
        "show_progress": "hidden", **in_order}).then(**view_step)

    # The shell's timers (outside gr.Tabs), read-only polls without the queue.
    # The fast one writes the status pill (`POLL_KEYS`) and the chips, and hands
    # every other change to a queued render in the group. That render reads
    # everything at its turn, so the token's default trigger ("always_last") is
    # the one wanted: tokens given while one render waits need one more render,
    # not one each.
    ctx.timer_fast.tick(_poll, inputs=[ctx.location, *view_inputs, polled],
                        outputs=[*(view_outputs[VIEW_KEYS.index(key)] for key in POLL_KEYS),
                                 changed, ctx.chips, polled],
                        queue=False, show_progress="hidden")
    changed.change(**{**view_step, "fn": _render_changed, "inputs": [*view_inputs, polled]})
    ctx.timer_slow.tick(_poll_resources, inputs=[ctx.location], outputs=[resources],
                        queue=False, show_progress="hidden")

    # The run bar's dataset.
    dataset.input(_on_dataset, inputs=[dataset, mode],
                  outputs=[ws, chip, ctx.training_workspace, mode, notice],
                  show_progress="hidden", **in_order, **every_one).then(**view_step)
    # The session's workspace changed (a hand choice above writes it, and so
    # do Datasets' "Use for training ›" and a render that follows a job
    # running in another workspace): with Continue… open, the parent list
    # follows it (a gated listing, then the chosen parent, the run bar and the
    # chips again). A follow therefore costs its render and this chain.
    chosen_step = {"fn": _on_parent_chosen,
                   "inputs": [continue_source, checkpoint_dropdown, manual_checkpoint, parents, ws,
                              mode], "outputs": [mode, compare, parent_details],
                   "show_progress": "hidden"}
    ctx.training_workspace.change(_on_training_workspace, inputs=[ctx.training_workspace, ws, mode],
                                  outputs=[ws, dataset, chip, mode],
                                  show_progress="hidden", **in_order, **every_one).then(
        _relist_parents, inputs=[ws, mode],
        outputs=[parents, continue_run, checkpoint_dropdown, parents_note, compare,
                 continue_preview, extend_epochs], show_progress="hidden").then(**chosen_step).then(
        **view_step).then(**ctx.chip_refresh)
    def _on_chip_click(evt: gr.EventData):
        """The chip's "All datasets ›": Research › Datasets."""
        return ctx.go("datasets") if event_act(evt) == "datasets" else ctx.no_nav()

    def _on_reason_click(evt: gr.EventData):
        """The reason line's "Open Test ›": Research › Test."""
        return ctx.go("test") if event_act(evt) == "open-test" else ctx.no_nav()

    chip.click(_on_chip_click, inputs=None, outputs=ctx.nav_outputs, show_progress="hidden")
    reason.click(_on_reason_click, inputs=None, outputs=ctx.nav_outputs, show_progress="hidden")
    banner.click(_on_banner_click, inputs=[dismissed, cache], outputs=[dismissed, banner],
                 show_progress="hidden")

    # Start, Stop.
    # Start also closes a Stop confirmation left from an earlier job.
    start_btn.click(_on_start, inputs=[ws, run_name, device, *base_fields], outputs=[notice],
                    show_progress="hidden").then(
        close_stop_question, inputs=[mode], outputs=[mode], show_progress="hidden").then(
        **view_step).then(**ctx.chip_refresh)
    stop_btn.click(_on_stop_request, inputs=[mode], outputs=[stop_text, mode],
                   show_progress="hidden").then(**view_step).then(**ctx.chip_refresh)
    stop_confirm_btn.click(_on_stop, inputs=[mode], outputs=[notice, mode],
                           show_progress="hidden").then(**view_step).then(**ctx.chip_refresh)
    stop_keep_btn.click(close_stop_question, inputs=[mode], outputs=[mode],
                        show_progress="hidden").then(**view_step)

    # Continue…
    cont_btn.click(_on_continue_open, inputs=[ws, mode],
                   outputs=[notice, mode, parents, continue_run, checkpoint_dropdown,
                            parents_note, continue_preview],
                   show_progress="hidden").then(
        _continue_device, inputs=[device], outputs=[continue_device],
        show_progress="hidden").then(**chosen_step).then(**view_step).then(**ctx.chip_refresh)
    for closer in (cancel_btn, close_btn):
        closer.click(_on_continue_close, inputs=[mode], outputs=[mode],
                     show_progress="hidden").then(**view_step)
    refresh_ckpt_btn.click(_refresh_checkpoints, inputs=[ws, mode, checkpoint_dropdown],
                           outputs=[parents, continue_run, checkpoint_dropdown, parents_note],
                           show_progress="hidden").then(**ctx.chip_refresh)
    # A run chosen: its checkpoints (the checkpoint select's change then
    # clears the last check and records the parent, as for any choice).
    continue_run.change(on_run_chosen, inputs=[continue_run, parents, checkpoint_dropdown],
                        outputs=[checkpoint_dropdown], show_progress="hidden")
    check_continue_btn.click(
        _on_check_continue,
        inputs=[continue_source, checkpoint_dropdown, manual_checkpoint, continue_device,
                extend_epochs, ws, parents, mode],
        outputs=[continue_preview, extend_epochs, mode, compare],
        show_progress="hidden").then(**view_step).then(**ctx.chip_refresh)
    resume_btn.click(
        _on_resume,
        inputs=[ws, continue_device, continue_source, checkpoint_dropdown, manual_checkpoint,
                extend_epochs, parents, mode],
        outputs=[notice, continue_preview, extend_epochs, mode],
        show_progress="hidden").then(**view_step).then(**ctx.chip_refresh)
    continue_source.change(fn=_toggle_continue_source, inputs=[continue_source],
                           outputs=[continue_run, checkpoint_dropdown, refresh_ckpt_btn,
                                    parent_details, manual_checkpoint])
    # A changed selection, or source, makes the last check and its Extra
    # epochs stale: both are cleared (contract M3a review of 094b90d, P2-2).
    # `.change` fires on updates from other events too (Refresh), so the run
    # list and the typed path clear only when they are the source.
    continue_source.change(fn=_on_continue_selection_changed, inputs=[],
                           outputs=[extend_epochs, continue_preview])
    checkpoint_dropdown.change(fn=_on_listed_changed, inputs=[continue_source],
                               outputs=[extend_epochs, continue_preview])
    manual_checkpoint.change(fn=_on_manual_changed, inputs=[continue_source],
                             outputs=[extend_epochs, continue_preview])
    continue_device.change(fn=_on_continue_device_changed, inputs=[], outputs=[continue_preview])
    # The session's record of the chosen parent (the run bar's actions).
    chosen_inputs = [continue_source, checkpoint_dropdown, manual_checkpoint, parents, ws, mode]
    for component in (continue_source, checkpoint_dropdown, manual_checkpoint):
        component.change(_on_parent_chosen, inputs=chosen_inputs,
                         outputs=[mode, compare, parent_details],
                         show_progress="hidden").then(**view_step)

    # The run bar's line for a continued run (while the drawer is open): a new
    # run of its own (contract M3a-2), continued from the checkpoint the drawer
    # shows chosen (named there once; the run view's header says whether the
    # run it shows is that one).
    def _continues(mode_value):
        chosen = ((mode_value or {}).get("parent") or {}).get("chosen")
        return ('<span class="shep-rb-k">Run</span> a new one, continued from the checkpoint '
                'chosen in Continue a run' if chosen else
                '<span class="shep-rb-k">Run</span> <span class="shep-muted">a new one, once a '
                'checkpoint is chosen in Continue a run</span>')

    mode.change(_continues, inputs=[mode], outputs=[continues], show_progress="hidden")


#: This page's element id (`shell.page_elem_id`), the scope of `PAGE_CSS`.
_PAGE = f"#shep-page-{TRAINING}"

#: The smallest plot area the run view draws at a wide size (M5-3: about 70 to
#: 100 px at 1536 x 760), and the room uPlot takes around it in the plot's box
#: (the epoch axis, 30 px, and its margins).
PLOT_MIN_HEIGHT = 140
PLOT_AXIS_ROOM = 47

#: The lock and the stop square drawn before a gated button's words (CSS only).
_LOCK_MASK = ("url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E"
              "%3Crect x='3' y='7' width='10' height='7.5' rx='1.6'/%3E%3Cpath d='M5.2 7V5.2a2.8 2.8 0 0 1 "
              "5.6 0V7' fill='none' stroke='black' stroke-width='1.6'/%3E%3C/svg%3E\")")

#: This page's style sheet (the page contract: listed in `src/webui/app.py`'s
#: ``PAGE_CSS``), with the chart component's. Nothing under 14 px.
PAGE_CSS = charts.CHART_CSS + f"""
{_PAGE} .shep-train {{ gap: 12px !important; }}
{_PAGE} #shep-train-runbar {{ background: var(--shep-panel); border: 1px solid var(--shep-border); border-radius: 10px; padding: 10px 16px !important; gap: 8px !important; flex: none; }}
{_PAGE} .shep-rb-row {{ align-items: center !important; gap: 10px 14px !important; flex-wrap: wrap !important; }}
{_PAGE} .shep-rb-row > * {{ min-width: 0; }}
{_PAGE} .shep-rb-row > .block {{ width: auto !important; }}
{_PAGE} #shep-train-dataset, {_PAGE} #run_name {{ flex: 0 0 auto !important; width: auto !important; }}
{_PAGE} #shep-train-dataset > div:not(.wrap), {_PAGE} #run_name > label {{ display: flex !important; align-items: center; gap: 8px; }}
{_PAGE} #shep-train-dataset span[id^="shep-lbl-"], {_PAGE} #run_name span[id^="shep-lbl-"] {{ position: static !important; display: block !important; width: 96px !important; height: auto !important; overflow: visible !important; clip: auto !important; clip-path: none !important; margin: 0 !important; padding: 0 !important; background: none !important; border: 0 !important; white-space: nowrap; font-size: 14px; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: var(--shep-text-2) !important; }}
{_PAGE} #shep-train-dataset .wrap {{ width: clamp(220px, 16vw, 320px); min-width: 0 !important; }}
{_PAGE} #shep-train-dataset input {{ flex: 1 1 auto !important; width: 0 !important; min-width: 0 !important; }}
{_PAGE} #shep-train-dataset .secondary-wrap {{ min-width: 0 !important; }}
{_PAGE} #shep-train-dataset .icon-wrap {{ flex: none !important; }}
/* The two selects drawn as fields: a border, the text clear of the chevron, grey while disabled
   (a value too long ends in the shell's titled ellipsis, the rule for every select). */
{_PAGE} #shep-train-dataset .wrap:has(> .wrap-inner), {_PAGE} #continue_run .wrap:has(> .wrap-inner), {_PAGE} #checkpoint_dropdown .wrap:has(> .wrap-inner) {{ border: 1px solid var(--shep-border-2) !important; border-radius: 8px !important; background: #fff !important; box-shadow: none !important; }}
{_PAGE} #shep-train-dataset .wrap:has(> .wrap-inner):focus-within, {_PAGE} #continue_run .wrap:has(> .wrap-inner):focus-within, {_PAGE} #checkpoint_dropdown .wrap:has(> .wrap-inner):focus-within {{ border-color: var(--shep-accent) !important; }}
{_PAGE} #shep-train-dataset input, {_PAGE} #continue_run input, {_PAGE} #checkpoint_dropdown input {{ padding-right: 30px !important; }}
{_PAGE} #shep-train-dataset .icon-wrap, {_PAGE} #continue_run .icon-wrap, {_PAGE} #checkpoint_dropdown .icon-wrap {{ color: var(--shep-text-2) !important; }}
{_PAGE} #shep-train-dataset .wrap:has(input:disabled), {_PAGE} #continue_run .wrap:has(input:disabled), {_PAGE} #checkpoint_dropdown .wrap:has(input:disabled) {{ background: var(--shep-grey-50) !important; border-color: var(--shep-grey-200) !important; }}
{_PAGE} #shep-train-dataset input:disabled, {_PAGE} #continue_run input:disabled, {_PAGE} #checkpoint_dropdown input:disabled {{ color: var(--shep-grey-600) !important; -webkit-text-fill-color: var(--shep-grey-600); cursor: not-allowed; }}
{_PAGE} #shep-train-dataset .wrap:has(input:disabled) .icon-wrap {{ color: var(--shep-grey-600) !important; }}
{_PAGE} #run_name .input-container {{ width: 260px; }}
{_PAGE} #run_name input {{ border: 1px solid var(--shep-border-2) !important; border-radius: 8px; padding: 6px 10px !important; background: #fff; }}
{_PAGE} #run_name input:disabled {{ background: var(--shep-grey-50); color: var(--shep-muted); }}
{_PAGE} .shep-rb-row > button {{ flex: none !important; width: auto !important; min-width: max-content !important; padding: 0 14px !important; height: 40px; white-space: nowrap; }}
{_PAGE} #shep-train-chip {{ flex: 1 1 360px !important; position: relative; z-index: {CHIP_Z_INDEX}; }}
{_PAGE} .shep-rb-chip {{ display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; font-size: 15px; position: relative; }}
/* The identity's parts wrap one by one (never hidden): "built …", the fingerprint, then the marks with "Details ›". */
{_PAGE} .shep-rb-chip > .shep-ds {{ display: contents; }}
{_PAGE} .shep-rb-chip .shep-ds-marks {{ display: inline-flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; }}
{_PAGE} .shep-rb-chip > .shep-ds > .shep-sep {{ margin: 0 -2px; }}
{_PAGE} .shep-rb-chip > .shep-ds > .shep-sep:has(+ .shep-ds-marks) {{ display: none; }}
{_PAGE} .shep-rb-inuse {{ color: var(--shep-text-2); font-size: 14px; }}
{_PAGE} .shep-link {{ border: 0; background: none; padding: 0; color: var(--shep-accent-text); font-size: 15px; cursor: pointer; font-weight: 500; }}
{_PAGE} .shep-link:hover {{ text-decoration: underline; }}
{_PAGE} .shep-pop {{ position: absolute; top: calc(100% + 8px); left: 0; z-index: 30; width: min({DETAILS_POPOVER_WIDTH}px, calc(100vw - 48px)); background: #fff; border: 1px solid var(--shep-border-2); border-radius: 10px; box-shadow: 0 10px 28px rgba(20,24,40,.16); padding: 12px 14px; display: flex; flex-direction: column; gap: 10px; font-size: 15px; }}
{_PAGE} .shep-pop[hidden] {{ display: none; }}
{_PAGE} .shep-pop-hd {{ display: flex; align-items: center; gap: 8px; overflow-wrap: anywhere; }}
{_PAGE} .shep-pop-x {{ margin-left: auto; width: 28px; height: 28px; border: 0; background: none; font-size: 20px; cursor: pointer; color: var(--shep-muted); }}
{_PAGE} .shep-pop-sub {{ font-weight: 600; font-size: 15px; }}
{_PAGE} .shep-pop .shep-kv-2 {{ grid-template-columns: max-content max-content max-content max-content; font-size: 14px; gap: 2px 12px; }}
{_PAGE} .shep-pop .shep-v {{ overflow-wrap: anywhere; }}
{_PAGE} .shep-pop .shep-link {{ align-self: flex-start; }}
{_PAGE} #shep-train-pill {{ margin-left: auto; flex: none !important; }}
{_PAGE} .shep-pill {{ display: inline-flex; align-items: center; gap: 6px; height: 30px; padding: 0 12px; border-radius: 999px; font-size: 14px; font-weight: 600; white-space: nowrap; border: 1px solid var(--shep-border-2); background: var(--shep-panel-2); color: var(--shep-text-2); }}
{_PAGE} .shep-pill-running {{ background: var(--shep-accent); border-color: var(--shep-accent); color: #fff; }}
{_PAGE} .shep-pill-accent {{ background: var(--shep-accent-50); border-color: var(--shep-accent-100); color: var(--shep-accent-text); }}
{_PAGE} .shep-pill-ok {{ background: var(--shep-green-50); border-color: var(--shep-green-200); color: var(--shep-green-800); }}
{_PAGE} .shep-pill-grey {{ background: var(--shep-grey-50); border-color: var(--shep-grey-200); color: var(--shep-grey-800); }}
{_PAGE} .shep-pill-red {{ background: var(--shep-red-50); border-color: var(--shep-red-200); color: var(--shep-red-700); }}
{_PAGE} .shep-dot-r, {_PAGE} .shep-dot-w {{ width: 8px; height: 8px; border-radius: 50%; background: var(--shep-red-700); }}
{_PAGE} .shep-dot-w {{ background: #fff; }}
{_PAGE} #shep-train-reason {{ flex: 1 1 260px !important; }}
{_PAGE} .shep-rb-reason {{ display: flex; align-items: center; justify-content: flex-end; gap: 12px; flex-wrap: wrap; }}
{_PAGE} .shep-rb-why {{ display: inline-flex; align-items: flex-start; gap: 6px; flex-wrap: nowrap; font-size: 15px; color: var(--shep-text-2); text-align: right; }}
{_PAGE} .shep-rb-why > svg, {_PAGE} .shep-rb-why > .shep-tag {{ flex: none; margin-top: 3px; }}
{_PAGE} .shep-rb-why > .shep-tag {{ margin-top: 0; }}
{_PAGE} .shep-rb-why > span:not(.shep-tag) {{ min-width: 0; text-wrap: pretty; }}
/* A reason of two sentences breaks between them (`renderers.word_groups`); a sentence wider than its line wraps inside its group without leaving one word alone (text-wrap: pretty). */
{_PAGE} .shep-rb-why .shep-wg {{ text-wrap: pretty; }}
{_PAGE} .shep-rb-why .shep-link {{ white-space: nowrap; }}
{_PAGE} .shep-rb-progress {{ flex: 1 1 160px; max-width: 380px; height: 8px; border-radius: 4px; background: var(--shep-grey-200); overflow: hidden; }}
{_PAGE} .shep-rb-progress > i {{ display: block; height: 100%; background: var(--shep-accent); }}
{_PAGE} .shep-gated::before {{ content: ""; display: inline-block; width: 12px; height: 12px; margin-right: 6px; background: currentColor; -webkit-mask: {_LOCK_MASK} center / contain no-repeat; mask: {_LOCK_MASK} center / contain no-repeat; opacity: .85; }}
{_PAGE} button.primary:not(:disabled) {{ background: var(--shep-accent) !important; border-color: var(--shep-accent) !important; color: #fff !important; }}
{_PAGE} button.primary:not(:disabled):hover {{ background: var(--shep-accent-600) !important; border-color: var(--shep-accent-600) !important; }}
{_PAGE} #shep-train-stop:not(:disabled) {{ color: var(--shep-red-700) !important; border-color: var(--shep-red-200) !important; background: #fff !important; }}
{_PAGE} .shep-rb-row > button:disabled {{ background: var(--shep-grey-50) !important; border: 1px solid var(--shep-grey-200) !important; color: var(--shep-grey-600) !important; opacity: 1 !important; box-shadow: none !important; cursor: not-allowed; }}
{_PAGE} .shep-drawer-only, {_PAGE} #shep-train-continues {{ display: none !important; }}
{_PAGE} .shep-rb-drawer .shep-drawer-only {{ display: inline-flex !important; }}
{_PAGE} .shep-rb-drawer #shep-train-continue, {_PAGE} .shep-rb-drawer #shep-train-stop, {_PAGE} .shep-rb-drawer #run_name {{ display: none !important; }}
{_PAGE} .shep-rb-drawer #shep-train-continues {{ display: block !important; flex: 0 1 auto !important; }}
{_PAGE} .shep-rb-k {{ font-size: 14px; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: var(--shep-text-2); margin-right: 8px; }}
{_PAGE} .shep-rb-confirm {{ display: none !important; }}
{_PAGE} .shep-rb-confirming .shep-rb-confirm {{ display: flex !important; }}
{_PAGE} .shep-rb-confirming .shep-rb-row2 {{ display: none !important; }}
{_PAGE} #shep-train-stop-text {{ flex: 1 1 400px !important; }}
{_PAGE} .shep-rb-confirm-q {{ background: var(--shep-red-50); border: 1px solid var(--shep-red-200); border-radius: 8px; padding: 8px 12px; font-size: 15px; }}
{_PAGE} #shep-train-notice:empty, {_PAGE} #shep-train-notice .html-container:empty {{ display: none; }}
{_PAGE} #shep-train-notice:has(.html-container:empty), {_PAGE} #shep-train-banner:not(:has(.shep-banner)) {{ display: none !important; }}
{_PAGE} #shep-train-notice {{ font-size: 15px; color: var(--shep-amber-800); overflow-wrap: anywhere; }}
{_PAGE} .shep-banner {{ border-radius: 8px; border: 1px solid; padding: 10px 14px; margin-bottom: 6px; font-size: 15px; }}
{_PAGE} .shep-banner-red {{ background: var(--shep-red-50); border-color: var(--shep-red-200); border-left: 4px solid var(--shep-red-700); color: var(--shep-red-700); }}
{_PAGE} .shep-banner-amber {{ background: var(--shep-amber-50); border-color: var(--shep-amber-300); border-left: 4px solid var(--shep-amber-600); color: var(--shep-amber-800); }}
{_PAGE} .shep-banner-hd {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
{_PAGE} .shep-banner-h {{ font-weight: 600; overflow-wrap: anywhere; }}
{_PAGE} .shep-banner-where {{ margin-left: auto; color: var(--shep-text-2); font-size: 14px; }}
/* The last error line the run printed, right under the headline: read without opening or scrolling the tail. */
{_PAGE} .shep-banner-cause {{ margin-top: 4px; color: var(--shep-text); line-height: 1.4; overflow-wrap: anywhere; }}
{_PAGE} .shep-banner-k {{ font-weight: 600; color: var(--shep-red-700); }}
{_PAGE} .shep-banner-v {{ font-family: var(--shep-mono); font-size: 14px; }}
{_PAGE} .shep-banner-more {{ margin-top: 6px; }}
{_PAGE} .shep-banner-more > summary {{ cursor: pointer; font-size: 14px; font-weight: 600; color: var(--shep-red-700); width: max-content; max-width: 100%; }}
{_PAGE} .shep-banner-n {{ font-weight: 400; color: var(--shep-text-2); }}
{_PAGE} .shep-banner-tail {{ margin: 6px 0 0; padding: 6px 10px; background: #fff; border: 1px solid var(--shep-red-200); border-radius: 6px; color: var(--shep-text); font-family: var(--shep-mono) !important; font-size: 14px; line-height: 1.35; white-space: pre-wrap; overflow-wrap: anywhere; max-height: calc(4 * 1.35em + 14px); overflow: auto; }}
{_PAGE} #shep-train-params-bd, {_PAGE} #shep-train-drawer-bd {{ padding: 4px 16px 12px !important; gap: 10px !important; }}
{_PAGE} #shep-train-drawer {{ display: none !important; }}
{_PAGE} .shep-train-drawer-open #shep-train-drawer {{ display: flex !important; }}
{_PAGE} .shep-train-drawer-open #shep-train-params {{ display: none !important; }}
{_PAGE} .shep-drawer-hd {{ flex-wrap: nowrap !important; }}
{_PAGE} #shep-train-drawer-title {{ flex: 1 1 auto !important; }}
{_PAGE} #shep-train-drawer-title .html-container > div {{ display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }}
{_PAGE} #shep-train-drawer-title h2 {{ font-size: 16px; font-weight: 600; margin: 0; }}
{_PAGE} .shep-x-btn {{ font-size: 0 !important; width: 32px; height: 32px; min-width: 0 !important; padding: 0 !important; background: none !important; border: 0 !important; box-shadow: none !important; }}
{_PAGE} .shep-x-btn::before {{ content: "×"; font-size: 22px; color: var(--shep-muted); }}
{_PAGE} .shep-cmp, {_PAGE} .shep-check {{ border-radius: 8px; padding: 10px 12px; font-size: 15px; border: 1px solid var(--shep-border); }}
{_PAGE} .shep-cmp p, {_PAGE} .shep-check p {{ margin: 4px 0 0; }}
{_PAGE} .shep-cmp-amber, {_PAGE} .shep-check-amber {{ background: var(--shep-amber-50); border-color: var(--shep-amber-300); color: var(--shep-amber-800); }}
{_PAGE} .shep-cmp-ok {{ background: var(--shep-green-50); border-color: var(--shep-green-200); color: var(--shep-green-800); }}
{_PAGE} .shep-cmp-grey {{ background: var(--shep-grey-50); border-color: var(--shep-grey-200); color: var(--shep-grey-800); }}
{_PAGE} .shep-check-ok {{ background: #fff; }}
{_PAGE} .shep-check-h {{ font-weight: 600; margin-bottom: 4px; }}
{_PAGE} .shep-check code, {_PAGE} .shep-rv-snap code {{ overflow-wrap: anywhere; word-break: normal; }}
{_PAGE} .shep-inherit {{ margin-top: 8px; border: 1px solid var(--shep-border); border-radius: 8px; background: #fff; }}
{_PAGE} .shep-inherit > summary {{ padding: 6px 10px; cursor: pointer; font-weight: 600; }}
{_PAGE} .shep-inherit .shep-ftbl {{ margin: 0 10px 10px; width: calc(100% - 20px); }}
{_PAGE} .shep-drawer-why {{ font-size: 15px; color: var(--shep-text-2); }}
{_PAGE} .shep-drawer-why .shep-rb-why {{ text-align: left; flex-wrap: nowrap; align-items: flex-start; }}
{_PAGE} #shep-train-drawer-why:not(:has(.shep-drawer-why)) {{ display: none !important; }}
{_PAGE} #shep-train-rv-top {{ gap: 0 !important; flex: none !important; }}
{_PAGE} .shep-rv-hd {{ gap: 6px 10px !important; }}
{_PAGE} .shep-rv-k {{ font-size: 14px; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: var(--shep-text-2); margin-right: 6px; }}
{_PAGE} .shep-rv-name {{ overflow-wrap: anywhere; }}
{_PAGE} .shep-rv-hd > .shep-muted {{ font-size: 14px; }}
/* Details, at the header's end: its popover opens under it, aligned to the right. */
{_PAGE} .shep-rv-tools {{ margin-left: auto; position: relative; }}
{_PAGE} .shep-rv-pop {{ left: auto; right: 0; width: min(560px, calc(100vw - 48px)); }}
{_PAGE} .shep-rv-pop .shep-kv {{ font-size: 14px; gap: 4px 14px; }}
{_PAGE} .shep-rv-snap {{ margin: 0; padding: 8px 16px 0; font-size: 15px; color: var(--shep-text-2); overflow-wrap: anywhere; }}
{_PAGE} #shep-train-charts {{ padding: 8px 16px 6px !important; }}
{_PAGE} .shep-rv-res {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 6px; padding: 8px 16px; border-top: 1px solid var(--shep-border); font-size: 15px; color: var(--shep-text-2); }}
{_PAGE} .shep-rv-res b {{ color: var(--shep-text); margin-right: 8px; }}
{_PAGE} .shep-lock {{ flex: none; }}
{_PAGE} .shep-html svg {{ display: inline-block; vertical-align: -2px; }}
{_PAGE} .shep-train-parent-row {{ align-items: flex-end !important; flex-wrap: nowrap !important; }}
{_PAGE} #continue_run {{ flex: 1 1 auto !important; min-width: 0 !important; }}
/* An open list at least as wide as its few words need (Gradio sizes the list and each entry to
   the box, inline), so an option reads on one line; still fixed and opaque, as Gradio draws it,
   and kept inside the window. */
{_PAGE} #continue_run .options, {_PAGE} #checkpoint_dropdown .options {{ min-width: min(28rem, calc(100vw - 48px)); }}
{_PAGE} #continue_run .options li, {_PAGE} #checkpoint_dropdown .options li {{ min-width: 100%; }}
/* The chosen checkpoint's details: a two-column table (`parent_details_html`). */
{_PAGE} #shep-train-parent-details:not(:has(.shep-ckpt)) {{ display: none !important; }}
{_PAGE} .shep-ckpt {{ border: 1px solid var(--shep-border); border-radius: 8px; background: #fff; padding: 8px 12px; }}
{_PAGE} .shep-ckpt .shep-kv {{ font-size: 14px; gap: 4px 14px; }}
{_PAGE} .shep-ckpt .shep-v {{ overflow-wrap: anywhere; min-width: 0; }}
{_PAGE} .shep-ckpt-amber {{ background: var(--shep-amber-50); border-color: var(--shep-amber-300); color: var(--shep-amber-800); }}
{_PAGE} #shep-train-refresh {{ flex: none !important; width: auto !important; min-width: max-content !important; }}
/* Extra epochs (when shown) beside Check checkpoint, above what the check says. */
{_PAGE} .shep-train-check-row {{ align-items: flex-end !important; flex-wrap: wrap !important; gap: 8px 12px !important; }}
{_PAGE} .shep-train-check-row > * {{ flex: 0 0 auto !important; width: auto !important; }}
{_PAGE} .shep-train-check-row > .form {{ background: none !important; border: 0 !important; box-shadow: none !important; }}
/* Its label beside its box, one line with the button: "Extra epochs ⓘ [ 10 ]  Check checkpoint". */
{_PAGE} #extend_epochs {{ padding: 0 !important; background: none !important; border: 0 !important; box-shadow: none !important; flex: none !important; min-width: max-content !important; }}
{_PAGE} #extend_epochs label.container {{ display: flex; align-items: center; gap: 8px; flex-wrap: nowrap; }}
{_PAGE} #extend_epochs [data-testid="block-info"] {{ margin: 0 !important; }}
{_PAGE} #extend_epochs input[type="number"] {{ width: 96px; }}
{_PAGE} #shep-train-check {{ align-self: flex-end; width: auto !important; min-width: max-content !important; }}
{_PAGE} #shep-train-parents-note {{ margin-top: -4px; }}
/* The base form's line on the reference settings (decision 5), and a value Start would refuse. */
{_PAGE} .shep-differs p {{ margin: 0; font-size: 14px; color: var(--shep-text-2); overflow-wrap: anywhere; }}
{_PAGE} .shep-differs p.shep-amber-t {{ margin-top: 4px; color: var(--shep-amber-800); }}
@media (min-width: 1400px) and (min-height: 760px) {{
  {_PAGE} > .column {{ height: 100%; min-height: 0 !important; }}
  {_PAGE} #shep-train {{ flex: 1 1 auto; min-height: 0; height: 100%; flex-wrap: nowrap !important; }}
  {_PAGE} #shep-train-page {{ flex: 1 1 auto; min-height: 0; height: auto; }}
  {_PAGE} #shep-train-params-bd, {_PAGE} #shep-train-drawer-bd {{ flex: 1 1 auto !important; min-height: 0; overflow: auto; flex-wrap: nowrap !important; }}
  {_PAGE} #shep-train-params-bd > *, {_PAGE} #shep-train-drawer-bd > * {{ flex-shrink: 0 !important; }}
  /* The fields as dense as drawn, so the three groups fit at 1920 × 1080 without scrolling. */
  {_PAGE} #shep-train-params-bd .form > .block {{ padding-top: 4px !important; padding-bottom: 4px !important; }}
  {_PAGE} #shep-train-params-bd .form > .block [data-testid="block-info"] {{ margin-bottom: 4px !important; padding-top: 2px !important; padding-bottom: 2px !important; }}
  {_PAGE} #shep-train-params-bd .form > .block input[type="number"], {_PAGE} #shep-train-params-bd .form > .block input[type="text"] {{ padding-top: 6px !important; padding-bottom: 6px !important; }}
  /* The Continue drawer as dense: it fits at 1920 x 1080 with Extra epochs on screen, as
     drawn (M5-3: 821 px of content in 684); a shorter window scrolls its body inside. */
  {_PAGE} #shep-train-drawer-bd {{ gap: 8px !important; }}
  {_PAGE} #shep-train-drawer-bd .block.padded {{ padding-top: 4px !important; padding-bottom: 4px !important; }}
  {_PAGE} #shep-train-drawer-bd [data-testid="block-info"] {{ margin-bottom: 4px !important; padding-top: 2px !important; padding-bottom: 2px !important; }}
  {_PAGE} #shep-train-drawer-bd .wrap label {{ padding-top: 4px !important; padding-bottom: 4px !important; }}
  {_PAGE} #shep-train-drawer-bd input[type="number"], {_PAGE} #shep-train-drawer-bd input[type="text"] {{ padding-top: 6px !important; padding-bottom: 6px !important; }}
  {_PAGE} #shep-train-drawer-bd .shep-cmp, {_PAGE} #shep-train-drawer-bd .shep-check {{ padding: 8px 12px; }}
  /* In every check result at 1920 x 1080 (M5-3: the first check of a finished parent and a
     parent whose dataset differs overflowed by 20 and 118 px): the source's label beside its
     choices, and the results' lines closer. */
  {_PAGE} #continue_source, {_PAGE} #continue_device {{ display: flex !important; flex-wrap: wrap; align-items: center; gap: 2px 10px; }}
  {_PAGE} #continue_source > [data-testid="block-info"], {_PAGE} #continue_device > [data-testid="block-info"] {{ margin: 0 !important; }}
  /* The run and checkpoint selects keep their labels above them, so each box has the drawer's
     width for its option's few words (the run's Refresh beside its box). */
  {_PAGE} #shep-train-drawer-bd {{ gap: 6px !important; }}
  {_PAGE} #shep-train-drawer-why {{ padding: 8px 16px !important; }}
  {_PAGE} #shep-train-drawer-why .html-container, {_PAGE} #shep-train-drawer-why .shep-html {{ padding: 0 !important; }}
  {_PAGE} #shep-train-drawer-bd .shep-cmp, {_PAGE} #shep-train-drawer-bd .shep-check {{ font-size: 14px; line-height: 1.4; }}
  {_PAGE} #shep-train-drawer-bd .shep-cmp p, {_PAGE} #shep-train-drawer-bd .shep-check p {{ margin-top: 2px; }}
  {_PAGE} #shep-train-drawer-bd .shep-check-h {{ margin-bottom: 2px; }}
  {_PAGE} #shep-train-drawer-bd .shep-inherit {{ margin-top: 4px; }}
  {_PAGE} #shep-train-drawer-bd .shep-inherit > summary {{ padding: 4px 10px; }}
  /* Never clipped, and never cramped: each of the four plots keeps a plot area of at least
     PLOT_MIN_HEIGHT px whatever its title's keys take (the minimum is the plot box's, and each
     box around it grows to hold its content). The run view is three rows: its header and
     state line, the charts, and the resources line. The charts take what the other two leave
     (the GB10 WebUI recheck: the resources line, kept over the plots, hid the last axis); when
     their minimum does not fit, the charts scroll inside their own row, the toolbar staying at
     its top, and nothing is ever drawn over an axis. */
  /* (Gradio writes "overflow: visible" on the block itself, inline: only !important wins.) */
  {_PAGE} #shep-train-runview > #shep-train-charts {{ flex: 1 1 0 !important; min-height: 0 !important; overflow-y: auto !important; display: flex; flex-direction: column; }}
  {_PAGE} #shep-train-charts .shep-chart-plot {{ min-height: {PLOT_MIN_HEIGHT + PLOT_AXIS_ROOM}px; }}
  /* The boxes between the charts' row and the grid take the row's height (min-height 0), so
     the grid's two rows share it, each at least what its own cards need (a card keeps its
     content's minimum: title, keys, the plot's minimum, the axis label); rows that need more
     than the row has overflow it, and the row scrolls. */
  {_PAGE} #shep-train-charts .shep-charts, {_PAGE} #shep-train-charts .shep-charts-body, {_PAGE} #shep-train-charts .shep-charts-grid {{ min-height: 0; }}
  {_PAGE} #shep-train-charts .shep-chart {{ min-height: auto; }}
  {_PAGE} #shep-train-charts .shep-charts-grid {{ grid-auto-rows: minmax(min-content, 1fr); }}
  {_PAGE} #shep-train-runview > #shep-train-rv-top {{ position: relative; z-index: 6; }}
  {_PAGE} #shep-train-runview > #resource_display {{ flex: none !important; }}
  {_PAGE} #shep-train-charts .shep-charts-bar {{ position: sticky; top: 0; z-index: 5; background: var(--shep-panel); }}
  {_PAGE} #shep-train-charts > .html-container {{ flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }}
  {_PAGE} #shep-train-charts > .html-container > div {{ flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }}
  {_PAGE} .shep-rb-row1 {{ flex-wrap: nowrap !important; }}
  {_PAGE} #shep-train-chip {{ flex: 1 1 auto !important; }}
  {_PAGE} #shep-train-charts .shep-charts {{ flex: 1 1 auto; }}
  {_PAGE} #shep-train-runview {{ container-type: inline-size; container-name: shep-runview; }}
}}
@media (min-width: 1400px) and (min-height: 760px) and (max-height: 899px) {{
  {_PAGE} .shep-rv-hd {{ min-height: 40px; padding-top: 6px; padding-bottom: 6px; }}
  {_PAGE} .shep-rv-snap {{ padding-top: 6px; }}
  {_PAGE} #shep-train-charts {{ padding-top: 6px !important; padding-bottom: 4px !important; }}
  {_PAGE} .shep-charts-bar {{ min-height: 34px; padding-bottom: 4px; }}
  {_PAGE} .shep-charts-btn {{ height: 30px; }}
  {_PAGE} .shep-charts-icon {{ height: 28px; }}
  {_PAGE} .shep-charts-grid {{ row-gap: 6px; }}
  {_PAGE} .shep-rv-res {{ padding-top: 6px; padding-bottom: 6px; }}
}}
@media (max-width: 1399px), (max-height: 759px) {{
  {_PAGE} #shep-train-params-bd .gr-group, {_PAGE} #shep-train-params-bd .gr-group > .styler, {_PAGE} #shep-train-params-bd .gr-group .styler > .form {{ background: transparent !important; border: 0 !important; box-shadow: none !important; }}
  {_PAGE} #shep-train-params-bd .gr-group .styler > .form {{ display: grid !important; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 8px 16px !important; border: 0 !important; }}
  {_PAGE} #shep-train-chip {{ flex: 1 1 240px !important; }}
  {_PAGE} .shep-rb-reason {{ justify-content: flex-start; }}
  {_PAGE} .shep-rb-why {{ text-align: left; }}
  {_PAGE} .shep-banner-tail {{ max-height: none; overflow: visible; }}
}}
@media (max-width: 1199px) {{
  {_PAGE} #shep-train-reason {{ flex: 1 1 100% !important; order: 10; }}
  {_PAGE} #shep-train-start {{ margin-left: auto; }}
  {_PAGE} #run_name .input-container {{ width: 200px; }}
}}
"""


__all__ = ["PAGE_CSS", "create_training_tab", "training_chart_csv", "training_chart_series",
           "training_charts"]
