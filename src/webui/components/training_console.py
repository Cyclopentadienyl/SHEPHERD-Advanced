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
training"). A failed run's error banner spans the page with its output tail.

**Start sends the full dataset fingerprint it showed** (the session's
``gr.State``, never the client's); the job runner reads the manifest again and
refuses a different one before anything is written or started
(`TrainingManager.start_training`'s ``expected_dataset_fingerprint``, the one
start path the API route uses too).

**Parameters "for the next run"**: the same fields, groups, defaults and
validation as before (Basic open, Advanced and Expert closed), their help
behind the shared ⓘ (`FIELD_HELP`). The Data Paths section and its path save
are gone: the workspace is the run bar's select.

**Continue…** opens a drawer over the parameters column: the parent list is
the run bar's workspace's runs only (`server_lists.list_checkpoints`, an
operator control since it opens every checkpoint), or a typed path; Check
checkpoint shows what continuing does (the shared run plan) and compares the
parent's recorded dataset fingerprint with the workspace's current one. While
it is open the run bar's actions are Start (disabled, with the reason),
🔒 Continue Training (disabled while the datasets differ, with the reason) and
Cancel. A listed parent is resolved by the server from its list again (never a
client path) and its bytes compared with the digest listed; the runner and the
training process still decide at start.

**The run view**: the run the job runner started last, or, before any since
the server started, the latest run of the run bar's workspace; its header
says which dataset fingerprint the run used (from its checkpoint: operators
only; a read-only request sees "not shown"), the snapshot sentence (operators:
where it can be continued from), the shared chart component
(`src/webui/components/charts.py`: Loss, MRR, Hits@K and the learning rate on
a log axis, one 2×2 figure sharing the epoch axis; the data from the run's
metrics log through the component's server function) and the resources line.

**Per browser session, no module state:** the selection, the drawer, the last
values sent and the banners a session dismissed are ``gr.State``. The page is
refreshed when it is shown (``ctx.on_page_open``) and by the shell's timers
(outside ``gr.Tabs``: ``ctx.timer_fast`` for the run bar and the chart token,
``ctx.timer_slow`` for the resources line), both ``queue=False``, doing no work
while another page is shown (``ctx.location``). Each tick sends only what
changed, and everything again every `FULL_SEND_TICKS` ticks.

**D8:** Start, Continue…, Continue Training, Check checkpoint, the parent list
and Stop (its confirmation reads the run's checkpoint) call the Gradio helper
first, on their own request, then the shared service in process; each is
followed by the chips' read. A read-only request sees every action disabled
with D8's reason as text, and no checkpoint path, snapshot or recorded
fingerprint. The metrics log and the workspaces' JSON records are not
operator-only.

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
import math
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import gradio as gr

from src.api.services.training_manager import training_manager
from src.config.runtime_presets import load_runtime_settings
from src.webui.components import charts
from src.webui.components import renderers as r
from src.webui.components.chips import D8_SHORT, gpu_chip_from, read_lock
from src.webui.components.help import copy_button, help_button
from src.webui.components.workspaces import read_workspaces, redact, workspaces_root
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
DEFAULT_WORKSPACE = "data/workspaces/default"
DISPLAY_PREFIX = "SHEPHERD-Advanced/"

# Max seed value. trainer._set_seed calls np.random.seed(), whose legal range is
# [0, 2**32 - 1] (the binding limit; torch.manual_seed is wider). Generated and
# accepted seeds stay in this range.
_MAX_SEED = 2**32 - 1

# Position of the `seed` value in `_collect_config`'s parameters. `_on_start`
# puts the workspace's directory first and the form's values after it, in this
# order, and swaps in a random seed at this index when "Randomize seed" is on.
SEED_PARAM_INDEX = 7

#: Every this many ticks the run bar is sent whole, whatever the session's
#: record of what it last sent says (a tick and a queued event may cross).
FULL_SEND_TICKS = 8
#: The GPU lock is read at most this often per session (seconds).
LOCK_READ_SECONDS = 5.0
#: The run bar's workspace's runs are listed again at most this often per
#: session (seconds), for the run view's "latest" run.
LATEST_READ_SECONDS = 10.0
#: Lines of a failed run's output the banner shows.
BANNER_TAIL_LINES = 12


def _random_seed() -> int:
    """A fresh seed in NumPy's legal range [0, _MAX_SEED]."""
    return random.randint(0, _MAX_SEED)


def _seed_field_lock(randomize: bool):
    """Lock (grey out) the seed field while "Randomize seed" is on; the random
    seed is generated on Start and written back, so manual editing is disabled."""
    return gr.update(interactive=not bool(randomize))


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


class ConfigValidationError(ValueError):
    """Raised when Training inputs are empty or illegal.

    The message is user-facing: callers surface it as a gr.Warning toast and
    abort without launching training.
    """


def _num(value, label, cast, errors, *, positive=False, lo=None, hi=None):
    """Coerce and range-check one numeric widget value, accumulating a friendly,
    field-named message into ``errors`` instead of leaking a raw
    TypeError/ValueError or Gradio's terse field-less bound error.

    A cleared gr.Number yields None; string/punctuation is already blocked by the
    number input client-side, but the try/except keeps a clean error contract for
    direct/programmatic calls (e.g. tests). ``lo``/``hi`` are inclusive bounds
    (replacing the widgets' minimum/maximum so the message names the field);
    ``positive`` is a strict > 0 check (learning_rate, which has no widget bound).
    Returns the cast value, or None when invalid (the caller aborts on a
    non-empty ``errors`` list).
    """
    if value is None:
        errors.append(f"{label} is empty — enter a value.")
        return None
    try:
        v = cast(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be a number (got {value!r}).")
        return None
    if positive and v <= 0:
        errors.append(f"{label} must be > 0 (got {v}).")
    if lo is not None and v < lo:
        errors.append(f"{label} must be >= {lo} (got {v}).")
    if hi is not None and v > hi:
        errors.append(f"{label} must be <= {hi} (got {v}).")
    return v


def _collect_config(
    # Paths
    data_dir: str,
    run_name: str,
    # Tier 1 — Basic
    num_epochs: int,
    learning_rate: float,
    batch_size: str,
    conv_type: str,
    device: str,
    seed: int,
    # Tier 2 — Advanced
    hidden_dim: str,
    num_layers: int,
    dropout: float,
    weight_decay: float,
    scheduler_type: str,
    warmup_steps: int,
    min_lr_ratio: float,
    early_stopping_patience: int,
    diagnosis_weight: float,
    link_prediction_weight: float,
    contrastive_weight: float,
    ortholog_weight: float,
    # Tier 3 — Expert
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    num_heads: str,
    use_ortholog_gate: bool,
    amp_mode: str,
    temperature: float,
    label_smoothing: float,
    margin: float,
    num_neighbors_str: str,
    max_subgraph_nodes: int,
) -> Dict[str, Any]:
    """Collect all parameter widgets into a config dict. `data_dir` is the
    workspace's directory, as the server resolved it from its list."""
    errors: List[str] = []

    # Parse comma-separated neighbors. An unparseable, empty or illegal entry surfaces as a
    # validation error rather than silently falling back to the default — a silent fallback
    # would start a run with a fan-out the user never asked for. Mirrors the API policy
    # (non-empty, each entry >= 1).
    num_neighbors: List[int] = []
    try:
        num_neighbors = [int(x.strip()) for x in num_neighbors_str.split(",") if x.strip()]
    except (ValueError, AttributeError):
        errors.append('Num Neighbors must be comma-separated integers (e.g. "15, 10, 5").')
    else:
        if not num_neighbors:
            errors.append('Num Neighbors must not be empty (e.g. "15, 10, 5").')
        elif any(n < 1 for n in num_neighbors):
            errors.append("Num Neighbors entries must each be >= 1.")

    # Strip the display prefix — training subprocess uses paths relative to project root
    def _strip_prefix(path: str) -> str:
        p = path.strip()
        if p.startswith("SHEPHERD-Advanced/"):
            p = p[len("SHEPHERD-Advanced/"):]
        return p

    # Numeric gr.Number fields can arrive as None when a user clears them; route
    # them through _num so an empty/illegal value becomes a friendly toast instead
    # of a raw TypeError. Sliders and (string-choice) dropdowns can't be None, so
    # they keep their bare casts. learning_rate is the one Number with no widget
    # minimum, so it also gets a >0 check (aligns with the API's Field(gt=0)).
    config: Dict[str, Any] = {
        "data_dir": _strip_prefix(data_dir) or "data/workspaces/default",
        # The run writes into a directory of its own, allocated under the
        # workspace (contract M3a-2); the name is optional and for people.
        "run_name": (run_name or "").strip(),
        # Tier 1
        "num_epochs": _num(num_epochs, "Epochs", int, errors, lo=1, hi=10000),
        "learning_rate": _num(learning_rate, "Learning Rate", float, errors, positive=True),
        "batch_size": int(batch_size),
        "conv_type": conv_type,
        "device": device,
        "seed": _num(seed, "Seed", int, errors, lo=0, hi=_MAX_SEED),
        # Tier 2
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "weight_decay": float(weight_decay),
        "scheduler_type": scheduler_type,
        "warmup_steps": _num(warmup_steps, "Warmup Steps", int, errors, lo=0),
        "min_lr_ratio": _num(min_lr_ratio, "Min LR Ratio", float, errors, lo=1e-4, hi=1.0),
        "early_stopping_patience": _num(early_stopping_patience, "Early Stopping Patience", int, errors, lo=1),
        "diagnosis_weight": float(diagnosis_weight),
        "link_prediction_weight": float(link_prediction_weight),
        "contrastive_weight": float(contrastive_weight),
        "ortholog_weight": float(ortholog_weight),
        # Tier 3
        "gradient_accumulation_steps": _num(gradient_accumulation_steps, "Gradient Accumulation Steps", int, errors, lo=1),
        "max_grad_norm": _num(max_grad_norm, "Max Grad Norm", float, errors, lo=0.01),
        "num_heads": int(num_heads),
        "use_ortholog_gate": use_ortholog_gate,
        "use_amp": False if (conv_type == "hgt" or amp_mode == "Off") else True,
        "amp_dtype": amp_mode if amp_mode != "Off" else "float16",
        "temperature": float(temperature),
        "label_smoothing": float(label_smoothing),
        "margin": float(margin),
        "num_neighbors": num_neighbors,
        "max_subgraph_nodes": _num(max_subgraph_nodes, "Max Subgraph Nodes", int, errors, lo=100),
        # torch.compile is configured in the Runtime Settings tab (per-run, no
        # restart). Read its persisted value here so it flows into the training
        # config exactly once, sourced from a single place.
        "compile": bool(load_runtime_settings().get("torch_compile", False)),
    }

    if errors:
        raise ConfigValidationError(
            "Cannot start training — please fix:\n• " + "\n• ".join(errors)
        )

    return config


# =============================================================================
# The run bar's workspace (the session's selection)
# =============================================================================
def _empty_workspace() -> Dict[str, Any]:
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


def _workspace_state(name: Optional[str], listed: Optional[Mapping[str, Any]] = None
                     ) -> Dict[str, Any]:
    """The session's run-bar workspace, read now (or from `listed`, a
    `read_workspaces` answer just read): the server's list (the dashboard's one
    reader, every reason without an absolute path) with the options' labels,
    the selected item, what its manifest binds, the listed fingerprints (to say
    what a parent trained on) and its latest run. The full fingerprint kept
    here is the one Start sends."""
    from src.webui.components.datasets_page import manifest_facts

    listed = listed if listed is not None else read_workspaces()
    rows = listed["rows"]
    state = _empty_workspace()
    state["choices"] = [list(choice) for choice in r.workspace_choices(rows)]
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


def _fingerprint(ws: Optional[Mapping[str, Any]]) -> Optional[str]:
    row = (ws or {}).get("row") or {}
    value = row.get("dataset_fingerprint")
    return value if r.is_digest(value) else None


def _dataset_choices_update(ws: Mapping[str, Any], *, interactive: bool = True) -> Any:
    choices = [tuple(c) for c in ws.get("choices") or []]
    return gr.update(choices=choices, value=ws.get("name"), interactive=interactive)


# --------------------------------------------------------------------------- the chip
_LOCK = ('<svg class="shep-lock" width="12" height="12" viewBox="0 0 16 16" aria-hidden="true" '
         'focusable="false"><rect x="3" y="7" width="10" height="7.5" rx="1.6" '
         'fill="currentColor"/><path d="M5.2 7V5.2a2.8 2.8 0 0 1 5.6 0V7" fill="none" '
         'stroke="currentColor" stroke-width="1.6"/></svg>')
_NO = ('<svg class="shep-no" width="14" height="14" viewBox="0 0 16 16" aria-hidden="true" '
       'focusable="false"><circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" '
       'stroke-width="1.5"/><path d="M4.6 8h6.8" stroke="currentColor" stroke-width="1.5"/></svg>')


def details_html(ws: Mapping[str, Any]) -> str:
    """The chip's "Details ›" popover: the full dataset fingerprint with Copy,
    what the manifest binds (12 hex each), the split, the sources, the path
    relative to the server's workspace root, and "All datasets ›"."""
    row = ws.get("row") or {}
    name = str(row.get("name") or "")
    record = (f'<span class="shep-tag shep-tag-ok">{r._CHECK} record verified</span>'
              if r.record_verified(row) else r.chip("record not verified", "grey"))
    parts = [f'<div class="shep-pop-hd"><b>{r.e(name)}</b>{record}'
             '<button type="button" class="shep-pop-x" data-act="close" '
             'aria-label="Close the dataset details">×</button></div>']
    fingerprint = row.get("dataset_fingerprint")
    if r.is_digest(fingerprint):
        parts.append('<div class="shep-digest"><span class="shep-dl">Dataset fingerprint · '
                     f'SHA-256 of split_manifest.json</span><code class="shep-full">'
                     f'{r.e(fingerprint)}</code>'
                     f'{copy_button(fingerprint, "the full dataset fingerprint of " + name)}</div>')
    facts = ws.get("facts")
    rows_kv: List[Tuple[str, str]] = []
    if facts:
        bound = "".join(f'<span class="shep-k">{r.e(label)}</span><code>{r.e(r.short_hex(d))}</code>'
                        for label, d in facts["bound"])
        parts.append('<div class="shep-pop-sub">The manifest binds '
                     '<span class="shep-muted">(sha256, first 12)</span></div>'
                     f'<div class="shep-kv shep-kv-2">{bound}</div>')
        split = []
        if facts.get("allocation_seed") is not None:
            split.append(f"allocation seed {facts['allocation_seed']}")
        if facts.get("feature_seed") is not None:
            split.append(f"feature seed {facts['feature_seed']}")
        for label, key in (("train", "train_samples"), ("val", "val_samples")):
            value = row.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                split.append(f"{label} {value:,}")
        rows_kv.append(("Split", r.e(" · ".join(split) or "not recorded")))
    elif row:
        why = row.get("not_trainable") or "its split manifest was not read"
        parts.append(f'<div class="shep-reason">The manifest was not read: {r.e(why)}</div>')
    sources = r.sources_text(row.get("declared_releases"))
    origin = r.origin_of(row)
    rows_kv.append(("Sources", r.e(sources) if sources else
                    ("none (synthetic)" if origin == "synthetic" else '<span class="shep-muted">—</span>')))
    built = r.built_text(row.get("build"))
    rows_kv.append(("Built", r.e(built.replace("built ", "", 1)) if built else "not recorded"))
    rows_kv.append(("Path", f'<code>{r.e(name)}</code> <span class="shep-muted">(in the server’s '
                            'workspace root)</span>'))
    parts.append('<div class="shep-kv">' + "".join(
        f'<span class="shep-k">{r.e(k)}</span><span class="shep-v">{v}</span>' for k, v in rows_kv)
        + "</div>")
    parts.append('<button type="button" class="shep-link" data-act="datasets">All datasets ›</button>')
    return "".join(parts)


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


#: The chip's popover and cross-link (element-scoped listeners; the chip is
#: never re-mounted, and a replaced value keeps them, being delegated).
_CHIP_JS = r"""
function pop() { return element.querySelector('.shep-pop'); }
function toggle(open) {
  var p = pop(), b = element.querySelector('[data-act="details"]');
  if (!p || !b) { return; }
  p.hidden = !open;
  b.setAttribute('aria-expanded', open ? 'true' : 'false');
  if (open) { var c = p.querySelector('.shep-copy'); if (c) { c.focus(); } }
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
_TRIGGER_JS = r"""
element.addEventListener('click', function (e) {
  var a = e.target.closest ? e.target.closest('[data-act]') : null;
  if (a) { trigger('click', {act: a.getAttribute('data-act'), key: a.getAttribute('data-key') || ''}); }
});
"""

#: A wide window at least this tall shows a failed run's output tail open;
#: a shorter one shows its summary line, so the charts keep their room.
BANNER_OPEN_HEIGHT = 1000

#: The banner: Dismiss (`_TRIGGER_JS`) and its output tail's disclosure, sent
#: open and closed here once per banner on a wide window under
#: `BANNER_OPEN_HEIGHT` px; once the user opens or closes it, their choice
#: holds for the banners that follow on this page.
_BANNER_JS = _TRIGGER_JS + r"""
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


#: Pill classes: never green for a failed, stopped or cancelled run.
PILL_KIND = {"idle": "idle", "starting": "accent", "running": "running", "stopping": "accent",
             "completed": "ok", "stopped": "grey", "failed": "red"}


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
    mark = {"ok": r._CHECK, "red": '<span class="shep-dot-r" aria-hidden="true"></span>',
            "running": '<span class="shep-dot-w" aria-hidden="true"></span>'}.get(kind, "")
    return (f'<div class="shep-pill shep-pill-{kind}" data-state="{r.e(state)}" role="status">'
            f'{mark}<span>{r.e(words)}</span></div>')


def _gpu_line(active: Optional[Mapping[str, Any]], lock: Optional[Mapping[str, Any]],
              test: Optional[Mapping[str, Any]], *, operator: bool) -> Optional[Dict[str, Any]]:
    """The GPU job holding the one slot when it is not this page's training
    job, from the job runner (a test run) or the GPU lock's holder record (a
    CLI, or a lock that cannot be used): ``{"text", "test"}`` or None."""
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
        view["reason"] = line(_LOCK, f"Operator controls only: {short}")
        view["drawer_reason"] = view["reason"]
        return view
    if training_busy:
        view["stop"] = s == "running" and not confirming(status, mode)
        view["reason"] = line(_NO, "Start and Continue… disabled: a training job is running."
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
        view["reason"] = line(_NO, "Start and Continue… disabled: select a dataset.")
        view["drawer_reason"] = view["reason"]
        return view
    if not row.get("trainable"):
        view["reason"] = line(_NO, f"Start and Continue… disabled: {row.get('name')} is not "
                                   f"trainable: {r.not_trainable_short(row)}.")
        view["drawer_reason"] = view["reason"]
        return view
    if not drawer:
        view["start"] = view["cont"] = True
        view["reason"] = line(_NO, "Stop disabled: no training is running.")
        return view
    parent = mode.get("parent") or {}
    why = continue_refusal(parent, ws)
    view["resume"] = why is None
    text = "Start disabled: close Continue… to start a new run."
    if why:
        text += f" Continue Training disabled: {why}"
        view["drawer_reason"] = line(_NO, f"Continue Training, in the run bar above, is disabled: {why}")
    view["reason"] = line(_NO, text)
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
    recorded, current = parent.get("fingerprint"), _fingerprint(ws)
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


def _render_error_block(message: str, *, headline: str = "Training failed",
                        run: Optional[str] = None) -> str:
    """The red block: the headline, the run and where its log is, Dismiss,
    and the output tail (monospace, escaped) in a disclosure, open as sent;
    the banner's script closes it on a wide window under `BANNER_OPEN_HEIGHT`
    px tall, where it would squeeze the charts (`_BANNER_JS`)."""
    first, _sep, tail = str(message).partition("\n\n--- Last output ---\n")
    lines = tail.splitlines()[-BANNER_TAIL_LINES:] if tail else []
    where = (f'<span class="shep-banner-where">run <code>{r.e(run)}</code> · full log in its '
             '<code>logs/</code></span>' if run else "")
    count = f'{len(lines)} line{"" if len(lines) == 1 else "s"}'
    body = ('<details class="shep-banner-more" open><summary>Last output '
            f'<span class="shep-banner-n">({count})</span></summary>'
            f'<pre class="shep-banner-tail">{r.e(chr(10).join(lines))}</pre></details>'
            if lines else "")
    return ('<div class="shep-banner shep-banner-red" role="alert">'
            f'<div class="shep-banner-hd"><span class="shep-banner-h">⚠ {r.e(headline)}: '
            f'{r.e(first.strip())}</span>{where}'
            '<button type="button" class="shep-btn shep-banner-x" data-act="dismiss">× Dismiss</button>'
            f'</div>{body}</div>')


def _render_warning_block(message: str) -> str:
    """The amber block for a non-fatal advisory."""
    return ('<div class="shep-banner shep-banner-amber" role="status">'
            f'<div class="shep-banner-hd"><span class="shep-banner-h">⚠ Warning: {r.e(message)}</span>'
            '<button type="button" class="shep-btn shep-banner-x" data-act="dismiss">× Dismiss</button>'
            '</div></div>')


def _build_error_banner_payload(
    status_info: Mapping[str, Any], pyg_warning: Optional[str], *, run: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """``(html, key)`` for the banner, or ``(None, None)`` when nothing must
    show. ``key`` names the content, so a session's dismissal holds until it
    changes (a new or different error shows again)."""
    blocks: List[str] = []
    key_parts: List[str] = []
    error_msg = status_info.get("error_message")
    if error_msg:
        epoch = _epochs(status_info)
        headline = f"Training failed at {epoch}" if epoch else "Training failed"
        blocks.append(_render_error_block(error_msg, headline=headline, run=run))
        key_parts.append("ERR:" + str(error_msg))
    if pyg_warning:
        blocks.append(_render_warning_block(pyg_warning))
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


def _runner_run(status: Mapping[str, Any], cache: Dict[str, Any]) -> Optional[RunRef]:
    """The run the job runner started last, named by its listed workspace
    (``<workspace>/<arch>/<run>``) or, when the workspace is not listed, by
    its job (``job:<id>``). The workspace's name is looked up once per run
    and session (one listing of the root)."""
    run_dir = training_manager.run_dir
    if run_dir is None or (status.get("status") or "idle") == "idle":
        return None
    key = str(run_dir)
    named = cache.get("runner_ws")
    if not named or named[0] != key:
        from src.api.services.server_lists import ServerListRefused, resolve_workspaces

        try:
            directories = resolve_workspaces()
        except ServerListRefused:
            directories = {}
        name = None
        try:
            resolved = Path(run_dir).resolve()
        except OSError:
            resolved = Path(run_dir)
        for listed, directory in directories.items():
            if directory in resolved.parents:
                name = listed
                break
        named = (key, name)
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
    joining the id onto a path), inside that workspace; or ``job:<id>``, the
    job runner's latest training job."""
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
    from src.config.model_types import SUPPORTED_CONV_TYPES
    from src.utils.checkpoint_paths import RUNS_DIRNAME

    if arch not in SUPPORTED_CONV_TYPES:
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


def run_head_html(run: Optional[RunRef], ws: Mapping[str, Any], *, operator: bool,
                  snapshot: Optional[Mapping[str, Any]]) -> str:
    """"Run gat/<run>" and the dataset fingerprint the run used: from its
    checkpoint for an operator ("✓ same as the run bar" / amber "≠ not the
    selected dataset" / "recorded at the first checkpoint"); for a read-only
    request "not shown · reading the checkpoint is an operator control"."""
    if run is None:
        return ('<div class="shep-panel-hd shep-rv-hd"><h2>Run</h2><span class="shep-muted">'
                'none to show yet</span></div>')
    latest = (f'<span class="shep-muted">latest in {r.e(run["workspace"])}</span>'
              if run.get("latest") else "")
    label = "used dataset fingerprint"
    if not operator:
        used = (f'<span class="shep-fpk">{label}:</span><span class="shep-tag shep-tag-grey">'
                f'{_LOCK} not shown · reading the checkpoint is an operator control</span>')
    elif snapshot is None:
        used = f'<span class="shep-fpk">{label}:</span><span class="shep-muted">recorded at the first checkpoint</span>'
    elif not snapshot.get("readable", True):
        used = f'<span class="shep-fpk">{label}:</span><span class="shep-muted">not read (the checkpoint cannot be read)</span>'
    else:
        recorded = snapshot.get("dataset_fingerprint")
        current = _fingerprint(ws)
        used = r.fingerprint_html(label, recorded, digits=ws.get("digits") or r.SHORT_HEX, copy=False)
        if r.is_digest(recorded) and current:
            used += (f'<span class="shep-tag shep-tag-ok">{r._CHECK} same as the run bar</span>'
                     if recorded == current else
                     '<span class="shep-tag shep-tag-amber">≠ not the selected dataset</span>')
    return (f'<div class="shep-panel-hd shep-rv-hd"><h2>Run <code>{r.e(run["label"])}</code></h2>'
            f'{latest}<span class="shep-rv-used">{used}'
            f'{help_button("training-used-dataset", "Used dataset fingerprint")}</span></div>')


def run_snapshot_html(run: Optional[RunRef], status: Mapping[str, Any], *, operator: bool,
                      snapshot: Optional[Mapping[str, Any]]) -> str:
    """The snapshot sentence: for an operator, where the run can be continued
    from (its checkpoint, named relative to the workspaces root); for a
    read-only request, the run's state and why the rest is not shown."""
    if run is None:
        return ('<p class="shep-rv-snap">No training job since the server started, and the run '
                'bar’s dataset has no runs yet: start one with the parameters on the left.</p>')
    logs = "Logs: the run directory’s <code>logs/</code>."
    s = (status.get("status") or "idle") if not run.get("latest") else "latest"
    live = s in ("running", "stopping")
    if not operator:
        state = (f"Running: {_epochs(status)}." if live and _epochs(status) else
                 "Running." if live else
                 f"{pill_state(status)[1].split(' · ')[0].capitalize()} ({_epochs(status)})."
                 if s in ("completed", "stopped", "failed") and _epochs(status) else "")
        return (f'<p class="shep-rv-snap">{r.e(state)} The checkpoint path and snapshot are shown '
                'where the operator controls are: reading the run’s checkpoint is checkpoint '
                'inspection. Logs: the run directory’s <code>logs/</code> on the server.</p>')
    if live:
        newest = _newest_checkpoint(run["path"])
        where = (f" The newest checkpoint file is {r.e(_shown_path(newest))}." if newest
                 else " No checkpoint has been written yet.")
        epochs = _epochs(status)
        return (f'<p class="shep-rv-snap">Running{": " + r.e(epochs) if epochs else ""}.{where} '
                f'{logs}</p>')
    sentence = _snapshot_sentence(dict(snapshot) if snapshot else None)
    return f'<p class="shep-rv-snap">{r.e(sentence)} · {logs}</p>'


# =============================================================================
# The charts: data from the run's metrics log (the server function)
# =============================================================================
#: The four charts, in figure order (2×2: Loss, MRR / Hits@K, learning rate).
CHART_TITLES = {
    "loss": "Loss",
    "mrr": "MRR (validation, Trainer top-20)",
    "hits": "Hits@K (validation, Trainer)",
    "lr": "Learning rate (log)",
}
SHARED_AXIS = "epoch (shared axis)"
_HITS = re.compile(r"^val_hits@(\d+)$")


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
    Loss (train, validation), MRR and Hits@K (the Trainer's validation
    figures; a Hits@K appears when the log has it) and the learning rate on a
    log axis. A series with no value yet is left out."""
    train = [e for e in history.get("train") or [] if isinstance(e, Mapping)]
    val = [e for e in history.get("validation") or [] if isinstance(e, Mapping)]
    specs = []

    def spec(chart_id, columns, *, y_scale="linear", x_label=None):
        epochs = sorted(set().union(*[set(c) for _label, c in columns])) if columns else []
        series = [(label, [col.get(x) for x in epochs]) for label, col in columns]
        series = [(label, values) for label, values in series if _has_value(values)]
        return charts.chart_spec(chart_id, CHART_TITLES[chart_id], epochs if series else [],
                                 series, y_scale=y_scale, x_label=x_label)

    specs.append(spec("loss", [("train", _by_epoch(train, "total", "train_loss")),
                               ("validation", _by_epoch(val, "val_loss"))]))
    specs.append(spec("mrr", [("MRR", _by_epoch(val, "val_mrr"))]))
    ks = sorted({int(m.group(1)) for e in val for k in e for m in [_HITS.match(str(k))] if m})
    specs.append(spec("hits", [(f"Hits@{k}", _by_epoch(val, f"val_hits@{k}")) for k in ks],
                      x_label=SHARED_AXIS))
    specs.append(spec("lr", [("learning rate", _by_epoch(train, "learning_rate"))],
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


def _chart_token(run: Optional[RunRef], cache: Dict[str, Any]) -> Dict[str, Any]:
    """The component's token for the run shown: its id and how many rows its
    metrics log holds, counted again only when the log file changed (a stat)."""
    if run is None:
        return charts.token(None, 0, empty="No run to show yet.")
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


def _gpu_view(active, cache: Dict[str, Any], operator: bool) -> Optional[Dict[str, Any]]:
    """The GPU line: the job runner each time; the lock (it probes the flock)
    at most every `LOCK_READ_SECONDS` per session, when the runner holds no job."""
    if active:
        test = training_manager.get_test_status() if active.get("job_type") == "test" else None
        return _gpu_line(active, None, test, operator=operator)
    now = datetime.now().timestamp()
    held = cache.get("lock")
    if not held or now - held[0] >= LOCK_READ_SECONDS:
        try:
            lock = read_lock()
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
    ws = dict(ws or _empty_workspace())
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
        ws = _workspace_state(run["workspace"])
        ws_changed = True
    gpu = None if in_use else _gpu_view(active, cache, access.allowed)
    view = run_bar_view(status, ws, mode, access=access, active=active, gpu=gpu)

    # The run's checkpoint record (the used fingerprint, the snapshot): read
    # for an operator only, and again only when its last.pt changed; while
    # the job runner's run is live, its recorded fingerprint only, once.
    if run is None or not access.allowed:
        snapshot = None
    elif in_use and not run.get("latest"):
        snapshot = _used_while_live(run, cache)
    else:
        snapshot = _snapshot_for(run, cache)
    banner_html, banner_key = _build_error_banner_payload(
        status if s == "failed" else {}, _get_pyg_warning(), run=run["label"] if run else None)
    if banner_html and not access.allowed:
        banner_html = redact(banner_html, workspaces_root()[0])
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
        "head": run_head_html(run, ws, operator=access.allowed, snapshot=snapshot),
        "snap": run_snapshot_html(run, status, operator=access.allowed, snapshot=snapshot),
        "charts": _chart_token(run, cache),
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


def view_updates(rendered: Mapping[str, Any], cache_before: Optional[Mapping[str, Any]], *,
                 full: bool = False) -> Tuple[Any, ...]:
    """The outputs for `VIEW_KEYS` and the cache: a piece that equals what
    this session was last sent is ``gr.update()`` (nothing sent), unless
    `full`."""
    sent = dict((cache_before or {}).get("sent") or {})
    pieces = rendered["pieces"]
    cache = dict(rendered["cache"])
    out = []
    for key in VIEW_KEYS:
        value = pieces[key]
        if key in ("ws", "training_workspace"):
            # The session's own state is written only when this render changed
            # it (a running job's dataset), never back over a newer selection.
            out.append(value if rendered.get("ws_changed") else gr.update())
            continue
        comparable = json.dumps(value, sort_keys=True, default=str)
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
    rendered["cache"]["ticks"] = 0
    return view_updates(rendered, cache, full=True)


def _poll(location, ws, mode, cache, dismissed, request: gr.Request):
    """The shell's fast timer (``queue=False``): while Training is shown, what
    changed in the run bar, the run view and the chart token; nothing while
    another page is shown."""
    if location != TRAINING:
        return tuple(gr.update() for _ in VIEW_KEYS) + (gr.update(),)
    try:
        rendered = render(ws, mode, cache, dismissed, request)
    except Exception:  # noqa: BLE001 - a tick that raises would stop the page's updates
        logger.exception("The Training page could not be read")
        return tuple(gr.update() for _ in VIEW_KEYS) + (gr.update(),)
    ticks = int((cache or {}).get("ticks") or 0) + 1
    rendered["cache"]["ticks"] = ticks
    return view_updates(rendered, cache, full=ticks % FULL_SEND_TICKS == 0)


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
    state = _workspace_state(choice, listed)
    return (state, _dataset_choices_update(state), chip_html(state), choice, {},
            _poll_resources(TRAINING))


def _on_dataset(name, mode, request: gr.Request):
    """A dataset chosen in the run bar: its manifest and record read now (the
    chip shows them), the session's run-bar workspace set (Datasets marks
    it), any check of a parent cleared (the drawer lists the new workspace's
    runs on its next refresh)."""
    state = _workspace_state(name)
    mode = dict(mode or {})
    if mode.get("parent"):
        mode["parent"] = {}
    return state, chip_html(state), state.get("name"), mode, ""


def _on_training_workspace(name, ws, mode):
    """Datasets' "Use for training ›" (``ctx.training_workspace``): the run
    bar selects that workspace (nothing changes on the server)."""
    if not name or name == (ws or {}).get("name"):
        return gr.update(), gr.update(), gr.update(), gr.update()
    state = _workspace_state(name)
    if state.get("name") is None:
        return gr.update(), gr.update(), gr.update(), gr.update()
    mode = dict(mode or {})
    mode["parent"] = {}
    return state, _dataset_choices_update(state), chip_html(state), mode


def _act(evt: Any) -> Optional[str]:
    """The ``act`` a page element's trigger sent (`_TRIGGER_JS`, `_CHIP_JS`)."""
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
    if _act(evt) != "dismiss" or not key:
        return gr.update(), gr.update()
    return [*(dismissed or []), key][-20:], ""


def _workspace_path(name: Any) -> Path:
    """The directory of the workspace the server lists as `name` (found in its
    list again: `server_lists.resolve_workspace`)."""
    from src.api.services.server_lists import resolve_workspace

    return resolve_workspace(name)


def _on_start(request: gr.Request, ws, *args):
    """Start: an operator control (D8), checked first on this event's own
    request; then the form's values (validated as before), the workspace
    resolved by the server from its list, and the job runner's start with the
    full dataset fingerprint this session showed. ``args`` are the form's
    values in `_collect_config`'s order after the directory, then "Randomize
    seed".

    Returns the notice (`notice_html`) and the seed field (echoing a drawn
    seed).
    """
    access = operator_access(request)
    if not access.allowed:
        return notice_html(access.refusal), gr.update()
    said, seed = _start(ws, args)
    return notice_html(said), seed


def _start(ws, args) -> Tuple[str, Any]:
    """Start's work after D8: ``(sentence, seed update)``, the sentence plain
    text ("" once started)."""
    randomize = bool(args[-1])
    row = (ws or {}).get("row") or {}
    fingerprint = _fingerprint(ws)
    if not row or not fingerprint:
        return "Not started: select a dataset from the server's list first.", gr.update()
    if not row.get("trainable"):
        return (f"Not started: {row.get('name')} is not trainable: {r.not_trainable_short(row)}.",
                gr.update())
    from src.api.services.server_lists import ServerListRefused

    try:
        directory = _workspace_path(row["name"])
    except ServerListRefused:
        return (f"Not started: {row['name']} is no longer listed by the server; select a dataset "
                "again.", gr.update())
    params = [str(directory), *args[:-1]]
    if randomize:
        # Swap in a fresh seed BEFORE validation so a locked/empty seed field
        # still launches; the used seed is echoed back to the (locked) field.
        params[SEED_PARAM_INDEX] = _random_seed()
    try:
        config = _collect_config(*params)
    except ConfigValidationError as e:
        gr.Warning(str(e), duration=12)
        return ("Not started: invalid parameters, listed in the notification.", gr.update())
    result = training_manager.start_training(config, expected_dataset_fingerprint=fingerprint)
    if result.get("success"):
        return "", gr.update(value=config["seed"])
    return f"Not started: {result.get('error', 'unknown error')}", gr.update()


def _running_job(status: Mapping[str, Any]) -> Optional[str]:
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
    return isinstance(asked, str) and asked == _running_job(status)


def _on_stop_request(request: gr.Request, mode):
    """Stop's first step: say what stopping loses before it happens. It reads
    the run's checkpoint, so D8 is checked first (refused: nothing is read).
    The confirmation is recorded for the job running now (its id).

    Returns the confirmation's text and the session's mode.
    """
    access = operator_access(request)
    mode = dict(mode or {})
    mode["confirm"] = None
    if not access.allowed:
        return notice_html(access.refusal), mode
    job = _running_job(training_manager.get_status())
    if job is None:
        return notice_html("Nothing to stop: no training job is running."), mode
    snapshot = training_manager.latest_snapshot()
    if snapshot and snapshot.get("readable", True) and snapshot.get("completed_epochs"):
        name = Path(str(snapshot["checkpoint"])).name
        text = (f"<b>Stop training now?</b> The last saved checkpoint is from epoch "
                f"{r.e(snapshot['completed_epochs'])} (<code>{r.e(name)}</code>). Progress since "
                f"then is lost; you can continue from epoch {r.e(snapshot['completed_epochs'])}.")
    else:
        text = ("<b>Stop training now?</b> No checkpoint has been saved yet, so all progress "
                "of this run is lost and nothing can be continued.")
    mode["confirm"] = job
    return f'<div class="shep-rb-confirm-q">{text}</div>', mode


def _on_stop_cancel(mode):
    """Keep training (and any Start: a confirmation left from an earlier job
    is closed)."""
    mode = dict(mode or {})
    mode["confirm"] = None
    return mode


def _on_stop(request: gr.Request, mode):
    """The confirmed Stop: an operator control (D8), checked on this event's
    own request; refused, nothing is stopped. A confirmation asked for another
    job than the one running now stops nothing (it described that job).

    Returns the notice and the session's mode (the confirmation closed).
    """
    mode = dict(mode or {})
    asked = mode.get("confirm")
    mode["confirm"] = None
    access = operator_access(request)
    if not access.allowed:
        return notice_html(access.refusal), mode
    if isinstance(asked, str) and asked != _running_job(training_manager.get_status()):
        return notice_html("Not stopped: the training job you asked to stop is no longer the "
                           "one running; click Stop again if this one should stop."), mode
    result = training_manager.stop_training()
    if result.get("success"):
        return notice_html("Stopped. " + _snapshot_sentence(result.get("outcome"))), mode
    return notice_html(f"Not stopped: {result.get('error', 'unknown error')}"), mode


# --------------------------------------------------------------------------- Continue…
#: The two sources of a parent checkpoint. Both go through the same check.
CONTINUE_FROM_LIST = "From run list"
CONTINUE_FROM_PATH = "Manual path"

#: The continue block's prompt before a check, and after the selection changes.
_CONTINUE_PROMPT = "Select a checkpoint, then click <b>Check checkpoint</b>."


#: What a checkpoint file holds, as its record says (`checkpoint_kind`).
CHECKPOINT_KINDS = {"training_state": "training state", "weights": "weights only"}


def _parent_label(item: Mapping[str, Any]) -> str:
    """A parent's option: the run, the epoch, the file and what it holds
    (a run's ``last.pt`` and its model file of the same epoch differ there),
    and its model fingerprint."""
    epoch = item.get("epoch")
    parts = [r.run_label(item.get("architecture"), item.get("run"))]
    if isinstance(epoch, int) and not isinstance(epoch, bool):
        parts.append(f"epoch {epoch}")
    kind = CHECKPOINT_KINDS.get(item.get("checkpoint_kind") or "")
    parts.append(f"{item.get('filename')} ({kind})" if kind else str(item.get("filename")))
    if item.get("problem") or not r.is_digest(item.get("sha256")):
        parts.append("metadata not readable")
    else:
        parts.append(f"model fingerprint {r.short_hex(item['sha256'])}")
    return " · ".join(parts)


def _list_parents(ws: Mapping[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """The run bar's workspace's checkpoints from the server's list (it opens
    every checkpoint: the caller checked D8): ``({key: item}, problem)``."""
    from dataclasses import asdict

    from src.api.services.server_lists import list_checkpoints

    name = ws.get("name")
    if not name:
        return {}, "Select a dataset in the run bar first."
    listing = list_checkpoints(name)
    if not listing.available:
        return {}, redact(listing.problem, workspaces_root()[0])
    parents: Dict[str, Any] = {}
    for item in listing.items:
        data = asdict(item)
        parents[f"{item.architecture}/{item.run}/{item.filename}"] = data
    return parents, None


def _parents_update(parents: Mapping[str, Any], keep: Optional[str] = None) -> Any:
    choices = [(_parent_label(item), key) for key, item in parents.items()]
    value = keep if keep in parents else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=value)


def _parents_note(ws: Mapping[str, Any], parents: Mapping[str, Any], problem: Optional[str]) -> str:
    if problem:
        return f'<span class="shep-muted">{r.e(problem)}</span>'
    return (f'<span class="shep-muted">{_LOCK} server list · {r.e(ws.get("name"))}’s runs · '
            f'latest first · {len(parents)}</span>')


def _on_continue_open(request: gr.Request, ws, mode):
    """Continue…: the drawer over the parameters, listing the run bar's
    workspace's checkpoints (an operator control: D8 first; refused, the
    drawer stays closed and nothing is opened).

    Returns the notice, the session's mode, the parents (state), the parent
    select, its note, and the check's text."""
    access = operator_access(request)
    mode = dict(mode or {})
    if not access.allowed:
        return (notice_html(access.refusal), mode, gr.update(), gr.update(), gr.update(),
                gr.update())
    parents, problem = _list_parents(ws or {})
    mode.update(drawer=True, confirm=None, parent={})
    update = _parents_update(parents)
    first = update.get("value") if isinstance(update, dict) else None
    mode["parent"] = _chosen(CONTINUE_FROM_LIST, first, "", parents, ws)
    return ("", mode, parents, update, _parents_note(ws or {}, parents, problem),
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
    """The parent list for `ws` as D8 allows it: ``(parents, select, note)``."""
    if not access.allowed:
        return {}, gr.update(choices=[], value=None), f'<span class="shep-muted">{r.e(access.refusal)}</span>'
    parents, problem = _list_parents(ws or {})
    return parents, _parents_update(parents, current), _parents_note(ws or {}, parents, problem)


def _relist_parents(request: gr.Request, ws, mode):
    """The run bar's workspace changed (chosen by hand, or by Datasets' "Use
    for training ›") while Continue… is open: the parent list becomes the new
    workspace's runs, through the Refresh's path (it opens every checkpoint,
    so D8 first, on this event's own request; refused, the list is emptied
    and the note says why), and the last check, the comparison and the Extra
    epochs, which were about the old list, are cleared. With the drawer
    closed nothing is listed (it lists when opened).

    Returns the parents (state), the parent select, its note, the
    comparison, the check's text and the Extra epochs field."""
    access = operator_access(request)
    if not (mode or {}).get("drawer"):
        return (gr.update(),) * 6
    parents, select, note = _parents_listed(access, ws, None)
    return (parents, select, note, "", f'<p class="shep-note">{_CONTINUE_PROMPT}</p>',
            gr.update(visible=False, value=None))


def _on_continue_close(mode):
    mode = dict(mode or {})
    mode.update(drawer=False, parent={})
    return mode


def _chosen(source: str, listed: Any, manual: Any, parents: Mapping[str, Any],
            ws: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The chosen parent as the session records it: whether one is chosen,
    its recorded dataset fingerprint when the list says it, and the run it
    continues (for the run bar's line)."""
    if source == CONTINUE_FROM_PATH:
        path = _strip_display_prefix(manual or "")
        return {"chosen": bool(path), "source": source, "fingerprint": None,
                "run_label": None}
    item = (parents or {}).get(listed) if listed else None
    if not item:
        return {"chosen": False, "source": source}
    return {"chosen": True, "source": source, "fingerprint": item.get("dataset_fingerprint"),
            "run_label": f"{item.get('architecture')}/{item.get('run')}", "key": listed,
            "workspace": item.get("workspace")}


def compare_html(parent: Mapping[str, Any], ws: Mapping[str, Any]) -> str:
    """The parent's recorded dataset fingerprint against the workspace's
    current one: ✓ the same, amber when it differs ("matches no listed
    workspace", or the listed workspace it matches), or what is not known yet."""
    if not parent.get("chosen"):
        return ""
    recorded, current = parent.get("fingerprint"), _fingerprint(ws)
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
        return (f'<div class="shep-cmp shep-cmp-ok">{r._CHECK} The parent trained on this '
                f'workspace’s current dataset · {r.fingerprint_html("dataset fingerprint", recorded, digits=digits, copy=False)}</div>')
    listed = ws.get("listed") or []
    trained = r.trained_on_html(recorded, listed, digits=digits)
    built = r.built_text((ws.get("row") or {}).get("build"))
    rebuilt = (f"<p>This workspace was {r.e(built)}. The start refuses a different dataset.</p>"
               if built else "<p>The start refuses a different dataset.</p>")
    return ('<div class="shep-cmp shep-cmp-amber"><b>⚠ The parent’s dataset is not this '
            f'workspace’s current one</b><p>parent {trained}</p><p>{now}</p>{rebuilt}</div>')


def _check_html(preview: Mapping[str, Any], source: str, *, differs: bool = False) -> str:
    """The check result, what continuing does, and the settings taken from the
    checkpoint (a closed disclosure), as a read-only summary (contract M3a-3;
    not disabled inputs, which screen readers skip)."""
    origin = "manual path" if source == CONTINUE_FROM_PATH else "run list"
    lines = [f'<div><b>Checkpoint</b> ({origin}): <code>{r.e(_shown_path(preview.get("parent", "")))}</code></div>']
    if preview.get("sha256"):
        arch = preview.get("architecture")
        lines.append('<div>' + r.fingerprint_html("model fingerprint", preview["sha256"], copy=False)
                     + (f", architecture <b>{r.e(str(arch).upper())}</b>" if arch else "") + "</div>")
    if not preview.get("ok"):
        return ('<div class="shep-check shep-check-amber"><div class="shep-check-h">⚠ Cannot '
                f'continue.</div>{"".join(lines)}<p>{r.e(preview.get("reason"))}</p></div>')
    head = ('<div class="shep-check-h">⚠ Checkpoint read; cannot continue on the current '
            'dataset.</div>' if differs else
            f'<div class="shep-check-h shep-ok-t">{r._CHECK} {r.e(preview["action"])}.</div>')
    lines.append(f"<p>Plan read from the checkpoint: {r.e(preview['action'])}.</p>")
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
    lines.append('<p class="shep-note">The start checks again against this workspace’s data '
                 'and graph.</p>')
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
    the dataset comparison)."""
    mode = dict(mode or {})
    if not mode.get("drawer"):
        return gr.update(), gr.update()
    mode["parent"] = _chosen(source, listed, manual, parents, ws)
    return mode, compare_html(mode["parent"], ws or {})


def _strip_display_prefix(path: str) -> str:
    """Strip the UI display prefix (``SHEPHERD-Advanced/...`` -> ``...``)."""
    p = (path or "").strip()
    return p[len("SHEPHERD-Advanced/"):] if p.startswith("SHEPHERD-Advanced/") else p


def _parent_path(source: str, listed: Any, manual: Any, ws: Mapping[str, Any],
                 parents: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str],
                                                      Optional[Mapping[str, Any]]]:
    """``(path, why_not, listed_item)`` for the chosen parent: a listed one
    resolved by the server from its list again (never a client path); a typed
    one as typed (an operator's)."""
    if source == CONTINUE_FROM_PATH:
        path = _strip_display_prefix(manual or "")
        return (path or None), (None if path else "enter a checkpoint path"), None
    item = (parents or {}).get(listed) if listed else None
    if not item:
        return None, "choose a checkpoint from the run list", None
    other = _other_workspace(item, ws)
    if other:
        return None, other.rstrip("."), None
    from src.api.services.server_lists import ServerListRefused, resolve_checkpoint

    try:
        path = resolve_checkpoint(ws.get("name"), item["architecture"], item["run"],
                                  item["filename"])
    except ServerListRefused as exc:
        return None, f"it is no longer listed ({redact(exc, workspaces_root()[0])}); refresh the list", None
    return str(path), None, item


def _preview_for(parent: str, device: Optional[str], extend_epochs) -> Dict[str, Any]:
    """The service's preview, with Extra epochs only for a finished
    checkpoint: the field is the console's, and for any other checkpoint a
    value in it is one left from an earlier selection, not a request. The API
    and the CLI send `extend_epochs` as given and are refused as before."""
    epochs = int(extend_epochs) if extend_epochs else None
    return training_manager.preview_continuation(
        parent, device=device or "auto", extend_epochs=epochs, extend_if_finished=True)


def _checked(source, listed, manual, device, extend_epochs, ws, parents, mode):
    """The check every Check and Continue Training runs: the parent resolved,
    the preview, the listed digest compared with the bytes read, the dataset
    compared. Returns (preview or None, why_not, the session's parent record)."""
    path, why, item = _parent_path(source, listed, manual, ws or {}, parents or {})
    record = dict((mode or {}).get("parent") or _chosen(source, listed, manual, parents, ws))
    if path is None:
        record.update(refused=True, checked=True)
        return None, why, record
    preview = _preview_for(path, device, extend_epochs)
    if item is not None and preview.get("sha256") and item.get("sha256") \
            and preview["sha256"] != item["sha256"]:
        preview = {**preview, "ok": False,
                   "reason": "The checkpoint changed since it was listed; refresh the list."}
    recorded = preview.get("dataset_fingerprint") or (item or {}).get("dataset_fingerprint")
    record.update(chosen=True, checked=True, refused=not preview.get("ok"), fingerprint=recorded)
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
    text = _check_html(preview, source, differs=bool(differs))
    if preview.get("end") == "finished" and not preview.get("ok") and not extend_epochs:
        text += '<p class="shep-note">Enter <b>Extra epochs</b> below, then click <b>Check checkpoint</b> again.</p>'
    return text, _extra_epochs_update(preview), mode, compare_html(record, ws or {})


def _on_resume(ws, device, source, listed, manual, extend_epochs, parents, mode,
               request: gr.Request):
    """Continue Training: an operator control (D8, checked first on this
    event's own request; refused, nothing is opened or started), then the same
    check as the preview, the dataset comparison, then the job runner's start
    with the full dataset fingerprint this session showed.

    Only operational fields are sent (contract M3a-3): every learning setting
    comes from the checkpoint, so the training form's values are not used.
    The job runner and the training process each check again.

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
    fingerprint = _fingerprint(ws)
    if not row or not fingerprint:
        return "Not started: select a dataset first.", gr.update(), gr.update(), mode
    preview, why, record = _checked(source, listed, manual, device, extend_epochs, ws, parents, mode)
    mode["parent"] = record
    if preview is None:
        return f"Not started: {why}.", gr.update(), gr.update(visible=False, value=None), mode
    refusal = continue_refusal(record, ws)
    text = _check_html(preview, source, differs=preview.get("ok") and refusal is not None)
    if not preview.get("ok"):
        if preview.get("end") == "finished" and not extend_epochs:
            text += ('<p class="shep-note">Enter <b>Extra epochs</b> below, then click '
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
    result = training_manager.start_training(config, expected_dataset_fingerprint=fingerprint)
    if result.get("success"):
        mode.update(drawer=False, parent={}, confirm=None)
        return "", text, _extra_epochs_update(preview), mode
    return f"Not started: {result.get('error', 'unknown error')}", text, _extra_epochs_update(preview), mode


def _toggle_continue_source(source):
    from_list = source != CONTINUE_FROM_PATH
    return (gr.update(visible=from_list), gr.update(visible=from_list),
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
        "The server's workspace list, the same as Research › Datasets. Choosing one reads its "
        "split manifest and dataset record; nothing changes on the server. Start sends the full "
        "dataset fingerprint shown here, and the server refuses the start if the workspace "
        "changed since.",
    ),
    "run_name": (
        "Run name",
        "Optional. Each run is saved in a new folder of its own: "
        "`<workspace>/runs/<architecture>/<date-time>_<name>_<id>/`. Letters, digits and "
        "hyphens are kept; blank uses \"run\". The run view names the folder once training "
        "starts.",
    ),
    "num_epochs": (
        "Epochs",
        "Smoke test: 2-5 | Quick: 10-50 | Full: 100-500",
    ),
    "learning_rate": (
        "Learning Rate",
        "Initial LR. Recommended: 1e-4 (GAT) | 5e-5 (HGT). Supports scientific notation (e.g."
        " 5e-5).",
    ),
    "batch_size": (
        "Batch Size",
        "Samples per step. Larger = faster but more VRAM. 32 = safe default; 1024 for "
        "large-VRAM GPUs (e.g. GB10); 2048 is experimental (HGT will OOM). Raise LR & warmup "
        "with batch.",
    ),
    "conv_type": (
        "GNN Conv Type",
        "GAT: attention-based (default) | HGT: heterogeneous transformer (no AMP, high VRAM "
        "requirements - batch size ≤16 for 16GB GPUs) | SAGE: mean aggregation",
    ),
    "device": (
        "Device",
        "auto: use GPU if available, otherwise CPU. Continue… has its own Device field, which "
        "starts from this one.",
    ),
    "seed": (
        "Seed",
        "Random seed for reproducibility. Change to try different initializations.",
    ),
    "randomize_seed": (
        "🎲 Randomize seed",
        "On: the seed field is locked and a fresh random seed (0..2^32-1) is drawn on each "
        "Start and shown here. Off: the entered seed is used.",
    ),
    "hidden_dim": (
        "Hidden Dim",
        "GNN embedding dimension. 256 balances quality/speed. 512 for large KGs.",
    ),
    "num_layers": (
        "Num Layers",
        "GNN message-passing depth. 3-4 typical. More layers = wider receptive field.",
    ),
    "dropout": (
        "Dropout",
        "Regularization. 0.1 typical. Increase to 0.2-0.3 if overfitting.",
    ),
    "weight_decay": (
        "Weight Decay",
        "L2 regularization. 0.01 typical for AdamW. Lower (1e-4) if underfitting.",
    ),
    "scheduler_type": (
        "Scheduler",
        "cosine: smooth decay (recommended) | onecycle: aggressive | linear: steady",
    ),
    "warmup_steps": (
        "Warmup Steps",
        "Steps to linearly ramp LR from 0. 500 typical. Set 0 to disable.",
    ),
    "min_lr_ratio": (
        "Min LR Ratio",
        "Floor of LR decay as a fraction of peak (must be > 0; onecycle uses 1/min_lr_ratio)."
        " 0.01 = decay to 1% of peak; raise (e.g. 0.1) to keep learning in late epochs.",
    ),
    "early_stopping_patience": (
        "Early Stopping Patience",
        "Epochs without improvement before stopping. 10 is a good default.",
    ),
    "diagnosis_weight": (
        "Diagnosis Weight",
        "Primary task weight. Keep at 1.0 (anchor). Other weights are relative to this.",
    ),
    "link_prediction_weight": (
        "Link Prediction Weight",
        "KG structure learning. 0.3-0.5 typical. Higher improves KG representation.",
    ),
    "contrastive_weight": (
        "Contrastive Weight",
        "Phenotype-disease similarity learning. 0.2-0.5 typical.",
    ),
    "ortholog_weight": (
        "Ortholog Weight",
        "Cross-species gene mapping. 0.1-0.3 typical. Set 0 to disable.",
    ),
    "gradient_accumulation_steps": (
        "Gradient Accumulation Steps",
        "Simulate larger batch by accumulating N steps. Useful when VRAM-limited.",
    ),
    "max_grad_norm": (
        "Max Grad Norm",
        "Gradient clipping threshold. 1.0 typical. Prevents training instability.",
    ),
    "num_heads": (
        "Attention Heads",
        "Multi-head attention count. 8 typical. Must evenly divide hidden_dim.",
    ),
    "use_ortholog_gate": (
        "Use Ortholog Gate",
        "Learnable gate for cross-species gene signals. Disable if no ortholog data.",
    ),
    "amp_mode": (
        "Mixed Precision (AMP)",
        "Faster & less VRAM on modern GPUs. Off: full float32. float16: wider GPU support. "
        "bfloat16: better numerical stability (RTX 30+, A100+).",
    ),
    "temperature": (
        "Contrastive Temperature",
        "Lower = sharper similarity. 0.07 typical for contrastive learning.",
    ),
    "label_smoothing": (
        "Label Smoothing",
        "Softens one-hot labels. 0.1 typical. Prevents overconfident predictions.",
    ),
    "margin": (
        "Margin",
        "Margin for ranking loss. 1.0 typical. Larger margin = stricter separation.",
    ),
    "num_neighbors_str": (
        "Num Neighbors (comma-separated)",
        "Neighbor samples per GNN layer (deepest to shallowest). E.g. '15, 10, 5'.",
    ),
    "max_subgraph_nodes": (
        "Max Subgraph Nodes",
        "Max nodes in sampled subgraph. Larger = more context but slower. 5000 typical.",
    ),
    "continue_source": (
        "Checkpoint source",
        "From run list: the run bar's workspace's runs, as the server lists them (listing "
        "opens each checkpoint, an operator control). Manual path: a .pt file on the server, "
        "in the run layout or not; it is checked the same way.",
    ),
    "checkpoint_dropdown": (
        "Run checkpoint",
        "The run bar's workspace's runs, latest first, each with its model fingerprint. The "
        "server resolves the one chosen from its list again and compares its bytes with the "
        "fingerprint listed. Refresh lists them again.",
    ),
    "manual_checkpoint": (
        "Checkpoint path",
        "A .pt file, in the run layout or not; checked the same way as a listed one.",
    ),
    "continue_device": (
        "Device",
        "Where the continued run trains: auto uses the GPU if available, otherwise CPU. Check "
        "checkpoint and Continue Training use it; mixed precision takes effect only on CUDA, so "
        "the check says when the checkpoint's setting would not hold on this device. Each time "
        "Continue… opens it starts from the Parameters' Device.",
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
        "Wheel over a chart zooms around the pointer: over the epoch axis only the epochs, "
        "over the value axis only the values, inside the plot both. Drag pans. Double-click "
        "or Reset view shows the whole run again. Keys, with a chart focused: + and − zoom, "
        "x and y zoom one axis (Shift zooms out), the arrows pan, End jumps to the latest "
        "epochs, 0 resets. The four charts share one epoch range; each keeps its own value "
        "range. While zoomed during a live run, a button counts the new epochs beyond the "
        "view. The view is kept per run in this browser tab. In the narrow layout the wheel "
        "scrolls the page.",
    ),
    "training-used-dataset": (
        "Used dataset fingerprint",
        "The dataset fingerprint the run's checkpoint records (the split manifest it trained "
        "on), compared in full with the run bar's dataset. Reading it opens the checkpoint, "
        "which is an operator control (checkpoint inspection); a read-only request is not "
        "shown it. Before the first checkpoint nothing records it yet.",
    ),
}


#: The AMP radio's refusal under HGT: state, so visible text (``info=``), never
#: behind the ⓘ.
AMP_HGT_REFUSAL = "Disabled: HGT requires full float32 (pyg_lib limitation)."


def _on_conv_type_change(ct):
    """HGT requires float32: AMP off and locked, with the refusal shown; any
    other architecture unlocks it and clears the refusal (``""``: Gradio 6.30
    keeps an info sent as None)."""
    if ct == "hgt":
        return gr.update(value="Off", interactive=False, info=AMP_HGT_REFUSAL)
    return gr.update(interactive=True, info="")


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

    ws = gr.State(_empty_workspace())
    mode = gr.State({})
    cache = gr.State({})
    dismissed = gr.State([])
    parents = gr.State({})

    with gr.Column(elem_id="shep-train", elem_classes=["shep-train"]):
        # ------------------------------------------------------------- run bar
        with gr.Column(elem_id="shep-train-runbar", elem_classes=["shep-runbar"]) as runbar:
            with gr.Row(elem_classes=["shep-rb-row", "shep-rb-row1"]):
                dataset = gr.Dropdown(label="Dataset", choices=[], value=None,
                                      elem_id="shep-train-dataset", scale=0, min_width=240,
                                      container=False)
                chip = plain_html(chip_html(None), elem_id="shep-train-chip", js_on_load=_CHIP_JS)
                pill = plain_html(pill_html({"status": "idle"}), elem_id="shep-train-pill")
            with gr.Row(elem_classes=["shep-rb-row", "shep-rb-row2"]):
                run_name = gr.Textbox(label="Run name", value="", max_lines=1, elem_id="run_name",
                                      scale=0, min_width=240, container=False)
                continues = plain_html("", elem_id="shep-train-continues")
                reason = plain_html('<div class="shep-rb-reason"></div>', elem_id="shep-train-reason",
                                    js_on_load=_TRIGGER_JS)
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
                           '<span class="shep-panel-right">for the next run</span></div>',
                           elem_id="shep-train-params-hd")
                with gr.Column(elem_id="shep-train-params-bd", elem_classes=["shep-train-bd"]):
                    with gr.Accordion("Basic Training Parameters", open=True,
                                      elem_id="shep-train-basic"):
                        with gr.Group(elem_classes=["shep-train-grid"]):
                            num_epochs = gr.Number(label="Epochs", value=100, precision=0,
                                                   elem_id="num_epochs")
                            learning_rate = gr.Number(label="Learning Rate", value=1e-4, precision=8,
                                                      elem_id="learning_rate")
                            batch_size = gr.Dropdown(
                                label="Batch Size",
                                choices=["8", "16", "32", "64", "128", "256", "512", "1024", "2048"],
                                value="32", elem_id="batch_size")
                            conv_type = gr.Radio(label="GNN Conv Type", choices=["gat", "hgt", "sage"],
                                                 value="gat", elem_id="conv_type")
                            device = gr.Radio(label="Device", choices=["auto", "cuda", "cpu"],
                                              value="auto", elem_id="device")
                            seed = gr.Number(label="Seed", value=42, precision=0, elem_id="seed")
                            randomize_seed = gr.Checkbox(label="🎲 Randomize seed", value=False,
                                                         elem_id="randomize_seed")
                    with gr.Accordion("Advanced Parameters", open=False):
                        with gr.Group():
                            hidden_dim = gr.Dropdown(label="Hidden Dim", choices=["128", "256", "512"],
                                                     value="256", elem_id="hidden_dim")
                            num_layers = gr.Slider(label="Num Layers", minimum=2, maximum=8, value=4,
                                                   step=1, elem_id="num_layers")
                            dropout = gr.Slider(label="Dropout", minimum=0.0, maximum=0.5, value=0.1,
                                                step=0.01, elem_id="dropout")
                            weight_decay = gr.Slider(label="Weight Decay", minimum=1e-5, maximum=0.1,
                                                     value=0.01, step=1e-5, elem_id="weight_decay")
                            scheduler_type = gr.Dropdown(
                                label="Scheduler", choices=["cosine", "onecycle", "linear", "none"],
                                value="cosine", elem_id="scheduler_type")
                            warmup_steps = gr.Number(label="Warmup Steps", value=500, precision=0,
                                                     elem_id="warmup_steps")
                            min_lr_ratio = gr.Number(label="Min LR Ratio", value=0.01,
                                                     elem_id="min_lr_ratio")
                            early_stopping_patience = gr.Number(
                                label="Early Stopping Patience", value=10, precision=0,
                                elem_id="early_stopping_patience")
                        gr.Markdown("##### Loss Weights")
                        with gr.Group():
                            diagnosis_weight = gr.Slider(label="Diagnosis Weight", minimum=0.0,
                                                         maximum=2.0, value=1.0, step=0.1,
                                                         elem_id="diagnosis_weight")
                            link_prediction_weight = gr.Slider(
                                label="Link Prediction Weight", minimum=0.0, maximum=2.0, value=0.5,
                                step=0.1, elem_id="link_prediction_weight")
                            contrastive_weight = gr.Slider(label="Contrastive Weight", minimum=0.0,
                                                           maximum=2.0, value=0.3, step=0.1,
                                                           elem_id="contrastive_weight")
                            ortholog_weight = gr.Slider(label="Ortholog Weight", minimum=0.0,
                                                        maximum=2.0, value=0.2, step=0.1,
                                                        elem_id="ortholog_weight")
                    with gr.Accordion("Expert Parameters", open=False):
                        with gr.Group():
                            gradient_accumulation_steps = gr.Number(
                                label="Gradient Accumulation Steps", value=1, precision=0,
                                elem_id="gradient_accumulation_steps")
                            max_grad_norm = gr.Number(label="Max Grad Norm", value=1.0,
                                                      elem_id="max_grad_norm")
                            num_heads = gr.Dropdown(label="Attention Heads", choices=["4", "8", "16"],
                                                    value="8", elem_id="num_heads")
                            use_ortholog_gate = gr.Checkbox(label="Use Ortholog Gate", value=True,
                                                            elem_id="use_ortholog_gate")
                            amp_mode = gr.Radio(label="Mixed Precision (AMP)",
                                                choices=["Off", "float16", "bfloat16"],
                                                value="float16", elem_id="amp_mode")
                            temperature = gr.Slider(label="Contrastive Temperature", minimum=0.01,
                                                    maximum=1.0, value=0.07, step=0.01,
                                                    elem_id="temperature")
                            label_smoothing = gr.Slider(label="Label Smoothing", minimum=0.0,
                                                        maximum=0.3, value=0.1, step=0.01,
                                                        elem_id="label_smoothing")
                            margin = gr.Slider(label="Margin", minimum=0.1, maximum=3.0, value=1.0,
                                               step=0.1, elem_id="margin")
                            num_neighbors_str = gr.Textbox(label="Num Neighbors (comma-separated)",
                                                           value="15, 10, 5",
                                                           elem_id="num_neighbors_str")
                            max_subgraph_nodes = gr.Number(label="Max Subgraph Nodes", value=5000,
                                                           precision=0, elem_id="max_subgraph_nodes")
                    # torch.compile is configured in the Runtime Settings tab (it is an
                    # execution/runtime knob, applied per training run).

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
                    with gr.Row(elem_classes=["shep-train-parent-row"]):
                        checkpoint_dropdown = gr.Dropdown(label="Run checkpoint", choices=[],
                                                          value=None, elem_id="checkpoint_dropdown")
                        refresh_ckpt_btn = gr.Button("Refresh", size="sm", scale=0,
                                                     elem_id="shep-train-refresh")
                    manual_checkpoint = gr.Textbox(label="Checkpoint path", value="", max_lines=1,
                                                   visible=False, elem_id="manual_checkpoint")
                    # The device is part of the check, so it is here, not only
                    # in the parameters the drawer covers.
                    continue_device = gr.Radio(label="Device", choices=["auto", "cuda", "cpu"],
                                               value="auto", elem_id="continue_device")
                    check_continue_btn = gr.Button("Check checkpoint", size="sm", scale=0,
                                                   elem_id="shep-train-check",
                                                   elem_classes=["shep-gated"])
                    compare = plain_html("", elem_id="shep-train-compare")
                    continue_preview = plain_html(f'<p class="shep-note">{_CONTINUE_PROMPT}</p>',
                                                  elem_id="continue_preview")
                    # No widget minimum: hidden and empty, a Number is sent as 0,
                    # which a minimum would reject before any handler runs. Empty
                    # or 0 means "not extending"; the run plan refuses below 1.
                    extend_epochs = gr.Number(label="Extra epochs", value=None, precision=0,
                                              visible=False, elem_id="extend_epochs")
                drawer_reason = plain_html("", elem_id="shep-train-drawer-why",
                                           elem_classes=["shep-panel-ft"])

            # ------------------------------------------------------- run view
            with gr.Column(elem_id="shep-train-runview", elem_classes=["shep-panel"], scale=22):
                run_head = plain_html(run_head_html(None, {}, operator=False, snapshot=None),
                                      elem_id="shep-train-run-head")
                run_snap = plain_html("", elem_id="shep-train-run-snap")
                figure = charts.chart_figure(
                    key="training", elem_id="shep-train-charts", series_fn=training_chart_series,
                    csv_fn=training_chart_csv, help_key="training-charts",
                    empty="No run to show yet.")
                resources = plain_html(_IDLE_RESOURCE_HTML, elem_id="resource_display")

    # =========================================================================
    # Wiring
    # =========================================================================
    form = [
        run_name,
        num_epochs, learning_rate, batch_size, conv_type, device, seed,
        hidden_dim, num_layers, dropout, weight_decay, scheduler_type,
        warmup_steps, min_lr_ratio, early_stopping_patience,
        diagnosis_weight, link_prediction_weight, contrastive_weight, ortholog_weight,
        gradient_accumulation_steps, max_grad_norm, num_heads,
        use_ortholog_gate, amp_mode,
        temperature, label_smoothing, margin, num_neighbors_str, max_subgraph_nodes,
    ]
    view_outputs = [ws, dataset, chip, pill, reason, start_btn, cont_btn, stop_btn, resume_btn,
                    runbar, page, run_name, banner, run_head, run_snap, figure, drawer_reason,
                    ctx.training_workspace, cache]
    view_inputs = [ws, mode, cache, dismissed]
    view_step = {"fn": _render_view, "inputs": view_inputs, "outputs": view_outputs,
                 "show_progress": "hidden"}

    # The page shown: the list read again, then the run bar whole.
    ctx.on_page_open(TRAINING, {
        "fn": _open, "inputs": [ws, ctx.training_workspace],
        "outputs": [ws, dataset, chip, ctx.training_workspace, cache, resources],
        "show_progress": "hidden"})
    ctx.page_tokens[TRAINING].change(**view_step)

    # The shell's timers (outside gr.Tabs), read-only polls without the queue.
    ctx.timer_fast.tick(_poll, inputs=[ctx.location, *view_inputs], outputs=view_outputs,
                        queue=False, show_progress="hidden")
    ctx.timer_slow.tick(_poll_resources, inputs=[ctx.location], outputs=[resources],
                        queue=False, show_progress="hidden")

    # The run bar's dataset.
    dataset.input(_on_dataset, inputs=[dataset, mode],
                  outputs=[ws, chip, ctx.training_workspace, mode, notice],
                  show_progress="hidden").then(**view_step)
    # The session's workspace changed (a hand choice above writes it, and so
    # does Datasets' "Use for training ›"): with Continue… open, the parent
    # list follows it (a gated listing, then the chosen parent, the run bar
    # and the chips again).
    chosen_step = {"fn": _on_parent_chosen,
                   "inputs": [continue_source, checkpoint_dropdown, manual_checkpoint, parents, ws,
                              mode], "outputs": [mode, compare], "show_progress": "hidden"}
    ctx.training_workspace.change(_on_training_workspace, inputs=[ctx.training_workspace, ws, mode],
                                  outputs=[ws, dataset, chip, mode],
                                  show_progress="hidden").then(
        _relist_parents, inputs=[ws, mode],
        outputs=[parents, checkpoint_dropdown, parents_note, compare, continue_preview,
                 extend_epochs], show_progress="hidden").then(**chosen_step).then(
        **view_step).then(**ctx.chip_refresh)
    def _on_chip_click(evt: gr.EventData):
        """The chip's "All datasets ›": Research › Datasets."""
        return ctx.go("datasets") if _act(evt) == "datasets" else ctx.no_nav()

    def _on_reason_click(evt: gr.EventData):
        """The reason line's "Open Test ›": Research › Test."""
        return ctx.go("test") if _act(evt) == "open-test" else ctx.no_nav()

    chip.click(_on_chip_click, inputs=None, outputs=ctx.nav_outputs, show_progress="hidden")
    reason.click(_on_reason_click, inputs=None, outputs=ctx.nav_outputs, show_progress="hidden")
    banner.click(_on_banner_click, inputs=[dismissed, cache], outputs=[dismissed, banner],
                 show_progress="hidden")

    # Start, Stop.
    # Start also closes a Stop confirmation left from an earlier job.
    start_btn.click(_on_start, inputs=[ws, *form, randomize_seed], outputs=[notice, seed],
                    show_progress="hidden").then(
        _on_stop_cancel, inputs=[mode], outputs=[mode], show_progress="hidden").then(
        **view_step).then(**ctx.chip_refresh)
    randomize_seed.change(fn=_seed_field_lock, inputs=[randomize_seed], outputs=[seed])
    stop_btn.click(_on_stop_request, inputs=[mode], outputs=[stop_text, mode],
                   show_progress="hidden").then(**view_step).then(**ctx.chip_refresh)
    stop_confirm_btn.click(_on_stop, inputs=[mode], outputs=[notice, mode],
                           show_progress="hidden").then(**view_step).then(**ctx.chip_refresh)
    stop_keep_btn.click(_on_stop_cancel, inputs=[mode], outputs=[mode],
                        show_progress="hidden").then(**view_step)

    # Continue…
    cont_btn.click(_on_continue_open, inputs=[ws, mode],
                   outputs=[notice, mode, parents, checkpoint_dropdown, parents_note,
                            continue_preview],
                   show_progress="hidden").then(
        _continue_device, inputs=[device], outputs=[continue_device],
        show_progress="hidden").then(**chosen_step).then(**view_step).then(**ctx.chip_refresh)
    for closer in (cancel_btn, close_btn):
        closer.click(_on_continue_close, inputs=[mode], outputs=[mode],
                     show_progress="hidden").then(**view_step)
    refresh_ckpt_btn.click(_refresh_checkpoints, inputs=[ws, mode, checkpoint_dropdown],
                           outputs=[parents, checkpoint_dropdown, parents_note],
                           show_progress="hidden").then(**ctx.chip_refresh)
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
                           outputs=[checkpoint_dropdown, refresh_ckpt_btn, manual_checkpoint])
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
        component.change(_on_parent_chosen, inputs=chosen_inputs, outputs=[mode, compare],
                         show_progress="hidden").then(**view_step)

    # HGT requires float32 (AMP disabled): the refusal is state, so it stays
    # visible text in `info=`; "" clears it (Gradio 6.30 keeps an info sent
    # as None).
    conv_type.change(fn=_on_conv_type_change, inputs=[conv_type], outputs=[amp_mode])

    # The run bar's line for a continued run (while the drawer is open).
    def _continues(mode_value):
        label = ((mode_value or {}).get("parent") or {}).get("run_label")
        return (f'<span class="shep-rb-k">Run</span> continues the parent’s run '
                f'<code>{r.e(label)}</code>' if label else
                '<span class="shep-rb-k">Run</span> <span class="shep-muted">continues the chosen '
                'checkpoint’s run</span>')

    mode.change(_continues, inputs=[mode], outputs=[continues], show_progress="hidden")


#: This page's element id (`shell.page_elem_id`), the scope of `PAGE_CSS`.
_PAGE = f"#shep-page-{TRAINING}"

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
{_PAGE} #shep-train-dataset input {{ text-overflow: ellipsis; flex: 1 1 auto !important; width: 0 !important; min-width: 0 !important; }}
{_PAGE} #shep-train-dataset .secondary-wrap {{ min-width: 0 !important; }}
{_PAGE} #shep-train-dataset .icon-wrap {{ flex: none !important; }}
/* The two selects drawn as fields: a border, the text clear of the chevron, grey while disabled. */
{_PAGE} #shep-train-dataset .wrap:has(> .wrap-inner), {_PAGE} #checkpoint_dropdown .wrap:has(> .wrap-inner) {{ border: 1px solid var(--shep-border-2) !important; border-radius: 8px !important; background: #fff !important; box-shadow: none !important; }}
{_PAGE} #shep-train-dataset .wrap:has(> .wrap-inner):focus-within, {_PAGE} #checkpoint_dropdown .wrap:has(> .wrap-inner):focus-within {{ border-color: var(--shep-accent) !important; }}
{_PAGE} #shep-train-dataset input, {_PAGE} #checkpoint_dropdown input {{ padding-right: 30px !important; text-overflow: ellipsis; }}
{_PAGE} #shep-train-dataset .icon-wrap, {_PAGE} #checkpoint_dropdown .icon-wrap {{ color: var(--shep-text-2) !important; }}
{_PAGE} #shep-train-dataset .wrap:has(input:disabled), {_PAGE} #checkpoint_dropdown .wrap:has(input:disabled) {{ background: var(--shep-grey-50) !important; border-color: var(--shep-grey-200) !important; }}
{_PAGE} #shep-train-dataset input:disabled, {_PAGE} #checkpoint_dropdown input:disabled {{ color: var(--shep-grey-600) !important; -webkit-text-fill-color: var(--shep-grey-600); cursor: not-allowed; }}
{_PAGE} #shep-train-dataset .wrap:has(input:disabled) .icon-wrap {{ color: var(--shep-grey-600) !important; }}
{_PAGE} #run_name .input-container {{ width: 260px; }}
{_PAGE} #run_name input {{ border: 1px solid var(--shep-border-2) !important; border-radius: 8px; padding: 6px 10px !important; background: #fff; }}
{_PAGE} #run_name input:disabled {{ background: var(--shep-grey-50); color: var(--shep-muted); }}
{_PAGE} .shep-rb-row > button {{ flex: none !important; width: auto !important; min-width: max-content !important; padding: 0 14px !important; height: 40px; white-space: nowrap; }}
{_PAGE} #shep-train-chip {{ flex: 1 1 360px !important; position: relative; z-index: 20; }}
{_PAGE} .shep-rb-chip {{ display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; font-size: 15px; position: relative; }}
/* The identity's parts wrap one by one (never hidden): "built …", the fingerprint, then the marks with "Details ›". */
{_PAGE} .shep-rb-chip > .shep-ds {{ display: contents; }}
{_PAGE} .shep-rb-chip .shep-ds-marks {{ display: inline-flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; }}
{_PAGE} .shep-rb-chip > .shep-ds > .shep-sep {{ margin: 0 -2px; }}
{_PAGE} .shep-rb-chip > .shep-ds > .shep-sep:has(+ .shep-ds-marks) {{ display: none; }}
{_PAGE} .shep-rb-inuse {{ color: var(--shep-text-2); font-size: 14px; }}
{_PAGE} .shep-link {{ border: 0; background: none; padding: 0; color: var(--shep-accent-text); font-size: 15px; cursor: pointer; font-weight: 500; }}
{_PAGE} .shep-link:hover {{ text-decoration: underline; }}
{_PAGE} .shep-pop {{ position: absolute; top: calc(100% + 8px); left: 0; z-index: 30; width: min(460px, calc(100vw - 48px)); background: #fff; border: 1px solid var(--shep-border-2); border-radius: 10px; box-shadow: 0 10px 28px rgba(20,24,40,.16); padding: 12px 14px; display: flex; flex-direction: column; gap: 10px; font-size: 15px; }}
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
{_PAGE} .shep-rb-why > span:not(.shep-tag) {{ min-width: 0; }}
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
{_PAGE} .shep-banner-more {{ margin-top: 6px; }}
{_PAGE} .shep-banner-more > summary {{ cursor: pointer; font-size: 14px; font-weight: 600; color: var(--shep-red-700); width: max-content; max-width: 100%; }}
{_PAGE} .shep-banner-n {{ font-weight: 400; color: var(--shep-text-2); }}
{_PAGE} .shep-banner-tail {{ margin: 6px 0 0; padding: 6px 10px; background: #fff; border: 1px solid var(--shep-red-200); border-radius: 6px; color: var(--shep-text); font-family: var(--shep-mono); font-size: 14px; line-height: 1.35; white-space: pre-wrap; overflow-wrap: anywhere; max-height: calc(4 * 1.35em + 14px); overflow: auto; }}
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
{_PAGE} .shep-check code, {_PAGE} .shep-rv-snap code {{ overflow-wrap: anywhere; word-break: break-all; }}
{_PAGE} .shep-inherit {{ margin-top: 8px; border: 1px solid var(--shep-border); border-radius: 8px; background: #fff; }}
{_PAGE} .shep-inherit > summary {{ padding: 6px 10px; cursor: pointer; font-weight: 600; }}
{_PAGE} .shep-inherit .shep-ftbl {{ margin: 0 10px 10px; width: calc(100% - 20px); }}
{_PAGE} .shep-drawer-why {{ font-size: 15px; color: var(--shep-text-2); }}
{_PAGE} .shep-drawer-why .shep-rb-why {{ text-align: left; flex-wrap: nowrap; align-items: flex-start; }}
{_PAGE} #shep-train-drawer-why:has(.html-container:empty) {{ display: none !important; }}
{_PAGE} .shep-rv-hd {{ gap: 6px 10px !important; }}
{_PAGE} .shep-rv-hd h2 code {{ font-size: 15px; font-weight: 600; }}
{_PAGE} .shep-rv-used {{ display: inline-flex; align-items: center; flex-wrap: wrap; gap: 6px; font-size: 15px; }}
{_PAGE} .shep-rv-snap {{ margin: 0; padding: 8px 16px 0; font-size: 15px; color: var(--shep-text-2); overflow-wrap: anywhere; }}
{_PAGE} #shep-train-charts {{ padding: 8px 16px 6px !important; }}
{_PAGE} .shep-rv-res {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 6px; padding: 8px 16px; border-top: 1px solid var(--shep-border); font-size: 15px; color: var(--shep-text-2); }}
{_PAGE} .shep-rv-res b {{ color: var(--shep-text); margin-right: 8px; }}
{_PAGE} .shep-lock {{ flex: none; }}
{_PAGE} .shep-html svg {{ display: inline-block; vertical-align: -2px; }}
{_PAGE} .shep-train-parent-row {{ align-items: flex-end !important; flex-wrap: nowrap !important; }}
{_PAGE} #checkpoint_dropdown {{ flex: 1 1 auto !important; min-width: 0 !important; }}
{_PAGE} #shep-train-refresh {{ flex: none !important; width: auto !important; min-width: max-content !important; }}
{_PAGE} #shep-train-check {{ align-self: flex-start; width: auto !important; min-width: max-content !important; }}
{_PAGE} #shep-train-parents-note {{ margin-top: -4px; }}
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
  /* Never clipped: when the charts' minimum does not fit, the run view scrolls inside. A failed
     run's banner takes height, so the charts then keep a usable minimum (plots of about 90 px). */
  {_PAGE} #shep-train-runview {{ overflow-y: auto !important; }}
  {_PAGE} #shep-train-runview > #shep-train-charts {{ flex: 1 1 auto !important; min-height: 240px; display: flex; flex-direction: column; }}
  {_PAGE} #shep-train:has(#shep-train-banner .shep-banner) #shep-train-runview > #shep-train-charts {{ min-height: 370px; }}
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
  {_PAGE} .shep-chart-xl {{ display: none; }}
  {_PAGE} .shep-rv-res {{ padding-top: 6px; padding-bottom: 6px; }}
}}
@container shep-runview (max-width: 1099px) {{
  {_PAGE} .shep-charts-hint {{ display: none; }}
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
