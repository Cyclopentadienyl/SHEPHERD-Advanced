"""
SHEPHERD-Advanced Training Console Component
==============================================
Gradio UI component for training control and monitoring (Tab 1).

Module: src/webui/components/training_console.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/webui/components/training_console.py

Purpose:
    Provide the Training Console tab with:
    - 3-tier hierarchical parameter panel (Basic / Advanced / Expert)
    - Start / Stop / Resume control buttons
    - Real-time loss and metrics curves (LinePlot)
    - GPU/RAM resource monitoring
    - Status display with polling via gr.Timer

Architecture Note:
    This component imports TrainingManager from src.api.services (allowed path).
    It does NOT import from src.training (forbidden by import-linter).
    All training operations go through the TrainingManager which launches
    subprocess calls to scripts/train_model.py.

    **Its operator controls check D8 first** (M5-1), on their own Gradio
    request (`src/webui/operator_access.py`): Start, Stop, Continue, Check
    checkpoint and the run list (both open checkpoint files, SPEC_4 C3), and
    Save Path. A refusal changes nothing and says why. The training routes
    apply the same check through their FastAPI dependency.

Dependencies:
    - gradio: UI components
    - pandas: DataFrame for chart data
    - src.api.services.training_manager: training_manager singleton

Version: 1.0.0
"""
from __future__ import annotations

import functools
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gradio as gr
import pandas as pd

from src.api.services.training_manager import training_manager
from src.config.runtime_presets import load_runtime_settings
from src.webui.operator_access import operator_access

logger = logging.getLogger(__name__)


TRAINING_CONFIG_FILE = Path(".shepherd_training_config.json")
DEFAULT_WORKSPACE = "data/workspaces/default"
DISPLAY_PREFIX = "SHEPHERD-Advanced/"

# Max seed value. trainer._set_seed calls np.random.seed(), whose legal range is
# [0, 2**32 - 1] (the binding limit; torch.manual_seed is wider). Generated and
# accepted seeds stay in this range.
_MAX_SEED = 2**32 - 1

# Position of the `seed` widget in all_params / _on_start's *args. MUST match the
# order of all_params (and _collect_config's signature) below — used by _on_start
# to swap in a random seed when "Randomize seed" is on.
SEED_PARAM_INDEX = 7


def _random_seed() -> int:
    """A fresh seed in NumPy's legal range [0, _MAX_SEED]."""
    return random.randint(0, _MAX_SEED)


def _seed_field_lock(randomize: bool):
    """Lock (grey out) the seed field while "Randomize seed" is on; the random
    seed is generated on Start and written back, so manual editing is disabled."""
    return gr.update(interactive=not bool(randomize))


def _load_training_config() -> Dict[str, str]:
    """Load saved training path config."""
    if TRAINING_CONFIG_FILE.exists():
        try:
            with open(TRAINING_CONFIG_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"data_dir": DEFAULT_WORKSPACE}


def _save_training_config(data_dir: str, request: gr.Request) -> str:
    """Save training path config: an operator control (D8), checked on this
    event's own request before anything is written."""
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


def _reset_training_config() -> Tuple[str, str]:
    """Reset training path config to defaults."""
    return f"{DISPLAY_PREFIX}{DEFAULT_WORKSPACE}", "Paths reset to defaults."


class ConfigValidationError(ValueError):
    """Raised when Training Console inputs are empty or illegal.

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
    """Collect all parameter widgets into a config dict."""
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
        # Paths (strip display prefix so backend receives relative-to-project-root paths)
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


def _on_start(request: gr.Request, *args):
    """Handle Start Training button click: an operator control (D8).

    `request` comes first: Gradio passes a request only to a positional
    parameter annotated ``gr.Request``, and the form's values follow as
    ``*args``. A request D8 refuses starts nothing and leaves the charts and
    buttons as they are.

    Returns status text, 4 empty plot DataFrames (to clear stale charts),
    and 3 button updates.
    """
    access = operator_access(request)
    if not access.allowed:
        no_change = gr.update()
        return (access.refusal, no_change, no_change, no_change, no_change,
                no_change, no_change, no_change, no_change)
    global _poll_cache_key
    _poll_cache_key = None  # force full refresh on next poll tick
    # Last input is the "Randomize seed" checkbox; the rest are all_params.
    randomize = bool(args[-1])
    params = list(args[:-1])
    if randomize:
        # Swap in a fresh seed BEFORE validation so a locked/empty seed field
        # still launches; the used seed is echoed back to the (locked) field.
        params[SEED_PARAM_INDEX] = _random_seed()
    try:
        config = _collect_config(*params)
    except ConfigValidationError as e:
        # Toast carries the detail (survives the ~1.5s status poll); the Status
        # field only shows a short marker.
        gr.Warning(str(e), duration=12)
        return (
            "⚠️ **Invalid configuration** — see the notification.",
            _EMPTY_LOSS_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF,
            gr.update(interactive=True),    # keep start enabled
            gr.update(interactive=False),   # keep stop disabled
            gr.update(interactive=True),    # keep resume enabled
            gr.update(),                    # seed: unchanged (nothing launched)
        )
    result = training_manager.start_training(config)
    if result.get("success"):
        return (
            f"**Status**: RUNNING\n**PID**: {result.get('pid')}\n**Phase**: Starting..."
            f"\n**Run directory**: `{result.get('run_dir')}`",
            _EMPTY_LOSS_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF,
            gr.update(interactive=False),   # disable start
            gr.update(interactive=True),    # enable stop
            gr.update(interactive=False),   # disable resume
            gr.update(value=config["seed"]),  # echo the seed actually used
        )
    return (
        f"Failed: {result.get('error', 'Unknown error')}",
        _EMPTY_LOSS_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF,
        gr.update(interactive=True),    # keep start enabled
        gr.update(interactive=False),   # keep stop disabled
        gr.update(interactive=True),    # keep resume enabled
        gr.update(),                    # seed: unchanged (nothing launched)
    )


def _snapshot_sentence(snapshot: Optional[Dict[str, Any]]) -> str:
    """Where a run can be continued from, in one sentence (contract M3a)."""
    if not snapshot:
        return "No checkpoint was saved, so nothing can be continued."
    if not snapshot.get("readable", True):
        return (f"The latest checkpoint `{snapshot['checkpoint']}` cannot be read "
                f"({snapshot.get('reason')}).")
    epoch, target = snapshot.get("completed_epochs"), snapshot.get("target_epochs")
    where = f"epoch {epoch} of {target}" if target else f"epoch {epoch}"
    end = snapshot.get("end")
    if end == "finished":
        return (f"Finished ({where}): `{snapshot['checkpoint']}`. To train further, "
                "extend it under Continue Training.")
    if end == "stopped_early":
        return f"Stopped early at {where} ({snapshot.get('stop_reason')}); it cannot be continued."
    label = "Paused at" if end == "paused" else "Latest resumable checkpoint:"
    return f"{label} {where}, `{snapshot['checkpoint']}`."


def _on_stop_request():
    """First step of Stop: say what stopping loses before it happens.

    Returns the confirmation group's visibility and its text.
    """
    snapshot = training_manager.latest_snapshot()
    if snapshot and snapshot.get("readable", True) and snapshot.get("completed_epochs"):
        text = (
            f"**Stop training now?** The last saved checkpoint is from epoch "
            f"{snapshot['completed_epochs']} (`{snapshot['checkpoint']}`). Progress since "
            f"then is lost; you can continue from epoch {snapshot['completed_epochs']}."
        )
    else:
        text = ("**Stop training now?** No checkpoint has been saved yet, so all progress "
                "of this run is lost and nothing can be continued.")
    return gr.update(visible=True), text


def _on_stop_cancel():
    return gr.update(visible=False)


def _on_stop(request: gr.Request):
    """Handle the confirmed Stop: an operator control (D8), checked on this
    event's own request; refused, nothing is stopped.

    Returns status text, gr.update() for 4 plots (no change), 3 button
    updates, and the confirmation group hidden.
    """
    no_change = gr.update()
    access = operator_access(request)
    if not access.allowed:
        return (access.refusal, no_change, no_change, no_change, no_change,
                no_change, no_change, no_change, gr.update(visible=False))
    global _poll_cache_key
    _poll_cache_key = None  # force full refresh on next poll tick
    result = training_manager.stop_training()
    if result.get("success"):
        return (
            "**Status**: STOPPED\n" + _snapshot_sentence(result.get("outcome")),
            no_change, no_change, no_change, no_change,
            gr.update(interactive=True),    # re-enable start
            gr.update(interactive=False),   # disable stop
            gr.update(interactive=True),    # re-enable continue
            gr.update(visible=False),
        )
    return (
        f"Failed: {result.get('error', 'Unknown error')}",
        no_change, no_change, no_change, no_change,
        gr.update(interactive=False),   # keep start disabled while running
        gr.update(interactive=True),    # keep stop enabled
        gr.update(interactive=False),   # keep continue disabled while running
        gr.update(visible=False),
    )


#: The two sources of a parent checkpoint. Both go through the same check.
CONTINUE_FROM_LIST = "From run list"
CONTINUE_FROM_PATH = "Manual path"


def _continue_parent(source: str, listed: Optional[str], manual: Optional[str]) -> str:
    """The parent checkpoint the operator chose, from the source they chose."""
    if source == CONTINUE_FROM_PATH:
        return _strip_display_prefix(manual or "")
    return str(listed or "").strip()


def _format_continue_preview(preview: Dict[str, Any], source: str) -> str:
    """The check result, what will happen, and the inherited settings, as a
    read-only summary (contract M3a-3; not disabled inputs, which screen
    readers skip)."""
    origin = "manual path" if source == CONTINUE_FROM_PATH else "run list"
    head = f"**Checkpoint** ({origin}): `{preview.get('parent', '')}`"
    if preview.get("sha256"):
        head += (f"  \nSHA-256 `{preview['sha256'][:16]}…`"
                 + (f", architecture **{preview['architecture']}**"
                    if preview.get("architecture") else ""))
    if not preview.get("ok"):
        return f"{head}\n\n❌ **Cannot continue.** {preview.get('reason')}"
    lines = [head, "", f"✅ **{preview['action']}.**", ""]
    inherited = preview.get("inherited") or {}
    if inherited:
        lines.append("Settings taken from this checkpoint (they cannot be changed when "
                     "continuing; to train with other settings, start a new run):")
        lines.append("")
        lines.append("| Setting | Value |")
        lines.append("|---|---|")
        for name in sorted(inherited):
            lines.append(f"| {name} | {inherited[name]!r} |")
        lines.append("")
    lines.append("The start checks again against this workspace's data and graph.")
    return "\n".join(lines)


#: The continue block's prompt before a check, and after the selection changes.
_CONTINUE_PROMPT = "Select a checkpoint, then click **Check checkpoint**."


def _extra_epochs_update(preview: Dict[str, Any]):
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
    is the run list: a Refresh or a conv_type change repopulates it even while
    a typed path is the source, and must not clear that path's check."""
    if source == CONTINUE_FROM_PATH:
        return gr.update(), gr.update()
    return _on_continue_selection_changed()


def _on_manual_changed(source):
    """The typed path changed; it is the chosen checkpoint only when the
    source is the typed path."""
    if source != CONTINUE_FROM_PATH:
        return gr.update(), gr.update()
    return _on_continue_selection_changed()


def _on_continue_device_changed():
    """The device is part of the check (mixed precision takes effect only on
    CUDA): the last check no longer describes what Continue will do. The Extra
    epochs entered stay; they belong to the checkpoint, not the device."""
    return _CONTINUE_PROMPT


def _preview_for(parent: str, device: Optional[str], extend_epochs) -> Dict[str, Any]:
    """The service's preview, with Extra epochs only for a finished
    checkpoint: the field is the console's, and for any other checkpoint a
    value in it is one left from an earlier selection, not a request. The API
    and the CLI send `extend_epochs` as given and are refused as before."""
    epochs = int(extend_epochs) if extend_epochs else None
    return training_manager.preview_continuation(
        parent, device=device or "auto", extend_epochs=epochs, extend_if_finished=True)


def _on_check_continue(source, listed, manual, device, extend_epochs, request: gr.Request):
    """Check button: preview what continuing from the chosen checkpoint does.
    It opens the checkpoint (SPEC_4 C3), so D8 is checked first, on this
    event's own request; refused, nothing is opened.

    Returns the preview text and the extend-epochs input's update: shown only
    for a finished checkpoint, where extending is the only way on.
    """
    access = operator_access(request)
    if not access.allowed:
        return access.refusal, gr.update()
    parent = _continue_parent(source, listed, manual)
    if not parent:
        return ("Select a checkpoint from the run list, or enter a path, then click "
                "**Check checkpoint**."), gr.update(visible=False, value=None)
    preview = _preview_for(parent, device, extend_epochs)
    text = _format_continue_preview(preview, source)
    if preview.get("end") == "finished" and not preview.get("ok") and not extend_epochs:
        text += "\n\nEnter **Extra epochs** below, then click **Check checkpoint** again."
    return text, _extra_epochs_update(preview)


def _on_resume(data_dir, run_name, device, source, listed, manual, extend_epochs,
               request: gr.Request):
    """Continue Training: an operator control (D8, checked first on this
    event's own request; refused, nothing is opened or started), then the same
    check as the preview, then start.

    Only operational fields are sent (contract M3a-3): every learning setting
    comes from the checkpoint, so the training form's values are not used.
    The training service and the training process each check again.
    """
    access = operator_access(request)
    if not access.allowed:
        no_change = gr.update()
        return (access.refusal, no_change, no_change, no_change, no_change,
                no_change, no_change, no_change, no_change, no_change)
    global _poll_cache_key
    _poll_cache_key = None
    no_start = (gr.update(interactive=True), gr.update(interactive=False),
                gr.update(interactive=True))
    empty = (_EMPTY_LOSS_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF)

    parent = _continue_parent(source, listed, manual)
    if not parent:
        message = "Select a checkpoint (run list or path) under Continue Training first."
        gr.Warning(message, duration=12)
        return ("⚠️ **Not started** — see Continue Training.", *empty, *no_start, message,
                gr.update(visible=False, value=None))
    preview = _preview_for(parent, device, extend_epochs)
    preview_text = _format_continue_preview(preview, source)
    if not preview.get("ok"):
        if preview.get("end") == "finished" and not extend_epochs:
            preview_text += ("\n\nEnter **Extra epochs** below, then click **Continue "
                             "Training** again.")
        gr.Warning("Cannot continue from this checkpoint; the reason is under Continue "
                   "Training.", duration=12)
        return ("⚠️ **Not started** — see Continue Training.", *empty, *no_start, preview_text,
                _extra_epochs_update(preview))

    config: Dict[str, Any] = {
        "data_dir": _strip_display_prefix(data_dir) or DEFAULT_WORKSPACE,
        "run_name": (run_name or "").strip(),
        "device": device or "auto",
        "resume_from": preview["parent"],
        "compile": bool(load_runtime_settings().get("torch_compile", False)),
    }
    if preview["mode"] == "extend":
        config["extend_epochs"] = preview["stage"]["target_epochs"]
    result = training_manager.start_training(config)
    if result.get("success"):
        return (
            f"**Status**: RUNNING\n**PID**: {result.get('pid')}\n"
            f"**Phase**: {preview['action']}...\n**Run directory**: `{result.get('run_dir')}`",
            *empty,
            gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False),
            preview_text, _extra_epochs_update(preview),
        )
    return (f"Failed: {result.get('error', 'Unknown error')}", *empty, *no_start, preview_text,
            _extra_epochs_update(preview))


def _toggle_continue_source(source):
    from_list = source != CONTINUE_FROM_PATH
    return (gr.update(visible=from_list), gr.update(visible=from_list),
            gr.update(visible=not from_list))


def _export_metrics_csv() -> Optional[str]:
    """Export training metrics history as a CSV file.

    Returns the path to the generated CSV, or None if no data.
    """
    import csv
    import tempfile

    metrics = training_manager.get_metrics_history()
    train_data = metrics.get("train", [])
    val_data = metrics.get("validation", [])

    if not train_data and not val_data:
        return None

    # Collect all unique keys across train and val entries
    all_keys: set = set()
    for entry in train_data:
        all_keys.update(entry.keys())
    for entry in val_data:
        all_keys.update(entry.keys())

    # Sort keys for consistent column order, with 'epoch' first
    sorted_keys = sorted(all_keys - {"epoch"})
    fieldnames = ["epoch", "split"] + sorted_keys

    fd, path = tempfile.mkstemp(suffix=".csv", prefix="training_metrics_")
    with os.fdopen(fd, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in train_data:
            row = {k: entry.get(k, "") for k in fieldnames}
            row["split"] = "train"
            if "epoch" in entry:
                row["epoch"] = entry["epoch"] + 1  # 1-indexed
            writer.writerow(row)
        for entry in val_data:
            row = {k: entry.get(k, "") for k in fieldnames}
            row["split"] = "validation"
            if "epoch" in entry:
                row["epoch"] = entry["epoch"] + 1
            writer.writerow(row)

    return path


def _workspace_field(name: Any) -> Optional[str]:
    """The Data Directory text for the workspace the server lists as `name`
    (Datasets › Use for training ›), or None when it is not listed: relative to
    the project root with the display prefix when inside it, as the field shows
    paths, else the resolved directory (the training process takes either)."""
    from src.api.services.server_lists import ServerListRefused, resolve_workspace
    from src.api.services.training_manager import PROJECT_ROOT

    if not name:
        return None
    try:
        path = resolve_workspace(name)
    except ServerListRefused:
        return None
    try:
        return DISPLAY_PREFIX + path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _from_datasets(name: Any) -> Tuple[Any, Any]:
    """Datasets › Use for training ›: this session's Data Directory and the
    Data Paths section opened to show it; nothing changes on the server."""
    value = _workspace_field(name)
    if value is None:
        return gr.update(), gr.update()
    return gr.update(value=value), gr.Accordion(open=True)


def _strip_display_prefix(path: str) -> str:
    """Strip the UI display prefix (``SHEPHERD-Advanced/...`` -> ``...``)."""
    p = (path or "").strip()
    return p[len("SHEPHERD-Advanced/"):] if p.startswith("SHEPHERD-Advanced/") else p


def _refresh_checkpoints(data_dir_value: str, conv_type_value: str, request: gr.Request):
    """Refresh the resume-checkpoint dropdown for the CURRENT training architecture.

    Lists that architecture's run checkpoints through the shared listing
    (`TrainingManager.get_checkpoints`, contract M3a-2), latest run first, so
    the dropdown always matches the selected conv_type. Wired to both the
    Refresh button and conv_type's change event.

    The listing opens every checkpoint (SPEC_4 C3), so D8 is checked first,
    on this event's own request, as for `GET /training/checkpoints`; refused,
    the list is emptied, nothing is opened, and a notice says why.
    """
    access = operator_access(request)
    if not access.allowed:
        gr.Warning(access.refusal, duration=12)
        return gr.update(choices=[], value=None)
    data_dir = _strip_display_prefix(data_dir_value) or DEFAULT_WORKSPACE
    checkpoints = training_manager.get_checkpoints(data_dir, conv_type_value or None)
    logger.info("Listed %d checkpoints in the %s runs of %s",
                len(checkpoints), conv_type_value, data_dir)
    choices = []
    for ckpt in checkpoints:
        size = ckpt.get("size_mb", 0)
        epoch = ckpt.get("epoch")
        label = f"{ckpt['run']} / {ckpt['filename']} ({size:.1f} MB"
        if epoch is not None:
            label += f", epoch {epoch}"
        label += ")"
        choices.append((label, ckpt["filepath"]))
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


_IDLE_RESOURCE_HTML = (
    '<div style="font-family:IBM Plex Sans,system-ui,sans-serif;padding:4px 0">'
    '<div style="font-size:13px;color:#6b7280">Loading resource info...</div>'
    '</div>'
)


def _poll_status(epoch_window: str = "All"):
    """
    Poll training status and metrics. Called by gr.Timer every 1.5 seconds.

    Smart refresh strategy:
        - **System Resources** are polled every tick (always fresh).
        - **Charts, status text, buttons** are only refreshed when training
          is actively running or when the state/window just changed.  When
          training is idle, completed, or stopped, these outputs return
          ``gr.update()`` (Gradio no-op) so Plotly chart zoom/pan and other
          component state are preserved.

    Args:
        epoch_window: Value from the Epoch Window dropdown.
            ``"All"`` shows every epoch; ``"Last N"`` trims charts to the
            most recent *N* epochs.

    Returns:
        (status_text, loss_df, mrr_df, hits_df, lr_df, resource_text,
         start_btn_update, stop_btn_update, resume_btn_update)
    """
    global _poll_cache_key, _current_banner_key

    status_info = training_manager.get_status()
    s = status_info.get("status", "idle")

    # System resources are ALWAYS polled (user checks GPU/RAM before starting)
    resources = training_manager.get_system_resources()
    resource_text = _format_resources(resources)

    # --- Error/warning banner (independent of the smart-refresh gating) -----
    # Cheap to recompute every tick; carried in all return paths so it stays
    # correct whether or not the charts/status are refreshed.
    try:
        banner_html, banner_key = _build_error_banner_payload(
            status_info, _get_pyg_warning()
        )
    except Exception as exc:  # banner must never break polling
        logger.debug("error banner build failed: %s", exc)
        banner_html, banner_key = None, None
    _current_banner_key = banner_key
    if banner_key is None or banner_key in _dismissed_banner_keys:
        banner_update = gr.update(visible=False, value="")
        dismiss_update = gr.update(visible=False)
    else:
        banner_update = gr.update(visible=True, value=banner_html)
        dismiss_update = gr.update(visible=True)

    # --- Smart refresh: skip re-renders when nothing changed ---------------
    cache_key = (s, epoch_window)
    is_stable = (
        s not in ("running", "stopping")
        and cache_key == _poll_cache_key
    )

    if is_stable:
        # State hasn't changed — only send fresh resource_text.
        # gr.update() is a Gradio no-op that preserves current component
        # state, including Plotly manual zoom/pan.
        _no_change = gr.update()
        return (
            _no_change,                                        # status_display
            _no_change, _no_change, _no_change, _no_change,   # 4 plots
            resource_text,                                     # always fresh
            banner_update, dismiss_update,                     # error banner
            _no_change, _no_change, _no_change,                # 3 buttons
        )

    # State changed or training is active → full refresh
    _poll_cache_key = cache_key

    # Button state
    is_running = s in ("running", "stopping")
    start_update = gr.update(interactive=not is_running)
    stop_update = gr.update(interactive=is_running)
    resume_update = gr.update(interactive=not is_running)

    if s == "idle":
        # No training ever started — skip metrics file reads
        return (
            _format_status(status_info),
            _EMPTY_LOSS_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF,
            resource_text,
            banner_update, dismiss_update,
            start_update, stop_update, resume_update,
        )

    # Training is or was active — do full poll
    metrics_history = training_manager.get_metrics_history()

    status_text = _format_status(status_info)

    train_data = metrics_history.get("train", [])
    val_data = metrics_history.get("validation", [])

    loss_df = _build_loss_df(train_data, val_data)
    mrr_df = _build_metric_df(val_data, "val_mrr", "MRR")
    hits_df = _build_hits_df(val_data)
    lr_df = _build_metric_df(train_data, "learning_rate", "Learning Rate")

    # Apply epoch windowing: rows only. The x axis follows the rows it is
    # given; an x_lim sent with the data would rebuild the chart on every tick
    # in Gradio 6.30 and draw the previous poll's rows.
    window = _parse_epoch_window(epoch_window)
    loss_df = _trim_df_to_window(loss_df, window)
    mrr_df = _trim_df_to_window(mrr_df, window)
    hits_df = _trim_df_to_window(hits_df, window)
    lr_df = _trim_df_to_window(lr_df, window)

    return (
        status_text,
        loss_df, mrr_df, hits_df, lr_df,
        resource_text,
        banner_update, dismiss_update,
        start_update, stop_update, resume_update,
    )


def _format_status(status_info: Dict[str, Any]) -> str:
    """Format training status as a readable string."""
    s = status_info.get("status", "idle")
    lines = [f"**Status**: {s.upper()}"]

    if status_info.get("pid"):
        lines.append(f"**PID**: {status_info['pid']}")
    if status_info.get("run_dir"):
        lines.append(f"**Run directory**: `{status_info['run_dir']}`")

    # Show phase if available (initializing, training, completed)
    phase = status_info.get("phase")
    if phase and phase == "initializing":
        lines.append("**Phase**: Initializing (loading data & model)...")

    epoch = status_info.get("current_epoch")
    total = status_info.get("total_epochs")
    if epoch is not None and total is not None:
        pct = (epoch / total * 100) if total > 0 else 0
        lines.append(f"**Epoch**: {epoch} / {total} ({pct:.0f}%)")

    # Show batch progress within current epoch
    batch = status_info.get("batch")
    total_batches = status_info.get("total_batches")
    if batch is not None and total_batches is not None and total_batches > 0:
        batch_pct = (batch / total_batches * 100)
        lines.append(f"**Batch**: {batch} / {total_batches} ({batch_pct:.0f}%)")
    elif batch is not None:
        lines.append(f"**Batch**: {batch}")

    elapsed = status_info.get("elapsed_seconds")
    if elapsed is not None:
        m, sec = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        lines.append(f"**Elapsed**: {h:02d}:{m:02d}:{sec:02d}")

    metrics = status_info.get("latest_metrics", {})
    if metrics:
        metric_strs = [f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))]
        if metric_strs:
            lines.append("**Metrics**: " + " | ".join(metric_strs[:6]))

    if s in ("stopped", "completed", "failed"):
        lines.append(_snapshot_sentence(status_info.get("outcome")))

    # NOTE: error_message is intentionally NOT rendered here. It is surfaced in
    # the dedicated error banner (see _build_error_banner_payload) so failures
    # stand alone instead of being buried in the normal status text.

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Error / warning banner
#
# A dedicated banner that is hidden during normal operation and only appears
# when the backend reports something the user must see:
#   * red  — a hard training error (subprocess crashed; error_message set)
#   * yellow — a non-fatal warning (PyG native extensions missing -> slow
#              fallback kernels)
# The banner is rendered separately from the normal status text and can be
# dismissed manually. A dismissal is keyed to the banner's *content*, so it
# stays dismissed until the underlying problem changes (a new/different error
# re-shows automatically).
# ---------------------------------------------------------------------------
def _get_pyg_warning() -> Optional[str]:
    """PyG native-extension fallback warning (probed + logged once, cached)."""
    global _pyg_warning_cache
    if _pyg_warning_cache is _UNSET:
        try:
            from src.utils.pyg_native_check import (
                get_fallback_warning,
                log_pyg_native_status,
            )

            log_pyg_native_status()
            _pyg_warning_cache = get_fallback_warning()
        except Exception as exc:  # torch stack may be unavailable in this process
            logger.debug("PyG native check unavailable in UI process: %s", exc)
            _pyg_warning_cache = None
    return _pyg_warning_cache


def _render_error_block(message: str) -> str:
    """Red error block (monospace, preserves the traceback tail layout)."""
    import html

    body = html.escape(str(message)).replace("\n", "<br>")
    return (
        '<div style="background:#fef2f2;border:1px solid #fecaca;'
        "border-left:4px solid #dc2626;border-radius:6px;padding:10px 12px;"
        "margin-bottom:6px;font-size:13px;color:#991b1b;"
        'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;overflow-x:auto">'
        "<b>&#9888;&#65039; Error</b><br>" + body + "</div>"
    )


def _render_warning_block(message: str) -> str:
    """Yellow warning block for non-fatal advisories."""
    import html

    body = html.escape(str(message))
    return (
        '<div style="background:#fffbeb;border:1px solid #fde68a;'
        "border-left:4px solid #d97706;border-radius:6px;padding:10px 12px;"
        'margin-bottom:6px;font-size:13px;color:#92400e">'
        "<b>&#9888;&#65039; Warning</b><br>" + body + "</div>"
    )


def _build_error_banner_payload(
    status_info: Dict[str, Any], pyg_warning: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Build ``(html, key)`` for the banner, or ``(None, None)`` if nothing to
    show. ``key`` identifies the content so a dismissal sticks until it changes.
    """
    blocks: List[str] = []
    key_parts: List[str] = []

    error_msg = status_info.get("error_message")
    if error_msg:
        blocks.append(_render_error_block(error_msg))
        key_parts.append("ERR:" + str(error_msg))

    if pyg_warning:
        blocks.append(_render_warning_block(pyg_warning))
        key_parts.append("PYG:" + str(pyg_warning))

    if not blocks:
        return None, None
    return "".join(blocks), "||".join(key_parts)


def _is_finite(v: Any) -> bool:
    """Check whether a value is a finite number (rejects inf, -inf, nan)."""
    try:
        return isinstance(v, (int, float)) and math.isfinite(v)
    except (TypeError, OverflowError):
        return False


def _parse_epoch_window(epoch_window: str) -> Optional[int]:
    """Parse 'Last N' dropdown value into an integer count, or None for 'All'."""
    if not epoch_window or epoch_window == "All":
        return None
    parts = epoch_window.split()
    if len(parts) == 2 and parts[0] == "Last":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _trim_df_to_window(df: pd.DataFrame, window: Optional[int]) -> pd.DataFrame:
    """Keep only rows from the last *window* epochs.

    If *window* is None or the DataFrame has fewer epochs than *window*,
    return the DataFrame unchanged.
    """
    if window is None or df.empty:
        return df
    max_epoch = df["epoch"].max()
    cutoff = max_epoch - window
    if cutoff <= 0:
        return df
    return df[df["epoch"] > cutoff].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Native plots on Gradio 6.30
#
# A LinePlot fixes the series it can draw (its colour domain) only when it
# rebuilds its chart, and since Gradio 5.33 (PR #11306) a new value alone does
# not rebuild it: a plot that starts empty never draws a series that appears
# later. A change of `colors_in_legend` does rebuild it, from the value already
# shown. So every event that sends plot data is followed (`.then`) by
# `_legend_step`, which sends `colors_in_legend` = the series now present, only
# when that set differs from what this browser session last received
# (gr.State), and [] when there are no rows. No placeholder rows are sent; the
# empty value is a typed empty DataFrame. Fixed colours (`color_map`) keep each
# series' colour across rebuilds.
# ---------------------------------------------------------------------------

#: The four metric plots, in output order: the column naming each series, and
#: each series' colour.
_PLOT_SERIES: Tuple[Tuple[str, Dict[str, str]], ...] = (
    ("split", {"train": "#4c78a8", "val": "#f58518"}),          # Loss Curves
    ("metric", {"MRR": "#4c78a8"}),                              # Mean Reciprocal Rank
    ("metric", {"Hits@1": "#4c78a8", "Hits@10": "#f58518"}),     # Hits@K
    ("metric", {"Learning Rate": "#4c78a8"}),                    # Learning Rate
)

#: The legends a new browser session starts with: every plot starts empty.
_NO_SERIES: List[List[str]] = [[] for _ in _PLOT_SERIES]


def _series_present(value: Any, column: str) -> Optional[List[str]]:
    """The series a plot output draws, sorted ([] for no rows), or None when
    the output leaves the plot as it is (``gr.update()`` without a value)."""
    if isinstance(value, dict):
        value = value.get("value")
    if not isinstance(value, pd.DataFrame):
        return None
    if value.empty:
        return []
    return sorted(str(name) for name in value[column].dropna().unique())


def _series_sent(plot_values: Sequence[Any]):
    """What a data step sent to the four plots, for the State `_legend_step`
    reads: per plot, the series drawn (None for a plot left as it was), or
    ``gr.update()`` when the step left all four as they were, so a no-op poll
    does not hide the series an earlier step sent before its legend step ran."""
    sent = [_series_present(value, column)
            for value, (column, _colors) in zip(plot_values, _PLOT_SERIES, strict=True)]
    if all(series is None for series in sent):
        return gr.update()
    return sent


def _legend_step(sent: Optional[Sequence[Optional[List[str]]]],
                 seen: Optional[Sequence[List[str]]]):
    """The step after each data step: per plot, ``gr.update(colors_in_legend=
    series)`` when the series last sent differ from the legend this browser
    session last received, else ``gr.update()``; and the legends now received.

    Returns:
        (loss_update, mrr_update, hits_update, lr_update, seen)
    """
    seen = [list(series) for series in (seen or _NO_SERIES)]
    updates = []
    for index, series in enumerate(sent or [None] * len(_PLOT_SERIES)):
        if series is None or series == seen[index]:
            updates.append(gr.update())
        else:
            updates.append(gr.update(colors_in_legend=list(series)))
            seen[index] = list(series)
    return (*updates, seen)


def _with_series_sent(handler: Callable[..., Tuple[Any, ...]]) -> Callable[..., Tuple[Any, ...]]:
    """Wrap an event handler whose outputs 1-4 are the four plots so that it
    also returns `_series_sent` of them, as one more output."""
    @functools.wraps(handler)
    def wrapped(*args):
        outputs = handler(*args)
        return (*outputs, _series_sent(outputs[1:5]))
    return wrapped


# ---------------------------------------------------------------------------
# Smart polling: skip re-renders when training is not active.
# Only System Resources needs continuous polling; charts, status text, and
# button states are static once training finishes (or before it starts).
# ---------------------------------------------------------------------------
_poll_cache_key: Optional[Tuple[str, str]] = None  # (status, epoch_window)

# Error-banner state (see _build_error_banner_payload).
_UNSET = object()
_pyg_warning_cache: Any = _UNSET  # None once probed and clean; str if fallback
_dismissed_banner_keys: set = set()  # content-keys the user dismissed this session
_current_banner_key: Optional[str] = None  # latest key, read by the dismiss handler


# Empty DataFrames with correct column structure for Gradio LinePlot.
# Gradio renders value=None as an icon placeholder without a title, so the
# plots always get a DataFrame — typed and empty when there is no data (the
# initial value too; see "Native plots on Gradio 6.30" above).
_EMPTY_LOSS_DF = pd.DataFrame({"epoch": pd.Series(dtype="float"),
                               "loss": pd.Series(dtype="float"),
                               "split": pd.Series(dtype="str")})
_EMPTY_METRIC_DF = pd.DataFrame({"epoch": pd.Series(dtype="float"),
                                 "value": pd.Series(dtype="float"),
                                 "metric": pd.Series(dtype="str")})


def _build_loss_df(
    train_data: List[Dict[str, Any]], val_data: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Build a DataFrame for the loss plot (train + val).

    Epochs are converted from 0-indexed (as stored in log files) to
    1-indexed to match the status display shown to the user.
    """
    rows = []

    for entry in train_data:
        epoch = entry.get("epoch")
        # Train loss can be 'total' or 'train_loss'
        loss = entry.get("total") or entry.get("train_loss")
        if epoch is not None and loss is not None and _is_finite(loss):
            rows.append({"epoch": epoch + 1, "loss": loss, "split": "train"})

    for entry in val_data:
        epoch = entry.get("epoch")
        loss = entry.get("val_loss")
        if epoch is not None and loss is not None and _is_finite(loss):
            rows.append({"epoch": epoch + 1, "loss": loss, "split": "val"})

    if not rows:
        return _EMPTY_LOSS_DF
    return pd.DataFrame(rows)


def _build_metric_df(
    data: List[Dict[str, Any]], key: str, label: str
) -> pd.DataFrame:
    """Build a single-series DataFrame for a metric.

    Epochs are converted from 0-indexed to 1-indexed.
    """
    rows = []
    for entry in data:
        epoch = entry.get("epoch")
        value = entry.get(key)
        if epoch is not None and value is not None and _is_finite(value):
            rows.append({"epoch": epoch + 1, "value": value, "metric": label})

    if not rows:
        return _EMPTY_METRIC_DF
    return pd.DataFrame(rows)


def _build_hits_df(val_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame for Hits@1 and Hits@10.

    Epochs are converted from 0-indexed to 1-indexed.
    """
    rows = []
    for entry in val_data:
        epoch = entry.get("epoch")
        h1 = entry.get("val_hits@1")
        h10 = entry.get("val_hits@10")
        if epoch is not None:
            if h1 is not None and _is_finite(h1):
                rows.append({"epoch": epoch + 1, "value": h1, "metric": "Hits@1"})
            if h10 is not None and _is_finite(h10):
                rows.append({"epoch": epoch + 1, "value": h10, "metric": "Hits@10"})

    if not rows:
        return _EMPTY_METRIC_DF
    return pd.DataFrame(rows)


def _bar_color(pct: float) -> str:
    """Return a CSS color based on utilization percentage."""
    if pct < 50:
        return "#22c55e"   # green
    if pct < 80:
        return "#eab308"   # yellow
    return "#ef4444"       # red


def _temp_color(temp_c: float) -> str:
    """Return a CSS color based on GPU temperature."""
    if temp_c < 60:
        return "#22c55e"
    if temp_c < 80:
        return "#eab308"
    return "#ef4444"


def _make_bar(label: str, value_text: str, pct: float, color: str) -> str:
    """Generate HTML for a single progress bar gauge."""
    pct_clamped = max(0, min(100, pct))
    return (
        f'<div style="margin-bottom:8px">'
        f'<div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:2px">'
        f'<span>{label}</span><span>{value_text}</span></div>'
        f'<div style="background:#e5e7eb;border-radius:4px;height:14px;overflow:hidden">'
        f'<div style="width:{pct_clamped:.1f}%;height:100%;background:{color};'
        f'border-radius:4px;transition:width 0.5s ease"></div></div></div>'
    )


def _format_resources(resources: Dict[str, Any]) -> str:
    """Format system resources as HTML dashboard with progress bar gauges."""
    bars: List[str] = []

    # On unified-memory machines (e.g. DGX Spark GB10) the GPU and CPU share one
    # physical pool, so a separate "VRAM" bar and "RAM" bar would show the SAME
    # bytes twice. When detected, drop the per-GPU VRAM bar and render a single
    # "Unified Memory" gauge (from system RAM, which sees the whole shared pool).
    unified = resources.get("unified_memory", False)

    gpu = resources.get("gpu", {})
    if gpu.get("available"):
        for dev in gpu.get("devices", []):
            name = dev.get("name", "GPU")
            idx = dev.get("index", 0)

            # GPU Utilization bar (value may be None if nvidia-smi reports [N/A])
            util_pct = dev.get("utilization_percent")
            if util_pct is None:
                bars.append(_make_bar(f"GPU {idx} Util", "N/A", 0, "#9ca3af"))
            else:
                bars.append(_make_bar(
                    f"GPU {idx} Util",
                    f"{util_pct:.0f}%",
                    util_pct,
                    _bar_color(util_pct),
                ))

            # GPU Memory bar — skipped under unified memory (shown as one shared
            # gauge below instead, to avoid double-counting the same pool).
            if not unified:
                mem_used = dev.get("memory_used_mb")
                mem_total = dev.get("memory_total_mb")
                if mem_used is None or mem_total is None or mem_total <= 0:
                    bars.append(_make_bar(f"GPU {idx} VRAM", "N/A", 0, "#9ca3af"))
                else:
                    mem_pct = mem_used / mem_total * 100
                    bars.append(_make_bar(
                        f"GPU {idx} VRAM",
                        f"{mem_used:.0f} / {mem_total:.0f} MB",
                        mem_pct,
                        _bar_color(mem_pct),
                    ))

            # Temperature bar (scale: 0-100 °C)
            temp = dev.get("temperature_c")
            if temp is None:
                bars.append(_make_bar(f"GPU {idx} Temp", "N/A", 0, "#9ca3af"))
            else:
                bars.append(_make_bar(
                    f"GPU {idx} Temp",
                    f"{temp:.0f}\u00b0C",
                    temp,
                    _temp_color(temp),
                ))

            # Add device name as a subtle label
            bars.append(
                f'<div style="font-size:11px;color:#6b7280;margin-bottom:10px">'
                f'{name}</div>'
            )
    else:
        bars.append(
            '<div style="font-size:13px;color:#6b7280;margin-bottom:8px">'
            'GPU: Not available</div>'
        )

    # Memory bar: one shared "Unified Memory" gauge on unified-memory machines,
    # otherwise the conventional separate system "RAM" gauge.
    ram = resources.get("ram", {})
    mem_label = "Unified Memory" if unified else "RAM"
    if "error" not in ram:
        total = ram.get("total_gb", 1)
        pct = ram.get("percent", 0)
        if unified and ram.get("available_gb") is not None:
            # On a shared pool "used = total - available" matches the bar % and
            # counts reclaimable page cache as free. psutil's .used can understate
            # pressure on huge-page LPDDR (e.g. GB10). Mirrors nvtop / sparkview.
            used = total - ram.get("available_gb", 0)
        else:
            used = ram.get("used_gb", 0)
        bars.append(_make_bar(
            mem_label,
            f"{used:.1f} / {total:.1f} GB",
            pct,
            _bar_color(pct),
        ))
        if unified:
            bars.append(
                '<div style="font-size:11px;color:#6b7280;margin-bottom:8px">'
                'Shared CPU + GPU pool (no separate VRAM)</div>'
            )
    else:
        bars.append(
            f'<div style="font-size:13px;color:#ef4444;margin-bottom:8px">'
            f'{mem_label}: {ram.get("error", "Unknown")}</div>'
        )

    return (
        '<div style="font-family:IBM Plex Sans,system-ui,sans-serif;padding:4px 0">'
        + "\n".join(bars)
        + "</div>"
    )


#: The fields' help, behind the shared ⓘ (`src/webui/components/help.py`):
#: ``{elem_id: (title, text)}``, registered by the tab builder. Only a
#: state-bearing text stays in ``info=``.
FIELD_HELP = {
    "data_dir": (
        "Data Directory (Workspace)",
        "Workspace folder with kg.json, node_features.pt, edge_indices.pt, etc.",
    ),
    "run_name": (
        "Run name (optional)",
        "Each run is saved in a new folder of its own: "
        "`<workspace>/runs/<architecture>/<date-time>_<name>_<id>/`. Letters, digits and "
        "hyphens are kept; blank uses \"run\". The folder is shown in Status once training "
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
        "auto: use GPU if available, otherwise CPU",
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
    "checkpoint_dropdown": (
        "Run checkpoint",
        "This architecture's runs, latest first. Click Refresh to update the list.",
    ),
    "manual_checkpoint": (
        "Checkpoint path",
        "A .pt file, in the run layout or not; checked the same way as a listed one.",
    ),
    "extend_epochs": (
        "Extra epochs (extend a finished run)",
        "Trained at the fixed rate shown in the check result.",
    ),
    "epoch_window": (
        "Epoch Window",
        "Limit charts to the most recent N epochs for detail.",
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
# Tab Builder
# =============================================================================
def create_training_tab(ctx: Any) -> None:
    """
    Build the Training Console tab content inside a gr.Blocks context.

    Must be called inside a `with gr.Tab(...)` or `with gr.Blocks()` context.
    `ctx` is the page context (`src/webui/shell.py`, the page contract): the
    fields' help goes behind its shared ⓘ (`FIELD_HELP`).
    """
    ctx.help.fields(FIELD_HELP)

    with gr.Row():
        # =====================================================================
        # Left Column: Parameters & Controls
        # =====================================================================
        with gr.Column(scale=1):
            # -----------------------------------------------------------------
            # Data Paths
            # -----------------------------------------------------------------
            with gr.Accordion("Data Paths", open=False) as data_paths:
                saved_train_cfg = _load_training_config()
                saved_data_dir = saved_train_cfg.get("data_dir", DEFAULT_WORKSPACE)
                with gr.Group():
                    data_dir = gr.Textbox(
                        label="Data Directory (Workspace)",
                        value=f"{DISPLAY_PREFIX}{saved_data_dir}",
                        elem_id="data_dir",
                    )
                    run_name = gr.Textbox(
                        label="Run name (optional)",
                        value="",
                        max_lines=1,
                        elem_id="run_name",
                    )
                with gr.Row():
                    save_path_btn = gr.Button("💾 Save Path", size="sm", variant="secondary")
                    reset_path_btn = gr.Button("↩️ Reset Default", size="sm", variant="secondary")
                path_msg = gr.Markdown(value="")

            # -----------------------------------------------------------------
            # Tier 1 — Basic Parameters (always visible)
            # -----------------------------------------------------------------
            gr.Markdown("### Basic Training Parameters")
            with gr.Group():
                num_epochs = gr.Number(
                    label="Epochs",
                    value=100,
                    precision=0,
                    elem_id="num_epochs",
                )
                learning_rate = gr.Number(
                    label="Learning Rate",
                    value=1e-4,
                    precision=8,
                    elem_id="learning_rate",
                )
                batch_size = gr.Dropdown(
                    label="Batch Size",
                    choices=["8", "16", "32", "64", "128", "256", "512", "1024", "2048"],
                    value="32",
                    elem_id="batch_size",
                )
                conv_type = gr.Radio(
                    label="GNN Conv Type",
                    choices=["gat", "hgt", "sage"],
                    value="gat",
                    elem_id="conv_type",
                )
                device = gr.Radio(
                    label="Device",
                    choices=["auto", "cuda", "cpu"],
                    value="auto",
                    elem_id="device",
                )
                seed = gr.Number(
                    label="Seed",
                    value=42,
                    precision=0,
                    elem_id="seed",
                )
                randomize_seed = gr.Checkbox(
                    label="🎲 Randomize seed",
                    value=False,
                    elem_id="randomize_seed",
                )

            # -----------------------------------------------------------------
            # Tier 2 — Advanced Parameters (collapsible)
            # -----------------------------------------------------------------
            with gr.Accordion("Advanced Parameters", open=False):
                with gr.Group():
                    hidden_dim = gr.Dropdown(
                        label="Hidden Dim",
                        choices=["128", "256", "512"],
                        value="256",
                        elem_id="hidden_dim",
                    )
                    num_layers = gr.Slider(
                        label="Num Layers",
                        minimum=2,
                        maximum=8,
                        value=4,
                        step=1,
                        elem_id="num_layers",
                    )
                    dropout = gr.Slider(
                        label="Dropout",
                        minimum=0.0,
                        maximum=0.5,
                        value=0.1,
                        step=0.01,
                        elem_id="dropout",
                    )
                    weight_decay = gr.Slider(
                        label="Weight Decay",
                        minimum=1e-5,
                        maximum=0.1,
                        value=0.01,
                        step=1e-5,
                        elem_id="weight_decay",
                    )
                    scheduler_type = gr.Dropdown(
                        label="Scheduler",
                        choices=["cosine", "onecycle", "linear", "none"],
                        value="cosine",
                        elem_id="scheduler_type",
                    )
                    warmup_steps = gr.Number(
                        label="Warmup Steps",
                        value=500,
                        precision=0,
                        elem_id="warmup_steps",
                    )
                    min_lr_ratio = gr.Number(
                        label="Min LR Ratio",
                        value=0.01,
                        elem_id="min_lr_ratio",
                    )
                    early_stopping_patience = gr.Number(
                        label="Early Stopping Patience",
                        value=10,
                        precision=0,
                        elem_id="early_stopping_patience",
                    )

                gr.Markdown("##### Loss Weights")
                with gr.Group():
                    diagnosis_weight = gr.Slider(
                        label="Diagnosis Weight",
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        elem_id="diagnosis_weight",
                    )
                    link_prediction_weight = gr.Slider(
                        label="Link Prediction Weight",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.5,
                        step=0.1,
                        elem_id="link_prediction_weight",
                    )
                    contrastive_weight = gr.Slider(
                        label="Contrastive Weight",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.3,
                        step=0.1,
                        elem_id="contrastive_weight",
                    )
                    ortholog_weight = gr.Slider(
                        label="Ortholog Weight",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.2,
                        step=0.1,
                        elem_id="ortholog_weight",
                    )

            # -----------------------------------------------------------------
            # Tier 3 — Expert Parameters (collapsible, default closed)
            # -----------------------------------------------------------------
            with gr.Accordion("Expert Parameters", open=False):
                with gr.Group():
                    gradient_accumulation_steps = gr.Number(
                        label="Gradient Accumulation Steps",
                        value=1,
                        precision=0,
                        elem_id="gradient_accumulation_steps",
                    )
                    max_grad_norm = gr.Number(
                        label="Max Grad Norm",
                        value=1.0,
                        elem_id="max_grad_norm",
                    )
                    num_heads = gr.Dropdown(
                        label="Attention Heads",
                        choices=["4", "8", "16"],
                        value="8",
                        elem_id="num_heads",
                    )
                    use_ortholog_gate = gr.Checkbox(
                        label="Use Ortholog Gate",
                        value=True,
                        elem_id="use_ortholog_gate",
                    )
                    amp_mode = gr.Radio(
                        label="Mixed Precision (AMP)",
                        choices=["Off", "float16", "bfloat16"],
                        value="float16",
                        elem_id="amp_mode",
                    )
                    temperature = gr.Slider(
                        label="Contrastive Temperature",
                        minimum=0.01,
                        maximum=1.0,
                        value=0.07,
                        step=0.01,
                        elem_id="temperature",
                    )
                    label_smoothing = gr.Slider(
                        label="Label Smoothing",
                        minimum=0.0,
                        maximum=0.3,
                        value=0.1,
                        step=0.01,
                        elem_id="label_smoothing",
                    )
                    margin = gr.Slider(
                        label="Margin",
                        minimum=0.1,
                        maximum=3.0,
                        value=1.0,
                        step=0.1,
                        elem_id="margin",
                    )
                    num_neighbors_str = gr.Textbox(
                        label="Num Neighbors (comma-separated)",
                        value="15, 10, 5",
                        elem_id="num_neighbors_str",
                    )
                    max_subgraph_nodes = gr.Number(
                        label="Max Subgraph Nodes",
                        value=5000,
                        precision=0,
                        elem_id="max_subgraph_nodes",
                    )

            # torch.compile moved to the Runtime Settings tab (it is an
            # execution/runtime knob, applied per training run).

            # -----------------------------------------------------------------
            # Control Buttons
            # -----------------------------------------------------------------
            gr.Markdown("### Control")
            with gr.Row():
                start_btn = gr.Button("Start Training", variant="primary")
                stop_btn = gr.Button("Stop Training", variant="stop", interactive=False)
                resume_btn = gr.Button("Continue Training")

            # Stop asks first, saying what stopping loses (contract M3a).
            with gr.Column(visible=False) as stop_confirm:
                stop_confirm_text = gr.Markdown("")
                with gr.Row():
                    stop_confirm_btn = gr.Button("Stop training", variant="stop", size="sm")
                    stop_keep_btn = gr.Button("Keep training", size="sm")

            # Continue a run from a checkpoint (contract M3a-3, M3a-4): one
            # check for a listed checkpoint and a typed path; what happens is
            # decided by the checkpoint's own record and shown before start.
            with gr.Accordion("Continue Training (from a saved checkpoint)", open=False):
                gr.Markdown(
                    "An unfinished run is resumed with its own settings; a finished run "
                    "can be extended at a fixed rate. Click **Check checkpoint** to see "
                    "what will happen, then **Continue Training**."
                )
                continue_source = gr.Radio(
                    choices=[CONTINUE_FROM_LIST, CONTINUE_FROM_PATH],
                    value=CONTINUE_FROM_LIST,
                    label="Checkpoint source",
                    elem_id="continue_source",
                )
                with gr.Row():
                    checkpoint_dropdown = gr.Dropdown(
                        label="Run checkpoint",
                        choices=[],
                        value=None,
                        elem_id="checkpoint_dropdown",
                    )
                    refresh_ckpt_btn = gr.Button("Refresh", size="sm", scale=0)
                manual_checkpoint = gr.Textbox(
                    label="Checkpoint path",
                    value="",
                    max_lines=1,
                    visible=False,
                    elem_id="manual_checkpoint",
                )
                # No widget minimum: hidden and empty, a Number is sent as 0,
                # which a minimum would reject before any handler runs. Empty
                # or 0 means "not extending"; the run plan refuses below 1.
                extend_epochs = gr.Number(
                    label="Extra epochs (extend a finished run)",
                    value=None,
                    precision=0,
                    visible=False,
                    elem_id="extend_epochs",
                )
                check_continue_btn = gr.Button("Check checkpoint", size="sm")
                continue_preview = gr.Markdown(
                    _CONTINUE_PROMPT,
                    elem_id="continue_preview",
                )

            status_display = gr.Markdown(
                value="**Status**: IDLE",
                label="Status",
                elem_id="status_display",
            )

            # Dedicated error/warning banner: hidden during normal operation,
            # shown (red error / yellow warning) only when the backend reports
            # a problem, separate from the status text above. Manually
            # dismissable; auto-clears when the underlying problem resolves.
            error_banner = gr.HTML(
                value="",
                visible=False,
                elem_id="error_banner",
                padding=True,  # Gradio 5's default; Gradio 6 defaults to none
            )
            error_dismiss_btn = gr.Button(
                "✕ Dismiss",
                visible=False,
                size="sm",
                elem_id="error_dismiss_btn",
            )

        # =====================================================================
        # Right Column: Metrics & Resources
        # =====================================================================
        with gr.Column(scale=2):
            gr.Markdown("### Training Metrics")

            with gr.Row():
                epoch_window = gr.Dropdown(
                    label="Epoch Window",
                    choices=["All", "Last 10", "Last 20", "Last 50", "Last 100"],
                    value="All",
                    elem_id="epoch_window",
                    scale=1,
                )

            # No width= (Gradio 6 has none; the row sizes the plots). y_lim
            # starts the y axis at 0, as Gradio 5 did by default.
            with gr.Row():
                loss_plot = gr.LinePlot(
                    value=_EMPTY_LOSS_DF,
                    x="epoch",
                    y="loss",
                    color="split",
                    color_map=_PLOT_SERIES[0][1],
                    title="Loss Curves",
                    x_title="Epoch",
                    y_title="Loss",
                    y_lim=[0, None],
                    height=280,
                    elem_id="loss_plot",
                )
                mrr_plot = gr.LinePlot(
                    value=_EMPTY_METRIC_DF,
                    x="epoch",
                    y="value",
                    color="metric",
                    color_map=_PLOT_SERIES[1][1],
                    title="Mean Reciprocal Rank",
                    x_title="Epoch",
                    y_title="MRR",
                    y_lim=[0, None],
                    height=280,
                    elem_id="mrr_plot",
                )

            with gr.Row():
                hits_plot = gr.LinePlot(
                    value=_EMPTY_METRIC_DF,
                    x="epoch",
                    y="value",
                    color="metric",
                    color_map=_PLOT_SERIES[2][1],
                    title="Hits@K",
                    x_title="Epoch",
                    y_title="Hits",
                    y_lim=[0, None],
                    height=280,
                    elem_id="hits_plot",
                )
                lr_plot = gr.LinePlot(
                    value=_EMPTY_METRIC_DF,
                    x="epoch",
                    y="value",
                    color="metric",
                    color_map=_PLOT_SERIES[3][1],
                    title="Learning Rate",
                    x_title="Epoch",
                    y_title="LR",
                    y_lim=[0, None],
                    height=280,
                    elem_id="lr_plot",
                )

            with gr.Row():
                export_csv_btn = gr.Button("Export Metrics CSV", size="sm")
                export_csv_file = gr.File(
                    label="Download",
                    visible=False,
                    elem_id="export_csv_file",
                )

            gr.Markdown("### System Resources")
            resource_display = gr.HTML(
                value='<div style="font-size:13px;color:#6b7280">Loading resource info...</div>',
                elem_id="resource_display",
                padding=True,  # Gradio 5's default; Gradio 6 defaults to none
            )

    # =========================================================================
    # Collect all parameter inputs in order matching _collect_config signature
    # =========================================================================
    all_params = [
        # Paths
        data_dir, run_name,
        # Tier 1
        num_epochs, learning_rate, batch_size, conv_type, device,
        seed,
        # Tier 2
        hidden_dim, num_layers, dropout, weight_decay, scheduler_type,
        warmup_steps, min_lr_ratio, early_stopping_patience,
        diagnosis_weight, link_prediction_weight, contrastive_weight, ortholog_weight,
        # Tier 3
        gradient_accumulation_steps, max_grad_norm, num_heads,
        use_ortholog_gate, amp_mode,
        temperature, label_smoothing, margin, num_neighbors_str, max_subgraph_nodes,
    ]

    # =========================================================================
    # Event Handlers
    # =========================================================================
    # Shared output list: status + 4 plots + 3 buttons
    plots = [loss_plot, mrr_plot, hits_plot, lr_plot]
    btn_outputs = [status_display, *plots, start_btn, stop_btn, resume_btn]

    # The plots' legends (see "Native plots on Gradio 6.30"): what the last
    # data step sent, and what this browser session's plots were last told.
    # Hidden progress: after each poll tick, the plots must not flash.
    plot_series_sent = gr.State(None)
    plot_series_seen = gr.State(_NO_SERIES)
    legend_step = {"fn": _legend_step, "inputs": [plot_series_sent, plot_series_seen],
                   "outputs": plots + [plot_series_seen], "show_progress": "hidden"}

    start_btn.click(
        fn=_with_series_sent(_on_start),
        inputs=all_params + [randomize_seed],
        outputs=btn_outputs + [seed, plot_series_sent],
    ).then(**legend_step).then(**ctx.chip_refresh)
    # Lock/grey the seed field while "Randomize seed" is on (Start draws + echoes
    # the actual seed used).
    randomize_seed.change(
        fn=_seed_field_lock,
        inputs=[randomize_seed],
        outputs=[seed],
    )
    stop_btn.click(
        fn=_on_stop_request,
        inputs=[],
        outputs=[stop_confirm, stop_confirm_text],
    )
    stop_confirm_btn.click(
        fn=_on_stop,
        inputs=[],
        outputs=btn_outputs + [stop_confirm],
    ).then(**ctx.chip_refresh)
    stop_keep_btn.click(fn=_on_stop_cancel, inputs=[], outputs=[stop_confirm])
    continue_inputs = [continue_source, checkpoint_dropdown, manual_checkpoint, device,
                       extend_epochs]
    check_continue_btn.click(
        fn=_on_check_continue,
        inputs=continue_inputs,
        outputs=[continue_preview, extend_epochs],
    )
    resume_btn.click(
        fn=_with_series_sent(_on_resume),
        inputs=[data_dir, run_name, device, continue_source, checkpoint_dropdown,
                manual_checkpoint, extend_epochs],
        outputs=btn_outputs + [continue_preview, extend_epochs, plot_series_sent],
    ).then(**legend_step).then(**ctx.chip_refresh)
    continue_source.change(
        fn=_toggle_continue_source,
        inputs=[continue_source],
        outputs=[checkpoint_dropdown, refresh_ckpt_btn, manual_checkpoint],
    )
    # A changed selection, or source, makes the last check and its Extra
    # epochs stale: both are cleared (contract M3a review of 094b90d, P2-2).
    # `.change` fires on updates from other events too (Refresh, conv_type), so
    # the run list and the typed path clear only when they are the source.
    continue_source.change(fn=_on_continue_selection_changed, inputs=[],
                           outputs=[extend_epochs, continue_preview])
    checkpoint_dropdown.change(fn=_on_listed_changed, inputs=[continue_source],
                               outputs=[extend_epochs, continue_preview])
    manual_checkpoint.change(fn=_on_manual_changed, inputs=[continue_source],
                             outputs=[extend_epochs, continue_preview])
    device.change(fn=_on_continue_device_changed, inputs=[], outputs=[continue_preview])
    refresh_ckpt_btn.click(
        fn=_refresh_checkpoints,
        inputs=[data_dir, conv_type],
        outputs=[checkpoint_dropdown],
    )
    # Auto-relink the resume list to the selected training architecture: changing
    # conv_type repopulates the dropdown with that architecture's checkpoints, so
    # a stale cross-architecture selection can't linger. (data_dir is free text —
    # not bound here to avoid re-scanning on every keystroke; the manual Refresh
    # and the _on_resume guard cover it.)
    conv_type.change(
        fn=_refresh_checkpoints,
        inputs=[data_dir, conv_type],
        outputs=[checkpoint_dropdown],
    )
    save_path_btn.click(
        fn=_save_training_config,
        inputs=[data_dir],
        outputs=[path_msg],
    )
    # Datasets › Use for training › sets ctx.training_workspace; until the run
    # bar (UI-C) reads it directly, it fills this session's Data Directory.
    ctx.training_workspace.change(
        fn=_from_datasets,
        inputs=[ctx.training_workspace],
        outputs=[data_dir, data_paths],
        show_progress="hidden",
    )
    reset_path_btn.click(
        fn=_reset_training_config,
        inputs=[],
        outputs=[data_dir, path_msg],
    )

    def _handle_export():
        path = _export_metrics_csv()
        if path is None:
            return gr.update(visible=False, value=None)
        return gr.update(visible=True, value=path)

    export_csv_btn.click(
        fn=_handle_export,
        inputs=[],
        outputs=[export_csv_file],
    )

    def _dismiss_error_banner():
        # Remember this exact content so the poll loop keeps it hidden until the
        # underlying problem changes (a new/different error re-shows on its own).
        if _current_banner_key is not None:
            _dismissed_banner_keys.add(_current_banner_key)
        return gr.update(visible=False, value=""), gr.update(visible=False)

    error_dismiss_btn.click(
        fn=_dismiss_error_banner,
        inputs=[],
        outputs=[error_banner, error_dismiss_btn],
    )

    # HGT requires float32 (AMP disabled) — update UI when conv type changes.
    # The refusal is state, so it stays visible text in `info=`; the field's
    # ordinary help is behind its ⓘ (FIELD_HELP), and "" clears the refusal
    # (Gradio 6.30 keeps an info sent as None).
    conv_type.change(
        fn=_on_conv_type_change,
        inputs=[conv_type],
        outputs=[amp_mode],
    )

    conv_type.change(
        fn=_on_conv_type_change,
        inputs=[conv_type],
        outputs=[amp_mode],
    )

    # =========================================================================
    # Polling Timer (1.5 second interval)
    # =========================================================================
    # Gradio 6 does not tick a timer in a tab not yet opened, so this one starts
    # when Research › Training is first opened (the page then mounts and stays
    # mounted). The page contract's shared timers (`ctx.timer_fast`) are the
    # ones outside gr.Tabs; UI-C moves this poll onto them.
    timer = gr.Timer(value=1.5)
    timer.tick(
        fn=_with_series_sent(_poll_status),
        inputs=[epoch_window],
        outputs=[
            status_display,
            loss_plot,
            mrr_plot,
            hits_plot,
            lr_plot,
            resource_display,
            error_banner,
            error_dismiss_btn,
            start_btn,
            stop_btn,
            resume_btn,
            plot_series_sent,
        ],
    ).then(**legend_step)


__all__ = ["create_training_tab"]
