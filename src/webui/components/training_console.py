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

Dependencies:
    - gradio: UI components
    - pandas: DataFrame for chart data
    - src.api.services.training_manager: training_manager singleton

Version: 1.0.0
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

from src.api.services.training_manager import training_manager

logger = logging.getLogger(__name__)


def _collect_config(
    # Paths
    data_dir: str,
    output_dir: str,
    checkpoint_dir: str,
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
    use_amp: bool,
    amp_dtype: str,
    temperature: float,
    label_smoothing: float,
    margin: float,
    num_neighbors_str: str,
    max_subgraph_nodes: int,
) -> Dict[str, Any]:
    """Collect all parameter widgets into a config dict."""
    # Parse comma-separated neighbors
    try:
        num_neighbors = [int(x.strip()) for x in num_neighbors_str.split(",")]
    except (ValueError, AttributeError):
        num_neighbors = [15, 10, 5]

    # Strip the display prefix — training subprocess uses paths relative to project root
    def _strip_prefix(path: str) -> str:
        p = path.strip()
        if p.startswith("SHEPHERD-Advanced/"):
            p = p[len("SHEPHERD-Advanced/"):]
        return p

    config: Dict[str, Any] = {
        # Paths (strip display prefix so backend receives relative-to-project-root paths)
        "data_dir": _strip_prefix(data_dir) or "data/processed",
        "output_dir": _strip_prefix(output_dir) or "outputs",
        "checkpoint_dir": _strip_prefix(checkpoint_dir) or "models/checkpoints",
        # Tier 1
        "num_epochs": int(num_epochs),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "conv_type": conv_type,
        "device": device,
        "seed": int(seed),
        # Tier 2
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "weight_decay": float(weight_decay),
        "scheduler_type": scheduler_type,
        "warmup_steps": int(warmup_steps),
        "early_stopping_patience": int(early_stopping_patience),
        "diagnosis_weight": float(diagnosis_weight),
        "link_prediction_weight": float(link_prediction_weight),
        "contrastive_weight": float(contrastive_weight),
        "ortholog_weight": float(ortholog_weight),
        # Tier 3
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "max_grad_norm": float(max_grad_norm),
        "num_heads": int(num_heads),
        "use_ortholog_gate": use_ortholog_gate,
        "use_amp": use_amp,
        "amp_dtype": amp_dtype,
        "temperature": float(temperature),
        "label_smoothing": float(label_smoothing),
        "margin": float(margin),
        "num_neighbors": num_neighbors,
        "max_subgraph_nodes": int(max_subgraph_nodes),
    }

    return config


def _on_start(*args):
    """Handle Start Training button click.

    Returns status text, 4 empty plot DataFrames (to clear stale charts),
    and 3 button updates.
    """
    config = _collect_config(*args)
    result = training_manager.start_training(config)
    if result.get("success"):
        return (
            f"**Status**: RUNNING\n**PID**: {result.get('pid')}\n**Phase**: Starting...",
            _EMPTY_LOSS_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF,
            gr.update(interactive=False),   # disable start
            gr.update(interactive=True),    # enable stop
            gr.update(interactive=False),   # disable resume
        )
    return (
        f"Failed: {result.get('error', 'Unknown error')}",
        _EMPTY_LOSS_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF,
        gr.update(interactive=True),    # keep start enabled
        gr.update(interactive=False),   # keep stop disabled
        gr.update(interactive=True),    # keep resume enabled
    )


def _on_stop():
    """Handle Stop Training button click.

    Returns status text, gr.update() for 4 plots (no change), and 3 button updates.
    """
    result = training_manager.stop_training()
    no_change = gr.update()
    if result.get("success"):
        return (
            "**Status**: COMPLETED\nTraining stopped by user.",
            no_change, no_change, no_change, no_change,
            gr.update(interactive=True),    # re-enable start
            gr.update(interactive=False),   # disable stop
            gr.update(interactive=True),    # re-enable resume
        )
    return (
        f"Failed: {result.get('error', 'Unknown error')}",
        no_change, no_change, no_change, no_change,
        gr.update(interactive=False),   # keep start disabled while running
        gr.update(interactive=True),    # keep stop enabled
        gr.update(interactive=False),   # keep resume disabled while running
    )


def _on_resume(*args):
    """Handle Resume Training button click (same as start with resume path).

    The last argument is the checkpoint dropdown selection (may be None).
    All preceding arguments are passed to _collect_config.

    Returns status text, 4 empty plot DataFrames (to clear stale charts),
    and 3 button updates.
    """
    # Last arg is checkpoint dropdown; rest go to _collect_config
    checkpoint_selection = args[-1] if args else None
    config = _collect_config(*args[:-1])

    # Use checkpoint from dropdown selection
    if checkpoint_selection and str(checkpoint_selection).strip():
        config["resume_from"] = str(checkpoint_selection).strip()

    # If no dropdown selection, auto-find the last checkpoint
    if "resume_from" not in config:
        checkpoints = training_manager.get_checkpoints()
        if checkpoints:
            # Use the last.pt if it exists
            for ckpt in checkpoints:
                if ckpt["filename"] == "last.pt":
                    config["resume_from"] = ckpt["filepath"]
                    break
            if "resume_from" not in config:
                config["resume_from"] = checkpoints[0]["filepath"]
        else:
            return (
                "No checkpoints found to resume from",
                _EMPTY_LOSS_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF,
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=True),
            )

    result = training_manager.start_training(config)
    if result.get("success"):
        ckpt_name = Path(config["resume_from"]).name
        return (
            f"**Status**: RUNNING\n**PID**: {result.get('pid')}\n**Phase**: Resuming from {ckpt_name}...",
            _EMPTY_LOSS_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF,
            gr.update(interactive=False),   # disable start
            gr.update(interactive=True),    # enable stop
            gr.update(interactive=False),   # disable resume
        )
    return (
        f"Failed: {result.get('error', 'Unknown error')}",
        _EMPTY_LOSS_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF,
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=True),
    )


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


def _refresh_checkpoints():
    """Refresh checkpoint dropdown choices."""
    checkpoints = training_manager.get_checkpoints()
    choices = []
    for ckpt in checkpoints:
        name = ckpt["filename"]
        size = ckpt.get("size_mb", 0)
        epoch = ckpt.get("epoch")
        label = f"{name} ({size:.1f} MB"
        if epoch is not None:
            label += f", epoch {epoch}"
        label += ")"
        choices.append((label, ckpt["filepath"]))
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


_IDLE_RESOURCE_HTML = (
    '<div style="font-family:system-ui,sans-serif;padding:4px 0">'
    '<div style="font-size:13px;color:#6b7280">Loading resource info...</div>'
    '</div>'
)


def _poll_status():
    """
    Poll training status and metrics. Called by gr.Timer every 1.5 seconds.

    Returns:
        (status_text, loss_df, mrr_df, hits_df, lr_df, resource_text,
         start_btn_update, stop_btn_update, resume_btn_update)
    """
    status_info = training_manager.get_status()
    s = status_info.get("status", "idle")

    # Button state
    is_running = s in ("running", "stopping")
    start_update = gr.update(interactive=not is_running)
    stop_update = gr.update(interactive=is_running)
    resume_update = gr.update(interactive=not is_running)

    # System resources are always polled so users can check load before training
    resources = training_manager.get_system_resources()
    resource_text = _format_resources(resources)

    if s == "idle":
        # No training ever started — skip metrics file reads
        return (
            _format_status(status_info),
            _EMPTY_LOSS_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF, _EMPTY_METRIC_DF,
            resource_text,
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

    return (
        status_text, loss_df, mrr_df, hits_df, lr_df, resource_text,
        start_update, stop_update, resume_update,
    )


def _format_status(status_info: Dict[str, Any]) -> str:
    """Format training status as a readable string."""
    s = status_info.get("status", "idle")
    lines = [f"**Status**: {s.upper()}"]

    if status_info.get("pid"):
        lines.append(f"**PID**: {status_info['pid']}")

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

    if status_info.get("error_message"):
        err = status_info["error_message"]
        if "\n" in err:
            lines.append(f"**Error**:\n```\n{err}\n```")
        else:
            lines.append(f"**Error**: {err}")

    return "\n".join(lines)


def _is_finite(v: Any) -> bool:
    """Check whether a value is a finite number (rejects inf, -inf, nan)."""
    try:
        return isinstance(v, (int, float)) and math.isfinite(v)
    except (TypeError, OverflowError):
        return False


# Empty DataFrames with correct column structure for Gradio LinePlot.
# Gradio 5.20+ renders value=None as a broken icon, so we always
# return a DataFrame — empty when there is no data.
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

    gpu = resources.get("gpu", {})
    if gpu.get("available"):
        for dev in gpu.get("devices", []):
            name = dev.get("name", "GPU")
            idx = dev.get("index", 0)

            # GPU Utilization bar
            util_pct = dev.get("utilization_percent", 0)
            bars.append(_make_bar(
                f"GPU {idx} Util",
                f"{util_pct:.0f}%",
                util_pct,
                _bar_color(util_pct),
            ))

            # GPU Memory bar
            mem_used = dev.get("memory_used_mb", 0)
            mem_total = dev.get("memory_total_mb", 1)
            mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0
            bars.append(_make_bar(
                f"GPU {idx} VRAM",
                f"{mem_used:.0f} / {mem_total:.0f} MB",
                mem_pct,
                _bar_color(mem_pct),
            ))

            # Temperature bar (scale: 0-100 °C)
            temp = dev.get("temperature_c", 0)
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

    # RAM bar
    ram = resources.get("ram", {})
    if "error" not in ram:
        used = ram.get("used_gb", 0)
        total = ram.get("total_gb", 1)
        pct = ram.get("percent", 0)
        bars.append(_make_bar(
            "RAM",
            f"{used:.1f} / {total:.1f} GB",
            pct,
            _bar_color(pct),
        ))
    else:
        bars.append(
            f'<div style="font-size:13px;color:#ef4444;margin-bottom:8px">'
            f'RAM: {ram.get("error", "Unknown")}</div>'
        )

    return (
        '<div style="font-family:system-ui,sans-serif;padding:4px 0">'
        + "\n".join(bars)
        + "</div>"
    )


# =============================================================================
# Tab Builder
# =============================================================================
def create_training_tab() -> None:
    """
    Build the Training Console tab content inside a gr.Blocks context.

    Must be called inside a `with gr.Tab(...)` or `with gr.Blocks()` context.
    """

    with gr.Row():
        # =====================================================================
        # Left Column: Parameters & Controls
        # =====================================================================
        with gr.Column(scale=1):
            # -----------------------------------------------------------------
            # Data Paths
            # -----------------------------------------------------------------
            with gr.Accordion("Data Paths", open=False):
                with gr.Group():
                    data_dir = gr.Textbox(
                        label="Data Directory",
                        info="Processed data with node_features.pt, edge_indices.pt, etc. (relative to project root)",
                        value="SHEPHERD-Advanced/data/processed",
                        elem_id="data_dir",
                    )
                    output_dir = gr.Textbox(
                        label="Output Directory",
                        info="Where final metrics and results are saved. (relative to project root)",
                        value="SHEPHERD-Advanced/outputs",
                        elem_id="output_dir",
                    )
                    checkpoint_dir = gr.Textbox(
                        label="Checkpoint Directory",
                        info="Where model checkpoints (.pt files) are saved. (relative to project root)",
                        value="SHEPHERD-Advanced/models/checkpoints",
                        elem_id="checkpoint_dir",
                    )

            # -----------------------------------------------------------------
            # Tier 1 — Basic Parameters (always visible)
            # -----------------------------------------------------------------
            gr.Markdown("### Basic Training Parameters")
            with gr.Group():
                num_epochs = gr.Number(
                    label="Epochs",
                    info="Smoke test: 2-5 | Quick: 10-50 | Full: 100-500",
                    value=100,
                    minimum=1,
                    maximum=10000,
                    precision=0,
                    elem_id="num_epochs",
                )
                learning_rate = gr.Slider(
                    label="Learning Rate",
                    info="Initial LR. Recommended: 1e-4 (GAT) | 5e-5 (HGT). Use with scheduler.",
                    minimum=1e-6,
                    maximum=1e-1,
                    value=1e-4,
                    step=1e-6,
                    elem_id="learning_rate",
                )
                batch_size = gr.Dropdown(
                    label="Batch Size",
                    info="Samples per step. Larger = faster but more VRAM. 32 is a safe default.",
                    choices=["8", "16", "32", "64"],
                    value="32",
                    elem_id="batch_size",
                )
                conv_type = gr.Radio(
                    label="GNN Conv Type",
                    info="GAT: attention-based (default) | HGT: heterogeneous transformer | SAGE: mean aggregation",
                    choices=["gat", "hgt", "sage"],
                    value="gat",
                    elem_id="conv_type",
                )
                device = gr.Radio(
                    label="Device",
                    info="auto: use GPU if available, otherwise CPU",
                    choices=["auto", "cuda", "cpu"],
                    value="auto",
                    elem_id="device",
                )
                seed = gr.Number(
                    label="Seed",
                    info="Random seed for reproducibility. Change to try different initializations.",
                    value=42,
                    minimum=0,
                    precision=0,
                    elem_id="seed",
                )

            # -----------------------------------------------------------------
            # Tier 2 — Advanced Parameters (collapsible)
            # -----------------------------------------------------------------
            with gr.Accordion("Advanced Parameters", open=False):
                with gr.Group():
                    hidden_dim = gr.Dropdown(
                        label="Hidden Dim",
                        info="GNN embedding dimension. 256 balances quality/speed. 512 for large KGs.",
                        choices=["128", "256", "512"],
                        value="256",
                        elem_id="hidden_dim",
                    )
                    num_layers = gr.Slider(
                        label="Num Layers",
                        info="GNN message-passing depth. 3-4 typical. More layers = wider receptive field.",
                        minimum=2,
                        maximum=8,
                        value=4,
                        step=1,
                        elem_id="num_layers",
                    )
                    dropout = gr.Slider(
                        label="Dropout",
                        info="Regularization. 0.1 typical. Increase to 0.2-0.3 if overfitting.",
                        minimum=0.0,
                        maximum=0.5,
                        value=0.1,
                        step=0.01,
                        elem_id="dropout",
                    )
                    weight_decay = gr.Slider(
                        label="Weight Decay",
                        info="L2 regularization. 0.01 typical for AdamW. Lower (1e-4) if underfitting.",
                        minimum=1e-5,
                        maximum=0.1,
                        value=0.01,
                        step=1e-5,
                        elem_id="weight_decay",
                    )
                    scheduler_type = gr.Dropdown(
                        label="Scheduler",
                        info="cosine: smooth decay (recommended) | onecycle: aggressive | linear: steady",
                        choices=["cosine", "onecycle", "linear", "none"],
                        value="cosine",
                        elem_id="scheduler_type",
                    )
                    warmup_steps = gr.Number(
                        label="Warmup Steps",
                        info="Steps to linearly ramp LR from 0. 500 typical. Set 0 to disable.",
                        value=500,
                        minimum=0,
                        precision=0,
                        elem_id="warmup_steps",
                    )
                    early_stopping_patience = gr.Number(
                        label="Early Stopping Patience",
                        info="Epochs without improvement before stopping. 10 is a good default.",
                        value=10,
                        minimum=1,
                        precision=0,
                        elem_id="early_stopping_patience",
                    )

                gr.Markdown("##### Loss Weights")
                with gr.Group():
                    diagnosis_weight = gr.Slider(
                        label="Diagnosis Weight",
                        info="Primary task weight. Keep at 1.0 (anchor). Other weights are relative to this.",
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        elem_id="diagnosis_weight",
                    )
                    link_prediction_weight = gr.Slider(
                        label="Link Prediction Weight",
                        info="KG structure learning. 0.3-0.5 typical. Higher improves KG representation.",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.5,
                        step=0.1,
                        elem_id="link_prediction_weight",
                    )
                    contrastive_weight = gr.Slider(
                        label="Contrastive Weight",
                        info="Phenotype-disease similarity learning. 0.2-0.5 typical.",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.3,
                        step=0.1,
                        elem_id="contrastive_weight",
                    )
                    ortholog_weight = gr.Slider(
                        label="Ortholog Weight",
                        info="Cross-species gene mapping. 0.1-0.3 typical. Set 0 to disable.",
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
                        info="Simulate larger batch by accumulating N steps. Useful when VRAM-limited.",
                        value=1,
                        minimum=1,
                        precision=0,
                        elem_id="gradient_accumulation_steps",
                    )
                    max_grad_norm = gr.Number(
                        label="Max Grad Norm",
                        info="Gradient clipping threshold. 1.0 typical. Prevents training instability.",
                        value=1.0,
                        minimum=0.01,
                        elem_id="max_grad_norm",
                    )
                    num_heads = gr.Dropdown(
                        label="Attention Heads",
                        info="Multi-head attention count. 8 typical. Must evenly divide hidden_dim.",
                        choices=["4", "8", "16"],
                        value="8",
                        elem_id="num_heads",
                    )
                    use_ortholog_gate = gr.Checkbox(
                        label="Use Ortholog Gate",
                        info="Learnable gate for cross-species gene signals. Disable if no ortholog data.",
                        value=True,
                        elem_id="use_ortholog_gate",
                    )
                    use_amp = gr.Checkbox(
                        label="Automatic Mixed Precision",
                        info="FP16/BF16 training. Faster & less VRAM on modern GPUs. Disable if NaN issues.",
                        value=True,
                        elem_id="use_amp",
                    )
                    amp_dtype = gr.Radio(
                        label="AMP Dtype",
                        info="float16: wider GPU support | bfloat16: better stability (RTX 30/40, A100+)",
                        choices=["float16", "bfloat16"],
                        value="float16",
                        elem_id="amp_dtype",
                    )
                    temperature = gr.Slider(
                        label="Contrastive Temperature",
                        info="Lower = sharper similarity. 0.07 typical for contrastive learning.",
                        minimum=0.01,
                        maximum=1.0,
                        value=0.07,
                        step=0.01,
                        elem_id="temperature",
                    )
                    label_smoothing = gr.Slider(
                        label="Label Smoothing",
                        info="Softens one-hot labels. 0.1 typical. Prevents overconfident predictions.",
                        minimum=0.0,
                        maximum=0.3,
                        value=0.1,
                        step=0.01,
                        elem_id="label_smoothing",
                    )
                    margin = gr.Slider(
                        label="Margin",
                        info="Margin for ranking loss. 1.0 typical. Larger margin = stricter separation.",
                        minimum=0.1,
                        maximum=3.0,
                        value=1.0,
                        step=0.1,
                        elem_id="margin",
                    )
                    num_neighbors_str = gr.Textbox(
                        label="Num Neighbors (comma-separated)",
                        info="Neighbor samples per GNN layer (deepest to shallowest). E.g. '15, 10, 5'.",
                        value="15, 10, 5",
                        elem_id="num_neighbors_str",
                    )
                    max_subgraph_nodes = gr.Number(
                        label="Max Subgraph Nodes",
                        info="Max nodes in sampled subgraph. Larger = more context but slower. 5000 typical.",
                        value=5000,
                        minimum=100,
                        precision=0,
                        elem_id="max_subgraph_nodes",
                    )

            # -----------------------------------------------------------------
            # Control Buttons
            # -----------------------------------------------------------------
            gr.Markdown("### Control")
            with gr.Row():
                start_btn = gr.Button("Start Training", variant="primary")
                stop_btn = gr.Button("Stop Training", variant="stop", interactive=False)
                resume_btn = gr.Button("Resume Training")

            # Checkpoint selection for resume
            with gr.Accordion("Resume Checkpoint Selection", open=False):
                with gr.Row():
                    checkpoint_dropdown = gr.Dropdown(
                        label="Resume Checkpoint",
                        info="Select a checkpoint to resume from. Click Refresh to update the list.",
                        choices=[],
                        value=None,
                        allow_custom_value=True,
                        elem_id="checkpoint_dropdown",
                    )
                    refresh_ckpt_btn = gr.Button("Refresh", size="sm", scale=0)

            status_display = gr.Markdown(
                value="**Status**: IDLE",
                label="Status",
                elem_id="status_display",
            )

        # =====================================================================
        # Right Column: Metrics & Resources
        # =====================================================================
        with gr.Column(scale=2):
            gr.Markdown("### Training Metrics")

            with gr.Row():
                loss_plot = gr.LinePlot(
                    value=_EMPTY_LOSS_DF,
                    x="epoch",
                    y="loss",
                    color="split",
                    title="Loss Curves",
                    x_title="Epoch",
                    y_title="Loss",
                    width=450,
                    height=280,
                    elem_id="loss_plot",
                )
                mrr_plot = gr.LinePlot(
                    value=_EMPTY_METRIC_DF,
                    x="epoch",
                    y="value",
                    color="metric",
                    title="Mean Reciprocal Rank",
                    x_title="Epoch",
                    y_title="MRR",
                    width=450,
                    height=280,
                    elem_id="mrr_plot",
                )

            with gr.Row():
                hits_plot = gr.LinePlot(
                    value=_EMPTY_METRIC_DF,
                    x="epoch",
                    y="value",
                    color="metric",
                    title="Hits@K",
                    x_title="Epoch",
                    y_title="Hits",
                    width=450,
                    height=280,
                    elem_id="hits_plot",
                )
                lr_plot = gr.LinePlot(
                    value=_EMPTY_METRIC_DF,
                    x="epoch",
                    y="value",
                    color="metric",
                    title="Learning Rate",
                    x_title="Epoch",
                    y_title="LR",
                    width=450,
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
            )

    # =========================================================================
    # Collect all parameter inputs in order matching _collect_config signature
    # =========================================================================
    all_params = [
        # Paths
        data_dir, output_dir, checkpoint_dir,
        # Tier 1
        num_epochs, learning_rate, batch_size, conv_type, device,
        seed,
        # Tier 2
        hidden_dim, num_layers, dropout, weight_decay, scheduler_type,
        warmup_steps, early_stopping_patience,
        diagnosis_weight, link_prediction_weight, contrastive_weight, ortholog_weight,
        # Tier 3
        gradient_accumulation_steps, max_grad_norm, num_heads,
        use_ortholog_gate, use_amp, amp_dtype,
        temperature, label_smoothing, margin, num_neighbors_str, max_subgraph_nodes,
    ]

    # =========================================================================
    # Event Handlers
    # =========================================================================
    # Shared output list: status + 4 plots + 3 buttons
    btn_outputs = [
        status_display,
        loss_plot, mrr_plot, hits_plot, lr_plot,
        start_btn, stop_btn, resume_btn,
    ]

    start_btn.click(
        fn=_on_start,
        inputs=all_params,
        outputs=btn_outputs,
    )
    stop_btn.click(
        fn=_on_stop,
        inputs=[],
        outputs=btn_outputs,
    )
    resume_btn.click(
        fn=_on_resume,
        inputs=all_params + [checkpoint_dropdown],
        outputs=btn_outputs,
    )
    refresh_ckpt_btn.click(
        fn=_refresh_checkpoints,
        inputs=[],
        outputs=[checkpoint_dropdown],
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

    # =========================================================================
    # Polling Timer (1.5 second interval)
    # =========================================================================
    timer = gr.Timer(value=1.5)
    timer.tick(
        fn=_poll_status,
        inputs=[],
        outputs=[
            status_display,
            loss_plot,
            mrr_plot,
            hits_plot,
            lr_plot,
            resource_display,
            start_btn,
            stop_btn,
            resume_btn,
        ],
    )


__all__ = ["create_training_tab"]
