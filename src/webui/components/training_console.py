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
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

from src.api.services.training_manager import training_manager

logger = logging.getLogger(__name__)


def _collect_config(
    # Tier 1 — Basic
    num_epochs: int,
    learning_rate: float,
    batch_size: str,
    conv_type: str,
    device: str,
    resume_from: str,
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

    config: Dict[str, Any] = {
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

    # Only include resume_from if non-empty
    if resume_from and resume_from.strip():
        config["resume_from"] = resume_from.strip()

    return config


def _on_start(*args) -> str:
    """Handle Start Training button click."""
    config = _collect_config(*args)
    result = training_manager.start_training(config)
    if result.get("success"):
        return f"Training started (PID: {result.get('pid')})"
    return f"Failed: {result.get('error', 'Unknown error')}"


def _on_stop() -> str:
    """Handle Stop Training button click."""
    result = training_manager.stop_training()
    if result.get("success"):
        return "Training stopped"
    return f"Failed: {result.get('error', 'Unknown error')}"


def _on_resume(*args) -> str:
    """Handle Resume Training button click (same as start with resume path)."""
    config = _collect_config(*args)
    # For resume, try to find the last checkpoint if no path specified
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
            return "No checkpoints found to resume from"

    result = training_manager.start_training(config)
    if result.get("success"):
        return f"Training resumed (PID: {result.get('pid')})"
    return f"Failed: {result.get('error', 'Unknown error')}"


def _poll_status() -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Poll training status and metrics. Called by gr.Timer every 2 seconds.

    Returns:
        (status_text, loss_df, mrr_df, hits_df, lr_df, resource_text)
    """
    status_info = training_manager.get_status()
    metrics_history = training_manager.get_metrics_history()
    resources = training_manager.get_system_resources()

    # Status text
    status_text = _format_status(status_info)

    # Build DataFrames for plots
    train_data = metrics_history.get("train", [])
    val_data = metrics_history.get("validation", [])

    loss_df = _build_loss_df(train_data, val_data)
    mrr_df = _build_metric_df(val_data, "val_mrr", "MRR")
    hits_df = _build_hits_df(val_data)
    lr_df = _build_metric_df(train_data, "lr_group_0", "Learning Rate")

    # Resource text
    resource_text = _format_resources(resources)

    return status_text, loss_df, mrr_df, hits_df, lr_df, resource_text


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


def _build_loss_df(
    train_data: List[Dict[str, Any]], val_data: List[Dict[str, Any]]
) -> Optional[pd.DataFrame]:
    """Build a DataFrame for the loss plot (train + val)."""
    rows = []

    for entry in train_data:
        epoch = entry.get("epoch")
        # Train loss can be 'total' or 'train_loss'
        loss = entry.get("total") or entry.get("train_loss")
        if epoch is not None and loss is not None and _is_finite(loss):
            rows.append({"epoch": epoch, "loss": loss, "split": "train"})

    for entry in val_data:
        epoch = entry.get("epoch")
        loss = entry.get("val_loss")
        if epoch is not None and loss is not None and _is_finite(loss):
            rows.append({"epoch": epoch, "loss": loss, "split": "val"})

    if not rows:
        return None
    return pd.DataFrame(rows)


def _build_metric_df(
    data: List[Dict[str, Any]], key: str, label: str
) -> Optional[pd.DataFrame]:
    """Build a single-series DataFrame for a metric."""
    rows = []
    for entry in data:
        epoch = entry.get("epoch")
        value = entry.get(key)
        if epoch is not None and value is not None and _is_finite(value):
            rows.append({"epoch": epoch, "value": value, "metric": label})

    if not rows:
        return None
    return pd.DataFrame(rows)


def _build_hits_df(val_data: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """Build a DataFrame for Hits@1 and Hits@10."""
    rows = []
    for entry in val_data:
        epoch = entry.get("epoch")
        h1 = entry.get("val_hits@1")
        h10 = entry.get("val_hits@10")
        if epoch is not None:
            if h1 is not None and _is_finite(h1):
                rows.append({"epoch": epoch, "value": h1, "metric": "Hits@1"})
            if h10 is not None and _is_finite(h10):
                rows.append({"epoch": epoch, "value": h10, "metric": "Hits@10"})

    if not rows:
        return None
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
            gr.Markdown("### Training Parameters")

            # -----------------------------------------------------------------
            # Tier 1 — Basic Parameters (always visible)
            # -----------------------------------------------------------------
            gr.Markdown("#### Basic")
            with gr.Group():
                num_epochs = gr.Number(
                    label="Epochs",
                    value=100,
                    minimum=1,
                    maximum=10000,
                    precision=0,
                    elem_id="num_epochs",
                )
                learning_rate = gr.Slider(
                    label="Learning Rate",
                    minimum=1e-6,
                    maximum=1e-1,
                    value=1e-4,
                    step=1e-6,
                    elem_id="learning_rate",
                )
                batch_size = gr.Dropdown(
                    label="Batch Size",
                    choices=["8", "16", "32", "64"],
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
                resume_from = gr.Textbox(
                    label="Resume From",
                    placeholder="checkpoint path (optional)",
                    value="",
                    elem_id="resume_from",
                )
                seed = gr.Number(
                    label="Seed",
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
                        minimum=0,
                        precision=0,
                        elem_id="warmup_steps",
                    )
                    early_stopping_patience = gr.Number(
                        label="Early Stopping Patience",
                        value=10,
                        minimum=1,
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
                        minimum=1,
                        precision=0,
                        elem_id="gradient_accumulation_steps",
                    )
                    max_grad_norm = gr.Number(
                        label="Max Grad Norm",
                        value=1.0,
                        minimum=0.01,
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
                    use_amp = gr.Checkbox(
                        label="Automatic Mixed Precision",
                        value=True,
                        elem_id="use_amp",
                    )
                    amp_dtype = gr.Radio(
                        label="AMP Dtype",
                        choices=["float16", "bfloat16"],
                        value="float16",
                        elem_id="amp_dtype",
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
                stop_btn = gr.Button("Stop Training", variant="stop")
                resume_btn = gr.Button("Resume Training")

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
                    value=None,
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
                    value=None,
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
                    value=None,
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
                    value=None,
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

            gr.Markdown("### System Resources")
            resource_display = gr.HTML(
                value='<div style="font-size:13px;color:#6b7280">Waiting for data...</div>',
                elem_id="resource_display",
            )

    # =========================================================================
    # Collect all parameter inputs in order matching _collect_config signature
    # =========================================================================
    all_params = [
        # Tier 1
        num_epochs, learning_rate, batch_size, conv_type, device,
        resume_from, seed,
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
    start_btn.click(
        fn=_on_start,
        inputs=all_params,
        outputs=status_display,
    )
    stop_btn.click(
        fn=_on_stop,
        inputs=[],
        outputs=status_display,
    )
    resume_btn.click(
        fn=_on_resume,
        inputs=all_params,
        outputs=status_display,
    )

    # =========================================================================
    # Polling Timer (2 second interval)
    # =========================================================================
    timer = gr.Timer(value=2)
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
        ],
    )


__all__ = ["create_training_tab"]
