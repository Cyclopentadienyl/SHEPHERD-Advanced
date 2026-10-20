"""
SHEPHERD-Advanced Gradio Dashboard
====================================
Main Gradio application with tabbed interface.

Module: src/webui/app.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/webui/app.py

Purpose:
    Build and export the Gradio Blocks application for the SHEPHERD-Advanced
    dashboard, and mount it into the FastAPI app at /ui (`mount_dashboard`).

Tabs:
    1. Training Console — Parameter tuning, training control, live metrics
    2. Inference Testing — (placeholder, Phase 2)
    3. Model Management — (placeholder, Phase 3)

Dependencies:
    - gradio==6.30.0 (pinned in pyproject.toml). From Gradio 6 a mounted app
      takes its theme, CSS, head and JS from gr.mount_gradio_app(), not from
      gr.Blocks(); `mount_dashboard` is the one place that passes them.
    - src.webui.components.training_console: create_training_tab

Called by:
    - src.api.main (`mount_dashboard`, at /ui)

Version: 1.0.0
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gradio as gr

from src.webui.components.training_console import create_training_tab
from src.webui.components.diagnosis_panel import create_diagnosis_tab
from src.webui.components.runtime_settings import create_runtime_settings_tab
from src.utils.version_checker import DEGRADED, NOTICE, format_runtime_line, probe_runtime

if TYPE_CHECKING:
    from fastapi import FastAPI


def _runtime_footer() -> str:
    """Markdown for the always-visible runtime line.

    Colour carries the severity so it reads at a glance: red when a capability
    is gone (torch or PyG will not import — GNN scoring cannot run), amber for
    performance-only or optional gaps, plain grey when everything is present.

    Wrapped: this footer exists to report a broken environment, so it must not
    itself break in one. Its own failure is rendered rather than raised.
    """
    try:
        report = probe_runtime()
        line = format_runtime_line(report)
        status = report.get("status")
        issues = report.get("issues", [])
    except Exception as exc:  # noqa: BLE001 — a footer that raises reports nothing
        return (
            '<span style="color:#d32f2f"><b>Runtime status unavailable</b> — '
            f"could not probe the environment: {exc}</span>"
        )

    if status == DEGRADED:
        detail = " ".join(issues)
        return (
            f'<span style="color:#d32f2f"><b>⛔ Degraded runtime</b> — {line}<br>'
            f"{detail}</span>"
        )
    if status == NOTICE:
        detail = " ".join(issues)
        return (
            f'<span style="color:#e65100"><b>⚠️ {line}</b><br>'
            f'<span style="font-size:0.9em">{detail}</span></span>'
        )
    return f'<span style="color:#888">{line}</span>'


#: The dashboard's CSS: the IBM Plex font files (served by the API at /static,
#: so the page needs no network), the polled panels' loading flash, and the
#: Runtime Settings badges. Passed to the mount (`mount_dashboard`).
DASHBOARD_CSS = """
    /* IBM Plex Sans — local font files (offline-safe) */
    @font-face {
        font-family: 'IBM Plex Sans';
        src: url('/static/fonts/ibm-plex-sans-400.woff2') format('woff2');
        font-weight: 400;
        font-style: normal;
        font-display: swap;
    }
    @font-face {
        font-family: 'IBM Plex Sans';
        src: url('/static/fonts/ibm-plex-sans-500.woff2') format('woff2');
        font-weight: 500;
        font-style: normal;
        font-display: swap;
    }
    @font-face {
        font-family: 'IBM Plex Sans';
        src: url('/static/fonts/ibm-plex-sans-600.woff2') format('woff2');
        font-weight: 600;
        font-style: normal;
        font-display: swap;
    }
    @font-face {
        font-family: 'IBM Plex Sans';
        src: url('/static/fonts/ibm-plex-sans-700.woff2') format('woff2');
        font-weight: 700;
        font-style: normal;
        font-display: swap;
    }

    /* CSS fallback stack (not in theme font= to avoid Gradio trying to load them as files) */
    .gradio-container, .gradio-container * {
        font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
    }

    /* Suppress Gradio's opacity/loading flash on polled components */
    #resource_display,
    #resource_display *,
    #status_display,
    #status_display *,
    div:has(> #resource_display),
    div:has(> #status_display) {
        animation: none !important;
        transition: none !important;
        opacity: 1 !important;
    }
    #resource_display.pending,
    #resource_display.generating,
    #status_display.pending,
    #status_display.generating,
    div:has(> #resource_display).pending,
    div:has(> #resource_display).generating,
    div:has(> #status_display).pending,
    div:has(> #status_display).generating {
        opacity: 1 !important;
        pointer-events: auto !important;
    }
    #resource_display .eta-bar,
    #status_display .eta-bar,
    div:has(> #resource_display) .eta-bar,
    div:has(> #status_display) .eta-bar {
        display: none !important;
    }

    /* Runtime Settings — coloured impact/scope pill badges */
    .shep-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 2px 0 8px;
    }
    .shep-badge {
        display: inline-flex;
        align-items: center;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        line-height: 1.55;
        letter-spacing: 0.01em;
        white-space: nowrap;
        border: 1px solid transparent;
    }
    .shep-badge-mem     { background: rgba(37,99,235,0.12);  color: #1d4ed8; border-color: rgba(37,99,235,0.30); }
    .shep-badge-speed   { background: rgba(22,163,74,0.12);  color: #15803d; border-color: rgba(22,163,74,0.30); }
    .shep-badge-prec    { background: rgba(217,119,6,0.14);  color: #b45309; border-color: rgba(217,119,6,0.34); }
    .shep-badge-exp     { background: rgba(147,51,234,0.12); color: #7e22ce; border-color: rgba(147,51,234,0.30); }
    .shep-badge-restart { background: rgba(220,38,38,0.12);  color: #b91c1c; border-color: rgba(220,38,38,0.30); }
    .shep-badge-nextrun { background: rgba(100,116,139,0.14);color: #475569; border-color: rgba(100,116,139,0.32); }

    /* Restart-locked warning (training in progress) */
    .shep-restart-locked {
        color: #dc2626;
        font-weight: 600;
        font-size: 0.85rem;
    }
"""


def _dashboard_theme() -> gr.themes.Base:
    """The Soft theme with IBM Plex (the font files are declared in DASHBOARD_CSS)."""
    return gr.themes.Soft(
        font=["IBM Plex Sans"],
        font_mono=["IBM Plex Mono"],
    )


def create_gradio_app() -> gr.Blocks:
    """
    Create the SHEPHERD-Advanced Gradio dashboard.

    Returns:
        gr.Blocks application instance. Mount it with `mount_dashboard`, which
        passes the theme, CSS and switches (`mount_options`).
    """
    with gr.Blocks(
        title="SHEPHERD-Advanced Dashboard",
        # No usage analytics and no version check (owner, 2026-10-12). Left to
        # Gradio's default, building this app started a version check against
        # api.gradio.app and an analytics request through huggingface_hub, and
        # the API builds it at import, so every start of the service tried to
        # reach both. This is the one place the dashboard is built. It switches
        # off only Gradio's own requests, not every Hugging Face request.
        analytics_enabled=False,
        # Theme and CSS are not set here: Gradio 6 drops them from a mounted
        # Blocks, with only a warning. `mount_dashboard` passes them.
    ) as app:
        gr.Markdown(
            "# SHEPHERD-Advanced Dashboard\n"
            "Rare Disease Diagnosis Engine — Training & Inference Console"
        )

        with gr.Tabs():
            with gr.Tab("Training Console", id="train"):
                create_training_tab()

            with gr.Tab("Diagnosis", id="infer"):
                create_diagnosis_tab(app)

            with gr.Tab("Runtime Settings", id="runtime") as runtime_tab:
                create_runtime_settings_tab(runtime_tab)

            with gr.Tab("Model Management", id="models"):
                gr.Markdown(
                    "### Model Management\n"
                    "*Coming soon — will provide checkpoint listing and metrics comparison.*"
                )

        # Runtime stack, always visible. Two jobs: say which torch/CUDA/PyG
        # combination is actually running, and make a degraded environment
        # impossible to miss. A deployment can lose its GNN without any error
        # appearing here, so absence has to be shown, not merely not-shown.
        gr.Markdown(_runtime_footer(), elem_id="runtime_footer")

    return app


#: The largest file Gradio's upload route accepts, per file (M5-1). The
#: dashboard has no upload control: its files go the other way (exports), and
#: import sources come from the server's intake directory, never a browser (D7).
#: Gradio still registers the route, and CORS does not stop a cross-origin
#: multipart POST from reaching it (a "simple" request needs no preflight), so
#: without a limit any page the operator's browser opens could write files of
#: any size into GRADIO_TEMP_DIR. 1 MB is far above anything the dashboard
#: sends and bounds what such a request can write; a larger file is refused
#: with 413 while it streams.
MAX_UPLOAD_SIZE = "1mb"


def mount_options() -> dict[str, Any]:
    """Everything `gr.mount_gradio_app` is given besides the app, the Blocks and the path.

    The look: theme and CSS (there is no ``head`` or ``js``; if one is added, it
    belongs here too, for the same reason).

    The switches, each off whatever the environment says, because left to
    Gradio a ``GRADIO_*`` variable can turn it on:

    - ``run_history``: Gradio 6 records each event's inputs and outputs in the
      browser by default and offers to save them to a Hugging Face bucket.
      Diagnosis inputs are a patient's phenotypes; they are not kept in the
      browser nor offered to a cloud store. Off, the run-history page answers
      404 and its other routes are not registered.
    - ``enable_monitoring``: only ``False`` refuses ``/monitoring/summary``
      (left at None, it serves the queue's event analytics).
    - ``mcp_server``: would publish every event handler as an MCP tool.
    - ``ssr_mode``: server-side rendering starts a Node.js server process.
    - ``max_file_size``: `MAX_UPLOAD_SIZE` (below).

    Two routes Gradio registers with no switch at all are refused in front of
    the mount: ``src/api/middleware/gradio_routes.py``.
    """
    return {
        "theme": _dashboard_theme(),
        "css": DASHBOARD_CSS,
        "run_history": False,
        "enable_monitoring": False,
        "mcp_server": False,
        "ssr_mode": False,
        "max_file_size": MAX_UPLOAD_SIZE,
    }


def mount_dashboard(app: FastAPI, path: str = "/ui") -> gr.Blocks:
    """Build the dashboard and mount it on ``app`` at ``path``; return the Blocks."""
    blocks = create_gradio_app()
    gr.mount_gradio_app(app, blocks, path=path, **mount_options())
    return blocks


__all__ = ["DASHBOARD_CSS", "MAX_UPLOAD_SIZE", "create_gradio_app", "mount_dashboard",
           "mount_options"]
