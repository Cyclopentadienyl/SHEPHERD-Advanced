"""
SHEPHERD-Advanced Gradio Dashboard
====================================
Main Gradio application with tabbed interface.

Module: src/webui/app.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/webui/app.py

Purpose:
    Build and export the Gradio Blocks application for the SHEPHERD-Advanced
    dashboard. Mounted into the FastAPI app at /ui via gr.mount_gradio_app().

Tabs:
    1. Training Console — Parameter tuning, training control, live metrics
    2. Inference Testing — (placeholder, Phase 2)
    3. Model Management — (placeholder, Phase 3)

Dependencies:
    - gradio>=5.20,<5.30
    - src.webui.components.training_console: create_training_tab

Called by:
    - src.api.main (Gradio mount)

Version: 1.0.0
"""
from __future__ import annotations

import gradio as gr

from src.webui.components.training_console import create_training_tab
from src.webui.components.diagnosis_panel import create_diagnosis_tab
from src.webui.components.runtime_settings import create_runtime_settings_tab


def create_gradio_app() -> gr.Blocks:
    """
    Create the SHEPHERD-Advanced Gradio dashboard.

    Returns:
        gr.Blocks application instance ready to be mounted or launched.
    """
    with gr.Blocks(
        title="SHEPHERD-Advanced Dashboard",
        theme=gr.themes.Soft(
            font=["IBM Plex Sans"],
            font_mono=["IBM Plex Mono"],
        ),
        css="""
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
        """,
    ) as app:
        gr.Markdown(
            "# SHEPHERD-Advanced Dashboard\n"
            "Rare Disease Diagnosis Engine — Training & Inference Console"
        )

        with gr.Tabs():
            with gr.Tab("Training Console", id="train"):
                create_training_tab()

            with gr.Tab("Diagnosis", id="infer"):
                create_diagnosis_tab()

            with gr.Tab("Runtime Settings", id="runtime"):
                create_runtime_settings_tab()

            with gr.Tab("Model Management", id="models"):
                gr.Markdown(
                    "### Model Management\n"
                    "*Coming soon — will provide checkpoint listing and metrics comparison.*"
                )

    return app


__all__ = ["create_gradio_app"]
