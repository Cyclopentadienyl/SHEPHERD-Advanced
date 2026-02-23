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


def create_gradio_app() -> gr.Blocks:
    """
    Create the SHEPHERD-Advanced Gradio dashboard.

    Returns:
        gr.Blocks application instance ready to be mounted or launched.
    """
    with gr.Blocks(
        title="SHEPHERD-Advanced Dashboard",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# SHEPHERD-Advanced Dashboard\n"
            "Rare Disease Diagnosis Engine — Training & Inference Console"
        )

        with gr.Tabs():
            with gr.Tab("Training Console", id="train"):
                create_training_tab()

            with gr.Tab("Inference Testing", id="infer"):
                gr.Markdown(
                    "### Inference Testing\n"
                    "*Coming soon — will provide HPO term input and diagnosis result display.*"
                )

            with gr.Tab("Model Management", id="models"):
                gr.Markdown(
                    "### Model Management\n"
                    "*Coming soon — will provide checkpoint listing and metrics comparison.*"
                )

    return app


__all__ = ["create_gradio_app"]
