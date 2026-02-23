"""
SHEPHERD-Advanced WebUI Module
================================
Gradio-based web dashboard for training, inference, and model management.

Module: src/webui/__init__.py

Version: 1.0.0
"""

from src.webui.app import create_gradio_app

__all__ = [
    "create_gradio_app",
]
