"""
SHEPHERD-Advanced WebUI Components
====================================
Gradio UI component modules.

Module: src/webui/components/__init__.py

Components:
    - training_console: Training control and monitoring tab

Version: 1.0.0
"""

from src.webui.components.training_console import create_training_tab

__all__ = [
    "create_training_tab",
]
