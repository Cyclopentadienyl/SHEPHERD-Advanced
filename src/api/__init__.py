"""
SHEPHERD-Advanced API Module
============================
REST API service for rare disease diagnosis.

Module: src/api/__init__.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/api/__init__.py

Purpose:
    Provide REST API endpoints for:
    - Disease diagnosis from patient phenotypes
    - HPO term search and lookup
    - Disease information retrieval
    - Health monitoring

Components:
    - main.py: FastAPI application and lifespan management
    - routes/: Endpoint implementations
        - diagnose.py: Diagnosis API
        - search.py: HPO search API
        - disease.py: Disease info API

Usage:
    # Run with uvicorn, on this machine only (SPEC_4 B1; another address is
    # the launcher's to guard, deployment guide §4.13)
    uvicorn src.api.main:app --host 127.0.0.1 --port 8000

    # Or use the CLI
    python -m src.api.main

**Importing the package builds nothing** (BACKLOG 35; M5-3 review EVAL-5):
`app`, `app_state` and `get_app_state` are `src.api.main`'s, imported when
first asked for. Building the application here made any import of a service
(`src.api.services...`, which the dashboard's pages import) build the app
too, so a program that imported the dashboard first got an application that
failed to mount it (a circular import, logged as "Gradio not available").

Version: 1.0.0
"""
from typing import Any

__all__ = [
    "app",
    "app_state",
    "get_app_state",
]


def __getattr__(name: str) -> Any:
    """`src.api.main`'s application objects, imported on first use."""
    if name in __all__:
        from src.api import main

        return getattr(main, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
