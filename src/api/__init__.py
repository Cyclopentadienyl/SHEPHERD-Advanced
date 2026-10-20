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
    # the launcher's to guard, deployment guide §4.12)
    uvicorn src.api.main:app --host 127.0.0.1 --port 8000

    # Or use the CLI
    python -m src.api.main

Version: 1.0.0
"""

from src.api.main import app, app_state, get_app_state

__all__ = [
    "app",
    "app_state",
    "get_app_state",
]
