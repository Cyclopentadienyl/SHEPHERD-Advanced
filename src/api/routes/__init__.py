"""
SHEPHERD-Advanced API Routes
============================
Route modules for the REST API.

Module: src/api/routes/__init__.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/api/routes/__init__.py

Routes:
    - diagnose: Disease diagnosis endpoints
    - search: HPO term search endpoints
    - disease: Disease information endpoints
    - training: Training control and monitoring endpoints
    - system: System resource monitoring endpoints

Version: 1.0.0
"""

from src.api.routes import diagnose, search, disease, training, system

__all__ = [
    "diagnose",
    "search",
    "disease",
    "training",
    "system",
]
