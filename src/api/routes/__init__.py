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

Version: 1.0.0
"""

from src.api.routes import diagnose, search, disease

__all__ = [
    "diagnose",
    "search",
    "disease",
]
