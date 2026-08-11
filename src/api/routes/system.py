"""
SHEPHERD-Advanced System API Routes
=====================================
REST endpoints for system resource monitoring.

Module: src/api/routes/system.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/api/routes/system.py

Purpose:
    Provide system monitoring endpoints:
    - GET /system/resources: GPU utilization, memory, RAM usage
    - GET /system/runtime:   which torch / PyG / backend stack this process runs on

Dependencies:
    - fastapi: Router
    - src.api.services.training_manager: TrainingManager (static method)
    - src.utils.version_checker: runtime stack probe

Version: 1.0.0
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter

from src.api.services.training_manager import TrainingManager
from src.utils.version_checker import probe_runtime

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/system/resources")
async def get_system_resources() -> Dict[str, Any]:
    """
    Get system resource utilization.

    Returns GPU utilization, GPU memory, RAM usage, and temperature.
    """
    return TrainingManager.get_system_resources()


@router.get("/system/runtime")
async def get_runtime_stack(force: bool = False) -> Dict[str, Any]:
    """
    Report the runtime stack this process is actually running on.

    Interpreter, platform, torch and its CUDA line, PyTorch Geometric, the PyG
    native extensions and the retrieval backends, plus a `status` of
    ok / notice / degraded and a list of human-readable `issues`.

    Read-only and side-effect free. The probe never raises — a broken torch is
    reported as data, since reporting that is the point.

    Pass `force=true` to re-probe after a deliberate reinstall; results are
    otherwise cached for the life of the process.
    """
    return probe_runtime(force=force)
