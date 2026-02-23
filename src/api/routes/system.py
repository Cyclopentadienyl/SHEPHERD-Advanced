"""
SHEPHERD-Advanced System API Routes
=====================================
REST endpoints for system resource monitoring.

Module: src/api/routes/system.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/api/routes/system.py

Purpose:
    Provide system monitoring endpoint:
    - GET /system/resources: GPU utilization, memory, RAM usage

Dependencies:
    - fastapi: Router
    - src.api.services.training_manager: TrainingManager (static method)

Version: 1.0.0
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter

from src.api.services.training_manager import TrainingManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/system/resources")
async def get_system_resources() -> Dict[str, Any]:
    """
    Get system resource utilization.

    Returns GPU utilization, GPU memory, RAM usage, and temperature.
    """
    return TrainingManager.get_system_resources()
