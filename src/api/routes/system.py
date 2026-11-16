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
    - GET /system/operator-access: whether D8 enables the operator controls
      for this request, or which condition fails (read-only; the UI's chip)
    - GET /settings/deployment: each data location in effect, its source,
      the saved and pending values and its problem (UI-06; read-only, paths
      hidden for a request D8 does not allow)

Dependencies:
    - fastapi: Router
    - src.api.services.training_manager: TrainingManager (static method)
    - src.utils.version_checker: runtime stack probe

Version: 1.0.0
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Request

from src.api.services import data_locations
from src.api.services.training_manager import TrainingManager
from src.utils import deployment_view
from src.utils.operator_access import check_operator_access
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


@router.get("/system/operator-access")
def get_operator_access(request: Request) -> Dict[str, Any]:
    """Whether the operator controls are enabled for this request (D8).

    Read-only, and open to everyone: it answers about the request that asks,
    with the same check every operator control makes (`check_operator_access`),
    so a page can show the state beside its controls. ``enabled``, the
    ``condition`` that decided it, and a ``reason`` an operator can act on,
    which repeats no header value. Showing it is not the control: each
    control checks its own request.
    """
    return check_operator_access(request.scope).as_status()


@router.get("/settings/deployment")
def get_deployment_settings(request: Request) -> Dict[str, Any]:
    """The data locations as this service reads them now (UI-06): for each,
    the value in effect and its ``source`` (``environment``: set when the
    service started, and it wins; ``saved``: saved in Settings › Data
    locations; null: not set), the saved value, whether a saved change is
    ``pending``, whether the environment overrides a saved value (``overridden``), and
    the ``problem`` that turns its feature off; the saved settings file, the
    deployment mode and the GPU lock; the pending keys this service's
    start-up settings hold back (``held_back``: they apply at a start without
    them). The same view as Settings and ``scripts/deployment_settings.py
    show`` (`src.utils.deployment_view`), read by this service
    (`data_locations.view`: why a pending change waits is the check Apply
    makes, this job runner's slot included).

    Read-only and open, like the operator-access status: for a request D8
    does not allow (`check_operator_access`), every path is null, a refused
    value's problem is said in fixed words that never quote the value, and
    every other text that may name a path is redacted (``paths_shown``
    false); what is set, from where and whether it is pending stay. Nothing here changes a
    setting: Save, Apply and Replace are operator controls of the Settings
    page."""
    access = check_operator_access(request.scope)
    answer = deployment_view.as_dict(data_locations.view(), paths_shown=access.allowed)
    answer["operator_access"] = access.as_status()
    return answer
