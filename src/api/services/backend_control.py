"""
SHEPHERD-Advanced Backend Control Service
=========================================
Self-restart for the backend process, driven by the Runtime Settings tab.

Module: src/api/services/backend_control.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/api/services/backend_control.py

Purpose:
    The backend runs as a single ``python -m uvicorn src.api.main:app`` process
    that serves BOTH the REST API and the Gradio UI (mounted at /ui). Some
    Runtime Settings (notably the CUDA memory allocator) are read once at
    process startup, so applying a new value requires relaunching that process.
    This service provides an in-process restart by re-exec'ing the interpreter
    with the same arguments (``os.execve``), which keeps the same PID and parent
    (the launcher's ``subprocess.run`` keeps waiting on it) while giving CUDA a
    fresh start under the newly chosen allocator.

Safety:
    A restart is REFUSED while training is in progress — the UI also greys the
    button out, but this server-side guard is the real safety net.

Allocator re-resolution:
    The launcher records HOW ``PYTORCH_ALLOC_CONF`` was set via the
    ``SHEPHERD_ALLOC_SOURCE`` env marker. If it came from a preset, we
    re-resolve it from the freshly saved settings on restart so a newly chosen
    allocator actually takes effect; an explicit user override is preserved.

Dependencies:
    - src.api.services.training_manager: training status (the restart lock)
    - src.config.runtime_presets: allocator preset resolution (gradio-free)

Called by:
    - src/webui/components/runtime_settings.py (Restart Backend button + poll)

Version: 1.0.0
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Callable, Dict, List

from src.api.services.training_manager import training_manager
from src.config.runtime_presets import load_runtime_settings, resolve_allocator

logger = logging.getLogger(__name__)

# Env marker the launcher sets to record where PYTORCH_ALLOC_CONF came from:
#   "preset" -> launcher applied it from the saved allocator preset; safe to
#               re-resolve on restart so a newly chosen preset takes effect.
#   "env"    -> an explicit override was present before launch; preserve it.
ALLOC_SOURCE_ENV = "SHEPHERD_ALLOC_SOURCE"

# Statuses during which a restart must be blocked (a subprocess is alive).
_BUSY_STATUSES = ("running", "stopping")


def is_training_active(default_on_error: bool = False) -> bool:
    """True if a training run is in progress (restart must be blocked).

    ``default_on_error`` is what to return when the status can't be read:
    callers that gate a destructive restart pass ``True`` (fail closed — treat
    an unknown state as "busy"), while the UI polling path keeps the default
    ``False`` (fail open — a transient read error must not break the page).
    """
    try:
        return training_manager.get_status().get("status") in _BUSY_STATUSES
    except Exception:
        logger.debug("is_training_active: status check failed", exc_info=True)
        return default_on_error


def resolve_restart_env(env: Dict[str, str]) -> Dict[str, str]:
    """Return a copy of ``env`` with the allocator re-resolved for restart.

    The saved allocator preset is applied when the current allocator is
    "preset-derived":
      - the launcher applied it from a preset (``SHEPHERD_ALLOC_SOURCE ==
        "preset"``), or
      - the backend was launched directly (e.g. bare ``uvicorn``) with **no**
        marker and **no** explicit allocator env var — so applying a setting via
        the UI and restarting actually takes effect.

    An explicit override is preserved untouched:
      - ``SHEPHERD_ALLOC_SOURCE == "env"`` (launcher saw an override at launch), or
      - a ``PYTORCH_ALLOC_CONF`` / ``PYTORCH_CUDA_ALLOC_CONF`` present without a
        marker (set directly by whoever started the process).
    """
    new_env = dict(env)
    marker = new_env.get(ALLOC_SOURCE_ENV)
    has_env_alloc = (
        "PYTORCH_ALLOC_CONF" in new_env or "PYTORCH_CUDA_ALLOC_CONF" in new_env
    )
    preset_derived = marker == "preset" or (marker is None and not has_env_alloc)
    if preset_derived:
        _preset, conf = resolve_allocator(load_runtime_settings().get("allocator_preset"))
        new_env["PYTORCH_ALLOC_CONF"] = conf
        new_env[ALLOC_SOURCE_ENV] = "preset"
    return new_env


def build_restart_argv(executable: str, argv: List[str]) -> List[str]:
    """Reconstruct the command to re-exec the current backend process.

    The backend is launched as ``python -m uvicorn src.api.main:app ...`` so
    ``argv[0]`` is uvicorn's ``__main__.py``; rebuild it as an explicit
    ``-m uvicorn`` invocation. For any other entry point, re-run the interpreter
    against the same argv verbatim.
    """
    argv0 = argv[0] if argv else ""
    looks_like_uvicorn = (
        "uvicorn" in os.path.basename(argv0)
        or f"{os.sep}uvicorn{os.sep}" in argv0
    )
    if looks_like_uvicorn:
        return [executable, "-m", "uvicorn", *argv[1:]]
    return [executable, *argv]


def restart_backend(
    delay: float = 1.5,
    _scheduler: Callable[[Callable[[], None]], None] | None = None,
    _execv: Callable[[str, List[str], Dict[str, str]], None] = os.execve,
) -> Dict[str, object]:
    """Restart the backend (uvicorn) process after a short delay.

    Refuses if training is in progress. The delay lets the triggering HTTP
    response flush to the browser before the process image is replaced; the page
    then reconnects once the fresh backend is up.

    The ``_scheduler`` / ``_execv`` hooks exist for testing — production uses a
    daemon thread and ``os.execve``.
    """
    # Fail closed for this destructive action: an unreadable status blocks it.
    if is_training_active(default_on_error=True):
        return {"success": False, "error": "Training in progress — restart is locked."}

    executable = sys.executable
    argv = build_restart_argv(executable, sys.argv)

    def _worker() -> None:
        time.sleep(delay)
        # Re-check after the delay: training may have started (via the REST API
        # or another tab) during the window. Aborting here prevents orphaning a
        # live training subprocess. Fail closed if the status can't be read.
        if is_training_active(default_on_error=True):
            logger.warning(
                "Backend restart aborted: training became active during the "
                "restart delay."
            )
            return
        env = resolve_restart_env(dict(os.environ))
        logger.info("Restarting backend: %s", " ".join(argv))
        try:
            _execv(executable, argv, env)
        except Exception:  # noqa: BLE001 — log and leave the process running
            logger.exception("Backend restart failed (execve)")

    scheduler = _scheduler or (
        lambda fn: threading.Thread(
            target=fn, daemon=True, name="backend-restart"
        ).start()
    )
    scheduler(_worker)
    return {"success": True}


__all__ = [
    "ALLOC_SOURCE_ENV",
    "is_training_active",
    "resolve_restart_env",
    "build_restart_argv",
    "restart_backend",
]
