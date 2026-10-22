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
    -- the launcher's ``subprocess.run`` waiting on it, or, where the launcher
    exec'd into the server (``--no-browser`` on POSIX), the service manager --
    while giving CUDA a fresh start under the newly chosen allocator.

Safety:
    A restart is REFUSED while a GPU job (training or a test run) is in
    progress — the UI also greys the button out, but this server-side guard is
    the real safety net. After its delay the restart closes the job runner's
    GPU slot only if it is still idle and open, in one step (`close_if_idle`),
    and keeps it closed through the exec, so no job can start in between; a
    backend that began to stop meanwhile is not restarted.

Allocator re-resolution:
    The launcher records HOW ``PYTORCH_ALLOC_CONF`` was set via the
    ``SHEPHERD_ALLOC_SOURCE`` env marker. If it came from a preset, we
    re-resolve it from the freshly saved settings on restart so a newly chosen
    allocator actually takes effect; an explicit user override is preserved.

Dependencies:
    - src.api.services.training_manager: the job runner's GPU slot (the restart lock)
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
from src.config.runtime_presets import (
    ALLOC_SOURCE_ENV,
    allocator_env,
    load_runtime_settings,
)

logger = logging.getLogger(__name__)

# ALLOC_SOURCE_ENV, the marker recording where PYTORCH_ALLOC_CONF came from, is
# defined beside the rule that reads it (`src.config.runtime_presets`) and
# re-exported here for the callers that already import it from this module.

def is_training_active(default_on_error: bool = False) -> bool:
    """True while any GPU job, training or a test run, is running or stopping
    (restart must be blocked: a re-exec would orphan its subprocess). The name
    predates test runs (M5); the job runner's one GPU slot is what it reads.

    ``default_on_error`` is what to return when the status can't be read:
    callers that gate a destructive restart pass ``True`` (fail closed — treat
    an unknown state as "busy"), while the UI polling path keeps the default
    ``False`` (fail open — a transient read error must not break the page).
    """
    try:
        return training_manager.active_job() is not None
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
    # The rule itself is `allocator_env`, shared with the service entry point, so
    # a restart and a start cannot come to disagree.
    return allocator_env(env, load_runtime_settings())


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

    Refuses while a GPU job (training or a test run) is in progress. The delay lets the triggering HTTP
    response flush to the browser before the process image is replaced; the page
    then reconnects once the fresh backend is up.

    The ``_scheduler`` / ``_execv`` hooks exist for testing — production uses a
    daemon thread and ``os.execve``.
    """
    # Fail closed for this destructive action: an unreadable status blocks it.
    if is_training_active(default_on_error=True):
        return {"success": False,
                "error": "A GPU job (training or a test run) is in progress — restart is locked."}

    executable = sys.executable
    argv = build_restart_argv(executable, sys.argv)

    def _worker() -> None:
        time.sleep(delay)
        # **The GPU slot is closed before the image is replaced** (root-cause
        # review, P23-1): checked idle and open, and closed, in one step under
        # the job runner's lock, so a job started during the delay (through
        # the REST API or another tab) is seen here and the restart aborted,
        # and one asked for after it is refused at the slot; none is orphaned
        # by the exec. A slot the backend's stop closed during the delay
        # aborts the restart too, so the exec never overrides a shutdown
        # (T2-V2). The slot stays closed through execve, which does not
        # return; only a failed execve opens it again, and only the close
        # this restart made, never a stop's that came meanwhile.
        if not training_manager.close_if_idle():
            logger.warning(
                "Backend restart aborted: a GPU job became active, or the backend "
                "began to stop, during the restart delay."
            )
            return
        try:
            env = resolve_restart_env(dict(os.environ))
            logger.info("Restarting backend: %s", " ".join(argv))
            _execv(executable, argv, env)
        except Exception:  # noqa: BLE001 — log and leave the process running
            logger.exception("Backend restart failed (execve)")
            training_manager.open(closed_by=training_manager.CLOSED_BY_RESTART)

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
