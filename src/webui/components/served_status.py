"""
What the server serves now, as the dashboard reads it (M5-2, UI-A).
===================================================================
One reader: `read_status` returns what ``GET /api/v1/pipeline/status`` answers,
computed by the route's own function (`src.api.routes.pipeline.live_status`) in
this process, so the dashboard and an API client read one status. It reads
application state only: no file, no HTTP self-call, nothing that waits behind a
reload.

`serving_workspace` names the listed workspace the status's
``current_data_dir`` is (Datasets' "serving reads this workspace" marker, the
Load confirmation's "serving workspace"), by comparing resolved directories:
the listed name is found through the server's list (`resolve_workspace`), never
derived from the status's path.

Module: src/webui/components/served_status.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Said when the status cannot be read (no exception text is shown).
STATUS_NOT_READ = "the pipeline status could not be read"


def read_status() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``(status, None)`` as the status route answers it, JSON-ready, or
    ``(None, reason)``."""
    try:
        from fastapi.encoders import jsonable_encoder

        from src.api.routes.pipeline import live_status

        return jsonable_encoder(live_status()), None
    except Exception:  # noqa: BLE001 - a page shows "cannot check", never a traceback
        logger.exception("The pipeline status could not be read")
        return None, STATUS_NOT_READ


def _part(status: Optional[Mapping[str, Any]], *keys: str) -> Any:
    value: Any = status
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def served_checkpoint_sha(status: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The full model fingerprint the status says is served, or None."""
    sha = _part(status, "served_identity", "checkpoint", "sha256")
    return sha if isinstance(sha, str) and len(sha) == 64 else None


def serving_workspace(status: Optional[Mapping[str, Any]], workspaces: Sequence[Any], *,
                      environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The name of the listed workspace the status's ``current_data_dir`` is,
    or None (none served, a graph served without a workspace, or one the
    server does not list)."""
    from src.api.services.server_lists import ServerListRefused, resolve_workspace

    data_dir = _part(status, "current_data_dir")
    if not isinstance(data_dir, str) or not data_dir:
        return None
    try:
        target = Path(data_dir).resolve()
    except (OSError, RuntimeError):
        return None
    for item in workspaces:
        name = item.get("name") if isinstance(item, Mapping) else getattr(item, "name", None)
        try:
            if resolve_workspace(name, environ=environ) == target:
                return name
        except ServerListRefused:
            continue
    return None


__all__ = ["STATUS_NOT_READ", "read_status", "served_checkpoint_sha", "serving_workspace"]
