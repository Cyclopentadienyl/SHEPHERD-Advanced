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
the listed name is found through the server's list (`resolve_workspaces`, one
listing of the root for every name), never derived from the status's path.
`served_key` is what is served as one comparable value (the whole served
identity and the data directory), so a change of any part is seen.

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
                      environ: Optional[Mapping[str, str]] = None,
                      directories: Optional[Mapping[str, Path]] = None) -> Optional[str]:
    """The name of the listed workspace the status's ``current_data_dir`` is,
    or None (none served, a graph served without a workspace, or one the
    server does not list).

    The names resolve through the server's list (`resolve_workspaces`: one
    listing of the root for every name, so many workspaces cost one listing);
    `directories` is that answer when the caller already has it."""
    from src.api.services.server_lists import ServerListRefused, resolve_workspaces

    data_dir = _part(status, "current_data_dir")
    if not isinstance(data_dir, str) or not data_dir:
        return None
    try:
        target = Path(data_dir).resolve()
    except (OSError, RuntimeError):
        return None
    if directories is None:
        try:
            directories = resolve_workspaces(environ=environ)
        except ServerListRefused:
            return None
    for item in workspaces:
        name = item.get("name") if isinstance(item, Mapping) else getattr(item, "name", None)
        if name in directories and directories[name] == target:
            return name
    return None


def served_key(status: Optional[Mapping[str, Any]]) -> Optional[str]:
    """What is served, as one comparable value: the whole served identity
    (checkpoint, graph, shortest-path table, dataset record, scoring mode and
    settings) and the served data directory; None when nothing can be read.
    Two reads with the same key serve the same thing."""
    import json

    if not isinstance(status, Mapping):
        return None
    return json.dumps({"identity": status.get("served_identity"),
                       "data_dir": status.get("current_data_dir")},
                      sort_keys=True, default=str)


__all__ = ["STATUS_NOT_READ", "read_status", "served_checkpoint_sha", "served_key",
           "serving_workspace"]
