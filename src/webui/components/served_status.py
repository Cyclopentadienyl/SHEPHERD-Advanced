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
`node_names` gives a result the served graph's names of the terms it names,
only while the graph served is that result's own (by its digest).

Module: src/webui/components/served_status.py
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

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


#: An HPO id as a request spells it (the Diagnosis box's canonical form).
_HPO_ID = re.compile(r"HP:\d{7}")


def node_names(kg_sha256: Any, ids: Iterable[str]) -> Dict[str, str]:
    """**The served graph's names of `ids`**, keyed as given: an HPO id as a
    request spells it ("HP:0001250") or a node as a path names it
    ("hpo:HP:0001250", "mondo:MONDO:0012091"). Only while the graph served
    now is the one `kg_sha256` names (a result's own
    ``served_identity.graph.kg_sha256``: the same digest is the same bytes,
    so these are that result's graph's names); else ``{}``, and the page shows
    ids alone (another graph served now, nothing served, a result that names
    no digest). An id the graph has no node for, or a node with no name, is
    left out. Reads application state only, in this process, as `read_status`
    does; nothing is opened or parsed."""
    if not isinstance(kg_sha256, str) or len(kg_sha256) != 64:
        return {}
    try:
        from src.api.main import app_state
        from src.core.types import DataSource, NodeID

        pipeline = app_state.snapshot()["pipeline"]
        if pipeline is None or pipeline.served_identity().graph.kg_sha256 != kg_sha256:
            return {}
        graph = pipeline.kg
        names: Dict[str, str] = {}
        for key in dict.fromkeys(ids):
            if not isinstance(key, str):
                continue
            try:
                node_id = (NodeID(DataSource.HPO, key) if _HPO_ID.fullmatch(key)
                           else NodeID.from_string(key))
            except ValueError:
                continue
            node = graph.get_node(node_id)
            name = getattr(node, "name", None)
            if isinstance(name, str) and name.strip():
                names[key] = name.strip()
        return names
    except Exception:  # noqa: BLE001 - names are an aid: ids are shown without them
        logger.exception("The served graph's names could not be read")
        return {}


__all__ = ["STATUS_NOT_READ", "node_names", "read_status", "served_checkpoint_sha",
           "served_key", "serving_workspace"]
