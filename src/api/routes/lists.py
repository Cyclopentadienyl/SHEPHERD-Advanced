"""
The server's lists, over the API (M5-1).
========================================
Thin routes over `src.api.services.server_lists`, the functions the WebUI's
handlers call in-process too, so an API client and the dashboard see the same
lists. Each answer is the list, or why the server offers none (`problem`): a
missing setting is a state of the deployment, not an error of the request.

- `GET /workspaces`: the workspaces under `SHEPHERD_WORKSPACES_ROOT`, each with
  its dataset identity. Reads small JSON files only; read-only and open.
- `GET /cohorts`: the cohort versions in `SHEPHERD_REGISTRY`, with aggregates
  only (never a case identifier, D7) and their report stores' runs. Read-only
  and open.
- `GET /workspaces/{name}/checkpoints`: one listed workspace's checkpoints, each
  read once. **Opens every checkpoint (SPEC_4 C3)**, so it is offered only
  where D8's operator controls are.

Plain `def` handlers: each reads files, so FastAPI runs them in its thread pool
instead of on the event loop.

Module: src/api/routes/lists.py
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter

from src.api.services import server_lists

router = APIRouter()


@router.get("/workspaces")
def get_workspaces() -> Dict[str, Any]:
    """The server's workspaces, by name and dataset fingerprint."""
    return server_lists.list_workspaces().as_dict("workspaces")


@router.get("/cohorts")
def get_cohort_versions() -> Dict[str, Any]:
    """The cohort versions in the server's registry, by label and digest."""
    # TODO(WP4): pass the registration's one "kept?"/"registered?" helper as
    # report_state; until then each run says its state was not read.
    return server_lists.list_cohort_versions().as_dict("versions")


# TODO(WP2): gate with D8 (C3: this opens every checkpoint of the workspace).
@router.get("/workspaces/{name}/checkpoints")
def get_workspace_checkpoints(name: str, architecture: Optional[str] = None) -> Dict[str, Any]:
    """A listed workspace's checkpoints, by run, file name and SHA-256."""
    return server_lists.list_checkpoints(name, architecture).as_dict("checkpoints")
