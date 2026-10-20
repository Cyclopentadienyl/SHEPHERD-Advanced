"""
The server's lists, over the API (M5-1).
========================================
Thin routes over `src.api.services.server_lists`, the functions the WebUI's
handlers call in-process too (from M5-2), so an API client and the dashboard
see the same lists. Each answer is the list, or why the server offers none
(`problem`): a missing setting is a state of the deployment, not an error of the
request.

- `GET /workspaces`: the workspaces under `SHEPHERD_WORKSPACES_ROOT`, each with
  its dataset identity. Reads small JSON files only; read-only and open.
- `GET /cohorts`: the cohort versions in `SHEPHERD_REGISTRY`, with aggregates
  only (never a case identifier, D7) and their report stores' runs. Read-only
  and open.
- A workspace's checkpoints: `get_workspace_checkpoints` below. **Not mounted
  at this revision.** It opens every checkpoint of the workspace (SPEC_4 C3),
  which D8 offers only where the operator controls are, and D8's check is
  M5-1's next package (WP2). Until WP2 mounts it with that check as
  `GET /workspaces/{name}/checkpoints`, no route offers it; an in-process
  caller (the WebUI's, from M5-2) calls `server_lists.list_checkpoints`.

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


# HOOK(WP2): not mounted until D8's check gates it (C3: this opens every checkpoint
# of the workspace). WP2 mounts it with its dependency, for example
#   router.add_api_route("/workspaces/{name}/checkpoints", get_workspace_checkpoints,
#                        methods=["GET"], dependencies=[Depends(<D8 check>)])
# and turns tests/unit/test_server_lists.py's "not mounted" test into a gate test.
def get_workspace_checkpoints(name: str, architecture: Optional[str] = None) -> Dict[str, Any]:
    """A listed workspace's checkpoints, by run, file name and SHA-256."""
    return server_lists.list_checkpoints(name, architecture).as_dict("checkpoints")
