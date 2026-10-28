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
  only (never a case identifier, D7) and their report stores' runs, each with
  whether its report is kept and registered as the registration reads them
  from the store and the ledger (`server_lists.stored_report_state`).
  Read-only and open. ``registered`` is read from the ledger beside the
  checkpoint the report records: ``registered`` when that ledger names it,
  ``cannot_be_checked`` otherwise, never ``not_registered``, since a
  registration filed beside another copy of the same weights is not looked
  for (the reason names the ledger that was). Each run with a ``report.json``
  also carries D5's state of its report's key (``contest``: uncontested,
  contested with the fields and runs that differ, or cannot_be_checked; never a
  problem's words), from the same run listing the Test page shows, so the two
  cannot disagree about a run; it is for the bytes the run's state read, so a
  report replaced between the two reads is cannot_be_checked. ``contest`` is
  null for a run directory with no ``report.json`` and for a run whose state
  was not read (its ``problem`` says why). The cost: one scan of each listed
  version's report store per request, each kept report read once more, on an
  open route (as the loaded-model view's records cost, BACKLOG 36).
- `GET /workspaces/{name}/checkpoints`: a listed workspace's checkpoints,
  optionally only those whose model record names a task or a learning stage.
  It opens every checkpoint of the workspace (SPEC_4 C3), which D8 offers only
  where the operator controls are, so it takes D8's one dependency
  (`src/api/operator_access.py`) and is refused, opening nothing, unless D8
  allows the request.
- `GET /workspaces/{name}/family`: the same checkpoints as families, each
  child under the exact parent file its record names (README §1), with the
  same filters, a filtered child's base kept as context. Gated as the list it
  is built from; a parent outside the workspace is looked for in the other
  listed workspaces' run of the recorded name only.

Plain `def` handlers: each reads files, so FastAPI runs them in its thread pool
instead of on the event loop.

Module: src/api/routes/lists.py
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter

from src.api.operator_access import OPERATOR_ONLY
from src.api.services import server_lists

router = APIRouter()


@router.get("/workspaces")
def get_workspaces() -> Dict[str, Any]:
    """The server's workspaces, by name and dataset fingerprint."""
    return server_lists.list_workspaces().as_dict("workspaces")


@router.get("/cohorts")
def get_cohort_versions() -> Dict[str, Any]:
    """The cohort versions in the server's registry, by label and digest, with
    each kept run's kept and registered state read from the disk, and D5's
    state of its report's key for those bytes (``contest``, without the
    problems' words; null for a run with no ``report.json``).
    Without a checkpoint file, ``registered`` is either ``registered`` (the
    ledger beside the checkpoint the report records names it) or
    ``cannot_be_checked``; a list never says ``not_registered``, because a
    registration beside another copy of the same weights is not looked for."""
    return server_lists.list_cohort_versions(
        report_state=server_lists.stored_report_state).as_dict("versions")


@router.get("/workspaces/{name}/checkpoints", dependencies=OPERATOR_ONLY)
def get_workspace_checkpoints(name: str, architecture: Optional[str] = None,
                              task: Optional[str] = None,
                              learning_stage: Optional[str] = None) -> Dict[str, Any]:
    """A listed workspace's checkpoints, by run, file name and SHA-256, each
    with its model record's summary; `task` and `learning_stage` keep the
    files whose record names them. An operator control (D8, SPEC_4 C3)."""
    return server_lists.list_checkpoints(
        name, architecture, task=task, learning_stage=learning_stage).as_dict("checkpoints")


@router.get("/workspaces/{name}/family", dependencies=OPERATOR_ONLY)
def get_workspace_family(name: str, task: Optional[str] = None,
                         learning_stage: Optional[str] = None) -> Dict[str, Any]:
    """A listed workspace's checkpoints as families: each child under the
    exact parent file its record names, the history and refused files after.
    An operator control (D8, SPEC_4 C3), as the list it is built from."""
    return server_lists.list_family(
        name, task=task, learning_stage=learning_stage).as_dict("rows")
