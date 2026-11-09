"""
D8 on the API: the one FastAPI dependency for operator routes.
==============================================================
`require_operator_access` applies D8's one check (`src/utils/operator_access.py`)
to the request the route is handling, its own ASGI scope, and refuses with 403
before the route runs: before its body is read into anything it acts on, and
before anything is written, started or listed. A route opts in with
``dependencies=OPERATOR_ONLY``; nothing else gates an API route.

Gated: every state-changing route and the checkpoint lists, which open
checkpoint files (SPEC_4 C3): ``POST /pipeline/reload``, ``POST
/pipeline/config``, ``POST /training/start``, ``POST /training/finetune``,
``POST /training/stop``, ``GET
/training/checkpoints``, ``GET /workspaces/{name}/checkpoints`` and ``GET
/workspaces/{name}/family``. A route
M5-2 adds that imports, starts or cancels a test, registers or loads a model is
to take the same dependency. Read-only routes (status, metrics, the lists of workspaces
and cohort versions, the recorded results, diagnosis) stay open.

The refusal's body names the condition that failed (`OperatorAccess.refusal`)
and repeats no header value.

Module: src/api/operator_access.py
"""
from __future__ import annotations

from typing import Any, List

from fastapi import Depends, HTTPException, Request, status

from src.utils.operator_access import check_operator_access


def require_operator_access(request: Request) -> None:
    """403 unless D8 allows this request (`check_operator_access` on its own
    scope). A plain `def`: it reads the scope and the environment only."""
    access = check_operator_access(request.scope)
    if not access.allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=access.refusal)


#: The dependency list an operator route passes as ``dependencies=``.
OPERATOR_ONLY: List[Any] = [Depends(require_operator_access)]


__all__ = ["OPERATOR_ONLY", "require_operator_access"]
