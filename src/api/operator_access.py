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
/pipeline/stop``, ``POST /training/start``, ``POST /training/finetune``,
``POST /training/branch``, ``POST /training/stop``, ``GET
/training/checkpoints``, ``GET /workspaces/{name}/checkpoints`` and ``GET
/workspaces/{name}/family``. A route
M5-2 adds that imports, starts or cancels a test, registers or loads a model is
to take the same dependency. Read-only routes (status, metrics, the lists of workspaces
and cohort versions, the recorded results, diagnosis) stay open.

The refusal's body names the condition that failed (`OperatorAccess.refusal`)
and repeats no header value.

**What an open read route answers** (`answer_for`, BACKLOG 53, D7): the status
routes (``GET /pipeline/status``, ``GET /training/status``) and the server's
lists (``GET /workspaces``, ``GET /cohorts``: an entry's reason names the file
or link it checked) answer a request D8 allows as they are, every path an
operator acts on included (the GPU lock's, the run directory, the served
workspace and checkpoint); any other request, the same answer with every
absolute server path replaced by "a server path". The other open routes answer
no server path of their own (``GET /settings/deployment`` hides its paths
itself; the rest name none). The pages read the same values in process and
show paths their own way.

Module: src/api/operator_access.py
"""
from __future__ import annotations

import os
from pathlib import PurePath
from typing import Any, List, Optional, Tuple

from fastapi import Depends, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder

from src.utils.operator_access import check_operator_access
from src.utils.path_redaction import SERVER_PATH, redact_value


async def require_operator_access(request: Request) -> None:
    """403 unless D8 allows this request (`check_operator_access` on its own
    scope). **An ``async def`` that awaits nothing**: it reads the scope and
    the environment only, so it runs on the event loop and takes no worker
    thread. A route's handler is reached without waiting for one, so a stop
    of inference closes admission as it arrives even while every worker of
    the threadpool runs a diagnosis (`pipeline.stop_pipeline`; checkpoint A's
    verification of I4)."""
    access = check_operator_access(request.scope)
    if not access.allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=access.refusal)


#: The dependency list an operator route passes as ``dependencies=``.
OPERATOR_ONLY: List[Any] = [Depends(require_operator_access)]


#: This service's own routes, which a reason may name ("… or POST
#: /api/v1/pipeline/reload"): a route, never a server path, so an answer keeps
#: it as it is (`answer_for`). Inside a longer path (``/srv/clinic/api/v1/…``)
#: it is that path's text and goes with it (`path_redaction`).
_ROUTES = (("/api/v1/", "/api/v1/"),)


def _a_server_directory(text: Any) -> bool:
    """Whether a data location's text names a server directory to hide: an
    absolute path, a filesystem root excepted. A relative value (refused:
    ``workspaces``, ``.``) names none; its text is a word of the answer, the
    refusal's own quote of it included, and as a hidden directory it took
    every such word, an object's key too ("workspaces", "data", each ". ").
    A root holds every absolute path, which the pattern replaces anyway; as
    a hidden directory it took every ``/`` (``train/val``)."""
    if not isinstance(text, str) or not text.strip():
        return False
    path = PurePath(text)
    return path.is_absolute() and path != PurePath(path.anchor)


def _data_locations() -> List[Tuple[str, Optional[str]]]:
    """The service's data locations as `redact_value`'s known, hidden
    directories (name None): each as it is set (the environment's text or the
    saved file's; the evidence folders' environment text entry by entry, as
    its reader splits it) and as it was checked, so a reason that names one
    whose own name holds a space ("… outside the workspaces root /srv/Jane
    (MRN 1) ws, so …") is replaced whole, which the pattern alone cannot see.
    Only the texts that are server directories (`_a_server_directory`). Read
    now, through the one read point (`server_settings.configured`); none when
    it cannot be read."""
    from src.config import server_settings

    texts: List[Any] = []
    try:
        for variable in (server_settings.ENV_WORKSPACES_ROOT, server_settings.ENV_REGISTRY,
                         server_settings.ENV_INTAKE_DIR,
                         server_settings.ENV_TRAINING_EVIDENCE_DIRS):
            raw = server_settings.configured(variable).raw
            if isinstance(raw, str):
                many = variable == server_settings.ENV_TRAINING_EVIDENCE_DIRS
                texts += raw.split(os.pathsep) if many else [raw]
            else:
                texts += list(raw or ())
        for setting in server_settings.all_settings()[1:]:
            value = setting.value
            texts += [str(v) for v in (value if isinstance(value, (tuple, list)) else (value,))
                      if v is not None]
    except Exception:  # noqa: BLE001 - the pattern still replaces every path it sees
        pass
    return [(text, None) for text in texts if _a_server_directory(text)]


def answer_for(request: Request, answer: Any) -> Any:
    """`answer` as an open read route gives it to `request`: as it is when D8
    allows the request (an operator acts on these paths: a GPU lock's holder,
    a run directory), and otherwise as JSON-ready data with every absolute
    server path in every text replaced, an object's keys included
    (`path_redaction.redact_value`; this service's own routes kept,
    `_ROUTES`; its data locations known and replaced whole, any path under
    them with them, `_data_locations`). The answer's shape and every other
    value stay, so its response model still holds."""
    if check_operator_access(request.scope).allowed:
        return answer
    roots = sorted([*_ROUTES, *_data_locations()], key=lambda root: -len(root[0]))
    return redact_value(jsonable_encoder(answer), roots, other=SERVER_PATH)


__all__ = ["OPERATOR_ONLY", "answer_for", "require_operator_access"]
