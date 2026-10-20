"""
D8 in the dashboard: the one helper every operator handler calls first.
=======================================================================
`operator_access` applies D8's one check (`src/utils/operator_access.py`) to the
request that triggered a Gradio event: ``request.request``, the
``fastapi.Request`` of the HTTP call that started it (the ``queue/join`` POST,
or ``call/<api>`` / ``run/<api>`` for a direct call, under the queue and with
``queue=False`` alike). A direct call through Gradio's HTTP API therefore meets
the same check as a click.

**Where the check is: in each state-changing handler, before it does anything**
(D8). A greyed-out button is not a control. The handler then calls the shared
service in process, never the API through a loopback self-call, which the API
would judge as local whoever clicked.

A handler receives the request only through a positional parameter annotated
``gr.Request``; one written as ``*args`` takes it as its first parameter.
Without a request (a call outside Gradio, or one whose request was dropped when
it was pickled), the answer is a refusal: D8 fails closed.

Module: src/webui/operator_access.py
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from src.utils.operator_access import OperatorAccess, check_operator_access

if TYPE_CHECKING:
    import gradio as gr


def operator_access(request: Optional["gr.Request"]) -> OperatorAccess:
    """D8's decision for the request that triggered this event.

    Reads ``request.request.scope``; a missing request, or one without a
    scope, is passed on as no scope, which the check refuses.
    """
    inner: Any = getattr(request, "request", None) if request is not None else None
    return check_operator_access(getattr(inner, "scope", None))


__all__ = ["operator_access"]
