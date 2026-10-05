"""
Record the address each request arrived on, for the in-process WebUI.
=====================================================================
`src/utils/server_address.py` holds the address and says why the Diagnosis tab
needs it. This is the only writer: every HTTP and WebSocket request passes through
here on its way into the app, Gradio's mount at `/ui` included, so for a server
listening on TCP the address is recorded before any WebUI callback can run. On a
Unix-domain socket there is no TCP address to record.

**Pure ASGI, not ``@app.middleware("http")``.** That decorator wraps the app in
Starlette's ``BaseHTTPMiddleware``, which sees HTTP only and stands between the
app and every streamed response. Gradio streams its queue over server-sent
events. Reading one field from the scope needs neither the request object nor
the response, so this touches nothing but the scope.

Module: src/api/middleware/server_address.py
"""
from typing import Any, Awaitable, Callable, MutableMapping

from src.utils.server_address import record_server_address

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class RecordServerAddress:
    """Pass every request through unchanged, noting the socket it arrived on."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") in ("http", "websocket"):
            record_server_address(scope.get("server"))
        await self.app(scope, receive, send)


__all__ = ["RecordServerAddress"]
