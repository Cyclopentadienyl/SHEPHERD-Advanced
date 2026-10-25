"""
Refuse to be framed: no other page may show the dashboard or the API in a frame.
================================================================================
D8 (`src/utils/operator_access.py`) decides who may use an operator control from
the request's own arrival: loopback, no forwarding header, a loopback ``Host``,
the page's own ``Origin`` and ``Sec-Fetch-Site``. **It cannot see framing.** A
dashboard loaded from ``http://localhost:<port>/ui/`` inside another site's
``<iframe>`` makes its own requests same-origin to itself, so D8 allows them, and
an invisible frame laid over a hostile page could turn the operator's click into
"Restart Backend" or a Load (clickjacking). So every response says it may not be
framed, by anyone, the dashboard's own pages included (none frames another):

- ``X-Frame-Options: DENY``, which every browser honours;
- ``Content-Security-Policy: frame-ancestors 'none'``, its standard form, added
  only when the response carries no policy of its own (Gradio sets none).

**Pure ASGI** like the other middleware here: it changes only the start of the
response, so Gradio's streamed queue (server-sent events) passes through as it
is.

Module: src/api/middleware/frame_guard.py
"""
from typing import Any, Awaitable, Callable, MutableMapping

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

#: What every HTTP response carries, as (name, value) in lower case bytes.
FRAME_HEADERS = (
    (b"x-frame-options", b"DENY"),
    (b"content-security-policy", b"frame-ancestors 'none'"),
)


class RefuseFraming:
    """Add `FRAME_HEADERS` to every HTTP response, keeping a header of the same
    name that the response already set."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: MutableMapping[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers") or [])
                present = {bytes(name).lower() for name, _ in headers}
                headers.extend((name, value) for name, value in FRAME_HEADERS
                               if name not in present)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


__all__ = ["FRAME_HEADERS", "RefuseFraming"]
