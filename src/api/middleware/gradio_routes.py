"""
Refuse two Gradio routes the dashboard does not use, before Gradio sees them.
============================================================================
Gradio 6.30 registers these on every mounted dashboard, and upstream offers no
switch to turn either off (no ``gr.Blocks``, ``mount_gradio_app`` or
``GRADIO_*`` option; ``allowed_paths`` / ``blocked_paths`` apply only to local
files):

* ``GET|HEAD <mount>/gradio_api/file=<target>`` and its older alias
  ``<mount>/gradio_api/file/<target>``. A local ``<target>`` is how ``gr.File``
  and ``gr.DownloadButton`` hand an export (CSV, Markdown) to the browser, and
  keeps working. An ``http(s)://`` ``<target>`` is fetched **by the server**
  since Gradio 6.20 (issue #13593, PR #13596 replaced the old redirect with a
  server-side fetch) and streamed back on this origin. That fetch ignores
  ``HTTPS_PROXY``, and for a ``.png``, ``.txt`` or ``.json`` URL whose server
  answers ``text/html`` it is served inline, so a remote page would run as this
  origin's own script (not reported upstream as of 2026-10-20).
* ``POST <mount>/gradio_api/process_recording`` (Gradio 5.31): reads a
  multipart video into temporary files, copies it into ``GRADIO_TEMP_DIR`` and
  may run ffmpeg. Nothing in the dashboard records the screen.

This middleware answers both with 403, a one-line reason and
``connection: close``, and never calls ``receive()``: the request body is not
read and nothing is written. Every other request passes through untouched,
local ``file=`` downloads (GET, HEAD and Range) included.

Matching uses the decoded path Starlette routes on (``scope["path"]`` less
``root_path``, as ``starlette._utils.get_route_path`` computes it), so a
percent-encoded or mixed-case scheme is caught too. A ``file=`` target is
refused when it matches Gradio's own ``route_utils.starts_with_protocol``
pattern (copied here, so a Gradio change cannot narrow the guard). Gradio
already refuses every such target except ``http(s)://``, which it fetches, so
the guard changes behaviour only for the URL form, and keeps covering it if a
later Gradio fetches more schemes.

**Pure ASGI, like** ``RecordServerAddress``: ``BaseHTTPMiddleware`` would stand
between Gradio and its server-sent-event stream. Registered inside the request
logging in ``src/api/main.py``, so each refusal is logged like any response.

Module: src/api/middleware/gradio_routes.py
"""
import re
from typing import Any, Awaitable, Callable, MutableMapping, Optional

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

#: ``gradio.route_utils.starts_with_protocol`` as of 6.30.0: a URL scheme, or a
#: ``//`` / backslash form Windows reads as a network path.
_PROTOCOL_OR_UNC = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+\-.]*://|//|\\\\|\\/|/\\)")

#: The reason sent with each refusal, by what was refused.
REFUSALS = {
    "file-url": "Refused: this server does not fetch external URLs through file=.",
    "process_recording": "Refused: screen-recording processing is disabled on this server.",
}


def _route_path(scope: Scope) -> str:
    """The path Starlette routes on: ``path`` less a leading ``root_path``."""
    path: str = scope.get("path", "")
    root_path: str = scope.get("root_path", "")
    if root_path and path.startswith(root_path):
        if path == root_path:
            return ""
        if path[len(root_path)] == "/":
            return path[len(root_path):]
    return path


class RefuseUnusedGradioRoutes:
    """403 for ``file=<URL>`` and ``process_recording`` under one Gradio mount."""

    def __init__(self, app: ASGIApp, mount_path: str = "/ui") -> None:
        self.app = app
        self.api_prefix = mount_path.rstrip("/") + "/gradio_api/"

    def refusal(self, scope: Scope) -> Optional[str]:
        """Which refusal applies to this HTTP scope (a key of ``REFUSALS``), or None."""
        path = _route_path(scope)
        if not path.startswith(self.api_prefix):
            return None
        rest = path[len(self.api_prefix):]
        if rest.rstrip("/") == "process_recording":
            return "process_recording"
        for form in ("file=", "file/"):
            if rest.startswith(form) and _PROTOCOL_OR_UNC.match(rest[len(form):]):
                return "file-url"
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "http":
            which = self.refusal(scope)
            if which is not None:
                await _refuse(send, REFUSALS[which])  # receive() is never called
                return
        await self.app(scope, receive, send)


async def _refuse(send: Send, reason: str) -> None:
    body = reason.encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 403,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
                # The body was never read: the connection is not reused for a
                # next request that would start inside it.
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


__all__ = ["REFUSALS", "RefuseUnusedGradioRoutes"]
