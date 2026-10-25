"""
Requests D8 allows, and ones it refuses, for tests (M5-1)
=========================================================
D8's one check (`src/utils/operator_access.py`) reads the triggering request's
own ASGI scope and the deployment setting. A test that drives an operator
control for another reason (a reload's refusals, a training start's settings)
needs a request D8 allows; one about the gate itself needs requests it refuses.

- `operator_mode(monkeypatch)` sets ``SHEPHERD_DEPLOYMENT_MODE=M1`` for the test.
- `gradio_request(...)` is a real ``gr.Request`` wrapping a Starlette request
  built from a scope: loopback arrival, loopback Host and no Origin unless the
  test says otherwise.
- `operator_request(monkeypatch)` is both: a Gradio handler's request D8 allows.
- `OPERATOR_BASE_URL`: a `TestClient` with this base URL sends its requests
  with ``scope["server"] == ["127.0.0.1", 80]`` and ``Host: 127.0.0.1``, which
  D8 allows under M1; Starlette's default ``http://testserver`` is refused, as
  a request on any address but loopback is.

Module: tests/fixtures/operator_access.py
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

OPERATOR_BASE_URL = "http://127.0.0.1"
DEPLOYMENT_MODE = "SHEPHERD_DEPLOYMENT_MODE"


def operator_mode(monkeypatch) -> None:
    """The deployment is in mode M1 for this test."""
    monkeypatch.setenv(DEPLOYMENT_MODE, "M1")


def http_scope(server: Any = ("127.0.0.1", 8264), *,
               headers: Optional[Iterable[Tuple[str, str]]] = None,
               host: Optional[str] = "127.0.0.1:8264", path: str = "/ui/gradio_api/queue/join",
               scope_type: str = "http") -> dict:
    """An ASGI scope as uvicorn builds one for a request arriving on `server`
    with these headers (`host` first, unless `None`)."""
    pairs = [] if host is None else [(b"host", host.encode("latin-1"))]
    pairs += [(name.lower().encode("latin-1"), value.encode("latin-1"))
              for name, value in (headers or ())]
    return {"type": scope_type, "method": "POST", "path": path, "root_path": "",
            "scheme": "http", "query_string": b"", "headers": pairs,
            "server": None if server is None else list(server), "client": ["127.0.0.1", 50000]}


def gradio_request(server: Any = ("127.0.0.1", 8264), **kwargs: Any):
    """A ``gr.Request`` for an event triggered by the request `http_scope`
    describes."""
    import gradio as gr
    from starlette.requests import Request

    return gr.Request(request=Request(http_scope(server, **kwargs)))


def operator_request(monkeypatch, **kwargs: Any):
    """A Gradio event request D8 allows: mode M1 and a loopback request."""
    operator_mode(monkeypatch)
    return gradio_request(**kwargs)


__all__ = ["DEPLOYMENT_MODE", "OPERATOR_BASE_URL", "gradio_request", "http_scope",
           "operator_mode", "operator_request"]
