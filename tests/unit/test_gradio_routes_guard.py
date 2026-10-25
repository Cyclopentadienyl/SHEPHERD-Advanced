"""
The guard in front of the Gradio mount, on its own.
===================================================
`RefuseUnusedGradioRoutes` (src/api/middleware/gradio_routes.py) refuses
``<mount>/gradio_api/process_recording``, ``upload`` and ``upload_progress``,
and ``file=`` / ``file/`` targets that start with a URL scheme, and passes
everything else through untouched. These
tests pin the matching (on the decoded path Starlette routes on), the refusal
(403, a plain reason, ``connection: close``) and that a refusal never reads
the request body. The guard on the real service, with Gradio behind it, is
tested in `test_dashboard_mount.py`.

Module: tests/unit/test_gradio_routes_guard.py
"""
import asyncio

import pytest

from src.api.middleware.gradio_routes import REFUSALS, RefuseUnusedGradioRoutes

GUARD = RefuseUnusedGradioRoutes(app=None, mount_path="/ui")


def _scope(path, root_path="", method="GET"):
    return {"type": "http", "method": method, "path": path, "root_path": root_path}


@pytest.mark.parametrize("path", [
    # The URL form Gradio 6.30 fetches server-side, in each spelling that
    # reaches it once Starlette has decoded the path.
    "/ui/gradio_api/file=http://example.com/a.png",
    "/ui/gradio_api/file=https://example.com/a.txt",
    "/ui/gradio_api/file=HTTP://example.com/a.json",
    "/ui/gradio_api/file=hTtPs://example.com/a",
    "/ui/gradio_api/file=https://169.254.169.254/latest/meta-data/",
    "/ui/gradio_api/file/http://example.com/a.png",
    # Other schemes and network-path forms (Gradio refuses these itself today).
    "/ui/gradio_api/file=ftp://example.com/a",
    "/ui/gradio_api/file=//example.com/a",
    "/ui/gradio_api/file=\\\\host\\share\\a",
    "/ui/gradio_api/file=\\/host/a",
])
def test_every_url_form_of_file_is_refused(path):
    assert GUARD.refusal(_scope(path)) == "file-url"


@pytest.mark.parametrize("path", [
    "/ui/gradio_api/process_recording",
    "/ui/gradio_api/process_recording/",
])
def test_process_recording_is_refused(path):
    for method in ("POST", "GET", "OPTIONS"):
        assert GUARD.refusal(_scope(path, method=method)) == "process_recording"


@pytest.mark.parametrize("path", [
    "/ui/gradio_api/upload",
    "/ui/gradio_api/upload/",
    "/ui/gradio_api/upload_progress",
])
def test_the_upload_routes_are_refused(path):
    """The dashboard has no upload control, and another site's page can post
    to the upload route with no preflight (M5-1)."""
    for method in ("POST", "GET", "OPTIONS"):
        assert GUARD.refusal(_scope(path, method=method)) == "upload"


@pytest.mark.parametrize("path", [
    # Local exports, as gr.File / gr.DownloadButton link them (Linux and Windows).
    "/ui/gradio_api/file=/tmp/gradio/abc/training_metrics_x.csv",
    "/ui/gradio_api/file=C:\\Users\\a\\AppData\\Local\\Temp\\gradio\\x.csv",
    "/ui/gradio_api/file=C:/Users/a/AppData/Local/Temp/gradio/x.md",
    "/ui/gradio_api/file//tmp/gradio/abc/x.csv",
    "/ui/gradio_api/file=/tmp/http://not-a-scheme-at-the-start.csv",
    # The rest of Gradio's API, and everything outside the mount.
    "/ui/gradio_api/uploads",
    "/ui/gradio_api/queue/join",
    "/ui/gradio_api/info",
    "/ui/config",
    "/ui/",
    "/api/v1/training/status",
    "/api/file=http://example.com",
    "/other/gradio_api/process_recording",
    "/other/gradio_api/upload",
    "/health",
])
def test_everything_else_passes(path):
    assert GUARD.refusal(_scope(path)) is None


def test_the_path_is_matched_below_root_path_as_starlette_routes_it():
    assert GUARD.refusal(_scope("/p/ui/gradio_api/process_recording", "/p")) == "process_recording"
    assert GUARD.refusal(_scope("/p/ui/gradio_api/file=http://x", "/p")) == "file-url"
    assert GUARD.refusal(_scope("/p/ui/gradio_api/file=/tmp/a.csv", "/p")) is None


def test_another_mount_path_is_guarded_instead():
    guard = RefuseUnusedGradioRoutes(app=None, mount_path="/dash/")
    assert guard.refusal(_scope("/dash/gradio_api/process_recording")) == "process_recording"
    assert guard.refusal(_scope("/ui/gradio_api/process_recording")) is None


def _run(app, scope):
    sent = []

    async def receive():
        raise AssertionError("the request body was read")

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return sent


@pytest.mark.parametrize("path,which", [
    ("/ui/gradio_api/process_recording", "process_recording"),
    ("/ui/gradio_api/file=http://example.com/evil.png", "file-url"),
    ("/ui/gradio_api/upload", "upload"),
])
def test_a_refusal_is_a_plain_403_that_never_reads_the_body(path, which):
    async def inner(scope, receive, send):
        raise AssertionError("Gradio was reached")

    sent = _run(RefuseUnusedGradioRoutes(inner, "/ui"), _scope(path, method="POST"))

    start, body = sent
    headers = dict(start["headers"])
    assert start["status"] == 403
    assert headers[b"connection"] == b"close"
    assert headers[b"content-type"].startswith(b"text/plain")
    assert body["body"] == REFUSALS[which].encode()
    assert headers[b"content-length"] == str(len(body["body"])).encode()


def test_other_requests_reach_the_app_with_their_own_receive():
    seen = []

    async def inner(scope, receive, send):
        seen.append((scope["path"], receive))

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    guard = RefuseUnusedGradioRoutes(inner, "/ui")
    asyncio.run(guard(_scope("/ui/gradio_api/queue/join", method="POST"), receive, None))
    assert seen == [("/ui/gradio_api/queue/join", receive)]


def test_websocket_and_lifespan_scopes_pass_through():
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    guard = RefuseUnusedGradioRoutes(inner, "/ui")
    for kind in ("websocket", "lifespan"):
        asyncio.run(guard({"type": kind, "path": "/ui/gradio_api/process_recording"}, None, None))
    assert seen == ["websocket", "lifespan"]
