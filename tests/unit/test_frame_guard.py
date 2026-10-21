"""
No page may frame the dashboard or the API (clickjacking).
==========================================================
D8 allows a request a dashboard makes to itself, and a dashboard loaded inside
another site's frame still makes same-origin requests, so the frame refusal is
the control against an invisible frame turning a click into an operator
control (`src/api/middleware/frame_guard.py`). These tests pin the headers on
the real application, the dashboard and a refused request included, and that
a response's own policy is kept.

**In a fresh interpreter** for the real application, as the service imports it
(`src.api.main` first; `tests/unit/test_dashboard_mount.py` says why).

Module: tests/unit/test_frame_guard.py
"""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.api.middleware.frame_guard import FRAME_HEADERS, RefuseFraming

REPO_ROOT = Path(__file__).resolve().parents[2]

PROBE = r'''
import json, logging
logging.disable(logging.CRITICAL)
import src.api.main as main
from fastapi.testclient import TestClient

client = TestClient(main.app)
answers = {}
for method, path, headers in (
    ("GET", "/", {}),
    ("GET", "/health", {}),
    ("GET", "/ui/", {}),
    ("GET", "/ui/config", {}),
    ("GET", "/api/v1/system/operator-access", {}),
    ("GET", "/ui/gradio_api/file=http://example.com/a.png", {}),
    ("OPTIONS", "/api/v1/pipeline/status",
     {"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"}),
):
    response = client.request(method, path, headers=headers)
    answers[f"{method} {path}"] = {
        "status": response.status_code,
        "xfo": response.headers.get_list("x-frame-options"),
        "csp": response.headers.get_list("content-security-policy"),
    }
print("ANSWERS" + json.dumps(answers))
'''


@pytest.fixture(scope="module")
def answers(tmp_path_factory):
    done = subprocess.run([sys.executable, "-c", PROBE], cwd=str(REPO_ROOT), capture_output=True,
                          text=True, timeout=600,
                          env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)})
    line = next((ln for ln in done.stdout.splitlines() if ln.startswith("ANSWERS")), None)
    assert line is not None, done.stdout[-2000:] + done.stderr[-4000:]
    return json.loads(line[len("ANSWERS"):])


def test_every_response_of_the_application_refuses_to_be_framed(answers):
    """The API, the dashboard, its config, a refused Gradio route and a refused
    cross-origin preflight: each says no page may frame it."""
    assert answers["GET /ui/"]["status"] == 200, answers
    assert answers["GET /ui/gradio_api/file=http://example.com/a.png"]["status"] == 403
    assert answers["OPTIONS /api/v1/pipeline/status"]["status"] == 400
    for request, answer in answers.items():
        assert answer["xfo"] == ["DENY"], (request, answer)
        assert answer["csp"] == ["frame-ancestors 'none'"], (request, answer)


def _run(app, scope):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(RefuseFraming(app)(scope, receive, send))
    return sent


def test_a_response_s_own_policy_is_kept_and_the_body_passes_unchanged():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-security-policy", b"default-src 'self'"),
                                (b"content-type", b"text/event-stream")]})
        await send({"type": "http.response.body", "body": b"data: 1\n\n", "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    sent = _run(app, {"type": "http", "method": "GET", "path": "/"})
    headers = sent[0]["headers"]
    assert (b"content-security-policy", b"default-src 'self'") in headers
    assert [v for k, v in headers if k == b"content-security-policy"] == [b"default-src 'self'"]
    assert (b"x-frame-options", b"DENY") in headers
    assert [m.get("body") for m in sent[1:]] == [b"data: 1\n\n", b""]


def test_a_websocket_and_a_lifespan_scope_pass_untouched():
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])
        await send({"type": "websocket.accept", "headers": []})

    sent = _run(app, {"type": "websocket", "path": "/"})
    assert seen == ["websocket"] and sent == [{"type": "websocket.accept", "headers": []}]
    assert dict(FRAME_HEADERS)[b"x-frame-options"] == b"DENY"
