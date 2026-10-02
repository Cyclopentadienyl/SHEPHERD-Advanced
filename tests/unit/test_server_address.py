"""
The in-process WebUI calls the API at the address the server actually bound.
============================================================================
The Diagnosis tab used to call a hard-coded `http://127.0.0.1:8000`. The systemd
unit starts the server on port 8264 (uvicorn takes the last `--port`), so under
the unit the tab called a port nothing listened on, and reported a serving
pipeline as not loaded.

The address is now recorded from each request's ASGI ``server`` field
(`src/utils/server_address.py`, written by
`src/api/middleware/server_address.py`). These tests pin the three things that
claim rests on: the recorder's rules, its registration on the real app, and —
with a real uvicorn socket — that the recorded port is the port the server bound.
"""
import threading
import time

import pytest

from src.utils import server_address
from src.utils.server_address import (
    ServerAddressUnknownError,
    api_base_url,
    record_server_address,
)


@pytest.fixture(autouse=True)
def _no_recorded_address(monkeypatch):
    """Each test starts with nothing recorded, and leaves nothing behind."""
    monkeypatch.setattr(server_address, "_address", None)


# ------------------------------------------------------------------ the recorder
def test_the_recorded_address_becomes_the_base_url():
    record_server_address(("127.0.0.1", 8264))
    assert api_base_url() == "http://127.0.0.1:8264"


def test_the_latest_address_wins():
    """Under a wildcard bind, requests arrive on different local addresses, and
    each is this server's own socket."""
    record_server_address(("127.0.0.1", 8264))
    record_server_address(("192.0.2.10", 8264))
    assert api_base_url() == "http://192.0.2.10:8264"


def test_an_ipv6_host_is_bracketed():
    record_server_address(("::1", 8264))
    assert api_base_url() == "http://[::1]:8264"


@pytest.mark.parametrize(
    "server",
    [None, ("/run/shepherd.sock", None), ("", 8264), ("127.0.0.1",), "127.0.0.1:8264"],
    ids=["absent", "unix-socket", "no-host", "one-field", "not-a-pair"],
)
def test_a_value_with_no_usable_host_and_port_is_ignored(server):
    """uvicorn reports a Unix-domain socket with no port; there is no http URL in
    it, and it must not erase a TCP address recorded earlier."""
    record_server_address(("127.0.0.1", 8264))
    record_server_address(server)
    assert api_base_url() == "http://127.0.0.1:8264"


def test_before_any_request_the_address_is_unknown_and_says_so():
    with pytest.raises(ServerAddressUnknownError, match="no request has reached it"):
        api_base_url()


# ------------------------------------------------------------- the middleware
def _tiny_app():
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    from src.api.middleware.server_address import RecordServerAddress

    async def ok(request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/ok", ok)])
    app.add_middleware(RecordServerAddress)
    return app


def test_the_middleware_records_each_request_and_passes_it_through():
    from starlette.testclient import TestClient

    response = TestClient(_tiny_app()).get("/ok")

    assert response.status_code == 200 and response.text == "ok"
    # Starlette's TestClient reports its server as ("testserver", 80).
    assert api_base_url() == "http://testserver:80"


def test_the_real_app_records_the_address():
    """Registered on `src.api.main.app`, outermost enough to see every request --
    the Gradio mount at /ui included, since it sits under the same app."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api.main import app

    TestClient(app).get("/health")

    assert api_base_url() == "http://testserver:80"


def test_under_a_real_server_the_recorded_port_is_the_bound_port():
    """The claim that matters: a server started on a port other than 8000 is
    called on that port. Port 0 lets the kernel choose, so the test cannot pass
    by coinciding with any default."""
    uvicorn = pytest.importorskip("uvicorn")
    import requests

    config = uvicorn.Config(_tiny_app(), host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.02)
        bound_port = server.servers[0].sockets[0].getsockname()[1]
        assert bound_port != 8000

        requests.get(f"http://127.0.0.1:{bound_port}/ok", timeout=5).raise_for_status()

        assert api_base_url() == f"http://127.0.0.1:{bound_port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


# ------------------------------------------------------------- the Diagnosis tab
def test_the_diagnosis_tab_calls_the_recorded_address(monkeypatch):
    pytest.importorskip("gradio")
    from src.webui.components import diagnosis_panel as panel

    called = []

    class _Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"initialized": True}

    def fake_get(url, timeout):
        called.append(url)
        return _Response()

    def fake_post(url, json, timeout):
        called.append(url)
        return _Response()

    monkeypatch.setattr(panel.requests, "get", fake_get)
    monkeypatch.setattr(panel.requests, "post", fake_post)
    record_server_address(("127.0.0.1", 8264))

    panel._get_pipeline_status()
    panel._reload_pipeline("data/workspaces/default", "")
    panel._call_diagnose(["HP:0001250"])

    assert called == [
        "http://127.0.0.1:8264/api/v1/pipeline/status",
        "http://127.0.0.1:8264/api/v1/pipeline/reload",
        "http://127.0.0.1:8264/api/v1/diagnose",
    ]


def test_an_unknown_address_is_reported_not_disguised_as_not_loaded(monkeypatch):
    """Before this, any failure to read the status rendered as "Pipeline not
    loaded", which is how the wrong port went unnoticed."""
    pytest.importorskip("gradio")
    from src.webui.components import diagnosis_panel as panel

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("no request should be attempted without an address")

    monkeypatch.setattr(panel.requests, "get", must_not_be_called)
    monkeypatch.setattr(panel.requests, "post", must_not_be_called)

    status = panel._get_pipeline_status()
    rendered = panel._format_pipeline_status(status)

    assert "Pipeline status unavailable" in rendered
    assert "no request has reached it" in rendered
    assert "not loaded" not in rendered
    assert "no request has reached it" in panel._reload_pipeline("d", "")["message"]
    assert "no request has reached it" in panel._call_diagnose(["HP:0001250"])["error"]


def test_an_unreachable_api_is_reported_too(monkeypatch):
    pytest.importorskip("gradio")
    from src.webui.components import diagnosis_panel as panel

    def refused(*args, **kwargs):
        raise panel.requests.ConnectionError("refused")

    monkeypatch.setattr(panel.requests, "get", refused)
    record_server_address(("127.0.0.1", 8264))

    rendered = panel._format_pipeline_status(panel._get_pipeline_status())

    assert "Pipeline status unavailable" in rendered
    assert "not loaded" not in rendered
