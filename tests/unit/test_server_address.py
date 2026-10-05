"""
The in-process WebUI calls the API at an address the server accepted a request on.
====================================================================================
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
    with pytest.raises(ServerAddressUnknownError, match="no TCP request has been recorded"):
        api_base_url()


# ------------------------------------------------------------- the middleware
def _tiny_app():
    """A Starlette app with the middleware, an HTTP route, a WebSocket route and a
    mounted sub-application, which is how the WebUI sits under the real app."""
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Mount, Route, WebSocketRoute

    from src.api.middleware.server_address import RecordServerAddress

    async def ok(request):
        return PlainTextResponse("ok")

    async def ws(websocket):
        await websocket.accept()
        await websocket.send_text("ok")
        await websocket.close()

    mounted = Starlette(routes=[Route("/config", ok)])
    app = Starlette(routes=[Route("/ok", ok), WebSocketRoute("/ws", ws), Mount("/ui", app=mounted)])
    app.add_middleware(RecordServerAddress)
    return app


def test_the_middleware_records_each_request_and_passes_it_through():
    from starlette.testclient import TestClient

    response = TestClient(_tiny_app()).get("/ok")

    assert response.status_code == 200 and response.text == "ok"
    # Starlette's TestClient reports its server as ("testserver", 80).
    assert api_base_url() == "http://testserver:80"


def test_a_websocket_connection_is_recorded_too():
    from starlette.testclient import TestClient

    with TestClient(_tiny_app()).websocket_connect("/ws") as connection:
        assert connection.receive_text() == "ok"

    assert api_base_url() == "http://testserver:80"


def test_a_request_to_a_mounted_application_passes_through_the_middleware():
    """The WebUI is a mount under the API app (`gr.mount_gradio_app(..., "/ui")`),
    so its requests reach the middleware as any route's do."""
    from starlette.testclient import TestClient

    response = TestClient(_tiny_app()).get("/ui/config")

    assert response.status_code == 200
    assert api_base_url() == "http://testserver:80"


def test_the_real_app_records_the_address():
    """Registered on `src.api.main.app` itself, not only on a test app."""
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

        # Proxy-free, as the tab's own calls are: a proxy variable in the test
        # environment must not route a loopback request away from the server.
        with requests.Session() as session:
            session.trust_env = False
            session.get(f"http://127.0.0.1:{bound_port}/ok", timeout=5).raise_for_status()

        assert api_base_url() == f"http://127.0.0.1:{bound_port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


# ------------------------------------------------------------- the Diagnosis tab
class _Response:
    def __init__(self, status=200, body=None, json_error=None):
        self.status_code = status
        self._body = {"initialized": True} if body is None else body
        self._json_error = json_error
        self.text = ""

    def raise_for_status(self):
        import requests

        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._body


def _record_self_calls(monkeypatch, panel, response=None, error=None):
    """Replace the HTTP layer under the tab; return the list of calls it saw."""
    calls = []

    def fake_request(session, method, url, **kwargs):
        calls.append((method, url, session.trust_env))
        if error is not None:
            raise error
        return response or _Response()

    monkeypatch.setattr(panel.requests.Session, "request", fake_request)
    return calls


def test_the_diagnosis_tab_calls_the_recorded_address(monkeypatch):
    pytest.importorskip("gradio")
    from src.webui.components import diagnosis_panel as panel

    calls = _record_self_calls(monkeypatch, panel)
    record_server_address(("127.0.0.1", 8264))

    panel._get_pipeline_status()
    panel._reload_pipeline("data/workspaces/default", "")
    panel._call_diagnose(["HP:0001250"])

    assert [(method, url) for method, url, _ in calls] == [
        ("GET", "http://127.0.0.1:8264/api/v1/pipeline/status"),
        ("POST", "http://127.0.0.1:8264/api/v1/pipeline/reload"),
        ("POST", "http://127.0.0.1:8264/api/v1/diagnose"),
    ]


def test_the_tab_never_sends_its_self_calls_through_a_proxy(monkeypatch):
    """The target can be a LAN address an institutional NO_PROXY list omits, and
    the diagnose payload carries patient phenotypes.

    Checked where requests hands a prepared request to the transport, so the real
    session code decides the proxies -- an attribute check alone would pass for a
    caller that turned trust_env off and then passed the environment's proxies in
    by hand. The control call shows the proxy variable is live in this test."""
    pytest.importorskip("gradio")
    from src.webui.components import diagnosis_panel as panel

    proxy = "http://proxy.invalid:3128"
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, proxy)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    sent = []

    def fake_send(adapter, request, **kwargs):
        sent.append((request.url, dict(kwargs.get("proxies") or {})))
        response = panel.requests.models.Response()
        response.status_code = 200
        response._content = b'{"initialized": true}'
        response.request = request
        response.url = request.url
        return response

    monkeypatch.setattr(panel.requests.adapters.HTTPAdapter, "send", fake_send)
    record_server_address(("192.0.2.10", 8264))

    with panel.requests.Session() as control:
        control.get("http://192.0.2.10:8264/control")
    assert sent[-1][1].get("http") == proxy, "control: the proxy variable is not live"

    sent.clear()
    panel._get_pipeline_status()
    panel._reload_pipeline("data/workspaces/default", "")
    panel._call_diagnose(["HP:0001250"])

    assert [url for url, _ in sent] == [
        "http://192.0.2.10:8264/api/v1/pipeline/status",
        "http://192.0.2.10:8264/api/v1/pipeline/reload",
        "http://192.0.2.10:8264/api/v1/diagnose",
    ]
    assert all(not proxies.get("http") and not proxies.get("all") for _, proxies in sent)


def test_an_unknown_address_is_reported_not_disguised_as_not_loaded(monkeypatch):
    """Before this, any failure to read the status rendered as "Pipeline not
    loaded", which is how the wrong port went unnoticed."""
    pytest.importorskip("gradio")
    from src.webui.components import diagnosis_panel as panel

    calls = _record_self_calls(monkeypatch, panel)

    status = panel._get_pipeline_status()
    rendered = panel._format_pipeline_status(status)

    assert "Pipeline status unavailable" in rendered
    assert "no TCP request has been recorded" in rendered
    assert "not loaded" not in rendered
    assert "no TCP request has been recorded" in panel._reload_pipeline("d", "")["message"]
    assert "no TCP request has been recorded" in panel._call_diagnose(["HP:0001250"])["error"]
    assert calls == [], "no request should be attempted without an address"


def test_an_unreachable_api_is_reported_too(monkeypatch):
    pytest.importorskip("gradio")
    from src.webui.components import diagnosis_panel as panel

    _record_self_calls(monkeypatch, panel, error=panel.requests.ConnectionError("refused"))
    record_server_address(("127.0.0.1", 8264))

    rendered = panel._format_pipeline_status(panel._get_pipeline_status())

    assert "Pipeline status unavailable:** API not reachable" in rendered
    assert "not loaded" not in rendered


@pytest.mark.parametrize(
    "error_name, reason",
    [("ConnectTimeout", "API not reachable"), ("ReadTimeout", "API did not answer within 5 s")],
)
def test_a_timeout_says_whether_the_connection_was_made(monkeypatch, error_name, reason):
    """A connect timeout never reached the server; a read timeout did, and is
    what a status read sees while a reload holds the server."""
    pytest.importorskip("gradio")
    from src.webui.components import diagnosis_panel as panel

    _record_self_calls(monkeypatch, panel, error=getattr(panel.requests, error_name)("slow"))
    record_server_address(("127.0.0.1", 8264))

    rendered = panel._format_pipeline_status(panel._get_pipeline_status())

    assert f"Pipeline status unavailable:** {reason}" in rendered


def test_an_api_that_answered_with_an_error_is_not_called_unreachable(monkeypatch):
    """It was reached; the reason shown says what it answered."""
    pytest.importorskip("gradio")
    from src.webui.components import diagnosis_panel as panel

    _record_self_calls(monkeypatch, panel, response=_Response(status=503))
    record_server_address(("127.0.0.1", 8264))

    rendered = panel._format_pipeline_status(panel._get_pipeline_status())

    assert "API error 503" in rendered
    assert "not reachable" not in rendered


def test_a_reply_that_is_not_json_is_named_as_such(monkeypatch):
    pytest.importorskip("gradio")
    from src.webui.components import diagnosis_panel as panel

    bad = panel.requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0)
    _record_self_calls(monkeypatch, panel, response=_Response(json_error=bad))
    record_server_address(("127.0.0.1", 8264))

    rendered = panel._format_pipeline_status(panel._get_pipeline_status())

    assert "the API's reply was not JSON" in rendered
    assert "not reachable" not in rendered


def test_a_request_that_was_never_sent_is_not_called_a_bad_reply(monkeypatch):
    """requests raises ValueError subclasses for a malformed URL or header too.
    Nothing was sent then, so "the API's reply was not JSON" would be false."""
    pytest.importorskip("gradio")
    from src.webui.components import diagnosis_panel as panel

    _record_self_calls(monkeypatch, panel, error=panel.requests.exceptions.InvalidURL("bad"))
    record_server_address(("127.0.0.1", 8264))

    rendered = panel._format_pipeline_status(panel._get_pipeline_status())

    assert "unexpected error" in rendered
    assert "not JSON" not in rendered
