"""
D8's one check, its FastAPI dependency and its Gradio helper (M5-1).
====================================================================
`PLAN_TEST_RESULTS.md` D8 and §6 ("M5's boundaries"): an operator control is
allowed only under deployment mode M1, for a request that arrived on a loopback
socket, carries no forwarding header, is addressed to a loopback host, when
it carries an Origin comes from that same address, and is not reported by the
browser as sent from another site (``Sec-Fetch-Site``). The check reads the
triggering request's own scope (never ``scope["client"]``), fails closed, and
its refusals name the condition without repeating a header's value.

The acceptance (§6): "a direct call to a gated API route or Gradio event meets
the same D8 check". The Gradio events are called here through Gradio's own HTTP
API on the real dashboard, as `gradio_client` or a script would call them.

Module: tests/unit/test_operator_access.py
"""
from __future__ import annotations

import inspect
import json

from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from src.utils.operator_access import (
    CONDITIONS,
    FORWARDING_HEADERS,
    OperatorAccess,
    check_operator_access,
)
from tests.fixtures.operator_access import (
    OPERATOR_BASE_URL,
    gradio_request,
    http_scope,
    operator_mode,
)

M1 = {"SHEPHERD_DEPLOYMENT_MODE": "M1"}
#: A value no refusal may repeat: header values are the client's.
CANARY = "canary-7f3a"


def check(scope, environ=M1) -> OperatorAccess:
    return check_operator_access(scope, environ=environ)


# --------------------------------------------------------------------------- allowed
@pytest.mark.parametrize("server, host, origin", [
    (("127.0.0.1", 8264), "127.0.0.1:8264", None),
    (("127.0.0.1", 8264), "127.0.0.1:8264", "http://127.0.0.1:8264"),
    (("127.0.0.1", 443), "localhost", "https://localhost"),
    (("127.0.0.1", 8264), "LocalHost:8264", "http://localhost:8264"),
    (("::1", 8264), "[::1]:8264", "http://[::1]:8264"),
    (("::ffff:127.0.0.1", 8264), "127.0.0.1:8264", None),
    (["127.0.0.53", 8264], "localhost:9000", None),
], ids=["no-origin", "same-origin", "https-default-port", "host-case", "ipv6", "v4-mapped",
        "other-loopback-and-forwarded-port"])
def test_m1_on_loopback_from_the_dashboard_itself_is_allowed(server, host, origin):
    headers = [("origin", origin)] if origin else []
    decision = check(http_scope(server, host=host, headers=headers))
    assert decision.allowed and decision.condition == "enabled"
    assert decision.refusal is None
    assert decision.as_status() == {"enabled": True, "condition": "enabled",
                                    "reason": decision.reason}


def test_the_client_address_is_never_read():
    """uvicorn rewrites ``client`` from X-Forwarded-For; only ``server``, the
    socket that accepted the request, decides."""
    lan_server = http_scope(("192.0.2.10", 8264))
    lan_server["client"] = ["127.0.0.1", 50000]
    assert check(lan_server).condition == "not_loopback"
    lan_client = http_scope()
    lan_client["client"] = ["192.0.2.99", 50000]
    assert check(lan_client).allowed
    del lan_client["client"]
    assert check(lan_client).allowed


# --------------------------------------------------------------------------- refused
@pytest.mark.parametrize("environ, says", [
    ({}, "SHEPHERD_DEPLOYMENT_MODE is not set"),
    ({"SHEPHERD_DEPLOYMENT_MODE": ""}, "SHEPHERD_DEPLOYMENT_MODE is not set"),
    ({"SHEPHERD_DEPLOYMENT_MODE": "m1"}, "not to M1"),
    ({"SHEPHERD_DEPLOYMENT_MODE": " M1"}, "not to M1"),
    ({"SHEPHERD_DEPLOYMENT_MODE": "M3"}, "not to M1"),
])
def test_without_mode_m1_every_request_is_refused(environ, says):
    decision = check(http_scope(), environ)
    assert (decision.allowed, decision.condition) == (False, "deployment_mode")
    assert says in decision.reason and "read-only" in decision.reason


def test_the_mode_is_read_at_each_check(monkeypatch):
    scope = http_scope()
    assert check_operator_access(scope).condition == "deployment_mode"
    operator_mode(monkeypatch)
    assert check_operator_access(scope).allowed
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE")
    assert not check_operator_access(scope).allowed


def test_the_host_check_reads_the_loopback_names(monkeypatch):
    """M5-3 review ARCH-6: the set of loopback Host names was a constant the
    check never read (a second copy in a regex); the check now reads it, so
    editing the set changes what is allowed."""
    from src.utils import operator_access

    scope = http_scope(("127.0.0.1", 8264), host="127.0.0.1:8264")
    assert check(scope).allowed
    monkeypatch.setattr(operator_access, "LOOPBACK_HOST_NAMES", ("localhost", "[::1]"))
    assert check(scope).condition == "host_not_loopback"
    assert check(http_scope(("127.0.0.1", 8264), host="localhost:8264")).allowed


@pytest.mark.parametrize("host", ["localhost:", "localhost:abc", "user@localhost",
                                  "localhost.", "127.1", "[::1", "::1", "localhost:65536",
                                  "[::2]:80", "localhost:80:80", ""])
def test_a_host_that_only_looks_like_loopback_is_refused(host):
    assert check(http_scope(("127.0.0.1", 8264), host=host)).condition == "host_not_loopback"


class _Exploding(dict):
    def get(self, *args, **kwargs):
        raise RuntimeError("unreadable")


@pytest.mark.parametrize("scope", [
    None,
    "not a scope",
    {"type": "lifespan"},
    http_scope(server=None),
    {**http_scope(), "server": ["127.0.0.1"]},
    {**http_scope(), "server": [None, 8264]},
    {**http_scope(), "server": "127.0.0.1:8264"},
    {**http_scope(), "headers": None},
    {**http_scope(), "headers": "host: 127.0.0.1"},
    {**http_scope(), "headers": [("host", "127.0.0.1")]},
    {**http_scope(), "headers": [(b"host",)]},
    _Exploding(http_scope()),
], ids=["none", "string", "lifespan", "no-server", "server-one-item", "server-no-host",
        "server-string", "headers-none", "headers-string", "headers-text", "header-half",
        "raises"])
def test_a_request_that_cannot_be_read_is_refused(scope):
    decision = check(scope)
    assert (decision.allowed, decision.condition) == (False, "unreadable_request")
    assert "fails closed" in decision.reason


@pytest.mark.parametrize("server", [
    ("192.0.2.10", 8264), ("0.0.0.0", 8264), ("testserver", 80), ("localhost", 8264),
    ("::ffff:192.0.2.1", 8264), ("fe80::1", 8264), ("/run/shepherd.sock", None),
], ids=["lan", "wildcard", "testclient-default", "a-name", "mapped-lan", "link-local",
        "unix-socket"])
def test_a_request_that_did_not_arrive_on_loopback_is_refused(server):
    decision = check(http_scope(server))
    assert (decision.allowed, decision.condition) == (False, "not_loopback")
    assert "ssh -L" in decision.reason


@pytest.mark.parametrize("name", FORWARDING_HEADERS)
def test_every_forwarding_header_is_refused_by_name_never_by_value(name):
    spelled = "-".join(part.capitalize() for part in name.split("-"))
    decision = check(http_scope(headers=[(spelled, CANARY)]))
    assert (decision.allowed, decision.condition) == (False, "forwarding_header")
    assert f"({name})" in decision.reason and CANARY not in decision.reason


def test_the_forwarding_list_is_the_extended_one():
    assert set(FORWARDING_HEADERS) == {
        "forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto", "x-real-ip",
        "x-original-forwarded-for", "cf-connecting-ip", "true-client-ip"}


@pytest.mark.parametrize("hosts", [
    [f"{CANARY}.example"], [f"{CANARY}.example:8264"], ["127.0.0.2:8264"], ["localhost."],
    ["127.0.0.1:99999"], ["127.0.0.1:"], ["[::1]x"], [], ["127.0.0.1", "localhost"],
], ids=["rebinding-name", "rebinding-name-port", "other-loopback-address", "trailing-dot",
        "port-out-of-range", "empty-port", "junk", "missing", "repeated"])
def test_a_host_that_is_not_a_loopback_name_is_refused(hosts):
    scope = http_scope(host=None, headers=[("host", value) for value in hosts])
    decision = check(scope)
    assert (decision.allowed, decision.condition) == (False, "host_not_loopback")
    assert "DNS-rebinding" in decision.reason and CANARY not in decision.reason


@pytest.mark.parametrize("origins", [
    [f"https://{CANARY}.example"], ["null"], ["http://127.0.0.1:9999"],
    ["http://localhost:8264"], ["http://127.0.0.1:8264/"], ["ftp://127.0.0.1:8264"],
    ["http://127.0.0.1:8264", "http://127.0.0.1:8264"],
], ids=["foreign-site", "null", "other-port", "other-name", "with-path", "other-scheme",
        "repeated"])
def test_an_origin_other_than_this_address_is_refused(origins):
    scope = http_scope(host="127.0.0.1:8264", headers=[("origin", o) for o in origins])
    decision = check(scope)
    assert (decision.allowed, decision.condition) == (False, "origin_mismatch")
    assert "another website" in decision.reason and CANARY not in decision.reason


#: The headers a browser sends with an image another site's page loads: no
#: Origin on a GET in no-cors mode, so only Sec-Fetch-Site says where it came from.
CROSS_SITE_IMAGE = [("sec-fetch-site", "cross-site"), ("sec-fetch-mode", "no-cors"),
                    ("sec-fetch-dest", "image")]


@pytest.mark.parametrize("sites", [
    ["cross-site"], ["same-site"], ["CROSS-SITE"], [CANARY], [""],
    ["same-origin", "same-origin"], ["same-origin", "cross-site"],
], ids=["cross-site", "same-site", "uppercase", "unknown-value", "empty", "repeated",
        "repeated-mixed"])
def test_a_request_another_site_sent_is_refused_though_it_carries_no_origin(sites):
    """A GET another site's page makes with an image, a script or a link
    carries no Origin; the browser says where it came from in Sec-Fetch-Site,
    and only ``same-origin`` and ``none`` (the address bar) pass."""
    scope = http_scope(headers=[("sec-fetch-site", s) for s in sites]
                       + [("sec-fetch-mode", "no-cors"), ("sec-fetch-dest", "image")])
    decision = check(scope)
    assert (decision.allowed, decision.condition) == (False, "cross_site")
    assert "another website" in decision.reason and CANARY not in decision.reason


@pytest.mark.parametrize("site", ["same-origin", "none", "Same-Origin"])
def test_a_request_from_the_dashboard_or_the_address_bar_is_allowed(site):
    with_origin = http_scope(headers=[("origin", "http://127.0.0.1:8264"),
                                      ("sec-fetch-site", site)])
    without = http_scope(headers=[("sec-fetch-site", site)])
    assert check(with_origin).allowed and check(without).allowed


def test_the_conditions_are_checked_in_order():
    """Mode first, then readable, loopback, forwarding, Host, Origin,
    Sec-Fetch-Site: the first that fails is the one reported."""
    everything_wrong = http_scope(("192.0.2.10", 8264), host="evil.example",
                                  headers=[("x-forwarded-for", "1.2.3.4"),
                                           ("origin", "https://evil.example")])
    assert check(everything_wrong, {}).condition == "deployment_mode"
    assert check(everything_wrong).condition == "not_loopback"
    everything_wrong["server"] = ["127.0.0.1", 8264]
    assert check(everything_wrong).condition == "forwarding_header"
    everything_wrong["headers"] = [h for h in everything_wrong["headers"]
                                   if h[0] != b"x-forwarded-for"]
    assert check(everything_wrong).condition == "host_not_loopback"
    everything_wrong["headers"] = [(b"host", b"127.0.0.1:8264"), (b"origin", b"https://x"),
                                   (b"sec-fetch-site", b"cross-site")]
    assert check(everything_wrong).condition == "origin_mismatch"
    everything_wrong["headers"][1] = (b"origin", b"http://127.0.0.1:8264")
    assert check(everything_wrong).condition == "cross_site"


def test_every_refusal_says_nothing_was_done_and_names_a_condition():
    decision = check(http_scope(("192.0.2.10", 8264)))
    assert decision.refusal == f"Refused: {decision.reason} Nothing was done."
    assert decision.condition in CONDITIONS
    assert decision.as_status()["enabled"] is False


def test_the_check_imports_nothing_posix_only_nor_the_service():
    """Windows: the backend must not crash for want of a POSIX module; and the
    check sits in src.utils, below the API it serves."""
    import ast
    from pathlib import Path

    import src.utils.operator_access as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                 for alias in node.names}
    assert not imported & {"fcntl", "pwd", "grp", "resource", "termios", "fastapi", "gradio"}
    assert not any(name and name.startswith("src.api") for name in imported)


# --------------------------------------------------------------------------- the Gradio helper
def test_the_gradio_helper_reads_the_triggering_requests_scope(monkeypatch):
    from src.webui.operator_access import operator_access

    operator_mode(monkeypatch)
    assert operator_access(gradio_request()).allowed
    assert operator_access(gradio_request(("192.0.2.10", 8264))).condition == "not_loopback"


def test_the_gradio_helper_fails_closed_without_a_request(monkeypatch):
    import gradio as gr

    from src.webui.operator_access import operator_access

    operator_mode(monkeypatch)
    for request in (None, gr.Request(), gr.Request(scope=http_scope()), object()):
        decision = operator_access(request)
        assert (decision.allowed, decision.condition) == (False, "unreadable_request")


@pytest.mark.parametrize("queue", [True, False], ids=["queued", "queue-false"])
def test_the_helper_sees_the_request_that_triggered_the_event(monkeypatch, queue):
    """Under the queue and with ``queue=False`` alike, ``gr.Request.request``
    is the HTTP call that started the event (`call/<api>` or `run/<api>`), so
    the helper judges that caller's own arrival, Host and Origin. (`probe`'s
    annotation is resolved in this module's globals, as Gradio resolves a
    handler's, hence the module-level `gr`.)"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.webui.operator_access import operator_access

    def probe(text, request: gr.Request):
        return operator_access(request).condition

    with gr.Blocks(analytics_enabled=False) as demo:
        box = gr.Textbox()
        gr.Button().click(probe, box, box, api_name="probe", queue=queue)
    app = gr.mount_gradio_app(FastAPI(), demo, path="/ui", run_history=False,
                              enable_monitoring=False, mcp_server=False, ssr_mode=False)
    operator_mode(monkeypatch)

    def condition(base_url, headers=None):
        with TestClient(app, base_url=base_url, headers=headers or {}) as client:
            if not queue:
                return client.post("/ui/gradio_api/run/probe",
                                   json={"data": ["x"]}).json()["data"][0]
            started = client.post("/ui/gradio_api/call/probe", json={"data": ["x"]}).json()
            stream = client.get(f"/ui/gradio_api/call/probe/{started['event_id']}").text
            return json.loads(stream.strip().splitlines()[-1][len("data:"):])[0]

    assert condition(OPERATOR_BASE_URL) == "enabled"
    assert condition("http://192.0.2.10") == "not_loopback"
    assert condition(OPERATOR_BASE_URL, {"Origin": "https://evil.example"}) == "origin_mismatch"


# --------------------------------------------------------------------------- the API
def _app_client(base_url=OPERATOR_BASE_URL, **kwargs):
    from fastapi.testclient import TestClient

    from src.api.main import app

    return TestClient(app, base_url=base_url, **kwargs)


def test_every_state_changing_route_takes_the_one_dependency():
    """Every POST on the API except diagnosis (C1, open by SPEC_4), and the two
    checkpoint lists (C3), are gated by `require_operator_access`, and nothing
    else is: read-only routes stay open."""
    from fastapi.routing import APIRoute

    from src.api.main import app
    from src.api.operator_access import require_operator_access

    gated, open_routes = set(), set()
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        deps = [d.dependency for d in route.dependencies]
        assert deps in ([], [require_operator_access]), route.path
        for method in route.methods:
            (gated if deps else open_routes).add((method, route.path))
    assert gated == {
        ("POST", "/api/v1/pipeline/reload"), ("POST", "/api/v1/pipeline/config"),
        ("POST", "/api/v1/training/start"), ("POST", "/api/v1/training/stop"),
        ("GET", "/api/v1/training/checkpoints"), ("GET", "/api/v1/workspaces/{name}/checkpoints"),
    }
    assert {method for method, _ in open_routes} == {"GET", "POST"}
    assert [path for method, path in open_routes if method == "POST"] == ["/api/v1/diagnose"]


def test_the_config_save_is_refused_and_writes_nothing(monkeypatch, tmp_path):
    from src.api.routes import pipeline

    target = tmp_path / "ui.json"
    monkeypatch.setattr(pipeline, "CONFIG_FILE", target)
    body = {"data_dir": "data/workspaces/x", "checkpoint_path": None}
    refused = _app_client("http://192.0.2.10").post("/api/v1/pipeline/config", json=body)
    assert refused.status_code == 403 and not target.exists()
    operator_mode(monkeypatch)
    assert _app_client().post("/api/v1/pipeline/config", json=body).status_code == 200
    assert json.loads(target.read_text()) == body


def test_the_reload_route_is_refused_before_the_reload_runs(monkeypatch):
    from src.api.routes import pipeline

    monkeypatch.setattr(pipeline, "run_reload",
                        lambda request: pytest.fail("a refused request reached the reload"))
    operator_mode(monkeypatch)
    response = _app_client().post("/api/v1/pipeline/reload", json={"data_dir": "x"},
                                  headers={"Origin": f"https://{CANARY}.example"})
    assert response.status_code == 403
    assert "another website" in response.json()["error"] and CANARY not in response.text


def test_the_reload_route_runs_the_reload_on_the_event_loop(monkeypatch):
    """The route calls `run_reload` on the event loop, never in a thread, as
    the dashboard's Reload does (`test_the_reload_handler_runs_the_reload_on_
    the_event_loop`): two reloads, or a reload and a diagnosis, then never run
    side by side, whichever entrance each came through (BACKLOG 28)."""
    import asyncio

    from src.api.routes import pipeline

    assert inspect.iscoroutinefunction(pipeline.reload_pipeline)
    assert not inspect.iscoroutinefunction(pipeline.run_reload)
    where = []

    def reload_spy(request):
        try:
            asyncio.get_running_loop()
            where.append("event loop")
        except RuntimeError:
            where.append("worker thread")
        return pipeline._refused("spy", {})

    monkeypatch.setattr(pipeline, "run_reload", reload_spy)
    operator_mode(monkeypatch)
    _app_client().post("/api/v1/pipeline/reload", json={"data_dir": "x"})
    assert where == ["event loop"]


@pytest.mark.parametrize("url, service", [
    ("/api/v1/training/checkpoints", "get_checkpoints"),
    ("/api/v1/workspaces/demo/checkpoints", "list_checkpoints"),
])
def test_another_sites_image_cannot_make_either_checkpoint_list_open_a_file(
        monkeypatch, url, service):
    """An ``<img>`` on another site's page sends a GET with no Origin; its
    Sec-Fetch-Site refuses it before a checkpoint is opened (C3), while the
    same address typed in the browser's address bar is served."""
    from src.api.services import server_lists
    from src.api.services.training_manager import training_manager

    reached = []
    monkeypatch.setattr(training_manager, "get_checkpoints",
                        lambda *a, **k: reached.append("get_checkpoints") or [])
    monkeypatch.setattr(server_lists, "list_checkpoints",
                        lambda *a, **k: reached.append("list_checkpoints")
                        or server_lists.Listing(()))
    operator_mode(monkeypatch)
    image = dict(CROSS_SITE_IMAGE)
    refused = _app_client().get(url, headers=image)
    assert refused.status_code == 403 and reached == []
    assert "another website" in refused.json()["error"]
    typed = _app_client().get(url, headers={"Sec-Fetch-Site": "none",
                                            "Sec-Fetch-Mode": "navigate"})
    assert typed.status_code == 200 and reached == [service]


def test_the_status_route_answers_for_the_request_that_asks(monkeypatch):
    assert _app_client().get("/api/v1/system/operator-access").json()["condition"] == (
        "deployment_mode")
    operator_mode(monkeypatch)
    assert _app_client().get("/api/v1/system/operator-access").json()["enabled"] is True
    lan = _app_client("http://192.0.2.10").get("/api/v1/system/operator-access")
    assert lan.status_code == 200 and lan.json()["condition"] == "not_loopback"
    forwarded = _app_client().get("/api/v1/system/operator-access",
                                  headers={"X-Real-IP": CANARY})
    assert forwarded.json()["condition"] == "forwarding_header"
    assert CANARY not in forwarded.text


# --------------------------------------------------------------------------- the dashboard
#: Every state-changing Gradio handler, by module and name. Each takes a
#: positional `gr.Request` (the one way Gradio passes it) and calls the helper.
GATED_HANDLERS = {
    "src.webui.components.diagnosis_panel": ("_on_change_model",),
    "src.webui.components.runtime_settings": ("_on_restart", "_on_apply"),
    "src.webui.components.training_console": (
        "_on_start", "_on_stop", "_on_resume", "_on_check_continue", "_refresh_checkpoints",
        "_save_training_config", "_on_stop_request", "_on_continue_open"),
}


def _gated():
    import importlib

    for module_name, names in GATED_HANDLERS.items():
        module = importlib.import_module(module_name)
        for name in names:
            yield module, name, getattr(module, name)


def test_every_gated_handler_takes_a_positional_gradio_request():
    import gradio as gr
    import typing

    for module, name, fn in _gated():
        hints = typing.get_type_hints(fn)
        positional = [p for p in inspect.signature(fn).parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        assert [p.name for p in positional if hints.get(p.name) is gr.Request] == ["request"], name
        assert "operator_access(request)" in inspect.getsource(fn), name


def test_no_gated_handler_calls_the_api_over_http():
    """In process, never `_self_request`: the API would judge a loopback
    self-call as local whoever clicked (D8). Diagnosis reloads through the
    shared Load (`model_list`, UI-A's own tests) and writes no configuration:
    its page calls neither route nor writer."""
    from src.webui.components import diagnosis_card
    from src.webui.components import diagnosis_panel as dp

    for name in ("_on_change_model", "open_change"):
        source = inspect.getsource(getattr(dp, name))
        assert "_self_request" not in source and "/pipeline/reload" not in source, name
    for module in (dp, diagnosis_card):
        source = inspect.getsource(module)
        for forbidden in ("run_reload(", "write_ui_config(", '"/api/v1/pipeline/reload',
                          '"/api/v1/pipeline/config'):
            assert forbidden not in source, (module.__name__, forbidden)


@pytest.fixture(scope="module")
def dashboard_app():
    """The dashboard mounted as the service mounts it (`mount_dashboard`), on a
    parent app with the service's own middleware, CORS included, copied from
    `src.api.main.app` in its order.

    Not `src.api.main.app` itself: when a test process happens to import
    `src.webui` before `src.api.main`, the service's mount fails at import
    (`tests/unit/test_dashboard_mount.py` says why) and that app has no /ui.
    The handlers, the helper and the services are the real ones."""
    from fastapi import FastAPI

    from src.api.main import app as service
    from src.webui.app import mount_dashboard

    parent = FastAPI()
    for middleware in reversed(service.user_middleware):
        parent.add_middleware(middleware.cls, *middleware.args, **middleware.kwargs)
    mount_dashboard(parent, "/ui")
    return parent


def _dashboard_client(app, base_url=OPERATOR_BASE_URL, **kwargs):
    from fastapi.testclient import TestClient

    return TestClient(app, base_url=base_url, **kwargs)


class _Dashboard:
    """The dashboard driven through Gradio's own HTTP API
    (`/ui/gradio_api/call/<api_name>`), as a script or `gradio_client` calls
    an event."""

    def __init__(self, client, app):
        self.client = client
        self.blocks = next(route.app.get_blocks() for route in app.routes
                           if getattr(route, "path", "") == "/ui")

    def api_name(self, fn) -> str:
        names = [d.api_name for d in self.blocks.fns.values()
                 if getattr(d.fn, "__wrapped__", d.fn) is fn]
        assert names, fn
        return names[0]

    def call(self, fn, data=None):
        dependency = next(d for d in self.blocks.fns.values()
                          if d.api_name == self.api_name(fn))
        if data is None:
            data = [getattr(block, "value", None) for block in dependency.inputs]
        name = self.api_name(fn)
        started = self.client.post(f"/ui/gradio_api/call/{name}", json={"data": data})
        assert started.status_code == 200, started.text
        stream = self.client.get(f"/ui/gradio_api/call/{name}/{started.json()['event_id']}")
        events = [line for line in stream.text.splitlines() if line.startswith(("event:", "data:"))]
        assert "event: complete" in events, stream.text
        return json.loads(events[events.index("event: complete") + 1][len("data:"):])


@pytest.fixture
def spies(monkeypatch, tmp_path):
    """Every service a gated handler reaches, replaced by a recorder, and the
    files they write pointed into tmp_path."""
    from src.api.routes import pipeline
    from src.api.services.training_manager import training_manager
    from src.webui.components import diagnosis_panel as dp
    from src.webui.components import runtime_settings as rs
    from src.webui.components import training_console as tc

    reached = []

    def recorder(name, answer):
        def record(*args, **kwargs):
            reached.append(name)
            return answer
        return record

    monkeypatch.setattr(pipeline, "run_reload", recorder("reload", pipeline._refused("spy", {})))
    monkeypatch.setattr(pipeline, "write_ui_config", recorder("write_ui_config", None))
    monkeypatch.setattr(dp, "_self_request",
                        lambda *a, **k: pytest.fail("a gated handler called the API over HTTP"))
    monkeypatch.setattr(rs, "restart_backend", recorder("restart", {"success": False,
                                                                    "error": "spy"}))
    monkeypatch.setattr(rs, "save_runtime_settings", recorder("save_runtime_settings", None))
    monkeypatch.setattr(training_manager, "start_training",
                        recorder("start_training", {"success": False, "error": "spy"}))
    monkeypatch.setattr(training_manager, "stop_training",
                        recorder("stop_training", {"success": False, "error": "spy"}))
    monkeypatch.setattr(training_manager, "get_checkpoints", recorder("get_checkpoints", []))
    from src.api.services import server_lists

    monkeypatch.setattr(server_lists, "list_checkpoints",
                        recorder("list_checkpoints", server_lists.Listing(())))
    monkeypatch.setattr(training_manager, "latest_snapshot", recorder("latest_snapshot", None))
    # A training job runs (a read, not a service: Stop's confirmation is
    # asked for the job running now).
    monkeypatch.setattr(training_manager, "get_status",
                        lambda: {"status": "running", "job_id": "spy-job"})
    monkeypatch.setattr(training_manager, "preview_continuation",
                        recorder("preview_continuation", {"ok": False, "reason": "spy"}))
    monkeypatch.setattr(tc, "TRAINING_CONFIG_FILE", tmp_path / "training.json")
    monkeypatch.setattr(tc.gr, "Warning",
                        lambda message, **k: reached.append(f"warning: {message}"))
    return reached


def _dashboard_calls():
    from src.webui.components import diagnosis_panel as dp
    from src.webui.components import runtime_settings as rs
    from src.webui.components import training_console as tc

    path = tc.CONTINUE_FROM_PATH
    # The Training page's handlers take the session's state (its run-bar
    # workspace, the drawer, the parent list): a fresh API session has no
    # dataset selected, so Start, Continue…, Check checkpoint, Continue
    # Training and the parent list stop before their service when allowed (no
    # service, `None`), and say so; refused, they say "Refused" as the others
    # do. A typed path only locates a checkpoint the session's run-bar
    # workspace lists (D8, 2026-10-19), and a State is the session's, never
    # what a client posts, so a direct call opens no typed file at all
    # (test_training_console_resume.py checks a listed one is previewed).
    return [
        (dp._on_change_model, [], None),
        (rs._on_restart, [], "restart"),
        (rs._on_apply, ["cuda_async", False], "save_runtime_settings"),
        (tc._on_start, None, None),
        (tc._on_stop, [{}], "stop_training"),
        (tc._on_stop_request, [{}], None),                 # asks, reading nothing
        (tc._on_continue_open, [{}, {}], None),
        (tc._refresh_checkpoints, [{}, {}, None], None),
        (tc._on_check_continue, [path, None, "/srv/x.pt", "cpu", None, {"name": "ws"}, {},
                                 {}], None),
        (tc._on_resume, [{}, "cpu", path, None, "/srv/x.pt", None, {}, {}], None),
    ]


@pytest.mark.parametrize("mode, base_url, headers", [
    (None, OPERATOR_BASE_URL, {}),
    ("M1", "http://192.0.2.10", {}),
    ("M1", OPERATOR_BASE_URL, {"X-Forwarded-For": "203.0.113.9"}),
    ("M1", OPERATOR_BASE_URL, {"Origin": "https://evil.example"}),
    ("M1", OPERATOR_BASE_URL, {"Sec-Fetch-Site": "cross-site"}),
], ids=["mode-unset", "lan", "forwarded", "foreign-origin", "cross-site"])
def test_a_direct_call_to_a_gated_event_meets_the_same_refusal(
        dashboard_app, spies, monkeypatch, tmp_path, mode, base_url, headers):
    """Every gated event, called through Gradio's HTTP API: refused, its
    outputs say why, and no service it would reach is touched."""
    if mode is not None:
        monkeypatch.setenv("SHEPHERD_DEPLOYMENT_MODE", mode)
    with _dashboard_client(dashboard_app, base_url, headers=headers) as client:
        dashboard = _Dashboard(client, dashboard_app)
        for fn, data, _service in _dashboard_calls():
            outputs = dashboard.call(fn, data)
            said = json.dumps(outputs, ensure_ascii=False)
            if fn.__name__ == "_refresh_checkpoints":
                # A list cannot hold a sentence: it is emptied, and the
                # refusal is the note beside it.
                assert {"__type__": "update", "choices": [], "value": None} in outputs
            assert "Refused: " in said, fn.__name__
    assert spies == [], "a refused event reached a service"
    assert not (tmp_path / "training.json").exists()


def test_a_direct_call_from_an_operator_on_loopback_reaches_each_service(
        dashboard_app, spies, monkeypatch, tmp_path):
    operator_mode(monkeypatch)
    with _dashboard_client(dashboard_app) as client:
        dashboard = _Dashboard(client, dashboard_app)
        for fn, data, service in _dashboard_calls():
            before = list(spies)
            outputs = dashboard.call(fn, data)
            assert "Refused: " not in json.dumps(outputs, ensure_ascii=False), fn.__name__
            if service is not None:
                reached = [name for name in spies[len(before):]
                           if not name.startswith("warning: ")]
                assert reached == [service], fn.__name__
            else:
                assert spies[len(before):] == [], fn.__name__


# --------------------------------------------------------------------------- CORS and uploads
@pytest.mark.parametrize("base_url", [OPERATOR_BASE_URL, "http://192.0.2.10:8264"],
                         ids=["loopback-host", "lan-host"])
def test_no_cross_origin_browser_access_is_granted(dashboard_app, base_url, monkeypatch):
    """No Access-Control-Allow-Origin for another origin, on /api, on the
    dashboard's config and on `file=`, whatever Host; preflights refused. The
    request itself still reaches the server, which is why D8, not CORS, guards
    the operator controls."""
    foreign = {"Origin": "https://evil.example"}
    preflight_headers = {**foreign, "Access-Control-Request-Method": "POST",
                         "Access-Control-Request-Headers": "content-type"}
    with _app_client(base_url) as api:
        response = api.get("/api/v1/workspaces", headers=foreign)
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers
        preflight = api.options("/api/v1/pipeline/reload", headers=preflight_headers)
        assert preflight.status_code == 400
        assert "access-control-allow-origin" not in preflight.headers
    with _dashboard_client(dashboard_app, base_url) as client:
        # A real file= download: a file in Gradio's own temporary directory,
        # where a download (Diagnosis' exports) is placed. (The Training
        # page's Export CSV is saved by the browser from a server function
        # since M5-2, with no file on the server.)
        import gradio.utils

        folder = Path(gradio.utils.get_upload_folder())
        folder.mkdir(parents=True, exist_ok=True)
        import uuid

        exported = folder / f"cors-probe-{uuid.uuid4().hex}.csv"
        exported.write_text("epoch,split,total\n1,train,1.5\n")
        path = f"/ui/gradio_api/file={exported}"
        for url in ("/ui/config", path):
            response = client.get(url, headers=foreign)
            assert response.status_code == 200, url
            assert "access-control-allow-origin" not in response.headers, url
            assert "access-control-allow-credentials" not in response.headers, url
        # The upload route is refused in front of CORS (403), preflight or not.
        for url, refused in (("/ui/gradio_api/queue/join", 400), ("/ui/gradio_api/upload", 403)):
            preflight = client.options(url, headers=preflight_headers)
            assert preflight.status_code == refused, url
            assert "access-control-allow-origin" not in preflight.headers, url
        same = client.get("/ui/config", headers={"Origin": base_url})
        assert same.status_code == 200


def test_the_parent_cors_middleware_is_the_class_gradio_defers_to():
    """Gradio stands down only for `CORSMiddleware` itself
    (`middleware.cls is CORSMiddleware`); a subclass leaves Gradio reflecting
    any origin on a LAN Host."""
    from fastapi.middleware.cors import CORSMiddleware

    from src.api.main import app

    (cors,) = [m for m in app.user_middleware if issubclass(m.cls, CORSMiddleware)]
    assert cors.cls is CORSMiddleware
    assert cors.kwargs == {"allow_origins": []}


@pytest.mark.parametrize("headers", [{"Origin": "https://evil.example"}, {}],
                         ids=["cross-origin", "same-origin"])
def test_the_upload_route_is_refused_whoever_posts(dashboard_app, headers):
    """The dashboard has no upload control, and another site's page can post
    a multipart body with no preflight, up to 1000 files a request: the route
    and its progress stream are refused in front of the mount, whatever the
    origin (`test_dashboard_mount.py` checks nothing is written). The per-file
    limit stays as a second line of defence."""
    from src.api.middleware.gradio_routes import REFUSALS
    from src.webui.app import MAX_UPLOAD_SIZE, mount_options

    assert mount_options()["max_file_size"] == MAX_UPLOAD_SIZE == "1mb"
    files = [("files", (f"f{i}.txt", b"x" * 1024, "text/plain")) for i in range(20)]
    with _dashboard_client(dashboard_app, headers=headers) as client:
        uploaded = client.post("/ui/gradio_api/upload", files=files)
        progress = client.get("/ui/gradio_api/upload_progress", params={"upload_id": "x"})
    assert uploaded.status_code == 403 and uploaded.text == REFUSALS["upload"]
    assert progress.status_code == 403
