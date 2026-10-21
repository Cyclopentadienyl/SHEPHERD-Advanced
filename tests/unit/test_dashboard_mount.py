"""
The dashboard as the service serves it: `src.api.main.app` on Gradio 6.30.
===========================================================================
From Gradio 6, a mounted app takes its theme, CSS, head and JS from
`gr.mount_gradio_app`, and silently drops those given to `gr.Blocks`; and
6.30 adds routes the dashboard does not use. This drives the real application
through Starlette's TestClient and checks what a browser and a script see:

- /ui is served with the Soft theme and the IBM Plex @font-face CSS, run
  history off and analytics off;
- the run-history, monitoring and MCP routes are absent or refused, with
  ``GRADIO_RUN_HISTORY`` and ``GRADIO_MCP_SERVER`` set to turn them on;
- the route guard: ``process_recording`` and ``upload`` are refused without
  their body being read (no temporary file), every URL form of ``file=`` is
  refused without any fetch, each refusal is in the request log, and a real
  download (a file in Gradio's own temporary directory, where the dashboard's
  downloads are placed) still downloads with GET, HEAD and Range. (The Training
  page's Export CSV is saved by the browser from a server function since M5-2,
  so it puts no file there.)

**In a fresh interpreter.** `src.api.main` mounts the dashboard at import, and
`src.webui` imports the API package back (its components use the API's
services): when a test process happens to import `src.webui` first, the
mount fails at import and the app has no /ui. The service always imports
`src.api.main` first (``uvicorn src.api.main:app``), and so does this probe.
Gradio's URL fetch is replaced by a counter, so nothing here touches the network.

Module: tests/unit/test_dashboard_mount.py
"""
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

REPO_ROOT = Path(__file__).resolve().parents[2]

URL_FORMS = (
    "/ui/gradio_api/file=http://example.com/evil.png",
    "/ui/gradio_api/file=https://example.com/evil.txt",
    "/ui/gradio_api/file=HTTP://example.com/evil.json",
    "/ui/gradio_api/file=hTtPs://example.com/a",
    "/ui/gradio_api/file=%68ttp://example.com/a.png",
    "/ui/gradio_api/file=http%3A%2F%2Fexample.com%2Fa.png",
    "/ui/gradio_api/file=https://169.254.169.254/latest/meta-data/",
    "/ui/gradio_api/file=http://example.com/a.png?x=1",
    "/ui/gradio_api/file=//example.com/a.png",
    "/ui/gradio_api/file/http://example.com/a.png",
    "/ui/gradio_api/file/https%3A//example.com/a.png",
)

PROBE = r'''
import json, logging, os, pathlib, re, sys

import src.api.main as main          # the service's own import order
import gradio.routes as gradio_routes
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

URL_FORMS = json.loads(sys.argv[1])
result = {"errors": {}}

fetched = []


async def fake_fetch(url, request):
    """Gradio's server-side URL fetch, counted and answered locally."""
    fetched.append(url)
    return PlainTextResponse("<script>alert(1)</script>", media_type="text/html")


gradio_routes.secure_url_stream_response = fake_fetch

log_lines = []


class Collect(logging.Handler):
    def emit(self, record):
        log_lines.append(record.getMessage())


logging.getLogger("src.api.main").addHandler(Collect())

receive_calls = []


async def counted(scope, receive, send):
    """The whole service, with each request's receive() calls counted."""
    calls = [0]

    async def counting_receive():
        calls[0] += 1
        return await receive()

    receive_calls.append(calls)
    await main.app(scope, counting_receive, send)


def files(path):
    return sorted(str(p) for p in pathlib.Path(path).rglob("*") if p.is_file())


with TestClient(counted) as client:
    try:
        r = client.get("/ui/config")
        config = r.json() if r.status_code == 200 else {}
        theme_css = client.get("/ui/theme.css").text
        primary = re.search(r"--primary-500:\s*([^;]+);", theme_css)
        font = re.search(r"--font:\s*([^;]+);", theme_css)
        result["config"] = {
            "status": r.status_code, "version": config.get("version"),
            "theme": config.get("theme"), "css": config.get("css") or "",
            "run_history": config.get("run_history"),
            "analytics_enabled": config.get("analytics_enabled"),
            "mcp_server": config.get("mcp_server"),
            "primary_500": primary.group(1).strip() if primary else None,
            "font": font.group(1).strip() if font else None,
        }
    except Exception as exc:
        result["errors"]["config"] = repr(exc)

    result["routes"] = {}
    for method, path in (("GET", "/ui/gradio_api/runs"),
                         ("POST", "/ui/gradio_api/run-history/connect"),
                         ("GET", "/ui/monitoring"),
                         ("GET", "/ui/monitoring/summary"),
                         ("GET", "/ui/gradio_api/mcp/sse"),
                         ("GET", "/ui/gradio_api/mcp/schema"),
                         ("POST", "/ui/gradio_api/mcp/messages/")):
        result["routes"][f"{method} {path}"] = client.request(method, path).status_code

    result["file_url"] = {}
    for path in URL_FORMS:
        for method in ("GET", "HEAD"):
            r = client.request(method, path)
            result["file_url"][f"{method} {path}"] = {
                "status": r.status_code, "body": r.text,
                "connection": r.headers.get("connection"),
                "receive_calls": receive_calls[-1][0],
            }
    result["url_fetches"] = list(fetched)

    try:
        tmp_dirs = [os.environ["TMPDIR"], os.environ["GRADIO_TEMP_DIR"]]
        before = [files(d) for d in tmp_dirs]
        boundary = "shepherdboundary"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; "
                f"filename=\"r.mp4\"\r\nContent-Type: video/mp4\r\n\r\n").encode()
        body += b"\x00" * 200_000 + f"\r\n--{boundary}--\r\n".encode()
        r = client.post("/ui/gradio_api/process_recording", content=body,
                        headers={"content-type": f"multipart/form-data; boundary={boundary}"})
        result["process_recording"] = {
            "status": r.status_code, "body": r.text,
            "connection": r.headers.get("connection"),
            "receive_calls": receive_calls[-1][0],
            "new_files": [sorted(set(files(d)) - set(b)) for d, b in zip(tmp_dirs, before)],
        }
    except Exception as exc:
        result["errors"]["process_recording"] = repr(exc)

    try:
        before = [files(d) for d in tmp_dirs]
        r = client.post("/ui/gradio_api/upload",
                        files=[("files", (f"f{i}.html", b"<script>1</script>" * 1000,
                                          "text/html")) for i in range(30)],
                        headers={"Origin": "https://evil.example"})
        result["upload"] = {
            "status": r.status_code, "body": r.text,
            "connection": r.headers.get("connection"),
            "receive_calls": receive_calls[-1][0],
            "new_files": [sorted(set(files(d)) - set(b)) for d, b in zip(tmp_dirs, before)],
        }
    except Exception as exc:
        result["errors"]["upload"] = repr(exc)

    try:
        temp_dir = pathlib.Path(os.environ["GRADIO_TEMP_DIR"]) / "export-probe"
        temp_dir.mkdir(parents=True, exist_ok=True)
        exported = temp_dir / "metrics.csv"
        exported.write_text("epoch,split,total,val_loss\n1,train,1.5,\n1,validation,,1.7\n")
        file_data = {"path": str(exported)}
        path = f"/ui/gradio_api/file={exported}"
        get, head = client.get(path), client.head(path)
        ranged = client.get(path, headers={"Range": "bytes=0-4"})
        result["export"] = {
            "in_gradio_temp_dir": file_data["path"].startswith(os.environ["GRADIO_TEMP_DIR"]),
            "get": [get.status_code, get.text],
            "head": [head.status_code, head.headers.get("content-length")],
            "range": [ranged.status_code, ranged.text],
        }
    except Exception as exc:
        result["errors"]["export"] = repr(exc)

result["log_lines"] = log_lines
print("PROBE_RESULT " + json.dumps(result))
'''


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """What the probe saw, run once for this module in a fresh interpreter."""
    root = tmp_path_factory.mktemp("dashboard")
    env = {k: v for k, v in os.environ.items()
           if k not in ("SHEPHERD_KG_PATH", "SHEPHERD_CHECKPOINT_PATH")}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    for name in ("TMPDIR", "GRADIO_TEMP_DIR"):
        (root / name).mkdir()
        env[name] = str(root / name)
    # The switches hold whatever the environment says: these variables would
    # turn run history on for a mount that left it to Gradio, and MCP too
    # wherever Gradio's `mcp` extra is installed (the lock does not have it;
    # `test_every_mount_option_is_a_parameter_of_this_gradios_mount` pins the
    # switch itself).
    env["GRADIO_RUN_HISTORY"] = "True"
    env["GRADIO_MCP_SERVER"] = "True"
    done = subprocess.run([sys.executable, "-c", PROBE, json.dumps(URL_FORMS)],
                          cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("PROBE_RESULT ")]
    assert done.returncode == 0 and lines, done.stderr[-4000:]
    return json.loads(lines[-1][len("PROBE_RESULT "):])


def test_ui_is_served_with_the_soft_theme_and_the_plex_fonts(served):
    assert not served["errors"].get("config"), served["errors"]
    config = served["config"]
    assert config["status"] == 200
    assert config["version"] == gr.__version__
    assert config["theme"] == "soft"
    assert config["primary_500"] == "#6366f1"          # Soft's indigo, not Default's orange
    assert config["font"] == "IBM Plex Sans"
    assert "@font-face" in config["css"]
    assert "/static/fonts/ibm-plex-sans-400.woff2" in config["css"]


def test_run_history_and_analytics_are_off(served):
    assert served["config"]["run_history"] is False
    assert served["config"]["analytics_enabled"] is False
    assert served["config"]["mcp_server"] is False


def test_run_history_monitoring_and_mcp_routes_are_absent_or_refused(served):
    routes = served["routes"]
    assert routes["GET /ui/gradio_api/runs"] == 404
    assert routes["POST /ui/gradio_api/run-history/connect"] == 404
    # Left at None, /monitoring/summary would serve the queue's analytics.
    assert routes["GET /ui/monitoring"] == 403
    assert routes["GET /ui/monitoring/summary"] == 403
    for path in ("GET /ui/gradio_api/mcp/sse", "GET /ui/gradio_api/mcp/schema",
                 "POST /ui/gradio_api/mcp/messages/"):
        assert routes[path] == 404, path


def test_every_url_form_of_file_is_refused_without_a_fetch(served):
    from src.api.middleware.gradio_routes import REFUSALS

    assert served["url_fetches"] == []
    for path in URL_FORMS:
        for method in ("GET", "HEAD"):
            seen = served["file_url"][f"{method} {path}"]
            assert seen["status"] == 403, (method, path, seen)
            assert seen["connection"] == "close", (method, path)
            assert seen["receive_calls"] == 0, (method, path)
            if method == "GET":
                assert seen["body"] == REFUSALS["file-url"], (path, seen["body"])


def test_process_recording_is_refused_before_its_body_is_read(served):
    from src.api.middleware.gradio_routes import REFUSALS

    assert not served["errors"].get("process_recording"), served["errors"]
    seen = served["process_recording"]
    assert seen["status"] == 403
    assert seen["body"] == REFUSALS["process_recording"]
    assert seen["connection"] == "close"
    assert seen["receive_calls"] == 0
    assert seen["new_files"] == [[], []], "a temporary file was written"


def test_an_upload_is_refused_before_its_body_is_read(served):
    """Another site's page can post files to Gradio's upload route with no
    preflight; the dashboard takes none, so nothing is written (M5-1)."""
    from src.api.middleware.gradio_routes import REFUSALS

    assert not served["errors"].get("upload"), served["errors"]
    seen = served["upload"]
    assert seen["status"] == 403
    assert seen["body"] == REFUSALS["upload"]
    assert seen["connection"] == "close"
    assert seen["receive_calls"] == 0
    assert seen["new_files"] == [[], []], "a temporary file was written"


def test_each_refusal_is_in_the_request_log(served):
    logged = served["log_lines"]
    assert any(line.startswith("GET /ui/gradio_api/file=") and " - 403 - " in line
               for line in logged), logged[-20:]
    assert any(line.startswith("POST /ui/gradio_api/process_recording - 403 - ")
               for line in logged), logged[-20:]
    assert any(line.startswith("POST /ui/gradio_api/upload - 403 - ")
               for line in logged), logged[-20:]


def test_a_real_export_still_downloads_with_get_head_and_range(served):
    assert not served["errors"].get("export"), served["errors"]
    export = served["export"]
    assert export["in_gradio_temp_dir"] is True
    status, text = export["get"]
    assert status == 200
    assert text.splitlines()[0] == "epoch,split,total,val_loss"
    assert export["head"] == [200, str(len(text.encode()))]
    assert export["range"] == [206, "epoch"]


def test_every_mount_option_is_a_parameter_of_this_gradios_mount():
    """A keyword `gr.mount_gradio_app` does not take raises at mount, which
    `src.api.main` logs as a warning and serves no dashboard."""
    from src.webui.app import mount_options

    parameters = inspect.signature(gr.mount_gradio_app).parameters
    options = mount_options()
    assert set(options) <= set(parameters), set(options) - set(parameters)
    assert {key: options[key] for key in ("run_history", "enable_monitoring",
                                          "mcp_server", "ssr_mode")} == {
        "run_history": False, "enable_monitoring": False,
        "mcp_server": False, "ssr_mode": False}
