"""
A page served by Gradio, and one browser session speaking to it, for tests
==========================================================================
The ordering of a page's events (which listener runs when, what a
``gr.State`` holds when an event starts and when its answer is written) is
Gradio's, so a test of it goes through Gradio's own queue and routes:

- `served(demo, **mount)`: the page's Blocks mounted on a FastAPI app as the
  dashboard mounts it (``/ui``; with the mount's options, `app.mount_options`,
  its theme, CSS and head too), served by uvicorn on a free loopback port;
  yields the base of Gradio's API.
- `ServedSession(base, demo)`: one browser session as Gradio 6.30's client
  speaks to the queue (`join`, `wait`) and to a ``queue=False`` listener such
  as a shell timer's tick (`predict`); `state(block)` reads the session's
  ``gr.State`` as the server holds it.
- `served_service(root, settings)`: the service itself in a process of its
  own (its lifespan, its API, the dashboard), for a browser test of the whole
  app on a workspace (the shared UI measurement check).

Moved here from the Test page's tests, so the Training page's use the same.

Module: tests/fixtures/served_page.py
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class ServedSession:
    """One browser session, spoken as Gradio 6.30's client speaks to the
    queue: an event joins with the values its non-State inputs had when it
    was asked (a State input is the session's, read when the event starts),
    and the answers come on the session's one event stream, which closes
    when nothing of the session is pending (closing it earlier would cancel
    them)."""

    def __init__(self, base: str, demo: Any) -> None:
        import httpx

        self.base, self.demo = base, demo
        self.session = uuid.uuid4().hex
        self.client = httpx.Client(timeout=60)
        self.done, self.order, self.cond = {}, [], threading.Condition()
        self.reader = None

    def _read(self) -> None:
        try:
            with self.client.stream("GET", f"{self.base}/queue/data",
                                    params={"session_hash": self.session}) as stream:
                for line in stream.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    msg = json.loads(line[len("data:"):])
                    if msg.get("msg") == "process_completed":
                        with self.cond:
                            self.done[msg["event_id"]] = msg
                            self.order.append(msg["event_id"])
                            self.cond.notify_all()
        except Exception:  # noqa: BLE001 - `wait` reads again while something is pending
            pass

    def join(self, fn: Any, data: list, event_data: Any = None) -> str:
        """Ask a queued listener `fn` with `data` (one value per input)."""
        answer = self.client.post(f"{self.base}/queue/join", json={
            "data": data, "fn_index": fn._id, "session_hash": self.session,
            "event_data": event_data, "trigger_id": None})
        answer.raise_for_status()
        if self.reader is None or not self.reader.is_alive():
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
        return answer.json()["event_id"]

    def wait(self, *events: str, timeout: float = 60.0) -> bool:
        """True once every event in `events` has answered (each successfully);
        False when `timeout` passed first."""
        deadline = time.monotonic() + timeout
        with self.cond:
            while not all(e in self.done for e in events):
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                if not self.reader.is_alive():
                    self.reader = threading.Thread(target=self._read, daemon=True)
                    self.reader.start()
                self.cond.wait(min(left, 0.1))
        for e in events:
            assert self.done[e].get("success"), self.done[e]
        return True

    def predict(self, fn: Any, data: list) -> list:
        """A ``queue=False`` listener's request, as the browser sends a shell
        timer's tick: answered on the request itself; returns its outputs."""
        answer = self.client.post(f"{self.base}/run/predict", json={
            "data": data, "fn_index": fn._id, "session_hash": self.session,
            "event_data": None, "trigger_id": None})
        answer.raise_for_status()
        return answer.json()["data"]

    def state(self, block: Any) -> Any:
        return self.demo._queue.server_app.state_holder[self.session][block._id]

    def set_state(self, block: Any, value: Any) -> None:
        """What another page's event wrote into a shared State (the shell's
        location, set when a page is shown)."""
        self.demo._queue.server_app.state_holder[self.session][block._id] = value

    def close(self) -> None:
        self.client.close()


@contextlib.contextmanager
def served(demo: Any, **mount: Any) -> Iterator[str]:
    """The page's Blocks mounted on a FastAPI app as the dashboard mounts it
    (``/ui``, with `mount`: what `gr.mount_gradio_app` is given besides the
    app, the Blocks and the path), served by uvicorn on a free loopback port;
    yields the base of Gradio's API (the page itself is its ``/ui/``).

    The dashboard's own files are served beside it at ``/static`` as the API
    serves them (`src/api/main.py`): its fonts (IBM Plex Sans,
    `app.DASHBOARD_CSS`), so a browser draws and measures the page in the
    type it is drawn in. Without them every width a test measured was the
    host's fallback font's (here Inter; DejaVu Sans where no Inter is
    installed, 40 px wider over the model list's tree at 1280 x 900), and a
    region's minimum fitted to one host's measure was a pixel short on
    another (the review of 6a2d52e: the model list's rows region, 616 px,
    scrolled to 617)."""
    import gradio as gr
    import uvicorn
    from fastapi import FastAPI
    from starlette.staticfiles import StaticFiles

    from src import webui

    app = FastAPI()
    app.mount("/static", StaticFiles(directory=str(Path(webui.__file__).parent / "static")),
              name="static")
    gr.mount_gradio_app(app, demo, path="/ui", **mount)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        assert time.monotonic() < deadline and thread.is_alive(), "the server did not start"
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/ui/gradio_api"
    finally:
        server.should_exit = True
        thread.join(10)


@contextlib.contextmanager
def served_service(root: Path, settings: dict[str, str]) -> Iterator[str]:
    """The service itself (``uvicorn src.api.main:app``: its lifespan, the API
    the Diagnosis page calls back, the dashboard at ``/ui``) in a process of its
    own on the loopback port it binds (``--port 0``, the port read from its
    start line: a port picked here and then given to it could be taken in
    between by another test's server, which would answer in its place), run
    from the repository; yields the dashboard's page URL once it answers.

    Its environment is this one with every ``SHEPHERD_*`` and ``GRADIO_*``
    variable removed and `settings` set, and a home, configuration, state and
    settings file of its own under `root` (never the account's); its output
    goes to ``root / "service.log"``, which the assertion that it did not start
    quotes. Stopped (terminated, then killed) on exit."""
    import re
    import subprocess
    import sys
    import urllib.request

    root = Path(root)
    for name in ("home", "config", "state", "settings"):
        (root / name).mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SHEPHERD_", "GRADIO_"))}
    env.update(HOME=str(root / "home"), XDG_CONFIG_HOME=str(root / "config"),
               XDG_STATE_HOME=str(root / "state"), SHEPHERD_STATE_DIR=str(root / "state"),
               SHEPHERD_SETTINGS_FILE=str(root / "settings" / "deployment.json"))
    env.update(settings)
    path = root / "service.log"
    log = open(path, "wb")
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "0"],
        cwd=Path(__file__).resolve().parents[2], env=env, stdout=log, stderr=subprocess.STDOUT)
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        deadline, page = time.monotonic() + 180, None
        while True:
            assert process.poll() is None, path.read_text(errors="replace")[-4000:]
            assert time.monotonic() < deadline, "the service did not answer in 180 s"
            if page is None:
                started = re.search(r"Uvicorn running on http://127\.0\.0\.1:(\d+)", path.read_text(errors="replace"))
                page = started and f"http://127.0.0.1:{started.group(1)}/ui/"
            if page is None:
                time.sleep(0.2)
                continue
            try:
                with direct.open(page, timeout=5) as answer:
                    if answer.status == 200:
                        break
            except OSError:
                time.sleep(0.2)
        yield page
    finally:
        process.terminate()
        try:
            process.wait(30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(10)
        log.close()
