"""
A page served by Gradio, and one browser session speaking to it, for tests
==========================================================================
The ordering of a page's events (which listener runs when, what a
``gr.State`` holds when an event starts and when its answer is written) is
Gradio's, so a test of it goes through Gradio's own queue and routes:

- `served(demo)`: the page's Blocks mounted on a FastAPI app as the dashboard
  mounts it (``/ui``), served by uvicorn on a free loopback port; yields the
  base of Gradio's API.
- `ServedSession(base, demo)`: one browser session as Gradio 6.30's client
  speaks to the queue (`join`, `wait`) and to a ``queue=False`` listener such
  as a shell timer's tick (`predict`); `state(block)` reads the session's
  ``gr.State`` as the server holds it.

Moved here from the Test page's tests, so the Training page's use the same.

Module: tests/fixtures/served_page.py
"""
from __future__ import annotations

import contextlib
import json
import threading
import time
import uuid
from collections.abc import Iterator
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
def served(demo: Any) -> Iterator[str]:
    """The page's Blocks mounted on a FastAPI app as the dashboard mounts it
    (``/ui``), served by uvicorn on a free loopback port; yields the base of
    Gradio's API."""
    import gradio as gr
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI()
    gr.mount_gradio_app(app, demo, path="/ui")
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
