"""
Stop inference on a real server: the admission cut-off is the stop's arrival
============================================================================
Checkpoint A's verification of the ops round's I4 found, on a real uvicorn
server, diagnoses sent after a stop answered 200 by the model being stopped
(39 of 39 in several runs) and the stop reporting "0 accepted request(s)
finished first": `/diagnose` ran its lease and `pipeline.run` on the event
loop, so a stop arriving during a diagnosis reached `stop_serving` only after
the loop had served the diagnoses queued behind it. The unit tests passed
because they held the lease in a thread of their own, which no diagnosis does.

Here the real application (`src.api.main.app`, its lifespan, middleware and
routes) runs under uvicorn on a loopback port of its own, in this process, and
every request is real HTTP: a diagnosis runs in a worker thread under the one
run lock inside its lease, the stop closes admission as it arrives and drains
in a thread, and a reload builds in a worker thread (BACKLOG 28). The served
model is a real disease child on the CPU (`trained_disease_child`), its
`pipeline.run` held or slowed by the test; a reload's build is the lifecycle
tests' stub, held by the test.

Module: tests/unit/test_serving_real_server.py
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional, Tuple

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

import src.api.main as api  # noqa: E402
from tests.fixtures.operator_access import operator_mode  # noqa: E402
from tests.fixtures.serving_lifecycle import (  # noqa: E402
    StubBuilds,
    checkpoint,
    served_workspace,
)

#: The words a diagnosis is refused with while a stop drains
#: (`AppState.not_serving_reason`).
STOPPING = "Inference is stopping"
DIAGNOSIS = {"phenotypes": ["HP:0001250", "HP:0001263"], "include_explanations": True}


#: The header a test names its request by, for `arrivals`.
REQUEST_ID = "X-Test-Request"


def arrival_recorder(app: Any, arrivals: Dict[str, float]) -> Any:
    """`app` behind a pure ASGI wrapper that records, by this process's
    monotonic clock, when each HTTP request naming itself in `REQUEST_ID`
    reaches the application: **the server's arrival order**, which a
    client's own send times are not (the client threads share this
    process's interpreter with the server, so a request stamped first can
    reach the server hundreds of milliseconds after one stamped later)."""
    key = REQUEST_ID.lower().encode("latin-1")

    async def recorded(scope, receive, send):
        if scope["type"] == "http":
            named = dict(scope["headers"]).get(key)
            if named is not None:
                arrivals[named.decode("latin-1")] = time.monotonic()
        await app(scope, receive, send)

    return recorded


@contextmanager
def real_server(arrivals: Optional[Dict[str, float]] = None) -> Iterator[str]:
    """The application under uvicorn on a free loopback port, in a thread of
    this process (its lifespan included); yields its base URL. With
    `arrivals`, each named request's arrival is recorded there
    (`arrival_recorder`)."""
    app = api.app if arrivals is None else arrival_recorder(api.app, arrivals)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        assert time.monotonic() < deadline and thread.is_alive(), "the server did not start"
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(60)


def call(base: str, method: str, path: str, body: Optional[Dict[str, Any]] = None,
         timeout: float = 120.0, name: Optional[str] = None) -> Tuple[int, Any, float, float]:
    """One real HTTP request, named `name` in `REQUEST_ID` when given:
    ``(status, JSON body, sent, answered)``, the two times by this process's
    monotonic clock."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if name is not None:
        headers[REQUEST_ID] = name
    request = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    sent = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return answer.status, json.loads(answer.read()), sent, time.monotonic()
    except urllib.error.HTTPError as refused:
        return refused.code, json.loads(refused.read()), sent, time.monotonic()


def in_thread(fn) -> SimpleNamespace:
    box = SimpleNamespace(result=None, error=None)

    def run() -> None:
        try:
            box.result = fn()
        except BaseException as exc:  # noqa: BLE001 - handed to the test
            box.error = exc

    box.thread = threading.Thread(target=run, daemon=True)
    box.thread.start()
    return box


def _wait(predicate, seconds: float = 20.0) -> None:
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.fixture(scope="module")
def served_child(trained_disease_child):
    """A real disease child's pipeline on the CPU, built once for this module
    (`trained_children.serve`); each test wraps its `run` and puts it back."""
    from tests.fixtures.trained_children import serve

    pipeline = serve(trained_disease_child)
    return pipeline, trained_disease_child


@pytest.fixture
def service(monkeypatch, served_child):
    """Nothing served but what the test publishes, operator mode on (D8
    allows loopback), and the published pipeline's `run` wrapped by the test
    (`gate` and `pause`), restored afterwards."""
    for name in ("pipeline", "kg", "_current_data_dir", "_current_checkpoint_path",
                 "serving_device", "served_sha256", "not_serving", "transition"):
        monkeypatch.setattr(api.app_state, name, None)
    for name in ("SHEPHERD_KG_PATH", "SHEPHERD_CHECKPOINT_PATH", "SHEPHERD_DATA_DIR",
                 "SHEPHERD_DEVICE", "SHEPHERD_BIND_HOST"):
        monkeypatch.delenv(name, raising=False)
    operator_mode(monkeypatch)
    pipeline, trained = served_child
    real_run = pipeline.run
    control = SimpleNamespace(entered=threading.Semaphore(0), gate=None, pause=0.0, runs=0,
                              concurrent=0, most=0, counted=threading.Lock(), ended=[])

    def run(*args, **kwargs):
        with control.counted:
            control.runs += 1
            control.concurrent += 1
            control.most = max(control.most, control.concurrent)
        try:
            control.entered.release()
            if control.gate is not None:
                assert control.gate.wait(60), "the test never let the diagnosis go on"
            if control.pause:
                time.sleep(control.pause)
            return real_run(*args, **kwargs)
        finally:
            with control.counted:
                control.concurrent -= 1
                control.ended.append(time.monotonic())

    monkeypatch.setattr(pipeline, "run", run, raising=False)

    def publish() -> None:
        api.publish_pipeline(api.PipelineBundle(
            kg=pipeline.kg, pipeline=pipeline, config=pipeline.get_pipeline_config(),
            data_dir=str(trained["data_dir"]), checkpoint_path=str(trained["disease"])))

    control.publish = publish
    yield control
    if control.gate is not None:
        control.gate.set()


def test_a_diagnosis_sent_while_a_stop_drains_is_refused_and_the_stop_waits_for_the_first(
        service):
    """A diagnosis runs; a stop arrives; a second diagnosis, sent while the
    stop is pending, is refused at once (503) with the stopping reason. The
    status answers meanwhile (the loop is free), and the stop answers only
    after the first diagnosis has, saying one accepted request finished
    first, the model released."""
    service.gate = threading.Event()
    service.publish()
    with real_server() as base:
        first = in_thread(lambda: call(base, "POST", "/api/v1/diagnose", DIAGNOSIS))
        assert service.entered.acquire(timeout=30), "the first diagnosis never ran"
        stop = in_thread(lambda: call(base, "POST", "/api/v1/pipeline/stop", {}))

        def stopping() -> bool:
            code, status, sent, answered = call(base, "GET", "/api/v1/pipeline/status",
                                                timeout=5)
            assert code == 200 and answered - sent < 5
            return status["serving"] == "stopping"

        _wait(stopping)
        code, said, sent, answered = call(base, "POST", "/api/v1/diagnose",
                                          {"phenotypes": ["HP:0001263"]}, timeout=10)
        assert code == 503 and said["error"].startswith(STOPPING), said
        assert answered - sent < 5, "the refusal waited for the running diagnosis"
        assert first.thread.is_alive() and stop.thread.is_alive()
        code, health, _, _ = call(base, "GET", "/health", timeout=5)
        assert code == 200

        service.gate.set()
        first.thread.join(60)
        stop.thread.join(60)
        diagnosed, stopped = first.result, stop.result
        assert first.error is None and stop.error is None
        assert diagnosed[0] == 200 and diagnosed[1]["result_kind"] == "disease_task/v1"
        assert stopped[0] == 200 and stopped[1]["released"] is True, stopped
        assert stopped[1]["message"].startswith(
            "Inference stopped: 1 accepted request(s) finished first"), stopped[1]
        # The stop answered after the diagnosis's run had ended (one clock:
        # the server runs in this process); the two answers' arrival order is
        # the client's.
        assert stopped[3] >= service.ended[0], "the stop answered before the diagnosis ended"
        assert stopped[1]["status"]["serving"] == "not_loaded"
        assert service.runs == 1


def test_in_a_flood_no_diagnosis_arriving_after_the_stop_is_answered_by_the_stopped_model(
        service):
    """The verifier's flood: concurrent diagnoses, each run a little slowed
    so they queue for the run lock (the first held in its run until the
    stop has closed admission, so the stop has accepted requests to wait
    for, however loaded the host), a stop sent after the 21st, and the flood
    kept up until twenty more were sent after the stop had reached the
    server. **Judged by the server's arrival order** (`arrival_recorder`),
    the order the cut-off is defined by: every diagnosis that reached the
    application after the stop did is refused (503, stopping or stopped);
    the ones answered all reached it before the stop did: those that ended
    before the stop and the accepted ones the stop says finished first. The
    runs never overlapped."""
    service.gate = threading.Event()
    service.pause = 0.02
    service.publish()
    rows, lock, info, arrivals = [], threading.Lock(), {}, {}
    with real_server(arrivals) as base:
        def diagnose(name: str) -> None:
            code, said, _, _ = call(base, "POST", "/api/v1/diagnose", DIAGNOSIS, name=name)
            with lock:
                rows.append((name, code, said))

        def stop() -> None:
            info["answer"] = call(base, "POST", "/api/v1/pipeline/stop", {}, name="stop")

        threads, after_arrival = [], 0
        for i in range(400):
            if after_arrival == 20:
                break
            if "stop" in arrivals:
                after_arrival += 1
            threads.append(threading.Thread(target=diagnose, args=(f"d{i}",), daemon=True))
            threads[-1].start()
            if i == 20:
                assert service.entered.acquire(timeout=30), "the first diagnosis never ran"
                threads.append(threading.Thread(target=stop, daemon=True))
                threads[-1].start()
            time.sleep(0.003)
        _wait(lambda: call(base, "GET", "/api/v1/pipeline/status",
                           timeout=5)[1]["serving"] == "stopping")
        service.gate.set()
        for thread in threads:
            thread.join(120)

    assert after_arrival == 20, "the stop never reached the server during the flood"
    code, said, _, _ = info["answer"]
    assert code == 200 and said["released"] is True, said
    accepted = int(said["message"].split("Inference stopped: ")[1].split(" accepted")[0])
    stop_arrived = arrivals["stop"]
    assert len(rows) == len(threads) - 1 and all(name in arrivals for name, _, _ in rows)
    answered = [name for name, code, _ in rows if code == 200]
    after = [(name, code, said) for name, code, said in rows if arrivals[name] > stop_arrived]
    assert len(after) >= 20, "fewer diagnoses reached the server after the stop than were sent"
    assert all(code == 503 for _, code, _ in after), [
        (name, code, round(arrivals[name] - stop_arrived, 4)) for name, code, _ in after
        if code != 503]
    assert all("stopped" in said["error"] or said["error"].startswith(STOPPING)
               for _, _, said in after)
    assert all(arrivals[name] < stop_arrived for name in answered)
    assert 0 < accepted <= len(answered) == service.runs
    assert service.most == 1, "two diagnoses ran their pipeline at once"


def test_a_reload_in_progress_does_not_stall_health_or_the_status(monkeypatch, service,
                                                                  tmp_path):
    """A reload builds in a worker thread: while its build is held, /health
    and /pipeline/status answer at once, the status saying a load runs; then
    the reload publishes its model."""
    builds = StubBuilds().install(monkeypatch)
    builds.hold = threading.Event()
    workspace = served_workspace(tmp_path / "ws")
    model = checkpoint(workspace, "a")
    with real_server() as base:
        loading = in_thread(lambda: call(base, "POST", "/api/v1/pipeline/reload", {
            "data_dir": str(workspace), "checkpoint_path": str(model), "device": "cpu"}))
        assert builds.started.wait(30), "the reload never built"
        for path in ("/health", "/api/v1/pipeline/status"):
            code, said, sent, answered = call(base, "GET", path, timeout=5)
            assert code == 200 and answered - sent < 5, (path, answered - sent)
        assert said["loading"] is True and said["serving"] == "not_loaded"
        assert loading.thread.is_alive()
        builds.hold.set()
        loading.thread.join(60)
    code, reloaded, _, _ = loading.result
    assert code == 200 and reloaded["success"] is True, reloaded
    assert reloaded["status"]["serving"] == "serving"
