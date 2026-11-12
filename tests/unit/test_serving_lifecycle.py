"""
The inference lifecycle in the one service state (the ops round's A3a, I1-I10)
==============================================================================
`src.api.main.AppState` holds what is served, how requests are admitted and
what serving holds: CUDA serving holds the deployment's one GPU lock
(`take_gpu_lock`), decided by the resolved device, taken before any candidate
is read and kept until nothing uses it; CPU serving takes no lock and checks
no job slot. One load or stop runs at a time, and a second is refused, never
queued. A stop refuses new requests at once, waits for the accepted ones (no
timeout, nothing killed), releases the model and the lock, and only then
reports released.

CUDA is faked by the shared serving-device resolver answering as on a host
with CUDA, and the build by a stub whose served identity names the device in
effect (`tests/fixtures/serving_lifecycle.py`); the lock is a real flock under
each test's own workspaces root (`tests/conftest.py`), and another process
holding it is `tests/fixtures/job_processes`' lock holder.

Module: tests/unit/test_serving_lifecycle.py
"""
from __future__ import annotations

import os
import threading
import time
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

import src.api.main as api  # noqa: E402
import src.api.services.training_manager as tm  # noqa: E402
from src.api.routes.pipeline import PipelineStopRequest, run_stop  # noqa: E402
from src.utils.gpu_lock import SERVING_JOB_TYPE, WORKSPACES_ROOT_ENV, lock_state  # noqa: E402
from tests.fixtures.job_processes import FakeProcess, start_lock_holder  # noqa: E402
from tests.fixtures.serving_lifecycle import (  # noqa: E402
    StubBuilds,
    checkpoint,
    fake_cuda,
    in_thread,
    reload,
    served_workspace,
    sha_of,
)


@pytest.fixture
def service(monkeypatch, tmp_path):
    """Nothing served, CUDA faked, builds stubbed, and a job runner of the
    test's own whose processes are stand-ins."""
    for name in ("pipeline", "kg", "_current_data_dir", "_current_checkpoint_path"):
        monkeypatch.setattr(api.app_state, name, None)
    fake_cuda(monkeypatch)
    builds = StubBuilds().install(monkeypatch)
    runner = tm.TrainingManager()
    monkeypatch.setattr(tm, "training_manager", runner)
    workspace = served_workspace(tmp_path / "ws")
    yield builds, runner, workspace
    if runner.active_job() is not None:
        runner.end_active_job()


def _held_by() -> dict:
    state = lock_state()
    return state["holder"] if state["held_by"] else None


def _base_start(workspace) -> dict:
    """A base start on the CPU: the training device is the host's (auto is
    refused without CUDA), whatever the faked serving resolver answers."""
    return {"data_dir": str(workspace), "learning_stage": "base", "device": "cpu"}


def _wait(predicate, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


# --------------------------------------------------------------------------- the holding
def test_a_cuda_load_holds_the_gpu_lock_and_names_inference(service):
    """I2, I8: the lock is held from before the build, its description names
    inference and how to stop it (no path), the status says so, and a job
    start is refused naming this service's inference."""
    builds, runner, workspace = service
    model = checkpoint(workspace, "a")

    loaded = reload(workspace, model, "cuda")

    assert loaded.success, loaded.message
    assert builds.lock_at_build[0] is not None, "the build ran before the lock was held"
    holder = _held_by()
    assert holder["job_type"] == SERVING_JOB_TYPE and holder["pid"] == os.getpid()
    assert "stop inference first" in holder["command"] and str(workspace) not in holder["command"]
    assert loaded.status.holds_gpu_lock is True and loaded.status.serving == "serving"
    assert loaded.status.loading is False and loaded.status.not_serving_reason is None
    with patch("subprocess.Popen", side_effect=lambda *a, **k: pytest.fail("a job started")):
        refused = runner.start_training(_base_start(workspace))
    assert refused["success"] is False
    assert "this service's inference" in refused["error"]
    assert "stop inference first" in refused["error"]


def test_a_cpu_load_takes_no_lock_and_a_job_still_starts(service):
    """I2: CPU serving takes no lock and checks no job slot: it loads while
    the runner's job runs (no blanket exclusion), and a job starts beside
    it."""
    builds, runner, workspace = service
    running = FakeProcess(pid=4321)
    with patch("subprocess.Popen", return_value=running):
        assert runner.start_training(_base_start(workspace))["success"]

    loaded = reload(workspace, checkpoint(workspace, "a"), "cpu")

    assert loaded.success, loaded.message
    assert len(builds.calls) == 1 and runner.active_job() is not None
    assert builds.lock_at_build == [None] and lock_state()["held_by"] is None
    assert loaded.status.holds_gpu_lock is False and api.app_state.gpu_lock is None
    running.finish(0)
    _wait(lambda: runner.active_job() is None)
    with patch("subprocess.Popen", side_effect=lambda *a, **k: FakeProcess(pid=4322)):
        assert runner.start_training(_base_start(workspace))["success"]


def test_a_load_is_refused_while_another_process_holds_the_lock(service, tmp_path):
    """Before any candidate is read: no build, the holder named, the lock
    left with its holder, and the refusal recorded as why nothing serves."""
    builds, _, workspace = service
    holder = start_lock_holder("sleep", {}, tmp_path)
    try:
        refused = reload(workspace, checkpoint(workspace, "a"), "cuda")
        reason = api.app_state.not_serving_reason
        still_held = lock_state()["held_by"]
    finally:
        holder.kill()
        holder.wait(10)

    assert refused.success is False
    assert "the GPU is in use by a training job (lock_holder.py" in refused.message
    assert builds.calls == [], "a candidate was built under another process's lock"
    assert still_held and api.app_state.gpu_lock is None
    assert reason.startswith("No model is loaded: the last load was refused")
    assert "a training job (lock_holder.py" in reason and str(tmp_path) not in reason


def test_a_busy_gpu_refuses_a_task_reload_before_any_candidate_is_read(service, tmp_path,
                                                                       monkeypatch):
    """I2: the lock is decided right after the device, before automatic
    selection scans the run layout: a candidate there is never read."""
    import src.utils.checkpoint_io as checkpoint_io
    from src.api.routes.pipeline import PipelineReloadRequest, run_reload

    builds, _, workspace = service
    run = workspace / "runs" / "gatv2" / "r1" / "checkpoints"
    run.mkdir(parents=True)
    (run / "last.pt").write_bytes(b"a candidate")
    monkeypatch.setattr(checkpoint_io, "read_checkpoint",
                        lambda *a, **k: pytest.fail("a candidate was read under a busy lock"))
    holder = start_lock_holder("sleep", {}, tmp_path)
    try:
        refused = run_reload(PipelineReloadRequest(data_dir=str(workspace), task="disease",
                                                   device="cuda"))
    finally:
        holder.kill()
        holder.wait(10)

    assert refused.success is False and "the GPU is in use by a training job" in refused.message
    assert builds.calls == []


def test_a_load_is_refused_while_the_runner_has_an_active_job(service):
    """I3: a job the runner spawned whose process has not taken the lock yet
    is seen through the runner's slot, read after the lock is held; the lock
    is given back and nothing is built."""
    builds, runner, workspace = service
    process = FakeProcess(pid=4321)
    with patch("subprocess.Popen", return_value=process):
        assert runner.start_training(_base_start(workspace))["success"]

    refused = reload(workspace, checkpoint(workspace, "a"), "cuda")

    assert refused.success is False and "job runner" in refused.message
    assert builds.calls == []
    assert lock_state()["held_by"] is None and api.app_state.gpu_lock is None
    process.finish(0)


def test_with_the_lock_not_configured_cuda_is_refused_and_cpu_still_loads(service,
                                                                           monkeypatch):
    """I2: nothing runs on the GPU without excluding the others, in the jobs'
    words (GPU_LOCK_NOT_CONFIGURED); CPU serving is unaffected."""
    from src.config.server_settings import GPU_LOCK_NOT_CONFIGURED

    builds, _, workspace = service
    monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)

    refused = reload(workspace, checkpoint(workspace, "a"), "cuda")
    loaded = reload(workspace, checkpoint(workspace, "b"), "cpu")

    assert refused.success is False and GPU_LOCK_NOT_CONFIGURED in refused.message
    assert len(builds.calls) == 1 and builds.calls[0]["device"] == "cpu"
    assert loaded.success, loaded.message
    assert api.app_state.serving_device == "cpu"


# --------------------------------------------------------------------------- switches
def test_a_failed_first_cuda_candidate_gives_the_lock_back(service):
    builds, _, workspace = service
    builds.fail = RuntimeError("the candidate is broken")

    refused = reload(workspace, checkpoint(workspace, "a"), "cuda")

    assert refused.success is False and "the candidate is broken" in refused.message
    assert builds.lock_at_build[0] is not None
    assert lock_state()["held_by"] is None and api.app_state.gpu_lock is None
    assert refused.status.holds_gpu_lock is False


def test_cuda_to_cuda_keeps_one_gpu_lock_and_a_failed_candidate_keeps_the_model(service):
    """I5: the same `GpuLock` instance throughout (a second taker in this
    process would refuse itself); a refused candidate leaves the model in use
    and its holding as they were."""
    builds, _, workspace = service
    first, second = checkpoint(workspace, "a"), checkpoint(workspace, "b")
    assert reload(workspace, first, "cuda").success
    lock, served = api.app_state.gpu_lock, api.app_state.pipeline

    builds.fail = RuntimeError("the candidate is broken")
    refused = reload(workspace, second, "cuda")

    assert refused.success is False and "still being served" in refused.message
    assert api.app_state.pipeline is served and api.app_state.served_sha256 == sha_of(first)
    assert api.app_state.gpu_lock is lock and lock.held
    builds.fail = None
    assert reload(workspace, second, "cuda").success
    assert api.app_state.gpu_lock is lock and lock.held
    assert api.app_state.served_sha256 == sha_of(second)


def test_cpu_to_cuda_takes_the_lock_before_the_build_and_gives_it_back_on_failure(service):
    builds, _, workspace = service
    assert reload(workspace, checkpoint(workspace, "a"), "cpu").success
    served = api.app_state.pipeline

    builds.fail = RuntimeError("the candidate is broken")
    refused = reload(workspace, checkpoint(workspace, "b"), "cuda")

    assert refused.success is False
    assert builds.lock_at_build == [None, builds.lock_at_build[1]] and builds.lock_at_build[1]
    assert api.app_state.pipeline is served and api.app_state.serving_device == "cpu"
    assert lock_state()["held_by"] is None and api.app_state.gpu_lock is None


def test_cuda_to_cpu_gives_the_lock_back_only_after_the_old_lease_ends(service):
    """I5: the replaced CUDA model keeps the lock while a request it accepted
    still runs (a lease held in a thread); the lock goes when it ends."""
    _, _, workspace = service
    assert reload(workspace, checkpoint(workspace, "a"), "cuda").success
    old = api.app_state.pipeline
    leased, release = threading.Event(), threading.Event()

    def accepted_request():
        with api.app_state.lease() as pipeline:
            leased.set()
            assert release.wait(30)
            return pipeline

    request = in_thread(accepted_request)
    assert leased.wait(10)
    switched = reload(workspace, checkpoint(workspace, "b"), "cpu")
    during = (lock_state()["held_by"], api.app_state.snapshot())
    release.set()
    request.thread.join(10)

    assert switched.success, switched.message
    assert api.app_state.serving_device == "cpu"
    assert during[0] is not None, "the lock went while the old model's request ran"
    assert during[1]["holds_gpu_lock"] is True and during[1]["in_flight"] == 1
    assert request.result is old, "the accepted request ran on another pipeline"
    assert lock_state()["held_by"] is None and api.app_state.gpu_lock is None


# --------------------------------------------------------------------------- stop
def test_a_stop_refuses_new_requests_waits_for_the_accepted_one_then_releases(service):
    """I4: new requests refused at once (503 "stopping"), the accepted one
    waited for with no timeout, "released" reported only after the model and
    the lock are given back; the reason then says it was stopped."""
    from fastapi.testclient import TestClient

    _, _, workspace = service
    model = checkpoint(workspace, "a")
    assert reload(workspace, model, "cuda").success
    client = TestClient(api.app)
    leased, release = threading.Event(), threading.Event()

    def accepted_request():
        with api.app_state.lease():
            leased.set()
            assert release.wait(30)

    request = in_thread(accepted_request)
    assert leased.wait(10)
    stop = in_thread(lambda: run_stop(PipelineStopRequest()))
    _wait(lambda: api.app_state.transition == api.STOPPING)
    refused = client.post("/api/v1/diagnose", json={"phenotypes": ["HP:0001250"]})
    status = client.get("/api/v1/pipeline/status").json()
    waiting = (stop.result, lock_state()["held_by"])
    release.set()
    stop.thread.join(30)
    request.thread.join(10)

    assert refused.status_code == 503 and refused.json()["error"].startswith(
        "Inference is stopping")
    assert status["serving"] == "stopping" and status["in_flight"] == 1
    assert status["holds_gpu_lock"] is True
    assert waiting[0] is None and waiting[1] is not None, "released before the request ended"
    stopped = stop.result
    assert stopped.success is True and stopped.released is True
    assert "1 accepted request(s) finished first" in stopped.message
    assert lock_state()["held_by"] is None and api.app_state.gpu_lock is None
    assert api.app_state.pipeline is None and api.app_state.kg is None
    assert api.app_state._current_checkpoint_path is None
    assert stopped.status.serving == "not_loaded" and stopped.status.holds_gpu_lock is False
    assert stopped.status.not_serving_reason.startswith(
        "Inference was stopped by an operator at ")
    assert sha_of(model)[:12] in stopped.status.not_serving_reason


def test_a_stop_clears_the_reference_sample_cache(service, monkeypatch):
    from src.api.routes import diagnose

    _, _, workspace = service
    assert reload(workspace, checkpoint(workspace, "a"), "cpu").success
    monkeypatch.setitem(diagnose._REFERENCE_SAMPLES, ("samples.jsonl", "f" * 64), object())

    assert run_stop(PipelineStopRequest()).released

    assert diagnose._REFERENCE_SAMPLES == {}


def test_a_stop_with_nothing_loaded_is_nothing_to_stop(service):
    stopped = run_stop(PipelineStopRequest())

    assert (stopped.success, stopped.released) == (False, False)
    assert stopped.message == "Nothing to stop: no model is loaded."
    assert api.app_state.not_serving is None


def test_a_stop_bound_to_a_model_refuses_another(service):
    """The stop of the model the caller was shown: another model in use keeps
    serving, and its lock."""
    _, _, workspace = service
    model = checkpoint(workspace, "a")
    assert reload(workspace, model, "cuda").success

    refused = run_stop(PipelineStopRequest(expected_checkpoint_sha256="b" * 64))

    assert (refused.success, refused.released) == (False, False)
    assert f"the model in use is {sha_of(model)[:12]}" in refused.message
    assert api.app_state.pipeline is not None and lock_state()["held_by"] is not None
    stopped = run_stop(PipelineStopRequest(expected_checkpoint_sha256=sha_of(model)))
    assert stopped.released and lock_state()["held_by"] is None


def test_the_stop_route_stops_through_the_one_stop(service, monkeypatch):
    """`POST /api/v1/pipeline/stop` (D8 allowing, a `{}` body) is `run_stop`."""
    from fastapi.testclient import TestClient

    from tests.fixtures.operator_access import OPERATOR_BASE_URL, operator_mode

    _, _, workspace = service
    assert reload(workspace, checkpoint(workspace, "a"), "cuda").success
    operator_mode(monkeypatch)

    answer = TestClient(api.app, base_url=OPERATOR_BASE_URL).post("/api/v1/pipeline/stop",
                                                                  json={})

    assert answer.status_code == 200 and answer.json()["released"] is True
    assert answer.json()["status"]["serving"] == "not_loaded"
    assert lock_state()["held_by"] is None


# --------------------------------------------------------------------------- one at a time
def test_a_concurrent_load_or_stop_is_refused_without_waiting(service):
    """One load or stop at a time: while a load builds, another load and a
    stop are refused at once (the event loop never waits for one), and while
    a stop waits for an accepted request, a load is refused at once too."""
    builds, _, workspace = service
    builds.hold = threading.Event()
    loading = in_thread(lambda: reload(workspace, checkpoint(workspace, "a"), "cuda"))
    assert builds.started.wait(10)

    began = time.monotonic()
    second = reload(workspace, checkpoint(workspace, "b"), "cuda")
    stop = run_stop(PipelineStopRequest())
    took = time.monotonic() - began
    status = api.app_state.snapshot()
    builds.hold.set()
    loading.thread.join(30)

    assert second.success is False and "A model is being loaded" in second.message
    assert stop.success is False and "A model is being loaded" in stop.message
    assert took < 2.0, f"a refused load or stop waited {took:.1f} s"
    assert status["loading"] is True and status["serving"] == "not_loaded"
    assert loading.result.success
    assert len(builds.calls) == 1

    builds.hold = None
    leased, release = threading.Event(), threading.Event()

    def accepted_request():
        with api.app_state.lease():
            leased.set()
            assert release.wait(30)

    request = in_thread(accepted_request)
    assert leased.wait(10)
    stopping = in_thread(lambda: run_stop(PipelineStopRequest()))
    _wait(lambda: api.app_state.transition == api.STOPPING)
    began = time.monotonic()
    refused = reload(workspace, checkpoint(workspace, "b"), "cuda")
    took = time.monotonic() - began
    release.set()
    stopping.thread.join(30)
    request.thread.join(10)

    assert refused.success is False and "Inference is stopping" in refused.message
    assert "still being served" not in refused.message, "a stopping model is not served"
    assert took < 2.0
    assert stopping.result.released is True


def test_a_model_on_cuda_is_never_published_without_the_lock(service):
    """The publisher's own refusal, before anything changes: a model whose
    device in effect is CUDA is served only while this service holds the
    lock (its load took it)."""
    builds, _, workspace = service
    bundle = builds(checkpoint_path=str(checkpoint(workspace, "a")),
                    data_dir=str(workspace), device="cuda")

    with pytest.raises(RuntimeError, match="only while this service holds the GPU lock"):
        api.publish_pipeline(bundle)

    assert api.app_state.pipeline is None and api.app_state.serving_device is None


def _wrapped_out_of_memory() -> BaseException:
    """Out of memory as the build's checkpoint read raises it: refused as an
    unreadable checkpoint, `from` the `MemoryError`."""
    from src.inference.pipeline import PipelineBuildError

    try:
        raise MemoryError("cannot allocate")
    except MemoryError as cause:
        try:
            raise PipelineBuildError(
                "Checkpoint b.pt could not be read (MemoryError: cannot allocate).") from cause
        except PipelineBuildError as wrapped:
            return wrapped


@pytest.mark.parametrize("failure", [lambda: MemoryError("cannot allocate"),
                                     _wrapped_out_of_memory],
                         ids=["raised", "wrapped by the build"])
def test_an_out_of_memory_candidate_is_refused_and_the_model_in_use_keeps_serving(service,
                                                                                failure):
    """I5: out of memory refuses the candidate with "stop inference first",
    raised as it is or wrapped by the build (a checkpoint read in host memory,
    the GPU's on the GB10's unified memory); nothing is unloaded to make
    room."""
    builds, _, workspace = service
    assert reload(workspace, checkpoint(workspace, "a"), "cuda").success
    served, lock = api.app_state.pipeline, api.app_state.gpu_lock

    builds.fail = failure()
    refused = reload(workspace, checkpoint(workspace, "b"), "cuda")

    assert refused.success is False
    assert "The model in use keeps serving; stop inference first" in refused.message
    assert "Nothing was unloaded to make room." in refused.message
    assert api.app_state.pipeline is served and api.app_state.gpu_lock is lock and lock.held


# --------------------------------------------------------------------------- giving the lock back
def test_the_status_says_the_lock_is_held_until_the_flock_is_given_back(service, monkeypatch):
    """The memory is freed before the lock is given back, and the flock is
    released in the step that takes it out of the service's state: while the
    CUDA cache is emptied (a failed CPU to CUDA load giving the lock back),
    the status still says the lock is held, as the flock is."""
    builds, _, workspace = service
    assert reload(workspace, checkpoint(workspace, "a"), "cpu").success
    seen = []
    real_free = api._free_device_memory

    def recording_free(device):
        if device == "cuda":
            seen.append((api.app_state.snapshot()["holds_gpu_lock"],
                         lock_state()["held_by"] is not None))
        real_free(device)

    monkeypatch.setattr(api, "_free_device_memory", recording_free)
    builds.fail = RuntimeError("the candidate is broken")

    refused = reload(workspace, checkpoint(workspace, "b"), "cuda")

    assert refused.success is False
    assert seen == [(True, True)], "the status said no lock while the flock was held"
    assert lock_state()["held_by"] is None and refused.status.holds_gpu_lock is False


def test_a_failure_giving_the_lock_back_still_ends_the_load(service, monkeypatch):
    """A release that reports an error (the descriptor's close, after the
    unlock) is logged: the load still ends, its lifecycle is free, and the next
    load is not refused as busy."""
    from src.utils import gpu_lock

    builds, _, workspace = service
    builds.fail = RuntimeError("the candidate is broken")
    real_release = gpu_lock.GpuLock.release
    calls = []

    def release_then_fail(self):
        calls.append(self)
        real_release(self)
        if len(calls) == 1:
            raise OSError(5, "Input/output error")

    monkeypatch.setattr(gpu_lock.GpuLock, "release", release_then_fail)

    refused = reload(workspace, checkpoint(workspace, "a"), "cuda")
    builds.fail = None
    after = reload(workspace, checkpoint(workspace, "b"), "cpu")

    assert refused.success is False and len(calls) == 1
    assert lock_state()["held_by"] is None and api.app_state.gpu_lock is None
    assert after.success, after.message
    assert not api.app_state.lifecycle_busy()


def test_a_request_ending_during_a_stop_leaves_the_lock_to_the_stop(service, monkeypatch):
    """I4, "released" only once the lock is given back: a request that ends
    while a stop waits for it neither frees the memory nor gives the lock
    back; the stop does both, so the flock is free when it answers and its
    message says so. The interleaving it used to lose: the request's thread
    descheduled right after its lease ended, until the stop had cleared the
    model, then emptying the CUDA cache before its release."""
    _, _, workspace = service
    assert reload(workspace, checkpoint(workspace, "a"), "cuda").success
    request_threads, freed_in_request = [], []
    stop_answered = threading.Event()
    real_release = api.AppState._release_gpu_if_unused
    real_free = api._free_device_memory

    def release(self, **kwargs):
        if threading.current_thread() in request_threads:
            _wait(lambda: self.pipeline is None)
        return real_release(self, **kwargs)

    def free(device):
        if threading.current_thread() in request_threads:
            freed_in_request.append(device)
            stop_answered.wait(5)
        real_free(device)

    monkeypatch.setattr(api.AppState, "_release_gpu_if_unused", release)
    monkeypatch.setattr(api, "_free_device_memory", free)
    leased, end = threading.Event(), threading.Event()

    def accepted_request():
        request_threads.append(threading.current_thread())
        with api.app_state.lease():
            leased.set()
            assert end.wait(30)

    request = in_thread(accepted_request)
    assert leased.wait(10)
    stop = in_thread(lambda: run_stop(PipelineStopRequest()))
    _wait(lambda: api.app_state.transition == api.STOPPING)
    end.set()
    stop.thread.join(30)
    at_answer = lock_state()["held_by"]
    stop_answered.set()
    request.thread.join(10)

    assert stop.result.released is True
    assert at_answer is None, "the stop reported released while the flock was still held"
    assert "and the GPU lock given back" in stop.result.message
    assert freed_in_request == [], "the request's own thread freed the memory, not the stop"


def test_a_stop_after_a_switch_to_the_cpu_frees_the_cuda_model_it_drained(service,
                                                                         monkeypatch):
    """A replaced CUDA model whose request still runs when an operator stops
    the CPU model served now: the stop waits for that request, empties the
    CUDA cache for the model it drained (the lock is held), gives the lock
    back, and only then answers released, the status and the flock agreeing."""
    _, _, workspace = service
    assert reload(workspace, checkpoint(workspace, "a"), "cuda").success
    frees, stop_threads = [], []
    real_free = api._free_device_memory

    def recording_free(device):
        frees.append((threading.current_thread(), device))
        real_free(device)

    monkeypatch.setattr(api, "_free_device_memory", recording_free)
    leased, end = threading.Event(), threading.Event()

    def accepted_request():
        with api.app_state.lease():
            leased.set()
            assert end.wait(30)

    def stop_now():
        stop_threads.append(threading.current_thread())
        return run_stop(PipelineStopRequest())

    request = in_thread(accepted_request)
    assert leased.wait(10)
    assert reload(workspace, checkpoint(workspace, "b"), "cpu").success
    assert lock_state()["held_by"] is not None, "the replaced CUDA model's request still runs"
    stop = in_thread(stop_now)
    _wait(lambda: api.app_state.transition == api.STOPPING)
    end.set()
    stop.thread.join(30)
    at_answer = (lock_state()["held_by"], stop.result.status.holds_gpu_lock)
    request.thread.join(10)

    assert stop.result.released is True
    assert "and the GPU lock given back" in stop.result.message
    assert at_answer == (None, False)
    assert (stop_threads[0], "cuda") in frees, "the CUDA cache was not emptied by the stop"
