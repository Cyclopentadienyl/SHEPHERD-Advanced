"""The deployment probe's S1: M1's minimal served readings, on CPU.

Research comparison Stage 10b (the owner's item D; BACKLOG 64): the probe
serves D1's disease child from a service of its own, started as the launcher
starts it, asserts readiness, then times a fixed workload of diagnose
requests, kept apart as cold start, warm-up and steady state, and reads the
server process's memory from its own /proc entries. On the GB10 it runs last
in phase M with ``--m2-acceptance --device cuda``; here it runs on CPU.

Shown here:

- end to end on CPU: the probe passes, its readings have their shape, the
  cold start, the warm-up and the steady state are apart with the fixed
  counts, each reading is a value or a state with its reason (the CUDA
  allocator peak is not applicable on CPU), the facts carry no path, and only
  the probe's own directory is written;
- a service that is not ready (a checkpoint without a model spec, refused at
  startup) or serves another model fails S1 before any request: no reading of
  another pipeline, and the facts keep what was read before (the readiness
  record, the cold start) with every later reading not taken;
- with a scripted server: the split by position, every failure class, a
  request with no whole answer ending the readings, a warm-up that ends
  leaving the steady state not taken, a server that has exited, and a
  service that never answers /health;
- the server's memory and allocator variables read from a /proc tree, and the
  states when they cannot be read; the allocator names one list for the
  probe's process and the server's;
- S1 on "cuda" through a scripted service, with this process's torch.cuda
  counters made to raise: the CUDA allocator peak is a state, never read here;
- S1 is in neither required list and outside the exit status, so a failed S1
  neither leaves the acceptance incomplete nor fails the run.

Module: tests/unit/test_probe_served_readings.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _probe():
    spec = importlib.util.spec_from_file_location(
        "probe_deployment", REPO / "scripts" / "probe_deployment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """The demo workspace and D1's disease child on it, trained once for this
    file by the production trainer (phase D, one epoch each); S1 reads both
    and writes neither."""
    from src.kg.workspace import write_workspace

    probe = _probe()
    work = tmp_path_factory.mktemp("served-readings")
    sound = work / "sound_workspace"
    write_workspace(probe._demo_kg(), sound, feature_dim=probe.FEATURE_DIM,
                    samples=probe._budget())
    report = probe.Report()
    models = probe.phase_training(report, work, sound, device="cpu", epochs=1)
    assert models is not None, [p.detail for p in report.probes if p.status != "passed"]
    return sound, models.child


def _read(probe, work, sound, checkpoint):
    report = probe.Report()
    probe._served_readings(report, work, sound, checkpoint, "cpu")
    (only,) = report.probes
    assert only.probe_id == probe.SERVED_READINGS_PROBE == "S1"
    return only


def _phase_shape(phase, planned):
    assert phase["state"] == "read"
    assert (phase["planned"], phase["sent"], phase["counted"]) == (planned,) * 3
    assert phase["failures"] == {} and phase["ended"] is None and phase["complete"] is True
    assert len(phase["seconds"]) == len(phase["server_handler_seconds"]) == planned
    assert all(isinstance(s, float) and s > 0 for s in phase["seconds"])
    for summary, values in ((phase["summary"], phase["seconds"]),
                            (phase["server_handler_summary"], phase["server_handler_seconds"])):
        assert summary["state"] == "read"
        assert summary["min_seconds"] == min(values) and summary["max_seconds"] == max(values)
        assert min(values) <= summary["median_seconds"] <= max(values)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads the server's /proc")
def test_s1_reads_the_served_child_apart_by_phase_and_is_never_required(trained, tmp_path):
    probe = _probe()
    sound, checkpoint = trained
    before = probe.snapshot(sound)

    s1 = _read(probe, tmp_path, sound, checkpoint)

    assert s1.status == "passed", s1.detail
    facts = s1.facts
    assert facts["required"] is False and facts["complete"] is True
    assert facts["version"] == probe.SERVED_READINGS_VERSION
    # The subject: the child, by the digest of its bytes, as the status
    # reported it at readiness (the whole identity, checked by the probe).
    import hashlib

    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert facts["subject"]["checkpoint_sha256"] == sha
    assert facts["subject"]["served_identity"]["checkpoint"] == {"state": "verified",
                                                                 "sha256": sha}
    assert facts["subject"]["served_identity"]["scoring_mode"] == "disease_task"
    assert facts["request"] == {"fields_sent": ["phenotypes"], "api_defaults": {
        "top_k": 10, "include_explanations": True, "include_paths": True}}
    assert facts["query"]["split"] == "val" and facts["query"]["phenotypes"] >= 1
    # Readiness first, then cold start, warm-up and steady state, apart, with
    # the fixed counts. The cold start is wait_ready's, to 0.1 s.
    assert facts["readiness"] == {"state": "ready", "ready_status": 200}
    cold = facts["cold_start"]
    assert cold["state"] == "read"
    assert isinstance(cold["seconds"], float) and cold["seconds"] > 0
    assert round(cold["seconds"], 1) == cold["seconds"]
    _phase_shape(facts["warm_up"], probe.SERVED_WARM_UP_REQUESTS)
    _phase_shape(facts["steady_state"], probe.SERVED_STEADY_REQUESTS)
    # The server process's own memory at each boundary; its peak since the
    # start never below what it holds.
    for memory in (cold["memory_at_ready"], facts["warm_up"]["memory_after"],
                   facts["steady_state"]["memory_after"]):
        assert memory["state"] == "read"
        assert 0 < memory["rss_bytes"] <= memory["peak_rss_since_start_bytes"]
    assert facts["allocator"]["state"] == "read"
    assert set(facts["allocator"]) == {"state", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
                                       "SHEPHERD_ALLOC_SOURCE"}
    # A field this run cannot have is a state with its reason, never a number.
    assert facts["cuda_allocator_peak"] == {"state": "not_applicable",
                                            "reason": "the model is served on the cpu"}
    assert set(facts["definitions"]) == set(probe.SERVED_READING_DEFINITIONS)
    assert facts["not_covered"] == list(probe.SERVED_READINGS_NOT_COVERED)
    # Plain JSON with no path; the workspace untouched, the service's files
    # under the probe's own directory.
    encoded = json.dumps(facts, allow_nan=False)
    assert str(tmp_path) not in encoded and str(sound) not in encoded
    probe.assert_untouched(before, sound, "S1")
    assert (tmp_path / "served-readings" / "server.log").is_file()
    # Never required: in neither list.
    assert probe.SERVED_READINGS_PROBE not in probe.M2_DEMO_REQUIRED
    assert probe.SERVED_READINGS_PROBE not in probe.M2_REAL_REQUIRED


def _no_request(monkeypatch, probe):
    """`_served_reading`, which sends every request S1 times, made to fail the
    test if it is called."""
    def never(*args, **kwargs):
        raise AssertionError("a request was sent to a service not shown ready")

    monkeypatch.setattr(probe, "_served_reading", never)


def _not_taken_after_readiness(probe, facts, failed_at, ready_status):
    """The facts of an S1 that failed at readiness: what was read before it,
    and every reading after it a state with its reason, never a number."""
    assert facts["required"] is False and facts["complete"] is False
    assert facts["version"] == probe.SERVED_READINGS_VERSION
    readiness = facts["readiness"]
    assert (readiness["state"], readiness["failed_at"]) == ("not_ready", failed_at)
    assert readiness.get("ready_status") == ready_status and readiness["reason"]
    not_taken = {"state": "not_taken",
                 "reason": "the service was not shown ready (readiness): no request was sent"}
    assert facts["warm_up"] == {**not_taken, "planned": probe.SERVED_WARM_UP_REQUESTS,
                                "complete": False}
    assert facts["steady_state"] == {**not_taken, "planned": probe.SERVED_STEADY_REQUESTS,
                                     "complete": False}
    assert facts["subject"]["served_identity"]["state"] == "not_taken"
    assert set(facts["definitions"]) == set(probe.SERVED_READING_DEFINITIONS)
    assert facts["not_covered"] == list(probe.SERVED_READINGS_NOT_COVERED)
    json.dumps(facts, allow_nan=False)
    return not_taken


def test_s1_fails_before_any_request_when_the_service_is_not_ready(trained, tmp_path,
                                                                   monkeypatch):
    """A checkpoint without a model spec is refused at the service's startup:
    the service runs and answers /health but serves nothing, and S1 fails at
    readiness, timing no request. Its facts keep the cold start and /ready's
    answer. (The query is the workspace's, asked of the child's own view,
    since the stripped copy has no record to bind it.)"""
    probe = _probe()
    sound, checkpoint = trained
    stripped = probe._without_model_spec(checkpoint, tmp_path / "no_model_spec.pt")
    query = probe._workspace_query
    monkeypatch.setattr(probe, "_workspace_query",
                        lambda _, workspace, digests: query(checkpoint, workspace, digests))
    _no_request(monkeypatch, probe)

    s1 = _read(probe, tmp_path / "work", sound, stripped)

    assert s1.status == "failed"
    assert "answered /ready 503" in s1.detail and "serves no pipeline" in s1.detail
    not_taken = _not_taken_after_readiness(probe, s1.facts, "ready", 503)
    assert s1.facts["readiness"]["reason"] == s1.detail
    cold = s1.facts["cold_start"]
    assert cold["state"] == "read" and cold["seconds"] > 0
    assert cold["memory_at_ready"] == not_taken
    # Read while the not-ready service ran: the environment it was started with.
    assert s1.facts["allocator"]["state"] == "read"
    assert s1.facts["cuda_allocator_peak"]["state"] == "not_applicable"
    assert str(tmp_path) not in json.dumps(s1.facts)


def test_s1_fails_before_any_request_when_the_service_serves_another_model(
        trained, tmp_path, monkeypatch):
    """Readiness holds the status's served identity to the probe's own read of
    the child: a service reporting another checkpoint is never timed, and the
    facts say at which step it failed."""
    probe = _probe()
    sound, checkpoint = trained
    real = probe._read_run_checkpoint
    monkeypatch.setattr(probe, "_read_run_checkpoint",
                        lambda path: {**real(path), "sha256": "0" * 64})
    _no_request(monkeypatch, probe)

    s1 = _read(probe, tmp_path, sound, checkpoint)

    assert s1.status == "failed"
    assert "not the checkpoint this probe read" in s1.detail
    _not_taken_after_readiness(probe, s1.facts, "served_identity", 200)
    assert s1.facts["cold_start"]["state"] == "read"


# ---------------------------------------------------------- a scripted server
IDENTITY = {"scoring_mode": "disease_task", "checkpoint": {"state": "verified", "sha256": "a" * 64}}


def _answer(identity=IDENTITY, milliseconds=2.5):
    return 200, {}, json.dumps({"served_identity": identity,
                                "inference_time_ms": milliseconds}).encode()


class _Scripted:
    """A stand-in for `_ProbeServer` that answers from a script: a tuple is
    the answer, an exception is raised. Its process is this test's, whose
    /proc entries exist."""

    def __init__(self, *answers, exits_after=None):
        self.answers = list(answers)
        self.sent = []
        self.exits_after = exits_after
        self.process = types.SimpleNamespace(pid=os.getpid(), poll=self._poll)

    def _poll(self):
        if self.exits_after is not None and len(self.sent) >= self.exits_after:
            return 1
        return None

    def request(self, method, path, headers=None, body=None, timeout=120.0):
        self.sent.append((method, path, json.loads(body)))
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class _ScriptedService(_Scripted):
    """`_Scripted` as `_served_readings` starts it, in place of `_ProbeServer`:
    /health answers after `health` seconds (or raises it), /ready answers
    `ready`, the status reports IDENTITY, and the diagnose requests answer
    from the script."""

    def __init__(self, *answers, health=1.5, ready=200):
        super().__init__(*answers)
        self.health, self.ready_status, self.stopped = health, ready, False

    def start(self):
        pass

    def wait_ready(self):
        if isinstance(self.health, BaseException):
            raise self.health
        return self.health

    def request(self, method, path, headers=None, body=None, timeout=120.0):
        if path == "/ready":
            return self.ready_status, {}, b""
        return super().request(method, path, headers, body, timeout)

    def json(self, path, headers=None):
        assert path == "/api/v1/pipeline/status"
        return {"served_identity": IDENTITY}

    def stop(self):
        self.stopped = True


def _scripted_s1(probe, monkeypatch, trained, work, service, device):
    """S1 itself (`_served_readings`) against `service`, its identity check
    passing the scripted identity (the check is R6's and M5's to show)."""
    sound, checkpoint = trained
    monkeypatch.setattr(probe, "_ProbeServer", lambda name, directory, settings: service)
    monkeypatch.setattr(probe, "_assert_served_identity", lambda identity, **_: identity)
    report = probe.Report()
    probe._served_readings(report, work, sound, checkpoint, device)
    (only,) = report.probes
    assert service.stopped
    return only


def _reading(probe, server):
    return probe._served_reading(server, ["HP:0000001", "HP:0000002"], IDENTITY,
                                 {"state": "read", "seconds": 1.5})


def test_the_first_requests_are_the_warm_up_and_the_rest_the_steady_state():
    probe = _probe()
    warm, steady = probe.SERVED_WARM_UP_REQUESTS, probe.SERVED_STEADY_REQUESTS
    server = _Scripted(*[_answer(milliseconds=10.0)] * warm, *[_answer()] * steady)

    reading = _reading(probe, server)

    assert reading["complete"] is True and reading["cold_start"]["seconds"] == 1.5
    assert reading["cold_start"]["memory_at_ready"]["state"] in ("read", "not_obtainable")
    assert server.sent == [("POST", "/api/v1/diagnose",
                            {"phenotypes": ["HP:0000001", "HP:0000002"]})] * (warm + steady)
    # By position: the warm-up's answers are the first ones, the steady
    # state's the rest.
    assert reading["warm_up"]["server_handler_seconds"] == [0.01] * warm
    assert reading["steady_state"]["server_handler_seconds"] == [0.0025] * steady
    assert reading["warm_up"]["counted"] == warm and reading["steady_state"]["counted"] == steady


def test_each_failure_is_counted_by_class_and_leaves_its_phase_incomplete():
    probe = _probe()
    warm, steady = probe.SERVED_WARM_UP_REQUESTS, probe.SERVED_STEADY_REQUESTS
    other = {**IDENTITY, "checkpoint": {"state": "verified", "sha256": "b" * 64}}
    server = _Scripted(
        (500, {}, b'{"detail": "CUDA out of memory"}'), _answer(identity=other),
        (200, {}, b"<html>"), (200, {}, json.dumps({"served_identity": IDENTITY}).encode()),
        _answer(), *[_answer()] * steady)

    reading = _reading(probe, server)

    warm_up = reading["warm_up"]
    assert warm_up["failures"] == {"http_500": 1, "out_of_memory_in_body": 1,
                                   "served_identity_mismatch": 1, "not_json": 1,
                                   "no_inference_time": 1}
    assert (warm_up["sent"], warm_up["counted"], warm_up["complete"]) == (warm, 1, False)
    assert len(warm_up["seconds"]) == 1
    # An HTTP error is a finished request: the steady state still runs.
    assert reading["steady_state"]["complete"] is True
    assert reading["complete"] is False


def test_a_request_with_no_whole_answer_ends_the_readings():
    probe = _probe()
    server = _Scripted(*[_answer()] * probe.SERVED_WARM_UP_REQUESTS, _answer(), _answer(),
                       TimeoutError("timed out"))

    reading = _reading(probe, server)

    steady = reading["steady_state"]
    assert (steady["sent"], steady["counted"], steady["ended"]) == (3, 2, "no_response")
    assert steady["failures"] == {"no_response": 1} and steady["complete"] is False
    assert steady["memory_after"]["state"] == "read"
    assert len(server.sent) == probe.SERVED_WARM_UP_REQUESTS + 3


def test_a_warm_up_that_ends_leaves_the_steady_state_not_taken():
    probe = _probe()
    server = _Scripted(_answer(), ConnectionResetError("reset"))

    reading = _reading(probe, server)

    assert reading["warm_up"]["ended"] == "no_response"
    assert reading["steady_state"] == {
        "state": "not_taken", "reason": "the warm-up ended: no_response",
        "planned": probe.SERVED_STEADY_REQUESTS, "complete": False}
    assert len(server.sent) == 2 and reading["complete"] is False


def test_a_server_that_has_exited_is_sent_nothing_more():
    probe = _probe()
    server = _Scripted(_answer(), _answer(), exits_after=2)

    reading = _reading(probe, server)

    warm_up = reading["warm_up"]
    assert (warm_up["sent"], warm_up["counted"], warm_up["ended"]) == (2, 2, "server_exited")
    assert warm_up["failures"] == {"server_exited": 1}
    assert reading["steady_state"]["state"] == "not_taken"


def test_a_service_that_never_answers_health_has_no_cold_start_figure(trained, tmp_path,
                                                                     monkeypatch):
    probe = _probe()
    service = _ScriptedService(health=AssertionError(
        "the S1 service did not answer /health within 300 s; its log is under the work "
        "directory"))
    _no_request(monkeypatch, probe)

    s1 = _scripted_s1(probe, monkeypatch, trained, tmp_path, service, "cpu")

    assert s1.status == "failed" and "did not answer /health" in s1.detail
    _not_taken_after_readiness(probe, s1.facts, "health", None)
    assert s1.facts["cold_start"] == {"state": "not_obtainable",
                                      "reason": "the service never answered /health"}
    assert service.sent == []


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_s1_s_service_names_a_workspaces_root_of_its_own(trained, tmp_path, monkeypatch,
                                                         device):
    """The ops round's A3b: a CUDA start-up load holds the GPU lock its root
    names (I2) and is refused with none, so S1's service is started with a
    root of its own under its directory, an existing empty one, never the
    operator's or another probe's."""
    probe = _probe()
    sound, checkpoint = trained
    started = []
    service = _ScriptedService(health=AssertionError("not asked to serve here"))
    _no_request(monkeypatch, probe)
    monkeypatch.setattr(probe, "_ProbeServer",
                        lambda name, directory, settings: started.append(
                            (directory, dict(settings))) or service)
    monkeypatch.setattr(probe, "_assert_served_identity", lambda identity, **_: identity)
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(tmp_path / "the-operators"))

    probe._served_readings(probe.Report(), tmp_path, sound, checkpoint, device)

    ((directory, settings),) = started
    root = Path(settings["SHEPHERD_WORKSPACES_ROOT"])
    assert root == tmp_path / "served-readings" / "workspaces" and root.parent == directory
    assert root.is_dir() and not any(root.iterdir())
    assert settings["SHEPHERD_DEVICE"] == device


def test_a_phase_with_nothing_counted_has_no_summary_figure():
    probe = _probe()
    assert probe._seconds_summary([]) == {"state": "not_obtainable",
                                          "reason": "no request of this phase was counted"}
    assert probe._seconds_summary([0.3, 0.1, 0.2]) == {
        "state": "read", "median_seconds": 0.2, "min_seconds": 0.1, "max_seconds": 0.3}


# ------------------------------------------------------- the server's process
def _proc(tmp_path, pid, status=None, environ=None):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "status").write_text("Name:\tself\n")
    if status is not None or environ is not None:
        (proc / str(pid)).mkdir()
    if status is not None:
        (proc / str(pid) / "status").write_text(status)
    if environ is not None:
        (proc / str(pid) / "environ").write_bytes(environ)
    return proc


def test_the_server_s_memory_is_read_from_its_own_status(tmp_path):
    probe = _probe()
    proc = _proc(tmp_path, 4242, status="Name:\tpython\nVmHWM:\t  2048 kB\nVmRSS:\t  1024 kB\n")

    assert probe._process_memory(4242, proc) == {
        "state": "read", "rss_bytes": 1024 * 1024, "peak_rss_since_start_bytes": 2048 * 1024}


def test_memory_that_cannot_be_read_is_a_state_with_its_reason(tmp_path):
    probe = _probe()
    exited = _proc(tmp_path / "a", 4242, status="Name:\tpython\nState:\tZ (zombie)\n")
    gone = _proc(tmp_path / "b", 4242)

    assert probe._process_memory(4242, exited) == {
        "state": "not_obtainable",
        "reason": "the server's process has exited: its status has no VmRSS or VmHWM"}
    assert probe._process_memory(4242, gone) == {
        "state": "not_obtainable", "reason": "the server's process is gone"}
    assert probe._process_memory(4242, tmp_path / "no-proc") == {
        "state": "not_obtainable",
        "reason": "this platform has no /proc/<pid>/status (Linux's process counters)"}


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_a_live_process_s_memory_is_read_from_proc():
    probe = _probe()
    memory = probe._process_memory(os.getpid())
    assert memory["state"] == "read"
    assert 0 < memory["rss_bytes"] <= memory["peak_rss_since_start_bytes"]


def test_the_server_s_allocator_variables_are_read_raw_from_its_environment(tmp_path):
    probe = _probe()
    proc = _proc(tmp_path, 4242,
                 environ=b"PATH=/usr/bin\0PYTORCH_ALLOC_CONF=backend:cudaMallocAsync\0"
                         b"SHEPHERD_ALLOC_SOURCE=preset\0")

    assert probe._process_allocator(4242, proc) == {
        "state": "read", "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync",
        "PYTORCH_CUDA_ALLOC_CONF": None, "SHEPHERD_ALLOC_SOURCE": "preset"}
    assert probe._process_allocator(4243, proc)["state"] == "not_obtainable"


def test_an_exited_process_s_empty_environment_is_no_allocator_reading(tmp_path):
    """Some kernels read an exited, unreaped process's environ as empty: that
    is no environment, never one without the variables."""
    probe = _probe()
    proc = _proc(tmp_path, 4242, environ=b"")

    assert probe._process_allocator(4242, proc) == {
        "state": "not_obtainable",
        "reason": "the server's process has exited: its environment reads empty"}


def test_the_allocator_names_are_one_list_for_both_processes(tmp_path):
    probe = _probe()
    names = probe._allocator_variables()
    proc = _proc(tmp_path, 4242, environ=b"PATH=/usr/bin\0")

    assert set(names) <= set(probe._allocator_record(None))
    assert set(probe._process_allocator(4242, proc)) == {"state", *names}


def test_s1_on_cuda_never_reads_a_device_counter_in_the_probe_s_process(
        trained, tmp_path, monkeypatch):
    """S1 itself, on "cuda", through a scripted service: this process's
    torch.cuda counters, peak resets and synchronise raise if anything calls
    them. The CUDA allocator peak is the server's, and no route exposes it, so
    it is a state with its reason; every phase is still read."""
    import torch

    probe = _probe()

    def never(*args, **kwargs):
        raise AssertionError("torch.cuda was read in the probe's process")

    for name in ("max_memory_allocated", "memory_allocated", "max_memory_reserved",
                 "memory_reserved", "memory_stats", "reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, never)
    warm, steady = probe.SERVED_WARM_UP_REQUESTS, probe.SERVED_STEADY_REQUESTS
    service = _ScriptedService(*[_answer()] * (warm + steady))

    s1 = _scripted_s1(probe, monkeypatch, trained, tmp_path, service, "cuda")

    assert s1.status == "passed", s1.detail
    facts = s1.facts
    assert facts["device"] == "cuda" and facts["complete"] is True
    assert facts["readiness"] == {"state": "ready", "ready_status": 200}
    assert facts["cold_start"]["seconds"] == 1.5
    assert (facts["warm_up"]["counted"], facts["steady_state"]["counted"]) == (warm, steady)
    peak = facts["cuda_allocator_peak"]
    assert peak["state"] == "not_obtainable" and set(peak) == {"state", "reason"}
    assert "inside the server's process" in peak["reason"]
    assert probe._cuda_allocator_peak("cpu") == {"state": "not_applicable",
                                                 "reason": "the model is served on the cpu"}


def _all_required_passed(probe):
    report = probe.Report()
    for probe_id in (*probe.M2_DEMO_REQUIRED, *probe.M2_REAL_REQUIRED):
        report.probes.append(probe.Probe(probe_id, "m2", "x", status="passed"))
    return report


@pytest.mark.parametrize("status", ["failed", "error"])
def test_a_failed_s1_neither_leaves_the_acceptance_incomplete_nor_fails_the_run(status):
    """S1 is in neither required list, and the run's exit status, which the
    external closing check reads, never counts it."""
    probe = _probe()
    report = _all_required_passed(probe)
    report.probes.append(probe.Probe(probe.SERVED_READINGS_PROBE, "m2", "x", status=status))

    verdict = probe.m2_acceptance(report, True)

    assert verdict["complete"] is True
    assert "S1" in verdict["scope"] and "never required" in verdict["scope"]
    assert probe.SERVED_READINGS_PROBE in probe.INFORMATIONAL_PROBES
    assert probe.run_exit_status(report, verdict) == 0
    # Its status stays in the report's summary as it is.
    assert report.summary()[status] == 1


def test_any_other_failure_or_an_incomplete_acceptance_still_fails_the_run():
    probe = _probe()
    report = _all_required_passed(probe)
    assert probe.run_exit_status(report, probe.m2_acceptance(report, True)) == 0
    assert probe.run_exit_status(report, None) == 0

    report.probes.append(probe.Probe("Z9", "m2", "x", status="error"))
    assert probe.run_exit_status(report, None) == 1

    report = _all_required_passed(probe)
    report.probes[0].status = "failed"
    verdict = probe.m2_acceptance(report, True)
    assert verdict["complete"] is False
    assert probe.run_exit_status(report, verdict) == 1
    # An incomplete acceptance alone (a required probe not run) fails it too.
    report = _all_required_passed(probe)
    report.probes.pop(0)
    assert probe.run_exit_status(report, probe.m2_acceptance(report, True)) == 1
