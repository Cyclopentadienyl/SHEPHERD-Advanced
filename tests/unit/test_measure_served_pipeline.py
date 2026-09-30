"""
The §7.3 measurement script, checked without CUDA and without the real service.

`scripts/measure_served_pipeline.py` implements PLAN_B04_PRODUCTIONISATION §7.3.
These tests hold it to that procedure: the counters it reads, the reset it
confirms while the server is paused, the workload it sends, the service it
starts and the evidence it writes. The service is replaced by a small HTTP
server so that a whole repeat -- R0 to R4, reset and all -- runs here in a
second or two; the real run happens on the deployment host.
"""
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from scripts import measure_served_pipeline as msp

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the script reads /proc"
)


# ------------------------------------------------------------------- helpers
def _fake_proc(tmp_path, pid=4242, *, stat=None, status=None, cmdline=None, environ=None,
               meminfo=None, vmstat=None):
    root = tmp_path / "proc"
    (root / str(pid)).mkdir(parents=True)
    for name, text in (("stat", stat), ("status", status)):
        if text is not None:
            (root / str(pid) / name).write_text(text)
    if cmdline is not None:
        (root / str(pid) / "cmdline").write_bytes(b"\0".join(cmdline) + b"\0")
    if environ is not None:
        (root / str(pid) / "environ").write_bytes(
            b"\0".join(f"{k}={v}".encode() for k, v in environ.items()) + b"\0")
    if meminfo is not None:
        (root / "meminfo").write_text(meminfo)
    if vmstat is not None:
        (root / "vmstat").write_text(vmstat)
    return root


def _sleeper(allocate_mb=0):
    """A child that optionally touches memory, frees it, then waits."""
    child = subprocess.Popen(
        [sys.executable, "-c",
         f"import time; b = bytearray({allocate_mb} * 1024 * 1024); del b; "
         "print('ready', flush=True); time.sleep(60)"],
        stdout=subprocess.PIPE, text=True)
    child.stdout.readline()
    return child


@pytest.fixture
def sleeper():
    child = _sleeper(allocate_mb=64)
    yield child
    child.send_signal(signal.SIGCONT)
    child.kill()
    child.wait()


def _reset_or_skip(pid, **kwargs):
    """The reset, skipped where this host refuses writes to clear_refs."""
    try:
        return msp.reset_high_water(pid, **kwargs)
    except PermissionError:
        pytest.skip("clear_refs is not writable here")


# --------------------------------------------------------------------- /proc
def test_status_is_read_in_bytes(tmp_path):
    proc = _fake_proc(tmp_path, status="Name:\tx\nVmHWM:\t  2048 kB\nVmRSS:\t  1024 kB\n")
    assert msp.read_status(4242, proc) == {"vm_hwm_bytes": 2048 * 1024,
                                           "vm_rss_bytes": 1024 * 1024}


def test_a_zombie_has_no_counters_and_is_reported_gone(tmp_path):
    proc = _fake_proc(tmp_path, status="Name:\tx\nState:\tZ (zombie)\n",
                      stat="4242 (x) Z 1 2 3")
    with pytest.raises(ProcessLookupError):
        msp.read_status(4242, proc)
    with pytest.raises(ProcessLookupError):
        msp.process_state(4242, proc)


def test_the_state_is_read_after_the_last_parenthesis(tmp_path):
    # A command name may contain spaces and parentheses of its own.
    proc = _fake_proc(tmp_path, stat="4242 (py (a) b) T 1 4242 4242 0")
    assert msp.process_state(4242, proc) == "T"


def test_meminfo_gives_system_and_swap_in_use(tmp_path):
    proc = _fake_proc(tmp_path, meminfo=textwrap.dedent("""\
        MemTotal:       1000 kB
        MemFree:         100 kB
        MemAvailable:    600 kB
        SwapTotal:       500 kB
        SwapFree:        450 kB
        """))
    assert msp.read_meminfo(proc) == {"system_in_use_bytes": 400 * 1024,
                                      "swap_in_use_bytes": 50 * 1024}


def test_the_oom_kill_counter_is_read_where_it_exists(tmp_path):
    proc = _fake_proc(tmp_path, vmstat="pgfault 9\noom_kill 3\n")
    assert msp.read_oom_kills(proc) == 3
    assert msp.read_oom_kills(tmp_path / "nowhere") is None


def test_the_measured_pid_must_be_uvicorn_serving_the_app(tmp_path):
    server = _fake_proc(tmp_path, cmdline=[b"/v/python", b"-m", b"uvicorn",
                                           b"src.api.main:app", b"--port", b"1"])
    assert msp.is_uvicorn_server(4242, server)
    launcher = _fake_proc(tmp_path / "l", cmdline=[b"/v/python", b"scripts/launch/shep_launch.py",
                                                   b"--no-browser"])
    assert not msp.is_uvicorn_server(4242, launcher)


def test_the_server_environment_is_read_raw(tmp_path):
    proc = _fake_proc(tmp_path, environ={"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync",
                                         "A": "x=y"})
    assert msp.read_environ(4242, proc) == {"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync",
                                            "A": "x=y"}


# ------------------------------------------------------------------ the reset
def test_the_reset_is_confirmed_with_the_server_paused_and_it_is_resumed(sleeper):
    before = msp.read_status(sleeper.pid)   # the 64 MB the child touched and freed
    outcome = _reset_or_skip(sleeper.pid)
    assert outcome["status"] == "confirmed"
    assert outcome["vm_hwm_bytes"] == outcome["vm_rss_bytes"] < before["vm_hwm_bytes"]
    time.sleep(0.05)
    assert msp.process_state(sleeper.pid) != "T"


def test_a_stop_never_seen_is_inconclusive_and_still_resumes(sleeper, monkeypatch):
    monkeypatch.setattr(msp, "process_state", lambda pid, proc=msp.PROC: "S")
    outcome = msp.reset_high_water(sleeper.pid, stop_wait_s=0.05)
    assert outcome["status"] == "inconclusive"
    assert "stop not confirmed" in outcome["reason"]
    monkeypatch.undo()
    time.sleep(0.05)
    assert msp.process_state(sleeper.pid) != "T"


def test_three_unequal_readings_are_inconclusive(sleeper, monkeypatch):
    reads = []
    monkeypatch.setattr(msp, "read_status", lambda pid, proc=msp.PROC: (
        reads.append(1) or {"vm_hwm_bytes": 2, "vm_rss_bytes": 1}))
    outcome = _reset_or_skip(sleeper.pid)
    assert outcome["status"] == "inconclusive" and outcome["tries"] == 3
    assert len(reads) == 3
    monkeypatch.undo()
    time.sleep(0.05)
    assert msp.process_state(sleeper.pid) != "T"


def test_an_unexpected_error_in_the_window_still_resumes(sleeper, monkeypatch):
    def _refuse(pid, proc=msp.PROC):
        raise PermissionError("refused")

    monkeypatch.setattr(msp, "read_status", _refuse)
    real_write = Path.write_text
    monkeypatch.setattr(Path, "write_text", lambda self, text, *a, **k: (
        None if self.name == "clear_refs" else real_write(self, text, *a, **k)))
    with pytest.raises(PermissionError):
        msp.reset_high_water(sleeper.pid)
    monkeypatch.undo()
    time.sleep(0.05)
    assert msp.process_state(sleeper.pid) != "T"


def test_a_process_that_is_gone_fails_the_reset():
    child = _sleeper()
    child.kill()
    child.wait()
    assert msp.reset_high_water(child.pid)["status"] == "failed"


def test_phase_status_follows_the_procedure():
    confirmed, inconclusive = {"status": "confirmed"}, {"status": "inconclusive"}
    assert msp._status(confirmed, True, complete=True) == "completed"
    assert msp._status(confirmed, True, complete=False) == "incomplete"
    assert msp._status(inconclusive, True, complete=True) == "inconclusive"
    assert msp._status(confirmed, False, complete=True) == "failed"
    assert msp._status({"status": "failed"}, True, complete=True) == "failed"


# ------------------------------------------------------------ system counter
def test_the_sampler_keeps_timed_readings_in_order():
    values = iter(range(1000))
    sampler = msp.SystemSampler(0.01, read=lambda: {"system_in_use_bytes": next(values),
                                                    "swap_in_use_bytes": 0})
    sampler.start()
    t0 = time.monotonic()
    time.sleep(0.1)
    t1 = time.monotonic()
    sampler.stop()
    seen = [r["system_in_use_bytes"] for r in sampler.window(t0, t1)]
    assert len(seen) >= 3 and seen == sorted(seen)
    assert sampler.window(t1 + 10, t1 + 20) == []


@pytest.mark.parametrize("swap, rise, met", [
    ([0, 8192, 0], 8192, False),          # rose and came back: still grew during R0
    ([0, 4096, 8192], 8192, False),       # kept growing
    ([4096, 4096, 4096], 0, True),        # a steady non-zero baseline
    ([8192, 4096, 0], 0, True),           # only falling
    ([4096, 0, 4096], 4096, False),       # fell, then grew back
])
def test_r0_fails_its_precondition_if_swap_grew_at_any_point(swap, rise, met):
    samples = [{"system_in_use_bytes": 1, "swap_in_use_bytes": v} for v in swap]
    base, r0 = msp.r0_fields(samples, 10.0)
    assert r0["swap_largest_rise_bytes"] == rise
    assert r0["measurement_precondition_met"] is met
    assert r0["swap_net_change_bytes"] == swap[-1] - swap[0]
    assert r0["swap_in_use_peak_bytes"] == max(swap) and base["swap"] == sorted(swap)[1]


def test_a_phase_peak_includes_the_direct_reading_at_its_end():
    readings = iter([{"system_in_use_bytes": 5, "swap_in_use_bytes": 1},
                     {"system_in_use_bytes": 9, "swap_in_use_bytes": 3}])
    sampler = msp.SystemSampler(read=lambda: next(readings))
    t0 = time.monotonic()
    sampler.sample()
    window = msp._window(sampler, t0, time.monotonic(), {"system": 4, "swap": 1}, None)
    assert window["system_in_use_sampled_peak_bytes"] == 9
    assert window["system_in_use_sampled_peak_less_r0_bytes"] == 5
    assert window["system_in_use_at_end_bytes"] == 9
    assert window["swap_increase_bytes"] == 2


# --------------------------------------------------------------- the schema
def test_the_schema_is_the_services_own_and_loads_without_the_application():
    # The measuring process's memory is inside the system counter; importing
    # through `src.api` would build the whole application in it.
    code = ("import sys; from scripts import measure_served_pipeline as m; "
            "s = m.load_diagnose_schema(); "
            "print(m.request_defaults(s), m.max_phenotypes(s), "
            "[x for x in ('torch', 'gradio', 'src.api.main') if x in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=msp.REPO_ROOT, check=True).stdout.strip()
    assert out == ("{'top_k': 10, 'include_explanations': True, 'include_paths': True} "
                   "100 []")


def test_the_loaded_schema_accepts_a_response_and_refuses_a_malformed_one():
    schema = msp.load_diagnose_schema()
    body = {"session_id": "s", "patient_id": "p", "timestamp": "t", "candidates": [
        {"rank": 1, "disease_id": "d", "disease_name": "n", "confidence_score": 0.5}],
        "inference_time_ms": 1.0, "model_version": "1.0.0"}
    schema.DiagnoseResponse.model_validate(body)
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        schema.DiagnoseResponse.model_validate({**body, "candidates": [{"rank": 1}]})


# -------------------------------------------------------------- the workload
@pytest.fixture
def workspace(tmp_path):
    from scripts.setup_demo import build_demo_kg

    root = tmp_path / "ws"
    root.mkdir()
    kg = build_demo_kg()
    kg.save_json(str(root / "kg.json"))
    samples = [{"patient_id": f"sim_val_{i:06d}", "phenotype_ids": [i % 10, (i + 3) % 10],
                "disease_id": i % 5} for i in range(8)]
    (root / "val_samples.json").write_text(json.dumps(samples))
    return root


def test_the_workload_is_seeded_and_sends_hpo_ids(workspace):
    first = msp.prepare_workload(str(workspace), 7, limit=4,
                                 validation_requests=3, maximum_requests=2)
    again = msp.prepare_workload(str(workspace), 7, limit=4,
                                 validation_requests=3, maximum_requests=2)
    assert first == again
    kinds = [r["kind"] for r in first["requests"]]
    assert sorted(kinds) == ["maximum"] * 2 + ["validation"] * 3
    for request in first["requests"]:
        assert all(p.startswith("HP:") for p in request["phenotypes"])
        if request["kind"] == "maximum":
            assert len(set(request["phenotypes"])) == 4
        else:
            assert len(request["phenotypes"]) == 2
    other = msp.prepare_workload(str(workspace), 8, limit=4,
                                 validation_requests=3, maximum_requests=2)
    assert other["requests"] != first["requests"]


def test_the_two_kinds_are_shuffled_together_as_the_rule_says(workspace):
    # Without the shuffle every validation request would precede every maximum one.
    orders = [[r["kind"] for r in msp.prepare_workload(
        str(workspace), seed, limit=4, validation_requests=3, maximum_requests=2)["requests"]]
        for seed in range(10)]
    assert any(order.index("maximum") < 3 for order in orders)
    assert "shuffled together" in msp.WORKLOAD_RULE


def test_validation_phenotypes_are_mapped_by_the_graphs_own_index(workspace):
    from src.kg.graph import KnowledgeGraph

    reverse = KnowledgeGraph.load_json(str(workspace / "kg.json")).get_reverse_node_mapping()
    (workspace / "val_samples.json").write_text(json.dumps(
        [{"patient_id": "sim_val_000000", "phenotype_ids": [2, 5], "disease_id": 0}]))
    out = msp.prepare_workload(str(workspace), 0, limit=4, validation_requests=1,
                               maximum_requests=0)
    expected = [reverse["phenotype"][i].split(":", 1)[1] for i in (2, 5)]
    assert out["requests"] == [{"kind": "validation", "phenotypes": expected}]


@pytest.mark.parametrize("samples, message", [
    ([{"phenotype_ids": [0, 99]}], "not an HPO node"),
    ([{"phenotype_ids": list(range(10))}], "the API accepts 1 to 4"),
    ([{"phenotype_ids": []}], "the API accepts 1 to 4"),
])
def test_a_sample_the_api_cannot_take_is_refused_not_trimmed(workspace, samples, message):
    (workspace / "val_samples.json").write_text(json.dumps(samples))
    with pytest.raises(SystemExit, match=message):
        msp.prepare_workload(str(workspace), 0, limit=4, validation_requests=1,
                             maximum_requests=0)


def test_too_few_validation_samples_is_refused(workspace):
    with pytest.raises(SystemExit, match="holds 8 samples"):
        msp.prepare_workload(str(workspace), 0, limit=4, validation_requests=9)


def test_the_graph_is_loaded_in_a_child_process_not_the_measuring_one(workspace):
    # The graph's Python objects would otherwise sit inside every system reading.
    code = ("import sys; from pathlib import Path; "
            "from scripts import measure_served_pipeline as m; "
            "m.VALIDATION_REQUESTS, m.MAXIMUM_REQUESTS = 3, 2; "
            f"out = m.build_workload(Path({str(workspace)!r}), 7, 4); "
            "print(len(out['requests']), 'src.kg.graph' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=msp.REPO_ROOT, check=True).stdout.strip()
    assert out == "5 False"


def test_the_workload_description_carries_counts_and_no_ids(workspace):
    prepared = msp.prepare_workload(str(workspace), 7, limit=4, validation_requests=3,
                                    maximum_requests=2)
    described = msp.describe_workload(prepared, 7, {"top_k": 10}, 4)
    assert described["phenotype_counts_by_kind"] == {
        "maximum": {"requests": 2, "min": 4, "median": 4, "max": 4},
        "validation": {"requests": 3, "min": 2, "median": 2, "max": 2},
    }
    assert described["val_samples_sha256"] and "HP:" not in json.dumps(described)


# --------------------------------------------------------- starting the service
def test_the_service_is_started_through_the_launcher_as_the_procedure_names():
    assert msp.launch_command("/v/python", 5555) == [
        "/v/python", str(msp.REPO_ROOT / "scripts" / "launch" / "shep_launch.py"),
        "--no-auto-install", "--no-browser", "--", "--host", "127.0.0.1", "--port", "5555"]


def test_the_shell_cannot_turn_the_preset_into_an_override(tmp_path):
    shell = {"PATH": "/bin", "PYTORCH_ALLOC_CONF": "backend:native",
             "PYTORCH_CUDA_ALLOC_CONF": "x", "SHEPHERD_ALLOC_SOURCE": "env",
             "SHEPHERD_SP_HOP_BOUND": "3", "ATTENTION_ORDER": "naive",
             "COMMANDLINE_ARGS": "-- --port 1", "SHEP_COMMANDLINE_ARGS": "--flash-attn"}
    env = msp.child_env(shell, tmp_path / "ws", tmp_path / "ck.pt")
    assert env == {
        "PATH": "/bin",
        "PYTHONUNBUFFERED": "1",
        "SHEPHERD_KG_PATH": str(tmp_path / "ws" / "kg.json"),
        "SHEPHERD_DATA_DIR": str(tmp_path / "ws"),
        "SHEPHERD_CHECKPOINT_PATH": str(tmp_path / "ck.pt"),
        "SHEPHERD_DEVICE": "cuda",
    }


def test_readiness_needs_every_field_the_procedure_names():
    ready = {"initialized": True, "gnn_ready": True, "has_model": True, "sp_ready": True,
             "sp_kg_binding": "verified", "scoring_mode": "gnn_plus_shortest_path",
             "fingerprint_warnings": ["w"], "current_data_dir": "/secret/path",
             "checkpoint_meta": {"device": "cuda:0", "params": 1}}
    reading = msp.readiness(ready)
    assert reading["asserted"] and reading["fingerprint_warning_count"] == 1
    assert reading["serving_device"] == "cuda:0"
    assert "/secret/path" not in json.dumps(reading)
    for change in ({"sp_ready": False}, {"has_model": False}, {"initialized": False},
                   {"sp_kg_binding": "unrecorded"}, {"scoring_mode": "gnn_only"},
                   {"checkpoint_meta": {"device": "cpu"}}, {"checkpoint_meta": {}},
                   {"checkpoint_meta": None}):
        assert not msp.readiness({**ready, **change})["asserted"], change


# ------------------------------------------------------------------ allocator
@pytest.mark.parametrize("values, backends, governing", [
    ({"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync", "PYTORCH_CUDA_ALLOC_CONF": None},
     {}, "PYTORCH_ALLOC_CONF"),
    ({"PYTORCH_ALLOC_CONF": None, "PYTORCH_CUDA_ALLOC_CONF": "backend:native"},
     {}, "PYTORCH_CUDA_ALLOC_CONF"),
    ({"PYTORCH_ALLOC_CONF": None, "PYTORCH_CUDA_ALLOC_CONF": None}, {}, None),
    ({"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync", "PYTORCH_CUDA_ALLOC_CONF": "backend:native"},
     {"backend:cudaMallocAsync": "cudaMallocAsync", "backend:native": "native",
      "both": "cudaMallocAsync"}, "PYTORCH_ALLOC_CONF"),
    ({"PYTORCH_ALLOC_CONF": "backend:native", "PYTORCH_CUDA_ALLOC_CONF": "backend:native"},
     {"backend:native": "native", "both": "native"}, "indistinguishable"),
])
def test_which_allocator_governs_is_torchs_answer(values, backends, governing):
    def probe(given):
        present = [v for v in given.values() if v is not None]
        key = "both" if len(present) == 2 else (present[0] if present else None)
        return {"allocator_backend": backends.get(key, "native"), "torch": "t"}

    reading, together = msp.allocator_reading(values, probe)
    assert reading["governing_variable"] == governing
    assert together["torch"] == "t"


# ----------------------------------------------------------------- summaries
def _repeat(r1="completed", r2="completed", r3="completed", r4="completed", precondition=True,
            wall=1.0):
    return {"phases": {"R0": {"measurement_precondition_met": precondition},
                       "R1": {"status": r1, "wall_seconds": wall, "vm_hwm_bytes": 10},
                       "R2": {"status": r2}, "R3": {"status": r3}, "R4": {"status": r4}}}


def test_readings_are_complete_only_with_three_repeats_all_completed():
    assert msp.readings_complete([_repeat()] * 3)
    assert not msp.readings_complete([_repeat()] * 2)
    assert not msp.readings_complete([_repeat()] * 2 + [_repeat(r4="inconclusive")])
    assert not msp.readings_complete([_repeat()] * 2 + [_repeat(precondition=False)])


def test_the_summary_is_median_and_maximum_over_completed_phases():
    summary = msp.summarise([_repeat(wall=1.0), _repeat(wall=5.0), _repeat(wall=2.0),
                             _repeat(r1="failed", wall=99.0)])
    assert summary["R1"]["wall_seconds"] == {"median": 2.0, "max": 5.0, "n": 3}
    assert summary["R2"]["vm_rss_after_bytes"] == {"n": 0}


# ------------------------------------------------------- a whole repeat, faked
FAKE_SERVER = textwrap.dedent('''\
    import json, os, sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    MODE = os.environ.get("FAKE_MODE", "ready")
    if MODE == "exit":
        sys.exit(3)
    if os.environ.get("FAKE_PIDFILE"):
        with open(os.environ["FAKE_PIDFILE"], "w") as f:
            f.write(str(os.getpid()))

    def status():
        return {"initialized": True, "gnn_ready": True, "has_model": True,
                "sp_ready": MODE != "not_ready", "sp_kg_binding": "verified",
                "scoring_mode": "gnn_plus_shortest_path", "fingerprint_warnings": [],
                "current_data_dir": os.environ["SHEPHERD_DATA_DIR"],
                "current_checkpoint_path": ("/elsewhere.pt" if MODE == "other_checkpoint"
                                            else os.environ["SHEPHERD_CHECKPOINT_PATH"]),
                "checkpoint_meta": {"device": "cpu" if MODE == "cpu" else "cuda"}}

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send(200, status())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/api/v1/diagnose":
                if MODE == "slow":
                    if os.environ.get("FAKE_INFLIGHT"):
                        open(os.environ["FAKE_INFLIGHT"], "w").close()
                    import time
                    time.sleep(float(os.environ.get("FAKE_SLOW_SECONDS", "3")))
                if MODE == "truncated":
                    self.send_response(200)
                    self.send_header("Content-Length", "1000")
                    self.end_headers()
                    self.wfile.write(b'{"session_id"')
                    self.close_connection = True
                    return
                if MODE == "oom" and len(body["phenotypes"]) > 2:
                    self.send(500, {"error": "CUDA out of memory"})
                    return
                self.send(200, {"session_id": "s", "patient_id": "p", "timestamp": "t",
                                "candidates": [], "inference_time_ms": 1.0,
                                "model_version": "1.0.0"})
            else:
                ok = body["checkpoint_path"] == os.environ["SHEPHERD_CHECKPOINT_PATH"]
                self.send(200, {"success": ok, "message": "m", "status": status()})

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
''')


@pytest.fixture
def fake_repeat(tmp_path, monkeypatch):
    server = tmp_path / "fake_server.py"
    server.write_text(FAKE_SERVER)
    logs = tmp_path / "logs"
    logs.mkdir()

    def run(mode="ready", workload=None, server_check=lambda pid: True):
        monkeypatch.setenv("FAKE_MODE", mode)
        monkeypatch.setenv("PYTORCH_ALLOC_CONF", "backend:native")  # the shell's, not passed on
        return msp.run_repeat(
            index=1, workspace=tmp_path / "ws", checkpoint=tmp_path / "ck.pt",
            workload=workload or [{"kind": "validation", "phenotypes": ["HP:0000001"]},
                                  {"kind": "maximum", "phenotypes": ["HP:1", "HP:2", "HP:3"]}],
            defaults={"top_k": 10, "include_explanations": True, "include_paths": True},
            schema=msp.load_diagnose_schema(), r0_seconds=0.3, ready_timeout_s=30,
            log_dir=logs, command=lambda port: [sys.executable, str(server), str(port)],
            server_check=server_check)

    return run


def test_a_whole_repeat_runs_r0_to_r4_and_records_no_paths(fake_repeat, tmp_path):
    result = fake_repeat()
    phases = result["phases"]
    reset_ok = phases["R2"]["reset"]["status"] == "confirmed"
    expected = "completed" if reset_ok else "inconclusive"
    assert phases["R0"]["samples"] >= 2
    assert phases["R1"]["status"] == "completed"
    assert phases["R1"]["readiness"]["asserted"] and phases["R1"]["serves_named_checkpoint"]
    assert [phases[p]["status"] for p in ("R2", "R3", "R4")] == [expected] * 3
    assert phases["R2"]["workload"] == {"planned": 2, "sent": 2, "accepted": 2,
                                        "failures": {}, "ended_without_response": False}
    assert phases["R1"]["readiness"]["serving_device"] == "cuda"
    assert phases["R3"]["success"] and phases["R3"]["readiness"]["asserted"]
    assert result["server_environment"]["PYTORCH_ALLOC_CONF"] is None
    assert result["server_exit_status"] is not None
    assert "error" not in result
    assert str(tmp_path) not in json.dumps(result)
    if reset_ok:
        assert phases["R4"]["vm_hwm_increase_bytes"] is not None


def test_a_failed_request_makes_the_workload_phase_incomplete(fake_repeat):
    result = fake_repeat(mode="oom")
    r2 = result["phases"]["R2"]
    assert r2["workload"]["failures"] == {"http_500": 1, "out_of_memory_in_body": 1}
    assert r2["status"] in ("incomplete", "inconclusive")
    # A complete error response finishes that request: the workload goes on.
    assert r2["workload"]["sent"] == 2 and "R3" in result["phases"]


def test_readiness_not_asserted_stops_the_repeat_at_r1(fake_repeat):
    result = fake_repeat(mode="not_ready")
    assert result["phases"]["R1"]["status"] == "failed"
    assert result["phases"]["R1"]["reason"] == "readiness not asserted"
    assert "R2" not in result["phases"]
    assert result["server_exit_status"] is not None


def test_a_server_that_exits_keeps_what_the_machine_went_through(fake_repeat):
    r1 = fake_repeat(mode="exit")["phases"]["R1"]
    assert (r1["status"], r1["reason"], r1["exit_status"]) == (
        "failed", "server exited before answering", 3)
    for field in ("wall_seconds", "system_in_use_sampled_peak_bytes",
                  "system_in_use_sampled_peak_less_r0_bytes", "swap_baseline_bytes",
                  "swap_sampled_peak_bytes", "swap_increase_bytes", "oom_kills_on_machine"):
        assert field in r1, field
    assert r1["vm_hwm_bytes"] is None and r1["vm_rss_bytes"] is None
    assert r1["process_counters"].startswith("unavailable")


def test_the_whole_script_writes_evidence_without_paths(workspace, tmp_path, monkeypatch):
    server = tmp_path / "fake_server.py"
    server.write_text(FAKE_SERVER)
    (workspace / "shortest_paths.pt").write_bytes(b"sp")
    checkpoint = workspace / "ck.pt"
    checkpoint.write_bytes(b"ck")
    monkeypatch.setattr(msp, "VALIDATION_REQUESTS", 3)
    monkeypatch.setattr(msp, "MAXIMUM_REQUESTS", 2)
    monkeypatch.setattr(msp, "max_phenotypes", lambda schema: 4)  # the demo graph has 10
    monkeypatch.setattr(msp, "launch_command",
                        lambda python, port: [python, str(server), str(port)])
    real_run_repeat = msp.run_repeat
    monkeypatch.setattr(msp, "run_repeat", lambda **kw: real_run_repeat(
        **kw, server_check=lambda pid: True))
    monkeypatch.setattr(msp, "torch_probe", lambda allocator: {"torch": "t", "torch_cuda": "c"})
    monkeypatch.setattr(msp, "driver_version", lambda: None)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "backend:native")  # the shell's; not passed on
    output = tmp_path / "readings.json"
    args = ["--workspace", str(workspace), "--checkpoint", str(checkpoint), "--output",
            str(output), "--repeats", "1", "--r0-seconds", "0.3", "--log-dir",
            str(tmp_path / "logs")]
    assert msp.main(args) == 1   # one repeat is a trial, never complete readings
    text = output.read_text()
    evidence = json.loads(text)
    assert str(tmp_path) not in text and "HP:" not in text
    assert evidence["readings_complete"] is False
    assert evidence["constants"]["repeats_run"] == 1
    assert evidence["workload"]["requests"] == 5
    # The fake server is started directly, not through the launcher, so nothing
    # applied a preset: the shell's value must not have reached it either.
    assert evidence["allocator"]["server_environment_raw"]["PYTORCH_ALLOC_CONF"] is None
    assert evidence["allocator"]["governing_variable"] is None
    assert evidence["code"]["commit"]
    assert evidence["subject"]["checkpoint_sha256"]
    with pytest.raises(SystemExit, match="exists"):
        msp.main(args)


def test_a_launcher_that_did_not_hand_over_is_not_measured(fake_repeat):
    result = fake_repeat(server_check=lambda pid: False)
    assert result["phases"]["R1"]["status"] == "failed"
    assert "did not hand over" in result["phases"]["R1"]["reason"]
    assert "R2" not in result["phases"]


def test_a_service_serving_another_checkpoint_fails_r1(fake_repeat):
    result = fake_repeat(mode="other_checkpoint")
    assert result["phases"]["R1"]["reason"] == "the service does not serve the named checkpoint"
    assert "R2" not in result["phases"]


def test_an_unconfirmed_reset_withholds_the_high_water_mark(fake_repeat, monkeypatch):
    monkeypatch.setattr(msp, "reset_high_water", lambda pid: {
        "status": "inconclusive", "reason": "stop not confirmed within 5 s"})
    phases = fake_repeat()["phases"]
    assert [phases[p]["status"] for p in ("R2", "R3", "R4")] == ["inconclusive"] * 3
    assert phases["R2"]["vm_hwm_bytes"] is None and phases["R2"]["vm_rss_after_bytes"]
    assert phases["R4"]["vm_hwm_increase_bytes"] is None


def test_an_unexpected_error_is_recorded_by_type_and_the_server_stopped(fake_repeat, monkeypatch):
    def _boom(pid, proc=msp.PROC):
        raise RuntimeError("/a/secret/path")

    monkeypatch.setattr(msp, "read_environ", _boom)
    result = fake_repeat()
    assert result["error"] == "RuntimeError"
    assert "/a/secret/path" not in json.dumps(result)
    assert result["server_exit_status"] is not None


def test_a_model_served_on_the_cpu_is_not_a_reading(fake_repeat):
    r1 = fake_repeat(mode="cpu")["phases"]["R1"]
    assert r1["status"] == "failed" and r1["reason"] == "readiness not asserted"
    assert r1["readiness"]["serving_device"] == "cpu"


# ------------------------------------------- requests whose outcome is unknown
def test_a_request_with_no_response_ends_the_repeat_before_any_reset(fake_repeat, monkeypatch):
    # The client stops waiting after 0.3 s; the server is still working for 3 s.
    # Resetting now would pause it with that request in flight (§7.3).
    monkeypatch.setattr(msp, "REQUEST_TIMEOUT_S", 0.3)
    resets = []
    real_reset = msp.reset_high_water
    monkeypatch.setattr(msp, "reset_high_water",
                        lambda pid: resets.append(pid) or real_reset(pid))
    result = fake_repeat(mode="slow")
    r2 = result["phases"]["R2"]
    assert r2["status"] == "incomplete" and "without a response" in r2["reason"]
    assert r2["workload"] == {"planned": 2, "sent": 1, "accepted": 0,
                              "failures": {"no_response": 1}, "ended_without_response": True}
    assert "R3" not in result["phases"] and "R4" not in result["phases"]
    assert len(resets) == 1   # R2's, taken before the request was sent
    assert result["server_exit_status"] is not None
    assert msp.summarise([result])["R2"]["wall_seconds"] == {"n": 0}


# ------------------------------------------------------------- stop signals
def _worker(code, tmp_path):
    """A measuring process of its own, so a real SIGTERM can be sent to it."""
    script = tmp_path / "worker.py"
    script.write_text(textwrap.dedent(f"""\
        import sys, time
        from pathlib import Path
        sys.path.insert(0, {str(msp.REPO_ROOT)!r})
        from scripts import measure_served_pipeline as m
        m.install_stop_handlers()
    """) + textwrap.dedent(code))
    return subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True,
                            cwd=msp.REPO_ROOT)


def test_sigterm_inside_the_reset_window_resumes_the_server(sleeper, tmp_path):
    worker = _worker(f"""\
        real = m.read_status
        def slow(pid, proc=m.PROC):
            print("in-window", flush=True)
            time.sleep(30)
            return real(pid, proc)
        m.read_status = slow
        m.reset_high_water({sleeper.pid})
    """, tmp_path)
    assert worker.stdout.readline().strip() == "in-window"
    assert msp.process_state(sleeper.pid) == "T"
    worker.send_signal(signal.SIGTERM)
    worker.wait(timeout=30)
    time.sleep(0.05)
    assert msp.process_state(sleeper.pid) != "T"


def test_sigterm_mid_workload_stops_and_reaps_the_server(tmp_path):
    server = tmp_path / "fake_server.py"
    server.write_text(FAKE_SERVER)
    pidfile, inflight = tmp_path / "server.pid", tmp_path / "inflight"
    (tmp_path / "logs").mkdir()
    worker = _worker(f"""\
        import os
        os.environ.update(FAKE_MODE="slow", FAKE_SLOW_SECONDS="60",
                          FAKE_PIDFILE={str(pidfile)!r}, FAKE_INFLIGHT={str(inflight)!r})
        m.run_repeat(index=1, workspace=Path("ws"), checkpoint=Path("ck.pt"),
                     workload=[{{"kind": "validation", "phenotypes": ["HP:1"]}}], defaults={{}},
                     schema=m.load_diagnose_schema(), r0_seconds=0.1, ready_timeout_s=30,
                     log_dir=Path({str(tmp_path / "logs")!r}),
                     command=lambda port: [sys.executable, {str(server)!r}, str(port)],
                     server_check=lambda pid: True)
    """, tmp_path)
    deadline = time.monotonic() + 30
    while not inflight.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert inflight.exists(), "the request never reached the server"
    server_pid = int(pidfile.read_text())
    try:
        worker.send_signal(signal.SIGTERM)
        worker.wait(timeout=60)
        # Reaped by the worker's cleanup. Left behind, it would still exist:
        # it runs in its own session and would outlive the worker.
        with pytest.raises(ProcessLookupError):
            os.kill(server_pid, 0)
    finally:
        try:
            os.kill(server_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_a_second_signal_cannot_cut_the_cleanup_short():
    # The first SIGTERM raises; one arriving while the cleanup it started is
    # still stopping the server must not raise again inside that `finally`.
    previous = msp.install_stop_handlers()
    try:
        with pytest.raises(msp.Stopped):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.05)   # the handler has run by now, and raised nothing
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def test_a_signal_already_ignored_stays_ignored():
    # Under nohup SIGHUP is ignored; installing a handler would undo that.
    before = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        previous = msp.install_stop_handlers()
        assert signal.SIGHUP not in previous
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    finally:
        signal.signal(signal.SIGHUP, before)


def test_a_stop_signal_writes_no_evidence_and_restores_the_handlers(workspace, tmp_path,
                                                                   monkeypatch):
    (workspace / "shortest_paths.pt").write_bytes(b"sp")
    (workspace / "ck.pt").write_bytes(b"ck")
    monkeypatch.setattr(msp, "build_workload", lambda *a: {
        "requests": [], "val_samples_sha256": "x", "val_samples_count": 0,
        "hpo_phenotype_nodes": 0})

    def _stopped(**kw):
        raise msp.Stopped(signal.SIGTERM)

    monkeypatch.setattr(msp, "run_repeat", _stopped)
    output = tmp_path / "readings.json"
    code = msp.main(["--workspace", str(workspace), "--checkpoint", str(workspace / "ck.pt"),
                     "--output", str(output), "--log-dir", str(tmp_path / "logs")])
    assert code == 128 + signal.SIGTERM
    assert not output.exists()
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


# ------------------------------------------------ metadata never costs the readings
def test_a_probe_that_times_out_is_recorded_not_raised(monkeypatch):
    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("python", 300)

    monkeypatch.setattr(msp.subprocess, "run", _timeout)
    assert msp.torch_probe({}) == {"probe_error": "TimeoutExpired"}


def test_an_allocator_reading_that_fails_after_the_repeats_keeps_them(workspace, tmp_path,
                                                                     monkeypatch):
    (workspace / "shortest_paths.pt").write_bytes(b"sp")
    (workspace / "ck.pt").write_bytes(b"ck")
    monkeypatch.setattr(msp, "build_workload", lambda *a: {
        "requests": [], "val_samples_sha256": "x", "val_samples_count": 0,
        "hpo_phenotype_nodes": 0})
    monkeypatch.setattr(msp, "run_repeat", lambda **kw: {
        **_repeat(), "repeat": kw["index"],
        "server_environment": {"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync",
                               "SHEPHERD_ALLOC_SOURCE": "preset"}})
    monkeypatch.setattr(msp.time, "sleep", lambda s: None)

    def _timeout(values, probe=None):
        raise subprocess.TimeoutExpired("python", 300)

    monkeypatch.setattr(msp, "allocator_reading", _timeout)
    monkeypatch.setattr(msp, "driver_version", lambda: None)
    output = tmp_path / "readings.json"
    code = msp.main(["--workspace", str(workspace), "--checkpoint", str(workspace / "ck.pt"),
                     "--output", str(output), "--log-dir", str(tmp_path / "logs")])
    evidence = json.loads(output.read_text())
    assert evidence["allocator"]["reading_error"] == "TimeoutExpired"
    assert evidence["allocator"]["source"] == "preset"
    assert len(evidence["repeats"]) == 3 and evidence["readings_complete"] is True
    assert code == 0


def test_a_response_cut_off_mid_body_is_an_unknown_outcome_too(fake_repeat):
    # http.client.IncompleteRead is not an OSError; it must not end the repeat
    # as an unexpected error, nor count as a finished request.
    result = fake_repeat(mode="truncated")
    r2 = result["phases"]["R2"]
    assert r2["workload"]["failures"] == {"no_response": 1}
    assert r2["workload"]["ended_without_response"] is True
    assert "R3" not in result["phases"] and "error" not in result
