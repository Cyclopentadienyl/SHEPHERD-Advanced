"""
Readings 1-4 of PLAN_B04 §13, taken from the running service.
==============================================================
Implements `docs/working/scorer-measurement/PLAN_B04_PRODUCTIONISATION.md`
§7.3, which was reviewed before this was written. **Read §7.3 before changing
anything here**: a change this script needs is made there first and reviewed.

**The service itself, started the way the unit starts it.** The launcher, with
`--no-auto-install --no-browser`, execs into uvicorn, so the PID this script
starts is the server's, and that PID is the one measured. Nothing builds the
pipeline in this process: a harness that did would leave out uvicorn, FastAPI
and the mounted Gradio dashboard, which is what §13 exists to include.

Per repeat, each in its own service process (§7.3 "Phases"):

  R0  system and swap in use on the quiet machine, before launch
  R1  launch to ready: wall time, VmHWM and VmRSS at ready, system peak
  R2  reset, 200 serial /diagnose requests: VmRSS after, VmHWM, system peak
  R3  reset, one reload of the same workspace and checkpoint: success, wall
      time, readiness re-asserted, VmRSS after
  R4  the same reload: VmHWM and system peak over it, less their values before

**The reset is confirmed with the server paused** (`reset_high_water`): SIGSTOP,
wait at most 5 s for state T, write 5 to clear_refs, accept only VmHWM ==
VmRSS, at most three tries, SIGCONT in a `finally`. Anything else makes the
phase inconclusive, not measured.

**A stop signal takes the cleanup path** (`install_stop_handlers`): SIGTERM or
SIGHUP resumes a paused server and stops every server the run started, and no
evidence is written. A request with no whole response ends its repeat before
any further pause, because the server may still be working on it.

**Output: one aggregate JSON**, BACKLOG §5.2's pattern. Bytes, seconds, counts,
digests, versions and readiness fields -- no paths, no host or operator names,
no phenotype ids. The server's own log goes to a directory named on stderr and
is never part of the evidence.

**Completing the readings is not passing a capacity gate** (§7.3). The numbers
are recorded, not judged.

Linux only, like `benchmark_sp_lookup.py`: every counter is read from `/proc`.

Usage:
    .venv/bin/python scripts/measure_served_pipeline.py \\
        --workspace data/workspaces/<ws> \\
        --checkpoint data/workspaces/<ws>/checkpoints/hgt/<file>.pt \\
        --output readings.json

Module: scripts/measure_served_pipeline.py
"""
from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import os
import platform
import random
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.fingerprint import file_sha256  # noqa: E402  (no torch import)

# =============================================================================
# Fixed by §7.3 -- changing one of these changes the procedure
# =============================================================================
REPEATS = 3
SAMPLE_INTERVAL_S = 0.2
STOP_WAIT_S = 5.0
RESET_TRIES = 3
VALIDATION_REQUESTS = 150
MAXIMUM_REQUESTS = 50

#: Readiness, asserted before anything counts (§7.3), plus the scoring mode
#: §7.1.1 says each reading asserts on the running service, and the device the
#: model was moved to, which must be CUDA: a reading from a model served on the
#: CPU would describe a deployment this project does not run.
READY_TRUE_FIELDS = ("initialized", "gnn_ready", "has_model", "sp_ready")
READY_BINDING = "verified"
READY_SCORING_MODE = "gnn_plus_shortest_path"
READY_DEVICE_PREFIX = "cuda"

#: Read raw from the server's /proc/<pid>/environ, each recorded as absent when
#: absent (§7.3 "With the allocator the deployment runs").
ALLOCATOR_VARS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
RECORDED_SERVER_VARS = ALLOCATOR_VARS + (
    "SHEPHERD_ALLOC_SOURCE", "ATTENTION_ORDER", "FLASHATTN_FORCE_DISABLE",
)

#: Not passed from the operator's shell to the launcher. The unit starts it with
#: none of them, so the launcher applies the saved preset and its own attention
#: settings. An allocator exported in the shell -- as it was for the training
#: run that produced the subject -- would otherwise reach the server as an
#: "explicit override" and describe a different start. Every `SHEPHERD_*` name
#: is removed too, and the four the procedure names are then set.
CHILD_ENV_REMOVED = ALLOCATOR_VARS + (
    "ATTENTION_ORDER", "ATTENTION_PLUGIN", "FLASHATTN_FORCE_DISABLE",
    "SHEP_COMMANDLINE_ARGS", "COMMANDLINE_ARGS",
)
CHILD_ENV_REMOVED_PREFIX = "SHEPHERD_"

# Operational limits, not part of what is measured.
READY_POLL_S = 0.5
REQUEST_TIMEOUT_S = 600.0
STOP_TIMEOUT_S = 120.0

PROC = Path("/proc")
DIAGNOSE_SCHEMA_FILE = REPO_ROOT / "src" / "api" / "routes" / "diagnose.py"
LAUNCHER = REPO_ROOT / "scripts" / "launch" / "shep_launch.py"


def require_linux() -> None:
    """Refuse where the counters are not Linux's; `--help` still works."""
    if not sys.platform.startswith("linux"):
        raise SystemExit(
            f"measure_served_pipeline reads /proc and cannot run on {sys.platform}. "
            "The readings are taken on the deployment host, which is Linux."
        )


class Stopped(BaseException):
    """A stop signal, raised where the script is, so every `finally` runs.

    A `BaseException`, like `KeyboardInterrupt`, so that a repeat's
    `except Exception` records it as nothing and lets it through to the cleanup.
    """


def install_stop_handlers() -> dict:
    """End the run on SIGTERM or SIGHUP through the same cleanup as an error.

    Their default action ends this process without running any `finally`. A
    server paused for a reset would then stay stopped, holding its memory, and a
    running server would be left behind in its own session. Raised as
    `Stopped` instead, they resume and stop the server on the way out. A signal
    already ignored, as under nohup, stays ignored. A second signal during that
    cleanup is ignored, so it cannot cut the cleanup short. SIGKILL cannot be
    caught, and nothing here claims to survive it.
    """
    raised = []

    def handler(signum, frame):
        if not raised:
            raised.append(signum)
            raise Stopped(signum)

    previous = {}
    for sig in (signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(sig) is signal.SIG_DFL:
            previous[sig] = signal.signal(sig, handler)
    return previous


# =============================================================================
# /proc
# =============================================================================
def read_status(pid: int, proc: Path = PROC) -> dict[str, int]:
    """VmRSS and VmHWM of `pid`, in bytes. A zombie has neither line."""
    values: dict[str, int] = {}
    for line in (proc / str(pid) / "status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            values["vm_rss_bytes"] = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            values["vm_hwm_bytes"] = int(line.split()[1]) * 1024
    if len(values) != 2:
        raise ProcessLookupError(pid)
    return values


def process_state(pid: int, proc: Path = PROC) -> str:
    """The state letter from /proc/<pid>/stat.

    Parsed after the **last** ')', because the command name inside the
    parentheses may itself contain spaces and parentheses. A zombie or a dead
    process is reported as gone rather than as a state.
    """
    text = (proc / str(pid) / "stat").read_text()
    state = text[text.rindex(")") + 2]
    if state in ("Z", "X"):
        raise ProcessLookupError(pid)
    return state


def read_meminfo(proc: Path = PROC) -> dict[str, int]:
    """System in use and swap in use, in bytes (§7.3's counter table)."""
    fields: dict[str, int] = {}
    for line in (proc / "meminfo").read_text().splitlines():
        name, _, rest = line.partition(":")
        if name in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            fields[name] = int(rest.split()[0]) * 1024
    return {
        "system_in_use_bytes": fields["MemTotal"] - fields["MemAvailable"],
        "swap_in_use_bytes": fields["SwapTotal"] - fields["SwapFree"],
    }


def read_oom_kills(proc: Path = PROC) -> int | None:
    """The machine's OOM-kill counter from /proc/vmstat; None where absent."""
    try:
        for line in (proc / "vmstat").read_text().splitlines():
            if line.startswith("oom_kill "):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def read_environ(pid: int, proc: Path = PROC) -> dict[str, str]:
    raw = (proc / str(pid) / "environ").read_bytes()
    pairs = (item.split(b"=", 1) for item in raw.split(b"\0") if b"=" in item)
    return {k.decode(errors="replace"): v.decode(errors="replace") for k, v in pairs}


def is_uvicorn_server(pid: int, proc: Path = PROC) -> bool:
    """Whether `pid` is now `python -m uvicorn src.api.main:app ...`.

    The launcher execs into that command, keeping the PID. If it did not --
    another platform, or a launcher change -- the PID is the launcher's and its
    counters are not the service's, so the run stops rather than measure it.
    """
    argv = (proc / str(pid) / "cmdline").read_bytes().split(b"\0")
    try:
        module = argv[argv.index(b"-m") + 1]
    except (ValueError, IndexError):
        return False
    return module == b"uvicorn" and b"src.api.main:app" in argv


# =============================================================================
# The reset, confirmed while paused (§7.3 "The reset is confirmed...")
# =============================================================================
def reset_high_water(
    pid: int,
    *,
    proc: Path = PROC,
    stop_wait_s: float = STOP_WAIT_S,
    tries: int = RESET_TRIES,
) -> dict[str, Any]:
    """Reset `pid`'s VmHWM to its VmRSS and confirm it, with `pid` stopped.

    A stopped process allocates nothing, so VmHWM == VmRSS after the write
    confirms the reset. Reclaim can still lower RSS while stopped, so an
    inequality is retried. SIGCONT is sent on every way out: confirmed, retries
    exhausted, a stop that never came, an error. The caller starts the phase's
    clock after this returns, which is after SIGCONT.
    """
    try:
        os.kill(pid, signal.SIGSTOP)
    except ProcessLookupError:
        return {"status": "failed", "reason": "process exited before the reset"}
    try:
        deadline = time.monotonic() + stop_wait_s
        while process_state(pid, proc) != "T":
            if time.monotonic() >= deadline:
                return {"status": "inconclusive",
                        "reason": f"stop not confirmed within {stop_wait_s:g} s"}
            time.sleep(0.01)
        values: dict[str, int] = {}
        for attempt in range(1, tries + 1):
            (proc / str(pid) / "clear_refs").write_text("5")
            values = read_status(pid, proc)
            if values["vm_hwm_bytes"] == values["vm_rss_bytes"]:
                return {"status": "confirmed", "tries": attempt, **values}
        return {"status": "inconclusive", "tries": tries,
                "reason": f"VmHWM != VmRSS after {tries} resets", **values}
    except (ProcessLookupError, FileNotFoundError):
        return {"status": "failed", "reason": "process exited inside the reset window"}
    finally:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass


# =============================================================================
# The system counter, sampled every 0.2 s from before launch
# =============================================================================
class SystemSampler:
    """Samples `read_meminfo` on a thread; a window's maximum is a sampled peak.

    An excursion shorter than the interval can be missed, which is why §7.3
    reports this beside the process high-water mark rather than instead of it.
    """

    def __init__(self, interval_s: float = SAMPLE_INTERVAL_S,
                 read: Callable[[], dict[str, int]] = read_meminfo) -> None:
        self.interval_s = interval_s
        self._read = read
        self._samples: list[tuple] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> SystemSampler:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        next_at = time.monotonic()
        while not self._stop.is_set():
            self.sample()
            next_at += self.interval_s
            self._stop.wait(max(0.0, next_at - time.monotonic()))

    def sample(self) -> dict[str, int]:
        """Take one reading now, keep it, and return it."""
        reading = self._read()
        with self._lock:
            self._samples.append((time.monotonic(), reading))
        return reading

    def window(self, t0: float, t1: float) -> list[dict[str, int]]:
        with self._lock:
            return [r for t, r in self._samples if t0 <= t <= t1]


# =============================================================================
# The response schema, loaded without the application
# =============================================================================
def load_diagnose_schema():
    """`src/api/routes/diagnose.py`, loaded by file.

    **Not imported through its package.** `src.api` imports `src.api.main`,
    which builds the whole application -- Gradio, torch, every route -- in this
    process: about 700 MB resident, measured, and this process's memory is
    part of the system counter. The file itself imports only FastAPI and
    Pydantic, so loading it alone gives the service's own request and response
    models without a copy of either.
    """
    name = "_measured_service_diagnose_schema"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DIAGNOSE_SCHEMA_FILE)
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs, as importlib's own recipe does: the file uses
    # postponed annotations, and Pydantic resolves them through sys.modules.
    # Unregistered, every response would fail validation as "not fully defined".
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def request_defaults(schema) -> dict[str, Any]:
    """The API's defaults, which the WebUI sends: top_k and the two flags."""
    fields = schema.DiagnoseRequest.model_fields
    return {name: fields[name].default
            for name in ("top_k", "include_explanations", "include_paths")}


def max_phenotypes(schema) -> int:
    """The request's phenotype limit, read from the model rather than restated."""
    for item in schema.DiagnoseRequest.model_fields["phenotypes"].metadata:
        if hasattr(item, "max_length"):
            return int(item.max_length)
    raise RuntimeError("DiagnoseRequest.phenotypes declares no max_length")


# =============================================================================
# The workload (§7.3 "The workload is fixed, seeded...")
# =============================================================================
WORKLOAD_RULE = (
    "validation: a seeded draw without replacement from val_samples.json, each "
    "request sending that sample's phenotypes, mapped from node index to HPO id "
    "by the graph's own index. maximum: for each request, a seeded draw of the "
    "API's maximum number of unique HPO ids among the graph's phenotype nodes. "
    "The two kinds are then shuffled together by the same generator, and the "
    "same list is sent in every repeat."
)


def prepare_workload(
    workspace: str,
    seed: int,
    limit: int,
    validation_requests: int = VALIDATION_REQUESTS,
    maximum_requests: int = MAXIMUM_REQUESTS,
) -> dict[str, Any]:
    """Build the request list; run in a child process (see `build_workload`).

    Phenotypes in `val_samples.json` are node indices. They are mapped to HPO
    ids by `KnowledgeGraph.get_reverse_node_mapping`, the index the samples
    were generated against, and a request sends a node's `local_id`, which is
    what `/diagnose` resolves. A sample the API could not take as generated is
    refused, never trimmed: that would silently change the workload.
    """
    from src.core.types import DataSource, NodeID
    from src.kg.graph import KnowledgeGraph

    root = Path(workspace)
    val_file = root / "val_samples.json"
    samples = json.loads(val_file.read_text(encoding="utf-8"))
    kg = KnowledgeGraph.load_json(str(root / "kg.json"))
    index_to_node = kg.get_reverse_node_mapping().get("phenotype", {})

    def hpo_id(node_str: str) -> str | None:
        node = NodeID.from_string(node_str)
        return node.local_id if node.source is DataSource.HPO else None

    hpo_by_index = {idx: hpo_id(node) for idx, node in index_to_node.items()}
    pool = sorted(h for h in hpo_by_index.values() if h is not None)

    if len(samples) < validation_requests:
        raise SystemExit(f"val_samples.json holds {len(samples)} samples; "
                         f"the workload draws {validation_requests}")
    if len(pool) < limit:
        raise SystemExit(f"the graph has {len(pool)} HPO phenotype nodes; "
                         f"a maximum request needs {limit}")

    rng = random.Random(seed)
    requests: list[dict[str, Any]] = []
    for position in rng.sample(range(len(samples)), validation_requests):
        indices = samples[position]["phenotype_ids"]
        ids = [hpo_by_index.get(int(i)) for i in indices]
        if None in ids:
            raise SystemExit("a drawn validation sample has a phenotype that is "
                             "not an HPO node in this graph")
        if not 1 <= len(ids) <= limit:
            raise SystemExit(f"a drawn validation sample has {len(ids)} phenotypes; "
                             f"the API accepts 1 to {limit}")
        requests.append({"kind": "validation", "phenotypes": ids})
    for _ in range(maximum_requests):
        requests.append({"kind": "maximum", "phenotypes": rng.sample(pool, limit)})
    rng.shuffle(requests)

    return {
        "requests": requests,
        "val_samples_sha256": file_sha256(val_file),
        "val_samples_count": len(samples),
        "hpo_phenotype_nodes": len(pool),
    }


def build_workload(workspace: Path, seed: int, limit: int) -> dict[str, Any]:
    """`prepare_workload` in a spawned child, so the graph it loads is gone.

    Loading `kg.json` into Python objects takes far more memory than the 200
    short lists it yields, and CPython need not return that memory to the
    system. Held by this process, it would sit inside every system reading.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=1,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        return pool.submit(prepare_workload, str(workspace), seed, limit,
                           VALIDATION_REQUESTS, MAXIMUM_REQUESTS).result()


def describe_workload(prepared: dict[str, Any], seed: int,
                      defaults: dict[str, Any], limit: int) -> dict[str, Any]:
    """What the evidence says about the workload: counts, never ids."""
    by_kind: dict[str, list[int]] = {}
    for request in prepared["requests"]:
        by_kind.setdefault(request["kind"], []).append(len(request["phenotypes"]))
    return {
        "seed": seed,
        "rule": WORKLOAD_RULE,
        "sent": "serially, one request at a time",
        "requests": len(prepared["requests"]),
        "request_fields": defaults,
        "api_phenotype_limit": limit,
        "val_samples_sha256": prepared["val_samples_sha256"],
        "val_samples_count": prepared["val_samples_count"],
        "hpo_phenotype_nodes": prepared["hpo_phenotype_nodes"],
        "phenotype_counts_by_kind": {
            kind: {"requests": len(counts), "min": min(counts),
                   "median": statistics.median(counts), "max": max(counts)}
            for kind, counts in sorted(by_kind.items())
        },
    }


# =============================================================================
# HTTP, to loopback only
# =============================================================================
#: No proxy. An operator's HTTP_PROXY would otherwise route loopback requests
#: through it.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

#: Nothing, or not a whole response, came back: refused, timed out, reset, or
#: cut off mid-body (`http.client.IncompleteRead` is not an `OSError`). Unlike an
#: HTTP error status, this does not say the server has finished the request.
NO_RESPONSE = (OSError, http.client.HTTPException)


def http_json(method: str, url: str, body: Any = None,
              timeout: float = REQUEST_TIMEOUT_S) -> tuple:
    """Return (status, parsed JSON or None, response text). Raises one of
    `NO_RESPONSE` when no whole response came back."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            text = response.read().decode(errors="replace")
            status = response.status
    except urllib.error.HTTPError as error:
        text = error.read().decode(errors="replace")
        status = error.code
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    return status, parsed, text


def readiness(status: dict[str, Any]) -> dict[str, Any]:
    """The readiness fields, and whether they assert readiness."""
    fields = {name: bool(status.get(name)) for name in READY_TRUE_FIELDS}
    fields["sp_kg_binding"] = status.get("sp_kg_binding")
    fields["scoring_mode"] = status.get("scoring_mode")
    fields["sp_max_hops"] = status.get("sp_max_hops")
    fields["kg_nodes"] = status.get("kg_nodes")
    fields["kg_edges"] = status.get("kg_edges")
    fields["fingerprint_warning_count"] = len(status.get("fingerprint_warnings") or [])
    meta = status.get("checkpoint_meta")
    fields["serving_device"] = meta.get("device") if isinstance(meta, dict) else None
    fields["asserted"] = (
        all(fields[name] for name in READY_TRUE_FIELDS)
        and fields["sp_kg_binding"] == READY_BINDING
        and fields["scoring_mode"] == READY_SCORING_MODE
        and str(fields["serving_device"]).startswith(READY_DEVICE_PREFIX)
    )
    return fields


def run_workload(base: str, requests: Sequence[dict[str, Any]],
                 defaults: dict[str, Any], schema, alive: Callable[[], bool]) -> dict[str, Any]:
    """Send the requests serially. A request counts only if it returns 200 with
    a body the response model accepts; everything else is a failure, by class.

    **A request with no whole response ends the workload.** The client stopped
    waiting; the server may still be working on it. §7.3 never pauses the
    server while a request is in flight, and nothing here can tell when this
    one ends, so the caller ends the repeat rather than reset. An HTTP error
    status is different: that request is finished, and the workload goes on.
    """
    accepted, sent = 0, 0
    failures: dict[str, int] = {}
    unknown = False

    def fail(kind: str) -> None:
        failures[kind] = failures.get(kind, 0) + 1

    for request in requests:
        if not alive():
            fail("server_exited")
            break
        payload = {"phenotypes": request["phenotypes"], **defaults}
        sent += 1
        try:
            status, body, text = http_json("POST", f"{base}/api/v1/diagnose", payload,
                                           timeout=REQUEST_TIMEOUT_S)
        except NO_RESPONSE:
            fail("no_response")
            unknown = True
            break
        if status != 200:
            fail(f"http_{status}")
            if "out of memory" in text.lower():
                fail("out_of_memory_in_body")
            continue
        try:
            schema.DiagnoseResponse.model_validate(body)
        except Exception:  # noqa: BLE001 -- any refusal by the model is a failure
            fail("schema_rejected")
            continue
        accepted += 1
    return {"planned": len(requests), "sent": sent, "accepted": accepted,
            "failures": failures, "ended_without_response": unknown}


# =============================================================================
# Starting and stopping the service
# =============================================================================
def child_env(base: dict[str, str], workspace: Path, checkpoint: Path) -> dict[str, str]:
    env = {k: v for k, v in base.items()
           if k not in CHILD_ENV_REMOVED and not k.startswith(CHILD_ENV_REMOVED_PREFIX)}
    env.update({
        "PYTHONUNBUFFERED": "1",   # as the unit sets it
        "SHEPHERD_KG_PATH": str(workspace / "kg.json"),
        "SHEPHERD_DATA_DIR": str(workspace),
        "SHEPHERD_CHECKPOINT_PATH": str(checkpoint),
        "SHEPHERD_DEVICE": "cuda",
    })
    return env


def launch_command(python: str, port: int) -> list[str]:
    return [python, str(LAUNCHER), "--no-auto-install", "--no-browser",
            "--", "--host", "127.0.0.1", "--port", str(port)]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stop_server(process: subprocess.Popen) -> int | None:
    """SIGTERM, then SIGKILL if it outlives the timeout; the exit status."""
    if process.poll() is None:
        process.send_signal(signal.SIGCONT)  # never leave it stopped
        process.terminate()
        try:
            process.wait(STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    return process.returncode


# =============================================================================
# One repeat: R0 to R4 in one service process
# =============================================================================
def _delta(value: float | None, base: float | None) -> float | None:
    return None if value is None or base is None else value - base


def _window(sampler: SystemSampler, t0: float, t1: float,
            base: dict[str, float], oom_before: int | None) -> dict[str, Any]:
    """System and swap over one phase, raw and less R0.

    The value at the phase's end is a direct reading taken now, and it joins
    the samples in [t0, t1] for the peak.
    """
    now = sampler.sample()
    samples = sampler.window(t0, t1) + [now]
    peak = max(r["system_in_use_bytes"] for r in samples)
    swap_peak = max(r["swap_in_use_bytes"] for r in samples)
    oom_after = read_oom_kills()
    return {
        "wall_seconds": t1 - t0,
        "system_in_use_at_end_bytes": now["system_in_use_bytes"],
        "system_in_use_at_end_less_r0_bytes": _delta(now["system_in_use_bytes"], base["system"]),
        "system_in_use_sampled_peak_bytes": peak,
        "system_in_use_sampled_peak_less_r0_bytes": _delta(peak, base["system"]),
        "swap_baseline_bytes": base["swap"],
        "swap_sampled_peak_bytes": swap_peak,
        "swap_increase_bytes": _delta(swap_peak, base["swap"]),
        "oom_kills_on_machine": _delta(oom_after, oom_before),
    }


def _status(reset: dict[str, Any], alive: bool, *, complete: bool) -> str:
    """Failed if the server died; inconclusive if the reset was not confirmed;
    incomplete if a request or the reload failed; otherwise completed."""
    if not alive or reset["status"] == "failed":
        return "failed"
    if reset["status"] != "confirmed":
        return "inconclusive"
    return "completed" if complete else "incomplete"


def largest_rise(series: Sequence[int]) -> int:
    """The largest amount any value exceeds the lowest value before it; 0 if the
    series never rises. A rise that falls back again still counts."""
    lowest, rise = series[0], 0
    for value in series[1:]:
        rise = max(rise, value - lowest)
        lowest = min(lowest, value)
    return rise


def r0_fields(samples: Sequence[dict[str, int]], seconds: float) -> tuple:
    """R0's baseline, and whether swap grew at any point while it was sampled.

    §7.3: if swap grows during R0, the run is recorded as *measurement
    precondition not met*. Judged over every sample, not first against last, so
    swap that rose and came back down still fails it. The net change is
    recorded beside it, and is not the criterion.
    """
    system = [r["system_in_use_bytes"] for r in samples]
    swap = [r["swap_in_use_bytes"] for r in samples]
    base = {"system": statistics.median(system), "swap": statistics.median(swap)}
    rise = largest_rise(swap)
    return base, {
        "samples": len(samples),
        "seconds": seconds,
        "system_in_use_median_bytes": base["system"],
        "system_in_use_min_bytes": min(system),
        "system_in_use_max_bytes": max(system),
        "swap_in_use_median_bytes": base["swap"],
        "swap_in_use_first_bytes": swap[0],
        "swap_in_use_peak_bytes": max(swap),
        "swap_net_change_bytes": swap[-1] - swap[0],
        "swap_largest_rise_bytes": rise,
        "measurement_precondition_met": rise == 0,
    }


def _measure_r0(sampler: SystemSampler, seconds: float) -> tuple:
    t0 = time.monotonic()
    sampler.sample()
    time.sleep(seconds)
    sampler.sample()
    return r0_fields(sampler.window(t0, time.monotonic()), seconds)


def _wait_ready(process: subprocess.Popen, base_url: str,
                timeout_s: float, t_launch: float) -> tuple:
    """Poll the status endpoint until it answers.

    uvicorn binds only after the lifespan has built the pipeline, so a refused
    connection means "still starting" and the first answer is final: a pipeline
    that failed to build answers too, with readiness false.
    """
    while True:
        if process.poll() is not None:
            return None, {"status": "failed", "reason": "server exited before answering",
                          "exit_status": process.returncode}
        if time.monotonic() - t_launch > timeout_s:
            return None, {"status": "failed", "reason": f"no answer within {timeout_s:g} s"}
        try:
            code, body, _ = http_json("GET", f"{base_url}/api/v1/pipeline/status", timeout=10)
        except NO_RESPONSE:
            time.sleep(READY_POLL_S)
            continue
        if code == 200 and isinstance(body, dict):
            return body, None
        return None, {"status": "failed", "reason": f"status endpoint answered {code}"}


def run_repeat(
    *,
    index: int,
    workspace: Path,
    checkpoint: Path,
    workload: Sequence[dict[str, Any]],
    defaults: dict[str, Any],
    schema,
    r0_seconds: float,
    ready_timeout_s: float,
    log_dir: Path,
    command: Callable[[int], list[str]] | None = None,
    server_check: Callable[[int], bool] = is_uvicorn_server,
) -> dict[str, Any]:
    """R0 to R4 in one fresh service process; what was read, by phase.

    An error the procedure has no rule for is recorded by its type alone --
    its message may carry a path -- and the repeat ends there.
    """
    command = command or (lambda port: launch_command(sys.executable, port))
    result: dict[str, Any] = {"repeat": index, "phases": {}}
    phases = result["phases"]
    sampler = SystemSampler().start()
    process: subprocess.Popen | None = None
    try:
        # ---------------------------------------------------------------- R0
        base, phases["R0"] = _measure_r0(sampler, r0_seconds)

        # ---------------------------------------------------------------- R1
        port = free_port()
        base_url = f"http://127.0.0.1:{port}"
        oom = read_oom_kills()
        t_launch = time.monotonic()
        with open(log_dir / f"server-repeat-{index}.log", "wb") as log:
            process = subprocess.Popen(
                command(port), cwd=str(REPO_ROOT),
                env=child_env(dict(os.environ), workspace, checkpoint),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid = process.pid

        def alive() -> bool:
            return process.poll() is None

        status, refused = _wait_ready(process, base_url, ready_timeout_s, t_launch)
        if refused:
            # What the machine went through up to here is kept: a start that
            # ran out of memory is the case these counters exist for.
            r1 = {**_window(sampler, t_launch, time.monotonic(), base, oom), **refused}
            try:
                r1.update(read_status(pid))
                r1["server_is_measured_pid"] = server_check(pid)
            except (ProcessLookupError, FileNotFoundError):
                r1.update(vm_hwm_bytes=None, vm_rss_bytes=None,
                          process_counters="unavailable: the process had exited")
            phases["R1"] = r1
            return result
        t_ready = time.monotonic()
        counters = read_status(pid)
        r1: dict[str, Any] = {**_window(sampler, t_launch, t_ready, base, oom), **counters}
        r1["system_in_use_at_end_less_r0_minus_vm_rss_bytes"] = _delta(
            r1["system_in_use_at_end_less_r0_bytes"], counters["vm_rss_bytes"])
        r1["readiness"] = readiness(status)
        r1["serves_named_checkpoint"] = status.get("current_checkpoint_path") == str(checkpoint)
        r1["server_is_measured_pid"] = server_check(pid)
        phases["R1"] = r1
        environ = read_environ(pid)
        result["server_environment"] = {n: environ.get(n) for n in RECORDED_SERVER_VARS}
        if not r1["server_is_measured_pid"]:
            r1.update(status="failed",
                      reason="the launcher did not hand over; the PID is not the server")
        elif not r1["readiness"]["asserted"]:
            r1.update(status="failed", reason="readiness not asserted")
        elif not r1["serves_named_checkpoint"]:
            r1.update(status="failed", reason="the service does not serve the named checkpoint")
        else:
            r1["status"] = "completed"
        if r1["status"] != "completed":
            return result

        # ---------------------------------------------------------------- R2
        reset = reset_high_water(pid)
        oom = read_oom_kills()
        t0 = time.monotonic()
        outcome = run_workload(base_url, workload, defaults, schema, alive)
        t1 = time.monotonic()
        r2: dict[str, Any] = {"reset": reset, "workload": outcome,
                              **_window(sampler, t0, t1, base, oom)}
        confirmed = reset["status"] == "confirmed"
        if alive():
            after = read_status(pid)
            r2["vm_rss_after_bytes"] = after["vm_rss_bytes"]
            r2["vm_hwm_bytes"] = after["vm_hwm_bytes"] if confirmed else None
        else:
            r2["exit_status"] = process.returncode
        r2["status"] = _status(reset, alive(),
                               complete=outcome["accepted"] == outcome["planned"])
        phases["R2"] = r2
        if not alive():
            return result
        if outcome["ended_without_response"]:
            r2["status"] = "incomplete"
            r2["reason"] = ("a request ended without a response and may still be running; "
                            "the repeat stops rather than pause the server")
            return result

        # ------------------------------------------------------------- R3/R4
        reset = reset_high_water(pid)
        confirmed = reset["status"] == "confirmed"
        before_hwm = read_status(pid)["vm_hwm_bytes"] if alive() else None
        before_system = sampler.sample()["system_in_use_bytes"]
        oom = read_oom_kills()
        t0 = time.monotonic()
        try:
            code, body, _ = http_json(
                "POST", f"{base_url}/api/v1/pipeline/reload",
                {"data_dir": str(workspace), "checkpoint_path": str(checkpoint),
                 "device": "cuda"},
                timeout=ready_timeout_s)
        except NO_RESPONSE:
            code, body = None, None
        t1 = time.monotonic()
        success = code == 200 and isinstance(body, dict) and body.get("success") is True
        window = _window(sampler, t0, t1, base, oom)
        r3: dict[str, Any] = {"reset": reset, "http_status": code, "success": success,
                              "wall_seconds": t1 - t0}
        r4: dict[str, Any] = {
            "reset": reset,
            "system_in_use_before_bytes": before_system,
            "system_in_use_sampled_peak_bytes": window["system_in_use_sampled_peak_bytes"],
            "system_in_use_sampled_peak_increase_bytes":
                _delta(window["system_in_use_sampled_peak_bytes"], before_system),
            "swap_baseline_bytes": base["swap"],
            "swap_sampled_peak_bytes": window["swap_sampled_peak_bytes"],
            "swap_increase_bytes": window["swap_increase_bytes"],
            "oom_kills_on_machine": window["oom_kills_on_machine"],
        }
        if alive():
            after = read_status(pid)
            r3["vm_rss_after_bytes"] = after["vm_rss_bytes"]
            r4["vm_hwm_before_bytes"] = before_hwm if confirmed else None
            r4["vm_hwm_over_reload_bytes"] = after["vm_hwm_bytes"] if confirmed else None
            r4["vm_hwm_increase_bytes"] = (
                _delta(after["vm_hwm_bytes"], before_hwm) if confirmed else None)
            try:
                _code, status, _ = http_json("GET", f"{base_url}/api/v1/pipeline/status",
                                             timeout=10)
            except NO_RESPONSE:
                status = None
            status = status if isinstance(status, dict) else {}
            r3["readiness"] = readiness(status)
            r3["serves_named_checkpoint"] = (
                status.get("current_checkpoint_path") == str(checkpoint))
        else:
            r3["exit_status"] = r4["exit_status"] = process.returncode
        ready_again = (bool(r3.get("readiness", {}).get("asserted"))
                       and bool(r3.get("serves_named_checkpoint")))
        r3["status"] = _status(reset, alive(), complete=success and ready_again)
        r4["status"] = _status(reset, alive(), complete=success)
        phases["R3"], phases["R4"] = r3, r4
        return result
    except Exception as exc:  # noqa: BLE001 -- recorded by type; see docstring
        result["error"] = type(exc).__name__
        return result
    finally:
        if process is not None:
            result["server_exit_status"] = stop_server(process)
        sampler.stop()


# =============================================================================
# Versions, digests, and the allocator torch reports
# =============================================================================
_PROBE = r"""
import json, torch
out = {"torch": torch.__version__, "torch_cuda": torch.version.cuda}
try:
    torch.zeros(1, device="cuda")
    out["device"] = torch.cuda.get_device_name(0)
    out["allocator_backend"] = torch.cuda.get_allocator_backend()
except Exception as exc:
    out["cuda_error"] = type(exc).__name__
print(json.dumps(out))
"""


def torch_probe(allocator: dict[str, str | None]) -> dict[str, Any]:
    """Ask this torch, in a fresh process, what it makes of these values.

    Run after the readings, with only the allocator variables taken from the
    server's environment, so the backend is torch's own answer rather than an
    assumption about its precedence rules (§7.3). It is a separate process
    given the same values, not the server itself.
    """
    env = {k: v for k, v in os.environ.items() if k not in ALLOCATOR_VARS}
    env.update({k: v for k, v in allocator.items() if v is not None})
    try:
        done = subprocess.run([sys.executable, "-c", _PROBE], env=env, cwd=str(REPO_ROOT),
                              capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"probe_error": type(exc).__name__}
    try:
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"probe_error": f"exit {done.returncode}"}


def allocator_reading(allocator: dict[str, str | None],
                      probe: Callable[[dict[str, str | None]], dict[str, Any]] = torch_probe
                      ) -> tuple:
    """Which variable governs, and the backend torch reports under it; also the
    probe's own answer, which carries the torch and CUDA versions.

    With one variable present, it governs. With both, each is probed alone as
    well as together, and the one whose backend matches governs; if both give
    the same backend, which one torch read cannot be told apart and is not
    claimed. With neither, torch's default applies -- never read as the
    project's preset.
    """
    present = [name for name in ALLOCATOR_VARS if allocator.get(name) is not None]
    together = probe(allocator)
    reading: dict[str, Any] = {"present": present,
                               "backend_reported_by_torch": together.get("allocator_backend")}
    if len(present) == 1:
        reading["governing_variable"] = present[0]
    elif not present:
        reading["governing_variable"] = None
    else:
        alone = {name: probe({name: allocator[name]}).get("allocator_backend")
                 for name in present}
        matches = [n for n in present if alone[n] == together.get("allocator_backend")]
        reading["backend_each_alone"] = alone
        reading["governing_variable"] = matches[0] if len(matches) == 1 else "indistinguishable"
    return reading, together


def code_version() -> dict[str, Any]:
    """The commit the readings ran at, and whether tracked files differ from it."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"),
                "tracked_files_modified": bool(git("status", "--porcelain",
                                                   "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_files_modified": None}


def driver_version() -> str | None:
    """For §7.1's re-take list. nvidia-smi's memory columns are not used."""
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip().splitlines()[0] if done.returncode == 0 and done.stdout else None


# =============================================================================
# Summary across repeats: median and maximum, never a mean (§7.3 "Repeats")
# =============================================================================
SUMMARISED = {
    "R1": ("wall_seconds", "vm_hwm_bytes", "vm_rss_bytes",
           "system_in_use_sampled_peak_less_r0_bytes", "system_in_use_at_end_less_r0_bytes",
           "system_in_use_at_end_less_r0_minus_vm_rss_bytes", "swap_increase_bytes"),
    "R2": ("wall_seconds", "vm_rss_after_bytes", "vm_hwm_bytes",
           "system_in_use_sampled_peak_less_r0_bytes", "swap_increase_bytes"),
    "R3": ("wall_seconds", "vm_rss_after_bytes"),
    "R4": ("vm_hwm_increase_bytes", "system_in_use_sampled_peak_increase_bytes",
           "swap_increase_bytes"),
}


def summarise(repeats: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Over the repeats whose phase completed; `n` says how many that was."""
    summary: dict[str, Any] = {}
    for phase, metrics in SUMMARISED.items():
        done = [r["phases"][phase] for r in repeats
                if r["phases"].get(phase, {}).get("status") == "completed"]
        summary[phase] = {}
        for metric in metrics:
            values = [p[metric] for p in done if p.get(metric) is not None]
            summary[phase][metric] = ({"median": statistics.median(values),
                                       "max": max(values), "n": len(values)}
                                      if values else {"n": 0})
    return summary


def readings_complete(repeats: Sequence[dict[str, Any]], required: int = REPEATS) -> bool:
    """§7.3: complete when all four phases finish with readiness asserted, in
    each of the three repeats, on a machine that met R0's precondition."""
    return len(repeats) == required and all(
        r["phases"].get("R0", {}).get("measurement_precondition_met")
        and all(r["phases"].get(p, {}).get("status") == "completed"
                for p in ("R1", "R2", "R3", "R4"))
        for r in repeats
    )


# =============================================================================
# Entry point
# =============================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PLAN_B04 §7.3: readings 1-4")
    parser.add_argument("--workspace", type=Path, required=True,
                        help="the workspace the service serves (SHEPHERD_DATA_DIR)")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="the designated checkpoint (SHEPHERD_CHECKPOINT_PATH)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing --output; evidence is cited by digest")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help=f"fewer than {REPEATS} is a trial, never complete readings")
    parser.add_argument("--r0-seconds", type=float, default=10.0)
    parser.add_argument("--settle-seconds", type=float, default=15.0,
                        help="wait after a server exits before the next R0")
    parser.add_argument("--ready-timeout", type=float, default=3600.0,
                        help="seconds allowed for a cold start, and for the reload")
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="where the server logs go; never part of the evidence")
    args = parser.parse_args(argv)
    require_linux()
    previous = install_stop_handlers()
    try:
        return _run(args)
    except Stopped as stop:
        signum = stop.args[0]
        print(f"stopped by signal {signum}: any server this run started was resumed "
              "and stopped; no evidence written", file=sys.stderr)
        return 128 + signum
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _run(args: argparse.Namespace) -> int:
    """Everything after argument parsing; `main` owns the stop signals."""
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite to replace it")
    workspace, checkpoint = args.workspace.resolve(), args.checkpoint.resolve()
    for required in (workspace / "kg.json", workspace / "shortest_paths.pt",
                     workspace / "val_samples.json", checkpoint):
        if not required.is_file():
            raise SystemExit(f"missing: {required}")
    log_dir = args.log_dir or Path(tempfile.mkdtemp(prefix="shepherd-readings-"))
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"server logs: {log_dir}", file=sys.stderr)

    schema = load_diagnose_schema()
    defaults, limit = request_defaults(schema), max_phenotypes(schema)
    code, driver = code_version(), driver_version()
    print("digests and workload...", file=sys.stderr)
    subject = {
        "graph_sha256": file_sha256(workspace / "kg.json"),
        "checkpoint_sha256": file_sha256(checkpoint),
        "sp_table_sha256": file_sha256(workspace / "shortest_paths.pt"),
    }
    prepared = build_workload(workspace, args.seed, limit)

    repeats: list[dict[str, Any]] = []
    for index in range(1, args.repeats + 1):
        if index > 1:
            time.sleep(args.settle_seconds)
        print(f"repeat {index}/{args.repeats}...", file=sys.stderr)
        repeats.append(run_repeat(
            index=index, workspace=workspace, checkpoint=checkpoint,
            workload=prepared["requests"], defaults=defaults, schema=schema,
            r0_seconds=args.r0_seconds, ready_timeout_s=args.ready_timeout,
            log_dir=log_dir))
        print(json.dumps({p: v.get("status", "-") for p, v in repeats[-1]["phases"].items()}),
              file=sys.stderr)

    # The allocator the server carried, and what this torch makes of it -- after
    # the readings, so the probe's CUDA context perturbs none of them.
    environments = [r["server_environment"] for r in repeats if r.get("server_environment")]
    allocator: dict[str, Any] = {
        "server_environment_raw": environments[0] if environments else None,
        "same_in_every_repeat": (all(e == environments[0] for e in environments)
                                 if environments else None),
    }
    # Metadata about the readings, never a reason to lose them: whatever fails
    # here is recorded by type, and the readings are written regardless.
    probe: dict[str, Any] = {}
    try:
        if environments:
            allocator["source"] = environments[0].get("SHEPHERD_ALLOC_SOURCE")
            reading, probe = allocator_reading(
                {n: environments[0].get(n) for n in ALLOCATOR_VARS})
            allocator.update(reading)
            allocator["backend_note"] = ("reported by torch in a separate process given "
                                         "the server's allocator values, after the readings")
        else:
            probe = torch_probe({})
    except Exception as exc:  # noqa: BLE001 -- recorded by type; see above
        allocator["reading_error"] = type(exc).__name__

    evidence = {
        "schema": "shepherd.served_readings/1",
        "procedure": "PLAN_B04_PRODUCTIONISATION.md §7.3",
        "script": {"file": "scripts/measure_served_pipeline.py",
                   "sha256": file_sha256(Path(__file__))},
        "code": code,
        "environment": {
            "kernel": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": probe.get("torch"),
            "torch_cuda": probe.get("torch_cuda"),
            "device": probe.get("device"),
            "driver": driver,
        },
        "subject": subject,
        "start": {
            "through": "scripts/launch/shep_launch.py --no-auto-install --no-browser "
                       "-- --host 127.0.0.1 --port <free port>",
            "environment_set": ["PYTHONUNBUFFERED", "SHEPHERD_KG_PATH", "SHEPHERD_DATA_DIR",
                                "SHEPHERD_CHECKPOINT_PATH", "SHEPHERD_DEVICE"],
            "environment_removed_if_present": sorted(CHILD_ENV_REMOVED) + [
                f"{CHILD_ENV_REMOVED_PREFIX}*"],
            "reload_request": "the same workspace and checkpoint, device cuda",
        },
        "allocator": allocator,
        "workload": describe_workload(prepared, args.seed, defaults, limit),
        "constants": {
            "repeats_required": REPEATS, "repeats_run": len(repeats),
            "sample_interval_seconds": SAMPLE_INTERVAL_S, "stop_wait_seconds": STOP_WAIT_S,
            "reset_tries": RESET_TRIES, "r0_seconds": args.r0_seconds,
            "settle_seconds": args.settle_seconds,
            "readiness": {"true": list(READY_TRUE_FIELDS), "sp_kg_binding": READY_BINDING,
                          "scoring_mode": READY_SCORING_MODE,
                          "serving_device_prefix": READY_DEVICE_PREFIX},
        },
        "notes": {
            "system_in_use": "MemTotal - MemAvailable: a whole-machine pressure indicator, "
                             "not the service's usage; it includes every other process",
            "sampled_peak": f"sampled every {SAMPLE_INTERVAL_S:g} s; a shorter excursion "
                            "can be missed",
            "gap": "system in use less R0 minus VmRSS is recorded and not named CUDA usage",
            "rss": "the kernel describes its RSS accounting as approximate",
            "quiet_machine": "no other GPU, training or build job during the run is the "
                             "operator's to ensure; the script checks only swap during R0",
        },
        "repeats": repeats,
        "summary": summarise(repeats),
        "readings_complete": readings_complete(repeats),
        "not_a_capacity_gate": True,
    }
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"readings complete: {evidence['readings_complete']}", file=sys.stderr)
    return 0 if evidence["readings_complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
