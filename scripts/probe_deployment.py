#!/usr/bin/env python3
"""
Adversarial deployment probe
============================
Runs the production pipeline on the machine that will serve it, and then attacks
it: every refusal this codebase claims to make is provoked here, on real files,
on real hardware, and checked for the property that matters — that a refusal
leaves nothing behind.

Module: scripts/probe_deployment.py

Why this exists separately from the test suite:
    The suite runs where there is no GPU. Three claims in this pipeline are
    therefore untested where they are made:

      * that a checkpoint trained and served on CUDA reports metadata the API
        can serialise (a metric logged as a device tensor is the realistic
        hazard, and it only exists on a machine with a device);
      * that reloading builds the candidate while the old pipeline is still
        resident, which is a memory claim nobody has measured;
      * that a refusal on a real filesystem leaves bytes and modification times
        untouched, rather than on tmpfs under pytest.

    Every probe is paired with a control. A refusal that fires for the wrong
    reason is indistinguishable from one that fires for the right one, so each
    group first proves the sound case works.

Output:
    A JSON report. This probe writes under --work-dir and to --report, and
    nowhere else. The report carries no absolute
    paths, no host or operator names, no sample identifiers and no per-disease
    lists (BACKLOG §5.2): every field in it is assembled from a fixed vocabulary
    in this file plus counts, booleans, digests and library version strings.
    Exception text is never copied into it — probes record which pattern matched,
    not the message, because messages carry paths.

Usage:
    python scripts/probe_deployment.py --work-dir /tmp/shepherd_probe \
        --report probe_report.json

    # add the real-data build (needs data/external; minutes, not seconds)
    python scripts/probe_deployment.py --work-dir /tmp/shepherd_probe \
        --report probe_report.json --external-dir data/external

    # only the environment and a real download of MONDO and HPO, in full,
    # from the configured sources into the work directory (no GPU needed)
    python scripts/probe_deployment.py --work-dir /tmp/shepherd_probe \
        --report probe_report.json --download-only

    # contract M2's acceptance: the demo part (phase M) and the real-data part
    # (phase R), on CUDA, the work directory kept as evidence. The real
    # workspace and the reference checkpoint are read, never written. The run
    # also carries contract M4's flow (T1-T6, RT1-RT3), whose cohort registry
    # is made under the work directory, so the work directory must be outside
    # the repository.
    python scripts/probe_deployment.py --m2-acceptance --device cuda \
        --work-dir <empty directory outside the repository> \
        --report <that directory>/m2_report.json \
        --real-workspace data/workspaces/<workspace> \
        --real-config <config.yaml with the established training settings> \
        --real-reference-checkpoint <path to the reference model's .pt file>

    A relative path is taken from the directory the probe is started in: every
    path argument is made absolute once, before any check or phase reads it
    (BACKLOG 49), so the forms above work from the repository root.

Contract M2's acceptance (`--m2-acceptance`), which also carries contract
M3a's checks:
    Phase M, on the demo workspace, after B-E: a valid same-shaped workspace
    over another graph (M1); a run paused after one of two planned epochs and
    resumed for the remaining one, recording its parent and its mixed precision,
    its gradient scaler's state checked right after the restore against the
    parent's (M2); measurement (M3) and serving
    with a verified shortest-path table (M4) of the resumed model; the five
    resume refusals, each leaving no run directory and nothing changed (M5);
    serving (M6) and measurement (M7) refusing a checkpoint over the same-shaped
    workspace; the inventory's verdicts (M8); the finished child extended by one
    epoch at a fixed rate (M9); resuming that finished child refused before
    anything exists (M10).
    Phase R, on the real workspace: it verifies (R1); the settings carry the
    reference's architecture (R2), the output locations an older config.yaml
    names dropped; one epoch of two from scratch, paused (R3), and its remaining
    epoch resumed (R4); measurement (R5) and serving with the table verified,
    and a diagnosis (R6); the inventory refuses the reference and passes the
    new model (R7); the workspace and the reference are unchanged by listing,
    size and modification time, not by content digest (R8), checked after
    every other probe of the phase, M4's included. Every run writes into its
    own run directory under --work-dir.
    The report stays inside --work-dir and replaces no existing file.
    Run as a program, the probe applies the project's allocator rule before
    torch is imported, as train_model.py does, and the report records the
    allocator variables, their source and the backend torch reports.
    The report's `m2_acceptance` says, per part, whether every required probe
    ran and passed; a missing, skipped or failed one leaves it incomplete, and
    the exit status is non-zero until both parts are complete.

Contract M4's flow, carried by the same acceptance run (functional only: not
model quality, and not an institutional test, which waits for real data, its
terms and a pre-declared protocol). Every step goes through its production
entry point, and the registry, the kept reports and the ledgers are all under
--work-dir:
    Phase M, after M11, on the demo workspace and D1's checkpoint: a source
    the probe builds from the demo graph's own HP and MONDO ids, with terms
    that do not map and a disease that is not a node, previewed (nothing
    written) and committed by the printed preview id, its graph the
    workspace's and its counts as built (T1); a copy of the checkpoint
    measured on that version with --record: the report kept with its ranks
    digest, R10 recorded with both scopes' evidence found in the workspace, one
    record built from the report's bytes (T2); the same report registered again
    is already registered and a damaged copy is refused, the ledger unchanged
    (T3); the version imported against M1's same-shaped workspace is refused by
    the measurement before the checkpoint is read (T4); served through the
    reload route, bound to the digest expected (as the dashboard's Load binds
    it), the measured checkpoint's record shown by the read-only view for the
    digest the status reports, uncontested in its version's report store (D5),
    and none for M2's checkpoint served beside the same ledger (T5); the overlap
    audit in registered mode (T6).
    Phase R, after R7: N of the real workspace's val cases written back as a
    source through the real graph's ids, imported with no exclusions (RT1);
    the checkpoint R4 trained measured on it with --record, training disease
    overlap none and validation phenotype-set overlap N of N, both as
    constructed (RT2); that record shown for that checkpoint served,
    uncontested (RT3).
    The probe's registry is the deployment's (`SHEPHERD_REGISTRY`) for each
    flow as a whole, set once around T1-T6 and RT1-RT3 whatever the operator's
    shell names, and put back after (`_registry_setting`; root-cause round,
    INT-1).

Contract M3c (what is served is identified) adds checks, not probes:
    E1 requires the served identity to name the checkpoint D1 trained, the
    workspace's graph, manifest and dataset record by the digests the probe's
    own reads took, and the device asked for, in effect; E2 that the status
    reports it unchanged; E5 that a refused reload leaves it as it was; E7 that
    a diagnosis carries it. M4 and R6 require the same of a diagnosis served
    with a verified shortest-path table, whose build_id is the sidecar's. The
    identity holds no paths and is recorded in E1's, M4's and R6's facts.

M5's acceptance, carried by the same run where it can run headless (M5-3;
functional only: not the dashboard's layout or a browser session, and no long
training). A5 records the installed packages against the uv.lock beside the
checkout, by name and version, and whether that lock is the one committed at
the tested commit: versions that differ, a dependency the project declares
that is not installed (torch_geometric among them, locked since BACKLOG 50), a
package installed more than once, and the packages outside the lock, the deploy
scripts' (PyG's native extensions, cuVS and what they require) told from the
ones nothing installed requires (an older lock's leftovers). A set that differs (a homelab's, say) is stated in the report and
on the console, never hidden, and said again beside the acceptance's verdict.
    Phase M, after T6, on the probe's own deployment under <work>/m5: a copy of
    the demo workspace under its own workspace root, M2's resumed checkpoint in
    a run there, T1's registry. The server's lists name the workspace, its
    checkpoints and the cohort versions by name and digest and resolve only
    what they list (W1). The service, started as the launcher starts it in its
    own process on a free loopback port, in mode M1 and read-only: D8 allows a
    loopback request in M1 and refuses a forwarding header, a foreign Host, a
    foreign Origin and a cross-site request on every API route its dependency
    gates and on three gated dashboard events (Training's Stop, Test's Start
    and Register), nothing done either way (W2); no Access-Control-Allow-Origin
    for a foreign Origin (W3); the route guard refuses the file= URL forms it
    is sent (ten spellings), none reaching Gradio's URL fetch, counted in this
    process, and process_recording and uploads before the body is read (W4);
    without M1 every gated API route and those events are refused and the
    read-only routes answer (W5). A request off loopback is refused, asked of
    the application in this process (W6). A busy GPU lock refuses the runner
    and both CLIs naming its holder, and a killed holder releases it (W7). A
    test run through the job runner, its inputs found in the server's lists:
    its outcome file, its report kept and not registered as the publication
    contract says, its usage-history entry, a training stop meanwhile refused
    (W8); that report registered from the store, a copy outside it refused
    (W9); a run started for another model or cohort fails before measuring,
    nothing kept (W10); a training job holds the lock, a test start, a test
    cancel and the measurement CLI are refused, and its stop ends it as
    stopped (W11).
    Phase R, after RT3 and before R8, which therefore covers them: the real
    workspace as the server lists it (RW1); a test run through the runner on
    it, on the device, kept and registered (RW2); a training job on it
    started and stopped, other jobs refused while it runs (RW3).
    **W11's and RW3's training job holds the GPU by configuration**, applied
    where it is started (`_held_config`: `_HOLD_EPOCHS` epochs, early stopping
    as long, no pause, a new run), so it ends only by the probe's stop. **One
    bracket covers the whole check**: that the job runs and is the lock's
    recorded holder is checked before a test start is tried and after the
    measurement CLI is, and the answers between are judged only then. The
    runner's record of training, read after each action, says what reached
    it: the first action after which the runner says a stop was asked of it
    fails as reaching training, and a job that ended with no stop asked fails
    naming the action during which it ended and how the runner says it ended
    (a crash and a signal sent outside the runner's stop look alike there);
    never as a pass or a skip (review of 8242b47, W11/RW3; root-cause round,
    W11-1, W11-2, and its verification). A test run that a start not refused
    began is stopped. Telling whether the job still runs needs `os.waitid` (not
    on Windows, nor on macOS before Python 3.13), without which W11 and RW3
    fail saying so before they start a job (W11-4).
    Every M5 setting is the probe's own while M5's probes run
    (`_M5_SETTINGS`), so they never use the deployment's lock, root or
    registry; W1 and RW1 read the lock the operator's settings name once, as
    M5's probes begin, and state whether a job holds it then. **The earlier
    measurement probes** (M3, M7, M11.measurement, T2, T4, R5, RT2) run
    measure_scorer in this process, which refuses without a configured GPU
    lock (P2-4), so phases M and R name a lock of the probe's own, under a
    workspaces root of its own, `<work>/probe-gpu-lock/`
    (`_probes_own_gpu_lock`); they therefore do not exclude a deployment job
    that runs meanwhile. **The training probes take
    no GPU lock at all**, not even the probe's own (D1, M2, M9, R3 and R4
    train; M5, M10 and M11.training are refused before they do): they call
    the production trainer in this process, and `train()` takes no lock, only
    the program entries do (`src/utils/gpu_lock.py`), so they exclude nothing
    either. The deployment's lock is only read, once at W1 and once at RW1, to
    say whether a job holds it at that moment; a job that ran during the
    training probes and ended before is not seen (root-cause round, INT-2, and
    its verification). Run the acceptance while no managed GPU job runs. The
    job runner's own temporary files (a training job's settings file, a test
    run's job directory) are made in the system's temporary directory and
    removed when the job ends.
    Stop a run with Ctrl-C or SIGTERM (``kill <pid>``): the services and the
    jobs it started are stopped on the way out. On Linux its services also end
    with it after a SIGKILL; **a runner job it started then keeps running**,
    and W11's or RW3's training job would never end by itself: stop it by its
    pid, which the GPU lock's holder record (``<lock file>.json``) names, with
    ``kill -INT <pid>``, or end its whole process group with
    ``kill -TERM -- -<pid>`` (the job leads its own group).
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import json
import logging
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

REPORT_SCHEMA_VERSION = 1

#: How this process's CUDA allocator was chosen, set by `_apply_allocator` when
#: this file is the program and before anything imports torch. Empty when
#: `main()` runs in a process that did not set it (a test, say).
ALLOCATOR: Dict[str, Any] = {}


def _apply_allocator() -> Dict[str, Any]:
    """The training entry point's allocator rule, applied before torch exists.

    `train_model.py` applies it only as a program (`__main__`), and the probe
    runs the trainer as a module, so a probe run used PyTorch's default backend
    whatever the saved preset said (M2's acceptance at `82d7573`, 2026-10-14).
    **The same shared rule** (`allocator_env(..., inherit=True)`, as
    `train_model._apply_allocator`): an allocator already in the environment --
    an explicit one, under either variable name, or a backend's preset -- is
    kept; otherwise the saved preset applies. Only the probe's own start-up
    calls this, so importing the module changes no caller's environment.
    """
    import os

    from src.config.runtime_presets import ALLOC_SOURCE_ENV, allocator_env, load_runtime_settings

    torch_first = "torch" in sys.modules
    inherited = "PYTORCH_ALLOC_CONF" in os.environ or "PYTORCH_CUDA_ALLOC_CONF" in os.environ
    resolved = allocator_env(os.environ, load_runtime_settings(), inherit=True)
    for key in ("PYTORCH_ALLOC_CONF", ALLOC_SOURCE_ENV):
        if key in resolved:
            os.environ[key] = resolved[key]
    if not inherited:
        source = "saved preset"
    elif os.environ.get(ALLOC_SOURCE_ENV) == "preset":
        source = "inherited from the backend (preset)"
    else:
        source = "environment (explicit)"
    return {"source": source, "torch_imported_before": torch_first}


def _allocator_record(backend: Optional[str]) -> Dict[str, Any]:
    """The allocator variables in effect, where they came from, and the backend
    torch reports (None when no CUDA device was resolved)."""
    import os

    from src.config.runtime_presets import ALLOC_SOURCE_ENV

    return {
        "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        ALLOC_SOURCE_ENV: os.environ.get(ALLOC_SOURCE_ENV),
        "source": ALLOCATOR.get("source", "not applied (main() called in process)"),
        "torch_imported_before": ALLOCATOR.get("torch_imported_before"),
        "backend": backend,
    }

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("probe")

# The demo graph allocates 3 training and 1 validation disease at f=0.2, and is
# the only graph this probe builds unless --external-dir is given.
FEATURE_DIM = 32
NUM_TRAIN = 60
NUM_VAL = 20
VAL_FRACTION = 0.2
SEED = 20260909


# =============================================================================
# Result plumbing
# =============================================================================
@dataclass
class Probe:
    """One check, its verdict, and nothing that could carry a path into a file."""

    probe_id: str
    phase: str
    claim: str
    status: str = "skipped"          # passed | failed | skipped | error
    detail: str = ""                 # this file's words, never an exception's
    facts: Dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0


class Report:
    def __init__(self) -> None:
        self.probes: List[Probe] = []

    def run(
        self, probe_id: str, phase: str, claim: str,
        fn: Callable[[], Tuple[str, Dict[str, Any]]],
    ) -> Probe:
        probe = Probe(probe_id=probe_id, phase=phase, claim=claim)
        started = time.time()
        try:
            detail, facts = fn()
            probe.status = "passed"
            probe.detail = detail
            probe.facts = facts
        except SkipProbe as skip:
            probe.status = "skipped"
            probe.detail = str(skip)
        except AssertionError as failure:
            probe.status = "failed"
            # Assertion text is written in this file, so it carries no paths.
            probe.detail = str(failure)
            # A `ProbeFailure` says *how* it failed in the same bounded
            # vocabulary a pass uses; the category is the evidence.
            probe.facts = dict(getattr(failure, "facts", {}) or {})
        except Exception:  # noqa: BLE001 — a probe crash is a result, not a stop
            probe.status = "error"
            probe.detail = "the probe itself raised; see the console traceback"
            traceback.print_exc()
        probe.seconds = round(time.time() - started, 3)
        self.probes.append(probe)
        mark = {"passed": "PASS", "failed": "FAIL",
                "skipped": "skip", "error": "ERR "}[probe.status]
        print(f"  [{mark}] {probe.probe_id:<6} {probe.claim}")
        if probe.status in ("failed", "error"):
            print(f"         -> {probe.detail}")
        return probe

    def summary(self) -> Dict[str, int]:
        counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
        for probe in self.probes:
            counts[probe.status] += 1
        return counts


class SkipProbe(Exception):
    """This probe cannot run here, and that is not a failure."""


class ProbeFailure(AssertionError):
    """A failure that carries facts: which category, not which message."""

    def __init__(self, detail: str, facts: Dict[str, Any]):
        super().__init__(detail)
        self.facts = facts


# =============================================================================
# Filesystem snapshots — the property every refusal has to satisfy
# =============================================================================
def snapshot(
    directory: Path, *, include_dirs: bool = False
) -> Dict[str, Tuple[int, str]]:
    """Every file under `directory`, by modification time and content digest.

    Both, deliberately. Bytes alone miss a rewrite with identical content;
    modification time alone is a weaker claim than "the same file".

    `include_dirs` adds every directory, by presence alone, so a refusal that
    leaves an empty directory behind is seen; a file listing cannot see one.
    Opt-in, because the download phase snapshots an operator's directories and
    only claims its files.
    """
    from src.utils.fingerprint import file_sha256

    if not directory.exists():
        return {}
    taken: Dict[str, Tuple[int, str]] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            taken[str(path.relative_to(directory))] = (path.stat().st_mtime_ns, file_sha256(path))
        elif include_dirs and path.is_dir():
            taken[str(path.relative_to(directory)) + "/"] = (0, "<directory>")
    return taken


def assert_untouched(
    before: Dict[str, Tuple[int, str]], directory: Path, what: str,
    *, include_dirs: bool = False,
) -> None:
    after = snapshot(directory, include_dirs=include_dirs)
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    assert not added, f"{what} created {added}"
    assert not removed, f"{what} removed {removed}"
    assert not changed, f"{what} rewrote {changed}"


def refuses(
    fn: Callable[[], Any], expected: type, fragment: Union[str, Tuple[str, ...]]
) -> BaseException:
    """Call `fn`, require it to raise `expected` with `fragment` in the message;
    a tuple of fragments requires every one of them. Returns the refusal, for a
    caller that records what it carries besides its message.

    The fragment is checked here and discarded here. It never reaches the
    report, because refusal messages name the paths they refused.
    """
    fragments = (fragment,) if isinstance(fragment, str) else tuple(fragment)
    try:
        fn()
    except expected as exc:
        missing = [part for part in fragments if part not in str(exc)]
        assert not missing, (
            f"refused, but not for the expected reason ({missing[0]!r})"
        )
        return exc
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"raised {type(exc).__name__} rather than {expected.__name__}"
        ) from None
    raise AssertionError(f"accepted an input that should have been refused ({fragment!r})")


def verify_workspace(directory: Path):
    """The workspace checks a graph consumer makes, from one read of each file.

    Contract M2.1: the manifest, `kg.json`, the graph tensors and both generated
    cohorts are each read once through the shared readers, and the read-form
    verifiers compare those reads with that one manifest reading — `kg.json` as
    serving checks it (`verify_graph_source_read`, on the identity of the bytes
    the graph was parsed from) and the cohorts as training and measurement check
    them (`verify_cohort_reads`). The dataset record is read once and checked
    against the same manifest reading and graph (`verify_dataset_record`,
    contract M3b). Nothing is hashed by path. Returns the manifest-bound digests
    of the four graph roles, the verified cohorts, the digest of each file as
    these reads returned it, by file name, and the verified dataset record.
    """
    from src.evaluation.cohort import GENERATED_SPLITS, resolve_cohort, verify_cohort_reads
    from src.kg.artifacts import (
        GRAPH_ARTIFACTS,
        read_split_manifest,
        verify_dataset_record,
        verify_graph_source_read,
    )
    from src.kg.graph import KnowledgeGraph
    from src.kg.storage.file_storage import read_graph_artifacts, read_samples

    for split in GENERATED_SPLITS:
        resolve_cohort(directory, split)
    manifest = read_split_manifest(directory)
    kg_path = directory / GRAPH_ARTIFACTS["kg"]
    if not kg_path.is_file():
        raise ValueError(
            f"{kg_path} does not exist, and {manifest.identity.path} records a kg artifact "
            "for this workspace"
        )
    _, graph = KnowledgeGraph.read_json(kg_path)
    tensors = read_graph_artifacts(directory, map_location="cpu")
    samples = {split: read_samples(directory, split) for split in GENERATED_SPLITS}
    bound = verify_graph_source_read(manifest, graph, tensors.reads)
    record = verify_dataset_record(manifest, bound["kg"])
    cohorts = verify_cohort_reads(manifest, samples)
    reads = [manifest.identity, graph, *tensors.reads.values(),
             *(read.identity for read in samples.values()), record.identity]
    return bound, cohorts, {Path(read.path).name: read.sha256 for read in reads}, record


def _kg_and_record(directory: Path) -> Tuple[str, Any]:
    """The `kg` digest and the dataset record the workspace check verifies."""
    bound, _, _, record = verify_workspace(directory)
    return bound["kg"], record


def _record_facts(record) -> Dict[str, Any]:
    """A verified dataset record as a report shows it (contract M3b): binding
    state and origin apart, each source by role with its digest and declared
    version (basenames only), and what the record says it does not establish."""
    summary = record.summary()
    return {
        "binding": summary["binding"],
        "origin": summary["origin"],
        "sha256": summary["sha256"],
        "sources": {entry["role"]: {"digest": entry["digest"],
                                    "declared_version": entry.get("declared_version")}
                    for entry in summary["sources"]},
        "incomplete_by_design": summary["incomplete_by_design"],
    }


def _sidecar_build_id(directory: Path) -> Optional[str]:
    """The `build_id` the workspace's shortest-path sidecar declares, through
    the sidecar's own reader and binding check (as R1 reads it); None when the
    workspace holds no table. The tensor is not read here: that the table and
    its sidecar are one pair is the service loader's verdict."""
    from src.inference.sp_artifact import (
        TENSOR_BUILD_ID_KEY,
        read_binding,
        read_sidecar,
        sidecar_path,
    )

    table = directory / "shortest_paths.pt"
    if not table.exists():
        return None
    assert sidecar_path(table).is_file(), "the workspace's table has no sidecar"
    binding = read_binding(read_sidecar(table).meta, (TENSOR_BUILD_ID_KEY,), source=table.name)
    assert binding.recorded, "the table's sidecar records no source binding"
    return binding.build_id


def _assert_served_identity(
    identity: Any,
    *,
    checkpoint_sha256: str,
    workspace_digests: Dict[str, str],
    record: Any,
    sp_build_id: Optional[str],
    device: str,
) -> Dict[str, Any]:
    """What serving says it served is what this probe verified (contract M3c).

    `identity` is the served identity in its JSON form, as
    `get_pipeline_config()["served_identity"]` and the API report it. **Every
    expected value comes from a read this probe already took**: the
    checkpoint's digest from `_read_run_checkpoint`, the `kg.json` and
    `split_manifest.json` digests from `verify_workspace`'s reads, the dataset
    record that check verified, the `build_id` the table's sidecar declares
    (None: the workspace has no table), and the device the probe asked for.
    Nothing is hashed again here. Returns the identity for the report: states,
    digests, identifiers and settings, and no paths.
    """
    from src.kg.artifacts import GRAPH_ARTIFACTS, MANIFEST_FILENAME

    assert isinstance(identity, dict), "serving reports no served identity"
    checkpoint = identity.get("checkpoint") or {}
    assert checkpoint == {"state": "verified", "sha256": checkpoint_sha256}, (
        f"the served identity names a checkpoint {checkpoint.get('state')} with digest "
        f"{str(checkpoint.get('sha256'))[:12]}..., not the checkpoint this probe read "
        f"({checkpoint_sha256[:12]}...)"
    )
    expected_graph = {
        "state": "verified",
        "kg_sha256": workspace_digests[GRAPH_ARTIFACTS["kg"]],
        "split_manifest_sha256": workspace_digests[MANIFEST_FILENAME],
    }
    graph = identity.get("graph") or {}
    assert graph == expected_graph, (
        f"the served identity names a graph {graph.get('state')} (kg "
        f"{str(graph.get('kg_sha256'))[:12]}..., manifest "
        f"{str(graph.get('split_manifest_sha256'))[:12]}...), not the workspace this "
        "probe verified"
    )
    summary = record.summary()
    expected_record = {"state": "verified", "origin": summary["origin"],
                       "sha256": summary["sha256"]}
    dataset_record = identity.get("dataset_record") or {}
    assert dataset_record == expected_record, (
        f"the served identity names a dataset record {dataset_record.get('state')}, "
        f"origin {dataset_record.get('origin')}, not the record the workspace check "
        f"verified (origin {summary['origin']})"
    )
    shortest_paths = identity.get("shortest_paths") or {}
    if sp_build_id is None:
        expected_mode = "gnn_only"
        assert shortest_paths == {"state": "absent", "build_id": None, "max_hops": None}, (
            f"the served identity names a shortest-path table {shortest_paths.get('state')} "
            "for a workspace that holds none"
        )
    else:
        expected_mode = "gnn_plus_shortest_path"
        assert (shortest_paths.get("state"), shortest_paths.get("build_id")) == (
            "verified", sp_build_id), (
            f"the served identity names a shortest-path table {shortest_paths.get('state')} "
            f"with build_id {str(shortest_paths.get('build_id'))[:12]}..., not the table "
            f"whose sidecar declares {sp_build_id[:12]}..."
        )
    assert identity.get("scoring_mode") == expected_mode, (
        f"the served identity's scoring mode is {identity.get('scoring_mode')}, not "
        f"{expected_mode}"
    )
    settings = identity.get("settings") or {}
    # The device in effect is torch's name for where the parameters are
    # ("cuda:0"); the one asked for is recorded as asked.
    assert (str(settings.get("device")).split(":")[0], settings.get("device_requested")) == (
        device, device), (
        f"the served identity records device {settings.get('device')} in effect and "
        f"{settings.get('device_requested')} requested, not the requested {device}"
    )
    return dict(identity)


# =============================================================================
# Phase A — what this machine is
# =============================================================================
def phase_environment(
    report: Report, device: str, *, describe_only: bool = False
) -> Dict[str, Any]:
    """What this machine is, and — unless `describe_only` — whether its device
    serves: A2 resolves it, A3 records its architectures, A4 launches kernels.

    `describe_only` is for runs whose verdict is about something else. A
    download check that launched kernels was failed by a GPU that could not run
    this build, with both ontologies published; A1 alone describes the stack
    and never raises.
    """
    print("\nA. Environment" + (" (description only)" if describe_only else ""))
    facts: Dict[str, Any] = {}

    def _runtime() -> Tuple[str, Dict[str, Any]]:
        from src.utils.version_checker import probe_runtime

        runtime = probe_runtime(force=True)
        torch_facts = runtime.get("torch", {})
        collected = {
            "python": runtime.get("python"),
            "platform": runtime.get("platform"),
            "torch_version": torch_facts.get("version"),
            "torch_cuda_build": torch_facts.get("cuda_build"),
            "cuda_available": torch_facts.get("cuda_available"),
            "device_name": torch_facts.get("device_name"),
            "torch_geometric": runtime.get("torch_geometric", {}).get("version"),
            "pyg_native_present": sorted(
                name for name, status in runtime.get("pyg_native", {}).items()
                if status.get("available")
            ),
            "runtime_status": runtime.get("status"),
        }
        facts.update(collected)
        return "runtime described", collected

    report.run("A1", "environment", "the runtime describes itself", _runtime)

    def _against_lock() -> Tuple[str, Dict[str, Any]]:
        collected = installed_against_lock(project_root / "uv.lock")
        collected["temporary_directory"] = _socket_room()
        facts["installed_against_lock"] = collected
        # **Stated, never hidden** (M5-3): a homelab whose packages moved away
        # from the committed lock says so on the console too, here and again
        # beside the acceptance's verdict at the end (`environment_verdict`).
        print(f"         {lock_sentence(collected)}")
        if collected["temporary_directory"]["socket_fits"] is False:
            print("         a DataLoader worker's socket cannot be made under the temporary "
                  "directory (its path is too long; see A5's facts): training with workers "
                  "fails here; use a short TMPDIR, such as TMPDIR=/tmp")
        differ, outside = collected["version_differs"], collected["installed_not_in_lock"]
        missing = collected.get("declared_not_installed") or []
        said = ("the installed versions are the lock's" if not differ and not missing else
                "; ".join(text for text in (
                    f"{len(differ)} installed version(s) differ from the lock" if differ else "",
                    f"{len(missing)} declared dependency(ies) not installed" if missing else "",
                ) if text))
        return f"{said}; {len(outside)} installed package(s) outside the lock", collected

    def _environment_record() -> None:
        report.run("A5", "environment",
                   "the installed packages are recorded against the lock "
                   "(informational)", _against_lock)

    if describe_only:
        _environment_record()
        return facts

    def _device() -> Tuple[str, Dict[str, Any]]:
        import torch

        resolved = device
        if resolved == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        if resolved == "cuda":
            assert torch.cuda.is_available(), (
                "cuda was requested but torch reports no device; this project "
                "requires CUDA and the probe will not silently fall back"
            )
            capability = torch.cuda.get_device_capability(0)
            collected = {
                "resolved_device": resolved,
                "device_count": torch.cuda.device_count(),
                "capability": f"{capability[0]}.{capability[1]}",
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 1
                ),
                # What this process actually runs, not what a marker claims.
                "allocator_backend": torch.cuda.get_allocator_backend(),
            }
        else:
            collected = {"resolved_device": resolved}
        facts.update(collected)
        return f"serving device is {resolved}", collected

    report.run("A2", "environment", "the serving device resolves", _device)

    def _arch_support() -> Tuple[str, Dict[str, Any]]:
        """What this torch build can emit, against what this device is.

        **Facts, and deliberately not a verdict.** On GB10 (sm_121) this probe
        will report "no native kernels" on every run this project will ever
        make: NVIDIA ships no sm_121-specific kernels, so no PyTorch build has
        any, and the device runs through the compatibility path permanently.
        That is upstream's settled position, not a deployment defect and not
        something to chase.

        Recording it as a status would therefore be a light that is always on,
        which is a light nobody reads. A4 carries the verdict — whether kernels
        actually execute and give the right answers — and this exists to make
        A4's result explicable rather than to grade the machine.
        """
        import torch

        if facts.get("resolved_device") != "cuda":
            raise SkipProbe("no device to compare against")
        arch_list = list(torch.cuda.get_arch_list())
        capability = facts.get("capability", "")
        native = f"sm_{capability.replace('.', '')}" in arch_list
        ptx = [name for name in arch_list if name.startswith("compute_")]
        collected = {
            "arch_list": arch_list,
            "device_capability": capability,
            "natively_compiled_for_this_device": native,
            "native_kernels_expected": None,  # set below: a judgement, not a probe
            "ptx_available": ptx,
        }
        # A device newer than anything the build names is the ordinary case for
        # recent hardware, and stays that way until the vendor ships kernels for
        # it. Saying so in the artifact keeps a later reader from re-opening it.
        collected["native_kernels_expected"] = bool(arch_list) and native
        facts.update({"cuda_arch_support": collected})
        note = (
            "natively compiled for this device" if native else
            "no native kernels for this device — the expected, permanent state "
            "for a capability the vendor ships none for; A4 is what decides "
            "whether that matters here"
        )
        return note, collected

    report.run(
        "A3", "environment",
        "the build's architectures are recorded (informational)", _arch_support,
    )

    def _kernel_runs() -> Tuple[str, Dict[str, Any]]:
        """**Availability is not usability.** `torch.cuda.is_available()` is
        true for a device this build has no kernel image for; the failure
        arrives at the first launch. This launches one and checks the answer.
        """
        import torch

        if facts.get("resolved_device") != "cuda":
            raise SkipProbe("no device to execute on")
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        left = torch.randn(256, 256, generator=generator)
        right = torch.randn(256, 256, generator=generator)
        expected = left @ right
        observed = (left.cuda() @ right.cuda()).cpu()
        assert torch.allclose(expected, observed, atol=1e-3), (
            "a matrix product on the device disagreed with the same product on "
            "the host: this build's kernels are not running correctly here"
        )
        index = torch.tensor([0, 1, 1, 2], device="cuda")
        source = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
        gathered = torch.zeros(3, device="cuda").scatter_add_(0, index, source)
        assert torch.allclose(
            gathered.cpu(), torch.tensor([1.0, 5.0, 4.0]), atol=1e-5
        ), "scatter_add on the device gave the wrong answer"
        return "kernels execute on the device and agree with the host", {
            "matmul_max_abs_error": round(float((expected - observed).abs().max()), 6),
        }

    report.run(
        "A4", "environment",
        "kernels actually execute on this device and give the right answer",
        _kernel_runs,
    )
    _environment_record()
    return facts


#: What a Python environment records of each installed distribution and what
#: `uv.lock` pins, compared by normalised name (PEP 503).
_DISTRIBUTION_NAME = re.compile(r"[-_.]+")


def _normalised(name: str) -> str:
    return _DISTRIBUTION_NAME.sub("-", name).lower()


def _same_version(installed: str, locked: str) -> bool:
    """Equal as PEP 440 versions (``2.8.0.post1`` and ``2.8.0-post1``), or as
    text when either is not one."""
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:  # pragma: no cover - packaging is in the lock
        return installed == locked
    try:
        return Version(installed) == Version(locked)
    except InvalidVersion:
        return installed == locked


#: What the deploy scripts install after ``uv sync``, outside the lock
#: (deploy.sh, deploy.cmd): PyTorch Geometric's native extensions, which come
#: per platform, and cuVS. These and what they require are the expected part of
#: "installed, not in the lock" (`installed_against_lock`). torch_geometric
#: itself is in the lock since BACKLOG 50, compared like any locked package.
DEPLOY_INSTALLED_ROOTS: Tuple[str, ...] = (
    "pyg-lib", "torch-scatter", "torch-sparse", "torch-cluster", "cuvs-cu13",
)
#: Why a package installed outside the lock is there, as A5 can tell.
OUTSIDE_LOCK_DEPLOY = "deploy_scripts"
OUTSIDE_LOCK_REQUIRED = "required_by_an_installed_package"
OUTSIDE_LOCK_LEFTOVER = "required_by_nothing_installed"


def _required_names(requirements: Any) -> set:
    """The normalised names a distribution's ``Requires-Dist`` asks for here:
    a requirement whose marker is false in this environment, or that only an
    extra asks for, is left out."""
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:  # pragma: no cover - packaging is in the lock
        return set()
    names = set()
    for text in requirements or ():
        try:
            requirement = Requirement(str(text))
            if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
                continue
        except (InvalidRequirement, ValueError, TypeError):
            continue
        names.add(_normalised(requirement.name))
    return names


def _lock_at_tested_commit(lock_path: Path, data: bytes) -> Optional[bool]:
    """Whether `data` (the lock as A5 read it) is the uv.lock committed at the
    checkout's HEAD, as git hashes it (its attributes and clean filters
    applied, so a line-ending conversion is no difference). False when the
    commit holds another or none; None when git cannot say (no git, or not a
    checkout)."""
    import subprocess

    def git(*args: str, given: Optional[bytes] = None) -> Optional[str]:
        try:
            done = subprocess.run(["git", *args], cwd=lock_path.parent, input=given,
                                  capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.decode("ascii", "replace").strip() if done.returncode == 0 else None

    if git("rev-parse", "--verify", "--quiet", "HEAD") is None:
        return None
    committed = git("rev-parse", "--verify", "--quiet", f"HEAD:./{lock_path.name}")
    if committed is None:
        return False
    read = git("hash-object", "--stdin", f"--path={lock_path.name}", given=data)
    return None if read is None else read == committed


def _is_this_project(package: Mapping[str, Any]) -> bool:
    """Whether a uv.lock package entry is this project itself (the checkout,
    installed editable or as a virtual root), whose ``dependencies`` are what
    the project declares."""
    source = package.get("source")
    return isinstance(source, dict) and (source.get("editable") == "." or
                                         source.get("virtual") == ".")


def _declared_here(dependencies: Any) -> set:
    """The normalised names of the project's declared dependencies whose
    marker is true in this environment (a dependency without one always is)."""
    try:
        from packaging.markers import InvalidMarker, Marker
    except ImportError:  # pragma: no cover - packaging is in the lock
        return set()
    names = set()
    for entry in dependencies or ():
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        marker = entry.get("marker")
        try:
            if marker and not Marker(str(marker)).evaluate({"extra": ""}):
                continue
        except (InvalidMarker, ValueError, TypeError):
            continue
        names.add(_normalised(str(entry["name"])))
    return names


def installed_against_lock(lock_path: Path) -> Dict[str, Any]:
    """This Python environment's installed distributions against the lock
    beside this checkout (M5-3): what the probe ran with, **stated, never
    hidden**.

    The homelab's check set had moved away from the lock M5-0 committed
    (huggingface-hub 1.33.0, click 8.5.0 and hf-xet 1.7.0 for 1.16.1, 8.3.3 and
    1.5.0, and aiofiles and ffmpy left over from Gradio 5.29.1), so a run there
    tests another set than the one reviewed. This says which, from one read of
    `uv.lock` (its digest recorded) and the distributions `importlib.metadata`
    finds on this interpreter's path:

    - ``lock_is_the_committed_one``: whether the lock read is the one
      committed at the tested commit (`_lock_at_tested_commit`); a lock made
      again on this machine and not committed reads False, so "the installed
      versions are the lock's" is never taken for the committed lock's;
    - ``version_differs``: installed under the lock's name with a version the
      lock does not pin (PEP 440 equality; a package the lock pins at several
      versions, one per platform, matches any of them), one entry per such
      installed version;
    - ``declared_not_installed``: a dependency the project itself declares
      (the lock's entry for this project, its marker true here) that is not
      installed at all, with the version the lock pins: missing is not the
      lock's set either (BACKLOG 50);
    - ``installed_more_than_once``: a name with more than one distribution on
      the path (a stale ``.dist-info`` beside the current one, say): each
      version is compared, and which one ``import`` finds is not told from
      the metadata, so both are stated;
    - ``installed_not_in_lock``: installed and not in the lock at all, each
      with why it is there as far as the installed metadata tells:
      `OUTSIDE_LOCK_DEPLOY` for what the deploy scripts install after the
      sync and what that requires (`DEPLOY_INSTALLED_ROOTS`, followed through
      ``Requires-Dist``), `OUTSIDE_LOCK_REQUIRED` for one another installed
      package requires, and `OUTSIDE_LOCK_LEFTOVER` for one nothing installed
      requires: a leftover of an older lock, or a hand install;
    - ``locked_not_installed``: in the lock and not installed (an extra not
      installed, a package for another platform, and the declared ones above).
      Names only.

    Names and versions only, no paths. Raises `SkipProbe` without a lock or a
    TOML reader to read it with, so an acceptance that requires the record is
    incomplete rather than silent.
    """
    from importlib import metadata

    from src.utils.fingerprint import read_once

    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.11 and later have it
        raise SkipProbe("this Python has no TOML reader to read uv.lock with") from None
    try:
        read = read_once(lock_path)
    except FileNotFoundError:
        raise SkipProbe("there is no uv.lock beside this checkout to compare with") from None
    lock = tomllib.loads(read.data.decode("utf-8"))
    locked: Dict[str, set] = {}
    declared: set = set()
    for package in lock.get("package", []):
        if isinstance(package, dict) and package.get("name") and package.get("version"):
            locked.setdefault(_normalised(str(package["name"])), set()).add(str(package["version"]))
        if isinstance(package, dict) and _is_this_project(package):
            declared |= _declared_here(package.get("dependencies"))
    installed: Dict[str, List[str]] = {}
    requires: Dict[str, set] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        if not name:
            continue
        key = _normalised(name)
        installed.setdefault(key, []).append(str(distribution.version))
        requires.setdefault(key, set()).update(
            _required_names(getattr(distribution, "requires", None)))
    differs = [
        {"name": name, "installed": version, "locked": sorted(locked[name])}
        for name, versions in sorted(installed.items()) if name in locked
        for version in sorted(set(versions))
        if not any(_same_version(version, pin) for pin in locked[name])
    ]
    # What the deploy scripts installed: their roots and, through
    # Requires-Dist, what those require that the lock does not hold (a
    # requirement the lock holds was installed by the sync).
    outside = set(installed) - set(locked)
    deployed: set = set()
    pending = [name for name in DEPLOY_INSTALLED_ROOTS if name in outside]
    while pending:
        name = pending.pop()
        if name not in deployed:
            deployed.add(name)
            pending.extend(sorted(requires.get(name, set()) & outside))
    required = {wanted for name, names in requires.items() for wanted in names if wanted != name}

    def why(name: str) -> str:
        if name in deployed:
            return OUTSIDE_LOCK_DEPLOY
        return OUTSIDE_LOCK_REQUIRED if name in required else OUTSIDE_LOCK_LEFTOVER

    missing = [{"name": name, "locked": sorted(locked.get(name, ()))}
               for name in sorted(declared) if name not in installed]
    return {
        "lock_sha256": read.identity.sha256,
        "lock_is_the_committed_one": _lock_at_tested_commit(lock_path, read.data),
        "lock_packages": len(locked),
        "installed_distributions": len(installed),
        "versions_match_lock": not differs and not missing,
        "version_differs": differs,
        "declared_not_installed": missing,
        "installed_more_than_once": [{"name": name, "versions": sorted(versions)}
                                     for name, versions in sorted(installed.items())
                                     if len(versions) > 1],
        "installed_not_in_lock": [{"name": name, "version": "/".join(sorted(set(versions))),
                                   "why": why(name)}
                                  for name, versions in sorted(installed.items())
                                  if name not in locked],
        "locked_not_installed": sorted(name for name in locked if name not in installed),
    }


def lock_sentence(facts: Mapping[str, Any]) -> str:
    """A5's record (`installed_against_lock`) as one console line: which lock,
    the versions that differ, every package installed more than once, and the
    packages outside the lock by why they are there, the deploy scripts' by
    count and every other by name."""
    committed = facts.get("lock_is_the_committed_one")
    which = {True: "this checkout's uv.lock (the one committed at the tested commit)",
             False: "this checkout's uv.lock, which is NOT the one committed at the tested "
                    "commit"}.get(committed, "this checkout's uv.lock (whether it is the "
                                             "committed one could not be read)")
    differ = facts.get("version_differs") or []
    parts = [f"{len(differ)} version(s) differ" + (
        " (" + ", ".join(f"{d['name']} {d['installed']}, lock {'/'.join(d['locked'])}"
                         for d in differ) + ")" if differ else "")]
    missing = facts.get("declared_not_installed") or []
    if missing:
        parts.append(f"{len(missing)} declared dependenc{'y is' if len(missing) == 1 else 'ies are'}"
                     " not installed (" + ", ".join(
                         f"{m['name']}, lock {'/'.join(m['locked']) or '?'}" for m in missing) + ")")
    twice = facts.get("installed_more_than_once") or []
    if twice:
        parts.append("installed more than once: " + ", ".join(
            f"{t['name']} ({', '.join(t['versions'])})" for t in twice))
    outside = facts.get("installed_not_in_lock") or []
    by_why = {key: [o for o in outside if o.get("why") == key]
              for key in (OUTSIDE_LOCK_DEPLOY, OUTSIDE_LOCK_REQUIRED, OUTSIDE_LOCK_LEFTOVER)}
    said = f"{len(outside)} installed package(s) are not in the lock"
    if outside:
        said += ": " + "; ".join(text for text in (
            f"{len(by_why[OUTSIDE_LOCK_DEPLOY])} from the deploy scripts (PyG's native "
            "extensions, cuVS and what they require)" if by_why[OUTSIDE_LOCK_DEPLOY] else "",
            "required by nothing installed: " + ", ".join(
                f"{o['name']} {o['version']}" for o in by_why[OUTSIDE_LOCK_LEFTOVER])
            if by_why[OUTSIDE_LOCK_LEFTOVER] else "",
            "required by another installed package: " + ", ".join(
                f"{o['name']} {o['version']}" for o in by_why[OUTSIDE_LOCK_REQUIRED])
            if by_why[OUTSIDE_LOCK_REQUIRED] else "",
        ) if text)
    parts.append(said)
    return f"installed packages against {which}: " + "; ".join(parts)


#: Run by `_socket_room` in a fresh interpreter: one AF_UNIX listener opened and
#: closed as `multiprocessing` opens one for DataLoader workers sharing tensors
#: (its resource sharer), under the temporary directory it is given. Whatever
#: that Python does with a long one (3.12 fails; 3.13 falls back to /tmp), this
#: is what it does.
_SOCKET_CHECK = (
    "import json\n"
    "from multiprocessing.connection import Listener\n"
    "try:\n"
    "    Listener(family='AF_UNIX').close()\n"
    "    said = {'fits': True}\n"
    "except OSError as exc:\n"
    "    said = {'fits': False, 'error': type(exc).__name__,\n"
    "            'path_too_long': 'path too long' in str(exc)}\n"
    "print(json.dumps(said))\n"
)


def _socket_room(python: Optional[str] = None) -> Dict[str, Any]:
    """Whether a DataLoader worker's socket can be made under this process's
    temporary directory: under a long one (a scratch path as ``TMPDIR``)
    training with workers fails with ``OSError: AF_UNIX path too long`` on
    Python 3.12 (M5-0's audit), while 3.13 puts its sockets under /tmp
    instead. **Asked, not computed**: `python` (this interpreter by default,
    the one the job runner starts) opens and closes one listener there, in a
    process of its own (`_SOCKET_CHECK`). ``socket_fits`` is None when the
    check could not run. The path's length only, never the path."""
    import subprocess
    import tempfile

    temp = tempfile.gettempdir()
    facts: Dict[str, Any] = {"path_bytes": len(os.fsencode(temp)), "socket_fits": None,
                             "refusal": None}
    if sys.platform == "win32":  # pragma: no cover - no AF_UNIX listener there
        return {**facts, "socket_fits": True, "checked": "not used on Windows"}
    facts["checked"] = ("an AF_UNIX listener opened and closed under it, as multiprocessing "
                        "opens one for DataLoader workers")
    try:
        done = subprocess.run([python or sys.executable, "-I", "-c", _SOCKET_CHECK],
                              env={**os.environ, "TMPDIR": temp}, capture_output=True,
                              text=True, timeout=120)
        said = json.loads((done.stdout.strip().splitlines() or ["{}"])[-1])
    except (OSError, subprocess.SubprocessError, ValueError):
        return {**facts, "checked": "the check could not run"}
    if said.get("fits") is True:
        return {**facts, "socket_fits": True}
    if said.get("fits") is False:
        return {**facts, "socket_fits": False, "refusal": (
            "AF_UNIX path too long" if said.get("path_too_long") else str(said.get("error")))}
    return {**facts, "checked": "the check could not run"}


# =============================================================================
# Phase B — the writer, attacked
# =============================================================================
def _demo_kg():
    from scripts.setup_demo import build_demo_kg

    return build_demo_kg()


def _other_kg():
    """The demo graph plus one disease with one phenotype edge, so its `kg.json`
    and every file of its export are genuinely other: the node changes
    `kg.json`, the features and the node counts, and the edge changes
    `edge_indices.pt`, which a node without edges would leave byte-identical."""
    from src.core.types import DataSource, EdgeType, NodeType
    from src.kg.graph import Edge, Node, NodeID

    kg = _demo_kg()
    disease = NodeID(source=DataSource.MONDO, local_id="MONDO:9999999")
    kg.add_node(Node(
        id=disease,
        node_type=NodeType.DISEASE,
        name="Probe-only disease",
        attributes={"mondo_id": "MONDO:9999999", "name": "Probe-only disease"},
    ))
    kg.add_edge(Edge(
        source_id=NodeID(source=DataSource.HPO, local_id="HP:0000486"),
        target_id=disease,
        edge_type=EdgeType.PHENOTYPE_OF_DISEASE,
    ))
    return kg


def _budget(**overrides):
    from src.kg.workspace import SampleBudget

    base = dict(
        num_train=NUM_TRAIN, num_val=NUM_VAL,
        val_disease_fraction=VAL_FRACTION, seed=SEED,
    )
    base.update(overrides)
    return SampleBudget(**base)


BAD_INPUTS: List[Tuple[str, str, Dict[str, Any], Dict[str, Any], str]] = [
    # probe_id, claim, write_kwargs, budget_kwargs, expected message fragment
    ("B2", "a zero feature width is refused",
     {"feature_dim": 0}, {}, "feature_dim must be >= 1"),
    ("B3", "a fractional feature width is refused",
     {"feature_dim": 1.5}, {}, "feature_dim must be an integer"),
    ("B4", "a zero phenotype floor is refused",
     {}, {"min_phenotypes": 0}, "min_phenotypes must be >= 1"),
    ("B5", "a boolean phenotype floor is refused",
     {}, {"min_phenotypes": True}, "min_phenotypes must be an integer"),
    ("B6", "a bytes seed is refused", {}, {"seed": b"42"}, "seed must be an integer"),
    ("B7", "an object seed is refused", {}, {"seed": object()}, "seed must be an integer"),
    ("B8", "a boolean seed is refused", {}, {"seed": True}, "seed must be an integer"),
    ("B9", "a boolean budget is refused", {}, {"num_train": True}, "must be an integer"),
    ("B10", "a budget too small to cover its partition is refused",
     {}, {"num_train": 1}, "cannot cover"),
    ("B11", "a validation fraction outside (0, 1) is refused",
     {}, {"val_disease_fraction": 1.5}, "val_fraction must be finite"),
    ("B12", "a universe too small to split is refused",
     {}, {"min_phenotypes": 50}, "cannot be cut into two non-empty"),
    # Contract M3b: a record from files that leaves a required input
    # unidentified is refused by every consumer, so it is not written.
    ("B23", "sources leaving a required input unidentified are refused",
     {"sources": [{"role": role, "filename": f"{role}.input", "digest": "0" * 64,
                   "declared_version": None}
                  for role in ("hpo", "phenotype_hpoa", "genes_to_phenotype")]},
     {}, "unidentified"),
]


def phase_writer(report: Report, work: Path) -> Optional[Path]:
    """Every refusal, and the control that keeps them meaningful."""
    print("\nB. The workspace writer, attacked")
    from src.kg.workspace import WorkspaceRefusal, write_workspace

    sound = work / "sound_workspace"

    def _control() -> Tuple[str, Dict[str, Any]]:
        written = write_workspace(
            _demo_kg(), sound, feature_dim=FEATURE_DIM, samples=_budget()
        )
        realised = written.manifest["realised"]
        return "a sound workspace was written", {
            "train_samples": len(written.train_samples),
            "val_samples": len(written.val_samples),
            "train_diseases": realised["train_diseases"],
            "val_diseases": realised["val_diseases"],
            "disjoint": written.manifest["disjoint"],
        }

    control = report.run(
        "B1", "writer", "CONTROL: a sound workspace is written", _control
    )
    if control.status != "passed":
        return None

    for probe_id, claim, write_kwargs, budget_kwargs, fragment in BAD_INPUTS:
        def _probe(write_kwargs=write_kwargs, budget_kwargs=budget_kwargs,
                   fragment=fragment):
            target = work / "refused_workspace"
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)
            (target / "sentinel.json").write_text('{"pre-existing": true}')
            before = snapshot(target)
            refuses(
                lambda: write_workspace(
                    _demo_kg(), target,
                    **{"feature_dim": FEATURE_DIM, **write_kwargs},
                    samples=_budget(**budget_kwargs),
                ),
                WorkspaceRefusal, fragment,
            )
            assert_untouched(before, target, "a refused write")
            shutil.rmtree(target)
            return "refused, and wrote nothing", {}

        report.run(probe_id, "writer", claim, _probe)

    def _fresh_directory() -> Tuple[str, Dict[str, Any]]:
        target = work / "never_created"
        refuses(
            lambda: write_workspace(
                _demo_kg(), target, feature_dim=0, samples=_budget()
            ),
            WorkspaceRefusal, "feature_dim",
        )
        assert not target.exists(), "a refused write created the directory"
        return "no directory was created", {}

    report.run(
        "B13", "writer", "a refusal creates no directory at all", _fresh_directory
    )

    # -------------------------------------------------------------------------
    # The direct persisting path.
    #
    # **Why these are not covered by B2-B12.** Those attack `write_workspace`,
    # which validates everything up front and always supplies a complete recipe
    # and complete digests — so its callers never reach the checks that used to
    # run *after* the cohort files were written. `generate_training_samples` is
    # public and persists on its own; that is where a caller can arrive with a
    # missing digest or a malformed recipe, and where the refusal used to land
    # on a workspace it had already overwritten.
    # -------------------------------------------------------------------------
    def _persisting_call(target: Path, digests=None, recipe=None):
        from src.kg.disease_allocation import allocate_diseases
        from src.kg.graph import (
            DEFAULT_FEATURE_SEED,
            FEATURE_INITIALISATION,
            FEATURE_INITIALISATION_VERSION,
        )
        from src.kg.sample_generator import (
            build_eligible_disease_profiles,
            generate_training_samples,
        )

        kg = _demo_kg()
        sound_digests = {role: "0" * 64 for role in
                         ("kg", "node_features", "edge_indices", "num_nodes", "provenance")}
        sound_recipe = {
            "feature_dim": FEATURE_DIM,
            "feature_seed": DEFAULT_FEATURE_SEED,
            "initialisation": FEATURE_INITIALISATION,
            "initialisation_version": FEATURE_INITIALISATION_VERSION,
        }
        return generate_training_samples(
            kg=kg,
            allocation=allocate_diseases(
                build_eligible_disease_profiles(kg, 2), VAL_FRACTION, seed=SEED
            ),
            num_train=NUM_TRAIN, num_val=NUM_VAL, output_dir=target,
            graph_digests=sound_digests if digests is None else digests,
            graph_export=sound_recipe if recipe is None else recipe,
        )

    def _populated(target: Path) -> Path:
        """A workspace with cohorts and a manifest already in it."""
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for name in ("train_samples.json", "val_samples.json", "split_manifest.json"):
            (target / name).write_text('{"from": "an earlier production event"}')
        return target

    #: Each entry is a persisting call a caller can get wrong, and the fragment
    #: its refusal must carry. The digest cases are shape only — a well-formed
    #: digest can still be another graph's, and what settles that is the
    #: consumer re-hashing the file, not anything checked here.
    DIRECT_REFUSALS = [
        ("B15", "a persisted call with an empty recipe is refused",
         dict(recipe={}), "graph_export"),
        ("B16", "a persisted call with an incomplete recipe is refused",
         dict(recipe={"feature_dim": FEATURE_DIM}), "graph_export"),
        ("B17", "a persisted call with a malformed recipe field is refused",
         dict(recipe={"feature_dim": 0, "feature_seed": 7,
                      "initialisation": "standard-normal",
                      "initialisation_version": 1}), "graph_export"),
        ("B18", "a persisted call missing a graph digest is refused",
         dict(digests={r: "0" * 64 for r in
                       ("kg", "node_features", "edge_indices")}), "unbound"),
        # The regression this campaign exists to guard: a 64-digit integer
        # stringifies to 64 valid hex characters, and the manifest would have
        # recorded a JSON number no consumer could ever match.
        ("B19", "a digest that is a 64-digit integer is refused",
         dict(digests={**{r: "0" * 64 for r in
                          ("kg", "node_features", "edge_indices", "provenance")},
                       "num_nodes": int("1" * 64)}), "not SHA-256"),
        # Contract M3b: the manifest binds the dataset record, or is not written.
        ("B22", "a persisted call that binds no dataset record is refused",
         dict(digests={r: "0" * 64 for r in
                       ("kg", "node_features", "edge_indices", "num_nodes")}),
         ("provenance", "unbound")),
    ]

    for probe_id, claim, kwargs, fragment in DIRECT_REFUSALS:
        def _direct(kwargs=kwargs, fragment=fragment):
            target = _populated(work / "direct_refusal")
            before = snapshot(target)
            refuses(lambda: _persisting_call(target, **kwargs), ValueError, fragment)
            assert_untouched(before, target, "a refused persisting call")
            shutil.rmtree(target)
            return "refused, and the workspace was untouched", {}

        report.run(probe_id, "writer", claim, _direct)

    def _direct_creates_nothing() -> Tuple[str, Dict[str, Any]]:
        """A refusal must not leave the directory it was about to fill."""
        target = work / "direct_never_created"
        refuses(lambda: _persisting_call(target, recipe={}), ValueError, "graph_export")
        assert not target.exists(), "a refused persisting call created the directory"
        return "no directory was created", {}

    report.run(
        "B20", "writer",
        "a refused persisted call creates no directory at all",
        _direct_creates_nothing,
    )

    def _direct_control() -> Tuple[str, Dict[str, Any]]:
        """CONTROL: without it every refusal above holds for a writer that
        refuses everything, and the digest recorded is checked against the file
        actually written -- which is what the serialise-once change had to keep
        true."""
        from src.utils.fingerprint import file_sha256

        target = work / "direct_sound"
        if target.exists():
            shutil.rmtree(target)
        _, _, manifest = _persisting_call(target)
        recorded = manifest["artifacts"]
        observed = {
            role: file_sha256(target / f"{role}.json")
            for role in ("train_samples", "val_samples")
        }
        mismatched = sorted(r for r in observed if observed[r] != recorded[r])
        assert not mismatched, (
            f"the manifest records a digest that is not the file's for {mismatched}"
        )
        shutil.rmtree(target)
        return "it wrote, and its digests are of the bytes on disk", {
            "cohort_digests_verified": len(observed),
        }

    report.run(
        "B21", "writer",
        "CONTROL: a sound persisted call writes, and its digests match the files",
        _direct_control,
    )

    def _checkpoint_guard() -> Tuple[str, Dict[str, Any]]:
        guard = work / "guarded_workspace"
        write_workspace(_demo_kg(), guard, feature_dim=FEATURE_DIM, samples=_budget())
        (guard / "checkpoints" / "gat").mkdir(parents=True, exist_ok=True)
        (guard / "checkpoints" / "gat" / "last.pt").write_bytes(b"trained on this graph")
        before = snapshot(guard)
        refuses(
            lambda: write_workspace(
                _demo_kg(), guard, feature_dim=FEATURE_DIM, samples=_budget()
            ),
            FileExistsError, "already holds trained checkpoints",
        )
        assert_untouched(before, guard, "a refused rebuild")
        return "the rebuild was refused and changed nothing", {}

    report.run(
        "B14", "writer",
        "rebuilding under trained checkpoints is refused and writes nothing",
        _checkpoint_guard,
    )
    return sound


# =============================================================================
# Phase C — the workspace, attacked
# =============================================================================
def phase_workspace(report: Report, work: Path, sound: Path) -> None:
    print("\nC. The workspace contract, attacked")
    from src.kg.artifacts import GRAPH_ARTIFACTS

    def _accepted() -> Tuple[str, Dict[str, Any]]:
        digests, cohorts, read, record = verify_workspace(sound)
        assert cohorts.verified == ("train", "val")
        # **Fully bound, and shown as synthetic** (contract M3b): the demo graph
        # is built in memory, so its record says so and still passes every check.
        assert (record.binding, record.origin) == ("verified", "synthetic"), (
            f"the demo workspace's record is {record.binding}, origin {record.origin}"
        )
        assert cohorts.disjointness_measured and cohorts.disjointness_claim_checked
        overlap = cohorts.disease_sets["train"] & cohorts.disease_sets["val"]
        assert not overlap, "the cohorts share a disease"
        # **The cross-machine control.** The demo graph is written out in
        # Python, so it is the same input everywhere — unlike a real build,
        # whose MONDO vintage differs between two sites deployed days apart.
        # Recording every file's digest here is what lets two reports isolate
        # the machine from the data: identical inputs, and the only remaining
        # variables are the architecture and the library versions.
        #
        # The open question it answers is whether a seeded `torch.Generator`
        # draws the same bytes on two architectures. Nothing in this project
        # asserts that it does; this measures it.
        #
        # The files the checks read are listed with the digest of the bytes they
        # verified; since contract M3b that includes the dataset record, so no
        # file is identified by path.
        return "every verifier accepted it", {
            # Counts and digests only: per-disease lists are forbidden in
            # evidence artifacts, and the identifiers are what those lists are.
            "kg_digest": digests["kg"],
            "workspace_digests": dict(sorted(read.items())),
            "dataset_record": {"binding": record.binding, "origin": record.origin},
            "train_diseases": len(cohorts.disease_sets["train"]),
            "val_diseases": len(cohorts.disease_sets["val"]),
            "disease_overlap": len(overlap),
        }

    report.run(
        "C1", "workspace", "CONTROL: the sound workspace verifies", _accepted
    )

    def _other() -> Path:
        # **A different graph, not the same graph under a different seed.** The
        # first version of C4 built the other workspace with seed+1 and copied
        # its kg.json across -- but the demo graph is deterministic, so the two
        # files were byte-identical and nothing was crossed. It failed loudly,
        # which is the only reason it was caught. The seed moves the
        # allocation; only the graph moves the graph.
        other = work / "other_workspace"
        if not other.exists():
            from src.kg.workspace import write_workspace

            write_workspace(
                _other_kg(), other, feature_dim=FEATURE_DIM, samples=_budget(),
            )
        return other

    for role in sorted(GRAPH_ARTIFACTS):
        def _replaced(role=role):
            # **Another workspace's valid file, not junk bytes.** The checks
            # parse each file before they compare it (contract M2.1), so junk
            # is refused by the reader that could not parse it; a file that
            # parses and is another export's is what only the digest catches.
            other = _other()
            name = GRAPH_ARTIFACTS[role]
            assert (other / name).read_bytes() != (sound / name).read_bytes(), (
                f"the two workspaces hold the same {name}, so this probe replaces nothing"
            )
            copy = work / f"replaced_{role}"
            if copy.exists():
                shutil.rmtree(copy)
            shutil.copytree(sound, copy)
            (copy / name).write_bytes((other / name).read_bytes())
            refuses(
                lambda: verify_workspace(copy), ValueError,
                "is not the graph" if role == "kg" else f"is not the {role} artifact",
            )
            shutil.rmtree(copy)
            return "the replacement was refused", {}

        report.run(
            f"C2.{role}", "workspace",
            f"a replaced {role} artifact is refused", _replaced,
        )

    def _no_manifest() -> Tuple[str, Dict[str, Any]]:
        copy = work / "no_manifest"
        if copy.exists():
            shutil.rmtree(copy)
        shutil.copytree(sound, copy)
        (copy / "split_manifest.json").unlink()
        refuses(
            lambda: verify_workspace(copy), ValueError,
            "has no split_manifest.json",
        )
        shutil.rmtree(copy)
        return "an unbound workspace was refused", {}

    report.run(
        "C3", "workspace", "a workspace with no manifest is refused", _no_manifest
    )

    def _crossed() -> Tuple[str, Dict[str, Any]]:
        other = _other()
        assert (other / "kg.json").read_bytes() != (sound / "kg.json").read_bytes(), (
            "the two workspaces hold the same graph, so this probe crosses nothing"
        )
        copy = work / "crossed"
        if copy.exists():
            shutil.rmtree(copy)
        shutil.copytree(sound, copy)
        (copy / "kg.json").write_bytes((other / "kg.json").read_bytes())
        refuses(lambda: verify_workspace(copy), ValueError, "is not the graph")
        shutil.rmtree(copy)
        return "another workspace's graph was refused", {}

    report.run(
        "C4", "workspace", "another workspace's kg.json is refused", _crossed
    )

    # -------------------------------------------------------------------------
    # The dataset record is required (contract M3b). Each case tampers with a
    # copy of the sound workspace, and each is refused with the binding state
    # it establishes. Where the manifest is rebound to the record written, only
    # the record's content can refuse it.
    # -------------------------------------------------------------------------
    from src.kg.artifacts import DATASET_RECORD_ROLE, DatasetRecordRefused, MANIFEST_FILENAME
    from src.kg.provenance import PROVENANCE_FILENAME, SOURCE_ROLES, encode_provenance

    def _rebind(copy: Path, record_bytes: bytes) -> None:
        from src.kg.artifacts import read_split_manifest
        from src.utils.fingerprint import write_digested

        digest = write_digested(copy / PROVENANCE_FILENAME, lambda sink: sink.write(record_bytes))
        manifest = read_split_manifest(copy).manifest
        manifest["artifacts"][DATASET_RECORD_ROLE] = digest
        (copy / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def _unbound(copy: Path) -> None:
        from src.kg.artifacts import read_split_manifest

        manifest = read_split_manifest(copy).manifest
        del manifest["artifacts"][DATASET_RECORD_ROLE]
        (copy / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def _unidentified(copy: Path) -> None:
        # A record from files that leaves the MONDO input unidentified, well
        # formed and naming this graph: what a build that skipped an input
        # would have written.
        record = verify_workspace(sound)[3].record
        sources = [{"role": role, "filename": f"{role}.input", "digest": "0" * 64,
                    "declared_version": None} for role in SOURCE_ROLES[1:]]
        _rebind(copy, encode_provenance({**record, "origin": "files", "sources": sources,
                                         "missing_roles": [SOURCE_ROLES[0]]}).encode("utf-8"))

    RECORD_CASES = [
        ("missing", "a workspace whose dataset record is gone is refused",
         lambda copy: (copy / PROVENANCE_FILENAME).unlink(), "unverifiable", "is not here"),
        ("replaced", "another workspace's dataset record in its place is refused",
         lambda copy: (copy / PROVENANCE_FILENAME).write_bytes(
             (_other() / PROVENANCE_FILENAME).read_bytes()),
         "mismatch", "is not the dataset record"),
        ("other_graph", "a record of another graph, rebound in the manifest, is refused",
         lambda copy: _rebind(copy, (_other() / PROVENANCE_FILENAME).read_bytes()),
         "mismatch", "describes the graph"),
        ("unbound", "a manifest that binds no dataset record is refused",
         _unbound, "unrecorded", "binds no"),
        ("unreadable", "a bound record that is not a record is refused",
         lambda copy: _rebind(copy, b"not a record"), "unverifiable", "is not one this revision reads"),
        ("unidentified", "a bound files record leaving an input unidentified is refused",
         _unidentified, "verified", "unidentified"),
    ]

    for case, claim, tamper, binding, fragment in RECORD_CASES:
        def _record_case(tamper=tamper, binding=binding, fragment=fragment, case=case):
            copy = work / f"record_{case}"
            if copy.exists():
                shutil.rmtree(copy)
            shutil.copytree(sound, copy)
            tamper(copy)
            refuses(lambda: verify_workspace(copy), DatasetRecordRefused, fragment)
            try:
                verify_workspace(copy)
            except DatasetRecordRefused as exc:
                assert exc.binding == binding, (
                    f"refused as {exc.binding}, and the state this case establishes is {binding}"
                )
            shutil.rmtree(copy)
            return "refused", {"binding": binding}

        report.run(f"C5.{case}", "workspace", claim, _record_case)


# =============================================================================
# Phase D — training on this machine, through the production entry point
# =============================================================================
def phase_training(
    report: Report, work: Path, sound: Path, device: str, epochs: int
) -> Optional[Path]:
    print("\nD. Training, through the production entry point")
    selected: Dict[str, Optional[Path]] = {"path": None}

    def _train() -> Tuple[str, Dict[str, Any]]:
        from scripts.train_model import TrainConfig, train
        from src.utils.checkpoint_paths import (
            propose_run_dir,
            run_checkpoint_dir,
            select_checkpoint_in_dir,
        )

        # **Named, not found afterwards**: the run's directory under the
        # workspace's runs (contract M3a-2), so the checkpoint read below is
        # this run's and no other's.
        run_dir = propose_run_dir(sound, "gat", "probe")
        config = TrainConfig(
            data_dir=str(sound),
            run_dir=str(run_dir),
            conv_type="gat",
            hidden_dim=FEATURE_DIM,
            num_layers=2,
            num_heads=4,
            num_epochs=epochs,
            batch_size=8,
            warmup_steps=0,
            num_workers=0,
            device=device,
            seed=SEED,
        )
        metrics = train(config)
        assert metrics, "training returned no metrics"

        chosen = select_checkpoint_in_dir(run_checkpoint_dir(run_dir))
        assert chosen is not None, "training wrote no checkpoint"
        selected["path"] = chosen

        import math as _math

        finite = {
            key: value for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        nonfinite = sorted(k for k, v in finite.items() if not _math.isfinite(float(v)))
        assert not nonfinite, f"training reported non-finite metrics: {nonfinite}"
        return "the production trainer produced a checkpoint", {
            "epochs": epochs,
            "metric_names": sorted(finite),
            "checkpoint_name": chosen.name,
        }

    trained = report.run(
        "D1", "training", "the production trainer runs here", _train
    )
    if trained.status != "passed":
        return None

    def _provenance() -> Tuple[str, Dict[str, Any]]:
        from src.kg.artifacts import GRAPH_ARTIFACTS
        from src.utils.checkpoint_io import read_checkpoint
        from src.utils.fingerprint import verify_training_record

        checkpoint = read_checkpoint(
            selected["path"], map_location="cpu", weights_only=True
        ).checkpoint
        digests, _, _, _ = verify_workspace(sound)
        # **Through the consumers' own comparator** (contract M2.4), not a
        # second comparison here: the four graph roles, `kg` the manifest-bound
        # source graph, present on both sides and equal, or it raises.
        verify_training_record(
            checkpoint,
            digests,
            tuple(GRAPH_ARTIFACTS),
            recorded_by=f"checkpoint {selected['path'].name}",
            compared_with="the workspace it was trained on",
        )
        return "the checkpoint names the workspace it was trained on", {
            "recorded_roles": sorted(checkpoint["training_input_digests"]),
        }

    report.run(
        "D2", "training",
        "the checkpoint carries the digests of the workspace it read", _provenance,
    )
    return selected["path"]


# =============================================================================
# Phase E — serving on this machine, attacked
# =============================================================================
def phase_serving(
    report: Report, work: Path, sound: Path, checkpoint: Path, device: str
) -> None:
    print("\nE. Serving, attacked")
    print("   (E5 and E6 provoke refusals; the ERROR lines they log are the "
          "evidence that they worked)")
    import torch

    state: Dict[str, Any] = {}

    def _build() -> Tuple[str, Dict[str, Any]]:
        import src.api.main as api

        bundle = api.build_pipeline(
            kg_path=str(sound / "kg.json"),
            checkpoint_path=str(checkpoint),
            data_dir=str(sound),
            device=device,
        )
        assert bundle is not None, "nothing was built from a sound workspace"
        config = bundle.config
        assert config.get("gnn_ready"), "the GNN did not come up"
        state["bundle"] = bundle
        # **What is served is identified by what this probe verified** (contract
        # M3c): the checkpoint D1 trained, by its digest; the workspace's graph
        # and manifest, by the digests the workspace check took; its dataset
        # record; and the device asked for, in effect.
        _, _, read, record = verify_workspace(sound)
        identity = _assert_served_identity(
            config.get("served_identity"),
            checkpoint_sha256=_read_run_checkpoint(checkpoint)["sha256"],
            workspace_digests=read,
            record=record,
            sp_build_id=_sidecar_build_id(sound),
            device=device,
        )
        return "the pipeline built, the GNN is ready, and what it serves is identified", {
            "scoring_mode": config.get("scoring_mode"),
            "sp_ready": bool(config.get("sp_ready")),
            "kg_nodes": config.get("kg_nodes"),
            "served_identity": identity,
        }

    built = report.run(
        "E1", "serving", "CONTROL: a sound workspace serves", _build
    )
    if built.status != "passed":
        return

    def _metadata() -> Tuple[str, Dict[str, Any]]:
        from fastapi.encoders import jsonable_encoder

        from src.api.routes.pipeline import _status_of

        meta = state["bundle"].config.get("checkpoint_meta", {})
        allowed = (int, float, str, type(None))
        offenders = sorted(k for k, v in meta.items() if not isinstance(v, allowed))
        assert not offenders, f"checkpoint metadata is not JSON-primitive: {offenders}"
        encoded = jsonable_encoder(_status_of(state["bundle"].config, None, None))
        # The status reports the served identity through the response's strict
        # model (contract M3c) and changes nothing in it on the way.
        assert encoded.get("served_identity") == state["bundle"].config["served_identity"], (
            "the status does not report the served identity the pipeline built"
        )
        return "metadata is JSON-primitive and the status encodes", {
            "metadata_types": {k: type(v).__name__ for k, v in sorted(meta.items())},
            "recorded_device": meta.get("device"),
        }

    report.run(
        "E2", "serving",
        "checkpoint metadata survives the encoder that serves it", _metadata,
    )

    def _device_tensor_metrics() -> Tuple[str, Dict[str, Any]]:
        """The hazard that only exists on a machine with a device: a trainer
        recording `loss.detach()` instead of `loss.item()`."""
        from src.inference.pipeline import _as_int, _as_metric

        where = "cuda" if device == "cuda" else "cpu"
        metric = _as_metric(torch.tensor(0.25, device=where))
        epoch = _as_int(torch.tensor(7, device=where))
        assert isinstance(metric, float) and abs(metric - 0.25) < 1e-6, (
            "a device metric did not normalise"
        )
        assert epoch == 7, "a device epoch did not normalise"
        assert _as_metric(torch.tensor([1.0, 2.0], device=where)) is None, (
            "a two-element tensor became a metric"
        )
        assert _as_metric(torch.tensor(float("nan"), device=where)) == "nan", (
            "a nan metric lost its name"
        )
        return f"metrics recorded as {where} tensors normalise", {"tensor_device": where}

    report.run(
        "E3", "serving",
        "a metric logged as a device tensor normalises rather than crashing",
        _device_tensor_metrics,
    )

    def _reload_and_measure() -> Tuple[str, Dict[str, Any]]:
        import asyncio

        from fastapi.encoders import jsonable_encoder

        import src.api.main as api
        from src.api.routes.pipeline import PipelineReloadRequest, reload_pipeline

        api.publish_pipeline(state["bundle"])
        facts: Dict[str, Any] = {}
        parameters = sum(
            p.numel() for p in state["bundle"].pipeline.model.parameters()
        )
        if device == "cuda":
            # **Emptied, not freed.** `empty_cache` returns the allocator's
            # unused blocks so the baseline is not whatever A4's matrices and
            # the training run left cached. It does not release the previous
            # pipeline: this probe still holds it in `state["bundle"]`, which is
            # deliberate — E5 needs it — and is one of the reasons the numbers
            # below are marked inconclusive rather than interpreted.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            before_bytes = torch.cuda.memory_allocated()
        result = asyncio.run(
            reload_pipeline(PipelineReloadRequest(data_dir=str(sound), device=device))
        )
        assert result.success, "a sound workspace failed to reload"
        jsonable_encoder(result)
        if device == "cuda":
            torch.cuda.synchronize()
            facts["allocated_before_mb"] = round(before_bytes / 1e6, 2)
            facts["peak_during_reload_mb"] = round(
                torch.cuda.max_memory_allocated() / 1e6, 2
            )
            facts["allocated_after_mb"] = round(torch.cuda.memory_allocated() / 1e6, 2)
            # **The scale this was measured at, so nobody over-reads it.** The
            # demo model is a few hundred kilobytes; a second copy of it is
            # inside the noise of an allocator that caches. This says the reload
            # works on the device and its response encodes. It does NOT settle
            # what double residency costs — that needs a checkpoint of
            # deployment size, and the report should not be read as if it did.
            facts["model_parameters"] = parameters
            facts["double_residency_conclusive"] = parameters > 10_000_000
        facts["scoring_mode"] = result.status.scoring_mode
        return "the reload succeeded and its response encodes", facts

    report.run(
        "E4", "serving",
        "reloading builds the candidate beside the live pipeline",
        _reload_and_measure,
    )

    def _failed_reload_keeps_serving() -> Tuple[str, Dict[str, Any]]:
        import asyncio

        from fastapi.encoders import jsonable_encoder

        import src.api.main as api
        from src.api.routes.pipeline import PipelineReloadRequest, reload_pipeline

        assert api.app_state.pipeline is not None, "nothing was serving before the attack"
        before = {
            "pipeline": api.app_state.pipeline,
            "kg": api.app_state.kg,
            # **What is served, from the live pipeline itself** (contract M3c),
            # never from shared state a reload swaps: the service keeps no
            # version beside the pipeline any more.
            "served_identity": api.app_state.pipeline.get_pipeline_config()["served_identity"],
            "data_dir": api.app_state._current_data_dir,
            "checkpoint": api.app_state._current_checkpoint_path,
        }
        broken = work / "broken_for_reload"
        if broken.exists():
            shutil.rmtree(broken)
        shutil.copytree(sound, broken)
        (broken / "node_features.pt").write_bytes(b"same shape, another workspace")

        result = asyncio.run(
            reload_pipeline(PipelineReloadRequest(data_dir=str(broken), device=device))
        )
        assert not result.success, "a mixed workspace was accepted"
        assert result.status.initialized, "the refusal claimed nothing was serving"
        jsonable_encoder(result)
        assert api.app_state.pipeline is before["pipeline"], "the live pipeline was replaced"
        assert api.app_state.kg is before["kg"], "the live graph was replaced"
        assert (api.app_state.pipeline.get_pipeline_config()["served_identity"]
                == before["served_identity"]), "the served identity moved"
        assert api.app_state._current_data_dir == before["data_dir"], "the workspace moved"
        assert api.app_state._current_checkpoint_path == before["checkpoint"], (
            "the checkpoint moved"
        )
        shutil.rmtree(broken)
        return "the refusal left every served field untouched", {
            "served_identity_unchanged": True,
        }

    report.run(
        "E5", "serving",
        "a refused reload leaves the running pipeline serving", _failed_reload_keeps_serving,
    )

    def _missing_manifest_reload() -> Tuple[str, Dict[str, Any]]:
        import asyncio

        import src.api.main as api
        from src.api.routes.pipeline import PipelineReloadRequest, reload_pipeline

        stripped = work / "stripped_for_reload"
        if stripped.exists():
            shutil.rmtree(stripped)
        shutil.copytree(sound, stripped)
        (stripped / "split_manifest.json").unlink()
        result = asyncio.run(
            reload_pipeline(PipelineReloadRequest(data_dir=str(stripped), device=device))
        )
        assert not result.success, "an unbound workspace was accepted"
        assert result.files_found.get("split_manifest.json") is False, (
            "the file report did not name the missing manifest"
        )
        assert api.app_state.pipeline is not None, "the live pipeline was lost"
        shutil.rmtree(stripped)
        return "the missing manifest was named in the file report", {}

    report.run(
        "E6", "serving",
        "a workspace with no manifest is named, not merely rejected",
        _missing_manifest_reload,
    )

    def _diagnose() -> Tuple[str, Dict[str, Any]]:
        from src.core.types import PatientPhenotypes

        pipeline = state["bundle"].pipeline
        result = pipeline.run(
            patient_input=PatientPhenotypes(
                patient_id="probe", phenotypes=["HP:0001250", "HP:0001263"],
            ),
            top_k=5,
            include_explanations=False,
        )
        candidates = getattr(result, "candidates", [])
        assert candidates, "the pipeline ranked nothing"
        import math as _math

        scores = [float(c.confidence_score) for c in candidates]
        assert all(_math.isfinite(score) for score in scores), (
            "a candidate came back with a non-finite confidence score"
        )
        assert scores == sorted(scores, reverse=True), "candidates came back unordered"
        # **The result says what produced it** (contract M3c): the identity it
        # carries, built when it was produced, is the one E1 verified.
        assert result.served_identity.to_dict() == state["bundle"].config["served_identity"], (
            "the diagnosis carries another served identity than the pipeline it ran on"
        )
        return "a diagnosis ran end to end", {
            "candidates": len(candidates),
            "score_range": [round(min(scores), 4), round(max(scores), 4)],
            "served_identity_is_the_built_pipelines": True,
        }

    report.run("E7", "serving", "a diagnosis runs on this machine", _diagnose)

    def _model_is_where_it_was_asked_to_be() -> Tuple[str, Dict[str, Any]]:
        """A silent fall back to the host would leave `gnn_ready` true and every
        other probe passing, while the machine bought for this served on CPU."""
        model = state["bundle"].pipeline.model
        assert model is not None, "no model was loaded"
        devices = {str(parameter.device).split(":")[0] for parameter in model.parameters()}
        assert devices == {device}, (
            f"the model's parameters are on {sorted(devices)}, not on the "
            f"requested {device}"
        )
        embeddings = state["bundle"].pipeline._node_embeddings or {}
        return f"every parameter is on {device}", {
            "parameter_devices": sorted(devices),
            "embedding_devices": sorted(
                {str(tensor.device).split(":")[0] for tensor in embeddings.values()}
            ),
        }

    report.run(
        "E8", "serving",
        "the served model's parameters are on the requested device",
        _model_is_where_it_was_asked_to_be,
    )


# =============================================================================
# Phase M — contract M2's acceptance on the demo workspace (--m2-acceptance)
# =============================================================================
#: The production trainer's settings for every demo run in phase M: D1's.
_DEMO_TRAINING: Dict[str, Any] = dict(
    conv_type="gat", hidden_dim=FEATURE_DIM, num_layers=2, num_heads=4,
    batch_size=8, warmup_steps=0, num_workers=0, seed=SEED,
)

#: The architecture fields `train_model.create_model_from_config` passes to
#: `ShepherdGNNConfig`, which a checkpoint records under `config.model_config`.
_ARCHITECTURE_FIELDS = (
    "conv_type", "hidden_dim", "num_layers", "num_heads", "dropout", "use_ortholog_gate",
)

#: Every probe contract M2's acceptance requires, by part (the plan's "Acceptance
#: for M2"). A required probe that is missing from the report, skipped or not
#: passed leaves its part incomplete; nothing is inferred from a probe not run.
M2_DEMO_REQUIRED: Tuple[str, ...] = (
    "A2", "B1", "C1", "D1", "D2", "E1", "E7", "E8",
    "M1", "M2", "M3", "M4",
    "M5.pre_contract", "M5.other_graph", "M5.other_samples",
    "M5.validated_parent", "M5.missing",
    "M6", "M7", "M8",
    # Contract M3a on the same runs: M2 above is a real pause and resume; M9
    # extends the finished child at a fixed rate; M10 refuses to resume it.
    "M9", "M10",
    # Contract M3b: the dataset record is required. C5 refuses each way a
    # record can fail to bind, and M11 refuses a workspace without one at each
    # entry point; C1, M3 and M4 show the demo's record as synthetic.
    "C5.missing", "C5.replaced", "C5.other_graph", "C5.unbound", "C5.unreadable",
    "C5.unidentified", "B22", "B23",
    "M11.training", "M11.serving", "M11.measurement",
    # Contract M4's flow on the same workspace and D1's checkpoint, functional
    # only: import preview and commit (T1); measure, keep and record (T2); the
    # same report again and a damaged copy (T3); a version mapped against M1's
    # graph (T4); the read-only view for the served checkpoint and another (T5);
    # the overlap audit on the registered version (T6).
    "T1", "T2", "T3", "T4", "T5", "T6",
    # M5's acceptance (M5-3): the environment against the lock (A5);
    # the server's lists (W1); D8 on the running service, CORS, the route
    # guard, read-only without M1 (W2-W5) and off loopback (W6); the GPU lock
    # (W7); a test run through the job runner, kept and registered as the
    # publication contract says, and registration from the store only (W8,
    # W9); the start's provenance check (W10); a training job's exclusion and
    # stop (W11).
    "A5", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10", "W11",
)
#: R8, the unchanged-workspace check, runs last, after M4's RT1-RT3 and M5's
#: RW1-RW3.
M2_REAL_REQUIRED: Tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8",
                                     "RT1", "RT2", "RT3", "RW1", "RW2", "RW3")


def m2_acceptance(report: Report, real_requested: bool) -> Dict[str, Any]:
    """Whether every probe M2's acceptance requires ran and passed, per part.

    The demo part and the real-data part are reported apart and neither stands
    in for the other: the acceptance is complete only when both are.
    """
    statuses = {probe.probe_id: probe.status for probe in report.probes}

    def part(required: Tuple[str, ...], requested: bool) -> Dict[str, Any]:
        if not requested:
            return {"requested": False, "complete": False}
        not_passed = {
            probe_id: statuses.get(probe_id, "not run")
            for probe_id in required if statuses.get(probe_id) != "passed"
        }
        return {"requested": True, "required": list(required),
                "not_passed": not_passed, "complete": not not_passed}

    demo = part(M2_DEMO_REQUIRED, True)
    real = part(M2_REAL_REQUIRED, real_requested)
    return {
        "demo": demo, "real_data": real,
        "complete": bool(demo["complete"] and real["complete"]),
        # What the acceptance ran with, beside its verdict (A5).
        "environment": environment_verdict(report),
        "scope": (
            "M2's source contract on this machine, through the production entry "
            "points. The real-data model is a short training that shows the "
            "pipeline and its checks work; it is not a model of serving quality, "
            "which is chosen by the established training and evaluation. The run "
            "also carries contract M4's flow (T1-T6, RT1-RT3): a synthetic "
            "supplied cohort imported, measured, kept, registered and read back "
            "through the read-only view. That is a functional check of the flow "
            "and its refusals, not a measure of model quality, and not an "
            "institutional test, which waits for real data, its terms and a "
            "pre-declared protocol. It carries M5's acceptance where it runs "
            "headless too (A5, W1-W11, RW1-RW3): the environment against the "
            "lock beside the checkout, the server's lists, D8, CORS and the route "
            "guard on the service started as the launcher starts it, the GPU lock, and test "
            "and training jobs through the job runner, started and stopped; not "
            "the dashboard's layout or a browser session, and no long training."
        ),
    }


def environment_verdict(report: Report) -> Dict[str, Any]:
    """A5's record in short, for the end of a run beside its verdict (the
    report's `m2_acceptance` and the console's last lines): whether the lock
    read is the committed one, and the names of what differs from it. A set
    that differs is said again where the run ends, not only an hour earlier."""
    a5 = next((probe for probe in report.probes if probe.probe_id == "A5"), None)
    if a5 is None or a5.status != "passed":
        return {"A5": a5.status if a5 is not None else "not run"}
    facts = a5.facts
    outside = facts.get("installed_not_in_lock") or []
    return {
        "A5": "passed",
        "lock_sha256": facts.get("lock_sha256"),
        "lock_is_the_committed_one": facts.get("lock_is_the_committed_one"),
        "versions_match_lock": facts.get("versions_match_lock"),
        "version_differs": [entry.get("name") for entry in facts.get("version_differs") or []],
        "declared_not_installed": [entry.get("name") for entry
                                   in facts.get("declared_not_installed") or []],
        "installed_more_than_once": [entry.get("name") for entry
                                     in facts.get("installed_more_than_once") or []],
        "not_in_lock": len(outside),
        "not_in_lock_from_deploy_scripts": sum(
            1 for entry in outside if entry.get("why") == OUTSIDE_LOCK_DEPLOY),
        "not_in_lock_required_by_nothing_installed": [
            entry.get("name") for entry in outside if entry.get("why") == OUTSIDE_LOCK_LEFTOVER],
    }


def environment_line(report: Report) -> Optional[str]:
    """A5's record as the run's closing console line (None when A5 did not
    run), so a set that differs from the lock is said where the run ends."""
    a5 = next((probe for probe in report.probes if probe.probe_id == "A5"), None)
    if a5 is None:
        return None
    if a5.status != "passed":
        return f"A5, the installed packages against the lock: {a5.status} ({a5.detail})"
    return "A5, " + lock_sentence(a5.facts)


def _tested_commit() -> Dict[str, Any]:
    """The commit this probe ran at, and whether tracked files were modified."""
    import subprocess

    def git(*args: str) -> Optional[str]:
        try:
            done = subprocess.run(["git", *args], cwd=project_root, capture_output=True,
                                  text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": head.strip() if head else None,
        "tracked_files_modified": bool(status.strip()) if status is not None else None,
    }


def _workspace_copy(source: Path, target: Path) -> Path:
    """`source`'s files, without any run or checkpoint written into it."""
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("checkpoints", "runs"))
    return target


def _train_run(
    data_dir: Path, run: Path, device: str, epochs: int, *,
    settings: Dict[str, Any], resume_from: Optional[Path] = None,
    pause_after: Optional[int] = None, extend_epochs: Optional[int] = None,
    observe: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """One run of the production trainer (`train_model.train`) with `run` as its
    run directory (contract M3a-2), so every output -- its checkpoints, logs and
    records -- is inside it and nothing lands in the workspace or anywhere else.
    Returns the run's checkpoint directory.

    **A run continued from a parent is given operational fields only**
    (contract M3a-3): it inherits every learning setting from the parent's
    record, the number of epochs included, and an explicit one would be
    refused if it differed.

    `observe`, for a continued run, receives what the restore left
    (`_observing_restore`)."""
    from scripts.train_model import TrainConfig, train
    from src.utils.checkpoint_paths import run_checkpoint_dir

    if resume_from is not None:
        fields: Dict[str, Any] = dict(
            data_dir=str(data_dir), run_dir=str(run), device=device,
            num_workers=settings.get("num_workers", 0), resume_from=str(resume_from),
            pause_after_epoch=pause_after, extend_epochs=extend_epochs,
        )
        with _observing_restore(observe):
            train(TrainConfig(**fields), {k for k, v in fields.items() if v is not None})
    else:
        fields = dict(settings)
        fields.update(data_dir=str(data_dir), run_dir=str(run), num_epochs=epochs,
                      device=device, pause_after_epoch=pause_after)
        train(TrainConfig(**fields))
    return run_checkpoint_dir(run)


def _scaler_numbers(state: Any) -> Optional[Dict[str, Any]]:
    """A gradient scaler's state as plain numbers (all GradScaler records), or
    `None` without one."""
    if not state:
        return None
    return {k: v for k, v in state.items() if isinstance(v, (int, float))}


@contextlib.contextmanager
def _observing_restore(observed: Optional[List[Dict[str, Any]]]):
    """While a continued run starts, what the production restore left.

    **Observed at the restore, not inferred from the ends** (contract M3a;
    the reviewer's review of `7e1e373`). The run is the production one: the
    probe only wraps `Trainer.continue_from` for its duration. Right after it
    restored the parent and before the first update, the gradient scaler's
    state is recorded beside the parent's `scaler_state_dict` from the same
    read the run restored from. A scaler's end values cannot show this: an
    overflow lowers the scale and resets its growth tracker, and a full growth
    interval later the scale is back where it was.
    """
    if observed is None:
        yield
        return
    from src.training.trainer import Trainer

    original = Trainer.continue_from

    def observing(self, checkpoint, plan, best_weights=None):
        original(self, checkpoint, plan, best_weights)
        observed.append({
            "parent": _scaler_numbers(checkpoint.get("scaler_state_dict")),
            "restored": (_scaler_numbers(self.scaler.state_dict())
                         if self.scaler is not None else None),
        })

    Trainer.continue_from = observing
    try:
        yield
    finally:
        Trainer.continue_from = original


def _last(checkpoint_dir: Path) -> Path:
    last = checkpoint_dir / "last.pt"
    assert last.is_file(), "the run wrote no last.pt"
    return last


def _read_run_checkpoint(path: Path) -> Dict[str, Any]:
    """A checkpoint as resume reads it (`read_checkpoint`), reduced to what a
    resume is judged by. The weights stay out of every report."""
    from src.utils.checkpoint_io import read_checkpoint

    read = read_checkpoint(path, map_location="cpu", weights_only=True)
    checkpoint = read.checkpoint
    steps = []
    for state in (checkpoint.get("optimizer_state_dict") or {}).get("state", {}).values():
        if isinstance(state, dict) and state.get("step") is not None:
            steps.append(float(state["step"]))
    return {
        "sha256": read.identity.sha256,
        "record": dict(checkpoint.get("training_input_digests") or {}),
        "optimizer_step": max(steps) if steps else None,
        "weights": checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or {},
        "progress": checkpoint.get("progress") or {},
        "rates": [float(g["lr"]) for g in
                  (checkpoint.get("optimizer_state_dict") or {}).get("param_groups", [])],
        # Mixed precision as it took effect, and the gradient scaler's state
        # (plain numbers), so a pause and resume under AMP, which only CUDA
        # runs, is shown in the report (contract M3a).
        "amp": ((checkpoint.get("config") or {}).get("effective") or {}).get("amp") or {},
        "scaler": _scaler_numbers(checkpoint.get("scaler_state_dict")),
        "loaders": ((checkpoint.get("config") or {}).get("effective") or {}).get("loaders") or {},
    }


def _assert_resumed(parent: Path, child: Path,
                    observed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A resume that trained, not one that only loaded: the child names its
    parent's bytes and the same workspace roles, its optimizer went on stepping
    from the parent's state, its weights moved, and its mixed precision and
    gradient scaler were restored (`observed`, from `_observing_restore`)."""
    import torch

    before, after = _read_run_checkpoint(parent), _read_run_checkpoint(child)
    assert before["record"].get("kg"), "the parent records no kg (contract M2.2)"
    assert after["record"].get("resume_checkpoint") == before["sha256"], (
        "the child does not name the bytes of its parent"
    )
    workspace = {k: v for k, v in before["record"].items() if k != "resume_checkpoint"}
    assert {k: v for k, v in after["record"].items() if k != "resume_checkpoint"} == workspace, (
        "the child's workspace roles are not its parent's"
    )
    assert before["optimizer_step"] is not None and after["optimizer_step"] is not None, (
        "a checkpoint records no optimizer step"
    )
    assert after["optimizer_step"] > before["optimizer_step"], (
        "the resumed run took no optimizer step beyond its parent"
    )
    changed = sum(
        1 for name, tensor in after["weights"].items()
        if name in before["weights"] and not torch.equal(tensor, before["weights"][name])
    )
    assert changed > 0, "the resumed run left every weight as its parent had it"
    amp = _scaler_restore_facts(before, after, observed)
    return {
        "parent_sha256": before["sha256"],
        "child_sha256": after["sha256"],
        "parent_optimizer_step": before["optimizer_step"],
        "child_optimizer_step": after["optimizer_step"],
        "weight_tensors_changed": changed,
        "weight_tensors": len(after["weights"]),
        "recorded_roles": sorted(after["record"]),
        **amp,
    }


def _assert_only_the_remaining_epochs(before: Dict[str, Any],
                                      after: Dict[str, Any]) -> Dict[str, Any]:
    """The child ran only the epochs its parent had left (contract M3a-3).

    **Counted by optimizer attempts, not successful updates.** AMP's gradient
    scaler skips an update on non-finite gradients, and two epochs need not
    skip alike: at the GB10 acceptance of `4b1571e` the parent's epoch made 8
    attempts and 7 updates, the remaining epoch 8 and 8, and a check that
    required twice the parent's updates refused a correct resume. An epoch
    takes `ceil(batches / accumulation)` attempts, from the parent's recorded
    schedule (and its train loader, when recorded); the child's must have
    grown by exactly that for each epoch it had left. Its successful updates
    must have grown, by no more than its attempts, and its scheduler steps by
    its successful updates (the schedule steps only when the optimizer ran).
    """
    p, c = before["progress"], after["progress"]
    stage = p.get("stage") or {}
    assert c.get("stage") == stage, "the resumed run is not in its parent's stage"
    target = int(stage.get("start_epoch", 0)) + int(stage.get("target_epochs", 0))
    assert c.get("end") == "finished" and c.get("completed_epochs") == target, (
        f"the resumed run did not finish its stage: {c.get('completed_epochs')}/{target}, "
        f"end {c.get('end')}")
    epochs_run = int(c["completed_epochs"]) - int(p["completed_epochs"])
    schedule = stage.get("schedule") or {}
    batches = schedule.get("steps_per_epoch")
    accumulation = int(schedule.get("gradient_accumulation_steps") or 1)
    assert batches, "the parent's schedule records no batches per epoch"
    loader = (before.get("loaders") or {}).get("train") or {}
    if loader.get("num_batches") is not None:
        assert int(loader["num_batches"]) == int(batches), (
            "the parent's train loader and schedule record different batches per epoch")
    per_epoch = -(-int(batches) // accumulation)
    attempts = int(c["optimizer_attempts"]) - int(p["optimizer_attempts"])
    assert attempts == epochs_run * per_epoch, (
        f"the resumed run made {attempts} optimizer attempts; the {epochs_run} epoch(s) its "
        f"parent had left take {epochs_run * per_epoch} ({per_epoch} each)")
    updates = int(c["optimizer_updates"]) - int(p["optimizer_updates"])
    assert 0 < updates <= attempts, (
        f"the resumed run's successful updates grew by {updates} over {attempts} attempts")
    steps = int(c["scheduler_steps"]) - int(p["scheduler_steps"])
    expected = 0 if schedule.get("type") in (None, "none") else updates
    assert steps == expected, (
        f"the resumed run's scheduler stepped {steps} times for {updates} successful updates")
    return {"epochs_run": epochs_run, "attempts_per_epoch": per_epoch,
            "child_run_attempts": attempts, "child_run_updates": updates,
            "child_run_skipped_updates": attempts - updates,
            "child_run_scheduler_steps": steps}


def _scaler_restore_facts(before: Dict[str, Any], after: Dict[str, Any],
                          observed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mixed precision took effect in the child as in its parent, and the
    gradient scaler was restored from the parent's state (contract M3a-3).

    **Judged at the restore** (`_observing_restore`): the scaler's state right
    after `Trainer.continue_from`, before the first update, must equal the
    parent's from the same read, which must be the parent file's. The scale and
    growth tracker at the two ends are recorded, not judged: equal or not, they
    can show neither a carried nor a restarted scaler.
    """
    assert after["amp"] == before["amp"], (
        f"the resumed run's mixed precision {after['amp']} is not its parent's {before['amp']}")
    assert len(observed) == 1, f"the continued run's restore was observed {len(observed)} times"
    restore = observed[0]
    facts: Dict[str, Any] = {"amp_enabled": bool(before["amp"].get("enabled")),
                             "grad_scaler": bool(before["amp"].get("grad_scaler"))}
    if not facts["grad_scaler"]:
        assert restore["parent"] is None and restore["restored"] is None, (
            f"a run without a gradient scaler restored one: {restore}")
        facts["scaler_restore"] = "no gradient scaler (mixed precision off, or bfloat16)"
        return facts
    assert restore["parent"] is not None and restore["parent"] == before["scaler"], (
        "the scaler state the run restored from is not the parent file's")
    assert restore["restored"] == restore["parent"], (
        f"the gradient scaler was not restored from the parent's state: "
        f"{restore['restored']} after the restore, the parent's {restore['parent']}")
    after_scaler = after["scaler"] or {}
    facts.update(
        scaler_restore="verified at restore: equal to the parent's before the first update",
        parent_scaler=restore["parent"],
        child_end_scale=after_scaler.get("scale"),
        child_end_growth_tracker=after_scaler.get("_growth_tracker"),
        child_run_updates=(after["progress"].get("optimizer_updates", 0)
                           - before["progress"].get("optimizer_updates", 0)),
        end_values="recorded, not judged",
    )
    return facts


def _measure(checkpoint: Path, data_dir: Path, output: Path, device: str) -> int:
    """The production measurement entry point, on the generated `val` cohort."""
    import scripts.measure_scorer as measure_scorer

    return measure_scorer.main([
        "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
        "--split", "val",
        "--output", str(output), "--device", device,
    ])


def _measurement_facts(output: Path, checkpoint_sha: str, kg_digest: str,
                       record: Any) -> Dict[str, Any]:
    """`record` is the dataset record the workspace check verified: the
    measurement manifest must carry it whole, sources and digest included."""
    from src.utils.fingerprint import file_sha256

    assert output.is_file(), "the measurement wrote no report"
    result = json.loads(output.read_text())
    digests = result["manifest"]["artifact_digests"]
    assert digests.get("checkpoint") == checkpoint_sha, (
        "the measurement does not name the checkpoint it was given"
    )
    assert digests.get("kg") == kg_digest, "the measurement records another source graph"
    # **The number says what its graph was built from** (contract M3b): the
    # record the workspace check verified, whole.
    recorded = result["manifest"].get("dataset_record")
    assert recorded == record.summary(), (
        f"the measurement records its dataset record as {recorded}, not the record "
        f"the workspace check verified ({record.binding}, origin {record.origin})"
    )
    return {
        "report_sha256": file_sha256(output),
        "checkpoint_sha256": checkpoint_sha,
        "dataset_record": {"binding": recorded["binding"], "origin": recorded["origin"],
                           "sha256": recorded["sha256"], "sources": len(recorded["sources"])},
        "n_ranked": result.get("n_ranked"),
        "authoritative_metrics": result.get("authoritative_metrics"),
    }


def _serve_and_diagnose(
    kg_path: Path, checkpoint: Path, data_dir: Path, device: str, record: Any,
    workspace_digests: Dict[str, str],
) -> Dict[str, Any]:
    """The production serving build (`build_pipeline`), shortest paths verified
    by its own loader, and one diagnosis through the built pipeline.

    `record` and `workspace_digests` are the dataset record and the per-file
    digests the caller's workspace check (`verify_workspace`) took; the
    diagnosis must say it was produced by those inputs (contract M3c).
    """
    import math as _math

    import src.api.main as api
    from src.core.types import PatientPhenotypes

    bundle = api.build_pipeline(
        kg_path=str(kg_path), checkpoint_path=str(checkpoint),
        data_dir=str(data_dir), device=device,
    )
    config = bundle.config
    assert config.get("gnn_ready"), "the GNN did not come up"
    assert config.get("sp_ready"), "shortest paths did not come up"
    assert config.get("sp_kg_binding") == "verified", (
        "the shortest-path table's source binding was not verified by the loader"
    )
    assert config.get("scoring_mode") == "gnn_plus_shortest_path", "the scoring mode is not GNN+SP"
    # The served workspace's record, binding state and origin apart (contract
    # M3b): the record the workspace check verified, whole.
    published = config.get("dataset_record")
    assert published == record.summary(), (
        f"serving publishes its dataset record as {published}, not the record the "
        f"workspace check verified ({record.binding}, origin {record.origin})"
    )
    devices = {str(p.device).split(":")[0] for p in bundle.pipeline.model.parameters()}
    assert devices == {device}, f"the model's parameters are on {sorted(devices)}"
    result = bundle.pipeline.run(
        patient_input=PatientPhenotypes(
            patient_id="probe", phenotypes=["HP:0001250", "HP:0001263"],
        ),
        top_k=5, include_explanations=False,
    )
    scores = [float(c.confidence_score) for c in getattr(result, "candidates", [])]
    assert scores, "the pipeline ranked nothing"
    assert all(_math.isfinite(score) for score in scores), "a non-finite score came back"
    # **The diagnosis names what produced it** (contract M3c): the checkpoint
    # this probe trained, by the digest of its own read; the graph and manifest
    # the workspace check verified; the table whose sidecar the probe read,
    # verified; the dataset record; the scoring mode; and the device in effect.
    # The result's identity is the one serving reports, built from one state.
    identity = _assert_served_identity(
        result.served_identity.to_dict(),
        checkpoint_sha256=_read_run_checkpoint(checkpoint)["sha256"],
        workspace_digests=workspace_digests,
        record=record,
        sp_build_id=_sidecar_build_id(data_dir),
        device=device,
    )
    assert identity == config.get("served_identity"), (
        "the diagnosis carries another served identity than serving reports"
    )
    return {
        "scoring_mode": config.get("scoring_mode"),
        "sp_kg_binding": config.get("sp_kg_binding"),
        "sp_hop_bound_source": config.get("sp_hop_bound_source"),
        "sp_max_hops": config.get("sp_max_hops"),
        "dataset_record": {"binding": published["binding"], "origin": published["origin"],
                           "sha256": published["sha256"], "sources": len(published["sources"])},
        "parameter_devices": sorted(devices),
        "candidates": len(scores),
        "served_identity": identity,
    }


def _same_shaped_kg():
    """The demo graph with one phenotype renamed: the same nodes, edges and
    counts, so the same shapes, and another `kg.json`."""
    from src.core.types import DataSource
    from src.kg.graph import NodeID

    kg = _demo_kg()
    node = kg.get_node(NodeID(source=DataSource.HPO, local_id="HP:0000486"))
    node.name = "Strabismus (probe: same shape, another graph)"
    return kg


# =============================================================================
# Contract M4's flow, carried by the acceptance (T1-T6 in phase M, RT1-RT3 in R)
# =============================================================================
#: The role of every cohort version the acceptance imports: what they are, a
#: synthetic supplied cohort this probe built, in every record and view.
M4_ROLE = "probe-synthetic"
#: The labels of those versions: the demo registry holds the first two (the
#: second mapped against M1's graph), the real-data registry the third.
M4_DEMO_LABEL = "probe-demo-1"
M4_OTHER_GRAPH_LABEL = "probe-other-graph-1"
M4_REAL_LABEL = "probe-real-val-1"
#: How many of the real workspace's val cases RT1 writes back as a source: the
#: first this many the source format can name, or all of them when fewer can.
M4_REAL_CASES = 20
#: Well-formed ids that are not nodes of the demo graph, so the demo import's
#: exclusions are exercised: two terms that do not map, a disease that is none.
_M4_UNKNOWN_HPO = ("HP:9999999", "HP:9999998")
_M4_UNKNOWN_MONDO = "MONDO:9999998"
#: What the importer must make of `_m4_demo_cases` under its default policy
#: (keep): ten ranked of twelve, one excluded per category, one case ranked on
#: fewer terms than it lists, two distinct terms unmappable, one repeat removed.
M4_DEMO_COUNTS: Dict[str, Any] = {
    "source_cases": 12, "ranked_eligible": 10,
    "excluded": {"ground_truth_absent": 1, "phenotype_unmappable": 1}, "n_excluded": 2,
    "cases_with_lost_terms": 1, "terms_unmappable": 2,
    "repeats_removed": 1, "cases_with_repeats": 1,
}


def _node_term(node: str, node_type: str) -> Optional[str]:
    """The `shepherd-cases/1` term that names graph node `node` (a `phenotype`
    or a `disease`), or None when that format cannot name it.

    **The importer's own rule, inverted, not a second one.** The importer maps
    a term to the node `str(NodeID(HPO, term))` or `str(NodeID(MONDO, term))`
    and accepts only a term its own pattern accepts (`cohort_registry`); a node
    is named by the term for which both hold.
    """
    from src.core.types import DataSource, NodeID
    from src.evaluation.cohort_registry import _HPO_TERM, _MONDO_TERM

    source, pattern = {"phenotype": (DataSource.HPO, _HPO_TERM),
                       "disease": (DataSource.MONDO, _MONDO_TERM)}[node_type]
    local = node.split(":", 1)[-1]
    return local if pattern.fullmatch(local) and str(NodeID(source, local)) == node else None


def _source_case(case_id: str, sample: Any,
                 reverse: Dict[str, Dict[int, str]]) -> Optional[Dict[str, Any]]:
    """A workspace sample written back as a `shepherd-cases/1` case through the
    graph's ids, or None when the format cannot name one of its nodes."""
    phenotypes = [_node_term(reverse["phenotype"][i], "phenotype") for i in sample.phenotype_ids]
    disease = _node_term(reverse["disease"][sample.disease_id], "disease")
    if disease is None or None in phenotypes:
        return None
    return {"case_id": case_id, "phenotypes": phenotypes, "disease": disease}


def _m4_demo_cases(data_dir: Path) -> List[Dict[str, Any]]:
    """A source built from the demo graph's own HP and MONDO ids, whose counts
    are `M4_DEMO_COUNTS`: eight plain cases, then one listing a term twice, one
    with a term the graph lacks, one whose disease is not a node and one with no
    term that maps."""
    from src.kg.graph import KnowledgeGraph

    reverse = KnowledgeGraph.read_json(data_dir / "kg.json").kg.get_reverse_node_mapping()
    p = [_node_term(reverse["phenotype"][i], "phenotype") for i in sorted(reverse["phenotype"])]
    d = [_node_term(reverse["disease"][i], "disease") for i in sorted(reverse["disease"])]
    assert None not in p + d, "a demo graph node is not an HP or MONDO term"
    assert len(p) >= 5 and len(d) >= 3, (
        "the demo graph has too few phenotypes or diseases for the probe's cohort source")
    cases = [{"case_id": f"probe-case-{i + 1:02d}",
              "phenotypes": [p[i % len(p)], p[(i + 3) % len(p)]], "disease": d[i % len(d)]}
             for i in range(8)]
    return cases + [
        # One repeat removed by the normalisation.
        {"case_id": "probe-case-09", "phenotypes": [p[0], p[1], p[0]], "disease": d[0]},
        # Kept under the default policy and ranked on the term that maps.
        {"case_id": "probe-case-10", "phenotypes": [p[2], _M4_UNKNOWN_HPO[0]], "disease": d[1]},
        # ground_truth_absent, then phenotype_unmappable.
        {"case_id": "probe-case-11", "phenotypes": [p[3], p[4]], "disease": _M4_UNKNOWN_MONDO},
        {"case_id": "probe-case-12", "phenotypes": [_M4_UNKNOWN_HPO[1]], "disease": d[2]},
    ]


def _write_cohort_source(path: Path, cases: List[Dict[str, Any]]) -> Path:
    from src.evaluation.cohort_registry import SOURCE_FORMAT

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": SOURCE_FORMAT, "cases": cases}, indent=2),
                    encoding="utf-8")
    return path


def _import_cohort(registry: Path, data_dir: Path, source: Path, label: str,
                   commit: Optional[str] = None) -> Tuple[int, str, str]:
    """`scripts/import_cohort.py`'s `main`, its exit code and what it printed."""
    import io

    import scripts.import_cohort as import_cohort

    argv = ["--registry", str(registry), "--data-dir", str(data_dir), "--source", str(source),
            "--label", label, "--role", M4_ROLE]
    if commit is not None:
        argv += ["--commit", commit]
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = import_cohort.main(argv)
    return code, out.getvalue(), err.getvalue()


def _import_version(registry: Path, data_dir: Path, source: Path, label: str) -> Tuple[Any, str]:
    """The importer's two steps through its CLI: the preview, which must leave
    the registry as it was, then the commit by the preview id it printed.
    Returns the version as `resolve_version` reads it, and the preview id.

    A refusal's message names paths, so it goes to the console and the report
    says what happened in this file's words. A registry inside the repository
    -- a --work-dir inside it -- is said as such.
    """
    from src.evaluation.cohort_registry import resolve_version

    before = snapshot(registry, include_dirs=True)
    code, out, err = _import_cohort(registry, data_dir, source, label)
    if code != 0:
        print(err, end="", file=sys.stderr)
        if "inside the repository" in err:
            raise AssertionError(
                "the importer refused the registry under --work-dir: it is inside the "
                "repository, and a cohort registry holds patient-level data, so it must "
                "be outside it. Run the acceptance with a --work-dir outside the repository")
        raise AssertionError(f"the import preview was refused (exit {code}); its message "
                             "is on the console")
    assert_untouched(before, registry, "the import preview", include_dirs=True)
    found = re.search(r"Preview id: ([0-9a-f]{16})", out)
    assert found, "the import preview printed no preview id"
    code, _, err = _import_cohort(registry, data_dir, source, label, commit=found.group(1))
    if code != 0:
        print(err, end="", file=sys.stderr)
        raise AssertionError(f"the commit by the printed preview id was refused (exit {code}); "
                             "its message is on the console")
    return resolve_version(registry, label), found.group(1)


def _measure_cohort(checkpoint: Path, data_dir: Path, registry: Path, label: str,
                    device: str, *extra: str) -> int:
    """The production measurement entry point on a registered cohort version
    (contract M4): `--cohort` with `--registry`, and `extra` (`--record`). Run
    inside an M4 flow, where that registry is the deployment's
    (`_registry_setting`, set once for the flow)."""
    import scripts.measure_scorer as measure_scorer

    return measure_scorer.main([
        "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
        "--cohort", label, "--registry", str(registry), "--device", device, *extra,
    ])


def _kept_report(registry: Path, label: str) -> Tuple[Path, Dict[str, Any], Any, int]:
    """The one report the version's report store keeps, parsed and identified
    from one read, and the number of rows of the ranks it names.

    **Through the shared reader** (`_read_json`): the report's content and its
    digest come from the same bytes, as registration takes them, and the ranks
    are compared with the digest the report names before they are parsed.
    Returns the report's path, its content, the identity of the bytes parsed
    and the ranks' row count.
    """
    from scripts.measure_scorer import RANKS_FILE, REPORT_FILE
    from src.kg.storage.file_storage import SamplesDigestMismatch, _read_json

    kept = sorted((registry / "versions" / label / "reports").glob(f"*/{REPORT_FILE}"))
    assert len(kept) == 1, f"the version's report store keeps {len(kept)} reports, not one"
    report, identity = _read_json(kept[0])
    ranks = report.get("ranks") or {}
    assert ranks.get("file") == RANKS_FILE and ranks.get("sha256"), (
        "the kept report names no ranks file by digest")
    try:
        rows, _ = _read_json(kept[0].parent / RANKS_FILE,
                             expected_sha256=frozenset({ranks["sha256"]}))
    except SamplesDigestMismatch:
        raise AssertionError("the kept ranks are not the bytes the report names") from None
    assert len(rows) == report.get("n_ranked"), "the kept ranks are not one row per ranked case"
    return kept[0], report, identity, len(rows)


def _r10_facts(r10: Dict[str, Any]) -> Dict[str, Any]:
    """R10's evidence state and each label's state and counts, per scope; the
    evidence path and any problem text stay in the report."""
    from src.evaluation.r10 import R10_LABELS, R10_SCOPES

    return {
        scope: {
            "evidence": r10[scope]["evidence"]["state"],
            **{label: {key: value for key, value in r10[scope][label].items()
                       if key in ("state", "k", "n", "cases")} for label in R10_LABELS},
        }
        for scope, _ in R10_SCOPES
    }


def _evidence_found_in(r10: Dict[str, Any], data_dir: Path, scopes: Tuple[str, ...]) -> None:
    """Each scope's evidence was found, among the workspace's own files."""
    for scope in scopes:
        evidence = r10[scope]["evidence"]
        assert evidence["state"] == "found", (
            f"R10's {scope} evidence is {evidence['state']}, not found")
        assert Path(evidence["path"]).resolve().parent == data_dir.resolve(), (
            f"R10's {scope} evidence was found outside the workspace")


def _the_record(ledger_dir: Path, report_identity: Any, checkpoint_sha256: str,
                version: Any) -> Tuple[Dict[str, Any], str]:
    """The ledger beside the checkpoint, through the one ledger reader, holding
    exactly one record: this checkpoint's, on this version, built from the kept
    report's bytes. Returns the record and the digest of the ledger read."""
    from src.evaluation.sidecar import LEDGER_FILENAME, read_ledger

    ledger, ledger_sha256 = read_ledger(ledger_dir / LEDGER_FILENAME)
    assert ledger_sha256 is not None, "no ledger was written beside the checkpoint"
    records = ledger["records"]
    assert len(records) == 1, f"the ledger holds {len(records)} records, not one"
    record = records[0]
    assert record.get("checkpoint_digest") == checkpoint_sha256, (
        "the record names another checkpoint")
    assert record.get("cohort_digest") == version.identity.sha256, (
        "the record names another cohort version")
    assert record.get("source_artifact_digest") == report_identity.sha256, (
        "the record was not built from the kept report's bytes")
    assert record.get("corroborating_reports") == [], "the record lists corroborating reports"
    return record, ledger_sha256


@contextlib.contextmanager
def _logged(logger: logging.Logger):
    """The messages `logger` logs at INFO and above while this is open, for an
    entry point that says its outcome only in its log (`record_evaluation.py`)."""
    messages: List[str] = []

    class _Kept(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler, level = _Kept(logging.INFO), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


@contextlib.contextmanager
def _observing_checkpoint_reads(observed: List[str]):
    """While this is open, each read through the one checkpoint reader
    (`read_checkpoint`) is noted by file name, then made as asked.

    **Observed, not inferred from the refusal's type**, as `_observing_restore`
    observes the restore: a refusal that must come before the checkpoint is
    read is shown to. Callers import the reader when they run
    (`read_measurement_inputs` does), so they reach this wrapper.
    """
    import src.utils.checkpoint_io as checkpoint_io

    original = checkpoint_io.read_checkpoint

    def observing(path, *args, **kwargs):
        observed.append(Path(path).name)
        return original(path, *args, **kwargs)

    checkpoint_io.read_checkpoint = observing
    try:
        yield
    finally:
        checkpoint_io.read_checkpoint = original


def _serve_and_view(data_dir: Path, checkpoint: Path, expected_sha256: str,
                    device: str) -> Tuple[str, Dict[str, Any]]:
    """`checkpoint` served through the production reload, named explicitly
    (E4's route, which publishes what it built) and **bound to the digest the
    probe expects** (`expected_checkpoint_sha256`, compared on the build's own
    checkpoint read before the bytes are parsed, as the dashboard's Load binds
    it), then the status and the loaded model's recorded results read as the
    Diagnosis panel reads them: two GETs on the application, status first.
    Called inside an M4 flow, so the route reads the probe's registry as the
    deployment's (`_registry_setting`), and each record's D5 state is from the
    probe's report store.

    Returns the digest the status reports for the served checkpoint, which
    must be the one expected, and the records answer, whose echoed digest must
    be that one.
    """
    import asyncio

    from fastapi.testclient import TestClient

    import src.api.main as api
    from src.api.routes.pipeline import PipelineReloadRequest, reload_pipeline

    result = asyncio.run(reload_pipeline(PipelineReloadRequest(
        data_dir=str(data_dir), checkpoint_path=str(checkpoint), device=device,
        expected_checkpoint_sha256=expected_sha256)))
    assert result.success, f"the reload naming the checkpoint was refused: {result.message}"
    # Not entered as a context, which would run the application's lifespan and
    # build a pipeline from any SHEPHERD_* configuration in the environment over
    # the one just published; the requests alone reach the routes.
    client = TestClient(api.app)
    try:
        status = client.get("/api/v1/pipeline/status")
        evaluations = client.get("/api/v1/evaluations/loaded-model")
    finally:
        client.close()
    assert (status.status_code, evaluations.status_code) == (200, 200), (
        f"the status answered {status.status_code} and the recorded results "
        f"{evaluations.status_code}")
    served = (status.json().get("served_identity") or {}).get("checkpoint") or {}
    assert served.get("state") == "verified", "the status does not identify the served checkpoint"
    assert served.get("sha256") == expected_sha256, "the status reports another checkpoint served"
    answer = evaluations.json()
    assert answer.get("checkpoint_sha256") == served["sha256"], (
        "the recorded results are for another checkpoint than the status reports")
    return served["sha256"], answer


#: A record's D5 state when the one report its key has in the store is the one
#: it was built from (`GET /api/v1/evaluations/loaded-model`, `contest`).
UNCONTESTED = {"state": "uncontested", "differing": [], "runs": []}


def _the_view_shows(answer: Dict[str, Any], record: Dict[str, Any], ledger_sha256: str,
                    scopes: Tuple[str, ...]) -> Dict[str, Any]:
    """The read-only view's answer shows exactly `record`, from the ledger read
    as `ledger_sha256`, with R10's evidence found for `scopes`, and uncontested
    in its version's report store (D5: the one report the flow kept for it).
    Returns its facts: the state, the record's digests and counts and its D5
    state, no path."""
    assert answer.get("state") == "records", f"the view's state is {answer.get('state')}"
    assert answer.get("records_not_shown") == 0, "a record for these weights is not shown"
    assert (answer.get("ledger") or {}).get("sha256") == ledger_sha256, (
        "the view read another ledger than the one the registration wrote")
    shown = answer.get("records") or []
    assert len(shown) == 1, f"the view shows {len(shown)} records, not one"
    view = shown[0]
    assert view["source_artifact_digest"] == record["source_artifact_digest"], (
        "the view shows a record built from another report")
    assert view["registration"]["sequence"] == record["registration"]["sequence"]
    assert (view["cohort"]["label"], view["cohort"]["role"], view["n_ranked"]) == (
        record["cohort"]["label"], record["cohort"]["role"], record["n_ranked"])
    for scope in scopes:
        assert view["r10"][scope]["evidence"]["state"] == "found", (
            f"the view shows R10's {scope} evidence as not found")
    # The review's P2-6, seen end to end (root-cause round, INT-1): the route
    # reads the probe's registry, so the record's state is from its store.
    contest = view.get("contest") or {}
    assert contest == UNCONTESTED, (
        f"the view shows the record's D5 state as {contest.get('state')}, not uncontested")
    return {"state": answer["state"], "checkpoint_sha256": answer["checkpoint_sha256"],
            "records": len(shown), "source_artifact_digest": view["source_artifact_digest"],
            "registration_sequence": view["registration"]["sequence"],
            "n_ranked": view["n_ranked"], "environment_class": view["environment_class"],
            "contest_state": contest["state"]}


# =============================================================================
# M5's acceptance, carried by the same run (W1-W11 in phase M, RW1-RW3 in R)
# =============================================================================
#: M5's probes in each part, every one required (`M2_DEMO_REQUIRED`,
#: `M2_REAL_REQUIRED`).
M5_DEMO_PROBES: Tuple[str, ...] = tuple(f"W{index}" for index in range(1, 12))
M5_REAL_PROBES: Tuple[str, ...] = ("RW1", "RW2", "RW3")
#: Phase M's M5 probes run on the probe's own deployment under <work>/m5: one
#: workspace under its workspace root, a copy of the demo workspace, with one
#: run holding a copy of M2's resumed checkpoint and one whose checkpoint is a
#: link out of the workspace (listed with that reason, never opened).
M5_WORKSPACE = "probe-demo"
M5_LISTED_RUN = "probe-m5-listed"
M5_LINKED_RUN = "probe-m5-link-out"
M5_CHECKPOINT = "last.pt"
#: The architecture of every demo run (`_DEMO_TRAINING`), and so of the runs'
#: directory in the workspace.
_M5_ARCHITECTURE = "gat"
#: D8's operator routes (`src/api/operator_access.py`: every state-changing
#: route and the checkpoint lists), each with the status it answers when D8
#: allows the probe's request. **Nothing runs either way**: a POST carries
#: ``[]`` as its JSON body, well formed and of the wrong shape, so D8's
#: dependency refuses it with 403 before the body is validated, and with D8
#: allowing, a route with a body refuses it in validation (422) before its
#: handler; the training stop answers 409 with no training running, and the
#: lists only list. A route any other way gated (`_gated_api_routes`) is asked
#: too, where D8 refuses only.
OPERATOR_ROUTES: Tuple[Tuple[str, str, int], ...] = (
    ("POST", "/api/v1/pipeline/reload", 422),
    ("POST", "/api/v1/pipeline/config", 422),
    ("POST", "/api/v1/training/start", 422),
    ("POST", "/api/v1/training/stop", 409),
    ("GET", "/api/v1/training/checkpoints", 200),
    ("GET", "/api/v1/workspaces/{name}/checkpoints", 200),
)
#: The gated dashboard events called directly through Gradio's HTTP API, as a
#: script or `gradio_client` would, each found by the handler it is registered
#: with (its module under `src/webui/components` and its name; its API name is
#: the dashboard's own, `_dashboard_events`), with the data sent and what its
#: answer says when D8 allows: Training's confirmed Stop with no training
#: running, Test's Start with no listed version chosen, Test's Register with
#: no kept report chosen. Allowed or refused, nothing is done.
GATED_EVENTS: Tuple[Tuple[str, str, Tuple[Any, ...], str], ...] = (
    ("training_console", "_on_stop", ({},), "No training is running"),
    ("test_page", "_on_start", ({}, {}, None, None, None, False, False), "Nothing was started."),
    ("test_page", "_on_register", ({}, {}, {}, None, None), "Nothing was registered."),
)
#: The deployment settings M5's probes depend on: the server's settings
#: (`src/config/server_settings.py`; the workspaces root also names the GPU
#: lock), the serving device a test run is given, and the pipeline the service
#: would build at startup.
#: For M5's probes each is the probe's own value or unset, in this process and
#: in every process it starts, and all are put back afterwards; an operator's
#: shell that holds the deployment's settings changes nothing here, and the
#: deployment's own lock, root and registry are never used.
_M5_SETTINGS: Tuple[str, ...] = (
    "SHEPHERD_DEPLOYMENT_MODE", "SHEPHERD_WORKSPACES_ROOT", "SHEPHERD_REGISTRY",
    "SHEPHERD_INTAKE_DIR", "SHEPHERD_TRAINING_EVIDENCE_DIRS", "SHEPHERD_ORDERING_ROLES",
    "SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND", "SHEPHERD_BIND_HOST", "SHEPHERD_DEVICE",
    "SHEPHERD_KG_PATH", "SHEPHERD_CHECKPOINT_PATH", "SHEPHERD_DATA_DIR",
)
#: How long M5's waits allow: a service to answer, a job to take the GPU lock,
#: a training job to reach its first epoch, and a test run to end.
_SERVICE_START_SECONDS = 300.0
_LOCK_TAKEN_SECONDS = 300.0
_FIRST_EPOCH_SECONDS = 1800.0
_TEST_RUN_SECONDS = 3600.0
#: The epochs, and the early-stopping patience, of the training job W11 and RW3
#: hold the GPU with (`_held_config`): far more than it can complete while the
#: probe watches, so it ends only by the probe's stop (or a failure, which the
#: checks name), never by itself during the check. With the trainer's default
#: patience the demo job stopped early at about epoch 10, within a second on a
#: fast CPU, and released the lock before the measurement CLI was tried (review
#: of 8242b47, W11/RW3).
#: Nothing in the trainer grows with the epoch count but its schedule's length.
_HOLD_EPOCHS = 1_000_000
#: The probe's own workspaces root, an empty directory under the work
#: directory whose `.shepherd-gpu.lock` is the GPU lock of the phases that run
#: the formal entry points in this process (`_probes_own_gpu_lock`).
PROBE_GPU_LOCK_ROOT = "probe-gpu-lock"
#: The GPU lock's setting (the workspaces root) of the operator's environment
#: while the probe's own lock is named in it (`_probes_own_gpu_lock`), for what
#: W1 and RW1 say of the deployment's lock (`_deployment_gpu_lock`); None
#: otherwise.
_OPERATOR_LOCK_SETTINGS: Optional[Dict[str, str]] = None
#: A process that takes the GPU lock and keeps it until it is killed: W7's
#: holder, through the lock's own entry point (`take_gpu_lock`). It also ends
#: when its standard input closes, so a probe that dies cannot leave it behind.
_LOCK_HOLDER = (
    "import os, sys\n"
    "from src.utils.gpu_lock import take_gpu_lock\n"
    "lock = take_gpu_lock('probe-holder', 'probe_deployment.py W7 holder')\n"
    "print('held', os.getpid(), flush=True)\n"
    "sys.stdin.read()\n"
)


@contextlib.contextmanager
def _deployment_settings(values: Mapping[str, Optional[str]]):
    """`values` as the M5 settings in effect (`_M5_SETTINGS`; any not named is
    unset) while this is open, then each put back as it was."""
    unknown = sorted(set(values) - set(_M5_SETTINGS))
    assert not unknown, f"not an M5 setting: {unknown}"
    saved = {name: os.environ.get(name) for name in _M5_SETTINGS}

    def apply(settings: Mapping[str, Optional[str]]) -> None:
        for name in _M5_SETTINGS:
            value = settings.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    apply(values)
    try:
        yield
    finally:
        apply(saved)


@contextlib.contextmanager
def _registry_setting(registry: Path):
    """`SHEPHERD_REGISTRY` naming the probe's own registry while this is open,
    then put back, through `_deployment_settings`; the other M5 settings stay
    as they are. A ``--cohort`` run measures, and a report is registered, only
    in the deployment's registry (BACKLOG 37), and for M4's probes that is the
    probe's. **Opened once around each M4 flow** (T1-T6, RT1-RT3; root-cause
    round, INT-1), never per call, so every step of the flow that reads the
    setting inherits it, the loaded-model route's D5 state included, whatever
    the operator's shell names."""
    current = {name: os.environ.get(name) for name in _M5_SETTINGS}
    with _deployment_settings({**current, "SHEPHERD_REGISTRY": str(registry)}):
        yield


@contextlib.contextmanager
def _probes_own_gpu_lock(work: Path):
    """`<work>/probe-gpu-lock/.shepherd-gpu.lock` as the GPU lock in effect
    while this is open, then the operator's setting put back (review of
    8242b47, P2-4).

    The formal entry points this probe runs in its own process (measure_scorer
    for M3, M7, M11.measurement, T2, T4, R5 and RT2) refuse without a
    configured lock, so the probe names one of its own and depends on nothing
    in the operator's shell; the deployment's lock is never taken, so those
    probes do not exclude a deployment job running meanwhile. M5's probes name
    theirs (`_M5_SETTINGS`). The operator's setting is kept for
    `_deployment_gpu_lock`.

    **The workspaces root is the lock's one name**
    (`server_settings.gpu_lock_location`, root-cause review P24-S1 and its
    verification), so the probe's own lock is the lock under a root of its
    own, an empty directory, as each test's is; the operator's root is
    replaced meanwhile. Nothing these phases run reads the root from the
    environment: the probe takes the real one from ``--workspaces-root`` and
    lists it through ``environ=``."""
    global _OPERATOR_LOCK_SETTINGS
    from src.utils.gpu_lock import WORKSPACES_ROOT_ENV

    saved = os.environ.get(WORKSPACES_ROOT_ENV)
    _OPERATOR_LOCK_SETTINGS = {} if saved is None else {WORKSPACES_ROOT_ENV: saved}
    own = work / PROBE_GPU_LOCK_ROOT
    own.mkdir(parents=True, exist_ok=True)
    os.environ[WORKSPACES_ROOT_ENV] = str(own)
    try:
        yield
    finally:
        _OPERATOR_LOCK_SETTINGS = None
        if saved is None:
            os.environ.pop(WORKSPACES_ROOT_ENV, None)
        else:
            os.environ[WORKSPACES_ROOT_ENV] = saved


def _deployment_gpu_lock() -> Dict[str, bool]:
    """The GPU lock the operator's own settings name (kept by
    `_probes_own_gpu_lock`, or this process's environment when no probe lock
    is named): whether one is configured, whether a job holds it now, and
    whether its setting is set but cannot be used (`unreadable`; a root not
    set is not configured, not unreadable). M5's probes run on their own lock,
    so a managed job holding the deployment's is not refused by them; this
    states it, in the report as booleans and on the console naming the holder
    (its description names a host, which the report never carries)."""
    from src.utils.gpu_lock import lock_state

    state = lock_state(_OPERATOR_LOCK_SETTINGS)
    held_by = state.get("held_by")
    if held_by:
        print(f"   the deployment's GPU lock is held by {held_by} while M5's probes run on "
              "their own lock: run the acceptance while no managed GPU job runs",
              file=sys.stderr)
    return {"configured": bool(state.get("configured")), "held": bool(held_by),
            "unreadable": bool(state.get("problem")) and not state.get("unset")}


def _wait_for(condition: Callable[[], Any], timeout: float, interval: float = 0.2) -> Any:
    """`condition()` until it is true or `timeout` seconds have passed; its last value."""
    deadline = time.monotonic() + timeout
    while True:
        value = condition()
        if value or time.monotonic() >= deadline:
            return value
        time.sleep(interval)


def _names_pid(text: str, pid: int) -> bool:
    """Whether a holder's description (`describe_holder`) names process `pid`."""
    return re.search(rf"\bpid {pid}\b", text) is not None


#: Why W11 and RW3 cannot run where `os.waitid` is missing.
_NO_WAITID = (
    "this Python has no os.waitid (Windows, or macOS before Python 3.13), so whether the "
    "training job's process still runs cannot be told without waiting for it, and the "
    "check that it held the GPU lock throughout cannot be made")


def _require_waitid() -> Callable[..., Any]:
    """`os.waitid`, or the probe's failure saying why it is needed
    (`_NO_WAITID`). The one check of it: `_start_training_job` makes it before
    anything is started, so W11 and RW3 start no job where it is missing
    (root-cause round, W11-4), and `_job_runs` takes waitid from it."""
    waitid = getattr(os, "waitid", None)
    if waitid is None:
        raise AssertionError(_NO_WAITID)
    return waitid


def _job_runs(pid: int) -> bool:
    """Whether the runner's job process `pid`, a child of this process (the
    runner runs in it), has not ended. `waitid` with WNOWAIT looks without
    waiting for it, so the runner's monitor still does; an ended job the
    monitor has not waited for yet is a zombie, which `kill(pid, 0)` and the
    holder description's liveness hint would still count as running."""
    waitid = _require_waitid()
    try:
        return waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None
    except ChildProcessError:  # waited for already: it has ended
        return False


def _holding_job(manager: Any, pid: int) -> str:
    """'' while the training job's process `pid` runs and the GPU lock's
    holder record names it; otherwise what is not so, in words that carry no
    path or host (the report's)."""
    from src.utils.gpu_lock import lock_state

    if not _job_runs(pid):
        return ("its process had ended (the runner says "
                f"{manager.get_job_status_word('training')})")
    held_by = lock_state().get("held_by") or ""
    if not ("a training job" in held_by and _names_pid(held_by, pid)):
        return "the GPU lock's holder record did not name its process"
    return ""


#: The free-text fields of a list item, which may name a server path for the
#: operator (`server_lists`: "a refusal's text may name a server path").
_REASON_FIELDS = frozenset({"problem", "not_trainable", "reason"})


def _identity_fields(value: Any) -> Any:
    """A list item's JSON form without its free-text reasons: what a client is
    given to name an item by, which must carry no server path."""
    if isinstance(value, dict):
        return {key: _identity_fields(item) for key, item in value.items()
                if key not in _REASON_FIELDS}
    if isinstance(value, (list, tuple)):
        return [_identity_fields(item) for item in value]
    return value


def _parse_response(raw: bytes) -> Tuple[Optional[int], Dict[str, str], bytes]:
    """A raw HTTP/1.1 response: its status (None when none came), its headers by
    lowercased name, and its body."""
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    try:
        status: Optional[int] = int(lines[0].split()[1])
    except (IndexError, ValueError):
        status = None
    headers = {}
    for line in lines[1:]:
        name, sep, value = line.partition(":")
        if sep:
            headers[name.strip().lower()] = value.strip()
    return status, headers, body


def _with_parent_death_signal() -> Optional[Callable[[], None]]:
    """On Linux, a ``preexec_fn`` after which the kernel sends the child SIGTERM
    when the thread that started it ends, however it ends (SIGKILL and the
    out-of-memory killer included), so a service this probe started never
    outlives it. The probe starts its services from its main thread. None
    elsewhere (macOS and Windows have no such request; there, stop the probe
    with Ctrl-C)."""
    if not sys.platform.startswith("linux"):
        return None
    import ctypes
    import signal

    libc = ctypes.CDLL(None, use_errno=True)      # loaded before the fork, not in the child
    parent = os.getpid()

    def _die_with_parent() -> None:
        libc.prctl(1, int(signal.SIGTERM), 0, 0, 0)  # PR_SET_PDEATHSIG
        if os.getppid() != parent:                  # the probe ended before the request
            os._exit(1)

    return _die_with_parent


def _exit_on_signal(number: int, frame: Any) -> None:
    """A termination signal ends the probe through `sys.exit`, so every
    ``finally`` that stops its services and the jobs it started runs."""
    print(f"\nsignal {number}: the probe stops its services and jobs, then exits",
          file=sys.stderr)
    sys.exit(128 + number)


def _stop_on_termination_signals() -> None:
    """Every signal that no longer reaches a runner job because the job leads
    its own process group (`JOB_DETACHED_SIGNALS`, the runner's one
    definition: SIGTERM from ``kill <pid>``, SIGHUP from a closed terminal,
    SIGQUIT from its Ctrl-\\, SIGBREAK on Windows) ends the probe as Ctrl-C
    does, through its ``finally`` blocks, instead of leaving the services and
    runner jobs it started running (root-cause review, P23-2: a Ctrl-\\ left
    W11's or RW3's job holding the lock). Ctrl-C itself already does
    (``KeyboardInterrupt``). A signal already set to something other than
    its default (SIGHUP ignored under ``nohup``, say) is left as it is."""
    import signal

    from src.api.services.training_manager import JOB_DETACHED_SIGNALS

    for name in JOB_DETACHED_SIGNALS:
        number = getattr(signal, name, None)
        if number is not None and signal.getsignal(number) is signal.SIG_DFL:
            signal.signal(number, _exit_on_signal)


@contextlib.contextmanager
def _job_ended_on_exit(manager: Any, job_type: str):
    """A job of `job_type` the runner `manager` still runs when this closes
    (a probe that failed, timed out or was told to stop) is stopped, so the
    probe never leaves one running on its workspace."""
    try:
        yield
    finally:
        if manager.get_job_status_word(job_type) in ("running", "stopping"):
            manager.stop(job_type)


class _ProbeServer:
    """The service as the launcher starts it, in its own process: uvicorn
    serving `src.api.main:app` on 127.0.0.1 with `--no-proxy-headers` (the
    launcher's own constants), the bound host passed in as
    `SHEPHERD_BIND_HOST` for startup's bind check, and this probe's M5
    settings (`_M5_SETTINGS`). Its Gradio files, temporary files and log stay
    under `directory`. A fresh interpreter, so the dashboard is mounted as the
    service mounts it, whatever this process imported first. Requests are
    plain HTTP to its port; it is stopped by this probe's own PID.
    """

    def __init__(self, name: str, directory: Path, settings: Mapping[str, Optional[str]]):
        self.name = name
        self.directory = directory
        self.settings = dict(settings)
        self.process: Any = None
        self.port: Optional[int] = None
        self.ready_seconds: Optional[float] = None
        self._log: Any = None
        self._started = 0.0

    @property
    def gradio_dir(self) -> Path:
        return self.directory / "gradio"

    def environment(self) -> Dict[str, str]:
        from src.config.server_settings import ENV_BIND_HOST

        env = {key: value for key, value in os.environ.items() if key not in _M5_SETTINGS}
        env.update({key: value for key, value in self.settings.items() if value is not None})
        env[ENV_BIND_HOST] = "127.0.0.1"
        env["GRADIO_TEMP_DIR"] = str(self.gradio_dir)
        env["TMPDIR"] = str(self.directory / "tmp")
        return env

    def start(self) -> None:
        import socket
        import subprocess

        from scripts.launch.shep_launch import DEFAULT_ENTRY, UVICORN_APP, UVICORN_FORCED_ARGS

        for directory in (self.gradio_dir, self.directory / "tmp"):
            directory.mkdir(parents=True, exist_ok=True)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        if self._log is None:
            self._log = open(self.directory / "server.log", "ab")
        self._started = time.monotonic()
        self.process = subprocess.Popen(
            [sys.executable, "-m", DEFAULT_ENTRY, UVICORN_APP, "--host", "127.0.0.1",
             "--port", str(self.port), *UVICORN_FORCED_ARGS],
            cwd=project_root, env=self.environment(), stdin=subprocess.DEVNULL,
            stdout=self._log, stderr=subprocess.STDOUT,
            # **It ends with the probe**, even one ended by SIGKILL (Linux).
            preexec_fn=_with_parent_death_signal())

    def wait_ready(self, timeout: float = _SERVICE_START_SECONDS) -> float:
        """Until /health answers 200; the seconds since the start. A port
        another process took meanwhile is tried once more on another."""
        retried = False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                log = (self.directory / "server.log").read_bytes()[-4000:]
                if not retried and b"address already in use" in log.lower():
                    retried = True
                    self.start()
                    continue
                raise AssertionError(
                    f"the {self.name} service exited (status {self.process.returncode}) before "
                    "it answered /health; its log is under the work directory")
            try:
                status = self.request("GET", "/health", timeout=2.0)[0]
            except OSError:
                status = None
            if status == 200:
                self.ready_seconds = round(time.monotonic() - self._started, 1)
                return self.ready_seconds
            time.sleep(0.5)
        raise AssertionError(f"the {self.name} service did not answer /health within "
                             f"{timeout:.0f} s; its log is under the work directory")

    @property
    def ready(self) -> bool:
        return self.ready_seconds is not None and self.process is not None \
            and self.process.poll() is None

    def stop(self) -> Optional[int]:
        """SIGINT, uvicorn's graceful shutdown (terminate where there is none),
        then a kill; the exit status, None when it was never started."""
        import signal
        import subprocess

        process, self.process = self.process, None
        if process is None:
            return None
        if process.poll() is None:
            if sys.platform == "win32":  # pragma: no cover - the backend is untested on Windows
                process.terminate()
            else:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        if self._log is not None:
            self._log.close()
            self._log = None
        return process.returncode

    def request(self, method: str, path: str, headers: Optional[Mapping[str, str]] = None,
                body: Optional[bytes] = None,
                timeout: float = 120.0) -> Tuple[int, Dict[str, str], bytes]:
        """One request on a new connection: ``Host: 127.0.0.1:<port>`` unless
        `headers` names another. Never through a proxy."""
        import http.client

        sent = {"Host": f"127.0.0.1:{self.port}"}
        sent.update(headers or {})
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            connection.request(method, path, body=body, headers=sent)
            response = connection.getresponse()
            return (response.status, {k.lower(): v for k, v in response.getheaders()},
                    response.read())
        finally:
            connection.close()

    def json(self, path: str, headers: Optional[Mapping[str, str]] = None) -> Any:
        status, _, body = self.request("GET", path, headers)
        assert status == 200, f"GET {path} answered {status} on the {self.name} service"
        return json.loads(body)

    def event(self, name: str, data: List[Any],
              headers: Optional[Mapping[str, str]] = None) -> str:
        """The Gradio event `name` called through Gradio's HTTP API
        (``/ui/gradio_api/call/<name>``) with these headers; the outputs of
        its completion, as text."""
        status, _, body = self.request(
            "POST", f"/ui/gradio_api/call/{name}",
            {"Content-Type": "application/json", **(headers or {})},
            json.dumps({"data": data}).encode("utf-8"))
        assert status == 200, f"the {name} event was not accepted ({status})"
        event_id = json.loads(body)["event_id"]
        status, _, stream = self.request("GET", f"/ui/gradio_api/call/{name}/{event_id}",
                                         headers)
        lines = stream.decode("utf-8", "replace").splitlines()
        assert "event: complete" in lines, f"the {name} event did not complete ({status})"
        return lines[lines.index("event: complete") + 1][len("data:"):]

    def raw(self, request: bytes, wait: float = 5.0) -> bytes:
        """`request` sent as is on a new connection; what came back before the
        server closed it or `wait` seconds passed."""
        import socket

        chunks = []
        deadline = time.monotonic() + wait
        with socket.create_connection(("127.0.0.1", self.port), timeout=wait) as connection:
            connection.sendall(request)
            while time.monotonic() < deadline:
                try:
                    chunk = connection.recv(65536)
                except TimeoutError:
                    break
                if not chunk:
                    break
                chunks.append(chunk)
        return b"".join(chunks)


def _asgi_body_requests(app: Any, method: str, path: str) -> Tuple[Optional[int], int]:
    """`app` called in this process as an ASGI application for one request
    announcing a 50 MB multipart body, offered to whoever asks for it: the
    status it answered, and how many times the body was asked for. A body that
    is read stops after sixteen chunks, so a regression fails, not hangs."""
    import asyncio

    asked = {"calls": 0}
    sent: List[Dict[str, Any]] = []

    async def receive() -> Dict[str, Any]:
        asked["calls"] += 1
        if asked["calls"] > 16:
            return {"type": "http.disconnect"}
        return {"type": "http.request", "body": b"x" * 65536, "more_body": True}

    async def send(message: Dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
        "scheme": "http", "path": path, "raw_path": path.encode("latin-1"), "root_path": "",
        "query_string": b"", "server": ("127.0.0.1", 8264), "client": ("127.0.0.1", 50000),
        "headers": [(b"host", b"127.0.0.1:8264"),
                    (b"content-type", b"multipart/form-data; boundary=B"),
                    (b"content-length", b"50000000")],
    }
    asyncio.run(app(scope, receive, send))
    status = next((m.get("status") for m in sent if m.get("type") == "http.response.start"),
                  None)
    return status, asked["calls"]


def _asgi_answer(app: Any, method: str, raw_path: str,
                 root_path: str = "") -> Tuple[Optional[int], bytes]:
    """`app` called in this process as an ASGI application for one request
    with no body to `raw_path`, decoded as uvicorn decodes it: its status and
    body."""
    import asyncio
    from urllib.parse import unquote

    sent: List[Dict[str, Any]] = []

    async def receive() -> Dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
        "scheme": "http", "path": unquote(raw_path), "raw_path": raw_path.encode("latin-1"),
        "root_path": root_path, "query_string": b"", "server": ("127.0.0.1", 8264),
        "client": ("127.0.0.1", 50000), "headers": [(b"host", b"127.0.0.1:8264")],
    }
    asyncio.run(app(scope, receive, send))
    status = next((m.get("status") for m in sent if m.get("type") == "http.response.start"),
                  None)
    return status, b"".join(m.get("body", b"") for m in sent
                            if m.get("type") == "http.response.body")


def _url_fetches(app: Any, forms: Tuple[str, ...]) -> Dict[str, Any]:
    """Each ``file=`` URL form sent to `app` in this process, GET and HEAD,
    while Gradio's URL fetch (`gradio.routes.secure_url_stream_response`, the
    name its ``file=`` route calls) is counted: how many requests reached it
    (none when the route guard answers them first), the requests not answered
    with the guard's refusal, and, as the control, how many one form sent to
    the mounted dashboard directly, past the guard, reached it (one: the count
    is where a fetch happens). Gradio refuses an address that is not public
    itself, so nothing is fetched in either case."""
    import gradio.routes as gradio_routes
    from starlette.routing import Mount

    from src.api.middleware.gradio_routes import REFUSALS

    original = gradio_routes.secure_url_stream_response
    reached: List[str] = []

    async def counted(url: str, request: Any) -> Any:
        reached.append(url)
        return await original(url, request)

    mount = next((route for route in app.routes if isinstance(route, Mount)
                  and getattr(route.app, "blocks", None) is not None), None)
    assert mount is not None, "the application mounts no dashboard"
    not_the_guard = []
    gradio_routes.secure_url_stream_response = counted
    try:
        for form in forms:
            for method in ("GET", "HEAD"):
                status, body = _asgi_answer(app, method, f"{mount.path}/gradio_api/{form}")
                if status != 403 or (method == "GET" and body.decode(
                        "utf-8", "replace") != REFUSALS["file-url"]):
                    not_the_guard.append(f"{method} {status}")
        through = len(reached)
        _asgi_answer(mount.app, "GET", f"{mount.path}/gradio_api/{forms[0]}",
                     root_path=mount.path)
    finally:
        gradio_routes.secure_url_stream_response = original
    return {"through_the_guard": through, "past_the_guard_control": len(reached) - through,
            "not_the_guard": not_the_guard}


def _route_key(method: str, path: str) -> str:
    return f"{method} {path}"


def _gated_api_routes(app: Any) -> List[Tuple[str, str]]:
    """Every API route of `app` that D8's dependency gates
    (`require_operator_access` among its dependencies), as (method, path
    template): a gated route added later is asked too, without a list here
    to keep up."""
    from fastapi.routing import APIRoute

    from src.api.operator_access import require_operator_access

    return sorted((method, route.path) for route in app.routes if isinstance(route, APIRoute)
                  and any(dependency.call is require_operator_access
                          for dependency in route.dependant.dependencies)
                  for method in route.methods)


def _dashboard(app: Any) -> Any:
    """The dashboard's Gradio Blocks, as `app` mounts it."""
    blocks = next((getattr(route.app, "blocks", None) for route in app.routes
                   if getattr(getattr(route, "app", None), "blocks", None) is not None), None)
    assert blocks is not None, "the application mounts no dashboard"
    return blocks


def _gated_events(app: Any) -> List[Tuple[str, str, List[Any], str]]:
    """`GATED_EVENTS` as the dashboard `app` mounts registers them: (label,
    API name, data, allowed answer). Each handler must be registered once,
    with as many inputs as the data the probe sends it."""
    found = []
    for module, name, data, allowed in GATED_EVENTS:
        matches = [dependency for dependency in _dashboard(app).fns.values()
                   if getattr(dependency.fn, "__module__", None) == f"src.webui.components.{module}"
                   and getattr(dependency.fn, "__name__", None) == name]
        assert len(matches) == 1 and len(matches[0].inputs) == len(data), (
            f"the dashboard registers {module}.{name} {len(matches)} time(s), or with other "
            "inputs than the probe sends")
        found.append((f"{module}.{name}", matches[0].api_name, list(data), allowed))
    return found


def _operator_requests(server: _ProbeServer, headers: Mapping[str, str],
                       routes: List[Tuple[str, str]],
                       events: List[Tuple[str, str, List[Any], str]]) -> Dict[str, Any]:
    """What one shape of request meets on `server`: D8's own status for it,
    each gated route in `routes` (a POST with ``[]`` as its JSON body, see
    `OPERATOR_ROUTES`) and each gated dashboard event in `events`
    (`_gated_events`)."""
    access = server.json("/api/v1/system/operator-access", headers)
    answers = {}
    for method, path in routes:
        target = re.sub(r"\{[^}]+\}", M5_WORKSPACE, path)
        if method == "GET":
            answers[_route_key(method, path)] = server.request(method, target, headers)
        else:
            answers[_route_key(method, path)] = server.request(
                method, target, {"Content-Type": "application/json", **headers}, b"[]")
    said = {label: server.event(api_name, data, headers)
            for label, api_name, data, _ in events}
    return {"access": access, "routes": answers, "events": said}


def _check_allowed(seen: Mapping[str, Any], listed_sha: str, label: str,
                   events: List[Tuple[str, str, List[Any], str]]) -> Dict[str, Any]:
    """D8 allowed this request (`_operator_requests`), and each operator route
    then answered as its handler or its validation does (`OPERATOR_ROUTES`:
    the list holds the listed checkpoint, the stop answered for the runner,
    nothing running; a body of the wrong shape refused in validation), and
    each event ran its handler, which did nothing."""
    access = seen["access"]
    assert access.get("enabled") is True and access.get("condition") == "enabled", (
        f"D8 refused {label}: {access.get('condition')}")
    routes = {}
    for method, path, expected in OPERATOR_ROUTES:
        key = _route_key(method, path)
        assert key in seen["routes"], f"{key} was not asked"
        status, _, body = seen["routes"][key]
        text = body.decode("utf-8", "replace")
        assert status == expected and "Refused: " not in text, (
            f"{key} answered {status} to {label}, not {expected}: D8 did not let it through")
        if path.endswith("/{name}/checkpoints"):
            items = json.loads(body).get("checkpoints") or []
            assert any(entry.get("sha256") == listed_sha for entry in items), (
                "the service's checkpoint list does not hold the listed checkpoint")
        if path.endswith("/training/stop"):
            assert "No training is running" in text, "the training stop did not answer for the runner"
        routes[key] = status
    for event_label, _, _, allowed in events:
        said = seen["events"][event_label]
        assert "Refused" not in said and allowed in said, (
            f"the Gradio event {event_label} did not run its handler")
    return {"condition": "enabled", "routes": routes,
            "gradio_events": {event_label: "ran" for event_label, _, _, _ in events}}


def _check_refused(seen: Mapping[str, Any], condition: str, values: Tuple[str, ...],
                   listed_sha: str) -> Dict[str, Any]:
    """D8 refused this request for `condition`, and so did every gated
    control asked: 403 with D8's refusal (which says nothing was done) from
    every route, each of `OPERATOR_ROUTES` among them, the refusal from every
    event, no checkpoint shown, and no header value (`values`) repeated
    anywhere."""
    access = seen["access"]
    assert access.get("enabled") is False and access.get("condition") == condition, (
        f"D8 answered {access.get('condition')}, not {condition}")
    missing = [_route_key(method, path) for method, path, _ in OPERATOR_ROUTES
               if _route_key(method, path) not in seen["routes"]]
    assert not missing and seen["events"], f"not asked: {missing or 'any dashboard event'}"
    texts = [json.dumps(access)]
    for key, (status, _, body) in seen["routes"].items():
        text = body.decode("utf-8", "replace")
        texts.append(text)
        assert status == 403 and "Refused: " in text and "Nothing was done." in text, (
            f"{key} answered {status}, not D8's refusal")
        assert listed_sha not in text, f"{key} showed a checkpoint when refused"
    for event_label, said in seen["events"].items():
        texts.append(said)
        assert "Refused: " in said and "Nothing was done." in said, (
            f"the Gradio event {event_label} was not refused")
    assert not [value for value in values if any(value in text for text in texts)], (
        "a refusal repeated a header's value")
    return {"condition": condition,
            "routes": {key: answer[0] for key, answer in seen["routes"].items()},
            "gradio_events": dict.fromkeys(seen["events"], "refused")}


def _cors_grants(headers: Mapping[str, str]) -> List[str]:
    """The CORS grants among a response's headers (by lowercased name)."""
    return sorted(name for name in headers
                  if name in ("access-control-allow-origin", "access-control-allow-credentials"))


def _check_guard_refusal(raw: bytes, body: Optional[str], what: str) -> None:
    """A raw response is the route guard's refusal: 403 as the first status
    line, so no interim "100 Continue" came before it (uvicorn sends one only
    when the application asks for the body), the connection closed (the body
    it did not read is not taken for a next request), and its own text
    (`body`; None for a HEAD, which has none)."""
    status, headers, received = _parse_response(raw)
    assert status == 403 and headers.get("connection") == "close", (
        f"{what} was answered {status}, not refused with the connection closed")
    assert body is None or received.decode("utf-8", "replace") == body, (
        f"{what} was refused, but not by the route guard")


def _store_runs(reports: Path) -> List[str]:
    """The run directories a version's report store holds now, by name."""
    from src.evaluation.cohort_registry import is_run_id

    try:
        return sorted(name for name in os.listdir(reports) if is_run_id(name))
    except FileNotFoundError:
        return []


def _listed_test_inputs(label: str, workspace: str, run: str, filename: str,
                        **choices: Any) -> Any:
    """A test run's inputs as the Test page builds them
    (`test_page.start_test_run`): the version, the workspace and the checkpoint
    found in the server's lists again by name, the digests those lists show
    passed as the expected ones, and the registry and evidence directories
    from their settings. `choices` are the operator's two (`record`,
    `deterministic`)."""
    from src.api.services import server_lists
    from src.api.services.training_manager import TestRunInputs
    from src.config import server_settings

    version = next((item for item in server_lists.list_cohort_versions().items
                    if item.label == label), None)
    assert version is not None and version.manifest_sha256, (
        f"the server lists no readable cohort version {label}")
    listed = next((item for item in server_lists.list_checkpoints(workspace).items
                   if (item.run, item.filename) == (run, filename)), None)
    assert listed is not None and listed.sha256, (
        f"the server lists no readable checkpoint {run}/{filename} in {workspace}")
    registry, evidence = (server_settings.cohort_registry(),
                          server_settings.training_evidence_dirs())
    assert registry.available and evidence.available, "the registry or evidence setting is refused"
    return TestRunInputs(
        registry=registry.value, version_label=label, version_sha256=version.manifest_sha256,
        workspace=server_lists.resolve_workspace(workspace),
        checkpoint=server_lists.resolve_checkpoint(workspace, listed.architecture, run,
                                                   filename),
        checkpoint_sha256=listed.sha256, evidence_dirs=tuple(evidence.value), **choices)


def _first_line(process: Any, timeout: float) -> str:
    """The first line `process` writes on its standard output, or "" when none
    came within `timeout` seconds (the process is then left to its caller)."""
    import threading

    line: List[str] = []
    reader = threading.Thread(target=lambda: line.append(process.stdout.readline()),
                              daemon=True)
    reader.start()
    reader.join(timeout)
    return line[0] if line else ""


def _job_ended(manager: Any, job_type: str, timeout: float) -> str:
    """The status of `manager`'s latest `job_type` job once it is neither
    running nor stopping (its monitor has read what it left)."""
    status = _wait_for(lambda: manager.get_job_status_word(job_type)
                       not in ("running", "stopping"), timeout, interval=0.5)
    word = manager.get_job_status_word(job_type)
    assert status, f"the {job_type} job had not ended after {timeout:.0f} s (still {word})"
    return word


def _job_said(manager: Any, job_type: str) -> str:
    """The job's own output tail, for the console when a probe fails on it
    (it can name paths, so it never goes into the report)."""
    lines = manager.get_log(job_type)["lines"]
    text = "\n".join(lines[-40:])
    print(f"--- the {job_type} job's last output ---\n{text}\n---", file=sys.stderr)
    return text


def _cli_measure(argv: List[str]) -> Tuple[int, str]:
    """`measure_scorer.py` as the operator runs it (`_run_as_program`, the
    program entry, in this process): its exit status and what it printed on
    standard error."""
    import io

    import scripts.measure_scorer as measure_scorer

    said = io.StringIO()
    with contextlib.redirect_stderr(said):
        code = measure_scorer._run_as_program(argv)
    return code, said.getvalue()


#: The settings a held job is never given: a parent, which would make it a
#: continuation (contract M3a-3), an extension of one (M3a-4), and a pause,
#: which would end it by itself (`_held_config`).
_NOT_HELD = ("resume_from", "extend_epochs", "extend_lr", "pause_after_epoch")


def _held_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """`config` as the job W11 and RW3 hold the GPU lock with runs it: a new
    run, whatever the settings name (no parent, no extension), that ends only
    by the probe's stop. The trainer's three ways of ending by itself are
    closed here, for both callers: the stage's end (`_HOLD_EPOCHS` epochs),
    early stopping (a patience as long, which cannot run out within the
    stage) and a pause (none). Root-cause round, W11-2: the hold is this
    function's, not a copy at each caller."""
    held = {name: value for name, value in config.items() if name not in _NOT_HELD}
    held.update(num_epochs=_HOLD_EPOCHS, early_stopping_patience=_HOLD_EPOCHS)
    return held


def _start_training_job(manager: Any, config: Dict[str, Any], fingerprint: str,
                        lock_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The training job W11 and RW3 hold the GPU lock with: `config` held
    (`_held_config`) and started through the runner, then waited for until its
    own process holds the GPU lock (its holder record names it) and it has
    begun its first epoch on the device (its progress file says so, as the
    runner's status reads it). Returns the start's answer and what the waits
    saw. A job that ends before either fails here, with its output on the
    console. Where `os.waitid` is missing nothing is started
    (`_require_waitid`): without it the check that the job held the lock
    throughout cannot be made."""
    _require_waitid()
    from src.utils.gpu_lock import lock_state

    started = manager.start_training(_held_config(config),
                                     expected_dataset_fingerprint=fingerprint)
    assert started.get("success"), "the runner refused the training start"
    pid = started["pid"]

    def holds() -> bool:
        held_by = lock_state().get("held_by") or ""
        return ("a training job" in held_by and _names_pid(held_by, pid)) or \
            manager.get_job_status_word("training") not in ("running", "stopping")

    _wait_for(holds, _LOCK_TAKEN_SECONDS, interval=0.2)
    if manager.get_job_status_word("training") != "running":
        _job_said(manager, "training")
        raise AssertionError("the training job ended before it held the GPU lock")
    held_by = lock_state().get("held_by") or ""
    assert "a training job" in held_by and _names_pid(held_by, pid), (
        "the GPU lock is not held by the training job's own process")
    def in_an_epoch() -> bool:
        return manager.get_status().get("phase") == "training"

    _wait_for(lambda: in_an_epoch() or manager.get_job_status_word("training") != "running",
              _FIRST_EPOCH_SECONDS, interval=0.5)
    if manager.get_job_status_word("training") != "running":
        _job_said(manager, "training")
        raise AssertionError("the training job ended before its first epoch")
    assert in_an_epoch(), (f"the training job had not begun an epoch after "
                           f"{_FIRST_EPOCH_SECONDS:.0f} s")
    return started, {"held_by_its_own_process": True, "reached_its_first_epoch": True,
                     "lock_file": lock_path.name}


def _exclusion_while_training(manager: Any, inputs: Any, pid: int, cli_argv: List[str],
                              outcome: Path) -> Dict[str, Any]:
    """While `manager`'s training job runs and holds the GPU lock: a test
    start on the same runner is refused for the slot, a test cancel does not
    reach training, and the measurement CLI, another process, is refused by
    the lock naming the job's own process. Nothing is started or written.

    **One bracket around the whole check** (review of 8242b47, W11/RW3;
    root-cause round, W11-1): that the job runs and is the lock's recorded
    holder is checked before anything is tried and once more after the CLI,
    and every answer between is observed, not judged, until then. A job holds
    the lock from its start to its end, so running and named both times means
    it held it throughout and every answer is the product's.

    **What reached training is read from the runner's own record after each
    action** (verification of W11-1, P-1): training's status word and whether
    its process runs, right after the test start (with the stop of a test run
    it began), the test cancel and the CLI. The runner says "stopping" or
    "stopped" only after a stop was asked of it (`TrainingManager._stop`,
    `_final_status`), so the first action after which it says so is the one
    that reached training, whether the job then ended or not. A job that
    ended with no stop asked (it cannot end by itself, `_held_config`) is said
    with the action during which it was first seen ended and how the runner
    says it ended. That record does not tell a crash from a signal sent
    outside the runner's stop (the trainer ends a SIGINT as a failure with
    exit code 1, as it ends a crash), so no cause is claimed. A job that
    still runs but is no longer the lock's recorded holder could not hold the
    lock. Never a pass or a skip, and never read as a start or the lock
    failing to refuse.
    """
    from src.api.services.training_manager import BUSY_STATUSES, RESULT_WAIT_SECONDS

    before = _holding_job(manager, pid)
    assert not before, (f"the training job could not hold the GPU lock for the check: "
                        f"before a test start was tried, {before}")
    # Observed, not judged yet: after each action, training's status word and
    # whether its process runs. A test run that a start not refused began is
    # stopped before the cancel is tried, whatever its answer turns out to mean.
    seen: List[Tuple[str, str, bool]] = []

    def observe(action: str) -> None:
        seen.append((action, manager.get_job_status_word("training"), _job_runs(pid)))

    with _job_ended_on_exit(manager, "test"):
        started = manager.start_test(inputs)
    observe("the test start")
    cancelled = manager.cancel_test()
    observe("the test cancel")
    code, said = _cli_measure(cli_argv)
    observe("the measurement CLI")
    after = _holding_job(manager, pid)
    if after and not _job_runs(pid):
        # Ended: how, once the runner's monitor has recorded it.
        _wait_for(lambda: manager.get_job_status_word("training") not in BUSY_STATUSES,
                  RESULT_WAIT_SECONDS)
        observe("the measurement CLI")
    reached = [(action, word) for action, word, _ in seen if word in ("stopping", "stopped")]
    assert not reached, (f"{reached[0][0]} reached training: the runner says {reached[0][1]} "
                         "right after it, which only a stop asked of the runner gives")
    if after:
        ended = [action for action, word, runs in seen if not runs or word != "running"]
        assert not ended, (
            f"the training job ended during {ended[0]} with no stop asked of the runner "
            f"({_how_training_ended(manager)}): a crash, or a signal sent outside the runner's "
            "stop, which its record does not tell apart; the exclusion was not exercised")
        raise AssertionError(f"the training job could not hold the GPU lock for the check: "
                             f"after a test start, a test cancel and the measurement CLI (exit "
                             f"{code}) were tried, {after}, so the exclusion was not exercised")
    start_error, cancel_error = started.get("error") or "", cancelled.get("error") or ""
    assert not started.get("success") and "Training is already running" in start_error \
        and "Nothing was started." in start_error, "a test start was not refused"
    assert not cancelled.get("success") and "No test run is running" in cancel_error \
        and "does not reach" in cancel_error, "a test cancel was not refused"
    assert code == 1 and "a training job" in said and _names_pid(said, pid) \
        and "Nothing was written." in said, (
            f"the measurement CLI was not refused by the lock naming the training job "
            f"(exit {code})")
    assert not outcome.exists(), "the refused CLI wrote its outcome file"
    return {"test_start": "refused: the GPU slot is the training job's",
            "test_cancel": "refused: a test cancel does not reach training",
            "measurement_cli": "refused by the GPU lock, naming the training job's process",
            "cli_exit": code}


def _how_training_ended(manager: Any) -> str:
    """How the runner `manager` says its latest training job ended: its
    status word and exit status, a signal named as the runner names it."""
    from src.api.services.training_manager import _signal_name

    status = manager.get_status()
    code = status.get("exit_code")
    how = ("its exit status not recorded" if code is None
           else f"ended by {_signal_name(code)}" if code < 0 else f"exit code {code}")
    return f"the runner says {status.get('status')}, {how}"


def _training_stop_does_not_reach_a_test_run(manager: Any) -> str:
    """Right after `manager` started a test run: its training stop is refused,
    saying it does not reach the test run, and the run is still running (the
    caller then requires it to complete)."""
    stopped = manager.stop_training()
    error = str(stopped.get("error") or "")
    assert not stopped.get("success") and "No training is running" in error \
        and "does not reach" in error, "a training stop was not refused while a test run ran"
    assert manager.get_job_status_word("test") in ("running", "completed"), (
        f"the training stop reached the test run ({manager.get_job_status_word('test')})")
    return "refused: a training stop does not reach a test run"


def _stop_training_job(manager: Any, lock_path: Path) -> Dict[str, Any]:
    """The runner's training stop (SIGINT, then SIGTERM, then SIGKILL): the job
    ends as stopped, never completed, and the GPU lock is free once it has."""
    from src.utils.gpu_lock import probe as lock_probe

    stopped = manager.stop_training()
    assert stopped.get("success") and stopped.get("status") == "stopped", (
        f"the stop answered {stopped.get('status')}, not stopped")
    status = manager.get_status()
    assert status["status"] == "stopped", f"the stopped job reads {status['status']}"
    assert _wait_for(lambda: lock_probe(lock_path) is None, 10.0), (
        "the GPU lock was still held after the training job ended")
    assert not manager.gpu_state()["busy"], "the runner still holds its GPU slot"
    return {"status": "stopped", "exit_code": status.get("exit_code"),
            "snapshot_left": stopped.get("outcome") is not None, "lock_released": True}


def _m5_demo(report: Report, work: Path, sound: Path, checkpoint: Path, device: str,
             child: Optional[Path], registry: Path, version: Any) -> None:
    """M5's acceptance on the demo workspace (W1-W11), after M4's flow, on the
    probe's own deployment under <work>/m5 with M2's resumed checkpoint
    (`child`) and T1's cohort version (`version` in `registry`).

    Functional only: what M5 adds to serving and evaluation (the server's
    lists, D8 on every operator control, no cross-origin access, the route
    guard, the GPU lock, test and training jobs through the job runner,
    registration from the report store), checked on this machine through the
    production entry points: the service started as the launcher starts it,
    in its own process on a free loopback port; the job runner and the CLIs
    running the real scripts. Not the UI's layout or a browser session.
    """
    print("   M5: the lists, D8, CORS, the route guard, the GPU lock and the jobs")
    from src.api.middleware.gradio_routes import REFUSALS
    from src.api.services import server_lists
    from src.api.services.training_manager import TrainingManager, build_test_argv
    from src.evaluation.cohort_registry import REPORT_FILE, is_run_id, resolve_version
    from src.evaluation.report_store import read_usage_history
    from src.kg.artifacts import MANIFEST_FILENAME
    from src.kg.storage.file_storage import _read_json
    from src.utils.checkpoint_paths import RUNS_DIRNAME, run_checkpoint_dir
    from src.utils.gpu_lock import lock_location, lock_state
    from src.utils.gpu_lock import probe as lock_probe

    m5 = work / "m5"
    root = m5 / "workspaces"
    reports = registry / "versions" / M4_DEMO_LABEL / "reports"
    state: Dict[str, Any] = {}
    settings = {"SHEPHERD_WORKSPACES_ROOT": str(root), "SHEPHERD_REGISTRY": str(registry),
                "SHEPHERD_DEVICE": device}
    servers = {
        "M1": _ProbeServer("M1", m5 / "service-m1",
                           {**settings, "SHEPHERD_DEPLOYMENT_MODE": "M1"}),
        "read-only": _ProbeServer("read-only", m5 / "service-read-only", settings),
    }

    def _need(*keys: str) -> None:
        missing = [key for key in keys if key not in state]
        assert not missing, f"W1's {missing[0]} is not available"

    def _listed() -> Tuple[str, Dict[str, Any]]:
        assert child is not None, "the resumed checkpoint (M2) is not available"
        assert version is not None, "the cohort version (T1) is not available"
        workspace = _workspace_copy(sound, root / M5_WORKSPACE)
        runs = workspace / RUNS_DIRNAME / _M5_ARCHITECTURE
        listed = run_checkpoint_dir(runs / M5_LISTED_RUN) / M5_CHECKPOINT
        listed.parent.mkdir(parents=True)
        shutil.copyfile(child, listed)
        linked = run_checkpoint_dir(runs / M5_LINKED_RUN) / M5_CHECKPOINT
        linked.parent.mkdir(parents=True)
        try:
            linked.symlink_to(checkpoint.resolve())
            link_made = True
        except OSError:                  # a filesystem without symbolic links
            link_made = False
        listed_sha = _read_run_checkpoint(listed)["sha256"]
        _, _, reads, record = verify_workspace(workspace)
        fingerprint = reads[MANIFEST_FILENAME]

        workspaces = server_lists.list_workspaces()
        assert workspaces.available, "the server lists no workspaces"
        assert [item.name for item in workspaces.items] == [M5_WORKSPACE], (
            "the probe's workspace root lists other entries than its one workspace")
        item = workspaces.items[0]
        assert item.dataset_fingerprint == fingerprint, (
            "the listed dataset fingerprint is not the digest of the workspace's manifest")
        assert item.trainable and item.not_trainable is None, "the workspace is not trainable"
        assert (item.dataset_record.get("binding"), item.dataset_record.get("origin")) == (
            "verified", record.origin), "the listed dataset record is not the verified one"
        checkpoints = server_lists.list_checkpoints(M5_WORKSPACE)
        assert checkpoints.available, "the server lists no checkpoints for the workspace"
        by_run = {entry.run: entry for entry in checkpoints.items}
        mine = by_run.get(M5_LISTED_RUN)
        assert mine is not None, "the listed run's checkpoint is not in the list"
        assert (mine.architecture, mine.filename, mine.sha256, mine.trained_on_this_dataset,
                mine.problem) == (_M5_ARCHITECTURE, M5_CHECKPOINT, listed_sha, True, None), (
            "the listed checkpoint is not named by its run, file and digest as read")
        if link_made:
            out = by_run.get(M5_LINKED_RUN)
            assert out is not None and out.sha256 is None and out.size_bytes is None and (
                "resolves outside the workspace" in (out.problem or "")), (
                "a checkpoint linked out of the workspace was opened or not listed with why")
        # **No item carries a server path** for a client to send back (a
        # reason may name one, for the operator).
        shown = json.dumps(_identity_fields(
            [workspaces.as_dict("workspaces"), checkpoints.as_dict("checkpoints")]))
        assert str(root.resolve()) not in shown and str(work.resolve()) not in shown, (
            "a listed item carries a server path")
        assert server_lists.resolve_checkpoint(
            M5_WORKSPACE, _M5_ARCHITECTURE, M5_LISTED_RUN, M5_CHECKPOINT) == listed
        refused = [
            refuses(lambda: server_lists.resolve_checkpoint(
                M5_WORKSPACE, _M5_ARCHITECTURE, M5_LISTED_RUN, "../" + M5_CHECKPOINT),
                server_lists.ServerListRefused, "is listed in the workspace"),
            refuses(lambda: server_lists.resolve_workspace("../m4"),
                    server_lists.ServerListRefused, "is not a workspace the server lists"),
        ]
        if link_made:
            refused.append(refuses(lambda: server_lists.resolve_checkpoint(
                M5_WORKSPACE, _M5_ARCHITECTURE, M5_LINKED_RUN, M5_CHECKPOINT),
                server_lists.ServerListRefused, "resolves outside the workspace"))
        versions = server_lists.list_cohort_versions(
            report_state=server_lists.stored_report_state)
        assert versions.available, "the server lists no cohort versions"
        demo = next((entry for entry in versions.items if entry.label == M4_DEMO_LABEL), None)
        assert demo is not None and demo.problem is None, "T1's version is not listed"
        assert (demo.manifest_sha256, demo.role, demo.offers_ordering) == (
            version.identity.sha256, M4_ROLE, False), (
            "the version is not listed by T1's digest and role, ordering off by default")
        # The store's run directories (its own files, the usage history and
        # the notes, are not runs).
        runs_listed = [run for run in demo.reports if is_run_id(run.run_id)]
        kept = [(run.state or {}).get("kept") for run in runs_listed]
        registered = [(run.state or {}).get("registered") for run in runs_listed]
        assert kept == ["kept"] and registered == ["registered"], (
            f"T2's run is listed as kept {kept} and registered {registered}")
        state.update(workspace=workspace, listed=listed, listed_sha=listed_sha,
                     fingerprint=fingerprint, link_made=link_made)
        return "listed by name and digest, resolved only from the lists, no server path", {
            "workspaces": [{"name": item.name, "dataset_fingerprint": item.dataset_fingerprint,
                            "trainable": item.trainable, "dataset_record": {
                                "binding": item.dataset_record.get("binding"),
                                "origin": item.dataset_record.get("origin")},
                            "build": item.build.get("state")}],
            "checkpoints": [{"run": entry.run, "filename": entry.filename,
                             "sha256": entry.sha256, "listed_with_a_problem": bool(entry.problem)}
                            for entry in checkpoints.items],
            "link_out_of_the_workspace": ("listed, not opened" if link_made
                                          else "not made: no symbolic links here"),
            "names_refused_by_resolution": len(refused),
            "cohort_version": {"label": demo.label, "manifest_sha256": demo.manifest_sha256,
                               "role": demo.role, "offers_ordering": demo.offers_ordering,
                               "reports": [{"kept": k, "registered": r}
                                           for k, r in zip(kept, registered, strict=True)]},
            "items_carry_no_server_path": True,
            "deployment_gpu_lock": deployment_lock,
        }

    # ---- The service itself, both modes, started side by side
    def _start_services() -> None:
        if "workspace" not in state:
            return
        for server in servers.values():
            try:
                server.start()
            except Exception as exc:  # noqa: BLE001 - said by every probe that needs it
                print(f"   the {server.name} service could not be started: {exc}",
                      file=sys.stderr)
        for server in servers.values():
            if server.process is not None:
                try:
                    server.wait_ready()
                except AssertionError as exc:
                    print(f"   {exc}", file=sys.stderr)

    def _server(name: str) -> _ProbeServer:
        server = servers[name]
        assert server.ready, (f"the {name} service is not running (W1 did not pass, or it "
                              "did not start; its log is under the work directory)")
        return server

    def _controls() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, List[Any], str]]]:
        """The gated routes and dashboard events W2 and W5 ask, from the
        application as this process imports it (the service runs the same
        code): D8's operator routes and any other route its dependency gates,
        and `GATED_EVENTS` by their registered names."""
        import src.api.main as api

        if "controls" not in state:
            routes = sorted({(method, path) for method, path, _ in OPERATOR_ROUTES}
                            | set(_gated_api_routes(api.app)))
            state["controls"] = (routes, _gated_events(api.app))
        return state["controls"]

    def _d8_on_the_service() -> Tuple[str, Dict[str, Any]]:
        _need("listed_sha")
        server = _server("M1")
        port = server.port
        routes, events = _controls()
        # **Allowed, nothing done**: only D8's own operator routes, whose
        # answers to the probe's requests run nothing (`OPERATOR_ROUTES`).
        operator = [(method, path) for method, path, _ in OPERATOR_ROUTES]
        results: Dict[str, Any] = {}
        for label, headers in (
            ("loopback, no Origin (a command-line client)", {}),
            ("the dashboard's own page", {"Host": f"localhost:{port}",
                                          "Origin": f"http://localhost:{port}",
                                          "Sec-Fetch-Site": "same-origin"}),
        ):
            results[label] = _check_allowed(_operator_requests(server, headers, operator, events),
                                            state["listed_sha"], label, events)
        for label, headers, condition, values in (
            ("a forwarding header (X-Forwarded-For)", {"X-Forwarded-For": "203.0.113.9"},
             "forwarding_header", ("203.0.113.9",)),
            ("a forwarding header (Forwarded)", {"Forwarded": "for=203.0.113.10"},
             "forwarding_header", ("203.0.113.10",)),
            ("a foreign Host (DNS rebinding)", {"Host": f"rebind.example:{port}"},
             "host_not_loopback", ("rebind.example",)),
            ("a foreign Origin", {"Origin": "https://other.example"}, "origin_mismatch",
             ("other.example",)),
            ("cross-site, no Origin (another site's image or link)",
             {"Sec-Fetch-Site": "cross-site"}, "cross_site", ()),
        ):
            results[label] = _check_refused(_operator_requests(server, headers, routes, events),
                                            condition, values, state["listed_sha"])
        return "allowed in M1 on loopback; refused, nothing done, for each other request", {
            "service": {"started_as": "uvicorn src.api.main:app on 127.0.0.1, "
                                      "--no-proxy-headers, SHEPHERD_DEPLOYMENT_MODE=M1",
                        "seconds_to_health": server.ready_seconds},
            "gated": {"api_routes": [_route_key(method, path) for method, path in routes],
                      "gradio_events": [label for label, _, _, _ in events]},
            "requests": results,
            "refusals_repeat_no_header_value": True,
        }

    def _cors() -> Tuple[str, Dict[str, Any]]:
        server = _server("M1")
        port = server.port
        foreign = {"Origin": "https://other.example"}
        preflight = {**foreign, "Access-Control-Request-Method": "POST",
                     "Access-Control-Request-Headers": "content-type"}
        seen: Dict[str, Any] = {}
        for label, method, path, headers, expected in (
            ("GET /api", "GET", "/api/v1/workspaces", foreign, 200),
            ("GET /ui/config", "GET", "/ui/config", foreign, 200),
            ("GET /ui/config, a LAN Host", "GET", "/ui/config",
             {**foreign, "Host": f"192.0.2.10:{port}"}, 200),
            ("preflight /api", "OPTIONS", "/api/v1/pipeline/reload", preflight, 400),
            ("preflight /ui", "OPTIONS", "/ui/gradio_api/queue/join", preflight, 400),
        ):
            status, headers_back, _ = server.request(method, path, headers)
            granted = _cors_grants(headers_back)
            assert status == expected and not granted, (
                f"{label} answered {status} with {granted or 'no grant'}")
            seen[label] = {"status": status, "access_control_allow_origin": False}
        status, _, _ = server.request("GET", "/ui/config",
                                      {"Origin": f"http://127.0.0.1:{port}"})
        assert status == 200, f"the same-origin control answered {status}"
        return "no Access-Control-Allow-Origin for a foreign Origin; preflights refused", {
            "foreign_origin": seen, "same_origin_control": status,
        }

    def _route_guard() -> Tuple[str, Dict[str, Any]]:
        import src.api.main as api

        server = _server("M1")
        own = f"127.0.0.1:{server.port}".encode()
        # A documentation address (TEST-NET-2), never routed: the forms are
        # spelled for the guard, not for a fetch to land anywhere.
        target = "198.51.100.7"
        forms = (f"file=http://{target}/probe.png", f"file=https://{target}/probe.json",
                 f"file=HTTP://{target}/probe.png", f"file=%68ttp://{target}/probe.png",
                 f"file=http:%2F%2F{target}/probe.txt", f"file=//{target}/probe.png",
                 f"file=\\\\{target}/probe.png", f"file=/\\{target}/probe.png",
                 f"file/http://{target}/probe.png",
                 f"file=http://user:pw@{target}/probe.txt")
        refused = 0
        for form in forms:
            for method in ("GET", "HEAD"):
                _check_guard_refusal(server.raw(
                    f"{method} /ui/gradio_api/{form} HTTP/1.1\r\n".encode("latin-1")
                    + b"Host: " + own + b"\r\nConnection: close\r\n\r\n"),
                    None if method == "HEAD" else REFUSALS["file-url"],
                    f"a file= URL form ({method})")
                refused += 1
        # **No fetch, counted where a fetch happens**: in this process, the
        # same forms through the application while Gradio's URL fetch is
        # counted (a listener cannot tell: Gradio refuses a loopback or
        # reserved address itself, before it connects).
        fetches = _url_fetches(api.app, forms)
        assert not fetches["not_the_guard"], (
            f"in process, file= URL requests were answered otherwise than by the route guard: "
            f"{fetches['not_the_guard']}")
        assert fetches["through_the_guard"] == 0, (
            f"{fetches['through_the_guard']} file= URL request(s) reached Gradio's URL fetch")
        assert fetches["past_the_guard_control"] == 1, (
            "the control sent past the route guard did not reach Gradio's URL fetch: the "
            "count is not where a fetch happens")
        # The control: a local download in Gradio's own directory still works.
        from urllib.parse import quote

        download = server.gradio_dir / "probe-download.csv"
        download.write_bytes(b"epoch,split,total\n1,train,1.5\n")
        status, _, body = server.request("GET", f"/ui/gradio_api/file={quote(str(download))}")
        assert status == 200 and body == download.read_bytes(), (
            f"a local file= download answered {status}")
        # **Refused before the body is read**: 50 MB announced, nothing sent,
        # 100-continue asked for. An answer that is the refusal, with no
        # interim "100 Continue", came without the body being asked for.
        bodies = 0
        for path, which in (("/ui/gradio_api/process_recording", "process_recording"),
                            ("/ui/gradio_api/process_recording/", "process_recording"),
                            ("/ui/gradio_api/process%5Frecording", "process_recording"),
                            ("/ui/gradio_api/upload", "upload"),
                            ("/ui/gradio_api/upload/", "upload"),
                            ("/ui%2Fgradio_api%2Fupload", "upload")):
            _check_guard_refusal(server.raw(
                f"POST {path} HTTP/1.1\r\n".encode("latin-1") + b"Host: " + own
                + b"\r\nContent-Type: multipart/form-data; boundary=B\r\n"
                + b"Content-Length: 50000000\r\nExpect: 100-continue\r\n\r\n"),
                REFUSALS[which], which)
            bodies += 1
        status, _, _ = server.request("GET", "/ui/gradio_api/upload_progress?upload_id=probe")
        assert status == 403, f"upload_progress answered {status}"
        asked = {}
        for path in ("/ui/gradio_api/process_recording", "/ui/gradio_api/upload"):
            status, calls = _asgi_body_requests(api.app, "POST", path)
            assert status == 403 and calls == 0, (
                f"in process, {path} answered {status} after asking for the body {calls} time(s)")
            asked[path.rsplit("/", 1)[-1]] = calls
        return "each file= URL form sent, process_recording and upload refused, no body read", {
            "file_url_requests_refused": refused, "file_url_forms": len(forms),
            "url_fetches_through_the_guard": fetches["through_the_guard"],
            "url_fetch_control_past_the_guard": fetches["past_the_guard_control"],
            "local_download_control": 200,
            "body_routes_refused_with_close": bodies, "interim_100_continue": False,
            "upload_progress": 403, "in_process_body_requests": asked,
        }

    def _read_only() -> Tuple[str, Dict[str, Any]]:
        _need("fingerprint")
        server = _server("read-only")
        routes, events = _controls()
        refused = _check_refused(_operator_requests(server, {}, routes, events),
                                 "deployment_mode", (), state["listed_sha"])
        open_routes = {}
        for path in ("/health", "/ui/"):
            status = server.request("GET", path)[0]
            assert status == 200, f"the read-only route {path} answered {status}"
            open_routes[path] = status
        listed = server.json("/api/v1/workspaces").get("workspaces") or []
        assert [(w.get("name"), w.get("dataset_fingerprint")) for w in listed] == [
            (M5_WORKSPACE, state["fingerprint"])], "the workspace list is not W1's"
        labels = [v.get("label") for v in server.json("/api/v1/cohorts").get("versions") or []]
        assert M4_DEMO_LABEL in labels, "the cohort list does not hold T1's version"
        open_routes.update({"/api/v1/workspaces": 200, "/api/v1/cohorts": 200})
        return "without M1 every gated API route and the dashboard events asked are refused; " \
               "the read-only routes answer", {
            "service": {"started_as": "uvicorn src.api.main:app on 127.0.0.1, "
                                      "--no-proxy-headers, SHEPHERD_DEPLOYMENT_MODE unset",
                        "seconds_to_health": server.ready_seconds},
            "loopback_request": refused, "read_only_routes": open_routes,
        }

    def _off_loopback() -> Tuple[str, Dict[str, Any]]:
        from fastapi.testclient import TestClient

        import src.api.main as api

        _need("listed_sha")
        seen = {}
        # **The arrival address is the ASGI scope's** `server`, which a test
        # client sets from its base URL: off loopback cannot be produced on a
        # service bound to loopback, so it is asked of the application here.
        with _deployment_settings({**settings, "SHEPHERD_DEPLOYMENT_MODE": "M1"}):
            for label, base_url, allowed in (("a LAN address", "http://192.0.2.10:8264", False),
                                             ("loopback (control)", "http://127.0.0.1:8264",
                                              True)):
                # Not entered as a context: no lifespan, no pipeline built.
                client = TestClient(api.app, base_url=base_url)
                try:
                    access = client.get("/api/v1/system/operator-access").json()
                    listed = client.get(f"/api/v1/workspaces/{M5_WORKSPACE}/checkpoints")
                    stop = client.post("/api/v1/training/stop")
                finally:
                    client.close()
                if allowed:
                    assert access.get("condition") == "enabled" and listed.status_code == 200 \
                        and stop.status_code == 409, f"the control was refused: {access}"
                else:
                    assert access.get("condition") == "not_loopback", (
                        f"D8 answered {access.get('condition')}, not not_loopback")
                    assert listed.status_code == 403 and state["listed_sha"] not in listed.text \
                        and stop.status_code == 403 and "Nothing was done." in stop.text, (
                            "a request off loopback was not refused")
                seen[label] = {"condition": access.get("condition"),
                               "checkpoint_list": listed.status_code,
                               "training_stop": stop.status_code}
        return "a request off loopback is refused in M1; the loopback control is allowed", {
            "arrival": "the ASGI scope's server, set by the test client's base URL",
            "requests": seen,
        }

    # ---- The GPU lock and the jobs, in this process's runner, under the
    # probe's settings: the lock under its workspace root, the device asked.
    def _lock_path() -> Path:
        path = lock_location()
        assert path.parent == root, "the GPU lock is not the probe's, under its workspace root"
        return path

    def _cli_argv(inputs: Any, outcome: Path) -> List[str]:
        """The command line the runner would run for `inputs`, less the
        interpreter and script (`build_test_argv`)."""
        from src.api.services.training_manager import deployment_device

        return build_test_argv(inputs, deployment_device(), outcome)[2:]

    def _gpu_lock() -> Tuple[str, Dict[str, Any]]:
        import subprocess

        import yaml

        _need("workspace", "fingerprint")
        lock_path = _lock_path()
        inputs = _listed_test_inputs(M4_DEMO_LABEL, M5_WORKSPACE, M5_LISTED_RUN, M5_CHECKPOINT)
        before = _store_runs(reports)
        holder = subprocess.Popen([sys.executable, "-c", _LOCK_HOLDER], cwd=project_root,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True)
        try:
            said = _first_line(holder, 120.0)
            assert said.startswith("held "), "the holder did not take the GPU lock"
            held_by = lock_state().get("held_by") or ""
            assert "a probe-holder job" in held_by and _names_pid(held_by, holder.pid), (
                "the lock's state does not name its holder")
            manager = TrainingManager()
            started = manager.start_test(inputs)
            assert not started.get("success") and "held by a probe-holder job" in started[
                "error"] and "Nothing was started." in started["error"], (
                "the runner started a test run on a busy lock")
            training = manager.start_training(
                {**_DEMO_TRAINING, "data_dir": str(state["workspace"]), "num_epochs": 1,
                 "run_dir": str(m5 / "jobs" / "refused-training"), "device": device},
                expected_dataset_fingerprint=state["fingerprint"])
            assert not training.get("success") and "held by a probe-holder job" in training[
                "error"], "the runner started training on a busy lock"
            assert manager.get_job_status_word("test") == "idle" and \
                manager.get_job_status_word("training") == "idle", "a job was started"
            outcome = m5 / "cli" / "outcome.json"
            outcome.parent.mkdir(parents=True, exist_ok=True)
            code, said = _cli_measure(_cli_argv(inputs, outcome))
            assert code == 1 and "held by a probe-holder job" in said and (
                "Nothing was written." in said), (
                f"the measurement CLI was not refused (exit {code})")
            assert not outcome.exists(), "the refused measurement CLI wrote its outcome file"
            config = m5 / "cli" / "train.yaml"
            refused_run = m5 / "cli" / "refused-run"
            config.write_text(yaml.safe_dump({
                **_DEMO_TRAINING, "data_dir": str(state["workspace"]), "num_epochs": 1,
                "run_dir": str(refused_run), "device": device}), encoding="utf-8")
            trained = subprocess.run(
                [sys.executable, str(project_root / "scripts" / "train_model.py"),
                 "--config", str(config)],
                cwd=project_root, capture_output=True, text=True, timeout=900)
            output = trained.stdout + trained.stderr
            assert trained.returncode == 1 and "refused" in output and (
                "held by a probe-holder job" in output) and "Nothing was written." in output, (
                f"the training CLI was not refused by the lock (exit {trained.returncode})")
            assert not refused_run.exists(), "the refused training CLI made its run directory"
            assert _store_runs(reports) == before, "a refused start left a report directory"
        finally:
            holder.kill()            # **SIGKILL**: the holder cannot release it itself
            holder.wait(timeout=30)
            holder.stdin.close()
            holder.stdout.close()
        released = _wait_for(lambda: lock_probe(lock_path) is None, 10.0)
        assert released, "the GPU lock stayed held after its holder was killed"
        return "a busy lock refused the runner and both CLIs, naming its holder; released " \
               "when the holder was killed", {
                   "lock": "under SHEPHERD_WORKSPACES_ROOT, the lock's one name",
                   "holder": "a process that took the lock through take_gpu_lock",
                   "refused": {"runner_test_start": "named the holder",
                               "runner_training_start": "named the holder",
                               "measure_scorer.py": f"exit {code}, named the holder",
                               "train_model.py": f"exit {trained.returncode}, named the holder"},
                   "nothing_started_or_written": True,
                   "released_after_SIGKILL": True,
               }

    def _test_job() -> Tuple[str, Dict[str, Any]]:
        _need("listed", "listed_sha")
        lock_path = _lock_path()
        inputs = _listed_test_inputs(M4_DEMO_LABEL, M5_WORKSPACE, M5_LISTED_RUN, M5_CHECKPOINT,
                                     record=False)
        before = _store_runs(reports)
        usage_before = len(read_usage_history(version)[0])
        manager = TrainingManager()
        with _job_ended_on_exit(manager, "test"):
            started = manager.start_test(inputs)
            assert started.get("success"), "the runner refused the test run"
            # **A stop reaches only its job type**, this way too: a training
            # stop while the run runs is refused and the run completes.
            not_reached = _training_stop_does_not_reach_a_test_run(manager)
            word = _job_ended(manager, "test", _TEST_RUN_SECONDS)
        status = manager.get_test_status()
        log = "\n".join(manager.get_log("test")["lines"])
        if word != "completed" or status["exit_code"] != 0:
            _job_said(manager, "test")
        assert word == "completed" and status["exit_code"] == 0, (
            f"the test run ended {word}, exit {status['exit_code']}")
        assert "GPU lock" in log and "held" in log, "the run did not take the GPU lock itself"
        created = sorted(set(_store_runs(reports)) - set(before))
        assert len(created) == 1, f"the run left {len(created)} report directories"
        outcome = status["outcome"] or {}
        report_file = reports / created[0] / REPORT_FILE
        assert outcome.get("report_dir") == str(reports / created[0]), (
            "the runner's outcome names another report directory than the run created")
        published = (outcome.get("run_outcome") or {}).get("report_sha256")
        # The kept report, parsed and identified from one read.
        kept, kept_read = _read_json(report_file)
        assert published == kept_read.sha256, (
            "the outcome file's digest is not the kept report's bytes")
        # **Kept and registered as the publication contract says**, never
        # inferred from the exit status: kept in the store, not registered
        # (the run was not asked to record).
        assert (outcome.get("kept"), outcome.get("registered")) == (True, False), (
            f"the outcome says kept {outcome.get('kept')}, registered {outcome.get('registered')}")
        assert kept["manifest"]["artifact_digests"]["checkpoint"] == state["listed_sha"]
        assert kept["manifest"]["cohort_version"]["manifest_sha256"] == version.identity.sha256
        entries, problems = read_usage_history(version)
        new = entries[usage_before:]
        assert not problems and [entry.get("kind") for entry in new] == ["test_run"], (
            f"the usage history gained {[entry.get('kind') for entry in new]}")
        entry = new[0]
        assert (entry.get("run_id"), entry.get("checkpoint_sha256"), entry.get("report_sha256"),
                entry.get("record_requested")) == (created[0], state["listed_sha"], published,
                                                   False), "the usage entry is not this run's"
        assert str(entry.get("device", "")).split(":")[0] == device, (
            f"the run measured on {entry.get('device')}, not {device}")
        assert lock_probe(lock_path) is None, "the GPU lock stayed held after the run ended"
        state["job"] = {"run_id": created[0], "report": report_file, "published": published}
        return "a test run from the server's lists: its report kept, not registered, in the " \
               "usage history", {
                   "inputs": "the version, workspace and checkpoint found in the server's lists "
                             "again, their digests passed as the expected ones",
                   "status": word, "exit_code": status["exit_code"],
                   "exit_meaning": status.get("exit_meaning"),
                   "outcome_file_names_the_kept_report": True, "report_sha256": published,
                   "kept": True, "registered": False,
                   "usage_history": {"kind": "test_run", "record_requested": False,
                                     "device": entry.get("device"),
                                     "n_ranked": entry.get("n_ranked")},
                   "gpu_lock": "taken by the run itself, released at its end",
                   "training_stop_while_it_ran": not_reached,
               }

    def _registered() -> Tuple[str, Dict[str, Any]]:
        from src.evaluation.registration import RegistrationRefused, register_report, report_state
        from src.evaluation.sidecar import read_ledger

        assert "job" in state, "the test run (W8) is not available"
        job, listed = state["job"], state["listed"]
        # **Found in the server's list of the version's runs**, as the Test
        # page's Register finds it, never from a client's path.
        path = server_lists.resolve_report(M4_DEMO_LABEL, job["run_id"])
        assert path == job["report"], "the listed run's report is not the run's"
        ledger_dir = listed.resolve().parent
        before = snapshot(ledger_dir, include_dirs=True)
        copy = m5 / "outside-the-store" / REPORT_FILE
        copy.parent.mkdir(parents=True)
        shutil.copyfile(path, copy)
        refuses(lambda: register_report(copy, checkpoint=listed), RegistrationRefused,
                ("is not a report the registry's report store holds", "Nothing was written"))
        assert_untouched(before, ledger_dir, "a refused registration", include_dirs=True)
        usage_before = len(read_usage_history(version)[0])
        result = register_report(path, checkpoint=listed)
        sequence = result.record["registration"]["sequence"]
        assert (result.outcome, sequence, result.usage_history_problem) == (
            "registered", 1, None), f"the registration was {result.outcome}"
        ledger, ledger_sha256 = read_ledger(ledger_dir / "evaluations.json")
        assert [record.get("source_artifact_digest") for record in ledger["records"]] == [
            job["published"]], "the ledger does not hold one record built from the kept report"
        after = report_state(path, checkpoint=listed, expected_sha256=job["published"])
        assert (after.kept, after.registered, after.sequence) == ("kept", "registered", 1), (
            f"the report reads {after.kept}, {after.registered}")
        new = read_usage_history(version)[0][usage_before:]
        assert [(e.get("kind"), e.get("run_id"), e.get("sequence")) for e in new] == [
            ("registered", job["run_id"], 1)], "the usage history did not gain the registration"
        demo = server_lists.cohort_version_item(M4_DEMO_LABEL,
                                                report_state=server_lists.stored_report_state)
        listed_run = next(run for run in demo.reports if run.run_id == job["run_id"])
        assert ((listed_run.state or {}).get("kept"), (listed_run.state or {}).get("registered")) \
            == ("kept", "registered"), "the server's list does not show the run registered"
        return "a copy outside the store refused; the store's report registered from the list", {
            "copy_outside_the_store": "refused, the ledger unchanged",
            "registered_from": "the report the server lists for the run",
            "outcome": result.outcome, "sequence": sequence, "ledger_sha256": ledger_sha256,
            "report_state": {"kept": after.kept, "registered": after.registered},
            "usage_history": "registered", "server_list": "kept, registered",
        }

    def _start_checked_again() -> Tuple[str, Dict[str, Any]]:
        import dataclasses

        _need("listed_sha")
        lock_path = _lock_path()
        inputs = _listed_test_inputs(M4_DEMO_LABEL, M5_WORKSPACE, M5_LISTED_RUN, M5_CHECKPOINT)
        other_checkpoint = _read_run_checkpoint(checkpoint)["sha256"]
        assert other_checkpoint != inputs.checkpoint_sha256
        try:
            other_version = resolve_version(registry, M4_OTHER_GRAPH_LABEL).identity.sha256
        except Exception:  # noqa: BLE001 - T4's version absent: another digest serves
            other_version = "0" * 64
        seen = {}
        for case, change in (("checkpoint", {"checkpoint_sha256": other_checkpoint}),
                             ("cohort version", {"version_sha256": other_version})):
            before, usage_before = _store_runs(reports), len(read_usage_history(version)[0])
            manager = TrainingManager()
            with _job_ended_on_exit(manager, "test"):
                started = manager.start_test(dataclasses.replace(inputs, **change))
                assert started.get("success"), "the runner refused the start itself"
                word = _job_ended(manager, "test", _TEST_RUN_SECONDS)
            outcome = manager.get_test_status()["outcome"] or {}
            log = "\n".join(manager.get_log("test")["lines"])
            if word != "failed":
                _job_said(manager, "test")
            # **A failed run is never a success**, and what it left is read
            # from the disk: no report published, nothing kept.
            assert word == "failed", f"a run started for another {case} ended {word}"
            assert "this run was started for" in log and "Nothing was written." in log, (
                f"the run was not refused for its {case} digest")
            assert (outcome.get("kept"), outcome.get("registered")) == (False, False), (
                f"the failed run reads kept {outcome.get('kept')}")
            assert _store_runs(reports) == before, "the refused run left a report directory"
            assert len(read_usage_history(version)[0]) == usage_before, (
                "the refused run is in the usage history")
            assert lock_probe(lock_path) is None, "the GPU lock stayed held"
            seen[case] = {"status": word, "kept": False, "registered": False,
                          "report_directory": False, "usage_history_entry": False}
        return "a run whose model or cohort is not the chosen one fails before measuring", seen

    def _training_job() -> Tuple[str, Dict[str, Any]]:
        _need("workspace", "fingerprint")
        lock_path = _lock_path()
        inputs = _listed_test_inputs(M4_DEMO_LABEL, M5_WORKSPACE, M5_LISTED_RUN, M5_CHECKPOINT)
        manager = TrainingManager()
        config = {**_DEMO_TRAINING, "data_dir": str(state["workspace"]),
                  "run_dir": str(m5 / "jobs" / "stopped-training"), "device": device}
        with _job_ended_on_exit(manager, "training"):
            started, held = _start_training_job(manager, config, state["fingerprint"],
                                                lock_path)
            outcome = m5 / "cli" / "outcome-while-training.json"
            refused = _exclusion_while_training(manager, inputs, started["pid"],
                                                _cli_argv(inputs, outcome), outcome)
            stopped = _stop_training_job(manager, lock_path)
        return "a training job held the lock; other jobs and the CLI refused; its stop ended " \
               "only it, as stopped", {**held, "while_it_ran": refused, "stop": stopped}

    # Read under the operator's settings, before M5's own replace them.
    deployment_lock = _deployment_gpu_lock()
    with _deployment_settings(settings):
        report.run("W1", "m2", "M5: the server lists workspaces, checkpoints and cohort "
                   "versions by name and digest, and resolves only what it lists", _listed)
        try:
            _start_services()
            report.run("W2", "m2", "M5: D8 on the running service allows M1 on loopback and "
                       "refuses forwarded, foreign-Host, foreign-Origin and cross-site requests "
                       "on every gated API route and three gated dashboard events",
                       _d8_on_the_service)
            report.run("W3", "m2", "M5: no cross-origin browser access to /api or /ui", _cors)
            report.run("W4", "m2", "M5: the route guard refuses file= URLs, never fetched, "
                       "and process_recording and uploads before the body is read", _route_guard)
            report.run("W5", "m2", "M5: without M1 the service refuses every gated API route "
                       "and the gated dashboard events asked; the read-only routes answer",
                       _read_only)
        finally:
            for server in servers.values():
                server.stop()
        report.run("W6", "m2", "M5: D8 refuses a request that did not arrive on loopback",
                   _off_loopback)
        report.run("W7", "m2", "M5: a busy GPU lock refuses the runner and the CLIs naming its "
                   "holder, and a killed holder releases it", _gpu_lock)
        report.run("W8", "m2", "M5: a test run through the job runner from the server's lists "
                   "says kept and registered as the publication contract does; a training stop "
                   "does not reach it", _test_job)
        report.run("W9", "m2", "M5: a report is registered only from the report store",
                   _registered)
        report.run("W10", "m2", "M5: a test run whose model or cohort is not the chosen one "
                   "fails before measuring, nothing kept", _start_checked_again)
        report.run("W11", "m2", "M5: a training job holds the GPU; a test start, a test cancel "
                   "and the CLI are refused meanwhile; its stop ends it as stopped", _training_job)


def _m5_real(report: Report, runs: Path, real: Path, workspaces_root: Path, device: str,
             state: Dict[str, Any]) -> None:
    """M5's acceptance on the real workspace (RW1-RW3), after M4's RT1-RT3 and
    before R8, which therefore covers them: the workspace as the server lists
    it, a test run through the job runner on the device, and a short training
    job started and stopped through it (no long training). Everything the runs
    write is under --work-dir: the GPU lock is the one under a workspaces root
    of the probe's own there (an empty directory, never listed), the registry is
    RT1's (named as `SHEPHERD_REGISTRY` while they run: a test run measures
    only in the deployment's registry, BACKLOG 37), the runs' directories and
    ledgers are beside R3's checkpoint or under <work>/real/m5. The
    deployment's own root, lock and registry are never used; the root is only
    read, to list."""
    from src.api.services import server_lists
    from src.api.services.training_manager import (
        TestRunInputs,
        TrainingManager,
        build_test_argv,
        deployment_device,
    )
    from src.evaluation.cohort_registry import REPORT_FILE
    from src.evaluation.report_store import read_usage_history
    from src.evaluation.sidecar import read_ledger
    from src.kg.artifacts import MANIFEST_FILENAME
    from src.kg.storage.file_storage import _read_json
    from src.utils.gpu_lock import LOCK_FILE_NAME
    from src.utils.gpu_lock import probe as lock_probe

    m5 = runs / "m5"
    lock_root = m5 / "gpu-lock-root"
    lock_path = lock_root / LOCK_FILE_NAME
    registry = runs / "m4" / "registry"
    reports = registry / "versions" / M4_REAL_LABEL / "reports"
    settings = {"SHEPHERD_WORKSPACES_ROOT": str(lock_root), "SHEPHERD_DEVICE": device,
                "SHEPHERD_REGISTRY": str(registry)}

    def _listed() -> Tuple[str, Dict[str, Any]]:
        from dataclasses import asdict

        assert "read" in state, "R1 did not pass"
        environ = {"SHEPHERD_WORKSPACES_ROOT": str(workspaces_root)}
        names = [name for name, path in server_lists.resolve_workspaces(environ=environ).items()
                 if path == real.resolve()]
        assert names, "the real workspace is not listed under --workspaces-root"
        listing = server_lists.list_workspaces(environ=environ)
        item = next(entry for entry in listing.items if entry.name == names[0])
        assert item.dataset_fingerprint == state["read"][MANIFEST_FILENAME], (
            "the listed dataset fingerprint is not the digest R1 read of the manifest")
        assert (item.dataset_record.get("binding"), item.dataset_record.get("origin")) == (
            "verified", "files"), "the listed dataset record is not verified, from files"
        assert item.trainable, "the real workspace is listed as not trainable"
        assert str(workspaces_root.resolve()) not in json.dumps(_identity_fields(asdict(item))), (
            "the listed item carries a server path")
        return "the real workspace listed by its fingerprint, record and build", {
            "dataset_fingerprint": item.dataset_fingerprint, "trainable": item.trainable,
            "dataset_record": {"binding": item.dataset_record.get("binding"),
                               "origin": item.dataset_record.get("origin")},
            "build": item.build.get("state"),
            "declared_releases": {entry.get("role"): entry.get("declared_version")
                                  for entry in item.declared_releases},
            "samples": {"train": item.train_samples, "val": item.val_samples},
            "workspaces_listed": len(listing.items), "item_carries_no_server_path": True,
            "deployment_gpu_lock": deployment_lock,
        }

    # Read under the operator's settings, before M5's own replace them.
    deployment_lock = _deployment_gpu_lock()
    report.run("RW1", "real", "M5: the server lists the real workspace by its dataset "
               "fingerprint", _listed)

    def _test_job() -> Tuple[str, Dict[str, Any]]:
        assert "record" in state.get("m4", {}), "the cohort version (RT1, RT2) is not available"
        assert "parent" in state, "R3 did not pass"
        version, parent = state["m4"]["version"], state["parent"]
        sha256 = _read_run_checkpoint(parent)["sha256"]
        before, usage_before = _store_runs(reports), len(read_usage_history(version)[0])
        manager = TrainingManager()
        with _job_ended_on_exit(manager, "test"):
            started = manager.start_test(TestRunInputs(
                registry=registry, version_label=M4_REAL_LABEL,
                version_sha256=version.identity.sha256, workspace=real, checkpoint=parent,
                checkpoint_sha256=sha256, record=True))
            assert started.get("success"), "the runner refused the test run"
            not_reached = _training_stop_does_not_reach_a_test_run(manager)
            word = _job_ended(manager, "test", _TEST_RUN_SECONDS)
        status = manager.get_test_status()
        if word != "completed" or status["exit_code"] != 0:
            _job_said(manager, "test")
        assert word == "completed" and status["exit_code"] == 0, (
            f"the test run ended {word}, exit {status['exit_code']}")
        created = sorted(set(_store_runs(reports)) - set(before))
        assert len(created) == 1, f"the run left {len(created)} report directories"
        outcome = status["outcome"] or {}
        published = (outcome.get("run_outcome") or {}).get("report_sha256")
        assert published == _read_json(reports / created[0] / REPORT_FILE)[1].sha256, (
            "the outcome file's digest is not the kept report's bytes")
        assert (outcome.get("kept"), outcome.get("registered")) == (True, True), (
            f"the outcome says kept {outcome.get('kept')}, registered {outcome.get('registered')}")
        ledger, _ = read_ledger(parent.resolve().parent / "evaluations.json")
        assert [record.get("source_artifact_digest") for record in ledger["records"]] == [
            published], "the ledger beside R3's checkpoint does not hold the run's one record"
        new = read_usage_history(version)[0][usage_before:]
        assert [(e.get("kind"), e.get("run_id")) for e in new] == [
            ("test_run", created[0]), ("registered", created[0])], (
            f"the usage history gained {[e.get('kind') for e in new]}")
        assert lock_probe(lock_path) is None, "the GPU lock stayed held after the run ended"
        return "a test run through the runner on the real workspace, kept and registered", {
            "checkpoint": "R3's paused checkpoint, under the work directory",
            "status": word, "exit_code": status["exit_code"], "report_sha256": published,
            "kept": True, "registered": True, "usage_history": ["test_run", "registered"],
            "device": new[0].get("device"), "n_ranked": new[0].get("n_ranked"),
            "gpu_lock": "taken by the run itself, released at its end",
            "training_stop_while_it_ran": not_reached,
        }

    def _training_job() -> Tuple[str, Dict[str, Any]]:
        from scripts.train_model import TrainConfig

        assert "settings" in state and "read" in state, "R1 or R2 did not pass"
        assert "m4" in state, "the cohort version (RT1) is not available"
        settings_now = dict(state["settings"])
        workers = int(settings_now.get("num_workers", TrainConfig.num_workers) or 0)
        room = _socket_room() if workers else None
        assert room is None or room["socket_fits"], (
            f"a DataLoader worker's socket could not be made under the temporary directory "
            f"(its path is {room['path_bytes']} bytes; {room['refusal'] or room['checked']}), "
            f"so this training (num_workers {workers}) would fail with 'AF_UNIX path too long'. "
            "Run the probe with a short TMPDIR, such as TMPDIR=/tmp")
        version, parent = state["m4"]["version"], state.get("parent", state.get("child"))
        inputs = TestRunInputs(
            registry=registry, version_label=M4_REAL_LABEL,
            version_sha256=version.identity.sha256, workspace=real, checkpoint=parent,
            checkpoint_sha256=_read_run_checkpoint(parent)["sha256"])
        manager = TrainingManager()
        # A new run held by configuration whatever the settings file holds
        # (`_start_training_job` applies `_held_config`): the real settings'
        # epochs and patience would let it end by itself during the check, and
        # a parent named there would make it a continuation (contract M3a-3).
        config = {**settings_now, "data_dir": str(real), "run_dir": str(m5 / "stopped-training"),
                  "device": device}
        with _job_ended_on_exit(manager, "training"):
            started, held = _start_training_job(manager, config,
                                                state["read"][MANIFEST_FILENAME], lock_path)
            outcome = m5 / "outcome-while-training.json"
            refused = _exclusion_while_training(
                manager, inputs, started["pid"],
                build_test_argv(inputs, deployment_device(), outcome)[2:], outcome)
            stopped = _stop_training_job(manager, lock_path)
        return "a training job on the real workspace held the GPU and was stopped", {
            **held, "num_workers": workers, "while_it_ran": refused, "stop": stopped,
        }

    lock_root.mkdir(parents=True, exist_ok=True)
    with _deployment_settings(settings):
        report.run("RW2", "real", "M5: a test run through the job runner on the real workspace "
                   "is kept and registered", _test_job)
        report.run("RW3", "real", "M5: a training job on the real workspace holds the GPU "
                   "lock and is stopped as stopped, other jobs refused", _training_job)


def _under_the_probes_own_gpu_lock(phase: Callable[..., None]) -> Callable[..., None]:
    """`phase(report, work, ...)` run with the lock under `<work>/probe-gpu-lock/`
    as the GPU lock in effect (`_probes_own_gpu_lock`), the operator's setting
    put back after."""
    @functools.wraps(phase)
    def run(report: Report, work: Path, *args: Any, **kwargs: Any) -> None:
        with _probes_own_gpu_lock(work):
            phase(report, work, *args, **kwargs)

    return run


@_under_the_probes_own_gpu_lock
def phase_m2_demo(
    report: Report, work: Path, sound: Path, checkpoint: Path, device: str
) -> None:
    """Contract M2's acceptance on the demo workspace: the successes first, then
    every refusal, each from the production entry point that makes it; then
    contract M4's flow on the same workspace (T1-T6), then M5's (W1-W11). The
    formal entry points take the probe's own GPU lock (`_probes_own_gpu_lock`)."""
    print("\nM. Contract M2 on the demo workspace, then M4's flow")
    import src.api.main as api
    from src.inference.pipeline import PipelineBuildError
    from src.kg.artifacts import GRAPH_ARTIFACTS
    from src.utils.fingerprint import TrainingRecordMismatch, file_sha256

    m2 = work / "m2"
    m2.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {}

    def _other_workspace() -> Tuple[str, Dict[str, Any]]:
        from src.kg.graph import DEFAULT_FEATURE_SEED
        from src.kg.storage.file_storage import read_graph_artifacts
        from src.kg.workspace import write_workspace
        from src.utils.fingerprint import compute_fingerprint

        other = m2 / "same_shaped_workspace"
        write_workspace(_same_shaped_kg(), other, feature_dim=FEATURE_DIM,
                        feature_seed=DEFAULT_FEATURE_SEED + 1, samples=_budget())
        verify_workspace(other)          # valid on its own: graph and cohorts
        shapes = [compute_fingerprint(read_graph_artifacts(w, map_location="cpu").graph_data)
                  for w in (sound, other)]
        assert shapes[0] == shapes[1], "the two workspaces differ in shape"
        differs = sorted(role for role, name in GRAPH_ARTIFACTS.items()
                         if (other / name).read_bytes() != (sound / name).read_bytes())
        assert {"kg", "node_features"} <= set(differs), (
            "the other workspace does not hold another graph and other features"
        )
        state["other"] = other
        return "a valid, same-shaped workspace over another graph", {
            "differing_roles": differs, "same_structural_fingerprint": True,
        }

    report.run("M1", "m2", "PRECONDITION: a valid same-shaped workspace over another graph",
               _other_workspace)

    def _resumed() -> Tuple[str, Dict[str, Any]]:
        # **A real pause and resume** (contract M3a-3): plan two epochs, pause
        # after one, resume the remaining one.
        paused = _last(_train_run(sound, m2 / "paused", device, 2,
                                  settings=_DEMO_TRAINING, pause_after=1))
        before = _read_run_checkpoint(paused)["progress"]
        assert before.get("end") == "paused" and before.get("completed_epochs") == 1, (
            f"the run did not pause after epoch 1: {before.get('end')}")
        observed: List[Dict[str, Any]] = []
        child = _last(_train_run(sound, m2 / "resume_ok", device, 2,
                                 settings=_DEMO_TRAINING, resume_from=paused, observe=observed))
        facts = _assert_resumed(paused, child, observed)
        facts.update(_assert_only_the_remaining_epochs(
            _read_run_checkpoint(paused), _read_run_checkpoint(child)))
        state["child"] = child
        facts.update(paused_end=before["end"], parent_attempts=before["optimizer_attempts"],
                     parent_updates=before["optimizer_updates"])
        return "paused after one of two epochs, resumed only the remaining one", facts

    report.run("M2", "m2", "CONTROL: a paused run resumes its remaining epoch and records it",
               _resumed)

    def _measured() -> Tuple[str, Dict[str, Any]]:
        # **The resumed model, never its parent in its place**: the flow this
        # part claims is build, train, resume, then measure and serve that model.
        assert "child" in state, "the resumed checkpoint (M2) is not available"
        output = m2 / "measured" / "measurement.json"
        assert _measure(state["child"], sound, output, device) == 0, "the measurement failed"
        return "the resumed checkpoint was measured over its own workspace", _measurement_facts(
            output, _read_run_checkpoint(state["child"])["sha256"], *_kg_and_record(sound))

    report.run("M3", "m2", "CONTROL: measurement accepts the resumed checkpoint over its graph",
               _measured)

    def _served_with_sp() -> Tuple[str, Dict[str, Any]]:
        import subprocess

        assert "child" in state, "the resumed checkpoint (M2) is not available"
        served = _workspace_copy(sound, m2 / "sp_workspace")
        produced = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "compute_shortest_paths.py"),
             "--kg-path", str(served / "kg.json"), "--output-dir", str(served),
             "--max-hops", "5", "--workers", "1"],
            cwd=project_root, capture_output=True, text=True, timeout=1800,
        )
        assert produced.returncode == 0, "the shortest-path producer failed"
        _, _, read, record = verify_workspace(served)
        facts = _serve_and_diagnose(served / "kg.json", state["child"], served, device,
                                    record, read)
        return "build, train, resume and serve, with the table verified", facts

    report.run("M4", "m2",
               "CONTROL: the resumed model serves with a verified shortest-path table",
               _served_with_sp)

    # ---- Resume refusals: nothing restored, no run directory, nothing changed
    def _refused_resume(case: str, data_dir: Path, parent: Path, expected: type,
                        fragment: str) -> Dict[str, Any]:
        run = m2 / "refused" / case
        workspace_before = snapshot(data_dir, include_dirs=True)
        parent_before = snapshot(parent.parent, include_dirs=True) if parent.exists() else None
        refuses(lambda: _train_run(data_dir, run, device, 1, settings=_DEMO_TRAINING,
                                   resume_from=parent), expected, fragment)
        assert not run.exists(), "the refused run created its run directory"
        assert_untouched(workspace_before, data_dir, "a refused resume", include_dirs=True)
        if parent_before is not None:
            assert_untouched(parent_before, parent.parent, "a refused resume",
                             include_dirs=True)
        return {"run_directory_created": False, "workspace_unchanged": True,
                "parent_unchanged": parent_before is not None}

    def _pre_contract() -> Path:
        import torch

        from src.utils.checkpoint_io import read_checkpoint

        if "pre_contract" not in state:
            stripped = m2 / "parents" / "pre_contract.pt"
            stripped.parent.mkdir(parents=True, exist_ok=True)
            contents = read_checkpoint(checkpoint, map_location="cpu",
                                       weights_only=True).checkpoint
            del contents["training_input_digests"]
            torch.save(contents, stripped)
            state["pre_contract"] = stripped
        return state["pre_contract"]

    def _refuse_pre_contract() -> Tuple[str, Dict[str, Any]]:
        return "refused before anything existed", _refused_resume(
            "pre_contract", sound, _pre_contract(), TrainingRecordMismatch,
            "records no training_input_digests")

    def _refuse_other_graph() -> Tuple[str, Dict[str, Any]]:
        assert "other" in state, "the same-shaped workspace (M1) is not available"
        return "refused before anything existed", _refused_resume(
            "other_graph", state["other"], checkpoint, TrainingRecordMismatch, "kg differs")

    def _refuse_other_samples() -> Tuple[str, Dict[str, Any]]:
        from src.kg.workspace import write_workspace

        other = m2 / "other_samples_workspace"
        write_workspace(_demo_kg(), other, feature_dim=FEATURE_DIM,
                        samples=_budget(seed=SEED + 1))
        for name in GRAPH_ARTIFACTS.values():
            assert (other / name).read_bytes() == (sound / name).read_bytes(), (
                f"the other-samples workspace holds another {name}"
            )
        assert (other / "train_samples.json").read_bytes() != (
            sound / "train_samples.json").read_bytes(), "the training samples are the same"
        return "refused before anything existed", _refused_resume(
            "other_samples", other, checkpoint, TrainingRecordMismatch,
            "train_samples differs")

    def _refuse_validated_parent() -> Tuple[str, Dict[str, Any]]:
        from src.kg.artifacts import read_split_manifest
        from src.kg.disease_allocation import train_only_allocation
        from src.kg.sample_generator import (
            build_eligible_disease_profiles,
            generate_training_samples,
        )
        from src.kg.storage.file_storage import read_samples

        assert _read_run_checkpoint(checkpoint)["record"].get("val_samples"), (
            "the parent did not run validation"
        )
        # The same graph files, cohorts cut again with no validation partition,
        # through the generator's own train-only path and a manifest it writes.
        other = m2 / "no_validation_workspace"
        other.mkdir(parents=True)
        for name in (*GRAPH_ARTIFACTS.values(), "kg.provenance.json"):
            shutil.copy2(sound / name, other / name)
        manifest = read_split_manifest(sound).manifest
        kg = _demo_kg()
        generate_training_samples(
            kg=kg,
            allocation=train_only_allocation(
                build_eligible_disease_profiles(kg, min_phenotypes=2), seed=SEED),
            num_train=NUM_TRAIN, num_val=0, min_phenotypes=2, output_dir=other,
            graph_digests={role: manifest["artifacts"][role]
                           for role in (*GRAPH_ARTIFACTS, "provenance")},
            graph_export=manifest["graph_export"],
        )
        assert read_samples(other, "val").samples == [], (
            "the workspace still has validation samples"
        )
        return "refused before anything existed", _refused_resume(
            "validated_parent", other, checkpoint, TrainingRecordMismatch,
            "val_samples is recorded and this run has none")

    def _refuse_missing() -> Tuple[str, Dict[str, Any]]:
        return "refused, and no run trained from scratch", _refused_resume(
            "missing", sound, m2 / "parents" / "absent.pt", FileNotFoundError,
            "does not train from scratch")

    for case, claim, probe in (
        ("pre_contract", "a parent from before the contract", _refuse_pre_contract),
        ("other_graph", "a parent from a same-shaped other graph", _refuse_other_graph),
        ("other_samples", "a parent that read other sample files", _refuse_other_samples),
        ("validated_parent", "a parent that ran validation, resumed by a run without it",
         _refuse_validated_parent),
        ("missing", "a requested parent that does not exist", _refuse_missing),
    ):
        report.run(f"M5.{case}", "m2",
                   f"resume refuses {claim}, before anything is restored or written", probe)

    # ---- The checkpoint against a same-shaped other workspace
    def _serving_refuses() -> Tuple[str, Dict[str, Any]]:
        assert "other" in state, "the same-shaped workspace (M1) is not available"
        other = state["other"]
        before, served = snapshot(other, include_dirs=True), api.app_state.pipeline
        refuses(lambda: api.build_pipeline(
            kg_path=str(other / "kg.json"), checkpoint_path=str(checkpoint),
            data_dir=str(other), device=device,
        ), PipelineBuildError, ("does not match this workspace's graph", "kg differs"))
        assert api.app_state.pipeline is served, "the served pipeline changed"
        assert_untouched(before, other, "a refused build", include_dirs=True)
        return "refused by its record, not its shape", {}

    report.run("M6", "m2", "serving refuses a checkpoint over a same-shaped other workspace",
               _serving_refuses)

    def _measurement_refuses() -> Tuple[str, Dict[str, Any]]:
        assert "other" in state, "the same-shaped workspace (M1) is not available"
        other = state["other"]
        output = m2 / "measure_refused" / "measurement.json"
        before = snapshot(other, include_dirs=True)
        refuses(lambda: _measure(checkpoint, other, output, device),
                TrainingRecordMismatch, "kg differs")
        assert not output.parent.exists(), "the refused measurement created its output directory"
        assert_untouched(before, other, "a refused measurement", include_dirs=True)
        return "refused before any model was built or output written", {}

    report.run("M7", "m2",
               "measurement refuses a checkpoint over a same-shaped other workspace",
               _measurement_refuses)

    def _inventory() -> Tuple[str, Dict[str, Any]]:
        import scripts.audit_checkpoint_family as audit

        assert "other" in state, "the same-shaped workspace (M1) is not available"
        root = m2 / "inventory_root"
        placed = {
            "a/checkpoints/m2/fresh.pt": checkpoint,
            "a/checkpoints/m2/pre_contract.pt": _pre_contract(),
            "b/checkpoints/m2/fresh.pt": checkpoint,
        }
        _workspace_copy(sound, root / "a")
        _workspace_copy(state["other"], root / "b")
        for name, source in placed.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, root / name)
        output = m2 / "inventory.json"
        assert audit.main(["--inventory", str(root), "--output", str(output)]) == 0
        inventory = json.loads(output.read_text())
        listing = inventory["workspaces"]

        def verdicts(workspace: str, name: str) -> Dict[str, Dict[str, Any]]:
            return listing[workspace]["checkpoints"][name]["m2_contract"]

        fresh = verdicts("a", "a/checkpoints/m2/fresh.pt")
        assert fresh["serve_and_measure"]["verdict"] == fresh["resume"]["verdict"] == "accepted"
        old = verdicts("a", "a/checkpoints/m2/pre_contract.pt")["serve_and_measure"]
        assert old["verdict"] == "refused" and "records no training_input_digests" in old["reason"]
        crossed = verdicts("b", "b/checkpoints/m2/fresh.pt")["serve_and_measure"]
        assert crossed["verdict"] == "refused" and "kg differs" in crossed["reason"]
        return "the inventory lists what M2 accepts and refuses", {
            "report_sha256": file_sha256(output),
            "counts": inventory["counts"],
        }

    report.run("M8", "m2", "the inventory lists the demo checkpoints' M2 verdicts", _inventory)

    def _extended() -> Tuple[str, Dict[str, Any]]:
        assert "child" in state, "the resumed checkpoint (M2) is not available"
        before = _read_run_checkpoint(state["child"])
        observed: List[Dict[str, Any]] = []
        extended = _last(_train_run(sound, m2 / "extended", device, 2, settings=_DEMO_TRAINING,
                                    resume_from=state["child"], extend_epochs=1,
                                    observe=observed))
        after = _read_run_checkpoint(extended)
        scaler = _scaler_restore_facts(before, after, observed)
        stage = after["progress"].get("stage") or {}
        assert after["record"].get("resume_checkpoint") == before["sha256"]
        assert after["progress"].get("completed_epochs") == 3
        assert stage.get("kind") == "extension" and stage.get("start_epoch") == 2
        assert stage.get("schedule", {}).get("lrs") == before["rates"] == after["rates"], (
            "the extension did not hold the finished run's final rate")
        assert after["progress"]["scheduler_steps"] == before["progress"]["scheduler_steps"], (
            "a scheduler stepped during a fixed-rate extension")
        return "extended by one epoch at the finished run's final rate", {
            "fixed_rates": after["rates"], "completed_epochs": 3,
            "optimizer_updates": after["progress"]["optimizer_updates"], **scaler,
        }

    report.run("M9", "m2", "CONTROL: a finished run extends at a fixed rate (M3a-4)", _extended)

    def _finished_refused() -> Tuple[str, Dict[str, Any]]:
        from src.utils.run_plan import RunPlanRefused

        assert "child" in state, "the resumed checkpoint (M2) is not available"
        return "a finished run is refused for resume, with the reason", _refused_resume(
            "finished", sound, state["child"], RunPlanRefused, "nothing left to resume")

    report.run("M10", "m2", "resuming a finished run is refused before anything exists",
               _finished_refused)

    # ---- Contract M3b: each entry point refuses a workspace without its record
    def _without_record() -> Path:
        from src.kg.provenance import PROVENANCE_FILENAME

        if "without_record" not in state:
            copy = _workspace_copy(sound, m2 / "no_record_workspace")
            (copy / PROVENANCE_FILENAME).unlink()
            state["without_record"] = copy
        return state["without_record"]

    def _training_refuses_record() -> Tuple[str, Dict[str, Any]]:
        from src.kg.artifacts import DatasetRecordRefused

        copy, run = _without_record(), m2 / "refused" / "no_record"
        before = snapshot(copy, include_dirs=True)
        refuses(lambda: _train_run(copy, run, device, 1, settings=_DEMO_TRAINING),
                DatasetRecordRefused, "is not here")
        assert not run.exists(), "the refused run created its run directory"
        assert_untouched(before, copy, "a refused training run", include_dirs=True)
        return "refused before the run directory existed", {}

    def _serving_refuses_record() -> Tuple[str, Dict[str, Any]]:
        from src.kg.artifacts import DatasetRecordRefused

        assert "child" in state, "the resumed checkpoint (M2) is not available"
        copy = _without_record()
        before, served = snapshot(copy, include_dirs=True), api.app_state.pipeline
        refuses(lambda: api.build_pipeline(
            kg_path=str(copy / "kg.json"), checkpoint_path=str(state["child"]),
            data_dir=str(copy), device=device,
        ), DatasetRecordRefused, "is not here")
        assert api.app_state.pipeline is served, "the served pipeline changed"
        assert_untouched(before, copy, "a refused build", include_dirs=True)
        return "refused before the checkpoint was read", {}

    def _measurement_refuses_record() -> Tuple[str, Dict[str, Any]]:
        from src.kg.artifacts import DatasetRecordRefused

        assert "child" in state, "the resumed checkpoint (M2) is not available"
        copy = _without_record()
        output = m2 / "measure_no_record" / "measurement.json"
        before = snapshot(copy, include_dirs=True)
        refuses(lambda: _measure(state["child"], copy, output, device),
                DatasetRecordRefused, "is not here")
        assert not output.parent.exists(), "the refused measurement created its output directory"
        assert_untouched(before, copy, "a refused measurement", include_dirs=True)
        return "refused before any model was built or output written", {}

    for use, probe in (("training", _training_refuses_record),
                       ("serving", _serving_refuses_record),
                       ("measurement", _measurement_refuses_record)):
        report.run(f"M11.{use}", "m2",
                   f"{use} refuses a workspace whose dataset record is missing (M3b)", probe)

    # ---- Contract M4's flow: import, measure, keep, register, read back
    # Everything under <work>/m4: the source, the registry, a copy of D1's
    # checkpoint (the ledger is written beside the measured file), the
    # damaged copy and the audit report. The workspace is only read.
    m4_dir = m2.parent / "m4"
    registry, measured = m4_dir / "registry", m4_dir / "checkpoints" / "measured.pt"
    m4: Dict[str, Any] = {}

    def _imported() -> Tuple[str, Dict[str, Any]]:
        registry.mkdir(parents=True)
        source = _write_cohort_source(m4_dir / "cases.json", _m4_demo_cases(sound))
        version, preview = _import_version(registry, sound, source, M4_DEMO_LABEL)
        bound, _, _, record = verify_workspace(sound)
        assert version.manifest["graph"] == bound, (
            "the version's graph digests are not the workspace's")
        assert version.manifest["dataset_record"]["sha256"] == record.summary()["sha256"], (
            "the version is bound to another dataset record than the workspace's")
        counts = version.manifest["counts"]
        assert counts == M4_DEMO_COUNTS, f"the import counted {counts}, not {M4_DEMO_COUNTS}"
        assert version.manifest["samples"]["n"] == counts["ranked_eligible"]
        m4.update(version=version, source=source)
        return "previewed with nothing written, committed by the printed id, bound to the graph", {
            "preview_wrote_nothing": True, "preview_id": preview,
            "manifest_sha256": version.identity.sha256,
            "samples_sha256": version.manifest["samples"]["sha256"],
            "graph_is_the_workspaces": True, "counts": counts,
            "partial_phenotypes": version.manifest["mapping"]["partial_phenotypes"],
        }

    def _measured_and_recorded() -> Tuple[str, Dict[str, Any]]:
        from src.evaluation.sidecar import REGISTERED_RELATIONS

        assert "version" in m4, "the cohort version (T1) is not available"
        version = m4["version"]
        measured.parent.mkdir(parents=True)
        shutil.copyfile(checkpoint, measured)          # the bytes D1 trained, copied
        sha256 = _read_run_checkpoint(measured)["sha256"]
        code = _measure_cohort(measured, sound, registry, M4_DEMO_LABEL, device, "--record")
        assert code == 0, f"the measurement with --record exited {code}"
        path, kept, identity, rows = _kept_report(registry, M4_DEMO_LABEL)
        manifest = kept["manifest"]
        assert manifest["artifact_digests"]["checkpoint"] == sha256, (
            "the report does not name the checkpoint it was given")
        assert manifest["cohort_version"]["manifest_sha256"] == version.identity.sha256, (
            "the report names another cohort version")
        assert manifest["relations"] == REGISTERED_RELATIONS, (
            f"the report's relation states are {manifest['relations']}")
        assert kept["n_ranked"] == M4_DEMO_COUNTS["ranked_eligible"]
        r10 = manifest["r10"]
        # D1 trained on this workspace, so the inputs it records are its files.
        _evidence_found_in(r10, sound, ("training", "validation"))
        record, ledger_sha256 = _the_record(measured.parent, identity, sha256, version)
        assert record["registration"]["sequence"] == 1
        m4.update(report=path, report_sha256=identity.sha256, record=record,
                  ledger_sha256=ledger_sha256, checkpoint_sha256=sha256, r10=r10)
        return "measured, the report kept, R10 recorded, one record from the report's bytes", {
            "exit_code": code, "checkpoint_sha256": sha256,
            "report_sha256": identity.sha256, "ranks_sha256": kept["ranks"]["sha256"],
            "ranks_rows": rows, "n_ranked": kept["n_ranked"],
            "cohort_manifest_sha256": version.identity.sha256,
            "relations": manifest["relations"], "r10": _r10_facts(r10),
            "r10_evidence_in_the_workspace": True,
            "ledger_sha256": ledger_sha256,
            "record": {"sequence": record["registration"]["sequence"],
                       "source_artifact_digest": record["source_artifact_digest"],
                       "corroborating_reports": len(record["corroborating_reports"])},
            "cuda_executed": manifest.get("cuda_executed"),
        }

    def _registered_again() -> Tuple[str, Dict[str, Any]]:
        import scripts.record_evaluation as record_evaluation

        assert "report" in m4, "the measurement (T2) is not available"
        before = snapshot(measured.parent, include_dirs=True)
        # The later registration in the form measure_scorer prints for it:
        # `--report` and `--checkpoint FILE`, the file the run measured, with
        # the probe's registry as the deployment's (the flow's).
        with _logged(record_evaluation.logger) as said:
            code = record_evaluation.main(["--report", str(m4["report"]),
                                           "--checkpoint", str(measured)])
        assert code == 0, f"registering the same report again exited {code}"
        assert any("already holds this report" in line for line in said), (
            "registering the same report again was not already_registered")
        assert_untouched(before, measured.parent, "registering the same report again",
                         include_dirs=True)
        data = m4["report"].read_bytes()
        damaged = m4_dir / "damaged" / "report.json"
        damaged.parent.mkdir(parents=True)
        damaged.write_bytes(data[: len(data) // 2])
        # Since M5-1 a report is registered only from the registry's report
        # store (D5, 2026-10-19): a copy outside it is refused for where it is,
        # before it is read, saying to register the store's original.
        refuses(lambda: record_evaluation.main(["--report", str(damaged),
                                                "--checkpoint", str(measured)]),
                SystemExit, ("is not a report the registry's report store holds",
                             "Nothing was written"))
        assert_untouched(before, measured.parent, "a refused registration", include_dirs=True)
        # **A copy of the whole registry is not the deployment's** (BACKLOG 37):
        # its report, the same bytes in the same structure, is refused before
        # anything in the copy is read, and the copy's usage history is untouched.
        whole = m4_dir / "registry-copy"
        shutil.copytree(registry, whole, symlinks=True)
        copy_before = snapshot(whole, include_dirs=True)
        refuses(lambda: record_evaluation.main([
            "--report", str(whole / m4["report"].relative_to(registry)),
            "--checkpoint", str(measured)]),
            SystemExit, ("is not registered: the registry", "is not the deployment's",
                         "Nothing was written"))
        assert_untouched(before, measured.parent, "a registration from a copied registry",
                         include_dirs=True)
        assert_untouched(copy_before, whole, "a registration from a copied registry",
                         include_dirs=True)
        return "already registered, a damaged copy outside the store and a copied registry " \
               "refused; the ledger unchanged", {
                   "same_report_again": "already_registered", "damaged_copy": "refused",
                   "damaged_copy_refused_as": "not in the registry's report store",
                   "damaged_copy_bytes": len(data) // 2,
                   "registry_copy": "refused: not the deployment's registry",
                   "ledger_unchanged": True,
               }

    def _other_graph_refused() -> Tuple[str, Dict[str, Any]]:
        from src.evaluation.cohort_registry import CohortBindingMismatch

        assert "record" in m4, "the measurement (T2) is not available"
        assert "other" in state, "the same-shaped workspace (M1) is not available"
        # The same source, mapped against M1's export of a same-shaped graph.
        _import_version(registry, state["other"], m4["source"], M4_OTHER_GRAPH_LABEL)
        before = snapshot(registry, include_dirs=True)
        ledger_before = snapshot(measured.parent, include_dirs=True)
        reads: List[str] = []
        with _observing_checkpoint_reads(reads):
            refusal = refuses(lambda: _measure_cohort(measured, sound, registry,
                                                      M4_OTHER_GRAPH_LABEL, device, "--record"),
                              CohortBindingMismatch, "mapped against another graph")
        assert not reads, "the checkpoint was read before the refusal"
        assert_untouched(before, registry, "a refused measurement", include_dirs=True)
        assert_untouched(ledger_before, measured.parent, "a refused measurement",
                         include_dirs=True)
        assert "kg" in refusal.roles, f"the refusal names {refusal.roles}, not the graph"
        return "refused by the binding before the checkpoint was read; nothing written", {
            "differing_roles": sorted(refusal.roles), "checkpoint_reads": 0,
            "report_directory_created": False, "ledger_unchanged": True,
        }

    def _read_only_view() -> Tuple[str, Dict[str, Any]]:
        assert "record" in m4, "the measurement (T2) is not available"
        assert "child" in state, "the resumed checkpoint (M2) is not available"
        served, answer = _serve_and_view(sound, measured, m4["checkpoint_sha256"], device)
        shown = _the_view_shows(answer, m4["record"], m4["ledger_sha256"],
                                ("training", "validation"))
        # **Another model, beside the same ledger**: found by digest, so the
        # measured checkpoint's record is not shown for it.
        other = measured.with_name("other.pt")
        shutil.copyfile(state["child"], other)
        # Its digest through the run reader, as T2's copy's (`_read_run_checkpoint`).
        served_other, beside = _serve_and_view(sound, other, _read_run_checkpoint(other)["sha256"],
                                               device)
        assert served_other != served, "the other checkpoint is the measured one"
        assert beside.get("state") == "no_records", (
            f"the view's state for another checkpoint is {beside.get('state')}")
        assert beside.get("records") == [] and beside.get("records_not_shown") == 0, (
            "the view shows a record for a checkpoint that has none")
        assert (beside.get("ledger") or {}).get("sha256") == m4["ledger_sha256"], (
            "the view read another ledger for the other checkpoint")
        return "the served checkpoint's record shown; none for another beside the same ledger", {
            "served": "reloaded through the reload route's handler with the checkpoint named "
                      "explicitly and bound to its digest (expected_checkpoint_sha256); read "
                      "through GET /api/v1/pipeline/status and GET "
                      "/api/v1/evaluations/loaded-model",
            "measured": shown,
            "other": {"state": beside["state"], "checkpoint_sha256": served_other,
                      "records": 0, "same_ledger": True},
        }

    def _overlap_audit() -> Tuple[str, Dict[str, Any]]:
        import scripts.audit_split_overlap as audit
        from src.kg.storage.file_storage import _read_json

        assert "version" in m4, "the cohort version (T1) is not available"
        version, output = m4["version"], m4_dir / "overlap_audit.json"
        assert audit.main(["--data-dir", str(sound), "--registry", str(registry),
                           "--cohort", M4_DEMO_LABEL, "--output", str(output)]) == 0
        audited, identity = _read_json(output)
        assert audited["schema_version"] == audit.REPORT_SCHEMA_VERSION
        cohort_version = audited.get("cohort_version") or {}
        expected = {"label": M4_DEMO_LABEL, "role": M4_ROLE,
                    "manifest_sha256": version.identity.sha256,
                    "samples_sha256": version.manifest["samples"]["sha256"]}
        recorded = {key: cohort_version.get(key) for key in expected}
        assert recorded == expected, f"the audit records the cohort version as {recorded}"
        return "the registered version audited against train, and recorded", {
            "report_sha256": identity.sha256, "cohort_version": recorded,
            "counts": audited["counts"], "overlap": audited["overlap"]["as_written"],
        }

    # **The probe's registry is the deployment's for the whole flow** (root-cause
    # round, INT-1), set once here as phases M and R set their GPU lock, so every
    # step that reads SHEPHERD_REGISTRY inherits it: measure_scorer --cohort,
    # record_evaluation and the loaded-model route's D5 state alike.
    with _registry_setting(registry):
        report.run("T1", "m2",
                   "M4: an import preview writes nothing and its id commits a version bound "
                   "to the graph", _imported)
        report.run("T2", "m2",
                   "M4: a registered version is measured with --record: report kept, R10 "
                   "recorded, one record", _measured_and_recorded)
        report.run("T3", "m2",
                   "M4: the same report again is already registered, a damaged copy outside "
                   "the report store and a copy of the whole registry refused, the ledger "
                   "unchanged", _registered_again)
        report.run("T4", "m2",
                   "M4: a version mapped against another graph is refused before the "
                   "checkpoint is read", _other_graph_refused)
        report.run("T5", "m2",
                   "M4: the read-only view shows the served checkpoint's record, and none for "
                   "another", _read_only_view)
        report.run("T6", "m2",
                   "M4: the overlap audit runs on the registered version and records it",
                   _overlap_audit)

    # ---- M5's acceptance on the same workspace: M2's resumed checkpoint, T1's
    # version and registry, the probe's own deployment under <work>/m5.
    _m5_demo(report, work, sound, checkpoint, device, state.get("child"), registry,
             m4.get("version"))


# =============================================================================
# Phase R — contract M2's acceptance on real data (--real-workspace)
# =============================================================================
def _stat_tree(directory: Path) -> Dict[str, Tuple[int, int]]:
    """Every file and directory under `directory` by size and modification time,
    without reading a byte: a workspace's tables can be gigabytes."""
    taken: Dict[str, Tuple[int, int]] = {}
    for path in sorted(directory.rglob("*")):
        status = path.stat()
        key = str(path.relative_to(directory)) + ("/" if path.is_dir() else "")
        taken[key] = (0, 0) if path.is_dir() else (status.st_size, status.st_mtime_ns)
    return taken


def _training_settings(config_path: Path) -> Tuple[Dict[str, Any], str]:
    """The training settings, parsed from one read of a `config.yaml` as
    `train_model.py` writes one, and the digest of those bytes. Unknown keys are
    refused rather than dropped, so every setting named is a setting applied.

    The output locations an older `config.yaml` names (`output_dir`,
    `checkpoint_dir`, `log_dir`, retired by contract M3a-2) are locations, not
    settings, and this probe names every run's directory itself; they are
    dropped, never applied."""
    import dataclasses

    import yaml

    from scripts.train_model import TrainConfig
    from src.utils.checkpoint_paths import RETIRED_LOCATION_FIELDS
    from src.utils.fingerprint import read_once

    read = read_once(config_path)
    loaded = yaml.safe_load(read.data.decode("utf-8")) or {}
    assert isinstance(loaded, dict), "the training settings are not a mapping"
    loaded = {k: v for k, v in loaded.items() if k not in RETIRED_LOCATION_FIELDS}
    known = {field.name for field in dataclasses.fields(TrainConfig)}
    unknown = sorted(set(loaded) - known)
    assert not unknown, f"the training settings name fields TrainConfig does not have: {unknown}"
    return loaded, read.identity.sha256


@_under_the_probes_own_gpu_lock
def phase_m2_real(
    report: Report, work: Path, real: Path, config_path: Path, reference: Path,
    workspaces_root: Path, device: str,
) -> None:
    """Contract M2's acceptance on real data: the full success path, from the
    production entry points, on a workspace that is read and never written;
    contract M4's flow on it (RT1-RT3) and M5's (RW1-RW3) before R8 checks it
    unchanged. The formal entry points take the probe's own GPU lock
    (`_probes_own_gpu_lock`), under the work directory."""
    print("\nR. Contract M2 on real data, with M4's flow")
    from src.utils.fingerprint import file_sha256

    runs = work / "real"
    runs.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {"before": _stat_tree(real)}
    reference_before = (reference.stat().st_size, reference.stat().st_mtime_ns)

    def _workspace() -> Tuple[str, Dict[str, Any]]:
        from src.inference.sp_artifact import (
            TENSOR_BUILD_ID_KEY,
            declares_binding,
            kg_binding_state,
            read_binding,
            read_sidecar,
            sidecar_path,
        )

        digests, cohorts, read, record = verify_workspace(real)
        # **Built from files, every input identified** (contract M3b): a real
        # workspace whose record were synthetic, or left an input unidentified,
        # would not be the real-data run this part claims.
        assert (record.binding, record.origin) == ("verified", "files"), (
            f"the real workspace's record is {record.binding}, origin {record.origin}"
        )
        table = real / "shortest_paths.pt"
        assert table.is_file() and sidecar_path(table).is_file(), (
            "the workspace has no shortest-path table with its sidecar"
        )
        # **A pre-check, before two epochs are spent**: the sidecar names this
        # workspace's graph. The verdict that the table is served verified is
        # the service loader's, in R6, which also reads the tensor.
        meta = read_sidecar(table).meta
        assert declares_binding(meta, ()), "the table's sidecar records no source binding"
        binding = read_binding(meta, (TENSOR_BUILD_ID_KEY,), source=table.name)
        assert kg_binding_state(binding, digests["kg"], source=table.name) == "verified", (
            "the table's sidecar does not name this workspace's graph"
        )
        state["kg"], state["record"], state["read"] = digests["kg"], record, read
        state["bound"] = dict(digests)
        return "the workspace's graph and training inputs verify", {
            "kg_digest": digests["kg"],
            "dataset_record": _record_facts(record),
            "train_diseases": len(cohorts.disease_sets["train"]),
            "val_diseases": len(cohorts.disease_sets["val"]),
            "sidecar_names_this_graph": True,
        }

    report.run("R1", "real",
               "the real workspace verifies, and its table's sidecar names its graph",
               _workspace)

    def _settings() -> Tuple[str, Dict[str, Any]]:
        from src.utils.checkpoint_io import read_checkpoint

        import dataclasses

        from scripts.train_model import TrainConfig

        settings, settings_sha256 = _training_settings(config_path)
        # **The values in effect**, TrainConfig's defaults included: a field the
        # file leaves out is still a setting the run uses.
        effective = dataclasses.asdict(TrainConfig(**settings))
        reference_read = read_checkpoint(reference, map_location="cpu", weights_only=True)
        recorded = reference_read.checkpoint
        architecture = ((recorded.get("config") or {}).get("model_config") or {})
        assert architecture.get("conv_type"), "the reference checkpoint records no architecture"
        compared = [name for name in _ARCHITECTURE_FIELDS if name in architecture]
        differ = sorted(name for name in compared if effective[name] != architecture[name])
        assert not differ, f"the training settings differ from the reference's architecture: {differ}"
        state["settings"] = settings
        return "the settings use the reference's architecture", {
            "settings_sha256": settings_sha256,
            "reference_sha256": reference_read.identity.sha256,
            "conv_type": effective["conv_type"],
            "architecture_fields_compared": compared,
            "fields_set_by_this_probe": sorted(
                ("data_dir", "run_dir", "num_epochs", "device", "resume_from",
                 "pause_after_epoch")),
        }

    report.run("R2", "real", "the training settings carry the reference's architecture",
               _settings)

    def _initial() -> Tuple[str, Dict[str, Any]]:
        assert "settings" in state and "kg" in state, "R1 or R2 did not pass"
        # One epoch of a two-epoch plan, paused (contract M3a-3), so R4 has
        # an unfinished stage to resume.
        parent = _last(_train_run(real, runs / "initial", device, 2,
                                  settings=state["settings"], pause_after=1))
        read = _read_run_checkpoint(parent)
        assert read["progress"].get("end") == "paused", "the initial run did not pause"
        missing = sorted({"kg", "node_features", "edge_indices", "num_nodes",
                          "split_manifest", "train_samples"} - set(read["record"]))
        assert not missing, f"the checkpoint does not record {missing}"
        assert read["record"]["kg"] == state["kg"], "the checkpoint records another graph"
        state["parent"] = parent
        return "one epoch of two from scratch, paused, recorded", {
            "sha256": read["sha256"], "recorded_roles": sorted(read["record"]),
            "optimizer_step": read["optimizer_step"],
        }

    report.run("R3", "real", "a short training from scratch pauses and records its inputs",
               _initial)

    def _resumed() -> Tuple[str, Dict[str, Any]]:
        assert "parent" in state, "R3 did not pass"
        observed: List[Dict[str, Any]] = []
        child = _last(_train_run(real, runs / "resumed", device, 2,
                                 settings=state["settings"], resume_from=state["parent"],
                                 observe=observed))
        facts = _assert_resumed(state["parent"], child, observed)
        progress = _read_run_checkpoint(child)["progress"]
        assert progress.get("end") == "finished" and progress.get("completed_epochs") == 2
        state["child"] = child
        return "the remaining epoch from the paused checkpoint, recorded", facts

    report.run("R4", "real", "a resume of the paused run trains its remaining epoch and names it",
               _resumed)

    def _measured() -> Tuple[str, Dict[str, Any]]:
        assert "child" in state, "R4 did not pass"
        output = runs / "measurement" / "measurement.json"
        assert _measure(state["child"], real, output, device) == 0, "the measurement failed"
        return "measured through the production entry point", _measurement_facts(
            output, _read_run_checkpoint(state["child"])["sha256"], state["kg"],
            state["record"])

    report.run("R5", "real", "measurement accepts and measures the new checkpoint", _measured)

    def _served() -> Tuple[str, Dict[str, Any]]:
        assert "child" in state, "R4 did not pass"
        facts = _serve_and_diagnose(real / "kg.json", state["child"], real, device,
                                    state["record"], state["read"])
        return "served with the table verified, and a diagnosis ran", facts

    report.run("R6", "real", "serving builds the new checkpoint, SP verified, and diagnoses",
               _served)

    def _inventory() -> Tuple[str, Dict[str, Any]]:
        import scripts.audit_checkpoint_family as audit

        assert "child" in state, "R4 did not pass"
        deployment = runs / "inventory_deployment.json"
        assert audit.main(["--inventory", str(workspaces_root),
                           "--output", str(deployment)]) == 0
        listing = json.loads(deployment.read_text())
        name = reference.resolve().relative_to(workspaces_root.resolve()).as_posix()
        workspace = name.split("/", 1)[0]
        old = listing["workspaces"][workspace]["checkpoints"][name]["m2_contract"]
        reason = old["serve_and_measure"]["reason"] or ""
        assert old["serve_and_measure"]["verdict"] == "refused", "the reference was accepted"
        assert ("records no training_input_digests" in reason or "kg is not recorded" in reason), (
            "the reference was refused for something other than its missing kg record"
        )
        # The new checkpoint lives outside the workspace, so it is listed through
        # a view: the workspace's own files, linked, and the checkpoint beside them.
        view = runs / "inventory_view" / real.name
        (view / "checkpoints" / "m2").mkdir(parents=True)
        for path in real.iterdir():
            if path.is_file():
                (view / path.name).symlink_to(path.resolve())
        (view / "checkpoints" / "m2" / "new.pt").symlink_to(state["child"].resolve())
        new_report = runs / "inventory_new_model.json"
        assert audit.main(["--inventory", str(view.parent), "--output", str(new_report)]) == 0
        new = json.loads(new_report.read_text())["workspaces"][real.name]["checkpoints"][
            f"{real.name}/checkpoints/m2/new.pt"]["m2_contract"]
        assert new["serve_and_measure"]["verdict"] == new["resume"]["verdict"] == "accepted", (
            "the new checkpoint did not pass M2's record checks"
        )
        return "the old model is refused for its missing record; the new one passes", {
            "deployment_report_sha256": file_sha256(deployment),
            "deployment_counts": listing["counts"],
            "reference_refusal": (
                "no record" if "records no training_input_digests" in reason else "no kg role"),
            "new_model_report_sha256": file_sha256(new_report),
        }

    report.run("R7", "real",
               "the inventory refuses the old model and passes the new one's record", _inventory)

    # ---- Contract M4's flow on real data, before R8 so R8 covers it. The
    # source, the registry, the kept report and the ledger (beside R4's
    # checkpoint, which is under --work-dir) are all under the work directory.
    m4_dir = runs / "m4"
    registry = m4_dir / "registry"

    def _real_cohort() -> Tuple[str, Dict[str, Any]]:
        from src.kg.graph import KnowledgeGraph
        from src.kg.storage.file_storage import read_samples

        assert "bound" in state, "R1 did not pass"
        # The real workspace's own val cases, written back through its graph's
        # ids: a synthetic supplied cohort whose relation to training is known.
        reverse = KnowledgeGraph.read_json(real / "kg.json").kg.get_reverse_node_mapping()
        cases, not_named = [], 0
        for sample in read_samples(real, "val").samples:
            if len(cases) == M4_REAL_CASES:
                break
            case = _source_case(f"probe-val-{len(cases) + 1:03d}", sample, reverse)
            if case is None:
                not_named += 1
            else:
                cases.append(case)
        del reverse
        assert cases, "no val case of the real workspace can be written as a shepherd-cases/1 case"
        registry.mkdir(parents=True)
        source = _write_cohort_source(m4_dir / "cases.json", cases)
        version, preview = _import_version(registry, real, source, M4_REAL_LABEL)
        assert version.manifest["graph"] == state["bound"], (
            "the version's graph digests are not the real workspace's")
        recorded = version.manifest["dataset_record"]["sha256"]
        assert recorded == state["record"].summary()["sha256"], (
            "the version is bound to another dataset record than the real workspace's")
        counts = version.manifest["counts"]
        assert (counts["source_cases"], counts["ranked_eligible"], counts["n_excluded"],
                counts["terms_unmappable"]) == (len(cases), len(cases), 0, 0), (
            f"the import of {len(cases)} val cases counted {counts}")
        state["m4"] = {"version": version, "n": len(cases)}
        return f"{len(cases)} val cases imported with no exclusions, bound to the real graph", {
            "val_cases_written": len(cases), "val_cases_not_nameable_skipped": not_named,
            "preview_wrote_nothing": True, "preview_id": preview,
            "manifest_sha256": version.identity.sha256, "graph_is_the_workspaces": True,
            "counts": counts,
        }

    def _real_measured() -> Tuple[str, Dict[str, Any]]:
        assert "m4" in state, "the cohort version (RT1) is not available"
        assert "child" in state, "R4 did not pass"
        version, n = state["m4"]["version"], state["m4"]["n"]
        # R4's checkpoint is under --work-dir, so the ledger --record writes
        # beside its resolved path is under it too.
        sha256 = _read_run_checkpoint(state["child"])["sha256"]
        code = _measure_cohort(state["child"], real, registry, M4_REAL_LABEL, device, "--record")
        assert code == 0, f"the measurement with --record exited {code}"
        _, kept, identity, rows = _kept_report(registry, M4_REAL_LABEL)
        assert kept["manifest"]["artifact_digests"]["checkpoint"] == sha256, (
            "the report does not name the checkpoint it was given")
        assert kept["n_ranked"] == n
        r10 = kept["manifest"]["r10"]
        # **As constructed**: the generated split is disease-disjoint, and the
        # cohort is the validation cases themselves.
        disease = r10["training"]["disease_overlap"]
        assert (disease["state"], disease["k"]) == ("none", 0), (
            f"R10's training disease overlap is {disease['state']} ({disease['k']}), not none")
        phenotype_sets = r10["validation"]["phenotype_set_overlap"]
        assert (phenotype_sets["state"], phenotype_sets["k"], phenotype_sets["n"]) == (
            "overlap", n, n), (
            f"R10's validation phenotype-set overlap is {phenotype_sets['state']} "
            f"({phenotype_sets['k']} of {phenotype_sets['n']}), not {n} of {n}")
        _evidence_found_in(r10, real, ("training", "validation"))
        record, ledger_sha256 = _the_record(state["child"].resolve().parent, identity, sha256,
                                            version)
        state["m4"].update(record=record, ledger_sha256=ledger_sha256, checkpoint_sha256=sha256)
        return "measured and recorded; R10 says what the construction makes true", {
            "exit_code": code, "checkpoint_sha256": sha256, "report_sha256": identity.sha256,
            "ranks_rows": rows, "n_ranked": kept["n_ranked"], "r10": _r10_facts(r10),
            "r10_evidence_in_the_workspace": True, "ledger_sha256": ledger_sha256,
            "record": {"sequence": record["registration"]["sequence"],
                       "source_artifact_digest": record["source_artifact_digest"]},
            "cuda_executed": kept["manifest"].get("cuda_executed"),
        }

    def _real_view() -> Tuple[str, Dict[str, Any]]:
        assert "record" in state.get("m4", {}), "the measurement (RT2) is not available"
        m4 = state["m4"]
        served, answer = _serve_and_view(real, state["child"], m4["checkpoint_sha256"], device)
        shown = _the_view_shows(answer, m4["record"], m4["ledger_sha256"],
                                ("training", "validation"))
        return "the served real checkpoint's record is shown by the read-only view", {
            "served": "reloaded through the reload route's handler with R4's checkpoint named "
                      "explicitly and bound to its digest (expected_checkpoint_sha256; R6 "
                      "builds its pipeline without publishing or keeping it); "
                      "read through GET /api/v1/pipeline/status and GET "
                      "/api/v1/evaluations/loaded-model",
            **shown,
        }

    # The probe's registry is the deployment's for the whole flow (INT-1).
    with _registry_setting(registry):
        report.run("RT1", "real",
                   "M4: a synthetic cohort of real val cases imports with no exclusions on the "
                   "real graph", _real_cohort)
        report.run("RT2", "real",
                   "M4: the new checkpoint measured on it with --record, R10 as constructed",
                   _real_measured)
        report.run("RT3", "real", "M4: the read-only view shows that record for the served "
                   "real checkpoint", _real_view)

    # ---- M5's acceptance on real data, before R8 so R8 covers it.
    _m5_real(report, runs, real, workspaces_root, device, state)

    def _untouched() -> Tuple[str, Dict[str, Any]]:
        after = _stat_tree(real)
        before = state["before"]
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
        assert not (added or removed or changed), (
            f"the real workspace changed: {len(added)} added, {len(removed)} removed, "
            f"{len(changed)} changed"
        )
        assert (reference.stat().st_size, reference.stat().st_mtime_ns) == reference_before, (
            "the reference checkpoint changed"
        )
        return "the same files and directories, sizes and modification times", {
            "entries_checked": len(after),
            "compared": "listing, size and modification time; not content digests",
        }

    report.run("R8", "real",
               "the real workspace and the old model are unchanged by listing, size and mtime",
               _untouched)


# =============================================================================
# Phase F — the real-data build (opt-in; minutes, not seconds)
# =============================================================================
def phase_real_build(
    report: Report, work: Path, external: Path, num_train: int, num_val: int,
    mondo_path: Optional[Path] = None, hpo_path: Optional[Path] = None,
) -> None:
    print("\nF. The real-data build")

    def _build() -> Tuple[str, Dict[str, Any]]:
        from scripts.build_knowledge_graph import build_knowledge_graph

        missing = [
            name for name in ("phenotype.hpoa", "genes_to_phenotype.txt")
            if not (external / name).exists()
        ]
        if missing:
            raise SkipProbe(f"the annotation files are not present ({', '.join(missing)})")

        workspace = work / "real_workspace"
        started = time.time()
        build_knowledge_graph(
            external_dir=external, workspace=workspace,
            generate_samples=True, num_train=num_train, num_val=num_val,
            val_disease_fraction=0.15, sample_seed=SEED,
            mondo_path=mondo_path, hpo_path=hpo_path,
        )
        elapsed = round(time.time() - started, 1)

        digests, cohorts, _, record = verify_workspace(workspace)
        overlap = cohorts.disease_sets["train"] & cohorts.disease_sets["val"]
        assert not overlap, "the real cohorts share a disease"
        # The builder identifies all four inputs, or stops (contract M3b).
        assert (record.binding, record.origin) == ("verified", "files"), (
            f"the real build's record is {record.binding}, origin {record.origin}"
        )
        return "a real workspace was built and verified", {
            "kg_digest": digests["kg"],
            "dataset_record": _record_facts(record),
            "train_diseases": len(cohorts.disease_sets["train"]),
            "val_diseases": len(cohorts.disease_sets["val"]),
            "disease_overlap": len(overlap),
            "build_seconds": elapsed,
        }

    report.run(
        "F1", "real_build", "a real workspace builds and verifies here", _build
    )

    def _rebuild_is_the_same_workspace() -> Tuple[str, Dict[str, Any]]:
        """Two builds from the same annotation files, compared file by file.

        **The manifest scheme rests on this and nothing tested it.** A digest
        binds artifacts to a production event; if the same inputs produced a
        different workspace each time, a digest would identify a run rather than
        a graph, and two sites could not confirm they held the same one.

        Every canonical file, not the four graph artifacts. The first version of this
        probe checked only those, which would have called a workspace reproduced
        while its cohorts were not — and the cohorts are what a measurement is
        taken over. The digests are recorded so two reports from two machines
        can be compared directly, which is the cross-site claim this exists for
        and which one machine cannot settle.
        """
        from scripts.build_knowledge_graph import build_knowledge_graph
        from src.utils.fingerprint import file_sha256

        first = work / "real_workspace"
        if not (first / "kg.json").exists():
            raise SkipProbe("the first build did not complete")
        second = work / "real_workspace_again"
        build_knowledge_graph(
            external_dir=external, workspace=second,
            generate_samples=True, num_train=num_train, num_val=num_val,
            val_disease_fraction=0.15, sample_seed=SEED,
            mondo_path=mondo_path, hpo_path=hpo_path,
        )
        # What a generated workspace is: the four graph artifacts, both cohorts,
        # the manifest, and the provenance record. Named rather than globbed, so
        # "the same workspace" means the same thing here as it does to the
        # verifiers, and an unrelated file dropped into the directory neither
        # joins the claim nor breaks it.
        #
        # Provenance belongs in this comparison rather than beside it: it is
        # derived from the same inputs, so two builds that agree on everything
        # else and disagree here would mean the record is picking up something
        # the build did not.
        from src.kg.artifacts import GRAPH_ARTIFACTS, MANIFEST_FILENAME
        from src.kg.provenance import PROVENANCE_FILENAME

        names = sorted(
            list(GRAPH_ARTIFACTS.values())
            + ["train_samples.json", "val_samples.json",
               MANIFEST_FILENAME, PROVENANCE_FILENAME]
        )
        # **Absence is checked before content, in both directions.** Hashing a
        # path that is not there raises out of the comprehension, and what the
        # operator would then read is a traceback about a missing file rather
        # than this probe's finding, which is that a build did not write one.
        # The comparison is only meaningful once both sides are known present.
        missing = sorted(
            f"{label}/{name}"
            for label, root in (("first", first), ("second", second))
            for name in names if not (root / name).exists()
        )
        assert not missing, f"a build did not write {missing}"
        digests = {name: file_sha256(first / name) for name in names}
        differing = sorted(
            name for name in names
            if file_sha256(second / name) != digests[name]
        )
        assert not differing, (
            f"two builds from the same annotation files differ in {differing}: "
            "a digest would then identify a run rather than a workspace"
        )
        return "the same inputs produced the same workspace", {
            "canonical_artifacts_compared": len(names),
            # Recorded so a report from another machine can be diffed against
            # this one without rebuilding anything.
            "digests": digests,
        }

    report.run(
        "F2", "real_build",
        "the same annotation files produce the same workspace",
        _rebuild_is_the_same_workspace,
    )


# =============================================================================
# Phase G — the download path, for real (opt-in; downloads whole ontologies)
# =============================================================================
#: The two ontologies a build consumes, and so the two this phase fetches.
DOWNLOAD_ONTOLOGIES = ("mondo", "hpo")


def _route_facts(urls, settings) -> List[Dict[str, Any]]:
    """What this machine's network looks like to each configured source.

    **Observed by the probe, beside the downloader, not instead of it.** These
    ask the production `DestinationPolicy` the same questions the downloader
    asks — does the name resolve here, will the request go to a proxy, is the
    host allow-listed — at the moment the probe runs. What the downloader
    actually decided is in the attempts; these say why that is plausible. Each
    source is named by its position in the configured list, never by its host.
    """
    from urllib.parse import urlsplit

    from src.ontology.download import DestinationPolicy, OntologyDownloadError

    policy = DestinationPolicy(allowed_hosts=settings.allowed_hosts)
    facts = []
    for index, url in enumerate(urls, start=1):
        parts = urlsplit(url)
        host = parts.hostname or ""
        try:
            policy.resolve(host)
            resolves = True
        except OntologyDownloadError:
            resolves = False
        facts.append({
            "index": index,
            "scheme": (parts.scheme or "").lower(),
            "host_resolves_here": resolves,
            "goes_through_proxy": bool(policy.routes_through_proxy(url)),
            "host_allow_listed": bool(policy.permits_host(host)),
        })
    return facts


#: The forms a declared release may take and still be written into a report:
#: the OBO Foundry release path every measured header uses
#: (`releases/2026-09-01`, `hp/releases/2026-09-01`,
#: `mondo/releases/2026-09-01/mondo-simple.owl`), or a bare date.
_RELEASE_LABEL = re.compile(
    r"(?:[a-z][a-z0-9_-]{0,31}/)?releases/\d{4}-\d{2}-\d{2}"
    r"(?:/[a-z][a-z0-9_.-]{0,63}\.(?:obo|owl))?"
    r"|\d{4}-\d{2}-\d{2}"
)
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _reportable_version(declared: Optional[str]) -> Dict[str, Any]:
    """The declared release, in a form a shared report may carry.

    **`data-version` is text the file chooses**, and a file can choose a URL
    with credentials in it, a host or a directory. So only the release forms
    above pass through as they are; anything else is withheld and reduced to
    the date it contains, if any. Nothing is lost that identifies the file:
    `source_digest` is the exact identity, and this is a label.
    """
    if declared is None:
        return {"data_version": None, "data_version_withheld": False, "data_version_date": None}
    date = _DATE.search(declared)
    if _RELEASE_LABEL.fullmatch(declared):
        return {"data_version": declared, "data_version_withheld": False,
                "data_version_date": date.group(0) if date else None}
    return {"data_version": None, "data_version_withheld": True,
            "data_version_date": date.group(0) if date else None}


def _outside_work_dir(settings) -> List[Path]:
    """Every place the download must leave alone: the configured roots and the
    default cache. The download goes into `--work-dir`, and nowhere else."""
    from src.ontology.loader import default_cache_dir

    return [Path(root) for root in settings.roots] + [default_cache_dir()]


def phase_download(report: Report, work: Path) -> None:
    """The production fetch path, end to end, on this machine's network.

    **The same call the build makes**, `_select_and_load(..., force_download=
    True)`, so the configured sources, the scheme and destination rules on every
    redirect, per-fetch staging, the imports policy, the role check and the
    publish all run as they do in a build. Nothing here decides a rule; this
    only reports what the rules decided. A successful HTTP transfer is not a
    pass: the file has to load, pass both checks and be published as the bytes
    that were verified.

    The cache is a fresh directory inside `--work-dir`, so the download really
    happens and nothing an operator keeps is written. Timeouts are the
    downloader's own, deliberately: if they are wrong for this network, that is
    a finding about production, not something for the probe to paper over.
    """
    print("\nG. The download path, for real")
    print("   Downloads MONDO and HPO in full from the configured sources into "
          "the work directory. Minutes on a slow link.")
    cache = work / "download_cache"
    state: Dict[str, Any] = {}

    def _before() -> Tuple[str, Dict[str, Any]]:
        from src.ontology.settings import load_ontology_settings

        try:
            settings = load_ontology_settings()
        except Exception as exc:  # noqa: BLE001 — classified, never copied
            raise ProbeFailure(
                "the ontology configuration could not be read",
                {"outcome": "configuration_invalid", "error_type": type(exc).__name__},
            ) from None
        state["settings"] = settings
        state["outside"] = {str(i): snapshot(p) for i, p in enumerate(_outside_work_dir(settings))}
        return "configuration read; locations outside the work directory recorded", {
            "locations_recorded": len(state["outside"]),
            "files_recorded": sum(len(v) for v in state["outside"].values()),
        }

    report.run("G0", "download", "the ontology configuration is readable", _before)

    def _download(name: str) -> Tuple[str, Dict[str, Any]]:
        from scripts.build_knowledge_graph import _select_and_load
        from src.ontology.loader import OntologyFetchError, OntologyImportError, OntologyLoader
        from src.ontology.roles import OntologyRoleError
        from src.ontology.settings import DEFAULT_ONTOLOGY_SOURCES
        from src.utils.fingerprint import file_sha256

        settings = state.get("settings")
        if settings is None:
            raise SkipProbe("the ontology configuration could not be read (G0)")
        urls = settings.urls_for(name)
        facts: Dict[str, Any] = {
            "ontology": name,
            "sources_configured": len(urls),
            "sources_are_project_defaults": tuple(urls) == tuple(DEFAULT_ONTOLOGY_SOURCES.get(name, ())),
            "network_at_probe_time": _route_facts(urls, settings),
        }
        loader = OntologyLoader(cache_dir=cache)

        def attempts(record) -> List[Dict[str, Any]]:
            return [{"index": a.index, "format": a.format, "outcome": a.outcome} for a in record]

        def failed(outcome: str, detail: str, **more: Any) -> ProbeFailure:
            facts.update(outcome=outcome, attempts=attempts(loader.last_fetch_attempts), **more)
            return ProbeFailure(f"{name}: {detail}", facts)

        try:
            ontology = _select_and_load(loader, name, None, cache, True)
        except OntologyFetchError as exc:
            facts["attempts"] = attempts(exc.attempts)
            facts["outcome"] = exc.reason
            raise ProbeFailure(f"{name}: no source delivered a usable file ({exc.reason})", facts) from None
        except OntologyImportError:
            raise failed("imports_declared", "the delivered file declares imports") from None
        except OntologyRoleError:
            raise failed("wrong_ontology", "the delivered file is not this ontology") from None
        except OSError as exc:
            raise failed("local_filesystem_error", "the work directory could not be written",
                         error_type=type(exc).__name__) from None
        except Exception as exc:  # noqa: BLE001 — a parse failure, most likely; type only
            raise failed("failed_otherwise", "the delivered file could not be loaded",
                         error_type=type(exc).__name__) from None

        published = ontology.source_path
        facts.update(
            outcome="published",
            attempts=attempts(loader.last_fetch_attempts),
            format=published.suffix.lstrip("."),
            source_digest=ontology.source_digest,
            **_reportable_version(ontology.declared_version),
            size_bytes=published.stat().st_size,
            term_count=ontology.num_terms,
        )
        assert published.parent == cache, f"{name}: published outside the work directory"
        assert file_sha256(published) == ontology.source_digest, (
            f"{name}: the published file is not the bytes that were verified"
        )
        leftovers = [p for p in cache.iterdir() if p.name.endswith(".staged")]
        assert not leftovers, f"{name}: {len(leftovers)} staging file(s) left behind"
        return f"{name} downloaded, verified and published", facts

    for index, name in enumerate(DOWNLOAD_ONTOLOGIES, start=1):
        report.run(f"G{index}", "download",
                   f"{name} downloads, verifies and publishes through the build's own path",
                   lambda name=name: _download(name))

    def _untouched() -> Tuple[str, Dict[str, Any]]:
        """By position, kind and count — **never by name**. The shared
        `assert_untouched` lists the files it found, which is right for the
        work directory and wrong here: these are the operator's directories,
        and a failure is exactly when their contents would reach the report."""
        settings = state.get("settings")
        if settings is None:
            raise SkipProbe("the ontology configuration could not be read (G0)")
        locations = _outside_work_dir(settings)
        changes = []
        for key, location in enumerate(locations):
            before, after = state["outside"][str(key)], snapshot(location)
            changes.append({
                "location": key,
                "kind": "default cache" if key == len(locations) - 1 else "configured root",
                "created": len(set(after) - set(before)),
                "removed": len(set(before) - set(after)),
                "rewritten": sum(1 for k in set(before) & set(after) if before[k] != after[k]),
            })
        touched = [c for c in changes if c["created"] or c["removed"] or c["rewritten"]]
        if touched:
            raise ProbeFailure(
                "the download phase changed files outside the work directory: " + "; ".join(
                    f"location {c['location']} ({c['kind']}): {c['created']} created, "
                    f"{c['removed']} removed, {c['rewritten']} rewritten" for c in touched),
                {"changes": changes},
            )
        return "nothing outside the work directory was touched", {
            "locations_checked": len(changes), "changes": changes,
        }

    report.run("G9", "download", "nothing outside the work directory was written",
               _untouched)


# =============================================================================
# Entry point
# =============================================================================
#: Every path argument the probe takes. `parse_args` makes each absolute once,
#: against the directory the probe was started in, before any check or phase
#: reads it (BACKLOG 49): the phases, the server's directory check (RW1 refuses
#: a relative workspace root) and the subprocesses run from the repository all
#: get the same absolute path, and a relative one given from elsewhere is
#: refused by the checks below before anything is trained.
PATH_ARGUMENTS: Tuple[str, ...] = (
    "work_dir", "report", "external_dir", "mondo_path", "hpo_path",
    "real_workspace", "real_config", "real_reference_checkpoint", "workspaces_root",
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attack this deployment and report what held. A relative path "
                    "is taken from the directory the probe is started in, made "
                    "absolute once before anything runs.",
    )
    parser.add_argument(
        "--work-dir", type=Path, required=True,
        help="Scratch directory for everything this probe writes. Must be empty "
             "or absent: the probe removes it on success and never writes "
             "outside it.",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Where to write the JSON report (default: <work-dir>/probe_report.json)",
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cuda", "cpu"),
        help="Serving and training device (default: auto)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Training epochs for the demo workspace (default: 3)",
    )
    parser.add_argument(
        "--external-dir", type=Path, default=None,
        help="Adds the real-data build. Directory holding phenotype.hpoa and "
             "genes_to_phenotype.txt. Takes minutes.",
    )
    parser.add_argument(
        "--mondo-path", type=Path, default=None,
        help="MONDO file for the real-data build, passed to it as given (made "
             "absolute, like every path here). "
             "Needed when the ontology roots hold more than one candidate, "
             "which the build refuses to choose between.",
    )
    parser.add_argument(
        "--hpo-path", type=Path, default=None,
        help="HPO file for the real-data build, with the same rule as --mondo-path.",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Adds the download phase: MONDO and HPO fetched in full from the "
             "configured sources, through the build's own path, into the work "
             "directory. Needs network access, not a GPU.",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Runs only the environment description and the download phase.",
    )
    parser.add_argument("--num-train", type=int, default=200000)
    parser.add_argument("--num-val", type=int, default=40000)
    parser.add_argument(
        "--keep", action="store_true",
        help="Keep the work directory instead of removing it at the end",
    )
    parser.add_argument(
        "--m2-acceptance", action="store_true",
        help="Adds contract M2's acceptance on the demo workspace (phase M) and "
             "accounts for every probe it requires. It also carries contract M4's "
             "flow (T1-T6; RT1-RT3 with --real-workspace), functional only: not model "
             "quality, not an institutional test; and M5's acceptance where it runs "
             "headless (A5, W1-W11; RW1-RW3 with --real-workspace): the service it "
             "starts on a free loopback port, the GPU lock, test and training jobs "
             "through the job runner. Needs --device cuda and a --work-dir outside the "
             "repository (M4's cohort registry is made under it); keeps the work "
             "directory as evidence. Run it while no managed GPU job runs.",
    )
    parser.add_argument(
        "--real-workspace", type=Path, default=None,
        help="Adds M2's acceptance on this real workspace (phase R), with M4's RT1-RT3: "
             "it is read, never written; every run, checkpoint, report, cohort registry "
             "and ledger goes under --work-dir.",
    )
    parser.add_argument(
        "--real-config", type=Path, default=None,
        help="The training settings for phase R: a config.yaml as train_model.py "
             "writes one. Its architecture must match --real-reference-checkpoint's.",
    )
    parser.add_argument(
        "--real-reference-checkpoint", type=Path, default=None,
        help="The model phase R's architecture is taken from, and which the "
             "inventory must refuse for its missing record. Not modified.",
    )
    parser.add_argument(
        "--workspaces-root", type=Path, default=None,
        help="The directory of workspaces the deployment inventory lists "
             "(default: the real workspace's parent).",
    )
    args = parser.parse_args(argv)
    # **Absolute once, here** (BACKLOG 49): every check below, the phases and
    # the server's directory check read the same path. Absolute and normalised,
    # not resolved: a symbolic link keeps the name it was given, and the
    # containment checks below resolve as they did.
    for name in PATH_ARGUMENTS:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(os.path.abspath(value)))
    if args.download_only and args.external_dir is not None:
        parser.error("--download-only runs only the environment and download "
                     "phases; drop --external-dir or --download-only")
    if args.m2_acceptance and args.download_only:
        parser.error("--m2-acceptance trains and serves; it cannot run with --download-only")
    if args.m2_acceptance and args.device != "cuda":
        parser.error("--m2-acceptance is an acceptance on the deployment's device: "
                     "pass --device cuda, so it cannot become a CPU run")
    # Inputs a later phase reads, refused now rather than after the training
    # that runs first (the real build runs after the demo's).
    if args.external_dir is not None and not args.external_dir.is_dir():
        parser.error(f"--external-dir is not a directory: {args.external_dir}")
    for name in ("mondo_path", "hpo_path"):
        if getattr(args, name) is not None and not getattr(args, name).is_file():
            parser.error(f"--{name.replace('_', '-')} is not a file: {getattr(args, name)}")
    real_args = (args.real_config, args.real_reference_checkpoint, args.workspaces_root)
    if args.real_workspace is None and any(value is not None for value in real_args):
        parser.error("--real-config, --real-reference-checkpoint and --workspaces-root "
                     "belong to --real-workspace")
    if args.real_workspace is not None:
        if not args.m2_acceptance:
            parser.error("--real-workspace is part of --m2-acceptance")
        if args.real_config is None or args.real_reference_checkpoint is None:
            parser.error("--real-workspace needs --real-config and --real-reference-checkpoint")
        if args.workspaces_root is None:
            args.workspaces_root = args.real_workspace.parent
        for name in ("real_workspace", "workspaces_root"):
            if not getattr(args, name).is_dir():
                parser.error(f"--{name.replace('_', '-')} is not a directory: "
                             f"{getattr(args, name)}")
        for name in ("real_config", "real_reference_checkpoint"):
            if not getattr(args, name).is_file():
                parser.error(f"--{name.replace('_', '-')} is not a file: {getattr(args, name)}")
        root = args.workspaces_root.resolve()
        if root not in args.real_reference_checkpoint.resolve().parents:
            parser.error("--real-reference-checkpoint must be inside --workspaces-root, "
                         "where the inventory lists it")
        work = args.work_dir.resolve()
        if work == args.real_workspace.resolve() or args.real_workspace.resolve() in work.parents:
            parser.error("--work-dir must not be inside --real-workspace, which is never written")
    if args.m2_acceptance and args.report is not None:
        # **The acceptance writes under its work directory and nowhere else**, its
        # report included, and replaces nothing: the report cannot be an input or
        # an earlier result. The final write refuses an existing file as well,
        # for one that appears during the run.
        report = args.report.resolve()
        if args.work_dir.resolve() not in report.parents:
            parser.error("with --m2-acceptance, --report must be inside --work-dir")
        if report.exists():
            parser.error("--report names an existing file; the acceptance replaces nothing")
        inputs = [args.real_config, args.real_reference_checkpoint]
        if any(path is not None and report == path.resolve() for path in inputs) or (
            args.real_workspace is not None and args.real_workspace.resolve() in report.parents
        ):
            parser.error("--report must not be an input of the acceptance")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # **Absolute from here on.** parse_args made every path argument absolute
    # (`PATH_ARGUMENTS`, BACKLOG 49). Every path the phases write is joined onto
    # `work`, and some of them go where a relative path changes meaning or is
    # refused: M4's cohort registry (`<work>/m4/registry`, and the real part's
    # under `<work>/real`) must be absolute or the importer refuses it, after
    # the training runs have already been spent; and a subprocess run with the
    # repository as its working directory would read a relative path against
    # the repository. Resolved as parse_args' containment checks resolve it.
    work = args.work_dir = args.work_dir.resolve()
    if work.exists() and any(work.iterdir()):
        # **Refuse rather than clean.** The operator names this directory, and a
        # probe that emptied whatever it was pointed at would be one typo away
        # from deleting something that mattered. Saying what to do about it is
        # this message's job, though: the usual reason it is not empty is that
        # the previous run failed and kept its evidence on purpose.
        print(
            f"--work-dir must be empty or absent; {work} is not.\n"
            f"A previous run that failed keeps its work directory so the "
            f"failure can be inspected. Remove it with `rm -rf {work}` once you "
            f"are done with it, or pass a different --work-dir.",
            file=sys.stderr,
        )
        return 2
    work.mkdir(parents=True, exist_ok=True)
    report_path = args.report or (work / "probe_report.json")

    print("=" * 66)
    print("SHEPHERD-Advanced — adversarial deployment probe")
    print("=" * 66)

    report = Report()
    environment = phase_environment(report, args.device, describe_only=args.download_only)
    # No serving device is resolved in a download-only run, and the report
    # says so rather than naming one that was never checked.
    device = environment.get("resolved_device", None if args.download_only else "cpu")

    stopped = args.m2_acceptance and device != "cuda"
    if stopped:
        # **An acceptance on the deployment's device does not become a CPU run.**
        # A2 could not resolve CUDA, so nothing after it runs, and every probe
        # the acceptance requires is reported as not run.
        print("\nCUDA did not resolve (A2): the M2 acceptance stops here, incomplete.")
    if not args.download_only and not stopped:
        sound = phase_writer(report, work)
        if sound is not None:
            phase_workspace(report, work, sound)
            checkpoint = phase_training(report, work, sound, device, args.epochs)
            if checkpoint is not None:
                phase_serving(report, work, sound, checkpoint, device)
                if args.m2_acceptance and device == "cuda":
                    phase_m2_demo(report, work, sound, checkpoint, device)
        if args.real_workspace is not None and device == "cuda":
            phase_m2_real(report, work, args.real_workspace, args.real_config,
                          args.real_reference_checkpoint, args.workspaces_root, device)
        if args.external_dir is not None:
            phase_real_build(report, work, args.external_dir, args.num_train, args.num_val,
                             mondo_path=args.mondo_path, hpo_path=args.hpo_path)
    if args.download or args.download_only:
        phase_download(report, work)

    summary = report.summary()
    acceptance = (
        m2_acceptance(report, args.real_workspace is not None) if args.m2_acceptance else None
    )
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tested_commit": _tested_commit(),
        "allocator": _allocator_record(environment.get("allocator_backend")),
        "environment": environment,
        "settings": {
            "device": device,
            "epochs": args.epochs,
            "feature_dim": FEATURE_DIM,
            "seed": SEED,
            "real_build_requested": args.external_dir is not None,
            "download_requested": bool(args.download or args.download_only),
            "download_only": bool(args.download_only),
            "m2_acceptance_requested": bool(args.m2_acceptance),
            "real_data_requested": args.real_workspace is not None,
        },
        "summary": summary,
        "m2_acceptance": acceptance,
        "probes": [
            {
                "id": probe.probe_id,
                "phase": probe.phase,
                "claim": probe.claim,
                "status": probe.status,
                "detail": probe.detail,
                "facts": probe.facts,
                "seconds": probe.seconds,
            }
            for probe in report.probes
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.m2_acceptance:
        # **Created, never replaced** (mode "x"): a file the run produced at this
        # path, or one put there while it ran, is not overwritten. The report
        # goes to standard output instead, so its evidence is not lost either.
        try:
            with open(report_path, "x", encoding="utf-8") as handle:
                handle.write(encoded)
        except FileExistsError:
            print(f"\n{report_path} exists and is not replaced; the report follows on "
                  "standard output. The acceptance did not write its report.",
                  file=sys.stderr)
            print(encoded)
            return 3
    else:
        report_path.write_text(encoded)

    print("\n" + "=" * 66)
    print(
        f"passed {summary['passed']}  failed {summary['failed']}  "
        f"errors {summary['error']}  skipped {summary['skipped']}"
    )
    print(f"report: {report_path}")
    print("=" * 66)

    failed = summary["failed"] + summary["error"]
    if acceptance is not None:
        for part, name in (("demo", "demo"), ("real_data", "real data")):
            verdict = acceptance[part]
            if not verdict["requested"]:
                print(f"M2 acceptance, {name}: not requested in this run")
            elif verdict["complete"]:
                print(f"M2 acceptance, {name}: every required probe passed")
            else:
                print(f"M2 acceptance, {name}: INCOMPLETE {verdict['not_passed']}")
        print("M2 acceptance: " + ("COMPLETE" if acceptance["complete"] else
                                   "NOT COMPLETE (both parts must run and pass)"))
        if not acceptance["complete"]:
            failed += 1
    # **What the run ran with, said again where it ends** (A5): a set that
    # differs from the lock is not left on a line an hour up the console.
    closing = environment_line(report)
    if closing:
        print(closing)
    # **Containment, not parent equality.** A report at `<work>/sub/report.json`
    # has a parent that is not `work`, and removing `work` would take the report
    # with it -- the first version of this check did exactly that.
    report_inside_work = work.resolve() in report_path.resolve().parents
    if not failed and not args.keep and not args.m2_acceptance:
        if report_inside_work:
            print("(work directory kept because the report lives inside it)")
        else:
            shutil.rmtree(work, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    # Before main(): the environment phase is the first thing to import torch.
    ALLOCATOR.update(_apply_allocator())
    _stop_on_termination_signals()
    sys.exit(main())
