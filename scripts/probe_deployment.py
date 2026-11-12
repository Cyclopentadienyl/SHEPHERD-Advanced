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
    # workspace, and a history file when one is given, are read, never
    # written. The run also carries contract M4's flow (T1-T6, RT1-RT3), whose
    # cohort registry is made under the work directory, so the work directory
    # must be outside the repository.
    python scripts/probe_deployment.py --m2-acceptance --device cuda \
        --work-dir <empty directory outside the repository> \
        --report <that directory>/m2_report.json \
        --real-workspace data/workspaces/<workspace>
    # optionally also --real-reference-checkpoint <a history file inside
    # --workspaces-root: a checkpoint without a model spec and without a
    # training record of its graph (no training_input_digests, or none for
    # kg), such as the deployment's model-02, which R2 and R7 refuse>

    A relative path is taken from the directory the probe is started in: every
    path argument is made absolute once, before any check or phase reads it
    (BACKLOG 49), so the forms above work from the repository root.

The models the chain trains, serves and measures are model bundles: a `gatv2`
base and one disease child finetuned from it, through the production trainer
(`train_model.train` with `learning_stage`, `task` and `finetune_from`), each
filed under its encoder kind. Phase D trains them on the demo workspace (D1,
`--epochs` each) and checks the child's training and model records (D2); phase
E serves the child, and phase M takes the child and the base from there. No
shortest-path table is loaded anywhere: a task model never reads one, and the
inventory (M8, R7) reads neither file of a pair left in a workspace (its
shortest-path check was removed in K8-5).

Contract M2's acceptance (`--m2-acceptance`), which also carries contract
M3a's checks:
    Phase M, on the demo workspace, after B-E: a valid same-shaped workspace
    over another graph (M1); a disease finetune of D1's base paused after one
    of two planned epochs and resumed for the remaining one, stating its
    mixed precision (float16, in effect on CUDA only) and recording its parent
    and that precision, its gradient scaler's state checked right after the
    restore against the parent's (M2); measurement by the disease protocol
    (M3) and disease-task serving (M4) of the resumed child; the six resume
    refusals of D1's child, each leaving no run directory and nothing changed
    (M5, the last a parent without a model spec); serving (M6) and measurement (M7)
    refusing D1's child over the same-shaped workspace; the inventory's
    verdicts, a file without a model spec refused for both uses (M8); the
    finished child extended by one epoch at a fixed rate (M9); resuming that
    finished child refused before anything exists (M10).
    Phase R, on the real workspace: it verifies (R1); a base trained by the
    reference recipe, one epoch of two, paused (R3), and its remaining epoch
    resumed (R4); a disease child finetuned from that resumed base
    (R4.finetune); a file without a model spec refused by a finetune and a
    resume before anything is written, a stripped copy of that child and the
    history file when one is given, and the stripped copy also by serving
    before any model is built (R2); measurement of the child by the disease
    protocol (R5) and disease-task serving, with a diagnosis of a query from
    the workspace's own validation cases (R6); the inventory passes
    the child, and refuses the history file when one is given (R7); the
    workspace and the history file are unchanged by listing, size and
    modification time, not by content digest (R8), checked after every other
    probe of the phase, M4's included. Every run writes into its own run
    directory under --work-dir. Each continued run is judged against the
    precision it asked for (`_scaler_restore_facts`): M2 and M9 against M2's
    stated float16, R4 against the base recipe's fp32, which R3 asks for.
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
    Phase M, after M11, on the demo workspace and D1's disease child: a source
    the probe builds from the demo graph's own HP and MONDO ids, with terms
    that do not map and a disease that is not a node, previewed (nothing
    written) and committed by the printed preview id, its graph the
    workspace's and its counts as built (T1); a copy of the child measured by
    the disease protocol on that version with --record: the report kept with its ranks
    digest, its ranks one row per case of the version in its order
    (`check_ranks_rows`), its binding counts and ranked cases the ones the
    probe computes itself from the child's graph view and the version's cases
    (`_own_binding`), R10 recorded
    with both scopes' evidence found in the workspace, one record built from
    the report's bytes (T2); the same report registered again
    is already registered and a damaged copy is refused, the ledger unchanged
    (T3); the version imported against M1's same-shaped workspace is refused by
    the measurement before the checkpoint is read (T4); served through the
    reload route, bound to the digest expected and naming its task (as the
    dashboard's Load binds it), the measured checkpoint's record shown by the
    read-only view for the digest the status reports, uncontested in its
    version's report store (D5), and none for M2's child served beside the
    same ledger (T5); the overlap audit in registered mode (T6).
    Phase R, after R7: N of the real workspace's val cases written back as a
    source through the real graph's ids, imported with no exclusions (RT1);
    the child R4.finetune trained measured on it by the disease protocol with
    --record, its binding the probe's own as in T2, training disease overlap none and
    validation phenotype-set overlap N of N, both as constructed (RT2); that
    record shown for that child served,
    uncontested (RT3).
    The probe's registry is the deployment's (`SHEPHERD_REGISTRY`) for each
    flow as a whole, set once around T1-T6 and RT1-RT3 whatever the operator's
    shell names, and put back after (`_registry_setting`; root-cause round,
    INT-1).

Contract M3c (what is served is identified) adds checks, not probes:
    E1 requires the served identity to name the child D1 trained, the
    workspace's graph, manifest and dataset record by the digests the probe's
    own reads took, the disease task's scoring mode and its shortest-path
    state from the task tables (`TASK_SCORING_MODES`, `SCORING_MODE_SP_STATES`),
    and the device asked for, in effect; E2 that the status reports it
    unchanged; E5 that a refused reload leaves it as it was; E7 that a
    diagnosis carries it. M4 and R6 require the same of a disease-task
    diagnosis. The identity holds no paths and is recorded in E1's, M4's and
    R6's facts.

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
Since research-comparison Stage 2b, A1 also records each named component as
every run's environment does (`src/utils/run_environment.collect_components`:
the PyG extensions' native build identity, and the source receipt's facts where
it matches what is installed), and the verdict states each one's identity and
route.
    Phase M, after T6, on the probe's own deployment under <work>/m5: a copy of
    the demo workspace under its own workspace root, M2's resumed disease
    child copied into a run there, filed under its encoder kind, and T1's
    registry; W7-W11's test runs measure that listed copy by the disease
    protocol (a test run measures a model by a protocol this build implements
    that its capabilities admit, which the runner judges on the listed model's
    capability description and the listed version, `src.evaluation.protocols`). The server's lists name
    the workspace, its
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
    the application in this process (W6). A busy GPU lock refuses the runner's
    test and base training starts and both CLIs naming its holder, and a
    killed holder releases it (W7). A
    test run through the job runner, its inputs found in the server's lists:
    its outcome file, its report kept and not registered as the publication
    contract says, its usage-history entry, a training stop meanwhile refused
    (W8); that report registered from the store, a copy outside it refused
    (W9); a run started for another model or cohort fails before measuring,
    nothing kept (W10); a base training job holds the lock, a test start, a
    test cancel and the measurement CLI are refused, and its stop ends it as
    stopped (W11).
    Phase R, after RT3 and before R8, which therefore covers them: the real
    workspace as the server lists it (RW1); a test run through the runner on
    it, on the device, kept and registered, of a disease child the probe
    finetunes from R4's resumed base by the reference recipe, one epoch, into
    a run directory of its own under --work-dir (RW2); a base training job on
    it started and stopped, other jobs refused while it runs (RW3).
    **W11's and RW3's training job holds the GPU by configuration**, applied
    where it is started (`_held_config`: a new base run of as many epochs as
    the field spec allows, early stopping off, no pause), so it ends only by
    the probe's stop. **One
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
    that runs meanwhile. The research comparison's probes (L1-L8, below) run
    under settings of their own the same way (`_rc_demo`), their lock under
    the workspaces root of their own deployment. **The training probes take
    no GPU lock at all**, not even the probe's own (D1, M2, M9, R3, R4,
    R4.finetune and RW2's child train, and so do L4.other_graph's and
    L4.other_view's bases and L7's children; M5, M10, M11.training and R2 are
    refused before they do):
    they call the production trainer in this process, and `train()` takes no
    lock, only the program entries do (`src/utils/gpu_lock.py`), so they
    exclude nothing either. The deployment's lock is only read, once at W1
    and once at RW1, to say whether a job holds it at that moment; a job that
    ran during the
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

The research comparison round's probes (its plan's §5 Stage 10, the owner's
O9; the field contract the external closing check reads is
docs/working/research-comparison/GB10_ACCEPTANCE.md), carried by the same run,
functional only: not model quality, and no figure of one model compared with
another's. A6, in phase A, requires A1's components: each named component's
facts in the shape every run records, its native build identity, and its route
only where the source receipt matched what is installed ("not recorded"
otherwise); no version is compared with anything. Phase M, after W11, on the
probe's own deployment under <work>/rc (a copy of the demo workspace under a
root of its own, listing copies of D1's base and disease child, and a registry),
every step through its production entry point:
    a KG edge set, the test part of the edge split D1's base records, imported
    by the importer (`--kind kg_edges --from-checkpoint`), previewed and
    committed (L1); the base measured on it by the KG link-prediction protocol
    through measure_scorer with --record (L2) and through the job runner from
    the server's lists with record (L3), each kept and registered, its exposure
    scoped to the base's record and its scores file, the version's pairs and
    the report's summary checked against each other (`_lp_scores_verified`);
    the refusals before anything is written (L4): a KG edge set of another
    graph (a base trained on M1's graph, its own set imported there), a pairs
    file changed after its import, a model of another graph view (a base
    trained without the ortholog relation: a stated capability gap), and a
    split record whose seed and digest contradict each other; D1's disease
    child measured by the disease protocol on a supplied cohort and by the KG
    link-prediction protocol on L1's set, two independent records read by the
    one reader (L5); the workspace's functional gene fixture derived
    (`scripts/derive_task_fixture.py`) and its generated val split imported
    with it, measured by that child and registered (L6); the GPU regression
    cases, a short disease, gene and patients-like-me child of the listed base
    each trained on the probe's device at the demo's sizes, measured by its own
    protocol (disease on the supplied cohort, the other two on the generated
    split's fixture) and registered (L7.disease, L7.gene,
    L7.patients_like_me); an fp32 frozen-backbone disease child of the listed
    base on the probe's device, paused after one of two epochs and resumed in
    its own run, its graph side's parameters and buffers checked equal to the
    base's, its downstream parts trained and alone in the optimizer, its
    record kept across the resume and its downstream start the full arm's,
    then measured by the disease protocol and registered (L7.disease_frozen,
    the ops-wiring round's decision F2; AMP and compile with a freeze are not
    covered); a same-settings branch of the listed base started
    through the job runner (`start_training` with `branch_from`, bound to the
    base's digest) and run on the probe's device to its end, listed under its
    parent's exact file with its changes `{}`, the parent's bytes unchanged,
    and loaded by the one loader (L7.branch, the ops-wiring round's
    decisions B1, B2); and the results export (`scripts/export_results.py`)
    over that deployment listing every run L2-L7.disease_frozen kept, with its
    identities, training settings, environment and exposure, no row capped
    (L8). On CUDA
    the measurements and the L7 children run on the device; this container's
    tests run the same phase on the CPU.

M1's minimal served readings (the round's Stage 10b, the owner's item D;
BACKLOG 64), carried by the same run, last in phase M after L8, and
**informational**: S1 is in neither required list, so `m2_acceptance` never
reads it and no reading can leave the acceptance incomplete, and the run's
exit status never counts it (`INFORMATIONAL_PROBES`, `run_exit_status`), so
not even a failed S1 fails the run; nothing it reads is judged, and there is
no threshold. It takes a minimal part of the retired
`scripts/measure_served_pipeline.py`'s reviewed procedure (PLAN_B04 §7.3),
which stays that script's record:
    D1's disease child served by a service of its own, started as the
    launcher starts it (`_ProbeServer`) with the child configured at startup,
    readiness asserted (/ready, and the status's served identity held to the
    probe's own reads, `_assert_served_identity`) before anything counts; a
    service not shown ready fails S1 with what was read before it (the
    readiness record, the cold start) and every later reading not taken. The
    readings: the cold start (launch to the first /health answer, the
    startup's pipeline build included), a warm-up and a steady state of a
    fixed count of serial diagnose requests (one query from the workspace's
    own validation cases, the API's defaults), each counted request timed on
    the probe's clock to its whole answer (on CUDA already synchronised by the server's
    copy of the scores to the host) beside the server's own handler time;
    each failure by class. The server process's own memory (VmRSS, and VmHWM,
    its peak since the start) at readiness and after each phase, and its
    allocator variables, from its PID's /proc entries. The server's CUDA
    allocator peak is recorded as not obtainable, with why: torch keeps it
    inside the server's process and no route exposes it, and a figure taken
    in the probe's process would describe the probe. What the readings do not
    cover is listed with them (`SERVED_READINGS_NOT_COVERED`).
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
from typing import (Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence,
                    Tuple, Union)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Standard library only (no torch), so it is safe before `_apply_allocator`.
from src.utils.pyg_native_check import normalised_name as _normalised  # noqa: E402

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


def _allocator_variables() -> Tuple[str, ...]:
    """The allocator variables a report records, by name: for the probe's own
    process (`_allocator_record`) and for the served readings' server process
    (`_process_allocator`), one list for both."""
    from src.config.runtime_presets import ALLOC_SOURCE_ENV

    return ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", ALLOC_SOURCE_ENV)


def _allocator_record(backend: Optional[str]) -> Dict[str, Any]:
    """The allocator variables in effect (`_allocator_variables`), where they
    came from, and the backend torch reports (None when no CUDA device was
    resolved)."""
    import os

    return {
        **{name: os.environ.get(name) for name in _allocator_variables()},
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


def _assert_served_identity(
    identity: Any,
    *,
    checkpoint_sha256: str,
    workspace_digests: Dict[str, str],
    record: Any,
    task: str,
    device: str,
) -> Dict[str, Any]:
    """What serving says it served is what this probe verified (contract M3c).

    `identity` is the served identity in its JSON form, as
    `get_pipeline_config()["served_identity"]` and the API report it. **Every
    expected value comes from a read this probe already took, or from the core
    tables**: the checkpoint's digest from `_read_run_checkpoint`, the
    `kg.json` and `split_manifest.json` digests from `verify_workspace`'s
    reads, the dataset record that check verified, the device the probe asked
    for, and for the `task` the model was finetuned for, the scoring mode it
    is served under (`TASK_SCORING_MODES`) and that mode's shortest-path term
    (`SCORING_MODE_SP_STATES`), built as the core type states it. Nothing is
    hashed again here. Returns the identity for the report: states, digests,
    identifiers and settings, and no paths.
    """
    from dataclasses import asdict

    from src.core.types import SCORING_MODE_SP_STATES, TASK_SCORING_MODES, ServedShortestPaths
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
    expected_mode = TASK_SCORING_MODES[task]
    expected_paths = asdict(ServedShortestPaths(state=SCORING_MODE_SP_STATES[expected_mode]))
    shortest_paths = identity.get("shortest_paths") or {}
    assert shortest_paths == expected_paths, (
        f"the served identity names a shortest-path term {shortest_paths.get('state')}, not "
        f"the {expected_paths['state']} term the {task} task is served with"
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
    serves: A2 resolves it, A3 records its architectures, A4 launches kernels;
    and A6 requires A1's native build identity of each named component.

    `describe_only` is for runs whose verdict is about something else. A
    download check that launched kernels was failed by a GPU that could not run
    this build, with both ontologies published; A1 alone describes the stack
    and never raises.
    """
    print("\nA. Environment" + (" (description only)" if describe_only else ""))
    facts: Dict[str, Any] = {}

    def _runtime() -> Tuple[str, Dict[str, Any]]:
        from src.utils.run_environment import (
            collect_components,
            native_identity,
            native_identity_text,
        )
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
            # Beside the keys above (research-comparison Stage 2b, BACKLOG 71):
            # each named component as every run's environment records it (the
            # one collector, read once per process), its native identity and,
            # where the source receipt matches what is installed, its source.
            "components": collect_components(),
        }
        facts.update(collected)
        print(f"         Native build: "
              f"{native_identity_text(native_identity(collected['components']))}")
        return "runtime described", collected

    report.run("A1", "environment", "the runtime describes itself", _runtime)

    def _against_lock() -> Tuple[str, Dict[str, Any]]:
        collected = installed_against_lock(project_root / "uv.lock")
        facts["installed_against_lock"] = collected
        # **Stated, never hidden** (M5-3): a homelab whose packages moved away
        # from the committed lock says so on the console too, here and again
        # beside the acceptance's verdict at the end (`environment_verdict`).
        print(f"         {lock_sentence(collected)}")
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

    def _native_identity_and_route() -> Tuple[str, Dict[str, Any]]:
        """**A1's components, required** (research comparison Stage 10; the
        owner's O9): each named component's facts in the one shape every run's
        environment records them in (`component_facts`), its native identity
        as the record views read it (`native_identity`, `RecordComponent`),
        and its route only where the collector recorded its source, which it
        does only when the source receipt matched what is installed; otherwise
        the route reads "not recorded". **Never a gate on versions**: no
        version, digest or route is compared with anything, so another build
        passes and is told apart by its identity."""
        from src.api.services.record_views import RecordComponent
        from src.utils.run_environment import (
            SOURCE_NOT_RECORDED,
            SOURCE_RECORDED,
            component_facts,
            native_identity,
            native_identity_text,
        )

        a1 = next((probe for probe in report.probes if probe.probe_id == "A1"), None)
        assert a1 is not None and a1.status == "passed", "A1 did not describe the runtime"
        components = a1.facts.get("components")
        assert isinstance(components, dict) and components, "A1 records no named component"
        identity = native_identity(components)
        routes: Dict[str, str] = {}
        for name in sorted(components):
            entry = components[name]
            try:
                held = component_facts(entry["kind"], **{key: value for key, value
                                                         in entry.items() if key != "kind"})
                RecordComponent.model_validate(identity[name])
            except (KeyError, TypeError, ValueError):
                raise AssertionError(f"the component {name}'s facts are not in the shape every "
                                     "run's environment records") from None
            assert held == entry, (
                f"the component {name}'s facts change when held to the one shape")
            source = entry["source"]
            routes[name] = (source["route"] if source["state"] == SOURCE_RECORDED
                            else SOURCE_NOT_RECORDED)
        return "each component's native identity, its route only where the receipt matched", {
            "components": len(components),
            "native_identity": identity,
            "routes": routes,
            "line": native_identity_text(identity),
        }

    report.run("A6", "environment",
               "each named component's native build identity and route are recorded, never "
               "compared with a version", _native_identity_and_route)
    return facts


#: What a Python environment records of each installed distribution and what
#: `uv.lock` pins are compared by normalised name (PEP 503): `_normalised`, the
#: install identity's one implementation, imported at the top.


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


def _editable_install_of(distribution: Any, root: Path) -> bool:
    """Whether a distribution is an editable install of the checkout at `root`,
    by its ``direct_url.json`` (PEP 610: ``dir_info.editable`` and a file URL
    naming that directory; `url2pathname` reads a Windows drive letter too)."""
    from urllib.parse import urlparse
    from urllib.request import url2pathname

    try:
        text = distribution.read_text("direct_url.json")
        data = json.loads(text) if text else None
    except (AttributeError, OSError, ValueError):
        return False
    if not isinstance(data, dict) or not isinstance(data.get("dir_info"), dict):
        return False
    url = data.get("url")
    if data["dir_info"].get("editable") is not True or not isinstance(url, str) \
            or not url.startswith("file://"):
        return False
    return Path(url2pathname(urlparse(url).path)).resolve() == root.resolve()


def _editable_paths(distribution: Any, root: Path) -> set:
    """The directories an editable install puts on the path: the lines of the
    ``.pth`` files its RECORD lists (``__editable__.<name>.pth``), or, when none
    can be read, the checkout's ``src`` (this project's layout)."""
    paths = set()
    for entry in getattr(distribution, "files", None) or ():
        if not str(entry).endswith(".pth"):
            continue
        try:
            text = Path(distribution.locate_file(entry)).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, TypeError, ValueError):
            continue
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "import ", "import\t")):
                paths.add(Path(line).resolve())
    return paths or {(root / "src").resolve()}


def _in_source_tree(distribution: Any, places: set) -> bool:
    """Whether a distribution's metadata is the project's own build output left
    where its editable install puts the project on the path (`places`,
    `_editable_paths`), with no ``direct_url.json``: the ``src/<name>.egg-info``
    an editable build leaves. A copy anywhere else (a nested worktree, a build
    directory) is not this, and stays reported."""
    try:
        where = Path(distribution.locate_file("")).resolve()
        installed_record = distribution.read_text("direct_url.json")
    except (AttributeError, OSError, TypeError):
        return False
    return not installed_record and where in places


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
      the metadata, so both are stated. **One exception, itself stated in
      ``counted_once``** (BACKLOG 59): this project's own metadata where its
      editable install puts it on the path (the ``.egg-info`` an editable build
      leaves in ``src/``, `_in_source_tree`, `_editable_paths`) beside that
      editable install of the same checkout (`_editable_install_of`), at the
      same version, and nothing else of the name, is one installation, not two;
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
    this_project = {_normalised(str(package["name"])) for package in lock.get("package", [])
                    if isinstance(package, dict) and package.get("name")
                    and _is_this_project(package)}
    installed: Dict[str, List[str]] = {}
    requires: Dict[str, set] = {}
    found: Dict[str, List[Any]] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        if not name:
            continue
        key = _normalised(name)
        installed.setdefault(key, []).append(str(distribution.version))
        found.setdefault(key, []).append(distribution)
        requires.setdefault(key, set()).update(
            _required_names(getattr(distribution, "requires", None)))
    # The project's own source-tree metadata beside its editable install of
    # this checkout is one installation (BACKLOG 59); said, never hidden.
    counted_once = []
    for key in sorted(this_project & set(found)):
        copies = found[key]
        editable = [d for d in copies if _editable_install_of(d, lock_path.parent)]
        # Exactly two: the editable install and one copy where it puts the
        # project on the path; a third copy anywhere is a real duplicate.
        if (len(copies) == 2 and len(editable) == 1 and len(set(installed[key])) == 1
                and all(d in editable or _in_source_tree(
                    d, _editable_paths(editable[0], lock_path.parent)) for d in copies)):
            installed[key] = [str(editable[0].version)]
            counted_once.append({"name": key, "version": installed[key][0],
                                 "why": "its editable install and the metadata its build "
                                        "left in the checkout's source tree"})
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
        "counted_once": counted_once,
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
class DemoModels(NamedTuple):
    """What phase D trained on the demo workspace: the disease child, which
    phases E, M, T and W serve, measure and list, and its base, which M2
    finetunes again (a base serves no diagnosis and measures no task)."""

    child: Path
    base: Path


def phase_training(
    report: Report, work: Path, sound: Path, device: str, epochs: int
) -> Optional[DemoModels]:
    print("\nD. Training, through the production entry point")
    trained: Dict[str, Any] = {}

    def _train() -> Tuple[str, Dict[str, Any]]:
        from scripts.setup_demo import demo_task_training
        from src.utils.checkpoint_paths import propose_run_dir

        # **Named, not found afterwards**: each run's directory under the
        # workspace's runs (contract M3a-2), filed under its model spec's
        # encoder kind (`_demo_encoder_kind`), so the checkpoints read below
        # are these runs' and no other's.
        sizes = demo_task_training()
        encoder = _demo_encoder_kind()
        base_run = propose_run_dir(sound, encoder, "probe-base")
        models = _train_disease_child(
            sound, propose_run_dir(sound, encoder, "probe-disease"), device,
            base_run=base_run, sizes=sizes, epochs=epochs)
        metrics = models.metrics
        assert metrics, "training returned no metrics"
        trained["models"] = DemoModels(child=models.child, base=models.base)

        import math as _math

        finite = {
            key: value for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        nonfinite = sorted(k for k, v in finite.items() if not _math.isfinite(float(v)))
        assert not nonfinite, f"training reported non-finite metrics: {nonfinite}"
        return "the production trainer produced a base and a disease child of it", {
            "epochs": epochs,
            "metric_names": sorted(finite),
            "checkpoint_name": models.child.name,
            "encoder_kind": encoder,
            "training_seconds": models.seconds,
        }

    done = report.run(
        "D1", "training", "the production trainer runs here", _train
    )
    if done.status != "passed":
        return None
    models: DemoModels = trained["models"]

    def _provenance() -> Tuple[str, Dict[str, Any]]:
        from src.kg.artifacts import GRAPH_ARTIFACTS
        from src.models.bundle.record import check_model_record, read_model_record
        from src.utils.checkpoint_io import read_checkpoint
        from src.utils.fingerprint import verify_training_record

        checkpoint = read_checkpoint(
            models.child, map_location="cpu", weights_only=True
        ).checkpoint
        digests, _, _, _ = verify_workspace(sound)
        # **Through the consumers' own comparator** (contract M2.4), not a
        # second comparison here: the four graph roles, `kg` the manifest-bound
        # source graph, present on both sides and equal, or it raises.
        verify_training_record(
            checkpoint,
            digests,
            tuple(GRAPH_ARTIFACTS),
            recorded_by=f"checkpoint {models.child.name}",
            compared_with="the workspace it was trained on",
        )
        # **And its model record, through the one reader** (contract §9.1): a
        # disease finetune whose parent is the base D1 trained, by the digest
        # of that base's bytes.
        record = read_model_record(checkpoint)
        check_model_record(record, learning_stage="finetune", task="disease")
        assert record.operation == "finetune", f"the child's operation is {record.operation}"
        base_sha256 = _read_run_checkpoint(models.base)["sha256"]
        assert (record.parent or {}).get("sha256") == base_sha256, (
            "the child's model record names another parent than the base D1 trained")
        return "the child names the workspace it was trained on and the base it came from", {
            "recorded_roles": sorted(checkpoint["training_input_digests"]),
            "learning_stage": record.learning_stage, "task": record.task,
            "operation": record.operation, "parent_run": record.parent["run"],
            "parent_sha256": base_sha256,
        }

    report.run(
        "D2", "training",
        "the child carries the digests of the workspace it read and names its base",
        _provenance,
    )
    return models


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
        # M3c): the child D1 trained, by its digest; the workspace's graph and
        # manifest, by the digests the workspace check took; its dataset
        # record; the disease task's scoring mode and shortest-path term; and
        # the device asked for, in effect.
        _, _, read, record = verify_workspace(sound)
        identity = _assert_served_identity(
            config.get("served_identity"),
            checkpoint_sha256=_read_run_checkpoint(checkpoint)["sha256"],
            workspace_digests=read,
            record=record,
            task="disease",
            device=device,
        )
        return "the pipeline built, the GNN is ready, and what it serves is identified", {
            "scoring_mode": identity["scoring_mode"],
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
        # **Selected by its task**, as the dashboard's Load names one: the
        # disease child among the workspace's runs, never the base beside it.
        result = asyncio.run(reload_pipeline(PipelineReloadRequest(
            data_dir=str(sound), task="disease", device=device)))
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
        assert facts["scoring_mode"] == "disease_task", (
            f"the reload by task serves {facts['scoring_mode']}, not the disease task")
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

        result = asyncio.run(reload_pipeline(PipelineReloadRequest(
            data_dir=str(broken), task="disease", device=device)))
        assert not result.success, "a mixed workspace was accepted"
        # **Refused for the replaced file**, not for a model it could not find.
        assert "node_features" in (result.message or ""), (
            "the refusal does not name the replaced node features")
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
        result = asyncio.run(reload_pipeline(PipelineReloadRequest(
            data_dir=str(stripped), task="disease", device=device)))
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
        pipeline = state["bundle"].pipeline
        result, scores = _disease_diagnosis(pipeline, DEMO_QUERY)
        # **The result says what produced it** (contract M3c): the identity it
        # carries, built when it was produced, is the one E1 verified.
        assert result.served_identity.to_dict() == state["bundle"].config["served_identity"], (
            "the diagnosis carries another served identity than the pipeline it ran on"
        )
        return "a diagnosis ran end to end", {
            "candidates": len(scores),
            "result_kind": result.result_kind,
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
        # The model bundle's full-view vectors, computed once when it was
        # built and read by every request (a private read, as E3's helpers).
        vectors = state["bundle"].pipeline._bundle_z
        assert vectors is not None, "the served model computed no vectors"
        return f"every parameter is on {device}", {
            "parameter_devices": sorted(devices),
            "embedding_devices": [str(vectors.device).split(":")[0]],
        }

    report.run(
        "E8", "serving",
        "the served model's parameters are on the requested device",
        _model_is_where_it_was_asked_to_be,
    )


# =============================================================================
# Phase M — contract M2's acceptance on the demo workspace (--m2-acceptance)
# =============================================================================
#: The mixed precision M2's paused run states beside the demo's sizes (Q13
#: (b)), on every device; it takes effect on CUDA only (contract M3a-1). Its
#: resume and M9's extension inherit it, and the three are judged against it
#: (`_scaler_restore_facts`), as `m2_acceptance`'s scope says.
M2_PRECISION: Dict[str, Any] = {"use_amp": True, "amp_dtype": "float16"}
#: The mixed precision R3's paused base states beside the reference recipe:
#: none, so the base recipe's (Q13 (b): the real part keeps the reference
#: recipe's fp32); an empty mapping means the recipe's. R3 is given it, its
#: resume (R4) inherits it, R4 is judged against the request built from it
#: (`_scaler_restore_facts`), and `m2_acceptance`'s scope says it
#: (`_precision_scope`).
R3_PRECISION: Dict[str, Any] = {}

#: Every probe contract M2's acceptance requires, by part (the plan's "Acceptance
#: for M2"). A required probe that is missing from the report, skipped or not
#: passed leaves its part incomplete; nothing is inferred from a probe not run.
M2_DEMO_REQUIRED: Tuple[str, ...] = (
    "A2", "B1", "C1", "D1", "D2", "E1", "E7", "E8",
    "M1", "M2", "M3", "M4",
    "M5.pre_contract", "M5.other_graph", "M5.other_samples",
    "M5.validated_parent", "M5.missing", "M5.no_model_spec",
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
    # Contract M4's flow on the same workspace and D1's disease child,
    # functional only: import preview and commit (T1); measure by the disease
    # protocol, keep and record (T2); the same report again and a damaged copy (T3); a
    # version mapped against M1's graph (T4); the read-only view for the
    # served child and another (T5); the overlap audit on the registered
    # version (T6).
    "T1", "T2", "T3", "T4", "T5", "T6",
    # M5's acceptance (M5-3): the environment against the lock (A5);
    # the server's lists (W1); D8 on the running service, CORS, the route
    # guard, read-only without M1 (W2-W5) and off loopback (W6); the GPU lock
    # (W7); a test run through the job runner, kept and registered as the
    # publication contract says, and registration from the store only (W8,
    # W9); the start's provenance check (W10); a training job's exclusion and
    # stop (W11).
    "A5", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10", "W11",
    # The research comparison round's (its plan's §5 Stage 10, the owner's
    # O9), functional only: each named component's native identity and route
    # (A6); a KG edge set imported from D1's base and measured by the KG
    # link-prediction protocol through measure_scorer and the job runner,
    # kept and registered (L1-L3); the refusals before anything is written
    # (L4); a disease child by both of its protocols (L5); a generated split
    # imported, measured and registered (L6); a short disease, gene and
    # patients-like-me child each on the probe's device, measured and
    # registered (L7, the GPU regression cases); and the results export (L8).
    # The ops-wiring round's, in the same flow: an fp32 frozen-backbone disease
    # child of the demo base, paused and resumed (L7.disease_frozen, decision
    # F2), and a same-settings branch of the demo base through the job runner
    # (L7.branch, decisions B1, B2).
    "A6", "L1", "L2", "L3",
    "L4.other_graph", "L4.tampered_pairs", "L4.other_view", "L4.split_record",
    "L5", "L6", "L7.disease", "L7.gene", "L7.patients_like_me", "L7.disease_frozen",
    "L7.branch", "L8",
)
#: R8, the unchanged-workspace check, runs last, after M4's RT1-RT3 and M5's
#: RW1-RW3; R2 runs after R4.finetune, whose child it strips.
M2_REAL_REQUIRED: Tuple[str, ...] = ("R1", "R2", "R3", "R4", "R4.finetune", "R5", "R6", "R7",
                                     "R8", "RT1", "RT2", "RT3", "RW1", "RW2", "RW3")


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
            "points, on the model-bundle flow: a base trained, paused and resumed, "
            "a disease child finetuned from a base, paused and resumed and "
            "extended, the child measured by the disease protocol and served for the "
            "disease task. The real-data models are short trainings by the reference "
            "recipe that show the pipeline and its checks work; they are not "
            "models of serving quality, which is chosen by the established "
            "training and evaluation. " + _architecture_scope() + " "
            + _precision_scope() + " The run "
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
            "the dashboard's layout or a browser session, and no long training. It "
            "carries the research comparison round's probes too (A6, L1-L8), on the demo "
            "workspace: each named component's native build identity and route, recorded "
            "and never compared with a version; a KG edge set taken from the demo base, "
            "measured by the KG link-prediction protocol through the measurement command "
            "and the job runner, its exposure, scores, pairs and summary checked against "
            "each other, and its refusals; a disease child measured by two protocols; a "
            "generated split imported, measured and registered; short disease, gene and "
            "patients-like-me children trained and measured on this device as regression "
            "cases, not studies; a frozen-backbone disease child of the demo base in fp32, "
            "paused and resumed, its graph side checked unchanged and its downstream parts "
            "trained (L7.disease_frozen; mixed precision and torch.compile with a freeze are "
            "not covered); a same-settings branch of the demo base started through the "
            "job runner, listed under its parent with no change (L7.branch); and the results "
            "export listing every record of that flow. "
            "Its figures are a functional check, never a comparison of models. Last, it "
            "records M1's served readings of the demo disease child (S1): the service's cold "
            "start, warm-up and steady-state request times and its process's memory, "
            "informational and never required, so no reading, however slow or large, leaves "
            "the acceptance incomplete, and nothing it reads is judged. Nor does its failure "
            "set the run's exit status."
        ),
    }


def _demo_encoder_kind() -> str:
    """The encoder kind of the demo's model spec (`setup_demo.demo_task_training`),
    which D1's base and child and every child finetuned from that base (M2's,
    and W1's copy of it) are built with: the run-layout key their runs are
    filed under, as `setup_demo.train_demo_model` files the demo's (K2: a
    bundle run is filed under its encoder kind)."""
    from scripts.setup_demo import demo_task_training
    from src.models.bundle.spec import ModelSpec

    return ModelSpec.from_dict(demo_task_training()["base"]["model_spec"]).encoder.kind


def _architecture_scope() -> str:
    """The encoder kind the acceptance covers, in words, read from the specs the
    probe trains and the kinds this build declares (never a literal): the demo
    part's from the demo's model spec (`_demo_encoder_kind`), the real part's
    from the reference recipe's base spec, which R3 trains as it states no
    model spec, and the declared kinds from `spec.ENCODER_DECLARATIONS`, which
    the builder's implementations equal. A second architecture is chosen
    after the ops-wiring round's fit assessment
    (docs/working/ops-wiring-and-ui-ux/README.md §10, C0) and proves its own
    devices and precisions then (Q13 (b))."""
    from src.models.bundle.spec import ENCODER_DECLARATIONS, reference_spec

    demo, real = _demo_encoder_kind(), reference_spec("base").encoder.kind
    declared = sorted(ENCODER_DECLARATIONS)
    listed = declared[0] if len(declared) == 1 else (
        f"{', '.join(declared[:-1])} and {declared[-1]}")
    return (f"Architecture, as the runs are built: the demo part trains the {demo} encoder "
            f"(the demo's model spec) and the real part the {real} encoder (the reference "
            "recipe's base spec, which R3 trains). Other encoder kinds are refused by the "
            "model-bundle builder and are outside this acceptance, their devices and "
            f"precisions included: this build declares and builds {listed} only, and a second "
            "architecture is chosen after the fit assessment of the ops-wiring round "
            "(docs/working/ops-wiring-and-ui-ux/README.md §10).")


def _precision_scope() -> str:
    """The precision the acceptance promises, in words, from the requests the
    probe makes (never a second statement of them): the demo part's from M2's
    (`M2_PRECISION`), the real part's from R3's (`R3_PRECISION`), which states
    none and so asks for the base recipe's. The acceptance runs on CUDA."""
    demo = _precision_request(M2_PRECISION, "cuda", "finetune", "disease")
    real = _precision_request(R3_PRECISION, "cuda", "base", None)

    def said(request: Mapping[str, Any]) -> str:
        if request["use_amp"] and request["amp_dtype"] == "float16":
            return ("mixed precision float16, as stated, and pass only with float16 in effect "
                    "and a gradient scaler restored from the parent's state")
        if request["use_amp"]:
            return f"mixed precision {request['amp_dtype']}, as stated, with no gradient scaler"
        return (f"fp32, as the {request['source']} asks, with no mixed-precision claim")

    return (f"Precision, as the runs ask for it: the demo part's M2 and M9 run in "
            f"{said(demo)}; the real part's R3 and R4 run in {said(real)}.")


def _native_verdict(report: Report) -> Dict[str, Any]:
    """A1's native build identity in short: each component's identity and
    route (`native_identity`: its route only where the source receipt matched
    what is installed, otherwise its source reads "not recorded"), or None
    when A1 did not pass."""
    from src.utils.run_environment import native_identity

    a1 = next((probe for probe in report.probes if probe.probe_id == "A1"), None)
    passed = a1 is not None and a1.status == "passed"
    return {"A1": a1.status if a1 is not None else "not run",
            "native_identity": native_identity(a1.facts.get("components")) if passed else None}


def environment_verdict(report: Report) -> Dict[str, Any]:
    """A5's record in short, for the end of a run beside its verdict (the
    report's `m2_acceptance` and the console's last lines): whether the lock
    read is the committed one, and the names of what differs from it. A set
    that differs is said again where the run ends, not only an hour earlier.
    Beside it, A1's native build identity (`_native_verdict`)."""
    native = _native_verdict(report)
    a5 = next((probe for probe in report.probes if probe.probe_id == "A5"), None)
    if a5 is None or a5.status != "passed":
        return {"A5": a5.status if a5 is not None else "not run", **native}
    facts = a5.facts
    outside = facts.get("installed_not_in_lock") or []
    return {
        **native,
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


#: The fields that say where a run reads and writes: given to every run, and
#: never part of the settings it states (`_train_run`, `_train_disease_child`).
_LOCATION_FIELDS = frozenset({"data_dir", "run_dir"})


def _stated(fields: Mapping[str, Any]) -> set:
    """The settings a run states: every field given a value, but its location
    fields. **Passed to the training entry as the explicit set** (Q13 (b), the
    reviewer's P2-1 (a)), so a setting stated at TrainConfig's default value
    (`use_amp=True`, `batch_size=32`) counts as stated and is never filled
    from the stage recipe in its place."""
    return {name for name, value in fields.items() if value is not None} - _LOCATION_FIELDS


def _train_run(
    data_dir: Path, run: Path, device: str, *, stage: str, task: Optional[str] = None,
    epochs: Optional[int] = None, settings: Optional[Mapping[str, Any]] = None,
    finetune_from: Optional[Path] = None, resume_from: Optional[Path] = None,
    pause_after: Optional[int] = None, extend_epochs: Optional[int] = None,
    observe: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """One model-bundle run of the production trainer (`train_model.train`)
    with `run` as its run directory (contract M3a-2), so every output -- its
    checkpoints, logs and records -- is inside it and nothing lands in the
    workspace or anywhere else. Returns the run's checkpoint directory.

    Every run states its learning stage, and a finetune its task, as the
    training service does (`TrainingManager.start_training`): the training
    entry refuses a stage-less bundle run, a finetune without its task, and a
    continuation of another stage or task than its parent's. A new run is a
    base, or a finetune of `finetune_from`, with `epochs` and `settings` (the
    stage recipe fills the rest and records which). **A run continued from a
    parent is given operational fields only** (contract M3a-3): it inherits
    every learning setting from the parent's record, the number of epochs
    included, and an explicit one would be refused if it differed. Either way
    the fields given are passed as the explicit set (`_stated`), never a bare
    TrainConfig whose stated defaults the entry would take for unset.

    `observe`, for a continued run, receives what the restore left
    (`_observing_restore`)."""
    from scripts.train_model import TrainConfig, train
    from src.utils.checkpoint_paths import run_checkpoint_dir

    fields: Dict[str, Any] = dict(data_dir=str(data_dir), run_dir=str(run), device=device,
                                  learning_stage=stage, task=task,
                                  pause_after_epoch=pause_after)
    if resume_from is not None:
        assert not (settings or epochs or finetune_from), (
            "a continued run inherits its settings and epochs from its parent")
        fields.update(resume_from=str(resume_from), extend_epochs=extend_epochs)
        given = {name: value for name, value in fields.items() if value is not None}
        with _observing_restore(observe):
            train(TrainConfig(**given), _stated(given))
    else:
        fields.update(settings or {})
        fields.update(num_epochs=epochs,
                      finetune_from=str(finetune_from) if finetune_from else None)
        given = {name: value for name, value in fields.items() if value is not None}
        train(TrainConfig(**given), _stated(given))
    return run_checkpoint_dir(run)


class TrainedChild(NamedTuple):
    """A disease child `_train_disease_child` trained, its base, the seconds
    each run took, and the metrics the child's run returned."""

    child: Path
    base: Path
    seconds: Dict[str, float]
    metrics: Dict[str, float]


def _train_disease_child(
    data_dir: Path, child_run: Path, device: str, *,
    base_run: Optional[Path] = None, base: Optional[Path] = None,
    sizes: Optional[Mapping[str, Mapping[str, Any]]] = None, epochs: int = 1,
) -> TrainedChild:
    """A disease child of a model-bundle base, by the production trainer
    (`train_model.train`, the model bundle path): a new base trained on
    `data_dir` into `base_run`, or the existing base `base` (its last.pt),
    then one disease child of it into `child_run` (`learning_stage`
    "finetune", task "disease", `finetune_from` the base), `epochs` each on
    `device`. D1 trains its base and child here, R4.finetune and RW2 their
    children from R4's resumed base.

    The stage's reference recipe fills every setting not given here and
    records which it filled (`apply_stage_recipe`); `sizes` (by "base" and
    "disease") replaces some for a smaller run (the demo graph's,
    `scripts/setup_demo.demo_task_training`, which `setup_demo.py
    --train-model` trains at), and with none the runs are the reference
    recipe's, the epochs aside. Every setting given counts as the operator's,
    as a configuration file's keys do (`_stated`). Each run writes only into
    its run directory (contract M3a-2)."""
    from scripts.train_model import TrainConfig, train
    from src.utils.checkpoint_paths import run_checkpoint_dir

    assert (base_run is None) != (base is None), "a child is trained from one base"
    sizes = sizes or {}
    seconds: Dict[str, float] = {}

    def run(stage: str, run_dir: Path, **fields: Any) -> Tuple[Path, Dict[str, float]]:
        fields = {**(sizes.get(stage) or {}), **fields, "data_dir": str(data_dir),
                  "run_dir": str(run_dir), "num_epochs": epochs, "device": device}
        started = time.monotonic()
        metrics = train(TrainConfig(**fields), _stated(fields))
        seconds[stage] = round(time.monotonic() - started, 1)
        return _last(run_checkpoint_dir(run_dir)), metrics

    if base is None:
        base, _ = run("base", base_run, learning_stage="base")
    child, metrics = run("disease", child_run, learning_stage="finetune", task="disease",
                         finetune_from=str(base))
    return TrainedChild(child=child, base=base, seconds=seconds, metrics=metrics)


def _without_model_spec(source: Path, target: Path) -> Path:
    """A copy of the checkpoint `source` with its model spec removed, written to
    `target`: a file written before the model spec, as a ShepherdGNN history
    file is, made from a file this run trained. The one helper M5.no_model_spec
    and R2 write their subject with; nothing else of the payload changes."""
    import torch

    from src.utils.checkpoint_io import read_checkpoint

    contents = read_checkpoint(source, map_location="cpu", weights_only=True).checkpoint
    del contents["model_spec"]
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(contents, target)
    return target


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
    config = checkpoint.get("config") or {}
    return {
        "sha256": read.identity.sha256,
        "record": dict(checkpoint.get("training_input_digests") or {}),
        "optimizer_step": max(steps) if steps else None,
        "weights": checkpoint.get("state_dict") or {},
        "progress": checkpoint.get("progress") or {},
        "rates": [float(g["lr"]) for g in
                  (checkpoint.get("optimizer_state_dict") or {}).get("param_groups", [])],
        # Mixed precision as it was asked for and as it took effect, and the
        # gradient scaler's state (plain numbers), so a pause and resume under
        # AMP, which only CUDA runs, is shown in the report (contract M3a).
        "amp": (config.get("effective") or {}).get("amp") or {},
        "scaler": _scaler_numbers(checkpoint.get("scaler_state_dict")),
        "loaders": (config.get("effective") or {}).get("loaders") or {},
        # The settings the run used, as recorded (`config.requested`).
        "requested": config.get("requested") or {},
        # Its last epoch's metrics, as the checkpoint written after it holds them.
        "logs": checkpoint.get("logs") or {},
    }


def _assert_resumed(parent: Path, child: Path, observed: List[Dict[str, Any]], *,
                    requested: Mapping[str, Any]) -> Dict[str, Any]:
    """A resume that trained, not one that only loaded: the child names its
    parent's bytes and the same workspace roles, its optimizer went on stepping
    from the parent's state, its weights moved, and its mixed precision is the
    one `requested` (`_precision_request`) asked for, its gradient scaler
    restored when it has one (`observed`, from `_observing_restore`)."""
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
    amp = _scaler_restore_facts(before, after, observed, requested=requested)
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
    """The child ran only the epoch its parent had left (contract M3a-3).

    **Counted by optimizer attempts, not successful updates.** AMP's gradient
    scaler skips an update on non-finite gradients, and two epochs need not
    skip alike: at the GB10 acceptance of `4b1571e` the parent's epoch made 8
    attempts and 7 updates, the remaining epoch 8 and 8, and a check that
    required twice the parent's updates refused a correct resume.

    **An epoch takes `ceil(learnable batches / accumulation)` attempts.** The
    batches per epoch are the parent's train loader's, as the run recorded
    them (`config.effective.loaders.train.batches`, which every model-bundle
    run writes); a schedule records its own count only when a scheduler is
    built, and where it does the two must agree (NEW-probe-2: the recipes'
    scheduler is "none"). A model-bundle run takes no attempt for a batch its
    task finds nothing to learn from, and counts those batches in the epoch's
    metrics (`train_steps_skipped`), which the checkpoint written after the
    epoch holds; so the learnable batches are the loader's less the child's
    skipped ones. The record holds them for its last epoch only, so the count
    is exact for one remaining epoch, which is what M2 resumes (two planned,
    paused after one), and a resume of more fails here as one its record
    cannot count. The accumulation is the schedule's where it records one,
    else the parent's recorded setting. The child's successful updates must
    have grown, by no more than its attempts, and its scheduler steps by its
    successful updates (the schedule steps only when the optimizer ran).
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
    batches = ((before.get("loaders") or {}).get("train") or {}).get("batches")
    assert batches, "the parent's train loader records no batches per epoch"
    if schedule.get("steps_per_epoch") is not None:
        assert int(schedule["steps_per_epoch"]) == int(batches), (
            "the parent's schedule and train loader record different batches per epoch")
    accumulation = schedule.get("gradient_accumulation_steps",
                                (before.get("requested") or {}).get(
                                    "gradient_accumulation_steps"))
    assert accumulation, "the parent records no gradient accumulation"
    logs = after.get("logs") or {}
    assert "train_steps_skipped" in logs, (
        "the resumed run's epoch records no count of the batches it skipped")
    assert epochs_run == 1, (
        f"the resumed run ran {epochs_run} epochs; its record counts the skipped batches of "
        "its last epoch only, so only a resume of one remaining epoch is counted")
    learnable = int(batches) - int(logs["train_steps_skipped"])
    per_epoch = -(-learnable // int(accumulation))
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
            "batches": int(batches), "batches_skipped": int(logs["train_steps_skipped"]),
            "child_run_attempts": attempts, "child_run_updates": updates,
            "child_run_skipped_updates": attempts - updates,
            "child_run_scheduler_steps": steps}


def _precision_request(settings: Mapping[str, Any], device: str, stage: str,
                       task: Optional[str]) -> Dict[str, Any]:
    """The mixed precision a new run asks for, which it and every run continued
    from it are judged against (`_scaler_restore_facts`; a continuation
    inherits its parent's): `use_amp`, and `amp_dtype`, as `settings` state
    them, else the stage recipe's `use_amp` (`src.config.recipe`, which the
    trainer fills in their place), on `device`. A run that asks for mixed
    precision states its dtype."""
    from src.config.recipe import reference_settings

    if "use_amp" in settings:
        use_amp, source = bool(settings["use_amp"]), "stated setting"
    else:
        use_amp, source = bool(reference_settings(stage, task)[0]["use_amp"]), f"{stage} recipe"
    amp_dtype = settings.get("amp_dtype")
    assert not use_amp or amp_dtype, "a run that asks for mixed precision states its dtype"
    return {"use_amp": use_amp, "amp_dtype": amp_dtype, "device": device, "source": source}


#: The keys of a recorded mixed precision (`Trainer.amp_record`), each read,
#: none defaulted; `dtype_requested` is read when the request states a dtype.
_AMP_KEYS = ("requested", "enabled", "dtype", "grad_scaler")


def _precision_in_effect(amp: Mapping[str, Any], requested: Mapping[str, Any],
                         side: str) -> None:
    """One side's recorded mixed precision against what the run asked for:
    every key present, the request recorded as asked, and the effect the
    request and the device decide (contract M3a-1: enabled exactly when AMP is
    asked for on CUDA, in the dtype asked for, else float32). float16 in
    effect requires a gradient scaler."""
    keys = _AMP_KEYS + (("dtype_requested",) if requested["amp_dtype"] else ())
    missing = [key for key in keys if key not in amp]
    assert not missing, f"the {side}'s mixed precision record has no {', '.join(missing)}"
    assert amp["requested"] is requested["use_amp"], (
        f"the {side} records mixed precision requested {amp['requested']}; the run asked for "
        f"{requested['use_amp']} ({requested['source']})")
    if requested["amp_dtype"]:
        assert amp["dtype_requested"] == requested["amp_dtype"], (
            f"the {side} records {amp['dtype_requested']} requested; the run asked for "
            f"{requested['amp_dtype']}")
    enabled = bool(requested["use_amp"]) and str(requested["device"]).split(":")[0] == "cuda"
    dtype = requested["amp_dtype"] if enabled else "float32"
    assert (amp["enabled"], amp["dtype"]) == (enabled, dtype), (
        f"the {side} records mixed precision {'on' if amp['enabled'] else 'off'} in "
        f"{amp['dtype']}, which the run did not ask for: {'on' if enabled else 'off'} in "
        f"{dtype} for use_amp {requested['use_amp']} on {requested['device']}")
    if amp["enabled"] and amp["dtype"] == "float16":
        assert amp["grad_scaler"] is True, (
            f"the {side} ran float16 without a gradient scaler")


def _scaler_restore_facts(before: Dict[str, Any], after: Dict[str, Any],
                          observed: List[Dict[str, Any]], *,
                          requested: Mapping[str, Any]) -> Dict[str, Any]:
    """Mixed precision took effect in the child as in its parent, as the run
    asked for it, and the gradient scaler was restored from the parent's state
    (contract M3a-3).

    **Judged against the request** (`requested`, `_precision_request`: what the
    run that started the line stated, or its stage recipe's): each side's
    record must hold every key, show the request, and show the effect the
    request and the device decide; float16 in effect requires a scaler
    (`_precision_in_effect`). So a run asked for float16 on CUDA that ran
    without a scaler, or with AMP off, fails, and so does an fp32 request
    whose record shows AMP in effect.

    **The restore is judged at the restore** (`_observing_restore`) whenever a
    scaler is recorded: the scaler's state right after `Trainer.continue_from`,
    before the first update, must equal the parent's from the same read,
    which must be the parent file's. The scale and growth tracker at the two
    ends are recorded, not judged: equal or not, they can show neither a
    carried nor a restarted scaler. "No gradient scaler" is said only when
    float16 is not in effect, with the reason: the CPU, an fp32 request, or
    bfloat16.
    """
    for side, amp in (("parent", before["amp"]), ("child", after["amp"])):
        _precision_in_effect(amp, requested, side)
    assert after["amp"] == before["amp"], (
        f"the resumed run's mixed precision {after['amp']} is not its parent's {before['amp']}")
    assert len(observed) == 1, f"the continued run's restore was observed {len(observed)} times"
    restore = observed[0]
    amp = before["amp"]
    facts: Dict[str, Any] = {"amp_requested": amp["requested"], "amp_enabled": amp["enabled"],
                             "amp_dtype": amp["dtype"], "grad_scaler": amp["grad_scaler"],
                             "precision_asked_by": requested["source"]}
    if not amp["grad_scaler"]:
        assert restore["parent"] is None and restore["restored"] is None, (
            f"a run without a gradient scaler restored one: {restore}")
        reason = ("bfloat16" if amp["dtype"] == "bfloat16"
                  else f"fp32 {requested['source'].split()[-1]}" if not requested["use_amp"]
                  else str(requested["device"]).split(":")[0].upper())
        facts["scaler_restore"] = f"no gradient scaler ({reason})"
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


def _measure(checkpoint: Path, data_dir: Path, output: Path, device: str, *,
             protocol: str) -> int:
    """The production measurement entry point on the generated `val` cohort,
    measuring the model by the evaluation `protocol` asked (`--mode`)."""
    import scripts.measure_scorer as measure_scorer

    return measure_scorer.main([
        "--mode", protocol, "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
        "--split", "val",
        "--output", str(output), "--device", device,
    ])


def _measurement_facts(output: Path, checkpoint_sha: str, kg_digest: str,
                       record: Any, *, protocol: str) -> Dict[str, Any]:
    """`record` is the dataset record the workspace check verified: the
    measurement manifest must carry it whole, sources and digest included, and
    name as its mode the evaluation `protocol` asked."""
    from src.utils.fingerprint import file_sha256

    assert output.is_file(), "the measurement wrote no report"
    result = json.loads(output.read_text())
    assert result["manifest"].get("mode") == protocol, (
        f"the report's mode is {result['manifest'].get('mode')!r}, not the protocol asked "
        f"({protocol})")
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
        "mode": protocol,
        "dataset_record": {"binding": recorded["binding"], "origin": recorded["origin"],
                           "sha256": recorded["sha256"], "sources": len(recorded["sources"])},
        "n_ranked": result.get("n_ranked"),
        "authoritative_metrics": result.get("authoritative_metrics"),
    }


#: The demo graph's two phenotypes E7 diagnoses on: the demo phase's fixed
#: query. A check that judges a workspace takes its query from that
#: workspace's own data (`_workspace_query`), never from here.
DEMO_QUERY = ("HP:0001250", "HP:0001263")


def _disease_diagnosis(pipeline: Any, phenotypes: Sequence[str],
                       top_k: int = 5) -> Tuple[Any, List[float]]:
    """One diagnosis of `phenotypes` (HPO terms) through a pipeline serving a
    disease child: a `disease_task/v1` result whose candidates are the first
    `top_k` of every disease of the model's graph view (fewer when the view
    holds fewer), ranked by finite scores in descending order
    (`candidate_score`). Returns the result and its scores."""
    import math as _math

    from src.core.types import DISEASE_TASK, PatientPhenotypes, candidate_score

    result = pipeline.run(
        patient_input=PatientPhenotypes(patient_id="probe", phenotypes=list(phenotypes)),
        top_k=top_k, include_explanations=False,
    )
    assert result.result_kind == DISEASE_TASK, (
        f"a disease child served, and the result is {result.result_kind!r}")
    candidates = list(getattr(result, "candidates", []))
    assert candidates, "the pipeline ranked nothing"
    # Every disease of the view is ranked (a private read of the served view,
    # as E8 reads the served vectors); the result shows its first `top_k`.
    diseases = int(pipeline._bundle_view.type_nodes("disease").numel())
    assert len(candidates) == min(top_k, diseases), (
        f"the result shows {len(candidates)} of the view's {diseases} diseases, top_k {top_k}")
    scores = [float(candidate_score(c)) for c in candidates]
    assert all(_math.isfinite(score) for score in scores), (
        "a candidate came back with a non-finite score")
    assert scores == sorted(scores, reverse=True), "candidates came back unordered"
    return result, scores


def _child_view(checkpoint: Path, workspace: Path) -> Any:
    """A model's own graph view (a disease child's for T2 and RT2, a base's or
    a child's for L2 and L5), as the probe computes it apart from serving and
    measurement: rebuilt from the workspace's graph by the rule the model
    record names, and required to be the view its record holds (by digest)."""
    from src.kg.graph_view import build_graph_view
    from src.kg.storage.file_storage import read_graph_artifacts
    from src.models.bundle.record import read_model_record
    from src.utils.checkpoint_io import read_checkpoint

    record = read_model_record(read_checkpoint(checkpoint, map_location="cpu",
                                               weights_only=True).checkpoint)
    graph = read_graph_artifacts(workspace, map_location="cpu").graph_data
    view = build_graph_view(graph, rule=record.data_roles["graph_view"]["rule"])
    assert view.digest == record.graph_view.digest, (
        "the child's graph view, rebuilt from the workspace by its recorded rule, is not the "
        "view its record names")
    return view


def _workspace_query(checkpoint: Path, workspace: Path,
                     workspace_digests: Mapping[str, str]) -> Tuple[List[str], Dict[str, Any]]:
    """A diagnosis query from the workspace being judged, never a constant (the
    reviewer's P2-1 of the K8-1 recheck: R6 asked about the demo graph's two
    phenotypes, so a deployment whose graph lacks them failed). It is the
    first case of the workspace's generated `val` split that the child's
    graph view binds (`_child_view`, by the one binder,
    `bind_disease_cases`) and whose phenotypes the query names as terms
    (`_node_term`, the importer's rule inverted, which serving's term to node
    mapping maps back to the same nodes). The split is read against the
    digest the caller's workspace check took; the graph that names its
    phenotypes is the one serving's identity must then name by that check's
    digest (`_assert_served_identity`). Returns the query's
    terms and, as evidence, where it came from (a position and counts, no
    identifier). A workspace none of whose validation cases gives a query
    fails, saying so: the check has nothing of that workspace's to ask."""
    from src.evaluation.cohort import resolve_cohort
    from src.kg.artifacts import GRAPH_ARTIFACTS
    from src.kg.graph import KnowledgeGraph
    from src.kg.storage.file_storage import read_samples_file
    from src.models.tasks.scoring import bind_disease_cases

    path = resolve_cohort(workspace, "val").samples
    samples = read_samples_file(path, expected_sha256=workspace_digests[path.name]).samples
    binding = bind_disease_cases(_child_view(checkpoint, workspace), samples)
    bound = {case.case_id for case in binding.cases}
    phenotypes = (KnowledgeGraph.read_json(workspace / GRAPH_ARTIFACTS["kg"]).kg
                  .get_reverse_node_mapping().get("phenotype", {}))
    for position, sample in enumerate(samples):
        if sample.patient_id not in bound:
            continue
        # A bound case's phenotype the graph does not name is the workspace's
        # split and graph disagreeing: that data condition, said as such.
        unnamed = [i for i in sample.phenotype_ids if i not in phenotypes]
        if unnamed:
            raise AssertionError(
                f"validation case {position} of the workspace names phenotype id "
                f"{unnamed[0]}, which the workspace's graph ({GRAPH_ARTIFACTS['kg']}) does not "
                f"hold among its {len(phenotypes)} phenotypes: the split and the graph disagree, "
                "so no query is asked of them")
        terms = [_node_term(phenotypes[i], "phenotype") for i in sample.phenotype_ids]
        if None not in terms:
            return terms, {"split": "val", "case_position": position,
                           "phenotypes": len(terms), "cases": binding.counts["cases"],
                           "cases_bound": binding.counts["bound"]}
    counts = binding.counts
    raise AssertionError(
        f"the workspace gives no diagnosis query: none of its {counts['cases']} validation "
        f"cases both binds in the child's graph view and has phenotypes a query can name as "
        f"HPO terms ({counts['bound']} bind, {counts['excluded']} do not)")


def _serve_and_diagnose(
    kg_path: Path, checkpoint: Path, data_dir: Path, device: str, record: Any,
    workspace_digests: Dict[str, str],
) -> Dict[str, Any]:
    """The disease-task serving check: the production serving build
    (`build_pipeline`) of a disease child, and one diagnosis through the built
    pipeline, of a query from the served workspace's own data
    (`_workspace_query`). No table is read: a task model reads none.

    `record` and `workspace_digests` are the dataset record and the per-file
    digests the caller's workspace check (`verify_workspace`) took; the
    diagnosis must say it was produced by those inputs, for the disease task
    (contract M3c).
    """
    import src.api.main as api

    query, asked = _workspace_query(checkpoint, data_dir, workspace_digests)
    bundle = api.build_pipeline(
        kg_path=str(kg_path), checkpoint_path=str(checkpoint),
        data_dir=str(data_dir), device=device,
    )
    config = bundle.config
    assert config.get("gnn_ready"), "the GNN did not come up"
    # The served workspace's record, binding state and origin apart (contract
    # M3b): the record the workspace check verified, whole.
    published = config.get("dataset_record")
    assert published == record.summary(), (
        f"serving publishes its dataset record as {published}, not the record the "
        f"workspace check verified ({record.binding}, origin {record.origin})"
    )
    devices = {str(p.device).split(":")[0] for p in bundle.pipeline.model.parameters()}
    assert devices == {device}, f"the model's parameters are on {sorted(devices)}"
    result, scores = _disease_diagnosis(bundle.pipeline, query)
    # **The diagnosis names what produced it** (contract M3c): the child this
    # probe trained, by the digest of its own read; the graph and manifest the
    # workspace check verified; the dataset record; the disease task's scoring
    # mode and shortest-path term; and the device in effect. The result's
    # identity is the one serving reports, built from one state.
    identity = _assert_served_identity(
        result.served_identity.to_dict(),
        checkpoint_sha256=_read_run_checkpoint(checkpoint)["sha256"],
        workspace_digests=workspace_digests,
        record=record,
        task="disease",
        device=device,
    )
    assert identity == config.get("served_identity"), (
        "the diagnosis carries another served identity than serving reports"
    )
    return {
        "scoring_mode": identity["scoring_mode"],
        "result_kind": result.result_kind,
        "dataset_record": {"binding": published["binding"], "origin": published["origin"],
                           "sha256": published["sha256"], "sources": len(published["sources"])},
        "parameter_devices": sorted(devices),
        "query": asked,
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


def _import_cohort(registry: Path, data_dir: Path, source: Optional[Path], label: str,
                   commit: Optional[str] = None, *,
                   kind_argv: Optional[Sequence[str]] = None) -> Tuple[int, str, str]:
    """`scripts/import_cohort.py`'s `main`, its exit code and what it printed.

    A supplied cohort from the case file `source` with M4's role, or, with
    `kind_argv`, the version those arguments name in its place (``--kind``
    and what that kind takes: a KG edge set's ``--from-checkpoint`` and
    ``--role``, a generated split's ``--role``; `_kg_edges_argv`,
    `_generated_argv`)."""
    import io

    import scripts.import_cohort as import_cohort

    argv = ["--registry", str(registry), "--data-dir", str(data_dir), "--label", label]
    argv += (list(kind_argv) if kind_argv is not None
             else ["--source", str(source), "--role", M4_ROLE])
    if commit is not None:
        argv += ["--commit", commit]
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = import_cohort.main(argv)
    return code, out.getvalue(), err.getvalue()


def _import_version(registry: Path, data_dir: Path, source: Optional[Path], label: str, *,
                    kind_argv: Optional[Sequence[str]] = None) -> Tuple[Any, str]:
    """The importer's two steps through its CLI: the preview, which must leave
    the registry as it was, then the commit by the preview id it printed.
    Returns the version as `resolve_version` reads it, and the preview id. A
    supplied cohort from `source`, or the version `kind_argv` names
    (`_import_cohort`).

    A refusal's message names paths, so it goes to the console and the report
    says what happened in this file's words. A registry inside the repository
    -- a --work-dir inside it -- is said as such.
    """
    from src.evaluation.cohort_registry import resolve_version

    before = snapshot(registry, include_dirs=True)
    code, out, err = _import_cohort(registry, data_dir, source, label, kind_argv=kind_argv)
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
    code, _, err = _import_cohort(registry, data_dir, source, label, commit=found.group(1),
                                  kind_argv=kind_argv)
    if code != 0:
        print(err, end="", file=sys.stderr)
        raise AssertionError(f"the commit by the printed preview id was refused (exit {code}); "
                             "its message is on the console")
    return resolve_version(registry, label), found.group(1)


def _measure_cohort(checkpoint: Path, data_dir: Path, registry: Path, label: str,
                    device: str, *extra: str, protocol: str) -> int:
    """The production measurement entry point on a registered cohort version
    (contract M4): `--cohort` with `--registry`, measuring the model by the
    evaluation `protocol` asked (`--mode`), and `extra` (`--record`). Run
    inside an M4 flow, where that registry is the deployment's
    (`_registry_setting`, set once for the flow)."""
    import scripts.measure_scorer as measure_scorer

    return measure_scorer.main([
        "--mode", protocol, "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
        "--cohort", label, "--registry", str(registry), "--device", device, *extra,
    ])


def _cohort_samples(version: Any) -> List[Any]:
    """A cohort version's samples, read against the digest its `cohort.json`
    binds: the cases T2 and RT2 judge a report by, in the version's order."""
    from src.kg.storage.file_storage import read_samples_file

    samples = read_samples_file(version.samples_path,
                                expected_sha256=version.manifest["samples"]["sha256"]).samples
    assert len(samples) == version.manifest["samples"]["n"], (
        f"the version's samples hold {len(samples)} cases, not the "
        f"{version.manifest['samples']['n']} its cohort.json counts")
    return samples


def _kept_report(registry: Path, label: str,
                 sample_ids: Sequence[str]) -> Tuple[Path, Dict[str, Any], Any, List[str]]:
    """The one report the version's report store keeps, parsed and identified
    from one read, and the cases its ranks rank.

    **Through the shared reader** (`_read_json`): the report's content and its
    digest come from the same bytes, as registration takes them, and the ranks
    are compared with the digest the report names before they are parsed.
    Then **the ranks rule** (`check_ranks_rows`, beside the writer): one row
    per case of the version, `sample_ids` as its `cohort.json` binds them, in
    their order, a case the model's graph view cannot rank kept with a null
    rank (the reviewer's P2-2 of the K8-1 recheck: a repeated, missing,
    foreign or moved case passed the digest and the counts). The rows with a
    rank are the cases the report counts as ranked. Returns the report's path,
    its content, the identity of the bytes parsed and the sample ids of the
    ranked rows, in the version's order.
    """
    from scripts.measure_scorer import RANKS_FILE, REPORT_FILE
    from src.evaluation.measurement import RanksRowsRefused, check_ranks_rows
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
    try:
        ranked = check_ranks_rows(rows, sample_ids)
    except RanksRowsRefused as refused:
        raise ProbeFailure(
            f"the kept ranks are not one row per case of the version, in its order "
            f"({refused.problem}, at position {refused.position})",
            {"ranks_rule": refused.problem, "position": refused.position}) from None
    assert len(ranked) == report.get("n_ranked"), (
        f"the kept ranks rank {len(ranked)} cases, not the {report.get('n_ranked')} the report "
        "counts")
    return kept[0], report, identity, ranked


def _own_binding(checkpoint: Path, workspace: Path, samples: Sequence[Any]) -> Any:
    """The binding of a cohort version's cases that the probe computes itself,
    apart from the measurement it checks: the disease child's own graph view
    (`_child_view`) and the version's samples (`_cohort_samples`), bound by
    the one binder (`src.models.tasks.scoring.bind_disease_cases`). Returns
    the binding: its counts, the form a task report's `binding` takes, and
    its bound cases, in the version's order."""
    from src.models.tasks.scoring import bind_disease_cases

    return bind_disease_cases(_child_view(checkpoint, workspace), samples)


def _ranked_cases(kept: Mapping[str, Any], ranked: Sequence[str], own: Any) -> Dict[str, Any]:
    """A task report against its cohort version's cases, whose ranks rows
    `_kept_report` held to the ranks rule: as many ranked as the cases the
    model's graph view binds (the report's binding counts; a case whose
    phenotypes are all outside the view is counted, never ranked), and the
    ranked rows are those very cases. **The binding is the probe's own**
    (`own`, `_own_binding`): the report's counts must be the ones the probe
    computed from the child's graph view and the version's cases, so a view
    or binding regression that leaves few cases ranked fails here, not only a
    report that disagrees with itself. Returns the binding counts, as
    evidence."""
    counts = dict(own.counts)
    binding = (kept.get("sampler_evidence") or {}).get("binding") or {}
    assert binding.get("cases") == counts["cases"] and kept["n_ranked"] == binding.get("bound"), (
        f"the report ranks {kept['n_ranked']} of {binding.get('cases')} cases; the view binds "
        f"{binding.get('bound')}")
    assert binding == counts, (
        f"the report binds {binding.get('bound')} and excludes {binding.get('excluded')} of "
        f"{binding.get('cases')} cases ({binding}); the child's graph view binds "
        f"{counts.get('bound')} and excludes {counts.get('excluded')} ({counts})")
    assert list(ranked) == [case.case_id for case in own.cases], (
        "the ranks rank other cases than the ones the child's graph view binds")
    assert kept["n_ranked"] > 0, "the report ranked no case"
    return dict(binding)


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
    report's bytes, naming the run it was registered from in a schema-4 ledger
    whose run identity starts at that record (K8-6r.2). Returns the record and
    the digest of the ledger read."""
    from src.evaluation.sidecar import (
        LEDGER_FILENAME,
        RUN_ID_PATTERN,
        RUN_IDENTITY_FROM_SEQUENCE,
        read_ledger,
    )

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
    assert isinstance(record.get("run_id"), str) and RUN_ID_PATTERN.fullmatch(record["run_id"]), (
        "the record does not name the run it was registered from")
    assert (ledger.get("schema_version"), ledger.get(RUN_IDENTITY_FROM_SEQUENCE)) == (4, 1), (
        "the ledger a registration created is not schema 4 with run identity from record 1")
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


@contextlib.contextmanager
def _observing_builds(observed: List[str]):
    """While this is open, each model built through the one builder
    (`src.models.bundle.builder.build_bundle`) is noted by its task, then built
    as asked.

    **Observed, not inferred from the refusal's words**, as
    `_observing_checkpoint_reads` observes the reads: a refusal that must come
    before any model is built is shown to. `load_bundle` imports the builder
    when it runs, so serving reaches this wrapper.
    """
    import src.models.bundle.builder as builder

    original = builder.build_bundle

    def observing(spec, *args, **kwargs):
        observed.append(str(getattr(spec, "task", None)))
        return original(spec, *args, **kwargs)

    builder.build_bundle = observing
    try:
        yield
    finally:
        builder.build_bundle = original


def _serve_and_view(data_dir: Path, checkpoint: Path, expected_sha256: str,
                    device: str) -> Tuple[str, Dict[str, Any]]:
    """`checkpoint`, a disease child, served through the production reload,
    named explicitly with its task (E4's route, which publishes what it built)
    and **bound to the digest the probe expects** (`expected_checkpoint_sha256`,
    compared on the build's own checkpoint read before the bytes are parsed,
    as the dashboard's Load sends path, digest and task), then the status and the loaded model's recorded results read as the
    Diagnosis panel reads them: two GETs on the application, status first.
    Called inside an M4 flow, so the route reads the probe's registry as the
    deployment's (`_registry_setting`), and each record's group description is from the
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
        data_dir=str(data_dir), checkpoint_path=str(checkpoint), task="disease", device=device,
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


#: A record's group description when the one report its key has in the store is the one
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
    assert (view.get("run_id"), view.get("historical_overlap")) == (record["run_id"], []), (
        "the view does not show the run the record names, or shows an overlap")
    assert (view["cohort"]["label"], view["cohort"]["role"], view["n_ranked"]) == (
        record["cohort"]["label"], record["cohort"]["role"], record["n_ranked"])
    for scope in scopes:
        state = view["r10"][scope]["evidence"]["state"]
        assert state == "found", f"the view shows R10's {scope} evidence as {state}, not found"
    # The review's P2-6, seen end to end (root-cause round, INT-1): the route
    # reads the probe's registry, so the record's state is from its store.
    contest = view.get("contest") or {}
    assert contest == UNCONTESTED, (
        f"the view shows the record's group description as {contest.get('state')}, not "
        "uncontested")
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
#: run holding a copy of M2's resumed disease child, the model W7-W11's test
#: runs measure (NEW-probe-1), and one whose checkpoint is a link out of the
#: workspace to D1's child (listed with that reason, never opened).
M5_WORKSPACE = "probe-demo"
M5_LISTED_RUN = "probe-m5-listed"
M5_LINKED_RUN = "probe-m5-link-out"
M5_CHECKPOINT = "last.pt"
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
    ("POST", "/api/v1/training/finetune", 422),
    ("POST", "/api/v1/training/branch", 422),
    ("POST", "/api/v1/training/stop", 409),
    ("GET", "/api/v1/training/checkpoints", 200),
    ("GET", "/api/v1/workspaces/{name}/checkpoints", 200),
    ("GET", "/api/v1/workspaces/{name}/family", 200),
)
#: The gated dashboard events called directly through Gradio's HTTP API, as a
#: script or `gradio_client` would, each found by the handler it is registered
#: with (its module under `src/webui/components` and its name; its API name is
#: the dashboard's own, `_dashboard_events`), with the data sent and what its
#: answer says when D8 allows: Training's confirmed Stop with no training
#: running, Test's Start with no listed version and no measurement chosen,
#: Test's Register with no kept report chosen. Allowed or refused, nothing is
#: done.
GATED_EVENTS: Tuple[Tuple[str, str, Tuple[Any, ...], str], ...] = (
    ("training_console", "_on_stop", ({},), "No training is running"),
    ("test_page", "_on_start", ({}, {}, None, None, None, None, False, False),
     "Nothing was started."),
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
    "SHEPHERD_INTAKE_DIR", "SHEPHERD_TRAINING_EVIDENCE_DIRS",
    "SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND", "SHEPHERD_BIND_HOST", "SHEPHERD_DEVICE",
    "SHEPHERD_KG_PATH", "SHEPHERD_CHECKPOINT_PATH", "SHEPHERD_DATA_DIR",
)
#: How long M5's waits allow: a service to answer, a job to take the GPU lock,
#: a training job to reach its first epoch, and a test run to end.
_SERVICE_START_SECONDS = 300.0
_LOCK_TAKEN_SECONDS = 300.0
_FIRST_EPOCH_SECONDS = 1800.0
_TEST_RUN_SECONDS = 3600.0
#: How long a short training job the runner runs to its end may take
#: (L7.branch: the demo base's epochs at the demo's sizes).
_TRAINING_RUN_SECONDS = 3600.0
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
    setting inherits it, the loaded-model route's group description included, whatever
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


def _listed_test_inputs(label: str, workspace: str, run: str, filename: str, *,
                        protocol: str, **choices: Any) -> Any:
    """A test run's inputs as the Test page builds them
    (`test_page.start_test_run`): the version, the workspace and the checkpoint
    found in the server's lists again by name, the digests those lists show
    passed as the expected ones, the evaluation `protocol` asked apart from the
    learning stage, task and capability description the listing's model
    record gives, the listed version's kind and counts, the registry from its
    setting and the evidence directories by the runner's one choice by the
    version's kind (`evidence_choice`: the setting's, which must not be
    refused, for a supplied version; none, the setting unread, for a KG edge
    set or a generated split). `protocol` and `choices`
    (`record`, `deterministic`) are the operator's. As the page's Start does,
    the protocol is first judged on that description and that version
    (`src.evaluation.protocols.refusal`); the runner judges it again before it
    starts, and the run on the bytes it reads. M5's test runs ask for the
    disease protocol on the probe's own disease children."""
    from src.api.services import server_lists
    from src.api.services.training_manager import TestRunInputs, evidence_choice
    from src.config import server_settings
    from src.evaluation import protocols

    version = next((item for item in server_lists.list_cohort_versions().items
                    if item.label == label), None)
    assert version is not None and version.manifest_sha256, (
        f"the server lists no readable cohort version {label}")
    listed = next((item for item in server_lists.list_checkpoints(workspace).items
                   if (item.run, item.filename) == (run, filename)), None)
    assert listed is not None and listed.sha256, (
        f"the server lists no readable checkpoint {run}/{filename} in {workspace}")
    why = protocols.refusal(protocol, listed.capabilities, version.kind, version.counts)
    assert why is None, (
        f"the {protocol} measurement is refused for the listed checkpoint {run}/{filename}: {why}")
    registry, evidence = server_settings.cohort_registry(), evidence_choice(version.kind)
    assert registry.available, "the registry setting is refused"
    assert evidence.problem is None, "the evidence setting is refused"
    return TestRunInputs(
        registry=registry.value, version_label=label, version_sha256=version.manifest_sha256,
        workspace=server_lists.resolve_workspace(workspace),
        checkpoint=server_lists.resolve_checkpoint(workspace, listed.architecture, run,
                                                   filename),
        checkpoint_sha256=listed.sha256, protocol=protocol,
        learning_stage=listed.learning_stage, task=listed.task,
        capabilities=listed.capabilities, version_kind=version.kind,
        version_counts=version.counts, evidence_dirs=evidence.dirs, **choices)


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


def _hold_epochs() -> int:
    """The epochs of the training job W11 and RW3 hold the GPU with: the most
    a run may be given, read from the field spec (`num_epochs`'s upper bound),
    never a copied number. Far more than it can complete while the probe
    watches (a demo base epoch takes seconds), so it ends only by the probe's
    stop (or a failure, which the checks name), never by itself during the
    check (review of 8242b47, W11/RW3)."""
    from src.config.training_fields import by_name

    return int(by_name("num_epochs").valid["le"])


def _held_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """`config` as the job W11 and RW3 hold the GPU lock with runs it: a new
    base run, whatever the settings name (no parent, no extension), that ends
    only by the probe's stop. The trainer's three ways of ending by itself are
    closed here, for both callers: the stage's end (`_hold_epochs()` epochs),
    early stopping (no patience, which disables it: `src.config.recipe`) and
    a pause (none). Root-cause round, W11-2: the hold is this function's, not
    a copy at each caller."""
    held = {name: value for name, value in config.items() if name not in _NOT_HELD}
    held.update(learning_stage="base", num_epochs=_hold_epochs(), early_stopping_patience=None)
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
             child: Optional[Path], registry: Path, version: Any,
             m4_runs: Sequence[str] = (), m4_registered: Sequence[str] = ()) -> None:
    """M5's acceptance on the demo workspace (W1-W11), after M4's flow, on the
    probe's own deployment under <work>/m5 with M2's resumed disease child
    (`child`, W1's listed run; `checkpoint` is D1's child, the link out of the
    workspace and W10's other model) and T1's cohort version (`version` in
    `registry`). W7-W11's test runs measure W1's listed copy, on the same
    workspace copy, the graph T1's version is bound to: they ask for the
    disease protocol, which takes a finetuned disease model, and the chain's
    own child is one (NEW-probe-1). `m4_runs` are the run ids M4's flow
    published in that version's store (T2's): W1 lists exactly those. `m4_registered` are those of them whose
    probe passed, so registered: W1 checks each is listed as registered.

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
        # The listed runs are filed under the encoder kind of the child they
        # hold, M2's child of D1's base (`_demo_encoder_kind`, as D1 files it).
        architecture = _demo_encoder_kind()
        runs = workspace / RUNS_DIRNAME / architecture
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
                mine.problem) == (architecture, M5_CHECKPOINT, listed_sha, True,
                                  None), (
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
            M5_WORKSPACE, architecture, M5_LISTED_RUN, M5_CHECKPOINT) == listed
        refused = [
            refuses(lambda: server_lists.resolve_checkpoint(
                M5_WORKSPACE, architecture, M5_LISTED_RUN, "../" + M5_CHECKPOINT),
                server_lists.ServerListRefused, "is listed in the workspace"),
            refuses(lambda: server_lists.resolve_workspace("../m4"),
                    server_lists.ServerListRefused, "is not a workspace the server lists"),
        ]
        if link_made:
            refused.append(refuses(lambda: server_lists.resolve_checkpoint(
                M5_WORKSPACE, architecture, M5_LINKED_RUN, M5_CHECKPOINT),
                server_lists.ServerListRefused, "resolves outside the workspace"))
        versions = server_lists.list_cohort_versions(
            report_state=server_lists.stored_report_state)
        assert versions.available, "the server lists no cohort versions"
        demo = next((entry for entry in versions.items if entry.label == M4_DEMO_LABEL), None)
        assert demo is not None and demo.problem is None, "T1's version is not listed"
        assert (demo.manifest_sha256, demo.role) == (version.identity.sha256, M4_ROLE), (
            "the version is not listed by T1's digest and role")
        # What the later probes need is kept before W1's own run checks, so a
        # failure here is W1's and never W2-W11's.
        state.update(workspace=workspace, listed=listed, listed_sha=listed_sha,
                     fingerprint=fingerprint, link_made=link_made)
        # The store's run directories (its own files, the usage history and
        # the notes, are not runs): exactly the runs M4's flow published, each
        # kept, and each whose probe passed (T2's) listed as registered.
        assert m4_runs, "T2's run (M4) is not available"
        listed_runs = {run.run_id: run.state or {} for run in demo.reports
                       if is_run_id(run.run_id)}
        assert sorted(listed_runs) == sorted(m4_runs), (
            f"the store lists {len(listed_runs)} runs, not the {len(m4_runs)} M4's flow "
            "published (T2's)")
        not_kept = [run for run in m4_runs if listed_runs[run].get("kept") != "kept"]
        assert not not_kept, f"M4's runs {not_kept} are not listed as kept"
        unregistered = [run for run in m4_registered
                        if listed_runs[run].get("registered") != "registered"]
        assert not unregistered, (
            f"M4's runs {unregistered} registered by their probes (T2) are not listed as "
            "registered")
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
                               "role": demo.role,
                               "reports": [{"kept": listed_runs[run].get("kept"),
                                            "registered": listed_runs[run].get("registered")}
                                           for run in m4_runs]},
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

    def _task_inputs(**choices: Any) -> Any:
        """The test-run inputs for W1's listed copy of M2's disease child,
        found in the server's lists again by name (`_listed_test_inputs`):
        listed by the digest of the bytes W1 read, and by its model record (a
        finetuned disease model); the disease protocol is asked of it."""
        _need("listed", "listed_sha")
        inputs = _listed_test_inputs(M4_DEMO_LABEL, M5_WORKSPACE, M5_LISTED_RUN, M5_CHECKPOINT,
                                     protocol="disease", **choices)
        assert (inputs.checkpoint, inputs.checkpoint_sha256, inputs.learning_stage,
                inputs.task) == (state["listed"], state["listed_sha"], "finetune", "disease"), (
            "the listed disease child is not listed by its run, its digest as read and its "
            "model record")
        return inputs

    def _gpu_lock() -> Tuple[str, Dict[str, Any]]:
        import subprocess

        import yaml

        from scripts.setup_demo import demo_task_training

        _need("workspace", "fingerprint")
        lock_path = _lock_path()
        inputs = _task_inputs()
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
            # A base start, as the dashboard's Training page makes one, in the
            # demo's sizes: it passes the start's own check of its settings
            # (`_new_bundle_run`), so its refusal is the lock's.
            training = manager.start_training(
                {"learning_stage": "base", **demo_task_training()["base"],
                 "data_dir": str(state["workspace"]), "num_epochs": 1,
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
                "learning_stage": "base", **demo_task_training()["base"],
                "data_dir": str(state["workspace"]), "num_epochs": 1,
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
        lock_path = _lock_path()
        inputs = _task_inputs(record=False)
        child = {"path": state["listed"], "sha256": state["listed_sha"]}
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
        assert kept["manifest"]["artifact_digests"]["checkpoint"] == child["sha256"]
        assert kept["manifest"]["cohort_version"]["manifest_sha256"] == version.identity.sha256
        # **Measured by the protocol asked**: the runner passes it as `--mode`
        # and the report's mode is that protocol's name.
        assert kept["manifest"]["mode"] == inputs.protocol, (
            f"the report's mode is {kept['manifest']['mode']!r}, not the protocol asked "
            f"({inputs.protocol})")
        entries, problems = read_usage_history(version)
        new = entries[usage_before:]
        assert not problems and [entry.get("kind") for entry in new] == ["test_run"], (
            f"the usage history gained {[entry.get('kind') for entry in new]}")
        entry = new[0]
        assert (entry.get("run_id"), entry.get("checkpoint_sha256"), entry.get("report_sha256"),
                entry.get("record_requested")) == (created[0], child["sha256"], published,
                                                   False), "the usage entry is not this run's"
        assert str(entry.get("device", "")).split(":")[0] == device, (
            f"the run measured on {entry.get('device')}, not {device}")
        assert lock_probe(lock_path) is None, "the GPU lock stayed held after the run ended"
        state["job"] = {"run_id": created[0], "report": report_file, "published": published,
                        "checkpoint": child["path"]}
        return "a test run from the server's lists: its report kept, not registered, in the " \
               "usage history", {
                   "inputs": "the version, workspace and checkpoint found in the server's lists "
                             "again, their digests passed as the expected ones",
                   "measured": {"model": "W1's listed copy of M2's resumed disease child",
                                "run": M5_LISTED_RUN, "sha256": child["sha256"],
                                "protocol": kept["manifest"]["mode"]},
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
        # The checkpoint W8 measured, W1's listed disease child: its ledger is
        # the one beside it, under runs/<encoder kind>/<run>/checkpoints.
        job, listed = state["job"], state["job"]["checkpoint"]
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
        # **The disease protocol's record**: its scoring mode is the protocol
        # W8 asked, and its metrics the disease evaluator's
        # (`task_measurement.TASK_RUNS["disease"]`).
        scoring, metrics = ledger["records"][0]["scoring"], ledger["records"][0]["metrics"]
        assert scoring.get("mode") == "disease" and "pooled_mrr" in metrics, (
            f"the record is scored as {scoring.get('mode')!r}, not by the disease protocol")
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
            "ledger": "beside the listed disease child's checkpoint, in its run",
            "scoring_mode": scoring["mode"],
            "outcome": result.outcome, "sequence": sequence, "ledger_sha256": ledger_sha256,
            "report_state": {"kept": after.kept, "registered": after.registered},
            "usage_history": "registered", "server_list": "kept, registered",
        }

    def _start_checked_again() -> Tuple[str, Dict[str, Any]]:
        import dataclasses

        lock_path = _lock_path()
        inputs = _task_inputs()
        # D1's child: another model of the same task and workspace.
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
        inputs = _task_inputs()
        manager = TrainingManager()
        from scripts.setup_demo import demo_task_training

        # The held job is a base run in the demo's sizes, held by
        # configuration where it is started (`_held_config`).
        config = {**demo_task_training()["base"], "data_dir": str(state["workspace"]),
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
             state: Dict[str, Any],
             task_sizes: Optional[Mapping[str, Mapping[str, Any]]] = None) -> None:
    """M5's acceptance on the real workspace (RW1-RW3), after M4's RT1-RT3 and
    before R8, which therefore covers them: the workspace as the server lists
    it, a test run through the job runner on the device of a disease child the
    probe finetunes from R4's resumed base (`state["base"]`) into a run
    directory of its own, one epoch (`_train_disease_child`; the reference
    recipe, or `task_sizes` over it), and a base training job started and
    stopped through the runner (no long training). RW2 never measures
    R4.finetune's child: RT2 already registered a record beside it, and RW2's
    run must be the one record beside its own. Everything the runs write is under
    --work-dir: the GPU lock is the one under a workspaces root of the probe's
    own there (an empty directory, never listed), the registry is RT1's (named
    as `SHEPHERD_REGISTRY` while they run: a test run measures only in the
    deployment's registry, BACKLOG 37), the runs' directories and ledgers,
    the disease child's among them, are under <work>/real/m5. The
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

    def _task_child() -> Dict[str, Any]:
        """The disease child RW2 measures and RW3's refused test start names:
        finetuned by the production trainer from R4's resumed base on the real
        workspace, which is only read, into a run directory of its own under
        <work>/real/m5 (`_train_disease_child`), with the learning stage, task
        and capability description its model record gives (through the one
        reader), which the test runs' inputs carry as the listed record's. Trained once, by RW2; a
        training that fails is RW2's failure, and RW3 then says it is not
        available."""
        from src.models.bundle.record import read_model_record
        from src.utils.checkpoint_io import read_checkpoint

        assert "read" in state, "R1 did not pass"
        assert "base" in state, "R4's resumed base is not available"
        if "task_child" not in state:
            assert "task_child_tried" not in state, (
                "the disease child could not be trained (RW2 says why)")
            state["task_child_tried"] = True
            trained = _train_disease_child(real, m5 / "disease-child", device,
                                           base=state["base"], sizes=task_sizes)
            read = read_checkpoint(trained.child, map_location="cpu", weights_only=True)
            record = read_model_record(read.checkpoint)
            state["task_child"] = {"path": trained.child, "sha256": read.identity.sha256,
                                   "learning_stage": record.learning_stage,
                                   "task": record.task,
                                   "capabilities": record.capabilities,
                                   "training_seconds": trained.seconds}
        return state["task_child"]

    def _test_job() -> Tuple[str, Dict[str, Any]]:
        assert "record" in state.get("m4", {}), "the cohort version (RT1, RT2) is not available"
        version = state["m4"]["version"]
        # The disease protocol asked of the probe's own disease child,
        # finetuned here from R4's resumed base; the runner judges the request
        # on the child's model record and the version's counts.
        child = _task_child()
        before, usage_before = _store_runs(reports), len(read_usage_history(version)[0])
        manager = TrainingManager()
        with _job_ended_on_exit(manager, "test"):
            started = manager.start_test(TestRunInputs(
                registry=registry, version_label=M4_REAL_LABEL,
                version_sha256=version.identity.sha256, workspace=real,
                checkpoint=child["path"], checkpoint_sha256=child["sha256"],
                protocol="disease", learning_stage=child["learning_stage"], task=child["task"],
                capabilities=child["capabilities"], version_kind=version.kind,
                version_counts=version.manifest.get("counts"), record=True))
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
        kept, kept_read = _read_json(reports / created[0] / REPORT_FILE)
        assert published == kept_read.sha256, (
            "the outcome file's digest is not the kept report's bytes")
        assert (kept["manifest"]["artifact_digests"]["checkpoint"], kept["manifest"]["mode"]) == (
            child["sha256"], "disease"), (
            "the kept report is not the disease child's, or its mode is not the protocol asked "
            "(disease)")
        assert (outcome.get("kept"), outcome.get("registered")) == (True, True), (
            f"the outcome says kept {outcome.get('kept')}, registered {outcome.get('registered')}")
        ledger, _ = read_ledger(child["path"].resolve().parent / "evaluations.json")
        assert [record.get("source_artifact_digest") for record in ledger["records"]] == [
            published], "the ledger beside the disease child does not hold the run's one record"
        assert ledger["records"][0]["scoring"].get("mode") == "disease", (
            "the registered record is not scored by the disease protocol")
        new = read_usage_history(version)[0][usage_before:]
        assert [(e.get("kind"), e.get("run_id")) for e in new] == [
            ("test_run", created[0]), ("registered", created[0])], (
            f"the usage history gained {[e.get('kind') for e in new]}")
        assert lock_probe(lock_path) is None, "the GPU lock stayed held after the run ended"
        return "a test run through the runner on the real workspace, kept and registered", {
            "checkpoint": "the probe's disease child, finetuned on the real workspace into the "
                          "work directory",
            "measured": {"model": "a disease child finetuned from R4's resumed base, one epoch, "
                                  "by the production trainer, in a run directory of its own",
                         "recipe": ("the reference recipe" if task_sizes is None
                                    else "the reference recipe at smaller sizes"),
                         "sha256": child["sha256"], "protocol": kept["manifest"]["mode"],
                         "training_seconds": child["training_seconds"]},
            "status": word, "exit_code": status["exit_code"], "report_sha256": published,
            "kept": True, "registered": True, "usage_history": ["test_run", "registered"],
            "device": new[0].get("device"), "n_ranked": new[0].get("n_ranked"),
            "gpu_lock": "taken by the run itself, released at its end",
            "training_stop_while_it_ran": not_reached,
        }

    def _training_job() -> Tuple[str, Dict[str, Any]]:
        assert "read" in state, "R1 did not pass"
        assert "m4" in state, "the cohort version (RT1) is not available"
        version = state["m4"]["version"]
        # The test run RW2 measured, the disease child's: only refused while
        # training runs (the slot, then the CLI's lock).
        child = _task_child()
        inputs = TestRunInputs(
            registry=registry, version_label=M4_REAL_LABEL,
            version_sha256=version.identity.sha256, workspace=real, checkpoint=child["path"],
            checkpoint_sha256=child["sha256"], protocol="disease",
            learning_stage=child["learning_stage"], task=child["task"],
            capabilities=child["capabilities"], version_kind=version.kind,
            version_counts=version.manifest.get("counts"))
        manager = TrainingManager()
        # A new base run by the reference recipe (or `task_sizes` over it, as
        # R3's), held by configuration (`_start_training_job` applies
        # `_held_config`): epochs and patience that would let it end by itself
        # during the check, and a parent that would make it a continuation
        # (contract M3a-3), are never passed.
        config = {**((task_sizes or {}).get("base") or {}), "data_dir": str(real),
                  "run_dir": str(m5 / "stopped-training"), "device": device}
        with _job_ended_on_exit(manager, "training"):
            started, held = _start_training_job(manager, config,
                                                state["read"][MANIFEST_FILENAME], lock_path)
            outcome = m5 / "outcome-while-training.json"
            refused = _exclusion_while_training(
                manager, inputs, started["pid"],
                build_test_argv(inputs, deployment_device(), outcome)[2:], outcome)
            stopped = _stop_training_job(manager, lock_path)
        return "a training job on the real workspace held the GPU and was stopped", {
            **held, "while_it_ran": refused, "stop": stopped,
        }

    lock_root.mkdir(parents=True, exist_ok=True)
    with _deployment_settings(settings):
        report.run("RW2", "real", "M5: a test run through the job runner on the real workspace "
                   "is kept and registered", _test_job)
        report.run("RW3", "real", "M5: a training job on the real workspace holds the GPU "
                   "lock and is stopped as stopped, other jobs refused", _training_job)


# =============================================================================
# The research comparison round's flow, carried by the acceptance (L1-L8 in M)
# =============================================================================
#: The round's probes in phase M (research comparison §5 Stage 10; the owner's
#: O9), in the order they run, every one required (`M2_DEMO_REQUIRED`).
RC_DEMO_PROBES: Tuple[str, ...] = (
    "L1", "L2", "L3", "L4.other_graph", "L4.tampered_pairs", "L4.other_view",
    "L4.split_record", "L5", "L6", "L7.disease", "L7.gene", "L7.patients_like_me",
    "L7.disease_frozen", "L7.branch", "L8",
)
#: The KG link-prediction protocol's name (`src.evaluation.protocols`).
LP_PROTOCOL = "kg_link_prediction"
#: The three task protocols L7's regression cases each train a short child for.
RC_TASKS: Tuple[str, ...] = ("disease", "gene", "patients_like_me")
#: The flow's own deployment under <work>/rc: one workspace under its own root,
#: a copy of the demo workspace whose runs list copies of D1's base and disease
#: child and L7's children, and one registry. Everything the flow measures is
#: listed there, so every ledger it writes is one the export reads.
RC_WORKSPACE = "probe-rc"
RC_BASE_RUN = "probe-rc-base"
RC_CHILD_RUN = "probe-rc-d1-disease"
RC_TASK_RUNS: Dict[str, str] = {task: "probe-rc-" + task.replace("_", "-") for task in RC_TASKS}
#: The run name of L7.branch's same-settings branch of the listed base (the
#: ops-wiring round, decisions B1, B2): part of the directory the job runner
#: allocates for it.
RC_BRANCH_RUN_NAME = "probe-rc-branch"
#: L7.disease_frozen's runs (ops-wiring round, decision F2): the
#: frozen-backbone disease child of the listed base, paused after
#: `RC_FROZEN_PAUSE_AFTER` of `RC_FROZEN_EPOCHS` epochs, and its resume in a run
#: of its own.
RC_FROZEN_RUNS: Dict[str, str] = {"paused": "probe-rc-disease-frozen",
                                  "resumed": "probe-rc-disease-frozen-resumed"}
RC_FROZEN_EPOCHS = 2
RC_FROZEN_PAUSE_AFTER = 1
#: What L7.disease_frozen states beside the demo's disease sizes: the freeze,
#: and float32 throughout (no mixed precision, no torch.compile), so the
#: check is an fp32 one and says so; AMP and compile with a freeze stay
#: outside it, recorded as not verified, and are never refused for it.
RC_FROZEN_SETTINGS: Dict[str, Any] = {"frozen_parts": "graph_side", "use_amp": False,
                                      "compile": False}
RC_CHECKPOINT = "last.pt"
#: The registry's versions: the KG edge set L1 imports from the listed base
#: (role test); the KG edge sets the refusals measure, each its own import
#: through the importer (another graph's; one whose pairs file is changed
#: after; one whose split record is made to contradict its graph after); a
#: supplied cohort (L5, L7.disease); and the workspace's own generated val
#: split with its functional gene fixture (L6, L7.gene, L7.patients_like_me).
RC_KG_LABEL = "probe-kg-test-1"
RC_KG_OTHER_GRAPH_LABEL = "probe-kg-other-graph-1"
RC_KG_TAMPERED_LABEL = "probe-kg-tampered-pairs-1"
RC_KG_SPLIT_LABEL = "probe-kg-split-record-1"
RC_SUPPLIED_LABEL = "probe-rc-supplied-1"
RC_GENERATED_LABEL = "probe-gen-val-1"
#: The relation a base of another graph view leaves out (L4.other_view): the
#: demo graph's one ortholog relation, so the view, its node and relation ids
#: and its split are another view's.
RC_OTHER_VIEW_EXCLUDES: Tuple[str, ...] = ("gene|human_mouse_ortholog|mouse_gene",)
#: How close the probe's own recomputation of an LP run's raw scores must come
#: to the scores file (`_lp_scores_verified`). On the CPU it is exact; CUDA's
#: message passing sums in an order it does not promise, so a recomputation
#: there may differ in the last bits, and the largest difference is recorded.
RC_SCORE_TOLERANCE: Dict[str, float] = {"rtol": 1e-4, "atol": 1e-5}
#: The export's names for its two sources (`scripts/export_results.py
#: --workspaces-root NAME=DIR --registry NAME=DIR`; two sources never share
#: one): what its rows show for them, never a path.
RC_EXPORT_SOURCES: Dict[str, str] = {"workspaces_root": "probe-rc-workspaces",
                                     "registry": "probe-rc-registry"}


def _kg_edges_argv(checkpoint: Path, role: str = "test") -> List[str]:
    """The importer's arguments for a KG edge set: the `role` part of the edge
    split `checkpoint`'s model record names (`_import_cohort`)."""
    return ["--kind", "kg_edges", "--from-checkpoint", str(checkpoint), "--role", role]


def _generated_argv(role: str) -> List[str]:
    """The importer's arguments for the workspace's own generated `role`
    split (`_import_cohort`)."""
    return ["--kind", "generated", "--role", role]


def _one_new_run(reports: Path, before: Sequence[str]) -> str:
    """The one run directory a measurement added to the report store
    `reports` (whose runs were `before`)."""
    created = sorted(set(_store_runs(reports)) - set(before))
    assert len(created) == 1, f"the run left {len(created)} report directories, not one"
    return created[0]


def _ran_on(manifest: Mapping[str, Any], device: str) -> None:
    """A kept report's measurement ran on the probe's `device`: its manifest
    records `cuda_executed` true exactly when that device is CUDA, the value
    the measurement command gives only to a run that executed there (it
    refuses `--device cuda` without CUDA)."""
    on_cuda = device.startswith("cuda")
    assert manifest.get("cuda_executed") is on_cuda, (
        f"the report records cuda_executed {manifest.get('cuda_executed')} for a run on {device}")


def _recorded_native_identity(report: Mapping[str, Any]) -> Dict[str, Any]:
    """The native identity a kept report's run recorded (its manifest's
    environment components), as every reader shows it: the record views' one
    projection (`native_identity`) through their model (`RecordComponent`),
    which is how an export row serialises it. What that run's own process
    recorded, never this one's."""
    from src.api.services.record_views import RecordComponent
    from src.utils.run_environment import native_identity

    components = report["manifest"]["environment"]["facts"]["components"]
    return {name: RecordComponent.model_validate(entry).model_dump(mode="json")
            for name, entry in native_identity(components).items()}


def _recorded_components(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Every fact of each component a kept report's run recorded (its
    manifest's environment components), as every reader passes them on: the
    record views' one projection (`recorded_components`) through their model
    (`RecordComponentFacts`), which is how an export row serialises them.
    What that run's own process recorded, never this one's."""
    from src.api.services.record_views import RecordComponentFacts
    from src.utils.run_environment import recorded_components

    components = report["manifest"]["environment"]["facts"]["components"]
    return {name: RecordComponentFacts.model_validate(entry).model_dump(mode="json")
            for name, entry in recorded_components(components).items()}


def _lp_scores_verified(run: Path, report: Mapping[str, Any], version: Any, checkpoint: Path,
                        workspace: Path, device: str) -> Dict[str, Any]:
    """**The scores file, the pairs file and the summary verify each other**
    (research comparison §7, LP artefacts), as the probe checks them apart
    from the measurement:

    - the scores file is the bytes the report names (digest, size, rows), keyed
      to the version's pairs file by its digest and the LP core's pairs digest,
      one row per pair in the pairs file's order, each row's state the one its
      raw scores give (`lp_score_states`);
    - its raw scores are the model's scores of those pairs, recomputed through
      the LP core (`score_lp_pairs`) over the version's message-passing graph
      (its recorded split's train part, rebuilt from the model's own view),
      within `RC_SCORE_TOLERANCE`;
    - the report's metrics are recomputed from the scores file alone, by the
      LP core's whole-set summary (`lp_set_metrics`) and the evaluator's status
      rule (`lp_metric_statuses`): the ranking metrics exactly, lp_loss (the
      head's hinge, which the run computed on its device) within float
      rounding.

    The pairs are read through the digest-checking reader
    (`read_kg_edge_pairs`). The scores file is read once (`read_once`), so
    the digest held to the report's is that of the bytes the LP core then
    parses (contract M2.1's rule for every reader). Nothing in the product
    reads it back (the measurement's digest is its writer's,
    `publish_chunks`): this read is the probe's own check of a file it had
    the measurement keep. Returns the evidence: counts, digests and the
    largest raw-score difference."""
    import math

    import numpy as np
    import torch

    from src.evaluation.cohort_registry import read_kg_edge_pairs
    from src.evaluation.task_measurement import LP_METRICS, lp_metric_statuses
    from src.kg.graph_view import rebuild_split
    from src.models.bundle.lp_head import (
        lp_score_states,
        lp_scores_from_table,
        lp_set_metrics,
        score_lp_pairs,
    )
    from src.models.bundle.record import load_bundle
    from src.utils.checkpoint_io import read_checkpoint
    from src.utils.fingerprint import read_once

    scores = report["scores"]
    read = read_once(run / scores["file"])
    data = read.data
    assert read.sha256 == scores["sha256"] and len(data) == scores["bytes"], (
        "the scores file is not the bytes the report names")
    pairs = read_kg_edge_pairs(version).pairs
    block = version.manifest["pairs"]
    assert scores["keyed_to"] == {"file": block["file"], "sha256": block["sha256"],
                                  "lp_pairs_digest": block["lp_pairs_digest"]}, (
        "the scores file is keyed to another pairs file than the version's")
    table = lp_scores_from_table(data, scores["layout"])
    n = int(pairs.head.numel())
    assert scores["rows"] == len(table) == block["rows"] == n, (
        f"the scores file holds {len(table)} rows for the version's {n} pairs")
    assert table["pair_row"].tolist() == list(range(n)), (
        "the scores file's rows are not the pairs file's, in its order")
    positive = torch.from_numpy(table["positive_raw"].copy())
    negative = torch.from_numpy(table["negative_raw"].copy())
    states = lp_score_states(positive, negative)
    assert table["state"].tolist() == states.tolist(), (
        "a row's state is not the one its raw scores give")

    # The model's raw scores of those pairs, recomputed through the LP core
    # over the version's message-passing graph, on the run's device.
    view = _child_view(checkpoint, workspace)
    payload = read_checkpoint(checkpoint, map_location="cpu", weights_only=True).checkpoint
    bundle, _record = load_bundle(payload, view_digest=view.digest)
    run_device = torch.device(device)
    bundle.to(run_device)
    split = rebuild_split(view, version.manifest["split"])
    passing = split.mask("train")
    again = score_lp_pairs(bundle, pairs, view.edge_index(passing), view.rel[passing],
                           run_device, node_type=view.node_type,
                           pair_chunk=report["manifest"]["batch_size"])
    sides = ((table["positive_raw"], again.positive.numpy()),
             (table["negative_raw"], again.negative.numpy()))
    # A score that is not finite is the same state on both sides, never a
    # difference (`lp_score_states` judged the stored ones above).
    differences = [float(np.nanmax(np.abs(stored - computed))) if n else 0.0
                   for stored, computed in sides]
    close = all(np.allclose(stored, computed, equal_nan=True, **RC_SCORE_TOLERANCE)
                for stored, computed in sides)
    assert close, (f"the raw scores in the scores file are not the model's scores of the pairs "
                   f"(largest difference {max(differences):.3g})")

    # The summary, from the scores file alone, by the LP core's one rule.
    recomputed = lp_set_metrics(bundle.lp_head, positive, negative)
    values, _statuses = lp_metric_statuses(recomputed, pairs=n, not_finite=int(states.sum()))
    stated = report["authoritative_metrics"]
    assert set(stated) == set(values), (
        f"the report gives the metrics {sorted(stated)}; the scores file gives {sorted(values)}")
    for name in values:
        same = (math.isclose(stated[name], values[name], rel_tol=1e-5, abs_tol=1e-7)
                if name == "lp_loss" else stated[name] == values[name])
        assert same, f"the report's {name} is not the one its scores file gives"
    return {"rows": n, "scores_sha256": scores["sha256"], "pairs_sha256": block["sha256"],
            "lp_pairs_digest": block["lp_pairs_digest"],
            "pairs_not_finite": int(states.sum()),
            "metrics_recomputed_from_the_scores": sorted(values),
            "metrics_defined": sorted(LP_METRICS),
            "raw_scores_recomputed": "through the LP core over the version's message-passing "
                                     "graph, on the run's device",
            "largest_raw_score_difference": max(differences)}


def _lp_run_facts(run: Path, version: Any, checkpoint: Path, checkpoint_sha256: str,
                  workspace: Path, device: str, *, stage: str,
                  exposure_state: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """A kept KG link-prediction run of the model `checkpoint` on the KG edge
    set `version`: its report parsed and identified from one read, measured by
    the protocol on that version's part, with no R10, on the probe's `device`
    (`_ran_on`); its exposure block scoped to the model's own record (its
    definition version, its sources: the model record by the checkpoint's
    digest and stage, the version by its label, kind, role, digest and split)
    and reading `exposure_state`'s words; and its scores file, pairs and
    summary verified against each other (`_lp_scores_verified`). Returns the
    evidence, and the report as parsed."""
    from src.evaluation.caveats import EXPOSURE_DEFINITION_VERSION, EXPOSURE_READINGS
    from src.evaluation.cohort_registry import REPORT_FILE
    from src.kg.storage.file_storage import _read_json

    report, identity = _read_json(run / REPORT_FILE)
    manifest = report["manifest"]
    role = version.manifest["role"]
    assert (manifest["mode"], manifest["cohort_kind"], manifest["split"]) == (
        LP_PROTOCOL, "kg_edges", role), (
        f"the report measured {manifest['mode']} on a {manifest['cohort_kind']} "
        f"{manifest['split']} part, not KG link prediction on the version's {role} part")
    assert manifest["cohort_version"]["manifest_sha256"] == version.identity.sha256, (
        "the report names another KG edge set")
    assert manifest["artifact_digests"]["checkpoint"] == checkpoint_sha256, (
        "the report does not name the checkpoint it was given")
    assert manifest["r10"] is None, "a KG edge set's report records R10"
    _ran_on(manifest, device)
    exposure = report["exposure"]
    sources = exposure.get("sources") or {}
    model_source = sources.get("model_record") or {}
    assert exposure.get("definition_version") == EXPOSURE_DEFINITION_VERSION, (
        f"the exposure block is of definition {exposure.get('definition_version')!r}")
    assert (model_source.get("checkpoint_sha256"), model_source.get("learning_stage")) == (
        checkpoint_sha256, stage), "the exposure block names another model record"
    assert sources.get("version") == {
        "label": version.manifest["label"], "kind": "kg_edges", "role": role,
        "manifest_sha256": version.identity.sha256,
        "split_digest": version.manifest["split"]["digest"]}, (
        "the exposure block names another version")
    assert (exposure.get("state"), exposure.get("reading")) == (
        exposure_state, EXPOSURE_READINGS[exposure_state]), (
        f"the exposure reads {exposure.get('state')!r}, not {exposure_state!r}")
    return {
        "run_id": run.name, "report_sha256": identity.sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "version": {"label": version.manifest["label"], "kind": "kg_edges", "role": role,
                    "manifest_sha256": version.identity.sha256},
        "exposure": {"definition_version": exposure["definition_version"],
                     "state": exposure["state"], "reading": exposure["reading"],
                     "measured_edges": (exposure.get("measured_edges") or {}).get("n")},
        "n_ranked": report["n_ranked"], "metrics": report["authoritative_metrics"],
        "scores": _lp_scores_verified(run, report, version, checkpoint, workspace, device),
        "cuda_executed": manifest["cuda_executed"],
    }, report


def _registered_record(checkpoint: Path, report_sha256: str, run_id: str, version: Any,
                       checkpoint_sha256: str) -> Dict[str, Any]:
    """The record the ledger beside `checkpoint` holds for the kept run
    `run_id` of `version`: found through the one ledger reader and read as its
    kind's view by the readers' one listing (`read_records`), built from the
    report's bytes, naming the version. Found by its run, since two runs
    whose reports are equal bytes are two records (research comparison §7).
    Returns its evidence."""
    from src.api.services.record_views import read_records
    from src.evaluation.sidecar import LEDGER_FILENAME, read_ledger

    ledger, ledger_sha256 = read_ledger(checkpoint.resolve().parent / LEDGER_FILENAME)
    assert ledger_sha256 is not None, "no ledger was written beside the checkpoint"
    viewed, unreadable = read_records(ledger, checkpoint_sha256)
    assert not unreadable, f"{len(unreadable)} records for the checkpoint cannot be read"
    # By its run: two runs whose reports are equal bytes are two records.
    found = [(record, view) for record, view in viewed if record.get("run_id") == run_id]
    assert len(found) == 1, f"the ledger holds {len(found)} records of the kept run, not one"
    record, view = found[0]
    assert (record.get("source_artifact_digest"), record.get("cohort_digest")) == (
        report_sha256, version.identity.sha256), (
        "the run's record was not built from the kept report's bytes, or names another version")
    return {"sequence": record["registration"]["sequence"], "run_id": record["run_id"],
            "mode": view.mode, "view": type(view).__name__, "ledger_sha256": ledger_sha256}


def _export_row_verified(row: Mapping[str, Any], entry: Mapping[str, Any]) -> None:
    """One row of the results export (L8) holds what the flow kept for its run
    (`entry`: the report's, the measured checkpoint's and the version's
    digests, the data kind and role, and the native identity and every
    component fact the run's report recorded, `_recorded_native_identity`,
    `_recorded_components`):

    - those identities, and the run kept and registered;
    - the recorded training settings, rules, learning and effective settings
      and loss identity, the environment class, the metric and its status; a
      finetune's weight transfer listing its parts, the frozen parts' forward
      mode and the downstream start (Stage 9b);
    - the native identity and every component fact its run recorded (each
      state, the fallback reason and the source's facts), as the record
      views show them;
    - a KG edge set's units and denominators, and the absence said for
      another kind's;
    - the exposure by kind: a KG edge set's state and reading, a supplied
      cohort's R10, a generated split's allocation note."""
    from src.api.services import results
    from src.evaluation.caveats import AFTER_FULL_VIEW_EXPOSURE, HELD_OUT

    probe_id = entry["probe"]
    said = (row["report_sha256"], row["checkpoint_sha256"], row["data_version_sha256"],
            row["data_kind"], row["role"])
    assert said == (entry["report_sha256"], entry["checkpoint_sha256"],
                    entry["version_sha256"], entry["kind"], entry["role"]), (
        f"{probe_id}'s run is exported with other identities")
    assert (row["report"], row["registered"]) == (results.REPORT_KEPT, results.REGISTERED), (
        f"{probe_id}'s run is exported as {row['report']}, {row['registered']}")
    for column in ("training_rules", "learning_settings", "effective_settings",
                   "loss_identity", "environment_class", "metric_name", "value_status"):
        assert row[column] is not None, f"{probe_id}'s run is exported without its {column}"
    # A KG edge set's run records each metric's unit and denominator; another
    # kind's says it records none.
    assert (row["unit"] is not None and row["denominator"] is not None
            if entry["kind"] == "kg_edges" else
            row["unit"] is None and "metric.unit" in (row["unknown"] or {})), (
        f"{probe_id}'s run is exported with its units unsaid")
    transfer = row["weight_transfer"]
    assert row["learning_stage"] != "finetune" or (
        isinstance(transfer, dict) and transfer.get("parts") and "frozen_forward" in transfer
        and transfer.get("downstream_initial_sha256")), (
        f"{probe_id}'s finetune is exported without its weight transfer")
    assert row["native_identity"] == entry["native_identity"], (
        f"{probe_id}'s run is exported with another native identity than its report records")
    assert row["components"] == entry["components"], (
        f"{probe_id}'s run is exported with other component facts than its report records")
    exposure = row["exposure_state"]
    if entry["kind"] == "kg_edges":
        assert exposure in (HELD_OUT, AFTER_FULL_VIEW_EXPOSURE) and (
            row["exposure_reading"] and row["exposure_definition"]), (
            f"{probe_id}'s KG edge set run is exported without its exposure")
    elif entry["kind"] == "supplied":
        assert row["r10"] is not None, f"{probe_id}'s run is exported without its R10"
    else:
        assert exposure == results.EXPOSURE_BY_ALLOCATION, (
            f"{probe_id}'s generated split run reads exposure {exposure!r}")


def _frozen_arm_verified(base: Path, paused: Path, resumed: Path, device: str, *,
                         full: Optional[Path] = None) -> Dict[str, Any]:
    """**The frozen-backbone control's functional check, fp32 only**
    (L7.disease_frozen; ops-wiring round, decision F2): `paused`, a disease
    child of `base` trained with `frozen_parts` graph_side and paused, and
    `resumed`, its resume in a run of its own, each read by the product's
    readers (`read_checkpoint`, `load_bundle`), checked in this order:

    - (a) **precision**: mixed precision off in effect (float32) and
      torch.compile not requested, in both files: the check covers fp32 only,
      and says so; AMP and compile with a freeze are not verified by it;
    - (b) **the record**: the weight transfer marks the graph side
      (`recipe.FROZEN_PARTS["graph_side"]`) frozen and the downstream parts
      trainable, the frozen parts' forward mode is `FROZEN_FORWARD`, the loss
      identity names the LP term, and only it, as a term without a trainable
      parameter, and the learning setting is graph_side;
    - (c) **the frozen state unchanged**: the graph side's parameters and
      buffers in each file equal the base's, key by key (the first difference
      named) and by `ModelBundle.parts_sha256`, with at least one BatchNorm
      buffer among them;
    - (d) **the downstream parts trained, in each run**: their digest in each
      file differs from the `downstream_initial_sha256` its record names, and
      the resumed file's from the paused file's (its `last.pt` is written at
      its last epoch's end, before early stopping could restore another, so a
      resume that learned nothing leaves them equal);
    - (e) **the optimizer holds the downstream parameters**: its groups hold
      as many parameters as the downstream parts have, each once, each with
      a saved state (a step was taken for it), and the saved moments' shapes
      are the downstream parameters' shapes, counted (a multiset: a
      graph-side parameter in place of a downstream one of another shape
      fails, though its shape is a downstream one's too), paused and resumed;
    - (f) **the resume**: `resumed` records operation resume, the paused
      file's weight transfer, the paused file's digest as its parent, and its
      stage's epochs completed, after the paused file's one;
    - (g) **one start for both arms**: `full`, the full arm of the same base
      (L7.disease), records the same `downstream_initial_sha256`, or "not
      compared" when it is None;
    - (h) **the device**: each file's environment names a GPU exactly when
      `device` is CUDA (the measurement's `cuda_executed` is `_ran_on`'s).

    Raises AssertionError naming what does not hold. Returns the facts."""
    from collections import Counter

    import torch

    from src.config.recipe import FROZEN_PARTS
    from src.models.bundle.builder import DOWNSTREAM_PARTS, FROZEN_FORWARD, GRAPH_SIDE_PARTS
    from src.models.bundle.record import load_bundle
    from src.utils.checkpoint_io import read_checkpoint

    def read(path: Path, stage: str) -> Tuple[Any, Any, Mapping[str, Any], str]:
        loaded = read_checkpoint(path, map_location="cpu", weights_only=True)
        bundle, record = load_bundle(loaded.checkpoint, learning_stage=stage,
                                     task="disease" if stage == "finetune" else None)
        return bundle, record, loaded.checkpoint, loaded.identity.sha256

    base_bundle, _base_record, _base_payload, base_sha = read(base, "base")
    arms = {"paused": read(paused, "finetune"), "resumed": read(resumed, "finetune")}
    frozen = tuple(FROZEN_PARTS["graph_side"])
    # (a) fp32 only: no mixed precision in effect, no torch.compile.
    for which, (_bundle, _record, payload, _sha) in arms.items():
        effective = (payload.get("config") or {}).get("effective") or {}
        amp, compiled = effective.get("amp") or {}, effective.get("compile") or {}
        assert amp.get("enabled") is False and amp.get("dtype") == "float32", (
            f"the {which} file ran with mixed precision {amp.get('dtype')!r} in effect: not the "
            "fp32 check (AMP with a freeze is not verified here)")
        assert compiled.get("requested") is False, (
            f"the {which} file asked for torch.compile: not the fp32 check (compile with a "
            "freeze is not verified here)")
    # (b) the record names the freeze it trained with.
    for which, (bundle, record, payload, _sha) in arms.items():
        transfer = record.weight_transfer or {}
        expected = {part: "frozen" if part in frozen else "trainable" for part in bundle.parts}
        assert transfer.get("parts") == expected, (
            f"the {which} file's record does not mark the graph side frozen and the downstream "
            f"parts trainable: {transfer.get('parts')}")
        assert transfer.get("frozen_forward") == FROZEN_FORWARD, (
            f"the {which} file records frozen_forward {transfer.get('frozen_forward')!r}")
        untrained = (record.loss_identity or {}).get("terms_without_trainable_parameters")
        assert untrained == ["lp"], (
            f"the {which} file's loss identity names {untrained!r} as the terms without a "
            "trainable parameter, not ['lp']")
        setting = ((payload.get("config") or {}).get("requested") or {}).get("frozen_parts")
        assert setting == "graph_side", (
            f"the {which} file's learning setting frozen_parts is {setting!r}")
    # (c) the graph side is the base's, every parameter and buffer.
    base_state = base_bundle.state_dict()
    graph_keys = sorted(key for key in base_state if key.split(".", 1)[0] in GRAPH_SIDE_PARTS)
    buffers = [key for key in graph_keys
               if key.endswith(("running_mean", "running_var", "num_batches_tracked"))]
    assert buffers, "the base's graph side holds no BatchNorm buffer, so the check is vacuous"
    base_digest = base_bundle.parts_sha256(GRAPH_SIDE_PARTS)
    for which, (bundle, _record, _payload, _sha) in arms.items():
        state = bundle.state_dict()
        own = sorted(key for key in state if key.split(".", 1)[0] in GRAPH_SIDE_PARTS)
        assert own == graph_keys, f"the {which} file's graph side holds other entries"
        changed = next((key for key in graph_keys
                        if not torch.equal(state[key].cpu(), base_state[key].cpu())), None)
        assert changed is None, f"the {which} file's graph side changed: {changed} is not the base's"
        assert bundle.parts_sha256(GRAPH_SIDE_PARTS) == base_digest, (
            f"the {which} file's graph-side digest is not the base's")
    # (d) the downstream parts trained.
    downstream = {}
    for which, (bundle, record, _payload, _sha) in arms.items():
        start = (record.weight_transfer or {}).get("downstream_initial_sha256")
        downstream[which] = bundle.parts_sha256(DOWNSTREAM_PARTS)
        assert start and downstream[which] != start, (
            f"the {which} file's downstream parts are still those training started from")
    assert downstream["resumed"] != downstream["paused"], (
        "the resumed file's downstream parts are the paused file's: the resumed run trained "
        "nothing")
    # (e) the optimizer holds exactly the downstream parameters.
    for which, (bundle, _record, payload, _sha) in arms.items():
        named = [(name, parameter) for name, parameter in bundle.named_parameters()]
        wanted = [parameter for name, parameter in named
                  if name.split(".", 1)[0] in DOWNSTREAM_PARTS]
        optimizer = payload.get("optimizer_state_dict") or {}
        held = [index for group in optimizer.get("param_groups") or []
                for index in group.get("params") or []]
        assert len(held) == len(set(held)) == len(wanted), (
            f"the {which} file's optimizer groups hold {len(held)} parameters, not the "
            f"{len(wanted)} of the downstream parts")
        saved = optimizer.get("state") or {}
        moments = [(saved.get(index) or {}).get("exp_avg") for index in held]
        stepless = sum(moment is None for moment in moments)
        assert not stepless, (
            f"{stepless} of the parameters the {which} file's optimizer holds have no saved "
            "state: no step was taken for them")
        surplus = sorted((Counter(tuple(moment.shape) for moment in moments)
                          - Counter(tuple(parameter.shape) for parameter in wanted)).elements())
        assert not surplus, (
            f"the {which} file's optimizer holds a parameter of shape {surplus[0]} beyond the "
            "downstream parts' parameters of that shape")
    # (f) the resume kept the stage's record and went on from the paused file.
    _pb, paused_record, paused_payload, paused_sha = arms["paused"]
    _rb, resumed_record, resumed_payload, resumed_sha = arms["resumed"]
    paused_epochs = (paused_payload.get("progress") or {}).get("completed_epochs")
    resumed_epochs = (resumed_payload.get("progress") or {}).get("completed_epochs")
    target = ((resumed_payload.get("progress") or {}).get("stage") or {}).get("target_epochs")
    assert resumed_record.operation == "resume", (
        f"the resumed file records operation {resumed_record.operation!r}")
    assert resumed_record.weight_transfer == paused_record.weight_transfer, (
        "the resumed file's weight transfer is not the paused file's")
    assert (resumed_record.parent or {}).get("sha256") == paused_sha, (
        "the resumed file does not name the paused file's bytes as its parent")
    assert paused_record.parent and paused_record.parent.get("sha256") == base_sha, (
        "the paused file does not name the base's bytes as its parent")
    assert (paused_epochs, resumed_epochs) == (RC_FROZEN_PAUSE_AFTER, target) and (
        target == RC_FROZEN_EPOCHS), (
        f"the runs completed {paused_epochs} and {resumed_epochs} epochs, not "
        f"{RC_FROZEN_PAUSE_AFTER} and {RC_FROZEN_EPOCHS}")
    # (g) both arms of the base start from one downstream initialisation.
    start = paused_record.weight_transfer["downstream_initial_sha256"]
    if full is None:
        same_start: Any = "not compared"
    else:
        _fb, full_record, _fp, _fs = read(full, "finetune")
        full_start = (full_record.weight_transfer or {}).get("downstream_initial_sha256")
        assert full_start == start, (
            "the full arm records another downstream start than the frozen arm "
            f"({str(full_start)[:12]}, not {start[:12]})")
        same_start = True
    # (h) the device: a GPU named exactly when the probe's device is CUDA.
    on_cuda = device.startswith("cuda")
    for which, (_bundle, _record, payload, _sha) in arms.items():
        gpu = (payload.get("environment") or {}).get("gpu")
        assert (gpu is not None) == on_cuda, (
            f"the {which} file's environment {'names no' if on_cuda else 'names a'} GPU for a "
            f"run on {device}")
    return {
        "precision": "fp32 only: mixed precision off and torch.compile not requested; AMP and "
                     "compile with a freeze are not verified by this check",
        "frozen_parts": "graph_side", "parts": paused_record.weight_transfer["parts"],
        "frozen_forward": paused_record.weight_transfer["frozen_forward"],
        "terms_without_trainable_parameters": ["lp"],
        "graph_side": {"base_sha256": base_digest, "paused_equal": True, "resumed_equal": True,
                       "entries": len(graph_keys), "batch_norm_buffers": len(buffers)},
        "downstream": {"initial_sha256": start, "paused_sha256": downstream["paused"],
                       "resumed_sha256": downstream["resumed"]},
        "optimizer_holds_only_downstream_parameters": True,
        "paused": {"sha256": paused_sha, "completed_epochs": paused_epochs,
                   "parent_sha256": base_sha},
        "resumed": {"sha256": resumed_sha, "operation": resumed_record.operation,
                    "completed_epochs": resumed_epochs, "parent_sha256": paused_sha,
                    "weight_transfer_kept": True},
        "same_downstream_start_as_the_full_arm": same_start,
        "trained_on_cuda": on_cuda,
    }


def _rc_demo(report: Report, work: Path, sound: Path, base: Path, child: Path, device: str,
             other: Optional[Path]) -> None:
    """The research comparison round's flow on the demo workspace (L1-L8),
    after M5's, on the probe's own deployment under <work>/rc: its settings
    (its workspaces root, which names its GPU lock, its registry and the
    device) are the M5 settings in effect while the flow runs, every other
    one unset, as M5's are (`_deployment_settings`).
    `base` and `child` are D1's base and disease child, whose bytes are copied
    into the flow's workspace and listed there; `other` is M1's same-shaped
    workspace over another graph (None when M1 did not pass).

    Functional only: every step through its production entry point (the
    importer, the measurement command, the job runner, the trainer, the
    fixture derivation and the export command), the outputs checked against
    what each left on disk. The figures say nothing of a model's quality, and
    no two of them are compared.
    """
    print("   Research comparison: KG link prediction, its refusals, two protocols, a "
          "generated split, the GPU regression cases, the frozen-backbone control, a branch, "
          "the export")
    from scripts.setup_demo import demo_task_training
    from src.api.services.training_manager import TrainingManager
    from src.evaluation.caveats import AFTER_FULL_VIEW_EXPOSURE, HELD_OUT
    from src.evaluation.cohort_registry import (
        REPORT_FILE,
        CohortBindingMismatch,
        CohortVersionRefused,
    )
    from src.evaluation.registration import report_state
    from src.models.bundle.record import read_model_record
    from src.utils.checkpoint_io import read_checkpoint
    from src.utils.checkpoint_paths import RUNS_DIRNAME, run_checkpoint_dir

    rc = work / "rc"
    root, registry = rc / "workspaces", rc / "registry"
    settings = {"SHEPHERD_WORKSPACES_ROOT": str(root), "SHEPHERD_REGISTRY": str(registry),
                "SHEPHERD_DEVICE": device}
    state: Dict[str, Any] = {}
    #: Each run the flow kept and registered, for L8: by run id, the report's
    #: digest, the measured checkpoint's, the version's, the probe's id, and
    #: the native identity and component facts the run's report recorded.
    kept: List[Dict[str, Any]] = []

    def _need(*keys: str) -> None:
        names = {"workspace": "the flow's workspace (L1)", "kg": "the KG edge set (L1)",
                 "supplied": "the supplied cohort (L5)", "generated": "the generated split (L6)",
                 "fixture": "the functional gene fixture (L6)"}
        missing = [key for key in keys if key not in state]
        assert not missing, f"{names.get(missing[0], missing[0])} is not available"

    def _reports(label: str) -> Path:
        return registry / "versions" / label / "reports"

    def _keep(probe_id: str, run_id: str, report_sha256: str, checkpoint_sha256: str,
              version: Any, record: Mapping[str, Any], kept_report: Mapping[str, Any]) -> None:
        kept.append({"probe": probe_id, "run_id": run_id, "report_sha256": report_sha256,
                     "checkpoint_sha256": checkpoint_sha256,
                     "version_sha256": version.identity.sha256, "kind": version.kind,
                     "role": version.manifest["role"], "sequence": record["sequence"],
                     "native_identity": _recorded_native_identity(kept_report),
                     "components": _recorded_components(kept_report)})

    def _measured_and_registered(probe_id: str, checkpoint: Path, checkpoint_sha256: str,
                                 label: str, version: Any, protocol: str
                                 ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """`checkpoint` measured by `protocol` on `version` through the
        measurement command with --record: one run kept in the version's
        store, its report naming the checkpoint, the version and the protocol,
        run on the probe's device (`_ran_on`), and one record of it beside the
        checkpoint, read as its kind's view. Returns the evidence, and the
        report as parsed from its one read (`_read_json`), so a caller checks
        what else it holds without reading the file again."""
        from src.kg.storage.file_storage import _read_json

        reports = _reports(label)
        before = _store_runs(reports)
        code = _measure_cohort(checkpoint, state["workspace"], registry, label, device,
                               "--record", protocol=protocol)
        assert code == 0, f"the measurement with --record exited {code}"
        run_id = _one_new_run(reports, before)
        kept_report, identity = _read_json(reports / run_id / REPORT_FILE)
        manifest = kept_report["manifest"]
        assert (manifest["mode"], manifest["cohort_kind"]) == (protocol, version.kind), (
            f"the report measured {manifest['mode']} on a {manifest['cohort_kind']} version, "
            f"not {protocol} on a {version.kind} one")
        assert manifest["cohort_version"]["manifest_sha256"] == version.identity.sha256, (
            "the report names another version")
        assert manifest["artifact_digests"]["checkpoint"] == checkpoint_sha256, (
            "the report does not name the checkpoint it was given")
        _ran_on(manifest, device)
        record = _registered_record(checkpoint, identity.sha256, run_id, version,
                                    checkpoint_sha256)
        _keep(probe_id, run_id, identity.sha256, checkpoint_sha256, version, record, kept_report)
        return {"run_id": run_id, "report_sha256": identity.sha256,
                "protocol": protocol, "n_ranked": kept_report.get("n_ranked"),
                "metrics": kept_report.get("authoritative_metrics"),
                "cuda_executed": manifest["cuda_executed"], "record": record}, kept_report

    # ---- L1: the flow's deployment and a KG edge set imported from the base
    def _kg_edge_set() -> Tuple[str, Dict[str, Any]]:
        from src.evaluation.cohort_registry import read_kg_edge_pairs

        workspace = _workspace_copy(sound, root / RC_WORKSPACE)
        runs = workspace / RUNS_DIRNAME / _demo_encoder_kind()
        for name, run, source in (("base", RC_BASE_RUN, base), ("child", RC_CHILD_RUN, child)):
            target = run_checkpoint_dir(runs / run) / RC_CHECKPOINT
            target.parent.mkdir(parents=True)
            shutil.copyfile(source, target)
            state[name], state[f"{name}_sha"] = target, _read_run_checkpoint(target)["sha256"]
        state["workspace"] = workspace
        registry.mkdir(parents=True)
        version, preview = _import_version(registry, workspace, None, RC_KG_LABEL,
                                           kind_argv=_kg_edges_argv(state["base"]))
        manifest = version.manifest
        assert (version.kind, manifest["role"]) == ("kg_edges", "test"), (
            f"the import is a {version.kind} version of role {manifest['role']}")
        bound, _, _, record = verify_workspace(workspace)
        assert manifest["graph"] == bound, "the KG edge set's graph digests are not the workspace's"
        assert manifest["dataset_record"]["sha256"] == record.summary()["sha256"], (
            "the KG edge set is bound to another dataset record than the workspace's")
        model = read_model_record(read_checkpoint(state["base"], map_location="cpu",
                                                  weights_only=True).checkpoint)
        assert manifest["graph_view"]["digest"] == model.graph_view.digest, (
            "the KG edge set's graph view is not the base's")
        assert manifest["split"]["digest"] == model.data_roles["split"]["digest"], (
            "the KG edge set's split is not the one the base records")
        assert manifest["taken_from"]["checkpoint_sha256"] == state["base_sha"], (
            "the KG edge set names another checkpoint than the listed base")
        pairs = read_kg_edge_pairs(version).pairs
        rows = int(pairs.head.numel())
        assert rows == manifest["counts"]["pairs"] > 0, (
            f"the pairs file holds {rows} pairs; the version counts {manifest['counts']['pairs']}")
        state["kg"] = version
        return ("previewed with nothing written, committed by the printed id, the base's view "
                "and split, its test part's pairs stored"), {
            "preview_wrote_nothing": True, "preview_id": preview,
            "kind": version.kind, "role": manifest["role"],
            "manifest_sha256": version.identity.sha256, "graph_is_the_workspaces": True,
            "graph_view_digest": manifest["graph_view"]["digest"],
            "split_digest": manifest["split"]["digest"],
            "taken_from_checkpoint_sha256": state["base_sha"],
            "pairs": {"sha256": manifest["pairs"]["sha256"],
                      "lp_pairs_digest": manifest["pairs"]["lp_pairs_digest"], "rows": rows},
            "counts": manifest["counts"],
        }

    # ---- L2: the protocol through the measurement command, kept and registered
    def _lp_kept_and_registered(probe_id: str, checkpoint: Path, checkpoint_sha256: str, *,
                                stage: str, exposure_state: str) -> Dict[str, Any]:
        """`checkpoint` measured by the KG link-prediction protocol on L1's
        set through the measurement command with --record: one run kept, its
        report and outputs checked (`_lp_run_facts`), and its record beside
        the checkpoint (`_registered_record`). Returns the evidence."""
        version, reports = state["kg"], _reports(RC_KG_LABEL)
        before = _store_runs(reports)
        code = _measure_cohort(checkpoint, state["workspace"], registry, RC_KG_LABEL, device,
                               "--record", protocol=LP_PROTOCOL)
        assert code == 0, f"the measurement with --record exited {code}"
        run_id = _one_new_run(reports, before)
        facts, kept_report = _lp_run_facts(reports / run_id, version, checkpoint,
                                           checkpoint_sha256, state["workspace"], device,
                                           stage=stage, exposure_state=exposure_state)
        record = _registered_record(checkpoint, facts["report_sha256"], run_id, version,
                                    checkpoint_sha256)
        _keep(probe_id, run_id, facts["report_sha256"], checkpoint_sha256, version, record,
              kept_report)
        return {**facts, "record": record}

    def _lp_measured() -> Tuple[str, Dict[str, Any]]:
        _need("workspace", "kg")
        facts = _lp_kept_and_registered("L2", state["base"], state["base_sha"], stage="base",
                                        exposure_state=HELD_OUT)
        assert facts["record"]["sequence"] == 1, (
            f"the record is the ledger's {facts['record']['sequence']}th")
        return ("measured, kept and registered; held out as the base's record says; the scores, "
                "pairs and summary agree"), {
            "measured": {"model": "D1's base, listed in the flow's workspace",
                         "run": RC_BASE_RUN, "protocol": LP_PROTOCOL}, **facts,
        }

    # ---- L3: the protocol through the job runner, kept and registered
    def _lp_through_the_runner() -> Tuple[str, Dict[str, Any]]:
        _need("workspace", "kg")
        version, reports = state["kg"], _reports(RC_KG_LABEL)
        inputs = _listed_test_inputs(RC_KG_LABEL, RC_WORKSPACE, RC_BASE_RUN, RC_CHECKPOINT,
                                     protocol=LP_PROTOCOL, record=True)
        assert (inputs.checkpoint, inputs.checkpoint_sha256, inputs.learning_stage) == (
            state["base"], state["base_sha"], "base"), (
            "the listed base is not listed by its run, its digest as read and its record")
        assert inputs.evidence_dirs == (), "R10's evidence is chosen for a KG edge set"
        before = _store_runs(reports)
        manager = TrainingManager()
        with _job_ended_on_exit(manager, "test"):
            started = manager.start_test(inputs)
            assert started.get("success"), "the runner refused the test run"
            word = _job_ended(manager, "test", _TEST_RUN_SECONDS)
        status = manager.get_test_status()
        if word != "completed" or status["exit_code"] != 0:
            _job_said(manager, "test")
        assert word == "completed" and status["exit_code"] == 0, (
            f"the test run ended {word}, exit {status['exit_code']}")
        run_id = _one_new_run(reports, before)
        outcome = status["outcome"] or {}
        assert outcome.get("report_dir") == str(reports / run_id), (
            "the runner's outcome names another report directory than the run created")
        published = (outcome.get("run_outcome") or {}).get("report_sha256")
        # **Kept and registered as the publication contract says**, never
        # inferred from the exit status: the run was asked to record.
        assert (outcome.get("kept"), outcome.get("registered")) == (True, True), (
            f"the outcome says kept {outcome.get('kept')}, registered {outcome.get('registered')}")
        facts, kept_report = _lp_run_facts(reports / run_id, version, state["base"],
                                           state["base_sha"], state["workspace"], device,
                                           stage="base", exposure_state=HELD_OUT)
        assert facts["report_sha256"] == published, (
            "the outcome file's digest is not the kept report's bytes")
        record = _registered_record(state["base"], published, run_id, version,
                                    state["base_sha"])
        after = report_state(reports / run_id / REPORT_FILE, checkpoint=state["base"],
                             expected_sha256=published)
        assert (after.kept, after.registered) == ("kept", "registered"), (
            f"the report reads {after.kept}, {after.registered}")
        _keep("L3", run_id, published, state["base_sha"], version, record, kept_report)
        return "a test run from the server's lists: kept and registered, its outputs verified", {
            "inputs": "the version, workspace and checkpoint found in the server's lists again, "
                      "their digests passed as the expected ones",
            "measured": {"model": "D1's base, listed in the flow's workspace",
                         "run": RC_BASE_RUN, "protocol": LP_PROTOCOL},
            "status": word, "exit_code": status["exit_code"],
            "outcome_file_names_the_kept_report": True, "kept": True, "registered": True,
            **facts, "record": record,
            "report_state": {"kept": after.kept, "registered": after.registered},
        }

    # ---- L4: the refusals before anything is written
    def _lp_refused(label: str, checkpoint: Path, error: type,
                    fragment: Union[str, Tuple[str, ...]]
                    ) -> Tuple[Dict[str, Any], BaseException]:
        """`checkpoint` measured by the protocol on `label` with --record,
        refused with `error` naming `fragment`: the registry, the flow's
        workspace (the listed checkpoints' ledgers) and the checkpoint's own
        directory unchanged, no run directory made. Returns how many reads of
        a checkpoint and how many model builds the refusal came after
        (`_observing_checkpoint_reads`, `_observing_builds`), and the refusal,
        for what it carries besides its message."""
        watched = [registry, state["workspace"], checkpoint.resolve().parent]
        before = [snapshot(directory, include_dirs=True) for directory in watched]
        runs_before = _store_runs(_reports(label))
        reads: List[str] = []
        builds: List[str] = []
        with _observing_checkpoint_reads(reads), _observing_builds(builds):
            refusal = refuses(lambda: _measure_cohort(checkpoint, state["workspace"], registry,
                                                      label, device, "--record",
                                                      protocol=LP_PROTOCOL),
                              error, fragment)
        for taken, directory in zip(before, watched, strict=True):
            assert_untouched(taken, directory, "a refused measurement", include_dirs=True)
        assert _store_runs(_reports(label)) == runs_before, (
            "the refused measurement made a run directory")
        return {"checkpoint_reads": len(reads), "models_built": len(builds),
                "nothing_written": True}, refusal

    def _refuse_other_graph() -> Tuple[str, Dict[str, Any]]:
        _need("workspace", "kg")
        assert other is not None, "the same-shaped workspace (M1) is not available"
        # **A KG edge set of another graph**, made as any is: a base trained
        # on M1's same-shaped other graph and the importer taking its test
        # part there. Then measured over the flow's workspace.
        other_base = _last(_train_run(other, rc / "other_graph_base", device, stage="base",
                                      epochs=1, settings=demo_task_training()["base"]))
        _import_version(registry, other, None, RC_KG_OTHER_GRAPH_LABEL,
                        kind_argv=_kg_edges_argv(other_base))
        refusal, refused = _lp_refused(RC_KG_OTHER_GRAPH_LABEL, state["base"],
                                       CohortBindingMismatch, "mapped against another graph")
        assert refusal["checkpoint_reads"] == 0, "the checkpoint was read before the refusal"
        assert "kg" in refused.roles, f"the refusal names {refused.roles}, not the graph"
        return "refused by the binding before the checkpoint was read; nothing written", {
            **refusal, "refused_as": "the KG edge set is bound to another graph",
            "differing_roles": sorted(refused.roles)}

    def _refuse_tampered_pairs() -> Tuple[str, Dict[str, Any]]:
        _need("workspace", "kg")
        # Its own import of the base's test part, then one byte of its stored
        # pairs changed: the version's digest no longer binds the file.
        version, _ = _import_version(registry, state["workspace"], None, RC_KG_TAMPERED_LABEL,
                                     kind_argv=_kg_edges_argv(state["base"]))
        pairs = version.identity.path.parent / version.manifest["pairs"]["file"]
        data = bytearray(pairs.read_bytes())
        data[3] ^= 1
        pairs.write_bytes(bytes(data))
        refusal, _ = _lp_refused(RC_KG_TAMPERED_LABEL, state["base"], CohortVersionRefused,
                                 "is not the pairs file")
        assert refusal["checkpoint_reads"] == 0, "the checkpoint was read before the refusal"
        return "refused before the pairs were parsed or the checkpoint read; nothing written", {
            **refusal, "refused_as": "the pairs file is not the bytes the version binds"}

    def _refuse_other_view() -> Tuple[str, Dict[str, Any]]:
        _need("workspace", "kg")
        # **A model of another graph view**: a base trained on the flow's
        # workspace with the ortholog relation left out of its view. Its view
        # has other node and relation ids, and this build has no mapping
        # between views: a stated capability gap, not a research rule.
        sizes = demo_task_training()["base"]
        spec = {**sizes["model_spec"],
                "graph_view": {**(sizes["model_spec"].get("graph_view") or {}),
                               "exclude_relations": list(RC_OTHER_VIEW_EXCLUDES)}}
        other_view = _last(_train_run(state["workspace"], rc / "other_view_base", device,
                                      stage="base", epochs=1,
                                      settings={**sizes, "model_spec": spec}))
        refusal, _ = _lp_refused(RC_KG_LABEL, other_view, ValueError,
                                 ("this build has no node mapping between graph views",
                                  "a capability this build lacks, not a research rule"))
        assert refusal["models_built"] == 0, "a model was built before the refusal"
        return "refused as a capability this build lacks, before any model was built or " \
               "anything written", {
            **refusal, "refused_as": "a capability gap: no node mapping between graph views"}

    def _refuse_split_record() -> Tuple[str, Dict[str, Any]]:
        _need("workspace", "kg")
        # Its own import, then its cohort.json's split seed changed, and its
        # pairs' seed with it, so the manifest stays whole and consistent in
        # shape (the reader refuses a pairs seed that is not the split's): the
        # split record's seed and its recorded digest now contradict each
        # other on the view, so the split does not rebuild from the graph the
        # version binds. The manifest changed is the one the import's
        # `resolve_version` parsed from the file's bytes, never a second read.
        version, _ = _import_version(registry, state["workspace"], None, RC_KG_SPLIT_LABEL,
                                     kind_argv=_kg_edges_argv(state["base"]))
        held = version.manifest
        seed = int(held["split"]["seed"]) + 1
        manifest = {**held, "split": {**held["split"], "seed": seed},
                    "pairs": {**held["pairs"], "seed": seed}}
        version.identity.path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                         encoding="utf-8")
        refusal, _ = _lp_refused(RC_KG_SPLIT_LABEL, state["base"], ValueError,
                                 ("records an edge split that does not rebuild",
                                  "its record contradicts the graph it binds"))
        return "refused before anything was scored or written", {
            **refusal, "refused_as": "the version's split record does not rebuild: its seed "
                                     "and its digest contradict each other on its graph's view"}

    # ---- L5: a disease child by both of its protocols, two records
    def _two_protocols() -> Tuple[str, Dict[str, Any]]:
        _need("workspace", "kg")
        from src.api.services.record_views import read_records
        from src.evaluation.sidecar import LEDGER_FILENAME, read_ledger

        # A supplied cohort built from the demo graph's own ids, as T1's.
        source = _write_cohort_source(rc / "cases.json", _m4_demo_cases(state["workspace"]))
        supplied, _ = _import_version(registry, state["workspace"], source, RC_SUPPLIED_LABEL)
        state["supplied"] = supplied
        child_path, child_sha = state["child"], state["child_sha"]
        disease, _ = _measured_and_registered("L5", child_path, child_sha, RC_SUPPLIED_LABEL,
                                              supplied, "disease")
        # A finetune passed messages over the full view in its validation, so
        # its figure on the base's test part reads after full-view exposure.
        lp = _lp_kept_and_registered("L5", child_path, child_sha, stage="finetune",
                                     exposure_state=AFTER_FULL_VIEW_EXPOSURE)
        # **The one reader**: both records, independent, each read as its
        # kind's view, in the one ledger beside the child.
        ledger, _ = read_ledger(child_path.resolve().parent / LEDGER_FILENAME)
        viewed, unreadable = read_records(ledger, child_sha)
        assert not unreadable and len(viewed) == 2, (
            f"the one reader reads {len(viewed)} records and {len(unreadable)} unreadable")
        seen = sorted((view.mode, type(view).__name__, record["run_id"],
                       record["registration"]["sequence"]) for record, view in viewed)
        assert [mode for mode, *_ in seen] == ["disease", LP_PROTOCOL], (
            f"the records are of the modes {[mode for mode, *_ in seen]}")
        assert len({run for _m, _v, run, _s in seen}) == 2 and sorted(
            sequence for *_rest, sequence in seen) == [1, 2], (
            "the two records are not two runs registered one after the other")
        return "two independent records, one per protocol, read by the one reader", {
            "measured": {"model": "D1's disease child, listed in the flow's workspace",
                         "run": RC_CHILD_RUN, "checkpoint_sha256": child_sha},
            "disease": {**disease, "version": {"label": RC_SUPPLIED_LABEL, "kind": "supplied",
                                               "manifest_sha256": supplied.identity.sha256}},
            "kg_link_prediction": lp,
            "records": [{"mode": mode, "view": view, "sequence": sequence}
                        for mode, view, _run, sequence in seen],
        }

    # ---- L6: a generated split imported, measured and registered
    def _generated_split() -> Tuple[str, Dict[str, Any]]:
        _need("workspace")
        import io

        import scripts.derive_task_fixture as derive_task_fixture
        from scripts.setup_demo import DEMO_FIXTURE_DISTRACTORS

        # The demo workspace's functional gene fixture, through its own entry
        # point, so the version keeps it (Stage 5b) and L7 trains on it.
        with contextlib.redirect_stdout(io.StringIO()):
            assert derive_task_fixture.main(["--data-dir", str(state["workspace"]),
                                             "--distractors",
                                             str(DEMO_FIXTURE_DISTRACTORS)]) == 0, (
                "the fixture derivation failed")
        state["fixture"] = True
        version, preview = _import_version(registry, state["workspace"], None,
                                           RC_GENERATED_LABEL, kind_argv=_generated_argv("val"))
        manifest = version.manifest
        assert (version.kind, manifest["role"]) == ("generated", "val"), (
            f"the import is a {version.kind} version of role {manifest['role']}")
        bound, _, _, _ = verify_workspace(state["workspace"])
        assert manifest["graph"] == bound, "the generated split's graph digests are not the " \
                                           "workspace's"
        assert manifest.get("task_fixture"), "the generated split keeps no functional fixture"
        state["generated"] = version
        measured, kept_report = _measured_and_registered(
            "L6", state["child"], state["child_sha"], RC_GENERATED_LABEL, version, "disease")
        assert (kept_report["manifest"]["split"], kept_report["manifest"]["r10"]) == (
            "val", None), "the report is not of the generated val split, without R10"
        assert measured["record"]["view"] == "GeneratedRecordView", (
            f"the record reads as {measured['record']['view']}")
        return "imported with its fixture, measured, kept and registered as a generated split", {
            "preview_wrote_nothing": True, "preview_id": preview,
            "version": {"label": RC_GENERATED_LABEL, "kind": version.kind,
                        "role": manifest["role"], "manifest_sha256": version.identity.sha256,
                        "counts": manifest["counts"]},
            "measured": {"model": "D1's disease child, listed in the flow's workspace",
                         "run": RC_CHILD_RUN, "checkpoint_sha256": state["child_sha"]},
            **measured,
        }

    # ---- L7: the GPU regression cases, one short child per task
    def _regression_case(task: str) -> Callable[[], Tuple[str, Dict[str, Any]]]:
        def case() -> Tuple[str, Dict[str, Any]]:
            _need("workspace", *(("supplied",) if task == "disease" else
                                 ("generated", "fixture")))
            run = state["workspace"] / RUNS_DIRNAME / _demo_encoder_kind() / RC_TASK_RUNS[task]
            started = time.monotonic()
            trained = _last(_train_run(state["workspace"], run, device, stage="finetune",
                                       task=task, epochs=1,
                                       settings=demo_task_training()[task],
                                       finetune_from=state["base"]))
            seconds = round(time.monotonic() - started, 1)
            payload = read_checkpoint(trained, map_location="cpu", weights_only=True).checkpoint
            record = read_model_record(payload)
            sha = _read_run_checkpoint(trained)["sha256"]
            assert (record.learning_stage, record.task) == ("finetune", task), (
                f"the child is a {record.learning_stage} of task {record.task}")
            assert (record.parent or {}).get("sha256") == state["base_sha"], (
                "the child names another parent than the listed base")
            gpu = (payload.get("environment") or {}).get("gpu")
            on_cuda = device.startswith("cuda")
            assert (gpu is not None) == on_cuda, (
                f"the child's environment {'names no' if on_cuda else 'names a'} GPU for a run "
                f"on {device}")
            label, version = ((RC_SUPPLIED_LABEL, state["supplied"]) if task == "disease"
                              else (RC_GENERATED_LABEL, state["generated"]))
            measured, _ = _measured_and_registered(f"L7.{task}", trained, sha, label, version,
                                                   task)
            assert measured["record"]["sequence"] == 1, "the child's ledger held a record before"
            # Kept once the case passed: the frozen arm compares its start
            # with the full disease arm's (L7.disease_frozen).
            state[f"trained_{task}"] = trained
            return f"a {task} child trained on {device}, measured by its protocol and registered", {
                "task": task, "device": device, "trained_on_cuda": gpu is not None,
                "training_seconds": seconds, "epochs": 1,
                "sizes": "the demo's (setup_demo.demo_task_training)",
                "child": {"run": RC_TASK_RUNS[task], "sha256": sha,
                          "parent_sha256": state["base_sha"],
                          "training_rules": payload.get("training_rules")},
                "version": {"label": label, "kind": version.kind,
                            "manifest_sha256": version.identity.sha256},
                **measured,
            }

        return case

    # ---- L7.disease_frozen: the frozen-backbone disease child, fp32, paused and resumed
    def _frozen_disease_child() -> Tuple[str, Dict[str, Any]]:
        """A demo-sized disease child of the listed base with `frozen_parts`
        graph_side on the probe's device, in fp32 (`RC_FROZEN_SETTINGS`),
        paused after `RC_FROZEN_PAUSE_AFTER` of `RC_FROZEN_EPOCHS` epochs and
        resumed in a run of its own, through the production trainer
        (`_train_run`, `train_model.train`), checked by `_frozen_arm_verified`
        against the base and the full arm (L7.disease, when it passed); then
        the resumed child measured by the disease protocol on the supplied
        cohort and registered, kept for L8's export (ops-wiring round,
        decision F2)."""
        _need("workspace", "supplied")
        runs = state["workspace"] / RUNS_DIRNAME / _demo_encoder_kind()
        started = time.monotonic()
        paused = _last(_train_run(state["workspace"], runs / RC_FROZEN_RUNS["paused"], device,
                                  stage="finetune", task="disease", epochs=RC_FROZEN_EPOCHS,
                                  settings={**demo_task_training()["disease"],
                                            **RC_FROZEN_SETTINGS},
                                  finetune_from=state["base"], pause_after=RC_FROZEN_PAUSE_AFTER))
        resumed = _last(_train_run(state["workspace"], runs / RC_FROZEN_RUNS["resumed"], device,
                                   stage="finetune", task="disease", resume_from=paused))
        seconds = round(time.monotonic() - started, 1)
        verified = _frozen_arm_verified(state["base"], paused, resumed, device,
                                        full=state.get("trained_disease"))
        sha = verified["resumed"]["sha256"]
        measured, _ = _measured_and_registered("L7.disease_frozen", resumed, sha,
                                               RC_SUPPLIED_LABEL, state["supplied"], "disease")
        assert measured["record"]["sequence"] == 1, "the child's ledger held a record before"
        return (f"an fp32 frozen-backbone disease child trained on {device}, paused and resumed, "
                "its graph side the base's, measured and registered"), {
            "task": "disease", "device": device, "training_seconds": seconds,
            "epochs": RC_FROZEN_EPOCHS, "paused_after": RC_FROZEN_PAUSE_AFTER,
            "sizes": "the demo's (setup_demo.demo_task_training)",
            "settings": dict(RC_FROZEN_SETTINGS),
            "runs": {"paused": RC_FROZEN_RUNS["paused"], "resumed": RC_FROZEN_RUNS["resumed"]},
            "parent_sha256": state["base_sha"],
            "training_rules": read_checkpoint(resumed, map_location="cpu",
                                              weights_only=True).checkpoint.get("training_rules"),
            "version": {"label": RC_SUPPLIED_LABEL, "kind": state["supplied"].kind,
                        "manifest_sha256": state["supplied"].identity.sha256},
            **verified, **measured,
        }

    # ---- L7.branch: a same-settings branch of the listed base, through the job runner
    def _same_settings_branch() -> Tuple[str, Dict[str, Any]]:
        """A branch of the listed base that changes no setting (the
        ops-wiring round, decisions B1, B2), started through the job runner's
        one start (`start_training` with `branch_from`), bound to the base's
        digest as the listing read it and to the dataset's, and run on the
        probe's device to its end: a new run in a directory of its own, listed
        under its parent's exact file by the server's family view, its
        changes `{}`, a new optimizer, its stage from epoch 0; the parent's
        bytes unchanged; read back by the one loader."""
        from src.api.services import server_lists
        from src.kg.artifacts import read_split_manifest
        from src.models.bundle.record import load_bundle

        _need("workspace")
        fingerprint = read_split_manifest(state["workspace"]).identity.sha256
        runs = state["workspace"] / RUNS_DIRNAME / _demo_encoder_kind()
        before = sorted(path.name for path in runs.iterdir())
        manager = TrainingManager()
        started_at = time.monotonic()
        with _job_ended_on_exit(manager, "training"):
            started = manager.start_training(
                {"data_dir": str(state["workspace"]), "branch_from": str(state["base"]),
                 "device": device, "run_name": RC_BRANCH_RUN_NAME},
                expected_dataset_fingerprint=fingerprint,
                expected_parent_sha256=state["base_sha"])
            assert started.get("success"), "the runner refused the branch start"
            word = _job_ended(manager, "training", _TRAINING_RUN_SECONDS)
        seconds = round(time.monotonic() - started_at, 1)
        status = manager.get_status()
        if word != "completed" or status["exit_code"] != 0:
            _job_said(manager, "training")
        assert word == "completed" and status["exit_code"] == 0, (
            f"the branch ended {word}, exit {status['exit_code']}")
        run = Path(started["run_dir"])
        assert run.parent == runs and run.name not in before, (
            "the branch was not filed in a run directory of its own under its parent's "
            "architecture")
        assert _read_run_checkpoint(state["base"])["sha256"] == state["base_sha"], (
            "the parent's bytes changed")
        last = run_checkpoint_dir(run) / RC_CHECKPOINT
        payload = read_checkpoint(last, map_location="cpu", weights_only=True).checkpoint
        _bundle, record = load_bundle(payload, learning_stage="base")
        family = read_model_record(read_checkpoint(state["base"], map_location="cpu",
                                                   weights_only=True).checkpoint).family_id
        assert (record.operation, record.family_id) == ("branch", family), (
            "the branch is not recorded as a branch of its parent's family")
        assert record.parent == {"sha256": state["base_sha"], "run": RC_BASE_RUN,
                                 "changes": {}}, "the branch records another parent or changes"
        assert (record.weight_transfer or {}).get("optimizer") == "new", (
            "the branch did not start a new optimizer")
        stage = (payload.get("progress") or {}).get("stage") or {}
        assert (stage.get("kind"), stage.get("start_epoch")) == ("original", 0), (
            "the branch did not run its stage from epoch 0")
        gpu = (payload.get("environment") or {}).get("gpu")
        assert (gpu is not None) == device.startswith("cuda"), (
            f"the branch's environment {'names no' if device.startswith('cuda') else 'names a'} "
            f"GPU for a run on {device}")
        sha = _read_run_checkpoint(last)["sha256"]
        rows = server_lists.list_family(RC_WORKSPACE)
        assert rows.available, "the family view of the flow's workspace is unavailable"
        keys = {(row.item.run, row.item.filename): row for row in rows.items}
        branch, parent = keys.get((run.name, RC_CHECKPOINT)), keys.get(
            (RC_BASE_RUN, RC_CHECKPOINT))
        assert branch is not None and parent is not None, "the branch or its parent is not listed"
        assert (branch.parent, branch.parent_state) == (parent.item.key, "listed"), (
            "the branch is not listed under its parent's exact file")
        assert (branch.item.sha256, branch.item.operation, branch.item.parent_changes) == (
            sha, "branch", {}), "the list does not show the branch as it is recorded"
        return f"a same-settings branch of D1's base through the job runner on {device}", {
            "device": device, "trained_on_cuda": gpu is not None, "training_seconds": seconds,
            "started_through": "the job runner's start, bound to the parent's digest shown and "
                               "the dataset's",
            "branch": {"run": run.name, "sha256": sha, "operation": record.operation,
                       "changes": record.parent["changes"],
                       "optimizer": record.weight_transfer["optimizer"],
                       "completed_epochs": (payload.get("progress") or {}).get(
                           "completed_epochs"),
                       "training_rules": payload.get("training_rules")},
            "parent": {"run": RC_BASE_RUN, "sha256": state["base_sha"], "unchanged": True},
            "listed_under_its_parent": True, "listed_changes": branch.item.parent_changes,
            "loaded": True,
        }

    # ---- L8: the results export, every record of the flow, uncapped
    def _exported() -> Tuple[str, Dict[str, Any]]:
        import io

        import scripts.export_results as export_results
        from src.api.services import results
        from src.utils.fingerprint import read_once
        # Models › Results' side-by-side cap: that view's only; a query and an
        # export are never capped (owner's O6), so the flow keeps more runs.
        from src.webui.components.results_segment import COMPARE_CAP

        assert kept, "the flow kept no run to export"
        output = rc / "export.json"
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                code = export_results.main([
                    "--workspaces-root", f"{RC_EXPORT_SOURCES['workspaces_root']}={root}",
                    "--registry", f"{RC_EXPORT_SOURCES['registry']}={registry}",
                    "--format", "json", "--output", str(output)])
        except SystemExit as refused:   # a usage error: argparse's exit
            raise AssertionError(f"the export refused its arguments (exit {refused.code})"
                                 ) from None
        assert code == 0, f"the export exited {code}"
        # One read: the digest recorded is of the bytes checked and parsed.
        written = read_once(output)
        export_sha256, text = written.sha256, written.data.decode("utf-8")
        del written
        assert str(work.resolve()) not in text and str(work) not in text, (
            "the export carries a server path")
        exported = json.loads(text)
        assert (exported.get("format"), exported.get("columns")) == (
            results.EXPORT_FORMAT, list(results.COLUMN_NAMES)), "the export is of another format"
        rows = exported["rows"]
        unreadable = [row for row in rows if row["result"] == results.UNREADABLE]
        assert not unreadable, f"{len(unreadable)} exported items could not be read"
        by_run: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            by_run.setdefault(row["run_id"], []).append(row)
        expected = {entry["run_id"]: entry for entry in kept}
        assert sorted(by_run) == sorted(expected), (
            f"the export lists {len(by_run)} runs; the flow kept and registered {len(expected)}")
        assert len(by_run) > COMPARE_CAP, (
            f"the flow kept {len(by_run)} runs, too few to show the export is not capped at "
            f"{COMPARE_CAP}")
        for run_id, group in by_run.items():
            for row in group:
                _export_row_verified(row, expected[run_id])
        return "every run of the flow exported with its identities, settings, environment and " \
               "exposure; none capped, none unreadable", {
                   "format": exported["format"], "export_sha256": export_sha256,
                   "results": len(by_run), "rows": len(rows), "unreadable": 0,
                   "ui_compare_cap": COMPARE_CAP, "capped": False,
                   "runs": [{"probe": entry["probe"], "run_id": entry["run_id"],
                             "report_sha256": entry["report_sha256"], "kind": entry["kind"],
                             "role": entry["role"],
                             "metric_rows": len(by_run[entry["run_id"]])} for entry in kept],
                   "carries_no_server_path": True,
               }

    with _deployment_settings(settings):
        report.run("L1", "m2", "research comparison: a KG edge set (role test) is imported by "
                   "the CLI importer from D1's base: the preview writes nothing, the commit "
                   "binds the base's view, split and test pairs", _kg_edge_set)
        report.run("L2", "m2", "research comparison: KG link prediction through measure_scorer "
                   "is kept and registered, its exposure scoped to the base's record and its "
                   "scores, pairs and summary verified against each other", _lp_measured)
        report.run("L3", "m2", "research comparison: KG link prediction through the job runner "
                   "from the server's lists is kept and registered", _lp_through_the_runner)
        for case, claim, probe in (
            ("other_graph", "a KG edge set of another graph", _refuse_other_graph),
            ("tampered_pairs", "a KG edge set whose pairs file was changed",
             _refuse_tampered_pairs),
            ("other_view", "a model of another graph view, as a stated capability gap",
             _refuse_other_view),
            ("split_record", "a KG edge set whose split record contradicts its graph",
             _refuse_split_record),
        ):
            report.run(f"L4.{case}", "m2", f"research comparison: KG link prediction refuses "
                       f"{claim}, before anything is written", probe)
        report.run("L5", "m2", "research comparison: a disease child measured by both of its "
                   "protocols gives two independent records, read by the one reader",
                   _two_protocols)
        report.run("L6", "m2", "research comparison: a generated split is imported with its "
                   "fixture, measured and registered", _generated_split)
        for task in RC_TASKS:
            report.run(f"L7.{task}", "m2", f"research comparison, GPU regression: a short {task} "
                       "child of D1's base trained on the probe's device, measured by its "
                       "protocol and registered", _regression_case(task))
        report.run("L7.disease_frozen", "m2", "ops wiring: an fp32 frozen-backbone disease child "
                   "of D1's base on the probe's device, paused and resumed: its graph side the "
                   "base's, its downstream parts trained and alone in the optimizer, its record "
                   "kept across the resume, then measured and registered", _frozen_disease_child)
        report.run("L7.branch", "m2", "ops wiring: a same-settings branch of D1's base through "
                   "the job runner, on the probe's device, listed under its parent with no "
                   "change, its parent unchanged, and loaded", _same_settings_branch)
        report.run("L8", "m2", "research comparison: the results export lists every run of the "
                   "flow with its identities, settings, environment and exposure, uncapped",
                   _exported)


# =============================================================================
# M1's served readings, informational (S1 in phase M)
# =============================================================================
#: The probe that takes M1's minimal served resource and latency readings
#: (research comparison §5 Stage 10, the owner's item D; BACKLOG 64), after the
#: retired `scripts/measure_served_pipeline.py` (PLAN_B04 §7.3, its reviewed
#: procedure, of which this takes a minimal part). Its own letter: the probe
#: ID M1 is M2's other-graph precondition, and M1 is also the deployment mode
#: M5's service runs in. **Informational**: in neither required list
#: (`M2_DEMO_REQUIRED`, `M2_REAL_REQUIRED`), so `m2_acceptance` never reads it
#: and no reading, however slow or large, can leave the acceptance incomplete;
#: and outside the run's exit status (`INFORMATIONAL_PROBES`), so not even a
#: failed S1 (a start slower than `_SERVICE_START_SECONDS`, say) fails the run.
#: There is no threshold: nothing it reads is judged.
SERVED_READINGS_PROBE = "S1"
#: The probes the report records with their status but the run's exit status
#: never counts (`run_exit_status`): informational readings, which hardware
#: variance alone can fail and no acceptance requires. The console prints a
#: failure of one as an INFO line.
INFORMATIONAL_PROBES: Tuple[str, ...] = (SERVED_READINGS_PROBE,)
#: The readings' definition, recorded with them (`SERVED_READING_DEFINITIONS`).
SERVED_READINGS_VERSION = "served-readings/1"
#: The fixed workload after readiness: the first requests are the warm-up, a
#: count fixed here and never detected, and the next ones the steady state,
#: all sent serially, one at a time.
SERVED_WARM_UP_REQUESTS = 5
SERVED_STEADY_REQUESTS = 50
#: What each reading is, recorded with the readings so the report says it.
SERVED_READING_DEFINITIONS: Dict[str, str] = {
    "cold_start.seconds": (
        "the probe's clock from launching the service's process (uvicorn serving the "
        "application, as the launcher starts it; the process is the server's own) to its "
        "first /health answer. The application builds the configured pipeline in its startup, "
        "before the service accepts a connection, so this covers the interpreter's start, the "
        "imports, the dashboard's mount, reading the workspace and the checkpoint, building "
        "the model and moving it to the device. /health is asked every 0.5 s, so it reads up "
        "to that much late, and the seconds are rounded to 0.1 s (`_ProbeServer.wait_ready`). "
        "`state` \"read\" with them, or \"not_obtainable\" with its reason when /health never "
        "answered"),
    "readiness": (
        "before anything counts: the first /health answer (the cold start), /ready answering "
        "200, and the status's served identity naming the child, the workspace and its dataset "
        "record by the digests this probe read, the disease task and the device asked for "
        "(`_assert_served_identity`). `state` \"ready\" with /ready's `ready_status`, or "
        "\"not_ready\" with `failed_at` (\"health\", \"ready\" or \"served_identity\"), its "
        "`reason` and /ready's `ready_status` once it answered. A service not shown ready is "
        "sent no request: what was read before stays, and every reading after is not taken"),
    "seconds": (
        "per counted request, the probe's clock (time.perf_counter) from sending one POST "
        "/api/v1/diagnose, on a new loopback connection, to reading its whole answer. The "
        "service answers after the disease head's scores are copied to the host, so on CUDA "
        "the time includes the device's work, synchronised by that copy in the server's "
        "process; the probe's process runs no part of the request, and no synchronise call "
        "is made in it"),
    "server_handler_seconds": (
        "per counted request, the answer's own inference_time_ms in seconds: the route's "
        "wall time from its start to the answer built, before it is validated and sent"),
    "warm_up": "the first requests after readiness, a fixed count, whatever they took",
    "steady_state": (
        "the requests after the warm-up, a fixed count, sent serially. Every request asks the "
        "same query, and the service keeps no result cache, so each answer is computed"),
    "counted": (
        "a request answered 200 with a JSON body whose served identity is the one the status "
        "reported at readiness; any other answer is a failure, by class, and a phase with one "
        "is incomplete. A request with no whole answer (timed out, reset or cut off) ends the "
        "readings: the server may still be working on it"),
    "memory": (
        "the server process's VmRSS (resident now) and VmHWM (its peak resident size since the "
        "process started) from /proc/<pid>/status, in bytes, approximate as the kernel's RSS "
        "accounting is. VmHWM is never reset here, so it is the peak since the start, never "
        "one phase's. Whether a unified-memory device's allocations count in it is not "
        "established"),
    "allocator": (
        "the allocator variables in the environment the server process was started with "
        "(/proc/<pid>/environ), raw, null when absent: `state` \"read\" with them, or "
        "\"not_obtainable\" with its reason when that environment cannot be read or the "
        "process has exited"),
    "cuda_allocator_peak": (
        "torch.cuda.max_memory_allocated of the server's process: torch keeps it inside that "
        "process, and no route of the service exposes it"),
}
#: What these readings do not cover, said in the report with them.
SERVED_READINGS_NOT_COVERED: Tuple[str, ...] = (
    "a peak per phase: VmHWM is not reset between phases (PLAN_B04 §7.3's reset pauses the "
    "server, and is not taken here)",
    "system-wide memory and swap, a baseline before the launch, and the machine's OOM kills "
    "(PLAN_B04 §7.3's R0 and system counters)",
    "a reload, and the old and new pipelines held together (PLAN_B04 §7.3's R3 and R4)",
    "concurrent requests, a varied or largest-size workload, and repeats over several cold "
    "starts: one query, the same each time, from one cold start",
    "the real workspace's children at deployment size, and gene and patients-like-me "
    "children: only D1's disease child on the demo workspace is served",
    "the server's CUDA allocator counters (cuda_allocator_peak)",
)


def _reading_not(state: str, reason: str) -> Dict[str, Any]:
    """A reading that has no value: its state (``not_obtainable``,
    ``not_applicable`` or ``not_taken``) and why, never a number."""
    return {"state": state, "reason": reason}


def _process_memory(pid: int, proc: Path = Path("/proc")) -> Dict[str, Any]:
    """`pid`'s VmRSS and VmHWM in bytes, read from its ``status`` under `proc`
    (Linux's process counters; SERVED_READING_DEFINITIONS["memory"]), or the
    reason they cannot be read: no /proc on this platform, or the process gone
    (a process that has exited has neither line)."""
    if not (proc / "self" / "status").is_file():
        return _reading_not("not_obtainable",
                            "this platform has no /proc/<pid>/status (Linux's process counters)")
    try:
        text = (proc / str(pid) / "status").read_text()
    except OSError:
        return _reading_not("not_obtainable", "the server's process is gone")
    values: Dict[str, int] = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if name in ("VmRSS", "VmHWM") and len(parts) == 2 and parts[1] == "kB" \
                and parts[0].isdigit():
            values[name] = int(parts[0]) * 1024
    if set(values) != {"VmRSS", "VmHWM"}:
        return _reading_not("not_obtainable",
                            "the server's process has exited: its status has no VmRSS or VmHWM")
    return {"state": "read", "rss_bytes": values["VmRSS"],
            "peak_rss_since_start_bytes": values["VmHWM"]}


def _process_allocator(pid: int, proc: Path = Path("/proc")) -> Dict[str, Any]:
    """The allocator variables `pid` was started with, by the names the
    report's `allocator` records for the probe's own process
    (`_allocator_variables`), raw, None when absent, from its ``environ``
    under `proc`; or why they cannot be read. A process that has exited but
    is not yet reaped has an ``environ`` that reads empty on some kernels:
    that is no environment, not one without the variables (the service is
    always started with some, `_ProbeServer.environment`)."""
    try:
        raw = (proc / str(pid) / "environ").read_bytes()
    except OSError:
        return _reading_not("not_obtainable",
                            "the server process's environment cannot be read here "
                            "(/proc/<pid>/environ)")
    if not raw.strip(b"\0"):
        return _reading_not("not_obtainable",
                            "the server's process has exited: its environment reads empty")
    pairs = dict(item.split(b"=", 1) for item in raw.split(b"\0") if b"=" in item)
    return {"state": "read", **{
        name: (pairs[name.encode()].decode("utf-8", "replace")
               if name.encode() in pairs else None) for name in _allocator_variables()}}


def _cuda_allocator_peak(device: str) -> Dict[str, Any]:
    """The server's CUDA allocator peak, which this probe cannot read
    (SERVED_READING_DEFINITIONS["cuda_allocator_peak"]): never a figure taken
    in the probe's own process, which would describe the probe."""
    if device != "cuda":
        return _reading_not("not_applicable", f"the model is served on the {device}")
    return _reading_not("not_obtainable",
                        "torch keeps its allocator counters inside the server's process and "
                        "no route of the service exposes them; a reading in the probe's "
                        "process would describe the probe, not the server")


def _seconds_summary(values: Sequence[float]) -> Dict[str, Any]:
    """The median, smallest and largest of `values` (no mean, as PLAN_B04 §7.3
    reports), or the state that none was counted."""
    import statistics

    if not values:
        return _reading_not("not_obtainable", "no request of this phase was counted")
    return {"state": "read", "median_seconds": round(statistics.median(values), 6),
            "min_seconds": min(values), "max_seconds": max(values)}


def _served_requests(server: Any, body: bytes, planned: int,
                     identity: Mapping[str, Any]) -> Dict[str, Any]:
    """`planned` diagnose requests of `body` to `server`, serially, each timed
    on the probe's clock (SERVED_READING_DEFINITIONS["seconds"]). A request
    counts only by SERVED_READING_DEFINITIONS["counted"]; a failure is
    counted by class (``http_<status>``, with ``out_of_memory_in_body``
    beside it when the body says so, ``not_json``, ``served_identity_mismatch``,
    ``no_inference_time``), and the phase goes on. A request with no whole
    answer (``no_response``) or a server that has exited (``server_exited``)
    ends it, said in ``ended``."""
    import http.client

    seconds: List[float] = []
    handler: List[float] = []
    failures: Dict[str, int] = {}
    sent, ended = 0, None

    def fail(kind: str) -> None:
        failures[kind] = failures.get(kind, 0) + 1

    for _ in range(planned):
        if server.process is None or server.process.poll() is not None:
            ended = "server_exited"
            fail(ended)
            break
        sent += 1
        started = time.perf_counter()
        try:
            status, _, answer = server.request(
                "POST", "/api/v1/diagnose", {"Content-Type": "application/json"}, body)
        except (OSError, http.client.HTTPException):
            ended = "no_response"
            fail(ended)
            break
        elapsed = time.perf_counter() - started
        if status != 200:
            fail(f"http_{status}")
            if b"out of memory" in answer.lower():
                fail("out_of_memory_in_body")
            continue
        try:
            result = json.loads(answer)
        except ValueError:
            fail("not_json")
            continue
        if not isinstance(result, dict) or result.get("served_identity") != identity:
            fail("served_identity_mismatch")
            continue
        try:
            handler_seconds = float(result["inference_time_ms"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            fail("no_inference_time")
            continue
        seconds.append(round(elapsed, 6))
        handler.append(round(handler_seconds, 6))
    return {
        "state": "read", "planned": planned, "sent": sent, "counted": len(seconds),
        "failures": failures, "ended": ended,
        "complete": sent == planned and len(seconds) == planned,
        "seconds": seconds, "summary": _seconds_summary(seconds),
        "server_handler_seconds": handler, "server_handler_summary": _seconds_summary(handler),
    }


def _served_readiness(
    server: Any, *, checkpoint_sha256: str, workspace_digests: Dict[str, str], record: Any,
    device: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    """S1's readiness, before anything counts
    (SERVED_READING_DEFINITIONS["readiness"]): the service's first /health
    answer (`wait_ready`, the cold start), /ready answering 200, and the
    status's served identity held to this probe's own reads
    (`_assert_served_identity`). Returns the readiness record, the cold start
    and the served identity. The identity is None when the service was not
    shown ready: the record then says at which step (`failed_at`) and why
    (`reason`, this file's words, with no path), with /ready's status once it
    answered, and the cold start keeps its seconds once /health answered."""
    try:
        seconds = server.wait_ready()
    except AssertionError as failure:
        return ({"state": "not_ready", "failed_at": "health", "reason": str(failure)},
                _reading_not("not_obtainable", "the service never answered /health"), None)
    cold_start = {"state": "read", "seconds": seconds}
    ready = server.request("GET", "/ready")[0]
    if ready != 200:
        return ({"state": "not_ready", "failed_at": "ready", "ready_status": ready,
                 "reason": f"the service answered /ready {ready}: it serves no pipeline (its "
                           "log is under the work directory)"}, cold_start, None)
    try:
        identity = _assert_served_identity(
            server.json("/api/v1/pipeline/status").get("served_identity"),
            checkpoint_sha256=checkpoint_sha256, workspace_digests=workspace_digests,
            record=record, task="disease", device=device)
    except AssertionError as failure:
        return ({"state": "not_ready", "failed_at": "served_identity", "ready_status": ready,
                 "reason": str(failure)}, cold_start, None)
    return {"state": "ready", "ready_status": ready}, cold_start, identity


def _served_not_taken(cold_start: Mapping[str, Any]) -> Dict[str, Any]:
    """The readings of a service not shown ready (`_served_readiness`), which
    is sent no request: the cold start as it was read, and every reading after
    readiness not taken, with why. The same keys as `_served_reading`'s."""
    not_taken = _reading_not("not_taken",
                             "the service was not shown ready (readiness): no request was sent")
    return {
        "cold_start": ({**cold_start, "memory_at_ready": not_taken}
                       if cold_start["state"] == "read" else dict(cold_start)),
        "warm_up": {**not_taken, "planned": SERVED_WARM_UP_REQUESTS, "complete": False},
        "steady_state": {**not_taken, "planned": SERVED_STEADY_REQUESTS, "complete": False},
        "complete": False,
    }


def _served_reading(server: Any, query: Sequence[str], identity: Mapping[str, Any],
                    cold_start: Mapping[str, Any]) -> Dict[str, Any]:
    """The readings of a service already shown ready and serving `identity`
    (`_served_readiness`), beside its cold start: the server's memory at
    readiness, then the warm-up and the steady state of `query`, each with the
    memory after it. Everything is read from the server's process (its PID's
    /proc entries) or its answers, never from the probe's process, which
    reads no device counter at all. A warm-up ended by a request with no whole
    answer or the server's exit leaves the steady state not taken."""
    pid = server.process.pid
    body = json.dumps({"phenotypes": list(query)}).encode("utf-8")
    memory_at_ready = _process_memory(pid)
    warm_up = _served_requests(server, body, SERVED_WARM_UP_REQUESTS, identity)
    warm_up["memory_after"] = _process_memory(pid)
    if warm_up["ended"] is None:
        steady = _served_requests(server, body, SERVED_STEADY_REQUESTS, identity)
        steady["memory_after"] = _process_memory(pid)
    else:
        steady = {**_reading_not("not_taken", f"the warm-up ended: {warm_up['ended']}"),
                  "planned": SERVED_STEADY_REQUESTS, "complete": False}
    return {
        "cold_start": {**cold_start, "memory_at_ready": memory_at_ready},
        "warm_up": warm_up,
        "steady_state": steady,
        "complete": bool(warm_up["complete"] and steady["complete"]),
    }


def _served_readings(report: Report, work: Path, sound: Path, checkpoint: Path,
                     device: str) -> None:
    """S1, M1's minimal served readings (`SERVED_READINGS_PROBE`): D1's disease
    child (`checkpoint`) on the demo workspace, served by a service of its own,
    started as the launcher starts it (`_ProbeServer`) with the pipeline
    configured at startup (`SHEPHERD_KG_PATH`, `SHEPHERD_DATA_DIR`,
    `SHEPHERD_CHECKPOINT_PATH`, `SHEPHERD_DEVICE`), its files under
    <work>/served-readings. Readiness is asserted before anything counts:
    /ready answers 200 and the status's served identity names the child, the
    workspace and its dataset record by the digests this probe read, the
    disease task and the device asked for, in effect (`_assert_served_identity`).
    (`_served_readiness`). Then the fixed workload of one query from the
    workspace's own validation cases (`_workspace_query`), sent with the API's
    defaults (`_served_reading`).

    **Informational, never required and outside the exit status**
    (`SERVED_READINGS_PROBE`, `INFORMATIONAL_PROBES`). The probe fails only
    when no reading of the child could be taken: the service did not start,
    was not ready or served something else. It then fails with its facts
    (`ProbeFailure`): what was read before (the readiness record, the cold
    start) and every reading after as not taken (`_served_not_taken`). Every
    reading that cannot be taken is recorded as a state with its reason,
    never a number, and an incomplete workload is recorded as incomplete, not
    as a failure."""
    def _read() -> Tuple[str, Dict[str, Any]]:
        from src.api.routes.diagnose import DiagnoseRequest
        from src.kg.artifacts import GRAPH_ARTIFACTS

        _, _, reads, record = verify_workspace(sound)
        query, asked = _workspace_query(checkpoint, sound, reads)
        checkpoint_sha256 = _read_run_checkpoint(checkpoint)["sha256"]
        server = _ProbeServer(SERVED_READINGS_PROBE, work / "served-readings", {
            "SHEPHERD_KG_PATH": str(sound / GRAPH_ARTIFACTS["kg"]),
            "SHEPHERD_DATA_DIR": str(sound),
            "SHEPHERD_CHECKPOINT_PATH": str(checkpoint),
            "SHEPHERD_DEVICE": device,
        })
        try:
            server.start()
            readiness, cold_start, identity = _served_readiness(
                server, checkpoint_sha256=checkpoint_sha256, workspace_digests=reads,
                record=record, device=device)
            reading = (_served_reading(server, query, identity, cold_start)
                       if identity is not None else _served_not_taken(cold_start))
            # Read while the server's process runs, whether it was ready or not.
            allocator = _process_allocator(server.process.pid)
        finally:
            server.stop()
        defaults = {name: DiagnoseRequest.model_fields[name].default
                    for name in ("top_k", "include_explanations", "include_paths")}
        facts = {
            "required": False,
            "version": SERVED_READINGS_VERSION,
            "subject": {"task": "disease", "checkpoint_sha256": checkpoint_sha256,
                        "served_identity": identity if identity is not None else _reading_not(
                            "not_taken", "the service was not shown ready (readiness)")},
            "device": device,
            "query": asked,
            "request": {"fields_sent": ["phenotypes"], "api_defaults": defaults},
            "readiness": readiness,
            **reading,
            "allocator": allocator,
            "cuda_allocator_peak": _cuda_allocator_peak(device),
            "definitions": dict(SERVED_READING_DEFINITIONS),
            "not_covered": list(SERVED_READINGS_NOT_COVERED),
        }
        if identity is None:
            raise ProbeFailure(readiness["reason"], facts)
        steady = reading["steady_state"]
        summary = steady.get("summary") or {}
        memory = steady.get("memory_after") or {}
        print(f"         Served readings: cold start {cold_start['seconds']} s; steady state "
              f"{steady.get('counted', 0)} of {steady['planned']} counted"
              + (f", median {summary['median_seconds']} s" if summary.get("state") == "read"
                 else "")
              + (f"; server resident {memory['rss_bytes'] / 1e9:.2f} GB, peak since start "
                 f"{memory['peak_rss_since_start_bytes'] / 1e9:.2f} GB"
                 if memory.get("state") == "read" else ""))
        detail = ("readings taken: cold start, warm-up and steady state, each counted request "
                  "timed" if reading["complete"] else
                  "readings incomplete: recorded as taken, each failure by class")
        return detail, facts

    report.run(SERVED_READINGS_PROBE, "m2",
               "M1's served readings of D1's disease child: cold start, warm-up and "
               "steady-state request times and the server's memory (informational: never "
               "required, never judged, outside the exit status)", _read)


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
    report: Report, work: Path, sound: Path, checkpoint: Path, base: Path, device: str
) -> None:
    """Contract M2's acceptance on the demo workspace: the successes first, then
    every refusal, each from the production entry point that makes it; then
    contract M4's flow on the same workspace (T1-T6), then M5's (W1-W11), then
    the research comparison round's (L1-L8, `_rc_demo`), and last M1's served
    readings (S1, `_served_readings`), informational. The formal entry
    points take the probe's own GPU lock (`_probes_own_gpu_lock`), and M5's
    and the research comparison's each that of their own deployment.

    `checkpoint` is D1's disease child, the resume refusals' parent and the
    subject of M6, M7, M8, T2, L5 and S1; `base` is D1's base, which M2 finetunes
    again into a run it pauses and resumes and the research comparison's flow
    lists, measures and finetunes."""
    print("\nM. Contract M2 on the demo workspace, then M4's flow")
    import src.api.main as api
    from scripts.setup_demo import demo_task_training
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

    # M2's request: the demo's disease sizes and the mixed precision it states
    # (Q13 (b)); its resume and M9's extension inherit it and are judged
    # against it.
    m2_settings = {**demo_task_training()["disease"], **M2_PRECISION}
    m2_precision = _precision_request(m2_settings, device, "finetune", "disease")

    def _resumed() -> Tuple[str, Dict[str, Any]]:
        # **A real pause and resume** (contract M3a-3) of a disease finetune
        # of D1's base: plan two epochs, pause after one, resume the remaining
        # one. A finetune, because measurement and serving take the resumed
        # file itself; a base's pause and resume is R3's and R4's.
        paused = _last(_train_run(sound, m2 / "paused", device, stage="finetune",
                                  task="disease", epochs=2, settings=m2_settings,
                                  finetune_from=base, pause_after=1))
        before = _read_run_checkpoint(paused)
        progress = before["progress"]
        assert progress.get("end") == "paused" and progress.get("completed_epochs") == 1, (
            f"the run did not pause after epoch 1: {progress.get('end')}")
        observed: List[Dict[str, Any]] = []
        child = _last(_train_run(sound, m2 / "resume_ok", device, stage="finetune",
                                 task="disease", resume_from=paused, observe=observed))
        facts = _assert_resumed(paused, child, observed, requested=m2_precision)
        facts.update(_assert_only_the_remaining_epochs(before, _read_run_checkpoint(child)))
        state["child"] = child
        facts.update(paused_end=progress["end"], parent_attempts=progress["optimizer_attempts"],
                     parent_updates=progress["optimizer_updates"])
        return "paused after one of two epochs, resumed only the remaining one", facts

    report.run("M2", "m2", "CONTROL: a paused run resumes its remaining epoch and records it",
               _resumed)

    def _measured() -> Tuple[str, Dict[str, Any]]:
        # **The resumed model, never its parent in its place**: the flow this
        # part claims is build, train, resume, then measure and serve that model.
        assert "child" in state, "the resumed checkpoint (M2) is not available"
        output = m2 / "measured" / "measurement.json"
        assert _measure(state["child"], sound, output, device, protocol="disease") == 0, (
            "the measurement failed")
        return "the resumed checkpoint was measured over its own workspace", _measurement_facts(
            output, _read_run_checkpoint(state["child"])["sha256"], *_kg_and_record(sound),
            protocol="disease")

    report.run("M3", "m2", "CONTROL: measurement accepts the resumed checkpoint over its graph",
               _measured)

    def _served() -> Tuple[str, Dict[str, Any]]:
        assert "child" in state, "the resumed checkpoint (M2) is not available"
        _, _, read, record = verify_workspace(sound)
        facts = _serve_and_diagnose(sound / "kg.json", state["child"], sound, device,
                                    record, read)
        return "build, train, resume and serve its task", facts

    report.run("M4", "m2", "CONTROL: the resumed model serves its task", _served)

    # ---- Resume refusals: nothing restored, no run directory, nothing changed.
    # Each resumes a disease child, stating its stage and task as the
    # training service does.
    def _refused_resume(case: str, data_dir: Path, parent: Path, expected: type,
                        fragment: str) -> Dict[str, Any]:
        run = m2 / "refused" / case
        workspace_before = snapshot(data_dir, include_dirs=True)
        parent_before = snapshot(parent.parent, include_dirs=True) if parent.exists() else None
        refuses(lambda: _train_run(data_dir, run, device, stage="finetune", task="disease",
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

    def _refuse_no_model_spec() -> Tuple[str, Dict[str, Any]]:
        from src.utils.run_plan import RunPlanRefused

        # D1's child with its model spec removed: written before the model
        # spec, as a history file is, so nothing is guessed to rebuild it.
        stripped = _without_model_spec(checkpoint, m2 / "parents" / "no_model_spec.pt")
        return "refused before anything existed", _refused_resume(
            "no_model_spec", sound, stripped, RunPlanRefused, "written before the model spec")

    for case, claim, probe in (
        ("pre_contract", "a parent from before the contract", _refuse_pre_contract),
        ("other_graph", "a parent from a same-shaped other graph", _refuse_other_graph),
        ("other_samples", "a parent that read other sample files", _refuse_other_samples),
        ("validated_parent", "a parent that ran validation, resumed by a run without it",
         _refuse_validated_parent),
        ("missing", "a requested parent that does not exist", _refuse_missing),
        ("no_model_spec", "a parent without a model spec", _refuse_no_model_spec),
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
        refuses(lambda: _measure(checkpoint, other, output, device, protocol="disease"),
                TrainingRecordMismatch, "kg differs")
        assert not output.parent.exists(), "the refused measurement created its output directory"
        assert_untouched(before, other, "a refused measurement", include_dirs=True)
        return "refused before any model was built or output written", {}

    report.run("M7", "m2",
               "measurement refuses a checkpoint over a same-shaped other workspace",
               _measurement_refuses)

    def _inventory() -> Tuple[str, Dict[str, Any]]:
        import scripts.audit_checkpoint_family as audit
        from src.config.model_types import WRITTEN_BEFORE_SPEC

        assert "other" in state, "the same-shaped workspace (M1) is not available"
        root = m2 / "inventory_root"
        placed = {
            "a/checkpoints/m2/fresh.pt": checkpoint,
            "a/checkpoints/m2/pre_contract.pt": _pre_contract(),
            # D1's child with its model spec removed (the one stripping
            # helper): it records this workspace's graph and inputs, so only
            # its missing model spec can refuse it.
            "a/checkpoints/m2/no_model_spec.pt": _without_model_spec(
                checkpoint, m2 / "parents" / "inventory_no_model_spec.pt"),
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
        specless = verdicts("a", "a/checkpoints/m2/no_model_spec.pt")
        for use, verdict in specless.items():
            assert (verdict["verdict"] == "refused"
                    and WRITTEN_BEFORE_SPEC in (verdict["reason"] or "")), (
                f"a file without a model spec reads {verdict['verdict']} for {use}")
        return ("the inventory lists what M2 accepts and refuses, and refuses a file without "
                "a model spec for both uses"), {
            "report_sha256": file_sha256(output),
            "counts": inventory["counts"],
        }

    report.run("M8", "m2", "the inventory lists the demo checkpoints' M2 verdicts, and refuses "
               "a file without a model spec for both uses", _inventory)

    def _extended() -> Tuple[str, Dict[str, Any]]:
        assert "child" in state, "the resumed checkpoint (M2) is not available"
        before = _read_run_checkpoint(state["child"])
        observed: List[Dict[str, Any]] = []
        extended = _last(_train_run(sound, m2 / "extended", device, stage="finetune",
                                    task="disease", resume_from=state["child"], extend_epochs=1,
                                    observe=observed))
        after = _read_run_checkpoint(extended)
        # The extension inherits M2's precision and is judged against it.
        scaler = _scaler_restore_facts(before, after, observed, requested=m2_precision)
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
        refuses(lambda: _train_run(copy, run, device, stage="base", epochs=1,
                                   settings=demo_task_training()["base"]),
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
        refuses(lambda: _measure(state["child"], copy, output, device, protocol="disease"),
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
    # disease child (the ledger is written beside the measured file), the
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
        shutil.copyfile(checkpoint, measured)          # the child D1 trained, copied
        sha256 = _read_run_checkpoint(measured)["sha256"]
        code = _measure_cohort(measured, sound, registry, M4_DEMO_LABEL, device, "--record",
                               protocol="disease")
        assert code == 0, f"the measurement with --record exited {code}"
        samples = _cohort_samples(version)
        path, kept, identity, ranked = _kept_report(registry, M4_DEMO_LABEL,
                                                    [s.patient_id for s in samples])
        manifest = kept["manifest"]
        assert manifest["artifact_digests"]["checkpoint"] == sha256, (
            "the report does not name the checkpoint it was given")
        assert manifest["cohort_version"]["manifest_sha256"] == version.identity.sha256, (
            "the report names another cohort version")
        assert manifest["relations"] == REGISTERED_RELATIONS, (
            f"the report's relation states are {manifest['relations']}")
        binding = _ranked_cases(kept, ranked, _own_binding(measured, sound, samples))
        assert version.manifest["samples"]["n"] == M4_DEMO_COUNTS["ranked_eligible"]
        r10 = manifest["r10"]
        # D1's child trained on this workspace, so the inputs it records
        # (train and validation samples) are its files.
        _evidence_found_in(r10, sound, ("training", "validation"))
        record, ledger_sha256 = _the_record(measured.parent, identity, sha256, version)
        assert record["registration"]["sequence"] == 1
        m4.update(report=path, report_sha256=identity.sha256, record=record,
                  ledger_sha256=ledger_sha256, checkpoint_sha256=sha256, r10=r10)
        return "measured, the report kept, R10 recorded, one record from the report's bytes", {
            "exit_code": code, "checkpoint_sha256": sha256,
            "report_sha256": identity.sha256, "ranks_sha256": kept["ranks"]["sha256"],
            "ranks_rows": len(samples), "n_ranked": kept["n_ranked"], "binding": binding,
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
                                                      M4_OTHER_GRAPH_LABEL, device, "--record",
                                                      protocol="disease"),
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
    # record_evaluation and the loaded-model route's group description alike.
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
             m4.get("version"),
             [m4["report"].parent.name] if m4.get("report") else [],
             [m4["report"].parent.name] if m4.get("record") else [])

    # ---- The research comparison round's flow on the same workspace: D1's
    # base and disease child, M1's other graph, the probe's own deployment
    # under <work>/rc.
    _rc_demo(report, work, sound, base, checkpoint, device, state.get("other"))

    # ---- M1's served readings (S1), informational and never required: D1's
    # disease child served by a service of its own, after every other probe
    # of the part, so no other service or job of the probe runs meanwhile.
    _served_readings(report, work, sound, checkpoint, device)


# =============================================================================
# Phase R — contract M2's acceptance on real data (--real-workspace)
# =============================================================================
def _size_and_mtime(path: Path) -> Tuple[int, int]:
    """A file by size and modification time, without reading a byte."""
    status = path.stat()
    return status.st_size, status.st_mtime_ns


def _stat_tree(directory: Path) -> Dict[str, Tuple[int, int]]:
    """Every file and directory under `directory` by size and modification time,
    without reading a byte: a workspace's tables can be gigabytes."""
    taken: Dict[str, Tuple[int, int]] = {}
    for path in sorted(directory.rglob("*")):
        key = str(path.relative_to(directory)) + ("/" if path.is_dir() else "")
        taken[key] = (0, 0) if path.is_dir() else _size_and_mtime(path)
    return taken


@_under_the_probes_own_gpu_lock
def phase_m2_real(
    report: Report, work: Path, real: Path, reference: Optional[Path],
    workspaces_root: Path, device: str, *,
    task_sizes: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    """Contract M2's acceptance on real data: the full success path, from the
    production entry points, on a workspace that is read and never written;
    contract M4's flow on it (RT1-RT3) and M5's (RW1-RW3) before R8 checks it
    unchanged. The formal entry points take the probe's own GPU lock
    (`_probes_own_gpu_lock`), under the work directory.

    Every model is trained by the reference recipe (Q9 (1) A): R3's base, two
    epochs paused after one and resumed by R4; R4.finetune's disease child of
    that resumed base, and RW2's, one epoch each. `task_sizes` replaces some
    of their sizes (`setup_demo.demo_task_training`), as the CPU test does for
    a workspace of the demo's size; the deployment's run passes none.
    `reference`, when given, is a history file inside the workspaces root: a
    checkpoint without a model spec and without a training record of its
    graph (no `training_input_digests`, or none for `kg`), such as the
    deployment's model-02. R2's finetune and resume refuse it for its missing
    model spec (serving is checked on R2's stripped copy, which records its
    graph, so serving's refusal is the model spec's and not the record's); R7's
    inventory refuses it for its missing record, checked first, or for its
    missing model spec. R8 checks it unchanged, and nothing is read from it to
    configure a run."""
    print("\nR. Contract M2 on real data, with M4's flow")
    from src.utils.fingerprint import file_sha256

    runs = work / "real"
    runs.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {"before": _stat_tree(real)}
    reference_before = _size_and_mtime(reference) if reference is not None else None
    sizes = task_sizes or {}
    # R3's request, from R3_PRECISION alone, as the acceptance's scope states
    # it: R4's resume inherits it and is judged against it.
    r3_settings = {**(sizes.get("base") or {}), **R3_PRECISION}
    r3_precision = _precision_request(R3_PRECISION, device, "base", None)

    def _workspace() -> Tuple[str, Dict[str, Any]]:
        digests, cohorts, read, record = verify_workspace(real)
        # **Built from files, every input identified** (contract M3b): a real
        # workspace whose record were synthetic, or left an input unidentified,
        # would not be the real-data run this part claims.
        assert (record.binding, record.origin) == ("verified", "files"), (
            f"the real workspace's record is {record.binding}, origin {record.origin}"
        )
        state["kg"], state["record"], state["read"] = digests["kg"], record, read
        state["bound"] = dict(digests)
        return "the workspace's graph and training inputs verify", {
            "kg_digest": digests["kg"],
            "dataset_record": _record_facts(record),
            "train_diseases": len(cohorts.disease_sets["train"]),
            "val_diseases": len(cohorts.disease_sets["val"]),
        }

    report.run("R1", "real", "the real workspace verifies", _workspace)

    def _initial() -> Tuple[str, Dict[str, Any]]:
        from src.models.bundle.record import check_model_record, read_model_record
        from src.utils.checkpoint_io import TRAINING_RULES_VERSION, read_checkpoint

        assert "kg" in state, "R1 did not pass"
        # One epoch of a two-epoch plan of a base, paused (contract M3a-3), so
        # R4 has an unfinished stage to resume.
        parent = _last(_train_run(real, runs / "initial", device, stage="base", epochs=2,
                                  settings=r3_settings, pause_after=1))
        read = _read_run_checkpoint(parent)
        assert read["progress"].get("end") == "paused", "the initial run did not pause"
        # A base reads the graph only, so it records the graph's roles.
        missing = sorted({"kg", "node_features", "edge_indices", "num_nodes",
                          "split_manifest"} - set(read["record"]))
        assert not missing, f"the checkpoint does not record {missing}"
        assert read["record"]["kg"] == state["kg"], "the checkpoint records another graph"
        checkpoint = read_checkpoint(parent, map_location="cpu", weights_only=True).checkpoint
        record = read_model_record(checkpoint)
        check_model_record(record, learning_stage="base")
        # The build's training rules, compared with the live constant (the
        # acceptance encodes no rules version; the observed one is a fact).
        assert checkpoint.get("training_rules") == TRAINING_RULES_VERSION, (
            f"the base records training rules {checkpoint.get('training_rules')}, not this "
            f"build's {TRAINING_RULES_VERSION}")
        recipe = (read["loaders"].get("train") or {}).get("recipe") or {}
        state["parent"] = parent
        return "one epoch of two of a base, by the reference recipe, paused, recorded", {
            "sha256": read["sha256"], "recorded_roles": sorted(read["record"]),
            "optimizer_step": read["optimizer_step"],
            "learning_stage": record.learning_stage, "encoder_kind": record.model_spec.encoder.kind,
            "training_rules": checkpoint.get("training_rules"),
            "recipe": {"operator_set": recipe.get("operator_set"),
                       "reference_filled": sorted(recipe.get("reference_filled") or {})},
        }

    report.run("R3", "real", "a base trained from scratch pauses and records its inputs",
               _initial)

    def _resumed() -> Tuple[str, Dict[str, Any]]:
        assert "parent" in state, "R3 did not pass"
        observed: List[Dict[str, Any]] = []
        child = _last(_train_run(real, runs / "resumed", device, stage="base",
                                 resume_from=state["parent"], observe=observed))
        facts = _assert_resumed(state["parent"], child, observed, requested=r3_precision)
        progress = _read_run_checkpoint(child)["progress"]
        assert progress.get("end") == "finished" and progress.get("completed_epochs") == 2
        state["base"] = child
        return "the remaining epoch from the paused base, recorded", facts

    report.run("R4", "real", "a resume of the paused run trains its remaining epoch and names it",
               _resumed)

    def _finetuned() -> Tuple[str, Dict[str, Any]]:
        from src.models.bundle.record import check_model_record, read_model_record
        from src.utils.checkpoint_io import read_checkpoint

        assert "base" in state, "R4 did not pass"
        # **From R4's actual resumed file**: a base serves no diagnosis, so the
        # measurement, serving and M4 flow below take its disease child.
        trained = _train_disease_child(real, runs / "finetune", device, base=state["base"],
                                       sizes=sizes)
        record = read_model_record(read_checkpoint(trained.child, map_location="cpu",
                                                   weights_only=True).checkpoint)
        check_model_record(record, learning_stage="finetune", task="disease")
        base_sha256 = _read_run_checkpoint(state["base"])["sha256"]
        assert (record.parent or {}).get("sha256") == base_sha256, (
            "the child does not name R4's resumed base as its parent")
        state["child"] = trained.child
        return "a disease child finetuned from R4's resumed base", {
            "sha256": _read_run_checkpoint(trained.child)["sha256"],
            "parent_sha256": base_sha256, "learning_stage": record.learning_stage,
            "task": record.task, "operation": record.operation,
            "training_seconds": trained.seconds,
        }

    report.run("R4.finetune", "real", "a disease child is finetuned from the resumed base",
               _finetuned)

    def _history_refused() -> Tuple[str, Dict[str, Any]]:
        import src.api.main as api
        from src.inference.pipeline import PipelineBuildError
        from src.config.model_types import WRITTEN_BEFORE_SPEC
        from src.utils.run_plan import RunPlanRefused

        assert "child" in state, "R4.finetune did not pass"
        history = runs / "history"
        # **A subject this run always has**: R4.finetune's child with its model
        # spec removed; the deployment's own history file beside it when given.
        stripped = _without_model_spec(state["child"], history / "no_model_spec.pt")
        subjects = {"stripped copy of R4.finetune's child": stripped}
        if reference is not None:
            subjects["history file given"] = reference
        seen: Dict[str, Any] = {}
        for index, (subject, path) in enumerate(subjects.items()):
            # What R2 claims to leave as it was, and no more: the stripped
            # copy's directory under --work-dir, by content; the history file
            # itself, by size and modification time as R8 compares it, never
            # the directory it is in, which can be the real workspace or the
            # workspaces root (R8 covers those, without reading a byte).
            before = snapshot(history) if path == stripped else _size_and_mtime(path)
            refused: Dict[str, str] = {}
            for use, fields in (("finetune", {"epochs": 1, "finetune_from": path}),
                                ("resume", {"resume_from": path})):
                run = history / "refused" / f"{use}-{index}"
                refuses(lambda run=run, fields=fields: _train_run(
                    real, run, device, stage="finetune", task="disease", **fields),
                    RunPlanRefused, "written before the model spec")
                assert not run.exists(), f"a refused {use} created its run directory"
                refused[use] = "refused before anything was written"
            if path == stripped:
                # **Serving, on the copy that records its graph**: the
                # production build over the real workspace, refused at load
                # by the model record (`read_model_record`) before any model
                # is built, observed at the one builder, and nothing published.
                # The given file is not served: serving would refuse it for its
                # missing training record first, which R7 covers.
                served, builds = api.app_state.pipeline, []
                with _observing_builds(builds):
                    refuses(lambda: api.build_pipeline(
                        kg_path=str(real / "kg.json"), checkpoint_path=str(stripped),
                        data_dir=str(real), device=device,
                    ), PipelineBuildError, WRITTEN_BEFORE_SPEC)
                assert builds == [], "a model was built from a file without a model spec"
                assert api.app_state.pipeline is served, "the served pipeline changed"
                refused["serving"] = "refused at load before any model was built"
                assert_untouched(before, history, "a refused finetune, resume or serving build")
            else:
                assert _size_and_mtime(path) == before, (
                    "a refused finetune or resume changed the history file")
            seen[subject] = refused
        if reference is None:
            seen["history file"] = "not given"
        return ("a file without a model spec is refused by a finetune and a resume, and the "
                "stripped copy also by serving before any model is built"), {"subjects": seen}

    report.run("R2", "real", "a file without a model spec is refused by a finetune and a resume, "
               "and the stripped copy also by serving before any model is built, on the real "
               "workspace, before anything is written", _history_refused)

    def _measured() -> Tuple[str, Dict[str, Any]]:
        assert "child" in state, "R4.finetune did not pass"
        output = runs / "measurement" / "measurement.json"
        assert _measure(state["child"], real, output, device, protocol="disease") == 0, (
            "the measurement failed")
        return ("measured by the disease protocol through the production entry point",
                _measurement_facts(output, _read_run_checkpoint(state["child"])["sha256"],
                                   state["kg"], state["record"], protocol="disease"))

    report.run("R5", "real", "measurement accepts the new child and measures it by the "
               "disease protocol", _measured)

    def _served() -> Tuple[str, Dict[str, Any]]:
        assert "child" in state, "R4.finetune did not pass"
        facts = _serve_and_diagnose(real / "kg.json", state["child"], real, device,
                                    state["record"], state["read"])
        return "served for the disease task, and a diagnosis ran", facts

    report.run("R6", "real", "serving builds the new child and diagnoses", _served)

    def _inventory() -> Tuple[str, Dict[str, Any]]:
        import scripts.audit_checkpoint_family as audit
        from src.config.model_types import WRITTEN_BEFORE_SPEC

        assert "child" in state, "R4.finetune did not pass"
        facts: Dict[str, Any] = {}
        if reference is not None:
            # The deployment's inventory, which must refuse the history file:
            # for its missing training record (checked first), or else for its
            # missing model spec (`read_model_record`'s reason).
            deployment = runs / "inventory_deployment.json"
            assert audit.main(["--inventory", str(workspaces_root),
                               "--output", str(deployment)]) == 0
            listing = json.loads(deployment.read_text())
            name = reference.resolve().relative_to(workspaces_root.resolve()).as_posix()
            workspace = name.split("/", 1)[0]
            old = listing["workspaces"][workspace]["checkpoints"][name]["m2_contract"]
            reason = old["serve_and_measure"]["reason"] or ""
            assert old["serve_and_measure"]["verdict"] == "refused", (
                "the history file was accepted")
            refusal = next((fact for fragment, fact in (
                ("records no training_input_digests", "no record"),
                ("kg is not recorded", "no kg role"),
                (WRITTEN_BEFORE_SPEC, "no model spec"),
            ) if fragment in reason), None)
            assert refusal is not None, (
                "the history file was refused for something other than its missing kg "
                "record or its missing model spec"
            )
            facts.update(deployment_report_sha256=file_sha256(deployment),
                         deployment_counts=listing["counts"],
                         reference_refusal=refusal)
        else:
            facts["reference_refusal"] = "not given"
        # The new child lives outside the workspace, so it is listed through a
        # view: the workspace's own files, linked, and the child beside them.
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
            "the new child did not pass M2's record checks"
        )
        facts["new_model_report_sha256"] = file_sha256(new_report)
        return ("the new child passes"
                + ("; the history file is refused for its missing record or model spec"
                   if reference is not None else "; no history file given")), facts

    report.run("R7", "real",
               "the inventory passes the new child's record, and refuses a history file given",
               _inventory)

    # ---- Contract M4's flow on real data, before R8 so R8 covers it. The
    # source, the registry, the kept report and the ledger (beside
    # R4.finetune's child, which is under --work-dir) are all under the work
    # directory.
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
        assert "child" in state, "R4.finetune did not pass"
        version, n = state["m4"]["version"], state["m4"]["n"]
        # R4.finetune's child is under --work-dir, so the ledger --record
        # writes beside its resolved path is under it too.
        sha256 = _read_run_checkpoint(state["child"])["sha256"]
        code = _measure_cohort(state["child"], real, registry, M4_REAL_LABEL, device, "--record",
                               protocol="disease")
        assert code == 0, f"the measurement with --record exited {code}"
        samples = _cohort_samples(version)
        _, kept, identity, ranked = _kept_report(registry, M4_REAL_LABEL,
                                                 [s.patient_id for s in samples])
        assert kept["manifest"]["artifact_digests"]["checkpoint"] == sha256, (
            "the report does not name the checkpoint it was given")
        binding = _ranked_cases(kept, ranked, _own_binding(state["child"], real, samples))
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
            "ranks_rows": len(samples), "n_ranked": kept["n_ranked"], "binding": binding,
            "r10": _r10_facts(r10),
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
            "served": "reloaded through the reload route's handler with R4.finetune's child "
                      "named explicitly with its task and bound to its digest "
                      "(expected_checkpoint_sha256; R6 "
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
    _m5_real(report, runs, real, workspaces_root, device, state, task_sizes)

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
        if reference is not None:
            assert _size_and_mtime(reference) == reference_before, "the history file changed"
        return "the same files and directories, sizes and modification times", {
            "entries_checked": len(after),
            "history_file": "unchanged" if reference is not None else "not given",
            "compared": "listing, size and modification time; not content digests",
        }

    report.run("R8", "real",
               "the real workspace and a history file given are unchanged by listing, size and "
               "mtime", _untouched)


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
    "real_workspace", "real_reference_checkpoint", "workspaces_root",
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
        help="Training epochs of each of phase D's two runs on the demo workspace, the base "
             "and its disease child (default: 3)",
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
             "and ledger goes under --work-dir. Its models are trained by the reference "
             "recipe.",
    )
    parser.add_argument(
        "--real-reference-checkpoint", type=Path, default=None,
        help="Optional: a history file inside --workspaces-root, a checkpoint without a "
             "model spec and without a training record of its graph (no "
             "training_input_digests, or none for kg), such as the deployment's model-02, "
             "which R2 refuses for its missing model spec and the inventory (R7) for its "
             "missing record. Never read to configure a run, and not modified.",
    )
    parser.add_argument(
        "--workspaces-root", type=Path, default=None,
        help="The directory of workspaces the deployment inventory lists "
             "(default: the real workspace's parent).",
    )
    args = parser.parse_args(argv)
    # **Absolute once, here** (BACKLOG 49): every check below, the phases and
    # the server's directory check read the same path. The starting directory
    # is put in front and nothing else changes: not resolved, so a symbolic
    # link keeps the name it was given, and not normalised, since collapsing
    # `link/..` by its spelling names another file than the one the file
    # system opens (the review of b14a6e9). The containment checks below
    # resolve as they did.
    for name in PATH_ARGUMENTS:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.absolute())
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
    real_args = (args.real_reference_checkpoint, args.workspaces_root)
    if args.real_workspace is None and any(value is not None for value in real_args):
        parser.error("--real-reference-checkpoint and --workspaces-root belong to "
                     "--real-workspace")
    if args.real_workspace is not None:
        if not args.m2_acceptance:
            parser.error("--real-workspace is part of --m2-acceptance")
        if args.workspaces_root is None:
            args.workspaces_root = args.real_workspace.parent
        for name in ("real_workspace", "workspaces_root"):
            if not getattr(args, name).is_dir():
                parser.error(f"--{name.replace('_', '-')} is not a directory: "
                             f"{getattr(args, name)}")
        reference = args.real_reference_checkpoint
        if reference is not None:
            if not reference.is_file():
                parser.error(f"--real-reference-checkpoint is not a file: {reference}")
            if args.workspaces_root.resolve() not in reference.resolve().parents:
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
        inputs = [args.real_reference_checkpoint]
        if any(path is not None and report == path.resolve() for path in inputs) or (
            args.real_workspace is not None and args.real_workspace.resolve() in report.parents
        ):
            parser.error("--report must not be an input of the acceptance")
    return args


def run_exit_status(report: Report, acceptance: Optional[Mapping[str, Any]]) -> int:
    """The run's exit status, the verdict an external check reads: 1 when a
    probe failed or raised, or a requested acceptance is not complete; else 0.
    **An informational probe never counts** (`INFORMATIONAL_PROBES`): the
    report keeps its status, and the console prints its failure as an INFO
    line, so hardware variance in a reading never fails the run."""
    failed = sum(1 for probe in report.probes
                 if probe.status in ("failed", "error")
                 and probe.probe_id not in INFORMATIONAL_PROBES)
    if acceptance is not None and not acceptance["complete"]:
        failed += 1
    return 1 if failed else 0


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
            # D1's disease child goes to phases E and M, its base to M2.
            trained = phase_training(report, work, sound, device, args.epochs)
            if trained is not None:
                phase_serving(report, work, sound, trained.child, device)
                if args.m2_acceptance and device == "cuda":
                    phase_m2_demo(report, work, sound, trained.child, trained.base, device)
        if args.real_workspace is not None and device == "cuda":
            phase_m2_real(report, work, args.real_workspace, args.real_reference_checkpoint,
                          args.workspaces_root, device)
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

    for probe in report.probes:
        if probe.probe_id in INFORMATIONAL_PROBES and probe.status in ("failed", "error"):
            print(f"INFO {probe.probe_id} {probe.status}, informational and outside the exit "
                  f"status: {probe.detail}")
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
    failed = run_exit_status(report, acceptance)
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
    return failed


if __name__ == "__main__":
    # Before main(): the environment phase is the first thing to import torch.
    ALLOCATOR.update(_apply_allocator())
    _stop_on_termination_signals()
    sys.exit(main())
