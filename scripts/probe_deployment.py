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
    # workspace and the reference checkpoint are read, never written.
    python scripts/probe_deployment.py --m2-acceptance --device cuda \
        --work-dir <empty directory> --report <empty directory>/m2_report.json \
        --real-workspace data/workspaces/<workspace> \
        --real-config <config.yaml with the established training settings> \
        --real-reference-checkpoint <path to the reference model's .pt file>

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
    size and modification time, not by content digest (R8). Every run writes
    into its own run directory under --work-dir.
    The report stays inside --work-dir and replaces no existing file.
    Run as a program, the probe applies the project's allocator rule before
    torch is imported, as train_model.py does, and the report records the
    allocator variables, their source and the backend torch reports.
    The report's `m2_acceptance` says, per part, whether every required probe
    ran and passed; a missing, skipped or failed one leaves it incomplete, and
    the exit status is non-zero until both parts are complete.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
) -> None:
    """Call `fn`, require it to raise `expected` with `fragment` in the message;
    a tuple of fragments requires every one of them.

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
        return
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
    them (`verify_cohort_reads`). Nothing is hashed by path. Returns the
    manifest-bound digests of the four graph roles, the verified cohorts, and the
    digest of each file as these reads returned it, by file name.
    """
    from src.evaluation.cohort import GENERATED_SPLITS, resolve_cohort, verify_cohort_reads
    from src.kg.artifacts import GRAPH_ARTIFACTS, read_split_manifest, verify_graph_source_read
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
    cohorts = verify_cohort_reads(manifest, samples)
    reads = [manifest.identity, graph, *tensors.reads.values(),
             *(read.identity for read in samples.values())]
    return bound, cohorts, {Path(read.path).name: read.sha256 for read in reads}


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
    if describe_only:
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
    return facts


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
                         ("kg", "node_features", "edge_indices", "num_nodes")}
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
                          ("kg", "node_features", "edge_indices")},
                       "num_nodes": int("1" * 64)}), "not SHA-256"),
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
        digests, cohorts, read = verify_workspace(sound)
        assert cohorts.verified == ("train", "val")
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
        # verified; a file they do not parse (the provenance record) is
        # identified by path, which is all a listing of it can be.
        from src.utils.fingerprint import file_sha256

        return "every verifier accepted it", {
            # Counts and digests only: per-disease lists are forbidden in
            # evidence artifacts, and the identifiers are what those lists are.
            "kg_digest": digests["kg"],
            "workspace_digests": {
                path.name: read[path.name] if path.name in read else file_sha256(path)
                for path in sorted(sound.iterdir()) if path.is_file()
            },
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
        digests, _, _ = verify_workspace(sound)
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
        return "the pipeline built and the GNN is ready", {
            "scoring_mode": config.get("scoring_mode"),
            "sp_ready": bool(config.get("sp_ready")),
            "kg_nodes": config.get("kg_nodes"),
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
        jsonable_encoder(_status_of(state["bundle"].config, None, None))
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
            "model_version": api.app_state.model_version,
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
        assert api.app_state.model_version == before["model_version"], "the version moved"
        assert api.app_state._current_data_dir == before["data_dir"], "the workspace moved"
        assert api.app_state._current_checkpoint_path == before["checkpoint"], (
            "the checkpoint moved"
        )
        shutil.rmtree(broken)
        return "the refusal left every served field untouched", {}

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
        return "a diagnosis ran end to end", {
            "candidates": len(candidates),
            "score_range": [round(min(scores), 4), round(max(scores), 4)],
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
)
M2_REAL_REQUIRED: Tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8")


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
        "scope": (
            "M2's source contract on this machine, through the production entry "
            "points. The real-data model is a short training that shows the "
            "pipeline and its checks work; it is not a model of serving quality, "
            "which is chosen by the established training and evaluation."
        ),
    }


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
        "--split", "val", "--cohort-kind", "generated",
        "--output", str(output), "--device", device,
    ])


def _measurement_facts(output: Path, checkpoint_sha: str, kg_digest: str) -> Dict[str, Any]:
    from src.utils.fingerprint import file_sha256

    assert output.is_file(), "the measurement wrote no report"
    result = json.loads(output.read_text())
    digests = result["manifest"]["artifact_digests"]
    assert digests.get("checkpoint") == checkpoint_sha, (
        "the measurement does not name the checkpoint it was given"
    )
    assert digests.get("kg") == kg_digest, "the measurement records another source graph"
    return {
        "report_sha256": file_sha256(output),
        "checkpoint_sha256": checkpoint_sha,
        "n_ranked": result.get("n_ranked"),
        "authoritative_metrics": result.get("authoritative_metrics"),
    }


def _serve_and_diagnose(
    kg_path: Path, checkpoint: Path, data_dir: Path, device: str
) -> Dict[str, Any]:
    """The production serving build (`build_pipeline`), shortest paths verified
    by its own loader, and one diagnosis through the built pipeline."""
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
    return {
        "scoring_mode": config.get("scoring_mode"),
        "sp_kg_binding": config.get("sp_kg_binding"),
        "sp_hop_bound_source": config.get("sp_hop_bound_source"),
        "sp_max_hops": config.get("sp_max_hops"),
        "parameter_devices": sorted(devices),
        "candidates": len(scores),
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


def phase_m2_demo(
    report: Report, work: Path, sound: Path, checkpoint: Path, device: str
) -> None:
    """Contract M2's acceptance on the demo workspace: the successes first, then
    every refusal, each from the production entry point that makes it."""
    print("\nM. Contract M2 on the demo workspace")
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
            output, _read_run_checkpoint(state["child"])["sha256"],
            verify_workspace(sound)[0]["kg"])

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
        facts = _serve_and_diagnose(served / "kg.json", state["child"], served, device)
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


def phase_m2_real(
    report: Report, work: Path, real: Path, config_path: Path, reference: Path,
    workspaces_root: Path, device: str,
) -> None:
    """Contract M2's acceptance on real data: the full success path, from the
    production entry points, on a workspace that is read and never written."""
    print("\nR. Contract M2 on real data")
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

        digests, cohorts, _ = verify_workspace(real)
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
        state["kg"] = digests["kg"]
        return "the workspace's graph and training inputs verify", {
            "kg_digest": digests["kg"],
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
            output, _read_run_checkpoint(state["child"])["sha256"], state["kg"])

    report.run("R5", "real", "measurement accepts and measures the new checkpoint", _measured)

    def _served() -> Tuple[str, Dict[str, Any]]:
        assert "child" in state, "R4 did not pass"
        facts = _serve_and_diagnose(real / "kg.json", state["child"], real, device)
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

        digests, cohorts, _ = verify_workspace(workspace)
        overlap = cohorts.disease_sets["train"] & cohorts.disease_sets["val"]
        assert not overlap, "the real cohorts share a disease"
        return "a real workspace was built and verified", {
            "kg_digest": digests["kg"],
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
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attack this deployment and report what held",
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
        help="MONDO file for the real-data build, passed to it unchanged. "
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
             "accounts for every probe it requires. Needs --device cuda; keeps the "
             "work directory as evidence.",
    )
    parser.add_argument(
        "--real-workspace", type=Path, default=None,
        help="Adds M2's acceptance on this real workspace (phase R): it is read, "
             "never written; every run, checkpoint and report goes under --work-dir.",
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
    if args.download_only and args.external_dir is not None:
        parser.error("--download-only runs only the environment and download "
                     "phases; drop --external-dir or --download-only")
    if args.m2_acceptance and args.download_only:
        parser.error("--m2-acceptance trains and serves; it cannot run with --download-only")
    if args.m2_acceptance and args.device != "cuda":
        parser.error("--m2-acceptance is an acceptance on the deployment's device: "
                     "pass --device cuda, so it cannot become a CPU run")
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
                parser.error(f"--{name.replace('_', '-')} is not a directory")
        for name in ("real_config", "real_reference_checkpoint"):
            if not getattr(args, name).is_file():
                parser.error(f"--{name.replace('_', '-')} is not a file")
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
    work = args.work_dir
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
    sys.exit(main())
