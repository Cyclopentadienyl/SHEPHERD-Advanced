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
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

REPORT_SCHEMA_VERSION = 1

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


# =============================================================================
# Filesystem snapshots — the property every refusal has to satisfy
# =============================================================================
def snapshot(directory: Path) -> Dict[str, Tuple[int, str]]:
    """Every file under `directory`, by modification time and content digest.

    Both, deliberately. Bytes alone miss a rewrite with identical content;
    modification time alone is a weaker claim than "the same file".
    """
    from src.utils.fingerprint import file_sha256

    if not directory.exists():
        return {}
    return {
        str(path.relative_to(directory)): (path.stat().st_mtime_ns, file_sha256(path))
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def assert_untouched(
    before: Dict[str, Tuple[int, str]], directory: Path, what: str
) -> None:
    after = snapshot(directory)
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    assert not added, f"{what} created {added}"
    assert not removed, f"{what} removed {removed}"
    assert not changed, f"{what} rewrote {changed}"


def refuses(fn: Callable[[], Any], expected: type, fragment: str) -> None:
    """Call `fn`, require it to raise `expected` with `fragment` in the message.

    The fragment is checked here and discarded here. It never reaches the
    report, because refusal messages name the paths they refused.
    """
    try:
        fn()
    except expected as exc:
        assert fragment in str(exc), (
            f"refused, but not for the expected reason ({fragment!r})"
        )
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"raised {type(exc).__name__} rather than {expected.__name__}"
        ) from None
    raise AssertionError(f"accepted an input that should have been refused ({fragment!r})")


# =============================================================================
# Phase A — what this machine is
# =============================================================================
def phase_environment(report: Report, device: str) -> Dict[str, Any]:
    print("\nA. Environment")
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
    """The demo graph plus one disease, so its `kg.json` is genuinely other."""
    from src.core.types import DataSource, NodeType
    from src.kg.graph import Node, NodeID

    kg = _demo_kg()
    kg.add_node(Node(
        id=NodeID(source=DataSource.MONDO, local_id="MONDO:9999999"),
        node_type=NodeType.DISEASE,
        name="Probe-only disease",
        attributes={"mondo_id": "MONDO:9999999", "name": "Probe-only disease"},
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
    from src.evaluation.cohort import verify_generated_cohorts
    from src.kg.artifacts import GRAPH_ARTIFACTS, verify_graph_artifacts, verify_graph_source

    def _accepted() -> Tuple[str, Dict[str, Any]]:
        digests = verify_graph_artifacts(sound)
        verify_graph_source(sound / "kg.json", sound)
        cohorts = verify_generated_cohorts(sound)
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
        from src.utils.fingerprint import file_sha256

        return "every verifier accepted it", {
            # Counts and digests only: per-disease lists are forbidden in
            # evidence artifacts, and the identifiers are what those lists are.
            "kg_digest": digests["kg"],
            "workspace_digests": {
                path.name: file_sha256(path)
                for path in sorted(sound.iterdir()) if path.is_file()
            },
            "train_diseases": len(cohorts.disease_sets["train"]),
            "val_diseases": len(cohorts.disease_sets["val"]),
            "disease_overlap": len(overlap),
        }

    report.run(
        "C1", "workspace", "CONTROL: the sound workspace verifies", _accepted
    )

    for role in sorted(GRAPH_ARTIFACTS):
        def _replaced(role=role):
            copy = work / f"replaced_{role}"
            if copy.exists():
                shutil.rmtree(copy)
            shutil.copytree(sound, copy)
            (copy / GRAPH_ARTIFACTS[role]).write_bytes(b"same shape, another workspace")
            refuses(
                lambda: verify_graph_artifacts(copy), ValueError,
                f"is not the {role} artifact",
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
            lambda: verify_graph_artifacts(copy), ValueError,
            "has no split_manifest.json",
        )
        shutil.rmtree(copy)
        return "an unbound workspace was refused", {}

    report.run(
        "C3", "workspace", "a workspace with no manifest is refused", _no_manifest
    )

    def _crossed() -> Tuple[str, Dict[str, Any]]:
        # **A different graph, not the same graph under a different seed.** The
        # first version of this probe built the other workspace with seed+1 and
        # copied its kg.json across -- but the demo graph is deterministic, so
        # the two files were byte-identical and nothing was crossed. It failed
        # loudly, which is the only reason it was caught. The seed moves the
        # allocation; only the graph moves the graph.
        other = work / "other_workspace"
        if not other.exists():
            from src.kg.workspace import write_workspace

            write_workspace(
                _other_kg(), other, feature_dim=FEATURE_DIM, samples=_budget(),
            )
        assert (other / "kg.json").read_bytes() != (sound / "kg.json").read_bytes(), (
            "the two workspaces hold the same graph, so this probe crosses nothing"
        )
        copy = work / "crossed"
        if copy.exists():
            shutil.rmtree(copy)
        shutil.copytree(sound, copy)
        (copy / "kg.json").write_bytes((other / "kg.json").read_bytes())
        refuses(
            lambda: verify_graph_artifacts(copy), ValueError, "is not the kg artifact"
        )
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
            resolve_checkpoint_dir,
            select_checkpoint_in_dir,
        )

        config = TrainConfig(
            data_dir=str(sound),
            # **Named, not defaulted.** TrainConfig writes to "outputs" and
            # "logs" relative to the working directory, so a probe that left
            # them alone would scatter run artifacts through the repository.
            output_dir=str(work / "train_outputs"),
            log_dir=str(work / "train_logs"),
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

        ckpt_dir = resolve_checkpoint_dir(str(sound), "gat", None)
        chosen = select_checkpoint_in_dir(ckpt_dir)
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
        import torch

        from src.kg.artifacts import verify_graph_artifacts

        checkpoint = torch.load(selected["path"], map_location="cpu", weights_only=False)
        digests = verify_graph_artifacts(sound)
        recorded = checkpoint.get("training_input_digests")
        assert recorded is not None, "the checkpoint records no training inputs"
        # **The roles must be present, not merely consistent.** Comparing only
        # the roles that happen to appear would pass against a checkpoint that
        # recorded an empty map -- the vacuous version of this check. `kg` is
        # deliberately not among them: training never opens kg.json, so the run
        # did not consume it.
        required = {"node_features", "edge_indices", "num_nodes"}
        assert required <= set(recorded), (
            f"the checkpoint omits the graph roles it read: "
            f"{sorted(required - set(recorded))}"
        )
        mismatched = sorted(
            role for role in required if recorded[role] != digests[role]
        )
        assert not mismatched, f"the checkpoint names other bytes for {mismatched}"
        return "the checkpoint names the workspace it was trained on", {
            "recorded_roles": sorted(recorded),
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
# Phase F — the real-data build (opt-in; minutes, not seconds)
# =============================================================================
def phase_real_build(
    report: Report, work: Path, external: Path, num_train: int, num_val: int
) -> None:
    print("\nF. The real-data build")

    def _build() -> Tuple[str, Dict[str, Any]]:
        from scripts.build_knowledge_graph import build_knowledge_graph
        from src.evaluation.cohort import verify_generated_cohorts
        from src.kg.artifacts import verify_graph_artifacts, verify_graph_source

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
        )
        elapsed = round(time.time() - started, 1)

        digests = verify_graph_artifacts(workspace)
        verify_graph_source(workspace / "kg.json", workspace)
        cohorts = verify_generated_cohorts(workspace)
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

        All seven files, not the four graph artifacts. The first version of this
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
        )
        # The seven files a generated workspace is: the four graph artifacts,
        # both cohorts and the manifest. Named rather than globbed, so "the same
        # workspace" means the same thing here as it does to the verifiers, and
        # an unrelated file dropped into the directory neither joins the claim
        # nor breaks it.
        from src.kg.artifacts import GRAPH_ARTIFACTS, MANIFEST_FILENAME

        names = sorted(
            list(GRAPH_ARTIFACTS.values())
            + ["train_samples.json", "val_samples.json", MANIFEST_FILENAME]
        )
        digests = {name: file_sha256(first / name) for name in names}
        differing = sorted(
            name for name in names
            if file_sha256(second / name) != digests[name]
        )
        missing = sorted(
            name for name in names if not (second / name).exists()
        )
        assert not missing, f"the second build did not write {missing}"
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
    parser.add_argument("--num-train", type=int, default=200000)
    parser.add_argument("--num-val", type=int, default=40000)
    parser.add_argument(
        "--keep", action="store_true",
        help="Keep the work directory instead of removing it at the end",
    )
    return parser.parse_args(argv)


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
    environment = phase_environment(report, args.device)
    device = environment.get("resolved_device", "cpu")

    sound = phase_writer(report, work)
    if sound is not None:
        phase_workspace(report, work, sound)
        checkpoint = phase_training(report, work, sound, device, args.epochs)
        if checkpoint is not None:
            phase_serving(report, work, sound, checkpoint, device)
    if args.external_dir is not None:
        phase_real_build(report, work, args.external_dir, args.num_train, args.num_val)

    summary = report.summary()
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment,
        "settings": {
            "device": device,
            "epochs": args.epochs,
            "feature_dim": FEATURE_DIM,
            "seed": SEED,
            "real_build_requested": args.external_dir is not None,
        },
        "summary": summary,
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
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))

    print("\n" + "=" * 66)
    print(
        f"passed {summary['passed']}  failed {summary['failed']}  "
        f"errors {summary['error']}  skipped {summary['skipped']}"
    )
    print(f"report: {report_path}")
    print("=" * 66)

    failed = summary["failed"] + summary["error"]
    # **Containment, not parent equality.** A report at `<work>/sub/report.json`
    # has a parent that is not `work`, and removing `work` would take the report
    # with it -- the first version of this check did exactly that.
    report_inside_work = work.resolve() in report_path.resolve().parents
    if not failed and not args.keep:
        if report_inside_work:
            print("(work directory kept because the report lives inside it)")
        else:
            shutil.rmtree(work, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
