#!/usr/bin/env python
"""
Mode A calibration — run both scorers on one seeded stream and compare.
======================================================================
**SUPERSEDED — this launcher cannot run, and its target no longer exists.**

    It drives `scripts/evaluate_model.py`, and no checkpoint in the scanned
    family carries the `metadata` / `in_channels_dict` keys that script's loader
    needs (BACKLOG §3.1, M1-M2). Frozen-evaluator bit parity is therefore
    unexecutable, not merely unrun, and it has been **retired as the acceptance
    target** rather than deferred.

    The replacement is `src/evaluation/differential.py`: the same batches handed
    to `Trainer._run_evaluation_pass` and to the Mode A harness, compared per
    sample. Its reference is the trainer's own validation calculation, which is
    code that runs, instead of an artifact that cannot be reproduced.

    **This file is kept, not deleted, and it is not yet rewritten.** Calibration
    still happens; only its reference changed, and the rewrite belongs with item
    7a, which is the institutional run and needs a designated loadable checkpoint
    to be verifiable against. Rewriting it now would produce a launcher nothing
    could execute and nothing could check. Everything below this banner describes
    the retired parity run and is retained as the reasoning behind it — history,
    not instructions.

The acceptance gate for B-0.2, made executable. It runs the frozen evaluator and
`scripts/measure_scorer.py` over the same data, then compares the only artifacts
the frozen evaluator writes: its report's `mrr`, and its per-sample predictions.

**Why a launcher exists at all.** The frozen evaluator seeds nothing, and it may
not be modified — its whole value is being the unmodified thing that produced the
reference numbers. Negative disease sampling draws from Python's `random`
(`src/kg/data_loader.py:667`) and neighbour sampling from `random.sample` /
`random.choice` (`:220, :263`), so two runs of the same command build different
subgraphs and therefore different candidate universes. Comparing them would
measure the sampler.

Seeding is therefore done **around** the frozen script rather than inside it: each
side runs in its own subprocess whose first act is to seed `random`, `numpy` and
`torch`, after which the script is executed unmodified via `runpy`. Nothing is
patched, monkeypatched or edited.

**Three settings are part of that stream, not preferences:**

  - **worker count.** Negatives are drawn inside dataloader worker processes, and
    PyTorch seeds each worker as `base_seed + worker_id` where `base_seed` is
    drawn from the parent's torch RNG. Seeding the parent therefore does determine
    the workers — but only for the same worker count. The frozen evaluator
    hardcodes `EvalConfig.num_workers = 4` (`scripts/evaluate_model.py:113`) with
    no flag to change it, so this launcher gives `measure_scorer.py` the same 4.
  - **batch size**, because Mode A's candidate universe is the batch's subgraph.
  - **device**, resolved here to an explicit value and passed to both, because the
    frozen evaluator's `auto` falls back to CPU in silence while
    `measure_scorer.py`'s refuses to. Letting each resolve `auto` on its own is
    the one way they could end up on different hardware.

**What this proves, and what it does not.** `comparison_passed` says the two
scorers produced identical per-sample rows and an identical aggregate on this
input with this seed. `cuda_executed` says the run happened on CUDA. Both are
machine-checkable, and both are named for exactly what they check.

**Neither is institutional acceptance, and this file cannot establish it.** A
synthetic workspace run on a CUDA machine satisfies both. Acceptance is a human
confirming that the recorded `artifact_digests` are the institution's real
checkpoint and cohort and that `deployment_host` is the deployment machine —
facts no code in this repository has any way to verify. The verdict records what
is needed for that confirmation and says in writing that it is external. **No
registry of approved artifacts is built here**; that would be a claim about
institutional process, which is not B-0.2's to make.

Seeding sufficiency is likewise *checked*, not proven: a divergent stream shows
up as a row mismatch rather than as a quietly different number. A FAIL is
therefore informative and is not a reason to retry with another seed.

    python scripts/calibrate_mode_a.py \\
        --checkpoint checkpoints/best.pt \\
        --data-dir data/processed \\
        --split val --seed 20260818 \\   # val is not held-out; see below
        --workdir reports/calibration

Module: scripts/calibrate_mode_a.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# One hashing implementation, shared. A second one here could differ from the
# harness's in exactly the way the digests exist to detect.
from scripts.measure_scorer import artifact_digests  # noqa: E402

logger = logging.getLogger("calibrate_mode_a")

# The frozen evaluator hardcodes this and exposes no flag (`evaluate_model.py:113`).
ORACLE_NUM_WORKERS = 4

SEED_BOOTSTRAP = """
import random, runpy, sys
import numpy, torch

seed = int(sys.argv[1])
script = sys.argv[2]
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

sys.argv = [script, *sys.argv[3:]]
runpy.run_path(script, run_name="__main__")
"""
"""Seed, then run the target script unmodified.

`runpy.run_path(..., run_name="__main__")` fires the target's own
`if __name__ == "__main__"` block, so its `sys.exit(main())` becomes this
process's exit status. Seeding happens before the script imports anything, which
is what a `--seed` flag inside the script would achieve — except that the frozen
evaluator has no such flag and may not be given one.
"""


def _run(seed: int, script: Path, args: List[str], cwd: Path,
         timeout: Optional[float]) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        [sys.executable, "-c", SEED_BOOTSTRAP, str(seed), str(script), *args],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout[-4000:])
        sys.stderr.write(completed.stderr[-4000:])
        raise SystemExit(f"{script.name} failed with exit code {completed.returncode}")
    return completed


def _resolve_device(requested: str) -> Tuple[str, bool]:
    """One explicit device for both sides, and whether it was CUDA.

    Resolved here rather than in each script, because the two resolve `auto`
    differently by design and a calibration run on two different devices compares
    nothing.

    The boolean is `cuda_executed` and means only that: the run happened on CUDA.
    It is not an eligibility verdict — see the note on `institutional_acceptance`
    in `main`.
    """
    import torch

    if requested != "auto":
        return requested, requested == "cuda"
    if torch.cuda.is_available():
        return "cuda", True
    logger.warning(
        "No CUDA here. Running the procedure on CPU: it exercises the comparison, "
        "but the result will be recorded as cuda_executed=false."
    )
    return "cpu", False


def run_oracle(seed: int, workdir: Path, args: argparse.Namespace, device: str) -> Tuple[dict, list]:
    """Run the frozen evaluator and read back both files it writes.

    `cwd=workdir` because it has no predictions-path option and writes
    `predictions_{split}.json` relative to the working directory.
    """
    report_path = workdir / "oracle_report.json"
    _run(
        seed,
        PROJECT_ROOT / "scripts" / "evaluate_model.py",
        [
            "--checkpoint", str(args.checkpoint),
            "--data-dir", str(args.data_dir),
            "--split", args.split,
            "--output", str(report_path),
            "--batch-size", str(args.batch_size),
            "--device", device,
            "--save-predictions",
        ],
        cwd=workdir,
        timeout=args.timeout,
    )
    predictions_path = workdir / f"predictions_{args.split}.json"
    if not predictions_path.exists():
        raise SystemExit(f"the frozen evaluator wrote no {predictions_path.name}")
    return json.loads(report_path.read_text()), json.loads(predictions_path.read_text())


def run_harness(seed: int, workdir: Path, args: argparse.Namespace, device: str) -> Tuple[dict, list]:
    measurement_path = workdir / "measurement.json"
    predictions_path = workdir / "measurement_predictions.json"
    _run(
        seed,
        PROJECT_ROOT / "scripts" / "measure_scorer.py",
        [
            "--checkpoint", str(args.checkpoint),
            "--data-dir", str(args.data_dir),
            "--split", args.split,
            "--output", str(measurement_path),
            "--predictions-output", str(predictions_path),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(ORACLE_NUM_WORKERS),
            "--device", device,
            "--seed", str(seed),
        ],
        cwd=workdir,
        timeout=args.timeout,
    )
    return json.loads(measurement_path.read_text()), json.loads(predictions_path.read_text())


def compare(
    oracle_report: dict,
    oracle_predictions: list,
    measurement: dict,
    harness_predictions: list,
    args: argparse.Namespace,
    digests_before: Dict[str, Optional[str]],
    digests_after: Dict[str, Optional[str]],
) -> List[str]:
    """Return the list of failures. Empty means the two agree.

    Every check is stated as a claim that could be false, and the preconditions
    come first: two MRRs truncated at different K, or computed over differently
    sized batches, are not the same quantity, and comparing them would produce a
    PASS or a FAIL that means nothing either way.
    """
    from src.evaluation.measurement import LEGACY_TRUNCATION_K

    failures: List[str] = []
    config = oracle_report.get("config", {})

    # --- did both runs consume the same bytes? ------------------------------
    # Taken before the oracle starts and again after the harness finishes, so a
    # checkpoint replaced or a samples file rewritten between the two runs is a
    # failure rather than an unexplained disagreement further down. The harness
    # hashes its own inputs *after* loading them, which is why its manifest is
    # checked against the before-image rather than trusted on its own.
    if digests_before != digests_after:
        changed = sorted(
            role for role in digests_before
            if digests_before[role] != digests_after.get(role)
        )
        failures.append(
            f"artifacts changed during the run: {', '.join(changed)}. The two runs "
            "did not consume the same bytes, so nothing below is a comparison of "
            "the same measurement"
        )
    if measurement["manifest"]["artifact_digests"] != digests_before:
        failures.append(
            "the harness manifest's digests do not match the artifacts as they "
            "were before the oracle ran; the harness measured different bytes"
        )

    # --- preconditions: is this comparison meaningful at all? ---------------
    oracle_k = max(config.get("top_k_values", [])) if config.get("top_k_values") else None
    if oracle_k != LEGACY_TRUNCATION_K:
        failures.append(
            f"the oracle truncated its MRR input at {oracle_k}, the harness at "
            f"{LEGACY_TRUNCATION_K}. These are different quantities and the "
            "comparison below would be meaningless"
        )
    if config.get("num_workers") != ORACLE_NUM_WORKERS:
        failures.append(
            f"the oracle ran with num_workers={config.get('num_workers')}, but the "
            f"harness was given {ORACLE_NUM_WORKERS}. Negatives are drawn in the "
            "workers, so the two consumed different random streams"
        )
    if config.get("batch_size") != args.batch_size:
        failures.append(
            f"the oracle ran with batch_size={config.get('batch_size')}, the harness "
            f"with {args.batch_size}. Mode A's candidate universe is the batch's "
            "subgraph, so this changes what was measured"
        )

    manifest = measurement["manifest"]
    if manifest["n_samples"] != oracle_report["metadata"]["num_samples"]:
        failures.append(
            f"cohort size differs: oracle {oracle_report['metadata']['num_samples']}, "
            f"harness {manifest['n_samples']}"
        )

    # --- the aggregate ------------------------------------------------------
    legacy_key = f"legacy_mrr_truncated_at_{LEGACY_TRUNCATION_K}"
    oracle_mrr = oracle_report["metrics"]["mrr"]
    harness_mrr = measurement["legacy_metrics"][legacy_key]
    if abs(oracle_mrr - harness_mrr) > 1e-12:
        failures.append(
            f"MRR differs: oracle {oracle_mrr!r}, harness {harness_mrr!r} "
            f"(delta {oracle_mrr - harness_mrr:+.3e})"
        )

    # --- the per-sample rows ------------------------------------------------
    if len(oracle_predictions) != len(harness_predictions):
        failures.append(
            f"prediction row count differs: oracle {len(oracle_predictions)}, "
            f"harness {len(harness_predictions)}"
        )
        return failures

    mismatched_rows = 0
    first_mismatch: Optional[str] = None
    for index, (left, right) in enumerate(zip(oracle_predictions, harness_predictions)):
        if left == right:
            continue
        mismatched_rows += 1
        if first_mismatch is None:
            reason = "predictions"
            if left["sample_id"] != right["sample_id"]:
                reason = "sample_id — the cohorts are not even in the same order"
            elif left["ground_truth"] != right["ground_truth"]:
                reason = "ground_truth — the id translation disagrees"
            first_mismatch = (
                f"row {index} ({left['sample_id']}) differs on {reason}:\n"
                f"      oracle  {left}\n"
                f"      harness {right}"
            )
    if mismatched_rows:
        failures.append(
            f"{mismatched_rows} of {len(oracle_predictions)} prediction rows differ. "
            f"{first_mismatch}"
        )
    return failures


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen evaluator and the Mode A harness on one seeded "
                    "stream and compare their artifacts",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", required=True,
                        choices=["train", "val", "test"],
                        help='Which samples file to measure. **Required — there is no default.** Generated workspaces normally contain train and val only; a test split exists only where an evaluation protocol created one. `val` is not held-out generalisation, for two independent reasons. (1) It is the checkpoint-selection split under the current trainer (early_stopping_monitor=val_mrr), so metrics measured on it are model-selection-contaminated. (2) `sample_generator.py` draws one sample pool, shuffles it and slices it into train and val — it never partitions by disease, so the two share diseases by construction. Audited on the deployment workspace at 100%%: all 7,970 val diseases appear in train (docs/working/EVIDENCE_M4.json, which records both split digests).')
    parser.add_argument("--workdir", type=Path, required=True,
                        help="Where both runs write. Created if absent")
    parser.add_argument("--seed", type=int, required=True,
                        help="Required, not optional: without it the two runs build "
                             "different subgraphs and the comparison measures the "
                             "sampler rather than the scorer")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="Resolved once and passed to both runs explicitly")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Per-run wall-clock limit in seconds. Unset by default, "
                             "because an institutional run over a real cohort must "
                             "not be killed part way through; a test that drives this "
                             "launcher passes one so a hang fails instead of hanging")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    args.checkpoint = args.checkpoint.resolve()
    args.data_dir = args.data_dir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)
    workdir = args.workdir.resolve()

    device, cuda_executed = _resolve_device(args.device)
    logger.info("seed=%d device=%s batch_size=%d workers=%d",
                args.seed, device, args.batch_size, ORACLE_NUM_WORKERS)

    digests_before = artifact_digests(args.checkpoint, args.data_dir, args.split)

    logger.info("Running the frozen evaluator...")
    oracle_report, oracle_predictions = run_oracle(args.seed, workdir, args, device)
    logger.info("Running the Mode A harness...")
    measurement, harness_predictions = run_harness(args.seed, workdir, args, device)

    digests_after = artifact_digests(args.checkpoint, args.data_dir, args.split)

    failures = compare(
        oracle_report, oracle_predictions, measurement, harness_predictions, args,
        digests_before, digests_after,
    )
    comparison_passed = not failures

    verdict: Dict[str, Any] = {
        # Named for exactly what each is evidence of. `comparison_passed` says the
        # two scorers agreed on this input; `cuda_executed` says the run happened
        # on CUDA. Neither is institutional acceptance, and the previous names
        # (`passed`, `gate_eligible`) implied otherwise -- a synthetic workspace on
        # a CUDA machine would have reported `gate_eligible: true`.
        "comparison_passed": comparison_passed,
        "cuda_executed": cuda_executed and measurement["manifest"]["cuda_executed"],
        "institutional_acceptance": (
            "EXTERNAL. This file cannot establish it. Acceptance is a human "
            "confirming that artifact_digests below are the institution's real "
            "checkpoint and cohort, and that deployment_host is the deployment "
            "machine. Nothing in this repository can verify either."
        ),
        "deployment_host": platform.node(),
        "seed": args.seed,
        "device": device,
        "batch_size": args.batch_size,
        "num_workers": ORACLE_NUM_WORKERS,
        "split": args.split,
        "n_samples": measurement["manifest"]["n_samples"],
        "oracle_mrr": oracle_report["metrics"]["mrr"],
        "artifact_digests": digests_before,
        "software_revision": measurement["manifest"]["software_revision"],
        "failures": failures,
    }
    verdict_path = workdir / "calibration.json"
    verdict_path.write_text(json.dumps(verdict, indent=2, allow_nan=False))

    print("\n" + "=" * 68)
    print(f"  Mode A comparison — {'PASS' if comparison_passed else 'FAIL'}")
    print("=" * 68)
    for failure in failures:
        print(f"  - {failure}")
    if comparison_passed and not verdict["cuda_executed"]:
        print("  The two scorers agree, but this run did not execute on CUDA.")
        print("  The procedure works; an acceptance claim needs the institutional")
        print("  hardware as well.")
    elif comparison_passed:
        print(f"  {verdict['n_samples']} samples, identical rows, MRR "
              f"{verdict['oracle_mrr']!r}.")
    print("\n  Institutional acceptance is external to this file: a human must")
    print("  confirm the recorded artifact digests and deployment host."),
    print(f"\n  Written to {verdict_path}\n")
    return 0 if comparison_passed else 1


if __name__ == "__main__":
    sys.exit(main())
