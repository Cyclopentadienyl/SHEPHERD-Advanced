#!/usr/bin/env python
"""
Measure the disease scorer — Mode A.
====================================
The offline counterpart to `scripts/evaluate_model.py`, which this replaces once
institutional Mode A calibration succeeds.

**A thin CLI.** Argument parsing, artifact loading, manifest assembly, and one
call into `src.evaluation.measurement`. No measurement logic lives here, so the
harness stays testable without a subprocess.

Mode A reproduces the legacy candidate construction *deliberately* — it is the
control against which modes B, C and D are read, and a control that has been
improved is no longer one. It reports two metric families: the legacy truncated
MRR, for comparison against the frozen evaluator, and the untruncated
authoritative metrics, which are what the modes are compared with each other on.

    python scripts/measure_scorer.py \
        --checkpoint checkpoints/best.pt \
        --data-dir data/processed \
        --split test \
        --output measurement.json

Two artifacts come out: `--output`, the measurement report a human reads, and
`--predictions-output`, the per-sample rows in the frozen evaluator's own shape.
The second exists to be diffed, and **is the calibration evidence** — the report
alone cannot show that the two scorers agree sample by sample.

**Do not run this by hand to calibrate.** `scripts/calibrate_mode_a.py` runs both
scorers on one seeded stream with matching batch size, worker count and device,
and compares what they wrote. Running the two separately leaves them on different
random streams, and a comparison of two different candidate universes is not a
calibration.

Module: scripts/measure_scorer.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def _file_sha256(path: Path) -> Optional[str]:
    """Raw content digest, or ``None`` if the file is not there.

    `hashlib.file_digest` (stdlib, 3.11+) reads in chunks, so a multi-gigabyte
    checkpoint is not loaded into memory to be identified. This hashes **bytes**
    and nothing else: no canonical form, no key ordering, no serialisation policy.
    Two runs quoting the same digest consumed the same file, which is the entire
    claim being made.
    """
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _artifact_digests(checkpoint: Path, data_dir: Path, split: str) -> Dict[str, Optional[str]]:
    """Every file a Mode A number depends on, by role.

    Paths are recorded too, but a path is not an identity — `checkpoints/best.pt`
    names a different file after every improvement — and the structural
    fingerprint is not one either, since two checkpoints trained on the same graph
    share it.
    """
    return {
        "checkpoint": _file_sha256(checkpoint),
        "samples": _file_sha256(data_dir / f"{split}_samples.json"),
        "node_features": _file_sha256(data_dir / "node_features.pt"),
        "edge_indices": _file_sha256(data_dir / "edge_indices.pt"),
        "num_nodes": _file_sha256(data_dir / "num_nodes.json"),
    }


def _resolve_device(requested: str) -> Tuple[torch.device, bool]:
    """Return the device and whether a run on it may clear the parity gate.

    **CUDA is a hard project requirement**, so `auto` falling back to CPU would
    silently produce a number that looks institutional and is not: different
    kernels, different reduction orders, and no statement at all about the
    hardware the tool actually runs on. `auto` therefore fails without CUDA rather
    than substituting a device nobody asked for.

    Explicit `--device cpu` stays available, because development in a container
    without a GPU is real work — but it is marked **calibration-ineligible** in the
    manifest, so the number cannot later be quoted as one that cleared the gate.
    """
    if requested == "auto":
        if not torch.cuda.is_available():
            raise SystemExit(
                "--device auto requires CUDA, which is not available here.\n"
                "CUDA is a hard requirement of this project and every deployment "
                "environment has it, so a silent CPU fallback would produce a "
                "number that reads as institutional but is not.\n"
                "Pass --device cpu explicitly for development; the run is then "
                "recorded as calibration-ineligible."
            )
        return torch.device("cuda"), True

    device = torch.device(requested)
    return device, device.type == "cuda"


def _software_revision() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — a missing revision is recorded as None, not fatal
        return None


def load_graph_and_samples(data_dir: Path, split: str) -> Tuple[Dict[str, Any], List[Any]]:
    """Read the same layout `scripts/evaluate_model.py:164-223` reads."""
    from src.kg.data_loader import DiagnosisSample

    samples_path = data_dir / f"{split}_samples.json"
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    samples = [
        DiagnosisSample(
            patient_id=s["patient_id"],
            phenotype_ids=s["phenotype_ids"],
            disease_id=s["disease_id"],
        )
        for s in json.loads(samples_path.read_text())
    ]

    graph_data: Dict[str, Any] = {}
    node_features = data_dir / "node_features.pt"
    if node_features.exists():
        graph_data["x_dict"] = torch.load(node_features, weights_only=True)
    edge_indices = data_dir / "edge_indices.pt"
    if edge_indices.exists():
        graph_data["edge_index_dict"] = torch.load(edge_indices, weights_only=True)
    num_nodes = data_dir / "num_nodes.json"
    if num_nodes.exists():
        graph_data["num_nodes_dict"] = json.loads(num_nodes.read_text())

    return graph_data, samples


def build_model(checkpoint_path: Path, device: torch.device) -> Any:
    """Rebuild the model from a checkpoint.

    `weights_only=True` — the safe loader. If a repository checkpoint proves
    incompatible with it, that is a checkpoint-format problem to fix explicitly,
    not a reason to fall back to executing pickled code.
    """
    from src.models.gnn.shepherd_gnn import ShepherdGNN, ShepherdGNNConfig

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config_dict = checkpoint.get("config", {})
    model = ShepherdGNN(
        metadata=checkpoint["metadata"],
        in_channels_dict=checkpoint["in_channels_dict"],
        config=ShepherdGNNConfig(
            hidden_dim=config_dict.get("hidden_dim", 256),
            num_layers=config_dict.get("num_layers", 4),
            num_heads=config_dict.get("num_heads", 8),
        ),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def build_manifest(args: argparse.Namespace, graph_data: Dict[str, Any],
                   n_samples: int, device: torch.device,
                   calibration_eligible: Optional[bool] = None) -> Any:
    from src.evaluation.measurement import LEGACY_TRUNCATION_K, MeasurementManifest
    from src.kg.data_loader import DataLoaderConfig
    from src.utils.fingerprint import compute_fingerprint

    loader_defaults = DataLoaderConfig()
    return MeasurementManifest(
        mode="A",
        split=args.split,
        n_samples=n_samples,
        candidate_construction="per-batch 2-hop subgraph seeded from answers and negatives",
        negative_sampling_strategy=loader_defaults.negative_sampling_strategy,
        num_negative_samples=loader_defaults.num_negative_samples,
        subgraph_strategy=loader_defaults.sampling_strategy,
        subgraph_hops=2,
        num_neighbors=list(loader_defaults.num_neighbors),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        score_semantics="raw cosine, no eta mixture and no shortest-path term",
        legacy_truncation_k=LEGACY_TRUNCATION_K,
        legacy_tie_policy="Tensor.sort on subgraph-local columns (frozen evaluator behaviour)",
        canonical_tie_policy_version="score-desc-then-global-id-asc/v1",
        checkpoint_path=str(args.checkpoint),
        data_dir=str(args.data_dir),
        graph_fingerprint=compute_fingerprint(graph_data),
        artifact_digests=_artifact_digests(args.checkpoint, args.data_dir, args.split),
        calibration_eligible=(
            device.type == "cuda" if calibration_eligible is None else calibration_eligible
        ),
        software_revision=_software_revision(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        device=str(device),
        dtype=str(torch.get_default_dtype()),
        amp_enabled=False,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic if torch.cuda.is_available() else None,
        cudnn_benchmark=torch.backends.cudnn.benchmark if torch.cuda.is_available() else None,
        python_seed=args.seed,
        numpy_seed=args.seed,
        torch_seed=args.seed,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the disease scorer (Mode A)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where the measurement JSON is written")
    parser.add_argument("--predictions-output", type=Path, default=None,
                        help="Where the per-sample calibration rows are written. "
                             "Defaults to <output stem>_predictions.json. This is "
                             "the artifact the frozen oracle's predictions file is "
                             "compared against; the measurement JSON carries only "
                             "the summary")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Part of Mode A's semantics, not a performance knob: "
                             "the candidate universe is the batch's subgraph")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Also semantics under calibration, not throughput. "
                             "Negatives are drawn in the worker processes, which "
                             "PyTorch seeds from the parent's torch RNG as "
                             "base_seed + worker_id, so a different worker count "
                             "consumes a different random stream and produces a "
                             "different candidate universe. The default matches "
                             "EvalConfig.num_workers=4, which the frozen evaluator "
                             "hardcodes and exposes no flag for")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="auto requires CUDA and fails without it. Explicit cpu "
                             "is permitted for development and marks the run "
                             "calibration-ineligible")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seeds Python, NumPy and torch. Recorded in the manifest")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device, calibration_eligible = _resolve_device(args.device)
    logger.info("Device: %s (%s)", device, platform.platform())
    if not calibration_eligible:
        logger.warning(
            "Calibration-ineligible run on %s. The result is a development number "
            "and may not be used to clear the Mode A parity gate.", device
        )

    from src.evaluation.measurement import run_mode_a
    from src.kg.data_loader import DataLoaderConfig, create_diagnosis_dataloader

    graph_data, samples = load_graph_and_samples(args.data_dir, args.split)
    model = build_model(args.checkpoint, device)
    dataloader = create_diagnosis_dataloader(
        samples=samples,
        graph_data=graph_data,
        config=DataLoaderConfig(
            batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
        ),
    )

    result = run_mode_a(
        model=model,
        dataloader=dataloader,
        manifest=build_manifest(args, graph_data, len(samples), device, calibration_eligible),
        device=device,
    )

    predictions_path = args.predictions_output or args.output.with_name(
        f"{args.output.stem}_predictions.json"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2, allow_nan=False))
    predictions_path.write_text(json.dumps(result.to_predictions(), indent=2, allow_nan=False))
    logger.info("Wrote %s and %s", args.output, predictions_path)

    print(f"\nMode A — {result.n_ranked} ranked, {result.n_ground_truth_absent} absent")
    for name, value in result.legacy_metrics.items():
        print(f"  {name:38s} {value:.6f}")
    for name, value in sorted(result.authoritative_metrics.items()):
        print(f"  {name:38s} {value:.6f}")

    candidates = result.sampler_evidence["candidate_columns"]
    negatives = result.sampler_evidence["negative_sampling"]
    print(
        f"\n  candidate columns per batch          "
        f"{candidates['min']}-{candidates['max']} (mean {candidates['mean']:.1f})"
    )
    print(f"  max subgraph nodes                   {result.sampler_evidence['max_subgraph_nodes']}")
    if negatives["observed"]:
        print(
            f"  negatives drawn / unique / repeated  "
            f"{negatives['total_drawn']} / {negatives['unique_global_ids']} / "
            f"{negatives['repeat_draws_within_sample']} within-sample"
        )

    if not calibration_eligible:
        print("\nCALIBRATION-INELIGIBLE — development run, not on CUDA.")
    print("\nNot calibrated. Institutional Mode A parity is a separate acceptance gate.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
