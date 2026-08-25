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
        --split val \        # see the note below: val is not held-out
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


def file_sha256(path: Path) -> Optional[str]:
    """Raw content digest, or ``None`` if the file is not there.

    **Public and shared.** `scripts/calibrate_mode_a.py` hashes the same artifacts
    before and after the two runs, and a second implementation there could differ
    from this one in exactly the way the digests exist to detect.

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


def artifact_digests(checkpoint: Path, data_dir: Path, split: str) -> Dict[str, Optional[str]]:
    """Every file a Mode A number depends on, by role.

    Paths are recorded too, but a path is not an identity — `checkpoints/best.pt`
    names a different file after every improvement — and the structural
    fingerprint is not one either, since two checkpoints trained on the same graph
    share it.
    """
    return {
        "checkpoint": file_sha256(checkpoint),
        "samples": file_sha256(data_dir / f"{split}_samples.json"),
        "node_features": file_sha256(data_dir / "node_features.pt"),
        "edge_indices": file_sha256(data_dir / "edge_indices.pt"),
        "num_nodes": file_sha256(data_dir / "num_nodes.json"),
    }


def _resolve_device(requested: str) -> Tuple[torch.device, bool]:
    """Return the device and whether it is CUDA.

    **CUDA is a hard project requirement**, so `auto` falling back to CPU would
    silently produce a number that looks institutional and is not: different
    kernels, different reduction orders, and no statement at all about the
    hardware the tool actually runs on. `auto` therefore fails without CUDA rather
    than substituting a device nobody asked for.

    Explicit `--device cpu` stays available, because development in a container
    without a GPU is real work — but the manifest records `cuda_executed: false`,
    so a development number cannot later be quoted as one produced on the
    deployment hardware.

    The boolean means only "this ran on CUDA". Whether the run is acceptable to
    the institution depends on which artifacts it consumed, which is checked by a
    person against the recorded digests, not here.
    """
    if requested == "auto":
        if not torch.cuda.is_available():
            raise SystemExit(
                "--device auto requires CUDA, which is not available here.\n"
                "CUDA is a hard requirement of this project and every deployment "
                "environment has it, so a silent CPU fallback would produce a "
                "number that reads as institutional but is not.\n"
                "Pass --device cpu explicitly for development; the run is then "
                "recorded as cuda_executed=false."
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


def load_legacy_mode_a_inputs(data_dir: Path, split: str) -> Tuple[Dict[str, Any], List[Any]]:
    """Read the same layout `scripts/evaluate_model.py:164-223` reads.

    **Named for its lifecycle, not its shape.** This is a deliberate duplicate of
    the frozen evaluator's loader, kept only so Mode A consumes its inputs exactly
    as the oracle does. **Modes B, C and D must not import it.** It retires with
    `scripts/evaluate_model.py` once institutional parity succeeds.

    **The reading is no longer duplicated here.** It delegates to
    `src/kg/storage/file_storage.py`, which does not retire with the frozen
    evaluator — Mode C needs the same files and may not reach them through this
    function. What stays legacy about this one is its *name and lifecycle*: it is
    the entry point Mode A uses, and it goes when the oracle goes.

    The same read still appears in `src/inference/pipeline.py:579-606`,
    `scripts/train_model.py`, `scripts/evaluate_model.py`,
    `scripts/build_index.py` and `scripts/setup_demo.py`. Every copy depends on
    the same filenames and serialisation format, so a format change breaks all of
    them at once and the duplication buys nothing.

    Migrating those five is P1 and stays out of this change; migrating them here
    would put the measurement behind an unrelated sweep.
    """
    from src.kg.storage.file_storage import read_graph_artifacts, read_samples

    return read_graph_artifacts(data_dir), read_samples(data_dir, split)


def build_legacy_mode_a_model(checkpoint_path: Path, device: torch.device) -> Any:
    """Rebuild the model the way the frozen evaluator does.

    **Named for its lifecycle.** This mirrors
    `scripts/evaluate_model.py:create_model_from_checkpoint` on purpose, including
    its hardcoded architecture fallbacks. Replacing it with the production
    architecture resolver before parity is established would change the control
    being measured, which is the one thing Mode A may not do.

    **Modes B, C and D must not import it** — they resolve architecture the way the
    deployed pipeline does. It retires with `scripts/evaluate_model.py`.

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


def build_loader_config(args: argparse.Namespace) -> Any:
    """The one `DataLoaderConfig` the run uses.

    Built here and handed to **both** the dataloader and `build_manifest`, so the
    manifest cannot describe a configuration different from the one that produced
    the batches. Previously the manifest read a freshly constructed
    `DataLoaderConfig()` while the loader got another instance: the values agreed
    only because both were defaults, and a changed default would have silently
    made the manifest describe a run that never happened.
    """
    from src.kg.data_loader import DataLoaderConfig

    return DataLoaderConfig(
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )


def build_manifest(args: argparse.Namespace, graph_data: Dict[str, Any],
                   n_samples: int, device: torch.device, loader_config: Any,
                   cuda_executed: Optional[bool] = None,
                   mode: str = "A",
                   candidate_construction: str =
                   "per-batch 2-hop subgraph seeded from answers and negatives",
                   model_construction: str = "frozen evaluator (legacy)",
                   model: Any = None) -> Any:
    """Build the manifest for one mode.

    `model` is optional and is used only to **observe** whether what ran was a
    `torch.compile` wrapper object. Omitting it records
    `torch_compile_wrapped=None` — "not observed", which is deliberately not the
    same claim as "not compiled". Neither value says a compiled graph executed;
    see `observe_torch_compile_wrapper`.
    """
    from src.evaluation.measurement import (
        LEGACY_TRUNCATION_K,
        MeasurementManifest,
        observe_torch_compile_wrapper,
    )
    from src.kg.data_loader import DIAGNOSIS_SUBGRAPH_HOPS
    from src.utils.fingerprint import compute_fingerprint

    return MeasurementManifest(
        mode=mode,
        split=args.split,
        n_samples=n_samples,
        candidate_construction=candidate_construction,
        negative_sampling_strategy=loader_config.negative_sampling_strategy,
        num_negative_samples=loader_config.num_negative_samples,
        subgraph_strategy=loader_config.sampling_strategy,
        subgraph_hops=DIAGNOSIS_SUBGRAPH_HOPS,
        num_neighbors=list(loader_config.num_neighbors),
        max_subgraph_nodes=loader_config.max_subgraph_nodes,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle,
        num_workers=loader_config.num_workers,
        score_semantics="raw cosine, no eta mixture and no shortest-path term",
        model_construction=model_construction,
        legacy_truncation_k=LEGACY_TRUNCATION_K,
        legacy_tie_policy="Tensor.sort on subgraph-local columns (frozen evaluator behaviour)",
        canonical_tie_policy_version="score-desc-then-global-id-asc/v1",
        checkpoint_path=str(args.checkpoint),
        data_dir=str(args.data_dir),
        graph_fingerprint=compute_fingerprint(graph_data),
        artifact_digests=artifact_digests(args.checkpoint, args.data_dir, args.split),
        cuda_executed=(
            device.type == "cuda" if cuda_executed is None else cuda_executed
        ),
        software_revision=_software_revision(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        device=str(device),
        dtype=str(torch.get_default_dtype()),
        # Structural, and enforced rather than trusted: no traversal in
        # `src/evaluation/measurement.py` opens an autocast context, and
        # `assert_no_autocast` refuses to run inside one opened by a caller. So
        # these two are facts about the run, not defaults that happen to be right.
        amp_enabled=False,
        amp_dtype=None,
        torch_compile_wrapped=observe_torch_compile_wrapper(model),
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
    parser.add_argument("--split", required=True,
                        choices=["train", "val", "test"],
                        help='Which samples file to measure. **Required — there is no default.** Generated workspaces normally contain train and val only; a test split exists only where an evaluation protocol created one. `val` is the checkpoint-selection split under the current trainer (early_stopping_monitor=val_mrr), so metrics measured on it are model-selection-contaminated and are not held-out generalisation.')
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
                             "is permitted for development and records "
                             "cuda_executed=false in the manifest")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seeds Python, NumPy and torch. Recorded in the manifest")
    parser.add_argument("--modes", default="A",
                        help="One of: A, A,B, C, A,B,C. Default A, which is the "
                             "calibration path and must stay the default. "
                             "**B requires A** — it is A's candidates under a "
                             "different encoder, so it is refused without A "
                             "rather than silently adding it")
    return parser.parse_args(argv)


SUPPORTED_MODE_SETS = (("A",), ("A", "B"), ("C",), ("A", "B", "C"))
"""The combinations that mean something, enumerated rather than derived.

Each is either one measurement or a **ladder** in which consecutive modes differ
by one thing. `A,C` is deliberately absent: it is two measurements whose
difference confounds encoder scope with candidate universe, and a run that emits
both invites exactly the attribution it cannot support. Ask for `A,B,C`, which
contains both and can attribute, or run them separately and say why.
"""


def parse_modes(spec: str) -> List[str]:
    """Normalise `--modes` against the supported combinations.

    Refusing rather than repairing, in both directions: an unsupported set is not
    silently completed to a supported one, because a caller who asked for `A,C`
    believes they asked for something attributable, and quietly handing them
    `A,B,C` would confirm it.
    """
    modes = [m.strip().upper() for m in spec.split(",") if m.strip()]
    unknown = sorted(set(modes) - {"A", "B", "C"})
    if unknown:
        raise SystemExit(f"unknown mode(s): {unknown}. Known modes are A, B and C")
    if not modes:
        raise SystemExit("--modes selected nothing")

    ordered = tuple(m for m in ("A", "B", "C") if m in set(modes))
    if ordered not in SUPPORTED_MODE_SETS:
        supported = ", ".join(",".join(combo) for combo in SUPPORTED_MODE_SETS)
        if ordered == ("A", "C"):
            raise SystemExit(
                "A,C is not a supported combination. A->C changes the encoder AND "
                "the candidate universe, so a difference between them attributes "
                "to neither. Use A,B,C, which contains both and can attribute, or "
                "run them as separate invocations if you want them independently."
            )
        if "B" in ordered and "A" not in ordered:
            raise SystemExit(
                "Mode B is Mode A's candidates under a full-graph encoder, so it "
                "is only meaningful beside A. Request --modes A,B"
            )
        raise SystemExit(f"unsupported mode combination {','.join(ordered)}. "
                         f"Supported: {supported}")
    return list(ordered)


def _assert_same_cohort(left: Any, right: Any) -> None:
    """Two modes are comparable only over the same patients in the same order.

    Checked rather than assumed, because the two reach their cohort by different
    routes: Mode A through the dataloader, Mode C straight from the samples file.
    A reordering in either would leave every metric well formed and every
    per-sample comparison wrong.
    """
    a_mode, b_mode = left.manifest.mode, right.manifest.mode
    if left.sample_ids != right.sample_ids:
        first = next(
            (i for i, (x, y) in enumerate(zip(left.sample_ids, right.sample_ids)) if x != y),
            min(len(left.sample_ids), len(right.sample_ids)),
        )
        raise SystemExit(
            f"modes {a_mode} and {b_mode} did not score the same cohort in the same "
            f"order: {len(left.sample_ids)} vs {len(right.sample_ids)} samples, first "
            f"difference at index {first}. Nothing comparing them would mean anything."
        )
    if left.truth_global_ids != right.truth_global_ids:
        raise SystemExit(
            f"modes {a_mode} and {b_mode} disagree on the ground truth for the same "
            "patients; one of the two id spaces is wrong"
        )


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device, cuda_executed = _resolve_device(args.device)
    logger.info("Device: %s (%s)", device, platform.platform())
    if not cuda_executed:
        logger.warning(
            "Not running on CUDA (%s). The manifest will record cuda_executed=false; "
            "this is a development number, not one from the deployment hardware.", device
        )

    from src.evaluation.measurement import (
        assert_constructions_agree,
        encode_full_graph,
        run_mode_c,
        run_modes_ab,
    )
    from src.kg.data_loader import create_diagnosis_dataloader

    modes = parse_modes(args.modes)
    wants_legacy = "A" in modes          # A, and B which rides A's traversal
    wants_production = bool({"B", "C"} & set(modes))

    # Loading and construction are dispatched by mode, not done unconditionally.
    # `load_legacy_mode_a_inputs` and `build_legacy_mode_a_model` retire with the
    # frozen evaluator; a C-only run that touched them would fail when they go,
    # and could fail today on a checkpoint the legacy loader cannot rebuild even
    # though production can.
    from src.kg.storage.file_storage import read_graph_artifacts, read_samples

    if wants_legacy:
        graph_data, samples = load_legacy_mode_a_inputs(args.data_dir, args.split)
    else:
        graph_data = read_graph_artifacts(args.data_dir)
        samples = read_samples(args.data_dir, args.split)

    legacy_model = build_legacy_mode_a_model(args.checkpoint, device) if wants_legacy else None

    # One config object, two consumers. Not two instances that happen to agree.
    loader_config = build_loader_config(args)

    def manifest_for(mode: str, candidates: str, construction: str, model: Any = None):
        return build_manifest(
            args, graph_data, len(samples), device, loader_config, cuda_executed,
            mode=mode, candidate_construction=candidates,
            model_construction=construction, model=model,
        )

    embeddings = None
    production_model = None
    if wants_production:
        from src.models.gnn.shepherd_gnn import build_shepherd_model

        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        production_model = build_shepherd_model(checkpoint, graph_data, device)
        # Only B is read against A as encoder scope, so only B needs the two
        # constructions to be the same model. C is never compared to A directly.
        if "B" in modes:
            assert_constructions_agree(legacy_model, production_model)
        embeddings = encode_full_graph(production_model, graph_data, device)

    results: Dict[str, Any] = {}
    if "A" in modes:
        results["A"], mode_b = run_modes_ab(
            model=legacy_model,
            dataloader=create_diagnosis_dataloader(
                samples=samples, graph_data=graph_data, config=loader_config
            ),
            # Which model each mode's `torch_compile_wrapped` describes, stated because
            # the modes do not all forward a model. A forwards `legacy_model` per
            # batch. B and C forward nothing — they index the embeddings
            # `encode_full_graph` produced, so the compile state that could have
            # moved their numbers is `production_model`'s, at the moment those
            # embeddings were computed.
            manifest_a=manifest_for(
                "A", "per-batch 2-hop subgraph seeded from answers and negatives",
                "frozen evaluator (legacy)", model=legacy_model,
            ),
            manifest_b=manifest_for(
                "B", "per-batch 2-hop subgraph seeded from answers and negatives",
                "production (build_shepherd_model)", model=production_model,
            ) if "B" in modes else None,
            full_graph_embeddings=embeddings,
            device=device,
        )
        if mode_b is not None:
            results["B"] = mode_b

    if "C" in modes:
        results["C"] = run_mode_c(
            full_graph_embeddings=embeddings,
            samples=samples,
            manifest=manifest_for(
                "C", "every disease in the knowledge graph",
                "production (build_shepherd_model)", model=production_model,
            ),
            device=device,
            batch_size=args.batch_size,
        )

    # The cohort claim, enforced before anything is written rather than printed
    # afterwards. A and C reach their patients by different routes — the
    # dataloader and the samples file — so their agreement is a fact to check,
    # not a consequence of the code's shape.
    if {"A", "C"} <= set(results):
        _assert_same_cohort(results["A"], results["C"])

    result = results.get("A")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # A single-mode run writes the mode the caller asked for to `--output`; a
    # multi-mode run keeps A there, because `scripts/calibrate_mode_a.py` reads
    # that path and must keep finding Mode A in it. One file per mode either way,
    # since a mode is one measurement and merging them would put two manifests in
    # one document.
    primary = modes[0] if len(modes) == 1 else "A"
    for mode, mode_result in results.items():
        path = (
            args.output if mode == primary
            else args.output.with_name(f"{args.output.stem}_mode{mode}.json")
        )
        path.write_text(json.dumps(mode_result.to_dict(), indent=2, allow_nan=False))
        ranks_path = path.with_name(f"{path.stem}_ranks.json")
        ranks_path.write_text(json.dumps(mode_result.to_ranks(), indent=2, allow_nan=False))
        logger.info("Mode %s -> %s, %s", mode, path, ranks_path)

    if result is not None:
        predictions_path = args.predictions_output or args.output.with_name(
            f"{args.output.stem}_predictions.json"
        )
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.write_text(
            json.dumps(result.to_predictions(), indent=2, allow_nan=False)
        )
        logger.info("Frozen-oracle comparison artifact -> %s", predictions_path)

    for mode, mode_result in results.items():
        print(f"\nMode {mode} — {mode_result.n_ranked} ranked, "
              f"{mode_result.n_ground_truth_absent} absent")
        legacy = getattr(mode_result, "legacy_metrics", {})
        for name, value in {**legacy, **mode_result.authoritative_metrics}.items():
            print(f"  {name:38s} {value:.6f}")
        candidates = mode_result.sampler_evidence["candidate_columns"]
        print(f"  candidate columns per batch            "
              f"{candidates['min']}-{candidates['max']} (mean {candidates['mean']:.1f})")

    rungs = []
    if {"A", "B"} <= set(results):
        rungs.append("A->B is encoder scope")
    if {"B", "C"} <= set(results):
        rungs.append("B->C is the candidate universe")
    if rungs:
        print("\n  Comparable only because these modes share this cohort in this order,")
        print(f"  which was checked before anything was written. {'; '.join(rungs)}.")

    if not cuda_executed:
        print("\nNOT ON CUDA — development run. Recorded as cuda_executed=false.")
    print("\nNot calibrated. Institutional parity is a separate acceptance gate, and\n"
          "no cross-mode conclusion may rest on a synthetic or CPU run.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
