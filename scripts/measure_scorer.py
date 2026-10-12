#!/usr/bin/env python
r"""
Measure the disease scorer — Mode C.
====================================
Every patient in a cohort scored against every disease in the knowledge graph,
with the model the deployed pipeline builds and raw cosine. This is the formal
Test's measurement (`docs/working/PLAN_TEST_RESULTS.md`, "This is Mode C").

**A thin CLI.** Argument parsing, one read of each input, manifest assembly, and
one call into `src.evaluation.measurement.run_mode_c`. No measurement logic
lives here, so the harness stays testable without a subprocess.

    # --split val is the checkpoint-selection split, not held-out data
    python scripts/measure_scorer.py \
        --checkpoint checkpoints/best.pt \
        --data-dir data/processed \
        --split val \
        --output measurement.json

Two files come out: `--output`, the measurement report a human reads, and
`<output stem>_ranks.json` beside it, the per-sample ranks, which the report
summarises but does not carry.

Module: scripts/measure_scorer.py
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.caveats import COHORT_KIND_HELP, SPLIT_ARGUMENT_HELP
from src.evaluation.cohort import (
    COHORT_KINDS,
    DEFAULT_COHORT_KIND,
    GENERATED_SPLITS,
    CohortProvenance,
    resolve_cohort,
)
from src.utils.fingerprint import file_sha256 as _file_sha256

if TYPE_CHECKING:
    from src.kg.artifacts import ManifestRead
    from src.kg.storage.file_storage import GraphArtifactsRead, SamplesRead
    from src.utils.checkpoint_io import CheckpointRead

logger = logging.getLogger(__name__)

#: What Mode C ranks each patient against, as its manifest records it. There is
#: no expansion to describe: the candidates are every disease in the graph.
CANDIDATE_CONSTRUCTION = "every disease in the knowledge graph"

#: How the scored model was built: through `build_shepherd_model`, the builder
#: the deployed pipeline uses.
MODEL_CONSTRUCTION = "production (build_shepherd_model)"


#: Re-exported so callers that import the name from here keep working. The
#: definition moved to `src/utils/fingerprint.py`, the bottom layer, because
#: consumers importing a hashing primitive **from a script** made an entry point
#: into a library.
#:
#: **Moving the definition did not by itself remove the edge**, and an earlier
#: revision claimed it had. `scripts/benchmark_sp_lookup.py` now imports from
#: `src.utils.fingerprint` directly, which is the edge that had no business
#: existing — the SP benchmark has nothing to do with scorer measurement. No
#: script imports it from here any more; the remaining callers are tests, and
#: `tests/unit/test_training_provenance.py` pins the re-export.
file_sha256 = _file_sha256


class MeasurementInputs(NamedTuple):
    """What a run read once each: the workspace files it checked, and the
    checkpoint, which M2.1 does not check (contract M2.1, S7).

    Carried whole from `read_measurement_inputs` to `measurement_input_digests`,
    so the tensors and samples Mode C scores, the checkpoint its model is built
    from and the digests its manifest records come from the same reads and
    cannot be paired with another read of a file that changed in between.

    The other generated split, read for its binding and the disjointness check,
    is not kept: it is checked, not consumed, and no role records it.
    """

    cohort: CohortProvenance
    manifest: ManifestRead
    graph: GraphArtifactsRead
    samples: SamplesRead
    checkpoint: CheckpointRead
    #: The digest the run's one manifest reading binds for each of the four
    #: graph roles, `kg` included, as `verify_graph_reads` returned it after
    #: comparing the three tensors read with it. Kept, not discarded: it is the
    #: `kg` role (M2.1 decision 2) and the opening for M2.4's comparison.
    bound: Dict[str, str]


def read_measurement_inputs(
    checkpoint: Path, data_dir: Path, split: str, cohort_kind: str, device: torch.device,
) -> MeasurementInputs:
    """Read the manifest, the graph tensors, the cohort and the checkpoint once
    each, and verify from those reads (contract M2.1, S7).

    **Before any model is built or any output written.** `main` calls this
    right after the device is resolved.

    **The cohort is resolved first, and that only stats.** A split that is not
    an identifier, a reserved name claimed as supplied, a non-generator name
    claimed as generated, or a generated cohort with no manifest is refused
    before any file is opened.

    **The digests compared are those of the bytes parsed.** The manifest is read
    once, for both cohort kinds, because the graph binding applies to both; each
    tensor's digest and, for a generated cohort, each sample file's digest is
    compared with that one reading. A file replaced after its read is the one
    scored and recorded. A graph tensor, or for a generated cohort a sample
    file, replaced before its read and after the manifest's is refused by name;
    a supplied cohort's samples and the checkpoint are bound to nothing and are
    recorded as read. Nothing is hashed again by path.

    `kg.json` is never parsed by measurement, so it gets the identification
    check (M2.1 decision 1): the file on disk is compared with this run's one
    manifest reading, and its digest is not recorded as a file read; the `kg`
    role is the manifest-bound source graph (`bound["kg"]`, decision 2).

    **A generated cohort reads its sibling split too**, with the default fields,
    so `verify_cohort_reads` checks both splits' bindings, the disease sets, and
    the manifest's disjointness claim and the measured disjointness, as the path
    form did. A supplied cohort carries no allocation, so it gets none of these
    and its samples are bound by their own digest only.

    The graph tensors load on CPU (`map_location="cpu"`, the shared reader's
    default and what this run passed before); `encode_full_graph` moves them.
    The checkpoint keeps this site's options: `map_location` the run's device,
    `weights_only=True`. It is read after the workspace checks, so a refused
    workspace reads no checkpoint; M2.1 checks nothing about it.
    """
    from src.evaluation.cohort import verify_cohort_reads
    from src.kg.artifacts import check_unparsed_kg_json, read_split_manifest, verify_graph_reads
    from src.kg.storage.file_storage import read_graph_artifacts, read_samples
    from src.utils.checkpoint_io import read_checkpoint

    # Stat only.
    cohort = resolve_cohort(data_dir, split, cohort_kind)
    siblings: Tuple[str, ...] = ()
    if cohort.is_generated:
        siblings = tuple(name for name in GENERATED_SPLITS if name != split)
        for name in siblings:
            resolve_cohort(data_dir, name)

    manifest = read_split_manifest(data_dir)
    graph = read_graph_artifacts(data_dir, map_location="cpu")
    samples = read_samples(data_dir, split)
    checked = {name: read_samples(data_dir, name) for name in siblings}

    bound = verify_graph_reads(manifest, graph.reads)
    check_unparsed_kg_json(data_dir, manifest)
    if cohort.is_generated:
        verify_cohort_reads(
            manifest,
            {name: samples if name == split else checked[name] for name in GENERATED_SPLITS},
        )
    # The sibling is checked, not consumed: nothing holds it past this point, so
    # it is not resident while the checkpoint is read.
    del checked

    read = read_checkpoint(checkpoint, map_location=device, weights_only=True)
    logger.info(
        "Read %d %s samples; checkpoint %s (sha256 %s)",
        len(samples.samples), split, read.identity.path, read.identity.sha256[:12],
    )
    return MeasurementInputs(cohort, manifest, graph, samples, read, bound)


def measurement_input_digests(inputs: MeasurementInputs) -> Dict[str, str]:
    """The record of what a run consumed, by role, built only from its reads
    (contract M2.1, S7).

    **Only from the readers' results.** Every file role is the digest of the
    bytes a reader parsed: the checkpoint the model was built from, the cohort
    scored, the three graph tensors encoded, and, for a generated cohort, the
    manifest those were verified against. Nothing here hashes a path, so a file
    replaced after its read cannot change what is recorded, and a digest that
    was never taken cannot be recorded as `None`.

    **`kg` is not a file this run read.** Measurement never parses `kg.json`;
    its role is the digest the run's manifest binds for the source graph, the
    graph the tensors scored were verified as the export of (M2.1 decision 2).
    The identification check compared `kg.json` on disk with that digest at the
    start of the run; it records nothing itself.

    `split_manifest` is recorded for a generated cohort only. A supplied cohort's
    run reads the manifest too, for the graph binding, but no allocation cut it,
    so the manifest says nothing about how its cohort was made.

    Nothing globs the directory, so an unrelated file beside these does not
    change the record of a run that never read it.
    """
    from src.kg.artifacts import GRAPH_TENSOR_ROLES

    reads: Dict[str, Any] = {
        "checkpoint": inputs.checkpoint.identity,
        "samples": inputs.samples.identity,
        **{role: inputs.graph.reads.get(role) for role in GRAPH_TENSOR_ROLES},
    }
    if inputs.cohort.is_generated:
        reads["split_manifest"] = inputs.manifest.identity

    missing = sorted(role for role, read in reads.items() if read is None)
    if missing:
        raise ValueError(
            f"no read of {', '.join(missing)}, so there is no digest to record; a "
            "measurement record names only bytes a reader parsed"
        )
    digests = {role: read.sha256 for role, read in reads.items()}
    digests["kg"] = inputs.bound["kg"]
    return digests


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


def build_manifest(args: argparse.Namespace, graph_data: Dict[str, Any],
                   n_samples: int, device: torch.device, *,
                   digests: Mapping[str, str],
                   regime: Any,
                   cuda_executed: Optional[bool] = None,
                   model: Any = None) -> Any:
    """Build the manifest of one Mode C run.

    `digests` is the record of what the run read, from
    `measurement_input_digests` (contract M2.1, S7); nothing here hashes a file.

    `regime` is the `AutocastRegime` `encode_full_graph` reported for the
    embeddings Mode C scores: they are computed before this manifest is built,
    so the regime observed here could be a different one. It is required, and
    `run_mode_c` refuses a manifest whose regime is not the embeddings'
    (`assert_manifest_describes_regime`).

    `model` is optional and is used only to **observe** whether what ran was a
    `torch.compile` wrapper object. Omitting it records
    `torch_compile_wrapped=None` — "not observed", which is deliberately not the
    same claim as "not compiled". Neither value says a compiled graph executed;
    see `observe_torch_compile_wrapper`.
    """
    from src.evaluation.measurement import (
        METRIC_SCHEMA_VERSION,
        MeasurementManifest,
        observe_torch_compile_wrapper,
    )
    from src.utils.fingerprint import compute_fingerprint


    return MeasurementManifest(
        mode="C",
        split=args.split,
        metric_schema_version=METRIC_SCHEMA_VERSION,
        cohort_kind=args.cohort_kind,
        n_samples=n_samples,
        candidate_construction=CANDIDATE_CONSTRUCTION,
        batch_size=args.batch_size,
        score_semantics="raw cosine, no eta mixture and no shortest-path term",
        model_construction=MODEL_CONSTRUCTION,
        canonical_tie_policy_version="score-desc-then-global-id-asc/v1",
        checkpoint_path=str(args.checkpoint),
        data_dir=str(args.data_dir),
        graph_fingerprint=compute_fingerprint(graph_data),
        artifact_digests=dict(digests),
        cuda_executed=(
            device.type == "cuda" if cuda_executed is None else cuda_executed
        ),
        software_revision=_software_revision(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        device=str(device),
        dtype=str(torch.get_default_dtype()),
        # Observed, not asserted. `run_mode_c` opens no autocast context, but a
        # caller can, and a literal `False` then described a run that did not
        # happen.
        amp_enabled=regime.enabled,
        amp_dtype=regime.dtype,
        torch_compile_wrapped=observe_torch_compile_wrapper(model),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic if torch.cuda.is_available() else None,
        cudnn_benchmark=torch.backends.cudnn.benchmark if torch.cuda.is_available() else None,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the disease scorer: every patient against every disease (Mode C)"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", required=True, help=SPLIT_ARGUMENT_HELP)
    parser.add_argument("--cohort-kind", default=DEFAULT_COHORT_KIND,
                        choices=COHORT_KINDS, help=COHORT_KIND_HELP)
    parser.add_argument("--output", type=Path, required=True,
                        help="Where the measurement JSON is written. The per-sample "
                             "ranks go beside it, as <output stem>_ranks.json")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Patients scored per chunk. Recorded on the manifest: "
                             "it can move the numbers at floating-point level")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="auto requires CUDA and fails without it. Explicit cpu "
                             "is permitted for development and records "
                             "cuda_executed=false in the manifest")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    device, cuda_executed = _resolve_device(args.device)
    logger.info("Device: %s (%s)", device, platform.platform())
    if not cuda_executed:
        logger.warning(
            "Not running on CUDA (%s). The manifest will record cuda_executed=false; "
            "this is a development number, not one from the deployment hardware.", device
        )

    from src.evaluation.measurement import encode_full_graph, run_mode_c
    from src.models.gnn.shepherd_gnn import build_shepherd_model

    # Each input read once, and verified and recorded from those reads, before
    # any model is built or any output written (contract M2.1, S7).
    inputs = read_measurement_inputs(
        args.checkpoint, args.data_dir, args.split, args.cohort_kind, device
    )
    graph_data, samples = inputs.graph.graph_data, inputs.samples.samples
    digests = measurement_input_digests(inputs)

    model = build_shepherd_model(inputs.checkpoint.checkpoint, graph_data, device)
    # Kept **whole**: `run_mode_c` checks that its manifest describes the regime
    # these embeddings were computed under, which a bare dict could not carry.
    encoded = encode_full_graph(model, graph_data, device)

    result = run_mode_c(
        full_graph_embeddings=encoded,
        samples=samples,
        manifest=build_manifest(
            args, graph_data, len(samples), device, digests=digests,
            cuda_executed=cuda_executed, model=model, regime=encoded.regime,
        ),
        device=device,
        batch_size=args.batch_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2, allow_nan=False))
    ranks_path = args.output.with_name(f"{args.output.stem}_ranks.json")
    ranks_path.write_text(json.dumps(result.to_ranks(), indent=2, allow_nan=False))
    logger.info("Mode C -> %s, %s", args.output, ranks_path)

    print(f"\nMode C — {result.n_ranked} ranked, {result.n_ground_truth_absent} absent")
    for name, value in result.authoritative_metrics.items():
        print(f"  {name:38s} {value:.6f}")
    print(f"  {'candidate diseases per patient':38s} "
          f"{result.sampler_evidence['candidate_columns']['max']}")

    if not cuda_executed:
        print("\nNOT ON CUDA — development run. Recorded as cuda_executed=false.")
    print("\nWhether this is the institution's number depends on the checkpoint and\n"
          "cohort it consumed, which a person checks against the recorded digests.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
