"""
Benchmark: shortest-path lookup cost — work item B-0.4, baseline stage.
======================================================================
Measures `sp_mean_distances` (`src/inference/scoring.py`) across the matrix
`docs/working/scorer-measurement/PLAN_B04.md` §5.4 defines, on **both caller
shapes**: the singleton loop production ships today, and the batched call B-1
and the offline harness will use.

**This is the baseline stage. No index prototype exists yet**, and per PLAN_B04
§3.1 none is built until the gate warrants it. Read the plan before extending
this script; several of its shapes are decisions rather than conveniences.

What this run may and may not conclude (PLAN_B04 §3.1):

  - A synthetic run validates the benchmark, compares implementations and
    exposes gross regressions.
  - It **may not** establish that the deployed baseline is fast enough, and may
    not trigger the "ship no replacement" outcome.
  - With no real artifact the honest verdict is *"benchmark complete; production
    replacement decision pending institutional run"*.

The provisional threshold below is **declared before results are examined**,
because declaring it afterwards is choosing the verdict.

Usage:
    python scripts/benchmark_sp_lookup.py --output reports/sp_lookup_baseline.json
    python scripts/benchmark_sp_lookup.py --artifact data/processed/shortest_paths.pt

Module: scripts/benchmark_sp_lookup.py
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# =============================================================================
# Declared before any measurement — see PLAN_B04 §3.1
# =============================================================================
#: Provisional, **non-institutional** decision threshold for the eager SP pass
#: at the institution's stated candidate list size. `selection_limit` and the
#: interaction latency target are both [OPEN] in DISEASE_SCORER_POLICY §1.3;
#: this is an engineering placeholder so the curves can be read, and it carries
#: no institutional authority. Final acceptance uses the institutional target.
PROVISIONAL_BUDGET_MS = 250.0
PROVISIONAL_BUDGET_AT = {"candidates": 200, "phenotypes": 20}

# =============================================================================
# The matrix — PLAN_B04 §5.4
# =============================================================================
CANDIDATE_COUNTS = (10, 50, 100, 200, 500)
#: 500 is a **stated provisional ceiling**, not a configured upper bound.
#: `selection_limit` has no institutional value yet.
PROVISIONAL_CEILING = 500

PHENOTYPE_COUNTS = (1, 20, 100)  # the API's contractual range, diagnose.py:56

DISTRIBUTIONS = ("representative", "dense_tail")

#: How the queried phenotypes are drawn from the table.
#:
#: A heavy-tailed *table* does not by itself exercise the tail: drawing `P`
#: phenotypes at random from a lognormal samples near its median, which sits
#: **below** its mean, so a first version of this benchmark reported `dense_tail`
#: as *cheaper* than `representative` — the opposite of the axis's purpose.
#:
#: `longest` is not merely a worst case. A patient's phenotypes are not a random
#: draw with respect to slice length: common, well-studied phenotypes have more
#: KG connections and therefore longer slices, and common phenotypes are exactly
#: the ones likely to appear on a patient's list. Which of the two selections is
#: realistic is a property of the deployed artifact and the cohort, and is
#: **[OPEN]** until measured there; both are reported rather than one chosen.
PHENOTYPE_SELECTIONS = ("random", "longest")

#: Mean slice length. The metadata sidecar records `num_pairs` and
#: `num_phenotypes`, from which only the **mean** is derivable, and no artifact
#: is present in development — so this is swept rather than assumed at one
#: value. When a real artifact is available its measured distribution replaces
#: these entirely.
SYNTHETIC_MEAN_SLICE_LENGTHS = (1_000, 10_000)

#: Declared synthetic target-space size: roughly 28k diseases plus 20k genes.
SYNTHETIC_TARGET_SPACE = 48_000

#: Cells whose predicted element-touches exceed this are skipped and **reported
#: as skipped**. A silent cap reads as "covered everything" when it did not.
WORK_CEILING = 2_000_000_000

MIN_REPEATS = 3
MAX_REPEATS = 11
TARGET_MEASURE_SECONDS = 0.75


# =============================================================================
# Synthetic table construction
# =============================================================================
def _slice_lengths(
    n_phenotypes: int, mean_length: int, distribution: str, generator: torch.Generator
) -> List[int]:
    """Per-phenotype slice lengths with the requested shape and the given mean.

    `representative` puts every phenotype near the mean. `dense_tail` keeps the
    same mean while giving a minority of phenotypes much longer slices, which is
    the shape a uniform synthetic table misrepresents and the one the cost is
    most sensitive to — `sp_mean_distances` scans a whole slice per candidate.
    """
    if distribution == "representative":
        jitter = torch.randint(
            -mean_length // 10, mean_length // 10 + 1, (n_phenotypes,), generator=generator
        )
        lengths = torch.full((n_phenotypes,), mean_length) + jitter
    elif distribution == "dense_tail":
        # Lognormal shape, rescaled to the requested mean. sigma is chosen so the
        # top decile carries several times the median, not so extreme that the
        # mean is one phenotype.
        raw = torch.exp(torch.randn(n_phenotypes, generator=generator) * 1.1)
        lengths = (raw / raw.mean() * mean_length).round().to(torch.int64)
    else:
        raise ValueError(f"unknown distribution: {distribution}")
    return [max(1, int(v)) for v in lengths]


def build_synthetic_lookup(
    n_phenotypes: int,
    mean_length: int,
    distribution: str,
    target_space: int,
    max_hops: int,
    seed: int,
) -> Any:
    """A `SPLookup` with the CSR shape `_load_shortest_paths` produces.

    Uniqueness of `(phenotype, target, target_type)` holds by construction,
    because targets are sampled without replacement within each phenotype — the
    same property the real table has (`scoring.py:91-94`) and the property a
    binary search would later depend on.
    """
    from src.inference.scoring import SPLookup

    generator = torch.Generator().manual_seed(seed)
    lengths = _slice_lengths(n_phenotypes, mean_length, distribution, generator)

    targets: List[torch.Tensor] = []
    types: List[torch.Tensor] = []
    distances: List[torch.Tensor] = []
    offsets: Dict[int, Any] = {}

    cursor = 0
    for phenotype, length in enumerate(lengths):
        length = min(length, target_space)
        picked = torch.randperm(target_space, generator=generator)[:length]
        targets.append(picked)
        types.append(torch.randint(0, 2, (length,), generator=generator, dtype=torch.int8))
        distances.append(
            torch.randint(1, max_hops + 1, (length,), generator=generator, dtype=torch.int8)
        )
        offsets[phenotype] = (cursor, cursor + length)
        cursor += length

    lookup = SPLookup(
        target=torch.cat(targets),
        target_type=torch.cat(types),
        distance=torch.cat(distances),
        offsets=offsets,
        max_hops=max_hops,
    )
    return lookup, [min(v, target_space) for v in lengths]


def measure_artifact_distribution(artifact: Path) -> Dict[str, Any]:
    """Slice lengths read from a real table, without new recording anywhere.

    The lengths **are** the run lengths of the phenotype column, which is what
    `_load_shortest_paths` already computes to build its offsets
    (`pipeline.py:509-519`). Adding distribution statistics to
    `compute_shortest_paths.py` would only help artifacts built afterwards, not
    one already deployed.
    """
    data = torch.load(artifact, weights_only=True)
    phenotype = data["phenotype"] if "phenotype" in data else data["phenotype_idx"]
    phenotype, _ = torch.sort(phenotype)
    boundaries = torch.where(phenotype[1:] != phenotype[:-1])[0] + 1
    starts = torch.cat([torch.zeros(1, dtype=torch.int64), boundaries])
    ends = torch.cat([boundaries, torch.tensor([len(phenotype)], dtype=torch.int64)])
    lengths = (ends - starts).to(torch.float64)
    quantiles = torch.tensor([0.5, 0.9, 0.99, 1.0], dtype=torch.float64)
    return {
        "source": "artifact",
        "path": str(artifact),
        "n_pairs": int(phenotype.numel()),
        "n_phenotypes": int(lengths.numel()),
        "mean_slice_length": float(lengths.mean()),
        "slice_length_quantiles": {
            f"p{int(q * 100)}": float(torch.quantile(lengths, q)) for q in quantiles
        },
    }


# =============================================================================
# Timing
# =============================================================================
def _time_once(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def _call_singleton(lookup, phenotypes: Sequence[int], candidates: Sequence[int]) -> None:
    """The shape production ships: one call per candidate, `[0]` taken."""
    from src.inference.scoring import sp_mean_distances

    for candidate in candidates:
        sp_mean_distances(lookup, phenotypes, [candidate], 1)


def _call_batched(lookup, phenotypes: Sequence[int], candidates: Sequence[int]) -> None:
    """The shape B-1 and the offline harness use: one call, every candidate."""
    from src.inference.scoring import sp_mean_distances

    sp_mean_distances(lookup, phenotypes, candidates, 1)


def _repeat(fn, *args) -> Dict[str, float]:
    """Adaptive repeats: enough to measure, capped so the matrix terminates."""
    fn(*args)  # warmup
    samples = [_time_once(fn, *args)]
    while (
        len(samples) < MAX_REPEATS
        and (len(samples) < MIN_REPEATS or sum(samples) < TARGET_MEASURE_SECONDS)
    ):
        samples.append(_time_once(fn, *args))
    return {
        "median_ms": statistics.median(samples) * 1000.0,
        "max_ms": max(samples) * 1000.0,
        "repeats": len(samples),
    }


# =============================================================================
# Driver
# =============================================================================
def run_matrix(
    mean_lengths: Sequence[int], target_space: int, max_hops: int, seed: int
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for mean_length in mean_lengths:
        for distribution in DISTRIBUTIONS:
            # One table per (mean, distribution); every cell below queries it.
            n_phenotypes = max(PHENOTYPE_COUNTS) * 2
            lookup, lengths = build_synthetic_lookup(
                n_phenotypes, mean_length, distribution, target_space, max_hops, seed
            )
            by_length = sorted(range(n_phenotypes), key=lambda i: -lengths[i])
            generator = torch.Generator().manual_seed(seed)

            for selection in PHENOTYPE_SELECTIONS:
                for n_phenotype in PHENOTYPE_COUNTS:
                    phenotypes = (
                        list(range(n_phenotype))
                        if selection == "random"
                        else by_length[:n_phenotype]
                    )
                    # The work this cell actually does, not the nominal mean.
                    touched = sum(lengths[p] for p in phenotypes)
                    for n_candidate in CANDIDATE_COUNTS:
                        work = n_candidate * touched
                        if work > WORK_CEILING:
                            skipped.append({
                                "candidates": n_candidate,
                                "phenotypes": n_phenotype,
                                "mean_slice_length": mean_length,
                                "distribution": distribution,
                                "phenotype_selection": selection,
                                "predicted_element_touches": work,
                                "reason": "exceeds WORK_CEILING",
                            })
                            continue
                        candidates = torch.randint(
                            0, target_space, (n_candidate,), generator=generator
                        ).tolist()
                        for shape, fn in (
                            ("singleton", _call_singleton),
                            ("batched", _call_batched),
                        ):
                            timing = _repeat(fn, lookup, phenotypes, candidates)
                            rows.append({
                                "implementation": "current",
                                "caller_shape": shape,
                                "candidates": n_candidate,
                                "phenotypes": n_phenotype,
                                "mean_slice_length": mean_length,
                                "distribution": distribution,
                                "phenotype_selection": selection,
                                "queried_slice_total": int(touched),
                                **timing,
                            })
                            print(json.dumps(rows[-1]), flush=True)

    return {"rows": rows, "skipped": skipped}


def provenance(args: argparse.Namespace) -> Dict[str, Any]:
    """PLAN_B04 §5.5. Recorded even for a development run, so that a curve can
    never be mistaken later for one that may choose `selection_limit`."""
    return {
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "seed": args.seed,
        "warmup_runs": 1,
        "min_repeats": MIN_REPEATS,
        "max_repeats": MAX_REPEATS,
        "target_measure_seconds": TARGET_MEASURE_SECONDS,
        "deployment_equivalent_cpu": False,
        "provisional_budget_ms": PROVISIONAL_BUDGET_MS,
        "provisional_budget_at": PROVISIONAL_BUDGET_AT,
        "provisional_ceiling_candidates": PROVISIONAL_CEILING,
        "budget_is_institutional": False,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="B-0.4 SP lookup baseline benchmark")
    parser.add_argument("--artifact", type=Path, default=None,
                        help="Real shortest_paths.pt. Its measured slice distribution "
                             "replaces the synthetic one.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)

    if args.artifact is not None:
        distribution_meta = measure_artifact_distribution(args.artifact)
        mean_lengths = (int(round(distribution_meta["mean_slice_length"])),)
    else:
        distribution_meta = {
            "source": "synthetic",
            "reason": "no shortest_paths.pt supplied; none exists in development",
            "declared_mean_slice_lengths": list(SYNTHETIC_MEAN_SLICE_LENGTHS),
            "declared_target_space": SYNTHETIC_TARGET_SPACE,
        }
        mean_lengths = SYNTHETIC_MEAN_SLICE_LENGTHS

    result = run_matrix(mean_lengths, SYNTHETIC_TARGET_SPACE, args.max_hops, args.seed)

    report = {
        "stage": "B-0.4 baseline",
        "implementations": ["current"],
        "slice_distribution": distribution_meta,
        "provenance": provenance(args),
        # PLAN_B04 §3.1: a synthetic run cannot accept the deployed baseline.
        "verdict": (
            "benchmark complete; production replacement decision pending "
            "institutional run"
            if distribution_meta["source"] == "synthetic"
            else "benchmark complete on a measured artifact distribution"
        ),
        **result,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.output}", file=sys.stderr)
    print(f"\nverdict: {report['verdict']}", file=sys.stderr)
    print(f"rows: {len(result['rows'])}, skipped: {len(result['skipped'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
