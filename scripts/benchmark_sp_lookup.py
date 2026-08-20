"""
Benchmark: shortest-path lookup cost — work item B-0.4, baseline stage.
======================================================================
Measures `sp_mean_distances` (`src/inference/scoring.py`) across the matrix
`docs/working/scorer-measurement/PLAN_B04.md` §5.4 defines, on **both caller
shapes**: the singleton loop production ships today, and the batched call B-1
and the offline harness will use.

The gate in PLAN_B04 §3.1 has since been crossed, so `--implementations` also
times the two prototypes in `scripts/sp_index_prototypes.py`. Read the plan
before extending this script; several of its shapes are decisions rather than
conveniences.

**Run one prototype per process.** `ru_maxrss` is a process high-water mark, so
a second index built in the same process inherits the first's peak and its own
cost stops being attributable. The report carries
`memory_attribution_isolated`, which is false whenever that was violated.

Two modes, and they are **not** interchangeable (PLAN_B04 §3.1):

  - **Synthetic** — a sensitivity sweep over declared slice-length shapes. It
    validates the benchmark, compares implementations and exposes gross
    regressions. It **cannot** close the institutional gate, and it may not be
    described as a measured artifact distribution however its parameters were
    chosen.
  - **Artifact** (`--artifact`) — times the primitive against the **real slices
    of a real table**. The phenotype subsets and candidate lists are sampled
    from that table under a recorded rule and seed; the slice contents that are
    scanned are the artifact's own.

An earlier version of this script conflated the two: it read an artifact, took
only its *mean* slice length, and then timed a synthetic table parameterised by
that mean — while reporting "benchmark complete on a measured artifact
distribution". The artifact's tail, phenotype-to-slice mapping and target layout
never entered the timed workload. That is why the two modes are now separate
code paths rather than one path with a parameter.

The provisional threshold below is **declared before results are examined**,
because declaring it afterwards is choosing the verdict.

Usage:
    python scripts/benchmark_sp_lookup.py --output reports/sp_lookup.json
    python scripts/benchmark_sp_lookup.py --artifact data/processed/shortest_paths.pt

Module: scripts/benchmark_sp_lookup.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

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

SYNTHETIC_DISTRIBUTIONS = ("representative", "dense_tail")

#: How the queried phenotypes are drawn.
#:
#: A heavy-tailed *table* does not by itself exercise the tail: drawing `P`
#: phenotypes at random samples near the lognormal's median, which sits **below**
#: its mean, so a first version of this benchmark reported `dense_tail` as
#: *cheaper* than `representative` — the opposite of that axis's purpose.
#:
#: `longest` is not merely a worst case. Common, well-studied phenotypes have
#: more KG connections and therefore longer slices, and common phenotypes are
#: exactly the ones likely to appear on a patient's list. Whether that
#: correlation holds is an **artifact–cohort relationship**, not a property of
#: the artifact alone, and it is [OPEN] until measured on both.
#:
#: `sampled` is **one seeded random subset — a single sensitivity example, not
#: an estimate of typical selection.** An earlier version used
#: `range(n_phenotype)`, a fixed prefix, and called it random. Estimating typical
#: selection needs a bounded set of seeds and belongs to the artifact run.
PHENOTYPE_SELECTIONS = ("sampled", "longest")

#: Mean slice length for synthetic tables. The metadata sidecar records
#: `num_pairs` and `num_phenotypes`, from which only the **mean** is derivable,
#: so this is swept rather than assumed at one value.
SYNTHETIC_MEAN_SLICE_LENGTHS = (1_000, 10_000)

#: Declared synthetic target-space size: roughly 28k diseases plus 20k genes.
SYNTHETIC_TARGET_SPACE = 48_000

#: Cells whose predicted element-touches exceed this are skipped and **reported
#: as skipped**. A silent cap reads as "covered everything" when it did not.
WORK_CEILING = 2_000_000_000

MIN_REPEATS = 3
MAX_REPEATS = 11
TARGET_MEASURE_SECONDS = 0.75

#: production passes 1 for diseases (`pipeline.py:1049`, "1 = disease, 0 = gene")
DISEASE_TYPE_IDX = 1


# =============================================================================
# Tables — synthetic
# =============================================================================
def _slice_lengths(
    n_phenotypes: int, mean_length: int, distribution: str, generator: torch.Generator
) -> List[int]:
    """Per-phenotype slice lengths with the requested shape and the given mean."""
    if distribution == "representative":
        jitter = torch.randint(
            -mean_length // 10, mean_length // 10 + 1, (n_phenotypes,), generator=generator
        )
        lengths = torch.full((n_phenotypes,), mean_length) + jitter
    elif distribution == "dense_tail":
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
) -> Tuple[Any, List[int], List[int]]:
    """A `SPLookup` with the CSR shape `_load_shortest_paths` produces.

    Uniqueness of `(phenotype, target, target_type)` holds by construction,
    because targets are sampled without replacement within each phenotype — the
    same property the real table has (`scoring.py:91-94`).

    Returns `(lookup, slice_lengths, disease_targets)`.
    """
    from src.inference.scoring import SPLookup

    generator = torch.Generator().manual_seed(seed)
    lengths = _slice_lengths(n_phenotypes, mean_length, distribution, generator)

    targets: List[torch.Tensor] = []
    types: List[torch.Tensor] = []
    distances: List[torch.Tensor] = []
    offsets: Dict[int, Tuple[int, int]] = {}

    cursor = 0
    actual: List[int] = []
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
        actual.append(length)

    lookup = SPLookup(
        target=torch.cat(targets),
        target_type=torch.cat(types),
        distance=torch.cat(distances),
        offsets=offsets,
        max_hops=max_hops,
    )
    return lookup, actual, list(range(target_space))


def synthetic_tables(
    max_hops: int, seed: int
) -> Iterator[Tuple[Dict[str, Any], Any, List[int], List[int]]]:
    n_phenotypes = max(PHENOTYPE_COUNTS) * 4
    for mean_length in SYNTHETIC_MEAN_SLICE_LENGTHS:
        for distribution in SYNTHETIC_DISTRIBUTIONS:
            lookup, lengths, targets = build_synthetic_lookup(
                n_phenotypes, mean_length, distribution, SYNTHETIC_TARGET_SPACE,
                max_hops, seed,
            )
            yield (
                {"mean_slice_length": mean_length, "distribution": distribution},
                lookup, lengths, targets,
            )


# =============================================================================
# Tables — the real artifact
# =============================================================================
def build_artifact_lookup(
    path: Path, max_hops: int
) -> Tuple[Any, List[int], List[int], Dict[str, Any]]:
    """An `SPLookup` over the artifact's **own slices**, not a table shaped like it.

    This mirrors `DiagnosisPipeline._load_shortest_paths` (`pipeline.py:470-545`):
    same required keys, same dtype compaction, same sort-by-phenotype, same
    offsets from run boundaries. **It is a second reader of that layout and is
    meant to stop being one: reconcile it whenever the production SP loader is
    next changed, whether or not B-0.4 selects a prototype.** Tying it to a
    prototype would leave the duplicate permanent if no prototype is selected.
    Recorded rather than left to be discovered, because
    `src/kg/storage/file_storage.py` exists to end exactly this kind of
    duplication.

    Returns `(lookup, slice_lengths, disease_targets, provenance)`.
    """
    from src.inference.scoring import SPLookup

    raw = torch.load(path, map_location="cpu", weights_only=True)
    required = {"phenotype_idx", "target_idx", "target_type", "distance"}
    missing = required - set(raw.keys())
    if missing:
        raise SystemExit(f"{path} is missing required keys: {sorted(missing)}")

    phenotype = raw["phenotype_idx"].to(torch.int32)
    order = phenotype.argsort()
    phenotype = phenotype[order]
    target = raw["target_idx"].to(torch.int32)[order]
    target_type = raw["target_type"].to(torch.int8)[order]
    distance = raw["distance"][order]
    del raw

    boundaries = torch.where(phenotype[1:] != phenotype[:-1])[0] + 1
    starts = torch.cat([torch.zeros(1, dtype=torch.int64), boundaries])
    ends = torch.cat([boundaries, torch.tensor([phenotype.numel()], dtype=torch.int64)])
    keys = phenotype[starts].tolist()
    starts_list, ends_list = starts.tolist(), ends.tolist()
    offsets = {k: (starts_list[i], ends_list[i]) for i, k in enumerate(keys)}
    lengths = [ends_list[i] - starts_list[i] for i in range(len(keys))]

    lookup = SPLookup(
        target=target, target_type=target_type, distance=distance,
        offsets=offsets, max_hops=max_hops,
    )

    # Candidates are drawn from the real disease target space, not an invented one.
    disease_targets = torch.unique(target[target_type == DISEASE_TYPE_IDX]).tolist()

    from scripts.measure_scorer import file_sha256

    quantiles = (0.5, 0.9, 0.99, 1.0)
    length_t = torch.tensor(lengths, dtype=torch.float64)
    provenance = {
        "path": str(path),
        # **Not `sha256(path.read_bytes())`.** That allocates the whole
        # multi-gigabyte artifact as one `bytes` object *after* the tensors are
        # already resident, and `ru_maxrss` never decreases — so the digest would
        # set a process high-water mark that hides every later prototype build,
        # and could OOM on unified memory. `file_sha256` reads in chunks and is
        # the one `measure_scorer.py` already uses; a second copy here would be
        # another place for the two to drift.
        "sha256": file_sha256(path),
        "n_pairs": int(phenotype.numel()),
        "n_phenotypes": len(keys),
        "n_disease_targets": len(disease_targets),
        "mean_slice_length": float(length_t.mean()),
        "slice_length_quantiles": {
            f"p{int(q * 100)}": float(torch.quantile(length_t, q)) for q in quantiles
        },
    }
    return lookup, lengths, disease_targets, provenance


# =============================================================================
# Timing
# =============================================================================
def _time_once(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def _callers(query_fn):
    """Both caller shapes for one query function.

    The prototypes take an index where the current primitive takes an `SPLookup`,
    but the signature is otherwise identical — deliberately, since PLAN_B04 §4.1
    keeps the caller unchanged. So one factory serves all three implementations
    and there is no second copy of the loop to drift.
    """

    def singleton(table, phenotypes: Sequence[int], candidates: Sequence[int]) -> None:
        """The shape production ships: one call per candidate, `[0]` taken."""
        for candidate in candidates:
            query_fn(table, phenotypes, [candidate], DISEASE_TYPE_IDX)

    def batched(table, phenotypes: Sequence[int], candidates: Sequence[int]) -> None:
        """The shape B-1 and the offline harness use: one call, every candidate."""
        query_fn(table, phenotypes, candidates, DISEASE_TYPE_IDX)

    return singleton, batched


def _rss_bytes() -> Dict[str, Optional[int]]:
    """Current **and** high-water RSS.

    `ru_maxrss` alone cannot attribute a prototype build: loading the artifact
    has already set a process peak far above anything the build adds, and a
    high-water mark never comes back down. Current RSS does move, so the pair
    says both "how much is resident now" and "how high has this process ever
    been" — and their disagreement is itself the signal that the peak belongs to
    something earlier.

    `/proc/self/status` on Linux, which is where the deployment runs; `None`
    elsewhere rather than a fabricated number. No profiling dependency.
    """
    import resource

    values: Dict[str, Optional[int]] = {
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "current_rss_bytes": None,
    }
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                values["current_rss_bytes"] = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    return values


def build_index(implementation: str, lookup) -> Tuple[Any, Dict[str, Any]]:
    """Build one prototype's index, timing it and recording what it costs.

    **Two memory numbers, deliberately not merged.** `resident_bytes_actual` is
    what the prototype object holds in *this* process, beside the loader's own
    tensors; `production_incremental_bytes_projected` is the steady-state
    increment if the loader reordered in place instead of keeping both copies.
    The second is a projection from the design and is labelled as one — reporting
    it as measured residence is how a memory verdict goes wrong.
    """
    from scripts.sp_index_prototypes import (
        build_global_key_index,
        build_slice_sorted_index,
    )

    builders = {"global": build_global_key_index, "slices": build_slice_sorted_index}
    before = _rss_bytes()
    start = time.perf_counter()
    index = builders[implementation](lookup)
    elapsed = time.perf_counter() - start
    after = _rss_bytes()
    return index, {
        "record": "index_build",
        "implementation": implementation,
        "build_seconds": elapsed,
        "prototype_resident_bytes_actual": index.resident_bytes_actual,
        "production_incremental_bytes_projected":
            index.production_incremental_bytes_projected,
        "rss_before": before,
        "rss_after": after,
        "rows": int(lookup.target.numel()),
    }


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
def time_table(
    labels: Dict[str, Any],
    lookup: Any,
    lengths: Sequence[int],
    targets: Sequence[int],
    phenotype_ids: Sequence[int],
    seed: int,
    rows: List[Dict[str, Any]],
    skipped: List[Dict[str, Any]],
    cell_counter: "itertools.count",
    tables: Optional[Dict[str, Any]] = None,
) -> None:
    """Time every requested implementation over the same cells.

    `tables` maps an implementation name to the object its query function takes —
    the `SPLookup` for `current`, a prototype index otherwise. All of them see the
    **same** phenotypes and candidates in the same cell, so a difference between
    two rows is the implementation and not the workload.
    """
    from src.inference.scoring import sp_mean_distances

    if tables is None:
        tables = {"current": lookup}
    query_fns = {"current": sp_mean_distances}
    if any(name != "current" for name in tables):
        from scripts.sp_index_prototypes import (
            sp_mean_distances_global,
            sp_mean_distances_slices,
        )

        query_fns["global"] = sp_mean_distances_global
        query_fns["slices"] = sp_mean_distances_slices

    by_length = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    generator = torch.Generator().manual_seed(seed)
    target_t = torch.tensor(targets)

    for selection in PHENOTYPE_SELECTIONS:
        for n_phenotype in PHENOTYPE_COUNTS:
            if n_phenotype > len(lengths):
                continue
            if selection == "longest":
                chosen = by_length[:n_phenotype]
            else:
                # A genuine seeded subset, not a prefix. One example, not an estimate.
                picked = torch.randperm(len(lengths), generator=generator)[:n_phenotype]
                chosen = picked.tolist()
            phenotypes = [phenotype_ids[i] for i in chosen]
            touched = sum(lengths[i] for i in chosen)

            for n_candidate in CANDIDATE_COUNTS:
                work = n_candidate * touched
                if work > WORK_CEILING:
                    skipped.append({
                        **labels, "candidates": n_candidate, "phenotypes": n_phenotype,
                        "phenotype_selection": selection,
                        "predicted_element_touches": int(work),
                        "reason": "exceeds WORK_CEILING",
                    })
                    continue
                # **Without replacement.** `torch.randint` draws with it, and a
                # repeated candidate is not a workload production can present:
                # the real disease candidate list is a set. Duplicates would also
                # flatter a binary search, whose repeated probes hit the same
                # cache lines.
                if n_candidate > len(targets):
                    skipped.append({
                        **labels, "candidates": n_candidate,
                        "phenotypes": n_phenotype,
                        "phenotype_selection": selection,
                        "available_targets": len(targets),
                        "reason": "requested more unique candidates than the "
                                  "target space holds",
                    })
                    continue
                picked = torch.randperm(len(targets), generator=generator)[:n_candidate]
                candidates = target_t[picked].tolist()

                # Rotate which implementation is measured first, independently of
                # the caller-shape alternation below. Iterating `tables` in
                # insertion order would put `current` first in every cell of every
                # documented command, so any warm-up or cache advantage would
                # accrue to the same implementation throughout.
                cell_index = next(cell_counter)
                ordered = list(tables.items())
                rotation = cell_index % len(ordered)
                ordered = ordered[rotation:] + ordered[:rotation]

                for position, (implementation, table) in enumerate(ordered):
                    singleton, batched = _callers(query_fns[implementation])
                    shapes = [("singleton", singleton), ("batched", batched)]
                    # Alternate which shape runs first, so a small difference
                    # between them is not confounded with a fixed measurement
                    # order.
                    #
                    # **Counted per timed cell, not from `len(rows)`.** Each cell
                    # appends two rows, so `len(rows)` is even at every cell
                    # boundary and a `len(rows) % 2` test never fires — the first
                    # version of this alternated nothing, and the evidence file
                    # recorded `measured_first="singleton"` for all 240 rows while
                    # §8.1 claimed otherwise.
                    #
                    # Derived from (cell, position) rather than a second draw, so
                    # shape order and implementation order vary independently
                    # instead of moving together.
                    if (cell_index + position) % 2:
                        shapes.reverse()
                    for shape, fn in shapes:
                        timing = _repeat(fn, table, phenotypes, candidates)
                        rows.append({
                            "implementation": implementation, **labels,
                            "caller_shape": shape, "candidates": n_candidate,
                            "phenotypes": n_phenotype,
                            "phenotype_selection": selection,
                            "queried_slice_total": int(touched),
                            "measured_first": shapes[0][0],
                            "implementation_position": position,
                            **timing,
                        })
                        print(json.dumps(rows[-1]), flush=True)


def provenance(args: argparse.Namespace, mode: str) -> Dict[str, Any]:
    """PLAN_B04 §5.5. Recorded even for a development run, so that a curve can
    never be mistaken later for one that may choose `selection_limit`."""
    return {
        "mode": mode,
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "seed": args.seed,
        "sampling_rule": (
            "phenotypes: 'longest' takes the n longest slices; 'sampled' takes one "
            "seeded randperm subset. candidates: seeded uniform draw with replacement "
            "from the disease target space."
        ),
        "warmup_runs": 1,
        "min_repeats": MIN_REPEATS,
        "max_repeats": MAX_REPEATS,
        "target_measure_seconds": TARGET_MEASURE_SECONDS,
        "deployment_equivalent_cpu": False,
        "provisional_budget_ms": PROVISIONAL_BUDGET_MS,
        "provisional_budget_at": PROVISIONAL_BUDGET_AT,
        "provisional_ceiling_candidates": PROVISIONAL_CEILING,
        "budget_is_institutional": False,
        "timing_reproducibility": (
            "the seed reproduces the same workload; timing observations are "
            "expected to vary"
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="B-0.4 SP lookup baseline benchmark")
    parser.add_argument("--artifact", type=Path, default=None,
                        help="Real shortest_paths.pt. Its own slices are timed.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--implementations", default="current",
        help=(
            "Comma-separated: current, global, slices. **Run one prototype per "
            "process** — peak RSS is a process high-water mark, so building two "
            "in one process attributes the second's cost to whichever ran first."
        ),
    )
    args = parser.parse_args(argv)

    requested = [name.strip() for name in args.implementations.split(",") if name.strip()]
    unknown = [name for name in requested if name not in ("current", "global", "slices")]
    if unknown:
        parser.error(f"unknown implementation(s): {', '.join(unknown)}")
    if not requested:
        parser.error("--implementations may not be empty")
    prototypes = [name for name in requested if name != "current"]

    torch.manual_seed(args.seed)
    cells = itertools.count()
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    builds: List[Dict[str, Any]] = []

    def prepare(lookup) -> Dict[str, Any]:
        tables: Dict[str, Any] = {}
        for name in requested:
            if name == "current":
                tables[name] = lookup
                continue
            index, record = build_index(name, lookup)
            tables[name] = index
            builds.append(record)
            print(json.dumps(record), flush=True)
        return tables

    if args.artifact is not None:
        mode = "artifact"
        lookup, lengths, targets, artifact_meta = build_artifact_lookup(
            args.artifact, args.max_hops
        )
        phenotype_ids = sorted(lookup.offsets)
        time_table({"table": "artifact"}, lookup, lengths, targets, phenotype_ids,
                   args.seed, rows, skipped, cells, prepare(lookup))
        source: Dict[str, Any] = {"source": "artifact", **artifact_meta}
        # A real artifact is one of §3.1's two requirements. The other is a
        # deployment-equivalent CPU, which this script cannot self-attest — and
        # must not gain a flag that would let it.
        verdict = (
            "artifact slices timed; baseline acceptance remains subject to the "
            "deployment-equivalent CPU gate"
        )
    else:
        mode = "synthetic"
        for labels, lookup, lengths, targets in synthetic_tables(args.max_hops, args.seed):
            time_table(labels, lookup, lengths, targets, list(range(len(lengths))),
                       args.seed, rows, skipped, cells, prepare(lookup))
        source = {
            "source": "synthetic",
            "reason": "no shortest_paths.pt supplied; none exists in development",
            "declared_mean_slice_lengths": list(SYNTHETIC_MEAN_SLICE_LENGTHS),
            "declared_target_space": SYNTHETIC_TARGET_SPACE,
        }
        # PLAN_B04 §3.1: a synthetic run cannot accept the deployed baseline, and
        # is never a measured artifact distribution however it was parameterised.
        verdict = (
            "synthetic sensitivity sweep complete; production replacement decision "
            "pending institutional run"
        )

    report = {
        "stage": "B-0.4 prototype" if prototypes else "B-0.4 baseline",
        "implementations": requested,
        "index_builds": builds,
        # `ru_maxrss` is a process high-water mark, so a second prototype built in
        # the same process inherits the first's peak. True only when exactly one
        # prototype was built here; the numbers are reported either way and the
        # flag says how to read them.
        "memory_attribution_isolated": len(prototypes) <= 1,
        "slice_source": source,
        "provenance": provenance(args, mode),
        "verdict": verdict,
        "rows": rows,
        "skipped": skipped,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.output}", file=sys.stderr)
    print(f"\nverdict: {verdict}", file=sys.stderr)
    print(f"rows: {len(rows)}, skipped: {len(skipped)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
