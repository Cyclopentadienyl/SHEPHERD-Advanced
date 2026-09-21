"""
Benchmark: shortest-path lookup cost — work item B-0.4, baseline stage.
======================================================================
Measures `sp_mean_distances` (`src/inference/sp_index.py`) across the matrix
`docs/working/scorer-measurement/PLAN_B04.md` §5.4 defines, on **both caller
shapes**: the singleton loop production ships today, and the batched call B-1
and the offline harness will use.

**One shipped implementation, one baseline.** The three-way comparison this
script ran during B-0.4 — the old tensor scan against two candidate indexes —
selected the global composite key, and 5a made it the only served primitive.
What remains is `indexed`, which is what ships, and `reference`, the
independent Python-dictionary scan the equivalence tests use, kept so the
speedup has something to be measured against. The second candidate is in git
history; it depended on the CSR offsets layout that the served lookup no longer
has, so keeping it would have meant keeping that layout alive.

Read the plan before extending this script; several of its shapes are decisions
rather than conveniences.

**Linux only, by design.** Memory residence is the quantity this benchmark
exists to weigh, and it is read the way Linux reports it: `/proc/self/status`,
and `ru_maxrss` in kilobytes. A number from another platform would not be the
number the B-0.4 decision needs — the platform under measurement is the GB10 the
deployment runs on. `require_linux_memory_accounting` refuses right after argument
parsing — `--help` still works everywhere — rather than failing later from inside
a timing loop.

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
    """The four raw columns, in the shape the artifact has on disk.

    Uniqueness of `(phenotype, target, target_type)` holds by construction,
    because targets are sampled without replacement within each phenotype — the
    same property the real table has, and the one `build_sp_index` refuses a
    table for lacking.

    **Returns columns, not a lookup.** Both implementations this benchmark times
    are built from the same rows, and building the served index here would make
    the reference a derivative of it.

    Returns `(columns, slice_lengths, disease_targets)`.
    """
    generator = torch.Generator().manual_seed(seed)
    lengths = _slice_lengths(n_phenotypes, mean_length, distribution, generator)

    phenotypes: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    types: List[torch.Tensor] = []
    distances: List[torch.Tensor] = []

    actual: List[int] = []
    for phenotype, length in enumerate(lengths):
        length = min(length, target_space)
        picked = torch.randperm(target_space, generator=generator)[:length]
        phenotypes.append(torch.full((length,), phenotype, dtype=torch.int64))
        targets.append(picked.to(torch.int64))
        types.append(
            torch.randint(0, 2, (length,), generator=generator, dtype=torch.int64)
        )
        distances.append(
            torch.randint(1, max_hops + 1, (length,), generator=generator, dtype=torch.int8)
        )
        actual.append(length)

    columns = (
        torch.cat(phenotypes),
        torch.cat(targets),
        torch.cat(types),
        torch.cat(distances),
    )
    return columns, actual, list(range(target_space))


def synthetic_tables(
    max_hops: int, seed: int
) -> Iterator[Tuple[Dict[str, Any], Any, List[int], List[int]]]:
    n_phenotypes = max(PHENOTYPE_COUNTS) * 4
    for mean_length in SYNTHETIC_MEAN_SLICE_LENGTHS:
        for distribution in SYNTHETIC_DISTRIBUTIONS:
            columns, lengths, targets = build_synthetic_lookup(
                n_phenotypes, mean_length, distribution, SYNTHETIC_TARGET_SPACE,
                max_hops, seed,
            )
            yield (
                {"mean_slice_length": mean_length, "distribution": distribution},
                columns, lengths, targets,
            )


# =============================================================================
# Tables — the real artifact
# =============================================================================
def build_artifact_lookup(
    path: Path, max_hops: int
) -> Tuple[Any, List[int], List[int], Dict[str, Any]]:
    """The artifact's **own rows**, not a table shaped like them.

    **This used to be a second reader of the loader's layout**, mirroring
    `_load_shortest_paths`'s dtype compaction, sort-by-phenotype and offsets from
    run boundaries, and it carried a note saying so and asking to be reconciled
    the next time the production loader changed. That is this change. It now
    reads the four columns, checks them with the same
    `validate_sp_artifact` the loader calls, and hands them back — the index is
    built from them exactly once, by the one builder, wherever it is needed.

    Slice lengths are still reported, because the workload shaping needs them;
    they are counted off the phenotype column rather than read out of an offsets
    table that no longer exists.

    Returns `(columns, slice_lengths, disease_targets, provenance)`.
    """
    from src.inference.sp_index import validate_sp_artifact

    raw = torch.load(path, map_location="cpu", weights_only=True)
    required = {"phenotype_idx", "target_idx", "target_type", "distance"}
    missing = required - set(raw.keys())
    if missing:
        raise SystemExit(f"{path} is missing required keys: {sorted(missing)}")

    phenotype = raw.pop("phenotype_idx")
    target = raw.pop("target_idx")
    target_type = raw.pop("target_type")
    distance = raw.pop("distance")
    del raw

    validate_sp_artifact(
        phenotype, target, target_type, distance, max_hops, source=str(path)
    )

    ordered = phenotype[phenotype.argsort()]
    boundaries = torch.where(ordered[1:] != ordered[:-1])[0] + 1
    starts = torch.cat([torch.zeros(1, dtype=torch.int64), boundaries])
    ends = torch.cat([boundaries, torch.tensor([ordered.numel()], dtype=torch.int64)])
    keys = ordered[starts].tolist()
    starts_list, ends_list = starts.tolist(), ends.tolist()
    lengths = [ends_list[i] - starts_list[i] for i in range(len(keys))]
    del ordered, boundaries, starts, ends

    columns = (phenotype, target, target_type, distance)

    # Candidates are drawn from the real disease target space, not an invented one.
    disease_targets = torch.unique(target[target_type == DISEASE_TYPE_IDX]).tolist()

    from src.utils.fingerprint import file_sha256

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
    return columns, lengths, disease_targets, provenance


# =============================================================================
# Timing
# =============================================================================
def _time_once(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def _callers(query_fn):
    """Both caller shapes for one query function.

    The served primitive and the reference take the same arguments — deliberately,
    since PLAN_B04 §4.1 keeps the caller unchanged and an adapter between them
    would be a place to hand the two sides different inputs. So one factory
    serves both and there is no second copy of the loop to drift.
    """

    def singleton(table, phenotypes: Sequence[int], candidates: Sequence[int]) -> None:
        """The shape production ships: one call per candidate, `[0]` taken."""
        for candidate in candidates:
            query_fn(table, phenotypes, [candidate], DISEASE_TYPE_IDX)

    def batched(table, phenotypes: Sequence[int], candidates: Sequence[int]) -> None:
        """The shape B-1 and the offline harness use: one call, every candidate."""
        query_fn(table, phenotypes, candidates, DISEASE_TYPE_IDX)

    return singleton, batched


def require_linux_memory_accounting() -> None:
    """Refuse to start on a host where this benchmark cannot measure what it
    exists to measure.

    **This is a platform declaration, not a missing feature.** Memory residence is
    the whole point of the B-0.4 comparison — approach A's 3.44 GB is the cost
    being weighed against its speed — and it is read the way Linux reports it:
    `/proc/self/status`, and `ru_maxrss` **in kilobytes**, which is why `_rss_bytes`
    multiplies by 1024. The measured platform is the GB10 the deployment runs on,
    so there is nothing to port. A number from elsewhere would not be the number
    the decision needs.

    **A positive Linux gate, not a Windows blocklist**, and the difference is not
    pedantic. The first version of this check tested `sys.platform == "win32"`,
    copied from `src/retrieval/backends/cuvs_backend.py` — but only its first line.
    There the win32 test is a shortcut and the real decision is `import cuvs`, so a
    non-Linux POSIX host lands correctly on the `ImportError`. Copied without that
    second half, the shortcut became the whole check, and every non-Linux POSIX
    host fell straight through into accounting that assumes Linux semantics.
    Review reports that `ru_maxrss` on macOS is already in bytes, which would make
    the peak read 1024x too large — and it would look like a valid measurement,
    which is the failure worth refusing over. That specific claim is the reviewer's
    and is not verified here; what *is* verified is that this code encodes a Linux
    assumption, and a Linux assumption belongs behind a Linux gate.

    Linux ARM and Linux x86 share both semantics, so one gate covers the two
    deployment architectures that matter.

    Called immediately after argument parsing: `--help` still works everywhere,
    and the refusal lands **before artifact loading and any timed workload**.
    That is the cost worth protecting — `shortest_paths.pt` is gigabytes and takes
    minutes, and discovering the host cannot account for memory after paying it,
    through a `ModuleNotFoundError` raised inside a timing loop, is a far worse
    failure than one line at the start. It is *not* before "anything is loaded";
    `torch` is imported when this module is.
    """
    if not sys.platform.startswith("linux"):
        raise SystemExit(
            f"benchmark_sp_lookup requires Linux memory accounting "
            f"(/proc/self/status, and ru_maxrss in kilobytes) and cannot run on "
            f"{sys.platform}. This is by design: the measurement is about memory "
            "residence on the deployment platform, which is Linux on ARM or x86. "
            "Nothing here needs a port — run it on the target host."
        )


def _rss_bytes() -> Dict[str, Optional[int]]:
    """Current **and** high-water RSS.

    `ru_maxrss` alone cannot attribute a prototype build: loading the artifact
    has already set a process peak far above anything the build adds, and a
    high-water mark never comes back down. Current RSS does move, so the pair
    says both "how much is resident now" and "how high has this process ever
    been" — and their disagreement is itself the signal that the peak belongs to
    something earlier.

    **Linux only, deliberately** — see `require_linux_memory_accounting`. An
    earlier version of this docstring said non-Linux hosts get `None` "rather than
    a fabricated number". That was false: the `/proc/self/status` read is guarded,
    but `import resource` is not, and `resource` does not exist on Windows. The
    module is unavailable there, not degraded, and the CLI now says so before it
    spends anything.

    `ru_maxrss` for the high-water mark, `/proc/self/status` for current. The
    `/proc` read stays guarded because it can be absent on a Linux host too — a
    restricted container, for instance — and a missing current-RSS line is worth
    reporting as `None` beside a peak that was read successfully. No profiling
    dependency.
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


def build_index(implementation: str, columns) -> Tuple[Any, Dict[str, Any]]:
    """Build one implementation's table, timing it and recording what it costs.

    **One memory number now, and it is measured.** The pair this replaces
    reported what a prototype object held *beside* the loader's own tensors, and
    separately projected what production would hold if the loader reordered in
    place instead of keeping both copies. The projection existed because the
    prototype was not what shipped. It is now: `resident_bytes` is the served
    lookup's own tensors, and there is no second copy to project away.
    `PLAN_B04.md`'s figures describe the design that kept the id columns and are
    not evidence about this one.

    The reference is built here too, so its cost is reported on the same terms
    rather than hidden — it is a Python dictionary over every row, which is the
    honest reason it is a correctness baseline and not a candidate.
    """
    from scripts.sp_scan_reference import build_reference_table
    from src.inference.sp_index import build_sp_index

    max_hops = int(columns[3].max()) if columns[3].numel() else 0
    builders = {
        "indexed": lambda: build_sp_index(*columns, max_hops),
        "reference": lambda: build_reference_table(*columns, max_hops),
    }
    before = _rss_bytes()
    start = time.perf_counter()
    table = builders[implementation]()
    elapsed = time.perf_counter() - start
    after = _rss_bytes()
    record = {
        "record": "index_build",
        "implementation": implementation,
        "build_seconds": elapsed,
        "rss_before": before,
        "rss_after": after,
        "rows": int(columns[0].numel()),
    }
    if implementation == "indexed":
        record["resident_bytes"] = table.resident_bytes()
    return table, record


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
    columns: Any,
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

    `tables` maps an implementation name to the object its query function takes.
    Both take the same argument shape, deliberately — an adapter between them
    would be a place to hand the two sides different inputs without anyone
    noticing. All of them see the **same** phenotypes and candidates in the same
    cell, so a difference between two rows is the implementation and not the
    workload.
    """
    from scripts.sp_scan_reference import sp_mean_distances_reference
    from src.inference.sp_index import sp_mean_distances

    query_fns = {
        "indexed": sp_mean_distances,
        "reference": sp_mean_distances_reference,
    }

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
                    # **From the cell alone, never from `position`.** An earlier
                    # version used `(cell_index + position) % 2`, intending to
                    # decorrelate the two orders. With two implementations the
                    # rotation moves `position` in lockstep with `cell_index`, so
                    # the sum is constant per implementation *identity*: `current`
                    # came out singleton-first in 60/60 rows and the prototype
                    # batched-first in 60/60. Rotating the implementations
                    # cancelled the shape alternation instead of decorrelating it.
                    #
                    # Keyed on the cell, every implementation in a cell shares one
                    # shape order and each alternates across cells.
                    if cell_index % 2:
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
            "seeded randperm subset. candidates: seeded randperm subset of the "
            "disease target space, **without replacement** — a cell requesting "
            "more unique candidates than the space holds is reported as skipped."
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
        "--overwrite", action="store_true",
        help="Replace an existing --output file. Off by default: measurement "
             "artifacts are cited by digest and must not be replaced silently.",
    )
    parser.add_argument(
        "--implementations", default="indexed",
        help=(
            "Comma-separated: indexed, reference. `indexed` is what ships; "
            "`reference` is the independent Python-dictionary scan the "
            "equivalence tests use, kept here so the speedup has a baseline — "
            "it is O(rows) in Python and is not a candidate. **Run one per "
            "process** — peak RSS is a process high-water mark, so building two "
            "in one process attributes the second's cost to whichever ran first."
        ),
    )
    args = parser.parse_args(argv)

    # After parsing so `--help` works everywhere; before any artifact is loaded or
    # workload is timed, which is the cost that matters. Not before *anything* is
    # loaded — `torch` is imported when this module is.
    require_linux_memory_accounting()

    requested = [name.strip() for name in args.implementations.split(",") if name.strip()]
    unknown = [name for name in requested if name not in ("indexed", "reference")]
    if unknown:
        parser.error(f"unknown implementation(s): {', '.join(unknown)}")
    if not requested:
        parser.error("--implementations may not be empty")

    torch.manual_seed(args.seed)
    cells = itertools.count()
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    builds: List[Dict[str, Any]] = []

    def prepare(columns) -> Dict[str, Any]:
        """Every table this run will time, each built from the same rows.

        **No implementation is exempt.** The shape this replaces handed
        `current` the loader's own object and built only the prototypes, so one
        of the three never appeared in the build records and its cost was
        invisible.
        """
        tables: Dict[str, Any] = {}
        for name in requested:
            table, record = build_index(name, columns)
            tables[name] = table
            builds.append(record)
            print(json.dumps(record), flush=True)
        return tables

    if args.artifact is not None:
        mode = "artifact"
        columns, lengths, targets, artifact_meta = build_artifact_lookup(
            args.artifact, args.max_hops
        )
        phenotype_ids = torch.unique(columns[0]).tolist()
        time_table({"table": "artifact"}, columns, lengths, targets, phenotype_ids,
                   args.seed, rows, skipped, cells, prepare(columns))
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
        for labels, columns, lengths, targets in synthetic_tables(args.max_hops, args.seed):
            time_table(labels, columns, lengths, targets, list(range(len(lengths))),
                       args.seed, rows, skipped, cells, prepare(columns))
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
        "stage": "5a served primitive" if requested == ["indexed"] else "5a comparison",
        "implementations": requested,
        "index_builds": builds,
        # `ru_maxrss` is a process high-water mark, so a second table built in the
        # same process inherits the first's peak. True only when exactly one was
        # built here; the numbers are reported either way and the flag says how
        # to read them.
        "memory_attribution_isolated": len(requested) <= 1,
        "slice_source": source,
        "provenance": provenance(args, mode),
        "verdict": verdict,
        "rows": rows,
        "skipped": skipped,
    }

    if args.output:
        # **Refuse to overwrite evidence.** An artifact run takes minutes and its
        # output is cited by SHA-256 from the plan; a second run pointed at the
        # same path silently replaces the file those citations describe. This
        # already happened once: the repeat run was given the first run's exact
        # output paths, and the shell's own `2>` redirection truncated the
        # `time_*.txt` companions at launch, before any replacement existed.
        #
        # Deliberately *not* an auto-generated unique name — that would leave the
        # operator guessing which file the plan means. Name the new run, or say
        # `--overwrite` and mean it.
        if args.output.exists() and not args.overwrite:
            parser.error(
                f"{args.output} already exists. A measurement artifact is cited "
                "by digest and must not be replaced silently — give the new run "
                "its own --output name, or pass --overwrite if replacing this "
                "file is what you intend."
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.output}", file=sys.stderr)
    print(f"\nverdict: {verdict}", file=sys.stderr)
    print(f"rows: {len(rows)}, skipped: {len(skipped)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
