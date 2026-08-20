"""B-0.4 prototypes — two indexed replacements for the linear slice scan.

`src/inference/scoring.py: sp_mean_distances` answers every
`(candidate, phenotype)` pair by linearly scanning that phenotype's whole slice:
`O(C x P x L)`. On the real artifact that misses the plan's provisional latency
budget by 1.7-2.5x (`docs/working/scorer-measurement/PLAN_B04.md` §9.3), which is
what moved the stage into this prototype phase.

**These are measurement subjects, not production code.** PLAN_B04 §5.2 says
"prototype both; productionise one", so both live here, next to the benchmark
that times them, and neither is importable from `src/`. When one wins, it moves
into the loader and the primitive and this file goes away with the loser. That is
also why there is no strategy base class, no registry and no runtime selector:
nothing here is meant to survive as a choice.

**What differs between them is memory, not asymptotics.** Both turn the `O(L)`
scan into `O(log L)`. The distinction PLAN_B04 §9.4 and §10.2 care about is that
approach A must *retain* a full-length int64 key column — 3.44 GB at the measured
429,971,678 rows — while approach B retains nothing beyond a reordering of
tensors that already exist. On SPARK that memory is unified with the model's, so
it is a first-order selection criterion rather than a footnote.

**Nothing on a query path may touch more than `O(log L)` rows.** Casting a slice
to a wider dtype to satisfy `torch.searchsorted` would be `O(L)` and would
quietly reintroduce the cost being removed, so the query values are cast to the
stored dtype instead, never the other way round.

Module: scripts/sp_index_prototypes.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor

if __package__ in (None, ""):  # direct execution, as the other scripts allow
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

__all__ = [
    "DuplicateRowError",
    "GlobalKeyIndex",
    "SliceSortedIndex",
    "build_global_key_index",
    "build_slice_sorted_index",
    "sp_mean_distances_global",
    "sp_mean_distances_slices",
]

INT64_MAX = 2**63 - 1

_DUPLICATE_MESSAGE = (
    "duplicate (phenotype, target, target_type) rows in the shortest-path table. "
    "Binary search returns the first match in *sorted* order, which need not be "
    "the first in original order, so the indexed and scanning implementations "
    "could disagree on the distance. Rebuild the table"
)


class DuplicateRowError(ValueError):
    """A `(phenotype, target, target_type)` triple appeared more than once.

    Its own type because PLAN_B04 §5.3.2 makes this a **startup** failure with a
    specific remedy — rebuild the table — rather than one more malformed-input
    error. The current first-match implementation loads such a table happily and
    silently answers with whichever row the loader's unstable `argsort` left
    first, so refusing is a behaviour change, and it is the intended one.
    """


# =============================================================================
# Shared helpers
# =============================================================================
@dataclass(frozen=True)
class _Domain:
    """The id ranges actually present in a table, and the strides built from them.

    **Every number comes from the tensors, never from
    `shortest_paths.meta.json`.** The loader treats that sidecar as optional and
    swallows any failure reading it (`pipeline.py:523-531`), and it can be stale
    or mismatched with the `.pt` beside it. Correctness may not depend on it.
    """

    max_phenotype: int
    max_target: int
    max_type: int
    stride_phenotype: int
    stride_type: int


def _derive_domain(phenotype: Tensor, target: Tensor, target_type: Tensor) -> _Domain:
    """Validate ids and size the composite key.

    **The overflow check runs in Python integers, before any int64 key exists.**
    Python integers are arbitrary precision; a check performed in int64 has
    already overflowed and would be comparing two wrapped values.
    """
    for name, tensor in (
        ("phenotype", phenotype),
        ("target", target),
        ("target_type", target_type),
    ):
        if tensor.numel() and int(tensor.min()) < 0:
            raise ValueError(
                f"{name} ids must be non-negative; the composite key is a "
                "positional encoding and a negative component would alias onto "
                "another triple"
            )

    max_phenotype = int(phenotype.max()) if phenotype.numel() else 0
    max_target = int(target.max()) if target.numel() else 0
    max_type = int(target_type.max()) if target_type.numel() else 0

    stride_type = max_target + 1
    stride_phenotype = stride_type * (max_type + 1)
    largest = max_phenotype * stride_phenotype + max_type * stride_type + max_target
    if largest > INT64_MAX:
        raise ValueError(
            f"the composite key domain needs {largest} > int64 max {INT64_MAX} "
            f"(phenotype<={max_phenotype}, target_type<={max_type}, "
            f"target<={max_target}). Approach A is not applicable to this table"
        )
    return _Domain(
        max_phenotype=max_phenotype,
        max_target=max_target,
        max_type=max_type,
        stride_phenotype=stride_phenotype,
        stride_type=stride_type,
    )


def _phenotype_column(lookup) -> Tensor:
    """Rebuild the phenotype column `SPLookup` discarded in favour of `offsets`.

    The loader keeps `_sp_ph` but `SPLookup` does not carry it, so approach A
    reconstructs it. Contiguity is guaranteed by the loader's phenotype sort
    (`pipeline.py:496-519`), so this is one assignment per phenotype rather than
    per row.
    """
    column = torch.empty(lookup.target.numel(), dtype=torch.int64)
    for phenotype_idx, (start, end) in lookup.offsets.items():
        column[start:end] = phenotype_idx
    return column


def _tensor_bytes(*tensors: Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _all_unreachable(n_candidates: int, unreachable: float) -> Tuple[Tensor, Tensor]:
    """Every candidate missed, but every candidate still *computed*.

    `available` is True: there were phenotypes and there were candidates, so
    something was measured. §5.3.4 keeps that Boolean narrow.
    """
    return (
        torch.full((n_candidates,), unreachable, dtype=torch.float64),
        torch.ones(n_candidates, dtype=torch.bool),
    )


def _empty_result(n_candidates: int) -> Tuple[Tensor, Tensor]:
    return (
        torch.zeros(n_candidates, dtype=torch.float64),
        torch.zeros(n_candidates, dtype=torch.bool),
    )


def _query_values(values: Sequence[int], stored: Tensor, ceiling: int) -> Tensor:
    """Cast the *query* to the stored dtype, and say which entries are in range.

    Returns ``(safe, in_range)``: a tensor safe to hand to `torch.searchsorted`
    against `stored`, and the mask the caller must apply to its hits. An id above
    the table's maximum cannot match any row, so it is clamped into range for the
    search and then excluded — **clamping without the mask would alias it onto a
    real row** and return that row's distance, which is a wrong answer rather
    than a missing one.
    """
    as_long = torch.as_tensor(list(values), dtype=torch.int64)
    in_range = (as_long >= 0) & (as_long <= ceiling)
    safe = as_long.clamp(min=0, max=max(ceiling, 0)).to(stored.dtype)
    return safe, in_range


# =============================================================================
# Approach A — one global composite key
# =============================================================================
@dataclass(frozen=True)
class GlobalKeyIndex:
    """A monotone int64 key over the whole table, plus distances in key order.

    `target`, `target_type` and `phenotype` are **not** retained: the key encodes
    all three. A production adoption would still have to keep them, because
    `pipeline.py:533-536` calls `_sp_tg`/`_sp_ty`/`_sp_di` part of the class's
    observable surface — so the memory this prototype reports is a **lower bound**
    on approach A's resident cost, not an estimate of it.
    """

    keys: Tensor
    distance: Tensor
    domain: _Domain
    unreachable_distance: float

    @property
    def resident_bytes_actual(self) -> int:
        """Every tensor this object holds while the original lookup is **also** alive.

        What the prototype really costs in the benchmark process. Not the same as
        what production would pay, and reported separately from it because
        conflating the two is how a memory verdict goes wrong.
        """
        return _tensor_bytes(self.keys, self.distance)

    @property
    def production_incremental_bytes_projected(self) -> int:
        """Steady-state increment **if** the loader reorders in place instead of
        retaining both copies: the key column, and nothing else.

        A projection from the design, not a measurement — and a **lower bound** on
        the real thing, since `pipeline.py:533-536` calls `_sp_tg`/`_sp_ty`/`_sp_di`
        part of the class's observable surface, so production would keep the
        reordered target and target_type as well. The transient cost of the
        reorder itself is not here; only the RSS figures capture that.
        """
        return _tensor_bytes(self.keys)


def build_global_key_index(lookup) -> GlobalKeyIndex:
    """Sort the whole table once by `(phenotype, target_type, target)`.

    One sort, doing all four of PLAN_B04 §5.2's jobs: phenotype is the most
    significant component so runs stay contiguous, the other two order rows
    within a run, and equal adjacent keys are exactly the duplicates §5.3.2
    requires be detected.
    """
    phenotype = _phenotype_column(lookup)
    domain = _derive_domain(phenotype, lookup.target, lookup.target_type)

    keys = (
        phenotype * domain.stride_phenotype
        + lookup.target_type.to(torch.int64) * domain.stride_type
        + lookup.target.to(torch.int64)
    )
    del phenotype

    order = keys.argsort()
    keys = keys[order]
    distance = lookup.distance[order]
    del order

    if keys.numel() > 1 and bool((keys[1:] == keys[:-1]).any()):
        raise DuplicateRowError(_DUPLICATE_MESSAGE)

    return GlobalKeyIndex(
        keys=keys,
        distance=distance,
        domain=domain,
        unreachable_distance=lookup.unreachable_distance,
    )


def sp_mean_distances_global(
    index: GlobalKeyIndex,
    phenotype_indices: Sequence[int],
    target_indices: Sequence[int],
    target_type_idx: int,
) -> Tuple[Tensor, Tensor]:
    """Approach A's `sp_mean_distances`. Same signature, same contract.

    One `searchsorted` over all `P x C` query keys, one gather, one masked
    reduction — a constant number of kernel launches regardless of `P` or `C`.
    That is the advantage the batched caller shape realises; the singleton caller
    B-0.4 retains makes `C = 1`, so it pays those launches `C` times instead.

    **float64 for the whole computation**, per the primitive's contract. The sum
    is exact whatever its order: distances are BFS hop counts stored as int8
    (`pipeline.py:493`) and `unreachable_distance` is `max_hops + 1`, so every
    term is a small integer and every partial sum is an integer well inside
    float64's exactly-representable range. Pairwise or sequential summation give
    the same bits, which is what lets the equivalence test assert equality rather
    than a tolerance.
    """
    n_candidates = len(target_indices)
    if not phenotype_indices or n_candidates == 0:
        return _empty_result(n_candidates)

    unreachable = index.unreachable_distance
    n_phenotypes = len(phenotype_indices)
    domain = index.domain
    n_rows = index.keys.numel()

    # An empty table: nothing can be found, so everything is unreachable. Answered
    # before the gather because `keys[clamped]` would index element 0 of an empty
    # tensor and raise, where the scanning primitive returns unreachable. Approach
    # B has no equivalent hazard — an empty table has no offsets, so every
    # phenotype takes its missing-bounds path.
    if n_rows == 0:
        return _all_unreachable(n_candidates, unreachable)

    # A target_type outside the table's range cannot match any row, so every
    # candidate misses. Answered here rather than folded into the key, where an
    # out-of-domain component could alias onto a stored triple.
    if not 0 <= target_type_idx <= domain.max_type:
        return _all_unreachable(n_candidates, unreachable)

    # A phenotype absent from `offsets` needs no special case: it has no rows, so
    # every lookup against it misses and contributes `unreachable` — exactly what
    # the scanning implementation does for it. Only ids outside the *key domain*
    # need masking, because their key would not be constructible.
    phenotypes = torch.as_tensor(list(phenotype_indices), dtype=torch.int64)
    phenotype_ok = (phenotypes >= 0) & (phenotypes <= domain.max_phenotype)
    phenotypes = phenotypes.clamp(min=0, max=max(domain.max_phenotype, 0))

    targets = torch.as_tensor(list(target_indices), dtype=torch.int64)
    target_ok = (targets >= 0) & (targets <= domain.max_target)
    targets = targets.clamp(min=0, max=max(domain.max_target, 0))

    query = (
        phenotypes.unsqueeze(1) * domain.stride_phenotype
        + int(target_type_idx) * domain.stride_type
        + targets.unsqueeze(0)
    ).reshape(-1)  # (P * C,)

    position = torch.searchsorted(index.keys, query)
    clamped = position.clamp(max=max(n_rows - 1, 0))
    hit = (position < n_rows) & (index.keys[clamped] == query)
    hit = hit.reshape(n_phenotypes, n_candidates)
    hit &= phenotype_ok.unsqueeze(1)
    hit &= target_ok.unsqueeze(0)

    gathered = index.distance[clamped].reshape(n_phenotypes, n_candidates)
    contribution = torch.where(
        hit,
        gathered.to(torch.float64),
        torch.tensor(unreachable, dtype=torch.float64),
    )
    return contribution.sum(dim=0) / n_phenotypes, torch.ones(
        n_candidates, dtype=torch.bool
    )


# =============================================================================
# Approach B — per-phenotype sorted slices
# =============================================================================
@dataclass(frozen=True)
class SliceSortedIndex:
    """The existing three tensors, reordered within each phenotype's slice.

    **No key column is retained, and none is built globally.** Within a slice the
    rows are ordered by `(target_type, target)`, so for a fixed `target_type` the
    targets form one contiguous ascending run that `searchsorted` addresses
    directly on the tensors the loader already holds. That is the whole of
    approach B's memory advantage over A.
    """

    target: Tensor
    target_type: Tensor
    distance: Tensor
    offsets: Dict[int, Tuple[int, int]]
    domain: _Domain
    unreachable_distance: float

    @property
    def resident_bytes_actual(self) -> int:
        """**Not zero.** The build clones all three tensors, so in the benchmark
        process this prototype holds a full second copy of them beside the
        loader's originals. Reporting zero here — as a first version did — would
        present the production projection as a measurement.
        """
        return _tensor_bytes(self.target, self.target_type, self.distance)

    @property
    def production_incremental_bytes_projected(self) -> int:
        """Zero steady-state: production reorders the loader's own tensors and
        keeps no key column.

        The clones above exist only so the benchmark can hold both
        implementations at once. The **transient** cost of the reorder is not
        zero and is not here — but because this prototype sorts one slice at a
        time, that transient is one slice rather than the whole table, which is
        the difference the RSS figures should show.
        """
        return 0


def build_slice_sorted_index(lookup) -> SliceSortedIndex:
    """Sort **within** each phenotype's slice, one slice at a time.

    A single global lexicographic sort would produce the same ordering faster,
    but it would allocate the same full-length int64 key approach A does — which
    would erase the difference the two prototypes exist to measure. Peak memory
    here is one slice, so the cost moves into build *time*, and build time is a
    reported figure (PLAN_B04 §6).
    """
    domain = _derive_domain(
        _phenotype_column(lookup), lookup.target, lookup.target_type
    )

    target = lookup.target.clone()
    target_type = lookup.target_type.clone()
    distance = lookup.distance.clone()

    for start, end in lookup.offsets.values():
        if end - start <= 1:
            continue
        slice_key = target_type[start:end].to(torch.int64) * domain.stride_type
        slice_key += target[start:end].to(torch.int64)
        order = slice_key.argsort()
        ordered = slice_key[order]
        if bool((ordered[1:] == ordered[:-1]).any()):
            raise DuplicateRowError(_DUPLICATE_MESSAGE)
        del slice_key, ordered
        target[start:end] = target[start:end][order]
        target_type[start:end] = target_type[start:end][order]
        distance[start:end] = distance[start:end][order]

    return SliceSortedIndex(
        target=target,
        target_type=target_type,
        distance=distance,
        offsets=dict(lookup.offsets),
        domain=domain,
        unreachable_distance=lookup.unreachable_distance,
    )


def sp_mean_distances_slices(
    index: SliceSortedIndex,
    phenotype_indices: Sequence[int],
    target_indices: Sequence[int],
    target_type_idx: int,
) -> Tuple[Tensor, Tensor]:
    """Approach B's `sp_mean_distances`. Same signature, same contract.

    Per phenotype: two `searchsorted` calls narrow the slice to the requested
    `target_type`'s contiguous run, then one more resolves all `C` candidates
    inside it at once. `P` iterations of a constant number of launches, against
    approach A's constant total — the difference the two caller shapes exercise.

    Every search runs against the **stored** dtypes; the query is what gets cast.
    Widening a slice to int64 here would be `O(L)` and would reinstate the scan
    this prototype exists to remove.

    The float64 exactness argument is approach A's, unchanged.
    """
    n_candidates = len(target_indices)
    if not phenotype_indices or n_candidates == 0:
        return _empty_result(n_candidates)

    unreachable = index.unreachable_distance
    n_phenotypes = len(phenotype_indices)
    domain = index.domain

    total = torch.zeros(n_candidates, dtype=torch.float64)
    unreachable_row = torch.full((n_candidates,), unreachable, dtype=torch.float64)

    if not 0 <= target_type_idx <= domain.max_type:
        return _all_unreachable(n_candidates, unreachable)

    targets, target_ok = _query_values(target_indices, index.target, domain.max_target)
    type_value = torch.tensor([int(target_type_idx)], dtype=index.target_type.dtype)

    for phenotype_idx in phenotype_indices:
        bounds = index.offsets.get(phenotype_idx)
        if bounds is None:
            total += unreachable
            continue
        start, end = bounds
        if end <= start:
            total += unreachable
            continue

        type_slice = index.target_type[start:end]
        low = int(torch.searchsorted(type_slice, type_value, right=False))
        high = int(torch.searchsorted(type_slice, type_value, right=True))
        if high <= low:
            total += unreachable
            continue

        run = slice(start + low, start + high)
        target_run = index.target[run]
        position = torch.searchsorted(target_run, targets)
        clamped = position.clamp(max=target_run.numel() - 1)
        hit = (position < target_run.numel()) & (target_run[clamped] == targets)
        hit &= target_ok
        gathered = index.distance[run][clamped]
        total += torch.where(hit, gathered.to(torch.float64), unreachable_row)

    return total / n_phenotypes, torch.ones(n_candidates, dtype=torch.bool)
