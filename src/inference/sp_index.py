"""
The shortest-path primitive: one index, one builder, one query.
===============================================================
**There is one served implementation, and it is the indexed one.** The scan this
replaces answered every (phenotype, candidate) pair by walking that phenotype's
slice — over 550,000 slice scans for a single full-universe request. B-0.4
measured the alternatives and selected the global composite key;
`docs/working/scorer-measurement/PLAN_B04_PRODUCTIONISATION.md` carries the
selection and the decisions behind this module's shape.

**Why the scan is gone rather than kept behind a condition.** An optional index
with a scan behind it is a runtime selector whose condition happens to be
`is None`: two algorithms reachable from one call, diverging wherever their edge
cases differ, both needing maintenance and only one of them exercised in
production. That divergence was not hypothetical. The scan compared an int32
column against a Python int, and torch wraps a scalar the dtype cannot hold
rather than refusing it, so a query for target `2**32` was answered with target
0's distance — marked available, indistinguishable from a real one. A `SPLookup`
that exists has an index; there is no second state to diverge in.

**One sort, and the tensor it produces is the one kept.** The composite key
orders `(phenotype, target_type, target)` in a single int64 column, so one sort
does all three jobs at once: each phenotype's rows become contiguous, rows order
within a run, and duplicate triples land adjacent where the uniqueness check can
see them. Nothing re-sorts and nothing copies the result.

**The key replaces the id columns rather than joining them.** After it is built,
nothing reads `phenotype`, `target` or `target_type` — the query reaches rows by
`torch.searchsorted` — so the builder folds each column into the key in place and
releases it. What a served lookup holds is the key and the distances, and that is
the whole of it.

  Arithmetic, **not a measurement**: int64 key + int8 distance is 9 bytes a row,
  against the 4+4+1+1 of the four parallel columns the loader used to keep, plus
  an offsets dict. Transient cost is the sort's permutation and the sorted copy.
  `PLAN_B04.md`'s resident and peak figures describe a design that kept the id
  columns, so they are not evidence about this one; §6.1 item 10 of the
  productionisation plan requires this be re-measured before it is quoted.

**Out-of-domain ids are masked, never clamped into a neighbour.** An id above the
table's maximum has no constructible key, so it is excluded by a mask *after*
being clamped for the search. Clamping without the mask is precisely the defect
above, in a different coordinate system: it would return a real row's distance
for an id that is not in the table.

**Generic algorithm, specific artifact.** `build_sp_index` accepts any
non-negative integer ids, because the index is an algorithm and its tests are
algorithm tests. What the *artifact* may contain is narrower — `target_type` in
{0, 1} and distances within the hop bound — and `validate_sp_artifact` checks
that separately, on the tensors as loaded and before any narrowing, because a
value a narrow type cannot hold wraps there for the same reason the query's did.

Module: src/inference/sp_index.py

Dependencies: torch. Import lazily from anywhere that must stay importable
without it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import torch
from torch import Tensor

__all__ = [
    "DuplicateRowError",
    "SPArtifactError",
    "SPLookup",
    "SP_TARGET_TYPES",
    "build_sp_index",
    "sp_mean_distances",
    "validate_sp_artifact",
    "validate_sp_columns",
]

#: Python's integers are arbitrary precision, so the domain check below can be
#: made *before* an int64 key exists. Performed in int64 it would already have
#: overflowed and would be comparing two wrapped values.
INT64_MAX = 2**63 - 1

#: What the producer writes: 0 = gene, 1 = disease
#: (`scripts/compute_shortest_paths.py:20`). The *index* does not care; the
#: artifact contract does, which is why this constant is used only by
#: `validate_sp_artifact`.
SP_TARGET_TYPES = (0, 1)

_DUPLICATE_MESSAGE = (
    "duplicate (phenotype, target, target_type) rows in the shortest-path table. "
    "Binary search returns the first match in *sorted* order, which need not be "
    "the first in original order, so two builds of the same table could disagree "
    "on the distance. Rebuild the table"
)


class SPArtifactError(ValueError):
    """A shortest-path table that cannot be turned into an index.

    A `ValueError` subclass because that is what the loader's callers already
    catch and what the existing tests assert on; a distinct type because "this
    artifact is unusable" and "this argument is wrong" deserve to be told apart
    by code that wants to.
    """


class DuplicateRowError(SPArtifactError):
    """The table contains a (phenotype, target, target_type) triple twice."""


@dataclass(frozen=True)
class KeyDomain:
    """The id ranges actually present, and the strides built from them.

    **Every number comes from the tensors, never from `shortest_paths.meta.json`.**
    A present sidecar is binding for the hop bound and the loader refuses rather
    than guessing when it cannot be read — but what it states is `max_hops`, not
    the id ranges, and it can be stale relative to the `.pt` beside it. The
    domains are a property of the rows, so they are read from the rows.
    """

    max_phenotype: int
    max_target: int
    max_type: int
    stride_phenotype: int
    stride_type: int


@dataclass(frozen=True)
class SPLookup:
    """The shortest-path table as the served primitive reads it.

    **Constructing one is building the index**, which is why there is no
    `index` field and no state in which this object exists without one. Use
    `build_sp_index`; assembling this by hand is the second construction path
    the design exists to prevent.

    `keys` is the sorted composite key, `distance` the hop counts in the same
    order. The id columns are not here: the key encodes them and nothing reads
    them back.
    """

    keys: Tensor
    distance: Tensor
    domain: KeyDomain
    max_hops: int

    @property
    def unreachable_distance(self) -> float:
        """What a phenotype contributes when it cannot reach the target.

        One more than the search bound, so an unreachable phenotype penalises
        the mean without erasing the contribution of the ones that connected.
        """
        return float(self.max_hops + 1)

    @property
    def n_rows(self) -> int:
        return int(self.keys.numel())

    def resident_bytes(self) -> int:
        """Bytes this object holds. **Measured off the tensors, not projected.**"""
        return sum(t.numel() * t.element_size() for t in (self.keys, self.distance))


def validate_sp_columns(
    phenotype: Any,
    target: Any,
    target_type: Any,
    distance: Any,
    *,
    source: str = "the shortest-path table",
) -> int:
    """Shape and sign, checked on the tensors **as loaded**. Returns the row count.

    **Before any narrowing, and that ordering is the point.** The loader used to
    cast to int32/int8 on the way in, and a value the narrow type cannot hold
    does not raise there — it wraps, exactly as the query's scalar comparison
    did. Checking afterwards inspects the wrapped value and finds it reasonable.
    """
    columns = {
        "phenotype_idx": phenotype,
        "target_idx": target,
        "target_type": target_type,
        "distance": distance,
    }
    for name, column in columns.items():
        if not isinstance(column, Tensor):
            raise SPArtifactError(
                f"{source}: {name} is {type(column).__name__}, not a tensor"
            )
        if column.dim() != 1:
            raise SPArtifactError(
                f"{source}: {name} has {column.dim()} dimensions; the table is "
                "four parallel columns and a second dimension means it is not "
                "the artifact this reads"
            )
        if (
            column.dtype.is_floating_point
            or column.dtype.is_complex
            or column.dtype == torch.bool
        ):
            raise SPArtifactError(
                f"{source}: {name} is {column.dtype}; ids and hop counts are "
                "integers, and a float column would silently truncate"
            )

    lengths = {name: column.numel() for name, column in columns.items()}
    if len(set(lengths.values())) != 1:
        raise SPArtifactError(
            f"{source}: the columns have different lengths ({lengths}); they are "
            "parallel, so a row in one has no partner in another"
        )

    for name in ("phenotype_idx", "target_idx", "target_type"):
        column = columns[name]
        if column.numel() and int(column.min()) < 0:
            raise SPArtifactError(
                f"{source}: {name} contains a negative id; the composite key is "
                "a positional encoding and a negative component would alias onto "
                "another triple"
            )

    return lengths["distance"]


def validate_sp_artifact(
    phenotype: Any,
    target: Any,
    target_type: Any,
    distance: Any,
    max_hops: int,
    *,
    source: str = "the shortest-path table",
) -> int:
    """`validate_sp_columns` plus what *the producer* promises. Returns the row count.

    Kept apart from the index's own checks because they answer different
    questions. `build_sp_index` works for any non-negative ids and is tested as
    an algorithm; this says what `scripts/compute_shortest_paths.py` writes —
    `target_type` in {0, 1}, distances no larger than the bound the table was
    built to. A fixture exercising a third type is testing the algorithm, not
    violating the artifact.

    The distance ceiling is the floor check that used to run after the offsets
    were built: `max(distance)` proves the declared bound is not too **low** and
    can never prove it is not too high, because a 3-hop table is consistent with
    a declared 5. That one direction is still worth checking — it is the
    direction a stale sidecar from a smaller run fails in, and there the
    unreachable sentinel lands *below* distances really in the table, so an
    unreachable phenotype outranks a connected one.
    """
    n_rows = validate_sp_columns(
        phenotype, target, target_type, distance, source=source
    )
    if not n_rows:
        return 0

    observed_types = set(int(v) for v in torch.unique(target_type).tolist())
    unexpected = sorted(observed_types - set(SP_TARGET_TYPES))
    if unexpected:
        raise SPArtifactError(
            f"{source} records target_type {unexpected}; the producer writes "
            f"{list(SP_TARGET_TYPES)} (0 = gene, 1 = disease) and the served "
            "query asks for those two. A third type means this table was built "
            "by something else, and which rows belong to which kind of node is "
            "a guess."
        )

    observed = int(distance.max())
    if observed > max_hops:
        raise SPArtifactError(
            f"{source} records a distance of {observed}, above the "
            f"max_hops={max_hops} it was read with. The unreachable sentinel "
            "would sit below distances that are really in the table, which "
            "reorders candidates rather than merely mis-scoring them."
        )
    return n_rows


def _derive_domain(phenotype: Tensor, target: Tensor, target_type: Tensor) -> KeyDomain:
    """Size the composite key against the ids actually present.

    Signs are `validate_sp_columns`'s; this needs the maxima, and refuses a
    table whose triples cannot be encoded in int64 at all.
    """
    max_phenotype = int(phenotype.max()) if phenotype.numel() else 0
    max_target = int(target.max()) if target.numel() else 0
    max_type = int(target_type.max()) if target_type.numel() else 0

    stride_type = max_target + 1
    stride_phenotype = stride_type * (max_type + 1)
    largest = max_phenotype * stride_phenotype + max_type * stride_type + max_target
    if largest > INT64_MAX:
        raise SPArtifactError(
            f"the composite key domain needs {largest} > int64 max {INT64_MAX} "
            f"(phenotype<={max_phenotype}, target_type<={max_type}, "
            f"target<={max_target}); this table cannot be indexed"
        )
    return KeyDomain(
        max_phenotype=max_phenotype,
        max_target=max_target,
        max_type=max_type,
        stride_phenotype=stride_phenotype,
        stride_type=stride_type,
    )


def build_sp_index(
    phenotype: Tensor,
    target: Tensor,
    target_type: Tensor,
    distance: Tensor,
    max_hops: int,
) -> SPLookup:
    """The only way a queryable `SPLookup` comes into existence.

    **The columns are read, never written.** An earlier draft folded them into
    the key in place — `phenotype.mul_(stride)` and two `add_`s — which saves
    one full-length int64 allocation and mutates the caller's tensor. At
    hundreds of millions of rows that allocation is real, and the footgun is
    worse: a fixture that builds two lookups from one set of columns would get a
    correct first and a silently corrupt second, with nothing raising. So the
    key is allocated once, by the multiply, and the other two components are
    added into *it*.

    **One allocation, and no materialised products.** `add_(other, alpha=k)`
    computes `key += k * other` without building `k * other`, so the peak is the
    caller's columns plus one int64 key, not plus a temporary per component.
    Dropping the builder's own references as it goes lets a caller that passed
    by `pop` release each column at once; one that kept its copies keeps them,
    which is its choice to make and not a silent one.

    Raises:
        SPArtifactError: ids that cannot be encoded, columns that do not line
            up, or a table that is not four parallel integer columns.
        DuplicateRowError: the same triple twice — a subclass, so a caller that
            only wants "unusable artifact" catches the parent.
    """
    validate_sp_columns(phenotype, target, target_type, distance)
    domain = _derive_domain(phenotype, target, target_type)

    # The multiply allocates the key; the two adds write into it. `del` drops
    # only the builder's reference — the caller decides whether that is the last
    # one — but it drops it at the earliest point it can.
    keys = phenotype.to(torch.int64) * domain.stride_phenotype
    del phenotype
    keys.add_(target_type.to(torch.int64), alpha=domain.stride_type)
    del target_type
    keys.add_(target.to(torch.int64))
    del target

    # **One sort.** It is what makes phenotype runs contiguous, orders rows
    # within a run, and puts duplicate triples next to each other; there is no
    # second ordering pass anywhere, and the tensors this returns are the ones
    # the lookup keeps.
    keys, order = torch.sort(keys)
    distance = distance[order]
    del order

    if keys.numel() > 1 and bool((keys[1:] == keys[:-1]).any()):
        raise DuplicateRowError(_DUPLICATE_MESSAGE)

    return SPLookup(
        keys=keys, distance=distance, domain=domain, max_hops=int(max_hops)
    )


def _all_unreachable(n_candidates: int, unreachable: float) -> Tuple[Tensor, Tensor]:
    """Every candidate missed, and every candidate still *computed*.

    `available` is True: there were phenotypes and there were candidates, so
    something was measured. That Boolean stays narrow on purpose — a candidate
    that is simply far away is available, with a large distance.
    """
    return (
        torch.full((n_candidates,), unreachable, dtype=torch.float64),
        torch.ones(n_candidates, dtype=torch.bool),
    )


def sp_mean_distances(
    lookup: SPLookup,
    phenotype_indices: Sequence[int],
    target_indices: Sequence[int],
    target_type_idx: int,
) -> Tuple[Tensor, Tensor]:
    """Mean hop distance from the patient's phenotypes to each candidate.

    Returns ``(mean_distance, available)``, both ``(C,)``. ``available`` is False
    only when there was nothing to measure from — no phenotype indices, or no
    candidates.

    A phenotype with no path to a candidate contributes
    `lookup.unreachable_distance` rather than being dropped, so a candidate all
    of whose phenotypes are unreachable is still computed: it has a real value,
    the largest one.

    **The result is float64, and that is a contract rather than a default.** The
    code this lineage replaces accumulated in Python doubles, so a float32
    result would not be behaviour-preserving: a mean such as 83/24 differs
    between the two at the eighth significant digit. The contract covers the
    *whole* computation — it is not satisfied by reducing in float32 and
    widening the rounded result afterwards.

    **An id outside the table's domain misses; it does not alias onto a
    neighbour.** Queries are clamped so `searchsorted` has something in range to
    look at, and then excluded by a mask built from the *unclamped* values. The
    clamp without the mask returns a real row's distance for an id that is not in
    the table, which is a wrong answer rather than a missing one.
    """
    n_candidates = len(target_indices)
    n_phenotypes = len(phenotype_indices)
    unreachable = lookup.unreachable_distance

    if n_phenotypes == 0 or n_candidates == 0:
        return (
            torch.zeros(n_candidates, dtype=torch.float64),
            torch.zeros(n_candidates, dtype=torch.bool),
        )

    n_rows = lookup.n_rows
    # An empty table: nothing can be found, so everything is unreachable.
    # Answered before the gather because indexing element 0 of an empty tensor
    # raises where the answer is simply "no".
    if n_rows == 0:
        return _all_unreachable(n_candidates, unreachable)

    domain = lookup.domain
    # A target_type outside the table's range cannot match any row. Answered
    # here rather than folded into the key, where an out-of-domain component
    # could alias onto a stored triple.
    if not 0 <= target_type_idx <= domain.max_type:
        return _all_unreachable(n_candidates, unreachable)

    # A phenotype simply absent from the table needs no special case: it has no
    # rows, so every lookup against it misses and contributes `unreachable`.
    # Only ids outside the *key domain* need masking, because their key would
    # not be constructible.
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

    position = torch.searchsorted(lookup.keys, query)
    clamped = position.clamp(max=max(n_rows - 1, 0))
    hit = (position < n_rows) & (lookup.keys[clamped] == query)
    hit = hit.reshape(n_phenotypes, n_candidates)
    hit &= phenotype_ok.unsqueeze(1)
    hit &= target_ok.unsqueeze(0)

    gathered = lookup.distance[clamped].reshape(n_phenotypes, n_candidates)
    contribution = torch.where(
        hit,
        gathered.to(torch.float64),
        torch.tensor(unreachable, dtype=torch.float64),
    )
    return (
        contribution.sum(dim=0) / n_phenotypes,
        torch.ones(n_candidates, dtype=torch.bool),
    )
