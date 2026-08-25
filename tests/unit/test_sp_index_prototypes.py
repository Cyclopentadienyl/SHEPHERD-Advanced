"""The B-0.4 prototypes must agree with the primitive they would replace.

PLAN_B04 §6 makes exact agreement the first acceptance condition, "including
every path in §5.3": unreachable phenotypes, absent phenotypes, the float64
contract and the narrow `available` semantics. **Exact, not approximate** — the
distances are BFS hop counts stored as int8 and the unreachable value is
`max_hops + 1`, so every partial sum is a small integer and summation order
cannot change the bits. A tolerance here would hide a real disagreement.

The other half of the file is the uniqueness assertion (§5.3.2), which is a
deliberate behaviour *change*: a table the scanning implementation loads happily
must now be refused, and the test that proves it constructs such a table.

Module: tests/unit/test_sp_index_prototypes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sp_index_prototypes import (  # noqa: E402
    DuplicateRowError,
    build_global_key_index,
    build_slice_sorted_index,
    sp_mean_distances_global,
    sp_mean_distances_slices,
)
from src.inference.scoring import SPLookup, sp_mean_distances  # noqa: E402

MAX_HOPS = 5
N_TYPES = 3


def build_lookup(
    n_phenotypes: int = 12,
    n_targets: int = 40,
    seed: int = 0,
    max_hops: int = MAX_HOPS,
) -> SPLookup:
    """A table with the two properties the real loader guarantees.

    Grouped by phenotype with contiguous offsets, and **exactly one row per
    `(phenotype, target, target_type)`** — the invariant the offline BFS produces
    and the one the prototypes' binary searches rely on. Slice-internal order is
    deliberately shuffled, because the loader's `argsort` is unstable and leaves
    it arbitrary (`pipeline.py:496`); a prototype that only worked on
    already-sorted slices would pass a test built any other way.
    """
    generator = torch.Generator().manual_seed(seed)
    targets, types, distances, offsets = [], [], [], {}
    cursor = 0

    for phenotype in range(n_phenotypes):
        pairs = [(t, ty) for t in range(n_targets) for ty in range(N_TYPES)]
        keep = torch.randperm(len(pairs), generator=generator)[: len(pairs) // 2]
        chosen = [pairs[int(i)] for i in keep]
        if not chosen:
            continue
        for target, target_type in chosen:
            targets.append(target)
            types.append(target_type)
        distances.extend(
            torch.randint(0, max_hops + 1, (len(chosen),), generator=generator).tolist()
        )
        offsets[phenotype] = (cursor, cursor + len(chosen))
        cursor += len(chosen)

    return SPLookup(
        target=torch.tensor(targets, dtype=torch.int32),
        target_type=torch.tensor(types, dtype=torch.int8),
        distance=torch.tensor(distances, dtype=torch.int8),
        offsets=offsets,
        max_hops=max_hops,
    )


PROTOTYPES = (
    ("global", build_global_key_index, sp_mean_distances_global),
    ("slices", build_slice_sorted_index, sp_mean_distances_slices),
)


@pytest.mark.parametrize("name,build,query", PROTOTYPES)
@pytest.mark.parametrize("target_type_idx", range(N_TYPES))
def test_agrees_exactly_with_the_scanning_primitive(name, build, query, target_type_idx):
    """The whole acceptance condition, over every target type in the table."""
    lookup = build_lookup()
    index = build(lookup)

    phenotypes = [0, 3, 7, 11]
    candidates = list(range(0, 40, 3))

    expected_d, expected_a = sp_mean_distances(
        lookup, phenotypes, candidates, target_type_idx
    )
    actual_d, actual_a = query(index, phenotypes, candidates, target_type_idx)

    assert torch.equal(actual_d, expected_d), name
    assert torch.equal(actual_a, expected_a), name
    assert actual_d.dtype is torch.float64, name


@pytest.mark.parametrize("name,build,query", PROTOTYPES)
def test_agrees_when_a_phenotype_is_absent_from_offsets(name, build, query):
    """An absent phenotype contributes `unreachable`, it is not dropped.

    Dropping it would divide by a smaller denominator and silently *improve* the
    candidate, which is the failure mode §5.3.3 names.
    """
    lookup = build_lookup()
    index = build(lookup)
    absent = 9_999
    assert absent not in lookup.offsets

    phenotypes = [1, absent, 4]
    candidates = [0, 5, 17]

    expected_d, expected_a = sp_mean_distances(lookup, phenotypes, candidates, 0)
    actual_d, actual_a = query(index, phenotypes, candidates, 0)

    assert torch.equal(actual_d, expected_d), name
    assert torch.equal(actual_a, expected_a), name


@pytest.mark.parametrize("name,build,query", PROTOTYPES)
def test_agrees_when_every_phenotype_is_unreachable(name, build, query):
    """A candidate no phenotype reaches is still *computed*, at the largest value."""
    lookup = build_lookup()
    index = build(lookup)

    unreachable_target = 10_000  # outside the table's target range entirely
    phenotypes = [0, 1, 2]

    expected_d, expected_a = sp_mean_distances(
        lookup, phenotypes, [unreachable_target], 0
    )
    actual_d, actual_a = query(index, phenotypes, [unreachable_target], 0)

    assert torch.equal(actual_d, expected_d), name
    assert float(actual_d[0]) == pytest.approx(lookup.unreachable_distance)
    assert bool(actual_a[0]) is True


@pytest.mark.parametrize("name,build,query", PROTOTYPES)
def test_agrees_on_an_out_of_range_target_type(name, build, query):
    """A target type the table never stores misses for every candidate."""
    lookup = build_lookup()
    index = build(lookup)

    phenotypes = [0, 2]
    candidates = [1, 2, 3]

    expected_d, expected_a = sp_mean_distances(lookup, phenotypes, candidates, N_TYPES)
    actual_d, actual_a = query(index, phenotypes, candidates, N_TYPES)

    assert torch.equal(actual_d, expected_d), name
    assert torch.equal(actual_a, expected_a), name


@pytest.mark.parametrize("name,build,query", PROTOTYPES)
def test_agrees_on_negative_and_out_of_domain_ids(name, build, query):
    """Ids outside the table's range are misses, never aliases onto a real row.

    A clamp alone would map `target = 10_000` onto the largest stored target and
    return *its* distance, which is a wrong answer rather than a missing one.
    """
    lookup = build_lookup()
    index = build(lookup)

    phenotypes = [0, -1, 500]
    candidates = [-1, 0, 10_000]

    expected_d, expected_a = sp_mean_distances(lookup, phenotypes, candidates, 0)
    actual_d, actual_a = query(index, phenotypes, candidates, 0)

    assert torch.equal(actual_d, expected_d), name
    assert torch.equal(actual_a, expected_a), name


@pytest.mark.parametrize("name,build,query", PROTOTYPES)
def test_available_semantics_stay_narrow(name, build, query):
    """False only for nothing-to-measure-from — §5.3.4, unchanged."""
    lookup = build_lookup()
    index = build(lookup)

    no_phenotypes_d, no_phenotypes_a = query(index, [], [1, 2, 3], 0)
    assert no_phenotypes_a.tolist() == [False, False, False], name
    assert no_phenotypes_d.tolist() == [0.0, 0.0, 0.0], name
    assert no_phenotypes_d.dtype is torch.float64, name

    no_candidates_d, no_candidates_a = query(index, [0, 1], [], 0)
    assert no_candidates_d.numel() == 0, name
    assert no_candidates_a.numel() == 0, name


@pytest.mark.parametrize("name,build,query", PROTOTYPES)
def test_single_phenotype_and_single_candidate(name, build, query):
    """The shape the deployed caller actually uses: `C = 1`, one call per candidate."""
    lookup = build_lookup()
    index = build(lookup)

    for candidate in range(6):
        expected_d, _ = sp_mean_distances(lookup, [2], [candidate], 1)
        actual_d, _ = query(index, [2], [candidate], 1)
        assert torch.equal(actual_d, expected_d), (name, candidate)


@pytest.mark.parametrize("name,build", [(n, b) for n, b, _ in PROTOTYPES])
def test_duplicate_rows_are_refused(name, build):
    """§5.3.2's deliberate behaviour change, proven on a table that violates it.

    The scanning implementation reads this table without complaint, so the test
    asserts both halves: the old path tolerates it, the new one refuses.
    """
    duplicated = SPLookup(
        target=torch.tensor([4, 4, 7], dtype=torch.int32),
        target_type=torch.tensor([1, 1, 1], dtype=torch.int8),
        distance=torch.tensor([2, 3, 1], dtype=torch.int8),
        offsets={0: (0, 3)},
        max_hops=MAX_HOPS,
    )

    tolerated, _ = sp_mean_distances(duplicated, [0], [4], 1)
    assert float(tolerated[0]) == 2.0, "the scanning path takes the first row"

    with pytest.raises(DuplicateRowError, match="Rebuild the table"):
        build(duplicated)


@pytest.mark.parametrize("name,build", [(n, b) for n, b, _ in PROTOTYPES])
def test_negative_ids_are_refused_at_build(name, build):
    """A negative component would alias onto another triple in a positional key."""
    negative = SPLookup(
        target=torch.tensor([-1, 2], dtype=torch.int32),
        target_type=torch.tensor([0, 0], dtype=torch.int8),
        distance=torch.tensor([1, 2], dtype=torch.int8),
        offsets={0: (0, 2)},
        max_hops=MAX_HOPS,
    )
    with pytest.raises(ValueError, match="non-negative"):
        build(negative)


def test_key_domain_overflow_is_caught_before_any_int64_key_exists():
    """The check must run in Python integers — §5.2.

    `max_target` near the int64 ceiling makes `stride_phenotype` overflow. If the
    check were performed in int64 it would compare two already-wrapped values and
    pass, so this asserts the failure rather than the arithmetic.
    """
    huge = SPLookup(
        target=torch.tensor([0, 2**62], dtype=torch.int64),
        target_type=torch.tensor([0, 3], dtype=torch.int8),
        distance=torch.tensor([1, 2], dtype=torch.int8),
        offsets={0: (0, 2)},
        max_hops=MAX_HOPS,
    )
    with pytest.raises(ValueError, match="int64 max"):
        build_global_key_index(huge)


@pytest.mark.parametrize("name,build,query", PROTOTYPES)
def test_slice_internal_order_does_not_change_the_answer(name, build, query):
    """Two tables differing only in within-slice row order must agree.

    The loader's phenotype `argsort` is unstable, so the order a real table
    arrives in is arbitrary. This is what makes the uniqueness assertion
    load-bearing rather than defensive: with exactly one row per triple, storage
    order cannot affect any answer.
    """
    lookup = build_lookup(seed=3)
    start, end = lookup.offsets[0]
    permutation = torch.randperm(end - start, generator=torch.Generator().manual_seed(9))

    shuffled = SPLookup(
        target=torch.cat([lookup.target[start:end][permutation], lookup.target[end:]]),
        target_type=torch.cat(
            [lookup.target_type[start:end][permutation], lookup.target_type[end:]]
        ),
        distance=torch.cat(
            [lookup.distance[start:end][permutation], lookup.distance[end:]]
        ),
        offsets=dict(lookup.offsets),
        max_hops=lookup.max_hops,
    )

    phenotypes = [0, 1]
    candidates = list(range(12))
    first, _ = query(build(lookup), phenotypes, candidates, 2)
    second, _ = query(build(shuffled), phenotypes, candidates, 2)
    assert torch.equal(first, second), name


@pytest.mark.parametrize("name,build,query", PROTOTYPES)
def test_an_empty_table_agrees_instead_of_raising(name, build, query):
    """Nothing to find means everything unreachable — not an IndexError.

    Approach A gathers `keys[position.clamp(...)]`, which on an empty table
    indexes element 0 of a zero-length tensor and raised, while the scanning
    primitive returned `unreachable` for every candidate. Approach B never had
    the hazard — an empty table has no offsets, so every phenotype takes its
    missing-bounds path — but both are asserted so the pair cannot drift.
    """
    empty = SPLookup(
        target=torch.zeros(0, dtype=torch.int32),
        target_type=torch.zeros(0, dtype=torch.int8),
        distance=torch.zeros(0, dtype=torch.int8),
        offsets={},
        max_hops=MAX_HOPS,
    )

    expected_d, expected_a = sp_mean_distances(empty, [0, 1], [3, 4], 0)
    actual_d, actual_a = query(build(empty), [0, 1], [3, 4], 0)

    assert torch.equal(actual_d, expected_d), name
    assert torch.equal(actual_a, expected_a), name
    assert actual_d.tolist() == [float(MAX_HOPS + 1)] * 2, name
    assert actual_a.tolist() == [True, True], name


def test_reported_memory_separates_measured_residence_from_projection():
    """§9.4's selection criterion must not be read off a projection.

    `SliceSortedIndex` clones all three tensors so the benchmark can hold both
    implementations at once, so its *actual* residence is not zero even though
    its production increment is. Reporting only the projection — as a first
    version did — would have shown approach B costing nothing at all.
    """
    lookup = build_lookup()
    table_bytes = sum(
        t.numel() * t.element_size()
        for t in (lookup.target, lookup.target_type, lookup.distance)
    )

    slices = build_slice_sorted_index(lookup)
    assert slices.production_incremental_bytes_projected == 0
    assert slices.resident_bytes_actual == table_bytes

    global_index = build_global_key_index(lookup)
    # The key column alone is the projection; the retained distance copy is not.
    assert global_index.production_incremental_bytes_projected == (
        global_index.keys.numel() * 8
    )
    assert global_index.resident_bytes_actual > (
        global_index.production_incremental_bytes_projected
    )
