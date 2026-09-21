"""The served shortest-path primitive, checked against an independent program.

**The comparison is the test, so the two sides must not be the same program.**
`src/inference/sp_index.py` builds a composite int64 key and binary-searches it;
`scripts/sp_scan_reference.py` puts every row in a Python dictionary and looks
triples up. No shared data structure, no shared arithmetic, no shared file. The
equivalence tests this replaces compared the indexed implementation against a
scan that dispatch would have turned *into* the indexed implementation for the
same input — a comparison that passes for the wrong reason.

**Expected values for the out-of-domain cases are computed, not inherited.** The
scan that shipped until this change answered a query for target `2**32` with
target 0's distance, because it compared an int32 column against a Python int
and torch wraps a scalar the dtype cannot hold. Asserting the old answer would
freeze the defect; the reference cannot produce it, because a dictionary key is
an exact integer.

**Algorithm tests and artifact tests are different tests.** `build_sp_index`
accepts any non-negative ids and is exercised here with three target types,
which the producer never writes. What the producer promises — `target_type` in
{0, 1}, distances within the hop bound — is `validate_sp_artifact`'s, and is
tested as such.

Module: tests/unit/test_sp_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sp_scan_reference import (  # noqa: E402
    build_reference_table,
    sp_mean_distances_reference,
)
from src.inference.sp_index import (  # noqa: E402
    DuplicateRowError,
    SPArtifactError,
    build_sp_index,
    sp_mean_distances,
    validate_sp_artifact,
    validate_sp_columns,
)

MAX_HOPS = 5
N_TYPES = 3


def build_columns(n_phenotypes=12, n_targets=40, seed=0, max_hops=MAX_HOPS):
    """Raw columns with the one property the offline BFS guarantees.

    **Exactly one row per `(phenotype, target, target_type)`** — the invariant
    the producer's first-reached-wins BFS creates and the one binary search
    relies on. Row order is deliberately shuffled across the whole table, not
    only within a phenotype: the builder sorts, and a builder that only worked
    on already-grouped input would pass a test built any other way.
    """
    generator = torch.Generator().manual_seed(seed)
    phenotypes, targets, types = [], [], []

    for phenotype in range(n_phenotypes):
        pairs = [(t, ty) for t in range(n_targets) for ty in range(N_TYPES)]
        keep = torch.randperm(len(pairs), generator=generator)[: len(pairs) // 2]
        for index in keep.tolist():
            target, target_type = pairs[index]
            phenotypes.append(phenotype)
            targets.append(target)
            types.append(target_type)

    n_rows = len(phenotypes)
    distances = torch.randint(0, max_hops + 1, (n_rows,), generator=generator)
    shuffle = torch.randperm(n_rows, generator=generator)

    return (
        torch.tensor(phenotypes, dtype=torch.int64)[shuffle],
        torch.tensor(targets, dtype=torch.int64)[shuffle],
        torch.tensor(types, dtype=torch.int64)[shuffle],
        distances.to(torch.int8)[shuffle],
    )


def both(columns, max_hops=MAX_HOPS):
    """The served lookup and the independent reference, over the same rows.

    They take the same arguments on purpose: an adapter between them would be a
    place to hand the two sides different inputs without anyone noticing.
    """
    return (
        build_sp_index(*columns, max_hops),
        build_reference_table(*columns, max_hops),
    )


def agree(lookup, reference, phenotypes, candidates, target_type_idx):
    got_d, got_a = sp_mean_distances(lookup, phenotypes, candidates, target_type_idx)
    want_d, want_a = sp_mean_distances_reference(
        reference, phenotypes, candidates, target_type_idx
    )
    assert got_d.dtype == torch.float64 and want_d.dtype == torch.float64
    assert torch.equal(got_d, want_d), (
        f"index {got_d.tolist()} vs reference {want_d.tolist()}"
    )
    assert torch.equal(got_a, want_a)
    return got_d


class TestItAgreesWithAnIndependentProgram:

    @pytest.mark.parametrize("target_type_idx", range(N_TYPES))
    def test_over_every_target_type_in_the_table(self, target_type_idx):
        lookup, reference = both(build_columns())
        agree(lookup, reference, [0, 3, 7, 11], list(range(40)), target_type_idx)

    def test_when_a_phenotype_has_no_rows_at_all(self):
        """No special case is needed for it: it misses everywhere and
        contributes `unreachable`, which is what the reference does too."""
        lookup, reference = both(build_columns(n_phenotypes=4))
        agree(lookup, reference, [0, 99], list(range(10)), 0)

    def test_when_every_phenotype_is_unreachable(self):
        lookup, reference = both(build_columns(n_phenotypes=4, n_targets=6))
        agree(lookup, reference, [0, 1], [500, 501], 0)

    def test_on_a_target_type_above_the_table_s_range(self):
        lookup, reference = both(build_columns())
        agree(lookup, reference, [0, 1], [0, 1, 2], N_TYPES)

    def test_with_one_phenotype_and_one_candidate(self):
        lookup, reference = both(build_columns())
        agree(lookup, reference, [2], [5], 1)

    def test_on_an_empty_table(self):
        """Answered before the gather: indexing element 0 of an empty tensor
        raises, where the answer is simply that nothing was found."""
        empty = tuple(
            torch.tensor([], dtype=d) for d in (torch.int64, torch.int64, torch.int64, torch.int8)
        )
        lookup, reference = both(empty)
        got = agree(lookup, reference, [0, 1], [3, 4], 0)
        assert got.tolist() == [MAX_HOPS + 1, MAX_HOPS + 1]

    def test_row_order_does_not_change_the_answer(self):
        """The builder sorts; two shuffles of the same rows must land on the
        same index. A builder that depended on input order would pass every
        test above and fail here."""
        columns = build_columns()
        other = torch.randperm(columns[0].numel(), generator=torch.Generator().manual_seed(7))
        shuffled = tuple(c[other] for c in columns)

        first = build_sp_index(*columns, MAX_HOPS)
        second = build_sp_index(*shuffled, MAX_HOPS)

        assert torch.equal(first.keys, second.keys)
        assert torch.equal(first.distance, second.distance)


class TestAnIdOutsideTheTableCannotReadAnotherNodesDistance:
    """**The defect the retired scan had**, and the reason it was retired rather
    than repaired.

    The scan compared an int32 column against a Python int; torch wraps a scalar
    the dtype cannot hold, so `target = 2**32` matched target 0 and returned its
    real distance, marked available. The index masks out-of-domain ids by
    construction — it clamps them so `searchsorted` has something in range, then
    excludes them with a mask built from the *unclamped* values.
    """

    @pytest.mark.parametrize(
        "candidate,why",
        [
            (2**32, "wraps to 0 in int32"),
            (2**31, "wraps to a negative in int32"),
            (-1, "negative"),
            (10_000, "simply absent"),
        ],
    )
    def test_candidate_ids_outside_the_domain_miss(self, candidate, why):
        lookup, reference = both(build_columns())
        got = agree(lookup, reference, [0, 1], [candidate], 0)
        assert got.tolist() == [MAX_HOPS + 1], why

    @pytest.mark.parametrize("target_type_idx", [256, 2**32, -1])
    def test_target_types_outside_the_domain_miss(self, target_type_idx):
        lookup, reference = both(build_columns())
        got = agree(lookup, reference, [0, 1], [0, 1], target_type_idx)
        assert got.tolist() == [MAX_HOPS + 1, MAX_HOPS + 1]

    @pytest.mark.parametrize("phenotype", [2**32, -1, 10_000])
    def test_phenotype_ids_outside_the_domain_miss(self, phenotype):
        lookup, reference = both(build_columns())
        agree(lookup, reference, [phenotype], [0, 1, 2], 0)

    def test_a_present_neighbour_is_not_what_an_absent_id_returns(self):
        """The control that makes the four above mean something: target 0 is in
        the table with a real distance, and that is the value the wrap used to
        hand back for `2**32`."""
        columns = (
            torch.tensor([0, 0], dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([0, 0], dtype=torch.int64),
            torch.tensor([1, 3], dtype=torch.int8),
        )
        lookup, _ = both(columns)

        present = sp_mean_distances(lookup, [0], [0], 0)[0]
        aliasing = sp_mean_distances(lookup, [0], [2**32], 0)[0]

        assert present.tolist() == [1.0]
        assert aliasing.tolist() == [MAX_HOPS + 1]
        assert aliasing.tolist() != present.tolist()


class TestAvailableStaysNarrow:

    def test_it_is_false_only_when_there_was_nothing_to_measure_from(self):
        lookup, reference = both(build_columns())
        for phenotypes, candidates in (([], [0, 1]), ([0, 1], [])):
            _, available = sp_mean_distances(lookup, phenotypes, candidates, 0)
            assert not available.any()

    def test_a_candidate_nothing_reaches_is_still_available(self):
        lookup, _ = both(build_columns(n_phenotypes=2, n_targets=4))
        distances, available = sp_mean_distances(lookup, [0, 1], [900], 0)
        assert bool(available[0]) is True
        assert distances.tolist() == [MAX_HOPS + 1]


class TestTheBuilderRefusesWhatItCannotIndex:

    def test_duplicate_rows(self):
        columns = (
            torch.tensor([0, 0], dtype=torch.int64),
            torch.tensor([4, 4], dtype=torch.int64),
            torch.tensor([1, 1], dtype=torch.int64),
            torch.tensor([2, 3], dtype=torch.int8),
        )
        with pytest.raises(DuplicateRowError, match="duplicate"):
            build_sp_index(*columns, MAX_HOPS)

    def test_duplicate_rows_are_an_artifact_error_too(self):
        """So a caller that only wants "this table is unusable" catches one
        type. `DuplicateRowError` is a subclass, not a sibling."""
        assert issubclass(DuplicateRowError, SPArtifactError)
        assert issubclass(SPArtifactError, ValueError)

    @pytest.mark.parametrize("column", [0, 1, 2])
    def test_negative_ids(self, column):
        columns = [
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([2, 3], dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([1, 2], dtype=torch.int8),
        ]
        columns[column] = columns[column] * -1
        with pytest.raises(SPArtifactError, match="negative"):
            build_sp_index(*columns, MAX_HOPS)

    def test_a_key_domain_int64_cannot_hold(self):
        """**Caught in Python integers, before any int64 key exists.** Checked
        after the key was built, the comparison would be between two values that
        had already wrapped."""
        huge = 2**40
        columns = (
            torch.tensor([huge], dtype=torch.int64),
            torch.tensor([huge], dtype=torch.int64),
            torch.tensor([huge], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int8),
        )
        with pytest.raises(SPArtifactError, match="int64 max"):
            build_sp_index(*columns, MAX_HOPS)

    def test_it_does_not_write_to_the_columns_it_is_given(self):
        """An earlier draft folded the key in place, which saves an allocation
        and silently corrupts the second of two builds from one set of columns.
        This is the test that stops it coming back."""
        columns = build_columns(n_phenotypes=3, n_targets=5)
        before = [c.clone() for c in columns]

        first = build_sp_index(*columns, MAX_HOPS)
        second = build_sp_index(*columns, MAX_HOPS)

        for got, want in zip(columns, before):
            assert torch.equal(got, want), "the builder wrote to its input"
        assert torch.equal(first.keys, second.keys)
        assert torch.equal(first.distance, second.distance)

    def test_it_sorts_exactly_once(self, monkeypatch):
        """D4's obligation after its mechanism went away: one sort, and the
        tensor it produces is the one kept. Counted rather than timed."""
        import src.inference.sp_index as module

        calls = []
        real = torch.sort

        def counted(*args, **kwargs):
            calls.append(args[0].numel())
            return real(*args, **kwargs)

        monkeypatch.setattr(module.torch, "sort", counted)
        build_sp_index(*build_columns(), MAX_HOPS)

        assert len(calls) == 1, f"expected one sort, saw {len(calls)}"


class TestValidationRunsOnTheTensorsAsLoaded:
    """**Before narrowing, which is the whole point.** A value a narrow type
    cannot hold does not raise on the way in — it wraps, exactly as the query's
    scalar comparison did — so a check afterwards inspects the wrapped value and
    finds it reasonable."""

    @staticmethod
    def _columns():
        return [
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([2, 3], dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([1, 2], dtype=torch.int8),
        ]

    def test_a_float_column_is_refused(self):
        columns = self._columns()
        columns[1] = columns[1].to(torch.float32)
        with pytest.raises(SPArtifactError, match="integers"):
            validate_sp_columns(*columns)

    def test_a_two_dimensional_column_is_refused(self):
        columns = self._columns()
        columns[0] = columns[0].unsqueeze(0)
        with pytest.raises(SPArtifactError, match="dimensions"):
            validate_sp_columns(*columns)

    def test_columns_of_different_lengths_are_refused(self):
        columns = self._columns()
        columns[3] = columns[3][:1]
        with pytest.raises(SPArtifactError, match="different lengths"):
            validate_sp_columns(*columns)

    def test_something_that_is_not_a_tensor_is_refused(self):
        columns = self._columns()
        columns[2] = [0, 1]
        with pytest.raises(SPArtifactError, match="not a tensor"):
            validate_sp_columns(*columns)

    def test_it_returns_the_row_count(self):
        assert validate_sp_columns(*self._columns()) == 2


class TestTheArtifactContractIsSeparateFromTheAlgorithm:
    """`build_sp_index` indexes any non-negative ids — the tests above use three
    target types. What `scripts/compute_shortest_paths.py` *writes* is narrower,
    and that is checked where the artifact is read, not where the index is
    built."""

    @staticmethod
    def _columns(types=(0, 1), distances=(1, 2)):
        return (
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([2, 3], dtype=torch.int64),
            torch.tensor(list(types), dtype=torch.int64),
            torch.tensor(list(distances), dtype=torch.int8),
        )

    def test_a_third_target_type_is_refused_as_an_artifact(self):
        with pytest.raises(SPArtifactError, match="target_type"):
            validate_sp_artifact(*self._columns(types=(0, 2)), MAX_HOPS)

    def test_the_same_third_type_indexes_fine_as_an_algorithm(self):
        """The distinction stated as a test, so neither check drifts into the
        other's job."""
        lookup = build_sp_index(*self._columns(types=(0, 2)), MAX_HOPS)
        assert lookup.n_rows == 2

    def test_a_distance_above_the_bound_is_refused(self):
        with pytest.raises(SPArtifactError, match="records a distance of 5"):
            validate_sp_artifact(*self._columns(distances=(1, 5)), 4)

    @pytest.mark.parametrize("below", [0, -1, -2])
    def test_a_distance_below_one_is_refused(self, below):
        """**The other end of the same rule, and the one that was missing.**
        `sp_scores_from_distances` is `1 / (1 + d)`: 0 scores a perfect 1.0, -1
        divides by zero, -2 scores -1.0. Each reaches the combined score as a
        number rather than as a refusal. The producer skips the source itself
        (`if dist == 0: continue`), so 1 is its floor."""
        with pytest.raises(SPArtifactError, match="records a distance of"):
            validate_sp_artifact(*self._columns(distances=(1, below)), MAX_HOPS)

    def test_a_distance_of_one_is_accepted(self):
        assert validate_sp_artifact(*self._columns(distances=(1, 1)), MAX_HOPS) == 2

    def test_the_same_zero_distance_indexes_fine_as_an_algorithm(self):
        """The split again: a hop count of 0 is not something the producer
        writes, and it is not something the index cannot handle."""
        lookup = build_sp_index(*self._columns(distances=(0, 2)), MAX_HOPS)
        assert lookup.n_rows == 2

    def test_a_distance_at_the_bound_is_accepted(self):
        assert validate_sp_artifact(*self._columns(distances=(1, 4)), 4) == 2

    def test_an_empty_table_passes_the_artifact_contract(self):
        empty = tuple(
            torch.tensor([], dtype=d)
            for d in (torch.int64, torch.int64, torch.int64, torch.int8)
        )
        assert validate_sp_artifact(*empty, MAX_HOPS) == 0


class TestWhatTheLookupReportsAboutItself:

    def test_resident_bytes_are_measured_off_the_tensors(self):
        """Not projected. The figure is the two tensors it holds, and nothing
        in `PLAN_B04.md` is quoted for it — that document measured a design
        that kept the id columns."""
        lookup = build_sp_index(*build_columns(), MAX_HOPS)
        expected = (
            lookup.keys.numel() * lookup.keys.element_size()
            + lookup.distance.numel() * lookup.distance.element_size()
        )
        assert lookup.resident_bytes() == expected
        assert lookup.n_rows == lookup.keys.numel()

    def test_the_id_columns_are_not_retained(self):
        """The saving is that they are gone, so this asserts their absence
        rather than trusting the docstring."""
        lookup = build_sp_index(*build_columns(), MAX_HOPS)
        for retired in ("target", "target_type", "offsets", "phenotype"):
            assert not hasattr(lookup, retired), f"{retired} came back"

    def test_unreachable_is_one_past_the_bound(self):
        lookup = build_sp_index(*build_columns(), MAX_HOPS)
        assert lookup.unreachable_distance == float(MAX_HOPS + 1)
