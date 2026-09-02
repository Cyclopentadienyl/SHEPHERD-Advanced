"""
Tests for the split feasibility audit.
======================================

**The probability tests use an independent exhaustive-subset oracle**, and they
exist because the specification's first version of
``p_no_training_representation`` was wrong and its brute-force check missed it.
That check used a single case, ``W = n_s``, which is precisely where the wrong
formula and the right one agree. A single case is not a check when the failure
mode is a degenerate coincidence, so every regime is enumerated here:
``W < n_s``, ``W = n_s``, ``W > n_s``, ``W > N - n_s``, ``n_s = 0`` and
``n_s = N``.

The oracle is enumeration of every ``C(N, W)`` subset and a direct count. It
shares no code with the implementation, which is the point: a closed form and a
telescoped product can both be wrong in the same way, but neither can be wrong in
the same way as counting.

Module: tests/unit/test_audit_split_feasibility.py
"""
from __future__ import annotations

import importlib.util
import json
import math
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_split_feasibility.py"

_spec = importlib.util.spec_from_file_location("audit_split_feasibility", SCRIPT)
audit = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(audit)


# --------------------------------------------------------------------------
# The oracle
# --------------------------------------------------------------------------


def oracle(n_total: int, n_stratum: int, withheld: int) -> Tuple[float, float]:
    """Enumerate every withheld subset and count. Shares nothing with the code."""
    stratum = set(range(n_stratum))
    draws = list(combinations(range(n_total), withheld))
    hits = [len(stratum & set(draw)) for draw in draws]
    return (
        sum(1 for h in hits if h == 0) / len(draws),
        sum(1 for h in hits if h == n_stratum) / len(draws),
    )


#: (N, n_s, W, which regime this covers). Every regime the specification names.
REGIMES: List[Tuple[int, int, int, str]] = [
    (12, 4, 2, "W < n_s"),
    (12, 4, 4, "W = n_s"),
    (12, 4, 6, "W > n_s"),
    (10, 3, 8, "W > N - n_s"),
    (9, 5, 5, "W = n_s, and W > N - n_s"),
    (11, 2, 3, "W > n_s, thin stratum"),
    (8, 6, 2, "W < n_s, fat stratum"),
    (10, 0, 3, "n_s = 0"),
    (7, 7, 7, "n_s = N, W = N"),
    (10, 10, 3, "n_s = N, W < N"),
]


@pytest.mark.parametrize("n_total,n_stratum,withheld,regime", REGIMES,
                         ids=[r[3] for r in REGIMES])
def test_zero_event_probabilities_match_exhaustive_enumeration(
    n_total, n_stratum, withheld, regime
):
    want_no_val, want_no_train = oracle(n_total, n_stratum, withheld)
    assert audit.p_no_validation_representation(n_total, n_stratum, withheld) == pytest.approx(
        want_no_val, abs=1e-12
    ), regime
    assert audit.p_no_training_representation(n_total, n_stratum, withheld) == pytest.approx(
        want_no_train, abs=1e-12
    ), regime


def test_the_two_events_are_distinct_where_they_can_be():
    """The bug that motivated these tests was two events treated as one.

    **The first version of this test picked a coincidence**, which is the same
    mistake in miniature. Besides the obvious ``n_s = 0`` and ``n_s = N``, the
    two probabilities also coincide whenever ``2W = N``, because
    ``C(N - n_s, W)`` and ``C(N - n_s, W - n_s)`` are then symmetric partners.
    The cases below avoid every such coincidence.
    """
    distinct = [(12, 4, 5), (12, 4, 3), (15, 5, 9), (20, 6, 4)]
    for n_total, n_stratum, withheld in distinct:
        assert 2 * withheld != n_total, "2W = N is a coincidence, not a test case"
        no_val = audit.p_no_validation_representation(n_total, n_stratum, withheld)
        no_train = audit.p_no_training_representation(n_total, n_stratum, withheld)
        assert no_val != no_train, (n_total, n_stratum, withheld)
        want_no_val, want_no_train = oracle(n_total, n_stratum, withheld)
        assert no_val == pytest.approx(want_no_val, abs=1e-12)
        assert no_train == pytest.approx(want_no_train, abs=1e-12)


def test_the_superseded_formula_disagrees_with_the_oracle():
    """Pins *why* the formula changed, so it cannot drift back.

    ``C(n_s, W) / C(N, W)`` is the probability the whole withheld set came from
    the stratum — the event ``X_s = W``. It is recomputed here rather than
    imported, because the implementation no longer contains it.
    """
    disagreements = 0
    for n_total, n_stratum, withheld, _ in REGIMES:
        superseded = (
            math.comb(n_stratum, withheld) / math.comb(n_total, withheld)
            if withheld <= n_stratum
            else 0.0
        )
        _, want_no_train = oracle(n_total, n_stratum, withheld)
        if abs(superseded - want_no_train) > 1e-12:
            disagreements += 1
    assert disagreements >= 5, (
        "the superseded formula must be shown to be wrong on several regimes; "
        "if it now agrees everywhere, this test has stopped testing anything"
    )


@pytest.mark.parametrize("n_total,n_stratum,withheld,regime", REGIMES,
                         ids=[r[3] for r in REGIMES])
def test_no_probability_call_raises_out_of_domain(n_total, n_stratum, withheld, regime):
    """The product form must be total, not merely correct where it is defined.

    ``math.comb`` raises on a negative argument, which is what the closed form
    would hit at ``W < n_s``. The telescoped product cannot: every factor is a
    non-negative integer over a positive one.
    """
    for value in (
        audit.p_no_validation_representation(n_total, n_stratum, withheld),
        audit.p_no_training_representation(n_total, n_stratum, withheld),
    ):
        assert 0.0 <= value <= 1.0, regime


# --------------------------------------------------------------------------
# W
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_eligible,fraction,expected",
    [
        (10576, 0.15, 1586),   # floor(1586.4 + 0.5)
        (100, 0.15, 15),
        (10, 0.15, 2),         # floor(1.5 + 0.5) = 2, round half up
        (10, 0.04, 1),         # clamped up off zero
        (10, 0.99, 9),         # clamped down off N
        (2, 0.5, 1),
    ],
)
def test_withheld_count_rounds_half_up_then_clamps(n_eligible, fraction, expected):
    assert audit.withheld_count(n_eligible, fraction) == expected


@pytest.mark.parametrize("n_eligible", [0, 1])
def test_a_universe_too_small_to_cut_is_refused_not_clamped(n_eligible):
    with pytest.raises(ValueError, match="two non-empty partitions"):
        audit.withheld_count(n_eligible, 0.15)


def test_withheld_count_never_empties_either_side():
    for n_eligible in range(2, 60):
        for fraction in (0.0, 0.001, 0.15, 0.5, 0.999, 1.0):
            withheld = audit.withheld_count(n_eligible, fraction)
            assert 1 <= withheld <= n_eligible - 1


# --------------------------------------------------------------------------
# Quotas
# --------------------------------------------------------------------------


def keys_for(labels):
    return [audit.band_sort_key(label, audit.PHENOTYPE_COUNT_BANDS) for label in labels]


def test_quotas_sum_exactly_to_withheld_and_respect_bucket_sizes():
    sizes = [7, 3, 11, 1, 0, 5]
    labels = ["0-1", "2-3", "4-5", "6-10", "11-20", "missing"]
    for withheld in range(0, sum(sizes)):
        quotas = audit.largest_remainder_quotas(sizes, withheld, keys_for(labels))
        assert sum(quotas) == withheld
        assert all(0 <= q <= s for q, s in zip(quotas, sizes))


def test_equal_remainders_break_on_canonical_bucket_order():
    """Two buckets of equal size tie exactly; the earlier band must win.

    The tie-break is over *buckets*, so it must not depend on anything about the
    diseases inside them.
    """
    sizes = [2, 2]
    labels = ["2-3", "4-5"]
    quotas = audit.largest_remainder_quotas(sizes, 1, keys_for(labels))
    assert quotas == [1, 0]


def test_the_missing_bucket_sorts_after_every_numeric_band():
    labels = ["0-1", "2-3", "51+", audit.MISSING_LABEL]
    ordered = sorted(
        labels, key=lambda label: audit.band_sort_key(label, audit.PHENOTYPE_COUNT_BANDS)
    )
    assert ordered[-1] == audit.MISSING_LABEL
    assert ordered[:3] == ["0-1", "2-3", "51+"]


def test_bucket_order_is_total_and_ascending():
    values = [0, 1, 2, 5, 12, 40, 900]
    bands = audit.bucket(values, audit.PHENOTYPE_COUNT_BANDS)
    lower_bounds = [
        audit.band_sort_key(label, audit.PHENOTYPE_COUNT_BANDS) for label, _ in bands
    ]
    assert lower_bounds == sorted(lower_bounds)
    assert sum(count for _, count in bands) == len(values)
    assert bands[-1][0] == audit.MISSING_LABEL


# --------------------------------------------------------------------------
# k, which is the phenotype count and nothing else
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_phenotypes,min_p,max_p,drop,expected",
    [
        (2, 2, 15, 0.3, 2),      # floor(1.4) = 1, raised to the minimum
        (10, 2, 15, 0.3, 7),
        (100, 2, 15, 0.3, 15),   # capped by max_phenotypes
        (3, 5, 15, 0.3, 3),      # capped by P itself
    ],
)
def test_retained_phenotype_count_mirrors_the_generator(
    n_phenotypes, min_p, max_p, drop, expected
):
    assert audit.retained_phenotype_count(n_phenotypes, min_p, max_p, drop) == expected


def test_retained_phenotype_count_matches_the_generator_on_a_sweep():
    """Restating the rule is the failure this guards against.

    The comparison is against ``_generate_samples``' own arithmetic, transcribed
    once here. If the generator's rule changes and this is not updated, capacity
    bands would describe a generator nobody runs.
    """
    for n_phen in range(2, 60):
        for drop in (0.0, 0.3, 0.5, 0.9):
            keep = max(2, int(n_phen * (1.0 - drop)))
            keep = min(keep, 15, n_phen)
            assert audit.retained_phenotype_count(n_phen, 2, 15, drop) == keep


# --------------------------------------------------------------------------
# Mean, SD
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_total,n_stratum,withheld,_regime", REGIMES)
def test_mean_and_sd_match_exhaustive_enumeration(n_total, n_stratum, withheld, _regime):
    hits = [
        len(set(range(n_stratum)) & set(draw))
        for draw in combinations(range(n_total), withheld)
    ]
    want_mean = sum(hits) / len(hits)
    want_sd = math.sqrt(sum((h - want_mean) ** 2 for h in hits) / len(hits))
    assert audit.hypergeometric_mean(n_total, n_stratum, withheld) == pytest.approx(
        want_mean, abs=1e-12
    )
    assert audit.hypergeometric_sd(n_total, n_stratum, withheld) == pytest.approx(
        want_sd, abs=1e-12
    )


def test_the_quota_and_the_expectation_are_not_the_same_object():
    """They differ by per-bucket rounding, and the report must show both."""
    sizes = [7, 3, 11]
    labels = ["0-1", "2-3", "4-5"]
    n_total, withheld = sum(sizes), 5
    quotas = audit.largest_remainder_quotas(sizes, withheld, keys_for(labels))
    means = [audit.hypergeometric_mean(n_total, size, withheld) for size in sizes]
    assert all(float(q).is_integer() for q in quotas)
    assert any(abs(q - m) > 1e-9 for q, m in zip(quotas, means))


# --------------------------------------------------------------------------
# End to end, against a real KnowledgeGraph
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_kg_path(tmp_path):
    """A KG small enough to reason about, written where the script expects one."""
    from src.core.types import DataSource, Node, NodeID, NodeType
    from src.kg.graph import Edge, EdgeType, KnowledgeGraph

    kg = KnowledgeGraph()
    phenotypes = [f"HP:000{i:04d}" for i in range(6)]
    for hp_id in phenotypes:
        kg.add_node(Node(id=NodeID(source=DataSource.HPO, local_id=hp_id),
                         node_type=NodeType.PHENOTYPE, name=hp_id))
    for gene in ("GENE_A", "GENE_B"):
        kg.add_node(Node(id=NodeID(source=DataSource.DISGENET, local_id=gene),
                         node_type=NodeType.GENE, name=gene))

    # Four diseases with 2, 3, 4 and 1 phenotypes: the last is below the
    # eligibility floor and must be excluded and counted as such.
    layout = {"MONDO:0000001": 2, "MONDO:0000002": 3, "MONDO:0000003": 4,
              "MONDO:0000004": 1}
    for mondo_id, n_phen in layout.items():
        kg.add_node(Node(id=NodeID(source=DataSource.MONDO, local_id=mondo_id),
                         node_type=NodeType.DISEASE, name=mondo_id))
        for hp_id in phenotypes[:n_phen]:
            kg.add_edge(Edge(
                source_id=NodeID(source=DataSource.HPO, local_id=hp_id),
                target_id=NodeID(source=DataSource.MONDO, local_id=mondo_id),
                edge_type=EdgeType.PHENOTYPE_OF_DISEASE,
            ))
        kg.add_edge(Edge(
            source_id=NodeID(source=DataSource.DISGENET, local_id="GENE_A"),
            target_id=NodeID(source=DataSource.MONDO, local_id=mondo_id),
            edge_type=EdgeType.GENE_ASSOCIATED_WITH_DISEASE,
        ))

    path = tmp_path / "kg.json"
    kg.save_json(str(path))
    return path


def test_report_is_aggregate_only_and_decides_nothing(tiny_kg_path):
    report = audit.build_report(
        tiny_kg_path,
        min_phenotypes=2, max_phenotypes=15, phenotype_drop_rate=0.3,
        train_budget=100, val_budget=20,
        fractions=[0.25, 0.5], samples_per_disease=[1, 5],
        relationship=audit.UNSTATED_RELATIONSHIP,
    )

    assert report["universe"]["disease_nodes"] == 4
    assert report["universe"]["eligible_diseases"] == 3
    assert report["universe"]["excluded_diseases"] == 1

    # The assumptions are echoed and labelled as assumptions, not observations.
    assert report["assumptions"]["min_phenotypes"] == 2
    assert report["assumptions"]["current_train_budget"] == 100
    assert "not recovered from the workspace" in report["assumptions"]["note"]

    for row in report["sensitivity"]:
        withheld = row["withheld_diseases"]
        assert 1 <= withheld <= report["universe"]["eligible_diseases"] - 1
        for bands in row["strata"].values():
            assert sum(band["quota"] for band in bands) == withheld
            assert all(0 <= band["quota"] <= band["diseases"] for band in bands)


def test_no_identifier_reaches_the_report(tiny_kg_path):
    """BACKLOG §5.2. The tiny KG's ids are distinctive so a leak would show."""
    report = audit.build_report(
        tiny_kg_path,
        min_phenotypes=2, max_phenotypes=15, phenotype_drop_rate=0.3,
        train_budget=100, val_budget=20,
        fractions=[0.25], samples_per_disease=[1],
        relationship=audit.UNSTATED_RELATIONSHIP,
    )
    rendered = json.dumps(report)
    for identifier in ("MONDO:", "HP:000", "GENE_A", "GENE_B", str(tiny_kg_path)):
        assert identifier not in rendered, identifier


def test_the_report_serialises_without_nan_or_infinity(tiny_kg_path):
    report = audit.build_report(
        tiny_kg_path,
        min_phenotypes=2, max_phenotypes=15, phenotype_drop_rate=0.3,
        train_budget=100, val_budget=20,
        fractions=list(audit.DEFAULT_FRACTIONS),
        samples_per_disease=list(audit.DEFAULT_SAMPLES_PER_DISEASE),
        relationship=audit.UNSTATED_RELATIONSHIP,
    )
    json.dumps(report, allow_nan=False, sort_keys=True)


def test_allocating_more_than_the_buckets_hold_is_refused():
    """The floor pass has no cap, so its precondition has to be enforced.

    ``withheld <= total`` is what makes every floored quota fit its bucket. An
    earlier version carried a ``min(..., size)`` there instead, which was dead
    code under that precondition and therefore untestable — removing it broke
    nothing, which is how it was found.
    """
    with pytest.raises(ValueError, match="cannot allocate"):
        audit.largest_remainder_quotas([2, 3], 6, keys_for(["2-3", "4-5"]))


def test_no_quota_ever_exceeds_its_bucket():
    """Codex acceptance criterion, kept as a property of the algorithm.

    No guard enforces this — under withheld <= total the largest-remainder
    method is naturally bounded, and both guards that once claimed to enforce it
    were dead code. The property is still worth pinning: it would fail under a
    *different* allocation algorithm, which is the change this guards against.
    """
    sizes = [1, 1, 20]
    labels = ["2-3", "4-5", "6-10"]
    for withheld in range(0, sum(sizes) + 1):
        quotas = audit.largest_remainder_quotas(sizes, withheld, keys_for(labels))
        assert sum(quotas) == withheld
        assert all(q <= s for q, s in zip(quotas, sizes))
