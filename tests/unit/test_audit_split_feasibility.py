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
# k, which is the phenotype count and nothing else — and lives in production
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_phenotypes,min_p,max_p,drop,expected",
    [
        (2, 2, 15, 0.3, 2),      # floor(1.4) = 1, raised to the minimum
        (10, 2, 15, 0.3, 7),
        (100, 2, 15, 0.3, 15),   # capped by max_phenotypes
        (3, 5, 15, 0.3, 3),      # capped by P itself
        (10, 2, 15, 0.0, 10),    # nothing dropped
        (10, 2, 15, 1.0, 2),     # everything droppable, floored at the minimum
    ],
)
def test_retained_phenotype_count_boundaries(n_phenotypes, min_p, max_p, drop, expected):
    from src.kg import retained_phenotype_count

    assert retained_phenotype_count(n_phenotypes, min_p, max_p, drop) == expected


def test_the_audit_uses_the_production_rule_rather_than_a_copy():
    """No transcribed formula here, and none in the audit.

    A copied rule would let production generation change while the audit and its
    tests stayed green, reporting capacity for a generator that no longer exists.
    The check is identity, not equality of two transcriptions.
    """
    import src.kg as kg_package

    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "retained_phenotype_count(" in source
    assert "1.0 - " not in source.split("def build_report")[1].split("strata =")[0], (
        "the audit must not recompute the retained-phenotype rule inline"
    )
    assert hasattr(kg_package, "retained_phenotype_count")


def test_generation_actually_calls_the_shared_rule(monkeypatch, tiny_kg_path):
    """Pins that ``_generate_samples`` routes through the helper.

    Without this, extracting the helper and leaving the inline arithmetic behind
    would look identical to having done the work.
    """
    from src.kg import graph as graph_module
    from src.kg import sample_generator

    calls = []
    original = sample_generator.retained_phenotype_count

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(sample_generator, "retained_phenotype_count", spy)
    kg = graph_module.KnowledgeGraph.load_json(str(tiny_kg_path))
    sample_generator.generate_training_samples(kg, num_train=4, num_val=0, min_phenotypes=2)
    assert calls, "generation did not go through the shared retained-phenotype rule"


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
        tiny_kg_path, settings(fractions=[0.25, 0.5], samples_per_disease=[1, 5]),
        audit.UNSTATED_RELATIONSHIP,
    )

    assert report["universe"]["disease_nodes"] == 4
    assert report["universe"]["eligible_diseases"] == 3
    assert report["universe"]["exclusions"]["below_min_phenotypes"] == 1
    assert "no mapping success rate is reported" in report["universe"]["exclusions"]["note"]

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
        tiny_kg_path, settings(fractions=[0.25], samples_per_disease=[1]),
        audit.UNSTATED_RELATIONSHIP,
    )
    rendered = json.dumps(report)
    for identifier in ("MONDO:", "HP:000", "GENE_A", "GENE_B", str(tiny_kg_path)):
        assert identifier not in rendered, identifier


def test_the_report_serialises_without_nan_or_infinity(tiny_kg_path):
    report = audit.build_report(
        tiny_kg_path, settings(), audit.UNSTATED_RELATIONSHIP,
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


def settings(**overrides):
    """Validated defaults for the end-to-end tests, overridable per case."""
    base = dict(
        min_phenotypes=2, max_phenotypes=15, phenotype_drop_rate=0.3,
        train_budget=100, val_budget=20,
        fractions=list(audit.DEFAULT_FRACTIONS),
        samples_per_disease=list(audit.DEFAULT_SAMPLES_PER_DISEASE),
    )
    base.update(overrides)
    return audit.validate_settings(**base)


# --------------------------------------------------------------------------
# The input boundary
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("min_phenotypes", 0, "min_phenotypes must be >= 1"),
        ("min_phenotypes", 2.0, "must be an integer"),
        ("min_phenotypes", True, "must be an integer"),
        ("max_phenotypes", 1, "max_phenotypes must be >= 2"),
        ("phenotype_drop_rate", -0.1, "finite and in"),
        ("phenotype_drop_rate", 1.5, "finite and in"),
        ("phenotype_drop_rate", float("nan"), "finite and in"),
        ("phenotype_drop_rate", float("inf"), "finite and in"),
        ("phenotype_drop_rate", "0.3", "must be a number"),
        ("train_budget", -1, "train_budget must be >= 0"),
        ("val_budget", -1, "val_budget must be >= 0"),
        ("fractions", [], "at least one fraction"),
        ("fractions", [0.0], "strictly between 0 and 1"),
        ("fractions", [1.0], "strictly between 0 and 1"),
        ("fractions", [-0.2], "strictly between 0 and 1"),
        ("fractions", [1.4], "strictly between 0 and 1"),
        ("fractions", [float("nan")], "strictly between 0 and 1"),
        ("fractions", [float("inf")], "strictly between 0 and 1"),
        ("samples_per_disease", [], "at least one samples_per_disease"),
        ("samples_per_disease", [0], "samples_per_disease must be >= 1"),
        ("samples_per_disease", [-3], "samples_per_disease must be >= 1"),
        ("samples_per_disease", [1.5], "must be an integer"),
        ("samples_per_disease", [True], "must be an integer"),
    ],
)
def test_invalid_assumptions_are_refused_at_the_api(field, value, message):
    """Refused at the API, not only at argparse — the audit is importable."""
    with pytest.raises(ValueError, match=message):
        settings(**{field: value})


def test_repeats_are_deduplicated_deterministically():
    got = settings(fractions=[0.15, 0.05, 0.15], samples_per_disease=[5, 5, 1])
    assert got.fractions == (0.15, 0.05)
    assert got.samples_per_disease == (5, 1)


def test_validated_settings_are_frozen():
    with pytest.raises(AttributeError):
        settings().min_phenotypes = 99


def test_a_zero_samples_per_disease_can_never_report_a_sufficient_budget():
    """The defect this boundary exists for: 0 required is trivially affordable."""
    with pytest.raises(ValueError):
        settings(samples_per_disease=[0])


# --------------------------------------------------------------------------
# Propagated support is not direct KG degree
# --------------------------------------------------------------------------


def test_profile_support_is_propagated_and_is_not_claimed_to_be_kg_degree(tmp_path):
    """One disease, one direct gene, and that gene carries many phenotypes.

    The disease has **one** directly incident neighbour. Its profile support is
    far larger, because ``_build_disease_profiles`` propagates the gene's
    phenotypes onto the disease. The report must band the propagated figure and
    must not call it KG degree.
    """
    from src.core.types import DataSource, Node, NodeID, NodeType
    from src.kg.graph import Edge, EdgeType, KnowledgeGraph

    kg = KnowledgeGraph()
    gene = NodeID(source=DataSource.DISGENET, local_id="HUB_GENE")
    kg.add_node(Node(id=gene, node_type=NodeType.GENE, name="HUB_GENE"))
    for i in range(12):
        hp = NodeID(source=DataSource.HPO, local_id=f"HP:009{i:04d}")
        kg.add_node(Node(id=hp, node_type=NodeType.PHENOTYPE, name=str(hp.local_id)))
        kg.add_edge(Edge(source_id=gene, target_id=hp,
                         edge_type=EdgeType.GENE_HAS_PHENOTYPE))

    # Two diseases so the universe can be cut; both borrow the hub gene's
    # phenotypes and neither has a phenotype edge of its own.
    for mondo in ("MONDO:0009001", "MONDO:0009002"):
        disease = NodeID(source=DataSource.MONDO, local_id=mondo)
        kg.add_node(Node(id=disease, node_type=NodeType.DISEASE, name=mondo))
        kg.add_edge(Edge(source_id=gene, target_id=disease,
                         edge_type=EdgeType.GENE_ASSOCIATED_WITH_DISEASE))

    path = tmp_path / "kg.json"
    kg.save_json(str(path))

    from src.kg import build_eligible_disease_profiles

    eligible = build_eligible_disease_profiles(
        KnowledgeGraph.load_json(str(path)), min_phenotypes=2
    )
    assert len(eligible) == 2
    for _, profile in eligible:
        # One direct neighbour, twelve propagated phenotypes.
        assert len(profile["gene_ids"]) == 1
        assert len(profile["phenotype_ids"]) == 12

    report = audit.build_report(
        path, settings(fractions=[0.5], samples_per_disease=[1]),
        audit.UNSTATED_RELATIONSHIP,
    )
    support = {row["band"]: row["diseases"] for row in
               report["distributions"]["profile_support_size"]}
    assert support["11-25"] == 2, support   # 12 phenotypes + 1 gene = 13

    definition = report["axis_definitions"]["profile_support_size"]
    assert "Not the disease node's direct KG degree" in definition
    assert "profile_degree" not in json.dumps(report)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def _argv(kg_path, out):
    return ["--kg-path", str(kg_path), "--output", str(out),
            "--train-budget", "100", "--val-budget", "20"]


def test_main_writes_finite_valid_json(tiny_kg_path, tmp_path):
    out = tmp_path / "report.json"
    assert audit.main(_argv(tiny_kg_path, out)) == 0
    report = json.loads(out.read_text())
    assert report["audit"] == "split_feasibility"
    json.dumps(report, allow_nan=False)


def test_main_refuses_an_existing_output_and_preserves_its_bytes(tiny_kg_path, tmp_path):
    out = tmp_path / "report.json"
    out.write_bytes(b'{"existing": true}')
    with pytest.raises(SystemExit, match="Pass --overwrite"):
        audit.main(_argv(tiny_kg_path, out))
    assert out.read_bytes() == b'{"existing": true}'


def test_main_overwrites_when_told_to(tiny_kg_path, tmp_path):
    out = tmp_path / "report.json"
    out.write_bytes(b'{"existing": true}')
    assert audit.main(_argv(tiny_kg_path, out) + ["--overwrite"]) == 0
    assert json.loads(out.read_text())["audit"] == "split_feasibility"


def test_main_rejects_invalid_assumptions_before_writing(tiny_kg_path, tmp_path):
    out = tmp_path / "report.json"
    with pytest.raises(SystemExit, match="strictly between 0 and 1"):
        audit.main(_argv(tiny_kg_path, out) + ["--fractions", "1.5"])
    assert not out.exists(), "nothing may be written when the inputs are refused"


def test_an_unknown_deployment_relationship_is_rejected(tiny_kg_path, tmp_path):
    out = tmp_path / "report.json"
    with pytest.raises(SystemExit):
        audit.parse_args(
            _argv(tiny_kg_path, out) + ["--deployment-relationship", "definitely-not-a-choice"]
        )


# --------------------------------------------------------------------------
# Validation happens before deduplication, and at every boundary
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values", [[1, True], [True, 1], [1, 1.0], [1.0, 1], [1, 1, True]],
    ids=["int-then-bool", "bool-then-int", "int-then-float", "float-then-int",
         "repeat-then-bool"],
)
def test_an_illegal_element_cannot_hide_behind_a_legal_duplicate(values):
    """``True == 1 == 1.0`` with matching hashes, so order used to decide this.

    Deduplicating first collapsed ``[1, True]`` to ``[1]`` and never checked the
    ``True``, while ``[True, 1]`` collapsed to ``[True]`` and was refused. Same
    multiset, opposite outcomes.
    """
    with pytest.raises(ValueError, match="must be an integer"):
        settings(samples_per_disease=values)


def test_legal_repeats_still_deduplicate():
    got = settings(samples_per_disease=[5, 5, 1], fractions=[0.15, 0.15, 0.05])
    assert got.samples_per_disease == (5, 1)
    assert got.fractions == (0.15, 0.05)


def test_validation_is_idempotent():
    """``build_report`` revalidates, so a second pass must not change anything."""
    once = settings()
    twice = audit.validate_settings(**once._asdict())
    assert twice == once


def test_build_report_refuses_invalid_settings_before_loading_the_graph(monkeypatch):
    """Immutability is not validity: the constructor is public and unchecked."""
    from src.kg import graph as graph_module

    def explode(*_args, **_kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("the knowledge graph must not be loaded")

    monkeypatch.setattr(graph_module.KnowledgeGraph, "load_json", staticmethod(explode))

    hand_built = audit.AuditSettings(
        min_phenotypes=-5, max_phenotypes=1, phenotype_drop_rate=9.9,
        train_budget=-1, val_budget=-1, fractions=(5.0,), samples_per_disease=(0,),
    )
    with pytest.raises(ValueError, match="min_phenotypes must be >= 1"):
        audit.build_report(Path("unused.json"), hand_built, audit.UNSTATED_RELATIONSHIP)


@pytest.mark.parametrize(
    "relationship",
    ["nozomi", "chung@nozomi", "/home/chung/workspaces", "", "identical_sibling"],
)
def test_build_report_refuses_a_relationship_outside_the_vocabulary(
    monkeypatch, relationship
):
    """The privacy schema forbids host and operator names; argparse is not enough.

    ``build_report`` writes this string verbatim into the artifact, and a
    programmatic caller never passes through ``parse_args``.
    """
    from src.kg import graph as graph_module

    def explode(*_args, **_kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("the knowledge graph must not be loaded")

    monkeypatch.setattr(graph_module.KnowledgeGraph, "load_json", staticmethod(explode))

    with pytest.raises(ValueError, match="deployment_relationship must be one of"):
        audit.build_report(Path("unused.json"), settings(), relationship)


def test_every_permitted_relationship_is_accepted(tiny_kg_path):
    for relationship in audit.DEPLOYMENT_RELATIONSHIPS:
        report = audit.build_report(
            tiny_kg_path, settings(fractions=[0.5], samples_per_disease=[1]),
            relationship,
        )
        assert report["deployment_relationship"] == relationship
