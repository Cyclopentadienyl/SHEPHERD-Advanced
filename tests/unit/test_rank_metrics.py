"""
`RankingMetrics.compute_from_ranks` — rank-based metrics, and the input contract
that keeps a fabricated number out of a report.
==============================================================================
The prediction-list entry points take truncated top-k lists, so a ground truth
outside the list is indistinguishable from one ranked last. That is why
`scripts/evaluate_model.py` could not report a mean rank at all, and reported a
fabricated `0.0` until commit `7dab728` removed it.

This entry point takes ranks directly. The tests below cover the arithmetic
against an independent computation, and the input contract — because the
failure this function exists to prevent is a plausible-looking number computed
from input that had no business producing one.

Ranking policy is not tested here because it is not here: score ordering, tie
breaking and missing-ground-truth handling belong to the measurement layer.
"""
import numpy as np
import pytest

from src.utils.metrics import RankingMetrics


@pytest.fixture
def metrics():
    return RankingMetrics()


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------
def test_matches_an_independent_computation(metrics):
    ranks = [1, 3, 7, 2, 50]

    got = metrics.compute_from_ranks(ranks, k_values=[1, 5, 10])

    assert got["mean_rank"] == pytest.approx(sum(ranks) / len(ranks))
    assert got["mrr"] == pytest.approx(sum(1 / r for r in ranks) / len(ranks))
    assert got["hits@1"] == pytest.approx(1 / 5)
    assert got["hits@5"] == pytest.approx(3 / 5)
    assert got["hits@10"] == pytest.approx(4 / 5)


def test_all_metrics_share_one_denominator(metrics):
    """Hits@k, MRR and mean rank must be over the same cohort. A per-metric
    denominator is how an evaluation quietly starts comparing different
    populations."""
    ranks = [1, 2, 3, 4]

    got = metrics.compute_from_ranks(ranks, k_values=[1, 2, 3, 4, 100])

    assert got["hits@100"] == 1.0
    assert got["hits@1"] == pytest.approx(0.25)
    assert got["mrr"] == pytest.approx((1 + 0.5 + 1 / 3 + 0.25) / 4)


def test_a_rank_beyond_every_k_still_counts_in_the_denominator(metrics):
    """The property a truncated list cannot express.

    Both cohorts have the same Hits@1, so a truncated report would call them
    identical. Mean rank and MRR separate them, because the distant rank is still
    in the denominator rather than having fallen off the end of a list.
    """
    near = metrics.compute_from_ranks([1, 2], k_values=[1])
    far = metrics.compute_from_ranks([1, 27990], k_values=[1])

    assert near["hits@1"] == far["hits@1"] == pytest.approx(0.5)
    assert far["mean_rank"] > near["mean_rank"]
    assert far["mrr"] < near["mrr"]


def test_untruncated_k_is_permitted(metrics):
    """k may exceed any list a caller would carry — that is the point."""
    got = metrics.compute_from_ranks([1, 500, 27990], k_values=[1, 100, 27990])

    assert got["hits@27990"] == 1.0
    assert got["hits@100"] == pytest.approx(1 / 3)


def test_defaults_are_used_when_k_values_are_omitted(metrics):
    got = metrics.compute_from_ranks([1, 2, 3])

    assert set(got) == {"mean_rank", "mrr"} | {
        f"hits@{k}" for k in RankingMetrics.default_k_values
    }


def test_numpy_integer_ranks_are_accepted(metrics):
    """Ranks arrive from tensor and array code, so numpy integers are ordinary
    input, not an exotic case."""
    got = metrics.compute_from_ranks(np.array([1, 4], dtype=np.int64), k_values=[1])

    assert got["mean_rank"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------
def test_empty_input_raises_rather_than_reporting_zero(metrics):
    """The defect this function is designed against. An empty cohort has no mean
    rank; `0.0` would be a fabricated value that reads as a perfect score."""
    with pytest.raises(ValueError):
        metrics.compute_from_ranks([])


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_non_positive_ranks_are_rejected(metrics, bad):
    """Ranks are 1-based. A 0 would make MRR infinite."""
    with pytest.raises(ValueError):
        metrics.compute_from_ranks([1, bad])


@pytest.mark.parametrize("bad", [True, False])
def test_boolean_ranks_are_rejected(metrics, bad):
    """`bool` is a subclass of `int`, so `isinstance(True, int)` is True and a
    boolean would otherwise be accepted as rank 1 — a silent wrong answer rather
    than an error."""
    with pytest.raises(ValueError):
        metrics.compute_from_ranks([1, bad])


@pytest.mark.parametrize("bad", [1.5, "3", None])
def test_non_integer_ranks_are_rejected(metrics, bad):
    with pytest.raises(ValueError):
        metrics.compute_from_ranks([1, bad])


@pytest.mark.parametrize("bad_k", [0, -5, True, 2.5])
def test_invalid_k_values_are_rejected(metrics, bad_k):
    with pytest.raises(ValueError):
        metrics.compute_from_ranks([1, 2], k_values=[bad_k])
