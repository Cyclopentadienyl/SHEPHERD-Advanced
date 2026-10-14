"""
The canonical ranking and the rank of the truth in it.
======================================================
Two failure modes are worth testing here, and they are different:

  - **Determinism.** `canonical_ranking` must depend on the scores and the global
    ids and on nothing else. If the order candidates arrive in, or how a cohort
    is chunked, could change the result, two runs over the same scores could
    report different ranks.
  - **Absence.** A ground truth outside the candidate set has no rank. It must
    stay `None` rather than becoming a large integer that flows into a mean.
"""
import pytest

torch = pytest.importorskip("torch")

from src.evaluation.measurement import (  # noqa: E402
    canonical_ranking,
    ranks_of_truth,
)


# ---------------------------------------------------------------------------
# Canonical ranking
# ---------------------------------------------------------------------------
def test_canonical_ranks_by_score_descending():
    scores = torch.tensor([[0.1, 0.9, 0.5]])
    ids = torch.tensor([70, 80, 90])

    assert canonical_ranking(scores, ids).tolist() == [[80, 90, 70]]


def test_canonical_breaks_ties_by_ascending_global_id():
    """The whole reason this stream exists."""
    scores = torch.tensor([[0.5, 0.5, 0.5]])
    ids = torch.tensor([90, 70, 80])

    assert canonical_ranking(scores, ids).tolist() == [[70, 80, 90]]


def test_canonical_is_independent_of_candidate_input_order():
    """The same (id, score) pairs in a different column order rank the same: the
    tie rule is the global id, not the position a candidate arrived in."""
    ids_a = torch.tensor([90, 70, 80, 60])
    scores_a = torch.tensor([[0.5, 0.5, 0.2, 0.5]])
    # same (id, score) pairs, shuffled into a different column order
    ids_b = torch.tensor([60, 80, 90, 70])
    scores_b = torch.tensor([[0.5, 0.2, 0.5, 0.5]])

    assert canonical_ranking(scores_a, ids_a).tolist() == canonical_ranking(scores_b, ids_b).tolist()
    assert canonical_ranking(scores_a, ids_a).tolist() == [[60, 70, 90, 80]]


def test_canonical_handles_a_batch():
    scores = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
    ids = torch.tensor([11, 22])

    assert canonical_ranking(scores, ids).tolist() == [[22, 11], [11, 22]]


def test_canonical_rejects_duplicate_global_ids():
    """A duplicated id makes the tie rule ambiguous and would silently pick one."""
    with pytest.raises(ValueError, match="duplicate"):
        canonical_ranking(torch.tensor([[0.1, 0.2]]), torch.tensor([7, 7]))


@pytest.mark.parametrize(
    "scores, ids",
    [
        (torch.tensor([0.1, 0.2]), torch.tensor([1, 2])),      # scores not 2-D
        (torch.tensor([[0.1, 0.2]]), torch.tensor([[1, 2]])),  # ids not 1-D
        (torch.tensor([[0.1, 0.2]]), torch.tensor([1, 2, 3])), # count mismatch
    ],
)
def test_canonical_rejects_malformed_input(scores, ids):
    with pytest.raises(ValueError):
        canonical_ranking(scores, ids)


# ---------------------------------------------------------------------------
# Input contracts
# ---------------------------------------------------------------------------
def test_canonical_rejects_non_integer_global_ids():
    with pytest.raises(ValueError, match="integer"):
        canonical_ranking(torch.tensor([[0.1, 0.2]]), torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), float("-inf")])
def test_canonical_rejects_non_finite_scores(bad_score):
    """A NaN sorts unpredictably and would become a plausible rank, then a
    plausible mean rank and MRR."""
    scores = torch.tensor([[0.5, bad_score, 0.1]])

    with pytest.raises(ValueError, match="NaN or infinity"):
        canonical_ranking(scores, torch.tensor([70, 80, 90]))


def test_rank_extraction_rejects_non_integer_ids():
    with pytest.raises(ValueError, match="integer"):
        ranks_of_truth(torch.tensor([[1.0, 2.0]]), torch.tensor([1]))


@pytest.mark.parametrize("bad", [torch.float32, torch.float64, torch.bool])
def test_ranking_identifiers_must_be_integers(bad):
    """`Tensor.long()` turns 1.7 into 1 and True into 1, so a float or boolean id
    wired in by mistake becomes a plausible index rather than an error. Both
    helpers Mode C ranks with refuse it, on every identifier they take."""
    ids = torch.tensor([0, 1])

    with pytest.raises(ValueError, match="integer"):
        canonical_ranking(torch.tensor([[0.1, 0.2]]), ids.to(bad))
    with pytest.raises(ValueError, match="integer"):
        ranks_of_truth(ids.unsqueeze(0).to(bad), torch.tensor([1]))
    with pytest.raises(ValueError, match="integer"):
        ranks_of_truth(ids.unsqueeze(0), torch.tensor([1]).to(bad))


# ---------------------------------------------------------------------------
# Rank extraction
# ---------------------------------------------------------------------------
def test_rank_is_one_based():
    ranked = torch.tensor([[80, 90, 70]])

    assert ranks_of_truth(ranked, torch.tensor([80])) == [1]
    assert ranks_of_truth(ranked, torch.tensor([70])) == [3]


def test_an_absent_truth_has_no_rank():
    """`None`, not a large integer. A sentinel rank would flow into a mean as
    though it had been measured."""
    ranked = torch.tensor([[80, 90, 70]])

    assert ranks_of_truth(ranked, torch.tensor([12345])) == [None]


def test_absent_and_present_truths_in_one_batch():
    ranked = torch.tensor([[80, 90, 70], [11, 22, 33]])

    assert ranks_of_truth(ranked, torch.tensor([90, 999])) == [2, None]


def test_rank_extraction_rejects_a_truth_count_mismatch():
    with pytest.raises(ValueError):
        ranks_of_truth(torch.tensor([[1, 2], [3, 4]]), torch.tensor([1]))


def test_ranks_feed_the_metrics_entry_point():
    """The seam this module was shaped around: filter the absences out, and what
    remains is exactly what `compute_from_ranks` accepts."""
    from src.utils.metrics import RankingMetrics

    ranked = torch.tensor([[80, 90, 70], [11, 22, 33], [5, 6, 7]])
    ranks = ranks_of_truth(ranked, torch.tensor([70, 999, 5]))
    assert ranks == [3, None, 1]

    present = [r for r in ranks if r is not None]
    metrics = RankingMetrics().compute_from_ranks(present, k_values=[1, 3])

    assert metrics["mean_rank"] == pytest.approx(2.0)
    assert metrics["hits@1"] == pytest.approx(0.5)
    assert metrics["hits@3"] == pytest.approx(1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_ranking_on_cuda():
    device = torch.device("cuda")
    scores = torch.tensor([[0.5, 0.5, 0.2]], device=device)
    ids = torch.tensor([90, 70, 80], device=device)

    ranked = canonical_ranking(scores, ids)

    assert ranked.device.type == "cuda"
    assert ranked.cpu().tolist() == [[70, 90, 80]]
    assert ranks_of_truth(ranked, torch.tensor([90], device=device)) == [2]
