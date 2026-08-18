"""
The two ranking streams, and the identifier translation between them.
=====================================================================
Three failure modes are worth testing here, and they are different:

  - **Direction.** Subgraph-local and global disease ids are both integers of a
    plausible magnitude, so translating in the wrong direction produces valid-
    looking ids and therefore valid-looking, wrong metrics. Nothing downstream
    can detect it.
  - **Determinism.** `canonical_ranking` must depend on the scores and the global
    ids and on nothing else. If the order candidates arrive in can change the
    result, then a number from Mode A cannot be compared with one from Mode C.
  - **Absence.** A ground truth outside the candidate set has no rank. It must
    stay `None` rather than becoming a large integer that flows into a mean.
"""
import pytest

torch = pytest.importorskip("torch")

from src.evaluation.measurement import (  # noqa: E402
    canonical_ranking,
    legacy_ranking,
    ranks_of_truth,
    to_global_disease_ids,
)


# ---------------------------------------------------------------------------
# Identifier translation
# ---------------------------------------------------------------------------
def test_local_ids_translate_through_original_indices():
    """`original_indices["disease"]` is indexed by LOCAL position and holds the
    GLOBAL index, so it is already the direction needed."""
    original = torch.tensor([12, 47, 300, 9001])  # local 0..3 -> these globals

    got = to_global_disease_ids(original, torch.tensor([2, 0, 3]))

    assert got.tolist() == [300, 12, 9001]


def test_translation_is_not_the_inverse_direction():
    """The guard against the error this function exists to prevent.

    `node_mapping` is the global->local dict. Using it where `original_indices`
    belongs, or vice versa, yields ids in range and completely wrong.
    """
    original = torch.tensor([5, 11, 40])

    assert to_global_disease_ids(original, torch.tensor([0])).tolist() == [5]
    assert to_global_disease_ids(original, torch.tensor([2])).tolist() == [40]


@pytest.mark.parametrize("bad_local", [[3], [99], [-1]])
def test_out_of_range_local_ids_are_rejected(bad_local):
    with pytest.raises(ValueError):
        to_global_disease_ids(torch.tensor([5, 11, 40]), torch.tensor(bad_local))


def test_translation_rejects_a_non_vector_table():
    with pytest.raises(ValueError):
        to_global_disease_ids(torch.tensor([[1, 2], [3, 4]]), torch.tensor([0]))


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
    """Determinism across modes. Mode A and Mode C will present the same
    candidates in different orders; the ranking must not notice."""
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
# Legacy ranking
# ---------------------------------------------------------------------------
def test_legacy_reproduces_tensor_sort_on_local_columns():
    """It must be `Tensor.sort` on the local columns — that is what
    `scripts/evaluate_model.py:295` calls — translated afterwards."""
    scores = torch.tensor([[0.1, 0.9, 0.5]])
    ids = torch.tensor([70, 80, 90])

    _, expected_local = scores.sort(dim=-1, descending=True)

    assert legacy_ranking(scores, ids).tolist() == ids[expected_local].tolist()


def test_the_two_streams_agree_when_no_scores_tie():
    """They differ only at ties. Anywhere else, a disagreement would mean one of
    them is simply wrong."""
    torch.manual_seed(0)
    scores = torch.randn(4, 9)
    ids = torch.tensor([31, 4, 77, 12, 90, 5, 68, 23, 51])

    assert canonical_ranking(scores, ids).tolist() == legacy_ranking(scores, ids).tolist()


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
