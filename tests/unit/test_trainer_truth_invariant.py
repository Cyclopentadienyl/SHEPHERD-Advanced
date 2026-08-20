"""The trainer must refuse a malformed disease truth, not correct it.

A ground-truth disease id outside `[0, n_subgraph_disease_rows)` means the
subgraph seeding, the id remap, the candidate alignment or the batch wiring is
wrong. Scoring it as a rank miss would turn a data-pipeline failure into
apparent model error and contaminate the loss, `val_mrr`, early stopping and
checkpoint selection. **The contract is refuse.**

This file exists in two halves, and the order matters:

  1. **Characterization** — the complete trainer path already refuses, through
     `DiagnosisLoss`. Frozen here *before* the guard below was added, so the
     guard can be shown to preserve behaviour rather than create it.
  2. **The explicit guard** — `_compute_model_outputs` now checks the range
     itself. Same outcome, named error, and raised before the silent clamp that
     used to stand in that position.

`_compute_model_outputs` touches no attribute of `self`, so it is exercised as
an unbound call rather than behind a constructed `Trainer` with an optimizer, a
dataloader and a device.

Module: tests/unit/test_trainer_truth_invariant.py
"""
from __future__ import annotations

import pytest
import torch

from src.training.loss_functions import DiagnosisLoss, LossConfig, MultiTaskLoss
from src.training.trainer import Trainer

N_DISEASES = 5
N_PHENOTYPES = 7
HIDDEN = 4


def make_batch(disease_ids):
    """One batch in the shape `diagnosis_collate_fn` produces after remapping.

    Phenotype ids carry a `-1` pad on purpose: that padding is legitimate, the
    mask discards it, and the clamp guarding it is **not** what this file is
    about. Only the disease truth is under test.
    """
    return {
        "phenotype_ids": torch.tensor([[0, 1, -1], [2, 3, -1]]),
        "phenotype_mask": torch.tensor([[True, True, False], [True, True, False]]),
        "disease_ids": torch.tensor(disease_ids),
        "patient_ids": ["p0", "p1"],
    }


def node_embeddings():
    torch.manual_seed(0)
    return {
        "disease": torch.randn(N_DISEASES, HIDDEN),
        "phenotype": torch.randn(N_PHENOTYPES, HIDDEN),
    }


def compute(disease_ids):
    return Trainer._compute_model_outputs(
        None, node_embeddings(), make_batch(disease_ids), {}, {}
    )


# ---------------------------------------------------------------------------
# 1. Characterization — what the complete path did before the guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("bad", [[0, -1], [0, N_DISEASES], [0, 99]])
def test_diagnosis_loss_refuses_a_malformed_truth(label_smoothing, bad):
    """`DiagnosisLoss` is what made the trainer refuse, on both its branches.

    Without label smoothing the refusal comes from `F.cross_entropy`; with it,
    from the `scatter_` inside `_label_smoothed_ce`. Both are asserted because a
    configuration change must not quietly move the path onto an unchecked one.
    """
    scores = torch.randn(2, N_DISEASES)
    loss = DiagnosisLoss(label_smoothing=label_smoothing)
    with pytest.raises((IndexError, RuntimeError)):
        loss(scores, torch.tensor(bad))


def test_the_loss_runs_before_predictions_are_collected():
    """Why the metric fragment is unreachable with a malformed truth.

    `_validate` calls `self.loss_fn(...)` (`trainer.py:640`) before collecting
    predictions (`:646-657`). `mean_reciprocal_rank` *would* score a `-1` truth
    as `0.0` in isolation — but the loss raises first, so the trainer never
    records that number. This asserts the isolated behaviour so the distinction
    stays visible: the primitive is permissive, the path is not.
    """
    from src.utils.metrics import RankingMetrics

    assert RankingMetrics().mean_reciprocal_rank([["0", "1", "2"]], ["-1"]) == 0.0

    outputs = compute([0, 1])
    total, _ = MultiTaskLoss(LossConfig())(make_batch([0, 1]), outputs)
    assert torch.isfinite(total), "a legal batch must still produce a finite loss"


# ---------------------------------------------------------------------------
# 2. The explicit guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad,offender", [([0, -1], -1), ([0, N_DISEASES], 5), ([99, 0], 99)])
def test_compute_model_outputs_refuses_a_malformed_truth(bad, offender):
    """Named, domain-specific, and raised before any gather."""
    with pytest.raises(ValueError, match="disease truth"):
        compute(bad)

    with pytest.raises(ValueError, match=str(offender)):
        compute(bad)


def test_the_error_names_the_candidate_count():
    """An id of 5 against 5 rows is a different bug from an id of 5 against 500."""
    with pytest.raises(ValueError, match=str(N_DISEASES)):
        compute([0, N_DISEASES])


def test_a_legal_truth_is_unchanged_by_the_guard():
    """Behaviour-preserving for valid input — the whole justification for adding it.

    The gather must use the **original** ids, not a clamped copy, so this pins
    the embeddings to the rows the ids name.
    """
    embeddings = node_embeddings()
    outputs = Trainer._compute_model_outputs(
        None, embeddings, make_batch([0, N_DISEASES - 1]), {}, {}
    )

    assert torch.equal(
        outputs["disease_embeddings"],
        embeddings["disease"][torch.tensor([0, N_DISEASES - 1])],
    )
    assert outputs["diagnosis_scores"].shape == (2, N_DISEASES)
    assert outputs["diagnosis_targets"].tolist() == [0, N_DISEASES - 1]


def test_boundary_ids_are_legal():
    """0 and n-1 are valid; only outside the half-open range is not."""
    for ids in ([0, 0], [N_DISEASES - 1, N_DISEASES - 1], [0, N_DISEASES - 1]):
        outputs = compute(ids)
        assert outputs["diagnosis_targets"].tolist() == ids


def test_padded_phenotype_ids_are_still_tolerated():
    """The phenotype clamp is legitimate and must survive.

    `diagnosis_collate_fn` pads phenotype ids with `-1` and the mask discards
    those positions. Removing that clamp along with the disease one would break
    every padded batch, which is every batch.
    """
    outputs = compute([0, 1])
    assert outputs["patient_embeddings"].shape == (2, HIDDEN)
    assert torch.isfinite(outputs["patient_embeddings"]).all()
