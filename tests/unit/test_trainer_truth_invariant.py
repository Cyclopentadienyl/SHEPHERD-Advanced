"""The trainer must refuse a malformed disease truth, not correct it.

A ground-truth disease id outside `[0, n_subgraph_disease_rows)` means the
subgraph seeding, the id remap, the candidate alignment or the batch wiring is
wrong. Scoring it as a rank miss would turn a data-pipeline failure into
apparent model error and contaminate the loss, `val_mrr`, early stopping and
checkpoint selection. **The contract is refuse.**

Three boundaries enforce it, each small and local, and this file covers two:

  1. **`DiagnosisDataLoader._assert_disease_truth_in_range`** — where the failure
     is created, on the host, immediately after the remap that can produce a `-1`
     hole. Covered in `tests/unit/test_data_pipeline.py`.
  2. **`DiagnosisLoss`** — the independent refusal for any caller that bypasses
     that loader. Covered below.
  3. **`to_global_ids`** — the measurement harness's own boundary. Covered in
     `tests/unit/test_measurement_ranking.py:52-56`.

**Scope of the claim below, stated narrowly on purpose.** These tests establish
refusal *at the loss*, plus the fact that `_compute_model_outputs` no longer
clamps. They do **not** drive `_validate` or `evaluate` end to end, so "the
complete path refuses" rests on those two facts plus the call ordering visible in
`trainer.py:640` — source ordering, not coverage. Full orchestration coverage is
backlog item 1b and is not built here.

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
# 1. Loss-level refusal — the independent second boundary
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


def test_the_ranking_primitive_is_permissive_in_isolation():
    """The distinction that made an earlier claim in this project wrong.

    `mean_reciprocal_rank` scores a `-1` truth as `0.0` when called directly.
    That is a fact about the *primitive*, and it was once written up as a fact
    about the trainer — which it is not, because `_validate` calls
    `self.loss_fn(...)` at `trainer.py:640` before collecting predictions at
    `:646-657`, and the loss refuses first.

    **This test does not prove that ordering**; it pins the permissive primitive
    so the difference between it and the path stays visible. The ordering is
    source-level and its coverage belongs to item 1b.
    """
    from src.utils.metrics import RankingMetrics

    assert RankingMetrics().mean_reciprocal_rank([["0", "1", "2"]], ["-1"]) == 0.0

    outputs = compute([0, 1])
    total, _ = MultiTaskLoss(LossConfig())(make_batch([0, 1]), outputs)
    assert torch.isfinite(total), "a legal batch must still produce a finite loss"


# ---------------------------------------------------------------------------
# 2. `_compute_model_outputs` no longer clamps, and adds no CUDA sync
# ---------------------------------------------------------------------------
def test_no_range_check_runs_in_the_hot_path():
    """The guard must not be reinstated here, and this says why in an assertion.

    `disease_ids` reaches `_compute_model_outputs` after `_move_to_device`, so any
    data-dependent branch on it — `bool(t.any())`, `if t.item()`, an `assert` over
    a device tensor — forces a host-device synchronisation on **every valid
    batch** of training and validation. A tensor that records every `bool()` taken
    of it stands in for the CUDA tensor a deployment run would pass.
    """
    calls = []

    class RecordingBool(torch.Tensor):
        @staticmethod
        def __new__(cls, data):
            return torch.Tensor._make_subclass(cls, data)

        def __bool__(self):
            calls.append("bool")
            return super().__bool__()

    batch = make_batch([0, 1])
    batch["disease_ids"] = RecordingBool(batch["disease_ids"])
    Trainer._compute_model_outputs(None, node_embeddings(), batch, {}, {})

    assert calls == [], (
        "a bool() was taken of the disease-id tensor in the hot path; on CUDA "
        "that is a per-batch host-device synchronisation"
    )


def test_a_legal_truth_gathers_the_rows_the_ids_name():
    """The gather uses the **original** ids — no clamp, no copy.

    A clamp here would silently score a different disease for a malformed truth,
    which is the defect this removal exists to prevent.
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
