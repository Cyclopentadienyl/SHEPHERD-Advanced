"""Hand oracles for the LP head, the task heads and losses, and the metrics.

README §8 regressions 1, 5 and 10 and §2.6 E7. Every expected value is hand
arithmetic or a few independent lines; expected and actual never both call
the code under test.

Module: tests/unit/test_task_losses_and_metrics.py
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from src.models.bundle.lp_head import DistMultHead, corrupt_tails  # noqa: E402
from src.models.bundle.metrics import (  # noqa: E402
    MODEL_FAILED,
    NO_OTHER_POSITIVE,
    RANKED,
    QueryOutcome,
    average_ranks,
    lp_metrics,
    ranking_report,
)
from src.models.bundle.spec import (  # noqa: E402
    LPHeadSpec,
    PatientEncoderSpec,
    TaskHeadSpec,
    GeneScoringSpec,
)
from src.models.tasks.heads import (  # noqa: E402
    DiseaseHead,
    GeneHead,
    PatientEncoder,
    PatientsLikeMeHead,
)
from src.models.tasks.losses import (  # noqa: E402
    multi_similarity_loss,
    nca_loss,
    normalised_sq_distance,
)

# --------------------------------------------------------------------------- LP (regression 1)


def _head():
    head = DistMultHead(LPHeadSpec(), num_relations=2, width=2)
    with torch.no_grad():
        head.relation_weights.copy_(torch.tensor([[0.5, -1.0], [1.0, 1.0]]))
    return head


def test_lp_score_and_hinge_equal_hand_values():
    head = _head()
    h, t = torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0, 1.0]])
    # relation 0: 1*0.5*2 + 2*(-1)*1 = -1; relation 1: 2 + 2 = 4.
    assert head.raw_score(h, torch.tensor([0]), t).item() == pytest.approx(-1.0)
    s_pos = head.score(h, torch.tensor([1]), t)
    s_neg = head.score(h, torch.tensor([0]), t)
    assert s_pos.item() == pytest.approx(math.tanh(4.0))
    hinge = head.hinge(s_pos, s_neg).item()
    assert hinge == pytest.approx(max(0.0, 1.0 - (math.tanh(4.0) - math.tanh(-1.0))))
    with pytest.raises(ValueError, match="no \\(positive, negative\\) pair"):
        head.hinge(torch.empty(0), torch.empty(0))


def test_lp_accuracy_equals_micro_f1_on_the_a5_example():
    # y = [1, 1, 0, 0], ŷ = [1, 0, 0, 1] → accuracy 0.5 = micro-F1.
    m = lp_metrics([2.0, -1.0], [-3.0, 0.5])
    assert m["lp_accuracy"] == 0.5


def test_lp_auc_ap_whole_set_and_undefined_without_a_class():
    # pos [0.9, 0.4], neg [0.5, 0.1]: pairs won 3 of 4 → AUC 0.75.
    # Sorted: 0.9+ 0.5- 0.4+ 0.1-: AP = 0.5*1 + 0.5*(2/3).
    m = lp_metrics([0.9, 0.4], [0.5, 0.1])
    assert m["lp_roc_auc"] == pytest.approx(0.75)
    assert m["lp_average_precision"] == pytest.approx(0.5 + 1 / 3)
    assert lp_metrics([1.0], [])["lp_roc_auc"] is None
    # Whole-set, not a mean of batch figures: batches [pos 2, neg -1] and
    # [neg 1, neg -2, neg -3] give accuracies 1 and 2/3 (mean 5/6); the whole
    # set has 4 of 5 right.
    assert lp_metrics([2.0], [-1.0, 1.0, -2.0, -3.0])["lp_accuracy"] == pytest.approx(0.8)


def test_corrupted_tail_never_equals_its_own():
    g = torch.Generator().manual_seed(0)
    rel = torch.tensor([0, 0, 0, 1, 1, 2])
    tails = torch.tensor([5, 6, 5, 7, 7, 8])
    chosen, ok = corrupt_tails(rel, tails, g)
    # Relation 1's two positives share a tail and relation 2 has one: none.
    assert ok.tolist() == [True, True, True, False, False, False]
    for i in range(3):
        assert tails[chosen[i]] != tails[i] and rel[chosen[i]] == rel[i]


# --------------------------------------------------------------------------- distances and NCA


def test_normalised_distance_keeps_zero_vectors_zero():
    d = normalised_sq_distance(torch.tensor([[0.0, 0.0], [3.0, 4.0]]),
                               torch.tensor([[1.0, 0.0]]), 1e-12)
    # Zero vector: ‖0 − e1‖² = 1 (the cosine identity would give 2).
    # [0.6, 0.8] vs [1, 0]: 0.16 + 0.64 = 0.8.
    assert d[:, 0].tolist() == pytest.approx([1.0, 0.8])


def test_nca_two_patients_three_diseases():
    p = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    positive = torch.tensor([[True, False, False], [False, True, False]])
    # Squared distances: p1 [0, 2, 4], p2 [2, 0, 2].
    l1 = math.log(1 + math.exp(-2) + math.exp(-4))
    l2 = math.log(1 + 2 * math.exp(-2))
    result = nca_loss(p, d, positive, scale=1.0, eps=1e-12)
    assert result.value.item() == pytest.approx((l1 + l2) / 2, abs=1e-6)
    assert result.units == 2


def test_nca_refuses_a_query_without_a_positive():
    p = torch.eye(2)
    with pytest.raises(ValueError, match="no positive"):
        nca_loss(p, p, torch.tensor([[True, False], [False, False]]), scale=1.0, eps=1e-12)


def test_patients_like_me_nca_excludes_self():
    # Three patients; 0 and 1 share a gene. Self excluded, so patient 0's
    # candidates are 1 (positive) and 2.
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    positive = torch.tensor([[False, True, False], [True, False, False], [False, False, False]])
    exclude = torch.eye(3, dtype=torch.bool)
    with pytest.raises(ValueError):
        nca_loss(z, z, positive, scale=1.0, eps=1e-12, exclude=exclude)
    r = nca_loss(z[:2], z, positive[:2], scale=1.0, eps=1e-12, exclude=exclude[:2])
    # Patient 0: d to 1 = 2, to 2 = 4. Patient 1: d to 0 = 2, to 2 = 2.
    l0 = -math.log(math.exp(-2) / (math.exp(-2) + math.exp(-4)))
    l1 = -math.log(0.5)
    assert r.value.item() == pytest.approx((l0 + l1) / 2, abs=1e-6)


# --------------------------------------------------------------------------- multi-similarity (C8)


def test_multi_similarity_counterexample_from_the_blueprint():
    sims = torch.tensor([[0.1, 0.9, 0.4]])
    positive = torch.tensor([[True, True, False]])
    valid = torch.ones_like(positive)
    mined = multi_similarity_loss(sims, positive, valid, alpha=1, beta=20, base=1, margin=0.4)
    # Mining keeps 0.1, drops 0.9 (0.9 − 0.4 ≥ 0.4), keeps 0.4.
    hand_mined = math.log(1 + math.exp(0.9)) + math.log(1 + math.exp(20 * (0.4 - 1))) / 20
    assert mined.value.item() == pytest.approx(hand_mined, abs=1e-6)
    assert mined.value.item() == pytest.approx(1.241154, abs=1e-6)
    unmined = multi_similarity_loss(sims, positive, valid, alpha=1, beta=20, base=1, margin=10)
    assert unmined.value.item() == pytest.approx(1.518369, abs=1e-6)


def test_multi_similarity_skips_and_counts_anchors_it_cannot_use():
    sims = torch.tensor([[0.2, 0.3], [0.5, 0.1]])
    positive = torch.tensor([[True, True], [False, False]])
    valid = torch.ones_like(positive)
    r = multi_similarity_loss(sims, positive, valid, alpha=1, beta=20, base=1, margin=0.4)
    assert r.value is None and r.units == 0
    assert r.skipped == {"no_negative": 1, "no_positive": 1}


# --------------------------------------------------------------------------- patient encoder and heads


def test_attention_pooling_weights_and_padding():
    enc = PatientEncoder(PatientEncoderSpec(transformer_layers=0), width=2)
    with torch.no_grad():
        enc.query.copy_(torch.tensor([[1.0, 0.0]]))
        enc.pool_weight.copy_(torch.eye(2))
    x = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [5.0, 5.0]]])
    mask = torch.tensor([[True, True, False]])
    pooled, w = enc(x, mask)
    e = math.exp(1.0)
    assert w[0].tolist() == pytest.approx([e / (e + 1), 1 / (e + 1), 0.0])
    assert pooled[0].tolist() == pytest.approx([e / (e + 1), 0.0])
    with pytest.raises(ValueError, match="without a phenotype"):
        enc(x, torch.zeros(1, 3, dtype=torch.bool))


def test_transformer_ignores_padded_positions():
    torch.manual_seed(0)
    enc = PatientEncoder(PatientEncoderSpec(transformer_layers=1, transformer_heads=2,
                                            feedforward=8), width=4).eval()
    x = torch.randn(1, 3, 4)
    mask = torch.tensor([[True, True, False]])
    a, _ = enc(x, mask)
    x2 = x.clone()
    x2[0, 2] = 100.0
    b, _ = enc(x2, mask)
    assert torch.allclose(a, b, atol=1e-6)


def test_towers_are_separate_parameter_sets():
    head = DiseaseHead(TaskHeadSpec(task="disease"), width=4)
    assert head.patient_tower.first.weight is not head.disease_tower.first.weight
    plm = PatientsLikeMeHead(TaskHeadSpec(task="patients_like_me"), width=4)
    assert not hasattr(plm, "disease_tower")
    # One leaky ReLU (slope 0.1) between two linears (C5).
    with torch.no_grad():
        for lin in (head.patient_tower.first, head.patient_tower.second):
            lin.weight.copy_(torch.eye(4))
            lin.bias.zero_()
    out = head.patient_tower(torch.tensor([[-1.0, 2.0, 0.0, -3.0]]))
    assert out[0].tolist() == pytest.approx([-0.1, 2.0, 0.0, -0.3])


def test_gene_embsim_hand_value():
    head = GeneHead(TaskHeadSpec(task="gene", gene=GeneScoringSpec()), width=2)
    with torch.no_grad():
        head.decoder_weight.copy_(torch.eye(2))
        head.decoder_bias.fill_(0.1)
    s = head.scores(torch.tensor([[3.0, 4.0]]), torch.tensor([[2.0, 0.0]]))
    assert s.item() == pytest.approx(math.tanh(0.6 + 0.1))


# --------------------------------------------------------------------------- ranking metrics (E7)


def test_e7_pooled_and_best_positive_mrr():
    r = ranking_report([QueryOutcome("q", RANKED, [1.0, 10.0])])
    assert r["pooled_mrr"] == pytest.approx(0.55)
    assert r["best_positive_mrr"] == pytest.approx(1.0)


def test_e7_recall_and_precision():
    # 20 relevant patients, 3 of them in the top 5.
    ranks = [1.0, 2.0, 4.0] + [float(r) for r in range(30, 47)]
    r = ranking_report([QueryOutcome("q", RANKED, ranks, n_positives=20)], ks=(5,))
    assert r["macro_recall@5"] == pytest.approx(3 / 20)
    assert r["macro_precision@5"] == pytest.approx(3 / 5)


def test_e7_two_queries_with_different_positive_counts():
    r = ranking_report([QueryOutcome("a", RANKED, [1.0]),
                        QueryOutcome("b", RANKED, [2.0, 4.0, 8.0])], ks=(3,))
    # Pooled over 4 pairs; macro averages per query.
    assert r["pooled_mrr"] == pytest.approx((1 + 0.5 + 0.25 + 0.125) / 4)
    assert r["macro_mrr"] == pytest.approx((1 + (0.5 + 0.25 + 0.125) / 3) / 2)
    assert r["pooled_top3"] == pytest.approx(2 / 4)
    assert r["macro_recall@3"] == pytest.approx((1 + 1 / 3) / 2)


def test_e7_average_tie_straddling_k():
    # Scores: the positive ties with two others for places 2–4: average rank 3.
    ranks = average_ranks([0.9, 0.5, 0.5, 0.5, 0.1])
    assert ranks.tolist() == [1.0, 3.0, 3.0, 3.0, 5.0]
    r = ranking_report([QueryOutcome("q", RANKED, [float(ranks[2])])], ks=(2, 3))
    assert r["pooled_top2"] == 0.0 and r["pooled_top3"] == 1.0


def test_e7_states_kept_apart_and_conservative_reading():
    r = ranking_report([
        QueryOutcome("a", RANKED, [2.0]),
        QueryOutcome("b", NO_OTHER_POSITIVE),
        QueryOutcome("c", MODEL_FAILED, n_positives=1),
    ])
    assert r["units_by_status"][NO_OTHER_POSITIVE] == 1
    assert r["ranked_units"] == 1 and r["coverage"] == pytest.approx(1 / 3)
    assert r["pooled_mrr"] == pytest.approx(0.5)
    # The failed unit's positive counts 0; the no-positive unit is not a recall 0.
    assert r["conservative_pooled_mrr"] == pytest.approx(0.25)
    assert r["conservative_denominator_pairs"] == 2
    with pytest.raises(ValueError):
        QueryOutcome("x", NO_OTHER_POSITIVE, [1.0])
