"""
The scoring primitives must agree with what they replaced, and the pipeline must
actually route through them.
=============================================================================
`src/inference/scoring.py` exists because the score a clinician sees and the
score an offline evaluation reports were computed by two separate
implementations of the same formulas. Two implementations drift; when they
drift, the evaluation stops describing the thing being evaluated.

**The evaluator's copy ended with its deletion, not by migration**:
`scripts/evaluate_model.py` was never migrated and is deleted (its code is at
`7dab728`). The measurement harness composes these primitives; the served
pipeline stopped at K8-4, when the served mixture retired, and its routing
tests left with it. The shortest-path lookup was removed in K8-5, and its
tests here left with it. The cosine is shared; the pool is still two deliberate
implementations (`pool_patient_embeddings`, `masked_mean_pool`), bound by
`test_masked_pool_and_served_pool_agree_on_one_unpadded_patient`.

Two properties worth testing, and they are different:

  - **Agreement.** The primitives compute what the previous inline code
    computed. Each test below reimplements the legacy formula *independently*
    rather than calling the primitive twice — a test that calls the thing it is
    testing to produce its own expectation cannot detect a wrong formula.
  - **Precision.** The scalar wrappers must reproduce the legacy Python-double
    arithmetic *exactly*. Rounding the mixture to float32 would let two
    candidates whose true scores differ below its resolution become an exact
    tie, resolved thereafter by input order — which can move a candidate across
    the top-k boundary a clinician sees.

Tolerances are declared where they belong and refused where they do not.
Comparing a batched reduction with a scalar loop needs one, because summation
order differs. The scalar mixture does not get one: its contract is behaviour
preservation, and a tolerance there would permit the very rounding these tests
forbid. Ranking is always asserted exactly.
"""
import pytest

torch = pytest.importorskip("torch")

from src.inference.scoring import (  # noqa: E402
    cosine_scores,
    mix_embedding_and_sp_scores,
    normalise_cosine_to_unit_interval,
    pool_patient_embeddings,
)

# Declared tolerances. Scores feed a ranking and a clinician-facing display;
# 1e-6 is far below any difference either could act on, and far above the
# summation-order noise between a loop and a reduction.
ATOL = 1e-6
RTOL = 1e-6


@pytest.fixture
def embeddings():
    torch.manual_seed(0)
    return torch.randn(40, 16)


# ---------------------------------------------------------------------------
# Patient pooling
# ---------------------------------------------------------------------------
def test_pooling_matches_an_independent_mean(embeddings):
    indices = [3, 11, 27]
    expected = (embeddings[3] + embeddings[11] + embeddings[27]) / 3.0

    got = pool_patient_embeddings(embeddings, indices)

    assert torch.allclose(got, expected, atol=ATOL, rtol=RTOL)


def test_pooling_clamps_out_of_range_indices(embeddings):
    """Behaviour inherited from the code this replaced: clamp rather than raise."""
    last = embeddings.size(0) - 1

    assert torch.allclose(
        pool_patient_embeddings(embeddings, [999]),
        embeddings[last],
        atol=ATOL, rtol=RTOL,
    )


def test_pooling_rejects_an_empty_phenotype_set(embeddings):
    """A mean over nothing has no value; the caller must not receive zeros that
    look like an answer."""
    with pytest.raises(ValueError):
        pool_patient_embeddings(embeddings, [])


# ---------------------------------------------------------------------------
# Embedding similarity
# ---------------------------------------------------------------------------
def test_cosine_matches_an_independent_computation(embeddings):
    patient = pool_patient_embeddings(embeddings, [0, 1, 2])
    candidates = embeddings[10:15]

    got = cosine_scores(patient, candidates)

    for i in range(candidates.size(0)):
        c = candidates[i]
        expected = float(
            torch.dot(patient, c) / (patient.norm() * c.norm())
        )
        assert got[i].item() == pytest.approx(expected, abs=ATOL, rel=RTOL)


def test_cosine_batch_of_one_matches_the_full_batch(embeddings):
    """Batch shape is the interface, not an optimisation. Scoring one candidate
    and scoring the universe must agree for the candidates they share."""
    patient = pool_patient_embeddings(embeddings, [4, 5])
    candidates = embeddings[20:30]

    full = cosine_scores(patient, candidates)

    for i in range(candidates.size(0)):
        single = cosine_scores(patient, candidates[i].unsqueeze(0))
        assert single[0].item() == pytest.approx(full[i].item(), abs=ATOL, rel=RTOL)


def test_normalisation_preserves_order(embeddings):
    """(x+1)/2 is monotone, so it may change the number but never the ranking."""
    patient = pool_patient_embeddings(embeddings, [7, 8, 9])
    raw = cosine_scores(patient, embeddings)

    assert torch.equal(raw.argsort(descending=True),
                       normalise_cosine_to_unit_interval(raw).argsort(descending=True))


# ---------------------------------------------------------------------------
# Mixture
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eta", [0.0, 0.3, 0.7, 1.0])
def test_mixture_matches_the_formula(eta):
    emb = torch.tensor([0.9, 0.2, 0.5])
    sp = torch.tensor([0.2, 0.4, 0.5])

    got = mix_embedding_and_sp_scores(emb, sp, eta)

    for i in range(3):
        expected = eta * float(emb[i]) + (1.0 - eta) * float(sp[i])
        assert got[i].item() == pytest.approx(expected, abs=ATOL, rel=RTOL)


def test_eta_one_ignores_the_shortest_path_term():
    emb = torch.tensor([0.9, 0.2])

    got = mix_embedding_and_sp_scores(emb, torch.tensor([99.0, -99.0]), 1.0)

    assert torch.allclose(got, emb, atol=ATOL)


# ---------------------------------------------------------------------------
# Determinism and ranking
# ---------------------------------------------------------------------------
def test_scores_are_deterministic(embeddings):
    patient = pool_patient_embeddings(embeddings, [1, 2, 3])

    first = cosine_scores(patient, embeddings)
    second = cosine_scores(patient, embeddings)

    assert torch.equal(first, second)


def test_exact_ties_break_by_candidate_order():
    """Candidate-order equivalence needs ties to resolve the same way every
    time. torch.argsort is stable, so equal scores keep their input order — this
    pins that rather than leaving it to chance."""
    scores = torch.tensor([0.5, 0.9, 0.5, 0.9])

    order = scores.argsort(descending=True, stable=True)

    assert order.tolist() == [1, 3, 0, 2]


# ---------------------------------------------------------------------------
# Precision — the extraction is only behaviour-preserving in double precision
# ---------------------------------------------------------------------------
# Two scores a fifth of a float32 step apart near 0.5. Distinct as Python
# doubles, identical once rounded to float32. Constructed rather than borrowed:
# a pair that does *not* collapse would let these tests pass with the bug
# present, which is the failure mode a regression test exists to prevent.
_F32_STEP_NEAR_HALF = 2.0**-24
NEAR_TIE_A = 0.5 + _F32_STEP_NEAR_HALF * 0.2
NEAR_TIE_B = 0.5 + _F32_STEP_NEAR_HALF * 0.4


def test_the_near_tie_pair_really_does_collapse_in_float32():
    """Pins the premise of the tests below. If a future torch changes rounding
    such that these no longer collapse, the regression tests would silently stop
    testing anything, and this fails first to say so."""
    assert NEAR_TIE_A != NEAR_TIE_B
    assert (
        torch.tensor([NEAR_TIE_A], dtype=torch.float32).item()
        == torch.tensor([NEAR_TIE_B], dtype=torch.float32).item()
    )


def _legacy_mixture(eta, emb, sp):
    """The η mixture as the pipeline computed it before extraction: Python
    doubles throughout."""
    return eta * emb + (1.0 - eta) * sp


@pytest.mark.parametrize("eta", [0.0, 0.3, 0.7, 1.0, 0.123456789])
def test_float64_mixture_is_exactly_the_legacy_arithmetic(eta):
    """Not "within tolerance" — exactly equal. The scalar wrapper's contract is
    behaviour preservation, and a tolerance here would permit precisely the
    rounding this test exists to forbid."""
    import random

    random.seed(20260817)
    for _ in range(2000):
        emb, sp = random.random(), random.random()
        got = float(
            mix_embedding_and_sp_scores(
                torch.tensor([emb], dtype=torch.float64),
                torch.tensor([sp], dtype=torch.float64),
                eta,
            )[0]
        )
        assert got == _legacy_mixture(eta, emb, sp)


def test_float32_mixture_would_have_lost_the_distinction():
    """Why the dtype is specified rather than left to default.

    This documents the defect the float64 requirement prevents: constructing the
    tensors without a dtype gives float32, and two candidates whose true scores
    differ below its resolution become an exact tie.
    """
    as_f32 = [
        float(mix_embedding_and_sp_scores(
            torch.tensor([v], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            1.0,
        )[0])
        for v in (NEAR_TIE_A, NEAR_TIE_B)
    ]
    as_f64 = [
        float(mix_embedding_and_sp_scores(
            torch.tensor([v], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
            1.0,
        )[0])
        for v in (NEAR_TIE_A, NEAR_TIE_B)
    ]

    assert as_f32[0] == as_f32[1], "float32 collapses them — the regression"
    assert as_f64[0] != as_f64[1], "float64 keeps them apart — the fix"


def test_near_tie_candidate_order_survives_the_mixture():
    """Ordering, not just arithmetic. Candidate B outranks A by less than a
    float32 step; the ranking must still put B first."""
    scores = torch.tensor([NEAR_TIE_A, NEAR_TIE_B], dtype=torch.float64)

    mixed = mix_embedding_and_sp_scores(scores, torch.zeros(2, dtype=torch.float64), 1.0)

    assert mixed.argsort(descending=True, stable=True).tolist() == [1, 0]


def test_top_k_boundary_is_not_moved_by_the_mixture():
    """The clinical consequence of a lost distinction: a candidate crossing the
    edge of what a clinician is shown."""
    top_k = 2
    emb = torch.tensor([0.9, NEAR_TIE_B, NEAR_TIE_A, 0.1], dtype=torch.float64)

    mixed = mix_embedding_and_sp_scores(emb, torch.zeros(4, dtype=torch.float64), 1.0)
    shown = mixed.argsort(descending=True, stable=True)[:top_k].tolist()

    assert shown == [0, 1], "candidate 1 must hold the second slot, not candidate 2"


# ---------------------------------------------------------------------------
# Device placement
# ---------------------------------------------------------------------------
def test_pooling_places_its_index_on_the_embedding_device(embeddings):
    """The index tensor follows the embeddings rather than defaulting to CPU.

    On CPU this is invisible; the full-universe evaluator will hold embeddings on
    an accelerator, where a CPU index tensor is at best an implicit conversion and
    at worst an error. Pinning it here means the deployed path is not relying on
    an undocumented indexing convenience.
    """
    out = pool_patient_embeddings(embeddings, [1, 2, 3])

    assert out.device == embeddings.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_pooling_and_cosine_work_on_an_accelerator(embeddings):
    device = torch.device("cuda")
    on_device = embeddings.to(device)

    patient = pool_patient_embeddings(on_device, [1, 2, 3])
    scores = cosine_scores(patient, on_device)

    assert patient.device.type == "cuda"
    assert scores.device.type == "cuda"
    assert torch.allclose(
        scores.cpu(),
        cosine_scores(pool_patient_embeddings(embeddings, [1, 2, 3]), embeddings),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Batched primitives — the offline-evaluation shapes (B-0.2 step 1)
# ---------------------------------------------------------------------------
# The served path scores one patient against a candidate matrix; the offline
# evaluator scores a padded batch of patients. These are the batched forms, and
# the tests below do two things: check each against an independent
# reimplementation, and pin the served path to the batched one so the two cannot
# drift.
from src.inference.scoring import (  # noqa: E402
    cosine_score_matrix,
    masked_mean_pool,
)


def _legacy_masked_mean(embeddings, mask):
    """`Trainer._compute_model_outputs` (trainer.py:744-751), reimplemented here so
    this test does not depend on the code under test being right."""
    weights = mask.unsqueeze(-1).float()
    summed = (embeddings * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp(min=1)
    return summed / counts


def test_masked_pool_matches_the_trainer_formula():
    torch.manual_seed(1)
    emb = torch.randn(4, 6, 8)
    mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    assert torch.equal(masked_mean_pool(emb, mask), _legacy_masked_mean(emb, mask))


def test_masked_pool_ignores_padding():
    """Padded positions must not reach the mean, whatever they contain. Filling
    them with a large value makes a leak impossible to miss."""
    emb = torch.zeros(1, 4, 3)
    emb[0, :2] = 1.0
    emb[0, 2:] = 1e6
    mask = torch.tensor([[True, True, False, False]])

    assert torch.equal(masked_mean_pool(emb, mask), torch.ones(1, 3))


def test_masked_pool_all_true_equals_a_plain_mean():
    torch.manual_seed(2)
    emb = torch.randn(3, 5, 7)

    got = masked_mean_pool(emb, torch.ones(3, 5, dtype=torch.bool))

    assert torch.allclose(got, emb.mean(dim=1), atol=ATOL, rtol=RTOL)


def test_masked_pool_all_false_is_a_zero_vector():
    """Behaviour preserved, not endorsed. The trainer clamps the denominator to
    one, so an empty row yields zeros rather than raising or producing NaN. The
    Trainer value tests take their expected rows from this function
    (`test_trainer_validation_values.py`), so it keeps the Trainer's formula;
    Mode C refuses a patient with no phenotypes before pooling."""
    emb = torch.randn(2, 4, 5)

    got = masked_mean_pool(emb, torch.zeros(2, 4, dtype=torch.bool))

    assert torch.equal(got, torch.zeros(2, 5))
    assert torch.isfinite(got).all()


@pytest.mark.parametrize(
    "emb_shape, mask_shape",
    [((2, 3, 4), (2, 4)), ((2, 3, 4), (3, 3)), ((2, 3, 4), (2,)), ((3, 4), (3, 4))],
)
def test_masked_pool_rejects_shape_mismatch(emb_shape, mask_shape):
    with pytest.raises(ValueError):
        masked_mean_pool(torch.randn(*emb_shape), torch.ones(*mask_shape, dtype=torch.bool))


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64, torch.int64, torch.uint8]
)
def test_masked_pool_rejects_a_non_boolean_mask(dtype):
    """A validity mask and a same-shaped floating-point tensor are otherwise
    indistinguishable at this boundary, and only the dtype tells them apart.
    `.float()` would silently turn the latter into a weighting vector: plausible
    numbers, wrong semantics, no error anywhere. A silent wrong answer is worse
    than a crash, and this is one of the few places the type system will not
    catch it.

    Requiring bool costs no parity: the production dataloader constructs
    `phenotype_mask` as `torch.bool` (`src/kg/data_loader.py:704-707`) and indexes
    with it (`:912`), which already requires bool.
    """
    mask = torch.ones(2, 3).to(dtype)

    with pytest.raises(ValueError, match="boolean"):
        masked_mean_pool(torch.randn(2, 3, 4), mask)


def test_masked_pool_keeps_its_device():
    emb = torch.randn(2, 3, 4)

    out = masked_mean_pool(emb, torch.ones(2, 3, dtype=torch.bool))

    assert out.device == emb.device


def test_masked_pool_and_served_pool_agree_on_one_unpadded_patient():
    """The equivalence that lets the two implementations stay separate.

    `pool_patient_embeddings` takes an index list over a node table;
    `masked_mean_pool` takes a padded batch. Given the same phenotypes and no
    padding they must produce the same vector — which is what makes keeping both
    a clarity choice rather than a drift risk.
    """
    torch.manual_seed(3)
    node_embeddings = torch.randn(20, 6)
    indices = [4, 9, 17]

    served = pool_patient_embeddings(node_embeddings, indices)
    batched = masked_mean_pool(
        node_embeddings[indices].unsqueeze(0), torch.ones(1, len(indices), dtype=torch.bool)
    )

    assert torch.allclose(served, batched[0], atol=ATOL, rtol=RTOL)


def test_cosine_matrix_matches_an_independent_computation():
    torch.manual_seed(4)
    patients = torch.randn(3, 9)
    candidates = torch.randn(5, 9)

    got = cosine_score_matrix(patients, candidates)

    assert got.shape == (3, 5)
    for b in range(3):
        for d in range(5):
            expected = float(
                torch.dot(patients[b], candidates[d])
                / (patients[b].norm() * candidates[d].norm())
            )
            assert got[b, d].item() == pytest.approx(expected, abs=ATOL, rel=RTOL)


def test_served_cosine_is_the_batch_of_one(embeddings):
    """The served signature is unchanged and now delegates. This pins that the
    delegation is exact, so the two shapes cannot diverge."""
    patient = pool_patient_embeddings(embeddings, [0, 1, 2])
    candidates = embeddings[10:20]

    served = cosine_scores(patient, candidates)
    batched = cosine_score_matrix(patient.unsqueeze(0), candidates)

    assert torch.equal(served, batched[0])


@pytest.mark.parametrize(
    "patients, candidates",
    [(torch.randn(4), torch.randn(3, 4)), (torch.randn(2, 4), torch.randn(4))],
)
def test_cosine_matrix_rejects_wrong_rank(patients, candidates):
    with pytest.raises(ValueError):
        cosine_score_matrix(patients, candidates)


# --- Accelerator coverage -------------------------------------------------
# Hardware acceleration is a hard requirement of every deployment and CUDA is
# currently the only supported backend, so these are the configurations
# that actually run in the hospital. They cannot be exercised in this
# environment; the deployment's own verification covers the device and dtype.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_batched_primitives_on_cuda():
    device = torch.device("cuda")
    emb = torch.randn(2, 3, 8, device=device)
    mask = torch.ones(2, 3, dtype=torch.bool, device=device)

    pooled = masked_mean_pool(emb, mask)
    scores = cosine_score_matrix(pooled, torch.randn(5, 8, device=device))

    assert pooled.device.type == "cuda"
    assert scores.shape == (2, 5)
    assert torch.allclose(
        pooled.cpu(), masked_mean_pool(emb.cpu(), mask.cpu()), atol=1e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_masked_pool_under_reduced_precision(dtype):
    """Dtype follows promotion, and the assertion is parity with the trainer's
    formula — not an assumed preservation rule."""
    device = torch.device("cuda")
    emb = torch.randn(2, 4, 8, device=device, dtype=dtype)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool, device=device)

    got = masked_mean_pool(emb, mask)
    expected = _legacy_masked_mean(emb, mask)

    assert got.dtype == expected.dtype
    assert torch.equal(got, expected)
