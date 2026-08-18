"""
The scoring primitives must agree with what they replaced, and the pipeline must
actually route through them.
=============================================================================
`src/inference/scoring.py` exists because the score a clinician sees and the
score an offline evaluation reports were computed by two separate
implementations of the same formulas. Two implementations drift; when they
drift, the evaluation stops describing the thing being evaluated.

**That condition is not yet ended.** The pipeline composes the primitives;
`scripts/evaluate_model.py` does not yet. These tests cover the migrated half.

Three properties worth testing, and they are different:

  - **Agreement.** The primitives compute what the previous inline code
    computed. Each test below reimplements the legacy formula *independently*
    rather than calling the primitive twice — a test that calls the thing it is
    testing to produce its own expectation cannot detect a wrong formula.
  - **Authority.** The pipeline actually routes through the primitives. A
    correct primitive that nobody calls prevents nothing, so each wrapper has a
    test that replaces its primitive and asserts the pipeline's answer follows.
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
    SPLookup,
    cosine_scores,
    mix_embedding_and_sp_scores,
    normalise_cosine_to_unit_interval,
    pool_patient_embeddings,
    sp_mean_distances,
    sp_scores_from_distances,
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


@pytest.fixture
def lookup():
    """Three phenotypes over a small table, covering every case that matters:
    a present pair, an absent pair (no path within max_hops), and a phenotype
    with no slice at all."""
    #            ph0        ph0        ph0        ph1        ph1
    target = torch.tensor([5, 7, 9, 5, 8], dtype=torch.int32)
    ttype = torch.tensor([1, 1, 0, 1, 1], dtype=torch.int8)
    distance = torch.tensor([2, 4, 1, 3, 5], dtype=torch.int8)
    offsets = {0: (0, 3), 1: (3, 5)}  # phenotype 2 deliberately absent
    return SPLookup(target=target, target_type=ttype, distance=distance,
                    offsets=offsets, max_hops=5)


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
# Shortest-path distance
# ---------------------------------------------------------------------------
def _legacy_sp_score(lookup, phenotype_indices, target_idx, target_type_idx):
    """The formula as it was written inline in the pipeline, reimplemented here
    so this test does not depend on the code under test being right."""
    unreachable = float(lookup.max_hops + 1)
    total = 0.0
    for ph in phenotype_indices:
        offsets = lookup.offsets.get(ph)
        if offsets is None:
            total += unreachable
            continue
        s, e = offsets
        mask = (lookup.target[s:e] == target_idx) & (lookup.target_type[s:e] == target_type_idx)
        hits = lookup.distance[s:e][mask]
        total += float(hits[0]) if len(hits) > 0 else unreachable
    return 1.0 / (1.0 + total / len(phenotype_indices))


@pytest.mark.parametrize("phenotypes", [[0], [0, 1], [0, 1, 2], [2]])
@pytest.mark.parametrize("target_idx", [5, 7, 8, 99])
def test_sp_score_matches_the_legacy_formula(lookup, phenotypes, target_idx):
    distances, available = sp_mean_distances(lookup, phenotypes, [target_idx], 1)

    assert bool(available[0])
    assert float(sp_scores_from_distances(distances)[0]) == pytest.approx(
        _legacy_sp_score(lookup, phenotypes, target_idx, 1), abs=ATOL, rel=RTOL
    )


def test_target_type_is_part_of_the_match(lookup):
    """Target 9 exists for phenotype 0 but as a gene, not a disease. Asking for
    the disease must not find it."""
    as_disease, _ = sp_mean_distances(lookup, [0], [9], 1)
    as_gene, _ = sp_mean_distances(lookup, [0], [9], 0)

    assert float(as_disease[0]) == 6.0   # unreachable: max_hops + 1
    assert float(as_gene[0]) == 1.0


def test_a_phenotype_with_no_slice_counts_as_unreachable(lookup):
    """Phenotype 2 has no entry in the table at all. It contributes the
    unreachable distance rather than being silently dropped, which would let a
    candidate look closer than it is."""
    with_missing, _ = sp_mean_distances(lookup, [0, 2], [5], 1)

    assert float(with_missing[0]) == pytest.approx((2.0 + 6.0) / 2.0)


def test_all_unreachable_is_still_a_computed_value(lookup):
    """The distinction C9 turns on: a candidate every one of whose phenotypes is
    unreachable has a real value — the largest one — and is NOT the same as a
    candidate that could not be evaluated."""
    distances, available = sp_mean_distances(lookup, [0, 1], [12345], 1)

    assert bool(available[0]), "unreachable is a distance, not an absence"
    assert float(distances[0]) == 6.0
    assert float(sp_scores_from_distances(distances)[0]) == pytest.approx(1.0 / 7.0)


def test_no_phenotypes_is_unavailable_not_zero_distance(lookup):
    """With nothing to measure from, there is no distance. The mask says so
    rather than the value pretending to."""
    _, available = sp_mean_distances(lookup, [], [5], 1)

    assert not bool(available[0])


def test_sp_batch_matches_one_at_a_time(lookup):
    targets = [5, 7, 8, 99]

    batched, _ = sp_mean_distances(lookup, [0, 1], targets, 1)

    for position, target in enumerate(targets):
        single, _ = sp_mean_distances(lookup, [0, 1], [target], 1)
        assert float(single[0]) == pytest.approx(float(batched[position]), abs=ATOL)


def test_computed_scores_stay_inside_the_documented_range(lookup):
    """[1/7, 1/2] at max_hops = 5 — never 0, which is why 0.0 as an
    unavailable sentinel is indistinguishable from nothing real."""
    distances, _ = sp_mean_distances(lookup, [0, 1], [5, 7, 8, 99], 1)
    scores = sp_scores_from_distances(distances)

    assert float(scores.min()) >= 1.0 / 7.0 - ATOL
    assert float(scores.max()) <= 1.0 / 2.0 + ATOL


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
def test_scores_are_deterministic(embeddings, lookup):
    patient = pool_patient_embeddings(embeddings, [1, 2, 3])

    first = cosine_scores(patient, embeddings)
    second = cosine_scores(patient, embeddings)
    d1, _ = sp_mean_distances(lookup, [0, 1], [5, 7], 1)
    d2, _ = sp_mean_distances(lookup, [0, 1], [5, 7], 1)

    assert torch.equal(first, second)
    assert torch.equal(d1, d2)


def test_exact_ties_break_by_candidate_order():
    """Candidate-order equivalence needs ties to resolve the same way every
    time. torch.argsort is stable, so equal scores keep their input order — this
    pins that rather than leaving it to chance."""
    scores = torch.tensor([0.5, 0.9, 0.5, 0.9])

    order = scores.argsort(descending=True, stable=True)

    assert order.tolist() == [1, 3, 0, 2]


# ---------------------------------------------------------------------------
# Authority — the pipeline must actually route through these
# ---------------------------------------------------------------------------
def test_pipeline_gnn_score_routes_through_the_primitive(monkeypatch):
    """A correct primitive that nobody calls prevents no drift.

    Replace `cosine_scores` and assert the pipeline's score follows it. The
    pipeline imports the primitives lazily inside the method, so patching the
    module attribute is what the call actually resolves.
    """
    from types import SimpleNamespace

    import src.inference.scoring as scoring
    from src.core.types import DataSource, NodeID
    from src.inference.pipeline import DiagnosisPipeline

    monkeypatch.setattr(
        scoring, "cosine_scores", lambda patient, candidates: torch.tensor([0.5])
    )

    fake = SimpleNamespace(
        _gnn_ready=True,
        _node_embeddings={"phenotype": torch.randn(5, 4), "disease": torch.randn(5, 4)},
        _node_id_to_idx={"phenotype": {"hpo:HP:0000001": 0}, "disease": {"mondo:D1": 1}},
    )
    source = NodeID(source=DataSource.HPO, local_id="HP:0000001")
    target = NodeID(source=DataSource.MONDO, local_id="D1")

    score = DiagnosisPipeline._calculate_gnn_score(fake, [source], target, None)

    # (0.5 + 1) / 2 — the patched cosine, carried through the normalisation.
    assert score == pytest.approx(0.75, abs=ATOL)


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
# Authority — every primitive the pipeline claims to route through
# ---------------------------------------------------------------------------
def _sp_ready_pipeline_stub(lookup):
    from types import SimpleNamespace

    return SimpleNamespace(
        _sp_ready=True,
        _sp_lookup=lookup,
        _sp_max_hops=lookup.max_hops,
        _node_id_to_idx={"phenotype": {"hpo:HP:0000001": 0}, "disease": {"mondo:D1": 5}},
    )


def _ids():
    from src.core.types import DataSource, NodeID

    return (
        NodeID(source=DataSource.HPO, local_id="HP:0000001"),
        NodeID(source=DataSource.MONDO, local_id="D1"),
    )


def test_pipeline_sp_score_routes_through_both_sp_primitives(monkeypatch, lookup):
    import src.inference.scoring as scoring
    from src.inference.pipeline import DiagnosisPipeline

    seen = {}

    def fake_distances(lk, phenotypes, targets, target_type):
        seen["called"] = (list(phenotypes), list(targets), target_type)
        return torch.tensor([3.0]), torch.tensor([True])

    monkeypatch.setattr(scoring, "sp_mean_distances", fake_distances)
    monkeypatch.setattr(scoring, "sp_scores_from_distances", lambda d: d * 100.0)

    source, target = _ids()
    got = DiagnosisPipeline._calculate_sp_score(
        _sp_ready_pipeline_stub(lookup), [source], target, 1
    )

    assert seen["called"] == ([0], [5], 1), "the pipeline must pass resolved indices"
    assert got == pytest.approx(300.0), "and must return what the transform gave it"


def test_pipeline_sp_score_honours_the_availability_mask(monkeypatch, lookup):
    """Unavailable collapses to 0.0 in this legacy wrapper. That is the behaviour
    being preserved, not the behaviour being endorsed — see the module docstring."""
    import src.inference.scoring as scoring
    from src.inference.pipeline import DiagnosisPipeline

    monkeypatch.setattr(
        scoring, "sp_mean_distances",
        lambda *a, **k: (torch.tensor([3.0]), torch.tensor([False])),
    )

    source, target = _ids()
    got = DiagnosisPipeline._calculate_sp_score(
        _sp_ready_pipeline_stub(lookup), [source], target, 1
    )

    assert got == 0.0


def test_pipeline_combined_score_routes_through_the_mixture(monkeypatch, lookup):
    """Routing, and the dtype the wrapper hands over.

    The dtype assertion pins the *mechanism* of the float32 fix: the wrapper is
    the only place that chooses how the two Python doubles are widened into
    tensors, and a `torch.tensor([x])` without a dtype gives float32. On its own
    it is a white-box check and would not notice a value rounded before the
    tensor was built — the two tests after this one cover that.
    """
    import src.inference.scoring as scoring
    from src.inference.pipeline import DiagnosisPipeline

    stub = _sp_ready_pipeline_stub(lookup)
    stub.config = type("C", (), {"eta": 0.7})()
    stub._calculate_gnn_score = lambda *a, **k: 0.8
    stub._calculate_sp_score = lambda *a, **k: 0.2

    seen = {}

    def fake_mixture(embedding_scores, sp_scores, eta):
        seen["dtypes"] = (embedding_scores.dtype, sp_scores.dtype)
        return torch.tensor([42.0])

    monkeypatch.setattr(scoring, "mix_embedding_and_sp_scores", fake_mixture)

    source, target = _ids()
    combined, emb, sp = DiagnosisPipeline._calculate_combined_score(
        stub, [source], target, None
    )

    assert combined == pytest.approx(42.0), "the wrapper must use the primitive"
    assert (emb, sp) == (0.8, 0.2), "and must report the components unchanged"
    assert seen["dtypes"] == (torch.float64, torch.float64), (
        "the wrapper must widen to float64; float32 would round the incoming "
        "doubles to ~7 significant digits"
    )


# ---------------------------------------------------------------------------
# Precision, at the place the defect actually occurred
# ---------------------------------------------------------------------------
# The tests in the previous section exercise `mix_embedding_and_sp_scores` with
# tensors they construct themselves as float64. They therefore say nothing about
# the production wrapper, which is where the float32 regression lived: reverting
# `pipeline.py` to `torch.tensor([emb_score])` left every one of them passing.
# The three tests below drive the wrapper itself.
def _combined_score_stub(lookup, eta, emb, sp):
    stub = _sp_ready_pipeline_stub(lookup)
    stub.config = type("C", (), {"eta": eta})()
    stub._calculate_gnn_score = lambda *a, **k: emb
    stub._calculate_sp_score = lambda *a, **k: sp
    return stub


@pytest.mark.parametrize("eta", [0.0, 0.3, 0.7, 1.0, 0.123456789])
def test_combined_score_wrapper_is_exactly_the_legacy_arithmetic(lookup, eta):
    """`_calculate_combined_score` must return the Python-double result, bit for
    bit — not a value within tolerance of it.

    This is the black-box form of the dtype check above: it fails for any loss of
    precision inside the wrapper, whether from the tensor dtype or from anything
    else introduced later.
    """
    import random

    from src.inference.pipeline import DiagnosisPipeline

    source, target = _ids()
    random.seed(20260818)
    for _ in range(2000):
        emb, sp = random.random(), random.random()

        combined, _, _ = DiagnosisPipeline._calculate_combined_score(
            _combined_score_stub(lookup, eta, emb, sp), [source], target, None
        )

        assert combined == _legacy_mixture(eta, emb, sp)


def test_combined_score_wrapper_keeps_a_sub_float32_distinction(lookup):
    """Two candidates closer together than a float32 step stay distinct through
    the wrapper. Under the regression they became an exact tie."""
    from src.inference.pipeline import DiagnosisPipeline

    source, target = _ids()
    scores = [
        DiagnosisPipeline._calculate_combined_score(
            _combined_score_stub(lookup, 0.7, emb, 0.3), [source], target, None
        )[0]
        for emb in (NEAR_TIE_A, NEAR_TIE_B)
    ]

    assert scores[0] != scores[1], (
        "the wrapper collapsed a real difference — this is the regression"
    )
    assert scores[1] > scores[0]


def _multi_phenotype_ids():
    """Three distinct phenotypes, mapping to lookup indices 0, 1 and 2.

    Against target 5 as a disease they contribute 2, 3 and — phenotype 2 having
    no slice at all — the unreachable 6. The mean is 11/3, which is the point:
    a sum that does not divide by the count, so the mean is a fraction no binary
    format represents exactly.
    """
    from src.core.types import DataSource, NodeID

    return [
        NodeID(source=DataSource.HPO, local_id=f"HP:000000{i + 1}") for i in range(3)
    ]


def test_sp_score_wrapper_is_exactly_the_legacy_arithmetic(lookup):
    """The same contract, one function away — and it was broken by the extraction.

    Before `337266f` the pipeline accumulated the total and computed
    `1 / (1 + total / n)` in Python doubles. The extracted `sp_mean_distances`
    first stored the mean in a float32 tensor, so the returned score differed from
    the legacy value at the eighth significant digit.

    **The multi-phenotype case is the one that matters, and a single phenotype
    does not exercise it.** With one phenotype the mean is an integer, exact in
    both formats; only a fractional mean — 11/3 here, 83/24 in the documented
    example — can be rounded by the accumulation itself. A single-phenotype test
    would still fail against a float32 implementation, but only because the
    reciprocal transform rounds, which is a different and weaker property.
    """
    from src.inference.pipeline import DiagnosisPipeline

    phenotypes = _multi_phenotype_ids()
    _, target = _ids()
    stub = _sp_ready_pipeline_stub(lookup)
    stub._node_id_to_idx = {
        "phenotype": {str(nid): i for i, nid in enumerate(phenotypes)},
        "disease": {str(target): 5},
    }

    got = DiagnosisPipeline._calculate_sp_score(stub, phenotypes, target, 1)

    assert got == _legacy_sp_score(lookup, [0, 1, 2], 5, 1)
    # Named so a change to the fixture that made the mean integral would be
    # visible rather than silently weakening the test.
    assert _legacy_sp_score(lookup, [0, 1, 2], 5, 1) == 1.0 / (1.0 + 11.0 / 3.0)


@pytest.mark.parametrize("target_idx", [5, 7, 99])
def test_sp_score_wrapper_matches_legacy_for_a_single_phenotype(lookup, target_idx):
    """The integral-mean case, kept as its own test rather than folded into the
    fractional one — the two catch different failures."""
    from src.inference.pipeline import DiagnosisPipeline

    source, target = _ids()
    stub = _sp_ready_pipeline_stub(lookup)
    stub._node_id_to_idx = {
        "phenotype": {str(source): 0},
        "disease": {str(target): target_idx},
    }

    got = DiagnosisPipeline._calculate_sp_score(stub, [source], target, 1)

    assert got == _legacy_sp_score(lookup, [0], target_idx, 1)


def test_reducing_in_float32_then_widening_loses_the_mean(lookup):
    """Pins the premise, and names the failure mode the output dtype cannot see.

    A vectorised implementation could sum and divide in float32 and then widen the
    already-rounded mean into a float64 output tensor. `sp_mean_distances` would
    still return float64, so `test_sp_mean_distances_are_float64` would pass, and
    so would every test that only inspects the output type. This shows the value
    is gone by then — which is why the contract in the docstring covers the whole
    computation and not just the result.
    """
    total, n = 11.0, 3

    in_double = total / n
    reduced_in_f32_then_widened = float(
        torch.tensor([total / n], dtype=torch.float32).to(torch.float64)[0]
    )

    assert reduced_in_f32_then_widened != in_double
    assert 1.0 / (1.0 + reduced_in_f32_then_widened) != 1.0 / (1.0 + in_double)


def test_sp_mean_distances_are_float64(lookup):
    """The dtype is the mechanism behind the test above; pin it directly so a
    vectorised reimplementation cannot quietly narrow it for memory."""
    distances, _ = sp_mean_distances(lookup, [0, 1], [5, 7, 99], 1)

    assert distances.dtype == torch.float64


def test_ranking_path_keeps_a_sub_float32_distinction(lookup):
    """The clinical consequence, through the real ranking code.

    Two candidates are separated by less than a float32 step. They are scored by
    the real `_calculate_combined_score` and ordered by the real
    `_score_and_rank_candidates`, whose `list.sort` is stable — so under the
    regression the tie resolves by insertion order and the weaker candidate keeps
    the slot. `top_k = 2` puts that tie exactly on the boundary a clinician sees.
    """
    from src.core.types import DataSource, NodeID
    from src.inference.pipeline import DiagnosisPipeline
    from src.reasoning.path_reasoning import ReasoningPath

    phenotype, _ = _ids()
    strong = NodeID(source=DataSource.MONDO, local_id="STRONG")
    lower = NodeID(source=DataSource.MONDO, local_id="LOWER")
    higher = NodeID(source=DataSource.MONDO, local_id="HIGHER")

    # Insertion order puts the *lower* candidate first, so a tie would keep it
    # ahead of the higher one. Python's sort is stable and `reverse=True` does
    # not reverse equal elements.
    embeddings = {strong: 0.9, lower: NEAR_TIE_A, higher: NEAR_TIE_B}
    all_paths = {
        str(disease): [ReasoningPath(nodes=[phenotype, disease], edges=[])]
        for disease in (strong, lower, higher)
    }

    stub = _sp_ready_pipeline_stub(lookup)
    stub.config = type("C", (), {"eta": 0.7})()
    stub.kg = type("KG", (), {"get_node": staticmethod(lambda _id: None)})()
    stub._gnn_ready = True
    stub._calculate_gnn_score = lambda sources, disease_id, patient: embeddings[disease_id]
    stub._calculate_sp_score = lambda *a, **k: 0.3
    stub._calculate_combined_score = lambda *a, **k: (
        DiagnosisPipeline._calculate_combined_score(stub, *a, **k)
    )
    stub._calculate_reasoning_score = lambda paths: 0.0
    stub._extract_supporting_genes = lambda paths: []
    stub._determine_evidence_sources = lambda paths: []

    shown = DiagnosisPipeline._score_and_rank_candidates(
        stub, all_paths, [phenotype], None, top_k=2, include_ortholog_evidence=False
    )

    assert [c.disease_id for c in shown] == [strong, higher], (
        "the higher candidate must hold the second slot; under the float32 "
        "regression the tie hands it to the lower one"
    )
    assert [c.rank for c in shown] == [1, 2]


def test_the_ranking_fixture_would_actually_catch_the_regression():
    """Pins the premise of the test above.

    The two candidates only sit on the top-k boundary if their combined scores
    genuinely collapse under float32. If a future change to eta, the SP component
    or the near-tie constants broke that, the ranking test would keep passing
    while testing nothing.
    """
    eta, sp = 0.7, 0.3
    as_f32 = [
        float(
            mix_embedding_and_sp_scores(
                torch.tensor([emb], dtype=torch.float32),
                torch.tensor([sp], dtype=torch.float32),
                eta,
            )[0]
        )
        for emb in (NEAR_TIE_A, NEAR_TIE_B)
    ]

    assert as_f32[0] == as_f32[1], "the fixture no longer exercises the regression"


def test_combined_score_without_sp_is_pure_gnn(lookup):
    """The documented degradation: no shortest-path table means effective eta=1,
    and the SP component is reported as 0.0 rather than invented."""
    from src.inference.pipeline import DiagnosisPipeline

    stub = _sp_ready_pipeline_stub(lookup)
    stub._sp_ready = False
    stub.config = type("C", (), {"eta": 0.7})()
    stub._calculate_gnn_score = lambda *a, **k: 0.8

    source, target = _ids()

    assert DiagnosisPipeline._calculate_combined_score(
        stub, [source], target, None
    ) == (0.8, 0.8, 0.0)


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
    one, so an empty row yields zeros rather than raising or producing NaN. Mode A
    exists to reproduce the legacy control, and changing this would move it off
    that control."""
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
# CUDA is a hard requirement of every deployment, so these are the configurations
# that actually run in the hospital. They cannot be exercised in this
# environment; institutional calibration verifies the deployed device and dtype.
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
