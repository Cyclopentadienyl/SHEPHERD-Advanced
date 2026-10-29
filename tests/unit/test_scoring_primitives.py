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
`7dab728`). The pipeline and the measurement harness compose these primitives.
The cosine is shared; the pool is still two deliberate implementations
(`pool_patient_embeddings`, `masked_mean_pool`), bound by
`test_masked_pool_and_served_pool_agree_on_one_unpadded_patient`.

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
    cosine_scores,
    mix_embedding_and_sp_scores,
    normalise_cosine_to_unit_interval,
    pool_patient_embeddings,
    sp_scores_from_distances,
)
from src.inference.sp_index import build_sp_index, sp_mean_distances  # noqa: E402

# Declared tolerances. Scores feed a ranking and a clinician-facing display;
# 1e-6 is far below any difference either could act on, and far above the
# summation-order noise between a loop and a reduction.
ATOL = 1e-6
RTOL = 1e-6


@pytest.fixture
def embeddings():
    torch.manual_seed(0)
    return torch.randn(40, 16)


#: The rows the `lookup` fixture is built from: a present pair, an absent pair
#: (no path within max_hops), a target that exists only as a gene, and a
#: phenotype — 2 — with no rows at all.
#:
#: **Held as rows, not read back off the fixture.** The legacy reference below
#: walks these; a reference that reached into `SPLookup` would be checking the
#: index against its own internals rather than against an independent
#: computation.
SP_ROWS = (
    # phenotype, target, target_type, distance
    (0, 5, 1, 2),
    (0, 7, 1, 4),
    (0, 9, 0, 1),
    (1, 5, 1, 3),
    (1, 8, 1, 5),
)
SP_MAX_HOPS = 5


@pytest.fixture
def lookup():
    """`SP_ROWS`, **built through the one builder** like every other lookup in
    the repository.

    Assembling the object field by field is the second construction path this
    refactor removed, and a fixture is where one would come back first: smallest
    blast radius, least chance of being noticed.
    """
    columns = list(zip(*SP_ROWS))
    return build_sp_index(
        torch.tensor(columns[0], dtype=torch.int64),
        torch.tensor(columns[1], dtype=torch.int64),
        torch.tensor(columns[2], dtype=torch.int64),
        torch.tensor(columns[3], dtype=torch.int8),
        max_hops=SP_MAX_HOPS,
    )


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
def _legacy_sp_score(phenotype_indices, target_idx, target_type_idx):
    """The formula as it was written inline in the pipeline, reimplemented over
    `SP_ROWS` so this test does not depend on the code under test being right.

    It walks rows in Python and compares exact integers. That is deliberately
    not how the served primitive works — and it is why this test would have
    caught the dtype aliasing the old tensor-comparing scan had, where a Python
    int wider than the column's dtype matched the wrong row.
    """
    unreachable = float(SP_MAX_HOPS + 1)
    total = 0.0
    for ph in phenotype_indices:
        hit = next(
            (d for p, t, y, d in SP_ROWS
             if p == ph and t == target_idx and y == target_type_idx),
            None,
        )
        total += unreachable if hit is None else float(hit)
    return 1.0 / (1.0 + total / len(phenotype_indices))


@pytest.mark.parametrize("phenotypes", [[0], [0, 1], [0, 1, 2], [2]])
@pytest.mark.parametrize("target_idx", [5, 7, 8, 99])
def test_sp_score_matches_the_legacy_formula(lookup, phenotypes, target_idx):
    distances, available, _unreachable = sp_mean_distances(lookup, phenotypes, [target_idx], 1)

    assert bool(available[0])
    assert float(sp_scores_from_distances(distances)[0]) == pytest.approx(
        _legacy_sp_score(phenotypes, target_idx, 1), abs=ATOL, rel=RTOL
    )


def test_target_type_is_part_of_the_match(lookup):
    """Target 9 exists for phenotype 0 but as a gene, not a disease. Asking for
    the disease must not find it."""
    as_disease, _, _unreachable = sp_mean_distances(lookup, [0], [9], 1)
    as_gene, _, _unreachable = sp_mean_distances(lookup, [0], [9], 0)

    assert float(as_disease[0]) == 6.0   # unreachable: max_hops + 1
    assert float(as_gene[0]) == 1.0


def test_a_phenotype_with_no_slice_counts_as_unreachable(lookup):
    """Phenotype 2 has no entry in the table at all. It contributes the
    unreachable distance rather than being silently dropped, which would let a
    candidate look closer than it is."""
    with_missing, _, _unreachable = sp_mean_distances(lookup, [0, 2], [5], 1)

    assert float(with_missing[0]) == pytest.approx((2.0 + 6.0) / 2.0)


def test_all_unreachable_is_still_a_computed_value(lookup):
    """The distinction C9 turns on: a candidate every one of whose phenotypes is
    unreachable has a real value — the largest one — and is NOT the same as a
    candidate that could not be evaluated."""
    distances, available, _unreachable = sp_mean_distances(lookup, [0, 1], [12345], 1)

    assert bool(available[0]), "unreachable is a distance, not an absence"
    assert float(distances[0]) == 6.0
    assert float(sp_scores_from_distances(distances)[0]) == pytest.approx(1.0 / 7.0)


def test_no_phenotypes_is_unavailable_not_zero_distance(lookup):
    """With nothing to measure from, there is no distance. The mask says so
    rather than the value pretending to."""
    _, available, _unreachable = sp_mean_distances(lookup, [], [5], 1)

    assert not bool(available[0])


def test_sp_batch_matches_one_at_a_time(lookup):
    targets = [5, 7, 8, 99]

    batched, _, _unreachable = sp_mean_distances(lookup, [0, 1], targets, 1)

    for position, target in enumerate(targets):
        single, _, _unreachable = sp_mean_distances(lookup, [0, 1], [target], 1)
        assert float(single[0]) == pytest.approx(float(batched[position]), abs=ATOL)


def test_computed_scores_stay_inside_the_documented_range(lookup):
    """[1/7, 1/2] at max_hops = 5 — never 0, which is why 0.0 as an
    unavailable sentinel is indistinguishable from nothing real."""
    distances, _, _unreachable = sp_mean_distances(lookup, [0, 1], [5, 7, 8, 99], 1)
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
    d1, _, _unreachable = sp_mean_distances(lookup, [0, 1], [5, 7], 1)
    d2, _, _unreachable = sp_mean_distances(lookup, [0, 1], [5, 7], 1)

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

    Replace `cosine_scores` and assert the pipeline's model term follows it. The
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
    target = NodeID(source=DataSource.MONDO, local_id="D1")

    score = DiagnosisPipeline._calculate_gnn_score(fake, torch.randn(4), target)

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


def _record(score, status="COMPUTED"):
    """A numeric SP record carrying `score`, as the mixture reads it."""
    from src.core.types import MAPPED_ONLY_DENOMINATOR, SPAnalysisRecord

    return SPAnalysisRecord(
        status=status, phenotypes_submitted=1, phenotypes_mapped=1, max_hops=5,
        mean_hop_distance=1.0 / score - 1.0 if score else 5.0, score=score,
        unreachable_pair_count=0, denominator_policy=MAPPED_ONLY_DENOMINATOR)


def test_the_one_sp_caller_routes_through_both_sp_primitives(monkeypatch, lookup):
    """`sp_analysis_records` is the one SP caller (Spec 1 §2.2): it passes the
    resolved indices to the lookup and the lookup's distances to the transform."""
    import src.inference.scoring as scoring
    import src.inference.sp_index as sp_index

    seen = {}

    def fake_distances(lk, phenotypes, targets, target_type):
        seen["called"] = (list(phenotypes), list(targets), target_type)
        return (torch.tensor([3.0], dtype=torch.float64), torch.tensor([True]),
                torch.tensor([0]))

    monkeypatch.setattr(sp_index, "sp_mean_distances", fake_distances)
    monkeypatch.setattr(scoring, "sp_scores_from_distances", lambda d: d * 100.0)

    source, target = _ids()
    stub = _sp_ready_pipeline_stub(lookup)
    [record] = sp_index.sp_analysis_records(
        lookup, stub._node_id_to_idx, [str(source)], [str(target)], 1)

    assert seen["called"] == ([0], [5], 1), "the caller must pass resolved indices"
    assert record.status == "COMPUTED"
    assert (record.mean_hop_distance, record.score) == (3.0, 300.0), (
        "and must carry what the lookup and the transform gave it")


def test_each_unavailable_sp_state_is_a_status_with_no_number(monkeypatch, lookup):
    """The legacy wrapper returned 0.0 for each of these (policy C9). Each is now
    its own status, the numbers absent, and the lookup is never asked: nothing
    was measured, so nothing can stand in for a measurement."""
    import src.inference.sp_index as sp_index

    def never(*args, **kwargs):
        raise AssertionError("the lookup was called for a candidate it cannot measure")

    monkeypatch.setattr(sp_index, "sp_mean_distances", never)
    source, target = _ids()
    mapping = _sp_ready_pipeline_stub(lookup)._node_id_to_idx
    cases = {
        "NO_TABLE": sp_index.sp_analysis_records(None, mapping, [str(source)], [str(target)], 1),
        "TARGET_UNMAPPED": sp_index.sp_analysis_records(
            lookup, mapping, [str(source)], ["mondo:ELSEWHERE"], 1),
        "NO_PHENOTYPE_MAPPED": sp_index.sp_analysis_records(
            lookup, mapping, ["hpo:HP:9999999"], [str(target)], 1),
    }
    for status, [record] in cases.items():
        assert record.status == status
        assert not record.numeric
        assert (record.mean_hop_distance, record.score, record.unreachable_pair_count,
                record.denominator_policy, record.mixed_as) == (None,) * 5
    assert cases["NO_TABLE"][0].max_hops is None
    assert cases["TARGET_UNMAPPED"][0].max_hops == lookup.max_hops
    # The phenotype no node was found for is named, never counted as mapped.
    assert cases["NO_PHENOTYPE_MAPPED"][0].unmapped_ids == ("hpo:HP:9999999",)
    assert cases["NO_PHENOTYPE_MAPPED"][0].phenotypes_mapped == 0


def test_a_partial_record_names_its_coverage(lookup):
    """COMPUTED_PARTIAL is numeric, over the mapped phenotypes only, and says how
    many of the submitted ones those are (Spec 1 §3.4). `status != COMPUTED` is
    not a test for unavailability."""
    from src.inference.sp_index import sp_analysis_records

    source, target = _ids()
    mapping = _sp_ready_pipeline_stub(lookup)._node_id_to_idx
    [full] = sp_analysis_records(lookup, mapping, [str(source)], [str(target)], 1)
    [partial] = sp_analysis_records(lookup, mapping, [str(source)], [str(target)], 1,
                                    unmapped_ids=["HP:9999999", "HP:9999999"])

    assert partial.status == "COMPUTED_PARTIAL" and partial.numeric
    assert (partial.phenotypes_mapped, partial.phenotypes_submitted) == (1, 2)
    assert partial.unmapped_ids == ("HP:9999999",)
    assert partial.score == full.score and partial.mean_hop_distance == full.mean_hop_distance


def test_the_served_score_routes_through_the_mixture(monkeypatch, lookup):
    """Routing, and the dtype the mixture is handed.

    The dtype assertion pins the *mechanism* of the float32 fix: `_served_score`
    is the only place that chooses how the two Python doubles are widened into
    tensors, and a `torch.tensor([x])` without a dtype gives float32. On its own
    it is a white-box check and would not notice a value rounded before the
    tensor was built — the two tests after this one cover that.
    """
    import src.inference.scoring as scoring
    from src.inference.pipeline import DiagnosisPipeline

    stub = _sp_ready_pipeline_stub(lookup)
    stub.config = type("C", (), {"eta": 0.7})()
    seen = {}

    def fake_mixture(embedding_scores, sp_scores, eta):
        seen["inputs"] = (embedding_scores.dtype, sp_scores.dtype,
                          float(embedding_scores[0]), float(sp_scores[0]))
        return torch.tensor([42.0])

    monkeypatch.setattr(scoring, "mix_embedding_and_sp_scores", fake_mixture)

    served = DiagnosisPipeline._served_score(stub, 0.8, _record(0.2))

    assert served == pytest.approx(42.0), "the served score must use the primitive"
    assert seen["inputs"] == (torch.float64, torch.float64, 0.8, 0.2), (
        "the components go in unchanged and widened to float64; float32 would round "
        "the incoming doubles to ~7 significant digits"
    )


def test_a_non_numeric_status_is_mixed_as_its_stated_zero(lookup):
    """The legacy ranking is kept (Spec 1 §2.1): a non-numeric status enters the
    mixture as its `mixed_as`, 0.0, exactly as the 0.0 the wrapper returned did."""
    from dataclasses import replace

    from src.core.types import SPAnalysisRecord
    from src.inference.pipeline import DiagnosisPipeline

    stub = _sp_ready_pipeline_stub(lookup)
    stub.config = type("C", (), {"eta": 0.7})()
    unmapped = replace(SPAnalysisRecord(status="TARGET_UNMAPPED", phenotypes_submitted=1,
                                        phenotypes_mapped=1, max_hops=5), mixed_as=0.0)

    assert DiagnosisPipeline._served_score(stub, 0.8, unmapped) == _legacy_mixture(0.7, 0.8, 0.0)


# ---------------------------------------------------------------------------
# Precision, at the place the defect actually occurred
# ---------------------------------------------------------------------------
# The tests in the previous section exercise `mix_embedding_and_sp_scores` with
# tensors they construct themselves as float64. They therefore say nothing about
# the production wrapper, which is where the float32 regression lived: reverting
# `pipeline.py` to `torch.tensor([emb_score])` left every one of them passing.
# The three tests below drive the wrapper itself.
def _served_score_stub(lookup, eta):
    stub = _sp_ready_pipeline_stub(lookup)
    stub.config = type("C", (), {"eta": eta})()
    return stub


@pytest.mark.parametrize("eta", [0.0, 0.3, 0.7, 1.0, 0.123456789])
def test_served_score_is_exactly_the_legacy_arithmetic(lookup, eta):
    """`_served_score` must return the Python-double result, bit for bit — not a
    value within tolerance of it.

    This is the black-box form of the dtype check above: it fails for any loss of
    precision inside the mixture, whether from the tensor dtype or from anything
    else introduced later.
    """
    import random

    from src.inference.pipeline import DiagnosisPipeline

    stub = _served_score_stub(lookup, eta)
    random.seed(20260818)
    for _ in range(2000):
        emb, sp = random.random(), random.random()

        served = DiagnosisPipeline._served_score(stub, emb, _record(sp))

        assert served == _legacy_mixture(eta, emb, sp)


def test_served_score_keeps_a_sub_float32_distinction(lookup):
    """Two candidates closer together than a float32 step stay distinct through
    the mixture. Under the regression they became an exact tie."""
    from src.inference.pipeline import DiagnosisPipeline

    stub = _served_score_stub(lookup, 0.7)
    scores = [DiagnosisPipeline._served_score(stub, emb, _record(0.3))
              for emb in (NEAR_TIE_A, NEAR_TIE_B)]

    assert scores[0] != scores[1], (
        "the mixture collapsed a real difference — this is the regression"
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


def test_the_sp_record_is_exactly_the_legacy_arithmetic(lookup):
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
    from src.inference.sp_index import sp_analysis_records

    phenotypes = _multi_phenotype_ids()
    _, target = _ids()
    mapping = {"phenotype": {str(nid): i for i, nid in enumerate(phenotypes)},
               "disease": {str(target): 5}}

    [record] = sp_analysis_records(lookup, mapping, [str(p) for p in phenotypes],
                                   [str(target)], 1)

    assert record.status == "COMPUTED"
    assert record.score == _legacy_sp_score([0, 1, 2], 5, 1)
    assert record.mean_hop_distance == 11.0 / 3.0
    # Phenotype 2 has no slice, so one pair of the three is unreachable.
    assert record.unreachable_pair_count == 1
    # Named so a change to the fixture that made the mean integral would be
    # visible rather than silently weakening the test.
    assert _legacy_sp_score([0, 1, 2], 5, 1) == 1.0 / (1.0 + 11.0 / 3.0)


@pytest.mark.parametrize("target_idx", [5, 7, 99])
def test_the_sp_record_matches_legacy_for_a_single_phenotype(lookup, target_idx):
    """The integral-mean case, kept as its own test rather than folded into the
    fractional one — the two catch different failures."""
    from src.inference.sp_index import sp_analysis_records

    source, target = _ids()
    mapping = {"phenotype": {str(source): 0}, "disease": {str(target): target_idx}}

    [record] = sp_analysis_records(lookup, mapping, [str(source)], [str(target)], 1)

    assert record.score == _legacy_sp_score([0], target_idx, 1)


def test_one_call_for_every_candidate_equals_one_call_each(lookup):
    """Batch 4 asks for every reached candidate in one call; the distances are
    sums of small integers, so each record is what a call for it alone gives."""
    from src.inference.sp_index import sp_analysis_records

    phenotypes = _multi_phenotype_ids()
    targets = ["mondo:A", "mondo:B", "mondo:C"]
    mapping = {"phenotype": {str(nid): i for i, nid in enumerate(phenotypes)},
               "disease": dict(zip(targets, [5, 7, 99]))}
    ids = [str(p) for p in phenotypes]

    together = sp_analysis_records(lookup, mapping, ids, targets, 1)
    alone = [sp_analysis_records(lookup, mapping, ids, [t], 1)[0] for t in targets]

    assert together == alone


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
    distances, _, _unreachable = sp_mean_distances(lookup, [0, 1], [5, 7, 99], 1)

    assert distances.dtype == torch.float64


def _ranking_stub(lookup, monkeypatch, embeddings, records, scoring_mode):
    """`_score_and_rank_candidates` over stubbed scores: the model terms given,
    the SP records given (by disease id), everything else inert."""
    import src.inference.sp_index as sp_index
    from src.inference.pipeline import DiagnosisPipeline

    monkeypatch.setattr(
        sp_index, "sp_analysis_records",
        lambda lookup_, mapping, phenotypes, targets, target_type, unmapped_ids=():
            [records[t] for t in targets])
    stub = _sp_ready_pipeline_stub(lookup)
    stub._sp_ready = scoring_mode == "gnn_plus_shortest_path"
    stub.config = type("C", (), {"eta": 0.7})()
    stub.kg = type("KG", (), {"get_node": staticmethod(lambda _id: None)})()
    stub._gnn_ready = True
    stub._scoring_mode = lambda: (scoring_mode, 0.7 if stub._sp_ready else 1.0)
    stub._patient_embedding = lambda rows: None
    stub._calculate_gnn_score = lambda patient, disease_id: embeddings[disease_id]
    stub._served_score = lambda term, record: DiagnosisPipeline._served_score(
        stub, term, record)
    stub._extract_supporting_genes = lambda paths: []
    stub._determine_evidence_sources = lambda paths: []
    return stub


def _paths_to(phenotype, diseases):
    from src.reasoning.path_reasoning import ReasoningPath

    return {str(d): [ReasoningPath(nodes=[phenotype, d], edges=[])] for d in diseases}


def test_ranking_path_keeps_a_sub_float32_distinction(lookup, monkeypatch):
    """The clinical consequence, through the real ranking code.

    Two candidates are separated by less than a float32 step. They are scored by
    the real `_served_score` and ordered by the real `_score_and_rank_candidates`,
    whose sort is stable — so under the regression the tie resolves by insertion
    order and the weaker candidate keeps the slot. `top_k = 2` puts that tie
    exactly on the boundary a clinician sees.
    """
    from src.core.types import DataSource, NodeID
    from src.inference.pipeline import DiagnosisPipeline

    phenotype, _ = _ids()
    strong = NodeID(source=DataSource.MONDO, local_id="STRONG")
    lower = NodeID(source=DataSource.MONDO, local_id="LOWER")
    higher = NodeID(source=DataSource.MONDO, local_id="HIGHER")

    # Insertion order puts the *lower* candidate first, so a tie would keep it
    # ahead of the higher one. Python's sort is stable and `reverse=True` does
    # not reverse equal elements.
    embeddings = {strong: 0.9, lower: NEAR_TIE_A, higher: NEAR_TIE_B}
    records = {str(d): _record(0.3) for d in embeddings}
    stub = _ranking_stub(lookup, monkeypatch, embeddings, records, "gnn_plus_shortest_path")

    ranked = DiagnosisPipeline._score_and_rank_candidates(
        stub, _paths_to(phenotype, embeddings), [phenotype], None,
        include_ortholog_evidence=False, rows=[0],
    )
    shown = ranked[:2]  # `run`'s cut at top_k = 2

    assert [c.disease_id for c in shown] == [strong, higher], (
        "the higher candidate must hold the second slot; under the float32 "
        "regression the tie hands it to the lower one"
    )
    assert [c.rank for c in shown] == [1, 2]


def test_a_mixed_status_ranks_as_the_legacy_zero_did_and_says_so(lookup, monkeypatch):
    """Forced through the ranking: a candidate whose SP status has no number is
    still mixed as 0, so its served score and its place are the legacy ones, and
    its record says `mixed_as` 0.0 (Spec 1 §2.1). A numeric record carries none."""
    from src.core.types import DataSource, NodeID, SPAnalysisRecord
    from src.inference.pipeline import DiagnosisPipeline

    phenotype, _ = _ids()
    a = NodeID(source=DataSource.MONDO, local_id="A")
    b = NodeID(source=DataSource.MONDO, local_id="B")
    records = {str(a): _record(0.3),
               str(b): SPAnalysisRecord(status="TARGET_UNMAPPED", phenotypes_submitted=1,
                                        phenotypes_mapped=1, max_hops=5)}
    stub = _ranking_stub(lookup, monkeypatch, {a: 0.6, b: 0.8}, records,
                         "gnn_plus_shortest_path")

    ranked = DiagnosisPipeline._score_and_rank_candidates(
        stub, _paths_to(phenotype, [a, b]), [phenotype], None,
        include_ortholog_evidence=False, rows=[0])
    by_id = {c.disease_id: c for c in ranked}

    assert by_id[b].served_score == _legacy_mixture(0.7, 0.8, 0.0)
    assert by_id[b].sp_term.mixed_as == 0.0 and by_id[b].sp_term.score is None
    assert by_id[a].served_score == _legacy_mixture(0.7, 0.6, 0.3)
    assert by_id[a].sp_term.mixed_as is None


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


def test_without_a_table_the_model_term_ranks_alone(lookup, monkeypatch):
    """The documented degradation, `gnn_only`: no table, so the served score is
    the model term itself, never mixed, and every SP record is NO_TABLE with no
    number — not the 0.0 the wrapper used to report."""
    from src.core.types import DataSource, NodeID
    from src.inference.pipeline import DiagnosisPipeline
    from src.inference.sp_index import sp_analysis_records

    phenotype, _ = _ids()
    disease = NodeID(source=DataSource.MONDO, local_id="D1")
    [no_table] = sp_analysis_records(None, {}, [str(phenotype)], [str(disease)], 1)
    stub = _ranking_stub(lookup, monkeypatch, {disease: 0.8}, {str(disease): no_table},
                         "gnn_only")

    [candidate] = DiagnosisPipeline._score_and_rank_candidates(
        stub, _paths_to(phenotype, [disease]), [phenotype], None,
        include_ortholog_evidence=False, rows=[0])

    assert candidate.served_score == candidate.model_term == 0.8
    assert candidate.sp_term.status == "NO_TABLE"
    assert candidate.sp_term.score is None and candidate.sp_term.mixed_as is None


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
