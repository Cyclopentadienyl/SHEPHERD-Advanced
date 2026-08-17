"""
The scoring primitives are the single authority, and they must agree with what
they replaced.
=============================================================================
`src/inference/scoring.py` exists because the score a clinician sees and the
score an offline evaluation reports were computed by two separate
implementations of the same formulas. Two implementations drift; when they
drift, the evaluation stops describing the thing being evaluated.

That makes two properties worth testing, and they are different:

  - **Agreement.** The primitives compute what the previous inline code
    computed. Each test below reimplements the legacy formula *independently*
    rather than calling the primitive twice — a test that calls the thing it is
    testing to produce its own expectation cannot detect a wrong formula.
  - **Authority.** The pipeline actually routes through the primitives. A
    correct primitive that nobody calls prevents nothing, so the last test
    replaces a primitive and asserts the pipeline's answer follows it.

Tolerances are declared rather than assumed: floating-point summation order
differs between a scalar loop and a batched reduction, so bit-identity is not
achievable and asserting it would produce a flaky test or a disabled one.
Ranking, unlike arithmetic, must match exactly.
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
