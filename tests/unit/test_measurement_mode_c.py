"""
Mode C — every patient scored against every disease, checked on its own terms.
==============================================================================
`run_mode_c` produces the formal Test's number (`docs/working/PLAN_TEST_RESULTS.md:289`),
so what is tested here is what that number is made of, with no other mode to
compare it against:

  - **the candidate universe is every disease**, and any disease can be the truth;
  - **the ranks are the ones the geometry gives**: a masked mean of the patient's
    phenotypes, cosine against every disease, ties to the lower id, whatever the
    chunking and the sample order;
  - **the metrics are exact** — untruncated MRR, mean rank, each hits@k and the
    key set — and the cohort is whole or the run is refused;
  - **the arithmetic is the served primitives'**: `run_mode_c` follows
    `masked_mean_pool` and `cosine_score_matrix` when they change;
  - **the regime travels with the embeddings**, and a run whose embeddings,
    manifest and scoring context disagree is refused.

Two kinds of input, for two kinds of claim. The workspace (`world`) is read the
way the CLI's Mode C reads it — `read_measurement_inputs`, `build_shepherd_model`
over the checkpoint that read, a Mode C manifest given the record built from
those reads — and it is degenerate on purpose
(`tests/fixtures/synthetic_workspace.py`), so **no rank from it is evidence about
the scorer**. Every value test therefore builds its own `EncodedGraph` by hand,
with vectors placed so each expected rank is a fact of geometry rather than the
output of a second implementation of the scorer.

Module: tests/unit/test_measurement_mode_c.py
"""
import json
import math

import pytest

torch = pytest.importorskip("torch")

from src.evaluation.measurement import (  # noqa: E402
    AutocastRegime,
    EncodedGraph,
    encode_full_graph,
    run_mode_c,
)
from tests.fixtures.synthetic_workspace import build_workspace  # noqa: E402

BATCH_SIZE = 3

#: The authoritative family, as the formal Test reports it: `mrr` under its
#: untruncated name, the mean rank, and hits at each K.
METRIC_KEYS = {
    "untruncated_mrr", "mean_rank",
    "hits@1", "hits@5", "hits@10", "hits@20", "hits@50", "hits@100",
}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """One workspace, read and built the way the CLI's Mode C reads and builds it:
    one read of each input through `read_measurement_inputs`, the model built from
    that read's checkpoint, and every manifest given the record built from those
    reads (contract M2.1, S7)."""
    import argparse

    from scripts.measure_scorer import (
        build_loader_config,
        build_manifest,
        measurement_input_digests,
        read_measurement_inputs,
    )
    from src.models.gnn.shepherd_gnn import build_shepherd_model

    root = tmp_path_factory.mktemp("mode_c")
    data_dir, checkpoint_path = build_workspace(root)
    device = torch.device("cpu")

    inputs = read_measurement_inputs(checkpoint_path, data_dir, "test", "supplied", device)
    digests = measurement_input_digests(inputs)
    graph_data, samples = inputs.graph.graph_data, inputs.samples.samples
    args = argparse.Namespace(
        checkpoint=checkpoint_path, data_dir=data_dir, split="test", cohort_kind="supplied",
        batch_size=BATCH_SIZE, num_workers=0, seed=None,
    )
    loader_config = build_loader_config(args)
    production_model = build_shepherd_model(inputs.checkpoint.checkpoint, graph_data, device)

    def manifest(mode, construction, n_samples=None, regime_here=False):
        """`regime_here=True` observes the caller's ambient context, which is what
        a manifest built alongside its own computation would record."""
        from src.evaluation.measurement import observe_autocast_regime

        return build_manifest(
            args, graph_data, len(samples) if n_samples is None else n_samples,
            device, loader_config,
            mode=mode, candidate_construction=construction,
            model_construction="production (build_shepherd_model)",
            model=production_model,
            regime=observe_autocast_regime(device) if regime_here else None,
            digests=digests,
        )

    return {
        "data_dir": data_dir,
        "checkpoint_path": checkpoint_path,
        "graph_data": graph_data,
        "samples": samples,
        "device": device,
        "production_model": production_model,
        "encoded": encode_full_graph(production_model, graph_data, device),
        "manifest": manifest,
    }


@pytest.fixture(scope="module")
def mode_c(world):
    return run_mode_c(
        full_graph_embeddings=world["encoded"],
        samples=world["samples"],
        manifest=world["manifest"]("C", "every disease in the graph"),
        device=world["device"],
        batch_size=BATCH_SIZE,
    )


def _encode_here(world):
    """Embeddings computed in the caller's ambient context."""
    return encode_full_graph(
        world["production_model"], world["graph_data"], world["device"])


# ---------------------------------------------------------------------------
# The regime travels with the embeddings
# ---------------------------------------------------------------------------
def test_mode_c_accepts_a_run_whose_embeddings_manifest_and_scoring_all_agree(world):
    """The capability the blanket refusal removed, and **all three** halves must
    be in the same regime for it.

    A first version of this test used the fixture's fp32 embeddings, built a
    bfloat16 manifest and scored under bfloat16, and asserted success. That is the
    mixed-regime run this module says a single field cannot describe — the test
    froze the defect as a specification, and the review caught it.
    """
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        result = run_mode_c(
            full_graph_embeddings=_encode_here(world),
            samples=world["samples"],
            manifest=world["manifest"](
                "C", "every disease in the graph", regime_here=True),
            device=world["device"],
            batch_size=BATCH_SIZE,
        )

    assert result.manifest.amp_enabled is True
    assert result.manifest.amp_dtype == "torch.bfloat16"


def test_fp32_embeddings_scored_under_autocast_are_refused(world):
    """Half the mixed-regime case: the embeddings predate the context the scoring
    runs in. The manifest can describe one or the other, never both."""
    fp32_encoded = _encode_here(world)
    assert fp32_encoded.regime.enabled is False, "the fixture must really be fp32"

    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        with pytest.raises(RuntimeError, match="records|computed under"):
            run_mode_c(
                full_graph_embeddings=fp32_encoded,
                samples=world["samples"],
                manifest=world["manifest"](
                    "C", "every disease in the graph", regime_here=True),
                device=world["device"],
                batch_size=BATCH_SIZE,
            )


def test_autocast_embeddings_scored_in_fp32_are_refused(world):
    """The other half, which only the embeddings' own regime can catch: the
    scoring context and the manifest agree with each other and are both wrong
    about where the numbers came from."""
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        bf16_encoded = _encode_here(world)
    assert bf16_encoded.regime.enabled is True

    with pytest.raises(RuntimeError, match="computed under"):
        run_mode_c(
            full_graph_embeddings=bf16_encoded,
            samples=world["samples"],
            manifest=world["manifest"]("C", "every disease in the graph"),
            device=world["device"],
            batch_size=BATCH_SIZE,
        )


def test_a_manifest_disagreeing_with_the_embeddings_regime_is_refused(world):
    """Stated on its own because it is the association the split-at-the-caller
    version could not check at all: everything runs in fp32 and the manifest
    claims bfloat16."""
    fp32_encoded = _encode_here(world)

    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        bf16_manifest = world["manifest"](
            "C", "every disease in the graph", regime_here=True)

    with pytest.raises(RuntimeError, match="computed under"):
        run_mode_c(
            full_graph_embeddings=fp32_encoded,
            samples=world["samples"],
            manifest=bf16_manifest,
            device=world["device"],
            batch_size=BATCH_SIZE,
        )


def test_embeddings_are_not_labelled_with_a_later_context(world):
    """**The mutation the review asked for, and the reason the regime travels.**

    The embeddings are computed under bfloat16; the manifest is constructed
    afterwards, outside that context. An implementation observing at manifest
    construction would label these embeddings fp32 — silently, since nothing
    downstream can recover the regime from the tensors.

    `encode_full_graph` observes inside itself and returns the pair, so the
    context it actually ran in is what reaches the manifest.
    """
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        encoded = encode_full_graph(
            world["production_model"], world["graph_data"], world["device"])

    # Out here it is fp32 again — and the reported regime must still be the
    # embeddings' own.
    assert encoded.regime.enabled is True
    assert encoded.regime.dtype == "torch.bfloat16"

    from src.evaluation.measurement import observe_autocast_regime

    assert observe_autocast_regime(world["device"]).enabled is False, (
        "the surrounding context must really be fp32, or this proves nothing"
    )


# ---------------------------------------------------------------------------
# What a Mode C run reports
# ---------------------------------------------------------------------------
def test_mode_c_scores_every_disease(mode_c, world):
    """The candidate universe is the graph's disease count, not a sampled subset —
    which is what the reference method does and what nothing here did before."""
    n_diseases = world["graph_data"]["x_dict"]["disease"].size(0)

    assert mode_c.sampler_evidence["candidate_columns"]["min"] == n_diseases
    assert mode_c.sampler_evidence["candidate_columns"]["max"] == n_diseases


def test_mode_c_ran_no_sampler_and_says_so(mode_c):
    """`observed: False` with a reason, rather than an empty structure that would
    read as "not recorded"."""
    negatives = mode_c.sampler_evidence["negative_sampling"]

    assert negatives["observed"] is False
    assert "every disease" in negatives["reason"]


def test_mode_c_reports_ids_and_truths_in_samples_order(mode_c, world):
    """Per-sample rows are only readable against the cohort if they come back in
    the cohort's order, and the truth recorded for each patient is the one its
    sample carries."""
    assert mode_c.sample_ids == [s.patient_id for s in world["samples"]]
    assert mode_c.truth_global_ids == [s.disease_id for s in world["samples"]]


def test_mode_c_cannot_lose_a_ground_truth(mode_c):
    """Absence is impossible by construction — the truth is a disease and every
    disease is a candidate — so a non-zero count would mean the ids are wrong."""
    assert mode_c.n_ground_truth_absent == 0
    assert mode_c.n_ranked == mode_c.manifest.n_samples


def test_mode_c_serialises_without_non_finite_values(mode_c):
    """The output is written with `allow_nan=False`; a NaN metric must fail here
    rather than in a report a reader would trust."""
    payload = json.loads(json.dumps(mode_c.to_dict(), allow_nan=False))

    assert payload["manifest"]["mode"] == "C"


def test_per_sample_ranks_line_up_with_the_cohort(mode_c):
    """Aggregate metrics hide a cohort where half the ranks improved and half
    collapsed, so the per-sample rows are what a mode comparison reads."""
    rows = mode_c.to_ranks()

    assert len(rows) == len(mode_c.sample_ids)
    assert rows[0]["sample_id"] == mode_c.sample_ids[0]
    assert all(row["rank"] >= 1 for row in rows)


def test_the_mode_c_report_and_its_rows_are_separate_artifacts(mode_c):
    """The summary a human reads must not carry thousands of per-sample rows, and
    the rows must still reach a file of their own."""
    report = mode_c.to_dict()

    assert set(report) == {
        "manifest", "authoritative_metrics", "n_ranked", "n_ground_truth_absent",
        "sampler_evidence",
    }
    assert len(mode_c.to_ranks()) == mode_c.manifest.n_samples


def test_the_mode_c_manifest_records_what_makes_the_number_mean_something(mode_c):
    manifest = mode_c.manifest

    assert manifest.mode == "C"
    assert "no eta" in manifest.score_semantics
    assert manifest.canonical_tie_policy_version
    assert manifest.graph_fingerprint
    assert manifest.torch_version
    # Recorded even when absent, so a reader can tell "no CUDA" from "not asked".
    assert manifest.device == "cpu"
    assert manifest.amp_enabled is False


def test_the_mode_c_manifest_records_the_numeric_regime_not_only_the_boolean(mode_c):
    """`amp_enabled=False` alone cannot tell an fp32 run from one whose dtype was
    never recorded. `torch_compile_wrapped` is observed from the production model
    the embeddings came from, which is unwrapped here, so `False` is an
    observation — `None` would mean nothing was observed at all."""
    assert mode_c.manifest.amp_dtype is None
    assert mode_c.manifest.torch_compile_wrapped is False


# ---------------------------------------------------------------------------
# The cohort is whole, or there is no number
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("declared_extra", [1, -1])
def test_a_cohort_other_than_the_one_the_manifest_declares_is_refused(world, declared_extra):
    """The manifest declares one sample more, or one fewer, than the samples
    handed over — what a dropped or a duplicated row would look like from the
    outside. A metric over a different cohort is not the one the manifest
    describes."""
    with pytest.raises(ValueError, match="cohort shrinkage"):
        run_mode_c(
            full_graph_embeddings=world["encoded"],
            samples=world["samples"],
            manifest=world["manifest"](
                "C", "every disease in the graph",
                n_samples=len(world["samples"]) + declared_extra),
            device=world["device"],
            batch_size=BATCH_SIZE,
        )


def test_an_empty_cohort_is_refused_rather_than_given_metrics(world):
    with pytest.raises(ValueError, match="nothing to measure"):
        run_mode_c(
            full_graph_embeddings=world["encoded"],
            samples=[],
            manifest=world["manifest"]("C", "every disease in the graph", n_samples=0),
            device=world["device"],
        )


def test_a_truth_missing_from_the_ranking_is_refused_as_absence(world, monkeypatch):
    """Absence cannot happen in Mode C — the truth is a disease and every disease
    is ranked — so it is made to happen: one truth is reported missing from the
    ranking. The run must refuse, and say absence rather than a count mismatch,
    which is what the one rank fewer than sample ids would otherwise read as."""
    import src.evaluation.measurement as measurement

    ranks_of_truth = measurement.ranks_of_truth

    def first_truth_missing(ranked, truth):
        return [None] + ranks_of_truth(ranked, truth)[1:]

    monkeypatch.setattr(measurement, "ranks_of_truth", first_truth_missing)

    with pytest.raises(ValueError, match="absent from the candidate set"):
        run_mode_c(
            full_graph_embeddings=world["encoded"],
            samples=world["samples"],
            manifest=world["manifest"]("C", "every disease in the graph"),
            device=world["device"],
            batch_size=BATCH_SIZE,
        )


# ---------------------------------------------------------------------------
# Mode C refuses ids it cannot honestly score
# ---------------------------------------------------------------------------
def _sample(patient_id, phenotype_ids, disease_id):
    from src.kg.data_loader import DiagnosisSample

    return DiagnosisSample(
        patient_id=patient_id, phenotype_ids=phenotype_ids, disease_id=disease_id
    )


@pytest.mark.parametrize("phenotypes, fragment", [
    ([-1, 0], "phenotype id -1"),
    ([0, 99999], "phenotype id 99999"),
])
def test_an_out_of_range_phenotype_is_refused_not_clamped(world, phenotypes, fragment):
    """The defect this was corrected for.

    Mode A clamps, and is right to: its out-of-range values are the dataloader's
    `-1` padding under a `False` mask, so the clamp only keeps the gather in
    bounds. In Mode C every id is a real value from a real patient with
    `mask=True`. Clamping would score phenotype -1 as phenotype 0 and 99999 as
    the last node in the graph — a real patient, a plausible rank, and the wrong
    one, indistinguishable downstream from a genuine result.
    """
    bad = [_sample("P-bad", phenotypes, 0)]

    with pytest.raises(ValueError, match=fragment):
        run_mode_c(
            full_graph_embeddings=world["encoded"],
            samples=bad,
            manifest=world["manifest"]("C", "every disease"),
            device=world["device"],
        )


def test_the_offending_patient_is_named(world):
    """An operator needs the row, not just the fact. A cohort of thousands with
    "an id is out of range" is a search; with the patient id it is a lookup."""
    bad = [_sample("P-good", [0], 0), _sample("P-offending", [0, -5], 0)]

    with pytest.raises(ValueError, match="P-offending"):
        run_mode_c(
            full_graph_embeddings=world["encoded"],
            samples=bad,
            manifest=world["manifest"]("C", "every disease"),
            device=world["device"],
        )


def test_an_out_of_range_ground_truth_is_refused(world):
    """Caught before ranking rather than surfacing later as an absence. Absence
    is Mode C's signal that the id spaces are wrong; letting a bad truth reach it
    would report the symptom instead of the cause."""
    bad = [_sample("P-truth", [0], 99999)]

    with pytest.raises(ValueError, match="ground-truth disease id 99999"):
        run_mode_c(
            full_graph_embeddings=world["encoded"],
            samples=bad,
            manifest=world["manifest"]("C", "every disease"),
            device=world["device"],
        )


def test_a_patient_with_no_phenotypes_is_refused(world):
    """Pooling nothing yields the zero vector, whose cosine against every disease
    is zero, whose ranking is an arbitrary tie-break over the whole graph. That
    is a rank, and it means nothing."""
    with pytest.raises(ValueError, match="no phenotypes"):
        run_mode_c(
            full_graph_embeddings=world["encoded"],
            samples=[_sample("P-empty", [], 0)],
            manifest=world["manifest"]("C", "every disease"),
            device=world["device"],
        )


def test_valid_ids_at_both_boundaries_are_accepted(world):
    """The negative control for the four refusals above: index 0 and the last
    node are in range, and a guard that rejected them would be worse than none.
    """
    n_phenotypes = world["encoded"].embeddings["phenotype"].size(0)
    n_diseases = world["encoded"].embeddings["disease"].size(0)
    edge = [_sample("P-edge", [0, n_phenotypes - 1], n_diseases - 1)]

    result = run_mode_c(
        full_graph_embeddings=world["encoded"],
        samples=edge,
        manifest=world["manifest"]("C", "every disease", n_samples=1),
        device=world["device"],
    )

    assert result.n_ranked == 1


@pytest.mark.parametrize("batch_size", [0, -1])
def test_a_batch_size_below_one_gives_no_result(world, batch_size):
    """Refused today without a message of its own: zero cannot step through the
    cohort and a negative step never enters it, which leaves every sample
    unscored. Either way no number comes out."""
    with pytest.raises(ValueError):
        run_mode_c(
            full_graph_embeddings=world["encoded"],
            samples=world["samples"],
            manifest=world["manifest"]("C", "every disease in the graph"),
            device=world["device"],
            batch_size=batch_size,
        )


@pytest.mark.parametrize("node_type", ["phenotype", "disease"])
def test_a_non_finite_embedding_is_refused_not_ranked(world, node_type):
    """A NaN in a row the patient reads, or in a disease row, makes a score NaN,
    and a NaN sorts unpredictably into a plausible rank. Mode C refuses the run
    rather than reporting one."""
    embeddings = {k: v.clone() for k, v in world["encoded"].embeddings.items()}
    embeddings[node_type][1] = float("nan")
    poisoned = EncodedGraph(embeddings=embeddings, regime=world["encoded"].regime)

    with pytest.raises(ValueError, match="NaN or infinity"):
        run_mode_c(
            full_graph_embeddings=poisoned,
            samples=[_sample("P-reads-row-1", [1], 0)],
            manifest=world["manifest"]("C", "every disease", n_samples=1),
            device=world["device"],
        )


# ---------------------------------------------------------------------------
# Values from geometry
# ---------------------------------------------------------------------------
# Every embedding below is written by hand. In the fan, disease `d` points at
# angle atan(d / (D - d)) from the x axis, so its cosine with the x axis falls as
# `d` grows and a patient whose only phenotype is the x axis ranks disease `d`
# at `d + 1`. Each disease's norm grows with `d` as well, so the raw dot product
# orders them differently from the cosine: a score that skipped normalising
# would not reproduce these ranks.
_FP32 = AutocastRegime(enabled=False, dtype=None)
_X_AXIS = [[1.0, 0.0]]


def _encoded(phenotypes, diseases):
    return EncodedGraph(
        embeddings={
            "phenotype": torch.tensor(phenotypes, dtype=torch.float32),
            "disease": torch.tensor(diseases, dtype=torch.float32),
        },
        regime=_FP32,
    )


def _fan(n_diseases):
    return [[(n_diseases - d) * (d + 1.0), d * (d + 1.0)] for d in range(n_diseases)]


def _patients_with_truths(truths):
    """One patient per truth, each with the x axis as its only phenotype."""
    return [_sample(f"P{t:03d}", [0], t) for t in truths]


def _run(world, encoded, samples, batch_size=32):
    return run_mode_c(
        full_graph_embeddings=encoded,
        samples=samples,
        manifest=world["manifest"](
            "C", "every disease in the knowledge graph", n_samples=len(samples)),
        device=world["device"],
        batch_size=batch_size,
    )


WIDE = 100


@pytest.fixture(scope="module")
def wide(world):
    """A hundred diseases, one patient for each as its truth, so every id is
    some patient's answer and the ranks run 1..100 — past the top-20 window and
    onto every hits@K boundary."""
    return _run(world, _encoded(_X_AXIS, _fan(WIDE)), _patients_with_truths(range(WIDE)))


def test_every_disease_is_a_candidate_and_any_can_be_the_truth(wide):
    assert wide.sampler_evidence["candidate_columns"]["min"] == WIDE
    assert wide.sampler_evidence["candidate_columns"]["max"] == WIDE
    assert wide.truth_global_ids == list(range(WIDE))
    assert wide.n_ranked == WIDE
    assert wide.n_ground_truth_absent == 0
    # Disease d is d-th from the x axis, so the last disease is ranked last.
    assert wide.canonical_ranks == list(range(1, WIDE + 1))


def test_metrics_on_ranks_one_two_and_four_are_exact(world):
    """Truths 0, 1 and 3 in a five-disease fan rank 1, 2 and 4."""
    result = _run(world, _encoded(_X_AXIS, _fan(5)), _patients_with_truths([0, 1, 3]))
    metrics = result.authoritative_metrics

    assert result.canonical_ranks == [1, 2, 4]
    assert set(metrics) == METRIC_KEYS
    assert metrics["untruncated_mrr"] == pytest.approx((1 + 1 / 2 + 1 / 4) / 3)
    assert metrics["mean_rank"] == pytest.approx((1 + 2 + 4) / 3)
    assert metrics["hits@1"] == pytest.approx(1 / 3)
    for k in (5, 10, 20, 50, 100):
        assert metrics[f"hits@{k}"] == pytest.approx(1.0), k


def test_metrics_count_every_rank_past_twenty_and_each_hits_boundary(wide):
    """Ranks 1..100: the MRR includes the eighty ranks past twenty, and each
    hits@K counts rank K itself."""
    metrics = wide.authoritative_metrics

    assert set(metrics) == METRIC_KEYS
    assert metrics["untruncated_mrr"] == pytest.approx(
        math.fsum(1 / r for r in range(1, WIDE + 1)) / WIDE)
    assert metrics["mean_rank"] == pytest.approx((WIDE + 1) / 2)
    for k in (1, 5, 10, 20, 50, 100):
        assert metrics[f"hits@{k}"] == pytest.approx(k / WIDE), k


# Even ids point along the x axis, odd ids along the y axis: two groups of
# identical rows, interleaved, so every even disease ties with every other at the
# patient's cosine 1 and every odd one at 0. The group is wide enough that an
# unstable sort reorders it.
TIED = 40


def _tied_rows():
    return [[1.0, 0.0] if d % 2 == 0 else [0.0, 1.0] for d in range(TIED)]


def _tied_rank(d):
    """Lower id first inside each group: the even group fills ranks 1..20 in id
    order, then the odd group 21..40."""
    return d // 2 + 1 if d % 2 == 0 else TIED // 2 + d // 2 + 1


@pytest.mark.parametrize("batch_size", [1, 7, TIED])
@pytest.mark.parametrize("reverse", [False, True])
def test_identical_diseases_rank_by_ascending_id_whatever_the_batching(
        world, batch_size, reverse):
    samples = _patients_with_truths(range(TIED))
    if reverse:
        samples = samples[::-1]

    result = _run(world, _encoded(_X_AXIS, _tied_rows()), samples, batch_size)

    assert dict(zip(result.sample_ids, result.canonical_ranks, strict=True)) == {
        s.patient_id: _tied_rank(s.disease_id) for s in samples
    }


def test_a_padded_phenotype_slot_does_not_reach_the_patient(world):
    """Mode C pads a short patient with phenotype 0 under a `False` mask. Here no
    patient has phenotype 0, and its row points somewhere distinctive: if the
    padded slot entered the mean, the short patient would point between its own
    phenotype and that row, and its truth would fall from first to second.
    """
    phenotypes = [
        [0.0, 0.0, 1.0],  # 0: the padding row, read by nobody
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
    ]
    diseases = [
        [1.0, 0.0, 0.0],  # 0: phenotype 1 alone
        [1.0, 0.0, 1.0],  # 1: phenotype 1 with the padding row mixed in
        [0.0, 1.0, 0.0],  # 2
        [1.0, 1.0, 0.0],  # 3: phenotypes 1 and 2
    ]
    short = _sample("P-short", [1], 0)
    long = _sample("P-long", [1, 2], 3)

    together = _run(world, _encoded(phenotypes, diseases), [short, long], batch_size=2)

    assert together.canonical_ranks == [1, 1]


# ---------------------------------------------------------------------------
# The arithmetic is the served primitives'
# ---------------------------------------------------------------------------
def test_mode_c_scores_with_the_shared_cosine(world, monkeypatch):
    """A correct primitive that Mode C did not call would prevent no drift.
    Replace `cosine_score_matrix` with one that prefers the higher id, and the
    ranks follow it — the opposite of what the fan gives."""
    import src.inference.scoring as scoring

    seen = []

    def higher_id_first(patients, candidates):
        seen.append(tuple(candidates.shape))
        return torch.arange(candidates.size(0), dtype=torch.float32).expand(
            patients.size(0), -1)

    monkeypatch.setattr(scoring, "cosine_score_matrix", higher_id_first)

    result = _run(world, _encoded(_X_AXIS, _fan(5)), _patients_with_truths([0, 1, 3]))

    assert result.canonical_ranks == [5, 4, 2]
    assert seen == [(5, 2)]


def test_mode_c_pools_with_the_shared_masked_mean(world, monkeypatch):
    """Replace `masked_mean_pool` with one that points every patient along the y
    axis, and the ranks follow it: the fan read from the other end."""
    import src.inference.scoring as scoring

    seen = []

    def along_y(embeddings, mask):
        seen.append((tuple(embeddings.shape), mask.dtype))
        return torch.tensor([0.0, 1.0]).expand(embeddings.size(0), -1)

    monkeypatch.setattr(scoring, "masked_mean_pool", along_y)

    result = _run(world, _encoded(_X_AXIS, _fan(5)), _patients_with_truths([0, 1, 3]))

    assert result.canonical_ranks == [5, 4, 2]
    assert seen == [((3, 1, 2), torch.bool)]
