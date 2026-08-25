"""
Modes B and C — the two rungs above the control.
================================================
A→B is meant to isolate **encoder scope**: same candidates, subgraph embeddings
replaced by full-graph ones. B→C is meant to isolate the **candidate universe**:
same encoder, the batch's subgraph candidates replaced by every disease.

Each of those claims is only true if the thing held constant really is constant,
so that is what is tested here — not that the modes produce numbers, but that
they produce numbers about what they say they are about.

> **This fixture cannot tell the three modes apart, by construction.** It is
> built so that sampling randomness cannot change the candidate set: every gene
> links every phenotype to every disease, so the 2-hop subgraph *is* the whole
> graph and all four diseases are always candidates. A, B and C therefore agree
> on it exactly — the encoders coincide and the candidate universes coincide.
> That is what makes the shared-traversal and cohort claims checkable here, and
> it is also why **no number from this fixture is evidence about the scorer**.
> Whether the modes actually differ is an institutional measurement on real data.

The last test in this file asserts that agreement, so the property is recorded
rather than discovered by someone reading three identical tables and concluding
something about the model.
"""
import json

import pytest

torch = pytest.importorskip("torch")

from src.evaluation.measurement import (  # noqa: E402
    ModeAResult,
    ModeResult,
    assert_constructions_agree,
    encode_full_graph,
    run_mode_c,
    run_modes_ab,
)
from tests.fixtures.synthetic_workspace import build_workspace  # noqa: E402

BATCH_SIZE = 3


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """One workspace, one model, one set of full-graph embeddings."""
    import argparse

    from scripts.measure_scorer import (
        build_legacy_mode_a_model,
        build_loader_config,
        build_manifest,
        load_legacy_mode_a_inputs,
    )
    from src.kg.data_loader import create_diagnosis_dataloader
    from src.models.gnn.shepherd_gnn import build_shepherd_model

    root = tmp_path_factory.mktemp("modes_bc")
    data_dir, checkpoint_path = build_workspace(root)
    device = torch.device("cpu")

    graph_data, samples = load_legacy_mode_a_inputs(data_dir, "test")
    args = argparse.Namespace(
        checkpoint=checkpoint_path, data_dir=data_dir, split="test",
        batch_size=BATCH_SIZE, num_workers=0, seed=None,
    )
    loader_config = build_loader_config(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    production_model = build_shepherd_model(checkpoint, graph_data, device)

    def manifest(mode, construction, n_samples=None, regime_here=False):
        """`regime_here=True` observes the caller's ambient context, which is what
        a manifest built alongside its own computation would record."""
        from src.evaluation.measurement import observe_autocast_regime

        base = build_manifest(
            args, graph_data, len(samples), device, loader_config,
            regime=observe_autocast_regime(device) if regime_here else None,
        )
        return type(base)(**{
            **{f: getattr(base, f) for f in base.__dataclass_fields__},
            "mode": mode,
            "candidate_construction": construction,
            "n_samples": len(samples) if n_samples is None else n_samples,
        })

    return {
        "data_dir": data_dir,
        "checkpoint_path": checkpoint_path,
        "graph_data": graph_data,
        "samples": samples,
        "device": device,
        "legacy_model": build_legacy_mode_a_model(checkpoint_path, device),
        "production_model": production_model,
        "encoded": encode_full_graph(production_model, graph_data, device),
        "loader": lambda: create_diagnosis_dataloader(
            samples=samples, graph_data=graph_data, config=loader_config
        ),
        "manifest": manifest,
    }


# ---------------------------------------------------------------------------
# Mode B — same candidates, different encoder
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ab(world):
    return run_modes_ab(
        model=world["legacy_model"],
        dataloader=world["loader"](),
        manifest_a=world["manifest"]("A", "per-batch 2-hop subgraph"),
        manifest_b=world["manifest"]("B", "per-batch 2-hop subgraph, full-graph encoder"),
        full_graph_embeddings=world["encoded"],
        device=world["device"],
    )


def test_a_and_b_come_out_of_one_traversal_over_the_same_cohort(ab):
    """The property the whole design of `run_modes_ab` exists for. Same patients,
    same order, same truths — so a per-sample rank comparison is meaningful."""
    result_a, result_b = ab

    assert isinstance(result_a, ModeAResult)
    assert isinstance(result_b, ModeResult) and not isinstance(result_b, ModeAResult)
    assert result_a.sample_ids == result_b.sample_ids
    assert result_a.truth_global_ids == result_b.truth_global_ids
    assert len(result_a.canonical_ranks) == len(result_b.canonical_ranks)


def test_mode_b_carries_no_legacy_metric(ab):
    """B has no frozen oracle to be compared against. A `legacy_mrr` on it would
    invite exactly the comparison that means nothing."""
    _, result_b = ab

    assert "legacy_metrics" not in result_b.to_dict()
    assert not hasattr(result_b, "legacy_top_k_local")


def test_mode_b_requires_the_embeddings_it_is_defined_by(world):
    """Without them B would fall back to the subgraph encoder and be Mode A under
    another name — a duplicate row in a comparison table, reported as a finding."""
    with pytest.raises(ValueError, match="full_graph_embeddings"):
        run_modes_ab(
            model=world["legacy_model"],
            dataloader=world["loader"](),
            manifest_a=world["manifest"]("A", "subgraph"),
            manifest_b=world["manifest"]("B", "subgraph"),
            full_graph_embeddings=None,
            device=world["device"],
        )


def test_omitting_mode_b_returns_the_mode_a_run_alone(world):
    """The calibration path is unchanged: `run_mode_a` is this case, and the
    frozen-oracle comparison must not have been perturbed by adding B."""
    result_a, result_b = run_modes_ab(
        model=world["legacy_model"],
        dataloader=world["loader"](),
        manifest_a=world["manifest"]("A", "subgraph"),
        device=world["device"],
    )

    assert result_b is None
    assert result_a.legacy_metrics


# ---------------------------------------------------------------------------
# Mode C — same encoder, every disease
# ---------------------------------------------------------------------------
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


def test_mode_c_covers_the_same_cohort_as_a(mode_c, ab):
    """B→C is only a candidate-universe comparison if the patients are the same
    patients in the same order. C reads them from the samples file; A and B get
    them through the dataloader. That they agree is checked, not assumed."""
    result_a, _ = ab

    assert mode_c.sample_ids == result_a.sample_ids
    assert mode_c.truth_global_ids == result_a.truth_global_ids


def test_mode_c_cannot_lose_a_ground_truth(mode_c):
    """Absence is impossible by construction — the truth is a disease and every
    disease is a candidate — so a non-zero count would mean the ids are wrong."""
    assert mode_c.n_ground_truth_absent == 0
    assert mode_c.n_ranked == mode_c.manifest.n_samples


def test_every_mode_serialises_without_non_finite_values(ab, mode_c):
    result_a, result_b = ab

    for result in (result_a, result_b, mode_c):
        payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        assert payload["manifest"]["mode"] in {"A", "B", "C"}


def test_per_sample_ranks_line_up_with_the_cohort(mode_c):
    """Aggregate metrics hide a cohort where half the ranks improved and half
    collapsed, so the per-sample rows are what a mode comparison reads."""
    rows = mode_c.to_ranks()

    assert len(rows) == len(mode_c.sample_ids)
    assert rows[0]["sample_id"] == mode_c.sample_ids[0]
    assert all(row["rank"] >= 1 for row in rows)


# ---------------------------------------------------------------------------
# The precondition on reading A→B at all
# ---------------------------------------------------------------------------
def test_identical_constructions_agree(world):
    """On this fixture the two loaders do produce the same model, so the guard
    passes — which is the only outcome under which A→B may be read as encoder
    scope."""
    assert_constructions_agree(world["legacy_model"], world["production_model"])


def test_a_different_architecture_is_reported_not_tolerated(world):
    """The failure mode this guard exists for. If the two constructions disagree,
    A→B is encoder scope *plus* architecture resolution, and the message names
    what differs so it can be acted on."""
    from src.models.gnn.shepherd_gnn import ShepherdGNN, ShepherdGNNConfig

    other = ShepherdGNN(
        metadata=world["production_model"].metadata,
        in_channels_dict={k: v.size(-1) for k, v in world["graph_data"]["x_dict"].items()},
        config=ShepherdGNNConfig(hidden_dim=8, num_layers=2, num_heads=2),
    )

    with pytest.raises(ValueError, match="parameter tensors differ|parameter names"):
        assert_constructions_agree(world["legacy_model"], other)


# ---------------------------------------------------------------------------
# What this fixture can and cannot show
# ---------------------------------------------------------------------------
def test_the_three_modes_agree_here_because_the_fixture_makes_them_agree(ab, mode_c, world):
    """Not a result about the scorer — a property of the fixture, asserted so it
    is not mistaken for one.

    Every gene links every phenotype to every disease, so the 2-hop subgraph is
    the whole graph and every disease is always a candidate. A's encoder scope
    equals B's and A's candidate universe equals C's, so all three *must* match.
    If this test ever fails, the fixture stopped being degenerate — and every
    other test in this file that relies on the modes being comparable needs
    re-reading before the failure is called a bug.
    """
    result_a, result_b = ab
    n_diseases = world["graph_data"]["x_dict"]["disease"].size(0)

    assert result_a.sampler_evidence["candidate_columns"]["max"] == n_diseases, (
        "the subgraph no longer covers every disease, so A and C are no longer "
        "comparable by construction"
    )
    assert result_a.canonical_ranks == result_b.canonical_ranks
    assert result_a.canonical_ranks == mode_c.canonical_ranks


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


def test_a_generator_yielding_from_inside_autocast_is_refused(world):
    """The hole an entry-only check cannot see.

    A generator that enters an autocast context and yields from inside it leaves
    that context **active while the consumer works** — verified by execution — and
    closes it between batches, so the regime can differ from batch to batch. A
    check taken once before iteration sees the context the stream was *created*
    in, not the one each forward runs in.

    Here the manifest and the entry check both see fp32 and agree. The first batch
    then arrives under bfloat16, and the per-batch check is the only thing that
    can refuse it.
    """
    def batches():
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
            for batch in world["loader"]():
                yield batch

    assert not torch.is_autocast_enabled("cpu"), "the entry check must see fp32"

    with pytest.raises(RuntimeError, match="records"):
        run_modes_ab(
            model=world["legacy_model"],
            dataloader=batches(),
            manifest_a=world["manifest"]("A", "per-batch 2-hop subgraph"),
            device=world["device"],
        )
