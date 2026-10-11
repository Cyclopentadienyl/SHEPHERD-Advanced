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
        checkpoint=checkpoint_path, data_dir=data_dir, split="test", cohort_kind="supplied",
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
    """B has no legacy counterpart; legacy metrics are Mode A's alone. A
    `legacy_mrr` on it would invite exactly the comparison that means nothing."""
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
    """Mode A's path is unchanged: `run_mode_a` is this case, and adding B must
    not perturb it."""
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


def test_mode_c_covers_the_same_cohort_as_a(mode_c, ab):
    """B→C is only a candidate-universe comparison if the patients are the same
    patients in the same order. C reads them from the samples file; A and B get
    them through the dataloader. That they agree is checked, not assumed."""
    result_a, _ = ab

    assert mode_c.sample_ids == result_a.sample_ids
    assert mode_c.truth_global_ids == result_a.truth_global_ids


def test_every_mode_serialises_without_non_finite_values(ab, mode_c):
    result_a, result_b = ab

    for result in (result_a, result_b, mode_c):
        payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        assert payload["manifest"]["mode"] in {"A", "B", "C"}


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
