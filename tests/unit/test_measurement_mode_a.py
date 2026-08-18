"""
The Mode A driver, on a synthetic workspace.
============================================
Mode A is the control the other modes are read against, so what is tested here
is that it *preserves* the legacy measurement rather than improving it, and that
it reports the two metric families as two different things.

The workspace is built so sampling randomness cannot change the answer
(`tests/fixtures/synthetic_workspace.py`). That precondition is asserted, not
assumed — if it stopped holding, everything below would fail intermittently,
which is worse than failing.
"""
import json

import pytest

torch = pytest.importorskip("torch")

from src.evaluation.measurement import (  # noqa: E402
    LEGACY_TRUNCATION_K,
    run_mode_a,
)
from tests.fixtures.synthetic_workspace import (  # noqa: E402
    assert_candidate_universe_is_stable,
    build_workspace,
)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("mode_a")
    data_dir, checkpoint = build_workspace(root)
    return root, data_dir, checkpoint


def _run(data_dir, checkpoint, batch_size=3):
    """Drive the harness the way the CLI does, without the subprocess."""
    import argparse

    from scripts.measure_scorer import build_manifest, build_model, load_graph_and_samples
    from src.kg.data_loader import DataLoaderConfig, create_diagnosis_dataloader

    device = torch.device("cpu")
    graph_data, samples = load_graph_and_samples(data_dir, "test")
    model = build_model(checkpoint, device)
    loader = create_diagnosis_dataloader(
        samples=samples,
        graph_data=graph_data,
        config=DataLoaderConfig(batch_size=batch_size, num_workers=0, shuffle=False),
    )
    args = argparse.Namespace(
        checkpoint=checkpoint, data_dir=data_dir, split="test",
        batch_size=batch_size, num_workers=0, seed=None,
    )
    manifest = build_manifest(args, graph_data, len(samples), device)
    return run_mode_a(model=model, dataloader=loader, manifest=manifest, device=device)


# ---------------------------------------------------------------------------
# The fixture's own precondition
# ---------------------------------------------------------------------------
def test_the_candidate_universe_does_not_move_with_the_seed(workspace):
    _, data_dir, _ = workspace

    assert_candidate_universe_is_stable(data_dir, "test")


# ---------------------------------------------------------------------------
# What Mode A reports
# ---------------------------------------------------------------------------
def test_mode_a_reports_both_metric_families(workspace):
    _, data_dir, checkpoint = workspace

    result = _run(data_dir, checkpoint)

    assert f"legacy_mrr_truncated_at_{LEGACY_TRUNCATION_K}" in result.legacy_metrics
    assert {"untruncated_mrr", "mean_rank"} <= set(result.authoritative_metrics)
    # Kept apart in the type, because only one of them is comparable across modes.
    assert not set(result.legacy_metrics) & set(result.authoritative_metrics)


def test_every_sample_is_ranked_and_none_is_absent(workspace):
    """In Mode A the truth is a subgraph seed, so it is always a candidate. A
    non-zero absence count would mean the harness is wrong, not the model."""
    _, data_dir, checkpoint = workspace

    result = _run(data_dir, checkpoint)

    assert result.n_ranked == result.manifest.n_samples
    assert result.n_ground_truth_absent == 0


def test_mean_rank_is_now_computable(workspace):
    """The metric `generate_report` had to emit as null, because a truncated
    prediction list cannot produce it."""
    _, data_dir, checkpoint = workspace

    result = _run(data_dir, checkpoint)

    assert result.authoritative_metrics["mean_rank"] >= 1.0


def test_legacy_top_k_is_local_and_truncated(workspace):
    """The oracle's observable artifact: subgraph-local column indices, cut at 20.
    Local because the oracle never persists the mapping needed to translate."""
    _, data_dir, checkpoint = workspace

    result = _run(data_dir, checkpoint)

    assert len(result.legacy_top_k_local) == result.manifest.n_samples
    for row in result.legacy_top_k_local:
        assert len(row) <= LEGACY_TRUNCATION_K
        assert all(isinstance(i, int) and i >= 0 for i in row)


def test_batch_size_changes_the_measurement(workspace):
    """Recorded in the manifest as semantics, not as a performance knob: Mode A's
    candidate universe is the batch's subgraph, so a different batch size is a
    different measurement — and the manifest has to say which one was made."""
    _, data_dir, checkpoint = workspace

    assert _run(data_dir, checkpoint, batch_size=3).manifest.batch_size == 3
    assert _run(data_dir, checkpoint, batch_size=6).manifest.batch_size == 6


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------
def test_the_manifest_records_what_makes_the_number_mean_something(workspace):
    _, data_dir, checkpoint = workspace

    manifest = _run(data_dir, checkpoint).manifest

    assert manifest.mode == "A"
    assert manifest.legacy_truncation_k == LEGACY_TRUNCATION_K
    assert "no eta" in manifest.score_semantics
    assert manifest.canonical_tie_policy_version
    assert manifest.graph_fingerprint
    assert manifest.torch_version
    # Recorded even when absent, so a reader can tell "no CUDA" from "not asked".
    assert manifest.device == "cpu"
    assert manifest.amp_enabled is False


def test_the_result_serialises_without_non_finite_values(workspace):
    """The output is written with `allow_nan=False`; a NaN metric must fail here
    rather than in a report a reader would trust."""
    _, data_dir, checkpoint = workspace

    payload = json.dumps(_run(data_dir, checkpoint).to_dict(), allow_nan=False)

    assert json.loads(payload)["manifest"]["mode"] == "A"


# ---------------------------------------------------------------------------
# Failure behaviour
# ---------------------------------------------------------------------------
def test_a_model_producing_no_embeddings_is_an_error_not_a_skip(workspace):
    """The legacy evaluator `continue`s past a batch with no embeddings, which
    silently shrinks the cohort. Mode A refuses instead."""
    _, data_dir, checkpoint = workspace

    from scripts.measure_scorer import build_manifest, build_model, load_graph_and_samples
    from src.kg.data_loader import DataLoaderConfig, create_diagnosis_dataloader

    import argparse

    device = torch.device("cpu")
    graph_data, samples = load_graph_and_samples(data_dir, "test")
    loader = create_diagnosis_dataloader(
        samples=samples, graph_data=graph_data,
        config=DataLoaderConfig(batch_size=3, num_workers=0, shuffle=False),
    )
    args = argparse.Namespace(checkpoint=checkpoint, data_dir=data_dir, split="test",
                              batch_size=3, num_workers=0, seed=None)

    class _Empty:
        def eval(self): return self
        def __call__(self, *a, **k): return {}

    with pytest.raises(ValueError, match="no disease or phenotype embeddings"):
        run_mode_a(
            model=_Empty(), dataloader=loader,
            manifest=build_manifest(args, graph_data, len(samples), device), device=device,
        )
