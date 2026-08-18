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

    import argparse

    from scripts.measure_scorer import build_manifest, load_graph_and_samples
    from src.kg.data_loader import DataLoaderConfig, create_diagnosis_dataloader

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


def test_a_cohort_smaller_than_the_manifest_claims_is_an_error(workspace):
    """Driven through the real call site rather than the helper: the manifest
    declares one more sample than the dataloader can produce, which is what a
    dropped last batch or a skipped batch would look like from the outside."""
    _, data_dir, checkpoint = workspace

    import argparse

    from scripts.measure_scorer import build_manifest, build_model, load_graph_and_samples
    from src.kg.data_loader import DataLoaderConfig, create_diagnosis_dataloader

    device = torch.device("cpu")
    graph_data, samples = load_graph_and_samples(data_dir, "test")
    loader = create_diagnosis_dataloader(
        samples=samples, graph_data=graph_data,
        config=DataLoaderConfig(batch_size=3, num_workers=0, shuffle=False),
    )
    args = argparse.Namespace(checkpoint=checkpoint, data_dir=data_dir, split="test",
                              batch_size=3, num_workers=0, seed=None)

    with pytest.raises(ValueError, match="cohort shrinkage"):
        run_mode_a(
            model=build_model(checkpoint, device), dataloader=loader,
            manifest=build_manifest(args, graph_data, len(samples) + 1, device),
            device=device,
        )


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"n_absent": 1}, "absent from the candidate set"),
        ({"n_canonical_ranks": 5}, "cohort shrinkage"),
        ({"n_legacy_rows": 5}, "cohort shrinkage"),
        ({"n_sample_ids": 5}, "cohort shrinkage"),
        ({"declared": 0}, "nothing to measure"),
    ],
)
def test_every_way_the_cohort_can_shrink_is_refused(workspace, kwargs, expected):
    """Absence gets its own message because it has its own cause. `n_absent` is
    checked before the count comparison so that a truth outside the candidate set
    is reported as what it is — a harness fault — rather than as an arithmetic
    mismatch pointing at the wrong place."""
    _, data_dir, checkpoint = workspace

    import argparse

    from scripts.measure_scorer import build_manifest, load_graph_and_samples
    from src.evaluation.measurement import _assert_cohort_is_intact

    graph_data, _ = load_graph_and_samples(data_dir, "test")
    args = argparse.Namespace(checkpoint=checkpoint, data_dir=data_dir, split="test",
                              batch_size=3, num_workers=0, seed=None)
    declared = kwargs.pop("declared", 6)
    manifest = build_manifest(args, graph_data, declared, torch.device("cpu"))

    call = {"n_legacy_rows": 6, "n_sample_ids": 6, "n_canonical_ranks": 6, "n_absent": 0}
    call.update(kwargs)

    with pytest.raises(ValueError, match=expected):
        _assert_cohort_is_intact(manifest=manifest, **call)


# ---------------------------------------------------------------------------
# The calibration artifact
# ---------------------------------------------------------------------------
def test_the_predictions_artifact_has_the_oracle_s_shape(workspace):
    """Mixed spaces on purpose. `scripts/evaluate_model.py:508-519` writes the
    **global** truth id beside **subgraph-local** prediction indices rendered as
    strings; the only job of this artifact is to diff against that one, and a
    tidier shape would not diff."""
    _, data_dir, checkpoint = workspace

    result = _run(data_dir, checkpoint)
    rows = result.to_predictions()

    assert len(rows) == result.manifest.n_samples
    assert set(rows[0]) == {"sample_id", "ground_truth", "predictions"}
    for row in rows:
        assert isinstance(row["sample_id"], str)
        assert isinstance(row["ground_truth"], int)
        assert all(isinstance(p, str) for p in row["predictions"])
        assert len(row["predictions"]) <= LEGACY_TRUNCATION_K


def test_the_report_and_the_rows_are_separate_artifacts(workspace):
    """The summary a human reads must not carry thousands of per-sample rows, and
    the rows must not be reachable only through an object that never reaches
    disk — which was the defect: `legacy_top_k_local` existed on the result and
    no file ever contained it."""
    _, data_dir, checkpoint = workspace

    result = _run(data_dir, checkpoint)
    report = result.to_dict()

    assert "legacy_top_k_local" not in report
    assert "sample_ids" not in report
    assert "sampler_evidence" in report
    assert result.to_predictions()


# ---------------------------------------------------------------------------
# Sampler evidence
# ---------------------------------------------------------------------------
def test_sampler_evidence_counts_negatives_in_global_space(workspace):
    """The batch dict is remapped to subgraph-local indices before it is handed
    over, so counting unique negatives without translating would count local
    column numbers — which repeat across batches for unrelated diseases."""
    _, data_dir, checkpoint = workspace

    result = _run(data_dir, checkpoint)
    negatives = result.sampler_evidence["negative_sampling"]

    assert negatives["observed"] is True
    assert negatives["total_drawn"] == (
        result.manifest.n_samples * result.manifest.num_negative_samples
    )
    # The fixture has four diseases, so every drawn id must be one of them.
    assert negatives["unique_global_ids"] <= 4
    assert negatives["repeat_draws_across_run"] == (
        negatives["total_drawn"] - negatives["unique_global_ids"]
    )


def test_sampler_evidence_records_what_was_scored_not_what_was_configured(workspace):
    """The manifest's claim and the observation are separate fields so they can
    disagree. A candidate universe far smaller than the configured negative count
    is exactly the discrepancy this is here to make visible."""
    _, data_dir, checkpoint = workspace

    result = _run(data_dir, checkpoint, batch_size=3)
    evidence = result.sampler_evidence

    assert evidence["n_batches"] == 2  # six samples at batch size three
    assert evidence["candidate_columns"]["min"] >= 1
    assert evidence["candidate_columns"]["max"] >= evidence["candidate_columns"]["min"]
    assert evidence["max_subgraph_nodes"]["disease"] >= evidence["candidate_columns"]["max"]


# ---------------------------------------------------------------------------
# The CLI's own guards — argument-level, not driver-level
# ---------------------------------------------------------------------------
def test_auto_device_refuses_to_fall_back_to_cpu(monkeypatch):
    """CUDA is a hard project requirement. A silent CPU fallback would produce a
    number that reads as institutional and is not."""
    from scripts import measure_scorer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit, match="requires CUDA"):
        measure_scorer._resolve_device("auto")


def test_explicit_cpu_is_permitted_but_ineligible(monkeypatch):
    """Development in a container without a GPU is real work; quoting its number
    as an acceptance result is not."""
    from scripts import measure_scorer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device, eligible = measure_scorer._resolve_device("cpu")

    assert device.type == "cpu"
    assert eligible is False


def test_artifact_digests_identify_content_not_paths(workspace, tmp_path):
    """Two files with identical content hash the same however they are named; a
    one-byte change does not."""
    import hashlib

    from scripts.measure_scorer import _file_sha256

    original = tmp_path / "a.bin"
    copy = tmp_path / "b.bin"
    altered = tmp_path / "c.bin"
    original.write_bytes(b"shepherd")
    copy.write_bytes(b"shepherd")
    altered.write_bytes(b"shepherE")

    assert _file_sha256(original) == hashlib.sha256(b"shepherd").hexdigest()
    assert _file_sha256(original) == _file_sha256(copy)
    assert _file_sha256(original) != _file_sha256(altered)
    assert _file_sha256(tmp_path / "absent.bin") is None
