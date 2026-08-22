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

    from scripts.measure_scorer import (
        build_legacy_mode_a_model,
        build_loader_config,
        build_manifest,
        load_legacy_mode_a_inputs,
    )
    from src.kg.data_loader import create_diagnosis_dataloader

    device = torch.device("cpu")
    graph_data, samples = load_legacy_mode_a_inputs(data_dir, "test")
    model = build_legacy_mode_a_model(checkpoint, device)
    args = argparse.Namespace(
        checkpoint=checkpoint, data_dir=data_dir, split="test",
        batch_size=batch_size, num_workers=0, seed=None,
    )
    # One config object to both consumers, exactly as the CLI does it — otherwise
    # this helper would be testing a wiring the CLI does not use.
    loader_config = build_loader_config(args)
    loader = create_diagnosis_dataloader(
        samples=samples, graph_data=graph_data, config=loader_config
    )
    # `model=` too, because this helper claims to drive the harness the way the
    # CLI does and the CLI now passes it: `torch_compiled` is observed from the
    # model that runs, so a helper omitting it would record 'not observed' and
    # quietly stop mirroring the thing it exists to mirror.
    manifest = build_manifest(
        args, graph_data, len(samples), device, loader_config, model=model
    )
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


def test_the_manifest_records_the_numeric_regime_not_only_the_boolean(workspace):
    """Backlog item D2. `amp_enabled=False` alone cannot distinguish an fp32 run
    from one whose dtype was never recorded, and BACKLOG §3.1.3 established that
    the AMP regime decides which question a comparison answered.

    `torch_compiled` is an **execution** fact: the project carries a compile toggle
    in `src/config/training_fields.py`, and what has to be recorded is what ran,
    not what was asked for. `_run` builds an uncompiled model, so `False` here is
    an observation — `None` would mean nothing was observed at all, which is a
    different claim.
    """
    _, data_dir, checkpoint = workspace

    manifest = _run(data_dir, checkpoint).manifest

    assert manifest.amp_dtype is None
    assert manifest.torch_compiled is False


def test_measuring_inside_an_autocast_block_is_refused(workspace):
    """The manifest's `amp_enabled=False` is a structural claim about this module
    — no traversal here opens an autocast context. A caller who wrapped the run in
    one would shift every score while the manifest went on recording fp32.

    So the claim is enforced, not asserted in prose. The refusal is the mutation
    check for it: this is exactly the state that would otherwise be misrecorded.
    """
    _, data_dir, checkpoint = workspace

    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        with pytest.raises(RuntimeError, match="autocast is enabled"):
            _run(data_dir, checkpoint)


# ---------------------------------------------------------------------------
# Observing the execution state, as opposed to reading the requested config
# ---------------------------------------------------------------------------
def test_a_compiled_model_is_observed_as_compiled():
    """Against a real `torch.compile` wrapper, not a stand-in with the right
    attribute — a stand-in would test the test."""
    import torch.nn as nn

    from src.evaluation.measurement import observe_torch_compiled

    plain = nn.Linear(4, 4)

    assert observe_torch_compiled(plain) is False
    assert observe_torch_compiled(torch.compile(plain)) is True


def test_no_model_is_not_observed_rather_than_not_compiled():
    """`None` and `False` are different claims and must not be flattened."""
    from src.evaluation.measurement import observe_torch_compiled

    assert observe_torch_compiled(None) is None


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

    from scripts.measure_scorer import (
        build_loader_config,
        build_manifest,
        load_legacy_mode_a_inputs,
    )
    from src.kg.data_loader import create_diagnosis_dataloader

    device = torch.device("cpu")
    graph_data, samples = load_legacy_mode_a_inputs(data_dir, "test")
    args = argparse.Namespace(checkpoint=checkpoint, data_dir=data_dir, split="test",
                              batch_size=3, num_workers=0, seed=None)
    loader_config = build_loader_config(args)
    loader = create_diagnosis_dataloader(
        samples=samples, graph_data=graph_data, config=loader_config
    )

    class _Empty:
        def eval(self): return self
        def __call__(self, *a, **k): return {}

    with pytest.raises(ValueError, match="no disease or phenotype embeddings"):
        run_mode_a(
            model=_Empty(), dataloader=loader,
            manifest=build_manifest(args, graph_data, len(samples), device, loader_config),
            device=device,
        )


def test_a_cohort_smaller_than_the_manifest_claims_is_an_error(workspace):
    """Driven through the real call site rather than the helper: the manifest
    declares one more sample than the dataloader can produce, which is what a
    dropped last batch or a skipped batch would look like from the outside."""
    _, data_dir, checkpoint = workspace

    import argparse

    from scripts.measure_scorer import (
        build_legacy_mode_a_model,
        build_loader_config,
        build_manifest,
        load_legacy_mode_a_inputs,
    )
    from src.kg.data_loader import create_diagnosis_dataloader

    device = torch.device("cpu")
    graph_data, samples = load_legacy_mode_a_inputs(data_dir, "test")
    args = argparse.Namespace(checkpoint=checkpoint, data_dir=data_dir, split="test",
                              batch_size=3, num_workers=0, seed=None)
    loader_config = build_loader_config(args)
    loader = create_diagnosis_dataloader(
        samples=samples, graph_data=graph_data, config=loader_config
    )

    with pytest.raises(ValueError, match="cohort shrinkage"):
        run_mode_a(
            model=build_legacy_mode_a_model(checkpoint, device), dataloader=loader,
            manifest=build_manifest(args, graph_data, len(samples) + 1, device, loader_config),
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

    from scripts.measure_scorer import (
        build_loader_config,
        build_manifest,
        load_legacy_mode_a_inputs,
    )
    from src.evaluation.measurement import _assert_cohort_is_intact

    graph_data, _ = load_legacy_mode_a_inputs(data_dir, "test")
    args = argparse.Namespace(checkpoint=checkpoint, data_dir=data_dir, split="test",
                              batch_size=3, num_workers=0, seed=None)
    declared = kwargs.pop("declared", 6)
    manifest = build_manifest(
        args, graph_data, declared, torch.device("cpu"), build_loader_config(args)
    )

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

    from scripts.measure_scorer import file_sha256

    original = tmp_path / "a.bin"
    copy = tmp_path / "b.bin"
    altered = tmp_path / "c.bin"
    original.write_bytes(b"shepherd")
    copy.write_bytes(b"shepherd")
    altered.write_bytes(b"shepherE")

    assert file_sha256(original) == hashlib.sha256(b"shepherd").hexdigest()
    assert file_sha256(original) == file_sha256(copy)
    assert file_sha256(original) != file_sha256(altered)
    assert file_sha256(tmp_path / "absent.bin") is None


# ---------------------------------------------------------------------------
# Oracle parity on padded phenotype ids
# ---------------------------------------------------------------------------
def test_padded_phenotype_ids_are_clamped_the_way_the_oracle_clamps(workspace):
    """Mode A must reproduce the frozen evaluator's index semantics, not a
    cancellation that happens to give the same answer.

    `diagnosis_collate_fn` pads phenotype ids with `-1` and `_remap_indices`
    leaves those positions at `-1`, so the oracle clamps before gathering and
    reads **row 0** for every padded slot. Indexing with `-1` instead reads the
    **last** row through Python negative indexing. For ordinary finite embeddings
    the mask multiplies both away and the pooled vector is the same — which is
    exactly why this test puts a `NaN` in the last phenotype row. `NaN * 0` is
    `NaN`, so the difference between the two operations stops being invisible:
    with the clamp the run completes, without it the score matrix is non-finite
    and `canonical_ranking` refuses it.

    A regression here is not cosmetic. Mode A is the control the whole ladder is
    read against, and a control that performs a *different* gather from the
    oracle is not one.
    """
    import argparse

    from scripts.measure_scorer import (
        build_loader_config,
        build_manifest,
        load_legacy_mode_a_inputs,
    )
    from src.kg.data_loader import DiagnosisSample, create_diagnosis_dataloader

    _, data_dir, checkpoint = workspace
    device = torch.device("cpu")
    graph_data, _ = load_legacy_mode_a_inputs(data_dir, "test")

    # Variable-length patients, so the collate pads and the padding is -1. The
    # phenotypes used are 0 and 1; phenotype 2 still enters the subgraph through
    # the 2-hop expansion, so it is the last row and **no patient reads it**.
    # That is what makes the last row reachable only by an unclamped -1.
    samples = [
        DiagnosisSample(patient_id="P-one", phenotype_ids=[0], disease_id=0),
        DiagnosisSample(patient_id="P-two", phenotype_ids=[0, 1], disease_id=1),
    ]
    args = argparse.Namespace(checkpoint=checkpoint, data_dir=data_dir, split="test",
                              batch_size=2, num_workers=0, seed=None)
    loader_config = build_loader_config(args)
    loader = create_diagnosis_dataloader(
        samples=samples, graph_data=graph_data, config=loader_config
    )
    batch_data = next(iter(loader))
    local_ids = batch_data["batch"]["phenotype_ids"]
    mask = batch_data["batch"]["phenotype_mask"]
    last_row = batch_data["subgraph_x_dict"]["phenotype"].size(0) - 1
    used = {int(i) for row, keep in zip(local_ids.tolist(), mask.tolist())
            for i, k in zip(row, keep) if k}

    # The test's own preconditions. If either stops holding it proves nothing,
    # and a green run would be worse than a red one.
    assert (local_ids == -1).any(), "no -1 padding in this batch"
    assert last_row not in used, (
        f"the last subgraph phenotype row {last_row} is read by a real patient, so "
        "it cannot distinguish the clamp from negative indexing"
    )

    class _NaNInLastPhenotypeRow:
        """Finite everywhere the oracle reads, NaN where only an unclamped -1 goes."""

        def eval(self):
            return self

        def __call__(self, x_dict, edge_index_dict):
            phenotypes = torch.ones(x_dict["phenotype"].size(0), 4)
            phenotypes[-1] = float("nan")
            diseases = torch.arange(
                1, x_dict["disease"].size(0) * 4 + 1, dtype=torch.float32
            ).reshape(x_dict["disease"].size(0), 4)
            return {"phenotype": phenotypes, "disease": diseases}

    manifest = build_manifest(args, graph_data, len(samples), device, loader_config)

    result = run_mode_a(
        model=_NaNInLastPhenotypeRow(),
        dataloader=create_diagnosis_dataloader(
            samples=samples, graph_data=graph_data, config=loader_config
        ),
        manifest=manifest,
        device=device,
    )

    assert result.n_ranked == len(samples)
    assert all(rank >= 1 for rank in result.canonical_ranks)
