"""
Synthetic legacy equivalence, through the real calibration procedure — NOT
calibration.
=========================================================================
`scripts/calibrate_mode_a.py` is driven end to end against a synthetic workspace:
it runs the frozen evaluator and `scripts/measure_scorer.py` as subprocesses on
one seeded stream, and compares **the files they write** rather than an in-memory
object. Both CLIs are therefore exercised, including the parts a direct call to
`run_mode_a` skips — argument parsing, artifact identity, the device gate, and
the predictions writer.

Comparing the written files matters more than it sounds. A result object can
carry a field that `to_dict()` never emits, and an in-memory comparison would
still pass while the artifact a reviewer actually receives is missing it. That is
a defect this harness has already had.

> **This is not institutional calibration and must never be reported as such.**
> It shows the procedure runs and the two agree on a synthetic graph in this
> container, on CPU. `calibration.json` says so itself: `cuda_executed` is false, and
> `institutional_acceptance` says in writing that no file here can establish it.
> Mode A is calibrated only by the institutional run — real data, a real
> checkpoint, CUDA hardware.

Two subprocesses, launched by the launcher that institutional operators will run.
No test-only orchestration.
"""
import json

import pytest

torch = pytest.importorskip("torch")

from src.evaluation.measurement import LEGACY_TRUNCATION_K  # noqa: E402
from tests.fixtures.synthetic_workspace import build_workspace  # noqa: E402

BATCH_SIZE = 3
SEED = 20260818
RUN_TIMEOUT_SECONDS = 900


@pytest.fixture(scope="module")
def calibration(tmp_path_factory):
    """Run the launcher once and hand back everything it and both CLIs wrote."""
    from scripts.calibrate_mode_a import main

    root = tmp_path_factory.mktemp("equivalence")
    data_dir, checkpoint = build_workspace(root)
    workdir = root / "calibration"

    exit_code = main([
        "--checkpoint", str(checkpoint),
        "--data-dir", str(data_dir),
        "--split", "test",
        "--workdir", str(workdir),
        "--seed", str(SEED),
        "--batch-size", str(BATCH_SIZE),
        "--device", "cpu",
        "--timeout", str(RUN_TIMEOUT_SECONDS),
    ])

    def read(name: str):
        path = workdir / name
        assert path.exists(), f"{name} was not written"
        return json.loads(path.read_text())

    return {
        "exit_code": exit_code,
        "verdict": read("calibration.json"),
        "oracle_report": read("oracle_report.json"),
        "oracle_predictions": read("predictions_test.json"),
        "measurement": read("measurement.json"),
        "harness_predictions": read("measurement_predictions.json"),
    }


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------
def test_the_calibration_procedure_passes(calibration):
    """The whole point: the launcher runs both CLIs and finds no disagreement."""
    assert calibration["exit_code"] == 0, calibration["verdict"]["failures"]
    assert calibration["verdict"]["comparison_passed"] is True
    assert calibration["verdict"]["failures"] == []


def test_aggregate_legacy_mrr_matches_the_oracle(calibration):
    """The oracle's `mrr` is computed over its top-20 truncated lists, so it is
    the legacy truncated metric under a different name."""
    oracle = calibration["oracle_report"]["metrics"]["mrr"]
    harness = calibration["measurement"]["legacy_metrics"][
        f"legacy_mrr_truncated_at_{LEGACY_TRUNCATION_K}"
    ]

    assert harness == pytest.approx(oracle, abs=1e-12)


def test_the_two_predictions_files_are_identical(calibration):
    """Not a prefix comparison. Both sides truncate at the same K, so the rows are
    either the same artifact or the run measured something else — and `sample_id`
    and `ground_truth` are compared too, because a matching prediction list under
    a mismatched id would be a coincidence, not agreement."""
    assert calibration["harness_predictions"] == calibration["oracle_predictions"]


# ---------------------------------------------------------------------------
# What the harness writes that the oracle cannot
# ---------------------------------------------------------------------------
def test_the_untruncated_metric_is_not_claimed_to_match(calibration):
    """They are different quantities and must not be conflated. Here the
    candidate set is smaller than 20 so they coincide numerically — which is
    exactly why the assertion is about the report's shape, not its value: the
    oracle has no untruncated metric to compare against at all."""
    report = calibration["oracle_report"]
    measurement = calibration["measurement"]

    assert "untruncated_mrr" in measurement["authoritative_metrics"]
    assert "untruncated_mrr" not in report["metrics"]
    assert report["metrics"]["mean_rank"] is None
    assert measurement["authoritative_metrics"]["mean_rank"] is not None


def test_the_measurement_artifact_carries_the_sampler_evidence(calibration):
    """The manifest states what was configured; this states what the sampler did.
    Only the pair is evidence — a manifest claiming a candidate construction
    beside an observation that contradicts it is what this is here to expose."""
    evidence = calibration["measurement"]["sampler_evidence"]
    manifest = calibration["measurement"]["manifest"]

    assert evidence["n_batches"] >= 1
    assert evidence["candidate_columns"]["max"] >= 1
    assert set(evidence["max_subgraph_nodes"]) <= {"phenotype", "gene", "disease"}

    negatives = evidence["negative_sampling"]
    assert negatives["observed"] is True
    assert negatives["total_drawn"] == manifest["n_samples"] * manifest["num_negative_samples"]
    assert negatives["unique_global_ids"] <= negatives["total_drawn"]


def test_every_consumed_file_is_identified_by_content(calibration):
    """Paths are not identities. `checkpoints/best.pt` names a different file after
    every improvement, and the structural fingerprint is shared by every checkpoint
    trained on the same graph."""
    digests = calibration["measurement"]["manifest"]["artifact_digests"]

    for role in ("checkpoint", "samples", "node_features", "edge_indices", "num_nodes"):
        assert len(digests[role]) == 64, f"{role} has no sha256"
    assert digests["checkpoint"] != digests["samples"]
    assert calibration["verdict"]["artifact_digests"] == digests


def test_the_cohort_is_whole(calibration):
    """No absences, no shrinkage. In Mode A the truth is a subgraph seed, so an
    absence would mean the harness is wrong — and a metric over a silently
    reduced cohort answers a question nobody asked."""
    measurement = calibration["measurement"]

    assert measurement["n_ground_truth_absent"] == 0
    assert measurement["n_ranked"] == measurement["manifest"]["n_samples"]
    assert len(calibration["harness_predictions"]) == measurement["manifest"]["n_samples"]


# ---------------------------------------------------------------------------
# The claim guard
# ---------------------------------------------------------------------------
def test_this_run_is_not_calibration(calibration):
    """A guard against the claim rather than the code. Nothing here touches real
    data, a real checkpoint or CUDA, so nothing here clears the acceptance gate —
    and both the manifest and the verdict have to say so in writing."""
    verdict = calibration["verdict"]
    manifest = calibration["measurement"]["manifest"]

    assert verdict["comparison_passed"] is True
    assert verdict["cuda_executed"] is False, (
        "a CPU run reported itself as having executed on CUDA; every assertion "
        "above would still pass while the recorded claim was false"
    )
    assert manifest["cuda_executed"] is False
    assert manifest["device"] == "cpu"
    assert manifest["n_samples"] < 100


def test_the_verdict_claims_only_what_it_can_check(calibration):
    """The two booleans are named for the narrow, machine-checkable facts they
    are. Neither is institutional acceptance — a synthetic workspace on a CUDA
    machine would satisfy both — so the verdict says in writing that acceptance is
    external, and records the two things a person needs in order to give it."""
    verdict = calibration["verdict"]

    assert set(verdict) >= {
        "comparison_passed", "cuda_executed", "institutional_acceptance",
        "deployment_host", "artifact_digests",
    }
    assert "EXTERNAL" in verdict["institutional_acceptance"]
    assert verdict["deployment_host"]
    # The old names are gone, not aliased. An alias would let a stale reader keep
    # reading the overclaiming field.
    assert "passed" not in verdict and "gate_eligible" not in verdict


def test_the_manifest_records_the_configured_ceiling_not_only_the_observed_one(calibration):
    """A run that never approached the subgraph cap and a run truncated by it look
    identical in the observation alone."""
    manifest = calibration["measurement"]["manifest"]
    observed = calibration["measurement"]["sampler_evidence"]["max_subgraph_nodes"]

    assert manifest["max_subgraph_nodes"] == 5000  # DataLoaderConfig default
    assert max(observed.values()) < manifest["max_subgraph_nodes"], (
        "the fixture is supposed to sit far below the cap; if it does not, the two "
        "fields no longer demonstrate anything different from each other"
    )


# ---------------------------------------------------------------------------
# Artifact identity across the two runs
# ---------------------------------------------------------------------------
def test_an_artifact_changing_between_the_runs_is_a_failure(calibration, tmp_path):
    """The digests are taken before the oracle starts and again after the harness
    finishes. A checkpoint swapped between the two runs would otherwise surface as
    an unexplained row mismatch, pointing the reader at the scorer instead of at
    the file that moved."""
    import argparse

    from scripts.calibrate_mode_a import compare

    before = dict(calibration["verdict"]["artifact_digests"])
    after = dict(before, checkpoint="0" * 64)
    args = argparse.Namespace(batch_size=BATCH_SIZE)

    failures = compare(
        calibration["oracle_report"], calibration["oracle_predictions"],
        calibration["measurement"], calibration["harness_predictions"],
        args, before, after,
    )

    assert any("artifacts changed during the run" in f and "checkpoint" in f for f in failures)


def test_a_harness_manifest_disagreeing_with_the_before_image_is_a_failure(calibration):
    """The harness hashes its own inputs *after* loading them, so its manifest is
    checked against the pre-run image rather than trusted on its own."""
    import argparse

    from scripts.calibrate_mode_a import compare

    before = dict(calibration["verdict"]["artifact_digests"], samples="0" * 64)
    args = argparse.Namespace(batch_size=BATCH_SIZE)

    failures = compare(
        calibration["oracle_report"], calibration["oracle_predictions"],
        calibration["measurement"], calibration["harness_predictions"],
        args, before, before,
    )

    assert any("harness manifest's digests do not match" in f for f in failures)


def test_the_unperturbed_comparison_reports_no_digest_failure(calibration):
    """The negative control for the two above: with the real digests on both
    sides, neither check fires. Without this, a compare() that always complained
    would satisfy both."""
    import argparse

    from scripts.calibrate_mode_a import compare

    digests = calibration["verdict"]["artifact_digests"]
    failures = compare(
        calibration["oracle_report"], calibration["oracle_predictions"],
        calibration["measurement"], calibration["harness_predictions"],
        argparse.Namespace(batch_size=BATCH_SIZE), digests, digests,
    )

    assert failures == []
