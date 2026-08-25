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


# ---------------------------------------------------------------------------
# The CLI runs the ladder, and Mode A's artifacts are untouched by it
# ---------------------------------------------------------------------------
def test_the_cli_runs_all_three_modes_without_disturbing_the_calibration_artifact(
    tmp_path,
):
    """Adding modes B and C must not change what the calibration reads.

    Mode A keeps `--output`'s filename and the predictions artifact; B and C sit
    beside them. If that ever stops being true, the launcher above is comparing
    the frozen oracle against a file that is no longer Mode A.
    """
    from scripts.measure_scorer import main
    from tests.fixtures.synthetic_workspace import build_workspace as build

    data_dir, checkpoint = build(tmp_path / "ws")
    output = tmp_path / "run" / "measurement.json"

    exit_code = main([
        "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
        "--split", "test", "--output", str(output),
        "--batch-size", str(BATCH_SIZE), "--num-workers", "0",
        "--device", "cpu", "--modes", "A,B,C",
    ])

    assert exit_code == 0
    written = {p.name for p in output.parent.iterdir()}
    assert {"measurement.json", "measurement_predictions.json",
            "measurement_modeB.json", "measurement_modeC.json"} <= written

    modes = {}
    for name, path in [("A", "measurement.json"), ("B", "measurement_modeB.json"),
                       ("C", "measurement_modeC.json")]:
        modes[name] = json.loads((output.parent / path).read_text())

    assert [modes[m]["manifest"]["mode"] for m in "ABC"] == ["A", "B", "C"]
    assert modes["A"]["manifest"]["model_construction"].startswith("frozen evaluator")
    assert modes["B"]["manifest"]["model_construction"].startswith("production")
    assert modes["C"]["manifest"]["candidate_construction"] == (
        "every disease in the knowledge graph"
    )
    # Only Mode A has a frozen oracle to be compared against.
    assert "legacy_metrics" in modes["A"] and "legacy_metrics" not in modes["B"]


def test_mode_b_without_mode_a_is_refused(tmp_path):
    """B is A's candidates under a different encoder, so B alone is a number with
    nothing to compare it to. Adding A silently would leave the caller believing
    otherwise."""
    from scripts.measure_scorer import main
    from tests.fixtures.synthetic_workspace import build_workspace as build

    data_dir, checkpoint = build(tmp_path / "ws")

    with pytest.raises(SystemExit, match="only meaningful beside A"):
        main([
            "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
            "--split", "test", "--output", str(tmp_path / "m.json"),
            "--device", "cpu", "--modes", "B",
        ])


def test_mode_c_alone_touches_no_retiring_legacy_path(tmp_path, monkeypatch):
    """The lifecycle claim, enforced rather than documented.

    `load_legacy_mode_a_inputs` and `build_legacy_mode_a_model` retire with the
    frozen evaluator. A C-only run that called them would break the day they go —
    and could fail today on a checkpoint the legacy loader cannot rebuild but
    production can.
    """
    import scripts.measure_scorer as cli
    from tests.fixtures.synthetic_workspace import build_workspace as build

    data_dir, checkpoint = build(tmp_path / "ws")

    def refuse(*args, **kwargs):
        raise AssertionError("a C-only run reached a retiring legacy entry point")

    monkeypatch.setattr(cli, "load_legacy_mode_a_inputs", refuse)
    monkeypatch.setattr(cli, "build_legacy_mode_a_model", refuse)

    output = tmp_path / "c_only" / "measurement.json"
    assert cli.main([
        "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
        "--split", "test", "--output", str(output),
        "--batch-size", str(BATCH_SIZE), "--num-workers", "0",
        "--device", "cpu", "--modes", "C",
    ]) == 0

    # A single-mode run writes the mode that was asked for to --output, rather
    # than leaving the requested path absent and a suffixed one beside it.
    written = json.loads(output.read_text())
    assert written["manifest"]["mode"] == "C"
    assert not (output.parent / "measurement_predictions.json").exists()


def test_a_and_c_must_agree_on_the_cohort_before_anything_is_written(tmp_path, monkeypatch):
    """The CLI prints that the modes share a cohort; that has to be checked, not
    announced. A and C reach their patients by different routes — the dataloader
    and the samples file — so a reordering in either is possible."""
    import scripts.measure_scorer as cli
    from tests.fixtures.synthetic_workspace import build_workspace as build

    data_dir, checkpoint = build(tmp_path / "ws")

    # `main` imports run_mode_c inside the function, so the patch has to land on
    # the source module rather than on the CLI's namespace.
    from src.evaluation import measurement

    original = measurement.run_mode_c

    def reordered(*args, **kwargs):
        result = original(*args, **kwargs)
        return type(result)(**{
            **{f: getattr(result, f) for f in result.__dataclass_fields__},
            "sample_ids": list(reversed(result.sample_ids)),
        })

    monkeypatch.setattr(measurement, "run_mode_c", reordered)

    with pytest.raises(SystemExit, match="same cohort in the same order"):
        cli.main([
            "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
            "--split", "test", "--output", str(tmp_path / "out" / "m.json"),
            "--batch-size", str(BATCH_SIZE), "--num-workers", "0",
            "--device", "cpu", "--modes", "A,B,C",
        ])


@pytest.mark.parametrize("spec, expected", [
    ("A,C", "not a supported combination"),
    ("B", "only meaningful beside A"),
    ("B,C", "only meaningful beside A"),
    ("D", "unknown mode"),
])
def test_unsupported_mode_combinations_are_refused_not_repaired(tmp_path, spec, expected):
    """`A,C` confounds encoder scope with candidate universe, so a run emitting
    both invites an attribution it cannot support. Silently completing it to
    `A,B,C` would confirm the caller's belief that they had asked for something
    attributable."""
    from scripts.measure_scorer import main
    from tests.fixtures.synthetic_workspace import build_workspace as build

    data_dir, checkpoint = build(tmp_path / "ws")

    with pytest.raises(SystemExit, match=expected):
        main([
            "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
            "--split", "test", "--output", str(tmp_path / "m.json"),
            "--device", "cpu", "--modes", spec,
        ])


# ==============================================================================
# --split is required, and a missing split says what the workspace has
# ==============================================================================
def test_split_has_no_default_on_either_entry_point():
    """Both defaulted to `test`, which the generator never writes.

    `src/kg/sample_generator.py` produces train and val only, so every entry
    point's default named a file no ordinary workspace contains. Requiring the
    flag is deliberate rather than switching the default to `val`: `val` is the
    checkpoint-selection split, and a default would let a caller measure on it
    without ever deciding to.
    """
    import scripts.calibrate_mode_a as calibrate
    import scripts.measure_scorer as measure

    for module, argv in (
        (measure, ["--checkpoint", "c.pt", "--data-dir", "d", "--output", "o.json"]),
        (calibrate, ["--checkpoint", "c.pt", "--data-dir", "d",
                     "--workdir", "w", "--seed", "0"]),
    ):
        with pytest.raises(SystemExit):
            module.parse_args(argv)


def test_missing_split_error_lists_what_the_workspace_actually_has(tmp_path):
    from src.kg.storage.file_storage import read_samples

    (tmp_path / "train_samples.json").write_text("[]")
    (tmp_path / "val_samples.json").write_text("[]")

    with pytest.raises(FileNotFoundError, match=r"this workspace has: train, val"):
        read_samples(tmp_path, "test")


def test_missing_split_error_when_nothing_is_there(tmp_path):
    from src.kg.storage.file_storage import read_samples

    with pytest.raises(FileNotFoundError, match="no \\*_samples.json files at all"):
        read_samples(tmp_path, "val")
