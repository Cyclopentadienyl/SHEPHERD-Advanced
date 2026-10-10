"""
Mode A's command line, run end to end on a synthetic workspace — NOT calibration.
=================================================================================
`scripts/measure_scorer.py` is run as an operator runs it, as a subprocess, and
the assertions read **the files it writes** rather than an in-memory object. The
parts a direct call to `run_mode_a` skips are therefore exercised: argument
parsing, artifact identity, the device gate, the worker processes and the
predictions writer.

Reading the written files matters more than it sounds. A result object can carry
a field that `to_dict()` never emits, and an in-memory check would still pass
while the artifact a reviewer actually receives is missing it. That is a defect
this harness has already had.

> **Nothing here is calibration, and nothing here may be reported as such.** It
> shows the command runs and writes complete, self-describing artifacts on a
> synthetic graph in this container, on CPU. The adopted acceptance is the
> same-batch differential calibration (`src/evaluation/differential.py`,
> `tests/unit/test_differential_calibration.py`); its institutional CUDA run is
> BACKLOG item 7a.

The rest of the file covers the mode ladder, the refusals of unsupported mode
combinations, and `--split`.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.evaluation.measurement import LEGACY_TRUNCATION_K  # noqa: E402
from tests.fixtures.synthetic_workspace import build_workspace  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MEASURE_SCORER = REPO_ROOT / "scripts" / "measure_scorer.py"

BATCH_SIZE = 3
SEED = 20260818
RUN_TIMEOUT_SECONDS = 900


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Run Mode A's command once and hand back what it wrote.

    Four workers and an explicit `--predictions-output`, so the worker processes
    and the predictions writer run end to end. sys.executable rather than
    "python": a bare name resolves against PATH, which is only this environment
    when a venv happens to be active.
    """
    root = tmp_path_factory.mktemp("mode_a_cli")
    data_dir, checkpoint = build_workspace(root)
    output = root / "run" / "measurement.json"
    predictions = root / "run" / "rows" / "predictions.json"

    completed = subprocess.run(
        [
            sys.executable, str(MEASURE_SCORER),
            "--checkpoint", str(checkpoint),
            "--data-dir", str(data_dir),
            "--split", "test", "--cohort-kind", "supplied",
            "--output", str(output),
            "--predictions-output", str(predictions),
            "--batch-size", str(BATCH_SIZE),
            "--num-workers", "4",
            "--device", "cpu",
            "--seed", str(SEED),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]

    def read(path: Path):
        assert path.exists(), f"{path.name} was not written"
        return json.loads(path.read_text())

    return {
        "measurement": read(output),
        "predictions": read(predictions),
        "default_predictions_path": output.parent / "measurement_predictions.json",
    }


# ---------------------------------------------------------------------------
# What Mode A's command writes
# ---------------------------------------------------------------------------
def test_the_predictions_go_where_they_were_asked_to(measured):
    """`--predictions-output` is honoured, not just defaulted: the rows are at the
    requested path and nothing is left at the default one beside the report."""
    assert measured["predictions"]
    assert not measured["default_predictions_path"].exists()


def test_both_metric_families_are_reported(measured):
    """The legacy truncated MRR and the untruncated authoritative metrics are
    different quantities and must not be conflated. Here the candidate set is
    smaller than 20 so they coincide numerically — which is exactly why the
    assertion is about the report's shape, not its value."""
    measurement = measured["measurement"]

    assert f"legacy_mrr_truncated_at_{LEGACY_TRUNCATION_K}" in measurement["legacy_metrics"]
    assert "untruncated_mrr" in measurement["authoritative_metrics"]
    assert measurement["authoritative_metrics"]["mean_rank"] is not None


def test_the_predictions_rows_are_truncated_and_identified(measured):
    """Each row names its sample and its truth, and carries at most K predictions.
    A prediction list under a missing id would say nothing about which patient it
    describes."""
    for row in measured["predictions"]:
        assert set(row) >= {"sample_id", "ground_truth", "predictions"}
        assert len(row["predictions"]) <= LEGACY_TRUNCATION_K


def test_the_measurement_artifact_carries_the_sampler_evidence(measured):
    """The manifest states what was configured; this states what the sampler did.
    Only the pair is evidence — a manifest claiming a candidate construction
    beside an observation that contradicts it is what this is here to expose."""
    evidence = measured["measurement"]["sampler_evidence"]
    manifest = measured["measurement"]["manifest"]

    assert evidence["n_batches"] >= 1
    assert evidence["candidate_columns"]["max"] >= 1
    assert set(evidence["max_subgraph_nodes"]) <= {"phenotype", "gene", "disease"}

    negatives = evidence["negative_sampling"]
    assert negatives["observed"] is True
    assert negatives["total_drawn"] == manifest["n_samples"] * manifest["num_negative_samples"]
    assert negatives["unique_global_ids"] <= negatives["total_drawn"]


def test_every_consumed_file_is_identified_by_content(measured):
    """Paths are not identities. `checkpoints/best.pt` names a different file after
    every improvement, and the structural fingerprint is shared by every checkpoint
    trained on the same graph."""
    digests = measured["measurement"]["manifest"]["artifact_digests"]

    for role in ("checkpoint", "samples", "node_features", "edge_indices", "num_nodes"):
        assert len(digests[role]) == 64, f"{role} has no sha256"
    assert digests["checkpoint"] != digests["samples"]


def test_the_cohort_is_whole(measured):
    """No absences, no shrinkage. In Mode A the truth is a subgraph seed, so an
    absence would mean the harness is wrong — and a metric over a silently
    reduced cohort answers a question nobody asked."""
    measurement = measured["measurement"]

    assert measurement["n_ground_truth_absent"] == 0
    assert measurement["n_ranked"] == measurement["manifest"]["n_samples"]
    assert len(measured["predictions"]) == measurement["manifest"]["n_samples"]


# ---------------------------------------------------------------------------
# The claim guard
# ---------------------------------------------------------------------------
def test_this_run_records_that_it_is_not_a_cuda_run(measured):
    """A guard against the claim rather than the code. Nothing here touches real
    data, a real checkpoint or CUDA, and the manifest has to say so in writing."""
    manifest = measured["measurement"]["manifest"]

    assert manifest["cuda_executed"] is False, (
        "a CPU run reported itself as having executed on CUDA; every assertion "
        "above would still pass while the recorded claim was false"
    )
    assert manifest["device"] == "cpu"
    assert manifest["n_samples"] < 100


def test_the_manifest_records_the_configured_ceiling_not_only_the_observed_one(measured):
    """A run that never approached the subgraph cap and a run truncated by it look
    identical in the observation alone."""
    manifest = measured["measurement"]["manifest"]
    observed = measured["measurement"]["sampler_evidence"]["max_subgraph_nodes"]

    assert manifest["max_subgraph_nodes"] == 5000  # DataLoaderConfig default
    assert max(observed.values()) < manifest["max_subgraph_nodes"], (
        "the fixture is supposed to sit far below the cap; if it does not, the two "
        "fields no longer demonstrate anything different from each other"
    )


# ---------------------------------------------------------------------------
# The CLI runs the ladder, and Mode A's artifacts keep their names
# ---------------------------------------------------------------------------
def test_the_cli_runs_all_three_modes_and_keeps_mode_a_at_its_paths(tmp_path):
    """Adding modes B and C must not move Mode A's artifacts.

    Mode A keeps `--output`'s filename and the predictions artifact; B and C sit
    beside them, one file per mode. A reader that opens `--output` after a ladder
    run therefore still finds Mode A there.
    """
    from scripts.measure_scorer import main
    from tests.fixtures.synthetic_workspace import build_workspace as build

    data_dir, checkpoint = build(tmp_path / "ws")
    output = tmp_path / "run" / "measurement.json"

    exit_code = main([
        "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
        "--split", "test", "--cohort-kind", "supplied", "--output", str(output),
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
    # Only Mode A carries the legacy truncated metrics.
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
            "--split", "test", "--cohort-kind", "supplied", "--output", str(tmp_path / "m.json"),
            "--device", "cpu", "--modes", "B",
        ])


def test_mode_c_alone_touches_no_retiring_legacy_path(tmp_path, monkeypatch):
    """The lifecycle claim, enforced rather than documented.

    `load_legacy_mode_a_inputs` is deleted in M2.1's S7, and
    `build_legacy_mode_a_model` goes with item 9's oracle-only surface. A C-only
    run that called either would break the day it goes — and could fail today on
    a checkpoint the legacy builder cannot rebuild but production can.
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
        "--split", "test", "--cohort-kind", "supplied", "--output", str(output),
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
            "--split", "test", "--cohort-kind", "supplied", "--output", str(tmp_path / "out" / "m.json"),
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
            "--split", "test", "--cohort-kind", "supplied", "--output", str(tmp_path / "m.json"),
            "--device", "cpu", "--modes", spec,
        ])


# ==============================================================================
# --split is required, and a missing split says what the workspace has
# ==============================================================================
def test_split_has_no_default_on_the_measurement_entry_point():
    """It defaulted to `test`, which the generator never writes.

    `src/kg/sample_generator.py` produces train and val only, so the default named
    a file no ordinary workspace contains. Requiring the flag is deliberate rather
    than switching the default to `val`: `val` is the checkpoint-selection split,
    and a default would let a caller measure on it without ever deciding to.
    """
    import scripts.measure_scorer as measure

    with pytest.raises(SystemExit):
        measure.parse_args(["--checkpoint", "c.pt", "--data-dir", "d", "--output", "o.json"])


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
