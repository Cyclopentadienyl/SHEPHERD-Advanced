"""
Mode C's command line, run end to end on a synthetic workspace.
===============================================================
`scripts/measure_scorer.py` is run as an operator runs it, as a subprocess, and
the assertions read **the files it writes** rather than an in-memory object. The
parts a direct call to `run_mode_c` skips are therefore exercised: argument
parsing, artifact identity, the device gate, the production model build, the
full-graph encode and the writers.

Reading the written files matters more than it sounds. A result object can carry
a field that `to_dict()` never emits, and an in-memory check would still pass
while the artifact a reviewer actually receives is missing it.

> **Nothing here is an acceptance result, and nothing here may be reported as
> one.** It shows the command runs and writes complete, self-describing
> artifacts on a synthetic graph in this container, on CPU.

Module: tests/integration/test_measure_scorer_run.py
"""
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.synthetic_workspace import build_workspace  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MEASURE_SCORER = REPO_ROOT / "scripts" / "measure_scorer.py"

BATCH_SIZE = 3
RUN_TIMEOUT_SECONDS = 900


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Run Mode C's command once and hand back what it wrote.

    sys.executable rather than "python": a bare name resolves against PATH,
    which is only this environment when a venv happens to be active.
    """
    root = tmp_path_factory.mktemp("mode_c_cli")
    data_dir, checkpoint = build_workspace(root)
    output = root / "run" / "measurement.json"

    completed = subprocess.run(
        [
            sys.executable, str(MEASURE_SCORER),
            "--checkpoint", str(checkpoint),
            "--data-dir", str(data_dir),
            "--split", "test", "--cohort-kind", "supplied",
            "--output", str(output),
            "--batch-size", str(BATCH_SIZE),
            "--device", "cpu",
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
        "ranks": read(output.with_name("measurement_ranks.json")),
        "written": sorted(p.name for p in output.parent.iterdir()),
        "checkpoint": checkpoint,
        "data_dir": data_dir,
    }


# ---------------------------------------------------------------------------
# What Mode C's command writes
# ---------------------------------------------------------------------------
def test_the_report_at_output_is_mode_c_scored_against_every_disease(measured):
    """The run writes Mode C's report to `--output`, beside its own ranks, and
    nothing else. Every batch is scored against every disease the workspace's
    graph has."""
    measurement = measured["measurement"]
    manifest = measurement["manifest"]
    n_diseases = json.loads((measured["data_dir"] / "num_nodes.json").read_text())["disease"]

    assert measured["written"] == ["measurement.json", "measurement_ranks.json"]
    assert manifest["mode"] == "C"
    assert manifest["candidate_construction"] == "every disease in the knowledge graph"
    assert manifest["model_construction"].startswith("production")

    columns = measurement["sampler_evidence"]["candidate_columns"]
    assert (columns["min"], columns["max"]) == (n_diseases, n_diseases)
    n_batches = measurement["sampler_evidence"]["n_batches"]
    assert n_batches == math.ceil(manifest["n_samples"] / BATCH_SIZE)


def test_the_report_carries_only_the_untruncated_metric_family(measured):
    """Mode C reports the untruncated metrics and nothing truncated. The report
    carries exactly its summary fields, so a second metric family, or the
    per-sample rows, cannot ride along in it unnoticed."""
    measurement = measured["measurement"]

    assert set(measurement) == {
        "manifest", "authoritative_metrics", "n_ranked", "n_ground_truth_absent",
        "sampler_evidence",
    }
    assert "untruncated_mrr" in measurement["authoritative_metrics"]
    assert "mrr" not in measurement["authoritative_metrics"]
    assert measurement["authoritative_metrics"]["mean_rank"] is not None


def test_the_manifest_records_the_chunk_size_the_run_used(measured):
    """`batch_size` is the one loader-era field Mode C's manifest keeps: it is the
    chunk size, it can move scores at floating-point level, and the ledger's
    semantics digest carries it. The run asked for 3, not the default 32."""
    assert BATCH_SIZE != 32
    assert measured["measurement"]["manifest"]["batch_size"] == BATCH_SIZE


def test_every_consumed_file_is_identified_by_content(measured):
    """Paths are not identities. `checkpoints/best.pt` names a different file after
    every improvement, and the structural fingerprint is shared by every checkpoint
    trained on the same graph."""
    digests = measured["measurement"]["manifest"]["artifact_digests"]

    for role in ("checkpoint", "samples", "node_features", "edge_indices", "num_nodes"):
        assert len(digests[role]) == 64, f"{role} has no sha256"
    assert digests["checkpoint"] != digests["samples"]

    # And each is the digest of the file the command was given, not merely a
    # digest-shaped string: a run that recorded another cohort's or another
    # checkpoint's digest would pass the checks above.
    data_dir = measured["data_dir"]
    given = {
        "checkpoint": measured["checkpoint"],
        "samples": data_dir / "test_samples.json",
        "node_features": data_dir / "node_features.pt",
        "edge_indices": data_dir / "edge_indices.pt",
        "num_nodes": data_dir / "num_nodes.json",
    }
    for role, path in given.items():
        assert digests[role] == hashlib.sha256(path.read_bytes()).hexdigest(), role


def test_the_kg_role_is_the_manifest_bound_source_graph(measured):
    """Mode C never parses `kg.json`, so its `kg` role is the digest the
    workspace's manifest binds for the source graph the tensors scored were
    verified as the export of (contract M2.1 decision 2), not a file it read. A
    supplied cohort records no allocation, and nothing else becomes a role."""
    digests = measured["measurement"]["manifest"]["artifact_digests"]
    manifest = json.loads((measured["data_dir"] / "split_manifest.json").read_text())

    assert digests["kg"] == manifest["artifacts"]["kg"]
    assert set(digests) == {
        "checkpoint", "samples", "node_features", "edge_indices", "num_nodes", "kg",
    }


def test_the_cohort_is_whole_and_ranked_in_the_order_given(measured):
    """No absences, no shrinkage, no reordering. Every disease is a candidate in
    Mode C, so an absence would mean the ids are wrong — and a metric over a
    silently reduced or reordered cohort answers a question nobody asked. The
    ranks file is compared with the samples file the command was given."""
    measurement = measured["measurement"]
    samples = json.loads((measured["data_dir"] / "test_samples.json").read_text())

    assert measurement["n_ground_truth_absent"] == 0
    assert measurement["n_ranked"] == measurement["manifest"]["n_samples"] == len(samples)
    assert [(row["sample_id"], row["ground_truth"]) for row in measured["ranks"]] == [
        (sample["patient_id"], sample["disease_id"]) for sample in samples
    ]


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


# ---------------------------------------------------------------------------
# In process, on a checkpoint in the training writer's shape
# ---------------------------------------------------------------------------
def test_an_in_process_run_writes_mode_c_at_output(tmp_path):
    """`main` called directly, as the unit tests call it, on the fixture's
    checkpoint, which carries only what the training writer writes: the model is
    built the way production builds it, and Mode C's report is at `--output`
    beside its ranks, with nothing else written."""
    import scripts.measure_scorer as cli
    from tests.fixtures.synthetic_workspace import build_workspace as build

    data_dir, checkpoint = build(tmp_path / "ws")

    output = tmp_path / "run" / "measurement.json"
    assert cli.main([
        "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
        "--split", "test", "--cohort-kind", "supplied", "--output", str(output),
        "--batch-size", str(BATCH_SIZE), "--device", "cpu",
    ]) == 0

    written = json.loads(output.read_text())
    assert written["manifest"]["mode"] == "C"
    assert sorted(p.name for p in output.parent.iterdir()) == [
        "measurement.json", "measurement_ranks.json",
    ]
