"""
Mode C's command line, run end to end.
======================================
`scripts/measure_scorer.py` is run as an operator runs it, as a subprocess, on a
registered test cohort version (`tests/fixtures/registered_cohort.py`: the demo
graph, a checkpoint over it and a version imported through the production
importer) with `--record`, and the assertions read **the files it writes**
rather than an in-memory object. The parts a direct call to `run_mode_c` skips
are therefore exercised: argument parsing, artifact identity, the device gate,
the production model build, the full-graph encode, the report store, the
registration and the console. A generated split, `--split val`, is run in
process on the synthetic workspace.

Reading the written files matters more than it sounds. A result object can carry
a field that `to_dict()` never emits, and an in-memory check would still pass
while the artifact a reviewer actually receives is missing it.

> **Nothing here is an acceptance result, and nothing here may be reported as
> one.** It shows the command runs and writes complete, self-describing
> artifacts on a synthetic graph in this container, on CPU, with an untrained
> checkpoint.

Module: tests/integration/test_measure_scorer_run.py
"""
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.registered_cohort import (  # noqa: E402
    GRAPH_SOURCE_COUNTS,
    build_registered_cohort,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MEASURE_SCORER = REPO_ROOT / "scripts" / "measure_scorer.py"

BATCH_SIZE = 3
RUN_TIMEOUT_SECONDS = 900
RUN_ID = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}-[0-9a-f]{8}")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Run Mode C's command once on a registered version, with `--record`, and
    hand back what it wrote.

    sys.executable rather than "python": a bare name resolves against PATH,
    which is only this environment when a venv happens to be active.
    """
    cohort = build_registered_cohort(tmp_path_factory.mktemp("mode_c_cli") / "root")
    label = cohort.version.manifest["label"]

    completed = subprocess.run(
        [
            sys.executable, str(MEASURE_SCORER),
            "--checkpoint", str(cohort.checkpoint),
            "--data-dir", str(cohort.data_dir),
            "--cohort", label, "--registry", str(cohort.registry),
            "--batch-size", str(BATCH_SIZE),
            "--device", "cpu",
            "--record",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]

    store = sorted((cohort.version.directory / "reports").iterdir())
    runs = [path for path in store if RUN_ID.fullmatch(path.name)]
    assert len(runs) == 1, runs
    # Beside the run directory, only the version's usage history (since M5-1).
    assert [path.name for path in store if path not in runs] == ["usage.jsonl"], store
    run_dir = runs[0]

    def read(path: Path):
        assert path.exists(), f"{path.name} was not written"
        return json.loads(path.read_text())

    return {
        "measurement": read(run_dir / "report.json"),
        "ranks": read(run_dir / "ranks.json"),
        "run_dir": run_dir,
        "written": sorted(p.name for p in run_dir.iterdir()),
        "cohort": cohort,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


# ---------------------------------------------------------------------------
# What Mode C's command writes, and where
# ---------------------------------------------------------------------------
def test_the_report_is_kept_in_a_new_run_directory_of_the_report_store(measured):
    """`<registry>/versions/<label>/reports/<run id>/` holds the ranks and the
    report and nothing else; the run id says when, which weights, and a random
    part, so two runs never share a directory."""
    cohort = measured["cohort"]
    run_dir = measured["run_dir"]

    assert run_dir.parent == cohort.version.directory / "reports"
    assert RUN_ID.fullmatch(run_dir.name)
    assert run_dir.name.split("-")[1] == cohort.checkpoint_sha256[:12]
    assert measured["written"] == ["ranks.json", "report.json"]


def test_the_run_and_its_registration_are_in_the_versions_usage_history(measured):
    """M5-1 (D4): the command-line run appended itself, then its registration,
    to the version's usage history, naming the report by its digest."""
    from src.evaluation.report_store import read_usage_history

    entries, problems = read_usage_history(measured["cohort"].version)
    digest = _sha(measured["run_dir"] / "report.json")

    assert problems == []
    assert [(e["kind"], e["run_id"], e["report_sha256"]) for e in entries] == [
        ("test_run", measured["run_dir"].name, digest),
        ("registered", measured["run_dir"].name, digest)]


def test_the_report_names_its_ranks_by_the_digest_written(measured):
    """The report is the commit point and carries the ranks' digest, so the two
    files are a pair a reader can check."""
    block = measured["measurement"]["ranks"]

    assert block == {"file": "ranks.json", "sha256": _sha(measured["run_dir"] / "ranks.json")}


def test_registry_files_are_written_as_every_registry_file_is(measured):
    """ASCII, sorted keys, two-space indent, no NaN: the registry's one encoding
    (`encode_json`), for the report and the ranks alike."""
    from src.evaluation.cohort_registry import encode_json

    for name in ("report.json", "ranks.json"):
        data = (measured["run_dir"] / name).read_bytes()
        assert data == encode_json(json.loads(data)), name


def test_the_report_is_mode_c_scored_against_every_disease(measured):
    """Every batch is scored against every disease the workspace's graph has."""
    measurement = measured["measurement"]
    manifest = measurement["manifest"]
    data_dir = measured["cohort"].data_dir
    n_diseases = json.loads((data_dir / "num_nodes.json").read_text())["disease"]

    assert manifest["mode"] == "C"
    assert manifest["candidate_construction"] == "every disease in the knowledge graph"
    assert manifest["model_construction"].startswith("production")

    columns = measurement["sampler_evidence"]["candidate_columns"]
    assert (columns["min"], columns["max"]) == (n_diseases, n_diseases)
    n_batches = measurement["sampler_evidence"]["n_batches"]
    assert n_batches == math.ceil(manifest["n_samples"] / BATCH_SIZE)


def test_the_report_carries_only_the_untruncated_metric_family(measured):
    """Mode C reports the untruncated metrics and nothing truncated. The report
    carries exactly its summary fields and the ranks' name and digest, so a
    second metric family, or the per-sample rows, cannot ride along in it."""
    measurement = measured["measurement"]

    assert set(measurement) == {
        "manifest", "authoritative_metrics", "n_ranked", "n_ground_truth_absent",
        "sampler_evidence", "ranks",
    }
    assert "untruncated_mrr" in measurement["authoritative_metrics"]
    assert "mrr" not in measurement["authoritative_metrics"]
    assert measurement["authoritative_metrics"]["mean_rank"] is not None


def test_the_manifest_records_the_chunk_size_the_run_used(measured):
    """`batch_size` is the chunk size, it can move scores at floating-point level,
    and the ledger's semantics digest carries it. The run asked for 3, not the
    default 32."""
    assert BATCH_SIZE != 32
    assert measured["measurement"]["manifest"]["batch_size"] == BATCH_SIZE


def test_every_consumed_file_is_identified_by_content(measured):
    """Paths are not identities. Each digest is the digest of the file the command
    was given, not merely a digest-shaped string: a run that recorded another
    cohort's or another checkpoint's digest would pass a shape check."""
    digests = measured["measurement"]["manifest"]["artifact_digests"]
    cohort = measured["cohort"]
    data_dir = cohort.data_dir
    given = {
        "checkpoint": cohort.checkpoint,
        "samples": cohort.version.samples_path,
        "node_features": data_dir / "node_features.pt",
        "edge_indices": data_dir / "edge_indices.pt",
        "num_nodes": data_dir / "num_nodes.json",
    }
    for role, path in given.items():
        assert digests[role] == _sha(path), role


def test_the_kg_role_is_the_manifest_bound_source_graph_and_no_allocation_is_recorded(measured):
    """Mode C never parses `kg.json`, so its `kg` role is the digest the
    workspace's manifest binds (contract M2.1 decision 2). A registered version
    was cut by no allocation of this workspace, so `split_manifest` is no role;
    the version's own manifest is identified in `cohort_version`."""
    digests = measured["measurement"]["manifest"]["artifact_digests"]
    manifest = json.loads((measured["cohort"].data_dir / "split_manifest.json").read_text())

    assert digests["kg"] == manifest["artifacts"]["kg"]
    assert set(digests) == {
        "checkpoint", "samples", "node_features", "edge_indices", "num_nodes", "kg",
    }


def test_the_manifest_describes_the_registered_version_scored(measured):
    """The role, the kind, the version block (aggregates only, the manifest digest
    of the `cohort.json` read), the relation states a registered run earns, R10
    and the environment."""
    from src.evaluation.measurement import expected_relations
    from src.evaluation.sidecar import COHORT_VERSION_FIELDS

    manifest = measured["measurement"]["manifest"]
    version = measured["cohort"].version

    assert manifest["split"] == version.manifest["role"]
    assert manifest["cohort_kind"] == "supplied"
    block = manifest["cohort_version"]
    assert set(block) == set(COHORT_VERSION_FIELDS)
    assert block["manifest_sha256"] == _sha(version.directory / "cohort.json")
    assert block["samples_sha256"] == _sha(version.samples_path)
    assert block["counts"] == GRAPH_SOURCE_COUNTS["keep"]
    assert manifest["relations"] == expected_relations(registered=True)
    assert set(manifest["r10"]) == {"definition_version", "candidates_examined",
                                    "training", "validation"}
    assert set(manifest["environment"]) == {"facts", "code"}


def test_the_cohort_is_whole_and_ranked_in_the_order_given(measured):
    """No absences, no shrinkage, no reordering. The ranks file is compared with
    the version's samples file the command read."""
    measurement = measured["measurement"]
    samples = json.loads(measured["cohort"].version.samples_path.read_text())

    assert measurement["n_ground_truth_absent"] == 0
    assert measurement["n_ranked"] == measurement["manifest"]["n_samples"] == len(samples)
    assert [(row["sample_id"], row["ground_truth"]) for row in measured["ranks"]] == [
        (sample["patient_id"], sample["disease_id"]) for sample in samples
    ]


def test_the_report_is_registered_beside_the_checkpoint(measured):
    """`--record` filed one record, keyed by the checkpoint read and the version's
    manifest digest, naming the report by the digest of its bytes."""
    cohort = measured["cohort"]
    ledger = json.loads((cohort.checkpoint.parent / "evaluations.json").read_text())

    assert ledger["schema_version"] == 3
    (record,) = ledger["records"]
    assert record["checkpoint_digest"] == cohort.checkpoint_sha256
    assert record["cohort_digest"] == _sha(cohort.version.directory / "cohort.json")
    assert record["source_artifact_digest"] == _sha(measured["run_dir"] / "report.json")
    assert record["registration"]["sequence"] == 1


def test_the_console_names_the_report_the_run_coverage_r10_and_the_registration(measured):
    """What an operator reads: where the report is, the run id, coverage with its
    denominators and every exclusion count, each R10 label's state and count,
    and what the registration did. No case id."""
    out = measured["stdout"]
    counts = GRAPH_SOURCE_COUNTS["keep"]
    report = measured["run_dir"] / "report.json"

    assert f"Report: {report}" in out
    assert f"Run id: {measured['run_dir'].name}" in out
    assert (f"coverage: {counts['ranked_eligible']} ranked of {counts['source_cases']} "
            f"source cases; {counts['n_excluded']} excluded") in out
    assert "ground_truth_absent 2, phenotype_unmappable 1" in out
    for name in ("Training disease overlap", "Training phenotype-set overlap",
                 "Validation disease overlap", "Validation phenotype-set overlap"):
        assert name in out
    assert "Registered as record 1" in out
    assert "case-" not in out + measured["stderr"], "a case id reached the console"


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
# In process, a generated split, on a checkpoint in the training writer's shape
# ---------------------------------------------------------------------------
def test_an_in_process_generated_run_writes_mode_c_at_output(tmp_path):
    """`main` called directly, as the unit tests call it, on the synthetic
    workspace's generated `val` and its checkpoint, which carries only what the
    training writer writes: the model is built the way production builds it,
    and Mode C's report is at `--output` beside its ranks, with nothing else
    written. A generated report keeps its shape: no cohort version, no R10, and
    no ranks block, since its rank file is named by `--output`."""
    import scripts.measure_scorer as cli
    from tests.fixtures.synthetic_workspace import build_workspace

    data_dir, checkpoint = build_workspace(tmp_path / "ws")

    output = tmp_path / "run" / "measurement.json"
    assert cli.main([
        "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
        "--split", "val", "--output", str(output),
        "--batch-size", str(BATCH_SIZE), "--device", "cpu",
    ]) == 0

    written = json.loads(output.read_text())
    assert written["manifest"]["mode"] == "C"
    assert written["manifest"]["split"] == "val"
    assert written["manifest"]["cohort_kind"] == "generated"
    assert written["manifest"]["cohort_version"] is None and written["manifest"]["r10"] is None
    assert "ranks" not in written
    ranks = json.loads((output.parent / "measurement_ranks.json").read_text())
    assert len(ranks) == written["n_ranked"] == 4
    assert sorted(p.name for p in output.parent.iterdir()) == [
        "measurement.json", "measurement_ranks.json",
    ]


def test_an_existing_output_is_replaced_staged(tmp_path, monkeypatch):
    """The generated path kept its overwrite of an existing `--output`, now
    through `publish_bytes`: staged beside the target and renamed into place, so
    no staging file is left and the old bytes are gone."""
    import scripts.measure_scorer as cli
    import src.utils.file_modes as file_modes
    from tests.fixtures.synthetic_workspace import build_workspace

    data_dir, checkpoint = build_workspace(tmp_path / "ws")
    output = tmp_path / "run" / "measurement.json"
    output.parent.mkdir()
    output.write_text("an earlier report")
    published = []
    real = file_modes.publish_bytes
    monkeypatch.setattr(file_modes, "publish_bytes",
                        lambda target, data: published.append(Path(target).name)
                        or real(target, data))

    assert cli.main([
        "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
        "--split", "val", "--output", str(output), "--device", "cpu",
    ]) == 0

    assert published == ["measurement_ranks.json", "measurement.json"], "the report is last"
    assert json.loads(output.read_text())["manifest"]["mode"] == "C"
    assert sorted(p.name for p in output.parent.iterdir()) == [
        "measurement.json", "measurement_ranks.json",
    ]
