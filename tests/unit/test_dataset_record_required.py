"""
The dataset record is required: every consumer refuses without it (contract M3b).
=================================================================================
Training, serving and measurement refuse a workspace whose `kg.provenance.json`
is missing, unreadable or does not match, each from its own entry point and
before it writes anything, builds a model or reads a checkpoint. A synthetic
workspace is fully bound, passes, and is shown as synthetic: in the training log,
on serving's published configuration and status, in the measurement manifest, and
in the inventory, where the binding state and the origin are separate fields.

The writers' half: the manifest is not written without the record's binding, the
digests every writer binds are of the bytes it wrote, and the builder stops
rather than record a required input as missing.

The record's own states, its one read and its rules are in
`test_kg_provenance.py`; this file is the consumers and the writers.

Module: tests/unit/test_dataset_record_required.py
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.generated_workspace import write_generated_workspace  # noqa: E402

IDS = {"train_ids": [0, 1, 2], "val_ids": [3]}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rebind(workspace, record_bytes):
    """Write `record_bytes` as the record and bind them in the manifest."""
    from src.kg.artifacts import MANIFEST_FILENAME
    from src.kg.provenance import PROVENANCE_FILENAME

    (workspace / PROVENANCE_FILENAME).write_bytes(record_bytes)
    manifest = json.loads((workspace / MANIFEST_FILENAME).read_text())
    manifest["artifacts"]["provenance"] = hashlib.sha256(record_bytes).hexdigest()
    (workspace / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))


def _missing(workspace):
    from src.kg.provenance import PROVENANCE_FILENAME

    (workspace / PROVENANCE_FILENAME).unlink()


def _replaced(workspace):
    """Another workspace's record in this one's place: well formed, not bound."""
    from src.kg.provenance import PROVENANCE_FILENAME

    other = write_generated_workspace(workspace.parent / "elsewhere",
                                      train_ids=[5, 6], val_ids=[7])[0]
    (workspace / PROVENANCE_FILENAME).write_bytes((other / PROVENANCE_FILENAME).read_bytes())


def _unbound(workspace):
    from src.kg.artifacts import MANIFEST_FILENAME

    manifest = json.loads((workspace / MANIFEST_FILENAME).read_text())
    del manifest["artifacts"]["provenance"]
    (workspace / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))


def _unidentified(workspace):
    """Bound, naming this graph, from files, with the MONDO input unidentified."""
    from src.kg.provenance import PROVENANCE_FILENAME, encode_provenance

    record = json.loads((workspace / PROVENANCE_FILENAME).read_text())
    record.update(origin="files", missing_roles=["mondo"], sources=[
        {"role": role, "filename": f"{role}.f", "digest": "a" * 64, "declared_version": None}
        for role in ("hpo", "phenotype_hpoa", "genes_to_phenotype")])
    _rebind(workspace, encode_provenance(record).encode())


#: Each way a workspace's record fails, with the state it establishes and a
#: fragment of the refusal.
FAILURES = {
    "missing": (_missing, "unverifiable", "is not here"),
    "replaced": (_replaced, "mismatch", "is not the dataset record"),
    "unbound": (_unbound, "unrecorded", "binds no kg.provenance.json"),
    "unidentified": (_unidentified, "verified", "unidentified"),
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", sorted(FAILURES))
def test_training_refuses_before_anything_is_written(tmp_path, monkeypatch, case):
    """At the top of the run: no run directory, no model, nothing restored."""
    import scripts.train_model as train_model
    from src.kg.artifacts import DatasetRecordRefused

    tamper, binding, fragment = FAILURES[case]
    workspace = write_generated_workspace(tmp_path / "ws", training_fields=True, **IDS)[0]
    tamper(workspace)
    for name in ("create_dataloaders", "create_model_from_config", "Trainer",
                 "read_resume_parent"):
        monkeypatch.setattr(train_model, name, lambda *a, _n=name, **k: (_ for _ in ()).throw(
            AssertionError(f"{_n} ran after a refusal")))
    run = tmp_path / "run"

    with pytest.raises(DatasetRecordRefused, match=fragment) as refused:
        train_model.train(train_model.TrainConfig(
            data_dir=str(workspace), run_dir=str(run), device="cpu", num_workers=0))

    assert refused.value.binding == binding
    assert not run.exists(), "a run artifact was written before the refusal"


def test_training_reads_a_synthetic_record_and_says_so(tmp_path, caplog):
    from scripts.train_model import read_training_inputs

    workspace = write_generated_workspace(tmp_path / "ws", training_fields=True, **IDS)[0]
    with caplog.at_level(logging.INFO, logger="scripts.train_model"):
        inputs = read_training_inputs(workspace)

    record = inputs.dataset_record
    assert (record.binding, record.origin) == ("verified", "synthetic")
    assert record.identity.sha256 == _sha(workspace / "kg.provenance.json")
    assert "binding verified, origin synthetic" in caplog.text


def test_the_training_record_names_the_manifest_that_binds_the_dataset_record(tmp_path):
    """No new role: the manifest a run records binds the record's digest, so the
    record is identified through it (the plan's §5.1 leaves the checkpoint's
    record unchanged by M3b)."""
    from scripts.train_model import read_training_inputs, training_input_digests
    from src.kg.artifacts import read_split_manifest

    workspace = write_generated_workspace(tmp_path / "ws", training_fields=True, **IDS)[0]
    digests = training_input_digests(read_training_inputs(workspace))

    assert "provenance" not in digests
    assert digests["split_manifest"] == _sha(workspace / "split_manifest.json")
    assert (read_split_manifest(workspace).manifest["artifacts"]["provenance"]
            == _sha(workspace / "kg.provenance.json"))


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def _measure(data_dir, checkpoint, output, *, cohort=None):
    """`--split val` with `--output`, or, with `cohort` (a `RegisteredCohort`),
    `--cohort` in its registry, whose report store decides where it goes."""
    import scripts.measure_scorer as cli

    argv = ["--checkpoint", str(checkpoint), "--data-dir", str(data_dir), "--device", "cpu"]
    if cohort is None:
        argv += ["--split", "val", "--output", str(output)]
    else:
        argv += ["--cohort", cohort.version.manifest["label"], "--registry", str(cohort.registry)]
    return cli.main(argv)


@pytest.fixture(scope="module")
def registered_base(tmp_path_factory):
    from tests.fixtures.registered_cohort import build_registered_cohort

    return build_registered_cohort(tmp_path_factory.mktemp("registered_base"))


@pytest.mark.parametrize("kind", ["generated", "registered"])
@pytest.mark.parametrize("case", sorted(FAILURES))
def test_measurement_refuses_before_a_checkpoint_or_model(
    tmp_path, monkeypatch, registered_base, case, kind
):
    """For a registered cohort version too: it is scored against a workspace's
    graph, and that graph's record is required as for a generated split."""
    import src.models.gnn.shepherd_gnn as shepherd_gnn
    import src.utils.checkpoint_io as checkpoint_io
    from src.kg.artifacts import DatasetRecordRefused
    from tests.fixtures.registered_cohort import copy_registered_cohort
    from tests.fixtures.synthetic_workspace import build_workspace

    tamper, binding, fragment = FAILURES[case]
    cohort = None
    if kind == "generated":
        data_dir, checkpoint = build_workspace(tmp_path / "ws")
    else:
        cohort = copy_registered_cohort(registered_base, tmp_path / "root")
        data_dir, checkpoint = cohort.data_dir, cohort.checkpoint
        # The deployment's registry: a --cohort run measures only there (BACKLOG 37).
        monkeypatch.setenv("SHEPHERD_REGISTRY", str(cohort.registry))
    tamper(data_dir)

    def forbidden(*args, **kwargs):
        raise AssertionError("reached after the workspace was refused")

    monkeypatch.setattr(shepherd_gnn, "build_shepherd_model", forbidden)
    monkeypatch.setattr(checkpoint_io, "read_checkpoint", forbidden)
    output = tmp_path / "out" / "measurement.json"

    with pytest.raises(DatasetRecordRefused, match=fragment) as refused:
        _measure(data_dir, checkpoint, output, cohort=cohort)

    assert refused.value.binding == binding
    assert not output.parent.exists()
    if cohort is not None:
        assert not (cohort.version.directory / "reports").exists()


def test_a_measurement_records_its_dataset_record_with_the_origin_apart(tmp_path):
    """A number over a synthetic workspace says so in its manifest; the binding
    state and the origin are separate fields."""
    from tests.fixtures.synthetic_workspace import build_workspace

    data_dir, checkpoint = build_workspace(tmp_path / "ws")
    output = tmp_path / "out" / "measurement.json"

    assert _measure(data_dir, checkpoint, output) == 0
    recorded = json.loads(output.read_text())["manifest"]["dataset_record"]
    assert (recorded["binding"], recorded["origin"]) == ("verified", "synthetic")
    assert recorded["sha256"] == _sha(data_dir / "kg.provenance.json")
    assert recorded["sources"] == []
    assert recorded["incomplete_by_design"], "what the record does not establish is shown"


def test_the_measurement_console_says_what_its_graph_was_built_from(tmp_path, capsys):
    """Where an operator reads the number, as in the manifest."""
    from tests.fixtures.synthetic_workspace import build_workspace

    data_dir, checkpoint = build_workspace(tmp_path / "ws")
    assert _measure(data_dir, checkpoint, tmp_path / "out" / "measurement.json") == 0

    out = capsys.readouterr().out
    assert "Dataset record: binding verified, origin synthetic" in out
    assert "SYNTHETIC WORKSPACE" in out


def test_the_dataset_record_does_not_enter_the_measurement_semantics():
    """It describes the graph's sources; no mode computes from it, so two runs
    over the same tensors are the same measurement whatever it says."""
    from src.evaluation.sidecar import (
        NON_SEMANTIC_FIELDS,
        SEMANTIC_MANIFEST_FIELDS,
        measurement_semantics_digest,
    )

    assert "dataset_record" in NON_SEMANTIC_FIELDS
    assert "dataset_record" not in SEMANTIC_MANIFEST_FIELDS
    base = {"mode": "C"}
    assert (measurement_semantics_digest({**base, "dataset_record": {"origin": "files"}})
            == measurement_semantics_digest({**base, "dataset_record": {"origin": "synthetic"}}))


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------
def _demo_workspace(root):
    from scripts.setup_demo import build_demo_kg
    from src.kg.workspace import SampleBudget, write_workspace

    write_workspace(build_demo_kg(), root, feature_dim=8,
                    samples=SampleBudget(num_train=20, num_val=5, val_disease_fraction=0.2))
    return root


#: The digest the stubbed model build records as the checkpoint read, as a real
#: build records its read's (contract M3c); no checkpoint bytes exist here.
STUB_CHECKPOINT_SHA256 = "c" * 64


def _built(self, checkpoint_path, device, *, expected_sha256=None):
    """Stands for a model built from one checkpoint read: it keeps the read's
    identity, as `_load_model_from_checkpoint` does, so the served identity names
    a verified checkpoint rather than an in-memory model."""
    from src.utils.fingerprint import ReadIdentity

    self._checkpoint_identity = ReadIdentity(
        path=Path(checkpoint_path), sha256=STUB_CHECKPOINT_SHA256)
    return object()


def _pipeline(monkeypatch, workspace, checkpoint_path, load=None):
    """Serving's own sequence with only the model build stubbed (`load`, which
    by default stands for a model built).

    `device="cpu"`: a checkpoint served with no device named asks for CUDA and
    is refused without it before the workspace is read (contract M3c, §5.2);
    these tests are about the workspace, so they name the development device."""
    import src.inference.pipeline as pipeline
    from src.kg.graph import KnowledgeGraph

    monkeypatch.setattr(
        pipeline.DiagnosisPipeline, "_load_model_from_checkpoint", load or _built,
    )
    monkeypatch.setattr(
        pipeline.DiagnosisPipeline, "_precompute_node_embeddings",
        lambda self, device=None: setattr(
            self, "_node_embeddings", {"phenotype": object(), "disease": object()}
        ),
    )
    return pipeline.DiagnosisPipeline(
        graph=KnowledgeGraph.read_json(workspace / "kg.json"),
        data_dir=str(workspace),
        checkpoint_path=str(checkpoint_path),
        device="cpu",
    )


@pytest.mark.parametrize("case", sorted(FAILURES))
def test_serving_refuses_before_the_checkpoint_is_read(tmp_path, monkeypatch, case):
    from src.kg.artifacts import DatasetRecordRefused

    tamper, binding, fragment = FAILURES[case]
    workspace = _demo_workspace(tmp_path / "ws")
    tamper(workspace)

    def forbidden(self, checkpoint_path, device, *, expected_sha256=None):
        raise AssertionError("the checkpoint was read over a refused workspace")

    with pytest.raises(DatasetRecordRefused, match=fragment) as refused:
        _pipeline(monkeypatch, workspace, tmp_path / "absent.pt", load=forbidden)

    assert refused.value.binding == binding


def test_serving_publishes_the_record_with_its_origin_apart(tmp_path, monkeypatch):
    from src.api.routes.pipeline import _status_of

    workspace = _demo_workspace(tmp_path / "ws")
    config = _pipeline(monkeypatch, workspace, tmp_path / "ck.pt").get_pipeline_config()

    record = config["dataset_record"]
    assert (record["binding"], record["origin"]) == ("verified", "synthetic")
    assert record["sha256"] == _sha(workspace / "kg.provenance.json")
    status = _status_of(config, str(workspace), str(tmp_path / "ck.pt"))
    assert status.dataset_record == record
    # The served identity names the same record, origin apart (contract M3c).
    served = status.served_identity.dataset_record
    assert (served.state, served.origin, served.sha256) == (
        "verified", "synthetic", record["sha256"])


def test_a_pipeline_over_no_workspace_publishes_no_record():
    """Nothing was verified, so nothing is claimed: an in-memory graph."""
    from scripts.setup_demo import build_demo_kg
    from src.inference.pipeline import DiagnosisPipeline

    assert DiagnosisPipeline(kg=build_demo_kg()).get_pipeline_config()["dataset_record"] is None


# ---------------------------------------------------------------------------
# The inventory and the one-time move
# ---------------------------------------------------------------------------
def test_the_inventory_shows_the_record_and_judges_no_checkpoint_over_a_refused_one(tmp_path):
    import scripts.audit_checkpoint_family as audit
    from scripts.train_model import read_training_inputs, training_input_digests

    root = tmp_path / "workspaces"
    sound = write_generated_workspace(root / "a", training_fields=True, **IDS)[0]
    broken = write_generated_workspace(root / "b", training_fields=True,
                                       train_ids=[5, 6], val_ids=[7])[0]
    for workspace in (sound, broken):
        record = training_input_digests(read_training_inputs(workspace))
        path = workspace / "checkpoints" / "hgt" / "good.pt"
        path.parent.mkdir(parents=True)
        torch.save({"state_dict": {"w": torch.zeros(2)}, "training_input_digests": record},
                   path)
    _missing(broken)

    report = audit.build_inventory(root, "unstated")
    a, b = report["workspaces"]["a"], report["workspaces"]["b"]

    assert a["dataset_record"]["verdict"] == "accepted"
    assert (a["dataset_record"]["binding"], a["dataset_record"]["origin"]) == (
        "verified", "synthetic")
    assert a["checkpoints"]["a/checkpoints/hgt/good.pt"]["m2_contract"][
        "serve_and_measure"]["verdict"] == "accepted"

    assert b["graph"]["verdict"] == "accepted"
    assert (b["dataset_record"]["verdict"], b["dataset_record"]["binding"]) == (
        "refused", "unverifiable")
    assert b["training"]["verdict"] == "refused"
    serve = b["checkpoints"]["b/checkpoints/hgt/good.pt"]["m2_contract"]["serve_and_measure"]
    assert serve == {"verdict": "not judged",
                     "reason": "the workspace's dataset record is refused"}
    assert report["counts"]["dataset_record:refused"] == 1
    assert report["counts"]["dataset_record origin:synthetic"] == 1


def test_one_unreadable_record_does_not_stop_the_inventory(tmp_path, monkeypatch):
    """A failure no rule names is listed as a refusal for that workspace, and the
    others are still listed."""
    import scripts.audit_checkpoint_family as audit
    import src.kg.artifacts as artifacts

    root = tmp_path / "workspaces"
    write_generated_workspace(root / "a", training_fields=True, **IDS)
    write_generated_workspace(root / "b", training_fields=True, train_ids=[5, 6], val_ids=[7])
    real = artifacts.verify_dataset_record

    def failing(manifest, kg_digest):
        if manifest.identity.path.parent.name == "b":
            raise OSError("the disk said no")
        return real(manifest, kg_digest)

    monkeypatch.setattr(artifacts, "verify_dataset_record", failing)
    report = audit.build_inventory(root, "unstated")

    assert report["workspaces"]["a"]["dataset_record"]["verdict"] == "accepted"
    b = report["workspaces"]["b"]["dataset_record"]
    assert (b["verdict"], b["binding"]) == ("refused", None)
    assert "the disk said no" in b["reason"]


def test_the_move_leaves_a_refused_workspaces_files_in_place(tmp_path):
    """Serving would refuse every file in it, so none is judged importable."""
    import scripts.migrate_checkpoints as tool
    from scripts.train_model import read_training_inputs, training_input_digests

    workspace = write_generated_workspace(tmp_path / "ws", training_fields=True, **IDS)[0]
    record = training_input_digests(read_training_inputs(workspace))
    path = workspace / "checkpoints" / "hgt" / "good.pt"
    path.parent.mkdir(parents=True)
    torch.save({"state_dict": {"w": torch.zeros(2)}, "training_input_digests": record,
                "config": {"model_config": {"conv_type": "hgt"}}}, path)
    _missing(workspace)

    [entry] = tool.classify(workspace)
    assert entry["kind"] == "unverifiable"
    assert "the workspace is refused" in entry["reason"]
    assert "DatasetRecordRefused" in entry["reason"]


# ---------------------------------------------------------------------------
# The writers
# ---------------------------------------------------------------------------
def test_a_manifest_that_would_bind_no_record_is_not_written(tmp_path):
    from scripts.setup_demo import build_demo_kg
    from src.kg.disease_allocation import allocate_diseases
    from src.kg.sample_generator import (
        build_eligible_disease_profiles,
        generate_training_samples,
    )
    from tests.fixtures.generated_workspace import default_graph_export

    kg = build_demo_kg()
    target = tmp_path / "ws"
    with pytest.raises(ValueError, match="would leave provenance unbound"):
        generate_training_samples(
            kg=kg, allocation=allocate_diseases(build_eligible_disease_profiles(kg, 2), 0.2,
                                                seed=42),
            num_train=20, num_val=5, output_dir=target,
            graph_digests={r: "0" * 64 for r in
                           ("kg", "node_features", "edge_indices", "num_nodes")},
            graph_export=default_graph_export(),
        )
    assert not target.exists()


def test_every_digest_the_workspace_writer_binds_is_of_the_bytes_written(tmp_path):
    """Computed here from the files with `hashlib`, never by the writer."""
    from src.kg.artifacts import GRAPH_ARTIFACTS

    workspace = _demo_workspace(tmp_path / "ws")
    bound = json.loads((workspace / "split_manifest.json").read_text())["artifacts"]

    for role, filename in {**GRAPH_ARTIFACTS, "provenance": "kg.provenance.json"}.items():
        assert bound[role] == _sha(workspace / filename), f"{role} is not the file's digest"


def test_the_graph_writers_return_the_digest_of_what_they_wrote(tmp_path):
    from scripts.setup_demo import build_demo_kg
    from src.kg.artifacts import GRAPH_ARTIFACTS
    from src.kg.graph import write_graph_data

    kg = build_demo_kg()
    assert kg.save_json(str(tmp_path / "kg.json")) == _sha(tmp_path / "kg.json")
    digests = write_graph_data(kg.export_graph_data(feature_dim=8), tmp_path)
    assert set(digests) == {"node_features", "edge_indices", "num_nodes"}
    for role, digest in digests.items():
        assert digest == _sha(tmp_path / GRAPH_ARTIFACTS[role])


def test_the_json_the_graph_writer_writes_is_what_json_dump_wrote(tmp_path):
    """The bytes did not move when the writer started digesting them: the same
    encoder options, in pieces, as UTF-8."""
    from scripts.setup_demo import build_demo_kg
    from src.kg.graph import _write_json

    graph = build_demo_kg()
    graph.save_json(str(tmp_path / "kg.json"))
    data = json.loads((tmp_path / "kg.json").read_text(encoding="utf-8"))
    expected = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    assert (tmp_path / "kg.json").read_bytes() == expected

    class Sink:
        def __init__(self):
            self.parts = []

        def write(self, data):
            self.parts.append(data)

    sink = Sink()
    payload = {"name": "é" * (1 << 20), "n": [1, 2]}
    _write_json(sink, payload, ensure_ascii=False, indent=2)
    assert len(sink.parts) > 1, "a large document is written in pieces"
    assert b"".join(sink.parts) == json.dumps(payload, ensure_ascii=False, indent=2).encode()


def test_the_record_writer_returns_the_digest_of_what_it_wrote(tmp_path):
    from src.kg.provenance import PROVENANCE_FILENAME, build_provenance, write_provenance

    digest = write_provenance(tmp_path, build_provenance("a" * 64, origin="synthetic"))
    assert digest == _sha(tmp_path / PROVENANCE_FILENAME)


def test_the_record_writer_refuses_a_record_every_consumer_refuses(tmp_path):
    from src.kg.provenance import ProvenanceError, build_provenance, write_provenance

    record = {**build_provenance("a" * 64, origin="synthetic"), "origin": "files"}
    with pytest.raises(ProvenanceError):
        write_provenance(tmp_path, record)
    assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------
def test_the_build_stops_rather_than_record_an_input_as_missing(monkeypatch, tmp_path):
    """An ontology the build parsed and cannot identify stops it, before the
    workspace exists. The check is not relaxed into a missing role."""
    from tests.unit.test_data_pipeline import TestBuildPathOrdering, _wide_kg_object

    build = TestBuildPathOrdering._stub_build_stages(monkeypatch, _wide_kg_object())
    TestBuildPathOrdering._satisfy_input_validation(tmp_path)

    class Unidentified:
        source_path = None
        source_digest = None
        declared_version = None

    monkeypatch.setattr(build, "_select_and_load", lambda loader, name, *a, **k: Unidentified())

    with pytest.raises(SystemExit, match="mondo ontology this build parsed cannot be identified"):
        build.build_knowledge_graph(
            external_dir=tmp_path, workspace=tmp_path / "ws",
            generate_samples=True, num_train=60, num_val=20,
        )
    assert not (tmp_path / "ws").exists()
