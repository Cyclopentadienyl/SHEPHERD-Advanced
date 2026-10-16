"""
What is served is identified, part by part (contract M3c, R8).
==============================================================
Every diagnosis result, `get_pipeline_config()`, `/diagnose` and
`/pipeline/status` carry one structured value naming what produced the answer:
the checkpoint, the graph and the workspace manifest by the digest of the bytes
read, the shortest-path table by its `build_id`, the dataset record by origin
and digest, the scoring mode, and the settings in effect. It replaces the two
version strings a result used to carry, a class constant and "unknown". What is
pinned here:

- each part refuses a state outside its vocabulary, a stated part without its
  digest or identifier, a digest that is not lowercase hexadecimal, and an
  absent part carrying a value; the whole refuses a contradiction between parts;
- the API's Pydantic mirror and the dataclasses have one shape and one set of
  states, and the mirror refuses an unknown key and a missing one;
- the identity a pipeline reports is built from its own reads: digests equal to
  `hashlib` over the files, the table's `build_id` and hop bound from its
  sidecar, the record's origin from the record, the device asked for and the one
  in effect;
- every result, valid, empty or refused, carries the pipeline's identity, and
  the API copies it from the result it answers with, never from shared state a
  reload could swap; the demo answer carries none;
- the reload route reports the metric of the checkpoint it served, from the
  build's own read (BACKLOG 19.2), calls it "best" only when the scan ranked the
  file by it, and says so when the file it chose was replaced before that read
  (19.1).

The device rule itself (`resolve_serving_device`) is in
`test_serving_device.py`.

Module: tests/unit/test_served_identity.py
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import shutil
import typing

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")

from src.core.types import (  # noqa: E402
    SERVED_CHECKPOINT_STATES,
    SERVED_DATASET_ORIGINS,
    SERVED_DATASET_RECORD_STATES,
    SERVED_GRAPH_STATES,
    SERVED_SCORING_MODES,
    SERVED_SHORTEST_PATHS_STATES,
    DataSource,
    InferenceResult,
    Node,
    NodeID,
    NodeType,
    PatientPhenotypes,
    ServedCheckpoint,
    ServedDatasetRecord,
    ServedGraph,
    ServedIdentity,
    ServedSettings,
    ServedShortestPaths,
)
from src.kg.graph import KnowledgeGraph  # noqa: E402
from tests.fixtures.generated_workspace import (  # noqa: E402
    graph_record,
    write_generated_workspace,
)
from tests.fixtures.opens import count_opens  # noqa: E402
from tests.fixtures.phenotype_kg import (  # noqa: E402
    DELAY,
    SEIZURE,
    UNKNOWN,
    build_two_phenotype_kg,
)
from tests.unit import test_serving_reads_once as reads_once  # noqa: E402
from tests.unit.test_serving_reads_once import _sha  # noqa: E402
from tests.unit.test_sp_artifact_binding import publish  # noqa: E402

#: Shared with `test_serving_reads_once.py` rather than copied: `api` is the app
#: state of a fresh process with SHEPHERD_DEVICE=cpu, `workspace` the synthetic
#: data directory and checkpoint its reload recipe uses.
api = reads_once.api
workspace = reads_once.workspace

SHA_A, SHA_B, SHA_C, SHA_D = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)
BUILD_ID = "0123456789abcdef0123456789abcdef"

#: The workspace directory's name, which `write_graph_export` puts in every node
#: identifier it writes.
WS = "ws"
#: A phenotype of that workspace's graph, with an edge to a disease.
PHENOTYPE = f"phenotype-0-of-{WS}"
#: A phenotype node with no edge at all, so path search finds nothing.
ISOLATED = "HP:0000001"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _checkpoint_for(data_dir, path, *, val_mrr=0.5):
    """A checkpoint of an untrained model over `data_dir`'s graph, recording
    that graph as training does (M2.2), so serving accepts it (M2.4).

    The model is built from the workspace's own tensors, so its metadata is the
    one `build_shepherd_model` derives from them; weights are seeded, not trained.
    """
    from src.kg.storage.file_storage import read_graph_artifacts
    from src.models.gnn.shepherd_gnn import ShepherdGNN, ShepherdGNNConfig

    graph_data = read_graph_artifacts(data_dir, map_location="cpu").graph_data
    x_dict, edge_index_dict = graph_data["x_dict"], graph_data["edge_index_dict"]
    torch.manual_seed(20261016)
    model = ShepherdGNN(
        metadata=(list(x_dict), list(edge_index_dict)),
        in_channels_dict={name: x.shape[1] for name, x in x_dict.items()},
        config=ShepherdGNNConfig(hidden_dim=8, num_layers=2, num_heads=2),
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"hidden_dim": 8, "num_layers": 2, "num_heads": 2},
            "epoch": 3,
            "logs": {"val_mrr": val_mrr},
            "training_input_digests": graph_record(data_dir),
        },
        path,
    )
    return path


@pytest.fixture
def served_workspace(tmp_path):
    """A generated workspace with a real graph, and a checkpoint trained on it.

    Returns ``(data_dir, checkpoint)``. No shortest-path table; `_with_table`
    publishes one.
    """
    data_dir, _ = write_generated_workspace(tmp_path / WS, train_ids=[0, 1], val_ids=[2])
    return data_dir, _checkpoint_for(data_dir, tmp_path / "checkpoint.pt")


def _with_table(data_dir, *, max_hops=4):
    """Publish a table bound to `data_dir`'s graph through the producer's own
    entry point; returns its sidecar as written."""
    from src.inference.sp_artifact import sidecar_path

    path = publish(data_dir, kg_digest=_sha(data_dir / "kg.json"),
                   extra={"max_hops": max_hops})
    return json.loads(sidecar_path(path).read_text())


def _full_pipeline(data_dir, checkpoint, config=None):
    from src.inference.pipeline import DiagnosisPipeline

    return DiagnosisPipeline(
        graph=KnowledgeGraph.read_json(data_dir / "kg.json"),
        data_dir=str(data_dir),
        checkpoint_path=str(checkpoint),
        config=config,
        device="cpu",
    )


def _patient(phenotypes, confidences=None):
    return PatientPhenotypes(patient_id="identity-patient", phenotypes=list(phenotypes),
                             phenotype_confidences=confidences)


# ---------------------------------------------------------------------------
# 1. The dataclasses refuse what no pipeline can serve
# ---------------------------------------------------------------------------
SETTINGS_WITH_MODEL = ServedSettings(
    device="cpu", device_requested="cpu", eta=0.7, max_path_length=4,
    path_length_penalty=0.9, aggregation_method="weighted_sum",
    max_paths_per_candidate=10, use_direct_path_optimization=True,
    phenotype_normalisation_version=1,
)
SETTINGS_NO_MODEL = dataclasses.replace(
    SETTINGS_WITH_MODEL, device=None, device_requested=None, eta=None
)
CHECKPOINTS = {
    "verified": ServedCheckpoint(state="verified", sha256=SHA_A),
    "supplied_in_memory": ServedCheckpoint(state="supplied_in_memory"),
    "absent": ServedCheckpoint(state="absent"),
}
GRAPHS = {
    "verified": ServedGraph(state="verified", kg_sha256=SHA_B, split_manifest_sha256=SHA_C),
    "unverified": ServedGraph(state="unverified", kg_sha256=SHA_B),
    "in_memory": ServedGraph(state="in_memory"),
}
TABLES = {
    "verified": ServedShortestPaths(state="verified", build_id=BUILD_ID, max_hops=5),
    "absent": ServedShortestPaths(state="absent"),
}
RECORDS = {
    "verified": ServedDatasetRecord(state="verified", origin="synthetic", sha256=SHA_D),
    "absent": ServedDatasetRecord(state="absent"),
}


def _identity(mode, checkpoint, graph, table, record):
    return ServedIdentity(
        scoring_mode=mode,
        checkpoint=CHECKPOINTS[checkpoint],
        graph=GRAPHS[graph],
        shortest_paths=TABLES[table],
        dataset_record=RECORDS[record],
        settings=SETTINGS_NO_MODEL if checkpoint == "absent" else SETTINGS_WITH_MODEL,
    )


#: Each a combination a pipeline builds: (mode, checkpoint, graph, table, record).
LEGAL = {
    "gnn and table over a workspace": (
        "gnn_plus_shortest_path", "verified", "verified", "verified", "verified"),
    "gnn alone over a workspace": (
        "gnn_only", "verified", "verified", "absent", "verified"),
    "a model in memory over a workspace with its table": (
        "gnn_plus_shortest_path", "supplied_in_memory", "verified", "verified", "verified"),
    "a model and graph data in memory": (
        "gnn_only", "supplied_in_memory", "in_memory", "absent", "absent"),
    "path reasoning over a graph file": (
        "path_reasoning_fallback", "absent", "unverified", "absent", "absent"),
    "path reasoning over a graph in memory": (
        "path_reasoning_fallback", "absent", "in_memory", "absent", "absent"),
}

#: Each part's refusals: (build, fragment of the refusal).
PART_REFUSALS = {
    "checkpoint state outside its vocabulary": (
        lambda: ServedCheckpoint(state="unknown"), "is not one of"),
    "verified checkpoint without its digest": (
        lambda: ServedCheckpoint(state="verified"), "lowercase hexadecimal SHA-256"),
    "checkpoint digest in uppercase": (
        lambda: ServedCheckpoint(state="verified", sha256=SHA_A.upper()), "lowercase"),
    "checkpoint digest not hexadecimal": (
        lambda: ServedCheckpoint(state="verified", sha256="g" * 64), "lowercase"),
    "checkpoint digest one character short": (
        lambda: ServedCheckpoint(state="verified", sha256="a" * 63), "lowercase"),
    "absent checkpoint carrying a digest": (
        lambda: ServedCheckpoint(state="absent", sha256=SHA_A), "carries no sha256"),
    "model in memory carrying a digest": (
        lambda: ServedCheckpoint(state="supplied_in_memory", sha256=SHA_A),
        "carries no sha256"),
    "graph state outside its vocabulary": (
        lambda: ServedGraph(state="cached", kg_sha256=SHA_B), "is not one of"),
    "verified graph without its kg digest": (
        lambda: ServedGraph(state="verified", split_manifest_sha256=SHA_C), "kg_sha256"),
    "verified graph without its manifest digest": (
        lambda: ServedGraph(state="verified", kg_sha256=SHA_B), "split_manifest_sha256"),
    "unverified graph without its kg digest": (
        lambda: ServedGraph(state="unverified"), "kg_sha256"),
    "kg digest in uppercase": (
        lambda: ServedGraph(state="unverified", kg_sha256=SHA_B.upper()), "lowercase"),
    "manifest digest not hexadecimal": (
        lambda: ServedGraph(state="verified", kg_sha256=SHA_B,
                            split_manifest_sha256="z" * 64), "lowercase"),
    "graph in memory carrying a kg digest": (
        lambda: ServedGraph(state="in_memory", kg_sha256=SHA_B), "carries no kg_sha256"),
    "unverified graph carrying a manifest digest": (
        lambda: ServedGraph(state="unverified", kg_sha256=SHA_B,
                            split_manifest_sha256=SHA_C), "checked against no manifest"),
    "graph in memory carrying a manifest digest": (
        lambda: ServedGraph(state="in_memory", split_manifest_sha256=SHA_C),
        "checked against no manifest"),
    "table state outside its vocabulary": (
        lambda: ServedShortestPaths(state="unrecorded"), "is not one of"),
    "verified table without its build_id": (
        lambda: ServedShortestPaths(state="verified", max_hops=5), "build_id"),
    "verified table with an empty build_id": (
        lambda: ServedShortestPaths(state="verified", build_id="", max_hops=5), "build_id"),
    "verified table without its hop bound": (
        lambda: ServedShortestPaths(state="verified", build_id=BUILD_ID), "max_hops"),
    "verified table with a boolean hop bound": (
        lambda: ServedShortestPaths(state="verified", build_id=BUILD_ID, max_hops=True),
        "max_hops"),
    "absent table carrying a build_id": (
        lambda: ServedShortestPaths(state="absent", build_id=BUILD_ID), "carries no build_id"),
    "absent table carrying a hop bound": (
        lambda: ServedShortestPaths(state="absent", max_hops=5), "carries no build_id"),
    "record state outside its vocabulary": (
        lambda: ServedDatasetRecord(state="mismatch"), "is not one of"),
    "verified record with an origin outside its vocabulary": (
        lambda: ServedDatasetRecord(state="verified", origin="unknown", sha256=SHA_D),
        "origin"),
    "verified record without its origin": (
        lambda: ServedDatasetRecord(state="verified", sha256=SHA_D), "origin"),
    "verified record without its digest": (
        lambda: ServedDatasetRecord(state="verified", origin="files"), "lowercase"),
    "record digest in uppercase": (
        lambda: ServedDatasetRecord(state="verified", origin="files", sha256=SHA_D.upper()),
        "lowercase"),
    "absent record carrying an origin": (
        lambda: ServedDatasetRecord(state="absent", origin="synthetic"), "carries no origin"),
    "absent record carrying a digest": (
        lambda: ServedDatasetRecord(state="absent", sha256=SHA_D), "carries no origin"),
}

#: Each combination of valid parts that contradict one another, built so only
#: the rule named fails: (mode, checkpoint, graph, table, record), fragment.
CONTRADICTIONS = {
    "a model scored by path reasoning": (
        ("path_reasoning_fallback", "supplied_in_memory", "in_memory", "absent", "absent"),
        "contradicts a checkpoint"),
    "a verified checkpoint scored by path reasoning": (
        ("path_reasoning_fallback", "verified", "verified", "absent", "verified"),
        "contradicts a checkpoint"),
    "GNN scoring with no model": (
        ("gnn_only", "absent", "in_memory", "absent", "absent"),
        "contradicts a checkpoint"),
    "shortest-path scoring with no table": (
        ("gnn_plus_shortest_path", "verified", "verified", "absent", "verified"),
        "contradicts a shortest-path table"),
    "a verified table that GNN-only scoring leaves out": (
        ("gnn_only", "verified", "verified", "verified", "verified"),
        "contradicts a shortest-path table"),
    "a verified graph without its record": (
        ("gnn_only", "verified", "verified", "absent", "absent"),
        "dataset record is verified exactly when"),
    "a verified record over an unverified graph": (
        ("path_reasoning_fallback", "absent", "unverified", "absent", "verified"),
        "dataset record is verified exactly when"),
    "a verified checkpoint over an unverified graph": (
        ("gnn_only", "verified", "unverified", "absent", "absent"),
        "verified checkpoint is served only over a verified workspace graph"),
    "a verified checkpoint over a graph in memory": (
        ("gnn_only", "verified", "in_memory", "absent", "absent"),
        "verified checkpoint is served only over a verified workspace graph"),
    "a verified table over a graph in memory": (
        ("gnn_plus_shortest_path", "supplied_in_memory", "in_memory", "verified", "absent"),
        "verified shortest-path table is served only over a verified workspace graph"),
    "a verified table over an unverified graph": (
        ("gnn_plus_shortest_path", "supplied_in_memory", "unverified", "verified", "absent"),
        "verified shortest-path table is served only over a verified workspace graph"),
}


@pytest.mark.parametrize("case", sorted(PART_REFUSALS))
def test_each_part_refuses_a_value_its_state_does_not_allow(case):
    """A part is either stated with what identifies it, or absent with nothing.
    Catches a part that accepts a blank, a placeholder digest, or a value beside
    a state that says there is none, which a reader would have to interpret."""
    build, fragment = PART_REFUSALS[case]

    with pytest.raises(ValueError, match=fragment):
        build()


@pytest.mark.parametrize("case", sorted(CONTRADICTIONS))
def test_the_identity_refuses_parts_that_contradict_each_other(case):
    """Valid parts can still describe no pipeline that exists. Catches a builder
    that, say, reports a verified checkpoint over a graph nothing verified, or a
    shortest-path table beside a mode that does not use one."""
    combination, fragment = CONTRADICTIONS[case]

    with pytest.raises(ValueError, match=fragment):
        _identity(*combination)


def test_the_identity_refuses_a_scoring_mode_outside_its_vocabulary():
    with pytest.raises(ValueError, match="is not one of"):
        dataclasses.replace(_identity(*LEGAL["gnn alone over a workspace"]),
                            scoring_mode="gnn")


@pytest.mark.parametrize("case", sorted(LEGAL))
def test_every_combination_a_pipeline_builds_is_accepted_and_round_trips(case):
    """The rules refuse contradictions, not the states a pipeline really serves.
    `to_dict()` is JSON primitives and is accepted unchanged by the response's
    schema: catches a part that serialises to something the API refuses."""
    from src.api.routes.diagnose import ServedIdentity as ResponseIdentity

    identity = _identity(*LEGAL[case])
    as_dict = identity.to_dict()

    assert json.loads(json.dumps(as_dict)) == as_dict
    assert ResponseIdentity.model_validate(as_dict).model_dump() == as_dict
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.scoring_mode = "gnn_only"


def test_a_result_cannot_be_built_without_an_identity():
    """Required, and the version strings it replaces are gone: catches a result
    path that could still return without saying what produced it."""
    from datetime import datetime

    with pytest.raises(TypeError, match="served_identity"):
        InferenceResult(patient_id="p", timestamp=datetime.now(), candidates=[])
    fields = {field.name for field in dataclasses.fields(InferenceResult)}
    assert "served_identity" in fields
    assert not {"model_version", "kg_version"} & fields


# ---------------------------------------------------------------------------
# 2. One shape, one set of states, in the dataclasses and the API
# ---------------------------------------------------------------------------
def _response_models():
    from src.api.routes import diagnose

    return {
        ServedIdentity: diagnose.ServedIdentity,
        ServedCheckpoint: diagnose.ServedCheckpoint,
        ServedGraph: diagnose.ServedGraph,
        ServedShortestPaths: diagnose.ServedShortestPaths,
        ServedDatasetRecord: diagnose.ServedDatasetRecord,
        ServedSettings: diagnose.ServedSettings,
    }


def _literal_values(annotation):
    """The values of a `Literal`, or of the `Literal` inside an `Optional`."""
    if typing.get_origin(annotation) is typing.Literal:
        return set(typing.get_args(annotation))
    values = set()
    for argument in typing.get_args(annotation):
        if argument is not type(None):
            values |= _literal_values(argument)
    return values


def test_each_part_has_the_same_fields_in_both_definitions():
    """The API defines its own mirror, because `scripts/measure_served_pipeline.py`
    loads diagnose.py without the application. Catches a field added, renamed or
    dropped on one side only."""
    for part, model in _response_models().items():
        assert {field.name for field in dataclasses.fields(part)} == set(
            model.model_fields
        ), part.__name__


def test_each_state_vocabulary_is_the_same_in_both_definitions():
    """Catches a state added to the dataclass that every response carrying it
    would fail on, or one the API accepts and no pipeline can produce."""
    from src.api.routes import diagnose
    from src.kg.provenance import ORIGINS

    vocabularies = [
        (diagnose.ServedCheckpoint, "state", SERVED_CHECKPOINT_STATES),
        (diagnose.ServedGraph, "state", SERVED_GRAPH_STATES),
        (diagnose.ServedShortestPaths, "state", SERVED_SHORTEST_PATHS_STATES),
        (diagnose.ServedDatasetRecord, "state", SERVED_DATASET_RECORD_STATES),
        (diagnose.ServedIdentity, "scoring_mode", SERVED_SCORING_MODES),
        # The record's origin, against the record reader's own vocabulary and
        # the dataclass's copy of it.
        (diagnose.ServedDatasetRecord, "origin", ORIGINS),
        (diagnose.ServedDatasetRecord, "origin", SERVED_DATASET_ORIGINS),
    ]
    for model, name, states in vocabularies:
        assert len(set(states)) == len(states), states
        assert _literal_values(model.model_fields[name].annotation) == set(states), (
            f"{model.__name__}.{name}"
        )


def test_the_dataclass_accepts_exactly_the_record_readers_origins():
    """`src.core` cannot import the record reader, so the dataclass checks
    origins against a copy of `src.kg.provenance.ORIGINS`
    (`SERVED_DATASET_ORIGINS`), the one state vocabulary not shared (contract
    M3c). Catches an origin added to the reader and the API alone: the dataclass
    would refuse it, so every identity over such a workspace would raise, and
    every startup and reload over it would be refused. Also catches a dataclass
    check that ignores the copy."""
    from src.api.routes import diagnose
    from src.kg.provenance import ORIGINS

    assert SERVED_DATASET_ORIGINS == ORIGINS
    assert set(SERVED_DATASET_ORIGINS) == _literal_values(
        diagnose.ServedDatasetRecord.model_fields["origin"].annotation
    )
    for origin in ORIGINS:
        record = ServedDatasetRecord(state="verified", origin=origin, sha256=SHA_D)
        assert record.origin == origin


def _nested_dicts(value, path=()):
    """Every dictionary in `value` with the path of keys that reaches it."""
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from _nested_dicts(child, (*path, key))


def _at(value, path):
    for key in path:
        value = value[key]
    return value


def test_the_response_schema_refuses_an_unknown_key_at_every_level():
    """The shape is the contract: catches a mirror that would quietly accept a
    field the dataclass has and it does not know, and so drift unnoticed."""
    from pydantic import ValidationError

    from src.api.routes.diagnose import ServedIdentity as ResponseIdentity

    identity = _identity(*LEGAL["gnn and table over a workspace"]).to_dict()
    for path, _ in list(_nested_dicts(identity)):
        tampered = json.loads(json.dumps(identity))
        _at(tampered, path)["not_in_the_contract"] = 1
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ResponseIdentity.model_validate(tampered)


def test_the_response_schema_requires_every_key():
    """Null is a value; a missing key is not. Catches a key given a default,
    which would let a response omit a part instead of stating its absence."""
    from pydantic import ValidationError

    from src.api.routes.diagnose import ServedIdentity as ResponseIdentity

    for model in _response_models().values():
        optional = [name for name, field in model.model_fields.items()
                    if not field.is_required()]
        assert optional == [], f"{model.__name__}: {optional}"

    identity = _identity(*LEGAL["path reasoning over a graph in memory"]).to_dict()
    for path, mapping in list(_nested_dicts(identity)):
        for key in mapping:
            tampered = json.loads(json.dumps(identity))
            del _at(tampered, path)[key]
            with pytest.raises(ValidationError, match="Field required"):
                ResponseIdentity.model_validate(tampered)


#: Every digest the identity carries, by its path in the JSON form, with the
#: dataclass part that holds it.
DIGEST_PATHS = {
    "checkpoint.sha256": (("checkpoint", "sha256"), ServedCheckpoint),
    "graph.kg_sha256": (("graph", "kg_sha256"), ServedGraph),
    "graph.split_manifest_sha256": (("graph", "split_manifest_sha256"), ServedGraph),
    "dataset_record.sha256": (("dataset_record", "sha256"), ServedDatasetRecord),
}
#: Values that are not a lowercase hexadecimal SHA-256.
BAD_DIGESTS = {
    "uppercase": SHA_A.upper(),
    "63 characters": "a" * 63,
    "not hexadecimal": "g" * 64,
}


@pytest.mark.parametrize("value", sorted(BAD_DIGESTS))
@pytest.mark.parametrize("digest", sorted(DIGEST_PATHS))
def test_the_response_schema_refuses_a_digest_the_dataclass_refuses(digest, value):
    """`scripts/measure_served_pipeline.py` checks a live service's identity
    with this mirror before citing it as the readings' subject, so the digest
    rule holds on every digest, as it does in the dataclass. Catches the pattern
    dropped or loosened on any one field, which would let an uppercase,
    truncated or non-hexadecimal digest through."""
    from pydantic import ValidationError

    from src.api.routes.diagnose import ServedIdentity as ResponseIdentity

    path, part = DIGEST_PATHS[digest]
    identity = _identity(*LEGAL["gnn and table over a workspace"]).to_dict()
    _at(identity, path[:-1])[path[-1]] = BAD_DIGESTS[value]

    with pytest.raises(ValueError, match="lowercase"):
        part(**_at(identity, path[:-1]))
    with pytest.raises(ValidationError, match=re.escape(digest)) as refusal:
        ResponseIdentity.model_validate(identity)
    assert "String should match pattern" in str(refusal.value)


# ---------------------------------------------------------------------------
# 3. A pipeline's identity is built from the reads it took
# ---------------------------------------------------------------------------
def _settings_of(config, *, device, device_requested, eta):
    from src.kg.phenotype_normalisation import PHENOTYPE_NORMALISATION_VERSION

    return ServedSettings(
        device=device, device_requested=device_requested, eta=eta,
        max_path_length=config.max_path_length,
        path_length_penalty=config.path_length_penalty,
        aggregation_method=config.aggregation_method,
        max_paths_per_candidate=config.max_paths_per_candidate,
        use_direct_path_optimization=config.use_direct_path_optimization,
        phenotype_normalisation_version=PHENOTYPE_NORMALISATION_VERSION,
    )


#: Every path setting the identity records, each away from its default, so an
#: identity built from the defaults instead of what is in effect cannot match.
NON_DEFAULT_PATHS = dict(max_path_length=3, path_length_penalty=0.8,
                         aggregation_method="max", max_paths_per_candidate=7,
                         use_direct_path_optimization=False)


def test_a_path_only_pipeline_over_a_graph_in_memory_states_every_absence():
    """No model, no file, no table, no workspace: each said in words. The path
    settings are the configuration's, since path search discovers the
    candidates in every mode. Non-default values catch a builder that reports
    the defaults instead of what is in effect."""
    from src.inference.pipeline import DiagnosisPipeline, PipelineConfig

    config = PipelineConfig(**NON_DEFAULT_PATHS)
    pipeline = DiagnosisPipeline(kg=build_two_phenotype_kg(), config=config)

    identity = pipeline.served_identity()

    assert identity.scoring_mode == "path_reasoning_fallback"
    assert identity.checkpoint == ServedCheckpoint(state="absent")
    assert identity.graph == ServedGraph(state="in_memory")
    assert identity.shortest_paths == ServedShortestPaths(state="absent")
    assert identity.dataset_record == ServedDatasetRecord(state="absent")
    assert identity.settings == _settings_of(config, device=None, device_requested=None,
                                             eta=None)
    assert pipeline.get_pipeline_config()["served_identity"] == identity.to_dict()
    assert "version" not in pipeline.get_pipeline_config()


def test_a_graph_read_from_a_file_and_checked_against_nothing_is_unverified(tmp_path):
    """BACKLOG 19.20: path-only serving reads no manifest, and says so. The
    digest is the read's, equal to `hashlib` over the file: catches a digest
    computed from another read, or a graph claimed verified that nothing
    checked."""
    from src.inference.pipeline import DiagnosisPipeline

    kg_file = tmp_path / "kg.json"
    build_two_phenotype_kg().save_json(str(kg_file))
    read = KnowledgeGraph.read_json(kg_file)

    identity = DiagnosisPipeline(graph=read).served_identity()

    assert read.identity.sha256 == _sha(kg_file)
    assert identity.graph == ServedGraph(state="unverified", kg_sha256=read.identity.sha256)
    assert identity.scoring_mode == "path_reasoning_fallback"
    assert identity.checkpoint.state == "absent"
    assert identity.dataset_record.state == "absent"


def test_a_workspace_beside_a_path_only_pipeline_is_not_claimed_verified(
    served_workspace,
):
    """With no model, no manifest is read even when a data directory is named
    (19.20), so the graph is unverified and no record is claimed: catches an
    identity that borrows the workspace's verification without performing it."""
    from src.inference.pipeline import DiagnosisPipeline

    data_dir, _ = served_workspace
    read = KnowledgeGraph.read_json(data_dir / "kg.json")

    with count_opens() as opens:
        identity = DiagnosisPipeline(graph=read, data_dir=str(data_dir)).served_identity()

    assert opens.opens(data_dir / "split_manifest.json") == 0
    assert identity.graph == ServedGraph(state="unverified", kg_sha256=_sha(data_dir / "kg.json"))
    assert identity.dataset_record.state == "absent"


def test_the_service_build_without_a_checkpoint_reports_an_unverified_graph(
    api, served_workspace
):
    data_dir, _ = served_workspace

    bundle = api.build_pipeline(kg_path=str(data_dir / "kg.json"), data_dir=str(data_dir))

    identity = bundle.config["served_identity"]
    assert identity["scoring_mode"] == "path_reasoning_fallback"
    assert identity["graph"] == {"state": "unverified",
                                 "kg_sha256": _sha(data_dir / "kg.json"),
                                 "split_manifest_sha256": None}
    assert identity["settings"]["device"] is None
    assert identity["settings"]["device_requested"] is None


def test_a_full_workspace_with_a_bound_table_is_verified_in_every_part(served_workspace):
    """Each value against an independent account of the files: `hashlib` over
    their bytes, the sidecar as written, the record as written. Catches any
    part filled from somewhere other than the read the pipeline served from."""
    from src.inference.pipeline import PipelineConfig
    from src.kg.provenance import PROVENANCE_FILENAME

    data_dir, checkpoint = served_workspace
    sidecar = _with_table(data_dir, max_hops=4)
    config = PipelineConfig(eta=0.55)

    pipeline = _full_pipeline(data_dir, checkpoint, config)
    identity = pipeline.served_identity()

    assert identity.scoring_mode == "gnn_plus_shortest_path"
    assert identity.checkpoint == ServedCheckpoint(state="verified", sha256=_sha(checkpoint))
    assert identity.graph == ServedGraph(
        state="verified", kg_sha256=_sha(data_dir / "kg.json"),
        split_manifest_sha256=_sha(data_dir / "split_manifest.json"),
    )
    assert identity.shortest_paths == ServedShortestPaths(
        state="verified", build_id=sidecar["build_id"], max_hops=4
    )
    record = json.loads((data_dir / PROVENANCE_FILENAME).read_text())
    assert identity.dataset_record == ServedDatasetRecord(
        state="verified", origin=record["origin"], sha256=_sha(data_dir / PROVENANCE_FILENAME)
    )
    assert identity.settings == _settings_of(config, device="cpu", device_requested="cpu",
                                             eta=0.55)
    assert pipeline.get_pipeline_config()["served_identity"] == identity.to_dict()


def test_a_full_workspace_without_a_table_serves_the_gnn_alone(served_workspace):
    """The mixing weight in effect is 1.0, not the configured one: catches an
    identity that reports the configuration rather than what scores."""
    from src.inference.pipeline import PipelineConfig

    data_dir, checkpoint = served_workspace

    identity = _full_pipeline(data_dir, checkpoint, PipelineConfig(eta=0.55)).served_identity()

    assert identity.scoring_mode == "gnn_only"
    assert identity.shortest_paths == ServedShortestPaths(state="absent")
    assert identity.settings.eta == 1.0
    assert identity.checkpoint.sha256 == _sha(checkpoint)
    assert identity.graph.state == identity.dataset_record.state == "verified"


# ---------------------------------------------------------------------------
# 4. Every result carries the identity of what produced it
# ---------------------------------------------------------------------------
def _kg_with_an_isolated_phenotype():
    kg = build_two_phenotype_kg()
    kg.add_node(Node(id=NodeID(source=DataSource.HPO, local_id=ISOLATED),
                     node_type=NodeType.PHENOTYPE, name=ISOLATED))
    return kg


#: (validate_phenotypes, phenotypes, confidences, kind, warning fragment)
RESULT_PATHS = {
    "scored": (True, [SEIZURE, DELAY], None, "scored", None),
    "empty, no phenotype in the graph": (
        False, [UNKNOWN], None, "empty", "No valid phenotypes found in knowledge graph"),
    "empty, no path": (True, [ISOLATED], None, "empty", "No paths found"),
    "refused, no valid phenotype": (
        True, [UNKNOWN], None, "empty", "No valid phenotypes after validation"),
    "refused, a confidence list of another length": (
        True, [SEIZURE, DELAY], [0.9], "empty", "phenotype_confidences has 1 entries"),
}


@pytest.mark.parametrize("case", sorted(RESULT_PATHS))
def test_every_result_path_carries_the_pipelines_identity(case):
    """Valid, empty and refused results are built in three places; catches one
    of them returning without the identity, or with one built differently. The
    path settings are the only values a path-only pipeline's identity has to
    lose, so they are not the defaults: an identity built from a default
    configuration differs."""
    from src.inference.pipeline import DiagnosisPipeline, PipelineConfig

    validate, phenotypes, confidences, kind, warning = RESULT_PATHS[case]
    pipeline = DiagnosisPipeline(
        kg=_kg_with_an_isolated_phenotype(),
        config=PipelineConfig(validate_phenotypes=validate, **NON_DEFAULT_PATHS),
    )

    result = pipeline.run(_patient(phenotypes, confidences), top_k=5)

    assert bool(result.candidates) == (kind == "scored"), result.warnings
    if warning:
        assert any(warning in line for line in result.warnings), result.warnings
    assert isinstance(result.served_identity, ServedIdentity)
    assert result.served_identity == pipeline.served_identity()
    assert result.served_identity.to_dict() == pipeline.get_pipeline_config()["served_identity"]
    assert result.served_identity.settings != _settings_of(
        PipelineConfig(), device=None, device_requested=None, eta=None
    ), "the path settings in effect are the defaults; this case cannot tell them apart"


#: The same three places over the full pipeline: (validate_phenotypes,
#: phenotypes, kind, warning fragment). Validation off lets an id the graph
#: lacks reach the empty-result builder.
FULL_RESULT_PATHS = {
    "scored": (True, [PHENOTYPE], "scored", None),
    "empty, no phenotype in the graph": (
        False, ["HP:9999999"], "empty", "No valid phenotypes found in knowledge graph"),
    "refused, no valid phenotype": (
        True, ["HP:9999999"], "empty", "No valid phenotypes after validation"),
}


@pytest.mark.parametrize("case", sorted(FULL_RESULT_PATHS))
def test_a_full_pipelines_results_carry_its_full_identity(served_workspace, case):
    """The same over a checkpoint and a bound table, where every part has a
    value to lose on the way to the result. An answer that found nothing to rank
    was still produced by the model and table served: catches an empty result
    that says no model or table produced it, an identity legal over a verified
    workspace, so no invariant refuses it."""
    from src.inference.pipeline import PipelineConfig

    validate, phenotypes, kind, warning = FULL_RESULT_PATHS[case]
    data_dir, checkpoint = served_workspace
    _with_table(data_dir)
    pipeline = _full_pipeline(data_dir, checkpoint,
                              PipelineConfig(validate_phenotypes=validate))

    result = pipeline.run(_patient(phenotypes), top_k=5)

    assert bool(result.candidates) == (kind == "scored"), result.warnings
    if warning:
        assert any(warning in line for line in result.warnings), result.warnings
    assert result.served_identity == pipeline.served_identity()
    assert result.served_identity.scoring_mode == "gnn_plus_shortest_path"
    assert result.served_identity.to_dict() == pipeline.get_pipeline_config()["served_identity"]


# ---------------------------------------------------------------------------
# 5. The API copies the identity from the result it answers with
# ---------------------------------------------------------------------------
@pytest.fixture
def app(monkeypatch):
    """The real application, with no deployment configured and nothing served."""
    from src.api import main as api_main

    monkeypatch.setattr(api_main.app_state, "pipeline", None, raising=False)
    monkeypatch.setattr(api_main.app_state, "real_pipeline_requested", False, raising=False)
    for name in ("SHEPHERD_KG_PATH", "SHEPHERD_DATA_DIR", "SHEPHERD_CHECKPOINT_PATH"):
        monkeypatch.delenv(name, raising=False)
    return api_main


def _client(api_main):
    from fastapi.testclient import TestClient

    return TestClient(api_main.app)


def test_a_diagnosis_carries_the_identity_of_the_pipeline_that_answered(app, monkeypatch):
    from src.inference.pipeline import DiagnosisPipeline

    pipeline = DiagnosisPipeline(kg=build_two_phenotype_kg())
    monkeypatch.setattr(app.app_state, "pipeline", pipeline)

    response = _client(app).post("/api/v1/diagnose", json={"phenotypes": [SEIZURE, DELAY]})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["candidates"]
    assert body["served_identity"] == pipeline.get_pipeline_config()["served_identity"]
    assert "model_version" not in body


def test_a_diagnosis_from_a_full_pipeline_carries_every_digest(app, monkeypatch, served_workspace):
    """The nested values survive the response model and the encoder: catches a
    field the schema drops or a value it coerces."""
    data_dir, checkpoint = served_workspace
    _with_table(data_dir)
    pipeline = _full_pipeline(data_dir, checkpoint)
    monkeypatch.setattr(app.app_state, "pipeline", pipeline)

    body = _client(app).post("/api/v1/diagnose", json={"phenotypes": [PHENOTYPE]}).json()

    assert body["candidates"]
    assert body["served_identity"] == pipeline.served_identity().to_dict()
    assert body["served_identity"]["checkpoint"]["sha256"] == _sha(checkpoint)


def test_the_demo_answer_identifies_nothing_and_says_it_is_a_demo(app):
    """Nothing was served, so the key is present and null; the warning stays."""
    body = _client(app).post("/api/v1/diagnose", json={"phenotypes": [SEIZURE]}).json()

    assert "served_identity" in body
    assert body["served_identity"] is None
    assert "Pipeline not fully initialized - using mock data" in body["warnings"]
    assert body["candidates"], "the demo still answers"


def test_the_identity_is_the_results_even_when_a_reload_lands_mid_request(app, monkeypatch):
    """A reload can publish another pipeline while a diagnosis runs. Catches a
    route that reads the identity from shared state after `run()`, which would
    attribute this result to a pipeline that did not produce it."""
    from src.inference.pipeline import DiagnosisPipeline, PipelineConfig

    answering = DiagnosisPipeline(kg=build_two_phenotype_kg())
    published = DiagnosisPipeline(kg=build_two_phenotype_kg(),
                                  config=PipelineConfig(max_path_length=2))
    assert answering.served_identity() != published.served_identity()
    original = answering.run

    def run_then_reload(**kwargs):
        result = original(**kwargs)
        app.app_state.pipeline = published
        return result

    monkeypatch.setattr(answering, "run", run_then_reload)
    monkeypatch.setattr(app.app_state, "pipeline", answering)

    body = _client(app).post("/api/v1/diagnose", json={"phenotypes": [SEIZURE]}).json()

    assert app.app_state.pipeline is published, "the reload did land"
    assert body["served_identity"] == answering.served_identity().to_dict()


def test_the_status_carries_the_live_pipelines_identity(app, monkeypatch, served_workspace):
    data_dir, checkpoint = served_workspace
    _with_table(data_dir)
    pipeline = _full_pipeline(data_dir, checkpoint)
    monkeypatch.setattr(app.app_state, "pipeline", pipeline)

    body = _client(app).get("/api/v1/pipeline/status").json()

    assert body["initialized"] is True
    assert body["served_identity"] == pipeline.get_pipeline_config()["served_identity"]
    # The top-level fields stay beside it.
    assert body["scoring_mode"] == body["served_identity"]["scoring_mode"]
    assert body["checkpoint_meta"]["device"] == "cpu"


def test_the_status_with_nothing_loaded_identifies_nothing(app):
    body = _client(app).get("/api/v1/pipeline/status").json()

    assert body["initialized"] is False
    assert body["served_identity"] is None


def test_a_diagnose_response_without_the_identity_key_is_refused():
    """Required-but-nullable: catches a default that would let a response path
    leave the key out instead of answering it."""
    from pydantic import ValidationError

    from src.api.routes.diagnose import DiagnoseResponse

    response = {"session_id": "s", "patient_id": "p", "timestamp": "t",
                "candidates": [], "inference_time_ms": 0.0}

    with pytest.raises(ValidationError, match="served_identity"):
        DiagnoseResponse.model_validate(response)
    assert DiagnoseResponse.model_validate({**response, "served_identity": None})


# ---------------------------------------------------------------------------
# 6. The reload route reports what it served (BACKLOG 19.1, 19.2)
# ---------------------------------------------------------------------------
def _with_val_mrr(source, target, val_mrr, *, shift=0.0):
    """`source` with another metric in its logs and, with `shift`, other weights."""
    contents = torch.load(source, map_location="cpu", weights_only=False)
    contents["logs"] = {"val_mrr": val_mrr}
    if shift:
        contents["state_dict"] = {
            name: value + shift if value.is_floating_point() else value
            for name, value in contents["state_dict"].items()
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(contents, target)
    return target


def _run_checkpoint(data_dir):
    return (data_dir / "runs" / "gat" / "20261015-000000_run_aaaaaa" / "checkpoints"
            / "model.pt")


def test_the_reported_metric_is_the_loaded_files_when_the_selection_was_replaced(
    api, workspace, tmp_path, monkeypatch
):
    """Chose A, reported B, served C is gone: the metric shown comes from the
    build's own read, and the reason says the choice was made on earlier bytes.
    Catches a metric read from the scan, or from a second read of the path."""
    from src.api.routes.pipeline import PipelineReloadRequest, reload_pipeline

    data_dir, checkpoint = workspace
    selected = _with_val_mrr(checkpoint, _run_checkpoint(data_dir), 0.25)
    scored = _sha(selected)
    replacement = _with_val_mrr(checkpoint, tmp_path / "b.pt", 0.8123, shift=1.0)
    loaded = _sha(replacement)
    real_build = api.build_pipeline

    def replace_then_build(**kwargs):
        shutil.copyfile(replacement, kwargs["checkpoint_path"])
        return real_build(**kwargs)

    monkeypatch.setattr(api, "build_pipeline", replace_then_build)

    result = asyncio.run(reload_pipeline(PipelineReloadRequest(data_dir=str(data_dir))))

    assert result.success, result.message
    assert "best val_mrr=0.8123" in result.selection_reason
    assert "0.2500" not in result.selection_reason
    assert f"(scored sha256 {scored}, loaded sha256 {loaded})" in result.selection_reason
    assert result.selection_reason.endswith(
        "the choice was made on the earlier bytes, and the metric shown is the loaded file's"
    )
    assert result.status.served_identity.checkpoint.sha256 == loaded
    assert result.status.checkpoint_meta["val_mrr"] == pytest.approx(0.8123)


def test_a_replacement_without_a_metric_is_not_said_to_show_one(
    api, workspace, tmp_path, monkeypatch
):
    """The same replacement, with loaded bytes that carry no ranking metric:
    nothing is shown, so the note does not say what the metric shown is, and
    the scan's value for the earlier bytes is not reported as the loaded
    file's. Catches the clause appended whether or not a metric was."""
    from src.api.routes.pipeline import PipelineReloadRequest, reload_pipeline

    data_dir, checkpoint = workspace
    selected = _with_val_mrr(checkpoint, _run_checkpoint(data_dir), 0.25)
    scored = _sha(selected)
    replacement = _with_val_mrr(checkpoint, tmp_path / "b.pt", None, shift=1.0)
    real_build = api.build_pipeline

    def replace_then_build(**kwargs):
        shutil.copyfile(replacement, kwargs["checkpoint_path"])
        return real_build(**kwargs)

    monkeypatch.setattr(api, "build_pipeline", replace_then_build)

    result = asyncio.run(reload_pipeline(PipelineReloadRequest(data_dir=str(data_dir))))

    assert result.success, result.message
    assert result.checkpoint_path == str(selected)
    assert result.selection_reason.endswith(
        f"(scored sha256 {scored}, loaded sha256 {_sha(replacement)}): "
        "the choice was made on the earlier bytes"
    )
    assert "metric shown" not in result.selection_reason
    assert "val_mrr" not in result.selection_reason
    assert "val_mrr" not in result.status.checkpoint_meta


#: Two candidates in one run, `model-a.pt` and `last.pt`, with these metrics in
#: their logs: (model-a.pt's, last.pt's, the file served, how the reason ends).
#: A 0-dim tensor is a metric the scan does not rank (it ranks Python numbers),
#: and the build's metadata turns it into a float.
SELECTIONS = {
    "ranked by a float metric": (0.9, 0.1, "model-a.pt", "; best val_mrr=0.9000"),
    "tensor metrics the scan does not rank": (
        torch.tensor(0.9), torch.tensor(0.1), "last.pt",
        "; not ranked by metric (no candidate carries one): last.pt, else the newest "
        "file; loaded file val_mrr=0.1000"),
    "no metric at all": (
        None, None, "last.pt",
        "; not ranked by metric (no candidate carries one): last.pt, else the newest "
        "file"),
}


@pytest.mark.parametrize("case", sorted(SELECTIONS))
def test_best_is_said_only_of_a_file_ranked_by_its_metric(api, workspace, case):
    """BACKLOG 19.2: the reason states how the file was chosen. When no
    candidate is ranked, selection falls back to `last.pt`, and the loaded
    file's metric, which its metadata may still have, is shown as that file's,
    never as the best: catches a fallback pick labelled best while a better
    candidate existed (here 0.1 beside 0.9), the loaded file's metric left out,
    and a ranked pick that loses "best"."""
    from src.api.routes.pipeline import PipelineReloadRequest, reload_pipeline

    data_dir, checkpoint = workspace
    first, last, served, ending = SELECTIONS[case]
    run_dir = _run_checkpoint(data_dir).parent
    _with_val_mrr(checkpoint, run_dir / "model-a.pt", first, shift=1.0)
    _with_val_mrr(checkpoint, run_dir / "last.pt", last)

    result = asyncio.run(reload_pipeline(PipelineReloadRequest(data_dir=str(data_dir))))

    assert result.success, result.message
    assert result.checkpoint_path == str(run_dir / served)
    assert result.status.served_identity.checkpoint.sha256 == _sha(run_dir / served)
    assert result.selection_reason.endswith(ending), result.selection_reason
    assert ("best" in result.selection_reason) == (served == "model-a.pt")
    assert "replaced" not in result.selection_reason


def test_an_unreplaced_selection_reports_its_metric_and_no_replacement(api, workspace):
    """The file chosen is the file served: no note, and the metric costs no
    third read (the scan's read and the build's are the only two)."""
    from src.api.routes.pipeline import PipelineReloadRequest, reload_pipeline

    data_dir, checkpoint = workspace
    selected = _with_val_mrr(checkpoint, _run_checkpoint(data_dir), 0.25)

    with count_opens() as opens:
        result = asyncio.run(reload_pipeline(PipelineReloadRequest(data_dir=str(data_dir))))

    assert result.success, result.message
    assert result.checkpoint_path == str(selected)
    assert "best val_mrr=0.2500" in result.selection_reason
    assert "replaced" not in result.selection_reason
    assert result.status.served_identity.checkpoint.sha256 == _sha(selected)
    assert opens.opens(selected) == 2
