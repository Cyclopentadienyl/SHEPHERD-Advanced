"""
The result union (Spec 1 §2.1, Batch 4): its invariants and the new refusals.
=============================================================================
`docs/working/results-review/SPEC_1_RESULTS_REVIEW.md` §2.1 replaces the flat
candidate with one kind per result. These tests pin what is new in Batch 4:

- **the kind is read off what produced the result**, never passed in, and
  `InferenceResult` refuses a result that breaks the kind's rules (another
  kind's candidate, ranks not 1..N, an order that rises); the served
  mixture's kind and modes are gone since K8-4;
- **a phenotype the model cannot place refuses the result**
  (`MODEL_TERM_UNAVAILABLE`, naming the node), never ranked without it; the
  case is forced here, since a verified workspace binds the files that would
  make it occur. The API answers a refusal as a deliberate 500 whose detail
  starts with the code, and the page reads it as a sentence.

Each kind's fields on valid input are pinned by its own serving tests
(`test_disease_serving.py`, `test_task_serving.py`).

Module: tests/unit/test_result_union.py
"""
from __future__ import annotations

import math
from datetime import datetime

import pytest

torch = pytest.importorskip("torch")

from src.core.types import (  # noqa: E402
    RESULT_KINDS,
    DataSource,
    DiseaseTaskCandidate,
    GeneTaskCandidate,
    InferenceResult,
    NodeID,
    ResultRefused,
    ServedCheckpoint,
    ServedDatasetRecord,
    ServedGraph,
    ServedIdentity,
    ServedSettings,
    ServedShortestPaths,
    result_kind_for,
    result_label,
)
DISEASE = NodeID(source=DataSource.MONDO, local_id="MONDO:1")
#: Two phenotypes of the demo graph, in the trained disease child's view.
SEIZURE, DELAY = "HP:0001250", "HP:0001263"


def _settings(eta):
    return ServedSettings(
        device="cpu", device_requested="cpu", eta=eta, max_path_length=4,
        path_length_penalty=0.9, max_paths_per_candidate=10,
        use_direct_path_optimization=True, phenotype_normalisation_version=1)


def _identity(scoring_mode):
    """A legal identity of a task model that uses no shortest-path term (the
    disease or patients-like-me task): a model read from a checkpoint over a
    verified workspace, the only identity a pipeline has (K7)."""
    return ServedIdentity(
        scoring_mode=scoring_mode, checkpoint=ServedCheckpoint(state="verified",
                                                               sha256="a" * 64),
        graph=ServedGraph(state="verified", kg_sha256="b" * 64,
                          split_manifest_sha256="c" * 64),
        shortest_paths=ServedShortestPaths(state="absent"),
        dataset_record=ServedDatasetRecord(state="verified", origin="synthetic",
                                           sha256="d" * 64),
        settings=_settings(None))


def _disease(rank, score):
    return DiseaseTaskCandidate(rank=rank, disease_id=DISEASE, disease_name="d", score=score)


def _result(candidates, mode="disease_task", **fields):
    return InferenceResult(patient_id="p", timestamp=datetime.now(), candidates=candidates,
                           served_identity=_identity(mode), **fields)


# ---------------------------------------------------------------- the kinds
def test_the_kind_is_read_off_what_produced_the_result():
    assert result_kind_for("disease_task") == "disease_task/v1"
    with pytest.raises(ValueError):
        result_kind_for("gnn_raw_cosine")  # batch 5's mode, not served yet
    # K8-4 (S1): the served mixture's modes and kind are gone, with no alias.
    for retired_mode in ("gnn_plus_shortest_path", "gnn_only"):
        with pytest.raises(ValueError):
            result_kind_for(retired_mode)
    assert "served_mixture/v1" not in RESULT_KINDS
    with pytest.raises(ValueError):
        result_label("served_mixture/v1")

    disease = _result([_disease(1, -0.6)])
    assert (disease.result_kind, disease.result_schema_version) == ("disease_task/v1", 2)
    assert disease.tie_policy == "score-desc-then-view-id-asc/v1"
    assert not hasattr(disease, "eta") and not hasattr(disease, "candidates_reached")


@pytest.mark.parametrize("retired_mode", ["path_reasoning_fallback", None])
def test_no_kind_is_answered_without_a_model(retired_mode):
    """K7 (PATH1, DEMO1): every result comes from a model. The path-reasoning
    kind and the demo answer are gone, with no alias: their modes map to no
    kind, their labels are unknown, and no result can be built without a
    checkpoint read."""
    assert "path_reasoning/v1" not in RESULT_KINDS and "demo/v1" not in RESULT_KINDS
    with pytest.raises(ValueError):
        result_kind_for(retired_mode)
    for retired_kind in ("path_reasoning/v1", "demo/v1"):
        with pytest.raises(ValueError):
            result_label(retired_kind)
    for state in ("absent", "supplied_in_memory"):
        with pytest.raises(ValueError):
            ServedCheckpoint(state=state, sha256="a" * 64)


@pytest.mark.parametrize("case, build", [
    ("a candidate of another kind",
     lambda: _result([_disease(1, -0.5), GeneTaskCandidate(
         rank=2, gene_id=DISEASE, gene_name="g", score=-0.7, embsim=-0.7, sp_similarity=0.0,
         mean_distance=1.0)])),
    ("ranks that are not 1..N",
     lambda: _result([_disease(1, -0.5), _disease(3, -0.6)])),
    ("an order that rises",
     lambda: _result([_disease(1, -0.6), _disease(2, -0.5)])),
])
def test_a_result_that_breaks_its_kinds_rules_is_refused(case, build):
    with pytest.raises(ValueError):
        build()


def test_a_score_that_is_not_finite_cannot_be_a_candidate():
    for value in (math.nan, math.inf):
        with pytest.raises(ValueError, match="finite"):
            _disease(1, value)


# -------------------------------------------- through the pipeline and the API
@pytest.fixture
def own_pipeline(trained_disease_child):
    """A disease child's pipeline of this test's own (the shared one is not
    changed), with its view's phenotypes read once, as a first request does."""
    from tests.fixtures.trained_children import serve

    pipeline = serve(trained_disease_child)
    pipeline.run(_patient(), top_k=1, include_explanations=False)
    return pipeline


def _patient():
    from src.core.types import PatientPhenotypes

    return PatientPhenotypes(patient_id="p", phenotypes=[SEIZURE, DELAY])


def _seizure_unmapped(pipeline):
    """The model's node mapping loses the seizure's row: a known phenotype
    the model cannot place (a model–graph mismatch)."""
    seizure = str(NodeID(source=DataSource.HPO, local_id=SEIZURE))
    pipeline._node_id_to_idx["phenotype"].pop(seizure)
    return seizure


def test_a_phenotype_the_model_does_not_map_refuses_the_whole_result(own_pipeline):
    """End to end: a phenotype in the model's view has no row in the node
    mapping. The result is refused naming it, never ranked without it."""
    ranked = own_pipeline.run(_patient(), top_k=5, include_explanations=False)
    assert ranked.result_kind == "disease_task/v1" and ranked.candidates
    missing = _seizure_unmapped(own_pipeline)

    with pytest.raises(ResultRefused) as refused:
        own_pipeline.run(_patient(), top_k=5, include_explanations=False)
    assert refused.value.code == "MODEL_TERM_UNAVAILABLE" and missing in str(refused.value)


def test_a_refusal_reaches_the_api_and_the_page_with_its_code(monkeypatch, own_pipeline):
    """A deliberate 500 whose detail starts with the code, not a validation
    accident; the page shows it as one sentence. A valid request on the same
    pipeline answers the disease task's union, and carries no removed field."""
    pytest.importorskip("fastapi")
    pytest.importorskip("gradio")
    from fastapi.testclient import TestClient

    from src.api import main as api_main
    from src.webui.components import diagnosis_panel as dp

    monkeypatch.setattr(api_main.app_state, "pipeline", own_pipeline, raising=False)
    client = TestClient(api_main.app)
    payload = {"phenotypes": [SEIZURE, DELAY], "include_explanations": True}

    body = client.post("/api/v1/diagnose", json=payload).json()
    assert body["result_kind"] == "disease_task/v1" and body["candidates"]
    assert not {"eta", "candidates_reached", "path_score_aggregation"} & set(body)
    for candidate in body["candidates"]:
        assert candidate["score"] <= 0
        assert not {"confidence_score", "gnn_score", "sp_score", "reasoning_score",
                    "confidence_label", "path_score", "served_score"} & set(candidate)
        assert "confidence_label" not in (candidate["evidence_package"] or {})

    missing = _seizure_unmapped(own_pipeline)
    response = client.post("/api/v1/diagnose", json=payload)
    assert response.status_code == 500
    error = response.json()["error"]
    assert error.startswith("MODEL_TERM_UNAVAILABLE: ") and missing in error
    shown = dp._describe_refused_result(response)
    assert shown.startswith("The service refused to produce a result") and missing in shown


def test_with_no_pipeline_nothing_is_answered(monkeypatch):
    """K7 (DEMO1): with no pipeline served, a diagnosis is refused with its
    reason (503), never answered with illustrative candidates."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api import main as api_main

    monkeypatch.setattr(api_main.app_state, "pipeline", None, raising=False)
    monkeypatch.setattr(api_main.app_state, "real_pipeline_requested", True, raising=False)
    response = TestClient(api_main.app).post(
        "/api/v1/diagnose", json={"phenotypes": [SEIZURE], "top_k": 3,
                                  "include_explanations": True})

    assert response.status_code == 503
    body = response.json()
    assert "candidates" not in body and "result_kind" not in body
    assert "none is being served" in body["error"]
