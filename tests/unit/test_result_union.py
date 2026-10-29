"""
The result union (Spec 1 §2.1, Batch 4): its invariants and the new refusals.
=============================================================================
`docs/working/results-review/SPEC_1_RESULTS_REVIEW.md` §2.1 replaces the flat
candidate with one kind per result. These tests pin what is new in Batch 4:

- **the kind is read off what produced the result**, never passed in, and
  `InferenceResult` refuses a result that breaks the kind's rules (another
  kind's candidate, ranks not 1..N, an order that rises, a mixture with no
  `candidates_reached`, `mixed_as` where nothing was mixed);
- **the SP record cannot hold an absent number as a zero** (§3.3);
- **the model term fails closed**: each state that used to return 0.0, and the
  two index clamps, refuse the result (`MODEL_TERM_UNAVAILABLE`, naming the
  node); a non-finite score refuses it too (`NON_FINITE_SCORE`). Each state is
  forced here, since a verified workspace binds the files that would make it
  occur. The API answers a refusal as a deliberate 500 whose detail starts with
  the code, and the page reads it as a sentence.

Valid input is unchanged: the golden test (`test_served_rank.py`) compares
every kind's fields with the capture taken before the union, exactly.

Module: tests/unit/test_result_union.py
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.core.types import (  # noqa: E402
    MAPPED_ONLY_DENOMINATOR,
    RESULT_KINDS,
    DataSource,
    DiseaseTaskCandidate,
    InferenceResult,
    NodeID,
    ResultRefused,
    SPAnalysisRecord,
    ServedCheckpoint,
    ServedDatasetRecord,
    ServedGraph,
    ServedIdentity,
    ServedMixtureCandidate,
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
    """A legal identity for each mode: a model read from a checkpoint over a
    verified workspace, the only identity a pipeline has (K7)."""
    return ServedIdentity(
        scoring_mode=scoring_mode, checkpoint=ServedCheckpoint(state="verified",
                                                               sha256="a" * 64),
        graph=ServedGraph(state="verified", kg_sha256="b" * 64,
                          split_manifest_sha256="c" * 64),
        shortest_paths=ServedShortestPaths(state="absent"),
        dataset_record=ServedDatasetRecord(state="verified", origin="synthetic",
                                           sha256="d" * 64),
        settings=_settings(1.0 if scoring_mode == "gnn_only" else None))


def _no_table(**overrides):
    return SPAnalysisRecord(status="NO_TABLE", phenotypes_submitted=1, phenotypes_mapped=1,
                            **overrides)


def _mixture(rank, score, record=None):
    return ServedMixtureCandidate(rank=rank, disease_id=DISEASE, disease_name="d",
                                  served_score=score, model_term=score,
                                  sp_term=record or _no_table())


def _disease(rank, score):
    return DiseaseTaskCandidate(rank=rank, disease_id=DISEASE, disease_name="d", score=score)


def _result(candidates, mode="gnn_only", **fields):
    return InferenceResult(patient_id="p", timestamp=datetime.now(), candidates=candidates,
                           served_identity=_identity(mode), **fields)


# ---------------------------------------------------------------- the kinds
def test_the_kind_is_read_off_what_produced_the_result():
    assert result_kind_for("gnn_plus_shortest_path") == "served_mixture/v1"
    assert result_kind_for("gnn_only") == "served_mixture/v1"
    assert result_kind_for("disease_task") == "disease_task/v1"
    with pytest.raises(ValueError):
        result_kind_for("gnn_raw_cosine")  # batch 5's mode, not served yet

    mixture = _result([_mixture(1, 0.6)], candidates_reached=3)
    assert (mixture.result_kind, mixture.eta, mixture.result_schema_version) == (
        "served_mixture/v1", 1.0, 2)
    assert mixture.tie_policy == "served-stable-discovery-order/v1"
    disease = _result([_disease(1, -0.6)], mode="disease_task")
    assert (disease.result_kind, disease.eta) == ("disease_task/v1", None)
    assert disease.tie_policy == "score-desc-then-view-id-asc/v1"


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


def test_the_labels_name_eta_from_the_result():
    assert result_label("served_mixture/v1", "gnn_plus_shortest_path", 0.55).endswith(
        "η × model term + (1 − η) × SP, η = 0.55")
    assert result_label("served_mixture/v1", "gnn_only", 1.0).endswith(
        "scored by the model term alone")
    with pytest.raises(ValueError):
        result_label("served_mixture/v1", "gnn_plus_shortest_path", None)


@pytest.mark.parametrize("case, build", [
    ("a candidate of another kind",
     lambda: _result([_disease(1, -0.5)], candidates_reached=1)),
    ("ranks that are not 1..N",
     lambda: _result([_mixture(1, 0.6), _mixture(3, 0.5)], candidates_reached=2)),
    ("an order that rises",
     lambda: _result([_mixture(1, 0.5), _mixture(2, 0.6)], candidates_reached=2)),
    ("a mixture that does not say how many were reached",
     lambda: _result([_mixture(1, 0.5)])),
    ("fewer reached than it holds",
     lambda: _result([_mixture(1, 0.6), _mixture(2, 0.5)], candidates_reached=1)),
    ("candidates_reached on a disease task result",
     lambda: _result([_disease(1, -0.5)], mode="disease_task", candidates_reached=1)),
    ("mixed_as where nothing was mixed",
     lambda: _result([_mixture(1, 0.5, _no_table(mixed_as=0.0))], candidates_reached=1)),
    ("a numeric record with no table served",
     lambda: _result([_mixture(1, 0.5, SPAnalysisRecord(
         status="COMPUTED", phenotypes_submitted=1, phenotypes_mapped=1, max_hops=5,
         mean_hop_distance=1.0, score=0.5, unreachable_pair_count=0,
         denominator_policy=MAPPED_ONLY_DENOMINATOR))], candidates_reached=1)),
])
def test_a_result_that_breaks_its_kinds_rules_is_refused(case, build):
    with pytest.raises(ValueError):
        build()


def test_a_score_that_is_not_finite_cannot_be_a_candidate():
    for value in (math.nan, math.inf):
        with pytest.raises(ValueError, match="finite"):
            _mixture(1, value)
        with pytest.raises(ValueError, match="finite"):
            _disease(1, value)


@pytest.mark.parametrize("fields", [
    dict(status="NO_TABLE", score=0.2),                      # a number on no table
    dict(status="TARGET_UNMAPPED", max_hops=5, mean_hop_distance=3.0),
    dict(status="COMPUTED", max_hops=5),                     # a numeric status, no numbers
    dict(status="TARGET_UNMAPPED"),                          # a table's status, no bound
    dict(status="NO_TABLE", max_hops=5),                     # no table, but a bound
    dict(status="NO_TABLE", mixed_as=0.5),                   # mixed as anything but 0.0
    dict(status="SOMETHING"),
])
def test_an_sp_record_cannot_hold_an_absent_number(fields):
    with pytest.raises(ValueError):
        SPAnalysisRecord(phenotypes_submitted=1, phenotypes_mapped=1, **fields)


def test_a_partial_record_names_what_it_did_not_map():
    numbers = dict(max_hops=5, mean_hop_distance=2.0, score=1 / 3, unreachable_pair_count=0,
                   denominator_policy=MAPPED_ONLY_DENOMINATOR)
    with pytest.raises(ValueError, match="COMPUTED_PARTIAL"):
        SPAnalysisRecord(status="COMPUTED_PARTIAL", phenotypes_submitted=1,
                         phenotypes_mapped=1, **numbers)
    with pytest.raises(ValueError, match="COMPUTED_PARTIAL"):
        SPAnalysisRecord(status="COMPUTED", phenotypes_submitted=2, phenotypes_mapped=1,
                         unmapped_ids=("HP:1",), **numbers)
    with pytest.raises(ValueError, match="submitted"):
        SPAnalysisRecord(status="COMPUTED_PARTIAL", phenotypes_submitted=3,
                         phenotypes_mapped=1, unmapped_ids=("HP:1",), **numbers)
    assert SPAnalysisRecord(status="COMPUTED_PARTIAL", phenotypes_submitted=2,
                            phenotypes_mapped=1, unmapped_ids=("HP:1",), **numbers).numeric


# ------------------------------------------------- the model term fails closed
def _model_stub(**overrides):
    stub = SimpleNamespace(
        _gnn_ready=True,
        _node_embeddings={"phenotype": torch.randn(3, 4), "disease": torch.randn(2, 4)},
        _node_id_to_idx={"phenotype": {"hpo:HP:1": 0}, "disease": {str(DISEASE): 1}},
        # The ShepherdGNN path: no model bundle served.
        _bundle_view=None,
    )
    for key, value in overrides.items():
        setattr(stub, key, value)
    return stub


@pytest.mark.parametrize("case, overrides, names", [
    ("no embeddings", dict(_node_embeddings=None), "no node embeddings"),
    ("not ready", dict(_gnn_ready=False), "no node embeddings"),
    ("no disease table", dict(_node_embeddings={"phenotype": torch.randn(3, 4)}),
     "no disease embedding table"),
    ("a disease with no row", dict(_node_id_to_idx={"phenotype": {}, "disease": {}}),
     str(DISEASE)),
    ("a row past the table (the clamp)",
     dict(_node_id_to_idx={"phenotype": {}, "disease": {str(DISEASE): 2}}), "outside"),
    ("a negative row", dict(_node_id_to_idx={"phenotype": {}, "disease": {str(DISEASE): -1}}),
     "outside"),
])
def test_each_state_that_returned_zero_now_refuses(case, overrides, names):
    from src.inference.pipeline import DiagnosisPipeline

    with pytest.raises(ResultRefused) as refused:
        DiagnosisPipeline._calculate_gnn_score(_model_stub(**overrides), torch.randn(4), DISEASE)
    assert refused.value.code == "MODEL_TERM_UNAVAILABLE", case
    assert names in str(refused.value), case


def test_a_model_term_that_is_not_finite_refuses(monkeypatch):
    import src.inference.scoring as scoring
    from src.inference.pipeline import DiagnosisPipeline

    monkeypatch.setattr(scoring, "cosine_scores", lambda p, c: torch.tensor([math.nan]))
    with pytest.raises(ResultRefused) as refused:
        DiagnosisPipeline._calculate_gnn_score(_model_stub(), torch.randn(4), DISEASE)
    assert refused.value.code == "NON_FINITE_SCORE"


@pytest.mark.parametrize("case, overrides, rows", [
    ("no embeddings", dict(_node_embeddings=None), [0]),
    ("no phenotype table", dict(_node_embeddings={"disease": torch.randn(2, 4)}), [0]),
    ("no rows", {}, []),
])
def test_the_patient_cannot_be_pooled_from_nothing(case, overrides, rows):
    from src.inference.pipeline import DiagnosisPipeline

    with pytest.raises(ResultRefused, match="MODEL_TERM_UNAVAILABLE"):
        DiagnosisPipeline._patient_embedding(_model_stub(**overrides), rows)


@pytest.mark.parametrize("case, mapping, names", [
    ("a phenotype with no row", {}, "has no row"),
    ("a row past the table (the clamp)", {"hpo:HP:1": 3}, "outside"),
    ("a row that is not an integer", {"hpo:HP:1": True}, "outside"),
])
def test_a_phenotype_with_no_usable_row_refuses_at_the_mapping_step(case, mapping, names):
    """Spec 1 §2.2: a known node with no tensor row is a model–graph mismatch,
    refused, never skipped (the old lookup dropped it and pooled the rest)."""
    from src.inference.pipeline import DiagnosisPipeline

    stub = _model_stub(_node_id_to_idx={"phenotype": mapping, "disease": {}})
    node = NodeID(source=DataSource.HPO, local_id="HP:1")
    with pytest.raises(ResultRefused) as refused:
        DiagnosisPipeline._phenotype_rows(stub, [node])
    assert refused.value.code == "MODEL_TERM_UNAVAILABLE" and names in str(refused.value)


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


@pytest.mark.parametrize("term", [-5.960464477539063e-08, 1.0000001192092896])
def test_a_model_term_an_ulp_outside_its_range_is_answered(term):
    """The internal review's finding: (cos + 1) / 2 is float32, and a patient
    parallel or anti-parallel to a disease rounds an ulp past [0, 1]. That is a
    valid result; the API mirror answers it rather than failing with a 500."""
    from dataclasses import asdict

    from src.api.routes.diagnose import DiagnoseResponse, _candidate_body

    candidate = _mixture(1, term)
    body = {"result_kind": "served_mixture/v1", "result_schema_version": 2,
            "session_id": "s", "patient_id": "p", "timestamp": "t", "inference_time_ms": 1.0,
            "served_identity": _identity("gnn_only").to_dict(), "phenotype_input": None,
            "candidates": [_candidate_body(candidate, True, True)], "eta": 1.0,
            "candidates_reached": 1, "tie_policy": "served-stable-discovery-order/v1"}
    answered = DiagnoseResponse.model_validate(body).root
    assert answered.candidates[0].model_term == term == answered.candidates[0].served_score
    assert asdict(candidate.sp_term)["status"] == "NO_TABLE"


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
