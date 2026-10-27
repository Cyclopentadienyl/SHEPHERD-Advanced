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
    DataSource,
    InferenceResult,
    NodeID,
    PathReasoningCandidate,
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
from tests.fixtures.phenotype_kg import DELAY, SEIZURE, build_two_phenotype_kg  # noqa: E402

DISEASE = NodeID(source=DataSource.MONDO, local_id="MONDO:1")


def _settings(eta):
    return ServedSettings(
        device="cpu", device_requested=None, eta=eta, max_path_length=4,
        path_length_penalty=0.9, aggregation_method="weighted_sum",
        max_paths_per_candidate=10, use_direct_path_optimization=True,
        phenotype_normalisation_version=1)


def _identity(scoring_mode):
    """A legal identity for each mode, over a graph supplied in memory."""
    if scoring_mode == "path_reasoning_fallback":
        return ServedIdentity(
            scoring_mode=scoring_mode, checkpoint=ServedCheckpoint(state="absent"),
            graph=ServedGraph(state="in_memory"), shortest_paths=ServedShortestPaths(
                state="absent"), dataset_record=ServedDatasetRecord(state="absent"),
            settings=_settings(None))
    return ServedIdentity(
        scoring_mode="gnn_only", checkpoint=ServedCheckpoint(state="supplied_in_memory"),
        graph=ServedGraph(state="in_memory"), shortest_paths=ServedShortestPaths(state="absent"),
        dataset_record=ServedDatasetRecord(state="absent"), settings=_settings(1.0))


def _no_table(**overrides):
    return SPAnalysisRecord(status="NO_TABLE", phenotypes_submitted=1, phenotypes_mapped=1,
                            **overrides)


def _mixture(rank, score, record=None):
    return ServedMixtureCandidate(rank=rank, disease_id=DISEASE, disease_name="d",
                                  served_score=score, model_term=score,
                                  sp_term=record or _no_table())


def _path(rank, score):
    return PathReasoningCandidate(rank=rank, disease_id=DISEASE, disease_name="d",
                                  path_score=score)


def _result(candidates, mode="gnn_only", **fields):
    return InferenceResult(patient_id="p", timestamp=datetime.now(), candidates=candidates,
                           served_identity=_identity(mode), **fields)


# ---------------------------------------------------------------- the kinds
def test_the_kind_is_read_off_what_produced_the_result():
    assert result_kind_for(None) == "demo/v1"
    assert result_kind_for("gnn_plus_shortest_path") == "served_mixture/v1"
    assert result_kind_for("gnn_only") == "served_mixture/v1"
    assert result_kind_for("path_reasoning_fallback") == "path_reasoning/v1"
    with pytest.raises(ValueError):
        result_kind_for("gnn_raw_cosine")  # batch 5's mode, not served yet

    mixture = _result([_mixture(1, 0.6)], candidates_reached=3)
    assert (mixture.result_kind, mixture.eta, mixture.result_schema_version) == (
        "served_mixture/v1", 1.0, 2)
    assert mixture.path_score_aggregation is None
    path = _result([_path(1, 0.6)], mode="path_reasoning_fallback")
    assert (path.result_kind, path.eta, path.path_score_aggregation) == (
        "path_reasoning/v1", None, "weighted_sum/v1")
    assert path.tie_policy == mixture.tie_policy == "served-stable-discovery-order/v1"


def test_the_labels_name_eta_from_the_result():
    assert result_label("served_mixture/v1", "gnn_plus_shortest_path", 0.55).endswith(
        "η × model term + (1 − η) × SP, η = 0.55")
    assert result_label("served_mixture/v1", "gnn_only", 1.0).endswith(
        "scored by the model term alone")
    with pytest.raises(ValueError):
        result_label("served_mixture/v1", "gnn_plus_shortest_path", None)
    assert result_label("demo/v1") == "Demo answer: no pipeline is configured; not a diagnosis"


@pytest.mark.parametrize("case, build", [
    ("a candidate of another kind",
     lambda: _result([_path(1, 0.5)], candidates_reached=1)),
    ("ranks that are not 1..N",
     lambda: _result([_mixture(1, 0.6), _mixture(3, 0.5)], candidates_reached=2)),
    ("an order that rises",
     lambda: _result([_mixture(1, 0.5), _mixture(2, 0.6)], candidates_reached=2)),
    ("a mixture that does not say how many were reached",
     lambda: _result([_mixture(1, 0.5)])),
    ("fewer reached than it holds",
     lambda: _result([_mixture(1, 0.6), _mixture(2, 0.5)], candidates_reached=1)),
    ("candidates_reached on path reasoning",
     lambda: _result([_path(1, 0.5)], mode="path_reasoning_fallback", candidates_reached=1)),
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
            _path(1, value)


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
    assert DiagnosisPipeline._phenotype_rows(_model_stub(_gnn_ready=False), [node]) is None


def test_a_path_score_that_is_not_finite_refuses():
    from src.inference.pipeline import DiagnosisPipeline
    from src.reasoning.path_reasoning import ReasoningPath

    phenotype = NodeID(source=DataSource.HPO, local_id="HP:1")
    stub = SimpleNamespace(
        _scoring_mode=lambda: ("path_reasoning_fallback", 0.0),
        kg=SimpleNamespace(get_node=lambda _id: None),
        _calculate_reasoning_score=lambda paths: math.nan,
        _extract_supporting_genes=lambda paths: [],
        _determine_evidence_sources=lambda paths: [],
    )
    paths = {str(DISEASE): [ReasoningPath(nodes=[phenotype, DISEASE], edges=[])]}
    with pytest.raises(ResultRefused) as refused:
        DiagnosisPipeline._score_and_rank_candidates(stub, paths, [phenotype], None, False)
    assert refused.value.code == "NON_FINITE_SCORE" and str(DISEASE) in str(refused.value)


# -------------------------------------------- through the pipeline and the API
@pytest.fixture
def gnn_pipeline():
    """A model over the two-phenotype graph, supplied in memory (gnn_only)."""
    from src.inference.pipeline import DiagnosisPipeline
    from tests.unit.test_inference import _build_gnn_test_fixtures

    kg = build_two_phenotype_kg()
    model, graph_data = _build_gnn_test_fixtures(kg)
    pipeline = DiagnosisPipeline(kg=kg, model=model, graph_data=graph_data)
    assert pipeline.served_identity().scoring_mode == "gnn_only"
    return pipeline


def _patient():
    from src.core.types import PatientPhenotypes

    return PatientPhenotypes(patient_id="p", phenotypes=[SEIZURE, DELAY])


def test_a_disease_the_model_does_not_map_refuses_the_whole_result(gnn_pipeline):
    """End to end: discovery reaches a disease the model's mapping lacks. The
    result is refused naming it, never ranked with a 0.0 for it."""
    ranked = gnn_pipeline.run(_patient(), top_k=5, include_explanations=False)
    assert ranked.result_kind == "served_mixture/v1" and ranked.candidates
    missing = str(ranked.candidates[0].disease_id)
    gnn_pipeline._node_id_to_idx["disease"].pop(missing)

    with pytest.raises(ResultRefused) as refused:
        gnn_pipeline.run(_patient(), top_k=5, include_explanations=False)
    assert refused.value.code == "MODEL_TERM_UNAVAILABLE" and missing in str(refused.value)


def test_a_refusal_reaches_the_api_and_the_page_with_its_code(monkeypatch, gnn_pipeline):
    """A deliberate 500 whose detail starts with the code, not a validation
    accident; the page shows it as one sentence. A valid request on the same
    pipeline answers the mixture's union, and carries no removed field."""
    pytest.importorskip("fastapi")
    pytest.importorskip("gradio")
    from fastapi.testclient import TestClient

    from src.api import main as api_main
    from src.webui.components import diagnosis_panel as dp

    monkeypatch.setattr(api_main.app_state, "pipeline", gnn_pipeline, raising=False)
    client = TestClient(api_main.app)
    payload = {"phenotypes": [SEIZURE, DELAY], "include_explanations": True}

    body = client.post("/api/v1/diagnose", json=payload).json()
    assert body["result_kind"] == "served_mixture/v1" and body["candidates"]
    assert body["candidates_reached"] >= len(body["candidates"]) and body["eta"] == 1.0
    for candidate in body["candidates"]:
        assert candidate["sp_term"]["status"] == "NO_TABLE"
        assert candidate["sp_term"]["score"] is None
        assert candidate["served_score"] == candidate["model_term"]
        assert not {"confidence_score", "gnn_score", "sp_score", "reasoning_score",
                    "confidence_label"} & set(candidate)
        assert "confidence_label" not in (candidate["evidence_package"] or {})

    missing = body["candidates"][0]["disease_id"]
    gnn_pipeline._node_id_to_idx["disease"].pop(missing)
    response = client.post("/api/v1/diagnose", json=payload)
    assert response.status_code == 500
    error = response.json()["error"]
    assert error.startswith("MODEL_TERM_UNAVAILABLE: ") and missing in error
    shown = dp._describe_refused_result(response)
    assert shown.startswith("The service refused to produce a result") and missing in shown


@pytest.mark.parametrize("term", [-5.960464477539063e-08, 1.0000001192092896])
def test_a_model_term_an_ulp_outside_its_range_is_answered(monkeypatch, gnn_pipeline, term):
    """The internal review's finding: (cos + 1) / 2 is float32, and a patient
    parallel or anti-parallel to a disease rounds an ulp past [0, 1]. That is a
    valid result; the API mirror answers it rather than failing with a 500."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api import main as api_main
    from src.inference.pipeline import DiagnosisPipeline

    monkeypatch.setattr(DiagnosisPipeline, "_calculate_gnn_score",
                        lambda self, patient_embedding, disease_id: term)
    monkeypatch.setattr(api_main.app_state, "pipeline", gnn_pipeline, raising=False)
    response = TestClient(api_main.app).post(
        "/api/v1/diagnose", json={"phenotypes": [SEIZURE, DELAY]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["candidates"] and {c["model_term"] for c in body["candidates"]} == {term}
    assert {c["served_score"] for c in body["candidates"]} == {term}     # gnn_only


def test_the_demo_answer_carries_no_score(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api import main as api_main

    monkeypatch.setattr(api_main.app_state, "pipeline", None, raising=False)
    monkeypatch.setattr(api_main.app_state, "real_pipeline_requested", True, raising=False)
    monkeypatch.setattr(type(api_main.app_state), "requested_pipeline_missing",
                        property(lambda self: False))
    body = TestClient(api_main.app).post(
        "/api/v1/diagnose", json={"phenotypes": [SEIZURE], "top_k": 3,
                                  "include_explanations": True}).json()

    assert body["result_kind"] == "demo/v1" and body["served_identity"] is None
    assert [c["rank"] for c in body["candidates"]] == [1, 2, 3]
    for candidate in body["candidates"]:
        assert set(candidate) == {"rank", "disease_id", "disease_name", "matching_phenotypes"}
