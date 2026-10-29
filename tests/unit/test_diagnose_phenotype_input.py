"""
The phenotype list through the real `/diagnose` route.
======================================================
docs/working/PLAN_PHENOTYPE_NORMALISATION.md, N1. Each test posts to the real
application, so what it pins is what a caller — the WebUI included — receives:

  - the 422 a list over the limit produces, in the shape the WebUI reads. A
    pydantic or FastAPI upgrade that changes the shape fails here, instead of
    quietly turning the WebUI's message into its general failure line;
  - a summary that counts the phenotypes scored, not the request's entries;
  - one confidence per phenotype, refused otherwise, and never a 500.

The pipeline is a disease child trained for real (`tests/fixtures/trained_children`):
since K7 nothing is served without a model, and with no pipeline the route
refuses before any phenotype is counted.

Module: tests/unit/test_diagnose_phenotype_input.py
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from tests.fixtures.phenotype_kg import (  # noqa: E402
    DELAY,
    SEIZURE,
    UNKNOWN,
    build_two_phenotype_kg,
)


@pytest.fixture(scope="module")
def pipeline(trained_disease_child):
    from tests.fixtures.trained_children import serve

    return serve(trained_disease_child)


@pytest.fixture
def client(monkeypatch, pipeline):
    """The real app, serving a real disease child, with every call to
    `pipeline.run` recorded."""
    from fastapi.testclient import TestClient

    from src.api import main as api_main

    calls = []
    original = pipeline.run

    def recording_run(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(pipeline, "run", recording_run)
    monkeypatch.setattr(api_main.app_state, "pipeline", pipeline, raising=False)
    test_client = TestClient(api_main.app)
    test_client.pipeline_calls = calls
    return test_client


def _post(client, **payload):
    return client.post("/api/v1/diagnose", json=payload)


# ------------------------------------------------------------------ the limit
def test_a_list_over_the_limit_gets_the_422_the_webui_reads(client):
    """The contract `diagnosis_panel` depends on: location, type, and both
    numbers in `ctx`. The pipeline is never reached."""
    from src.api.routes.diagnose import DiagnoseRequest

    limit = next(
        item.max_length
        for item in DiagnoseRequest.model_fields["phenotypes"].metadata
        if hasattr(item, "max_length")
    )
    response = _post(client, phenotypes=[SEIZURE] * limit + [DELAY])

    assert response.status_code == 422
    entries = response.json()["detail"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["loc"] == ["body", "phenotypes"]
    assert entry["type"] == "too_long"
    assert entry["ctx"]["max_length"] == limit
    assert entry["ctx"]["actual_length"] == limit + 1
    assert client.pipeline_calls == []


# ------------------------------------------------------------------ the summary
def test_the_summary_counts_the_phenotypes_scored(client):
    response = _post(client, phenotypes=[SEIZURE, SEIZURE, DELAY])

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["summary"].endswith("for 2 phenotypes (3 received)")
    assert "Removed 1 repeated phenotype entry; each phenotype is scored once." in (
        body["warnings"]
    )


def test_an_unknown_id_is_not_counted_as_scored(client):
    response = _post(client, phenotypes=[SEIZURE, UNKNOWN, DELAY])

    body = response.json()
    assert body["summary"].endswith("for 2 phenotypes (3 received)")
    assert f"Unknown phenotype: {UNKNOWN}" in body["warnings"]


def test_a_list_scored_as_sent_shows_one_count(client):
    body = _post(client, phenotypes=[SEIZURE, DELAY]).json()

    assert body["summary"].endswith("for 2 phenotypes")


def test_with_no_pipeline_nothing_is_counted(monkeypatch):
    """No pipeline configured at all: the request is refused (K7, DEMO1), so
    there is no summary and no account of phenotypes used."""
    from fastapi.testclient import TestClient

    from src.api import main as api_main

    monkeypatch.setattr(api_main.app_state, "pipeline", None, raising=False)
    monkeypatch.setattr(api_main.app_state, "real_pipeline_requested", False, raising=False)
    monkeypatch.delenv("SHEPHERD_KG_PATH", raising=False)

    response = TestClient(api_main.app).post(
        "/api/v1/diagnose", json={"phenotypes": [SEIZURE, SEIZURE, DELAY]})

    assert response.status_code == 503
    assert not {"summary", "phenotype_input", "candidates"} & set(response.json())


def test_a_graph_alone_is_refused_and_never_ranked(monkeypatch, tmp_path):
    """K7 (PATH1): a deployment that names a graph and no model is refused at
    its build, and every diagnosis then answers 503; no path-reasoning result
    is produced from the graph."""
    from fastapi.testclient import TestClient

    from src.api import main as api_main
    from src.inference.pipeline import PipelineBuildError

    kg_file = tmp_path / "kg.json"
    build_two_phenotype_kg().save_json(str(kg_file))
    monkeypatch.setattr(api_main.app_state, "pipeline", None, raising=False)
    monkeypatch.setattr(api_main.app_state, "real_pipeline_requested", False, raising=False)
    monkeypatch.setenv("SHEPHERD_KG_PATH", str(kg_file))
    monkeypatch.setenv("SHEPHERD_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("SHEPHERD_CHECKPOINT_PATH", raising=False)

    with pytest.raises(PipelineBuildError, match="No model is configured"):
        api_main.build_pipeline()
    response = TestClient(api_main.app).post("/api/v1/diagnose",
                                             json={"phenotypes": [SEIZURE, DELAY]})

    assert response.status_code == 503
    assert "candidates" not in response.json()
    assert api_main.app_state.pipeline is None


def test_the_route_summary_labels_an_unmapped_request_as_received():
    """A request refused before its phenotypes were mapped has no account, so
    the summary counts what was received."""
    from src.api.routes.diagnose import _summary

    assert _summary(0, 3, None).endswith("for 3 phenotypes received")


# ------------------------------------------------------------------ confidences
@pytest.mark.parametrize("confidences", [None, [0.9, 0.8]])
def test_no_confidences_or_one_per_phenotype_pass(client, confidences):
    response = _post(client, phenotypes=[SEIZURE, DELAY], phenotype_confidences=confidences)

    assert response.status_code == 200, response.text


@pytest.mark.parametrize("confidences", [[], [0.9], [0.9, 0.8, 0.7]])
def test_a_confidence_list_of_another_length_is_refused(client, confidences):
    response = _post(client, phenotypes=[SEIZURE, DELAY], phenotype_confidences=confidences)

    assert response.status_code == 422
    (entry,) = response.json()["detail"]
    assert (
        f"phenotype_confidences has {len(confidences)} entries for 2 phenotypes"
        in entry["msg"]
    )
    assert client.pipeline_calls == []


def test_confidences_survive_unknown_ids_and_repeats_at_their_positions(client, pipeline,
                                                                        monkeypatch):
    """`[X, A, A, B]` keeps positions 1 and 3, so scoring sees the confidences
    sent with A and B, and the route does not fail on the reduction."""
    seen = {}
    original = pipeline._rank_by_disease_task

    def capture(patient_input, *args, **kwargs):
        seen["input"] = patient_input
        return original(patient_input, *args, **kwargs)

    monkeypatch.setattr(pipeline, "_rank_by_disease_task", capture)
    response = _post(
        client,
        phenotypes=[UNKNOWN, SEIZURE, SEIZURE, DELAY],
        phenotype_confidences=[0.1, 0.2, 0.3, 0.4],
    )

    assert response.status_code == 200, response.text
    assert seen["input"].phenotypes == [SEIZURE, DELAY]
    assert seen["input"].phenotype_confidences == [0.2, 0.4]
