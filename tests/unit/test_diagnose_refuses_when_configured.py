"""A configured pipeline that failed is not a demo.

`_generate_mock_candidates` answers with real MONDO identifiers, real disease
names, real gene symbols and confidence scores from 0.95 down, over HTTP 200.
That is the intended reply when nobody configured a pipeline — someone trying
the service out. It is the wrong reply when a deployment *was* configured and
its pipeline could not be built: the caller asked a clinical question of a
system that cannot answer it, and a warning string beside a well-formed ranked
list is not a refusal.

This became reachable for artifacts that previously loaded when the shortest-path
loader gained its hop-bound refusals, which is why it is closed here rather than
left as pre-existing behaviour.

Module: tests/unit/test_diagnose_refuses_when_configured.py
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")


def _client():
    from fastapi.testclient import TestClient

    from src.api.main import app

    return TestClient(app)


def _request():
    return {"phenotypes": ["HP:0001250", "HP:0001263"], "top_k": 3}


@pytest.fixture(autouse=True)
def _pipeline_unavailable(monkeypatch):
    """No pipeline, and lazy initialisation cannot make one."""
    from src.api import main as api_main

    monkeypatch.setattr(api_main.app_state, "pipeline", None, raising=False)
    monkeypatch.setattr(
        api_main, "initialize_pipeline",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("hop bound unknown")),
    )


def test_a_configured_deployment_refuses_rather_than_inventing(monkeypatch):
    """503, not a ranked list of diseases nobody scored."""
    monkeypatch.setenv("SHEPHERD_KG_PATH", "/some/configured/kg.json")

    response = _client().post("/api/v1/diagnose", json=_request())

    assert response.status_code == 503, (
        f"a configured deployment answered a clinical request with "
        f"{response.status_code}"
    )
    assert "could not be initialized" in response.text


def test_the_refusal_is_not_relabelled_as_an_internal_error(monkeypatch):
    """`HTTPException` is an `Exception`, so the route's outer handler would turn
    a considered 503 into a 500 — which reads to a caller as a bug in the service
    rather than the service declining to answer."""
    monkeypatch.setenv("SHEPHERD_KG_PATH", "/some/configured/kg.json")

    response = _client().post("/api/v1/diagnose", json=_request())

    assert response.status_code != 500


def test_no_fabricated_candidate_reaches_a_configured_caller(monkeypatch):
    """The specific thing being prevented, named rather than implied."""
    monkeypatch.setenv("SHEPHERD_KG_PATH", "/some/configured/kg.json")

    body = _client().post("/api/v1/diagnose", json=_request()).text

    assert "Marfan" not in body and "MONDO:0007947" not in body


def test_an_unconfigured_deployment_still_gets_the_demo(monkeypatch):
    """The control, and the half that must not regress. Without a configured
    pipeline the mock reply is the intended behaviour, and removing it was never
    the point."""
    monkeypatch.delenv("SHEPHERD_KG_PATH", raising=False)

    response = _client().post("/api/v1/diagnose", json=_request())

    assert response.status_code == 200
    assert "using mock data" in response.text
