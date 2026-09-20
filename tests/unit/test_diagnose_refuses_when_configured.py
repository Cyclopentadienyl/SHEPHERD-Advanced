"""A configured pipeline that failed is not a demo.

`_generate_mock_candidates` answers with real MONDO identifiers, real disease
names, real gene symbols and confidence scores from 0.95 down, over HTTP 200.
That is the intended reply when nobody configured a pipeline — someone trying
the service out. It is the wrong reply when a deployment *was* configured and
its pipeline could not be built.

**What "configured" means took two attempts.** The first version keyed on
`SHEPHERD_KG_PATH`, which covers startup and misses the reload API entirely: a
caller can point `/pipeline/reload` at a real workspace, have the candidate
refused, and reach `/diagnose` with no environment variable in sight. These
tests therefore drive the real chain rather than replacing `initialize_pipeline`
with a stub — the stub was what let the first version look covered.

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
def _clean_state(monkeypatch):
    """No pipeline, and no request recorded by an earlier test."""
    from src.api import main as api_main

    monkeypatch.setattr(api_main.app_state, "pipeline", None, raising=False)
    monkeypatch.setattr(
        api_main.app_state, "real_pipeline_requested", False, raising=False
    )
    monkeypatch.delenv("SHEPHERD_KG_PATH", raising=False)


def test_a_startup_configured_deployment_refuses(monkeypatch, tmp_path):
    """The real lazy path: `initialize_pipeline` runs, `build_pipeline` records
    the request, the build fails, and nothing is published."""
    monkeypatch.setenv("SHEPHERD_KG_PATH", str(tmp_path / "absent_kg.json"))

    response = _client().post("/api/v1/diagnose", json=_request())

    assert response.status_code == 503
    assert "could not be initialized" in response.text


def test_a_workspace_named_only_through_reload_refuses_too(monkeypatch, tmp_path):
    """The path the first version missed, with no environment variable set.

    A caller points the reload API at a workspace, the candidate is refused, and
    the next diagnosis must not be answered with invented candidates.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()

    client = _client()
    reload_response = client.post(
        "/api/v1/pipeline/reload", json={"data_dir": str(workspace)}
    )
    assert reload_response.status_code in (200, 400, 409, 503)
    assert '"success":true' not in reload_response.text.replace(" ", "")

    response = client.post("/api/v1/diagnose", json=_request())

    assert response.status_code == 503, (
        "a caller who named a real workspace was treated as a demo user"
    )
    assert "Marfan" not in response.text


def test_no_fabricated_candidate_reaches_a_configured_caller(monkeypatch, tmp_path):
    monkeypatch.setenv("SHEPHERD_KG_PATH", str(tmp_path / "absent_kg.json"))

    body = _client().post("/api/v1/diagnose", json=_request()).text

    assert "Marfan" not in body and "MONDO:0007947" not in body


def test_the_refusal_is_not_relabelled_as_an_internal_error(monkeypatch, tmp_path):
    """`HTTPException` is an `Exception`, so the route's outer handler would turn
    a considered 503 into a 500 — which reads to a caller as a bug in the service
    rather than the service declining to answer."""
    monkeypatch.setenv("SHEPHERD_KG_PATH", str(tmp_path / "absent_kg.json"))

    assert _client().post("/api/v1/diagnose", json=_request()).status_code != 500


def test_an_unconfigured_deployment_still_gets_the_demo():
    """The control, and the half that must not regress. With no workspace named
    by any route, the mock reply is the intended behaviour."""
    response = _client().post("/api/v1/diagnose", json=_request())

    assert response.status_code == 200
    assert "using mock data" in response.text


def test_a_served_pipeline_is_unaffected_by_a_refused_reload(monkeypatch, tmp_path):
    """The other control: recording the request must not disturb a pipeline that
    is already serving. A refused candidate leaves the live one in place."""
    from src.api import main as api_main

    class _Served:
        def run(self, **kwargs):
            raise AssertionError("not called; this test only checks it survives")

    monkeypatch.setattr(api_main.app_state, "pipeline", _Served(), raising=False)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _client().post("/api/v1/pipeline/reload", json={"data_dir": str(workspace)})

    assert api_main.app_state.pipeline is not None, (
        "a refused reload removed the pipeline that was serving"
    )
