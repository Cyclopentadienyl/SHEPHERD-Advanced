"""No pipeline served, no diagnosis: every answer comes from a model (K7, DEMO1).

`/api/v1/diagnose` used to answer with fixed illustrative candidates (the demo
answer, `demo/v1`) whenever nobody had configured a pipeline, and to refuse
only a deployment that had configured one and failed. Since K7 (the owner's
decision 1) there is one state, **no pipeline served**, and one definition of
it (`AppState.no_pipeline_reason`): `/diagnose` and `/ready` both answer 503
with the reason, and `/health` stays 200, so the launcher (which checks
`/health`) starts the service and the dashboard's Load entry can load a model.
The reason still tells "nothing is configured" from "a configured pipeline
could not be initialized".

**What "configured" means took two attempts.** The first version keyed on
`SHEPHERD_KG_PATH`, which covers startup and misses the reload API entirely: a
caller can point `/pipeline/reload` at a real workspace, have the candidate
refused, and reach `/diagnose` with no environment variable in sight. These
tests therefore drive the real chain rather than replacing `initialize_pipeline`
with a stub — the stub was what let the first version look covered.

`TestTheNoModelMatrix` is the reviewer's matrix (K7 audit §2A), over the real
lifespan, the real routes and a disease child trained for real
(`tests/fixtures/trained_children`).

Module: tests/unit/test_diagnose_refuses_when_configured.py
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

#: The two reasons `AppState.no_pipeline_reason` gives, by their first words.
NOTHING_LOADED = "No model is loaded"
CONFIGURED_BUT_FAILED = "A pipeline is configured"


def _client(app=None, *, base_url=None):
    """A client D8 allows (mode M1, set by `_clean_state`, and a loopback
    request): the reload route is an operator control since M5-1."""
    from fastapi.testclient import TestClient

    from src.api.main import app as api_app
    from tests.fixtures.operator_access import OPERATOR_BASE_URL

    return TestClient(app or api_app, base_url=base_url or OPERATOR_BASE_URL)


def _request():
    return {"phenotypes": ["HP:0001250", "HP:0001263"], "top_k": 3}


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """No pipeline, no deployment configured, and no request recorded by an
    earlier test; the served fields a publish writes are put back afterwards."""
    from src.api import main as api_main

    for field in ("pipeline", "kg", "_current_data_dir", "_current_checkpoint_path"):
        monkeypatch.setattr(api_main.app_state, field, None, raising=False)
    monkeypatch.setattr(
        api_main.app_state, "real_pipeline_requested", False, raising=False
    )
    for name in ("SHEPHERD_KG_PATH", "SHEPHERD_DATA_DIR", "SHEPHERD_CHECKPOINT_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")
    from tests.fixtures.operator_access import operator_mode

    operator_mode(monkeypatch)


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
    the next diagnosis is refused as a configured pipeline that could not be
    built.
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

    assert response.status_code == 503
    assert response.json()["error"].startswith(CONFIGURED_BUT_FAILED), (
        "a caller who named a real workspace was told nothing was configured"
    )
    assert "candidates" not in response.json()


def test_a_reload_d8_refuses_requests_nothing(monkeypatch, tmp_path):
    """A reload refused by D8 (here: arriving on a LAN address) is refused
    before the route runs, so it is not a request for a real pipeline: the
    refusal still says nothing is loaded, and a read-only caller cannot load
    one. Before D8, any caller's reload, refused or not, switched every
    frontend's diagnosis to "configured but failed" (the multi-frontend
    audit)."""
    from src.api import main as api_main

    workspace = tmp_path / "ws"
    workspace.mkdir()
    refused = _client(base_url="http://192.0.2.10").post(
        "/api/v1/pipeline/reload", json={"data_dir": str(workspace)})
    assert refused.status_code == 403
    assert api_main.app_state.pipeline is None
    assert api_main.app_state.real_pipeline_requested is False

    response = _client(base_url="http://192.0.2.10").post("/api/v1/diagnose", json=_request())
    assert response.status_code == 503
    assert response.json()["error"].startswith(NOTHING_LOADED)


def test_no_fabricated_candidate_reaches_any_caller(monkeypatch, tmp_path):
    """Neither a configured deployment that failed nor one with nothing
    configured is handed the retired demo's candidates."""
    unconfigured = _client().post("/api/v1/diagnose", json=_request()).text
    monkeypatch.setenv("SHEPHERD_KG_PATH", str(tmp_path / "absent_kg.json"))
    configured = _client().post("/api/v1/diagnose", json=_request()).text

    for body in (unconfigured, configured):
        assert "Marfan" not in body and "MONDO:0007947" not in body
        assert "demo/v1" not in body and "mock" not in body


def test_the_refusal_is_not_relabelled_as_an_internal_error(monkeypatch, tmp_path):
    """`HTTPException` is an `Exception`, so the route's outer handler would turn
    a considered 503 into a 500 — which reads to a caller as a bug in the service
    rather than the service declining to answer."""
    monkeypatch.setenv("SHEPHERD_KG_PATH", str(tmp_path / "absent_kg.json"))

    assert _client().post("/api/v1/diagnose", json=_request()).status_code == 503
    monkeypatch.delenv("SHEPHERD_KG_PATH")
    assert _client().post("/api/v1/diagnose", json=_request()).status_code == 503


def test_an_unconfigured_deployment_is_refused_with_its_reason():
    """K7 (DEMO1): with no workspace named by any route, nothing answers. The
    retired demo answer is not given; the reason says what to do."""
    response = _client().post("/api/v1/diagnose", json=_request())

    assert response.status_code == 503
    body = response.json()
    assert body["error"].startswith(NOTHING_LOADED)
    assert "POST /api/v1/pipeline/reload" in body["error"]
    assert "candidates" not in body and "result_kind" not in body


def test_every_no_model_reason_names_the_load_entry_the_page_shows(monkeypatch):
    """Both reasons a 503 carries, and the build's refusal when no checkpoint is
    named, name the Diagnosis page's entry as the page labels it with no model
    loaded (`diagnosis_card.change_button`), say that loading is an operator
    control, and name the reload route. Compared here, so the API never imports
    the WebUI and the two cannot drift apart."""
    pytest.importorskip("gradio")
    from src.api import main as api_main
    from src.inference.pipeline import PipelineBuildError, require_model_source
    from src.webui.components import diagnosis_card as card

    label = card.change_button({"ok": True, "status": {"initialized": False},
                                "operator": True})["value"]
    entry = label.removeprefix("🔒").strip()
    assert entry and entry != label
    reasons = []
    for requested in (False, True):
        monkeypatch.setattr(api_main.app_state, "real_pipeline_requested", requested)
        reasons.append(api_main.app_state.no_pipeline_reason)
    with pytest.raises(PipelineBuildError) as refused:
        require_model_source(None, None)
    reasons.append(str(refused.value))
    for reason in reasons:
        assert f"“{entry}” on the Diagnosis page" in reason
        assert "An operator loads" in reason and "POST /api/v1/pipeline/reload" in reason


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


# ==============================================================================
# The reviewer's matrix (K7 audit §2A), through the real lifespan
# ==============================================================================
class TestTheNoModelMatrix:
    """Nothing configured, a graph alone, or a model that failed to load: the
    same three answers. A model loaded later serves; a bad reload keeps it."""

    @staticmethod
    def _surfaces(client):
        """What each surface answers, side by side, and the two refusals' words."""
        health, ready = client.get("/health"), client.get("/ready")
        diagnose = client.post("/api/v1/diagnose", json=_request())
        return {"health": health.status_code, "ready": ready.status_code,
                "diagnose": diagnose.status_code}, ready, diagnose

    NOT_SERVING = {"health": 200, "ready": 503, "diagnose": 503}
    SERVING = {"health": 200, "ready": 200, "diagnose": 200}

    @staticmethod
    def _configure(monkeypatch, trained, start, tmp_path):
        """The environment a deployment starts with, for each start state."""
        data_dir = trained["data_dir"]
        if start == "nothing configured":
            return None
        monkeypatch.setenv("SHEPHERD_KG_PATH", str(data_dir / "kg.json"))
        monkeypatch.setenv("SHEPHERD_DATA_DIR", str(data_dir))
        if start == "a graph alone":
            return None
        corrupt = tmp_path / "corrupt.pt"
        corrupt.write_bytes(b"not a checkpoint")
        monkeypatch.setenv("SHEPHERD_CHECKPOINT_PATH", str(corrupt))
        return corrupt

    @pytest.mark.parametrize("start, reason", [
        ("nothing configured", NOTHING_LOADED),
        ("a graph alone", CONFIGURED_BUT_FAILED),
        ("a model that failed to load", CONFIGURED_BUT_FAILED),
    ])
    def test_with_no_pipeline_ready_and_diagnose_refuse_and_health_answers(
            self, monkeypatch, trained_disease_child, tmp_path, start, reason):
        """Diagnose 503, ready 503 with the same reason, health 200. A graph
        named without a model is refused before `kg.json` is parsed: it is never
        served by path reasoning."""
        from src.api import main as api_main
        from src.kg.graph import KnowledgeGraph

        self._configure(monkeypatch, trained_disease_child, start, tmp_path)
        reads = []
        real_read = KnowledgeGraph.read_json.__func__

        def counted(cls, path):
            reads.append(path)
            return real_read(cls, path)

        monkeypatch.setattr(KnowledgeGraph, "read_json", classmethod(counted))

        with _client() as client:
            answers, ready, diagnose = self._surfaces(client)

        assert answers == self.NOT_SERVING
        assert ready.json()["error"] == diagnose.json()["error"]
        assert diagnose.json()["error"].startswith(reason)
        assert "candidates" not in diagnose.json()
        assert api_main.app_state.pipeline is None
        if start == "a graph alone":
            assert reads == [], "a graph with no model was read"

    def test_repeated_diagnoses_do_not_read_a_failed_source_again(
            self, monkeypatch, trained_disease_child, tmp_path):
        """The startup's one build read the bad checkpoint once; every later
        diagnosis is refused from the recorded state, with no build and no read
        (the way to try again is a reload or a restart)."""
        from src.api import main as api_main
        from tests.fixtures.opens import count_opens

        corrupt = self._configure(monkeypatch, trained_disease_child,
                                  "a model that failed to load", tmp_path)
        builds = []
        real_build = api_main.build_pipeline

        def counted(*args, **kwargs):
            builds.append(kwargs)
            return real_build(*args, **kwargs)

        monkeypatch.setattr(api_main, "build_pipeline", counted)

        with _client() as client:
            assert len(builds) == 1, "startup did not try the build once"
            with count_opens() as opens:
                statuses = [client.post("/api/v1/diagnose", json=_request()).status_code
                            for _ in range(3)]

        assert statuses == [503, 503, 503]
        assert len(builds) == 1, f"{len(builds)} builds for one startup and 3 diagnoses"
        assert opens.opens(corrupt) == 0
        assert opens.opens(trained_disease_child["data_dir"] / "kg.json") == 0

    @pytest.mark.parametrize("start", ["nothing configured", "a model that failed to load"])
    def test_a_valid_model_loaded_afterwards_serves(self, monkeypatch, trained_disease_child,
                                                    tmp_path, start):
        """The way back: an operator's reload of a valid model makes ready and
        diagnose answer, the diagnosis from that model."""
        self._configure(monkeypatch, trained_disease_child, start, tmp_path)

        with _client() as client:
            assert self._surfaces(client)[0] == self.NOT_SERVING
            loaded = client.post("/api/v1/pipeline/reload", json={
                "data_dir": str(trained_disease_child["data_dir"]),
                "checkpoint_path": str(trained_disease_child["disease"])}).json()
            assert loaded["success"] is True, loaded["message"]
            answers, ready, diagnose = self._surfaces(client)

        assert answers == self.SERVING
        assert ready.json()["pipeline_loaded"] is True
        assert diagnose.json()["result_kind"] == "disease_task/v1"

    def test_a_bad_reload_keeps_the_loaded_model_and_its_readiness(
            self, monkeypatch, trained_disease_child, tmp_path):
        """With a valid model served, a reload of a bad file is refused and
        changes nothing: the same pipeline, ready, and diagnoses from it."""
        from src.api import main as api_main

        data_dir = trained_disease_child["data_dir"]
        monkeypatch.setenv("SHEPHERD_KG_PATH", str(data_dir / "kg.json"))
        monkeypatch.setenv("SHEPHERD_DATA_DIR", str(data_dir))
        monkeypatch.setenv("SHEPHERD_CHECKPOINT_PATH", str(trained_disease_child["disease"]))
        corrupt = tmp_path / "corrupt.pt"
        corrupt.write_bytes(b"not a checkpoint")

        with _client() as client:
            answers, _ready, before = self._surfaces(client)
            assert answers == self.SERVING
            served = api_main.app_state.pipeline

            refused = client.post("/api/v1/pipeline/reload", json={
                "data_dir": str(data_dir), "checkpoint_path": str(corrupt)}).json()
            answers, _ready, after = self._surfaces(client)

        assert refused["success"] is False and "still being served" in refused["message"]
        assert api_main.app_state.pipeline is served
        assert answers == self.SERVING
        assert after.json()["served_identity"] == before.json()["served_identity"]

    def test_the_diagnosis_page_offers_the_load_and_keeps_its_gate(self):
        """With no model loaded the page's card says so, turns Run off with
        its reason and offers "Load a model…" to an operator; a read-only
        request sees the same reason and the load disabled, with D8's words."""
        pytest.importorskip("gradio")
        from src.webui.components import diagnosis_card as card
        from src.webui.components import diagnosis_panel as panel

        status = {"initialized": False}
        for operator in (True, False):
            state = {"ok": True, "status": status, "operator": operator,
                     "d8_short": None if operator else "request not local"}
            assert card.run_blocked(state)
            button = card.change_button(state)
            assert button["value"] == "🔒 Load a model…"
            assert button["interactive"] is operator
            assert ("Operator controls only" in card.change_reason_html(state)) is not operator
            assert "No model loaded" in card.body_html(state)
        assert "An operator" in panel.RUN_DISABLED_NO_MODEL_READ_ONLY
