"""
Unit tests for the Diagnosis panel: tolerant HPO parsing and result export.
"""
import os

import pytest

# diagnosis_panel imports gradio at module load.
pytest.importorskip("gradio")
import gradio as gr  # noqa: E402

from src.webui.components import diagnosis_panel as dp  # noqa: E402


# ------------------------------------------------------------------- HPO parsing
def test_parse_strict_one_per_line():
    assert dp._parse_hpo_ids("HP:0001250 — Seizure\nHP:0001263 — Delay") == [
        "HP:0001250",
        "HP:0001263",
    ]


def test_parse_comma_and_space_separated():
    assert dp._parse_hpo_ids("HP:0001250, HP:0001263  HP:0002376") == [
        "HP:0001250",
        "HP:0001263",
        "HP:0002376",
    ]


def test_parse_tolerates_case_missing_colon_and_junk():
    text = "  hp:0001250 ;; HP_0001263 , seizure(HP:0002376)!!  hp 0004322"
    assert dp._parse_hpo_ids(text) == [
        "HP:0001250",
        "HP:0001263",
        "HP:0002376",
        "HP:0004322",
    ]


def test_parse_keeps_order_and_repeats():
    """Repeats are the service's to remove, by graph node, and to count
    (docs/working/PLAN_PHENOTYPE_NORMALISATION.md, decision W)."""
    assert dp._parse_hpo_ids("HP:0001263\nHP:0001250\nHP:0001263") == [
        "HP:0001263",
        "HP:0001250",
        "HP:0001263",
    ]


def test_parse_does_not_grab_prefix_of_longer_number():
    # An 8+ digit run must not yield a bogus 7-digit id.
    assert dp._parse_hpo_ids("HP:00012501234 HP:0009999") == ["HP:0009999"]


def test_parse_no_ids_returns_empty():
    assert dp._parse_hpo_ids("patient has seizures and developmental delay") == []
    assert dp._parse_hpo_ids("") == []


# ------------------------------------------------------------------- export
def _sample_result():
    return {
        "session_id": "sess_abc",
        "patient_id": "pt_01",
        "timestamp": "2026-07-07T16:00:00Z",
        "model_version": "1.0.0",
        "inference_time_ms": 123.4,
        "warnings": ["low phenotype count"],
        "_query_phenotypes": ["HP:0001250", "HP:0001263"],
        "candidates": [
            {
                "rank": 1,
                "disease_id": "mondo:MONDO:0011073",
                "disease_name": "Dravet syndrome",
                "confidence_score": 0.72,
                "gnn_score": 0.81,
                "sp_score": 0.55,
                "confidence_label": "Strong path support",
                "matching_phenotypes": ["HP:0001250"],
                "supporting_genes": ["SCN1A"],
                "explanation": "Because ...",
                "evidence_package": {
                    "mode": "direct_path",
                    "summary": "Direct 2-hop path",
                    "min_path_length": 2,
                    "direct_paths": [["hp:HP:0001250", "mondo:MONDO:0011073"]],
                    "analogies": [],
                },
            },
            {
                "rank": 2,
                "disease_id": "omim:OMIM:123",
                "disease_name": "Other disease",
                "confidence_score": 0.44,
                "confidence_label": "Analogy-based",
                "matching_phenotypes": [],
                "supporting_genes": [],
                "explanation": None,
                "evidence_package": {"mode": "analogy_based", "summary": "Analogy"},
            },
        ],
    }


def test_csv_has_row_per_candidate_with_cleaned_ids():
    csv_text = dp._build_results_csv(_sample_result())
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("rank,disease_id,disease_name,confidence_score")
    assert len(lines) - 1 == 2  # header + 2 candidates
    assert "Dravet syndrome" in csv_text
    assert "MONDO:0011073" in csv_text and "mondo:MONDO" not in csv_text  # cleaned id
    assert "SCN1A" in csv_text


def test_report_includes_meta_and_all_candidates_with_cleaned_ids():
    md = dp._build_results_report_md(_sample_result())
    assert "# SHEPHERD-Advanced Diagnosis Report" in md
    assert "pt_01" in md  # patient id
    assert "HP:0001250" in md  # query phenotypes
    assert "Dravet syndrome" in md and "Other disease" in md  # every candidate
    assert "Full explanation" in md
    assert "`MONDO:0011073`" in md and "mondo:MONDO" not in md  # cleaned id


# ------------------------------------------------------------------- disease id cleaning
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("mondo:MONDO:0019441", "MONDO:0019441"),
        ("omim:OMIM:123", "OMIM:123"),
        ("orphanet:ORPHA:456", "ORPHA:456"),
        ("orpha:ORPHA:456", "ORPHA:456"),
        ("mondo:OMIM:123", "mondo:OMIM:123"),  # cross-namespace: untouched
        ("MONDO:0019441", "MONDO:0019441"),  # bare CURIE: untouched
        ("  mondo:MONDO:0019441  ", "MONDO:0019441"),  # surrounding whitespace
        ("mondo:MONDO:0019441\n", "MONDO:0019441"),  # trailing newline
        ("  MONDO:0019441  ", "MONDO:0019441"),  # padded bare CURIE
        ("", ""),
        (None, ""),
        ("weird", "weird"),
    ],
)
def test_clean_disease_id(raw, expected):
    assert dp._clean_disease_id(raw) == expected


# ------------------------------------------------------------------- export files
def test_write_exports_names_by_patient_and_session():
    csv_path, md_path = dp._write_exports(_sample_result())
    assert os.path.exists(csv_path) and os.path.exists(md_path)
    assert os.path.basename(csv_path) == "diagnosis_pt_01_sess_abc.csv"
    assert os.path.basename(md_path) == "diagnosis_pt_01_sess_abc_report.md"
    # both files live in the single module-level export dir
    assert os.path.dirname(csv_path) == os.path.dirname(md_path)


def test_write_exports_reuses_single_dir():
    p1, _ = dp._write_exports(_sample_result())
    p2, _ = dp._write_exports(_sample_result())
    assert os.path.dirname(p1) == os.path.dirname(p2)


def test_export_basename_falls_back_when_ids_missing():
    base = dp._export_basename({"candidates": []})
    assert base == "diagnosis_webui_patient_run"


# ------------------------------------------------------------------- stale-state clear
def test_phenotype_change_noops_without_results():
    out = dp._on_phenotype_change(None)
    assert len(out) == 7
    assert out[4] is None  # results_state stays cleared/None
    # component outputs are gr.update() no-ops (leave UI untouched)
    assert out[0] is not dp._DOWNLOAD_DISABLED


def test_phenotype_change_clears_after_a_run():
    out = dp._on_phenotype_change(_sample_result())
    assert len(out) == 7
    assert "Inputs changed" in out[0]
    assert out[4] is None  # results_state cleared
    assert out[5] is dp._DOWNLOAD_DISABLED and out[6] is dp._DOWNLOAD_DISABLED


# ------------------------------------------------------------------- diagnose download wiring
def test_on_diagnose_no_hpo_disables_downloads():
    out = dp._on_diagnose("no ids here", "", 10)
    assert len(out) == 7
    assert out[4] is None  # results_state
    assert out[5] is dp._DOWNLOAD_DISABLED and out[6] is dp._DOWNLOAD_DISABLED


def test_on_diagnose_success_sets_download_values(monkeypatch):
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: _sample_result())
    out = dp._on_diagnose("HP:0001250", "", 10)
    assert len(out) == 7
    assert out[4] is not None and out[4].get("candidates")  # results_state populated
    # downloads enabled with a real file value (not the disabled sentinel)
    assert out[5] is not dp._DOWNLOAD_DISABLED and out[6] is not dp._DOWNLOAD_DISABLED


def test_on_diagnose_export_failure_still_shows_results(monkeypatch):
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: _sample_result())

    def boom(_result):
        raise OSError("disk full")

    monkeypatch.setattr(dp, "_write_exports", boom)
    out = dp._on_diagnose("HP:0001250", "", 10)
    assert len(out) == 7
    # results still shown, with a note; state still populated
    assert "Dravet syndrome" in out[0] and "could not be written" in out[0]
    assert out[4] is not None and out[4].get("candidates")
    # downloads stay disabled (recoverable by re-running once disk is free)
    assert out[5] is dp._DOWNLOAD_DISABLED and out[6] is dp._DOWNLOAD_DISABLED


# ------------------------------------------------------------ status on page load
def _build_tab(monkeypatch):
    """Build the tab and fail if the build made any call to the API.

    Calls are recorded, not refused: the status helpers catch every exception, so
    a fake that raised would be swallowed and the build would look clean. Three
    layers are watched -- the status read itself, the HTTP helper under every call,
    and requests' own Session.request beneath both -- so a status read or any other
    HTTP call during the build is caught whether or not an address is recorded.
    """
    calls = []
    real_status = dp._get_pipeline_status

    def watched_status():
        calls.append("_get_pipeline_status")
        return real_status()

    def watched_request(method, path, **kwargs):
        calls.append(f"{method} {path}")
        raise AssertionError("no API call is expected while the app is built")

    def watched_session_request(session, method, url, **kwargs):
        calls.append(f"requests {method} {url}")
        raise AssertionError("no HTTP request is expected while the app is built")

    monkeypatch.setattr(dp, "_get_pipeline_status", watched_status)
    monkeypatch.setattr(dp, "_self_request", watched_request)
    # Underneath both: any requests call, including a direct requests.get.
    monkeypatch.setattr(dp.requests.Session, "request", watched_session_request)
    with gr.Blocks() as demo:
        dp.create_diagnosis_tab(demo)
    assert calls == [], f"the tab called the API while the app was being built: {calls}"
    return demo


def test_the_model_status_is_not_read_while_the_app_is_built(monkeypatch):
    """The app is built before the server serves, so a status read then was stale
    on arrival -- and under the systemd unit it was also aimed at the wrong port.
    `_build_tab` fails on any API call during the build; the placeholder stands
    until the first page load reads the real status."""
    demo = _build_tab(monkeypatch)

    placeholders = [
        block for block in demo.blocks.values()
        if isinstance(block, gr.Markdown) and block.value == dp.STATUS_CHECKING
    ]
    assert len(placeholders) == 1


def test_the_model_status_is_read_on_every_page_load(monkeypatch):
    demo = _build_tab(monkeypatch)

    on_load = [
        fn for fn in demo.fns.values()
        if fn.fn is dp._on_load_status and (demo._id, "load") in fn.targets
    ]
    assert len(on_load) == 1
    (status_md,) = on_load[0].outputs
    assert isinstance(status_md, gr.Markdown) and status_md.value == dp.STATUS_CHECKING


def test_the_reload_button_still_writes_the_same_status(monkeypatch):
    """Reload already refreshed the status; the load event targets the same
    component rather than a second copy of it."""
    demo = _build_tab(monkeypatch)

    (on_load,) = [fn for fn in demo.fns.values() if fn.fn is dp._on_load_status]
    (on_reload,) = [fn for fn in demo.fns.values() if fn.fn is dp._on_reload_pipeline]
    assert on_reload.outputs[0] is on_load.outputs[0]


# ------------------------------------------------- the phenotype list and the API
# docs/working/PLAN_PHENOTYPE_NORMALISATION.md, N1 and decision W: the panel sends
# what it parsed, repeats included; the service removes repeats and enforces the
# list's limit; the panel says in words why a request was refused. These tests
# drive the real handler against the real application, with `_self_request`
# routed into a TestClient instead of a socket.
from tests.fixtures.phenotype_kg import (  # noqa: E402
    DELAY,
    SEIZURE,
    UNKNOWN,
    build_two_phenotype_kg,
)

_TOO_MANY = "HPO phenotype entries (repeats included)"


def _as_requests_response(response):
    """A TestClient (httpx) response as the `requests.Response` the panel expects,
    so `raise_for_status` raises `requests.HTTPError` as it does in production."""
    import requests

    converted = requests.Response()
    converted.status_code = response.status_code
    converted._content = response.content
    converted.headers.update(response.headers)
    converted.url = str(response.url)
    converted.encoding = "utf-8"
    return converted


@pytest.fixture
def served(monkeypatch):
    """The panel's API calls reach the real app, which serves a real
    path-reasoning pipeline. Returns the payloads sent and the pipeline runs."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api import main as api_main
    from src.inference.pipeline import DiagnosisPipeline

    pipeline = DiagnosisPipeline(kg=build_two_phenotype_kg())
    record = {"payloads": [], "runs": 0}
    original_run = pipeline.run

    def counted_run(**kwargs):
        record["runs"] += 1
        return original_run(**kwargs)

    monkeypatch.setattr(pipeline, "run", counted_run)
    monkeypatch.setattr(api_main.app_state, "pipeline", pipeline, raising=False)
    client = TestClient(api_main.app)

    def self_request(method, path, **kwargs):
        kwargs.pop("timeout", None)
        record["payloads"].append(kwargs.get("json"))
        return _as_requests_response(client.request(method, path, **kwargs))

    monkeypatch.setattr(dp, "_self_request", self_request)
    return record


def _assert_refused_without_results(out):
    assert out[1] == "" and out[2] == ""
    assert out[4] is None  # results_state cleared: no earlier result passes as this one
    assert out[5] is dp._DOWNLOAD_DISABLED and out[6] is dp._DOWNLOAD_DISABLED


def test_one_hundred_repeats_then_another_is_refused_with_both_numbers(served):
    """Several ids to a line: the message counts entries, not lines."""
    text = "\n".join(" ".join([SEIZURE] * 10) for _ in range(10)) + f"\n{DELAY}"

    out = dp._on_diagnose(text, "", 10)

    assert served["payloads"][0]["phenotypes"] == [SEIZURE] * 100 + [DELAY]
    assert "Recognised 101 " + _TOO_MANY in out[0]
    assert "at most 100" in out[0]
    assert "was not shortened" in out[0]
    assert served["runs"] == 0  # the pipeline is never reached
    _assert_refused_without_results(out)


def test_one_hundred_and_one_distinct_ids_get_the_same_refusal(served):
    text = ", ".join(f"HP:{n:07d}" for n in range(1, 102))

    out = dp._on_diagnose(text, "", 10)

    assert "Recognised 101 " + _TOO_MANY in out[0]
    assert served["runs"] == 0
    _assert_refused_without_results(out)


def test_repeats_within_the_limit_are_sent_and_removed_by_the_service(served):
    out = dp._on_diagnose(f"{SEIZURE} {SEIZURE} {DELAY}", "", 10)

    assert served["payloads"][0]["phenotypes"] == [SEIZURE, SEIZURE, DELAY]
    assert served["runs"] == 1
    assert out[4]["summary"].endswith("for 2 phenotypes (3 received)")
    assert "Removed 1 repeated phenotype entry" in out[0]


def test_a_corrected_list_shows_its_own_result_and_no_stale_error(served):
    refused = dp._on_diagnose(" ".join([SEIZURE] * 101), "", 10)
    assert _TOO_MANY in refused[0]

    out = dp._on_diagnose(f"{SEIZURE}\n{DELAY}", "", 10)

    assert _TOO_MANY not in out[0] and "Error" not in out[0]
    assert out[4] is not None and out[4].get("candidates")
    assert out[4]["summary"].endswith("for 2 phenotypes")


def test_the_input_box_is_never_an_output(monkeypatch):
    """A refusal leaves the user's text where it was: the handler cannot write
    to the box it reads from."""
    demo = _build_tab(monkeypatch)
    (handler,) = [f for f in demo.fns.values() if f.fn is dp._on_diagnose]

    phenotype_box = handler.inputs[0]
    assert isinstance(phenotype_box, gr.Textbox)
    assert phenotype_box not in handler.outputs


# ------------------------------------------------ reading a refusal, never failing
def _response(status, body=None, text=None):
    import json as _json

    import requests

    response = requests.Response()
    response.status_code = status
    response._content = (text if text is not None else _json.dumps(body)).encode()
    response.encoding = "utf-8"
    return response


def _too_long_entry(**ctx):
    return {"type": "too_long", "loc": ["body", "phenotypes"],
            "msg": "List should have at most 100 items after validation, not 101",
            "input": [SEIZURE] * 101, "ctx": ctx}


def test_the_limit_message_takes_both_numbers_from_the_server():
    message = dp._describe_refused_request(
        _response(422, {"detail": [_too_long_entry(max_length=7, actual_length=9)]})
    )

    assert message.startswith("Recognised 9 " + _TOO_MANY)
    assert "at most 7" in message
    assert SEIZURE not in message  # the echoed input is never shown


@pytest.mark.parametrize("ctx", [
    {},
    {"max_length": 100},
    {"actual_length": 101},
    {"max_length": "100", "actual_length": 101},
    {"max_length": 100, "actual_length": True},
])
def test_without_both_numbers_no_limit_is_guessed(ctx):
    message = dp._describe_refused_request(_response(422, {"detail": [_too_long_entry(**ctx)]}))

    assert _TOO_MANY not in message
    assert "100" not in message.replace("at most 100 items", "")
    assert message == (
        "Request refused — phenotypes: List should have at most 100 items after "
        "validation, not 101"
    )


def test_a_too_long_entry_without_ctx_at_all_falls_back_too():
    entry = _too_long_entry()
    del entry["ctx"]

    message = dp._describe_refused_request(_response(422, {"detail": [entry]}))

    assert message.startswith("Request refused — phenotypes:")


@pytest.mark.parametrize("response", [
    _response(422, text="<html>gateway</html>"),
    _response(422, {"error": "no detail key"}),
    _response(422, {"detail": "a string, not a list"}),
    _response(422, {"detail": []}),
    _response(422, {"detail": ["not an entry", 3, None]}),
    _response(422, {"detail": [{"loc": ["body"], "msg": ""}]}),
    _response(422, ["a", "list", "body"]),
])
def test_an_unreadable_refusal_gets_the_general_line(response):
    assert dp._describe_refused_request(response) == dp._UNREADABLE_REFUSAL


def test_a_confidence_refusal_is_not_presented_as_too_many_phenotypes():
    """The UI sends no confidences, so the real API's refusal is taken from the
    route itself and read by the panel."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api.main import app

    api_response = TestClient(app).post(
        "/api/v1/diagnose",
        json={"phenotypes": [SEIZURE, DELAY], "phenotype_confidences": [0.5]},
    )
    assert api_response.status_code == 422

    message = dp._describe_refused_request(_as_requests_response(api_response))

    assert _TOO_MANY not in message
    assert message == (
        "Request refused — request: phenotype_confidences has 1 entries for 2 "
        "phenotypes; it needs one per phenotype, or none at all"
    )


def test_a_connection_failure_is_reported_without_raising(monkeypatch):
    import requests

    def unreachable(method, path, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(dp, "_self_request", unreachable)

    out = dp._on_diagnose(SEIZURE, "", 10)

    assert "API server not reachable" in out[0]
    _assert_refused_without_results(out)


def test_other_http_errors_keep_their_message(monkeypatch):
    monkeypatch.setattr(
        dp, "_self_request",
        lambda method, path, **kwargs: _response(500, text="Internal Server Error"),
    )

    result = dp._call_diagnose(phenotypes=[SEIZURE])

    assert result == {"error": "API error: 500 — Internal Server Error"}


def test_an_unknown_id_is_reported_and_not_counted(served):
    out = dp._on_diagnose(f"{SEIZURE} {UNKNOWN} {DELAY}", "", 10)

    assert f"Unknown phenotype: {UNKNOWN}" in out[0]
    assert out[4]["summary"].endswith("for 2 phenotypes (3 received)")
