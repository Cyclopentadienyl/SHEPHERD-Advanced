"""
Unit tests for the Diagnosis panel: tolerant HPO parsing and result export.
"""
import os

import pytest

# diagnosis_panel imports gradio at module load.
pytest.importorskip("gradio")
import gradio as gr  # noqa: E402

from src.webui.components import diagnosis_panel as dp  # noqa: E402
from src.webui.shell import standalone_context  # noqa: E402


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
def _digest(name):
    import hashlib

    return hashlib.sha256(name.encode()).hexdigest()


CHECKPOINT_SHA = _digest("checkpoint.pt")
KG_SHA = _digest("kg.json")
MANIFEST_SHA = _digest("split_manifest.json")
RECORD_SHA = _digest("kg.provenance.json")
BUILD_ID = "0123456789abcdef0123456789abcdef"


def _settings(**overrides):
    settings = {
        "device": "cuda:0", "device_requested": "auto", "eta": 0.7,
        "max_path_length": 4, "path_length_penalty": 0.9,
        "aggregation_method": "weighted_sum", "max_paths_per_candidate": 10,
        "use_direct_path_optimization": True, "phenotype_normalisation_version": 1,
    }
    settings.update(overrides)
    return settings


def _served_identity(origin="synthetic"):
    """What the API sends for a checkpoint served over a verified workspace with
    a verified shortest-path table (contract M3c), from a synthetic workspace by
    default."""
    return {
        "scoring_mode": "gnn_plus_shortest_path",
        "checkpoint": {"state": "verified", "sha256": CHECKPOINT_SHA},
        "graph": {"state": "verified", "kg_sha256": KG_SHA,
                  "split_manifest_sha256": MANIFEST_SHA},
        "shortest_paths": {"state": "verified", "build_id": BUILD_ID, "max_hops": 5},
        "dataset_record": {"state": "verified", "origin": origin, "sha256": RECORD_SHA},
        "settings": _settings(),
    }


def _path_only_identity():
    """Serving without a checkpoint from a graph file: no workspace verified."""
    return {
        "scoring_mode": "path_reasoning_fallback",
        "checkpoint": {"state": "absent", "sha256": None},
        "graph": {"state": "unverified", "kg_sha256": KG_SHA, "split_manifest_sha256": None},
        "shortest_paths": {"state": "absent", "build_id": None, "max_hops": None},
        "dataset_record": {"state": "absent", "origin": None, "sha256": None},
        "settings": _settings(device=None, device_requested=None, eta=None),
    }


def _gnn_only_identity():
    identity = _served_identity(origin="files")
    identity["scoring_mode"] = "gnn_only"
    identity["shortest_paths"] = {"state": "absent", "build_id": None, "max_hops": None}
    identity["settings"] = _settings(eta=1.0)
    return identity


def _in_memory_identity():
    """A model object and a graph supplied in memory (tests, notebooks)."""
    return {
        "scoring_mode": "gnn_only",
        "checkpoint": {"state": "supplied_in_memory", "sha256": None},
        "graph": {"state": "in_memory", "kg_sha256": None, "split_manifest_sha256": None},
        "shortest_paths": {"state": "absent", "build_id": None, "max_hops": None},
        "dataset_record": {"state": "absent", "origin": None, "sha256": None},
        "settings": _settings(device=None, device_requested=None, eta=1.0),
    }


def _sample_result():
    return {
        "session_id": "sess_abc",
        "patient_id": "pt_01",
        "timestamp": "2026-07-07T16:00:00Z",
        "served_identity": _served_identity(),
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
    result = _sample_result()
    result["served_identity"] = _served_identity(origin="files")
    csv_path, md_path = dp._write_exports(result)
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
        dp.create_diagnosis_tab(standalone_context(demo))
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
    """One load event writes the status and the two path fields (BACKLOG 17,
    F5), from the one status read `_on_load_status` makes."""
    demo = _build_tab(monkeypatch)

    on_load = [
        fn for fn in demo.fns.values()
        if fn.fn is dp._on_load_status and (demo._id, "load") in fn.targets
    ]
    assert len(on_load) == 1
    status_md, data_dir, checkpoint = on_load[0].outputs
    assert isinstance(status_md, gr.Markdown) and status_md.value == dp.STATUS_CHECKING
    assert isinstance(data_dir, gr.Textbox) and data_dir.label == "Data Directory"
    assert isinstance(checkpoint, gr.Textbox)
    assert checkpoint.label == "Checkpoint Path (optional)"


def test_the_reload_button_still_writes_the_same_status(monkeypatch):
    """Reload already refreshed the status; the load event targets the same
    component rather than a second copy of it. The path fields the page load
    sets are the ones Reload sends and, after it, sets."""
    demo = _build_tab(monkeypatch)

    (on_load,) = [fn for fn in demo.fns.values() if fn.fn is dp._on_load_status]
    (on_reload,) = [fn for fn in demo.fns.values() if fn.fn is dp._on_reload_pipeline]
    assert on_reload.outputs[0] is on_load.outputs[0]
    _, data_dir, checkpoint = on_load.outputs
    _, _, set_dir, set_checkpoint = on_reload.outputs
    sent_dir, sent_checkpoint, _ = on_reload.inputs
    assert set_dir is data_dir and set_checkpoint is checkpoint
    assert sent_dir is data_dir and sent_checkpoint is checkpoint


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


# -------------------------------------------- what produced the result (M3c)
# Contract M3c: the report, the CSV, the tab and the model status say what
# produced a result, part by part, from the `served_identity` the API copied
# from that result. Every part states its state; nothing is blank or "unknown".
import csv as _csv  # noqa: E402
import io as _io  # noqa: E402
import re as _re  # noqa: E402

#: The candidate columns as they were before M3c, in order: the identity is
#: appended after them, so a reader taking columns by position is not broken.
_ORIGINAL_CSV_COLUMNS = [
    "rank", "disease_id", "disease_name", "confidence_score", "confidence_label",
    "gnn_score", "sp_score", "reasoning_score", "evidence_mode", "evidence_summary",
    "min_path_length", "num_direct_paths", "num_analogies",
    "matching_phenotypes", "supporting_genes",
]
_SERVED_COLUMNS = [
    "served_scoring_mode", "served_checkpoint_state", "served_checkpoint_sha256",
    "served_kg_state", "served_kg_sha256", "served_split_manifest_sha256",
    "served_sp_state", "served_sp_build_id", "served_sp_max_hops",
    "served_dataset_record_state", "served_dataset_origin",
    "served_dataset_record_sha256", "served_device", "served_device_requested",
    "served_eta", "served_max_path_length", "served_path_length_penalty",
    "served_aggregation_method", "served_max_paths_per_candidate",
    "served_direct_path_optimization", "served_phenotype_normalisation_version",
]
_DIGEST_COLUMNS = [
    "served_checkpoint_sha256", "served_kg_sha256", "served_split_manifest_sha256",
    "served_dataset_record_sha256",
]
_CLAIM_BOUNDARY = (
    "Checked: these files are the ones their records name, and the records agree "
    "on the relations marked ✓. Not checked: who changed them, or anything a "
    "person able to edit both files and records could alter."
)
_NOT_A_DEVICE = "Not a certified medical device; decision support for a qualified clinician."
#: The qualifier under the boundary when nothing is marked ✓, in the words the
#: demo answer has always carried.
_NOTHING_CHECKED = (
    "No relation is marked ✓ for this result: nothing in it was checked against a record."
)
#: The same said of the loaded pipeline, under the model status.
_NOTHING_CHECKED_LOADED = (
    "No relation is marked ✓ for the loaded pipeline: nothing it serves was "
    "checked against a record."
)
_CHECKS_NOT_REPORTED = "The response did not say what produced it or what was checked."


def _path_reasoning_over_a_verified_workspace():
    """No checkpoint, over a verified workspace: the graph, its manifest and the
    dataset record are marked ✓, the checkpoint and the table are absent."""
    identity = _path_only_identity()
    identity["graph"] = {"state": "verified", "kg_sha256": KG_SHA,
                         "split_manifest_sha256": MANIFEST_SHA}
    identity["dataset_record"] = {"state": "verified", "origin": "files", "sha256": RECORD_SHA}
    return identity


_IDENTITIES = {
    "verified_synthetic": _served_identity,
    "verified_files": lambda: _served_identity(origin="files"),
    "path_only": _path_only_identity,
    "gnn_only": _gnn_only_identity,
    "in_memory": _in_memory_identity,
    "demo": lambda: None,
}


def _result_with(identity):
    result = _sample_result()
    result["served_identity"] = identity
    return result


def _report_section(md):
    return md[md.index("## What produced this result"):md.index("## Ranked candidates")]


def _table_rows(text):
    """The identity table's body rows as cell lists, split on unescaped pipes."""
    rows = []
    for line in text.splitlines():
        if line.startswith("| ") and not line.startswith("| Part |"):
            rows.append([cell.strip() for cell in _re.split(r"(?<!\\)\|", line)[1:-1]])
    return rows


def _csv_records(text):
    rows = list(_csv.reader(_io.StringIO(text)))
    return rows[0], [dict(zip(rows[0], row, strict=True)) for row in rows[1:]], rows


# ------------------------------------------------------------------ the report
def test_report_says_what_produced_the_result_before_the_candidates():
    md = dp._build_results_report_md(_sample_result())

    assert md.index("## What produced this result") < md.index("## Ranked candidates")
    section = _report_section(md)
    for digest in (CHECKPOINT_SHA, KG_SHA, MANIFEST_SHA, RECORD_SHA):
        assert f"`sha256:{digest}`" in section  # in full, never shortened
    assert f"`{BUILD_ID}`" in section and "max\\_hops 5" in section
    assert "`cuda:0` in effect; requested: `auto`" in section
    assert dp.IDENTITY_CLAIM_BOUNDARY == _CLAIM_BOUNDARY  # the plan's line, verbatim
    assert _CLAIM_BOUNDARY in section and _NOT_A_DEVICE in section
    assert "Model version" not in md and "1.0.0" not in md
    assert "unknown" not in md.lower()


@pytest.mark.parametrize("name", sorted(_IDENTITIES))
def test_every_identity_row_is_filled_and_never_unknown(name):
    section = _report_section(dp._build_results_report_md(_result_with(_IDENTITIES[name]())))
    rows = _table_rows(section)

    assert [row[0] for row in rows] == list(dp._IDENTITY_PARTS)
    for row in rows:
        assert len(row) == 3 and all(cell for cell in row), row
    assert "unknown" not in section.lower()


@pytest.mark.parametrize("name, banner", [
    ("verified_synthetic", "SYNTHETIC WORKSPACE: not a clinical result."),
    ("path_only", "NO TRAINED MODEL: path-reasoning fallback."),
    ("demo", "DEMO ANSWER: not computed."),
    ("gnn_only", "NO SHORTEST-PATH TABLE: GNN score alone."),
])
def test_report_banners_state_each_non_normal_state_in_words(name, banner):
    section = _report_section(dp._build_results_report_md(_result_with(_IDENTITIES[name]())))

    assert f"> **{banner}**" in section
    assert section.index(banner) < section.index("| Part |")


def test_a_normal_result_has_no_banner():
    section = _report_section(dp._build_results_report_md(_result_with(_served_identity("files"))))

    assert not [line for line in section.splitlines() if line.startswith(">")]


def test_a_response_without_an_identity_is_not_called_a_demo():
    result = _sample_result()
    del result["served_identity"]

    section = _report_section(dp._build_results_report_md(result))

    assert "> **WHAT PRODUCED THIS RESULT WAS NOT REPORTED.**" in section
    assert "DEMO" not in section
    assert all(row[1] == "not\\_reported" for row in _table_rows(section))


def test_absent_parts_are_written_as_their_state_and_reason():
    rows = {row[0]: row for row in _table_rows(dp._build_results_report_md(
        _result_with(_path_only_identity())))}

    checkpoint = rows["Checkpoint"]
    assert checkpoint[1] == "absent"
    assert checkpoint[2].startswith("absent — no trained model is served")
    assert rows["Workspace manifest (`split_manifest.json`)"][2].startswith("absent — ")
    assert rows["Knowledge graph (`kg.json`)"][1] == "unverified"
    assert f"`sha256:{KG_SHA}`" in rows["Knowledge graph (`kg.json`)"][2]
    assert rows["Device (in effect; requested)"][2].startswith("none — no model is served")
    assert rows["Mixing weight eta"][2].startswith("none — ")


def test_verified_states_carry_the_mark_the_claim_boundary_refers_to():
    rows = _table_rows(dp._build_results_report_md(_sample_result()))
    states = {row[0]: row[1] for row in rows}

    assert states["Checkpoint"] == "✓ verified"
    assert states["Shortest-path table"] == "✓ verified"
    # Never "verified" alone for a synthetic record (contract M3b).
    assert states["Dataset record (`kg.provenance.json`)"] == "✓ verified · synthetic"
    assert dp.IDENTITY_NOTHING_CHECKED not in dp._build_results_report_md(_sample_result())


@pytest.mark.parametrize("name", ["path_only", "in_memory", "demo"])
def test_with_nothing_marked_the_report_says_nothing_was_checked(name):
    section = _report_section(dp._build_results_report_md(_result_with(_IDENTITIES[name]())))

    assert "✓" not in "".join(row[1] for row in _table_rows(section))
    assert dp.IDENTITY_NOTHING_CHECKED in section


def _without_an_identity(how):
    result = _sample_result()
    if how == "missing":
        del result["served_identity"]
    else:
        result["served_identity"] = "not an object"
    return result


@pytest.mark.parametrize("how", ["missing", "not_an_object"])
def test_a_response_without_an_identity_does_not_say_nothing_was_checked(how):
    """Catches the report's qualifier asserting a negative the panel cannot
    know: a response that carried no served identity (a server from before
    contract M3c may well have checked its inputs) is reported as one that did
    not say what was checked, never as one in which nothing was."""
    section = _report_section(dp._build_results_report_md(_without_an_identity(how)))

    assert "nothing in it was checked" not in section
    assert dp.IDENTITY_CHECKS_NOT_REPORTED == _CHECKS_NOT_REPORTED
    assert section.index(_CLAIM_BOUNDARY) < section.index(_CHECKS_NOT_REPORTED)
    assert section.index(_CHECKS_NOT_REPORTED) < section.index(_NOT_A_DEVICE)


def test_values_from_the_api_cannot_break_the_table():
    identity = _served_identity()
    identity["shortest_paths"]["build_id"] = "a|b`c*d<script>"
    identity["settings"]["aggregation_method"] = "x|y\nz"

    rows = _table_rows(_report_section(dp._build_results_report_md(_result_with(identity))))

    assert len(rows) == len(dp._IDENTITY_PARTS)
    assert all(len(row) == 3 for row in rows)
    assert "``a\\|b`c*d<script>``" in rows[3][2]
    # Outside a table a code span keeps backslashes, so the pipe is not escaped.
    summary = dp._identity_summary_md("served", identity)
    assert "SP table ``a|b`c*d<scri…`` ✓" in summary


# ------------------------------------------------------------------- the CSV
def test_csv_keeps_one_header_and_appends_the_identity_after_the_candidates():
    header, records, rows = _csv_records(dp._build_results_csv(_sample_result()))

    assert header[:15] == _ORIGINAL_CSV_COLUMNS
    assert header[15:] == _SERVED_COLUMNS
    assert len(records) == 2 and all(len(row) == len(header) for row in rows)
    assert header not in rows[1:]  # exactly one header row, no preamble
    served = [{k: v for k, v in r.items() if k.startswith("served_")} for r in records]
    assert served[0] == served[1]  # the same identity on every row
    assert served[0] == {
        "served_scoring_mode": "gnn_plus_shortest_path",
        "served_checkpoint_state": "verified",
        "served_checkpoint_sha256": f"sha256:{CHECKPOINT_SHA}",
        "served_kg_state": "verified",
        "served_kg_sha256": f"sha256:{KG_SHA}",
        "served_split_manifest_sha256": f"sha256:{MANIFEST_SHA}",
        "served_sp_state": "verified",
        "served_sp_build_id": BUILD_ID,
        "served_sp_max_hops": "5",
        "served_dataset_record_state": "verified",
        "served_dataset_origin": "synthetic",
        "served_dataset_record_sha256": f"sha256:{RECORD_SHA}",
        "served_device": "cuda:0",
        "served_device_requested": "auto",
        "served_eta": "0.7",
        "served_max_path_length": "4",
        "served_path_length_penalty": "0.9",
        "served_aggregation_method": "weighted_sum",
        "served_max_paths_per_candidate": "10",
        "served_direct_path_optimization": "true",
        "served_phenotype_normalisation_version": "1",
    }


@pytest.mark.parametrize("name", sorted(_IDENTITIES))
def test_no_identity_cell_is_empty(name):
    _, records, _ = _csv_records(dp._build_results_csv(_result_with(_IDENTITIES[name]())))

    for record in records:
        for column in _SERVED_COLUMNS:
            assert record[column], f"{name}: {column} is empty"
        for column in _DIGEST_COLUMNS:
            value = record[column]
            assert _re.fullmatch(r"sha256:[0-9a-f]{64}", value) or value in (
                "absent", "none", "demo"), f"{name}: {column} = {value!r}"
        assert "unknown" not in ",".join(record.values()).lower()


def test_the_demo_answer_says_demo_in_every_identity_column():
    _, records, _ = _csv_records(dp._build_results_csv(_result_with(None)))

    assert {record[c] for record in records for c in _SERVED_COLUMNS} == {"demo"}


def test_a_response_without_an_identity_says_not_reported_in_every_column():
    result = _sample_result()
    del result["served_identity"]

    _, records, _ = _csv_records(dp._build_results_csv(result))

    assert {record[c] for record in records for c in _SERVED_COLUMNS} == {"not_reported"}


def test_absent_and_inapplicable_values_are_words():
    _, (path_only, _), _ = _csv_records(dp._build_results_csv(_result_with(_path_only_identity())))
    assert path_only["served_checkpoint_state"] == "absent"
    assert path_only["served_checkpoint_sha256"] == "absent"
    assert path_only["served_kg_state"] == "unverified"
    assert path_only["served_kg_sha256"] == f"sha256:{KG_SHA}"
    assert path_only["served_split_manifest_sha256"] == "absent"
    assert path_only["served_sp_build_id"] == "absent"
    assert path_only["served_dataset_origin"] == "absent"
    assert path_only["served_device"] == "none"
    assert path_only["served_eta"] == "none"

    _, (in_memory, _), _ = _csv_records(dp._build_results_csv(_result_with(_in_memory_identity())))
    assert in_memory["served_checkpoint_sha256"] == "none"
    assert in_memory["served_kg_sha256"] == "none"
    assert in_memory["served_device"] == "not_stated"
    assert in_memory["served_device_requested"] == "none"


def test_a_text_cell_a_spreadsheet_would_run_is_defused():
    result = _sample_result()
    result["candidates"][0]["disease_name"] = '=HYPERLINK("http://x")'
    result["candidates"][0]["gnn_score"] = -0.25
    result["served_identity"]["settings"]["aggregation_method"] = "@sum"

    _, (first, _), _ = _csv_records(dp._build_results_csv(result))

    assert first["disease_name"] == '\'=HYPERLINK("http://x")'
    assert first["served_aggregation_method"] == "'@sum"
    assert first["gnn_score"] == "-0.2500"  # numbers are not text cells


def test_export_names_carry_synthetic_and_demo():
    assert dp._export_basename(_sample_result()) == "diagnosis_SYNTHETIC_pt_01_sess_abc"
    assert dp._export_basename(_result_with(None)) == "diagnosis_DEMO_pt_01_sess_abc"
    assert dp._export_basename(_result_with(_path_only_identity())) == "diagnosis_pt_01_sess_abc"


# ------------------------------------------------------------------- the tab
def test_the_tab_shows_what_produced_the_result(monkeypatch):
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: _sample_result())

    shown = dp._on_diagnose("HP:0001250", "", 10)[0]

    # The banner, above the table.
    assert shown.index("**SYNTHETIC WORKSPACE: not a clinical result.**") < shown.index("| Rank |")
    # One summary line under the meta line, with short digests.
    assert shown.index("candidates found in") < shown.index("**Produced by:**")
    summary = shown[shown.index("**Produced by:**"):].splitlines()[0]
    assert f"checkpoint `sha256:{CHECKPOINT_SHA[:12]}…` ✓" in summary
    assert CHECKPOINT_SHA not in summary
    assert "`gnn_plus_shortest_path`" in summary and "dataset synthetic ✓" in summary
    # The full identity, one click away, closed, and the report's own table.
    assert "<details><summary>" in shown and "<details open" not in shown
    details = shown[shown.index("<details>"):shown.index("</details>")]
    assert f"`sha256:{CHECKPOINT_SHA}`" in details
    report = dp._build_results_report_md(_sample_result())
    assert _table_rows(details) == _table_rows(_report_section(report))
    assert _CLAIM_BOUNDARY in details and _NOT_A_DEVICE in details


def test_the_tab_marks_the_demo_answer(monkeypatch):
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: _result_with(None))

    shown = dp._on_diagnose("HP:0001250", "", 10)[0]

    assert shown.startswith("⚠️ **DEMO ANSWER: not computed.**")
    assert "**Produced by:** demo answer" in shown


def _details_block(shown):
    return shown[shown.index("<details>"):shown.index("</details>")]


def test_the_tab_does_not_say_nothing_was_checked_without_an_identity(monkeypatch):
    """Catches the tab's identity block claiming nothing was checked for a
    response that carried no served identity: like the report, it says the
    response did not say, under the same claim boundary."""
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: _without_an_identity("missing"))

    shown = dp._on_diagnose("HP:0001250", "", 10)[0]

    assert "nothing in it was checked" not in shown
    details = _details_block(shown)
    assert details.index(_CLAIM_BOUNDARY) < details.index(_CHECKS_NOT_REPORTED)


def test_the_demo_answer_still_says_nothing_was_checked(monkeypatch):
    """Catches the not-reported wording spreading to the demo answer, where
    nothing was served: its report and its tab block keep saying, in the same
    words as before, that nothing in it was checked against a record."""
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: _result_with(None))

    report = _report_section(dp._build_results_report_md(_result_with(None)))
    details = _details_block(dp._on_diagnose("HP:0001250", "", 10)[0])

    assert dp.IDENTITY_NOTHING_CHECKED == _NOTHING_CHECKED
    for text in (report, details):
        assert text.index(_CLAIM_BOUNDARY) < text.index(_NOTHING_CHECKED)
        assert _CHECKS_NOT_REPORTED not in text


def test_the_tab_shows_the_served_pipelines_own_identity(served):
    """End to end through the real route: a path-reasoning pipeline over an
    in-memory graph says so beside its result, and in its export."""
    out = dp._on_diagnose(f"{SEIZURE}\n{DELAY}", "", 10)

    identity = out[4]["served_identity"]
    assert identity["scoring_mode"] == "path_reasoning_fallback"
    assert identity["checkpoint"] == {"state": "absent", "sha256": None}
    assert "**NO TRAINED MODEL: path-reasoning fallback.**" in out[0]
    assert "no trained model (checkpoint absent)" in out[0]
    assert "graph in memory (no file)" in out[0]
    _, records, _ = _csv_records(dp._build_results_csv(out[4]))
    assert {r["served_checkpoint_state"] for r in records} == {"absent"}
    assert {r["served_kg_state"] for r in records} == {"in_memory"}


# ---------------------------------------------------------- the model status
def _loaded_status(**extra):
    status = {
        "initialized": True, "gnn_ready": True, "sp_ready": True,
        "scoring_mode": "gnn_plus_shortest_path", "kg_nodes": 10, "kg_edges": 20,
        "served_identity": _served_identity(),
    }
    status.update(extra)
    return status


def test_the_model_status_shows_what_the_loaded_pipeline_serves():
    rendered = dp._format_pipeline_status(_loaded_status())

    serving = next(line for line in rendered.splitlines() if "**Serving:**" in line)
    assert f"checkpoint `sha256:{CHECKPOINT_SHA[:12]}…` ✓" in serving
    assert f"SP table `{BUILD_ID[:12]}…` ✓" in serving
    assert _CLAIM_BOUNDARY in rendered
    assert "SYNTHETIC WORKSPACE" in rendered


@pytest.mark.parametrize("name", ["path_only", "in_memory"])
def test_a_status_with_nothing_marked_says_nothing_was_checked(name):
    """Catches the model status showing the §6 line alone under an identity
    with no part marked ✓, which reads as a claim that its graph file was
    checked against a record; the report and the tab add the qualifier."""
    rendered = dp._format_pipeline_status(_loaded_status(served_identity=_IDENTITIES[name]()))

    serving = next(line for line in rendered.splitlines() if "**Serving:**" in line)
    assert "✓" not in serving
    after = rendered[rendered.index(serving):]
    assert after.index(f"<sub>{_CLAIM_BOUNDARY}</sub>") < after.index(
        f"<sub>{_NOTHING_CHECKED_LOADED}</sub>")
    # Said of the loaded pipeline, not of a result: the status shows no result.
    assert _NOTHING_CHECKED not in rendered
    assert _CHECKS_NOT_REPORTED not in rendered


@pytest.mark.parametrize("name", ["verified_synthetic", "verified_files", "gnn_only"])
def test_a_status_with_a_verified_part_does_not_say_nothing_was_checked(name):
    """Catches the qualifier being added to every status: where a part is
    marked ✓, the §6 line refers to it and "nothing was checked" would be
    false."""
    rendered = dp._format_pipeline_status(_loaded_status(served_identity=_IDENTITIES[name]()))

    assert f"<sub>{_CLAIM_BOUNDARY}</sub>" in rendered
    assert "nothing in it was checked" not in rendered
    assert _NOTHING_CHECKED_LOADED not in rendered
    assert _CHECKS_NOT_REPORTED not in rendered


_STATUS_IDENTITIES = {
    **{name: make for name, make in _IDENTITIES.items() if name != "demo"},
    "path_reasoning_verified_workspace": _path_reasoning_over_a_verified_workspace,
}


@pytest.mark.parametrize("name", sorted(_STATUS_IDENTITIES))
def test_the_status_and_the_report_agree_on_whether_anything_was_checked(name):
    """Catches the model status deciding "nothing checked" by a rule of its own
    (for example from the checkpoint alone): for every served identity it adds
    the qualifier exactly when the report's identity table does."""
    make = _STATUS_IDENTITIES[name]
    status = dp._format_pipeline_status(_loaded_status(served_identity=make()))
    section = _report_section(dp._build_results_report_md(_result_with(make())))
    nothing_marked = "✓" not in "".join(row[1] for row in _table_rows(section))

    assert (_NOTHING_CHECKED in section) is nothing_marked
    assert (_NOTHING_CHECKED_LOADED in status) is nothing_marked
    assert _CLAIM_BOUNDARY in status and _CLAIM_BOUNDARY in section


def test_a_status_without_an_identity_says_so():
    status = _loaded_status()
    del status["served_identity"]

    rendered = dp._format_pipeline_status(status)

    assert "**Serving:** not reported" in rendered
    assert "SYNTHETIC" not in rendered


def test_fingerprint_warnings_are_named_for_what_they_compare():
    rendered = dp._format_pipeline_status(_loaded_status(fingerprint_warnings=[
        "Graph structure differs from the checkpoint's structural fingerprint.",
        "Node count mismatch for 'gene': checkpoint expects 3, current data has 4",
    ]))

    assert "KG version mismatch" not in rendered
    assert "structural fingerprint** (2 warnings;" in rendered
    assert "not the graph's identity" in rendered


def test_the_panel_no_longer_reads_a_model_version():
    from pathlib import Path

    assert "model_version" not in Path(dp.__file__).read_text(encoding="utf-8")
