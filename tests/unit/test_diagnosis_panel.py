"""
Unit tests for the Diagnosis page: tolerant HPO parsing, result export, the
handlers (Run Diagnosis, the phenotypes, the table and the details), the
results badge, Run Diagnosis' states, Change model…'s D8 check, the identity
drawer, and the page as built in the shell.
"""
import os

import pytest

# diagnosis_panel imports gradio at module load.
pytest.importorskip("gradio")
import gradio as gr  # noqa: E402

from src.webui.components import diagnosis_panel as dp  # noqa: E402
from tests.fixtures.clock_zone import utc_clock  # noqa: E402,F401


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


# ------------------------------------------------------------------- the page's handlers
def _no_model_status():
    return {"initialized": False}


def _loaded_status(**extra):
    """A status serving `_served_identity()` (the records tests use it too)."""
    status = {
        "initialized": True, "gnn_ready": True, "sp_ready": True,
        "scoring_mode": "gnn_plus_shortest_path", "kg_nodes": 10, "kg_edges": 20,
        "served_identity": _served_identity(),
    }
    status.update(extra)
    return status


def _status_serving(identity, **extra):
    status = {"initialized": True, "gnn_ready": True, "served_identity": identity,
              "checkpoint_meta": {"epoch": 5}, "fingerprint_warnings": []}
    status.update(extra)
    return status


# The local stamps these tests write are in UTC, the zone the page shows.
pytestmark = pytest.mark.usefixtures("utc_clock")


@pytest.fixture
def status_read(monkeypatch):
    """The card's status read, answered by the test (a list, popped in order;
    the last answer stays); returns the list of reads made."""
    from src.webui.components import diagnosis_card as card

    answers = [(_no_model_status(), None)]
    reads = []

    def read(seconds=card.STATUS_TIMEOUT_SECONDS, reader=None):
        reads.append(seconds)
        return answers.pop(0) if len(answers) > 1 else answers[0]

    monkeypatch.setattr(card, "read_status_within", read)
    monkeypatch.setattr(card, "served_facts", lambda status, **kw: {
        "key": dp.card.served_key(status), "operator": kw.get("operator"), "workspaces": [],
        "serving": None, "item": None, "item_problem": None, "recorded": None, "trained": None})
    return {"answers": answers, "reads": reads}


def _run(text, patient="", top_k=10, explanations=True, state=None, phase="idle", view=None,
         request=None):
    """Run Diagnosis as the page does; the outputs by name."""
    return dp.diagnose(text, patient, top_k, explanations, state, phase, view or {}, request)


def test_phenotype_change_noops_without_results():
    assert dp.phenotypes_changed(None, None) == {}


def test_phenotype_change_clears_after_a_run():
    out = dp.phenotypes_changed(_sample_result(), None)
    assert "Inputs changed" in out["results_body"]
    assert out["results_state"] is None and out["selected"] is None and out["badge"] == ""
    assert out["export_csv"] is dp._EXPORTS_HIDDEN
    assert out["export_report"] is dp._EXPORTS_HIDDEN


def test_on_diagnose_no_hpo_disables_downloads(status_read):
    out = _run("no ids here")
    assert "No valid HPO terms found" in out["results_body"]
    assert out["results_state"] is None
    assert out["export_csv"] is dp._EXPORTS_HIDDEN and out["export_report"] is dp._EXPORTS_HIDDEN
    assert status_read["reads"] == []   # nothing was asked of the server


def test_on_diagnose_success_sets_download_values(monkeypatch, status_read):
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: _sample_result())
    out = _run("HP:0001250")
    assert out["results_state"] is not None and out["results_state"].get("candidates")
    assert out["export_csv"] is not dp._DOWNLOAD_DISABLED
    assert out["export_report"] is not dp._DOWNLOAD_DISABLED
    # A result shows the exports, which no result had shown (M5-3).
    assert out["export_csv"]["visible"] is True and out["export_csv"]["interactive"] is True
    assert out["export_report"]["visible"] is True and out["export_report"]["interactive"] is True
    assert out["selected"] == 1 and "Dravet syndrome" in out["details_hd"]


def test_on_diagnose_export_failure_still_shows_results(monkeypatch, status_read):
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: _sample_result())

    def boom(_result):
        raise OSError("disk full")

    monkeypatch.setattr(dp, "_write_exports", boom)
    out = _run("HP:0001250")
    assert "Dravet syndrome" in out["results_body"]
    assert "could not be written" in out["results_body"]
    assert out["results_state"] is not None and out["results_state"].get("candidates")
    assert out["export_csv"] is dp._DOWNLOAD_DISABLED and out["export_report"] is dp._DOWNLOAD_DISABLED
    # A result exists: its exports are shown, disabled, beside the reason.
    assert dp._DOWNLOAD_DISABLED["visible"] is True


def test_no_export_buttons_are_shown_before_a_result():
    """M5-3: the first screen showed two disabled Export buttons; the mockup
    shows none until a result exists. They are built hidden (kept mounted),
    no result keeps them hidden, and their ⓘ is hidden with them."""
    from src.webui import shell

    pages = {page: (lambda ctx: None) for page in (*shell.RESEARCH_PAGES, shell.SETTINGS)}
    with gr.Blocks() as blocks:
        shell.build_shell(blocks, {**pages, shell.CLINICAL: dp.create_diagnosis_tab})
    buttons = {c.elem_id: c for c in blocks.blocks.values()
               if getattr(c, "elem_id", None) in ("shep-dx-export-csv", "shep-dx-export-md")}
    assert set(buttons) == {"shep-dx-export-csv", "shep-dx-export-md"}
    assert all(b.visible == "hidden" for b in buttons.values())
    assert dp._EXPORTS_HIDDEN["visible"] == "hidden" and dp._EXPORTS_HIDDEN["value"] is None
    assert "#shep-dx-results-ft:has(#shep-dx-export-csv.hidden) #shep-dx-exports-help" in dp.PAGE_CSS


def test_a_blank_patient_id_is_sent_as_none(monkeypatch):
    sent = []

    class _Ok:
        def raise_for_status(self):
            return None

        def json(self):
            return {"candidates": []}

    monkeypatch.setattr(dp, "_self_request",
                        lambda method, path, **kw: sent.append(kw["json"]) or _Ok())
    dp._call_diagnose(["HP:0001250"], patient_id="   ")
    dp._call_diagnose(["HP:0001250"], patient_id="")
    dp._call_diagnose(["HP:0001250"], patient_id=" pt_7 ")
    assert [payload["patient_id"] for payload in sent] == [None, None, "pt_7"]


def test_the_status_is_read_again_after_each_diagnosis(monkeypatch, status_read):
    """The badge compares the result with a read made after it: one read per
    diagnosis, within the 5 s limit, after the API answered."""
    order = []
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: order.append("api") or _sample_result())
    status_read["answers"][:] = [(_status_serving(_served_identity()), None)]
    real = dp.card.read_status_within

    def read(*a, **k):
        order.append("status")
        return real(*a, **k)

    monkeypatch.setattr(dp.card, "read_status_within", read)
    out = _run("HP:0001250")
    assert order == ["api", "status"]
    assert "shep-dx-badge-ok" in out["badge"]


# ------------------------------------------------------------ the page as built
def _build_tab(monkeypatch):
    """Build the page in the real shell and fail if the build made any call to
    the API or read the status: the app is built before the server serves, so a
    read then would be stale on arrival."""
    from src.webui import shell
    from src.webui.components import diagnosis_card as card

    calls = []

    def watched_request(method, path, **kwargs):
        calls.append(f"{method} {path}")
        raise AssertionError("no API call is expected while the app is built")

    def watched_session_request(session, method, url, **kwargs):
        calls.append(f"requests {method} {url}")
        raise AssertionError("no HTTP request is expected while the app is built")

    monkeypatch.setattr(dp, "_self_request", watched_request)
    monkeypatch.setattr(dp.requests.Session, "request", watched_session_request)
    monkeypatch.setattr(card, "read_status_within",
                        lambda *a, **k: calls.append("status") or (None, "x"))
    with gr.Blocks() as demo:
        # Every page needs a builder (a missing one is a KeyError): the others
        # build nothing here.
        ctx = shell.build_shell(demo, {**{page: (lambda ctx: None)
                                          for page in (*shell.RESEARCH_PAGES, shell.SETTINGS)},
                                       shell.CLINICAL: dp.create_diagnosis_tab})
    assert calls == [], f"the page called the API while the app was being built: {calls}"
    return demo, ctx


def _named(demo, name):
    found = [fn for fn in demo.fns.values() if fn.name == name]
    assert found, name
    return found


def test_the_model_in_use_is_not_read_while_the_app_is_built(monkeypatch):
    """`_build_tab` fails on any read during the build; the card says it is
    checking until the first page load reads the real status."""
    demo, _ctx = _build_tab(monkeypatch)
    bodies = [b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "shep-dx-card-body"]
    assert len(bodies) == 1 and "Checking the model in use" in bodies[0].value


def test_the_card_is_read_on_load_when_shown_after_a_load_and_on_the_timer(monkeypatch):
    demo, ctx = _build_tab(monkeypatch)
    refreshes = _named(demo, "_card_refresh")
    targets = {t for fn in refreshes for t in fn.targets}
    assert (demo._id, "load") in targets
    assert (ctx.tabs["clinical"]._id, "select") in targets
    assert (ctx.served_model_changed._id, "change") in targets
    assert all(fn.queue is not False for fn in refreshes)   # may open the checkpoint list
    (poll,) = _named(demo, "_card_poll")
    assert poll.targets == [(ctx.timer_slow._id, "tick")]
    assert poll.queue is False and poll.inputs[0] is ctx.location


def test_change_model_asks_d8_then_reads_the_list_then_the_chips(monkeypatch):
    from src.webui.components.chips import read_chips

    demo, ctx = _build_tab(monkeypatch)
    (opened,) = [fn for fn in demo.fns.values() if fn.fn is dp._on_change_model]
    listing = next(fn for fn in demo.fns.values() if fn.trigger_after == opened._id)
    assert listing.name == "_refresh" and listing.fn.__qualname__.startswith("create_model_list")
    chips = next(fn for fn in demo.fns.values() if fn.trigger_after == listing._id)
    assert chips.fn is read_chips


def test_run_diagnosis_is_off_while_a_model_change_is_open(monkeypatch):
    demo, _ctx = _build_tab(monkeypatch)
    (changed,) = _named(demo, "_phase_changed")
    assert changed.targets[0][1] == "change"
    run = [b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "shep-dx-run-btn"][0]
    assert run in changed.outputs


def test_the_page_css_is_registered_and_scoped_to_the_page():
    from src.webui import app, shell

    assert dp.PAGE_CSS in app.PAGE_CSS
    assert dp._TAB == "#" + shell.tab_elem_id(shell.CLINICAL)
    assert dp.CLINICAL == shell.CLINICAL
    rules = [line for line in dp.PAGE_CSS.splitlines() if "{" in line and not line.startswith(("@", "}"))]
    assert all(line.lstrip().startswith(dp._TAB) for line in rules), [
        line for line in rules if not line.lstrip().startswith(dp._TAB)]


def test_the_page_css_hides_a_view_by_its_class_and_keeps_the_confirmation_in_view():
    """The views are hidden by `VIEW_OFF` (never ``visible``), so the page's
    stylesheet must hide that class. Seen in review: the Load confirmation's
    from → to rows and its warning sentence scrolled out of view at 1707x870,
    1440x900 and 1536x760 (checked in a browser at the wide sizes); while the
    confirmation box is mounted the list's footer is hidden and the list gives
    up its height to it (all of it below 820 px of height)."""
    css = " ".join(dp.PAGE_CSS.split())
    assert f"{dp._TAB} .{dp.VIEW_OFF} {{ display: none !important; }}" in css
    confirming = f"{dp._TAB} #shep-dx-change .shep-model-list:has(.shep-model-confirm-box)"
    assert f"{confirming} > .shep-model-foot {{ display: none !important; }}" in css
    wide = css.index("@media (min-width: 1400px) and (min-height: 760px) {")
    assert css.index(f"{confirming} > .shep-model-body {{ min-height: 0 !important; }}") > wide
    low = css.index("@media (min-width: 1400px) and (min-height: 760px) and (max-height: 819px) {")
    assert css.index(f"{confirming} > .shep-model-body {{ display: none !important; }}") > low


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
def served(monkeypatch, status_read):
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
    assert "After a diagnosis" in out["details_html"]["value"]
    assert dp.VIEW_OFF not in out["details_html"]["elem_classes"]
    assert out["results_state"] is None  # no earlier result passes as this one
    assert out["export_csv"] is dp._EXPORTS_HIDDEN
    assert out["export_report"] is dp._EXPORTS_HIDDEN


def test_one_hundred_repeats_then_another_is_refused_with_both_numbers(served):
    """Several ids to a line: the message counts entries, not lines."""
    text = "\n".join(" ".join([SEIZURE] * 10) for _ in range(10)) + f"\n{DELAY}"

    out = _run(text)

    assert served["payloads"][0]["phenotypes"] == [SEIZURE] * 100 + [DELAY]
    assert "Recognised 101 " + _TOO_MANY in out["results_body"]
    assert "at most 100" in out["results_body"]
    assert "was not shortened" in out["results_body"]
    assert served["runs"] == 0  # the pipeline is never reached
    _assert_refused_without_results(out)


def test_one_hundred_and_one_distinct_ids_get_the_same_refusal(served):
    text = ", ".join(f"HP:{n:07d}" for n in range(1, 102))

    out = _run(text)

    assert "Recognised 101 " + _TOO_MANY in out["results_body"]
    assert served["runs"] == 0
    _assert_refused_without_results(out)


def test_repeats_within_the_limit_are_sent_and_removed_by_the_service(served):
    out = _run(f"{SEIZURE} {SEIZURE} {DELAY}")

    assert served["payloads"][0]["phenotypes"] == [SEIZURE, SEIZURE, DELAY]
    assert served["runs"] == 1
    assert out["results_state"]["summary"].endswith("for 2 phenotypes (3 received)")
    assert "Removed 1 repeated phenotype entry" in out["results_body"]


def test_a_corrected_list_shows_its_own_result_and_no_stale_error(served):
    refused = _run(" ".join([SEIZURE] * 101))
    assert _TOO_MANY in refused["results_body"]

    out = _run(f"{SEIZURE}\n{DELAY}")

    assert _TOO_MANY not in out["results_body"] and "Error" not in out["results_body"]
    assert out["results_state"] is not None and out["results_state"].get("candidates")
    assert out["results_state"]["summary"].endswith("for 2 phenotypes")


def test_the_input_box_is_never_an_output(monkeypatch):
    """A refusal leaves the user's text where it was: the handler cannot write
    to the box it reads from."""
    demo, _ctx = _build_tab(monkeypatch)
    (handler,) = _named(demo, "_run_diagnosis")

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


def test_a_connection_failure_is_reported_without_raising(monkeypatch, status_read):
    import requests

    def unreachable(method, path, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(dp, "_self_request", unreachable)

    out = _run(SEIZURE)

    assert "API server not reachable" in out["results_body"]
    _assert_refused_without_results(out)


def test_other_http_errors_keep_their_message(monkeypatch):
    monkeypatch.setattr(
        dp, "_self_request",
        lambda method, path, **kwargs: _response(500, text="Internal Server Error"),
    )

    result = dp._call_diagnose(phenotypes=[SEIZURE])

    assert result == {"error": "API error: 500 — Internal Server Error"}


def test_an_unknown_id_is_reported_and_not_counted(served):
    out = _run(f"{SEIZURE} {UNKNOWN} {DELAY}")

    assert f"Unknown phenotype: {UNKNOWN}" in out["results_body"]
    assert out["results_state"]["summary"].endswith("for 2 phenotypes (3 received)")


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


# ------------------------------------------------------------------- the page
def test_the_page_shows_what_produced_the_result(monkeypatch, status_read):
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: _sample_result())

    out = _run("HP:0001250")

    body = out["results_body"]
    # The banner, above the table; its detail behind the shared ⓘ.
    assert body.index("SYNTHETIC WORKSPACE: not a clinical result.") < body.index("<table")
    assert 'aria-describedby="shep-help-text-dx-banner-synthetic"' in body
    # "This result" (the identity drawer): the result's own rows, as the
    # report reads them, with full digests shortened in the table only.
    shown = dp.result_identity_md(out["results_state"], out["card_state"])
    for part, state, value in dp._identity_rows("served", _served_identity()):
        assert f"| {part} | {dp._short_digests(f'{state} · {value}')} |" in shown
    assert CHECKPOINT_SHA not in shown and f"{CHECKPOINT_SHA[:12]}…" in shown
    assert _CLAIM_BOUNDARY in shown and _NOT_A_DEVICE in shown
    digests = dp.result_digests_html(out["results_state"])
    for digest in (CHECKPOINT_SHA, KG_SHA, MANIFEST_SHA, RECORD_SHA):
        assert f'data-copy="{digest}"' in digests


def test_the_page_marks_the_demo_answer(monkeypatch, status_read):
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: _result_with(None))

    out = _run("HP:0001250")

    assert "DEMO ANSWER: not computed." in out["results_body"]
    assert "shep-dx-banner-red" in out["results_body"]
    assert "This is the demo answer" in out["badge"]
    assert "Produced by the model in use" not in out["badge"]


def test_the_page_does_not_say_nothing_was_checked_without_an_identity(monkeypatch, status_read):
    """Like the report, "This result" says the response did not say, under the
    same claim boundary."""
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: _without_an_identity("missing"))

    out = _run("HP:0001250")
    shown = dp.result_identity_md(out["results_state"], out["card_state"])

    assert "nothing in it was checked" not in shown
    assert shown.index(_CLAIM_BOUNDARY) < shown.index(_CHECKS_NOT_REPORTED)
    assert "did not say what produced it" in out["badge"]


def test_the_demo_answer_still_says_nothing_was_checked(monkeypatch, status_read):
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: _result_with(None))

    report = _report_section(dp._build_results_report_md(_result_with(None)))
    out = _run("HP:0001250")
    shown = dp.result_identity_md(out["results_state"], out["card_state"])

    assert dp.IDENTITY_NOTHING_CHECKED == _NOTHING_CHECKED
    for text in (report, shown):
        assert text.index(_CLAIM_BOUNDARY) < text.index(_NOTHING_CHECKED)
        assert _CHECKS_NOT_REPORTED not in text


def test_the_page_shows_the_served_pipelines_own_identity(served):
    """End to end through the real route: a path-reasoning pipeline over an
    in-memory graph says so beside its result, and in its export."""
    out = _run(f"{SEIZURE}\n{DELAY}")

    identity = out["results_state"]["served_identity"]
    assert identity["scoring_mode"] == "path_reasoning_fallback"
    assert identity["checkpoint"] == {"state": "absent", "sha256": None}
    assert "NO TRAINED MODEL: path-reasoning fallback." in out["results_body"]
    shown = dp.result_identity_md(out["results_state"], out["card_state"])
    assert "absent — no trained model is served" in shown
    assert "none — the graph was supplied in memory" in shown
    _, records, _ = _csv_records(dp._build_results_csv(out["results_state"]))
    assert {r["served_checkpoint_state"] for r in records} == {"absent"}
    assert {r["served_kg_state"] for r in records} == {"in_memory"}


# -------------------------------------------------- the badge: the result against what is served
_NOW = "2026-07-07T16:00:00"


def _card(status=None, problem=None, *, at=_NOW, state=None, operator=True):
    from datetime import datetime

    return dp.card.advance(state, status, problem, now=datetime.fromisoformat(at),
                           operator=operator, d8_short=None if operator else "request not local",
                           open_checkpoints=False, facts=lambda s, **k: {
                               "key": dp.card.served_key(s), "operator": operator,
                               "workspaces": [], "serving": None, "item": None,
                               "item_problem": None, "recorded": None, "trained": None})


def _produced(at="2026-07-07T15:59:59"):
    result = _sample_result()
    result["timestamp"] = at
    return result


def test_the_badge_says_equal_only_for_the_whole_identity_read_after_the_result():
    state = _card(_status_serving(_served_identity()))
    assert dp.compare_with_served(_produced(), state)["state"] == "equal"
    assert "shep-dx-badge-ok" in dp.badge_html(_produced(), state)
    # One part of the served identity differs (the SP table): never equal.
    other = _served_identity()
    other["shortest_paths"] = {"state": "absent", "build_id": None, "max_hops": None}
    other["scoring_mode"] = "gnn_only"
    other["settings"] = _settings(eta=1.0)
    compared = dp.compare_with_served(_produced(), _card(_status_serving(other)))
    assert compared["state"] == "differs"
    assert compared["differ"] == ["Shortest-path table", "Scoring mode", "Mixing weight eta"]
    badge = dp.badge_html(_produced(), _card(_status_serving(other)))
    assert "Produced by an earlier model or pipeline" in badge and "3 parts differ" in badge


def test_a_stale_or_unreadable_status_never_leaves_equal_showing():
    good = _card(_status_serving(_served_identity()))
    stale = dp.compare_with_served(_produced("2026-07-07T16:00:30"), good)
    assert stale["state"] == "unknown"
    assert ("last read at 16:00:00 UTC, before this result was produced (16:00:30 UTC)"
            in stale["reason"])
    failed = _card(None, "timed out (5 s)", state=good)
    unknown = dp.compare_with_served(_produced(), failed)
    # The mockup's words, one level of parentheses (seen in review: "failed
    # (timed out (5 s))").
    assert unknown["state"] == "unknown"
    assert unknown["reason"].startswith("The status read failed (timed out after 5 s), so ")
    from src.webui.components.served_status import STATUS_NOT_READ

    unreadable = dp.compare_with_served(_produced(), _card(None, STATUS_NOT_READ, state=good))
    assert f"The status read failed ({STATUS_NOT_READ}), so " in unreadable["reason"]
    for state, result in ((good, _produced("2026-07-07T16:00:30")), (failed, _produced())):
        badge = dp.badge_html(result, state)
        assert "Cannot check against the model in use yet" in badge
        assert "shep-dx-badge-ok" not in badge and "shep-dx-badge-grey" in badge
        assert "keeps its own identity" in badge


def test_a_result_older_than_a_load_says_how_many_parts_differ():
    load = _card(_status_serving(_served_identity()))
    other = _served_identity()
    other["checkpoint"] = {"state": "verified", "sha256": "b" * 64}
    later = _card(_status_serving(other), state=load, at="2026-07-07T16:31:12")
    assert dp.compare_with_served(_produced(), later)["differ"] == ["Checkpoint"]
    assert "1 part differs" in dp.badge_html(_produced(), later)
    shown = dp.result_identity_md(_produced(), later)
    assert "1 of 10 parts differs" in shown and "status read 16:31:12" in shown
    assert "· differs**" in shown and "= same" in shown


# -------------------------------------------------- Run Diagnosis
def test_run_is_greyed_with_a_bubble_and_a_reason_when_no_model_is_loaded():
    from src.webui.components.help import text_id

    run = dp.run_controls(_card(_no_model_status()), "idle")
    assert run["run_btn"]["interactive"] is False
    assert f'aria-describedby="{text_id("dx-run-no-model")}"' in run["run_cover"]
    assert 'class="shep-i shep-i-cover"' in run["run_cover"]
    assert dp.RUN_DISABLED_NO_MODEL in run["run_reason"]
    read_only = dp.run_controls(_card(_no_model_status(), operator=False), "idle")
    assert dp._e(dp.RUN_DISABLED_NO_MODEL_READ_ONLY) in read_only["run_reason"]
    assert "dx-run-no-model" in dp.TOPIC_HELP


def test_run_stays_on_for_path_reasoning_and_an_unreadable_status_but_not_a_model_change():
    graph = _status_serving(_path_only_identity(), gnn_ready=False)
    assert dp.run_controls(_card(graph), "idle")["run_btn"]["interactive"] is True
    assert dp.run_controls(_card(None, "timed out (5 s)"), "idle")["run_cover"] == ""
    for phase in ("confirming", "loading"):
        change = dp.run_controls(_card(_status_serving(_served_identity())), phase)
        assert change["run_btn"]["interactive"] is False
        assert dp.RUN_DISABLED_CHANGE in change["run_reason"] and "dx-run-change" in change["run_cover"]


# -------------------------------------------------- the card's outputs and the timer
def test_the_card_sends_only_what_changed():
    state = _card(_status_serving(_served_identity()))
    first = dp.card_outputs(state, None, "idle")
    assert {"card_body", "badge", "run_btn", "change_btn"} <= set(first)
    again = dp.card_outputs(first["card_state"], None, "idle")
    assert set(again) == {"card_state"}


def test_the_timer_reads_only_while_diagnosis_is_shown_and_opens_no_checkpoint(monkeypatch):
    seen = []

    def read_card(state, request, *, open_checkpoints, reread=False, now=None):
        seen.append(open_checkpoints)
        return _card(_no_model_status())

    monkeypatch.setattr(dp, "read_card", read_card)
    assert dp.poll_card("datasets", None, None, "idle", {}, None) == {}
    out = dp.poll_card("clinical", None, None, "idle", {}, None)
    assert seen == [False] and "No model loaded" in out["card_body"]


# -------------------------------------------------- Change model…
def test_change_model_is_refused_with_the_reason_and_opens_nothing(monkeypatch):
    from tests.fixtures.operator_access import gradio_request, operator_request

    refused = dp.open_change(gradio_request())
    assert "Refused: " in refused["change_reason"]
    assert refused["change_drawer"]["visible"] is False
    allowed = dp.open_change(operator_request(monkeypatch))
    assert allowed["change_drawer"]["visible"] is True
    assert allowed["identity_drawer"]["visible"] is False
    outputs = dp._on_change_model(operator_request(monkeypatch))
    assert len(outputs) == len(dp.CHANGE_OUTPUTS)


def test_closing_the_change_drawer_drops_an_open_confirmation():
    closed = dp.close_change("confirming")
    assert closed["plan"] is None and closed["phase"] == "idle"
    assert closed["change_drawer"]["visible"] is False and closed["change_closed"] is True
    # Nothing inside the drawer is updated in the response that hides it (seen
    # in review: +738 detached elements per open and Cancel); the second step
    # resets the confirmation inside the hidden drawer.
    assert set(closed) == {"change_drawer", "plan", "phase", "change_closed"}
    reset = dp.reset_change(closed["change_closed"])
    assert reset["confirm"]["visible"] is False and reset["confirm_html"] == ""
    assert reset["list_reason"] == "" and reset["change_closed"] is False
    assert dp.reset_change(False) == {}


def test_a_running_reload_keeps_the_change_drawer_open():
    """The reload cannot be stopped and its outcome must be seen: × and Cancel
    keep the drawer, the plan and the phase while it runs."""
    kept = dp.close_change("loading")
    assert kept == {"change_drawer": kept["change_drawer"], "change_closed": False}
    assert kept["change_drawer"]["visible"] is True
    assert dp.reset_change(kept["change_closed"]) == {}


@pytest.mark.parametrize("phase", ["confirming", "loading"])
def test_details_never_hide_a_model_change_in_progress(phase):
    """Seen in review: Details ›, Recorded tests › or the badge during a Load
    confirmation hid the change drawer and kept the phase, so Run Diagnosis
    stayed disabled ("finish or cancel the model change first") with no
    model change on screen."""
    state = _card(_status_serving(_served_identity()))
    for seg, section in (("model", "identity"), ("model", "evidence"), ("result", "identity")):
        out = dp.open_identity(state, _sample_result(), seg, section, phase)
        assert out == {"change_drawer": out["change_drawer"]}
        assert out["change_drawer"]["visible"] is True
    opened = dp.open_identity(state, _sample_result(), "model", "identity", "idle")
    assert opened["identity_view"]["open"] is True and opened["change_drawer"]["visible"] is False


def test_the_page_passes_the_model_change_phase_to_the_identity_drawer(monkeypatch):
    demo, _ctx = _build_tab(monkeypatch)
    phases = {fn.name: fn for fn in demo.fns.values()
              if fn.name in ("_open_details", "_open_result", "_card_link")}
    assert set(phases) == {"_open_details", "_open_result", "_card_link"}
    phase_states = [b for b in demo.blocks.values()
                    if isinstance(b, gr.State) and b.value == "idle"]
    for name, fn in phases.items():
        assert any(i in phase_states for i in fn.inputs), name


def test_closing_the_change_drawer_resets_it_in_a_second_step(monkeypatch):
    demo, _ctx = _build_tab(monkeypatch)
    drawer = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "shep-dx-change")
    inside = {b for b in demo.blocks.values()
              if getattr(b, "elem_id", None) in ("shep-models-clinical-confirm",
                                                 "shep-models-clinical-reason")}
    assert len(inside) == 2
    for name in ("_close_change_drawer", "_cancel_change"):
        (closing,) = _named(demo, name)
        assert drawer in closing.outputs and not inside & set(closing.outputs), name
        reset = next(fn for fn in demo.fns.values() if fn.trigger_after == closing._id)
        assert reset.name == "_reset_change" and inside <= set(reset.outputs)


# -------------------------------------------------- the candidates and the details
def test_values_from_the_api_are_escaped_in_the_table_and_the_details():
    result = _sample_result()
    evil = "<img src=x onerror=alert(1)>"
    result["candidates"][0].update(disease_name=evil, supporting_genes=[evil],
                                   matching_phenotypes=[evil])
    result["candidates"][0]["evidence_package"].update(summary=evil, direct_paths=[[evil, "b"]])
    html = (dp.results_body_html(result, 1) + dp.details_header_html(result["candidates"][0])
            + dp.evidence_html(result["candidates"][0]) + dp.paths_html(result["candidates"][0])
            + dp.phenotypes_html(result["candidates"][0]))
    assert "<img" not in html and "&lt;img src=x onerror=alert(1)&gt;" in html


def test_the_sp_column_reads_a_dash_when_no_table_is_served():
    result = _result_with(_gnn_only_identity())
    result["candidates"][0]["sp_score"] = 0.0
    assert ">0.000</td>" not in dp.results_body_html(result, 1).split("<tbody>")[1].split(
        "</tr>")[0].replace(">0.720</td>", "")
    assert "SP — (no table)" in dp.details_header_html(result["candidates"][0], True)
    assert ">0.550</td>" in dp.results_body_html(_sample_result(), 1)


def test_a_selected_row_shows_its_details_with_its_counts():
    out = dp.select_candidate(_sample_result(), "evidence", "2")
    assert out["selected"] == 2 and "Other disease" in out["details_hd"]
    choices = dp.segment_choices(_sample_result()["candidates"][0])
    assert choices == [("Evidence", "evidence"), ("Explanation", "explanation"),
                       ("Matching phenotypes 1", "phenotypes"), ("Paths 1", "paths")]
    assert dp.select_candidate(_sample_result(), "evidence", "99") == {}
    paths = dp.details_outputs(_sample_result(), 1, "paths")
    assert "MONDO:0011073" in paths["details_html"]["value"]
    explanation = dp.details_outputs(_sample_result(), 1, "explanation")
    assert dp.VIEW_OFF not in explanation["details_md"]["elem_classes"]
    assert explanation["details_md"]["value"] == "Because ..."
    assert dp.VIEW_OFF in explanation["details_html"]["elem_classes"]


#: What may still be shown or hidden with ``visible``: the two drawers (whole
#: columns) and Details › (which changes only with what is served).
_VISIBLE_ALLOWED = {"identity_drawer", "change_drawer", "details_btn"}


def _shown(update):
    return dp.VIEW_OFF not in update["elem_classes"]


def _assert_views_switch_by_class(out):
    """Seen in review: switching ``visible`` on single components (+77 and +114
    detached elements per switch) and updating a component in the response
    that hides its column (+738) leave detached page elements in Gradio 6.30.
    The views are shown and hidden by a class instead, which keeps them
    mounted, and a hidden view is not rendered."""
    for name, value in out.items():
        if not isinstance(value, dict):
            continue
        if "visible" in value:
            assert name in _VISIBLE_ALLOWED, name
        if "elem_classes" in value:
            assert value["elem_classes"], name          # never empty (ignored by Gradio)
            if not _shown(value):
                assert set(value) == {"__type__", "elem_classes"}, (name, value)


def test_the_details_switch_views_by_class():
    result = _sample_result()
    for rank in (None, 1, 2):
        for segment in ("evidence", "explanation", "phenotypes", "paths"):
            out = dp.details_outputs(result, rank, segment)
            _assert_views_switch_by_class(out)
            assert _shown(out["details_seg"]) is (rank is not None)
            assert _shown(out["details_html"]) is not _shown(out["details_md"])
            assert "shep-html" in out["details_html"]["elem_classes"]
    _assert_views_switch_by_class(dp.select_candidate(result, "explanation", "1"))


def test_the_identity_drawer_switches_views_by_class():
    state = _card(_status_serving(_served_identity()))
    result = _produced()
    for seg in ("model", "result"):
        view = {"open": True, "seg": seg, "section": "identity"}
        out = dp.identity_outputs(view, state, result)
        _assert_views_switch_by_class(out)
        assert _shown(out["id_model"]) is (seg == "model")
        assert _shown(out["id_result_md"]) is _shown(out["id_result_html"]) is (seg == "result")
        _assert_views_switch_by_class(dp.card_outputs(state, result, "idle", view))
    _assert_views_switch_by_class(dp.open_identity(state, result, "result"))


def test_the_views_start_as_their_first_update_leaves_them(monkeypatch):
    """The built components' classes are the ones `view_update` sends for the
    first screen (no result): the segmented control and the explanation
    hidden, the identity drawer on "Model in use"."""
    demo, _ctx = _build_tab(monkeypatch)
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    first = dp.details_outputs(None, None)
    for elem_id, name in (("shep-dx-seg", "details_seg"), ("shep-dx-details-body", "details_html"),
                          ("shep-dx-explanation", "details_md")):
        assert list(by_id[elem_id].elem_classes) == first[name]["elem_classes"], elem_id
    model = dp.identity_outputs({"open": True, "seg": "model"},
                                _card(_status_serving(_served_identity())), None)
    for elem_id, name in (("shep-dx-id-model", "id_model"), ("shep-dx-id-result", "id_result_md"),
                          ("shep-dx-id-digests", "id_result_html")):
        assert list(by_id[elem_id].elem_classes) == model[name]["elem_classes"], elem_id
    for block in demo.blocks.values():
        if getattr(block, "elem_id", "") in ("shep-dx-seg", "shep-dx-details-body",
                                              "shep-dx-explanation", "shep-dx-id-model",
                                              "shep-dx-id-result", "shep-dx-id-digests"):
            assert block.visible is True


# -------------------------------------------------- the model in use (identity drawer)
def test_the_model_in_use_shows_the_boundary_for_what_is_loaded():
    html = dp.model_identity_html(_card(_status_serving(_served_identity())))
    assert _CLAIM_BOUNDARY in html and _NOT_A_DEVICE in html
    assert _NOTHING_CHECKED_LOADED not in html and "SYNTHETIC WORKSPACE" in html


@pytest.mark.parametrize("name", ["path_only", "in_memory"])
def test_a_model_in_use_with_nothing_marked_says_nothing_was_checked(name):
    html = dp.model_identity_html(_card(_status_serving(_IDENTITIES[name]())))
    assert html.index(_CLAIM_BOUNDARY) < html.index(_NOTHING_CHECKED_LOADED)
    assert _NOTHING_CHECKED not in html and _CHECKS_NOT_REPORTED not in html


_STATUS_IDENTITIES = {
    **{name: make for name, make in _IDENTITIES.items() if name != "demo"},
    "path_reasoning_verified_workspace": _path_reasoning_over_a_verified_workspace,
}


@pytest.mark.parametrize("name", sorted(_STATUS_IDENTITIES))
def test_the_model_in_use_and_the_report_agree_on_whether_anything_was_checked(name):
    """The drawer adds the qualifier exactly when the report's identity table
    does, for every served identity (one rule, `_claim_boundary_lines`)."""
    make = _STATUS_IDENTITIES[name]
    html = dp.model_identity_html(_card(_status_serving(make())))
    section = _report_section(dp._build_results_report_md(_result_with(make())))
    nothing_marked = "✓" not in "".join(row[1] for row in _table_rows(section))

    assert (_NOTHING_CHECKED in section) is nothing_marked
    assert (_NOTHING_CHECKED_LOADED in html) is nothing_marked
    assert _CLAIM_BOUNDARY in html and _CLAIM_BOUNDARY in section


def test_a_status_without_an_identity_says_so():
    status = _status_serving(_served_identity())
    del status["served_identity"]
    state = _card(status)
    assert "The status did not say what is served." in dp.card.body_html(state)
    assert _CHECKS_NOT_REPORTED in dp.model_identity_html(state)


def test_the_panel_no_longer_reads_a_model_version():
    from pathlib import Path

    assert "model_version" not in Path(dp.__file__).read_text(encoding="utf-8")



def test_a_code_chip_in_this_result_is_text_in_its_line():
    """M5-3: Gradio draws a Markdown code chip as an inline flex box, before
    which a line may break, and the rule broke anywhere inside it: "Knowledge
    graph (" / "kg.json)". The chip is inline text, breaking only where it
    must."""
    start = dp.PAGE_CSS.index("#shep-dx-id-result code {")
    rule = dp.PAGE_CSS[start:dp.PAGE_CSS.index("}", start)]
    assert "display: inline;" in rule and "word-break: normal" in rule
    assert "break-all" not in rule
