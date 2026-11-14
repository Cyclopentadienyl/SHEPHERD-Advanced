"""
Unit tests for the Diagnosis page: tolerant HPO parsing, result export, the
handlers (Run Diagnosis, the phenotypes, the table and the details), the
results badge, Run Diagnosis' states, Change model…'s D8 check, the identity
drawer, and the page as built in the shell.
"""
import os
import re

import pytest

# diagnosis_panel imports gradio at module load.
pytest.importorskip("gradio")
import gradio as gr  # noqa: E402

from src.webui.components import diagnosis_panel as dp  # noqa: E402
from src.webui.components import renderers as r  # noqa: E402
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


def _settings(**overrides):
    settings = {
        "device": "cuda:0", "device_requested": "auto", "eta": None,
        "max_path_length": 4, "path_length_penalty": 0.9,
        "max_paths_per_candidate": 10,
        "use_direct_path_optimization": True, "phenotype_normalisation_version": 1,
    }
    settings.update(overrides)
    return settings


def _served_identity(origin="synthetic"):
    """What the API sends for a disease task model served over a verified
    workspace (contract M3c), from a synthetic workspace by default: no
    shortest-path term, no mixing weight."""
    return {
        "scoring_mode": "disease_task",
        "checkpoint": {"state": "verified", "sha256": CHECKPOINT_SHA},
        "graph": {"state": "verified", "kg_sha256": KG_SHA,
                  "split_manifest_sha256": MANIFEST_SHA},
        "shortest_paths": {"state": "absent"},
        "dataset_record": {"state": "verified", "origin": origin, "sha256": RECORD_SHA},
        "settings": _settings(),
    }


def _parts_left_out():
    """A served identity whose response left out every part a record checks
    (the checkpoint, graph, table and dataset record): no part is marked ✓.
    Since K7 every result comes from a model read from its checkpoint, so only
    a malformed reply has nothing marked."""
    return {"scoring_mode": "disease_task", "settings": _settings()}


def _candidates():
    """Two candidates of `disease_task/v1` (Spec 1 §2.1), ranked by the disease
    head's score: the first with a direct path, the second with no usable
    evidence."""
    first = {"rank": 1, "disease_id": "mondo:MONDO:0011073", "disease_name": "Dravet syndrome",
             "matching_phenotypes": ["HP:0001250"], "score": -0.28}
    second = {"rank": 2, "disease_id": "omim:OMIM:123", "disease_name": "Other disease",
              "matching_phenotypes": [], "score": -0.56}
    first.update(supporting_genes=["SCN1A"], explanation="Because ...",
                 evidence_label="Strong path support",
                 evidence_package={"mode": "direct_path", "summary": "Direct 2-hop path",
                                   "evidence_label": "Strong path support",
                                   "min_path_length": 2,
                                   "direct_paths": [["hp:HP:0001250", "mondo:MONDO:0011073"]]})
    second.update(supporting_genes=[], explanation=None,
                  evidence_label="Insufficient evidence",
                  evidence_package={"mode": "insufficient", "summary": "No usable evidence",
                                    "evidence_label": "Insufficient evidence"})
    return [first, second]


def _sample_result():
    """A `disease_task/v1` result of a disease task model."""
    return {
        "result_kind": "disease_task/v1",
        "result_schema_version": 3,
        "session_id": "sess_abc",
        "patient_id": "pt_01",
        "timestamp": "2026-07-07T16:00:00Z",
        "served_identity": _served_identity(),
        "phenotype_input": {"received": 2, "unknown": [], "repeats_removed": 0,
                            "used": ["HP:0001250", "HP:0001263"]},
        "tie_policy": "score-desc-then-view-id-asc/v1",
        "inference_time_ms": 123.4,
        "warnings": ["low phenotype count"],
        "_query_phenotypes": ["HP:0001250", "HP:0001263"],
        "candidates": _candidates(),
    }


def test_csv_has_row_per_candidate_with_cleaned_ids():
    csv_text = dp._build_results_csv(_sample_result())
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("result_kind,rank,disease_id,disease_name,score")
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


def test_the_report_names_the_format_its_response_says_it_has():
    """The report is the kept copy of a response, so it names the response's
    format version (K8-4r, review P3-2: K8-4 changed the served identity's
    shape, and the version now marks it). The value is read as given, by no
    version rule: a response of the format before still reads, and says so. A
    response that names no format, or names it with something that is not a
    number, gets no format line. Catches a report that cannot tell which shape
    the response it keeps had."""
    line = "- **Result format**: result_schema_version {}"
    md = dp._build_results_report_md(_sample_result())
    assert line.format(3) in md
    assert md.index("- **Result kind**") < md.index(line.format(3)) < md.index(
        "- **Candidates**")
    assert line.format(2) in dp._build_results_report_md(
        {**_sample_result(), "result_schema_version": 2})
    for unnamed in (None, True, "3"):
        assert "Result format" not in dp._build_results_report_md(
            {**_sample_result(), "result_schema_version": unnamed}), unnamed
    result = _sample_result()
    del result["result_schema_version"]
    assert "Result format" not in dp._build_results_report_md(result)


def test_the_score_is_named_for_what_it_is_and_the_csv_names_the_kinds_fields():
    """The disease head's score is a ranking score, not a calibrated
    probability: the page and the report say "Score" and say so; the CSV names
    each kind's own fields (Spec 1 §2.1), never the removed confidence_score
    or the retired mixture's terms."""
    result = _sample_result()
    md = dp._build_results_report_md(result)
    table = md.split("## Ranked candidates")[1].split("## Per-candidate detail")[0]
    assert "| Rank | Disease | Score | Evidence |" in table
    assert dp.SCORE_NOTE in table and "not a calibrated disease probability" in dp.SCORE_NOTE
    assert "· score -0.280" in md and "· confidence" not in md
    page = dp.results_body_html(result, 1) + dp.details_header_html(
        result["candidates"][0], "disease_task/v1")
    assert "Confidence" not in page and "> Score <b>-0.280</b>" in page
    assert 'data-help="dx-score"' in page and "dx-score" in dp.TOPIC_HELP
    header = dp._build_results_csv(result).splitlines()[0]
    assert header.startswith("result_kind,rank,disease_id,disease_name,score,evidence_label")
    assert "confidence" not in header and "gnn_score" not in header
    assert "served_score" not in header and "sp_term" not in header

def test_write_exports_reuses_single_dir():
    p1, _ = dp._write_exports(_sample_result())
    p2, _ = dp._write_exports(_sample_result())
    assert os.path.dirname(p1) == os.path.dirname(p2)


def test_export_basename_falls_back_when_ids_missing():
    base = dp._export_basename({"candidates": []})
    assert base == "diagnosis_webui_patient_run"


# ------------------------------------------------------------------- the page's handlers
#: The lifecycle's fields as the status route answers them
#: (`PipelineStatusResponse`, `AppState.snapshot`): a model serving, and
#: nothing loaded with the service's reason.
_SERVING = {"serving": "serving", "loading": False, "in_flight": 0, "holds_gpu_lock": False,
            "not_serving_reason": None}
_NOT_LOADED = {"serving": "not_loaded", "loading": False, "in_flight": 0, "holds_gpu_lock": False,
               "not_serving_reason": "No model is loaded, so no diagnosis can be produced."}


def _no_model_status():
    return {"initialized": False, **_NOT_LOADED}


def _loaded_status(**extra):
    """A status serving `_served_identity()` (the records tests use it too)."""
    status = {
        "initialized": True, "gnn_ready": True,
        "scoring_mode": "disease_task", "kg_nodes": 10, "kg_edges": 20,
        "served_identity": _served_identity(), **_SERVING,
    }
    status.update(extra)
    return status


def _status_serving(identity, **extra):
    status = {"initialized": True, "gnn_ready": True, "served_identity": identity,
              "checkpoint_meta": {"epoch": 5}, **_SERVING}
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


def test_after_a_stop_the_card_points_back_to_research(monkeypatch):
    """B4 (2): after a stop the card's "Back to research ›" (its state
    Stopped, from the backend) opens Research › Training, as the header's
    Research switch would; a click on the card for anything else navigates
    nowhere."""
    demo, ctx = _build_tab(monkeypatch)
    body = next(b for b in demo.blocks.values()
                if getattr(b, "elem_id", None) == "shep-dx-card-body")
    (handler,) = _named(demo, "_to_research")
    assert [t[0] for t in handler.targets] == [body._id]
    assert list(handler.outputs) == list(ctx.nav_outputs)

    class Event:
        def __init__(self, key):
            self._data = {"key": key}

    nav, research_nav, page = handler.fn(Event("research"))
    assert (nav.selected, research_nav.selected, page) == ("research", "training", "training")
    assert handler.fn(Event("refresh")) == ctx.no_nav()


def test_the_load_drawer_says_the_chosen_workspace_once_by_the_shared_line(monkeypatch):
    """B4 (b): the drawer's workspace select has the identity line Training,
    Fine-tune, Models and Test show, the shared list's one line
    (`model_list.workspace_line_html`), following the listing whichever
    event read it."""
    from src.webui.components import model_list as ml

    demo, _ctx = _build_tab(monkeypatch)
    line = next(b for b in demo.blocks.values()
                if getattr(b, "elem_id", None) == "shep-dx-change-ws")
    (follows,) = [fn for fn in demo.fns.values()
                  if fn.fn is ml.workspace_line_html and line in fn.outputs]
    assert [t[1] for t in follows.targets] == ["change"]
    title = next(b for b in demo.blocks.values()
                 if getattr(b, "elem_id", None) == "shep-dx-change-title")
    assert "<h2>Load for inference</h2>" in title.value


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


def test_the_stop_questions_buttons_keep_their_words_width():
    """B4 browser check at 1850x960: Gradio gives a scale-0 button a 0% flex
    basis, so the stop question's two buttons were 12 px wide, "Keep serving"
    drawn over "Stop inference now", and a click on one hit the other. Each
    keeps its words' width."""
    css = " ".join(dp.PAGE_CSS.split())
    assert (f"{dp._TAB} .shep-dx-stop-row > button "
            "{ flex: 0 0 auto !important; width: auto !important; }") in css


def test_the_sticky_card_and_run_row_never_cover_a_focused_control(monkeypatch):
    """K6 closing smoke at 1536x760: the sticky Model-in-use card and Run row
    (bottom 0 inside the input column's scroll region) covered a focused
    Reference cohort select, Patient ID, Top-K or Explanations (the disease
    model's Explanations too). The scroll region keeps a scroll padding as tall
    as the sticky block, which a ResizeObserver writes as a CSS variable (as the
    shell writes ``--shep-chips-h``), so focusing a control scrolls it above
    the block. The browser re-measure is the integrator's."""
    css = " ".join(dp.PAGE_CSS.split())
    wide = css.index("@media (min-width: 1400px) and (min-height: 760px) {")
    narrow = css.index("@media (max-width: 1399px), (max-height: 759px) {")
    rule = (f"{dp._TAB} #shep-dx-input-bd {{ display: flex; flex-direction: column; "
            "scroll-padding-bottom: var(--shep-dx-sticky-h, 0px); }")
    assert wide < css.index(rule) < narrow
    assert f"{dp._TAB} .shep-dx-sticky {{ position: sticky; bottom: 0;" in css
    js = dp.STICKY_JS
    assert "element.closest('#shep-dx-sticky')" in js
    assert "sticky.closest('#shep-dx-input-bd')" in js
    assert "body.style.setProperty('--shep-dx-sticky-h', sticky.offsetHeight + 'px')" in js
    assert "new ResizeObserver(fit).observe(sticky)" in js
    # Wired on a component inside the sticky block, so its observer finds it.
    demo, _ctx = _build_tab(monkeypatch)
    (host,) = [b for b in demo.blocks.values() if getattr(b, "js_on_load", None) == js]
    ancestors, block = [], host
    while block is not None:
        ancestors.append(getattr(block, "elem_id", None))
        block = getattr(block, "parent", None)
    assert "shep-dx-sticky" in ancestors and "shep-dx-input-bd" in ancestors, ancestors


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
def served(monkeypatch, status_read, disease_pipeline):
    """The panel's API calls reach the real app, which serves a disease child
    trained for real (since K7 nothing is served without a model). Returns the
    payloads sent and the pipeline runs."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api import main as api_main

    pipeline = disease_pipeline
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
    _response(422, {"error": 5}),
    _response(422, {"error": "   "}),
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

#: The candidate columns of each kind (Spec 1 §2.1), in order; the identity is
#: appended after them.
_KIND_CSV_COLUMNS = {
    "disease_task/v1": [
        "result_kind", "rank", "disease_id", "disease_name", "score",
        "evidence_label", "evidence_mode", "evidence_summary", "min_path_length",
        "num_direct_paths", "supporting_genes", "matching_phenotypes"],
    # A task kind's own columns, then what the result carries for every row (K6).
    "gene_task/v1": [
        "result_kind", "rank", "gene_id", "gene_name", "score", "embsim", "sp_similarity",
        "mean_distance", "sp_bounds_lower", "sp_bounds_upper"],
    "patients_like_me/v1": [
        "result_kind", "rank", "case_id", "similarity", "disease_id", "disease_name",
        "causal_genes", "phenotypes", "reference_cohort_label",
        "reference_cohort_version_sha256", "reference_cohort_samples_sha256",
        "reference_cohort_patients"],
}
_SERVED_COLUMNS = [
    "served_scoring_mode", "served_checkpoint_state", "served_checkpoint_sha256",
    "served_kg_state", "served_kg_sha256", "served_split_manifest_sha256",
    "served_sp_state",
    "served_dataset_record_state", "served_dataset_origin",
    "served_dataset_record_sha256", "served_device", "served_device_requested",
    "served_eta", "served_max_path_length", "served_path_length_penalty",
    "served_max_paths_per_candidate",
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
#: The qualifier under the boundary when nothing is marked ✓.
_NOTHING_CHECKED = (
    "No relation is marked ✓ for this result: nothing in it was checked against a record."
)
#: The same said of the loaded pipeline, under the model status.
_NOTHING_CHECKED_LOADED = (
    "No relation is marked ✓ for the loaded pipeline: nothing it serves was "
    "checked against a record."
)
_CHECKS_NOT_REPORTED = "The response did not say what produced it or what was checked."


def _on_page(sentence: str) -> str:
    """A boundary sentence as the identity drawer shows it: the report's words,
    its ✓ drawn as the page's verified mark (the mark the drawer shows)."""
    return dp._e(sentence).replace("✓", r.status_mark("ok", "verified"))


_IDENTITIES = {
    "verified_synthetic": _served_identity,
    "verified_files": lambda: _served_identity(origin="files"),
    "gene_task": lambda: _task_identity("gene_task"),
}


def _result_with(identity):
    """A result served with `identity`, of the kind its scoring mode names
    (`result_kind_for`): a gene task model's genes, a disease result otherwise."""
    result = _gene_result() if identity.get("scoring_mode") == "gene_task" else _sample_result()
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
    assert "absent — no table is served; a disease task model scores without one" in section
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


def test_an_absent_table_is_written_as_its_state_and_reason():
    """A disease task model uses no shortest-path term and mixes nothing: the
    rows say so and why, and every other part is the model's, marked ✓ (K7: no
    absent checkpoint and no unverified graph is served)."""
    rows = {row[0]: row for row in _table_rows(dp._build_results_report_md(
        _result_with(_served_identity("files"))))}

    table = rows["Shortest-path table"]
    assert table[1] == "absent"
    assert table[2] == "absent — no table is served; a disease task model scores without one"
    assert rows["Checkpoint"][1] == "✓ verified"
    assert rows["Knowledge graph (`kg.json`)"][1] == "✓ verified"
    assert rows["Mixing weight eta"][1:] == [
        "none", "none — a disease task model has no terms to mix"]


def test_verified_states_carry_the_mark_the_claim_boundary_refers_to():
    rows = _table_rows(dp._build_results_report_md(_sample_result()))
    states = {row[0]: row[1] for row in rows}

    assert states["Checkpoint"] == "✓ verified"
    assert states["Shortest-path table"] == "absent"
    # Never "verified" alone for a synthetic record (contract M3b).
    assert states["Dataset record (`kg.provenance.json`)"] == "✓ verified · synthetic"
    assert dp.IDENTITY_NOTHING_CHECKED not in dp._build_results_report_md(_sample_result())


def test_with_nothing_marked_the_report_says_nothing_was_checked():
    section = _report_section(dp._build_results_report_md(_result_with(_parts_left_out())))

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
    identity["scoring_mode"] = "a|b`c*d<script>"
    identity["settings"]["device_requested"] = "x|y\nz"

    rows = _table_rows(_report_section(dp._build_results_report_md(_result_with(identity))))

    assert len(rows) == len(dp._IDENTITY_PARTS)
    assert all(len(row) == 3 for row in rows)
    assert "``a\\|b`c*d<script>``" in rows[5][2]
    # The drawer's "This result" (B4: a summary list from the same cells)
    # draws the value as text in its code, never as markup.
    shown = dp.result_identity_html(_result_with(identity), dp.card.new_state())
    assert "<code>a|b`c*d&lt;script&gt;</code>" in shown and "<script>" not in shown
    assert "<code>x|y z</code>" in shown


def test_the_report_and_the_drawer_draw_the_same_identity_pieces_each_in_its_own_form():
    """B4 review: the identity drawer reads no Markdown back. The one builder
    (`_identity_rows`) gives each value as pieces (words, code, a digest) and
    the report and the drawer draw them, each in its own form; so a value
    from the API in a text cell with backticks or escapes reads as written
    on the page (gr.Markdown did so before B4), and the report keeps its
    escapes."""
    identity = _served_identity()
    identity["settings"]["eta"] = "a`b`c\\_*"
    identity["dataset_record"]["origin"] = "s`y|n<t>"
    identity["settings"]["phenotype_normalisation_version"] = "v[1]#!"
    rows = {part: (state, value) for part, state, value in dp._identity_rows("served", identity)}
    eta = rows["Mixing weight eta"][1]
    assert eta == (dp._Piece("text", "a`b`c\\_*"),)
    assert dp._value_md(eta) == "a\\`b\\`c\\\\\\_\\*"
    assert dp._value_html(eta) == "a`b`c\\_*"
    record = dp._value_html(rows["Dataset record (`kg.provenance.json`)"][1])
    assert record.endswith(" — origin s`y|n&lt;t&gt;: not described by this page")
    assert f'title="{RECORD_SHA}">{RECORD_SHA[:12]}</code>' in record
    assert dp._value_html(rows["Phenotype normalisation rule"][1]) == "version v[1]#!"
    shown = dp.result_identity_html(_result_with(identity), dp.card.new_state())
    assert "a`b`c\\_*" in shown and "<code>b" not in shown and "&lt;t&gt;" in shown
    # The report's table cells are the same pieces in Markdown.
    table = _table_rows(_report_section(dp._build_results_report_md(_result_with(identity))))
    assert table[7][2] == "a\\`b\\`c\\\\\\_\\*"
    assert table[4][2].endswith("— origin s\\`y\\|n\\<t\\>: not described by this page")
    # Nothing on the page parses a Markdown cell any more.
    assert not hasattr(dp, "_md_cell_html")


# ------------------------------------------------------------------- the CSV
def test_csv_keeps_one_header_and_appends_the_identity_after_the_candidates():
    header, records, rows = _csv_records(dp._build_results_csv(_sample_result()))

    width = len(_KIND_CSV_COLUMNS["disease_task/v1"])
    assert header[:width] == _KIND_CSV_COLUMNS["disease_task/v1"]
    assert header[width:] == _SERVED_COLUMNS
    assert len(records) == 2 and all(len(row) == len(header) for row in rows)
    assert header not in rows[1:]  # exactly one header row, no preamble
    # The identity block by its own names: `score` is a candidate column.
    served = [{k: r[k] for k in _SERVED_COLUMNS} for r in records]
    assert served[0] == served[1]  # the same identity on every row
    assert served[0] == {
        "served_scoring_mode": "disease_task",
        "served_checkpoint_state": "verified",
        "served_checkpoint_sha256": f"sha256:{CHECKPOINT_SHA}",
        "served_kg_state": "verified",
        "served_kg_sha256": f"sha256:{KG_SHA}",
        "served_split_manifest_sha256": f"sha256:{MANIFEST_SHA}",
        "served_sp_state": "absent",
        "served_dataset_record_state": "verified",
        "served_dataset_origin": "synthetic",
        "served_dataset_record_sha256": f"sha256:{RECORD_SHA}",
        "served_device": "cuda:0",
        "served_device_requested": "auto",
        "served_eta": "none",
        "served_max_path_length": "4",
        "served_path_length_penalty": "0.9",
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
            assert _re.fullmatch(r"sha256:[0-9a-f]{64}", value), f"{name}: {column} = {value!r}"
        assert "unknown" not in ",".join(record.values()).lower()


@pytest.mark.parametrize("how", ["missing", "null"])
def test_a_response_without_an_identity_says_not_reported_in_every_column(how):
    """A null identity is not a demo answer (K7, DEMO1: there is none): it is
    a response that did not say what produced it."""
    result = _sample_result()
    if how == "missing":
        del result["served_identity"]
    else:
        result["served_identity"] = None

    _, records, _ = _csv_records(dp._build_results_csv(result))

    assert {record[c] for record in records for c in _SERVED_COLUMNS} == {"not_reported"}


def test_absent_and_inapplicable_values_are_words():
    _, (disease, _), _ = _csv_records(dp._build_results_csv(
        _result_with(_served_identity("files"))))
    assert disease["served_sp_state"] == "absent"
    assert disease["served_eta"] == "none"
    assert disease["served_checkpoint_sha256"] == f"sha256:{CHECKPOINT_SHA}"


def test_a_text_cell_a_spreadsheet_would_run_is_defused():
    result = _sample_result()
    result["candidates"][0]["disease_name"] = '=HYPERLINK("http://x")'
    result["candidates"][0]["score"] = -0.25
    result["served_identity"]["settings"]["device_requested"] = "@sum"

    _, (first, _), _ = _csv_records(dp._build_results_csv(result))

    assert first["disease_name"] == '\'=HYPERLINK("http://x")'
    assert first["served_device_requested"] == "'@sum"
    assert first["score"] == "-0.2500"  # numbers are not text cells


def test_export_names_carry_synthetic():
    assert dp._export_basename(_sample_result()) == "diagnosis_SYNTHETIC_pt_01_sess_abc"
    assert dp._export_basename(_result_with(_served_identity("files"))) == (
        "diagnosis_pt_01_sess_abc")
    assert dp._export_basename(_without_an_identity("missing")) == "diagnosis_pt_01_sess_abc"


# ------------------------------------------------------------------- the page
def test_the_page_shows_what_produced_the_result(monkeypatch, status_read):
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: _sample_result())

    out = _run("HP:0001250")

    body = out["results_body"]
    # The banner, above the table; its detail behind the shared ⓘ.
    assert body.index("SYNTHETIC WORKSPACE: not a clinical result.") < body.index("<table")
    assert 'aria-describedby="shep-help-text-dx-banner-synthetic"' in body
    # "This result" (the identity drawer): the result's own rows, as the
    # report reads them, one summary list (B4: B0c's details), a part a row,
    # a verified part by the shared mark, a digest by its 12 hex (the whole
    # in its title; "This result in full" has its Copy).
    shown = dp.result_identity_html(out["results_state"], out["card_state"])
    for part, state, value in dp._identity_rows("served", _served_identity()):
        assert (f'<dt class="shep-k">{dp._e(dp._part_key(part))}</dt>'
                f'<dd class="shep-v">{dp._part_html(state, value)}</dd>') in shown
    assert f'title="{CHECKPOINT_SHA}">{CHECKPOINT_SHA[:12]}</code>' in shown
    # No ✓ anywhere on the page: the boundary names the mark the drawer shows.
    assert r.status_mark("ok", "verified") in shown and "✓" not in shown
    assert _on_page(_CLAIM_BOUNDARY) in shown and _NOT_A_DEVICE in shown
    # This fixture's status serves nothing: compared with nothing, never as
    # ten parts that differ (B4), the result keeping its own identity.
    assert "<b>Not compared:</b> Nothing is served now" in shown and "<table" not in shown
    digests = dp.result_digests_html(out["results_state"])
    for digest in (CHECKPOINT_SHA, KG_SHA, MANIFEST_SHA, RECORD_SHA):
        assert f'data-copy="{digest}"' in digests
    assert digests.count('<dt class="shep-k shep-kv-wide">') == 4


def test_the_page_does_not_say_nothing_was_checked_without_an_identity(monkeypatch, status_read):
    """Like the report, "This result" says the response did not say, under the
    same claim boundary."""
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: _without_an_identity("missing"))

    out = _run("HP:0001250")
    shown = dp.result_identity_html(out["results_state"], out["card_state"])

    assert "nothing in it was checked" not in shown
    assert shown.index(_on_page(_CLAIM_BOUNDARY)) < shown.index(_CHECKS_NOT_REPORTED)
    assert "did not say what produced it" in out["badge"]


def test_a_result_with_nothing_marked_says_nothing_was_checked(monkeypatch, status_read):
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: _result_with(_parts_left_out()))

    report = _report_section(dp._build_results_report_md(_result_with(_parts_left_out())))
    out = _run("HP:0001250")
    shown = dp.result_identity_html(out["results_state"], out["card_state"])

    assert dp.IDENTITY_NOTHING_CHECKED == _NOTHING_CHECKED
    for text, said in ((report, str), (shown, _on_page)):
        assert text.index(said(_CLAIM_BOUNDARY)) < text.index(said(_NOTHING_CHECKED))
        assert _CHECKS_NOT_REPORTED not in text


def test_the_page_shows_the_served_pipelines_own_identity(served):
    """End to end through the real route: a disease child read from its
    checkpoint says so beside its result, and in its export, part by part."""
    out = _run(f"{SEIZURE}\n{DELAY}")

    result = out["results_state"]
    identity = result["served_identity"]
    assert result["result_kind"] == "disease_task/v1"
    assert identity["scoring_mode"] == "disease_task"
    assert identity["checkpoint"]["state"] == "verified"
    shown = dp.result_identity_html(result, out["card_state"])
    assert f">{identity['checkpoint']['sha256'][:12]}</code>" in shown
    assert "a model finetuned for the disease task" in shown
    _, records, _ = _csv_records(dp._build_results_csv(result))
    assert {r["served_checkpoint_state"] for r in records} == {"verified"}
    assert {r["served_kg_state"] for r in records} == {"verified"}
    assert {r["served_checkpoint_sha256"] for r in records} == {
        f"sha256:{identity['checkpoint']['sha256']}"}


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
    # Parts of the served identity differ (another task's model): never equal.
    other = _served_identity()
    other["shortest_paths"] = {"state": "computed"}
    other["scoring_mode"] = "gene_task"
    other["settings"] = _settings(eta=0.5)
    compared = dp.compare_with_served(_produced(), _card(_status_serving(other)))
    assert compared["state"] == "differs"
    assert compared["differ"] == ["Shortest-path table", "Scoring mode", "Mixing weight eta",
                                  "Path settings"]
    badge = dp.badge_html(_produced(), _card(_status_serving(other)))
    assert "Produced by an earlier model or pipeline" in badge and "4 parts differ" in badge


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
    shown = dp.result_identity_html(_produced(), later)
    assert "1 of 10 parts differs" in shown and "status read 16:31:12" in shown
    # B4: the parts that differ only, side by side (the skill's §4), then the
    # result's own identity: the model it used, never relabelled by the load.
    table = shown[shown.index("<table"):shown.index("</table>")]
    assert table.count('<th scope="row">') == 1 and '<th scope="row">Checkpoint</th>' in table
    assert f'>{CHECKPOINT_SHA[:12]}</code>' in table and ">bbbbbbbbbbbb</code>" in table
    own = shown[shown.index("What produced this result"):]
    assert f'>{CHECKPOINT_SHA[:12]}</code>' in own and "bbbbbbbbbbbb" not in own


# -------------------------------------------------- Run Diagnosis
def test_run_is_greyed_with_a_bubble_and_a_reason_when_no_model_is_loaded():
    from src.webui.components.help import text_id

    run = dp.run_controls(_card(_no_model_status()), "idle")
    assert run["run_btn"]["interactive"] is False
    assert f'aria-describedby="{text_id("dx-run-no-model")}"' in run["run_cover"]
    assert 'class="shep-i shep-i-cover"' in run["run_cover"]
    assert dp.RUN_DISABLED_NO_MODEL in run["run_reason"]
    # B4: one reason, "Disabled until a model is loaded", whoever reads it:
    # the card says why, and who can load one (its gated action).
    assert dp.RUN_DISABLED_NO_MODEL == "Disabled until a model is loaded."
    read_only = dp.run_controls(_card(_no_model_status(), operator=False), "idle")
    assert dp.RUN_DISABLED_NO_MODEL in read_only["run_reason"]
    assert "dx-run-no-model" in dp.TOPIC_HELP


def test_run_stays_on_for_a_served_model_and_an_unreadable_status_but_not_a_model_change():
    served = _status_serving(_served_identity())
    assert dp.run_controls(_card(served), "idle")["run_btn"]["interactive"] is True
    assert dp.run_controls(_card(None, "timed out (5 s)"), "idle")["run_cover"] == ""
    change = dp.run_controls(_card(_status_serving(_served_identity())), "confirming")
    assert change["run_btn"]["interactive"] is False
    assert dp.RUN_DISABLED_CHANGE in change["run_reason"] and "dx-run-change" in change["run_cover"]
    # While this page's Load reloads, the reason is the load's (the mockup's state 3).
    loading = dp.run_controls(_card(_status_serving(_served_identity())), "loading")
    assert loading["run_btn"]["interactive"] is False
    assert dp.RUN_DISABLED_LOADING in loading["run_reason"]
    assert "dx-run-loading" in loading["run_cover"] and "dx-run-loading" in dp.TOPIC_HELP


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
    assert seen == [False] and "Not loaded" in out["card_body"]


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
    """The reload cannot be stopped and its outcome must be seen: a close that
    still finds the phase "loading" (one the page did not hold until the
    answer, `CLOSE_WHILE_LOADING`) keeps the drawer, the plan and the phase."""
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


def test_this_pages_load_is_said_by_the_server_and_run_waits_for_its_answer(monkeypatch):
    """B4 (README §6, the note from A3b's review): the page's own "Loading
    <12 hex> for every session…" line folds into the server's Loading state,
    said once (`diagnosis_card.serving_state`, `loading_too`): the page draws
    no line of its own. Run Diagnosis is off with the load's reason in the
    step before the reload, and taken back after its answer, whatever it
    was."""
    plan = {"to": {"sha256": "e" * 64}}
    state = _card(_status_serving(_served_identity()))
    run_btn, cover, reason = dp.card_loading_outputs(plan, state)
    assert run_btn["interactive"] is False
    assert "dx-run-loading" in cover and dp.RUN_DISABLED_LOADING in reason
    assert dp.card_loading_outputs(None, state) == (gr.update(),) * 3
    run_btn, cover, reason = dp.card_loaded_outputs(state, "idle")
    assert run_btn["interactive"] is True and cover == "" and reason == ""
    assert not hasattr(dp, "loading_line_html")
    # The page as built: no line of its own; Run rides on the list's step
    # before the reload and is taken back by a step after its answer.
    demo, _ctx = _build_tab(monkeypatch)
    assert not any(getattr(b, "elem_id", None) == "shep-dx-card-loading"
                   for b in demo.blocks.values())
    run_button = next(b for b in demo.blocks.values()
                      if getattr(b, "elem_id", None) == "shep-dx-run-btn")
    (loading,) = _named(demo, "_loading")
    assert run_button in loading.outputs
    (confirm,) = _named(demo, "_confirm")
    after = [fn for fn in demo.fns.values() if fn.trigger_after == confirm._id]
    answered = [fn for fn in after if fn.name == "_reload_answered"]
    assert len(answered) == 1 and run_button in answered[0].outputs
    # The server's load is the card's, beside Serving while a model serves.
    reloading = dp.card_outputs(_card({**_status_serving(_served_identity()), "loading": True}),
                                None, "loading")
    assert reloading["card_body"].count("Loading") == 1
    assert "for every session" not in reloading["card_body"]


def test_the_compare_link_keeps_each_arrow_with_its_word(monkeypatch):
    """M5-3: at 1440 x 900 "Compare in Research › Models ›" left its last "›"
    alone on a line; each is held to the word before it."""
    assert dp.COMPARE_LINK == "Compare in Research\u00a0› Models\u00a0›"
    demo, _ctx = _build_tab(monkeypatch)
    link = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "shep-dx-compare")
    assert dp.COMPARE_LINK in link.value and "Models ›" not in link.value


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


def test_while_a_reload_runs_the_drawer_closes_at_once_and_its_close_follows_the_answer(
        monkeypatch):
    """M5-3 review: Close and × did nothing while the reload held the server,
    then closed the drawer after it in no fixed order with its answer. Close,
    × and Cancel now hide the whole drawer at once in the page (the card keeps
    its "Loading …" line); × is clicked for the server after the answer;
    "Change model…" shows the drawer again meanwhile; and each close ends
    with the list's page step that drops the page's own hiding."""
    from src.webui.components import model_list as ml
    from src.webui.components.help import as_json_for_script

    demo, _ctx = _build_tab(monkeypatch)
    guard = ml.load_guard("shep-models-clinical", dp.CLOSE_WHILE_LOADING)
    assert guard["hide"] == "#shep-dx-change" and guard["then"] == "#shep-dx-change-close button"
    assert guard["closers"] == ["#shep-models-clinical-keep", "#shep-dx-change-close button",
                                "#shep-dx-change-cancel"]
    assert guard["openers"] == ["#shep-dx-change-btn"]
    (loading,) = _named(demo, "_loading")
    assert as_json_for_script(guard) in loading.js
    blocks = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    assert {"shep-dx-change", "shep-dx-change-cancel", "shep-dx-change-btn",
            "shep-models-clinical-keep"} <= set(blocks)
    assert '<button type="button" class="shep-dx-close" data-key="close"' in \
        blocks["shep-dx-change-close"].value
    for name in ("_close_change_drawer", "_cancel_change"):
        (closing,) = _named(demo, name)
        reset = next(fn for fn in demo.fns.values() if fn.trigger_after == closing._id)
        done = next(fn for fn in demo.fns.values() if fn.trigger_after == reset._id)
        assert done.fn is None and done.js == ml.load_closed_js("shep-models-clinical"), name


# -------------------------------------------------- the candidates and the details
def test_values_from_the_api_are_escaped_in_the_table_and_the_details():
    result = _sample_result()
    evil = "<img src=x onerror=alert(1)>"
    result["candidates"][0].update(disease_name=evil, supporting_genes=[evil],
                                   matching_phenotypes=[evil])
    result["candidates"][0]["evidence_package"].update(summary=evil, direct_paths=[[evil, "b"]])
    html = (dp.results_body_html(result, 1)
            + dp.details_header_html(result["candidates"][0], "disease_task/v1")
            + dp.evidence_html(result["candidates"][0]) + dp.paths_html(result["candidates"][0])
            + dp.phenotypes_html(result["candidates"][0]))
    assert "<img" not in html and "&lt;img src=x onerror=alert(1)&gt;" in html


def test_a_response_that_does_not_name_its_kind_shows_no_candidates(monkeypatch, status_read):
    result = _sample_result()
    del result["result_kind"]
    assert dp.KIND_NOT_REPORTED in dp.results_body_html(result, 1)
    assert "<table" not in dp.results_body_html(result, 1)
    _, records, _ = _csv_records(dp._build_results_csv(result))
    assert records == []
    report = dp._build_results_report_md(result)
    assert report.count(dp.KIND_NOT_REPORTED) == 1 and ".." not in report
    # The page selects nothing and exports nothing for the candidates it hides.
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: dict(result))
    out = _run("HP:0001250")
    assert out["selected"] is None
    assert out["export_csv"] is dp._DOWNLOAD_DISABLED
    assert out["export_report"] is dp._DOWNLOAD_DISABLED


@pytest.mark.parametrize("kind", ["gnn_ranking/v1", "path_reasoning/v1", "demo/v1"])
def test_a_kind_the_page_does_not_know_shows_no_candidates(kind):
    """A kind this page does not know (a newer API, or one retired in K7 that
    an old server may still send) is treated like an absent kind on every
    surface: the notice, no table, no CSV row; the report keeps the raw value
    it was given."""
    result = _sample_result()
    result["result_kind"] = kind
    body = dp.results_body_html(result, 1)
    assert dp.KIND_NOT_REPORTED in body and "<table" not in body
    assert dp.kind_label(result) == dp.KIND_NOT_REPORTED
    _, records, _ = _csv_records(dp._build_results_csv(result))
    assert records == []
    report = dp._build_results_report_md(result)
    assert report.count(dp.KIND_NOT_REPORTED) == 1 and "### #1" not in report
    assert f"`{result['result_kind']}` (not readable here)" in report


def test_the_score_help_stays_within_its_budget():
    """The skill's §7 budget (about 60-80 words) and §5's sentence limit."""
    text = dp.TOPIC_HELP["dx-score"][1]
    assert len(text.split()) <= 80
    assert max(len(s.split()) for s in re.split(r"(?<=[.!?])\s+", text)) <= 25


def test_a_missing_count_or_time_is_never_written_as_zero():
    """No evidence package (explanations off): the CSV's path counts are empty,
    not 0; a missing inference time is left out of the report, not 0.0."""
    result = _sample_result()
    for candidate in result["candidates"]:
        candidate["evidence_package"] = None
    del result["inference_time_ms"]
    _, records, _ = _csv_records(dp._build_results_csv(result))
    assert {r["num_direct_paths"] for r in records} == {""}
    assert "Inference time" not in dp._build_results_report_md(result)

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
        assert _shown(out["id_result_html"]) is (seg == "result")
        assert "id_result_md" not in out
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
    for elem_id, name in (("shep-dx-id-model", "id_model"),
                          ("shep-dx-id-result", "id_result_html")):
        assert list(by_id[elem_id].elem_classes) == model[name]["elem_classes"], elem_id
    for block in demo.blocks.values():
        if getattr(block, "elem_id", "") in ("shep-dx-seg", "shep-dx-details-body",
                                              "shep-dx-explanation", "shep-dx-id-model",
                                              "shep-dx-id-result"):
            assert block.visible is True


# -------------------------------------------------- the model in use (identity drawer)
def test_the_model_in_use_shows_the_boundary_for_what_is_loaded():
    html = dp.model_identity_html(_card(_status_serving(_served_identity())))
    assert _on_page(_CLAIM_BOUNDARY) in html and _NOT_A_DEVICE in html
    assert _on_page(_NOTHING_CHECKED_LOADED) not in html and "SYNTHETIC WORKSPACE" in html
    # The sentence names the mark the served facts show a verified part by.
    assert "✓" not in html and r.status_mark("ok", "verified") in html.split("shep-dx-boundary")[0]


def test_a_model_in_use_with_nothing_marked_says_nothing_was_checked():
    html = dp.model_identity_html(_card(_status_serving(_parts_left_out())))
    assert html.index(_on_page(_CLAIM_BOUNDARY)) < html.index(_on_page(_NOTHING_CHECKED_LOADED))
    assert _on_page(_NOTHING_CHECKED) not in html and _CHECKS_NOT_REPORTED not in html


_STATUS_IDENTITIES = {**_IDENTITIES, "parts_left_out": _parts_left_out}


@pytest.mark.parametrize("name", sorted(_STATUS_IDENTITIES))
def test_the_model_in_use_and_the_report_agree_on_whether_anything_was_checked(name):
    """The drawer adds the qualifier exactly when the report's identity table
    does, for every served identity (one rule, `_claim_boundary_lines`)."""
    make = _STATUS_IDENTITIES[name]
    html = dp.model_identity_html(_card(_status_serving(make())))
    section = _report_section(dp._build_results_report_md(_result_with(make())))
    nothing_marked = "✓" not in "".join(row[1] for row in _table_rows(section))

    assert (_NOTHING_CHECKED in section) is nothing_marked
    assert (_on_page(_NOTHING_CHECKED_LOADED) in html) is nothing_marked
    assert _on_page(_CLAIM_BOUNDARY) in html and _CLAIM_BOUNDARY in section


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


# -------------------------------------------------- Diagnosis by task (K6 decision 15)
# A gene task model ranks the request's candidate genes; a patients-like-me
# model ranks a registered reference cohort version's patients. The page offers
# the served task's input, sends it only to its task, and draws both kinds'
# rows, details and exports escaped (README §8, "In scope for the UI").
VERSION_SHA = _digest("cohort.json")
SAMPLES_SHA = _digest("reference_samples.json")
_EVIL = "<img src=x onerror=alert(1)>"
_EVIL_SHOWN = "&lt;img src=x onerror=alert(1)&gt;"


def _task_identity(mode):
    """What the API sends for a finetuned bundle of `mode`: no table (the gene
    model's distances computed from its view), the gene model's own η."""
    identity = _served_identity(origin="files")
    identity["scoring_mode"] = mode
    identity["shortest_paths"] = {"state": "computed" if mode == "gene_task" else "absent"}
    identity["settings"] = _settings(eta=0.5 if mode == "gene_task" else None)
    return identity


def _version(label, binding="bound", problem=None):
    """A reference cohort version as the card lists it
    (`diagnosis_card.reference_cohorts`)."""
    words = {"bound": "bound to this graph",
             "another": "mapped against another graph (node_features differ)",
             "unknown": "the version cannot be read"}
    return {"label": label, "manifest_sha256": None if problem else VERSION_SHA,
            "problem": problem, "binding": "unknown" if problem else binding,
            "binding_words": words["unknown" if problem else binding]}


def _cohorts(*versions, problem=None):
    return {"versions": list(versions), "problem": problem}


def _task_card(mode, cohorts=None, *, operator=True):
    """The card's state serving a task model of `mode`, its facts holding
    `cohorts` (the reference cohort versions, for patients-like-me)."""
    from datetime import datetime

    return dp.card.advance(None, _status_serving(_task_identity(mode)), None,
                           now=datetime.fromisoformat(_NOW), operator=operator, d8_short=None,
                           open_checkpoints=False, facts=lambda s, **k: {
                               "key": dp.card.served_key(s), "operator": operator,
                               "workspaces": [], "serving": None, "item": None,
                               "item_problem": None, "recorded": None, "trained": None,
                               "cohorts": cohorts})


def _gene_result():
    """A `gene_task/v1` result as the API sends it (`GeneTaskResponse`)."""
    return {
        "result_kind": "gene_task/v1", "result_schema_version": 3, "session_id": "sess_g",
        "patient_id": "pt_g", "timestamp": "2026-07-07T16:00:00Z",
        "served_identity": _task_identity("gene_task"),
        "phenotype_input": {"received": 1, "unknown": [], "repeats_removed": 0,
                            "used": ["HP:0001250"]},
        "tie_policy": "score-desc-then-view-id-asc/v1", "sp_bounds": [1.0, 6.84],
        "inference_time_ms": 12.0, "summary": "Found 2 candidate genes",
        "warnings": ["2 of 3 candidate genes scored; 1 map to no gene"],
        "_query_phenotypes": ["HP:0001250"],
        "_query_candidate_genes": ["SCN1A", "disgenet:PTEN", "NOT_A_GENE"],
        "candidates": [
            {"rank": 1, "gene_id": "disgenet:SCN1A", "gene_name": "SCN1A", "score": 0.61,
             "embsim": 0.4, "sp_similarity": 0.82, "mean_distance": 1.5},
            {"rank": 2, "gene_id": "disgenet:PTEN", "gene_name": "PTEN", "score": -0.2,
             "embsim": -0.3, "sp_similarity": -0.1, "mean_distance": 4.0}],
    }


def _patients_result():
    """A `patients_like_me/v1` result (`PatientsLikeMeResponse`): a patient
    with a recorded diagnosis and causal gene, and one with neither."""
    return {
        "result_kind": "patients_like_me/v1", "result_schema_version": 3, "session_id": "sess_p",
        "patient_id": "pt_p", "timestamp": "2026-07-07T16:00:00Z",
        "served_identity": _task_identity("patients_like_me"),
        "phenotype_input": {"received": 1, "unknown": [], "repeats_removed": 0,
                            "used": ["HP:0001250"]},
        "tie_policy": "similarity-desc-then-case-id-asc/v1",
        "reference_cohort": {"label": "reference-1", "version_sha256": VERSION_SHA,
                             "samples_sha256": SAMPLES_SHA, "patients": 12},
        "inference_time_ms": 9.0, "summary": "Found 2 similar patients", "warnings": [],
        "_query_phenotypes": ["HP:0001250"],
        "candidates": [
            {"rank": 1, "case_id": "case-07", "similarity": -0.12,
             "phenotypes": ["HP:0001250", "HP:0001263"], "disease_id": "mondo:MONDO:0011073",
             "disease_name": "Dravet syndrome", "causal_genes": ["SCN1A"]},
            {"rank": 2, "case_id": "case-02", "similarity": -0.5, "phenotypes": ["HP:0002376"],
             "disease_id": None, "disease_name": None, "causal_genes": []}],
    }


@pytest.mark.parametrize("mode, genes, cohort, explanations", [
    ("gene_task", True, False, False), ("patients_like_me", False, True, False),
    ("disease_task", False, False, True)])
def test_the_inputs_follow_the_served_task(mode, genes, cohort, explanations):
    """1: Candidate genes for a gene task model, Reference cohort for a
    patients-like-me model, neither otherwise; Explanations for a disease
    task model only (B4 (3): each task asks only for its own inputs: a gene's
    or a reference patient's rows carry no explanation); each shown by its
    container's class (`view_update`), never `visible`."""
    state = _task_card(mode, _cohorts(_version("reference-1")))
    out = dp.card_outputs(state, None, "idle")
    assert _shown(out["genes_box"]) is genes and _shown(out["cohort_box"]) is cohort
    assert _shown(out["explanations_box"]) is explanations
    _assert_views_switch_by_class(out)
    assert ("cohort" in out) is cohort
    for state in (_card(_no_model_status()), _card(None, "timed out (5 s)")):
        out = dp.card_outputs(state, None, "idle")
        assert not _shown(out["genes_box"]) and not _shown(out["cohort_box"])
        assert _shown(out["explanations_box"])


def _sent(monkeypatch, result):
    """`_self_request` answered with `result`; returns the payloads sent."""
    sent = []

    class _Ok:
        def raise_for_status(self):
            return None

        def json(self):
            return dict(result)

    monkeypatch.setattr(dp, "_self_request",
                        lambda method, path, **kw: sent.append(kw["json"]) or _Ok())
    return sent


def test_each_task_gets_only_its_own_input(monkeypatch, status_read):
    """2: the gene list goes to a gene task model, split on whitespace, commas
    and semicolons with order, repeats and case kept; the reference cohort
    label to a patients-like-me model; neither to any other model."""
    genes = "SCN1A\ndisgenet:PTEN, ;scn1a  "
    sent = _sent(monkeypatch, _gene_result())
    out = dp.diagnose("HP:0001250", "", 10, True, _task_card("gene_task"), "idle", {}, None,
                      genes_text=genes, cohort="reference-1")
    assert sent[-1]["candidate_genes"] == ["SCN1A", "disgenet:PTEN", "scn1a"]
    assert "reference_cohort" not in sent[-1]
    # Explanations is a disease task model's own option (B4): never asked
    # of a gene task model, whatever the hidden box holds.
    assert sent[-1]["include_explanations"] is False
    assert out["results_state"]["_query_candidate_genes"] == ["SCN1A", "disgenet:PTEN", "scn1a"]
    sent = _sent(monkeypatch, _patients_result())
    dp.diagnose("HP:0001250", "", 10, True,
                _task_card("patients_like_me", _cohorts(_version("reference-1"))), "idle", {},
                None, genes_text=genes, cohort="reference-1")
    assert sent[-1]["reference_cohort"] == "reference-1" and "candidate_genes" not in sent[-1]
    assert sent[-1]["include_explanations"] is False
    for state in (_task_card("disease_task"), _card(_status_serving(_served_identity()))):
        sent = _sent(monkeypatch, _sample_result())
        dp.diagnose("HP:0001250", "", 10, True, state, "idle", {}, None, genes_text=genes,
                    cohort="reference-1")
        assert "candidate_genes" not in sent[-1] and "reference_cohort" not in sent[-1]
        assert sent[-1]["include_explanations"] is True


def test_an_empty_gene_list_is_said_at_the_click_and_nothing_is_asked(monkeypatch, status_read):
    """3: a gene task model with no candidate genes: a notice, no API call."""
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kw: pytest.fail("the API was called"))
    out = dp.diagnose("HP:0001250", "", 10, True, _task_card("gene_task"), "idle", {}, None,
                      genes_text=" ,; \n")
    assert dp.NO_CANDIDATE_GENES in out["results_body"] and out["results_state"] is None
    assert out["export_csv"] is dp._EXPORTS_HIDDEN and status_read["reads"] == []


def test_run_waits_for_a_readable_reference_cohort_version():
    """4: a patients-like-me model with a registry problem, no version or no
    readable version: Run off, with the reason visible in its own words per
    cause, saying who can act (skill §9), and its registered topic; the
    registry setting's own words (its name, a command's flag) are an
    operator's and never reach the reason, read-only or not (skill §5). A
    readable version turns Run on."""
    from src.config.server_settings import cohort_registry
    from src.webui.components.help import text_id

    unset = cohort_registry(environ={}).problem
    assert "SHEPHERD_REGISTRY" in unset and "--registry" in unset
    for cohorts, words in ((_cohorts(problem=unset), dp.NO_COHORT_REGISTRY),
                           (_cohorts(problem="the cohort registry could not be listed"),
                            dp.NO_COHORT_REGISTRY),
                           (_cohorts(), dp.NO_COHORT_VERSION),
                           (_cohorts(_version("torn-1", problem="<registry>/versions/torn-1 has "
                                                                 "no cohort.json")),
                            dp.NO_READABLE_VERSION),
                           (None, dp.NO_COHORT_VERSION)):
        for operator in (True, False):
            run = dp.run_controls(_task_card("patients_like_me", cohorts, operator=operator),
                                  "idle")
            assert run["run_btn"]["interactive"] is False
            assert run["run_reason"].endswith(
                f"{dp._e(f'{dp.RUN_DISABLED_NO_COHORT} {words}')}</p>")
            assert "operator" in words and "SHEPHERD_" not in run["run_reason"]
            assert "--registry" not in run["run_reason"] and "<registry>" not in run["run_reason"]
            assert f'aria-describedby="{text_id("dx-run-no-cohort")}"' in run["run_cover"]
    assert "dx-run-no-cohort" in dp.TOPIC_HELP
    # The cover's bubble explains this reason to whoever reads it (a clinician
    # too): the registry setting's own words stay off it (K6 closing review).
    help_text = " ".join(dp.TOPIC_HELP["dx-run-no-cohort"])
    assert "SHEPHERD_" not in help_text and "--registry" not in help_text
    assert "the server's cohort registry setting" in help_text
    # The empty select says what it holds, not that the user should pick.
    assert dp.cohort_info(_cohorts(problem=unset), None) == "No version is listed."
    ready = _task_card("patients_like_me", _cohorts(_version("other-1", "another"),
                                                    _version("reference-1")))
    assert dp.run_controls(ready, "idle")["run_btn"]["interactive"] is True
    # Another model is never gated on cohorts.
    assert dp.run_controls(_task_card("gene_task"), "idle")["run_btn"]["interactive"] is True
    assert dp.run_controls(_task_card("patients_like_me", _cohorts()), "confirming")[
        "run_reason"].count(dp.RUN_DISABLED_CHANGE) == 1


def _assert_options_tell_versions_apart(options):
    """Every option at most 40 characters, with at most one separator, and
    telling its version from every other listed one: read as a pattern (the
    mark dropped, each "…" standing for hidden characters) it fits its own
    label and no other; so the texts are unique too."""
    labels = [value for _text, value in options]
    for text, label in options:
        assert len(text) <= r.OPTION_CHARS and text.count(" · ") <= 1, text
        shown = text.split(" · ")[0]
        pattern = ".*".join(re.escape(piece) for piece in shown.split("…"))
        assert [other for other in labels if re.fullmatch(pattern, other)] == [label], text


def test_long_reference_cohort_labels_show_the_part_that_tells_them_apart():
    """5 (options): labels over 40 characters that share their start and end
    show where they differ (the date, the version token), bound or marked;
    a label told apart by its middle cut keeps it."""
    v1 = "hospital_a_rare_disease_reference_v1_cohort_2026_10_import"
    v2 = v1.replace("_v1_", "_v2_")
    first, second = (f"cohort_2026_10_{day}_hospital_a_rare_disease_reference_import"
                     for day in ("01", "15"))
    options, _chosen = dp.cohort_choices(_cohorts(_version(v1), _version(v2), _version(first),
                                                  _version(second, "another")), None)
    _assert_options_tell_versions_apart(options)
    texts = [text for text, _value in options]
    assert texts[:2] == ["…v1_cohort_2026_10_import", "…v2_cohort_2026_10_import"]
    assert texts[2] == "cohort_2026_10_01_ho…se_reference_import"
    assert texts[3] == "…15_hospital_a_rare_dis… · another graph"


def test_the_reference_cohort_options_are_short_unique_and_keep_the_full_label():
    """5: every option at most 40 characters, unique, with at most one
    separator; the values are the full labels; "another graph" and "cannot
    be read" marked; the default the first bound version, a current value
    kept while listed."""
    long_a = "a" * 30 + "-left-" + "z" * 28
    long_b = "a" * 30 + "-rght-" + "z" * 28
    assert len(long_a) == len(long_b) == 64
    cohorts = _cohorts(_version(long_a, "another"), _version(long_b, "another"),
                       _version("torn-1", problem="x"), _version("reference-1"),
                       _version("reference-2"))
    options, chosen = dp.cohort_choices(cohorts, None)
    texts, values = [t for t, _v in options], [v for _t, v in options]
    assert values == [long_a, long_b, "torn-1", "reference-1", "reference-2"]
    _assert_options_tell_versions_apart(options)
    assert texts[0].endswith(" · another graph") and "…" in texts[0]
    # Two long labels that differ only inside the middle cut: each option leads
    # with the token where it differs (skill §8: the distinguishing token first).
    assert texts[0] == "…left-" + "z" * 17 + "… · another graph"
    assert texts[1] == "…rght-" + "z" * 17 + "… · another graph"
    assert texts[2] == "torn-1 · cannot be read" and texts[3] == "reference-1"
    assert chosen == "reference-1"
    assert dp.cohort_choices(cohorts, "reference-2")[1] == "reference-2"
    assert dp.cohort_choices(cohorts, "gone")[1] == "reference-1"
    assert dp.cohort_choices(_cohorts(_version("other-1", "another")), None)[1] == "other-1"
    assert dp.cohort_choices(_cohorts(_version("torn-1", problem="x")), None)[1] is None
    info = dp.cohort_info(cohorts, "reference-1")
    assert info == f"reference-1: version {VERSION_SHA[:12]}, bound to this graph"
    assert dp.cohort_info(cohorts, "torn-1") == "torn-1: cannot be read: x"


def test_the_timer_never_redraws_the_reference_cohort_select_for_nothing():
    """5 (timer): a read with the same options sends no select update, and a
    user's pick (its `.input`) is not sent back by the next read."""
    cohorts = _cohorts(_version("reference-1"), _version("reference-2"))
    state = _task_card("patients_like_me", cohorts)
    first = dp.card_outputs(state, None, "idle", cohort=None)
    update = first["cohort"]
    assert update["value"] == "reference-1"
    assert update["choices"] == [("reference-1", "reference-1"), ("reference-2", "reference-2")]
    assert update["info"].startswith("reference-1: version ")
    again = dp.card_outputs(first["card_state"], None, "idle", cohort="reference-1")
    assert "cohort" not in again
    picked = dp.cohort_picked(None, again["card_state"], "reference-2")
    assert picked["cohort"]["info"].startswith("reference-2: version ")
    assert "value" not in picked["cohort"] and "results_state" not in picked
    # The pick writes no card state back over a read running beside it.
    assert "card_state" not in picked
    ticked = dp.card_outputs(again["card_state"], None, "idle", cohort="reference-2")
    assert "cohort" not in ticked and "genes_box" not in ticked
    # A pick after a result clears it, as edited phenotypes do.
    cleared = dp.cohort_picked(_patients_result(), again["card_state"], "reference-2")
    assert "Inputs changed" in cleared["results_body"] and cleared["results_state"] is None


def test_a_timer_read_racing_a_pick_never_redraws_the_reference_cohort_select():
    """5 (timer, a race): the slow timer runs outside the queue, so a read
    that started before a pick can write the card's state after the pick's
    event, and a read sent just after the pick can still get the state from
    before it. The select's cache holds the server's facts, never the user's
    pick: neither read sends the select again. Facts that change still do."""
    cohorts = _cohorts(_version("reference-1"), _version("reference-2"))
    before = dp.card_outputs(_task_card("patients_like_me", cohorts), None, "idle")["card_state"]
    dp.cohort_picked(None, before, "reference-2")
    late = dp.card_outputs(before, None, "idle", cohort="reference-1")
    assert "cohort" not in late
    assert "cohort" not in dp.card_outputs(late["card_state"], None, "idle", cohort="reference-2")
    assert "cohort" not in dp.card_outputs(before, None, "idle", cohort="reference-2")
    # The chosen version's facts change (now mapped against another graph).
    moved = _task_card("patients_like_me", _cohorts(_version("reference-1"),
                                                    _version("reference-2", "another")))
    update = dp.card_outputs({**moved, "rendered": late["card_state"]["rendered"]}, None, "idle",
                             cohort="reference-2")["cohort"]
    assert update["info"].startswith("reference-2: version ") and "another graph" in update["info"]
    assert "value" not in update


def test_the_task_inputs_are_never_run_outputs_and_clear_on_input(monkeypatch):
    """6: the genes box and the select are inputs of Run Diagnosis, never its
    outputs; the card's reads show and fill them; the user's edit (`.input`,
    never `.change`, which a card update fires) clears the result; Clear
    empties the genes box too."""
    demo, _ctx = _build_tab(monkeypatch)
    blocks = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    genes, cohort = blocks["dx_candidate_genes"], blocks["dx_reference_cohort"]
    genes_box, cohort_box = blocks["shep-dx-genes-box"], blocks["shep-dx-cohort-box"]
    assert genes.label == "Candidate genes" and cohort.label == "Reference cohort"
    assert cohort.info is None and dp.VIEW_OFF in genes_box.elem_classes
    (run,) = _named(demo, "_run_diagnosis")
    assert genes in run.inputs and cohort in run.inputs
    assert not {genes, cohort, genes_box, cohort_box} & set(run.outputs)
    for name in ("_card_refresh", "_card_poll", "_card_reread", "_card_link"):
        for fn in _named(demo, name):
            assert {genes_box, cohort_box, cohort} <= set(fn.outputs), name
            assert genes not in fn.outputs, name
    triggers = {(fn.name, t) for fn in demo.fns.values() for t in fn.targets}
    assert ("_phenotypes_edited", (genes._id, "input")) in triggers
    assert ("_cohort_picked", (cohort._id, "input")) in triggers
    assert not {t for _n, t in triggers if t in ((genes._id, "change"), (cohort._id, "change"))}
    clear = blocks["shep-dx-clear"]
    assert any(genes in fn.outputs for fn in demo.fns.values()
               if (clear._id, "click") in fn.targets)


def test_a_gene_result_shows_its_genes_and_their_values():
    """7: the kind's label, its columns and formatted numbers; no disease or
    evidence column; no segment; the details add the mean hop distance and the
    model's SP bounds."""
    result = _gene_result()
    body = dp.results_body_html(result, 1)
    assert "Gene task model: your candidate genes" in body
    for head in (">Gene</th>", ">Model similarity</th>", ">SP similarity</th>"):
        assert head in body
    assert 'data-help="dx-gene-score"' in body and ">0.610</td>" in body and ">0.820</td>" in body
    assert "Disease" not in body and "Evidence" not in body and "0.000" not in body
    assert "2 of 3 candidate genes scored" in body
    out = dp.details_outputs(result, 1)
    assert dp.VIEW_OFF in out["details_seg"]["elem_classes"]
    details = out["details_html"]["value"]
    assert "Mean hop distance" in details and ">1.500<" in details
    assert "1.000 to 6.840" in details
    assert "#1</span> SCN1A</h2>" in out["details_hd"]
    assert "<code>disgenet:SCN1A</code> Score <b>0.610</b>" in out["details_hd"]
    _assert_views_switch_by_class(out)
    assert dp.select_candidate(result, "evidence", "2")["selected"] == 2


def test_a_patients_like_me_result_shows_reference_patients_and_its_version():
    """8: research id, rank, similarity, recorded diagnosis and causal genes,
    "not recorded" where nothing was; the reference cohort label and 12-hex
    over the table, never the full digest there; the details list the recorded
    HPO terms as context; the full digests to copy in "This result in full"."""
    result = _patients_result()
    body = dp.results_body_html(result, 1)
    for head in (">Research ID</th>", ">Diagnosis</th>", ">Causal genes</th>"):
        assert head in body
    assert 'data-help="dx-similarity"' in body and ">-0.120</td>" in body
    assert "Dravet syndrome" in body and ">SCN1A</td>" in body
    assert body.count("not recorded") == 2
    assert (f"Reference cohort <b>reference-1</b> · version <code>{VERSION_SHA[:12]}</code>"
            in body)
    assert VERSION_SHA not in body and SAMPLES_SHA not in body
    out = dp.details_outputs(result, 1)
    details = out["details_html"]["value"]
    assert "<code>HP:0001250</code> <code>HP:0001263</code>" in details
    assert "Recorded HPO terms (2)" in details and dp.REFERENCE_CONTEXT_NOTE in details
    assert "<code>MONDO:0011073</code>" in details
    assert "#1</span> case-07</h2>" in out["details_hd"] and "<b>-0.120</b>" in out["details_hd"]
    second = dp.details_outputs(result, 2)["details_html"]["value"]
    assert second.count("not recorded") == 2
    digests = dp.result_digests_html(result)
    for digest in (VERSION_SHA, SAMPLES_SHA):
        assert f'<code class="shep-full" title="{digest}">{digest}</code>' in digests
        assert f'data-copy="{digest}"' in digests
    assert "Reference cohort reference-1 · version (cohort.json)" in digests


def test_the_task_tables_mark_the_selected_row_without_aria_selected():
    """7, 8: a plain table's rows take no `aria-selected` (skill §8): the
    selected gene or reference patient is `shep-on` and `aria-current`."""
    for result in (_gene_result(), _patients_result()):
        body = dp.results_body_html(result, 2)
        rows = re.findall(r"<tr data-key[^>]*>", body)
        assert len(rows) == 2 and "aria-selected" not in body
        assert rows == ['<tr data-key="1" tabindex="0">',
                        '<tr data-key="2" tabindex="0" class="shep-on" aria-current="true">']


def _escaped_results():
    gene = _gene_result()
    gene["candidates"][0].update(gene_name=_EVIL, gene_id=_EVIL)
    patients = _patients_result()
    patients["candidates"][0].update(case_id=_EVIL, disease_name=_EVIL, phenotypes=[_EVIL],
                                     causal_genes=[_EVIL])
    patients["candidates"][1].update(disease_id=_EVIL)
    patients["reference_cohort"]["label"] = _EVIL
    return gene, patients


def test_values_of_both_task_kinds_are_escaped_on_every_surface():
    """9: markup in a gene's name or id, a reference patient's research id,
    diagnosis, HPO terms or causal genes, or the reference label is never
    drawn as markup: not in the table, details, header or digests; the report
    escapes it or keeps it in a code span; the CSV defuses a formula."""
    for result in _escaped_results():
        kind = result["result_kind"]
        html = (dp.results_body_html(result, 1) + dp.result_digests_html(result)
                + dp.result_identity_html(result, dp.card.new_state()))
        for rank in (1, 2):
            out = dp.details_outputs(result, rank)
            html += out["details_hd"] + out["details_html"]["value"]
            html += dp.details_header_html(result["candidates"][rank - 1], kind)
        assert "<img" not in html and _EVIL_SHOWN in html, kind
        # Markdown: escaped (``\<img``), or literal inside a code span.
        report = _re.sub(r"(`+)[^`]*?\1", "", dp._build_results_report_md(result))
        assert not _re.search(r"(?<!\\)<img", report) and "\\<img" in report, kind
    _, patients = _escaped_results()
    patients["candidates"][0]["case_id"] = "=HYPERLINK(\"http://x\")"
    _, (first, _second), _ = _csv_records(dp._build_results_csv(patients))
    assert first["case_id"] == "'=HYPERLINK(\"http://x\")"
    assert first["reference_cohort_label"] == _EVIL


def test_each_task_kinds_csv_holds_its_own_columns_before_the_served_ones():
    """10: the kind's columns, then what the result carries for every row, then
    the unchanged served columns; numbers to four places, digests sha256:."""
    header, (first, second), rows = _csv_records(dp._build_results_csv(_gene_result()))
    assert header == _KIND_CSV_COLUMNS["gene_task/v1"] + _SERVED_COLUMNS
    assert (first["gene_id"], first["score"], first["embsim"]) == ("disgenet:SCN1A", "0.6100",
                                                                   "0.4000")
    assert (first["sp_bounds_lower"], first["sp_bounds_upper"]) == ("1.0000", "6.8400")
    assert second["sp_bounds_upper"] == "6.8400" and second["score"] == "-0.2000"
    assert first["served_scoring_mode"] == "gene_task" and first["served_eta"] == "0.5"
    header, (first, second), _ = _csv_records(dp._build_results_csv(_patients_result()))
    assert header == _KIND_CSV_COLUMNS["patients_like_me/v1"] + _SERVED_COLUMNS
    assert first["reference_cohort_version_sha256"] == f"sha256:{VERSION_SHA}"
    assert first["reference_cohort_samples_sha256"] == f"sha256:{SAMPLES_SHA}"
    assert (first["reference_cohort_label"], first["reference_cohort_patients"]) == (
        "reference-1", "12")
    assert (first["phenotypes"], first["causal_genes"]) == ("HP:0001250; HP:0001263", "SCN1A")
    assert (first["disease_id"], first["similarity"]) == ("MONDO:0011073", "-0.1200")
    assert (second["disease_name"], second["causal_genes"]) == ("", "")


def test_each_task_kinds_report_says_what_its_numbers_are_and_what_it_ranked():
    """11: the kind's own note (never "disease probability"), the submitted
    genes and the SP bounds, or the reference cohort version; no evidence or
    explanation for either kind."""
    gene = dp._build_results_report_md(_gene_result())
    assert dp.GENE_SCORE_NOTE in gene and "disease probability" not in gene
    assert "- **Candidate genes submitted**: SCN1A, disgenet:PTEN, NOT\\_A\\_GENE" in gene
    assert "- **SP similarity bounds** (the model's own): 1.000 to 6.840" in gene
    assert "| 1 | **SCN1A** | `disgenet:SCN1A` | 0.610 | 0.400 | 0.820 | 1.500 |" in gene
    assert "## Per-candidate detail" not in gene and "**Evidence**" not in gene
    patients = dp._build_results_report_md(_patients_result())
    assert dp.SIMILARITY_NOTE in patients and "disease probability" not in patients
    assert "- **Reference cohort**: `reference-1`" in patients
    assert f"- **Reference cohort version** (cohort.json): `sha256:{VERSION_SHA}`" in patients
    assert f"- **Reference cohort samples**: `sha256:{SAMPLES_SHA}`" in patients
    assert "- **Reference cohort patients**: 12" in patients
    assert "| 2 | `case-02` | -0.500 | not recorded | not recorded |" in patients
    assert "**Recorded HPO terms**: HP:0001250; HP:0001263" in patients
    assert dp.REFERENCE_CONTEXT_NOTE in patients and "**Evidence**" not in patients
    assert "| Path settings | not applicable |" in patients


def test_a_refusal_the_route_raises_is_shown_in_words_without_a_path(monkeypatch, tmp_path):
    """12: every reference cohort refusal reaches the page as
    ``{"error": "<detail>"}``; its words are shown, the registry's paths
    named ``<registry>/…`` and any other server path hidden."""
    registry = tmp_path / "registry"
    registry.mkdir()
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(registry))
    detail = (f"reference cohort 'x': {registry}/versions/x was mapped against another graph "
              "(node_features differ); see /srv/private/notes.txt")
    said = dp._describe_refused_request(_response(422, {"error": detail, "status_code": 422}))
    assert said == ("Request refused — reference cohort 'x': <registry>/versions/x was mapped "
                    "against another graph (node_features differ); see a server path")
    monkeypatch.setattr(dp, "_self_request",
                        lambda method, path, **kw: _response(422, {"error": detail}))
    assert dp._call_diagnose(["HP:0001250"], reference_cohort="x")["error"] == said
    # The server's own condition (503, no registry) is said in its words too.
    monkeypatch.setattr(dp, "_self_request", lambda method, path, **kw: _response(
        503, {"error": f"no reference cohort version can be read: {registry} is refused"}))
    assert dp._call_diagnose(["HP:0001250"])["error"] == (
        "API error: 503 — no reference cohort version can be read: <registry> is refused")


def test_path_settings_are_not_applicable_where_no_path_is_discovered():
    """13: a gene task or patients-like-me model discovers no paths."""
    assert dp._path_settings_cells(_task_identity("gene_task")) == (
        "not applicable", "not applicable — a gene task model discovers no paths")
    assert dp._path_settings_cells(_task_identity("patients_like_me")) == (
        "not applicable", "not applicable — a patients-like-me model discovers no paths")
    for identity in (_task_identity("disease_task"), _served_identity()):
        state, value = dp._path_settings_cells(identity)
        assert state == "in effect" and "`max_path_length` 4" in value


def test_the_task_help_texts_answer_first_within_their_budget():
    """Skill §5/§7: each new text leads with its answer (about 90 characters),
    about 80 words, sentences of 25 words at most; each field's title is its
    label (`test_webui_help.py` pins that against the built app)."""
    texts = [dp.FIELD_HELP["dx_candidate_genes"], dp.FIELD_HELP["dx_reference_cohort"],
             dp.TOPIC_HELP["dx-gene-score"], dp.TOPIC_HELP["dx-similarity"],
             dp.TOPIC_HELP["dx-run-no-cohort"]]
    for title, text in texts:
        lead = re.split(r"(?<=[.!?])\s+", text)[0]
        assert len(lead) <= 95, (title, lead)
        assert len(text.split()) <= 80, title
        assert max(len(s.split()) for s in re.split(r"(?<=[.!?])\s+", text)) <= 25, title
    assert [dp.FIELD_HELP[k][0] for k in ("dx_candidate_genes", "dx_reference_cohort")] == [
        "Candidate genes", "Reference cohort"]


def test_the_change_drawer_filters_its_list_by_task(monkeypatch):
    """The drawer's Task control (K6 decision 14, map_models §D): All tasks,
    Disease, Gene, Patients-like-me, All tasks first; its `.input` sets the
    shared list's view (``{"task": …}``), which the list redraws on; the
    compare link stays beside it."""
    demo, _ctx = _build_tab(monkeypatch)
    blocks = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    control = blocks["shep-dx-change-task"]
    assert isinstance(control, gr.Radio) and control.value == "all"
    labels = [label for label, _value in control.choices]
    assert labels == ["All tasks", "Disease", "Gene", "Patients-like-me"]
    assert [value for _label, value in control.choices] == [
        "all", "disease", "gene", "patients_like_me"]
    assert len(set(labels)) == len(labels)
    assert all(len(label) <= 40 and label.count(" · ") <= 1 for label in labels)
    (fn,) = [f for f in demo.fns.values() if f.fn is dp.task_view]
    assert fn.targets == [(control._id, "input")]
    (view,) = fn.outputs
    assert any(f.name == "_redraw" and (view._id, "change") in f.targets
               for f in demo.fns.values())
    assert dp.task_view("gene") == {"task": "gene"}
    assert dp.task_view("history") == {"task": "all"}
    assert "shep-dx-compare" in blocks


# --------------------------------------------------------------------------- Stop inference (A3b)
_SERVED_SHA = "b" * 64


def _serving_card(sha=_SERVED_SHA, serving="serving"):
    from src.webui.components import diagnosis_card as card

    status = {"initialized": True, "serving": serving, "loading": False, "in_flight": 1,
              "served_identity": {"scoring_mode": "disease_task",
                                  "checkpoint": {"state": "verified", "sha256": sha}}}
    return {**card.new_state(), "ok": True, "status": status, "operator": True}


def _no_stop(*args, **kwargs):
    raise AssertionError("the stop ran")


def test_stop_inference_asks_first_naming_and_binding_the_model_shown():
    from src.utils.operator_access import ENABLED

    asked = dp.ask_stop_inference(ENABLED, _serving_card())
    assert f"<code>{_SERVED_SHA[:12]}</code>" in asked["stop_text"]
    assert "Diagnoses already accepted finish; new diagnoses are refused until a model is " \
           "loaded" in asked["stop_text"]
    assert asked["stop_asked"]["sha256"] == _SERVED_SHA
    assert _shown(asked["stop_now_btn"]) and _shown(asked["stop_keep_btn"])
    nothing = dp.ask_stop_inference(ENABLED, _serving_card(serving="stopping"))
    assert "sha256" not in nothing["stop_asked"] and "Nothing to stop" in nothing["stop_text"]
    assert not _shown(nothing["stop_now_btn"])


def test_a_read_only_request_is_refused_and_nothing_is_asked_or_stopped():
    from src.utils.operator_access import OperatorAccess

    refused = OperatorAccess(False, "not_loopback", "This request did not arrive on loopback.")
    asked = dp.ask_stop_inference(refused, _serving_card())
    assert "sha256" not in asked["stop_asked"] and "Refused: " in asked["stop_text"]
    assert not _shown(asked["stop_now_btn"])
    done = dp.confirm_stop_inference(refused, {"sha256": _SERVED_SHA}, stop=_no_stop)
    assert "Refused: " in done["stop_notice"] and "Nothing was done." in done["stop_notice"]


def test_the_confirmed_stop_is_the_one_stop_bound_to_the_model_shown():
    from src.api.routes.pipeline import PipelineStopResponse, PipelineStatusResponse
    from src.utils.operator_access import ENABLED

    seen = []

    def stop(request):
        seen.append(request.expected_checkpoint_sha256)
        return PipelineStopResponse(
            success=True, released=True,
            message="Inference stopped: 1 accepted request(s) finished first <b>x</b>.",
            status=PipelineStatusResponse(initialized=False))

    resources = {"gpu": {"available": False}, "ram": {"total_gb": 128.0, "used_gb": 20.5},
                 "unified_memory": False}
    done = dp.confirm_stop_inference(ENABLED, {"sha256": _SERVED_SHA}, stop=stop,
                                     resources=resources)
    assert seen == [_SERVED_SHA]
    assert "1 accepted request(s) finished first &lt;b&gt;x&lt;/b&gt;." in done["stop_notice"]
    assert "<b>x</b>" not in done["stop_notice"]
    assert "not a promise that memory was returned" in done["stop_notice"]
    assert "memory 20.5 / 128.0 GB" in done["stop_notice"]
    # Bound to what the stop left served (nothing), never confirmable again.
    assert done["stop_asked"]["notice_for"] is None and "sha256" not in done["stop_asked"]
    assert not _shown(done["stop_now_btn"])

    def not_released(request):
        return PipelineStopResponse(success=False, released=False,
                                    message="Not stopped: the model in use is aaaaaaaaaaaa.",
                                    status=PipelineStatusResponse(initialized=True))

    kept = dp.confirm_stop_inference(ENABLED, {"sha256": _SERVED_SHA}, stop=not_released,
                                     resources=resources)
    assert "Not stopped" in kept["stop_notice"] and "Read after the stop" not in kept[
        "stop_notice"]


def test_a_confirmation_never_asked_stops_nothing():
    from src.utils.operator_access import ENABLED

    for asked in (None, {}, {"sha256": "not a digest"}):
        done = dp.confirm_stop_inference(ENABLED, asked, stop=_no_stop)
        assert "Nothing was stopped" in done["stop_notice"]


def test_the_stop_handler_calls_run_stop_after_d8_on_its_own_request(monkeypatch):
    """`_on_stop_inference` reads D8 on the event's own request, then calls the
    one stop in process (`run_stop`) bound to the asked model; refused, it
    never reaches it."""
    from src.api.routes import pipeline
    from src.api.routes.pipeline import PipelineStopResponse, PipelineStatusResponse
    from tests.fixtures.operator_access import gradio_request, operator_mode

    seen = []

    def stop(request):
        seen.append(request.expected_checkpoint_sha256)
        return PipelineStopResponse(success=False, released=False, message="Nothing to stop.",
                                    status=PipelineStatusResponse(initialized=False))

    monkeypatch.setattr(pipeline, "run_stop", stop)
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    refused = dict(zip(dp.STOP_OUTPUTS, dp._on_stop_inference(gradio_request(),
                                                               {"sha256": _SERVED_SHA})))
    assert seen == [] and "Refused: " in refused["stop_notice"]
    operator_mode(monkeypatch)
    allowed = dict(zip(dp.STOP_OUTPUTS, dp._on_stop_inference(gradio_request(),
                                                               {"sha256": _SERVED_SHA})))
    assert seen == [_SERVED_SHA] and "Nothing to stop." in allowed["stop_notice"]


def _stop_released(request):
    from src.api.routes.pipeline import PipelineStatusResponse, PipelineStopResponse

    return PipelineStopResponse(
        success=True, released=True,
        message="Inference stopped: 0 accepted request(s) finished first, then model "
                "bbbbbbbbbbbb and its graph were released. New diagnoses are refused until a "
                "model is loaded.",
        status=PipelineStatusResponse(initialized=False))


_RESOURCES = {"gpu": {"available": False}, "ram": {}, "unified_memory": False}
_NOTHING_SERVED = {"initialized": False, "serving": "not_loaded", "loading": False,
                   "in_flight": 0, "not_serving_reason": "Inference was stopped by an operator."}


def _read(card_state, stop, status=None):
    """A card read (`card_outputs`, as the reads call it) of `status` (or the
    card's own), after `card_state`'s reads, with this session's binding."""
    state = dict(card_state) if status is None else {**card_state, "status": status}
    return dp.card_outputs(state, None, "idle", stop=stop)


def test_a_stopped_notice_lasts_while_nothing_serves_and_goes_once_a_model_is_loaded():
    """Review of A3b (P2): what a confirmed stop said is the page's, and a read
    of the server's status owns it: kept while nothing is served (the read
    right after the stop included), cleared once a model serves again (this
    session's Load, another browser's, a Restart that loads one), once, and
    again after the next stop."""
    from src.utils.operator_access import ENABLED

    before = _read(_serving_card(), None)["card_state"]
    done = dp.confirm_stop_inference(ENABLED, {"sha256": _SERVED_SHA}, stop=_stop_released,
                                     resources=_RESOURCES)
    assert "New diagnoses are refused until a model is loaded" in done["stop_notice"]
    after = _read(before, done["stop_asked"], _NOTHING_SERVED)
    assert not set(dp.STOP_VIEWS) & set(after), "the read after the stop closed its notice"
    again = _read(after["card_state"], done["stop_asked"], _NOTHING_SERVED)
    assert not set(dp.STOP_VIEWS) & set(again)
    for served in (_SERVED_SHA, "c" * 64):           # the same model, or another
        loaded = _read(again["card_state"], done["stop_asked"], _serving_card(served)["status"])
        assert loaded["stop_notice"] == "", "the notice outlived the stop's state"
        assert "stop_text" not in loaded, "a notice closes only the notice"
        assert loaded["card_state"]["rendered"]["stop_closed"] == done["stop_asked"]
        assert "stop_notice" not in _read(loaded["card_state"], done["stop_asked"])
    second = dp.confirm_stop_inference(ENABLED, {"sha256": _SERVED_SHA}, stop=_stop_released,
                                       resources=_RESOURCES)
    assert second["stop_asked"] != done["stop_asked"], "two stops are two bindings"
    reloaded = _read(loaded["card_state"], second["stop_asked"])
    assert reloaded["stop_notice"] == ""


def test_a_refused_stop_says_so_while_the_model_it_found_serves():
    """A stop refused because another model serves says so for as long as
    the card offers that model (the read right after it included), and goes
    once that one is no longer served."""
    from types import SimpleNamespace

    from src.utils.operator_access import ENABLED

    other = _serving_card("c" * 64)

    def another_serves(request):
        # The stop's answer, its status standing in for the route's model.
        return SimpleNamespace(
            success=False, released=False,
            message="Not stopped: the model in use is cccccccccccc, not bbbbbbbbbbbb.",
            status=SimpleNamespace(model_dump=lambda: dict(other["status"])))

    done = dp.confirm_stop_inference(ENABLED, {"sha256": _SERVED_SHA}, stop=another_serves)
    assert done["stop_asked"]["notice_for"] == "c" * 64
    kept = _read(other, done["stop_asked"])
    assert "stop_notice" not in kept
    gone = _read(kept["card_state"], done["stop_asked"], _NOTHING_SERVED)
    assert gone["stop_notice"] == ""


def test_an_open_question_closes_once_the_card_no_longer_offers_its_model():
    """Stop inference's question is bound to the model shown: it stays while
    the card offers that model, and closes (its line, both buttons) once a
    read finds it stopped elsewhere or another model served; the binding is
    left as the handler wrote it."""
    from src.utils.operator_access import ENABLED

    asked = dp.ask_stop_inference(ENABLED, _serving_card())
    still = _read(_serving_card(), asked["stop_asked"])
    assert not set(dp.STOP_VIEWS) & set(still)
    for status in (_NOTHING_SERVED, _serving_card("c" * 64)["status"],
                   _serving_card(serving="stopping")["status"]):
        closed = _read(still["card_state"], asked["stop_asked"], status)
        assert closed["stop_text"] == "" and not _shown(closed["stop_now_btn"])
        assert not _shown(closed["stop_keep_btn"]) and "stop_notice" not in closed
        assert "stop_asked" not in closed, "a read wrote the binding"
    nothing = dp.ask_stop_inference(ENABLED, {**_serving_card(), "status": _NOTHING_SERVED})
    loaded = _read(_serving_card(), nothing["stop_asked"])
    assert loaded["stop_text"] == "", "'Nothing to stop' outlived the state it described"


def test_run_diagnosis_never_writes_the_stop_views():
    """Run Diagnosis's card update passes no binding: the stop views are the
    card's reads' alone."""
    out = dp.card_outputs(_serving_card(), None, "idle")
    assert not set(dp.STOP_VIEWS) & set(out)


def test_after_a_stop_run_is_off_with_the_backends_reason_and_the_result_keeps_its_model():
    """B4: after a stop, Run Diagnosis is off ("Disabled until a model is
    loaded"), the card says the backend's state and lead (Stopped, who and
    when, the whole reason one click away), and the result produced before
    it keeps the model and data it used: its table is not redrawn, the badge
    names its own model and compares it with nothing (never "10 parts
    differ"), and "This result" lists its own identity."""
    served = _card(_status_serving(_served_identity()))
    result = _produced()
    assert dp.compare_with_served(result, served)["state"] == "equal"
    stopped = _card({"initialized": False, "serving": "not_loaded", "loading": False,
                     "in_flight": 0, "holds_gpu_lock": False,
                     "not_serving_reason": "Inference was stopped by an operator at "
                                           "2026-07-07T16:00:30+00:00: model "
                                           f"{CHECKPOINT_SHA[:12]} is no longer loaded.",
                     "not_serving": {"state": "stopped", "lead": "By an operator",
                                     "at": "2026-07-07T16:00:30+00:00",
                                     "model_sha12": CHECKPOINT_SHA[:12]}},
                    state=served, at="2026-07-07T16:00:40")
    out = dp.card_outputs(stopped, result, "idle")
    assert out["run_btn"]["interactive"] is False
    assert dp.RUN_DISABLED_NO_MODEL in out["run_reason"]
    assert "By an operator at 16:00:30 UTC" in out["card_body"]
    assert "Inference was stopped by an operator" in out["card_body"]   # under Details ›
    assert "results_body" not in out, "a result is never redrawn by a card read"
    compared = dp.compare_with_served(result, stopped)
    assert compared["state"] == "nothing" and compared["differ"] == []
    badge = out["badge"]
    assert f"Produced by model {CHECKPOINT_SHA[:12]} · not compared" in badge
    assert "parts differ" not in badge and "the model in use" not in badge.split("dx-badge")[1]
    shown = dp.result_identity_html(result, stopped)
    assert "<b>Not compared:</b>" in shown and "<table" not in shown
    own = shown[shown.index("What produced this result"):]
    assert f'title="{CHECKPOINT_SHA}">{CHECKPOINT_SHA[:12]}</code>' in own
    assert "no_model" not in shown and "no model is loaded" not in shown


def test_run_diagnosis_is_off_while_inference_stops():
    run = dp.run_controls(_serving_card(serving="stopping"), "idle")
    assert run["run_btn"]["interactive"] is False
    # The reason refers to the card's state (B4); the bubble is the stop's.
    assert dp.RUN_DISABLED_NO_MODEL in run["run_reason"]
    assert 'data-help="dx-run-stopping"' in run["run_cover"]
    assert dp.run_controls(_serving_card(), "idle")["run_btn"]["interactive"] is True


def test_the_card_shows_stop_inference_by_class_while_a_model_serves():
    out = dp.card_outputs(_serving_card(), None, "idle")
    assert _shown(out["stop_btn"]) and out["stop_btn"]["value"] == "■ Stop inference"
    assert "visible" not in out["stop_btn"]
    hidden = dp.card_outputs({**_serving_card(), "status": {"initialized": False}}, None, "idle")
    assert not _shown(hidden["stop_btn"])
