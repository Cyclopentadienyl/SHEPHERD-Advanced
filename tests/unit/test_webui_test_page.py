"""
Research › Evaluate › Test (M5-2, UI-D): import, run, keep and register.
========================================================================
`src/webui/components/test_page.py`. What these pin:

- **D8 on every operator control's own request, then the shared service**:
  the import (listing the intake directory, preview, commit), start, cancel,
  registration and investigation notes are refused for a read-only request
  without calling the service; the checkpoint list (it opens every
  checkpoint) is refused and opens nothing; a direct call through Gradio's
  HTTP API meets the same check; each act's chain reads the chips again;
- **one path per concern**: the job runner's one start with inputs from the
  server's lists only and **the digests shown as the expected ones**; the
  importer's two steps (`prepare_import`, `commit_import` under the preview id
  shown, the form the one previewed); the one registration function with the
  ledger beside the **listed** checkpoint holding the weights (found by full
  digest, never the report's recorded path), a report that changed since it
  was shown refused;
- **the measurement** (K2, R1): the Measurement radio offers the protocols the
  chosen model's listed record admits (`protocols.available`), Start's reason
  and the start take the chosen protocol and say the protocol definition's
  reasons (a base is offered the KG link-prediction protocol, research
  comparison Stage 5), and the runner is given the protocol
  apart from the model's stage and task; under test-only protocols p1 and p2 a
  disease child offers both and starts p2 with its task still disease;
- **D5** (policy rev 16): a group whose runs differ is described in neutral
  words ("runs under these conditions differ"), flagged in amber with ▲ and
  "— investigate" only when it records deterministic algorithms on; each run
  can be registered; a note is kept with who, when and what and lifts nothing;
- **D7**: the preview shows exclusion reasons and their counts, never a case
  id; there is no upload control;
- **R10 evidence**: an evidence directory the server cannot check never
  disables Start; the row says which and why in amber with ▲, redacted, and
  the run is given it; one known not to be a directory still disables Start
  (K8-6 completion review, 2026-11-08);
- **every message true**: the job's pill is never green for a failed or
  cancelled run; the outcome is the runner's (kept, registered, "no report was
  published"); *unverifiable* is a word, never a 0; no absolute server path is
  shown; values are escaped;
- **the page's wiring**: refreshed when the page is shown; polled on the
  shell's slow timer with ``queue=False``, doing work only while it is shown
  and never touching a select, the selection or Start (no listener on the fast
  timer), the one writer of what the session saw of the job, each change
  handed once to a queued listener, the only one that chooses an ended run's
  report;
  every write of the selection in one queued group of one, so a late answer
  never writes back the selection it started from (the page served by Gradio,
  one read held); the report of a run that has just ended selected; the
  import's preview and commit one at a time; field help registered with titles equal to the labels; its style
  sheet registered and never under 14 px; the batch size shown is the
  measurement's default.

Module: tests/unit/test_webui_test_page.py
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
from types import SimpleNamespace

import pytest

gr = pytest.importorskip("gradio")
pytest.importorskip("torch")

from src.config.model_types import WRITTEN_BEFORE_SPEC  # noqa: E402
from src.webui.components import model_list as ml  # noqa: E402
from src.webui.components import renderers  # noqa: E402
from src.webui.components import test_page as tp  # noqa: E402
from src.webui.components.help import sentence_lines  # noqa: E402
from tests.fixtures.capabilities import described  # noqa: E402
from tests.fixtures.operator_access import (  # noqa: E402
    OPERATOR_BASE_URL,
    gradio_request,
    operator_mode,
    operator_request,
)
from tests.fixtures.r10_table import cell_text, r10_cells, without_r10  # noqa: E402
from tests.fixtures.served_page import ServedSession, served  # noqa: E402
from tests.fixtures.trained_children import CHILD_RUNS  # noqa: E402

RUN = "20261017-091500_run_3fa9c1"
LABEL = "demo-test-1"


# --------------------------------------------------------------------------- the reports table
def _report_rows(html: str) -> dict:
    """The reports table's selectable rows as their cells' plain words, by
    run id (`tp.reports_html`: Report, Model, Measurement, State)."""
    import html as html_lib

    rows = {}
    for key, body in re.findall(r'<tr class="[^"]*" data-key="([^"]+)"[^>]*>(.*?)</tr>', html):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", body)
        rows[html_lib.unescape(key)] = [
            " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", cell)).split()) for cell in cells]
    return rows


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def built(trained_disease_child, tmp_path_factory):
    """A workspaces root with the demo workspace holding two runs: the
    session's trained disease child (`trained_disease_child`, copied) and a
    checkpoint without a model spec, which the listing lists as refused; a
    registry with one version holding one kept, unregistered report (the real
    disease task measurement of the child, on CPU: a 1-epoch model, so no
    number here is a result); and an intake directory with one source."""
    import scripts.measure_scorer as measure_scorer
    from src.utils.fingerprint import file_sha256
    from tests.fixtures.registered_cohort import (
        graph_source_cases,
        import_version,
        write_checkpoint_without_model_spec,
        write_source,
    )

    base = tmp_path_factory.mktemp("testpage")
    workspaces = base / "ws"
    workspaces.mkdir()
    demo = workspaces / "demo"
    shutil.copytree(trained_disease_child["data_dir"], demo)
    child = demo / "runs" / "gatv2" / CHILD_RUNS["disease"] / "checkpoints" / "last.pt"
    child.parent.mkdir(parents=True)
    shutil.copy2(trained_disease_child["disease"], child)
    history = demo / "runs" / "hgt" / RUN / "checkpoints" / "best.pt"
    history.parent.mkdir(parents=True)
    write_checkpoint_without_model_spec(history, data_dir=demo)
    registry = base / "c" / "registry"
    registry.mkdir(parents=True)
    source = write_source(base / "c" / "cases.json", graph_source_cases(demo))
    version = import_version(registry, demo, source)
    intake = base / "intake"
    intake.mkdir()
    shutil.copy2(source, intake / "cases.json")
    # The registry is the deployment's while the run measures (BACKLOG 37).
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SHEPHERD_REGISTRY", str(registry))
        code = measure_scorer.main(["--mode", "disease", "--checkpoint", str(child),
                                    "--data-dir", str(demo), "--cohort", LABEL, "--registry",
                                    str(registry), "--device", "cpu"])
    assert code == 0
    return SimpleNamespace(base=base, sha=file_sha256(child), history_sha=file_sha256(history),
                           version_sha=version.identity.sha256)


@pytest.fixture
def site(built, tmp_path, monkeypatch):
    """A copy of the built tree, with the server settings naming it."""
    from src.webui.components import served_status

    root = tmp_path / "site"
    shutil.copytree(built.base, root, symlinks=True)
    registry = root / "c" / "registry"
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root / "ws"))
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(registry))
    monkeypatch.setenv("SHEPHERD_INTAKE_DIR", str(root / "intake"))
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")
    monkeypatch.delenv("SHEPHERD_TRAINING_EVIDENCE_DIRS", raising=False)
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    monkeypatch.setattr(served_status, "read_status", lambda: (None, "not read"))
    reports = registry / "versions" / LABEL / "reports"
    (run_dir,) = [p for p in reports.iterdir() if p.is_dir()]
    return SimpleNamespace(
        root=root, ws=root / "ws", demo=root / "ws" / "demo", registry=registry,
        intake=root / "intake", reports=reports, run_id=run_dir.name, sha=built.sha,
        version_sha=built.version_sha,
        checkpoint=(root / "ws" / "demo" / "runs" / "gatv2" / CHILD_RUNS["disease"]
                    / "checkpoints" / "last.pt"),
        key=f"gatv2/{CHILD_RUNS['disease']}/last.pt",
        history_key=f"hgt/{RUN}/best.pt", history_sha=built.history_sha)


def _readonly():
    return gradio_request(server=("192.0.2.10", 8264))


def _page(monkeypatch, *, operator=True):
    request = operator_request(monkeypatch) if operator else _readonly()
    return tp.read_page(request), request


def _models(page, request):
    return tp.read_models("demo", request)


#: Synthetic bundle runs filed beside the fixture's trained child and its file
#: without a model spec (listed, its record refused with why; K6 decision 1).
CHILD_RUN = "20261020-000000_disease_bbbbbb"
BUNDLE_BASE_RUN = "20261019-000000_base_aaaaaa"


def _put_bundle(site, run=CHILD_RUN, task="disease", **change):
    """A model bundle's checkpoint under ``runs/gatv2/<run>`` (a finetune child
    of `task`, or a base for `task` None), its record as the training payload
    writes it (`test_task_first_selection._record_payload`, K6 decision 10's
    precedent for listing tests, `test_finetune_children`), `change` overriding
    payload keys; for the tests that start a run through `FakeManager`, so its
    weights are never measured. Returns its path and listing key."""
    import time

    from tests.unit.test_task_first_selection import _put, _record_payload

    path = _put(site.demo, "gatv2", run, {**_record_payload(task), **change}, time.time())
    return path, f"gatv2/{run}/last.pt"


class FakeManager:
    """The job runner's surface the page uses, recording what it is given."""

    def __init__(self, status=None):
        self.started, self.cancelled = [], 0
        self.status = status or {"status": "idle", "job_id": None, "inputs": None,
                                 "outcome": None}

    def start_test(self, inputs):
        self.started.append(inputs)
        return {"success": True, "job_id": "test-1", "status": "running"}

    def cancel_test(self):
        self.cancelled += 1
        return {"success": True, "status": "cancelled", "outcome": None}

    def get_test_status(self):
        return dict(self.status)

    def get_log(self, job_type):
        return {"lines": [f"line {i}" for i in range(40)]}

    def active_job(self):
        return None


def _disagreeing_copy(site, delta=0.0004):
    """A hand-edited copy of the kept report under a new run id: the same key,
    another number (what a disagreement looks like in the store, D5)."""
    source = site.reports / site.run_id
    copy = site.reports / f"20261020T180000Z-{site.sha[:12]}-0badc0de"
    shutil.copytree(source, copy)
    report = json.loads((copy / "report.json").read_text())
    report["authoritative_metrics"]["pooled_mrr"] += delta
    (copy / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return copy.name


# --------------------------------------------------------------------------- pure pieces
def test_no_absolute_path_is_shown():
    roots = [("/srv/registry", "<registry>"), ("/srv/ws", "<workspaces>")]
    text = ("A report was kept at /srv/registry/versions/x/reports/r1; ledger /srv/ws/demo/"
            "runs/gatv2/r/checkpoints/evaluations.json; other /etc/passwd and C:\\Users\\x; "
            "/srv/ws2/elsewhere stays out; 'at /srv/registry'.")
    out = tp.redact(text, sorted(roots, key=lambda p: -len(p[0])))
    assert "<registry>/versions/x/reports/r1" in out
    assert "<workspaces>/demo/runs/gatv2/r/checkpoints/evaluations.json" in out
    assert "/etc" not in out and "C:\\" not in out and "/srv/ws2" not in out
    assert out.count(tp.SERVER_PATH) == 3 and "'at <registry>'" in out


def test_the_path_roots_are_the_server_settings_and_the_repository(site):
    roots = {token: prefix for prefix, token in tp.path_roots()}
    assert roots["<registry>"] == str(site.registry.resolve())
    assert roots["<workspaces>"] == str(site.ws.resolve())
    assert roots["<intake>"] == str(site.intake.resolve())
    assert "<repository>" in roots


def test_a_run_id_names_its_time_and_model():
    run = "20261017T152412Z-15f098277b54-9c1d02aa"
    assert tp.run_time(run) == "2026-10-17 15:24 UTC" and tp.run_sha12(run) == "15f098277b54"
    assert tp.run_time("usage.jsonl") is None


def test_the_batch_size_shown_is_the_measurements_default():
    import scripts.measure_scorer as measure_scorer

    args = measure_scorer.parse_args(["--mode", "disease", "--checkpoint", "c", "--data-dir",
                                      "d", "--cohort", "x", "--registry", "/r"])
    assert args.batch_size == tp.BATCH_SIZE_SHOWN


def _outside_details(html: str) -> str:
    """What a reader sees of a card with its disclosures closed."""
    return re.sub(r'<details class="[^"]*">.*?</details>', " ", html, flags=re.S)


def test_a_runs_output_is_folded_in_its_one_details_and_a_failure_shown():
    """B52-E: a run's output tail, whatever its state, is the card's one
    "Details ›", never the main space (12 unfolded lines of R10 definitions
    and a record_evaluation command); a failed run's first error line is
    said in the open. Fails on 97c8ec8's card, which drew a completed or
    failed run's output unfolded."""
    job = {"status": "cancelled", "inputs": {}, "log": ["Traceback", "KeyboardInterrupt"]}
    for status in ("cancelled", "failed", "completed", "running"):
        html = tp.job_body_html(dict(job, status=status, error_message="Process exited with "
                                     "code 1"), [], ("busy", "GPU job"))
        assert html.count("<details") == 1 and html.count('<details class="shep-more">') == 1
        assert "KeyboardInterrupt" in html and "KeyboardInterrupt" not in _outside_details(html)
        assert "Output, last 2 lines" in html
        assert ("Process exited with code 1" in _outside_details(html)) is (status == "failed")


def test_every_class_the_page_styles_is_one_it_draws():
    """No dead style: each ``shep-test-*`` class PAGE_CSS styles is named by
    the page's code outside that style. Fails on f6b9289, whose ``.shep-test-
    fold`` rules outlived the last element with the class (a cancelled run's
    output and "Recorded metrics", both now the shared Details)."""
    import inspect

    source = inspect.getsource(tp)
    start = source.index('PAGE_CSS = f"""')
    code = source[:start] + source[source.index('"""', start + len('PAGE_CSS = f"""')) + 3:]
    styled = set(re.findall(r"\.(shep-test-[\w-]+)", tp.PAGE_CSS))
    assert styled and not {name for name in styled if name not in code}


def test_the_pill_is_never_green_for_a_failed_or_cancelled_run():
    for status, kind in (("completed", "ok"), ("failed", "red"), ("cancelled", "grey"),
                         ("running", "accent")):
        html = tp.job_html({"status": status, "inputs": {}}, [], ("none", ""))
        assert f'shep-tag-{kind} shep-test-pill" data-status="{status}"' in html
    # R-81: no run started since the server did is one muted line, no pill
    # and no card (a 479 x 95 bordered card and "Test run Idle" before).
    idle = tp.job_html({"status": "idle", "inputs": {}}, [], ("none", ""))
    assert "shep-test-pill" not in idle and "shep-test-job\"" not in idle
    assert idle == (f'<div class="shep-test-job-hd"><span class="shep-sub">{tp.NO_JOB_YET}'
                    "</span></div>")
    assert tp.job_body_html({}, [], ("none", "")) == ""


def test_the_card_leads_with_what_the_run_left_and_never_says_a_registration():
    """B52-E and NEW-test-3: an ended run's card says first what it left, as
    the runner read it from disk (a kept report is Result's; no report kept;
    cannot be checked), and **never a registration of its own**: the Result
    reads it from the ledger now, so after a report is registered from the
    page no visible line of the card says "Not registered" or gives the
    record_evaluation command (both did, beside Result's "Registered record
    2"). That moment's reading, its paths and its output are one click away."""
    tail = ["Report: <registry>/versions/v/reports/r/report.json",
            "Not registered (no --record). Register it with:",
            '  python scripts/record_evaluation.py --report "<registry>/x" --checkpoint "y"']
    summary = ("When the run ended, a report was kept at /srv/reg/versions/v/reports/r; it was "
               "not registered (at that time: …).")
    job = {"status": "completed", "inputs": {}, "exit_code": 0, "exit_meaning": "measured",
           "log": tail, "outcome": {"kept": True, "registered": False, "summary": summary}}
    card = tp.job_body_html(job, [("/srv/reg", "<registry>")], ("none", ""))
    shown = _outside_details(card)
    assert tp.OUTCOME_KEPT in shown and shown.index(tp.OUTCOME_KEPT) < shown.index("Version")
    for line in ("Not registered", "not registered", "record_evaluation", "<registry>",
                 "Report kept", "Exit status", "When the run ended"):
        assert line not in shown and line not in tp.outcome_lead_html(job["outcome"]), line
    assert "Exit status" in card and "At its end" in card and "record_evaluation.py" in card
    assert "/srv/reg" not in card and "&lt;registry&gt;/versions/v/reports/r;" in card
    none = tp.outcome_lead_html({"kept": False, "registered": False, "summary": "No report."})
    assert "No report kept" in none and "egistered" not in none and "No report." not in none
    unknown = tp.outcome_lead_html({"kept": None, "registered": None, "summary": ""})
    assert "Kept: cannot be checked" in unknown and "egistered" not in unknown


def test_the_job_card_shows_the_tail_redacted_and_the_failure():
    job = {"status": "failed", "inputs": {"version_label": LABEL, "checkpoint_sha256": "a" * 64,
                                          "workspace": "/srv/ws/demo", "record": True},
           "error_message": "Process exited with code 1\n\n--- Last output ---\n/srv/ws/x",
           "exit_code": 1, "exit_meaning": "refused before anything was written",
           "log": ["reading /srv/ws/demo/kg.json", "<script>x</script>"],
           "outcome": {"kept": False, "registered": False, "summary": "No report was published."}}
    html = tp.job_html(job, [("/srv/ws", "<workspaces>")], ("none", ""))
    assert "&lt;workspaces&gt;/demo/kg.json" in html and "/srv/ws" not in html
    assert "<script>" not in html and "Process exited with code 1" in html
    assert "No report kept" in html and _identity_rows(html)["Workspace"] == "demo"


def test_the_import_form_offers_the_versions_own_roles_and_asks_for_one():
    """Guard (K8-6d): the cohort-import form's Role select and its note belong
    to the form, not to the retired ordering. The select offers the listed
    supplied versions' own roles, then mygene2 and institutional_acceptance,
    each once; a KG edge set's role (a part of an edge split, which a supplied
    import refuses for train and val) is not offered (research comparison
    Stage 4). With no role chosen the note asks for one."""
    page = {"versions": [{"role": "research_selection", "kind": "supplied"},
                         {"role": "mygene2", "kind": "supplied"}, {"role": None, "kind": None},
                         {"role": "train", "kind": "kg_edges"},
                         {"role": "val", "kind": "kg_edges"},
                         {"role": "exploration", "kind": "supplied"}, {}]}
    assert tp.role_choices(page) == ["research_selection", "mygene2", "exploration",
                                     "institutional_acceptance"]
    assert tp.role_choices({}) == ["mygene2", "institutional_acceptance"]
    prompt = ("Choose the version's role: its declared purpose (for example mygene2, "
              "research_selection or institutional_acceptance).")
    for unchosen in (None, "", "  "):
        assert tp.role_line(unchosen) == prompt


def test_the_retired_ordering_roles_setting_changes_nothing_on_the_test_page(site, monkeypatch):
    """A deployment that still sets SHEPHERD_ORDERING_ROLES is simply not read
    (K8-6d): the version list and the chosen version's line, the import form's
    role choices and its note are the same with and without it; a chosen role
    gets no note, and nothing says a role offers ordering."""
    def shown():
        page, _request = _page(monkeypatch)
        version = page["versions"][0]
        return (page["versions"], tp.version_html(page, version["label"], None),
                tp.role_choices(page),
                [tp.role_line(role) for role in (None, version["role"], "institutional_acceptance")])

    without = shown()
    role = without[0][0]["role"]
    monkeypatch.setenv("SHEPHERD_ORDERING_ROLES", f"{role},institutional_acceptance")
    assert shown() == without
    assert "offers_ordering" not in without[0][0]
    assert "ordering" not in without[1] and without[3][1:] == ["", ""]


def test_unverifiable_is_a_word_never_a_count():
    r10 = {"training": {"evidence": {"state": "found", "path": "/srv/secret/train.json"},
                        "disease_overlap": {"state": "overlap", "k": 3, "n": 5, "cases": 7},
                        "phenotype_set_overlap": {"state": "none", "k": 0, "n": 10}},
           "validation": {"evidence": {"state": "not_found"},
                          "disease_overlap": {"state": "unverifiable", "k": None, "n": 5,
                                              "cases": None},
                          "phenotype_set_overlap": {"state": "unverifiable", "k": None, "n": 10}}}
    html = renderers.r10_table_html(r10)  # the one R10 rendering the page shows
    cells = r10_cells(html)
    for column in ("Disease overlap", "Phenotype-set overlap"):
        validation = cells["Validation"][column]
        assert validation.startswith(renderers.status_mark("neutral", "Unverifiable"))
        assert " of " not in cell_text(validation) and not re.search(r"\d", cell_text(validation))
    assert cell_text(cells["Training"]["Disease overlap"]) == (
        "▲ Overlap 3 of 5 distinct test diseases, 7 test cases with one of them")
    assert cell_text(cells["Training"]["Phenotype-set overlap"]).endswith("0 of 10 test cases")
    assert "/srv/secret" not in html


#: An unverifiable label's words by its scope's evidence state, as the Test
#: page and Models show them (`caveats.R10_UNVERIFIABLE_REASON`).
NOT_FOUND_WORDS = ("no file looked at holds the bytes this checkpoint records, so there "
                   "is no count")
SEARCH_INCOMPLETE_WORDS = ("a place that may hold the bytes this checkpoint records could "
                           "not be read, so there is no count. Make it readable to the "
                           "account that measures, and measure again; the report names "
                           "the place")


@pytest.mark.parametrize("version, state, evidence_words, reason", [
    (1, "not_found", "evidence not found", NOT_FOUND_WORDS),
    (2, "search_incomplete", "evidence search incomplete", SEARCH_INCOMPLETE_WORDS),
], ids=["version-1-not-found", "search-incomplete"])
def test_a_kept_reports_r10_reads_in_its_states_words_never_a_count(
    tmp_path, version, state, evidence_words, reason
):
    """D11 (guard; Decision 7) and the K8-6r.3 words. A kept report whose R10
    block is definition version 1 with a `not_found` scope reads with its
    version-1 meaning, "no file looked at holds the bytes"; a version-2 block
    whose search was left incomplete says a place could not be read and what
    to do. Either way the labels are *unverifiable*, with no count, no 0, no
    dash and no "none", and the place's path is not shown."""
    from tests.fixtures.model_records import keep
    from tests.unit.test_evaluation_sidecar import r10_block, registered_report

    block = r10_block()
    block["definition_version"] = version
    block["training"]["evidence"] = {
        "state": state, "path": None,
        "problem": None if state == "not_found" else (
            "no candidate that was read holds the recorded bytes, and these places could "
            "not be looked in: /srv/secret/train_samples.json (PermissionError: "
            "Permission denied)")}
    path, record = keep(tmp_path / "registry", registered_report(r10=block))
    run_id = path.parent.name
    reports = {"runs": [{"run_id": run_id, "record": record,
                         "state": {"kept": "kept", "registered": "not_registered"}}]}

    html = tp.result_html(reports, run_id, [], notes=[])

    table = renderers.r10_table_html(record["r10"])
    assert table in html
    training = r10_cells(table)["Training"]
    assert training["evidence"] == evidence_words
    for column in ("Disease overlap", "Phenotype-set overlap"):
        cell = training[column]
        assert cell == (renderers.status_mark("neutral", "Unverifiable")
                        + f'<div class="shep-r10-n">{reason[:1].upper() + reason[1:]}</div>')
        text = cell_text(cell)
        assert not re.search(r"\d", text) and "—" not in text and "none" not in text.lower()
    assert "/srv/secret" not in html


def test_binding_compares_the_versions_graph_with_the_workspaces(site):
    from src.kg.artifacts import read_split_manifest

    page = tp.read_page(_readonly())
    version = tp.find(page["versions"], "label", LABEL)
    row = tp.find(page["workspaces"], "name", "demo")
    binding = tp.workspace_binding(site.demo, row)
    assert tp.binding_state(version, binding) == ("bound", "bound to this graph")
    other = dict(binding, graph={**binding["graph"], "node_features": "0" * 64})
    state, words = tp.binding_state(version, other)
    assert state == "another" and "node_features" in words
    assert tp.binding_state(version, None)[0] == "unknown"
    assert set(binding["graph"]) == {"kg", "node_features", "edge_indices", "num_nodes"}
    assert binding["graph"]["kg"] == read_split_manifest(site.demo).manifest["artifacts"]["kg"]


# --------------------------------------------------------------------------- reading
def test_a_read_only_request_sees_the_versions_and_reports_and_no_checkpoint(site, monkeypatch):
    from src.api.services import server_lists

    monkeypatch.setattr(server_lists, "list_checkpoints",
                        lambda *a, **k: pytest.fail("a read-only request listed checkpoints"))
    operator_mode(monkeypatch)      # mode M1, but the request does not arrive on loopback
    page = tp.read_page(_readonly())
    assert page["access"]["allowed"] is False
    assert [v["label"] for v in page["versions"]] == [LABEL]
    models = tp.read_models("demo", _readonly())
    assert models["refused"] is True and models["items"] == []
    # Why, said once above the Run and Checkpoint fields (V-15), not under them.
    assert tp.listing_why_html(page, models).count("Operator controls only: request not local.") == 1
    assert tp.model_html(page, models, None) == ""
    reports = tp.read_reports(LABEL)
    assert [r["run_id"] for r in reports["runs"]] == [site.run_id]
    allowed, why = tp.start_state(page, models, "demo", LABEL, None, None, ("none", ""))
    assert allowed is False and why == "Operator controls only: request not local."


def test_an_operator_lists_the_workspaces_checkpoints(site, monkeypatch):
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    # Latest run first (by its newest file): the history file is written after
    # the child is copied with its training time kept.
    assert [tp.item_key(i) for i in models["items"]] == [site.history_key, site.key]
    assert [i["sha256"] for i in models["items"]] == [site.history_sha, site.sha]
    # The model as Run → Checkpoint, the shared pair (`parents_update`): the
    # runs latest first, each valued by its run's key; the chosen run's
    # checkpoints, each valued by the listing's key (never a 91-character
    # option: the skill's §8, at most 40 characters and one separator).
    by_key = {tp.item_key(i): i for i in models["items"]}
    runs, checkpoints = ml.parents_update(tp.model_parents(models), site.key, first=False)
    assert [value for _label, value in runs["choices"]] == [
        ml.item_run_key(by_key[site.history_key]), ml.item_run_key(by_key[site.key])]
    assert runs["value"] == ml.item_run_key(by_key[site.key])
    assert checkpoints["choices"] == [(
        f"Latest save · {renderers.epoch_words(by_key[site.key])}", site.key)]
    assert checkpoints["value"] == site.key
    labels = [label for label, _value in runs["choices"] + checkpoints["choices"]]
    assert all(len(label) <= renderers.OPTION_CHARS and label.count(" · ") <= 1
               for label in labels), labels
    assert len(set(labels)) == len(labels)
    # Nothing chosen: no run either, until the operator picks one.
    none_run, none_checkpoint = ml.parents_update(tp.model_parents(models), None, first=False)
    assert none_run["value"] is None and none_checkpoint["choices"] == []
    # What each file is leads its details, never its option: the fixture's
    # file without a model spec a record not readable, its trained child by
    # its task (`renderers.model_kind_text`).
    assert '<dd class="shep-v">Disease</dd>' in tp.model_html(page, models, site.key)
    history = tp.model_html(page, models, site.history_key)
    assert '<dd class="shep-v">Record not readable</dd>' in history


def test_the_reports_of_a_version_with_their_state_and_record(site):
    reports = tp.read_reports(LABEL)
    (run,) = reports["runs"]
    assert run["run_id"] == site.run_id and run["sha12"] == site.sha[:12]
    assert run["state"]["kept"] == "kept"
    assert run["state"]["registered"] == "cannot_be_checked", "the list knows no checkpoint file"
    assert run["record"]["checkpoint_digest"] == site.sha
    assert run["contest"]["state"] == "uncontested"
    assert "usage.jsonl" not in [r["run_id"] for r in reports["runs"]]


def test_the_selected_reports_registration_is_asked_beside_the_listed_checkpoint(site, monkeypatch):
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    reports = tp.read_reports(LABEL)
    refined = tp.refine_registration(page, reports, site.run_id, models)
    (run,) = refined["runs"]
    assert run["state"]["registered"] == "not_registered"
    assert run["state"]["checked_beside"] == site.key
    readonly = tp.read_page(_readonly())
    assert tp.refine_registration(readonly, reports, site.run_id, models) is reports
    # A ledger that cannot be read again leaves the list's answer, never a page error.
    import src.evaluation.registration as registration

    def unreadable(*_a, **_k):
        raise OSError("ledger unreadable")

    monkeypatch.setattr(registration, "report_state", unreadable)
    assert tp.refine_registration(page, reports, site.run_id, models) is reports


def test_runs_that_differ_are_described_and_each_can_be_registered(site, monkeypatch):
    """G3, G9 (K8-6r.2; policy rev 16, Decision 3): a group whose runs differ
    (``contested``, kept as a machine value) is described in plain words on the
    row and in the result, and Register stays enabled for every run of it: the
    description decides nothing. Each run's record is its own (looked up by its
    run id), and the row keeps one " · " (the skill's §5). HEAD: Register
    disabled with "runs disagree — investigate"."""
    other = _disagreeing_copy(site)
    page, _request = _page(monkeypatch)
    reports = tp.read_reports(LABEL)
    runs = {r["run_id"]: r for r in reports["runs"]}
    assert runs[other]["contest"]["state"] == "contested"
    assert "pooled_mrr" in runs[other]["contest"]["differing"]
    assert runs[other]["record"]["source_artifact_digest"] != (
        runs[site.run_id]["record"]["source_artifact_digest"])
    html = tp.reports_html(reports, other)
    # The table's State says the group's state; what differs is the result's.
    assert _report_rows(html)[other][3] == "runs differ"
    assert "on: pooled_mrr" in tp.result_html(reports, other, [], notes=[])
    # The site's runs record deterministic algorithms off: the difference is
    # described in neutral words, never amber and never "investigate".
    assert runs[other]["record"]["regime"]["deterministic_algorithms"] is False
    assert "investigate" not in html and "shep-tag-amber" not in html and "▲" not in html
    assert "disagree" not in html
    for run_id in (site.run_id, other):
        assert tp.register_state(page, reports, run_id) == (True, None)
        assert all(cell.count(" · ") <= 1 for cell in _report_rows(html)[run_id])
    # The R10 table's ▲ marks an overlap, its own condition (R-63), not this.
    result = without_r10(tp.result_html(reports, site.run_id, [], notes=[]))
    assert "<b>Runs under these conditions differ</b> on: pooled_mrr" in result
    assert "investigate" not in result and "▲" not in result
    assert "shep-tag-amber" not in result and "shep-notice-amber" not in result
    assert "each can be registered" in result and "block" not in result.replace(
        "shep-test-block", "")
    assert tp.notes_visible(reports, site.run_id)


def _twin_copy(site):
    """The kept report's bytes in a new run directory: a second run whose
    report is byte-identical (two CPU runs of one test)."""
    source = site.reports / site.run_id
    copy = site.reports / f"20261020T190000Z-{site.sha[:12]}-0be11e5d"
    shutil.copytree(source, copy)
    return copy.name


def test_the_route_and_the_page_read_a_byte_identical_run_by_its_own_record(site, monkeypatch):
    """G2, G3, A3 (K8-6r.2): after run 1 is registered from the page, a second
    run with byte-identical report bytes is not registered on
    ``GET /api/v1/cohorts`` nor on the Test page, whose Register stays enabled
    for it; "Already registered." applies to run 1's own record only, and
    registering run 2 adds record 2 naming it. HEAD: run 2 read registered by
    its bytes, and Register said "Already registered." for it."""
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    done, said = tp.register_selected(page, tp.read_reports(LABEL), models, "demo", site.run_id,
                                      request)
    assert done is True and said.startswith("Registered as record 1."), said
    twin = _twin_copy(site)

    # The route and the page's listing say the same of each run (the list
    # knows no checkpoint file, so neither reads "registered" for the twin).
    route = _cohorts_route_runs(LABEL)
    listed = {run["run_id"]: run for run in tp.read_reports(LABEL)["runs"]}
    for run_id in (site.run_id, twin):
        assert route[run_id]["state"] == listed[run_id]["state"]
        assert listed[run_id]["contest"]["state"] == "uncontested"
    assert listed[twin]["state"]["registered"] != "registered"
    # Asked beside the listed checkpoint Register would use, by run.
    reports = tp.refine_rows(page, tp.read_reports(LABEL), models)
    reports = tp.refine_registration(page, reports, twin, models)
    states = {run["run_id"]: run["state"] for run in reports["runs"]}
    assert states[site.run_id]["registered"] == "registered"
    assert states[twin]["registered"] == "not_registered"
    assert "holds no record naming this run" in states[twin]["why_registered"]
    assert tp.register_state(page, reports, site.run_id) == (False, "Already registered.")
    assert tp.register_state(page, reports, twin) == (True, None)

    done, said = tp.register_selected(page, reports, models, "demo", twin, request)
    assert done is True and said.startswith("Registered as record 2."), said
    ledger = json.loads((site.checkpoint.parent / "evaluations.json").read_text())
    assert [r["run_id"] for r in ledger["records"]] == [site.run_id, twin]
    assert len({r["source_artifact_digest"] for r in ledger["records"]}) == 1


def test_a_store_whose_runs_cannot_be_listed_says_why_register_is_off():
    """C5 (Decision 10): a version whose report store's runs cannot be listed
    has no report the page can confirm, so Register is off for an operator,
    saying why and that the command line registers a named report; a
    read-only request is told only that operator controls are off. A version
    that could not be read at all is not this case."""
    reports = {"runs": [], "version_sha256": "a" * 64,
               "problem": "the report store of 'v' could not be listed (PermissionError)"}
    allowed, why = tp.register_state({"access": {"allowed": True}}, reports, None)
    assert allowed is False and why.startswith(
        "Register disabled: the report store could not be listed, so a run's report cannot be "
        "confirmed here: the report store of 'v' could not be listed (PermissionError).")
    assert "scripts/record_evaluation.py --report" in why
    assert tp.register_state({"access": {"allowed": True}},
                             {**reports, "version_sha256": None}, None) == (
        False, "Register disabled: no report is listed yet.")


def test_a_run_whose_bytes_a_merged_record_holds_shows_the_possible_overlap(site):
    """A7 on the page (K8-6r.2): a run not registered whose bytes a record
    merged before each run had its own record holds says so beside its
    registration, as a possible overlap, never as registered; a registered
    run never shows it."""
    run = {r["run_id"]: r for r in tp.read_reports(LABEL)["runs"]}[site.run_id]
    state = {"kept": "kept", "registered": "not_registered", "checked_beside": "x/y/last.pt",
             "historical_overlap": [1]}
    import html as html_lib

    text = html_lib.unescape(re.sub(r"<[^>]+>", "", tp._record_html(run["record"], run, state)))
    assert "not registered (the ledger beside x/y/last.pt was read)" in text
    # The overlap on a line of its own, in sentences (the owner's decision of 2026-11-23).
    assert ("Possible overlap with record 1, merged before each run had its own record. "
            "Whether this run is one of them is not recorded. Registering adds this run's "
            "own record.") in text
    registered = {**state, "registered": "registered", "sequence": 2}
    assert "possible overlap" not in tp._record_html(run["record"], run, registered)


def _with_determinism(reports, flag):
    """The listing with every run's record saying deterministic algorithms
    `flag` (the setting is in the group's semantics, so all its runs share it)."""
    runs = []
    for run in reports["runs"]:
        record = run.get("record")
        if isinstance(record, dict):
            run = {**run, "record": {**record, "regime": {**record["regime"],
                                                          "deterministic_algorithms": flag}}}
        runs.append(run)
    return {**reports, "runs": runs}


def test_deterministic_runs_that_differ_are_flagged_for_investigation(site):
    """A6 (K8-6r.2; policy rev 16): when the runs of a group differ and the
    group records deterministic algorithms on, they were expected to agree, so
    the table row's State, the Result's chip and its block flag it in amber
    with ▲ and "— investigate". A group without them describes the same
    difference in neutral words: no amber, no ▲, no "investigate"; the row's
    State is then the run's own state ("● Registered #N" once registered), the
    Result's block still describing the difference. A determinism the record
    does not say (None) is neutral too, and the block does not claim it was
    off. Neither disables anything, and the notes stay shown either way."""
    from html import unescape

    other = _disagreeing_copy(site)
    for flag in (True, False, None):
        reports = _with_determinism(tp.read_reports(LABEL), flag)
        run = {r["run_id"]: r for r in reports["runs"]}[other]
        row = "".join(_report_rows(tp.reports_html(reports, other))[other])
        chip = tp._row_chip(run)
        # The R10 table's ▲ marks an overlap, its own condition (R-63), not this.
        result = without_r10(tp.result_html(reports, other, [], notes=[]))
        block = tp.block_html(run, [], [])
        assert tp.notes_visible(reports, other)
        assert "Runs under these conditions differ" in block
        if flag:
            assert chip == tp.r.warn_chip("runs differ")
            assert chip in tp.reports_html(reports, other)
            assert tp.r.warn_chip("Runs differ — investigate") in result
            assert block.startswith('<div class="shep-notice shep-notice-amber shep-test-block">')
            assert "Runs under these conditions differ — investigate." in unescape(block)
            assert ("Deterministic algorithms were on, so these runs were expected to agree."
                    in block)
        else:
            assert "investigate" not in row + chip + result and "▲" not in row + chip + result
            assert "amber" not in row + chip + block
            assert "shep-tag-amber" not in result and "shep-notice-amber" not in result
            assert chip == '<span class="shep-tag shep-tag-grey">runs differ</span>'
            assert '<span class="shep-tag shep-tag-grey">Runs differ</span>' in result
            assert block.startswith('<div class="shep-reason shep-reason-grey shep-test-block">')
            assert "<b>Runs under these conditions differ</b> on: pooled_mrr" in block
            if flag is False:
                assert ("Without deterministic algorithms, runs can differ. Every run is kept "
                        "as recorded and each can be registered." in block)
            else:   # not recorded: nothing is said about it either way
                assert "deterministic" not in block.lower()
                assert "Every run is kept as recorded and each can be registered." in block
            assert "An investigation note records what was looked at." in block
            assert "expected to agree" not in block
        # Once registered, the row's chip: amber "runs differ" stays first in a
        # deterministic group; otherwise the run's own "● Registered #3" (the
        # shared mark), and the Result's block still describes the difference.
        registered = {**run, "state": {**run["state"], "registered": "registered",
                                       "sequence": 3}}
        assert tp._row_chip(registered) == (chip if flag else (
            tp.r.status_mark("ok", "Registered #3", as_chip=True)))
        assert "on: pooled_mrr" in tp.block_html(registered, [], [])


def _uncomparable_copy(site):
    """A copy of the kept report under a new run id that `build_record` refuses
    (its metrics removed by hand) while its manifest still names its key."""
    source = site.reports / site.run_id
    copy = site.reports / f"20261020T170000Z-{site.sha[:12]}-0badf00d"
    shutil.copytree(source, copy)
    report = json.loads((copy / "report.json").read_text())
    del report["authoritative_metrics"]
    (copy / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return copy.name


def _unreadable_copy(site):
    """A run directory whose report.json is not JSON: its key cannot be told."""
    copy = site.reports / f"20261020T160000Z-{site.sha[:12]}-0dead0ff"
    copy.mkdir()
    (copy / "report.json").write_text("not a report")
    return copy.name


def _cohorts_route_runs(label):
    """``GET /api/v1/cohorts`` through the real route: each run of `label`'s
    store as the open route lists it to an operator, by run id (a request D8
    does not allow gets its reasons' paths as "a server path", B53)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routes import lists
    from tests.fixtures.operator_access import OPERATOR_BASE_URL, operator_mode

    app = FastAPI()
    app.include_router(lists.router, prefix="/api/v1")
    with pytest.MonkeyPatch.context() as patch:
        operator_mode(patch)
        body = TestClient(app, base_url=OPERATOR_BASE_URL).get("/api/v1/cohorts").json()
    (version,) = [v for v in body["versions"] if v["label"] == label]
    return {run["run_id"]: run for run in version["reports"]}


#: Each case: how the store is changed, and the D5 state every run with a report
#: then has (the run ids are the original's and the added copy's).
D5_CASES = {
    "uncontested": (lambda site, mp: None, "uncontested"),
    "contested": (lambda site, mp: _disagreeing_copy(site), "contested"),
    "a report that cannot be compared": (lambda site, mp: _uncomparable_copy(site),
                                         "cannot_be_checked"),
    "a report whose key cannot be told": (lambda site, mp: _unreadable_copy(site),
                                          "cannot_be_checked"),
    "a store that cannot be scanned": (lambda site, mp: mp.setattr(
        __import__("src.evaluation.report_store", fromlist=["x"]), "scan_report_store",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("unreadable"))),
        "cannot_be_checked"),
}


@pytest.mark.parametrize("case", list(D5_CASES))
def test_the_cohort_route_and_the_test_page_give_each_run_the_same_d5_state(
        site, monkeypatch, case):
    """EB-3: D5's state of each run's key is part of the one run listing both
    surfaces share (`server_lists._report_items`), so ``GET /api/v1/cohorts``
    and the Test page say the same for every run: uncontested, contested, or
    cannot be checked -- a report the scan could not compare, one whose key
    cannot be told (named by its run id, never recovered from a problem's
    words), and every run of a store that cannot be scanned; never no state
    for a run with a report. The open route carries no problem's words; the
    Test page keeps them for the operator. Catches a route without the state,
    and a page that judges it on its own."""
    change, expected = D5_CASES[case]
    added = change(site, monkeypatch)
    route = _cohorts_route_runs(LABEL)
    page = {run["run_id"]: run for run in tp.read_reports(LABEL)["runs"]}

    with_report = [site.run_id] + ([added] if isinstance(added, str) else [])
    assert sorted(route) == sorted(page) == sorted(with_report)
    for run_id in with_report:
        shown, listed = page[run_id]["contest"], route[run_id]["contest"]
        # An unreadable other report is its own run's problem only (C1,
        # K8-6r.2): it never marks another run.
        wanted = ("uncontested" if case == "a report whose key cannot be told"
                  and run_id == site.run_id else expected)
        assert shown["state"] == listed["state"] == wanted, (run_id, shown, listed)
        assert shown["differing"] == listed["differing"]
        assert shown["disagreeing_runs"] == listed["disagreeing_runs"]
        assert listed["problems"] == []
        assert bool(shown["problems"]) == (wanted == "cannot_be_checked")
    assert str(site.root) not in json.dumps([run["contest"] for run in route.values()])
    if case == "contested":
        assert "pooled_mrr" in route[added]["contest"]["differing"]
        assert sorted(route[added]["contest"]["disagreeing_runs"]) == sorted(with_report)
    operator = {"access": {"allowed": True}}
    if case == "a report whose key cannot be told":
        # C1 (K8-6r.2): the broken run is listed with its own reason only; the
        # valid run reads uncontested and its Register stays enabled.
        assert page[added]["key"] is None and "could not be read" in page[added]["unbuilt"]
        assert page[added]["contest"]["problems"] == [page[added]["unbuilt"]]
        reports = tp.read_reports(LABEL)
        assert tp.register_state(operator, reports, site.run_id) == (True, None)
        # Not JSON, so the helper reads it as no report kept: its own reason.
        assert tp.register_state(operator, reports, added) == (
            False, "Register disabled: no report was kept in this run.")
        block = tp.block_html(page[added], [], [])
        assert "This run&#x27;s report cannot be compared" in block and "marks no other run" in block
        assert "Comparison incomplete" not in block
    if case == "a report that cannot be compared":
        # C2 (K8-6r.2): the group's description names that file as
        # cannot_be_checked; the complete run's Register stays enabled.
        assert page[added]["key"] == page[site.run_id]["key"]
        assert "cannot be compared" in page[added]["unbuilt"]
        assert page[added]["record"] is None and page[site.run_id]["record"] is not None
        reports = tp.read_reports(LABEL)
        assert tp.register_state(operator, reports, site.run_id) == (True, None)
        assert tp.register_state(operator, reports, added)[1].startswith(
            "Register disabled: this report cannot be registered:")
    if case == "a store that cannot be scanned":
        # C5 (Decision 10): no report of the store can be confirmed here, so
        # Register is off saying why and where a run is registered instead.
        reports = tp.read_reports(LABEL)
        allowed, why = tp.register_state(operator, reports, site.run_id)
        assert allowed is False and why.startswith(
            "Register disabled: the report store could not be listed, so this run's report cannot be "
            "confirmed here: its report store could not be read (RuntimeError).")
        assert "Register it from the command line by naming its report" in why


@pytest.mark.parametrize("replacement", ["damaged", "another report"])
def test_a_report_replaced_between_the_scan_and_its_state_is_never_shown_uncontested(
        site, monkeypatch, replacement):
    """The verification of EB-3 (R-1): a run's kept/registered state and its
    key's D5 state come from two reads of its ``report.json`` (the store's scan,
    then the registration's helper). A report replaced between them -- damaged,
    or another report under the same key -- is ``cannot_be_checked`` on
    ``GET /api/v1/cohorts`` and on the Test page, saying so for the operator,
    never the scan's "uncontested" for bytes the state did not read; the page
    then gives the run no key (so no note is filed under the scan's), no record
    and no Register. The same store unreplaced is the control. Catches a
    listing that takes the scan's state without checking which bytes it read."""
    import src.evaluation.report_store as report_store

    report = site.reports / site.run_id / "report.json"
    original = report.read_bytes()
    assert _cohorts_route_runs(LABEL)[site.run_id]["contest"]["state"] == "uncontested"
    control = {run["run_id"]: run for run in tp.read_reports(LABEL)["runs"]}[site.run_id]
    assert control["contest"]["state"] == "uncontested" and control["key"] is not None

    if replacement == "damaged":
        replaced = b"not a report"
    else:
        changed = json.loads(original)
        changed["authoritative_metrics"]["pooled_mrr"] += 0.0004
        replaced = json.dumps(changed, indent=2, sort_keys=True).encode()
    real = report_store.scan_report_store

    def scan_then_replace(version, **kwargs):
        report.write_bytes(original)  # each surface's scan reads the original
        try:
            return real(version, **kwargs)
        finally:
            report.write_bytes(replaced)

    monkeypatch.setattr(report_store, "scan_report_store", scan_then_replace)
    listed = _cohorts_route_runs(LABEL)[site.run_id]
    reports = tp.read_reports(LABEL)
    shown = {run["run_id"]: run for run in reports["runs"]}[site.run_id]

    for run in (listed, shown):
        if replacement == "damaged":
            assert run["state"]["kept"] == "not_kept" and run["state"]["report_sha256"] is None
        else:
            assert run["state"]["kept"] == "kept"
            assert run["state"]["report_sha256"] == hashlib.sha256(replaced).hexdigest()
        assert run["contest"]["state"] == "cannot_be_checked", run["contest"]
    assert listed["contest"]["problems"] == []
    assert any("replaced between the two reads" in p for p in shown["contest"]["problems"])
    assert shown["key"] is None and shown["record"] is None
    assert tp.register_state({"access": {"allowed": True}}, reports, site.run_id)[0] is False
    assert not tp.notes_visible(reports, site.run_id)


def test_a_store_that_cannot_be_scanned_reads_the_same_for_its_records_and_its_runs(
        site, monkeypatch):
    """The verification of EB-3 (R-2): the record-level D5 state (Models, the
    Diagnosis card, ``GET /api/v1/evaluations/loaded-model``) and the run-level
    one (the Test page, ``GET /api/v1/cohorts``) turn a scan into a shown state
    through one layer, so a store whose scan fails is caught and worded once:
    the same words for the record and the run it came from, and none on an open
    surface. Catches a second copy of the failure handling with its own words."""
    import src.evaluation.report_store as report_store
    from src.api.services.record_states import CohortVersions, contest_states

    record = {run["run_id"]: run for run in tp.read_reports(LABEL)["runs"]}[site.run_id]["record"]
    assert record is not None

    def fails(*args, **kwargs):
        raise RuntimeError("unreadable")

    monkeypatch.setattr(report_store, "scan_report_store", fails)
    run = {run["run_id"]: run for run in tp.read_reports(LABEL)["runs"]}[site.run_id]
    (state,) = contest_states([record], CohortVersions.listed()).values()
    (open_state,) = contest_states([record], CohortVersions.listed(), operator=False).values()

    assert run["contest"]["state"] == state["state"] == "cannot_be_checked"
    assert run["contest"]["problems"] == state["problems"] == [
        "its report store could not be read (RuntimeError)"]
    assert open_state["problems"] == _cohorts_route_runs(LABEL)[site.run_id]["contest"][
        "problems"] == []


def test_the_result_shows_aggregates_and_states_never_a_case_or_a_path(site):
    reports = tp.read_reports(LABEL)
    html = tp.result_html(reports, site.run_id, tp.path_roots())
    # A disease task report: its tiles, and its coverage over the version's
    # import-eligible cases, of which it ranks those whose truth is in the
    # model's graph view (the report's `n_ranked`).
    ranked = json.loads((site.reports / site.run_id / "report.json").read_text())["n_ranked"]
    assert 0 < ranked <= 10
    assert 'tile-l">Pooled MRR<' in html and 'tile-l">Coverage<' in html
    # The coverage in its one wording (R-82): what it counted, then how the
    # data stands under it, on lines of their own, never a chain of " · ".
    assert (f"<div><b>{ranked} of 10</b> import-eligible cases ranked by the model</div>"
            '<div class="shep-sub">13 source cases, 3 excluded at import: ground truth absent '
            "2, phenotype unmappable 1</div>") in html
    assert "Validation MRR" not in html, "Trainer validation and a test result never share a place"
    assert "case-" not in html and str(site.root) not in html
    assert "synthetic</b> — a generated stand-in" in html
    assert "Graph files match the workspace manifest (R1)" in html


def _native_row(html):
    """The value of the result's one "Native build" row."""
    assert html.count('<dt class="shep-k">Native build</dt>') == 1
    match = re.search(r'<dt class="shep-k">Native build</dt><dd class="shep-v">(.*?)</dd>',
                      html)
    return match.group(1)


def _native_lines(html):
    return [_plain(line) for line in re.findall(r"<div>(.*?)</div>", _native_row(html))]


def test_the_result_names_its_native_build_as_a_lead_and_one_line_per_component(site):
    """research-comparison Stage 2b: the record's native build identity in the
    run environment's one projection and wording, as Models shows it, here as
    a lead (the route once, and the count) and one line per component: in
    this column the one line ran to four (the webui-design skill's §4, seen
    at 1850 x 960 in the 2b review). A record whose environment names no
    components (recorded before) reads "not recorded"; what it shows is
    escaped."""
    from src.utils.run_environment import (
        PYG_NATIVE_KIND,
        component_facts,
        native_identity,
        native_identity_lines,
        native_identity_text,
    )

    reports = tp.read_reports(LABEL)
    run = tp.find(reports["runs"], "run_id", site.run_id)
    record = run["record"]
    components = record["environment"]["facts"]["components"]
    assert components, "the measured run's environment names its components"

    lines = _native_lines(tp._record_html(record, run, {}))
    identity = native_identity(components)
    assert lines == native_identity_lines(identity)
    assert len(lines) == len(components) + 1
    assert lines[0].endswith(f" · {len(components)} components")
    # The same words as the one line Models shows, with the route said once.
    assert native_identity_text(identity) == f"{lines[0].split(' · ')[0]} — " + ", ".join(
        lines[1:])
    assert [line.split(" ")[0] for line in lines[1:]] == sorted(components)

    facts = {key: value for key, value in record["environment"]["facts"].items()
             if key != "components"}
    older = {**record, "environment": {**record["environment"], "facts": facts}}
    assert _native_lines(tp._record_html(older, run, {})) == ["not recorded"]
    hostile = {**record, "environment": {**record["environment"], "facts": {
        **facts, "components": {"<img src=x onerror=alert(1)>": component_facts(
            PYG_NATIVE_KIND, version="<b>1</b>")}}}}
    row = _native_row(tp._record_html(hostile, run, {}))
    assert "<img" not in row and "<b>" not in row and "&lt;img src=x" in row


def test_the_result_shows_the_one_readers_view_of_its_record(site, monkeypatch):
    """Stage 8b review (one reader of a result for every surface): every field
    the Result shows but the metric values is the one reader's view of the
    record (`record_views.result_view`, what the results service and Models
    read by) and its R10 the one rendering (`renderers.r10_table_html`), never the
    page's own read of the raw record: a view that says otherwise than the
    record is what is shown."""
    import html as html_lib

    from src.api.services import record_views

    reports = tp.read_reports(LABEL)
    run = tp.find(reports["runs"], "run_id", site.run_id)
    record = run["record"]
    assert record["cohort"]["kind"] == "supplied"
    real = record_views.result_view

    class Said:
        def __init__(self, view):
            self.view = view

        def model_dump(self, mode="json"):
            dump = self.view.model_dump(mode=mode)
            dump["cohort"]["label"] = "label-from-the-view"
            dump["environment_class"] = "class-from-the-view"
            dump["score_semantics"] = "semantics-from-the-view"
            dump["dataset_origin"] = "files"
            dump["deterministic_algorithms"] = None
            dump["native_identity"] = None
            dump["r10"]["training"]["evidence"]["state"] = "not_recorded"
            return dump

    monkeypatch.setattr(record_views, "result_view", lambda value: Said(real(value)))
    raw = tp._record_html(record, run, {})
    text = html_lib.unescape(re.sub(r"<[^>]+>", " ", raw))
    assert "label-from-the-view (role" in text
    assert f"{record['cohort']['label']} (role" not in text
    assert "class-from-the-view" in text and "semantics-from-the-view" in text
    assert " ".join(text.split()).count("Deterministic algorithms not recorded") == 1
    assert record["dataset_record"]["origin"] != "files" and "files — built from" in text
    assert _native_lines(raw) == ["not recorded"]
    table = renderers.r10_table_html(
        {**record["r10"], "training": {**record["r10"]["training"],
                                       "evidence": {"state": "not_recorded"}}})
    assert table in raw
    assert r10_cells(raw)["Training"]["evidence"] == "evidence not recorded by the checkpoint"


def test_a_result_of_another_or_no_mode_is_shown_as_recorded(site):
    """A record whose mode this version does not describe, or whose mode cannot
    be read, is named for what it is by the one display for every mode (R2):
    its own label, no tiles and no claim of what its cases are, its metrics as
    recorded in one "Recorded metrics" row; never another mode's words. A
    record whose mode cannot be read is one the one reader does not read as a
    result (research comparison Stage 8b: the result view the results service
    and Models read by): it says so with the field named, never another mode's
    words. Each report row names its mode first (ui_api-5), a kept Mode C
    report's as "mode C, not described by this version", and what the Run
    panel fixes names no measurement: the Measurement radio offers the
    protocols."""
    import html as html_lib

    c_run = _keep_mode_c_report(site)
    reports = tp.read_reports(LABEL)
    run = {r["run_id"]: r for r in reports["runs"]}[c_run]
    record = run["record"]
    changed = {**record, "scoring": {**record["scoring"], "mode": "E"}}
    raw = tp._record_html(changed, run, {})
    text = html_lib.unescape(re.sub(r"<[^>]+>", "", raw))
    assert "Mode E (a measurement mode this version does not describe)" in text
    assert "Mode C" not in text and "shep-rec-tile-l" not in raw
    assert "source cases ranked" not in text and "not described for this mode" in text
    assert '<dt class="shep-k">Recorded metrics</dt>' in raw
    assert '<div class="shep-rec-ln"><code>untruncated_mrr</code> 0.5</div>' in raw
    unread = {**record, "scoring": {**record["scoring"], "mode": None}}
    text = html_lib.unescape(re.sub(r"<[^>]+>", "", tp._record_html(unread, run, {})))
    assert text.startswith("This report's record cannot be read as a result") and \
        "scoring.mode" in text
    assert "Mode C" not in text and "Recorded metrics" not in text
    assert _report_rows(tp.reports_html(reports, None))[c_run][2] == (
        "Mode C, not described by this version")
    fixed = tp.fixed_html({})
    assert "Mode C" not in fixed and "Mode D" not in fixed and "Measurement" not in fixed
    # The server's fixed rows sit among the Run panel's fields: their keys are
    # drawn as its field labels (R-16, one label style per page).
    assert fixed.startswith(f'<dl class="shep-kv {tp.r.FIELD_ROWS}">')


def test_a_run_without_a_report_says_none_was_kept(site):
    interrupted = site.reports / f"20261020T190000Z-{site.sha[:12]}-0000dead"
    interrupted.mkdir()
    reports = tp.read_reports(LABEL)
    run = tp.find(reports["runs"], "run_id", interrupted.name)
    assert run["state"]["kept"] == "not_kept"
    # The table's row says its state; its measurement, never read from a
    # report that is not there, is Unknown; why, in the Result, redacted.
    row = _report_rows(tp.reports_html(reports, None))[interrupted.name]
    assert row[2:] == ["○ Unknown", "no report kept"]
    roots = tp.path_roots()
    result = _plain(tp.result_html(reports, interrupted.name, roots))
    assert "No report was kept in this run directory: there is no <registry>/" in result
    assert "report.json" in result and str(site.registry) not in result
    assert tp.register_state({"access": {"allowed": True}}, reports, interrupted.name) == (
        False, "Register disabled: no report was kept in this run.")


def _plain(html):
    import html as html_lib

    return " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", html)).split())


@pytest.mark.parametrize("damage", ["does not parse", "other bytes"])
def test_a_report_json_that_is_there_is_never_said_to_be_absent(site, damage):
    """K8-6e.1 (policy rev 16; the webui-design skill keeps "cannot be read"
    apart from absent): a run directory whose report.json is there but is not
    a report, or holds bytes other than the run published, keeps no report
    under the publication contract (``not_kept``), and the Test page's row
    says so and its Result why, in the registration helper's own words,
    redacted, never
    "report.json ... is not there". HEAD: both said the fixed sentence "is not
    there (an interrupted or refused run)", and the Result then contradicted
    itself with the comparison's "could not be read"."""
    from src.evaluation.registration import report_state

    roots = tp.path_roots()
    if damage == "does not parse":
        added = _unreadable_copy(site)
        reports = tp.read_reports(LABEL)
        run = tp.find(reports["runs"], "run_id", added)
        wanted = "report.json is not a report: it does not parse as UTF-8 JSON"
    else:
        # The refined state (`_refined_state`): the bytes the list read, then
        # replaced before the registration's helper read them again.
        reports = tp.read_reports(LABEL)
        run = tp.find(reports["runs"], "run_id", site.run_id)
        report = site.reports / site.run_id / "report.json"
        report.write_bytes(report.read_bytes() + b"\n")
        fresh = report_state(report, expected_sha256=run["state"]["report_sha256"])
        run = {**run, "record": None,
               "state": {**run["state"], "kept": fresh.kept, "why_kept": fresh.why_kept}}
        reports = {**reports, "runs": [run]}
        added = site.run_id
        wanted = "the run's report was replaced or changed"
    assert run["state"]["kept"] == "not_kept"
    row = _report_rows(tp.reports_html(reports, None, roots=roots))[added]
    result = _plain(tp.result_html(reports, added, roots))
    assert wanted in result, result
    assert "is not there" not in result and "is not there" not in " ".join(row)
    assert "<registry>/" in result and str(site.registry) not in result
    assert row[3] == "no report kept"
    assert "No report was kept in this run directory: " in result
    assert tp.register_state({"access": {"allowed": True}}, reports, added) == (
        False, "Register disabled: no report was kept in this run.")


def test_start_waits_for_every_choice_and_says_why(site, monkeypatch):
    _child, child = _put_bundle(site)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    binding = tp.workspace_binding(site.demo, tp.find(page["workspaces"], "name", "demo"))
    state = lambda ws, label, key, gpu=("none", ""), protocol="disease": (  # noqa: E731
        tp.start_state(page, models, ws, label, key, binding, gpu, protocol=protocol))
    assert state(None, LABEL, child) == (False, "Start disabled: choose a workspace.")
    assert state("demo", None, child) == (False, "Start disabled: choose a data version.")
    # NEW-test-1: the choice is named by its two selects, the one place the
    # page asks for it (it said "choose a model", a control it no longer has).
    assert state("demo", LABEL, None) == (False, tp.CHOOSE_A_CHECKPOINT) == (
        False, "Start disabled: choose the run, then the checkpoint to test.")
    assert tp.model_html(page, models, None) == ""
    # No measurement chosen (the radio's value None) is said, never filled in.
    assert state("demo", LABEL, child, protocol=None) == (
        False, "Start disabled: choose a measurement.")
    assert state("demo", LABEL, child) == (True, None)
    busy = state("demo", LABEL, child, ("busy", "GPU job: training (job runner)"))
    assert busy == (False, "Start disabled: GPU job: training (job runner).")
    # This service's CUDA inference: the chip's words and where to stop it,
    # never a job's (review of A3b: inference is not a training or test job).
    serving = state("demo", LABEL, child, ("busy", "GPU: inference (this service); stop it with "
                                                   "Stop inference on Diagnosis"))
    assert serving == (False, "Start disabled: GPU: inference (this service); stop it with Stop "
                              "inference on Diagnosis.")
    another = dict(binding, graph={**binding["graph"], "kg": "0" * 64})
    allowed, why = tp.start_state(page, models, "demo", LABEL, child, another, ("none", ""))
    assert allowed is False and "mapped against another graph" in why
    # The fixture's file without a model spec: listed, never measured, with
    # the record reader's one phrase (N2).
    assert state("demo", LABEL, site.history_key) == (
        False, f"Start disabled: Record not readable ({WRITTEN_BEFORE_SPEC}): not measured here. "
               "Choose a model with a readable record.")


def test_a_lone_testable_checkpoint_is_chosen_as_the_shared_list_chooses_one(site, monkeypatch):
    """RJ-61b: the model choice preselected nothing even where there was
    nothing to choose between. A workspace that lists one checkpoint, which
    can be tested, has it chosen by the shared list's one rule
    (`model_list.lone_key`, R-27, which the Models list uses too), Run and
    Checkpoint drawn on it and Start's reason past the choice; of two or
    more the page still picks none itself, and a lone file it cannot test
    is not chosen."""
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    child = tp.find_model(models, site.key)
    history = tp.find_model(models, site.history_key)
    assert tp.testable(child) and not tp.testable(history)
    assert len(models["items"]) >= 2 and tp.chosen_model(models, None) is None
    assert tp.chosen_model(models, site.key) == site.key
    lone = {**models, "items": [child]}
    assert tp.chosen_model(lone, None) == site.key
    assert tp.chosen_model({**models, "items": [history]}, None) is None
    # The model list's own lone rule is the same function (no copy of it).
    assert ml.lone_key([child], tp.testable) == site.key
    assert ml.lone_key([child, child], tp.testable) is None
    # Drawn by the page's open.
    monkeypatch.setattr(tp, "read_job", lambda **k: {"status": "idle", "inputs": None,
                                                    "outcome": None, "log": []})
    monkeypatch.setattr(tp, "gpu_line", lambda **k: ("none", "GPU jobs: none"))
    monkeypatch.setattr(tp, "read_models", lambda workspace, request, **k: dict(lone))
    demo, _ctx, _registry = _page_component()
    refresh = _named(demo, "_refresh")
    out = dict(zip([b.elem_id for b in refresh.outputs],
                   refresh.fn(None, None, None, None, {}, None, None, request), strict=True))
    assert out["shep-test-model"]["value"] == site.key
    assert out["shep-test-model-run"]["value"] == ml.item_run_key(child)
    assert "choose the run" not in out["shep-test-start-reason"]
    assert "Model fingerprint" in out["shep-test-model-info"]


def test_the_register_option_and_button_say_one_act_and_when():
    """R-79: "Register the result" (a run option) and "Register this report"
    (a button) were two near-identical names. The act has one name, Register;
    the option says when (as the run ends), the button which report."""
    assert tp.RECORD_LABEL == "Register when the run ends"
    assert tp.FIELD_HELP["shep-test-record"][0] == tp.RECORD_LABEL
    assert tp.GATED_LABELS["register"] == "Register this report"
    assert "Register this report" in tp.FIELD_HELP["shep-test-record"][1]
    assert "the result" not in tp.RECORD_LABEL


def test_a_version_whose_kept_file_changed_is_not_bound_and_start_says_why(site, monkeypatch):
    """UI-12 (B72-T3): "Bound to this graph" says two facts, the version's
    record names this graph and its kept files were read as the bytes that
    record binds, with when. One byte of its samples.json changed (on the
    site's copy): its cohort.json still names this graph, but the version
    no longer reads bound, the refusal under it names the file and both
    digests (no path), and Start is off with a reason pointing there. The
    run's own content refusal is unchanged. At ba6e85e the listing read
    cohort.json alone: "● Bound to this graph" and Start enabled, and only
    the run refused."""
    _child, child = _put_bundle(site)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    binding = tp.workspace_binding(site.demo, tp.find(page["workspaces"], "name", "demo"))
    version = tp.find(page["versions"], "label", LABEL)
    assert version["contents"]["state"] == "verified"
    assert tp.binding_state(version, binding) == ("bound", "bound to this graph")
    shown = tp.version_html(page, LABEL, binding)
    checked = version["contents"]["checked_at"].replace("T", " ").replace("Z", " UTC")
    assert (tp.r.status_mark("ok", "Bound to this graph")
            + f' <span class="shep-sub">{tp.r.word_groups("files checked", checked)}</span>'
            ) in shown
    assert tp.start_state(page, models, "demo", LABEL, child, binding, ("none", ""),
                          protocol="disease") == (True, None)

    samples = site.registry / "versions" / LABEL / "samples.json"
    data = bytearray(samples.read_bytes())
    data[len(data) // 2] ^= 0x01
    samples.write_bytes(bytes(data))
    page, request = _page(monkeypatch)
    version = tp.find(page["versions"], "label", LABEL)
    assert version["manifest_sha256"] == site.version_sha      # its record is unchanged
    assert version["contents"]["state"] == "differs"
    assert tp.binding_state(version, binding) == (
        "changed", "not bound: its files changed after the import")
    shown = tp.version_html(page, LABEL, binding)
    assert "Bound to this graph" not in shown
    assert tp.r.status_mark("error", "Not bound: its files changed after the import") in shown
    reason = version["contents"]["files"][0]["reason"]
    assert reason.startswith("the version's samples.json holds ") and str(site.root) not in reason
    assert tp.r.status_mark("error", tp.CHANGED_FILES.format(reason=reason)) in shown
    assert tp.start_state(page, models, "demo", LABEL, child, binding, ("none", ""),
                          protocol="disease") == (
        False, "Start disabled: the data version's files changed after its import (see Data).")


# --------------------------------------------------------------------------- the operator controls
def test_start_is_refused_for_a_read_only_request_and_starts_nothing(site, monkeypatch):
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    manager = FakeManager()
    result = tp.start_test_run(page, models, "demo", LABEL, site.key, True, False, _readonly(),
                               manager=manager)
    assert result["success"] is False and result["error"].startswith("Refused: ")
    assert manager.started == []


def test_start_passes_the_digests_shown_and_the_servers_paths(site, monkeypatch):
    evidence = site.root / "evidence"
    evidence.mkdir()
    monkeypatch.setenv("SHEPHERD_TRAINING_EVIDENCE_DIRS", str(evidence))
    from src.utils.fingerprint import file_sha256

    path, child = _put_bundle(site)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    manager = FakeManager()
    result = tp.start_test_run(page, models, "demo", LABEL, child, True, True, request,
                               protocol="disease", manager=manager)
    assert result["success"] is True
    (inputs,) = manager.started
    assert inputs.checkpoint_sha256 == file_sha256(path)
    assert inputs.version_sha256 == site.version_sha
    assert inputs.registry == site.registry.resolve() and inputs.version_label == LABEL
    assert inputs.workspace == site.demo.resolve()
    assert inputs.checkpoint.resolve() == path.resolve()
    # The measurement asked, apart from the model's stage and task and the
    # version's counts, which are the listing's: never the client's.
    assert inputs.protocol == "disease"
    assert (inputs.learning_stage, inputs.task) == ("finetune", "disease")
    assert inputs.version_counts == tp.find(page["versions"], "label", LABEL)["counts"]
    assert inputs.evidence_dirs == (evidence.resolve(),)
    assert inputs.record is True and inputs.deterministic is True


def _start_inputs(site, monkeypatch):
    """A listed disease child, the page and the session's models and binding,
    as Start reads them."""
    _path, child = _put_bundle(site)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    binding = tp.workspace_binding(site.demo, tp.find(page["workspaces"], "name", "demo"))
    return page, request, models, binding, child


def test_an_evidence_directory_that_cannot_be_checked_never_disables_start(site, monkeypatch):
    """A8, P2-B of the K8-6 completion review (2026-11-08). The setting names
    an evidence directory whose own stat fails (behind a parent the service
    account may not search) and one that is fine. Start stays enabled; the
    R10 evidence row lists both and, on a line of its own, in amber with ▲ and
    words (readable without colour), which one cannot be checked and why, its
    path redacted and the reason escaped; the start passes both, in order, as
    `--training-evidence` in the job's command line. HEAD: Start disabled
    ("the evidence directories setting is refused") and the start refused."""
    import errno
    import os

    from src.api.services.training_manager import build_test_argv
    from tests.unit.test_r10 import _stat_fails

    locked, good = (site.root / "locked-evidence").resolve(), (site.root / "evidence").resolve()
    locked.mkdir()
    good.mkdir()
    monkeypatch.setenv("SHEPHERD_TRAINING_EVIDENCE_DIRS", os.pathsep.join([str(locked),
                                                                          str(good)]))
    _stat_fails(monkeypatch, locked, PermissionError(errno.EACCES, "Permission <denied>"))
    page, request, models, binding, child = _start_inputs(site, monkeypatch)

    assert tp.start_state(page, models, "demo", LABEL, child, binding, ("none", ""),
                          protocol="disease") == (True, None)
    row = tp.evidence_html(page, "demo")
    assert ("<div><code>locked-evidence</code>, <code>evidence</code> (the server's setting)"
            '</div><div><span class="shep-st shep-st-warn"><span class="shep-st-shape" aria-hidden="true">▲</span><span class="shep-st-sp">\u00a0</span><span class="shep-st-w">Cannot be checked: '
            "&lt;evidence 1&gt; (PermissionError: Permission &lt;denied&gt;). R10 cannot look "
            "in it: inputs found nowhere else read unverifiable, naming it.</span></span></div>"
            "</dd>") in row
    assert str(site.root) not in row and "<denied>" not in row
    assert row.startswith(f'<dl class="shep-kv {tp.r.FIELD_ROWS}">')         # a field's label (R-16)
    manager = FakeManager()
    result = tp.start_test_run(page, models, "demo", LABEL, child, False, False, request,
                               protocol="disease", manager=manager)
    assert result["success"] is True
    (inputs,) = manager.started
    assert inputs.evidence_dirs == (locked, good)
    argv = build_test_argv(inputs, None, site.root / "outcome.json")
    assert [argv[i + 1] for i, arg in enumerate(argv) if arg == "--training-evidence"] == [
        str(locked), str(good)]


def test_an_evidence_entry_that_is_not_a_directory_still_disables_start(site, monkeypatch):
    """A8 (guard). An entry known not to be a directory (here missing) still
    refuses the whole setting: Start is disabled saying so, the start is
    refused, and nothing is started."""
    monkeypatch.setenv("SHEPHERD_TRAINING_EVIDENCE_DIRS", str(site.root / "missing"))
    page, request, models, binding, child = _start_inputs(site, monkeypatch)

    allowed, why = tp.start_state(page, models, "demo", LABEL, child, binding, ("none", ""),
                                  protocol="disease")
    assert allowed is False
    assert why.startswith("Start disabled: the evidence directories setting is refused: "
                          "The training evidence folders set at start "
                          "(Ref: SHEPHERD_TRAINING_EVIDENCE_DIRS) are refused: evidence "
                          "directory 1 ")
    assert "does not exist or is not a directory" in why
    manager = FakeManager()
    result = tp.start_test_run(page, models, "demo", LABEL, child, False, False, request,
                               protocol="disease", manager=manager)
    assert result["success"] is False and "does not exist or is not a directory" in result["error"]
    assert manager.started == []


def test_start_runs_only_what_the_session_was_shown(site, monkeypatch):
    path, child = _put_bundle(site)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    manager = FakeManager()
    # a key that is not in the session's list, a path, and another workspace
    for ws, key in (("demo", "gatv2/x/y.pt"), ("demo", str(path)), ("other", child)):
        result = tp.start_test_run(page, models, ws, LABEL, key, False, False, request,
                                   manager=manager)
        assert result["success"] is False and "Nothing was started." in result["error"]
        assert "Choose a model the server lists in this workspace" in result["error"]
    result = tp.start_test_run(page, models, "demo", "not-a-version", child, False, False,
                               request, manager=manager)
    assert result["success"] is False and manager.started == []
    # A listed workspace holding the same run and file: the model was listed
    # for "demo", so it is not started in "copy" (where it was never shown).
    shutil.copytree(site.demo, site.ws / "copy", symlinks=True)
    page = tp.read_page(request)
    assert "copy" in page["directories"]
    result = tp.start_test_run(page, models, "copy", LABEL, child, False, False, request,
                               manager=manager)
    assert result["success"] is False and "in this workspace" in result["error"]
    assert manager.started == []


def test_a_checkpoint_the_server_no_longer_lists_is_refused_at_start(site, monkeypatch):
    path, child = _put_bundle(site)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    path.unlink()
    manager = FakeManager()
    result = tp.start_test_run(page, models, "demo", LABEL, child, False, False, request,
                               protocol="disease", manager=manager)
    assert result["success"] is False and manager.started == []
    assert result["error"].endswith("Nothing was started.")
    assert "this checkpoint" not in result["error"].lower(), "refused by the list, not the record"


def test_cancel_is_refused_for_a_read_only_request(monkeypatch):
    manager = FakeManager()
    assert tp.cancel_test_run(_readonly(), manager=manager)["success"] is False
    assert manager.cancelled == 0
    assert tp.cancel_test_run(operator_request(monkeypatch), manager=manager)["success"] is True
    assert manager.cancelled == 1


def test_register_files_the_record_beside_the_listed_checkpoint(site, monkeypatch):
    from src.evaluation.cohort_registry import resolve_version
    from src.evaluation.report_store import read_usage_history

    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    reports = tp.read_reports(LABEL)
    done, said = tp.register_selected(page, reports, models, "demo", site.run_id, request)
    assert done is True and said.startswith("Registered as record 1.")
    ledger = json.loads((site.checkpoint.parent / "evaluations.json").read_text())
    assert [r["checkpoint_digest"] for r in ledger["records"]] == [site.sha]
    entries, _ = read_usage_history(resolve_version(site.registry, LABEL))
    assert [e["kind"] for e in entries][-1] == "registered"
    again, said = tp.register_selected(page, tp.read_reports(LABEL), models, "demo",
                                       site.run_id, request)
    assert again is True and "Already registered as record 1" in said


def test_a_registered_run_whose_report_was_replaced_says_so_before_register(site, monkeypatch):
    """A5 on the Test page (review of K8-6r.2): a run registered from the page
    whose report.json is then replaced by other valid bytes reads "not
    registered", and the page says why before Register is pressed: the row and
    the Result name the replacement, the Result with the registration's own
    words (the record and both digests), redacted. Pressing Register is refused
    with the same words; the ledger is unchanged. Before the fix the page said
    only "not registered"."""
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    done, said = tp.register_selected(page, tp.read_reports(LABEL), models, "demo",
                                      site.run_id, request)
    assert done is True and said.startswith("Registered as record 1."), said
    ledger = site.checkpoint.parent / "evaluations.json"
    before = ledger.read_bytes()
    path = site.reports / site.run_id / "report.json"
    data = json.loads(path.read_text())
    data["manifest"]["data_dir"] = str(data["manifest"].get("data_dir")) + "-moved"
    path.write_text(json.dumps(data, indent=2, sort_keys=True))

    reports = tp.refine_rows(page, tp.read_reports(LABEL), models)
    reports = tp.refine_registration(page, reports, site.run_id, models)
    run = tp.find(reports["runs"], "run_id", site.run_id)

    assert run["state"]["registered"] == "not_registered"
    assert run["state"]["replaced_after_registration"] is True
    assert _report_rows(tp.reports_html(reports, None))[site.run_id][3] == "not registered"
    text = " ".join(re.sub(r"<[^>]+>", " ", tp.result_html(reports, site.run_id, page["roots"],
                                                            notes=[])).split())
    assert "Not registered: " in text and "is already registered as record 1 from report" in text
    assert "so this file was replaced after it was registered" in text
    assert str(site.checkpoint.parent) not in text
    done, said = tp.register_selected(page, reports, models, "demo", site.run_id, request)
    assert done is False and "is already registered as record 1 from report" in said
    assert ledger.read_bytes() == before


def test_a_registration_the_listed_checkpoints_ledger_cannot_tell_gives_its_reason():
    """Review of K8-6r.2: when the ledger beside the listed checkpoint was read
    and cannot tell (here a record that lost its sequence), the Result gives
    the registration's own reason, redacted and escaped; the list's own
    "cannot be checked" (no checkpoint looked in) keeps its one sentence and
    adds nothing."""
    roots = [("/srv/ws", "<workspaces>")]
    state = {"kept": "kept", "registered": "cannot_be_checked",
             "checked_beside": "gatv2/r1/last.pt",
             "why_registered": ("/srv/ws/demo/runs/gatv2/r1/checkpoints/evaluations.json: the "
                                "ledger's record 0 has no registration sequence, so the next "
                                "sequence cannot be allocated safely")}

    html = tp._registration_reason_html(state, roots)

    assert ("Registration cannot be checked: &lt;workspaces&gt;/demo/runs/gatv2/r1/checkpoints/"
            "evaluations.json: the ledger&#x27;s record 0 has no registration sequence") in html
    assert "/srv/ws" not in html
    assert tp._registration_reason_html({**state, "checked_beside": None}, roots) == ""
    assert tp._registration_reason_html({**state, "registered": "not_registered"}, roots) == ""


def test_register_is_refused_for_a_read_only_request(site, monkeypatch):
    from src.evaluation import registration

    monkeypatch.setattr(registration, "register_report",
                        lambda *a, **k: pytest.fail("a read-only request registered"))
    page = tp.read_page(_readonly())
    done, said = tp.register_selected(page, tp.read_reports(LABEL), {}, "demo", site.run_id,
                                      _readonly())
    assert done is False and said.startswith("Refused: ")
    assert not (site.checkpoint.parent / "evaluations.json").exists()


def test_a_busy_ledger_refuses_register_inline_and_register_stays_available(
        site, monkeypatch):
    """F-e and G10 (K8-6r.1), the Test page's Register: while another
    registration holds the ledger's lock past the (patched) grace, the handler
    returns ``(False, message)`` with the four facts -- the ledger is busy,
    nothing was written this time, the report is kept, register it again --
    no ledger is written, and Register stays available (the report reads not
    registered and is never marked invalid). Registering again then works."""
    import src.evaluation.sidecar as sidecar
    from src.evaluation.sidecar import ledger_lock_path
    from src.utils.file_lock import exclusive_lock

    monkeypatch.setattr(sidecar, "LEDGER_LOCK_GRACE_SECONDS", 0.2)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    ledger = site.checkpoint.parent / "evaluations.json"
    with exclusive_lock(ledger_lock_path(ledger), grace=0):
        done, said = tp.register_selected(page, tp.read_reports(LABEL), models, "demo",
                                          site.run_id, request)
    assert done is False
    assert "is not registered: the ledger is busy (another registration was writing" in said
    assert "The report stays where it is kept; register it again." in said
    assert said.endswith("this time.") and "Nothing was written to" in said
    assert not ledger.exists()
    assert tp.register_state(page, tp.read_reports(LABEL), site.run_id) == (True, None)

    done, said = tp.register_selected(page, tp.read_reports(LABEL), models, "demo",
                                      site.run_id, request)
    assert done is True and said == "Registered as record 1."


def test_register_says_when_the_usage_history_was_not_written(site, monkeypatch):
    """F-i (K8-6r.1): the ledger written and the usage history busy past its
    (patched) grace: Register succeeds and its message says the usage history
    was not written, with the reason; the record stands."""
    import src.evaluation.report_store as report_store
    from src.utils.file_lock import exclusive_lock

    monkeypatch.setattr(report_store, "APPEND_LOCK_GRACE_SECONDS", 0.2)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    with exclusive_lock(site.reports / "usage.jsonl.lock", grace=0):
        usage_before = _usage_lines(site)
        done, said = tp.register_selected(page, tp.read_reports(LABEL), models, "demo",
                                          site.run_id, request)
        assert _usage_lines(site) == usage_before
    assert done is True
    lock = site.reports / "usage.jsonl.lock"
    assert said.startswith("Registered as record 1. The usage history was not written: it was "
                           f"busy (another writer held {lock} for the 0.2 s this one waited). ")
    assert "FileLockBusy" not in said and said.endswith(" stands.")
    ledger = json.loads((site.checkpoint.parent / "evaluations.json").read_text())
    assert [r["registration"]["sequence"] for r in ledger["records"]] == [1]


def test_a_busy_notes_file_keeps_nothing_and_says_add_it_again(site, monkeypatch):
    """K8-6r.1, the Test page's note: while another writer holds the notes
    file's lock past the (patched) grace, the note is refused in words that say
    the notes file is busy, nothing was written this time, and it may be added
    again; nothing is appended. Added again, it is kept, and the version's
    report list still holds the run alone (F-k: the lock files are not runs)."""
    import src.evaluation.report_store as report_store
    from src.utils.file_lock import exclusive_lock

    monkeypatch.setattr(report_store, "APPEND_LOCK_GRACE_SECONDS", 0.2)
    request = operator_request(monkeypatch)
    reports = tp.read_reports(LABEL)
    lock = site.reports / "notes.jsonl.lock"
    with exclusive_lock(lock, grace=0):
        done, said = tp.add_note(reports, site.run_id, "ops", "found", request)
    assert done is False
    assert said == ("The note was not kept: another writer held the notes file for the 0.2 s "
                    "this one waited. Nothing was written this time. Add it again.")
    assert not (site.reports / "notes.jsonl").exists()

    done, _said = tp.add_note(reports, site.run_id, "ops", "found", request)
    assert done is True
    again = tp.read_reports(LABEL)
    assert [r["run_id"] for r in again["runs"]] == [site.run_id]
    assert len(again["notes"]) == 1


def _usage_lines(site):
    usage = site.reports / "usage.jsonl"
    return usage.read_bytes() if usage.exists() else b""


def _shown(markup: str) -> str:
    """Rendered HTML as a person reads it: tags dropped, entities decoded."""
    import html as html_lib

    return html_lib.unescape(re.sub(r"<[^>]+>", "", markup))


def _site_manifest(site):
    """The site report's manifest less the fields a Mode C report sets
    (`MODE_C_SEMANTICS`): its paths, digests, cohort block and environment."""
    from tests.unit.test_evaluation_sidecar import MODE_C_SEMANTICS

    report = json.loads((site.reports / site.run_id / "report.json").read_text())
    return {k: v for k, v in report["manifest"].items() if k not in MODE_C_SEMANTICS}


def _keep_mode_c_report(site) -> str:
    """A kept Mode C report of the site's checkpoint on its version, built
    (`mode_c_report`) on the site report's other manifest fields, whatever mode
    that report was measured in: a subject of the one display for every mode
    (R2, K8-6e.2). Its mode is part of its key, so it never contests the site's
    disease report. Its run id."""
    from src.evaluation.cohort_registry import resolve_version
    from tests.fixtures.report_store import store_bytes
    from tests.unit.test_evaluation_sidecar import mode_c_report

    manifest = _site_manifest(site)
    eligible = manifest["cohort_version"]["counts"]["ranked_eligible"]
    report = mode_c_report(n_ranked=eligible, **manifest)
    path = store_bytes(site.registry, json.dumps(report, indent=2).encode("utf-8"),
                       version=resolve_version(site.registry, LABEL), checkpoint_sha256=site.sha)
    return path.parent.name


def _keep_mode_d_report(site) -> str:
    """A kept Mode D report of the site's checkpoint on its version, as Batch 3
    kept them in a version's store: Mode D no longer runs (K7), so it is written
    from the site report's manifest with Mode D's values on Mode C's
    (`mode_d_report`). Its run id."""
    from src.evaluation.cohort_registry import resolve_version
    from tests.fixtures.report_store import store_bytes
    from tests.unit.test_evaluation_sidecar import mode_d_report

    manifest = _site_manifest(site)
    eligible = manifest["cohort_version"]["counts"]["ranked_eligible"]
    report = mode_d_report(n_ranked=eligible - 1, n_eligible=eligible, **manifest)
    path = store_bytes(site.registry, json.dumps(report, indent=2).encode("utf-8"),
                       version=resolve_version(site.registry, LABEL), checkpoint_sha256=site.sha)
    return path.parent.name


def test_a_kept_mode_d_report_is_shown_as_recorded_and_its_register_is_enabled(
        site, monkeypatch):
    """A kept Mode D report beside a Mode C one (K7 removed Mode D, not its
    reports), shown by the one display for every mode (R2): its generic label,
    no tiles, every metric by its recorded name and value, its score semantics
    verbatim (no η derived, no case-set words), the generic coverage; each
    report row says its mode as one this version does not describe; no Mode C
    or D help topic, and no ⓘ on either label. Register is on, as the Mode C
    report's is: no rule refuses a report by its mode (K8-6c, policy rev 15). A
    click registers it by the general rules, under its own mode: record 1 in
    the ledger beside the listed checkpoint, mode D, a `registered` line in the
    version's usage history, and the report reads as kept, its record and its
    key state as before; a second click answers that it is already registered,
    writing nothing."""
    from src.evaluation.caveats import mode_label
    from src.evaluation.cohort_registry import resolve_version
    from src.evaluation.report_store import read_usage_history

    c_run = _keep_mode_c_report(site)
    d_run = _keep_mode_d_report(site)
    reports = tp.read_reports(LABEL)
    runs = {run["run_id"]: run for run in reports["runs"]}
    record = runs[d_run]["record"]
    assert record["scoring"]["mode"] == "D"
    d_html = tp.result_html(reports, d_run, [])
    result = _shown(d_html)
    assert mode_label("D") in result and "Mode D (" in result
    assert "shep-rec-tile-l" not in d_html and "Mean rank" not in result
    assert '<div class="shep-rec-ln"><code>post_import_mrr</code> 0.35</div>' in d_html
    assert '<div class="shep-rec-ln"><code>reached_mean_rank</code> 2.43</div>' in d_html
    assert ('<dt class="shep-k">Score semantics</dt><dd class="shep-v"><code>'
            'served:gnn_only:eta=1.0:sp=none</code></dd>') in d_html
    for gone in ("η =", "import-eligible)", "reached only", "reached by the served pipeline",
                 "Served ranking", "approved scorer policy", "Mode C"):
        assert gone not in result, gone
    assert "test-mode-d" not in tp.TOPIC_HELP and "test-mode-c" not in tp.TOPIC_HELP
    rows = _report_rows(tp.reports_html(reports, None))
    assert rows[d_run][2] == "Mode D, not described by this version"
    assert rows[c_run][2] == "Mode C, not described by this version"
    c_html = tp.result_html(reports, c_run, [])
    c_result = _shown(c_html)
    assert mode_label("C") in c_result and "Mode D" not in c_result
    for label_html in (c_html, d_html):
        label = label_html[label_html.index('class="shep-rec-label"'):]
        label = label[:label.index("</div>")]
        assert "data-help" not in label, label

    page, request = _page(monkeypatch)
    assert tp.register_state(page, reports, d_run) == (True, None)
    assert tp.register_state(page, reports, c_run) == (True, None)
    models = tp.read_models("demo", request)
    done, said = tp.register_selected(page, reports, models, "demo", d_run, request)
    assert done is True and said.startswith("Registered as record 1.")
    ledger = json.loads((site.checkpoint.parent / "evaluations.json").read_text())
    assert [(r["scoring"]["mode"], r["source_artifact_digest"]) for r in ledger["records"]] == [
        ("D", runs[d_run]["state"]["report_sha256"])]
    entries, _ = read_usage_history(resolve_version(site.registry, LABEL))
    last = entries[-1]
    assert (last["kind"], last["run_id"], last["outcome"]) == ("registered", d_run, "registered")
    shown_again = tp.read_reports(LABEL)
    again = {run["run_id"]: run for run in shown_again["runs"]}[d_run]
    assert again["state"]["kept"] == "kept" and again["record"] == record
    assert again["contest"]["state"] == runs[d_run]["contest"]["state"] == "uncontested"
    repeat, said = tp.register_selected(page, shown_again, models, "demo", d_run, request)
    assert repeat is True and "Already registered as record 1. Nothing was written" in said


def _swap(path, replacement: bytes, how: str):
    """Replace a report's bytes in place or by an atomic rename."""
    import os

    if how == "overwrite":
        path.write_bytes(replacement)
    else:
        staged = path.with_name("report.json.staged")
        staged.write_bytes(replacement)
        os.replace(staged, path)


def _another_valid_report(path) -> bytes:
    """The reviewer's counter-example (P2-1): another valid report of the same
    checkpoint and version, its MRR moved by 0.0004."""
    replacement = json.loads(path.read_bytes())
    replacement["authoritative_metrics"]["pooled_mrr"] += 0.0004
    return json.dumps(replacement, indent=2, sort_keys=True).encode()


def _never_decoded(monkeypatch):
    """Fail the test if a report's bytes are decoded as JSON by the one
    report reader (`read_report`), the page's or the registration's."""
    import src.evaluation.report_store as report_store

    class Undecoded:
        def __getattr__(self, name):
            return getattr(json, name)

        @staticmethod
        def loads(*args, **kwargs):
            pytest.fail("a report that is not the one shown was decoded")

    monkeypatch.setattr(report_store, "json", Undecoded())


@pytest.mark.parametrize("bytes_now", ["another valid report", "damaged bytes"])
@pytest.mark.parametrize("how", ["overwrite", "rename"])
def test_a_report_swapped_after_it_was_shown_is_refused_by_the_registration_alone(
    site, monkeypatch, how, bytes_now
):
    """Round 3 of the review fixes (F2): Register has one read and one judge.
    The reviewer's swaps (P2-1) -- another valid report of the same checkpoint
    and version, or damaged bytes, put in place by an overwrite or a rename
    after the report was shown and before Register -- are refused by the
    registration's own comparison, on the one read it would register, before
    the bytes are decoded, in its words ("not the report that was shown ...
    select it again"): never the page's own "changed since it was shown",
    never a parse error. No ledger is written and the usage history is
    unchanged."""
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    reports = tp.read_reports(LABEL)
    path = site.reports / site.run_id / "report.json"
    shown = tp.find(reports["runs"], "run_id", site.run_id)["state"]["report_sha256"]
    usage_before = _usage_lines(site)
    _swap(path, _another_valid_report(path) if bytes_now == "another valid report"
          else b"{not json", how)
    _never_decoded(monkeypatch)

    done, said = tp.register_selected(page, reports, models, "demo", site.run_id, request)

    now = hashlib.sha256(path.read_bytes()).hexdigest()
    assert done is False
    assert said.startswith(
        f"{path} is not the report that was shown: it now holds bytes {now[:12]}..., and "
        f"{shown[:12]}... was shown. It was replaced or changed after it was shown; select it "
        "again. Nothing was written."), said
    assert not (site.checkpoint.parent / "evaluations.json").exists()
    assert _usage_lines(site) == usage_before


def test_the_bytes_registered_are_the_bytes_shown_whatever_replaces_them_after_that_read(
    site, monkeypatch
):
    """The one read is the judge and the source: a report replaced right after
    the registration read it is registered as the bytes it read and compared
    with the digest shown, never refused by a second read and never recorded
    under the new bytes."""
    from tests.fixtures.replacement import replace_after_read

    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    reports = tp.read_reports(LABEL)
    path = site.reports / site.run_id / "report.json"
    shown = tp.find(reports["runs"], "run_id", site.run_id)["state"]["report_sha256"]
    swap = replace_after_read(monkeypatch, after=path, target=path,
                              data=_another_valid_report(path), how="rename")

    done, said = tp.register_selected(page, reports, models, "demo", site.run_id, request)

    assert swap.fired == 1
    assert done is True and said.startswith("Registered as record 1."), said
    ledger = json.loads((site.checkpoint.parent / "evaluations.json").read_text())
    assert [r["source_artifact_digest"] for r in ledger["records"]] == [shown]
    assert hashlib.sha256(path.read_bytes()).hexdigest() != shown


@pytest.mark.parametrize("how", ["overwrite", "rename"])
@pytest.mark.parametrize("hook", ["resolve_report", "ledger_checkpoint"])
def test_the_reviewers_swap_is_refused_before_the_one_read_and_registers_the_bytes_shown_after_it(
    site, monkeypatch, hook, how
):
    """The reviewer's reproduction of P2-1, hooked as it was written: another
    valid report swapped in, in place or by a rename, when `ledger_checkpoint`
    returns. Since round 3 of the review fixes (F2) the page's placement runs
    inside the registration, after its one read compared the digest shown, so
    that swap lands after the read: the bytes shown are registered, the record
    names the digest shown, and the replacement is never recorded (refusing it
    would need a second read, a second judge; the round's verification, A-1).
    The same swap one step earlier, when the page has found the report
    (`resolve_report` returns), lands before that read and is refused in the
    registration's words: no ledger, the usage history unchanged."""
    import src.api.services.server_lists as server_lists

    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    reports = tp.read_reports(LABEL)
    path = site.reports / site.run_id / "report.json"
    shown = tp.find(reports["runs"], "run_id", site.run_id)["state"]["report_sha256"]
    replacement = _another_valid_report(path)
    usage_before = _usage_lines(site)
    owner = server_lists if hook == "resolve_report" else tp
    real = getattr(owner, hook)
    fired = []

    def swap_on_return(*args, **kwargs):
        result = real(*args, **kwargs)
        if not fired:
            fired.append(hook)
            _swap(path, replacement, how)
        return result

    monkeypatch.setattr(owner, hook, swap_on_return)

    done, said = tp.register_selected(page, reports, models, "demo", site.run_id, request)

    assert fired == [hook]
    ledger = site.checkpoint.parent / "evaluations.json"
    now = hashlib.sha256(replacement).hexdigest()
    if hook == "resolve_report":
        assert done is False and said.startswith(
            f"{path} is not the report that was shown: it now holds bytes {now[:12]}..., and "
            f"{shown[:12]}... was shown."), said
        assert not ledger.exists()
        assert _usage_lines(site) == usage_before
    else:
        assert done is True and said.startswith("Registered as record 1."), said
        assert [r["source_artifact_digest"]
                for r in json.loads(ledger.read_text())["records"]] == [shown]
        assert now not in ledger.read_text()


def test_register_takes_the_deployments_registry_from_the_settings_it_was_given(
    site, monkeypatch
):
    """The settings the page found the report by (`environ`) say which registry
    is the deployment's for the registration too: with the process's own
    `SHEPHERD_REGISTRY` unset, the settings passed are used, and the report
    registers (BACKLOG 37)."""
    import os

    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    reports = tp.read_reports(LABEL)
    environ = dict(os.environ)
    monkeypatch.delenv("SHEPHERD_REGISTRY")

    done, said = tp.register_selected(page, reports, models, "demo", site.run_id, request,
                                      environ=environ)

    assert done is True and said.startswith("Registered as record 1."), said


def test_no_listed_checkpoint_with_the_weights_no_registration(site, monkeypatch):
    page, request = _page(monkeypatch)
    reports = tp.read_reports(LABEL)
    other = site.checkpoint.parent / "other.pt"
    site.checkpoint.rename(other)
    other.write_bytes(other.read_bytes() + b"x")
    models = tp.read_models("demo", request)
    usage_before = _usage_lines(site)
    done, said = tp.register_selected(page, reports, models, "demo", site.run_id, request)
    assert done is False and "No checkpoint the server lists (demo) holds the weights" in said
    assert said.endswith("--report … --checkpoint …. Nothing was written."), said
    assert _usage_lines(site) == usage_before


def test_each_run_of_a_group_whose_runs_differ_registers_with_the_note(site, monkeypatch):
    """G3 (K8-6r.2): the Test page's Register, through the one registration,
    registers each run of a group whose runs differ as a record of its own,
    and its message carries the group note, which decides nothing. HEAD:
    refused, "runs disagree — investigate", nothing written."""
    other = _disagreeing_copy(site)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    done, said = tp.register_selected(page, tp.read_reports(LABEL), models, "demo", site.run_id,
                                      request)
    assert done is True and said.startswith("Registered as record 1."), said
    done, said = tp.register_selected(page, tp.read_reports(LABEL), models, "demo", other,
                                      request)
    assert done is True and said.startswith("Registered as record 2."), said
    assert "Runs under these conditions differ: record 2 (this run) differs from record 1" in said
    assert "pooled_mrr" in said
    ledger = json.loads((site.checkpoint.parent / "evaluations.json").read_text())
    assert [r["run_id"] for r in ledger["records"]] == [site.run_id, other]


def test_a_note_is_kept_with_the_group_and_changes_nothing(site, monkeypatch):
    _disagreeing_copy(site)
    reports = tp.read_reports(LABEL)
    done, said = tp.add_note(reports, site.run_id, "", "found", operator_request(monkeypatch))
    assert done is False and "'who'" in said
    refused, said = tp.add_note(reports, site.run_id, "ops", "found", _readonly())
    assert refused is False and said.startswith("Refused: ")
    done, _said = tp.add_note(reports, site.run_id, "ops <b>", "cuDNN differs",
                              operator_request(monkeypatch))
    assert done is True
    reports = tp.read_reports(LABEL)
    page = tp.read_page(operator_request(monkeypatch))
    html = tp.result_html(reports, site.run_id, [], notes=reports["notes"])
    assert "ops &lt;b&gt;" in html and "cuDNN differs" in html and "Investigation notes (1)" in html
    # A note records what was found and changes nothing: Register was enabled
    # before it and stays so (K8-6r.2), whatever the group's description.
    assert tp.register_state(page, reports, site.run_id) == (True, None)
    assert _said == ("The note was kept with this group's reports. It records what was found "
                     "and changes nothing.")


# --------------------------------------------------------------------------- the import
def test_the_import_is_refused_for_a_read_only_request(site, monkeypatch):
    from src.api.services import server_lists
    from src.evaluation import cohort_registry

    monkeypatch.setattr(server_lists, "list_intake_files",
                        lambda **k: pytest.fail("a read-only request listed the intake"))
    monkeypatch.setattr(cohort_registry, "prepare_import",
                        lambda *a, **k: pytest.fail("a read-only request previewed"))
    assert tp.open_import(_readonly())["refused"].startswith("Refused: ")
    preview, refusal = tp.preview_import("demo", "cases.json", "v2", "demo", "keep", _readonly())
    assert preview is None and refusal.startswith("Refused: ")
    done, said, label = tp.commit_import_ui({"preview_id": "0" * 16}, "demo", "cases.json", "v2",
                                            "demo", "keep", _readonly())
    assert done is False and said.startswith("Refused: ") and label is None


def test_the_preview_shows_reasons_and_counts_never_a_case_id(site, monkeypatch):
    listing = tp.open_import(operator_request(monkeypatch))
    assert [i["name"] for i in listing["items"]] == ["cases.json"]
    preview, refusal = tp.preview_import("demo", "cases.json", "demo-test-2", "demo", "keep",
                                         operator_request(monkeypatch))
    assert refusal is None and re.fullmatch(r"[0-9a-f]{16}", preview["preview_id"])
    assert "exclusions" not in json.dumps(preview) and "case-" not in json.dumps(preview)
    html = tp.preview_html(preview)
    assert "case-" not in html
    assert "<b>3</b> of 13: ground truth absent 2 · phenotype unmappable 1" in html
    assert "HP:9999999</code> listed by 2 cases" in html
    assert "SYNTHETIC WORKSPACE" in html and "nothing written" in html
    assert not (site.registry / "versions" / "demo-test-2").exists()


def test_the_preview_takes_a_listed_file_only(site, monkeypatch):
    for name in ("../intake/cases.json", str(site.intake / "cases.json"), "missing.json"):
        preview, refusal = tp.preview_import("demo", name, "demo-test-2", "demo", "keep",
                                             operator_request(monkeypatch))
        assert preview is None and "is not an import source the server lists" in refusal
        assert refusal.endswith("Nothing was written.")


def test_commit_publishes_what_was_previewed(site, monkeypatch):
    request = operator_request(monkeypatch)
    preview, _ = tp.preview_import("demo", "cases.json", "demo-test-2", "demo", "keep", request)
    changed, said, _ = tp.commit_import_ui(preview, "demo", "cases.json", "demo-test-2",
                                           "demo", "exclude", request)
    assert changed is False and "The form changed since the preview" in said
    done, said, label = tp.commit_import_ui(preview, "demo", "cases.json", "demo-test-2", "demo",
                                            "keep", request)
    assert done is True and label == "demo-test-2"
    assert said.startswith("Committed cohort version demo-test-2 (role demo): 10 ranked of 13")
    assert (site.registry / "versions" / "demo-test-2" / "cohort.json").is_file()


def test_commit_refuses_inputs_that_changed_since_the_preview(site, monkeypatch):
    request = operator_request(monkeypatch)
    preview, _ = tp.preview_import("demo", "cases.json", "demo-test-2", "demo", "keep", request)
    source = site.intake / "cases.json"
    cases = json.loads(source.read_text())
    cases["cases"] = cases["cases"][:-1]
    source.write_text(json.dumps(cases))
    done, said, _ = tp.commit_import_ui(preview, "demo", "cases.json", "demo-test-2", "demo",
                                        "keep", request)
    assert done is False and "the inputs changed since the preview" in said
    assert said.endswith("Nothing was written.") and said.count("Nothing was written.") == 1
    assert not (site.registry / "versions" / "demo-test-2").exists()


def test_a_commit_refused_after_the_source_copy_was_kept_says_so(site, monkeypatch):
    """The importer's refusal from the commit itself is said as it is: one
    raised after the source copy was written says what was written, and the
    page never adds "Nothing was written." to it."""
    import src.evaluation.cohort_registry as cohort_registry

    request = operator_request(monkeypatch)
    preview, _ = tp.preview_import("demo", "cases.json", "demo-test-2", "demo", "keep", request)

    def taken(*_a, **_k):
        raise cohort_registry.CohortVersionRefused(
            f"{site.registry}/versions/demo-test-2 was created by another import while this one "
            "ran, so the label 'demo-test-2' is taken. Only the source copy was written (under "
            f"{site.registry}/sources, named by its digest, which is harmless to keep).")

    monkeypatch.setattr(cohort_registry, "commit_import", taken)
    done, said, _ = tp.commit_import_ui(preview, "demo", "cases.json", "demo-test-2", "demo",
                                        "keep", request)
    assert done is False and "Only the source copy was written" in said
    assert "Nothing was written" not in said


def test_the_intake_note_names_what_is_not_offered_without_its_path(site):
    outside = site.root / "elsewhere.json"
    outside.write_text("{}")
    (site.intake / "leaves.json").symlink_to(outside)
    from dataclasses import asdict

    from src.api.services.server_lists import list_intake_files

    listing = {"items": [asdict(i) for i in list_intake_files().items], "problem": None}
    assert [label for label, _name in tp.intake_choices(listing)][0].startswith("cases.json · ")
    assert [name for _l, name in tp.intake_choices(listing)] == ["cases.json"]
    note = tp.intake_note_html(listing, tp.path_roots())
    assert "leaves.json resolves to a server path" in note and str(site.root) not in note


def test_with_no_import_folder_the_drawer_says_its_notice_not_the_whole_sentence(monkeypatch):
    """S13 of the sweep's integration (the lanes' L3 R4): with no import folder set the import
    drawer said the setting's whole sentence (`server_settings.INTAKE_DIR_NOT_SET`, its variable
    and the folder's rule in the flow). It says the folder's notice
    (`server_settings.INTAKE_DIR_NOTICE` through `workspaces.setting_notice`, chosen by the
    setting's flags): one line, its link to Settings › Data locations, the rest under
    "Details ›" (the variable only in its Ref line); a folder set but refused says it cannot be
    used and where to see why, never its path."""
    from src.config import server_settings as ss
    from tests.fixtures.operator_access import operator_request

    monkeypatch.delenv("SHEPHERD_INTAKE_DIR", raising=False)
    listing = tp.open_import(operator_request(monkeypatch), environ={})
    assert listing["items"] == [] and listing["notice"]["key"] == "intake_dir"
    note = tp.intake_note_html(listing, [])
    assert note.count('data-notice="intake_dir"') == 1 and 'data-shep-go="settings"' in note
    line = note.split("<details", 1)[0]
    assert ss.NO_IMPORT_FOLDER in line and "SHEPHERD_INTAKE_DIR" not in line
    assert ss.INTAKE_DIR_NOT_SET not in note
    assert "SHEPHERD_INTAKE_DIR" in re.sub(r"<[^>]+>", "", note.split("<details", 1)[1])
    refused = tp.open_import(operator_request(monkeypatch), environ={"SHEPHERD_INTAKE_DIR": "relative"})
    said = tp.intake_note_html(refused, [])
    assert "The import folder cannot be used." in said and "relative" not in said


def test_the_import_source_is_named_by_the_folder_settings_chooses():
    """T06: the select read "File in the intake directory (server)" while
    Settings › Data locations, where the folder is chosen, names it "Import
    folder". The select, its help and its empty note now use that name."""
    from src.config.deployment_settings import KEY_INTAKE_DIR
    from src.utils.deployment_view import FIELD_WORDS

    name = FIELD_WORDS[KEY_INTAKE_DIR][0].lower()
    assert name == "import folder" and name in tp.INTAKE_LABEL
    demo, _ctx, _registry = _page_component()
    (select,) = [b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "shep-test-intake"]
    assert select.label == tp.INTAKE_LABEL
    assert tp.FIELD_HELP["shep-test-intake"][0] == tp.INTAKE_LABEL
    empty = tp.intake_note_html({"items": [], "problem": None}, [])
    assert "No import file is in the import folder" in empty and "intake" not in empty


# --------------------------------------------------------------------------- the page's wiring
def _page_component():
    from src.webui.components.help import HelpRegistry
    from src.webui.shell import standalone_context

    registry = HelpRegistry()
    with gr.Blocks() as demo:
        ctx = standalone_context(demo, registry)
        tp.create_test_page(ctx)
    return demo, ctx, registry


def _named(demo, name):
    found = [fn for fn in demo.fns.values() if fn.name == name]
    assert len(found) == 1, (name, len(found))
    return found[0]


def _chain(demo, fn, depth=4):
    names, current = [], fn
    for _ in range(depth):
        nxt = [d for d in demo.fns.values() if d.trigger_after == current._id]
        if not nxt:
            break
        current = nxt[0]
        names.append(current.fn)
    return names


def test_every_operator_control_is_followed_by_the_chips_read():
    from src.webui.components.chips import read_chips

    demo, _ctx, _registry = _page_component()
    for name in ("_on_start", "_on_cancel", "_on_register", "_on_note", "_on_open_import",
                 "_on_preview", "_on_commit", "_on_model_reload", "_on_workspace"):
        assert read_chips in _chain(demo, _named(demo, name)), name


def test_the_page_is_read_when_shown_and_polled_only_while_shown(monkeypatch):
    demo, ctx, _registry = _page_component()
    refresh = _named(demo, "_refresh")
    assert (ctx.page_tokens["test"]._id, "change") in refresh.targets
    poll = _named(demo, "_poll")
    assert (ctx.timer_slow._id, "tick") in poll.targets and poll.queue is False
    assert poll.inputs[0] is ctx.location
    selects = [b for b in poll.outputs if isinstance(b, (gr.Dropdown, gr.Checkbox, gr.Radio))]
    assert selects == [], "a poll never touches a select"
    selection = _named(demo, "_on_report").outputs[:2]
    assert not [b for b in poll.outputs if b in selection], "a poll never writes the selection"
    monkeypatch.setattr(tp, "read_job", lambda **k: pytest.fail("polled while not shown"))
    quiet = poll.fn("clinical", {"access": {}}, {"status": "running"}, None)
    assert all(update == gr.update() for update in quiet)


def test_the_poll_reads_and_writes_nothing_drawn_from_the_selection():
    """F6 and B46-races: the poll writes what the session saw of the job and
    the hand-offs; never Start or its reason, which are drawn from the
    selection (a tick that read the selection before a choice wrote Start
    back for the selection it read), and it is given no part of the
    selection to draw them from; nor the job's card, which the selection's
    group draws again on the hand-off (`test_the_job_card_has_one_kind_of_
    writer_the_selections_group`)."""
    demo, _ctx, _registry = _page_component()
    poll = _named(demo, "_poll")
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    start, reason = by_id["shep-test-start"], by_id["shep-test-start-reason"]
    assert start not in poll.outputs and reason not in poll.outputs
    selection = [by_id[i] for i in ("shep-test-workspace", "shep-test-version", "shep-test-model",
                                    "shep-test-measurement")]
    selection += _named(demo, "_on_report").outputs[:2] + [_named(demo, "_on_model_reload")
                                                           .outputs[0]]
    assert not [b for b in poll.inputs if b in selection]
    assert not {by_id[i] for i in ("shep-test-job-head", "shep-test-job", "shep-test-cancel",
                                   "shep-test-cancel-reason")} & set(poll.outputs)
    writers = [fn for fn in demo.fns.values() if {start, reason} & set(fn.outputs)]
    assert {"_on_model", "_on_measurement", "_after_act",
            "_after_job_change"} <= {fn.name for fn in writers}
    for fn in writers:
        assert fn.queue is not False, fn.name
        assert (fn.concurrency_id, fn.concurrency_limit) == (tp.SELECTION_CONCURRENCY, 1), fn.name


def test_no_listener_of_the_page_sends_a_request_on_the_fast_timer():
    """A shell timer's tick is a request per listener whatever it does; the
    page polls on the slow timer only."""
    demo, ctx, _registry = _page_component()
    assert not [fn for fn in demo.fns.values() if (ctx.timer_fast._id, "tick") in fn.targets]


GPU_NONE = ["none", "GPU jobs: none"]


def test_a_poll_hands_the_store_read_over_only_when_the_job_changed(site, monkeypatch):
    """The poll never reads the store: on the tick that sees the job change it
    gives the hidden hand-off a new value naming what the session saw before
    and what it sees now, and the hand-off's listener reads the store again,
    for the version the session's reports were read for (not the select as
    it was when the hand-off was asked)."""
    demo, ctx, _registry = _page_component()
    poll, after = _named(demo, "_poll"), _named(demo, "_after_job_change")
    handoff = poll.outputs[-2]
    assert isinstance(handoff, gr.JSON) and not isinstance(handoff, gr.State)
    assert (handoff._id, "change") in after.targets
    manager = FakeManager({"status": "completed", "job_id": "test-1", "inputs": {},
                           "outcome": None})
    monkeypatch.setattr(tp, "read_job", lambda **k: {**manager.get_test_status(), "log": []})
    monkeypatch.setattr(tp, "gpu_line", lambda **k: tuple(GPU_NONE))
    reads = []
    monkeypatch.setattr(tp, "read_reports", lambda label, **k: reads.append(label) or
                        {"label": label, "runs": [], "problem": None, "notes": []})
    page = tp.read_page(_readonly())
    unchanged = poll.fn("test", page, {"job_id": "test-1", "status": "completed",
                                       "gpu": GPU_NONE}, _readonly())
    assert reads == [] and unchanged[-2] == gr.update()
    ended = poll.fn("test", page, {"job_id": "test-1", "status": "running", "gpu": GPU_NONE},
                    _readonly())
    assert reads == [] and {k: ended[0][k] for k in ("job_id", "status", "gpu")} == {
        "job_id": "test-1", "status": "completed", "gpu": GPU_NONE}
    assert ended[-2]["seen"] == {"job_id": "test-1", "status": "running"}
    assert ended[-2]["now"] == {"job_id": "test-1", "status": "completed"}
    again = poll.fn("test", page, {"job_id": "test-1", "status": "running", "gpu": GPU_NONE},
                    _readonly())
    assert again[-2]["at"] != ended[-2]["at"], "each hand-off is a new value"
    out = after.fn(page, {}, None, "demo", None, None, None, {"label": LABEL}, ended[-2],
                   _readonly())
    assert reads == [LABEL] and out[0]["label"] == LABEL


def test_the_gpu_line_changing_hands_start_over_to_the_selections_group(site, monkeypatch):
    """F6: Start's first reason is the GPU line, and the poll writes no Start:
    a tick that sees only the GPU line change hands the read over too, and
    the hand-off writes Start and its reason from the selection it is given
    (the session's, in its group) and the GPU line read now; it decides no
    end, so the report chosen stays."""
    demo, _ctx, _registry = _page_component()
    poll, after = _named(demo, "_poll"), _named(demo, "_after_job_change")
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    start_at = after.outputs.index(by_id["shep-test-start"])
    reason_at = after.outputs.index(by_id["shep-test-start-reason"])
    job = {"status": "idle", "job_id": None, "inputs": {}, "log": [], "outcome": None}
    monkeypatch.setattr(tp, "read_job", lambda **k: dict(job))
    line = ("busy", "GPU job: training (job runner)")
    monkeypatch.setattr(tp, "gpu_line", lambda **k: line)
    _path, child = _put_bundle(site)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    binding = tp.workspace_binding(site.demo, tp.find(page["workspaces"], "name", "demo"))
    reports = tp.read_reports(LABEL)
    seen = {"job_id": None, "status": "idle", "gpu": list(line)}
    assert poll.fn("test", page, seen, request)[-2] == gr.update()
    line = ("none", "GPU jobs: none")
    handoff = poll.fn("test", page, seen, request)[-2]
    assert handoff["seen"] == handoff["now"] == {"job_id": None, "status": "idle"}
    out = after.fn(page, models, binding, "demo", child, "disease", site.run_id, reports,
                   handoff, request)
    assert out[1] == site.run_id
    assert out[start_at] == gr.update(value="▶ Start test", interactive=True)
    assert out[reason_at] == ""
    line = ("busy", "GPU job: training (job runner)")
    out = after.fn(page, models, binding, "demo", child, "disease", site.run_id, reports,
                   handoff, request)
    assert out[start_at] == gr.update(value="▶ Start test", interactive=False)   # no lock
    assert "Start disabled: GPU job: training (job runner)." in out[reason_at]


def test_the_header_chips_follow_the_gpu_line_on_the_tick_that_sees_it_change(site, monkeypatch):
    """M5-3: the header's chips are read again on the slow tick whose GPU line
    (the GPU chip read now) is not the one this session saw last; an unchanged
    tick sends nothing for them."""
    from src.webui.components import chips

    demo, ctx, _registry = _page_component()
    poll = _named(demo, "_poll")
    assert poll.outputs[-1] is ctx.chips
    monkeypatch.setattr(tp, "read_job", lambda **k: {"status": "idle", "job_id": None, "log": []})
    line = ["none", "GPU jobs: none"]
    monkeypatch.setattr(tp, "gpu_line", lambda **k: tuple(line))
    read = []
    monkeypatch.setattr(chips, "read_chips", lambda request: read.append(1) or "<chips>")
    monkeypatch.setattr(tp, "read_reports", lambda label, **k:
                        {"label": label, "runs": [], "problem": None, "notes": []})
    page = tp.read_page(_readonly())
    seen = {"job_id": None, "status": "idle", "gpu": ["none", "GPU jobs: none"]}
    same = poll.fn("test", page, seen, _readonly())
    assert same[-1] == gr.update() and read == []
    line[:] = ["busy", "GPU job: training (job runner)"]
    moved = poll.fn("test", page, seen, _readonly())
    assert moved[-1] == "<chips>" and read == [1]
    assert moved[0]["gpu"] == ["busy", "GPU job: training (job runner)"]


def test_the_report_of_a_run_that_has_just_ended_is_selected(site, monkeypatch):
    """A run this session saw running ends with a report: the read the poll
    hands over selects it, even over the report the operator had chosen; once
    seen ended, the poll hands nothing over. So does a hand-off whose run the
    poll never saw busy because it started and ended between two ticks (the
    poll saw the run before it, or no run). Only that hand-off decides an
    end: an act's read after it keeps the operator's choice, and so does a
    hand-off that names no end (the GPU line changed, or the session's first
    tick, which saw nothing before) or whose ended run is no longer the job
    read now (even when that job ended with another listed report)."""
    other = _disagreeing_copy(site)
    demo, ctx, _registry = _page_component()
    poll, after = _named(demo, "_poll"), _named(demo, "_after_job_change")
    after_act = next(fn for fn in demo.fns.values() if fn.name == "_after_act")
    job = {"status": "completed", "job_id": "test-2", "inputs": {}, "log": [],
           "outcome": {"report_dir": str(site.reports / site.run_id), "kept": True,
                       "registered": False, "summary": "kept"}}
    monkeypatch.setattr(tp, "read_job", lambda **k: dict(job))
    monkeypatch.setattr(tp, "gpu_line", lambda **k: tuple(GPU_NONE))
    page = tp.read_page(_readonly())
    reports = tp.read_reports(LABEL)
    handoff = poll.fn("test", page, {"job_id": "test-2", "status": "running", "gpu": GPU_NONE},
                      _readonly())[-2]
    ended = after.fn(page, {}, None, "demo", None, None, other, reports, handoff, _readonly())
    assert ended[1] == site.run_id
    quiet = poll.fn("test", page, {"job_id": "test-2", "status": "completed", "gpu": GPU_NONE},
                    _readonly())
    assert quiet[-2] == gr.update(), "nothing is handed over once the end was seen"
    acted = after_act.fn(page, {}, None, "demo", None, None, other, reports, _readonly())
    assert acted[1] == other, "an act's read decides no end"
    completed = {"job_id": "test-2", "status": "completed"}
    no_end = after.fn(page, {}, None, "demo", None, None, other, reports,
                      {"seen": completed, "now": completed, "at": 1}, _readonly())
    assert no_end[1] == other, "a hand-off that names no end keeps the operator's choice"
    for before in ({"job_id": "test-1", "status": "completed"}, {"job_id": None, "status": "idle"}):
        between = after.fn(page, {}, None, "demo", None, None, other, reports,
                           {"seen": before, "now": completed, "at": 2}, _readonly())
        assert between[1] == site.run_id, f"a run that started and ended after {before}"
    first = poll.fn("test", page, {}, _readonly())[-2]
    assert first["seen"] == {"job_id": None, "status": None} and first["now"] == completed
    first_tick = after.fn(page, {}, None, "demo", None, None, other, reports, first,
                          _readonly())
    assert first_tick[1] == other, "the session's first tick saw nothing before: no end"
    job.update(job_id="test-3", status="completed",
               outcome={"report_dir": str(site.reports / site.run_id), "kept": True,
                        "registered": False, "summary": "kept"})
    moved_on = after.fn(page, {}, None, "demo", None, None, other, reports, handoff,
                        _readonly())
    assert moved_on[1] == other, "the ended run is no longer the job read now"


def test_a_poll_that_sees_the_job_change_chooses_again_when_the_shown_report_is_gone(site,
                                                                                    monkeypatch):
    """A run started elsewhere changes the job: the store is read again, and a
    report this session showed that is gone from it gives way to the newest."""
    demo, ctx, _registry = _page_component()
    poll, after = _named(demo, "_poll"), _named(demo, "_after_job_change")
    job = {"status": "running", "job_id": "test-3", "inputs": {}, "log": [], "outcome": None}
    monkeypatch.setattr(tp, "read_job", lambda **k: dict(job))
    monkeypatch.setattr(tp, "gpu_line", lambda **k: ("busy", "GPU job: test run (job runner)"))
    page = tp.read_page(_readonly())
    gone = "20261001T000000Z-" + "a" * 12 + "-" + "b" * 8
    handoff = poll.fn("test", page, {"job_id": "test-2", "status": "completed",
                                     "gpu": ["none", "GPU jobs: none"]}, _readonly())[-2]
    out = after.fn(page, {}, None, "demo", None, None, gone, {"label": LABEL}, handoff,
                   _readonly())
    assert out[1] == site.run_id


def test_the_import_is_one_at_a_time_and_there_is_no_upload():
    demo, _ctx, _registry = _page_component()
    preview, commit = _named(demo, "_on_preview"), _named(demo, "_on_commit")
    assert preview.concurrency_id == commit.concurrency_id == "shep-test-import"
    assert preview.concurrency_limit == commit.concurrency_limit == 1
    kinds = {type(b).__name__ for b in demo.blocks.values()}
    assert not kinds & {"File", "UploadButton", "Files"}


def test_a_commits_follow_up_reads_the_page_with_the_committed_version_chosen(site, monkeypatch):
    """A Commit's follow-up (`_after_commit`, chained after `_on_commit` as
    the page wires it) reads the page again as its opening does, with the
    version just committed chosen: the Data panel, the version select and the
    Run panel drawn for it, each in its own output. R-68 gave the opening a
    hand-off (`ctx.test_model`, an input and an output) and the follow-up
    still called it with the old positions: a TypeError after every
    successful Commit (a red "Error" toast; the panels left as they were
    until the page was opened again). A commit hands no model over: the
    follow-up neither reads nor writes the hand-off."""
    monkeypatch.setattr(tp, "read_job", lambda **k: {"status": "idle", "inputs": None,
                                                    "outcome": None, "log": []})
    monkeypatch.setattr(tp, "gpu_line", lambda **k: tuple(GPU_NONE))
    demo, ctx, _registry = _page_component()
    commit = _named(demo, "_on_commit")
    (after,) = [fn for fn in demo.fns.values() if fn.trigger_after == commit._id]
    assert after.name == "_after_commit"
    assert ctx.test_model not in after.inputs and ctx.test_model not in after.outputs
    request = operator_request(monkeypatch)
    page = tp.read_page(request)
    preview, _ = tp.preview_import("demo", "cases.json", "demo-test-2", "demo", "keep", request)
    written = dict(zip([b._id for b in commit.outputs],
                       commit.fn(page, preview, "demo", "cases.json", "demo-test-2", "demo",
                                 "keep", request), strict=True))
    committed = written[after.inputs[0]._id]
    assert committed == "demo-test-2"
    # The rest of its inputs as the page holds them: the workspace, the
    # version chosen before the import, no model or measurement chosen yet.
    out = dict(zip([b.elem_id for b in after.outputs],
                   after.fn(committed, "demo", LABEL, None, None, {}, None, request),
                   strict=True))
    chosen = out["shep-test-version"]
    assert chosen["value"] == "demo-test-2"
    assert {"demo-test-1", "demo-test-2"} <= {value for _label, value in chosen["choices"]}
    # The version's own lines, drawn for the version committed (13 source
    # cases, 10 ranked; bound to this workspace's graph), not an empty read.
    info = out["shep-test-version-info"]
    assert "No data version" not in info and "Bound to this graph" in info
    assert "13 source cases, 10 ranked-eligible" in info
    assert out["shep-test-workspace"]["value"] == "demo"
    runs = [value for _label, value in out["shep-test-model-run"]["choices"]]
    assert any(CHILD_RUNS["disease"] in str(value) for value in runs), runs


def test_every_write_of_the_selection_is_in_one_queued_group_of_one():
    """The session's selection (the checkpoint listing, the version's
    reports, the report shown) is written only by queued listeners of
    `SELECTION_CONCURRENCY`, one at a time: page open, the workspace, the
    version, a report's row, ↻, the read after each act and after an import,
    and the poll's hand-off. The version writes the select too (a workspace's
    event asked before it set it). A choice the operator makes and the poll's
    hand-off join every time they are asked ("multiple"): with Gradio's
    defaults the browser drops a second trigger of a listener whose event
    waits ("once") or asks it again with the first one's data
    ("always_last"), and in the group an event waits. What the session saw of
    the job has one writer, the poll (F6)."""
    demo, ctx, _registry = _page_component()
    selected_report, reports_state = _named(demo, "_on_report").outputs[:2]
    models_state = _named(demo, "_on_model_reload").outputs[0]
    selection = (models_state, reports_state, selected_report)
    writers = [fn for fn in demo.fns.values() if set(selection) & set(fn.outputs)]
    assert sorted(fn.name for fn in writers) == sorted(
        ["_refresh", "_on_workspace", "_on_version", "_on_report", "_on_model_reload",
         "_after_act", "_after_act", "_after_act", "_after_act", "_after_commit",
         "_after_job_change"])
    for fn in writers:
        assert fn.queue is not False, fn.name
        assert (fn.concurrency_id, fn.concurrency_limit) == (tp.SELECTION_CONCURRENCY, 1), fn.name
    for name in ("_on_workspace", "_on_version", "_on_run", "_on_model", "_on_measurement",
                 "_on_report", "_after_job_change"):
        assert _named(demo, name).trigger_mode == "multiple", name
    version = _named(demo, "_on_version")
    assert version.inputs[4] in version.outputs
    poll = _named(demo, "_poll")
    job_state = poll.outputs[0]
    assert isinstance(job_state, gr.State) and job_state in poll.inputs
    assert [fn.name for fn in demo.fns.values() if job_state in fn.outputs] == ["_poll"]
    assert [fn.name for fn in demo.fns.values() if job_state in fn.inputs] == ["_poll"]



def test_a_late_read_never_writes_back_the_selection_it_started_from(site, monkeypatch):
    """P2-5-F2, through the page served by Gradio: one session, an event that
    writes the selection held inside its store read (the slow part: it builds
    every kept report) while the operator acts. With the read the poll hands
    over after the job changed held, a report's row clicked meanwhile stays
    chosen, and a version chosen meanwhile is the one whose reports the
    session holds; with a workspace's event held, a version chosen meanwhile
    is the one the select ends on. Each listener being its own group, the
    held event ended last and wrote back the selection it started from: the
    click undone, the select over another version's reports."""
    from tests.fixtures.registered_cohort import import_version

    other_label = "demo-test-2"
    import_version(site.registry, site.demo, site.intake / "cases.json", label=other_label)
    other = _disagreeing_copy(site)
    job = {"status": "completed", "job_id": "test-9", "inputs": {}, "log": [], "outcome": None}
    monkeypatch.setattr(tp, "read_job", lambda **k: dict(job))
    monkeypatch.setattr(tp, "gpu_line", lambda **k: ("none", "GPU jobs: none"))
    real = tp.read_reports
    hold, entered, release = threading.Event(), threading.Event(), threading.Event()

    def read_reports(label, **kwargs):
        if hold.is_set():
            hold.clear()
            entered.set()
            assert release.wait(30), "never released"
        return real(label, **kwargs)

    monkeypatch.setattr(tp, "read_reports", read_reports)
    demo, _ctx, _registry = _page_component()
    refresh, on_workspace, on_version, on_report, after = (
        _named(demo, n) for n in ("_refresh", "_on_workspace", "_on_version", "_on_report",
                                  "_after_job_change"))
    selected_report, reports_state = on_report.outputs[:2]
    version_select = next(b for b in demo.blocks.values()
                          if getattr(b, "elem_id", None) == "shep-test-version")

    def held(first, act):
        """`first` asked and held in its store read; `act` asked meanwhile and
        left 1.5 s to finish before `first` goes on (in the group it cannot:
        `first` was asked first)."""
        hold.set()
        entered.clear()
        release.clear()
        events = [first()]
        assert entered.wait(30), "the held event did not start"
        events.append(act())
        browser.wait(events[1], timeout=1.5)
        release.set()
        assert browser.wait(*events)

    def shown():
        """The version select as the browser shows it: the operator's own
        picks and the answers that write it, in the order they came."""
        value = None
        for event in browser.order:
            if isinstance(event, tuple):                  # ("picked", label)
                value = event[1]
                continue
            data, fn = browser.done[event]["output"]["data"], asked[event]
            if version_select in fn.outputs:
                value = data[fn.outputs.index(version_select)]["value"]
        return value

    def pick(label):
        with browser.cond:
            browser.order.append(("picked", label))
        return join(on_version, [None, None, None, "demo", label, None, None, None])

    asked = {}

    def join(fn, data, event_data=None):
        event = browser.join(fn, data, event_data)
        asked[event] = fn
        return event

    with served(demo) as base:
        browser = ServedSession(base, demo)
        assert browser.wait(join(refresh, [None] * len(refresh.inputs)))
        assert browser.wait(pick(LABEL))
        runs = [r["run_id"] for r in browser.state(reports_state)["runs"]]
        assert sorted(runs) == sorted([site.run_id, other]) and runs[0] == site.run_id
        assert browser.state(selected_report) == site.run_id and shown() == LABEL
        handoff = {"seen": {"job_id": "test-9", "status": "running"},
                   "now": {"job_id": "test-9", "status": "completed"}, "at": 1}
        read_again = lambda: join(after, [None, None, None, "demo", None, None, None,  # noqa: E731
                                          None, handoff])

        # a report's row clicked while the hand-off's read is held
        held(read_again, lambda: join(on_report, [None] * 7, {"key": other}))
        assert browser.state(selected_report) == other, "the click was undone"
        assert browser.state(reports_state)["label"] == LABEL == shown()

        # another version chosen while the hand-off's read is held
        held(read_again, lambda: pick(other_label))
        assert shown() == other_label
        assert browser.state(reports_state)["label"] == other_label, \
            "the select shows one version and the session holds another's reports"
        assert browser.state(selected_report) is None

        # a version chosen while a workspace's event (asked with the version
        # shown then) is held: the select ends on the version chosen
        held(lambda: join(on_workspace, [None, "demo", other_label, None, None, None, None]),
             lambda: pick(LABEL))
        assert browser.state(reports_state)["label"] == LABEL
        assert shown() == LABEL, "the select shows one version and the session holds another's"
        assert browser.state(selected_report) == site.run_id
        browser.close()


def test_the_job_card_has_one_kind_of_writer_the_selections_group():
    """B46-races, the contract: the job's title row, its details, Cancel and
    Cancel's reason (`job_outputs`) are written only by queued listeners of
    the selection's group, one at a time, first asked first run (the page
    open, the read after an act, the poll's hand-offs). The slow poll
    (``queue=False``) writes none of them: on the old wiring it wrote the
    card too, so a tick that read the job before an act's read and answered
    after it brought back the older card for 5 s."""
    demo, _ctx, _registry = _page_component()
    poll = _named(demo, "_poll")
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    card = {by_id[elem_id] for elem_id in ("shep-test-job-head", "shep-test-job",
                                           "shep-test-cancel", "shep-test-cancel-reason")}
    writers = [fn for fn in demo.fns.values() if card & set(fn.outputs)]
    assert writers and not card & set(poll.outputs)
    outside = sorted({fn.name for fn in writers
                      if (fn.concurrency_id, fn.concurrency_limit)
                      != (tp.SELECTION_CONCURRENCY, 1) or fn.queue is False})
    assert outside == [], outside


def test_a_tick_that_crosses_an_acts_read_never_brings_back_an_older_job_card(
        site, monkeypatch):
    """B46-races, through the page served by Gradio, one session: the slow
    tick's read of the job is held (the run going); meanwhile the run ends
    and an act's read (as after Cancel) draws the card ("Completed").
    Released, the tick answers last. On the old wiring its own card, the
    running one it read before, was applied after the act's and stayed for a
    tick (5 s). Now the tick writes no card: it hands a change over, and the
    card is drawn again in the selection's group from the job as it is."""
    job = {"status": "running", "job_id": "test-7", "inputs": {}, "log": [], "outcome": None,
           "elapsed_seconds": 12}
    hold, entered, release = threading.Event(), threading.Event(), threading.Event()

    def read_job(**_kwargs):
        now = dict(job)
        if hold.is_set():
            hold.clear()
            entered.set()
            assert release.wait(30), "never released"
        return now

    monkeypatch.setattr(tp, "read_job", read_job)
    monkeypatch.setattr(tp, "gpu_line", lambda **k: ("none", "GPU jobs: none"))
    demo, ctx, _registry = _page_component()
    refresh, poll = _named(demo, "_refresh"), _named(demo, "_poll")
    after_act = next(fn for fn in demo.fns.values() if fn.name == "_after_act")
    head = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None)
                == "shep-test-job-head")

    def card_of(fn, data):
        return data[fn.outputs.index(head)] if head in fn.outputs else None

    with served(demo) as base:
        browser = ServedSession(base, demo)
        assert browser.wait(browser.join(refresh, [None] * len(refresh.inputs)))
        browser.set_state(ctx.location, tp.PAGE)
        hold.set()
        answers = []
        ticking = threading.Thread(
            target=lambda: answers.append(browser.predict(poll, [None] * len(poll.inputs))),
            daemon=True)
        ticking.start()
        assert entered.wait(30), "the tick's read did not start"
        job.update(status="completed", end_time="2026-11-16T10:00:00+00:00",
                   elapsed_seconds=None)
        act = browser.join(after_act, [None] * len(after_act.inputs))
        assert browser.wait(act)
        release.set()
        ticking.join(30)
        (tick,) = answers
        shown = None
        for fn, data in ((after_act, browser.done[act]["output"]["data"]), (poll, tick)):
            value = card_of(fn, data)
            if value is not None and not (isinstance(value, dict)
                                          and value.get("__type__") == "update"):
                shown = value
        assert shown is not None and "Completed" in shown and "elapsed" not in shown, \
            "the tick brought back the card it read before the act's"
        # What the tick handed over draws the card from the job as it is.
        for fn in demo.fns.values():
            if fn.queue is not False and any(
                    (block._id, "change") in fn.targets for block in poll.outputs
                    if isinstance(block, gr.JSON)) and head in fn.outputs:
                token = tick[poll.outputs.index(next(
                    block for block in poll.outputs if (block._id, "change") in fn.targets))]
                if isinstance(token, dict) and token.get("__type__") != "update":
                    event = browser.join(fn, [None] * (len(fn.inputs) - 1) + [token])
                    assert browser.wait(event)
                    drawn = card_of(fn, browser.done[event]["output"]["data"])
                    assert drawn is None or "update" in str(drawn) or "Completed" in drawn
        browser.close()


def test_an_ended_run_is_handed_over_once_and_a_click_after_it_stays(site, monkeypatch):
    """F6, through the page served by Gradio: an act's read is held in its
    store read (it read the job, still running, before it) while the run ends
    with the older report. The poll (``queue=False``) sees the end and hands
    it over; the act ends after that tick. The hand-off chooses the ended
    run's report, and the tick after the late act hands nothing over, so a
    report clicked after the hand-off stays chosen. When the group's reads
    wrote what the session saw of the job too, the late act wrote back the
    running job it read, and the next tick handed the same end over again."""
    other = _disagreeing_copy(site)
    job = {"status": "running", "job_id": "test-9", "inputs": {}, "log": [], "outcome": None}
    monkeypatch.setattr(tp, "read_job", lambda **k: dict(job))
    monkeypatch.setattr(tp, "gpu_line", lambda **k: ("none", "GPU jobs: none"))
    real = tp.read_reports
    hold, entered, release = threading.Event(), threading.Event(), threading.Event()

    def read_reports(label, **kwargs):
        if hold.is_set():
            hold.clear()
            entered.set()
            assert release.wait(30), "never released"
        return real(label, **kwargs)

    monkeypatch.setattr(tp, "read_reports", read_reports)
    demo, ctx, _registry = _page_component()
    refresh, on_report, poll, after = (
        _named(demo, n) for n in ("_refresh", "_on_report", "_poll", "_after_job_change"))
    after_act = next(fn for fn in demo.fns.values() if fn.name == "_after_act")
    selected_report = on_report.outputs[0]

    with served(demo) as base:
        browser = ServedSession(base, demo)

        def tick():
            """The slow timer's request, as the browser sends it (queue=False);
            returns the hand-off it wrote."""
            return browser.predict(poll, [None, None, None])[poll.outputs.index(after.inputs[-1])]

        def hand_over(change):
            """The hand-off's listener, joined as the browser joins it."""
            return browser.join(after, [None, None, None, "demo", None, None, None, None,
                                        change])

        running = {"job_id": "test-9", "status": "running"}
        assert browser.wait(browser.join(refresh, [None] * len(refresh.inputs)))
        browser.set_state(ctx.location, tp.PAGE)
        newest = browser.state(selected_report)
        assert newest == site.run_id != other
        # The page open does not write what the session saw of the job: the
        # first tick does, and hands that over once.
        opened = tick()
        assert opened["now"] == running
        assert browser.wait(hand_over(opened))
        assert tick().get("__type__") == "update", "nothing changed since the last tick"

        hold.set()
        act = browser.join(after_act, [None, None, None, "demo", None, None, None, None])
        assert entered.wait(30), "the act's read did not start"
        job.update(status="completed", outcome={"report_dir": str(site.reports / other),
                                                "kept": True, "registered": False,
                                                "summary": "kept"})
        first = tick()
        assert first["seen"] == running
        assert first["now"] == {"job_id": "test-9", "status": "completed"}
        handed = hand_over(first)
        release.set()
        assert browser.wait(act, handed)
        assert browser.state(selected_report) == other, "the ended run's report is chosen"

        assert browser.wait(browser.join(on_report, [None] * 7, {"key": newest}))
        second = tick()
        assert second.get("__type__") == "update", \
            "the late act wrote back the job it read, so the poll handed the end over again"
        browser.close()
        assert browser.state(selected_report) == newest, "the click was undone"


def test_a_run_started_here_that_ends_before_the_next_tick_is_chosen(site, monkeypatch):
    """Through the page served by Gradio: Start's read (the act's, which
    writes nothing the poll reads) sees the run it started running, and the
    run ends with its report before the poll's next tick, so the poll never
    sees it busy. That tick hands over the job it saw last (the run before)
    and the one it reads now (this run, ended), and the hand-off chooses this
    run's report over the report the operator had chosen; the tick after
    hands nothing over. When only a run the poll saw busy could end, the
    operator's choice stayed and this run's report was never chosen."""
    other = _disagreeing_copy(site)
    job = {"status": "completed", "job_id": "test-1", "inputs": {}, "log": [], "outcome": None}
    monkeypatch.setattr(tp, "read_job", lambda **k: dict(job))
    monkeypatch.setattr(tp, "gpu_line", lambda **k: ("none", "GPU jobs: none"))
    demo, ctx, _registry = _page_component()
    refresh, on_report, poll, after = (
        _named(demo, n) for n in ("_refresh", "_on_report", "_poll", "_after_job_change"))
    after_act = next(fn for fn in demo.fns.values() if fn.name == "_after_act")
    selected_report = on_report.outputs[0]

    with served(demo) as base:
        browser = ServedSession(base, demo)

        def tick():
            return browser.predict(poll, [None, None, None])[poll.outputs.index(after.inputs[-1])]

        def hand_over(change):
            return browser.join(after, [None, None, None, "demo", None, None, None, None,
                                        change])

        assert browser.wait(browser.join(refresh, [None] * len(refresh.inputs)))
        browser.set_state(ctx.location, tp.PAGE)
        assert browser.wait(hand_over(tick()))
        assert tick().get("__type__") == "update"
        assert browser.wait(browser.join(on_report, [None] * 7, {"key": other}))
        assert browser.state(selected_report) == other != site.run_id

        job.update(job_id="test-2", status="running")
        assert browser.wait(browser.join(after_act, [None, None, None, "demo", None, None, None,
                                                     None]))
        assert browser.state(selected_report) == other, "an act's read decides no end"
        job.update(status="completed", outcome={"report_dir": str(site.reports / site.run_id),
                                                "kept": True, "registered": False,
                                                "summary": "kept"})
        handoff = tick()
        assert handoff["seen"] == {"job_id": "test-1", "status": "completed"}
        assert handoff["now"] == {"job_id": "test-2", "status": "completed"}
        assert browser.wait(hand_over(handoff))
        assert tick().get("__type__") == "update", "the end is handed over once"
        browser.close()
        assert browser.state(selected_report) == site.run_id, \
            "the run started here ended between two ticks and its report was not chosen"


def test_a_late_reload_never_writes_back_another_workspaces_checkpoints(site, monkeypatch):
    """Through the page served by Gradio, for an operator: ↻ held in its
    checkpoint listing while another workspace is chosen. The listing the
    session keeps (each row's registration and Start are asked from it) is the
    workspace chosen last. With ↻ in its own group the workspace's event ran
    beside it, and ↻ ended last and wrote back the first workspace's
    listing."""
    shutil.copytree(site.demo, site.ws / "other", symlinks=True)
    operator_mode(monkeypatch)
    job = {"status": "idle", "job_id": None, "inputs": {}, "log": [], "outcome": None}
    monkeypatch.setattr(tp, "read_job", lambda **k: dict(job))
    monkeypatch.setattr(tp, "gpu_line", lambda **k: ("none", "GPU jobs: none"))
    real = tp.read_models
    hold, entered, release = threading.Event(), threading.Event(), threading.Event()

    def read_models(name, request, **kwargs):
        if hold.is_set():
            hold.clear()
            entered.set()
            assert release.wait(30), "never released"
        return real(name, request, **kwargs)

    monkeypatch.setattr(tp, "read_models", read_models)
    demo, _ctx, _registry = _page_component()
    refresh, on_workspace, reload_ = (
        _named(demo, n) for n in ("_refresh", "_on_workspace", "_on_model_reload"))
    models_state = reload_.outputs[0]

    with served(demo) as base:
        browser = ServedSession(base, demo)
        assert browser.wait(browser.join(refresh, [None] * len(refresh.inputs)))
        listing = browser.state(models_state)
        assert listing["workspace"] == "demo" and listing["items"], listing
        hold.set()
        reloaded = browser.join(reload_, [None, None, "demo", LABEL, None, None, None, None,
                                          None])
        assert entered.wait(30), "the reload did not start"
        chosen = browser.join(on_workspace, [None, "other", LABEL, None, None, None, None])
        browser.wait(chosen, timeout=1.5)
        release.set()
        assert browser.wait(reloaded, chosen)
        browser.close()
        assert browser.state(models_state)["workspace"] == "other", \
            "the reload wrote back the first workspace's checkpoints"


def test_the_field_help_is_registered_with_the_labels():
    demo, _ctx, registry = _page_component()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    for elem_id, (title, _text) in tp.FIELD_HELP.items():
        assert elem_id in registry.targets() and by_id[elem_id].label == title
    assert all(key in registry for key in tp.TOPIC_HELP)
    assert not [b for b in demo.blocks.values() if getattr(b, "info", None)]


def test_the_page_is_registered_with_its_style_sheet():
    from src.webui import app, shell

    assert app.PAGES[shell.TEST] is tp.create_test_page
    assert tp.PAGE_CSS in app.PAGE_CSS and tp.PAGE_CSS in app.dashboard_head()
    for value, unit in re.findall(r"font-size:\s*([0-9.]+)(px|rem|em)", tp.PAGE_CSS):
        assert unit == "px" and float(value) >= 14
    from src.webui.style import SIDE, STACK

    selectors = [sel.strip() for head in re.findall(r"([^{}]+)\{", tp.PAGE_CSS)
                 if not head.strip().startswith(("@", SIDE, STACK)) for sel in head.split(",")]
    assert selectors and all(sel.startswith("#shep-page-test ") for sel in selectors), selectors


def test_a_direct_call_through_gradios_api_meets_the_same_check(site, monkeypatch):
    """The start event called as `gradio_client` would: refused off loopback
    before anything is started; from loopback under M1 the lists it starts
    from are this session's state, never what the client posts: the handler
    is given the session's own (empty) page and model list, not the forged
    ones, and a forged workspace, version and model are refused as choices the
    session was never offered. Nothing is started either way."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    manager = FakeManager()
    real = tp.start_test_run
    given = []

    def spy(*a, **k):
        given.append((a[0], a[1]))
        return real(*a, **{**k, "manager": manager})

    monkeypatch.setattr(tp, "start_test_run", spy)
    demo, _ctx, _registry = _page_component()
    app = FastAPI()
    gr.mount_gradio_app(app, demo, path="/ui")
    start = _named(demo, "_on_start")
    operator_mode(monkeypatch)
    forged_page = tp.read_page(operator_request(monkeypatch))
    forged_models = tp.read_models("demo", operator_request(monkeypatch))
    assert forged_page["versions"] and forged_models["items"]

    def call(base_url, choices):
        with TestClient(app, base_url=base_url) as client:
            started = client.post(f"/ui/gradio_api/call/{start.api_name}",
                                  json={"data": [forged_page, forged_models, *choices,
                                                 True, False]})
            assert started.status_code == 200, started.text
            stream = client.get(f"/ui/gradio_api/call/{start.api_name}/"
                                f"{started.json()['event_id']}")
        lines = stream.text.splitlines()
        event = next(x for x in lines if x.startswith("event: ") and x != "event: heartbeat")
        return event, lines[lines.index(event) + 1][len("data:"):]

    event, refused = call("http://192.0.2.10", [None, None, None, None, None])
    assert event == "event: complete" and "Refused: " in refused
    event, loopback = call(OPERATOR_BASE_URL, [None, None, None, None, None])
    assert event == "event: complete" and "Choose a data version the server lists" in loopback
    assert given and all(page == {} and models.get("items") == [] for page, models in given), \
        "the handler was given what the client posted"
    event, forged = call(OPERATOR_BASE_URL, ["demo", LABEL, None, site.key, None])
    assert event == "event: error" and "not in the list of choices" in forged
    assert manager.started == []


# --------------------------------------------------------------------------- review fixes (2026-10-21)
def _source(site, name, edit):
    """A copy of the intake source with `edit` applied to its cases."""
    source = json.loads((site.intake / "cases.json").read_text())
    edit(source["cases"])
    (site.intake / name).write_text(json.dumps(source))
    return source["cases"]


def test_a_malformed_source_is_refused_naming_no_case(site, monkeypatch):
    """D7: the importer lists a malformed source's problems by case index and
    case id; the page shows them by index only and says where the ids are."""
    request = operator_request(monkeypatch)

    def edit(cases):
        repeated = cases[2]["case_id"]
        cases[0] = {k: v for k, v in cases[0].items() if k != "phenotypes"}
        cases[0]["case_id"] = "MRN-0012345"
        cases[1] = dict(cases[1], case_id="MRN-7654321", disease=12)
        cases.append(dict(cases[3], case_id=repeated))
        cases.append(dict(cases[4], case_id="MRN\x07bell"))

    cases = _source(site, "bad.json", edit)
    preview, refusal = tp.preview_import("demo", "bad.json", "v-bad", "demo", "keep", request)
    assert preview is None
    for case_id in ("MRN-0012345", "MRN-7654321", cases[2]["case_id"], "bell", "case_id '"):
        assert case_id not in refusal, case_id
    assert "(4 problems)" in refusal and "cases[0]: missing ['phenotypes']" in refusal
    assert "cases[1]: disease is a number" in refusal
    assert f"cases[{len(cases) - 2}]: case_id repeats cases[2]" in refusal
    assert f"cases[{len(cases) - 1}]: case_id is not printable text" in refusal
    assert "scripts/import_cohort.py" in refusal and refusal.endswith("Nothing was written.")
    assert "MRN" not in tp._outcome_box("amber", refusal, tp.path_roots())


def test_a_long_list_of_problems_is_cut_and_counted(site, monkeypatch):
    def edit(cases):
        for index in range(12):
            cases[index] = {k: v for k, v in cases[index].items() if k != "phenotypes"}

    _source(site, "many.json", edit)
    _preview, refusal = tp.preview_import("demo", "many.json", "v-bad", "demo", "keep",
                                          operator_request(monkeypatch))
    assert "(12 problems)" in refusal and "cases[9]: missing" in refusal
    assert "cases[10]" not in refusal and "… 2 more" in refusal


def test_commit_meeting_a_malformed_source_names_no_case(site, monkeypatch):
    request = operator_request(monkeypatch)
    preview, _ = tp.preview_import("demo", "cases.json", "demo-test-2", "demo", "keep", request)
    assert preview is not None
    _source(site, "cases.json", lambda cases: cases.__setitem__(
        1, dict(cases[1], case_id="MRN-7654321", disease=12)))
    done, said, label = tp.commit_import_ui(preview, "demo", "cases.json", "demo-test-2", "demo",
                                            "keep", request)
    assert done is False and label is None
    assert "MRN-7654321" not in said and "cases[1]: disease is a number" in said
    assert not (site.registry / "versions" / "demo-test-2").exists()


def test_the_job_output_names_no_case():
    """A refusal in a run's output that names a case by its id (the importer's
    wording, `cohort_registry.parse_source`) is shown on the job card without
    it, in the failure line and in the output (D7)."""
    text = "MalformedSource: cases[3] (case_id 'MRN-9'): no phenotypes"
    assert "MRN-9" not in tp.case_free(text)
    job = {"status": "failed", "inputs": {}, "error_message": text, "log": ["Traceback", text]}
    html = tp.job_html(job, [], ("none", ""))
    assert "MRN-9" not in html and "cases[3]: no phenotypes" in html


def test_notes_that_cannot_be_read_are_said_never_no_note(site, monkeypatch):
    other = _disagreeing_copy(site)
    request = operator_request(monkeypatch)
    done, _said = tp.add_note(tp.read_reports(LABEL), other, "Dr A", "compared", request)
    assert done
    notes = site.reports / "notes.jsonl"
    notes.write_bytes(notes.read_bytes() + b"\xff\xfe broken\n")
    reports = tp.read_reports(LABEL)
    assert reports["notes"] == [] and reports["notes_problems"]
    # Through the page's own wiring: choosing the report renders the block.
    demo, _ctx, _registry = _page_component()
    monkeypatch.setattr(tp, "read_job", lambda **k: {"status": "idle", "inputs": None,
                                                    "outcome": None, "log": []})
    page = tp.read_page(request)
    out = _named(demo, "_on_report").fn(page, reports, {}, None, "demo", LABEL, None,
                                        SimpleNamespace(_data={"key": other}), request)
    html = out[4]          # the selection, then the reports' title, table and result
    assert "No investigation note is kept" not in html
    assert "The investigation notes cannot all be read" in html and str(site.root) not in html


def _point_report_at_site(site):
    """The kept report as if measured on the site's own demo workspace (the
    module's measurement ran on the tree the site was copied from)."""
    path = site.reports / site.run_id / "report.json"
    report = json.loads(path.read_text())
    report["manifest"]["data_dir"] = str(site.demo)
    report["manifest"]["checkpoint_path"] = str(site.checkpoint)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))


def test_register_from_a_copy_files_beside_the_checkpoint_the_run_read(site, monkeypatch):
    """The page shows a copy of the workspace the run measured: Register files
    the record beside the file the run read (where the list looks), so the
    list says registered and the same report is not filed in a second ledger."""
    _point_report_at_site(site)
    shutil.copytree(site.demo, site.ws / "copy", symlinks=True)
    page, request = _page(monkeypatch)
    demo_ledger = site.checkpoint.parent / "evaluations.json"
    copy_ledger = (site.ws / "copy" / site.checkpoint.relative_to(site.demo)).parent / \
        "evaluations.json"
    models_copy = tp.read_models("copy", request)
    refined = tp.refine_registration(page, tp.read_reports(LABEL), site.run_id, models_copy,
                                     request=request)
    state = tp.find(refined["runs"], "run_id", site.run_id)["state"]
    assert state["registered"] == "not_registered"
    assert state["checked_beside"] == f"{site.key} in workspace demo"
    done, said = tp.register_selected(page, refined, models_copy, "copy", site.run_id, request)
    assert done is True and said.startswith("Registered as record 1.")
    assert demo_ledger.exists() and not copy_ledger.exists()
    reports = tp.read_reports(LABEL)
    assert tp.find(reports["runs"], "run_id", site.run_id)["state"]["registered"] == "registered"
    assert tp.register_state(page, reports, site.run_id) == (False, "Already registered.")


def test_a_report_changed_since_the_list_is_never_placed_from_its_new_bytes(site, monkeypatch):
    """Round 3 of the review fixes (F2): the selected report's registration is
    asked again with the ledger placed from the bytes the list showed (the
    report read bound to their digest), so a report replaced since the list was
    read is never decoded to place a ledger, and the list's own answer stands.
    The control: the same report unchanged reads "not registered" beside the
    file the run read."""
    _point_report_at_site(site)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    reports = tp.read_reports(LABEL)
    assert tp.find(reports["runs"], "run_id", site.run_id)["state"]["registered"] == \
        "cannot_be_checked"
    control = tp.refine_registration(page, reports, site.run_id, models, request=request)
    state = tp.find(control["runs"], "run_id", site.run_id)["state"]
    assert state["registered"] == "not_registered" and state["checked_beside"] == site.key

    path = site.reports / site.run_id / "report.json"
    _swap(path, _another_valid_report(path), "rename")
    _never_decoded(monkeypatch)

    assert tp.refine_registration(page, reports, site.run_id, models,
                                  request=request) is reports


def test_the_registered_answer_reads_the_report_once_the_registrations_read(site, monkeypatch):
    """Round 3's verification (A-3): the selected report's registered answer
    places the ledger from the one read the registration's helper makes
    (`report_state`'s ``place``), so the page reads no report itself: the
    report is read once for the answer, and the answer is the ledger beside
    the listed file Register would use."""
    from tests.fixtures.replacement import hook_read_once

    _point_report_at_site(site)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    reports = tp.read_reports(LABEL)
    path = (site.reports / site.run_id / "report.json").resolve()
    reads = []
    hook_read_once(monkeypatch,
                   on_entry=lambda read: reads.append(read) if read.resolve() == path else None)

    refined = tp.refine_registration(page, reports, site.run_id, models, request=request)

    state = tp.find(refined["runs"], "run_id", site.run_id)["state"]
    assert state["registered"] == "not_registered"
    assert state["checked_beside"] == site.key
    assert len(reads) == 1, reads


def test_every_row_says_not_registered_for_an_operator_and_keeps_it(site, monkeypatch):
    """The list's "cannot be checked" is asked again for every row from the
    session's checkpoint listing (no checkpoint opened), so an unregistered
    report reads "not registered" whichever row is selected; a row whose
    place needs another workspace's listing keeps the list's answer."""
    from src.api.services import server_lists

    _point_report_at_site(site)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    shutil.copytree(site.demo, site.ws / "copy", symlinks=True)
    models_copy = tp.read_models("copy", request)
    page = tp.read_page(request)
    monkeypatch.setattr(server_lists, "list_checkpoints",
                        lambda *a, **k: pytest.fail("a row opened another workspace's checkpoints"))
    reports = tp.read_reports(LABEL)
    rows = tp.refine_rows(page, reports, models)
    state = tp.find(rows["runs"], "run_id", site.run_id)["state"]
    assert state["registered"] == "not_registered" and state["checked_beside"] == site.key
    html = tp.reports_html(rows, None, models=models)
    assert _report_rows(html)[site.run_id][3] == "not registered"
    assert "cannot be checked from the list" not in html
    from_copy = tp.refine_rows(page, reports, models_copy)
    assert tp.find(from_copy["runs"], "run_id", site.run_id)["state"]["registered"] == \
        "cannot_be_checked"
    readonly = tp.read_page(_readonly())
    assert tp.refine_rows(readonly, reports, models) is reports


def test_a_refused_report_lookup_is_said_not_raised(site, monkeypatch):
    """A report gone from the store between the list and Register is said
    ("Nothing was registered."), never a page error."""
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    reports = tp.read_reports(LABEL)
    shutil.rmtree(site.reports / site.run_id)
    done, said = tp.register_selected(page, reports, models, "demo", site.run_id, request)
    assert done is False and said.endswith("Nothing was registered.")


def test_the_jobs_times_are_utc_as_the_report_rows_are():
    import os
    import time

    before = os.environ.get("TZ")
    os.environ["TZ"] = "Asia/Taipei"
    time.tzset()
    try:
        job = {"status": "completed", "start_time": "2026-10-21T10:06:07",
               "end_time": "2026-10-21T10:07:30", "outcome": None,
               "inputs": {"version_label": LABEL, "checkpoint_sha256": "a" * 64}}
        assert "ended 02:07:30 UTC" in tp.job_head_html(job)
        assert _identity_rows(tp.job_body_html(job, [], ("none", "")))["Started"] == (
            "02:06:07 UTC")
        running = dict(job, status="running", end_time=None)
        assert "started 02:06:07 UTC" in tp.reports_html({"label": LABEL, "runs": []}, None,
                                                          job=running)
    finally:
        if before is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = before
        time.tzset()


def test_a_busy_gpu_is_the_start_reason_before_any_choice(site, monkeypatch):
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    busy = ("busy", "GPU job: test run (job runner)")
    for ws, label, key in ((None, None, None), ("demo", LABEL, None)):
        allowed, why = tp.start_state(page, models, ws, label, key, None, busy)
        assert allowed is False and why.startswith("Start disabled: GPU job: test run (job runner).")
    allowed, why = tp.start_state(page, models, "demo", LABEL, None, None, ("unread", ""))
    assert why.startswith("Start disabled: the GPU lock cannot be read")


def _runs(n):
    return [{"run_id": f"202610{20 - i // 24:02d}T{23 - i % 24:02d}0000Z-{'a' * 12}-{i:08x}",
             "when": f"run {i}", "sha12": "a" * 12, "state": {"kept": "kept",
                                                            "registered": "not_registered"}}
            for i in range(n)]


def test_report_rows_that_would_read_the_same_are_told_apart_by_their_run_id():
    """The 2026-10-25 review's Test screenshot: two rows read "2026-10-24 11:10
    UTC · model f6f7e5cab280" with different registration states. The Report
    cell of rows whose times read alike gets the seconds, then the run id's
    last part; other rows keep the minute; every row still selects by its full
    run id. Their model is its own cell (its 12 hex digits for weights the
    list does not hold)."""
    sha = "f6f7e5cab280"
    ids = [f"20261024T111047Z-{sha}-1885e8bd", f"20261024T111012Z-{sha}-0c0ffee0",
           f"20261024T111012Z-{sha}-7a7a7a7a", f"20261024T095501Z-{sha}-11111111"]
    runs = [{"run_id": x, "when": tp.run_time(x), "sha12": sha,
             "state": {"kept": "kept", "registered": "not_registered"}} for x in ids]
    html = tp.reports_html({"label": LABEL, "runs": runs}, ids[0])
    rows = _report_rows(html)
    assert list(rows) == ids
    text = {key: cells[0] for key, cells in rows.items()}
    assert len(set(text.values())) == 4
    assert text[ids[0]] == "2026-10-24 11:10:47 UTC"
    assert text[ids[1]] == "2026-10-24 11:10:12 UTC · run …0c0ffee0"
    assert text[ids[2]] == "2026-10-24 11:10:12 UTC · run …7a7a7a7a"
    assert text[ids[3]] == "2026-10-24 09:55 UTC"
    assert all(cells[1] == sha for cells in rows.values())
    # The selected row is the plain table's `aria-current` (the skill's §8).
    assert f'data-key="{ids[0]}" tabindex="0" aria-current="true">' in html
    assert html.count("aria-current") == 1 and "aria-selected" not in html
    assert tp.run_time(ids[0], seconds=True) == "2026-10-24 11:10:47 UTC"


def test_report_rows_are_named_by_the_shared_run_time_and_distinct_names(monkeypatch):
    """Review of the browser-check fixes (AGENTS.md: no second implementation):
    the Test page names its report rows by the one `renderers.run_time` and
    tells alike names apart by the one `renderers.distinct_names`, as Models ›
    Results does, not by copies of its own."""
    from src.webui.components import renderers as r

    assert tp.run_time is r.run_time
    calls = []
    shared = r.distinct_names

    def spy(items, name_at, **kwargs):
        calls.append(list(items))
        return shared(items, name_at, **kwargs)

    monkeypatch.setattr(r, "distinct_names", spy)
    sha = "f6f7e5cab280"
    ids = [f"20261024T111012Z-{sha}-0c0ffee0", f"20261024T111012Z-{sha}-7a7a7a7a"]
    runs = [{"run_id": x, "when": tp.run_time(x), "sha12": sha,
             "state": {"kept": "kept", "registered": "not_registered"}} for x in ids]
    html = tp.reports_html({"label": LABEL, "runs": runs}, ids[0])
    # The rows' names (the listed models' Run options, `model_list.pair_words`,
    # are told apart by the same rule, here over no listed run).
    assert [items for items in calls if items] == [ids]
    assert _report_rows(html)[ids[1]][0] == "2026-10-24 11:10:12 UTC · run …7a7a7a7a"


def test_a_long_report_list_shows_the_newest_and_keeps_the_selected():
    runs = _runs(20)
    reports = {"label": LABEL, "runs": runs}
    html = tp.reports_html(reports, runs[15]["run_id"])
    old = re.findall(r'<tr class="[^"]*shep-test-rep-old[^"]*" data-key="([^"]+)"', html)
    assert old == [x["run_id"] for i, x in enumerate(runs) if i >= tp.ROWS_SHOWN and i != 15]
    assert f'data-key="{runs[15]["run_id"]}"' in html
    assert "Show all 20 reports ›" in html and f"Show the newest {tp.ROWS_SHOWN} only" in html
    short = tp.reports_html({"label": LABEL, "runs": runs[:tp.ROWS_SHOWN]}, None)
    assert "shep-test-rep-old" not in short and "Show all" not in short
    # A compact single-select table (the skill's §4, §8): one row per run,
    # named by the table, its columns one or two words each; its title and
    # count are the panel's (`reports_header_html`), "Show all" and the notes
    # outside the table; an older report's row is hidden whole.
    assert html.startswith('<div class="shep-test-replist"><div class="shep-tbl-wrap '
                           'shep-test-repwrap"><table class="shep-tbl shep-test-reps" '
                           f'aria-label="Reports of {LABEL}"><thead><tr>')
    assert re.findall(r'<th scope="col">([^<]*)</th>', html) == list(tp.REPORT_COLUMNS)
    assert all(len(head.split()) <= 2 for head in tp.REPORT_COLUMNS)
    assert len(_report_rows(html)) == 20 and "role=\"list" not in html
    assert '<div class="shep-test-more">' in html.split("</table>")[1]
    assert ".shep-test-all .shep-test-rep-old { display: table-row !important; }" in tp.PAGE_CSS
    head = tp.reports_header_html(reports)
    assert "<h2>Reports</h2>" in head and "20 of the selected version" in head
    # Its rows select by click and by Enter or Space on a focused row.
    assert "tr[data-key]" in tp._REPORTS_JS and "'Enter'" in tp._REPORTS_JS
    # A version with no report: its empty state, and no empty table.
    empty = tp.reports_html({"label": LABEL, "runs": []}, None)
    assert "<table" not in empty and tp.e(tp.REPORTS_NONE_YET) in empty


def test_a_run_that_ended_with_no_report_is_a_row_until_the_next():
    cancelled = {"status": "cancelled", "start_time": "2026-10-21T02:00:00",
                 "end_time": "2026-10-21T02:01:00",
                 "inputs": {"version_label": LABEL, "checkpoint_sha256": "b" * 64},
                 "outcome": {"report_dir": None, "kept": False, "registered": False}}
    html = tp.reports_html({"label": LABEL, "runs": []}, None, job=cancelled)
    assert 'data-status="cancelled"' in html and "interrupted: cancelled" in html
    assert "no report kept" in html and tp.e(tp.REPORTS_NONE_YET) not in html
    failed = dict(cancelled, status="failed", exit_meaning="refused before anything was written")
    assert "failed (refused before anything was written)" in tp.reports_html(
        {"label": LABEL, "runs": []}, None, job=failed)
    runs = _runs(1)
    listed = dict(cancelled, outcome={"report_dir": f"/x/{runs[0]['run_id']}", "kept": False})
    assert 'data-status="cancelled"' not in tp.reports_html({"label": LABEL, "runs": runs}, None,
                                                            job=listed)
    other = dict(cancelled, inputs={"version_label": "another", "checkpoint_sha256": "b" * 64})
    assert 'data-status' not in tp.reports_html({"label": LABEL, "runs": []}, None, job=other)


def test_the_r10_help_sits_beside_the_r10_lines(site):
    reports = tp.read_reports(LABEL)
    html = tp.result_html(reports, site.run_id, [])
    label = html[html.index('class="shep-rec-label"'):]
    label = label[:label.index("</div>")]
    help_label = f"Help: {renderers.R10_TITLE}"
    assert help_label not in label and "Help: Disease task result" in label
    r10 = html[html.index(f">{renderers.R10_TITLE}"):]
    assert r10.index(help_label) < r10.index('class="shep-v') < r10.index(
        '<table class="shep-r10"')


def _css_rules(css):
    """(selector, declarations) for each rule of a page's style sheet, the
    page's own id taken off the selector and its @media blocks opened."""
    css = re.sub(r"@media[^{]*\{", "", css).replace(tp._PAGE + " ", "")
    return [(sel.strip(), body) for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)]


def _tokens(block):
    """The ids and classes of `block` and every component inside it."""
    own = {"#" + block.elem_id} if getattr(block, "elem_id", None) else set()
    own |= {"." + c for c in (getattr(block, "elem_classes", None) or [])}
    for child in getattr(block, "children", []):
        own |= _tokens(child)
    return own


def test_the_run_panels_foot_keeps_start_the_jobs_state_and_cancel_out_of_its_one_scroll_region():
    """The skill's §1.1 and §2. At a wide size the Run panel is three rows: its
    title; its body, the one scroll region, holding the choices and then the
    job's details (what it was given, what it left, its output) under their
    heading; and a foot in flow that never scrolls, holding Start, its reason,
    the action line and the job's title row (its state, Cancel beside it).
    The round's closing check measured two stacked regions (the choices 238
    px of 484 shown at 1850 x 960); one region holding everything then put
    Start under its fold before any job, and the job's state and Cancel
    scrolled away with the output."""
    demo, _ctx, _registry = _page_component()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}

    def names(block):
        return [getattr(c, "elem_id", None) or (c.elem_classes or [type(c).__name__])[0]
                for c in block.children]

    run = by_id["shep-test-run"]
    assert names(run) == ["shep-test-run-hd", "shep-test-bd", "shep-test-runft"]
    body, foot = run.children[1], run.children[2]
    assert names(body)[-2:] == ["shep-test-fixed", "shep-test-job"]
    # A refused or empty listing's reason first (V-15), the Run row's note
    # (L7 SW-LK10), then the choices.
    assert names(body)[0] == "shep-test-model-row"
    assert names(body.children[0])[0] == "shep-test-model-why"
    # The card's heading names the run once, in the card (none while no run
    # was started, R-81); its state is the foot's alone.
    running = {"status": "running", "inputs": {}, "log": []}
    card = tp.job_body_html(running, [], ("busy", "GPU"))
    assert "Latest test run" in card and "shep-test-pill" not in card
    assert by_id["shep-test-job"].value == ""
    assert by_id["shep-test-cancel"].visible is False
    assert names(foot) == ["shep-test-start", "shep-test-start-reason", "shep-test-action",
                           "shep-test-job-top", "shep-test-cancel-reason"]
    assert names(foot.children[3]) == ["shep-test-job-head", "shep-test-cancel"]
    # The classes the job's two parts draw, running and ended.
    job = {"status": "running", "inputs": {"version_label": LABEL}, "log": ["a line"],
           "elapsed_seconds": 5}
    ended = dict(job, status="cancelled", end_time="2026-11-14T05:00:00Z",
                 outcome={"kept": False, "registered": False, "summary": "Cancelled."})

    def drawn(*html):
        return {"." + c for attr in re.findall(r'class="([^"]+)"', "".join(html))
                for c in attr.split()}

    head = drawn(tp.job_head_html(job), tp.job_head_html(ended))
    details = drawn(tp.job_body_html(job, [], ("busy", "GPU")),
                    tp.job_body_html(ended, [], ("none", "")))
    around = {"#shep-test-row", ".shep-page", ".shep-test", ".shep-panel", ".shep-test-col",
              "#shep-test-run"}
    in_panel = _tokens(run) | head | details | around
    in_foot = _tokens(foot) | head | around
    scrolls = re.compile(r"overflow(-y)?: (auto|scroll)|max-height|position: (sticky|fixed)")
    regions = []
    for selectors, rule in _css_rules(tp.PAGE_CSS):
        for selector in selectors.split(","):
            needs = set(re.findall(r"[#.][\w-]+", selector))
            if not needs or not scrolls.search(rule):
                continue
            # No rule that could reach the foot or anything in it scrolls,
            # caps or pins it.
            assert not needs <= in_foot, selector
            if needs <= in_panel:
                regions.append(selector.strip())
    # The body is the panel's one scroll region.
    assert regions == [".shep-test-col > .shep-test-bd"]
    from tests.unit.test_webui_shell import side

    wide = side(tp.PAGE_CSS)
    assert re.search(r"\.shep-test-col > \.shep-test-bd \{ flex: 1 1 auto !important; "
                     r"min-height: 0; overflow: auto !important;", wide)
    # The foot keeps its own height, the body taking what is left.
    assert "#shep-test-runft { flex: none !important; }" in wide


def test_the_import_drawer_lies_over_the_panels_at_every_size():
    """Wide, the drawer covers the panels' right side; narrow, the window. Its
    body scrolls inside it under its title either way."""
    from tests.unit.test_webui_shell import side, stack

    wide, narrow = side(tp.PAGE_CSS), stack(tp.PAGE_CSS)
    assert "{ position: fixed !important;" in narrow.split(".shep-test-drawer ", 1)[1]
    assert re.search(r"\.shep-test-drawer > \.shep-test-bd \{[^}]*overflow: auto", narrow)
    # Over the panels, Gradio's own field labels included (position: relative;
    # z-index: 40 in 6.30), which would otherwise show through the drawer.
    for css in (wide, narrow):
        z = re.search(r"\.shep-test-drawer \{[^}]*z-index: (\d+)", css)
        assert z and int(z.group(1)) > 40, css[:40]


def inside_ids(block, elem_id):
    """Whether a component with `elem_id` is anywhere inside `block`."""
    return any(getattr(c, "elem_id", None) == elem_id or inside_ids(c, elem_id)
               for c in getattr(block, "children", []))


def test_add_note_sits_at_the_result_panels_foot_outside_the_scrolling_report():
    """While runs disagree, "Add note" is the one action left on the Result
    panel (Register is off). At wide sizes the reports table and the report
    scroll inside the panel's one body, under its title (in flow); the note
    form is a region of its own between that body and Register's footer, so
    Add note is in view without scrolling the report."""
    demo, _ctx, _registry = _page_component()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    panel = [getattr(c, "elem_id", None) or c.elem_classes[0]
             for c in by_id["shep-test-result"].children]
    assert panel == ["shep-test-reports-hd", "shep-test-bd", "shep-test-noteform",
                     "shep-test-note-out", "shep-panel-ft"]
    body = by_id["shep-test-result"].children[1]
    assert [getattr(c, "elem_id", None) for c in body.children] == [
        "shep-test-reports", "shep-test-result-body"]
    # The Data panel no longer holds the reports (they head the Result panel).
    assert not inside_ids(by_id["shep-test-cohort"], "shep-test-reports")

    def inside(block, elem_id):
        return any(getattr(c, "elem_id", None) == elem_id or inside(c, elem_id)
                   for c in getattr(block, "children", []))

    for field in ("shep-test-note-who", "shep-test-note-what", "shep-test-note-add",
                  "shep-test-note-reason"):
        assert inside(by_id["shep-test-noteform"], field), field
    from tests.unit.test_webui_shell import side

    wide = side(tp.PAGE_CSS)
    assert re.search(r"#shep-test-noteform \{[^}]*max-height: 50%[^}]*overflow: auto", wide)


def test_a_long_note_scrolls_inside_what_was_found_and_leaves_add_note_in_view():
    """"What was found" grows from two lines to at most four, then scrolls
    inside itself. Unbounded (Gradio's default is 20 lines), an eight-line
    note filled the note form's region (at most half the panel, wide) and
    pushed Add note out of it at every wide size, squeezing the report above
    to its title (measured in the browser at 1400x760, 1536x760, 1920x1080)."""
    demo, _ctx, _registry = _page_component()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    what = by_id["shep-test-note-what"]
    assert what.lines == 2
    assert (what.max_lines or max(what.lines, 20)) <= 4


def test_opening_the_import_moves_the_focus_to_the_drawer():
    demo, _ctx, _registry = _page_component()
    current, js = _named(demo, "_on_open_import"), []
    for _ in range(4):
        nxt = [d for d in demo.fns.values() if d.trigger_after == current._id]
        if not nxt:
            break
        current = nxt[0]
        js.append(current.js or "")
    assert any("#shep-test-import-title h2" in text and ".focus(" in text for text in js)


def test_a_long_choice_ends_in_an_ellipsis_with_its_whole_text_as_title():
    demo, _ctx, _registry = _page_component()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    for elem_id in ("shep-test-workspace", "shep-test-version", "shep-test-model-run",
                    "shep-test-model", "shep-test-intake", "shep-test-role", "shep-test-partial"):
        assert "shep-test-select" in by_id[elem_id].elem_classes, elem_id
    # The ellipsis ends before the chevron Gradio draws over the input's right.
    assert ".shep-test-select input { padding-right: 26px !important; }" in tp.PAGE_CSS
    # The titled ellipsis is the shell's one rule for every select (M5-3): the
    # page's own hover script, which left a stale title when the value changed
    # after the pointer had passed, is gone.
    from src.webui import shell, style

    assert '.gradio-container .wrap-inner input[role="combobox"] { text-overflow: ellipsis; }' \
        in style.SHELL_CSS
    assert 'input[role="combobox"]' in shell.SHELL_HEAD
    assert by_id["shep-test-cohort-hd"].js_on_load == renderers.KEEP_OPEN_JS
    # Run → Checkpoint: each option short (at most 40 characters and one
    # separator), the distinguishing token first, so an option seldom needs
    # the ellipsis at all; a file's kind is its details' first row, never its
    # option (the 91-character option the ops-wiring survey measured).
    item = {"architecture": "gatv2", "run": RUN, "filename": "best.pt", "epoch": 12,
            "sha256": "c" * 64}
    models = {"items": [item]}
    runs, checkpoints = ml.parents_update(tp.model_parents(models), tp.item_key(item), first=False)
    assert runs["choices"] == [("2026-10-17 09:15 · GATv2", ml.item_run_key(item))]
    assert checkpoints["choices"] == [("best.pt · recorded epoch 12", tp.item_key(item))]
    # A record never read says its state's own word in the details.
    assert '<dd class="shep-v">Record unknown</dd>' in tp.model_html({}, models, tp.item_key(item))


def test_a_read_only_request_cannot_type_a_note(site, monkeypatch):
    _disagreeing_copy(site)
    demo, _ctx, _registry = _page_component()
    monkeypatch.setattr(tp, "read_job", lambda **k: {"status": "idle", "inputs": None,
                                                    "outcome": None, "log": []})
    page = tp.read_page(_readonly())
    reports = tp.read_reports(LABEL)
    out = _named(demo, "_on_report").fn(page, reports, {}, None, "demo", LABEL, None,
                                        SimpleNamespace(_data={"key": site.run_id}), _readonly())
    note_btn, note_who, note_what, note_reason = out[-4:]
    assert note_who == gr.update(interactive=False) and note_what == gr.update(interactive=False)
    # Refused: disabled, with the lock, beside D8's reason (the WebUI skill §3).
    assert note_btn == gr.update(value="🔒 Add note", interactive=False)
    assert "Operator controls only" in note_reason


#: Each gated button of the page and the element that holds its reason beside it.
GATED_REASONS = {"shep-test-import-open": "shep-test-import-reason",
                 "shep-test-model-reload": "shep-test-model-why",
                 "shep-test-start": "shep-test-start-reason",
                 "shep-test-cancel": "shep-test-cancel-reason",
                 "shep-test-note-add": "shep-test-note-reason",
                 "shep-test-register": "shep-test-register-reason"}


@pytest.mark.parametrize("status", ["idle", "running"])
def test_every_gated_button_is_locked_only_when_refused_and_then_says_why(site, monkeypatch,
                                                                          status):
    """The one gated-button rule on each of the page's gated buttons (the WebUI
    skill's §3): for an operator, no lock and no "Operator controls only"; for
    a read-only request, disabled with the lock and D8's short reason beside
    it, whatever the job's state or the selection (B0a review: Cancel with no
    job running and Register with no report chosen showed the lock alone)."""
    monkeypatch.setattr(tp, "read_job", lambda **k: {"status": status, "inputs": None,
                                                    "outcome": None, "log": []})
    monkeypatch.setattr(tp, "gpu_line", lambda **k: ("none", "GPU jobs: none"))
    demo, _ctx, _registry = _page_component()
    refresh = _named(demo, "_refresh")
    assert set(GATED_REASONS) == {f"shep-test-{x}" for x in (
        "import-open", "model-reload", "start", "cancel", "note-add", "register")}
    assert {by for by in GATED_REASONS} <= {b.elem_id for b in refresh.outputs}
    for operator in (True, False):
        request = operator_request(monkeypatch) if operator else _readonly()
        out = dict(zip([b.elem_id for b in refresh.outputs],
                       refresh.fn(None, None, None, None, {}, None, None, request)))
        for button, beside in GATED_REASONS.items():
            label, reason = out[button]["value"], out[beside]
            if button == "shep-test-cancel" and status not in ("running", "stopping"):
                # R-81: Cancel is drawn only while a run is busy, so no
                # reason stands beside a button that is not there.
                assert out[button]["visible"] is False and reason == "", (status, reason)
                continue
            if operator:
                assert tp.LOCK not in label and "Operator controls only" not in reason, button
            else:
                assert label.startswith(f"{tp.LOCK} ") and out[button]["interactive"] is False
                assert "Operator controls only: " in reason and tp.LOCK in reason, button
        assert out["shep-test-cancel"]["visible"] is (status in ("running", "stopping"))
        if operator:
            assert out["shep-test-cancel"]["interactive"] is (status == "running")
    # No report chosen (none kept yet): Register's reason is still D8's.
    refused = {"access": {"allowed": False, "condition": "not_loopback"}}
    assert tp.register_state(refused, {"runs": []}, None) == (
        False, "Operator controls only: request not local.")
    # L7 SW-LK6: and an operator's says why, in the shared form (never a
    # silent disabled button): the registry the Data panel names, else what
    # to choose.
    operator = {"access": {"allowed": True}}
    assert tp.register_state(operator, {"runs": []}, None) == (
        False, "Register disabled: no report is listed yet.")
    assert tp.register_state(operator, {"runs": [{"run_id": "r"}]}, None) == (
        False, "Register disabled: choose a kept report in the list first.")
    first_use = {**operator, "registry_notice": {
        "key": "registry", "kind": "neutral", "summary": "No results registry is set.",
        "short": "no results registry is set"}}
    assert tp.register_state(first_use, {"runs": []}, None) == (
        False, "Register disabled: no results registry is set (see Data).")


def test_the_empty_run_and_checkpoint_selects_say_why_they_are_empty():
    """L7 SW-LK10: at first use and in an empty folder the Run and Checkpoint
    selects were empty grey boxes, and the ↻ beside Run off, with no words
    (the skill's §9: an empty region says what will appear). Their note
    under the Run row (`listing_why_html`) is Run's relation to the Data
    panel's workspaces notice, never the notice again; with the folder set,
    what to choose there; nothing once a workspace is listed."""
    operator = {"access": {"allowed": True}}
    none = {"items": [], "workspace": None}
    first_use = {**operator, "workspaces_notice": {
        "key": "workspaces_root", "kind": "neutral", "summary": "No workspaces folder is set.",
        "short": "no workspaces folder is set"}}
    said = tp.listing_why_html(first_use, none)
    assert 'data-notice-rel="workspaces_root"' in said and "data-notice=" not in said
    assert "Run disabled: no workspaces folder is set (see Data)." in said
    assert "Run disabled: no workspace yet (see Data)." in tp.listing_why_html(operator, none)
    assert "Run disabled: choose a workspace in Data first." in tp.listing_why_html(
        {**operator, "workspaces": [{"name": "demo"}]}, none)
    assert tp.listing_why_html(operator, {"items": [], "workspace": "demo"}) == ""



#: The Test page for a read-only browser (no deployment mode, so D8 refuses
#: every request) at three windows: the ↻ beside the Run field once its lock
#: is drawn, brought into view as a reader brings it, and the nearest visible
#: "Operator controls only: …" text, as the verification's part RB2 measured
#: them (scratchpad/verify/rb2/run2/pw/a33_beside.cjs): the gap between the
#: two boxes, where the reason sits and how far it is from the ↻'s row (the
#: Run field's label is all between them), whether each is wholly on screen
#: (inside the window and every box that clips it), and how often the
#: listing's refusal is said in the Run panel. A screen of each window under
#: argv[3].
_RELOAD_REASON_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const [base, shots] = [process.argv[2], process.argv[3]];
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h] of [[1850, 960], [1440, 1600], [1280, 720]]) {
    const page = await (await browser.newContext({ viewport: { width: w, height: h },
                                                   deviceScaleFactor: 1 })).newPage();
    await page.addInitScript(QUEUE_JS);
    const idle = () => page.waitForFunction(() => window.__shepQueue && window.__shepQueue.idle(),
                                            null, { timeout: 60000 });
    await page.goto(base, { waitUntil: 'load' });
    await page.waitForSelector('#shep-chips .shep-chip', { timeout: 60000 });
    await idle();
    await page.click('#shep-tab-research-button'); await idle();
    await page.click('#shep-page-test-button');
    await page.waitForSelector('#shep-page-test-button[aria-selected="true"]');
    await idle();
    await page.waitForFunction(() => /🔒/.test(
      (document.getElementById('shep-test-model-reload') || {}).innerText || ''), null, { timeout: 30000 });
    await page.waitForFunction(() => /listing/.test(
      (document.getElementById('shep-test-run') || {}).innerText || ''), null, { timeout: 30000 });
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    out[w + 'x' + h] = await page.evaluate(() => {
      const T = (s) => (s || '').replace(/\s+/g, ' ').trim();
      const b = document.getElementById('shep-test-model-reload');
      b.scrollIntoView({ block: 'nearest' });
      const box = (e) => { const r = e.getBoundingClientRect(); return { l: r.left, t: r.top, r: r.right, b: r.bottom }; };
      const vis = (e) => { const r = e.getBoundingClientRect(), s = getComputedStyle(e);
        return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
      // Wholly on screen: inside the window and inside every box around it that clips.
      const whole = (e) => { const q = box(e); let x = { l: 0, t: 0, r: innerWidth, b: innerHeight };
        for (let a = e.parentElement; a; a = a.parentElement) { const s = getComputedStyle(a);
          if (/(auto|scroll|hidden|clip)/.test(s.overflowY + s.overflowX)) { const p = box(a);
            x = { l: Math.max(x.l, p.l), t: Math.max(x.t, p.t), r: Math.min(x.r, p.r), b: Math.min(x.b, p.b) }; } }
        return q.l >= x.l - 1 && q.t >= x.t - 1 && q.r <= x.r + 1 && q.b <= x.b + 1; };
      const said = (e) => /Operator controls only:/.test(e.innerText || '');
      const reasons = [...document.querySelectorAll('[role=tabpanel] *')]
        .filter((e) => vis(e) && said(e) && ![...e.children].some(said));
      const a = box(b); let best = null;
      for (const r of reasons) { const q = box(r);
        const dx = Math.max(0, q.l - a.r, a.l - q.r), dy = Math.max(0, q.t - a.b, a.t - q.b);
        const vO = Math.min(a.b, q.b) - Math.max(a.t, q.t), hO = Math.min(a.r, q.r) - Math.max(a.l, q.l);
        const where = vO > 0 ? (q.l >= a.r - 1 ? 'same row, right' : q.r <= a.l + 1 ? 'same row, left' : 'overlapping')
          : hO > 0 ? (q.t >= a.b - 1 ? 'directly below' : 'directly above') : 'apart';
        const gap = Math.round(Math.hypot(dx, dy));
        if (!best || gap < best.gap) { best = { gap, where, text: T(r.innerText), whole: whole(r), bottom: q.b }; } }
      // From the reason to the top of the ↻'s row (the Run field's label).
      if (best) { best.toRow = Math.round(b.closest('.shep-field-row').getBoundingClientRect().top - best.bottom); }
      const run = document.getElementById('shep-test-run').innerText;
      return { button: T(b.innerText), disabled: b.disabled, whole: whole(b), reason: best,
               listing: run.split('listing the checkpoints opens them').length - 1
                 + run.split('listing checkpoints opens them').length - 1 };
    });
    await page.screenshot({ path: `${shots}/reload-reason_${w}x${h}.png` });
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_read_only_viewer_reads_why_the_reload_is_locked_beside_it(site, tmp_path):
    """UI ledger V-15 (the verification of fcc3dea, part RB2): for a
    read-only browser the Test page's ↻ beside the Run field showed the lock
    (the one gated-button rule, `chips.gated`), and its reason was the
    Checkpoint field's note, 123 px under it, off screen with it at
    390 x 844. The listing's refusal is now said once above the fields it
    turns off (the Run row with its ↻, then Checkpoint), as Settings says its
    panel's (T06-D): in a real browser (skipped without Node and
    Playwright), served with no deployment mode (the site's) at 1850 x 960,
    1440 x 1600 (stacked) and the display floor's 1280 x 720 (390 x 844, below it, is
    drawn as at it), the ↻ is locked and the nearest "Operator controls
    only: …" is that refusal, on the line just above the ↻'s row (within
    24 px of it: only the Run field's label between), both wholly on screen,
    and the refusal is said once in the Run panel. Fails on d8d1828 (122 px
    under the ↻, the Checkpoint field between; at 390 x 844 not wholly on
    screen)."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    shots = tmp_path / "shots"
    shots.mkdir()
    script = _RELOAD_REASON_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(script, tmp_path, base[:-len("gradio_api")], str(shots))
    (tmp_path / "seen.json").write_text(json.dumps(seen, ensure_ascii=False), encoding="utf-8")
    for size, got in seen.items():
        reason = got["reason"]
        assert got["button"] == f"{tp.LOCK} ↻" and got["disabled"], (size, got)
        assert got["whole"] and reason and reason["whole"], (size, got)
        # The line just above the ↻'s row: only the Run field's label between.
        assert 0 <= reason["toRow"] <= 24, (size, got)
        assert "listing the checkpoints opens them" in reason["text"], (size, got)
        assert got["listing"] == 1, (size, got)


def test_the_answer_asked_for_a_chosen_report_stays_with_it(site, monkeypatch):
    """A row whose place needs another workspace's listing is asked when it is
    chosen; the answer is kept in the session's reports, so choosing another
    row does not turn it back into "cannot be checked"."""
    _point_report_at_site(site)
    shutil.copytree(site.demo, site.ws / "copy", symlinks=True)
    page, request = _page(monkeypatch)
    models_copy = tp.read_models("copy", request)
    monkeypatch.setattr(tp, "read_job", lambda **k: {"status": "idle", "inputs": None,
                                                    "outcome": None, "log": []})
    demo, _ctx, _registry = _page_component()
    reports = tp.read_reports(LABEL)
    out = _named(demo, "_on_report").fn(page, reports, models_copy, None, "demo", LABEL, None,
                                        SimpleNamespace(_data={"key": site.run_id}), request)
    kept = tp.find(out[1]["runs"], "run_id", site.run_id)["state"]
    assert kept["registered"] == "not_registered"
    assert _report_rows(tp.reports_html(out[1], None, models=models_copy))[site.run_id][3] == (
        "not registered")


def test_another_workspaces_checkpoints_are_listed_only_for_the_asking_request(site, monkeypatch):
    """Asking a chosen report's registration may list another workspace's
    checkpoints (it opens them, C3): D8 is asked on the triggering request,
    as the model list asks it; a refused request, or none, lists nothing and
    the list's answer stands. Each pass lists that workspace once."""
    from src.api.services import server_lists

    _point_report_at_site(site)
    shutil.copytree(site.demo, site.ws / "copy", symlinks=True)
    page, request = _page(monkeypatch)
    models_copy = tp.read_models("copy", request)
    real, listed = server_lists.list_checkpoints, []
    monkeypatch.setattr(server_lists, "list_checkpoints",
                        lambda name, **k: listed.append(name) or real(name, **k))
    monkeypatch.setattr(tp, "read_job", lambda **k: {"status": "idle", "inputs": None,
                                                    "outcome": None, "log": []})
    demo, _ctx, _registry = _page_component()
    reports = tp.read_reports(LABEL)
    for asking in (_readonly(), None):
        out = _named(demo, "_on_report").fn(page, reports, models_copy, None, "demo", LABEL,
                                            None, SimpleNamespace(_data={"key": site.run_id}),
                                            asking)
        assert listed == []
        assert tp.find(out[1]["runs"], "run_id", site.run_id)["state"]["registered"] == \
            "cannot_be_checked"
    _named(demo, "_on_report").fn(page, reports, models_copy, None, "demo", LABEL, None,
                                  SimpleNamespace(_data={"key": site.run_id}), request)
    assert listed == ["demo"]
    # No listed file holds the weights: the list's answer stands, and the
    # choice still lists the other workspace once, not once per output built.
    listed.clear()
    monkeypatch.setattr(server_lists, "list_checkpoints",
                        lambda name, **k: listed.append(name) or server_lists.Listing((), None))
    out = _named(demo, "_on_report").fn(page, reports, dict(models_copy, items=[]), None, "demo",
                                        LABEL, None, SimpleNamespace(_data={"key": site.run_id}),
                                        request)
    assert listed == ["demo"], "listed once for the choice, never again for the same pass"
    assert tp.find(out[1]["runs"], "run_id", site.run_id)["state"]["registered"] == \
        "cannot_be_checked"


def test_the_ledger_order_is_the_file_then_the_workspace_the_run_read(site, monkeypatch):
    """`ledger_checkpoint`: (1) the file the run read when it is listed, even
    when another listed copy comes first in its workspace; (2) else a copy in
    the workspace the run measured, before (3) the workspace the page shows."""
    _point_report_at_site(site)
    manifest = json.loads((site.reports / site.run_id / "report.json").read_text())["manifest"]
    newer = site.demo / "runs" / "hgt" / "20261019-080000_run_aaaaaa" / "checkpoints"
    newer.mkdir(parents=True)
    # Written now, so its run is the most recently trained (`list_runs` orders
    # runs by their newest file): it lists before the fixture's file without a
    # model spec too.
    shutil.copy(site.checkpoint, newer / "best.pt")
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    assert models["items"][0]["run"] == "20261019-080000_run_aaaaaa", "the newer copy lists first"
    place = tp.ledger_checkpoint(manifest, page, models, "demo")
    assert place.path.resolve() == site.checkpoint.resolve() and place.workspaces == ("demo",)
    # The file the run read is gone from the list: the measured workspace's
    # copy, before the copy in the workspace the page shows.
    shutil.rmtree(newer.parent)
    site.checkpoint.rename(site.checkpoint.with_name("renamed.pt"))
    shutil.copytree(site.demo, site.ws / "copy", symlinks=True)
    page = tp.read_page(request)
    models_copy = tp.read_models("copy", request)
    place = tp.ledger_checkpoint(manifest, page, models_copy, "copy")
    assert place.workspaces == ("demo", "copy") and place.item["workspace"] == "demo"
    assert place.path.resolve() == site.checkpoint.with_name("renamed.pt").resolve()
    assert tp.ledger_checkpoint(manifest, page, models_copy, "copy", read_others=False).unread
    # A run measured on no listed workspace: the page's own workspace.
    elsewhere = dict(manifest, data_dir="/nowhere/ws", checkpoint_path="/nowhere/best.pt")
    place = tp.ledger_checkpoint(elsewhere, page, models_copy, "copy")
    assert place.workspaces == ("copy",) and place.item["workspace"] == "copy"


@pytest.mark.parametrize("text, kept", [
    ("Failed reading /mnt/clinic share/MRN 0012345/cases.json", "Failed reading "),
    ("[Errno 2] No such file or directory: '/mnt/clinic share/MRN 0012345.json'",
     "No such file or directory: '"),
    ("/data/MRN 0012345.json could not be read", " could not be read"),
    ('a "C:\\Users\\Jane Doe\\MRN 0012345\\x.json" link', 'a "'),
])
def test_a_path_with_spaces_outside_every_root_is_replaced_whole(text, kept):
    """M5-3 review SECURITY-4: the path ended at its first space, so the rest
    (``share/MRN 0012345/cases.json``) reached a read-only browser through the
    job tail."""
    out = tp.redact(text, [("/srv/registry", "<registry>")])
    assert "0012345" not in out and "MRN" not in out and "Jane" not in out, out
    assert tp.SERVER_PATH in out and kept in out, out


@pytest.mark.parametrize("name", ["demo (copy)", "ws🧬", "ws+", "a\xa0b", "工作 区"])
def test_a_path_under_a_root_keeps_every_name_whole(name):
    """The review of B0c: the one pattern started a new absolute path at any
    slash after a character that is no word's, so the rest of a path under a
    root ("<workspaces>/demo (copy)/runs/…", an emoji or a "+" ending a name)
    was cut and called a server path. The rest of a path under a root is read with it
    (`path_redaction._ROOTED`), so each name stays whole, here and through the
    dashboard's redaction, which joins plain whitespace only."""
    from pathlib import Path

    from src.webui.components import workspaces

    text = f"/srv/ws/{name}/runs/gatv2/r/checkpoints/last.pt does not exist."
    assert tp.redact(text, [("/srv/ws", "<workspaces>")]) == (
        f"<workspaces>/{name}/runs/gatv2/r/checkpoints/last.pt does not exist.")
    assert workspaces.redact(text, Path("/srv/ws")) == (
        f"{name}/runs/gatv2/r/checkpoints/last.pt does not exist.")


@pytest.mark.parametrize("text", [
    "cmd >/srv/clinic/MRN0012345/x.json failed", "🔒/srv/clinic/MRN0012345/x.json",
    "a→/srv/clinic/MRN0012345/x.json", "(see above)/srv/clinic/MRN0012345/x.json",
    "/srv/ws/demo (copy)/runs/a and /srv/clinic/MRN0012345/x.json",
])
def test_a_path_after_any_mark_that_is_no_words_is_still_replaced(text):
    """The review of B0c's fixes: names under a root are kept whole by reading
    the rest of a root's path with it (`path_redaction._ROOTED`), not by
    narrowing where a path may start, so a path right after a symbol, an
    emoji, an arrow or a closing bracket is replaced as before, under one root
    or the dashboard's."""
    from pathlib import Path

    from src.webui.components import workspaces

    for out in (tp.redact(text, [("/srv/ws", "<workspaces>")]),
                workspaces.redact(text, Path("/srv/ws"))):
        assert "/srv/clinic" not in out and "MRN0012345" not in out, out


@pytest.mark.parametrize("text, said", [
    ("/mnt/old/srv/ws/Jane Doe (MRN 1)/x.json does not exist.", "{} does not exist."),
    ("No such file: '/mnt/old/srv/ws/Jane Doe/x.json'", "No such file: '{}'"),
    ("Cannot read /mnt/Jane Doe/srv/ws/MRN 1/x.json: denied", "Cannot read {}: denied"),
    ("/mnt/Jane (MRN 1)/srv/ws/x.pt in 3.2 s", "{} in 3.2 s"),
])
def test_a_roots_text_inside_a_longer_path_is_that_paths_own(text, said):
    """UI-13 (a residual of B53): a root found anywhere in a text was taken as
    its start, inside another path too, so "/mnt/old/srv/ws/Jane Doe (MRN
    1)/x.json" read "a server path<workspaces>/Jane Doe (MRN 1)/x.json": a path
    outside the root shown under its name, its private part on. A path that
    runs into a root's text is one path, replaced whole, through the Test
    page's redaction and the dashboard's; a root that starts a path is named
    as before (the tests above)."""
    from pathlib import Path

    from src.webui.components import workspaces

    assert tp.redact(text, [("/srv/ws", "<workspaces>")]) == said.format(tp.SERVER_PATH)
    out = workspaces.redact(text, Path("/srv/ws"))
    assert out == said.format(workspaces.OUTSIDE_PATH), out


@pytest.mark.parametrize("text, kept", [
    ("/data/MRN(0012345)/cases.json does not exist.", " does not exist."),
    ("/srv/x/(0012345)/z.pt and (/srv/q)", " and ("),
    ("the cohort's set (/srv/MRN 0012345/y.json) is not the one", "the cohort's set ("),
])
def test_a_bracketed_name_in_a_path_outside_every_root_is_replaced_whole(text, kept):
    """A bracketed part with no space inside a name is the path's own; a
    bracket around a path is not (the closing one stays)."""
    out = tp.redact(text, [("/srv/registry", "<registry>")])
    assert "0012345" not in out and "MRN" not in out and "/srv" not in out, out
    assert tp.SERVER_PATH in out and kept in out, out


@pytest.mark.parametrize("text, said", [
    ("Cannot read /data/Jane (MRN 1)/x.json: denied", "Cannot read {}: denied"),
    ("Cannot read /data/other/x.json: denied", "Cannot read {}: denied"),
    ("see /data/Jöhn Doe (MRN 1)/x.json.", "see {}."),
    ("/mnt/clinic share/MRN 0012345.json: denied", "{}: denied"),
    ("Is /data/Jane (MRN 1).json readable? no", "Is {} readable? no"),
    ("/data/患者 (病历 1)/x.json! retry", "{}! retry"),
    ("Cannot read /data/x.json (permission denied)", "Cannot read {} (permission denied)"),
])
def test_a_bracketed_spaced_name_goes_whole_and_the_sentence_keeps_its_punctuation(text, said):
    """UI-13 (B72-N5, NEW-shell-nav-theme-3): a path outside every root whose
    name holds a bracketed part with a space ended at that space, so
    "(MRN 1)/x.json" was shown, and a path took the sentence's ': ' after it
    ("…root denied"). Every part of the name goes, the punctuation after it
    stays, through the Test page's redaction and the dashboard's; a bracket
    that only follows a path after a space is prose and stays."""
    from pathlib import Path

    from src.webui.components import workspaces

    assert tp.redact(text, [("/srv/registry", "<registry>")]) == said.format(tp.SERVER_PATH)
    out = workspaces.redact(text, Path("/srv/workspaces"))
    assert out == said.format(workspaces.OUTSIDE_PATH)
    assert "MRN" not in out and "Jane" not in out and "x.json" not in out and "患者" not in out


def test_the_words_after_a_path_stay_and_two_paths_are_two():
    roots = [("/srv/registry", "<registry>")]
    assert tp.redact("/x/model.pt in 3.2 s", roots) == f"{tp.SERVER_PATH} in 3.2 s"
    assert tp.redact("/etc/passwd and C:\\Users\\x end", roots) == (
        f"{tp.SERVER_PATH} and {tp.SERVER_PATH} end")
    assert tp.redact("see /srv/registry/versions/a b/r.json now", roots) == (
        "see <registry>/versions/a b/r.json now")


def test_the_test_page_and_datasets_share_one_redaction():
    """M5-3 review ARCH-7: one pattern and one implementation for "no absolute
    server path is shown" (they had already drifted apart); since research
    comparison Stage 8 in `src.utils.path_redaction`, which the results service
    uses too."""
    from src.utils import path_redaction
    from src.webui.components import workspaces

    assert not hasattr(tp, "_ABSOLUTE") and not hasattr(workspaces, "_ABSOLUTE")
    text = "x /mnt/a b/c d"
    assert tp.redact(text, []) == path_redaction.redact_paths(text, [], other=tp.SERVER_PATH)
    assert workspaces.redact(text, None) == path_redaction.redact_paths(
        text, [], other=workspaces.OUTSIDE_PATH)
    # Models' notes problems and the Test page read one pair, in the shared
    # module (a page reaches nothing of another page).
    assert (tp.redact, tp.path_roots) == (workspaces.redact_server, workspaces.path_roots)
    assert tp.SERVER_PATH == path_redaction.SERVER_PATH


def test_every_surface_says_a_server_path_in_the_one_words():
    """Verification of fcc3dea (V-AUD-4): the investigation notes on Test,
    Models and the API's record contest read a hidden path as "<path>",
    where every other surface says `path_redaction.SERVER_PATH`; and four
    modules each defined those words again. One definition now, read by
    every surface. Fails on fcc3dea."""
    import re
    from pathlib import Path

    from src.api.services import record_states
    from src.utils import deployment_view, path_redaction
    from src.webui.components import diagnosis_card

    said = path_redaction.SERVER_PATH
    assert record_states._open_note("seed drift; see /srv/ws/demo/notes.txt") == (
        f"seed drift; see {said}")
    assert diagnosis_card.SERVER_PATH == deployment_view.HIDDEN_PATH == said
    src = Path(path_redaction.__file__).resolve().parents[1]
    defined = re.compile(r"(?:=|other=)\s*[\"']" + re.escape(said) + r"[\"']")
    again = sorted(str(path.relative_to(src)) for path in src.rglob("*.py")
                   if defined.search(path.read_text(encoding="utf-8")))
    assert again == ["utils/path_redaction.py"], again



def test_the_notes_who_field_holds_a_normal_name():
    """M5-3: "Who" was 166 px wide at 1920 x 1080, so "research screenshots v2"
    showed without "v2". It takes two fifths of the note row, never under
    NOTE_FIELD_MIN_WIDTH px (the row wraps first), beside "What was found"."""
    demo, _ctx, _registry = _page_component()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    who, what = by_id["shep-test-note-who"], by_id["shep-test-note-what"]

    def row_of(block):
        while block is not None and "shep-test-noterow" not in (block.elem_classes or []):
            block = block.parent
        return block

    assert row_of(who) is not None and row_of(who) is row_of(what)
    assert (who.scale, what.scale) == (2, 3)
    assert who.min_width == what.min_width == tp.NOTE_FIELD_MIN_WIDTH
    assert tp.NOTE_FIELD_MIN_WIDTH >= 240


def test_a_refusal_worded_outside_the_known_forms_still_names_no_case_id():
    """M5-3 review ARCH-10 (b): `case_free` knows the importer's present
    wordings; a future one ("cases[2]: duplicate of case_id 'PAT-9'", which it
    leaves as it is) is replaced by the problem's case index, so no case id
    reaches the browser (D7)."""
    from src.evaluation.cohort_registry import MalformedSource

    problem = "cases[2]: duplicate of case_id 'PAT-9'"
    assert tp.case_free(problem) == problem                # not a known wording
    said = tp.import_refusal(MalformedSource("intake.json", [problem, "cases[0]: no terms"]))
    assert "PAT-9" not in said and "case_id 'PAT" not in said
    assert "cases[2]: a problem that names its case id" in said and "cases[0]: no terms" in said
    unplaced = tp.import_refusal(MalformedSource("intake.json", ["duplicate case_id \"Q-1\""]))
    assert "Q-1" not in unplaced and "a case: a problem that names its case id" in unplaced


def test_the_preview_shows_the_gene_counts_when_the_source_lists_genes():
    """README §8: the operator sees, before commit, how many gene identifiers
    map; a source without genes shows no gene row."""
    from src.webui.components.test_page import preview_html

    genes = {"identifiers_listed": 5, "identifiers_unmappable": 2, "identifiers_repeated": 1,
             "cases_with_causal_genes": 3, "cases_with_unmapped_causal_genes": 1,
             "cases_with_candidate_genes": 2}
    html = preview_html({"counts": {"source_cases": 3, "genes": genes}, "samples_n": 3})
    assert "Gene identifiers" in html and "2 name no gene of the graph" in html
    assert "1 with some unmapped" in html
    assert "Gene identifiers" not in preview_html({"counts": {"source_cases": 3},
                                                   "samples_n": 3})


# --------------------------------------------------------------------------- K6: by task
HOSTILE = "<img src=x onerror=alert(1)>"
#: A version's gene counts (`cohort_registry`, written only for a source that
#: lists genes), by what the gate reads.
NO_GENES = {"source_cases": 3, "ranked_eligible": 3}
GENES = {**NO_GENES, "genes": {"cases_with_causal_genes": 3, "cases_with_candidate_genes": 3}}
NO_CANDIDATES = {**NO_GENES, "genes": {"cases_with_causal_genes": 3,
                                       "cases_with_candidate_genes": 0}}
NO_CAUSAL = {**NO_GENES, "genes": {"cases_with_causal_genes": 0,
                                   "cases_with_candidate_genes": 3}}


def _model(task="disease", stage="finetune", record="model", **extra):
    """A listed model as the server's list gives it: its record's stage and
    task, and the capability description the protocols judge (none for a
    record not read as a model)."""
    return {"record": record, "learning_stage": stage, "task": task,
            "capabilities": described(stage, task) if record == "model" else None, **extra}


#: A listed supplied cohort version's kind, as the server's list gives it.
SUPPLIED = "supplied"


def test_measure_refusal_says_why_each_kind_is_not_measured():
    """A record's state is said in the word the Model option leads with
    (`renderers.record_kind_text`, K6 decision 6; the closing review found the
    page's own copies), then the page's own reason. Past it every reason is
    the protocol definition's (`protocols`, K2): with no measurement chosen,
    why none takes the record or "choose a measurement"; with one chosen, its
    refusal. The page words none itself."""
    from src.evaluation import protocols
    from src.webui.components import renderers as r

    refusal = tp.measure_refusal
    assert refusal(_model("disease"), {"kind": SUPPLIED, "counts": NO_GENES}, "disease") is None
    # No measurement chosen while the record admits one is said, never filled in.
    assert refusal(_model("disease"), {"kind": SUPPLIED, "counts": NO_GENES}, None) == (
        tp.CHOOSE_A_MEASUREMENT)
    # A base: the KG link-prediction protocol takes it (research comparison
    # Stage 5), on a KG edge set only; a task protocol says its own reason.
    # With none chosen on a supplied cohort, none is offered (Stage 7: by the
    # version's kind), and the reason is the definition's.
    base, supplied = _model(None, "base"), {"kind": SUPPLIED, "counts": NO_GENES}
    assert refusal(base, supplied, None) == protocols.available(
        described("base", None), SUPPLIED)[1] == (
        "no measurement that takes this checkpoint's model measures a supplied cohort version: "
        "it can be measured by KG link prediction, which measures a KG edge set. Ref: "
        "kg_link_prediction.")
    assert refusal(base, {"kind": "kg_edges", "counts": {"pairs": 4}}, None) == (
        tp.CHOOSE_A_MEASUREMENT)
    assert refusal(base, None, None) == tp.CHOOSE_A_MEASUREMENT
    # B72-T1: the measurement by its plain words, its code only in the Ref.
    assert refusal(base, supplied, "kg_link_prediction") == (
        "the KG link prediction measurement does not measure a supplied cohort version; it "
        "measures a KG edge set. Ref: kg_link_prediction.")
    assert refusal(base, {"kind": "kg_edges", "counts": {"pairs": 4}}, "kg_link_prediction") is None
    for protocol in ("disease", "gene"):
        assert refusal(base, supplied, protocol) == protocols.model_refusal(
            protocol, described("base", None))
        assert refusal(base, supplied, protocol).endswith(
            "It can be measured by KG link prediction. Ref: kg_link_prediction.")
    specless, refused, unread = ({"record": "refused", "record_problem": WRITTEN_BEFORE_SPEC},
                                 {"record": "refused", "record_problem": "record version 9"},
                                 {"record": None})
    for item in (specless, refused, unread):
        for protocol in (None, "disease"):
            said = refusal(item, None, protocol)
            assert said.startswith(r.record_kind_text(item)) and said.endswith(
                tp.NOT_MEASURED_HERE), said
    assert refusal(specless, None, None) == (
        f"Record not readable ({WRITTEN_BEFORE_SPEC}): " + tp.NOT_MEASURED_HERE)
    assert refusal(refused, None, None) == (
        "Record not readable (record version 9): " + tp.NOT_MEASURED_HERE)
    assert refusal(unread, None, None) == "Record unknown: " + tp.NOT_MEASURED_HERE
    # A record no protocol takes, a protocol the record does not meet, and a
    # name that is no protocol: the definition's sentences, by its ordinary rules.
    from tests.fixtures.capabilities import LP_HEAD_UNSTATED

    assert refusal(_model(LP_HEAD_UNSTATED, "base"), None, None) == protocols.available(
        described("base", LP_HEAD_UNSTATED))[1]
    assert refusal(_model("C"), None, None) == tp.CHOOSE_A_MEASUREMENT
    assert refusal(_model("gene"), None, "disease") == (
        protocols.model_refusal("disease", described("finetune", "gene")))
    assert ("It can be measured by gene or KG link prediction. Ref: kg_link_prediction."
            in refusal(_model("gene"), None, "disease"))
    for name in ("C", "D", "Disease"):
        assert refusal(_model("disease"), None, name) == (
            protocols.model_refusal(name, described("finetune", "disease")))
        assert "is not a measurement this build implements" in refusal(_model("disease"), None,
                                                                        name)


def test_a_listed_base_is_offered_the_kg_link_prediction_measurement(site, monkeypatch):
    """§5.4, the page's half, since research comparison Stage 5: the
    Measurement radio offers a listed base the KG link-prediction protocol, the
    one that takes it; on a supplied cohort version Start's reason and the
    start's refusal are that protocol's sentence for the data kind, and no
    runner is asked."""
    from src.evaluation import protocols

    _path, base = _put_bundle(site, BUNDLE_BASE_RUN, None)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    binding = tp.workspace_binding(site.demo, tp.find(page["workspaces"], "name", "demo"))
    choices = tp.measurement_choices(tp.find_model(models, base))
    assert choices == [("KG link prediction", "kg_link_prediction")]
    assert tp.chosen_measurement(choices, None) == "kg_link_prediction"
    message = protocols.inputs_refusal("kg_link_prediction", SUPPLIED, None)
    assert message == ("the KG link prediction measurement does not measure a supplied cohort "
                       "version; it measures a KG edge set. Ref: kg_link_prediction.")
    allowed, why = tp.start_state(page, models, "demo", LABEL, base, binding, ("none", ""),
                                  protocol="kg_link_prediction")
    assert (allowed, why) == (False, f"Start disabled: {message}")
    manager = FakeManager()
    result = tp.start_test_run(page, models, "demo", LABEL, base, False, False, request,
                               protocol="kg_link_prediction", manager=manager)
    assert result == {"success": False,
                      "error": f"{message[:1].upper()}{message[1:]} Nothing was started."}
    assert manager.started == []


@pytest.mark.parametrize("protocol, counts, missing", [
    ("gene", NO_GENES, "causal"), ("gene", NO_CAUSAL, "causal"),
    ("gene", NO_CANDIDATES, "candidate"), ("gene", GENES, None),
    ("patients_like_me", NO_GENES, "causal"), ("patients_like_me", NO_CAUSAL, "causal"),
    # Candidate genes are the gene measurement's list; patients-like-me needs none.
    ("patients_like_me", NO_CANDIDATES, None),
    ("disease", NO_GENES, None), ("disease", NO_CAUSAL, None), ("disease", GENES, None),
])
def test_a_measurement_needs_a_version_that_lists_its_genes(protocol, counts, missing):
    """The version's inputs, for a model the protocol takes: the definition's
    sentence (`protocols.inputs_refusal`), naming the genes missing and the
    source field that lists them."""
    from src.evaluation import protocols

    model = _model(protocols.PROTOCOLS[protocol].head[1])
    got = tp.measure_refusal(model, {"kind": SUPPLIED, "counts": counts}, protocol)
    assert got == protocols.inputs_refusal(protocol, SUPPLIED, counts)
    if missing is None:
        assert got is None
    else:
        words = protocols._said(protocols.PROTOCOLS[protocol])
        assert got == (f"this cohort version lists no {missing} genes, which the {words} "
                       f"measurement needs. Import a source that lists {missing}_genes."
                       + (f" Ref: {protocol}." if words != protocol else ""))


def test_start_and_its_reason_refuse_what_is_not_measured_with_one_rule(site, monkeypatch):
    """The listing's real records: a base, a record this build refuses and the
    fixture's file without a model spec are listed; Start says why (`start_state`,
    "Start disabled: …") and the start refuses them before the runner is asked
    ("… Nothing was started."), from the one rule (`measure_refusal`). A gene
    child on the fixture version, which lists no genes, is refused for that
    by the gene measurement, in the definition's words."""
    from src.evaluation import protocols

    _put_bundle(site, BUNDLE_BASE_RUN, None)
    _put_bundle(site, "20261021-000000_refused_cccccc", "disease", record_version=999)
    _put_bundle(site, "20261022-000000_gene_dddddd", "gene")
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    binding = tp.workspace_binding(site.demo, tp.find(page["workspaces"], "name", "demo"))
    manager = FakeManager()
    gene = "gatv2/20261022-000000_gene_dddddd/last.pt"
    version = tp.find(page["versions"], "label", LABEL)
    counts = version["counts"]
    assert version["kind"] == SUPPLIED
    cases = {
        f"gatv2/{BUNDLE_BASE_RUN}/last.pt": (
            "kg_link_prediction", protocols.inputs_refusal("kg_link_prediction", SUPPLIED,
                                                           counts)),
        "gatv2/20261021-000000_refused_cccccc/last.pt":
            (None, "Record not readable (checkpoint record version 999"),
        site.history_key: (None, f"Record not readable ({WRITTEN_BEFORE_SPEC})"),
        gene: ("gene", protocols.inputs_refusal("gene", SUPPLIED, counts)),
    }
    assert cases[gene][1].startswith("this cohort version lists no causal genes")
    for key, (protocol, says) in cases.items():
        allowed, why = tp.start_state(page, models, "demo", LABEL, key, binding, ("none", ""),
                                      protocol=protocol)
        assert allowed is False and why.startswith(f"Start disabled: {says}") and why.endswith("."), why
        result = tp.start_test_run(page, models, "demo", LABEL, key, False, False, request,
                                   protocol=protocol, manager=manager)
        assert result["success"] is False, key
        assert result["error"].startswith(says[:1].upper() + says[1:]), result["error"]
        assert result["error"].endswith(". Nothing was started.")
    assert manager.started == []
    # The version listing causal and candidate genes lifts the gene child's refusal.
    page["versions"][0]["counts"] = {**page["versions"][0]["counts"], **GENES}
    assert tp.start_state(page, models, "demo", LABEL, gene, binding, ("none", ""),
                          protocol="gene") == (True, None)
    assert tp.start_test_run(page, models, "demo", LABEL, gene, False, False, request,
                             protocol="gene", manager=manager)["success"] is True
    assert [(i.protocol, i.task) for i in manager.started] == [("gene", "gene")]


def test_a_refusal_naming_a_record_problem_is_shown_escaped():
    page = {"access": {"allowed": True}, "roots": [],
            "workspaces": [{"name": "demo"}],
            "versions": [{"label": LABEL, "counts": NO_GENES, "manifest_sha256": "e" * 64}]}
    models = {"workspace": "demo", "items": [{
        "architecture": "gatv2", "run": "r", "filename": "last.pt", "sha256": "c" * 64,
        "record": "refused", "record_problem": HOSTILE}]}
    allowed, why = tp.start_state(page, models, "demo", LABEL, "gatv2/r/last.pt", None,
                                  ("none", ""))
    assert allowed is False and HOSTILE in why            # plain text, as the select's
    shown = tp._reason_html(why)
    assert HOSTILE not in shown and "&lt;img src=x onerror=alert(1)&gt;" in shown


def test_every_checkpoint_is_listed_led_by_what_it_is(site, monkeypatch):
    """K6 decision 13: the Run → Checkpoint pair lists every checkpoint (each
    run's, under its run), the values the listing's keys, the options short
    and unique; what each file is leads its details
    (`renderers.model_kind_text`: the frozen-backbone control by its freeze,
    A2), never its option."""
    from src.webui.components import renderers as r

    _put_bundle(site, BUNDLE_BASE_RUN, None)
    for task, run in (("disease", CHILD_RUN), ("gene", "20261022-000000_gene_dddddd"),
                      ("patients_like_me", "20261023-000000_plm_eeeeee")):
        _put_bundle(site, run, task)
    _put_bundle(site, "20261021-000000_refused_cccccc", "disease", record_version=999)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    keys = []
    runs, _checkpoints = ml.parents_update(tp.model_parents(models), None, first=False)
    run_labels = [label for label, _value in runs["choices"]]
    assert len(set(run_labels)) == len(run_labels)
    for _label, run in runs["choices"]:
        key = ml.chosen_in_run(tp.model_parents(models), run, None)
        _runs, checkpoints = ml.parents_update(tp.model_parents(models), key, first=False)
        assert checkpoints["value"] == key
        keys += [value for _label, value in checkpoints["choices"]]
        for label, _value in runs["choices"] + checkpoints["choices"]:
            assert len(label) <= r.OPTION_CHARS and label.count(" · ") <= 1, label
            assert not label.startswith(("Disease", "Base", "Record")), label
    assert sorted(keys) == sorted(tp.item_key(i) for i in models["items"])
    kinds = sorted(
        re.search(r'<dt class="shep-k">Model</dt><dd class="shep-v">([^<]*)</dd>',
                  tp.model_html(page, models, tp.item_key(item))).group(1)
        for item in models["items"] if r.is_digest(item.get("sha256")))
    # Two disease children: the fixture's trained one and the one put here.
    assert kinds == sorted(["Base", "Disease", "Disease", "Gene", "Patients-like-me",
                            "Record not readable", "Record not readable"])
    frozen = {**models["items"][0], "record": "model", "learning_stage": "finetune",
              "task": "disease", "weight_transfer": {
                  "trainable": ["patient_encoder", "task_head"],
                  "parts": {"input": "frozen", "encoder": "frozen", "lp_head": "frozen",
                            "patient_encoder": "trainable", "task_head": "trainable"}}}
    assert ('<dt class="shep-k">Model</dt><dd class="shep-v">Disease · frozen graph backbone</dd>'
            in tp.model_html(page, {"items": [frozen]}, tp.item_key(frozen)))


def _task_record(mode, **metrics):
    """A kept task report's record (`build_record`, as the store's scan builds
    it, so the one reader reads it, research comparison Stage 8b) with these
    metrics and a small cohort's counts."""
    from tests.unit.test_evaluation_sidecar import registered_report

    base = {"pooled_mrr": 0.3125, "pooled_top1": 0.25, "pooled_top3": 0.5, "pooled_top10": 0.75,
            "macro_recall@1": 0.2, "macro_recall@5": 0.4, "macro_recall@10": 0.6,
            "macro_precision@1": 0.25, "macro_mrr": 0.33, "best_positive_mrr": 0.4,
            "conservative_pooled_mrr": 0.21, "coverage": 0.8}
    record = _kept_record(registered_report(mode=mode))
    record["metrics"] = {**base, **metrics}
    record["n_ranked"] = 8
    record["cohort"]["label"] = "v"
    record["cohort"]["counts"] = {"source_cases": 13, "ranked_eligible": 10, "n_excluded": 3,
                                  "excluded": {"ground_truth_absent": 3}}
    return record


@pytest.mark.parametrize("mode", ["disease", "gene", "patients_like_me"])
def test_a_task_record_shows_its_own_tiles_coverage_and_figures(mode):
    """A task record (`task_measurement`): its three tiles and coverage, each
    named by `caveats.metric_text`; coverage "n of M import-eligible cases";
    the conservative pooled MRR with what it counts; no mean rank and no Mode
    C; its own help topic; a metric of another name shown escaped."""
    import html as html_lib

    from src.evaluation.caveats import TASK_NAMES, metric_text, mode_label, mode_note

    raw = tp._record_html(_task_record(mode, **{HOSTILE: 0.5}), {}, {})
    text = " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", raw)).split())
    tiles = renderers.RECORD_TILES[mode]
    assert len(tiles) == 4 and tiles[0] == "pooled_mrr" and tiles[-1] == "coverage"
    # R-71: a tile's name less the task's, which the label above says once.
    for name in tiles:
        words = metric_text(mode, name)[len(TASK_NAMES[mode]) + 1:]
        assert f'shep-rec-tile-l">{html_lib.escape(words[:1].upper() + words[1:])}<' in raw
        assert f'shep-rec-tile-l">{html_lib.escape(metric_text(mode, name))}<' not in raw
    assert mode_label(mode) in text
    assert "8 of 10 import-eligible cases ranked by the model 13 source cases, 3 excluded " \
        "at import: ground truth absent 3" in text
    # The counts are the Coverage row's alone (rule 1.2): a tile's footer says
    # what its figure is over, never "n of M" or the n ranked again.
    notes = re.findall(r'shep-rec-tile-n">([^<]*)<', raw)
    assert len(notes) == 4 and "pooled over positive pairs" in notes
    assert notes[-1] == "share of import-eligible cases ranked"
    assert not any(ch.isdigit() for note in notes for ch in note), notes
    assert text.count("8 of 10") == 1
    assert "Conservative pooled MRR" in text and "0.2100 an unranked case whose truth can be " \
        "assessed counts 0" in text and "0.2100 ·" not in text
    assert "Mean rank" not in text and "Mode C" not in text and "Mode D" not in text
    assert "Macro MRR" in text and "Best-positive MRR" in text and "Pooled top-k" in text
    # A tile's figure is not repeated in its family's row (rule 1.2).
    pooled = raw.split('>Pooled top-k</dt><dd class="shep-v">')[1].split("</dd>")[0]
    shown = {f"k={name[len('pooled_top'):]} " for name in tiles if name.startswith("pooled_top")}
    assert shown and not any(k in pooled for k in shown)
    assert all(f"k={k} " in pooled for k in (1, 3, 10) if f"k={k} " not in shown)
    assert f'data-help="test-mode-{mode}"' in raw
    # The governed note in its own words, a sentence a line (its lead first).
    assert tp.TOPIC_HELP[f"test-mode-{mode}"] == (f"{TASK_NAMES[mode]} task result",
                                                  sentence_lines(mode_note(mode)))
    assert tp.TOPIC_HELP[f"test-mode-{mode}"][1].replace("\n", " ") == mode_note(mode)
    assert HOSTILE not in raw and html_lib.escape(HOSTILE, quote=False) in raw


def test_a_results_determinism_reads_by_the_one_value_words_rule():
    """The review of B0c: the Result identity kept a page copy of the on/off
    rule. It is `renderers.setting_words`' (on, off); a record without the
    flag holds none, "not recorded", never "not set"."""
    from src.webui.components import renderers as r

    record = _task_record("disease")
    for flag, words in ((True, "on"), (False, "off")):
        record["regime"]["deterministic_algorithms"] = flag
        assert (f'Deterministic algorithms</dt><dd class="shep-v">{words}</dd>'
                in tp._record_html(record, {}, {}))
        assert words == r.setting_words("deterministic_algorithms", flag)


def _kept_record(report):
    """The record the ledger would hold for `report`, as the Test page's
    Result reads it (`build_record`, through the store's scan)."""
    from src.evaluation.sidecar import build_record

    return build_record(report, "f" * 64)


def _result_text(raw):
    import html as html_lib

    return " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", raw)).split())


def test_a_kept_mode_c_record_is_shown_as_recorded_by_the_one_display():
    """A kept Mode C record (R2, K8-6e.2): its generic label, no tiles and no
    "Mean rank" row; one "Recorded metrics" row listing every metric by its
    recorded name and its value unrounded; its score semantics exactly as
    recorded; the generic coverage (source cases, exclusions by reason, the
    cases its record counts, with no claim of what they are); no ⓘ beside its
    label, since no help topic describes it."""
    from src.evaluation.caveats import mode_label
    from tests.unit.test_evaluation_sidecar import mode_c_report

    record = _kept_record(mode_c_report())
    raw = tp._record_html(record, {}, {})
    text = _result_text(raw)
    assert mode_label("C") in text and "approved scorer policy" not in text
    assert "shep-rec-tile-l" not in raw and "Mean rank" not in text
    recorded = raw.split('<dt class="shep-k">Recorded metrics</dt><dd class="shep-v">')[1]
    recorded = recorded.split("</dd>")[0]
    assert re.findall(r'<div class="shep-rec-ln"><code>([^<]*)</code> ([^<]*)</div>',
                      recorded) == [
        (name, json.dumps(value)) for name, value in record["metrics"].items()]
    assert {"untruncated_mrr": "0.5", "hits@1": "0.25", "hits@10": "0.75",
            "mean_rank": "3.0"} == dict(re.findall(r"<code>([^<]*)</code> ([^<]*)<", recorded))
    assert (f"Score semantics {record['scoring']['score_semantics']}" in text
            and record["scoring"]["score_semantics"] == MODE_C_SEMANTICS_TEXT)
    assert ("100 source cases, 20 excluded: ground truth absent 12, phenotype unmappable 8 "
            "80 cases counted by the record. What they are is not described for this mode."
            ) in text
    label = raw[raw.index('class="shep-rec-label"'):]
    assert "data-help" not in label[:label.index("</div>")]


#: The score semantics a kept Mode C report records (`MODE_C_SEMANTICS`).
MODE_C_SEMANTICS_TEXT = "raw cosine, no eta mixture and no shortest-path term"


def test_a_record_of_a_mode_no_build_has_words_for_is_shown_as_recorded():
    """A "kg_lp_reader_only" record (a mode this build has no words for) and a
    record whose mode no build has ever named: the recorded metrics are the
    main result, with no tiles, no "Mean rank" row and no list of modes that
    decides what is shown (the three tasks are not a read whitelist)."""
    from src.evaluation.caveats import mode_label
    from tests.unit.test_evaluation_sidecar import registered_report

    for mode in ("kg_lp_reader_only", "zz_protocol_never_seen"):
        record = _kept_record(registered_report(mode=mode))
        raw = tp._record_html(record, {}, {})
        text = _result_text(raw)
        assert mode_label(mode) in text and f"Mode {mode} (" in text
        assert "shep-rec-tile-l" not in raw and "Mean rank" not in text
        assert "macro_precision@3 0.16666666666666666" in text
        assert len(re.findall(r'<div class="shep-rec-ln"><code>', raw)) == len(record["metrics"])
        assert "Other metrics" not in text and "Pooled top-k" not in text


def test_a_task_record_keeps_its_tiles_and_shows_every_metric_one_click_away():
    """A task record (R2): its tiles and rows unchanged; every metric as
    recorded behind a closed native disclosure, "Recorded metrics (N)", under
    its figures; its score semantics as recorded in its own row."""
    from src.webui.components.renderers import e
    from tests.unit.test_evaluation_sidecar import DISEASE_METRICS, registered_report

    record = _kept_record(registered_report())
    raw = tp._record_html(record, {}, {})
    assert len(re.findall(r'shep-rec-tile-l">', raw)) == 4
    assert "Conservative pooled MRR" in raw and "Pooled top-k" in raw
    # B52-E: the metrics as recorded and the engineering rows are the
    # record's one "Details ›", after its other rows.
    fold = re.search(r'<details class="shep-more"><summary>Details ›</summary>(.*?)</details>',
                     raw)
    count = re.search(r'<dt class="shep-k">Recorded metrics \((\d+)\)</dt>', fold.group(1))
    assert count and int(count.group(1)) == len(DISEASE_METRICS)
    assert ('<div class="shep-rec-ln"><code>macro_precision@3</code> 0.16666666666666666</div>'
            in fold.group(1))
    assert raw.index("shep-rec-rows") < raw.index("shep-more")
    assert ('<dt class="shep-k">Score semantics</dt><dd class="shep-v"><code>'
            + e(record["scoring"]["score_semantics"]) + "</code></dd>") in fold.group(1)
    for key in renderers.RECORD_DETAIL_ROWS:
        assert f'<dt class="shep-k">{key}</dt>' in fold.group(1), key
        assert f'<dt class="shep-k">{key}</dt>' not in _outside_details(raw), key


def test_only_the_described_measurements_have_help_topics():
    """No help topic exists only for a mode this build does not describe: the
    Mode C and Mode D topics retired (R2); each measurement this build
    describes has its topic (`caveats.MEASUREMENT_NAMES`), each task's as it
    was and, since the browser check of 7e1d09e, KG link prediction's."""
    from src.evaluation.caveats import MEASUREMENT_NAMES

    assert "test-mode-c" not in tp.TOPIC_HELP and "test-mode-d" not in tp.TOPIC_HELP
    assert sorted(k for k in tp.TOPIC_HELP if k.startswith("test-mode-")) == sorted(
        f"test-mode-{mode}" for mode in MEASUREMENT_NAMES)


def test_a_hostile_metric_name_and_score_semantics_are_literal_text_on_test():
    """A metric name and a score semantics holding markup are shown as literal
    text on the Result, for a described measurement (in its disclosure) and
    any other mode (its recorded metrics row)."""
    from tests.unit.test_evaluation_sidecar import registered_report

    for mode in ("disease", "zz_new"):
        record = _kept_record(registered_report({HOSTILE: 0.5, "pooled_mrr": 0.25}, mode=mode,
                                                score_semantics=HOSTILE))
        raw = tp._record_html(record, {}, {})
        assert HOSTILE not in raw, mode
        assert f"{HOSTILE} 0.5" in _result_text(raw)
        assert f"Score semantics {HOSTILE}" in _result_text(raw)


def test_a_task_report_row_names_its_task_first():
    record = _task_record("gene")
    run = {"run_id": "r", "record": record, "state": {"kept": "kept",
                                                      "registered": "not_registered"}}
    assert tp._row_mode(run) == "Gene task" and "not registered" in tp._row_chip(run)
    plm = dict(run, record=_task_record("patients_like_me"))
    assert tp._row_mode(plm) == "Patients-like-me task"


def _kind_record(kind, **report):
    """A kept report's record of a KG edge set or a generated split
    (`build_record`, as the store's scan builds it)."""
    from tests.unit.test_ledger_by_data_kind import generated_report, kg_edges_report

    return _kept_record({"kg_edges": kg_edges_report, "generated": generated_report}[kind](
        **report))


def test_a_kg_edge_set_or_generated_result_of_the_selected_version_says_no_count_of_datas():
    """R-70 (counts once when the result's version is the selected one) for
    the two kinds whose counts the Result folds: Data's version line says a
    KG edge set's or a generated split's counts (`_version_counts`, every
    count by name in its "Counts and units" fold), so a result of that very
    version leaves out its "Data version counts" fold (the shared record put
    it back: positive_edges 8, pairs 6, edges_without_negative 2 and
    message_passing_edges 50, or samples 5 and diseases 1, a second time on
    one screen) and a KG edge set's graph and pairs line their counts (its
    part, pair rule and seed stay); a KG edge set's run's own counts, which
    can differ, stay one click away; a result of another version keeps its
    version's counts. Fails on d6c1efa, whose Result of the selected version
    drew the fold and "… 50 edges; … 6 rows"."""
    from src.api.services.record_views import result_view

    for kind, said in (("kg_edges", ("positive_edges", "edges_without_negative")),
                       ("generated", ("samples", "diseases"))):
        record = _kind_record(kind)
        view = result_view(dict(record)).model_dump(mode="json")
        version = {"kind": kind, "counts": view["counts"], "count_units": view["count_units"]}
        data = _plain(tp._version_counts(version))
        selected = tp.Selection(None, None, None, view["cohort"]["version_manifest_sha256"])
        result = tp._record_html(record, {"run_id": "r"}, {}, selection=selected)
        assert tp.SELECTED_VERSION in result, kind
        assert "Data version counts" not in result, kind
        assert ("Run counts" in result) is (kind == "kg_edges"), kind
        for name in said:
            # Data names it; the Result only in a KG edge set's run's counts.
            assert name in data, (kind, name)
            assert result.count(f"<code>{name}</code>") == (kind == "kg_edges"), (kind, name)
        keys = re.findall(r'<dt class="shep-k[^"]*"[^>]*>([^<]*)', result)
        assert ("Counts" in keys) is (kind == "kg_edges"), (kind, keys)
        if kind == "kg_edges":
            run = result[result.index("Run counts"):]
            assert "pairs_scored" in run and "pairs_not_finite" in run
            # Its graph and pairs line keeps their part, rule and seed, not
            # the version's counts Data's lead gives ("… message passing
            # over 50 edges"; "6 pairs").
            cohort = view["cohort"]
            counted = (f"{cohort['message_passing']['edges']} edges",
                       f"{cohort['pairs']['rows']} rows")
            assert all(words.split()[0] in data for words in counted)
            assert not any(words in _plain(result) for words in counted), counted
            line = _plain(dict(renderers.data_kind_lines(view, version_counts=False))[
                "Graph and pairs"])
            assert line.startswith("Message passing over the split's") and "seed" in line
        other = tp._record_html(record, {"run_id": "r"}, {},
                                selection=tp.Selection(None, None, None, "f" * 64))
        assert tp.SELECTED_VERSION not in other and "Data version counts" in other, kind
        if kind == "kg_edges":
            assert all(words in _plain(other) for words in counted), counted


def test_a_kg_link_prediction_report_row_names_its_measurement_first():
    """The browser check of 7e1d09e: every kept KG link-prediction row read
    "mode kg_link_prediction, not described by this version", on a page that
    offers that measurement as "KG link prediction". The row's words come
    from the one measurement vocabulary (`caveats.MEASUREMENT_NAMES`), so a
    measurement this build describes is never named by its code as one it
    does not; a mode it does not describe still is."""
    record = _kind_record("kg_edges")
    assert record["scoring"]["mode"] == "kg_link_prediction"
    run = {"run_id": "r", "record": record, "state": {"kept": "kept", "registered": "registered",
                                                      "sequence": 2}}
    assert tp._row_mode(run) == "KG link prediction"
    assert "Registered #2" in tp._row_chip(run)
    unknown = dict(run, record={**record, "scoring": {**record["scoring"], "mode": "zz_new"}})
    assert tp._row_mode(unknown) == "Mode zz_new, not described by this version"
    hostile = dict(run, record={**record, "scoring": {**record["scoring"], "mode": HOSTILE}})
    assert HOSTILE not in tp._row_mode(hostile)


def test_a_kg_link_prediction_result_has_its_measurement_note():
    """The browser check of 7e1d09e: an LP result showed no note, where every
    task result has one; the note is the measurement's own
    (`caveats.MODE_RESULT_NOTES`, the one Models prints above its LP table),
    behind the ⓘ beside the result's label ("KG link prediction result").
    Its metrics stay as recorded, as Models shows them."""
    from src.evaluation.caveats import KG_LINK_PREDICTION_NOTE, mode_label

    raw = tp._record_html(_kind_record("kg_edges"), {}, {})
    label = raw[raw.index('class="shep-rec-label"'):]
    label = label[:label.index("</div>")]
    assert 'data-help="test-mode-kg_link_prediction"' in label
    assert tp.TOPIC_HELP["test-mode-kg_link_prediction"] == (
        "KG link prediction result", sentence_lines(KG_LINK_PREDICTION_NOTE))
    assert mode_label("kg_link_prediction") in _result_text(raw)
    assert "shep-rec-tile-l" not in raw


def test_a_kg_link_prediction_records_metrics_read_in_plain_names_on_test_and_models():
    """B72-C2: a KG link-prediction record showed 'lp_accuracy
    0.3333333333333333 / lp_roc_auc …', raw names, unrounded, no units, on
    Test and Models. Each is a row under its plain name
    (`caveats.LP_METRIC_WORDS`: "ROC-AUC"), its value to four decimals and
    its unit and denominator said once under it, as the record's
    ``measured`` holds them; every value as recorded stays one click away
    ("Recorded metrics (4)") and in every export. The same rows on both
    pages (one rendering, R-82)."""
    from src.evaluation.caveats import LP_METRIC_WORDS
    from src.webui.components import models_page as mp

    record = _kind_record("kg_edges")
    raw = tp._record_html(record, {}, {})
    import html as html_lib

    rows = {html_lib.unescape(key): value for key, value in re.findall(
        r'<dt class="shep-k">([^<]*)</dt><dd class="shep-v">(.*?)</dd>', raw)}
    assert '<dt class="shep-k">Recorded metrics</dt>' not in raw
    statuses = record["measured"]["metrics"]
    for name, value in record["metrics"].items():
        words = LP_METRIC_WORDS[name]
        cell = rows[words[:1].upper() + words[1:]]
        status = statuses[name]
        if status["status"] == "computed":
            assert cell.startswith(f"{value:.4f}<div"), (name, cell)
        else:
            assert "Not computable" in cell
        assert renderers.e(status["unit"]) in cell
        assert f"over {status['denominator']['count']} " in cell
        assert name not in _result_text(cell)
    fold = re.search(r'<dt class="shep-k">Recorded metrics \((\d+)\)</dt>'
                     r'<dd class="shep-v">(.*?)</dd>', raw)
    assert fold and int(fold.group(1)) == len(record["metrics"])
    for name, value in record["metrics"].items():
        assert f"<code>{name}</code> {json.dumps(value)}" in fold.group(2)
    # Models draws the same rows for the same record (the shared rendering).
    from src.evaluation.sidecar import append_record, empty_ledger

    registered = append_record(empty_ledger(), record, registered_at="2026-10-27T12:00:00Z",
                               run_id=f"20261027T120000Z-{record['checkpoint_digest'][:12]}-0000"
                                      "0001")[1]["records"][0]
    models = mp.record_block_html(mp.record_view(registered), None)
    words = {renderers.e(w[:1].upper() + w[1:]) for w in LP_METRIC_WORDS.values()}
    lp_rows = [key for key in re.findall(r'<dt class="shep-k">([^<]*)</dt>', raw)
               if key in words]
    assert len(lp_rows) == len(record["metrics"]) and lp_rows == [
        key for key in re.findall(r'<dt class="shep-k">([^<]*)</dt>', models)
        if key in words]


@pytest.mark.parametrize("kind, mode", [("kg_edges", "kg_link_prediction"),
                                        ("generated", "disease"),
                                        ("generated", "patients_like_me")])
def test_a_kg_edge_set_or_generated_split_result_never_says_cohort(kind, mode):
    """The browser check of 7e1d09e: a KG edge set's and a generated split's
    results read "cohort version mapped against this graph" (Relations) and
    a patients-like-me result "every other patient of the cohort". The
    caveats' words every kind shares fit each kind (research comparison §5
    Stage 4: never "cohort" for these kinds), so the Test page and Models say
    them alike: "data version bound to this graph", "every other patient
    measured"; a supplied cohort's result still names its cohort (R10)."""
    from src.evaluation.caveats import RELATION_DISPLAY_NAMES, mode_label
    from src.webui.components import models_page as mp
    from tests.unit.test_evaluation_sidecar import registered_report

    record = _kind_record(kind, **({} if kind == "kg_edges" else {"mode": mode}))
    assert (record["scoring"]["mode"], record["cohort"]["kind"]) == (mode, kind)
    raw = tp._record_html(record, {}, {})
    text = _result_text(raw)
    assert "cohort" not in text.lower(), text
    assert mode_label(mode) in text
    # Each relation a row of its own, its name visible whole (the one
    # rendering both pages show, `renderers.relation_rows`).
    bound = (f'<dt class="shep-k">Data version bound to this graph</dt><dd class="shep-v">'
             f'{tp.r.status_mark("ok", "Verified")}</dd>')
    assert bound in raw and RELATION_DISPLAY_NAMES["cohort_graph"] == (
        "data version bound to this graph")
    assert not hasattr(mp, "relations_html")     # Models' own chain of short names retired
    rows = dict(tp.r.relation_rows(record["relations"]))
    assert "cohort" not in " ".join(rows).lower() and "Data version bound to this graph" in rows
    supplied = tp._record_html(_kept_record(registered_report()), {}, {})
    assert renderers.R10_TITLE in _result_text(supplied) and bound in supplied


def test_the_measurement_options_are_the_protocols_the_record_admits():
    """§5.3: the Measurement radio's options are `protocols.available` of the
    chosen model's listed record, each labelled with the measurement's words
    (`caveats.MEASUREMENT_NAMES`: a task's by its task's name, "KG link
    prediction"; never a protocol's code as the label, skill §5) and valued by
    the protocol's name, short and unique (skill §8). The only option is
    chosen; of several, none is until the operator picks one, and a pick no
    longer offered goes. What the server fixes names no measurement."""
    lp = ("KG link prediction", "kg_link_prediction")
    for task, words in (("disease", "Disease"), ("gene", "Gene"),
                        ("patients_like_me", "Patients-like-me")):
        # A finetune is taken by its task's protocol and by the KG
        # link-prediction protocol (research comparison Stage 5).
        choices = tp.measurement_choices(_model(task))
        assert choices == [(words, task), lp]
        assert tp.chosen_measurement(choices, None) is None
        assert tp.chosen_measurement(choices, task) == task
        assert tp.chosen_measurement(choices, "another") is None
    for item in (_model(None, "base"), _model("C")):
        choices = tp.measurement_choices(item)
        assert choices == [lp]
        assert tp.chosen_measurement(choices, None) == "kg_link_prediction"
        assert tp.chosen_measurement(choices, "another") == "kg_link_prediction"
    for item in (None, {"record": "refused", "record_problem": WRITTEN_BEFORE_SPEC},
                 {"record": None}, {"record": "refused", "record_problem": "x"}):
        assert tp.measurement_choices(item) == []
    assert tp.chosen_measurement([], "disease") is None
    several = [("p1", "p1"), ("p2", "p2")]
    assert tp.chosen_measurement(several, None) is None
    assert tp.chosen_measurement(several, "p2") == "p2"
    assert tp.chosen_measurement(several, "disease") is None
    labels = [label for label, _value in tp.measurement_choices(_model("patients_like_me"))]
    assert all(len(label) <= 40 and label.count(" · ") <= 1 for label in labels)
    fixed = tp.fixed_html({})
    assert "Batch size" in fixed and "Device" in fixed and "Measurement" not in fixed


def test_the_measurement_radio_sits_in_the_run_panel_for_an_operator():
    """A radio (skill §8: under five options) labelled "Measurement", in the
    Run panel, beside the Model select; written with the selection, so Start's
    reason is drawn from the value it shows. The value it shows is written
    with it into a State, which the selection's listeners read in place of
    the radio (a value an event was asked with may no longer be among the
    choices, and Gradio refuses it before the handler runs); only the
    radio's own pick and Start take the radio."""
    demo, _ctx, _registry = _page_component()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    radio = by_id["shep-test-measurement"]
    assert isinstance(radio, gr.Radio) and radio.label == "Measurement"
    parent, panels = radio.parent, []
    while parent is not None:
        panels.append(getattr(parent, "elem_id", None))
        parent = parent.parent
    assert "shep-test-run" in panels
    pick = _named(demo, "_on_measurement")
    mirror = pick.outputs[-1]
    assert isinstance(mirror, gr.State) and radio in pick.outputs and radio in pick.inputs
    for name in ("_refresh", "_on_workspace", "_on_version", "_on_model", "_on_model_reload",
                 "_after_commit"):
        fn = _named(demo, name)
        assert {radio, mirror} <= set(fn.outputs) and mirror in fn.inputs, name
    start = by_id["shep-test-start"]
    for fn in demo.fns.values():
        if start in fn.outputs and fn is not pick:
            assert mirror in fn.inputs and radio not in fn.inputs, fn.name
    assert radio in _named(demo, "_on_start").inputs


def test_the_job_card_names_the_runs_protocol_escaped():
    """The card names the measurement the run was asked for (`inputs["protocol"]`),
    not the model's task: the two can differ. In the words its report's row
    will carry (`measurement_words`: review of B3, the job row said
    "Disease" beside its report's "Disease task"), a row of the card's one
    summary list, never a chain of " · " (the skill's §5)."""
    import html as html_lib

    job = {"status": "running", "inputs": {"version_label": LABEL, "protocol": "gene",
                                           "learning_stage": "finetune", "task": "gene",
                                           "checkpoint_sha256": "c" * 64, "workspace": "/w/ws"},
           "log": []}
    body = tp.job_body_html(job, [], ("busy", "GPU job: test run"))
    rows = _identity_rows(body)
    assert rows["Measurement"] == "Gene task" and " · " not in "".join(rows.values())
    assert rows["Model"] == "c" * 12 and f'title="{"c" * 64}"' in body
    # What it measures in view; its settings one Details › away, so the card
    # at the Run panel's foot leaves the choices above it their room.
    assert list(rows) == ["Version", "Model", "Measurement", "Workspace", "Register",
                          "Deterministic algorithms", "Device", "R10 evidence"]
    shown = re.sub(r'<details class="shep-more">.*?</details>', "", body, flags=re.S)
    assert list(_identity_rows(shown)) == ["Version", "Model", "Measurement"]
    other = dict(job, inputs={**job["inputs"], "protocol": "p2", "task": "disease"})
    body = tp.job_body_html(other, [], ("busy", ""))
    assert _identity_rows(body)["Measurement"] == "Mode p2, not described by this version"
    assert "Disease" not in body
    hostile = dict(job, inputs={**job["inputs"], "protocol": HOSTILE})
    body = tp.job_body_html(hostile, [], ("busy", ""))
    assert HOSTILE not in body and f"Mode {html_lib.escape(HOSTILE)}, not described" in body
    for missing in (None, "", 7):
        body = tp.job_body_html(dict(job, inputs={**job["inputs"], "protocol": missing}), [],
                                ("busy", ""))
        assert _identity_rows(body)["Measurement"] == "○ Not recorded"
    # The running row and the kept report's row of one protocol: one cell.
    run = {"run_id": f"20261113T213218Z-{'c' * 12}-743716b1", "when": "2026-11-13 21:32 UTC",
           "record": {"checkpoint_digest": "c" * 64, "scoring": {"mode": "disease"}},
           "state": {"kept": "kept", "registered": "registered", "sequence": 2}}
    running = dict(job, inputs={**job["inputs"], "protocol": "disease"})
    table = tp.reports_html({"label": LABEL, "runs": [run]}, None, job=running)
    cells = [" ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", c)).split())
             for c in re.findall(r"<td[^>]*>(.*?)</td>", table)]
    assert cells[2] == cells[6] == "Disease task", cells


def test_the_help_says_what_the_measurement_options_are():
    """Skill §12: the titles equal the labels (pinned by the registration
    test), the leads answer the question within the budget, and the
    measurement is said where the choice is made: never Mode C, never "the
    model's own task"."""
    # R-79: the batch size and the device each have their own ⓘ (one beside
    # Batch size explained both, and Device had none).
    assert "test-fixed" not in tp.TOPIC_HELP
    for topic, title, says in (("test-batch", "Batch size", "fixed by the job runner"),
                               ("test-device", "Device", "deployment's device")):
        got, text = tp.TOPIC_HELP[topic]
        lead = text.split("\n")[0]
        assert got == title and len(lead) <= 100 and says in lead and "Mode C" not in text
        assert len(text.split()) <= 80
    fixed = tp.fixed_html({"device": {"value": "cpu"}})
    assert fixed.count('class="shep-i') == 2
    assert 'data-help="test-batch"' in fixed and 'data-help="test-device"' in fixed
    title, text = tp.FIELD_HELP["shep-test-measurement"]
    lead = text.split("\n")[0]
    assert title == "Measurement" and len(lead) <= 100 and len(text.split()) <= 80
    assert "this build implements" in lead
    # A base is offered the KG link-prediction measurement (research comparison
    # Stage 5), never told its test is not implemented.
    assert "A base: offered KG link prediction only" in text and "Mode C" not in text
    assert "not implemented" not in text
    # Offered by what the model offers (Stage 3), not by a stage list.
    assert "capabilities meet" in text and "learning stage" not in text
    title, text = tp.FIELD_HELP["shep-test-model"]
    assert title == "Checkpoint" and len(text.split("\n")[0]) <= 100
    assert tp.FIELD_HELP["shep-test-model-run"][0] == "Run"
    assert "No model spec: the file is listed, never measured" in text
    assert "its LP head" in text and "a base" not in text
    assert "measured by that task" not in text and "capabilities meet" in text
    assert len(text.split()) <= 85


@pytest.fixture
def test_only_protocols(monkeypatch):
    """§5.5's test-only protocols, the product table untouched: p1 and p2 both
    take a finetuned model with the disease head and run the disease
    evaluator, so neither name is a task."""
    from dataclasses import replace

    from src.evaluation import protocols

    table = {name: replace(protocols.PROTOCOLS["disease"], name=name, label="")
             for name in ("p1", "p2")}
    monkeypatch.setattr(protocols, "PROTOCOLS", table)
    return table


def test_a_listed_disease_child_offers_the_test_only_protocols_and_starts_one(
        site, monkeypatch, test_only_protocols):
    """§5.5 wiring, the Test page's part (3): under p1 and p2, nothing on the
    page treats a protocol's name as the model's task. A listed disease child
    offers exactly p1 and p2 (named as they are, none chosen, Start waiting
    for a pick); `start_state(..., protocol="p2")` is (True, None); the start
    hands the runner protocol p2 apart from the model's task disease, and the
    job card names p2. "disease" is no protocol here: the definition's
    ordinary sentence. The base asked as p1 gets the definition's one reason
    (no head p1 scores with, and nothing in this table takes it) from Start
    and from the start, which asks no runner."""
    from src.evaluation import protocols

    _path, child = _put_bundle(site)
    _put_bundle(site, BUNDLE_BASE_RUN, None)
    base = f"gatv2/{BUNDLE_BASE_RUN}/last.pt"
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    binding = tp.workspace_binding(site.demo, tp.find(page["workspaces"], "name", "demo"))
    offered = [("p1", "p1"), ("p2", "p2")]
    assert tp.measurement_choices(tp.find_model(models, child)) == offered
    state = lambda key, protocol: tp.start_state(  # noqa: E731
        page, models, "demo", LABEL, key, binding, ("none", ""), protocol=protocol)
    assert state(child, None) == (False, "Start disabled: choose a measurement.")
    assert state(child, "p2") == (True, None)
    assert state(child, "disease") == (
        False, "Start disabled: 'disease' is not a measurement this build implements: p1 or p2.")

    # The radio as the model's listener writes it, and a pick.
    demo, _ctx, _registry = _page_component()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    on_model, on_pick = _named(demo, "_on_model"), _named(demo, "_on_measurement")
    radio_at = on_model.outputs.index(by_id["shep-test-measurement"])
    start_at = on_model.outputs.index(by_id["shep-test-start"])
    reason_at = on_model.outputs.index(by_id["shep-test-start-reason"])
    out = on_model.fn(page, models, binding, "demo", LABEL, child, None, {}, None)
    assert out[radio_at] == gr.update(choices=offered, value=None, visible=True, interactive=True)
    assert out[start_at] == gr.update(value="▶ Start test", interactive=False)
    assert "Start disabled: choose a measurement." in out[reason_at]
    picked = on_pick.fn(page, models, binding, "demo", LABEL, child, "p2")
    assert picked == [gr.update(choices=offered, value="p2", visible=True, interactive=True),
                      gr.update(value="▶ Start test", interactive=True), "", "p2"]
    # Like the Model select, the radio is an operator's control only.
    readonly = dict(page, access={"allowed": False, "condition": "not_local"})
    assert on_model.fn(readonly, models, binding, "demo", LABEL, child, "p2", {},
                       None)[radio_at] == (
        gr.update(choices=offered, value="p2", visible=True, interactive=False))
    # With no model chosen there is nothing to offer: the group is not drawn
    # (the independent check of the UI sweep, C6: its label stood over an
    # empty group and a 30 px gap at every size and in every state).
    assert by_id["shep-test-measurement"].visible is False
    assert on_model.fn(page, models, binding, "demo", LABEL, None, None, {}, None)[radio_at] == (
        gr.update(choices=[], value=None, visible=False, interactive=False))

    manager = FakeManager()
    result = tp.start_test_run(page, models, "demo", LABEL, child, False, False, request,
                               protocol="p2", manager=manager)
    assert result["success"] is True
    (inputs,) = manager.started
    assert (inputs.protocol, inputs.learning_stage, inputs.task) == ("p2", "finetune", "disease")
    card = tp.job_body_html({"status": "running", "inputs": inputs.summary(), "log": []}, [],
                            ("busy", ""))
    assert _identity_rows(card)["Measurement"] == "Mode p2, not described by this version"

    version = tp.find(page["versions"], "label", LABEL)
    reason = protocols.refusal("p1", tp.find_model(models, base)["capabilities"],
                               version["kind"], version["counts"])
    assert reason == ("the p1 measurement takes a finetune model with the disease head; this "
                      "checkpoint is a base model with no task head.")
    assert tp.measurement_choices(tp.find_model(models, base)) == []
    assert state(base, "p1") == (False, f"Start disabled: {reason}")
    assert tp.start_test_run(page, models, "demo", LABEL, base, False, False, request,
                             protocol="p1", manager=manager) == {
        "success": False, "error": f"{reason[:1].upper()}{reason[1:]} Nothing was started."}
    assert len(manager.started) == 1


def test_a_model_picked_after_another_answers_its_second_event_without_an_error(
        site, monkeypatch):
    """Through the page served by Gradio, for an operator: one pick of a Run
    or Checkpoint option sends its input event twice, each with the values
    the browser held when it was asked. On the supplied version, the disease
    child's run chosen (its latest save chosen with it; the radio offers its
    one measurement for that kind, "disease", chosen; research comparison
    Stage 7), the operator's pick "disease", then the base's run: the first of
    its events writes the base's checkpoints and choices for a supplied
    cohort, none, and the second still carries the child's checkpoint and
    "disease". Both answer, the radio ends with no choice and Start's reason
    is the definition's sentence for a model none of whose measurements
    measures a supplied cohort. When the selection's listeners took the
    radio itself, Gradio checked the second event's value against the
    choices the first had written and answered with an error before the
    page's own handling ran (`chosen_measurement`); the chosen checkpoint is
    read the same way (`model_state`), since a run's pick changes the
    Checkpoint select's choices too."""
    import html as html_lib

    from src.evaluation import protocols

    _put_bundle(site, BUNDLE_BASE_RUN, None)
    base = f"gatv2/{BUNDLE_BASE_RUN}/last.pt"
    operator_mode(monkeypatch)
    job = {"status": "idle", "job_id": None, "inputs": {}, "log": [], "outcome": None}
    monkeypatch.setattr(tp, "read_job", lambda **k: dict(job))
    monkeypatch.setattr(tp, "gpu_line", lambda **k: ("none", "GPU jobs: none"))
    demo, _ctx, _registry = _page_component()
    refresh, on_run = _named(demo, "_refresh"), _named(demo, "_on_run")
    on_model, on_pick = _named(demo, "_on_model"), _named(demo, "_on_measurement")
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    radio_at = on_run.outputs.index(by_id["shep-test-measurement"])
    reason_at = on_run.outputs.index(by_id["shep-test-start-reason"])
    checkpoint_at = on_run.outputs.index(by_id["shep-test-model"])
    run_of = lambda key: key.rsplit("/", 1)[0]  # noqa: E731

    with served(demo) as url:
        browser = ServedSession(url, demo)
        assert browser.wait(browser.join(refresh, [None] * len(refresh.inputs)))
        # The child's run picked: its latest save is chosen with it.
        chosen = browser.join(on_run, [None, None, None, "demo", LABEL, run_of(site.key), None,
                                       None, None, None])
        assert browser.wait(chosen)
        out = browser.done[chosen]["output"]["data"]
        assert out[checkpoint_at]["value"] == site.key, out[checkpoint_at]
        assert out[radio_at]["value"] == "disease" and len(out[radio_at]["choices"]) == 1
        # The checkpoint picked, as the select offers it now: twice, no error.
        again = [browser.join(on_model, [None, None, None, "demo", LABEL, site.key, None,
                                         None, None]) for _ in range(2)]
        assert browser.wait(*again)
        # The operator picks "disease", the child's one.
        picked = browser.join(on_pick, [None, None, None, "demo", LABEL, None, "disease"])
        assert browser.wait(picked)
        radio = browser.done[picked]["output"]["data"][0]
        assert radio["value"] == "disease", radio
        # The base's run picked once: two events, each asked with the radio's
        # "disease"; the second is asked while the first takes the child's
        # checkpoint away from the Checkpoint select's choices.
        events = [browser.join(on_run, [None, None, None, "demo", LABEL, run_of(base), None,
                                        "disease", None, None])
                  for _ in range(2)]
        assert browser.wait(*events)
        browser.close()
    _names, said = protocols.available(described("base", None), SUPPLIED)
    for event in events:
        out = browser.done[event]["output"]["data"]
        assert ([list(c) for c in out[radio_at]["choices"]], out[radio_at]["value"]) == (
            [], None)
        assert html_lib.escape(said) in out[reason_at], out[reason_at]


def test_a_gatv2_childs_report_is_registered_beside_its_own_file(site, monkeypatch):
    """K5a fixed the gatv2 layout key the ledger placement resolves through;
    the page's placement for a bundle run had no test: a manifest naming the
    child's digest and path is placed beside that listed file."""
    from src.utils.fingerprint import file_sha256

    path, child = _put_bundle(site)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    manifest = {"artifact_digests": {"checkpoint": file_sha256(path)},
                "checkpoint_path": str(path), "data_dir": str(site.demo)}
    place = tp.ledger_checkpoint(manifest, page, models, "demo")
    assert place.path is not None and place.path.resolve() == path.resolve()
    assert tp.item_key(place.item) == child and place.item["architecture"] == "gatv2"
    assert place.path.parent / "evaluations.json" == \
        site.demo / "runs" / "gatv2" / CHILD_RUN / "checkpoints" / "evaluations.json"


# --------------------------------------------------------------------------- every data kind
# Research comparison Stage 7: the page lists, measures and registers a KG edge
# set and a generated split as it does a supplied cohort, through the one job
# runner and the one registration, with R10's evidence chosen once by kind.
KG_LABEL, GEN_LABEL = "kg-test", "gen-val"


@pytest.fixture
def kinds(site):
    """The site's registry also holding a KG edge set (the test part of the
    disease child's recorded edge split) and a generated split (the demo
    workspace's own val split), each imported by the importer's two steps."""
    from src.evaluation.cohort_registry import (
        commit_import,
        prepare_generated_import,
        prepare_kg_edges_import,
    )

    edges = prepare_kg_edges_import(site.registry, site.demo, site.checkpoint, KG_LABEL, "test")
    generated = prepare_generated_import(site.registry, site.demo, GEN_LABEL, "val")
    return SimpleNamespace(
        edges=commit_import(edges, site.registry, edges.preview_id),
        generated=commit_import(generated, site.registry, generated.preview_id))


def _visible(html: str) -> str:
    """The text a renderer's markup shows, its entities read back."""
    import html as html_lib

    return " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", html)).split())


def test_a_long_version_label_is_shortened_in_its_option_never_in_its_value():
    """The skill's §8 for the version select: a label past the option's room
    keeps its start and end, alike ones are numbered, an unreadable version
    says so; each value is the full label, whatever the words."""
    long = "hospital-cohort-2026-09-wide-phenotype-set-"
    versions = [{"label": f"{long}a", "kind": "supplied", "role": "test", "counts": {}},
                {"label": f"{long}b", "kind": "supplied", "role": "test", "counts": {}},
                {"label": "kg-x", "kind": "kg_edges", "role": "test", "counts": {}},
                {"label": "torn", "problem": "it cannot be read"}]
    choices = tp.version_choices({"versions": versions})
    labels = [label for label, _value in choices]
    assert [value for _label, value in choices] == [f"{long}a", f"{long}b", "torn", "kg-x"]
    assert len(set(labels)) == len(labels), labels
    for label in labels:
        assert len(label) <= renderers.OPTION_CHARS and label.count(" · ") == 1, label
    assert labels[0].startswith("hospital") and labels[0].endswith("-a · role test")
    assert labels[2:] == ["torn · cannot be read", "kg-x · KG edge set (test)"]


def test_the_data_panel_lists_every_kind_by_kind_and_role_with_its_counts(
        site, kinds, monkeypatch):
    """The version select lists the supplied cohort as it always read, then
    the KG edge set and the generated split by kind and role, each option
    short with one separator (skill §8), also for a version mapped against
    another graph, whose state is its details' tag; the chosen version's
    details give its counts in its own units, each count beside the unit its
    manifest records, never a supplied cohort's words, and never its kind and
    role again (skill §1.2; the browser check of 7e1d09e: "KG edge set (test)"
    under "kg-test · KG edge set (test)"); no KG edge sets' empty state once
    one is listed. The panel and the select read "Data", not "Cohort"."""
    page, _request = _page(monkeypatch)
    binding = tp.workspace_binding(site.demo, tp.find(page["workspaces"], "name", "demo"))
    choices = tp.version_choices(page)
    assert [value for _label, value in choices] == [LABEL, KG_LABEL, GEN_LABEL]
    labels = dict((value, label) for label, value in choices)
    assert labels[KG_LABEL] == "kg-test · KG edge set (test)"
    assert labels[GEN_LABEL] == "gen-val · Generated split (val)"
    # One rule for the select (skill §8): every option, the supplied cohort's
    # too ("label · role R", its counts its details), unique, at most 40
    # characters and one separator, its value the full label.
    assert len(set(labels.values())) == len(labels)
    for value, label in labels.items():
        assert len(label) <= 40 and label.count(" · ") == 1 and label.startswith(value), label
    assert labels[LABEL].startswith(f"{LABEL} · role ") and "ranked-eligible" not in labels[LABEL]
    # The version chosen is the first option, whatever order the registry
    # lists versions in (the review of Stage 7: it was the registry's first).
    for versions in (page["versions"], list(reversed(page["versions"]))):
        shuffled = {**page, "versions": versions}
        for bound in (binding, None):
            assert tp.default_version(shuffled, bound, None) == (
                tp.version_choices(shuffled)[0][1]), [v["label"] for v in versions]
    # A version mapped against another graph: its option is the same (one
    # separator, no " · another graph"), its details say so by the shared
    # mark (▲, a condition) with its refusal beside the version, never only
    # at Start (BACKLOG 72; the ops-wiring round's B3), and Start points there.
    another = dict(binding, graph={**binding["graph"], "kg": "0" * 64})

    for version, words in ((kinds.edges, "KG edge set (test)"),
                           (kinds.generated, "Generated split (val)")):
        label = version.manifest["label"]
        shown = _visible(tp.version_html(page, label, binding))
        assert words not in shown and words in labels[label], shown
        mapped = tp.version_html(page, label, another)
        assert "Mapped against another graph (kg differ)" in _visible(mapped)
        assert tp.r.status_mark("warn", "Mapped against another graph (kg differ)") in mapped
        assert tp.r.status_mark("error", tp.ANOTHER_GRAPH) in mapped
        assert tp.ANOTHER_GRAPH not in tp.version_html(page, label, binding)
        assert tp.r.status_mark("ok", "Bound to this graph") in tp.version_html(page, label,
                                                                               binding)
        counts, units = version.manifest["counts"], version.manifest["count_units"]
        assert f"Counts and units ({len(counts)})" in shown
        # Each count's unit by name; its value only where the words above do
        # not give it, so no count is shown twice (skill §1.2; the review of
        # Stage 7).
        said = {"kg_edges": ("pairs", "positive_edges", "message_passing_edges"),
                "generated": ("samples", "diseases", "task_fixture_samples")}[
            version.manifest["kind"]]
        read = shown.replace(" :", ":")  # `_visible` puts a space where </code> was
        for name, unit in units.items():
            assert (f"{name}: {unit}" if name in said else f"{name} {counts[name]}: {unit}"
                    ) in read, name
            assert f"{name} {counts[name]}:" not in read or name not in said, name
        assert any(name not in said for name in units) or version is kinds.generated
        for said in ("source cases", "ranked-eligible", "cohort", "No KG edge set"):
            assert said not in shown, (label, said)
    edges = kinds.edges.manifest["counts"]
    assert (f"{edges['pairs']} pairs from {edges['positive_edges']} positive edges"
            in _visible(tp.version_html(page, KG_LABEL, binding)))
    assert "source cases" in tp.version_html(page, LABEL, binding)
    assert "<h2>Data</h2>" in tp.cohort_header_html(page)
    demo, _ctx, _registry = _page_component()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    assert by_id["shep-test-version"].label == "Data version"
    assert tp.FIELD_HELP["shep-test-version"][0] == "Data version"


def test_with_no_kg_edge_set_the_data_panel_names_the_import_command(site, monkeypatch):
    """The KG edge sets' empty state (skill §9): with none in the registry,
    one line under the chosen version, whatever it is, says what one is for;
    the command that imports one on the machine, escaped, is a row of its own
    under "Details ›" and the guide's section is on its Ref line (the text
    brief's §8: never a command or a section in the sentence); none when the
    registry is not available."""
    page, _request = _page(monkeypatch)
    empty = tp.kg_edges_empty_html(page)
    assert empty and "No KG edge set in the registry yet" in empty
    assert ("scripts/import_cohort.py --kind kg_edges --from-checkpoint &lt;checkpoint&gt;"
            in empty and "<checkpoint>" not in empty)
    lead, details = empty.split("<details", 1)
    assert "scripts/" not in lead and "§" not in lead and "<code" not in lead
    assert "scripts/import_cohort.py" in details and "Ref: deployment guide §4.9" in details
    assert tp.version_html(page, LABEL, None).endswith(empty)
    assert tp.version_html(page, None, None).endswith(empty)
    assert tp.version_html({**page, "versions": []}, None, None).startswith(
        tp._reason_html("No data version is in the registry yet: import a cohort with "
                        "“Import a new version…”."))
    assert tp.kg_edges_empty_html({**page, "registry_problem": "unset"}) == ""
    assert tp.version_html({**page, "registry_problem": "unset"}, LABEL, None) == ""


def _first_use_screen(page, gpu=("unread", "GPU jobs: not set up")):
    """What the Test page draws for `page` with nothing chosen (an operator):
    Data, Run (its fields' details, Start's reason) and Reports, as the page
    open writes them; `gpu` the GPU chip's line (not set up while no
    workspaces folder is)."""
    allowed = bool((page.get("access") or {}).get("allowed"))
    empty = {"label": None, "runs": []}
    can, why = tp.start_state(page, dict(tp._EMPTY_MODELS), None, None, None, None, gpu)
    return {
        "data": tp.cohort_header_html(page) + tp.workspace_html(page, None)
        + tp.version_html(page, None, None) + tp.import_reason_html(page, None),
        "run": tp.model_html(page, dict(tp._EMPTY_MODELS), None) + tp.evidence_html(page, None)
        + tp.fixed_html(page) + tp.start_reason_html(page, allowed, can, why),
        "reports": tp.reports_header_html(empty) + tp.reports_html(empty, None, page=page)
        + tp.result_html(empty, None, []),
    }, (can, why)


def _unset(monkeypatch, *names):
    for name in names:
        monkeypatch.delenv(name, raising=False)


def test_at_first_use_the_screen_says_each_setting_once_and_each_control_its_relation(
        monkeypatch, tmp_path):
    """The sweep's B-09 and R2 (the skill's §9; the owner's first start, GB10-UI-02):
    with no results registry and no workspaces folder, the screen says each
    once, in the Data panel's one notice (its state, what it means here, one
    link to Settings › Data locations, the rest under Details ›); the header's
    right side is empty while it shows; the Workspace select draws nothing
    under it; Import and Start each say a one-line relation to it (Start's
    reason the words `start_state` gives); the Reports panel says why it is
    empty. It said the registry's whole sentence in the Data header, the
    workspaces folder's under the Workspace select, "registry: not
    available" at the header's right and "Disabled: …" lines beside Import and
    Start, with the Reports panel blank. Then each setting alone."""
    from src.config import server_settings

    _unset(monkeypatch, "SHEPHERD_WORKSPACES_ROOT", "SHEPHERD_REGISTRY")
    page = tp.read_page(operator_request(monkeypatch))
    screen, (can, why) = _first_use_screen(page)
    html = "".join(screen.values())
    keys = re.findall(r'data-notice="([^"]+)"', html)
    assert sorted(k for found in keys for k in found.split()) == ["registry", "workspaces_root"]
    text = _plain(html)
    for said in ("No results registry is set.", "No workspaces folder is set."):
        assert text.count(said) == 1, said
    # One link for both, in the settings' words for several (the sweep's S12).
    assert html.count("data-shep-go=") == 1 and server_settings.CHOOSE_THEM_IN_DATA_LOCATIONS in text
    # The settings' whole sentences, the env vars and the old lines are said nowhere.
    for whole in (server_settings.REGISTRY_NOT_SET, server_settings.WORKSPACES_ROOT_NOT_SET):
        assert whole[:40] not in text
    outside = _plain(re.sub(r"<details.*?</details>", " ", html, flags=re.S))
    assert "SHEPHERD_" not in outside and "registry: " not in text
    # Each "… disabled: …" line is a relation to a notice, by its short words.
    shorts = {server_settings.REGISTRY_NOTICE.short, server_settings.WORKSPACES_ROOT_NOTICE.short}
    causes = re.findall(r"\b(?:Start|Import) disabled: ([^.(]+)", text)
    assert causes and {c.strip() for c in causes} <= shorts, causes
    assert 'class="shep-panel-right"' not in screen["data"] + screen["reports"]
    assert tp.workspace_html(page, None) == ""
    # Each control a relation, never the notice: Import's and Start's, one line each.
    rels = re.findall(r'data-notice-rel="([^"]+)"', html)
    assert rels == ["registry", "registry"]
    assert "Import disabled: no results registry is set." in _plain(screen["data"])
    assert (can, why) == (False, "Start disabled: no results registry is set (see Data).")
    assert why in _plain(screen["run"])
    # The Reports panel's empty state, in the flow.
    assert f'<p class="shep-empty">{tp.e(tp.REPORTS_NO_REGISTRY)}</p>' in screen["reports"]

    # The registry set (empty), the workspaces folder not: its notice alone,
    # and the controls' relations to it.
    registry = tmp_path / "registry"
    registry.mkdir()
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(registry))
    page = tp.read_page(operator_request(monkeypatch))
    screen, (can, why) = _first_use_screen(page)
    html = "".join(screen.values())
    assert re.findall(r'data-notice="([^"]+)"', html) == ["workspaces_root"]
    assert _plain(html).count("No workspaces folder is set.") == 1
    assert "Testing waits for it." in _plain(screen["data"])
    assert (can, why) == (False, "Start disabled: no workspaces folder is set (see Data).")
    assert "Import disabled: no workspaces folder is set." in _plain(screen["data"])
    assert tp.e(tp.REPORTS_NO_VERSION) in screen["reports"]

    # Both set, both empty: no notice; each empty place in its own words.
    folder = tmp_path / "workspaces"
    folder.mkdir()
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(folder))
    page = tp.read_page(operator_request(monkeypatch))
    screen, (can, why) = _first_use_screen(page, ("none", "GPU jobs: none"))
    html = "".join(screen.values())
    assert "data-notice" not in html and "registry: server setting" in screen["data"]
    assert tp.NO_WORKSPACE_YET in _plain(screen["data"])
    assert why == "Start disabled: no workspace is listed (see Data)."
    assert "Import disabled: no workspace to import against." in _plain(screen["data"])
    assert tp.e(tp.REPORTS_NO_VERSION) in screen["reports"]


def test_the_reports_panel_says_why_it_is_empty_in_its_own_words_per_cause():
    """The skill's §9: an empty Reports panel says what will appear and the
    next step, by its cause, in the flow (`.shep-empty`): no registry, a
    registry set and not usable, no version in it, none chosen, a version no
    run has kept a report on; a store that could not be listed says its
    problem instead, never "no report". Its title says a count only while it
    lists any (the empty state says why there is none, once)."""
    neutral = {"key": "registry", "kind": "neutral", "short": "no results registry is set"}
    refused = dict(neutral, kind="error")
    none = {"label": None, "runs": []}
    for page, words in (({"registry_notice": neutral, "registry_problem": "x"},
                         tp.REPORTS_NO_REGISTRY),
                        ({"registry_notice": refused, "registry_problem": "x"},
                         tp.REPORTS_REGISTRY_REFUSED),
                        ({"versions": []}, tp.REPORTS_NO_VERSION),
                        ({"versions": [{"label": LABEL}]}, tp.REPORTS_CHOOSE_VERSION)):
        html = tp.reports_html(none, None, page=page)
        assert html == ('<div class="shep-test-replist"><p class="shep-empty">'
                        f'{tp.e(words)}</p></div>')
    assert tp.reports_html(none, None) == ""          # before the page is read
    empty = tp.reports_html({"label": LABEL, "runs": []}, None)
    assert f'<p class="shep-empty">{tp.e(tp.REPORTS_NONE_YET)}</p>' in empty and "<table" not in empty
    broken = tp.reports_html({"label": LABEL, "runs": [], "problem": "the store is busy"}, None)
    assert "shep-empty" not in broken and "the store is busy" in broken
    for reports in (none, {"label": LABEL, "runs": []}):
        assert "shep-panel-right" not in tp.reports_header_html(reports)
    assert "2 of the selected version" in tp.reports_header_html(
        {"label": LABEL, "runs": _runs(2)})
    for words in (tp.REPORTS_NO_REGISTRY, tp.REPORTS_REGISTRY_REFUSED, tp.REPORTS_NO_VERSION,
                  tp.REPORTS_CHOOSE_VERSION, tp.REPORTS_NONE_YET):
        assert ";" not in words and len(words.split(". ")) <= 2, words


def test_the_panels_are_sized_by_what_they_hold():
    """The sweep's M4 and B-04 (DECISIONS 7): the three panels carry no
    Gradio share fitted to one window (`scale=`) but the shared sizing
    (`.shep-sized`): a basis, a growth and a floor in rem each, the basis at
    least the floor; while the Reports panel draws no table (an empty state)
    its floor and growth are smaller, so Data and Run take the room it left
    blank (at 1366 x 768 at first use the Run panel scrolled 100 px while
    Reports stood 90 % blank). The import drawer's width is in rem too."""
    from tests.unit.test_webui_shell import side, stack

    demo, _ctx, _registry = _page_component()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    rules = dict(_css_rules(tp.PAGE_CSS))
    sizes = {}
    for panel in ("shep-test-cohort", "shep-test-run", "shep-test-result"):
        column = by_id[panel]
        assert "shep-sized" in column.elem_classes and column.scale in (None, 1), panel
        found = dict(re.findall(r"--shep-panel-(\w+): ([\d.]+)(?:rem)?;", rules[f"#{panel}"]))
        assert set(found) == {"basis", "grow", "min"}, panel
        sizes[panel] = {k: float(v) for k, v in found.items()}
        assert re.search(r"--shep-panel-(basis|min): [\d.]+rem", rules[f"#{panel}"]), panel
        assert sizes[panel]["basis"] >= sizes[panel]["min"], panel
    blank = {k: float(v) for k, v in re.findall(
        r"--shep-panel-(\w+): ([\d.]+)", rules["#shep-test-result:not(:has(.shep-test-reps))"])}
    full = sizes["shep-test-result"]
    assert blank["min"] < full["min"] and blank["grow"] < full["grow"]
    assert blank["grow"] < min(sizes["shep-test-cohort"]["grow"], sizes["shep-test-run"]["grow"])
    for css in (side(tp.PAGE_CSS), stack(tp.PAGE_CSS)):
        drawer = re.search(r"\.shep-test-drawer \{[^}]*width: (min\([^)]*\))", css).group(1)
        assert "rem" in drawer and "px" not in drawer, drawer


def test_the_run_panels_rows_are_bars_of_whole_groups():
    """The sweep's M2 (`.shep-bar`): the Run select and its ↻ are one group of
    a bar (they never part, the button on the box's line); the foot's job row
    is a bar whose state grows and whose Cancel sits at the line's end, on a
    line of its own when the state leaves it no room (it was a row that never
    wrapped, so a long state squeezed). The job's output is drawn in the
    monospace face (`.shep-mono`; the page's own family lost to the mount's
    rule, B-15)."""
    demo, _ctx, _registry = _page_component()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    def row_of(block):
        while not isinstance(block, gr.Row):
            block = block.parent
        return block

    row = row_of(by_id["shep-test-model-run"])
    assert {"shep-grp", "shep-grp-grow", "shep-field-row"} <= set(row.elem_classes)
    assert row_of(by_id["shep-test-model-reload"]) is row
    # Its note first, why the lists are empty, on the line above the group
    # (V-15; L7 SW-LK10: a reason of the bar's, the ↻'s for the shared rule).
    assert "shep-bar" in row.parent.elem_classes
    assert row.parent.children == [by_id["shep-test-model-why"], row]
    assert "shep-bar-note" in by_id["shep-test-model-why"].elem_classes
    top = row_of(by_id["shep-test-job-head"])
    assert {"shep-test-job-top", "shep-bar"} <= set(top.elem_classes)
    assert "shep-grp-grow" in by_id["shep-test-job-head"].elem_classes
    assert "shep-grp-end" in by_id["shep-test-cancel"].elem_classes
    assert "nowrap" not in dict(_css_rules(tp.PAGE_CSS)).get(".shep-test-job-top", "")
    ended = {"status": "failed", "inputs": {}, "log": ["a line"], "end_time": "2026-11-14T05:00:00Z",
             "outcome": {"kept": False, "registered": False, "summary": "Failed."}}
    assert 'class="shep-test-log shep-mono"' in tp.job_body_html(ended, [], ("none", ""))
    assert "font-family" not in dict(_css_rules(tp.PAGE_CSS))[".shep-test-log"]


def test_the_pages_help_texts_meet_the_text_budget():
    """The sweep's text brief §11 (`help.TEXT_BUDGET`): each of the page's
    registered help texts has a lead that answers the question in two bubble
    lines, then none or two to four items of one form, each two lines at most,
    80 words at most, Ref last, plain words. Before: 14 of 21 over. A
    described measurement's result note is the governed caveat's own text
    (`caveats.mode_note`, shown on Models and Results too), each sentence a
    line: it is kept as written, never reworded here."""
    from src.evaluation.caveats import mode_note
    from src.webui.components.help import TEXT_BUDGET, budget_notes, budget_problems

    texts = {**{k: v for k, (_t, v) in tp.FIELD_HELP.items()},
             **{k: v for k, (_t, v) in tp.TOPIC_HELP.items()}}
    governed = {tp._MODE_HELP[mode]: mode for mode in tp._RESULT_WORDS}
    for key, text in texts.items():
        if key in governed:
            assert text == sentence_lines(mode_note(governed[key])), key
            continue
        assert budget_problems(text, TEXT_BUDGET) == (), (key, budget_problems(text, TEXT_BUDGET))
        assert budget_notes(text, TEXT_BUDGET) == (), key


def test_the_measurement_options_follow_the_model_and_the_chosen_versions_kind(
        site, kinds, monkeypatch):
    """Item 2 (and Stage 5's interim, fixed): the options are what the model's
    capabilities meet that measure the chosen version's kind, in plain words
    (skill §5), with the preselection rule kept. The disease child: its task's
    measurement alone on the supplied cohort and the generated split, KG link
    prediction alone on the KG edge set, each chosen; both, none chosen, with
    no version chosen. The base: KG link prediction alone, chosen, on the KG
    edge set; nothing on the others, and Start says why in the definition's
    words."""
    from src.evaluation import protocols

    _put_bundle(site, BUNDLE_BASE_RUN, None)
    base = f"gatv2/{BUNDLE_BASE_RUN}/last.pt"
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    binding = tp.workspace_binding(site.demo, tp.find(page["workspaces"], "name", "demo"))
    child, based = tp.find_model(models, site.key), tp.find_model(models, base)
    version = {label: tp.find(page["versions"], "label", label)
               for label in (LABEL, KG_LABEL, GEN_LABEL)}
    lp, disease = ("KG link prediction", "kg_link_prediction"), ("Disease", "disease")
    for label, offered in ((LABEL, [disease]), (KG_LABEL, [lp]), (GEN_LABEL, [disease])):
        choices = tp.measurement_choices(child, version[label])
        assert choices == offered, label
        assert tp.chosen_measurement(choices, None) == offered[0][1]
    both = tp.measurement_choices(child)
    assert both == [disease, lp] and tp.chosen_measurement(both, None) is None
    assert all(code not in label for label, code in both)
    assert tp.measurement_choices(based, version[KG_LABEL]) == [lp]
    for label, kind in ((LABEL, protocols.SUPPLIED), (GEN_LABEL, protocols.GENERATED)):
        assert tp.measurement_choices(based, version[label]) == []
        said = protocols.available(based["capabilities"], kind)[1]
        assert ("it can be measured by KG link prediction, which measures a KG edge set. "
                "Ref: kg_link_prediction.") in said
        assert tp.start_state(page, models, "demo", label, base, binding, ("none", ""),
                              protocol=None) == (False, f"Start disabled: {said}")
    assert tp.start_state(page, models, "demo", KG_LABEL, base, binding, ("none", ""),
                          protocol="kg_link_prediction") == (True, None)


@pytest.mark.parametrize("setting", ["set", "refused"])
def test_a_kg_or_generated_start_needs_no_evidence_setting_and_is_given_none(
        site, kinds, monkeypatch, setting):
    """Item 1, the page's part: Start's check and the start use the runner's
    one choice by the version's kind. With the evidence setting set or
    refused, a KG edge set's and a generated split's Start is enabled and the
    start hands the runner no evidence directory; their R10 row reads "Not
    applicable" with why. A supplied cohort takes the setting's directories,
    and a refused setting disables its Start (a red ✕ with words) and refuses
    its start."""
    from src.evaluation import protocols

    evidence = site.root / "evidence"
    evidence.mkdir()
    monkeypatch.setenv("SHEPHERD_TRAINING_EVIDENCE_DIRS",
                       str(evidence) if setting == "set" else "relative/evidence")
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    binding = tp.workspace_binding(site.demo, tp.find(page["workspaces"], "name", "demo"))
    manager = FakeManager()
    for label, protocol, kind in ((KG_LABEL, "kg_link_prediction", protocols.KG_EDGES),
                                  (GEN_LABEL, "disease", protocols.GENERATED)):
        assert tp.start_state(page, models, "demo", label, site.key, binding, ("none", ""),
                              protocol=protocol) == (True, None), label
        result = tp.start_test_run(page, models, "demo", label, site.key, False, False,
                                   request, protocol=protocol, manager=manager)
        assert result["success"] is True, result
        inputs = manager.started[-1]
        assert (inputs.version_label, inputs.version_kind, inputs.evidence_dirs) == (
            label, kind, ())
        inputs.reports_dir()  # the runner's own judgment takes them
        row = tp.evidence_html(page, "demo", tp.find(page["versions"], "label", label))
        assert tp.r.status_mark("neutral", "Not applicable") in row
        # Each fact a line of its own (L7 SW-LK7), no " · " chain.
        assert _visible(row) == (f"R10 evidence ○ Not applicable "
                                 f"{protocols.training_evidence_refusal(kind)}"), row
        assert " · " not in row
        assert "Refused" not in row and "relative/evidence" not in row
        card = tp.job_body_html({"status": "running", "inputs": inputs.summary(), "log": []},
                                [], ("busy", ""))
        assert _identity_rows(card)["R10 evidence"] == "○ Not applicable"
    supplied = tp.find(page["versions"], "label", LABEL)
    allowed, why = tp.start_state(page, models, "demo", LABEL, site.key, binding, ("none", ""),
                                  protocol="disease")
    row = tp.evidence_html(page, "demo", supplied)
    result = tp.start_test_run(page, models, "demo", LABEL, site.key, False, False, request,
                               protocol="disease", manager=manager)
    if setting == "set":
        assert (allowed, why) == (True, None) and result["success"] is True
        assert manager.started[-1].evidence_dirs == (evidence.resolve(),)
        assert "<code>evidence</code> (the server's setting)" in row and "Not applicable" not in row
    else:
        assert allowed is False and why.startswith(
            "Start disabled: the evidence directories setting is refused: ")
        assert result["success"] is False and result["error"].endswith("Nothing was started.")
        assert '<span class="shep-st shep-st-error"><span class="shep-st-shape" aria-hidden="true">✕</span><span class="shep-st-sp">\u00a0</span><span class="shep-st-w">Refused: ' in row
        assert len(manager.started) == 2


def test_the_r10_row_renders_each_state_escaped():
    """The R10 row's renderer on its own (skill §6, §11): "Not applicable"
    grey with a hollow ○ and the reason for a KG edge set and a generated
    split; for a supplied cohort (and before a version is chosen) a refused
    setting as a red ✕ with "Refused:" and its words, escaped and redacted;
    the setting's directory names escaped. Never a colour without its mark
    and word."""
    from src.evaluation import protocols

    hostile = {"evidence": {"dirs": [], "names": [], "problem": f"refused: {HOSTILE} at /srv/x",
                            "note": None},
               "roots": [("/srv", "<srv>")]}
    for version in (None, {"kind": "supplied"}):
        row = tp.evidence_html(hostile, "demo", version)
        assert HOSTILE not in row and "&lt;img src=x onerror=alert(1)&gt;" in row
        assert "/srv/x" not in row and "&lt;srv&gt;/x" in row
        assert '<span class="shep-st shep-st-error"><span class="shep-st-shape" aria-hidden="true">✕</span><span class="shep-st-sp">\u00a0</span><span class="shep-st-w">Refused: ' in row
    for kind in (protocols.KG_EDGES, protocols.GENERATED):
        row = tp.evidence_html(hostile, "demo", {"kind": kind})
        assert "Refused" not in row and HOSTILE not in row and "onerror" not in row
        assert tp.r.status_mark("neutral", "Not applicable") in row
    named = {"evidence": {"dirs": [f"/srv/{HOSTILE}"], "names": [HOSTILE], "problem": None,
                          "note": None}}
    row = tp.evidence_html(named, "demo", {"kind": "supplied"})
    assert HOSTILE not in row and f"<code>{tp.e(HOSTILE)}</code>" in row
    assert tp.run_evidence(named, {"kind": "supplied"}).dirs == (tp.Path(f"/srv/{HOSTILE}"),)
    assert tp.run_evidence(named, {"kind": protocols.KG_EDGES}).dirs == ()
    # A run's card says what its own inputs recorded, whatever is chosen now.
    for kind, said in (("supplied", "the workspace"),
                       (protocols.KG_EDGES, "○ Not applicable"),
                       (protocols.GENERATED, "○ Not applicable")):
        card = tp.job_body_html({"status": "running", "inputs": {"version_kind": kind},
                                 "log": []}, [], ("busy", ""))
        assert _identity_rows(card)["R10 evidence"] == said, kind


def test_register_takes_a_kept_report_of_every_kind_through_the_one_registration(
        site, kinds, monkeypatch):
    """Register works for every kind through `register_report` (and the
    record help's "it can be registered later from this page" holds for each):
    a KG edge set's run and a generated split's run, each measured by the
    real CLI on CPU and kept unregistered, are each registered from the page
    beside the listed checkpoint that holds the weights; the ledger keeps
    each record under its own kind, and the page then reads it registered."""
    import scripts.measure_scorer as measure_scorer
    from src.evaluation.sidecar import read_ledger

    for label, mode in ((KG_LABEL, "kg_link_prediction"), (GEN_LABEL, "disease")):
        assert measure_scorer.main(["--mode", mode, "--checkpoint", str(site.checkpoint),
                                    "--data-dir", str(site.demo), "--cohort", label,
                                    "--registry", str(site.registry), "--device", "cpu"]) == 0
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    for label, kind in ((KG_LABEL, "kg_edges"), (GEN_LABEL, "generated")):
        reports = tp.read_reports(label)
        (run,) = reports["runs"]
        assert run["record"]["cohort"]["kind"] == kind
        assert tp.register_state(page, reports, run["run_id"]) == (True, None)
        done, said = tp.register_selected(page, reports, models, "demo", run["run_id"], request)
        assert done is True, said
        again = tp.read_reports(label)
        assert tp.register_state(page, again, run["run_id"]) == (False, "Already registered.")
    ledger, _ = read_ledger(site.checkpoint.parent / "evaluations.json")
    assert [r["cohort"]["kind"] for r in ledger["records"]] == ["kg_edges", "generated"]


# --------------------------------------------------------------------------- one fact once (B3)
def _identity_rows(html):
    """The Result's identity rows, key → plain words."""
    return {key: _plain(value) for key, value in re.findall(
        r'<dt class="shep-k">([^<]*)</dt><dd class="shep-v">(.*?)</dd>', html)}


def test_a_past_result_keeps_its_own_identity_after_the_selection_changes(site, monkeypatch):
    """The skill's §1.2 (the ops-wiring round's B3): the Result says its
    workspace, model and version by their relation to what is chosen now when
    they are the same, "the selected …", and in full when they are not, drawn
    from the record each time: a past result keeps its own identity when
    another checkpoint, version or workspace is chosen, never taking the
    selection's. The reports table's Model cell says the same relation."""
    _path, other = _put_bundle(site)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    binding = tp.workspace_binding(site.demo, tp.find(page["workspaces"], "name", "demo"))
    reports = tp.read_reports(LABEL)

    def shown(workspace, label, key, bound=binding):
        chosen = tp.selection_of(page, models, bound, workspace, label, key)
        result = tp.result_html(reports, site.run_id, [], selection=chosen, models=models,
                                workspaces=page["workspaces"])
        table = tp.reports_html(reports, site.run_id, models=models, selection=chosen)
        return _identity_rows(result), _report_rows(table)[site.run_id][1]

    rows, cell = shown("demo", LABEL, site.key)
    assert (rows["Dataset"], rows["Model"], rows["Version"]) == (
        tp.SELECTED_DATASET, tp.SELECTED_MODEL, tp.SELECTED_VERSION)
    # Its run id, which names the model's fingerprint again, one click away.
    assert cell == tp.SELECTED_MODEL and rows["Run id"] == site.run_id and "Report" not in rows
    # Another checkpoint chosen: the report's own model, by what it is, its
    # run and its file (the words of its Run → Checkpoint pair) and, in the
    # Result, its fingerprint.
    item = tp.find_model(models, site.key)
    run_option, file_option = ml.pair_words(tp.model_parents(models))[site.key]
    rows, cell = shown("demo", LABEL, other)
    whole_run = renderers.run_words(item["architecture"], item["run"], fit=False)
    assert rows["Model"].startswith(
        f"{renderers.model_kind_text(item)} {whole_run} {file_option}")
    assert site.sha[:12] in rows["Model"] and tp.SELECTED_MODEL not in rows["Model"]
    assert cell == f"{renderers.model_kind_text(item)} {run_option} {file_option}"
    # Nothing chosen: the same, never "the selected model".
    rows, _cell = shown("demo", LABEL, None)
    assert site.sha[:12] in rows["Model"]
    # Another version chosen (or none): its own label, role and fingerprint.
    rows, _cell = shown("demo", None, site.key)
    version = tp.find(page["versions"], "label", LABEL)
    assert rows["Version"].startswith(f"{LABEL} (role ")
    assert version["manifest_sha256"][:12] in rows["Version"]
    # Another workspace's dataset record chosen: the listed workspaces whose
    # dataset record the report measured.
    rows, _cell = shown("copy", LABEL, site.key, {**binding, "record_sha256": "0" * 64})
    assert rows["Dataset"].startswith("demo's dataset record")


def test_a_result_is_the_selected_workspaces_only_by_the_relation_its_record_holds(
        site, monkeypatch):
    """NEW-test-5: the Result said "the selected workspace" whenever the
    report's dataset record digest was the selected workspace's, so a report
    measured in another workspace built on the same dataset record was
    attributed to the selected one. A report records its dataset record,
    never the workspace it read: the Result says exactly that relation, "the
    selected workspace's dataset record", and names every other listed
    workspace holding the same record, the one it was measured in among them;
    a record no listed workspace holds is said so, with its origin."""
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    reports = tp.read_reports(LABEL)
    binding = tp.workspace_binding(site.demo, tp.find(page["workspaces"], "name", "demo"))
    demo = tp.find(page["workspaces"], "name", "demo")
    # A second listed workspace on the same dataset record (a copy of demo),
    # chosen: the report was measured in demo.
    workspaces = [*page["workspaces"], {**demo, "name": "demo-copy"}]
    chosen = tp.selection_of(page, models, binding, "demo-copy", LABEL, site.key)
    result = tp.result_html(reports, site.run_id, [], selection=chosen, models=models,
                            workspaces=workspaces)
    dataset = _identity_rows(result)["Dataset"]
    assert dataset.startswith(tp.SELECTED_DATASET)
    assert ("demo holds the same record. The report does not record which workspace was read."
            in dataset)
    assert _plain(result).count("the selected workspace") == 1     # only "…'s dataset record"
    assert "the selected workspace " not in _plain(result).replace(tp.SELECTED_DATASET, "")
    # Demo chosen: the copy is named the same way.
    chosen = tp.selection_of(page, models, binding, "demo", LABEL, site.key)
    rows = _identity_rows(tp.result_html(reports, site.run_id, [], selection=chosen,
                                         models=models, workspaces=workspaces))
    assert "demo-copy holds the same record" in rows["Dataset"]
    # Alone on its record: the relation and nothing more.
    rows = _identity_rows(tp.result_html(reports, site.run_id, [], selection=chosen,
                                         models=models, workspaces=page["workspaces"]))
    assert rows["Dataset"] == tp.SELECTED_DATASET
    # Another workspace's record chosen, the report's held by no listed
    # workspace: so, and its origin.
    other = tp.selection_of(page, models, {**binding, "record_sha256": "0" * 64}, "demo",
                            LABEL, site.key)
    rows = _identity_rows(tp.result_html(reports, site.run_id, [], selection=other,
                                         models=models, workspaces=[]))
    assert rows["Dataset"].startswith("○ No listed workspace holds its dataset record")
    assert "synthetic — a generated stand-in" in rows["Dataset"]


def test_a_model_not_chosen_is_named_by_its_kind_its_run_and_its_file():
    """Review of B3 (the skill's §1.2, decision F1): a report of a listed
    model other than the chosen one is named, in the reports table and in
    the Result, by what it is (`renderers.model_kind_text`) and by the words
    of its Run → Checkpoint pair (`model_list.pair_words`), so two files of
    one run, both "1 epoch completed", never read alike, and the
    frozen-backbone control's past result never reads like the full
    fine-tune's. Seen in the browser check: two files of one run with
    identical Model cells, and the frozen child's row without "frozen"."""
    def item(run, filename, sha, **more):
        return {"architecture": "gatv2", "run": run, "filename": filename, "sha256": sha,
                "record": "model", "learning_stage": "finetune", "task": "disease",
                "completed_epochs": 1, "checkpoint_kind": "full", **more}

    full_run, frozen_run = "20261113-193200_disease_aa11bb", "20261113-193600_disease_cc22dd"
    frozen_transfer = {"trainable": ["patient_encoder", "task_head"],
                       "parts": {"input": "frozen", "encoder": "frozen", "lp_head": "frozen",
                                 "patient_encoder": "trainable", "task_head": "trainable"}}
    latest = item(full_run, "last.pt", "a" * 64)
    saved = item(full_run, "model-01-0.2760.pt", "b" * 64)
    frozen = item(frozen_run, "last.pt", "c" * 64, weight_transfer=frozen_transfer)
    chosen = item("20261113-194000_other_ee33ff", "last.pt", "d" * 64)
    models = {"items": [chosen, frozen, latest, saved]}
    selection = tp.Selection("demo", None, chosen, None)
    runs = [{"run_id": f"20261113T21{i}000Z-{m['sha256'][:12]}-0000000{i}",
             "when": f"2026-11-13 21:{i}0 UTC",
             "record": {"checkpoint_digest": m["sha256"], "scoring": {"mode": "disease"}},
             "state": {"kept": "kept", "registered": "not_registered"}}
            for i, m in enumerate((latest, saved, frozen))]
    cells = {key: row[1] for key, row in _report_rows(tp.reports_html(
        {"label": LABEL, "runs": runs}, None, models=models, selection=selection)).items()}
    latest_cell, saved_cell, frozen_cell = (cells[run["run_id"]] for run in runs)
    assert latest_cell != saved_cell
    assert latest_cell.startswith("Disease disease · GATv2 2026-11-13 19:32 Latest save")
    assert "model-01-0.2760.pt" in saved_cell
    assert frozen_cell.startswith(f"Disease · {renderers.FROZEN_BACKBONE_WORDS} ")
    assert renderers.FROZEN_BACKBONE_WORDS not in latest_cell
    view = {"cohort": {"version_manifest_sha256": "e" * 64, "kind": "supplied", "role": "test",
                       "label": LABEL}}

    def model_row(model):
        rows = dict(tp.result_identity_rows({"checkpoint_digest": model["sha256"]}, view,
                                            selection, models, []))
        return _plain(rows["Model"])

    assert model_row(latest) != model_row(saved)
    assert model_row(frozen).startswith(f"Disease · {renderers.FROZEN_BACKBONE_WORDS} ")
    assert "c" * 12 in model_row(frozen) and renderers.FROZEN_BACKBONE_WORDS not in model_row(
        latest)


def test_one_screen_says_the_models_fingerprint_once(site, monkeypatch):
    """The skill's §1.2 and §12, counted in one screen's visible text (Data,
    Run and Result with its reports table) for the report of the chosen
    model: the model's 12 hex digits once (its details, with its Copy; the
    table and the Result say "the selected model", the report's run id is
    its own identifier, once), the dataset fingerprint once (under the
    workspace), the version's once (its details; the Result says "the
    selected version"), the workspace's name never in the panels' text (its
    select's, and "the selected workspace" elsewhere, R10's row included),
    and no Run header repeating the selection."""
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    binding = tp.workspace_binding(site.demo, tp.find(page["workspaces"], "name", "demo"))
    reports = tp.read_reports(LABEL)
    chosen = tp.selection_of(page, models, binding, "demo", LABEL, site.key)
    version = tp.find(page["versions"], "label", LABEL)
    screen = " ".join([
        tp.workspace_html(page, "demo"), tp.version_html(page, LABEL, binding),
        tp.run_header_html(), tp.model_html(page, models, site.key),
        tp.evidence_html(page, "demo", version), tp.reports_header_html(reports),
        tp.reports_html(reports, site.run_id, models=models, selection=chosen),
        tp.result_html(reports, site.run_id, [], selection=chosen, models=models,
                       workspaces=page["workspaces"])])
    text = _plain(re.sub(r'<details class="[^"]*">.*?</details>', " ", screen, flags=re.S))
    # R-70, BACKLOG 51 (d): the report's run id, which names the model's 12
    # hex digits again, is one click away (it was the Result's Report row).
    assert text.count(site.run_id) == 0 and site.run_id in screen
    without_run_id = text
    assert without_run_id.count(site.sha[:12]) == 1, without_run_id
    # 51 (b): the version's role once, by its option (no Role row under it);
    # 51 (c): its counts and exclusions once, Data's, the Result of the
    # selected version saying its own count ("n of M ranked") alone.
    option = {value: words for words, value in tp.version_choices(page)}[LABEL]
    assert f"role {version['role']}" in option and "Role" not in text
    assert text.count("source cases") == 1 and text.count("excluded") == 1, text
    dataset = tp.find(page["workspaces"], "name", "demo")["dataset_fingerprint"]
    assert without_run_id.count(dataset[:12]) == 1
    assert without_run_id.count(version["manifest_sha256"][:12]) == 1
    # (The fixture's version role is "demo" too: its option's, not the panels'.)
    assert not re.search(r"\bdemo\b(?!-)", without_run_id), without_run_id
    assert tp.run_header_html() == '<div class="shep-panel-hd"><h2>Run</h2></div>'
    # With a test run of the chosen model on the card (review of B3): the
    # card says its own inputs, once each, a row each, drawn from the job
    # alone (the poll writes it without the selection, so it holds while
    # another model or version is chosen: the two can differ, the skill's
    # §1.2), so the model's and the version's 12 hex digits each appear once
    # more, never more.
    job = {"status": "running", "start_time": None, "log": [],
           "inputs": {"version_label": LABEL, "version_sha256": version["manifest_sha256"],
                      "checkpoint_sha256": site.sha, "workspace": str(site.demo),
                      "protocol": "disease", "record": True, "deterministic": False,
                      "device": "cpu", "version_kind": "supplied"}}
    card = _plain(re.sub(r'<details class="[^"]*">.*?</details>', " ",
                         tp.job_head_html(job) + tp.job_body_html(job, [], ("busy", "GPU job")),
                         flags=re.S))
    with_card = without_run_id + " " + card
    assert card.count(site.sha[:12]) == 1 and with_card.count(site.sha[:12]) == 2
    assert with_card.count(version["manifest_sha256"][:12]) == 2
    assert with_card.count(dataset[:12]) == 1
    # Its workspace is one Details › away, not in the screen's text.
    assert not re.search(r"(?<!Role )\bdemo\b(?!-)", card), card


# --------------------------------------------------------------------------- in a browser
#: The Test page drawn in a real browser (Node and Playwright): the page as
#: it opens (no run since the server started), then a run and its kept
#: report chosen, at the owner's window, two laptop windows and the stacked
#: layout: the job's idle line and Cancel, each run option's label and its ⓘ,
#: and the Result's tile values with their names as drawn and with the first
#: name made to wrap (`browser.LOOK_JS`'s `tops` and `textBox`).
_TEST_PAGE_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__, LOOK_JS = __LOOK_JS__;
(async () => {
  const [base, runWords] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 } })).newPage();
  await page.addInitScript(QUEUE_JS);
  await page.addInitScript(LOOK_JS);
  const idle = () => page.waitForFunction(() => window.__shepQueue && window.__shepQueue.idle(),
                                          null, { timeout: 60000 });
  const frames = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  await page.goto(base, { waitUntil: 'load' });
  await page.waitForSelector('#shep-chips .shep-chip', { timeout: 60000 });
  await idle();
  await page.click('#shep-tab-research-button'); await idle();
  await page.click('#shep-page-test-button');
  await page.waitForSelector('#shep-page-test-button[aria-selected="true"]');
  await idle();
  await page.waitForSelector('#shep-test-reports tr[data-key]', { timeout: 60000 });
  await frames();
  const out = {};
  out.idle = await page.evaluate(() => {
    const cancel = document.querySelector('#shep-test-cancel');
    return { card: document.querySelector('#shep-test-run').innerText.includes('Latest test run'),
             cancel: !!cancel && cancel.getClientRects().length > 0,
             head: document.querySelector('#shep-test-job-head').innerText.trim() };
  });
  await page.click('#shep-test-model-run input');
  await page.locator('[role=listbox] li[role=option]', { hasText: runWords }).first().click();
  await page.waitForSelector('[role=listbox]', { state: 'detached' }).catch(() => {});
  await idle();
  await page.waitForFunction(() => /Model fingerprint/.test(
    (document.querySelector('#shep-test-model-info') || {}).innerText || ''), null, { timeout: 60000 });
  await page.click('#shep-test-reports tr[data-key]');
  await idle();
  await page.waitForSelector('#shep-test-result-body .shep-rec-tile-v', { timeout: 60000 });
  out.sizes = {};
  for (const [w, h] of [[1850, 960], [1536, 864], [1440, 900], [1440, 1600]]) {
    await page.setViewportSize({ width: w, height: h });
    await frames();
    out.sizes[`${w}x${h}`] = await page.evaluate(async () => {
      const look = window.__shepLook, values = '#shep-test-result-body .shep-rec-tile-v';
      const options = {};
      for (const id of ['shep-test-record', 'shep-test-deterministic']) {
        const text = look.textBox(document.querySelector(`#${id} label > span`));
        const i = document.querySelector(`#${id} .shep-i`).getBoundingClientRect();
        options[id] = { text, info: { left: Math.round(i.left), top: Math.round(i.top) },
                        block: Math.round(document.getElementById(id).getBoundingClientRect().right) };
      }
      const first = document.querySelector('#shep-test-result-body .shep-rec-tile-l');
      const named = first.textContent;
      const tiles = look.tops(values);
      first.textContent = named + ' over a name long enough to take three lines in its tile,'
        + ' whatever the window and however narrow the type it is drawn in';
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const wrapped = { tops: look.tops(values), lines: look.textBox(first).lines };
      first.textContent = named;
      return { options, tiles, wrapped,
               names: [...document.querySelectorAll('#shep-test-result-body .shep-rec-tile-l')]
                 .map((e) => e.textContent) };
    });
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_the_run_options_the_idle_job_and_the_result_tiles_as_drawn(site, monkeypatch, tmp_path):
    """In a real browser (skipped without Node and Playwright), the page on
    the site's workspace for an operator:

    - R-81: with no run since the server started, no job card and no Cancel
      (a 479 x 95 bordered card and a disabled Cancel nothing could use), the
      foot's one muted line instead;
    - R-79: each run option's label on one line and its ⓘ right after its
      words ("Deterministic execution" wrapped to two lines at 1850 x 960 with
      its ⓘ at the block's right edge, 26 px from the words);
    - R-71, NEW-test-4: the task's tiles named without the task, which their
      label says once, and their values on one line at 1850, 1536 and 1440
      wide, also with a name that wraps to three lines (19-20 px apart at
      1536 x 864 and 1440 x 900); stacked, two to a row, each row's alike.
    Fails on 6db401b's page: the idle card and Cancel are drawn, the
    Deterministic label takes two lines, and a wrapped name drops its value."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import LOOK_JS, QUEUE_JS, node_with_playwright, run_node

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    operator_mode(monkeypatch)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    parents = tp.model_parents(models)
    run_words = {value: words for words, value in ml.run_choices(parents)}[
        ml.item_run_key(parents[site.key])]
    script = (_TEST_PAGE_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
              .replace("__LOOK_JS__", json.dumps(LOOK_JS)))
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(script, tmp_path, base[:-len("gradio_api")], run_words)
    assert seen["idle"] == {"card": False, "cancel": False, "head": tp.NO_JOB_YET}, seen["idle"]
    for size, facts in seen["sizes"].items():
        assert facts["names"] == ["Pooled MRR", "Pooled top-1", "Pooled top-10", "Coverage"]
        for option, drawn in facts["options"].items():
            text, info = drawn["text"], drawn["info"]
            assert text["lines"] == 1, (size, option, drawn)
            assert 0 <= info["left"] - text["right"] <= 12, (size, option, drawn)
            assert info["left"] < drawn["block"], (size, option, drawn)
        assert facts["wrapped"]["lines"] >= 2, (size, facts)
        for tops in (facts["tiles"], facts["wrapped"]["tops"]):
            rows = [tops] if size != "1440x1600" else [tops[:2], tops[2:]]
            for row in rows:
                assert max(row) - min(row) <= 2, (size, tops)


#: The job card as the page draws it (`#shep-test-job`, its own
#: ``js_on_load``), redrawn by the server while a run prints lines, as the
#: poll's `_job_card_now` redraws it: the reader opens its Details, lines
#: arrive, the reader closes it, a line arrives, the reader opens it again
#: and the run ends. Each redraw is waited for by the output it brings.
_JOB_FOLD_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const base = process.argv[2];
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 } })).newPage();
  await page.goto(base, { waitUntil: 'networkidle' });
  const card = '#shep-test-job';
  await page.waitForSelector(card + ' details > summary');
  const read = () => page.evaluate((c) => {
    const fold = document.querySelector(c + ' details');
    const log = document.querySelector(c + ' pre');
    return { open: fold.open, log: log ? log.textContent : '',
             shown: !!log && log.checkVisibility(),
             lead: document.querySelector(c).innerText.includes('Its report was kept') };
  }, card);
  const redrawn = (button, words) => page.click(button).then(() => page.waitForFunction(
    ([c, w]) => document.querySelector(c).textContent.includes(w), [card, words], { timeout: 20000 }));
  const toggle = () => page.click(card + ' details > summary');
  const out = {};
  out.drawn = await read();
  await toggle(); out.opened = await read();
  await redrawn('#line', 'scored batch 1'); out.line1 = await read();
  await redrawn('#line', 'scored batch 2'); out.line2 = await read();
  await toggle(); out.closed = await read();
  await redrawn('#line', 'scored batch 3'); out.line3 = await read();
  await toggle(); await redrawn('#end', 'Its report was kept'); out.ended = await read();
  await redrawn('#next', 'next job batch 0'); out.nextJob = await read();
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_the_job_cards_details_stay_as_the_reader_left_them_while_the_run_prints(tmp_path):
    """A running test's card is drawn again each time its output gains a line
    (the poll's `_job_card_now`); Gradio's redraw sets each element's
    attributes to the new HTML's, so the card's Details ›, which holds the
    output tail (B52-E), closed under the reader on each new line: opened,
    then one line later ``{open: false}``. In a real browser (skipped
    without Node and Playwright), the card with the page's own
    ``js_on_load``: opened, it stays open as lines arrive and its output
    shows the new line; closed by the reader, a line leaves it closed; opened
    again, it stays open when the run ends and its output folds under the
    lead. Fails on f6b9289's page, whose card has no script: the first line
    closed it. The card of the next job (another job id, the same "Details ›")
    is drawn closed: what was kept was this job's (T08-flow-2: the card keyed
    by its summary's words and place alone kept it open for the next job;
    the shared keeper names the job, `renderers.keep_as`)."""
    from tests.fixtures.browser import node_with_playwright, run_node

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    demo, _ctx, _registry = _page_component()
    (drawn,) = [b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "shep-test-job"]
    assert drawn.js_on_load == renderers.KEEP_OPEN_JS
    (result,) = [b for b in demo.blocks.values()
                 if getattr(b, "elem_id", None) == "shep-test-result-body"]
    # The result is drawn again when the poll sees a job start or end.
    assert result.js_on_load == renderers.KEEP_OPEN_JS
    job = {"job_id": "job-1", "status": "running",
           "inputs": {"version_label": LABEL, "device": "cpu"},
           "log": ["scored batch 0"], "elapsed_seconds": 5}

    def card():
        return tp.job_body_html(job, [], ("busy", "GPU job: test run"))

    def line():
        job["log"] = [*job["log"], f"scored batch {len(job['log'])}"]
        return card()

    def end():
        job.update(status="completed", exit_code=0, exit_meaning="measured",
                   outcome={"kept": True, "registered": False, "summary": "Kept."})
        return card()

    def next_job():
        job.clear()
        job.update({"job_id": "job-2", "status": "running", "elapsed_seconds": 1,
                    "inputs": {"version_label": LABEL, "device": "cpu"},
                    "log": ["next job batch 0"]})
        return card()

    with gr.Blocks() as page:
        shown = tp.plain_html(card(), elem_id="shep-test-job")
        gr.Button("Line", elem_id="line").click(line, outputs=[shown])
        gr.Button("End", elem_id="end").click(end, outputs=[shown])
        gr.Button("Next", elem_id="next").click(next_job, outputs=[shown])
    with served(page) as base:
        seen = run_node(_JOB_FOLD_JS, tmp_path, base[:-len("gradio_api")])
    assert seen["drawn"]["open"] is False and seen["opened"]["shown"] is True, seen
    for step in ("line1", "line2"):
        assert seen[step]["open"] is True and seen[step]["shown"] is True, (step, seen)
    assert seen["line2"]["log"].endswith("scored batch 2"), seen
    assert seen["closed"]["open"] is False and seen["line3"]["open"] is False, seen
    assert "scored batch 3" in seen["line3"]["log"], seen
    assert seen["ended"]["lead"] is True and seen["ended"]["open"] is True, seen
    assert seen["ended"]["shown"] is True and "scored batch 3" in seen["ended"]["log"], seen
    assert seen["nextJob"]["open"] is False and seen["nextJob"]["shown"] is False, seen
