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
- **D5**: a key whose runs disagree is shown "runs disagree — investigate",
  its reports cannot be registered, a note is kept with who, when and what and
  lifts nothing;
- **D7**: the preview shows exclusion reasons and their counts, never a case
  id; there is no upload control;
- **every message true**: the job's pill is never green for a failed or
  cancelled run; the outcome is the runner's (kept, registered, "no report was
  published"); *unverifiable* is a word, never a 0; no absolute server path is
  shown; values are escaped;
- **the page's wiring**: refreshed when the page is shown; polled on the
  shell's slow timer with ``queue=False``, doing work only while it is shown
  and never touching a select (no listener on the fast timer); the report of
  a run that has just ended selected; the import's preview and commit one at
  a time; field help registered with titles equal to the labels; its style
  sheet registered and never under 14 px; the batch size shown is the
  measurement's default.

Module: tests/unit/test_webui_test_page.py
"""
from __future__ import annotations

import json
import re
import shutil
from types import SimpleNamespace

import pytest

gr = pytest.importorskip("gradio")
pytest.importorskip("torch")

from src.webui.components import test_page as tp  # noqa: E402
from tests.fixtures.operator_access import (  # noqa: E402
    OPERATOR_BASE_URL,
    gradio_request,
    operator_mode,
    operator_request,
)

RUN = "20261017-091500_run_3fa9c1"
LABEL = "demo-test-1"


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A workspaces root with the demo workspace and its checkpoint in a run, a
    registry with one version holding one kept, unregistered report (the real
    measurement, on CPU: an untrained model, so no number here is a result),
    and an intake directory with one source."""
    import scripts.measure_scorer as measure_scorer
    from tests.fixtures.registered_cohort import build_registered_cohort

    base = tmp_path_factory.mktemp("testpage")
    cohort = build_registered_cohort(base / "c")
    workspaces = base / "ws"
    workspaces.mkdir()
    demo = workspaces / "demo"
    shutil.move(str(cohort.data_dir), demo)
    checkpoints = demo / "runs" / "gat" / RUN / "checkpoints"
    checkpoints.mkdir(parents=True)
    shutil.copy2(cohort.checkpoint, checkpoints / "best.pt")
    intake = base / "intake"
    intake.mkdir()
    shutil.copy2(cohort.source, intake / "cases.json")
    # The registry is the deployment's while the run measures (BACKLOG 37).
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SHEPHERD_REGISTRY", str(cohort.registry))
        code = measure_scorer.main(["--checkpoint", str(checkpoints / "best.pt"),
                                    "--data-dir", str(demo), "--cohort", LABEL, "--registry",
                                    str(cohort.registry), "--device", "cpu"])
    assert code == 0
    return SimpleNamespace(base=base, sha=cohort.checkpoint_sha256,
                           version_sha=cohort.version.identity.sha256)


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
    monkeypatch.delenv("SHEPHERD_ORDERING_ROLES", raising=False)
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    monkeypatch.setattr(served_status, "read_status", lambda: (None, "not read"))
    reports = registry / "versions" / LABEL / "reports"
    (run_dir,) = [p for p in reports.iterdir() if p.is_dir()]
    return SimpleNamespace(
        root=root, ws=root / "ws", demo=root / "ws" / "demo", registry=registry,
        intake=root / "intake", reports=reports, run_id=run_dir.name, sha=built.sha,
        version_sha=built.version_sha,
        checkpoint=root / "ws" / "demo" / "runs" / "gat" / RUN / "checkpoints" / "best.pt",
        key=f"gat/{RUN}/best.pt")


def _readonly():
    return gradio_request(server=("192.0.2.10", 8264))


def _page(monkeypatch, *, operator=True):
    request = operator_request(monkeypatch) if operator else _readonly()
    return tp.read_page(request), request


def _models(page, request):
    return tp.read_models("demo", request)


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
    report["authoritative_metrics"]["untruncated_mrr"] += delta
    (copy / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return copy.name


# --------------------------------------------------------------------------- pure pieces
def test_no_absolute_path_is_shown():
    roots = [("/srv/registry", "<registry>"), ("/srv/ws", "<workspaces>")]
    text = ("A report was kept at /srv/registry/versions/x/reports/r1; ledger /srv/ws/demo/"
            "runs/gat/r/checkpoints/evaluations.json; other /etc/passwd and C:\\Users\\x; "
            "/srv/ws2/elsewhere stays out; 'at /srv/registry'.")
    out = tp.redact(text, sorted(roots, key=lambda p: -len(p[0])))
    assert "<registry>/versions/x/reports/r1" in out
    assert "<workspaces>/demo/runs/gat/r/checkpoints/evaluations.json" in out
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

    args = measure_scorer.parse_args(["--checkpoint", "c", "--data-dir", "d", "--cohort", "x",
                                      "--registry", "/r"])
    assert args.batch_size == tp.BATCH_SIZE_SHOWN


def test_a_cancelled_runs_output_is_folded_and_a_failed_runs_shown():
    job = {"status": "cancelled", "inputs": {}, "log": ["Traceback", "KeyboardInterrupt"]}
    html = tp.job_html(job, [], ("none", ""))
    assert '<details class="shep-test-fold">' in html and "KeyboardInterrupt" in html
    failed = tp.job_html(dict(job, status="failed"), [], ("none", ""))
    assert "<details" not in failed and "KeyboardInterrupt" in failed


def test_the_pill_is_never_green_for_a_failed_or_cancelled_run():
    for status, kind in (("completed", "ok"), ("failed", "red"), ("cancelled", "grey"),
                         ("running", "accent"), ("idle", "grey")):
        html = tp.job_html({"status": status, "inputs": {}}, [], ("none", ""))
        assert f'shep-tag-{kind} shep-test-pill" data-status="{status}"' in html


def test_the_outcome_is_the_runners_kept_and_registered():
    kept = tp.outcome_html({"kept": True, "registered": False, "summary": "A report was kept."},
                           [])
    assert "Report kept" in kept and "Not registered" in kept
    none = tp.outcome_html({"kept": False, "registered": False,
                            "summary": "No report was published: the run wrote no outcome file"},
                           [])
    assert "No report kept" in none and "Registered" not in none and "No report was published" in none
    unknown = tp.outcome_html({"kept": None, "registered": None, "summary": ""}, [])
    assert "Kept: cannot be checked" in unknown and "Registered: cannot be checked" in unknown


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
    assert "No report kept" in html and "workspace demo" in html


def test_a_role_says_whether_it_offers_ordering(monkeypatch):
    monkeypatch.setenv("SHEPHERD_ORDERING_ROLES", "research_selection,institutional_acceptance")
    assert tp.role_line("research_selection").startswith(
        "Role research_selection offers ordering by test score in Models")
    assert "never offers ordering" in tp.role_line("institutional_acceptance")
    assert "does not offer ordering" in tp.role_line("mygene2")
    assert tp.role_line("").startswith("Choose the version's role")


def test_unverifiable_is_a_word_never_a_count():
    r10 = {"training": {"evidence": {"state": "found", "path": "/srv/secret/train.json"},
                        "disease_overlap": {"state": "overlap", "k": 3, "n": 5, "cases": 7},
                        "phenotype_set_overlap": {"state": "none", "k": 0, "n": 10}},
           "validation": {"evidence": {"state": "not_found"},
                          "disease_overlap": {"state": "unverifiable", "k": None, "n": 5,
                                              "cases": None},
                          "phenotype_set_overlap": {"state": "unverifiable", "k": None, "n": 10}}}
    lines = dict(tp._r10_lines(r10))
    validation = lines["R10, validation"]
    assert "<b>unverifiable</b>" in validation and " of " not in validation
    assert "3 of 5 distinct test diseases; 7 test cases" in lines["R10, training"]
    assert "0 of 10 test cases" in lines["R10, training"]
    assert "/srv/secret" not in "".join(lines.values())


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
    assert "operator control" in tp.model_html(page, models, None)
    reports = tp.read_reports(LABEL)
    assert [r["run_id"] for r in reports["runs"]] == [site.run_id]
    allowed, why = tp.start_state(page, models, "demo", LABEL, None, None, ("none", ""))
    assert allowed is False and why == "Operator controls only: request not local."


def test_an_operator_lists_the_workspaces_checkpoints(site, monkeypatch):
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    assert [tp.item_key(i) for i in models["items"]] == [site.key]
    assert models["items"][0]["sha256"] == site.sha
    (label, key), = tp.model_choices(models)
    assert key == site.key and f"model fingerprint {site.sha[:12]}" in label


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
    assert run["state"]["checked_beside"] == f"gat/{RUN}/best.pt"
    readonly = tp.read_page(_readonly())
    assert tp.refine_registration(readonly, reports, site.run_id, models) is reports
    # A ledger that cannot be read again leaves the list's answer, never a page error.
    import src.evaluation.registration as registration

    def unreadable(*_a, **_k):
        raise OSError("ledger unreadable")

    monkeypatch.setattr(registration, "report_state", unreadable)
    assert tp.refine_registration(page, reports, site.run_id, models) is reports


def test_a_disagreeing_key_is_shown_runs_disagree_and_cannot_be_registered(site, monkeypatch):
    other = _disagreeing_copy(site)
    page, _request = _page(monkeypatch)
    reports = tp.read_reports(LABEL)
    runs = {r["run_id"]: r for r in reports["runs"]}
    assert runs[other]["contest"]["state"] == "contested"
    assert "untruncated_mrr" in runs[other]["contest"]["differing"]
    html = tp.reports_html(reports, other)
    assert "runs disagree — investigate" in html
    allowed, why = tp.register_state(page, reports, site.run_id)
    assert allowed is False and why.startswith("Disabled: runs disagree — investigate.")
    result = tp.result_html(reports, site.run_id, [], notes=[])
    assert "Runs disagree — investigate." in result and "does not lift the block" in result
    assert tp.notes_visible(reports, site.run_id)


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
    store as the open route lists it, by run id."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routes import lists

    app = FastAPI()
    app.include_router(lists.router, prefix="/api/v1")
    body = TestClient(app).get("/api/v1/cohorts").json()
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
        assert shown["state"] == listed["state"] == expected, (run_id, shown, listed)
        assert shown["differing"] == listed["differing"]
        assert shown["disagreeing_runs"] == listed["disagreeing_runs"]
        assert listed["problems"] == []
        assert bool(shown["problems"]) == (expected == "cannot_be_checked")
    assert str(site.root) not in json.dumps([run["contest"] for run in route.values()])
    if case == "contested":
        assert "untruncated_mrr" in route[added]["contest"]["differing"]
        assert sorted(route[added]["contest"]["disagreeing_runs"]) == sorted(with_report)
    if case == "a report whose key cannot be told":
        assert page[added]["key"] is None and "could not be read" in page[added]["unbuilt"]
        assert tp.register_state({"access": {"allowed": True}}, tp.read_reports(LABEL),
                                 site.run_id)[0] is False
    if case == "a report that cannot be compared":
        assert page[added]["key"] == page[site.run_id]["key"]
        assert "cannot be compared" in page[added]["unbuilt"]
        assert page[added]["record"] is None and page[site.run_id]["record"] is not None


def test_the_result_shows_aggregates_and_states_never_a_case_or_a_path(site):
    reports = tp.read_reports(LABEL)
    html = tp.result_html(reports, site.run_id, tp.path_roots())
    assert "Mode C MRR (untruncated)" in html and "over 10 ranked" in html
    assert "10 of 13</b> source cases ranked" in html
    assert "ground truth absent 2 · phenotype unmappable 1" in html
    assert "Validation MRR" not in html, "Trainer validation and Mode C never share a place"
    assert "case-" not in html and str(site.root) not in html
    assert "synthetic</b> — a generated stand-in" in html
    assert "R1 graph files match the workspace manifest" in html


def test_a_run_without_a_report_says_none_was_kept(site):
    interrupted = site.reports / f"20261020T190000Z-{site.sha[:12]}-0000dead"
    interrupted.mkdir()
    reports = tp.read_reports(LABEL)
    run = tp.find(reports["runs"], "run_id", interrupted.name)
    assert run["state"]["kept"] == "not_kept"
    assert "no report kept" in tp.reports_html(reports, None)
    assert "No report was kept in this run directory" in tp.result_html(reports, interrupted.name, [])
    assert tp.register_state({"access": {"allowed": True}}, reports, interrupted.name) == (
        False, "Disabled: no report was kept in this run.")


def test_start_waits_for_every_choice_and_says_why(site, monkeypatch):
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    binding = tp.workspace_binding(site.demo, tp.find(page["workspaces"], "name", "demo"))
    state = lambda ws, label, key, gpu=("none", ""): tp.start_state(  # noqa: E731
        page, models, ws, label, key, binding, gpu)
    assert state(None, LABEL, site.key) == (False, "Disabled: choose a workspace.")
    assert state("demo", None, site.key) == (False, "Disabled: choose a cohort version.")
    assert state("demo", LABEL, None) == (False, "Disabled: choose a model.")
    assert state("demo", LABEL, site.key) == (True, None)
    busy = state("demo", LABEL, site.key, ("busy", "GPU job: training (job runner)"))
    assert busy == (False, "Disabled: GPU job: training (job runner). One managed GPU job at a "
                           "time (training or evaluation).")
    another = dict(binding, graph={**binding["graph"], "kg": "0" * 64})
    allowed, why = tp.start_state(page, models, "demo", LABEL, site.key, another, ("none", ""))
    assert allowed is False and "mapped against another graph" in why


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
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    manager = FakeManager()
    result = tp.start_test_run(page, models, "demo", LABEL, site.key, True, True, request,
                               manager=manager)
    assert result["success"] is True
    (inputs,) = manager.started
    assert inputs.checkpoint_sha256 == site.sha and inputs.version_sha256 == site.version_sha
    assert inputs.registry == site.registry.resolve() and inputs.version_label == LABEL
    assert inputs.workspace == site.demo.resolve()
    assert inputs.checkpoint.resolve() == site.checkpoint.resolve()
    assert inputs.evidence_dirs == (evidence.resolve(),)
    assert inputs.record is True and inputs.deterministic is True


def test_start_runs_only_what_the_session_was_shown(site, monkeypatch):
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    manager = FakeManager()
    # a key that is not in the session's list, a path, and another workspace
    for ws, key in (("demo", "gat/x/y.pt"), ("demo", str(site.checkpoint)), ("other", site.key)):
        result = tp.start_test_run(page, models, ws, LABEL, key, False, False, request,
                                   manager=manager)
        assert result["success"] is False and "Nothing was started." in result["error"]
    result = tp.start_test_run(page, models, "demo", "not-a-version", site.key, False, False,
                               request, manager=manager)
    assert result["success"] is False and manager.started == []
    # A listed workspace holding the same run and file: the model was listed
    # for "demo", so it is not started in "copy" (where it was never shown).
    shutil.copytree(site.demo, site.ws / "copy", symlinks=True)
    page = tp.read_page(request)
    assert "copy" in page["directories"]
    result = tp.start_test_run(page, models, "copy", LABEL, site.key, False, False, request,
                               manager=manager)
    assert result["success"] is False and "in this workspace" in result["error"]
    assert manager.started == []


def test_a_checkpoint_the_server_no_longer_lists_is_refused_at_start(site, monkeypatch):
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    site.checkpoint.unlink()
    manager = FakeManager()
    result = tp.start_test_run(page, models, "demo", LABEL, site.key, False, False, request,
                               manager=manager)
    assert result["success"] is False and manager.started == []
    assert result["error"].endswith("Nothing was started.")


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


def test_register_is_refused_for_a_read_only_request(site, monkeypatch):
    from src.evaluation import registration

    monkeypatch.setattr(registration, "register_report",
                        lambda *a, **k: pytest.fail("a read-only request registered"))
    page = tp.read_page(_readonly())
    done, said = tp.register_selected(page, tp.read_reports(LABEL), {}, "demo", site.run_id,
                                      _readonly())
    assert done is False and said.startswith("Refused: ")
    assert not (site.checkpoint.parent / "evaluations.json").exists()


def test_a_report_that_changed_since_it_was_shown_is_not_registered(site, monkeypatch):
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    reports = tp.read_reports(LABEL)
    path = site.reports / site.run_id / "report.json"
    path.write_text(path.read_text() + "\n")
    done, said = tp.register_selected(page, reports, models, "demo", site.run_id, request)
    assert done is False and "changed since it was shown" in said
    assert not (site.checkpoint.parent / "evaluations.json").exists()


def _usage_lines(site):
    usage = site.reports / "usage.jsonl"
    return usage.read_bytes() if usage.exists() else b""


def _replaced_after_the_check(site, monkeypatch, replacement: bytes, how: str):
    """Replace the shown report's bytes right after the page's own comparison,
    when the ledger's checkpoint has been found (`ledger_checkpoint` returns),
    in place or by an atomic rename: the window between the page's read and the
    registration's (the M5 review's P2-1)."""
    import os

    path = site.reports / site.run_id / "report.json"
    real = tp.ledger_checkpoint

    def found_then_replaced(*args, **kwargs):
        place = real(*args, **kwargs)
        if how == "overwrite":
            path.write_bytes(replacement)
        else:
            staged = path.with_name("report.json.staged")
            staged.write_bytes(replacement)
            os.replace(staged, path)
        return place

    monkeypatch.setattr(tp, "ledger_checkpoint", found_then_replaced)


@pytest.mark.parametrize("how", ["overwrite", "rename"])
def test_a_valid_report_swapped_in_after_the_check_is_refused_by_the_registration(
    site, monkeypatch, how
):
    """The reviewer's counter-example (P2-1): another valid report of the same
    checkpoint and version, its MRR moved by 0.0004, swapped in after the page
    compared the shown digest. The registration compares that digest again on
    the one read it parses (`expected_report_sha256`) and refuses: no ledger,
    and the version's usage history unchanged."""
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    reports = tp.read_reports(LABEL)
    path = site.reports / site.run_id / "report.json"
    replacement = json.loads(path.read_bytes())
    replacement["authoritative_metrics"]["untruncated_mrr"] += 0.0004
    usage_before = _usage_lines(site)
    _replaced_after_the_check(site, monkeypatch,
                              json.dumps(replacement, indent=2, sort_keys=True).encode(), how)

    done, said = tp.register_selected(page, reports, models, "demo", site.run_id, request)

    assert done is False
    assert f"{path} is not the report that was shown" in said
    assert "select it again" in said and "Nothing was written to " in said
    assert not (site.checkpoint.parent / "evaluations.json").exists()
    assert _usage_lines(site) == usage_before


def test_damaged_bytes_swapped_in_after_the_check_are_refused_before_they_are_parsed(
    site, monkeypatch
):
    """The digest refuses them, never the parser: the message is the shown
    report's, not "not a measurement report"."""
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    reports = tp.read_reports(LABEL)
    usage_before = _usage_lines(site)
    _replaced_after_the_check(site, monkeypatch, b"{not json", "overwrite")

    done, said = tp.register_selected(page, reports, models, "demo", site.run_id, request)

    assert done is False and "is not the report that was shown" in said
    assert "not a measurement report" not in said
    assert not (site.checkpoint.parent / "evaluations.json").exists()
    assert _usage_lines(site) == usage_before


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
    done, said = tp.register_selected(page, reports, models, "demo", site.run_id, request)
    assert done is False and "No checkpoint the server lists (demo) holds the weights" in said
    assert "Nothing was registered." in said


def test_a_disagreeing_keys_report_is_refused_by_the_one_registration(site, monkeypatch):
    _disagreeing_copy(site)
    page, request = _page(monkeypatch)
    models = tp.read_models("demo", request)
    done, said = tp.register_selected(page, tp.read_reports(LABEL), models, "demo", site.run_id,
                                      request)
    assert done is False and "runs disagree — investigate" in said
    assert not (site.checkpoint.parent / "evaluations.json").exists()


def test_a_note_is_kept_with_the_key_and_lifts_nothing(site, monkeypatch):
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
    assert tp.register_state(page, reports, site.run_id)[0] is False, "a note lifts nothing"


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
    selects = [b for b in poll.outputs if isinstance(b, (gr.Dropdown, gr.Checkbox))]
    assert selects == [], "a poll never touches a select"
    monkeypatch.setattr(tp, "read_job", lambda **k: pytest.fail("polled while not shown"))
    quiet = poll.fn("clinical", {"access": {}}, {}, None, "demo", LABEL, None,
                    {"status": "running"}, None, {}, None)
    assert all(update == gr.update() for update in quiet)


def test_no_listener_of_the_page_sends_a_request_on_the_fast_timer():
    """A shell timer's tick is a request per listener whatever it does; the
    page polls on the slow timer only."""
    demo, ctx, _registry = _page_component()
    assert not [fn for fn in demo.fns.values() if (ctx.timer_fast._id, "tick") in fn.targets]


def test_a_poll_reads_the_store_again_only_when_the_job_changed(site, monkeypatch):
    demo, ctx, _registry = _page_component()
    poll = _named(demo, "_poll")
    manager = FakeManager({"status": "completed", "job_id": "test-1", "inputs": {},
                           "outcome": None})
    monkeypatch.setattr(tp, "read_job", lambda **k: {**manager.get_test_status(), "log": []})
    monkeypatch.setattr(tp, "gpu_line", lambda **k: ("none", "GPU jobs: none"))
    reads = []
    monkeypatch.setattr(tp, "read_reports", lambda label, **k: reads.append(label) or
                        {"label": label, "runs": [], "problem": None, "notes": []})
    page = tp.read_page(_readonly())
    unchanged = poll.fn("test", page, {}, None, "demo", LABEL, None,
                        {"job_id": "test-1", "status": "completed"}, None, {"label": LABEL},
                        _readonly())
    assert reads == [] and unchanged[0] == gr.update()
    ended = poll.fn("test", page, {}, None, "demo", LABEL, None,
                    {"job_id": "test-1", "status": "running"}, None, {"label": LABEL},
                    _readonly())
    assert reads == [LABEL] and ended[2] == {"job_id": "test-1", "status": "completed",
                                             "gpu": ["none", "GPU jobs: none"]}


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
    same = poll.fn("test", page, {}, None, "demo", LABEL, None, seen, None, {"label": LABEL},
                   _readonly())
    assert same[-1] == gr.update() and read == []
    line[:] = ["busy", "GPU job: training (job runner)"]
    moved = poll.fn("test", page, {}, None, "demo", LABEL, None, seen, None, {"label": LABEL},
                    _readonly())
    assert moved[-1] == "<chips>" and read == [1]
    assert moved[2]["gpu"] == ["busy", "GPU job: training (job runner)"]


def test_the_report_of_a_run_that_has_just_ended_is_selected(site, monkeypatch):
    """A run this session saw running ends with a report: the poll selects it,
    even over the report the operator had chosen; once seen ended, the
    choice is the operator's again."""
    other = _disagreeing_copy(site)
    demo, ctx, _registry = _page_component()
    poll = _named(demo, "_poll")
    job = {"status": "completed", "job_id": "test-2", "inputs": {}, "log": [],
           "outcome": {"report_dir": str(site.reports / site.run_id), "kept": True,
                       "registered": False, "summary": "kept"}}
    monkeypatch.setattr(tp, "read_job", lambda **k: dict(job))
    monkeypatch.setattr(tp, "gpu_line", lambda **k: ("none", "GPU jobs: none"))
    page = tp.read_page(_readonly())
    reports = tp.read_reports(LABEL)
    ended = poll.fn("test", page, {}, None, "demo", LABEL, None,
                    {"job_id": "test-2", "status": "running"}, other, reports, _readonly())
    assert ended[1] == site.run_id
    seen = poll.fn("test", page, {}, None, "demo", LABEL, None,
                   {"job_id": "test-2", "status": "completed"}, other, reports, _readonly())
    assert seen[1] == gr.update(), "the operator's choice stands once the end was seen"


def test_a_poll_that_sees_the_job_change_chooses_again_when_the_shown_report_is_gone(site,
                                                                                    monkeypatch):
    """A run started elsewhere changes the job: the store is read again, and a
    report this session showed that is gone from it gives way to the newest."""
    demo, ctx, _registry = _page_component()
    poll = _named(demo, "_poll")
    job = {"status": "running", "job_id": "test-3", "inputs": {}, "log": [], "outcome": None}
    monkeypatch.setattr(tp, "read_job", lambda **k: dict(job))
    monkeypatch.setattr(tp, "gpu_line", lambda **k: ("busy", "GPU job: test run (job runner)"))
    page = tp.read_page(_readonly())
    gone = "20261001T000000Z-" + "a" * 12 + "-" + "b" * 8
    out = poll.fn("test", page, {}, None, "demo", LABEL, None,
                  {"job_id": "test-2", "status": "completed"}, gone, {"label": LABEL},
                  _readonly())
    assert out[1] == site.run_id


def test_the_import_is_one_at_a_time_and_there_is_no_upload():
    demo, _ctx, _registry = _page_component()
    preview, commit = _named(demo, "_on_preview"), _named(demo, "_on_commit")
    assert preview.concurrency_id == commit.concurrency_id == "shep-test-import"
    assert preview.concurrency_limit == commit.concurrency_limit == 1
    kinds = {type(b).__name__ for b in demo.blocks.values()}
    assert not kinds & {"File", "UploadButton", "Files"}


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
    selectors = [sel.strip() for head in re.findall(r"([^{}]+)\{", tp.PAGE_CSS)
                 if not head.strip().startswith("@") for sel in head.split(",")]
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

    event, refused = call("http://192.0.2.10", [None, None, None])
    assert event == "event: complete" and "Refused: " in refused
    event, loopback = call(OPERATOR_BASE_URL, [None, None, None])
    assert event == "event: complete" and "Choose a cohort version the server lists" in loopback
    assert given and all(page == {} and models.get("items") == [] for page, models in given), \
        "the handler was given what the client posted"
    event, forged = call(OPERATOR_BASE_URL, ["demo", LABEL, site.key])
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
    """The measurement refuses a sample by its patient id; the job card shows
    the refusal without it (D7)."""
    from src.evaluation.measurement import _assert_ids_in_range

    with pytest.raises(ValueError) as refused:
        _assert_ids_in_range([SimpleNamespace(patient_id="MRN-9", phenotype_ids=[],
                                              disease_id=0)], 10, 10)
    text = f"ValueError: {refused.value}"
    assert "MRN-9" in text and "MRN-9" not in tp.case_free(text)
    job = {"status": "failed", "inputs": {}, "error_message": text, "log": ["Traceback", text]}
    html = tp.job_html(job, [], ("none", ""))
    assert "MRN-9" not in html and "a sample has no phenotypes" in html


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
    out = _named(demo, "_on_report").fn(page, reports, {}, {},
                                        SimpleNamespace(_data={"key": other}), request)
    html = out[3]
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
    assert state["checked_beside"] == f"gat/{RUN}/best.pt in workspace demo"
    done, said = tp.register_selected(page, refined, models_copy, "copy", site.run_id, request)
    assert done is True and said.startswith("Registered as record 1.")
    assert demo_ledger.exists() and not copy_ledger.exists()
    reports = tp.read_reports(LABEL)
    assert tp.find(reports["runs"], "run_id", site.run_id)["state"]["registered"] == "registered"
    assert tp.register_state(page, reports, site.run_id) == (False, "Already registered.")


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
    assert state["registered"] == "not_registered" and state["checked_beside"] == f"gat/{RUN}/best.pt"
    html = tp.reports_html(rows, None, models=models)
    assert "report kept · not registered" in html and "cannot be checked from the list" not in html
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
        assert "started 02:06:07 UTC" in tp.job_body_html(job, [], ("none", ""))
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
        assert allowed is False and why.startswith("Disabled: GPU job: test run (job runner).")
    allowed, why = tp.start_state(page, models, "demo", LABEL, None, None, ("unread", ""))
    assert why.startswith("Disabled: the GPU lock cannot be read")


def _runs(n):
    return [{"run_id": f"202610{20 - i // 24:02d}T{23 - i % 24:02d}0000Z-{'a' * 12}-{i:08x}",
             "when": f"run {i}", "sha12": "a" * 12, "state": {"kept": "kept",
                                                            "registered": "not_registered"}}
            for i in range(n)]


def test_a_long_report_list_shows_the_newest_and_keeps_the_selected():
    runs = _runs(20)
    reports = {"label": LABEL, "runs": runs}
    html = tp.reports_html(reports, runs[15]["run_id"])
    old = re.findall(r'class="shep-test-rep[^"]*shep-test-rep-old"[^>]*data-key="([^"]+)"', html)
    assert old == [x["run_id"] for i, x in enumerate(runs) if i >= tp.ROWS_SHOWN and i != 15]
    assert f'data-key="{runs[15]["run_id"]}"' in html
    assert "Show all 20 reports ›" in html and f"Show the newest {tp.ROWS_SHOWN} only" in html
    short = tp.reports_html({"label": LABEL, "runs": runs[:tp.ROWS_SHOWN]}, None)
    assert "shep-test-rep-old" not in short and "Show all" not in short


def test_a_run_that_ended_with_no_report_is_a_row_until_the_next():
    cancelled = {"status": "cancelled", "start_time": "2026-10-21T02:00:00",
                 "end_time": "2026-10-21T02:01:00",
                 "inputs": {"version_label": LABEL, "checkpoint_sha256": "b" * 64},
                 "outcome": {"report_dir": None, "kept": False, "registered": False}}
    html = tp.reports_html({"label": LABEL, "runs": []}, None, job=cancelled)
    assert 'data-status="cancelled"' in html and "interrupted: cancelled" in html
    assert "no report kept" in html and "No run has kept a report" not in html
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
    label = html[html.index('class="shep-test-label"'):]
    label = label[:label.index("</div>")]
    assert "Help: R10 labels" not in label and "Help: Mode C result" in label
    r10 = html[html.index(">R10, training"):]
    assert r10.index("Help: R10 labels") < r10.index('class="shep-v"')


def test_start_and_the_job_sit_in_the_run_panels_foot():
    """Start and the job card are one region after the choices: at wide sizes
    it keeps to the panel's foot (the choices scroll) and the job's output
    scrolls inside its card under the title row with Cancel."""
    demo, _ctx, _registry = _page_component()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    foot = [getattr(c, "elem_id", None) for c in by_id["shep-test-runft"].children]
    assert foot == ["shep-test-start", "shep-test-start-reason", "shep-test-action",
                    "shep-test-jobcard"]
    wide, narrow = tp.PAGE_CSS.split("@media (max-width: 1399px)")
    wide = wide[wide.index("@media (min-width: 1400px)"):]
    assert "#shep-test-runft {" in wide and "#shep-test-jobcard > .shep-test-job-top { position: sticky" in wide
    # The region never shrinks below its content (the choices shrink and
    # scroll); past its cap the job card scrolls inside.
    assert "#shep-test-runft { flex: 0 0 auto !important;" in wide and "max-height: 64%" in wide
    assert "{ position: fixed !important;" in narrow.split(".shep-test-drawer ", 1)[1]
    assert re.search(r"\.shep-test-drawer > \.shep-test-bd \{[^}]*overflow: auto", narrow)
    # Over the panels, Gradio's own field labels included (position: relative;
    # z-index: 40 in 6.30), which would otherwise show through the drawer.
    for css in (wide, narrow):
        z = re.search(r"\.shep-test-drawer \{[^}]*z-index: (\d+)", css)
        assert z and int(z.group(1)) > 40, css[:40]


def test_add_note_sits_at_the_result_panels_foot_outside_the_scrolling_report():
    """While runs disagree, "Add note" is the one action left on the Result
    panel (Register is off). At wide sizes the report scrolls inside its body;
    the note form is a region of its own between that body and Register's
    footer, so Add note is in view without scrolling the report."""
    demo, _ctx, _registry = _page_component()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    panel = [getattr(c, "elem_id", None) or c.elem_classes[0]
             for c in by_id["shep-test-result"].children]
    assert panel == ["shep-test-bd", "shep-test-noteform", "shep-test-note-out", "shep-panel-ft"]
    body = by_id["shep-test-result"].children[0]
    assert [getattr(c, "elem_id", None) for c in body.children] == ["shep-test-result-body"]

    def inside(block, elem_id):
        return any(getattr(c, "elem_id", None) == elem_id or inside(c, elem_id)
                   for c in getattr(block, "children", []))

    for field in ("shep-test-note-who", "shep-test-note-what", "shep-test-note-add",
                  "shep-test-note-reason"):
        assert inside(by_id["shep-test-noteform"], field), field
    wide = tp.PAGE_CSS.split("@media (max-width: 1399px)")[0]
    wide = wide[wide.index("@media (min-width: 1400px)"):]
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
    for elem_id in ("shep-test-workspace", "shep-test-version", "shep-test-model",
                    "shep-test-intake", "shep-test-role", "shep-test-partial"):
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
    assert not by_id["shep-test-cohort-hd"].js_on_load
    (label, _key), = tp.model_choices({"items": [{
        "architecture": "gat", "run": RUN, "filename": "best.pt", "epoch": 12, "sha256": "c" * 64}]})
    assert label.startswith("epoch 12 · GAT · run 2026-10-17 09:15") and label.endswith("· best.pt")


def test_a_read_only_request_cannot_type_a_note(site, monkeypatch):
    _disagreeing_copy(site)
    demo, _ctx, _registry = _page_component()
    monkeypatch.setattr(tp, "read_job", lambda **k: {"status": "idle", "inputs": None,
                                                    "outcome": None, "log": []})
    page = tp.read_page(_readonly())
    reports = tp.read_reports(LABEL)
    out = _named(demo, "_on_report").fn(page, reports, {}, {},
                                        SimpleNamespace(_data={"key": site.run_id}), _readonly())
    note_btn, note_who, note_what, note_reason = out[-4:]
    assert note_who == gr.update(interactive=False) and note_what == gr.update(interactive=False)
    assert note_btn == gr.update(interactive=False) and "Operator controls only" in note_reason


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
    out = _named(demo, "_on_report").fn(page, reports, models_copy, {},
                                        SimpleNamespace(_data={"key": site.run_id}), request)
    kept = tp.find(out[1]["runs"], "run_id", site.run_id)["state"]
    assert kept["registered"] == "not_registered"
    assert "report kept · not registered" in tp.reports_html(out[1], None, models=models_copy)


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
        out = _named(demo, "_on_report").fn(page, reports, models_copy, {},
                                            SimpleNamespace(_data={"key": site.run_id}), asking)
        assert listed == []
        assert tp.find(out[1]["runs"], "run_id", site.run_id)["state"]["registered"] == \
            "cannot_be_checked"
    _named(demo, "_on_report").fn(page, reports, models_copy, {},
                                  SimpleNamespace(_data={"key": site.run_id}), request)
    assert listed == ["demo"]
    # No listed file holds the weights: the list's answer stands, and the
    # choice still lists the other workspace once, not once per output built.
    listed.clear()
    monkeypatch.setattr(server_lists, "list_checkpoints",
                        lambda name, **k: listed.append(name) or server_lists.Listing((), None))
    out = _named(demo, "_on_report").fn(page, reports, dict(models_copy, items=[]), {},
                                        SimpleNamespace(_data={"key": site.run_id}), request)
    assert listed == ["demo"], "listed once for the choice, never again for the same pass"
    assert tp.find(out[1]["runs"], "run_id", site.run_id)["state"]["registered"] == \
        "cannot_be_checked"


def test_the_ledger_order_is_the_file_then_the_workspace_the_run_read(site, monkeypatch):
    """`ledger_checkpoint`: (1) the file the run read when it is listed, even
    when another listed copy comes first in its workspace; (2) else a copy in
    the workspace the run measured, before (3) the workspace the page shows."""
    _point_report_at_site(site)
    manifest = json.loads((site.reports / site.run_id / "report.json").read_text())["manifest"]
    newer = site.demo / "runs" / "gat" / "20261019-080000_run_aaaaaa" / "checkpoints"
    newer.mkdir(parents=True)
    shutil.copy2(site.checkpoint, newer / "best.pt")
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


def test_the_words_after_a_path_stay_and_two_paths_are_two():
    roots = [("/srv/registry", "<registry>")]
    assert tp.redact("/x/model.pt in 3.2 s", roots) == f"{tp.SERVER_PATH} in 3.2 s"
    assert tp.redact("/etc/passwd and C:\\Users\\x end", roots) == (
        f"{tp.SERVER_PATH} and {tp.SERVER_PATH} end")
    assert tp.redact("see /srv/registry/versions/a b/r.json now", roots) == (
        "see <registry>/versions/a b/r.json now")


def test_the_test_page_and_datasets_share_one_redaction():
    """M5-3 review ARCH-7: one pattern and one implementation for "no absolute
    server path is shown" (they had already drifted apart)."""
    from src.webui.components import workspaces

    assert not hasattr(tp, "_ABSOLUTE")
    text = "x /mnt/a b/c d"
    assert tp.redact(text, []) == workspaces.redact_paths(text, [], other=tp.SERVER_PATH)
    assert workspaces.redact(text, None) == workspaces.redact_paths(
        text, [], other=workspaces.OUTSIDE_PATH)



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
