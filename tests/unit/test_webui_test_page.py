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
    code = measure_scorer.main(["--checkpoint", str(checkpoints / "best.pt"), "--data-dir",
                                str(demo), "--cohort", LABEL, "--registry",
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
    assert [tp.model_key(i) for i in models["items"]] == [site.key]
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
    assert not (site.registry / "versions" / "demo-test-2").exists()


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
                    {"status": "running"}, None, {})
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
                        {"job_id": "test-1", "status": "completed"}, None, {"label": LABEL})
    assert reads == [] and unchanged[0] == gr.update()
    ended = poll.fn("test", page, {}, None, "demo", LABEL, None,
                    {"job_id": "test-1", "status": "running"}, None, {"label": LABEL})
    assert reads == [LABEL] and ended[2] == {"job_id": "test-1", "status": "completed"}


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
                    {"job_id": "test-2", "status": "running"}, other, reports)
    assert ended[1] == site.run_id
    seen = poll.fn("test", page, {}, None, "demo", LABEL, None,
                   {"job_id": "test-2", "status": "completed"}, other, reports)
    assert seen[1] == gr.update(), "the operator's choice stands once the end was seen"


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
    from are this session's state, never what the client posts (a forged page
    naming a version and a model starts nothing)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    manager = FakeManager()
    real = tp.start_test_run
    monkeypatch.setattr(tp, "start_test_run",
                        lambda *a, **k: real(*a, **{**k, "manager": manager}))
    demo, _ctx, _registry = _page_component()
    app = FastAPI()
    gr.mount_gradio_app(app, demo, path="/ui")
    start = _named(demo, "_on_start")
    operator_mode(monkeypatch)
    forged_page = tp.read_page(operator_request(monkeypatch))
    forged_models = tp.read_models("demo", operator_request(monkeypatch))

    def call(base_url):
        with TestClient(app, base_url=base_url) as client:
            started = client.post(f"/ui/gradio_api/call/{start.api_name}",
                                  json={"data": [forged_page, forged_models, None, None,
                                                 None, True, False]})
            assert started.status_code == 200, started.text
            stream = client.get(f"/ui/gradio_api/call/{start.api_name}/"
                                f"{started.json()['event_id']}")
        lines = stream.text.splitlines()
        return json.dumps(json.loads(lines[lines.index("event: complete") + 1][len("data:"):]))

    refused = call("http://192.0.2.10")
    assert "Refused: " in refused and manager.started == []
    loopback = call(OPERATOR_BASE_URL)
    assert "Choose a cohort version the server lists" in loopback and manager.started == []
