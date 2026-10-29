"""
The shared model list and the one Load (M5-2, UI-A).
====================================================
`src/webui/components/model_list.py`. What these pin:

- **listing opens checkpoints, so it is an operator control**: D8's helper on
  the triggering request first; refused, nothing is listed or opened, and the
  reason is said; a direct call through Gradio's HTTP API meets the same check;
- **the rows**: "In use" pinned on top by the **full** model fingerprint the
  status serves, marking the checkpoint only; the clinical preset shows models
  finetuned for a task only, the latest 4 runs then "Show older runs"; the
  research preset every run; an unreadable checkpoint greyed; each row's
  recorded tests a state read from the ledger beside the checkpoint and, for
  contested, the report store;
- **the presets and the page's view**: the research preset is a table with
  "In use (Diagnosis)" on its own row, not pinned, and its listing carries the
  family view of the same one listing; a page's view lists one filter of the
  one vocabulary (all, base, a task, history, task models) or its explicit
  rows, draws its groups in its order (the rest after them), fills one extra
  column, and puts a note beside Load; the list redraws when the view changes;
  each filter's empty state and count say it in its own words;
- **the Load**: D8 on the "Load" request; a base, a history file or a refused
  record refused before anything of the file is read (the research Load off
  with its reason); the item found in the server's list again and its bytes
  hashed again (a list read earlier is caught), one confirmation (from → to:
  the task and scoring mode from the one table, the SP row a state of the
  task, the changed rows marked, what it replaces); then D8 again on "Reload
  with …", **both columns read again**: a change of what is served (any part
  of the served identity or the served workspace, the From column rebuilt
  from the plan's own listing) or of what would be loaded (the file's bytes,
  the workspace's graph; a file rewritten into a base is refused) is said with
  the new values and nothing is reloaded; a workspace SP table never changes a
  task model's To side; WP2's reload with the workspace, the checkpoint the
  server resolves (never a client path) and the task its record names, bound
  to the model fingerprint shown: through the real `run_reload` on a disease
  child trained in the test, bytes that replace the file after the Load hashed
  it are refused on the reload's own read and never served;
  "served model changed" set when any part of what is served changed; a usage
  history entry only when the list was ordered by a cohort version's score,
  and not when that version changed since;
- **the wiring**: Load and Reload refresh the chips, Reload sets the "served
  model changed" trigger and the phase and re-reads the list, Keep closes the
  confirmation; the 180 s count runs in the list's own confirmation, so both
  presets can sit in one page;
- **escaped**: a workspace name with markup in the confirmation.

Module: tests/unit/test_webui_model_list.py
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from src.utils.fingerprint import file_sha256  # noqa: E402
from src.webui.components import model_list as ml  # noqa: E402
from tests.fixtures.operator_access import (  # noqa: E402
    OPERATOR_BASE_URL,
    gradio_request,
    operator_mode,
    operator_request,
)

#: The base every task model here is finetuned from (its family's root run).
BASE_RUN = "20261001-000000_base_aaaaaa"
#: (architecture, run, files, kind) newest run first, as the listing orders
#: them: model bundles filed under ``gatv2`` (a base and its disease, gene and
#: patients-like-me children), a file without a model spec (history) under
#: ``gat`` and one that cannot be read under ``sage``.
RUNS = [
    ("gatv2", "20261018-091200_disease_8c2e7a", ("model-12.pt", "model-08.pt"), "disease"),
    ("gatv2", "20261017-091500_disease_3fa9c1", ("model-05.pt",), "disease"),
    ("sage", "20261016-083000_run_e4c2a9", ("broken.pt",), None),
    ("gatv2", "20261015-080000_gene_aaaaaa", ("model-03.pt",), "gene"),
    ("gatv2", "20261014-080000_patients_like_me_bbbbbb", ("model-02.pt",), "patients_like_me"),
    ("gatv2", "20261013-080000_disease_cccccc", ("model-01.pt",), "disease"),
    ("gat", "20261012-080000_run_dddddd", ("model-07.pt",), "history"),
    ("gatv2", BASE_RUN, ("base.pt",), "base"),
]


def _key(name):
    """A file's listed key (architecture/run/file) by its file name."""
    for arch, run, files, _kind in RUNS:
        if name in files:
            return f"{arch}/{run}/{name}"
    raise KeyError(name)


def payload(kind, epoch, fingerprint, parent=None):
    """A checkpoint's payload as the trainer writes it: a model bundle's record
    (a base, or a child of `parent` for a task; `test_task_first_selection`'s
    writer) or, for ``history``, a ShepherdGNN file without a model spec."""
    meta = {"checkpoint_kind": "weights", "epoch": epoch,
            "training_input_digests": {"split_manifest": fingerprint}}
    if kind == "history":
        return {**meta, "logs": {"val_mrr": 0.3 + epoch / 100}}
    from tests.unit import test_task_first_selection as bundles

    assert bundles.BASE_RUN == BASE_RUN
    task = None if kind == "base" else kind
    metric = "val_lp_accuracy" if task is None else "val_pooled_mrr"
    return {**bundles._record_payload(task, logs={metric: 0.3 + epoch / 100}, parent=parent),
            **meta}


@pytest.fixture(scope="module")
def demo_workspace(tmp_path_factory):
    from tests.fixtures.registered_cohort import write_demo_workspace

    return write_demo_workspace(tmp_path_factory.mktemp("mlbuild") / "demo")


@pytest.fixture
def root(demo_workspace, tmp_path, monkeypatch):
    """A workspaces root with one workspace and eight runs of checkpoints."""
    from src.utils.checkpoint_io import write_checkpoint

    root = tmp_path / "root"
    workspace = shutil.copytree(demo_workspace, root / "demo")
    fingerprint = file_sha256(workspace / "split_manifest.json")
    now = time.time()
    shas = {}
    base = workspace / "runs" / "gatv2" / BASE_RUN / "checkpoints" / "base.pt"
    base.parent.mkdir(parents=True)
    shas["base.pt"] = write_checkpoint(payload("base", 1, fingerprint), base)
    parent = {"sha256": shas["base.pt"], "run": BASE_RUN}
    for index, (arch, run, files, kind) in enumerate(RUNS):
        directory = workspace / "runs" / arch / run / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        for offset, name in enumerate(files):
            path = directory / name
            if kind is None:
                path.write_bytes(b"not a checkpoint")
            elif kind != "base":
                epoch = int(name.split("-")[1].split(".")[0])
                shas[name] = write_checkpoint(payload(kind, epoch, fingerprint, parent), path)
            stamp = now - index * 3600 - offset * 60
            os.utime(path, (stamp, stamp))
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    monkeypatch.delenv("SHEPHERD_REGISTRY", raising=False)
    return {"root": root, "workspace": workspace, "fingerprint": fingerprint, "shas": shas,
            "parent": parent}


def _status(root, served="model-05.pt", data_dir=None):
    """What the pipeline serves: a disease task model, as a bundle is served
    (`disease_task`, no shortest-path table, no mixing weight)."""
    sha = root["shas"].get(served)
    kg = json.loads((root["workspace"] / "split_manifest.json").read_text())["artifacts"]["kg"]
    return {
        "initialized": True, "current_data_dir": str(data_dir or root["workspace"]),
        "checkpoint_meta": {"epoch": 5},
        "served_identity": {
            "scoring_mode": "disease_task",
            "checkpoint": {"state": "verified" if sha else "absent", "sha256": sha},
            "graph": {"state": "verified", "kg_sha256": kg,
                      "split_manifest_sha256": root["fingerprint"]},
            "shortest_paths": {"state": "absent", "build_id": None, "max_hops": None},
            "dataset_record": {"state": "verified", "origin": "synthetic", "sha256": "e" * 64},
            "settings": {"device": "cpu", "device_requested": "cpu", "eta": None,
                         "max_path_length": 4, "path_length_penalty": 0.5,
                         "max_paths_per_candidate": 10,
                         "use_direct_path_optimization": True,
                         "phenotype_normalisation_version": 1},
        },
    }


@pytest.fixture
def serving(root, monkeypatch):
    status = {"value": _status(root)}
    monkeypatch.setattr(ml, "read_status", lambda: (status["value"], None))
    return status


# --------------------------------------------------------------------------- listing
def test_a_read_only_request_lists_nothing_and_opens_nothing(root, serving, monkeypatch):
    from src.api.services import server_lists

    monkeypatch.setattr(server_lists, "list_checkpoints",
                        lambda *a, **k: pytest.fail("a refused request listed checkpoints"))
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    listing = ml.list_models("demo", gradio_request())
    assert listing["refused"] is True and listing["items"] == []
    assert listing["problem"].startswith("Refused: ")
    operator_mode(monkeypatch)
    listing = ml.list_models("demo", gradio_request(server=("192.0.2.10", 8264)))
    assert listing["refused"] is True and "loopback" in listing["problem"]
    # A listing is no act (M5-3 review UI-6): the short line every page uses.
    shown = ml.list_html(listing, None, ml.PRESETS["clinical"])
    assert "Operator controls only: request not local." in shown
    assert "Refused" not in shown and "Nothing was done" not in shown


def test_an_operator_lists_the_serving_workspace_with_in_use_pinned(root, serving, monkeypatch):
    listing = ml.list_models(None, operator_request(monkeypatch))
    assert listing["workspace"] == "demo" and listing["serving"] == "demo"
    assert listing["served_sha"] == root["shas"]["model-05.pt"]
    assert len(listing["items"]) == 9
    html = ml.list_html(listing, None, ml.PRESETS["clinical"])
    first_row = html.index('class="shep-row')
    assert html.index("In use now") < first_row
    assert html[first_row:].startswith('class="shep-row"') or "epoch 5" in html[first_row:first_row + 400]
    assert html.count('shep-tag-accent">In use<') == 1
    assert "in use, pinned above" in html
    # the clinical preset: models finetuned for a task only (a base offers no
    # diagnosis; history and unreadable files are never run), the latest four
    # runs, then the rest on request
    for other in ("base_aaaaaa", "run_dddddd", "run_e4c2a9", "metadata not readable"):
        assert other not in html, other
    assert "Show older runs (1)" in html and "disease_cccccc" not in html
    older = ml.list_html(listing, None, ml.PRESETS["clinical"], show_older=True)
    assert "disease_cccccc" in older and "Show older runs" not in older
    research = ml.list_html(listing, None, ml.PRESETS["research"])
    assert "disease_cccccc" in research and 'In use (Diagnosis)<' in research
    # The drawer's own rule is its scope, not a filter it names: no "· Task
    # models only, of 9 listed" (the 9 would count files it never shows).
    assert ml.count_line(listing, ml.PRESETS["clinical"], False) == (
        "newest first · latest 4 runs · 5 checkpoints (of 6 in 5 runs)")


def test_in_use_is_matched_by_the_full_fingerprint_only(root, serving, monkeypatch):
    sha = root["shas"]["model-05.pt"]
    serving["value"]["served_identity"]["checkpoint"]["sha256"] = sha[:12] + "f" * 52
    listing = ml.list_models("demo", operator_request(monkeypatch))
    html = ml.list_html(listing, None, ml.PRESETS["clinical"])
    assert "In use now" not in html and "In use<" not in html


def test_each_rows_recorded_tests_come_from_the_ledger_beside_it(root, serving, monkeypatch):
    from src.evaluation.sidecar import empty_ledger, write_ledger

    checkpoints = root["workspace"] / "runs" / "gatv2"
    write_ledger(checkpoints / RUNS[0][1] / "checkpoints" / "evaluations.json",
                 empty_ledger(), None)
    (checkpoints / RUNS[1][1] / "checkpoints" / "evaluations.json").write_text("{not json")
    listing = ml.list_models("demo", operator_request(monkeypatch))
    recorded = listing["recorded"]
    assert recorded[_key("model-12.pt")]["state"] == "none"
    assert recorded[_key("model-05.pt")]["state"] == "not_read"
    assert recorded[_key("model-03.pt")]["state"] == "no_ledger"
    assert _key("broken.pt") not in recorded                      # unreadable: no state


def test_contested_and_unchecked_are_counted_apart_from_the_report_store(root, monkeypatch):
    """A record contested in its version's report store is counted; one whose
    state cannot be checked (its store, or a version the registry does not
    list) is counted apart and never as uncontested, and never hides a known
    contested one (M5-3 review EVAL-3: one uncheckable record turned
    "2 · 1 contested" into "3 · contested state not checked")."""
    import src.evaluation.report_store as report_store
    import src.evaluation.sidecar as sidecar
    from src.webui.components import renderers as r

    sha = root["shas"]["model-12.pt"]
    records = [{"checkpoint_digest": sha, "cohort_role": "research_selection",
                "cohort_digest": d, "measurement_semantics_digest": "a" * 64}
               for d in ("1" * 64, "1" * 64, "2" * 64, "3" * 64)]
    monkeypatch.setattr(sidecar, "read_ledger", lambda path: ({"records": records}, "x" * 64))
    monkeypatch.setattr(sidecar, "records_for", lambda ledger, digest: ledger["records"]
                        if digest == sha else [])

    class Version:
        def __init__(self, digest):
            self.identity = type("I", (), {"sha256": digest})()

    class Versions:
        def by_digest(self, digest):
            if digest == "3" * 64:
                return None, "its cohort version is not one the server's registry lists"
            return Version(digest), None

    class Status:
        def __init__(self, state):
            self.state, self.differing, self.disagreeing_runs, self.problems = state, (), (), ()

    class Scan:
        reports: dict = {}

        def __init__(self, state):
            self._state = state

        def status(self, key):
            return Status(self._state)

    from src.api.services.record_states import CohortVersions

    monkeypatch.setattr(CohortVersions, "listed",
                        classmethod(lambda cls, environ=None: Versions()))
    states = {"1" * 64: "contested", "2" * 64: "uncontested"}
    monkeypatch.setattr(report_store, "scan_report_store",
                        lambda version: Scan(states[version.identity.sha256]))
    item = {"architecture": "gatv2", "run": RUNS[0][1], "filename": "model-12.pt",
            "sha256": sha, "problem": None}
    result = ml.recorded_tests_for(root["workspace"], [item])[ml.item_key(item)]
    assert (result.state, result.count, result.contested, result.unchecked) == (
        "records", 4, 2, 1)
    assert r.recorded_tests_text(result) == (
        "Recorded tests: 4 · 2 contested · contested state of 1 not checked")
    states["2" * 64] = "cannot_be_checked"
    result = ml.recorded_tests_for(root["workspace"], [item])[ml.item_key(item)]
    assert (result.count, result.contested, result.unchecked) == (4, 2, 2)
    states["1" * 64] = "uncontested"
    states["2" * 64] = "uncontested"
    result = ml.recorded_tests_for(root["workspace"], [item])[ml.item_key(item)]
    assert r.recorded_tests_text(result) == (
        "Recorded tests: 4 · contested state of 1 not checked")


# --------------------------------------------------------------------------- the Load
def _listing(root, monkeypatch):
    return ml.list_models("demo", operator_request(monkeypatch))


def test_load_needs_d8_on_its_own_request_and_resolves_nothing_when_refused(
        root, serving, monkeypatch):
    from src.api.services import server_lists

    listing = _listing(root, monkeypatch)
    key = _key("model-12.pt")
    monkeypatch.setattr(server_lists, "resolve_checkpoint",
                        lambda *a, **k: pytest.fail("a refused Load resolved a checkpoint"))
    plan, html = ml.request_load(listing, key, None, gradio_request(server=("192.0.2.10", 8264)))
    assert plan is None and "Refused: " in html


def _changed_rows(html):
    rows = [row for row in html.split("<tr") if 'class="shep-changed"' in row]
    return [row.split('<th scope="row">')[1].split("<")[0] for row in rows]


def test_load_shows_one_confirmation_from_to_with_the_changed_rows_marked(
        root, serving, monkeypatch):
    listing = _listing(root, monkeypatch)
    key = _key("model-12.pt")
    plan, html = ml.request_load(listing, key, None, operator_request(monkeypatch))
    new, old = root["shas"]["model-12.pt"][:12], root["shas"]["model-05.pt"][:12]
    assert plan["to"]["sha256"] == root["shas"]["model-12.pt"]
    assert plan["from"]["sha256"] == root["shas"]["model-05.pt"]
    assert f"Reload the pipeline with <code>{new}</code> for every session?" in html
    assert ml.REPLACES_NOTE in html
    assert _changed_rows(html) == ["Model", "Model fingerprint"]
    assert "GATv2 · run 2026-10-17 09:15 · epoch 5" in html and old in html
    assert "same dataset" in html and ">same<" in html
    # The task and its scoring mode from the one table, on both sides; the
    # disease task uses no shortest-path table, said as such on both sides.
    assert html.count("Disease (<code>disease_task</code>)") == 2
    assert html.count("not used by this task") == 2 and ">absent<" not in html
    assert "eta" not in html                         # no mixing weight is predicted
    # not a client path: the plan names the item, never a file
    assert all("/" not in str(plan[k]) or k == "run" for k in ("workspace", "filename"))
    # A gene model's To side: its task and the term it computes from its view.
    plan, html = ml.request_load(listing, _key("model-03.pt"), None,
                                 operator_request(monkeypatch))
    assert plan["to"]["mode"] == "gene_task" and plan["to"]["sp"] == "computed"
    assert _changed_rows(html) == ["Model", "Model fingerprint", "Task and scoring", "SP table"]
    assert "Gene (<code>gene_task</code>)" in html
    assert "computed from the model's graph view" in html


def test_an_unloadable_item_cannot_be_loaded(root, serving, monkeypatch):
    listing = _listing(root, monkeypatch)
    plan, html = ml.request_load(listing, _key("broken.pt"), None,
                                 operator_request(monkeypatch))
    assert plan is None and "Select a model that can be loaded" in html
    plan, html = ml.request_load(listing, "../../../etc/passwd", None,
                                 operator_request(monkeypatch))
    assert plan is None


def _refused_record(root):
    """A file whose model record this build refuses (another record version),
    in a run of its own; its key."""
    from src.utils.checkpoint_io import write_checkpoint

    run = "20261019-080000_disease_ffffff"
    path = root["workspace"] / "runs" / "gatv2" / run / "checkpoints" / "model-09.pt"
    path.parent.mkdir(parents=True)
    write_checkpoint({**payload("disease", 9, root["fingerprint"], root["parent"]),
                      "record_version": 999}, path)
    return f"gatv2/{run}/model-09.pt"


def test_a_base_a_history_file_and_a_refused_record_are_refused_before_any_read(
        root, serving, monkeypatch):
    """README §5 ("a base offers no diagnosis"), §4 and K6 decision 1 (a file
    without a model spec is history, never run): such a row is refused from
    its listed record, before the file is resolved, hashed or the status read
    (the pipeline would refuse a base only after reading the graph, and would
    still serve a history file)."""
    from src.api.services import server_lists
    from src.evaluation import sidecar

    refused = _refused_record(root)
    listing = _listing(root, monkeypatch)
    assert ml.find_item(listing, refused)["record"] == "refused"
    for target, name in ((server_lists, "resolve_checkpoint"), (sidecar, "checkpoint_matches"),
                         (ml, "read_status")):
        monkeypatch.setattr(target, name, lambda *a, **k: pytest.fail("a refused row was read"))
    for key, why in ((_key("base.pt"), ml.BASE_REFUSAL), (_key("model-07.pt"), ml.HISTORY_REFUSAL),
                     (refused, "Record not readable: ")):
        plan, html = ml.request_load(listing, key, None, operator_request(monkeypatch))
        assert plan is None and why in html and "Nothing was done." in html, key
        assert not ml.can_load(ml.find_item(listing, key))
    assert "record version 999" in ml.load_refusal(ml.find_item(listing, refused))
    assert ml.can_load(ml.find_item(listing, _key("model-12.pt")))
    # The reason beside Load is escaped (a record's problem is the server's words).
    evil = "<img src=x onerror=alert(1)>"
    shown = ml.load_reason_html({"record": "refused", "record_problem": evil,
                                 "sha256": "a" * 64, "problem": None})
    assert evil not in shown and "Record not readable: &lt;img src=x" in shown


def _plan(root, monkeypatch, ordering=None):
    listing = _listing(root, monkeypatch)
    plan, _html = ml.request_load(listing, _key("model-12.pt"), ordering,
                                  operator_request(monkeypatch))
    return plan


def test_confirm_needs_d8_again_and_a_refusal_reloads_nothing(root, serving, monkeypatch):
    plan = _plan(root, monkeypatch)
    calls = []
    step = ml.finish_load(plan, gradio_request(headers=[("Origin", "https://evil.example")]),
                          reload=lambda request: calls.append(request))
    assert calls == [] and "Refused: " in step.reason_html and step.plan == plan
    assert step.changed is False


def test_a_model_in_use_that_changed_since_is_said_never_reloaded_over(root, serving, monkeypatch):
    plan = _plan(root, monkeypatch)
    serving["value"] = _status(root, served="model-08.pt")
    calls = []
    step = ml.finish_load(plan, operator_request(monkeypatch),
                          reload=lambda request: calls.append(request))
    assert calls == [] and step.plan is not None and step.changed is False
    assert ml.FROM_CHANGED in step.confirm_html and ml.NOTHING_RELOADED in step.confirm_html
    assert step.plan["from"]["sha256"] == root["shas"]["model-08.pt"]
    # The From column is read as the confirmation reads it, from the plan's
    # own listing: the served workspace is named, the model and what it was
    # trained on are known, and an unchanged row is not marked.
    assert step.plan["from"]["model"] == "GATv2 · run 2026-10-18 09:12 · epoch 8"
    assert "not a listed workspace" not in step.confirm_html
    assert "not in this workspace's list" not in step.confirm_html
    assert _changed_rows(step.confirm_html) == ["Model", "Model fingerprint"]
    # Confirmed again as shown, it reloads.
    sent = []
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    again = ml.finish_load(step.plan, operator_request(monkeypatch), reload=lambda request: (
        sent.append(request) or PipelineReloadResponse(success=True, message="ok",
                                                       status=PipelineStatusResponse(**_status(
                                                           root, served="model-12.pt")))))
    assert len(sent) == 1 and "Reloaded." in again.confirm_html


@pytest.mark.parametrize("change", ["workspace", "sp_table", "eta"])
def test_any_part_of_what_is_served_that_changed_is_said(root, serving, monkeypatch, change,
                                                       tmp_path):
    """The same checkpoint, served by another session with another workspace,
    shortest-path table or eta: the Load says so rather than reloading over it."""
    plan = _plan(root, monkeypatch)
    status = _status(root)
    if change == "workspace":
        other = tmp_path / "elsewhere"
        other.mkdir()
        status["current_data_dir"] = str(other)
    elif change == "sp_table":
        status["served_identity"]["shortest_paths"] = {"state": "verified", "build_id": "b1",
                                                       "max_hops": 4}
    else:
        status["served_identity"]["settings"]["eta"] = 0.5
    serving["value"] = status
    calls = []
    step = ml.finish_load(plan, operator_request(monkeypatch),
                          reload=lambda request: calls.append(request))
    assert calls == [] and step.plan is not None and ml.FROM_CHANGED in step.confirm_html


def test_a_reload_of_the_same_checkpoint_with_another_workspace_is_a_change(
        root, serving, monkeypatch, tmp_path):
    """"served model changed" follows the whole served identity: the same
    checkpoint served from another directory is a change for the card."""
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    other = tmp_path / "elsewhere"
    other.mkdir()
    serving["value"] = _status(root, served="model-05.pt", data_dir=other)
    listing = _listing(root, monkeypatch)
    plan, _html = ml.request_load(listing, _key("model-05.pt"), None,
                                  operator_request(monkeypatch))
    step = ml.finish_load(plan, operator_request(monkeypatch), reload=lambda request: (
        PipelineReloadResponse(success=True, message="ok", status=PipelineStatusResponse(
            **_status(root, served="model-05.pt")))))
    assert "Reloaded." in step.confirm_html and step.served == root["shas"]["model-05.pt"]
    assert step.changed is True
    unchanged = ml.finish_load(plan, operator_request(monkeypatch), reload=lambda request: (
        PipelineReloadResponse(success=False, message="refused", status=PipelineStatusResponse(
            **_status(root, served="model-05.pt", data_dir=other)))))
    assert unchanged.changed is False


def _rewrite(root, name, epoch, kind="disease"):
    """Write other bytes into a listed checkpoint, as a training run that
    overwrites ``last.pt`` each epoch does (by default another epoch of the
    same disease model)."""
    from src.utils.checkpoint_io import write_checkpoint

    path = root["workspace"] / "runs" / "gatv2" / RUNS[0][1] / "checkpoints" / name
    return write_checkpoint(payload(kind, epoch, root["fingerprint"], root["parent"]),
                            path), path


def test_bytes_that_changed_after_the_confirmation_are_shown_never_loaded(
        root, serving, monkeypatch):
    """The trainer rewrites the file between "Load" and "Reload with …": the
    confirmation is shown again with what the file holds now; nothing is
    reloaded until that is confirmed."""
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    plan = _plan(root, monkeypatch)
    new_sha, path = _rewrite(root, "model-12.pt", 13)
    calls = []
    step = ml.finish_load(plan, operator_request(monkeypatch),
                          reload=lambda request: calls.append(request))
    assert calls == [] and step.plan is not None and step.changed is False
    assert ml.TO_CHANGED in step.confirm_html
    assert step.plan["to"]["sha256"] == new_sha and step.plan["to"]["model"].endswith("epoch 13")
    assert f"Reload the pipeline with <code>{new_sha[:12]}</code>" in step.confirm_html
    sent = []
    again = ml.finish_load(step.plan, operator_request(monkeypatch), reload=lambda request: (
        sent.append(request) or PipelineReloadResponse(
            success=True, message="ok", status=PipelineStatusResponse(**{
                **_status(root), "served_identity": {
                    **_status(root)["served_identity"],
                    "checkpoint": {"state": "verified", "sha256": new_sha}}}))))
    assert [Path(r.checkpoint_path) for r in sent] == [path]
    assert "Reloaded." in again.confirm_html and again.served == new_sha


def test_a_file_rewritten_into_a_base_after_the_confirmation_is_never_reloaded(
        root, serving, monkeypatch):
    """The refusal is checked again on what the file holds at "Reload with
    …": a base written over the confirmed file is refused, never shown to
    confirm and never reloaded."""
    plan = _plan(root, monkeypatch)
    _rewrite(root, "model-12.pt", 3, kind="base")
    calls = []
    step = ml.finish_load(plan, operator_request(monkeypatch),
                          reload=lambda request: calls.append(request))
    assert calls == [] and step.plan is None and step.changed is False
    assert "Not reloaded." in step.confirm_html and ml.BASE_REFUSAL in step.confirm_html
    # Asked for again from the list read before: refused from what it holds now.
    plan, html = ml.request_load(_listing(root, monkeypatch), _key("model-12.pt"), None,
                                 operator_request(monkeypatch))
    assert plan is None and ml.BASE_REFUSAL in html


def test_a_workspace_sp_table_never_changes_a_task_models_to_side(root, serving, monkeypatch):
    """The To side was guessed from the workspace (an SP table there read as
    "gnn_plus_shortest_path" with the server's eta), which a task model's
    serving never reads: the mode is the task's, from the one table, so a
    table appearing after the confirmation changes nothing and the Load goes
    on with the task its record names."""
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    plan = _plan(root, monkeypatch)
    assert (plan["to"]["mode"], plan["to"]["sp"]) == ("disease_task", "absent")
    (root["workspace"] / "shortest_paths.pt").write_bytes(b"a table")
    sent = []
    step = ml.finish_load(plan, operator_request(monkeypatch), reload=lambda request: (
        sent.append(request) or PipelineReloadResponse(success=True, message="ok",
                                                       status=PipelineStatusResponse(**_status(
                                                           root, served="model-12.pt")))))
    assert len(sent) == 1 and ml.TO_CHANGED not in step.confirm_html
    assert "Reloaded." in step.confirm_html and sent[0].task == "disease"


def test_the_to_sides_sp_state_is_the_served_identitys_one_rule(root, serving, monkeypatch):
    """The SP state the To column says for a task model is the one its scoring
    mode is served with (`core.types.SCORING_MODE_SP_STATES`, the rule
    `ServedIdentity` checks and the pipeline builds from), read from that one
    table and never from a copy: a change of the rule changes what the
    confirmation says, so reloading the same model never marks its SP row
    changed."""
    from src.core import types

    listing = _listing(root, monkeypatch)
    item = ml.find_item(listing, _key("model-12.pt"))
    for task, mode in types.TASK_SCORING_MODES.items():
        side = ml._to_side({**item, "task": task}, "demo", root["workspace"], [])
        assert (side["mode"], side["sp"]) == (mode, types.SCORING_MODE_SP_STATES[mode])
    monkeypatch.setitem(types.SCORING_MODE_SP_STATES, "patients_like_me", "computed")
    side = ml._to_side({**item, "task": "patients_like_me"}, "demo", root["workspace"], [])
    assert side["sp"] == "computed"


def test_a_list_read_earlier_is_caught_when_load_is_asked(root, serving, monkeypatch):
    listing = _listing(root, monkeypatch)
    new_sha, _path = _rewrite(root, "model-12.pt", 14)
    plan, html = ml.request_load(listing, _key("model-12.pt"), None,
                                 operator_request(monkeypatch))
    assert plan["to"]["sha256"] == new_sha and plan["to"]["model"].endswith("epoch 14")
    assert "The file changed since the list was read" in html and new_sha[:12] in html


def test_load_finds_the_item_in_the_servers_list_again(root, serving, monkeypatch):
    """A listed item whose file is gone since the list was read is refused,
    and nothing is opened from the listing's word for it."""
    listing = _listing(root, monkeypatch)
    (root["workspace"] / "runs" / "gatv2" / RUNS[0][1] / "checkpoints" / "model-12.pt").unlink()
    plan, html = ml.request_load(listing, _key("model-12.pt"), None,
                                 operator_request(monkeypatch))
    assert plan is None and "Refused: " in html and "is listed in the workspace" in html


def test_the_reload_gets_what_the_server_resolves_and_the_fingerprint_shown(
        root, serving, monkeypatch):
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    plan = _plan(root, monkeypatch)
    sent = []

    def reload(request):
        sent.append(request)
        status = _status(root, served="model-12.pt")
        return PipelineReloadResponse(success=True, message="ok",
                                      status=PipelineStatusResponse(**status))

    step = ml.finish_load(plan, operator_request(monkeypatch), reload=reload)
    (request,) = sent
    expected = (root["workspace"] / "runs" / "gatv2" / RUNS[0][1] / "checkpoints" / "model-12.pt")
    assert Path(request.data_dir) == root["workspace"].resolve()
    assert Path(request.checkpoint_path) == expected
    assert request.expected_checkpoint_sha256 == root["shas"]["model-12.pt"]
    # The task the confirmed model's record names: the reload refuses a build
    # that serves another (switching task means loading another model).
    assert request.task == "disease"
    assert step.changed is True and step.served == root["shas"]["model-12.pt"]
    assert "Reloaded." in step.confirm_html and step.plan is None


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A complete workspace with a base and a disease child trained on it for
    real (`tests/fixtures/trained_children`, 1 epoch on CPU), once per module."""
    from tests.fixtures.registered_cohort import write_demo_workspace
    from tests.fixtures.trained_children import train_children

    return train_children(write_demo_workspace(tmp_path_factory.mktemp("mltrained") / "data"),
                          ("disease",))


@pytest.fixture
def buildable(trained, tmp_path, monkeypatch):
    """A workspaces root whose one workspace (``data``) the real reload builds:
    a graph, and a disease model finetuned on it as ``last.pt`` of its run, the
    file a running training job rewrites every epoch. Nothing is served yet;
    the device is the CPU."""
    import src.api.main as api_main
    from tests.fixtures.trained_children import CHILD_RUNS

    for field in ("pipeline", "kg", "_current_data_dir", "_current_checkpoint_path"):
        monkeypatch.setattr(api_main.app_state, field, None, raising=False)
    monkeypatch.setattr(api_main.app_state, "is_ready", False, raising=False)
    for name in ("SHEPHERD_KG_PATH", "SHEPHERD_DATA_DIR", "SHEPHERD_CHECKPOINT_PATH",
                 "SHEPHERD_REGISTRY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")
    root = tmp_path / "root"
    data_dir = shutil.copytree(trained["data_dir"], root / "data")
    run = CHILD_RUNS["disease"]
    last = data_dir / "runs" / "gatv2" / run / "checkpoints" / "last.pt"
    source = tmp_path / "source.pt"
    shutil.copyfile(last, source)
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    return {"api": api_main, "data_dir": data_dir, "last": last, "source": source,
            "key": f"gatv2/{run}/last.pt"}


def _shifted(source, shift):
    """The bytes of `source`'s model with every weight moved by `shift`: another
    model the build accepts (the same architecture, trained on the same graph)."""
    import io

    import torch

    contents = torch.load(source, map_location="cpu", weights_only=False)
    contents["state_dict"] = {name: value + shift if value.is_floating_point() else value
                              for name, value in contents["state_dict"].items()}
    buffer = io.BytesIO()
    torch.save(contents, buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("how", ["rename", "rewrite"])
def test_bytes_replaced_before_the_reloads_own_read_are_refused_never_served(
        buildable, monkeypatch, tmp_path, how):
    """EB-1, through the real `run_reload` and `build_pipeline`. A model loaded
    as shown is served (the positive control). Then another model is listed and
    confirmed, and the file is replaced after the Load hashed it, while the build
    reads the workspace's graph and before it reads the checkpoint, as a
    training run's next epoch renames a new ``last.pt`` into place (and once in
    place). The reload compares its own read with the fingerprint shown: nothing
    is published, the model in use stays, the page says not reloaded and why,
    and the usage history records the load as not made. Catches the reload
    given no fingerprint (or not comparing it) and a check after publication."""
    import src.evaluation.cohort_registry as cohort_registry
    import src.evaluation.report_store as report_store
    from src.config import server_settings
    from tests.fixtures.replacement import replace_after_read, replace_file

    api, last = buildable["api"], buildable["last"]
    written = []
    monkeypatch.setattr(report_store, "record_model_load",
                        lambda version, **fields: written.append(fields))
    version = type("V", (), {"identity": type("I", (), {"sha256": "c" * 64})()})()
    monkeypatch.setattr(cohort_registry, "resolve_version",
                        lambda registry, label, **kwargs: version)
    registry = tmp_path / "registry"
    registry.mkdir()
    monkeypatch.setattr(server_settings, "cohort_registry",
                        lambda environ=None: server_settings.Setting("SHEPHERD_REGISTRY", registry))
    ordering = {"version": "mygene2-2026-09", "version_sha256": "c" * 64,
                "metric": "untruncated_mrr", "scoring_semantics_digest": "s" * 64}
    published = []
    publish = api.publish_pipeline

    def spy(bundle):
        published.append(bundle.config["served_identity"]["checkpoint"]["sha256"])
        return publish(bundle)

    monkeypatch.setattr(api, "publish_pipeline", spy)

    def plan_for_last():
        listing = ml.list_models("data", operator_request(monkeypatch))
        plan, html = ml.request_load(listing, buildable["key"], ordering,
                                     operator_request(monkeypatch))
        assert plan is not None, html
        return plan

    first = file_sha256(last)
    loaded = ml.finish_load(plan_for_last(), operator_request(monkeypatch))
    assert "Reloaded." in loaded.confirm_html, loaded.confirm_html
    assert published == [first] and loaded.served == first
    serving = api.app_state.pipeline
    assert serving.served_identity().checkpoint.sha256 == first

    replace_file(last, _shifted(buildable["source"], 1.0), how)
    shown = file_sha256(last)
    plan = plan_for_last()
    assert plan["to"]["sha256"] == shown
    other = _shifted(buildable["source"], 2.0)
    swap = replace_after_read(monkeypatch, after=buildable["data_dir"] / "kg.json",
                              target=last, data=other, how=how)
    step = ml.finish_load(plan, operator_request(monkeypatch))

    assert swap.fired == 1
    held = file_sha256(last)
    assert held not in (first, shown)
    assert published == [first]
    assert api.app_state.pipeline is serving
    assert serving.served_identity().checkpoint.sha256 == first
    assert "Not reloaded; the model in use is unchanged." in step.confirm_html
    assert "is not the model that was shown" in step.confirm_html
    assert f"it holds {held[:12]} now, not {shown[:12]}" in step.confirm_html
    assert "Reloaded" not in step.confirm_html.replace("Not reloaded", "")
    assert step.served == first and step.changed is False and step.plan is None
    assert [entry["loaded"] for entry in written] == [True, False]
    assert written[-1]["checkpoint_sha256"] == shown
    assert "is not the model that was shown" in written[-1]["why"]


@pytest.mark.parametrize("bytes_now", ["torn", "another graph", "not a dict", "gone"])
def test_any_other_bytes_at_the_shown_file_are_refused_as_not_the_model_shown(
        buildable, monkeypatch, tmp_path, bytes_now):
    """Root-cause round 3 (INT-R1), through the real `run_reload` and
    `build_pipeline`: whatever replaced the confirmed file after the Load
    hashed it -- torn bytes, a checkpoint recorded for another graph, a pickle
    that is not a checkpoint, or nothing -- is refused on the build's one
    checkpoint read before torch parses it, as "not the model that was shown":
    never as unreadable, another graph's or not a checkpoint. No model is
    built, nothing is published, the model in use stays, and the usage history
    records the load as not made, with that reason."""
    from tests.unit.test_served_identity import (
        change_after_the_graph_read,
        never_parsed_or_built,
        not_the_shown_bytes,
    )

    api, last = buildable["api"], buildable["last"]
    _version, ordering, written = _ordered_by_a_version(monkeypatch, tmp_path)

    def plan_for_last():
        listing = ml.list_models("data", operator_request(monkeypatch))
        plan, html = ml.request_load(listing, buildable["key"], ordering,
                                     operator_request(monkeypatch))
        assert plan is not None, html
        return plan

    shown = file_sha256(last)
    loaded = ml.finish_load(plan_for_last(), operator_request(monkeypatch))
    assert "Reloaded." in loaded.confirm_html, loaded.confirm_html
    serving = api.app_state.pipeline
    plan = plan_for_last()
    data = not_the_shown_bytes(last, bytes_now)
    fired = change_after_the_graph_read(monkeypatch, buildable["data_dir"], last, data)
    never_parsed_or_built(monkeypatch)

    step = ml.finish_load(plan, operator_request(monkeypatch))

    assert fired, "the build never read the graph"
    held = (f"it is gone; {shown[:12]} was shown" if data is None
            else f"it holds {file_sha256(last)[:12]} now, not {shown[:12]}")
    assert "Not reloaded; the model in use is unchanged." in step.confirm_html
    assert f"is not the model that was shown: {held}. Nothing was published" in \
        step.confirm_html, step.confirm_html
    assert api.app_state.pipeline is serving
    assert step.served == shown and step.changed is False and step.plan is None
    assert [entry["loaded"] for entry in written] == [True, False]
    assert written[-1]["checkpoint_sha256"] == shown
    assert "is not the model that was shown" in written[-1]["why"]


def test_a_refused_reload_leaves_the_model_in_use_and_says_so(root, serving, monkeypatch):
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    plan = _plan(root, monkeypatch)
    step = ml.finish_load(plan, operator_request(monkeypatch), reload=lambda request: (
        PipelineReloadResponse(success=False, message="No checkpoint found.",
                               status=PipelineStatusResponse(**_status(root)))))
    assert "Not reloaded; the model in use is unchanged." in step.confirm_html
    assert "No checkpoint found." in step.confirm_html and step.changed is False
    raising = ml.finish_load(plan, operator_request(monkeypatch),
                             reload=lambda request: (_ for _ in ()).throw(RuntimeError("x")))
    assert "Not reloaded" in raising.confirm_html and "RuntimeError" in raising.confirm_html


def _ordered_by_a_version(monkeypatch, tmp_path):
    """A registry holding one committed version, the deployment's, and the
    ordering of a list by that version's score; the usage history entries a
    Load writes are collected instead of appended."""
    import src.evaluation.report_store as report_store
    from src.config import server_settings
    from tests.fixtures.report_store import store_version
    from tests.unit.test_evaluation_sidecar import cohort_version

    written = []
    monkeypatch.setattr(report_store, "record_model_load",
                        lambda version, **fields: written.append(fields))
    registry = tmp_path / "registry"
    version = store_version(registry, cohort_version())
    monkeypatch.setattr(server_settings, "cohort_registry",
                        lambda environ=None: server_settings.Setting("SHEPHERD_REGISTRY", registry))
    ordering = {"version": version.manifest["label"], "version_sha256": version.identity.sha256,
                "metric": "untruncated_mrr", "scoring_semantics_digest": "s" * 64}
    return version, ordering, written


def test_the_usage_history_is_written_only_for_a_list_ordered_by_a_versions_score(
        root, serving, monkeypatch, tmp_path):
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    _version, ordering, written = _ordered_by_a_version(monkeypatch, tmp_path)

    def reload(request):
        return PipelineReloadResponse(success=True, message="ok", status=PipelineStatusResponse(
            **_status(root, served="model-12.pt")))

    unordered = ml.finish_load(_plan(root, monkeypatch), operator_request(monkeypatch),
                               reload=reload)
    # an unordered list: no entry is tried, so none is said to be missing
    assert written == [] and "usage history" not in unordered.confirm_html
    serving["value"] = _status(root)
    step = ml.finish_load(_plan(root, monkeypatch, ordering), operator_request(monkeypatch),
                          reload=reload)
    assert written == [{"checkpoint_sha256": root["shas"]["model-12.pt"],
                        "metric": "untruncated_mrr", "scoring_semantics_digest": "s" * 64,
                        "loaded": True, "why": None}]
    assert "usage history records this load" in step.confirm_html
    serving["value"] = _status(root)
    ml.finish_load(_plan(root, monkeypatch, ordering), operator_request(monkeypatch),
                   reload=lambda request: PipelineReloadResponse(
                       success=False, message="refused", status=PipelineStatusResponse(
                           **_status(root))))
    assert written[-1]["loaded"] is False and written[-1]["why"] == "refused"
    # The cohort version changed since the list was ordered: no entry is
    # written against it, and the page says why.
    count = len(written)
    stale = {**ordering, "version_sha256": "d" * 64}
    step = ml.finish_load(_plan(root, monkeypatch, stale), operator_request(monkeypatch),
                          reload=reload)
    assert len(written) == count
    assert "the cohort version changed since the list was ordered" in step.confirm_html


@pytest.mark.parametrize("bytes_now", ["torn", "another schema", "gone"])
def test_a_version_whose_bytes_changed_since_the_ordering_gets_no_entry_and_is_said_changed(
        root, serving, monkeypatch, tmp_path, bytes_now):
    """Root-cause round 3 (INT-R2): the usage history's version is read bound
    to the digest the list was ordered by, compared before it is decoded
    (`resolve_version(expected_sha256=)`), so other bytes at the version --
    torn, another schema, or none -- are said as "changed since the list was
    ordered", never as a version that cannot be read, and no entry is
    written."""
    import src.evaluation.cohort_registry as cohort_registry
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    version, ordering, written = _ordered_by_a_version(monkeypatch, tmp_path)
    path = version.identity.path
    if bytes_now == "gone":
        path.unlink()
    elif bytes_now == "torn":
        path.write_bytes(path.read_bytes()[:40])
    else:
        manifest = json.loads(path.read_bytes())
        manifest["schema_version"] = 99
        path.write_bytes(cohort_registry.encode_json(manifest))

    step = ml.finish_load(_plan(root, monkeypatch, ordering), operator_request(monkeypatch),
                          reload=lambda request: PipelineReloadResponse(
                              success=True, message="ok", status=PipelineStatusResponse(
                                  **_status(root, served="model-12.pt"))))

    assert written == []
    assert ("But the usage history was not written: the cohort version changed since the "
            "list was ordered.") in step.confirm_html, step.confirm_html


def test_a_markup_workspace_name_is_escaped_in_the_confirmation(root, serving, monkeypatch):
    listing = _listing(root, monkeypatch)
    plan, _html = ml.request_load(listing, _key("model-12.pt"), None,
                                  operator_request(monkeypatch))
    plan["to"]["workspace"] = "<b>x</b>"
    html = ml.confirmation_html(plan)
    assert "<b>x</b>" not in html and "&lt;b&gt;x&lt;/b&gt;" in html


# --------------------------------------------------------------------------- presets and views
def test_the_research_preset_is_a_table_with_in_use_on_its_own_row(root, serving, monkeypatch):
    listing = _listing(root, monkeypatch)
    research = ml.list_html(listing, None, ml.PRESETS["research"])
    assert '<table class="shep-tbl shep-mtbl"' in research and "In use now" not in research
    for column in ("Model", "Fingerprints: model · dataset trained on",
                   "Validation (Trainer)", "Recorded tests"):
        assert f">{column}</th>" in research, column
    served_row = next(row for row in research.split("<tr") if "disease_3fa9c1" in row)
    assert "In use (Diagnosis)" in served_row and "0.3500" in served_row
    assert research.count("In use (Diagnosis)") == 1
    assert "metadata not readable · cannot be loaded" in research
    assert research.count(" data-key=") == 9


def test_a_pages_view_orders_filters_and_adds_a_column(root, serving, monkeypatch):
    listing = _listing(root, monkeypatch)
    keys = [ml.item_key(i) for i in listing["items"]]
    first, second = _key("model-01.pt"), _key("model-12.pt")
    view = {"task": "disease",
            "groups": [{"title": "Ordered by Mode C MRR (untruncated) on v1 · highest first",
                        "keys": [first, second, "gatv2/nope/x.pt"]},
                       {"title": "Contested on this version · not ordered", "keys": []}],
            "rest_title": "Not tested on this cohort version · newest first",
            "column": "Mode C MRR, v1 · regime",
            "cells": {first: '0.2310<div class="shep-sub">CUDA · deterministic</div>'},
            "note": "The usage history records <ordered by …>."}
    html = ml.list_html(listing, None, ml.PRESETS["research"], view=view)
    ordered = html.index("Ordered by Mode C MRR")
    assert ordered < html.index("disease_cccccc") < html.index("disease_8c2e7a") < html.index(
        "Contested on this version") < html.index("Not tested on this cohort version")
    for other in ("run_e4c2a9", "gene_aaaaaa", "base_aaaaaa", "run_dddddd"):
        assert other not in html, other                    # not disease models: filtered out
    assert ">Mode C MRR, v1 · regime</th>" in html and "CUDA · deterministic" in html
    assert ml.count_line(listing, ml.PRESETS["research"], False, view) == (
        "in the groups below · 4 checkpoints in 3 runs · Disease only, of 9 listed")
    assert ml.note_html(view) == ('<span class="shep-note">The usage history records '
                                  '&lt;ordered by …&gt;.</span>')
    # the clinical rows draw a page's cell as one more line of the row
    rows = ml.list_html(listing, None, ml.PRESETS["clinical"],
                        view={"cells": {second: "<b>regime</b>"}})
    assert '<span class="shep-row-sub"><b>regime</b></span></button>' in rows
    assert len(keys) == 9
    # A view's explicit rows (a family view's, its context rows included).
    rows = ml.shown_items(listing, {"task": "disease", "keys": [_key("base.pt"), first]})
    assert [ml.item_key(i) for i in rows] == [first, _key("base.pt")]
    # Its count says the filter's rows and the context rows apart: the base
    # shown as the parent of the disease models is never counted as one, and
    # the line never says "Disease only" beside it.
    family = [_key(name) for name in ("base.pt", "model-12.pt", "model-08.pt", "model-05.pt",
                                      "model-01.pt")]
    view = {"task": "disease", "keys": family,
            "groups": [{"title": "Family from run 2026-10-01 00:00 · 5 files", "keys": family}]}
    assert ml.count_line(listing, ml.PRESETS["research"], False, view) == (
        "in the groups below · 4 checkpoints in 3 runs · Disease, with 1 context row, of 9 "
        "listed")


def test_one_filter_vocabulary_and_the_drawer_lists_task_models_only(root, serving, monkeypatch):
    """The list's filters (`TASK_FILTERS`) by the server's one predicate; the
    clinical preset shows models finetuned for a task whatever the view asks,
    so a direct call with no view (or one asking for bases) lists no base or
    history file; each filter's empty state is said in its own words."""
    listing = _listing(root, monkeypatch)
    research, clinical = ml.PRESETS["research"], ml.PRESETS["clinical"]

    def shown(view, preset=research):
        return sorted(i["filename"] for i in ml.shown_items(listing, view, preset))

    assert shown({"task": "base"}) == ["base.pt"]
    assert shown({"task": "disease"}) == ["model-01.pt", "model-05.pt", "model-08.pt",
                                          "model-12.pt"]
    assert shown({"task": "gene"}) == ["model-03.pt"]
    assert shown({"task": "patients_like_me"}) == ["model-02.pt"]
    assert shown({"task": "history"}) == ["model-07.pt"]
    assert shown({"task": "task_models"}) == shown(None, clinical) == shown({"task": "all"},
                                                                            clinical)
    assert len(shown({"task": "all"})) == len(shown(None)) == len(shown({"task": "x"})) == 9
    assert "base.pt" not in shown(None, clinical) and "model-07.pt" not in shown(None, clinical)
    assert shown({"task": "base"}, clinical) == shown({"task": "history"}, clinical) == []
    assert ml.list_html(listing, None, clinical, view={"task": "base"}) == (
        '<div class="shep-reason">No base is listed in this workspace.</div>')
    no_gene = {**listing, "items": [i for i in listing["items"] if i.get("task") != "gene"]}
    assert "No gene model is listed in this workspace." in ml.list_html(
        no_gene, None, research, view={"task": "gene"})
    assert ml.count_line(no_gene, research, False, {"task": "gene"}) == (
        "no checkpoint shown · Gene only, of 8 listed")
    assert ml.count_line(listing, research, False, {"task": "patients_like_me"}).endswith(
        "· Patients-like-me only, of 9 listed")
    # Each count in its own number (K6 closing smoke: "1 checkpoints in 1 runs").
    assert ml.count_line(listing, research, False, {"task": "gene"}) == (
        "newest first · 1 checkpoint in 1 run · Gene only, of 9 listed")
    # The drawer's task filter is counted against what the drawer lists (its
    # six task models), never against the bases and history files it hides;
    # asking for every row it shows names no filter at all.
    assert ml.count_line(listing, clinical, False, {"task": "disease"}) == (
        "newest first · 4 checkpoints in 3 runs · Disease only, of 6 listed")
    assert ml.count_line(listing, clinical, False, {"task": "task_models"}) == ml.count_line(
        listing, clinical, False, {"task": "all"}) == ml.count_line(listing, clinical, False)


def test_the_list_redraws_when_the_page_changes_its_view(root, serving, monkeypatch):
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list("research", head=lambda: gr.Button("Full records"),
                                       foot=lambda: gr.Button("Compare"))
    (redraw,) = [fn for fn in demo.fns.values() if (handle.view._id, "change") in fn.targets]
    assert redraw.outputs == [handle.selected, handle.rows, handle.count, handle.facts,
                              handle.load_button, handle.reason, handle.note]
    listing = _listing(root, monkeypatch)
    broken = _key("broken.pt")
    key, rows, count, facts, load, reason, note = redraw.fn(listing, broken, False,
                                                            {"task": "disease"})
    assert key is None and "run_e4c2a9" not in rows and load["interactive"] is False
    assert reason == ""
    disease = _key("model-12.pt")
    key, _rows, _count, _facts, load, reason, _note = redraw.fn(listing, disease, False,
                                                                {"task": "disease"})
    assert key == disease and load["interactive"] is True and reason == ""
    assert handle.head_extra.value == "Full records" and handle.foot_extra.value == "Compare"
    assert handle.refresh["inputs"] == [handle.workspace, handle.show_older, handle.view,
                                        handle.listing, handle.selected]


def test_the_research_load_is_off_for_a_base_or_history_row_with_its_reason(
        root, serving, monkeypatch):
    """Research › Models lists every row; a base or a history file selected
    there turns Load off, its reason beside it (in the selection's, the
    redraw's and the refresh's answers), before anything is asked."""
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list("research")
    select = next(fn for fn in demo.fns.values() if fn.name == "_select")
    assert select.outputs[-2:] == [handle.load_button, handle.reason]
    listing = _listing(root, monkeypatch)
    for name, why in (("base.pt", ml.BASE_REFUSAL), ("model-07.pt", ml.HISTORY_REFUSAL)):
        out = select.fn(listing, False, None, gr.EventData(None, {"key": _key(name)}))
        assert out[0] == _key(name) and out[-2]["interactive"] is False
        assert out[-1] == f'<div class="shep-reason shep-reason-grey">{why}</div>'
    out = select.fn(listing, False, None, gr.EventData(None, {"key": _key("model-05.pt")}))
    assert out[-2]["interactive"] is True and out[-1] == ""
    operator_mode(monkeypatch)
    refreshed = handle.refresh["fn"]("demo", False, None, listing, _key("base.pt"),
                                     gradio_request())
    by = dict(zip(map(id, handle.refresh["outputs"]), refreshed, strict=True))
    assert by[id(handle.load_button)]["interactive"] is False
    assert ml.BASE_REFUSAL in by[id(handle.reason)]


def test_a_view_shaped_for_another_workspace_never_hides_this_workspaces_rows(
        root, serving, monkeypatch):
    """Choosing another workspace redraws the list (the shared refresh) with
    the page's view before the page shapes a new one. What a view names by key
    (a family view's rows, its groups and cells) was read from one workspace's
    listing, so on another workspace's listing it is left out and the view's
    filter, column and note apply: never "No checkpoint is listed in this
    workspace." over a workspace that holds checkpoints, nor another
    workspace's family titles. The same workspace's listing (the refresh after
    a Load) keeps the view whole."""
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list("research")
    elsewhere = "gatv2/20260901-000000_base_ffffff/base.pt"
    family = {"task": "disease", "keys": [elsewhere],
              "groups": [{"title": "Family from run 2026-09-01 00:00 · 1 file",
                          "keys": [elsewhere]}],
              "column": "Lineage", "cells": {elsewhere: "New run"}, "note": "Reloads."}
    operator_mode(monkeypatch)

    def refresh(view, previous):
        out = handle.refresh["fn"]("demo", False, view, previous, None, gradio_request())
        return dict(zip(map(id, handle.refresh["outputs"]), out, strict=True))

    by = refresh(family, {**ml.empty_listing(), "workspace": "other"})
    rows = by[id(handle.rows)]
    for name in ("model-12.pt", "model-08.pt", "model-05.pt", "model-01.pt"):
        assert f'data-key="{_key(name)}"' in rows, name
    assert "Family from run" not in rows and ">Lineage</th>" in rows
    assert by[id(handle.count)] == (
        "newest first · 4 checkpoints in 3 runs · Disease only, of 9 listed")
    assert by[id(handle.note)] == '<span class="shep-note">Reloads.</span>'
    # The same workspace: the view's own rows and groups.
    listing = by[id(handle.listing)]
    mine = {**family, "keys": [_key("model-05.pt")],
            "groups": [{"title": "Family from run 2026-10-01 00:00 · 1 file",
                        "keys": [_key("model-05.pt")]}]}
    rows = refresh(mine, listing)[id(handle.rows)]
    assert rows.count(" data-key=") == 1 and "Family from run 2026-10-01" in rows


def _row_of(html, name):
    """The row (a drawer's button, or a table's row) of the file `name`."""
    at = html.index(f'data-key="{_key(name)}"')
    start = max(html.rfind("<button", 0, at), html.rfind("<tr", 0, at))
    return html[start:html.index("</button>" if html[start:].startswith("<button") else "</tr>",
                                 at)]


def test_every_row_says_what_its_file_is_in_both_presets(root, serving, monkeypatch):
    """README §5 (map B7): every row leads with its kind word
    (`renderers.record_kind_text`), whatever the view or the order: the
    drawer's rows, which list every task's models together, and the research
    table's Model cell under any order (here, groups as By test score draws
    them), its history and unreadable files included."""
    listing = _listing(root, monkeypatch)
    rows = ml.list_html(listing, None, ml.PRESETS["clinical"], show_older=True)
    for name, kind in (("model-12.pt", "Disease"), ("model-05.pt", "Disease"),
                       ("model-03.pt", "Gene"), ("model-02.pt", "Patients-like-me")):
        row = _row_of(rows, name)
        assert (f'"><span class="shep-row-sub shep-row-kind">{kind}</span>'
                f'<span class="shep-row-main">') in row, (name, row)
    view = {"groups": [{"title": "Ordered by Mode C MRR (untruncated) on v1 · highest first",
                        "keys": [_key("model-07.pt")]}],
            "column": "Mode C MRR, v1 · regime", "cells": {}}
    table = ml.list_html(listing, None, ml.PRESETS["research"], view=view)
    for name, kind in (("base.pt", "Base"), ("model-07.pt", "History · never run"),
                       ("broken.pt", "Record not read"), ("model-03.pt", "Gene")):
        row = _row_of(table, name)
        assert f'"><td><div class="shep-sub shep-row-kind">{kind}</div>' in row, (name, row)


def test_the_research_listing_carries_the_family_view_from_its_one_listing(
        root, serving, monkeypatch):
    """`listing["family"]`: per filter, the server's family view
    (`list_family(listing=)`) of the same listing the rows came from, so the
    files are listed once; parent first, the base kept as a context row of a
    task's view; a parent that is not available says why with no server path
    (a file's problem may name one)."""
    from src.api.services import server_lists
    from src.utils.checkpoint_io import write_checkpoint

    # A child whose recorded parent run holds a file that cannot be read, in a
    # family of its own (its record names its own run as the family's root).
    run = "20261012-070000_disease_eeeeee"
    orphan = root["workspace"] / "runs" / "gatv2" / run / "checkpoints" / "model-04.pt"
    orphan.parent.mkdir(parents=True)
    write_checkpoint({**payload("disease", 4, root["fingerprint"],
                                {"sha256": "9" * 64, "run": RUNS[2][1]}), "family_id": run},
                     orphan)
    os.utime(orphan, (time.time() - 10 * 3600,) * 2)
    calls = []
    real = server_lists.list_checkpoints
    monkeypatch.setattr(server_lists, "list_checkpoints",
                        lambda *a, **k: calls.append(a) or real(*a, **k))
    listing = ml.list_models("demo", operator_request(monkeypatch), family=True)
    assert len(calls) == 1
    family = listing["family"]
    assert set(family) == {"all", "base", "disease", "gene", "patients_like_me"}
    disease = family["disease"]
    assert [(row["key"], row["depth"], row["context"]) for row in disease] == [
        (_key("base.pt"), 0, True), *((_key(name), 1, False) for name in (
            "model-12.pt", "model-08.pt", "model-05.pt", "model-01.pt")),
        (f"gatv2/{run}/model-04.pt", 0, False)]
    assert all(row["parent"] == _key("base.pt") for row in disease[1:5])
    assert [row["group"] for row in family["all"]][-2:] == ["history", "refused"]
    assert [row["key"] for row in family["base"]] == [_key("base.pt")]
    assert [row["key"] for row in family["gene"]] == [_key("base.pt"), _key("model-03.pt")]
    lost = disease[-1]
    assert lost["parent_state"] == "unavailable" and "could not be read" in lost[
        "parent_problem"]
    assert str(root["root"]) not in json.dumps(family)
    # The drawer's listing reads no family view.
    assert "family" not in ml.list_models("demo", operator_request(monkeypatch))
    # A parent's problem that names a server path (an OSError's words, say) is
    # redacted as the list's own problems are.
    import dataclasses

    named = f"[Errno 13] Permission denied: '{root['workspace']}/runs/gatv2/x/last.pt'"
    real_family = server_lists.list_family
    monkeypatch.setattr(server_lists, "list_family", lambda *a, **k: server_lists.Listing(tuple(
        dataclasses.replace(row, parent_problem=named) for row in real_family(*a, **k).items)))
    listing = ml.list_models("demo", operator_request(monkeypatch), family=True)
    assert str(root["root"]) not in json.dumps(listing["family"])
    assert "Permission denied" in listing["family"]["disease"][-1]["parent_problem"]


# --------------------------------------------------------------------------- the wiring
def _component(preset="clinical", both=False):
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list(preset)
        other = ctx.create_model_list("research") if both else None
    return demo, ctx, handle, other


def _named(demo, name, handle=None):
    found = [fn for fn in demo.fns.values() if fn.name == name
             and (handle is None or set(map(id, fn.outputs)) & {id(handle.reason),
                                                                 id(handle.phase)})]
    assert len(found) == 1, (name, len(found))
    return found[0]


def _after(demo, fn):
    nxt = [d for d in demo.fns.values() if d.trigger_after == fn._id]
    assert len(nxt) == 1, fn.name
    return nxt[0]


def test_load_and_reload_refresh_the_chips_and_reload_sets_the_trigger(root, serving, monkeypatch):
    from src.webui.components.chips import read_chips

    demo, ctx, handle, _ = _component()
    request = _named(demo, "_request")
    assert handle.phase in request.outputs and handle.plan in request.outputs
    chips = _after(demo, request)
    assert chips.fn is read_chips and chips.outputs == [ctx.chips]
    loading = _named(demo, "_loading")
    assert loading.outputs == [handle.phase, handle.reload_button, handle.keep_button]
    assert loading.queue is False
    assert '"shep-models-clinical-confirm"' in loading.js
    assert '"shep-models-clinical-reload"' in loading.js        # Reload off at once, in the page
    confirm = _after(demo, loading)
    assert confirm.name == "_confirm" and handle.reload_done is not None
    assert ctx.served_model_changed in confirm.outputs and handle.phase in confirm.outputs
    answered = _after(demo, confirm)                    # the page's step (`load_answered_js`)
    assert answered.fn is None and answered.js == ml.load_answered_js(handle.elem_id)
    chips = _after(demo, answered)
    assert chips.fn is read_chips
    assert _after(demo, chips).fn is handle.refresh["fn"]


def test_confirm_returns_the_phase_and_the_served_model_changed_token(root, serving, monkeypatch):
    import asyncio

    from src.api.routes import pipeline as pipeline_routes
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    demo, ctx, handle, _ = _component()
    confirm = _named(demo, "_confirm")
    phase_at = confirm.outputs.index(handle.phase)
    token_at = confirm.outputs.index(ctx.served_model_changed)
    plan = _plan(root, monkeypatch)
    monkeypatch.setattr(pipeline_routes, "run_reload", lambda request: PipelineReloadResponse(
        success=True, message="ok", status=PipelineStatusResponse(
            **_status(root, served="model-12.pt"))))
    out = asyncio.run(confirm.fn(plan, operator_request(monkeypatch)))
    assert out[0] is None and out[phase_at] == "idle"
    assert out[token_at].startswith(root["shas"]["model-12.pt"] + "@")
    # shown again (what is served changed): still confirming, no token
    serving["value"] = _status(root)
    shown = _plan(root, monkeypatch)
    serving["value"] = _status(root, served="model-08.pt")
    out = asyncio.run(confirm.fn(shown, operator_request(monkeypatch)))
    assert out[phase_at] == "confirming" and out[token_at] == gr.update()
    # an unexpected error never leaves the page "loading"
    monkeypatch.setattr(ml, "finish_load", lambda *a, **k: (_ for _ in ()).throw(KeyError("x")))
    out = asyncio.run(confirm.fn(plan, operator_request(monkeypatch)))
    assert out[phase_at] == "idle" and "The Load stopped with an error." in out[2]


def test_while_a_reload_runs_the_confirmation_shows_only_its_status_and_close():
    """M5-3 (the mockup's footer after "Reload with …"): Reload and Keep were
    still drawn while the reload ran, and a second click could ask for a second
    reload. The step before the reload hides Reload and makes Keep "Close"
    (closing does not stop it); the page's script disables Reload at once, and
    counts in any element marked for this load (Diagnosis' card)."""
    phase, reload_, keep = ml.loading_buttons({"to": {"sha256": "a" * 64}})
    assert phase == "loading" and reload_["visible"] is False and keep["value"] == "Close"
    assert ml.loading_buttons(None) == ("idle", gr.update(), gr.update())
    script = ml.start_wait_js("x-confirm", "x-reload")
    assert 'getElementById("x-reload")' in script and "again.disabled = true" in script
    assert "data-shep-load-elapsed" in script and "data-shep-load-card" in script
    # A page's own outputs ride on the same step, before the reload holds the server.
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        extra_in, extra_out = gr.State("s"), gr.HTML("")
        seen = []
        handle = ctx.create_model_list(
            "clinical", on_loading={"fn": lambda plan, s: seen.append((plan, s)) or ("line",),
                                    "inputs": [extra_in], "outputs": [extra_out]})
    loading = _named(demo, "_loading")
    assert loading.inputs == [handle.plan, extra_in]
    assert loading.outputs == [handle.phase, handle.reload_button, handle.keep_button, extra_out]
    plan = {"to": {"sha256": "b" * 64}}
    out = loading.fn(plan, "s")
    assert out[0] == "loading" and out[2]["value"] == "Close" and out[3] == "line"
    assert seen == [(plan, "s")]
    assert loading.fn(None, "s")[0] == "idle" and seen[-1] == (None, "s")


def test_a_close_while_the_reload_runs_is_done_in_the_page_and_sent_after_its_answer():
    """M5-3 review: the reload holds the server's event loop, so Close (and
    the drawer's ×) did nothing until the reload answered; then its close and
    the answer came back in no fixed order, and the box could be left open,
    empty, with "Reload with <the model now served>". While a reload runs the
    page closes it at once (`load_guard`) and clicks Close again only after
    the answer (`load_answered_js`), so the server handles it in order; the
    page's own hiding goes when that close is done (`load_closed_js`)."""
    assert ml.load_guard("x") == {"list": "x", "closers": ["#x-keep"], "hide": "#x-confirm-box",
                                  "then": "#x-keep", "openers": ["#x-load"]}
    drawer = {"hide": "#d", "then": "#d-x button", "also": ["#d-x button", "#d-cancel"],
              "openers": ["#open"]}
    assert ml.load_guard("x", drawer) == {"list": "x", "closers": ["#x-keep", "#d-x button",
                                                                   "#d-cancel"],
                                          "hide": "#d", "then": "#d-x button",
                                          "openers": ["#open"]}
    demo, _ctx, handle, _ = _component()
    assert handle.confirm.elem_id == "shep-models-clinical-confirm-box"
    loading = _named(demo, "_loading")
    from src.webui.components.help import as_json_for_script

    assert as_json_for_script(ml.load_guard("shep-models-clinical")) in loading.js
    for part in ("document.addEventListener('click'", "e.stopImmediatePropagation()",
                 "'shep-load-closed-' + id", "{ display: none !important; }",
                 ":not(#shep-load-x1):not(#shep-load-x2):not(#shep-load-x3)",
                 "g.openers.some(hit)"):
        assert part in loading.js, part
    # A close and an opener are both held from the page's own handlers.
    assert loading.js.count("e.preventDefault(); e.stopImmediatePropagation();") == 2
    answered = _after(demo, _after(demo, loading))
    assert answered.fn is None and "st.busy = false" in answered.js
    assert "document.querySelector(st.guard.then)" in answered.js and "el.click()" in answered.js
    assert '"shep-models-clinical"' in answered.js
    # Close's own chain ends with the page's step that drops its hiding.
    done = _after(demo, _after(demo, _named(demo, "_keep")))
    assert done.fn is None and done.js == ml.load_closed_js(handle.elem_id) == handle.after_close_js
    assert "!st.busy" in done.js and "css.remove()" in done.js
    # A page that puts the list in a drawer names it, and its own closes.
    from src.webui.shell import standalone_context

    with gr.Blocks() as page:
        ctx = standalone_context(page)
        listed = ctx.create_model_list("clinical", close_while_loading=drawer)
    script = _named(page, "_loading").js
    assert as_json_for_script(ml.load_guard(listed.elem_id, drawer)) in script


def test_keep_closes_the_confirmation(root, serving, monkeypatch):
    demo, _ctx, handle, _ = _component()
    keep = _named(demo, "_keep")
    out = dict(zip(map(id, keep.outputs), keep.fn(), strict=True))
    assert out[id(handle.plan)] is None and out[id(handle.confirm)]["visible"] is False
    assert out[id(handle.phase)] == "idle"


def test_keep_changes_nothing_inside_the_box_it_hides_and_resets_it_next():
    """A response that hides a column and updates a component inside it
    leaves detached elements and listeners in the browser on every cycle
    (Gradio 6.30, measured: about 113 elements and 2 listeners per
    Load-then-Keep). Keep hides the box; the step after it empties the box and
    puts Reload back, unless a Load has opened a confirmation meanwhile."""
    demo, _ctx, handle, _ = _component()
    keep = _named(demo, "_keep")
    inside = {id(handle.confirm_html), id(handle.reload_button), id(handle.keep_button)}
    assert id(handle.confirm) in set(map(id, keep.outputs))
    assert not inside & set(map(id, keep.outputs))
    after = _after(demo, keep)
    assert after.name == "_after_keep" and after.inputs == [handle.plan]
    assert after.outputs == [handle.confirm_html, handle.reload_button]
    html_, reload_ = after.fn(None)
    assert html_ == "" and reload_["visible"] is True
    assert after.fn({"to": {"sha256": "a" * 64}}) == (gr.update(), gr.update())


def test_both_presets_sit_in_one_page_each_counting_in_its_own_confirmation():
    demo, _ctx, clinical, research = _component(both=True)
    assert (clinical.elem_id, research.elem_id) == ("shep-models-clinical", "shep-models-research")
    ids = [getattr(b, "elem_id", None) for b in demo.blocks.values() if getattr(b, "elem_id", None)]
    assert len(ids) == len(set(ids))
    scripts = sorted(fn.js for fn in demo.fns.values() if fn.name == "_loading")
    assert len(scripts) == 2
    assert '"shep-models-clinical-confirm"' in scripts[0] + scripts[1]
    # Each names only its own list's confirmation and Reload button.
    import re

    for script in scripts:
        named = re.findall(r'"(shep-models-[a-z]+)-(confirm|reload)"', script)
        assert sorted(kind for _p, kind in named) == ["confirm", "reload"]
        assert len({prefix for prefix, _k in named}) == 1
    assert "document.querySelector('.shep-model-confirm')" not in "".join(scripts)


# --------------------------------------------------------------------------- through Gradio
def test_a_direct_call_through_gradios_api_meets_the_same_check(root, serving, monkeypatch):
    """The list's events, called as `gradio_client` would: refused off
    loopback, listed for an operator. No workspace is posted (the dropdown has
    no choices until the first read), so the server picks the serving one."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list("clinical")
    app = FastAPI()
    gr.mount_gradio_app(app, demo, path="/ui")
    refresh = next(d for d in demo.fns.values() if d.name == "_refresh")
    operator_mode(monkeypatch)

    def call(base_url):
        with TestClient(app, base_url=base_url) as client:
            started = client.post(f"/ui/gradio_api/call/{refresh.api_name}",
                                  json={"data": [None, False, None, None, None]})
            assert started.status_code == 200, started.text
            stream = client.get(f"/ui/gradio_api/call/{refresh.api_name}/"
                                f"{started.json()['event_id']}")
        lines = stream.text.splitlines()
        return json.loads(lines[lines.index("event: complete") + 1][len("data:"):])

    # The outputs a client sees: the listing itself is session state (not
    # returned), then the workspace choice, the count, the rows, the facts.
    refused = call("http://192.0.2.10")
    assert "Operator controls only: request not local." in json.dumps(refused)
    assert "disease_8c2e7a" not in json.dumps(refused)
    listed = call(OPERATOR_BASE_URL)
    _state, _selected, workspace, count, rows = listed[:5]
    assert workspace["value"] == "demo"
    assert "5 checkpoints (of 6 in 5 runs)" in count and "listed" not in count
    assert "run 2026-10-18 09:12 · epoch 12" in rows
    assert handle.load_button is not None and handle.refresh["fn"] is refresh.fn


def test_the_list_shows_no_absolute_server_path(root, serving, monkeypatch, tmp_path):
    """M5-3 review SECURITY-3: a workspace that is a link out of the root was
    refused with its absolute target and the root's path, which the list
    printed escaped but not redacted (Datasets redacts the same reason)."""
    outside = tmp_path / "outside" / "elsewhere_ws"
    outside.mkdir(parents=True)
    (root["root"] / "leaving_ws").symlink_to(outside)
    listing = ml.list_models("leaving_ws", operator_request(monkeypatch))
    assert listing["workspace"] == "leaving_ws" and listing["items"] == []
    shown = (ml.list_html(listing, None, ml.PRESETS["clinical"])
             + ml.count_line(listing, ml.PRESETS["clinical"], False) + repr(listing["problem"]))
    assert listing["problem"] and "outside the workspaces root" in listing["problem"]
    assert str(tmp_path) not in shown and str(root["root"]) not in shown, shown


def test_a_selection_made_after_a_load_survives_the_lists_re_read(root, serving, monkeypatch):
    """M5-3 review UI-3: the re-read chained after every Load reset the list,
    dropping a row picked once the outcome was closed (its facts gone, Load
    disabled, nothing said). A selection still listed in the same workspace
    stays; one of another workspace, or no longer listed, does not."""
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list("clinical")
    refresh = handle.refresh
    assert refresh["inputs"][-2:] == [handle.listing, handle.selected]
    operator_mode(monkeypatch)
    request = gradio_request()
    before = ml.list_models("demo", request)
    picked = _key("model-12.pt")
    out = refresh["fn"]("demo", False, None, before, picked, request)
    by = dict(zip(map(id, refresh["outputs"]), out, strict=True))
    assert by[id(handle.selected)] == picked
    assert by[id(handle.load_button)]["interactive"] is True
    assert 'aria-pressed="true"' in by[id(handle.rows)]
    assert "Select a model" not in by[id(handle.facts)]
    elsewhere = ({**before, "workspace": "other"}, picked)
    gone = (before, "gatv2/x/gone.pt")
    for previous, key in (elsewhere, gone):
        out = refresh["fn"]("demo", False, None, previous, key, request)
        by = dict(zip(map(id, refresh["outputs"]), out, strict=True))
        assert by[id(handle.selected)] is None
        assert by[id(handle.load_button)]["interactive"] is False
