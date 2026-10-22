"""
The shared model list and the one Load (M5-2, UI-A).
====================================================
`src/webui/components/model_list.py`. What these pin:

- **listing opens checkpoints, so it is an operator control**: D8's helper on
  the triggering request first; refused, nothing is listed or opened, and the
  reason is said; a direct call through Gradio's HTTP API meets the same check;
- **the rows**: "In use" pinned on top by the **full** model fingerprint the
  status serves, marking the checkpoint only; the clinical preset shows the
  latest 4 runs then "Show older runs"; the research preset every run; an
  unreadable checkpoint greyed; each row's recorded tests a state read from
  the ledger beside the checkpoint and, for contested, the report store;
- **the presets and the page's view**: the research preset is a table with
  "In use (Diagnosis)" on its own row, not pinned; a page's view lists one
  architecture, draws its groups in its order (the rest after them), fills one
  extra column, and puts a note beside Load; the list redraws when the view
  changes;
- **the Load**: D8 on the "Load" request, the item found in the server's list
  again and its bytes hashed again (a list read earlier is caught), one
  confirmation (from → to, the changed rows marked, what it replaces); then D8
  again on "Reload with …", **both columns read again**: a change of what is
  served (any part of the served identity or the served workspace, the From
  column rebuilt from the plan's own listing) or of what would be loaded (the
  file's bytes, the workspace's graph or SP table) is said with the new values
  and nothing is reloaded; WP2's reload with the workspace and the checkpoint
  the server resolves (never a client path); the served fingerprint checked
  against the one shown; "served model changed" set when any part of what is
  served changed; a usage history entry only when the list was ordered by a
  cohort version's score, and not when that version changed since;
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

#: (architecture, run, files) newest run first, as the listing orders them.
RUNS = [
    ("gat", "20261018-091200_run_8c2e7a", ("model-12.pt", "model-08.pt")),
    ("gat", "20261017-091500_run_3fa9c1", ("model-05.pt",)),
    ("sage", "20261016-083000_run_e4c2a9", ("broken.pt",)),
    ("gat", "20261015-080000_run_aaaaaa", ("model-03.pt",)),
    ("gat", "20261014-080000_run_bbbbbb", ("model-02.pt",)),
    ("gat", "20261013-080000_run_cccccc", ("model-01.pt",)),
]


@pytest.fixture(scope="module")
def demo_workspace(tmp_path_factory):
    from tests.fixtures.registered_cohort import write_demo_workspace

    return write_demo_workspace(tmp_path_factory.mktemp("mlbuild") / "demo")


@pytest.fixture
def root(demo_workspace, tmp_path, monkeypatch):
    """A workspaces root with one workspace and six runs of checkpoints."""
    from src.utils.checkpoint_io import write_checkpoint

    root = tmp_path / "root"
    workspace = shutil.copytree(demo_workspace, root / "demo")
    fingerprint = file_sha256(workspace / "split_manifest.json")
    now = time.time()
    shas = {}
    for index, (arch, run, files) in enumerate(RUNS):
        directory = workspace / "runs" / arch / run / "checkpoints"
        directory.mkdir(parents=True)
        for offset, name in enumerate(files):
            path = directory / name
            if name == "broken.pt":
                path.write_bytes(b"not a checkpoint")
            else:
                epoch = int(name.split("-")[1].split(".")[0])
                shas[name] = write_checkpoint(
                    {"checkpoint_kind": "weights", "epoch": epoch,
                     "logs": {"val_mrr": 0.3 + epoch / 100},
                     "training_input_digests": {"split_manifest": fingerprint}}, path)
            stamp = now - index * 3600 - offset * 60
            os.utime(path, (stamp, stamp))
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    monkeypatch.delenv("SHEPHERD_REGISTRY", raising=False)
    return {"root": root, "workspace": workspace, "fingerprint": fingerprint, "shas": shas}


def _status(root, served="model-05.pt", data_dir=None):
    sha = root["shas"].get(served)
    kg = json.loads((root["workspace"] / "split_manifest.json").read_text())["artifacts"]["kg"]
    return {
        "initialized": True, "current_data_dir": str(data_dir or root["workspace"]),
        "eta_configured": 0.7, "checkpoint_meta": {"epoch": 5},
        "served_identity": {
            "scoring_mode": "gnn_only",
            "checkpoint": {"state": "verified" if sha else "absent", "sha256": sha},
            "graph": {"state": "verified", "kg_sha256": kg,
                      "split_manifest_sha256": root["fingerprint"]},
            "shortest_paths": {"state": "absent", "build_id": None, "max_hops": None},
            "dataset_record": {"state": "verified", "origin": "synthetic", "sha256": "e" * 64},
            "settings": {"device": "cpu", "device_requested": "cpu", "eta": 1.0,
                         "max_path_length": 4, "path_length_penalty": 0.5,
                         "aggregation_method": "max", "max_paths_per_candidate": 10,
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
    assert len(listing["items"]) == 7
    html = ml.list_html(listing, None, ml.PRESETS["clinical"])
    first_row = html.index('class="shep-row')
    assert html.index("In use now") < first_row
    assert html[first_row:].startswith('class="shep-row"') or "epoch 5" in html[first_row:first_row + 400]
    assert html.count('shep-tag-accent">In use<') == 1
    assert "in use, pinned above" in html
    assert "metadata not readable · cannot be loaded" in html
    # the clinical preset: the latest four runs, then the rest on request
    assert "Show older runs (2)" in html and "run_cccccc" not in html
    older = ml.list_html(listing, None, ml.PRESETS["clinical"], show_older=True)
    assert "run_cccccc" in older and "Show older runs" not in older
    research = ml.list_html(listing, None, ml.PRESETS["research"])
    assert "run_cccccc" in research and 'In use (Diagnosis)<' in research
    assert ml.count_line(listing, ml.PRESETS["clinical"], False) == (
        "newest first · latest 4 runs · 5 checkpoints (of 7 in 6 runs)")


def test_in_use_is_matched_by_the_full_fingerprint_only(root, serving, monkeypatch):
    sha = root["shas"]["model-05.pt"]
    serving["value"]["served_identity"]["checkpoint"]["sha256"] = sha[:12] + "f" * 52
    listing = ml.list_models("demo", operator_request(monkeypatch))
    html = ml.list_html(listing, None, ml.PRESETS["clinical"])
    assert "In use now" not in html and "In use<" not in html


def test_each_rows_recorded_tests_come_from_the_ledger_beside_it(root, serving, monkeypatch):
    from src.evaluation.sidecar import empty_ledger, write_ledger

    checkpoints = root["workspace"] / "runs" / "gat"
    write_ledger(checkpoints / RUNS[0][1] / "checkpoints" / "evaluations.json",
                 empty_ledger(), None)
    (checkpoints / RUNS[1][1] / "checkpoints" / "evaluations.json").write_text("{not json")
    listing = ml.list_models("demo", operator_request(monkeypatch))
    recorded = listing["recorded"]
    assert recorded[f"gat/{RUNS[0][1]}/model-12.pt"]["state"] == "none"
    assert recorded[f"gat/{RUNS[1][1]}/model-05.pt"]["state"] == "not_read"
    assert recorded[f"gat/{RUNS[3][1]}/model-03.pt"]["state"] == "no_ledger"
    assert f"sage/{RUNS[2][1]}/broken.pt" not in recorded          # unreadable: no state


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

    monkeypatch.setattr(ml.CohortVersions, "listed",
                        classmethod(lambda cls, environ=None: Versions()))
    states = {"1" * 64: "contested", "2" * 64: "uncontested"}
    monkeypatch.setattr(report_store, "scan_report_store",
                        lambda version: Scan(states[version.identity.sha256]))
    item = {"architecture": "gat", "run": RUNS[0][1], "filename": "model-12.pt",
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
    key = f"gat/{RUNS[0][1]}/model-12.pt"
    monkeypatch.setattr(server_lists, "resolve_checkpoint",
                        lambda *a, **k: pytest.fail("a refused Load resolved a checkpoint"))
    plan, html = ml.request_load(listing, key, None, gradio_request(server=("192.0.2.10", 8264)))
    assert plan is None and "Refused: " in html


def test_load_shows_one_confirmation_from_to_with_the_changed_rows_marked(
        root, serving, monkeypatch):
    listing = _listing(root, monkeypatch)
    key = f"gat/{RUNS[0][1]}/model-12.pt"
    plan, html = ml.request_load(listing, key, None, operator_request(monkeypatch))
    new, old = root["shas"]["model-12.pt"][:12], root["shas"]["model-05.pt"][:12]
    assert plan["to"]["sha256"] == root["shas"]["model-12.pt"]
    assert plan["from"]["sha256"] == root["shas"]["model-05.pt"]
    assert f"Reload the pipeline with <code>{new}</code> for every session?" in html
    assert ml.REPLACES_NOTE in html
    changed = [row for row in html.split("<tr") if 'class="shep-changed"' in row]
    labels = [row.split("<th scope=\"row\">")[1].split("<")[0] for row in changed]
    assert labels == ["Model", "Model fingerprint"]
    assert "GAT · run 2026-10-17 09:15 · epoch 5" in html and old in html
    assert "same dataset" in html and ">same<" in html
    assert "absent" in html                                    # SP table, both sides
    # not a client path: the plan names the item, never a file
    assert all("/" not in str(plan[k]) or k == "run" for k in ("workspace", "filename"))


def test_an_unloadable_item_cannot_be_loaded(root, serving, monkeypatch):
    listing = _listing(root, monkeypatch)
    plan, html = ml.request_load(listing, f"sage/{RUNS[2][1]}/broken.pt", None,
                                 operator_request(monkeypatch))
    assert plan is None and "Select a model that can be loaded" in html
    plan, html = ml.request_load(listing, "../../../etc/passwd", None,
                                 operator_request(monkeypatch))
    assert plan is None


def _plan(root, monkeypatch, ordering=None):
    listing = _listing(root, monkeypatch)
    plan, _html = ml.request_load(listing, f"gat/{RUNS[0][1]}/model-12.pt", ordering,
                                  operator_request(monkeypatch))
    return plan


def test_confirm_needs_d8_again_and_a_refusal_reloads_nothing(root, serving, monkeypatch):
    plan = _plan(root, monkeypatch)
    calls = []
    step = ml.finish_load(plan, gradio_request(headers=[("Origin", "https://evil.example")]),
                          reload=lambda request: calls.append(request))
    assert calls == [] and "Refused: " in step.reason_html and step.plan == plan
    assert step.changed is False


def _changed_rows(html):
    rows = [row for row in html.split("<tr") if 'class="shep-changed"' in row]
    return [row.split('<th scope="row">')[1].split("<")[0] for row in rows]


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
    assert step.plan["from"]["model"] == "GAT · run 2026-10-18 09:12 · epoch 8"
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
    plan, _html = ml.request_load(listing, f"gat/{RUNS[1][1]}/model-05.pt", None,
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


def _rewrite(root, name, epoch):
    """Write other bytes into a listed checkpoint, as a training run that
    overwrites ``last.pt`` each epoch does."""
    from src.utils.checkpoint_io import write_checkpoint

    path = root["workspace"] / "runs" / "gat" / RUNS[0][1] / "checkpoints" / name
    return write_checkpoint({"checkpoint_kind": "weights", "epoch": epoch,
                             "logs": {"val_mrr": 0.5},
                             "training_input_digests": {"split_manifest": root["fingerprint"]}},
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


def test_a_workspace_whose_table_appeared_after_the_confirmation_is_shown_again(
        root, serving, monkeypatch):
    plan = _plan(root, monkeypatch)
    (root["workspace"] / "shortest_paths.pt").write_bytes(b"a table")
    calls = []
    step = ml.finish_load(plan, operator_request(monkeypatch),
                          reload=lambda request: calls.append(request))
    assert calls == [] and ml.TO_CHANGED in step.confirm_html
    assert step.plan["to"]["sp"] is True and step.plan["to"]["mode"] == "gnn_plus_shortest_path"


def test_a_list_read_earlier_is_caught_when_load_is_asked(root, serving, monkeypatch):
    listing = _listing(root, monkeypatch)
    new_sha, _path = _rewrite(root, "model-12.pt", 14)
    plan, html = ml.request_load(listing, f"gat/{RUNS[0][1]}/model-12.pt", None,
                                 operator_request(monkeypatch))
    assert plan["to"]["sha256"] == new_sha and plan["to"]["model"].endswith("epoch 14")
    assert "The file changed since the list was read" in html and new_sha[:12] in html


def test_load_finds_the_item_in_the_servers_list_again(root, serving, monkeypatch):
    """A listed item whose file is gone since the list was read is refused,
    and nothing is opened from the listing's word for it."""
    listing = _listing(root, monkeypatch)
    (root["workspace"] / "runs" / "gat" / RUNS[0][1] / "checkpoints" / "model-12.pt").unlink()
    plan, html = ml.request_load(listing, f"gat/{RUNS[0][1]}/model-12.pt", None,
                                 operator_request(monkeypatch))
    assert plan is None and "Refused: " in html and "is listed in the workspace" in html


def test_the_reload_gets_what_the_server_resolves_and_the_served_fingerprint_is_checked(
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
    expected = (root["workspace"] / "runs" / "gat" / RUNS[0][1] / "checkpoints" / "model-12.pt")
    assert Path(request.data_dir) == root["workspace"].resolve()
    assert Path(request.checkpoint_path) == expected
    assert step.changed is True and step.served == root["shas"]["model-12.pt"]
    assert "Reloaded." in step.confirm_html and step.plan is None


def test_other_bytes_served_than_shown_are_said(root, serving, monkeypatch):
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    plan = _plan(root, monkeypatch)
    step = ml.finish_load(plan, operator_request(monkeypatch), reload=lambda request: (
        PipelineReloadResponse(success=True, message="ok", status=PipelineStatusResponse(
            **_status(root, served="model-08.pt")))))
    assert "Reloaded, but not with the model shown." in step.confirm_html
    assert step.changed is True


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


def test_the_usage_history_is_written_only_for_a_list_ordered_by_a_versions_score(
        root, serving, monkeypatch, tmp_path):
    import src.evaluation.cohort_registry as cohort_registry
    import src.evaluation.report_store as report_store
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse
    from src.config import server_settings

    written = []
    monkeypatch.setattr(report_store, "record_model_load",
                        lambda version, **fields: written.append(fields))
    version = type("V", (), {"identity": type("I", (), {"sha256": "c" * 64})()})()
    monkeypatch.setattr(cohort_registry, "resolve_version", lambda registry, label: version)
    registry = tmp_path / "registry"
    registry.mkdir()
    monkeypatch.setattr(server_settings, "cohort_registry",
                        lambda environ=None: server_settings.Setting("SHEPHERD_REGISTRY", registry))

    def reload(request):
        return PipelineReloadResponse(success=True, message="ok", status=PipelineStatusResponse(
            **_status(root, served="model-12.pt")))

    unordered = ml.finish_load(_plan(root, monkeypatch), operator_request(monkeypatch),
                               reload=reload)
    # an unordered list: no entry is tried, so none is said to be missing
    assert written == [] and "usage history" not in unordered.confirm_html
    serving["value"] = _status(root)
    ordering = {"version": "mygene2-2026-09", "version_sha256": "c" * 64,
                "metric": "untruncated_mrr", "scoring_semantics_digest": "s" * 64}
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


def test_a_markup_workspace_name_is_escaped_in_the_confirmation(root, serving, monkeypatch):
    listing = _listing(root, monkeypatch)
    plan, _html = ml.request_load(listing, f"gat/{RUNS[0][1]}/model-12.pt", None,
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
                   "Validation MRR (Trainer, top-20)", "Recorded tests"):
        assert f">{column}</th>" in research, column
    served_row = next(row for row in research.split("<tr") if "run_3fa9c1" in row)
    assert "In use (Diagnosis)" in served_row and "0.3500" in served_row
    assert research.count("In use (Diagnosis)") == 1
    assert "metadata not readable · cannot be loaded" in research
    assert research.count(" data-key=") == 7


def test_a_pages_view_orders_filters_and_adds_a_column(root, serving, monkeypatch):
    listing = _listing(root, monkeypatch)
    keys = [ml.item_key(i) for i in listing["items"]]
    first, second = f"gat/{RUNS[3][1]}/model-03.pt", f"gat/{RUNS[0][1]}/model-12.pt"
    view = {"architecture": "gat",
            "groups": [{"title": "Ordered by Mode C MRR (untruncated) on v1 · highest first",
                        "keys": [first, second, "gat/nope/x.pt"]},
                       {"title": "Contested on this version · not ordered", "keys": []}],
            "rest_title": "Not tested on this cohort version · newest first",
            "column": "Mode C MRR, v1 · regime",
            "cells": {first: '0.2310<div class="shep-sub">CUDA · deterministic</div>'},
            "note": "The usage history records <ordered by …>."}
    html = ml.list_html(listing, None, ml.PRESETS["research"], view=view)
    ordered = html.index("Ordered by Mode C MRR")
    assert ordered < html.index("run_aaaaaa") < html.index("run_8c2e7a") < html.index(
        "Contested on this version") < html.index("Not tested on this cohort version")
    assert "run_e4c2a9" not in html                       # the SAGE run is filtered out
    assert ">Mode C MRR, v1 · regime</th>" in html and "CUDA · deterministic" in html
    assert ml.count_line(listing, ml.PRESETS["research"], False, view) == (
        "in the groups below · 6 checkpoints in 5 runs · GAT only, of 7 listed")
    assert ml.note_html(view) == ('<span class="shep-note">The usage history records '
                                  '&lt;ordered by …&gt;.</span>')
    # the clinical rows draw a page's cell as one more line of the row
    rows = ml.list_html(listing, None, ml.PRESETS["clinical"],
                        view={"cells": {second: "<b>regime</b>"}})
    assert '<span class="shep-row-sub"><b>regime</b></span></button>' in rows
    assert len(keys) == 7


def test_the_list_redraws_when_the_page_changes_its_view(root, serving, monkeypatch):
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list("research", head=lambda: gr.Button("Full records"),
                                       foot=lambda: gr.Button("Compare"))
    (redraw,) = [fn for fn in demo.fns.values() if (handle.view._id, "change") in fn.targets]
    assert redraw.outputs == [handle.selected, handle.rows, handle.count, handle.facts,
                              handle.load_button, handle.note]
    listing = _listing(root, monkeypatch)
    sage = f"sage/{RUNS[2][1]}/broken.pt"
    key, rows, count, facts, load, note = redraw.fn(listing, sage, False, {"architecture": "gat"})
    assert key is None and "run_e4c2a9" not in rows and load["interactive"] is False
    gat = f"gat/{RUNS[0][1]}/model-12.pt"
    key, *_rest = redraw.fn(listing, gat, False, {"architecture": "gat"})
    assert key == gat
    assert handle.head_extra.value == "Full records" and handle.foot_extra.value == "Compare"
    assert handle.refresh["inputs"] == [handle.workspace, handle.show_older, handle.view,
                                        handle.listing, handle.selected]


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
    assert "run_8c2e7a" not in json.dumps(refused)
    listed = call(OPERATOR_BASE_URL)
    _state, _selected, workspace, count, rows = listed[:5]
    assert workspace["value"] == "demo" and "of 7 in 6 runs" in count
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
    picked = f"gat/{RUNS[0][1]}/model-12.pt"
    out = refresh["fn"]("demo", False, None, before, picked, request)
    by = dict(zip(map(id, refresh["outputs"]), out, strict=True))
    assert by[id(handle.selected)] == picked
    assert by[id(handle.load_button)]["interactive"] is True
    assert 'aria-pressed="true"' in by[id(handle.rows)]
    assert "Select a model" not in by[id(handle.facts)]
    elsewhere = ({**before, "workspace": "other"}, picked)
    gone = (before, "gat/x/gone.pt")
    for previous, key in (elsewhere, gone):
        out = refresh["fn"]("demo", False, None, previous, key, request)
        by = dict(zip(map(id, refresh["outputs"]), out, strict=True))
        assert by[id(handle.selected)] is None
        assert by[id(handle.load_button)]["interactive"] is False
