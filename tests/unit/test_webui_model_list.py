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
- **the Load**: D8 on the "Load" request, the item found in the server's list
  again, one confirmation (from → to, the changed rows marked, what it
  replaces); then D8 again on "Reload with …", the status read again and a
  model in use that changed since said, never reloaded over; WP2's reload with
  the workspace and the checkpoint the server resolves (never a client path);
  the served fingerprint checked against the one shown; a usage history entry
  only when the list was ordered by a cohort version's score;
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
    assert "Refused:" in ml.list_html(listing, None, ml.PRESETS["clinical"])


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


def test_contested_is_counted_from_the_report_store_and_unchecked_is_not_zero(
        root, monkeypatch, tmp_path):
    import src.evaluation.sidecar as sidecar
    from src.config import server_settings

    sha = root["shas"]["model-12.pt"]
    records = [{"checkpoint_digest": sha, "cohort_role": "research_selection",
                "cohort_digest": d, "measurement_semantics_digest": "m" * 64}
               for d in ("1" * 64, "1" * 64, "2" * 64)]
    monkeypatch.setattr(sidecar, "read_ledger", lambda path: ({"records": records}, "x" * 64))
    monkeypatch.setattr(sidecar, "records_for", lambda ledger, digest: ledger["records"]
                        if digest == sha else [])

    class Version:
        def __init__(self, digest):
            self.identity = type("I", (), {"sha256": digest})()

    class Listed:
        def __init__(self, digest):
            self.version = Version(digest)

    class Scan:
        def __init__(self, state):
            self._state = state

        def state(self, key):
            return self._state

    registry = tmp_path / "registry"
    registry.mkdir()
    monkeypatch.setattr(server_settings, "cohort_registry",
                        lambda environ=None: server_settings.Setting("SHEPHERD_REGISTRY", registry))
    import src.evaluation.cohort_registry as cohort_registry
    import src.evaluation.report_store as report_store

    monkeypatch.setattr(cohort_registry, "list_versions",
                        lambda path: [Listed("1" * 64), Listed("2" * 64)])
    states = {"1" * 64: "contested", "2" * 64: "uncontested"}
    monkeypatch.setattr(report_store, "scan_report_store",
                        lambda version: Scan(states[version.identity.sha256]))
    item = {"architecture": "gat", "run": RUNS[0][1], "filename": "model-12.pt",
            "sha256": sha, "problem": None}
    result = ml.recorded_tests_for(root["workspace"], [item])[ml.item_key(item)]
    assert (result.state, result.count, result.contested) == ("records", 3, 2)
    states["2" * 64] = "cannot_be_checked"
    result = ml.recorded_tests_for(root["workspace"], [item])[ml.item_key(item)]
    assert (result.count, result.contested) == (3, None)       # never shown as zero


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


def test_a_model_in_use_that_changed_since_is_said_never_reloaded_over(root, serving, monkeypatch):
    plan = _plan(root, monkeypatch)
    serving["value"] = _status(root, served="model-08.pt")
    calls = []
    step = ml.finish_load(plan, operator_request(monkeypatch),
                          reload=lambda request: calls.append(request))
    assert calls == [] and step.plan is not None
    assert "The model in use changed since this was shown" in step.confirm_html
    assert step.plan["from"]["sha256"] == root["shas"]["model-08.pt"]


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


def test_a_markup_workspace_name_is_escaped_in_the_confirmation(root, serving, monkeypatch):
    listing = _listing(root, monkeypatch)
    plan, _html = ml.request_load(listing, f"gat/{RUNS[0][1]}/model-12.pt", None,
                                  operator_request(monkeypatch))
    plan["to"]["workspace"] = "<b>x</b>"
    html = ml.confirmation_html(plan)
    assert "<b>x</b>" not in html and "&lt;b&gt;x&lt;/b&gt;" in html


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
                                  json={"data": [None, False]})
            assert started.status_code == 200, started.text
            stream = client.get(f"/ui/gradio_api/call/{refresh.api_name}/"
                                f"{started.json()['event_id']}")
        lines = stream.text.splitlines()
        return json.loads(lines[lines.index("event: complete") + 1][len("data:"):])

    # The outputs a client sees: the listing itself is session state (not
    # returned), then the workspace choice, the count, the rows, the facts.
    refused = call("http://192.0.2.10")
    assert "Refused: " in json.dumps(refused) and "run_8c2e7a" not in json.dumps(refused)
    listed = call(OPERATOR_BASE_URL)
    _state, _selected, workspace, count, rows = listed[:5]
    assert workspace["value"] == "demo" and "of 7 in 6 runs" in count
    assert "run 2026-10-18 09:12 · epoch 12" in rows
    assert handle.load_button is not None and handle.refresh["fn"] is refresh.fn
