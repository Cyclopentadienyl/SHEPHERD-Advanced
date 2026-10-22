"""
Start sends the dataset fingerprint it showed (M5-2, UI-C).
===========================================================
The Training run bar shows a workspace's dataset fingerprint (the SHA-256 of
its ``split_manifest.json``) and Start sends the full value; the job runner
reads the manifest again and refuses a different one ("The dataset changed
since it was shown; select it again.") **before anything is written or
started**: no run directory named, no config file, no process, no parent
read. One start path: `TrainingManager.start_training`, which the API route
calls with the same check (an optional query parameter) and the dashboard's
Start and Continue Training call in process.

Also here, the runner's reads the Training page's run view uses: a given run's
metrics log, and the dataset fingerprint a checkpoint records.

Module: tests/unit/test_training_start_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

import src.api.services.training_manager as tm  # noqa: E402
from src.api.services.training_manager import TrainingManager  # noqa: E402
from tests.fixtures.job_processes import FakeProcess  # noqa: E402
from tests.fixtures.operator_access import OPERATOR_BASE_URL, operator_mode  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from tests.fixtures.registered_cohort import write_demo_workspace

    return write_demo_workspace(tmp_path_factory.mktemp("startds") / "demo")


@pytest.fixture
def workspace(built, tmp_path):
    ws = tmp_path / "root" / "demo"
    shutil.copytree(built, ws)
    return ws


def _fingerprint(ws: Path) -> str:
    from src.kg.artifacts import read_split_manifest

    return read_split_manifest(ws).identity.sha256


def _rewrite_manifest(ws: Path) -> None:
    """The workspace rebuilt in place after it was shown: another manifest."""
    path = ws / "split_manifest.json"
    manifest = json.loads(path.read_text())
    manifest.setdefault("generation", {})["note"] = "rebuilt"
    path.write_text(json.dumps(manifest))


def _nothing_happened(manager: TrainingManager, popen, ws: Path, before: set) -> None:
    popen.assert_not_called()
    assert manager.run_dir is None and manager._config is None
    assert set(p.name for p in ws.rglob("*")) == before, "something was written"


# --------------------------------------------------------------------------- the runner
def test_the_dataset_shown_is_still_the_workspaces_one_so_the_run_starts(workspace):
    manager = TrainingManager()
    with patch("subprocess.Popen", return_value=FakeProcess(pid=7)) as popen:
        result = manager.start_training({"data_dir": str(workspace)},
                                        expected_dataset_fingerprint=_fingerprint(workspace))
    assert result["success"] is True and popen.call_count == 1
    assert Path(result["run_dir"]).parent.parent.parent == workspace


def test_a_dataset_that_changed_since_it_was_shown_is_refused_before_anything_happens(
        workspace):
    shown = _fingerprint(workspace)
    _rewrite_manifest(workspace)
    before = set(p.name for p in workspace.rglob("*"))
    manager = TrainingManager()
    with patch("subprocess.Popen") as popen, \
            patch("tempfile.mkstemp", side_effect=AssertionError("a config was written")):
        result = manager.start_training({"data_dir": str(workspace)},
                                        expected_dataset_fingerprint=shown)
    assert result["success"] is False and result["dataset_changed"] is True
    assert result["error"] == ("The dataset changed since it was shown; select it again. "
                               "Nothing was started.")
    _nothing_happened(manager, popen, workspace, before)


def test_a_continued_run_on_a_changed_dataset_is_refused_before_its_parent_is_read(
        workspace, monkeypatch):
    shown = _fingerprint(workspace)
    _rewrite_manifest(workspace)
    manager = TrainingManager()
    monkeypatch.setattr(manager, "preview_continuation",
                        lambda *a, **k: pytest.fail("the parent was read"))
    with patch("subprocess.Popen") as popen:
        result = manager.start_training(
            {"data_dir": str(workspace), "resume_from": str(workspace / "x.pt")},
            expected_dataset_fingerprint=shown)
    assert result["success"] is False and "dataset changed" in result["error"]
    popen.assert_not_called()


@pytest.mark.parametrize("expected", ["abc", "A" * 64, "g" * 64, 12, ""])
def test_a_value_that_is_not_a_full_digest_is_refused(workspace, expected):
    manager = TrainingManager()
    with patch("subprocess.Popen") as popen:
        result = manager.start_training({"data_dir": str(workspace)},
                                        expected_dataset_fingerprint=expected)
    assert result["success"] is False and "not a full SHA-256" in result["error"]
    popen.assert_not_called()


def test_a_manifest_that_cannot_be_read_now_is_refused(workspace):
    shown = _fingerprint(workspace)
    (workspace / "split_manifest.json").unlink()
    manager = TrainingManager()
    with patch("subprocess.Popen") as popen:
        result = manager.start_training({"data_dir": str(workspace)},
                                        expected_dataset_fingerprint=shown)
    assert result["success"] is False
    assert result["error"].startswith("The dataset changed since it was shown; select it again.")
    assert "cannot be read now" in result["error"] and str(workspace) not in result["error"]
    popen.assert_not_called()


def test_without_an_expected_fingerprint_nothing_is_checked(workspace):
    """A caller that shows no dataset (the CLI, an API client that sends no
    fingerprint) starts as before."""
    _rewrite_manifest(workspace)
    with patch("subprocess.Popen", return_value=FakeProcess(pid=8)):
        result = TrainingManager().start_training({"data_dir": str(workspace)})
    assert result["success"] is True


# --------------------------------------------------------------------------- the route
def _client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routes import training as training_routes

    operator_mode(monkeypatch)
    app = FastAPI()
    app.include_router(training_routes.router, prefix="/api/v1")
    return TestClient(app, base_url=OPERATOR_BASE_URL), training_routes


def test_the_route_takes_the_same_check_through_the_one_start_path(workspace, monkeypatch):
    client, routes = _client(monkeypatch)
    fresh = TrainingManager()
    monkeypatch.setattr(routes, "training_manager", fresh)
    shown = _fingerprint(workspace)
    with patch("subprocess.Popen", return_value=FakeProcess(pid=9)) as popen:
        ok = client.post(f"/api/v1/training/start?expected_dataset_fingerprint={shown}",
                         json={"data_dir": str(workspace)})
        assert ok.status_code == 200, ok.text
        written = Path(ok.json()["config_path"]).read_text()
        # The process gets it as an argument it checks on its own manifest read
        # (review of 8242b47, P2-2), never as a configuration key it records.
        assert "expected_dataset_fingerprint" not in written
        assert popen.call_args[0][0][-2:] == ["--expect-dataset-sha256", shown]
        fresh._last["training"].process.finish(0)
        fresh._last["training"].done.wait(5)
        _rewrite_manifest(workspace)
        refused = client.post(f"/api/v1/training/start?expected_dataset_fingerprint={shown}",
                              json={"data_dir": str(workspace)})
    assert refused.status_code == 409
    assert "The dataset changed since it was shown; select it again." in refused.text
    assert popen.call_count == 1


def test_the_route_refuses_a_malformed_fingerprint_before_the_service(monkeypatch):
    client, routes = _client(monkeypatch)
    monkeypatch.setattr(routes.training_manager, "start_training",
                        lambda *a, **k: pytest.fail("the service was reached"))
    response = client.post("/api/v1/training/start?expected_dataset_fingerprint=abc", json={})
    assert response.status_code == 422


def test_the_route_passes_the_fingerprint_to_the_service(monkeypatch):
    client, routes = _client(monkeypatch)
    seen = {}

    def start(config, **kwargs):
        seen.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(routes.training_manager, "start_training", start)
    assert client.post("/api/v1/training/start", json={}).status_code == 200
    assert seen["expected_dataset_fingerprint"] is None
    client.post(f"/api/v1/training/start?expected_dataset_fingerprint={'b' * 64}", json={})
    assert seen["expected_dataset_fingerprint"] == "b" * 64


# --------------------------------------------------------------------------- the dashboard's Start
def test_the_run_bars_start_sends_what_it_showed_and_the_runner_refuses_a_rebuild(
        workspace, monkeypatch):
    """End to end in process: the session reads the workspace (its chip shows
    the fingerprint), the workspace is rebuilt in place, Start is clicked:
    the runner refuses, and nothing is started."""
    from src.webui.components import training_console as tc
    from tests.fixtures.operator_access import gradio_request

    operator_mode(monkeypatch)
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(workspace.parent))
    fresh = TrainingManager()
    monkeypatch.setattr(tc, "training_manager", fresh)
    state = tc._workspace_state("demo")
    assert state["row"]["dataset_fingerprint"] == _fingerprint(workspace)
    assert state["row"]["dataset_fingerprint"][:12] in tc.chip_html(state)
    _rewrite_manifest(workspace)

    form = [""] + [100, 1e-4, "32", "gat", "cpu", 42, "256", 4, 0.1, 0.01, "cosine", 500, 0.01,
                   10, 1.0, 0.5, 0.3, 0.2, 1, 1.0, "8", True, "float16", 0.07, 0.1, 1.0,
                   "15, 10, 5", 5000]
    with patch("subprocess.Popen") as popen:
        notice, _seed = tc._on_start(gradio_request(), state, *form, False)
    assert notice == tc.notice_html("Not started: The dataset changed since it was shown; "
                                    "select it again. Nothing was started.")
    popen.assert_not_called()
    assert fresh.run_dir is None


# --------------------------------------------------------------------------- the run view's reads
def test_the_metrics_of_a_given_run_are_read_from_its_own_log(tmp_path):
    run = tmp_path / "runs" / "gat" / "20261018-142000_run_d41f8b"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "train_20261018_142000.json").write_text(
        json.dumps({"train": [{"epoch": 0, "total": 1.5}], "validation": []}))
    manager = TrainingManager()
    assert manager.metrics_log_file(run).name == "train_20261018_142000.json"
    assert manager.get_metrics_history(run)["train"] == [{"epoch": 0, "total": 1.5}]
    # The run started last stays the default (the API's /training/metrics).
    assert manager.get_metrics_history() == {"train": [], "validation": []}
    assert manager.metrics_log_file(tmp_path / "no-run") is None


def test_a_snapshot_and_a_preview_name_the_dataset_the_checkpoint_records(tmp_path):
    from src.utils.checkpoint_io import write_checkpoint

    run = tmp_path / "runs" / "gat" / "r"
    (run / "checkpoints").mkdir(parents=True)
    last = run / "checkpoints" / "last.pt"
    write_checkpoint({"checkpoint_kind": "training_state", "epoch": 0,
                      "training_input_digests": {"split_manifest": "d" * 64},
                      "progress": {"completed_epochs": 1, "end": "finished",
                                   "stage": {"start_epoch": 0, "target_epochs": 1}}}, last)
    manager = TrainingManager()
    assert manager.latest_snapshot(run)["dataset_fingerprint"] == "d" * 64
    assert manager.preview_continuation(str(last), device="cpu")["dataset_fingerprint"] == "d" * 64
    assert tm._recorded_dataset({"training_input_digests": {"split_manifest": "not-hex"}}) is None
    assert tm._recorded_dataset(None) is None
