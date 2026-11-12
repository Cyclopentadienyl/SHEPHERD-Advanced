"""
# ==============================================================================
# Module: tests/unit/test_training_api.py
# ==============================================================================
# Purpose: Unit tests for training API routes and TrainingManager service
#
# Tests:
#   - TrainingManager lifecycle (start, status, stop)
#   - TrainingManager metrics reading from log files
#   - TrainingManager checkpoint listing
#   - TrainingManager system resources
#   - Training API endpoint responses
#   - System resources API endpoint
# ==============================================================================
"""
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Skip if fastapi/pydantic not available
pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from src.api.services.training_manager import TrainingManager
from src.config.model_types import WRITTEN_BEFORE_SPEC
from tests.fixtures.job_processes import FakeProcess
from tests.fixtures.operator_access import OPERATOR_BASE_URL, operator_mode


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture
def tmp_dirs(tmp_path):
    """A workspace with one run directory (contract M3a-2): its logs and
    checkpoints are inside it."""
    data_dir = tmp_path / "ws"
    run_dir = data_dir / "runs" / "gatv2" / "20260223-100000_run_aaaaaa"
    log_dir = run_dir / "logs"
    checkpoint_dir = run_dir / "checkpoints"
    log_dir.mkdir(parents=True)
    checkpoint_dir.mkdir()
    return {
        "data_dir": str(data_dir),
        "run_dir": str(run_dir),
        "log_dir": str(log_dir),
        "checkpoint_dir": str(checkpoint_dir),
    }


@pytest.fixture
def manager(tmp_dirs):
    """A TrainingManager whose last run is the fixture's run."""
    manager = TrainingManager()
    manager.run_dir = Path(tmp_dirs["run_dir"])
    return manager


@pytest.fixture
def sample_config():
    """Minimal training config for testing: a base start (TM12: a new run is
    a base), on the reference spec, with a few of the recipe's settings."""
    return {
        "learning_stage": "base",
        "num_epochs": 10,
        "learning_rate": 1e-4,
        "batch_size": 16,
        "device": "cpu",
        "seed": 42,
    }


@pytest.fixture
def base_body(tmp_dirs):
    """A base start as POST /training/start takes it (K7, TM11): the WebUI
    base form's call, over HTTP."""
    return {"data_dir": tmp_dirs["data_dir"], "learning_stage": "base", "device": "cpu",
            "num_epochs": 10}


@pytest.fixture
def sample_metrics_log(tmp_dirs):
    """Write a sample metrics log file."""
    log_dir = Path(tmp_dirs["log_dir"])
    log_file = log_dir / "train_20260223_100000.json"
    data = {
        "train": [
            {"epoch": 0, "total": 2.5, "timestamp": "2026-02-23T10:00:00"},
            {"epoch": 1, "total": 2.0, "lr_group_0": 1e-4, "timestamp": "2026-02-23T10:01:00"},
            {"epoch": 2, "total": 1.5, "lr_group_0": 9e-5, "timestamp": "2026-02-23T10:02:00"},
        ],
        "validation": [
            {"epoch": 0, "val_loss": 3.0, "val_pooled_mrr": 0.1, "val_pooled_top1": 0.05,
             "val_pooled_top10": 0.2},
            {"epoch": 1, "val_loss": 2.5, "val_pooled_mrr": 0.15, "val_pooled_top1": 0.08,
             "val_pooled_top10": 0.25},
            {"epoch": 2, "val_loss": 2.0, "val_pooled_mrr": 0.2, "val_pooled_top1": 0.1,
             "val_pooled_top10": 0.3},
        ],
    }
    with open(log_file, "w") as f:
        json.dump(data, f)
    return log_file


# ==============================================================================
# TrainingManager Tests
# ==============================================================================
class TestTrainingManagerInit:
    """Test TrainingManager initialization."""

    def test_initial_status_is_idle(self, manager):
        status = manager.get_status()
        assert status["status"] == "idle"
        assert status["pid"] is None

    def test_logs_are_read_from_the_last_runs_directory(self, manager, tmp_dirs):
        assert str(manager.log_dir) == tmp_dirs["log_dir"]
        fresh = TrainingManager()
        assert fresh.run_dir is None and fresh.log_dir is None
        assert fresh.get_metrics_history() == {"train": [], "validation": []}
        assert fresh.get_status()["run_dir"] is None


class TestTrainingManagerStartStop:
    """Test TrainingManager start/stop (with mocked subprocess)."""

    def test_start_training_success(self, manager, sample_config):
        """Start training with mocked subprocess."""
        mock_proc = FakeProcess(pid=12345)

        with patch("subprocess.Popen", return_value=mock_proc):
            result = manager.start_training(sample_config)

        assert result["success"] is True
        assert result["pid"] == 12345
        assert result["status"] == "running"

    def test_start_names_a_new_run_directory_and_creates_nothing(self, tmp_path):
        """The directory is named here and claimed by the training process
        after its refusals (contract M3a-2), so a refused run leaves nothing.
        A base is filed under its model spec's encoder kind (the reference
        spec's gatv2 here)."""
        manager = TrainingManager()
        mock_proc = FakeProcess(pid=1)
        config = {"data_dir": str(tmp_path / "ws"), "learning_stage": "base",
                  "run_name": "Pilot A"}

        with patch("subprocess.Popen", return_value=mock_proc) as popen:
            result = manager.start_training(config)

        run_dir = Path(result["run_dir"])
        assert run_dir.parent == tmp_path / "ws" / "runs" / "gatv2"
        assert "_pilot-a_" in run_dir.name
        assert not run_dir.exists()
        assert manager.run_dir == run_dir and manager.get_status()["run_dir"] == str(run_dir)
        import yaml
        written = yaml.safe_load(Path(result["config_path"]).read_text())
        assert written["run_dir"] == str(run_dir), "the process is not given the directory named"
        assert popen.call_args[0][0][-1] == result["config_path"]
        assert "run_dir" not in config, "the caller's dict was changed"

    @pytest.mark.parametrize("start", ["base", "continuation"])
    @pytest.mark.parametrize("retired", ["output_dir", "checkpoint_dir", "log_dir"])
    def test_a_retired_location_is_refused_not_ignored(self, retired, start, tmp_path,
                                                       request):
        """A new run (a base) is refused by its whole-configuration check, a
        retired location being no setting a base takes; a continuation of a
        real paused base, whose settings are the parent's, by the preview of
        that parent, which records no such setting to inherit. Nothing is
        spawned either way."""
        manager = TrainingManager()
        if start == "base":
            config = {"data_dir": str(tmp_path), "learning_stage": "base", retired: "x"}
            said = f"{retired} is not a setting a base run reads"
        else:
            ws, parent = request.getfixturevalue("paused_base")
            config = {"data_dir": str(ws), "resume_from": str(parent), "device": "cpu",
                      retired: "x"}
            said = f"does not record {retired}, so the run cannot inherit them"
        with patch("subprocess.Popen") as popen:
            result = manager.start_training(config)
        assert result["success"] is False and said in result["error"], result
        popen.assert_not_called()

    @pytest.mark.parametrize("retired", ["output_dir", "checkpoint_dir", "log_dir"])
    def test_the_retired_location_guard_refuses_a_continuation_the_preview_passed(
            self, retired, tmp_path):
        """The run layout's retired-location guard alone
        (`refuse_retired_locations`, which `start_training` calls once a
        continuation's preview passes), with the preview stubbed to pass: no
        real continuation reaches the guard today, since the preview of a real
        parent refuses a retired location first (the case above)."""
        manager = TrainingManager()
        config = {"data_dir": str(tmp_path), "resume_from": "p.pt", retired: "x"}
        preview = {"ok": True, "architecture": "gatv2", "sha256": "c" * 64,
                   "learning_stage": "base"}
        with patch.object(manager, "preview_continuation", return_value=preview), \
                patch("subprocess.Popen") as popen:
            result = manager.start_training(config)
        assert result["success"] is False, result
        assert f"sets {retired}, which no longer exist" in result["error"], result
        popen.assert_not_called()

    def test_an_explicit_run_directory_that_holds_files_is_refused(self, tmp_path):
        used = tmp_path / "used"
        used.mkdir()
        (used / "config.yaml").write_text("x")
        manager = TrainingManager()
        with patch("subprocess.Popen") as popen:
            result = manager.start_training({"learning_stage": "base", "run_dir": str(used)})
        assert result["success"] is False and "already holds files" in result["error"]
        popen.assert_not_called()

    @pytest.mark.parametrize("explicit", [False, True], ids=["allocated", "explicit"])
    def test_a_run_directory_behind_a_link_out_of_the_workspace_is_refused(self, tmp_path,
                                                                          explicit):
        """The workspace's runs/ is a link to another disk: a run named there
        would land where the reload's automatic selection never reads (the
        reviewer's K8-1 recheck), so the start is refused before a process is
        started or anything is written, saying why and what to do."""
        workspace, elsewhere = tmp_path / "ws", tmp_path / "elsewhere"
        workspace.mkdir()
        elsewhere.mkdir()
        (workspace / "runs").symlink_to(elsewhere, target_is_directory=True)
        config = {"data_dir": str(workspace), "learning_stage": "base"}
        if explicit:
            config["run_dir"] = str(workspace / "runs" / "gatv2" / "mine")
        manager = TrainingManager()
        with patch("subprocess.Popen") as popen:
            result = manager.start_training(config)
        assert result["success"] is False
        assert "resolves outside the workspace, so it is not written" in result["error"]
        assert "explicit checkpoint_path" in result["error"]
        popen.assert_not_called()
        assert list(elsewhere.iterdir()) == []

    def test_start_training_already_running(self, manager, sample_config):
        """Cannot start training when already running."""
        mock_proc = FakeProcess(pid=12345)

        with patch("subprocess.Popen", return_value=mock_proc):
            manager.start_training(sample_config)
            result = manager.start_training(sample_config)

        assert result["success"] is False
        assert "already running" in result["error"]

    def test_stop_training_not_running(self, manager):
        """Stop when nothing is running returns error."""
        result = manager.stop_training()
        assert result["success"] is False
        assert "No training" in result["error"]

    def test_a_stop_is_reported_as_stopped_with_the_latest_snapshot(self, tmp_path):
        """Contract M3a: an operator's stop is not a completed stage, even
        when the process exits 0 on it."""
        manager = TrainingManager()
        process = FakeProcess(pid=7, on_sigint=0)
        with patch("subprocess.Popen", return_value=process):
            manager.start_training({"data_dir": str(tmp_path / "ws"), "learning_stage": "base"})

        result = manager.stop_training()

        assert result == {"success": True, "status": "stopped", "outcome": None}
        assert manager.get_status()["status"] == "stopped"
        assert manager.get_status()["outcome"] is None, "no last.pt was saved"

    def test_a_continuation_the_parent_does_not_allow_is_refused_before_launch(self, tmp_path):
        manager = TrainingManager()
        parent = tmp_path / "old.pt"
        import torch
        torch.save({"state_dict": {}}, parent)   # no record at all
        with patch("subprocess.Popen") as popen:
            result = manager.start_training({"resume_from": str(parent)})
        assert result["success"] is False
        assert f"The parent old.pt: {WRITTEN_BEFORE_SPEC}" in result["error"]
        popen.assert_not_called()

    def test_a_continued_run_is_placed_under_its_parents_architecture(self, tmp_path):
        manager = TrainingManager()
        mock_proc = FakeProcess(pid=3)
        preview = {"ok": True, "architecture": "hgt", "sha256": "c" * 64}
        with patch.object(manager, "preview_continuation", return_value=preview), \
                patch("subprocess.Popen", return_value=mock_proc) as popen:
            result = manager.start_training({"data_dir": str(tmp_path / "ws"),
                                             "resume_from": "p.pt"})
        assert Path(result["run_dir"]).parent == tmp_path / "ws" / "runs" / "hgt"
        # The process is bound to the parent this placement was decided on (P22-S1).
        argv = popen.call_args[0][0]
        assert argv[argv.index("--expect-resume-sha256") + 1] == "c" * 64


class TestTrainingManagerMetrics:
    """Test metrics reading from log files."""

    def test_get_metrics_history(self, manager, sample_metrics_log):
        metrics = manager.get_metrics_history()
        assert len(metrics["train"]) == 3
        assert len(metrics["validation"]) == 3
        assert metrics["train"][0]["total"] == 2.5
        assert metrics["validation"][2]["val_pooled_mrr"] == 0.2

    def test_get_metrics_no_logs(self, manager):
        metrics = manager.get_metrics_history()
        assert metrics == {"train": [], "validation": []}

    def test_latest_metrics_in_status(self, manager, sample_metrics_log):
        status = manager.get_status()
        assert status["status"] == "idle"
        # Even when idle, we can read existing logs
        latest = status.get("latest_metrics")
        if latest:
            assert latest["epoch"] == 2

    def test_corrupt_metrics_log(self, manager, tmp_dirs):
        """Corrupt JSON should be handled gracefully."""
        log_dir = Path(tmp_dirs["log_dir"])
        corrupt_file = log_dir / "train_20260223_999999.json"
        corrupt_file.write_text("not valid json{{{")

        metrics = manager.get_metrics_history()
        assert metrics == {"train": [], "validation": []}


class TestTrainingManagerCheckpoints:
    """Test checkpoint listing."""

    def test_get_checkpoints_empty(self, manager):
        checkpoints = manager.get_checkpoints()
        assert checkpoints == []

    def test_get_checkpoints_with_files(self, manager, tmp_dirs):
        """Create dummy checkpoint files and list them."""
        ckpt_dir = Path(tmp_dirs["checkpoint_dir"])
        # Create dummy .pt files (they won't be loadable by torch, that's ok)
        (ckpt_dir / "model-01-0.5000.pt").write_bytes(b"dummy")
        (ckpt_dir / "last.pt").write_bytes(b"dummy")

        checkpoints = manager.get_checkpoints(tmp_dirs["data_dir"])
        assert len(checkpoints) == 2
        assert {c["run"] for c in checkpoints} == {"20260223-100000_run_aaaaaa"}
        assert {c["architecture"] for c in checkpoints} == {"gatv2"}
        filenames = [c["filename"] for c in checkpoints]
        assert "last.pt" in filenames
        assert "model-01-0.5000.pt" in filenames
        # size_mb should be very small for dummy files
        assert all(c["size_mb"] >= 0 for c in checkpoints)
        # Unloadable: listed with why, no digest, no metadata (M5-1), never dropped.
        for entry in checkpoints:
            assert entry["metadata_problem"] and entry["sha256"] is None
            assert entry["validation"] is None and "epoch" not in entry

    def test_each_entry_carries_its_reads_digest_and_record(self, manager, tmp_dirs):
        """One read per file (`read_run_checkpoints`): the digest, the Trainer's
        validation figure and the dataset fingerprint come from the bytes the
        metadata came from (BACKLOG 19.3)."""
        from src.utils.checkpoint_io import write_checkpoint

        ckpt_dir = Path(tmp_dirs["checkpoint_dir"])
        digest = write_checkpoint(
            {"checkpoint_kind": "training_state", "epoch": 3,
             "logs": {"val_pooled_mrr": 0.4, "val_loss": 0.2},
             "training_input_digests": {"split_manifest": "5" * 64}},
            ckpt_dir / "last.pt")

        (entry,) = manager.get_checkpoints(tmp_dirs["data_dir"])
        assert entry["sha256"] == digest
        assert entry["validation"] == {"metric": "val_pooled_mrr", "value": 0.4}
        assert entry["dataset_fingerprint"] == "5" * 64
        assert entry["metadata_problem"] is None
        assert (entry["epoch"], entry["checkpoint_kind"]) == (3, "training_state")
        assert entry["metrics"] == {"val_pooled_mrr": 0.4, "val_loss": 0.2}

    def test_the_manager_no_longer_loads_a_file_itself(self):
        """The one read is `read_checkpoint`'s, inside the listing."""
        import inspect

        source = inspect.getsource(TrainingManager.get_checkpoints)
        assert "torch.load" not in source and "read_run_checkpoints" in source


class TestTrainingManagerResources:
    """Test system resource monitoring."""

    def test_get_system_resources_structure(self):
        resources = TrainingManager.get_system_resources()
        assert "gpu" in resources
        assert "ram" in resources
        assert "timestamp" in resources

    def test_gpu_info_without_nvidia_smi(self):
        """When nvidia-smi is not available, gpu.available should be False."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            info = TrainingManager._get_gpu_info()
        assert info["available"] is False
        assert info["devices"] == []

    def test_gpu_info_with_nvidia_smi(self):
        """Mock nvidia-smi output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "0, NVIDIA RTX 5070 Ti, 45, 2048, 17408, 55\n"

        with patch("subprocess.run", return_value=mock_result):
            info = TrainingManager._get_gpu_info()

        assert info["available"] is True
        assert len(info["devices"]) == 1
        assert info["devices"][0]["name"] == "NVIDIA RTX 5070 Ti"
        assert info["devices"][0]["utilization_percent"] == 45.0

    def test_ram_info_returns_data(self):
        ram = TrainingManager._get_ram_info()
        # Should have data from psutil or /proc/meminfo or an error
        assert isinstance(ram, dict)


# ==============================================================================
# API Endpoint Tests
# ==============================================================================
class TestTrainingAPIEndpoints:
    """Test FastAPI training endpoints using TestClient."""

    @pytest.fixture(autouse=True)
    def setup_client(self, manager, monkeypatch):
        """Patch the global training_manager with our test instance. The
        client is one D8 allows (mode M1, loopback): start, stop and the
        checkpoint list are operator routes; `TestTrainingRoutesAreGated`
        covers the refusals."""
        monkeypatch.setattr(
            "src.api.routes.training.training_manager", manager
        )
        from fastapi.testclient import TestClient
        from src.api.routes.training import router
        from fastapi import FastAPI

        operator_mode(monkeypatch)
        test_app = FastAPI()
        test_app.include_router(router, prefix="/api/v1")
        self.client = TestClient(test_app, base_url=OPERATOR_BASE_URL)

    def test_get_status_idle(self):
        resp = self.client.get("/api/v1/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"

    def test_get_metrics_empty(self):
        resp = self.client.get("/api/v1/training/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["train"] == []
        assert data["validation"] == []

    def test_get_checkpoints_empty(self):
        resp = self.client.get("/api/v1/training/checkpoints")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_the_route_passes_the_reads_digest_on(self, manager, tmp_dirs):
        """The response model keeps the fields the listing adds (M5-1)."""
        from src.utils.checkpoint_io import write_checkpoint

        digest = write_checkpoint(
            {"epoch": 2, "logs": {"val_pooled_mrr": 0.1},
             "training_input_digests": {"split_manifest": "6" * 64}},
            Path(tmp_dirs["checkpoint_dir"]) / "last.pt")
        (Path(tmp_dirs["checkpoint_dir"]) / "broken.pt").write_bytes(b"x")
        manager._config = {"data_dir": tmp_dirs["data_dir"]}

        listed = {c["filename"]: c for c in self.client.get(
            "/api/v1/training/checkpoints").json()}
        assert listed["last.pt"]["sha256"] == digest
        assert listed["last.pt"]["validation"] == {"metric": "val_pooled_mrr", "value": 0.1}
        assert listed["last.pt"]["dataset_fingerprint"] == "6" * 64
        assert listed["last.pt"]["metadata_problem"] is None
        assert listed["broken.pt"]["sha256"] is None and listed["broken.pt"]["metadata_problem"]

    def test_a_link_whose_target_is_gone_does_not_break_the_list(self, manager, tmp_dirs):
        """A `.pt` link whose target is gone (or a checkpoint training removed
        between the listing and its `stat`) used to fail the whole list with
        `FileNotFoundError`. The file has no size or time to show, so this
        list leaves it out, as before for a file gone since it was listed
        (M5-1 review)."""
        from src.utils.checkpoint_io import write_checkpoint

        checkpoints = Path(tmp_dirs["checkpoint_dir"])
        write_checkpoint({"epoch": 1}, checkpoints / "last.pt")
        (checkpoints / "gone.pt").symlink_to(checkpoints / "removed.pt")
        manager._config = {"data_dir": tmp_dirs["data_dir"]}

        response = self.client.get("/api/v1/training/checkpoints")
        assert response.status_code == 200
        assert [c["filename"] for c in response.json()] == ["last.pt"]

    def test_start_training(self, base_body):
        mock_proc = FakeProcess(pid=99999)

        with patch("subprocess.Popen", return_value=mock_proc):
            resp = self.client.post("/api/v1/training/start", json=base_body)

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["status"] == "running"
        assert "/runs/gatv2/" in data["run_dir"], "filed under the reference spec's encoder"

    def test_stop_training_not_running(self):
        resp = self.client.post("/api/v1/training/stop")
        assert resp.status_code == 409

    @pytest.mark.parametrize("payload, named", [
        ({"checkpoint_kind": "training_state", "progress": {"end": "\udc80"}},
         "its progress.end is not text: character 0 is a lone surrogate"),
        ({"progress": {"end": "stopped_early", "stop_reason": "patience\ud800"}},
         "its progress.stop_reason is not text: character 8 is a lone surrogate"),
        ({"progress": {"completed_epochs": "tensor", "end": None}},
         "its progress.completed_epochs is a Tensor, not a whole number"),
        ({"progress": {"stage": {"start_epoch": "x", "target_epochs": 2}}},
         "its progress.stage.start_epoch is a str, not a whole number"),
        (b"legacy", "RuntimeError: Invalid protocol version: \\udc80proto"),
    ], ids=["end-surrogate", "stop-reason-surrogate", "completed-tensor", "start-epoch-text",
            "legacy-protocol-surrogate"])
    def test_a_stopped_runs_snapshot_that_cannot_be_shown_as_read_is_not_readable(
            self, base_body, payload, named):
        """The K8-4r4 verifier, defect 2: the status put the end, the stop
        reason, the epoch counts and the loader's message of the job's own
        ``last.pt`` in its answer as read, so a lone surrogate in one (the
        legacy loader's message names the file's protocol version) failed the
        answer's encoding, a tensor its serialisation and a start epoch of
        "x" the snapshot itself: the stop and every later status failed. Now
        both answer 200, the snapshot not readable, the reason naming the
        field (as the listing checks it) or escaping the message."""
        import pickle

        import torch

        process = FakeProcess(pid=8, on_sigint=0)
        with patch("subprocess.Popen", return_value=process):
            run = Path(self.client.post("/api/v1/training/start", json=base_body).json()[
                "run_dir"])
        last = run / "checkpoints" / "last.pt"
        last.parent.mkdir(parents=True, exist_ok=True)
        if payload == b"legacy":
            last.write_bytes(pickle.dumps(0x1950A86A20F9469CFC6C, protocol=2)
                             + pickle.dumps("\udc80proto", protocol=2))
        else:
            progress = payload["progress"]
            if progress.get("completed_epochs") == "tensor":
                progress["completed_epochs"] = torch.tensor([1, 2])
            torch.save(payload, last)

        for response in (self.client.post("/api/v1/training/stop"),
                         self.client.get("/api/v1/training/status")):
            assert response.status_code == 200, response.text[:300]
            outcome = response.json()["outcome"]
            assert outcome["readable"] is False and outcome["checkpoint"] == str(last)
            assert outcome["reason"].startswith(named), outcome
            assert "resumable" not in outcome and "end" not in outcome

    def test_a_retired_location_is_refused_by_the_api(self, base_body):
        with patch("subprocess.Popen") as popen:
            resp = self.client.post("/api/v1/training/start",
                                    json={**base_body, "checkpoint_dir": "elsewhere"})
        assert resp.status_code == 422
        assert "checkpoint_dir" in resp.text
        popen.assert_not_called()

    def test_only_the_fields_the_caller_set_reach_the_training_process(self, tmp_dirs):
        """The reference recipe fills every base setting not set, and the
        training process records which (contract M3a-3): the configuration
        written is the caller's fields, the stage and the run directory, as
        the WebUI base form's start writes it."""
        import yaml
        mock_proc = FakeProcess(pid=4)
        with patch("subprocess.Popen", return_value=mock_proc):
            resp = self.client.post("/api/v1/training/start", json={
                "data_dir": tmp_dirs["data_dir"], "learning_stage": "base", "num_epochs": 3,
                "device": "cpu"})
        written = yaml.safe_load(Path(resp.json()["config_path"]).read_text())
        assert written == {"data_dir": tmp_dirs["data_dir"], "learning_stage": "base",
                           "num_epochs": 3, "device": "cpu", "run_dir": resp.json()["run_dir"]}

    def test_a_retired_location_is_refused_even_empty(self, base_body):
        """The request is strict (BACKLOG 54): a field it does not define, a
        retired location among them, is refused, empty or not, never dropped."""
        with patch("subprocess.Popen") as popen:
            resp = self.client.post("/api/v1/training/start",
                                    json={**base_body, "checkpoint_dir": ""})
        assert resp.status_code == 422 and "checkpoint_dir" in resp.text
        popen.assert_not_called()


class TestTrainingRoutesAreGated:
    """Start, stop and the checkpoint list are operator controls (D8, M5-1):
    refused with 403 before the route runs, nothing started, stopped or
    opened, unless the deployment is in mode M1 and the request arrived on
    loopback. Status and metrics stay open."""

    GATED = [("post", "/api/v1/training/start", {"data_dir": "x", "learning_stage": "base"}),
             ("post", "/api/v1/training/stop", None),
             ("get", "/api/v1/training/checkpoints", None)]

    @pytest.fixture
    def guarded(self, monkeypatch):
        """The routes over a manager whose every action fails the test."""
        from fastapi import FastAPI

        from src.api.routes.training import router

        class Untouchable:
            def __getattr__(self, name):
                pytest.fail(f"a refused request reached the service ({name})")

        monkeypatch.setattr("src.api.routes.training.training_manager", Untouchable())
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        return app

    @pytest.mark.parametrize("method, path, body", GATED)
    @pytest.mark.parametrize("mode, base_url, headers, says", [
        (None, "http://127.0.0.1", {}, "SHEPHERD_DEPLOYMENT_MODE is not set"),
        ("M1", "http://192.0.2.10", {}, "did not arrive on a loopback address"),
        ("M1", "http://127.0.0.1", {"X-Forwarded-For": "203.0.113.9"}, "forwarding header"),
        ("M1", "http://127.0.0.1", {"Origin": "https://evil.example"}, "origin"),
        ("M1", "http://127.0.0.1", {"Host": "evil.example"}, "not addressed to localhost"),
    ], ids=["mode-unset", "lan", "forwarded", "foreign-origin", "rebinding-host"])
    def test_refused_before_the_service_is_reached(self, guarded, monkeypatch, method, path,
                                                   body, mode, base_url, headers, says):
        from fastapi.testclient import TestClient

        if mode is not None:
            monkeypatch.setenv("SHEPHERD_DEPLOYMENT_MODE", mode)
        client = TestClient(guarded, base_url=base_url)
        response = getattr(client, method)(path, headers=headers,
                                           **({"json": body} if body is not None else {}))
        assert response.status_code == 403, response.text
        assert says in response.json()["detail"]
        assert "203.0.113.9" not in response.text and "evil.example" not in response.text

    def test_an_invalid_body_from_a_refused_request_is_refused_not_validated(self, guarded):
        """The gate comes first: a request D8 refuses learns nothing about
        which settings the service would accept."""
        from fastapi.testclient import TestClient

        response = TestClient(guarded, base_url="http://192.0.2.10").post(
            "/api/v1/training/start", json={"num_epochs": -5})
        assert response.status_code == 403

    def test_status_and_metrics_stay_open(self, manager, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.api.routes.training import router

        monkeypatch.setattr("src.api.routes.training.training_manager", manager)
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app, base_url="http://192.0.2.10")
        assert client.get("/api/v1/training/status").status_code == 200
        assert client.get("/api/v1/training/metrics").status_code == 200


class TestSystemAPIEndpoint:
    """Test system resources endpoint."""

    def test_get_system_resources(self):
        from fastapi.testclient import TestClient
        from src.api.routes.system import router
        from fastapi import FastAPI

        test_app = FastAPI()
        test_app.include_router(router, prefix="/api/v1")
        client = TestClient(test_app)

        resp = client.get("/api/v1/system/resources")
        assert resp.status_code == 200
        data = resp.json()
        assert "gpu" in data
        assert "ram" in data


class TestTheBaseStartRequest:
    """TrainingStartRequest (K7, TM11): a strict base start, its fields'
    bounds the field spec's."""

    def _req(self, **overrides):
        from src.api.routes.training import TrainingStartRequest

        return TrainingStartRequest(**{"data_dir": "data/workspaces/x", "learning_stage": "base",
                                       **overrides})

    def test_a_base_start_needs_its_workspace_and_its_stage(self):
        from pydantic import ValidationError

        from src.api.routes.training import TrainingStartRequest

        for missing in ({"data_dir": "x"}, {"learning_stage": "base"}, {}):
            with pytest.raises(ValidationError):
                TrainingStartRequest(**missing)
        assert self._req().learning_stage == "base"

    @pytest.mark.parametrize("stage", ["finetune", "pretrain", None, ""])
    def test_another_stage_is_refused(self, stage):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            self._req(learning_stage=stage)

    @pytest.mark.parametrize("field, value", [
        ("conv_type", "gat"), ("hidden_dim", 256), ("num_heads", 8), ("num_workers", 0),
        ("diagnosis_weight", 1.0), ("use_ortholog_gate", True), ("task", "disease"),
        ("finetune_from", "base.pt"), ("frozen_parts", "none"), ("run_dir", "/r"),
        ("unknown", 1),
    ])
    def test_a_shepherdgnn_or_unknown_field_is_refused(self, field, value):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            self._req(**{field: value})

    def test_a_setting_left_unset_is_the_recipes_not_the_requests(self):
        """Nothing the caller did not set is sent: every base setting
        defaults to unset, and the recipe fills it."""
        from src.config.recipe import BASE

        sent = self._req(num_epochs=5).model_dump(exclude_unset=True, exclude_none=True)
        assert sent == {"data_dir": "data/workspaces/x", "learning_stage": "base",
                        "num_epochs": 5}
        assert {name: getattr(self._req(), name) for name in BASE} == dict.fromkeys(BASE)

    # -- WS3: batch_size upper bound raised to 2048 ----------------------------
    def test_batch_size_2048_accepted(self):
        assert self._req(batch_size=2048).batch_size == 2048

    def test_batch_size_over_2048_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            self._req(batch_size=2049)

    # -- WS6: seed constrained to numpy's [0, 2**32 - 1] -----------------------
    def test_seed_zero_and_max_accepted(self):
        assert self._req(seed=0).seed == 0
        assert self._req(seed=2**32 - 1).seed == 2**32 - 1

    def test_seed_over_numpy_max_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            self._req(seed=2**32)

    def test_extend_needs_a_parent(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="extend_epochs continues a parent"):
            self._req(extend_epochs=2)


# ==============================================================================
# The base start through the real route and job runner (K7, TM11)
# ==============================================================================
@pytest.fixture(scope="module")
def paused_base(tmp_path_factory):
    """A real paused base (a small spec, one of its two epochs) on a base
    workspace: a parent POST /training/start continues."""
    import sys

    scripts = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import train_model
    from tests.fixtures.base_workspace import write_base_workspace
    from tests.unit.test_train_base_stage import _config

    ws = write_base_workspace(tmp_path_factory.mktemp("paused-base"))
    run = ws / "runs" / "gatv2" / "20261001-000000_base_aaaaaa"
    train_model.train(*_config(ws, run, pause_after_epoch=1))
    return ws, run / "checkpoints" / "last.pt"


@pytest.fixture(scope="module")
def specless_parent(paused_base, tmp_path_factory):
    """A paused training-state checkpoint with no model spec, its record as a
    ShepherdGNN run wrote one (record version 1, no stage requested): the
    paused base's file with its model record taken out, as
    test_train_base_stage.py builds a parent written before the model spec.
    Not a base, so the base start does not continue it."""
    import torch

    from src.config.model_types import MODEL_RECORD_KEYS

    ws, parent = paused_base
    payload = torch.load(parent, map_location="cpu", weights_only=True)
    for key in MODEL_RECORD_KEYS:
        payload.pop(key, None)
    payload["record_version"] = 1
    payload["config"] = {**payload["config"],
                         "requested": {**payload["config"]["requested"], "learning_stage": None}}
    path = tmp_path_factory.mktemp("specless-parent") / "last.pt"
    torch.save(payload, path)
    return ws, path


@pytest.fixture
def real_client(monkeypatch):
    """The real route over a real TrainingManager; only the process is stubbed."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes.training import router

    manager = TrainingManager()
    monkeypatch.setattr("src.api.routes.training.training_manager", manager)
    operator_mode(monkeypatch)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    process = FakeProcess(pid=11)
    with patch("subprocess.Popen", return_value=process) as popen:
        yield TestClient(app, base_url=OPERATOR_BASE_URL), popen, manager
    process.finish(0)


def _nothing_started(manager, popen, *workspaces):
    """A refused start started no process, claimed no GPU slot and named no
    run directory (none exists under the workspaces)."""
    popen.assert_not_called()
    assert manager.active_job() is None and manager.run_dir is None
    assert manager.get_status()["status"] == "idle"
    for workspace in workspaces:
        assert not (Path(workspace) / "runs" / "gat").exists()
        assert not list(Path(workspace).glob("runs/*/*_run_*"))


class TestTheBaseStartThroughTheRunner:
    def test_a_refused_request_reaches_no_service(self, monkeypatch, tmp_path):
        """Refused by the request itself (422): the service is never called."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.api.routes.training import router

        class Untouchable:
            def __getattr__(self, name):
                pytest.fail(f"a refused request reached the service ({name})")

        monkeypatch.setattr("src.api.routes.training.training_manager", Untouchable())
        operator_mode(monkeypatch)
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app, base_url=OPERATOR_BASE_URL)
        base = {"data_dir": str(tmp_path), "learning_stage": "base"}
        for body in ({"data_dir": str(tmp_path)}, {**base, "learning_stage": "finetune"},
                     {**base, "conv_type": "gat"}, {**base, "hidden_dim": 64},
                     {**base, "task": "disease"}, {**base, "unknown": 1}, {},
                     {**base, "num_epochs": 0}, {**base, "extend_epochs": 2}):
            assert client.post("/api/v1/training/start", json=body).status_code == 422, body

    @pytest.mark.parametrize("settings, fragment", [
        ({"num_neighbors": [15, 10]}, "num_neighbors needs 3 numbers, one hop per encoder "
                                      "layer, not 2"),
        ({"num_neighbors": [15, 10, 5, 5]}, "not 4"),
    ])
    def test_a_setting_the_runner_refuses_starts_nothing(self, real_client, tmp_path, settings,
                                                         fragment):
        """The runner's one check (`_new_bundle_run`) refuses what the recipe or
        the reference spec does not take (422), before the slot or a run
        directory is claimed."""
        client, popen, manager = real_client
        resp = client.post("/api/v1/training/start", json={
            "data_dir": str(tmp_path / "ws"), "learning_stage": "base", "device": "cpu",
            **settings})
        assert resp.status_code == 422, resp.text
        assert fragment in resp.text and "Nothing was started." in resp.text
        _nothing_started(manager, popen, tmp_path / "ws")

    def test_a_base_continues_with_its_settings(self, paused_base, real_client):
        """resume_from a paused base: started, every learning setting
        inherited (none is written), the stage the parent's."""
        import yaml

        ws, parent = paused_base
        client, popen, manager = real_client
        resp = client.post("/api/v1/training/start", json={
            "data_dir": str(ws), "learning_stage": "base", "resume_from": str(parent),
            "device": "cpu"})
        assert resp.status_code == 200, resp.text
        popen.assert_called_once()
        written = yaml.safe_load(Path(resp.json()["config_path"]).read_text())
        assert written["learning_stage"] == "base" and written["resume_from"] == str(parent)
        assert not {"num_epochs", "batch_size", "model_spec"} & set(written), written
        assert "/runs/gatv2/" in resp.json()["run_dir"]

    def test_a_setting_other_than_the_parents_is_refused(self, paused_base, real_client):
        ws, parent = paused_base
        client, popen, manager = real_client
        resp = client.post("/api/v1/training/start", json={
            "data_dir": str(ws), "learning_stage": "base", "resume_from": str(parent),
            "device": "cpu", "batch_size": 64})
        assert resp.status_code == 409 and "batch_size: the parent's 16" in resp.text
        _nothing_started(manager, popen, ws)

    def test_a_parent_that_is_not_a_base_is_refused(self, specless_parent,
                                                     trained_disease_child, real_client):
        """A finetune child (here continued by extending it) is not a base: the
        runner's preview of the parent refuses the stage asked for, on its one
        read, before anything starts. A checkpoint with no model spec (record
        version 1) is refused before that, by its record version."""
        client, popen, manager = real_client
        for ws, parent, extra, said in (
                (trained_disease_child["data_dir"], trained_disease_child["disease"],
                 {"extend_epochs": 1}, "learning_stage: the parent's 'finetune', requested "
                                       "'base'"),
                (*specless_parent, {}, WRITTEN_BEFORE_SPEC)):
            resp = client.post("/api/v1/training/start", json={
                "data_dir": str(ws), "learning_stage": "base", "resume_from": str(parent),
                "device": "cpu", **extra})
            assert resp.status_code == 409 and said in resp.text, resp.text
            _nothing_started(manager, popen, ws)



# ==============================================================================
# The branch start through the real route and job runner (ops-wiring round, B2)
# ==============================================================================
def _sha256(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TestTheBranchStartThroughTheRunner:
    def test_a_refused_request_reaches_no_service(self, monkeypatch, tmp_path):
        """Refused by the request itself (422): a field it does not define, a
        name in settings that is not a training setting, a device the grammar
        does not take. The service is never called."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.api.routes.training import router

        class Untouchable:
            def __getattr__(self, name):
                pytest.fail(f"a refused request reached the service ({name})")

        monkeypatch.setattr("src.api.routes.training.training_manager", Untouchable())
        operator_mode(monkeypatch)
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app, base_url=OPERATOR_BASE_URL)
        body = {"data_dir": str(tmp_path), "branch_from": "last.pt"}
        for sent in ({}, {"data_dir": str(tmp_path)}, {**body, "learning_rate": 1e-3},
                     {**body, "resume_from": "last.pt"}, {**body, "device": "gpu"},
                     {**body, "settings": {"run_dir": "/elsewhere"}},
                     {**body, "settings": {"conv_type": "gat"}},
                     {**body, "settings": {"extend_epochs": 1}}, {**body, "settings": []}):
            assert client.post("/api/v1/training/branch", json=sent).status_code == 422, sent

    def test_a_same_settings_branch_starts_bound_to_its_parent(self, paused_base, real_client):
        """``settings`` {} (decision B1): started through the one start, its
        stage the parent's, no setting written, the process bound to the
        parent's digest."""
        import yaml

        ws, parent = paused_base
        client, popen, manager = real_client
        resp = client.post("/api/v1/training/branch", json={
            "data_dir": str(ws), "branch_from": str(parent), "device": "cpu", "settings": {}})
        assert resp.status_code == 200, resp.text
        argv = popen.call_args.args[0]
        assert argv[4:6] == ["--expect-parent-sha256", _sha256(parent)]
        written = yaml.safe_load(Path(resp.json()["config_path"]).read_text())
        assert written == {"data_dir": str(ws), "branch_from": str(parent), "device": "cpu",
                           "run_name": "", "learning_stage": "base",
                           "run_dir": resp.json()["run_dir"]}
        assert "/runs/gatv2/" in resp.json()["run_dir"]

    @pytest.mark.parametrize("settings, fragment", [
        ({"learning_rate": -1.0}, "learning_rate must be"),
        ({"split_seed": 9}, "keeps every setting of its parent but those on its list"),
        ({"task_loss_weight": 0.5}, "task_loss_weight is not a setting a base run reads"),
        ({"num_neighbors": [4]}, "num_neighbors needs 2 numbers"),
    ])
    def test_settings_the_runner_refuses_answer_422(self, paused_base, real_client, settings,
                                                    fragment):
        ws, parent = paused_base
        client, popen, manager = real_client
        resp = client.post("/api/v1/training/branch", json={
            "data_dir": str(ws), "branch_from": str(parent), "device": "cpu",
            "settings": settings})
        assert resp.status_code == 422 and fragment in resp.text, resp.text
        _nothing_started(manager, popen, ws)

    def test_a_parent_changed_since_it_was_shown_answers_409(self, paused_base, real_client):
        ws, parent = paused_base
        client, popen, manager = real_client
        resp = client.post(f"/api/v1/training/branch?expected_parent_sha256={'0' * 64}", json={
            "data_dir": str(ws), "branch_from": str(parent), "device": "cpu"})
        assert resp.status_code == 409, resp.text
        assert "The checkpoint changed since it was chosen" in resp.text
        _nothing_started(manager, popen, ws)
        assert client.post("/api/v1/training/branch?expected_parent_sha256=abc", json={
            "data_dir": str(ws), "branch_from": str(parent)}).status_code == 422

    def test_the_route_passes_the_settings_and_both_digests_to_the_service(self, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.api.routes import training as training_routes

        seen = []
        monkeypatch.setattr(training_routes.training_manager, "start_training",
                            lambda config, **kwargs: seen.append((config, kwargs))
                            or {"success": True})
        operator_mode(monkeypatch)
        app = FastAPI()
        app.include_router(training_routes.router, prefix="/api/v1")
        client = TestClient(app, base_url=OPERATOR_BASE_URL)
        body = {"data_dir": "data/workspaces/x", "branch_from": "runs/a/checkpoints/last.pt",
                "settings": {"learning_rate": 1e-3}}
        assert client.post("/api/v1/training/branch", json=body).status_code == 200
        assert client.post(f"/api/v1/training/branch?expected_parent_sha256={'b' * 64}"
                           f"&expected_dataset_fingerprint={'d' * 64}",
                           json={**body, "settings": {}}).status_code == 200
        assert seen == [
            ({"data_dir": "data/workspaces/x", "branch_from": "runs/a/checkpoints/last.pt",
              "device": "auto", "run_name": "", "learning_rate": 1e-3},
             {"expected_dataset_fingerprint": None, "expected_parent_sha256": None}),
            ({"data_dir": "data/workspaces/x", "branch_from": "runs/a/checkpoints/last.pt",
              "device": "auto", "run_name": ""},
             {"expected_dataset_fingerprint": "d" * 64, "expected_parent_sha256": "b" * 64})]

    def test_the_branch_route_does_not_run_on_the_event_loop(self):
        """It reads the whole parent checkpoint: a plain `def`, as the
        Fine-tune start is."""
        import inspect

        from src.api.routes.training import start_branch

        assert not inspect.iscoroutinefunction(start_branch)


def test_the_services_422_does_not_use_the_name_starlette_deprecated(monkeypatch):
    """The route's own 422 (settings the service refuses): Starlette 1.x, which
    Gradio 6.30 brings, renamed 422 after RFC 9110 and warns on the old
    HTTP_422_UNPROCESSABLE_ENTITY. Same status code, without the warning."""
    import warnings

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routes import training as training_routes

    monkeypatch.setattr(training_routes.training_manager, "start_training",
                        lambda config, **kwargs: {"success": False, "invalid_settings": True,
                                                  "error": "refused by the service"})
    operator_mode(monkeypatch)
    app = FastAPI()
    app.include_router(training_routes.router, prefix="/api/v1")
    client = TestClient(app, base_url=OPERATOR_BASE_URL)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resp = client.post("/api/v1/training/start",
                           json={"data_dir": "x", "learning_stage": "base"})
    assert resp.status_code == 422 and "refused by the service" in resp.text
    assert not [w for w in caught if "HTTP_422" in str(w.message)], [str(w.message) for w in caught]
