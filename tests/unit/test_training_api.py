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
    run_dir = data_dir / "runs" / "gat" / "20260223-100000_run_aaaaaa"
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
    """Minimal training config for testing."""
    return {
        "num_epochs": 10,
        "learning_rate": 1e-4,
        "batch_size": 16,
        "conv_type": "gat",
        "device": "cpu",
        "seed": 42,
    }


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
            {"epoch": 0, "val_loss": 3.0, "val_mrr": 0.1, "val_hits@1": 0.05, "val_hits@10": 0.2},
            {"epoch": 1, "val_loss": 2.5, "val_mrr": 0.15, "val_hits@1": 0.08, "val_hits@10": 0.25},
            {"epoch": 2, "val_loss": 2.0, "val_mrr": 0.2, "val_hits@1": 0.1, "val_hits@10": 0.3},
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
        after its refusals (contract M3a-2), so a refused run leaves nothing."""
        manager = TrainingManager()
        mock_proc = FakeProcess(pid=1)
        config = {"data_dir": str(tmp_path / "ws"), "conv_type": "hgt", "run_name": "Pilot A"}

        with patch("subprocess.Popen", return_value=mock_proc) as popen:
            result = manager.start_training(config)

        run_dir = Path(result["run_dir"])
        assert run_dir.parent == tmp_path / "ws" / "runs" / "hgt"
        assert "_pilot-a_" in run_dir.name
        assert not run_dir.exists()
        assert manager.run_dir == run_dir and manager.get_status()["run_dir"] == str(run_dir)
        import yaml
        written = yaml.safe_load(Path(result["config_path"]).read_text())
        assert written["run_dir"] == str(run_dir), "the process is not given the directory named"
        assert popen.call_args[0][0][-1] == result["config_path"]
        assert "run_dir" not in config, "the caller's dict was changed"

    @pytest.mark.parametrize("retired", ["output_dir", "checkpoint_dir", "log_dir"])
    def test_a_retired_location_is_refused_not_ignored(self, retired, tmp_path):
        manager = TrainingManager()
        with patch("subprocess.Popen") as popen:
            result = manager.start_training({"data_dir": str(tmp_path), retired: "x"})
        assert result["success"] is False and retired in result["error"]
        popen.assert_not_called()

    def test_an_explicit_run_directory_that_holds_files_is_refused(self, tmp_path):
        used = tmp_path / "used"
        used.mkdir()
        (used / "config.yaml").write_text("x")
        manager = TrainingManager()
        with patch("subprocess.Popen") as popen:
            result = manager.start_training({"run_dir": str(used)})
        assert result["success"] is False and "already holds files" in result["error"]
        popen.assert_not_called()

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
            manager.start_training({"data_dir": str(tmp_path / "ws")})

        result = manager.stop_training()

        assert result == {"success": True, "status": "stopped", "outcome": None}
        assert manager.get_status()["status"] == "stopped"
        assert manager.get_status()["outcome"] is None, "no last.pt was saved"

    def test_a_continuation_the_parent_does_not_allow_is_refused_before_launch(self, tmp_path):
        manager = TrainingManager()
        parent = tmp_path / "old.pt"
        import torch
        torch.save({"state_dict": {}}, parent)   # no M3a record
        with patch("subprocess.Popen") as popen:
            result = manager.start_training({"resume_from": str(parent)})
        assert result["success"] is False and "before contract M3a" in result["error"]
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
        assert metrics["validation"][2]["val_mrr"] == 0.2

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
        assert {c["architecture"] for c in checkpoints} == {"gat"}
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
             "logs": {"val_mrr": 0.4, "val_loss": 0.2},
             "training_input_digests": {"split_manifest": "5" * 64}},
            ckpt_dir / "last.pt")

        (entry,) = manager.get_checkpoints(tmp_dirs["data_dir"])
        assert entry["sha256"] == digest
        assert entry["validation"] == {"metric": "val_mrr", "value": 0.4}
        assert entry["dataset_fingerprint"] == "5" * 64
        assert entry["metadata_problem"] is None
        assert (entry["epoch"], entry["checkpoint_kind"]) == (3, "training_state")
        assert entry["metrics"] == {"val_mrr": 0.4, "val_loss": 0.2}

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
            {"epoch": 2, "logs": {"val_hits@1": 0.1},
             "training_input_digests": {"split_manifest": "6" * 64}},
            Path(tmp_dirs["checkpoint_dir"]) / "last.pt")
        (Path(tmp_dirs["checkpoint_dir"]) / "broken.pt").write_bytes(b"x")
        manager._config = {"data_dir": tmp_dirs["data_dir"]}

        listed = {c["filename"]: c for c in self.client.get(
            "/api/v1/training/checkpoints").json()}
        assert listed["last.pt"]["sha256"] == digest
        assert listed["last.pt"]["validation"] == {"metric": "val_hits@1", "value": 0.1}
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

    def test_start_training(self, sample_config):
        mock_proc = FakeProcess(pid=99999)

        with patch("subprocess.Popen", return_value=mock_proc):
            resp = self.client.post("/api/v1/training/start", json=sample_config)

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["status"] == "running"

    def test_stop_training_not_running(self):
        resp = self.client.post("/api/v1/training/stop")
        assert resp.status_code == 409

    def test_a_retired_location_is_refused_by_the_api(self, sample_config):
        with patch("subprocess.Popen") as popen:
            resp = self.client.post("/api/v1/training/start",
                                    json={**sample_config, "checkpoint_dir": "elsewhere"})
        assert resp.status_code == 422
        assert "checkpoint_dir" in resp.text
        popen.assert_not_called()

    def test_only_the_fields_the_caller_set_reach_the_training_process(self):
        """Defaults are the training script's own; a continued run inherits
        what the caller did not set (contract M3a-3)."""
        import yaml
        mock_proc = FakeProcess(pid=4)
        with patch("subprocess.Popen", return_value=mock_proc):
            resp = self.client.post("/api/v1/training/start",
                                    json={"num_epochs": 3, "device": "cpu"})
        written = yaml.safe_load(Path(resp.json()["config_path"]).read_text())
        assert set(written) == {"num_epochs", "device", "run_dir"}

    def test_a_retired_location_is_refused_even_empty(self, sample_config):
        """The request is strict (BACKLOG 54): a field it does not define, a
        retired location among them, is refused, empty or not, never dropped."""
        with patch("subprocess.Popen") as popen:
            resp = self.client.post("/api/v1/training/start",
                                    json={**sample_config, "checkpoint_dir": ""})
        assert resp.status_code == 422 and "checkpoint_dir" in resp.text
        popen.assert_not_called()


class TestTrainingRoutesAreGated:
    """Start, stop and the checkpoint list are operator controls (D8, M5-1):
    refused with 403 before the route runs, nothing started, stopped or
    opened, unless the deployment is in mode M1 and the request arrived on
    loopback. Status and metrics stay open."""

    GATED = [("post", "/api/v1/training/start", {"num_epochs": 1, "device": "cpu"}),
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


class TestTrainingRequestValidation:
    """TrainingStartRequest field + cross-field validation."""

    def _req(self, **overrides):
        from src.api.routes.training import TrainingStartRequest

        return TrainingStartRequest(**overrides)

    # -- WS1b: onecycle requires min_lr_ratio > 0; cosine/linear allow 0 --------
    def test_onecycle_zero_min_lr_ratio_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            self._req(scheduler_type="onecycle", min_lr_ratio=0.0)

    def test_cosine_zero_min_lr_ratio_allowed(self):
        req = self._req(scheduler_type="cosine", min_lr_ratio=0.0)
        assert req.min_lr_ratio == 0.0

    def test_linear_zero_min_lr_ratio_allowed(self):
        req = self._req(scheduler_type="linear", min_lr_ratio=0.0)
        assert req.min_lr_ratio == 0.0

    def test_onecycle_positive_min_lr_ratio_ok(self):
        req = self._req(scheduler_type="onecycle", min_lr_ratio=0.01)
        assert req.scheduler_type == "onecycle"

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


# ==============================================================================
# A continued run's rules are checked on the settings it will use
# (contract M3a review of 094b90d, P2-3)
# ==============================================================================
@pytest.fixture(scope="module")
def heads3_parent(tmp_path_factory):
    """A real paused parent with hidden_dim=18 and num_heads=3: valid for gat,
    and invalid against the new-training default hidden_dim of 256."""
    import scripts.train_model as train_model
    from tests.fixtures.trainable_workspace import write_trainable_workspace

    root = tmp_path_factory.mktemp("heads3")
    ws = write_trainable_workspace(root / "ws")
    train_model.train(train_model.TrainConfig(
        data_dir=str(ws), run_dir=str(root / "parent"), device="cpu", hidden_dim=18,
        num_layers=1, num_heads=3, num_workers=0, use_amp=False, num_epochs=2,
        batch_size=5, warmup_steps=1, pause_after_epoch=1))
    return ws, root / "parent" / "checkpoints" / "last.pt"


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
        yield TestClient(app, base_url=OPERATOR_BASE_URL), popen
    process.finish(0)


class TestContinuationRulesOnResolvedSettings:
    def test_a_partial_request_inherits_before_its_rules_are_checked(self, heads3_parent,
                                                                     real_client):
        import yaml

        ws, parent = heads3_parent
        client, popen = real_client
        resp = client.post("/api/v1/training/start", json={
            "data_dir": str(ws), "resume_from": str(parent), "device": "cpu", "num_heads": 3})

        assert resp.status_code == 200, resp.text
        popen.assert_called_once()
        written = yaml.safe_load(Path(resp.json()["config_path"]).read_text())
        assert "hidden_dim" not in written, "hidden_dim is the parent's (18), inherited"

    def test_an_explicit_value_other_than_the_parents_is_still_refused(self, heads3_parent,
                                                                       real_client):
        ws, parent = heads3_parent
        client, popen = real_client
        for heads in (6, 4):     # 18 % 6 == 0; 18 % 4 != 0: refused as a conflict either way
            resp = client.post("/api/v1/training/start", json={
                "data_dir": str(ws), "resume_from": str(parent), "device": "cpu",
                "num_heads": heads})
            assert resp.status_code == 409 and "num_heads: the parent's 3" in resp.text
        popen.assert_not_called()

    def test_resolved_settings_that_break_a_rule_are_refused_with_422(self, heads3_parent,
                                                                      real_client, tmp_path):
        """A parent whose recorded settings break the rule (here edited; one
        trained through the CLI, which does not apply the API's rules, could
        record such a pair) is refused on the settings the run would use."""
        ws, parent = heads3_parent
        client, popen = real_client
        record = torch_load(parent)
        record["config"]["requested"]["num_heads"] = 4
        edited = tmp_path / "edited.pt"
        import torch
        torch.save(record, edited)

        resp = client.post("/api/v1/training/start", json={
            "data_dir": str(ws), "resume_from": str(edited), "device": "cpu"})
        assert resp.status_code == 422
        assert "hidden_dim (18) must be divisible by num_heads (4)" in resp.text
        popen.assert_not_called()

    def test_a_new_run_still_validates_its_defaults(self, real_client):
        client, popen = real_client
        for body, fragment in (
            ({"num_heads": 3}, "hidden_dim (256) must be divisible by num_heads (3)"),
            ({"hidden_dim": 100, "num_heads": 8}, "hidden_dim (100) must be divisible"),
            ({"scheduler_type": "onecycle", "min_lr_ratio": 0.0}, "onecycle"),
            ({"extend_epochs": 2}, "extend_epochs continues a parent; give resume_from"),
        ):
            resp = client.post("/api/v1/training/start", json=body)
            assert resp.status_code == 422 and fragment in resp.text, (body, resp.text)
        popen.assert_not_called()


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
        resp = client.post("/api/v1/training/start", json={})
    assert resp.status_code == 422 and "refused by the service" in resp.text
    assert not [w for w in caught if "HTTP_422" in str(w.message)], [str(w.message) for w in caught]


def torch_load(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)
