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


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture
def tmp_dirs(tmp_path):
    """Create temp directories for logs, checkpoints, and outputs."""
    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "outputs"
    log_dir.mkdir()
    checkpoint_dir.mkdir()
    output_dir.mkdir()
    return {
        "log_dir": str(log_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
    }


@pytest.fixture
def manager(tmp_dirs):
    """Create a TrainingManager instance with temp directories."""
    return TrainingManager(
        log_dir=tmp_dirs["log_dir"],
        checkpoint_dir=tmp_dirs["checkpoint_dir"],
        output_dir=tmp_dirs["output_dir"],
    )


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

    def test_directories_are_set(self, manager, tmp_dirs):
        assert str(manager.log_dir) == tmp_dirs["log_dir"]
        assert str(manager.checkpoint_dir) == tmp_dirs["checkpoint_dir"]
        assert str(manager.output_dir) == tmp_dirs["output_dir"]


class TestTrainingManagerStartStop:
    """Test TrainingManager start/stop (with mocked subprocess)."""

    def test_start_training_success(self, manager, sample_config):
        """Start training with mocked subprocess."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.return_value = ""

        with patch("subprocess.Popen", return_value=mock_proc):
            result = manager.start_training(sample_config)

        assert result["success"] is True
        assert result["pid"] == 12345
        assert result["status"] == "running"

    def test_start_training_already_running(self, manager, sample_config):
        """Cannot start training when already running."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.return_value = ""

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

        checkpoints = manager.get_checkpoints()
        assert len(checkpoints) == 2
        filenames = [c["filename"] for c in checkpoints]
        assert "last.pt" in filenames
        assert "model-01-0.5000.pt" in filenames
        # size_mb should be very small for dummy files
        assert all(c["size_mb"] >= 0 for c in checkpoints)


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
        """Patch the global training_manager with our test instance."""
        monkeypatch.setattr(
            "src.api.routes.training.training_manager", manager
        )
        from fastapi.testclient import TestClient
        from src.api.routes.training import router
        from fastapi import FastAPI

        test_app = FastAPI()
        test_app.include_router(router, prefix="/api/v1")
        self.client = TestClient(test_app)

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

    def test_start_training(self, sample_config):
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.poll.return_value = None
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.return_value = ""

        with patch("subprocess.Popen", return_value=mock_proc):
            resp = self.client.post("/api/v1/training/start", json=sample_config)

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["status"] == "running"

    def test_stop_training_not_running(self):
        resp = self.client.post("/api/v1/training/stop")
        assert resp.status_code == 409


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
