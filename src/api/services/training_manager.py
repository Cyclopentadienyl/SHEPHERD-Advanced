"""
SHEPHERD-Advanced Training Manager Service
============================================
Manages training subprocess lifecycle, metrics reading, and system resource monitoring.

Module: src/api/services/training_manager.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/api/services/training_manager.py

Purpose:
    Provide an API-layer service that:
    - Launches scripts/train_model.py as a subprocess with config via temp YAML
    - Tracks subprocess PID, status (idle/running/completed/failed/stopping)
    - Reads metrics from MetricsLogger's JSON log files
    - Sends SIGTERM to stop training gracefully
    - Lists checkpoint files with metadata
    - Reports GPU/RAM resource stats via nvidia-smi subprocess

Architecture Note:
    This module lives in src.api.services, NOT in src.training.
    The webui layer imports this service (webui -> api is allowed).
    Training is invoked via subprocess to respect import-linter rules
    and avoid blocking the API event loop.

Dependencies:
    - subprocess: Launch training process
    - signal: Send stop signals
    - json: Read metrics logs
    - yaml: Write config files
    - pathlib: File path handling
    - shutil: Disk usage (optional)
    - psutil (optional): RAM monitoring

Called by:
    - src/api/routes/training.py (API endpoints)
    - src/webui/components/training_console.py (direct call via AppState)

Version: 1.0.0
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Project root (two levels up from src/api/services/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@dataclass
class TrainingStatus:
    """Snapshot of training state."""

    status: str = "idle"  # idle, running, completed, failed, stopping
    pid: Optional[int] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    start_time: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    latest_metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None


class TrainingManager:
    """
    Manages training subprocess lifecycle and metrics retrieval.

    Thread-safe singleton designed to be shared across API routes
    and Gradio event handlers in the same FastAPI process.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        output_dir: str = "outputs",
    ):
        self._process: Optional[subprocess.Popen] = None
        self._status: str = "idle"
        self._config: Optional[Dict[str, Any]] = None
        self._start_time: Optional[datetime] = None
        self._error_message: Optional[str] = None
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._temp_config_path: Optional[Path] = None

        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)

    # =========================================================================
    # Training Control
    # =========================================================================

    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start a training run as a subprocess.

        Args:
            config: Training configuration dict (mirrors TrainConfig fields).

        Returns:
            Status dict with success/error info.

        Raises:
            RuntimeError: If training is already running.
        """
        with self._lock:
            if self._status == "running":
                return {
                    "success": False,
                    "error": "Training is already running",
                    "status": self._status,
                }

            self._config = config
            self._error_message = None

            # Ensure directories exist
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Apply directory paths to config
            config.setdefault("log_dir", str(self.log_dir))
            config.setdefault("checkpoint_dir", str(self.checkpoint_dir))
            config.setdefault("output_dir", str(self.output_dir))

            # Write config to temp YAML
            temp_fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="train_config_")
            self._temp_config_path = Path(temp_path)
            try:
                with os.fdopen(temp_fd, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
            except Exception:
                os.close(temp_fd)
                raise

            # Launch subprocess
            train_script = PROJECT_ROOT / "scripts" / "train_model.py"
            cmd = [sys.executable, str(train_script), "--config", str(self._temp_config_path)]

            logger.info(f"Starting training: {' '.join(cmd)}")

            try:
                self._process = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    # On Windows use CREATE_NEW_PROCESS_GROUP for graceful stop
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        if sys.platform == "win32"
                        else 0
                    ),
                )
            except Exception as e:
                self._status = "failed"
                self._error_message = str(e)
                logger.error(f"Failed to start training: {e}")
                return {"success": False, "error": str(e), "status": "failed"}

            self._status = "running"
            self._start_time = datetime.now()

            # Start background monitor thread
            self._monitor_thread = threading.Thread(
                target=self._monitor_process,
                daemon=True,
                name="training-monitor",
            )
            self._monitor_thread.start()

            logger.info(f"Training started (PID: {self._process.pid})")
            return {
                "success": True,
                "pid": self._process.pid,
                "status": "running",
                "config_path": str(self._temp_config_path),
            }

    def stop_training(self) -> Dict[str, Any]:
        """
        Stop the running training subprocess gracefully.

        Returns:
            Status dict with result info.
        """
        with self._lock:
            if self._status != "running" or self._process is None:
                return {
                    "success": False,
                    "error": "No training is running",
                    "status": self._status,
                }

            self._status = "stopping"
            pid = self._process.pid

        logger.info(f"Stopping training (PID: {pid})...")

        try:
            if sys.platform == "win32":
                # On Windows, send CTRL_BREAK_EVENT to the process group
                self._process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self._process.send_signal(signal.SIGINT)

            # Wait up to 30 seconds for graceful shutdown
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Training did not stop gracefully, terminating...")
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()

        except OSError as e:
            logger.error(f"Error stopping training: {e}")
            with self._lock:
                self._status = "failed"
                self._error_message = f"Stop error: {e}"
            return {"success": False, "error": str(e), "status": "failed"}

        with self._lock:
            self._status = "completed"
        logger.info("Training stopped")
        return {"success": True, "status": "completed"}

    def _monitor_process(self) -> None:
        """Background thread that monitors the training subprocess."""
        if self._process is None:
            return

        # Read stdout in background (prevents pipe buffer from filling up)
        try:
            while True:
                line = self._process.stdout.readline()
                if not line and self._process.poll() is not None:
                    break
                if line:
                    # Log training output at DEBUG level
                    logger.debug(f"[train] {line.rstrip()}")
        except Exception:
            pass

        returncode = self._process.poll()
        with self._lock:
            if self._status == "stopping":
                self._status = "completed"
            elif returncode == 0:
                self._status = "completed"
            else:
                self._status = "failed"
                self._error_message = f"Process exited with code {returncode}"

        # Clean up temp config
        if self._temp_config_path and self._temp_config_path.exists():
            try:
                self._temp_config_path.unlink()
            except OSError:
                pass

        logger.info(f"Training process ended (code={returncode}, status={self._status})")

    # =========================================================================
    # Status & Metrics
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current training status snapshot."""
        with self._lock:
            elapsed = None
            if self._start_time and self._status == "running":
                elapsed = (datetime.now() - self._start_time).total_seconds()

            result = {
                "status": self._status,
                "pid": self._process.pid if self._process else None,
                "start_time": self._start_time.isoformat() if self._start_time else None,
                "elapsed_seconds": elapsed,
                "error_message": self._error_message,
                "config": self._config,
            }

        # Augment with latest metrics from log files
        metrics = self._read_latest_metrics()
        if metrics:
            result["current_epoch"] = metrics.get("epoch")
            result["total_epochs"] = (
                self._config.get("num_epochs") if self._config else None
            )
            result["latest_metrics"] = metrics

        return result

    def get_metrics_history(self) -> Dict[str, Any]:
        """
        Read full metrics history from MetricsLogger JSON logs.

        Returns:
            Dict with 'train' and 'validation' metric lists.
        """
        log_file = self._find_latest_log_file()
        if log_file is None:
            return {"train": [], "validation": []}

        try:
            with open(log_file) as f:
                data = json.load(f)
            return {
                "train": data.get("train", []),
                "validation": data.get("validation", []),
            }
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read metrics log: {e}")
            return {"train": [], "validation": []}

    def _read_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Read the most recent metrics entry from log files."""
        log_file = self._find_latest_log_file()
        if log_file is None:
            return None

        try:
            with open(log_file) as f:
                data = json.load(f)
            train_history = data.get("train", [])
            if train_history:
                return train_history[-1]
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _find_latest_log_file(self) -> Optional[Path]:
        """Find the most recently modified training log file."""
        if not self.log_dir.exists():
            return None

        log_files = sorted(
            self.log_dir.glob("train_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return log_files[0] if log_files else None

    # =========================================================================
    # Checkpoint Management
    # =========================================================================

    def get_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List checkpoint files with metadata.

        Returns:
            List of checkpoint info dicts.
        """
        if not self.checkpoint_dir.exists():
            return []

        checkpoints = []
        for ckpt_path in sorted(
            self.checkpoint_dir.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            stat = ckpt_path.stat()
            info: Dict[str, Any] = {
                "filename": ckpt_path.name,
                "filepath": str(ckpt_path),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }

            # Try to read checkpoint metadata without loading full tensors
            try:
                import torch

                meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(meta, dict):
                    info["epoch"] = meta.get("epoch")
                    info["metrics"] = meta.get("logs", {})
            except Exception:
                pass

            checkpoints.append(info)

        return checkpoints

    # =========================================================================
    # System Resource Monitoring
    # =========================================================================

    @staticmethod
    def get_system_resources() -> Dict[str, Any]:
        """
        Get GPU and RAM resource utilization.

        Uses nvidia-smi subprocess for GPU stats (avoids importing torch in webui).
        Uses /proc/meminfo or psutil for RAM.

        Returns:
            Dict with gpu and ram fields.
        """
        resources: Dict[str, Any] = {
            "gpu": TrainingManager._get_gpu_info(),
            "ram": TrainingManager._get_ram_info(),
            "timestamp": datetime.now().isoformat(),
        }
        return resources

    @staticmethod
    def _get_gpu_info() -> Dict[str, Any]:
        """Query GPU info via nvidia-smi."""
        gpu_info: Dict[str, Any] = {
            "available": False,
            "devices": [],
        }

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                gpu_info["available"] = True
                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        gpu_info["devices"].append(
                            {
                                "index": int(parts[0]),
                                "name": parts[1],
                                "utilization_percent": float(parts[2]),
                                "memory_used_mb": float(parts[3]),
                                "memory_total_mb": float(parts[4]),
                                "temperature_c": float(parts[5]),
                            }
                        )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        return gpu_info

    @staticmethod
    def _get_ram_info() -> Dict[str, Any]:
        """Get RAM usage info."""
        ram_info: Dict[str, Any] = {}

        try:
            import psutil

            mem = psutil.virtual_memory()
            ram_info = {
                "total_gb": round(mem.total / (1024**3), 2),
                "used_gb": round(mem.used / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "percent": mem.percent,
            }
        except ImportError:
            # Fallback: read /proc/meminfo on Linux
            try:
                meminfo: Dict[str, float] = {}
                with open("/proc/meminfo") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            key = parts[0].rstrip(":")
                            value_kb = float(parts[1])
                            meminfo[key] = value_kb

                total = meminfo.get("MemTotal", 0) / (1024 * 1024)
                available = meminfo.get("MemAvailable", 0) / (1024 * 1024)
                ram_info = {
                    "total_gb": round(total, 2),
                    "used_gb": round(total - available, 2),
                    "available_gb": round(available, 2),
                    "percent": round((total - available) / total * 100, 1) if total > 0 else 0,
                }
            except OSError:
                ram_info = {"error": "Unable to read memory info"}

        return ram_info


# Module-level singleton instance
training_manager = TrainingManager()

__all__ = [
    "TrainingManager",
    "TrainingStatus",
    "training_manager",
]
