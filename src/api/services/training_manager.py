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
from typing import Any, Callable, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Project root (two levels up from src/api/services/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@dataclass
class TrainingStatus:
    """Snapshot of training state."""

    # idle, running, stopping, stopped (by the operator), completed (the
    # process exited cleanly; how the stage stood is the run's `outcome`),
    # failed.
    status: str = "idle"
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

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._status: str = "idle"
        self._user_stopped: bool = False
        self._config: Optional[Dict[str, Any]] = None
        self._start_time: Optional[datetime] = None
        self._error_message: Optional[str] = None
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._temp_config_path: Optional[Path] = None

        #: The directory of the run started last (contract M3a-2). Named here
        #: and claimed by the training process once its inputs pass, so a
        #: refused run leaves nothing behind; its logs are read from inside it.
        self.run_dir: Optional[Path] = None
        self._snapshot_cache: Optional[tuple] = None

    @property
    def log_dir(self) -> Optional[Path]:
        """The logs of the run started last, or `None` before any."""
        from src.utils.checkpoint_paths import run_log_dir

        return run_log_dir(self.run_dir) if self.run_dir is not None else None

    @staticmethod
    def workspace_path(data_dir: str) -> Path:
        """A workspace path as the training process resolves it: relative to
        the project root, where the process runs."""
        path = Path(data_dir or "data/workspaces/default")
        return path if path.is_absolute() else PROJECT_ROOT / path

    # =========================================================================
    # Training Control
    # =========================================================================

    def start_training(
        self,
        config: Dict[str, Any],
        effective_rules: Optional[Callable[[Dict[str, Any]], List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Start a training run as a subprocess.

        Args:
            config: Training configuration dict (mirrors TrainConfig fields).
            effective_rules: for a run continued from a parent, rules checked on
                the settings the run plan resolves (the API's cross-field
                rules); a violation is refused with `invalid_settings`.

        Returns:
            Status dict with success/error info.

        Raises:
            RuntimeError: If training is already running.
        """
        # A run continued from a parent is decided by the parent's record
        # (contract M3a-3): refused here as the training process would refuse
        # it, and placed under the parent's architecture, which it inherits.
        # The process decides again on the bytes it restores. Read before the
        # lock, so the status stays readable meanwhile.
        architecture = config.get("conv_type")
        if config.get("resume_from"):
            preview = self.preview_continuation(
                config["resume_from"], device=config.get("device", "auto"),
                extend_epochs=config.get("extend_epochs"),
                requested={k: v for k, v in config.items() if k != "resume_from"})
            if not preview["ok"]:
                return {"success": False, "error": preview["reason"], "status": self._status}
            errors = effective_rules(preview["learning_settings"]) if effective_rules else []
            if errors:
                return {"success": False, "invalid_settings": True, "status": self._status,
                        "error": "The settings this run would use, its parent's with this "
                                 "request's: " + " ".join(errors)}
            architecture = preview.get("architecture") or architecture

        with self._lock:
            if self._status == "running":
                return {
                    "success": False,
                    "error": "Training is already running",
                    "status": self._status,
                }

            # **The run's own directory** (contract M3a-2): named here, so the
            # status and the console know where its logs and checkpoints will be,
            # and handed to the training process, which claims it after its
            # inputs pass. An explicit one must be free; nothing is created here.
            from src.utils.checkpoint_paths import (
                check_run_dir_free, propose_run_dir, refuse_retired_locations,
            )

            try:
                refuse_retired_locations(config, source="This training request")
                if config.get("run_dir"):
                    run_dir = check_run_dir_free(config["run_dir"])
                else:
                    run_dir = propose_run_dir(
                        self.workspace_path(config.get("data_dir", "")),
                        architecture, config.get("run_name"))
            except (ValueError, OSError) as e:
                return {"success": False, "error": str(e), "status": self._status}

            config = {**config, "run_dir": str(run_dir)}
            self._config = config
            self._error_message = None
            self._user_stopped = False
            self.run_dir = run_dir

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
                    stdin=subprocess.DEVNULL,
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

            logger.info(f"Training started (PID: {self._process.pid}), run directory {run_dir}")
            return {
                "success": True,
                "pid": self._process.pid,
                "status": "running",
                "config_path": str(self._temp_config_path),
                "run_dir": str(run_dir),
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
            self._user_stopped = True
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
            self._status = "stopped"
        logger.info("Training stopped")
        # **Stopped, not completed** (contract M3a): the stage did not finish.
        # What can be continued is the run's last saved snapshot, if any;
        # progress after it is lost.
        return {"success": True, "status": "stopped", "outcome": self.latest_snapshot()}

    def _monitor_process(self) -> None:
        """Background thread that monitors the training subprocess."""
        if self._process is None:
            return

        # Read stdout in background (prevents pipe buffer from filling up)
        # Keep a rolling buffer of recent lines for error reporting
        recent_lines: list[str] = []
        max_recent = 50
        try:
            while True:
                line = self._process.stdout.readline()
                if not line and self._process.poll() is not None:
                    break
                if line:
                    stripped = line.rstrip()
                    logger.debug(f"[train] {stripped}")
                    recent_lines.append(stripped)
                    if len(recent_lines) > max_recent:
                        recent_lines.pop(0)
        except Exception:
            pass

        returncode = self._process.poll()
        with self._lock:
            if self._status == "stopping" or self._user_stopped:
                # Operator-initiated stop: stopped, whatever the exit code.
                self._status = "stopped"
            elif returncode == 0:
                self._status = "completed"
            else:
                self._status = "failed"
                # Filter out Intel MKL/Fortran noise from error output
                filtered = [
                    line for line in recent_lines[-20:]
                    if "forrtl:" not in line
                    and "KERNELBASE" not in line
                    and "KERNEL32" not in line
                    and "ntdll.dll" not in line
                    and "Unknown               Unknown  Unknown" not in line
                ]
                tail = "\n".join(filtered) if filtered else ""
                self._error_message = (
                    f"Process exited with code {returncode}"
                    + (f"\n\n--- Last output ---\n{tail}" if tail else "")
                )

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

    def _read_progress_file(self) -> Optional[Dict[str, Any]]:
        """Read the lightweight progress.json file."""
        if self.log_dir is None:
            return None
        progress_file = self.log_dir / "progress.json"
        if not progress_file.exists():
            return None
        try:
            with open(progress_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

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
                "run_dir": str(self.run_dir) if self.run_dir is not None else None,
            }

        # How the run stands, from its last snapshot's own record, once the
        # process is not running: finished, paused, stopped early, or where it
        # can be continued from (contract M3a).
        if result["status"] in ("stopped", "completed", "failed"):
            result["outcome"] = self.latest_snapshot()

        # Read progress file for real-time batch-level progress
        progress = self._read_progress_file()
        if progress:
            result["phase"] = progress.get("phase")
            result["batch"] = progress.get("batch")
            result["total_batches"] = progress.get("total_batches")

        # Augment with latest metrics from log files
        metrics = self._read_latest_metrics()
        if metrics:
            # Convert 0-indexed epoch to 1-indexed for display
            raw_epoch = metrics.get("epoch")
            result["current_epoch"] = (raw_epoch + 1) if raw_epoch is not None else None
            # The run's own count (its active stage's end), which an inherited
            # or extended run does not carry in the request.
            result["total_epochs"] = (progress or {}).get("total_epochs") or (
                self._config.get("num_epochs") if self._config else None
            )
            result["latest_metrics"] = metrics
        elif progress:
            # No epoch-level metrics yet; use progress file for epoch info
            raw_epoch = progress.get("epoch")
            result["current_epoch"] = (raw_epoch + 1) if raw_epoch is not None else None
            result["total_epochs"] = progress.get("total_epochs") or (
                self._config.get("num_epochs") if self._config else None
            )

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
        if self.log_dir is None or not self.log_dir.exists():
            return None

        log_files = sorted(
            self.log_dir.glob("train_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return log_files[0] if log_files else None

    def latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """The run's latest saved training state (`last.pt`) as its own record
        describes it, or `None` when the run saved none.

        Read whole with the safe loader, so only when asked (the stop
        confirmation, a status after the process ended) and cached by the
        file's size and modification time.
        """
        from src.utils.checkpoint_paths import run_checkpoint_dir

        if self.run_dir is None:
            return None
        last = run_checkpoint_dir(self.run_dir) / "last.pt"
        try:
            stat = last.stat()
        except OSError:
            return None
        key = (str(last), stat.st_mtime_ns, stat.st_size)
        if self._snapshot_cache and self._snapshot_cache[0] == key:
            return self._snapshot_cache[1]
        try:
            from src.utils.checkpoint_io import read_checkpoint

            record = read_checkpoint(last, map_location="cpu", weights_only=True).checkpoint
        except Exception as exc:  # noqa: BLE001 - shown, not raised
            snapshot = {"checkpoint": str(last), "readable": False,
                        "reason": f"{type(exc).__name__}: {exc}"}
        else:
            progress = (record.get("progress") or {}) if isinstance(record, dict) else {}
            stage = progress.get("stage") or {}
            end = progress.get("end")
            snapshot = {
                "checkpoint": str(last),
                "readable": True,
                "completed_epochs": progress.get("completed_epochs"),
                "target_epochs": (int(stage.get("start_epoch", 0))
                                  + int(stage.get("target_epochs", 0))) if stage else None,
                "end": end,
                "stop_reason": progress.get("stop_reason"),
                "resumable": (isinstance(record, dict)
                              and record.get("checkpoint_kind") == "training_state"
                              and end in (None, "paused")),
            }
        self._snapshot_cache = (key, snapshot)
        return snapshot

    def preview_continuation(
        self,
        parent_path: str,
        *,
        device: str = "auto",
        extend_epochs: Optional[int] = None,
        requested: Optional[Dict[str, Any]] = None,
        extend_if_finished: bool = False,
    ) -> Dict[str, Any]:
        """What starting from `parent_path` would do, for the console to show
        before start (contract M3a-3, M3a-4).

        `requested` is a training request's own fields, set explicitly: a
        learning setting among them that differs from the parent's is refused
        here, as the training process would refuse it.

        `extend_if_finished` is the console's: its Extra epochs field belongs to
        a finished parent, so for any other parent the value is not sent
        (`training_console._preview_for`). Every other caller leaves it off, and
        an `extend_epochs` given for an unfinished parent is refused.

        **The same decision the training process makes**
        (`src.utils.run_plan.plan_continuation`), on the parent's own
        record read with the safe loader. A preview is not the check: the
        training process reads the parent again, checks it against the
        workspace's data and graph (M2.3) and decides again on those bytes.
        """
        from src.utils.run_plan import OPERATIONAL_FIELDS, RunPlanRefused, plan_continuation
        from src.utils.checkpoint_io import read_checkpoint

        path = Path(parent_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            return {"ok": False, "reason": f"{path} does not exist."}
        try:
            read = read_checkpoint(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - a refusal to show
            return {"ok": False,
                    "reason": f"{path.name} cannot be read as a checkpoint "
                              f"({type(exc).__name__}: {exc})."}
        record = read.checkpoint if isinstance(read.checkpoint, dict) else {}
        progress = record.get("progress") or {}
        stage = progress.get("stage") or {}
        if extend_if_finished and progress.get("end") != "finished":
            extend_epochs = None
        facts = {
            "parent": str(path),
            "sha256": read.identity.sha256,
            "end": progress.get("end"),
            "completed_epochs": progress.get("completed_epochs"),
            "target_epochs": (int(stage.get("start_epoch", 0))
                              + int(stage.get("target_epochs", 0))) if stage else None,
            "architecture": ((record.get("config") or {}).get("model_config") or {}).get(
                "conv_type"),
        }
        # The parent's own settings, with the request's fields over them; only
        # the request's fields count as set explicitly, so every other learning
        # setting shows as inherited.
        explicit = dict(requested or {})
        recorded = (record.get("config") or {}).get("requested")
        request = {**(recorded if isinstance(recorded, dict) else {}), **explicit,
                   "resume_from": str(path), "run_dir": "", "extend_epochs": extend_epochs,
                   "extend_lr": explicit.get("extend_lr"), "pause_after_epoch": None,
                   "device": device}
        effective_device = "cuda" if device == "auto" else device
        try:
            plan = plan_continuation(record, path, request, set(explicit),
                                     device=effective_device,
                                     parent_sha256=read.identity.sha256)
        except RunPlanRefused as exc:
            return {**facts, "ok": False, "reason": str(exc)}
        return {**facts, "ok": True, "mode": plan.mode, "action": plan.action,
                "inherited": plan.inherited, "start_epoch": plan.start_epoch,
                "end_epoch": plan.end_epoch, "stage": plan.stage,
                # The learning settings the run would use (the parent's, with
                # the request's explicit values), for checks that need them
                # complete: the API's cross-field rules. Operational fields are
                # left out: the process takes them from the request or its own
                # defaults, which this preview does not know.
                "learning_settings": {k: v for k, v in plan.settings.items()
                                      if k not in OPERATIONAL_FIELDS}}

    # =========================================================================
    # Checkpoint Management
    # =========================================================================

    def get_checkpoints(
        self, data_dir: Optional[str] = None, conv_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List a workspace's run checkpoints with metadata (contract M3a-2).

        Every run's, or one architecture's, through the shared listing, each
        file read once (`checkpoint_paths.read_run_checkpoints`, M5-1): runs
        most recently trained first, newest file first within a run. The
        workspace defaults to the last training request's. What each file is
        comes from its own record; a listed file is not thereby verified, and a
        file outside the run layout is reached by its path.

        Each entry carries the SHA-256 of the bytes read (`sha256`), the
        Trainer's validation figure (`validation`: metric and value, or None),
        the dataset fingerprint it records (`dataset_fingerprint`) and, for a
        file the read refused, why (`metadata_problem`), beside the keys the
        console and the route already read.

        Returns:
            List of checkpoint info dicts.
        """
        from src.utils.checkpoint_paths import read_run_checkpoints, run_architecture

        if data_dir is None:
            if not self._config:
                return []
            data_dir = self._config.get("data_dir", "")
        workspace = self.workspace_path(data_dir)

        checkpoints = []
        for entry in read_run_checkpoints(workspace, conv_type):
            try:
                stat = entry.path.stat()
            except OSError:
                continue  # removed since it was listed: there is nothing to show
            info: Dict[str, Any] = {
                "filename": entry.path.name,
                "filepath": str(entry.path),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "run": entry.run_dir.name,
                "run_dir": str(entry.run_dir),
                "architecture": run_architecture(entry.run_dir),
                "sha256": entry.sha256,
                "validation": ({"metric": entry.validation[0], "value": entry.validation[1]}
                               if entry.validation else None),
                "dataset_fingerprint": entry.dataset_fingerprint,
                "metadata_problem": entry.problem,
            }
            # The file's own record, from the same read as the digest; a file
            # the read refused has none.
            if entry.logs is not None:
                info["epoch"] = entry.epoch
                info["metrics"] = entry.logs
                info["checkpoint_kind"] = entry.checkpoint_kind

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
        gpu_info = TrainingManager._get_gpu_info()
        ram_info = TrainingManager._get_ram_info()
        resources: Dict[str, Any] = {
            "gpu": gpu_info,
            "ram": ram_info,
            "unified_memory": TrainingManager._detect_unified_memory(gpu_info, ram_info),
            "timestamp": datetime.now().isoformat(),
        }
        return resources

    # Truly integrated SoCs where the GPU and CPU share ONE physical pool and
    # nvidia-smi (and NVML) report GPU memory as [N/A] / NOT_SUPPORTED, so there
    # is no separate VRAM figure to show. Matched case-insensitively vs GPU name.
    # NOTE: Grace Hopper (GH200) and Grace-Blackwell superchips (GB200/GB300) are
    # deliberately EXCLUDED — they have discrete HBM that nvidia-smi reports
    # correctly, so their VRAM bar is meaningful and must stay visible.
    _UNIFIED_GPU_TAGS = ("GB10", "JETSON", "ORIN", "XAVIER", "THOR", "TEGRA")

    @staticmethod
    def _detect_unified_memory(gpu_info: Dict[str, Any], ram_info: Dict[str, Any]) -> bool:
        """Detect a unified-memory architecture (e.g. DGX Spark GB10, Grace Hopper).

        Two independent signals, either is sufficient:
          1. GPU name matches a known integrated/unified SoC.
          2. GPU-visible memory is about as large as total system RAM — physically
             only possible when they are the same shared pool (a discrete GPU's
             VRAM is far smaller than system RAM).
        torch is intentionally not imported here (webui avoids the torch dep).
        """
        devices = gpu_info.get("devices", [])
        if not gpu_info.get("available") or not devices:
            return False

        # Signal 1: known integrated SoC by name.
        for dev in devices:
            name = str(dev.get("name", "")).upper()
            if any(tag in name for tag in TrainingManager._UNIFIED_GPU_TAGS):
                return True

        # Signal 2: GPU memory ~= system RAM (shared pool). Conservative ratio so
        # high-VRAM discrete cards on small-RAM hosts don't false-positive.
        ram_total_mb = float(ram_info.get("total_gb", 0) or 0) * 1024.0
        gpu_total_mb = max((float(d.get("memory_total_mb", 0) or 0) for d in devices), default=0.0)
        if ram_total_mb > 0 and gpu_total_mb > 0:
            if gpu_total_mb / ram_total_mb >= 0.85:
                return True

        return False

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
                        # nvidia-smi prints "[N/A]" / "[Not Supported]" for fields a
                        # driver can't read — integrated GPUs like the GB10 return
                        # [N/A] for memory. Parse numerics defensively (-> None) so a
                        # placeholder doesn't crash the whole resource poll; the
                        # frontend renders "N/A" for missing values.
                        try:
                            index = int(parts[0])
                        except ValueError:
                            continue
                        gpu_info["devices"].append(
                            {
                                "index": index,
                                "name": parts[1],
                                "utilization_percent": TrainingManager._safe_float(parts[2]),
                                "memory_used_mb": TrainingManager._safe_float(parts[3]),
                                "memory_total_mb": TrainingManager._safe_float(parts[4]),
                                "temperature_c": TrainingManager._safe_float(parts[5]),
                            }
                        )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        return gpu_info

    @staticmethod
    def _safe_float(value: str) -> Optional[float]:
        """Parse an nvidia-smi numeric field, returning None for non-numeric
        placeholders like '[N/A]' or '[Not Supported]' (seen on integrated GPUs)."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

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
            # Fallback 1: read /proc/meminfo on Linux
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
                # Fallback 2: Windows via ctypes kernel32
                try:
                    import ctypes

                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]

                    mem_status = MEMORYSTATUSEX()
                    mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status))  # type: ignore[attr-defined]

                    total_gb = mem_status.ullTotalPhys / (1024**3)
                    avail_gb = mem_status.ullAvailPhys / (1024**3)
                    used_gb = total_gb - avail_gb
                    ram_info = {
                        "total_gb": round(total_gb, 2),
                        "used_gb": round(used_gb, 2),
                        "available_gb": round(avail_gb, 2),
                        "percent": round(used_gb / total_gb * 100, 1) if total_gb > 0 else 0,
                    }
                except (OSError, AttributeError, ValueError):
                    ram_info = {"error": "Unable to read memory info"}

        return ram_info


# Module-level singleton instance
training_manager = TrainingManager()

__all__ = [
    "TrainingManager",
    "TrainingStatus",
    "training_manager",
]
