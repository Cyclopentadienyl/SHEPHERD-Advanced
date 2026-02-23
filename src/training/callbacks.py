"""
SHEPHERD-Advanced Training Callbacks
=====================================
Callback system for training monitoring, checkpointing, and control flow.

Module: src/training/callbacks.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/training/callbacks.py

Purpose:
    Provide extensible callback hooks for the training loop, enabling:
    - Early stopping based on validation metrics
    - Model checkpoint saving with top-k retention
    - Learning rate monitoring and logging
    - Training metrics logging to file and console
    - Gradient norm tracking

Components:
    - Callback: Abstract base class defining hook interface
    - CallbackList: Manager for multiple callbacks
    - EarlyStopping: Stop training when metric plateaus
    - ModelCheckpoint: Save best/last model weights
    - LearningRateMonitor: Track LR changes across epochs
    - MetricsLogger: Log metrics to JSON and console
    - ProgressBar: tqdm-based progress display
    - GradientClipping: Monitor gradient norms

Dependencies:
    - torch: Model state_dict, tensor operations
    - torch.nn: Module type hints
    - json: Metrics file serialization
    - logging: Console output
    - pathlib: Checkpoint path handling
    - datetime: Timestamp generation
    - abc: Abstract base class
    - tqdm (optional): Progress bar display

Input (Callback hooks):
    - trainer: Trainer instance with model, optimizer, scheduler attributes
    - epoch: Current epoch number
    - batch_idx: Current batch index
    - logs: Dict[str, float] - Current metrics

Output:
    - Side effects: File I/O (checkpoints, logs), training state modification
    - CallbackList.should_stop() -> bool: Early stopping signal

Called by:
    - src/training/trainer.py (training loop hooks)

Note:
    Callbacks use forward references ("Trainer") for type hints since the
    Trainer class imports this module. At runtime, any object with model,
    optimizer, scheduler, and config attributes will work.

Version: 1.0.0
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Base Callback
# =============================================================================
class Callback(ABC):
    """回調基類"""

    def on_train_begin(self, trainer: "Trainer", **kwargs) -> None:
        """訓練開始時調用"""
        pass

    def on_train_end(self, trainer: "Trainer", **kwargs) -> None:
        """訓練結束時調用"""
        pass

    def on_epoch_begin(self, trainer: "Trainer", epoch: int, **kwargs) -> None:
        """每個 epoch 開始時調用"""
        pass

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float], **kwargs
    ) -> None:
        """每個 epoch 結束時調用"""
        pass

    def on_batch_begin(
        self, trainer: "Trainer", batch_idx: int, **kwargs
    ) -> None:
        """每個 batch 開始時調用"""
        pass

    def on_batch_end(
        self, trainer: "Trainer", batch_idx: int, logs: Dict[str, float], **kwargs
    ) -> None:
        """每個 batch 結束時調用"""
        pass

    def on_validation_begin(self, trainer: "Trainer", **kwargs) -> None:
        """驗證開始時調用"""
        pass

    def on_validation_end(
        self, trainer: "Trainer", logs: Dict[str, float], **kwargs
    ) -> None:
        """驗證結束時調用"""
        pass


# =============================================================================
# Early Stopping
# =============================================================================
@dataclass
class EarlyStoppingConfig:
    """早停配置"""
    monitor: str = "val_loss"
    mode: str = "min"  # "min" or "max"
    patience: int = 10
    min_delta: float = 1e-4
    restore_best_weights: bool = True


class EarlyStopping(Callback):
    """
    早停回調

    監控指標在指定步數內未改善時停止訓練
    """

    def __init__(self, config: Optional[EarlyStoppingConfig] = None):
        self.config = config or EarlyStoppingConfig()
        self.best_value: Optional[float] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.stopped_epoch: int = 0
        self.best_weights: Optional[Dict[str, Any]] = None
        self.should_stop: bool = False

    def on_train_begin(self, trainer: "Trainer", **kwargs) -> None:
        self.best_value = None
        self.best_epoch = 0
        self.counter = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.should_stop = False

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float], **kwargs
    ) -> None:
        current_value = logs.get(self.config.monitor)

        if current_value is None:
            logger.warning(
                f"EarlyStopping: metric '{self.config.monitor}' not found in logs"
            )
            return

        if self._is_improvement(current_value):
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0

            if self.config.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
                }
        else:
            self.counter += 1
            if self.counter >= self.config.patience:
                self.should_stop = True
                self.stopped_epoch = epoch
                logger.info(
                    f"EarlyStopping: stopped at epoch {epoch}. "
                    f"Best {self.config.monitor}: {self.best_value:.4f} at epoch {self.best_epoch}"
                )

    def on_train_end(self, trainer: "Trainer", **kwargs) -> None:
        if self.config.restore_best_weights and self.best_weights is not None:
            logger.info(
                f"EarlyStopping: restoring best weights from epoch {self.best_epoch}"
            )
            trainer.model.load_state_dict(self.best_weights)

    def _is_improvement(self, current: float) -> bool:
        if self.best_value is None:
            return True

        if self.config.mode == "min":
            return current < self.best_value - self.config.min_delta
        else:
            return current > self.best_value + self.config.min_delta


# =============================================================================
# Model Checkpoint
# =============================================================================
@dataclass
class ModelCheckpointConfig:
    """模型檢查點配置"""
    dirpath: str = "checkpoints"
    filename: str = "model-{epoch:02d}-{val_loss:.4f}"
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 3
    save_last: bool = True
    save_weights_only: bool = False


class ModelCheckpoint(Callback):
    """
    模型檢查點回調

    定期保存模型，保留最佳的 k 個檢查點
    """

    def __init__(self, config: Optional[ModelCheckpointConfig] = None):
        self.config = config or ModelCheckpointConfig()
        self.best_k_models: List[Dict[str, Any]] = []
        self.dirpath = Path(self.config.dirpath)
        self.last_filepath: Optional[Path] = None

    def on_train_begin(self, trainer: "Trainer", **kwargs) -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.best_k_models = []

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float], **kwargs
    ) -> None:
        current_value = logs.get(self.config.monitor)

        if current_value is None:
            logger.warning(
                f"ModelCheckpoint: metric '{self.config.monitor}' not found"
            )
            return

        # 格式化檔名
        filename = self.config.filename.format(
            epoch=epoch,
            **{k: v for k, v in logs.items() if isinstance(v, (int, float))}
        )
        filepath = self.dirpath / f"{filename}.pt"

        # 檢查是否應該保存
        should_save = self._should_save(current_value)

        if should_save:
            self._save_checkpoint(trainer, filepath, epoch, logs)

            # 更新最佳模型列表
            self.best_k_models.append({
                "filepath": filepath,
                "value": current_value,
                "epoch": epoch,
            })

            # 排序並移除多餘的檢查點
            self._sort_and_prune()

        # 保存最後一個檢查點
        if self.config.save_last:
            last_path = self.dirpath / "last.pt"
            self._save_checkpoint(trainer, last_path, epoch, logs)
            self.last_filepath = last_path

    def _should_save(self, current_value: float) -> bool:
        if len(self.best_k_models) < self.config.save_top_k:
            return True

        # 比較最差的
        worst = self.best_k_models[-1]["value"]

        if self.config.mode == "min":
            return current_value < worst
        else:
            return current_value > worst

    def _save_checkpoint(
        self,
        trainer: "Trainer",
        filepath: Path,
        epoch: int,
        logs: Dict[str, float],
    ) -> None:
        if self.config.save_weights_only:
            checkpoint = {"state_dict": trainer.model.state_dict()}
        else:
            checkpoint = {
                "epoch": epoch,
                "state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "logs": logs,
                "config": trainer.config.__dict__ if hasattr(trainer, "config") else {},
            }

            if trainer.scheduler is not None:
                checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"ModelCheckpoint: saved to {filepath}")

    def _sort_and_prune(self) -> None:
        # 排序
        reverse = self.config.mode == "max"
        self.best_k_models.sort(key=lambda x: x["value"], reverse=reverse)

        # 移除多餘的檢查點
        while len(self.best_k_models) > self.config.save_top_k:
            removed = self.best_k_models.pop()
            if removed["filepath"].exists():
                removed["filepath"].unlink()
                logger.info(f"ModelCheckpoint: removed {removed['filepath']}")


# =============================================================================
# Learning Rate Monitor
# =============================================================================
class LearningRateMonitor(Callback):
    """
    學習率監控回調

    記錄每個 epoch 的學習率
    """

    def __init__(self):
        self.lr_history: List[Dict[str, float]] = []

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float], **kwargs
    ) -> None:
        if trainer.optimizer is not None:
            lrs = {}
            for i, param_group in enumerate(trainer.optimizer.param_groups):
                lrs[f"lr_group_{i}"] = param_group["lr"]

            self.lr_history.append({"epoch": epoch, **lrs})
            logs.update(lrs)


# =============================================================================
# Metrics Logger
# =============================================================================
@dataclass
class MetricsLoggerConfig:
    """指標記錄器配置"""
    log_dir: str = "logs"
    log_to_file: bool = True
    log_to_console: bool = True
    console_interval: int = 10  # 每 N 個 batch 輸出一次


class MetricsLogger(Callback):
    """
    指標記錄回調

    記錄訓練和驗證指標到文件和控制台
    """

    def __init__(self, config: Optional[MetricsLoggerConfig] = None):
        self.config = config or MetricsLoggerConfig()
        self.log_dir = Path(self.config.log_dir)
        self.train_history: List[Dict[str, Any]] = []
        self.val_history: List[Dict[str, Any]] = []
        self.batch_count: int = 0
        self.epoch_metrics: Dict[str, List[float]] = {}
        self._progress_file: Optional[Path] = None
        self._last_progress_write: float = 0.0

    def _write_progress(self, data: Dict[str, Any]) -> None:
        """Write lightweight progress file for WebUI polling."""
        if self._progress_file is None:
            return
        try:
            tmp_path = self._progress_file.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(data, f)
            tmp_path.replace(self._progress_file)
        except OSError:
            pass

    def on_train_begin(self, trainer: "Trainer", **kwargs) -> None:
        if self.config.log_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.json"
            self._progress_file = self.log_dir / "train_progress.json"

        self.train_history = []
        self.val_history = []

        self._write_progress({
            "phase": "initializing",
            "total_epochs": trainer.config.num_epochs,
            "timestamp": datetime.now().isoformat(),
        })

    def on_epoch_begin(self, trainer: "Trainer", epoch: int, **kwargs) -> None:
        self.batch_count = 0
        self.epoch_metrics = {}
        self._last_progress_write = 0.0

        try:
            total_batches = len(trainer.train_dataloader)
        except TypeError:
            total_batches = None

        self._write_progress({
            "phase": "training",
            "epoch": epoch,
            "total_epochs": trainer.config.num_epochs,
            "batch": 0,
            "total_batches": total_batches,
            "timestamp": datetime.now().isoformat(),
        })

    def on_batch_end(
        self, trainer: "Trainer", batch_idx: int, logs: Dict[str, float], **kwargs
    ) -> None:
        self.batch_count += 1

        # 累積指標
        for key, value in logs.items():
            if key not in self.epoch_metrics:
                self.epoch_metrics[key] = []
            self.epoch_metrics[key].append(value)

        # 輸出到控制台
        if (
            self.config.log_to_console
            and self.batch_count % self.config.console_interval == 0
        ):
            metrics_str = " | ".join(
                f"{k}: {v:.4f}" for k, v in logs.items()
            )
            logger.info(f"Batch {batch_idx}: {metrics_str}")

        # Throttled progress file update (every 5 seconds)
        now = time.time()
        if now - self._last_progress_write >= 5.0:
            self._last_progress_write = now

            try:
                total_batches = len(trainer.train_dataloader)
            except TypeError:
                total_batches = None

            self._write_progress({
                "phase": "training",
                "epoch": trainer.state.epoch,
                "total_epochs": trainer.config.num_epochs,
                "batch": batch_idx + 1,
                "total_batches": total_batches,
                "batch_loss": logs.get("batch_loss"),
                "timestamp": datetime.now().isoformat(),
            })

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float], **kwargs
    ) -> None:
        # 計算 epoch 平均
        epoch_avg = {
            k: sum(v) / len(v) for k, v in self.epoch_metrics.items() if v
        }

        record = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **epoch_avg,
            **logs,
        }

        self.train_history.append(record)

        if self.config.log_to_console:
            metrics_str = " | ".join(
                f"{k}: {v:.4f}" for k, v in logs.items() if isinstance(v, (int, float))
            )
            logger.info(f"Epoch {epoch}: {metrics_str}")

        if self.config.log_to_file:
            self._save_history()

    def on_validation_end(
        self, trainer: "Trainer", logs: Dict[str, float], **kwargs
    ) -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            **logs,
        }
        self.val_history.append(record)

        if self.config.log_to_console:
            metrics_str = " | ".join(
                f"{k}: {v:.4f}" for k, v in logs.items() if isinstance(v, (int, float))
            )
            logger.info(f"Validation: {metrics_str}")

    def on_train_end(self, trainer: "Trainer", **kwargs) -> None:
        self._write_progress({
            "phase": "completed",
            "epoch": trainer.state.epoch,
            "total_epochs": trainer.config.num_epochs,
            "timestamp": datetime.now().isoformat(),
        })

    def _save_history(self) -> None:
        data = {
            "train": self.train_history,
            "validation": self.val_history,
        }
        with open(self.log_file, "w") as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Progress Bar (Optional)
# =============================================================================
class ProgressBar(Callback):
    """
    進度條回調

    顯示訓練進度
    """

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.pbar = None

    def on_train_begin(self, trainer: "Trainer", **kwargs) -> None:
        try:
            from tqdm import tqdm
            self.pbar = tqdm(total=self.total_epochs, desc="Training", unit="epoch")
        except ImportError:
            logger.warning("tqdm not installed, progress bar disabled")
            self.pbar = None

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float], **kwargs
    ) -> None:
        if self.pbar is not None:
            self.pbar.update(1)
            # 更新進度條描述
            desc = " | ".join(
                f"{k}: {v:.4f}"
                for k, v in logs.items()
                if isinstance(v, (int, float)) and "loss" in k.lower()
            )
            self.pbar.set_postfix_str(desc)

    def on_train_end(self, trainer: "Trainer", **kwargs) -> None:
        if self.pbar is not None:
            self.pbar.close()


# =============================================================================
# Gradient Clipping
# =============================================================================
class GradientClipping(Callback):
    """
    梯度裁剪回調
    """

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.grad_norms: List[float] = []

    def on_batch_end(
        self, trainer: "Trainer", batch_idx: int, logs: Dict[str, float], **kwargs
    ) -> None:
        # 記錄梯度範數 (在 optimizer.step() 之前)
        total_norm = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(self.norm_type)
                total_norm += param_norm.item() ** self.norm_type
        total_norm = total_norm ** (1.0 / self.norm_type)

        self.grad_norms.append(total_norm)
        logs["grad_norm"] = total_norm


# =============================================================================
# Callback List
# =============================================================================
class CallbackList:
    """回調列表管理器"""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_train_begin(self, trainer: "Trainer", **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(trainer, **kwargs)

    def on_train_end(self, trainer: "Trainer", **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer, **kwargs)

    def on_epoch_begin(self, trainer: "Trainer", epoch: int, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch, **kwargs)

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float], **kwargs
    ) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, logs, **kwargs)

    def on_batch_begin(self, trainer: "Trainer", batch_idx: int, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch_idx, **kwargs)

    def on_batch_end(
        self, trainer: "Trainer", batch_idx: int, logs: Dict[str, float], **kwargs
    ) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_idx, logs, **kwargs)

    def on_validation_begin(self, trainer: "Trainer", **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_validation_begin(trainer, **kwargs)

    def on_validation_end(
        self, trainer: "Trainer", logs: Dict[str, float], **kwargs
    ) -> None:
        for callback in self.callbacks:
            callback.on_validation_end(trainer, logs, **kwargs)

    def should_stop(self) -> bool:
        """檢查是否有回調請求停止訓練"""
        for callback in self.callbacks:
            if isinstance(callback, EarlyStopping) and callback.should_stop:
                return True
        return False


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "EarlyStoppingConfig",
    "ModelCheckpoint",
    "ModelCheckpointConfig",
    "LearningRateMonitor",
    "MetricsLogger",
    "MetricsLoggerConfig",
    "ProgressBar",
    "GradientClipping",
]
