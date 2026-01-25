"""
SHEPHERD-Advanced Model Trainer
===============================
Complete training loop for ShepherdGNN with multi-task learning.

Module: src/training/trainer.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/training/trainer.py

Purpose:
    Provide a complete training loop for the ShepherdGNN model, including:
    - Multi-task loss optimization (diagnosis, link prediction, contrastive)
    - Mixed precision training (FP16/BF16) for memory efficiency
    - Gradient accumulation for effective larger batch sizes
    - Learning rate scheduling with warmup
    - Evaluation and validation loops
    - Callback system integration (checkpoints, early stopping, logging)

Components:
    - TrainerConfig: Training hyperparameters and settings
    - Trainer: Main training loop orchestrator
    - TrainingState: Tracks epoch, step, and metric history

Dependencies:
    - torch: Core training operations, autocast, GradScaler
    - torch.nn: Module, Optimizer base classes
    - torch.optim: AdamW, lr_scheduler
    - src.training.loss_functions: MultiTaskLoss, LossConfig
    - src.training.callbacks: CallbackList, EarlyStopping, ModelCheckpoint
    - src.kg.data_loader: DiagnosisDataLoader
    - src.utils.metrics: DiagnosisMetrics
    - src.models.gnn.shepherd_gnn: ShepherdGNN, PhenotypeDiseaseMatcher

Input:
    - model: ShepherdGNN instance
    - train_dataloader: DiagnosisDataLoader for training
    - val_dataloader: DiagnosisDataLoader for validation (optional)
    - config: TrainerConfig with hyperparameters

Output:
    - Trained model saved to checkpoints directory
    - Training metrics logged to file
    - Final evaluation metrics dict

Called by:
    - scripts/train_model.py (main entry point)

Version: 1.0.0
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    OneCycleLR,
    LinearLR,
    SequentialLR,
)
from torch.cuda.amp import GradScaler, autocast

from src.training.loss_functions import LossConfig, MultiTaskLoss
from src.training.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    EarlyStoppingConfig,
    ModelCheckpoint,
    ModelCheckpointConfig,
    MetricsLogger,
    MetricsLoggerConfig,
    LearningRateMonitor,
    GradientClipping,
)
from src.utils.metrics import DiagnosisMetrics, RankingMetrics

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class TrainerConfig:
    """訓練器配置"""
    # Basic training
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Batch and accumulation
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Learning rate scheduling
    scheduler_type: str = "cosine"  # "cosine", "onecycle", "linear", "none"
    warmup_steps: int = 500
    warmup_ratio: float = 0.1  # Alternative: fraction of total steps
    min_lr_ratio: float = 0.01  # Final LR = learning_rate * min_lr_ratio

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16", "bfloat16"

    # Validation
    eval_every_n_epochs: int = 1
    eval_every_n_steps: Optional[int] = None

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 3
    save_last: bool = True

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_monitor: str = "val_mrr"
    early_stopping_mode: str = "max"

    # Logging
    log_dir: str = "logs"
    log_every_n_steps: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    seed: Optional[int] = 42

    # Loss configuration
    loss_config: Optional[LossConfig] = None


@dataclass
class TrainingState:
    """訓練狀態追蹤"""
    epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None
    best_epoch: int = 0
    train_loss_history: List[float] = field(default_factory=list)
    val_metric_history: List[Dict[str, float]] = field(default_factory=list)


# =============================================================================
# Trainer
# =============================================================================
class Trainer:
    """
    ShepherdGNN 訓練器

    完整的訓練循環，包含：
    - 多任務損失優化
    - 混合精度訓練
    - 梯度累積
    - 學習率調度
    - 回調系統
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: Iterator[Dict[str, Any]],
        val_dataloader: Optional[Iterator[Dict[str, Any]]] = None,
        config: Optional[TrainerConfig] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        """
        Args:
            model: ShepherdGNN 模型
            train_dataloader: 訓練資料載入器
            val_dataloader: 驗證資料載入器 (可選)
            config: 訓練配置
            loss_fn: 損失函數 (預設使用 MultiTaskLoss)
            optimizer: 優化器 (預設使用 AdamW)
            scheduler: 學習率調度器 (預設根據 config 創建)
            callbacks: 回調列表
        """
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)

        # Model
        self.model = model.to(self.device)

        # Data
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Loss
        self.loss_fn = loss_fn or MultiTaskLoss(
            config=self.config.loss_config or LossConfig()
        )

        # Optimizer
        self.optimizer = optimizer or self._create_optimizer()

        # Scheduler (created after optimizer)
        self.scheduler = scheduler or self._create_scheduler()

        # Mixed precision
        self._setup_amp()

        # Callbacks
        self.callbacks = CallbackList(callbacks or self._create_default_callbacks())

        # Metrics
        self.metrics = DiagnosisMetrics()

        # State
        self.state = TrainingState()

        # Set seed
        if self.config.seed is not None:
            self._set_seed(self.config.seed)

    def _create_optimizer(self) -> Optimizer:
        """創建優化器"""
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        param_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def _create_scheduler(self) -> Optional[LRScheduler]:
        """創建學習率調度器"""
        if self.config.scheduler_type == "none":
            return None

        # Estimate total steps
        total_steps = self._estimate_total_steps()

        # Warmup steps
        if self.config.warmup_steps > 0:
            warmup_steps = self.config.warmup_steps
        else:
            warmup_steps = int(total_steps * self.config.warmup_ratio)

        if self.config.scheduler_type == "cosine":
            # Warmup + Cosine decay
            warmup = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.config.learning_rate * self.config.min_lr_ratio,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )

        elif self.config.scheduler_type == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=self.config.warmup_ratio,
                div_factor=25,
                final_div_factor=1 / self.config.min_lr_ratio,
            )

        elif self.config.scheduler_type == "linear":
            return LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_lr_ratio,
                total_iters=total_steps,
            )

        return None

    def _estimate_total_steps(self) -> int:
        """估算總訓練步數"""
        try:
            steps_per_epoch = len(self.train_dataloader)
        except TypeError:
            # Iterator doesn't have len
            steps_per_epoch = 1000  # Default estimate

        return (
            self.config.num_epochs * steps_per_epoch
            // self.config.gradient_accumulation_steps
        )

    def _setup_amp(self) -> None:
        """設置混合精度訓練"""
        self.use_amp = self.config.use_amp and self.device.type == "cuda"

        if self.use_amp:
            if self.config.amp_dtype == "bfloat16":
                self.amp_dtype = torch.bfloat16
                self.scaler = None  # BF16 doesn't need scaling
            else:
                self.amp_dtype = torch.float16
                self.scaler = GradScaler()
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

    def _create_default_callbacks(self) -> List[Callback]:
        """創建預設回調"""
        callbacks = []

        # Early stopping
        callbacks.append(EarlyStopping(
            config=EarlyStoppingConfig(
                monitor=self.config.early_stopping_monitor,
                mode=self.config.early_stopping_mode,
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
            )
        ))

        # Model checkpoint
        callbacks.append(ModelCheckpoint(
            config=ModelCheckpointConfig(
                dirpath=self.config.checkpoint_dir,
                filename="model-{epoch:02d}-{val_mrr:.4f}",
                monitor=self.config.early_stopping_monitor,
                mode=self.config.early_stopping_mode,
                save_top_k=self.config.save_top_k,
                save_last=self.config.save_last,
            )
        ))

        # Metrics logger
        callbacks.append(MetricsLogger(
            config=MetricsLoggerConfig(
                log_dir=self.config.log_dir,
                log_to_file=True,
                log_to_console=True,
                console_interval=self.config.log_every_n_steps,
            )
        ))

        # Learning rate monitor
        callbacks.append(LearningRateMonitor())

        # Gradient clipping monitor
        callbacks.append(GradientClipping(max_norm=self.config.max_grad_norm))

        return callbacks

    def _set_seed(self, seed: int) -> None:
        """設置隨機種子"""
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # =========================================================================
    # Training Loop
    # =========================================================================
    def train(self) -> Dict[str, float]:
        """
        執行完整訓練循環

        Returns:
            最終評估指標
        """
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Config: {self.config}")

        self.callbacks.on_train_begin(self)

        try:
            for epoch in range(self.config.num_epochs):
                self.state.epoch = epoch

                # Training epoch
                train_metrics = self._train_epoch(epoch)

                # Validation
                val_metrics = {}
                if (
                    self.val_dataloader is not None
                    and (epoch + 1) % self.config.eval_every_n_epochs == 0
                ):
                    val_metrics = self._validate(epoch)

                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}

                # Callbacks
                self.callbacks.on_epoch_end(self, epoch, epoch_metrics)

                # Check early stopping
                if self.callbacks.should_stop():
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        finally:
            self.callbacks.on_train_end(self)

        # Final evaluation
        if self.val_dataloader is not None:
            final_metrics = self._validate(self.state.epoch)
        else:
            final_metrics = {}

        return final_metrics

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """訓練一個 epoch"""
        self.model.train()
        self.callbacks.on_epoch_begin(self, epoch)

        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0

        epoch_start_time = time.time()

        for batch_idx, batch_data in enumerate(self.train_dataloader):
            self.callbacks.on_batch_begin(self, batch_idx)

            # Move data to device
            batch = self._move_to_device(batch_data["batch"])
            subgraph_x = self._move_to_device(batch_data["subgraph_x_dict"])
            subgraph_edges = self._move_to_device(batch_data["subgraph_edge_index_dict"])

            # Forward pass with mixed precision
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                # Forward through GNN
                node_embeddings = self.model(subgraph_x, subgraph_edges)

                # Compute diagnosis scores
                model_outputs = self._compute_model_outputs(
                    node_embeddings, batch, subgraph_x, subgraph_edges
                )

                # Compute loss
                loss, loss_dict = self.loss_fn(batch, model_outputs)

                # Scale for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item()

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()

                # Zero gradients
                self.optimizer.zero_grad()

                # Update state
                self.state.global_step += 1
                total_loss += accumulated_loss * self.config.gradient_accumulation_steps
                accumulated_loss = 0.0
                num_batches += 1

            # Batch metrics
            batch_metrics = {
                "batch_loss": loss.item() * self.config.gradient_accumulation_steps,
                **{f"loss_{k}": v for k, v in loss_dict.items()},
            }

            self.callbacks.on_batch_end(self, batch_idx, batch_metrics)

            # Step-based validation
            if (
                self.config.eval_every_n_steps is not None
                and self.state.global_step % self.config.eval_every_n_steps == 0
                and self.val_dataloader is not None
            ):
                self._validate(epoch)
                self.model.train()

        # Epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / max(num_batches, 1)

        self.state.train_loss_history.append(avg_loss)

        return {
            "train_loss": avg_loss,
            "epoch_time": epoch_time,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def _validate(self, epoch: int) -> Dict[str, float]:
        """驗證模型"""
        self.model.eval()
        self.callbacks.on_validation_begin(self)

        all_predictions = []
        all_ground_truths = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_data in self.val_dataloader:
                # Move data to device
                batch = self._move_to_device(batch_data["batch"])
                subgraph_x = self._move_to_device(batch_data["subgraph_x_dict"])
                subgraph_edges = self._move_to_device(batch_data["subgraph_edge_index_dict"])

                # Forward pass
                with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    node_embeddings = self.model(subgraph_x, subgraph_edges)
                    model_outputs = self._compute_model_outputs(
                        node_embeddings, batch, subgraph_x, subgraph_edges
                    )

                    # Compute loss
                    loss, _ = self.loss_fn(batch, model_outputs)

                total_loss += loss.item()
                num_batches += 1

                # Collect predictions for metrics
                if "diagnosis_scores" in model_outputs:
                    scores = model_outputs["diagnosis_scores"]
                    targets = batch.get("diagnosis_targets", batch.get("disease_ids"))

                    # Get ranked predictions
                    _, indices = scores.sort(dim=-1, descending=True)

                    for i in range(scores.size(0)):
                        pred_indices = indices[i].tolist()
                        # Convert indices to IDs (simplified)
                        all_predictions.append([str(idx) for idx in pred_indices[:20]])
                        all_ground_truths.append(str(targets[i].item()))

        # Compute metrics
        val_metrics = {}

        if all_predictions:
            ranking_metrics = RankingMetrics().compute_all(
                all_predictions, all_ground_truths
            )
            val_metrics.update({f"val_{k}": v for k, v in ranking_metrics.items()})

        val_metrics["val_loss"] = total_loss / max(num_batches, 1)

        # Track best metric
        monitor_value = val_metrics.get(self.config.early_stopping_monitor)
        if monitor_value is not None:
            is_best = self._is_best_metric(monitor_value)
            if is_best:
                self.state.best_metric = monitor_value
                self.state.best_epoch = epoch

        self.state.val_metric_history.append(val_metrics)
        self.callbacks.on_validation_end(self, val_metrics)

        return val_metrics

    def _is_best_metric(self, current: float) -> bool:
        """檢查是否為最佳指標"""
        if self.state.best_metric is None:
            return True

        if self.config.early_stopping_mode == "max":
            return current > self.state.best_metric
        else:
            return current < self.state.best_metric

    def _move_to_device(self, data: Union[Dict, Tensor, Any]) -> Union[Dict, Tensor, Any]:
        """移動資料到設備"""
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        elif isinstance(data, Tensor):
            return data.to(self.device)
        else:
            return data

    def _compute_model_outputs(
        self,
        node_embeddings: Dict[str, Tensor],
        batch: Dict[str, Any],
        subgraph_x: Dict[str, Tensor],
        subgraph_edges: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Tensor]:
        """
        計算模型輸出

        整合節點嵌入並計算診斷分數
        """
        outputs = {"node_embeddings": node_embeddings}

        # Get disease embeddings
        disease_emb = node_embeddings.get("disease")
        if disease_emb is None:
            return outputs

        # Get phenotype embeddings
        phenotype_emb = node_embeddings.get("phenotype")
        if phenotype_emb is None:
            return outputs

        # Get batch phenotype indices
        phenotype_ids = batch.get("phenotype_ids")
        phenotype_mask = batch.get("phenotype_mask")
        disease_ids = batch.get("disease_ids")

        if phenotype_ids is None or disease_ids is None:
            return outputs

        # Gather patient phenotype embeddings
        batch_size = phenotype_ids.size(0)
        max_phenotypes = phenotype_ids.size(1)

        # Handle invalid indices
        valid_ids = phenotype_ids.clamp(min=0, max=phenotype_emb.size(0) - 1)
        patient_phenotype_emb = phenotype_emb[valid_ids.view(-1)].view(
            batch_size, max_phenotypes, -1
        )

        # Aggregate patient phenotypes (masked mean)
        if phenotype_mask is not None:
            mask = phenotype_mask.unsqueeze(-1).float()
            summed = (patient_phenotype_emb * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            patient_embeddings = summed / counts
        else:
            patient_embeddings = patient_phenotype_emb.mean(dim=1)

        outputs["patient_embeddings"] = patient_embeddings

        # Get disease embeddings for targets
        valid_disease_ids = disease_ids.clamp(min=0, max=disease_emb.size(0) - 1)
        target_disease_emb = disease_emb[valid_disease_ids]
        outputs["disease_embeddings"] = target_disease_emb

        # Compute diagnosis scores (similarity to all diseases)
        patient_norm = torch.nn.functional.normalize(patient_embeddings, dim=-1)
        disease_norm = torch.nn.functional.normalize(disease_emb, dim=-1)
        diagnosis_scores = torch.mm(patient_norm, disease_norm.t())

        outputs["diagnosis_scores"] = diagnosis_scores
        outputs["diagnosis_targets"] = disease_ids

        return outputs

    # =========================================================================
    # Checkpoint Operations
    # =========================================================================
    def save_checkpoint(self, filepath: Union[str, Path]) -> None:
        """保存檢查點"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state": self.state.__dict__,
            "config": self.config.__dict__,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: Union[str, Path]) -> None:
        """載入檢查點"""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore state
        self.state.epoch = checkpoint.get("epoch", 0)
        self.state.global_step = checkpoint.get("global_step", 0)

        if "state" in checkpoint:
            for key, value in checkpoint["state"].items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)

        logger.info(f"Checkpoint loaded from {filepath}")


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "TrainerConfig",
    "TrainingState",
    "Trainer",
]
