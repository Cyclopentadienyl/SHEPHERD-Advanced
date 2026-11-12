"""
SHEPHERD-Advanced Model Trainer
===============================
The training loop of a model bundle, driven by its stage's task module.

Module: src/training/trainer.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/training/trainer.py

Purpose:
    Run the training loop a stage's task module drives (README §3.1 lifecycle
    hooks: `training_step` and `validate`), owning what every stage shares:
    - Mixed precision training (FP16/BF16) for memory efficiency
    - Gradient accumulation, the epoch's tail applied as one more update
    - Learning rate scheduling with warmup
    - Validation through the task, with the best value and its history
    - Callback system integration (checkpoints, early stopping, logging)
    - Continuing a parent's training as the run plan decided

Components:
    - TrainerConfig: Training hyperparameters and settings
    - Trainer: Main training loop orchestrator
    - TrainingState: Tracks epoch, step, and metric history

Dependencies:
    - torch: Core training operations, autocast, GradScaler
    - torch.nn: Module, Optimizer base classes
    - torch.optim: AdamW, lr_scheduler
    - src.training.callbacks: CallbackList, EarlyStopping, ModelCheckpoint

Input:
    - model: a model bundle (src.models.bundle)
    - train_dataloader: the stage's sampler, yielding the task's batches
    - task: the stage's task module (src.training.lp_task, finetune_task)
    - config: TrainerConfig with hyperparameters

Output:
    - Trained model saved to checkpoints directory
    - Training metrics logged to file
    - Final validation metrics dict

Called by:
    - scripts/train_model.py (main entry point)

Version: 1.0.0
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LRScheduler,
    OneCycleLR,
    LinearLR,
)
from torch.amp import GradScaler, autocast

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
)

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

    # Checkpointing
    checkpoint_dir: str = ""  # empty = auto-derive from data_dir + /checkpoints
    save_top_k: int = 3
    save_last: bool = True

    # Early stopping. Off for the bundle path's reference recipe (README C6):
    # the full schedule runs and ModelCheckpoint keeps the best file.
    early_stopping: bool = True
    early_stopping_patience: int = 10
    # The metric early stopping and ModelCheckpoint select by. None: the
    # task's own selection metric (`task.selection_metric`), which the
    # Trainer puts in its config when it is built.
    early_stopping_monitor: Optional[str] = None
    early_stopping_mode: str = "max"

    # Logging
    log_dir: str = "logs"
    log_every_n_steps: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    seed: Optional[int] = 42

    # Stop at a safe boundary once this many epochs of the run's total progress
    # are complete, recorded as paused (contract M3a-3). Not inherited.
    pause_after_epoch: Optional[int] = None


@dataclass
class TrainingState:
    """訓練狀態追蹤"""
    epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None
    best_epoch: int = 0
    train_loss_history: List[float] = field(default_factory=list)
    val_metric_history: List[Dict[str, float]] = field(default_factory=list)
    # Progress counted three ways (contract M3a-1). `global_step` above is the
    # optimizer attempts; these are the epochs completed, the updates that
    # actually ran (AMP's GradScaler skips one on non-finite gradients) and
    # the scheduler steps taken, which follow the updates.
    completed_epochs: int = 0
    optimizer_updates: int = 0
    scheduler_steps: int = 0


# =============================================================================
# Scheduler input validation
# =============================================================================
def _validate_min_lr_ratio(min_lr_ratio: float, scheduler_type: str) -> None:
    """Validate ``min_lr_ratio`` for the chosen scheduler.

    Rules (matched to PyTorch's actual behaviour, not stricter):
      - ``scheduler_type == "none"``: the ratio is never consumed -> no check.
      - A negative ratio is invalid for every scheduler that uses it. PyTorch's
        LambdaLR (our cosine path) and OneCycleLR do not validate it and would
        silently train at a negative learning rate, so reject it here.
      - ``onecycle`` additionally requires a strictly positive ratio: it computes
        ``final_div_factor = 1 / min_lr_ratio`` (0 -> ZeroDivisionError; OneCycleLR
        does not validate the factor). **That rule is the field spec's**
        (`training_fields.CROSS_FIELD_RULES`), refused here in its one sentence
        (`rule_problem`), the sentence the training process and the job
        runner's start refuse a stated ratio with before a Trainer exists.
      - ``cosine`` / ``linear`` accept 0 (legitimate decay-to-zero via a floor
        factor), so 0 is NOT rejected for them.
    """
    from src.config.training_fields import rule_problem

    if scheduler_type == "none":
        return
    if min_lr_ratio < 0:
        raise ValueError(
            f"min_lr_ratio must be >= 0 (got {min_lr_ratio!r}) for "
            f"scheduler_type={scheduler_type!r}."
        )
    problem = rule_problem("onecycle_requires_positive_min_lr_ratio",
                           {"min_lr_ratio": min_lr_ratio, "scheduler_type": scheduler_type})
    if problem:
        raise ValueError(problem)


# =============================================================================
# Trainer
# =============================================================================
class Trainer:
    """
    The training loop of a model bundle (README §3.1): the stage's task module
    computes each step's loss and the validation metrics; the Trainer keeps
    the optimizer, mixed precision, accumulation, the schedule, callbacks and
    saving.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: Iterator[Dict[str, Any]],
        config: Optional[TrainerConfig] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        callbacks: Optional[List[Callback]] = None,
        *,
        task: Any,
    ):
        """
        Args:
            model: the model bundle to train
            train_dataloader: the stage's sampler, yielding the task's batches
            config: 訓練配置
            optimizer: 優化器 (預設使用 AdamW)
            scheduler: 學習率調度器 (預設根據 config 創建)
            callbacks: 回調列表
            task: the stage's task module (README §3.1 lifecycle hooks),
                required: `training_step(model, batch, device)` returns a
                `StepResult` (a loss, or None for a step with nothing to learn
                from), and `validate(model, device)` returns the validation
                metrics. It is the Trainer's one training path; `None` is
                refused (a TypeError). When the config states no
                `early_stopping_monitor`, the Trainer selects by the task's
                `selection_metric`.
        """
        if task is None:
            raise TypeError("Trainer needs the stage's task module (task): it computes each "
                            "step's loss and the validation metrics")
        self.config = config or TrainerConfig()
        if self.config.early_stopping_monitor is None:
            # A copy: the caller's config is left as given.
            self.config = replace(self.config, early_stopping_monitor=task.selection_metric)
        self.device = torch.device(self.config.device)

        # Model
        self.model = model.to(self.device)
        # Whether some parameter does not train (a frozen part, frozen before
        # the Trainer is built: `ModelBundle.freeze`), so the optimizer below
        # holds none of it. Only then may a step's loss reach no trainable
        # parameter (`_train_epoch`); without one, such a loss is a fault and
        # fails at its backward, as it always did.
        self._has_frozen_parameters = any(not p.requires_grad for p in self.model.parameters())

        # Data
        self.train_dataloader = train_dataloader

        # The stage's task module.
        self.task = task

        # Optimizer
        self.optimizer = optimizer or self._create_optimizer()

        # Scheduler (created after optimizer). `schedule_record` is the schedule
        # as built, recorded in every checkpoint (contract M3a-1).
        self.schedule_record: Dict[str, Any] = {
            "type": "external" if scheduler is not None else self.config.scheduler_type,
        }
        self.scheduler = scheduler or self._create_scheduler()

        # Mixed precision
        self._setup_amp()

        # Callbacks
        self.callbacks = CallbackList(callbacks or self._create_default_callbacks())

        # State
        self.state = TrainingState()

        # Set when the model holds weights from an earlier step than the
        # optimizer (early stopping restores its best at the end of `train`);
        # the checkpoint payload then writes weights only (contract M3a-1).
        self.weights_restored_from: Optional[Dict[str, Any]] = None

        # The active stage: the original schedule, from epoch 0 for
        # `num_epochs`, unless `continue_from` restores another or opens an
        # extension (contract M3a-3, M3a-4). Recorded in every checkpoint.
        self.stage: Dict[str, Any] = {
            "kind": "original",
            "start_epoch": 0,
            "target_epochs": int(self.config.num_epochs),
            "schedule": self.schedule_record,
        }
        # The next epoch index to run; a continued run starts after its
        # parent's last completed epoch.
        self.start_epoch: int = 0
        # Set at the epoch boundary where `pause_after_epoch` stops the run,
        # before the callbacks write that epoch's checkpoint.
        self.pausing: bool = False
        # Set when a stop (KeyboardInterrupt) ended `train`.
        self.interrupted: bool = False
        # The parent's callback state, applied after `on_train_begin` resets it.
        self._pending_callbacks: Optional[Dict[str, Any]] = None

        # Set seed
        if self.config.seed is not None:
            self._set_seed(self.config.seed)

        # Report PyG native-extension status once. PyG falls back to slower
        # pure-Python / torch.scatter_reduce kernels *silently* when the native
        # extensions are missing, so surfacing it here makes a degraded build
        # visible in the training log instead of going unnoticed.
        try:
            from src.utils.pyg_native_check import log_pyg_native_status

            log_pyg_native_status()
        except Exception as exc:  # observability must never break training
            logger.debug("PyG native-extension check skipped: %s", exc)

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
        self.schedule_record = {"type": self.config.scheduler_type,
                                "base_lr": float(self.config.learning_rate)}
        if self.config.scheduler_type == "none":
            return None

        # Fail loud on an illegal min_lr_ratio before building the scheduler. This
        # covers every entry path (CLI / YAML / API / UI) since all funnel here.
        _validate_min_lr_ratio(self.config.min_lr_ratio, self.config.scheduler_type)

        # Estimate total steps
        total_steps = self._estimate_total_steps()

        # Warmup steps (capped to avoid exceeding total steps)
        original_warmup = self.config.warmup_steps
        if self.config.warmup_steps > 0:
            warmup_steps = min(self.config.warmup_steps, total_steps // 2)
        else:
            warmup_steps = int(total_steps * self.config.warmup_ratio)

        # Ensure at least 1 step for main scheduler phase
        warmup_steps = min(warmup_steps, max(0, total_steps - 1))

        self.schedule_record.update({
            "steps_per_epoch": self._steps_per_epoch(),
            "gradient_accumulation_steps": int(self.config.gradient_accumulation_steps),
            "total_steps": int(total_steps),
            "warmup_steps_requested": int(original_warmup),
            "warmup_ratio": float(self.config.warmup_ratio),
            "min_lr_ratio": float(self.config.min_lr_ratio),
        })

        if original_warmup > 0 and warmup_steps < original_warmup:
            logger.warning(
                f"warmup_steps reduced from {original_warmup} to {warmup_steps} "
                f"(total_steps={total_steps}). Consider increasing epochs or reducing warmup."
            )

        if self.config.scheduler_type == "cosine":
            import math
            min_lr_ratio = self.config.min_lr_ratio
            ws = max(1, warmup_steps)
            cs = max(1, total_steps - warmup_steps)
            # What the curve below uses: at least one warmup step, at least one
            # decay step.
            self.schedule_record.update({"warmup_steps_effective": ws, "decay_steps": cs})

            def lr_lambda(current_step: int) -> float:
                if current_step < ws:
                    # Linear warmup: 0.01 -> 1.0
                    return 0.01 + 0.99 * current_step / ws
                # Cosine decay: 1.0 -> min_lr_ratio
                progress = (current_step - ws) / cs
                return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )

            return LambdaLR(self.optimizer, lr_lambda)

        elif self.config.scheduler_type == "onecycle":
            self.schedule_record.update({
                "pct_start": float(self.config.warmup_ratio), "div_factor": 25.0,
                "final_div_factor": 1 / self.config.min_lr_ratio,
                "total_steps_scheduled": max(1, total_steps),
            })
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=max(1, total_steps),
                pct_start=self.config.warmup_ratio,
                div_factor=25,
                final_div_factor=1 / self.config.min_lr_ratio,
            )

        elif self.config.scheduler_type == "linear":
            self.schedule_record.update({
                "start_factor": 1.0, "end_factor": float(self.config.min_lr_ratio),
                "total_iters": max(1, total_steps), "warmup_steps_effective": 0,
            })
            return LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_lr_ratio,
                total_iters=max(1, total_steps),
            )

        return None

    def _steps_per_epoch(self) -> Optional[int]:
        """Batches per epoch, or `None` for a loader without a length."""
        try:
            return int(len(self.train_dataloader))
        except TypeError:
            return None

    def _estimate_total_steps(self) -> int:
        """估算總訓練步數"""
        steps_per_epoch = self._steps_per_epoch()
        if steps_per_epoch is None:
            # Iterator doesn't have len
            steps_per_epoch = 1000  # Default estimate

        # Updates per epoch: every full accumulation group, and the tail as one
        # more (`_train_epoch`), so the schedule counts the updates taken.
        accumulation = self.config.gradient_accumulation_steps
        return self.config.num_epochs * (-(-steps_per_epoch // accumulation))

    def _setup_amp(self) -> None:
        """設置混合精度訓練"""
        self.use_amp = self.config.use_amp and self.device.type == "cuda"

        if self.use_amp:
            if self.config.amp_dtype == "bfloat16":
                self.amp_dtype = torch.bfloat16
                self.scaler = None  # BF16 doesn't need scaling
            else:
                self.amp_dtype = torch.float16
                self.scaler = GradScaler('cuda')
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

    def amp_record(self) -> Dict[str, Any]:
        """AMP as requested and as it took effect: on only on CUDA, and bf16
        without a scaler (contract M3a-1)."""
        return {
            "requested": bool(self.config.use_amp),
            "dtype_requested": str(self.config.amp_dtype),
            "enabled": bool(self.use_amp),
            "dtype": str(self.amp_dtype).replace("torch.", ""),
            "grad_scaler": self.scaler is not None,
        }

    def stage_record(self) -> Dict[str, Any]:
        """The active stage, as recorded in a checkpoint."""
        return dict(self.stage)

    def stage_end(self) -> int:
        """The epoch the active stage ends at (exclusive): its start plus its
        target number of epochs."""
        return int(self.stage["start_epoch"]) + int(self.stage["target_epochs"])

    def _create_default_callbacks(self) -> List[Callback]:
        """創建預設回調"""
        callbacks = []

        # Early stopping
        if self.config.early_stopping:
            callbacks.append(EarlyStopping(
                config=EarlyStoppingConfig(
                    monitor=self.config.early_stopping_monitor,
                    mode=self.config.early_stopping_mode,
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                )
            ))

        # Model checkpoint, named by the monitored metric
        callbacks.append(ModelCheckpoint(
            config=ModelCheckpointConfig(
                dirpath=self.config.checkpoint_dir,
                filename="model-{epoch:02d}-{" + self.config.early_stopping_monitor + ":.4f}",
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

        return callbacks

    def _set_seed(self, seed: int) -> None:
        """設置隨機種子

        The rule is `apply_seed`, at the end of this module. `train_model.train`
        applies it before the data loaders and the model are built, so the seed
        fixes the initial weights (contract M3a). Applied again here, it makes
        the stream training draws from (negatives, shuffling, dropout) start
        from the seed whatever building the loaders and the model consumed.
        An out-of-range seed raises inside `apply_seed`, in `np.random.seed`,
        before anything is logged, as it did when the rule was written out
        here.
        """
        apply_seed(seed)

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
        self.weights_restored_from = None
        self.pausing = False
        self.interrupted = False

        self.callbacks.on_train_begin(self)
        # A continued run's callback state, after `on_train_begin` reset it.
        if self._pending_callbacks is not None:
            self._restore_callbacks(self._pending_callbacks)
            self._pending_callbacks = None

        end = self.stage_end()
        pause = self.config.pause_after_epoch
        if pause is not None and pause >= end:
            logger.info(f"The stage ends at epoch {end}, before pause_after_epoch {pause}.")
        try:
            for epoch in range(self.start_epoch, end):
                self.state.epoch = epoch

                # Training epoch
                train_metrics = self._train_epoch(epoch)

                # Validation
                val_metrics = {}
                if (epoch + 1) % self.config.eval_every_n_epochs == 0:
                    val_metrics = self._validate(epoch)

                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}

                # The epoch is complete before the callbacks see it, so a
                # checkpoint written at its end records it as done, and as
                # paused when the pause stops the run here.
                self.state.completed_epochs = epoch + 1
                self.pausing = pause is not None and epoch + 1 == pause and epoch + 1 < end

                # Callbacks
                self.callbacks.on_epoch_end(self, epoch, epoch_metrics)

                # Check early stopping
                if self.callbacks.should_stop():
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                if self.pausing:
                    logger.info(
                        f"Paused after epoch {epoch + 1} of {end} (pause_after_epoch); "
                        "resume from this run's last.pt.")
                    break

        except KeyboardInterrupt:
            # A stop (SIGINT) inside an epoch: that epoch's progress is lost;
            # the last epoch's `last.pt` is the state to continue from.
            self.interrupted = True
            logger.info("Training interrupted by user")

        finally:
            self.callbacks.on_train_end(self)

        # Final evaluation, of a run that ended, not of one stopped: a stop
        # should end promptly, and its run has no final metrics.
        if not self.interrupted:
            final_metrics = self._validate(self.state.epoch)
        else:
            final_metrics = {}

        return final_metrics

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """訓練一個 epoch

        **The accumulation tail is applied at the epoch's end** (contract
        M3a-1). With gradient accumulation, the batches after the last full
        group used to leave their gradients in `.grad`, where the next epoch's
        first update absorbed them (and the last epoch's were dropped), while
        the schedule's step count assumed a fraction of an update. Now the tail
        of `r` batches is one more update, its gradients rescaled by
        `accumulation / r` so that, like every full group, it applies the mean
        of the batches it holds. No gradient crosses an epoch boundary, and an
        epoch takes `ceil(batches / accumulation)` updates, which is what
        `_estimate_total_steps` counts. With an accumulation of 1 nothing
        changes.

        **`train_loss` is the mean loss per batch** (contract M3a-1 review). It
        used to be the mean, over updates, of each group's summed loss, which
        is `accumulation` times the batch mean, and adding the tail as its own
        update mixed in a mean; one denominator now, independent of updates.
        With an accumulation of 1 the value is unchanged.
        """
        self.model.train()
        self.callbacks.on_epoch_begin(self, epoch)

        accumulation = self.config.gradient_accumulation_steps
        # `train_loss` is the mean loss per batch over the epoch: one unit, one
        # denominator, kept apart from how many updates the batches made.
        loss_sum = 0.0
        batch_count = 0
        pending = 0

        epoch_start_time = time.time()

        set_epoch = getattr(self.train_dataloader, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        skipped_steps = 0

        for batch_idx, batch_data in enumerate(self.train_dataloader):
            self.callbacks.on_batch_begin(self, batch_idx)

            with autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                step = self.task.training_step(self.model, batch_data, self.device)
            if step.loss is None or (not step.loss.requires_grad
                                     and self._has_frozen_parameters):
                # Nothing to learn from: no loss (README §3.1, no empty mean
                # and no invented loss), or, in a model with a frozen part,
                # one that reaches no trainable parameter (research comparison
                # Stage 9: a gene step without a task term, whose LP term
                # reads only a frozen graph side). No backward, no update,
                # counted, its terms logged as given. A model without a frozen
                # part has every term reach a trainable parameter, so there a
                # loss without a gradient is a fault: it fails at its
                # backward below, never skipped.
                skipped_steps += 1
                self.callbacks.on_batch_end(self, batch_idx, {
                    "batch_skipped": 1, **{f"loss_{k}": v for k, v in step.terms.items()},
                    **{f"count_{k}": v for k, v in step.counts.items()}})
                continue
            loss = step.loss / accumulation
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_sum += loss.item() * accumulation
            batch_count += 1
            pending += 1
            if pending == accumulation:
                self._optimizer_update()
                pending = 0
            self.callbacks.on_batch_end(self, batch_idx, {
                "batch_loss": loss.item() * accumulation,
                **{f"loss_{k}": v for k, v in step.terms.items()},
                **{f"count_{k}": v for k, v in step.counts.items()}})

        # The accumulation tail: one more update, as the mean of its batches.
        if pending:
            factor = accumulation / pending
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(factor)
            self._optimizer_update()

        # Epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_loss = loss_sum / max(batch_count, 1)

        self.state.train_loss_history.append(avg_loss)

        return {
            "train_loss": avg_loss,
            "epoch_time": epoch_time,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "train_steps_skipped": skipped_steps,
        }

    def _optimizer_update(self) -> None:
        """Clip, step the optimizer (through the scaler where AMP uses one),
        step the scheduler if the optimizer ran, zero the gradients and count.

        `global_step` counts the attempts; `optimizer_updates` and
        `scheduler_steps` count what ran (contract M3a-1).
        """
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
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # GradScaler skips optimizer.step() on NaN/Inf gradients,
            # indicated by scale decreasing. Only step the scheduler
            # when the optimizer actually ran.
            optimizer_was_run = self.scaler.get_scale() >= scale_before
        else:
            self.optimizer.step()
            optimizer_was_run = True

        if optimizer_was_run:
            self.state.optimizer_updates += 1

        # Scheduler step (only if optimizer actually updated weights)
        if self.scheduler is not None and optimizer_was_run:
            self.scheduler.step()
            self.state.scheduler_steps += 1

        # Zero gradients
        self.optimizer.zero_grad()

        # Update state
        self.state.global_step += 1

    def _validate(self, epoch: int) -> Dict[str, float]:
        """驗證模型

        The task validates (`task.validate`); what stays here is the
        validation lifecycle every stage shares: the callbacks, the `val_`
        prefix, and the best-metric and history bookkeeping.

        `model.eval()` comes **before** `on_validation_begin`: a callback
        receives the trainer and may read `model.training`, so the order is
        observable.
        """
        self.model.eval()
        self.callbacks.on_validation_begin(self)

        # Numbers, and None for a metric the task found undefined (an AUC
        # without one class, README F4): stated in the logs, not dropped.
        val_metrics = {f"val_{k}": v for k, v in self.task.validate(self.model, self.device).items()
                       if v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))}

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

    # =========================================================================
    # Checkpoint Operations
    # =========================================================================
    def _serialize_config(self) -> Dict[str, Any]:
        """The trainer's configuration as plain data, for the checkpoint's
        `config` (every `TrainerConfig` field is a plain value). The model's
        architecture is not here: it is the model record's spec."""
        return dict(self.config.__dict__)

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        data_fingerprint: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write the trainer's checkpoint to `filepath`; return its SHA-256.

        **The same payload `ModelCheckpoint` writes** (contract M3a-1), from
        `src.training.checkpoint_payload.build_checkpoint_payload`, written whole
        by `src.utils.checkpoint_io.write_checkpoint`. `data_fingerprint`, when
        given, replaces the one the training script attached.
        """
        from src.training.checkpoint_payload import build_checkpoint_payload
        from src.utils.checkpoint_io import write_checkpoint
        from src.utils.run_environment import assert_primitive

        checkpoint = build_checkpoint_payload(self, epoch=self.state.epoch, logs={})
        if data_fingerprint is not None:
            assert_primitive(data_fingerprint, "data_fingerprint")
            checkpoint["data_fingerprint"] = data_fingerprint

        digest = write_checkpoint(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath} (sha256 {digest[:12]})")
        return digest

    def continue_from(
        self,
        checkpoint: Mapping[str, Any],
        plan: Any,
        best_weights: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Continue a parent's training from a checkpoint already read, as
        `plan` (`src.utils.run_plan.ContinuationPlan`) decided.

        **Reads nothing** (contract M2.1, S8): the entry point read the parent
        once and records that read's identity. The Trainer was built with the
        plan's settings, the parent's learning settings, so the schedule it
        built is the parent's; each part is restored as the mode allows
        (contract M3a, "room for a branch"):

        - **resume** restores the active stage. The original schedule is
          rebuilt from the settings and must equal the one the parent recorded
          before its scheduler's progress is loaded: never a curve built from
          other settings with the parent's progress. A paused extension keeps
          its fixed rates.
        - **extend** keeps the model, the optimizer and the progress, and opens
          a stage at each group's rate after the restore (M3a-4).

        The model, the optimizer, the scaler and the progress counters are
        restored in both; the callbacks' state is applied after
        `on_train_begin` (`train`), which would otherwise reset it.

        `best_weights` is early stopping's best weights when the plan names a
        file the entry point read and checked; for `{"kind": "self"}` they are
        the parent's own.

        Raises:
            RunPlanRefused: the parent's record cannot rebuild its schedule, or
                its scaler state does not fit this run.
        """
        from src.utils.run_plan import RunPlanRefused

        stage = dict(plan.stage)
        recorded_schedule = (checkpoint.get("progress") or {}).get("stage", {}).get("schedule")
        if plan.mode == "resume" and stage.get("kind", "original") == "original":
            if recorded_schedule != self.schedule_record:
                differ = sorted(
                    k for k in set(recorded_schedule or {}) | set(self.schedule_record)
                    if (recorded_schedule or {}).get(k) != self.schedule_record.get(k))
                raise RunPlanRefused(
                    "The parent's schedule cannot be rebuilt from its record: rebuilt here, "
                    f"{', '.join(differ) or 'it'} differ"
                    f"{'s' if len(differ) == 1 else ''} from what the parent recorded. A run "
                    "is never resumed on a curve built from other settings.")

        has_scaler = "scaler_state_dict" in checkpoint
        if has_scaler != (self.scaler is not None):
            raise RunPlanRefused(
                "The parent's mixed-precision scaler state does not fit this run "
                f"(the parent {'has' if has_scaler else 'has no'} scaler state; this run "
                f"{'uses' if self.scaler is not None else 'does not use'} a scaler).")

        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if stage.get("kind", "original") == "original":
            if self.scheduler is not None:
                if "scheduler_state_dict" not in checkpoint:
                    raise RunPlanRefused("The parent records a schedule but no scheduler state.")
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            # A fixed-rate stage steps no scheduler (contract M3a-4). A new
            # extension sets each group's rate as the plan fixed it (the
            # restored rate, or `extend_lr` for a rate that ended at 0); a
            # paused one must hold the rates its stage recorded.
            self.scheduler = None
            fixed = [float(r) for r in stage["schedule"]["lrs"]]
            if len(fixed) != len(self.optimizer.param_groups):
                raise RunPlanRefused(
                    f"The extension fixes {len(fixed)} rates for "
                    f"{len(self.optimizer.param_groups)} parameter groups.")
            if plan.mode == "extend":
                for group, rate in zip(self.optimizer.param_groups, fixed):
                    group["lr"] = rate
            else:
                rates = [float(group["lr"]) for group in self.optimizer.param_groups]
                if rates != fixed:
                    raise RunPlanRefused(
                        f"The parent's extension records fixed rates {fixed}, and its "
                        f"optimizer holds {rates}.")

        state = checkpoint.get("state") or {}
        for key in ("epoch", "global_step", "best_metric", "best_epoch",
                    "train_loss_history", "val_metric_history", "completed_epochs",
                    "optimizer_updates", "scheduler_steps"):
            if key in state:
                setattr(self.state, key, state[key])

        self.stage = stage
        self.start_epoch = int(plan.start_epoch)
        if best_weights is None and (plan.best_weights or {}).get("kind") == "self":
            best_weights = checkpoint["state_dict"]
        self._pending_callbacks = {
            "recorded": plan.callbacks or {},
            "best_k": list(plan.best_k),
            "best_weights": ({k: v.detach().cpu().clone() for k, v in best_weights.items()}
                             if best_weights is not None else None),
        }
        logger.info(
            f"Continuing ({plan.mode}) from epoch {self.start_epoch} to {self.stage_end()}")

    def _restore_callbacks(self, pending: Mapping[str, Any]) -> None:
        """Early stopping's and best-k's state, as the parent recorded them.

        The parent's best-k files are ranked with this run's but belong to the
        run that wrote them: `ModelCheckpoint` never deletes them.
        """
        from src.training.callbacks import EarlyStopping, ModelCheckpoint

        recorded = pending["recorded"]
        # Early stopping first: best-k keeps its best epoch's file listed.
        for callback in self.callbacks.callbacks:
            if isinstance(callback, EarlyStopping) and recorded.get("early_stopping"):
                callback.restore_state(recorded["early_stopping"], pending["best_weights"])
        for callback in self.callbacks.callbacks:
            if isinstance(callback, ModelCheckpoint):
                callback.restore_foreign(pending["best_k"], self)

# =============================================================================
# Seeding
# =============================================================================
def apply_seed(seed: int) -> None:
    """Seed Python's `random`, NumPy and torch, every CUDA device included.

    The one seeding rule training uses: `scripts/train_model.py` applies it
    before the data loaders and the model are built (contract M3a), and
    `Trainer.__init__` applies it again through `_set_seed`.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Logged only after every RNG above is seeded successfully — an
    # out-of-range seed raises in np.random.seed before reaching here.
    logger.info("Random seed set to %s", seed)


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "TrainerConfig",
    "TrainingState",
    "Trainer",
    "apply_seed",
]
