"""
SHEPHERD-Advanced Training Module
==================================
Training infrastructure for model bundles.

Module: src/training/__init__.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/training/__init__.py

Purpose:
    Provide the training infrastructure every stage shares:
    - Callback system for checkpointing and monitoring
    - Trainer class for the training loop a stage's task module drives
    The stages' task modules are `src.training.lp_task` (the KG base) and
    `src.training.finetune_task` (the task finetunes), imported directly.

Components (re-exported):
    From callbacks:
        - Callback: Base callback class
        - CallbackList: Callback manager
        - EarlyStopping, EarlyStoppingConfig: Early stopping callback
        - ModelCheckpoint, ModelCheckpointConfig: Model saving
        - LearningRateMonitor: LR tracking
        - MetricsLogger, MetricsLoggerConfig: Metrics logging

    From trainer:
        - TrainerConfig: Training hyperparameters
        - TrainingState: Training state tracking
        - Trainer: Main training loop

Dependencies:
    - torch: Training operations
    - Internal: src.training.callbacks, src.training.trainer

Usage:
    from src.training import Trainer, TrainerConfig

Version: 1.0.0
"""

# Callbacks
from src.training.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    EarlyStoppingConfig,
    ModelCheckpoint,
    ModelCheckpointConfig,
    LearningRateMonitor,
    MetricsLogger,
    MetricsLoggerConfig,
)

# Trainer
from src.training.trainer import (
    TrainerConfig,
    TrainingState,
    Trainer,
)


__all__ = [
    # Callbacks
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "EarlyStoppingConfig",
    "ModelCheckpoint",
    "ModelCheckpointConfig",
    "LearningRateMonitor",
    "MetricsLogger",
    "MetricsLoggerConfig",
    # Trainer
    "TrainerConfig",
    "TrainingState",
    "Trainer",
]
