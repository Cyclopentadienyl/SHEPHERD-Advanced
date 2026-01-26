"""
SHEPHERD-Advanced Training Module
==================================
Training infrastructure for ShepherdGNN model.

Module: src/training/__init__.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/training/__init__.py

Purpose:
    Provide complete training infrastructure including:
    - Multi-task loss functions for disease diagnosis
    - Callback system for checkpointing and monitoring
    - Trainer class for orchestrating training loops

Components (re-exported):
    From loss_functions:
        - LossConfig: Loss weight configuration
        - DiagnosisLoss: Classification + ranking loss
        - LinkPredictionLoss: KG embedding loss
        - ContrastiveLoss: InfoNCE for discriminative embeddings
        - OrthologConsistencyLoss: Cross-species alignment
        - MultiTaskLoss: Weighted combination of all losses

    From callbacks:
        - Callback: Base callback class
        - CallbackList: Callback manager
        - EarlyStopping, EarlyStoppingConfig: Early stopping callback
        - ModelCheckpoint, ModelCheckpointConfig: Model saving
        - LearningRateMonitor: LR tracking
        - MetricsLogger, MetricsLoggerConfig: Metrics logging
        - ProgressBar: tqdm progress display
        - GradientClipping: Gradient norm monitoring

    From trainer:
        - TrainerConfig: Training hyperparameters
        - TrainingState: Training state tracking
        - Trainer: Main training loop

Dependencies:
    - torch: Training operations
    - Internal: src.training.loss_functions, src.training.callbacks, src.training.trainer

Usage:
    from src.training import Trainer, TrainerConfig, MultiTaskLoss

Version: 1.0.0
"""

# Loss functions
from src.training.loss_functions import (
    LossConfig,
    DiagnosisLoss,
    LinkPredictionLoss,
    ContrastiveLoss,
    OrthologConsistencyLoss,
    MultiTaskLoss,
)

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
    ProgressBar,
    GradientClipping,
)

# Trainer
from src.training.trainer import (
    TrainerConfig,
    TrainingState,
    Trainer,
)


__all__ = [
    # Loss functions
    "LossConfig",
    "DiagnosisLoss",
    "LinkPredictionLoss",
    "ContrastiveLoss",
    "OrthologConsistencyLoss",
    "MultiTaskLoss",
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
    "ProgressBar",
    "GradientClipping",
    # Trainer
    "TrainerConfig",
    "TrainingState",
    "Trainer",
]
