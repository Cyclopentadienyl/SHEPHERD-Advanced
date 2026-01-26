"""
SHEPHERD-Advanced Configuration Module
======================================
Centralized configuration management for all system components.

Module: src/config/__init__.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/config/__init__.py

Purpose:
    Provide unified configuration management including:
    - Hyperparameter definitions with frontend API support
    - Configuration validation
    - Schema loading and validation
    - Persistence to YAML/JSON

Components (re-exported):
    From hyperparameters:
        - ParameterType: Enum for parameter types
        - HyperparameterSpec: Single parameter specification
        - TrainingHyperparameters: Training-phase parameters
        - InferenceHyperparameters: Inference-phase parameters
        - ModelHyperparameters: Model architecture parameters
        - HyperparameterManager: Central parameter manager
        - get_hyperparameter_manager: Get global manager instance

Dependencies:
    - pydantic: Validation
    - yaml: Configuration files
    - Internal: src.config.hyperparameters

Usage:
    # Get current hyperparameters
    from src.config import get_hyperparameter_manager
    manager = get_hyperparameter_manager()

    # Get all specs for frontend
    specs = manager.get_all_specs()

    # Update a parameter
    result = manager.update_parameter("learning_rate", 0.001)

    # Save configuration
    manager.save_to_yaml("configs/current.yaml")

Version: 1.0.0
"""

from src.config.hyperparameters import (
    ParameterType,
    HyperparameterSpec,
    TrainingHyperparameters,
    InferenceHyperparameters,
    ModelHyperparameters,
    HyperparameterManager,
    get_hyperparameter_manager,
    reset_hyperparameter_manager,
)


__all__ = [
    # Parameter types
    "ParameterType",
    "HyperparameterSpec",
    # Parameter classes
    "TrainingHyperparameters",
    "InferenceHyperparameters",
    "ModelHyperparameters",
    # Manager
    "HyperparameterManager",
    "get_hyperparameter_manager",
    "reset_hyperparameter_manager",
]
