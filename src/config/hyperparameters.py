"""
SHEPHERD-Advanced Hyperparameter Configuration
===============================================
Centralized hyperparameter definitions with frontend API support.

Module: src/config/hyperparameters.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/config/hyperparameters.py

Purpose:
    Provide a unified hyperparameter management system that:
    - Defines all tunable parameters for training and inference
    - Exports JSON Schema for frontend validation
    - Supports runtime parameter updates via API
    - Persists configurations to YAML/JSON files
    - Validates parameter ranges and types

Components:
    - HyperparameterSpec: Single parameter specification with metadata
    - TrainingHyperparameters: Training-phase parameters
    - InferenceHyperparameters: Inference-phase parameters
    - ModelHyperparameters: Model architecture parameters
    - HyperparameterManager: Central manager for all parameters

Dependencies:
    - pydantic: Validation and serialization
    - yaml: Configuration file I/O
    - json: JSON Schema export
    - pathlib: File path handling

Input:
    - Configuration files (YAML/JSON)
    - API requests with parameter updates
    - Default values from code

Output:
    - Validated hyperparameter objects
    - JSON Schema for frontend
    - Serialized configuration files

Called by:
    - src/api/ (parameter query and update endpoints)
    - scripts/train_model.py (training configuration)
    - src/inference/pipeline.py (inference configuration)
    - Frontend WebUI (parameter adjustment interface)

Version: 1.0.0
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, get_type_hints

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Parameter Specification
# =============================================================================
class ParameterType(str, Enum):
    """Parameter types for frontend rendering"""
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    SELECT = "select"  # Dropdown with options
    RANGE = "range"    # Slider


@dataclass
class HyperparameterSpec:
    """
    Single hyperparameter specification

    Provides metadata for frontend rendering and validation.
    """
    name: str
    value: Any
    param_type: ParameterType
    description: str
    category: str  # "training", "inference", "model"

    # Constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[Any]] = None  # For SELECT type

    # UI hints
    display_name: Optional[str] = None
    unit: Optional[str] = None
    advanced: bool = False  # Hide in basic mode
    requires_restart: bool = False  # Needs model reload

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "name": self.name,
            "value": self.value,
            "type": self.param_type.value,
            "description": self.description,
            "category": self.category,
            "display_name": self.display_name or self.name.replace("_", " ").title(),
            "advanced": self.advanced,
            "requires_restart": self.requires_restart,
        }

        if self.min_value is not None:
            result["min"] = self.min_value
        if self.max_value is not None:
            result["max"] = self.max_value
        if self.step is not None:
            result["step"] = self.step
        if self.options is not None:
            result["options"] = self.options
        if self.unit is not None:
            result["unit"] = self.unit

        return result

    def validate(self, value: Any) -> bool:
        """Validate a value against constraints"""
        if self.param_type == ParameterType.SELECT:
            return value in (self.options or [])

        if self.param_type in (ParameterType.FLOAT, ParameterType.INT, ParameterType.RANGE):
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False

        return True


# =============================================================================
# Training Hyperparameters
# =============================================================================
@dataclass
class TrainingHyperparameters:
    """
    Training-phase hyperparameters

    All parameters that can be tuned during model training.
    """
    # Learning rate
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    warmup_steps: int = 500
    warmup_ratio: float = 0.1

    # Optimization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Batch and epochs
    batch_size: int = 32
    num_epochs: int = 100
    eval_every_n_epochs: int = 1

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # Learning rate scheduling
    scheduler_type: str = "cosine"  # "cosine", "linear", "onecycle", "none"

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16", "bfloat16"

    # Loss weights (multi-task learning)
    diagnosis_loss_weight: float = 1.0
    link_prediction_weight: float = 0.5
    contrastive_loss_weight: float = 0.3
    ortholog_loss_weight: float = 0.2

    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1

    @classmethod
    def get_specs(cls) -> List[HyperparameterSpec]:
        """Get parameter specifications for frontend"""
        return [
            # Learning rate group
            HyperparameterSpec(
                name="learning_rate",
                value=1e-4,
                param_type=ParameterType.FLOAT,
                description="Initial learning rate for optimizer",
                category="training",
                min_value=1e-7,
                max_value=1e-1,
                step=1e-5,
                display_name="Learning Rate",
            ),
            HyperparameterSpec(
                name="min_learning_rate",
                value=1e-6,
                param_type=ParameterType.FLOAT,
                description="Minimum learning rate after decay",
                category="training",
                min_value=0,
                max_value=1e-3,
                advanced=True,
            ),
            HyperparameterSpec(
                name="warmup_steps",
                value=500,
                param_type=ParameterType.INT,
                description="Number of warmup steps",
                category="training",
                min_value=0,
                max_value=10000,
                step=100,
            ),

            # Optimization group
            HyperparameterSpec(
                name="weight_decay",
                value=0.01,
                param_type=ParameterType.FLOAT,
                description="L2 regularization weight",
                category="training",
                min_value=0,
                max_value=0.5,
                step=0.001,
            ),
            HyperparameterSpec(
                name="max_grad_norm",
                value=1.0,
                param_type=ParameterType.FLOAT,
                description="Maximum gradient norm for clipping",
                category="training",
                min_value=0.1,
                max_value=10.0,
                step=0.1,
            ),
            HyperparameterSpec(
                name="gradient_accumulation_steps",
                value=1,
                param_type=ParameterType.INT,
                description="Accumulate gradients over N steps (effective batch = batch_size * N)",
                category="training",
                min_value=1,
                max_value=64,
                advanced=True,
            ),

            # Batch and epochs
            HyperparameterSpec(
                name="batch_size",
                value=32,
                param_type=ParameterType.INT,
                description="Training batch size",
                category="training",
                min_value=1,
                max_value=512,
                step=8,
            ),
            HyperparameterSpec(
                name="num_epochs",
                value=100,
                param_type=ParameterType.INT,
                description="Maximum training epochs",
                category="training",
                min_value=1,
                max_value=1000,
            ),

            # Early stopping
            HyperparameterSpec(
                name="early_stopping_patience",
                value=10,
                param_type=ParameterType.INT,
                description="Stop if no improvement for N epochs",
                category="training",
                min_value=1,
                max_value=100,
            ),

            # Scheduler
            HyperparameterSpec(
                name="scheduler_type",
                value="cosine",
                param_type=ParameterType.SELECT,
                description="Learning rate scheduler type",
                category="training",
                options=["cosine", "linear", "onecycle", "none"],
            ),

            # Mixed precision
            HyperparameterSpec(
                name="use_amp",
                value=True,
                param_type=ParameterType.BOOL,
                description="Use automatic mixed precision training",
                category="training",
            ),
            HyperparameterSpec(
                name="amp_dtype",
                value="float16",
                param_type=ParameterType.SELECT,
                description="Mixed precision data type",
                category="training",
                options=["float16", "bfloat16"],
                advanced=True,
            ),

            # Loss weights
            HyperparameterSpec(
                name="diagnosis_loss_weight",
                value=1.0,
                param_type=ParameterType.FLOAT,
                description="Weight for diagnosis classification loss",
                category="training",
                min_value=0,
                max_value=10.0,
                step=0.1,
                display_name="Diagnosis Loss Weight",
            ),
            HyperparameterSpec(
                name="link_prediction_weight",
                value=0.5,
                param_type=ParameterType.FLOAT,
                description="Weight for link prediction loss",
                category="training",
                min_value=0,
                max_value=10.0,
                step=0.1,
            ),
            HyperparameterSpec(
                name="contrastive_loss_weight",
                value=0.3,
                param_type=ParameterType.FLOAT,
                description="Weight for contrastive learning loss",
                category="training",
                min_value=0,
                max_value=10.0,
                step=0.1,
            ),
            HyperparameterSpec(
                name="ortholog_loss_weight",
                value=0.2,
                param_type=ParameterType.FLOAT,
                description="Weight for ortholog consistency loss (P1 feature)",
                category="training",
                min_value=0,
                max_value=10.0,
                step=0.1,
                advanced=True,
            ),

            # Regularization
            HyperparameterSpec(
                name="dropout",
                value=0.1,
                param_type=ParameterType.FLOAT,
                description="Dropout rate for regularization",
                category="training",
                min_value=0,
                max_value=0.9,
                step=0.05,
            ),
            HyperparameterSpec(
                name="label_smoothing",
                value=0.1,
                param_type=ParameterType.FLOAT,
                description="Label smoothing factor",
                category="training",
                min_value=0,
                max_value=0.5,
                step=0.05,
                advanced=True,
            ),
        ]


# =============================================================================
# Inference Hyperparameters
# =============================================================================
@dataclass
class InferenceHyperparameters:
    """
    Inference-phase hyperparameters

    Parameters that affect prediction behavior at runtime.
    """
    # Ranking
    top_k: int = 10
    min_confidence_threshold: float = 0.1
    use_reasoning_paths: bool = True

    # Path reasoning
    max_path_length: int = 4
    max_paths_per_target: int = 5
    path_score_weight: float = 0.3

    # Evidence aggregation
    evidence_weight: float = 0.2
    min_evidence_count: int = 1

    # Ortholog (P1)
    use_ortholog_evidence: bool = True
    ortholog_confidence_threshold: float = 0.5
    ortholog_species: List[str] = field(default_factory=lambda: ["mouse", "zebrafish"])

    # Explanation generation
    include_explanations: bool = True
    max_explanation_paths: int = 3
    explanation_detail_level: str = "standard"  # "minimal", "standard", "detailed"

    # Performance
    use_caching: bool = True
    cache_ttl_seconds: int = 3600

    @classmethod
    def get_specs(cls) -> List[HyperparameterSpec]:
        """Get parameter specifications for frontend"""
        return [
            # Ranking
            HyperparameterSpec(
                name="top_k",
                value=10,
                param_type=ParameterType.INT,
                description="Number of top diagnoses to return",
                category="inference",
                min_value=1,
                max_value=100,
                display_name="Top K Results",
            ),
            HyperparameterSpec(
                name="min_confidence_threshold",
                value=0.1,
                param_type=ParameterType.FLOAT,
                description="Minimum confidence score to include in results",
                category="inference",
                min_value=0,
                max_value=1.0,
                step=0.05,
            ),
            HyperparameterSpec(
                name="use_reasoning_paths",
                value=True,
                param_type=ParameterType.BOOL,
                description="Include reasoning path analysis",
                category="inference",
            ),

            # Path reasoning
            HyperparameterSpec(
                name="max_path_length",
                value=4,
                param_type=ParameterType.INT,
                description="Maximum hops in reasoning paths (Phenotype→Gene→Disease)",
                category="inference",
                min_value=2,
                max_value=6,
                advanced=True,
            ),
            HyperparameterSpec(
                name="max_paths_per_target",
                value=5,
                param_type=ParameterType.INT,
                description="Maximum paths to consider per disease candidate",
                category="inference",
                min_value=1,
                max_value=20,
                advanced=True,
            ),
            HyperparameterSpec(
                name="path_score_weight",
                value=0.3,
                param_type=ParameterType.FLOAT,
                description="Weight of path reasoning score in final ranking",
                category="inference",
                min_value=0,
                max_value=1.0,
                step=0.1,
            ),

            # Evidence
            HyperparameterSpec(
                name="evidence_weight",
                value=0.2,
                param_type=ParameterType.FLOAT,
                description="Weight of literature evidence in scoring",
                category="inference",
                min_value=0,
                max_value=1.0,
                step=0.1,
            ),

            # Ortholog (P1)
            HyperparameterSpec(
                name="use_ortholog_evidence",
                value=True,
                param_type=ParameterType.BOOL,
                description="Include ortholog (cross-species) evidence",
                category="inference",
                display_name="Use Ortholog Evidence (P1)",
            ),
            HyperparameterSpec(
                name="ortholog_confidence_threshold",
                value=0.5,
                param_type=ParameterType.FLOAT,
                description="Minimum confidence for ortholog matches",
                category="inference",
                min_value=0,
                max_value=1.0,
                step=0.1,
                advanced=True,
            ),

            # Explanation
            HyperparameterSpec(
                name="include_explanations",
                value=True,
                param_type=ParameterType.BOOL,
                description="Generate human-readable explanations",
                category="inference",
            ),
            HyperparameterSpec(
                name="explanation_detail_level",
                value="standard",
                param_type=ParameterType.SELECT,
                description="Level of detail in explanations",
                category="inference",
                options=["minimal", "standard", "detailed"],
            ),

            # Performance
            HyperparameterSpec(
                name="use_caching",
                value=True,
                param_type=ParameterType.BOOL,
                description="Cache inference results for repeated queries",
                category="inference",
                advanced=True,
            ),
            HyperparameterSpec(
                name="cache_ttl_seconds",
                value=3600,
                param_type=ParameterType.INT,
                description="Cache time-to-live in seconds",
                category="inference",
                min_value=60,
                max_value=86400,
                unit="seconds",
                advanced=True,
            ),
        ]


# =============================================================================
# Model Hyperparameters
# =============================================================================
@dataclass
class ModelHyperparameters:
    """
    Model architecture hyperparameters

    Parameters that define the neural network structure.
    Changes require model retraining.
    """
    # Architecture
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8

    # GNN type
    conv_type: str = "gat"  # "gat", "hgt", "sage"
    aggregation: str = "mean"  # "mean", "sum", "max"

    # Attention
    attention_dropout: float = 0.1
    use_edge_features: bool = True

    # Ortholog gate (P1)
    use_ortholog_gate: bool = True
    ortholog_gate_hidden: int = 128

    # Positional encoding
    use_positional_encoding: bool = True
    pe_dim: int = 32
    pe_type: str = "laplacian"  # "laplacian", "rwse", "both"

    # Feature encoding
    node_feature_dim: int = 256
    edge_feature_dim: int = 64

    @classmethod
    def get_specs(cls) -> List[HyperparameterSpec]:
        """Get parameter specifications for frontend"""
        return [
            # Architecture
            HyperparameterSpec(
                name="hidden_dim",
                value=256,
                param_type=ParameterType.INT,
                description="Hidden dimension size for all layers",
                category="model",
                min_value=64,
                max_value=1024,
                step=64,
                requires_restart=True,
            ),
            HyperparameterSpec(
                name="num_layers",
                value=4,
                param_type=ParameterType.INT,
                description="Number of GNN message passing layers",
                category="model",
                min_value=1,
                max_value=12,
                requires_restart=True,
            ),
            HyperparameterSpec(
                name="num_heads",
                value=8,
                param_type=ParameterType.INT,
                description="Number of attention heads",
                category="model",
                min_value=1,
                max_value=32,
                requires_restart=True,
            ),

            # GNN type
            HyperparameterSpec(
                name="conv_type",
                value="gat",
                param_type=ParameterType.SELECT,
                description="Graph convolution type",
                category="model",
                options=["gat", "hgt", "sage"],
                requires_restart=True,
                display_name="GNN Type",
            ),
            HyperparameterSpec(
                name="aggregation",
                value="mean",
                param_type=ParameterType.SELECT,
                description="Neighborhood aggregation method",
                category="model",
                options=["mean", "sum", "max"],
                requires_restart=True,
                advanced=True,
            ),

            # Attention
            HyperparameterSpec(
                name="attention_dropout",
                value=0.1,
                param_type=ParameterType.FLOAT,
                description="Dropout rate for attention weights",
                category="model",
                min_value=0,
                max_value=0.9,
                step=0.05,
                requires_restart=True,
            ),
            HyperparameterSpec(
                name="use_edge_features",
                value=True,
                param_type=ParameterType.BOOL,
                description="Include edge type features in message passing",
                category="model",
                requires_restart=True,
                advanced=True,
            ),

            # Ortholog gate
            HyperparameterSpec(
                name="use_ortholog_gate",
                value=True,
                param_type=ParameterType.BOOL,
                description="Enable ortholog gating mechanism (P1 feature)",
                category="model",
                requires_restart=True,
                display_name="Ortholog Gate (P1)",
            ),

            # Positional encoding
            HyperparameterSpec(
                name="use_positional_encoding",
                value=True,
                param_type=ParameterType.BOOL,
                description="Add positional encodings to node features",
                category="model",
                requires_restart=True,
                advanced=True,
            ),
            HyperparameterSpec(
                name="pe_type",
                value="laplacian",
                param_type=ParameterType.SELECT,
                description="Type of positional encoding",
                category="model",
                options=["laplacian", "rwse", "both"],
                requires_restart=True,
                advanced=True,
            ),

            # Feature dimensions
            HyperparameterSpec(
                name="node_feature_dim",
                value=256,
                param_type=ParameterType.INT,
                description="Input node feature dimension",
                category="model",
                min_value=32,
                max_value=1024,
                requires_restart=True,
                advanced=True,
            ),
        ]


# =============================================================================
# Hyperparameter Manager
# =============================================================================
class HyperparameterManager:
    """
    Central manager for all hyperparameters

    Provides:
    - Unified access to all parameters
    - JSON Schema export for frontend
    - Parameter update with validation
    - Configuration persistence
    """

    def __init__(
        self,
        training: Optional[TrainingHyperparameters] = None,
        inference: Optional[InferenceHyperparameters] = None,
        model: Optional[ModelHyperparameters] = None,
    ):
        self.training = training or TrainingHyperparameters()
        self.inference = inference or InferenceHyperparameters()
        self.model = model or ModelHyperparameters()

        self._specs_cache: Optional[Dict[str, HyperparameterSpec]] = None

    # =========================================================================
    # Frontend API Methods
    # =========================================================================
    def get_all_specs(self) -> List[Dict[str, Any]]:
        """
        Get all parameter specifications for frontend

        Returns:
            List of parameter specs as dicts, suitable for JSON response
        """
        specs = []
        specs.extend([s.to_dict() for s in TrainingHyperparameters.get_specs()])
        specs.extend([s.to_dict() for s in InferenceHyperparameters.get_specs()])
        specs.extend([s.to_dict() for s in ModelHyperparameters.get_specs()])
        return specs

    def get_specs_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get parameter specs for a specific category"""
        if category == "training":
            return [s.to_dict() for s in TrainingHyperparameters.get_specs()]
        elif category == "inference":
            return [s.to_dict() for s in InferenceHyperparameters.get_specs()]
        elif category == "model":
            return [s.to_dict() for s in ModelHyperparameters.get_specs()]
        else:
            return []

    def get_current_values(self) -> Dict[str, Any]:
        """
        Get current values of all parameters

        Returns:
            Dict with category -> parameter -> value structure
        """
        return {
            "training": asdict(self.training),
            "inference": asdict(self.inference),
            "model": asdict(self.model),
        }

    def get_json_schema(self) -> Dict[str, Any]:
        """
        Export JSON Schema for frontend form generation

        Returns:
            JSON Schema compatible dict
        """
        specs = self.get_all_specs()

        properties = {}
        for spec in specs:
            prop = {
                "title": spec["display_name"],
                "description": spec["description"],
            }

            if spec["type"] == "float":
                prop["type"] = "number"
                if "min" in spec:
                    prop["minimum"] = spec["min"]
                if "max" in spec:
                    prop["maximum"] = spec["max"]
            elif spec["type"] == "int":
                prop["type"] = "integer"
                if "min" in spec:
                    prop["minimum"] = spec["min"]
                if "max" in spec:
                    prop["maximum"] = spec["max"]
            elif spec["type"] == "bool":
                prop["type"] = "boolean"
            elif spec["type"] == "select":
                prop["type"] = "string"
                prop["enum"] = spec.get("options", [])
            else:
                prop["type"] = "string"

            prop["default"] = spec["value"]
            properties[spec["name"]] = prop

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": properties,
        }

    def update_parameter(
        self,
        name: str,
        value: Any,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update a single parameter

        Args:
            name: Parameter name
            value: New value
            category: Optional category hint

        Returns:
            Dict with success status and any warnings
        """
        result = {"success": False, "warnings": [], "requires_restart": False}

        # Find the parameter
        spec = self._find_spec(name)
        if spec is None:
            result["error"] = f"Unknown parameter: {name}"
            return result

        # Validate
        if not spec.validate(value):
            result["error"] = f"Invalid value for {name}: {value}"
            return result

        # Update the appropriate config
        if spec.category == "training" and hasattr(self.training, name):
            setattr(self.training, name, value)
        elif spec.category == "inference" and hasattr(self.inference, name):
            setattr(self.inference, name, value)
        elif spec.category == "model" and hasattr(self.model, name):
            setattr(self.model, name, value)
        else:
            result["error"] = f"Parameter {name} not found in {spec.category}"
            return result

        result["success"] = True
        result["requires_restart"] = spec.requires_restart

        if spec.requires_restart:
            result["warnings"].append(
                f"Parameter '{name}' requires model restart to take effect"
            )

        logger.info(f"Updated parameter {name} = {value}")
        return result

    def update_parameters(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update multiple parameters at once

        Args:
            updates: Dict of name -> value pairs

        Returns:
            Dict with results for each parameter
        """
        results = {
            "success": True,
            "updated": [],
            "failed": [],
            "requires_restart": False,
        }

        for name, value in updates.items():
            result = self.update_parameter(name, value)
            if result["success"]:
                results["updated"].append(name)
                if result.get("requires_restart"):
                    results["requires_restart"] = True
            else:
                results["failed"].append({"name": name, "error": result.get("error")})
                results["success"] = False

        return results

    def _find_spec(self, name: str) -> Optional[HyperparameterSpec]:
        """Find parameter spec by name"""
        if self._specs_cache is None:
            self._specs_cache = {}
            for spec in TrainingHyperparameters.get_specs():
                self._specs_cache[spec.name] = spec
            for spec in InferenceHyperparameters.get_specs():
                self._specs_cache[spec.name] = spec
            for spec in ModelHyperparameters.get_specs():
                self._specs_cache[spec.name] = spec

        return self._specs_cache.get(name)

    # =========================================================================
    # Persistence
    # =========================================================================
    def save_to_yaml(self, path: Union[str, Path]) -> None:
        """Save current configuration to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "training": asdict(self.training),
            "inference": asdict(self.inference),
            "model": asdict(self.model),
        }

        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {path}")

    def load_from_yaml(self, path: Union[str, Path]) -> None:
        """Load configuration from YAML file"""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            config = yaml.safe_load(f)

        if "training" in config:
            for key, value in config["training"].items():
                if hasattr(self.training, key):
                    setattr(self.training, key, value)

        if "inference" in config:
            for key, value in config["inference"].items():
                if hasattr(self.inference, key):
                    setattr(self.inference, key, value)

        if "model" in config:
            for key, value in config["model"].items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)

        logger.info(f"Configuration loaded from {path}")

    def save_to_json(self, path: Union[str, Path]) -> None:
        """Save current configuration to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "training": asdict(self.training),
            "inference": asdict(self.inference),
            "model": asdict(self.model),
        }

        with open(path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Configuration saved to {path}")

    # =========================================================================
    # Conversion to component configs
    # =========================================================================
    def to_trainer_config(self) -> Dict[str, Any]:
        """Convert to TrainerConfig compatible dict"""
        return {
            "num_epochs": self.training.num_epochs,
            "learning_rate": self.training.learning_rate,
            "weight_decay": self.training.weight_decay,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "max_grad_norm": self.training.max_grad_norm,
            "scheduler_type": self.training.scheduler_type,
            "warmup_steps": self.training.warmup_steps,
            "use_amp": self.training.use_amp,
            "amp_dtype": self.training.amp_dtype,
            "early_stopping_patience": self.training.early_stopping_patience,
        }

    def to_pipeline_config(self) -> Dict[str, Any]:
        """Convert to PipelineConfig compatible dict"""
        return {
            "top_k": self.inference.top_k,
            "min_confidence_threshold": self.inference.min_confidence_threshold,
            "use_reasoning_paths": self.inference.use_reasoning_paths,
            "max_path_length": self.inference.max_path_length,
            "include_explanations": self.inference.include_explanations,
            "use_ortholog_evidence": self.inference.use_ortholog_evidence,
            "ortholog_confidence_threshold": self.inference.ortholog_confidence_threshold,
        }

    def to_model_config(self) -> Dict[str, Any]:
        """Convert to ShepherdGNNConfig compatible dict"""
        return {
            "hidden_dim": self.model.hidden_dim,
            "num_layers": self.model.num_layers,
            "num_heads": self.model.num_heads,
            "conv_type": self.model.conv_type,
            "dropout": self.training.dropout,
            "use_ortholog_gate": self.model.use_ortholog_gate,
        }


# =============================================================================
# Singleton instance for global access
# =============================================================================
_default_manager: Optional[HyperparameterManager] = None


def get_hyperparameter_manager() -> HyperparameterManager:
    """Get the global hyperparameter manager instance"""
    global _default_manager
    if _default_manager is None:
        _default_manager = HyperparameterManager()
    return _default_manager


def reset_hyperparameter_manager() -> None:
    """Reset the global manager (for testing)"""
    global _default_manager
    _default_manager = None


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "ParameterType",
    "HyperparameterSpec",
    "TrainingHyperparameters",
    "InferenceHyperparameters",
    "ModelHyperparameters",
    "HyperparameterManager",
    "get_hyperparameter_manager",
    "reset_hyperparameter_manager",
]
