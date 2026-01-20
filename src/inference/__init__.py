"""
# ==============================================================================
# Module: src/inference/__init__.py
# ==============================================================================
# Purpose: End-to-end inference pipeline for rare disease diagnosis
#
# Dependencies:
#   - External: None (pure Python, torch optional)
#   - Internal: src.core.types, src.kg, src.reasoning
#
# Exports:
#   - DiagnosisPipeline: Main inference pipeline
#   - PipelineConfig: Pipeline configuration
#   - InputValidator: Input validation
#   - ExtensibleInputValidator: Input validator with custom hooks
#   - ValidationResult: Validation result
#   - create_diagnosis_pipeline: Factory function
#   - create_input_validator: Factory function
#
# Usage:
#   from src.inference import DiagnosisPipeline, create_diagnosis_pipeline
#
#   pipeline = create_diagnosis_pipeline(kg=knowledge_graph)
#   result = pipeline.run(patient_phenotypes, top_k=10)
#
# Design Notes:
#   - P0 Core: Phenotype → Gene → Disease reasoning
#   - P1 Feature: Ortholog evidence (interfaces preserved)
#   - Two-stage: Path reasoning + optional GNN scoring
#   - Production-ready: Validation, logging, error handling
#   - Extensible: Custom scorers, validation hooks
# ==============================================================================
"""

from src.inference.pipeline import (
    DiagnosisPipeline,
    PipelineConfig,
    create_diagnosis_pipeline,
)
from src.inference.input_validator import (
    InputValidator,
    ExtensibleInputValidator,
    ValidationResult,
    create_input_validator,
)

__all__ = [
    # Pipeline
    "DiagnosisPipeline",
    "PipelineConfig",
    "create_diagnosis_pipeline",
    # Validation
    "InputValidator",
    "ExtensibleInputValidator",
    "ValidationResult",
    "create_input_validator",
]
