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
#   - create_diagnosis_pipeline: Factory function
#
#   Input is validated by `DiagnosisPipeline.validate_input`. The separate
#   `InputValidator` had no production caller and was removed
#   (docs/working/PLAN_PHENOTYPE_NORMALISATION.md §5).
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
#   - Extensible: Custom scorers
# ==============================================================================
"""

from src.inference.pipeline import (
    DiagnosisPipeline,
    PipelineConfig,
    create_diagnosis_pipeline,
)

__all__ = [
    # Pipeline
    "DiagnosisPipeline",
    "PipelineConfig",
    "create_diagnosis_pipeline",
]
