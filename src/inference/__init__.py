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
#   from src.inference import create_diagnosis_pipeline
#
#   pipeline = create_diagnosis_pipeline(
#       graph=KnowledgeGraph.read_json(workspace / "kg.json"),
#       checkpoint_path=str(checkpoint), data_dir=str(workspace))
#   result = pipeline.run(patient_phenotypes, top_k=10)
#
# Design Notes:
#   - Always a trained model read from a checkpoint; no pipeline without one
#   - Path reasoning explains a disease row, never ranks it
#   - P1 Feature: Ortholog evidence (interfaces preserved)
#   - Production-ready: Validation, logging, error handling
# ==============================================================================
"""

from src.inference.pipeline import (
    DiagnosisPipeline,
    PipelineBuildError,
    PipelineConfig,
    create_diagnosis_pipeline,
)

__all__ = [
    # Pipeline
    "DiagnosisPipeline",
    "PipelineBuildError",
    "PipelineConfig",
    "create_diagnosis_pipeline",
]
