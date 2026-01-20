"""
# ==============================================================================
# Module: src/reasoning/__init__.py
# ==============================================================================
# Purpose: Reasoning module for medical diagnosis inference
#
# Dependencies:
#   - External: numpy
#   - Internal: src.core.types, src.kg
#
# Exports:
#   - PathReasoner: Multi-hop path reasoning
#   - PathReasoningConfig: Configuration for path reasoning
#   - ReasoningPath: Path data structure
#   - DirectPathFinder: Optimized direct path finding
#   - ExplanationGenerator: Human-readable explanation generation
#   - EvidenceSummary: Structured evidence summary
#   - create_path_reasoner: Factory function
#   - create_explanation_generator: Factory function
#
# Usage:
#   from src.reasoning import PathReasoner, ExplanationGenerator
#
#   # Path reasoning
#   reasoner = create_path_reasoner()
#   candidates = reasoner.get_top_candidates(
#       source_ids=patient_phenotypes,
#       kg=knowledge_graph,
#       top_k=10,
#   )
#
#   # Generate explanations
#   generator = ExplanationGenerator()
#   explanation = generator.generate_explanation(candidate, phenotypes, kg)
#
# Design Notes:
#   - Path reasoning: Phenotype → Gene → Disease
#   - Supports pathway-mediated and ortholog paths (P1 feature)
#   - Scoring based on edge weights and path length
#   - Interpretable: Returns paths and human-readable explanations
# ==============================================================================
"""

from src.reasoning.path_reasoning import (
    PathReasoner,
    PathReasoningConfig,
    ReasoningPath,
    DirectPathFinder,
    create_path_reasoner,
)
from src.reasoning.explanation_generator import (
    ExplanationGenerator,
    EvidenceSummary,
    EvidenceItem,
    create_explanation_generator,
)

__all__ = [
    # Path reasoning
    "PathReasoner",
    "PathReasoningConfig",
    "ReasoningPath",
    "DirectPathFinder",
    "create_path_reasoner",
    # Explanation generation
    "ExplanationGenerator",
    "EvidenceSummary",
    "EvidenceItem",
    "create_explanation_generator",
]
