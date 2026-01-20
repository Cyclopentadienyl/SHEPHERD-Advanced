"""
# ==============================================================================
# Module: src/inference/pipeline.py
# ==============================================================================
# Purpose: End-to-end inference pipeline for rare disease diagnosis
#
# Dependencies:
#   - External: None (pure Python, torch optional for GNN)
#   - Internal: src.core.types, src.kg, src.reasoning
#
# Input:
#   - PatientPhenotypes: Patient's phenotype data (HPO IDs)
#   - KnowledgeGraph: Pre-built knowledge graph
#   - Model (optional): Pre-trained GNN model for scoring
#
# Output:
#   - InferenceResult: Complete diagnosis result with candidates and explanations
#
# Design Notes:
#   - Core functionality (P0): Phenotype -> Gene -> Disease reasoning
#   - Ortholog support (P1): Cross-species evidence (interfaces preserved)
#   - Two-stage scoring: Path reasoning + optional GNN scoring
#   - Interpretable: Full evidence paths and human-readable explanations
#   - Production-ready: Input validation, error handling, logging
#
# Usage:
#   from src.inference import DiagnosisPipeline
#
#   pipeline = DiagnosisPipeline(kg=knowledge_graph)
#   result = pipeline.run(patient_phenotypes, top_k=10)
# ==============================================================================
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from src.core.types import (
    DataSource,
    DiagnosisCandidate,
    EvidenceSource,
    InferenceResult,
    NodeID,
    NodeType,
    PatientPhenotypes,
)
from src.reasoning import (
    PathReasoner,
    PathReasoningConfig,
    ReasoningPath,
    DirectPathFinder,
    ExplanationGenerator,
    create_path_reasoner,
    create_explanation_generator,
)

if TYPE_CHECKING:
    from src.kg import KnowledgeGraph

logger = logging.getLogger(__name__)


# ==============================================================================
# Type Aliases for Callbacks
# ==============================================================================
# Callback for custom scoring: (candidate, paths, kg) -> modified_score
ScoringCallback = Callable[
    ["DiagnosisCandidate", List[ReasoningPath], "KnowledgeGraph"],
    float
]
# Callback for post-processing candidates
PostProcessCallback = Callable[
    [List["DiagnosisCandidate"], PatientPhenotypes],
    List["DiagnosisCandidate"]
]


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class PipelineConfig:
    """Configuration for the diagnosis pipeline."""

    # Path reasoning
    max_path_length: int = 4
    path_length_penalty: float = 0.9
    aggregation_method: str = "weighted_sum"

    # Scoring weights
    reasoning_weight: float = 0.5
    gnn_weight: float = 0.5
    ortholog_weight: float = 0.3  # P1: Weight for ortholog evidence

    # Output control
    include_explanations: bool = True
    include_ortholog_evidence: bool = True
    include_literature_evidence: bool = True
    max_paths_per_candidate: int = 10

    # Validation
    validate_phenotypes: bool = True
    min_phenotypes: int = 1
    max_phenotypes: int = 100

    # Performance
    use_direct_path_optimization: bool = True

    # P1 Ortholog Configuration
    ortholog_species: List[str] = field(
        default_factory=lambda: ["mouse", "zebrafish", "rat"]
    )
    min_ortholog_confidence: float = 0.5

    # Extensibility: Custom scoring callbacks (P1-ready)
    # These will be called during scoring phase
    custom_scorers: List[ScoringCallback] = field(default_factory=list)
    post_process_callbacks: List[PostProcessCallback] = field(default_factory=list)


# ==============================================================================
# Result Types
# ==============================================================================
@dataclass
class ValidationResult:
    """Result of input validation."""

    is_valid: bool
    validated_phenotypes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ==============================================================================
# Diagnosis Pipeline
# ==============================================================================
class DiagnosisPipeline:
    """
    End-to-end diagnosis inference pipeline.

    Combines:
    - Knowledge graph path reasoning (P0 core)
    - Optional GNN scoring (P0 enhancement)
    - Ortholog evidence integration (P1 feature)
    - Human-readable explanations

    Usage:
        pipeline = DiagnosisPipeline(kg=knowledge_graph)
        result = pipeline.run(patient_phenotypes, top_k=10)
    """

    # Version info
    VERSION = "1.0.0"

    def __init__(
        self,
        kg: "KnowledgeGraph",
        config: Optional[PipelineConfig] = None,
        model: Optional[Any] = None,  # Optional GNN model
    ):
        """
        Initialize the diagnosis pipeline.

        Args:
            kg: Knowledge graph for reasoning
            config: Pipeline configuration
            model: Optional pre-trained GNN model for enhanced scoring
        """
        self.kg = kg
        self.config = config or PipelineConfig()
        self.model = model

        # Initialize components
        self._init_reasoning_components()
        self._init_explanation_generator()

        logger.info(
            f"DiagnosisPipeline initialized: "
            f"version={self.VERSION}, "
            f"kg_nodes={len(kg._nodes)}, "
            f"has_model={model is not None}"
        )

    def _init_reasoning_components(self) -> None:
        """Initialize path reasoning components."""
        reasoning_config = PathReasoningConfig(
            max_path_length=self.config.max_path_length,
            length_penalty=self.config.path_length_penalty,
            aggregation_method=self.config.aggregation_method,
        )
        self.path_reasoner = create_path_reasoner(config=reasoning_config)

        if self.config.use_direct_path_optimization:
            self.direct_finder = DirectPathFinder(self.kg)
        else:
            self.direct_finder = None

    def _init_explanation_generator(self) -> None:
        """Initialize explanation generator."""
        self.explanation_generator = create_explanation_generator(
            include_ortholog_evidence=self.config.include_ortholog_evidence,
            include_literature_evidence=self.config.include_literature_evidence,
        )

    def run(
        self,
        patient_input: PatientPhenotypes,
        top_k: int = 10,
        include_explanations: Optional[bool] = None,
        include_ortholog_evidence: Optional[bool] = None,
        include_literature_evidence: Optional[bool] = None,
    ) -> InferenceResult:
        """
        Execute the full diagnosis pipeline.

        Args:
            patient_input: Patient phenotype data
            top_k: Number of top candidates to return
            include_explanations: Override config for explanation generation
            include_ortholog_evidence: Override config for ortholog evidence
            include_literature_evidence: Override config for literature evidence

        Returns:
            InferenceResult with diagnosis candidates and explanations
        """
        start_time = time.time()
        warnings: List[str] = []

        # Use config defaults if not overridden
        if include_explanations is None:
            include_explanations = self.config.include_explanations
        if include_ortholog_evidence is None:
            include_ortholog_evidence = self.config.include_ortholog_evidence
        if include_literature_evidence is None:
            include_literature_evidence = self.config.include_literature_evidence

        logger.info(
            f"Starting diagnosis for patient {patient_input.patient_id} "
            f"with {len(patient_input.phenotypes)} phenotypes"
        )

        # Step 1: Validate input
        validation = self.validate_input(patient_input)
        if not validation.is_valid:
            logger.error(f"Input validation failed: {validation.errors}")
            return InferenceResult(
                patient_id=patient_input.patient_id,
                timestamp=datetime.now(),
                candidates=[],
                warnings=validation.errors,
                model_version=self.VERSION,
                inference_time_ms=(time.time() - start_time) * 1000,
            )
        warnings.extend(validation.warnings)

        # Step 2: Convert phenotypes to NodeIDs
        source_ids = self._phenotypes_to_node_ids(validation.validated_phenotypes)
        if not source_ids:
            logger.warning("No valid phenotype NodeIDs found")
            return self._create_empty_result(
                patient_input.patient_id,
                start_time,
                warnings + ["No valid phenotypes found in knowledge graph"],
            )

        # Step 3: Find reasoning paths
        logger.debug("Finding reasoning paths...")
        all_paths = self._find_all_paths(source_ids, include_ortholog_evidence)
        if not all_paths:
            logger.warning("No paths found from phenotypes to diseases")
            return self._create_empty_result(
                patient_input.patient_id,
                start_time,
                warnings + ["No paths found from phenotypes to diseases"],
            )

        # Step 4: Score and rank candidates
        logger.debug("Scoring candidates...")
        candidates = self._score_and_rank_candidates(
            all_paths=all_paths,
            source_ids=source_ids,
            patient_input=patient_input,
            top_k=top_k,
            include_ortholog_evidence=include_ortholog_evidence,
        )

        # Step 5: Generate explanations
        if include_explanations:
            logger.debug("Generating explanations...")
            candidates = self._add_explanations(
                candidates=candidates,
                all_paths=all_paths,
                patient_input=patient_input,
            )

        # Step 6: Create summary explanation
        summary = None
        if include_explanations and candidates:
            summary = self._generate_summary_explanation(
                candidates[:min(3, len(candidates))],
                patient_input,
            )

        # Build result
        inference_time = (time.time() - start_time) * 1000
        logger.info(
            f"Diagnosis complete: {len(candidates)} candidates "
            f"in {inference_time:.1f}ms"
        )

        return InferenceResult(
            patient_id=patient_input.patient_id,
            timestamp=datetime.now(),
            candidates=candidates,
            summary_explanation=summary,
            model_version=self.VERSION,
            kg_version=getattr(self.kg, "version", "unknown"),
            inference_time_ms=inference_time,
            warnings=warnings,
        )

    def validate_input(
        self,
        patient_input: PatientPhenotypes,
    ) -> ValidationResult:
        """
        Validate patient input.

        Args:
            patient_input: Patient phenotype data

        Returns:
            ValidationResult with validated phenotypes and any warnings/errors
        """
        errors: List[str] = []
        warnings: List[str] = []
        validated_phenotypes: List[str] = []

        # Check patient_id
        if not patient_input.patient_id:
            errors.append("patient_id is required")

        # Check phenotype count
        if len(patient_input.phenotypes) < self.config.min_phenotypes:
            errors.append(
                f"At least {self.config.min_phenotypes} phenotype(s) required, "
                f"got {len(patient_input.phenotypes)}"
            )

        if len(patient_input.phenotypes) > self.config.max_phenotypes:
            warnings.append(
                f"More than {self.config.max_phenotypes} phenotypes provided, "
                f"using first {self.config.max_phenotypes}"
            )

        # Validate each phenotype
        for pheno_id in patient_input.phenotypes[:self.config.max_phenotypes]:
            if self._validate_phenotype(pheno_id):
                validated_phenotypes.append(pheno_id)
            else:
                warnings.append(f"Unknown phenotype: {pheno_id}")

        if not validated_phenotypes and not errors:
            errors.append("No valid phenotypes after validation")

        return ValidationResult(
            is_valid=len(errors) == 0,
            validated_phenotypes=validated_phenotypes,
            warnings=warnings,
            errors=errors,
        )

    def _validate_phenotype(self, pheno_id: str) -> bool:
        """Check if a phenotype ID is valid in the KG."""
        if not self.config.validate_phenotypes:
            return True

        # Check if phenotype exists in KG
        node_id = NodeID(source=DataSource.HPO, local_id=pheno_id)
        return self.kg.has_node(node_id)

    def _phenotypes_to_node_ids(self, phenotypes: List[str]) -> List[NodeID]:
        """Convert HPO ID strings to NodeID objects."""
        node_ids = []
        for pheno_id in phenotypes:
            node_id = NodeID(source=DataSource.HPO, local_id=pheno_id)
            if self.kg.has_node(node_id):
                node_ids.append(node_id)
        return node_ids

    def _find_all_paths(
        self,
        source_ids: List[NodeID],
        include_ortholog: bool = True,
    ) -> Dict[str, List[ReasoningPath]]:
        """
        Find all reasoning paths from phenotypes to diseases.

        Args:
            source_ids: Phenotype NodeIDs
            include_ortholog: Include ortholog paths (P1 feature)

        Returns:
            Dictionary mapping disease ID strings to their supporting paths
        """
        all_paths: Dict[str, List[ReasoningPath]] = {}

        # Use direct path finder for optimization if available
        if self.direct_finder and self.config.use_direct_path_optimization:
            direct_paths = self.direct_finder.find_phenotype_gene_disease_paths(
                source_ids
            )
            for path in direct_paths:
                disease_key = str(path.target)
                if disease_key not in all_paths:
                    all_paths[disease_key] = []
                all_paths[disease_key].append(path)

        # Find general paths via BFS
        general_paths = self.path_reasoner.find_paths(
            source_ids=source_ids,
            target_type=NodeType.DISEASE,
            kg=self.kg,
            max_length=self.config.max_path_length,
        )

        # Score paths
        scored_paths = self.path_reasoner.score_paths(general_paths, self.kg)

        # Group by disease
        for path in scored_paths:
            disease_key = str(path.target)
            if disease_key not in all_paths:
                all_paths[disease_key] = []

            # Avoid duplicates
            existing_path_strs = {
                str([str(n) for n in p.nodes]) for p in all_paths[disease_key]
            }
            path_str = str([str(n) for n in path.nodes])
            if path_str not in existing_path_strs:
                all_paths[disease_key].append(path)

        # Sort paths within each disease by score
        for disease_key in all_paths:
            all_paths[disease_key].sort(key=lambda p: p.score, reverse=True)
            # Keep only top paths per candidate
            all_paths[disease_key] = all_paths[disease_key][
                :self.config.max_paths_per_candidate
            ]

        return all_paths

    def _score_and_rank_candidates(
        self,
        all_paths: Dict[str, List[ReasoningPath]],
        source_ids: List[NodeID],
        patient_input: PatientPhenotypes,
        top_k: int,
        include_ortholog_evidence: bool,
    ) -> List[DiagnosisCandidate]:
        """
        Score and rank disease candidates.

        Args:
            all_paths: Paths grouped by disease
            source_ids: Patient phenotype NodeIDs
            patient_input: Patient phenotype data
            top_k: Number of top candidates to return
            include_ortholog_evidence: Include ortholog evidence

        Returns:
            List of DiagnosisCandidate sorted by score
        """
        candidates: List[DiagnosisCandidate] = []

        for disease_key, paths in all_paths.items():
            if not paths:
                continue

            # Get disease info
            disease_id = paths[0].target
            disease_node = self.kg.get_node(disease_id)
            disease_name = disease_node.name if disease_node else str(disease_id)

            # Calculate reasoning score from paths
            reasoning_score = self._calculate_reasoning_score(paths)

            # Calculate GNN score if model available
            gnn_score = 0.0
            if self.model is not None:
                gnn_score = self._calculate_gnn_score(
                    source_ids, disease_id, patient_input
                )

            # Combined confidence score
            if self.model is not None:
                confidence_score = (
                    self.config.reasoning_weight * reasoning_score
                    + self.config.gnn_weight * gnn_score
                )
            else:
                confidence_score = reasoning_score

            # Extract supporting genes
            supporting_genes = self._extract_supporting_genes(paths)

            # Extract reasoning path nodes
            reasoning_paths = [list(p.nodes) for p in paths[:5]]

            # Determine evidence sources
            evidence_sources = self._determine_evidence_sources(paths)

            # Create candidate
            candidate = DiagnosisCandidate(
                rank=0,  # Will be set after sorting
                disease_id=disease_id,
                disease_name=disease_name,
                confidence_score=confidence_score,
                gnn_score=gnn_score,
                reasoning_score=reasoning_score,
                supporting_genes=supporting_genes,
                reasoning_paths=reasoning_paths,
                evidence_sources=evidence_sources,
            )
            candidates.append(candidate)

        # Sort by confidence score
        candidates.sort(key=lambda c: c.confidence_score, reverse=True)

        # Assign ranks and limit to top_k
        for i, candidate in enumerate(candidates[:top_k]):
            candidate.rank = i + 1

        return candidates[:top_k]

    def _calculate_reasoning_score(self, paths: List[ReasoningPath]) -> float:
        """Calculate aggregate reasoning score from paths."""
        if not paths:
            return 0.0

        # Use configured aggregation method
        method = self.config.aggregation_method
        scores = [p.score for p in paths]

        if method == "max":
            return max(scores)
        elif method == "mean":
            return sum(scores) / len(scores)
        elif method == "sum":
            return min(sum(scores), 1.0)  # Cap at 1.0
        elif method == "weighted_sum":
            # Weighted by path count with diminishing returns
            weighted = sum(s * (0.8 ** i) for i, s in enumerate(scores))
            return min(weighted, 1.0)
        else:
            return max(scores)

    def _calculate_gnn_score(
        self,
        source_ids: List[NodeID],
        disease_id: NodeID,
        patient_input: PatientPhenotypes,
    ) -> float:
        """
        Calculate GNN-based score for a disease candidate.

        This is a placeholder for future GNN integration.
        Currently returns 0.0 if model is not available.
        """
        # TODO: Implement actual GNN scoring when model is integrated
        # This would involve:
        # 1. Convert patient phenotypes to node embeddings
        # 2. Run GNN message passing
        # 3. Compute phenotype-disease similarity
        return 0.0

    def _extract_supporting_genes(
        self, paths: List[ReasoningPath]
    ) -> List[str]:
        """Extract unique supporting genes from paths."""
        genes = set()
        for path in paths:
            for node_id in path.nodes:
                node = self.kg.get_node(node_id)
                if node and node.node_type == NodeType.GENE:
                    genes.add(node.name)
        return sorted(list(genes))

    def _determine_evidence_sources(
        self, paths: List[ReasoningPath]
    ) -> List[EvidenceSource]:
        """Determine evidence sources from paths."""
        sources: List[EvidenceSource] = []
        seen_sources: set = set()

        for path in paths:
            for node_id in path.nodes:
                source_key = f"{node_id.source.value}:{node_id.local_id}"
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    sources.append(
                        EvidenceSource(
                            source_type=node_id.source,
                            source_id=node_id.local_id,
                        )
                    )

        # Limit to top sources to avoid overwhelming results
        return sources[:10]

    def _add_explanations(
        self,
        candidates: List[DiagnosisCandidate],
        all_paths: Dict[str, List[ReasoningPath]],
        patient_input: PatientPhenotypes,
    ) -> List[DiagnosisCandidate]:
        """Add explanations to candidates."""
        for candidate in candidates:
            disease_key = str(candidate.disease_id)
            paths = all_paths.get(disease_key, [])

            explanation = self.explanation_generator.generate_explanation(
                candidate=candidate,
                phenotypes=patient_input,
                kg=self.kg,
                paths=paths,
            )
            candidate.explanation = explanation

        return candidates

    def _generate_summary_explanation(
        self,
        top_candidates: List[DiagnosisCandidate],
        patient_input: PatientPhenotypes,
    ) -> str:
        """Generate overall summary explanation."""
        if not top_candidates:
            return "No diagnosis candidates found."

        lines = [
            f"## Diagnosis Summary for Patient {patient_input.patient_id}",
            "",
            f"Based on {len(patient_input.phenotypes)} input phenotypes, "
            f"the following top candidates were identified:",
            "",
        ]

        for candidate in top_candidates:
            confidence_level = (
                "HIGH" if candidate.confidence_score >= 0.7
                else "MEDIUM" if candidate.confidence_score >= 0.4
                else "LOW"
            )
            lines.append(
                f"**{candidate.rank}. {candidate.disease_name}** "
                f"(Confidence: {confidence_level}, Score: {candidate.confidence_score:.3f})"
            )
            if candidate.supporting_genes:
                lines.append(
                    f"   Key genes: {', '.join(candidate.supporting_genes[:3])}"
                )
            lines.append("")

        lines.extend([
            "---",
            "*Note: These results are computational predictions for clinical review. "
            "Final diagnosis should be made by qualified healthcare professionals.*",
        ])

        return "\n".join(lines)

    def _create_empty_result(
        self,
        patient_id: str,
        start_time: float,
        warnings: List[str],
    ) -> InferenceResult:
        """Create an empty result with warnings."""
        return InferenceResult(
            patient_id=patient_id,
            timestamp=datetime.now(),
            candidates=[],
            warnings=warnings,
            model_version=self.VERSION,
            inference_time_ms=(time.time() - start_time) * 1000,
        )

    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration as dictionary."""
        return {
            "version": self.VERSION,
            "max_path_length": self.config.max_path_length,
            "path_length_penalty": self.config.path_length_penalty,
            "aggregation_method": self.config.aggregation_method,
            "reasoning_weight": self.config.reasoning_weight,
            "gnn_weight": self.config.gnn_weight,
            "include_explanations": self.config.include_explanations,
            "include_ortholog_evidence": self.config.include_ortholog_evidence,
            "include_literature_evidence": self.config.include_literature_evidence,
            "has_model": self.model is not None,
        }


# ==============================================================================
# Factory Function
# ==============================================================================
def create_diagnosis_pipeline(
    kg: "KnowledgeGraph",
    config: Optional[PipelineConfig] = None,
    model: Optional[Any] = None,
) -> DiagnosisPipeline:
    """
    Factory function to create a diagnosis pipeline.

    Args:
        kg: Knowledge graph
        config: Pipeline configuration
        model: Optional GNN model

    Returns:
        Configured DiagnosisPipeline instance
    """
    return DiagnosisPipeline(kg=kg, config=config, model=model)
