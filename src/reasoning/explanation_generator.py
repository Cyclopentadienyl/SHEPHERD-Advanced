"""
# ==============================================================================
# Module: src/reasoning/explanation_generator.py
# ==============================================================================
# Purpose: Generate human-readable explanations for diagnosis results
#
# Dependencies:
#   - External: None (pure Python)
#   - Internal: src.core.types, src.kg, src.reasoning.path_reasoning
#
# Input:
#   - DiagnosisCandidate with paths and evidence
#   - Patient phenotypes
#   - Knowledge Graph for context lookup
#
# Output:
#   - Human-readable explanation string
#   - Structured evidence summary
#
# Design Notes:
#   - Focuses on interpretability for clinical use
#   - Separates direct evidence from inferred evidence
#   - Supports ortholog evidence chains (P1 feature)
#   - Multilingual-ready (currently English)
#
# Key Principles:
#   1. Transparency: Show WHY a diagnosis is suggested
#   2. Evidence-based: Cite specific paths and data sources
#   3. Uncertainty quantification: Express confidence levels
#   4. Actionable: Highlight key phenotype-gene-disease connections
# ==============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from src.core.types import (
    DataSource,
    DiagnosisCandidate,
    EdgeType,
    EvidenceSource,
    Node,
    NodeID,
    NodeType,
    PatientPhenotypes,
)
from src.reasoning.path_reasoning import ReasoningPath

if TYPE_CHECKING:
    from src.kg.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


# ==============================================================================
# Evidence Data Structures
# ==============================================================================
@dataclass
class EvidenceItem:
    """A single piece of evidence supporting a diagnosis."""

    evidence_type: str  # "direct", "pathway", "ortholog", "literature"
    description: str
    confidence: float
    source_nodes: List[NodeID] = field(default_factory=list)
    supporting_path: Optional[ReasoningPath] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.evidence_type,
            "description": self.description,
            "confidence": self.confidence,
            "sources": [str(n) for n in self.source_nodes],
        }


@dataclass
class EvidenceSummary:
    """Structured evidence summary for a diagnosis candidate."""

    disease_id: NodeID
    disease_name: str
    total_score: float
    confidence_level: str  # "high", "medium", "low"

    direct_evidence: List[EvidenceItem] = field(default_factory=list)
    pathway_evidence: List[EvidenceItem] = field(default_factory=list)
    ortholog_evidence: List[EvidenceItem] = field(default_factory=list)
    literature_evidence: List[EvidenceItem] = field(default_factory=list)

    matching_phenotypes: List[str] = field(default_factory=list)
    key_genes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "disease_id": str(self.disease_id),
            "disease_name": self.disease_name,
            "total_score": self.total_score,
            "confidence_level": self.confidence_level,
            "matching_phenotypes": self.matching_phenotypes,
            "key_genes": self.key_genes,
            "evidence": {
                "direct": [e.to_dict() for e in self.direct_evidence],
                "pathway": [e.to_dict() for e in self.pathway_evidence],
                "ortholog": [e.to_dict() for e in self.ortholog_evidence],
                "literature": [e.to_dict() for e in self.literature_evidence],
            },
        }


# ==============================================================================
# Explanation Generator
# ==============================================================================
class ExplanationGenerator:
    """
    Generate human-readable explanations for diagnosis results.

    Provides:
    - Narrative explanation for clinical review
    - Structured evidence summary for programmatic use
    - Confidence assessment based on evidence strength

    Usage:
        generator = ExplanationGenerator()
        explanation = generator.generate_explanation(
            candidate=diagnosis_candidate,
            phenotypes=patient_phenotypes,
            kg=knowledge_graph,
        )
        summary = generator.generate_evidence_summary(candidate, kg)
    """

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    MEDIUM_CONFIDENCE_THRESHOLD = 0.4

    def __init__(
        self,
        include_ortholog_evidence: bool = True,
        include_literature_evidence: bool = True,
        max_paths_per_type: int = 5,
    ):
        """
        Args:
            include_ortholog_evidence: Include cross-species evidence (P1 feature)
            include_literature_evidence: Include PubMed citations
            max_paths_per_type: Maximum paths to include per evidence type
        """
        self.include_ortholog_evidence = include_ortholog_evidence
        self.include_literature_evidence = include_literature_evidence
        self.max_paths_per_type = max_paths_per_type

    def generate_explanation(
        self,
        candidate: DiagnosisCandidate,
        phenotypes: PatientPhenotypes,
        kg: "KnowledgeGraph",
        paths: Optional[List[ReasoningPath]] = None,
    ) -> str:
        """
        Generate a human-readable explanation for a diagnosis candidate.

        Args:
            candidate: Diagnosis candidate with score
            phenotypes: Patient's phenotypes
            kg: Knowledge graph for context
            paths: Supporting paths (optional)

        Returns:
            Human-readable explanation string
        """
        # Get disease name
        disease_node = kg.get_node(candidate.disease_id)
        disease_name = disease_node.name if disease_node else str(candidate.disease_id)

        # Build explanation sections
        sections = []

        # Header
        confidence = self._get_confidence_level(candidate.confidence_score)
        sections.append(
            f"## Diagnosis Candidate: {disease_name}\n"
            f"Confidence: {confidence.upper()} (score: {candidate.confidence_score:.3f})\n"
        )

        # Matching phenotypes
        matching = self._find_matching_phenotypes(
            candidate.disease_id, phenotypes, kg
        )
        if matching:
            sections.append("### Matching Phenotypes")
            for pheno_id, pheno_name in matching[:10]:
                sections.append(f"- {pheno_name} ({pheno_id})")
            sections.append("")

        # Key genes
        genes = self._find_connecting_genes(
            phenotypes, candidate.disease_id, kg, paths
        )
        if genes:
            sections.append("### Key Associated Genes")
            for gene_name, evidence_count in genes[:5]:
                sections.append(f"- {gene_name} ({evidence_count} evidence paths)")
            sections.append("")

        # Evidence summary
        if paths:
            sections.append("### Evidence Paths")

            # Direct evidence
            direct_paths = [p for p in paths if self._is_direct_path(p)]
            if direct_paths:
                sections.append("**Direct Evidence:**")
                for path in direct_paths[:3]:
                    sections.append(f"- {self._format_path(path, kg)}")
                sections.append("")

            # Pathway evidence
            pathway_paths = [p for p in paths if self._is_pathway_path(p)]
            if pathway_paths:
                sections.append("**Pathway-Mediated Evidence:**")
                for path in pathway_paths[:3]:
                    sections.append(f"- {self._format_path(path, kg)}")
                sections.append("")

            # Ortholog evidence (P1 feature)
            if self.include_ortholog_evidence:
                ortholog_paths = [p for p in paths if self._is_ortholog_path(p)]
                if ortholog_paths:
                    sections.append("**Cross-Species Evidence (Ortholog):**")
                    for path in ortholog_paths[:3]:
                        sections.append(f"- {self._format_path(path, kg)}")
                    sections.append("")

        # Disclaimer
        sections.append(
            "---\n"
            "*Note: This is a computational prediction for clinical review. "
            "Final diagnosis should be made by qualified healthcare professionals "
            "based on comprehensive clinical evaluation.*"
        )

        return "\n".join(sections)

    def generate_evidence_summary(
        self,
        candidate: DiagnosisCandidate,
        kg: "KnowledgeGraph",
        paths: Optional[List[ReasoningPath]] = None,
        phenotypes: Optional[PatientPhenotypes] = None,
    ) -> EvidenceSummary:
        """
        Generate a structured evidence summary.

        Args:
            candidate: Diagnosis candidate
            kg: Knowledge graph
            paths: Supporting paths
            phenotypes: Patient phenotypes

        Returns:
            EvidenceSummary with categorized evidence
        """
        disease_node = kg.get_node(candidate.disease_id)
        disease_name = disease_node.name if disease_node else str(candidate.disease_id)

        summary = EvidenceSummary(
            disease_id=candidate.disease_id,
            disease_name=disease_name,
            total_score=candidate.confidence_score,
            confidence_level=self._get_confidence_level(candidate.confidence_score),
        )

        # Extract evidence from paths
        if paths:
            for path in paths:
                evidence = self._create_evidence_item(path, kg)

                if self._is_direct_path(path):
                    if len(summary.direct_evidence) < self.max_paths_per_type:
                        summary.direct_evidence.append(evidence)
                elif self._is_pathway_path(path):
                    if len(summary.pathway_evidence) < self.max_paths_per_type:
                        summary.pathway_evidence.append(evidence)
                elif self._is_ortholog_path(path) and self.include_ortholog_evidence:
                    if len(summary.ortholog_evidence) < self.max_paths_per_type:
                        summary.ortholog_evidence.append(evidence)

        # Find matching phenotypes
        if phenotypes:
            matching = self._find_matching_phenotypes(
                candidate.disease_id, phenotypes, kg
            )
            summary.matching_phenotypes = [
                f"{name} ({pid})" for pid, name in matching[:10]
            ]

        # Find key genes
        if paths:
            genes = self._find_connecting_genes(
                phenotypes or PatientPhenotypes(phenotype_ids=[]),
                candidate.disease_id,
                kg,
                paths,
            )
            summary.key_genes = [name for name, _ in genes[:5]]

        return summary

    def _get_confidence_level(self, score: float) -> str:
        """Determine confidence level from score."""
        if score >= self.HIGH_CONFIDENCE_THRESHOLD:
            return "high"
        elif score >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return "medium"
        else:
            return "low"

    def _find_matching_phenotypes(
        self,
        disease_id: NodeID,
        phenotypes: PatientPhenotypes,
        kg: "KnowledgeGraph",
    ) -> List[Tuple[str, str]]:
        """
        Find patient phenotypes that match the disease.

        Returns:
            List of (phenotype_id, phenotype_name)
        """
        matching = []

        # Get disease's known phenotypes
        disease_phenotypes = set()
        neighbors = kg.get_neighbors(
            disease_id,
            edge_types=[EdgeType.PHENOTYPE_OF_DISEASE],
            direction="in",
        )
        for pheno_id, _ in neighbors:
            disease_phenotypes.add(str(pheno_id))

        # Check which patient phenotypes match
        # PatientPhenotypes.phenotypes contains HPO ID strings like "HP:0001"
        for hpo_id in phenotypes.phenotypes:
            # Convert string HPO ID to NodeID for KG lookup
            pheno_node_id = NodeID(source=DataSource.HPO, local_id=hpo_id)
            pheno_str = str(pheno_node_id)

            if pheno_str in disease_phenotypes:
                pheno_node = kg.get_node(pheno_node_id)
                pheno_name = pheno_node.name if pheno_node else hpo_id
                matching.append((hpo_id, pheno_name))

        return matching

    def _find_connecting_genes(
        self,
        phenotypes: PatientPhenotypes,
        disease_id: NodeID,
        kg: "KnowledgeGraph",
        paths: Optional[List[ReasoningPath]],
    ) -> List[Tuple[str, int]]:
        """
        Find genes that connect phenotypes to the disease.

        Returns:
            List of (gene_name, evidence_count) sorted by evidence count
        """
        gene_counts: Dict[str, int] = {}

        if paths:
            for path in paths:
                if path.target != disease_id:
                    continue

                # Find genes in path
                for node_id in path.nodes:
                    node = kg.get_node(node_id)
                    if node and node.node_type == NodeType.GENE:
                        gene_name = node.name
                        gene_counts[gene_name] = gene_counts.get(gene_name, 0) + 1

        # Sort by count
        sorted_genes = sorted(
            gene_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_genes

    def _is_direct_path(self, path: ReasoningPath) -> bool:
        """Check if path is direct (no pathway or ortholog)."""
        edge_values = {e.value for e in path.edges}

        ortholog_edges = {
            EdgeType.ORTHOLOG_OF.value,
            EdgeType.HUMAN_MOUSE_ORTHOLOG.value,
            EdgeType.HUMAN_ZEBRAFISH_ORTHOLOG.value,
        }
        pathway_edges = {
            EdgeType.GENE_IN_PATHWAY.value,
            EdgeType.PATHWAY_INVOLVES_GENE.value,
        }

        return not (edge_values & ortholog_edges) and not (edge_values & pathway_edges)

    def _is_pathway_path(self, path: ReasoningPath) -> bool:
        """Check if path involves pathways."""
        edge_values = {e.value for e in path.edges}
        pathway_edges = {
            EdgeType.GENE_IN_PATHWAY.value,
            EdgeType.PATHWAY_INVOLVES_GENE.value,
        }
        return bool(edge_values & pathway_edges)

    def _is_ortholog_path(self, path: ReasoningPath) -> bool:
        """Check if path involves orthologs (P1 feature)."""
        edge_values = {e.value for e in path.edges}
        ortholog_edges = {
            EdgeType.ORTHOLOG_OF.value,
            EdgeType.HUMAN_MOUSE_ORTHOLOG.value,
            EdgeType.HUMAN_ZEBRAFISH_ORTHOLOG.value,
            EdgeType.MOUSE_GENE_HAS_PHENOTYPE.value,
        }
        return bool(edge_values & ortholog_edges)

    def _format_path(
        self,
        path: ReasoningPath,
        kg: "KnowledgeGraph",
    ) -> str:
        """Format a path as human-readable string."""
        parts = []

        for i, node_id in enumerate(path.nodes):
            node = kg.get_node(node_id)
            node_name = node.name if node else str(node_id)

            parts.append(node_name)

            if i < len(path.edges):
                edge_type = path.edges[i].value.replace("_", " ")
                parts.append(f" --[{edge_type}]--> ")

        path_str = "".join(parts)
        return f"{path_str} (score: {path.score:.3f})"

    def _create_evidence_item(
        self,
        path: ReasoningPath,
        kg: "KnowledgeGraph",
    ) -> EvidenceItem:
        """Create an EvidenceItem from a path."""
        # Determine evidence type
        if self._is_ortholog_path(path):
            evidence_type = "ortholog"
        elif self._is_pathway_path(path):
            evidence_type = "pathway"
        else:
            evidence_type = "direct"

        # Build description
        description = self._format_path(path, kg)

        return EvidenceItem(
            evidence_type=evidence_type,
            description=description,
            confidence=path.score,
            source_nodes=list(path.nodes),
            supporting_path=path,
        )


# ==============================================================================
# Factory Function
# ==============================================================================
def create_explanation_generator(
    include_ortholog_evidence: bool = True,
    include_literature_evidence: bool = True,
) -> ExplanationGenerator:
    """
    Factory function to create ExplanationGenerator.

    Args:
        include_ortholog_evidence: Include cross-species evidence (P1 feature)
        include_literature_evidence: Include PubMed citations

    Returns:
        Configured ExplanationGenerator instance
    """
    return ExplanationGenerator(
        include_ortholog_evidence=include_ortholog_evidence,
        include_literature_evidence=include_literature_evidence,
    )
