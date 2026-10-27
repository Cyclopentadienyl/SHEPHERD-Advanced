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
#   - A candidate of the result union (its score and the kind's label)
#   - Patient phenotypes
#   - Knowledge Graph for context lookup
#
# Output:
#   - Human-readable explanation string
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
#   3. Honest scores: name a score for what it is, never a confidence
#   4. Actionable: Highlight key phenotype-gene-disease connections
# ==============================================================================
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from src.core.types import (
    Candidate,
    DataSource,
    EdgeType,
    EvidenceSource,
    Node,
    NodeID,
    NodeType,
    PatientPhenotypes,
    candidate_score,
)
from src.reasoning.path_reasoning import ReasoningPath

if TYPE_CHECKING:
    from src.kg.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

#: What a candidate's score is, wherever the pipeline writes it for a reader
#: (each candidate's explanation, the run's summary): the value its result
#: kind ranks by (`served_score` or `path_score`, Spec 1 §2.1). It is not a
#: calibrated probability, so the text names no confidence band.
SCORE_MEANING = "a ranking score, not a calibrated disease probability"


# ==============================================================================
# Matching Phenotypes
# ==============================================================================
def find_matching_phenotypes(
    disease_id: NodeID,
    phenotypes: Sequence[str],
    kg: "KnowledgeGraph",
) -> List[str]:
    """The phenotypes the knowledge graph annotates directly to this disease.

    **One definition for the candidate field and the explanation.** The pipeline
    fills each candidate's `matching_phenotypes` with this, and the
    explanation's "Matching Phenotypes" section lists the first ten of the same
    ids, in the same order, then how many more the field holds: the section is
    cut for reading, never silently.

    A phenotype matches when the graph has a `phenotype_of_disease` edge from
    that exact term to the disease. An ontology ancestor or descendant of an
    annotated term does not match, and neither does a phenotype linked to the
    disease only through a gene, however short that path. An empty list
    therefore does not mean the case and the disease share no clinical features.

    `phenotypes` are HPO ids, kept in their order and spelling. The pipeline
    passes the request's phenotypes as used — known to the graph, repeats
    removed, in the request's spelling (`DiagnosisPipeline.validate_input`) —
    so what comes back is a subset of the request. Repeats are not removed
    here; that is the phenotype rule's job, applied before scoring.

    Cost: one scan of the disease's incoming edges.
    """
    annotated = {
        str(pheno_id)
        for pheno_id, _ in kg.get_neighbors(
            disease_id,
            edge_types=[EdgeType.PHENOTYPE_OF_DISEASE],
            direction="in",
        )
    }
    return [
        hpo_id
        for hpo_id in phenotypes
        if str(NodeID(source=DataSource.HPO, local_id=hpo_id)) in annotated
    ]


# ==============================================================================
# Explanation Generator
# ==============================================================================
class ExplanationGenerator:
    """
    Generate human-readable explanations for diagnosis results.

    Provides a narrative explanation for clinical review, which names the
    candidate's score with its result kind's label.

    Usage:
        generator = ExplanationGenerator()
        explanation = generator.generate_explanation(
            candidate=candidate,
            phenotypes=patient_phenotypes,
            kg=knowledge_graph,
            score_label=result.label,
        )
    """

    # Matching phenotypes listed by name in the explanation; the candidate's
    # `matching_phenotypes` carries all of them.
    MATCHING_PHENOTYPES_LISTED = 10

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
        candidate: Candidate,
        phenotypes: PatientPhenotypes,
        kg: "KnowledgeGraph",
        paths: Optional[List[ReasoningPath]] = None,
        score_label: Optional[str] = None,
    ) -> str:
        """
        Generate a human-readable explanation for a diagnosis candidate.

        Args:
            candidate: A candidate of the result union, with its kind's score
            phenotypes: Patient's phenotypes
            kg: Knowledge graph for context
            paths: Supporting paths (optional)
            score_label: The result kind's label (`InferenceResult.label`), so
                the score line says what the number is

        Returns:
            Human-readable explanation string
        """
        # Get disease name
        disease_node = kg.get_node(candidate.disease_id)
        disease_name = disease_node.name if disease_node else str(candidate.disease_id)

        # Build explanation sections
        sections = []

        # Header
        sections.append(
            f"## Diagnosis Candidate: {disease_name}\n"
            f"Score: {candidate_score(candidate):.3f} ({SCORE_MEANING})\n"
            # A paragraph of its own: one newline joins the lines in Markdown.
            + (f"\nRanking: {score_label}\n" if score_label else "")
        )

        # Matching phenotypes
        matching = self._find_matching_phenotypes(
            candidate.disease_id, phenotypes, kg
        )
        if matching:
            sections.append("### Matching Phenotypes")
            for pheno_id, pheno_name in matching[: self.MATCHING_PHENOTYPES_LISTED]:
                sections.append(f"- {pheno_name} ({pheno_id})")
            # A cut list must say it was cut: the field, and the WebUI's CSV
            # from it, carry every match.
            unlisted = len(matching) - self.MATCHING_PHENOTYPES_LISTED
            if unlisted > 0:
                sections.append(f"- …and {unlisted} more")
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

    def _find_matching_phenotypes(
        self,
        disease_id: NodeID,
        phenotypes: PatientPhenotypes,
        kg: "KnowledgeGraph",
    ) -> List[Tuple[str, str]]:
        """
        Find patient phenotypes that match the disease, as
        `find_matching_phenotypes` defines a match, with their names.

        Returns:
            List of (phenotype_id, phenotype_name)
        """
        matching = []
        for hpo_id in find_matching_phenotypes(disease_id, phenotypes.phenotypes, kg):
            pheno_node = kg.get_node(NodeID(source=DataSource.HPO, local_id=hpo_id))
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
