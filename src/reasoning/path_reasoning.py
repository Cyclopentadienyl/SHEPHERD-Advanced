"""
# ==============================================================================
# Module: src/reasoning/path_reasoning.py
# ==============================================================================
# Purpose: Multi-hop path reasoning for disease diagnosis (DR.KNOWS style)
#
# Dependencies:
#   - External: numpy
#   - Internal: src.core.types, src.kg (KnowledgeGraph)
#
# Input:
#   - Source nodes: Patient phenotypes (HPO terms)
#   - Target type: Disease nodes
#   - Knowledge Graph with phenotype-gene-disease relationships
#
# Output:
#   - Ranked paths from phenotypes to diseases
#   - Aggregated disease scores with evidence
#
# Design Notes:
#   - Uses BFS for path enumeration (efficient for bounded lengths)
#   - Path scoring considers edge weights and path length
#   - Supports multiple path aggregation strategies (max, mean, weighted)
#   - Ortholog paths handled separately for interpretability
#
# Key Paths for Rare Disease Diagnosis:
#   1. Phenotype → Gene → Disease (direct association)
#   2. Phenotype → Gene → Pathway → Gene → Disease (pathway-mediated)
#   3. Phenotype → Gene → Ortholog → Disease (cross-species, P1 feature)
#
# References:
#   - DR.KNOWS: Path-based reasoning for rare disease
#   - SHEPHERD: Heterogeneous graph reasoning for diagnosis
# ==============================================================================
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from src.core.types import (
    EdgeType,
    EvidenceSource,
    Node,
    NodeID,
    NodeType,
)

if TYPE_CHECKING:
    from src.kg.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


# ==============================================================================
# Path Data Structures
# ==============================================================================
@dataclass
class ReasoningPath:
    """
    A reasoning path from source to target.

    Attributes:
        nodes: List of NodeIDs in the path
        edges: List of EdgeTypes connecting nodes
        score: Path score (higher = more relevant)
        evidence: Supporting evidence for this path
    """
    nodes: List[NodeID]
    edges: List[EdgeType]
    score: float = 0.0
    evidence: List[EvidenceSource] = field(default_factory=list)

    @property
    def length(self) -> int:
        """Path length (number of edges)"""
        return len(self.edges)

    @property
    def source(self) -> NodeID:
        """Source node"""
        return self.nodes[0]

    @property
    def target(self) -> NodeID:
        """Target node"""
        return self.nodes[-1]

    def __repr__(self) -> str:
        path_str = " → ".join(str(n) for n in self.nodes)
        return f"ReasoningPath({path_str}, score={self.score:.4f})"


@dataclass
class PathReasoningConfig:
    """Configuration for path reasoning."""

    # Path search parameters
    max_path_length: int = 4
    max_paths_per_source: int = 100
    max_total_paths: int = 10000

    # Scoring parameters
    length_penalty: float = 0.9  # Multiply score by this for each hop
    edge_weight_power: float = 1.0  # Power to raise edge weights

    # Aggregation method: "max", "mean", "sum", "weighted_sum"
    aggregation_method: str = "weighted_sum"

    # Path type weights (for weighted aggregation)
    direct_path_weight: float = 1.0
    pathway_path_weight: float = 0.8
    ortholog_path_weight: float = 0.6

    # Edge type preferences (higher = preferred)
    edge_type_weights: Dict[str, float] = field(default_factory=lambda: {
        # Direct associations (strongest)
        EdgeType.GENE_ASSOCIATED_WITH_DISEASE.value: 1.0,
        EdgeType.GENE_CAUSES_DISEASE.value: 1.2,
        EdgeType.PHENOTYPE_OF_DISEASE.value: 1.0,
        EdgeType.GENE_HAS_PHENOTYPE.value: 0.9,
        # Hierarchy
        EdgeType.IS_A.value: 0.7,
        # Pathway
        EdgeType.GENE_IN_PATHWAY.value: 0.6,
        EdgeType.PATHWAY_INVOLVES_GENE.value: 0.6,
        # Ortholog (P1 feature)
        EdgeType.ORTHOLOG_OF.value: 0.5,
        EdgeType.HUMAN_MOUSE_ORTHOLOG.value: 0.5,
        EdgeType.MOUSE_GENE_HAS_PHENOTYPE.value: 0.4,
    })


# ==============================================================================
# Path Reasoner
# ==============================================================================
class PathReasoner:
    """
    Multi-hop path reasoning for disease diagnosis.

    Finds and scores paths from patient phenotypes to candidate diseases
    through the knowledge graph.

    Usage:
        reasoner = PathReasoner(config)
        paths = reasoner.find_paths(phenotype_ids, NodeType.DISEASE, kg)
        scored_paths = reasoner.score_paths(paths, kg)
        disease_scores = reasoner.aggregate_path_scores(scored_paths)
    """

    def __init__(self, config: Optional[PathReasoningConfig] = None):
        """
        Args:
            config: Reasoning configuration
        """
        self.config = config or PathReasoningConfig()

    def find_paths(
        self,
        source_ids: List[NodeID],
        target_type: NodeType,
        kg: "KnowledgeGraph",
        max_length: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[ReasoningPath]:
        """
        Find paths from source nodes to target type.

        Uses BFS to enumerate all paths up to max_length.

        Args:
            source_ids: Source nodes (e.g., patient phenotypes)
            target_type: Target node type (e.g., DISEASE)
            kg: Knowledge graph
            max_length: Override max path length
            top_k: Override max paths per source

        Returns:
            List of ReasoningPath objects
        """
        max_len = max_length or self.config.max_path_length
        max_paths = top_k or self.config.max_paths_per_source

        all_paths: List[ReasoningPath] = []

        for source_id in source_ids:
            if not kg.has_node(source_id):
                logger.warning(f"Source node {source_id} not in graph, skipping")
                continue

            paths = self._bfs_find_paths(
                source_id=source_id,
                target_type=target_type,
                kg=kg,
                max_length=max_len,
                max_paths=max_paths,
            )
            all_paths.extend(paths)

            if len(all_paths) >= self.config.max_total_paths:
                logger.info(f"Reached max total paths ({self.config.max_total_paths})")
                break

        logger.info(f"Found {len(all_paths)} paths from {len(source_ids)} sources")
        return all_paths

    def _bfs_find_paths(
        self,
        source_id: NodeID,
        target_type: NodeType,
        kg: "KnowledgeGraph",
        max_length: int,
        max_paths: int,
    ) -> List[ReasoningPath]:
        """
        BFS path enumeration from single source.

        Args:
            source_id: Starting node
            target_type: Target node type to reach
            kg: Knowledge graph
            max_length: Maximum path length
            max_paths: Maximum paths to return

        Returns:
            List of paths to target type nodes
        """
        paths: List[ReasoningPath] = []

        # BFS state: (current_node, path_nodes, path_edges)
        queue: List[Tuple[NodeID, List[NodeID], List[EdgeType]]] = [
            (source_id, [source_id], [])
        ]

        visited_states: Set[Tuple[str, int]] = set()  # (node_id, path_length)

        while queue and len(paths) < max_paths:
            current_id, path_nodes, path_edges = queue.pop(0)
            current_len = len(path_edges)

            if current_len >= max_length:
                continue

            # Get neighbors
            neighbors = kg.get_neighbors(current_id, direction="both")

            for neighbor_id, edge_type in neighbors:
                # Avoid cycles
                if neighbor_id in path_nodes:
                    continue

                # State for cycle detection
                state = (str(neighbor_id), current_len + 1)
                if state in visited_states:
                    continue
                visited_states.add(state)

                # Build new path
                new_nodes = path_nodes + [neighbor_id]
                new_edges = path_edges + [edge_type]

                # Check if reached target type
                neighbor_node = kg.get_node(neighbor_id)
                if neighbor_node and neighbor_node.node_type == target_type:
                    paths.append(ReasoningPath(
                        nodes=new_nodes,
                        edges=new_edges,
                    ))

                    if len(paths) >= max_paths:
                        break

                # Continue BFS if not at max length
                if len(new_edges) < max_length:
                    queue.append((neighbor_id, new_nodes, new_edges))

        return paths

    def score_paths(
        self,
        paths: List[ReasoningPath],
        kg: "KnowledgeGraph",
    ) -> List[ReasoningPath]:
        """
        Score each path based on edge weights and path length.

        Scoring formula:
            score = Π(edge_weight * edge_type_weight) * length_penalty^length

        Args:
            paths: Paths to score
            kg: Knowledge graph (for edge weights)

        Returns:
            Paths with updated scores (sorted by score descending)
        """
        for path in paths:
            score = 1.0

            for i, edge_type in enumerate(path.edges):
                # Get edge weight from KG
                if i < len(path.nodes) - 1:
                    edge = kg.get_edge(path.nodes[i], path.nodes[i + 1], edge_type)
                    if edge is None:
                        # Try reverse direction
                        edge = kg.get_edge(path.nodes[i + 1], path.nodes[i], edge_type)

                    edge_weight = edge.weight if edge else 0.5

                    # Apply edge type weight
                    type_weight = self.config.edge_type_weights.get(
                        edge_type.value, 0.5
                    )

                    # Combine weights
                    score *= (edge_weight ** self.config.edge_weight_power) * type_weight

            # Apply length penalty
            score *= self.config.length_penalty ** path.length

            path.score = score

        # Sort by score (descending)
        paths.sort(key=lambda p: p.score, reverse=True)

        return paths

    def aggregate_path_scores(
        self,
        scored_paths: List[ReasoningPath],
        target_ids: Optional[List[NodeID]] = None,
    ) -> Dict[NodeID, float]:
        """
        Aggregate scores for paths leading to the same target.

        Args:
            scored_paths: Paths with scores
            target_ids: Optional filter for specific targets

        Returns:
            {target_id: aggregated_score}
        """
        # Group paths by target
        target_paths: Dict[str, List[ReasoningPath]] = defaultdict(list)

        for path in scored_paths:
            target_key = str(path.target)
            if target_ids is None or path.target in target_ids:
                target_paths[target_key].append(path)

        # Aggregate scores
        target_scores: Dict[NodeID, float] = {}

        for target_key, paths in target_paths.items():
            if not paths:
                continue

            target_id = paths[0].target
            scores = [p.score for p in paths]

            if self.config.aggregation_method == "max":
                agg_score = max(scores)
            elif self.config.aggregation_method == "mean":
                agg_score = sum(scores) / len(scores)
            elif self.config.aggregation_method == "sum":
                agg_score = sum(scores)
            elif self.config.aggregation_method == "weighted_sum":
                # Weight by path type
                weighted_scores = []
                for p in paths:
                    weight = self._get_path_type_weight(p)
                    weighted_scores.append(p.score * weight)
                agg_score = sum(weighted_scores)
            else:
                agg_score = max(scores)

            target_scores[target_id] = agg_score

        return target_scores

    def _get_path_type_weight(self, path: ReasoningPath) -> float:
        """
        Determine path type and return corresponding weight.

        Path types:
        - Direct: Phenotype → Gene → Disease
        - Pathway: Contains pathway nodes
        - Ortholog: Contains ortholog edges (P1 feature)
        """
        edge_types_set = set(e.value for e in path.edges)

        # Check for ortholog edges
        ortholog_edges = {
            EdgeType.ORTHOLOG_OF.value,
            EdgeType.HUMAN_MOUSE_ORTHOLOG.value,
            EdgeType.HUMAN_ZEBRAFISH_ORTHOLOG.value,
            EdgeType.MOUSE_GENE_HAS_PHENOTYPE.value,
        }
        if edge_types_set & ortholog_edges:
            return self.config.ortholog_path_weight

        # Check for pathway edges
        pathway_edges = {
            EdgeType.GENE_IN_PATHWAY.value,
            EdgeType.PATHWAY_INVOLVES_GENE.value,
        }
        if edge_types_set & pathway_edges:
            return self.config.pathway_path_weight

        # Default: direct path
        return self.config.direct_path_weight

    def get_top_candidates(
        self,
        source_ids: List[NodeID],
        kg: "KnowledgeGraph",
        target_type: NodeType = NodeType.DISEASE,
        top_k: int = 10,
    ) -> List[Tuple[NodeID, float, List[ReasoningPath]]]:
        """
        Get top disease candidates with supporting paths.

        Convenience method combining find_paths, score_paths, and aggregation.

        Args:
            source_ids: Patient phenotypes
            kg: Knowledge graph
            target_type: Target type (usually DISEASE)
            top_k: Number of top candidates to return

        Returns:
            List of (disease_id, score, supporting_paths)
        """
        # Find and score paths
        paths = self.find_paths(source_ids, target_type, kg)
        scored_paths = self.score_paths(paths, kg)

        # Aggregate by target
        target_scores = self.aggregate_path_scores(scored_paths)

        # Sort by score
        sorted_targets = sorted(
            target_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Collect supporting paths for each target
        results = []
        for target_id, score in sorted_targets:
            supporting = [p for p in scored_paths if p.target == target_id]
            results.append((target_id, score, supporting[:10]))  # Top 10 paths per disease

        return results


# ==============================================================================
# Specialized Path Finders
# ==============================================================================
class DirectPathFinder:
    """
    Find direct paths: Phenotype → Gene → Disease.

    Optimized for the most common diagnostic reasoning pattern.
    """

    def __init__(self, kg: "KnowledgeGraph"):
        self.kg = kg

    def find_phenotype_gene_disease_paths(
        self,
        phenotype_ids: List[NodeID],
        max_genes: int = 100,
    ) -> List[ReasoningPath]:
        """
        Find Phenotype → Gene → Disease paths.

        Args:
            phenotype_ids: Patient phenotypes
            max_genes: Max intermediate genes to consider

        Returns:
            List of direct paths
        """
        paths: List[ReasoningPath] = []

        for pheno_id in phenotype_ids:
            # Step 1: Phenotype → Gene
            gene_edges = [
                EdgeType.GENE_HAS_PHENOTYPE,  # Reverse direction
            ]

            # Get genes associated with this phenotype
            neighbors = self.kg.get_neighbors(
                pheno_id,
                edge_types=gene_edges,
                direction="in"  # Genes point to phenotypes
            )

            genes = [(n, et) for n, et in neighbors
                     if self.kg.get_node(n) and
                     self.kg.get_node(n).node_type == NodeType.GENE][:max_genes]

            for gene_id, pheno_gene_edge in genes:
                # Step 2: Gene → Disease
                disease_edges = [
                    EdgeType.GENE_ASSOCIATED_WITH_DISEASE,
                    EdgeType.GENE_CAUSES_DISEASE,
                ]

                disease_neighbors = self.kg.get_neighbors(
                    gene_id,
                    edge_types=disease_edges,
                    direction="out"
                )

                for disease_id, gene_disease_edge in disease_neighbors:
                    disease_node = self.kg.get_node(disease_id)
                    if disease_node and disease_node.node_type == NodeType.DISEASE:
                        paths.append(ReasoningPath(
                            nodes=[pheno_id, gene_id, disease_id],
                            edges=[pheno_gene_edge, gene_disease_edge],
                        ))

        return paths


# ==============================================================================
# Factory Function
# ==============================================================================
def create_path_reasoner(
    config: Optional[PathReasoningConfig] = None,
) -> PathReasoner:
    """
    Factory function to create PathReasoner.

    Args:
        config: Optional configuration

    Returns:
        Configured PathReasoner instance
    """
    return PathReasoner(config=config)
