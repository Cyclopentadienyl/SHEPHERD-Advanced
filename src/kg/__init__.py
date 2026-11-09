"""
# ==============================================================================
# Module: src/kg/__init__.py
# ==============================================================================
# Purpose: Knowledge Graph module - heterogeneous graph for medical reasoning
#
# Dependencies:
#   - External: networkx, numpy, torch (optional), scipy (optional)
#   - Internal: src.core.types, src.core.schema, src.ontology
#
# Exports:
#   - KnowledgeGraph: Core graph data structure
#   - KnowledgeGraphBuilder: Graph construction from data sources
#   - KGBuilderConfig: Builder configuration
#   - create_kg_builder: Factory function
#   - DiseaseAllocation, allocate_diseases, train_only_allocation: disease split
#   - build_eligible_disease_profiles, generate_training_samples,
#     retained_phenotype_count: sample generation
#
# Usage:
#   from src.kg import KnowledgeGraph, KnowledgeGraphBuilder
#   from src.ontology import OntologyLoader
#
#   # Build KG from ontologies
#   loader = OntologyLoader()
#   hpo = loader.load_hpo()
#
#   builder = KnowledgeGraphBuilder()
#   builder.add_ontology(hpo, NodeType.PHENOTYPE)
#   kg = builder.build()
#
#   # Query graph
#   neighbors = kg.get_neighbors(node_id)
#   subgraph = kg.get_subgraph(seed_nodes, num_hops=2)
# ==============================================================================
"""

from src.kg.graph import KnowledgeGraph
from src.kg.builder import (
    KGBuilderConfig,
    KnowledgeGraphBuilder,
    create_kg_builder,
)
from src.kg.disease_allocation import (
    DiseaseAllocation,
    allocate_diseases,
    train_only_allocation,
)
from src.kg.sample_generator import (
    build_eligible_disease_profiles,
    generate_training_samples,
    retained_phenotype_count,
)

__all__ = [
    # Core graph
    "KnowledgeGraph",
    # Builder
    "KGBuilderConfig",
    "KnowledgeGraphBuilder",
    "create_kg_builder",
    # Sample generation
    "DiseaseAllocation",
    "allocate_diseases",
    "build_eligible_disease_profiles",
    "train_only_allocation",
    "generate_training_samples",
    "retained_phenotype_count",
]
