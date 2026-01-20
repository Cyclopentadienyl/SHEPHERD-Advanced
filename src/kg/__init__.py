"""
# ==============================================================================
# Module: src/kg/__init__.py
# ==============================================================================
# Purpose: Knowledge Graph module - heterogeneous graph for medical reasoning
#
# Dependencies:
#   - External: networkx, numpy, torch_geometric (optional), scipy (optional)
#   - Internal: src.core.types, src.core.schema, src.ontology
#
# Exports:
#   - KnowledgeGraph: Core graph data structure
#   - KnowledgeGraphBuilder: Graph construction from data sources
#   - KGBuilderConfig: Builder configuration
#   - create_kg_builder: Factory function
#   - preprocess_for_gnn: Preprocessing pipeline for GNN input
#   - compute_laplacian_pe: Laplacian positional encoding
#   - compute_rwse: Random walk structural encoding
#
# Usage:
#   from src.kg import KnowledgeGraph, KnowledgeGraphBuilder, preprocess_for_gnn
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
#
#   # Prepare for GNN
#   data, pe_dict = preprocess_for_gnn(kg)
#   metadata = kg.metadata()
#   model = ShepherdGNN(metadata=metadata, ...)
# ==============================================================================
"""

from src.kg.graph import KnowledgeGraph
from src.kg.builder import (
    KGBuilderConfig,
    KnowledgeGraphBuilder,
    create_kg_builder,
)
from src.kg.preprocessing import (
    compute_laplacian_pe,
    compute_rwse,
    compute_degree_features,
    preprocess_for_gnn,
)

__all__ = [
    # Core graph
    "KnowledgeGraph",
    # Builder
    "KGBuilderConfig",
    "KnowledgeGraphBuilder",
    "create_kg_builder",
    # Preprocessing
    "preprocess_for_gnn",
    "compute_laplacian_pe",
    "compute_rwse",
    "compute_degree_features",
]
