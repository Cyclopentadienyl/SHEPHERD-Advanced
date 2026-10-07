"""
A two-phenotype graph for tests of the phenotype list through the API and UI.
=============================================================================
Two HPO phenotypes reach one disease through two genes, so a path-reasoning
pipeline built on it produces a real result for either phenotype or both. Shared
by tests/unit/test_diagnose_phenotype_input.py and
tests/unit/test_diagnosis_panel.py, which drive the same `/diagnose` route from
the API's side and from the WebUI's.

Module: tests/fixtures/phenotype_kg.py
"""
from __future__ import annotations

from src.core.types import DataSource, Edge, EdgeType, Node, NodeID, NodeType

SEIZURE = "HP:0001250"
DELAY = "HP:0001263"
#: Well formed, and not a node of this graph.
UNKNOWN = "HP:9999999"


def build_two_phenotype_kg():
    """Seizure and developmental delay, each through its own gene, to one disease."""
    from src.kg import KnowledgeGraph

    def node(source, local_id, node_type):
        return Node(
            id=NodeID(source=source, local_id=local_id),
            node_type=node_type,
            name=local_id,
            data_sources={source},
        )

    def edge(src, tgt, edge_type):
        return Edge(source_id=src.id, target_id=tgt.id, edge_type=edge_type, weight=0.9)

    kg = KnowledgeGraph()
    seizure = node(DataSource.HPO, SEIZURE, NodeType.PHENOTYPE)
    delay = node(DataSource.HPO, DELAY, NodeType.PHENOTYPE)
    scn1a = node(DataSource.DISGENET, "SCN1A", NodeType.GENE)
    mecp2 = node(DataSource.DISGENET, "MECP2", NodeType.GENE)
    disease = node(DataSource.MONDO, "MONDO:0100135", NodeType.DISEASE)
    for item in (seizure, delay, scn1a, mecp2, disease):
        kg.add_node(item)
    kg.add_edge(edge(scn1a, seizure, EdgeType.GENE_HAS_PHENOTYPE))
    kg.add_edge(edge(mecp2, delay, EdgeType.GENE_HAS_PHENOTYPE))
    kg.add_edge(edge(scn1a, disease, EdgeType.GENE_ASSOCIATED_WITH_DISEASE))
    kg.add_edge(edge(mecp2, disease, EdgeType.GENE_ASSOCIATED_WITH_DISEASE))
    return kg
