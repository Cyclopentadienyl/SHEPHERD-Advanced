"""
A two-phenotype graph, and the phenotype ids the API and UI tests send.
=======================================================================
Two HPO phenotypes reach one disease through two genes. Since K7 no graph is
served without a trained model, so the graph is the input of the no-model
refusal (`tests/unit/test_diagnose_phenotype_input.py`): a graph alone is
refused, never ranked by path reasoning. The ids are the demo graph's, so the
tests that rank (served by a disease child, `tests/fixtures/trained_children`)
send the same ones.

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
