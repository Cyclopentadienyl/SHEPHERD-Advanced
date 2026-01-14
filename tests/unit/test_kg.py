"""
Unit Tests for Knowledge Graph Module
=====================================
Tests for KnowledgeGraph and KnowledgeGraphBuilder
"""
import pytest
from pathlib import Path

from src.core.types import (
    DataSource,
    Edge,
    EdgeType,
    Node,
    NodeID,
    NodeType,
)
from src.kg import (
    KnowledgeGraph,
    KnowledgeGraphBuilder,
    KGBuilderConfig,
    create_kg_builder,
)


# =============================================================================
# Helper Functions
# =============================================================================
def make_node_id(source: DataSource, local_id: str) -> NodeID:
    """Create a NodeID"""
    return NodeID(source=source, local_id=local_id)


def make_node(source: DataSource, local_id: str, node_type: NodeType, name: str, **kwargs) -> Node:
    """Create a Node"""
    return Node(
        id=make_node_id(source, local_id),
        node_type=node_type,
        name=name,
        data_sources={source},
        **kwargs
    )


def make_edge(
    src_source: DataSource, src_id: str,
    tgt_source: DataSource, tgt_id: str,
    edge_type: EdgeType,
    weight: float = 1.0,
) -> Edge:
    """Create an Edge"""
    return Edge(
        source_id=make_node_id(src_source, src_id),
        target_id=make_node_id(tgt_source, tgt_id),
        edge_type=edge_type,
        weight=weight,
    )


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def sample_nodes():
    """Create sample nodes for testing"""
    return [
        make_node(DataSource.HPO, "HP:0001250", NodeType.PHENOTYPE, "Seizure"),
        make_node(DataSource.HPO, "HP:0001251", NodeType.PHENOTYPE, "Ataxia"),
        make_node(DataSource.HPO, "HP:0000118", NodeType.PHENOTYPE, "Phenotypic abnormality"),
        make_node(DataSource.MONDO, "MONDO:0001234", NodeType.DISEASE, "Test Disease"),
        make_node(DataSource.DISGENET, "BRCA1", NodeType.GENE, "BRCA1", attributes={"symbol": "BRCA1", "entrez_id": "672"}),
    ]


@pytest.fixture
def sample_edges():
    """Create sample edges for testing"""
    return [
        # IS_A hierarchy
        make_edge(DataSource.HPO, "HP:0001250", DataSource.HPO, "HP:0000118", EdgeType.IS_A),
        make_edge(DataSource.HPO, "HP:0001251", DataSource.HPO, "HP:0000118", EdgeType.IS_A),
        # Phenotype-Disease
        make_edge(DataSource.HPO, "HP:0001250", DataSource.MONDO, "MONDO:0001234", EdgeType.PHENOTYPE_OF_DISEASE, 0.8),
        # Gene-Disease
        make_edge(DataSource.DISGENET, "BRCA1", DataSource.MONDO, "MONDO:0001234", EdgeType.GENE_ASSOCIATED_WITH_DISEASE, 0.9),
    ]


@pytest.fixture
def graph(sample_nodes, sample_edges):
    """Create a populated graph for testing"""
    g = KnowledgeGraph()
    g.add_nodes(sample_nodes)
    g.add_edges(sample_edges)
    return g


# =============================================================================
# Test KnowledgeGraph Basic Operations
# =============================================================================
class TestKnowledgeGraphBasic:
    """Test basic KnowledgeGraph operations"""

    def test_create_empty_graph(self):
        """Test creating an empty graph"""
        g = KnowledgeGraph()
        assert g.total_nodes == 0
        assert g.total_edges == 0

    def test_add_node(self, sample_nodes):
        """Test adding nodes"""
        g = KnowledgeGraph()
        g.add_node(sample_nodes[0])

        assert g.total_nodes == 1
        assert g.has_node(sample_nodes[0].id)

    def test_add_duplicate_node(self, sample_nodes):
        """Test adding duplicate node (should be ignored)"""
        g = KnowledgeGraph()
        g.add_node(sample_nodes[0])
        g.add_node(sample_nodes[0])  # Duplicate

        assert g.total_nodes == 1

    def test_add_nodes(self, sample_nodes):
        """Test adding multiple nodes"""
        g = KnowledgeGraph()
        g.add_nodes(sample_nodes)

        assert g.total_nodes == len(sample_nodes)

    def test_get_node(self, graph, sample_nodes):
        """Test getting a node"""
        node = graph.get_node(sample_nodes[0].id)
        assert node is not None
        assert node.name == "Seizure"

    def test_get_node_not_found(self, graph):
        """Test getting non-existent node"""
        fake_id = make_node_id(DataSource.HPO, "fake123")
        node = graph.get_node(fake_id)
        assert node is None

    def test_has_node(self, graph, sample_nodes):
        """Test node existence check"""
        assert graph.has_node(sample_nodes[0].id)
        assert not graph.has_node(make_node_id(DataSource.HPO, "fake"))

    def test_get_nodes_by_type(self, graph):
        """Test getting nodes by type"""
        phenotypes = graph.get_nodes_by_type(NodeType.PHENOTYPE)
        assert len(phenotypes) == 3

        genes = graph.get_nodes_by_type(NodeType.GENE)
        assert len(genes) == 1


# =============================================================================
# Test KnowledgeGraph Edge Operations
# =============================================================================
class TestKnowledgeGraphEdges:
    """Test edge operations"""

    def test_add_edge(self, sample_nodes):
        """Test adding an edge"""
        g = KnowledgeGraph()
        g.add_nodes(sample_nodes[:2])

        edge = make_edge(
            DataSource.HPO, "HP:0001250",
            DataSource.HPO, "HP:0001251",
            EdgeType.IS_A
        )
        g.add_edge(edge)

        assert g.total_edges == 1

    def test_add_edge_missing_node(self, sample_nodes):
        """Test adding edge with missing node (should skip)"""
        g = KnowledgeGraph()
        g.add_node(sample_nodes[0])

        # Target node not in graph
        edge = make_edge(
            DataSource.HPO, "HP:0001250",
            DataSource.HPO, "HP:0001251",
            EdgeType.IS_A
        )
        g.add_edge(edge)

        assert g.total_edges == 0

    def test_get_edges_by_source(self, graph):
        """Test getting edges by source"""
        source_id = make_node_id(DataSource.HPO, "HP:0001250")
        edges = graph.get_edges(source_id=source_id)

        assert len(edges) == 2  # IS_A + PHENOTYPE_OF_DISEASE

    def test_get_edges_by_type(self, graph):
        """Test getting edges by type"""
        edges = graph.get_edges(edge_type=EdgeType.IS_A)
        assert len(edges) == 2

        edges = graph.get_edges(edge_type=EdgeType.GENE_ASSOCIATED_WITH_DISEASE)
        assert len(edges) == 1

    def test_num_edges_by_type(self, graph):
        """Test edge count by type"""
        counts = graph.num_edges
        assert counts[EdgeType.IS_A] == 2
        assert counts[EdgeType.PHENOTYPE_OF_DISEASE] == 1


# =============================================================================
# Test KnowledgeGraph Neighbor Queries
# =============================================================================
class TestKnowledgeGraphNeighbors:
    """Test neighbor queries"""

    def test_get_neighbors_outgoing(self, graph):
        """Test getting outgoing neighbors"""
        node_id = make_node_id(DataSource.HPO, "HP:0001250")
        neighbors = graph.get_neighbors(node_id, direction="out")

        assert len(neighbors) == 2
        neighbor_ids = [str(n[0]) for n in neighbors]
        assert "hpo:HP:0000118" in neighbor_ids
        assert "mondo:MONDO:0001234" in neighbor_ids

    def test_get_neighbors_incoming(self, graph):
        """Test getting incoming neighbors"""
        node_id = make_node_id(DataSource.HPO, "HP:0000118")
        neighbors = graph.get_neighbors(node_id, direction="in")

        assert len(neighbors) == 2  # Two phenotypes point to it

    def test_get_neighbors_filtered(self, graph):
        """Test getting neighbors filtered by edge type"""
        node_id = make_node_id(DataSource.HPO, "HP:0001250")
        neighbors = graph.get_neighbors(
            node_id,
            edge_types=[EdgeType.IS_A],
            direction="out"
        )

        assert len(neighbors) == 1
        assert neighbors[0][1] == EdgeType.IS_A


# =============================================================================
# Test KnowledgeGraph Subgraph
# =============================================================================
class TestKnowledgeGraphSubgraph:
    """Test subgraph extraction"""

    def test_get_subgraph_one_hop(self, graph):
        """Test extracting 1-hop subgraph"""
        seed_id = make_node_id(DataSource.HPO, "HP:0001250")
        subgraph = graph.get_subgraph([seed_id], num_hops=1)

        # Should include seed + direct neighbors
        assert subgraph.total_nodes >= 2
        assert subgraph.has_node(seed_id)

    def test_get_subgraph_two_hops(self, graph):
        """Test extracting 2-hop subgraph"""
        seed_id = make_node_id(DataSource.DISGENET, "BRCA1")
        subgraph = graph.get_subgraph([seed_id], num_hops=2)

        # BRCA1 -> Disease -> Phenotype (2 hops)
        assert subgraph.total_nodes >= 2


# =============================================================================
# Test KnowledgeGraph Export
# =============================================================================
class TestKnowledgeGraphExport:
    """Test export methods"""

    def test_to_networkx(self, graph):
        """Test export to NetworkX"""
        G = graph.to_networkx()

        assert G.number_of_nodes() == graph.total_nodes
        assert G.number_of_edges() == graph.total_edges

    def test_to_edge_list(self, graph):
        """Test export to edge list"""
        edges = graph.to_edge_list()

        assert len(edges) == graph.total_edges
        assert all(len(e) == 4 for e in edges)  # (src, tgt, type, weight)

    def test_get_statistics(self, graph):
        """Test getting statistics"""
        stats = graph.get_statistics()

        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "nodes_by_type" in stats
        assert "edges_by_type" in stats


# =============================================================================
# Test KnowledgeGraphBuilder
# =============================================================================
class TestKnowledgeGraphBuilder:
    """Test KnowledgeGraphBuilder"""

    def test_create_builder(self):
        """Test creating builder"""
        builder = KnowledgeGraphBuilder()
        assert builder.graph is not None
        assert builder.graph.total_nodes == 0

    def test_create_builder_with_config(self):
        """Test creating builder with config"""
        config = KGBuilderConfig(
            include_ontology_hierarchy=False,
            include_orthologs=False,
        )
        builder = KnowledgeGraphBuilder(config=config)
        assert builder.config.include_ontology_hierarchy is False

    def test_add_gene_disease_associations(self):
        """Test adding gene-disease associations"""
        builder = KnowledgeGraphBuilder()

        # First add a disease node
        disease_node = make_node(DataSource.MONDO, "MONDO:0001234", NodeType.DISEASE, "Test Disease")
        builder.graph.add_node(disease_node)

        # Add associations
        associations = [
            {
                "gene_id": "BRCA1",
                "gene_symbol": "BRCA1",
                "disease_id": "MONDO:0001234",
                "score": 0.9,
            },
            {
                "gene_id": "TP53",
                "gene_symbol": "TP53",
                "disease_id": "MONDO:0001234",
                "score": 0.8,
            },
        ]

        genes_added, edges_added = builder.add_gene_disease_associations(associations)

        assert genes_added == 2
        assert edges_added == 2

    def test_add_phenotype_disease_annotations(self):
        """Test adding phenotype-disease annotations"""
        builder = KnowledgeGraphBuilder()

        # Add phenotype and disease nodes
        pheno_node = make_node(DataSource.HPO, "HP:0001250", NodeType.PHENOTYPE, "Seizure")
        disease_node = make_node(DataSource.MONDO, "MONDO:0001234", NodeType.DISEASE, "Test Disease")
        builder.graph.add_node(pheno_node)
        builder.graph.add_node(disease_node)

        # Add annotations
        annotations = [
            {
                "phenotype_id": "HP:0001250",
                "disease_id": "MONDO:0001234",
                "frequency": 0.75,
            },
        ]

        edges_added = builder.add_phenotype_disease_annotations(annotations)
        assert edges_added == 1

    def test_build(self):
        """Test building final graph"""
        builder = KnowledgeGraphBuilder()

        # Add some nodes
        for i in range(5):
            node = make_node(DataSource.HPO, f"HP:000000{i}", NodeType.PHENOTYPE, f"Test {i}")
            builder.graph.add_node(node)

        kg = builder.build()
        assert kg.total_nodes == 5

    def test_get_build_summary(self):
        """Test getting build summary"""
        builder = KnowledgeGraphBuilder()
        summary = builder.get_build_summary()

        assert "graph_stats" in summary
        assert "sources_added" in summary


# =============================================================================
# Test Factory Function
# =============================================================================
class TestFactoryFunction:
    """Test factory function"""

    def test_create_kg_builder(self):
        """Test factory function"""
        builder = create_kg_builder()
        assert builder is not None
        assert isinstance(builder, KnowledgeGraphBuilder)

    def test_create_kg_builder_with_config(self):
        """Test factory with config"""
        config = KGBuilderConfig(min_association_score=0.5)
        builder = create_kg_builder(config=config)
        assert builder.config.min_association_score == 0.5


# =============================================================================
# Integration Test: Builder with Ontology
# =============================================================================
class TestBuilderOntologyIntegration:
    """Test builder with ontology integration"""

    @pytest.fixture
    def mini_ontology(self):
        """Create a minimal ontology for testing"""
        from src.ontology import OBOParser, Ontology
        from pathlib import Path

        fixtures_dir = Path(__file__).parent.parent / "fixtures"
        mini_hpo_path = fixtures_dir / "mini_hpo.obo"

        if mini_hpo_path.exists():
            parser = OBOParser()
            header, terms = parser.parse_file(mini_hpo_path)
            return Ontology(header, terms)
        return None

    def test_add_ontology(self, mini_ontology):
        """Test adding ontology to builder"""
        if mini_ontology is None:
            pytest.skip("mini_hpo.obo not found")

        builder = KnowledgeGraphBuilder()
        nodes_added = builder.add_ontology(mini_ontology, NodeType.PHENOTYPE)

        assert nodes_added > 0
        assert builder.graph.total_nodes == nodes_added

        # Check IS_A edges were added
        is_a_edges = builder.graph.get_edges(edge_type=EdgeType.IS_A)
        assert len(is_a_edges) > 0

    def test_add_ontology_no_hierarchy(self, mini_ontology):
        """Test adding ontology without hierarchy"""
        if mini_ontology is None:
            pytest.skip("mini_hpo.obo not found")

        builder = KnowledgeGraphBuilder()
        nodes_added = builder.add_ontology(
            mini_ontology,
            NodeType.PHENOTYPE,
            include_hierarchy=False
        )

        assert nodes_added > 0
        # No IS_A edges should be added
        is_a_edges = builder.graph.get_edges(edge_type=EdgeType.IS_A)
        assert len(is_a_edges) == 0
