"""
Unit Tests for Reasoning Module
================================
Tests for PathReasoner and related components
"""
import pytest
from typing import List

from src.core.types import (
    DataSource,
    Edge,
    EdgeType,
    Node,
    NodeID,
    NodeType,
)
from src.kg import KnowledgeGraph
from src.reasoning import (
    PathReasoner,
    PathReasoningConfig,
    ReasoningPath,
    DirectPathFinder,
    create_path_reasoner,
    ExplanationGenerator,
    create_explanation_generator,
)
from src.core.types import DiseaseTaskCandidate, PatientPhenotypes


# =============================================================================
# Helper Functions
# =============================================================================
def make_node_id(source: DataSource, local_id: str) -> NodeID:
    """Create a NodeID"""
    return NodeID(source=source, local_id=local_id)


def make_node(source: DataSource, local_id: str, node_type: NodeType, name: str) -> Node:
    """Create a Node"""
    return Node(
        id=make_node_id(source, local_id),
        node_type=node_type,
        name=name,
        data_sources={source},
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
def simple_kg():
    """
    Create a simple KG for testing path reasoning.

    Structure:
        Phenotype1 (HP:0001) <-- Gene1 (BRCA1) --> Disease1 (MONDO:001)
        Phenotype2 (HP:0002) <-- Gene2 (TP53)  --> Disease1 (MONDO:001)
                                 Gene2 (TP53)  --> Disease2 (MONDO:002)
    """
    kg = KnowledgeGraph()

    # Add phenotypes
    kg.add_node(make_node(DataSource.HPO, "HP:0001", NodeType.PHENOTYPE, "Seizure"))
    kg.add_node(make_node(DataSource.HPO, "HP:0002", NodeType.PHENOTYPE, "Ataxia"))

    # Add genes
    kg.add_node(make_node(DataSource.DISGENET, "BRCA1", NodeType.GENE, "BRCA1"))
    kg.add_node(make_node(DataSource.DISGENET, "TP53", NodeType.GENE, "TP53"))

    # Add diseases
    kg.add_node(make_node(DataSource.MONDO, "MONDO:001", NodeType.DISEASE, "Disease A"))
    kg.add_node(make_node(DataSource.MONDO, "MONDO:002", NodeType.DISEASE, "Disease B"))

    # Gene -> Phenotype edges
    kg.add_edge(make_edge(
        DataSource.DISGENET, "BRCA1",
        DataSource.HPO, "HP:0001",
        EdgeType.GENE_HAS_PHENOTYPE,
        weight=0.9,
    ))
    kg.add_edge(make_edge(
        DataSource.DISGENET, "TP53",
        DataSource.HPO, "HP:0002",
        EdgeType.GENE_HAS_PHENOTYPE,
        weight=0.8,
    ))

    # Gene -> Disease edges
    kg.add_edge(make_edge(
        DataSource.DISGENET, "BRCA1",
        DataSource.MONDO, "MONDO:001",
        EdgeType.GENE_ASSOCIATED_WITH_DISEASE,
        weight=0.95,
    ))
    kg.add_edge(make_edge(
        DataSource.DISGENET, "TP53",
        DataSource.MONDO, "MONDO:001",
        EdgeType.GENE_ASSOCIATED_WITH_DISEASE,
        weight=0.85,
    ))
    kg.add_edge(make_edge(
        DataSource.DISGENET, "TP53",
        DataSource.MONDO, "MONDO:002",
        EdgeType.GENE_ASSOCIATED_WITH_DISEASE,
        weight=0.7,
    ))

    return kg


@pytest.fixture
def patient_phenotypes():
    """Sample patient phenotypes"""
    return [
        make_node_id(DataSource.HPO, "HP:0001"),
        make_node_id(DataSource.HPO, "HP:0002"),
    ]


# =============================================================================
# Test ReasoningPath
# =============================================================================
class TestReasoningPath:
    """Test ReasoningPath data structure"""

    def test_path_creation(self):
        """Test creating a reasoning path"""
        nodes = [
            make_node_id(DataSource.HPO, "HP:0001"),
            make_node_id(DataSource.DISGENET, "BRCA1"),
            make_node_id(DataSource.MONDO, "MONDO:001"),
        ]
        edges = [
            EdgeType.GENE_HAS_PHENOTYPE,
            EdgeType.GENE_ASSOCIATED_WITH_DISEASE,
        ]

        path = ReasoningPath(nodes=nodes, edges=edges, score=0.5)

        assert path.length == 2
        assert path.source == nodes[0]
        assert path.target == nodes[-1]
        assert path.score == 0.5

    def test_path_repr(self):
        """Test path string representation"""
        nodes = [
            make_node_id(DataSource.HPO, "HP:0001"),
            make_node_id(DataSource.MONDO, "MONDO:001"),
        ]
        edges = [EdgeType.PHENOTYPE_OF_DISEASE]

        path = ReasoningPath(nodes=nodes, edges=edges, score=0.75)
        repr_str = repr(path)

        assert "ReasoningPath" in repr_str
        assert "0.75" in repr_str


# =============================================================================
# Test PathReasoningConfig
# =============================================================================
class TestPathReasoningConfig:
    """Test PathReasoningConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = PathReasoningConfig()

        assert config.max_path_length == 4
        assert config.length_penalty == 0.9

    def test_custom_config(self):
        """Test custom configuration"""
        config = PathReasoningConfig(
            max_path_length=3,
            length_penalty=0.8,
        )

        assert config.max_path_length == 3
        assert config.length_penalty == 0.8


# =============================================================================
# Test PathReasoner
# =============================================================================
class TestPathReasoner:
    """Test PathReasoner"""

    def test_create_reasoner(self):
        """Test creating a path reasoner"""
        reasoner = PathReasoner()
        assert reasoner.config is not None

    def test_create_reasoner_with_config(self):
        """Test creating reasoner with custom config"""
        config = PathReasoningConfig(max_path_length=3)
        reasoner = PathReasoner(config=config)
        assert reasoner.config.max_path_length == 3

    def test_find_paths(self, simple_kg, patient_phenotypes):
        """Test finding paths from phenotypes to diseases"""
        reasoner = PathReasoner()
        paths = reasoner.find_paths(
            source_ids=patient_phenotypes,
            target_type=NodeType.DISEASE,
            kg=simple_kg,
            max_length=3,
        )

        # Should find paths to both diseases
        assert len(paths) > 0

        # Check path structure
        for path in paths:
            assert path.source in patient_phenotypes
            # Target should be a disease node
            target_node = simple_kg.get_node(path.target)
            assert target_node.node_type == NodeType.DISEASE

    def test_find_paths_missing_source(self, simple_kg):
        """Test finding paths with missing source node"""
        reasoner = PathReasoner()
        missing_id = make_node_id(DataSource.HPO, "HP:9999")

        paths = reasoner.find_paths(
            source_ids=[missing_id],
            target_type=NodeType.DISEASE,
            kg=simple_kg,
        )

        assert len(paths) == 0

    def test_score_paths(self, simple_kg, patient_phenotypes):
        """Test scoring paths"""
        reasoner = PathReasoner()
        paths = reasoner.find_paths(
            source_ids=patient_phenotypes,
            target_type=NodeType.DISEASE,
            kg=simple_kg,
        )

        scored_paths = reasoner.score_paths(paths, simple_kg)

        # Paths should be scored
        assert all(p.score > 0 for p in scored_paths)

        # Should be sorted by score (descending)
        scores = [p.score for p in scored_paths]
        assert scores == sorted(scores, reverse=True)

    def test_paths_rank_no_disease(self):
        """K7 (PATH2): a path score orders a disease's evidence paths; nothing
        aggregates path scores into a ranking of diseases."""
        for retired in ("aggregate_path_scores", "get_top_candidates", "_get_path_type_weight"):
            assert not hasattr(PathReasoner, retired)


# =============================================================================
# Test DirectPathFinder
# =============================================================================
class TestDirectPathFinder:
    """Test DirectPathFinder"""

    def test_find_direct_paths(self, simple_kg, patient_phenotypes):
        """Test finding direct phenotype-gene-disease paths"""
        finder = DirectPathFinder(simple_kg)
        paths = finder.find_phenotype_gene_disease_paths(patient_phenotypes)

        # Should find direct paths
        assert len(paths) > 0

        # All paths should be length 2 (phenotype -> gene -> disease)
        for path in paths:
            assert path.length == 2


# =============================================================================
# Test Factory Function
# =============================================================================
class TestFactoryFunction:
    """Test factory function"""

    def test_create_path_reasoner(self):
        """Test factory function"""
        reasoner = create_path_reasoner()
        assert isinstance(reasoner, PathReasoner)

    def test_create_path_reasoner_with_config(self):
        """Test factory with config"""
        config = PathReasoningConfig(max_path_length=5)
        reasoner = create_path_reasoner(config=config)
        assert reasoner.config.max_path_length == 5


# =============================================================================
# Test ExplanationGenerator
# =============================================================================
class TestExplanationGenerator:
    """Test ExplanationGenerator"""

    @pytest.fixture
    def patient_phenos(self):
        """Create PatientPhenotypes for testing"""
        return PatientPhenotypes(
            patient_id="test_patient_001",
            phenotypes=["HP:0001", "HP:0002"],
        )

    @pytest.fixture
    def diagnosis_candidate(self):
        """Create a candidate (disease_task/v1) for testing"""
        return DiseaseTaskCandidate(
            rank=1,
            disease_id=make_node_id(DataSource.MONDO, "MONDO:001"),
            disease_name="Disease A",
            score=-0.85,
        )

    def test_create_generator(self):
        """Test creating explanation generator"""
        generator = ExplanationGenerator()
        assert generator is not None
        assert generator.include_ortholog_evidence is True

    def test_create_generator_no_ortholog(self):
        """Test creating generator without ortholog evidence"""
        generator = ExplanationGenerator(include_ortholog_evidence=False)
        assert generator.include_ortholog_evidence is False

    def test_generate_explanation(
        self, simple_kg, diagnosis_candidate, patient_phenos
    ):
        """Test generating explanation"""
        generator = ExplanationGenerator()

        # Convert string HPO IDs to NodeIDs for path finding
        source_ids = [
            make_node_id(DataSource.HPO, hpo_id)
            for hpo_id in patient_phenos.phenotypes
        ]

        # Get paths for explanation
        reasoner = PathReasoner()
        paths = reasoner.find_paths(
            source_ids=source_ids,
            target_type=NodeType.DISEASE,
            kg=simple_kg,
        )
        scored_paths = reasoner.score_paths(paths, simple_kg)

        # Filter paths to this disease
        disease_paths = [
            p for p in scored_paths
            if str(p.target) == str(diagnosis_candidate.disease_id)
        ]

        explanation = generator.generate_explanation(
            candidate=diagnosis_candidate,
            phenotypes=patient_phenos,
            kg=simple_kg,
            paths=disease_paths,
        )

        # Should produce explanation text
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "Disease A" in explanation or "MONDO:001" in explanation
        # The score line prints the candidate's own score, and the kind's label
        # when the pipeline passes it.
        assert "Score: -0.850 (" in explanation
        labelled = generator.generate_explanation(
            candidate=diagnosis_candidate, phenotypes=patient_phenos, kg=simple_kg,
            paths=disease_paths, score_label="Disease task model: every disease ranked",
        )
        assert "Ranking: Disease task model: every disease ranked" in labelled
        # A paragraph of its own: Markdown joins lines one newline apart.
        assert ")\n\nRanking: Disease task model" in labelled

# =============================================================================
# Test Explanation Factory Function
# =============================================================================
class TestExplanationFactoryFunction:
    """Test explanation generator factory"""

    def test_create_explanation_generator(self):
        """Test factory function"""
        generator = create_explanation_generator()
        assert isinstance(generator, ExplanationGenerator)

    def test_create_with_options(self):
        """Test factory with options"""
        generator = create_explanation_generator(
            include_ortholog_evidence=False,
            include_literature_evidence=False,
        )
        assert generator.include_ortholog_evidence is False
        assert generator.include_literature_evidence is False
