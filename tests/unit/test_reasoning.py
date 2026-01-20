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
    EvidenceSummary,
    EvidenceItem,
    create_explanation_generator,
)
from src.core.types import DiagnosisCandidate, PatientPhenotypes


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
        assert config.aggregation_method == "weighted_sum"

    def test_custom_config(self):
        """Test custom configuration"""
        config = PathReasoningConfig(
            max_path_length=3,
            length_penalty=0.8,
            aggregation_method="max",
        )

        assert config.max_path_length == 3
        assert config.length_penalty == 0.8
        assert config.aggregation_method == "max"


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

    def test_aggregate_path_scores(self, simple_kg, patient_phenotypes):
        """Test aggregating path scores"""
        reasoner = PathReasoner()
        paths = reasoner.find_paths(
            source_ids=patient_phenotypes,
            target_type=NodeType.DISEASE,
            kg=simple_kg,
        )
        scored_paths = reasoner.score_paths(paths, simple_kg)
        target_scores = reasoner.aggregate_path_scores(scored_paths)

        # Should have scores for diseases
        assert len(target_scores) > 0

        # All scores should be positive
        assert all(score > 0 for score in target_scores.values())

    def test_get_top_candidates(self, simple_kg, patient_phenotypes):
        """Test getting top disease candidates"""
        reasoner = PathReasoner()
        candidates = reasoner.get_top_candidates(
            source_ids=patient_phenotypes,
            kg=simple_kg,
            top_k=5,
        )

        # Should return candidates
        assert len(candidates) > 0

        # Each candidate should have (disease_id, score, paths)
        for disease_id, score, paths in candidates:
            assert isinstance(score, float)
            assert score > 0
            assert isinstance(paths, list)

        # Should be sorted by score (descending)
        scores = [score for _, score, _ in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_aggregation_methods(self, simple_kg, patient_phenotypes):
        """Test different aggregation methods"""
        for method in ["max", "mean", "sum", "weighted_sum"]:
            config = PathReasoningConfig(aggregation_method=method)
            reasoner = PathReasoner(config=config)

            candidates = reasoner.get_top_candidates(
                source_ids=patient_phenotypes,
                kg=simple_kg,
                top_k=5,
            )

            assert len(candidates) > 0


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
        """Create a DiagnosisCandidate for testing"""
        return DiagnosisCandidate(
            rank=1,
            disease_id=make_node_id(DataSource.MONDO, "MONDO:001"),
            disease_name="Disease A",
            confidence_score=0.85,
            gnn_score=0.8,
            reasoning_score=0.9,
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

    def test_generate_evidence_summary(
        self, simple_kg, diagnosis_candidate, patient_phenos
    ):
        """Test generating evidence summary"""
        generator = ExplanationGenerator()

        # Convert string HPO IDs to NodeIDs for path finding
        source_ids = [
            make_node_id(DataSource.HPO, hpo_id)
            for hpo_id in patient_phenos.phenotypes
        ]

        # Get paths
        reasoner = PathReasoner()
        paths = reasoner.find_paths(
            source_ids=source_ids,
            target_type=NodeType.DISEASE,
            kg=simple_kg,
        )
        scored_paths = reasoner.score_paths(paths, simple_kg)

        disease_paths = [
            p for p in scored_paths
            if str(p.target) == str(diagnosis_candidate.disease_id)
        ]

        summary = generator.generate_evidence_summary(
            candidate=diagnosis_candidate,
            kg=simple_kg,
            paths=disease_paths,
            phenotypes=patient_phenos,
        )

        # Check summary structure
        assert isinstance(summary, EvidenceSummary)
        assert summary.disease_name is not None
        assert summary.total_score == diagnosis_candidate.confidence_score
        assert summary.confidence_level in ["high", "medium", "low"]

    def test_evidence_summary_to_dict(
        self, simple_kg, diagnosis_candidate, patient_phenos
    ):
        """Test evidence summary serialization"""
        generator = ExplanationGenerator()

        summary = generator.generate_evidence_summary(
            candidate=diagnosis_candidate,
            kg=simple_kg,
            phenotypes=patient_phenos,
        )

        summary_dict = summary.to_dict()

        assert "disease_id" in summary_dict
        assert "disease_name" in summary_dict
        assert "total_score" in summary_dict
        assert "evidence" in summary_dict

    def test_confidence_levels(self):
        """Test confidence level determination"""
        generator = ExplanationGenerator()

        # High confidence
        assert generator._get_confidence_level(0.8) == "high"
        # Medium confidence
        assert generator._get_confidence_level(0.5) == "medium"
        # Low confidence
        assert generator._get_confidence_level(0.2) == "low"


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
