"""
Unit Tests for Inference Pipeline
=================================
Tests for DiagnosisPipeline and related components
"""
import pytest
from datetime import datetime
from typing import List

from src.core.types import (
    DataSource,
    Edge,
    EdgeType,
    Node,
    NodeID,
    NodeType,
    PatientPhenotypes,
    InferenceResult,
    DiagnosisCandidate,
)
from src.kg import KnowledgeGraph
from src.inference import (
    DiagnosisPipeline,
    PipelineConfig,
    create_diagnosis_pipeline,
)


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
    Create a simple KG for testing.

    Structure:
        Phenotype1 (HP:0001234) <-- Gene1 (BRCA1) --> Disease1 (MONDO:0001234)
        Phenotype2 (HP:0002345) <-- Gene2 (TP53)  --> Disease1 (MONDO:0001234)
                                    Gene2 (TP53)  --> Disease2 (MONDO:0003456)
    """
    kg = KnowledgeGraph()

    # Add phenotypes
    kg.add_node(make_node(DataSource.HPO, "HP:0001234", NodeType.PHENOTYPE, "Seizure"))
    kg.add_node(make_node(DataSource.HPO, "HP:0002345", NodeType.PHENOTYPE, "Ataxia"))

    # Add genes
    kg.add_node(make_node(DataSource.DISGENET, "BRCA1", NodeType.GENE, "BRCA1"))
    kg.add_node(make_node(DataSource.DISGENET, "TP53", NodeType.GENE, "TP53"))

    # Add diseases
    kg.add_node(make_node(DataSource.MONDO, "MONDO:0001234", NodeType.DISEASE, "Disease A"))
    kg.add_node(make_node(DataSource.MONDO, "MONDO:0003456", NodeType.DISEASE, "Disease B"))

    # Gene -> Phenotype edges
    kg.add_edge(make_edge(
        DataSource.DISGENET, "BRCA1",
        DataSource.HPO, "HP:0001234",
        EdgeType.GENE_HAS_PHENOTYPE,
        weight=0.9,
    ))
    kg.add_edge(make_edge(
        DataSource.DISGENET, "TP53",
        DataSource.HPO, "HP:0002345",
        EdgeType.GENE_HAS_PHENOTYPE,
        weight=0.8,
    ))

    # Gene -> Disease edges
    kg.add_edge(make_edge(
        DataSource.DISGENET, "BRCA1",
        DataSource.MONDO, "MONDO:0001234",
        EdgeType.GENE_ASSOCIATED_WITH_DISEASE,
        weight=0.95,
    ))
    kg.add_edge(make_edge(
        DataSource.DISGENET, "TP53",
        DataSource.MONDO, "MONDO:0001234",
        EdgeType.GENE_ASSOCIATED_WITH_DISEASE,
        weight=0.85,
    ))
    kg.add_edge(make_edge(
        DataSource.DISGENET, "TP53",
        DataSource.MONDO, "MONDO:0003456",
        EdgeType.GENE_ASSOCIATED_WITH_DISEASE,
        weight=0.7,
    ))

    return kg


@pytest.fixture
def patient_phenotypes():
    """Sample patient phenotypes with valid KG IDs"""
    return PatientPhenotypes(
        patient_id="test_patient_001",
        phenotypes=["HP:0001234", "HP:0002345"],
    )


@pytest.fixture
def patient_phenotypes_with_invalid():
    """Patient phenotypes with some invalid IDs"""
    return PatientPhenotypes(
        patient_id="test_patient_002",
        phenotypes=["HP:0001234", "HP:9999999", "HP:0002345"],
    )


# =============================================================================
# Test DiagnosisPipeline
# =============================================================================
class TestDiagnosisPipeline:
    """Test DiagnosisPipeline"""

    def test_create_pipeline(self, simple_kg):
        """Test creating pipeline"""
        pipeline = DiagnosisPipeline(kg=simple_kg)
        assert pipeline is not None
        assert pipeline.kg is simple_kg
        assert pipeline.config is not None

    def test_create_pipeline_with_config(self, simple_kg):
        """Test creating pipeline with custom config"""
        config = PipelineConfig(
            max_path_length=3,
            include_explanations=False,
        )
        pipeline = DiagnosisPipeline(kg=simple_kg, config=config)

        assert pipeline.config.max_path_length == 3
        assert pipeline.config.include_explanations is False

    def test_validate_input(self, simple_kg, patient_phenotypes):
        """Test input validation"""
        pipeline = DiagnosisPipeline(kg=simple_kg)
        result = pipeline.validate_input(patient_phenotypes)

        assert result.is_valid
        assert len(result.validated_phenotypes) == 2

    def test_run_basic(self, simple_kg, patient_phenotypes):
        """Test basic pipeline run"""
        pipeline = DiagnosisPipeline(kg=simple_kg)
        result = pipeline.run(
            patient_input=patient_phenotypes,
            top_k=5,
            include_explanations=False,
        )

        # Check result structure
        assert isinstance(result, InferenceResult)
        assert result.patient_id == patient_phenotypes.patient_id
        assert isinstance(result.timestamp, datetime)
        assert result.inference_time_ms >= 0

        # Should find candidates
        assert len(result.candidates) > 0

        # Check candidate structure
        for candidate in result.candidates:
            assert isinstance(candidate, DiagnosisCandidate)
            assert candidate.rank > 0
            assert candidate.confidence_score >= 0
            assert candidate.disease_name is not None

    def test_run_with_explanations(self, simple_kg, patient_phenotypes):
        """Test pipeline run with explanations"""
        pipeline = DiagnosisPipeline(kg=simple_kg)
        result = pipeline.run(
            patient_input=patient_phenotypes,
            top_k=5,
            include_explanations=True,
        )

        # Should have candidates with explanations
        assert len(result.candidates) > 0
        for candidate in result.candidates:
            assert candidate.explanation is not None
            assert len(candidate.explanation) > 0

        # Should have summary explanation
        assert result.summary_explanation is not None

    def test_run_candidates_ranked(self, simple_kg, patient_phenotypes):
        """Test that candidates are ranked correctly"""
        pipeline = DiagnosisPipeline(kg=simple_kg)
        result = pipeline.run(
            patient_input=patient_phenotypes,
            top_k=10,
        )

        # Ranks should be sequential
        for i, candidate in enumerate(result.candidates):
            assert candidate.rank == i + 1

        # Scores should be descending
        scores = [c.confidence_score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_run_with_invalid_phenotypes(self, simple_kg, patient_phenotypes_with_invalid):
        """Test pipeline with some invalid phenotypes"""
        config = PipelineConfig(validate_phenotypes=False)
        pipeline = DiagnosisPipeline(kg=simple_kg, config=config)
        result = pipeline.run(
            patient_input=patient_phenotypes_with_invalid,
            top_k=5,
        )

        # Should still work with valid phenotypes
        assert isinstance(result, InferenceResult)
        # May have warnings about invalid phenotypes

    def test_run_empty_phenotypes(self, simple_kg):
        """Test pipeline with no valid phenotypes"""
        pipeline = DiagnosisPipeline(kg=simple_kg)
        patient = PatientPhenotypes(
            patient_id="test",
            phenotypes=[],
        )
        result = pipeline.run(patient_input=patient, top_k=5)

        # Should return empty result with errors
        assert len(result.candidates) == 0
        assert len(result.warnings) > 0

    def test_run_nonexistent_phenotypes(self, simple_kg):
        """Test pipeline with phenotypes not in KG"""
        pipeline = DiagnosisPipeline(kg=simple_kg)
        patient = PatientPhenotypes(
            patient_id="test",
            phenotypes=["HP:9999999", "HP:8888888"],
        )
        result = pipeline.run(patient_input=patient, top_k=5)

        # Should handle gracefully
        assert isinstance(result, InferenceResult)

    def test_get_pipeline_config(self, simple_kg):
        """Test getting pipeline config"""
        pipeline = DiagnosisPipeline(kg=simple_kg)
        config_dict = pipeline.get_pipeline_config()

        assert "version" in config_dict
        assert "max_path_length" in config_dict
        assert "aggregation_method" in config_dict
        assert config_dict["has_model"] is False


# =============================================================================
# Test Factory Functions
# =============================================================================
class TestFactoryFunctions:
    """Test factory functions"""

    def test_create_diagnosis_pipeline(self, simple_kg):
        """Test factory function"""
        pipeline = create_diagnosis_pipeline(kg=simple_kg)
        assert isinstance(pipeline, DiagnosisPipeline)

    def test_create_diagnosis_pipeline_with_config(self, simple_kg):
        """Test factory with config"""
        config = PipelineConfig(max_path_length=5)
        pipeline = create_diagnosis_pipeline(kg=simple_kg, config=config)
        assert pipeline.config.max_path_length == 5


# =============================================================================
# Test PipelineConfig
# =============================================================================
class TestPipelineConfig:
    """Test PipelineConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = PipelineConfig()

        assert config.max_path_length == 4
        assert config.eta == 0.7  # Default 70% GNN, 30% shortest path
        assert config.sp_optional is True
        assert config.include_explanations is True
        assert config.include_ortholog_evidence is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = PipelineConfig(
            max_path_length=3,
            eta=0.5,
            sp_optional=False,
            include_explanations=False,
            include_ortholog_evidence=False,
        )

        assert config.max_path_length == 3
        assert config.eta == 0.5
        assert config.sp_optional is False
        assert config.include_explanations is False
        assert config.include_ortholog_evidence is False

    def test_eta_extreme_values(self):
        """eta=1.0 = pure GNN; eta=0.0 = pure shortest path"""
        config_pure_gnn = PipelineConfig(eta=1.0)
        assert config_pure_gnn.eta == 1.0

        config_pure_sp = PipelineConfig(eta=0.0)
        assert config_pure_sp.eta == 0.0


# =============================================================================
# Test GNN Inference Integration
# =============================================================================
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


def _build_gnn_test_fixtures(simple_kg):
    """
    Build a small GNN model + graph data that matches simple_kg's topology.

    Returns (model, graph_data, node_id_to_idx) for pipeline initialization.
    The model is untrained but structurally valid — enough to verify that
    GNN scoring is actually wired through (score != 0.0).
    """
    from src.models.gnn import ShepherdGNN, ShepherdGNNConfig

    hidden_dim = 32

    # Node ID → index mappings (must match x_dict dimensions)
    node_mapping = simple_kg.get_node_id_mapping()
    # e.g. {"phenotype": {"hpo:HP:0001234": 0, "hpo:HP:0002345": 1}, ...}

    num_phenotypes = len(node_mapping.get("phenotype", {}))
    num_genes = len(node_mapping.get("gene", {}))
    num_diseases = len(node_mapping.get("disease", {}))

    # Build x_dict with random features
    torch.manual_seed(42)
    x_dict = {
        "phenotype": torch.randn(num_phenotypes, hidden_dim),
        "gene": torch.randn(num_genes, hidden_dim),
        "disease": torch.randn(num_diseases, hidden_dim),
    }

    # Build edge_index_dict from KG edges
    # simple_kg has edges: Gene→Phenotype, Gene→Disease
    # We need bidirectional edges for message passing
    pheno_map = node_mapping["phenotype"]
    gene_map = node_mapping["gene"]
    disease_map = node_mapping["disease"]

    # gene -> phenotype (and reverse)
    gp_src, gp_dst = [], []
    # gene -> disease (and reverse)
    gd_src, gd_dst = [], []

    for edge in simple_kg._edges:
        src_str = str(edge.source_id)
        tgt_str = str(edge.target_id)

        if src_str in gene_map and tgt_str in pheno_map:
            gp_src.append(gene_map[src_str])
            gp_dst.append(pheno_map[tgt_str])
        elif src_str in gene_map and tgt_str in disease_map:
            gd_src.append(gene_map[src_str])
            gd_dst.append(disease_map[tgt_str])

    edge_index_dict = {}

    if gp_src:
        gp_edges = torch.tensor([gp_src, gp_dst], dtype=torch.long)
        edge_index_dict[("gene", "has_phenotype", "phenotype")] = gp_edges
        edge_index_dict[("phenotype", "rev_has_phenotype", "gene")] = gp_edges.flip(0)

    if gd_src:
        gd_edges = torch.tensor([gd_src, gd_dst], dtype=torch.long)
        edge_index_dict[("gene", "associated_with", "disease")] = gd_edges
        edge_index_dict[("disease", "rev_associated_with", "gene")] = gd_edges.flip(0)

    graph_data = {
        "x_dict": x_dict,
        "edge_index_dict": edge_index_dict,
        "num_nodes_dict": {
            "phenotype": num_phenotypes,
            "gene": num_genes,
            "disease": num_diseases,
        },
    }

    # Build metadata for GNN
    node_types = sorted(x_dict.keys())
    edge_types = list(edge_index_dict.keys())
    metadata = (node_types, edge_types)

    in_channels_dict = {k: v.shape[1] for k, v in x_dict.items()}

    config = ShepherdGNNConfig(
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        conv_type="gat",
        dropout=0.0,
    )

    model = ShepherdGNN(
        metadata=metadata,
        in_channels_dict=in_channels_dict,
        config=config,
    )
    model.eval()

    return model, graph_data


class TestGNNInference:
    """Test that GNN scoring is properly wired into the inference pipeline."""

    @pytest.fixture
    def gnn_pipeline(self, simple_kg):
        """Create a pipeline with a real (untrained) GNN model."""
        model, graph_data = _build_gnn_test_fixtures(simple_kg)
        pipeline = DiagnosisPipeline(
            kg=simple_kg,
            model=model,
            graph_data=graph_data,
        )
        return pipeline

    def test_gnn_ready(self, gnn_pipeline):
        """GNN should be marked as ready when model + graph_data are provided."""
        assert gnn_pipeline._gnn_ready is True

    def test_node_embeddings_precomputed(self, gnn_pipeline):
        """Node embeddings should be populated after init."""
        emb = gnn_pipeline._node_embeddings
        assert emb is not None
        assert "phenotype" in emb
        assert "disease" in emb
        # Embeddings should be on CPU
        assert emb["disease"].device.type == "cpu"

    def test_gnn_score_not_zero(self, gnn_pipeline, patient_phenotypes):
        """GNN score should NOT be 0.0 (the old stub behavior)."""
        result = gnn_pipeline.run(
            patient_input=patient_phenotypes,
            top_k=5,
            include_explanations=False,
        )
        assert len(result.candidates) > 0
        for c in result.candidates:
            # With a real model, gnn_score should be non-zero
            assert c.gnn_score != 0.0, (
                f"{c.disease_name}: gnn_score is 0.0 — GNN not wired correctly"
            )

    def test_confidence_uses_gnn_score(self, gnn_pipeline, patient_phenotypes):
        """When GNN is active, confidence_score should equal gnn_score."""
        result = gnn_pipeline.run(
            patient_input=patient_phenotypes,
            top_k=5,
            include_explanations=False,
        )
        for c in result.candidates:
            assert c.confidence_score == c.gnn_score, (
                f"{c.disease_name}: confidence={c.confidence_score} != "
                f"gnn={c.gnn_score}"
            )

    def test_gnn_score_in_valid_range(self, gnn_pipeline, patient_phenotypes):
        """GNN score should be normalized to [0, 1]."""
        result = gnn_pipeline.run(
            patient_input=patient_phenotypes,
            top_k=5,
            include_explanations=False,
        )
        for c in result.candidates:
            assert 0.0 <= c.gnn_score <= 1.0, (
                f"{c.disease_name}: gnn_score={c.gnn_score} out of [0,1]"
            )

    def test_pipeline_config_reports_gnn(self, gnn_pipeline):
        """get_pipeline_config should report GNN status.

        scoring_mode values (Step B+):
          - "gnn_plus_shortest_path" — GNN + SP table both loaded
          - "gnn_only"                — GNN loaded, SP table not found (eta=1.0 effective)
          - "path_reasoning_fallback" — no GNN (PathReasoner only)
        This fixture doesn't generate shortest_paths.pt → expect "gnn_only".
        """
        config = gnn_pipeline.get_pipeline_config()
        assert config["gnn_ready"] is True
        assert config["has_model"] is True
        assert config["scoring_mode"] in ("gnn_only", "gnn_plus_shortest_path")
        # Step C: evidence is decoupled from scoring
        assert config["path_reasoner_role"] == "explanation_only"

    def test_path_reasoner_still_runs(self, gnn_pipeline, patient_phenotypes):
        """Path reasoning should still provide explanation paths alongside GNN."""
        result = gnn_pipeline.run(
            patient_input=patient_phenotypes,
            top_k=5,
            include_explanations=False,
        )
        # At least some candidates should have reasoning paths
        candidates_with_paths = [
            c for c in result.candidates if len(c.reasoning_paths) > 0
        ]
        assert len(candidates_with_paths) > 0

    def test_without_gnn_uses_reasoning_fallback(self, simple_kg, patient_phenotypes):
        """Without GNN, confidence should fall back to reasoning score."""
        pipeline = DiagnosisPipeline(kg=simple_kg)
        assert pipeline._gnn_ready is False

        result = pipeline.run(
            patient_input=patient_phenotypes,
            top_k=5,
            include_explanations=False,
        )
        for c in result.candidates:
            assert c.gnn_score == 0.0
            assert c.confidence_score == c.reasoning_score


# =============================================================================
# The phenotype-list rule, through pipeline.run
# =============================================================================
# docs/working/PLAN_PHENOTYPE_NORMALISATION.md: repeats are removed by graph node
# after mapping, first occurrence kept, and nothing is truncated. These tests go
# through `run` (or `validate_input`, which `run` calls first) rather than
# restating the rule; the rule alone is tested in test_phenotype_normalisation.py.
SEIZURE = "HP:0001234"
ATAXIA = "HP:0002345"
UNKNOWN = "HP:9999999"


def _patient(phenotypes, confidences=None):
    return PatientPhenotypes(
        patient_id="norm_patient",
        phenotypes=list(phenotypes),
        phenotype_confidences=confidences,
    )


def _ranking(result):
    return [
        (c.disease_id, c.rank, c.confidence_score, c.gnn_score, c.reasoning_score)
        for c in result.candidates
    ]


class TestPhenotypeListRule:
    """The served path applies the one phenotype-list rule."""

    def test_a_repeat_is_scored_once(self, simple_kg):
        pipeline = DiagnosisPipeline(kg=simple_kg)

        with_repeat = pipeline.run(_patient([SEIZURE, SEIZURE, ATAXIA]), top_k=5)
        without = pipeline.run(_patient([SEIZURE, ATAXIA]), top_k=5)

        assert _ranking(with_repeat) == _ranking(without)
        assert with_repeat.phenotype_input.used == (SEIZURE, ATAXIA)
        assert with_repeat.phenotype_input.repeats_removed == 1
        assert "Removed 1 repeated phenotype entry; each phenotype is scored once." in (
            with_repeat.warnings
        )

    def test_a_repeat_is_scored_once_by_the_gnn_too(self, simple_kg):
        """The GNN mean and the SP mean were where a repeat weighed most.

        **This is the test whose scores move.** On `simple_kg` path reasoning
        happens to rank `[A, A, B]` and `[A, B]` identically even without the
        rule — checked by disabling it — so the test above relies on its account
        and warning assertions. With the rule disabled, this one fails."""
        model, graph_data = _build_gnn_test_fixtures(simple_kg)
        pipeline = DiagnosisPipeline(kg=simple_kg, model=model, graph_data=graph_data)
        assert pipeline._gnn_ready

        with_repeat = pipeline.run(_patient([SEIZURE, SEIZURE, ATAXIA]), top_k=5)
        without = pipeline.run(_patient([SEIZURE, ATAXIA]), top_k=5)

        assert _ranking(with_repeat) == _ranking(without)
        assert any(c.gnn_score != 0.0 for c in with_repeat.candidates)

    def test_one_hundred_repeats_then_another_phenotype_keeps_both(self, simple_kg):
        """The previous version truncated to the first 100 entries before
        removing anything, so this input scored the seizure alone."""
        pipeline = DiagnosisPipeline(kg=simple_kg)

        result = pipeline.run(_patient([SEIZURE] * 100 + [ATAXIA]), top_k=5)
        both = pipeline.run(_patient([SEIZURE, ATAXIA]), top_k=5)

        assert _ranking(result) == _ranking(both)
        assert result.phenotype_input.received == 101
        assert result.phenotype_input.used == (SEIZURE, ATAXIA)
        assert "Removed 99 repeated phenotype entries; each phenotype is scored once." in (
            result.warnings
        )
        assert not any("using first" in w for w in result.warnings)

    def test_too_many_distinct_phenotypes_are_refused_not_truncated(self, simple_kg):
        pipeline = DiagnosisPipeline(kg=simple_kg, config=PipelineConfig(max_phenotypes=1))

        result = pipeline.run(_patient([SEIZURE, ATAXIA]), top_k=5)

        assert result.candidates == []
        assert any(
            "2 distinct phenotypes found in the graph; at most 1 are accepted" in w
            and "not shortened" in w
            for w in result.warnings
        )

    def test_the_count_is_taken_after_repeats_are_removed(self, simple_kg):
        pipeline = DiagnosisPipeline(kg=simple_kg, config=PipelineConfig(max_phenotypes=1))

        result = pipeline.run(_patient([SEIZURE, SEIZURE, SEIZURE]), top_k=5)

        assert result.candidates
        assert result.phenotype_input.used == (SEIZURE,)

    def test_unknown_ids_keep_their_policy_and_are_accounted_for(self, simple_kg):
        pipeline = DiagnosisPipeline(kg=simple_kg)

        result = pipeline.run(_patient([UNKNOWN, SEIZURE, ATAXIA]), top_k=5)

        assert f"Unknown phenotype: {UNKNOWN}" in result.warnings
        assert result.phenotype_input.unknown == (UNKNOWN,)
        assert result.phenotype_input.used == (SEIZURE, ATAXIA)

    def test_everything_after_validation_reads_the_list_as_used(
        self, simple_kg, monkeypatch
    ):
        """Scoring receives the normalised input, with each confidence still at
        the phenotype it was sent with: `[X, A, A, B]` keeps positions 1 and 3."""
        pipeline = DiagnosisPipeline(kg=simple_kg)
        seen = {}
        original = pipeline._score_and_rank_candidates

        def capture(**kwargs):
            seen["input"] = kwargs["patient_input"]
            return original(**kwargs)

        monkeypatch.setattr(pipeline, "_score_and_rank_candidates", capture)

        result = pipeline.run(
            _patient([UNKNOWN, SEIZURE, SEIZURE, ATAXIA], confidences=[0.1, 0.2, 0.3, 0.4]),
            top_k=5,
            include_explanations=True,
        )

        assert seen["input"].phenotypes == [SEIZURE, ATAXIA]
        assert seen["input"].phenotype_confidences == [0.2, 0.4]
        assert "Based on 2 input phenotypes" in result.summary_explanation

    @pytest.mark.parametrize("confidences", [[], [0.5], [0.5, 0.5, 0.5]])
    def test_a_confidence_list_of_the_wrong_length_is_refused(self, simple_kg, confidences):
        pipeline = DiagnosisPipeline(kg=simple_kg)

        result = pipeline.run(_patient([SEIZURE, ATAXIA], confidences=confidences), top_k=5)

        assert result.candidates == []
        assert result.phenotype_input is None
        assert any(
            f"phenotype_confidences has {len(confidences)} entries for 2 phenotypes" in w
            for w in result.warnings
        )

    def test_no_confidences_and_the_right_length_both_pass(self, simple_kg):
        pipeline = DiagnosisPipeline(kg=simple_kg)

        assert pipeline.validate_input(_patient([SEIZURE, ATAXIA])).is_valid
        assert pipeline.validate_input(_patient([SEIZURE, ATAXIA], [0.9, 0.8])).is_valid
