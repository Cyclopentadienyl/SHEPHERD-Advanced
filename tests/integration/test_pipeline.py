"""
Integration Tests: GNN Pipeline + Vector Index
===============================================
End-to-end tests that verify:
  1. GNN model → precompute embeddings → score candidates
  2. Build vector index from embeddings → save → load → ANN search
  3. Full pipeline: BFS paths + GNN scoring + ANN candidate discovery

These tests use small synthetic data and run on CPU in seconds.
No GPU or large datasets required.

Module: tests/integration/test_pipeline.py
"""
from __future__ import annotations

import json
import pytest
import numpy as np
from pathlib import Path
from typing import Dict

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.core.types import (
    DataSource,
    Edge,
    EdgeType,
    Node,
    NodeID,
    NodeType,
    PatientPhenotypes,
)
from src.kg import KnowledgeGraph
from src.inference import DiagnosisPipeline, PipelineConfig, create_diagnosis_pipeline
from src.models.gnn import ShepherdGNN, ShepherdGNNConfig


# =============================================================================
# Fixtures
# =============================================================================
def _make_node_id(source: DataSource, local_id: str) -> NodeID:
    return NodeID(source=source, local_id=local_id)


def _make_node(source: DataSource, local_id: str, ntype: NodeType, name: str) -> Node:
    return Node(
        id=_make_node_id(source, local_id),
        node_type=ntype,
        name=name,
        data_sources={source},
    )


def _make_edge(
    src_source: DataSource, src_id: str,
    tgt_source: DataSource, tgt_id: str,
    edge_type: EdgeType, weight: float = 1.0,
) -> Edge:
    return Edge(
        source_id=_make_node_id(src_source, src_id),
        target_id=_make_node_id(tgt_source, tgt_id),
        edge_type=edge_type,
        weight=weight,
    )


@pytest.fixture
def medium_kg():
    """
    Medium-sized KG for integration testing (larger than unit test KG).

    Structure (10 phenotypes, 6 genes, 5 diseases):
      HP:0000001 ← GeneA → Disease1
      HP:0000002 ← GeneA → Disease1
      HP:0000003 ← GeneB → Disease2
      HP:0000004 ← GeneB → Disease2
      HP:0000005 ← GeneC → Disease3
      HP:0000006 ← GeneC → Disease3
      HP:0000007 ← GeneD → Disease4
      HP:0000008 ← GeneD → Disease4
      HP:0000009 ← GeneE → Disease5
      HP:0010 ← GeneE → Disease5
      GeneF (isolated — connected to Disease3 only, no phenotype edges)
    """
    kg = KnowledgeGraph()

    phenotypes = [f"HP:{i:07d}" for i in range(1, 11)]
    genes = ["GeneA", "GeneB", "GeneC", "GeneD", "GeneE", "GeneF"]
    diseases = [f"MONDO:{i:07d}" for i in range(1, 6)]

    for p in phenotypes:
        kg.add_node(_make_node(DataSource.HPO, p, NodeType.PHENOTYPE, f"Phenotype {p}"))
    for g in genes:
        kg.add_node(_make_node(DataSource.DISGENET, g, NodeType.GENE, g))
    for d in diseases:
        kg.add_node(_make_node(DataSource.MONDO, d, NodeType.DISEASE, f"Disease {d}"))

    # Gene → Phenotype edges (pairs)
    gene_pheno_pairs = [
        ("GeneA", "HP:0000001"), ("GeneA", "HP:0000002"),
        ("GeneB", "HP:0000003"), ("GeneB", "HP:0000004"),
        ("GeneC", "HP:0000005"), ("GeneC", "HP:0000006"),
        ("GeneD", "HP:0000007"), ("GeneD", "HP:0000008"),
        ("GeneE", "HP:0000009"), ("GeneE", "HP:0000010"),
    ]
    for gene, pheno in gene_pheno_pairs:
        kg.add_edge(_make_edge(
            DataSource.DISGENET, gene, DataSource.HPO, pheno,
            EdgeType.GENE_HAS_PHENOTYPE, weight=0.9,
        ))

    # Gene → Disease edges
    gene_disease_pairs = [
        ("GeneA", "MONDO:0000001"), ("GeneB", "MONDO:0000002"),
        ("GeneC", "MONDO:0000003"), ("GeneD", "MONDO:0000004"),
        ("GeneE", "MONDO:0000005"),
        ("GeneF", "MONDO:0000003"),  # GeneF → Disease3 (no phenotype link)
    ]
    for gene, disease in gene_disease_pairs:
        kg.add_edge(_make_edge(
            DataSource.DISGENET, gene, DataSource.MONDO, disease,
            EdgeType.GENE_ASSOCIATED_WITH_DISEASE, weight=0.85,
        ))

    return kg


@pytest.fixture
def gnn_model_and_data(medium_kg):
    """
    Build a small GNN model + graph data matching medium_kg.

    Uses kg.metadata() and kg.export_graph_data() so that edge type names
    match what _load_model_from_checkpoint() will produce when reconstructing
    the model. This ensures checkpoint round-trip works in tests.
    """
    hidden_dim = 32

    # Build graph data using the same code path the production pipeline uses
    torch.manual_seed(42)
    graph_data = medium_kg.export_graph_data(
        output_dir=None, feature_dim=hidden_dim
    )

    # Build model using kg.metadata() — same as _load_model_from_checkpoint
    metadata = medium_kg.metadata()
    in_channels_dict = {
        k: v.shape[1] for k, v in graph_data["x_dict"].items()
    }

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


# =============================================================================
# Test: GNN Pipeline End-to-End
# =============================================================================
class TestGNNPipelineE2E:
    """End-to-end tests for GNN-powered diagnosis pipeline."""

    def test_full_diagnosis_with_gnn(self, medium_kg, gnn_model_and_data):
        """Full pipeline: phenotypes → BFS paths → GNN scoring → ranked results."""
        model, graph_data = gnn_model_and_data
        pipeline = DiagnosisPipeline(
            kg=medium_kg,
            model=model,
            graph_data=graph_data,
        )
        assert pipeline._gnn_ready

        patient = PatientPhenotypes(
            patient_id="e2e_test_001",
            phenotypes=["HP:0000001", "HP:0000002"],  # Should lead to Disease1 via GeneA
        )

        result = pipeline.run(patient, top_k=5, include_explanations=True)

        # Basic sanity
        assert len(result.candidates) > 0
        assert result.inference_time_ms > 0

        # All candidates should have GNN scores
        for c in result.candidates:
            assert c.gnn_score != 0.0
            assert c.confidence_score == c.gnn_score
            assert 0.0 <= c.gnn_score <= 1.0

        # Top candidate should have reasoning paths
        top = result.candidates[0]
        assert top.rank == 1
        assert top.explanation is not None

    def test_different_phenotype_sets_give_different_embeddings(
        self, medium_kg, gnn_model_and_data
    ):
        """Different phenotype inputs should produce different patient embeddings."""
        model, graph_data = gnn_model_and_data
        pipeline = DiagnosisPipeline(
            kg=medium_kg, model=model, graph_data=graph_data,
        )

        node_mapping = pipeline._node_id_to_idx
        pheno_type = NodeType.PHENOTYPE.value
        pheno_emb = pipeline._node_embeddings[pheno_type]

        # Get embedding for phenotype set A (indices 0, 1) vs set B (indices 6, 7)
        pheno_map = node_mapping[pheno_type]
        idx_a = [pheno_map[f"hpo:HP:0000001"], pheno_map[f"hpo:HP:0000002"]]
        idx_b = [pheno_map[f"hpo:HP:0000007"], pheno_map[f"hpo:HP:0000008"]]

        emb_a = pheno_emb[torch.tensor(idx_a)].mean(dim=0)
        emb_b = pheno_emb[torch.tensor(idx_b)].mean(dim=0)

        # Different phenotype sets should have different aggregated embeddings
        assert not torch.allclose(emb_a, emb_b, atol=1e-6), (
            "Different phenotype sets should produce different patient embeddings"
        )

    def test_gnn_scores_reproducible(self, medium_kg, gnn_model_and_data):
        """Same input should produce identical GNN scores (deterministic)."""
        model, graph_data = gnn_model_and_data
        pipeline = DiagnosisPipeline(
            kg=medium_kg, model=model, graph_data=graph_data,
        )

        patient = PatientPhenotypes(
            patient_id="repro", phenotypes=["HP:0000003", "HP:0000004"],
        )

        result1 = pipeline.run(patient, top_k=5, include_explanations=False)
        result2 = pipeline.run(patient, top_k=5, include_explanations=False)

        scores1 = [c.gnn_score for c in result1.candidates]
        scores2 = [c.gnn_score for c in result2.candidates]
        assert scores1 == scores2


# =============================================================================
# Test: Vector Index Build + Load + Pipeline Integration
# =============================================================================
class TestVectorIndexE2E:
    """End-to-end tests for vector index build → load → ANN search in pipeline."""

    @pytest.fixture
    def voyager_available(self):
        try:
            import voyager
            return True
        except ImportError:
            pytest.skip("voyager not installed")

    @pytest.fixture
    def built_index_path(self, medium_kg, gnn_model_and_data, tmp_path, voyager_available):
        """Build a vector index from GNN embeddings and save to tmp_path."""
        from src.retrieval import create_index

        model, graph_data = gnn_model_and_data
        hidden_dim = 32

        # Run GNN forward pass
        with torch.no_grad():
            embeddings = model(graph_data["x_dict"], graph_data["edge_index_dict"])

        disease_emb = embeddings["disease"].cpu().numpy().astype(np.float32)
        node_mapping = medium_kg.get_node_id_mapping()
        disease_mapping = node_mapping.get("disease", {})

        # Build entity_id → embedding dict
        idx_to_id = {idx: nid for nid, idx in disease_mapping.items()}
        entity_embeddings = {
            idx_to_id.get(i, str(i)): disease_emb[i]
            for i in range(disease_emb.shape[0])
        }

        # Create and build index
        index = create_index(backend="voyager", dim=hidden_dim)
        index.build_index(entity_embeddings)

        # Save
        index_base = tmp_path / "test_vector_index"
        disease_path = index_base.parent / f"{index_base.name}_disease"
        index.save(disease_path)

        return str(index_base)

    def test_build_and_load_index(self, built_index_path, voyager_available):
        """Vector index should be saveable and loadable."""
        from src.retrieval import create_index

        disease_path = Path(built_index_path).parent / f"{Path(built_index_path).name}_disease"

        index = create_index(backend="voyager", dim=32)
        index.load(disease_path)

        assert len(index) == 5  # 5 diseases in medium_kg
        assert index.dim == 32

    def test_index_search_returns_results(self, built_index_path, voyager_available):
        """Search should return ranked results."""
        from src.retrieval import create_index

        disease_path = Path(built_index_path).parent / f"{Path(built_index_path).name}_disease"

        index = create_index(backend="voyager", dim=32)
        index.load(disease_path)

        # Random query
        np.random.seed(123)
        query = np.random.randn(32).astype(np.float32)
        results = index.search(query, top_k=3)

        assert len(results) == 3
        for entity_id, distance in results:
            assert isinstance(entity_id, str)
            assert isinstance(distance, float)

    def test_pipeline_with_vector_index(
        self, medium_kg, gnn_model_and_data, built_index_path, voyager_available,
    ):
        """Pipeline should load vector index and report it as ready."""
        model, graph_data = gnn_model_and_data

        config = PipelineConfig(vector_index_path=built_index_path)
        pipeline = DiagnosisPipeline(
            kg=medium_kg,
            model=model,
            graph_data=graph_data,
            config=config,
        )

        assert pipeline._gnn_ready
        assert pipeline._vector_index_ready
        assert pipeline._vector_index is not None

        pipe_config = pipeline.get_pipeline_config()
        assert pipe_config["vector_index_ready"] is True
        assert pipe_config["vector_index_size"] == 5

    def test_full_pipeline_with_ann(
        self, medium_kg, gnn_model_and_data, built_index_path, voyager_available,
    ):
        """Full pipeline with both BFS paths and ANN search."""
        model, graph_data = gnn_model_and_data

        config = PipelineConfig(
            vector_index_path=built_index_path,
            ann_top_k=10,
            ann_score_threshold=0.0,  # Accept all ANN results for testing
        )
        pipeline = DiagnosisPipeline(
            kg=medium_kg,
            model=model,
            graph_data=graph_data,
            config=config,
        )

        patient = PatientPhenotypes(
            patient_id="ann_test",
            phenotypes=["HP:0000001", "HP:0000002"],
        )

        result = pipeline.run(patient, top_k=10, include_explanations=False)

        assert len(result.candidates) > 0
        for c in result.candidates:
            assert c.rank > 0
            assert 0.0 <= c.confidence_score <= 1.0

    def test_ann_candidates_have_correct_structure(
        self, medium_kg, gnn_model_and_data, built_index_path, voyager_available,
    ):
        """ANN-only candidates should still have valid DiagnosisCandidate fields."""
        model, graph_data = gnn_model_and_data

        config = PipelineConfig(
            vector_index_path=built_index_path,
            ann_top_k=50,
            ann_score_threshold=0.0,
        )
        pipeline = DiagnosisPipeline(
            kg=medium_kg,
            model=model,
            graph_data=graph_data,
            config=config,
        )

        patient = PatientPhenotypes(
            patient_id="struct_test",
            phenotypes=["HP:0000005", "HP:0000006"],
        )

        result = pipeline.run(patient, top_k=10, include_explanations=False)

        for c in result.candidates:
            assert c.disease_id is not None
            assert c.disease_name is not None
            assert isinstance(c.supporting_genes, list)
            assert isinstance(c.reasoning_paths, list)
            assert isinstance(c.evidence_sources, list)

    def test_pipeline_without_index_still_works(
        self, medium_kg, gnn_model_and_data,
    ):
        """Pipeline without vector_index_path should work normally (no ANN)."""
        model, graph_data = gnn_model_and_data

        pipeline = DiagnosisPipeline(
            kg=medium_kg,
            model=model,
            graph_data=graph_data,
        )

        assert pipeline._vector_index_ready is False

        patient = PatientPhenotypes(
            patient_id="no_ann",
            phenotypes=["HP:0000001", "HP:0000002"],
        )

        result = pipeline.run(patient, top_k=5, include_explanations=False)
        assert len(result.candidates) > 0


# =============================================================================
# Test: Checkpoint Save → Load → Inference (full bridge)
# =============================================================================
class TestCheckpointBridge:
    """Verify the train → save checkpoint → load → inference chain."""

    def test_checkpoint_roundtrip(self, medium_kg, gnn_model_and_data, tmp_path):
        """
        Train model -> save checkpoint -> load fresh pipeline from disk paths,
        verify GNN scoring works end-to-end (the actual production path).
        """
        import torch.nn.functional as F

        model, graph_data = gnn_model_and_data

        # --- Train a few steps so weights diverge from random init ---
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(5):
            optimizer.zero_grad()
            emb = model(graph_data["x_dict"], graph_data["edge_index_dict"])
            p_emb = emb.get("phenotype")
            d_emb = emb.get("disease")
            if p_emb is None or d_emb is None:
                pytest.skip("Model did not produce phenotype/disease embeddings")
            p = F.normalize(p_emb, dim=-1)
            d = F.normalize(d_emb, dim=-1)
            n = min(p.size(0), d.size(0))
            loss = F.cross_entropy(torch.mm(p[:n], d[:n].t()), torch.arange(n))
            loss.backward()
            optimizer.step()
        model.eval()

        # --- Save checkpoint (Trainer.save_checkpoint format) ---
        ckpt_path = tmp_path / "checkpoint.pt"
        torch.save({
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "config": {
                "hidden_dim": 32,
                "num_layers": 2,
                "num_heads": 4,
                "conv_type": "gat",
                "dropout": 0.0,
            },
        }, ckpt_path)

        # --- Save graph data files ---
        # Save the SAME features+edges the model was trained on so the
        # precomputed embeddings match training. We use export_graph_data() to
        # create the directory + num_nodes.json, then overwrite the tensor
        # files with the fixture's exact data (which is what the model saw).
        data_dir = tmp_path / "graph_data"
        medium_kg.export_graph_data(output_dir=data_dir, feature_dim=32)
        torch.save(graph_data["x_dict"], data_dir / "node_features.pt")
        torch.save(graph_data["edge_index_dict"], data_dir / "edge_indices.pt")

        # --- Load fresh pipeline from disk (production path) ---
        pipeline = DiagnosisPipeline(
            kg=medium_kg,
            checkpoint_path=str(ckpt_path),
            data_dir=str(data_dir),
            device="cpu",
        )

        # GNN must be ready after loading
        assert pipeline._gnn_ready, (
            "GNN should be ready after loading checkpoint + graph data from disk"
        )
        assert pipeline._node_embeddings is not None
        assert "phenotype" in pipeline._node_embeddings
        assert "disease" in pipeline._node_embeddings

        # --- Run inference and verify GNN-primary scoring ---
        patient = PatientPhenotypes(
            patient_id="bridge_test",
            phenotypes=["HP:0000001", "HP:0000002"],
        )

        result = pipeline.run(patient, top_k=5, include_explanations=False)
        assert len(result.candidates) > 0, "Pipeline should produce candidates"

        for c in result.candidates:
            # GNN-primary: confidence == gnn_score (not weighted combo)
            assert abs(c.confidence_score - c.gnn_score) < 1e-5, (
                f"GNN-primary: confidence ({c.confidence_score}) "
                f"should equal gnn_score ({c.gnn_score})"
            )
            # Scores should be in valid range
            assert 0.0 <= c.gnn_score <= 1.0

        # At least one candidate should have non-zero GNN score (model is trained)
        assert any(c.gnn_score > 0.0 for c in result.candidates), (
            "Trained model should produce non-zero GNN scores"
        )

        # --- Reproducibility: same input → same scores ---
        result2 = pipeline.run(patient, top_k=5, include_explanations=False)
        scores1 = [(c.disease_id, c.gnn_score) for c in result.candidates]
        scores2 = [(c.disease_id, c.gnn_score) for c in result2.candidates]
        assert scores1 == scores2, "Same input should produce identical scores"

    def test_export_graph_data_matches_kg_structure(self, medium_kg, tmp_path):
        """KG.export_graph_data() should produce files matching KG node/edge counts."""
        data_dir = tmp_path / "export_test"
        graph_data = medium_kg.export_graph_data(output_dir=data_dir, feature_dim=64)

        # Check returned dict
        assert "phenotype" in graph_data["x_dict"]
        assert "gene" in graph_data["x_dict"]
        assert "disease" in graph_data["x_dict"]
        assert graph_data["num_nodes_dict"]["phenotype"] == 10
        assert graph_data["num_nodes_dict"]["gene"] == 6
        assert graph_data["num_nodes_dict"]["disease"] == 5
        assert graph_data["x_dict"]["phenotype"].shape == (10, 64)

        # Check files on disk
        assert (data_dir / "node_features.pt").exists()
        assert (data_dir / "edge_indices.pt").exists()
        assert (data_dir / "num_nodes.json").exists()

        # Edge indices should include reverse edges
        edge_types = list(graph_data["edge_index_dict"].keys())
        rev_count = sum(1 for et in edge_types if et[1].startswith("rev_"))
        fwd_count = len(edge_types) - rev_count
        assert rev_count == fwd_count, "Every forward edge type should have a reverse"
