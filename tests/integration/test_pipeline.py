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

    CRITICAL: This fixture mirrors the production training path
    (scripts/train_model.py:create_model_from_config) which derives metadata
    from graph_data["edge_index_dict"].keys() — INCLUDING reverse rev_* edges
    for bidirectional message passing. Do NOT switch to kg.metadata() (forward
    edges only) — that would diverge from the trainer and let checkpoint
    round-trip bugs slip through CI.

    The matching change is in pipeline._load_model_from_checkpoint, which
    must also derive metadata from graph_data, not kg.metadata().
    """
    hidden_dim = 32

    # Build graph data using the same code path the production pipeline uses
    torch.manual_seed(42)
    graph_data = medium_kg.export_graph_data(
        output_dir=None, feature_dim=hidden_dim
    )

    # Build model metadata from graph_data — matches scripts/train_model.py
    node_types = list(graph_data["num_nodes_dict"].keys())
    edge_types = list(graph_data["edge_index_dict"].keys())
    metadata = (node_types, edge_types)
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


# =============================================================================
# Test: Shortest Path Integration (Step B)
# =============================================================================
class TestShortestPathIntegration:
    """
    Verify the original SHEPHERD scoring formula:
        confidence = eta * embedding_sim + (1 - eta) * SP_sim

    The shortest path lookup table is generated by running the BFS pre-compute
    logic from scripts/compute_shortest_paths.py against medium_kg.
    """

    def _build_sp_lookup(self, medium_kg, data_dir, max_hops=5):
        """Generate shortest_paths.pt for medium_kg via the production script."""
        # Import the precompute logic from the script
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "compute_shortest_paths",
            Path(__file__).parent.parent.parent / "scripts" / "compute_shortest_paths.py",
        )
        sp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sp_module)

        sp_data = sp_module.compute_shortest_paths(medium_kg, max_hops=max_hops)
        meta = {
            "max_hops": max_hops,
            "num_pairs": int(sp_data["distance"].numel()),
        }
        sp_module.save_shortest_paths(
            sp_data, data_dir / "shortest_paths.pt", meta
        )

    def test_sp_table_loads_when_present(
        self, medium_kg, gnn_model_and_data, tmp_path
    ):
        """Pipeline should detect and load shortest_paths.pt automatically."""
        model, graph_data = gnn_model_and_data

        # Set up data_dir with both graph data and SP table
        data_dir = tmp_path / "data"
        medium_kg.export_graph_data(output_dir=data_dir, feature_dim=32)
        torch.save(graph_data["x_dict"], data_dir / "node_features.pt")
        torch.save(graph_data["edge_index_dict"], data_dir / "edge_indices.pt")
        self._build_sp_lookup(medium_kg, data_dir, max_hops=5)

        # Save checkpoint and load via disk path so _load_shortest_paths runs
        ckpt_path = tmp_path / "ckpt.pt"
        torch.save({
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "config": {"hidden_dim": 32, "num_layers": 2, "num_heads": 4, "conv_type": "gat"},
        }, ckpt_path)

        pipeline = DiagnosisPipeline(
            kg=medium_kg,
            checkpoint_path=str(ckpt_path),
            data_dir=str(data_dir),
            device="cpu",
        )
        assert pipeline._gnn_ready
        assert pipeline._sp_ready
        assert pipeline._sp_index is not None
        assert len(pipeline._sp_index) > 0

    def test_sp_optional_fallback_when_missing(
        self, medium_kg, gnn_model_and_data, tmp_path
    ):
        """When shortest_paths.pt is absent and sp_optional=True, pipeline
        falls back to pure GNN scoring (effective eta=1.0)."""
        model, graph_data = gnn_model_and_data

        data_dir = tmp_path / "data"
        medium_kg.export_graph_data(output_dir=data_dir, feature_dim=32)
        torch.save(graph_data["x_dict"], data_dir / "node_features.pt")
        torch.save(graph_data["edge_index_dict"], data_dir / "edge_indices.pt")
        # Note: NO shortest_paths.pt

        ckpt_path = tmp_path / "ckpt.pt"
        torch.save({
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "config": {"hidden_dim": 32, "num_layers": 2, "num_heads": 4, "conv_type": "gat"},
        }, ckpt_path)

        pipeline = DiagnosisPipeline(
            kg=medium_kg,
            checkpoint_path=str(ckpt_path),
            data_dir=str(data_dir),
            device="cpu",
        )
        assert pipeline._gnn_ready
        assert not pipeline._sp_ready

        cfg = pipeline.get_pipeline_config()
        assert cfg["scoring_mode"] == "gnn_only"
        assert cfg["eta_effective"] == 1.0

    def test_combined_score_matches_formula(
        self, medium_kg, gnn_model_and_data, tmp_path
    ):
        """Verify confidence = eta * gnn_score + (1-eta) * sp_score
        when both signals are available."""
        model, graph_data = gnn_model_and_data

        data_dir = tmp_path / "data"
        medium_kg.export_graph_data(output_dir=data_dir, feature_dim=32)
        torch.save(graph_data["x_dict"], data_dir / "node_features.pt")
        torch.save(graph_data["edge_index_dict"], data_dir / "edge_indices.pt")
        self._build_sp_lookup(medium_kg, data_dir, max_hops=5)

        ckpt_path = tmp_path / "ckpt.pt"
        torch.save({
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "config": {"hidden_dim": 32, "num_layers": 2, "num_heads": 4, "conv_type": "gat"},
        }, ckpt_path)

        eta_value = 0.6
        config = PipelineConfig(eta=eta_value)
        pipeline = DiagnosisPipeline(
            kg=medium_kg,
            checkpoint_path=str(ckpt_path),
            data_dir=str(data_dir),
            config=config,
            device="cpu",
        )
        assert pipeline._sp_ready

        patient = PatientPhenotypes(
            patient_id="sp_test",
            phenotypes=["HP:0000001", "HP:0000002"],
        )
        result = pipeline.run(patient, top_k=5, include_explanations=False)
        assert len(result.candidates) > 0

        for c in result.candidates:
            expected = eta_value * c.gnn_score + (1.0 - eta_value) * c.sp_score
            assert abs(c.confidence_score - expected) < 1e-5, (
                f"confidence {c.confidence_score:.6f} != "
                f"eta*gnn + (1-eta)*sp = {expected:.6f} "
                f"(gnn={c.gnn_score}, sp={c.sp_score}, eta={eta_value})"
            )

    def test_eta_one_equals_pure_gnn(
        self, medium_kg, gnn_model_and_data, tmp_path
    ):
        """eta=1.0 with SP loaded should produce same confidence as gnn_score."""
        model, graph_data = gnn_model_and_data

        data_dir = tmp_path / "data"
        medium_kg.export_graph_data(output_dir=data_dir, feature_dim=32)
        torch.save(graph_data["x_dict"], data_dir / "node_features.pt")
        torch.save(graph_data["edge_index_dict"], data_dir / "edge_indices.pt")
        self._build_sp_lookup(medium_kg, data_dir, max_hops=5)

        ckpt_path = tmp_path / "ckpt.pt"
        torch.save({
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "config": {"hidden_dim": 32, "num_layers": 2, "num_heads": 4, "conv_type": "gat"},
        }, ckpt_path)

        pipeline = DiagnosisPipeline(
            kg=medium_kg,
            checkpoint_path=str(ckpt_path),
            data_dir=str(data_dir),
            config=PipelineConfig(eta=1.0),
            device="cpu",
        )

        patient = PatientPhenotypes(
            patient_id="pure_gnn",
            phenotypes=["HP:0000001", "HP:0000002"],
        )
        result = pipeline.run(patient, top_k=5, include_explanations=False)

        for c in result.candidates:
            assert abs(c.confidence_score - c.gnn_score) < 1e-5

    def test_eta_zero_equals_pure_sp(
        self, medium_kg, gnn_model_and_data, tmp_path
    ):
        """eta=0.0 with SP loaded should produce confidence == sp_score."""
        model, graph_data = gnn_model_and_data

        data_dir = tmp_path / "data"
        medium_kg.export_graph_data(output_dir=data_dir, feature_dim=32)
        torch.save(graph_data["x_dict"], data_dir / "node_features.pt")
        torch.save(graph_data["edge_index_dict"], data_dir / "edge_indices.pt")
        self._build_sp_lookup(medium_kg, data_dir, max_hops=5)

        ckpt_path = tmp_path / "ckpt.pt"
        torch.save({
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "config": {"hidden_dim": 32, "num_layers": 2, "num_heads": 4, "conv_type": "gat"},
        }, ckpt_path)

        pipeline = DiagnosisPipeline(
            kg=medium_kg,
            checkpoint_path=str(ckpt_path),
            data_dir=str(data_dir),
            config=PipelineConfig(eta=0.0),
            device="cpu",
        )

        patient = PatientPhenotypes(
            patient_id="pure_sp",
            phenotypes=["HP:0000001", "HP:0000002"],
        )
        result = pipeline.run(patient, top_k=5, include_explanations=False)

        for c in result.candidates:
            assert abs(c.confidence_score - c.sp_score) < 1e-5
            # SP score should be in [0, 1]
            assert 0.0 <= c.sp_score <= 1.0

    def test_sp_score_higher_for_closer_phenotypes(self, medium_kg):
        """Direct test of _calculate_sp_score: closer phenotypes should
        produce higher SP similarity than farther ones."""
        # Pipeline with no model (just enough state for SP lookup)
        pipeline = DiagnosisPipeline(kg=medium_kg)
        pipeline._node_id_to_idx = medium_kg.get_node_id_mapping()

        # Build SP table directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "compute_shortest_paths",
            Path(__file__).parent.parent.parent / "scripts" / "compute_shortest_paths.py",
        )
        sp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sp_module)
        sp_data = sp_module.compute_shortest_paths(medium_kg, max_hops=5)

        # Manually populate the SP state
        ph_arr = sp_data["phenotype_idx"].tolist()
        tg_arr = sp_data["target_idx"].tolist()
        ty_arr = sp_data["target_type"].tolist()
        di_arr = sp_data["distance"].tolist()
        pipeline._sp_index = {
            (ph_arr[i], tg_arr[i], ty_arr[i]): di_arr[i]
            for i in range(len(ph_arr))
        }
        pipeline._sp_lookup = sp_data
        pipeline._sp_max_hops = 5
        pipeline._sp_ready = True

        # Patient with phenotypes directly connected to GeneA→Disease1
        from src.core.types import NodeID, DataSource

        ph_a1 = NodeID(source=DataSource.HPO, local_id="HP:0000001")
        ph_a2 = NodeID(source=DataSource.HPO, local_id="HP:0000002")
        disease1 = NodeID(source=DataSource.MONDO, local_id="MONDO:0000001")
        disease5 = NodeID(source=DataSource.MONDO, local_id="MONDO:0000005")

        sp_close = pipeline._calculate_sp_score(
            [ph_a1, ph_a2], disease1, target_type_idx=1
        )
        sp_far = pipeline._calculate_sp_score(
            [ph_a1, ph_a2], disease5, target_type_idx=1
        )

        # disease1 is the canonical target for HP:0000001/2 (via GeneA)
        # disease5 is on the other side of the graph (via GeneE)
        assert sp_close > sp_far, (
            f"Closer disease should have higher SP score: "
            f"close={sp_close:.4f}, far={sp_far:.4f}"
        )
        assert 0.0 <= sp_close <= 1.0


# =============================================================================
# Test: Evidence Panel (Step C)
# =============================================================================
class TestEvidencePanel:
    """
    Verify Step C: PathReasoner is decoupled from scoring; evidence is
    presented as Mode A (direct path) or Mode B (analogy-based) with
    confidence labels.
    """

    def test_mode_a_strong_path_for_close_disease(
        self, medium_kg, gnn_model_and_data
    ):
        """A disease reachable in 2 hops should get STRONG_PATH label."""
        from src.reasoning import (
            EvidencePanel, ConfidenceLabel, EvidenceMode,
        )
        from src.core.types import NodeID, DataSource

        model, graph_data = gnn_model_and_data
        pipeline = DiagnosisPipeline(
            kg=medium_kg, model=model, graph_data=graph_data,
        )

        patient = PatientPhenotypes(
            patient_id="mode_a_test",
            phenotypes=["HP:0000001", "HP:0000002"],  # GeneA → Disease1 (2 hops)
        )
        result = pipeline.run(patient, top_k=5, include_explanations=True)

        # The top candidate should have a Mode A evidence package
        top = result.candidates[0]
        assert top.evidence_package is not None
        assert top.evidence_package["mode"] in (
            EvidenceMode.DIRECT_PATH.value, EvidenceMode.ANALOGY_BASED.value
        )

        # At least one candidate should be Mode A with STRONG label
        # (the one reachable via GeneA in 2 hops)
        strong_found = any(
            c.evidence_package
            and c.evidence_package["mode"] == EvidenceMode.DIRECT_PATH.value
            and c.confidence_label == ConfidenceLabel.STRONG_PATH.value
            for c in result.candidates
        )
        assert strong_found, (
            f"Expected at least one STRONG_PATH candidate. Got labels: "
            f"{[c.confidence_label for c in result.candidates]}"
        )

    def test_evidence_has_min_path_length(self, medium_kg, gnn_model_and_data):
        """Mode A evidence should report the minimum path length."""
        from src.reasoning import EvidenceMode

        model, graph_data = gnn_model_and_data
        pipeline = DiagnosisPipeline(
            kg=medium_kg, model=model, graph_data=graph_data,
        )

        patient = PatientPhenotypes(
            patient_id="path_len_test",
            phenotypes=["HP:0000001", "HP:0000002"],
        )
        result = pipeline.run(patient, top_k=5, include_explanations=True)

        for c in result.candidates:
            if (
                c.evidence_package
                and c.evidence_package["mode"] == EvidenceMode.DIRECT_PATH.value
            ):
                assert c.evidence_package["min_path_length"] is not None
                assert c.evidence_package["min_path_length"] > 0

    def test_evidence_panel_used_when_explanations_enabled(
        self, medium_kg, gnn_model_and_data
    ):
        """include_explanations=True should populate evidence_package."""
        model, graph_data = gnn_model_and_data
        pipeline = DiagnosisPipeline(
            kg=medium_kg, model=model, graph_data=graph_data,
        )

        patient = PatientPhenotypes(
            patient_id="enabled",
            phenotypes=["HP:0000001"],
        )
        result = pipeline.run(patient, top_k=3, include_explanations=True)

        # Every candidate should have an evidence package and label
        for c in result.candidates:
            assert c.evidence_package is not None, (
                "evidence_package missing when include_explanations=True"
            )
            assert c.confidence_label is not None
            # Confidence label should be one of the known values
            assert c.confidence_label in (
                "Strong path support",
                "Weak path support",
                "Analogy-based (no direct KG path)",
                "Insufficient evidence",
            )

    def test_evidence_skipped_when_explanations_disabled(
        self, medium_kg, gnn_model_and_data
    ):
        """include_explanations=False should NOT populate evidence_package."""
        model, graph_data = gnn_model_and_data
        pipeline = DiagnosisPipeline(
            kg=medium_kg, model=model, graph_data=graph_data,
        )

        patient = PatientPhenotypes(
            patient_id="disabled",
            phenotypes=["HP:0000001"],
        )
        result = pipeline.run(patient, top_k=3, include_explanations=False)

        for c in result.candidates:
            assert c.evidence_package is None
            assert c.confidence_label is None

    def test_evidence_panel_does_not_affect_ranking(
        self, medium_kg, gnn_model_and_data
    ):
        """Critical invariant: enabling evidence must not change scores
        or ranking. Evidence is decorative/explanatory, not a scoring input."""
        model, graph_data = gnn_model_and_data

        pipeline_a = DiagnosisPipeline(
            kg=medium_kg, model=model, graph_data=graph_data,
        )
        pipeline_b = DiagnosisPipeline(
            kg=medium_kg, model=model, graph_data=graph_data,
        )

        patient = PatientPhenotypes(
            patient_id="invariant_test",
            phenotypes=["HP:0000003", "HP:0000004"],
        )

        result_no_evidence = pipeline_a.run(
            patient, top_k=5, include_explanations=False
        )
        result_with_evidence = pipeline_b.run(
            patient, top_k=5, include_explanations=True
        )

        # Same number of candidates, same ordering, same scores
        assert len(result_no_evidence.candidates) == len(
            result_with_evidence.candidates
        )
        for c_no, c_yes in zip(
            result_no_evidence.candidates, result_with_evidence.candidates
        ):
            assert c_no.disease_id == c_yes.disease_id, (
                "Evidence panel changed ranking order"
            )
            assert abs(c_no.confidence_score - c_yes.confidence_score) < 1e-9, (
                "Evidence panel changed confidence scores"
            )

    def test_evidence_panel_direct_unit_test(self, medium_kg):
        """Direct unit test of EvidencePanel without going through pipeline."""
        from src.reasoning import (
            EvidencePanel, EvidencePanelConfig, EvidenceMode, ConfidenceLabel,
        )
        from src.core.types import (
            NodeID, DataSource, DiagnosisCandidate,
        )

        panel = EvidencePanel(
            kg=medium_kg,
            config=EvidencePanelConfig(
                strong_path_max_hops=2,
                weak_path_max_hops=4,
            ),
        )

        # Disease1 is reachable from HP:0000001 via GeneA (2 hops)
        candidate = DiagnosisCandidate(
            rank=1,
            disease_id=NodeID(source=DataSource.MONDO, local_id="MONDO:0000001"),
            disease_name="Disease 1",
            confidence_score=0.9,
            gnn_score=0.9,
            reasoning_score=0.5,
        )
        patient = PatientPhenotypes(
            patient_id="unit",
            phenotypes=["HP:0000001"],
        )
        source_ids = [NodeID(source=DataSource.HPO, local_id="HP:0000001")]

        package = panel.build_evidence(
            candidate=candidate,
            patient_input=patient,
            source_ids=source_ids,
            existing_paths=None,  # force fresh path search
        )

        # Without GNN embeddings, Mode B is unavailable. Result must be
        # either DIRECT_PATH (if KG has a path) or INSUFFICIENT.
        assert package.mode in (
            EvidenceMode.DIRECT_PATH, EvidenceMode.INSUFFICIENT,
        )
        if package.mode == EvidenceMode.DIRECT_PATH:
            assert package.confidence_label in (
                ConfidenceLabel.STRONG_PATH, ConfidenceLabel.WEAK_PATH,
            )
            assert package.min_path_length is not None
            assert len(package.direct_paths) > 0

    def test_evidence_panel_insufficient_when_no_paths_no_embeddings(
        self, medium_kg
    ):
        """When neither paths nor embeddings are available → INSUFFICIENT."""
        from src.reasoning import EvidencePanel, EvidenceMode, ConfidenceLabel
        from src.core.types import (
            NodeID, DataSource, DiagnosisCandidate, NodeType, Node,
        )

        # Build a candidate for a disease the patient phenotype CANNOT
        # reach by adding an isolated disease to the KG.
        kg = medium_kg
        isolated_id = NodeID(source=DataSource.MONDO, local_id="MONDO:9999999")
        kg.add_node(Node(
            id=isolated_id,
            node_type=NodeType.DISEASE,
            name="Isolated Disease",
            data_sources={DataSource.MONDO},
        ))

        panel = EvidencePanel(kg=kg)

        candidate = DiagnosisCandidate(
            rank=1,
            disease_id=isolated_id,
            disease_name="Isolated Disease",
            confidence_score=0.5,
            gnn_score=0.5,
            reasoning_score=0.0,
        )
        patient = PatientPhenotypes(
            patient_id="iso",
            phenotypes=["HP:0000001"],
        )
        source_ids = [NodeID(source=DataSource.HPO, local_id="HP:0000001")]

        package = panel.build_evidence(
            candidate=candidate,
            patient_input=patient,
            source_ids=source_ids,
            existing_paths=None,
            node_embeddings=None,  # no GNN → no Mode B
            node_id_to_idx=None,
        )

        assert package.mode == EvidenceMode.INSUFFICIENT
        assert package.confidence_label == ConfidenceLabel.INSUFFICIENT
        assert 0.0 <= sp_far <= 1.0
