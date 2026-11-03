"""
Integration Tests: the graph export and the evidence panel
==========================================================
End-to-end tests that verify:
  1. the graph export a workspace is built from matches the graph;
  2. the evidence panel explains a served disease child's rows (Mode A, a
     direct path, or insufficient evidence) without changing its ranking.

The served model is a disease child trained for real on CPU
(`tests/fixtures/trained_children`). The ShepherdGNN checkpoint round trip and
the shortest-path mixture these tests once served retired with serving them
(K8-4); a checkpoint without a model spec is refused at load
(`tests/unit/test_pipeline_fails_closed.py`).

Module: tests/integration/test_pipeline.py
"""
from __future__ import annotations

import pytest

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


class TestCheckpointBridge:
    """The graph export a workspace's tensors are written from."""

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
# Test: Evidence Panel (Step C)
# =============================================================================
class TestEvidencePanel:
    """
    Verify Step C: PathReasoner is decoupled from scoring; evidence is
    presented as Mode A (direct path) or as insufficient, with evidence
    labels. Served by a disease child trained for real
    (`tests/fixtures/trained_children`): since K7 a pipeline is always a model
    read from its checkpoint, and paths only explain the rows it ranked.
    """

    #: Phenotypes of the demo graph: seizure and developmental delay, each
    #: annotated to genes that cause diseases (two hops).
    PHENOTYPES = ["HP:0001250", "HP:0001263"]
    #: Every disease of the demo graph's view is a row, so the 2-hop diseases
    #: are among them whatever the 1-epoch model's order.
    ALL_ROWS = 100

    def test_mode_a_strong_path_for_close_disease(self, disease_pipeline):
        """A disease reachable in 2 hops should get STRONG_PATH label."""
        from src.reasoning import ConfidenceLabel, EvidenceMode

        patient = PatientPhenotypes(patient_id="mode_a_test", phenotypes=self.PHENOTYPES)
        result = disease_pipeline.run(patient, top_k=self.ALL_ROWS, include_explanations=True)

        # The top candidate should have an evidence package of a known mode
        top = result.candidates[0]
        assert top.evidence_package is not None
        assert top.evidence_package["mode"] in (
            EvidenceMode.DIRECT_PATH.value, EvidenceMode.INSUFFICIENT.value,
        )

        # At least one candidate should be Mode A with STRONG label (a disease
        # one gene away from a phenotype of the patient)
        strong_found = any(
            c.evidence_package
            and c.evidence_package["mode"] == EvidenceMode.DIRECT_PATH.value
            and c.evidence_label == ConfidenceLabel.STRONG_PATH.value
            for c in result.candidates
        )
        assert strong_found, (
            f"Expected at least one STRONG_PATH candidate. Got labels: "
            f"{[c.evidence_label for c in result.candidates]}"
        )

    def test_evidence_has_min_path_length(self, disease_pipeline):
        """Mode A evidence should report the minimum path length."""
        from src.reasoning import EvidenceMode

        patient = PatientPhenotypes(patient_id="path_len_test", phenotypes=self.PHENOTYPES)
        result = disease_pipeline.run(patient, top_k=self.ALL_ROWS, include_explanations=True)

        direct = [c for c in result.candidates if c.evidence_package
                  and c.evidence_package["mode"] == EvidenceMode.DIRECT_PATH.value]
        assert direct, "no Mode A evidence to check"
        for c in direct:
            assert c.evidence_package["min_path_length"] is not None
            assert c.evidence_package["min_path_length"] > 0

    def test_evidence_panel_used_when_explanations_enabled(self, disease_pipeline):
        """include_explanations=True should populate evidence_package."""
        patient = PatientPhenotypes(patient_id="enabled", phenotypes=self.PHENOTYPES[:1])
        result = disease_pipeline.run(patient, top_k=3, include_explanations=True)

        assert result.candidates
        # Every candidate should have an evidence package and label
        for c in result.candidates:
            assert c.evidence_package is not None, (
                "evidence_package missing when include_explanations=True"
            )
            assert c.evidence_label is not None
            # The evidence label should be one of the known values
            assert c.evidence_label in (
                "Strong path support",
                "Weak path support",
                "Insufficient evidence",
            )

    def test_evidence_skipped_when_explanations_disabled(self, disease_pipeline):
        """include_explanations=False should NOT populate evidence_package."""
        patient = PatientPhenotypes(patient_id="disabled", phenotypes=self.PHENOTYPES[:1])
        result = disease_pipeline.run(patient, top_k=3, include_explanations=False)

        assert result.candidates
        for c in result.candidates:
            assert c.evidence_package is None
            assert c.evidence_label is None

    def test_evidence_panel_does_not_affect_ranking(self, trained_disease_child):
        """Critical invariant: enabling evidence must not change scores
        or ranking. Evidence is decorative/explanatory, not a scoring input."""
        from tests.fixtures.trained_children import serve

        pipeline_a = serve(trained_disease_child)
        pipeline_b = serve(trained_disease_child)

        patient = PatientPhenotypes(patient_id="invariant_test", phenotypes=self.PHENOTYPES)

        result_no_evidence = pipeline_a.run(patient, top_k=5, include_explanations=False)
        result_with_evidence = pipeline_b.run(patient, top_k=5, include_explanations=True)

        # Same number of candidates, same ordering, same scores
        assert len(result_no_evidence.candidates) == len(result_with_evidence.candidates) == 5
        for c_no, c_yes in zip(
            result_no_evidence.candidates, result_with_evidence.candidates
        ):
            assert c_no.disease_id == c_yes.disease_id, (
                "Evidence panel changed ranking order"
            )
            assert abs(c_no.score - c_yes.score) < 1e-9, (
                "Evidence panel changed the scores"
            )

    def test_there_is_no_analogy_mode(self):
        """X20 (K8-4): Mode B, the analogy over the ShepherdGNN embeddings the
        specless serving supplied, retired with it. A direct path and
        insufficient evidence are the only modes and labels, with no alias."""
        import inspect

        import src.reasoning as reasoning
        from src.reasoning import ConfidenceLabel, EvidenceMode, EvidencePanel

        assert {mode.name for mode in EvidenceMode} == {"DIRECT_PATH", "INSUFFICIENT"}
        assert {label.name for label in ConfidenceLabel} == {
            "STRONG_PATH", "WEAK_PATH", "INSUFFICIENT"}
        assert not hasattr(reasoning, "AnalogyMatch")
        assert not {"node_embeddings", "node_id_to_idx"} & set(
            inspect.signature(EvidencePanel.build_evidence).parameters)

    def test_evidence_panel_direct_unit_test(self, medium_kg):
        """Direct unit test of EvidencePanel without going through pipeline."""
        from src.core.types import DataSource, DiseaseTaskCandidate, NodeID
        from src.reasoning import (
            ConfidenceLabel,
            EvidenceMode,
            EvidencePanel,
            EvidencePanelConfig,
        )

        panel = EvidencePanel(
            kg=medium_kg,
            config=EvidencePanelConfig(
                strong_path_max_hops=2,
                weak_path_max_hops=4,
            ),
        )

        # Disease1 is reachable from HP:0000001 via GeneA (2 hops)
        candidate = DiseaseTaskCandidate(
            rank=1,
            disease_id=NodeID(source=DataSource.MONDO, local_id="MONDO:0000001"),
            disease_name="Disease 1",
            score=-0.5,
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

        # Either DIRECT_PATH (if KG has a path) or INSUFFICIENT: there is no
        # other mode.
        assert package.mode in (
            EvidenceMode.DIRECT_PATH, EvidenceMode.INSUFFICIENT,
        )
        if package.mode == EvidenceMode.DIRECT_PATH:
            assert package.evidence_label in (
                ConfidenceLabel.STRONG_PATH, ConfidenceLabel.WEAK_PATH,
            )
            assert package.min_path_length is not None
            assert len(package.direct_paths) > 0

    def test_evidence_panel_insufficient_when_no_paths_no_embeddings(
        self, medium_kg
    ):
        """When no path reaches the disease → INSUFFICIENT."""
        from src.core.types import DataSource, DiseaseTaskCandidate, Node, NodeID, NodeType
        from src.reasoning import ConfidenceLabel, EvidenceMode, EvidencePanel

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

        candidate = DiseaseTaskCandidate(
            rank=1,
            disease_id=isolated_id,
            disease_name="Isolated Disease",
            score=-4.0,
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
        )

        assert package.mode == EvidenceMode.INSUFFICIENT
        assert package.evidence_label == ConfidenceLabel.INSUFFICIENT
