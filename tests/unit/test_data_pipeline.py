"""
Unit Tests for Data Pipeline
==============================
Tests for HPOAnnotationParser, generate_training_samples,
and build_knowledge_graph validation logic.
"""
import json
import pytest
import tempfile
from pathlib import Path

from src.core.types import (
    DataSource,
    Edge,
    EdgeType,
    Node,
    NodeID,
    NodeType,
)
from src.kg.graph import KnowledgeGraph
from src.kg.sample_generator import generate_training_samples
from src.data_sources.hpo_annotations import HPOAnnotationParser


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def demo_kg():
    """Small KG with known structure for testing sample generation."""
    kg = KnowledgeGraph()

    # 3 phenotypes
    for hp_id, name in [
        ("HP:0001250", "Seizure"),
        ("HP:0001263", "Global developmental delay"),
        ("HP:0001252", "Hypotonia"),
    ]:
        kg.add_node(Node(
            id=NodeID(source=DataSource.HPO, local_id=hp_id),
            node_type=NodeType.PHENOTYPE,
            name=name,
        ))

    # 2 genes
    for gene_id in ["SCN1A", "MECP2"]:
        kg.add_node(Node(
            id=NodeID(source=DataSource.DISGENET, local_id=gene_id),
            node_type=NodeType.GENE,
            name=gene_id,
        ))

    # 2 diseases
    for mondo_id, name in [
        ("MONDO:0011073", "Dravet syndrome"),
        ("MONDO:0010582", "Rett syndrome"),
    ]:
        kg.add_node(Node(
            id=NodeID(source=DataSource.MONDO, local_id=mondo_id),
            node_type=NodeType.DISEASE,
            name=name,
        ))

    # Gene-disease edges
    kg.add_edge(Edge(
        source_id=NodeID(source=DataSource.DISGENET, local_id="SCN1A"),
        target_id=NodeID(source=DataSource.MONDO, local_id="MONDO:0011073"),
        edge_type=EdgeType.GENE_ASSOCIATED_WITH_DISEASE,
        weight=0.95,
    ))
    kg.add_edge(Edge(
        source_id=NodeID(source=DataSource.DISGENET, local_id="MECP2"),
        target_id=NodeID(source=DataSource.MONDO, local_id="MONDO:0010582"),
        edge_type=EdgeType.GENE_ASSOCIATED_WITH_DISEASE,
        weight=0.95,
    ))

    # Gene-phenotype edges
    kg.add_edge(Edge(
        source_id=NodeID(source=DataSource.DISGENET, local_id="SCN1A"),
        target_id=NodeID(source=DataSource.HPO, local_id="HP:0001250"),
        edge_type=EdgeType.GENE_HAS_PHENOTYPE,
        weight=0.9,
    ))
    kg.add_edge(Edge(
        source_id=NodeID(source=DataSource.DISGENET, local_id="SCN1A"),
        target_id=NodeID(source=DataSource.HPO, local_id="HP:0001263"),
        edge_type=EdgeType.GENE_HAS_PHENOTYPE,
        weight=0.8,
    ))
    kg.add_edge(Edge(
        source_id=NodeID(source=DataSource.DISGENET, local_id="MECP2"),
        target_id=NodeID(source=DataSource.HPO, local_id="HP:0001263"),
        edge_type=EdgeType.GENE_HAS_PHENOTYPE,
        weight=0.9,
    ))
    kg.add_edge(Edge(
        source_id=NodeID(source=DataSource.DISGENET, local_id="MECP2"),
        target_id=NodeID(source=DataSource.HPO, local_id="HP:0001252"),
        edge_type=EdgeType.GENE_HAS_PHENOTYPE,
        weight=0.85,
    ))

    # Phenotype-disease edges
    kg.add_edge(Edge(
        source_id=NodeID(source=DataSource.HPO, local_id="HP:0001250"),
        target_id=NodeID(source=DataSource.MONDO, local_id="MONDO:0011073"),
        edge_type=EdgeType.PHENOTYPE_OF_DISEASE,
        weight=0.8,
    ))

    return kg


@pytest.fixture
def phenotype_hpoa_file(tmp_dir):
    """Create a minimal phenotype.hpoa test fixture."""
    content = """\
#description: HPO annotations test fixture
#date: 2026-01-01
#tracker: n/a
database_id	disease_name	qualifier	hpo_id	reference	evidence	onset	frequency	sex	modifier	aspect	biocuration
MONDO:0011073	Dravet syndrome		HP:0001250	PMID:123	PCS		HP:0040281			P	HPO:test[2026-01-01]
MONDO:0011073	Dravet syndrome		HP:0001263	PMID:123	PCS		HP:0040282			P	HPO:test[2026-01-01]
MONDO:0011073	Dravet syndrome	NOT	HP:0000256	PMID:456	PCS					P	HPO:test[2026-01-01]
OMIM:312750	Rett syndrome		HP:0001252	PMID:789	PCS		50%			P	HPO:test[2026-01-01]
ORPHA:99999	Orphanet disease		HP:0001250	PMID:000	PCS					P	HPO:test[2026-01-01]
"""
    path = tmp_dir / "phenotype.hpoa"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def genes_to_phenotype_file(tmp_dir):
    """Create a minimal genes_to_phenotype.txt test fixture."""
    content = """\
gene_id	gene_symbol	hpo_id	hpo_name	frequency	disease_id
2565	SCN1A	HP:0001250	Seizure	-	MONDO:0011073
2565	SCN1A	HP:0001263	Global developmental delay	-	MONDO:0011073
4204	MECP2	HP:0001252	Hypotonia	-	OMIM:312750
4204	MECP2	HP:0001263	Global developmental delay	-	OMIM:312750
999999	FAKEGENE	HP:0001250	Seizure	-	OMIM:999999
"""
    path = tmp_dir / "genes_to_phenotype.txt"
    path.write_text(content, encoding="utf-8")
    return path


# =============================================================================
# HPOAnnotationParser Tests
# =============================================================================
class TestHPOAnnotationParser:
    """Tests for HPOAnnotationParser"""

    def test_parse_phenotype_hpoa_basic(self, phenotype_hpoa_file):
        """Parse phenotype.hpoa and get phenotype-disease annotations."""
        parser = HPOAnnotationParser()
        annotations = parser.parse_phenotype_hpoa(phenotype_hpoa_file)

        # MONDO:0011073 rows should pass (2 valid, 1 NOT filtered)
        mondo_annots = [a for a in annotations if a["disease_id"] == "MONDO:0011073"]
        assert len(mondo_annots) == 2

        hpo_ids = {a["phenotype_id"] for a in mondo_annots}
        assert "HP:0001250" in hpo_ids
        assert "HP:0001263" in hpo_ids

    def test_parse_phenotype_hpoa_filters_not_qualifier(self, phenotype_hpoa_file):
        """NOT-qualified rows should be excluded."""
        parser = HPOAnnotationParser()
        annotations = parser.parse_phenotype_hpoa(phenotype_hpoa_file)

        all_phenos = {a["phenotype_id"] for a in annotations}
        assert "HP:0000256" not in all_phenos

    def test_parse_phenotype_hpoa_skips_unmapped_omim(self, phenotype_hpoa_file):
        """OMIM IDs without MONDO mapping should be skipped (no mondo_ontology provided)."""
        parser = HPOAnnotationParser()
        annotations = parser.parse_phenotype_hpoa(phenotype_hpoa_file)

        disease_ids = {a["disease_id"] for a in annotations}
        assert all(d.startswith("MONDO:") for d in disease_ids)

    def test_parse_phenotype_hpoa_skips_orpha(self, phenotype_hpoa_file):
        """ORPHA IDs should be skipped (not mapped)."""
        parser = HPOAnnotationParser()
        annotations = parser.parse_phenotype_hpoa(phenotype_hpoa_file)

        disease_ids = {a["disease_id"] for a in annotations}
        assert not any("ORPHA" in d for d in disease_ids)

    def test_parse_frequency(self):
        """Test frequency parsing for various formats."""
        parse = HPOAnnotationParser._parse_frequency

        assert parse("HP:0040280") == 1.0       # Obligate
        assert parse("HP:0040281") == 0.90       # Very frequent
        assert parse("HP:0040284") == 0.02       # Very rare
        assert parse("45%") == 0.45
        assert parse("3/12") == 0.25
        assert parse("") == 1.0                  # Empty -> default
        assert parse("unknown_value") == 1.0     # Unknown -> default

    def test_parse_genes_to_phenotype(self, genes_to_phenotype_file):
        """Parse genes_to_phenotype.txt into gene-pheno and gene-disease lists."""
        parser = HPOAnnotationParser()
        gene_pheno, gene_disease = parser.parse_genes_to_phenotype(
            genes_to_phenotype_file
        )

        # SCN1A has 2 phenotype links
        scn1a_phenos = [gp for gp in gene_pheno if gp["gene_id"] == "SCN1A"]
        assert len(scn1a_phenos) == 2

        # SCN1A has 1 disease link (MONDO:0011073)
        scn1a_diseases = [gd for gd in gene_disease if gd["gene_id"] == "SCN1A"]
        assert len(scn1a_diseases) == 1
        assert scn1a_diseases[0]["disease_id"] == "MONDO:0011073"

    def test_parse_genes_to_phenotype_skips_unmapped_omim(self, genes_to_phenotype_file):
        """OMIM disease IDs without MONDO mapping should be skipped in gene-disease output."""
        parser = HPOAnnotationParser()
        _, gene_disease = parser.parse_genes_to_phenotype(genes_to_phenotype_file)

        # MECP2's OMIM:312750 has no MONDO mapping -> no gene-disease entry
        mecp2_diseases = [gd for gd in gene_disease if gd["gene_id"] == "MECP2"]
        assert len(mecp2_diseases) == 0

    def test_parse_genes_to_phenotype_deduplicates(self, tmp_dir):
        """Duplicate gene-phenotype pairs should appear only once."""
        content = """\
gene_id	gene_symbol	hpo_id	hpo_name	frequency	disease_id
2565	SCN1A	HP:0001250	Seizure	-	MONDO:0011073
2565	SCN1A	HP:0001250	Seizure	-	MONDO:0011073
"""
        path = tmp_dir / "genes_to_phenotype.txt"
        path.write_text(content, encoding="utf-8")

        parser = HPOAnnotationParser()
        gene_pheno, _ = parser.parse_genes_to_phenotype(path)
        assert len(gene_pheno) == 1

    def test_build_omim_to_mondo_map_without_ontology(self):
        """Parser works without mondo_ontology (empty OMIM map)."""
        parser = HPOAnnotationParser(mondo_ontology=None)
        assert parser._omim_to_mondo == {}


# =============================================================================
# Sample Generator Tests
# =============================================================================
class TestSampleGenerator:
    """Tests for generate_training_samples"""

    def test_generate_samples_basic(self, demo_kg):
        """Generate samples from demo KG."""
        train, val = generate_training_samples(
            demo_kg, num_train=10, num_val=5, min_phenotypes=1
        )
        assert len(train) == 10
        assert len(val) == 5

    def test_sample_format(self, demo_kg):
        """Each sample has required fields with correct types."""
        train, _ = generate_training_samples(
            demo_kg, num_train=5, num_val=0, min_phenotypes=1
        )

        for sample in train:
            assert "patient_id" in sample
            assert "phenotype_ids" in sample
            assert "disease_id" in sample
            assert isinstance(sample["patient_id"], str)
            assert isinstance(sample["phenotype_ids"], list)
            assert isinstance(sample["disease_id"], int)
            assert len(sample["phenotype_ids"]) > 0

    def test_sample_ids_are_valid_indices(self, demo_kg):
        """Phenotype and disease IDs should be valid node indices."""
        node_mapping = demo_kg.get_node_id_mapping()
        n_phenotypes = len(node_mapping.get("phenotype", {}))
        n_diseases = len(node_mapping.get("disease", {}))

        train, _ = generate_training_samples(
            demo_kg, num_train=20, num_val=0, min_phenotypes=1
        )

        for sample in train:
            assert 0 <= sample["disease_id"] < n_diseases
            for pid in sample["phenotype_ids"]:
                assert 0 <= pid < n_phenotypes

    def test_deterministic_with_seed(self, demo_kg):
        """Same seed should produce same samples."""
        t1, v1 = generate_training_samples(
            demo_kg, num_train=5, num_val=2, min_phenotypes=1, seed=42
        )
        t2, v2 = generate_training_samples(
            demo_kg, num_train=5, num_val=2, min_phenotypes=1, seed=42
        )
        assert t1 == t2
        assert v1 == v2

    def test_different_seed_different_results(self, demo_kg):
        """Different seeds should produce different samples."""
        t1, _ = generate_training_samples(
            demo_kg, num_train=20, num_val=0, min_phenotypes=1, seed=1
        )
        t2, _ = generate_training_samples(
            demo_kg, num_train=20, num_val=0, min_phenotypes=1, seed=2
        )
        assert t1 != t2

    def test_output_files(self, demo_kg, tmp_dir):
        """Samples should be saved to JSON files when output_dir is given."""
        generate_training_samples(
            demo_kg, num_train=5, num_val=3, min_phenotypes=1,
            output_dir=tmp_dir,
        )

        assert (tmp_dir / "train_samples.json").exists()
        assert (tmp_dir / "val_samples.json").exists()

        with open(tmp_dir / "train_samples.json") as f:
            train = json.load(f)
        with open(tmp_dir / "val_samples.json") as f:
            val = json.load(f)

        assert len(train) == 5
        assert len(val) == 3

    def test_empty_kg_returns_empty(self):
        """KG with no disease-phenotype edges should return empty lists."""
        kg = KnowledgeGraph()
        kg.add_node(Node(
            id=NodeID(source=DataSource.MONDO, local_id="MONDO:0000001"),
            node_type=NodeType.DISEASE,
            name="Test",
        ))

        train, val = generate_training_samples(kg, num_train=5, num_val=2)
        assert train == []
        assert val == []

    def test_min_phenotypes_filter(self, demo_kg):
        """Diseases with fewer phenotypes than min_phenotypes should be excluded."""
        # With min_phenotypes=10, no disease qualifies in the small demo KG
        train, val = generate_training_samples(
            demo_kg, num_train=5, num_val=2, min_phenotypes=10
        )
        assert train == []
        assert val == []

    def test_gene_ids_included_when_available(self, demo_kg):
        """Samples should include gene_ids if the disease has associated genes."""
        train, _ = generate_training_samples(
            demo_kg, num_train=20, num_val=0, min_phenotypes=1
        )
        samples_with_genes = [s for s in train if "gene_ids" in s]
        assert len(samples_with_genes) > 0


# =============================================================================
# Build Script Validation Tests
# =============================================================================
class TestBuildScriptValidation:
    """Test that build_knowledge_graph.py validates input correctly."""

    def test_missing_files_exits_with_error(self, tmp_dir):
        """Script should exit with code 1 when required files are missing."""
        import subprocess

        result = subprocess.run(
            [
                "python", "scripts/build_knowledge_graph.py",
                "--workspace", str(tmp_dir / "output"),
                "--external-dir", str(tmp_dir / "nonexistent"),
            ],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )

        assert result.returncode == 1
        assert "Missing" in result.stderr or "not found" in result.stderr
