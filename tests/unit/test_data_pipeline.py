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
from src.kg.disease_allocation import (
    DiseaseAllocation,
    allocate_diseases,
    derive_stream,
    train_only_allocation,
    universe_digest,
)
from src.kg.sample_generator import (
    build_eligible_disease_profiles,
    generate_training_samples,
)
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


@pytest.fixture
def fake_mondo():
    """Minimal stand-in for a MONDO Ontology exposing get_all_terms/get_term.

    Note the prefix difference exercised here: MONDO xrefs use "Orphanet:<n>",
    while the annotation files use "ORPHA:<n>".
    """
    class FakeMondo:
        def get_all_terms(self, include_obsolete=False):
            return ["MONDO:0000001", "MONDO:0000002"]

        def get_term(self, term_id):
            xrefs = {
                "MONDO:0000001": ["OMIM:123456", "Orphanet:558"],
                "MONDO:0000002": ["Orphanet:999"],
            }.get(term_id)
            return {"xrefs": xrefs} if xrefs is not None else None

    return FakeMondo()


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
        parse = HPOAnnotationParser.parse_frequency

        assert parse("HP:0040280") == 1.0       # Obligate
        assert parse("HP:0040281") == 0.90       # Very frequent
        assert parse("HP:0040284") == 0.02       # Very rare
        assert parse("45%") == 0.45
        assert parse("3/12") == 0.25
        assert parse("") == 1.0                  # Empty -> default
        assert parse("unknown_value") == 1.0     # Unknown -> default

    @pytest.mark.parametrize(
        "token",
        ["200%", "-5%", "3/2", "nan%", "inf%", "-inf%", "1/0"],
    )
    def test_a_computed_frequency_outside_the_documented_range_is_malformed(self, token):
        """`float()` parses more than a frequency.

        These all used to be returned as-is while the docstring promised [0, 1].
        Two consumers depended on that promise and neither checked it: the KG
        builder writes the value straight onto the edge as `weight`, and the
        generator audit reads any value other than 1.0 as a *certainly usable*
        frequency -- so an out-of-range token inflated the lower bound of that
        measurement rather than its ambiguous middle.

        An invalid computed value is a malformed token and takes the same 1.0
        fallback an unparseable one does, which puts it in the ambiguous bucket
        where "this told us nothing" belongs.
        """
        from src.data_sources.hpo_annotations import HPOAnnotationParser

        assert HPOAnnotationParser.parse_frequency(token) == 1.0

    def test_valid_boundary_frequencies_still_parse(self):
        """The range check must not reject the ends of the range it enforces."""
        from src.data_sources.hpo_annotations import HPOAnnotationParser

        parse = HPOAnnotationParser.parse_frequency
        assert parse("0%") == 0.0
        assert parse("100%") == 1.0
        assert parse("0/5") == 0.0
        assert parse("12/12") == 1.0

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
        """Parser works without mondo_ontology (empty OMIM/ORPHA maps)."""
        parser = HPOAnnotationParser(mondo_ontology=None)
        assert parser._omim_to_mondo == {}
        assert parser._orpha_to_mondo == {}

    def test_xref_maps_built_from_ontology(self, fake_mondo):
        """OMIM and ORPHA maps are built from MONDO xrefs, re-keying Orphanet->ORPHA."""
        parser = HPOAnnotationParser(fake_mondo)

        assert parser._omim_to_mondo == {"OMIM:123456": "MONDO:0000001"}
        assert parser._orpha_to_mondo == {
            "ORPHA:558": "MONDO:0000001",
            "ORPHA:999": "MONDO:0000002",
        }

    def test_resolve_disease_id_all_prefixes(self, fake_mondo):
        """_resolve_disease_id resolves MONDO/OMIM/ORPHA; rejects unmapped + DECIPHER."""
        parser = HPOAnnotationParser(fake_mondo)

        assert parser._resolve_disease_id("MONDO:0000123") == "MONDO:0000123"  # passthrough
        assert parser._resolve_disease_id("OMIM:123456") == "MONDO:0000001"
        assert parser._resolve_disease_id("ORPHA:558") == "MONDO:0000001"
        assert parser._resolve_disease_id("ORPHA:999") == "MONDO:0000002"
        assert parser._resolve_disease_id("ORPHA:000000") is None  # unmapped ORPHA
        assert parser._resolve_disease_id("DECIPHER:1") is None

    def test_orpha_annotation_mapped_with_ontology(self, tmp_dir, fake_mondo):
        """An ORPHA-keyed phenotype.hpoa row is kept and mapped to MONDO."""
        content = (
            "database_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\t"
            "onset\tfrequency\tsex\tmodifier\taspect\tbiocuration\n"
            "ORPHA:558\tMarfan syndrome\t\tHP:0001166\tPMID:1\tPCS\t\t\t\t\tP\tHPO:test\n"
        )
        path = tmp_dir / "phenotype.hpoa"
        path.write_text(content, encoding="utf-8")

        parser = HPOAnnotationParser(fake_mondo)
        annotations = parser.parse_phenotype_hpoa(path)

        assert len(annotations) == 1
        assert annotations[0]["disease_id"] == "MONDO:0000001"


# =============================================================================
# Sample Generator Tests
# =============================================================================
class TestSampleGenerator:
    """Tests for allocation-driven sample generation.

    The interface changed with the disease-disjoint split: generation consumes an
    allocation rather than deciding one. These tests were rewritten rather than
    adapted, because several of them pinned the superseded contract — a pooled
    draw sliced by index — which is the defect the change removes.
    """

    @staticmethod
    def allocate(kg, min_phenotypes=1, val_fraction=0.5, seed=42):
        eligible = build_eligible_disease_profiles(kg, min_phenotypes)
        return allocate_diseases(eligible, val_fraction, seed=seed)

    def test_generate_samples_basic(self, demo_kg):
        allocation = self.allocate(demo_kg)
        train, val, _ = generate_training_samples(
            demo_kg, allocation, num_train=10, num_val=5, min_phenotypes=1
        )
        assert len(train) == 10
        assert len(val) == 5

    def test_sample_format(self, demo_kg):
        allocation = self.allocate(demo_kg)
        train, _, _ = generate_training_samples(
            demo_kg, allocation, num_train=10, num_val=5, min_phenotypes=1
        )
        for sample in train:
            assert isinstance(sample["patient_id"], str)
            assert isinstance(sample["phenotype_ids"], list)
            assert isinstance(sample["disease_id"], int)
            assert len(sample["phenotype_ids"]) > 0

    def test_sample_ids_are_valid_indices(self, demo_kg):
        allocation = self.allocate(demo_kg)
        train, val, _ = generate_training_samples(
            demo_kg, allocation, num_train=10, num_val=5, min_phenotypes=1
        )
        mapping = demo_kg.get_node_id_mapping()
        n_diseases = len(mapping.get("disease", {}))
        n_phenotypes = len(mapping.get("phenotype", {}))
        for sample in train + val:
            assert 0 <= sample["disease_id"] < n_diseases
            for pid in sample["phenotype_ids"]:
                assert 0 <= pid < n_phenotypes

    def test_deterministic_with_seed(self, demo_kg):
        first = generate_training_samples(
            demo_kg, self.allocate(demo_kg, seed=7), num_train=10, num_val=5,
            min_phenotypes=1,
        )
        second = generate_training_samples(
            demo_kg, self.allocate(demo_kg, seed=7), num_train=10, num_val=5,
            min_phenotypes=1,
        )
        assert first == second

    def test_different_seed_different_results(self, demo_kg):
        # A different seed must move the cut, and with it the cohorts.
        one = self.allocate(demo_kg, seed=1)
        two = self.allocate(demo_kg, seed=2)
        assert one.val_ids != two.val_ids or one.train_ids != two.train_ids

    def test_output_files_include_the_split_manifest(self, demo_kg, tmp_dir):
        allocation = self.allocate(demo_kg)
        generate_training_samples(
            demo_kg, allocation, num_train=5, num_val=3, min_phenotypes=1,
            output_dir=tmp_dir, graph_digests=_stub_graph_digests(tmp_dir), graph_export=_stub_graph_export(),
        )
        assert (tmp_dir / "train_samples.json").exists()
        assert (tmp_dir / "val_samples.json").exists()
        assert (tmp_dir / "split_manifest.json").exists()

    def test_an_empty_kg_cannot_be_allocated(self):
        """A universe with no eligible disease is refused, not silently emptied.

        The superseded generator returned two empty lists here, which reads as a
        successful run that produced nothing.
        """
        kg = KnowledgeGraph()
        kg.add_node(Node(
            id=NodeID(source=DataSource.MONDO, local_id="MONDO:0000001"),
            node_type=NodeType.DISEASE,
            name="Test",
        ))
        with pytest.raises(ValueError, match="two non-empty partitions"):
            self.allocate(kg)

    def test_min_phenotypes_filter_leaves_nothing_to_allocate(self, demo_kg):
        with pytest.raises(ValueError, match="two non-empty partitions"):
            self.allocate(demo_kg, min_phenotypes=10)

    def test_gene_ids_included_when_available(self, demo_kg):
        allocation = self.allocate(demo_kg)
        train, _, _ = generate_training_samples(
            demo_kg, allocation, num_train=20, num_val=5, min_phenotypes=1
        )
        assert [s for s in train if "gene_ids" in s]


class TestDiseaseDisjointness:
    """The property the change exists to establish."""

    def test_no_disease_appears_in_both_cohorts(self, demo_kg):
        eligible = build_eligible_disease_profiles(demo_kg, 1)
        allocation = allocate_diseases(eligible, 0.5, seed=42)
        train, val, manifest = generate_training_samples(
            demo_kg, allocation, num_train=40, num_val=20, min_phenotypes=1
        )
        train_diseases = {s["disease_id"] for s in train}
        val_diseases = {s["disease_id"] for s in val}
        assert not (train_diseases & val_diseases)
        assert manifest["disjoint"] is True

    def test_patient_ids_do_not_collide_across_cohorts(self, demo_kg):
        """Each partition has its own namespace; a shared counter restarts."""
        eligible = build_eligible_disease_profiles(demo_kg, 1)
        allocation = allocate_diseases(eligible, 0.5, seed=42)
        train, val, _ = generate_training_samples(
            demo_kg, allocation, num_train=10, num_val=10, min_phenotypes=1
        )
        train_ids = {s["patient_id"] for s in train}
        val_ids = {s["patient_id"] for s in val}
        assert len(train_ids) == len(train)
        assert len(val_ids) == len(val)
        assert not (train_ids & val_ids)

    def test_changing_the_training_budget_does_not_move_the_validation_cohort(
        self, demo_kg
    ):
        """Independent streams. One shared generator would shift every later draw."""
        eligible = build_eligible_disease_profiles(demo_kg, 1)
        allocation = allocate_diseases(eligible, 0.5, seed=42)
        _, val_small, _ = generate_training_samples(
            demo_kg, allocation, num_train=10, num_val=8, min_phenotypes=1
        )
        _, val_large, _ = generate_training_samples(
            demo_kg, allocation, num_train=90, num_val=8, min_phenotypes=1
        )
        assert val_small == val_large

    def test_every_allocated_disease_receives_a_sample(self, demo_kg):
        """Coverage by construction, not by a large enough budget."""
        eligible = build_eligible_disease_profiles(demo_kg, 1)
        allocation = allocate_diseases(eligible, 0.5, seed=42)
        train, val, manifest = generate_training_samples(
            demo_kg, allocation, num_train=len(allocation.train),
            num_val=len(allocation.val), min_phenotypes=1,
        )
        assert {s["disease_id"] for s in train} == set(allocation.train_ids)
        assert {s["disease_id"] for s in val} == set(allocation.val_ids)
        assert manifest["realised"]["train_diseases"] == len(allocation.train)

    def test_a_budget_too_small_to_cover_the_partition_is_refused(self, demo_kg):
        eligible = build_eligible_disease_profiles(demo_kg, 1)
        allocation = allocate_diseases(eligible, 0.5, seed=42)
        with pytest.raises(ValueError, match="cannot cover"):
            generate_training_samples(
                demo_kg, allocation, num_train=0, num_val=5, min_phenotypes=1
            )

    def test_a_zero_budget_against_a_populated_partition_is_refused(self, demo_kg):
        """An early return on count == 0 used to skip the coverage check, so a
        partition with diseases and no budget produced nothing and still looked
        like a successful run."""
        eligible = build_eligible_disease_profiles(demo_kg, 1)
        allocation = allocate_diseases(eligible, 0.5, seed=42)
        assert allocation.val, "the fixture must allocate a validation partition"
        # Caught by the earlier intent guard, which names the mismatch precisely:
        # a validation partition was withheld and then asked for no samples.
        with pytest.raises(ValueError, match="num_val is 0 but the allocation"):
            generate_training_samples(
                demo_kg, allocation, num_train=20, num_val=0, min_phenotypes=1
            )

    def test_train_only_is_explicit_rather_than_a_fraction_of_zero(self, demo_kg):
        eligible = build_eligible_disease_profiles(demo_kg, 1)
        allocation = train_only_allocation(eligible, seed=42)
        train, val, manifest = generate_training_samples(
            demo_kg, allocation, num_train=20, num_val=0, min_phenotypes=1
        )
        assert val == []
        assert manifest["realised"]["val_diseases"] == 0
        assert len(train) == 20
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            allocate_diseases(eligible, 0.0, seed=42)

    def test_a_validation_budget_without_a_validation_partition_is_refused(
        self, demo_kg
    ):
        eligible = build_eligible_disease_profiles(demo_kg, 1)
        with pytest.raises(ValueError, match="no validation partition"):
            generate_training_samples(
                demo_kg, train_only_allocation(eligible, seed=42),
                num_train=20, num_val=5, min_phenotypes=1,
            )

    def test_profile_lists_are_sorted(self, demo_kg):
        """Canonical order, so sampling cannot depend on set layout."""
        for _, profile in build_eligible_disease_profiles(demo_kg, 1):
            assert profile["phenotype_ids"] == sorted(profile["phenotype_ids"])
            assert profile["gene_ids"] == sorted(profile["gene_ids"])

    def test_the_manifest_derives_realised_sets_from_the_emitted_records(
        self, demo_kg
    ):
        """Copying allocation metadata would restate it, not evidence it."""
        eligible = build_eligible_disease_profiles(demo_kg, 1)
        allocation = allocate_diseases(eligible, 0.5, seed=42)
        _, _, manifest = generate_training_samples(
            demo_kg, allocation, num_train=20, num_val=10, min_phenotypes=1
        )
        assert manifest["realised"]["derived_from"].startswith("the emitted")
        assert manifest["allocation"]["allocated"]["train_digest"] == \
            manifest["realised"]["train_digest"]
        assert manifest["generation"]["algorithm_version"] >= 1


# =============================================================================
# Build Script Validation Tests
# =============================================================================
class TestBuildScriptValidation:
    """Test that build_knowledge_graph.py validates input correctly."""

    def test_missing_files_exits_with_error(self, tmp_dir):
        """Script should exit with code 1 when required files are missing."""
        import subprocess
        import sys

        # sys.executable, not "python": the latter resolves against PATH, which is
        # only the environment running the suite when a venv happens to be active.
        # Otherwise the subprocess runs a different interpreter without the
        # project's dependencies, and the script dies on an unrelated ImportError
        # before it can report the missing input files this test is about.
        result = subprocess.run(
            [
                sys.executable, "scripts/build_knowledge_graph.py",
                "--workspace", str(tmp_dir / "output"),
                "--external-dir", str(tmp_dir / "nonexistent"),
            ],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )

        assert result.returncode == 1
        assert "Missing" in result.stderr or "not found" in result.stderr


# =============================================================================
# The disease-truth range invariant, at the boundary that creates the failure
# =============================================================================
class TestDiseaseTruthRangeInvariant:
    """`_assert_disease_truth_in_range` — the first of three boundaries.

    `_remap_indices` sends each global disease id through a mapping tensor that
    is `-1` at every unsampled position, so this is where a legal global truth
    can become an illegal local one. Catching it here keeps the check on the
    **host**: the equivalent test inside `Trainer._compute_model_outputs` would
    run after `_move_to_device` and force a host-device synchronisation on every
    valid batch.

    The two downstream boundaries — `DiagnosisLoss` and `to_global_ids` — are
    covered by their own files and are deliberately independent of this one.
    """

    @staticmethod
    def _check(disease_ids, n_rows=4):
        import torch

        from src.kg.data_loader import DiagnosisDataLoader

        DiagnosisDataLoader._assert_disease_truth_in_range(
            {
                "disease_ids": torch.tensor(disease_ids),
                "patient_ids": [f"p{i}" for i in range(len(disease_ids))],
            },
            {"disease": torch.arange(n_rows)},
        )

    def test_legal_truths_pass(self):
        self._check([0, 1, 3])
        self._check([0])
        self._check([3, 3])

    def test_a_minus_one_hole_is_refused(self):
        """The unsampled-disease case: in range globally, absent from the subgraph."""
        with pytest.raises(ValueError, match=r"disease truth out of range"):
            self._check([0, -1, 2])

    def test_an_id_beyond_the_subgraph_is_refused(self):
        """`_remap_indices` leaves an out-of-range global id unchanged, so it
        arrives here as a number that is not a local index at all."""
        with pytest.raises(ValueError, match=r"not in \[0, 4\)"):
            self._check([0, 4])

    def test_the_error_names_the_offending_patients(self):
        """A cohort-sized failure needs to say *which* samples, not just that one."""
        with pytest.raises(ValueError, match="p1"):
            self._check([0, -1, 2])

    def test_a_truth_is_never_clamped(self):
        """Refuse, never correct. A clamp would silently score another disease."""
        import torch

        from src.kg.data_loader import DiagnosisDataLoader

        batch = {"disease_ids": torch.tensor([0, 2]), "patient_ids": ["a", "b"]}
        DiagnosisDataLoader._assert_disease_truth_in_range(
            batch, {"disease": torch.arange(4)}
        )
        assert batch["disease_ids"].tolist() == [0, 2], "the check must not mutate"

    def test_a_batch_without_disease_ids_is_not_an_error(self):
        """Not every consumer carries a disease truth; absence is not malformity."""
        import torch

        from src.kg.data_loader import DiagnosisDataLoader

        DiagnosisDataLoader._assert_disease_truth_in_range({}, {"disease": torch.arange(4)})
        DiagnosisDataLoader._assert_disease_truth_in_range(
            {"disease_ids": torch.tensor([0])}, {}
        )


class TestSplitRegimeBoundary:
    """Two split regimes must not share a checkpoint directory."""

    @staticmethod
    def _allocation(kg):
        return allocate_diseases(build_eligible_disease_profiles(kg, 1), 0.5, seed=42)

    def test_regenerating_where_checkpoints_exist_is_refused(self, demo_kg, tmp_dir):
        (tmp_dir / "checkpoints" / "hgt").mkdir(parents=True)
        (tmp_dir / "checkpoints" / "hgt" / "model-01-0.5000.pt").write_bytes(b"x")
        with pytest.raises(FileExistsError, match="two split regimes"):
            generate_training_samples(
                demo_kg, self._allocation(demo_kg), num_train=10, num_val=5,
                min_phenotypes=1, output_dir=tmp_dir, graph_digests=_stub_graph_digests(tmp_dir), graph_export=_stub_graph_export(),
            )

    def test_the_refusal_writes_nothing(self, demo_kg, tmp_dir):
        (tmp_dir / "checkpoints" / "gat").mkdir(parents=True)
        with pytest.raises(FileExistsError):
            generate_training_samples(
                demo_kg, self._allocation(demo_kg), num_train=10, num_val=5,
                min_phenotypes=1, output_dir=tmp_dir, graph_digests=_stub_graph_digests(tmp_dir), graph_export=_stub_graph_export(),
            )
        assert not (tmp_dir / "train_samples.json").exists()
        assert not (tmp_dir / "split_manifest.json").exists()

    def test_an_empty_checkpoint_directory_does_not_block(self, demo_kg, tmp_dir):
        """Nothing was trained there, so nothing can be mixed."""
        (tmp_dir / "checkpoints").mkdir()
        generate_training_samples(
            demo_kg, self._allocation(demo_kg), num_train=10, num_val=5,
            min_phenotypes=1, output_dir=tmp_dir, graph_digests=_stub_graph_digests(tmp_dir), graph_export=_stub_graph_export(),
        )
        assert (tmp_dir / "split_manifest.json").exists()

    def test_a_fresh_workspace_is_unaffected(self, demo_kg, tmp_dir):
        generate_training_samples(
            demo_kg, self._allocation(demo_kg), num_train=10, num_val=5,
            min_phenotypes=1, output_dir=tmp_dir, graph_digests=_stub_graph_digests(tmp_dir), graph_export=_stub_graph_export(),
        )
        assert (tmp_dir / "train_samples.json").exists()


def _wide_kg_object():
    """The wide fixture as a plain callable, for tests outside its scope."""
    return _build_wide_kg()


@pytest.fixture
def wide_kg():
    """A universe big enough that coincidences stop hiding defects.

    The demo KG has two eligible diseases and phenotype indices 0-2, which is
    small enough that ``list(set(...))`` comes out sorted by accident and
    replacement sampling covers everything by accident. Three mutation tests
    passed against it while the guarantees they check were removed. This fixture
    has twelve diseases and non-contiguous phenotype ids, where neither
    coincidence holds.
    """
    kg = KnowledgeGraph()
    # Sixty-four phenotypes, so the *indices* a profile holds are large and
    # scattered. Index magnitude is what matters, not the HPO string: the profile
    # stores node indices assigned by insertion order, and a handful of small
    # contiguous ones come out of a set already sorted.
    for i in range(64):
        hp_id = f"HP:{i:07d}"
        kg.add_node(Node(id=NodeID(source=DataSource.HPO, local_id=hp_id),
                         node_type=NodeType.PHENOTYPE, name=hp_id))
    for i in range(12):
        mondo = f"MONDO:{i:07d}"
        kg.add_node(Node(id=NodeID(source=DataSource.MONDO, local_id=mondo),
                         node_type=NodeType.DISEASE, name=mondo))
        for offset in (0, 17, 33, 49):
            kg.add_edge(Edge(
                source_id=NodeID(
                    source=DataSource.HPO, local_id=f"HP:{(i * 5 + offset) % 64:07d}"
                ),
                target_id=NodeID(source=DataSource.MONDO, local_id=mondo),
                edge_type=EdgeType.PHENOTYPE_OF_DISEASE,
            ))
    return kg


_build_wide_kg = wide_kg.__wrapped__ if hasattr(wide_kg, "__wrapped__") else None


class TestGuaranteesThatNeedAWideUniverse:
    """The three properties a two-disease fixture cannot distinguish."""

    def test_coverage_gives_every_allocated_disease_exactly_one_sample(self, wide_kg):
        """At ``count == len(partition)`` the coverage pass makes this exact.

        Replacement sampling would hit every disease of a 10-disease partition in
        10 draws with probability 10!/10^10, about 0.036%. The assertion is
        deterministic under coverage-first and effectively impossible without it.
        """
        eligible = build_eligible_disease_profiles(wide_kg, 2)
        allocation = allocate_diseases(eligible, 0.2, seed=42)
        train, val, _ = generate_training_samples(
            wide_kg, allocation, num_train=len(allocation.train),
            num_val=len(allocation.val), min_phenotypes=2,
        )
        assert len(allocation.train) >= 8, "the fixture must be wide enough to matter"
        assert sorted(s["disease_id"] for s in train) == sorted(allocation.train_ids)
        assert sorted(s["disease_id"] for s in val) == sorted(allocation.val_ids)

    def test_profile_lists_are_sorted_at_realistic_identifier_magnitudes(self, wide_kg):
        """``list(set(...))`` is sorted for tiny ints and is not for these."""
        profiles = build_eligible_disease_profiles(wide_kg, 2)
        # The fixture must actually exercise the failure: without sorting, most
        # of these profiles come out of their set in a non-ascending order.
        unsorted_without_the_fix = sum(
            1 for _, p in profiles if list(set(p["phenotype_ids"])) != sorted(p["phenotype_ids"])
        )
        assert unsorted_without_the_fix >= 8, (
            "the fixture no longer distinguishes sorted from set order"
        )
        for _, profile in profiles:
            assert profile["phenotype_ids"] == sorted(profile["phenotype_ids"])


class TestRandomStreams:
    """The derivation contract, tested where it lives rather than through effects."""

    def test_named_streams_are_independent(self):
        one = derive_stream(42, "train").random()
        two = derive_stream(42, "val").random()
        three = derive_stream(42, "allocation").random()
        assert len({one, two, three}) == 3

    def test_the_same_name_and_seed_reproduce(self):
        assert derive_stream(42, "train").random() == derive_stream(42, "train").random()

    def test_a_different_root_seed_moves_every_stream(self):
        for name in ("allocation", "train", "val"):
            assert derive_stream(1, name).random() != derive_stream(2, name).random()

    def test_the_separator_cannot_be_smuggled_into_a_stream_name(self):
        """``f"{seed}|{name}"`` is only unambiguous if names carry no separator."""
        with pytest.raises(ValueError, match="must not contain"):
            derive_stream(42, "train|val")


class TestWorkspaceSafety:
    """Refusal must happen before the first byte, not before the last."""

    @staticmethod
    def _prepopulate(workspace):
        """A workspace as it would be after a real build plus training."""
        (workspace / "checkpoints" / "hgt").mkdir(parents=True)
        payload = {
            "kg.json": b'{"original": "graph"}',
            "node_features.pt": b"ORIGINAL-FEATURES",
            "edge_indices.pt": b"ORIGINAL-EDGES",
            "num_nodes.json": b'{"disease": 1}',
            "train_samples.json": b'[{"original": true}]',
            "val_samples.json": b'[{"original": true}]',
            "checkpoints/hgt/model-01-0.5000.pt": b"ORIGINAL-WEIGHTS",
        }
        for name, data in payload.items():
            (workspace / name).write_bytes(data)
        return payload

    def test_a_rebuild_refuses_before_touching_any_workspace_byte(self, tmp_dir):
        """The defect: kg.json and the graph tensors were written first.

        The generator's guard fired only just before the sample files, so a
        rebuild overwrote the graph under trained checkpoints and *then* refused
        — leaving old weights paired with a new graph, which is worse than either
        rebuilding cleanly or not rebuilding at all.
        """
        import scripts.build_knowledge_graph as build

        payload = self._prepopulate(tmp_dir)
        with pytest.raises(FileExistsError, match="two split regimes"):
            build.build_knowledge_graph(
                external_dir=tmp_dir / "no_such_external_dir",
                workspace=tmp_dir,
            )
        for name, original in payload.items():
            assert (tmp_dir / name).read_bytes() == original, f"{name} was modified"

    def test_the_preflight_precedes_input_validation(self, tmp_dir):
        """Ordering matters: a missing-input error would mask the real problem.

        With no annotation files present, the build would normally fail on input
        validation. The checkpoint refusal must win, so the operator is told the
        thing that determines what they should do next.
        """
        import scripts.build_knowledge_graph as build

        self._prepopulate(tmp_dir)
        with pytest.raises(FileExistsError):
            build.build_knowledge_graph(
                external_dir=tmp_dir / "no_such_external_dir",
                workspace=tmp_dir,
            )


class TestAllocationBoundary:
    """DiseaseAllocation is publicly constructible; immutability is not validity."""

    @staticmethod
    def _eligible(kg):
        return build_eligible_disease_profiles(kg, 1)

    def test_a_hand_built_overlapping_allocation_is_refused(self, demo_kg):
        eligible = self._eligible(demo_kg)
        overlapping = DiseaseAllocation(
            train=tuple(eligible), val=tuple(eligible[:1]),
            val_fraction_requested=0.5, seed=42,
            universe_digest=universe_digest(eligible),
        )
        with pytest.raises(ValueError, match="appear in both partitions"):
            generate_training_samples(
                demo_kg, overlapping, num_train=10, num_val=5, min_phenotypes=1
            )

    def test_duplicate_disease_ids_within_a_partition_are_refused(self, demo_kg):
        eligible = self._eligible(demo_kg)
        duplicated = DiseaseAllocation(
            train=tuple(eligible) + (eligible[0],), val=(),
            val_fraction_requested=0.0, seed=42,
            universe_digest=universe_digest(eligible),
        )
        with pytest.raises(ValueError, match="duplicate disease ids"):
            generate_training_samples(
                demo_kg, duplicated, num_train=10, num_val=0, min_phenotypes=1
            )

    def test_a_bool_disease_id_is_refused(self, demo_kg):
        eligible = self._eligible(demo_kg)
        bad = DiseaseAllocation(
            train=((True, eligible[0][1]),), val=(),
            val_fraction_requested=0.0, seed=42, universe_digest="unused",
        )
        with pytest.raises(ValueError, match="must be an integer"):
            generate_training_samples(demo_kg, bad, num_train=5, num_val=0)

    def test_an_allocation_from_a_different_graph_is_refused(self, demo_kg, wide_kg):
        """Same disease indices, different phenotype content."""
        foreign = allocate_diseases(self._eligible(wide_kg), 0.5, seed=42)
        with pytest.raises(ValueError, match="different disease universe"):
            generate_training_samples(
                demo_kg, foreign, num_train=10, num_val=5, min_phenotypes=1
            )

    def test_modified_profile_content_is_refused(self, demo_kg):
        """An id-only digest would not catch this; the content digest does."""
        eligible = self._eligible(demo_kg)
        allocation = allocate_diseases(eligible, 0.5, seed=42)
        tampered = [(d, {**p, "phenotype_ids": p["phenotype_ids"][:1] + [999]})
                    for d, p in allocation.train]
        # Caught by self-consistency, which names the cause precisely: the
        # contents no longer match the digest recorded when the cut was made.
        with pytest.raises(ValueError, match="modified since it was cut"):
            generate_training_samples(
                demo_kg,
                allocation._replace(train=tuple(tampered)),
                num_train=10, num_val=5, min_phenotypes=1,
            )

    def test_an_under_eligible_profile_is_refused(self, wide_kg):
        """Cut at one min_phenotypes, generated at a stricter one."""
        allocation = allocate_diseases(build_eligible_disease_profiles(wide_kg, 2), 0.2, seed=42)
        # max_phenotypes must stay >= min_phenotypes, or the input validator
        # fires first and names that instead — a different, also-real problem.
        with pytest.raises(ValueError, match="below the min_phenotypes"):
            generate_training_samples(
                wide_kg, allocation, num_train=20, num_val=5,
                min_phenotypes=99, max_phenotypes=99,
            )


def _stub_graph_export():
    """The recipe that goes beside those digests.

    The digests say which bytes `node_features.pt` is; schema 3 also promises it
    can be remade, and the reader collects that promise. A test writing a
    workspace has to produce both or it produces one no consumer opens -- which
    would be this file testing the binding through a workspace that fails for an
    unrelated reason. Shared with the fixture so there is one answer to "what
    does a written workspace carry".
    """
    from tests.fixtures.generated_workspace import default_graph_export

    return default_graph_export()


def _stub_graph_digests(root):
    """Stand-in graph artifacts plus their digests, as the export writer supplies.

    A workspace-writing call now binds the whole graph export, so a test that
    writes one has to produce it. The bytes are arbitrary: what is under test is
    the binding, not the tensors.
    """
    from src.kg.artifacts import GRAPH_ARTIFACTS
    from src.utils.fingerprint import file_sha256

    root.mkdir(parents=True, exist_ok=True)
    digests = {}
    for role, filename in GRAPH_ARTIFACTS.items():
        (root / filename).write_bytes(f"{role}-bytes".encode())
        digests[role] = file_sha256(root / filename)
    return digests


class TestManifestBinding:
    """The manifest must describe the bytes on disk, not an equivalent object."""

    @staticmethod
    def _graph_digests(root, *, roles=None):
        """Write stand-in graph artifacts and digest them, as the writer does."""
        from src.kg.artifacts import GRAPH_ARTIFACTS
        from src.utils.fingerprint import file_sha256

        digests = {}
        for role, filename in GRAPH_ARTIFACTS.items():
            if roles is not None and role not in roles:
                continue
            (root / filename).write_bytes(f"{role}-bytes".encode())
            digests[role] = file_sha256(root / filename)
        return digests

    def test_the_manifest_digests_the_files_that_were_written(self, wide_kg, tmp_dir):
        from src.utils.fingerprint import file_sha256

        digests = self._graph_digests(tmp_dir)
        allocation = allocate_diseases(build_eligible_disease_profiles(wide_kg, 2), 0.2, seed=42)
        _, _, manifest = generate_training_samples(
            wide_kg, allocation, num_train=30, num_val=10, min_phenotypes=2,
            output_dir=tmp_dir, graph_digests=digests, graph_export=_stub_graph_export(),
        )
        for role, filename in (
            ("train_samples", "train_samples.json"),
            ("val_samples", "val_samples.json"),
            ("kg", "kg.json"),
            ("node_features", "node_features.pt"),
            ("edge_indices", "edge_indices.pt"),
            ("num_nodes", "num_nodes.json"),
        ):
            assert manifest["artifacts"][role] == file_sha256(tmp_dir / filename)

    def test_generation_does_not_hash_the_graph_files_itself(self, wide_kg, tmp_dir):
        """**The first version of this test demonstrated the hole it should close.**

        It placed arbitrary bytes at ``kg.json`` and asserted that those unrelated
        bytes were hashed — recording one graph's universe digest beside another
        file's digest as if they were one provenance chain. Generation cannot
        vouch for files it did not write, so it hashes none of them: the digests
        it records are the ones the export writer handed it.
        """
        digests = self._graph_digests(tmp_dir)
        (tmp_dir / "kg.json").write_bytes(b'{"some other": "graph"}')
        allocation = allocate_diseases(build_eligible_disease_profiles(wide_kg, 2), 0.2, seed=42)
        _, _, manifest = generate_training_samples(
            wide_kg, allocation, num_train=30, num_val=10, min_phenotypes=2,
            output_dir=tmp_dir, graph_digests=digests, graph_export=_stub_graph_export(),
        )
        assert manifest["artifacts"]["kg"] == digests["kg"]

    @pytest.mark.parametrize(
        "present", [("kg",), ("kg", "node_features"),
                    ("kg", "node_features", "edge_indices"), ()],
    )
    def test_a_workspace_binding_only_part_of_the_export_is_refused(
        self, wide_kg, tmp_dir, present
    ):
        """The four artifacts are one call to `export_graph_data` from one graph.
        A manifest binding some of them describes half a production event, and the
        half it omits is the half a model consumes."""
        digests = self._graph_digests(tmp_dir, roles=present)
        allocation = allocate_diseases(build_eligible_disease_profiles(wide_kg, 2), 0.2, seed=42)
        with pytest.raises(ValueError, match="would leave"):
            generate_training_samples(
                wide_kg, allocation, num_train=30, num_val=10, min_phenotypes=2,
                output_dir=tmp_dir, graph_digests=digests, graph_export=_stub_graph_export(),
            )

    def test_tampering_with_a_sample_file_breaks_its_recorded_digest(
        self, wide_kg, tmp_dir
    ):
        from src.utils.fingerprint import file_sha256

        digests = self._graph_digests(tmp_dir)
        allocation = allocate_diseases(build_eligible_disease_profiles(wide_kg, 2), 0.2, seed=42)
        _, _, manifest = generate_training_samples(
            wide_kg, allocation, num_train=30, num_val=10, min_phenotypes=2,
            output_dir=tmp_dir, graph_digests=digests, graph_export=_stub_graph_export(),
        )
        (tmp_dir / "train_samples.json").write_bytes(b'[{"tampered": true}]')
        assert file_sha256(tmp_dir / "train_samples.json") != manifest["artifacts"]["train_samples"]

    def test_an_in_memory_manifest_binds_nothing_and_is_not_asked_to(self, wide_kg):
        """No workspace was produced, so there is no export to bind. The binding
        requirement belongs where a workspace is written, not where a cut is
        inspected."""
        allocation = allocate_diseases(build_eligible_disease_profiles(wide_kg, 2), 0.2, seed=42)
        _, _, manifest = generate_training_samples(
            wide_kg, allocation, num_train=30, num_val=10, min_phenotypes=2
        )
        assert manifest["artifacts"] == {"train_samples": None, "val_samples": None}


class TestSharedBudgetDomain:
    """One validator, called from both entry points.

    The two-phase preflight is only sound if its cheap half knows the *whole*
    domain of a budget. The build path previously knew two facts — not ``None``,
    not smaller than the partition — and a large float satisfies both, so the
    graph was written before the generator refused it.
    """

    @pytest.mark.parametrize(
        "num_train,num_val,message",
        [
            (None, 5, "num_train is required"),
            (10, None, "num_val is required"),
            (50000.0, 5, "num_train must be an integer"),
            (10, 5.0, "num_val must be an integer"),
            (True, 5, "num_train must be an integer"),
            (10, True, "num_val must be an integer"),
            (-1, 5, "num_train must be >= 0"),
            (10, -1, "num_val must be >= 0"),
        ],
    )
    def test_the_library_refuses_every_unusable_budget(self, num_train, num_val, message):
        from src.kg.sample_generator import validate_sample_budgets

        with pytest.raises(ValueError, match=message):
            validate_sample_budgets(num_train, num_val)

    def test_usable_budgets_pass(self):
        from src.kg.sample_generator import validate_sample_budgets

        validate_sample_budgets(0, 0)
        validate_sample_budgets(10, 5)

    @pytest.mark.parametrize(
        "num_train,num_val",
        [(None, 5), (10, None), (50000.0, 5), (10, 5.0), (True, 5), (10, True),
         (-1, 5), (10, -1)],
    )
    def test_the_script_refuses_the_same_set(self, num_train, num_val):
        """Phase one delegates rather than restating, so the sets cannot drift.

        Parametrised identically to the library case on purpose: were the script
        to grow its own copy of the rules, one of these pairs would diverge.
        """
        import scripts.build_knowledge_graph as build

        with pytest.raises(SystemExit, match="must both be supplied with --generate-samples"):
            build.require_usable_budgets(num_train, num_val)

    def test_the_script_accepts_what_the_library_accepts(self):
        import scripts.build_knowledge_graph as build

        build.require_usable_budgets(0, 0)
        build.require_usable_budgets(10, 5)

    def test_both_entry_points_route_through_the_one_validator(
        self, monkeypatch, demo_kg
    ):
        """Not "both refuse the same inputs" — both run the *same code*.

        Two independent copies would agree on today's rules and pass the
        parametrised pairs above; they would diverge the first time one is
        extended. Replacing the single definition and watching both paths change
        behaviour is what distinguishes sharing from coincidence.
        """
        import scripts.build_knowledge_graph as build
        import src.kg.sample_generator as generator

        class _Sentinel(Exception):
            pass

        seen = []

        def _refuse(num_train, num_val):
            seen.append((num_train, num_val))
            raise _Sentinel("replaced")

        monkeypatch.setattr(generator, "validate_sample_budgets", _refuse)

        allocation = allocate_diseases(
            build_eligible_disease_profiles(demo_kg, 1), 0.5, seed=42
        )
        with pytest.raises(_Sentinel):
            generator.generate_training_samples(
                demo_kg, allocation, num_train=4, num_val=2
            )
        with pytest.raises(_Sentinel):
            build.require_usable_budgets(7, 3)
        assert seen == [(4, 2), (7, 3)]


class TestBudgetPreflight:
    """Budget refusals must precede every workspace write, not follow them."""

    @staticmethod
    def _existing(workspace):
        payload = {"kg.json": b'{"before": true}', "train_samples.json": b"[]"}
        for name, data in payload.items():
            (workspace / name).write_bytes(data)
        return payload

    @staticmethod
    def _allocation():
        return allocate_diseases(
            build_eligible_disease_profiles(_wide_kg_object(), 2), 0.2, seed=42
        )

    def test_a_budget_too_small_for_its_partition_is_refused(self):
        """`_generate_partition` refuses too, but only after the graph is saved.

        The comparison lives in the writer, which is what runs it before the
        first byte; the script supplies the flag names its operator typed.
        """
        from src.kg.workspace import WorkspaceRefusal, require_budget_coverage

        allocation = self._allocation()
        with pytest.raises(WorkspaceRefusal, match="cannot cover"):
            require_budget_coverage(1, 999, allocation)
        with pytest.raises(WorkspaceRefusal, match="--num-val"):
            require_budget_coverage(
                999, 0, allocation,
                train_label="--num-train", val_label="--num-val",
            )

    def test_sufficient_budgets_pass(self):
        from src.kg.workspace import require_budget_coverage

        allocation = self._allocation()
        require_budget_coverage(
            len(allocation.train), len(allocation.val), allocation
        )

    def test_the_refusal_is_the_kind_a_caller_may_call_unwritten(self):
        """`WorkspaceRefusal` is what lets the build say "Nothing was written" and
        be right: every other failure in the writer can happen after the graph
        is on disk."""
        from src.kg.workspace import WorkspaceRefusal, require_budget_coverage

        with pytest.raises(WorkspaceRefusal):
            require_budget_coverage(1, 1, self._allocation())
        assert issubclass(WorkspaceRefusal, ValueError)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("num_train", True, "must be an integer"),
        ("num_train", 1.5, "must be an integer"),
        ("num_train", -1, "must be >= 0"),
        ("num_val", True, "must be an integer"),
        ("min_phenotypes", 0, "must be >= 1"),
        ("min_phenotypes", True, "must be an integer"),
        ("max_phenotypes", 1, "must be >= 2"),
        ("phenotype_drop_rate", -0.1, "finite and in"),
        ("phenotype_drop_rate", 1.5, "finite and in"),
        ("phenotype_drop_rate", float("nan"), "finite and in"),
        ("phenotype_drop_rate", "0.3", "must be a number"),
    ],
)
def test_generation_inputs_are_validated_at_the_api(demo_kg, field, value, message):
    """Importable, so argparse is not the only entry point that must be sound."""
    allocation = allocate_diseases(build_eligible_disease_profiles(demo_kg, 1), 0.5, seed=42)
    kwargs = dict(num_train=10, num_val=5, min_phenotypes=2, max_phenotypes=15,
                  phenotype_drop_rate=0.3)
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        generate_training_samples(demo_kg, allocation, **kwargs)


class TestBuildPathOrdering:
    """The wiring, not just the helpers.

    Testing ``require_sufficient_budgets`` in isolation leaves the call site
    untested: removing it from the build path broke no test, which is how this
    gap was found. These stub the expensive stages so the ordering itself can be
    exercised.
    """

    @staticmethod
    def _stub_build_stages(monkeypatch, kg):
        import scripts.build_knowledge_graph as build

        class _Loader:
            def __init__(self, *a, **k): pass
            def load_mondo(self): return object()
            def load_hpo(self): return object()

        class _Parser:
            def __init__(self, *a, **k): pass
            def parse_phenotype_hpoa(self, *a, **k): return []
            def parse_genes_to_phenotype(self, *a, **k): return ([], [])

        class _Builder:
            def __init__(self, *a, **k): pass
            def add_ontology(self, *a, **k): return 0
            def add_phenotype_disease_annotations(self, *a, **k): return 0
            def add_gene_disease_associations(self, *a, **k): return (0, 0)
            def add_gene_phenotype_associations(self, *a, **k): return 0
            def build(self): return kg

        monkeypatch.setattr(build, "OntologyLoader", _Loader)
        monkeypatch.setattr(build, "HPOAnnotationParser", _Parser)
        monkeypatch.setattr(build, "KnowledgeGraphBuilder", _Builder)
        return build

    @staticmethod
    def _satisfy_input_validation(external_dir):
        """The annotation files the build checks for before doing anything."""
        for name in ("phenotype.hpoa", "genes_to_phenotype.txt"):
            (external_dir / name).write_text("")

    @pytest.mark.parametrize(
        "num_train,num_val,expected",
        [
            (None, None, "must both be supplied"),
            (50000.0, 5, "num_train must be an integer"),
            (5, 50000.0, "num_val must be an integer"),
            (True, 5, "num_train must be an integer"),
            (5, True, "num_val must be an integer"),
            (-1, 5, "num_train must be >= 0"),
            (1, 1, "cannot cover"),
        ],
        ids=["missing", "float-train", "float-val", "bool-train", "bool-val",
             "negative", "insufficient"],
    )
    def test_a_budget_refusal_leaves_the_workspace_untouched(
        self, monkeypatch, tmp_dir, wide_kg, num_train, num_val, expected
    ):
        """A large float and a ``True`` used to reach the generator.

        Both cleared the old build-path checks — a float is never less than a
        disease count, and ``True`` covers a one-disease partition — so kg.json
        and the graph tensors were written and only then was the run refused.
        """
        build = self._stub_build_stages(monkeypatch, wide_kg)
        self._satisfy_input_validation(tmp_dir)
        before = {"kg.json": b'{"pre-existing": true}'}
        for name, data in before.items():
            (tmp_dir / name).write_bytes(data)

        with pytest.raises(SystemExit, match=expected):
            build.build_knowledge_graph(
                external_dir=tmp_dir, workspace=tmp_dir, generate_samples=True,
                num_train=num_train, num_val=num_val,
            )

        for name, original in before.items():
            assert (tmp_dir / name).read_bytes() == original, f"{name} was rewritten"
        for written in ("node_features.pt", "edge_indices.pt", "num_nodes.json",
                        "train_samples.json", "split_manifest.json"):
            assert not (tmp_dir / written).exists(), f"{written} should not exist"

    def test_phase_one_refuses_before_anything_is_read(self, monkeypatch, tmp_dir):
        """The cheap half must precede the expensive stages, not merely precede
        the writes.

        Ordering is the whole value of splitting the preflight: a budget's own
        domain needs no ontology, so an operator who mistypes one should not pay
        for a MONDO load, an HPO load, two annotation parses and a KG build first.
        The stubs here refuse to be constructed, so any of those stages running
        is a failure rather than a slow success.
        """
        import scripts.build_knowledge_graph as build

        class _MustNotRun:
            def __init__(self, *args, **kwargs):
                raise AssertionError(
                    "an expensive build stage ran before the budgets were checked"
                )

        for name in ("OntologyLoader", "HPOAnnotationParser", "KnowledgeGraphBuilder"):
            monkeypatch.setattr(build, name, _MustNotRun)
        self._satisfy_input_validation(tmp_dir)

        with pytest.raises(SystemExit, match="must both be supplied"):
            build.build_knowledge_graph(
                external_dir=tmp_dir, workspace=tmp_dir, generate_samples=True,
                num_train=None, num_val=None,
            )
        assert sorted(path.name for path in tmp_dir.iterdir()) == [
            "genes_to_phenotype.txt", "phenotype.hpoa"
        ]

    def test_a_build_without_samples_needs_no_budgets(self, monkeypatch, tmp_dir, wide_kg):
        """Phase one is gated on ``--generate-samples``, as the flags are.

        Without it there is no allocation and no cohort, so requiring budgets
        would refuse a legitimate graph-only build.
        """
        build = self._stub_build_stages(monkeypatch, wide_kg)
        self._satisfy_input_validation(tmp_dir)

        build.build_knowledge_graph(
            external_dir=tmp_dir, workspace=tmp_dir, generate_samples=False,
        )
        assert (tmp_dir / "kg.json").exists()
        assert not (tmp_dir / "split_manifest.json").exists()

    def test_a_sufficient_build_writes_a_manifest_bound_to_the_graph_it_wrote(
        self, monkeypatch, tmp_dir, wide_kg
    ):
        """The vouched digest reaches the manifest through the orchestration."""
        import json as _json

        from src.utils.fingerprint import file_sha256

        build = self._stub_build_stages(monkeypatch, wide_kg)
        self._satisfy_input_validation(tmp_dir)
        eligible = build_eligible_disease_profiles(wide_kg, 2)
        allocation = allocate_diseases(eligible, 0.15, seed=42)

        build.build_knowledge_graph(
            external_dir=tmp_dir, workspace=tmp_dir, generate_samples=True,
            num_train=max(30, len(allocation.train)),
            num_val=max(10, len(allocation.val)),
            val_disease_fraction=0.15, sample_seed=42,
        )
        manifest = _json.loads((tmp_dir / "split_manifest.json").read_text())
        assert manifest["allocation"]["universe_digest"] == universe_digest(eligible)
        assert manifest["disjoint"] is True

        # **The writer computed these, and they are the bytes it exported.** The
        # tensors are what a model consumes and used to be bound to nothing; this
        # is the orchestration that closes it, not a helper in isolation.
        from src.evaluation.cohort import verify_graph_artifacts
        from src.kg.artifacts import GRAPH_ARTIFACTS

        for role, filename in GRAPH_ARTIFACTS.items():
            assert manifest["artifacts"][role] == file_sha256(tmp_dir / filename)
        assert set(verify_graph_artifacts(tmp_dir)) == set(GRAPH_ARTIFACTS)
