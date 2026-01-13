"""
Unit Tests for Ontology Module
==============================
測試本體載入、層次結構操作、語義相似度計算等功能
"""
import pytest
from pathlib import Path

from src.ontology import (
    OBOParser,
    OntologyLoader,
    Ontology,
    OntologyConstraintChecker,
    create_ontology_loader,
    create_constraint_checker,
)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def fixtures_dir() -> Path:
    """獲取 fixtures 目錄路徑"""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def mini_hpo_path(fixtures_dir: Path) -> Path:
    """獲取 mini HPO 檔案路徑"""
    return fixtures_dir / "mini_hpo.obo"


@pytest.fixture
def parser() -> OBOParser:
    """創建 OBO 解析器"""
    return OBOParser()


@pytest.fixture
def ontology(mini_hpo_path: Path, parser: OBOParser) -> Ontology:
    """載入測試用本體"""
    header, terms = parser.parse_file(mini_hpo_path)
    return Ontology(header, terms)


@pytest.fixture
def constraint_checker(ontology: Ontology) -> OntologyConstraintChecker:
    """創建約束檢查器"""
    return OntologyConstraintChecker(ontology)


# =============================================================================
# Test OBO Parser
# =============================================================================
class TestOBOParser:
    """測試 OBO 解析器"""

    def test_parse_file(self, mini_hpo_path: Path, parser: OBOParser):
        """測試解析 OBO 檔案"""
        header, terms = parser.parse_file(mini_hpo_path)

        assert header is not None
        assert header.format_version == "1.4"
        assert header.ontology == "hp"

        assert len(terms) > 0
        assert "HP:0001250" in terms  # Seizure

    def test_parse_term_attributes(self, mini_hpo_path: Path, parser: OBOParser):
        """測試解析術語屬性"""
        _, terms = parser.parse_file(mini_hpo_path)

        seizure = terms.get("HP:0001250")
        assert seizure is not None
        assert seizure.name == "Seizure"
        assert "HP:0012638" in seizure.is_a
        assert len(seizure.synonyms) > 0

    def test_parse_obsolete_term(self, mini_hpo_path: Path, parser: OBOParser):
        """測試解析廢棄術語"""
        _, terms = parser.parse_file(mini_hpo_path)

        obsolete = terms.get("HP:9999999")
        assert obsolete is not None
        assert obsolete.is_obsolete is True
        assert obsolete.replaced_by == "HP:0001250"


# =============================================================================
# Test Ontology Properties
# =============================================================================
class TestOntologyProperties:
    """測試本體屬性"""

    def test_name_and_version(self, ontology: Ontology):
        """測試名稱和版本"""
        assert ontology.name == "hp"
        assert ontology.version is not None

    def test_num_terms(self, ontology: Ontology):
        """測試術語數量 (不含廢棄)"""
        # mini_hpo.obo has 11 terms, 1 is obsolete
        assert ontology.num_terms == 10

    def test_root_terms(self, ontology: Ontology):
        """測試根節點"""
        roots = ontology.root_terms
        assert "HP:0000001" in roots  # All


# =============================================================================
# Test Term Access
# =============================================================================
class TestTermAccess:
    """測試術語訪問"""

    def test_get_term(self, ontology: Ontology):
        """測試獲取術語"""
        term = ontology.get_term("HP:0001250")
        assert term is not None
        assert term["name"] == "Seizure"
        assert term["id"] == "HP:0001250"

    def test_get_term_not_found(self, ontology: Ontology):
        """測試獲取不存在的術語"""
        term = ontology.get_term("HP:INVALID")
        assert term is None

    def test_get_term_name(self, ontology: Ontology):
        """測試獲取術語名稱"""
        name = ontology.get_term_name("HP:0001250")
        assert name == "Seizure"

    def test_has_term(self, ontology: Ontology):
        """測試術語存在性"""
        assert ontology.has_term("HP:0001250") is True
        assert ontology.has_term("HP:INVALID") is False

    def test_is_obsolete(self, ontology: Ontology):
        """測試廢棄狀態"""
        assert ontology.is_obsolete("HP:9999999") is True
        assert ontology.is_obsolete("HP:0001250") is False


# =============================================================================
# Test Hierarchy Traversal
# =============================================================================
class TestHierarchyTraversal:
    """測試層次結構遍歷"""

    def test_get_parents(self, ontology: Ontology):
        """測試獲取父節點"""
        parents = ontology.get_parents("HP:0001250")  # Seizure
        assert "HP:0012638" in parents  # Abnormal nervous system physiology

    def test_get_children(self, ontology: Ontology):
        """測試獲取子節點"""
        children = ontology.get_children("HP:0012638")  # Abnormal nervous system physiology
        assert "HP:0001250" in children  # Seizure
        assert "HP:0002311" in children  # Incoordination
        assert "HP:0001252" in children  # Hypotonia

    def test_get_ancestors(self, ontology: Ontology):
        """測試獲取所有祖先"""
        ancestors = ontology.get_ancestors("HP:0001250")  # Seizure

        assert "HP:0012638" in ancestors  # Abnormal nervous system physiology
        assert "HP:0000707" in ancestors  # Abnormality of the nervous system
        assert "HP:0000118" in ancestors  # Phenotypic abnormality
        assert "HP:0000001" in ancestors  # All

        # Should not include self
        assert "HP:0001250" not in ancestors

    def test_get_ancestors_include_self(self, ontology: Ontology):
        """測試獲取祖先 (包含自身)"""
        ancestors = ontology.get_ancestors("HP:0001250", include_self=True)
        assert "HP:0001250" in ancestors

    def test_get_descendants(self, ontology: Ontology):
        """測試獲取所有後代"""
        descendants = ontology.get_descendants("HP:0000707")  # Abnormality of the nervous system

        assert "HP:0012638" in descendants  # Abnormal nervous system physiology
        assert "HP:0001250" in descendants  # Seizure
        assert "HP:0002311" in descendants  # Incoordination
        assert "HP:0001251" in descendants  # Ataxia

    def test_get_depth(self, ontology: Ontology):
        """測試獲取深度"""
        # HP:0000001 (All) is root, depth = 0
        assert ontology.get_depth("HP:0000001") == 0

        # HP:0000118 -> HP:0000001, depth = 1
        assert ontology.get_depth("HP:0000118") == 1

        # HP:0001250 -> HP:0012638 -> HP:0000707 -> HP:0000118 -> HP:0000001, depth = 4
        assert ontology.get_depth("HP:0001250") == 4


# =============================================================================
# Test LCA (Lowest Common Ancestor)
# =============================================================================
class TestLCA:
    """測試最低共同祖先"""

    def test_get_common_ancestors(self, ontology: Ontology):
        """測試共同祖先"""
        # Seizure and Ataxia both under Abnormal nervous system physiology
        common = ontology.get_common_ancestors("HP:0001250", "HP:0001251")

        assert "HP:0012638" in common  # Abnormal nervous system physiology
        assert "HP:0000707" in common  # Abnormality of the nervous system
        assert "HP:0000118" in common  # Phenotypic abnormality
        assert "HP:0000001" in common  # All

    def test_get_lowest_common_ancestors(self, ontology: Ontology):
        """測試最低共同祖先"""
        # Seizure and Ataxia
        lcas = ontology.get_lowest_common_ancestors("HP:0001250", "HP:0001251")

        # LCA should be HP:0012638 (Abnormal nervous system physiology)
        assert "HP:0012638" in lcas

    def test_lca_same_term(self, ontology: Ontology):
        """測試相同術語的 LCA"""
        lcas = ontology.get_lowest_common_ancestors("HP:0001250", "HP:0001250")
        assert "HP:0001250" in lcas


# =============================================================================
# Test Information Content
# =============================================================================
class TestInformationContent:
    """測試 Information Content"""

    def test_ic_computed(self, ontology: Ontology):
        """測試 IC 計算"""
        ic_seizure = ontology.get_information_content("HP:0001250")
        ic_root = ontology.get_information_content("HP:0000001")

        # More specific term should have higher IC
        assert ic_seizure > ic_root

    def test_ic_hierarchy(self, ontology: Ontology):
        """測試 IC 層次關係"""
        ic_seizure = ontology.get_information_content("HP:0001250")
        ic_nervous = ontology.get_information_content("HP:0000707")
        ic_pheno = ontology.get_information_content("HP:0000118")

        # More specific terms should have higher IC
        assert ic_seizure > ic_nervous > ic_pheno


# =============================================================================
# Test Semantic Similarity
# =============================================================================
class TestSemanticSimilarity:
    """測試語義相似度"""

    def test_similarity_same_term(self, ontology: Ontology):
        """測試相同術語的相似度"""
        # Lin similarity of same term should be 1.0
        sim = ontology.compute_similarity("HP:0001250", "HP:0001250", method="lin")
        assert sim == 1.0

    def test_similarity_resnik(self, ontology: Ontology):
        """測試 Resnik 相似度"""
        sim = ontology.compute_similarity("HP:0001250", "HP:0001251", method="resnik")
        assert sim > 0  # Should have some similarity (share common ancestor)

    def test_similarity_lin(self, ontology: Ontology):
        """測試 Lin 相似度"""
        sim = ontology.compute_similarity("HP:0001250", "HP:0001251", method="lin")
        assert 0 <= sim <= 1  # Lin similarity is normalized

    def test_similarity_jiang(self, ontology: Ontology):
        """測試 Jiang-Conrath 相似度"""
        sim = ontology.compute_similarity("HP:0001250", "HP:0001251", method="jiang")
        assert 0 <= sim <= 1

    def test_similarity_jaccard(self, ontology: Ontology):
        """測試 Jaccard 相似度"""
        sim = ontology.compute_similarity("HP:0001250", "HP:0001251", method="jaccard")
        assert 0 <= sim <= 1

    def test_similarity_sibling_vs_distant(self, ontology: Ontology):
        """測試兄弟節點 vs 遠親節點的相似度"""
        # Seizure and Incoordination are siblings (both under HP:0012638)
        sim_sibling = ontology.compute_similarity("HP:0001250", "HP:0002311", method="lin")

        # Seizure and Abnormality of the head (different branch)
        sim_distant = ontology.compute_similarity("HP:0001250", "HP:0000234", method="lin")

        # Siblings should be more similar
        assert sim_sibling > sim_distant


# =============================================================================
# Test Set Similarity
# =============================================================================
class TestSetSimilarity:
    """測試集合相似度"""

    def test_set_similarity_bma(self, ontology: Ontology):
        """測試 Best Match Average 相似度"""
        terms1 = ["HP:0001250", "HP:0001252"]  # Seizure, Hypotonia
        terms2 = ["HP:0001251", "HP:0002311"]  # Ataxia, Incoordination

        sim = ontology.compute_set_similarity(terms1, terms2, method="bma")
        assert sim > 0

    def test_set_similarity_identical(self, ontology: Ontology):
        """測試相同集合的相似度"""
        terms = ["HP:0001250", "HP:0001252"]

        sim = ontology.compute_set_similarity(terms, terms, method="bma")
        # Should be very high (though not necessarily 1.0 due to BMA calculation)
        assert sim > 0.9


# =============================================================================
# Test Search
# =============================================================================
class TestSearch:
    """測試搜尋功能"""

    def test_search_exact_match(self, ontology: Ontology):
        """測試精確匹配"""
        results = ontology.search("Seizure")

        assert len(results) > 0
        assert results[0][0] == "HP:0001250"
        assert results[0][2] == 1.0  # Exact match score

    def test_search_prefix_match(self, ontology: Ontology):
        """測試前綴匹配"""
        results = ontology.search("Seiz")

        assert len(results) > 0
        # Should find Seizure
        ids = [r[0] for r in results]
        assert "HP:0001250" in ids

    def test_search_contains_match(self, ontology: Ontology):
        """測試包含匹配"""
        results = ontology.search("nervous")

        assert len(results) > 0
        ids = [r[0] for r in results]
        assert "HP:0000707" in ids  # Abnormality of the nervous system

    def test_search_synonym(self, ontology: Ontology):
        """測試同義詞搜尋"""
        results = ontology.search("Convulsion", include_synonyms=True)

        assert len(results) > 0
        ids = [r[0] for r in results]
        assert "HP:0001250" in ids  # Seizure (has synonym "Convulsion")


# =============================================================================
# Test Constraint Checker
# =============================================================================
class TestConstraintChecker:
    """測試約束檢查器"""

    def test_validate_valid_phenotypes(self, constraint_checker: OntologyConstraintChecker):
        """測試驗證有效表型"""
        phenotypes = ["HP:0001250", "HP:0001251"]
        is_valid, violations = constraint_checker.validate_phenotype_set(phenotypes)

        assert is_valid is True
        # May have info-level violations (redundant ancestors) but no errors
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0

    def test_validate_invalid_phenotypes(self, constraint_checker: OntologyConstraintChecker):
        """測試驗證無效表型"""
        phenotypes = ["HP:INVALID", "HP:0001250"]
        is_valid, violations = constraint_checker.validate_phenotype_set(phenotypes)

        assert is_valid is False
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) > 0
        assert "HP:INVALID" in errors[0].terms

    def test_validate_obsolete_phenotypes(self, constraint_checker: OntologyConstraintChecker):
        """測試驗證廢棄表型"""
        phenotypes = ["HP:9999999", "HP:0001250"]
        is_valid, violations = constraint_checker.validate_phenotype_set(phenotypes)

        # Should be valid (obsolete is warning, not error)
        assert is_valid is True
        warnings = [v for v in violations if v.severity == "warning"]
        assert len(warnings) > 0

    def test_remove_redundant_ancestors(self, constraint_checker: OntologyConstraintChecker):
        """測試移除冗餘祖先"""
        # Include both Seizure and its ancestor Abnormal nervous system physiology
        phenotypes = ["HP:0001250", "HP:0012638"]
        cleaned = constraint_checker.remove_redundant_ancestors(phenotypes)

        # Should keep only Seizure (more specific)
        assert "HP:0001250" in cleaned
        assert "HP:0012638" not in cleaned

    def test_expand_to_ancestors(self, constraint_checker: OntologyConstraintChecker):
        """測試擴展到祖先"""
        phenotypes = ["HP:0001250"]  # Seizure
        expanded = constraint_checker.expand_to_ancestors(phenotypes)

        assert "HP:0001250" in expanded
        assert "HP:0012638" in expanded  # Abnormal nervous system physiology
        assert "HP:0000001" in expanded  # All

    def test_replace_obsolete_terms(self, constraint_checker: OntologyConstraintChecker):
        """測試替換廢棄術語"""
        phenotypes = ["HP:9999999"]  # Obsolete, replaced by HP:0001250
        updated, replacements = constraint_checker.replace_obsolete_terms(phenotypes)

        assert "HP:0001250" in updated
        assert "HP:9999999" in replacements
        assert replacements["HP:9999999"] == "HP:0001250"


# =============================================================================
# Test Factory Functions
# =============================================================================
class TestFactoryFunctions:
    """測試工廠函數"""

    def test_create_ontology_loader(self, tmp_path: Path):
        """測試創建本體載入器"""
        loader = create_ontology_loader(cache_dir=tmp_path)
        assert loader is not None
        assert loader.cache_dir == tmp_path

    def test_create_constraint_checker(self, ontology: Ontology):
        """測試創建約束檢查器"""
        checker = create_constraint_checker(ontology)
        assert checker is not None
        assert checker.ontology == ontology


# =============================================================================
# Integration Test with Loader
# =============================================================================
class TestOntologyLoader:
    """測試本體載入器"""

    def test_load_from_file(self, mini_hpo_path: Path):
        """測試從檔案載入"""
        loader = OntologyLoader()
        ontology = loader.load(mini_hpo_path)

        assert ontology is not None
        assert ontology.num_terms > 0
        assert ontology.has_term("HP:0001250")
