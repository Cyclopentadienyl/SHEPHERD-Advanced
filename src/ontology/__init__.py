"""
SHEPHERD-Advanced Ontology Module
=================================
本體處理模組，支援 HPO/MONDO/GO/MP 等生物醫學本體

主要功能:
- 載入 OBO 格式本體檔案
- 層次結構遍歷 (祖先、後代、LCA)
- 語義相似度計算 (Resnik, Lin, Jiang-Conrath)
- Information Content (IC) 計算
- 術語搜尋
- 約束驗證

使用範例:
    from src.ontology import OntologyLoader, Ontology

    # 載入 HPO
    loader = OntologyLoader()
    hpo = loader.load_hpo()

    # 查詢術語
    term = hpo.get_term("HP:0001250")  # Seizure
    print(f"Name: {term['name']}")

    # 獲取祖先
    ancestors = hpo.get_ancestors("HP:0001250")
    print(f"Ancestors: {ancestors}")

    # 計算語義相似度
    sim = hpo.compute_similarity("HP:0001250", "HP:0002311", method="lin")
    print(f"Similarity: {sim}")

    # 驗證表型集合
    from src.ontology import OntologyConstraintChecker
    checker = OntologyConstraintChecker(hpo)
    is_valid, violations = checker.validate_phenotype_set(["HP:0001250", "HP:0002311"])

版本: 1.0.0
"""

# Loader
from src.ontology.loader import (
    OBOTerm,
    OBOHeader,
    OBOParser,
    OntologyLoader,
    create_ontology_loader,
)

# Hierarchy operations
from src.ontology.hierarchy import Ontology

# Constraints
from src.ontology.constraints import (
    ConstraintViolation,
    OntologyConstraintChecker,
    create_constraint_checker,
)

__all__ = [
    # Data classes
    "OBOTerm",
    "OBOHeader",
    # Parser
    "OBOParser",
    # Loader
    "OntologyLoader",
    "create_ontology_loader",
    # Ontology
    "Ontology",
    # Constraints
    "ConstraintViolation",
    "OntologyConstraintChecker",
    "create_constraint_checker",
]
