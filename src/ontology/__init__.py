"""
SHEPHERD-Advanced Ontology Module
==================================
本體處理模組

支援的本體:
- HPO (Human Phenotype Ontology) - 人類表型
- MONDO (Mondo Disease Ontology) - 疾病
- GO (Gene Ontology) - 基因功能
- MP (Mammalian Phenotype Ontology) - 小鼠表型 (用於同源基因)

使用方式:
    # 載入本體
    from src.ontology import load_hpo, load_mondo, load_mp

    hpo = load_hpo()

    # 查詢術語
    term = hpo.get_term("HP:0001250")  # Seizure
    ancestors = hpo.get_ancestors("HP:0001250")

    # 計算相似度
    sim = hpo.compute_similarity("HP:0001250", "HP:0007359", method="lin")

    # 約束驗證
    from src.ontology import OntologyConstraints

    constraints = OntologyConstraints(hpo)
    is_valid, errors = constraints.validate_phenotype_set(phenotypes)
    cleaned = constraints.remove_redundant_ancestors(phenotypes)

版本: 1.0.0
"""

# Base classes
from src.ontology.base import (
    Ontology,
    OntologyTerm,
)

# Loader
from src.ontology.loader import (
    OntologyLoader,
    load_hpo,
    load_mondo,
    load_go,
    load_mp,
    ONTOLOGY_URLS,
)

# Constraints
from src.ontology.constraints import (
    OntologyConstraints,
    ConstraintConfig,
    ConstraintViolation,
)

__all__ = [
    # Base classes
    "Ontology",
    "OntologyTerm",
    # Loader
    "OntologyLoader",
    "load_hpo",
    "load_mondo",
    "load_go",
    "load_mp",
    "ONTOLOGY_URLS",
    # Constraints
    "OntologyConstraints",
    "ConstraintConfig",
    "ConstraintViolation",
]
