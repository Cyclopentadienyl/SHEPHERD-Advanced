"""
SHEPHERD-Advanced Ontology Constraints
======================================
本體約束驗證，用於檢查推理結果是否符合本體邏輯

版本: 1.0.0
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from src.ontology.hierarchy import Ontology

logger = logging.getLogger(__name__)


# =============================================================================
# Constraint Types
# =============================================================================
class ConstraintViolation:
    """約束違規記錄"""

    def __init__(
        self,
        violation_type: str,
        message: str,
        terms: List[str],
        severity: str = "warning",  # "error", "warning", "info"
    ):
        self.violation_type = violation_type
        self.message = message
        self.terms = terms
        self.severity = severity

    def __repr__(self) -> str:
        return f"ConstraintViolation({self.violation_type}: {self.message})"


# =============================================================================
# Ontology Constraint Checker
# =============================================================================
class OntologyConstraintChecker:
    """
    本體約束檢查器

    檢查類型:
    1. 術語有效性 - 術語是否存在於本體中
    2. 廢棄術語 - 使用廢棄術語的警告
    3. 冗餘祖先 - 同時包含術語及其祖先
    4. 互斥表型 - 本體中定義的互斥關係
    """

    def __init__(self, ontology: Ontology):
        """
        Args:
            ontology: 要檢查的本體
        """
        self.ontology = ontology

    # =========================================================================
    # Validation
    # =========================================================================
    def validate_phenotype_set(
        self,
        phenotypes: List[str],
    ) -> Tuple[bool, List[ConstraintViolation]]:
        """
        驗證表型集合是否符合本體約束

        Args:
            phenotypes: HPO 術語 ID 列表

        Returns:
            (is_valid, list of violations)
            is_valid 為 True 表示沒有嚴重錯誤
        """
        violations = []
        has_errors = False

        # 1. Check term existence
        invalid_terms = self._check_term_existence(phenotypes)
        if invalid_terms:
            has_errors = True
            violations.append(ConstraintViolation(
                violation_type="invalid_term",
                message=f"Unknown terms: {', '.join(invalid_terms)}",
                terms=invalid_terms,
                severity="error",
            ))

        # Filter to valid terms for subsequent checks
        valid_phenotypes = [p for p in phenotypes if p not in invalid_terms]

        # 2. Check obsolete terms
        obsolete_terms = self._check_obsolete_terms(valid_phenotypes)
        if obsolete_terms:
            violations.append(ConstraintViolation(
                violation_type="obsolete_term",
                message=f"Obsolete terms used: {', '.join(obsolete_terms)}",
                terms=obsolete_terms,
                severity="warning",
            ))

        # 3. Check redundant ancestors
        redundant = self._check_redundant_ancestors(valid_phenotypes)
        if redundant:
            violations.append(ConstraintViolation(
                violation_type="redundant_ancestor",
                message=f"Redundant ancestor terms: {', '.join(redundant)}",
                terms=redundant,
                severity="info",
            ))

        # 4. Check logical consistency (if ontology has disjoint axioms)
        # This is optional and depends on ontology support
        # inconsistent = self._check_logical_consistency(valid_phenotypes)

        return not has_errors, violations

    def _check_term_existence(self, phenotypes: List[str]) -> List[str]:
        """檢查術語是否存在"""
        invalid = []
        for term_id in phenotypes:
            if not self.ontology.has_term(term_id):
                invalid.append(term_id)
        return invalid

    def _check_obsolete_terms(self, phenotypes: List[str]) -> List[str]:
        """檢查廢棄術語"""
        obsolete = []
        for term_id in phenotypes:
            if self.ontology.is_obsolete(term_id):
                obsolete.append(term_id)
        return obsolete

    def _check_redundant_ancestors(self, phenotypes: List[str]) -> List[str]:
        """
        檢查冗餘祖先

        如果 A 是 B 的祖先，同時包含 A 和 B 是冗餘的
        """
        phenotype_set = set(phenotypes)
        redundant = []

        for term in phenotypes:
            ancestors = self.ontology.get_ancestors(term)
            overlap = ancestors & phenotype_set
            if overlap:
                redundant.extend(overlap)

        return list(set(redundant))

    # =========================================================================
    # Phenotype Set Operations
    # =========================================================================
    def remove_redundant_ancestors(
        self,
        phenotypes: List[str],
    ) -> List[str]:
        """
        移除冗餘的祖先表型

        保留最具體的表型 (即沒有後代在集合中)

        Args:
            phenotypes: HPO 術語 ID 列表

        Returns:
            去除冗餘後的術語列表
        """
        phenotype_set = set(phenotypes)
        to_remove = set()

        for term in phenotypes:
            ancestors = self.ontology.get_ancestors(term)
            to_remove.update(ancestors & phenotype_set)

        return [p for p in phenotypes if p not in to_remove]

    def expand_to_ancestors(
        self,
        phenotypes: List[str],
        max_depth: Optional[int] = None,
    ) -> Set[str]:
        """
        擴展表型集合以包含所有祖先

        用於表型推理 (如果患者有某表型，也隱含有其祖先表型)

        Args:
            phenotypes: HPO 術語 ID 列表
            max_depth: 最大深度限制

        Returns:
            擴展後的術語集合
        """
        expanded = set(phenotypes)

        for term in phenotypes:
            ancestors = self.ontology.get_ancestors(term)
            expanded.update(ancestors)

        return expanded

    def get_implied_phenotypes(
        self,
        phenotypes: List[str],
    ) -> Set[str]:
        """
        獲取隱含的表型 (基於本體推理)

        Args:
            phenotypes: 顯式表型列表

        Returns:
            隱含的表型集合 (所有祖先)
        """
        return self.expand_to_ancestors(phenotypes) - set(phenotypes)

    # =========================================================================
    # Obsolete Term Handling
    # =========================================================================
    def replace_obsolete_terms(
        self,
        phenotypes: List[str],
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        替換廢棄術語為其推薦替代

        Args:
            phenotypes: HPO 術語 ID 列表

        Returns:
            (updated_phenotypes, replacements_made)
        """
        updated = []
        replacements = {}

        for term in phenotypes:
            if self.ontology.is_obsolete(term):
                replacement = self.ontology.get_replacement(term)
                if replacement and self.ontology.has_term(replacement):
                    updated.append(replacement)
                    replacements[term] = replacement
                    logger.info(f"Replaced obsolete term {term} with {replacement}")
                else:
                    # No valid replacement, keep original with warning
                    updated.append(term)
                    logger.warning(f"Obsolete term {term} has no valid replacement")
            else:
                updated.append(term)

        return updated, replacements

    # =========================================================================
    # Phenotype Profile Validation
    # =========================================================================
    def validate_for_diagnosis(
        self,
        phenotypes: List[str],
        disease_id: str,
        disease_phenotypes: List[str],
    ) -> Tuple[float, List[str]]:
        """
        驗證患者表型與疾病表型的一致性

        Args:
            phenotypes: 患者表型
            disease_id: 疾病 ID
            disease_phenotypes: 疾病的典型表型

        Returns:
            (consistency_score, inconsistencies)
        """
        patient_expanded = self.expand_to_ancestors(phenotypes)
        disease_expanded = self.expand_to_ancestors(disease_phenotypes)

        # Calculate overlap
        overlap = patient_expanded & disease_expanded

        # Consistency score based on overlap
        if not disease_expanded:
            return 0.0, []

        # What percentage of disease phenotypes are covered
        coverage = len(overlap & set(disease_phenotypes)) / len(disease_phenotypes)

        # Find inconsistencies (this would require disjoint axioms from ontology)
        # For now, just return empty list
        inconsistencies = []

        return coverage, inconsistencies


# =============================================================================
# Factory Function
# =============================================================================
def create_constraint_checker(ontology: Ontology) -> OntologyConstraintChecker:
    """
    工廠函數: 創建約束檢查器

    Args:
        ontology: 本體實例

    Returns:
        OntologyConstraintChecker 實例
    """
    return OntologyConstraintChecker(ontology)
