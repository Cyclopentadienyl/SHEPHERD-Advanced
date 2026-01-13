"""
Ontology Constraints
====================
本體約束驗證模組

用於:
1. 驗證表型集合的有效性
2. 移除冗餘祖先表型
3. 獲取隱含表型
4. 驗證推理結果是否符合本體約束

版本: 1.0.0
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core import OntologyConstraintProtocol
from src.ontology.base import Ontology

logger = logging.getLogger(__name__)


# =============================================================================
# Constraint Configuration
# =============================================================================
@dataclass
class ConstraintConfig:
    """約束配置"""
    # Validation settings
    allow_obsolete_terms: bool = False
    allow_unknown_terms: bool = False

    # Redundancy removal
    remove_ancestor_phenotypes: bool = True

    # Inference settings
    infer_ancestor_phenotypes: bool = False
    max_inferred_depth: int = 3

    # Severity levels
    obsolete_term_severity: str = "warning"  # "error", "warning", "ignore"
    unknown_term_severity: str = "error"


# =============================================================================
# Constraint Violation
# =============================================================================
@dataclass
class ConstraintViolation:
    """約束違規記錄"""
    term_id: str
    violation_type: str  # "obsolete", "unknown", "redundant", "invalid"
    message: str
    severity: str = "warning"  # "error", "warning", "info"
    suggested_replacement: Optional[str] = None


# =============================================================================
# Ontology Constraint Checker
# =============================================================================
class OntologyConstraints(OntologyConstraintProtocol):
    """
    本體約束檢查器

    使用方式:
        constraints = OntologyConstraints(hpo)
        is_valid, violations = constraints.validate_phenotype_set(phenotypes)
        cleaned = constraints.remove_redundant_ancestors(phenotypes)
    """

    def __init__(
        self,
        ontology: Ontology,
        config: Optional[ConstraintConfig] = None,
    ):
        self.ontology = ontology
        self.config = config or ConstraintConfig()

        logger.debug(f"OntologyConstraints initialized for {ontology.name}")

    # =========================================================================
    # Validation
    # =========================================================================
    def validate_phenotype_set(
        self,
        phenotypes: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        驗證表型集合是否符合本體約束

        Args:
            phenotypes: 表型 ID 列表

        Returns:
            (is_valid, list of violation messages)
        """
        violations: List[ConstraintViolation] = []

        for phenotype in phenotypes:
            # Check if term exists
            if not self.ontology.has_term(phenotype):
                if not self.config.allow_unknown_terms:
                    violations.append(ConstraintViolation(
                        term_id=phenotype,
                        violation_type="unknown",
                        message=f"Unknown term: {phenotype}",
                        severity=self.config.unknown_term_severity,
                    ))
                continue

            # Check if term is obsolete
            term = self.ontology.get_term_object(phenotype)
            if term and term.is_obsolete:
                if not self.config.allow_obsolete_terms:
                    violations.append(ConstraintViolation(
                        term_id=phenotype,
                        violation_type="obsolete",
                        message=f"Obsolete term: {phenotype} ({term.name})",
                        severity=self.config.obsolete_term_severity,
                    ))

        # Check for redundant ancestors
        if self.config.remove_ancestor_phenotypes:
            redundant = self._find_redundant_ancestors(phenotypes)
            for term_id in redundant:
                term = self.ontology.get_term_object(term_id)
                name = term.name if term else term_id
                violations.append(ConstraintViolation(
                    term_id=term_id,
                    violation_type="redundant",
                    message=f"Redundant ancestor term: {term_id} ({name})",
                    severity="info",
                ))

        # Determine overall validity
        has_errors = any(v.severity == "error" for v in violations)
        is_valid = not has_errors

        # Convert to string messages
        messages = [v.message for v in violations]

        return is_valid, messages

    def validate_term(self, term_id: str) -> Tuple[bool, Optional[str]]:
        """
        驗證單個術語

        Returns:
            (is_valid, error_message)
        """
        if not self.ontology.has_term(term_id):
            return False, f"Unknown term: {term_id}"

        term = self.ontology.get_term_object(term_id)
        if term and term.is_obsolete and not self.config.allow_obsolete_terms:
            return False, f"Obsolete term: {term_id}"

        return True, None

    # =========================================================================
    # Redundancy Removal
    # =========================================================================
    def remove_redundant_ancestors(
        self,
        phenotypes: List[str],
    ) -> List[str]:
        """
        移除冗餘的祖先表型

        如果一個表型是另一個表型的祖先，則移除祖先
        (保留更具體的表型)

        Example:
            Input: ["HP:0001250", "HP:0007359"]  # Seizure, Focal seizure
            Output: ["HP:0007359"]  # 只保留 Focal seizure (更具體)
        """
        if not phenotypes:
            return []

        # Filter valid phenotypes
        valid_phenotypes = [
            p for p in phenotypes
            if self.ontology.has_term(p)
        ]

        if not valid_phenotypes:
            return []

        # Find all ancestors of each phenotype
        phenotype_set = set(valid_phenotypes)
        all_ancestors: Set[str] = set()

        for phenotype in valid_phenotypes:
            ancestors = self.ontology.get_ancestors(phenotype, include_self=False)
            all_ancestors.update(ancestors)

        # Remove phenotypes that are ancestors of other phenotypes
        result = [
            p for p in valid_phenotypes
            if p not in all_ancestors
        ]

        if len(result) < len(valid_phenotypes):
            logger.debug(
                f"Removed {len(valid_phenotypes) - len(result)} redundant ancestors"
            )

        return result

    def _find_redundant_ancestors(self, phenotypes: List[str]) -> Set[str]:
        """找出冗餘的祖先表型"""
        valid_phenotypes = [
            p for p in phenotypes
            if self.ontology.has_term(p)
        ]

        phenotype_set = set(valid_phenotypes)
        all_ancestors: Set[str] = set()

        for phenotype in valid_phenotypes:
            ancestors = self.ontology.get_ancestors(phenotype, include_self=False)
            all_ancestors.update(ancestors)

        # Redundant are phenotypes that are in ancestor set
        return phenotype_set & all_ancestors

    # =========================================================================
    # Phenotype Inference
    # =========================================================================
    def get_implied_phenotypes(
        self,
        phenotypes: List[str],
    ) -> Set[str]:
        """
        獲取隱含的表型 (基於本體推理)

        根據 IS_A 關係，子表型隱含其所有祖先表型

        Args:
            phenotypes: 表型 ID 列表

        Returns:
            隱含表型集合 (包含輸入表型及其所有祖先)
        """
        implied: Set[str] = set()

        for phenotype in phenotypes:
            if self.ontology.has_term(phenotype):
                # Add the phenotype itself
                implied.add(phenotype)

                # Add ancestors up to max depth
                if self.config.infer_ancestor_phenotypes:
                    ancestors = self._get_ancestors_up_to_depth(
                        phenotype,
                        self.config.max_inferred_depth,
                    )
                    implied.update(ancestors)
                else:
                    # Add all ancestors
                    implied.update(
                        self.ontology.get_ancestors(phenotype, include_self=False)
                    )

        return implied

    def _get_ancestors_up_to_depth(
        self,
        term_id: str,
        max_depth: int,
    ) -> Set[str]:
        """獲取指定深度內的祖先"""
        ancestors: Set[str] = set()
        current_level = {term_id}

        for _ in range(max_depth):
            next_level: Set[str] = set()
            for term in current_level:
                parents = self.ontology.get_parents(term)
                next_level.update(parents)
            ancestors.update(next_level)
            current_level = next_level

            if not current_level:
                break

        return ancestors

    # =========================================================================
    # Phenotype Normalization
    # =========================================================================
    def normalize_phenotypes(
        self,
        phenotypes: List[str],
        remove_redundant: bool = True,
        resolve_obsolete: bool = True,
    ) -> List[str]:
        """
        標準化表型列表

        Args:
            phenotypes: 表型 ID 列表
            remove_redundant: 是否移除冗餘祖先
            resolve_obsolete: 是否嘗試解析過時術語

        Returns:
            標準化後的表型列表
        """
        normalized = []

        for phenotype in phenotypes:
            # Skip unknown terms
            if not self.ontology.has_term(phenotype):
                logger.warning(f"Skipping unknown term: {phenotype}")
                continue

            term = self.ontology.get_term_object(phenotype)
            if term is None:
                continue

            # Handle obsolete terms
            if term.is_obsolete and resolve_obsolete:
                # Try to find replacement in alt_ids or consider
                # For now, just skip obsolete terms
                logger.warning(f"Skipping obsolete term: {phenotype}")
                continue

            normalized.append(phenotype)

        # Remove redundant ancestors
        if remove_redundant:
            normalized = self.remove_redundant_ancestors(normalized)

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for p in normalized:
            if p not in seen:
                seen.add(p)
                result.append(p)

        return result

    # =========================================================================
    # Specificity Analysis
    # =========================================================================
    def get_specificity_score(self, term_id: str) -> float:
        """
        獲取術語的特異性分數

        基於 Information Content，值越高越具體

        Returns:
            特異性分數 [0, 1]
        """
        if not self.ontology.has_term(term_id):
            return 0.0

        ic = self.ontology.get_information_content(term_id)

        # Normalize IC to [0, 1] using max possible IC
        max_ic = self.ontology.get_information_content(
            next(iter(self.ontology.all_term_ids()))
        )

        # Find actual max IC
        # (This is expensive, so we use an approximation)
        # Max IC is typically around 10-15 for medical ontologies
        max_ic_approx = 12.0

        return min(ic / max_ic_approx, 1.0)

    def rank_by_specificity(
        self,
        phenotypes: List[str],
        descending: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        按特異性排序表型

        Returns:
            List of (term_id, specificity_score) sorted by score
        """
        scored = [
            (p, self.get_specificity_score(p))
            for p in phenotypes
            if self.ontology.has_term(p)
        ]

        return sorted(scored, key=lambda x: x[1], reverse=descending)

    # =========================================================================
    # Coverage Analysis
    # =========================================================================
    def compute_phenotype_coverage(
        self,
        patient_phenotypes: List[str],
        disease_phenotypes: List[str],
    ) -> Dict[str, Any]:
        """
        計算表型覆蓋率

        用於評估患者表型與疾病表型的匹配程度

        Returns:
            {
                "exact_matches": [...],
                "ancestor_matches": [...],
                "descendant_matches": [...],
                "coverage_score": float,
            }
        """
        patient_set = set(patient_phenotypes)
        disease_set = set(disease_phenotypes)

        # Exact matches
        exact = patient_set & disease_set

        # Patient phenotypes that are ancestors of disease phenotypes
        # (Patient has more general symptom)
        ancestor_matches = set()
        for patient_p in patient_set:
            if patient_p in exact:
                continue
            patient_ancestors = self.ontology.get_ancestors(patient_p, include_self=False)
            if patient_ancestors & disease_set:
                ancestor_matches.add(patient_p)

        # Patient phenotypes that are descendants of disease phenotypes
        # (Patient has more specific symptom - this is a good match!)
        descendant_matches = set()
        for patient_p in patient_set:
            if patient_p in exact:
                continue
            patient_descendants = self.ontology.get_descendants(patient_p, include_self=False)
            if patient_descendants & disease_set:
                descendant_matches.add(patient_p)

        # Also check if patient phenotype is a descendant of any disease phenotype
        for patient_p in patient_set:
            if patient_p in exact or patient_p in descendant_matches:
                continue
            patient_ancestors = self.ontology.get_ancestors(patient_p, include_self=False)
            for disease_p in disease_set:
                if disease_p in patient_ancestors:
                    descendant_matches.add(patient_p)
                    break

        # Compute coverage score
        if not disease_set:
            coverage_score = 0.0
        else:
            # Weighted score: exact > descendant > ancestor
            exact_weight = 1.0
            descendant_weight = 0.8  # More specific is good
            ancestor_weight = 0.4   # Less specific is partial match

            score = (
                len(exact) * exact_weight +
                len(descendant_matches) * descendant_weight +
                len(ancestor_matches) * ancestor_weight
            )
            coverage_score = score / len(disease_set)

        return {
            "exact_matches": list(exact),
            "ancestor_matches": list(ancestor_matches),
            "descendant_matches": list(descendant_matches),
            "unmatched_patient": list(
                patient_set - exact - ancestor_matches - descendant_matches
            ),
            "unmatched_disease": list(
                disease_set - exact -
                {d for d in disease_set if any(
                    d in self.ontology.get_ancestors(p, include_self=False)
                    for p in patient_set
                )}
            ),
            "coverage_score": min(coverage_score, 1.0),
        }
