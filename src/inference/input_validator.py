"""
# ==============================================================================
# Module: src/inference/input_validator.py
# ==============================================================================
# Purpose: Validate patient input for diagnosis pipeline
#
# Dependencies:
#   - External: None (pure Python)
#   - Internal: src.core.types
#
# Input:
#   - PatientPhenotypes: Patient phenotype data to validate
#   - OntologyProtocol (optional): For semantic validation
#
# Output:
#   - ValidationResult: Validation status with warnings/errors
#
# Design Notes:
#   - Supports HPO phenotype ID format validation
#   - Optional semantic validation against ontology
#   - Returns warnings for non-fatal issues
#   - Production-ready error messages
# ==============================================================================
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, TYPE_CHECKING

from src.core.types import PatientPhenotypes

if TYPE_CHECKING:
    from src.core.protocols import OntologyProtocol

logger = logging.getLogger(__name__)


# ==============================================================================
# Validation Result
# ==============================================================================
@dataclass
class ValidationResult:
    """Result of input validation."""

    is_valid: bool
    validated_phenotypes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.is_valid


# ==============================================================================
# Input Validator
# ==============================================================================
class InputValidator:
    """
    Validates patient input for the diagnosis pipeline.

    Performs:
    - Format validation (HPO ID format)
    - Semantic validation (against ontology if provided)
    - Duplicate detection
    - Range checks

    Usage:
        validator = InputValidator()
        result = validator.validate_patient_input(patient_phenotypes)
        if result.is_valid:
            # proceed with diagnosis
    """

    # HPO ID pattern: HP:XXXXXXX (7 digits)
    HPO_PATTERN = re.compile(r"^HP:\d{7}$")

    # Alternate patterns for flexibility
    HPO_PATTERN_FLEXIBLE = re.compile(r"^HP:\d{4,7}$")

    def __init__(
        self,
        strict_hpo_format: bool = False,
        min_phenotypes: int = 1,
        max_phenotypes: int = 100,
        allow_duplicates: bool = False,
    ):
        """
        Initialize validator.

        Args:
            strict_hpo_format: Require exactly 7 digits in HPO IDs
            min_phenotypes: Minimum required phenotypes
            max_phenotypes: Maximum allowed phenotypes
            allow_duplicates: Allow duplicate phenotype IDs
        """
        self.strict_hpo_format = strict_hpo_format
        self.min_phenotypes = min_phenotypes
        self.max_phenotypes = max_phenotypes
        self.allow_duplicates = allow_duplicates

        self._hpo_pattern = (
            self.HPO_PATTERN if strict_hpo_format else self.HPO_PATTERN_FLEXIBLE
        )

    def validate_patient_input(
        self,
        patient_input: PatientPhenotypes,
        ontology: Optional["OntologyProtocol"] = None,
    ) -> ValidationResult:
        """
        Validate complete patient input.

        Args:
            patient_input: Patient phenotype data
            ontology: Optional ontology for semantic validation

        Returns:
            ValidationResult with validation status and details
        """
        errors: List[str] = []
        warnings: List[str] = []
        validated_phenotypes: List[str] = []

        # Validate patient_id
        if not patient_input.patient_id:
            errors.append("patient_id is required")
        elif not patient_input.patient_id.strip():
            errors.append("patient_id cannot be empty or whitespace")

        # Validate phenotype list exists
        if not patient_input.phenotypes:
            errors.append("phenotypes list is required and cannot be empty")
            return ValidationResult(
                is_valid=False,
                validated_phenotypes=[],
                warnings=warnings,
                errors=errors,
            )

        # Validate phenotype count
        if len(patient_input.phenotypes) < self.min_phenotypes:
            errors.append(
                f"At least {self.min_phenotypes} phenotype(s) required, "
                f"got {len(patient_input.phenotypes)}"
            )

        if len(patient_input.phenotypes) > self.max_phenotypes:
            warnings.append(
                f"More than {self.max_phenotypes} phenotypes provided, "
                f"using first {self.max_phenotypes}"
            )

        # Validate individual phenotypes
        seen_phenotypes: Set[str] = set()
        phenotypes_to_validate = patient_input.phenotypes[:self.max_phenotypes]

        for pheno_id in phenotypes_to_validate:
            # Check for duplicates
            if pheno_id in seen_phenotypes:
                if not self.allow_duplicates:
                    warnings.append(f"Duplicate phenotype ignored: {pheno_id}")
                    continue
            seen_phenotypes.add(pheno_id)

            # Validate format
            format_result = self.validate_phenotype_format(pheno_id)
            if not format_result.is_valid:
                warnings.extend(format_result.warnings)
                errors.extend(format_result.errors)
                continue

            # Validate against ontology if provided
            if ontology is not None:
                semantic_result = self.validate_phenotype_semantic(
                    pheno_id, ontology
                )
                if not semantic_result.is_valid:
                    warnings.extend(semantic_result.warnings)
                    # Don't add to errors - just warn
                    continue

            validated_phenotypes.append(pheno_id)

        # Check if any valid phenotypes remain
        if not validated_phenotypes and not errors:
            errors.append("No valid phenotypes after validation")

        # Validate phenotype confidences if provided
        if patient_input.phenotype_confidences:
            conf_result = self._validate_confidences(
                patient_input.phenotypes,
                patient_input.phenotype_confidences,
            )
            warnings.extend(conf_result.warnings)
            errors.extend(conf_result.errors)

        return ValidationResult(
            is_valid=len(errors) == 0 and len(validated_phenotypes) > 0,
            validated_phenotypes=validated_phenotypes,
            warnings=warnings,
            errors=errors,
        )

    def validate_phenotypes(
        self,
        phenotypes: List[str],
        ontology: Optional["OntologyProtocol"] = None,
    ) -> ValidationResult:
        """
        Validate a list of phenotype IDs.

        Args:
            phenotypes: List of HPO IDs
            ontology: Optional ontology for semantic validation

        Returns:
            ValidationResult
        """
        errors: List[str] = []
        warnings: List[str] = []
        validated: List[str] = []

        for pheno_id in phenotypes:
            # Format validation
            format_result = self.validate_phenotype_format(pheno_id)
            if not format_result.is_valid:
                warnings.extend(format_result.warnings)
                continue

            # Semantic validation
            if ontology is not None:
                semantic_result = self.validate_phenotype_semantic(
                    pheno_id, ontology
                )
                if not semantic_result.is_valid:
                    warnings.extend(semantic_result.warnings)
                    continue

            validated.append(pheno_id)

        if not validated:
            errors.append("No valid phenotypes found")

        return ValidationResult(
            is_valid=len(validated) > 0,
            validated_phenotypes=validated,
            warnings=warnings,
            errors=errors,
        )

    def validate_phenotype_format(self, pheno_id: str) -> ValidationResult:
        """
        Validate HPO ID format.

        Args:
            pheno_id: HPO phenotype ID

        Returns:
            ValidationResult
        """
        warnings: List[str] = []
        errors: List[str] = []

        # Check type
        if not isinstance(pheno_id, str):
            errors.append(f"Phenotype ID must be string, got {type(pheno_id)}")
            return ValidationResult(
                is_valid=False,
                validated_phenotypes=[],
                warnings=warnings,
                errors=errors,
            )

        # Normalize
        pheno_id = pheno_id.strip()

        # Check empty
        if not pheno_id:
            errors.append("Phenotype ID cannot be empty")
            return ValidationResult(
                is_valid=False,
                validated_phenotypes=[],
                warnings=warnings,
                errors=errors,
            )

        # Check format
        if not self._hpo_pattern.match(pheno_id):
            warnings.append(
                f"Invalid HPO format: {pheno_id} "
                f"(expected HP:XXXXXXX)"
            )
            return ValidationResult(
                is_valid=False,
                validated_phenotypes=[],
                warnings=warnings,
                errors=[],
            )

        return ValidationResult(
            is_valid=True,
            validated_phenotypes=[pheno_id],
            warnings=warnings,
            errors=errors,
        )

    def validate_phenotype_semantic(
        self,
        pheno_id: str,
        ontology: "OntologyProtocol",
    ) -> ValidationResult:
        """
        Validate phenotype against ontology.

        Args:
            pheno_id: HPO phenotype ID
            ontology: Ontology instance for validation

        Returns:
            ValidationResult
        """
        warnings: List[str] = []

        try:
            # Check if term exists in ontology
            if not ontology.has_term(pheno_id):
                warnings.append(f"Unknown phenotype in ontology: {pheno_id}")
                return ValidationResult(
                    is_valid=False,
                    validated_phenotypes=[],
                    warnings=warnings,
                    errors=[],
                )

            # Check if term is obsolete
            term = ontology.get_term(pheno_id)
            if hasattr(term, "is_obsolete") and term.is_obsolete:
                warnings.append(f"Obsolete phenotype: {pheno_id}")
                # Still valid but with warning

        except Exception as e:
            logger.warning(f"Error validating phenotype {pheno_id}: {e}")
            warnings.append(f"Could not validate phenotype: {pheno_id}")

        return ValidationResult(
            is_valid=True,
            validated_phenotypes=[pheno_id],
            warnings=warnings,
            errors=[],
        )

    def _validate_confidences(
        self,
        phenotypes: List[str],
        confidences: List[float],
    ) -> ValidationResult:
        """Validate phenotype confidence scores."""
        warnings: List[str] = []
        errors: List[str] = []

        # Check length match
        if len(confidences) != len(phenotypes):
            errors.append(
                f"Confidence list length ({len(confidences)}) "
                f"does not match phenotype list length ({len(phenotypes)})"
            )

        # Check value range
        for i, conf in enumerate(confidences):
            if not isinstance(conf, (int, float)):
                errors.append(
                    f"Confidence at index {i} must be numeric, got {type(conf)}"
                )
            elif conf < 0.0 or conf > 1.0:
                warnings.append(
                    f"Confidence at index {i} ({conf}) outside [0, 1] range"
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            validated_phenotypes=[],
            warnings=warnings,
            errors=errors,
        )


# ==============================================================================
# Factory Function
# ==============================================================================
def create_input_validator(
    strict_hpo_format: bool = False,
    min_phenotypes: int = 1,
    max_phenotypes: int = 100,
) -> InputValidator:
    """
    Factory function to create an InputValidator.

    Args:
        strict_hpo_format: Require exactly 7 digits in HPO IDs
        min_phenotypes: Minimum required phenotypes
        max_phenotypes: Maximum allowed phenotypes

    Returns:
        Configured InputValidator instance
    """
    return InputValidator(
        strict_hpo_format=strict_hpo_format,
        min_phenotypes=min_phenotypes,
        max_phenotypes=max_phenotypes,
    )
