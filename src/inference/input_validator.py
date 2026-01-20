"""
# ==============================================================================
# Module: src/inference/input_validator.py
# ==============================================================================
# Purpose: Validate patient input for diagnosis pipeline
#
# Dependencies:
#   - External: None (pure Python)
#   - Internal: src.core.types, src.core.protocols
#
# Input:
#   - PatientPhenotypes: Patient phenotype data to validate
#   - OntologyProtocol (optional): For semantic validation
#   - Dict[str, Any]: Raw patient input for API use
#
# Output:
#   - ValidationResult: Internal validation status
#   - Result[T]: Protocol-compliant result type
#
# Protocol Compliance:
#   - Implements InputValidatorProtocol from src.core.protocols
#
# Design Notes:
#   - Supports HPO phenotype ID format validation
#   - Optional semantic validation against ontology
#   - Returns warnings for non-fatal issues
#   - Production-ready error messages
#   - P1-ready: Extensible validation hooks
# ==============================================================================
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from src.core.types import PatientPhenotypes, Result

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

    # ==========================================================================
    # Protocol-Compliant Methods (InputValidatorProtocol)
    # ==========================================================================
    # These methods provide protocol-compliant interface using Result[T]

    def validate_phenotypes_result(
        self,
        phenotypes: List[str],
        ontology: Optional["OntologyProtocol"] = None,
    ) -> Result[List[str]]:
        """
        Protocol-compliant phenotype validation.

        Returns Result[List[str]] as specified in InputValidatorProtocol.
        """
        result = self.validate_phenotypes(phenotypes, ontology)

        if result.is_valid:
            return Result.ok(
                result.validated_phenotypes,
                warnings=result.warnings,
            )
        else:
            error_msg = "; ".join(result.errors) if result.errors else "Validation failed"
            return Result.fail(error_msg, warnings=result.warnings)

    def validate_patient_input_dict(
        self,
        patient_input: Dict[str, Any],
        ontology: Optional["OntologyProtocol"] = None,
    ) -> Result[PatientPhenotypes]:
        """
        Protocol-compliant patient input validation from Dict.

        Accepts raw Dict input (e.g., from API) and validates it.
        Returns Result[PatientPhenotypes] as specified in InputValidatorProtocol.

        Expected Dict format:
            {
                "patient_id": str,
                "phenotypes": List[str],
                "phenotype_confidences": Optional[List[float]],
                "candidate_genes": Optional[List[str]],
                "age": Optional[int],
                "sex": Optional[str],
            }
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Extract and validate patient_id
        patient_id = patient_input.get("patient_id")
        if not patient_id or not isinstance(patient_id, str):
            errors.append("patient_id is required and must be a string")

        # Extract and validate phenotypes
        phenotypes = patient_input.get("phenotypes")
        if not phenotypes or not isinstance(phenotypes, list):
            errors.append("phenotypes is required and must be a list")
            return Result.fail("; ".join(errors))

        # Early return if basic validation fails
        if errors:
            return Result.fail("; ".join(errors))

        # Create PatientPhenotypes object
        try:
            patient = PatientPhenotypes(
                patient_id=patient_id,
                phenotypes=phenotypes,
                phenotype_confidences=patient_input.get("phenotype_confidences"),
                candidate_genes=patient_input.get("candidate_genes"),
                variants=patient_input.get("variants"),
                age=patient_input.get("age"),
                sex=patient_input.get("sex"),
            )
        except Exception as e:
            return Result.fail(f"Failed to create PatientPhenotypes: {e}")

        # Validate using existing method
        result = self.validate_patient_input(patient, ontology)

        if result.is_valid:
            # Update phenotypes to validated ones
            patient.phenotypes = result.validated_phenotypes
            return Result.ok(patient, warnings=result.warnings)
        else:
            error_msg = "; ".join(result.errors) if result.errors else "Validation failed"
            return Result.fail(error_msg, warnings=result.warnings)


# ==============================================================================
# Validation Hooks (Extensibility)
# ==============================================================================
# Type alias for validation hooks
ValidationHook = Callable[[str], Optional[str]]  # Returns error message or None


class ExtensibleInputValidator(InputValidator):
    """
    Extended InputValidator with custom validation hooks.

    Use this for adding custom validation rules without subclassing.
    P1-ready: Supports future validation extensions.

    Usage:
        validator = ExtensibleInputValidator()
        validator.add_phenotype_hook(my_custom_validator)
        result = validator.validate_patient_input(patient)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._phenotype_hooks: List[ValidationHook] = []
        self._patient_hooks: List[Callable[[PatientPhenotypes], Optional[str]]] = []

    def add_phenotype_hook(self, hook: ValidationHook) -> None:
        """Add a custom phenotype validation hook."""
        self._phenotype_hooks.append(hook)

    def add_patient_hook(
        self,
        hook: Callable[[PatientPhenotypes], Optional[str]],
    ) -> None:
        """Add a custom patient-level validation hook."""
        self._patient_hooks.append(hook)

    def validate_phenotype_format(self, pheno_id: str) -> ValidationResult:
        """Override to apply custom hooks."""
        result = super().validate_phenotype_format(pheno_id)

        # Apply custom hooks
        for hook in self._phenotype_hooks:
            error = hook(pheno_id)
            if error:
                result.warnings.append(error)

        return result

    def validate_patient_input(
        self,
        patient_input: PatientPhenotypes,
        ontology: Optional["OntologyProtocol"] = None,
    ) -> ValidationResult:
        """Override to apply custom patient hooks."""
        result = super().validate_patient_input(patient_input, ontology)

        # Apply custom patient hooks
        for hook in self._patient_hooks:
            error = hook(patient_input)
            if error:
                result.warnings.append(error)

        return result


# ==============================================================================
# Factory Functions
# ==============================================================================
def create_input_validator(
    strict_hpo_format: bool = False,
    min_phenotypes: int = 1,
    max_phenotypes: int = 100,
    extensible: bool = False,
) -> InputValidator:
    """
    Factory function to create an InputValidator.

    Args:
        strict_hpo_format: Require exactly 7 digits in HPO IDs
        min_phenotypes: Minimum required phenotypes
        max_phenotypes: Maximum allowed phenotypes
        extensible: If True, return ExtensibleInputValidator with hook support

    Returns:
        Configured InputValidator instance
    """
    validator_class = ExtensibleInputValidator if extensible else InputValidator
    return validator_class(
        strict_hpo_format=strict_hpo_format,
        min_phenotypes=min_phenotypes,
        max_phenotypes=max_phenotypes,
    )
