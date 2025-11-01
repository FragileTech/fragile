"""Standardized validation infrastructure for mathematical entity refinement.

This module provides reusable validation scripts that all skills and agents can use,
eliminating the need for custom validation implementations in each agent.

Module Structure:
    - base_validator: Base classes and utilities for validation
    - entity_validators: Entity-specific validators (TheoremValidator, AxiomValidator, etc.)
    - schema_validator: Pydantic schema validation
    - relationship_validator: Cross-reference and dependency validation
    - framework_validator: Framework consistency checks via Gemini
    - validation_report: Report generation utilities
    - cli: Command-line interface

Usage:
    # Schema validation only
    python -m fragile.mathster.tools.validation --refined-dir PATH --mode schema

    # Complete validation (includes framework consistency)
    python -m fragile.mathster.tools.validation --refined-dir PATH --mode complete

    # Entity-specific validation
    python -m fragile.mathster.tools.validation --refined-dir PATH --entity-types theorems axioms

Programmatic Usage:
    from fragile.mathster.tools.validation import TheoremValidator, ValidationReport

    validator = TheoremValidator()
    result = validator.validate_file("theorems/thm-example.json")

    if not result.is_valid:
        print(f"Errors: {result.errors}")
"""

from mathster.tools.validation.base_validator import (
    BaseValidator,
    ValidationError,
    ValidationResult,
    ValidationWarning,
)
from mathster.tools.validation.entity_validators import (
    AxiomValidator,
    EquationValidator,
    ObjectValidator,
    ParameterValidator,
    ProofValidator,
    RemarkValidator,
    TheoremValidator,
)
from mathster.tools.validation.framework_validator import FrameworkValidator
from mathster.tools.validation.relationship_validator import RelationshipValidator
from mathster.tools.validation.schema_validator import SchemaValidator
from mathster.tools.validation.validation_report import ValidationReport


__all__ = [
    "AxiomValidator",
    "BaseValidator",
    "EquationValidator",
    "FrameworkValidator",
    "ObjectValidator",
    "ParameterValidator",
    "ProofValidator",
    "RelationshipValidator",
    "RemarkValidator",
    "SchemaValidator",
    "TheoremValidator",
    "ValidationError",
    "ValidationReport",
    "ValidationResult",
    "ValidationWarning",
]
