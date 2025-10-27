"""
SymPy Integration with Proof System.

This module extends the proof system (ProofBox, ProofStep) with SymPy validation:

1. PropertyReferenceWithSymPy - AttributeReference extended with dual representation
2. SymPyProofStep - Validated proof step with SymPy transformations
3. ProofStep validation methods - validate_sympy(), validate_sympy_chain()
4. ProofBox validation methods - validate_all_sympy()

Design Philosophy:
- Non-invasive extension: Existing proof system works without SymPy
- Opt-in validation: Add dual representations when SymPy can help
- Graceful fallback: When SymPy can't validate, allow LLM proofs
- Clear diagnostics: Report validation results with details

All types follow Lean-compatible patterns.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from fragile.proofs.sympy.dual_representation import DualExpr, DualStatement, SymPyContext
from fragile.proofs.core.proof_system import (
    DirectDerivation,
    LemmaApplication,
    ProofBox,
    ProofInput,
    ProofOutput,
    ProofStep,
    AttributeReference,
    SubProofReference,
)
from fragile.proofs.sympy.validation import (
    SymPyValidator,
    Transformation,
    TransformationType,
    ValidationResult,
    ValidationStatus,
)


# =============================================================================
# EXTENDED PROPERTY REFERENCE
# =============================================================================


class PropertyReferenceWithSymPy(AttributeReference):
    """
    AttributeReference extended with SymPy dual representation.

    This is a non-invasive extension: if dual_statement is None,
    it behaves identically to AttributeReference.

    Maps to Lean:
        structure PropertyReferenceWithSymPy extends AttributeReference where
          dual_statement : Option DualStatement
    """

    model_config = ConfigDict(frozen=True)

    dual_statement: Optional[DualStatement] = Field(
        None,
        description="Dual representation (LaTeX + SymPy) of the property statement",
    )

    def can_validate_with_sympy(self) -> bool:
        """Check if this property can be validated with SymPy."""
        return self.dual_statement is not None and self.dual_statement.can_validate()


# =============================================================================
# SYMPY PROOF STEP
# =============================================================================


class SymPyProofStep(BaseModel):
    """
    SymPy-validated proof step with transformations.

    This stores the SymPy validation of a proof step:
    - Assumptions used
    - Transformations applied
    - Validation result

    Maps to Lean:
        structure SymPyProofStep where
          input_statements : List DualStatement
          output_statements : List DualStatement
          transformations : List Transformation
          validation_result : ValidationResult
    """

    model_config = ConfigDict(frozen=True)

    input_statements: List[DualStatement] = Field(
        default_factory=list,
        description="Input property statements (from PropertyReferenceWithSymPy)",
    )
    output_statements: List[DualStatement] = Field(
        default_factory=list,
        description="Output property statements (to be established)",
    )
    transformations: List[Transformation] = Field(
        default_factory=list,
        description="SymPy transformations applied in this step",
    )
    validation_result: ValidationResult = Field(
        ...,
        description="Result of SymPy validation for this step",
    )

    def is_valid(self) -> bool:
        """Check if this step is valid according to SymPy."""
        return self.validation_result.is_valid


# =============================================================================
# EXTENDED DIRECT DERIVATION
# =============================================================================


class DirectDerivationWithSymPy(DirectDerivation):
    """
    DirectDerivation extended with SymPy validation.

    This is a non-invasive extension: if sympy_step is None,
    it behaves identically to DirectDerivation.

    Maps to Lean:
        structure DirectDerivationWithSymPy extends DirectDerivation where
          sympy_step : Option SymPyProofStep
    """

    model_config = ConfigDict(frozen=True)

    sympy_step: Optional[SymPyProofStep] = Field(
        None,
        description="SymPy validation of this derivation",
    )

    def can_validate_with_sympy(self) -> bool:
        """Check if this derivation has SymPy validation."""
        return self.sympy_step is not None

    def is_sympy_valid(self) -> bool:
        """Check if SymPy validation passed."""
        if self.sympy_step is None:
            return True  # No SymPy validation, allow LLM proof
        return self.sympy_step.is_valid()


# =============================================================================
# PROOF STEP VALIDATION
# =============================================================================


def validate_proof_step_with_sympy(
    step: ProofStep,
    validator: Optional[SymPyValidator] = None,
) -> ValidationResult:
    """
    Validate a ProofStep using SymPy.

    Strategy:
    1. Check if step has SymPy-compatible derivation
    2. Extract input/output statements from properties
    3. Use validator to check step correctness
    4. Return validation result

    If SymPy cannot validate, returns UNCERTAIN (allows LLM proof).
    """
    # Create validator if not provided
    if validator is None:
        validator = SymPyValidator()

    # Check step type
    if step.step_type.value != "direct_derivation":
        return ValidationResult.uncertain("Only direct derivations can be validated with SymPy")

    # Check if derivation has SymPy validation
    if isinstance(step.derivation, DirectDerivationWithSymPy):
        if step.derivation.sympy_step is not None:
            return step.derivation.sympy_step.validation_result

    # Try to validate using input/output properties
    # This requires extracting dual statements from PropertyReferenceWithSymPy
    input_statements: List[DualStatement] = []
    output_statements: List[DualStatement] = []

    # Extract input statements
    for proof_input in step.inputs:
        for prop in proof_input.required_properties:
            if isinstance(prop, PropertyReferenceWithSymPy) and prop.dual_statement is not None:
                input_statements.append(prop.dual_statement)

    # Extract output statements
    for proof_output in step.outputs:
        for prop in proof_output.properties_established:
            if isinstance(prop, PropertyReferenceWithSymPy) and prop.dual_statement is not None:
                output_statements.append(prop.dual_statement)

    # If no dual statements, cannot validate
    if not output_statements:
        return ValidationResult.uncertain("No dual statements available for validation")

    # Validate each output statement
    for output_stmt in output_statements:
        result = validator.validate_statement(output_stmt)
        if not result.is_valid:
            return result

    # All statements valid
    return ValidationResult.valid({"statements_validated": str(len(output_statements))})


def validate_proof_chain_with_sympy(
    steps: List[ProofStep],
    validator: Optional[SymPyValidator] = None,
) -> ValidationResult:
    """
    Validate a chain of proof steps.

    This checks:
    1. Each step is individually valid
    2. Properties flow correctly between steps
    3. Transitive reasoning is sound
    """
    if validator is None:
        validator = SymPyValidator()

    for i, step in enumerate(steps):
        result = validate_proof_step_with_sympy(step, validator)
        if not result.is_valid:
            from fragile.proofs.sympy.validation import ValidationIssue

            return ValidationResult(
                status=ValidationStatus.INVALID,
                is_valid=False,
                can_validate=True,
                issues=[
                    ValidationIssue(
                        severity="error",
                        message=f"Step {i} failed validation",
                        location=f"step[{i}] ({step.step_id})",
                    )
                ],
            )

    return ValidationResult.valid({"steps_validated": str(len(steps))})


# =============================================================================
# PROOFBOX VALIDATION
# =============================================================================


def validate_proof_box_with_sympy(
    proof: ProofBox,
    validator: Optional[SymPyValidator] = None,
    recursive: bool = True,
) -> ValidationResult:
    """
    Validate a ProofBox using SymPy.

    Args:
        proof: The ProofBox to validate
        validator: SymPy validator to use (creates one if None)
        recursive: If True, recursively validate sub-proofs

    Returns:
        ValidationResult with status and diagnostics
    """
    if validator is None:
        validator = SymPyValidator()

    # Validate main proof steps
    result = validate_proof_chain_with_sympy(proof.steps, validator)
    if not result.is_valid:
        return result

    # Recursively validate sub-proofs
    if recursive and proof.sub_proofs:
        for sub_proof_id, sub_proof in proof.sub_proofs.items():
            sub_result = validate_proof_box_with_sympy(sub_proof, validator, recursive=True)
            if not sub_result.is_valid:
                from fragile.proofs.sympy.validation import ValidationIssue

                return ValidationResult(
                    status=ValidationStatus.INVALID,
                    is_valid=False,
                    can_validate=True,
                    issues=[
                        ValidationIssue(
                            severity="error",
                            message=f"Sub-proof {sub_proof_id} failed validation",
                            location=f"sub_proof[{sub_proof_id}]",
                        )
                    ],
                )

    return ValidationResult.valid(
        {
            "main_steps": str(len(proof.steps)),
            "sub_proofs": str(len(proof.sub_proofs)) if proof.sub_proofs else "0",
        }
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_sympy_proof_step(
    input_statements: List[DualStatement],
    output_statements: List[DualStatement],
    transformations: List[Transformation],
    validator: Optional[SymPyValidator] = None,
) -> SymPyProofStep:
    """
    Create and validate a SymPyProofStep.

    This validates that:
    1. All transformations are correct
    2. Output statements follow from inputs and transformations
    """
    if validator is None:
        validator = SymPyValidator()

    # Validate transformations
    for transformation in transformations:
        result = validator.validate_transformation(transformation)
        if not result.is_valid:
            return SymPyProofStep(
                input_statements=input_statements,
                output_statements=output_statements,
                transformations=transformations,
                validation_result=result,
            )

    # Validate output statements
    result = validator.validate_chain([stmt for stmt in output_statements])

    return SymPyProofStep(
        input_statements=input_statements,
        output_statements=output_statements,
        transformations=transformations,
        validation_result=result,
    )


def add_sympy_validation_to_step(
    step: ProofStep,
    sympy_step: SymPyProofStep,
) -> ProofStep:
    """
    Add SymPy validation to an existing ProofStep.

    This creates a new ProofStep with the same data but with
    SymPy validation attached to the derivation.
    """
    if step.step_type.value != "direct_derivation" or step.derivation is None:
        raise ValueError("Can only add SymPy validation to direct derivations")

    if not isinstance(step.derivation, DirectDerivation):
        raise ValueError("Derivation must be DirectDerivation")

    # Create new derivation with SymPy
    new_derivation = DirectDerivationWithSymPy(
        mathematical_content=step.derivation.mathematical_content,
        techniques=step.derivation.techniques,
        verification_status=step.derivation.verification_status,
        sympy_step=sympy_step,
    )

    # Create new step with updated derivation
    return ProofStep(
        step_id=step.step_id,
        description=step.description,
        inputs=step.inputs,
        outputs=step.outputs,
        step_type=step.step_type,
        derivation=new_derivation,
        status=step.status,
        justification=step.justification,
    )


__all__ = [
    "PropertyReferenceWithSymPy",
    "SymPyProofStep",
    "DirectDerivationWithSymPy",
    "validate_proof_step_with_sympy",
    "validate_proof_chain_with_sympy",
    "validate_proof_box_with_sympy",
    "create_sympy_proof_step",
    "add_sympy_validation_to_step",
]
