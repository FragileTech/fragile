"""
Proof System Integration with TheoremBox.

This module bridges the compositional proof system (ProofBox) with the
theorem pipeline (TheoremBox), enabling:
1. Validation that mathster match theorem claims
2. Automatic relationship extraction from mathster
3. Complete workflow from theorem → proof → verification

All types follow Lean-compatible patterns:
- Pure functions (no side effects)
- Total functions (Optional[T] instead of exceptions)
- Explicit validation with clear error messages
"""

from pydantic import BaseModel, ConfigDict, Field

from mathster.proof_pipeline.pipeline_types import (
    Attribute,
    MathematicalObject,
    Relationship,
    RelationshipAttribute,
    TheoremBox,
)
from mathster.core.proof_system import AttributeReference, ProofBox, ProofInput, ProofOutput


# =============================================================================
# VALIDATION TYPES
# =============================================================================


class ProofTheoremMismatch(BaseModel):
    """
    Describes a mismatch between proof and theorem.

    Maps to Lean:
        structure ProofTheoremMismatch where
          mismatch_type : String
          description : String
          expected : String
          actual : String
    """

    model_config = ConfigDict(frozen=True)

    mismatch_type: str = Field(..., description="Type of mismatch (input/output/property)")
    description: str = Field(..., description="Human-readable description")
    expected: str = Field(..., description="What the theorem expects")
    actual: str = Field(..., description="What the proof provides")


class ProofValidationResult(BaseModel):
    """
    Result of validating proof against theorem.

    Maps to Lean:
        structure ProofValidationResult where
          is_valid : Bool
          mismatches : List ProofTheoremMismatch
          warnings : List String
    """

    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(..., description="Whether proof matches theorem")
    mismatches: list[ProofTheoremMismatch] = Field(
        default_factory=list, description="Validation errors"
    )
    warnings: list[str] = Field(default_factory=list, description="Non-blocking warnings")


# =============================================================================
# INTEGRATION FUNCTIONS
# =============================================================================


def validate_proof_for_theorem(
    proof: ProofBox, theorem: TheoremBox, objects: dict[str, MathematicalObject]
) -> ProofValidationResult:
    """
    Validate that a proof correctly implements a theorem.

    Checks:
    1. Proof proves the correct theorem (label match)
    2. Proof inputs match theorem's required properties
    3. Proof outputs establish theorem's claimed properties
    4. All required objects are available

    Args:
        proof: ProofBox to validate
        theorem: TheoremBox it should prove
        objects: Available mathematical objects

    Returns:
        ProofValidationResult with validation status and any errors

    Maps to Lean:
        def validate_proof_for_theorem
          (proof : ProofBox)
          (theorem : TheoremBox)
          (objects : HashMap String MathematicalObject)
          : ProofValidationResult :=
          ...
    """
    mismatches: list[ProofTheoremMismatch] = []
    warnings: list[str] = []

    # Check 1: Proof claims to prove this theorem
    if proof.proves != theorem.label:
        mismatches.append(
            ProofTheoremMismatch(
                mismatch_type="label_mismatch",
                description="Proof claims to prove different theorem",
                expected=theorem.label,
                actual=proof.proves,
            )
        )

    # Check 2: Validate inputs match theorem's property requirements
    # Build map of proof's input properties
    proof_input_properties: dict[str, set[str]] = {}
    for proof_input in proof.inputs:
        if proof_input.object_id not in proof_input_properties:
            proof_input_properties[proof_input.object_id] = set()
        proof_input_properties[proof_input.object_id].update(
            prop.property_id for prop in proof_input.required_properties
        )

    # Check that theorem's required properties are covered by proof inputs
    for obj_label, required_props in theorem.attributes_required.items():
        if obj_label not in proof_input_properties:
            mismatches.append(
                ProofTheoremMismatch(
                    mismatch_type="missing_input_object",
                    description=f"Theorem requires object '{obj_label}' but proof doesn't use it",
                    expected=f"Proof uses object {obj_label}",
                    actual=f"Proof inputs: {list(proof_input_properties.keys())}",
                )
            )
            continue

        proof_props = proof_input_properties[obj_label]
        missing_props = set(required_props) - proof_props

        if missing_props:
            mismatches.append(
                ProofTheoremMismatch(
                    mismatch_type="missing_required_properties",
                    description=f"Theorem requires properties that proof doesn't use for {obj_label}",
                    expected=f"Properties: {required_props}",
                    actual=f"Proof uses: {list(proof_props)}, missing: {list(missing_props)}",
                )
            )

        # Check if proof uses extra properties (warning, not error)
        extra_props = proof_props - set(required_props)
        if extra_props:
            warnings.append(
                f"Proof uses extra properties for {obj_label}: {list(extra_props)}. "
                f"These are not required by the theorem."
            )

    # Check 3: Validate outputs match theorem's claimed properties
    # Build map of proof's output properties
    proof_output_properties: dict[str, set[str]] = {}
    for proof_output in proof.outputs:
        if proof_output.object_id not in proof_output_properties:
            proof_output_properties[proof_output.object_id] = set()
        proof_output_properties[proof_output.object_id].update(
            prop.property_id for prop in proof_output.properties_established
        )

    # Check that theorem's claimed properties are established by proof
    for added_property in theorem.attributes_added:
        obj_label = added_property.object_label
        prop_label = added_property.label

        if obj_label not in proof_output_properties:
            mismatches.append(
                ProofTheoremMismatch(
                    mismatch_type="missing_output_object",
                    description=f"Theorem claims to add property to '{obj_label}' but proof doesn't output it",
                    expected=f"Proof outputs for {obj_label}",
                    actual=f"Proof outputs: {list(proof_output_properties.keys())}",
                )
            )
            continue

        if prop_label not in proof_output_properties[obj_label]:
            mismatches.append(
                ProofTheoremMismatch(
                    mismatch_type="missing_output_property",
                    description=f"Theorem claims to establish '{prop_label}' for '{obj_label}' but proof doesn't",
                    expected=f"Attribute {prop_label} established",
                    actual=f"Proof establishes: {list(proof_output_properties[obj_label])}",
                )
            )

    # Check 4: Verify all required objects exist
    for obj_label in theorem.input_objects:
        if obj_label not in objects:
            mismatches.append(
                ProofTheoremMismatch(
                    mismatch_type="object_not_found",
                    description=f"Theorem requires object '{obj_label}' but it's not available",
                    expected=f"Object {obj_label} exists",
                    actual=f"Available objects: {list(objects.keys())}",
                )
            )

    # Return validation result
    is_valid = len(mismatches) == 0

    return ProofValidationResult(is_valid=is_valid, mismatches=mismatches, warnings=warnings)


def extract_relationships_from_proof(proof: ProofBox, theorem: TheoremBox) -> list[Relationship]:
    """
    Extract relationships from proof outputs.

    Analyzes proof outputs to identify relationships between objects
    and creates Relationship objects that match theorem.relations_established.

    Args:
        proof: ProofBox with outputs
        theorem: TheoremBox that proof implements

    Returns:
        List of Relationship objects extracted from proof

    Maps to Lean:
        def extract_relationships_from_proof
          (proof : ProofBox)
          (theorem : TheoremBox)
          : List Relationship :=
          ...
    """
    relationships: list[Relationship] = []

    # For each proof output pair, check if it establishes a relationship
    for i, output1 in enumerate(proof.outputs):
        for output2 in proof.outputs[i + 1 :]:
            # Check if theorem claims a relationship between these objects
            for rel in theorem.relations_established:
                if (
                    rel.source_object == output1.object_id
                    and rel.target_object == output2.object_id
                ) or (
                    rel.source_object == output2.object_id
                    and rel.target_object == output1.object_id
                ):
                    # Relationship matches - extract properties established
                    properties: list[RelationshipAttribute] = []

                    for prop_ref in (
                        output1.properties_established + output2.properties_established
                    ):
                        properties.append(
                            RelationshipAttribute(
                                label=prop_ref.property_id,
                                expression=prop_ref.property_statement,
                                description=f"Established by {proof.proof_id}",
                            )
                        )

                    relationships.append(rel)

    return relationships


def create_proof_inputs_from_theorem(
    theorem: TheoremBox, objects: dict[str, MathematicalObject]
) -> list[ProofInput]:
    """
    Create ProofInput list from theorem's required properties.

    Converts theorem's attributes_required into ProofInput objects with
    AttributeReference for each required property.

    Args:
        theorem: TheoremBox with property requirements
        objects: Available mathematical objects (for property statements)

    Returns:
        List of ProofInput objects

    Maps to Lean:
        def create_proof_inputs_from_theorem
          (theorem : TheoremBox)
          (objects : HashMap String MathematicalObject)
          : List ProofInput :=
          ...
    """
    proof_inputs: list[ProofInput] = []

    for obj_label, required_props in theorem.attributes_required.items():
        # Get object to extract property statements
        obj = objects.get(obj_label)

        property_refs: list[AttributeReference] = []

        for prop_label in required_props:
            # Try to find property statement from object
            property_statement = f"Attribute {prop_label}"  # Default

            if obj is not None:
                # Find matching property in object
                for prop in obj.current_attributes:
                    if prop.label == prop_label:
                        property_statement = prop.expression
                        break

            property_refs.append(
                AttributeReference(
                    object_id=obj_label,
                    property_id=prop_label,
                    property_statement=property_statement,
                )
            )

        proof_inputs.append(
            ProofInput(
                object_id=obj_label, required_properties=property_refs, required_assumptions=[]
            )
        )

    return proof_inputs


def create_proof_outputs_from_theorem(
    theorem: TheoremBox, objects: dict[str, MathematicalObject]
) -> list[ProofOutput]:
    """
    Create ProofOutput list from theorem's added properties.

    Converts theorem's attributes_added into ProofOutput objects with
    AttributeReference for each property to establish.

    Args:
        theorem: TheoremBox with properties to add
        objects: Available mathematical objects

    Returns:
        List of ProofOutput objects

    Maps to Lean:
        def create_proof_outputs_from_theorem
          (theorem : TheoremBox)
          (objects : HashMap String MathematicalObject)
          : List ProofOutput :=
          ...
    """
    proof_outputs: list[ProofOutput] = []

    # Group properties by object
    properties_by_object: dict[str, list[Attribute]] = {}
    for prop in theorem.attributes_added:
        obj_label = prop.object_label
        if obj_label not in properties_by_object:
            properties_by_object[obj_label] = []
        properties_by_object[obj_label].append(prop)

    # Create ProofOutput for each object
    for obj_label, props in properties_by_object.items():
        property_refs: list[AttributeReference] = []

        for prop in props:
            property_refs.append(
                AttributeReference(
                    object_id=obj_label,
                    property_id=prop.label,
                    property_statement=prop.expression,
                )
            )

        proof_outputs.append(
            ProofOutput(object_id=obj_label, properties_established=property_refs)
        )

    return proof_outputs


# =============================================================================
# WORKFLOW ORCHESTRATION
# =============================================================================


def create_proof_sketch_from_theorem(
    theorem: TheoremBox,
    objects: dict[str, MathematicalObject],
    strategy: str,
    num_steps: int = 3,
) -> ProofBox:
    """
    Create initial proof sketch from theorem specification.

    Auto-generates:
    - Proof inputs from theorem.attributes_required
    - Proof outputs from theorem.attributes_added
    - Sketched proof steps (to be expanded by LLM)

    Args:
        theorem: TheoremBox to create proof for
        objects: Available mathematical objects (for property statements)
        strategy: High-level proof strategy description
        num_steps: Number of proof steps to create (default: 3)

    Returns:
        ProofBox in SKETCHED status with auto-generated structure

    Maps to Lean:
        def create_proof_sketch_from_theorem
          (theorem : TheoremBox)
          (objects : HashMap String MathematicalObject)
          (strategy : String)
          (num_steps : Nat)
          : ProofBox :=
          let inputs := create_proof_inputs_from_theorem theorem objects
          let outputs := create_proof_outputs_from_theorem theorem objects
          let steps := generate_sketched_steps inputs outputs num_steps
          ProofBox.mk
            (proof_id := s!"proof-{theorem.label}")
            (label := s!"Proof of {theorem.name}")
            (proves := theorem.label)
            (inputs := inputs)
            (outputs := outputs)
            (strategy := strategy)
            (steps := steps)
            (theorem := some theorem)
    """
    from mathster.core.proof_system import (
        ProofBox,
        ProofStep,
        ProofStepStatus,
        ProofStepType,
    )

    # Auto-generate inputs/outputs
    inputs = create_proof_inputs_from_theorem(theorem, objects)
    outputs = create_proof_outputs_from_theorem(theorem, objects)

    # Create sketched steps
    steps = []
    for i in range(num_steps):
        # First step uses proof inputs, last step produces proof outputs
        step_inputs = inputs if i == 0 else []
        step_outputs = outputs if i == num_steps - 1 else []

        steps.append(
            ProofStep(
                step_id=f"step-{i + 1}",
                description=f"Step {i + 1} of {num_steps} (to be expanded by LLM)",
                inputs=step_inputs,
                outputs=step_outputs,
                step_type=ProofStepType.DIRECT_DERIVATION,
                status=ProofStepStatus.SKETCHED,
                estimated_complexity="simple" if num_steps <= 3 else "moderate",
            )
        )

    # Create proof ID from theorem label
    proof_id = (
        f"proof-{theorem.label.replace('thm-', '').replace('lem-', '').replace('prop-', '')}"
    )

    # Create ProofBox with back-reference to theorem
    return ProofBox(
        proof_id=proof_id,
        label=f"Proof of {theorem.name}",
        proves=theorem.label,
        inputs=inputs,
        outputs=outputs,
        strategy=strategy,
        steps=steps,
        theorem=theorem,  # Back-reference enables bidirectional navigation
    )


def attach_proof_to_theorem(
    theorem: TheoremBox,
    proof: ProofBox,
    objects: dict[str, MathematicalObject],
    validate: bool = True,
) -> TheoremBox:
    """
    Attach proof to theorem with optional validation.

    Creates new TheoremBox with proof attached (immutable update).
    Optionally validates proof against theorem specification.

    Args:
        theorem: TheoremBox to attach proof to
        proof: ProofBox to attach
        objects: Available objects (for validation)
        validate: Whether to validate proof against theorem (default: True)

    Returns:
        New TheoremBox with proof attached and proof_status updated

    Raises:
        ValueError: If validation fails

    Maps to Lean:
        def attach_proof_to_theorem
          (theorem : TheoremBox)
          (proof : ProofBox)
          (objects : HashMap String MathematicalObject)
          (validate : Bool)
          : Except String TheoremBox :=
          if validate then
            let result := validate_proof_for_theorem proof theorem objects
            if !result.is_valid then
              Except.error s!"Proof validation failed: {result.mismatches}"
            else
              let status := compute_proof_status proof
              Except.ok { theorem with proof := some proof, proof_status := status }
          else
            let status := compute_proof_status proof
            Except.ok { theorem with proof := some proof, proof_status := status }
    """
    # Validate if requested
    if validate:
        result = validate_proof_for_theorem(proof, theorem, objects)
        if not result.is_valid:
            errors = "\n".join(
                f"  - {m.mismatch_type}: {m.description}" for m in result.mismatches
            )
            raise ValueError(f"Proof validation failed:\n{errors}")

    # Determine proof status based on step completion
    if proof.all_steps_expanded():
        # Check if all steps are verified
        from mathster.core.proof_system import ProofStepStatus

        if all(step.status == ProofStepStatus.VERIFIED for step in proof.steps):
            status = "verified"
        else:
            status = "expanded"
    elif any(step.is_expanded() for step in proof.steps):
        status = "sketched"  # Some steps expanded, but not all
    else:
        status = "sketched"  # All steps still sketched

    # Attach proof to theorem (immutable update)
    return theorem.model_copy(update={"proof": proof, "proof_status": status})


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def print_validation_result(result: ProofValidationResult) -> None:
    """
    Pretty-print validation result.

    Args:
        result: ProofValidationResult to display
    """
    if result.is_valid:
        print("✓ Proof validation PASSED")
    else:
        print(f"✗ Proof validation FAILED ({len(result.mismatches)} errors)")

    if result.mismatches:
        print("\nValidation Errors:")
        for i, mismatch in enumerate(result.mismatches, 1):
            print(f"  {i}. {mismatch.mismatch_type}: {mismatch.description}")
            print(f"     Expected: {mismatch.expected}")
            print(f"     Actual: {mismatch.actual}")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  ⚠ {warning}")


def get_proof_statistics(proof: ProofBox) -> dict[str, int]:
    """
    Get statistics about a proof.

    Args:
        proof: ProofBox to analyze

    Returns:
        Dictionary with proof statistics
    """
    from mathster.core.proof_system import ProofStepStatus, ProofStepType

    return {
        "total_steps": len(proof.steps),
        "direct_derivations": sum(
            1 for step in proof.steps if step.step_type == ProofStepType.DIRECT_DERIVATION
        ),
        "sub_proofs": sum(1 for step in proof.steps if step.step_type == ProofStepType.SUB_PROOF),
        "lemma_applications": sum(
            1 for step in proof.steps if step.step_type == ProofStepType.LEMMA_APPLICATION
        ),
        "sketched": sum(1 for step in proof.steps if step.status == ProofStepStatus.SKETCHED),
        "expanded": sum(1 for step in proof.steps if step.status == ProofStepStatus.EXPANDED),
        "verified": sum(1 for step in proof.steps if step.status == ProofStepStatus.VERIFIED),
        "nested_sub_proofs": len(proof.sub_proofs),
        "total_inputs": len(proof.inputs),
        "total_outputs": len(proof.outputs),
    }
