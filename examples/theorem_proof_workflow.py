"""
Theorem-Proof Workflow Example: Organic High-to-Low Level Integration.

This example demonstrates the complete workflow for bridging TheoremBox
(high-level validation) with ProofBox (low-level proof development):

1. Create theorem (high-level specification)
2. Generate proof sketch from theorem (auto-generates inputs/outputs)
3. Attach proof to theorem (validates against theorem spec)
4. Expand proof steps (LLM fills in mathematical details)
5. Validate proof (automatic dataflow + theorem match validation)

This shows the organic connection between theorem validation and proof development.

Version: 1.0.0
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fragile.proofs import (
    MathematicalObject,
    ObjectType,
    Property,
    TheoremBox,
    TheoremOutputType,
    attach_proof_to_theorem,
    create_proof_sketch_from_theorem,
    create_simple_object,
    create_simple_theorem,
    print_validation_result,
)


def main():
    """Demonstrate complete theorem → proof → validation workflow."""

    print("=" * 80)
    print("THEOREM-PROOF WORKFLOW: Organic High-to-Low Level Integration")
    print("=" * 80)
    print()

    # =========================================================================
    # STEP 1: Create Mathematical Objects (The Foundation)
    # =========================================================================

    print("STEP 1: Creating mathematical objects...")
    print("-" * 80)

    # Create discrete system object
    discrete_system = create_simple_object(
        label="obj-euclidean-gas-discrete",
        name="Discrete Euclidean Gas",
        expr="S_N = (x_1, v_1, ..., x_N, v_N) evolving via Ψ_kin ∘ Ψ_clone",
        obj_type=ObjectType.STRUCTURE,
        tags=["discrete", "euclidean-gas", "particle-system"],
    )

    # Add properties to discrete system
    discrete_system = discrete_system.add_property(
        Property(
            label="prop-bounded-potential",
            expression="|U(x)| ≤ C_U for all x ∈ 𝒳",
            object_label="obj-euclidean-gas-discrete",
            established_by="thm-euclidean-gas-properties",
        ),
        timestamp=0,
    )

    discrete_system = discrete_system.add_property(
        Property(
            label="prop-lipschitz-potential",
            expression="|∇U(x) - ∇U(y)| ≤ L_U |x - y|",
            object_label="obj-euclidean-gas-discrete",
            established_by="thm-euclidean-gas-properties",
        ),
        timestamp=0,
    )

    # Create continuous system object
    continuous_system = create_simple_object(
        label="obj-euclidean-gas-continuous",
        name="Continuous Euclidean Gas (PDE)",
        expr="∂_t μ = L_kin μ + L_clone μ where μ(t) is measure on 𝒳 × ℝ^d",
        obj_type=ObjectType.STRUCTURE,
        tags=["continuous", "euclidean-gas", "pde"],
    )

    # Store objects in registry
    objects = {
        "obj-euclidean-gas-discrete": discrete_system,
        "obj-euclidean-gas-continuous": continuous_system,
    }

    print(f"✓ Created {len(objects)} mathematical objects")
    print(f"  - {discrete_system.name}: {len(discrete_system.current_properties)} properties")
    print(f"  - {continuous_system.name}: {len(continuous_system.current_properties)} properties")
    print()

    # =========================================================================
    # STEP 2: Define Theorem (High-Level Specification)
    # =========================================================================

    print("STEP 2: Defining theorem (high-level specification)...")
    print("-" * 80)

    theorem = TheoremBox(
        label="thm-mean-field-convergence",
        name="Mean Field Convergence Rate",
        statement_type="theorem",
        # Inputs: what objects and properties we need
        input_objects=["obj-euclidean-gas-discrete"],
        properties_required={
            "obj-euclidean-gas-discrete": ["prop-bounded-potential", "prop-lipschitz-potential"]
        },
        # Output: what properties we establish
        output_type=TheoremOutputType.RELATION,
        properties_added=[
            Property(
                label="prop-mean-field-equivalence",
                expression="W_2(μ_N, μ_t) = O(N^{-1/d}) where μ_N is empirical measure",
                object_label="obj-euclidean-gas-continuous",
                established_by="thm-mean-field-convergence",
            )
        ],
    )

    print(f"✓ Theorem defined: {theorem.name}")
    print(f"  Label: {theorem.label}")
    print(f"  Inputs: {theorem.input_objects}")
    print(f"  Required properties: {list(theorem.properties_required.keys())}")
    print(f"  Output properties: {len(theorem.properties_added)}")
    print(f"  Has proof: {theorem.has_proof()}")  # NEW METHOD!
    print(f"  Proof status: {theorem.proof_status}")  # NEW FIELD!
    print()

    # =========================================================================
    # STEP 3: Generate Proof Sketch from Theorem (Auto-Generation)
    # =========================================================================

    print("STEP 3: Generating proof sketch from theorem...")
    print("-" * 80)

    # NEW FUNCTION: Auto-generates proof structure from theorem
    proof_sketch = create_proof_sketch_from_theorem(
        theorem=theorem,
        objects=objects,
        strategy="Use LSI inequality + Poincaré + Grönwall to bound Wasserstein distance",
        num_steps=5,
    )

    print(f"✓ Proof sketch generated: {proof_sketch.label}")
    print(f"  Proof ID: {proof_sketch.proof_id}")
    print(f"  Proves: {proof_sketch.proves}")
    print(f"  Strategy: {proof_sketch.strategy}")
    print(f"  Number of steps: {len(proof_sketch.steps)}")
    print(f"  All steps sketched: {all(step.is_sketched() for step in proof_sketch.steps)}")
    print()

    print("  Auto-generated proof structure:")
    for i, step in enumerate(proof_sketch.steps, 1):
        print(f"    Step {i}: {step.description}")
        print(f"      Status: {step.status.value}")
        print(f"      Type: {step.step_type.value}")
        print(f"      Inputs: {len(step.inputs)} objects")
        print(f"      Outputs: {len(step.outputs)} objects")

    print()

    # =========================================================================
    # STEP 4: Attach Proof to Theorem (Bidirectional Connection)
    # =========================================================================

    print("STEP 4: Attaching proof to theorem with validation...")
    print("-" * 80)

    # NEW FUNCTION: Attach proof with automatic validation
    try:
        theorem_with_proof = attach_proof_to_theorem(
            theorem=theorem, proof=proof_sketch, objects=objects, validate=True  # Auto-validates!
        )

        print(f"✓ Proof attached successfully!")
        print(f"  Theorem has proof: {theorem_with_proof.has_proof()}")  # NEW METHOD!
        print(f"  Proof status: {theorem_with_proof.proof_status}")  # Updated automatically!
        print(f"  Is proven: {theorem_with_proof.is_proven()}")  # NEW METHOD!
        print()

        # Show bidirectional navigation
        print("  Bidirectional navigation:")
        print(f"    Theorem → Proof: {theorem_with_proof.proof.proof_id}")
        print(f"    Proof → Theorem: {theorem_with_proof.proof.theorem.label}")
        print()

    except ValueError as e:
        print(f"✗ Proof validation failed:")
        print(f"  {e}")
        return

    # =========================================================================
    # STEP 5: Validate Proof Against Theorem (Comprehensive Validation)
    # =========================================================================

    print("STEP 5: Comprehensive proof validation...")
    print("-" * 80)

    # NEW METHOD: Validate proof against theorem specification
    validation_result = theorem_with_proof.validate_proof(objects)

    print_validation_result(validation_result)
    print()

    # =========================================================================
    # STEP 6: Check Proof Dataflow (Step-by-Step Validation)
    # =========================================================================

    print("STEP 6: Validating proof dataflow...")
    print("-" * 80)

    dataflow_errors = theorem_with_proof.proof.validate_dataflow()

    if not dataflow_errors:
        print("✓ Proof dataflow is valid!")
        print("  All step inputs are satisfied by previous outputs")
        print("  Final outputs match proof specification")
    else:
        print(f"✗ Proof dataflow has {len(dataflow_errors)} errors:")
        for error in dataflow_errors:
            print(f"  - {error}")

    print()

    # =========================================================================
    # STEP 7: Summary of Theorem-Proof Integration
    # =========================================================================

    print("=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    print()

    print("✓ High-level specification (TheoremBox):")
    print(f"  - Theorem: {theorem_with_proof.label}")
    print(f"  - Input objects: {len(theorem_with_proof.input_objects)}")
    print(f"  - Required properties: {sum(len(props) for props in theorem_with_proof.properties_required.values())}")
    print(f"  - Output properties: {len(theorem_with_proof.properties_added)}")
    print()

    print("✓ Low-level proof (ProofBox):")
    print(f"  - Proof ID: {theorem_with_proof.proof.proof_id}")
    print(f"  - Proof steps: {len(theorem_with_proof.proof.steps)}")
    print(f"  - Sketched steps: {len(theorem_with_proof.proof.get_sketched_steps())}")
    print(f"  - Expanded steps: {len([s for s in theorem_with_proof.proof.steps if s.is_expanded()])}")
    print()

    print("✓ Integration status:")
    print(f"  - Theorem has proof: {theorem_with_proof.has_proof()}")
    print(f"  - Proof status: {theorem_with_proof.proof_status}")
    print(f"  - Is proven: {theorem_with_proof.is_proven()}")
    print(f"  - Validation passed: {validation_result.is_valid}")
    print(f"  - Dataflow valid: {len(dataflow_errors) == 0}")
    print()

    print("✓ Next steps:")
    print("  1. Expand sketched proof steps with LLM")
    print("  2. Submit for dual review (Gemini 2.5 Pro + Codex)")
    print("  3. Implement feedback after critical evaluation")
    print("  4. Mark steps as VERIFIED after validation")
    print("  5. Proof status will update: sketched → expanded → verified")
    print()

    print("=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print()

    print("Key features demonstrated:")
    print("  ✓ TheoremBox.proof field (optional ProofBox)")
    print("  ✓ TheoremBox.proof_status field (unproven/sketched/expanded/verified)")
    print("  ✓ create_proof_sketch_from_theorem() - auto-generation")
    print("  ✓ attach_proof_to_theorem() - with validation")
    print("  ✓ theorem.has_proof() - convenience method")
    print("  ✓ theorem.is_proven() - check completion")
    print("  ✓ theorem.validate_proof() - comprehensive validation")
    print("  ✓ Bidirectional navigation (theorem ↔ proof)")
    print("  ✓ Automatic proof status tracking")
    print()


if __name__ == "__main__":
    main()
