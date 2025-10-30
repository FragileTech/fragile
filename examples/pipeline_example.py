"""
Example: Theorem-Proving-as-Pipeline with Property Accumulation.

Demonstrates:
1. Creating mathematical objects
2. Theorems with properties_required (API signatures)
3. Computed conditionality (dynamic checking)
4. Property accumulation
5. Automatic conditionality upgrades
6. Self-contained theorems (internal assumption discharge)
"""

from pathlib import Path
import sys


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fragile.proofs import (
    Axiom,
    create_simple_object,
    ObjectType,
    Property,
    TheoremBox,
    TheoremOutputType,
)


def main() -> None:
    """Run pipeline example."""
    print("=" * 70)
    print("THEOREM-PROVING-AS-PIPELINE EXAMPLE")
    print("=" * 70)
    print()

    # ==========================================================================
    # STEP 1: Create Objects (Definitions Only)
    # ==========================================================================
    print("STEP 1: Create Mathematical Objects")
    print("-" * 70)

    obj_swarm = create_simple_object(
        label="obj-swarm",
        name="Swarm Configuration",
        expr="S = {x_i(t)}_{i=1}^N ⊂ ℝ^d",
        obj_type=ObjectType.SET,
    )

    print(f"✓ Created: {obj_swarm.label}")
    print(f"  Name: {obj_swarm.name}")
    print(f"  Expression: {obj_swarm.mathematical_expression}")
    print(f"  Properties: {list(obj_swarm.get_property_labels())}")
    print()

    # ==========================================================================
    # STEP 2: Create Axiom (Immutable Foundational Truth)
    # ==========================================================================
    print("STEP 2: Create Axiom")
    print("-" * 70)

    axiom_bounded = Axiom(
        label="axiom-bounded-displacement",
        statement="φ is 1-Lipschitz",
        mathematical_expression="d_alg(φ(x), φ(y)) ≤ d_𝒳(x, y)",
        foundational_framework="Fragile Gas Framework",
    )

    print(f"✓ Created: {axiom_bounded.label}")
    print(f"  Statement: {axiom_bounded.statement}")
    print()

    # ==========================================================================
    # STEP 3: Create Theorem (No Properties Required → Unconditional)
    # ==========================================================================
    print("STEP 3: Create Theorem with NO Properties Required")
    print("-" * 70)

    thm_energy = TheoremBox(
        label="thm-energy-bound",
        name="Energy Bound Theorem",
        input_objects=["obj-swarm"],
        input_axioms=["axiom-bounded-displacement"],
        properties_required={},  # No properties required!
        output_type=TheoremOutputType.PROPERTY,
        properties_added=[
            Property(
                label="prop-bounded-energy",
                expression="E(S) ≤ C·N",
                object_label="obj-swarm",
                established_by="thm-energy-bound",
            )
        ],
    )

    print(f"✓ Created: {thm_energy.label}")
    print(f"  Input Objects: {thm_energy.input_objects}")
    print(f"  Properties Required: {thm_energy.properties_required}")
    print(f"  Properties Added: {[p.label for p in thm_energy.properties_added]}")
    print()

    # Check conditionality
    objects_dict = {"obj-swarm": obj_swarm}
    is_conditional = thm_energy.is_conditional(objects_dict)
    print(f"  Is Conditional? {is_conditional}")
    print("  Reason: No properties required → UNCONDITIONAL")
    print()

    # ==========================================================================
    # STEP 4: Execute First Theorem (Add Property)
    # ==========================================================================
    print("STEP 4: Execute Theorem → Add Property to Object")
    print("-" * 70)

    # Simulate execution: add property to object
    prop_energy = thm_energy.properties_added[0]
    obj_swarm_v2 = obj_swarm.add_property(prop_energy, timestamp=1)

    print(f"✓ Executed: {thm_energy.label}")
    print(f"  Properties Added to {obj_swarm_v2.label}:")
    for prop in obj_swarm_v2.current_properties:
        print(f"    - {prop.label}: {prop.expression}")
    print()

    # ==========================================================================
    # STEP 5: Create Theorem Requiring Property (Conditional)
    # ==========================================================================
    print("STEP 5: Create Theorem with Property Required")
    print("-" * 70)

    thm_convergence = TheoremBox(
        label="thm-convergence-rate",
        name="Convergence Rate Theorem",
        input_objects=["obj-swarm"],
        properties_required={
            "obj-swarm": ["prop-bounded-variance"]  # API: Needs this property!
        },
        output_type=TheoremOutputType.PROPERTY,
        properties_added=[
            Property(
                label="prop-exponential-convergence",
                expression="‖μ_t - μ_∞‖ ≤ C·e^(-λt)",
                object_label="obj-swarm",
                established_by="thm-convergence-rate",
            )
        ],
    )

    print(f"✓ Created: {thm_convergence.label}")
    print(f"  Properties Required: {thm_convergence.properties_required}")
    print()

    # Check conditionality (BEFORE property added)
    objects_dict_v2 = {"obj-swarm": obj_swarm_v2}
    missing_before = thm_convergence.compute_conditionality(objects_dict_v2)
    is_conditional_before = thm_convergence.is_conditional(objects_dict_v2)

    print("  Conditionality Check (BEFORE proving property):")
    print(f"    Is Conditional? {is_conditional_before}")
    print(f"    Missing Properties: {missing_before}")
    print("    Reason: obj-swarm missing 'prop-bounded-variance'")
    print()

    # ==========================================================================
    # STEP 6: Prove Missing Property (Modular Proof)
    # ==========================================================================
    print("STEP 6: Prove Missing Property (Modular Proof)")
    print("-" * 70)

    thm_variance = TheoremBox(
        label="thm-variance-proof",
        name="Variance Bound Proof",
        input_objects=["obj-swarm"],
        properties_required={},  # No requirements
        output_type=TheoremOutputType.PROPERTY,
        properties_added=[
            Property(
                label="prop-bounded-variance",
                expression="Var[X] < ∞",
                object_label="obj-swarm",
                established_by="thm-variance-proof",
            )
        ],
    )

    print(f"✓ Created: {thm_variance.label}")
    print(f"  Properties Added: {[p.label for p in thm_variance.properties_added]}")
    print()

    # Execute: add property
    prop_variance = thm_variance.properties_added[0]
    obj_swarm_v3 = obj_swarm_v2.add_property(prop_variance, timestamp=2)

    print(f"✓ Executed: {thm_variance.label}")
    print(f"  Properties now on {obj_swarm_v3.label}:")
    for prop in obj_swarm_v3.current_properties:
        print(f"    - {prop.label}")
    print()

    # ==========================================================================
    # STEP 7: Check Automatic Conditionality Upgrade
    # ==========================================================================
    print("STEP 7: Automatic Conditionality Upgrade")
    print("-" * 70)

    objects_dict_v3 = {"obj-swarm": obj_swarm_v3}
    missing_after = thm_convergence.compute_conditionality(objects_dict_v3)
    is_conditional_after = thm_convergence.is_conditional(objects_dict_v3)

    print("  Conditionality Check (AFTER proving property):")
    print(f"    Is Conditional? {is_conditional_after}")
    print(f"    Missing Properties: {missing_after}")
    print("    Result: NOW UNCONDITIONAL! ✨")
    print()

    print("  Explanation:")
    print("    - Required: prop-bounded-variance")
    print(f"    - Object has: {list(obj_swarm_v3.get_property_labels())}")
    print("    - All requirements satisfied → Automatic upgrade!")
    print()

    # ==========================================================================
    # STEP 8: Self-Contained Theorem (Internal Assumption Discharge)
    # ==========================================================================
    print("STEP 8: Self-Contained Theorem (Proves Own Assumptions)")
    print("-" * 70)

    obj_new = create_simple_object(
        label="obj-potential",
        name="Potential Function",
        expr="U: ℝ^d → ℝ",
        obj_type=ObjectType.FUNCTION,
    )

    thm_self_contained = TheoremBox(
        label="thm-self-contained-convergence",
        name="Self-Contained Convergence Theorem",
        input_objects=["obj-potential"],
        properties_required={
            "obj-potential": ["prop-convex"]  # Requires convexity...
        },
        output_type=TheoremOutputType.PROPERTY,
        properties_added=[
            Property(
                label="prop-convex",
                expression="∇²U ≥ 0",
                object_label="obj-potential",
                established_by="thm-self-contained-convergence",
            ),  # ...but proves it internally!
            Property(
                label="prop-fast-convergence",
                expression="Rate = O(e^(-αt))",
                object_label="obj-potential",
                established_by="thm-self-contained-convergence",
            ),
        ],
    )

    print(f"✓ Created: {thm_self_contained.label}")
    print(f"  Properties Required: {thm_self_contained.properties_required}")
    print(f"  Properties Added: {[p.label for p in thm_self_contained.properties_added]}")
    print()

    # Check BEFORE execution
    objects_initial = {"obj-potential": obj_new}
    missing_initial = thm_self_contained.compute_conditionality(objects_initial)
    print("  Conditionality (BEFORE execution):")
    print(f"    Missing: {missing_initial}")
    print("    Status: CONDITIONAL")
    print()

    # Simulate execution: add ALL properties (including the required one!)
    obj_updated = obj_new
    for prop in thm_self_contained.properties_added:
        obj_updated = obj_updated.add_property(prop, timestamp=3)

    objects_final = {"obj-potential": obj_updated}
    missing_final = thm_self_contained.compute_conditionality(objects_final)
    print("  Conditionality (AFTER execution):")
    print(f"    Missing: {missing_final}")
    print("    Status: UNCONDITIONAL ✨")
    print("    Reason: Theorem proved its own assumption internally!")
    print()

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("✅ Demonstrated:")
    print("  1. Object creation (definitions only)")
    print("  2. Property accumulation via theorems")
    print("  3. Computed conditionality (properties_required vs current_properties)")
    print("  4. Automatic upgrades when properties proved")
    print("  5. Modular proofs (prove once, reuse everywhere)")
    print("  6. Self-contained theorems (internal assumption discharge)")
    print()
    print("✅ Lean-Compatible Patterns:")
    print("  - All models frozen=True (immutable)")
    print("  - Pure functions (no side effects)")
    print("  - Total functions (Optional[T], no exceptions)")
    print("  - Explicit types (no Any)")
    print("  - Validators → Lean proof obligations")
    print()
    print("🎯 Ready for pipeline executor implementation!")
    print()


if __name__ == "__main__":
    main()
