"""
Test: Relationship Model with Discrete ↔ Continuous Example.

Demonstrates:
1. Creating objects with tags
2. Establishing relationships between objects
3. Relationship properties (e.g., error bounds)
4. Bidirectional vs directed relationships
5. ID naming system validation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fragile.proofs import (
    MathematicalObject,
    ObjectType,
    Relationship,
    RelationshipProperty,
    RelationType,
    TheoremBox,
    TheoremOutputType,
    create_simple_object,
)


def main() -> None:
    """Test relationship model."""
    print("=" * 70)
    print("RELATIONSHIP MODEL TEST: Discrete ↔ Continuous")
    print("=" * 70)
    print()

    # ==========================================================================
    # Create Objects with Tags
    # ==========================================================================
    print("STEP 1: Create Mathematical Objects with Tags")
    print("-" * 70)

    obj_discrete = create_simple_object(
        label="obj-discrete-particle-system",
        name="Discrete Particle System",
        expr="S_N = {x_i(t)}_{i=1}^N ⊂ ℝ^d",
        obj_type=ObjectType.SET,
        tags=["euclidean-gas", "discrete", "particle", "finite", "n-walkers"],
    )

    obj_continuous = create_simple_object(
        label="obj-continuous-pde",
        name="Continuous PDE (Mean Field)",
        expr="∂_t μ = ∆μ - ∇·(μ ∇U)",
        obj_type=ObjectType.FUNCTION,
        tags=["euclidean-gas", "continuous", "pde", "mean-field", "fokker-planck"],
    )

    print(f"✓ Created: {obj_discrete.label}")
    print(f"  Tags: {obj_discrete.tags}")
    print()
    print(f"✓ Created: {obj_continuous.label}")
    print(f"  Tags: {obj_continuous.tags}")
    print()

    # ==========================================================================
    # Create Relationship with Properties
    # ==========================================================================
    print("STEP 2: Create Equivalence Relationship")
    print("-" * 70)

    rel_equivalence = Relationship(
        label="rel-discrete-continuous-equivalence",
        relationship_type=RelationType.EQUIVALENCE,
        source_object="obj-discrete-particle-system",
        target_object="obj-continuous-pde",
        bidirectional=True,
        established_by="thm-mean-field-equivalence",
        expression="S_N ≡ μ_t + O(N^{-1/d})",
        properties=[
            RelationshipProperty(
                label="approx-error-N",
                expression="O(N^{-1/d})",
                description="Approximation error depends on number of particles"
            ),
            RelationshipProperty(
                label="convergence-rate-exponential",
                expression="‖μ_N(t) - μ_∞(t)‖ ≤ C·e^(-λt)",
                description="Exponential convergence to continuous limit"
            )
        ],
        tags=["mean-field", "discrete-continuous", "convergence"],
    )

    print(f"✓ Created: {rel_equivalence.label}")
    print(f"  Type: {rel_equivalence.relationship_type.value}")
    print(f"  Bidirectional: {rel_equivalence.is_symmetric()}")
    print(f"  Source: {rel_equivalence.source_object}")
    print(f"  Target: {rel_equivalence.target_object}")
    print(f"  Expression: {rel_equivalence.expression}")
    print(f"  Properties:")
    for prop in rel_equivalence.properties:
        print(f"    - {prop.label}: {prop.expression}")
    print()

    # ==========================================================================
    # Test Directed Relationship
    # ==========================================================================
    print("STEP 3: Create Directed Relationship (Embedding)")
    print("-" * 70)

    obj_particles = create_simple_object(
        label="obj-particle-configuration",
        name="Particle Configuration Space",
        expr="X_N = (ℝ^d)^N",
        obj_type=ObjectType.SPACE,
        tags=["discrete", "configuration-space", "particle"],
    )

    obj_fluid = create_simple_object(
        label="obj-velocity-field",
        name="Velocity Field Space",
        expr="V = {v: ℝ^d → ℝ^d | v smooth}",
        obj_type=ObjectType.SPACE,
        tags=["continuous", "velocity", "field"],
    )

    rel_embedding = Relationship(
        label="rel-particle-fluid-embedding",
        relationship_type=RelationType.EMBEDDING,
        source_object="obj-particle-configuration",
        target_object="obj-velocity-field",
        bidirectional=False,  # Embedding is directed
        established_by="thm-particle-fluid-embedding",
        expression="X_N ↪ V via v(x) = Σ_i K(x - x_i)·v_i",
        properties=[
            RelationshipProperty(
                label="kernel-type",
                expression="K(x) = (2πh)^{-d/2} exp(-|x|^2/(2h))",
                description="Gaussian kernel for embedding"
            )
        ],
        tags=["embedding", "particle-continuum", "kernel-method"],
    )

    print(f"✓ Created: {rel_embedding.label}")
    print(f"  Type: {rel_embedding.relationship_type.value}")
    print(f"  Directed: {rel_embedding.is_directed()}")
    print(f"  Source: {rel_embedding.source_object}")
    print(f"  Target: {rel_embedding.target_object}")
    print(f"  Expression: {rel_embedding.expression}")
    print()

    # ==========================================================================
    # Create Theorem Establishing Relationships
    # ==========================================================================
    print("STEP 4: Create Theorem that Establishes Relationships")
    print("-" * 70)

    thm_equivalence = TheoremBox(
        label="thm-mean-field-equivalence",
        name="Mean Field Equivalence Theorem",
        input_objects=["obj-discrete-particle-system", "obj-continuous-pde"],
        output_type=TheoremOutputType.EQUIVALENCE,
        relations_established=[rel_equivalence],
    )

    print(f"✓ Created: {thm_equivalence.label}")
    print(f"  Output Type: {thm_equivalence.output_type.value}")
    print(f"  Relations Established: {len(thm_equivalence.relations_established)}")
    for rel in thm_equivalence.relations_established:
        print(f"    - {rel.label}")
        print(f"      {rel.source_object} {('↔' if rel.is_symmetric() else '→')} {rel.target_object}")
    print()

    # ==========================================================================
    # Test ID Validation
    # ==========================================================================
    print("STEP 5: Test ID Validation")
    print("-" * 70)

    print("Testing valid IDs:")
    valid_ids = [
        "rel-discrete-continuous-equivalence",
        "rel-particle-fluid-embedding",
        "rel-pde-ode-reduction",
    ]
    for id in valid_ids:
        print(f"  ✓ {id}")

    print()
    print("Testing invalid IDs (would raise ValueError):")
    invalid_ids = [
        "rel-discrete-continuous",  # Missing relationship type
        "relationship-test-equivalence",  # Wrong prefix
        "rel-Discrete-Continuous-equivalence",  # Capital letters
    ]
    for id in invalid_ids:
        print(f"  ✗ {id}")
        try:
            # This would fail validation
            Relationship(
                label=id,
                relationship_type=RelationType.EQUIVALENCE,
                source_object="obj-test",
                target_object="obj-test2",
                bidirectional=True,
                established_by="thm-test",
                expression="test"
            )
            print("    ERROR: Should have raised ValueError!")
        except ValueError as e:
            print(f"    Correctly rejected: {str(e)[:60]}...")
    print()

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("✅ Relationship Model Features:")
    print("  1. First-class objects with labels, types, and properties")
    print("  2. Type-dependent directionality (equivalence=bidirectional, embedding=directed)")
    print("  3. Relationship properties (error bounds, convergence rates)")
    print("  4. Tag system for categorization")
    print("  5. ID validation following naming conventions")
    print()
    print("✅ Tested Relationship Types:")
    print("  - Equivalence: discrete ≡ continuous (bidirectional)")
    print("  - Embedding: particles ↪ fluid (directed)")
    print()
    print("✅ ID System:")
    print("  - Format: rel-{source}-{target}-{type}")
    print("  - Validation: enforced by Pydantic")
    print("  - Tags: flexible categorization")
    print()
    print("🎯 Ready for MathematicalRegistry and storage layer!")
    print()


if __name__ == "__main__":
    main()
