"""
Test: MathematicalRegistry with Tags and Relationships.

Demonstrates:
1. Adding objects to registry
2. Querying by tags (any/all/none)
3. Querying by type
4. Querying relationships
5. Reference resolution
6. Referential integrity validation
7. Registry statistics
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
from fragile.proofs import CombinedTagQuery, TagQuery
from fragile.proofs import MathematicalRegistry


def main() -> None:
    """Test mathematical registry."""
    print("=" * 70)
    print("MATHEMATICAL REGISTRY TEST")
    print("=" * 70)
    print()

    # ==========================================================================
    # Create Registry and Add Objects
    # ==========================================================================
    print("STEP 1: Create Registry and Add Objects")
    print("-" * 70)

    registry = MathematicalRegistry()

    # Create objects with different tags
    obj1 = create_simple_object(
        label="obj-discrete-particle-system",
        name="Discrete Particle System",
        expr="S_N = {x_i(t)}_{i=1}^N ⊂ ℝ^d",
        obj_type=ObjectType.SET,
        tags=["euclidean-gas", "discrete", "particle", "finite"],
    )

    obj2 = create_simple_object(
        label="obj-continuous-pde",
        name="Continuous PDE",
        expr="∂_t μ = ∆μ - ∇·(μ ∇U)",
        obj_type=ObjectType.FUNCTION,
        tags=["euclidean-gas", "continuous", "pde", "mean-field"],
    )

    obj3 = create_simple_object(
        label="obj-adaptive-gas-system",
        name="Adaptive Gas System",
        expr="S_adaptive = {x_i, v_i}",
        obj_type=ObjectType.SET,
        tags=["adaptive-gas", "discrete", "particle", "kinetic"],
    )

    # Add to registry
    registry.add_all([obj1, obj2, obj3])

    print(f"✓ Added {registry.count_total()} objects to registry")
    print(f"  Statistics: {registry.count_by_type()}")
    print()

    # ==========================================================================
    # Query by Tags (ANY)
    # ==========================================================================
    print("STEP 2: Query by Tags (ANY)")
    print("-" * 70)

    query_any = TagQuery(tags=["discrete", "continuous"], mode="any")
    result = registry.query_by_tag(query_any)

    print(f"Query: Objects with tags 'discrete' OR 'continuous'")
    print(f"Found: {result.count()} objects")
    for obj in result.matches:
        print(f"  - {obj.label}: {obj.tags}")
    print()

    # ==========================================================================
    # Query by Tags (ALL)
    # ==========================================================================
    print("STEP 3: Query by Tags (ALL)")
    print("-" * 70)

    query_all = TagQuery(tags=["euclidean-gas", "discrete"], mode="all")
    result = registry.query_by_tag(query_all)

    print(f"Query: Objects with tags 'euclidean-gas' AND 'discrete'")
    print(f"Found: {result.count()} objects")
    for obj in result.matches:
        print(f"  - {obj.label}: {obj.tags}")
    print()

    # ==========================================================================
    # Query by Tags (COMBINED)
    # ==========================================================================
    print("STEP 4: Query by Tags (COMBINED)")
    print("-" * 70)

    query_combined = CombinedTagQuery(
        must_have=["discrete"],
        any_of=["particle", "swarm"],
        must_not_have=["deprecated"]
    )
    result = registry.query_by_tags(query_combined)

    print(f"Query: Objects that:")
    print(f"  - MUST have: discrete")
    print(f"  - ANY of: particle, swarm")
    print(f"  - MUST NOT have: deprecated")
    print(f"Found: {result.count()} objects")
    for obj in result.matches:
        print(f"  - {obj.label}: {obj.tags}")
    print()

    # ==========================================================================
    # Add Relationships
    # ==========================================================================
    print("STEP 5: Add Relationships")
    print("-" * 70)

    rel1 = Relationship(
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
                expression="O(N^{-1/d})"
            )
        ],
        tags=["mean-field", "discrete-continuous"],
    )

    rel2 = Relationship(
        label="rel-euclidean-adaptive-extension",
        relationship_type=RelationType.EXTENSION,
        source_object="obj-discrete-particle-system",
        target_object="obj-adaptive-gas-system",
        bidirectional=False,
        established_by="thm-adaptive-extension",
        expression="Euclidean Gas ⊂ Adaptive Gas",
        tags=["framework-extension"],
    )

    registry.add_all([rel1, rel2])

    print(f"✓ Added {len(registry.get_all_relationships())} relationships")
    for rel in registry.get_all_relationships():
        direction = "↔" if rel.is_symmetric() else "→"
        print(f"  - {rel.label}")
        print(f"    {rel.source_object} {direction} {rel.target_object}")
    print()

    # ==========================================================================
    # Query Relationships
    # ==========================================================================
    print("STEP 6: Query Relationships")
    print("-" * 70)

    obj_id = "obj-discrete-particle-system"
    related = registry.get_related_objects(obj_id)

    print(f"Query: Objects related to '{obj_id}'")
    print(f"Found: {len(related)} related objects")
    for rel_obj_id in related:
        print(f"  - {rel_obj_id}")
    print()

    # Get relationships for object
    rels = registry.get_relationships_for_object(obj_id)
    print(f"Relationships involving '{obj_id}': {len(rels)}")
    for rel in rels:
        print(f"  - {rel.label} ({rel.relationship_type.value})")
    print()

    # ==========================================================================
    # Query by Relationship Type
    # ==========================================================================
    print("STEP 7: Query by Relationship Type")
    print("-" * 70)

    equiv_rels = registry.get_relationships_by_type("equivalence")
    print(f"Equivalence relationships: {len(equiv_rels)}")
    for rel in equiv_rels:
        print(f"  - {rel.label}")
        print(f"    {rel.source_object} ≡ {rel.target_object}")
    print()

    # ==========================================================================
    # Add Theorem
    # ==========================================================================
    print("STEP 8: Add Theorem Establishing Relationships")
    print("-" * 70)

    thm = TheoremBox(
        label="thm-mean-field-equivalence",
        name="Mean Field Equivalence Theorem",
        input_objects=["obj-discrete-particle-system", "obj-continuous-pde"],
        output_type=TheoremOutputType.EQUIVALENCE,
        relations_established=[rel1],
    )

    registry.add(thm)

    print(f"✓ Added theorem: {thm.label}")
    print(f"  Input objects: {thm.input_objects}")
    print(f"  Relations established: {len(thm.relations_established)}")
    print()

    # ==========================================================================
    # Validation
    # ==========================================================================
    print("STEP 9: Referential Integrity Validation")
    print("-" * 70)

    missing = registry.validate_referential_integrity()

    if not missing:
        print("✓ All references are valid (referential integrity maintained)")
    else:
        print(f"✗ Found {len(missing)} missing references:")
        for msg in missing:
            print(f"  - {msg}")
    print()

    # ==========================================================================
    # Statistics
    # ==========================================================================
    print("STEP 10: Registry Statistics")
    print("-" * 70)

    stats = registry.get_statistics()

    print(f"Total objects: {stats['total_objects']}")
    print(f"Counts by type:")
    for type_name, count in stats['counts_by_type'].items():
        if count > 0:
            print(f"  - {type_name}: {count}")
    print(f"Total unique tags: {stats['total_tags']}")
    print(f"All tags: {', '.join(stats['all_tags'])}")
    print()

    # ==========================================================================
    # Query by Type
    # ==========================================================================
    print("STEP 11: Query by Type")
    print("-" * 70)

    result = registry.query_by_type("MathematicalObject")
    print(f"Query: All MathematicalObject instances")
    print(f"Found: {result.count()} objects")
    for obj in result.matches:
        print(f"  - {obj.label}")
    print()

    # ==========================================================================
    # Get Specific Objects
    # ==========================================================================
    print("STEP 12: Get Specific Objects")
    print("-" * 70)

    obj = registry.get_object("obj-discrete-particle-system")
    if obj:
        print(f"✓ Retrieved: {obj.label}")
        print(f"  Name: {obj.name}")
        print(f"  Tags: {obj.tags}")
        print(f"  Type: {obj.object_type.value}")
    else:
        print("✗ Object not found")
    print()

    rel = registry.get_relationship("rel-discrete-continuous-equivalence")
    if rel:
        print(f"✓ Retrieved: {rel.label}")
        print(f"  Type: {rel.relationship_type.value}")
        print(f"  Bidirectional: {rel.is_symmetric()}")
        print(f"  Expression: {rel.expression}")
    print()

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("✅ Mathematical Registry Features:")
    print("  1. Add/get/remove operations with uniqueness checking")
    print("  2. Tag-based queries (any/all/none/combined)")
    print("  3. Type-based queries")
    print("  4. Relationship queries (by object, by type)")
    print("  5. Referential integrity validation")
    print("  6. Statistics and counts")
    print()
    print(f"✅ Current Registry State:")
    print(f"  - {registry.count_total()} total objects")
    print(f"  - {len(registry.get_all_objects())} mathematical objects")
    print(f"  - {len(registry.get_all_relationships())} relationships")
    print(f"  - {len(registry.get_all_theorems())} theorems")
    print(f"  - {len(registry.get_all_tags())} unique tags")
    print()
    print("🎯 Ready for storage layer (save/load JSON with references)!")
    print()


if __name__ == "__main__":
    main()
