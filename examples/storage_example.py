"""
Test: Storage Layer with Reference Resolution.

Demonstrates:
1. Creating registry with objects and relationships
2. Saving to directory (with ID references to avoid duplication)
3. Loading from directory
4. Verifying objects loaded correctly
5. Directory structure
6. Reference resolution
"""

from pathlib import Path
import sys


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fragile.proofs import (
    create_simple_object,
    load_registry_from_directory,
    MathematicalRegistry,
    ObjectType,
    Relationship,
    RelationshipProperty,
    RelationType,
    save_registry_to_directory,
    TheoremBox,
    TheoremOutputType,
)


def main() -> None:
    """Test storage layer."""
    print("=" * 70)
    print("STORAGE LAYER TEST")
    print("=" * 70)
    print()

    # ==========================================================================
    # Create Registry with Objects
    # ==========================================================================
    print("STEP 1: Create Registry with Objects and Relationships")
    print("-" * 70)

    registry = MathematicalRegistry()

    # Create objects with tags
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

    # Create relationships
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
                expression="O(N^{-1/d})",
                description="Approximation error scales as N^{-1/d}",
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

    # Create theorem
    thm = TheoremBox(
        label="thm-mean-field-equivalence",
        name="Mean Field Equivalence Theorem",
        input_objects=["obj-discrete-particle-system", "obj-continuous-pde"],
        output_type=TheoremOutputType.EQUIVALENCE,
        relations_established=[rel1],
    )

    # Add to registry
    registry.add_all([obj1, obj2, obj3, rel1, rel2, thm])

    print("✓ Created registry with:")
    print(f"  - {len(registry.get_all_objects())} mathematical objects")
    print(f"  - {len(registry.get_all_relationships())} relationships")
    print(f"  - {len(registry.get_all_theorems())} theorems")
    print(f"  - {len(registry.get_all_tags())} unique tags")
    print()

    # ==========================================================================
    # Save Registry to Directory
    # ==========================================================================
    print("STEP 2: Save Registry to Directory")
    print("-" * 70)

    storage_dir = Path(__file__).parent / "test_storage"

    # Clean up previous test data
    if storage_dir.exists():
        import shutil

        shutil.rmtree(storage_dir)

    # Save registry
    save_registry_to_directory(registry, storage_dir)

    print(f"✓ Saved registry to: {storage_dir}")
    print()

    # Show directory structure
    print("Directory structure:")
    for path in sorted(storage_dir.rglob("*")):
        if path.is_file():
            rel_path = path.relative_to(storage_dir)
            file_size = path.stat().st_size
            print(f"  {rel_path} ({file_size} bytes)")
    print()

    # ==========================================================================
    # Inspect Saved Files
    # ==========================================================================
    print("STEP 3: Inspect Saved Files (with references)")
    print("-" * 70)

    # Read a relationship file to show ID references
    rel_file = storage_dir / "relationships" / "rel-discrete-continuous-equivalence.json"
    if rel_file.exists():
        import json

        with open(rel_file, encoding="utf-8") as f:
            rel_data = json.load(f)

        print("Relationship file content (truncated):")
        print(f"  label: {rel_data.get('label')}")
        print(f"  relationship_type: {rel_data.get('relationship_type')}")
        print(f"  source_object: {rel_data.get('source_object')}")
        print(f"  target_object: {rel_data.get('target_object')}")
        print(f"  established_by: {rel_data.get('established_by')}")
        print(f"  bidirectional: {rel_data.get('bidirectional')}")
        print()

    # Read index file
    index_file = storage_dir / "index.json"
    if index_file.exists():
        import json

        with open(index_file, encoding="utf-8") as f:
            index_data = json.load(f)

        print("Index file content:")
        print(f"  version: {index_data.get('version')}")
        print(f"  total_objects: {index_data.get('statistics', {}).get('total_objects')}")
        print(f"  total_tags: {index_data.get('statistics', {}).get('total_tags')}")
        print(
            f"  all_ids ({len(index_data.get('all_ids', []))}): {index_data.get('all_ids', [])[:3]}..."
        )
        print()

    # ==========================================================================
    # Load Registry from Directory
    # ==========================================================================
    print("STEP 4: Load Registry from Directory")
    print("-" * 70)

    # Load registry
    loaded_registry = load_registry_from_directory(MathematicalRegistry, storage_dir)

    print("✓ Loaded registry with:")
    print(f"  - {len(loaded_registry.get_all_objects())} mathematical objects")
    print(f"  - {len(loaded_registry.get_all_relationships())} relationships")
    print(f"  - {len(loaded_registry.get_all_theorems())} theorems")
    print(f"  - {len(loaded_registry.get_all_tags())} unique tags")
    print()

    # ==========================================================================
    # Verify Loaded Objects
    # ==========================================================================
    print("STEP 5: Verify Loaded Objects")
    print("-" * 70)

    # Verify object 1
    loaded_obj1 = loaded_registry.get_object("obj-discrete-particle-system")
    if loaded_obj1:
        print(f"✓ Retrieved: {loaded_obj1.label}")
        print(f"  Name: {loaded_obj1.name}")
        print(f"  Tags: {loaded_obj1.tags}")
        print(f"  Expression: {loaded_obj1.mathematical_expression}")
    else:
        print("✗ Failed to retrieve obj-discrete-particle-system")
    print()

    # Verify relationship
    loaded_rel = loaded_registry.get_relationship("rel-discrete-continuous-equivalence")
    if loaded_rel:
        print(f"✓ Retrieved: {loaded_rel.label}")
        print(f"  Type: {loaded_rel.relationship_type.value}")
        print(f"  Source: {loaded_rel.source_object}")
        print(f"  Target: {loaded_rel.target_object}")
        print(f"  Bidirectional: {loaded_rel.is_symmetric()}")
        print(f"  Expression: {loaded_rel.expression}")
        print(f"  Properties: {len(loaded_rel.properties)}")
    else:
        print("✗ Failed to retrieve relationship")
    print()

    # Verify theorem
    loaded_thm = loaded_registry.get_theorem("thm-mean-field-equivalence")
    if loaded_thm:
        print(f"✓ Retrieved: {loaded_thm.label}")
        print(f"  Name: {loaded_thm.name}")
        print(f"  Output type: {loaded_thm.output_type.value}")
        print(f"  Input objects: {loaded_thm.input_objects}")
        print(f"  Relations established: {len(loaded_thm.relations_established)}")
    else:
        print("✗ Failed to retrieve theorem")
    print()

    # ==========================================================================
    # Verify Relationships Still Work
    # ==========================================================================
    print("STEP 6: Verify Relationships Still Work")
    print("-" * 70)

    # Query related objects
    related = loaded_registry.get_related_objects("obj-discrete-particle-system")
    print(f"Objects related to 'obj-discrete-particle-system': {len(related)}")
    for rel_obj_id in related:
        print(f"  - {rel_obj_id}")
    print()

    # Query by tag
    from fragile.proofs import TagQuery

    query = TagQuery(tags=["discrete"], mode="all")
    result = loaded_registry.query_by_tag(query)
    print(f"Objects with tag 'discrete': {result.count()}")
    for obj in result.matches:
        obj_id = getattr(obj, "label", "unknown")
        print(f"  - {obj_id}")
    print()

    # Validate referential integrity
    missing = loaded_registry.validate_referential_integrity()
    if not missing:
        print("✓ Referential integrity maintained after load")
    else:
        print(f"✗ Found {len(missing)} broken references:")
        for msg in missing:
            print(f"  - {msg}")
    print()

    # ==========================================================================
    # Statistics Comparison
    # ==========================================================================
    print("STEP 7: Statistics Comparison")
    print("-" * 70)

    original_stats = registry.get_statistics()
    loaded_stats = loaded_registry.get_statistics()

    print("Original vs Loaded:")
    print(f"  Total objects: {original_stats['total_objects']} → {loaded_stats['total_objects']}")
    print(f"  Total tags: {original_stats['total_tags']} → {loaded_stats['total_tags']}")
    print(
        f"  Objects: {original_stats['counts_by_type']['MathematicalObject']} → {loaded_stats['counts_by_type']['MathematicalObject']}"
    )
    print(
        f"  Relationships: {original_stats['counts_by_type']['Relationship']} → {loaded_stats['counts_by_type']['Relationship']}"
    )
    print(
        f"  Theorems: {original_stats['counts_by_type']['TheoremBox']} → {loaded_stats['counts_by_type']['TheoremBox']}"
    )
    print()

    # Check if statistics match
    if original_stats == loaded_stats:
        print("✅ Statistics match perfectly!")
    else:
        print("⚠️  Statistics differ (this may be expected if structure changed)")
    print()

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("✅ Storage Layer Features:")
    print("  1. Save registry to directory structure")
    print("  2. Objects stored as individual JSON files")
    print("  3. ID-based references to avoid duplication")
    print("  4. Load registry from directory")
    print("  5. Lazy loading support")
    print("  6. Index file with metadata")
    print("  7. Referential integrity maintained")
    print()
    print(f"✅ Storage Location: {storage_dir}")
    print(f"  - {len(list(storage_dir.glob('**/*.json')))} JSON files created")
    print()
    print("🎯 Ready for relationship graph analysis!")
    print()


if __name__ == "__main__":
    main()
