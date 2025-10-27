"""
Complete Workflow: End-to-End Demonstration.

This example demonstrates the complete relationship system workflow:
1. Create mathematical objects and relationships
2. Add to registry with tag-based organization
3. Save to JSON storage
4. Load from storage
5. Build relationship graph
6. Perform graph analysis
7. Compute equivalence classes
8. Analyze framework flow

This is the recommended workflow for using the relationship system.
"""

import sys
from pathlib import Path
import shutil

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
from fragile.proofs import load_registry_from_directory, save_registry_to_directory
from fragile.proofs import (
    EquivalenceClassifier,
    ObjectLineage,
    build_framework_flow_from_registry,
    build_relationship_graph_from_registry,
)


def main() -> None:
    """Demonstrate complete workflow."""
    print("=" * 70)
    print("COMPLETE WORKFLOW: RELATIONSHIP SYSTEM")
    print("=" * 70)
    print()
    print("This example demonstrates the recommended workflow:")
    print("  Registry → Storage → Graph Analysis → Insights")
    print()

    # ==========================================================================
    # STEP 1: Create Mathematical Framework
    # ==========================================================================
    print("STEP 1: Create Mathematical Framework")
    print("-" * 70)

    registry = MathematicalRegistry()

    # Create core objects
    obj_discrete = create_simple_object(
        label="obj-euclidean-gas-discrete",
        name="Euclidean Gas (Discrete)",
        expr="S_N = {(x_i, v_i, r_i)}_{i=1}^N",
        obj_type=ObjectType.SET,
        tags=["euclidean-gas", "discrete", "particle-system", "core"],
    )

    obj_continuous = create_simple_object(
        label="obj-euclidean-gas-continuous",
        name="Euclidean Gas (Continuous)",
        expr="∂_t μ = L_kin μ + L_clone μ",
        obj_type=ObjectType.FUNCTION,
        tags=["euclidean-gas", "continuous", "pde", "mean-field", "core"],
    )

    obj_qsd = create_simple_object(
        label="obj-quasi-stationary-distribution",
        name="Quasi-Stationary Distribution",
        expr="QSD: μ_∞ = lim_{t→∞} μ_t",
        obj_type=ObjectType.FUNCTION,
        tags=["euclidean-gas", "convergence", "equilibrium"],
    )

    obj_adaptive = create_simple_object(
        label="obj-adaptive-gas",
        name="Adaptive Gas",
        expr="S_adaptive = {(x_i, v_i, F_i)}",
        obj_type=ObjectType.SET,
        tags=["adaptive-gas", "discrete", "particle-system", "extended"],
    )

    obj_geometric = create_simple_object(
        label="obj-geometric-gas",
        name="Geometric Gas (Riemannian)",
        expr="S_M = {(x_i, v_i)} on (M, g)",
        obj_type=ObjectType.SET,
        tags=["geometric-gas", "riemannian", "manifold", "extended"],
    )

    # Create relationships
    rel_mean_field = Relationship(
        label="rel-discrete-continuous-equivalence",
        relationship_type=RelationType.EQUIVALENCE,
        source_object="obj-euclidean-gas-discrete",
        target_object="obj-euclidean-gas-continuous",
        bidirectional=True,
        established_by="thm-mean-field-limit",
        expression="S_N ≡ μ_t + O(N^{-1/d}) as N → ∞",
        properties=[
            RelationshipProperty(
                label="convergence-rate",
                expression="O(N^{-1/d})",
                description="Wasserstein distance convergence rate",
            )
        ],
        tags=["mean-field", "convergence", "fundamental"],
    )

    rel_qsd_convergence = Relationship(
        label="rel-continuous-qsd-reduction",
        relationship_type=RelationType.REDUCTION,
        source_object="obj-euclidean-gas-continuous",
        target_object="obj-quasi-stationary-distribution",
        bidirectional=False,
        established_by="thm-exponential-convergence",
        expression="μ_t → μ_∞ exponentially fast",
        properties=[
            RelationshipProperty(
                label="convergence-rate-exponential",
                expression="O(e^{-λt})",
                description="Exponential convergence with rate λ",
            )
        ],
        tags=["convergence", "qsd", "fundamental"],
    )

    rel_adaptive_extension = Relationship(
        label="rel-euclidean-adaptive-extension",
        relationship_type=RelationType.EXTENSION,
        source_object="obj-euclidean-gas-discrete",
        target_object="obj-adaptive-gas",
        bidirectional=False,
        established_by="thm-adaptive-gas-extension",
        expression="Euclidean Gas ⊂ Adaptive Gas (ε_F = 0)",
        tags=["framework-extension", "adaptive"],
    )

    rel_geometric = Relationship(
        label="rel-euclidean-geometric-generalization",
        relationship_type=RelationType.GENERALIZATION,
        source_object="obj-euclidean-gas-discrete",
        target_object="obj-geometric-gas",
        bidirectional=False,
        established_by="thm-geometric-generalization",
        expression="Euclidean Gas → Geometric Gas (M = ℝ^d, g = I)",
        tags=["framework-extension", "geometric"],
    )

    # Create theorems
    thm_mean_field = TheoremBox(
        label="thm-mean-field-limit",
        name="Mean Field Limit Theorem",
        input_objects=["obj-euclidean-gas-discrete"],
        output_type=TheoremOutputType.EQUIVALENCE,
        relations_established=[rel_mean_field],
    )

    thm_convergence = TheoremBox(
        label="thm-exponential-convergence",
        name="Exponential Convergence to QSD",
        input_objects=["obj-euclidean-gas-continuous"],
        output_type=TheoremOutputType.PROPERTY,
        relations_established=[rel_qsd_convergence],
    )

    thm_adaptive = TheoremBox(
        label="thm-adaptive-gas-extension",
        name="Adaptive Gas Extension",
        input_objects=["obj-euclidean-gas-discrete"],
        output_type=TheoremOutputType.EXTENSION,
        relations_established=[rel_adaptive_extension],
    )

    thm_geometric = TheoremBox(
        label="thm-geometric-generalization",
        name="Geometric Gas Generalization",
        input_objects=["obj-euclidean-gas-discrete"],
        output_type=TheoremOutputType.CONSTRUCTION,
        relations_established=[rel_geometric],
    )

    # Add all to registry
    registry.add_all([obj_discrete, obj_continuous, obj_qsd, obj_adaptive, obj_geometric])
    registry.add_all([rel_mean_field, rel_qsd_convergence, rel_adaptive_extension, rel_geometric])
    registry.add_all([thm_mean_field, thm_convergence, thm_adaptive, thm_geometric])

    print(f"✓ Created framework:")
    print(f"  Objects: {len(registry.get_all_objects())}")
    print(f"  Relationships: {len(registry.get_all_relationships())}")
    print(f"  Theorems: {len(registry.get_all_theorems())}")
    print(f"  Tags: {len(registry.get_all_tags())}")
    print()

    # ==========================================================================
    # STEP 2: Query by Tags
    # ==========================================================================
    print("STEP 2: Query by Tags")
    print("-" * 70)

    # Simple query
    query = TagQuery(tags=["core"], mode="all")
    result = registry.query_by_tag(query)
    print(f"Core objects (tag='core'): {result.count()}")
    for obj in result.matches:
        print(f"  - {obj.label}")
    print()

    # Combined query
    combined_query = CombinedTagQuery(
        must_have=["discrete"],
        any_of=["euclidean-gas", "adaptive-gas"],
        must_not_have=["deprecated"],
    )
    result2 = registry.query_by_tags(combined_query)
    print(f"Discrete particle systems: {result2.count()}")
    for obj in result2.matches:
        print(f"  - {obj.label}")
    print()

    # ==========================================================================
    # STEP 3: Save to Storage
    # ==========================================================================
    print("STEP 3: Save to Storage")
    print("-" * 70)

    storage_dir = Path(__file__).parent / "complete_workflow_storage"

    # Clean up previous data
    if storage_dir.exists():
        shutil.rmtree(storage_dir)

    # Save
    save_registry_to_directory(registry, storage_dir)

    file_count = len(list(storage_dir.rglob("*.json")))
    print(f"✓ Saved to: {storage_dir}")
    print(f"  Files created: {file_count}")
    print()

    # ==========================================================================
    # STEP 4: Load from Storage
    # ==========================================================================
    print("STEP 4: Load from Storage")
    print("-" * 70)

    loaded_registry = load_registry_from_directory(MathematicalRegistry, storage_dir)

    print(f"✓ Loaded from storage:")
    print(f"  Objects: {len(loaded_registry.get_all_objects())}")
    print(f"  Relationships: {len(loaded_registry.get_all_relationships())}")
    print(f"  Theorems: {len(loaded_registry.get_all_theorems())}")
    print()

    # Verify integrity
    missing = loaded_registry.validate_referential_integrity()
    if not missing:
        print("✓ Referential integrity verified")
    else:
        print(f"✗ Found {len(missing)} broken references")
    print()

    # ==========================================================================
    # STEP 5: Build Relationship Graph
    # ==========================================================================
    print("STEP 5: Build Relationship Graph")
    print("-" * 70)

    graph = build_relationship_graph_from_registry(loaded_registry)

    print(f"✓ Built graph:")
    print(f"  Nodes: {graph.node_count()}")
    print(f"  Edges: {graph.edge_count()}")
    print()

    # Show connectivity
    component = graph.get_connected_component("obj-euclidean-gas-discrete")
    print(f"Connected component from 'obj-euclidean-gas-discrete':")
    print(f"  Size: {len(component)} nodes")
    print(f"  Nodes: {sorted(component)}")
    print()

    # ==========================================================================
    # STEP 6: Compute Equivalence Classes
    # ==========================================================================
    print("STEP 6: Compute Equivalence Classes")
    print("-" * 70)

    classifier = EquivalenceClassifier(graph)
    eq_classes = classifier.compute_equivalence_classes()

    print(f"Equivalence classes: {len(eq_classes)}")
    for i, eq_class in enumerate(eq_classes, 1):
        if eq_class.size() > 1:
            print(f"  Class {i} (size {eq_class.size()}):")
            print(f"    {' ≡ '.join(eq_class.members)}")
    print()

    # Test equivalence
    if classifier.are_equivalent("obj-euclidean-gas-discrete", "obj-euclidean-gas-continuous"):
        print("✓ Discrete and continuous formulations are equivalent (mean field limit)")
    print()

    # ==========================================================================
    # STEP 7: Trace Object Lineage
    # ==========================================================================
    print("STEP 7: Trace Object Lineage")
    print("-" * 70)

    lineage = ObjectLineage(graph)

    # Get descendants
    descendants = lineage.get_descendants("obj-euclidean-gas-discrete", max_depth=2)
    print(f"Descendants of 'obj-euclidean-gas-discrete':")
    print(f"  Count: {len(descendants)}")
    for desc in sorted(descendants):
        print(f"    → {desc}")
    print()

    # Find path
    path = graph.find_path("obj-euclidean-gas-discrete", "obj-quasi-stationary-distribution")
    if path:
        print(f"Path to QSD:")
        print(f"  {' → '.join(path)}")
    print()

    # ==========================================================================
    # STEP 8: Analyze Framework Flow
    # ==========================================================================
    print("STEP 8: Analyze Framework Flow")
    print("-" * 70)

    flow = build_framework_flow_from_registry(loaded_registry, graph)

    # Get framework layers
    layers = flow.get_framework_layers()
    print(f"Framework has {len(layers)} layers:")
    for i, layer in enumerate(layers):
        print(f"  Layer {i}: {len(layer)} theorems")
        for thm_id in sorted(layer):
            thm = loaded_registry.get_theorem(thm_id)
            if thm:
                print(f"    - {thm_id}: {thm.name}")
    print()

    # Show dependencies
    thm_id = "thm-exponential-convergence"
    deps = flow.get_theorem_dependencies(thm_id)
    print(f"Dependencies of '{thm_id}':")
    if deps:
        print(f"  Depends on: {', '.join(sorted(deps))}")
    else:
        print(f"  No dependencies (base theorem)")
    print()

    # ==========================================================================
    # STEP 9: Framework Statistics
    # ==========================================================================
    print("STEP 9: Framework Statistics")
    print("-" * 70)

    stats = loaded_registry.get_statistics()

    print("Framework overview:")
    print(f"  Total objects: {stats['total_objects']}")
    print(f"  Total tags: {stats['total_tags']}")
    print()

    print("Objects by type:")
    for obj_type, count in stats['counts_by_type'].items():
        if count > 0:
            print(f"  - {obj_type}: {count}")
    print()

    print("Most used tags:")
    all_tags = sorted(stats['all_tags'])
    print(f"  {', '.join(all_tags[:10])}")
    if len(all_tags) > 10:
        print(f"  ... and {len(all_tags) - 10} more")
    print()

    # ==========================================================================
    # STEP 10: Summary
    # ==========================================================================
    print("=" * 70)
    print("WORKFLOW SUMMARY")
    print("=" * 70)
    print()
    print("✅ Complete workflow demonstrated:")
    print("  1. ✓ Created mathematical framework")
    print("  2. ✓ Organized with tags")
    print("  3. ✓ Saved to JSON storage")
    print("  4. ✓ Loaded from storage")
    print("  5. ✓ Built relationship graph")
    print("  6. ✓ Computed equivalence classes")
    print("  7. ✓ Traced object lineage")
    print("  8. ✓ Analyzed framework flow")
    print("  9. ✓ Generated statistics")
    print()
    print("📊 Framework metrics:")
    print(f"  - {stats['total_objects']} mathematical objects")
    print(f"  - {len(eq_classes)} equivalence classes")
    print(f"  - {len(layers)} theorem dependency layers")
    print(f"  - {len(component)} objects in main component")
    print()
    print("💾 Storage:")
    print(f"  - Location: {storage_dir}")
    print(f"  - Files: {file_count} JSON files")
    print()
    print("🎯 System ready for production use!")
    print()
    print("Next steps:")
    print("  - Add more mathematical objects and relationships")
    print("  - Use tag queries to filter and analyze")
    print("  - Build visualizations from graph structure")
    print("  - Export to other formats (GraphML, DOT, etc.)")
    print()


if __name__ == "__main__":
    main()
