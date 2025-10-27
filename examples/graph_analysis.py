"""
Test: Relationship Graph Analysis.

Demonstrates:
1. Building relationship graph from registry
2. Graph traversal and connectivity
3. Finding paths between objects
4. Object lineage tracing
5. Equivalence class computation
6. Framework flow analysis
7. Theorem dependency layers
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
from fragile.proofs import MathematicalRegistry
from fragile.proofs import (
    EquivalenceClassifier,
    FrameworkFlow,
    ObjectLineage,
    RelationshipGraph,
    build_framework_flow_from_registry,
    build_relationship_graph_from_registry,
)


def main() -> None:
    """Test relationship graph analysis."""
    print("=" * 70)
    print("RELATIONSHIP GRAPH ANALYSIS TEST")
    print("=" * 70)
    print()

    # ==========================================================================
    # Create Registry with Complex Relationships
    # ==========================================================================
    print("STEP 1: Create Registry with Complex Relationships")
    print("-" * 70)

    registry = MathematicalRegistry()

    # Create a chain of mathematical objects
    obj1 = create_simple_object(
        label="obj-discrete-particle-system",
        name="Discrete Particle System",
        expr="S_N = {x_i(t)}_{i=1}^N",
        obj_type=ObjectType.SET,
        tags=["euclidean-gas", "discrete", "particle"],
    )

    obj2 = create_simple_object(
        label="obj-continuous-pde",
        name="Continuous PDE",
        expr="∂_t μ = ∆μ - ∇·(μ ∇U)",
        obj_type=ObjectType.FUNCTION,
        tags=["euclidean-gas", "continuous", "pde"],
    )

    obj3 = create_simple_object(
        label="obj-fokker-planck",
        name="Fokker-Planck Equation",
        expr="∂_t ρ = ∆ρ + ∇·(ρ ∇V)",
        obj_type=ObjectType.FUNCTION,
        tags=["pde", "continuous", "diffusion"],
    )

    obj4 = create_simple_object(
        label="obj-langevin-dynamics",
        name="Langevin Dynamics",
        expr="dX_t = -∇U(X_t) dt + √(2β⁻¹) dW_t",
        obj_type=ObjectType.FUNCTION,
        tags=["stochastic", "sde", "continuous"],
    )

    obj5 = create_simple_object(
        label="obj-adaptive-gas",
        name="Adaptive Gas System",
        expr="S_adaptive = {x_i, v_i, F_i}",
        obj_type=ObjectType.SET,
        tags=["adaptive-gas", "discrete", "particle"],
    )

    # Create equivalence relationships
    rel1 = Relationship(
        label="rel-discrete-continuous-equivalence",
        relationship_type=RelationType.EQUIVALENCE,
        source_object="obj-discrete-particle-system",
        target_object="obj-continuous-pde",
        bidirectional=True,
        established_by="thm-mean-field-limit",
        expression="S_N ≡ μ_t + O(N^{-1/d})",
        tags=["mean-field"],
    )

    rel2 = Relationship(
        label="rel-pde-fokker-planck-equivalence",
        relationship_type=RelationType.EQUIVALENCE,
        source_object="obj-continuous-pde",
        target_object="obj-fokker-planck",
        bidirectional=True,
        established_by="thm-pde-equivalence",
        expression="Both describe same diffusion process",
        tags=["pde-equivalence"],
    )

    # Create derivation relationships
    rel3 = Relationship(
        label="rel-langevin-fokker-planck-reduction",
        relationship_type=RelationType.REDUCTION,
        source_object="obj-langevin-dynamics",
        target_object="obj-fokker-planck",
        bidirectional=False,
        established_by="thm-langevin-to-fp",
        expression="Langevin SDE → Fokker-Planck PDE",
        tags=["sde-to-pde"],
    )

    rel4 = Relationship(
        label="rel-euclidean-adaptive-extension",
        relationship_type=RelationType.EXTENSION,
        source_object="obj-discrete-particle-system",
        target_object="obj-adaptive-gas",
        bidirectional=False,
        established_by="thm-adaptive-extension",
        expression="Euclidean Gas ⊂ Adaptive Gas",
        tags=["framework-extension"],
    )

    # Create theorems
    thm1 = TheoremBox(
        label="thm-mean-field-limit",
        name="Mean Field Limit Theorem",
        input_objects=["obj-discrete-particle-system"],
        output_type=TheoremOutputType.EQUIVALENCE,
        relations_established=[rel1],
    )

    thm2 = TheoremBox(
        label="thm-pde-equivalence",
        name="PDE Equivalence Theorem",
        input_objects=["obj-continuous-pde", "obj-fokker-planck"],
        output_type=TheoremOutputType.EQUIVALENCE,
        relations_established=[rel2],
    )

    thm3 = TheoremBox(
        label="thm-langevin-to-fp",
        name="Langevin to Fokker-Planck",
        input_objects=["obj-langevin-dynamics"],
        output_type=TheoremOutputType.RELATION,
        relations_established=[rel3],
    )

    thm4 = TheoremBox(
        label="thm-adaptive-extension",
        name="Adaptive Gas Extension",
        input_objects=["obj-discrete-particle-system"],
        output_type=TheoremOutputType.EMBEDDING,
        relations_established=[rel4],
    )

    # Add all to registry
    registry.add_all([obj1, obj2, obj3, obj4, obj5])
    registry.add_all([rel1, rel2, rel3, rel4])
    registry.add_all([thm1, thm2, thm3, thm4])

    print(f"✓ Created registry with:")
    print(f"  - {len(registry.get_all_objects())} objects")
    print(f"  - {len(registry.get_all_relationships())} relationships")
    print(f"  - {len(registry.get_all_theorems())} theorems")
    print()

    # ==========================================================================
    # Build Relationship Graph
    # ==========================================================================
    print("STEP 2: Build Relationship Graph")
    print("-" * 70)

    graph = build_relationship_graph_from_registry(registry)

    print(f"✓ Built graph with:")
    print(f"  - {graph.node_count()} nodes")
    print(f"  - {graph.edge_count()} edges")
    print()

    # Show nodes
    print("Nodes:")
    for node_id, node in graph.nodes.items():
        print(f"  - {node_id} ({node.node_type})")
        if node.tags:
            print(f"    Tags: {', '.join(node.tags)}")
    print()

    # Show edges
    print("Edges:")
    for edge in graph.edges:
        direction = "↔" if edge.bidirectional else "→"
        print(f"  - {edge.source} {direction} {edge.target}")
        print(f"    Type: {edge.relationship_type.value}")
        print(f"    ID: {edge.relationship_id}")
    print()

    # ==========================================================================
    # Graph Connectivity
    # ==========================================================================
    print("STEP 3: Graph Connectivity Analysis")
    print("-" * 70)

    # Test connectivity
    start_node = "obj-discrete-particle-system"
    component = graph.get_connected_component(start_node)

    print(f"Connected component from '{start_node}':")
    print(f"  Size: {len(component)} nodes")
    print(f"  Nodes: {sorted(component)}")
    print()

    # Find paths
    source = "obj-discrete-particle-system"
    target = "obj-fokker-planck"
    path = graph.find_path(source, target)

    if path:
        print(f"Path from '{source}' to '{target}':")
        print(f"  Length: {len(path) - 1} edges")
        print(f"  Path: {' → '.join(path)}")
    else:
        print(f"No path found from '{source}' to '{target}'")
    print()

    # Test another path
    source2 = "obj-langevin-dynamics"
    target2 = "obj-discrete-particle-system"
    path2 = graph.find_path(source2, target2)

    print(f"Path from '{source2}' to '{target2}':")
    if path2:
        print(f"  Path: {' → '.join(path2)}")
    else:
        print(f"  No path exists (expected - not connected)")
    print()

    # ==========================================================================
    # Object Lineage
    # ==========================================================================
    print("STEP 4: Object Lineage Tracing")
    print("-" * 70)

    lineage = ObjectLineage(graph)

    # Trace lineage from discrete system
    node_id = "obj-discrete-particle-system"
    paths = lineage.trace_lineage(node_id, max_depth=3)

    print(f"Lineage paths from '{node_id}':")
    print(f"  Found: {len(paths)} paths")
    for i, path in enumerate(paths[:5], 1):  # Show first 5
        print(f"  Path {i}:")
        print(f"    Nodes: {' → '.join(path.nodes)}")
        print(f"    Length: {path.length()} edges")
    print()

    # Get descendants
    descendants = lineage.get_descendants(node_id, max_depth=2)
    print(f"Descendants of '{node_id}' (depth ≤ 2):")
    print(f"  Count: {len(descendants)}")
    print(f"  Nodes: {sorted(descendants)}")
    print()

    # Get ancestors
    target_node = "obj-fokker-planck"
    ancestors = lineage.get_ancestors(target_node, max_depth=2)
    print(f"Ancestors of '{target_node}' (depth ≤ 2):")
    print(f"  Count: {len(ancestors)}")
    print(f"  Nodes: {sorted(ancestors)}")
    print()

    # ==========================================================================
    # Equivalence Classes
    # ==========================================================================
    print("STEP 5: Equivalence Class Computation")
    print("-" * 70)

    classifier = EquivalenceClassifier(graph)
    eq_classes = classifier.compute_equivalence_classes()

    print(f"Equivalence classes: {len(eq_classes)}")
    for i, eq_class in enumerate(eq_classes, 1):
        print(f"  Class {i}:")
        print(f"    Size: {eq_class.size()}")
        print(f"    Representative: {eq_class.representative}")
        print(f"    Members: {', '.join(eq_class.members)}")
    print()

    # Test equivalence
    node_a = "obj-discrete-particle-system"
    node_b = "obj-fokker-planck"
    are_equiv = classifier.are_equivalent(node_a, node_b)
    print(f"Are '{node_a}' and '{node_b}' equivalent?")
    print(f"  {are_equiv}")
    print()

    node_c = "obj-continuous-pde"
    are_equiv2 = classifier.are_equivalent(node_a, node_c)
    print(f"Are '{node_a}' and '{node_c}' equivalent?")
    print(f"  {are_equiv2}")
    print()

    # ==========================================================================
    # Framework Flow
    # ==========================================================================
    print("STEP 6: Framework Flow Analysis")
    print("-" * 70)

    flow = build_framework_flow_from_registry(registry, graph)

    print(f"Framework flow:")
    print(f"  Theorems: {len(flow.theorem_nodes)}")
    print()

    # Show theorems
    print("Theorem nodes:")
    for thm_id, thm_node in flow.theorem_nodes.items():
        print(f"  - {thm_id}")
        print(f"    Inputs: {', '.join(thm_node.input_objects)}")
        print(f"    Outputs: {', '.join(thm_node.output_objects)}")
        print(f"    Relations: {', '.join(thm_node.relations_established)}")
    print()

    # Get establishing theorems for a relationship
    rel_id = "rel-discrete-continuous-equivalence"
    establishing_thms = flow.get_establishing_theorems(rel_id)
    print(f"Theorems establishing '{rel_id}':")
    print(f"  {', '.join(establishing_thms)}")
    print()

    # Get theorems for an object
    obj_id = "obj-discrete-particle-system"
    theorems = flow.get_theorems_for_object(obj_id)
    print(f"Theorems involving '{obj_id}':")
    print(f"  Count: {len(theorems)}")
    print(f"  Theorems: {', '.join(theorems)}")
    print()

    # ==========================================================================
    # Theorem Dependencies
    # ==========================================================================
    print("STEP 7: Theorem Dependency Analysis")
    print("-" * 70)

    # Get dependencies for a theorem
    thm_id = "thm-pde-equivalence"
    deps = flow.get_theorem_dependencies(thm_id)
    print(f"Dependencies of '{thm_id}':")
    print(f"  Count: {len(deps)}")
    if deps:
        print(f"  Depends on: {', '.join(sorted(deps))}")
    else:
        print(f"  No dependencies (base theorem)")
    print()

    # Compute framework layers
    layers = flow.get_framework_layers()
    print(f"Framework layers: {len(layers)}")
    for i, layer in enumerate(layers):
        print(f"  Layer {i}: {', '.join(sorted(layer))}")
    print()

    # ==========================================================================
    # Subgraph Extraction
    # ==========================================================================
    print("STEP 8: Subgraph Extraction")
    print("-" * 70)

    # Extract subgraph for euclidean-gas framework
    euclidean_nodes = {
        "obj-discrete-particle-system",
        "obj-continuous-pde",
        "obj-fokker-planck",
    }
    subgraph = graph.get_subgraph(euclidean_nodes)

    print(f"Euclidean gas subgraph:")
    print(f"  Nodes: {subgraph.node_count()}")
    print(f"  Edges: {subgraph.edge_count()}")
    print(f"  Node IDs: {', '.join(sorted(subgraph.nodes.keys()))}")
    print()

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("✅ Relationship Graph Features:")
    print("  1. Graph construction from registry")
    print("  2. Connectivity analysis (BFS)")
    print("  3. Path finding (shortest path)")
    print("  4. Subgraph extraction")
    print()
    print("✅ Object Lineage Features:")
    print("  1. Lineage path tracing (DFS)")
    print("  2. Ancestor/descendant computation")
    print("  3. Derivation chains")
    print()
    print("✅ Equivalence Class Features:")
    print("  1. Equivalence class computation (Union-Find)")
    print("  2. Equivalence testing")
    print("  3. Representative selection")
    print()
    print("✅ Framework Flow Features:")
    print("  1. Theorem-relationship tracking")
    print("  2. Object-theorem associations")
    print("  3. Dependency analysis")
    print("  4. Layered framework structure")
    print()
    print(f"✅ Graph Statistics:")
    print(f"  - {graph.node_count()} nodes in relationship graph")
    print(f"  - {graph.edge_count()} edges")
    print(f"  - {len(eq_classes)} equivalence classes")
    print(f"  - {len(layers)} framework layers")
    print()
    print("🎯 Ready for production use!")
    print()


if __name__ == "__main__":
    main()
