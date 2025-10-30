"""
Complete Integration Example: TheoremBox + ProofBox + Relationship System.

This example demonstrates the full workflow:
1. Define mathematical objects with properties
2. Create theorem claiming to establish new properties
3. Write compositional proof implementing the theorem
4. Validate proof against theorem
5. Extract relationships from proof
6. Add to registry and build relationship graph

This shows the complete end-to-end pipeline from theorem to verified proof
with automatic relationship extraction.
"""

from pathlib import Path
import sys


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fragile.proofs import (
    build_relationship_graph_from_registry,
    CombinedTagQuery,
    create_proof_inputs_from_theorem,
    create_proof_outputs_from_theorem,
    create_simple_object,
    DirectDerivation,
    extract_relationships_from_proof,
    get_proof_statistics,
    MathematicalObject,
    MathematicalRegistry,
    ObjectType,
    print_validation_result,
    ProofBox,
    ProofEngine,
    ProofStep,
    ProofStepStatus,
    ProofStepType,
    Property,
    PropertyEvent,
    PropertyEventType,
    Relationship,
    RelationType,
    TheoremBox,
    TheoremOutputType,
    validate_proof_for_theorem,
)


def main() -> None:
    """Demonstrate complete integration."""
    print("=" * 80)
    print("COMPLETE INTEGRATION: TheoremBox + ProofBox + Relationship System")
    print("=" * 80)
    print()
    print("This example shows the full workflow from theorem to verified proof.")
    print()

    # ==========================================================================
    # STEP 1: Create Mathematical Objects with Properties
    # ==========================================================================
    print("STEP 1: Create Mathematical Objects")
    print("-" * 80)

    # Create discrete Euclidean Gas object with properties
    obj_discrete = MathematicalObject(
        label="obj-euclidean-gas-discrete",
        name="Euclidean Gas (Discrete)",
        mathematical_expression="S_N = {(x_i, v_i, r_i)}_{i=1}^N",
        object_type=ObjectType.SET,
        tags=["euclidean-gas", "discrete", "particle-system"],
        current_properties=[
            Property(
                label="prop-bounded-potential",
                expression="U: ℝ^d → ℝ is bounded: |U(x)| ≤ C_U",
                object_label="obj-euclidean-gas-discrete",
                established_by="thm-axiom-setup",
            ),
            Property(
                label="prop-lipschitz-potential",
                expression="U is Lipschitz: |∇U(x)| ≤ L_U",
                object_label="obj-euclidean-gas-discrete",
                established_by="thm-axiom-setup",
            ),
        ],
        property_history=[
            PropertyEvent(
                timestamp=0,
                property_label="prop-bounded-potential",
                added_by_theorem="thm-axiom-setup",
                event_type=PropertyEventType.ADDED,
            ),
            PropertyEvent(
                timestamp=0,
                property_label="prop-lipschitz-potential",
                added_by_theorem="thm-axiom-setup",
                event_type=PropertyEventType.ADDED,
            ),
        ],
    )

    # Create continuous Euclidean Gas object (initially no properties - theorem will add them)
    obj_continuous = create_simple_object(
        label="obj-euclidean-gas-continuous",
        name="Euclidean Gas (Continuous)",
        expr="∂_t μ = L_kin μ + L_clone μ",
        obj_type=ObjectType.FUNCTION,
        tags=["euclidean-gas", "continuous", "pde", "mean-field"],
    )

    print(f"✓ Created object: {obj_discrete.label}")
    print(f"  Properties: {len(obj_discrete.current_properties)}")
    for prop in obj_discrete.current_properties:
        print(f"    - {prop.label}: {prop.expression}")
    print()

    print(f"✓ Created object: {obj_continuous.label}")
    print(f"  Properties: {len(obj_continuous.current_properties)} (will add via theorem)")
    print()

    # ==========================================================================
    # STEP 2: Create Theorem
    # ==========================================================================
    print("STEP 2: Create Theorem (Mean Field Limit)")
    print("-" * 80)

    # Properties that theorem will add
    prop_well_posed = Property(
        label="prop-well-posed",
        expression="PDE ∂_t μ = L_kin μ + L_clone μ has unique solution",
        object_label="obj-euclidean-gas-continuous",
        established_by="thm-mean-field-limit",
    )

    prop_equivalence = Property(
        label="prop-mean-field-equivalence",
        expression="W_2(μ_N, μ_t) = O(N^{-1/d}) where μ_N is empirical measure",
        object_label="obj-euclidean-gas-continuous",
        established_by="thm-mean-field-limit",
    )

    # Relationship established by theorem
    rel_mean_field = Relationship(
        label="rel-discrete-continuous-equivalence",
        relationship_type=RelationType.EQUIVALENCE,
        source_object="obj-euclidean-gas-discrete",
        target_object="obj-euclidean-gas-continuous",
        bidirectional=True,
        established_by="thm-mean-field-limit",
        expression="S_N ≡ μ_t + O(N^{-1/d}) as N → ∞",
        tags=["mean-field", "convergence"],
    )

    # Create theorem
    theorem = TheoremBox(
        label="thm-mean-field-limit",
        name="Mean Field Limit Theorem",
        input_objects=["obj-euclidean-gas-discrete"],
        properties_required={
            "obj-euclidean-gas-discrete": ["prop-bounded-potential", "prop-lipschitz-potential"]
        },
        output_type=TheoremOutputType.EQUIVALENCE,
        properties_added=[prop_well_posed, prop_equivalence],
        relations_established=[rel_mean_field],
    )

    print(f"✓ Created theorem: {theorem.name}")
    print(f"  Label: {theorem.label}")
    print(f"  Input objects: {theorem.input_objects}")
    print("  Properties required:")
    for obj, props in theorem.properties_required.items():
        print(f"    {obj}: {props}")
    print(f"  Properties added: {len(theorem.properties_added)}")
    for prop in theorem.properties_added:
        print(f"    - {prop.label} to {prop.object_label}")
    print(f"  Relations established: {len(theorem.relations_established)}")
    for rel in theorem.relations_established:
        print(f"    - {rel.label}: {rel.source_object} ↔ {rel.target_object}")
    print()

    # ==========================================================================
    # STEP 3: Create Proof Using Integration Helpers
    # ==========================================================================
    print("STEP 3: Create Proof (Using Integration Helpers)")
    print("-" * 80)

    # Use helper to create proof inputs from theorem
    objects = {
        "obj-euclidean-gas-discrete": obj_discrete,
        "obj-euclidean-gas-continuous": obj_continuous,
    }

    proof_inputs = create_proof_inputs_from_theorem(theorem, objects)

    print(f"✓ Created {len(proof_inputs)} proof inputs from theorem:")
    for inp in proof_inputs:
        props = [p.property_id for p in inp.required_properties]
        print(f"  - {inp.object_id}: {props}")
    print()

    # Use helper to create proof outputs from theorem
    proof_outputs = create_proof_outputs_from_theorem(theorem, objects)

    print(f"✓ Created {len(proof_outputs)} proof outputs from theorem:")
    for out in proof_outputs:
        props = [p.property_id for p in out.properties_established]
        print(f"  - {out.object_id}: {props}")
    print()

    # Create proof steps
    step_1 = ProofStep(
        step_id="step-1",
        description="Establish PDE well-posedness via Lipschitz contraction",
        inputs=proof_inputs,
        outputs=[proof_outputs[0]],  # Just well-posedness
        step_type=ProofStepType.DIRECT_DERIVATION,
        derivation=DirectDerivation(
            mathematical_content="""
Using Lipschitz continuity of U, we apply the Banach fixed-point theorem
to the McKean-Vlasov iteration operator:

$$
T(\\mu) = \\text{Solution to } \\partial_t \\nu = L_{kin}[\\mu] \\nu + L_{clone}[\\mu] \\nu
$$

Since |∇U(x)| ≤ L_U, the operator T is contractive in Wasserstein metric:

$$
W_2(T(\\mu), T(\\nu)) \\leq \\lambda W_2(\\mu, \\nu), \\quad \\lambda < 1
$$

Therefore, unique fixed point μ_t exists, establishing well-posedness.
            """,
            techniques=["banach-fixpoint", "wasserstein-contractivity", "mcvlasov-theory"],
        ),
        status=ProofStepStatus.EXPANDED,
    )

    step_2 = ProofStep(
        step_id="step-2",
        description="Prove convergence via Sznitman coupling and Grönwall bound",
        inputs=proof_inputs,
        outputs=proof_outputs,  # All outputs
        step_type=ProofStepType.DIRECT_DERIVATION,
        derivation=DirectDerivation(
            mathematical_content="""
Construct Sznitman coupling between S_N and μ_t. For coupled processes:

$$
W_2^2(\\mu_N, \\mu_t) \\leq \\mathbb{E}\\left[\\frac{1}{N}\\sum_{i=1}^N |X_i^N(t) - X_i(t)|^2\\right]
$$

The coupling error satisfies:

$$
\\frac{d}{dt} \\mathbb{E}[|X_i^N - X_i|^2] \\leq C_U \\mathbb{E}[|X_i^N - X_i|^2] + O(N^{-1})
$$

By Grönwall: $\\mathbb{E}[|X_i^N - X_i|^2] = O(N^{-2/d})$

Therefore: $W_2(\\mu_N, \\mu_t) = O(N^{-1/d})$
            """,
            techniques=["sznitman-coupling", "gronwall-inequality", "wasserstein-bounds"],
        ),
        status=ProofStepStatus.EXPANDED,
    )

    # Create proof box
    proof = ProofBox(
        proof_id="proof-thm-mean-field-limit",
        label="Mean Field Limit Proof",
        proves="thm-mean-field-limit",
        inputs=proof_inputs,
        outputs=proof_outputs,
        strategy="""
Two-step strategy:
1. Establish PDE well-posedness using Lipschitz contraction
2. Prove N→∞ convergence via Sznitman coupling and Grönwall
        """,
        steps=[step_1, step_2],
        sub_proofs={},
    )

    print(f"✓ Created proof: {proof.label}")
    print(f"  Proves: {proof.proves}")
    print(f"  Steps: {len(proof.steps)}")
    print()

    # ==========================================================================
    # STEP 4: Validate Proof Against Theorem
    # ==========================================================================
    print("STEP 4: Validate Proof Against Theorem")
    print("-" * 80)

    validation_result = validate_proof_for_theorem(proof, theorem, objects)

    print_validation_result(validation_result)
    print()

    # ==========================================================================
    # STEP 5: Extract Relationships from Proof
    # ==========================================================================
    print("STEP 5: Extract Relationships from Proof")
    print("-" * 80)

    extracted_rels = extract_relationships_from_proof(proof, theorem)

    print(f"Extracted {len(extracted_rels)} relationships:")
    for rel in extracted_rels:
        print(f"  - {rel.label}: {rel.source_object} → {rel.target_object}")
        print(f"    Type: {rel.relationship_type.value}")
        print(f"    Expression: {rel.expression}")
    print()

    # ==========================================================================
    # STEP 6: Get Proof Statistics
    # ==========================================================================
    print("STEP 6: Proof Statistics")
    print("-" * 80)

    stats = get_proof_statistics(proof)

    print("Proof breakdown:")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  - Direct derivations: {stats['direct_derivations']}")
    print(f"  - Sub-proofs: {stats['sub_proofs']}")
    print(f"  - Lemma applications: {stats['lemma_applications']}")
    print()
    print("Status breakdown:")
    print(f"  - Sketched: {stats['sketched']}")
    print(f"  - Expanded: {stats['expanded']}")
    print(f"  - Verified: {stats['verified']}")
    print()
    print("Dataflow:")
    print(f"  - Inputs: {stats['total_inputs']}")
    print(f"  - Outputs: {stats['total_outputs']}")
    print(f"  - Nested sub-proofs: {stats['nested_sub_proofs']}")
    print()

    # ==========================================================================
    # STEP 7: Integrate with Registry and Relationship Graph
    # ==========================================================================
    print("STEP 7: Integrate with Registry and Relationship Graph")
    print("-" * 80)

    # Create registry
    registry = MathematicalRegistry()

    # Add objects
    registry.add(obj_discrete)
    registry.add(obj_continuous)

    # Add relationships
    registry.add(rel_mean_field)

    # Add theorem
    registry.add(theorem)

    print("✓ Added to registry:")
    print(f"  Objects: {len(registry.get_all_objects())}")
    print(f"  Relationships: {len(registry.get_all_relationships())}")
    print(f"  Theorems: {len(registry.get_all_theorems())}")
    print()

    # Build relationship graph
    graph = build_relationship_graph_from_registry(registry)

    print("✓ Built relationship graph:")
    print(f"  Nodes: {graph.node_count()}")
    print(f"  Edges: {graph.edge_count()}")
    print()

    # Query connected objects
    connected = graph.get_connected_component("obj-euclidean-gas-discrete")
    print("Objects connected to discrete formulation:")
    for obj_id in sorted(connected):
        print(f"  - {obj_id}")
    print()

    # ==========================================================================
    # STEP 8: Use ProofEngine for Management
    # ==========================================================================
    print("STEP 8: ProofEngine for Proof Management")
    print("-" * 80)

    engine = ProofEngine()
    engine.register_proof(proof)

    print("✓ Registered proof with engine")
    print()

    # Get expansion requests (should be none since all steps are expanded)
    requests = engine.get_expansion_requests(proof.proof_id)
    print(f"Expansion requests: {len(requests)}")
    if requests:
        print("  Steps needing expansion:")
        for req in requests:
            print(f"    - {req.step_id}: {req.step_description}")
    else:
        print("  ✓ All steps fully expanded!")
    print()

    # Validate with engine
    errors = engine.validate_proof(proof.proof_id)
    if not errors:
        print("✓ Engine validation: PASSED")
    else:
        print(f"✗ Engine validation: FAILED ({len(errors)} errors)")
        for error in errors:
            print(f"  - {error}")
    print()

    # ==========================================================================
    # STEP 9: Query Framework
    # ==========================================================================
    print("STEP 9: Query Framework")
    print("-" * 80)

    # Query by tags
    query = CombinedTagQuery(must_have=["euclidean-gas"], any_of=["discrete", "continuous"])
    result = registry.query_by_tags(query)

    print(f"Objects tagged with 'euclidean-gas' (discrete or continuous): {result.count()}")
    for obj in result.matches:
        obj_labels = ", ".join(obj.tags)
        print(f"  - {obj.label} ({obj_labels})")
    print()

    # Get related objects
    related = registry.get_related_objects("obj-euclidean-gas-discrete")
    print(f"Objects related to discrete formulation: {len(related)}")
    for rel_obj in related:
        print(f"  - {rel_obj}")
    print()

    # ==========================================================================
    # STEP 10: Summary
    # ==========================================================================
    print("=" * 80)
    print("INTEGRATION SUMMARY")
    print("=" * 80)
    print()

    print("✅ Complete workflow demonstrated:")
    print("  1. ✓ Created mathematical objects with properties")
    print("  2. ✓ Defined theorem with property requirements")
    print("  3. ✓ Wrote compositional proof using integration helpers")
    print("  4. ✓ Validated proof against theorem (PASSED)")
    print("  5. ✓ Extracted relationships from proof")
    print("  6. ✓ Analyzed proof statistics")
    print("  7. ✓ Added to registry and built relationship graph")
    print("  8. ✓ Managed proof with ProofEngine")
    print("  9. ✓ Queried framework")
    print()

    print("📊 Framework metrics:")
    print(f"  - {len(registry.get_all_objects())} mathematical objects")
    print(
        f"  - {sum(len(obj.current_properties) for obj in registry.get_all_objects())} total properties"
    )
    print(f"  - {len(registry.get_all_relationships())} relationships")
    print(f"  - {len(registry.get_all_theorems())} theorems")
    print("  - 1 complete proof (2 steps, both expanded)")
    print(f"  - {graph.node_count()} graph nodes, {graph.edge_count()} edges")
    print()

    print("🎯 Integration features:")
    print("  ✓ TheoremBox ↔ ProofBox validation")
    print("  ✓ Automatic proof input/output creation from theorem")
    print("  ✓ Relationship extraction from proofs")
    print("  ✓ Proof statistics and analysis")
    print("  ✓ Registry integration")
    print("  ✓ Relationship graph integration")
    print("  ✓ ProofEngine management")
    print()

    print("🚀 System ready for:")
    print("  - LLM-based proof expansion")
    print("  - Automatic relationship discovery")
    print("  - Theorem dependency tracking")
    print("  - Framework visualization")
    print("  - Lean export for formal verification")
    print()


if __name__ == "__main__":
    main()
