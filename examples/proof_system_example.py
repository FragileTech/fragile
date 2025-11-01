"""
Proof System Example: Compositional Proofs with Property-Level Dataflow.

This example demonstrates the proof system's core features:
1. Property-level granularity (object has 10 properties, proof needs 2)
2. Hierarchical/recursive architecture (ProofBox contains sub-mathster)
3. Three expansion modes (DirectDerivation, SubProof, LemmaApplication)
4. Dataflow validation (properties flow correctly through steps)
5. Integration with relationship system

We'll prove the Mean Field Limit theorem as a concrete example.
"""

from pathlib import Path
import sys


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fragile.proofs import (
    AssumptionReference,
    DirectDerivation,
    LemmaApplication,
    ProofBox,
    ProofEngine,
    ProofInput,
    ProofOutput,
    ProofStep,
    ProofStepStatus,
    ProofStepType,
    PropertyReference,
    SubProofReference,
)


def create_mean_field_proof() -> ProofBox:
    """Create a proof of the Mean Field Limit theorem.

    Theorem: The discrete Euclidean Gas converges to the continuous PDE
    in Wasserstein distance at rate O(N^{-1/d}).

    This proof demonstrates:
    - Property-level inputs (discrete system needs specific properties)
    - Multi-step proof structure with dataflow
    - Sub-proof expansion for complex steps
    - Lemma application for reusable components
    """

    # ==========================================================================
    # STEP 0: Define Properties of Input Objects
    # ==========================================================================

    # Properties of the discrete system
    discrete_bounded_potential = PropertyReference(
        object_id="obj-euclidean-gas-discrete",
        property_id="prop-bounded-potential",
        property_statement="U: ℝ^d → ℝ is bounded: |U(x)| ≤ C_U",
    )

    discrete_lipschitz_potential = PropertyReference(
        object_id="obj-euclidean-gas-discrete",
        property_id="prop-lipschitz-potential",
        property_statement="U is Lipschitz: |∇U(x)| ≤ L_U",
    )

    discrete_bounded_domain = PropertyReference(
        object_id="obj-euclidean-gas-discrete",
        property_id="prop-bounded-domain",
        property_statement="State space 𝒳 is compact with diam(𝒳) ≤ D",
    )

    discrete_finite_particles = PropertyReference(
        object_id="obj-euclidean-gas-discrete",
        property_id="prop-finite-particles",
        property_statement="Number of particles N < ∞",
    )

    # Properties we'll establish for the continuous system
    continuous_well_posed = PropertyReference(
        object_id="obj-euclidean-gas-continuous",
        property_id="prop-well-posed",
        property_statement="PDE ∂_t μ = L_kin μ + L_clone μ has unique solution",
    )

    continuous_equivalence = PropertyReference(
        object_id="obj-euclidean-gas-continuous",
        property_id="prop-mean-field-equivalence",
        property_statement="W_2(μ_N, μ_t) = O(N^{-1/d}) where μ_N is empirical measure",
    )

    # ==========================================================================
    # STEP 1: Define Proof Inputs (What We Need)
    # ==========================================================================

    proof_inputs = [
        ProofInput(
            object_id="obj-euclidean-gas-discrete",
            required_properties=[
                discrete_bounded_potential,
                discrete_lipschitz_potential,
                discrete_bounded_domain,
                discrete_finite_particles,
            ],
            required_assumptions=[
                AssumptionReference(
                    object_id="obj-euclidean-gas-discrete",
                    assumption_id="assump-independent-initial",
                    assumption_statement="Initial positions {x_i(0)} are i.i.d. from μ_0",
                    justification="Standard assumption for mean field limit",
                )
            ],
        )
    ]

    # ==========================================================================
    # STEP 2: Define Proof Outputs (What We'll Prove)
    # ==========================================================================

    proof_outputs = [
        ProofOutput(
            object_id="obj-euclidean-gas-continuous",
            properties_established=[continuous_well_posed, continuous_equivalence],
        )
    ]

    # ==========================================================================
    # STEP 3: Create Proof Steps (The Proof Strategy)
    # ==========================================================================

    # Step 1: Prove PDE well-posedness (complex → needs sub-proof)
    step_1 = ProofStep(
        step_id="step-1",
        description="Establish well-posedness of McKean-Vlasov PDE",
        inputs=[
            ProofInput(
                object_id="obj-euclidean-gas-discrete",
                required_properties=[discrete_lipschitz_potential, discrete_bounded_domain],
                required_assumptions=[],
            )
        ],
        outputs=[
            ProofOutput(
                object_id="obj-euclidean-gas-continuous",
                properties_established=[continuous_well_posed],
            )
        ],
        step_type=ProofStepType.SUB_PROOF,
        derivation=SubProofReference(
            proof_id="proof-pde-wellposedness",
            proof_label="PDE Well-Posedness",
        ),
        status=ProofStepStatus.SKETCHED,
    )

    # Step 2: Couple discrete and continuous dynamics (uses existing lemma)
    step_2_intermediate = PropertyReference(
        object_id="obj-coupling",
        property_id="prop-coupling-exists",
        property_statement="∃ coupling (S_N, μ_t) on common probability space",
    )

    step_2 = ProofStep(
        step_id="step-2",
        description="Construct coupling between discrete and continuous systems",
        inputs=[
            ProofInput(
                object_id="obj-euclidean-gas-discrete",
                required_properties=[discrete_finite_particles],
                required_assumptions=[],
            ),
            ProofInput(
                object_id="obj-euclidean-gas-continuous",
                required_properties=[continuous_well_posed],
                required_assumptions=[],
            ),
        ],
        outputs=[
            ProofOutput(
                object_id="obj-coupling",
                properties_established=[step_2_intermediate],
            )
        ],
        step_type=ProofStepType.LEMMA_APPLICATION,
        derivation=LemmaApplication(
            lemma_id="lem-sznitman-coupling",
            input_mapping={
                "discrete-system": "obj-euclidean-gas-discrete",
                "continuous-system": "obj-euclidean-gas-continuous",
            },
            justification="Apply with μ_0 i.i.d. initial condition",
        ),
        status=ProofStepStatus.SKETCHED,
    )

    # Step 3: Derive convergence rate (simple → direct LLM derivation)
    step_3 = ProofStep(
        step_id="step-3",
        description="Bound Wasserstein distance using coupling",
        inputs=[
            ProofInput(
                object_id="obj-euclidean-gas-discrete",
                required_properties=[discrete_bounded_domain],
                required_assumptions=[],
            ),
            ProofInput(
                object_id="obj-coupling",
                required_properties=[step_2_intermediate],
                required_assumptions=[],
            ),
        ],
        outputs=[
            ProofOutput(
                object_id="obj-euclidean-gas-continuous",
                properties_established=[continuous_equivalence],
            )
        ],
        step_type=ProofStepType.DIRECT_DERIVATION,
        derivation=DirectDerivation(
            mathematical_content="""
By the coupling from Step 2, we have:

$$
W_2^2(\\mu_N, \\mu_t) \\leq \\mathbb{E}\\left[\\frac{1}{N}\\sum_{i=1}^N |X_i^N(t) - X_i(t)|^2\\right]
$$

where $(X_i^N, X_i)$ are coupled. Using the SDE analysis:

$$
d(X_i^N - X_i) = \\text{(interaction error)} + \\text{(discretization error)}
$$

The interaction error scales as $O(N^{-1})$ (finite-size effects).
Since we're in dimension $d$, the Wasserstein-2 distance inherits scaling:

$$
W_2(\\mu_N, \\mu_t) = O(N^{-1/d})
$$

This establishes the claimed convergence rate.
            """,
            techniques=["coupling", "wasserstein-distance", "sde-analysis"],
        ),
        status=ProofStepStatus.EXPANDED,  # Already has full derivation
    )

    # ==========================================================================
    # STEP 4: Create the ProofBox
    # ==========================================================================

    return ProofBox(
        proof_id="proof-thm-mean-field-limit",
        label="Mean Field Limit",
        proves="thm-mean-field-limit",
        inputs=proof_inputs,
        outputs=proof_outputs,
        strategy="""
Three-step strategy:
1. Establish PDE well-posedness (ensures continuous limit exists)
2. Construct Sznitman coupling (relates discrete and continuous)
3. Bound Wasserstein distance via coupling (quantitative convergence)
        """,
        steps=[step_1, step_2, step_3],
        sub_proofs={},  # Will be populated when step-1 is expanded
    )


def create_pde_wellposedness_subproof() -> ProofBox:
    """Create the sub-proof for PDE well-posedness.

    This demonstrates how complex steps expand into nested ProofBoxes.
    """

    # Input properties (from parent proof step-1)
    lipschitz_potential = PropertyReference(
        object_id="obj-euclidean-gas-discrete",
        property_id="prop-lipschitz-potential",
        property_statement="U is Lipschitz: |∇U(x)| ≤ L_U",
    )

    bounded_domain = PropertyReference(
        object_id="obj-euclidean-gas-discrete",
        property_id="prop-bounded-domain",
        property_statement="State space 𝒳 is compact with diam(𝒳) ≤ D",
    )

    # Output property
    well_posed = PropertyReference(
        object_id="obj-euclidean-gas-continuous",
        property_id="prop-well-posed",
        property_statement="PDE ∂_t μ = L_kin μ + L_clone μ has unique solution",
    )

    # Sub-proof steps
    step_1_existence = ProofStep(
        step_id="step-1-1",
        description="Prove existence via fixed-point theorem",
        inputs=[
            ProofInput(
                object_id="obj-euclidean-gas-discrete",
                required_properties=[lipschitz_potential, bounded_domain],
                required_assumptions=[],
            )
        ],
        outputs=[
            ProofOutput(
                object_id="obj-euclidean-gas-continuous",
                properties_established=[
                    PropertyReference(
                        object_id="obj-euclidean-gas-continuous",
                        property_id="prop-solution-exists",
                        property_statement="∃ solution μ_t ∈ C([0,T], P(𝒳))",
                    )
                ],
            )
        ],
        step_type=ProofStepType.LEMMA_APPLICATION,
        derivation=LemmaApplication(
            lemma_id="lem-banach-fixpoint",
            input_mapping={
                "banach-space": "obj-probability-measures",
                "contraction": "obj-mcvlasov-operator",
            },
            justification="Apply to Picard iteration for McKean-Vlasov SDE",
        ),
        status=ProofStepStatus.SKETCHED,
    )

    step_2_uniqueness = ProofStep(
        step_id="step-1-2",
        description="Prove uniqueness via Grönwall inequality",
        inputs=[
            ProofInput(
                object_id="obj-euclidean-gas-discrete",
                required_properties=[lipschitz_potential],
                required_assumptions=[],
            ),
            ProofInput(
                object_id="obj-euclidean-gas-continuous",
                required_properties=[
                    PropertyReference(
                        object_id="obj-euclidean-gas-continuous",
                        property_id="prop-solution-exists",
                        property_statement="∃ solution μ_t ∈ C([0,T], P(𝒳))",
                    )
                ],
                required_assumptions=[],
            ),
        ],
        outputs=[
            ProofOutput(
                object_id="obj-euclidean-gas-continuous",
                properties_established=[well_posed],
            )
        ],
        step_type=ProofStepType.DIRECT_DERIVATION,
        derivation=DirectDerivation(
            mathematical_content="""
Assume μ_t and ν_t are two solutions. Then:

$$
\\frac{d}{dt} W_2^2(\\mu_t, \\nu_t) \\leq 2 L_U W_2^2(\\mu_t, \\nu_t)
$$

By Grönwall's inequality:

$$
W_2^2(\\mu_t, \\nu_t) \\leq W_2^2(\\mu_0, \\nu_0) e^{2 L_U t}
$$

Since μ_0 = ν_0, we have W_2(μ_t, ν_t) = 0, hence μ_t ≡ ν_t.
            """,
            techniques=["gronwall-inequality", "wasserstein-contractivity"],
        ),
        status=ProofStepStatus.EXPANDED,
    )

    return ProofBox(
        proof_id="proof-pde-wellposedness",
        label="PDE Well-Posedness",
        proves="lem-pde-wellposedness",
        inputs=[
            ProofInput(
                object_id="obj-euclidean-gas-discrete",
                required_properties=[lipschitz_potential, bounded_domain],
                required_assumptions=[],
            )
        ],
        outputs=[
            ProofOutput(
                object_id="obj-euclidean-gas-continuous",
                properties_established=[well_posed],
            )
        ],
        strategy="Existence via fixed-point, uniqueness via Grönwall",
        steps=[step_1_existence, step_2_uniqueness],
        sub_proofs={},
    )


def main() -> None:
    """Demonstrate the proof system."""
    print("=" * 80)
    print("PROOF SYSTEM EXAMPLE: Mean Field Limit Theorem")
    print("=" * 80)
    print()

    # ==========================================================================
    # STEP 1: Create the main proof
    # ==========================================================================
    print("STEP 1: Create Main Proof (Mean Field Limit)")
    print("-" * 80)

    proof = create_mean_field_proof()

    print(f"✓ Created proof: {proof.label}")
    print(f"  Proof ID: {proof.proof_id}")
    print(f"  Proves: {proof.proves}")
    print(f"  Steps: {len(proof.steps)}")
    print()

    print("Proof inputs (property-level granularity):")
    for inp in proof.inputs:
        print(f"  Object: {inp.object_id}")
        print(f"    Required properties ({len(inp.required_properties)}):")
        for prop in inp.required_properties:
            print(f"      - {prop.property_id}: {prop.property_statement}")
        if inp.required_assumptions:
            print(f"    Assumptions ({len(inp.required_assumptions)}):")
            for assump in inp.required_assumptions:
                print(f"      - {assump.assumption_id}: {assump.assumption_statement}")
    print()

    print("Proof outputs (what we establish):")
    for out in proof.outputs:
        print(f"  Object: {out.object_id}")
        print(f"    Properties established ({len(out.properties_established)}):")
        for prop in out.properties_established:
            print(f"      - {prop.property_id}: {prop.property_statement}")
    print()

    # ==========================================================================
    # STEP 2: Show proof structure
    # ==========================================================================
    print("STEP 2: Proof Structure (3 steps)")
    print("-" * 80)

    for i, step in enumerate(proof.steps, 1):
        print(f"Step {i} [{step.step_id}]: {step.description}")
        print(f"  Type: {step.step_type.value}")
        print(f"  Status: {step.status.value}")

        print("  Inputs:")
        for inp in step.inputs:
            prop_ids = [p.property_id for p in inp.required_properties]
            print(f"    - {inp.object_id}: {prop_ids}")

        print("  Outputs:")
        for out in step.outputs:
            prop_ids = [p.property_id for p in out.properties_established]
            print(f"    - {out.object_id}: {prop_ids}")

        if step.derivation:
            if step.step_type == ProofStepType.SUB_PROOF:
                sub_ref = step.derivation
                print(f"  Sub-proof: {sub_ref.proof_id} ({sub_ref.proof_label})")
            elif step.step_type == ProofStepType.LEMMA_APPLICATION:
                lemma = step.derivation
                print(f"  Lemma: {lemma.lemma_id}")
            elif step.step_type == ProofStepType.DIRECT_DERIVATION:
                print("  ✓ Full mathematical derivation provided")
        print()

    # ==========================================================================
    # STEP 3: Validate dataflow
    # ==========================================================================
    print("STEP 3: Validate Dataflow")
    print("-" * 80)

    errors = proof.validate_dataflow()

    if not errors:
        print("✓ Dataflow is valid!")
        print("  All step inputs are satisfied by previous outputs or proof inputs")
        print("  All properties flow correctly through the proof")
    else:
        print(f"✗ Found {len(errors)} dataflow errors:")
        for error in errors:
            print(f"  - {error}")
    print()

    # ==========================================================================
    # STEP 4: Use ProofEngine to manage expansion
    # ==========================================================================
    print("STEP 4: ProofEngine - Manage Proof Expansion")
    print("-" * 80)

    engine = ProofEngine()
    engine.register_proof(proof)

    print("✓ Registered proof with engine")
    print()

    # Get expansion requests (sketched steps that need work)
    expansion_requests = engine.get_expansion_requests(proof.proof_id)

    print(f"Expansion requests: {len(expansion_requests)}")
    for req in expansion_requests:
        print(f"  - Step {req.step_id}: {req.step_description}")
        print(f"    Context: {req.context or 'None'}")
        print()

    # ==========================================================================
    # STEP 5: Add sub-proof for complex step
    # ==========================================================================
    print("STEP 5: Expand Complex Step with Sub-Proof")
    print("-" * 80)

    sub_proof = create_pde_wellposedness_subproof()

    print(f"✓ Created sub-proof: {sub_proof.label}")
    print(f"  Sub-proof ID: {sub_proof.proof_id}")
    print(f"  Steps: {len(sub_proof.steps)}")
    print()

    # Add sub-proof to engine
    success = engine.add_sub_proof(proof.proof_id, sub_proof)

    if success:
        print("✓ Sub-proof added successfully")
        print(f"  Parent proof now has {len(proof.sub_proofs)} sub-proof(s)")

        # Update step-1 status to EXPANDED
        proof.steps[0].status = ProofStepStatus.EXPANDED
        print(f"  Step-1 status updated: {proof.steps[0].status.value}")
    print()

    # ==========================================================================
    # STEP 6: Validate complete proof
    # ==========================================================================
    print("STEP 6: Validate Complete Proof (with sub-mathster)")
    print("-" * 80)

    validation_errors = engine.validate_proof(proof.proof_id)

    if not validation_errors:
        print("✓ Complete proof is valid!")
        print("  Main proof dataflow: ✓")
        print("  All sub-mathster dataflow: ✓")
    else:
        print(f"✗ Found {len(validation_errors)} validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
    print()

    # ==========================================================================
    # STEP 7: Convert to graph for visualization
    # ==========================================================================
    print("STEP 7: Export to Graph (for visualization)")
    print("-" * 80)

    graph = proof.to_graph()

    print("✓ Converted to graph:")
    print(f"  Nodes: {len(graph['nodes'])}")
    print(f"  Edges: {len(graph['edges'])}")
    print()

    print("Node types:")
    node_types = {}
    for node in graph["nodes"]:
        node_type = node["type"]
        node_types[node_type] = node_types.get(node_type, 0) + 1

    for node_type, count in node_types.items():
        print(f"  - {node_type}: {count}")
    print()

    print("Sample nodes:")
    for node in graph["nodes"][:5]:
        print(f"  {node['id']} ({node['type']}): {node.get('label', 'N/A')}")
    print()

    print("Sample edges (dataflow):")
    for edge in graph["edges"][:5]:
        props = edge.get("properties", [])
        if props:
            print(f"  {edge['source']} → {edge['target']} (properties: {', '.join(props[:3])}...)")
        else:
            print(f"  {edge['source']} → {edge['target']}")
    print()

    # ==========================================================================
    # STEP 8: Show recursive structure
    # ==========================================================================
    print("STEP 8: Recursive Proof Structure")
    print("-" * 80)

    def print_proof_tree(proof_box: ProofBox, indent: int = 0) -> None:
        """Recursively print proof structure."""
        prefix = "  " * indent
        print(f"{prefix}📦 {proof_box.label} ({proof_box.proof_id})")
        print(f"{prefix}   Steps: {len(proof_box.steps)}")

        for i, step in enumerate(proof_box.steps, 1):
            status_icon = "✓" if step.status == ProofStepStatus.EXPANDED else "○"
            print(f"{prefix}   {status_icon} Step {i}: {step.step_type.value}")

            # If step references sub-proof, recurse
            if step.step_type == ProofStepType.SUB_PROOF and step.derivation:
                sub_id = step.derivation.proof_id
                if sub_id in proof_box.sub_proofs:
                    print_proof_tree(proof_box.sub_proofs[sub_id], indent + 2)

    print_proof_tree(proof)
    print()

    # ==========================================================================
    # STEP 9: Summary
    # ==========================================================================
    print("=" * 80)
    print("PROOF SYSTEM SUMMARY")
    print("=" * 80)
    print()

    print("✅ Key Features Demonstrated:")
    print()
    print("1. Property-Level Granularity")
    print("   - Object has 4 properties (bounded, Lipschitz, compact, finite)")
    print("   - Each step uses only the properties it needs")
    print("   - Example: Step-3 only needs 'bounded-domain', not all 4")
    print()

    print("2. Hierarchical/Recursive Architecture")
    print("   - Main proof (3 steps)")
    print("   - Step-1 expands into sub-proof (2 steps)")
    print("   - Can nest arbitrarily deep")
    print()

    print("3. Three Expansion Modes")
    print("   - DirectDerivation: LLM provides full mathematical content (Step-3)")
    print("   - SubProof: Complex step becomes ProofBox (Step-1)")
    print("   - LemmaApplication: Reuse existing results (Step-2)")
    print()

    print("4. Dataflow Validation")
    print("   - Tracks available properties after each step")
    print("   - Ensures each step's inputs are satisfied")
    print("   - Detects missing properties early")
    print()

    print("5. Graph Representation")
    print(f"   - {len(graph['nodes'])} nodes (objects, steps, properties)")
    print(f"   - {len(graph['edges'])} edges (dataflow)")
    print("   - Ready for visualization (D3.js, Graphviz, etc.)")
    print()

    print("📊 Proof Statistics:")
    print(f"  Main proof: {len(proof.steps)} steps")
    print(f"  Sub-mathster: {len(proof.sub_proofs)}")
    print(
        f"  Total steps (including sub-mathster): {len(proof.steps) + sum(len(sp.steps) for sp in proof.sub_proofs.values())}"
    )
    print(f"  Input objects: {len(proof.inputs)}")
    print(f"  Output objects: {len(proof.outputs)}")
    print(
        f"  Total properties tracked: {sum(len(inp.required_properties) for inp in proof.inputs)}"
    )
    print()

    print("🎯 System ready for LLM proving pipeline!")
    print()
    print("Next steps:")
    print("  - Integrate with TheoremBox from pipeline_types.py")
    print("  - Create LLM prompt templates for expansion requests")
    print("  - Build proof visualization dashboard")
    print("  - Add automated proof checking (Lean export)")
    print()


if __name__ == "__main__":
    main()
