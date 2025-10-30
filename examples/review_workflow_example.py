"""
Review Workflow Example: Demonstrating the Review System.

This example shows how to:
1. Create a proof
2. Validate it automatically (dataflow)
3. Submit for LLM review (simulated)
4. Analyze review results
5. Iterate on issues
6. Track improvement over iterations

Run with: python examples/review_workflow_example.py
"""

from pathlib import Path
import sys


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fragile.proofs import (
    get_review_registry,
    # Core types
    ProofBox,
    ProofInput,
    ProofOutput,
    ProofStep,
    ProofStepStatus,
    ProofStepType,
    Property,
    PropertyReference,
    # Review system
    ReviewAnalyzer,
    ReviewBuilder,
    ReviewSource,
    TheoremBox,
    TheoremOutputType,
)


def create_example_theorem() -> TheoremBox:
    """Create a simple theorem for demonstration."""
    return TheoremBox(
        label="thm-convergence-example",
        name="Convergence Example",
        input_objects=["obj-discrete-system"],
        properties_required={"obj-discrete-system": ["prop-lipschitz", "prop-bounded"]},
        output_type=TheoremOutputType.PROPERTY,
        properties_added=[
            Property(
                label="prop-converges",
                expression=r"\\lim_{t \\to \\infty} \\|\\mu_t - \\mu_*\\| = 0",
                object_label="obj-discrete-system",
                established_by="thm-convergence-example",
            )
        ],
    )


def create_example_proof_with_error() -> ProofBox:
    """
    Create a proof with an intentional error (circular dependency).

    This will trigger validation failure.
    """
    # Inputs
    lipschitz_prop = PropertyReference(
        object_id="obj-discrete-system",
        property_id="prop-lipschitz",
        property_statement="Function is Lipschitz continuous",
    )

    # Outputs
    converges_prop = PropertyReference(
        object_id="obj-discrete-system",
        property_id="prop-converges",
        property_statement="System converges to equilibrium",
    )

    # Step 1: Circular dependency - uses prop that step 2 produces!
    step_1 = ProofStep(
        step_id="step-1",
        description="Apply Grönwall inequality",
        inputs=[
            ProofInput(
                object_id="obj-discrete-system",
                required_properties=[
                    lipschitz_prop,
                    # ERROR: This property doesn't exist yet!
                    PropertyReference(
                        object_id="obj-discrete-system",
                        property_id="prop-bounded-derivative",
                        property_statement="Derivative is bounded",
                    ),
                ],
            )
        ],
        outputs=[
            ProofOutput(
                object_id="obj-discrete-system",
                properties_established=[converges_prop],
            )
        ],
        step_type=ProofStepType.DIRECT_DERIVATION,
        status=ProofStepStatus.SKETCHED,
    )

    # Step 2: Produces property that step 1 needs (circular!)
    step_2 = ProofStep(
        step_id="step-2",
        description="Bound the derivative",
        inputs=[
            ProofInput(
                object_id="obj-discrete-system",
                required_properties=[converges_prop],  # Needs step-1 output
            )
        ],
        outputs=[
            ProofOutput(
                object_id="obj-discrete-system",
                properties_established=[
                    PropertyReference(
                        object_id="obj-discrete-system",
                        property_id="prop-bounded-derivative",
                        property_statement="Derivative is bounded",
                    )
                ],
            )
        ],
        step_type=ProofStepType.DIRECT_DERIVATION,
        status=ProofStepStatus.SKETCHED,
    )

    return ProofBox(
        proof_id="proof-thm-convergence-example",
        label="Convergence Proof (with error)",
        proves="thm-convergence-example",
        inputs=[
            ProofInput(
                object_id="obj-discrete-system",
                required_properties=[lipschitz_prop],
            )
        ],
        outputs=[
            ProofOutput(
                object_id="obj-discrete-system",
                properties_established=[converges_prop],
            )
        ],
        strategy="Apply Grönwall inequality after bounding derivative",
        steps=[step_1, step_2],
    )


def simulate_llm_review_response() -> str:
    """Simulate an LLM review response (what Gemini/Codex would return)."""
    return """## Overall Assessment

This proof has a critical circular dependency issue. Step 1 requires a property
(prop-bounded-derivative) that is only established in Step 2, but Step 2 requires
the output of Step 1. This makes the proof invalid.

Additionally, Step 1 is marked as SKETCHED but provides no mathematical content,
violating the requirement for complete proofs.

## Issues

1. [CRITICAL] Circular Dependency in Proof Steps
   Location: step-1, step-2
   Problem: Step 1 requires prop-bounded-derivative (from Step 2), but Step 2 requires prop-converges (from Step 1)
   Mechanism: The dataflow graph has a cycle: step-1 → step-2 → step-1
   Evidence: Step 1 inputs require prop-bounded-derivative, which is in step-2 outputs. Step 2 inputs require prop-converges, which is in step-1 outputs.
   Impact: thm-convergence-example, obj-discrete-system
   Fix: Reorder steps: first establish prop-bounded-derivative using only Lipschitz property, then apply Grönwall

2. [MAJOR] Missing Mathematical Content in Step 1
   Location: step-1
   Problem: Step is marked SKETCHED but should be EXPANDED with full derivation
   Mechanism: ProofStepStatus.SKETCHED indicates missing content
   Evidence: step-1 has no DirectDerivation with mathematical_content field
   Impact: proof-thm-convergence-example
   Fix: Add DirectDerivation with complete Grönwall inequality application

3. [MINOR] Vague Proof Strategy
   Location: strategy field
   Problem: Strategy says "Apply Grönwall inequality after bounding derivative" but doesn't explain why this approach works
   Mechanism: Missing mathematical intuition for strategy choice
   Fix: Add explanation: "The Lipschitz property gives us the derivative bound needed for Grönwall, which then yields exponential convergence"

## Strengths
- Clear proof structure with well-defined inputs and outputs
- Correct use of property-level granularity
- Appropriate choice of Grönwall inequality for convergence

## Scores
Rigor: 3/10
Soundness: 2/10
Completeness: 4/10
Clarity: 6/10
Framework Consistency: 7/10
"""


def main():
    """Demonstrate the Review workflow."""
    print("=" * 80)
    print("REVIEW SYSTEM DEMONSTRATION")
    print("=" * 80)
    print()

    # ==========================================================================
    # STEP 1: Create proof with error
    # ==========================================================================
    print("STEP 1: Create proof with intentional error (circular dependency)")
    print("-" * 80)

    proof = create_example_proof_with_error()
    print(f"✓ Created proof: {proof.label}")
    print(f"  Proof ID: {proof.proof_id}")
    print(f"  Steps: {len(proof.steps)}")
    print()

    # ==========================================================================
    # STEP 2: Automated validation (dataflow)
    # ==========================================================================
    print("STEP 2: Run automated dataflow validation")
    print("-" * 80)

    dataflow_errors = proof.validate_dataflow()

    if dataflow_errors:
        print(f"✗ Dataflow validation FAILED with {len(dataflow_errors)} errors:")
        for error in dataflow_errors:
            print(f"  - {error}")
        print()

        # Create review from validation failure
        validation_review = ReviewBuilder.from_validation_failure(
            validator="dataflow",
            errors=dataflow_errors,
            warnings=[],
            object_id=proof.proof_id,
            object_type="ProofBox",
            iteration=0,
        )

        # Add to registry
        registry = get_review_registry()
        registry.add_review(validation_review)

        print(f"✓ Created validation review: {validation_review.review_id}")
        print(f"  Status: {validation_review.get_status().value}")
        print(f"  Blocking issues: {validation_review.blocking_issue_count}")
        print()
    else:
        print("✓ Dataflow validation passed")
        print()

    # ==========================================================================
    # STEP 3: Simulated LLM review
    # ==========================================================================
    print("STEP 3: Simulated LLM review (Gemini 2.5 Pro)")
    print("-" * 80)

    llm_response = simulate_llm_review_response()
    print("LLM Response (truncated):")
    print(llm_response[:300] + "...")
    print()

    # Parse LLM response into Review object
    llm_review = ReviewBuilder.from_llm_response(
        response_text=llm_response,
        object_id=proof.proof_id,
        object_type="ProofBox",
        iteration=0,
        source=ReviewSource.GEMINI_2_5_PRO,
    )

    # Add to registry
    registry.add_review(llm_review)

    print(f"✓ Created LLM review: {llm_review.review_id}")
    print(f"  Issues identified: {len(llm_review.issues)}")
    print(f"  Average score: {llm_review.get_average_score():.1f}/10")
    print(f"  Status: {llm_review.get_status().value}")
    print()

    # ==========================================================================
    # STEP 4: Analyze review results
    # ==========================================================================
    print("STEP 4: Analyze review results")
    print("-" * 80)

    # Get blocking issues
    blocking = ReviewAnalyzer.identify_blocking_issues(llm_review)
    print(f"Blocking issues ({len(blocking)}):")
    for issue in blocking:
        print(f"  [{issue.severity.value.upper()}] {issue.title}")
        print(f"    Location: {issue.location}")
        print(f"    Actionability: {issue.actionability_score:.1f}")
    print()

    # Suggest next action
    next_action = ReviewAnalyzer.suggest_next_action(llm_review)
    print(f"Suggested next action: {next_action}")
    print()

    # Compute metrics
    metrics = ReviewAnalyzer.compute_review_metrics(llm_review)
    print("Review metrics:")
    print(f"  Average score: {metrics['avg_score']:.1f}/10")
    print(f"  Blocking ratio: {metrics['blocking_ratio']:.1%}")
    print(f"  Mean actionability: {metrics['actionability_mean']:.2f}")
    print(f"  Severity distribution: {metrics['severity_distribution']}")
    print()

    # ==========================================================================
    # STEP 5: Query review registry
    # ==========================================================================
    print("STEP 5: Query review registry")
    print("-" * 80)

    # Get all reviews for proof
    history = registry.get_review_history(proof.proof_id)
    print(f"Review history for {proof.proof_id}: {len(history)} reviews")
    for review in history:
        print(
            f"  - Iteration {review.iteration} ({review.source.value}): {review.get_status().value}"
        )
    print()

    # Get current status
    status = registry.get_status(proof.proof_id)
    print(f"Current status: {status.value}")
    print()

    # Get unresolved issues
    unresolved = registry.get_unresolved_issues(proof.proof_id)
    print(f"Unresolved issues: {len(unresolved)}")
    print()

    # Get all blocked objects
    blocked_objects = registry.get_blocked_objects()
    print(f"All blocked objects in registry: {blocked_objects}")
    print()

    # ==========================================================================
    # STEP 6: Registry statistics
    # ==========================================================================
    print("STEP 6: Registry statistics")
    print("-" * 80)

    stats = registry.get_statistics()
    print("Registry statistics:")
    print(f"  Total objects reviewed: {stats['total_objects']}")
    print(f"  Total reviews: {stats['total_reviews']}")
    print(f"  Ready for publication: {stats['ready_count']}")
    print(f"  Blocked: {stats['blocked_count']}")
    print(f"  Average iterations: {stats['avg_iterations']:.1f}")
    print()

    print("  By status:")
    for status, count in stats["by_status"].items():
        if count > 0:
            print(f"    {status}: {count}")
    print()

    print("  By source:")
    for source, count in stats["by_source"].items():
        print(f"    {source}: {count}")
    print()

    # ==========================================================================
    # STEP 7: Export review history
    # ==========================================================================
    print("STEP 7: Export review history to JSON")
    print("-" * 80)

    export_path = Path("/tmp/review_registry.json")
    registry.export_to_json(export_path)
    print(f"✓ Exported to: {export_path}")
    print(f"  File size: {export_path.stat().st_size} bytes")
    print()

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("=" * 80)
    print("REVIEW SYSTEM SUMMARY")
    print("=" * 80)
    print()

    print("✅ Demonstrated Features:")
    print()
    print("1. Automated Validation")
    print("   - Dataflow validation detected circular dependency")
    print("   - Created ValidationResult and Review from failure")
    print()

    print("2. LLM Review Integration")
    print("   - Simulated Gemini response parsing")
    print("   - ReviewBuilder creates structured Review from text")
    print("   - Issues extracted with severity, location, actionability")
    print()

    print("3. Review Analysis")
    print("   - Identified blocking issues automatically")
    print("   - Computed actionability scores")
    print("   - Suggested next action based on review status")
    print()

    print("4. Review Registry")
    print("   - Centralized storage of all reviews")
    print("   - Query by status, severity, iteration")
    print("   - Track improvement trajectory")
    print("   - Export/import for persistence")
    print()

    print("📊 Next Steps in Real Workflow:")
    print()
    print("1. Fix blocking issues (circular dependency)")
    print("2. Re-submit for review (iteration 1)")
    print("3. Compare reviews across iterations")
    print("4. Dual review protocol (Gemini + Codex)")
    print("5. Synthesize consensus review")
    print("6. Iterate until ready for publication")
    print()

    print("🔗 Integration Points:")
    print()
    print("- MCP Tools: mcp__gemini-cli__ask-gemini, mcp__codex__codex")
    print("- Validators: dataflow, SymPy, framework_checker")
    print("- ProofBox: validate_dataflow(), to_graph()")
    print("- TheoremBox: compute_conditionality(), can_execute()")
    print()


if __name__ == "__main__":
    main()
