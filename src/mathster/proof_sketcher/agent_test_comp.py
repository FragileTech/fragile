#!/usr/bin/env python3
"""Example script demonstrating ProofSketchAgent usage."""

from __future__ import annotations

from datetime import datetime
import json
import logging
import sys
from typing import Any

import flogging

from mathster.parsing.config import configure_dspy
from mathster.proof_sketcher.sketcher import ProofSketchAgent


# Configure logging BEFORE flogging.setup() to ensure our handler is used
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
    force=True,  # Force reconfiguration even if handlers exist
)

# Now setup flogging
flogging.setup(allow_trailing_dot=True)

# Set proof sketcher modules to DEBUG level
logging.getLogger("mathster.proof_sketcher.sketcher").setLevel(logging.DEBUG)
logging.getLogger("mathster.proof_sketcher.sketch_validator").setLevel(logging.DEBUG)
logging.getLogger("mathster.proof_sketcher.sketch_referee_analysis").setLevel(logging.DEBUG)


def run_demo() -> dict[str, Any]:
    """Instantiate ProofSketchAgent, generate a proof sketch, and display results."""

    # Configure DSPy with a fast model for testing
    configure_dspy(model="gemini/gemini-2.5-flash-lite-preview-09-2025")

    print("=" * 80)
    print("Testing ProofSketchAgent")
    print("=" * 80)

    # Create the agent
    agent = ProofSketchAgent()

    # Sample theorem data
    sample_data = {
        "title_hint": "KL Convergence of Euclidean Gas",
        "theorem_label": "thm-euclidean-gas-kl-convergence",
        "theorem_type": "Theorem",
        "theorem_statement": (
            "Under the Euclidean Gas axioms with confining potential assumption, "
            "the swarm law μ_t converges exponentially fast in KL divergence to "
            "the unique quasi-stationary distribution (QSD) π: "
            "KL(μ_t || π) ≤ C e^{-λt} KL(μ_0 || π) for constants C, λ > 0."
        ),
        "document_source": "docs/source/1_euclidean_gas/09_kl_convergence.md",
        "creation_date": datetime.now().strftime("%Y-%m-%d"),
        "proof_status": "Sketch",
        "framework_context": (
            "Available results: LSI theory (thm-lsi-target), "
            "Bakry-Émery criterion (lem-bakry-emery), "
            "Grönwall's lemma (lem-gronwall), "
            "confinement potential bounds (ax-confining-potential)."
        ),
        "operator_notes": "Prefer LSI-based approach. Focus on explicit constants.",
    }

    print(f"\nGenerating proof sketch for: {sample_data['title_hint']}")
    print(f"Label: {sample_data['theorem_label']}")
    print(f"Type: {sample_data['theorem_type']}\n")

    # Run the agent
    prediction = agent(**sample_data)

    # Extract the proof sketch
    sketch = prediction.sketch

    # Display key components
    print("\n" + "=" * 80)
    print("PROOF SKETCH GENERATED")
    print("=" * 80)

    print("\n--- Metadata ---")
    print(f"Title: {sketch.title}")
    print(f"Label: {sketch.label}")
    print(f"Type: {sketch.type}")
    print(f"Status: {sketch.status}")
    print(f"Date: {sketch.date}")
    print(f"Source: {sketch.source}")

    print("\n--- Statement ---")
    formal_preview = (
        sketch.statement.formal[:200] + "..."
        if len(sketch.statement.formal) > 200
        else sketch.statement.formal
    )
    print(f"Formal: {formal_preview}")
    print(f"Informal: {sketch.statement.informal}")

    print("\n--- Strategy Synthesis ---")
    print(f"Number of strategies evaluated: {len(sketch.strategySynthesis.strategies)}")
    print(f"Chosen method: {sketch.strategySynthesis.recommendedApproach.chosenMethod}")
    rationale_preview = (
        sketch.strategySynthesis.recommendedApproach.rationale[:150] + "..."
        if len(sketch.strategySynthesis.recommendedApproach.rationale) > 150
        else sketch.strategySynthesis.recommendedApproach.rationale
    )
    print(f"Rationale: {rationale_preview}")

    print("\n--- Dependencies ---")
    deps = sketch.dependencies
    print(f"Verified dependencies: {len(deps.verifiedDependencies)}")
    if deps.verifiedDependencies:
        print("  Sample verified:")
        for dep in deps.verifiedDependencies[:3]:
            purpose_preview = dep.purpose[:60] + "..." if len(dep.purpose) > 60 else dep.purpose
            print(f"    - {dep.label} ({dep.type}): {purpose_preview}")

    if deps.missingOrUncertainDependencies:
        missing = deps.missingOrUncertainDependencies
        print(f"Lemmas to prove: {len(missing.lemmasToProve)}")
        print(f"Uncertain assumptions: {len(missing.uncertainAssumptions)}")

    print("\n--- Detailed Proof ---")
    proof = sketch.detailedProof
    overview_preview = (
        proof.overview[:150] + "..." if len(proof.overview) > 150 else proof.overview
    )
    print(f"Overview: {overview_preview}")
    print(f"Top-level outline ({len(proof.topLevelOutline)} items):")
    for i, item in enumerate(proof.topLevelOutline, 1):
        print(f"  {i}. {item}")
    print(f"Detailed steps: {len(proof.steps)}")
    conclusion_preview = (
        proof.conclusion[:100] + "..." if len(proof.conclusion) > 100 else proof.conclusion
    )
    print(f"Conclusion: {conclusion_preview}")

    print("\n--- Technical Deep Dives ---")
    print(f"Number of challenges identified: {len(sketch.technicalDeepDives)}")
    for i, dive in enumerate(sketch.technicalDeepDives, 1):
        print(f"\n  Challenge {i}: {dive.challengeTitle}")
        diff_preview = (
            dive.difficultyDescription[:80] + "..."
            if len(dive.difficultyDescription) > 80
            else dive.difficultyDescription
        )
        sol_preview = (
            dive.proposedSolution[:80] + "..."
            if len(dive.proposedSolution) > 80
            else dive.proposedSolution
        )
        print(f"    Difficulty: {diff_preview}")
        print(f"    Solution: {sol_preview}")

    print("\n--- Validation Checklist ---")
    checklist = sketch.validationChecklist
    print(f"  Logical Completeness: {checklist.logicalCompleteness}")
    print(f"  Hypothesis Usage: {checklist.hypothesisUsage}")
    print(f"  Conclusion Derivation: {checklist.conclusionDerivation}")
    print(f"  Framework Consistency: {checklist.frameworkConsistency}")
    print(f"  No Circular Reasoning: {checklist.noCircularReasoning}")

    print("\n--- Alternative Approaches ---")
    print(f"Number documented: {len(sketch.alternativeApproaches)}")

    if sketch.futureWork:
        print("\n--- Future Work ---")
        fw = sketch.futureWork
        print(f"  Remaining gaps: {len(fw.remainingGaps)}")
        print(f"  Conjectures: {len(fw.conjectures)}")
        print(f"  Extensions: {len(fw.extensions)}")

    if sketch.expansionRoadmap:
        print("\n--- Expansion Roadmap ---")
        roadmap = sketch.expansionRoadmap
        print(f"  Total phases: {len(roadmap.phases)}")
        print(f"  Estimated time: {roadmap.totalEstimatedTime}")

    print("\n--- Special Notes ---")
    if sketch.specialNotes:
        print(f"{sketch.specialNotes}")
    else:
        print("None")

    # Display the two strategies that were generated
    print("\n" + "=" * 80)
    print("DUAL STRATEGY COMPARISON")
    print("=" * 80)

    print("\n--- Strategy 1 (Primary) ---")
    s1 = prediction.strategy_1
    print(f"Strategist: {s1.strategist}")
    print(f"Method: {s1.method}")
    print(f"Confidence: {s1.confidenceScore}")
    print(f"Key steps: {len(s1.keySteps)}")

    print("\n--- Strategy 2 (Secondary) ---")
    s2 = prediction.strategy_2
    print(f"Strategist: {s2.strategist}")
    print(f"Method: {s2.method}")
    print(f"Confidence: {s2.confidenceScore}")
    print(f"Key steps: {len(s2.keySteps)}")

    # Optionally dump full JSON
    print("\n" + "=" * 80)
    print("FULL PROOF SKETCH (JSON)")
    print("=" * 80)
    print(json.dumps(sketch.model_dump(), indent=2))

    return sketch.model_dump()


if __name__ == "__main__":
    run_demo()
