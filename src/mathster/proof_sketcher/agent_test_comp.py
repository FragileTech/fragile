#!/usr/bin/env python3
"""Example script demonstrating ProofSketchAgent usage."""

from __future__ import annotations

from datetime import datetime
import json
import logging
import sys
from typing import Any

import dspy
import flogging

from mathster.parsing.config import configure_dspy
from mathster.proof_sketcher.sketch_pipeline import AgentSketchPipeline, ProofSketchWorkflowResult


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
logging.getLogger("mathster.proof_sketcher.sketch_pipeline").setLevel(logging.DEBUG)


def run_demo() -> dict[str, Any]:
    """Instantiate AgentSketchPipeline, generate a proof sketch, and display results."""

    # Configure DSPy with a fast model for testing
    configure_dspy(
        model="gemini/gemini-2.5-flash-lite-preview-09-2025",
        #model="xai/grok-4-fast-reasoning-latest",
        temperature=0.1, max_tokens=50000
    )

    print("=" * 80)
    print("Testing AgentSketchPipeline (Full Workflow)")
    print("=" * 80)

    def reward_fn(args, pred: dspy.Prediction):
        return pred.result.scores.get_score()

    # Create the pipeline agent
    agent = dspy.Refine(
        module=AgentSketchPipeline(), reward_fn=reward_fn, N=5, threshold=60, fail_count=5
    )

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

    # Run the pipeline (drafting + validation)
    prediction = agent(**sample_data)

    # Extract the workflow result
    result: ProofSketchWorkflowResult = prediction.result
    sketch = result.sketch

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

    # Display validation report
    print("\n" + "=" * 80)
    print("VALIDATION REPORT")
    print("=" * 80)

    validation_report = result.validation_report
    scores = result.scores

    print("\n--- Overall Scores ---")
    print(f"Gemini Score: {scores.gemini_overall_score}/5")
    print(f"Codex Score: {scores.codex_overall_score}/5")
    print(f"Final score: {scores.get_score():.2f}/100")

    print("\n--- Final Decision ---")
    print(f"Recommendation: {validation_report.synthesisAndActionPlan.finalDecision}")

    # print("\n--- Gemini Review ---")
    # gemini_review = result.review_1
    # print(f"Reviewer: {gemini_review.reviewer}")
    # print(f"Timestamp: {gemini_review.timestamp}")
    # print(f"Overall Score: {gemini_review.overallAssessment.score}/5")
    # print(f"Confidence: {gemini_review.overallAssessment.confidenceScore}/5")
    # print(f"Recommendation: {gemini_review.overallAssessment.recommendation}")
    # print(f"Completeness Score: {gemini_review.completenessAndCorrectness.score}/5 ({len(gemini_review.completenessAndCorrectness.identifiedErrors)} errors)")
    # print(f"Logical Flow Score: {gemini_review.logicalFlowValidation.score}/5 (sound={gemini_review.logicalFlowValidation.isSound}, {len(gemini_review.logicalFlowValidation.identifiedGaps)} gaps)")
    # print(f"Dependency Status: {gemini_review.dependencyValidation.status} ({len(gemini_review.dependencyValidation.issues)} issues)")
    # print(f"Technical Dive Score: {gemini_review.technicalDeepDiveValidation.score}/5 ({len(gemini_review.technicalDeepDiveValidation.critiques)} critiques)")
    #
    # print("\n--- Codex Review ---")
    # codex_review = result.review_2
    # print(f"Reviewer: {codex_review.reviewer}")
    # print(f"Timestamp: {codex_review.timestamp}")
    # print(f"Overall Score: {codex_review.overallAssessment.score}/5")
    # print(f"Confidence: {codex_review.overallAssessment.confidenceScore}/5")
    # print(f"Recommendation: {codex_review.overallAssessment.recommendation}")
    # print(f"Completeness Score: {codex_review.completenessAndCorrectness.score}/5 ({len(codex_review.completenessAndCorrectness.identifiedErrors)} errors)")
    # print(f"Logical Flow Score: {codex_review.logicalFlowValidation.score}/5 (sound={codex_review.logicalFlowValidation.isSound}, {len(codex_review.logicalFlowValidation.identifiedGaps)} gaps)")
    # print(f"Dependency Status: {codex_review.dependencyValidation.status} ({len(codex_review.dependencyValidation.issues)} issues)")
    # print(f"Technical Dive Score: {codex_review.technicalDeepDiveValidation.score}/5 ({len(codex_review.technicalDeepDiveValidation.critiques)} critiques)")

    # Optionally dump full JSON
    print("\n" + "=" * 80)
    print("FULL PROOF SKETCH (JSON)")
    print("=" * 80)
    print(json.dumps(sketch.model_dump(), indent=2))

    return sketch.model_dump()


if __name__ == "__main__":
    run_demo()
