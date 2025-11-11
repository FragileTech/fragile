#!/usr/bin/env python3
"""Example script that demonstrates SketchRefereeAgent usage."""

from __future__ import annotations

import json
from typing import Any

from mathster.parsing.config import configure_dspy
from mathster.proof_sketcher.sketch_validator import SketchValidator


def run_demo() -> dict[str, Any]:
    """Instantiate the agent, run a mock review, and return the serialized output."""

    # Create a mock ProofSketch dictionary matching the ProofSketch schema
    sample_proof_sketch = {
        "title": "KL Convergence of Euclidean Gas",
        "label": "thm-euclidean-gas-kl-convergence",
        "type": "Theorem",
        "source": "docs/source/1_euclidean_gas/09_kl_convergence.md",
        "date": "2025-01-15",
        "status": "Sketch",
        "statement": {
            "formal": (
                "Under the Euclidean Gas axioms and the confining potential assumption, "
                "the swarm law converges exponentially fast in KL divergence to the unique QSD."
            ),
            "informal": "The Euclidean Gas converges to equilibrium exponentially fast.",
        },
        "strategySynthesis": {
            "evaluatedStrategies": [
                {
                    "name": "LSI + Grönwall",
                    "description": "Use log-Sobolev inequality with Grönwall lemma",
                    "pros": ["Well-established", "Quantitative rates"],
                    "cons": ["Requires LSI constant"],
                }
            ],
            "chosenStrategy": {
                "name": "LSI + Grönwall",
                "rationale": "Standard approach for exponential convergence",
            },
        },
        "dependencies": {
            "verifiedDependencies": [
                {"label": "def-kl-divergence", "title": "KL Divergence", "type": "Definition"}
            ],
            "missingDependencies": ["LSI constant computation"],
        },
        "detailedProof": {
            "overview": (
                "We establish exponential KL convergence by first showing tightness "
                "via the confinement potential, then applying the log-Sobolev inequality "
                "combined with Grönwall's lemma to obtain exponential decay."
            ),
            "topLevelOutline": [
                "Establish tightness via confinement potential",
                "Apply log-Sobolev inequality for KL contraction",
                "Use Grönwall lemma to obtain exponential rate",
                "Verify uniqueness of QSD",
            ],
            "steps": [
                {
                    "stepNumber": 1,
                    "title": "Tightness from Confinement",
                    "goal": "Show the swarm remains in a compact region",
                    "action": "Apply confinement potential bounds",
                    "justification": "Confining potential ensures bounded support",
                    "expectedResult": "Tightness of the measure sequence",
                    "dependencies": ["def-confining-potential"],
                },
                {
                    "stepNumber": 2,
                    "title": "KL Contraction via LSI",
                    "goal": "Establish KL divergence decay rate",
                    "action": "Apply Bakry-Émery criterion",
                    "justification": "Log-Sobolev inequality implies entropy decay",
                    "expectedResult": "d/dt KL(ρ_t || π) ≤ -λ KL(ρ_t || π)",
                    "dependencies": ["lem-bakry-emery"],
                    "potentialIssues": "LSI constant not explicitly computed",
                },
                {
                    "stepNumber": 3,
                    "title": "Exponential Rate via Grönwall",
                    "goal": "Convert differential inequality to exponential bound",
                    "action": "Apply Grönwall lemma to KL differential inequality",
                    "justification": "Standard ODE comparison argument",
                    "expectedResult": "KL(ρ_t || π) ≤ e^{-λt} KL(ρ_0 || π)",
                    "dependencies": ["lem-gronwall"],
                },
                {
                    "stepNumber": 4,
                    "title": "Uniqueness of QSD",
                    "goal": "Show π is the unique quasi-stationary distribution",
                    "action": "Use ergodicity from exponential convergence",
                    "justification": "Exponential convergence implies unique limit",
                    "expectedResult": "π is the unique QSD",
                    "dependencies": ["thm-ergodic-convergence"],
                },
            ],
            "conclusion": "Therefore, the Euclidean Gas converges exponentially in KL divergence. Q.E.D.",
        },
        "validationChecklist": {
            "allClaimsCovered": False,
            "allDependenciesVerified": False,
            "logicalFlowSound": True,
            "technicalGapsIdentified": ["LSI constant not computed"],
            "readyForExpansion": False,
        },
        "technicalDeepDives": [
            {
                "challengeTitle": "Log-Sobolev Constant",
                "difficultyDescription": "Computing the explicit LSI constant is non-trivial",
                "proposedSolution": "Use Bakry-Émery Γ2 criterion with curvature bounds",
                "mathematicalDetail": "Need to verify Γ2(V, V) ≥ λ ||∇V||^2 for some λ > 0",
            }
        ],
        "alternativeApproaches": [],
        "futureWork": None,
        "expansionRoadmap": None,
        "crossReferences": None,
        "specialNotes": "This is a test sketch for validation workflow",
    }

    from datetime import datetime
    import uuid

    configure_dspy(model="gemini/gemini-2.5-flash-lite-preview-09-2025")
    print("Testing SketchValidator")
    validator = SketchValidator()

    # Generate required metadata
    validation_cycle_id = str(uuid.uuid4())
    validation_timestamp = datetime.utcnow().isoformat() + "Z"
    sketch_label = sample_proof_sketch["label"]

    prediction = validator(
        proof_sketch=sample_proof_sketch,
        sketch_label=sketch_label,
        validation_cycle_id=validation_cycle_id,
        validation_timestamp=validation_timestamp,
        reviewer_context="Focus on mathematical rigor and completeness of arguments.",
        final_decision_context="Assess if the proof sketch is ready for expansion or needs revisions.",
        confidence_context="Evaluate the overall viability of the proof strategy.",
    )

    # Extract the full validation report
    report = prediction.report

    print("\n=== Sketch Validation Report ===")
    print(f"Sketch Label: {report.reportMetadata.sketchLabel}")
    print(f"Validation Cycle ID: {report.reportMetadata.validationCycleId}")
    print(f"Timestamp: {report.reportMetadata.validationTimestamp}")

    print("\n--- Synthesis and Action Plan ---")
    synthesis = report.synthesisAndActionPlan
    print(f"Final Decision: {synthesis.finalDecision}")
    print(f"\nConfidence Statement:\n{synthesis.confidenceStatement}")

    print("\n--- Consensus Analysis ---")
    consensus = synthesis.consensusAnalysis
    print(f"Summary: {consensus.summaryOfFindings}")
    print(f"\nPoints of Agreement ({len(consensus.pointsOfAgreement)}):")
    for point in consensus.pointsOfAgreement:
        print(f"  - {point}")
    print(f"\nPoints of Disagreement ({len(consensus.pointsOfDisagreement)}):")
    for disagreement in consensus.pointsOfDisagreement:
        print(f"  Topic: {disagreement.topic}")
        print(f"    Gemini: {disagreement.geminiView}")
        print(f"    Codex: {disagreement.codexView}")
        print(f"    Resolution: {disagreement.resolution}")

    print(f"\n--- Actionable Items ({len(synthesis.actionableItems)}) ---")
    for item in synthesis.actionableItems:
        print(f"  [{item.priority}] {item.itemId}: {item.description}")
        if item.references:
            print(f"    References: {', '.join(item.references)}")

    print("\n--- Individual Reviews ---")
    print(f"Number of reviews: {len(report.reviews)}")
    for i, review in enumerate(report.reviews, 1):
        print(f"\nReview {i}:")
        print(f"  Reviewer: {review.get('reviewer', 'Unknown')}")
        if "overallAssessment" in review:
            print(f"  Overall Score: {review['overallAssessment'].get('score', 'N/A')}")
            print(f"  Recommendation: {review['overallAssessment'].get('recommendation', 'N/A')}")

    print("\n--- Full Report JSON ---")
    print(json.dumps(report.model_dump(), indent=2))
    print("REviews")
    print(json.dumps(prediction.review_1, indent=2))
    print("\n--- Validation Scores ---")
    print(json.dumps(prediction.scores.model_dump(), indent=2))

    return report.model_dump()


if __name__ == "__main__":
    run_demo()
