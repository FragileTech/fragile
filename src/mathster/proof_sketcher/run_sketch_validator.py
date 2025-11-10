#!/usr/bin/env python3
"""Integration harness for SketchValidator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

CACHE_DIR = Path("/tmp/dspy_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("DSPY_CACHEDIR", str(CACHE_DIR))

from mathster.parsing.config import configure_dspy
from mathster.proof_sketcher.sketch_validator import SketchValidator

SKETCH_JSON = json.dumps(
    {
        "title": "Euclidean Gas KL Convergence Sketch",
        "label": "thm-main-kl-convergence",
        "type": "Theorem",
        "source": "docs/source/1_euclidean_gas/09_kl_convergence.md",
        "date": "2025-02-14",
        "status": "Sketch",
        "statement": {
            "formal": (
                "Let μ_t be the law of the Euclidean Gas swarm governed by the cloning operator "
                "Ψ_clone at time t. If the potential V ensures a unique quasi-stationary distribution π_∞, "
                "and π_∞ satisfies an LSI with constant λ>0, then H(μ_t | π_∞) ≤ e^{-λ t} H(μ_0 | π_∞)."
            ),
            "informal": (
                "Entropy dissipation shows the swarm rapidly converges to equilibrium. "
                "The cloning dynamics do not degrade the exponential rate, which remains uniform in N."
            ),
        },
        "strategySynthesis": {
            "strategies": [
                {
                    "strategist": "Claude Sonnet 4.5 via SketchStrategist",
                    "method": "LSI Dissipation + Grönwall Iteration",
                    "keySteps": [
                        "Differentiate KL divergence w.r.t. time under Ψ_clone dynamics.",
                        "Use LSI to bound entropy dissipation uniformly in N.",
                        "Control cloning-induced cross terms via lem-cloning-conditional-independence.",
                        "Apply Grönwall to obtain exponential decay.",
                    ],
                    "strengths": [
                        "Directly leverages structural assumptions.",
                        "Gives explicit convergence rate.",
                    ],
                    "weaknesses": [
                        "Requires careful handling of residual cloning terms.",
                        "Uniformity in N must be justified rigorously.",
                    ],
                },
                {
                    "strategist": "GPT-5 Codex via SketchStrategist",
                    "method": "Generator Analysis + Tensorized LSI",
                    "keySteps": [
                        "Write generator for cloning flow and identify symmetric part.",
                        "Invoke tensorization to transfer single-particle LSI to N particles.",
                        "Bootstrap boundary regularity to prevent loss in λ.",
                    ],
                    "strengths": ["Highlights generator structure.", "Emphasizes tensorization."],
                    "weaknesses": ["Needs sharp boundary control.", "More technical preconditions."],
                },
            ],
            "recommendedApproach": {
                "chosenMethod": "LSI Dissipation + Grönwall Iteration",
                "rationale": "Cleaner entropy argument with fewer auxiliary lemmas.",
                "verificationStatus": {
                    "frameworkDependencies": "Verified",
                    "circularReasoning": "No circularity detected",
                    "keyAssumptions": "All assumptions are standard",
                    "crossValidation": "Consensus between strategists",
                },
            },
        },
        "dependencies": {
            "verifiedDependencies": [
                {
                    "type": "Theorem",
                    "label": "thm-lsi-target",
                    "sourceDocument": "docs/source/1_euclidean_gas/04_lsi.md",
                    "purpose": "Provides the uniform LSI constant λ.",
                    "usedInSteps": ["Step 2"],
                },
                {
                    "type": "Lemma",
                    "label": "lem-cloning-conditional-independence",
                    "sourceDocument": "docs/source/1_euclidean_gas/03_cloning.md",
                    "purpose": "Controls residual cloning terms.",
                    "usedInSteps": ["Step 3"],
                },
            ],
            "missingOrUncertainDependencies": None,
        },
        "detailedProof": {
            "overview": "Differentiate KL divergence, bound by LSI, control residual terms, apply Grönwall.",
            "topLevelOutline": [
                "Entropy evolution equation",
                "LSI-based dissipation bound",
                "Cloning correction control",
                "Application of Grönwall",
            ],
            "steps": [
                {
                    "stepNumber": 1,
                    "title": "Entropy Evolution",
                    "goal": "Compute d/dt H(μ_t | π_∞).",
                    "action": "Differentiate KL divergence using generator formalism.",
                    "justification": "Standard entropy calculus for Markov semigroups.",
                    "expectedResult": "dH/dt expressed via Dirichlet form + error.",
                    "dependencies": ["def-relative-entropy"],
                    "potentialIssues": None,
                },
                {
                    "stepNumber": 2,
                    "title": "LSI Dissipation",
                    "goal": "Bound Dirichlet form by λ H(μ_t | π_∞).",
                    "action": "Invoke thm-lsi-target to lower-bound entropy dissipation.",
                    "justification": "LSI implies -dH/dt ≥ 2λH.",
                    "expectedResult": "Inequality dH/dt ≤ -λH + cloning error.",
                    "dependencies": ["thm-lsi-target"],
                    "potentialIssues": "Need uniform λ in N.",
                },
            ],
            "conclusion": "Inequality yields exponential decay after Grönwall.",
        },
        "validationChecklist": {
            "logicalCompleteness": True,
            "hypothesisUsage": True,
            "conclusionDerivation": True,
            "frameworkConsistency": True,
            "noCircularReasoning": True,
            "constantTracking": True,
            "edgeCases": False,
            "regularityAssumptions": True,
        },
        "technicalDeepDives": [
            {
                "challengeTitle": "Uniform LSI Constant",
                "difficultyDescription": "Need λ independent of N.",
                "proposedSolution": "Use tensorization and Bakry-Émery curvature bounds.",
                "mathematicalDetail": "λ_N ≥ λ_single due to product structure.",
                "references": ["thm-lsi-target"],
            }
        ],
        "alternativeApproaches": [],
        "futureWork": {
            "remainingGaps": ["Boundary corrections for cloning residual terms."],
            "conjectures": [],
            "extensions": ["Extension to adaptive cloning with drift."],
        },
        "expansionRoadmap": {
            "phases": [
                {
                    "phaseTitle": "Stabilize LSI Argument",
                    "estimatedTime": "3 Days",
                    "tasks": [
                        {
                            "taskName": "Verify tensorization assumptions",
                            "strategy": "Check lemmas covering independence after cloning.",
                            "difficulty": "Medium",
                        }
                    ],
                }
            ],
            "totalEstimatedTime": "5 Days",
        },
        "crossReferences": {
            "theoremsUsed": ["thm-lsi-target", "thm-gronwall"],
            "definitionsUsed": ["def-relative-entropy"],
            "axiomsUsed": ["Axiom (Confining Potential)"],
            "relatedProofs": ["sketch_lsi_entropy_decay"],
            "downstreamConsequences": ["Establishes exponential mixing rate."],
        },
        "specialNotes": "Initial sketch assembled via dual strategist workflow.",
    },
    indent=2,
)


def _print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()


def run_sketch_validator_example() -> None:
    """Execute SketchValidator on the embedded KL convergence sketch."""
    configure_dspy(model="gemini/gemini-2.5-flash-lite-preview-09-2025")
    _print_header("Testing SketchValidator Agent")

    sketch = json.loads(SKETCH_JSON)
    sketch_label = sketch["label"]
    validation_cycle_id = str(uuid4())
    validation_timestamp = datetime.now(timezone.utc).isoformat()

    print("Input Proof Sketch Summary")
    print("-" * 80)
    print(f"Label: {sketch_label}")
    print(f"Title: {sketch['title']} ({sketch['status']})")
    print(f"Source: {sketch['source']}")
    print(f"Date:   {sketch['date']}")
    print()

    validator = SketchValidator(project_root=".")
    print("Running dual reviewer workflow (Gemini + Codex)...\n")

    prediction = validator(
        proof_sketch=sketch,
        sketch_label=sketch_label,
        validation_cycle_id=validation_cycle_id,
        validation_timestamp=validation_timestamp,
        reviewer_context=(
            "Audit the KL convergence proof sketch, focusing on LSI usage, cloning operator subtleties, "
            "and uniformity in N."
        ),
        final_decision_context="Base the final call on both reviewers' confidence and identified tasks.",
        confidence_context="Assumes all critical items listed in the action plan are completed.",
    )
    report = prediction.report

    _print_header("Validation Report Metadata")
    print(report.reportMetadata)
    print()

    _print_header("Decision & Confidence")
    sap = report.synthesisAndActionPlan
    print(f"Final Decision: {sap.finalDecision}")
    print(f"Confidence Statement: {sap.confidenceStatement}")
    print()

    _print_header("Consensus Analysis")
    ca = sap.consensusAnalysis
    print("Points of Agreement:")
    for point in ca.pointsOfAgreement:
        print(f"  • {point}")
    if ca.pointsOfDisagreement:
        print("\nPoints of Disagreement:")
        for entry in ca.pointsOfDisagreement:
            print(f"  Topic: {entry.topic}")
            print(f"    Gemini: {entry.geminiView}")
            print(f"    Codex:  {entry.codexView}")
            print(f"    Resolution: {entry.resolution}")
    print()
    print("Summary of Findings:")
    print(ca.summaryOfFindings)
    print()

    _print_header("Action Items")
    if sap.actionableItems:
        for item in sap.actionableItems:
            print(f"[{item.priority}] {item.itemId}: {item.description}")
            if item.references:
                print(f"  References: {', '.join(item.references)}")
            print()
    else:
        print("No outstanding action items. Sketch approved as-is.\n")

    _print_header("Complete Validation Report (JSON)")
    print(json.dumps(report.model_dump(), indent=2))
    print()
    print("✓ SketchValidator workflow completed successfully!")


def main() -> None:
    run_sketch_validator_example()


if __name__ == "__main__":
    main()
