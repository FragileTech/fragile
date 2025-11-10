#!/usr/bin/env python3
"""Integration test for ProofSketchAgent using Gemini 2.5 Flash Lite."""

from __future__ import annotations

import json
from pathlib import Path

from mathster.proof_sketcher.sketcher import ProofSketchAgent
from mathster.parsing.config import configure_dspy


def _print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()


def run_sketch_agent() -> None:
    """Execute ProofSketchAgent on a representative KL convergence theorem."""
    configure_dspy(model="gemini/gemini-2.5-flash-lite-preview-09-2025")
    _print_header("Testing ProofSketchAgent")

    # Metadata / inputs
    title_hint = "Euclidean Gas KL Convergence Sketch"
    theorem_label = "thm-main-kl-convergence"
    theorem_type = "Theorem"
    document_source = "docs/source/1_euclidean_gas/09_kl_convergence.md"
    creation_date = "2025-02-14"
    proof_status = "Sketch"

    theorem_statement = """
Under the following assumptions:
1. Axiom (Confining Potential): The potential V ensures a unique quasi-stationary distribution π_∞.
2. Axiom (QSD Log-Concave): The QSD π_∞ satisfies a log-Sobolev inequality (LSI) with constant λ > 0.

Let μ_t denote the law of the Euclidean Gas swarm with cloning operator Ψ_clone at time t.
Then μ_t converges exponentially fast in relative entropy to π_∞:
    H(μ_t | π_∞) ≤ e^{-λ t} H(μ_0 | π_∞),
with a rate λ that is uniform in the number of walkers N.
"""

    framework_context = """
Supporting results likely required:
- thm-lsi-target: Establishes LSI for the target measure with explicit constant λ.
- lem-cloning-contraction: Shows Ψ_clone contracts Wasserstein distance.
- thm-bakry-emery: Bakry–Émery criterion for LSI via Γ_2 calculus.
- lem-tensorization: Tensorization of LSI for product measures (needed for N walkers).
- lem-cloning-conditional-independence: Controls dependencies introduced by cloning.
- thm-gronwall: Grönwall's lemma used to convert differential inequality into exponential bound.
- def-relative-entropy: Defines KL divergence for stochastic processes.
"""

    operator_notes = """
Guidance for the proof pipeline:
- Prefer strategies that combine LSI + Grönwall arguments.
- Require all constants to be uniform in N.
- Avoid vague compactness arguments; highlight exact lemmas that guarantee limits.
- Document any assumptions about cloning regularity or boundary effects.
"""

    print("Input Overview")
    print("-" * 80)
    print(f"Title: {title_hint}")
    print(f"Label: {theorem_label}")
    print(f"Type: {theorem_type}")
    print(f"Document Source: {document_source}")
    print(f"Status: {proof_status} (created {creation_date})")
    print()

    try:
        print("Initializing ProofSketchAgent...")
        agent = ProofSketchAgent()
        print("✓ ProofSketchAgent ready\n")

        print("Generating complete proof sketch (this will orchestrate many sub-agents)...\n")
        prediction = agent(
            title_hint=title_hint,
            theorem_label=theorem_label,
            theorem_type=theorem_type,
            theorem_statement=theorem_statement,
            document_source=document_source,
            creation_date=creation_date,
            proof_status=proof_status,
            framework_context=framework_context,
            operator_notes=operator_notes,
        )
        sketch = prediction.sketch
        print("✓ ProofSketch generated successfully!\n")

        # Display key sections
        _print_header("Proof Statement")
        print("Formal Statement:")
        print(sketch.statement.formal.strip())
        print()
        print("Informal Statement:")
        print(sketch.statement.informal.strip())
        print()

        _print_header("Strategy Synthesis")
        rec = sketch.strategySynthesis.recommendedApproach
        print(f"Chosen Method: {rec.chosenMethod}")
        print(f"Rationale: {rec.rationale}")
        print(f"Verification Status: {rec.verificationStatus}")
        print()

        _print_header("Dependency Ledger Summary")
        print(f"Verified Dependencies: {len(sketch.dependencies.verifiedDependencies)} entries")
        missing = sketch.dependencies.missingOrUncertainDependencies
        if missing:
            print("Missing Lemmas:", len(missing.lemmasToProve))
            print("Uncertain Assumptions:", len(missing.uncertainAssumptions))
        else:
            print("Missing Lemmas: 0")
            print("Uncertain Assumptions: 0")
        print()

        _print_header("Detailed Proof Outline")
        print("Top-level Outline:")
        for i, entry in enumerate(sketch.detailedProof.topLevelOutline, 1):
            print(f"  {i}. {entry}")
        print()
        print("First 2 Proof Steps:")
        for step in sketch.detailedProof.steps[:2]:
            print(f"- Step {step.stepNumber}: {step.title}")
            print(f"  Goal: {step.goal}")
            print(f"  Action: {step.action}")
            print()

        _print_header("Technical Deep Dives")
        if sketch.technicalDeepDives:
            for dive in sketch.technicalDeepDives:
                print(f"* {dive.challengeTitle}: {dive.difficultyDescription}")
                print(f"  Solution: {dive.proposedSolution}")
                if dive.references:
                    print(f"  References: {', '.join(dive.references)}")
                print()
        else:
            print("No technical deep dives recorded.\n")

        _print_header("Validation Checklist")
        print(json.dumps(sketch.validationChecklist.model_dump(), indent=2))
        print()

        _print_header("Future Work & Expansion Roadmap")
        print("Future Work:")
        if sketch.futureWork is not None:
            print(json.dumps(sketch.futureWork.model_dump(), indent=2))
        else:
            print("  (Future work section empty)")
        print()
        print("Expansion Roadmap:")
        if sketch.expansionRoadmap is not None:
            print(json.dumps(sketch.expansionRoadmap.model_dump(), indent=2))
        else:
            print("  (Expansion roadmap not provided)")
        print()

        _print_header("Full ProofSketch (JSON)")
        print(json.dumps(sketch.model_dump(), indent=2))
        print()
        print("✓ ProofSketchAgent test completed successfully!")

    except Exception as exc:  # noqa: BLE001
        _print_header("✗ ProofSketchAgent encountered an error")
        print(f"{type(exc).__name__}: {exc}")
        print()
        import traceback

        traceback.print_exc()
        raise


def main() -> None:
    """CLI entrypoint."""
    run_sketch_agent()


if __name__ == "__main__":
    main()
