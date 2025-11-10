#!/usr/bin/env python3
"""Test script for SketchStrategist agent with sample theorem from Fragile framework."""

from __future__ import annotations

import json
from pathlib import Path

from mathster.agent_schemas.signatures import SketchStrategist
from mathster.parsing.config import configure_dspy


def run_sketch_strategist():
    """Test SketchStrategist with a realistic KL convergence theorem."""
    configure_dspy(model="gemini/gemini-2.5-flash-lite-preview-09-2025")
    print("=" * 80)
    print("Testing SketchStrategist Agent")
    print("=" * 80)
    print()

    # Sample theorem from 09_kl_convergence
    theorem_label = "thm-main-kl-convergence"

    theorem_statement = """
Under the following axioms:
1. Axiom (Confining Potential): The potential V satisfies confining conditions
   ensuring the existence of a unique quasi-stationary distribution (QSD)
2. Axiom (QSD Log-Concave): The QSD π_∞ satisfies log-Sobolev inequality (LSI)
   with constant λ > 0

The Euclidean Gas with cloning operator Ψ_clone converges exponentially fast
in relative entropy (KL divergence) to the unique QSD π_∞:

    H(μ_t | π_∞) ≤ e^{-λt} H(μ_0 | π_∞)

where μ_t is the law of the swarm at time t, and the rate λ is independent
of the number of walkers N.
"""

    framework_context = """
Available framework results:
- thm-lsi-target: Log-Sobolev inequality for the target measure with explicit constant
- lem-cloning-contraction: Cloning operator contracts Wasserstein distance
- thm-bakry-emery: Bakry-Émery criterion for LSI via Γ_2 calculus
- lem-tensorization: Tensorization of LSI for product measures
- lem-cloning-conditional-independence: Conditional independence after cloning
- thm-gronwall: Grönwall's lemma for differential inequalities
- def-relative-entropy: Definition of KL divergence H(μ|ν)
- ax-bounded-displacement: Axiom ensuring Lipschitz bounds on projections
"""

    operator_notes = """
Preferences and constraints:
- Prefer LSI-based convergence approach over direct Lyapunov construction
- Use Grönwall iteration for exponential rate
- Ensure all bounds are uniform in N (number of walkers)
- Avoid non-constructive compactness arguments
- Leverage tensorization for independence structure
"""

    print("Input Configuration:")
    print("-" * 80)
    print(f"Theorem Label: {theorem_label}")
    print(f"\nTheorem Statement:{theorem_statement}")
    print(f"\nFramework Context:{framework_context}")
    print(f"\nOperator Notes:{operator_notes}")
    print()

    try:
        # Initialize strategist
        print("Initializing SketchStrategist...")
        strategist = SketchStrategist()
        print("✓ SketchStrategist initialized successfully")
        print()

        # Generate strategy
        print("Generating proof sketch strategy...")
        print("(This may take a moment as the agent orchestrates multiple tool calls)")
        print()

        result = strategist(
            theorem_label=theorem_label,
            theorem_statement=theorem_statement,
            framework_context=framework_context,
            operator_notes=operator_notes,
        )

        print("✓ Strategy generation complete!")
        print()

        # Extract strategy from result
        strategy = result.strategy

        # Display results in structured format
        print("=" * 80)
        print("GENERATED PROOF SKETCH STRATEGY")
        print("=" * 80)
        print()

        # Basic Strategy Items
        print("1. BASIC STRATEGY ITEMS")
        print("-" * 80)
        print(f"Strategist: {strategy.strategist}")
        print(f"Method: {strategy.method}")
        print()
        print("Summary:")
        print(f"  {strategy.summary}")
        print()

        print("Key Steps:")
        for i, step in enumerate(strategy.keySteps, 1):
            print(f"  {i}. {step}")
        print()

        print("Strengths:")
        for i, strength in enumerate(strategy.strengths, 1):
            print(f"  + {strength}")
        print()

        print("Weaknesses:")
        for i, weakness in enumerate(strategy.weaknesses, 1):
            print(f"  - {weakness}")
        print()

        # Framework Dependencies
        print("2. FRAMEWORK DEPENDENCIES")
        print("-" * 80)

        deps = strategy.frameworkDependencies
        print(f"Theorems ({len(deps.theorems)}):")
        for dep in deps.theorems:
            print(f"  • {dep.label} ({dep.document})")
            print(f"    Purpose: {dep.purpose}")
            if dep.usedInSteps:
                print(f"    Used in steps: {', '.join(dep.usedInSteps)}")
        print()

        print(f"Lemmas ({len(deps.lemmas)}):")
        for dep in deps.lemmas:
            print(f"  • {dep.label} ({dep.document})")
            print(f"    Purpose: {dep.purpose}")
            if dep.usedInSteps:
                print(f"    Used in steps: {', '.join(dep.usedInSteps)}")
        print()

        print(f"Axioms ({len(deps.axioms)}):")
        for dep in deps.axioms:
            print(f"  • {dep.label} ({dep.document})")
            print(f"    Purpose: {dep.purpose}")
            if dep.usedInSteps:
                print(f"    Used in steps: {', '.join(dep.usedInSteps)}")
        print()

        print(f"Definitions ({len(deps.definitions)}):")
        for dep in deps.definitions:
            print(f"  • {dep.label} ({dep.document})")
            print(f"    Purpose: {dep.purpose}")
            if dep.usedInSteps:
                print(f"    Used in steps: {', '.join(dep.usedInSteps)}")
        print()

        # Technical Deep Dives
        print("3. TECHNICAL DEEP DIVES")
        print("-" * 80)
        print(f"Number of technical challenges identified: {len(strategy.technicalDeepDives)}")
        print()

        for i, dive in enumerate(strategy.technicalDeepDives, 1):
            print(f"Challenge {i}: {dive.challengeTitle}")
            print(f"  Difficulty: {dive.difficultyDescription}")
            print(f"  Proposed Solution: {dive.proposedSolution}")
            if dive.references:
                print(f"  References: {', '.join(dive.references)}")
            print()

        # Confidence Score
        print("4. CONFIDENCE ASSESSMENT")
        print("-" * 80)
        print(f"Confidence Score: {strategy.confidenceScore}")
        print()

        # Summary Statistics
        print("=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        total_deps = (
            len(deps.theorems) + len(deps.lemmas) +
            len(deps.axioms) + len(deps.definitions)
        )
        print(f"Total framework dependencies: {total_deps}")
        print(f"  - Theorems: {len(deps.theorems)}")
        print(f"  - Lemmas: {len(deps.lemmas)}")
        print(f"  - Axioms: {len(deps.axioms)}")
        print(f"  - Definitions: {len(deps.definitions)}")
        print()
        print(f"Technical challenges: {len(strategy.technicalDeepDives)}")
        print(f"Proof steps: {len(strategy.keySteps)}")
        print(f"Confidence: {strategy.confidenceScore}")
        print()

        # Full JSON dump
        print("=" * 80)
        print("FULL STRATEGY (JSON)")
        print("=" * 80)
        print(json.dumps(strategy.model_dump(), indent=2))
        print()

        print("=" * 80)
        print("✓ TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)

        return strategy

    except Exception as e:
        print("=" * 80)
        print("✗ ERROR DURING STRATEGY GENERATION")
        print("=" * 80)
        print(f"\nError type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print()

        import traceback
        print("Full traceback:")
        print("-" * 80)
        traceback.print_exc()
        print()

        raise


def main():
    """Run the test."""
    run_sketch_strategist()


if __name__ == "__main__":
    main()
