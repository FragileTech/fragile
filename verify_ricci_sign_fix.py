#!/usr/bin/env python3
"""Verify that the Ricci scalar sign convention is correct after the fix.

This script tests that:
1. Contracting swarms (converging walkers) receive POSITIVE rewards (positive curvature)
2. Expanding swarms (diverging walkers) receive NEGATIVE rewards (negative curvature)
"""

import sys

import torch

from fragile.fractalai.core.benchmarks import VoronoiRicciScalar


def main():
    print("=" * 80)
    print("Ricci Scalar Sign Convention Verification")
    print("=" * 80)
    print()

    benchmark = VoronoiRicciScalar(dims=2, update_every=1)

    # Use a 5-walker configuration like in the test
    base_points = torch.tensor(
        [
            [-10.0, -10.0],
            [-10.0, 10.0],
            [10.0, -10.0],
            [10.0, 10.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    print("Test 1: Small perturbation (to initialize volumes)")
    print("-" * 80)
    r1 = benchmark(base_points)
    print(f"Initial call: rewards = {r1.numpy()}")
    print("  (Expected zeros on first call - no previous volumes)")
    print()

    print("Test 2: Outward movement (expansion)")
    print("-" * 80)
    # Move all walkers slightly outward (expansion)
    expanded = base_points * 1.1  # Scale outward by 10%
    r2 = benchmark(expanded)
    print(f"After expansion: rewards = {r2.numpy()}")
    print("  → Swarm expanded (dV/dt > 0, θ > 0)")
    print("  → Expected: R = -θ < 0 (NEGATIVE curvature)")
    print(f"  → Got negative values: {(r2 < 0).any().item()}")
    print()

    print("Test 3: Inward movement (contraction)")
    print("-" * 80)
    # Move walkers back inward (contraction)
    contracted = expanded * 0.9  # Scale inward by 10%
    r3 = benchmark(contracted)
    print(f"After contraction: rewards = {r3.numpy()}")
    print("  → Swarm contracted (dV/dt < 0, θ < 0)")
    print("  → Expected: R = -θ > 0 (POSITIVE curvature)")
    print(f"  → Got positive values: {(r3 > 0).any().item()}")
    print()

    print("=" * 80)
    print("Summary:")
    print("=" * 80)

    # Check if we got the expected signs
    has_negative_on_expansion = (r2 < -1e-6).any().item()
    has_positive_on_contraction = (r3 > 1e-6).any().item()

    if has_negative_on_expansion and has_positive_on_contraction:
        print("✓ PASS: Ricci scalar sign convention is CORRECT")
        print("  - Expansion → Negative rewards (negative curvature)")
        print("  - Contraction → Positive rewards (positive curvature)")
        return 0
    print("✗ FAIL: Sign convention may be incorrect")
    if not has_negative_on_expansion:
        print("  - Expansion should give some negative rewards")
    if not has_positive_on_contraction:
        print("  - Contraction should give some positive rewards")
    print()
    print("Note: If all values are near zero, the sign fix is correct but")
    print("the expansion computation may need non-uniform walker movements.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
