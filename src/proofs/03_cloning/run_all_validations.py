"""
Symbolic Validation Suite for 03_cloning.md

This module provides sympy-based validation of algebraic manipulations
in the Keystone Principle proof. Each function validates one algebraic step.

Generated: 2025-10-24
Agent: Math Verifier v1.0
Source: docs/source/1_euclidean_gas/03_cloning.md
"""

from pathlib import Path
import sys


# Import all validation functions
sys.path.insert(0, str(Path(__file__).parent))

from test_companion_fitness_gap import test_companion_fitness_gap_algebra
from test_mean_decomposition_gap import test_mean_decomposition_gap
from test_mean_deviations import test_mean_deviations_from_total
from test_variance_identity_gemini import test_between_group_variance_identity
from test_within_group_variance_bound import test_within_group_variance_bound


def run_all_validations():
    """
    Run all validation tests for 03_cloning.md key theorems

    Returns:
        tuple: (passed_count, failed_count)
    """
    tests = [
        ("Between-group variance identity (Claim 1B)", test_between_group_variance_identity),
        ("Mean deviations from total (Claim 1A)", test_mean_deviations_from_total),
        ("Within-group variance bound (Claim 1C)", test_within_group_variance_bound),
        ("Companion fitness gap algebra (Claim 2B)", test_companion_fitness_gap_algebra),
        ("Mean decomposition gap (Claim 2C)", test_mean_decomposition_gap),
    ]

    passed = 0
    failed = 0

    print("=" * 80)
    print("SYMBOLIC VALIDATION SUITE FOR 03_CLONING.MD")
    print("=" * 80)
    print(f"\nRunning {len(tests)} algebraic validation tests...\n")

    for i, (name, test_func) in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] Testing: {name}")
        print("-" * 80)
        try:
            test_func()
            passed += 1
            print(f"✅ PASSED: {name}\n")
        except AssertionError as e:
            print(f"❌ FAILED: {name}")
            print(f"   Error: {e}\n")
            failed += 1
        except Exception as e:
            print(f"⚠️  ERROR: {name}")
            print(f"   Unexpected error: {e}\n")
            failed += 1

    print("=" * 80)
    print(f"VALIDATION SUMMARY: {passed} passed, {failed} failed")
    print("=" * 80)

    if failed == 0:
        print("\n✅ All algebraic validations PASSED!")
    else:
        print(f"\n⚠️  {failed} validation(s) FAILED - manual review required")

    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_validations()
    sys.exit(0 if failed == 0 else 1)
