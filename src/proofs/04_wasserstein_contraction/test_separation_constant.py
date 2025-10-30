"""
Symbolic Validation for Separation Constant Factorization

Source: docs/source/1_euclidean_gas/04_wasserstein_contraction.md
Theorem Label: cor-between-group-dominance
Generated: 2025-10-24
Agent: Math Verifier v1.0

This module validates the c_sep constant factorization in between-group variance.
"""

import sympy


def test_separation_constant_factorization():
    """
    Verify: c_sep factorization in between-group variance
    Source: docs/source/1_euclidean_gas/04_wasserstein_contraction.md, lines 362-367
    """
    # Define positive constants based on the problem description
    f_UH = sympy.Symbol("f_UH", positive=True)
    c_pack = sympy.Symbol("c_pack", positive=True)
    lambda_v = sympy.Symbol("lambda_v", positive=True)
    V_struct = sympy.Symbol("V_struct", positive=True)

    # Define the left-hand side (LHS) of the inequality, which is the starting expression
    # f_I f_J \|\mu_x(I_k) - \mu_x(J_k)\|^2 >= ...
    # We are verifying the algebraic manipulation on the right side of the inequality.

    # Original expression from the inequality's right-hand side
    lhs_expression = (f_UH / 2) * c_pack * (V_struct / (1 + lambda_v))

    # Define the separation constant c_sep
    c_sep = (f_UH * c_pack) / (2 * (1 + lambda_v))

    # Define the right-hand side (RHS) using the c_sep substitution
    rhs_expression = c_sep * V_struct

    # Verify that the LHS and RHS are algebraically equivalent.
    # The difference between the two expressions should simplify to zero.
    difference = lhs_expression - rhs_expression
    simplified_difference = sympy.simplify(difference)

    assert simplified_difference == 0, f"Verification failed! Difference: {simplified_difference}"

    print("✓ Separation constant factorization verified successfully.")
    print("  c_sep = f_UH · c_pack / (2(1 + λ_v))")
    print("  Verified: (f_UH/2)·c_pack·V_struct/(1+λ_v) = c_sep·V_struct")


def run_all_validations():
    """Run all validation tests for separation constant"""
    tests = [
        test_separation_constant_factorization,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"VALIDATION SUMMARY: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    return passed, failed


if __name__ == "__main__":
    run_all_validations()
