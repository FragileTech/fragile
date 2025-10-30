"""
Symbolic Validation for Quadratic Distance Identity

Source: docs/source/1_euclidean_gas/04_wasserstein_contraction.md
Theorem Label: lem-expected-distance-change
Generated: 2025-10-24
Agent: Math Verifier v1.0

This module validates the quadratic identity used in expected distance change calculation.
"""

from sympy import expand, simplify, symbols


def test_quadratic_identity_distance_change():
    """
    Verify: ||a-c||² - ||b-c||² = ||a-b||² + 2⟨a-b, b-c⟩
    Source: docs/source/1_euclidean_gas/04_wasserstein_contraction.md, line 657
    """
    # 1. Symbol Definitions
    # Use scalar components for a concrete verification
    # For vectors in R^d, we use component notation
    a1, a2, a3 = symbols("a1 a2 a3", real=True)
    b1, b2, b3 = symbols("b1 b2 b3", real=True)
    c1, c2, c3 = symbols("c1 c2 c3", real=True)

    # Define squared norm and inner product using components
    def squared_norm_3d(v1, v2, v3, u1, u2, u3):
        """Compute ||v - u||^2"""
        return (v1 - u1) ** 2 + (v2 - u2) ** 2 + (v3 - u3) ** 2

    def inner_product_3d(v1, v2, v3, u1, u2, u3):
        """Compute ⟨v, u⟩"""
        return (v1 * u1) + (v2 * u2) + (v3 * u3)

    # 2. Construct LHS: ||a-c||² - ||b-c||²
    lhs = squared_norm_3d(a1, a2, a3, c1, c2, c3) - squared_norm_3d(b1, b2, b3, c1, c2, c3)

    # 3. Construct RHS: ||a-b||² + 2⟨a-b, b-c⟩
    # First compute ||a-b||²
    norm_ab_sq = squared_norm_3d(a1, a2, a3, b1, b2, b3)

    # Then compute ⟨a-b, b-c⟩
    inner_ab_bc = inner_product_3d(
        a1 - b1,
        a2 - b2,
        a3 - b3,  # a - b
        b1 - c1,
        b2 - c2,
        b3 - c3,  # b - c
    )

    rhs = norm_ab_sq + 2 * inner_ab_bc

    # 4. Verify the identity by expansion and simplification
    lhs_expanded = expand(lhs)
    rhs_expanded = expand(rhs)

    difference = lhs_expanded - rhs_expanded
    simplified_difference = simplify(difference)

    # 5. Assert the result
    assert (
        simplified_difference == 0
    ), f"Verification failed! Difference is not zero: {simplified_difference}"

    print("✅ Algebraic identity verified successfully.")
    print("   Source: docs/source/1_euclidean_gas/04_wasserstein_contraction.md, line 657")
    print("   Identity: ||a-c||² - ||b-c||² = ||a-b||² + 2⟨a-b, b-c⟩")
    print("   Verified in R³ (generalizes to arbitrary dimension)")
    print(f"   Simplified difference: {simplified_difference} (equals 0 ✓)")


def run_all_validations():
    """Run all validation tests for quadratic identity"""
    tests = [
        test_quadratic_identity_distance_change,
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
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"VALIDATION SUMMARY: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    return passed, failed


if __name__ == "__main__":
    run_all_validations()
