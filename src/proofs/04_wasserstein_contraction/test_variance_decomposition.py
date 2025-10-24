"""
Symbolic Validation for Variance Decomposition by Clusters

Source: docs/source/1_euclidean_gas/04_wasserstein_contraction.md
Theorem Label: lem-variance-decomposition
Generated: 2025-10-24
Agent: Math Verifier v1.0

This module provides sympy-based validation of algebraic manipulations
in the proof of variance decomposition lemma.
"""

from sympy import symbols, simplify, factor

def test_variance_decomposition_by_clusters():
    """
    Verify: Variance decomposition formula - factorization step

    Source: docs/source/1_euclidean_gas/04_wasserstein_contraction.md, lines 259-328
    Category: Variance Decomposition
    """

    # 1. Define symbols with appropriate assumptions from the proof
    # f_I, f_J: population fractions (positive, sum to 1)
    # k: total swarm size (positive integer)
    # I_k_size, J_k_size: cluster sizes, represented as |I_k| and |J_k|
    f_I, f_J = symbols('f_I f_J', real=True, positive=True)
    k = symbols('k', integer=True, positive=True)
    I_k_size, J_k_size = symbols('|I_k| |J_k|', integer=True, positive=True)

    # 2. Define the Left-Hand Side (LHS) and Right-Hand Side (RHS) of the identity to be verified
    # Identity: |I_k| * f_J**2 + |J_k| * f_I**2 = k * f_I * f_J
    lhs = I_k_size * f_J**2 + J_k_size * f_I**2
    rhs = k * f_I * f_J

    # 3. Define the constraints from the lemma as substitution rules
    # Constraint 1: |I_k| = f_I * k
    # Constraint 2: |J_k| = f_J * k
    # Constraint 3: f_I + f_J = 1
    substitutions = {
        I_k_size: f_I * k,
        J_k_size: f_J * k,
    }

    # 4. Apply the size substitutions to the LHS
    lhs_substituted = lhs.subs(substitutions)

    # 5. Factor the substituted LHS to reveal the (f_I + f_J) term, mimicking the proof's logic
    # Expect: k*f_I*f_J*(f_I + f_J)
    factored_lhs = factor(lhs_substituted)

    # 6. Apply the final constraint, f_I + f_J = 1
    final_lhs = factored_lhs.subs({f_I + f_J: 1})

    # 7. Verify that the fully simplified LHS equals the RHS
    # A robust way to check for algebraic equivalence is to see if their difference simplifies to zero.
    difference = final_lhs - rhs
    simplified_difference = simplify(difference)

    assertion_message = (
        f"Verification failed! The algebraic identity does not hold.\n"
        f"LHS (after substitutions): {final_lhs}\n"
        f"RHS: {rhs}\n"
        f"Simplified Difference (LHS - RHS): {simplified_difference} (expected 0)"
    )

    assert simplified_difference == 0, assertion_message

    print("✓ Variance decomposition factorization step verified successfully.")
    print(f"  Identity: |I_k|*f_J**2 + |J_k|*f_I**2 = k*f_I*f_J")
    print(f"  Verified using constraints: |I_k|=k*f_I, |J_k|=k*f_J, f_I+f_J=1")


def run_all_validations():
    """Run all validation tests for variance decomposition"""
    tests = [
        test_variance_decomposition_by_clusters,
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

    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    return passed, failed

if __name__ == "__main__":
    run_all_validations()
