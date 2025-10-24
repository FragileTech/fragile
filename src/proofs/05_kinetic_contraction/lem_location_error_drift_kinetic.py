"""
Symbolic Validation for Lemma: Drift of Location Error Under Kinetics

Source: docs/source/1_euclidean_gas/05_kinetic_contraction.md
Theorem Label: lem-location-error-drift-kinetic
Generated: 2025-10-24
Agent: Math Verifier v1.0

This module provides sympy-based validation of algebraic manipulations
in the proof of lem-location-error-drift-kinetic. Each function validates one algebraic step.
"""

from sympy import symbols, simplify, Matrix, eye, zeros
import pytest

# ========================================
# FRAMEWORK SYMBOLS (from glossary.md)
# ========================================

# γ (gamma): friction coefficient
# λ_v (lambda_v): velocity weight in hypocoercive norm
# b: coupling parameter in hypocoercive norm

# ========================================
# VALIDATION FUNCTIONS
# ========================================

def test_drift_matrix_calculation():
    """
    Verify: Drift matrix D = M^T Q + QM calculation

    Source: docs/source/1_euclidean_gas/05_kinetic_contraction.md, lines 1323-1335
    Category: Matrix Forms and Quadratic Functions (Hypocoercive Cost)

    Matrices:
      - Q = [[I_d, (b/2) I_d], [(b/2) I_d, lambda_v I_d]]
      - M = [[0, I_d], [0, -gamma I_d]]
      - D = M^T Q + Q M
    """

    # Define symbols with appropriate assumptions (match framework names)
    gamma, lambda_v, b = symbols('gamma lambda_v b', real=True, positive=True)

    # Verify both scalar (d=1) and block (d=3) cases to check structure and symmetry
    for d in (1, 3):
        Id = eye(d)
        Z = zeros(d)

        # Define matrices M and Q in block form
        M = Matrix.vstack(Matrix.hstack(Z, Id),
                          Matrix.hstack(Z, -gamma * Id))

        Q = Matrix.vstack(Matrix.hstack(Id, (b / 2) * Id),
                          Matrix.hstack((b / 2) * Id, lambda_v * Id))

        # Step 1: Compute M^T Q
        MTQ = M.T * Q
        MTQ_expected = Matrix.vstack(
            Matrix.hstack(zeros(d), zeros(d)),
            Matrix.hstack((1 - (b * gamma) / 2) * Id, (b / 2 - gamma * lambda_v) * Id)
        )
        diff_MTQ = (MTQ - MTQ_expected).applyfunc(simplify)
        assert diff_MTQ == zeros(2 * d), (
            f"Step 1 failed for d={d}: M^T Q mismatch.\n"
            f"Difference:\n{diff_MTQ}"
        )

        # Step 2: Compute QM
        QM = Q * M
        QM_expected = Matrix.vstack(
            Matrix.hstack(zeros(d), (1 - (b * gamma) / 2) * Id),
            Matrix.hstack(zeros(d), (b / 2 - gamma * lambda_v) * Id)
        )
        diff_QM = (QM - QM_expected).applyfunc(simplify)
        assert diff_QM == zeros(2 * d), (
            f"Step 2 failed for d={d}: Q M mismatch.\n"
            f"Difference:\n{diff_QM}"
        )

        # Step 3: Compute D = M^T Q + Q M
        D = MTQ + QM
        D_expected = Matrix.vstack(
            Matrix.hstack(zeros(d), (1 - (b * gamma) / 2) * Id),
            Matrix.hstack((1 - (b * gamma) / 2) * Id, (b - 2 * gamma * lambda_v) * Id)
        )
        diff_D = (D - D_expected).applyfunc(simplify)
        assert diff_D == zeros(2 * d), (
            f"Step 3 failed for d={d}: D = M^T Q + Q M mismatch.\n"
            f"Difference:\n{diff_D}"
        )

        # Edge-case checks: block structure and symmetry
        assert D[0:d, 0:d] == zeros(d), f"Top-left block not zero for d={d}."
        assert D[0:d, d:2*d] == (1 - (b * gamma) / 2) * Id, f"Top-right block incorrect for d={d}."
        assert D[d:2*d, 0:d] == (1 - (b * gamma) / 2) * Id, f"Bottom-left block incorrect for d={d}."
        assert D[d:2*d, d:2*d] == (b - 2 * gamma * lambda_v) * Id, f"Bottom-right block incorrect for d={d}."
        assert D == D.T, f"D is not symmetric for d={d}."

    print("✓ Drift matrix calculation verified")

# ========================================
# TEST RUNNER
# ========================================

def run_all_validations():
    """Run all validation tests for lem-location-error-drift-kinetic"""
    tests = [
        test_drift_matrix_calculation,
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
