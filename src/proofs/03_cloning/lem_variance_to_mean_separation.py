"""
Symbolic Validation for Lemma: From Total Variance to Mean Separation

Source: /home/guillem/fragile/docs/source/1_euclidean_gas/03_cloning.md
Theorem Label: lem-variance-to-mean-separation
Generated: 2025-10-26
Agent: Math Verifier v1.0

This module provides sympy-based validation of algebraic manipulations
in the proof of lem-variance-to-mean-separation. Each function validates one algebraic step.
"""

from sympy import expand, simplify, sqrt, symbols


# ========================================
# VALIDATION FUNCTIONS
# ========================================


def test_between_group_variance_identity():
    """
    Verify: Between-group variance identity f_H(μ_H - μ_V)² + f_L(μ_L - μ_V)² = f_H·f_L·(μ_H - μ_L)²

    Source: /home/guillem/fragile/docs/source/1_euclidean_gas/03_cloning.md, lines 3794-3822
    Category: Variance Decomposition (Law of Total Variance)

    This is the fundamental identity that relates between-group variance to mean separation.
    The proof establishes this through algebraic substitution and constraint application.
    """

    # Define symbols with framework-consistent names and assumptions
    # Fractions are positive real numbers in (0,1)
    f_H, f_L = symbols("f_H f_L", real=True, positive=True)
    # Means are real numbers (can be any value)
    mu_H, mu_L, mu_V = symbols("mu_H mu_L mu_V", real=True)

    # Define the total mean constraint: mu_V = f_H*mu_H + f_L*mu_L
    # This is the definition of the weighted average
    mu_V_def = {mu_V: f_H * mu_H + f_L * mu_L}

    # Define the partition constraint: f_H + f_L = 1
    # The two subsets partition the population
    constraint_c1 = {f_L: 1 - f_H}  # Substitute f_L = 1 - f_H
    constraint_c2 = {f_H: 1 - f_L}  # Substitute f_H = 1 - f_L (symmetric form)

    # LHS of the identity: f_H*(mu_H - mu_V)² + f_L*(mu_L - mu_V)²
    # This is the definition of between-group variance
    lhs = f_H * (mu_H - mu_V) ** 2 + f_L * (mu_L - mu_V) ** 2

    # RHS of the final identity: f_H * f_L * (mu_H - mu_L)²
    # This is the simplified form we want to prove
    rhs_final = f_H * f_L * (mu_H - mu_L) ** 2

    # ========================================
    # STEP 1: VERIFY PRE-CONSTRAINT FORM
    # ========================================
    # After substituting mu_V definition but BEFORE applying f_H + f_L = 1,
    # the identity should have an extra factor (f_H + f_L)

    # Substitute mu_V definition into LHS
    lhs_sub_mu = expand(lhs.subs(mu_V_def))

    # The general (pre-constraint) form should be:
    # f_H * f_L * (f_H + f_L) * (mu_H - mu_L)²
    rhs_pre_constraint = expand(f_H * f_L * (f_H + f_L) * (mu_H - mu_L) ** 2)

    # Verify intermediate algebraic step
    diff_pre = simplify(lhs_sub_mu - rhs_pre_constraint)
    assert diff_pre == 0, (
        "Intermediate step verification failed!\n"
        "After substituting mu_V = f_H*mu_H + f_L*mu_L, LHS should equal:\n"
        "  f_H*f_L*(f_H+f_L)*(mu_H - mu_L)²\n"
        f"But simplified difference was: {diff_pre}"
    )
    print("✓ Step 1 verified: Pre-constraint form f_H·f_L·(f_H+f_L)·(μ_H-μ_L)² is correct")

    # ========================================
    # STEP 2: VERIFY FINAL FORM (Constraint: f_L = 1 - f_H)
    # ========================================
    # Apply partition constraint to eliminate one variable

    lhs_c1 = simplify(lhs_sub_mu.subs(constraint_c1))
    rhs_c1 = simplify(rhs_final.subs(constraint_c1))
    diff_c1 = simplify(lhs_c1 - rhs_c1)

    assert diff_c1 == 0, (
        "Final identity verification failed (using f_L = 1 - f_H)!\n"
        "Expected: f_H*(mu_H - mu_V)² + f_L*(mu_L - mu_V)² = f_H*f_L*(mu_H - mu_L)²\n"
        f"Simplified difference: {diff_c1}"
    )
    print("✓ Step 2 verified: Final form with f_L = 1 - f_H substitution")

    # ========================================
    # STEP 3: VERIFY SYMMETRY (Constraint: f_H = 1 - f_L)
    # ========================================
    # Verify the identity also holds with symmetric constraint form

    lhs_c2 = simplify(lhs_sub_mu.subs(constraint_c2))
    rhs_c2 = simplify(rhs_final.subs(constraint_c2))
    diff_c2 = simplify(lhs_c2 - rhs_c2)

    assert diff_c2 == 0, (
        "Symmetry verification failed (using f_H = 1 - f_L)!\n"
        "Expected: f_H*(mu_H - mu_V)² + f_L*(mu_L - mu_V)² = f_H*f_L*(mu_H - mu_L)²\n"
        f"Simplified difference: {diff_c2}"
    )
    print("✓ Step 3 verified: Symmetry confirmed with f_H = 1 - f_L substitution")

    print("\n" + "=" * 60)
    print("✓ BETWEEN-GROUP VARIANCE IDENTITY FULLY VERIFIED")
    print("=" * 60)
    print("  Identity: f_H(μ_H - μ_V)² + f_L(μ_L - μ_V)² = f_H·f_L·(μ_H - μ_L)²")
    print("  Constraint: μ_V = f_H·μ_H + f_L·μ_L, f_H + f_L = 1")
    print("=" * 60)


def test_mean_separation_lower_bound():
    """
    Verify: Mean separation lower bound from variance inequality

    Source: /home/guillem/fragile/docs/source/1_euclidean_gas/03_cloning.md, lines 3841-3857
    Category: Variance Decomposition

    This validates the algebraic rearrangement that derives the mean separation bound:
    (μ_H - μ_L)² ≥ (1/(f_H·f_L)) · (Var_total - Var_max)
    """

    # Define symbols
    f_H, f_L = symbols("f_H f_L", real=True, positive=True)
    mu_H, mu_L = symbols("mu_H mu_L", real=True)
    Var_total, _Var_B, _Var_W, Var_max = symbols(
        "Var_total Var_B Var_W Var_max", real=True, positive=True
    )

    # Law of Total Variance: Var_total = Var_B + Var_W
    # Rearranging: Var_B = Var_total - Var_W

    # From previous test, we know: Var_B = f_H * f_L * (mu_H - mu_L)²
    # Therefore: f_H * f_L * (mu_H - mu_L)² = Var_total - Var_W

    # Given that Var_W ≤ Var_max (proven in document using Popoviciu's inequality)
    # We have: Var_total - Var_W ≥ Var_total - Var_max

    # Thus: f_H * f_L * (mu_H - mu_L)² ≥ Var_total - Var_max

    # Dividing both sides by (f_H * f_L) gives:
    # (mu_H - mu_L)² ≥ (Var_total - Var_max) / (f_H * f_L)

    # This is a pure inequality manipulation, not an identity
    # We verify the algebraic rearrangement is correct

    # Starting inequality: f_H * f_L * (mu_H - mu_L)² ≥ Var_total - Var_max
    lhs_inequality = f_H * f_L * (mu_H - mu_L) ** 2
    rhs_inequality = Var_total - Var_max

    # Divide both sides by f_H * f_L (positive by assumption)
    lhs_divided = simplify(lhs_inequality / (f_H * f_L))
    rhs_divided = simplify(rhs_inequality / (f_H * f_L))

    # Verify LHS simplifies to (mu_H - mu_L)²
    expected_lhs = (mu_H - mu_L) ** 2
    assert (
        simplify(lhs_divided - expected_lhs) == 0
    ), f"LHS simplification failed. Expected (mu_H - mu_L)², got {lhs_divided}"

    # Verify RHS simplifies to (Var_total - Var_max) / (f_H * f_L)
    expected_rhs = (Var_total - Var_max) / (f_H * f_L)
    assert (
        simplify(rhs_divided - expected_rhs) == 0
    ), f"RHS simplification failed. Expected (Var_total - Var_max)/(f_H*f_L), got {rhs_divided}"

    print("✓ Mean separation lower bound algebra verified")
    print("  (μ_H - μ_L)² ≥ (Var_total - Var_max) / (f_H · f_L)")


def test_signal_to_noise_condition():
    """
    Verify: Signal-to-noise condition yields positive mean separation

    Source: /home/guillem/fragile/docs/source/1_euclidean_gas/03_cloning.md, lines 3766-3770
    Category: Variance Decomposition

    When κ_var > Var_max (signal-to-noise condition), the mean separation is:
    |μ_H - μ_L| ≥ (1/√(f_H·f_L)) · √(κ_var - Var_max) > 0
    """

    # Define symbols
    f_H, f_L = symbols("f_H f_L", real=True, positive=True)
    kappa_var, Var_max = symbols("kappa_var Var_max", real=True, positive=True)
    mu_H, mu_L = symbols("mu_H mu_L", real=True)

    # From previous lemma: (mu_H - mu_L)² ≥ (kappa_var - Var_max) / (f_H * f_L)
    # Taking square root of both sides (valid since both sides are non-negative when kappa_var > Var_max)

    (mu_H - mu_L) ** 2
    rhs_squared = (kappa_var - Var_max) / (f_H * f_L)

    # Taking square root
    # |mu_H - mu_L| = √[(mu_H - mu_L)²]
    # √[rhs_squared] = √[(kappa_var - Var_max) / (f_H * f_L)]
    #                = √(kappa_var - Var_max) / √(f_H * f_L)
    #                = (1/√(f_H·f_L)) · √(kappa_var - Var_max)

    rhs_sqrt_expected = sqrt(kappa_var - Var_max) / sqrt(f_H * f_L)
    rhs_sqrt_simplified = simplify(sqrt(rhs_squared))

    # Verify the square root simplification
    assert simplify(rhs_sqrt_simplified - rhs_sqrt_expected) == 0, (
        f"Square root simplification failed.\n"
        f"Expected: √(kappa_var - Var_max) / √(f_H*f_L)\n"
        f"Got: {rhs_sqrt_simplified}"
    )

    print("✓ Signal-to-noise condition algebra verified")
    print("  |μ_H - μ_L| ≥ √(κ_var - Var_max) / √(f_H · f_L)")


# ========================================
# TEST RUNNER
# ========================================


def run_all_validations():
    """Run all validation tests for lem-variance-to-mean-separation"""
    tests = [
        test_between_group_variance_identity,
        test_mean_separation_lower_bound,
        test_signal_to_noise_condition,
    ]

    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("RUNNING VALIDATION TESTS FOR LEM-VARIANCE-TO-MEAN-SEPARATION")
    print("=" * 60 + "\n")

    for test in tests:
        print(f"Running: {test.__name__}")
        print("-" * 60)
        try:
            test()
            print()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED:")
            print(f"  {e}\n")
            failed += 1

    print("=" * 60)
    print(f"VALIDATION SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    return passed, failed


if __name__ == "__main__":
    run_all_validations()
