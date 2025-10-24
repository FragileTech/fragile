from sympy import symbols, simplify, expand

def test_between_group_variance_identity():
    """
    Verify: Between-group variance identity

    Var_B = f_H f_L (mu_H - mu_L)^2

    Source: docs/source/1_euclidean_gas/03_cloning.md, lines 3810-3822
    Category: Variance Decomposition
    """

    # 1. Define symbols with appropriate assumptions from the framework
    f_H, f_L = symbols('f_H f_L', positive=True, real=True)
    mu_H, mu_L, mu_V = symbols('mu_H mu_L mu_V', real=True)

    # 2. Define the Left-Hand Side (LHS) from the definition of between-group variance
    # Var_B = f_H(mu_H - mu_V)^2 + f_L(mu_L - mu_V)^2
    lhs = f_H * (mu_H - mu_V)**2 + f_L * (mu_L - mu_V)**2

    # 3. Substitute the definition of the total mean (mu_V) into the LHS
    # mu_V = f_H*mu_H + f_L*mu_L
    lhs_substituted = lhs.subs(mu_V, f_H * mu_H + f_L * mu_L)

    # 4. Define the Right-Hand Side (RHS) of the identity to be verified
    rhs = f_H * f_L * (mu_H - mu_L)**2

    # 5. Create the expression representing the difference between LHS and RHS
    difference = lhs_substituted - rhs

    # 6. Apply the constraint f_H + f_L = 1 by substituting f_L = 1 - f_H.
    # The `simplify` function will perform the algebraic manipulation to confirm
    # that the difference is zero, thus proving the identity.
    simplified_difference = simplify(difference.subs(f_L, 1 - f_H))

    # 7. Assert that the identity holds true. If not, provide a descriptive error.
    assert simplified_difference == 0, (
        f"Verification failed. The difference between LHS and RHS should be 0, but it is {simplified_difference}.\n"
        f"LHS (substituted and expanded): {expand(lhs_substituted.subs(f_L, 1 - f_H))}\n"
        f"RHS (substituted and expanded): {expand(rhs.subs(f_L, 1 - f_H))}"
    )

    # 8. Print confirmation message on success
    print("✓ Between-group variance identity verified successfully.")
    print("Identity: Var_B = f_H * f_L * (mu_H - mu_L)**2")

if __name__ == "__main__":
    test_between_group_variance_identity()
