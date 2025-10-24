from sympy import symbols, Eq, solve, simplify

def test_mean_decomposition_gap():
    """
    Verify: mu_V_k - mu_U = f_F * (mu_F - mu_U)
    Source: docs/source/1_euclidean_gas/03_cloning.md, lines 4786-4788
    """
    # 1. Define symbols
    # Fractions are positive, means are real numbers
    f_U, f_F = symbols('f_U f_F', positive=True)
    mu_U, mu_F = symbols('mu_U mu_F', real=True)

    # 2. Define the overall mean
    mu_V_k = f_U * mu_U + f_F * mu_F

    # 3. Define the two sides of the equation to be verified
    # LHS: mu_V_k - mu_U
    lhs = mu_V_k - mu_U

    # RHS: f_F * (mu_F - mu_U)
    rhs = f_F * (mu_F - mu_U)

    # 4. Define the constraint
    constraint = Eq(f_U + f_F, 1)

    # We can express f_U in terms of f_F from the constraint
    # f_U = 1 - f_F
    f_U_solution = solve(constraint, f_U)[0]

    # 5. Verify the identity by checking if LHS - RHS is zero after substitution
    # Substitute f_U in the expression (LHS - RHS)
    difference = lhs - rhs
    simplified_difference = difference.subs(f_U, f_U_solution)

    # The simplify function will perform the algebraic manipulation
    final_result = simplify(simplified_difference)

    # 6. Assert that the result is zero
    assert final_result == 0

    print("Validation successful: The algebraic identity holds true.")
    print(f"LHS expression: {lhs}")
    print(f"RHS expression: {rhs}")
    print(f"Constraint: {constraint}")
    print(f"LHS - RHS after substituting constraint and simplifying: {final_result}")

if __name__ == '__main__':
    # Execute the validation function
    test_mean_decomposition_gap()
