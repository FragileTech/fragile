import sympy

def test_within_group_variance_bound():
    """
    Verify: Var_W <= Var_max simplifies to Var_max using constraint
    Source: docs/source/1_euclidean_gas/03_cloning.md, lines 3834-3836
    """
    # Define symbols: f_H, f_L (positive), Var_max (positive)
    f_H, f_L = sympy.symbols('f_H f_L', positive=True)
    Var_max = sympy.symbols('Var_max', positive=True)

    # Start with: Var_W_upper = f_H * Var_max + f_L * Var_max
    Var_W_upper = f_H * Var_max + f_L * Var_max

    # Factor out Var_max and apply constraint f_H + f_L = 1
    # We can substitute f_L with 1 - f_H
    simplified_expr = sympy.simplify(Var_W_upper.subs(f_L, 1 - f_H))

    # Verify result equals Var_max
    assert simplified_expr == Var_max, f"Simplification failed: Expected {Var_max}, got {simplified_expr}"

    # Alternative approach: Factoring
    factored_expr = sympy.factor(Var_W_upper)
    final_result = factored_expr.subs(f_H + f_L, 1)

    # Verify the factored result as well
    assert final_result == Var_max, f"Factoring failed: Expected {Var_max}, got {final_result}"

    print("Validation successful.")
    print(f"Original expression: {Var_W_upper}")
    print(f"Factored form: {factored_expr}")
    print(f"Final result after applying constraint (f_H + f_L = 1): {final_result}")

if __name__ == "__main__":
    test_within_group_variance_bound()
