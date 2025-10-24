import sympy

def test_mean_deviations_from_total():
    """
    Verify two mean deviation identities
    Source: docs/source/1_euclidean_gas/03_cloning.md, lines 3802-3804
    """
    # 1. Define symbols
    f_H, f_L = sympy.symbols('f_H f_L', positive=True)
    mu_H, mu_L = sympy.symbols('mu_H mu_L', real=True)

    # 2. Define the total mean
    mu_V_expr = f_H * mu_H + f_L * mu_L

    # --- Verification of Claim 1 ---
    # Claim: mu_H - mu_V = f_L * (mu_H - mu_L)
    # We verify that (mu_H - mu_V) - f_L * (mu_H - mu_L) simplifies to 0

    lhs_1 = mu_H - mu_V_expr
    rhs_1 = f_L * (mu_H - mu_L)

    # Substitute the constraint f_H = 1 - f_L to allow simplification
    difference_1 = (lhs_1 - rhs_1).subs(f_H, 1 - f_L)
    simplified_1 = sympy.simplify(difference_1)

    assert simplified_1 == 0, f"Claim 1 failed: expected 0, got {simplified_1}"

    # --- Verification of Claim 2 ---
    # Claim: mu_L - mu_V = -f_H * (mu_H - mu_L)
    # We verify that (mu_L - mu_V) + f_H * (mu_H - mu_L) simplifies to 0

    lhs_2 = mu_L - mu_V_expr
    rhs_2 = -f_H * (mu_H - mu_L)

    # Substitute the constraint f_L = 1 - f_H to allow simplification
    difference_2 = (lhs_2 - rhs_2).subs(f_L, 1 - f_H)
    simplified_2 = sympy.simplify(difference_2)

    assert simplified_2 == 0, f"Claim 2 failed: expected 0, got {simplified_2}"

if __name__ == '__main__':
    try:
        test_mean_deviations_from_total()
        print("Both algebraic claims successfully verified with SymPy.")
    except AssertionError as e:
        print(f"Verification failed: {e}")
