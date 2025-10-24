import sympy

def test_companion_fitness_gap_algebra():
    """
    Verify: mu_comp_i - V_k_i = (k/(k-1)) * (mu_V_k - V_k_i)
    Source: docs/source/1_euclidean_gas/03_cloning.md, lines 4770-4772
    """
    # 1. Define symbols
    # k: Number of alive walkers (positive integer, k >= 2)
    # mu_V_k: Mean fitness (real)
    # V_k_i: Individual walker fitness (real)
    k = sympy.Symbol('k', integer=True, positive=True)
    mu_V_k = sympy.Symbol('mu_V_k', real=True)
    V_k_i = sympy.Symbol('V_k_i', real=True)

    # 2. Define mu_comp_i based on the given identity
    # mu_comp_i = (1/(k-1)) * (k * mu_V_k - V_k_i)
    mu_comp_i = (k * mu_V_k - V_k_i) / (k - 1)

    # 3. Compute the Left-Hand Side (LHS) of the equation
    # LHS = mu_comp_i - V_k_i
    lhs = mu_comp_i - V_k_i

    # 4. Compute the Right-Hand Side (RHS) of the equation
    # RHS = (k/(k-1)) * (mu_V_k - V_k_i)
    rhs = (k / (k - 1)) * (mu_V_k - V_k_i)

    # 5. Verify that LHS - RHS simplifies to 0
    # This confirms the algebraic equivalence.
    difference = lhs - rhs
    simplified_difference = sympy.simplify(difference)

    print(f"Starting expression: {difference}")
    print(f"Simplified difference: {simplified_difference}")

    # The simplification should result in 0 if the identity is correct.
    # We add an assertion to make this a formal test.
    assert simplified_difference == 0, f"Validation failed! Expected 0, got {simplified_difference}"

    print("\nAlgebraic identity successfully verified.")
    print(f"LHS simplified: {sympy.simplify(lhs)}")
    print(f"RHS simplified: {sympy.simplify(rhs)}")

# Execute the validation function
if __name__ == "__main__":
    test_companion_fitness_gap_algebra()
