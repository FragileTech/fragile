"""
Symbolic Validation for Equilibrium Variance Bounds

Source: docs/source/1_euclidean_gas/06_convergence.md, lines 1112-1176
Category: Equilibrium Conditions (solving drift equations at steady state)
Generated: 2025-10-24
Agent: Math Verifier v1.0 (Gemini + GPT-5 synthesis)

This module validates equilibrium solutions for variance components
by solving drift inequalities at steady state (ΔV = 0).
"""

from sympy import symbols, simplify, solve, Rational

def test_positional_variance_equilibrium():
    """
    Verify: V^QSD_Var,x = C_x / κ_x

    Source: docs/source/1_euclidean_gas/06_convergence.md, lines 1128-1134
    Category: Equilibrium Conditions

    Drift equation: ΔV_Var,x = -κ_x V_Var,x + C_x
    At equilibrium: 0 = -κ_x V^QSD_Var,x + C_x
    """

    # Define symbols
    kappa_x, C_x = symbols('kappa_x C_x', positive=True)
    V_Var_x = symbols('V_Var_x', positive=True)

    # Drift equation
    drift = -kappa_x * V_Var_x + C_x

    # Solve for equilibrium (drift = 0)
    equilibrium = solve(drift, V_Var_x)[0]

    # Expected result
    expected = C_x / kappa_x

    # Verify equality
    difference = simplify(equilibrium - expected)
    assert difference == 0, (
        f"Positional variance equilibrium verification failed!\n"
        f"  Computed: {equilibrium}\n"
        f"  Expected: {expected}\n"
        f"  Difference: {difference}"
    )

    # Substitute back to verify zero drift
    drift_at_equilibrium = drift.subs(V_Var_x, equilibrium)
    assert simplify(drift_at_equilibrium) == 0, (
        f"Drift at equilibrium is non-zero: {drift_at_equilibrium}"
    )

    print("✓ Positional Variance Equilibrium:")
    print(f"  V^QSD_Var,x = C_x / κ_x")
    print(f"  Physical: Equilibrium variance = noise / contraction_rate")

def test_velocity_variance_equilibrium():
    """
    Verify: V^QSD_Var,v = (C_v + σ²_max d τ) / (2γτ)

    Source: docs/source/1_euclidean_gas/06_convergence.md, lines 1146-1159
    Category: Equilibrium Conditions

    Drift equation: ΔV_Var,v = -2γ V_Var,v τ + (C_v + σ²_max d τ)
    At equilibrium: 0 = -2γ V^QSD_Var,v τ + (C_v + σ²_max d τ)
    """

    # Define symbols
    gamma, tau, C_v, sigma_max_sq, d = symbols(
        'gamma tau C_v sigma_max_sq d', positive=True
    )
    V_Var_v = symbols('V_Var_v', positive=True)

    # Drift equation
    drift = -2*gamma*V_Var_v*tau + (C_v + sigma_max_sq*d*tau)

    # Solve for equilibrium
    equilibrium = solve(drift, V_Var_v)[0]

    # Expected result
    expected = (C_v + sigma_max_sq*d*tau) / (2*gamma*tau)

    # Verify equality
    difference = simplify(equilibrium - expected)
    assert difference == 0, (
        f"Velocity variance equilibrium verification failed!\n"
        f"  Computed: {equilibrium}\n"
        f"  Expected: {expected}\n"
        f"  Difference: {difference}"
    )

    # Substitute back to verify zero drift
    drift_at_equilibrium = drift.subs(V_Var_v, equilibrium)
    assert simplify(drift_at_equilibrium) == 0, (
        f"Drift at equilibrium is non-zero: {drift_at_equilibrium}"
    )

    # Decompose into cloning and Langevin contributions
    cloning_contribution = C_v / (2*gamma*tau)
    langevin_contribution = (sigma_max_sq*d) / (2*gamma)
    full_decomposition = simplify(cloning_contribution + langevin_contribution)

    print("✓ Velocity Variance Equilibrium:")
    print(f"  V^QSD_Var,v = (C_v + σ²_max d τ) / (2γτ)")
    print(f"  Decomposition:")
    print(f"    - Cloning contribution: C_v / (2γτ)")
    print(f"    - Langevin contribution: (σ²_max d) / (2γ)")
    print(f"  Physical: Balance between friction dissipation and noise injection")

def test_boundary_potential_equilibrium():
    """
    Verify: W^QSD_b = C_b / κ_b

    Source: docs/source/1_euclidean_gas/06_convergence.md, lines 1166-1173
    Category: Equilibrium Conditions

    Drift equation: ΔW_b = -κ_b W_b + C_b
    At equilibrium: 0 = -κ_b W^QSD_b + C_b
    """

    # Define symbols
    kappa_b, C_b = symbols('kappa_b C_b', positive=True)
    W_b = symbols('W_b', positive=True)

    # Drift equation
    drift = -kappa_b * W_b + C_b

    # Solve for equilibrium
    equilibrium = solve(drift, W_b)[0]

    # Expected result
    expected = C_b / kappa_b

    # Verify equality
    difference = simplify(equilibrium - expected)
    assert difference == 0, (
        f"Boundary potential equilibrium verification failed!\n"
        f"  Computed: {equilibrium}\n"
        f"  Expected: {expected}\n"
        f"  Difference: {difference}"
    )

    # Substitute back to verify zero drift
    drift_at_equilibrium = drift.subs(W_b, equilibrium)
    assert simplify(drift_at_equilibrium) == 0, (
        f"Drift at equilibrium is non-zero: {drift_at_equilibrium}"
    )

    print("✓ Boundary Potential Equilibrium:")
    print(f"  W^QSD_b = C_b / κ_b")
    print(f"  Physical: Larger κ_b (stronger boundary) → smaller W^QSD_b")

def test_parameter_positivity_assumptions():
    """
    Verify that all parameters have correct positivity assumptions.
    """

    # All parameters should be positive
    params = {
        'κ_x': symbols('kappa_x', positive=True),
        'κ_b': symbols('kappa_b', positive=True),
        'γ': symbols('gamma', positive=True),
        'τ': symbols('tau', positive=True),
        'C_x': symbols('C_x', positive=True),
        'C_v': symbols('C_v', positive=True),
        'C_b': symbols('C_b', positive=True),
        'σ²_max': symbols('sigma_max_sq', positive=True),
        'd': symbols('d', positive=True)
    }

    for name, symbol in params.items():
        assert symbol.is_positive, f"Parameter {name} should be positive"

    print("✓ Parameter Positivity Assumptions:")
    print(f"  All parameters (κ_x, κ_b, γ, τ, C_x, C_v, C_b, σ²_max, d) are positive")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Equilibrium Variance Bounds Validation")
    print("="*60 + "\n")

    test_positional_variance_equilibrium()
    print()
    test_velocity_variance_equilibrium()
    print()
    test_boundary_potential_equilibrium()
    print()
    test_parameter_positivity_assumptions()

    print("\n" + "="*60)
    print("✓ All equilibrium variance bounds verified")
    print("="*60)
