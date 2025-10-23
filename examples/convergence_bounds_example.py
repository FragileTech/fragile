"""Example usage of convergence_bounds module.

Demonstrates how to:
1. Compute convergence bounds for the Euclidean Gas
2. Validate parameter regimes for the Geometric Gas
3. Perform sensitivity analysis
4. Optimize parameters for balanced convergence

"""

import numpy as np
from fragile import convergence_bounds as cb


def example_1_euclidean_gas_analysis():
    """Example 1: Complete convergence analysis for Euclidean Gas."""
    print("=" * 80)
    print("Example 1: Euclidean Gas Convergence Analysis")
    print("=" * 80)

    # Define parameters
    gamma = 1.0  # Friction coefficient
    lambda_alg = 1.0  # Cloning rate
    sigma_v = 0.1  # Langevin noise
    tau = 0.01  # Time step
    lambda_min = 1.0  # Min eigenvalue of confining potential
    delta_f_boundary = 0.5  # Fitness drop at boundary

    # Compute component contraction rates
    kappa_x = cb.kappa_x(lambda_alg, tau)
    kappa_v = cb.kappa_v(gamma, tau)
    kappa_W = cb.kappa_W(gamma, lambda_min, c_hypo=0.1)
    kappa_b = cb.kappa_b(lambda_alg, delta_f_boundary)

    print(f"\nComponent contraction rates:")
    print(f"  Position (κ_x):    {kappa_x:.4f}")
    print(f"  Velocity (κ_v):    {kappa_v:.4f}")
    print(f"  Wasserstein (κ_W): {kappa_W:.4f}")
    print(f"  Boundary (κ_b):    {kappa_b:.4f}")

    # Total rate
    kappa_total = cb.kappa_total(kappa_x, kappa_v, kappa_W, kappa_b)
    print(f"\nTotal convergence rate (κ_total): {kappa_total:.4f}")

    # Identify bottleneck
    bottlenecks = cb.convergence_timescale_ratio(kappa_x, kappa_v, kappa_W, kappa_b)
    print(f"\nBottleneck component: {bottlenecks['bottleneck']}")
    print(f"Timescale ratios:")
    for component, ratio in bottlenecks.items():
        if component != "bottleneck":
            print(f"  {component:12s}: {ratio:.2f}x slower than fastest")

    # Mixing time
    epsilon = 0.01  # Target accuracy (1%)
    V_init = 10.0  # Initial Lyapunov value
    C_total = 1.0  # Equilibrium constant
    T_mix = cb.T_mix(epsilon, kappa_total, V_init, C_total)
    print(f"\nMixing time to reach {epsilon*100}% accuracy: {T_mix:.1f} steps")

    # Equilibrium variances
    d = 3
    var_x = cb.equilibrium_variance_x(sigma_v, tau, gamma, lambda_alg)
    var_v = cb.equilibrium_variance_v(d, sigma_v, gamma)
    print(f"\nEquilibrium variances:")
    print(f"  Position: {var_x:.6f}")
    print(f"  Velocity: {var_v:.4f}")


def example_2_geometric_gas_validation():
    """Example 2: Validate parameter regime for Geometric Gas."""
    print("\n" + "=" * 80)
    print("Example 2: Geometric Gas Parameter Regime Validation")
    print("=" * 80)

    # Ellipticity parameters
    epsilon_Sigma = 2.0  # Diffusion regularization
    H_max = 1.0  # Maximum Hessian eigenvalue

    print(f"\nEllipticity parameters:")
    print(f"  ε_Σ:   {epsilon_Sigma}")
    print(f"  H_max: {H_max}")

    # Validate ellipticity
    is_valid = cb.validate_ellipticity(epsilon_Sigma, H_max)
    print(f"  Uniform ellipticity valid: {is_valid}")

    # Compute bounds
    c_min_val = cb.c_min(epsilon_Sigma, H_max)
    c_max_val = cb.c_max(epsilon_Sigma, H_max)
    print(f"\nEllipticity bounds:")
    print(f"  c_min: {c_min_val:.4f}")
    print(f"  c_max: {c_max_val:.4f}")

    # Condition number
    kappa_geom = cb.condition_number_geometry(c_min_val, c_max_val)
    print(f"  Geometric condition number: {kappa_geom:.2f}")

    # Critical adaptive force threshold
    F_adapt_max = 10.0  # Maximum adaptive force
    epsilon_F_star = cb.epsilon_F_star(1.0, c_min_val, F_adapt_max)
    print(f"\nAdaptive force regime:")
    print(f"  F_adapt_max:  {F_adapt_max}")
    print(f"  ε_F* (threshold): {epsilon_F_star:.4f}")

    # Test different ε_F values
    test_epsilon_F = [0.01, 0.05, 0.1, 0.2]
    for eps_F in test_epsilon_F:
        valid = cb.validate_hypocoercivity(eps_F, epsilon_F_star, nu=0.1)
        status = "✓" if valid else "✗"
        print(f"  ε_F = {eps_F:.2f}: {status}")

    # LSI constant
    gamma = 1.0
    kappa_conf = 1.0
    kappa_W = 0.1
    C_LSI = cb.C_LSI_geometric(1.0, c_min_val, c_max_val, gamma, kappa_conf, kappa_W)
    print(f"\nN-uniform LSI constant: {C_LSI:.2f}")


def example_3_sensitivity_analysis():
    """Example 3: Sensitivity analysis and parameter optimization."""
    print("\n" + "=" * 80)
    print("Example 3: Sensitivity Analysis")
    print("=" * 80)

    # Parameter dictionary
    params = {
        "gamma": 1.0,
        "lambda_alg": 1.0,
        "sigma_v": 0.1,
        "tau": 0.01,
        "lambda_min": 1.0,
        "delta_f_boundary": 0.5,
    }

    print("\nBase parameters:")
    for k, v in params.items():
        print(f"  {k:20s}: {v}")

    # Rate sensitivity matrix
    M_kappa = cb.rate_sensitivity_matrix(params)
    print(f"\nRate sensitivity matrix shape: {M_kappa.shape}")
    print("  (4 rates × 6 parameters)")

    # Principal coupling modes
    modes = cb.principal_coupling_modes(M_kappa, k=3)
    print(f"\nTop 3 principal coupling modes:")
    print(f"  Singular values: {modes['singular_values']}")
    print("  (Measure strength of parameter coupling)")

    # Condition number
    kappa_params = cb.condition_number_parameters(M_kappa)
    print(f"\nParameter condition number: {kappa_params:.2f}")
    print("  (Measures robustness to parameter errors)")


def example_4_optimal_parameters():
    """Example 4: Compute optimal balanced parameters."""
    print("\n" + "=" * 80)
    print("Example 4: Optimal Parameter Selection")
    print("=" * 80)

    # Problem characteristics
    lambda_min = 1.0  # Landscape minimum curvature
    lambda_max = 10.0  # Landscape maximum curvature
    d = 3  # Dimension
    V_target = 1.0  # Target exploration variance

    print(f"\nProblem characteristics:")
    print(f"  Dimension:      {d}")
    print(f"  λ_min:          {lambda_min}")
    print(f"  λ_max:          {lambda_max}")
    print(f"  Target variance: {V_target}")

    # Compute optimal balanced parameters
    optimal = cb.balanced_parameters_closed_form(lambda_min, lambda_max, d, V_target)

    print(f"\nOptimal balanced parameters:")
    for param, value in optimal.items():
        print(f"  {param:12s}: {value:.4f}")

    print("\n✓ These parameters eliminate convergence bottlenecks")
    print("  by balancing all component contraction rates.")


def example_5_pareto_frontier():
    """Example 5: Rate-variance trade-off (Pareto frontier)."""
    print("\n" + "=" * 80)
    print("Example 5: Rate-Variance Pareto Frontier")
    print("=" * 80)

    # Compute Pareto frontier
    kappa_range = (0.1, 1.0)
    C_range = (0.5, 5.0)
    frontier = cb.pareto_frontier_rate_variance(kappa_range, C_range, n_points=10)

    print("\nPareto frontier (rate vs. equilibrium variance):")
    print("  κ_total  |  Var_eq")
    print("  " + "-" * 20)
    for kappa, var in frontier:
        print(f"  {kappa:7.3f}  |  {var:7.3f}")

    print("\n✓ Frontier shows achievable trade-offs:")
    print("  - Higher rate → faster convergence")
    print("  - Lower variance → more focused exploration")
    print("  - Cannot optimize both simultaneously!")


if __name__ == "__main__":
    example_1_euclidean_gas_analysis()
    example_2_geometric_gas_validation()
    example_3_sensitivity_analysis()
    example_4_optimal_parameters()
    example_5_pareto_frontier()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
