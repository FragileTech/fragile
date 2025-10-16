"""
Gas Parameter Optimization and Convergence Analysis

This module implements all formulas and parameter analysis from
clean_build/source/04_convergence.md Chapter 8 and Section 9.10.

It provides:
1. Convergence rate computation (κ_x, κ_v, κ_W, κ_b, κ_total)
2. Equilibrium constant computation (C_x, C_v, C_W, C_b, C_total)
3. Optimal parameter selection algorithms
4. Parameter sensitivity analysis
5. Mixing time estimation
6. Empirical rate estimation from trajectories

Mathematical References:
- Section 8: Explicit Parameter Dependence and Convergence Rates
- Section 9.10: Rate-Space Optimization: Computing Optimal Parameters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from torch import Tensor


@dataclass
class LandscapeParams:
    """Landscape characterization parameters.

    Attributes:
        lambda_min: Smallest eigenvalue of Hessian ∇²U(x) in relevant region
        lambda_max: Largest eigenvalue of Hessian ∇²U(x)
        d: Spatial dimension
        f_typical: Typical fitness scale
        Delta_f_boundary: Fitness gap at boundary (interior - boundary)
    """

    lambda_min: float
    lambda_max: float
    d: int
    f_typical: float = 1.0
    Delta_f_boundary: float = 10.0


@dataclass
class GasParams:
    """Euclidean Gas algorithm parameters.

    Mathematical notation from 04_convergence.md:
    - τ (tau): Timestep
    - γ (gamma): Friction coefficient
    - σ_v (sigma_v): Thermal velocity fluctuation intensity
    - λ (lambda_clone): Cloning rate
    - N: Number of walkers
    - σ_x (sigma_x): Position jitter scale
    - λ_alg (lambda_alg): Velocity weight in algorithmic distance
    - α_rest (alpha_rest): Restitution coefficient
    - d_safe: Safe Harbor distance
    - κ_wall (kappa_wall): Boundary stiffness
    """

    tau: float
    gamma: float
    sigma_v: float
    lambda_clone: float
    N: int
    sigma_x: float
    lambda_alg: float
    alpha_rest: float
    d_safe: float
    kappa_wall: float


@dataclass
class ConvergenceRates:
    """Convergence rates for all Lyapunov components.

    From Theorem 8.5 (Total Convergence Rate):
    κ_total = min(κ_x, κ_v, κ_W, κ_b) * (1 - ε_coupling)
    """

    kappa_x: float  # Position variance contraction rate
    kappa_v: float  # Velocity variance dissipation rate
    kappa_W: float  # Wasserstein contraction rate
    kappa_b: float  # Boundary contraction rate
    kappa_total: float  # Total geometric convergence rate
    epsilon_coupling: float  # Expansion-to-contraction ratio


@dataclass
class EquilibriumConstants:
    """Equilibrium constants for all Lyapunov components.

    From Section 8: C_i determines equilibrium variance V_i^eq = C_i / κ_i
    """

    C_x: float  # Position variance source
    C_v: float  # Velocity variance source
    C_W: float  # Wasserstein source
    C_b: float  # Boundary source
    C_total: float  # Total source term


# ==============================================================================
# Rate Computation (Chapter 8)
# ==============================================================================


def compute_velocity_rate(params: GasParams) -> float:
    """Compute velocity variance dissipation rate κ_v.

    From Proposition 8.1 (Velocity Dissipation Rate):
    κ_v = 2γ - O(τ)

    Args:
        params: Gas parameters

    Returns:
        Velocity dissipation rate (1/time)
    """
    tau_correction = 0.1 * params.tau  # O(τ) correction factor
    return 2.0 * params.gamma * (1.0 - tau_correction)


def estimate_fitness_correlation(lambda_alg: float, epsilon_c: float) -> float:
    """Estimate fitness-variance correlation coefficient c_fit.

    From cloning theory (03_cloning.md), typical values:
    c_fit ≈ 0.5 - 0.8 for well-separated landscapes

    Args:
        lambda_alg: Velocity weight in algorithmic distance
        epsilon_c: Position jitter scale

    Returns:
        Correlation coefficient (dimensionless)
    """
    # Heuristic: stronger pairing (smaller λ_alg) → better correlation
    # Typical range: 0.5 to 0.8
    base_correlation = 0.65
    lambda_effect = np.exp(-0.5 * lambda_alg)  # Decreases with λ_alg
    return base_correlation * (0.8 + 0.2 * lambda_effect)


def compute_position_rate(params: GasParams, landscape: LandscapeParams) -> float:
    """Compute position variance contraction rate κ_x.

    From Proposition 8.2 (Positional Contraction Rate):
    κ_x = λ · Cov(f_i, ||x_i - x̄||²) / E[||x_i - x̄||²] + O(τ)

    Args:
        params: Gas parameters
        landscape: Landscape characterization

    Returns:
        Position contraction rate (1/time)
    """
    c_fit = estimate_fitness_correlation(params.lambda_alg, params.sigma_x)
    tau_correction = 0.1 * params.tau
    return params.lambda_clone * c_fit * (1.0 - tau_correction)


def compute_wasserstein_rate(
    params: GasParams, landscape: LandscapeParams, c_hypo: float = 0.5
) -> float:
    """Compute Wasserstein contraction rate κ_W.

    From Proposition 8.3 (Wasserstein Contraction Rate):
    κ_W = c_hypo² · γ / (1 + γ/λ_min)

    Args:
        params: Gas parameters
        landscape: Landscape characterization
        c_hypo: Hypocoercivity constant (typically 0.1 - 1.0)

    Returns:
        Wasserstein contraction rate (1/time)
    """
    return (c_hypo**2 * params.gamma) / (1.0 + params.gamma / landscape.lambda_min)


def compute_boundary_rate(params: GasParams, landscape: LandscapeParams) -> float:
    """Compute boundary contraction rate κ_b.

    From Proposition 8.4 (Boundary Contraction Rate):
    κ_b = min(λ · Δf_boundary/f_typical, κ_wall + γ)

    Args:
        params: Gas parameters
        landscape: Landscape characterization

    Returns:
        Boundary contraction rate (1/time)
    """
    kappa_clone = params.lambda_clone * landscape.Delta_f_boundary / landscape.f_typical
    kappa_kinetic = params.kappa_wall + params.gamma
    return min(kappa_clone, kappa_kinetic)


def compute_convergence_rates(
    params: GasParams, landscape: LandscapeParams, c_hypo: float = 0.5
) -> ConvergenceRates:
    """Compute all convergence rates and total rate.

    From Theorem 8.5 (Total Convergence Rate):
    κ_total = min(κ_x, κ_v, κ_W, κ_b) · (1 - ε_coupling)

    Args:
        params: Gas parameters
        landscape: Landscape characterization
        c_hypo: Hypocoercivity constant

    Returns:
        All convergence rates
    """
    kappa_x = compute_position_rate(params, landscape)
    kappa_v = compute_velocity_rate(params)
    kappa_W = compute_wasserstein_rate(params, landscape, c_hypo)
    kappa_b = compute_boundary_rate(params, landscape)

    # Coupling ratio (expansion-to-contraction)
    # Heuristic: ε_coupling ≈ O(τ) for well-tuned parameters
    epsilon_coupling = 0.05 * params.tau / 0.01  # Normalized to τ = 0.01
    epsilon_coupling = min(epsilon_coupling, 0.5)  # Cap at 50%

    kappa_total = min(kappa_x, kappa_v, kappa_W, kappa_b) * (1.0 - epsilon_coupling)

    return ConvergenceRates(
        kappa_x=kappa_x,
        kappa_v=kappa_v,
        kappa_W=kappa_W,
        kappa_b=kappa_b,
        kappa_total=kappa_total,
        epsilon_coupling=epsilon_coupling,
    )


# ==============================================================================
# Equilibrium Constants (Chapter 8)
# ==============================================================================


def compute_velocity_equilibrium(params: GasParams, landscape: LandscapeParams) -> float:
    """Compute velocity equilibrium constant C_v.

    From Proposition 8.1:
    C_v' = d·σ_v² / γ

    Args:
        params: Gas parameters
        landscape: Landscape characterization

    Returns:
        Velocity source term
    """
    return landscape.d * params.sigma_v**2 / params.gamma


def compute_position_equilibrium(params: GasParams, landscape: LandscapeParams) -> float:
    """Compute position equilibrium constant C_x.

    From Proposition 8.2:
    C_x = O(σ_v² τ² / (γ·λ))

    Args:
        params: Gas parameters
        landscape: Landscape characterization

    Returns:
        Position source term
    """
    return (params.sigma_v**2 * params.tau**2) / (params.gamma * params.lambda_clone)


def compute_wasserstein_equilibrium(params: GasParams, landscape: LandscapeParams) -> float:
    """Compute Wasserstein equilibrium constant C_W.

    From Proposition 8.3:
    C_W' = O(σ_v² τ / N^(1/d))

    Args:
        params: Gas parameters
        landscape: Landscape characterization

    Returns:
        Wasserstein source term
    """
    N_factor = params.N ** (1.0 / landscape.d)
    return (params.sigma_v**2 * params.tau) / N_factor


def compute_boundary_equilibrium(params: GasParams, landscape: LandscapeParams) -> float:
    """Compute boundary equilibrium constant C_b.

    From Proposition 8.4:
    C_b = O(σ_v² τ / d_safe²)

    Args:
        params: Gas parameters
        landscape: Landscape characterization

    Returns:
        Boundary source term
    """
    return (params.sigma_v**2 * params.tau) / (params.d_safe**2)


def compute_equilibrium_constants(
    params: GasParams, landscape: LandscapeParams
) -> EquilibriumConstants:
    """Compute all equilibrium constants.

    Args:
        params: Gas parameters
        landscape: Landscape characterization

    Returns:
        All equilibrium constants
    """
    C_x = compute_position_equilibrium(params, landscape)
    C_v = compute_velocity_equilibrium(params, landscape)
    C_W = compute_wasserstein_equilibrium(params, landscape)
    C_b = compute_boundary_equilibrium(params, landscape)

    # Total source term (weighted sum)
    # Weights chosen to ensure synergy (Section 6)
    alpha_v = 1.0
    alpha_W = 1.0
    alpha_b = 1.0
    C_total = C_x + alpha_v * C_v + alpha_W * C_W + alpha_b * C_b

    return EquilibriumConstants(C_x=C_x, C_v=C_v, C_W=C_W, C_b=C_b, C_total=C_total)


# ==============================================================================
# Mixing Time Estimation (Section 8.6)
# ==============================================================================


def compute_mixing_time(
    params: GasParams, landscape: LandscapeParams, epsilon: float = 0.01, V_init: float = 1.0
) -> dict[str, float]:
    """Compute mixing time to reach ε-proximity to equilibrium.

    From Proposition 8.6 (Mixing Time):
    T_mix(ε) = (1/κ_total) · ln(V_init / (ε · C_total))

    Args:
        params: Gas parameters
        landscape: Landscape characterization
        epsilon: Target relative error
        V_init: Initial Lyapunov value

    Returns:
        Dictionary with:
            - T_mix_time: Mixing time (continuous time units)
            - T_mix_steps: Mixing time (number of steps)
            - kappa_total: Total convergence rate
            - V_eq: Equilibrium Lyapunov value
    """
    rates = compute_convergence_rates(params, landscape)
    constants = compute_equilibrium_constants(params, landscape)

    V_eq = constants.C_total / rates.kappa_total

    # Mixing time formula
    if V_init <= epsilon * V_eq:
        # Already at equilibrium
        T_mix_time = 0.0
    else:
        log_arg = (V_init * rates.kappa_total) / (epsilon * constants.C_total)
        T_mix_time = np.log(log_arg) / rates.kappa_total

    T_mix_steps = int(np.ceil(T_mix_time / params.tau))

    return {
        "T_mix_time": T_mix_time,
        "T_mix_steps": T_mix_steps,
        "kappa_total": rates.kappa_total,
        "V_eq": V_eq,
        "rates": rates,
        "constants": constants,
    }


# ==============================================================================
# Optimal Parameter Selection (Section 9.10.1)
# ==============================================================================


def compute_optimal_parameters(
    landscape: LandscapeParams, V_target: float = 0.1, gamma_budget: float | None = None
) -> GasParams:
    """Compute optimal parameters using closed-form solution.

    From Theorem 9.10.1 (Closed-Form Balanced Optimum):

    Step 1: γ* = λ_min (maximize κ_W)
    Step 2: λ* = 2γ*/c_fit ≈ 3λ_min (balanced)
    Step 3: τ* = min(0.5/γ*, 1/√λ_max, 0.01)
    Step 4: σ_v* = √(γ* · V_target)
    Step 5: σ_x* = σ_v* τ* / √γ*
    Step 6: λ_alg* = σ_x*² / σ_v*²
    Step 7: α_rest* = √(2 - 2γ_budget/γ*)
    Step 8: d_safe* = 3√V_target, κ_wall* = 10λ_min

    Args:
        landscape: Landscape characterization
        V_target: Target exploration width (position variance)
        gamma_budget: Available friction (default 1.5 * gamma*)

    Returns:
        Optimal gas parameters
    """
    # Step 1: Friction from landscape
    gamma_opt = landscape.lambda_min

    # Step 2: Cloning rate from balance
    c_fit_estimate = 0.65
    lambda_opt = 2.0 * gamma_opt / c_fit_estimate  # ≈ 3 * lambda_min

    # Step 3: Timestep from stability
    tau_opt = min(
        0.5 / gamma_opt,  # Friction stability
        1.0 / np.sqrt(landscape.lambda_max),  # Symplectic stability
        0.01,  # Practical upper bound
    )

    # Step 4: Exploration noise from target
    sigma_v_opt = np.sqrt(gamma_opt * V_target)

    # Step 5: Position jitter from crossover
    sigma_x_opt = sigma_v_opt * tau_opt / np.sqrt(gamma_opt)

    # Step 6: Geometric parameters
    lambda_alg_opt = sigma_x_opt**2 / sigma_v_opt**2

    # Step 7: Restitution coefficient
    if gamma_budget is None:
        gamma_budget = 1.5 * gamma_opt

    restitution_arg = 2.0 - 2.0 * gamma_budget / gamma_opt
    if restitution_arg >= 0:
        alpha_rest_opt = np.sqrt(restitution_arg)
    else:
        alpha_rest_opt = 0.0  # Fully inelastic

    # Step 8: Boundary parameters
    d_safe_opt = 3.0 * np.sqrt(V_target)
    kappa_wall_opt = 10.0 * landscape.lambda_min

    # Determine N from Wasserstein accuracy requirement
    # Heuristic: N ≥ (10 * d)^d for good statistical behavior
    N_opt = max(100, int((10 * landscape.d) ** (landscape.d / 2)))

    return GasParams(
        tau=tau_opt,
        gamma=gamma_opt,
        sigma_v=sigma_v_opt,
        lambda_clone=lambda_opt,
        N=N_opt,
        sigma_x=sigma_x_opt,
        lambda_alg=lambda_alg_opt,
        alpha_rest=alpha_rest_opt,
        d_safe=d_safe_opt,
        kappa_wall=kappa_wall_opt,
    )


# ==============================================================================
# Parameter Sensitivity Matrix (Section 9)
# ==============================================================================


def compute_sensitivity_matrix(
    params: GasParams, landscape: LandscapeParams, delta: float = 1e-4
) -> NDArray[np.float64]:
    """Compute sensitivity matrix M_κ: ∂κ_i / ∂log(P_j).

    From Section 9: Sensitivities show how each rate responds to parameter changes.

    Returns 4×10 matrix where:
    - Rows: [κ_x, κ_v, κ_W, κ_b]
    - Cols: [tau, gamma, sigma_v, lambda_clone, N, sigma_x, lambda_alg,
             alpha_rest, d_safe, kappa_wall]

    Args:
        params: Current parameters
        landscape: Landscape characterization
        delta: Finite difference step (relative)

    Returns:
        Sensitivity matrix (4, 10)
    """
    param_names = [
        "tau",
        "gamma",
        "sigma_v",
        "lambda_clone",
        "N",
        "sigma_x",
        "lambda_alg",
        "alpha_rest",
        "d_safe",
        "kappa_wall",
    ]

    # Baseline rates
    rates_base = compute_convergence_rates(params, landscape)
    kappa_base = np.array([
        rates_base.kappa_x,
        rates_base.kappa_v,
        rates_base.kappa_W,
        rates_base.kappa_b,
    ])

    M_kappa = np.zeros((4, len(param_names)))

    for j, param_name in enumerate(param_names):
        # Perturb parameter by delta (multiplicative)
        params_pert = GasParams(**vars(params))
        old_value = getattr(params_pert, param_name)
        setattr(params_pert, param_name, old_value * (1.0 + delta))

        # Compute perturbed rates
        rates_pert = compute_convergence_rates(params_pert, landscape)
        kappa_pert = np.array([
            rates_pert.kappa_x,
            rates_pert.kappa_v,
            rates_pert.kappa_W,
            rates_pert.kappa_b,
        ])

        # Sensitivity: ∂log(κ_i) / ∂log(P_j) ≈ Δκ_i / (κ_i · δ)
        M_kappa[:, j] = (kappa_pert - kappa_base) / (kappa_base * delta)

    return M_kappa


# ==============================================================================
# Trajectory Analysis (Section 9.10.4)
# ==============================================================================


def estimate_rates_from_trajectory(
    trajectory_data: dict[str, Tensor | NDArray], tau: float
) -> ConvergenceRates:
    """Estimate empirical convergence rates from trajectory.

    From Algorithm 9.10.4 (Adaptive Parameter Tuning):
    Fit exponential decay: V_i(t) ≈ C_i/κ_i + (V_i(0) - C_i/κ_i) * exp(-κ_i * t)

    Args:
        trajectory_data: Dictionary with keys:
            - 'V_Var_x': Position variance over time [T]
            - 'V_Var_v': Velocity variance over time [T]
            - 'V_W': Wasserstein distance over time [T] (optional)
            - 'W_b': Boundary potential over time [T] (optional)
        tau: Timestep size

    Returns:
        Estimated convergence rates
    """

    def fit_exponential_rate(V: NDArray | Tensor, times: NDArray) -> float:
        """Fit V(t) = C + A * exp(-κ*t) and extract κ."""
        if isinstance(V, Tensor):
            V = V.cpu().numpy()

        # Skip if not enough data
        if len(V) < 10:
            return 0.0

        # Use log-linear regression on transient part
        # Assume equilibrium is last 20% of trajectory
        idx_eq_start = int(0.8 * len(V))
        V_eq_estimate = np.mean(V[idx_eq_start:])

        # Transient: V(t) - V_eq ≈ A * exp(-κ*t)
        V_transient = V[:idx_eq_start] - V_eq_estimate
        V_transient = np.maximum(V_transient, 1e-10)  # Avoid log(0)

        log_V_transient = np.log(V_transient)
        times_transient = times[:idx_eq_start]

        # Linear fit: log(V_transient) = log(A) - κ*t
        if len(times_transient) > 2:
            poly = np.polyfit(times_transient, log_V_transient, 1)
            kappa = -poly[0]  # Negative slope
            return max(kappa, 0.0)
        return 0.0

    # Extract trajectories
    V_Var_x = trajectory_data.get("V_Var_x", np.zeros(1))
    V_Var_v = trajectory_data.get("V_Var_v", np.zeros(1))
    V_W = trajectory_data.get("V_W", np.zeros(1))
    W_b = trajectory_data.get("W_b", np.zeros(1))

    T = len(V_Var_x)
    times = np.arange(T) * tau

    # Fit rates
    kappa_x = fit_exponential_rate(V_Var_x, times)
    kappa_v = fit_exponential_rate(V_Var_v, times)
    kappa_W = fit_exponential_rate(V_W, times) if len(V_W) > 1 else 0.0
    kappa_b = fit_exponential_rate(W_b, times) if len(W_b) > 1 else 0.0

    epsilon_coupling = 0.05  # Default estimate
    kappa_total = (
        min(kappa_x, kappa_v, kappa_W, kappa_b) * (1.0 - epsilon_coupling)
        if all(k > 0 for k in [kappa_x, kappa_v, kappa_W, kappa_b])
        else 0.0
    )

    return ConvergenceRates(
        kappa_x=kappa_x,
        kappa_v=kappa_v,
        kappa_W=kappa_W,
        kappa_b=kappa_b,
        kappa_total=kappa_total,
        epsilon_coupling=epsilon_coupling,
    )


# ==============================================================================
# Adaptive Tuning (Section 9.10.4)
# ==============================================================================


def adaptive_parameter_tuning(
    trajectory_data: dict[str, Tensor | NDArray],
    params_init: GasParams,
    landscape: LandscapeParams,
    max_iterations: int = 10,
    alpha_init: float = 0.2,
    verbose: bool = True,
) -> tuple[GasParams, list[dict]]:
    """Iteratively improve parameters using empirical measurements.

    From Algorithm 9.10.4 (Adaptive Parameter Tuning):
    1. Measure empirical rates from trajectory
    2. Identify bottleneck
    3. Compute adjustment direction from sensitivity matrix
    4. Update parameters
    5. Validate improvement

    Args:
        trajectory_data: Trajectory measurements (see estimate_rates_from_trajectory)
        params_init: Initial parameter guess
        landscape: Landscape characterization
        max_iterations: Maximum tuning iterations
        alpha_init: Initial step size
        verbose: Print progress

    Returns:
        Tuple of (tuned_params, history) where history is list of dicts with:
            - iteration, params, rates, bottleneck, improvement
    """
    params = GasParams(**vars(params_init))
    alpha = alpha_init
    history = []

    # Estimate current rates from trajectory
    rates_emp = estimate_rates_from_trajectory(trajectory_data, params.tau)
    kappa_base = min(rates_emp.kappa_x, rates_emp.kappa_v, rates_emp.kappa_W, rates_emp.kappa_b)

    if verbose:
        print(f"Initial rate: κ_total = {kappa_base:.6f}")

    for iteration in range(max_iterations):
        # Identify bottleneck
        rate_values = [rates_emp.kappa_x, rates_emp.kappa_v, rates_emp.kappa_W, rates_emp.kappa_b]
        bottleneck_idx = np.argmin(rate_values)
        bottleneck_names = ["Position", "Velocity", "Wasserstein", "Boundary"]
        bottleneck = bottleneck_names[bottleneck_idx]
        kappa_min = rate_values[bottleneck_idx]

        if verbose:
            print(f"\nIter {iteration}: Bottleneck = {bottleneck}, κ = {kappa_min:.6f}")

        # Compute sensitivity matrix
        M_kappa = compute_sensitivity_matrix(params, landscape)

        # Gradient for bottleneck (which parameters improve it)
        grad = M_kappa[bottleneck_idx, :]

        # Estimate gap to achievable rate
        rates_theoretical = compute_convergence_rates(params, landscape)
        kappa_target = min(
            rates_theoretical.kappa_x,
            rates_theoretical.kappa_v,
            rates_theoretical.kappa_W,
            rates_theoretical.kappa_b,
        )
        gap = kappa_target - kappa_min

        # Adaptive step size
        if gap > 0:
            alpha = 0.2 * gap / (kappa_min + 1e-8)
        else:
            alpha = 0.05

        # Update parameters (multiplicative)
        param_names = [
            "tau",
            "gamma",
            "sigma_v",
            "lambda_clone",
            "N",
            "sigma_x",
            "lambda_alg",
            "alpha_rest",
            "d_safe",
            "kappa_wall",
        ]

        params_new = GasParams(**vars(params))
        for j, param_name in enumerate(param_names):
            old_value = getattr(params_new, param_name)
            adjustment = 1.0 + alpha * grad[j]
            new_value = old_value * adjustment
            setattr(params_new, param_name, new_value)

        # Project onto constraints
        params_new = project_parameters_onto_constraints(params_new, landscape)

        # Validate improvement
        rates_new = compute_convergence_rates(params_new, landscape)
        kappa_new = min(rates_new.kappa_x, rates_new.kappa_v, rates_new.kappa_W, rates_new.kappa_b)

        improvement = kappa_new - kappa_min

        if improvement > 0 or iteration == 0:
            params = params_new
            rates_emp = rates_new
            if verbose:
                print(f"  → Accepted: κ_new = {kappa_new:.6f} (Δκ = {improvement:.6f})")
        else:
            alpha *= 0.5
            if verbose:
                print(f"  → Rejected: Reducing step size to α = {alpha:.4f}")

        # Record history
        history.append({
            "iteration": iteration,
            "params": GasParams(**vars(params)),
            "rates": rates_emp,
            "bottleneck": bottleneck,
            "kappa_total": kappa_new if improvement > 0 else kappa_min,
            "improvement": improvement,
        })

        # Convergence check
        if abs(improvement) < 1e-5:
            if verbose:
                print(f"\nConverged after {iteration + 1} iterations")
            break

    return params, history


def project_parameters_onto_constraints(
    params: GasParams, landscape: LandscapeParams
) -> GasParams:
    """Project parameters onto feasible constraint set.

    Enforces:
    - Positivity: all parameters > 0
    - Stability: γ·τ < 0.5, √λ_max · τ < 1
    - Bounds: α_rest ∈ [0, 1], N ≥ 10

    Args:
        params: Parameters to project
        landscape: Landscape characterization

    Returns:
        Projected parameters
    """
    params_proj = GasParams(**vars(params))

    # Positivity
    for attr in vars(params_proj):
        value = getattr(params_proj, attr)
        if isinstance(value, int | float):
            setattr(params_proj, attr, max(value, 1e-8))

    # Stability constraints
    params_proj.tau = min(params_proj.tau, 0.5 / params_proj.gamma)
    params_proj.tau = min(params_proj.tau, 1.0 / np.sqrt(landscape.lambda_max))
    params_proj.tau = min(params_proj.tau, 0.1)  # Practical upper bound

    # Restitution bounds
    params_proj.alpha_rest = np.clip(params_proj.alpha_rest, 0.0, 1.0)

    # Minimum swarm size
    params_proj.N = max(int(params_proj.N), 10)

    # Lambda bounds (avoid too high cloning rate)
    params_proj.lambda_clone = min(params_proj.lambda_clone, 10.0)

    return params_proj


# ==============================================================================
# Evaluation Functions
# ==============================================================================


def evaluate_gas_convergence(
    params: GasParams, landscape: LandscapeParams, verbose: bool = True
) -> dict[str, Any]:
    """Complete convergence analysis for given parameters.

    Args:
        params: Gas parameters to evaluate
        landscape: Landscape characterization
        verbose: Print summary

    Returns:
        Dictionary with all convergence metrics
    """
    rates = compute_convergence_rates(params, landscape)
    constants = compute_equilibrium_constants(params, landscape)
    mixing = compute_mixing_time(params, landscape)

    # Identify bottleneck
    rate_names = ["Position (κ_x)", "Velocity (κ_v)", "Wasserstein (κ_W)", "Boundary (κ_b)"]
    rate_values = [rates.kappa_x, rates.kappa_v, rates.kappa_W, rates.kappa_b]
    bottleneck_idx = np.argmin(rate_values)
    bottleneck = rate_names[bottleneck_idx]

    results = {
        "rates": rates,
        "constants": constants,
        "mixing_time": mixing["T_mix_time"],
        "mixing_steps": mixing["T_mix_steps"],
        "V_equilibrium": mixing["V_eq"],
        "bottleneck": bottleneck,
        "bottleneck_rate": rate_values[bottleneck_idx],
    }

    if verbose:
        print("=" * 60)
        print("EUCLIDEAN GAS CONVERGENCE ANALYSIS")
        print("=" * 60)
        print("\nParameters:")
        print(f"  γ = {params.gamma:.4f}, λ = {params.lambda_clone:.4f}, τ = {params.tau:.6f}")
        print(f"  σ_v = {params.sigma_v:.4f}, N = {params.N}")
        print("\nConvergence Rates:")
        for name, value in zip(rate_names, rate_values):
            marker = " ⚠ BOTTLENECK" if value == rates.kappa_total else ""
            print(f"  {name:20s} = {value:.6f}{marker}")
        print(f"  Total (κ_total)      = {rates.kappa_total:.6f}")
        print("\nMixing Time:")
        print(f"  T_mix = {mixing['T_mix_time']:.2f} time units ({mixing['T_mix_steps']} steps)")
        print("\nEquilibrium:")
        print(f"  V_eq = {mixing['V_eq']:.6f}")
        print("=" * 60)

    return results
