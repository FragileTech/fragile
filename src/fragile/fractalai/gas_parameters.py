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
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor


if TYPE_CHECKING:
    from fragile.fractalai.core.history import RunHistory
    from fragile.experiments.gas_config_dashboard import GasConfig


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


@dataclass
class FitDiagnostics:
    """Diagnostic information for exponential rate fitting."""

    r_squared: float  # Coefficient of determination
    n_negative: int  # Number of negative transients (clamped to 1e-10)
    n_valid: int  # Number of valid points used in fit
    equilibrium_reached: bool  # Whether system appears to have equilibrated
    V_eq_estimate: float  # Estimated equilibrium value
    equilibrium_index: int  # Index where equilibrium starts (adaptive detection)
    confidence_interval: tuple[float, float] | None  # (lower, upper) 95% CI for κ
    A_estimate: float  # Amplitude of exponential decay
    kappa_std: float  # Standard error of κ estimate


def detect_equilibrium_index(V: NDArray, window_size: int = 20) -> int:
    """Detect where trajectory transitions from transient to equilibrium.

    Uses sliding window variance detection to find the point where
    the trajectory stabilizes (variance becomes small and constant).

    Algorithm:
    1. Compute sliding window variance over trajectory
    2. Find where variance drops below threshold (mean of variances / 10)
    3. Return index where equilibrium begins

    Args:
        V: Trajectory values [T]
        window_size: Size of sliding window for variance computation

    Returns:
        Index where equilibrium starts (0 if not detected)
    """
    if len(V) < 3 * window_size:
        # Not enough data, use heuristic (last 30%)
        return int(0.7 * len(V))

    # Compute sliding window variance
    variances = []
    for i in range(len(V) - window_size):
        window_var = np.var(V[i : i + window_size])
        variances.append(window_var)

    variances = np.array(variances)

    # Find threshold: mean variance / 10 (indicating stabilization)
    threshold = np.mean(variances) / 10.0

    # Find first index where variance stays below threshold for at least window_size steps
    for i in range(len(variances) - window_size):
        if np.all(variances[i : i + window_size] < threshold):
            # Found equilibrium region
            return i + window_size // 2

    # No clear equilibrium detected, use last 30%
    return int(0.7 * len(V))


def bootstrap_confidence_interval(
    times: NDArray, log_V_transient: NDArray, n_bootstrap: int = 100, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for exponential decay rate κ.

    Uses residual resampling bootstrap to estimate uncertainty in κ.

    Algorithm:
    1. Fit baseline model to get residuals
    2. Resample residuals with replacement
    3. Refit model to bootstrapped data
    4. Compute percentiles of bootstrap distribution

    Args:
        times: Time values [N]
        log_V_transient: Log of transient values [N]
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (kappa_mean, lower_bound, upper_bound)
    """
    # Baseline fit
    poly_baseline = np.polyfit(times, log_V_transient, 1)
    kappa_baseline = -poly_baseline[0]
    residuals = log_V_transient - (poly_baseline[0] * times + poly_baseline[1])

    # Bootstrap samples
    kappa_samples = []
    for _ in range(n_bootstrap):
        # Resample residuals
        resampled_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
        # Create bootstrapped data
        y_boot = (poly_baseline[0] * times + poly_baseline[1]) + resampled_residuals
        # Refit
        poly_boot = np.polyfit(times, y_boot, 1)
        kappa_boot = -poly_boot[0]
        if kappa_boot > 0:  # Only keep positive rates
            kappa_samples.append(kappa_boot)

    if len(kappa_samples) == 0:
        # Bootstrap failed, return baseline with no CI
        return kappa_baseline, kappa_baseline, kappa_baseline

    kappa_samples = np.array(kappa_samples)

    # Compute percentiles
    alpha = 1.0 - confidence
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    lower_bound = np.percentile(kappa_samples, lower_percentile)
    upper_bound = np.percentile(kappa_samples, upper_percentile)
    kappa_mean = np.mean(kappa_samples)

    return kappa_mean, lower_bound, upper_bound


def estimate_rates_from_trajectory(
    trajectory_data: dict[str, Tensor | NDArray], tau: float, verbose: bool = False
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
        verbose: If True, print diagnostic information

    Returns:
        Estimated convergence rates

    Notes:
        - Returns diagnostics in trajectory_data['_diagnostics'] if verbose=True
        - Rates may be unreliable if trajectory is too short (<50 steps)
        - Check fit quality (R²) before trusting empirical rates
    """

    def fit_exponential_rate(
        V: NDArray | Tensor, times: NDArray, component_name: str = "unknown"
    ) -> tuple[float, FitDiagnostics]:
        """Fit V(t) = C + A * exp(-κ*t) and extract κ with diagnostics."""
        if isinstance(V, Tensor):
            V = V.cpu().numpy()

        # Initialize diagnostics with default values
        diag = FitDiagnostics(
            r_squared=0.0,
            n_negative=0,
            n_valid=0,
            equilibrium_reached=False,
            V_eq_estimate=0.0,
            equilibrium_index=0,
            confidence_interval=None,
            A_estimate=0.0,
            kappa_std=0.0,
        )

        # Skip if not enough data
        if len(V) < 10:
            if verbose:
                print(f"  {component_name}: Insufficient data ({len(V)} points < 10)")
            return 0.0, diag

        # ===== ADAPTIVE EQUILIBRIUM DETECTION =====
        # Use sliding window variance to detect equilibrium transition
        idx_eq_start = detect_equilibrium_index(V, window_size=max(20, len(V) // 10))
        diag.equilibrium_index = idx_eq_start

        # Check if equilibrium appears to be reached
        # Compare variance of detected equilibrium region vs early trajectory
        idx_early_end = min(int(0.3 * len(V)), idx_eq_start)
        var_early = np.var(V[:idx_early_end]) if idx_early_end > 1 else np.var(V[: len(V) // 2])
        var_late = np.var(V[idx_eq_start:]) if idx_eq_start < len(V) - 1 else 0.0
        diag.equilibrium_reached = var_late < 0.1 * var_early and idx_eq_start < 0.9 * len(V)

        # Use robust equilibrium estimation (median of detected equilibrium region)
        V_eq_estimate = np.median(V[idx_eq_start:]) if idx_eq_start < len(V) else V[-1]
        diag.V_eq_estimate = V_eq_estimate

        # Transient: V(t) - V_eq ≈ A * exp(-κ*t)
        V_transient_raw = V[:idx_eq_start] - V_eq_estimate

        # Count and filter negative transients instead of clamping
        negative_mask = V_transient_raw <= 0
        diag.n_negative = int(np.sum(negative_mask))

        # Filter out negative values (don't clamp)
        valid_mask = ~negative_mask
        V_transient = V_transient_raw[valid_mask]
        times_transient = times[:idx_eq_start][valid_mask]
        diag.n_valid = len(V_transient)

        if diag.n_valid < 3:
            if verbose:
                print(
                    f"  {component_name}: Too many negative transients "
                    f"({diag.n_negative}/{idx_eq_start})"
                )
            return 0.0, diag

        # Log-linear regression on valid points
        log_V_transient = np.log(V_transient)

        # Linear fit: log(V_transient) = log(A) - κ*t
        poly = np.polyfit(times_transient, log_V_transient, 1)
        kappa = -poly[0]  # Negative slope
        log_A = poly[1]
        diag.A_estimate = np.exp(log_A)

        # Compute R² for fit quality
        y_pred = poly[0] * times_transient + poly[1]
        ss_res = np.sum((log_V_transient - y_pred) ** 2)
        ss_tot = np.sum((log_V_transient - np.mean(log_V_transient)) ** 2)
        diag.r_squared = 1.0 - (ss_res / (ss_tot + 1e-10))

        # ===== BOOTSTRAP CONFIDENCE INTERVAL =====
        # Only compute if we have enough data and fit quality is reasonable
        if diag.n_valid >= 20 and diag.r_squared > 0.5:
            kappa_mean, lower_bound, upper_bound = bootstrap_confidence_interval(
                times_transient, log_V_transient, n_bootstrap=100, confidence=0.95
            )
            diag.confidence_interval = (lower_bound, upper_bound)
            diag.kappa_std = (upper_bound - lower_bound) / (2 * 1.96)  # Approx std from 95% CI
            # Use bootstrap mean as kappa estimate
            kappa = kappa_mean
        else:
            diag.confidence_interval = (kappa, kappa)  # No uncertainty
            diag.kappa_std = 0.0

        if verbose:
            status = "✓" if diag.r_squared > 0.7 else "⚠"
            eq_status = "equilibrated" if diag.equilibrium_reached else "transient"
            eq_pct = (idx_eq_start / len(V)) * 100

            # Format confidence interval
            if diag.confidence_interval and diag.kappa_std > 0:
                ci_str = f"[{diag.confidence_interval[0]:.6f}, {diag.confidence_interval[1]:.6f}]"
            else:
                ci_str = "N/A"

            print(
                f"  {component_name}: κ={kappa:.6f} (95% CI: {ci_str}), "
                f"R²={diag.r_squared:.3f} {status}, "
                f"{diag.n_valid}/{idx_eq_start} valid points, {eq_status} @ {eq_pct:.0f}%"
            )

        return max(kappa, 0.0), diag

    # Extract trajectories
    V_Var_x = trajectory_data.get("V_Var_x", np.zeros(1))
    V_Var_v = trajectory_data.get("V_Var_v", np.zeros(1))
    V_W = trajectory_data.get("V_W", np.zeros(1))
    W_b = trajectory_data.get("W_b", np.zeros(1))

    T = len(V_Var_x)
    times = np.arange(T) * tau

    if verbose:
        print("\nEmpirical Rate Estimation Diagnostics:")
        print(f"  Trajectory length: {T} steps ({times[-1]:.2f} time units)")

    # Fit rates with diagnostics
    kappa_x, diag_x = fit_exponential_rate(V_Var_x, times, "V_Var_x")
    kappa_v, diag_v = fit_exponential_rate(V_Var_v, times, "V_Var_v")

    if len(V_W) > 1:
        kappa_W, diag_W = fit_exponential_rate(V_W, times, "V_W (Wasserstein proxy)")
    else:
        kappa_W = 0.0
        diag_W = FitDiagnostics(0.0, 0, 0, False, 0.0)

    if len(W_b) > 1:
        kappa_b, diag_b = fit_exponential_rate(W_b, times, "W_b (boundary)")
    else:
        kappa_b = 0.0
        diag_b = FitDiagnostics(0.0, 0, 0, False, 0.0)

    epsilon_coupling = 0.05  # Default estimate
    kappa_total = (
        min(kappa_x, kappa_v, kappa_W, kappa_b) * (1.0 - epsilon_coupling)
        if all(k > 0 for k in [kappa_x, kappa_v, kappa_W, kappa_b])
        else 0.0
    )

    # Store diagnostics in trajectory_data for later inspection
    if "_diagnostics" not in trajectory_data:
        trajectory_data["_diagnostics"] = {}
    trajectory_data["_diagnostics"]["fits"] = {
        "kappa_x": diag_x,
        "kappa_v": diag_v,
        "kappa_W": diag_W,
        "kappa_b": diag_b,
    }

    if verbose:
        avg_r2 = np.mean([diag_x.r_squared, diag_v.r_squared, diag_W.r_squared, diag_b.r_squared])
        if avg_r2 < 0.5:
            print(f"\n⚠ WARNING: Average fit quality R²={avg_r2:.3f} is poor (<0.5)")
            print("  → Empirical rates may be unreliable")
            print("  → Consider: longer simulation, different initial conditions")
        elif avg_r2 < 0.7:
            print(f"\n⚠ CAUTION: Average fit quality R²={avg_r2:.3f} is moderate")
            print("  → Empirical rates should be interpreted carefully")

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
# Empirical Environment Estimation from RunHistory
# ==============================================================================


def estimate_landscape_from_history(
    history: RunHistory,
    use_bounds_analysis: bool = True,
) -> LandscapeParams:
    """Estimate landscape parameters from simulation data.

    Extracts empirical estimates of landscape curvature (λ_min, λ_max) and
    boundary fitness gap (Δf_boundary) from RunHistory.

    From Algorithm 9.10.3 (Empirical Landscape Characterization):
    1. Extract Hessian eigenvalues (if available)
    2. Estimate λ_min, λ_max from percentiles
    3. Analyze boundary behavior for Δf_boundary (if applicable)
    4. Fall back to variance-based estimates if no Hessian data

    Args:
        history: RunHistory from simulation
        use_bounds_analysis: Estimate Δf_boundary from boundary deaths
                            Only valid when pbc=False and walkers die at boundaries

    Returns:
        LandscapeParams with empirical estimates

    Raises:
        ValueError: If history is empty or invalid

    Notes:
        - Requires history.fitness_hessians_diag or fitness_hessians_full
        - Falls back to variance-based λ estimation if no Hessian data
        - Boundary analysis only meaningful when use_bounds_analysis=True
    """
    if history.n_recorded < 2:
        msg = "History too short for landscape estimation (need n_recorded >=2)"
        raise ValueError(msg)

    d = history.d

    # Try to extract Hessian eigenvalues
    lambda_vals = None

    if history.fitness_hessians_diag is not None:
        # Diagonal Hessian: eigenvalues are diagonal elements
        # Shape: [n_recorded-1, N, d]
        hessians_diag = history.fitness_hessians_diag
        # Take absolute values (Hessian eigenvalues can be negative for non-convex)
        lambda_vals = torch.abs(hessians_diag).flatten().cpu().numpy()

    elif history.fitness_hessians_full is not None:
        # Full Hessian: compute eigenvalues
        # Shape: [n_recorded-1, N, d, d]
        hessians_full = history.fitness_hessians_full
        # Flatten to [n_samples, d, d]
        n_samples = hessians_full.shape[0] * hessians_full.shape[1]
        H_flat = hessians_full.reshape(n_samples, d, d)

        # Compute eigenvalues for each sample
        try:
            eigenvals = torch.linalg.eigvalsh(H_flat)  # Symmetric eigenvalue decomp
            lambda_vals = torch.abs(eigenvals).flatten().cpu().numpy()
        except Exception:
            # Eigenvalue computation failed, fall back to variance
            lambda_vals = None

    # Estimate λ_min, λ_max from eigenvalue distribution
    if lambda_vals is not None and len(lambda_vals) > 0:
        # Filter out very small values (numerical noise)
        lambda_vals = lambda_vals[lambda_vals > 1e-6]

        if len(lambda_vals) > 0:
            # Use percentiles to avoid outliers
            lambda_min = float(np.percentile(lambda_vals, 5))
            lambda_max = float(np.percentile(lambda_vals, 95))
        else:
            # All values too small, use defaults
            lambda_min = 1.0
            lambda_max = 10.0
    else:
        # No Hessian data: estimate from position variance
        # λ ~ 1 / Var[x] (harmonic mean over dimensions)
        x_data = history.x_final  # [n_recorded, N, d]
        alive_masks = history.alive_mask  # [n_recorded-1, N]

        # Compute variance across alive walkers at each timestep
        variances_per_dim = []
        for t in range(history.n_recorded - 1):
            x_t = x_data[t]  # [N, d]
            alive_t = alive_masks[t]  # [N]
            x_alive = x_t[alive_t]  # [N_alive, d]

            if len(x_alive) > 1:
                var_t = torch.var(x_alive, dim=0, unbiased=True).cpu().numpy()  # [d]
                variances_per_dim.append(var_t)

        if len(variances_per_dim) > 0:
            variances = np.stack(variances_per_dim, axis=0)  # [T, d]
            mean_var = np.mean(variances, axis=0)  # [d]

            # λ ~ 1 / Var (inverse relationship for quadratic potentials)
            lambda_estimates = 1.0 / (mean_var + 1e-6)
            lambda_min = float(np.min(lambda_estimates))
            lambda_max = float(np.max(lambda_estimates))
        else:
            # Fallback defaults
            lambda_min = 1.0
            lambda_max = 10.0

    # Estimate boundary fitness gap Δf_boundary
    Delta_f_boundary = 0.5  # Default

    if use_bounds_analysis and not history.bounds.pbc:
        # Analyze walker deaths at boundaries
        n_alive = history.n_alive  # [n_recorded]
        death_indices = np.where(np.diff(n_alive.cpu().numpy()) < 0)[0]

        if len(death_indices) > 0:
            # Compute fitness differences near boundary deaths
            fitness_vals = history.fitness  # [n_recorded-1, N]
            alive_masks = history.alive_mask  # [n_recorded-1, N]

            boundary_fitness_gaps = []

            for t_idx in death_indices:
                if t_idx < len(fitness_vals):
                    # Compare fitness of alive vs. dead walkers
                    f_t = fitness_vals[t_idx]  # [N]
                    alive_t = alive_masks[t_idx]  # [N]

                    f_alive = f_t[alive_t]
                    f_dead = f_t[~alive_t]

                    if len(f_alive) > 0 and len(f_dead) > 0:
                        # Boundary gap: median(alive) - median(dead)
                        gap = float(
                            torch.median(f_alive).cpu().item() - torch.median(f_dead).cpu().item()
                        )
                        boundary_fitness_gaps.append(max(gap, 0.0))

            if len(boundary_fitness_gaps) > 0:
                Delta_f_boundary = float(np.mean(boundary_fitness_gaps))

    # Typical fitness scale (use median of fitness values)
    if history.fitness is not None:
        fitness_all = history.fitness.flatten().cpu().numpy()
        fitness_all = fitness_all[np.isfinite(fitness_all)]
        if len(fitness_all) > 0:
            f_typical = float(np.median(np.abs(fitness_all)))
        else:
            f_typical = 1.0
    else:
        f_typical = 1.0

    return LandscapeParams(
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        d=d,
        f_typical=f_typical,
        Delta_f_boundary=Delta_f_boundary,
    )


def compute_wasserstein_proxy_improved(
    history: RunHistory, stage: str = "final", reference_frame: int | None = None
) -> np.ndarray:
    """Compute improved Wasserstein distance proxy using empirical distributions.

    Instead of using V_W = V_Var_x + V_Var_v (crude approximation), compute
    the actual mean pairwise distance to a reference distribution.

    Algorithm:
    1. Select reference distribution (final frame or specified frame)
    2. For each frame t, compute mean distance from walkers(t) to reference
    3. Use this as Wasserstein-1 proxy

    Args:
        history: RunHistory from simulation
        stage: Which state to analyze ("before_clone", "after_clone", "final")
        reference_frame: Index of reference frame (default: last frame)

    Returns:
        Array of Wasserstein proxies [T]

    Notes:
        - This is still an approximation (not true Wasserstein distance)
        - More accurate than simple variance sum
        - Computationally efficient for large N
    """
    # Determine which position/velocity data to use
    if stage == "before_clone":
        x_data = history.x_before_clone  # [n_recorded-1, N, d]
        history.v_before_clone if hasattr(history, "v_before_clone") else None
        n_frames = history.n_recorded - 1
        frame_offset = 1
    elif stage == "after_clone":
        x_data = history.x_after_clone  # [n_recorded-1, N, d]
        history.v_after_clone if hasattr(history, "v_after_clone") else None
        n_frames = history.n_recorded - 1
        frame_offset = 1
    else:  # "final"
        x_data = history.x_final  # [n_recorded, N, d]
        n_frames = history.n_recorded
        frame_offset = 0

    alive_masks = history.alive_mask  # [n_recorded-1, N]

    # Select reference frame (default: last frame)
    if reference_frame is None:
        reference_frame = n_frames - 1

    # Extract reference distribution
    x_ref = x_data[reference_frame].cpu().numpy()  # [N, d]
    if stage != "before_clone" and frame_offset == 0 and reference_frame > 0:
        alive_ref = alive_masks[reference_frame - 1].cpu().numpy()
    else:
        alive_ref = np.ones(x_ref.shape[0], dtype=bool)

    x_ref_alive = x_ref[alive_ref]  # [N_ref, d]

    # Compute Wasserstein proxy for each frame
    W_proxy = np.zeros(n_frames)

    for t in range(n_frames):
        # Extract current distribution
        x_t = x_data[t].cpu().numpy()  # [N, d]

        # Get alive mask for current frame
        if stage != "before_clone" and t + frame_offset > 0:
            alive_t = alive_masks[t + frame_offset - 1].cpu().numpy()
        else:
            alive_t = np.ones(x_t.shape[0], dtype=bool)

        x_t_alive = x_t[alive_t]  # [N_t, d]

        if len(x_t_alive) == 0 or len(x_ref_alive) == 0:
            W_proxy[t] = 0.0
            continue

        # Compute mean pairwise distance (Wasserstein-1 proxy)
        # W_1(μ_t, μ_ref) ≈ mean_{i in t} min_{j in ref} ||x_i - x_j||
        # This is an upper bound on the true W_1 distance

        # For efficiency, use subsample if N is large
        if len(x_t_alive) > 100:
            idx_sample = np.random.choice(len(x_t_alive), size=100, replace=False)
            x_t_sample = x_t_alive[idx_sample]
        else:
            x_t_sample = x_t_alive

        # Compute distances to reference
        # Shape: [N_sample, N_ref]
        dists = np.linalg.norm(x_t_sample[:, None, :] - x_ref_alive[None, :, :], axis=2)

        # Mean minimum distance
        W_proxy[t] = np.mean(np.min(dists, axis=1))

    return W_proxy


def extract_trajectory_data_from_history(
    history: RunHistory,
    stage: str = "final",
    use_improved_wasserstein: bool = True,
) -> dict[str, np.ndarray]:
    """Extract trajectory data for convergence rate estimation.

    Computes Lyapunov function components (V_Var_x, V_Var_v) and Wasserstein
    proxy from RunHistory for use with estimate_rates_from_trajectory().

    From Algorithm 9.10.4 (Adaptive Parameter Tuning):
    Trajectory data includes:
    - V_Var_x: Position variance over time
    - V_Var_v: Velocity variance over time
    - V_W: Wasserstein distance proxy (optional)
    - W_b: Boundary potential proxy (optional)

    Args:
        history: RunHistory from simulation
        stage: Which stage to extract ("final", "after_clone", "before_clone")
        use_improved_wasserstein: If True, use better Wasserstein proxy (slower)

    Returns:
        Dictionary with trajectory arrays:
        - "V_Var_x": Position variance [T]
        - "V_Var_v": Velocity variance [T]
        - "V_W": Wasserstein proxy [T] (if sufficient data)
        - "W_b": Boundary proxy [T] (if applicable)

    Notes:
        - Uses alive walkers only (masked by alive_mask)
        - V_W computed using improved empirical distance (if enabled)
        - W_b computed from boundary death rate
    """
    # Import here to avoid circular dependency
    from fragile.lyapunov import compute_lyapunov_trajectory

    # Extract Lyapunov components
    lyap_data = compute_lyapunov_trajectory(
        history=history,
        stage=stage,
        use_alive_mask=True,
    )

    trajectory_data = {
        "V_Var_x": lyap_data["V_var_x"],  # Already numpy array
        "V_Var_v": lyap_data["V_var_v"],
    }

    # Compute Wasserstein proxy
    if use_improved_wasserstein:
        # Use improved proxy based on empirical distances
        try:
            trajectory_data["V_W"] = compute_wasserstein_proxy_improved(
                history, stage=stage, reference_frame=None
            )
        except Exception:
            # Fall back to simple proxy if improved version fails
            trajectory_data["V_W"] = lyap_data["V_total"]
    else:
        # Simple proxy: V_W ~ V_total = V_Var_x + V_Var_v
        trajectory_data["V_W"] = lyap_data["V_total"]

    # Compute boundary proxy from walker death rate (if applicable)
    if not history.bounds.pbc:
        n_alive = history.n_alive.cpu().numpy()  # [n_recorded]
        # Boundary potential proxy: negative log of survival probability
        # W_b ~ -log(N_alive / N_initial)
        N_initial = float(n_alive[0])
        W_b = -np.log((n_alive / N_initial) + 1e-10)
        trajectory_data["W_b"] = W_b
    else:
        # No boundary deaths with periodic boundaries
        trajectory_data["W_b"] = np.zeros(len(trajectory_data["V_Var_x"]))

    return trajectory_data


def create_fit_diagnostic_plots(
    trajectory_data: dict[str, np.ndarray | Tensor], tau: float
) -> dict[str, Any]:
    """Create diagnostic plots showing V(t) decay and exponential fits.

    Generates HoloViews plots for visual inspection of fit quality and
    equilibrium detection.

    Args:
        trajectory_data: Dictionary with trajectory arrays (from extract_trajectory_data_from_history)
        tau: Timestep size

    Returns:
        Dictionary with HoloViews plot objects:
            - "V_Var_x_plot": Position variance trajectory with fit
            - "V_Var_v_plot": Velocity variance trajectory with fit
            - "V_W_plot": Wasserstein proxy trajectory with fit
            - "combined_plot": Combined overlay of all components

    Notes:
        - Requires trajectory_data["_diagnostics"] to be populated
        - Uses holoviews for plotting (bokeh backend)
        - Automatically loads bokeh extension if not already loaded
    """
    import holoviews as hv

    # Ensure bokeh extension is loaded
    try:
        hv.extension("bokeh")
    except Exception:
        pass  # Already loaded or not available

    # Check if diagnostics are available
    if "_diagnostics" not in trajectory_data or "fits" not in trajectory_data["_diagnostics"]:
        # Return empty plot with message
        return {
            "message": "No diagnostics available. Run estimate_rates_from_trajectory with verbose=True first."
        }

    fits = trajectory_data["_diagnostics"]["fits"]

    # Extract trajectories
    V_Var_x = trajectory_data.get("V_Var_x", np.zeros(1))
    V_Var_v = trajectory_data.get("V_Var_v", np.zeros(1))
    V_W = trajectory_data.get("V_W", np.zeros(1))

    T = len(V_Var_x)
    times = np.arange(T) * tau

    plots = {}

    # Helper function to create individual plot
    def create_component_plot(V: np.ndarray, times: np.ndarray, component_name: str, fit_key: str):
        """Create plot for one component."""
        fit_diag = fits.get(fit_key)
        if fit_diag is None or fit_diag.r_squared == 0.0:
            return None

        # Plot data
        data_curve = hv.Curve((times, V), kdims="Time", vdims="Value").opts(
            color="blue", line_width=2, alpha=0.7, title=f"{component_name} Trajectory"
        )

        # Plot equilibrium line
        V_eq = fit_diag.V_eq_estimate
        eq_line = hv.HLine(V_eq).opts(color="green", line_dash="dashed", line_width=1.5)

        # Plot exponential fit
        idx_eq = fit_diag.equilibrium_index
        times_fit = times[:idx_eq]
        V_eq + fit_diag.A_estimate * np.exp(-fit_diag.r_squared * times_fit)

        # Recompute fit using stored kappa and A
        # Reconstruct: V(t) = C + A * exp(-κ*t)
        # We need to get kappa from the diagnostics
        # The fit was done on log(V - V_eq) = log(A) - κ*t
        # So V = V_eq + A * exp(-kappa * t)

        # Get kappa from fit (it's not stored directly in FitDiagnostics)
        # We'll approximate it from the data
        V_transient = V[:idx_eq] - V_eq
        V_transient_valid = V_transient[V_transient > 0]
        times_valid = times[:idx_eq][V_transient[: len(times_fit)] > 0]

        if len(V_transient_valid) > 2 and len(times_valid) > 2:
            log_V_transient = np.log(V_transient_valid)
            poly = np.polyfit(times_valid, log_V_transient, 1)
            kappa = -poly[0]
            A = np.exp(poly[1])

            # Generate fit curve
            times_fit_dense = np.linspace(0, times[idx_eq], 100)
            V_fit_dense = V_eq + A * np.exp(-kappa * times_fit_dense)

            fit_curve = hv.Curve((times_fit_dense, V_fit_dense), kdims="Time", vdims="Value").opts(
                color="red", line_width=2, line_dash="dotted", alpha=0.8
            )
        else:
            fit_curve = hv.Curve([]).opts(visible=False)

        # Add vertical line at equilibrium index
        eq_vline = hv.VLine(times[idx_eq]).opts(
            color="orange", line_dash="dashdot", line_width=1.5
        )

        # Add confidence interval shading if available
        if fit_diag.confidence_interval and fit_diag.kappa_std > 0:
            kappa_lower, kappa_upper = fit_diag.confidence_interval
            V_fit_lower = V_eq + A * np.exp(-kappa_upper * times_fit_dense)
            V_fit_upper = V_eq + A * np.exp(-kappa_lower * times_fit_dense)

            # Create area for CI
            ci_area = hv.Area((times_fit_dense, V_fit_lower, V_fit_upper), vdims=["y", "y2"]).opts(
                color="red", alpha=0.15
            )

            overlay = data_curve * eq_line * fit_curve * eq_vline * ci_area
        else:
            overlay = data_curve * eq_line * fit_curve * eq_vline

        # Add labels
        label_text = (
            f"R² = {fit_diag.r_squared:.3f}\n"
            f"V_eq = {V_eq:.4f}\n"
            f"Equilibrium @ {(idx_eq / len(V)) * 100:.0f}%"
        )
        label = hv.Text(times[len(times) // 4], np.max(V) * 0.9, label_text).opts(
            text_align="left", text_font_size="10pt"
        )

        return (overlay * label).opts(
            width=600,
            height=400,
            xlabel="Time",
            ylabel="Value",
            show_legend=False,
            tools=["hover"],
            fontsize={"title": 14, "labels": 12, "xticks": 10, "yticks": 10},
        )

    # Create plots for each component
    plots["V_Var_x_plot"] = create_component_plot(V_Var_x, times, "V_Var_x (Position)", "kappa_x")
    plots["V_Var_v_plot"] = create_component_plot(V_Var_v, times, "V_Var_v (Velocity)", "kappa_v")

    if len(V_W) > 1:
        plots["V_W_plot"] = create_component_plot(V_W, times, "V_W (Wasserstein)", "kappa_W")

    # Create combined plot with all trajectories normalized
    def normalize(V):
        """Normalize to [0, 1] range."""
        V_min, V_max = np.min(V), np.max(V)
        if V_max - V_min < 1e-10:
            return np.zeros_like(V)
        return (V - V_min) / (V_max - V_min)

    combined_curves = []
    if len(V_Var_x) > 1:
        combined_curves.append(
            hv.Curve((times, normalize(V_Var_x)), label="V_Var_x").opts(color="blue", line_width=2)
        )
    if len(V_Var_v) > 1:
        combined_curves.append(
            hv.Curve((times, normalize(V_Var_v)), label="V_Var_v").opts(
                color="green", line_width=2
            )
        )
    if len(V_W) > 1:
        combined_curves.append(
            hv.Curve((times, normalize(V_W)), label="V_W").opts(color="red", line_width=2)
        )

    if combined_curves:
        plots["combined_plot"] = hv.Overlay(combined_curves).opts(
            width=800,
            height=400,
            xlabel="Time",
            ylabel="Normalized Value [0, 1]",
            title="Combined Trajectories (Normalized)",
            legend_position="right",
            show_legend=True,
            tools=["hover"],
        )

    return plots


def optimize_parameters_multi_strategy(
    strategy: str,
    landscape: LandscapeParams,
    current_params: GasParams,
    trajectory_data: dict[str, np.ndarray] | None = None,
    V_target: float = 0.1,
) -> tuple[GasParams, dict[str, Any]]:
    """Optimize Gas parameters using selected strategy.

    Provides multiple optimization strategies:
    - **balanced**: Eliminate bottlenecks using closed-form solution
    - **empirical**: Adaptive tuning from trajectory data
    - **conservative**: Safe parameters with stability margins
    - **aggressive**: Maximum convergence rate (push stability limits)

    From Section 9.10 (Rate-Space Optimization):
    Each strategy computes optimal parameters targeting different objectives:
    - Balanced: min{κ_x, κ_v, κ_W, κ_b} maximized (no bottleneck)
    - Empirical: Fit to observed convergence behavior
    - Conservative: Reduce rates, increase safety margins (γ·τ << 0.5)
    - Aggressive: Maximize rates, push limits (γ·τ → 0.5)

    Args:
        strategy: One of "balanced", "empirical", "conservative", "aggressive"
        landscape: Landscape characterization (empirical or default)
        current_params: Current Gas parameters (for empirical/conservative)
        trajectory_data: Trajectory data (required for empirical strategy)
        V_target: Target equilibrium variance

    Returns:
        Tuple of (optimized_params, diagnostics) where diagnostics contains:
        - strategy: Strategy name
        - improvement_ratio: Expected κ_total improvement
        - expected_T_mix: Predicted mixing time
        - bottleneck_before: Bottleneck before optimization
        - bottleneck_after: Bottleneck after optimization

    Raises:
        ValueError: If empirical strategy without trajectory data
        ValueError: If unknown strategy

    Notes:
        - Balanced strategy uses compute_optimal_parameters()
        - Empirical strategy uses adaptive_parameter_tuning()
        - Conservative/Aggressive modify balanced solution
    """
    # Validate strategy
    valid_strategies = ["balanced", "empirical", "conservative", "aggressive"]
    if strategy not in valid_strategies:
        raise ValueError(f"Unknown strategy '{strategy}'. Must be one of {valid_strategies}")

    # Compute current rates for comparison
    rates_current = compute_convergence_rates(current_params, landscape)
    kappa_before = rates_current.kappa_total

    # Identify current bottleneck
    rate_values_before = [
        rates_current.kappa_x,
        rates_current.kappa_v,
        rates_current.kappa_W,
        rates_current.kappa_b,
    ]
    bottleneck_names = ["Position", "Velocity", "Wasserstein", "Boundary"]
    bottleneck_before = bottleneck_names[np.argmin(rate_values_before)]

    # Execute strategy
    if strategy == "balanced":
        # Closed-form balanced optimum
        optimal_params = compute_optimal_parameters(
            landscape=landscape,
            V_target=V_target,
            gamma_budget=None,
        )

    elif strategy == "empirical":
        # Adaptive tuning from trajectory
        if trajectory_data is None:
            msg = "Empirical strategy requires trajectory_data"
            raise ValueError(msg)

        optimal_params, _history = adaptive_parameter_tuning(
            trajectory_data=trajectory_data,
            params_init=current_params,
            landscape=landscape,
            max_iterations=10,
            alpha_init=0.2,
            verbose=False,
        )

    elif strategy == "conservative":
        # Start with balanced, then add safety margins
        optimal_params = compute_optimal_parameters(
            landscape=landscape,
            V_target=V_target,
            gamma_budget=None,
        )

        # Reduce rates by 20% for safety
        optimal_params = GasParams(
            tau=optimal_params.tau * 0.8,  # Smaller timestep
            gamma=optimal_params.gamma * 0.9,  # Slightly less friction
            sigma_v=optimal_params.sigma_v,
            lambda_clone=optimal_params.lambda_clone * 0.8,  # Slower cloning
            N=optimal_params.N,
            sigma_x=optimal_params.sigma_x,
            lambda_alg=optimal_params.lambda_alg,
            alpha_rest=optimal_params.alpha_rest,
            d_safe=optimal_params.d_safe,
            kappa_wall=optimal_params.kappa_wall,
        )

        # Ensure stability margins: γ·τ < 0.4 (not 0.5)
        if optimal_params.gamma * optimal_params.tau > 0.4:
            optimal_params = GasParams(**{
                **vars(optimal_params),
                "tau": 0.4 / optimal_params.gamma,
            })

    elif strategy == "aggressive":
        # Start with balanced, then push limits
        optimal_params = compute_optimal_parameters(
            landscape=landscape,
            V_target=V_target,
            gamma_budget=None,
        )

        # Increase rates by 15%
        optimal_params = GasParams(
            tau=min(optimal_params.tau * 1.1, 0.5 / optimal_params.gamma),  # Push timestep
            gamma=optimal_params.gamma * 1.1,  # More friction
            sigma_v=optimal_params.sigma_v,
            lambda_clone=min(optimal_params.lambda_clone * 1.15, 10.0),  # Faster cloning
            N=optimal_params.N,
            sigma_x=optimal_params.sigma_x,
            lambda_alg=optimal_params.lambda_alg,
            alpha_rest=optimal_params.alpha_rest,
            d_safe=optimal_params.d_safe,
            kappa_wall=optimal_params.kappa_wall,
        )

    # Project onto constraints to ensure validity
    optimal_params = project_parameters_onto_constraints(optimal_params, landscape)

    # Compute optimized rates
    rates_optimal = compute_convergence_rates(optimal_params, landscape)
    kappa_after = rates_optimal.kappa_total

    # Identify new bottleneck
    rate_values_after = [
        rates_optimal.kappa_x,
        rates_optimal.kappa_v,
        rates_optimal.kappa_W,
        rates_optimal.kappa_b,
    ]
    bottleneck_after = bottleneck_names[np.argmin(rate_values_after)]

    # Compute expected mixing time
    mixing = compute_mixing_time(optimal_params, landscape)

    # Compute improvement ratio
    if kappa_before > 0:
        improvement_ratio = (kappa_after - kappa_before) / kappa_before
    else:
        improvement_ratio = float("inf")

    # Build diagnostics
    diagnostics = {
        "strategy": strategy,
        "improvement_ratio": improvement_ratio,
        "kappa_before": kappa_before,
        "kappa_after": kappa_after,
        "expected_T_mix": mixing["T_mix_time"],
        "expected_T_mix_steps": mixing["T_mix_steps"],
        "bottleneck_before": bottleneck_before,
        "bottleneck_after": bottleneck_after,
        "rates_before": rates_current,
        "rates_after": rates_optimal,
    }

    return optimal_params, diagnostics


# ==============================================================================
# GasParams ↔ GasConfig Conversion Utilities
# ==============================================================================


def gas_params_from_config(gas_config: GasConfig) -> GasParams:
    """Extract GasParams from GasConfig for optimization.

    Maps GasConfig UI parameters to mathematical GasParams structure.

    Key mappings:
    - tau ← delta_t (time step)
    - gamma ← gamma (friction)
    - sigma_v ← sqrt(2/(gamma*beta)) (Langevin noise scale)
    - lambda_clone ← estimated from fitness parameters
    - N ← N (number of walkers)
    - sigma_x ← sigma_x (cloning jitter)
    - lambda_alg ← lambda_alg (velocity weight)
    - alpha_rest ← alpha_restitution (restitution coefficient)
    - d_safe, kappa_wall ← defaults (not exposed in UI)

    Args:
        gas_config: GasConfig object from dashboard

    Returns:
        GasParams for theoretical analysis

    Notes:
        - lambda_clone estimation is approximate (from alpha_fit/beta_fit)
        - sigma_v derived from thermodynamic relation
        - Some parameters (d_safe, kappa_wall) use reasonable defaults
    """
    # Direct mappings
    tau = float(gas_config.delta_t)
    gamma = float(gas_config.gamma)
    N = int(gas_config.N)
    sigma_x = float(gas_config.sigma_x)
    lambda_alg = float(gas_config.lambda_alg)
    alpha_rest = float(gas_config.alpha_restitution)

    # Derive sigma_v from beta (thermal noise)
    # From Langevin equilibrium: sigma_v = sqrt(2/(gamma*beta))
    beta = float(gas_config.beta)
    sigma_v = np.sqrt(2.0 / (gamma * beta))

    # Estimate lambda_clone from fitness parameters
    # Approximate: lambda ~ alpha_fit / (step_size)
    # This is rough because actual cloning rate depends on fitness landscape
    alpha_fit = float(gas_config.alpha_fit)
    lambda_clone = alpha_fit / tau  # Rough estimate

    # Default values for parameters not in GasConfig UI
    d_safe = 3.0  # Safe harbor distance
    kappa_wall = 10.0  # Boundary stiffness

    return GasParams(
        tau=tau,
        gamma=gamma,
        sigma_v=sigma_v,
        lambda_clone=lambda_clone,
        N=N,
        sigma_x=sigma_x,
        lambda_alg=lambda_alg,
        alpha_rest=alpha_rest,
        d_safe=d_safe,
        kappa_wall=kappa_wall,
    )


def gas_params_to_config_dict(
    params: GasParams,
    preserve_adaptive: bool = True,
) -> dict[str, Any]:
    """Convert GasParams to GasConfig parameter dictionary.

    Creates dictionary of parameter updates for GasConfig.
    Only core parameters are updated; adaptive features preserved by default.

    Key mappings:
    - delta_t ← tau
    - gamma ← gamma
    - beta ← 2/(gamma*sigma_v²) (derived from sigma_v)
    - N ← N
    - sigma_x ← sigma_x
    - lambda_alg ← lambda_alg
    - alpha_restitution ← alpha_rest

    Args:
        params: GasParams from optimization
        preserve_adaptive: If True, only update core params (default)
                          If False, reset adaptive features (epsilon_F=0, nu=0, etc.)

    Returns:
        Dictionary of GasConfig parameter updates

    Notes:
        - Beta computed from thermodynamic relation
        - Fitness parameters (alpha_fit, beta_fit) preserved (complex mapping)
        - Adaptive parameters (epsilon_F, nu, epsilon_Sigma) preserved if flag set
    """
    # Core parameter mappings
    config_dict = {
        "delta_t": float(params.tau),
        "gamma": float(params.gamma),
        "N": int(params.N),
        "sigma_x": float(params.sigma_x),
        "lambda_alg": float(params.lambda_alg),
        "alpha_restitution": float(params.alpha_rest),
    }

    # Derive beta from sigma_v
    # sigma_v = sqrt(2/(gamma*beta)) → beta = 2/(gamma*sigma_v²)
    if params.sigma_v > 0 and params.gamma > 0:
        beta = 2.0 / (params.gamma * params.sigma_v**2)
        config_dict["beta"] = float(beta)

    # Reset adaptive parameters if requested
    if not preserve_adaptive:
        config_dict.update({
            "epsilon_F": 0.0,
            "use_fitness_force": False,
            "epsilon_Sigma": 0.0,
            "use_anisotropic_diffusion": False,
            "nu": 0.0,
            "use_viscous_coupling": False,
        })

    return config_dict


def apply_gas_params_to_config(
    params: GasParams,
    gas_config: GasConfig,
    preserve_adaptive: bool = True,
) -> None:
    """Apply optimized parameters to GasConfig in-place.

    Updates GasConfig object with optimized parameters from GasParams.
    Triggers Panel UI refresh for all updated widgets.

    Args:
        params: Optimized GasParams
        gas_config: GasConfig object to update
        preserve_adaptive: Preserve adaptive parameters (default: True)

    Side Effects:
        - Modifies gas_config parameters in-place
        - Triggers Panel param watchers
        - Updates UI widgets

    Notes:
        - Use this after optimization to update dashboard configuration
        - User can review updated parameters before running simulation
        - Does NOT trigger simulation automatically
    """
    # Get parameter update dictionary
    config_dict = gas_params_to_config_dict(params, preserve_adaptive=preserve_adaptive)

    # Update each parameter (triggers Panel watchers)
    for param_name, value in config_dict.items():
        if hasattr(gas_config, param_name):
            setattr(gas_config, param_name, value)


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
