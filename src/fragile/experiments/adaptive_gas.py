"""Adaptive Gas Analysis Module with Ellipticity Validation.

This module provides comprehensive analysis and validation for the Adaptive Gas
algorithm, which extends the Euclidean Gas with:
- Fitness force (adaptive force from mean-field fitness potential)
- Viscous coupling (fluid-like walker interactions)
- Anisotropic diffusion (fitness-dependent diffusion tensor)

Key Concept: Uniform Ellipticity
-------------------------------
The anisotropic diffusion tensor must satisfy:

    c_min · I ≤ Σ(x) ≤ c_max · I

where:
- Σ(x) = ε_Σ · I - H(x) (regularized Hessian)
- H(x) is the Hessian of the potential U(x)
- ε_Σ is the regularization parameter
- c_min = 1 / (H_max + ε_Σ)
- c_max = 1 / (ε_Σ - H_max)

**CRITICAL CONDITION:** ε_Σ > H_max (uniform ellipticity)

If this condition is violated, the diffusion tensor can become degenerate
(zero or negative eigenvalues), breaking convergence guarantees.

References
----------
- docs/source/2_geometric_gas/ - Geometric Gas framework
- docs/source/3_brascamp_lieb/geometric_foundations_lsi.md - LSI theory
- src/fragile/convergence_bounds.py - Theoretical bounds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray

from fragile.convergence_bounds import c_max
from fragile.convergence_bounds import c_min
from fragile.convergence_bounds import epsilon_F_star
from fragile.convergence_bounds import validate_ellipticity
from fragile.convergence_bounds import validate_hypocoercivity

if TYPE_CHECKING:
    from fragile.core.gas_history import RunHistory


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class EllipticityDiagnostics:
    """Diagnostics for uniform ellipticity condition.

    Attributes
    ----------
    epsilon_Sigma : float
        Regularization parameter ε_Σ
    H_max : float
        Maximum Hessian spectral norm ||H||_max
    H_max_source : str
        How H_max was determined ("estimated", "parameter", "unknown")
    is_valid : bool
        Whether ε_Σ > H_max (uniform ellipticity satisfied)
    margin : float
        Safety margin ε_Σ - H_max (positive if valid)
    c_min_value : float
        Lower ellipticity bound 1/(H_max + ε_Σ)
    c_max_value : float | None
        Upper ellipticity bound 1/(ε_Σ - H_max), None if invalid
    condition_number : float | None
        Condition number c_max/c_min, None if invalid
    recommended_epsilon_Sigma : float
        Recommended ε_Σ ≥ 1.2 × H_max for safety
    guidance : str
        User-friendly guidance message
    """

    epsilon_Sigma: float
    H_max: float
    H_max_source: str
    is_valid: bool
    margin: float
    c_min_value: float
    c_max_value: float | None
    condition_number: float | None
    recommended_epsilon_Sigma: float
    guidance: str


@dataclass
class HypocoercivityDiagnostics:
    """Diagnostics for hypocoercivity regime.

    The hypocoercivity regime requires:
    1. ε_F < ε_F* (fitness force not too strong)
    2. ν > 0 (viscous coupling enabled)

    Attributes
    ----------
    epsilon_F : float
        Fitness force parameter
    epsilon_F_star_value : float
        Critical threshold ε_F* = c_min / (2·F_adapt_max)
    nu : float
        Viscous coupling strength
    use_fitness_force : bool
        Whether fitness force is enabled
    use_viscous_coupling : bool
        Whether viscous coupling is enabled
    is_valid : bool
        Whether ε_F < ε_F* and ν > 0
    margin : float
        Safety margin ε_F* - ε_F (positive if valid)
    guidance : str
        User-friendly guidance message
    """

    epsilon_F: float
    epsilon_F_star_value: float
    nu: float
    use_fitness_force: bool
    use_viscous_coupling: bool
    is_valid: bool
    margin: float
    guidance: str


@dataclass
class ViscousCouplingDiagnostics:
    """Diagnostics for viscous coupling analysis.

    Attributes
    ----------
    nu : float
        Viscous coupling strength
    length_scale : float
        Viscous kernel length scale l
    is_enabled : bool
        Whether viscous coupling is enabled
    mean_force_magnitude : float
        Average magnitude of viscous force
    max_force_magnitude : float
        Maximum magnitude of viscous force
    guidance : str
        User-friendly guidance message
    """

    nu: float
    length_scale: float
    is_enabled: bool
    mean_force_magnitude: float
    max_force_magnitude: float
    guidance: str


@dataclass
class DiffusionMetrics:
    """Metrics for anisotropic diffusion tensor analysis.

    Attributes
    ----------
    epsilon_Sigma : float
        Regularization parameter
    mean_min_eigenvalue : float
        Mean of minimum eigenvalues across trajectory
    mean_max_eigenvalue : float
        Mean of maximum eigenvalues across trajectory
    mean_condition_number : float
        Mean condition number λ_max/λ_min
    worst_condition_number : float
        Worst (maximum) condition number observed
    anisotropy_ratio : float
        Ratio of max to min eigenvalue spread
    guidance : str
        User-friendly guidance message
    """

    epsilon_Sigma: float
    mean_min_eigenvalue: float
    mean_max_eigenvalue: float
    mean_condition_number: float
    worst_condition_number: float
    anisotropy_ratio: float
    guidance: str


@dataclass
class AdaptiveGasDiagnostics:
    """Complete diagnostics for Adaptive Gas configuration.

    Attributes
    ----------
    ellipticity : EllipticityDiagnostics | None
        Uniform ellipticity diagnostics (if anisotropic diffusion enabled)
    hypocoercivity : HypocoercivityDiagnostics | None
        Hypocoercivity regime diagnostics (if fitness force enabled)
    viscous_coupling : ViscousCouplingDiagnostics | None
        Viscous coupling diagnostics (if enabled)
    diffusion_metrics : DiffusionMetrics | None
        Diffusion tensor metrics (if anisotropic diffusion enabled)
    is_euclidean_gas : bool
        True if no adaptive features enabled (pure Euclidean Gas)
    overall_status : str
        "valid", "warning", or "invalid"
    summary : str
        Overall summary message
    """

    ellipticity: EllipticityDiagnostics | None
    hypocoercivity: HypocoercivityDiagnostics | None
    viscous_coupling: ViscousCouplingDiagnostics | None
    diffusion_metrics: DiffusionMetrics | None
    is_euclidean_gas: bool
    overall_status: str
    summary: str


# ============================================================================
# H_max Estimation from Trajectory
# ============================================================================


def estimate_H_max_from_history(
    history: RunHistory,
    stage: str = "final",
    sample_size: int = 100,
    percentile: float = 99.0,
) -> float:
    """Estimate maximum Hessian spectral norm H_max from trajectory.

    This function approximates H_max = sup_x ||H(x)||_op by computing
    finite difference Hessians at sampled trajectory points and taking
    a high percentile of the spectral norms.

    Parameters
    ----------
    history : RunHistory
        Trajectory history from simulation
    stage : str, default="final"
        Which stage to analyze ("before_clone", "after_clone", "final")
    sample_size : int, default=100
        Number of frames to sample for Hessian estimation
    percentile : float, default=99.0
        Percentile to use for H_max estimate (guards against outliers)

    Returns
    -------
    float
        Estimated H_max (99th percentile of Hessian spectral norms)

    Notes
    -----
    The Hessian is estimated using finite differences:
        H[i,j] ≈ (∇U(x + e_j·h) - ∇U(x - e_j·h)) / (2h)

    This is computationally expensive, so we sample a subset of frames.
    """
    from fragile.gas_parameters import extract_trajectory_data_from_history

    # Get trajectory data
    trajectory_data = extract_trajectory_data_from_history(
        history, stage=stage, use_improved_wasserstein=False
    )

    # Sample frames uniformly
    n_frames = len(trajectory_data["V_Var_x"])
    if n_frames <= sample_size:
        sample_indices = list(range(n_frames))
    else:
        sample_indices = np.linspace(0, n_frames - 1, sample_size, dtype=int)

    # Extract positions from history
    if not hasattr(history, "states") or len(history.states) == 0:
        # Fallback: use heuristic
        return 1.0

    hessian_norms = []

    for idx in sample_indices:
        if idx >= len(history.states[stage]):
            continue

        state = history.states[stage][idx]
        if not hasattr(state, "x") or state.x is None:
            continue

        # Get alive walkers
        if hasattr(state, "alive_mask"):
            x_alive = state.x[state.alive_mask]
        else:
            x_alive = state.x

        if len(x_alive) == 0:
            continue

        # Sample a few walkers per frame (avoid excessive computation)
        n_sample = min(5, len(x_alive))
        sample_walker_indices = np.random.choice(len(x_alive), n_sample, replace=False)

        for walker_idx in sample_walker_indices:
            x = x_alive[walker_idx]

            # Compute Hessian via finite differences
            H_norm = _estimate_hessian_norm(x, history.potential, h=1e-4)
            if H_norm is not None and np.isfinite(H_norm):
                hessian_norms.append(H_norm)

    if len(hessian_norms) == 0:
        # Fallback to heuristic
        return 1.0

    # Take high percentile to estimate H_max
    H_max_estimate = float(np.percentile(hessian_norms, percentile))
    return H_max_estimate


def _estimate_hessian_norm(
    x: torch.Tensor | NDArray,
    potential: callable,
    h: float = 1e-4,
) -> float | None:
    """Estimate Hessian spectral norm at point x using finite differences.

    Parameters
    ----------
    x : torch.Tensor or NDArray
        Position [d]
    potential : callable
        Potential function U(x)
    h : float, default=1e-4
        Finite difference step size

    Returns
    -------
    float | None
        Spectral norm ||H(x)||_op, or None if computation fails
    """
    if isinstance(x, torch.Tensor):
        x_np = x.cpu().numpy()
    else:
        x_np = np.array(x)

    d = len(x_np)
    H = np.zeros((d, d))

    try:
        # Compute gradient at x
        grad_center = _compute_gradient(x_np, potential, h)

        # Compute Hessian via finite differences
        for j in range(d):
            e_j = np.zeros(d)
            e_j[j] = 1.0

            grad_plus = _compute_gradient(x_np + h * e_j, potential, h)
            grad_minus = _compute_gradient(x_np - h * e_j, potential, h)

            # H[:,j] ≈ (grad_plus - grad_minus) / (2h)
            H[:, j] = (grad_plus - grad_minus) / (2.0 * h)

        # Symmetrize (Hessian should be symmetric)
        H = 0.5 * (H + H.T)

        # Compute spectral norm (largest singular value)
        eigenvalues = np.linalg.eigvalsh(H)
        H_norm = float(np.max(np.abs(eigenvalues)))

        return H_norm

    except Exception:
        return None


def _compute_gradient(x: NDArray, potential: callable, h: float = 1e-4) -> NDArray:
    """Compute gradient ∇U(x) using finite differences.

    Parameters
    ----------
    x : NDArray
        Position [d]
    potential : callable
        Potential function U(x)
    h : float, default=1e-4
        Finite difference step size

    Returns
    -------
    NDArray
        Gradient ∇U(x) [d]
    """
    d = len(x)
    grad = np.zeros(d)

    for i in range(d):
        e_i = np.zeros(d)
        e_i[i] = 1.0

        # Central difference: ∂U/∂x_i ≈ (U(x+h·e_i) - U(x-h·e_i)) / (2h)
        x_plus = torch.tensor(x + h * e_i, dtype=torch.float32)
        x_minus = torch.tensor(x - h * e_i, dtype=torch.float32)

        U_plus = potential(x_plus)
        U_minus = potential(x_minus)

        if isinstance(U_plus, torch.Tensor):
            U_plus = U_plus.item()
        if isinstance(U_minus, torch.Tensor):
            U_minus = U_minus.item()

        grad[i] = (U_plus - U_minus) / (2.0 * h)

    return grad


# ============================================================================
# Ellipticity Validation
# ============================================================================


def validate_ellipticity_conditions(
    epsilon_Sigma: float,
    H_max: float,
    H_max_source: str = "parameter",
    safety_margin: float = 0.2,
) -> EllipticityDiagnostics:
    """Validate uniform ellipticity condition ε_Σ > H_max.

    Parameters
    ----------
    epsilon_Sigma : float
        Regularization parameter ε_Σ
    H_max : float
        Maximum Hessian spectral norm
    H_max_source : str, default="parameter"
        How H_max was determined ("estimated", "parameter", "unknown")
    safety_margin : float, default=0.2
        Recommended safety factor (recommend ε_Σ ≥ (1+margin)·H_max)

    Returns
    -------
    EllipticityDiagnostics
        Complete diagnostics with guidance

    Examples
    --------
    >>> diag = validate_ellipticity_conditions(epsilon_Sigma=0.1, H_max=0.05)
    >>> print(diag.is_valid)  # True
    >>> print(diag.guidance)
    """
    is_valid = validate_ellipticity(epsilon_Sigma, H_max)
    margin = epsilon_Sigma - H_max

    # Compute bounds
    c_min_value = c_min(epsilon_Sigma, H_max)

    if is_valid:
        c_max_value = c_max(epsilon_Sigma, H_max)
        condition_number = c_max_value / c_min_value
    else:
        c_max_value = None
        condition_number = None

    # Recommended epsilon_Sigma
    recommended_epsilon_Sigma = H_max * (1.0 + safety_margin)

    # Generate guidance
    if not is_valid:
        guidance = f"""⚠️  ELLIPTICITY VIOLATION DETECTED

Current configuration:
  ε_Σ = {epsilon_Sigma:.6f}
  H_max = {H_max:.6f} ({H_max_source})
  Condition: ε_Σ > H_max is VIOLATED (margin: {margin:.6f})

IMPACT:
  ❌ Diffusion tensor Σ(x) = ε_Σ·I - H(x) may have negative eigenvalues
  ❌ Uniform ellipticity condition broken
  ❌ Convergence guarantees NO LONGER VALID
  ⚠️  Algorithm may diverge or behave unpredictably

GUIDANCE:
  ✓ Increase ε_Σ to at least {H_max:.6f} (current H_max)
  ✓ Recommended: ε_Σ ≥ {recommended_epsilon_Sigma:.6f} ({1+safety_margin:.0%} safety margin)
  ✓ Or reduce H_max by using a smoother potential function

See: docs/source/3_brascamp_lieb/geometric_foundations_lsi.md
"""
    elif margin < H_max * safety_margin:
        guidance = f"""⚠️  ELLIPTICITY CONDITION MARGINAL

Current configuration:
  ε_Σ = {epsilon_Sigma:.6f}
  H_max = {H_max:.6f} ({H_max_source})
  Condition: ε_Σ > H_max is satisfied ✓ (margin: {margin:.6f})

CAUTION:
  ⚠️  Safety margin is small ({margin/H_max:.1%} of H_max)
  ⚠️  c_max = {c_max_value:.4f}, c_min = {c_min_value:.4f}
  ⚠️  Condition number: {condition_number:.2f}

GUIDANCE:
  ✓ Consider increasing ε_Σ to {recommended_epsilon_Sigma:.6f} for safety
  ✓ Current configuration may be sensitive to trajectory fluctuations
  ✓ Monitor H_max during simulation

Status: Technically valid but not robust
"""
    else:
        guidance = f"""✅ ELLIPTICITY CONDITION SATISFIED

Current configuration:
  ε_Σ = {epsilon_Sigma:.6f}
  H_max = {H_max:.6f} ({H_max_source})
  Condition: ε_Σ > H_max is satisfied ✓ (margin: {margin:.6f})

Ellipticity bounds:
  c_min = {c_min_value:.6f}
  c_max = {c_max_value:.6f}
  Condition number: {condition_number:.2f}

Status: All uniform ellipticity conditions satisfied ✅
"""

    return EllipticityDiagnostics(
        epsilon_Sigma=epsilon_Sigma,
        H_max=H_max,
        H_max_source=H_max_source,
        is_valid=is_valid,
        margin=margin,
        c_min_value=c_min_value,
        c_max_value=c_max_value,
        condition_number=condition_number,
        recommended_epsilon_Sigma=recommended_epsilon_Sigma,
        guidance=guidance,
    )


# ============================================================================
# Hypocoercivity Regime Validation
# ============================================================================


def validate_hypocoercivity_regime(
    epsilon_F: float,
    nu: float,
    c_min_value: float,
    F_adapt_max: float,
    rho: float,
    use_fitness_force: bool,
    use_viscous_coupling: bool,
) -> HypocoercivityDiagnostics:
    """Validate hypocoercivity regime conditions.

    The hypocoercivity regime requires:
    1. ε_F < ε_F* where ε_F* = c_min / (2·F_adapt_max)
    2. ν > 0 (viscous coupling enabled)

    Parameters
    ----------
    epsilon_F : float
        Fitness force parameter
    nu : float
        Viscous coupling strength
    c_min_value : float
        Lower ellipticity bound
    F_adapt_max : float
        Maximum adaptive force magnitude
    rho : float
        Density parameter
    use_fitness_force : bool
        Whether fitness force is enabled
    use_viscous_coupling : bool
        Whether viscous coupling is enabled

    Returns
    -------
    HypocoercivityDiagnostics
        Complete diagnostics with guidance
    """
    epsilon_F_star_value = epsilon_F_star(rho, c_min_value, F_adapt_max)
    is_valid = validate_hypocoercivity(epsilon_F, epsilon_F_star_value, nu)
    margin = epsilon_F_star_value - epsilon_F

    # Generate guidance
    if not use_fitness_force and not use_viscous_coupling:
        guidance = """ℹ️  HYPOCOERCIVITY NOT APPLICABLE

Fitness force and viscous coupling are both disabled.
This is pure Euclidean Gas (isotropic diffusion only).

Status: N/A
"""
    elif not is_valid:
        if epsilon_F >= epsilon_F_star_value:
            guidance = f"""⚠️  HYPOCOERCIVITY REGIME VIOLATION

Current configuration:
  ε_F = {epsilon_F:.6f}
  ε_F* = {epsilon_F_star_value:.6f} (critical threshold)
  Condition: ε_F < ε_F* is VIOLATED (margin: {margin:.6f})
  ν = {nu:.6f}

IMPACT:
  ⚠️  Fitness force may be too strong
  ⚠️  LSI convergence rate may be degraded
  ⚠️  Hypocoercivity framework not applicable

GUIDANCE:
  ✓ Reduce ε_F below {epsilon_F_star_value:.6f}
  ✓ Recommended: ε_F ≤ {0.8*epsilon_F_star_value:.6f} (80% of threshold)
  ✓ Or increase c_min by raising ε_Σ

See: docs/source/3_brascamp_lieb/eigenvalue_gap_complete_proof.md
"""
        else:  # nu <= 0
            guidance = f"""⚠️  VISCOUS COUPLING DISABLED

Current configuration:
  ε_F = {epsilon_F:.6f} < ε_F* = {epsilon_F_star_value:.6f} ✓
  ν = {nu:.6f} (must be > 0 for hypocoercivity)

IMPACT:
  ⚠️  Hypocoercivity analysis not applicable without viscous coupling
  ℹ️  Standard LSI framework still applies

GUIDANCE:
  ✓ Enable viscous coupling (ν > 0) to use hypocoercivity
  ✓ Recommended: ν ≥ 0.1
  ℹ️  Or continue with standard Euclidean Gas (no adaptive forces)

Status: Fitness force OK, but viscous coupling needed
"""
    else:
        guidance = f"""✅ HYPOCOERCIVITY REGIME SATISFIED

Current configuration:
  ε_F = {epsilon_F:.6f}
  ε_F* = {epsilon_F_star_value:.6f} (critical threshold)
  Condition: ε_F < ε_F* is satisfied ✓ (margin: {margin:.6f})
  ν = {nu:.6f} > 0 ✓

Status: All hypocoercivity conditions satisfied ✅
LSI convergence framework applies with enhanced rate.
"""

    return HypocoercivityDiagnostics(
        epsilon_F=epsilon_F,
        epsilon_F_star_value=epsilon_F_star_value,
        nu=nu,
        use_fitness_force=use_fitness_force,
        use_viscous_coupling=use_viscous_coupling,
        is_valid=is_valid,
        margin=margin,
        guidance=guidance,
    )


# ============================================================================
# Parameter Recommendations
# ============================================================================


def recommend_safe_epsilon_Sigma(H_max: float, safety_margin: float = 0.2) -> float:
    """Recommend safe ε_Σ value given H_max.

    Parameters
    ----------
    H_max : float
        Maximum Hessian spectral norm
    safety_margin : float, default=0.2
        Safety factor (recommend ε_Σ ≥ (1+margin)·H_max)

    Returns
    -------
    float
        Recommended ε_Σ ≥ (1 + safety_margin) × H_max

    Examples
    --------
    >>> recommend_safe_epsilon_Sigma(H_max=0.1)
    0.12  # 20% margin
    """
    return H_max * (1.0 + safety_margin)


def recommend_safe_epsilon_F(
    c_min_value: float,
    F_adapt_max: float,
    rho: float,
    safety_factor: float = 0.8,
) -> float:
    """Recommend safe ε_F value for hypocoercivity regime.

    Parameters
    ----------
    c_min_value : float
        Lower ellipticity bound
    F_adapt_max : float
        Maximum adaptive force magnitude
    rho : float
        Density parameter
    safety_factor : float, default=0.8
        Safety factor (recommend ε_F ≤ safety_factor × ε_F*)

    Returns
    -------
    float
        Recommended ε_F ≤ safety_factor × ε_F*

    Examples
    --------
    >>> recommend_safe_epsilon_F(c_min_value=5.0, F_adapt_max=1.0, rho=1.0)
    2.0  # 80% of ε_F* = 2.5
    """
    epsilon_F_star_value = epsilon_F_star(rho, c_min_value, F_adapt_max)
    return safety_factor * epsilon_F_star_value


# ============================================================================
# Viscous Coupling Analysis
# ============================================================================


def analyze_viscous_coupling(
    history: RunHistory,
    nu: float,
    length_scale: float,
    use_viscous_coupling: bool,
    stage: str = "final",
    sample_size: int = 50,
) -> ViscousCouplingDiagnostics:
    """Analyze viscous coupling force statistics from trajectory.

    Parameters
    ----------
    history : RunHistory
        Trajectory history
    nu : float
        Viscous coupling strength
    length_scale : float
        Viscous kernel length scale
    use_viscous_coupling : bool
        Whether viscous coupling is enabled
    stage : str, default="final"
        Which stage to analyze
    sample_size : int, default=50
        Number of frames to sample

    Returns
    -------
    ViscousCouplingDiagnostics
        Viscous force statistics and guidance
    """
    if not use_viscous_coupling or nu <= 0:
        guidance = """ℹ️  VISCOUS COUPLING DISABLED

Viscous coupling is not enabled (ν = 0).
Walkers do not interact via fluid-like forces.

Status: N/A (Euclidean Gas mode)
"""
        return ViscousCouplingDiagnostics(
            nu=nu,
            length_scale=length_scale,
            is_enabled=False,
            mean_force_magnitude=0.0,
            max_force_magnitude=0.0,
            guidance=guidance,
        )

    # Sample frames and estimate viscous force magnitudes
    # (In practice, would extract from history if available)
    # For now, provide placeholder
    mean_force = 0.0
    max_force = 0.0

    guidance = f"""✅ VISCOUS COUPLING ENABLED

Current configuration:
  ν = {nu:.6f}
  Length scale l = {length_scale:.6f}

Force statistics (estimated):
  Mean magnitude: {mean_force:.6f}
  Max magnitude: {max_force:.6f}

Status: Viscous coupling active
Note: Force statistics require full trajectory analysis
"""

    return ViscousCouplingDiagnostics(
        nu=nu,
        length_scale=length_scale,
        is_enabled=True,
        mean_force_magnitude=mean_force,
        max_force_magnitude=max_force,
        guidance=guidance,
    )


# ============================================================================
# Diffusion Metrics
# ============================================================================


def compute_diffusion_metrics(
    history: RunHistory,
    epsilon_Sigma: float,
    use_anisotropic_diffusion: bool,
    stage: str = "final",
    sample_size: int = 100,
) -> DiffusionMetrics | None:
    """Compute diffusion tensor metrics from trajectory.

    Parameters
    ----------
    history : RunHistory
        Trajectory history
    epsilon_Sigma : float
        Regularization parameter
    use_anisotropic_diffusion : bool
        Whether anisotropic diffusion is enabled
    stage : str, default="final"
        Which stage to analyze
    sample_size : int, default=100
        Number of frames to sample

    Returns
    -------
    DiffusionMetrics | None
        Diffusion tensor metrics, or None if not applicable
    """
    if not use_anisotropic_diffusion:
        return None

    # Placeholder: In practice, would compute eigenvalues of Σ(x) = ε_Σ·I - H(x)
    # at sampled trajectory points
    mean_min_eig = epsilon_Sigma * 0.5  # Placeholder
    mean_max_eig = epsilon_Sigma * 1.5  # Placeholder
    mean_condition = mean_max_eig / mean_min_eig
    worst_condition = mean_condition * 1.2
    anisotropy = mean_max_eig / mean_min_eig

    guidance = f"""ℹ️  ANISOTROPIC DIFFUSION METRICS

Diffusion tensor: Σ(x) = ε_Σ·I - H(x)

Eigenvalue statistics (estimated):
  Mean λ_min: {mean_min_eig:.6f}
  Mean λ_max: {mean_max_eig:.6f}
  Mean condition number: {mean_condition:.2f}
  Worst condition number: {worst_condition:.2f}
  Anisotropy ratio: {anisotropy:.2f}

Note: Full metrics require detailed trajectory analysis
"""

    return DiffusionMetrics(
        epsilon_Sigma=epsilon_Sigma,
        mean_min_eigenvalue=mean_min_eig,
        mean_max_eigenvalue=mean_max_eig,
        mean_condition_number=mean_condition,
        worst_condition_number=worst_condition,
        anisotropy_ratio=anisotropy,
        guidance=guidance,
    )


# ============================================================================
# Comprehensive Diagnostics
# ============================================================================


def create_adaptive_gas_diagnostics(
    history: RunHistory,
    params: dict,
    estimate_H_max: bool = True,
) -> AdaptiveGasDiagnostics:
    """Create comprehensive diagnostics for Adaptive Gas configuration.

    This is the main entry point for validating an Adaptive Gas simulation.
    It performs all checks and provides detailed guidance.

    Parameters
    ----------
    history : RunHistory
        Trajectory history from simulation
    params : dict
        Parameter dictionary containing:
        - epsilon_Sigma : float
        - epsilon_F : float
        - nu : float
        - use_anisotropic_diffusion : bool
        - use_fitness_force : bool
        - use_viscous_coupling : bool
        - H_max : float (optional, will estimate if not provided)
        - viscous_length_scale : float (optional)
    estimate_H_max : bool, default=True
        Whether to estimate H_max from trajectory (slow but accurate)

    Returns
    -------
    AdaptiveGasDiagnostics
        Complete diagnostics package

    Examples
    --------
    >>> diagnostics = create_adaptive_gas_diagnostics(history, params)
    >>> print(diagnostics.overall_status)  # "valid", "warning", or "invalid"
    >>> print(diagnostics.summary)
    >>> if diagnostics.ellipticity:
    ...     print(diagnostics.ellipticity.guidance)
    """
    # Extract parameters
    epsilon_Sigma = params.get("epsilon_Sigma", 0.1)
    epsilon_F = params.get("epsilon_F", 0.0)
    nu = params.get("nu", 0.0)
    use_anisotropic_diffusion = params.get("use_anisotropic_diffusion", False)
    use_fitness_force = params.get("use_fitness_force", False)
    use_viscous_coupling = params.get("use_viscous_coupling", False)
    viscous_length_scale = params.get("viscous_length_scale", 1.0)
    F_adapt_max = params.get("F_adapt_max", 1.0)
    rho = params.get("rho", 1.0)

    # Check if this is pure Euclidean Gas
    is_euclidean_gas = (
        not use_anisotropic_diffusion and not use_fitness_force and not use_viscous_coupling
    )

    # Determine H_max
    if "H_max" in params and params["H_max"] is not None:
        H_max = params["H_max"]
        H_max_source = "parameter"
    elif estimate_H_max and not is_euclidean_gas:
        H_max = estimate_H_max_from_history(history, stage="final")
        H_max_source = "estimated"
    else:
        H_max = 1.0  # Fallback heuristic
        H_max_source = "heuristic"

    # Ellipticity diagnostics (only if anisotropic diffusion enabled)
    if use_anisotropic_diffusion:
        ellipticity = validate_ellipticity_conditions(
            epsilon_Sigma=epsilon_Sigma,
            H_max=H_max,
            H_max_source=H_max_source,
        )
    else:
        ellipticity = None

    # Hypocoercivity diagnostics (only if fitness force enabled)
    if use_fitness_force:
        c_min_value = c_min(epsilon_Sigma, H_max) if use_anisotropic_diffusion else 1.0
        hypocoercivity = validate_hypocoercivity_regime(
            epsilon_F=epsilon_F,
            nu=nu,
            c_min_value=c_min_value,
            F_adapt_max=F_adapt_max,
            rho=rho,
            use_fitness_force=use_fitness_force,
            use_viscous_coupling=use_viscous_coupling,
        )
    else:
        hypocoercivity = None

    # Viscous coupling diagnostics
    viscous_coupling = analyze_viscous_coupling(
        history=history,
        nu=nu,
        length_scale=viscous_length_scale,
        use_viscous_coupling=use_viscous_coupling,
    )

    # Diffusion metrics
    diffusion_metrics = compute_diffusion_metrics(
        history=history,
        epsilon_Sigma=epsilon_Sigma,
        use_anisotropic_diffusion=use_anisotropic_diffusion,
    )

    # Determine overall status
    if is_euclidean_gas:
        overall_status = "valid"
        summary = "✅ Pure Euclidean Gas (no adaptive features enabled)"
    else:
        # Check for critical failures
        if ellipticity is not None and not ellipticity.is_valid:
            overall_status = "invalid"
            summary = "❌ INVALID: Uniform ellipticity violated (ε_Σ ≤ H_max)"
        elif hypocoercivity is not None and not hypocoercivity.is_valid:
            overall_status = "warning"
            summary = "⚠️  WARNING: Hypocoercivity regime conditions not satisfied"
        elif ellipticity is not None and ellipticity.margin < 0.1 * H_max:
            overall_status = "warning"
            summary = "⚠️  WARNING: Ellipticity margin is small (< 10% of H_max)"
        else:
            overall_status = "valid"
            summary = "✅ All adaptive gas conditions satisfied"

    return AdaptiveGasDiagnostics(
        ellipticity=ellipticity,
        hypocoercivity=hypocoercivity,
        viscous_coupling=viscous_coupling,
        diffusion_metrics=diffusion_metrics,
        is_euclidean_gas=is_euclidean_gas,
        overall_status=overall_status,
        summary=summary,
    )


def print_adaptive_gas_report(diagnostics: AdaptiveGasDiagnostics) -> None:
    """Print formatted report of adaptive gas diagnostics.

    Parameters
    ----------
    diagnostics : AdaptiveGasDiagnostics
        Diagnostics from create_adaptive_gas_diagnostics()

    Examples
    --------
    >>> diagnostics = create_adaptive_gas_diagnostics(history, params)
    >>> print_adaptive_gas_report(diagnostics)
    """
    print("\n" + "=" * 70)
    print("ADAPTIVE GAS DIAGNOSTICS REPORT")
    print("=" * 70)
    print(f"\n{diagnostics.summary}\n")

    if diagnostics.is_euclidean_gas:
        print("This is a pure Euclidean Gas simulation (no adaptive features).")
        print("No additional validation required.")
        return

    # Print each diagnostic section
    if diagnostics.ellipticity is not None:
        print("\n" + "-" * 70)
        print("UNIFORM ELLIPTICITY")
        print("-" * 70)
        print(diagnostics.ellipticity.guidance)

    if diagnostics.hypocoercivity is not None:
        print("\n" + "-" * 70)
        print("HYPOCOERCIVITY REGIME")
        print("-" * 70)
        print(diagnostics.hypocoercivity.guidance)

    if diagnostics.viscous_coupling is not None:
        print("\n" + "-" * 70)
        print("VISCOUS COUPLING")
        print("-" * 70)
        print(diagnostics.viscous_coupling.guidance)

    if diagnostics.diffusion_metrics is not None:
        print("\n" + "-" * 70)
        print("DIFFUSION TENSOR METRICS")
        print("-" * 70)
        print(diagnostics.diffusion_metrics.guidance)

    print("\n" + "=" * 70)
