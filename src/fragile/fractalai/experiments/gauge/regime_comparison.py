"""Regime comparison and crossover analysis (mean-field ↔ local).

This module provides tools for comparing mean-field and local regimes by scanning
the locality parameter ρ to map the phase diagram and identify critical scales.

**Framework Reference:**
old_docs/source/13_fractal_set_new/04c_test_cases.md - Test Case 3 (Crossover)

**Key Questions:**
1. At what ρ/L does local → mean-field transition occur?
2. Is the transition sharp (phase transition) or smooth (crossover)?
3. Does gauge structure emerge/disappear at critical ρ_c?

**Observables vs ρ:**
- Correlation length ξ(ρ)
- Field gradient |∇d'|(ρ)
- Perturbation response range R_resp(ρ)
- Fitness variance σ²(V_fit)(ρ)
"""

import numpy as np
from pydantic import BaseModel, Field
from torch import Tensor

from fragile.fractalai.experiments.gauge.locality_tests import (
    LocalityTestsConfig,
    test_field_gradients,
    test_spatial_correlation,
)
from fragile.fractalai.experiments.gauge.observables import (
    ObservablesConfig,
)


class RegimeComparisonConfig(BaseModel):
    """Configuration for regime comparison scans.

    Attributes:
        rho_values: List of ρ values to scan
        num_samples: Number of random configurations per ρ
    """

    rho_values: list[float] = Field(
        default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        description="ρ values for crossover scan",
    )
    num_samples: int = Field(default=5, gt=0, description="Samples per ρ value")


def scan_correlation_length(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor,
    companions: Tensor,
    alive: Tensor,
    rho_values: list[float],
    obs_config: ObservablesConfig | None = None,
    test_config: LocalityTestsConfig | None = None,
) -> dict:
    """Scan correlation length ξ(ρ) across parameter space.

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N]
        companions: Diversity companion indices [N]
        alive: Boolean mask [N]
        rho_values: List of ρ values to scan
        obs_config: Observables configuration
        test_config: Test configuration

    Returns:
        Dictionary with keys:
            - "rho_values": ρ values scanned
            - "xi_values": Correlation lengths ξ(ρ)
            - "fit_quality": R² values for each fit

    Example:
        >>> results = scan_correlation_length(
        ...     positions,
        ...     velocities,
        ...     rewards,
        ...     companions,
        ...     alive,
        ...     rho_values=[0.01, 0.05, 0.1, 0.5, None],
        ... )
    """
    xi_values = []
    fit_qualities = []

    # Include mean-field (ρ=None) at end
    rho_scan = [*list(rho_values), None]

    for rho in rho_scan:
        result = test_spatial_correlation(
            positions, velocities, rewards, companions, alive, rho, obs_config, test_config
        )
        xi_values.append(result["xi"])
        fit_qualities.append(result["r_squared"])

    return {
        "rho_values": rho_scan,
        "xi_values": np.array(xi_values),
        "fit_quality": np.array(fit_qualities),
    }


def scan_field_gradients(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor,
    companions: Tensor,
    alive: Tensor,
    rho_values: list[float],
    obs_config: ObservablesConfig | None = None,
    test_config: LocalityTestsConfig | None = None,
) -> dict:
    """Scan field gradient |∇d'|(ρ) across parameter space.

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N]
        companions: Diversity companion indices [N]
        alive: Boolean mask [N]
        rho_values: List of ρ values to scan
        obs_config: Observables configuration
        test_config: Test configuration

    Returns:
        Dictionary with keys:
            - "rho_values": ρ values scanned
            - "gradient_mean": Mean |∇d'| for each ρ
            - "gradient_std": Std |∇d'| for each ρ

    Example:
        >>> results = scan_field_gradients(
        ...     positions,
        ...     velocities,
        ...     rewards,
        ...     companions,
        ...     alive,
        ...     rho_values=[0.01, 0.05, 0.1, 0.5, None],
        ... )
    """
    gradient_means = []
    gradient_stds = []

    rho_scan = [*list(rho_values), None]

    for rho in rho_scan:
        result = test_field_gradients(
            positions, velocities, rewards, companions, alive, rho, obs_config, test_config
        )
        gradient_means.append(result["mean"])
        gradient_stds.append(result["std"])

    return {
        "rho_values": rho_scan,
        "gradient_mean": np.array(gradient_means),
        "gradient_std": np.array(gradient_stds),
    }


def identify_critical_scale(
    rho_values: np.ndarray,
    observable_values: np.ndarray,
    threshold: float = 0.5,
) -> float | None:
    """Identify critical scale ρ_c where observable transitions.

    Uses simple threshold crossing to estimate ρ_c. For more sophisticated
    analysis, use phase transition detection methods.

    Args:
        rho_values: ρ values (must be sorted)
        observable_values: Observable O(ρ) values
        threshold: Threshold value for transition detection

    Returns:
        Critical ρ_c where O crosses threshold, or None if no transition

    Example:
        >>> # Normalize ξ(ρ) and find where it drops to 50% of max
        >>> xi_norm = xi_values / xi_values.max()
        >>> rho_c = identify_critical_scale(rho_values, xi_norm, threshold=0.5)
    """
    # Filter out None values
    valid = np.array([r is not None for r in rho_values])
    rho_finite = np.array([r for r in rho_values if r is not None])
    obs_finite = observable_values[valid]

    if len(rho_finite) < 2:
        return None

    # Find threshold crossing
    above_threshold = obs_finite > threshold
    if not above_threshold.any() or above_threshold.all():
        return None  # No crossing

    # Find first crossing
    crossing_idx = np.where(~above_threshold)[0][0]
    if crossing_idx == 0:
        return None

    # Linear interpolation between points
    rho_before = rho_finite[crossing_idx - 1]
    rho_after = rho_finite[crossing_idx]
    obs_before = obs_finite[crossing_idx - 1]
    obs_after = obs_finite[crossing_idx]

    # Interpolate
    if obs_before != obs_after:
        frac = (threshold - obs_after) / (obs_before - obs_after)
        rho_c = rho_after + frac * (rho_before - rho_after)
    else:
        rho_c = (rho_before + rho_after) / 2

    return float(rho_c)


def compare_regimes(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor,
    companions: Tensor,
    alive: Tensor,
    obs_config: ObservablesConfig | None = None,
    test_config: LocalityTestsConfig | None = None,
    regime_config: RegimeComparisonConfig | None = None,
) -> dict:
    """Compare mean-field vs local regimes and scan crossover.

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N]
        companions: Diversity companion indices [N]
        alive: Boolean mask [N]
        obs_config: Observables configuration
        test_config: Test configuration
        regime_config: Regime comparison configuration

    Returns:
        Dictionary with comprehensive regime comparison data

    Example:
        >>> results = compare_regimes(positions, velocities, rewards, companions, alive)
        >>> print(f"Critical scale: ρ_c ≈ {results['rho_c_correlation']:.4f}")
    """
    if regime_config is None:
        regime_config = RegimeComparisonConfig()

    # Scan correlation length
    correlation_scan = scan_correlation_length(
        positions,
        velocities,
        rewards,
        companions,
        alive,
        regime_config.rho_values,
        obs_config,
        test_config,
    )

    # Scan field gradients
    gradient_scan = scan_field_gradients(
        positions,
        velocities,
        rewards,
        companions,
        alive,
        regime_config.rho_values,
        obs_config,
        test_config,
    )

    # Identify critical scales
    xi_values = correlation_scan["xi_values"]
    xi_norm = xi_values / (xi_values.max() + 1e-10)
    rho_c_correlation = identify_critical_scale(
        correlation_scan["rho_values"], xi_norm, threshold=0.5
    )

    gradient_values = gradient_scan["gradient_mean"]
    grad_norm = gradient_values / (gradient_values.max() + 1e-10)
    rho_c_gradient = identify_critical_scale(
        gradient_scan["rho_values"], 1 - grad_norm, threshold=0.5
    )

    return {
        "correlation_scan": correlation_scan,
        "gradient_scan": gradient_scan,
        "rho_c_correlation": rho_c_correlation,
        "rho_c_gradient": rho_c_gradient,
        "rho_values_scanned": regime_config.rho_values,
    }


def generate_regime_comparison_report(results: dict) -> str:
    """Generate report from regime comparison analysis.

    Args:
        results: Output from compare_regimes()

    Returns:
        Formatted text report
    """
    rho_c_corr = results.get("rho_c_correlation")
    rho_c_grad = results.get("rho_c_gradient")

    # Format critical scale values (extract conditionals outside f-string)
    rho_c_corr_str = f"{rho_c_corr:.6f}" if rho_c_corr is not None else "N/A"
    rho_c_grad_str = f"{rho_c_grad:.6f}" if rho_c_grad is not None else "N/A"

    return f"""
================================================================================
REGIME COMPARISON REPORT
================================================================================

Scanned ρ values: {results["rho_values_scanned"]}

--------------------------------------------------------------------------------
Critical Scales:
--------------------------------------------------------------------------------

From correlation length: ρ_c ≈ {rho_c_corr_str}
From field gradients:    ρ_c ≈ {rho_c_grad_str}

{_interpret_critical_scales(rho_c_corr, rho_c_grad)}

--------------------------------------------------------------------------------
Regime Classification:
--------------------------------------------------------------------------------

{_classify_regime(results)}

================================================================================
"""


def _interpret_critical_scales(rho_c_corr: float | None, rho_c_grad: float | None) -> str:
    """Interpret critical scale findings."""
    if rho_c_corr is not None and rho_c_grad is not None:
        if abs(rho_c_corr - rho_c_grad) < 0.1:
            return f"""✓ Both observables agree on critical scale: ρ_c ≈ {(rho_c_corr + rho_c_grad) / 2:.4f}

This suggests a genuine crossover between local and mean-field regimes.
For ρ < ρ_c: Local field theory
For ρ > ρ_c: Mean-field theory"""
        return f"""Observables give different critical scales:
  Correlation: ρ_c ≈ {rho_c_corr:.4f}
  Gradient:    ρ_c ≈ {rho_c_grad:.4f}

This suggests different aspects transition at different scales."""
    if rho_c_corr is not None:
        return f"Critical scale from correlation: ρ_c ≈ {rho_c_corr:.4f}"
    if rho_c_grad is not None:
        return f"Critical scale from gradient: ρ_c ≈ {rho_c_grad:.4f}"
    return "No clear critical scale detected. System may be in single regime across scan range."


def _classify_regime(results: dict) -> str:
    """Classify the observed regime structure."""
    rho_c = results.get("rho_c_correlation")

    if rho_c is None:
        return "Unable to identify clear regime structure from scan."

    return f"""For the scanned parameter range:

  LOCAL REGIME:      ρ < {rho_c:.4f}
    - Finite correlation length ξ ≈ ρ
    - Strong field gradients |∇d'| ~ 1/ρ
    - Localized perturbation response
    - Gauge covariance test applicable

  MEAN-FIELD REGIME: ρ > {rho_c:.4f}
    - Diverging correlation length
    - Weak field gradients
    - Non-local perturbation response
    - Gauge invariance expected

  CROSSOVER:         ρ ≈ {rho_c:.4f}
    - Most interesting physics
    - Emergent gauge structure?
    - Study this region in detail"""
