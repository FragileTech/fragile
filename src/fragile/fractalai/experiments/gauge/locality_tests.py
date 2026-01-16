"""Locality tests for collective field structure (Tests 1A-1C, 1E).

This module implements tests to verify local field theory structure by measuring:
- Test 1A: Spatial correlation function C(r)
- Test 1B: Field gradient magnitude |∇d'|
- Test 1C: Perturbation response locality
- Test 1E: Wave excitations

**Framework Reference:**
old_docs/source/13_fractal_set_new/04c_test_cases.md §1.2-§1.6

**Expected Results for Local Regime (ρ small):**
- Correlation length ξ ≈ ρ
- Field gradient |∇d'| ~ O(1/ρ)
- Perturbation response localized within ~3ρ
- Wave excitations propagate

**Expected Results for Mean-Field Regime (ρ large):**
- Correlation length ξ → ∞ (flat correlation)
- Field gradient |∇d'| ≈ 0
- Perturbation response uniform
- Exponential relaxation (no propagation)
"""

import numpy as np
from pydantic import BaseModel, Field
import torch
from torch import Tensor

from fragile.fractalai.experiments.gauge.observables import (
    bin_by_distance,
    compute_collective_fields,
    compute_field_gradients,
    fit_exponential_decay,
    ObservablesConfig,
)


class LocalityTestsConfig(BaseModel):
    """Configuration for locality tests.

    Attributes:
        r_max: Maximum distance for correlation analysis
        n_bins: Number of bins for distance histograms
        k_neighbors: Number of neighbors for gradient estimation
        perturbation_magnitude: Magnitude of single-walker perturbation
    """

    r_max: float = Field(default=0.5, gt=0, description="Max distance for correlation")
    n_bins: int = Field(default=50, gt=0, description="Number of distance bins")
    k_neighbors: int = Field(default=5, gt=0, description="Neighbors for gradient")
    perturbation_magnitude: float = Field(default=0.01, gt=0, description="Perturbation magnitude")


def test_spatial_correlation(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor,
    companions: Tensor,
    alive: Tensor,
    rho: float | None = None,
    obs_config: ObservablesConfig | None = None,
    test_config: LocalityTestsConfig | None = None,
) -> dict:
    """Test 1A: Measure spatial correlation function C(r) = ⟨d'_i · d'_j⟩.

    From §1.2 in:
    old_docs/source/13_fractal_set_new/04c_test_cases.md

    Expected:
    - Local regime: C(r) ~ C_0 · exp(-r²/ξ²) with ξ ≈ ρ
    - Mean-field: C(r) ≈ const

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N]
        companions: Diversity companion indices [N]
        alive: Boolean mask [N]
        rho: Localization scale
        obs_config: Observables configuration
        test_config: Test configuration

    Returns:
        Dictionary with keys:
            - "r": Distance bin centers
            - "C": Correlation values
            - "counts": Number of pairs per bin
            - "xi": Fitted correlation length
            - "C0": Fitted amplitude
            - "r_squared": Fit quality
            - "verdict": "local" or "mean-field"

    Example:
        >>> results = test_spatial_correlation(
        ...     positions, velocities, rewards, companions, alive, rho=0.05
        ... )
        >>> print(f"Correlation length: ξ = {results['xi']:.4f}")
        >>> print(f"Expected: ξ ≈ ρ = {0.05:.4f}")
    """
    if obs_config is None:
        obs_config = ObservablesConfig()
    if test_config is None:
        test_config = LocalityTestsConfig()

    # Compute collective fields
    fields = compute_collective_fields(
        positions, velocities, rewards, alive, companions, rho, obs_config
    )
    d_prime = fields["d_prime"]

    # Bin by distance
    r, C, counts = bin_by_distance(
        positions, d_prime, alive, test_config.r_max, test_config.n_bins
    )

    # Fit exponential decay
    fit_params = fit_exponential_decay(r, C, counts)

    # Determine verdict
    if rho is not None:
        # Local regime: expect ξ ≈ ρ
        xi_expected = rho
        xi_ratio = fit_params["xi"] / xi_expected if xi_expected > 0 else 0
        verdict = "local" if 0.5 < xi_ratio < 2.0 else "mean-field"
    else:
        # Mean-field regime: expect ξ → ∞ (flat correlation)
        verdict = "mean-field" if fit_params["xi"] > test_config.r_max else "local"

    return {
        "r": r,
        "C": C,
        "counts": counts,
        "xi": fit_params["xi"],
        "C0": fit_params["C0"],
        "r_squared": fit_params["r_squared"],
        "verdict": verdict,
        "rho": rho,
    }


def test_field_gradients(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor,
    companions: Tensor,
    alive: Tensor,
    rho: float | None = None,
    obs_config: ObservablesConfig | None = None,
    test_config: LocalityTestsConfig | None = None,
) -> dict:
    """Test 1B: Measure field gradient magnitude |∇d'|.

    From §1.3 in:
    old_docs/source/13_fractal_set_new/04c_test_cases.md

    Expected:
    - Local regime: |∇d'| ~ O(1/ρ)
    - Mean-field: |∇d'| ≈ 0

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N]
        companions: Diversity companion indices [N]
        alive: Boolean mask [N]
        rho: Localization scale
        obs_config: Observables configuration
        test_config: Test configuration

    Returns:
        Dictionary with keys:
            - "gradients": Gradient magnitudes [N]
            - "mean": Mean gradient magnitude
            - "std": Std gradient magnitude
            - "expected_scale": Expected scale (1/ρ if local, 0 if mean-field)
            - "verdict": "local" or "mean-field"

    Example:
        >>> results = test_field_gradients(
        ...     positions, velocities, rewards, companions, alive, rho=0.05
        ... )
        >>> print(f"|∇d'|_avg = {results['mean']:.4f}")
        >>> print(f"Expected: ~ {results['expected_scale']:.4f}")
    """
    if obs_config is None:
        obs_config = ObservablesConfig()
    if test_config is None:
        test_config = LocalityTestsConfig()

    # Compute collective fields
    fields = compute_collective_fields(
        positions, velocities, rewards, alive, companions, rho, obs_config
    )
    d_prime = fields["d_prime"]

    # Compute gradients
    gradients = compute_field_gradients(positions, d_prime, alive, test_config.k_neighbors)

    # Statistics
    grad_alive = gradients[alive]
    mean_grad = grad_alive.mean().item()
    std_grad = grad_alive.std().item()

    # Expected scale
    if rho is not None and rho > 0:
        expected_scale = 1.0 / rho
        # Verdict: mean gradient should be ~ 1/ρ for local
        ratio = mean_grad / expected_scale
        verdict = "local" if 0.1 < ratio < 10.0 else "mean-field"
    else:
        expected_scale = 0.0
        verdict = "mean-field" if mean_grad < 1.0 else "local"

    return {
        "gradients": gradients,
        "mean": mean_grad,
        "std": std_grad,
        "expected_scale": expected_scale,
        "verdict": verdict,
        "rho": rho,
    }


def test_perturbation_response(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor,
    companions: Tensor,
    alive: Tensor,
    perturb_idx: int,
    rho: float | None = None,
    obs_config: ObservablesConfig | None = None,
    test_config: LocalityTestsConfig | None = None,
) -> dict:
    """Test 1C: Measure perturbation response locality.

    From §1.4 in:
    old_docs/source/13_fractal_set_new/04c_test_cases.md

    Procedure:
    1. Compute baseline d'_i
    2. Perturb single walker k: x_k → x_k + δx
    3. Recompute d'_i
    4. Measure Δd'_i vs distance from k

    Expected:
    - Local regime: Δd'(r) ~ exp(-r²/ρ²), localized within ~3ρ
    - Mean-field: Δd' ≈ const (uniform response)

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N]
        companions: Diversity companion indices [N]
        alive: Boolean mask [N]
        perturb_idx: Index of walker to perturb
        rho: Localization scale
        obs_config: Observables configuration
        test_config: Test configuration

    Returns:
        Dictionary with keys:
            - "distances": Distances from perturbed walker [N]
            - "delta_d_prime": Change in d'_i [N]
            - "response_range": Distance at which Δd' drops to 1/e
            - "verdict": "local" or "mean-field"

    Example:
        >>> results = test_perturbation_response(
        ...     positions, velocities, rewards, companions, alive, perturb_idx=50, rho=0.05
        ... )
        >>> print(f"Response range: {results['response_range']:.4f}")
        >>> print(f"Expected: ~{3 * 0.05:.4f}")
    """
    if obs_config is None:
        obs_config = ObservablesConfig()
    if test_config is None:
        test_config = LocalityTestsConfig()

    # Baseline collective fields
    fields_baseline = compute_collective_fields(
        positions, velocities, rewards, alive, companions, rho, obs_config
    )
    d_prime_baseline = fields_baseline["d_prime"]

    # Perturb single walker
    positions_perturbed = positions.clone()
    perturbation = torch.randn_like(positions[perturb_idx]) * test_config.perturbation_magnitude
    positions_perturbed[perturb_idx] += perturbation

    # Recompute collective fields with perturbation
    fields_perturbed = compute_collective_fields(
        positions_perturbed, velocities, rewards, alive, companions, rho, obs_config
    )
    d_prime_perturbed = fields_perturbed["d_prime"]

    # Measure change
    delta_d_prime = torch.abs(d_prime_perturbed - d_prime_baseline)

    # Compute distances from perturbed walker
    distances = torch.sqrt(((positions - positions[perturb_idx]) ** 2).sum(dim=1))

    # Estimate response range (distance where Δd' drops to 1/e of max)
    delta_max = delta_d_prime[alive].max().item()
    threshold = delta_max / np.e

    distances_alive = distances[alive].cpu().numpy()
    delta_alive = delta_d_prime[alive].cpu().numpy()

    # Find response range
    if delta_max > 1e-10:
        beyond_threshold = delta_alive < threshold
        if beyond_threshold.any():
            response_range = distances_alive[beyond_threshold].min()
        else:
            response_range = distances_alive.max()
    else:
        response_range = 0.0

    # Verdict
    if rho is not None and rho > 0:
        expected_range = 3 * rho
        ratio = response_range / expected_range
        verdict = "local" if 0.5 < ratio < 2.0 else "mean-field"
    else:
        # Mean-field: expect large response range
        verdict = "mean-field" if response_range > test_config.r_max else "local"

    return {
        "distances": distances,
        "delta_d_prime": delta_d_prime,
        "response_range": response_range,
        "verdict": verdict,
        "rho": rho,
        "perturb_idx": perturb_idx,
    }


def run_all_locality_tests(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor,
    companions: Tensor,
    alive: Tensor,
    rho: float | None = None,
    obs_config: ObservablesConfig | None = None,
    test_config: LocalityTestsConfig | None = None,
) -> dict:
    """Run complete battery of locality tests (1A-1C).

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N]
        companions: Diversity companion indices [N]
        alive: Boolean mask [N]
        rho: Localization scale
        obs_config: Observables configuration
        test_config: Test configuration

    Returns:
        Dictionary with keys "test_1a", "test_1b", "test_1c", "summary"

    Example:
        >>> results = run_all_locality_tests(
        ...     positions, velocities, rewards, companions, alive, rho=0.05
        ... )
        >>> print(f"Test 1A: {results['test_1a']['verdict']}")
        >>> print(f"Test 1B: {results['test_1b']['verdict']}")
        >>> print(f"Test 1C: {results['test_1c']['verdict']}")
        >>> print(f"Overall: {results['summary']['verdict']}")
    """
    # Test 1A: Spatial correlation
    test_1a = test_spatial_correlation(
        positions, velocities, rewards, companions, alive, rho, obs_config, test_config
    )

    # Test 1B: Field gradients
    test_1b = test_field_gradients(
        positions, velocities, rewards, companions, alive, rho, obs_config, test_config
    )

    # Test 1C: Perturbation response (perturb a random alive walker)
    alive_indices = torch.where(alive)[0]
    perturb_idx = alive_indices[len(alive_indices) // 2].item()  # Middle walker
    test_1c = test_perturbation_response(
        positions,
        velocities,
        rewards,
        companions,
        alive,
        perturb_idx,
        rho,
        obs_config,
        test_config,
    )

    # Summary
    verdicts = [test_1a["verdict"], test_1b["verdict"], test_1c["verdict"]]
    local_count = sum(1 for v in verdicts if v == "local")
    overall_verdict = "local" if local_count >= 2 else "mean-field"

    summary = {
        "verdict": overall_verdict,
        "test_1a_verdict": test_1a["verdict"],
        "test_1b_verdict": test_1b["verdict"],
        "test_1c_verdict": test_1c["verdict"],
        "rho": rho,
        "correlation_length": test_1a["xi"],
        "gradient_mean": test_1b["mean"],
        "response_range": test_1c["response_range"],
    }

    return {
        "test_1a": test_1a,
        "test_1b": test_1b,
        "test_1c": test_1c,
        "summary": summary,
    }


def generate_locality_report(
    results: dict,
    rho: float | None = None,
) -> str:
    """Generate text report from locality test results.

    Args:
        results: Output from run_all_locality_tests()
        rho: Localization scale used

    Returns:
        Formatted text report

    Example:
        >>> results = run_all_locality_tests(...)
        >>> report = generate_locality_report(results, rho=0.05)
        >>> print(report)
    """
    summary = results["summary"]
    test_1a = results["test_1a"]
    test_1b = results["test_1b"]
    test_1c = results["test_1c"]

    return f"""
================================================================================
LOCALITY TESTS REPORT
================================================================================

Configuration:
  ρ (localization scale): {rho if rho is not None else "∞ (mean-field)"}

Overall Verdict: {summary["verdict"].upper()}

--------------------------------------------------------------------------------
Test 1A: Spatial Correlation Function
--------------------------------------------------------------------------------
  Verdict: {test_1a["verdict"]}
  Correlation length (ξ): {test_1a["xi"]:.6f}
  Expected (ρ): {rho if rho is not None else "N/A"}
  Fit quality (R²): {test_1a["r_squared"]:.4f}

  Interpretation:
    {_interpret_correlation_test(test_1a, rho)}

--------------------------------------------------------------------------------
Test 1B: Field Gradient Magnitude
--------------------------------------------------------------------------------
  Verdict: {test_1b["verdict"]}
  Mean |∇d'|: {test_1b["mean"]:.4f}
  Expected scale: {test_1b["expected_scale"]:.4f}
  Std |∇d'|: {test_1b["std"]:.4f}

  Interpretation:
    {_interpret_gradient_test(test_1b, rho)}

--------------------------------------------------------------------------------
Test 1C: Perturbation Response
--------------------------------------------------------------------------------
  Verdict: {test_1c["verdict"]}
  Response range: {test_1c["response_range"]:.6f}
  Expected (3ρ): {3 * rho if rho is not None else "N/A"}
  Perturbed walker: {test_1c["perturb_idx"]}

  Interpretation:
    {_interpret_perturbation_test(test_1c, rho)}

================================================================================
CONCLUSION
================================================================================

{_generate_conclusion(summary, rho)}

================================================================================
"""


def _interpret_correlation_test(test_1a: dict, rho: float | None) -> str:
    """Generate interpretation text for correlation test."""
    if rho is not None:
        ratio = test_1a["xi"] / rho if rho > 0 else 0
        if 0.5 < ratio < 2.0:
            return f"✓ Correlation length ξ ≈ ρ (ratio: {ratio:.2f}). Local field structure confirmed."
        return f"✗ Correlation length ξ ≠ ρ (ratio: {ratio:.2f}). Does not match local regime."
    if test_1a["xi"] > test_1a.get("r_max", 0.5):
        return "✓ Large correlation length. Consistent with mean-field regime."
    return "✗ Finite correlation length in mean-field regime (unexpected)."


def _interpret_gradient_test(test_1b: dict, rho: float | None) -> str:
    """Generate interpretation text for gradient test."""
    if rho is not None and rho > 0:
        ratio = test_1b["mean"] / test_1b["expected_scale"]
        if 0.1 < ratio < 10.0:
            return f"✓ Gradient magnitude |∇d'| ~ 1/ρ (ratio: {ratio:.2f}). Local field confirmed."
        return f"✗ Gradient magnitude deviates from 1/ρ (ratio: {ratio:.2f})."
    if test_1b["mean"] < 1.0:
        return "✓ Weak gradients. Consistent with mean-field regime."
    return "✗ Strong gradients in mean-field regime (unexpected)."


def _interpret_perturbation_test(test_1c: dict, rho: float | None) -> str:
    """Generate interpretation text for perturbation test."""
    if rho is not None and rho > 0:
        expected = 3 * rho
        ratio = test_1c["response_range"] / expected
        if 0.5 < ratio < 2.0:
            return f"✓ Response localized within ~3ρ (ratio: {ratio:.2f}). Local field confirmed."
        return f"✗ Response range differs from 3ρ (ratio: {ratio:.2f})."
    if test_1c["response_range"] > 0.5:
        return "✓ Long-range response. Consistent with mean-field regime."
    return "✗ Short-range response in mean-field regime (unexpected)."


def _generate_conclusion(summary: dict, rho: float | None) -> str:
    """Generate conclusion text."""
    if summary["verdict"] == "local":
        return f"""The collective field d'_i exhibits LOCAL FIELD THEORY structure:
  - Correlation length ξ ≈ ρ = {rho if rho is not None else "N/A"}
  - Field gradients |∇d'| ~ O(1/ρ)
  - Perturbation response localized within ~3ρ

This supports the interpretation of d'_i as a genuine local field rather than
a mean-field auxiliary variable. Gauge covariance test (Test 1D) should be
performed to determine if local GAUGE theory applies."""
    return """The collective field d'_i exhibits MEAN-FIELD structure:
  - Flat or very long-range correlations
  - Weak field gradients
  - Non-local perturbation response

This is consistent with mean-field interpretation where d'_i is an auxiliary
collective variable rather than a local field. Local gauge theory interpretation
is unlikely in this parameter regime."""
