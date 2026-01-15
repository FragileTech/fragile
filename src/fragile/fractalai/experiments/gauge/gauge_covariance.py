"""Gauge covariance test (Test 1D - CRITICAL EXPERIMENT).

This module implements the definitive test that determines whether the collective
field structure supports local gauge theory or operates as gauge-invariant
mean-field variables.

**Framework Reference:**
old_docs/source/13_fractal_set_new/04c_test_cases.md §1.5

**Research Question:**
Does d'_i transform non-trivially under local U(1) fitness-phase transformation α_i(x)?

**Test Procedure:**
1. Define local gauge transformation on region R
2. Apply phase shift α_i to walkers in R
3. Modify companion selection probabilities: P'(k|i) ∝ P(k|i) · exp(i(α_i-α_k)/ℏ_eff)
4. Recompute d'_i with transformed probabilities
5. Measure change Δd'_i inside/outside/boundary of R

**Expected Outcomes:**
- **GAUGE COVARIANT**: Δd' ~ O(α) locally → Local gauge theory viable ✓✓✓
- **GAUGE INVARIANT**: Δd' ≈ 0 everywhere → Mean-field interpretation

**THIS IS THE CRITICAL TEST THAT DETERMINES THE INTERPRETATION!**
"""

from pydantic import BaseModel, Field
import torch
from torch import Tensor

from fragile.core.companion_selection import compute_algorithmic_distance_matrix
from fragile.experiments.gauge.observables import (
    compute_collective_fields,
    ObservablesConfig,
)


class GaugeTransformConfig(BaseModel):
    """Configuration for gauge transformation test.

    Attributes:
        h_eff: Effective Planck constant (ℏ_eff)
        alpha_0: Phase shift magnitude (α_0)
        epsilon_d: Diversity companion range
        lambda_alg: Velocity weight in distance
        region_center: Center of transformation region [x_min, x_max, y_min, y_max]
        boundary_width: Width of boundary region for analysis
    """

    h_eff: float = Field(default=1.0, gt=0, description="Effective Planck constant")
    alpha_0: float = Field(default=1.0, description="Phase shift magnitude")
    epsilon_d: float = Field(default=0.1, gt=0, description="Diversity companion range")
    lambda_alg: float = Field(default=0.0, ge=0, description="Velocity weight")
    region_center: tuple[float, float, float, float] = Field(
        default=(0.4, 0.6, 0.4, 0.6),
        description="Region bounds [x_min, x_max, y_min, y_max]",
    )
    boundary_width: float = Field(default=0.05, gt=0, description="Boundary analysis width")


def define_gauge_transformation_region(
    positions: Tensor,
    region_bounds: tuple[float, float, float, float],
) -> tuple[Tensor, Tensor, Tensor]:
    """Define spatial regions for gauge transformation test.

    Args:
        positions: Walker positions [N, 2]
        region_bounds: (x_min, x_max, y_min, y_max) defining region R

    Returns:
        Tuple of (inside_mask, boundary_mask, outside_mask):
            - inside_mask: Walkers inside region R [N]
            - boundary_mask: Walkers at boundary of R [N]
            - outside_mask: Walkers outside R [N]

    Example:
        >>> positions = torch.rand(1000, 2)  # Unit square
        >>> inside, boundary, outside = define_gauge_transformation_region(
        ...     positions, (0.4, 0.6, 0.4, 0.6)
        ... )
        >>> print(f"Inside: {inside.sum()}, Outside: {outside.sum()}")
    """
    x_min, x_max, y_min, y_max = region_bounds

    # Inside region R
    inside = (
        (positions[:, 0] >= x_min)
        & (positions[:, 0] <= x_max)
        & (positions[:, 1] >= y_min)
        & (positions[:, 1] <= y_max)
    )

    # Compute distance to region boundary
    # Distance to nearest edge
    dist_to_x_min = torch.abs(positions[:, 0] - x_min)
    dist_to_x_max = torch.abs(positions[:, 0] - x_max)
    dist_to_y_min = torch.abs(positions[:, 1] - y_min)
    dist_to_y_max = torch.abs(positions[:, 1] - y_max)

    dist_to_boundary = torch.min(
        torch.min(dist_to_x_min, dist_to_x_max),
        torch.min(dist_to_y_min, dist_to_y_max),
    )

    # Boundary: within boundary_width of edge
    boundary_width = 0.05  # Default
    boundary = inside & (dist_to_boundary < boundary_width)

    # Outside
    outside = ~inside

    return inside, boundary, outside


def apply_gauge_transformation_to_phases(
    positions: Tensor,
    alpha_0: float,
    region_mask: Tensor,
) -> Tensor:
    """Apply local gauge transformation α_i to walkers in region.

    α_i = α_0 for i ∈ R
    α_i = 0 for i ∉ R

    The fitness phase transforms as θ_i → θ_i + α_i.

    Args:
        positions: Walker positions [N, d]
        alpha_0: Phase shift magnitude
        region_mask: Boolean mask [N] indicating walkers in region R

    Returns:
        Phase shifts [N] where α[i] = α_0 if in R, else 0

    Example:
        >>> alpha = apply_gauge_transformation_to_phases(
        ...     positions, alpha_0=1.0, region_mask=inside_mask
        ... )
    """
    N = positions.shape[0]
    alpha = torch.zeros(N, device=positions.device, dtype=positions.dtype)
    alpha[region_mask] = alpha_0
    return alpha


def modify_companion_probabilities_with_gauge(
    positions: Tensor,
    velocities: Tensor,
    alive: Tensor,
    alpha: Tensor,
    config: GaugeTransformConfig,
) -> Tensor:
    """Modify companion selection probabilities with gauge transformation.

    P'(k|i) ∝ P(k|i) · exp(i(α_i - α_k) / ℏ_eff)

    From §1.5 (Step 1) in:
    old_docs/source/13_fractal_set_new/04c_test_cases.md

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        alive: Boolean mask [N]
        alpha: Phase shifts [N]
        config: Gauge transformation configuration

    Returns:
        Modified probability matrix [N, N] where P'[i,j] = P'(j|i)

    Note:
        The phase factor exp(i(α_i - α_k)/ℏ_eff) modulates the complex amplitudes.
        Taking |·|² cancels the phase, so P'(k|i) ∝ P(k|i) in pure probability form.

        Therefore, for this test, we use the phase directly in the companion
        selection (not just the probability).
    """
    N = positions.shape[0]

    # Compute base probabilities
    dist_sq = compute_algorithmic_distance_matrix(positions, velocities, config.lambda_alg)
    weights = torch.exp(-dist_sq / (2 * config.epsilon_d**2))

    # Apply gauge-dependent phase modulation
    # Phase difference matrix: (α_i - α_k) [N, N]
    alpha_diff = alpha.unsqueeze(1) - alpha.unsqueeze(0)  # [N, N]

    # Apply phase modulation to weights
    # For real probabilities, we use: w' = w · cos((α_i - α_k)/ℏ_eff)
    # Or we could use: w' = w · exp(phase_factor) where phase affects the real part
    phase_factor = alpha_diff / config.h_eff
    weights_transformed = weights * torch.cos(phase_factor)

    # Ensure non-negative
    weights_transformed = torch.clamp(weights_transformed, min=0.0)

    # Mask dead walkers and self-pairing
    alive_mask = alive.unsqueeze(1) & alive.unsqueeze(0)
    self_mask = ~torch.eye(N, device=alive.device, dtype=torch.bool)
    weights_transformed = weights_transformed * alive_mask.float() * self_mask.float()

    # Normalize to probabilities
    return weights_transformed / (weights_transformed.sum(dim=1, keepdim=True) + 1e-10)


def sample_companions_from_modified_probabilities(
    probs_transformed: Tensor,
    alive: Tensor,
) -> Tensor:
    """Sample new companions from gauge-transformed probabilities.

    Args:
        probs_transformed: Modified probability matrix [N, N]
        alive: Boolean mask [N]

    Returns:
        Companion indices [N]

    Example:
        >>> companions_transformed = sample_companions_from_modified_probabilities(
        ...     probs_transformed, alive
        ... )
    """
    N = probs_transformed.shape[0]
    companions = torch.zeros(N, dtype=torch.long, device=probs_transformed.device)

    for i in range(N):
        if alive[i]:
            # Sample from transformed probability distribution
            companions[i] = torch.multinomial(probs_transformed[i], num_samples=1).item()
        else:
            companions[i] = 0  # Dead walkers (value doesn't matter)

    return companions


def test_gauge_covariance(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor,
    companions_baseline: Tensor,
    alive: Tensor,
    rho: float | None = None,
    obs_config: ObservablesConfig | None = None,
    gauge_config: GaugeTransformConfig | None = None,
    num_trials: int = 10,
) -> dict:
    """Test 1D: Gauge covariance test (CRITICAL EXPERIMENT).

    From §1.5 in:
    old_docs/source/13_fractal_set_new/04c_test_cases.md

    This test determines whether d'_i is gauge-covariant or gauge-invariant.

    Args:
        positions: Walker positions [N, 2]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N]
        companions_baseline: Baseline companion indices [N]
        alive: Boolean mask [N]
        rho: Localization scale
        obs_config: Observables configuration
        gauge_config: Gauge transformation configuration
        num_trials: Number of sampling trials for statistical significance

    Returns:
        Dictionary with keys:
            - "baseline_d_prime": Baseline d'_i [N]
            - "transformed_d_prime": Transformed d'_i [N] (averaged over trials)
            - "delta_d_prime": |transformed - baseline| [N]
            - "delta_inside": Mean Δd' inside region
            - "delta_boundary": Mean Δd' at boundary
            - "delta_outside": Mean Δd' outside region
            - "alpha_0": Phase shift magnitude used
            - "verdict": "covariant" or "invariant"
            - "confidence": Statistical confidence measure

    Example:
        >>> results = test_gauge_covariance(
        ...     positions, velocities, rewards, companions, alive, rho=0.05
        ... )
        >>> print(f"Verdict: {results['verdict']}")
        >>> print(f"Δd'_inside: {results['delta_inside']:.6f}")
        >>> if results["verdict"] == "covariant":
        ...     print("✓✓✓ LOCAL GAUGE THEORY VIABLE!")
        ... else:
        ...     print("Mean-field interpretation applies.")
    """
    if obs_config is None:
        obs_config = ObservablesConfig()
    if gauge_config is None:
        gauge_config = GaugeTransformConfig()

    # Compute baseline collective fields
    fields_baseline = compute_collective_fields(
        positions, velocities, rewards, alive, companions_baseline, rho, obs_config
    )
    d_prime_baseline = fields_baseline["d_prime"]

    # Define transformation region
    inside, boundary, outside = define_gauge_transformation_region(
        positions, gauge_config.region_center
    )

    # Apply gauge transformation
    alpha = apply_gauge_transformation_to_phases(positions, gauge_config.alpha_0, inside)

    # Run multiple trials with resampling
    d_prime_transformed_trials = []

    for trial in range(num_trials):
        # Modify companion probabilities with gauge transformation
        probs_transformed = modify_companion_probabilities_with_gauge(
            positions, velocities, alive, alpha, gauge_config
        )

        # Sample new companions
        companions_transformed = sample_companions_from_modified_probabilities(
            probs_transformed, alive
        )

        # Recompute collective fields with transformed companions
        fields_transformed = compute_collective_fields(
            positions,
            velocities,
            rewards,
            alive,
            companions_transformed,
            rho,
            obs_config,
        )
        d_prime_transformed_trials.append(fields_transformed["d_prime"])

    # Average over trials
    d_prime_transformed = torch.stack(d_prime_transformed_trials).mean(dim=0)

    # Measure change
    delta_d_prime = torch.abs(d_prime_transformed - d_prime_baseline)

    # Compute region-wise statistics
    delta_inside = delta_d_prime[inside & alive].mean().item() if (inside & alive).any() else 0.0
    delta_boundary = (
        delta_d_prime[boundary & alive].mean().item() if (boundary & alive).any() else 0.0
    )
    delta_outside = (
        delta_d_prime[outside & alive].mean().item() if (outside & alive).any() else 0.0
    )

    # Determine verdict
    # Covariant: Δd' ~ O(α) inside, negligible outside
    # Invariant: Δd' ≈ 0 everywhere
    threshold_covariant = 0.1 * gauge_config.alpha_0  # 10% of α_0
    threshold_invariant = 0.01 * gauge_config.alpha_0  # 1% of α_0

    if delta_inside > threshold_covariant and delta_outside < threshold_invariant:
        verdict = "covariant"
        confidence = delta_inside / (delta_outside + 1e-10)
    elif delta_inside < threshold_invariant and delta_outside < threshold_invariant:
        verdict = "invariant"
        confidence = 1.0 / (delta_inside + 1e-10)
    else:
        verdict = "inconclusive"
        confidence = 0.0

    return {
        "baseline_d_prime": d_prime_baseline,
        "transformed_d_prime": d_prime_transformed,
        "delta_d_prime": delta_d_prime,
        "delta_inside": delta_inside,
        "delta_boundary": delta_boundary,
        "delta_outside": delta_outside,
        "alpha_0": gauge_config.alpha_0,
        "verdict": verdict,
        "confidence": confidence,
        "inside_mask": inside,
        "boundary_mask": boundary,
        "outside_mask": outside,
        "rho": rho,
        "num_trials": num_trials,
    }


def generate_gauge_covariance_report(results: dict) -> str:
    """Generate comprehensive report from gauge covariance test.

    Args:
        results: Output from test_gauge_covariance()

    Returns:
        Formatted text report

    Example:
        >>> results = test_gauge_covariance(...)
        >>> report = generate_gauge_covariance_report(results)
        >>> print(report)
    """
    verdict = results["verdict"]
    alpha_0 = results["alpha_0"]
    delta_inside = results["delta_inside"]
    delta_boundary = results["delta_boundary"]
    delta_outside = results["delta_outside"]
    results["confidence"]
    rho = results["rho"]

    return f"""
================================================================================
GAUGE COVARIANCE TEST (TEST 1D) - CRITICAL EXPERIMENT
================================================================================

Configuration:
  ρ (localization scale): {rho if rho is not None else "∞ (mean-field)"}
  α_0 (phase shift): {alpha_0:.4f}
  Number of trials: {results["num_trials"]}

VERDICT: {verdict.upper()}

--------------------------------------------------------------------------------
Measurement Results:
--------------------------------------------------------------------------------

Δd' (change in collective field) by region:

  Inside region R:     Δd'_in  = {delta_inside:.6f}
  Boundary (∂R):       Δd'_bd  = {delta_boundary:.6f}
  Outside region:      Δd'_out = {delta_outside:.6f}

  Response ratio:      Δd'_in / Δd'_out = {delta_inside / (delta_outside + 1e-10):.2f}

--------------------------------------------------------------------------------
Interpretation:
--------------------------------------------------------------------------------

{_interpret_gauge_test(results)}

--------------------------------------------------------------------------------
Physical Implications:
--------------------------------------------------------------------------------

{_explain_implications(verdict)}

================================================================================
"""


def _interpret_gauge_test(results: dict) -> str:
    """Generate interpretation of gauge covariance test results."""
    verdict = results["verdict"]
    delta_inside = results["delta_inside"]
    delta_outside = results["delta_outside"]
    alpha_0 = results["alpha_0"]

    if verdict == "covariant":
        return f"""✓✓✓ GAUGE COVARIANT behavior detected!

The collective field d'_i transforms NON-TRIVIALLY under local gauge transformation:
  - Inside region: Δd' ~ O(α) = O({alpha_0:.4f})
  - Outside region: Δd' ≈ 0 (localized response)

This indicates that d'_i is a GAUGE-COVARIANT LOCAL FIELD, not a gauge-invariant
observable. The transformation compensates for the phase shift within the ρ-neighborhood.

CONCLUSION: Local gauge theory interpretation is VIABLE!"""

    if verdict == "invariant":
        return """GAUGE INVARIANT behavior detected.

The collective field d'_i does NOT transform under local gauge transformation:
  - Inside region: Δd' ≈ 0
  - Outside region: Δd' ≈ 0

This indicates that d'_i is a GAUGE-INVARIANT OBSERVABLE built from gauge-invariant
primitives. It does not support local gauge transformation.

CONCLUSION: Mean-field interpretation applies. Local gauge theory is NOT viable."""

    return f"""INCONCLUSIVE results.

  - Inside: Δd' = {delta_inside:.6f}
  - Outside: Δd' = {delta_outside:.6f}

The response pattern does not clearly match either gauge-covariant or gauge-invariant
behavior. Possible reasons:
  1. Intermediate regime (ρ neither small nor large)
  2. Insufficient trials (increase num_trials)
  3. Wrong transformation ansatz
  4. Numerical issues

RECOMMENDATION: Repeat test with different parameters."""


def _explain_implications(verdict: str) -> str:
    """Explain physical implications of verdict."""
    if verdict == "covariant":
        return """If gauge covariance is confirmed:

1. **Theoretical Framework**: Structure supports LOCAL GAUGE THEORY
   - Can define gauge connection A_μ from collective fields
   - Can construct Yang-Mills action
   - Can derive gauge boson spectrum
   - Strong Standard Model correspondence

2. **Algorithm Physics**: Using processed perception (d'_i from ρ-localized stats)
   creates genuine gauge structure rather than just symmetry.

3. **Publication**: High-impact result (local gauge structure from optimization algorithm)

4. **Next Steps**:
   - Derive gauge connection A_μ
   - Compute Wilson loops
   - Construct conserved Noether currents
   - Study gauge boson excitations"""

    if verdict == "invariant":
        return """If gauge invariance is confirmed:

1. **Theoretical Framework**: Mean-field collective field theory
   - d'_i are auxiliary mean-field variables
   - Analogous to Hartree-Fock, BCS theory
   - Condensed matter interpretation (phonons, magnons)

2. **Algorithm Physics**: Collective fields reflect algorithm's global coordination
   rather than local gauge structure.

3. **Publication**: Still interesting (emergent collective behavior)

4. **Next Steps**:
   - Develop mean-field formalism
   - Study collective modes
   - Find condensed matter analogs
   - Phase transitions in (ρ, ε) space"""

    return """Inconclusive results require further investigation:

1. Vary ρ systematically to find crossover
2. Increase num_trials for better statistics
3. Try different transformation ansätze
4. Check numerical stability"""
