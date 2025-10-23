"""U(1)_fitness symmetry structure tests.

This module implements tests for the U(1) fitness gauge symmetry, comparing the
current framework (raw distance-based phases) with the proposed framework (collective
field-based phases).

**Framework References:**
- old_docs/source/13_fractal_set_new/03_yang_mills_noether.md - Current U(1) structure
- old_docs/source/13_fractal_set_new/04_symmetry_redefinition_viability_analysis.md §3.1 - Proposed U(1)

**Current U(1) Structure:**
- Phase: θ_ik^(div) = -d_alg²(i,k)/(2ε_d² ℏ_eff) (pairwise, geometric)
- Amplitude: √P_comp(k|i) from softmax over distances
- Dressed walker: |ψ_i⟩ = Σ_k √P(k|i) · e^(iθ_ik) |k⟩

**Proposed U(1) Structure:**
- Phase: θ_i = (d'_i)^β / ℏ_eff (collective field value)
- Amplitude: SAME as current
- Dressed walker: |ψ_i⟩ = Σ_k √P(k|i) · e^(iθ_i) |k⟩

**Key Difference:**
- Current: Pairwise phases θ_ik depend on (i,k) pair geometry
- Proposed: Walker phases θ_i depend on collective field d'_i (entire swarm through ρ-stats)
"""

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from fragile.core.companion_selection import (
    compute_algorithmic_distance_matrix,
    select_companions_softmax,
)
from fragile.experiments.gauge.observables import (
    ObservablesConfig,
    compute_collective_fields,
)


class U1Config(BaseModel):
    """Configuration for U(1) symmetry tests.

    Attributes:
        h_eff: Effective Planck constant (ℏ_eff)
        epsilon_d: Diversity companion selection range (ε_d)
        lambda_alg: Velocity weight in algorithmic distance
    """

    h_eff: float = Field(default=1.0, gt=0, description="Effective Planck constant")
    epsilon_d: float = Field(default=0.1, gt=0, description="Diversity companion range")
    lambda_alg: float = Field(default=0.0, ge=0, description="Velocity weight")


def compute_u1_phase_current(
    positions: Tensor,
    velocities: Tensor,
    companions: Tensor,
    alive: Tensor,
    config: U1Config | None = None,
) -> Tensor:
    """Compute current U(1) phases: θ_ik = -d_alg²(i,k)/(2ε_d² ℏ_eff).

    From {prf:ref}`def-dressed-walker-state` in:
    old_docs/source/13_fractal_set_new/03_yang_mills_noether.md §1.2

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        companions: Diversity companion indices [N]
        alive: Boolean mask [N]
        config: U(1) configuration

    Returns:
        Phases [N] where phase[i] = θ_i(c_div(i))

    Example:
        >>> positions = torch.randn(100, 2)
        >>> velocities = torch.randn(100, 2) * 0.1
        >>> companions = torch.randint(0, 100, (100,))
        >>> alive = torch.ones(100, dtype=torch.bool)
        >>> phases = compute_u1_phase_current(positions, velocities, companions, alive)
    """
    if config is None:
        config = U1Config()

    # Compute full distance matrix [N, N]
    dist_sq = compute_algorithmic_distance_matrix(
        positions, velocities, config.lambda_alg
    )

    # Extract distances to companions [N]
    N = positions.shape[0]
    dist_sq_companion = dist_sq[torch.arange(N), companions]

    # Compute phases: θ_ik = -d_alg²(i,k) / (2ε_d² ℏ_eff)
    phases = -dist_sq_companion / (2 * config.epsilon_d**2 * config.h_eff)

    # Mask dead walkers
    return torch.where(alive, phases, torch.zeros_like(phases))


def compute_u1_phase_proposed(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor,
    companions: Tensor,
    alive: Tensor,
    rho: float | None = None,
    obs_config: ObservablesConfig | None = None,
    u1_config: U1Config | None = None,
) -> Tensor:
    """Compute proposed U(1) phases: θ_i = (d'_i)^β / ℏ_eff.

    From §3.1.2 in:
    old_docs/source/13_fractal_set_new/04_symmetry_redefinition_viability_analysis.md

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N]
        companions: Diversity companion indices [N]
        alive: Boolean mask [N]
        rho: Localization scale (None for mean-field)
        obs_config: Observables configuration
        u1_config: U(1) configuration

    Returns:
        Phases [N] where phase[i] = θ_i = (d'_i)^β / ℏ_eff

    Example:
        >>> # Mean-field regime
        >>> phases_mf = compute_u1_phase_proposed(
        ...     positions, velocities, rewards, companions, alive, rho=None
        ... )
        >>>
        >>> # Local regime
        >>> phases_local = compute_u1_phase_proposed(
        ...     positions, velocities, rewards, companions, alive, rho=0.05
        ... )
    """
    if obs_config is None:
        obs_config = ObservablesConfig()
    if u1_config is None:
        u1_config = U1Config()

    # Compute collective fields
    fields = compute_collective_fields(
        positions, velocities, rewards, alive, companions, rho, obs_config
    )

    # Extract d'_i
    d_prime = fields["d_prime"]

    # Compute phases: θ_i = (d'_i)^β / ℏ_eff
    phases = (d_prime ** obs_config.beta) / u1_config.h_eff

    # Mask dead walkers
    return torch.where(alive, phases, torch.zeros_like(phases))


def compute_u1_amplitude(
    positions: Tensor,
    velocities: Tensor,
    alive: Tensor,
    config: U1Config | None = None,
) -> Tensor:
    """Compute U(1) amplitude (SAME for current and proposed).

    Amplitude: √P_comp(k|i) = softmax probability

    From {prf:ref}`def-dressed-walker-state` in:
    old_docs/source/13_fractal_set_new/03_yang_mills_noether.md §1.2

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        alive: Boolean mask [N]
        config: U(1) configuration

    Returns:
        Companion selection probabilities [N, N] where P[i,j] = P(j|i)

    Example:
        >>> amplitudes = compute_u1_amplitude(positions, velocities, alive)
        >>> # amplitudes[i, j] = probability walker i chooses companion j
    """
    if config is None:
        config = U1Config()

    N = positions.shape[0]

    # Compute distance matrix
    dist_sq = compute_algorithmic_distance_matrix(
        positions, velocities, config.lambda_alg
    )

    # Compute softmax weights: exp(-d²/(2ε²))
    weights = torch.exp(-dist_sq / (2 * config.epsilon_d**2))

    # Mask dead walkers and self-pairing
    alive_mask = alive.unsqueeze(1) & alive.unsqueeze(0)
    self_mask = ~torch.eye(N, device=alive.device, dtype=torch.bool)
    weights = weights * alive_mask.float() * self_mask.float()

    # Normalize to probabilities
    probs = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)

    return probs


def compare_u1_phases(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor,
    companions: Tensor,
    alive: Tensor,
    rho: float | None = None,
    obs_config: ObservablesConfig | None = None,
    u1_config: U1Config | None = None,
) -> dict[str, Tensor]:
    """Compare current vs proposed U(1) phase structures.

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N]
        companions: Diversity companion indices [N]
        alive: Boolean mask [N]
        rho: Localization scale
        obs_config: Observables configuration
        u1_config: U(1) configuration

    Returns:
        Dictionary with keys:
            - "current": Current phases [N]
            - "proposed": Proposed phases [N]
            - "difference": |proposed - current| [N]
            - "correlation": Correlation coefficient (scalar)
            - "current_mean": Mean current phase (scalar)
            - "current_std": Std current phase (scalar)
            - "proposed_mean": Mean proposed phase (scalar)
            - "proposed_std": Std proposed phase (scalar)

    Example:
        >>> comparison = compare_u1_phases(
        ...     positions, velocities, rewards, companions, alive, rho=0.05
        ... )
        >>> print(f"Correlation: {comparison['correlation']:.4f}")
        >>> print(f"Mean difference: {comparison['difference'].mean():.4f}")
    """
    # Compute both phase structures
    current = compute_u1_phase_current(
        positions, velocities, companions, alive, u1_config
    )
    proposed = compute_u1_phase_proposed(
        positions, velocities, rewards, companions, alive, rho, obs_config, u1_config
    )

    # Extract alive values
    current_alive = current[alive]
    proposed_alive = proposed[alive]

    # Compute correlation
    if len(current_alive) > 1:
        correlation = torch.corrcoef(
            torch.stack([current_alive, proposed_alive])
        )[0, 1].item()
    else:
        correlation = 0.0

    return {
        "current": current,
        "proposed": proposed,
        "difference": torch.abs(proposed - current),
        "correlation": correlation,
        "current_mean": current_alive.mean().item(),
        "current_std": current_alive.std().item(),
        "proposed_mean": proposed_alive.mean().item(),
        "proposed_std": proposed_alive.std().item(),
    }


def compute_dressed_walker_state(
    phases: Tensor,
    amplitudes: Tensor,
    walker_idx: int,
    alive: Tensor,
) -> Tensor:
    """Compute dressed walker state |ψ_i⟩ = Σ_k √P(k|i) · e^(iθ) |k⟩.

    Args:
        phases: Phases [N] (either θ_ik for current or θ_i for proposed)
        amplitudes: Companion probabilities [N, N]
        walker_idx: Index of walker to compute state for
        alive: Boolean mask [N]

    Returns:
        Complex state vector [N] where component k = √P(k|i) · e^(iθ)
        For current: θ = θ_ik (pairwise)
        For proposed: θ = θ_i (same for all k)

    Note:
        For proposed structure, phase θ_i is constant across all companions,
        making it a global phase on walker i's state.
    """
    # Get probabilities for this walker
    probs_i = amplitudes[walker_idx]  # [N]

    # For current: need pairwise phases θ_ik (requires full phase matrix)
    # For proposed: use walker's phase θ_i (broadcast)
    # Here we assume phase is either [N] (proposed) or should be [N,N] (current)

    # Since we only have phases[N], interpret as proposed structure
    phase_i = phases[walker_idx]

    # Compute complex coefficients: ψ_ik = √P(k|i) · e^(iθ_i)
    psi = torch.sqrt(probs_i) * torch.exp(1j * phase_i)

    # Mask dead companions
    psi = torch.where(alive, psi, torch.zeros_like(psi))

    return psi


def compute_u1_observable(dressed_state: Tensor) -> float:
    """Compute gauge-invariant observable |⟨ψ_i|ψ_i⟩|².

    Physical observables must be gauge-invariant (same under U(1) transformation).

    Args:
        dressed_state: Complex state vector [N]

    Returns:
        Norm squared (should be ≈ 1 for normalized state)
    """
    return torch.abs(torch.dot(dressed_state.conj(), dressed_state)).item()
