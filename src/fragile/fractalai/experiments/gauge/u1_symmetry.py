"""U(1)_fitness symmetry structure tests.

This module implements tests for the U(1) fitness gauge symmetry using the unified
definition:

**U(1) Structure:**
- Phase: θ_ik = -(Φ_k - Φ_i)/ℏ_eff (fitness phase difference)
- Amplitude: √P_comp(k|i) from diversity companion selection
- Dressed walker: |ψ_i⟩ = Σ_k √P(k|i) · e^(iθ_ik) |k⟩

The additive fitness baseline Φ → Φ + c shifts all θ_i equally and leaves θ_ik
invariant, generating the U(1) redundancy.
"""

from pydantic import BaseModel, Field
import torch
from torch import Tensor

from fragile.core.companion_selection import (
    compute_algorithmic_distance_matrix,
)
from fragile.experiments.gauge.observables import (
    compute_collective_fields,
    ObservablesConfig,
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


def compute_u1_phase(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor | None,
    companions: Tensor,
    alive: Tensor,
    rho: float | None = None,
    obs_config: ObservablesConfig | None = None,
    u1_config: U1Config | None = None,
    fitness: Tensor | None = None,
) -> Tensor:
    """Compute U(1) phases: θ_ik = -(Φ_k - Φ_i)/ℏ_eff.

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N] (required if fitness is None)
        companions: Diversity companion indices [N]
        alive: Boolean mask [N]
        rho: Localization scale (None for mean-field)
        obs_config: Observables configuration
        u1_config: U(1) configuration
        fitness: Optional precomputed fitness potential Φ [N]

    Returns:
        Phases [N] where phase[i] = θ_i(c_div(i))

    Example:
        >>> phases = compute_u1_phase(
        ...     positions, velocities, rewards, companions, alive, rho=0.05
        ... )
    """
    if u1_config is None:
        u1_config = U1Config()
    if fitness is None:
        if rewards is None:
            raise ValueError("rewards or fitness must be provided to compute U(1) phases.")
        if obs_config is None:
            obs_config = ObservablesConfig()
        fields = compute_collective_fields(
            positions, velocities, rewards, alive, companions, rho, obs_config
        )
        fitness = fields["fitness"]

    # Extract fitness values to companions [N]
    fitness_companion = fitness[companions]

    # Compute phases: θ_ik = -(Φ_k - Φ_i) / ℏ_eff
    phases = -(fitness_companion - fitness) / u1_config.h_eff

    # Mask dead walkers
    return torch.where(alive, phases, torch.zeros_like(phases))


def compute_u1_phase_current(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor | None,
    companions: Tensor,
    alive: Tensor,
    rho: float | None = None,
    obs_config: ObservablesConfig | None = None,
    u1_config: U1Config | None = None,
    fitness: Tensor | None = None,
) -> Tensor:
    """Legacy alias for compute_u1_phase (fitness-based U(1) phases)."""
    return compute_u1_phase(
        positions,
        velocities,
        rewards,
        companions,
        alive,
        rho=rho,
        obs_config=obs_config,
        u1_config=u1_config,
        fitness=fitness,
    )


def compute_u1_phase_proposed(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor | None,
    companions: Tensor,
    alive: Tensor,
    rho: float | None = None,
    obs_config: ObservablesConfig | None = None,
    u1_config: U1Config | None = None,
    fitness: Tensor | None = None,
) -> Tensor:
    """Legacy alias for compute_u1_phase (fitness-based U(1) phases)."""
    return compute_u1_phase(
        positions,
        velocities,
        rewards,
        companions,
        alive,
        rho=rho,
        obs_config=obs_config,
        u1_config=u1_config,
        fitness=fitness,
    )


def compute_u1_amplitude(
    positions: Tensor,
    velocities: Tensor,
    alive: Tensor,
    config: U1Config | None = None,
) -> Tensor:
    """Compute U(1) amplitude from diversity companion selection.

    Amplitude: √P_comp(k|i) = softmax probability

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
    dist_sq = compute_algorithmic_distance_matrix(positions, velocities, config.lambda_alg)

    # Compute softmax weights: exp(-d²/(2ε²))
    weights = torch.exp(-dist_sq / (2 * config.epsilon_d**2))

    # Mask dead walkers and self-pairing
    alive_mask = alive.unsqueeze(1) & alive.unsqueeze(0)
    self_mask = ~torch.eye(N, device=alive.device, dtype=torch.bool)
    weights = weights * alive_mask.float() * self_mask.float()

    # Normalize to probabilities
    return weights / (weights.sum(dim=1, keepdim=True) + 1e-10)


def compare_u1_phases(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor,
    companions: Tensor,
    alive: Tensor,
    rho: float | None = None,
    obs_config: ObservablesConfig | None = None,
    u1_config: U1Config | None = None,
) -> dict[str, Tensor | float]:
    """Compare legacy U(1) phase APIs (now identical under unified definition).

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
            - "current": U(1) phases [N]
            - "proposed": U(1) phases [N] (alias of current)
            - "difference": |proposed - current| [N] (zeros)
            - "correlation": Correlation coefficient (scalar)
            - "current_mean": Mean phase (scalar)
            - "current_std": Std phase (scalar)
            - "proposed_mean": Mean phase (scalar)
            - "proposed_std": Std phase (scalar)

    Example:
        >>> comparison = compare_u1_phases(
        ...     positions, velocities, rewards, companions, alive, rho=0.05
        ... )
        >>> print(f"Correlation: {comparison['correlation']:.4f}")
        >>> print(f"Mean difference: {comparison['difference'].mean():.4f}")
    """
    # Compute both phase structures
    current = compute_u1_phase(
        positions,
        velocities,
        rewards,
        companions,
        alive,
        rho=rho,
        obs_config=obs_config,
        u1_config=u1_config,
    )
    proposed = current

    # Extract alive values
    current_alive = current[alive]
    proposed_alive = proposed[alive]

    # Compute correlation (identical arrays under unified definition)
    if len(current_alive) > 1 and current_alive.std().item() > 0:
        correlation = torch.corrcoef(torch.stack([current_alive, proposed_alive]))[0, 1].item()
    else:
        correlation = 1.0

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
    """Compute dressed walker state |ψ_i⟩ = Σ_k √P(k|i) · e^(iθ_ik) |k⟩.

    Args:
        phases: Phases [N] or [N, N] (θ_i or θ_ik)
        amplitudes: Companion probabilities [N, N]
        walker_idx: Index of walker to compute state for
        alive: Boolean mask [N]

    Returns:
        Complex state vector [N] where component k = √P(k|i) · e^(iθ)
        If phases is [N, N], it is interpreted as pairwise θ_ik.
        If phases is [N], it is interpreted as a per-walker phase (e.g., θ_i or θ_i(c_i)).
    """
    # Get probabilities for this walker
    probs_i = amplitudes[walker_idx]  # [N]

    if phases.ndim == 2:
        phase_i = phases[walker_idx]
    else:
        phase_i = phases[walker_idx]

    # Compute complex coefficients: ψ_ik = √P(k|i) · e^(iθ_ik)
    psi = torch.sqrt(probs_i) * torch.exp(1j * phase_i)

    # Mask dead companions
    return torch.where(alive, psi, torch.zeros_like(psi))


def compute_u1_observable(dressed_state: Tensor) -> float:
    """Compute gauge-invariant observable |⟨ψ_i|ψ_i⟩|².

    Physical observables must be gauge-invariant (same under U(1) transformation).

    Args:
        dressed_state: Complex state vector [N]

    Returns:
        Norm squared (should be ≈ 1 for normalized state)
    """
    return torch.abs(torch.dot(dressed_state.conj(), dressed_state)).item()
