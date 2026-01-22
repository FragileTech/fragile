"""SU(2)_weak symmetry structure tests.

This module implements tests for the SU(2) weak isospin gauge symmetry using the
unified definition:

**SU(2) Structure:**
- Phase: θ_ij = S_i(j) / ℏ_eff where S_i(j) = (V_j - V_i)/(V_i + ε_clone)
- Amplitude: Pairing probability from cloning companion selection
- Interaction state: |Ψ_ij⟩ = |↑⟩⊗|ψ_i⟩ + |↓⟩⊗|ψ_j⟩

The doublet structure comes from the asymmetric cloning rule: only one direction
of cloning is allowed for any ordered pair (i, j) with V_i ≠ V_j.
"""

from pydantic import BaseModel, Field
import torch
from torch import Tensor

from fragile.fractalai.core.companion_selection import (
    compute_algorithmic_distance_matrix,
)
from fragile.fractalai.experiments.gauge.observables import (
    compute_cloning_score,
    compute_collective_fields,
    ObservablesConfig,
)


class SU2Config(BaseModel):
    """Configuration for SU(2) symmetry tests.

    Attributes:
        h_eff: Effective Planck constant (ℏ_eff)
        epsilon_c: Cloning companion selection range (ε_c)
        epsilon_clone: Regularization for cloning score
        lambda_alg: Velocity weight in algorithmic distance
    """

    h_eff: float = Field(default=1.0, gt=0, description="Effective Planck constant")
    epsilon_c: float = Field(default=0.1, gt=0, description="Cloning companion range")
    epsilon_clone: float = Field(default=1e-8, gt=0, description="Cloning score regularization")
    lambda_alg: float = Field(default=0.0, ge=0, description="Velocity weight")


def compute_su2_phase(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor | None,
    diversity_companions: Tensor,
    clone_companions: Tensor,
    alive: Tensor,
    rho: float | None = None,
    obs_config: ObservablesConfig | None = None,
    su2_config: SU2Config | None = None,
    fitness: Tensor | None = None,
) -> Tensor:
    """Compute SU(2) phases: θ_ij = S_i(j) / ℏ_eff.

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N] (required if fitness is None)
        diversity_companions: Diversity companion indices [N]
        clone_companions: Cloning companion indices [N]
        alive: Boolean mask [N]
        rho: Localization scale
        obs_config: Observables configuration
        su2_config: SU(2) configuration
        fitness: Optional precomputed fitness potential V [N]

    Returns:
        Phases [N] where phase[i] = θ_i(c_clone(i))
    """
    if obs_config is None:
        obs_config = ObservablesConfig()
    if su2_config is None:
        su2_config = SU2Config()

    if fitness is None:
        if rewards is None:
            msg = "rewards or fitness must be provided to compute SU(2) phases."
            raise ValueError(msg)
        fields = compute_collective_fields(
            positions, velocities, rewards, alive, diversity_companions, rho, obs_config
        )
        fitness = fields["fitness"]

    # Compute cloning scores
    scores = compute_cloning_score(fitness, alive, clone_companions, su2_config.epsilon_clone)

    # Compute phases: θ_ij = S_i(j) / ℏ_eff
    phases = scores / su2_config.h_eff

    # Mask dead walkers (already done in compute_cloning_score, but ensure)
    return torch.where(alive, phases, torch.zeros_like(phases))


def compute_su2_phase_current(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor | None,
    diversity_companions: Tensor,
    clone_companions: Tensor,
    alive: Tensor,
    rho: float | None = None,
    obs_config: ObservablesConfig | None = None,
    su2_config: SU2Config | None = None,
    fitness: Tensor | None = None,
) -> Tensor:
    """Legacy alias for compute_su2_phase (cloning-score phases)."""
    return compute_su2_phase(
        positions,
        velocities,
        rewards,
        diversity_companions,
        clone_companions,
        alive,
        rho=rho,
        obs_config=obs_config,
        su2_config=su2_config,
        fitness=fitness,
    )


def compute_su2_phase_proposed(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor | None,
    diversity_companions: Tensor,
    clone_companions: Tensor,
    alive: Tensor,
    rho: float | None = None,
    obs_config: ObservablesConfig | None = None,
    su2_config: SU2Config | None = None,
    fitness: Tensor | None = None,
) -> Tensor:
    """Legacy alias for compute_su2_phase (cloning-score phases)."""
    return compute_su2_phase(
        positions,
        velocities,
        rewards,
        diversity_companions,
        clone_companions,
        alive,
        rho=rho,
        obs_config=obs_config,
        su2_config=su2_config,
        fitness=fitness,
    )


def compute_su2_pairing_probability(
    positions: Tensor,
    velocities: Tensor,
    alive: Tensor,
    config: SU2Config | None = None,
) -> Tensor:
    """Compute pairing probabilities for cloning companion selection.

    This is used as the amplitude in the SU(2) structure.

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        alive: Boolean mask [N]
        config: SU(2) configuration

    Returns:
        Pairing probabilities [N, N] where P[i,j] = P(i ↔ j | pairing)

    Note:
        We use ε_c (cloning range) to determine pairing probabilities.
    """
    if config is None:
        config = SU2Config()

    N = positions.shape[0]

    # Compute distance matrix
    dist_sq = compute_algorithmic_distance_matrix(positions, velocities, config.lambda_alg)

    # Compute softmax weights: exp(-d²/(2ε²))
    weights = torch.exp(-dist_sq / (2 * config.epsilon_c**2))

    # Mask dead walkers and self-pairing
    alive_mask = alive.unsqueeze(1) & alive.unsqueeze(0)
    self_mask = ~torch.eye(N, device=alive.device, dtype=torch.bool)
    weights = weights * alive_mask.float() * self_mask.float()

    # Normalize to probabilities
    return weights / (weights.sum(dim=1, keepdim=True) + 1e-10)


def compute_isospin_doublet_state(
    phases: Tensor,
    amplitudes: Tensor,
    walker_i: int,
    walker_j: int,
    alive: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute weak isospin doublet state for interaction pair (i,j).

    |Ψ_ij⟩ = |↑⟩⊗|ψ_i⟩ + |↓⟩⊗|ψ_j⟩ ∈ ℂ² ⊗ ℂ^(N-1)

    From {prf:ref}`def-dressed-walker-state` in:
    old_docs/source/13_fractal_set_new/03_yang_mills_noether.md §1.2

    Args:
        phases: Phases [N] (cloning-score phases)
        amplitudes: Companion probabilities [N, N]
        walker_i: Cloner walker index
        walker_j: Target walker index
        alive: Boolean mask [N]

    Returns:
        Tuple of (up_component, down_component):
            - up_component: |↑⟩⊗|ψ_i⟩ [N] (cloner component)
            - down_component: |↓⟩⊗|ψ_j⟩ [N] (target component)

    Note:
        The full state is the sum: |Ψ_ij⟩ = up_component + down_component
    """
    # Get probabilities for both walkers
    probs_i = amplitudes[walker_i]  # [N]
    probs_j = amplitudes[walker_j]  # [N]

    # Get phases
    phase_i = phases[walker_i]
    phase_j = phases[walker_j]

    # Compute components: ψ = √P · e^(iθ)
    up_component = torch.sqrt(probs_i) * torch.exp(1j * phase_i)
    down_component = torch.sqrt(probs_j) * torch.exp(1j * phase_j)

    # Mask dead companions
    up_component = torch.where(alive, up_component, torch.zeros_like(up_component))
    down_component = torch.where(alive, down_component, torch.zeros_like(down_component))

    return up_component, down_component


def compare_su2_phases(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor,
    diversity_companions: Tensor,
    clone_companions: Tensor,
    alive: Tensor,
    rho: float | None = None,
    obs_config: ObservablesConfig | None = None,
    su2_config: SU2Config | None = None,
) -> dict[str, Tensor | float]:
    """Compare legacy SU(2) phase APIs (now identical under unified definition).

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N]
        diversity_companions: Diversity companion indices [N]
        clone_companions: Cloning companion indices [N]
        alive: Boolean mask [N]
        rho: Localization scale
        obs_config: Observables configuration
        su2_config: SU(2) configuration

    Returns:
        Dictionary with keys:
            - "current": SU(2) phases [N]
            - "proposed": SU(2) phases [N] (alias of current)
            - "difference": |proposed - current| [N] (zeros)
            - "correlation": Correlation coefficient (scalar)
            - "current_mean": Mean phase (scalar)
            - "current_std": Std phase (scalar)
            - "proposed_mean": Mean phase (scalar)
            - "proposed_std": Std phase (scalar)

    Example:
        >>> comparison = compare_su2_phases(
        ...     positions,
        ...     velocities,
        ...     rewards,
        ...     diversity_companions,
        ...     clone_companions,
        ...     alive,
        ...     rho=0.05,
        ... )
        >>> print(f"Correlation: {comparison['correlation']:.4f}")
        >>> print(f"Mean difference: {comparison['difference'].mean():.4f}")
    """
    # Compute both phase structures (identical under unified definition)
    current = compute_su2_phase(
        positions,
        velocities,
        rewards,
        diversity_companions,
        clone_companions,
        alive,
        rho=rho,
        obs_config=obs_config,
        su2_config=su2_config,
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


def compute_su2_observable(
    up_component: Tensor,
    down_component: Tensor,
) -> float:
    """Compute gauge-invariant observable from isospin doublet state.

    Physical observable: |⟨Ψ_ij|Ψ_ij⟩|²

    Args:
        up_component: |↑⟩⊗|ψ_i⟩ [N]
        down_component: |↓⟩⊗|ψ_j⟩ [N]

    Returns:
        Norm squared of full interaction state
    """
    full_state = up_component + down_component
    return torch.abs(torch.dot(full_state.conj(), full_state)).item()
