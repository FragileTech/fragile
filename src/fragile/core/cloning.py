from typing import Any

import numpy as np
import panel as pn
import param
import torch
from torch import Tensor

from fragile.core.panel_model import INPUT_WIDTH, PanelModel


def compute_cloning_score(
    fitness: Tensor,
    companion_fitness: Tensor,
    epsilon_clone: float = 0.01,
) -> Tensor:
    """Compute cloning score comparing walker fitness to companion fitness.

    Implements the Canonical Cloning Score from Definition def-cloning-score in 03_cloning.md:

        S_i(c_i) = (V_fit,c_i - V_fit,i) / (V_fit,i + ε_clone)

    The cloning score measures the relative fitness advantage of the companion over
    the walker. Positive scores indicate the walker is less fit than its companion
    and should be replaced (cloned). Negative scores indicate the walker is fitter
    and should persist.

    Reference: Chapter 5, Section 5.7.2 in 03_cloning.md

    Args:
        fitness: Fitness values for all walkers [N]
        companion_fitness: Fitness values of selected companions [N]
        epsilon_clone: Regularization constant preventing division by zero (default: 0.01)

    Returns:
        Cloning scores [N]. Positive scores favor cloning, negative favor persistence.

    Note:
        - The scores are anti-symmetric: S_i(c) = -S_c(i) (approximately)
        - Only the less fit walker in a pair receives a positive score
        - Information flows from high-fitness to low-fitness regions
        - Dead walkers should have fitness = 0.0, giving them maximum cloning pressure
    """
    return (companion_fitness - fitness) / (fitness + epsilon_clone)


def compute_cloning_probability(
    cloning_scores: Tensor,
    p_max: float = 0.75,
) -> Tensor:
    """Convert cloning scores to cloning probabilities via clipping function.

    Implements the clipping function π(S) = min(1, max(0, S/p_max)) from the
    cloning decision mechanism in 03_cloning.md.

    The total cloning probability for walker i is:
        p_i = E[π(S_i(c_i))] where expectation is over companion selection

    For a single companion choice, this function computes:
        π(S_i) = min(1, max(0, S_i / p_max))

    Reference: Chapter 5, Section 5.7.3 in 03_cloning.md

    Args:
        cloning_scores: Cloning scores for all walkers [N]
        p_max: Maximum cloning probability threshold (default: 0.75)

    Returns:
        Cloning probabilities [N] in range [0, 1]

    Note:
        - Scores ≤ 0 → probability = 0 (persist)
        - Scores ≥ p_max → probability = 1 (guaranteed clone)
        - 0 < Scores < p_max → linear interpolation
    """
    return torch.clamp(cloning_scores / p_max, min=0.0, max=1.0)


def inelastic_collision_velocity(
    velocities: Tensor,
    companions: Tensor,
    will_clone: Tensor,
    alpha_restitution: float = 0.5,
) -> Tensor:
    """Compute velocities after multi-body inelastic collision.

    Implements Definition 5.7.4 from 03_cloning.md:
    - Groups walkers by their companion (only those that will actually clone)
    - Conserves momentum within each collision group
    - Applies restitution coefficient to relative velocities

    Physics:
    - alpha_restitution = 0: fully inelastic (all velocities → V_COM)
    - alpha_restitution = 1: perfectly elastic (magnitudes preserved)

    Reference: Chapter 5, Section 5.7.4 in 03_cloning.md

    Args:
        velocities: Current velocities [N, d]
        companions: Companion indices [N]
        will_clone: Boolean mask [N], True for walkers that will clone
        alpha_restitution: Restitution coefficient in [0, 1] (default: 0.5)

    Returns:
        New velocities [N, d] after collision. Walkers that don't clone keep
        their original velocities.

    Note:
        - Only walkers with will_clone=True participate in collisions
        - Each collision group consists of: companion + all walkers cloning to it
        - Momentum is conserved within each group independently
        - The companion itself may be cloning to another walker
    """
    v_new = velocities.clone()  # Start with original velocities

    # Get indices of walkers that will actually clone
    cloning_walker_indices = torch.where(will_clone)[0]

    if cloning_walker_indices.numel() == 0:
        # No walkers cloning, return unchanged velocities
        return v_new

    # Get unique companions that have at least one walker cloning to them
    unique_companions = torch.unique(companions[will_clone])

    for c_idx in unique_companions:
        # Find all walkers that will clone to this companion
        # Must satisfy: companions[i] == c_idx AND will_clone[i] == True
        cloners_mask = (companions == c_idx) & will_clone  # [N]
        cloner_indices = torch.where(cloners_mask)[0]  # [M]

        if cloner_indices.numel() == 0:
            continue  # No actual cloners for this companion

        # Build collision group: companion + cloners (excluding companion from cloners)
        # This prevents double-counting when a walker clones to itself
        cloner_indices_no_companion = cloner_indices[cloner_indices != c_idx]

        # Collision group: [companion, cloner_1, ..., cloner_M]
        group_indices = torch.cat([c_idx.unsqueeze(0), cloner_indices_no_companion])
        group_velocities = velocities[group_indices]  # [M+1, d] where M is number of OTHER cloners

        # Step 1: Compute center-of-mass velocity (conserved quantity)
        V_COM = torch.mean(group_velocities, dim=0)  # [d]

        # Step 2: Compute relative velocities in COM frame
        u_relative = group_velocities - V_COM.unsqueeze(0)  # [M+1, d]

        # Step 3: Apply restitution (scale relative velocities)
        # Note: We apply restitution without individual rotations to preserve momentum
        # The stochasticity comes from the random companion selection process
        u_new = alpha_restitution * u_relative  # [M+1, d]

        # Step 4: Transform back to lab frame
        v_group_new = V_COM.unsqueeze(0) + u_new  # [M+1, d]

        # Step 5: Assign new velocities to all members of collision group
        v_new[group_indices] = v_group_new

    return v_new


def clone_position(
    positions: Tensor,
    companions: Tensor,
    will_clone: Tensor,
    sigma_x: float = 0.1,
) -> Tensor:
    """Clone positions with Gaussian jitter.

    Implements the position update from Definition def-inelastic-collision-update
    in 03_cloning.md:

        x'_i = x_{c_i} + σ_x ζ_i^x  where ζ_i^x ~ N(0, I_d)

    Walkers that clone receive their companion's position plus Gaussian jitter.
    Walkers that persist keep their original position unchanged.

    Reference: Chapter 9, Section 9.3 in 03_cloning.md

    Args:
        positions: Current positions [N, d]
        companions: Companion indices [N]
        will_clone: Boolean mask [N], True for walkers that will clone
        sigma_x: Position jitter scale (default: 0.1)

    Returns:
        New positions [N, d]. Cloners receive companion position + jitter,
        persisters keep original position.

    Note:
        - Gaussian jitter breaks spatial correlations in coupled swarms
        - The jitter scale σ_x controls positional desynchronization
        - Companions themselves don't change position from this interaction
        - Dead walkers should have will_clone=True and receive new positions
    """
    x_new = positions.clone()

    if not will_clone.any():
        # No walkers cloning, return unchanged positions
        return x_new

    # Get indices of walkers that will clone
    cloner_indices = torch.where(will_clone)[0]

    # Get companion positions for cloners
    companion_positions = positions[companions[cloner_indices]]  # [M, d]

    # Generate Gaussian jitter: ζ_i^x ~ N(0, I_d)
    d = positions.shape[1]  # Dimensionality
    device = positions.device
    zeta = torch.randn(cloner_indices.numel(), d, device=device)  # [M, d]

    # Apply position update: x'_i = x_{c_i} + σ_x ζ_i^x
    x_new[cloner_indices] = companion_positions + sigma_x * zeta

    return x_new


def clone_walkers(
    positions: Tensor,
    velocities: Tensor,
    fitness: Tensor,
    companions: Tensor,
    alive: Tensor,
    p_max: float = 1.0,
    epsilon_clone: float = 0.01,
    sigma_x: float = 0.1,
    alpha_restitution: float = 0.5,
    **clone_tensor_kwargs,
) -> tuple[Tensor, Tensor, dict[str, Any], dict]:
    """Execute complete cloning operator Ψ_clone.

    Implements the full cloning pipeline from Chapter 9 of 03_cloning.md:
    1. Compute cloning scores: S_i = (V_c - V_i) / (V_i + ε)
    2. Convert scores to probabilities: π(S) = clip(S / p_max, 0, 1)
    3. Make stochastic cloning decisions
    4. Update positions: x'_i = x_c + σ_x ζ_i^x for cloners
    5. Update velocities: inelastic collision model

    Reference: Chapter 9, Sections 9.3-9.4 in 03_cloning.md

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        fitness: Fitness potential values [N] from compute_fitness
        companions: Companion indices [N] from compute_fitness
        alive: Boolean mask [N], True for alive walkers
        p_max: Maximum cloning probability threshold (default: 1.0)
        epsilon_clone: Regularization for cloning score (default: 0.01)
        sigma_x: Position jitter scale (default: 0.1)
        alpha_restitution: Velocity restitution coefficient (default: 0.5)
        **clone_tensor_kwargs: Additional tensors to be cloned alongside positions/velocities. \
            Each tensor should have shape [N]. These tensors will be cloned in the same way \
            as positions/velocities, using the same companion indices and cloning decisions. \
            Those tensors will be returned inside a dictionary.

    Returns:
        positions_new: Updated positions [N, d]
        velocities_new: Updated velocities [N, d]
        cloned_tensors: Dictionary of updated additional tensors. Each tensor has shape [N, ...].
        info: Dictionary with intermediate results:
            - 'cloning_scores': Cloning scores [N]
            - 'cloning_probs': Cloning probabilities [N]
            - 'will_clone': Boolean mask [N] of walkers that cloned
            - 'num_cloned': Number of walkers that cloned
            - 'companions': Companion indices [N] (same as input)

    Note:
        - Dead walkers (alive=False) should have fitness=0, giving them maximum
          cloning pressure to be revived
        - All output walkers have alive_new=True (intermediate all-alive state)
        - Positions updated before velocities to maintain proper state
        - Momentum conserved within each collision group
    """
    N = positions.shape[0]
    device = positions.device
    with torch.no_grad():
        # Dead walkers should always clone (they need to be revived)
        # They should have fitness=0 from compute_fitness, which gives maximum score
        # But we enforce it explicitly here for robustness

        # Step 1: Compute cloning scores for all walkers
        # S_i = (V_fit,c_i - V_fit,i) / (V_fit,i + ε_clone)
        companion_fitness = fitness[companions]
        cloning_scores = compute_cloning_score(
            fitness, companion_fitness, epsilon_clone=epsilon_clone
        )

        # Dead walkers get maximum positive score to guarantee cloning
        if not alive.all():
            cloning_scores = torch.where(
                alive, cloning_scores, torch.tensor(float("inf"), device=device)
            )

        # Step 2: Convert scores to probabilities via clipping function
        # π(S) = min(1, max(0, S / p_max))
        cloning_probs = compute_cloning_probability(cloning_scores, p_max=p_max)

        # Step 3: Make stochastic cloning decisions
        # Sample uniform thresholds and compare to probabilities
        thresholds = torch.rand(N, device=device)
        will_clone = cloning_probs > thresholds
        will_clone[~alive] = True  # Ensure dead walkers always clone

        # Step 4: Update positions with Gaussian jitter
        # x'_i = x_c + σ_x ζ_i^x for cloners
        positions_new = clone_position(positions, companions, will_clone, sigma_x=sigma_x)

        # Step 5: Update velocities via inelastic collision
        # Conserves momentum within each collision group
        velocities_new = inelastic_collision_velocity(
            velocities, companions, will_clone, alpha_restitution=alpha_restitution
        )

        # Step 6: All walkers are alive in intermediate state
        alive_new = torch.ones_like(alive)

        # Collect intermediate results for analysis
        info = {
            "cloning_scores": cloning_scores,
            "cloning_probs": cloning_probs,
            "will_clone": will_clone,
            "alive_new": alive_new,
            "num_cloned": will_clone.sum().item(),
            "companions": companions,
        }
        other_cloned = {
            k: clone_tensor(x, companions, will_clone) for k, x in clone_tensor_kwargs.items()
        }

        return positions_new, velocities_new, other_cloned, info


def clone_tensor(
    x: Tensor | np.ndarray, compas_ix: Tensor, will_clone: Tensor
) -> Tensor | np.ndarray:
    """Clone the data from the compas indexes."""
    if not will_clone.any():
        return x
    if isinstance(x, torch.Tensor):
        x[will_clone] = x[compas_ix][will_clone]
    elif isinstance(x, np.ndarray):
        compas_ix, will_clone = compas_ix.cpu().numpy(), will_clone.cpu().numpy()
        x[will_clone] = x[compas_ix][will_clone]
    else:
        raise ValueError(f"Unsupported type {type(x)}")
    return x


class CloneOperator(PanelModel):
    """Stateless cloning operator wrapping the functional clone_walkers interface.

    This class provides a parameter-validated wrapper around the clone_walkers function,
    storing default parameters while allowing per-call overrides.

    Mathematical notation from 03_cloning.md:
    - p_max: Maximum cloning probability threshold
    - ε_clone (epsilon_clone): Regularization for cloning score
    - σ_x (sigma_x): Position jitter scale
    - α_rest (alpha_restitution): Restitution coefficient

    Reference: Chapter 9, Sections 9.3-9.4 in 03_cloning.md

    Example:
        >>> operator = CloneOperator(p_max=0.75, sigma_x=0.1)
        >>> pos_new, vel_new, alive_new, info = operator(
        ...     positions, velocities, fitness, companions, alive
        ... )
    """

    _n_widget_columns = param.Integer(default=2, bounds=(1, None), doc="Number of widget columns")
    _max_widget_width = param.Integer(default=800, bounds=(0, None), doc="Maximum widget width")

    p_max = param.Number(
        default=1.0,
        bounds=(0, 1),
        softbounds=(0.2, 1.0),
        doc="Maximum cloning probability threshold",
    )
    epsilon_clone = param.Number(
        default=0.01,
        bounds=(0, None),
        softbounds=(1e-4, 0.05),
        doc="Regularization for cloning score (ε_clone)",
    )
    sigma_x = param.Number(
        default=0.1,
        bounds=(0, None),
        softbounds=(0.01, 1.0),
        doc="Position jitter scale (σ_x)",
    )
    alpha_restitution = param.Number(
        default=0.5,
        bounds=(0, 1),
        softbounds=(0.0, 1.0),
        doc="Velocity restitution coefficient (α_rest): 0=fully inelastic, 1=elastic",
    )

    @property
    def widgets(self) -> dict[str, dict]:
        """Widget configurations for cloning parameters."""
        return {
            "p_max": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "p_max (max cloning prob)",
                "start": 0.2,
                "end": 10.0,
                "step": 0.1,
            },
            "epsilon_clone": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "ε_clone (regularization)",
                "start": 1e-4,
                "end": 0.05,
                "step": 1e-4,
            },
            "sigma_x": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "σ_x (position jitter)",
                "start": 0.05,
                "end": 2.0,
                "step": 0.05,
            },
            "alpha_restitution": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "α_rest (restitution)",
                "start": 0.0,
                "end": 1.0,
                "step": 0.05,
            },
        }

    @property
    def widget_parameters(self) -> list[str]:
        """Parameters to display in UI."""
        return ["p_max", "epsilon_clone", "sigma_x", "alpha_restitution"]

    def __call__(
        self,
        positions: Tensor,
        velocities: Tensor,
        fitness: Tensor,
        companions: Tensor,
        alive: Tensor,
        p_max: float | None = None,
        epsilon_clone: float | None = None,
        sigma_x: float | None = None,
        alpha_restitution: float | None = None,
        **clone_tensor_kwargs,
    ) -> tuple[Tensor, Tensor, dict[str, Any], dict]:
        """Execute cloning operator using clone_walkers function.

        Parameters provided to this call override the instance defaults.

        Args:
            positions: Walker positions [N, d]
            velocities: Walker velocities [N, d]
            fitness: Fitness potential values [N] from compute_fitness
            companions: Companion indices [N] from compute_fitness
            alive: Boolean mask [N], True for alive walkers
            p_max: Override maximum cloning probability threshold
            epsilon_clone: Override regularization for cloning score
            sigma_x: Override position jitter scale
            alpha_restitution: Override velocity restitution coefficient
            **clone_tensor_kwargs: Additional tensors to be cloned alongside
                positions/velocities. Each tensor should have shape [N]. These
                tensors will be cloned in the same way as positions/velocities,
                using the same companion indices and cloning decisions.
                Those tensors will be returned inside a dictionary.

        Returns:
            positions_new: Updated positions [N, d]
            velocities_new: Updated velocities [N, d]
            cloned_tensors: Dictionary of cloned additional tensors with same keys as input.
            info: Dictionary with intermediate results:
                - 'cloning_scores': Cloning scores [N]
                - 'cloning_probs': Cloning probabilities [N]
                - 'will_clone': Boolean mask [N] of walkers that cloned
                - 'num_cloned': Number of walkers that cloned
                - 'companions': Companion indices [N] (same as input)

        Note:
            This is a stateless wrapper - no internal state is modified during calls.
        """
        return clone_walkers(
            positions=positions,
            velocities=velocities,
            fitness=fitness,
            companions=companions,
            alive=alive,
            p_max=p_max if p_max is not None else self.p_max,
            epsilon_clone=epsilon_clone if epsilon_clone is not None else self.epsilon_clone,
            sigma_x=sigma_x if sigma_x is not None else self.sigma_x,
            alpha_restitution=alpha_restitution
            if alpha_restitution is not None
            else self.alpha_restitution,
            **clone_tensor_kwargs,
        )
