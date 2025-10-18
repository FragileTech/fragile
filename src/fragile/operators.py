import torch
from torch import Tensor

from fragile.companion_selection import CompanionSelection


def logistic_rescale(z: Tensor, A: float = 1.0) -> Tensor:
    """Logistic rescale function mapping R -> [0, A].

    Implements g_A(z) = A / (1 + exp(-z)), a smooth, bounded, monotone increasing
    function used in the fitness potential V_fit[f, ρ](x) = g_A(Z_ρ[f, d, x]).

    Reference: Definition def-localized-mean-field-fitness in 11_geometric_gas.md

    Args:
        z: Input tensor (typically Z-scores)
        A: Upper bound of the output range (default: 1.0)

    Returns:
        Tensor with values in [0, A]
    """
    return A / (1.0 + torch.exp(-z))


def patched_standardization(
    values: Tensor,
    alive: Tensor,
    rho: float | None = None,
    sigma_min: float = 1e-8,
) -> Tensor:
    """Compute Z-scores using only alive walkers for statistics.

    Implements the patched standardization Z_ρ[f, d, x] where statistics (mean, std)
    are computed only over alive walkers to prevent contamination from dead walkers.

    For the global case (rho=None), computes:
        Z[f, d, x_i] = (d(x_i) - μ[d|alive]) / σ'[d|alive]

    where μ and σ are computed using only alive walkers, and σ' includes regularization:
        σ'[d|alive] = sqrt(σ²[d|alive] + σ²_min)

    Reference: Definition def-unified-z-score in 11_geometric_gas.md

    Args:
        values: Tensor of shape [N] containing measurement values for all walkers
        alive: Boolean tensor of shape [N], True for alive walkers
        rho: Localization scale parameter (not yet implemented for finite rho)
        sigma_min: Regularization constant ensuring σ' ≥ σ_min > 0

    Returns:
        Z-scores tensor of shape [N]. Dead walkers receive Z-score of 0.0.

    Note:
        Current implementation is for the global case (rho → ∞). For finite rho,
        localization kernel K_ρ(x_i, x_j) would weight contributions from nearby
        alive walkers. See def-localized-mean-field-moments in 11_geometric_gas.md.
    """
    if rho is not None:
        msg = "Localized standardization (finite rho) not yet implemented"
        raise NotImplementedError(msg)

    # Extract alive walker values
    alive_values = values[alive]

    if alive_values.numel() == 0:
        # No alive walkers - return zeros
        return torch.zeros_like(values)

    # Compute statistics over alive walkers only
    mu = alive_values.mean()
    sigma_sq = alive_values.var(unbiased=False)  # Population variance

    # Regularized standard deviation: σ'[d|alive] = sqrt(σ²[d|alive] + σ²_min)
    sigma_reg = torch.sqrt(sigma_sq + sigma_min**2)

    # Compute Z-scores for all walkers
    z_scores = (values - mu) / sigma_reg

    # Set Z-scores of dead walkers to 0.0 (they don't participate in dynamics)
    z_scores = torch.where(alive, z_scores, torch.zeros_like(z_scores))

    return z_scores



def compute_fitness(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor,
    alive: Tensor,
    companion_selection: CompanionSelection,
    alpha: float = 1.0,
    beta: float = 1.0,
    eta: float = 0.1,
    lambda_alg: float = 0.0,
    sigma_min: float = 1e-8,
    A: float = 2.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute fitness potential using the Euclidean Gas measurement pipeline.

    Implements the complete fitness pipeline from Definition def-fitness-potential-operator
    in 03_cloning.md:

    1. Select random alive companions for diversity measurement
    2. Compute algorithmic distances: d_alg(i,j)² = ||x_i - x_j||² + λ_alg ||v_i - v_j||²
    3. Standardize rewards using patched standardization (only alive walkers)
    4. Standardize distances using patched standardization (only alive walkers)
    5. Apply logistic rescale: g_A(z) = A / (1 + exp(-z))
    6. Add positivity floor η
    7. Combine channels: V_i = (d'_i)^β · (r'_i)^α

    Reference: Chapter 5, Section 5.6 in 03_cloning.md

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N]
        alive: Boolean mask [N], True for alive walkers
        alpha: Reward channel exponent (default: 1.0)
        beta: Diversity channel exponent (default: 1.0)
        eta: Positivity floor parameter (default: 0.1)
        lambda_alg: Algorithmic distance velocity weight (default: 0.0, position-only)
        sigma_min: Regularization for patched standardization (default: 1e-8)
        A: Upper bound for logistic rescale (default: 2.0)

    Returns:
        fitness: Fitness potential vector [N], zero for dead walkers
        distances: Algorithmic distances to companions [N]
        companions: Companion indices [N]

    Note:
        - Dead walkers receive fitness = 0.0 and are excluded from statistics
        - Companion selection is uniform random from alive walkers
        - For λ_alg = 0: position-only distance (spatial proximity)
        - For λ_alg > 0: phase-space distance (kinematic similarity)
        - For λ_alg = 1: balanced phase-space model
    """
    # Step 1: Select random alive companions for diversity measurement
    companions = companion_selection(x=positions, v=velocities, alive_mask=alive, lambda_alg=lambda_alg)

    # Step 2: Compute algorithmic distances in phase space
    # d_alg(i,j)² = ||x_i - x_j||² + λ_alg ||v_i - v_j||²
    pos_diff = positions - positions[companions]
    vel_diff = velocities - velocities[companions]
    distances = torch.sqrt(
        (pos_diff**2).sum(dim=-1) + lambda_alg * (vel_diff**2).sum(dim=-1)
    )

    # Step 3-4: Patched standardization for both channels (only alive walkers)
    z_rewards = patched_standardization(rewards, alive, rho=None, sigma_min=sigma_min)
    z_distances = patched_standardization(distances, alive, rho=None, sigma_min=sigma_min)

    # Step 5-6: Logistic rescale + positivity floor
    # r'_i = g_A(z_r,i) + η, d'_i = g_A(z_d,i) + η
    r_prime = logistic_rescale(z_rewards, A=A) + eta
    d_prime = logistic_rescale(z_distances, A=A) + eta

    # Step 7: Combine channels into fitness potential
    # V_i = (d'_i)^β · (r'_i)^α
    fitness = (d_prime**beta) * (r_prime**alpha)

    # Dead walkers receive fitness = 0.0 (they don't participate in cloning)
    fitness = torch.where(alive, fitness, torch.zeros_like(fitness))

    return fitness, distances, companions


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