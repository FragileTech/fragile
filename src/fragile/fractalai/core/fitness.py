import warnings

import panel as pn
import param
import torch
from torch import Tensor

from fragile.fractalai.core.panel_model import INPUT_WIDTH, PanelModel


try:
    from torch.func import jacrev

    TORCH_FUNC_AVAILABLE = True
except ImportError:
    TORCH_FUNC_AVAILABLE = False


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


def compute_localization_weights(
    positions: Tensor,
    velocities: Tensor,
    alive: Tensor,
    rho: float,
    lambda_alg: float = 0.0,
    bounds=None,
    pbc: bool = False,
) -> Tensor:
    """Compute ρ-localized weights K_ρ(i,j) for statistical neighborhoods.

    Implements the localization kernel from Definition def-localized-mean-field-moments
    in 13_fractal_set_new/01_fractal_set.md:

        K_ρ(i,j) = exp(-d_alg²(i,j) / (2ρ²))

    where d_alg²(i,j) = ||x_i - x_j||² + λ_alg ||v_i - v_j||² is the algorithmic
    distance in phase space.

    The weights define a ρ-neighborhood around each walker i, with contributions
    from walker j decaying exponentially with algorithmic distance. Dead walkers
    are excluded (weight = 0).

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        alive: Boolean mask [N], True for alive walkers
        rho: Localization scale parameter (controls neighborhood size)
        lambda_alg: Velocity weight in algorithmic distance (default: 0.0)
        bounds: Domain bounds (required if pbc=True)
        pbc: If True, use periodic boundary conditions for position distances

    Returns:
        weights: Tensor [N, N] where weights[i,j] = K_ρ(i,j) * alive[j]
                 Dead walkers have weights[i, j] = 0 for all i

    Note:
        - For small ρ: local regime (few neighbors contribute)
        - For large ρ: mean-field regime (many neighbors contribute)
        - ρ → ∞ recovers uniform weights (global statistics)
        - Fully differentiable for gradient-based methods
        - With pbc=True, position distances use minimum image convention
    """
    # Compute pairwise algorithmic distances [N, N] using periodic distance if enabled
    from fragile.fractalai.core.companion_selection import compute_algorithmic_distance_matrix

    d_alg_sq = compute_algorithmic_distance_matrix(positions, velocities, lambda_alg, bounds, pbc)

    # Localization kernel: K_ρ(i,j) = exp(-d_alg²/(2ρ²))
    K_rho = torch.exp(-d_alg_sq / (2 * rho**2))  # [N, N]

    # Mask dead walkers: only alive walkers contribute to statistics
    # Both source and target must be alive for nonzero weight
    alive_mask_2d = alive.unsqueeze(1).float() * alive.unsqueeze(0).float()  # [N, N]
    # Use multiplication instead of *= to avoid in-place operation (preserves gradients)
    return K_rho * alive_mask_2d


def localized_statistics(
    values: Tensor,
    weights: Tensor,
    sigma_min: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Compute ρ-localized mean and standard deviation for each walker.

    Implements ρ-localized statistics from Definition def-localized-mean-field-moments
    in 13_fractal_set_new/01_fractal_set.md:

        μ_ρ(i) = Σ_j K_ρ(i,j) v_j / Σ_j K_ρ(i,j)
        σ²_ρ(i) = Σ_j K_ρ(i,j) (v_j - μ_ρ(i))² / Σ_j K_ρ(i,j)

    where K_ρ(i,j) are the localization weights computed by compute_localization_weights().

    Each walker i has its own local mean μ_ρ(i) and local std σ_ρ(i) computed from
    its ρ-neighborhood, making the statistics local fields rather than global constants.

    Args:
        values: Tensor [N] containing measurement values for all walkers
        weights: Localization weights [N, N] from compute_localization_weights()
        sigma_min: Regularization constant ensuring σ'_ρ(i) ≥ σ_min > 0

    Returns:
        mu_rho: Local means [N], where mu_rho[i] = μ_ρ(i)
        sigma_rho: Regularized local stds [N], where sigma_rho[i] = √(σ²_ρ(i) + σ²_min)

    Note:
        - Uses regularized std: σ'_ρ(i) = √(σ²_ρ(i) + σ²_min) for stability
        - Fully differentiable for gradient-based methods
        - If all weights for walker i are zero (isolated dead walker), returns
          mu_rho[i] = 0, sigma_rho[i] = sigma_min
    """
    # Normalize weights for each walker
    # weight_sum[i] = Σ_j K_ρ(i,j) (total weight from all neighbors)
    weight_sum = weights.sum(dim=1, keepdim=True)  # [N, 1]

    # Handle edge case: walker with no alive neighbors (avoid division by zero)
    weight_sum_safe = torch.clamp(weight_sum, min=1e-10)

    # Normalized weights: w_ij = K_ρ(i,j) / Σ_k K_ρ(i,k)
    weights_norm = weights / weight_sum_safe  # [N, N]

    # Local mean: μ_ρ(i) = Σ_j w_ij v_j
    mu_rho = torch.matmul(weights_norm, values)  # [N]

    # Local variance: σ²_ρ(i) = Σ_j w_ij (v_j - μ_ρ(i))²
    # Broadcasting: values[j] - mu_rho[i] for all pairs (i,j)
    centered = values.unsqueeze(0) - mu_rho.unsqueeze(1)  # [N, N]
    sigma_sq_rho = torch.sum(weights_norm * centered**2, dim=1)  # [N]

    # Regularized std: σ'_ρ(i) = √(σ²_ρ(i) + σ²_min)
    sigma_rho = torch.sqrt(sigma_sq_rho + sigma_min**2)  # [N]

    return mu_rho, sigma_rho


def patched_standardization(
    values: Tensor,
    alive: Tensor,
    positions: Tensor | None = None,
    velocities: Tensor | None = None,
    rho: float | None = None,
    lambda_alg: float = 0.0,
    sigma_min: float = 1e-8,
    return_statistics: bool = False,
    bounds=None,
    pbc: bool = False,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """Compute Z-scores using only alive walkers for statistics (fully differentiable).

    Implements the patched standardization Z_ρ[f, d, x] where statistics (mean, std)
    are computed only over alive walkers to prevent contamination from dead walkers.

    **Two regimes:**

    1. **Mean-field regime** (rho=None, default):
       Computes global statistics over all alive walkers:
           Z[f, d, x_i] = (v_i - μ[v|alive]) / σ'[v|alive]
       where μ and σ are global constants computed from all alive walkers.
       Computational cost: O(N) - efficient for large swarms.

    2. **Local regime** (rho is finite):
       Computes ρ-localized statistics for each walker:
           Z_ρ[f, d, x_i] = (v_i - μ_ρ(i)) / σ'_ρ(i)
       where μ_ρ(i), σ_ρ(i) are computed from the ρ-neighborhood of walker i
       using localization kernel K_ρ(i,j) = exp(-d_alg²(i,j)/(2ρ²)).
       Computational cost: O(N²) - accurate for local gauge theory.

    Reference: Definition def-unified-z-score and def-localized-mean-field-moments
    in 13_fractal_set_new/01_fractal_set.md

    **Differentiability**: Uses masked element-wise operations instead of boolean
    indexing to preserve second-order gradients for Hessian computation.

    Args:
        values: Tensor [N] containing measurement values for all walkers
        alive: Boolean tensor [N], True for alive walkers
        positions: Walker positions [N, d], required if rho is not None
        velocities: Walker velocities [N, d], required if rho is not None
        rho: Localization scale parameter. If None, uses global statistics (mean-field).
             If finite, uses ρ-localized statistics (local regime).
        lambda_alg: Velocity weight in algorithmic distance (default: 0.0, position-only)
        sigma_min: Regularization constant ensuring σ' ≥ σ_min > 0
        return_statistics: If True, return (z_scores, mu, sigma) tuple.
            For mean-field: mu is scalar, sigma is scalar
            For local: mu is [N], sigma is [N]
        bounds: Domain bounds (required if pbc=True)
        pbc: If True, use periodic boundary conditions for distance calculations

    Returns:
        If return_statistics=False:
            Z-scores tensor [N]. Dead walkers receive Z-score of 0.0.
        If return_statistics=True:
            - Mean-field regime: (z_scores [N], mu [scalar], sigma [scalar])
            - Local regime: (z_scores [N], mu [N], sigma [N])

    Raises:
        ValueError: If rho is not None but positions or velocities are None

    Note:
        For small ρ (local regime): statistics become local fields, enabling local
        gauge theory interpretation.
        For large ρ or rho=None (mean-field): statistics are global, giving mean-field
        interpretation.
    """

    def _global_stats(values_tensor: Tensor, alive_mask: Tensor) -> tuple[Tensor, Tensor]:
        n_alive = alive_mask.sum()
        n_alive_safe = torch.clamp(n_alive, min=1.0)
        mu = (values_tensor * alive_mask).sum() / n_alive_safe
        centered = values_tensor - mu
        sigma_sq = ((centered**2) * alive_mask).sum() / n_alive_safe
        sigma_reg = torch.sqrt(sigma_sq + sigma_min**2)
        return mu, sigma_reg

    # Branch 1: MEAN-FIELD REGIME (rho=None) - Global statistics, O(N) cost
    if rho is None:
        # Convert boolean mask to float for differentiable operations
        # This preserves gradients where boolean indexing would break them
        alive_mask = alive.float()  # [N], 1.0 for alive, 0.0 for dead

        mu, sigma_reg = _global_stats(values, alive_mask)

        # Compute Z-scores for all walkers
        z_scores = (values - mu) / sigma_reg

        # Mask dead walkers (set to 0.0)
        # Using multiplication instead of torch.where to preserve gradients
        z_scores_masked = z_scores * alive_mask

        if return_statistics:
            return z_scores_masked, mu, sigma_reg
        return z_scores_masked

    # Branch 2: LOCAL REGIME (finite rho) - ρ-localized statistics, O(N²) cost
    # Validate required arguments
    if positions is None or velocities is None:
        msg = (
            "positions and velocities are required for localized standardization (rho is not None)"
        )
        raise ValueError(msg)

    # Compute localization weights K_ρ(i,j) for all pairs
    weights = compute_localization_weights(
        positions=positions,
        velocities=velocities,
        alive=alive,
        rho=rho,
        lambda_alg=lambda_alg,
        bounds=bounds,
        pbc=pbc,
    )  # [N, N]

    if weights.shape[0] > 0:
        eye = torch.eye(weights.shape[0], device=weights.device, dtype=weights.dtype)
        # Avoid self-dominated neighborhoods collapsing local statistics.
        weights = weights * (1.0 - eye)

    # Compute ρ-localized statistics for each walker
    mu_rho, sigma_rho = localized_statistics(
        values=values, weights=weights, sigma_min=sigma_min
    )  # [N], [N]

    alive_mask = alive.to(values.dtype)
    neighbor_mass = weights.sum(dim=1)
    min_local_mass = torch.finfo(weights.dtype).eps
    fallback_mask = (neighbor_mass <= min_local_mass) & alive
    if fallback_mask.any():
        mu_global, sigma_global = _global_stats(values, alive_mask)
        mu_rho = torch.where(fallback_mask, mu_global, mu_rho)
        sigma_rho = torch.where(fallback_mask, sigma_global, sigma_rho)

    # Compute Z-scores using LOCAL statistics
    # Z_ρ[f, d, x_i] = (v_i - μ_ρ(i)) / σ'_ρ(i)
    z_scores = (values - mu_rho) / sigma_rho  # [N]

    # Mask dead walkers (set to 0.0)
    z_scores_masked = z_scores * alive_mask

    if return_statistics:
        return z_scores_masked, mu_rho, sigma_rho
    return z_scores_masked


def compute_fitness(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor,
    alive: Tensor,
    companions: Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    eta: float = 0.1,
    lambda_alg: float = 0.0,
    sigma_min: float = 1e-8,
    A: float = 2.0,
    epsilon_dist: float = 1e-8,
    rho: float | None = None,
    bounds=None,
    pbc: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
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

    **Two regimes based on rho parameter:**

    - **Mean-field** (rho=None): Uses global statistics μ, σ computed over all alive walkers.
      Computational cost: O(N). Suitable for large swarms and mean-field physics.

    - **Local** (finite rho): Uses ρ-localized statistics μ_ρ(i), σ_ρ(i) computed from
      each walker's ρ-neighborhood. Computational cost: O(N²). Enables local gauge
      theory interpretation.

    Reference: Chapter 5, Section 5.6 in 03_cloning.md

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N]
        alive: Boolean mask [N], True for alive walkers
        companions: Companion indices [N] (must be provided, not selected here)
        alpha: Reward channel exponent (default: 1.0)
        beta: Diversity channel exponent (default: 1.0)
        eta: Positivity floor parameter (default: 0.1)
        lambda_alg: Algorithmic distance velocity weight (default: 0.0, position-only)
        sigma_min: Regularization for patched standardization (default: 1e-8)
        A: Upper bound for logistic rescale (default: 2.0)
        epsilon_dist: Distance regularization for C^∞ smoothness (default: 1e-8)
        rho: Localization scale parameter (default: None for mean-field).
             If finite, enables ρ-localized statistics for local regime.

    Returns:
        fitness: Fitness potential vector [N], zero for dead walkers
        info: Dictionary with intermediate values for diagnostics and data tracking. Each \
            entry is a tensor of shape [N]. Keys:
            - "distances": Algorithmic distances to companions [N]
            - "companions": Companion indices [N]
            - "z_rewards": Z-scores of rewards [N]
            - "z_distances": Z-scores of distances [N]
            - "pos_squared_differences": Squared position differences ||x_i - x_j||² [N]
            - "vel_squared_differences": Squared velocity differences ||v_i - v_j||² [N]
            - "rescaled_rewards": Rescaled rewards r'_i [N]
            - "rescaled_distances": Rescaled distances d'_i [N]

    Note:
        - Dead walkers receive fitness = 0.0 and are excluded from statistics
        - Distance uses regularization: d = sqrt(||Δx||² + ε²) to ensure smoothness at origin
        - This prevents NaN gradients when walkers are self-paired (companions[i] = i)
        - For λ_alg = 0: position-only distance (spatial proximity)
        - For λ_alg > 0: phase-space distance (kinematic similarity)
        - For λ_alg = 1: balanced phase-space model
    """
    # Step 2: Compute regularized algorithmic distances in phase space
    # Regularized distance: d_alg(i,j) = sqrt(||x_i - x_j||² + λ_alg ||v_i - v_j||² + ε²)
    # The epsilon term ensures C^∞ differentiability at the origin (prevents NaN gradients)
    pos_diff = positions - positions[companions]

    # Apply minimum image convention for periodic boundary conditions
    # This ensures shortest distance through wrapping: dx_wrapped = dx - L * round(dx / L)
    if pbc and bounds is not None:
        L = bounds.high - bounds.low  # Domain size [d]
        pos_diff -= L * torch.round(pos_diff / L)  # Wrap to [-L/2, L/2]

    vel_diff = velocities - velocities[companions]  # Velocities never use PBC
    pos_sq = (pos_diff**2).sum(dim=-1)
    vel_sq = (vel_diff**2).sum(dim=-1)
    distances = torch.sqrt(pos_sq + lambda_alg * vel_sq + epsilon_dist**2)

    # Step 3-4: Patched standardization for both channels (only alive walkers)
    # Get statistics for localized mean-field analysis
    # Pass positions/velocities for localized regime (rho not None)
    z_rewards, mu_rewards, sigma_rewards = patched_standardization(
        values=rewards,
        alive=alive,
        positions=positions,
        velocities=velocities,
        rho=rho,
        lambda_alg=lambda_alg,
        sigma_min=sigma_min,
        return_statistics=True,
        bounds=bounds,
        pbc=pbc,
    )
    z_distances, mu_distances, sigma_distances = patched_standardization(
        values=distances,
        alive=alive,
        positions=positions,
        velocities=velocities,
        rho=rho,
        lambda_alg=lambda_alg,
        sigma_min=sigma_min,
        return_statistics=True,
        bounds=bounds,
        pbc=pbc,
    )

    # Step 5-6: Logistic rescale + positivity floor
    # r'_i = g_A(z_r,i) + η, d'_i = g_A(z_d,i) + η
    r_prime = logistic_rescale(z_rewards, A=A) + eta
    d_prime = logistic_rescale(z_distances, A=A) + eta

    # Step 7: Combine channels into fitness potential
    # V_i = (d'_i)^β · (r'_i)^α
    fitness = (d_prime**beta) * (r_prime**alpha)

    # Dead walkers receive fitness = 0.0 (they don't participate in cloning)
    fitness = torch.where(alive, fitness, torch.zeros_like(fitness))
    info = {
        "distances": distances,
        "companions": companions,
        "z_rewards": z_rewards,
        "z_distances": z_distances,
        "pos_squared_differences": pos_sq,
        "vel_squared_differences": vel_sq,
        "rescaled_rewards": r_prime,
        "rescaled_distances": d_prime,
        # Localized statistics (global case: rho → ∞)
        "mu_rewards": mu_rewards,  # μ_ρ[r|alive]
        "sigma_rewards": sigma_rewards,  # σ'_ρ[r|alive]
        "mu_distances": mu_distances,  # μ_ρ[d|alive]
        "sigma_distances": sigma_distances,  # σ'_ρ[d|alive]
    }

    return fitness, info


class FitnessOperator(PanelModel):
    """Fitness operator with automatic differentiation for Langevin dynamics.

    This class provides:
    1. Fitness potential V_i(x, v, rewards, alive, companions)
    2. First derivative ∂V_i/∂x_i for fitness-based force in Langevin dynamics
    3. Second derivative ∂²V_i/∂x_i² for state-dependent diffusion tensor

    The fitness potential is computed using the Euclidean Gas measurement pipeline
    from Definition def-fitness-potential-operator in 03_cloning.md.

    Mathematical notation from Definition def-fitness-potential-operator in 03_cloning.md:
    - α (alpha): Reward channel exponent (default: 1.0)
    - β (beta): Diversity channel exponent (default: 1.0)
    - η (eta): Positivity floor parameter (default: 0.1)
    - λ_alg (lambda_alg): Velocity weight in algorithmic distance (default: 0.0)
    - σ_min (sigma_min): Regularization for patched standardization (default: 1e-8)
    - ε_dist (epsilon_dist): Regularization for distance smoothness (default: 1e-8)
    - A: Upper bound for logistic rescale (default: 2.0)

    Reference: Chapter 5, Section 5.6 in 03_cloning.md

    Method Selection Guide (based on benchmarks):
    -----------------------------------------------

    **For GRADIENTS** (first derivative):
        - **Recommended**: `compute_gradient_func()` using torch.func.jacrev
        - ~2x faster than autograd loops
        - Requires PyTorch >= 2.0
        - Fallback: `compute_gradient()` for PyTorch < 2.0

    **For HESSIAN DIAGONAL** (second derivative diagonal):
        - **For small N (<50)**: `compute_hessian(..., diagonal_only=True)` (autograd loops)
        - **For large N (>50)**: `compute_hessian_func(..., diagonal_only=True)` (torch.func)
        - torch.func.jacrev can be faster for moderate N, but slower for very small N
        - HVP method is slower and not recommended

    **For FULL HESSIAN** (complete second derivative tensor):
        - **Always use**: `compute_hessian(..., diagonal_only=False)` (autograd loops)
        - Most memory-efficient (O(Nd²) vs O(N²d²))
        - Fastest for small N (typical use case)

    Note:
        Derivatives are computed w.r.t. positions x, treating companions as fixed,
        and extracting only per-walker blocks (no cross-derivatives through other walkers).
        This gives the "instantaneous force" for the current companion assignment.
        For mean-field forces, expectation over companions would be needed (future work).
        Companion selection is handled externally (by EuclideanGas) - this operator
        only computes fitness given pre-selected companions.
    """

    _n_widget_columns = param.Integer(default=2, bounds=(1, None), doc="Number of widget columns")
    _max_widget_width = param.Integer(default=800, bounds=(0, None), doc="Maximum widget width")

    # Fitness parameters
    alpha = param.Number(
        default=1.0,
        bounds=(0, None),
        softbounds=(0.01, 5.0),
        inclusive_bounds=(False, True),
        doc="Reward channel exponent (α)",
    )
    beta = param.Number(
        default=1.0,
        bounds=(0, None),
        softbounds=(0.01, 5.0),
        inclusive_bounds=(False, True),
        doc="Diversity channel exponent (β)",
    )
    eta = param.Number(
        default=0.1,
        bounds=(0, None),
        softbounds=(0.001, 0.5),
        inclusive_bounds=(False, True),
        doc="Positivity floor parameter (η)",
    )
    lambda_alg = param.Number(
        default=0.0,
        bounds=(0, None),
        softbounds=(0.0, 1.0),
        doc="Velocity weight in algorithmic distance (λ_alg)",
    )
    sigma_min = param.Number(
        default=1e-8,
        bounds=(0, None),
        softbounds=(1e-9, 1e-3),
        inclusive_bounds=(False, True),
        doc="Regularization for patched standardization (σ_min)",
    )
    epsilon_dist = param.Number(
        default=1e-8,
        bounds=(0, None),
        softbounds=(1e-10, 1e-6),
        inclusive_bounds=(False, True),
        doc=(
            "Distance regularization for smoothness: "
            "d = sqrt(||Δx||² + ε²) ensures C^∞ differentiability"
        ),
    )
    A = param.Number(
        default=2.0,
        bounds=(0, None),
        softbounds=(1.0, 5.0),
        inclusive_bounds=(False, True),
        doc="Upper bound for logistic rescale",
    )
    rho = param.Number(
        default=None,
        allow_None=True,
        doc=(
            "Localization scale parameter (ρ). "
            "If None (default), uses global statistics (mean-field regime, O(N) cost). "
            "If finite, uses ρ-localized statistics (local regime, O(N²) cost). "
            "Small ρ enables local gauge theory interpretation."
        ),
    )

    @property
    def widgets(self) -> dict[str, dict]:
        """Widget configurations for fitness parameters."""
        return {
            "alpha": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "α (reward exponent)",
                "start": 0.1,
                "end": 5.0,
                "step": 0.05,
            },
            "beta": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "β (diversity exponent)",
                "start": 0.5,
                "end": 5.0,
                "step": 0.1,
            },
            "eta": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "η (positivity floor)",
                "start": 0.001,
                "end": 0.1,
                "step": 0.001,
            },
            "lambda_alg": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "λ_alg (velocity weight)",
                "start": 0.0,
                "end": 3.0,
                "step": 0.05,
            },
            "sigma_min": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "σ_min (standardization reg)",
                "start": 1e-9,
                "end": 1e-3,
                "step": 1e-9,
                "format": "%.1e",
            },
            "epsilon_dist": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "ε_dist (distance reg)",
                "start": 1e-10,
                "end": 1e-6,
                "step": 1e-10,
                "format": "%.1e",
            },
            "A": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "A (rescale bound)",
                "start": 1.0,
                "end": 5.0,
                "step": 0.1,
            },
            "rho": {
                "type": pn.widgets.FloatInput,
                "width": INPUT_WIDTH,
                "name": "ρ (localization scale)",
                "value": None,
            },
        }

    @property
    def widget_parameters(self) -> list[str]:
        """Parameters to display in UI."""
        return ["alpha", "beta", "eta", "lambda_alg", "sigma_min", "epsilon_dist", "A", "rho"]

    def __call__(
        self,
        positions: Tensor,
        velocities: Tensor,
        rewards: Tensor,
        alive: Tensor,
        companions: Tensor,
        bounds=None,
        pbc: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute fitness potential using the Euclidean Gas measurement pipeline.

        Wraps the `compute_fitness` function with the operator's parameters.

        Args:
            positions: Walker positions [N, d]
            velocities: Walker velocities [N, d]
            rewards: Raw reward values [N]
            alive: Boolean mask [N], True for alive walkers
            companions: Companion indices [N] (required, selected by EuclideanGas)
            bounds: Domain bounds (required if pbc=True)
            pbc: If True, use periodic boundary conditions for distance calculations

        Returns:
            fitness: Fitness potential vector [N], zero for dead walkers
            info: Dictionary with intermediate values for diagnostics and data tracking. Each \
                entry is a tensor of shape [N]. Keys:
                - "distances": Algorithmic distances to companions [N]
                - "companions": Companion indices [N]
                - "z_rewards": Z-scores of rewards [N]
                - "z_distances": Z-scores of distances [N]
                - "pos_squared_differences": Squared position differences ||x_i - x_j||² [N]
                - "vel_squared_differences": Squared velocity differences ||v_i - v_j||² [N]
                - "rescaled_rewards": Rescaled rewards r'_i [N]
                - "rescaled_distances": Rescaled distances d'_i [N]
        """
        return compute_fitness(
            positions=positions,
            velocities=velocities,
            rewards=rewards,
            alive=alive,
            companions=companions,
            alpha=self.alpha,
            beta=self.beta,
            eta=self.eta,
            lambda_alg=self.lambda_alg,
            sigma_min=self.sigma_min,
            A=self.A,
            epsilon_dist=self.epsilon_dist,
            bounds=bounds,
            pbc=pbc,
            rho=self.rho,
        )

    def compute_gradient(
        self,
        positions: Tensor,
        velocities: Tensor,
        rewards: Tensor,
        alive: Tensor,
        companions: Tensor,
    ) -> Tensor:
        """Compute gradient ∂V/∂x for fitness-based force in Langevin dynamics.

        Uses automatic differentiation to compute the gradient of each walker's
        fitness potential w.r.t. its own position. The gradient provides the force term
        for adaptive Langevin dynamics:

            F_fit,i(x) = -∂V_i/∂x_i

        Args:
            positions: Walker positions [N, d] (requires_grad will be enabled)
            velocities: Walker velocities [N, d]
            rewards: Raw reward values [N]
            alive: Boolean mask [N], True for alive walkers
            companions: Companion indices [N] (required, selected by EuclideanGas)

        Returns:
            Gradient tensor [N, d] where grad[i] = ∂V_i/∂x_i

        Note:
            The gradient is computed treating companions as fixed and extracting
            only the per-walker block (no cross-derivatives through other walkers).
            This gives the instantaneous force for the current companion assignment.
        """
        # Enable gradient tracking on positions
        positions_grad = positions.clone().detach().requires_grad_(True)  # noqa: FBT003

        # Compute fitness
        fitness, _ = compute_fitness(
            positions=positions_grad,
            velocities=velocities,
            rewards=rewards,
            alive=alive,
            companions=companions,
            alpha=self.alpha,
            beta=self.beta,
            eta=self.eta,
            lambda_alg=self.lambda_alg,
            sigma_min=self.sigma_min,
            A=self.A,
            epsilon_dist=self.epsilon_dist,
            rho=self.rho,
        )
        if not fitness.requires_grad:
            warnings.warn(
                "Fitness gradient unavailable (fitness does not require grad); "
                "returning zero gradients.",
                RuntimeWarning,
            )
            return torch.zeros_like(positions_grad)

        # Compute per-walker gradient: ∂V_i/∂positions_i (ignore cross-terms)
        N = fitness.shape[0]
        grad = torch.zeros_like(positions_grad)
        for i in range(N):
            if not fitness[i].requires_grad:
                continue
            (grad_i,) = torch.autograd.grad(
                outputs=fitness[i],
                inputs=positions_grad,
                create_graph=False,
                retain_graph=i < N - 1,
                allow_unused=True,
            )
            if grad_i is None:
                continue
            grad[i] = grad_i[i]

        return grad

    def compute_hessian(
        self,
        positions: Tensor,
        velocities: Tensor,
        rewards: Tensor,
        alive: Tensor,
        companions: Tensor,
        diagonal_only: bool = True,
    ) -> Tensor:
        """Compute Hessian ∂²V/∂x² for state-dependent diffusion tensor.

        Uses automatic differentiation to compute the Hessian (second derivative)
        of each walker's fitness potential w.r.t. its own position. The Hessian provides
        the state-dependent diffusion tensor for adaptive Langevin dynamics:

            D(x) = f(∂²V/∂x²)

        where f is some function (e.g., absolute value, eigenvalue decomposition).

        Args:
            positions: Walker positions [N, d] (requires_grad will be enabled)
            velocities: Walker velocities [N, d]
            rewards: Raw reward values [N]
            alive: Boolean mask [N], True for alive walkers
            companions: Companion indices [N] (required, selected by EuclideanGas)
            diagonal_only: If True, return only diagonal elements [N, d].
                          If False, return full Hessian [N, d, d] (expensive!)

        Returns:
            If diagonal_only=True: Diagonal Hessian [N, d] where hess[i, j] = ∂²V_i/∂x_ij²
            If diagonal_only=False: Full Hessian [N, d, d] where hess[i, j, k] = ∂²V_i/∂x_ij∂x_ik

        Note:
            - Full Hessian computation is O(N*d²) and can be very expensive
            - Diagonal approximation is O(N*d) and sufficient for many diffusion models
            - The Hessian is computed treating companions as fixed and extracting
              only per-walker blocks (no cross-derivatives through other walkers)
        """
        N, d = positions.shape

        # Enable gradient tracking on positions
        positions_grad = positions.clone().detach().requires_grad_(True)  # noqa: FBT003

        # Compute fitness
        fitness, _ = compute_fitness(
            positions=positions_grad,
            velocities=velocities,
            rewards=rewards,
            alive=alive,
            companions=companions,
            alpha=self.alpha,
            beta=self.beta,
            eta=self.eta,
            lambda_alg=self.lambda_alg,
            sigma_min=self.sigma_min,
            A=self.A,
            epsilon_dist=self.epsilon_dist,
            rho=self.rho,
        )
        if not fitness.requires_grad:
            warnings.warn(
                "Fitness Hessian unavailable (fitness does not require grad); "
                "returning zero Hessian.",
                RuntimeWarning,
            )
            if diagonal_only:
                return torch.zeros_like(positions_grad)
            return torch.zeros(N, d, d, dtype=positions.dtype, device=positions.device)

        if diagonal_only:
            # Compute diagonal Hessian elements ∂²V_i/∂positions_i²
            hessian_diag = torch.zeros_like(positions_grad)

            for i in range(N):
                if not fitness[i].requires_grad:
                    continue
                (grad_i,) = torch.autograd.grad(
                    outputs=fitness[i],
                    inputs=positions_grad,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )
                if grad_i is None:
                    continue
                for j in range(d):
                    grad_component = grad_i[i, j]
                    if not grad_component.requires_grad:
                        continue
                    (hess_i,) = torch.autograd.grad(
                        outputs=grad_component,
                        inputs=positions_grad,
                        create_graph=False,
                        retain_graph=not (i == N - 1 and j == d - 1),
                        allow_unused=True,
                    )
                    if hess_i is None:
                        continue
                    hessian_diag[i, j] = hess_i[i, j]

            return hessian_diag

        # Compute full Hessian blocks (expensive!)
        hessian_full = torch.zeros(N, d, d, dtype=positions.dtype, device=positions.device)

        for i in range(N):
            if not fitness[i].requires_grad:
                continue
            (grad_i,) = torch.autograd.grad(
                outputs=fitness[i],
                inputs=positions_grad,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            if grad_i is None:
                continue
            for j in range(d):
                grad_component = grad_i[i, j]
                if not grad_component.requires_grad:
                    continue
                (hess_i,) = torch.autograd.grad(
                    outputs=grad_component,
                    inputs=positions_grad,
                    create_graph=False,
                    retain_graph=not (i == N - 1 and j == d - 1),
                    allow_unused=True,
                )
                if hess_i is None:
                    continue
                hessian_full[i, :, j] = hess_i[i]

        return hessian_full

    def compute_gradient_func(
        self,
        positions: Tensor,
        velocities: Tensor,
        rewards: Tensor,
        alive: Tensor,
        companions: Tensor,
    ) -> Tensor:
        """Compute gradient ∂V/∂x using torch.func.jacrev (fully vectorized, no loops).

        This method uses torch.func.jacrev to compute the full Jacobian of the fitness
        vector w.r.t. all positions, then extracts the diagonal blocks to get each
        walker's gradient w.r.t. its own position.

        This is fully vectorized and typically faster than the autograd loop version
        for moderate numbers of walkers.

        Args:
            positions: Walker positions [N, d]
            velocities: Walker velocities [N, d]
            rewards: Raw reward values [N]
            alive: Boolean mask [N], True for alive walkers
            companions: Companion indices [N] (required, selected by EuclideanGas)

        Returns:
            Gradient tensor [N, d] where grad[i] = ∂fitness_i/∂position_i

        Raises:
            RuntimeError: If torch.func is not available (PyTorch < 2.0)

        Note:
            - Computes full Jacobian [N, N, d] then extracts diagonal [N, d]
            - Memory usage: O(N²d) for Jacobian
            - Suitable for N < ~1000 walkers
            - For very large N, use compute_gradient() which uses O(Nd) memory
        """
        if not TORCH_FUNC_AVAILABLE:
            msg = "torch.func not available. Use compute_gradient() or upgrade PyTorch >= 2.0"
            raise RuntimeError(msg)

        N = positions.shape[0]

        # Define fitness function: positions -> vector of per-walker fitness
        def fitness_vec_fn(pos: Tensor) -> Tensor:
            """Compute fitness for all walkers given all positions."""
            fitness, _ = compute_fitness(
                positions=pos,
                velocities=velocities,
                rewards=rewards,
                alive=alive,
                companions=companions,
                alpha=self.alpha,
                beta=self.beta,
                eta=self.eta,
                lambda_alg=self.lambda_alg,
                sigma_min=self.sigma_min,
                A=self.A,
                epsilon_dist=self.epsilon_dist,
                rho=self.rho,
            )
            return fitness

        # Compute full Jacobian: [N, N, d], then take diagonal blocks
        jac = jacrev(fitness_vec_fn)(positions)
        idx = torch.arange(N, device=positions.device)
        return jac[idx, idx]

    def compute_hessian_func(
        self,
        positions: Tensor,
        velocities: Tensor,
        rewards: Tensor,
        alive: Tensor,
        companions: Tensor,
        diagonal_only: bool = True,
    ) -> Tensor:
        """Compute Hessian ∂²V/∂x² using torch.func.jacrev (fully vectorized, no loops).

        This method uses torch.func.jacrev to compute the per-walker gradient field,
        then differentiates it to extract each walker's Hessian block.

        Args:
            positions: Walker positions [N, d]
            velocities: Walker velocities [N, d]
            rewards: Raw reward values [N]
            alive: Boolean mask [N], True for alive walkers
            companions: Companion indices [N] (required, selected by EuclideanGas)
            diagonal_only: If True, return only diagonal elements [N, d].
                          If False, return full per-walker Hessian [N, d, d]

        Returns:
            If diagonal_only=True: Diagonal Hessian [N, d]
            If diagonal_only=False: Full per-walker Hessian [N, d, d]

        Raises:
            RuntimeError: If torch.func is not available (PyTorch < 2.0)

        Note:
            - Memory usage: O(N²d²) for full Hessian computation
            - Suitable for small to moderate N (< ~100 walkers)
            - For large N, use compute_hessian() which computes elements iteratively
        """
        if not TORCH_FUNC_AVAILABLE:
            msg = "torch.func not available. Use compute_hessian() or upgrade PyTorch >= 2.0"
            raise RuntimeError(msg)

        N, _d = positions.shape

        # Define fitness function: positions -> vector of per-walker fitness
        def fitness_vec_fn(pos: Tensor) -> Tensor:
            """Compute fitness for all walkers given all positions."""
            fitness, _ = compute_fitness(
                positions=pos,
                velocities=velocities,
                rewards=rewards,
                alive=alive,
                companions=companions,
                alpha=self.alpha,
                beta=self.beta,
                eta=self.eta,
                lambda_alg=self.lambda_alg,
                sigma_min=self.sigma_min,
                A=self.A,
                epsilon_dist=self.epsilon_dist,
                rho=self.rho,
            )
            return fitness

        def per_walker_grad(pos: Tensor) -> Tensor:
            jac = jacrev(fitness_vec_fn)(pos)  # [N, N, d]
            idx = torch.arange(pos.shape[0], device=pos.device)
            return jac[idx, idx]  # [N, d]

        # Differentiate the per-walker gradient field
        hess_full = jacrev(per_walker_grad)(positions)  # [N, d, N, d]
        idx = torch.arange(N, device=positions.device)
        hess_full = hess_full[idx, :, idx, :]  # [N, d, d]

        if diagonal_only:
            return hess_full.diagonal(dim1=1, dim2=2)
        return hess_full

    def compute_hessian_hvp(
        self,
        positions: Tensor,
        velocities: Tensor,
        rewards: Tensor,
        alive: Tensor,
        companions: Tensor,
        diagonal_only: bool = True,
    ) -> Tensor:
        """Compute Hessian ∂²V/∂x² using HVP (Hessian-Vector Products).

        This method uses Hessian-Vector Products to compute only the diagonal blocks
        of the Hessian without materializing the full [N, d, N, d] tensor.
        More memory-efficient than compute_hessian_func, but uses loops.

        Args:
            positions: Walker positions [N, d]
            velocities: Walker velocities [N, d]
            rewards: Raw reward values [N]
            alive: Boolean mask [N], True for alive walkers
            companions: Companion indices [N] (required, selected by EuclideanGas)
            diagonal_only: If True, return only diagonal elements [N, d].
                          If False, return full per-walker Hessian [N, d, d]

        Returns:
            If diagonal_only=True: Diagonal Hessian [N, d]
            If diagonal_only=False: Full per-walker Hessian [N, d, d]

        Note:
            - Memory usage: O(Nd²) for diagonal blocks only (vs O(N²d²) for full)
            - More efficient than compute_hessian_func for large N
            - Uses loops over walkers and dimensions (not fully vectorized)
        """
        N, d = positions.shape

        # Compute all Hessian blocks using HVP
        hess_blocks = []

        for i in range(N):

            def fitness_i_fn(pos: Tensor, idx: int = i) -> Tensor:
                fitness, _ = compute_fitness(
                    positions=pos,
                    velocities=velocities,
                    rewards=rewards,
                    alive=alive,
                    companions=companions,
                    alpha=self.alpha,
                    beta=self.beta,
                    eta=self.eta,
                    lambda_alg=self.lambda_alg,
                    sigma_min=self.sigma_min,
                    A=self.A,
                    epsilon_dist=self.epsilon_dist,
                    rho=self.rho,
                )
                return fitness[idx]

            # Compute Hessian block H[i, :, i, :] using HVPs
            hess_block_i = torch.zeros((d, d), device=positions.device, dtype=positions.dtype)

            for j in range(d):
                # Create tangent vector: only perturb position[i, j]
                tangents = torch.zeros_like(positions)
                tangents[i, j] = 1.0

                # Compute HVP: (H @ e_j) where e_j is the j-th basis vector for walker i
                _, hvp_result = torch.autograd.functional.hvp(
                    fitness_i_fn, positions, tangents, create_graph=False
                )

                # Extract the i-th walker's response (this is column j of H[i, :, i, :])
                hess_block_i[:, j] = hvp_result[i]

            hess_blocks.append(hess_block_i)

        # Stack into [N, d, d]
        hess_blocks = torch.stack(hess_blocks)

        if diagonal_only:
            # Extract diagonal elements [N, d]
            return torch.stack([hess_blocks[i].diagonal() for i in range(N)])
        return hess_blocks
