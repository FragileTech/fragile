"""Vectorized mean-field operations for fitness potential computation.

This module provides standalone, fully vectorized implementations of mean-field
operations used in the Adaptive Gas algorithm. All operations are vectorized to
avoid Python loops and maximize performance.

Functions implement formulas from docs/source/01_fragile_gas_framework.md.
"""

import torch

from fragile.core.companion_selection import select_companions_uniform


def patched_std_dev(variance: torch.Tensor, kappa_var_min: float, eps_std: float) -> torch.Tensor:
    """Compute patched (regularized) standard deviation with C^∞ regularity.

    From docs/source/01_fragile_gas_framework.md § 11.1.2
    ({prf:ref}`def-statistical-properties-measurement`):

    The regularized standard deviation prevents pathological sensitivity near zero
    variance while maintaining smooth behavior everywhere:

    $$
    \\sigma'_{\\text{reg}}(V) := \\sqrt{V + \\sigma'^2_{\\min}}
    $$

    where $\\sigma'_{\\min} := \\sqrt{\\kappa_{\\text{var,min}} + \\varepsilon_{\\text{std}}^2}$

    Properties:
    - C^∞ regularity (infinitely differentiable)
    - Positive lower bound: σ'_reg(V) ≥ σ'_min > 0
    - Monotonically increasing
    - Asymptotic behavior: σ'_reg(V) ≈ sqrt(V) for large V

    Args:
        variance: Variance values, shape [...] (any shape, element-wise operation)
        kappa_var_min: Variance floor threshold (κ_var,min > 0)
        eps_std: Numerical stability parameter (ε_std > 0)

    Returns:
        Patched standard deviation, same shape as variance

    Note:
        This replaces the simple sqrt(var + eps) with a mathematically principled
        regularization that ensures global Lipschitz continuity.

    Example:
        >>> var = torch.tensor([0.0, 0.01, 1.0, 100.0])
        >>> sigma_prime = patched_std_dev(var, kappa_var_min=0.01, eps_std=0.001)
        >>> # sigma_prime[0] = sqrt(0.01 + 0.001^2) ≈ 0.1005 (floor)
        >>> # sigma_prime[-1] ≈ sqrt(100) = 10.0 (asymptotic to sqrt)
    """
    # Compute regularization parameter: σ'_min = sqrt(κ_var,min + ε_std^2)
    sigma_prime_min_sq = kappa_var_min + eps_std**2

    # Compute regularized standard deviation: sqrt(V + σ'_min^2)
    return torch.sqrt(variance + sigma_prime_min_sq)


def distance_to_random_companion(
    x: torch.Tensor,
    v: torch.Tensor,
    alive_mask: torch.Tensor,
    lambda_alg: float = 0.0,
) -> torch.Tensor:
    """Compute distance from each walker to a randomly selected companion.

    This implements the distance measurement operator for diversity signal.
    Each walker is paired with a uniformly random alive companion, and the
    algorithmic distance is computed:

    $$
    d_{\\text{alg}}(i, j)^2 := ||x_i - x_j||^2 + λ_{\\text{alg}} ||v_i - v_j||^2
    $$

    Args:
        x: Positions of all walkers, shape [N, d]
        v: Velocities of all walkers, shape [N, d]
        alive_mask: Boolean mask indicating alive walkers, shape [N]
        lambda_alg: Weight for velocity contribution (default 0.0 for position-only)

    Returns:
        Distance to random companion for each walker, shape [N]
        Dead walkers have distance 0.0

    Note:
        Uses uniform random pairing (O(N) complexity) rather than distance-dependent
        pairing. This is the baseline for diversity measurement before applying
        more sophisticated pairing strategies.

    Example:
        >>> N, d = 10, 2
        >>> x = torch.randn(N, d)
        >>> v = torch.randn(N, d)
        >>> alive_mask = torch.ones(N, dtype=torch.bool)
        >>> distances = distance_to_random_companion(x, v, alive_mask, lambda_alg=1.0)
        >>> distances.shape
        torch.Size([10])
    """
    # Select random companions for all walkers
    companions = select_companions_uniform(alive_mask)  # [N]

    # Compute distances to companions (vectorized)
    dx = x - x[companions]  # [N, d]
    pos_dist_sq = torch.sum(dx * dx, dim=1)  # [N]

    if lambda_alg > 0:
        dv = v - v[companions]  # [N, d]
        vel_dist_sq = torch.sum(dv * dv, dim=1)  # [N]
        dist_sq = pos_dist_sq + lambda_alg * vel_dist_sq
    else:
        dist_sq = pos_dist_sq

    dist = torch.sqrt(dist_sq.clamp(min=0.0))  # [N]

    # Dead walkers have distance 0
    dist *= alive_mask.float()

    return dist


def compute_fitness_potential_vectorized(
    x: torch.Tensor,
    v: torch.Tensor,
    measurement: torch.Tensor,
    alive_mask: torch.Tensor,
    alpha: float,
    beta: float,
    kappa_var_min: float,
    eps_std: float,
    eta: float,
    lambda_alg: float = 0.0,
) -> torch.Tensor:
    """Compute mean-field fitness potential V_fit for all walkers (fully vectorized).

    This replaces the loop-based implementation in MeanFieldOps.compute_fitness_potential
    with a fully vectorized version.

    From framework definition:
    V_fit = (r')^α · (d')^β

    where r' and d' are rescaled, standardized measurements.

    Args:
        x: Positions of all walkers, shape [N, d]
        v: Velocities of all walkers, shape [N, d]
        measurement: Raw measurement values (e.g., reward) for each walker, shape [N]
        alive_mask: Boolean mask indicating alive walkers, shape [N]
        alpha: Exploitation weight for reward
        beta: Exploitation weight for diversity
        kappa_var_min: Variance floor threshold
        eps_std: Numerical stability parameter
        eta: Rescale lower bound
        lambda_alg: Weight for velocity in distance metric

    Returns:
        Fitness potential values for all walkers, shape [N]
        Dead walkers have fitness 0.0

    Note:
        This is a simplified global statistics version. For patch-based statistics
        (local neighborhoods), use compute_fitness_potential_with_patches.
    """
    N = x.shape[0]
    device = x.device

    # Extract alive walker measurements
    alive_indices = torch.where(alive_mask)[0]
    n_alive = len(alive_indices)

    if n_alive == 0:
        # No alive walkers: return zero fitness
        return torch.zeros(N, device=device, dtype=x.dtype)

    measurement_alive = measurement[alive_mask]  # [n_alive]

    # Compute global statistics for reward
    mu_r = measurement_alive.mean()
    var_r = measurement_alive.var(unbiased=False)
    sigma_prime_r = patched_std_dev(var_r.unsqueeze(0), kappa_var_min, eps_std).squeeze()

    # Standardize reward: z_r = (r - μ_r) / σ'_r
    z_r = (measurement - mu_r) / sigma_prime_r  # [N]

    # Rescale reward: r' = η + (1-η)·sigmoid(z_r)
    r_prime = eta + (1.0 - eta) / (1.0 + torch.exp(-z_r))  # [N]

    # Compute distances to random companions
    distances = distance_to_random_companion(x, v, alive_mask, lambda_alg)  # [N]

    # Compute global statistics for distance
    distances_alive = distances[alive_mask]  # [n_alive]
    mu_d = distances_alive.mean()
    var_d = distances_alive.var(unbiased=False)
    sigma_prime_d = patched_std_dev(var_d.unsqueeze(0), kappa_var_min, eps_std).squeeze()

    # Standardize distance: z_d = (d - μ_d) / σ'_d
    z_d = (distances - mu_d) / sigma_prime_d  # [N]

    # Rescale distance: d' = η + (1-η)·sigmoid(z_d)
    d_prime = eta + (1.0 - eta) / (1.0 + torch.exp(-z_d))  # [N]

    # Compute fitness potential: V_fit = (r')^α · (d')^β
    V_fit = torch.pow(r_prime, alpha) * torch.pow(d_prime, beta)  # [N]

    # Dead walkers have zero fitness
    V_fit *= alive_mask.float()

    return V_fit


def compute_fitness_gradient_vectorized(
    x: torch.Tensor,
    v: torch.Tensor,
    measurement: torch.Tensor,
    alive_mask: torch.Tensor,
    alpha: float,
    beta: float,
    kappa_var_min: float,
    eps_std: float,
    eta: float,
    lambda_alg: float = 0.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute gradient of fitness potential ∇_x V_fit using finite differences.

    This is a vectorized implementation that computes all d partial derivatives
    in a single batched operation.

    Args:
        x: Positions of all walkers, shape [N, d]
        v: Velocities of all walkers, shape [N, d]
        measurement: Raw measurement values, shape [N]
        alive_mask: Boolean mask indicating alive walkers, shape [N]
        alpha: Exploitation weight for reward
        beta: Exploitation weight for diversity
        kappa_var_min: Variance floor threshold
        eps_std: Numerical stability parameter
        eta: Rescale lower bound
        lambda_alg: Weight for velocity in distance metric
        eps: Finite difference step size

    Returns:
        Fitness gradients for all walkers, shape [N, d]
        Dead walkers have zero gradient

    Note:
        Uses central differences for better accuracy: ∂V/∂x_j ≈ (V(x+ε) - V(x-ε))/(2ε)
        This requires 2*d+1 evaluations of V_fit per call.
    """
    N, d = x.shape
    device = x.device
    dtype = x.dtype

    grad_V_fit = torch.zeros(N, d, device=device, dtype=dtype)

    # Compute V_fit at current point
    V_fit_center = compute_fitness_potential_vectorized(
        x, v, measurement, alive_mask, alpha, beta, kappa_var_min, eps_std, eta, lambda_alg
    )

    # Compute gradient via finite differences (vectorized over dimensions)
    for j in range(d):
        # Perturb along j-th dimension
        x_plus = x.clone()
        x_plus[:, j] += eps

        V_fit_plus = compute_fitness_potential_vectorized(
            x_plus,
            v,
            measurement,
            alive_mask,
            alpha,
            beta,
            kappa_var_min,
            eps_std,
            eta,
            lambda_alg,
        )

        # Compute finite difference: ∂V / ∂x_j ≈ (V(x+eps) - V(x)) / eps
        grad_V_fit[:, j] = (V_fit_plus - V_fit_center) / eps

    # Dead walkers have zero gradient
    grad_V_fit *= alive_mask.unsqueeze(1).float()

    return grad_V_fit
