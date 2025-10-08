"""
Lyapunov Functions for Swarm Convergence Analysis

This module implements Lyapunov functions to numerically verify convergence properties
of Euclidean Gas swarms. It compares two swarm states and returns all relevant terms
as tensors for analysis.

Mathematical Background:
- Lyapunov functions V(t) measure "distance" from equilibrium
- If dV/dt ≤ 0, the system converges
- We decompose V into multiple components:
  * Position variance: spread in state space
  * Velocity variance: kinetic energy distribution
  * Cross-swarm distances: relative positions between swarms
  * Wasserstein-like metrics: distribution matching

References:
- See docs/source/1_fractal_calculus/theorems/ for convergence proofs
"""

from __future__ import annotations

import torch
from torch import Tensor

from fragile.euclidean_gas import EuclideanGas, SwarmState, VectorizedOps


class LyapunovTerms:
    """Container for all Lyapunov function terms."""

    def __init__(
        self,
        # Variance terms (within-swarm spread)
        var_x_1: Tensor,
        var_x_2: Tensor,
        var_v_1: Tensor,
        var_v_2: Tensor,
        # Cross-swarm distance terms
        mean_cross_distance_x: Tensor,
        mean_cross_distance_v: Tensor,
        wasserstein_x: Tensor,  # 1-Wasserstein approximation for positions
        wasserstein_v: Tensor,  # 1-Wasserstein approximation for velocities
        # Mean distance terms (center of mass)
        com_distance_x: Tensor,  # ||μ_x^(1) - μ_x^(2)||
        com_distance_v: Tensor,  # ||μ_v^(1) - μ_v^(2)||
        # Total Lyapunov function
        total: Tensor,
    ):
        """
        Initialize Lyapunov terms.

        Args:
            var_x_1: Position variance of swarm 1
            var_x_2: Position variance of swarm 2
            var_v_1: Velocity variance of swarm 1
            var_v_2: Velocity variance of swarm 2
            mean_cross_distance_x: Mean cross-swarm position distance
            mean_cross_distance_v: Mean cross-swarm velocity distance
            wasserstein_x: 1-Wasserstein distance approximation for positions
            wasserstein_v: 1-Wasserstein distance approximation for velocities
            com_distance_x: Distance between position centers of mass
            com_distance_v: Distance between velocity centers of mass
            total: Total Lyapunov function value
        """
        self.var_x_1 = var_x_1
        self.var_x_2 = var_x_2
        self.var_v_1 = var_v_1
        self.var_v_2 = var_v_2
        self.mean_cross_distance_x = mean_cross_distance_x
        self.mean_cross_distance_v = mean_cross_distance_v
        self.wasserstein_x = wasserstein_x
        self.wasserstein_v = wasserstein_v
        self.com_distance_x = com_distance_x
        self.com_distance_v = com_distance_v
        self.total = total

    def to_dict(self) -> dict[str, Tensor]:
        """
        Convert all terms to a dictionary of tensors.

        Returns:
            Dictionary mapping term names to tensor values
        """
        return {
            "var_x_1": self.var_x_1,
            "var_x_2": self.var_x_2,
            "var_v_1": self.var_v_1,
            "var_v_2": self.var_v_2,
            "mean_cross_distance_x": self.mean_cross_distance_x,
            "mean_cross_distance_v": self.mean_cross_distance_v,
            "wasserstein_x": self.wasserstein_x,
            "wasserstein_v": self.wasserstein_v,
            "com_distance_x": self.com_distance_x,
            "com_distance_v": self.com_distance_v,
            "total": self.total,
        }


def compute_wasserstein_1d_approx(x1: Tensor, x2: Tensor) -> Tensor:
    """
    Compute 1-Wasserstein distance approximation between two point clouds.

    Uses the Hungarian algorithm approximation via sorting along each dimension
    and averaging. This is exact for 1D and a good approximation for higher dimensions.

    W_1(μ_1, μ_2) ≈ (1/d) Σ_j W_1(μ_1^j, μ_2^j)

    where μ_i^j is the marginal distribution along dimension j.

    Args:
        x1: Points from distribution 1 [N1, d]
        x2: Points from distribution 2 [N2, d]

    Returns:
        Approximate 1-Wasserstein distance (scalar)
    """
    # Handle different sample sizes by repeating smaller set
    N1, d = x1.shape
    N2 = x2.shape[0]

    if N1 != N2:
        # Match sizes by repeating
        N_max = max(N1, N2)
        if N1 < N_max:
            # Repeat x1
            repeats = (N_max + N1 - 1) // N1  # Ceiling division
            x1 = x1.repeat(repeats, 1)[:N_max]
        if N2 < N_max:
            # Repeat x2
            repeats = (N_max + N2 - 1) // N2
            x2 = x2.repeat(repeats, 1)[:N_max]

    # For each dimension, sort both distributions and compute L1 distance
    wasserstein_per_dim = torch.zeros(d, device=x1.device, dtype=x1.dtype)

    for j in range(d):
        x1_sorted = torch.sort(x1[:, j])[0]
        x2_sorted = torch.sort(x2[:, j])[0]
        wasserstein_per_dim[j] = torch.mean(torch.abs(x1_sorted - x2_sorted))

    # Average over dimensions
    return torch.mean(wasserstein_per_dim)


def compute_cross_distance_mean(x1: Tensor, x2: Tensor) -> Tensor:
    """
    Compute mean pairwise distance between two point clouds.

    D_cross = (1/(N1*N2)) Σ_i Σ_j ||x1_i - x2_j||

    Args:
        x1: Points from cloud 1 [N1, d]
        x2: Points from cloud 2 [N2, d]

    Returns:
        Mean cross-distance (scalar)
    """
    # Pairwise distances [N1, N2]
    diff = x1.unsqueeze(1) - x2.unsqueeze(0)  # [N1, N2, d]
    distances = torch.norm(diff, dim=2)  # [N1, N2]

    return torch.mean(distances)


def compute_lyapunov(
    state1: SwarmState,
    state2: SwarmState,
    weight_var: float = 1.0,
    weight_cross: float = 1.0,
    weight_wasserstein: float = 1.0,
    weight_com: float = 1.0,
) -> LyapunovTerms:
    """
    Compute all Lyapunov function terms for two swarm states.

    The total Lyapunov function is a weighted sum:

    V(t) = w_var * (var_x_1 + var_x_2 + var_v_1 + var_v_2)
         + w_cross * (D_cross_x + D_cross_v)
         + w_wasserstein * (W_x + W_v)
         + w_com * (||Δμ_x|| + ||Δμ_v||)

    Args:
        state1: First swarm state
        state2: Second swarm state
        weight_var: Weight for variance terms
        weight_cross: Weight for cross-distance terms
        weight_wasserstein: Weight for Wasserstein terms
        weight_com: Weight for center-of-mass distance terms

    Returns:
        LyapunovTerms object containing all terms
    """
    # Within-swarm variance terms
    var_x_1 = VectorizedOps.variance_position(state1)
    var_x_2 = VectorizedOps.variance_position(state2)
    var_v_1 = VectorizedOps.variance_velocity(state1)
    var_v_2 = VectorizedOps.variance_velocity(state2)

    # Cross-swarm distance terms
    mean_cross_distance_x = compute_cross_distance_mean(state1.x, state2.x)
    mean_cross_distance_v = compute_cross_distance_mean(state1.v, state2.v)

    # Wasserstein distance approximations
    wasserstein_x = compute_wasserstein_1d_approx(state1.x, state2.x)
    wasserstein_v = compute_wasserstein_1d_approx(state1.v, state2.v)

    # Center of mass distances
    μ_x_1 = torch.mean(state1.x, dim=0)  # [d]
    μ_x_2 = torch.mean(state2.x, dim=0)  # [d]
    μ_v_1 = torch.mean(state1.v, dim=0)  # [d]
    μ_v_2 = torch.mean(state2.v, dim=0)  # [d]

    com_distance_x = torch.norm(μ_x_1 - μ_x_2)
    com_distance_v = torch.norm(μ_v_1 - μ_v_2)

    # Total Lyapunov function
    total = (
        weight_var * (var_x_1 + var_x_2 + var_v_1 + var_v_2)
        + weight_cross * (mean_cross_distance_x + mean_cross_distance_v)
        + weight_wasserstein * (wasserstein_x + wasserstein_v)
        + weight_com * (com_distance_x + com_distance_v)
    )

    return LyapunovTerms(
        var_x_1=var_x_1,
        var_x_2=var_x_2,
        var_v_1=var_v_1,
        var_v_2=var_v_2,
        mean_cross_distance_x=mean_cross_distance_x,
        mean_cross_distance_v=mean_cross_distance_v,
        wasserstein_x=wasserstein_x,
        wasserstein_v=wasserstein_v,
        com_distance_x=com_distance_x,
        com_distance_v=com_distance_v,
        total=total,
    )


def compute_lyapunov_from_gas(
    gas1: EuclideanGas,
    gas2: EuclideanGas,
    state1: SwarmState | None = None,
    state2: SwarmState | None = None,
    **kwargs,
) -> dict[str, Tensor]:
    """
    Convenience function to compute Lyapunov terms from two EuclideanGas instances.

    Args:
        gas1: First Euclidean Gas instance
        gas2: Second Euclidean Gas instance
        state1: Optional state for gas1 (if None, initializes new state)
        state2: Optional state for gas2 (if None, initializes new state)
        **kwargs: Additional arguments passed to compute_lyapunov (weights, etc.)

    Returns:
        Dictionary of tensors containing all Lyapunov terms

    Example:
        >>> from fragile.euclidean_gas import EuclideanGas, EuclideanGasParams, ...
        >>> from fragile.lyapunov import compute_lyapunov_from_gas
        >>>
        >>> # Create two gas instances with different parameters
        >>> params1 = EuclideanGasParams(...)
        >>> params2 = EuclideanGasParams(...)
        >>> gas1 = EuclideanGas(params1)
        >>> gas2 = EuclideanGas(params2)
        >>>
        >>> # Run both for some steps
        >>> state1 = gas1.initialize_state()
        >>> state2 = gas2.initialize_state()
        >>> for _ in range(100):
        >>>     _, state1 = gas1.step(state1)
        >>>     _, state2 = gas2.step(state2)
        >>>
        >>> # Compare using Lyapunov function
        >>> lyapunov = compute_lyapunov_from_gas(gas1, gas2, state1, state2)
        >>> print(f"Total Lyapunov: {lyapunov['total'].item():.6f}")
    """
    # Initialize states if not provided
    if state1 is None:
        state1 = gas1.initialize_state()
    if state2 is None:
        state2 = gas2.initialize_state()

    # Compute Lyapunov terms
    terms = compute_lyapunov(state1, state2, **kwargs)

    return terms.to_dict()
