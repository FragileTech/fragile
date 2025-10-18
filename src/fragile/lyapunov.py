"""
Lyapunov Functions for Swarm Convergence Analysis

This module implements Lyapunov functions following the framework defined in
docs/source/1_euclidean_gas/03_cloning.md.

Mathematical Framework:
The synergistic Lyapunov function is defined as:

    V_total(S) = V_Var,x(S) + V_Var,v(S)

where:
- V_Var,x: Positional internal variance (N-normalized)
- V_Var,v: Velocity internal variance (N-normalized)

For a single swarm S with N walkers and k_alive alive walkers:

    V_Var,x(S) = (1/N) Σ_{i ∈ A(S)} ||δ_x,i||²
    V_Var,v(S) = (1/N) Σ_{i ∈ A(S)} ||δ_v,i||²

where:
- δ_x,i = x_i - μ_x is deviation from center of mass (position)
- δ_v,i = v_i - μ_v is deviation from center of mass (velocity)
- μ_x = (1/k_alive) Σ_{i ∈ A(S)} x_i
- μ_v = (1/k_alive) Σ_{i ∈ A(S)} v_i
- A(S) is the set of alive walker indices

The N-normalization ensures drift inequalities are N-uniform (independent of swarm size).

References:
- docs/source/1_euclidean_gas/03_cloning.md § 3.2 (def-full-synergistic-lyapunov-function)
- docs/source/1_euclidean_gas/03_cloning.md lines 866-909 (Three Variance Notations)
"""

from __future__ import annotations

import torch
from torch import Tensor

from fragile.euclidean_gas import SwarmState


def compute_internal_variance_position(
    state: SwarmState,
    alive_mask: Tensor | None = None
) -> Tensor:
    """Compute positional internal variance V_Var,x(S) (N-normalized).

    Following the framework definition (03_cloning.md line 877):

        V_Var,x(S) = (1/N) Σ_{i ∈ A(S)} ||δ_x,i||²

    where δ_x,i = x_i - μ_x is the deviation from center of mass.

    Args:
        state: Swarm state
        alive_mask: Optional boolean mask [N] for alive walkers.
                   If None, all walkers are considered alive.

    Returns:
        Scalar tensor: V_Var,x (N-normalized variance)

    Example:
        >>> V_var_x = compute_internal_variance_position(state, alive_mask)
        >>> print(f"Positional variance: {V_var_x.item():.6f}")
    """
    N = state.x.shape[0]

    if alive_mask is None:
        # All walkers alive
        x_alive = state.x
        k_alive = N
    else:
        # Filter to alive walkers
        x_alive = state.x[alive_mask]
        k_alive = alive_mask.sum().item()

    if k_alive == 0:
        return torch.tensor(0.0, device=state.x.device, dtype=state.x.dtype)

    # Center of mass (alive walkers only)
    mu_x = x_alive.mean(dim=0)  # [d]

    # Deviations
    delta_x = x_alive - mu_x  # [k_alive, d]

    # Squared deviations
    squared_deviations = torch.sum(delta_x**2, dim=1)  # [k_alive]

    # N-normalized sum (framework definition)
    V_var_x = squared_deviations.sum() / N

    return V_var_x


def compute_internal_variance_velocity(
    state: SwarmState,
    alive_mask: Tensor | None = None
) -> Tensor:
    """Compute velocity internal variance V_Var,v(S) (N-normalized).

    Following the framework definition (analogous to position):

        V_Var,v(S) = (1/N) Σ_{i ∈ A(S)} ||δ_v,i||²

    where δ_v,i = v_i - μ_v is the deviation from velocity center of mass.

    Args:
        state: Swarm state
        alive_mask: Optional boolean mask [N] for alive walkers.
                   If None, all walkers are considered alive.

    Returns:
        Scalar tensor: V_Var,v (N-normalized variance)

    Example:
        >>> V_var_v = compute_internal_variance_velocity(state, alive_mask)
        >>> print(f"Velocity variance: {V_var_v.item():.6f}")
    """
    N = state.v.shape[0]

    if alive_mask is None:
        # All walkers alive
        v_alive = state.v
        k_alive = N
    else:
        # Filter to alive walkers
        v_alive = state.v[alive_mask]
        k_alive = alive_mask.sum().item()

    if k_alive == 0:
        return torch.tensor(0.0, device=state.v.device, dtype=state.v.dtype)

    # Velocity center of mass (alive walkers only)
    mu_v = v_alive.mean(dim=0)  # [d]

    # Deviations
    delta_v = v_alive - mu_v  # [k_alive, d]

    # Squared deviations
    squared_deviations = torch.sum(delta_v**2, dim=1)  # [k_alive]

    # N-normalized sum (framework definition)
    V_var_v = squared_deviations.sum() / N

    return V_var_v


def compute_total_lyapunov(
    state: SwarmState,
    alive_mask: Tensor | None = None
) -> Tensor:
    """Compute total Lyapunov function V_total(S).

    Following the framework definition:

        V_total(S) = V_Var,x(S) + V_Var,v(S)

    This is the simplified version for single-swarm analysis (no inter-swarm terms).

    Args:
        state: Swarm state
        alive_mask: Optional boolean mask [N] for alive walkers

    Returns:
        Scalar tensor: Total Lyapunov function value

    Example:
        >>> V_total = compute_total_lyapunov(state, alive_mask)
        >>> print(f"Total Lyapunov: {V_total.item():.6f}")
    """
    V_var_x = compute_internal_variance_position(state, alive_mask)
    V_var_v = compute_internal_variance_velocity(state, alive_mask)

    return V_var_x + V_var_v
