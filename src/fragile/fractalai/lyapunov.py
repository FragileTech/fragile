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

Usage Examples:

    # Single-state analysis (existing API)
    from fragile.fractalai.core import EuclideanGas
    from fragile.lyapunov import compute_total_lyapunov

    gas = EuclideanGas(N=100, d=2, ...)
    state = gas.initialize_state()
    V = compute_total_lyapunov(state)
    print(f"Lyapunov: {V.item():.6f}")

    # Time series analysis from RunHistory (new API)
    from fragile.lyapunov import compute_lyapunov_trajectory

    history = gas.run(n_steps=1000, record_every=10)
    traj = compute_lyapunov_trajectory(history, stage="final")

    # Visualize exponential decay
    import holoviews as hv
    hv.extension('bokeh')
    hv.Curve((traj["time"], traj["V_total"]), label="V_total").opts(logy=True)

    # Component analysis
    from fragile.lyapunov import compute_lyapunov_components_trajectory

    comp = compute_lyapunov_components_trajectory(history)
    print(f"Position dominance: {comp['position_ratio'][-1]:.2%}")
    print(f"Velocity dominance: {comp['velocity_ratio'][-1]:.2%}")

References:
- docs/source/1_euclidean_gas/03_cloning.md § 3.2 (def-full-synergistic-lyapunov-function)
- docs/source/1_euclidean_gas/03_cloning.md lines 866-909 (Three Variance Notations)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from fragile.fractalai.core.euclidean_gas import SwarmState


if TYPE_CHECKING:
    from fragile.fractalai.core.history import RunHistory


def compute_internal_variance_position(
    state: SwarmState, alive_mask: Tensor | None = None
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
    return squared_deviations.sum() / N


def compute_internal_variance_velocity(
    state: SwarmState, alive_mask: Tensor | None = None
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
    return squared_deviations.sum() / N


def compute_total_lyapunov(state: SwarmState, alive_mask: Tensor | None = None) -> Tensor:
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


# ============================================================================
# RunHistory-based Analysis Functions
# ============================================================================


def _extract_state_from_history(
    history: RunHistory, time_idx: int, stage: str = "final"
) -> SwarmState:
    """Extract SwarmState from RunHistory at a specific time index and stage.

    This is a helper function for computing Lyapunov trajectories from recorded history.

    Args:
        history: RunHistory object containing execution trace
        time_idx: Index in recorded arrays (0 to n_recorded-1)
        stage: Which state to extract ("before_clone", "after_clone", "final")

    Returns:
        SwarmState at the specified time and stage

    Raises:
        ValueError: If stage is not recognized or time_idx is out of bounds

    Example:
        >>> history = gas.run(n_steps=100, record_every=10)
        >>> state_t5 = _extract_state_from_history(history, 5, "final")
        >>> print(f"State at recorded timestep 5: N={state_t5.N}, d={state_t5.d}")
    """
    # Validate time index
    max_idx = history.n_recorded - 1
    if time_idx < 0 or time_idx > max_idx:
        msg = f"time_idx {time_idx} out of bounds [0, {max_idx}]"
        raise ValueError(msg)

    # Extract state based on stage
    if stage == "before_clone":
        x = history.x_before_clone[time_idx]  # [N, d]
        v = history.v_before_clone[time_idx]  # [N, d]
    elif stage == "after_clone":
        # Note: No "after_clone" state at t=0
        if time_idx == 0:
            msg = "No 'after_clone' state at time_idx=0 (initial state)"
            raise ValueError(msg)
        x = history.x_after_clone[time_idx - 1]  # Offset by 1
        v = history.v_after_clone[time_idx - 1]
    elif stage == "final":
        x = history.x_final[time_idx]  # [N, d]
        v = history.v_final[time_idx]  # [N, d]
    else:
        msg = f"Unknown stage: {stage}. Must be 'before_clone', 'after_clone', or 'final'"
        raise ValueError(msg)

    return SwarmState(x, v)


def compute_lyapunov_trajectory(
    history: RunHistory, stage: str = "final", use_alive_mask: bool = True
) -> dict[str, Tensor]:
    """Compute Lyapunov function trajectory from RunHistory.

    Computes V_total(S) = V_Var,x(S) + V_Var,v(S) at each recorded timestep,
    enabling convergence analysis over the full execution trace.

    Args:
        history: RunHistory object containing execution trace
        stage: Which state to analyze ("before_clone", "after_clone", "final")
        use_alive_mask: If True, compute variance only over alive walkers

    Returns:
        Dictionary containing:
            - "time": Absolute step numbers [n_recorded] or [n_recorded-1] for after_clone
            - "V_total": Total Lyapunov values [n_recorded] or [n_recorded-1]
            - "V_var_x": Position variance component [n_recorded] or [n_recorded-1]
            - "V_var_v": Velocity variance component [n_recorded] or [n_recorded-1]

    Example:
        >>> history = gas.run(n_steps=1000, record_every=10)
        >>> traj = compute_lyapunov_trajectory(history, stage="final")
        >>> print(f"Lyapunov at t=0: {traj['V_total'][0]:.6f}")
        >>> print(f"Lyapunov at final: {traj['V_total'][-1]:.6f}")
        >>>
        >>> # Plot exponential decay
        >>> import matplotlib.pyplot as plt
        >>> plt.semilogy(traj["time"], traj["V_total"])
        >>> plt.xlabel("Time step")
        >>> plt.ylabel("V_total (log scale)")

    Notes:
        - For stage="after_clone", time array starts at step record_every (no t=0 state)
        - Alive mask filtering is only applied if use_alive_mask=True
        - All Lyapunov values are N-normalized (framework convention)

    Reference:
        - docs/source/1_euclidean_gas/03_cloning.md § 3.2 (Lyapunov function definition)
    """
    # Determine number of timesteps to process
    if stage == "after_clone":
        n_timesteps = history.n_recorded - 1  # No after_clone at t=0
        time_offset = 1
    else:
        n_timesteps = history.n_recorded
        time_offset = 0

    # Preallocate output arrays
    V_total = torch.zeros(n_timesteps, dtype=history.x_final.dtype, device=history.x_final.device)
    V_var_x = torch.zeros(n_timesteps, dtype=history.x_final.dtype, device=history.x_final.device)
    V_var_v = torch.zeros(n_timesteps, dtype=history.x_final.dtype, device=history.x_final.device)

    # Compute Lyapunov at each timestep
    for t_idx in range(n_timesteps):
        # Extract state
        state = _extract_state_from_history(history, t_idx + time_offset, stage)

        # Get alive mask if requested
        alive_mask = None
        if use_alive_mask and stage != "before_clone":
            # alive_mask has shape [n_recorded-1, N], indexed by step after t=0
            if t_idx + time_offset > 0:
                alive_mask = history.alive_mask[t_idx + time_offset - 1]

        # Compute Lyapunov components
        V_var_x[t_idx] = compute_internal_variance_position(state, alive_mask)
        V_var_v[t_idx] = compute_internal_variance_velocity(state, alive_mask)
        V_total[t_idx] = V_var_x[t_idx] + V_var_v[t_idx]

    # Create time array
    time = (
        torch.arange(time_offset, time_offset + n_timesteps, dtype=torch.long)
        * history.record_every
    )

    return {
        "time": time,
        "V_total": V_total,
        "V_var_x": V_var_x,
        "V_var_v": V_var_v,
    }


def compute_lyapunov_components_trajectory(
    history: RunHistory, stage: str = "final", use_alive_mask: bool = True
) -> dict[str, Tensor]:
    """Compute detailed Lyapunov components over time with separate position/velocity.

    This is an extended version of compute_lyapunov_trajectory() that provides
    both the total Lyapunov and its individual variance components, useful for
    understanding the relative contributions of positional vs. velocity spreading.

    Args:
        history: RunHistory object containing execution trace
        stage: Which state to analyze ("before_clone", "after_clone", "final")
        use_alive_mask: If True, compute variance only over alive walkers

    Returns:
        Dictionary containing:
            - "time": Absolute step numbers [n_timesteps]
            - "V_total": Total Lyapunov V_Var,x + V_Var,v [n_timesteps]
            - "V_var_x": Position internal variance [n_timesteps]
            - "V_var_v": Velocity internal variance [n_timesteps]
            - "position_ratio": V_var_x / V_total [n_timesteps]
            - "velocity_ratio": V_var_v / V_total [n_timesteps]

    Example:
        >>> history = gas.run(n_steps=1000, record_every=10)
        >>> comp = compute_lyapunov_components_trajectory(history)
        >>>
        >>> # Analyze component evolution
        >>> print(f"Initial position dominance: {comp['position_ratio'][0]:.2%}")
        >>> print(f"Final position dominance: {comp['position_ratio'][-1]:.2%}")
        >>>
        >>> # Plot component breakdown
        >>> import holoviews as hv
        >>> hv.extension("bokeh")
        >>> position_curve = hv.Curve((comp["time"], comp["V_var_x"]), label="Position")
        >>> velocity_curve = hv.Curve((comp["time"], comp["V_var_v"]), label="Velocity")
        >>> (position_curve * velocity_curve).opts(logy=True)

    Notes:
        - Ratios are computed as V_var_x / (V_var_x + V_var_v + ε) to avoid division by zero
        - ε = 1e-10 for numerical stability
        - Both ratios sum to approximately 1.0 at each timestep
        - Useful for understanding which variance component dominates dynamics

    Reference:
        - docs/source/1_euclidean_gas/03_cloning.md § 3.2 (Variance components)
    """
    # Get base trajectory
    traj = compute_lyapunov_trajectory(history, stage, use_alive_mask)

    # Compute ratios (with numerical stability)
    eps = 1e-10
    total_safe = traj["V_total"] + eps

    position_ratio = traj["V_var_x"] / total_safe
    velocity_ratio = traj["V_var_v"] / total_safe

    return {
        "time": traj["time"],
        "V_total": traj["V_total"],
        "V_var_x": traj["V_var_x"],
        "V_var_v": traj["V_var_v"],
        "position_ratio": position_ratio,
        "velocity_ratio": velocity_ratio,
    }
