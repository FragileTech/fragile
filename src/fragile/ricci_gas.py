"""
Ricci Fragile Gas: Geometry-Driven Swarm Optimization

This module implements the Ricci Fragile Gas as defined in docs/source/12_fractal_gas.md,
a novel variant where exploration is driven by the Ricci curvature of the emergent
manifold created by the swarm's distribution.

Key concepts:
- Push-pull dynamics: Force aggregates (pull), cloning disperses (push)
- Phase transitions at critical feedback strength α_c
- Multi-layered stability via geometry and diversity
- 3D-optimized implementation using eigenvalue-based Ricci proxy
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor


class RicciGasParams(BaseModel):
    """Parameters for the Ricci Fragile Gas.

    Physics analogy:
    - epsilon_R: Gravitational coupling constant (aggregation strength)
    - kde_bandwidth: Planck length (minimum geometric scale)
    - R_crit: Event horizon radius (singularity threshold)
    - epsilon_Sigma: Quantum fluctuation scale
    """

    # Core Ricci parameters
    epsilon_R: float = Field(default=0.1, ge=0.0, description="Ricci force strength (α)")
    kde_bandwidth: float = Field(default=0.5, gt=0.0, description="KDE smoothing length (ℓ)")
    epsilon_Ric: float = Field(default=0.01, gt=0.0, description="Ricci reward regularization")
    R_crit: float | None = Field(default=None, description="Critical curvature for status killing")

    # Variant selection
    force_mode: Literal["pull", "push", "none"] = Field(
        default="pull",
        description="Force direction: 'pull' (+∇R), 'push' (-∇R), 'none' (0)",
    )
    reward_mode: Literal["inverse", "negative", "none"] = Field(
        default="inverse",
        description="Reward type: 'inverse' (1/R), 'negative' (max(0,-R)), 'none' (0)",
    )

    # Cloning parameters
    epsilon_clone: float = Field(default=0.3, gt=0.0, description="Cloning interaction range (ε_c)")
    sigma_clone: float = Field(default=0.1, gt=0.0, description="Positional jitter scale (σ_c)")

    # Boundary parameters
    x_min: float | None = Field(default=None, description="Lower bound for all dimensions")
    x_max: float | None = Field(default=None, description="Upper bound for all dimensions")

    # Adaptive Gas parameters (inherited)
    epsilon_Sigma: float = Field(default=0.01, gt=0.0, description="Metric regularization")

    # Numerical stability
    gradient_clip: float | None = Field(default=10.0, description="Max gradient norm")
    use_tree_kde: bool = Field(default=False, description="Use tree-based KDE (faster)")
    truncation_radius: float | None = Field(default=None, description="KDE truncation (in units of ℓ)")

    class Config:
        arbitrary_types_allowed = True


class SwarmState(BaseModel):
    """Swarm state for Ricci Gas."""

    x: Tensor = Field(..., description="Positions [N, d]")
    v: Tensor = Field(..., description="Velocities [N, d]")
    s: Tensor = Field(..., description="Status (0=dead, 1=alive) [N]")

    # Cached geometric quantities (optional, for efficiency)
    R: Tensor | None = Field(default=None, description="Ricci curvature per walker [N]")
    H: Tensor | None = Field(default=None, description="Hessian per walker [N, d, d]")

    class Config:
        arbitrary_types_allowed = True


def gaussian_kernel(x: Tensor, bandwidth: float = 1.0) -> Tensor:
    """Gaussian KDE kernel with given bandwidth.

    Args:
        x: [N, d] positions or [N, N, d] pairwise differences
        bandwidth: Smoothing length ℓ

    Returns:
        [N] or [N, N] kernel weights
    """
    d = x.shape[-1]
    norm_sq = (x / bandwidth).pow(2).sum(dim=-1)
    # Use torch.tensor to ensure pi is on the correct device
    pi = torch.tensor(3.14159265358979323846, device=x.device, dtype=x.dtype)
    normalizer = (2 * pi * bandwidth**2) ** (-d / 2)
    return normalizer * torch.exp(-0.5 * norm_sq)


def compute_kde_density(
    x: Tensor,  # [N, d]
    x_eval: Tensor,  # [M, d]
    bandwidth: float,
    alive_mask: Tensor | None = None,  # [N]
    truncation: float | None = None,
) -> Tensor:
    """Compute KDE density at evaluation points.

    Args:
        x: Walker positions [N, d]
        x_eval: Evaluation points [M, d]
        bandwidth: KDE bandwidth ℓ
        alive_mask: Only include alive walkers [N]
        truncation: Ignore walkers beyond this distance (in units of ℓ)

    Returns:
        rho: Density at evaluation points [M]
    """
    # Pairwise differences: [M, N, d]
    diff = x_eval.unsqueeze(1) - x.unsqueeze(0)

    # Compute kernel weights
    weights = gaussian_kernel(diff, bandwidth)  # [M, N]

    # Apply truncation if requested
    if truncation is not None:
        dist = diff.norm(dim=-1)
        weights = weights * (dist < truncation * bandwidth).float()

    # Mask dead walkers
    if alive_mask is not None:
        weights = weights * alive_mask.unsqueeze(0)

    # Normalize
    N_alive = alive_mask.sum() if alive_mask is not None else x.shape[0]
    rho = weights.sum(dim=1) / N_alive

    return rho


def compute_kde_hessian(
    x: Tensor,  # [N, d]
    x_eval: Tensor,  # [M, d]
    bandwidth: float,
    alive_mask: Tensor | None = None,
) -> Tensor:
    """Compute Hessian of KDE density via automatic differentiation.

    Args:
        x: Walker positions [N, d]
        x_eval: Evaluation points [M, d] (requires grad)
        bandwidth: KDE bandwidth
        alive_mask: Alive walkers mask [N]

    Returns:
        H: Hessian tensor [M, d, d]
    """
    # Clone and detach to avoid issues with existing computational graph
    x_eval = x_eval.clone().detach().requires_grad_(True)

    # Compute density
    rho = compute_kde_density(x, x_eval, bandwidth, alive_mask)

    # First derivatives (gradient)
    grad_rho = torch.autograd.grad(
        rho.sum(),
        x_eval,
        create_graph=True,
        retain_graph=True,
    )[0]  # [M, d]

    # Second derivatives (Hessian)
    M, d = x_eval.shape
    H = torch.zeros(M, d, d, device=x.device, dtype=x.dtype)

    for i in range(d):
        H[:, i, :] = torch.autograd.grad(
            grad_rho[:, i].sum(),
            x_eval,
            retain_graph=(i < d - 1),
        )[0]

    return H


def compute_ricci_proxy(H: Tensor) -> Tensor:
    """Compute Ricci curvature proxy from Hessian (works in any dimension).

    R = tr(H) - λ_min(H)

    This proxy captures the essence of Ricci curvature:
    - tr(H): Average curvature (sum of principal curvatures)
    - λ_min(H): Most negative expansion direction
    - High R: Strong focusing in all directions (high density region)
    - Low R: Expansion dominates (saddle point or low density)

    Args:
        H: Hessian tensor [N, d, d] for any d >= 1

    Returns:
        R: Ricci scalar proxy [N]
    """
    # Trace (average curvature)
    trace = torch.diagonal(H, dim1=-2, dim2=-1).sum(dim=-1)

    # Minimum eigenvalue (most negative expansion direction)
    eigenvalues = torch.linalg.eigvalsh(H)  # [N, d], sorted ascending
    lambda_min = eigenvalues[..., 0]

    return trace - lambda_min


def compute_ricci_proxy_3d(H: Tensor) -> Tensor:
    """Compute 3D Ricci curvature proxy from Hessian.

    DEPRECATED: Use compute_ricci_proxy() instead (works in any dimension).

    Args:
        H: Hessian tensor [N, 3, 3]

    Returns:
        R: Ricci scalar proxy [N]
    """
    return compute_ricci_proxy(H)


def compute_ricci_gradient(
    x: Tensor,  # [N, d]
    H: Tensor,  # [N, d, d]
    bandwidth: float,
    alive_mask: Tensor,
) -> Tensor:
    """Compute gradient of Ricci proxy via finite differences.

    Args:
        x: Walker positions [N, d]
        H: Hessians at walker positions [N, d, d]
        bandwidth: KDE bandwidth (used to set step size)
        alive_mask: Alive walkers [N]

    Returns:
        grad_R: Gradient of Ricci at each walker [N, d]
    """
    N, d = x.shape
    grad_R = torch.zeros_like(x)

    # Finite difference step size (fraction of bandwidth)
    eps = 0.01 * bandwidth

    for i in range(d):
        # Perturb positions
        x_plus = x.clone()
        x_plus[:, i] += eps

        x_minus = x.clone()
        x_minus[:, i] -= eps

        # Compute Hessians at perturbed positions
        H_plus = compute_kde_hessian(x, x_plus, bandwidth, alive_mask)
        H_minus = compute_kde_hessian(x, x_minus, bandwidth, alive_mask)

        # Ricci at perturbed positions (dimension-agnostic)
        R_plus = compute_ricci_proxy(H_plus)
        R_minus = compute_ricci_proxy(H_minus)

        # Central difference
        grad_R[:, i] = (R_plus - R_minus) / (2 * eps)

    return grad_R


class RicciGas:
    """Ricci Fragile Gas implementation for 3D physics.

    Implements the push-pull architecture:
    - Force: F = ε_R * ∇R (aggregation toward high curvature)
    - Reward: r ∝ 1/R (dispersion via cloning)

    Usage:
        params = RicciGasParams(epsilon_R=0.5, kde_bandwidth=0.3)
        gas = RicciGas(params)

        # Single step
        state = SwarmState(x=..., v=..., s=...)
        state_new = gas.step(state, dt=0.1)

        # Compute geometry
        R, H = gas.compute_curvature(state)
    """

    def __init__(self, params: RicciGasParams, device: torch.device | str | None = None):
        self.params = params
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

    def compute_curvature(
        self,
        state: SwarmState,
        cache: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Compute Ricci curvature and Hessian for all walkers.

        Args:
            state: Current swarm state
            cache: If True, store R and H in state

        Returns:
            R: Ricci proxy [N]
            H: Hessian [N, d, d]
        """
        x = state.x.to(self.device)
        alive = state.s.bool().to(self.device)

        # Compute Hessian of KDE at each walker position
        H = compute_kde_hessian(
            x,
            x,
            self.params.kde_bandwidth,
            alive,
        )

        # Compute Ricci proxy (dimension-agnostic)
        R = compute_ricci_proxy(H)

        if cache:
            state.R = R
            state.H = H

        return R, H

    def compute_reward(self, state: SwarmState) -> Tensor:
        """Compute Ricci-based reward.

        Args:
            state: Swarm state (must have R computed)

        Returns:
            reward: [N] reward values
        """
        if state.R is None:
            R, _ = self.compute_curvature(state, cache=True)
        else:
            R = state.R

        if self.params.reward_mode == "inverse":
            # r = 1 / (R + ε)
            reward = 1.0 / (R + self.params.epsilon_Ric)
        elif self.params.reward_mode == "negative":
            # r = max(0, -R)
            reward = torch.clamp(-R, min=0.0)
        elif self.params.reward_mode == "none":
            # Use standard reward (would come from environment)
            reward = torch.zeros_like(R)
        else:
            raise ValueError(f"Unknown reward_mode: {self.params.reward_mode}")

        return reward

    def compute_force(self, state: SwarmState) -> Tensor:
        """Compute Ricci-driven force.

        Args:
            state: Swarm state (must have R, H computed)

        Returns:
            force: [N, d] force vectors
        """
        if state.R is None or state.H is None:
            self.compute_curvature(state, cache=True)

        x = state.x.to(self.device)
        alive = state.s.bool().to(self.device)

        if self.params.force_mode == "none":
            return torch.zeros_like(x)

        # Compute gradient of Ricci
        grad_R = compute_ricci_gradient(
            x,
            state.H.to(self.device),
            self.params.kde_bandwidth,
            alive,
        )

        # Clip gradient for numerical stability
        if self.params.gradient_clip is not None:
            grad_norm = grad_R.norm(dim=-1, keepdim=True)
            grad_R = grad_R * torch.clamp(
                grad_norm / self.params.gradient_clip,
                max=1.0,
            )

        # Apply force direction
        if self.params.force_mode == "pull":
            # F = +ε_R ∇R (toward high curvature)
            force = self.params.epsilon_R * grad_R
        elif self.params.force_mode == "push":
            # F = -ε_R ∇R (toward low curvature)
            force = -self.params.epsilon_R * grad_R
        else:
            raise ValueError(f"Unknown force_mode: {self.params.force_mode}")

        # Zero force for dead walkers
        force = force * alive.unsqueeze(-1).float()

        return force

    def apply_cloning(self, state: SwarmState) -> SwarmState:
        """Apply distance-based cloning operator.

        Companions are selected based on spatial proximity using Gaussian weights:
        - w_ij = exp(-||x_i - x_j||² / (2 * epsilon_clone²))
        - P(i chooses j) = w_ij / Σ_k w_ik

        When epsilon_clone is large, selection becomes nearly uniform.
        When epsilon_clone is small, walkers only clone from nearby companions.

        Args:
            state: Swarm state

        Returns:
            Updated state after cloning
        """
        N = state.x.shape[0]
        d = state.x.shape[1]

        # Only alive walkers can be chosen as companions
        alive = state.s.bool()

        # Compute pairwise distances: [N, N]
        diff = state.x.unsqueeze(0) - state.x.unsqueeze(1)  # [N, N, d]
        dist_sq = (diff**2).sum(dim=-1)  # [N, N]

        # Mask self-selection by setting diagonal to inf
        dist_sq_masked = dist_sq.clone()
        dist_sq_masked.fill_diagonal_(float('inf'))

        # Mask dead walkers: they can't be chosen as companions
        dist_sq_masked[:, ~alive] = float('inf')

        # Compute Gaussian spatial weights: w_ij = exp(-d²_ij / (2ε²))
        epsilon = self.params.epsilon_clone
        exponent = -dist_sq_masked / (2.0 * epsilon**2)  # [N, N]

        # Use log-sum-exp trick for numerical stability
        max_exp = torch.max(exponent, dim=1, keepdim=True)[0]  # [N, 1]
        max_exp = torch.where(torch.isinf(max_exp), torch.zeros_like(max_exp), max_exp)

        weights = torch.exp(exponent - max_exp)  # [N, N]
        weights_sum = weights.sum(dim=1, keepdim=True)  # [N, 1]

        # Handle edge case: if a walker has no valid companions, clone from self
        no_companions = weights_sum.squeeze() == 0
        if no_companions.any():
            # For walkers with no companions, set self-cloning weight
            weights[no_companions, :] = 0.0
            for idx in torch.where(no_companions)[0]:
                weights[idx, idx] = 1.0
            weights_sum = weights.sum(dim=1, keepdim=True)

        probs = weights / weights_sum  # [N, N]

        # Sample companions for all walkers
        companions = torch.multinomial(probs, num_samples=1).squeeze(1)  # [N]

        # Clone positions with jitter
        sigma = self.params.sigma_clone
        zeta_x = torch.randn(N, d, device=self.device, dtype=state.x.dtype)
        x_companion = state.x[companions]  # [N, d]
        x_new = x_companion + sigma * zeta_x

        # Clone velocities (simple copy)
        v_new = state.v[companions]

        # Revive all walkers: cloning brings dead walkers back to life
        # This is the key mechanism for maintaining population
        s_new = torch.ones(N, device=self.device, dtype=state.s.dtype)

        # Update state
        state_new = SwarmState(x=x_new, v=v_new, s=s_new)

        return state_new

    def apply_boundary_enforcement(self, state: SwarmState) -> SwarmState:
        """Kill walkers that leave the boundary region.

        Args:
            state: Swarm state

        Returns:
            state: Updated state with modified status
        """
        if self.params.x_min is None and self.params.x_max is None:
            return state

        # Check which walkers are out of bounds
        out_of_bounds = torch.zeros(state.x.shape[0], dtype=torch.bool, device=state.x.device)

        if self.params.x_min is not None:
            out_of_bounds |= (state.x < self.params.x_min).any(dim=-1)

        if self.params.x_max is not None:
            out_of_bounds |= (state.x > self.params.x_max).any(dim=-1)

        # Kill out-of-bounds walkers
        state.s = state.s * (~out_of_bounds).float()

        return state

    def apply_singularity_regulation(self, state: SwarmState) -> SwarmState:
        """Kill walkers that enter high-curvature regions.

        Implements "bouncing singularity" mechanism.

        Args:
            state: Swarm state with R computed

        Returns:
            state: Updated state with modified status
        """
        if self.params.R_crit is None:
            return state

        if state.R is None:
            self.compute_curvature(state, cache=True)

        # Kill walkers with R > R_crit
        high_curv_mask = state.R > self.params.R_crit
        state.s = state.s * (~high_curv_mask).float()

        return state

    def step(
        self,
        state: SwarmState,
        dt: float = 0.1,
        gamma: float = 0.9,
        noise_std: float = 0.05,
        do_clone: bool = True,
    ) -> SwarmState:
        """Perform one step of Ricci Gas dynamics.

        Sequence:
        1. Cloning (if enabled): Distance-based companion selection
        2. Curvature computation: R, H from KDE
        3. Force computation: F = ε_R * ∇R
        4. Langevin update: v' = γv + (1-γ)F + noise, x' = x + v'*dt

        Args:
            state: Current swarm state
            dt: Time step size
            gamma: Friction coefficient (velocity damping)
            noise_std: Standard deviation of Langevin noise
            do_clone: Whether to apply cloning operator

        Returns:
            Updated swarm state
        """
        # Step 1: Cloning (dispersive push via spatial selection)
        if do_clone:
            state = self.apply_cloning(state)

        # Step 2: Compute curvature
        R, H = self.compute_curvature(state, cache=True)

        # Step 3: Compute force (aggregative pull toward high curvature)
        force = self.compute_force(state)

        # Step 4: Langevin dynamics
        noise = torch.randn_like(state.v, device=self.device) * noise_std
        v_new = gamma * state.v + (1 - gamma) * force + noise
        x_new = state.x + v_new * dt

        # Update state
        state_new = SwarmState(x=x_new, v=v_new, s=state.s.clone(), R=R, H=H)

        # Step 5: Apply boundary enforcement (kill out-of-bounds walkers)
        state_new = self.apply_boundary_enforcement(state_new)

        # Step 6: Apply singularity regulation (kill high-curvature walkers)
        state_new = self.apply_singularity_regulation(state_new)

        return state_new

    def visualize_curvature(
        self,
        state: SwarmState,
        grid_resolution: int = 50,
        zlevel: float = 0.0,
    ) -> dict:
        """Create curvature heatmap visualization data.

        Args:
            state: Swarm state
            grid_resolution: Number of grid points per dimension
            zlevel: Z-coordinate for 2D slice

        Returns:
            dict with 'x_grid', 'y_grid', 'R_grid', 'walkers'
        """
        x = state.x.detach().cpu()

        # Create 2D grid (XY plane at zlevel)
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

        x_grid = torch.linspace(x_min, x_max, grid_resolution)
        y_grid = torch.linspace(y_min, y_max, grid_resolution)
        xx, yy = torch.meshgrid(x_grid, y_grid, indexing="ij")

        # Evaluation points (include z=zlevel)
        x_eval = torch.stack([
            xx.flatten(),
            yy.flatten(),
            torch.full_like(xx.flatten(), zlevel),
        ], dim=-1)

        # Compute Hessian and Ricci on grid
        H_grid = compute_kde_hessian(
            state.x,
            x_eval,
            self.params.kde_bandwidth,
            state.s.bool(),
        )
        R_grid = compute_ricci_proxy_3d(H_grid)
        R_grid = R_grid.reshape(grid_resolution, grid_resolution)

        return {
            "x_grid": x_grid.numpy(),
            "y_grid": y_grid.numpy(),
            "R_grid": R_grid.detach().cpu().numpy(),
            "walkers": x.numpy(),
            "alive": state.s.bool().cpu().numpy(),
        }


def create_ricci_gas_variants() -> dict[str, RicciGasParams]:
    """Create parameter sets for ablation study variants.

    Returns dictionary with keys:
    - 'ricci': Push-pull (force=pull, reward=inverse) [Variant A]
    - 'aligned': Both seek flat (force=push, reward=inverse) [Variant B]
    - 'force_only': Pure aggregation (force=pull, reward=none) [Variant C]
    - 'reward_only': Pure dispersion (force=none, reward=inverse) [Variant D]
    """
    base_params = {
        "epsilon_R": 0.5,
        "kde_bandwidth": 0.3,
        "epsilon_Ric": 0.01,
        "epsilon_Sigma": 0.01,
    }

    return {
        "ricci": RicciGasParams(
            **base_params,
            force_mode="pull",
            reward_mode="inverse",
        ),
        "aligned": RicciGasParams(
            **base_params,
            force_mode="push",
            reward_mode="inverse",
        ),
        "force_only": RicciGasParams(
            **base_params,
            force_mode="pull",
            reward_mode="none",
        ),
        "reward_only": RicciGasParams(
            **base_params,
            force_mode="none",
            reward_mode="inverse",
        ),
    }


# Example toy problems

def double_well(x: Tensor) -> Tensor:
    """Double-well potential (works in any dimension).

    V(x) = (x₁² - 1)² + Σᵢ₌₂ⁿ xᵢ²

    Minima at (±1, 0, ..., 0)

    Args:
        x: Positions [N, d] for any d >= 1

    Returns:
        V: Potential values [N]
    """
    if x.shape[-1] == 1:
        # 1D case
        return (x[:, 0]**2 - 1) ** 2
    else:
        # Multi-dimensional case
        return (x[:, 0]**2 - 1) ** 2 + (x[:, 1:]**2).sum(dim=-1)


def rastrigin(x: Tensor) -> Tensor:
    """Rastrigin function (works in any dimension, many local minima).

    V(x) = A·d + Σᵢ (xᵢ² - A·cos(2π·xᵢ))

    Args:
        x: Positions [N, d] for any d >= 1

    Returns:
        V: Potential values [N]
    """
    A = 10
    d = x.shape[-1]
    pi = torch.tensor(3.14159265358979323846, device=x.device, dtype=x.dtype)
    return A * d + (x**2 - A * torch.cos(2 * pi * x)).sum(dim=-1)


def sphere(x: Tensor) -> Tensor:
    """Sphere function (works in any dimension).

    V(x) = ||x||²

    Global minimum at origin.

    Args:
        x: Positions [N, d] for any d >= 1

    Returns:
        V: Potential values [N]
    """
    return (x**2).sum(dim=-1)


# Backward compatibility: 3D-specific versions
def double_well_3d(x: Tensor) -> Tensor:
    """3D double-well potential with minima at (±1, 0, 0).

    DEPRECATED: Use double_well() instead (works in any dimension).

    Args:
        x: Positions [N, 3]

    Returns:
        V: Potential values [N]
    """
    return double_well(x)


def rastrigin_3d(x: Tensor) -> Tensor:
    """3D Rastrigin function (many local minima).

    DEPRECATED: Use rastrigin() instead (works in any dimension).

    Args:
        x: Positions [N, 3]

    Returns:
        V: Potential values [N]
    """
    return rastrigin(x)
