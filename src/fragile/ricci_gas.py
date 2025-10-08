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

from fragile.fragile_typing import TensorType


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

    x: TensorType["N", "d"] = Field(..., description="Positions")
    v: TensorType["N", "d"] = Field(..., description="Velocities")
    s: TensorType["N"] = Field(..., description="Status (0=dead, 1=alive)")

    # Cached geometric quantities (optional, for efficiency)
    R: TensorType["N"] | None = Field(default=None, description="Ricci curvature per walker")
    H: TensorType["N", "d", "d"] | None = Field(default=None, description="Hessian per walker")

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
    normalizer = (2 * torch.pi * bandwidth**2) ** (-d / 2)
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
    x_eval = x_eval.requires_grad_(True)

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


def compute_ricci_proxy_3d(H: Tensor) -> Tensor:
    """Compute 3D Ricci curvature proxy from Hessian.

    R = tr(H) - λ_min(H)

    Args:
        H: Hessian tensor [N, 3, 3]

    Returns:
        R: Ricci scalar proxy [N]
    """
    assert H.shape[-2:] == (3, 3), "Only 3D is supported"

    # Trace (average curvature)
    trace = torch.diagonal(H, dim1=-2, dim2=-1).sum(dim=-1)

    # Minimum eigenvalue (most negative expansion direction)
    eigenvalues = torch.linalg.eigvalsh(H)  # [N, 3], sorted ascending
    lambda_min = eigenvalues[..., 0]

    return trace - lambda_min


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

        # Ricci at perturbed positions
        R_plus = compute_ricci_proxy_3d(H_plus)
        R_minus = compute_ricci_proxy_3d(H_minus)

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

    def __init__(self, params: RicciGasParams):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        alive = state.s.bool()

        # Compute Hessian of KDE at each walker position
        H = compute_kde_hessian(
            x,
            x,
            self.params.kde_bandwidth,
            alive,
        )

        # Compute Ricci proxy
        R = compute_ricci_proxy_3d(H)

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
        alive = state.s.bool()

        if self.params.force_mode == "none":
            return torch.zeros_like(x)

        # Compute gradient of Ricci
        grad_R = compute_ricci_gradient(
            x,
            state.H,
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


# Example toy problem: Double-well potential
def double_well_3d(x: Tensor) -> Tensor:
    """3D double-well potential with minima at (±1, 0, 0).

    V(x) = (x² - 1)² + y² + z²

    Args:
        x: Positions [N, 3]

    Returns:
        V: Potential values [N]
    """
    return (x[:, 0]**2 - 1) ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2


def rastrigin_3d(x: Tensor) -> Tensor:
    """3D Rastrigin function (many local minima).

    Args:
        x: Positions [N, 3]

    Returns:
        V: Potential values [N]
    """
    A = 10
    d = 3
    return A * d + (x**2 - A * torch.cos(2 * torch.pi * x)).sum(dim=-1)
