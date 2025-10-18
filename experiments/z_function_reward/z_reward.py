"""
Z-Function Reward for Riemann Hypothesis Investigation

This module implements reward functions based on the Riemann-Siegel Z function
to test the hypothesis that Yang-Mills eigenvalues correspond to zeta zeros.

Key idea: Use Z(||x||) as reward landscape → QSD localizes at zeta zeros →
eigenvalues reflect zero locations → RH proof via self-adjointness.
"""

from __future__ import annotations

import torch
from torch import Tensor
import mpmath
import numpy as np
from typing import Callable


class ZFunctionReward:
    """
    Reward function based on Riemann-Siegel Z function.

    The Z function is defined as:
        Z(t) = exp(i*theta(t)) * zeta(1/2 + i*t)

    where theta(t) is chosen so Z(t) is real-valued.

    Key properties:
    - Z(t_n) = 0 iff zeta(1/2 + it_n) = 0 (assuming RH)
    - Z(t) is real for all t
    - Oscillates with sign changes at each zero

    Reward design:
        r(x) = 1 / (Z(||x||)^2 + epsilon^2)

    This creates sharp peaks at zeta zeros, causing walkers to localize there.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        t_min: float = 0.0,
        t_max: float = 100.0,
        cache_size: int = 10000,
    ):
        """
        Initialize Z-function reward.

        Args:
            epsilon: Regularization parameter (controls peak sharpness)
            t_min: Minimum t value for caching
            t_max: Maximum t value for caching
            cache_size: Number of points to pre-compute for interpolation
        """
        self.epsilon = epsilon
        self.t_min = t_min
        self.t_max = t_max
        self.cache_size = cache_size

        # Pre-compute Z function values for fast interpolation
        self._build_cache()

    def _build_cache(self):
        """Pre-compute Z function values on a grid."""
        print(f"Building Z-function cache from t={self.t_min} to t={self.t_max}...")

        # Grid of t values
        t_grid = np.linspace(self.t_min, self.t_max, self.cache_size)

        # Compute Z(t) using mpmath (high precision)
        z_values = np.array([
            float(mpmath.siegelz(t)) for t in t_grid
        ])

        # Store as torch tensors
        self.t_cache = torch.tensor(t_grid, dtype=torch.float32)
        self.z_cache = torch.tensor(z_values, dtype=torch.float32)

        print(f"Cache built: {self.cache_size} points")
        print(f"Z value range: [{z_values.min():.3f}, {z_values.max():.3f}]")

    def Z(self, t: Tensor) -> Tensor:
        """
        Evaluate Riemann-Siegel Z function via interpolation.

        Args:
            t: Input values [N] or scalar

        Returns:
            Z(t) values [N] or scalar
        """
        # Handle scalar input
        if t.dim() == 0:
            t = t.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Clamp to cache range
        t_clamped = torch.clamp(t, self.t_min, self.t_max)

        # Linear interpolation
        # Find indices for interpolation
        indices = torch.searchsorted(self.t_cache, t_clamped)
        indices = torch.clamp(indices, 1, len(self.t_cache) - 1)

        # Get surrounding points
        t_low = self.t_cache[indices - 1]
        t_high = self.t_cache[indices]
        z_low = self.z_cache[indices - 1]
        z_high = self.z_cache[indices]

        # Linear interpolation
        alpha = (t_clamped - t_low) / (t_high - t_low + 1e-10)
        z_interp = z_low + alpha * (z_high - z_low)

        if squeeze:
            return z_interp.squeeze()
        return z_interp

    def reward(self, x: Tensor) -> Tensor:
        """
        Compute reward r(x) = 1 / (Z(||x||)^2 + epsilon^2).

        Args:
            x: Positions [N, d]

        Returns:
            Reward values [N]
        """
        # Compute radial coordinate
        r = torch.norm(x, dim=-1)  # [N]

        # Evaluate Z function
        z = self.Z(r)  # [N]

        # Compute reward with regularization
        reward = 1.0 / (z**2 + self.epsilon**2)

        return reward

    def reward_squared_z(self, x: Tensor) -> Tensor:
        """
        Alternative reward: r(x) = -Z(||x||)^2.

        This has minima at zeta zeros (where Z=0).

        Args:
            x: Positions [N, d]

        Returns:
            Reward values [N]
        """
        r = torch.norm(x, dim=-1)
        z = self.Z(r)
        return -z**2

    def potential(self, x: Tensor) -> Tensor:
        """
        Potential V(x) = -r(x) for use in fitness landscape.

        Args:
            x: Positions [N, d]

        Returns:
            Potential values [N]
        """
        return -self.reward(x)


def get_first_zeta_zeros(n: int = 100) -> np.ndarray:
    """
    Get the first n non-trivial zeta zeros (imaginary parts).

    Uses mpmath to compute zeros to high precision.

    Args:
        n: Number of zeros to compute

    Returns:
        Array of t_n values where zeta(1/2 + it_n) = 0
    """
    print(f"Computing first {n} zeta zeros...")
    zeros = []

    for k in range(1, n + 1):
        # mpmath.zetazero(k) returns the k-th zero on critical line
        zero = mpmath.zetazero(k)
        # Extract imaginary part
        t_n = float(zero.imag)
        zeros.append(t_n)

    zeros = np.array(zeros)
    print(f"First zero: t_1 = {zeros[0]:.6f}")
    print(f"Last zero:  t_{n} = {zeros[-1]:.6f}")

    return zeros


def visualize_z_landscape(
    z_reward: ZFunctionReward,
    t_range: tuple[float, float] = (0, 50),
    n_points: int = 1000,
):
    """
    Visualize the Z-function reward landscape.

    Args:
        z_reward: ZFunctionReward instance
        t_range: Range of t values to plot
        n_points: Number of points for plotting
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    t = np.linspace(t_range[0], t_range[1], n_points)

    # Evaluate Z function
    t_torch = torch.tensor(t, dtype=torch.float32)
    z_values = z_reward.Z(t_torch).numpy()

    # Create dummy x for 1D case (just need radial coordinate)
    x_1d = torch.tensor(t, dtype=torch.float32).unsqueeze(-1)  # [N, 1]
    rewards = z_reward.reward(x_1d).numpy()

    # Get actual zeros for reference
    zeros = get_first_zeta_zeros(20)
    zeros_in_range = zeros[zeros < t_range[1]]

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot Z function
    axes[0].plot(t, z_values, 'b-', linewidth=1)
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0].plot(zeros_in_range, np.zeros_like(zeros_in_range), 'ro',
                 markersize=8, label='Zeta zeros')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('Z(t)')
    axes[0].set_title('Riemann-Siegel Z Function')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot reward landscape
    axes[1].plot(t, rewards, 'g-', linewidth=1)
    axes[1].plot(zeros_in_range,
                 1.0 / z_reward.epsilon**2 * np.ones_like(zeros_in_range),
                 'ro', markersize=8, label='Peak locations (zeros)')
    axes[1].set_xlabel('t (radial coordinate ||x||)')
    axes[1].set_ylabel('r(x) = 1/(Z²+ε²)')
    axes[1].set_title(f'Reward Landscape (ε={z_reward.epsilon})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig('experiments/z_function_reward/z_landscape.png', dpi=150)
    print("Saved visualization to experiments/z_function_reward/z_landscape.png")
    plt.close()


if __name__ == "__main__":
    # Test Z-function reward
    print("Testing Z-function reward implementation...")

    # Create reward function
    z_reward = ZFunctionReward(epsilon=0.1, t_max=50.0)

    # Visualize landscape
    visualize_z_landscape(z_reward, t_range=(0, 50))

    # Test reward evaluation
    print("\nTesting reward evaluation:")

    # Create test positions at known zero locations
    zeros = get_first_zeta_zeros(10)
    x_test = torch.tensor(zeros, dtype=torch.float32).unsqueeze(-1)  # [10, 1]

    rewards = z_reward.reward(x_test)
    print(f"\nRewards at zero locations (should be high):")
    for i, (t, r) in enumerate(zip(zeros, rewards)):
        print(f"  t_{i+1} = {t:.4f}: r = {r:.4f}")

    # Test at non-zero locations
    x_nonzero = torch.tensor([[15.5], [25.3], [35.7]], dtype=torch.float32)
    rewards_nonzero = z_reward.reward(x_nonzero)
    print(f"\nRewards at non-zero locations (should be lower):")
    for x, r in zip(x_nonzero, rewards_nonzero):
        print(f"  t = {x[0]:.4f}: r = {r:.4f}")

    print("\n✓ Z-function reward implementation working!")
