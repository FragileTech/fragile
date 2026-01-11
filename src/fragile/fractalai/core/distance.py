"""Distance computation utilities for Euclidean Gas algorithms.

This module provides distance functions that handle both standard Euclidean
distances and periodic boundary conditions (PBC). With PBC enabled, the space
wraps around at boundaries (torus topology), so distances account for wrapping.

Mathematical Background:
-----------------------
With periodic boundaries on domain [low, high]^d:
- Domain size: L = high - low (per dimension)
- Minimum image convention: dx_wrapped = dx - L * round(dx / L)
- This ensures dx ∈ [-L/2, L/2], giving shortest distance through wrapping

Example (1D):
    Domain: [0, 1]
    Point A at x=0.1, Point B at x=0.9
    - Direct distance: |0.9 - 0.1| = 0.8
    - Periodic distance: min(0.8, 1.0 - 0.8) = 0.2 (wraps around)

Reference:
    - PBC implementation: fragile/core/bounds.py TorchBounds.apply_bounds()
    - Minimum image convention: Allen & Tildesley, Computer Simulation of Liquids
"""

from __future__ import annotations

import torch
from torch import Tensor

from fragile.bounds import TorchBounds


__all__ = [
    "compute_periodic_distance_matrix",
    "compute_periodic_distance_pairwise",
]


def compute_periodic_distance_matrix(
    x: Tensor,
    y: Tensor | None = None,
    bounds: TorchBounds | None = None,
    pbc: bool = False,
) -> Tensor:
    """Compute pairwise distance matrix with optional periodic boundary conditions.

    Args:
        x: Positions [N, d] or [M, d]
        y: Optional second set of positions [M, d]. If None, computes x-x distances.
        bounds: Domain bounds. Required if pbc=True.
        pbc: If True, use periodic boundary conditions (minimum image convention).

    Returns:
        Distance matrix:
        - If y is None: [N, N] symmetric matrix where [i,j] = distance(x[i], x[j])
        - If y is not None: [N, M] matrix where [i,j] = distance(x[i], y[j])

    Raises:
        ValueError: If pbc=True but bounds is None.

    Example:
        >>> x = torch.tensor([[0.1, 0.1], [0.9, 0.9]])
        >>> bounds = TorchBounds(low=torch.zeros(2), high=torch.ones(2))
        >>>
        >>> # Standard Euclidean distance
        >>> d_std = compute_periodic_distance_matrix(x, bounds=bounds, pbc=False)
        >>> # d_std[0,1] ≈ 1.13 (diagonal of unit square)
        >>>
        >>> # Periodic distance (wraps around)
        >>> d_pbc = compute_periodic_distance_matrix(x, bounds=bounds, pbc=True)
        >>> # d_pbc[0,1] ≈ 0.28 (shorter through wrapping)
    """
    if pbc and bounds is None:
        msg = "bounds required when pbc=True"
        raise ValueError(msg)

    # If not using PBC, delegate to standard torch.cdist
    if not pbc:
        if y is None:
            return torch.cdist(x, x, p=2)
        return torch.cdist(x, y, p=2)

    # Use PBC: minimum image convention
    # Reference: Allen & Tildesley § 1.3

    # Get domain size L = high - low
    L = bounds.high - bounds.low  # [d]

    # Compute displacement vectors
    if y is None:
        # Self-distance matrix: dx[i,j] = x[j] - x[i]
        dx = x.unsqueeze(0) - x.unsqueeze(1)  # [N, N, d]
    else:
        # Cross-distance matrix: dx[i,j] = y[j] - x[i]
        dx = y.unsqueeze(0) - x.unsqueeze(1)  # [N, M, d]

    # Apply minimum image convention: shift to [-L/2, L/2]
    # dx_wrapped = dx - L * round(dx / L)
    # This gives the shortest vector accounting for periodic wrapping
    dx_wrapped = dx - L * torch.round(dx / L)

    # Compute Euclidean distance from wrapped displacement
    return torch.norm(dx_wrapped, dim=-1, p=2)


def compute_periodic_distance_pairwise(
    x1: Tensor,
    x2: Tensor,
    bounds: TorchBounds | None = None,
    pbc: bool = False,
) -> Tensor:
    """Compute pairwise distances between corresponding points with optional PBC.

    Unlike compute_periodic_distance_matrix which computes all pairs, this
    computes distances between corresponding indices: distance(x1[i], x2[i]).

    Args:
        x1: First set of positions [N, d]
        x2: Second set of positions [N, d]
        bounds: Domain bounds. Required if pbc=True.
        pbc: If True, use periodic boundary conditions.

    Returns:
        Distances [N] where result[i] = distance(x1[i], x2[i])

    Raises:
        ValueError: If x1 and x2 have different shapes, or if pbc=True but bounds is None.

    Example:
        >>> x1 = torch.tensor([[0.1, 0.1], [0.9, 0.5]])
        >>> x2 = torch.tensor([[0.9, 0.9], [0.1, 0.5]])
        >>> bounds = TorchBounds(low=torch.zeros(2), high=torch.ones(2))
        >>>
        >>> # Standard distances
        >>> d_std = compute_periodic_distance_pairwise(x1, x2, bounds=bounds, pbc=False)
        >>> # d_std ≈ [1.13, 0.8]
        >>>
        >>> # Periodic distances
        >>> d_pbc = compute_periodic_distance_pairwise(x1, x2, bounds=bounds, pbc=True)
        >>> # d_pbc ≈ [0.28, 0.2] (shorter through wrapping)
    """
    if x1.shape != x2.shape:
        msg = f"x1 and x2 must have same shape, got {x1.shape} and {x2.shape}"
        raise ValueError(msg)

    if pbc and bounds is None:
        msg = "bounds required when pbc=True"
        raise ValueError(msg)

    # Compute displacement
    dx = x2 - x1  # [N, d]

    # If not using PBC, return standard Euclidean distance
    if not pbc:
        return torch.norm(dx, dim=-1, p=2)

    # Apply minimum image convention
    L = bounds.high - bounds.low  # [d]
    dx_wrapped = dx - L * torch.round(dx / L)

    # Compute distance
    return torch.norm(dx_wrapped, dim=-1, p=2)
