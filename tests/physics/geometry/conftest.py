"""Shared fixtures for physics/geometry test suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from fragile.physics.geometry.delaunai import build_delaunay_edges


def _build_knn_graph(positions: Tensor, k: int) -> Tensor:
    """Build a symmetric kNN graph from positions using cdist + topk."""
    dists = torch.cdist(positions, positions)
    # Set self-distance to inf so we don't pick self as neighbor
    dists.fill_diagonal_(float("inf"))
    _, indices = dists.topk(k, dim=1, largest=False)
    N = positions.shape[0]
    src = torch.arange(N).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = indices.reshape(-1)
    # Stack and make symmetric
    edge_index = torch.stack([src, dst], dim=0)
    edge_index_rev = torch.stack([dst, src], dim=0)
    edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
    # Remove duplicates
    return torch.unique(edge_index, dim=1)


# ---------------------------------------------------------------------------
# Point cloud fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grid_2d_positions() -> Tensor:
    """10x10 regular grid on [-2, 2]^2, shape [100, 2]."""
    xs = torch.linspace(-2, 2, 10)
    ys = torch.linspace(-2, 2, 10)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), dim=-1)
    return grid.reshape(-1, 2)


@pytest.fixture
def grid_3d_positions() -> Tensor:
    """5x5x5 regular grid on [-1, 1]^3, shape [125, 3]."""
    xs = torch.linspace(-1, 1, 5)
    ys = torch.linspace(-1, 1, 5)
    zs = torch.linspace(-1, 1, 5)
    grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)
    return grid.reshape(-1, 3)


@pytest.fixture
def random_2d_positions() -> Tensor:
    """50 random points, seeded, shape [50, 2]."""
    gen = torch.Generator().manual_seed(42)
    return torch.randn(50, 2, generator=gen)


@pytest.fixture
def clustered_2d_positions() -> Tensor:
    """60 points in 3 Gaussian clusters, shape [60, 2]."""
    gen = torch.Generator().manual_seed(123)
    centers = torch.tensor([[0.0, 0.0], [3.0, 3.0], [-3.0, 2.0]])
    points = []
    for c in centers:
        pts = 0.3 * torch.randn(20, 2, generator=gen) + c
        points.append(pts)
    return torch.cat(points, dim=0)


# ---------------------------------------------------------------------------
# Edge / graph fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def delaunay_2d_edges(grid_2d_positions: Tensor) -> Tensor:
    """Delaunay edges from grid_2d, shape [2, E]."""
    edges_np = build_delaunay_edges(grid_2d_positions.numpy())
    return torch.as_tensor(edges_np, dtype=torch.long).t()


@pytest.fixture
def random_2d_edges(random_2d_positions: Tensor) -> Tensor:
    """kNN(k=8) edges from random_2d, shape [2, E]."""
    return _build_knn_graph(random_2d_positions, k=8)


@pytest.fixture
def empty_edge_index() -> Tensor:
    """Empty edge index, shape [2, 0]."""
    return torch.zeros(2, 0, dtype=torch.long)


# ---------------------------------------------------------------------------
# Metric fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def identity_metric_2d() -> Tensor:
    """Identity [2,2] for N=100, shape [100, 2, 2]."""
    return torch.eye(2).unsqueeze(0).expand(100, -1, -1).clone()


@pytest.fixture
def diagonal_metric_2d() -> Tensor:
    """diag(2, 0.5) for N=100, shape [100, 2, 2]."""
    m = torch.diag(torch.tensor([2.0, 0.5]))
    return m.unsqueeze(0).expand(100, -1, -1).clone()


@pytest.fixture
def full_spd_metric_2d() -> Tensor:
    """Random SPD 2x2 for N=100, seeded, shape [100, 2, 2]."""
    gen = torch.Generator().manual_seed(99)
    A = torch.randn(100, 2, 2, generator=gen)
    # M = A @ A^T + eps*I  guarantees SPD
    return A @ A.transpose(-1, -2) + 0.1 * torch.eye(2).unsqueeze(0)


# ---------------------------------------------------------------------------
# Fitness landscape fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def quadratic_fitness(grid_2d_positions: Tensor, delaunay_2d_edges: Tensor) -> dict:
    """V = x^2 + 2y^2 on 10x10 grid with Delaunay edges.

    Returns dict with positions, fitness, edge_index, hessian_true, grad_fn.
    """
    pos = grid_2d_positions
    x, y = pos[:, 0], pos[:, 1]
    fitness = x**2 + 2 * y**2
    hessian_true = torch.zeros(pos.shape[0], 2, 2)
    hessian_true[:, 0, 0] = 2.0
    hessian_true[:, 1, 1] = 4.0

    def grad_fn(p: Tensor) -> Tensor:
        return torch.stack([2 * p[:, 0], 4 * p[:, 1]], dim=1)

    return {
        "positions": pos,
        "fitness": fitness,
        "edge_index": delaunay_2d_edges,
        "hessian_true": hessian_true,
        "grad_fn": grad_fn,
    }


@pytest.fixture
def linear_fitness(grid_2d_positions: Tensor, delaunay_2d_edges: Tensor) -> dict:
    """V = 3x + 5y on 10x10 grid with Delaunay edges.

    Returns dict with positions, fitness, edge_index, grad_true.
    """
    pos = grid_2d_positions
    fitness = 3 * pos[:, 0] + 5 * pos[:, 1]
    grad_true = torch.zeros(pos.shape[0], 2)
    grad_true[:, 0] = 3.0
    grad_true[:, 1] = 5.0
    return {
        "positions": pos,
        "fitness": fitness,
        "edge_index": delaunay_2d_edges,
        "grad_true": grad_true,
    }


# ---------------------------------------------------------------------------
# CSR fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_csr() -> dict:
    """5-node graph in CSR format.

    Graph: 0-1, 0-2, 1-2, 2-3, 3-4  (symmetric)
    """
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 2, 3, 3, 4], [1, 2, 0, 2, 0, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    from fragile.physics.geometry.neighbors import build_csr_from_coo

    csr = build_csr_from_coo(edge_index, n_nodes=5)
    return {
        "edge_index": edge_index,
        "n_nodes": 5,
        **csr,
    }
