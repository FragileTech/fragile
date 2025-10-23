"""Shared fixtures for gauge symmetry tests."""

import pytest
import torch


@pytest.fixture
def simple_swarm_2d():
    """Create a simple 2D swarm configuration for testing.

    Returns:
        Dictionary with:
            - positions: [N, 2] walker positions
            - velocities: [N, 2] walker velocities
            - rewards: [N] reward values
            - diversity_companions: [N] diversity companion indices
            - clone_companions: [N] cloning companion indices
            - alive: [N] boolean mask
    """
    N = 50
    d = 2

    # Random positions in [0, 1]^2
    positions = torch.rand(N, d)

    # Small random velocities
    velocities = torch.randn(N, d) * 0.1

    # Rewards with some variation
    rewards = torch.randn(N) * 0.5 + 1.0

    # Random companion selections
    diversity_companions = torch.randint(0, N, (N,))
    clone_companions = torch.randint(0, N, (N,))

    # All alive
    alive = torch.ones(N, dtype=torch.bool)

    return {
        "positions": positions,
        "velocities": velocities,
        "rewards": rewards,
        "diversity_companions": diversity_companions,
        "clone_companions": clone_companions,
        "alive": alive,
        "N": N,
        "d": d,
    }


@pytest.fixture
def partially_dead_swarm_2d():
    """Create a 2D swarm with some dead walkers for robustness testing."""
    N = 50
    d = 2

    positions = torch.rand(N, d)
    velocities = torch.randn(N, d) * 0.1
    rewards = torch.randn(N) * 0.5 + 1.0
    diversity_companions = torch.randint(0, N, (N,))
    clone_companions = torch.randint(0, N, (N,))

    # 20% dead walkers
    alive = torch.rand(N) > 0.2

    return {
        "positions": positions,
        "velocities": velocities,
        "rewards": rewards,
        "diversity_companions": diversity_companions,
        "clone_companions": clone_companions,
        "alive": alive,
        "N": N,
        "d": d,
    }


@pytest.fixture
def clustered_swarm_2d():
    """Create a 2D swarm with clear clustering for locality tests.

    Creates 3 clusters at different locations.
    """
    N = 60
    d = 2
    cluster_size = 20

    # Create 3 clusters
    cluster_centers = torch.tensor([[0.2, 0.2], [0.5, 0.8], [0.8, 0.3]])

    positions = []
    for center in cluster_centers:
        cluster_pos = center + torch.randn(cluster_size, d) * 0.05
        positions.append(cluster_pos)
    positions = torch.cat(positions, dim=0)

    velocities = torch.randn(N, d) * 0.05
    rewards = torch.randn(N) * 0.5 + 1.0
    diversity_companions = torch.randint(0, N, (N,))
    clone_companions = torch.randint(0, N, (N,))
    alive = torch.ones(N, dtype=torch.bool)

    return {
        "positions": positions,
        "velocities": velocities,
        "rewards": rewards,
        "diversity_companions": diversity_companions,
        "clone_companions": clone_companions,
        "alive": alive,
        "N": N,
        "d": d,
    }
