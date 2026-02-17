"""Shared fixtures for physics/fractal_gas test suite."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Basic tensor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def N() -> int:
    """Default number of walkers."""
    return 20


@pytest.fixture
def d() -> int:
    """Default spatial dimension."""
    return 3


@pytest.fixture
def positions(N: int, d: int) -> Tensor:
    """Random walker positions [N, d], seeded."""
    gen = torch.Generator().manual_seed(42)
    return torch.randn(N, d, generator=gen)


@pytest.fixture
def velocities(N: int, d: int) -> Tensor:
    """Random walker velocities [N, d], seeded."""
    gen = torch.Generator().manual_seed(43)
    return torch.randn(N, d, generator=gen)


@pytest.fixture
def fitness_values(N: int) -> Tensor:
    """Positive fitness values [N], seeded."""
    gen = torch.Generator().manual_seed(44)
    return torch.rand(N, generator=gen) + 0.1  # ensure positive


@pytest.fixture
def rewards(N: int) -> Tensor:
    """Random reward values [N], seeded."""
    gen = torch.Generator().manual_seed(45)
    return torch.randn(N, generator=gen)


@pytest.fixture
def companions(N: int) -> Tensor:
    """Random companion indices [N] as mutual pairs."""
    from fragile.physics.fractal_gas.euclidean_gas import random_pairing_fisher_yates

    torch.manual_seed(46)
    return random_pairing_fisher_yates(N)


@pytest.fixture
def will_clone_half(N: int) -> Tensor:
    """Boolean mask where first half clones, second half doesn't."""
    mask = torch.zeros(N, dtype=torch.bool)
    mask[: N // 2] = True
    return mask


@pytest.fixture
def will_clone_none(N: int) -> Tensor:
    """Boolean mask where nobody clones."""
    return torch.zeros(N, dtype=torch.bool)


# ---------------------------------------------------------------------------
# SwarmState fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def swarm_state(positions: Tensor, velocities: Tensor):
    """A SwarmState instance."""
    from fragile.physics.fractal_gas.euclidean_gas import SwarmState

    return SwarmState(positions.clone(), velocities.clone())


# ---------------------------------------------------------------------------
# Operator fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kinetic_op():
    """A default KineticOperator for testing."""
    from fragile.physics.fractal_gas.kinetic_operator import KineticOperator

    return KineticOperator(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        temperature=0.5,
        nu=0.1,
        use_viscous_coupling=True,
        beta_curl=0.0,
    )


@pytest.fixture
def fitness_op():
    """A default FitnessOperator for testing."""
    from fragile.physics.fractal_gas.fitness import FitnessOperator

    return FitnessOperator(alpha=1.0, beta=1.0, eta=0.0, sigma_min=1e-8, A=2.0)


@pytest.fixture
def clone_op():
    """A default CloneOperator for testing."""
    from fragile.physics.fractal_gas.cloning import CloneOperator

    return CloneOperator(p_max=1.0, epsilon_clone=0.01, sigma_x=0.1, alpha_restitution=0.5)


# ---------------------------------------------------------------------------
# Neighbor edge fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_neighbor_edges(N: int) -> Tensor:
    """Simple chain-like neighbor edges [E, 2] for testing viscous coupling."""
    edges = []
    for i in range(N - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    return torch.tensor(edges, dtype=torch.long)


@pytest.fixture
def simple_edge_weights(simple_neighbor_edges: Tensor) -> Tensor:
    """Uniform edge weights [E] for neighbor edges."""
    E = simple_neighbor_edges.shape[0]
    return torch.ones(E) / 2  # normalized per source (2 neighbors each)
