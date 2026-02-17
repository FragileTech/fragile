"""Shared fixtures for physics/qft_utils test suite."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------


@pytest.fixture
def T() -> int:
    """Number of time frames."""
    return 10


@pytest.fixture
def N() -> int:
    """Number of walkers."""
    return 20


@pytest.fixture
def d() -> int:
    """Spatial dimension."""
    return 5


# ---------------------------------------------------------------------------
# Companion indices
# ---------------------------------------------------------------------------


@pytest.fixture
def companions_distance(T: int, N: int) -> Tensor:
    """Companion distance indices [T, N] — cyclic shift by 1."""
    return torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).clone()


@pytest.fixture
def companions_clone(T: int, N: int) -> Tensor:
    """Companion clone indices [T, N] — cyclic shift by 2."""
    return torch.arange(N).roll(-2).unsqueeze(0).expand(T, -1).clone()


# ---------------------------------------------------------------------------
# MockRunHistory for color_states and helpers
# ---------------------------------------------------------------------------


def _build_ring_graph(
    n_recorded: int,
    N: int,
) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    """Build a connected ring graph (i <-> i+1 mod N) with uniform weights."""
    src = torch.arange(N, dtype=torch.long)
    dst = (src + 1) % N
    # Undirected: both directions
    edges_frame = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=1)  # [2*N, 2]
    n_edges = edges_frame.shape[0]
    weights_frame = {
        "riemannian_kernel_volume": torch.ones(n_edges, dtype=torch.float32),
    }
    neighbor_edges = [edges_frame.clone() for _ in range(n_recorded)]
    edge_weights = [{k: v.clone() for k, v in weights_frame.items()} for _ in range(n_recorded)]
    return neighbor_edges, edge_weights


class MockRunHistory:
    """Minimal RunHistory mock for testing qft_utils functions."""

    def __init__(
        self,
        N: int = 20,
        d: int = 5,
        n_steps: int = 100,
        n_recorded: int = 50,
        seed: int = 42,
        with_neighbor_graph: bool = False,
    ):
        self.N = N
        self.d = d
        self.n_steps = n_steps
        self._n_recorded = n_recorded

        gen = torch.Generator().manual_seed(seed)

        # Recorded steps: evenly spaced
        step_interval = max(1, n_steps // max(n_recorded, 1))
        self._recorded_steps = [i * step_interval for i in range(n_recorded)]

        # History tensors [n_recorded, N, d]
        self.x_before_clone = torch.randn(n_recorded, N, d, generator=gen)
        self.v_before_clone = torch.randn(n_recorded, N, d, generator=gen)
        self.force_viscous = torch.randn(n_recorded, N, d, generator=gen)

        # Companion indices [n_recorded, N]
        self.companions_distance = (
            torch.arange(N).roll(-1).unsqueeze(0).expand(n_recorded, -1).clone()
        )
        self.companions_clone = (
            torch.arange(N).roll(-2).unsqueeze(0).expand(n_recorded, -1).clone()
        )

        # Cloning scores [n_recorded, N]
        self.cloning_scores = torch.randn(n_recorded, N, generator=gen)

        # Optional neighbor graph data
        if with_neighbor_graph and N > 1:
            self.neighbor_edges, self.edge_weights = _build_ring_graph(n_recorded, N)
        else:
            self.neighbor_edges: list[Tensor] | None = None
            self.edge_weights: list[dict[str, Tensor]] | None = None

    @property
    def n_recorded(self) -> int:
        return self._n_recorded

    @property
    def recorded_steps(self) -> list[int]:
        return self._recorded_steps

    def get_step_index(self, step: int) -> int:
        return self._recorded_steps.index(step)


@pytest.fixture
def mock_history() -> MockRunHistory:
    """A default MockRunHistory for testing."""
    return MockRunHistory()
