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


class MockRunHistory:
    """Minimal RunHistory mock for testing qft_utils functions."""

    def __init__(
        self,
        N: int = 20,
        d: int = 5,
        n_steps: int = 100,
        n_recorded: int = 50,
        seed: int = 42,
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
