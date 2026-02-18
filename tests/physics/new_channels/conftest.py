"""Shared fixtures for physics/new_channels regression test suite."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Parity helper: assert two dataclass outputs are field-by-field identical
# ---------------------------------------------------------------------------


def assert_outputs_equal(old_out, new_out):
    """Assert that two dataclass outputs have identical fields.

    - Tensor fields: exact bit-for-bit equality via ``torch.equal``.
    - Optional Tensor fields: both None or both equal.
    - Scalar / list / tuple / str fields: ``==``.
    """
    old_fields = {f.name for f in dataclasses.fields(old_out)}
    new_fields = {f.name for f in dataclasses.fields(new_out)}
    assert old_fields == new_fields, f"Field mismatch: {old_fields ^ new_fields}"

    for f in dataclasses.fields(old_out):
        old_val = getattr(old_out, f.name)
        new_val = getattr(new_out, f.name)
        if old_val is None and new_val is None:
            continue
        if isinstance(old_val, Tensor):
            assert isinstance(
                new_val, Tensor
            ), f"Field {f.name}: old is Tensor, new is {type(new_val)}"
            assert torch.equal(old_val, new_val), (
                f"Field {f.name}: tensors differ.\n"
                f"  max abs diff = {(old_val - new_val).abs().max().item()}"
            )
        else:
            assert old_val == new_val, f"Field {f.name}: {old_val!r} != {new_val!r}"


# ---------------------------------------------------------------------------
# MockRunHistory (seeded, deterministic)
# ---------------------------------------------------------------------------


class MockRunHistory:
    """Minimal RunHistory mock for testing companion channel functions.

    All random tensors are generated from a seeded torch.Generator for
    deterministic, platform-independent test data.

    Companions use cyclic shifts so that all (i, j, k) triplets are
    distinct, which is required by the baryon/glueball channels.
    """

    def __init__(
        self,
        N: int = 20,
        d: int = 4,
        n_recorded: int = 30,
        seed: int = 42,
    ):
        self.N = N
        self.d = d
        self._n_recorded = n_recorded
        self.pbc = False
        self.bounds = None

        gen = torch.Generator().manual_seed(seed)

        # Recorded steps: evenly spaced, starting from 0
        self.record_every = 10
        self._recorded_steps = [i * self.record_every for i in range(n_recorded)]

        # History tensors [n_recorded, N, d]
        self.x_before_clone = torch.randn(n_recorded, N, d, generator=gen)
        self.x_final = torch.randn(n_recorded, N, d, generator=gen)
        self.v_before_clone = torch.randn(n_recorded, N, d, generator=gen)
        self.force_viscous = torch.randn(n_recorded, N, d, generator=gen)

        # Alive mask [n_recorded, N] — all alive
        self.alive_mask = torch.ones(n_recorded, N, dtype=torch.bool)

        # Companion indices [n_recorded, N] — cyclic shifts guarantee distinct triplets
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


# ---------------------------------------------------------------------------
# Fixtures: MockRunHistory
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_history() -> MockRunHistory:
    """Default seeded MockRunHistory: N=20, d=4, n_recorded=30."""
    return MockRunHistory()


# ---------------------------------------------------------------------------
# Tiny fixtures for analytical / Layer-A tests
# ---------------------------------------------------------------------------

_TINY_T = 5
_TINY_N = 3
_TINY_D = 3


@pytest.fixture
def tiny_color_states() -> Tensor:
    """Seeded complex color states [5, 3, 3], unit-normalized."""
    gen = torch.Generator().manual_seed(7)
    real = torch.randn(_TINY_T, _TINY_N, _TINY_D, generator=gen)
    imag = torch.randn(_TINY_T, _TINY_N, _TINY_D, generator=gen)
    c = torch.complex(real, imag)
    norms = c.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return c / norms


@pytest.fixture
def tiny_color_valid() -> Tensor:
    """All-valid color mask [5, 3]."""
    return torch.ones(_TINY_T, _TINY_N, dtype=torch.bool)


@pytest.fixture
def tiny_companions_distance() -> Tensor:
    """Cyclic shift-1 companion indices [5, 3]."""
    return torch.arange(_TINY_N).roll(-1).unsqueeze(0).expand(_TINY_T, -1).clone()


@pytest.fixture
def tiny_companions_clone() -> Tensor:
    """Cyclic shift-2 companion indices [5, 3]."""
    return torch.arange(_TINY_N).roll(-2).unsqueeze(0).expand(_TINY_T, -1).clone()


@pytest.fixture
def tiny_alive() -> Tensor:
    """All-True alive mask [5, 3]."""
    return torch.ones(_TINY_T, _TINY_N, dtype=torch.bool)


@pytest.fixture
def tiny_scores() -> Tensor:
    """Seeded cloning scores [5, 3]."""
    gen = torch.Generator().manual_seed(8)
    return torch.randn(_TINY_T, _TINY_N, generator=gen)


@pytest.fixture
def tiny_positions() -> Tensor:
    """Seeded 3D positions [5, 3, 3]."""
    gen = torch.Generator().manual_seed(9)
    return torch.randn(_TINY_T, _TINY_N, _TINY_D, generator=gen)
