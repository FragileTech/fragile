"""Shared fixtures for physics/electroweak parity test suite."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from tests.physics.new_channels.conftest import assert_outputs_equal, MockRunHistory


# ---------------------------------------------------------------------------
# Parity helpers
# ---------------------------------------------------------------------------


def assert_dict_tensors_equal(old: dict[str, Tensor], new: dict[str, Tensor]) -> None:
    """Compare ``dict[str, Tensor]`` key-by-key with ``torch.equal``."""
    assert set(old.keys()) == set(new.keys()), f"Key mismatch: {sorted(set(old) ^ set(new))}"
    for key in old:
        assert torch.allclose(old[key], new[key], atol=1e-6, rtol=1e-5), (
            f"Key {key!r}: tensors differ.\n"
            f"  max abs diff = {(old[key] - new[key]).abs().max().item()}"
        )


def assert_dict_floats_equal(old: dict[str, float], new: dict[str, float]) -> None:
    """Compare ``dict[str, float]`` key-by-key (exact for finite, both-NaN check)."""
    assert set(old.keys()) == set(new.keys()), f"Key mismatch: {sorted(set(old) ^ set(new))}"
    for key in old:
        o, n = old[key], new[key]
        if math.isnan(o) and math.isnan(n):
            continue
        assert o == n, f"Key {key!r}: {o!r} != {n!r}"


def _assert_tensors_equal_nan_aware(old_val: Tensor, new_val: Tensor, label: str) -> None:
    """Assert two tensors are identical, treating NaN==NaN as True."""
    both_nan = torch.isnan(old_val) & torch.isnan(new_val)
    old_finite = torch.where(both_nan, torch.zeros_like(old_val), old_val)
    new_finite = torch.where(both_nan, torch.zeros_like(new_val), new_val)
    assert torch.allclose(old_finite, new_finite, atol=1e-6, rtol=1e-5), (
        f"{label}: tensors differ (NaN-aware).\n"
        f"  max abs diff = {(old_finite - new_finite).abs().max().item()}"
    )


def _assert_dataclass_equal_nan_aware(old_out, new_out) -> None:
    """Like assert_outputs_equal but handles NaN in tensors."""
    import dataclasses as _dc

    old_fields = {f.name for f in _dc.fields(old_out)}
    new_fields = {f.name for f in _dc.fields(new_out)}
    assert old_fields == new_fields, f"Field mismatch: {old_fields ^ new_fields}"

    for f in _dc.fields(old_out):
        old_val = getattr(old_out, f.name)
        new_val = getattr(new_out, f.name)
        if old_val is None and new_val is None:
            continue
        if isinstance(old_val, Tensor):
            assert isinstance(
                new_val, Tensor
            ), f"Field {f.name}: old is Tensor, new is {type(new_val)}"
            _assert_tensors_equal_nan_aware(old_val, new_val, f"Field {f.name}")
        elif isinstance(old_val, dict):
            assert isinstance(new_val, dict)
            assert set(old_val.keys()) == set(new_val.keys())
            for k in old_val:
                ov, nv = old_val[k], new_val[k]
                if isinstance(ov, Tensor):
                    _assert_tensors_equal_nan_aware(ov, nv, f"Field {f.name}[{k!r}]")
                elif isinstance(ov, float):
                    if math.isnan(ov) and math.isnan(nv):
                        continue
                    assert math.isclose(
                        ov, nv, rel_tol=1e-5, abs_tol=1e-6
                    ), f"Field {f.name}[{k!r}]: {ov!r} != {nv!r}"
                else:
                    assert ov == nv, f"Field {f.name}[{k!r}]: {ov!r} != {nv!r}"
        else:
            assert old_val == new_val, f"Field {f.name}: {old_val!r} != {new_val!r}"


def assert_channel_results_equal(old: dict, new: dict) -> None:
    """Compare ``dict[str, ChannelCorrelatorResult]`` by iterating keys (NaN-aware)."""
    assert set(old.keys()) == set(new.keys()), f"Key mismatch: {sorted(set(old) ^ set(new))}"
    for key in old:
        _assert_dataclass_equal_nan_aware(old[key], new[key])


# ---------------------------------------------------------------------------
# ElectroweakMockRunHistory
# ---------------------------------------------------------------------------


class ElectroweakMockRunHistory(MockRunHistory):
    """Extended MockRunHistory with electroweak-specific attributes.

    Adds params dict, delta_t, and forces the companion path by
    setting neighbor_edges / edge_weights / will_clone to None.
    """

    def __init__(
        self,
        N: int = 20,
        d: int = 4,
        n_recorded: int = 30,
        seed: int = 42,
    ):
        super().__init__(N=N, d=d, n_recorded=n_recorded, seed=seed)
        self.params = {
            "companion_selection": {"epsilon": 1.0},
            "companion_selection_clone": {"epsilon": 1.0},
            "cloning": {"epsilon_clone": 1e-8},
            "kinetic": {"nu": 0.1},
        }
        self.delta_t = 0.01
        self.neighbor_edges = None
        self.edge_weights = None
        self.will_clone = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TINY_T = 5
_TINY_N = 3
_TINY_D = 3


@pytest.fixture
def mock_history() -> ElectroweakMockRunHistory:
    """Default seeded ElectroweakMockRunHistory: N=20, d=4, n_recorded=30."""
    return ElectroweakMockRunHistory()


@pytest.fixture
def mock_history_with_will_clone() -> ElectroweakMockRunHistory:
    """ElectroweakMockRunHistory with a seeded will_clone tensor."""
    h = ElectroweakMockRunHistory()
    gen = torch.Generator().manual_seed(99)
    h.will_clone = torch.randint(
        0,
        2,
        (h.n_recorded, h.N),
        generator=gen,
        dtype=torch.bool,
    )
    return h


@pytest.fixture
def tiny_positions() -> Tensor:
    """Seeded 3D positions [5, 3, 3]."""
    gen = torch.Generator().manual_seed(9)
    return torch.randn(_TINY_T, _TINY_N, _TINY_D, generator=gen)


@pytest.fixture
def tiny_velocities() -> Tensor:
    """Seeded 3D velocities [5, 3, 3]."""
    gen = torch.Generator().manual_seed(11)
    return torch.randn(_TINY_T, _TINY_N, _TINY_D, generator=gen)


@pytest.fixture
def tiny_fitness() -> Tensor:
    """Seeded positive fitness values [3]."""
    gen = torch.Generator().manual_seed(10)
    return torch.rand(_TINY_N, generator=gen).clamp(min=1e-6)


@pytest.fixture
def tiny_alive() -> Tensor:
    """All-True alive mask [3]."""
    return torch.ones(_TINY_N, dtype=torch.bool)


@pytest.fixture
def tiny_will_clone() -> Tensor:
    """Seeded bool will_clone [3]."""
    gen = torch.Generator().manual_seed(12)
    return torch.randint(0, 2, (_TINY_N,), generator=gen, dtype=torch.bool)


@pytest.fixture
def tiny_neighbors():
    """PackedNeighbors from cyclic companions for N=3 walkers."""
    from fragile.physics.electroweak.electroweak_observables import (
        pack_neighbors_from_edges,
    )

    N = _TINY_N
    src = torch.arange(N, dtype=torch.long)
    dst = src.roll(-1)
    edges = torch.stack([src, dst], dim=1)
    weights = torch.ones(N, dtype=torch.float32)
    alive = torch.ones(N, dtype=torch.bool)
    return pack_neighbors_from_edges(
        edges=edges,
        edge_weights=weights,
        alive=alive,
        n_walkers=N,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
