"""Shared fixtures and helpers for AIC parity test suite.

Re-exports helpers from tests.physics.new_channels.conftest and adds
AIC-specific comparison utilities.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from torch import Tensor

from tests.physics.new_channels.conftest import (
    assert_outputs_equal,
    MockRunHistory,
)


__all__ = [
    "MockRunHistory",
    "assert_dict_results_equal",
    "assert_mass_fit_equal",
    "assert_outputs_equal",
    "assert_tensor_or_nan_equal",
    "mock_history",
]


# ---------------------------------------------------------------------------
# NaN-tolerant tensor comparison
# ---------------------------------------------------------------------------


def assert_tensor_or_nan_equal(old: Tensor, new: Tensor, label: str = "") -> None:
    """Assert tensors are equal, tolerating matching NaN positions."""
    prefix = f"{label}: " if label else ""
    assert old.shape == new.shape, f"{prefix}shape mismatch {old.shape} vs {new.shape}"
    old_nan = torch.isnan(old)
    new_nan = torch.isnan(new)
    assert torch.equal(old_nan, new_nan), f"{prefix}NaN pattern differs"
    finite = ~old_nan
    if finite.any():
        assert torch.equal(old[finite], new[finite]), (
            f"{prefix}finite values differ, "
            f"max abs diff = {(old[finite] - new[finite]).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# Mass fit dict comparison
# ---------------------------------------------------------------------------


def assert_mass_fit_equal(
    old_fit: dict[str, Any], new_fit: dict[str, Any], label: str = ""
) -> None:
    """Compare mass_fit dicts: tensor values via torch.equal, scalars via ==, NaN-tolerant."""
    prefix = f"{label}: " if label else ""
    assert set(old_fit.keys()) == set(
        new_fit.keys()
    ), f"{prefix}key mismatch: {set(old_fit.keys()) ^ set(new_fit.keys())}"
    for key in old_fit:
        old_val = old_fit[key]
        new_val = new_fit[key]
        if old_val is None and new_val is None:
            continue
        if isinstance(old_val, Tensor) and isinstance(new_val, Tensor):
            assert_tensor_or_nan_equal(old_val, new_val, label=f"{prefix}fit[{key}]")
        elif isinstance(old_val, float) and isinstance(new_val, float):
            if math.isnan(old_val) and math.isnan(new_val):
                continue
            assert old_val == new_val, f"{prefix}fit[{key}]: {old_val!r} != {new_val!r}"
        elif isinstance(old_val, list) and isinstance(new_val, list):
            assert old_val == new_val, f"{prefix}fit[{key}]: lists differ"
        else:
            assert old_val == new_val, f"{prefix}fit[{key}]: {old_val!r} != {new_val!r}"


# ---------------------------------------------------------------------------
# ChannelCorrelatorResult dict comparison
# ---------------------------------------------------------------------------


def assert_dict_results_equal(
    old: dict[str, Any],
    new: dict[str, Any],
    label: str = "",
) -> None:
    """Compare dict[str, ChannelCorrelatorResult] field-by-field."""
    prefix = f"{label}: " if label else ""
    assert set(old.keys()) == set(
        new.keys()
    ), f"{prefix}channel key mismatch: {set(old.keys()) ^ set(new.keys())}"
    for ch in old:
        assert_outputs_equal(old[ch], new[ch])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_history() -> MockRunHistory:
    """Default seeded MockRunHistory: N=20, d=4, n_recorded=30."""
    return MockRunHistory()
