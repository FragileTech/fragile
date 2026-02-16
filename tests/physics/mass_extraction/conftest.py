"""Shared fixtures for mass extraction tests.

Provides synthetic exponential correlator data with known masses
for verifying the extraction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Known masses and amplitudes for synthetic data
# ---------------------------------------------------------------------------

KNOWN_MASS_0 = 0.3  # ground state
KNOWN_MASS_1 = 0.8  # first excited
KNOWN_AMP_0 = 1.0
KNOWN_AMP_1 = 0.3
MAX_LAG = 40
T_SERIES = 2000  # operator time series length


# ---------------------------------------------------------------------------
# Synthetic correlator
# ---------------------------------------------------------------------------


def make_synthetic_correlator(
    mass0: float = KNOWN_MASS_0,
    mass1: float = KNOWN_MASS_1,
    amp0: float = KNOWN_AMP_0,
    amp1: float = KNOWN_AMP_1,
    max_lag: int = MAX_LAG,
    noise_level: float = 1e-4,
    seed: int = 42,
) -> Tensor:
    """Create a synthetic 2-exponential correlator with noise.

    ``C(t) = amp0^2 * exp(-mass0 * t) + amp1^2 * exp(-mass1 * t) + noise``
    """
    rng = np.random.default_rng(seed)
    t = np.arange(max_lag + 1, dtype=np.float64)
    signal = amp0**2 * np.exp(-mass0 * t) + amp1**2 * np.exp(-mass1 * t)
    noise = rng.normal(0, noise_level, size=len(t))
    return torch.from_numpy(signal + noise).float()


def make_synthetic_operator_series(
    mass0: float = KNOWN_MASS_0,
    amp0: float = KNOWN_AMP_0,
    T: int = T_SERIES,
    noise_level: float = 0.1,
    seed: int = 42,
) -> Tensor:
    """Create a synthetic operator time series.

    Simple oscillating signal with noise: ``O(t) = amp0 * exp(-mass0/2 * mod(t, period))``
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float64)
    period = 50
    signal = amp0 * np.exp(-mass0 / 2 * (t % period))
    noise = rng.normal(0, noise_level, size=T)
    return torch.from_numpy(signal + noise).float()


# ---------------------------------------------------------------------------
# Pipeline result mock
# ---------------------------------------------------------------------------


@dataclass
class MockPipelineResult:
    """Mimics PipelineResult for testing without RunHistory dependency."""

    operators: dict[str, Tensor] = field(default_factory=dict)
    correlators: dict[str, Tensor] = field(default_factory=dict)
    prepared_data: object = None
    scales: Tensor | None = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def known_masses():
    """Known masses for verification."""
    return {"ground": KNOWN_MASS_0, "excited": KNOWN_MASS_1}


@pytest.fixture
def synthetic_correlator() -> Tensor:
    """Single synthetic 2-exponential correlator [max_lag+1]."""
    return make_synthetic_correlator()


@pytest.fixture
def synthetic_operator_series() -> Tensor:
    """Synthetic operator time series [T]."""
    return make_synthetic_operator_series()


@pytest.fixture
def synthetic_correlators() -> dict[str, Tensor]:
    """Multiple synthetic correlators mimicking real pipeline output."""
    return {
        "scalar": make_synthetic_correlator(seed=42),
        "pseudoscalar": make_synthetic_correlator(
            mass0=0.3, mass1=0.8, amp0=0.8, amp1=0.2, seed=43
        ),
        "vector": make_synthetic_correlator(
            mass0=0.5, mass1=1.0, amp0=0.9, amp1=0.25, seed=44
        ),
    }


@pytest.fixture
def synthetic_operators() -> dict[str, Tensor]:
    """Synthetic operator time series for all channels."""
    return {
        "scalar": make_synthetic_operator_series(seed=42),
        "pseudoscalar": make_synthetic_operator_series(seed=43),
        "vector": make_synthetic_operator_series(seed=44),
    }


@pytest.fixture
def mock_pipeline_result(synthetic_correlators, synthetic_operators) -> MockPipelineResult:
    """Mock PipelineResult with synthetic data."""
    return MockPipelineResult(
        operators=synthetic_operators,
        correlators=synthetic_correlators,
    )


@pytest.fixture
def multiscale_correlators() -> dict[str, Tensor]:
    """Multiscale correlators [S, max_lag+1]."""
    n_scales = 3
    corrs = []
    for s in range(n_scales):
        corrs.append(make_synthetic_correlator(seed=42 + s))
    return {"scalar": torch.stack(corrs)}
