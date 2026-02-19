"""Regression tests for the FFT correlator engine."""

from __future__ import annotations

import pytest
import torch

from fragile.fractalai.qft.correlator_channels import (
    _fft_correlator_batched,
    _fft_correlator_single,
    compute_effective_mass_torch,
)

# New-location aliases for parity tests
from fragile.physics.new_channels.correlator_channels import (
    _fft_correlator_batched as new_fft_batched,
    _fft_correlator_single as new_fft_single,
    compute_effective_mass_torch as new_effective_mass,
)


# ── Shape ─────────────────────────────────────────────────────────────────────


class TestFFTCorrelatorShape:
    def test_batched_output_shape(self):
        gen = torch.Generator().manual_seed(0)
        series = torch.randn(4, 50, generator=gen)
        result = _fft_correlator_batched(series, max_lag=10)
        assert result.shape == (4, 11)

    def test_batched_output_shape_large_lag(self):
        gen = torch.Generator().manual_seed(0)
        series = torch.randn(2, 20, generator=gen)
        result = _fft_correlator_batched(series, max_lag=30)
        # max_lag > T: should still return shape [B, max_lag+1]
        assert result.shape == (2, 31)


# ── Constant series ──────────────────────────────────────────────────────────


class TestFFTCorrelatorConstant:
    def test_connected_constant_is_zero(self):
        series = torch.full((3, 40), 5.0)
        result = _fft_correlator_batched(series, max_lag=10, use_connected=True)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    def test_disconnected_constant(self):
        series = torch.full((1, 40), 3.0)
        result = _fft_correlator_batched(series, max_lag=5, use_connected=False)
        expected = torch.full((1, 6), 9.0)
        assert torch.allclose(result, expected, atol=1e-5)


# ── Single vs batched ────────────────────────────────────────────────────────


class TestSingleVsBatched:
    def test_single_matches_batched_first_row(self):
        gen = torch.Generator().manual_seed(11)
        series_1d = torch.randn(60, generator=gen)
        single = _fft_correlator_single(series_1d, max_lag=15)
        batched = _fft_correlator_batched(series_1d.unsqueeze(0), max_lag=15)
        assert torch.allclose(single, batched.squeeze(0), atol=1e-6)


# ── FFT vs direct loop ───────────────────────────────────────────────────────


class TestFFTVsDirectLoop:
    def test_fft_matches_direct_correlation(self):
        gen = torch.Generator().manual_seed(22)
        T = 100
        series = torch.randn(T, generator=gen)
        max_lag = 20

        # Direct double-loop computation
        direct = torch.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            vals = []
            mean = series.mean()
            for t in range(T - lag):
                vals.append((series[t] - mean) * (series[t + lag] - mean))
            if vals:
                direct[lag] = torch.stack(vals).mean()

        fft_result = _fft_correlator_batched(
            series.unsqueeze(0), max_lag=max_lag, use_connected=True
        ).squeeze(0)
        assert torch.allclose(fft_result, direct, atol=1e-5)


# ── Connected C(0) = variance ────────────────────────────────────────────────


class TestConnectedVariance:
    def test_c0_equals_variance(self):
        gen = torch.Generator().manual_seed(33)
        series = torch.randn(1, 500, generator=gen)
        result = _fft_correlator_batched(series, max_lag=5, use_connected=True)
        expected_var = series.var(dim=-1, correction=0)
        assert torch.allclose(result[:, 0], expected_var, atol=1e-4)


# ── max_lag > T ──────────────────────────────────────────────────────────────


class TestMaxLagExceedsT:
    def test_pads_with_zeros(self):
        gen = torch.Generator().manual_seed(44)
        series = torch.randn(1, 10, generator=gen)
        result = _fft_correlator_batched(series, max_lag=20)
        assert result.shape == (1, 21)
        # Beyond T, values should be zero
        assert torch.allclose(result[:, 11:], torch.zeros(1, 10), atol=1e-6)


# ── Empty series ─────────────────────────────────────────────────────────────


class TestEmptySeries:
    def test_empty_returns_zeros(self):
        series = torch.zeros(2, 0)
        result = _fft_correlator_batched(series, max_lag=5)
        assert result.shape == (2, 6)
        assert torch.allclose(result, torch.zeros_like(result))


# ── Effective mass ───────────────────────────────────────────────────────────


class TestEffectiveMass:
    def test_pure_exponential(self):
        dt = 0.5
        m_true = 0.3
        t = torch.arange(20, dtype=torch.float32)
        correlator = torch.exp(-m_true * t)
        m_eff = compute_effective_mass_torch(correlator, dt)
        expected = torch.full_like(m_eff, m_true / dt)
        assert torch.allclose(m_eff, expected, atol=1e-4)

    def test_empty_correlator(self):
        correlator = torch.tensor([])
        result = compute_effective_mass_torch(correlator, dt=1.0)
        assert result.numel() == 0

    def test_single_point(self):
        correlator = torch.tensor([1.0])
        result = compute_effective_mass_torch(correlator, dt=1.0)
        assert result.numel() == 0


# =============================================================================
# Layer C: Old-vs-New Parity Tests
# =============================================================================


class TestParityFFT:
    """Verify new-location FFT functions produce identical results to originals."""

    def test_fft_batched_parity(self):
        gen = torch.Generator().manual_seed(100)
        series = torch.randn(4, 50, generator=gen)
        old = _fft_correlator_batched(series, max_lag=10)
        new = new_fft_batched(series, max_lag=10)
        assert torch.allclose(old, new, atol=1e-6)

    def test_fft_batched_connected_parity(self):
        gen = torch.Generator().manual_seed(101)
        series = torch.randn(3, 40, generator=gen)
        old = _fft_correlator_batched(series, max_lag=8, use_connected=True)
        new = new_fft_batched(series, max_lag=8, use_connected=True)
        assert torch.allclose(old, new, atol=1e-6)

    def test_fft_single_parity(self):
        gen = torch.Generator().manual_seed(102)
        series = torch.randn(60, generator=gen)
        old = _fft_correlator_single(series, max_lag=15)
        new = new_fft_single(series, max_lag=15)
        assert torch.allclose(old, new, atol=1e-6)

    def test_effective_mass_parity(self):
        t = torch.arange(20, dtype=torch.float32)
        correlator = torch.exp(-0.3 * t)
        old = compute_effective_mass_torch(correlator, dt=0.5)
        new = new_effective_mass(correlator, dt=0.5)
        assert torch.allclose(old, new, atol=1e-6)
