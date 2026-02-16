"""Tests for FFT-based correlator computation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fragile.physics.qft_utils.fft import _fft_correlator_batched


class TestFftCorrelatorBatchedBasic:
    """Basic functionality tests."""

    def test_output_shape(self):
        """Output shape is [B, max_lag+1]."""
        B, T = 3, 100
        max_lag = 20
        series = torch.randn(B, T)
        result = _fft_correlator_batched(series, max_lag)
        assert result.shape == (B, max_lag + 1)

    def test_single_constant_series_connected_true(self):
        """Single constant series with connected=True: all zeros (mean subtracted = 0)."""
        series = torch.ones(1, 50) * 5.0
        result = _fft_correlator_batched(series, max_lag=10, use_connected=True)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)

    def test_single_constant_series_connected_false(self):
        """Single constant series with connected=False: C(0) = constant^2, C(t>0) = constant^2."""
        constant = 3.0
        T = 50
        series = torch.ones(1, T) * constant
        result = _fft_correlator_batched(series, max_lag=10, use_connected=False)
        expected = constant**2
        assert torch.allclose(result, torch.full_like(result, expected), atol=1e-5)

    def test_all_zero_series(self):
        """All-zero series: all-zero correlator."""
        series = torch.zeros(2, 50)
        result = _fft_correlator_batched(series, max_lag=10, use_connected=True)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-7)

    def test_dtype_is_float32(self):
        """dtype is float32 regardless of input dtype."""
        series_int = torch.randint(0, 10, (2, 50))
        result_int = _fft_correlator_batched(series_int, max_lag=10)
        assert result_int.dtype == torch.float32

        series_double = torch.randn(2, 50, dtype=torch.float64)
        result_double = _fft_correlator_batched(series_double, max_lag=10)
        assert result_double.dtype == torch.float32

    def test_max_lag_zero(self):
        """max_lag=0 returns shape [B, 1]."""
        series = torch.randn(3, 50)
        result = _fft_correlator_batched(series, max_lag=0)
        assert result.shape == (3, 1)

    def test_single_batch_element(self):
        """B=1 single series works."""
        series = torch.randn(1, 100)
        result = _fft_correlator_batched(series, max_lag=20)
        assert result.shape == (1, 21)

    def test_multiple_max_lags(self):
        """Different max_lag values produce correct output shapes."""
        series = torch.randn(2, 100)
        for max_lag in [5, 20, 50, 99]:
            result = _fft_correlator_batched(series, max_lag)
            assert result.shape == (2, max_lag + 1)


class TestFftCorrelatorBatchedValidation:
    """Input validation tests."""

    def test_wrong_ndim_1d(self):
        """Wrong ndim (1D) raises ValueError."""
        series = torch.randn(50)
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            _fft_correlator_batched(series, max_lag=10)

    def test_wrong_ndim_3d(self):
        """Wrong ndim (3D) raises ValueError."""
        series = torch.randn(2, 50, 3)
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            _fft_correlator_batched(series, max_lag=10)

    def test_empty_batch_dimension(self):
        """Empty batch dimension (B=0): returns [0, max_lag+1]."""
        series = torch.randn(0, 50)
        result = _fft_correlator_batched(series, max_lag=10)
        assert result.shape == (0, 11)


class TestFftCorrelatorBatchedMath:
    """Mathematical properties tests."""

    def test_c0_non_negative_connected(self):
        """C(0) >= 0 for connected correlator (variance is non-negative)."""
        series = torch.randn(5, 100)
        result = _fft_correlator_batched(series, max_lag=20, use_connected=True)
        assert torch.all(result[:, 0] >= -1e-6)  # Allow small numerical error

    def test_sinusoidal_signal(self):
        """Sinusoidal signal: C(0) is maximum, C(period/2) is negative, C(period) close to C(0)."""
        T = 200
        period = 40
        t = torch.arange(T, dtype=torch.float32)
        series = torch.sin(2 * np.pi * t / period).unsqueeze(0)

        result = _fft_correlator_batched(series, max_lag=period, use_connected=True)

        c0 = result[0, 0].item()
        c_half_period = result[0, period // 2].item()
        c_period = result[0, period].item()

        # C(0) should be positive and maximum (or at least among the largest)
        assert c0 > 0
        # For a sinusoid, C(0) and |C(period/2)| are approximately equal
        assert abs(c0 - abs(c_half_period)) < 0.1

        # C(period/2) should be negative
        assert c_half_period < 0

        # C(period) should be close to C(0) (periodic)
        assert abs(c_period - c0) < 0.1 * abs(c0)

    def test_delta_function(self):
        """Delta function (impulse at t=0): C(0) = 1/T, C(t>0) near 0 (after mean subtraction)."""
        T = 100
        series = torch.zeros(1, T)
        series[0, 0] = 1.0

        result = _fft_correlator_batched(series, max_lag=20, use_connected=True)

        # After mean subtraction, the impulse becomes (1 - 1/T) at t=0
        # and -1/T elsewhere. The autocorrelation is more complex.
        # Just check that C(t>0) is much smaller than C(0)
        assert result[0, 0] > 0
        assert torch.max(torch.abs(result[0, 1:])) < 0.5 * result[0, 0]

    def test_linear_ramp(self):
        """Linear ramp: known correlator structure."""
        T = 100
        series = torch.arange(T, dtype=torch.float32).unsqueeze(0)

        result = _fft_correlator_batched(series, max_lag=20, use_connected=True)

        # For a linear ramp with mean subtracted, the correlator should be smooth
        # and decrease with lag. C(0) should be positive (variance of linear ramp).
        assert result[0, 0] > 0
        # Check monotonic decrease (with some tolerance)
        for i in range(1, 10):
            assert result[0, i] < result[0, i - 1] + 1.0  # Allow some numerical noise

    def test_independent_batch_elements(self):
        """Two independent batch elements produce independent correlators."""
        series1 = torch.randn(1, 100)
        series2 = torch.randn(1, 100) + 10.0  # Different mean

        combined = torch.cat([series1, series2], dim=0)
        result_combined = _fft_correlator_batched(combined, max_lag=20)

        result1 = _fft_correlator_batched(series1, max_lag=20)
        result2 = _fft_correlator_batched(series2, max_lag=20)

        assert torch.allclose(result_combined[0], result1[0], atol=1e-5)
        assert torch.allclose(result_combined[1], result2[0], atol=1e-5)

    def test_max_lag_greater_than_T_minus_1(self):
        """max_lag > T-1: entries beyond T-1 are zero-padded."""
        T = 20
        series = torch.randn(2, T)
        max_lag = 50

        result = _fft_correlator_batched(series, max_lag=max_lag)

        # Entries beyond T-1 should be zero
        assert torch.allclose(result[:, T:], torch.zeros_like(result[:, T:]), atol=1e-7)

    def test_symmetry_real_series(self):
        """Symmetry check: for a real series, C(t) is real (no imaginary component)."""
        series = torch.randn(3, 100)
        result = _fft_correlator_batched(series, max_lag=20)

        # Result should be real (dtype is float32)
        assert result.dtype == torch.float32
        # All values should be finite
        assert torch.all(torch.isfinite(result))

    def test_normalization_constant_series(self):
        """Normalization: C(0) with connected=False and constant series equals constant^2."""
        constant = 7.0
        series = torch.ones(2, 100) * constant

        result = _fft_correlator_batched(series, max_lag=10, use_connected=False)

        assert torch.allclose(result[:, 0], torch.tensor(constant**2), atol=1e-5)

    def test_large_max_lag(self):
        """Large max_lag (much larger than T): output has correct shape with zero padding."""
        T = 30
        max_lag = 1000
        series = torch.randn(2, T)

        result = _fft_correlator_batched(series, max_lag=max_lag)

        assert result.shape == (2, max_lag + 1)
        # Most entries should be zero
        assert torch.allclose(result[:, T:], torch.zeros_like(result[:, T:]), atol=1e-7)

    def test_batch_identical_series(self):
        """Batch of identical series: all correlators identical."""
        series_single = torch.randn(1, 100)
        series_batch = series_single.repeat(5, 1)

        result = _fft_correlator_batched(series_batch, max_lag=20)

        for i in range(1, 5):
            assert torch.allclose(result[i], result[0], atol=1e-5)


class TestFftCorrelatorBatchedConnected:
    """Tests for connected vs not-connected correlator."""

    def test_connected_differs_from_not_connected(self):
        """connected=True vs connected=False differ for non-zero-mean series."""
        series = torch.randn(2, 100) + 5.0  # Non-zero mean

        result_connected = _fft_correlator_batched(series, max_lag=20, use_connected=True)
        result_not_connected = _fft_correlator_batched(series, max_lag=20, use_connected=False)

        # They should be different
        assert not torch.allclose(result_connected, result_not_connected, atol=1e-3)

    def test_connected_true_gives_variance(self):
        """connected=True gives C(0) = variance."""
        series = torch.randn(3, 100) + 10.0

        result = _fft_correlator_batched(series, max_lag=20, use_connected=True)

        # C(0) should equal variance
        expected_variance = series.var(dim=1, unbiased=False)
        assert torch.allclose(result[:, 0], expected_variance, atol=1e-4)

    def test_connected_false_gives_mean_of_squared(self):
        """connected=False gives C(0) = mean of x_t^2 (not variance)."""
        series = torch.randn(3, 100) + 5.0

        result = _fft_correlator_batched(series, max_lag=20, use_connected=False)

        # C(0) should equal mean of x_t^2
        expected_mean_squared = (series**2).mean(dim=1)
        assert torch.allclose(result[:, 0], expected_mean_squared, atol=1e-4)

    def test_zero_mean_series_connected_equivalence(self):
        """Zero-mean series: connected and not-connected give same result."""
        series = torch.randn(2, 100)
        # Ensure exactly zero mean
        series = series - series.mean(dim=1, keepdim=True)

        result_connected = _fft_correlator_batched(series, max_lag=20, use_connected=True)
        result_not_connected = _fft_correlator_batched(series, max_lag=20, use_connected=False)

        # They should be the same for zero-mean series
        assert torch.allclose(result_connected, result_not_connected, atol=1e-4)
