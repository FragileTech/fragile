"""Comprehensive tests for fragile.physics.operators.correlators."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from fragile.physics.operators.correlators import compute_correlators_batched


# ============================================================================
# Helpers
# ============================================================================


def _constant_series(value: float, T: int) -> Tensor:
    """Return a constant scalar series [T]."""
    return torch.full((T,), value, dtype=torch.float32)


def _sinusoidal_series(T: int, period: int) -> Tensor:
    """Return a sinusoidal scalar series [T] with the given period."""
    t = torch.arange(T, dtype=torch.float32)
    return torch.sin(2.0 * math.pi * t / period)


# ============================================================================
# TestComputeCorrelatorsSingleScale
# ============================================================================


class TestComputeCorrelatorsSingleScale:
    """Tests for compute_correlators_batched with n_scales=1 (default)."""

    # --- Structural / edge-case tests ---

    def test_empty_operators_returns_empty(self):
        """Empty dict input returns an empty dict."""
        result = compute_correlators_batched({}, max_lag=5)
        assert result == {}

    def test_single_scalar_series_shape(self):
        """A single scalar series [T] produces a correlator of shape [max_lag+1]."""
        T, max_lag = 20, 8
        ops = {"scalar_op": torch.randn(T)}
        result = compute_correlators_batched(ops, max_lag=max_lag)
        assert "scalar_op" in result
        assert result["scalar_op"].shape == (max_lag + 1,)

    def test_multiple_scalar_series_shapes(self):
        """Multiple scalar series each produce the correct shape."""
        T, max_lag = 30, 10
        ops = {
            "alpha": torch.randn(T),
            "beta": torch.randn(T),
            "gamma": torch.randn(T),
        }
        result = compute_correlators_batched(ops, max_lag=max_lag)
        assert len(result) == 3
        for name in ops:
            assert result[name].shape == (max_lag + 1,)

    def test_vector_series_contracted_shape(self):
        """A vector series [T, 3] produces a contracted scalar correlator [max_lag+1]."""
        T, max_lag = 25, 6
        ops = {"vec_op": torch.randn(T, 3)}
        result = compute_correlators_batched(ops, max_lag=max_lag)
        assert result["vec_op"].shape == (max_lag + 1,)

    def test_tensor_series_contracted_shape(self):
        """A tensor series [T, 5] produces a contracted scalar correlator [max_lag+1]."""
        T, max_lag = 25, 6
        ops = {"tens_op": torch.randn(T, 5)}
        result = compute_correlators_batched(ops, max_lag=max_lag)
        assert result["tens_op"].shape == (max_lag + 1,)

    def test_mixed_scalar_and_vector(self):
        """Mixed scalar + vector operators all get correct shapes."""
        T, max_lag = 20, 5
        ops = {
            "scalar": torch.randn(T),
            "vector": torch.randn(T, 3),
            "tensor": torch.randn(T, 5),
        }
        result = compute_correlators_batched(ops, max_lag=max_lag)
        assert result["scalar"].shape == (max_lag + 1,)
        assert result["vector"].shape == (max_lag + 1,)
        assert result["tensor"].shape == (max_lag + 1,)

    # --- Value / physics tests ---

    def test_connected_autocorrelation_at_zero_nonnegative(self):
        """C(0) >= 0 for a connected correlator (auto-correlation at lag 0)."""
        T, max_lag = 50, 10
        torch.manual_seed(123)
        ops = {"s": torch.randn(T)}
        result = compute_correlators_batched(ops, max_lag=max_lag, use_connected=True)
        assert result["s"][0].item() >= 0.0

    def test_constant_series_connected_gives_zero(self):
        """Constant series with connected=True should yield C(tau)=0 for all tau."""
        T, max_lag = 30, 10
        ops = {"const": _constant_series(7.5, T)}
        result = compute_correlators_batched(ops, max_lag=max_lag, use_connected=True)
        torch.testing.assert_close(
            result["const"],
            torch.zeros(max_lag + 1),
            atol=1e-5,
            rtol=0.0,
        )

    def test_constant_series_not_connected(self):
        """Constant series with connected=False -> C(tau) = constant^2."""
        T, max_lag = 30, 10
        val = 3.0
        ops = {"const": _constant_series(val, T)}
        result = compute_correlators_batched(ops, max_lag=max_lag, use_connected=False)
        expected = torch.full((max_lag + 1,), val * val)
        torch.testing.assert_close(result["const"], expected, atol=1e-4, rtol=1e-4)

    def test_sinusoidal_signal_peak_at_period(self):
        """Correlator of a sinusoidal signal peaks at multiples of the period."""
        period = 10
        T = 200  # long enough for several periods
        max_lag = 30
        ops = {"sin": _sinusoidal_series(T, period)}
        result = compute_correlators_batched(ops, max_lag=max_lag, use_connected=True)
        corr = result["sin"]

        # C(0) should be a peak (the maximum)
        assert corr[0] > corr[1]
        # C(period) should be a local peak higher than neighboring lags
        assert corr[period] > corr[period - 1]
        assert corr[period] > corr[period + 1]
        # Near half-period, correlator should be negative (anti-correlated)
        half = period // 2
        assert corr[half] < 0.0

    def test_zero_series_gives_zero_correlator(self):
        """A series of all zeros should give a zero correlator."""
        T, max_lag = 20, 8
        ops = {"zero": torch.zeros(T)}
        result = compute_correlators_batched(ops, max_lag=max_lag)
        torch.testing.assert_close(
            result["zero"],
            torch.zeros(max_lag + 1),
            atol=1e-7,
            rtol=0.0,
        )

    def test_empty_element_series_gives_zero_filled(self):
        """An empty-element series (numel=0) is filled with zeros."""
        T, max_lag = 20, 5
        ops = {
            "good": torch.randn(T),
            "empty": torch.zeros(0),
        }
        result = compute_correlators_batched(ops, max_lag=max_lag)
        assert result["empty"].shape == (max_lag + 1,)
        torch.testing.assert_close(
            result["empty"],
            torch.zeros(max_lag + 1),
            atol=1e-7,
            rtol=0.0,
        )

    def test_max_lag_greater_than_T_minus_1_zero_padded(self):
        """When max_lag > T-1, extra entries are zero-padded."""
        T = 5
        max_lag = 20
        ops = {"s": torch.randn(T)}
        result = compute_correlators_batched(ops, max_lag=max_lag, use_connected=True)
        assert result["s"].shape == (max_lag + 1,)
        # Entries beyond T-1 should be zero
        beyond = result["s"][T:]
        torch.testing.assert_close(
            beyond,
            torch.zeros_like(beyond),
            atol=1e-7,
            rtol=0.0,
        )

    def test_max_lag_zero_gives_just_c0(self):
        """max_lag=0 produces a single-element correlator [C(0)]."""
        T = 20
        ops = {"s": torch.randn(T)}
        result = compute_correlators_batched(ops, max_lag=0, use_connected=True)
        assert result["s"].shape == (1,)
        assert result["s"][0].item() >= 0.0

    def test_vector_contraction_equals_sum_of_component_correlators(self):
        """Vector correlator equals sum of per-component scalar correlators."""
        T, max_lag = 40, 8
        torch.manual_seed(7)
        vec = torch.randn(T, 3)

        # Compute via the batched function as a vector
        vec_result = compute_correlators_batched({"v": vec}, max_lag=max_lag, use_connected=True)[
            "v"
        ]

        # Compute each component separately and sum
        comp_sum = torch.zeros(max_lag + 1)
        for c in range(3):
            comp_corr = compute_correlators_batched(
                {"c": vec[:, c]}, max_lag=max_lag, use_connected=True
            )["c"]
            comp_sum += comp_corr

        torch.testing.assert_close(vec_result, comp_sum, atol=1e-5, rtol=1e-5)


# ============================================================================
# TestComputeCorrelatorsMultiscale
# ============================================================================


class TestComputeCorrelatorsMultiscale:
    """Tests for compute_correlators_batched with n_scales > 1."""

    def test_scalar_multiscale_shape(self):
        """n_scales=3 with [S, T] input produces shape [S, max_lag+1]."""
        S, T, max_lag = 3, 20, 7
        ops = {"ms": torch.randn(S, T)}
        result = compute_correlators_batched(ops, max_lag=max_lag, n_scales=S)
        assert result["ms"].shape == (S, max_lag + 1)

    def test_multicomponent_multiscale_shape(self):
        """n_scales=3 with [S, T, C] input produces shape [S, max_lag+1]."""
        S, T, C, max_lag = 3, 20, 5, 7
        ops = {"ms_vec": torch.randn(S, T, C)}
        result = compute_correlators_batched(ops, max_lag=max_lag, n_scales=S)
        assert result["ms_vec"].shape == (S, max_lag + 1)

    def test_single_scale_fallback(self):
        """n_scales=1 falls back to single-scale mode (1D series)."""
        T, max_lag = 20, 5
        ops = {"s": torch.randn(T)}
        result = compute_correlators_batched(ops, max_lag=max_lag, n_scales=1)
        assert result["s"].shape == (max_lag + 1,)
        assert result["s"].ndim == 1

    def test_empty_series_multiscale_zero_filled(self):
        """Empty series in multiscale mode produces zeros [S, max_lag+1]."""
        S, max_lag = 4, 6
        ops = {
            "good": torch.randn(S, 15),
            "empty": torch.zeros(0),
        }
        result = compute_correlators_batched(ops, max_lag=max_lag, n_scales=S)
        assert result["empty"].shape == (S, max_lag + 1)
        torch.testing.assert_close(
            result["empty"],
            torch.zeros(S, max_lag + 1),
            atol=1e-7,
            rtol=0.0,
        )

    def test_each_scale_independent(self):
        """Different series per scale yield different correlators per scale."""
        S, T, max_lag = 3, 50, 10
        torch.manual_seed(42)
        # Each scale is a different random series
        series = torch.randn(S, T)
        # Make scales genuinely different
        series[0] *= 1.0
        series[1] *= 5.0
        series[2] *= 0.1

        result = compute_correlators_batched(
            {"op": series}, max_lag=max_lag, n_scales=S, use_connected=True
        )
        corr = result["op"]  # [S, max_lag+1]

        # C(0) for each scale should differ significantly
        c0_values = corr[:, 0]
        assert not torch.allclose(c0_values[0:1], c0_values[1:2], atol=1e-2)

    def test_multiscale_matches_per_scale_computation(self):
        """Multiscale result matches computing each scale independently."""
        S, T, max_lag = 3, 30, 8
        torch.manual_seed(55)
        series = torch.randn(S, T)

        # Multiscale computation
        ms_result = compute_correlators_batched(
            {"op": series}, max_lag=max_lag, n_scales=S, use_connected=True
        )["op"]

        # Per-scale computation (using single-scale mode)
        for s in range(S):
            single_result = compute_correlators_batched(
                {"op": series[s]}, max_lag=max_lag, n_scales=1, use_connected=True
            )["op"]
            torch.testing.assert_close(ms_result[s], single_result, atol=1e-5, rtol=1e-5)

    def test_multiscale_multicomponent_output_shape_and_properties(self):
        """Multiscale [S, T, C] produces correct shape and non-negative C(0)."""
        S, T, C, max_lag = 2, 40, 3, 6
        torch.manual_seed(77)
        series = torch.randn(S, T, C)

        result = compute_correlators_batched(
            {"op": series}, max_lag=max_lag, n_scales=S, use_connected=True
        )["op"]

        assert result.shape == (S, max_lag + 1)
        # C(0) is a sum of per-component auto-correlations, so it must be >= 0
        for s in range(S):
            assert result[s, 0].item() >= 0.0

    def test_multiscale_multicomponent_contraction_internal(self):
        """Multiscale [S, T, C] contraction matches manual reshape-sum path."""
        S, T, C, max_lag = 2, 40, 3, 6
        torch.manual_seed(77)
        series = torch.randn(S, T, C)

        from fragile.physics.qft_utils.fft import _fft_correlator_batched

        # Full multiscale multi-component via public API
        full_result = compute_correlators_batched(
            {"op": series}, max_lag=max_lag, n_scales=S, use_connected=True
        )["op"]  # [S, max_lag+1]

        # Reproduce the internal reshape logic manually
        flat = series.reshape(S * C, T)
        corr = _fft_correlator_batched(flat, max_lag=max_lag, use_connected=True)
        manual = corr.reshape(S, C, -1).sum(dim=1)

        torch.testing.assert_close(full_result, manual, atol=1e-6, rtol=1e-6)

    def test_multiscale_constant_connected_zero(self):
        """Constant multiscale series with connected=True -> all zeros."""
        S, T, max_lag = 3, 20, 5
        series = torch.full((S, T), 4.0)
        result = compute_correlators_batched(
            {"op": series}, max_lag=max_lag, n_scales=S, use_connected=True
        )["op"]
        torch.testing.assert_close(
            result,
            torch.zeros(S, max_lag + 1),
            atol=1e-5,
            rtol=0.0,
        )
