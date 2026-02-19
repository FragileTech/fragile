"""AIC parity tests for gevp_channels module.

Verifies that the AIC copy produces identical results to the original
``fragile.fractalai.qft.gevp_channels`` module for all public and
semi-public helpers.
"""

from __future__ import annotations

import math

import pytest
import torch

from fragile.fractalai.qft.gevp_channels import (
    _build_whitener,
    _fft_cross_correlator_lags,
    _fft_cross_correlator_lags_batched,
    _fit_mass_from_correlator,
    _mass_from_fit_mode,
    _sanitize_mode,
    COMPANION_GEVP_BASE_CHANNELS,
    get_companion_gevp_basis_channels,
    GEVPConfig,
)
from fragile.physics.aic.gevp_channels import (
    _build_whitener as new_whitener,
    _fft_cross_correlator_lags as new_cross_corr,
    _fft_cross_correlator_lags_batched as new_cross_corr_batched,
    _fit_mass_from_correlator as new_fit_mass,
    _mass_from_fit_mode as new_mass_fit_mode,
    _sanitize_mode as new_sanitize,
    COMPANION_GEVP_BASE_CHANNELS as NEW_COMPANION_GEVP,
    get_companion_gevp_basis_channels as new_get_basis,
    GEVPConfig as NewGEVPConfig,
)
from tests.physics.aic.conftest import assert_mass_fit_equal, assert_tensor_or_nan_equal


# ---------------------------------------------------------------------------
# Cross-correlator parity
# ---------------------------------------------------------------------------


class TestParityCrossCorrelator:
    """Parity tests for FFT cross-correlator lag functions."""

    @pytest.mark.parametrize("seed", [42, 99, 2025])
    def test_cross_correlator_lags_parity(self, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        series = torch.randn(3, 50, generator=gen)

        old = _fft_cross_correlator_lags(series, max_lag=20, use_connected=True)
        new = new_cross_corr(series, max_lag=20, use_connected=True)

        assert_tensor_or_nan_equal(old, new, label=f"cross_corr seed={seed}")

    @pytest.mark.parametrize("seed", [42, 99, 2025])
    def test_batched_parity(self, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        series = torch.randn(4, 3, 50, generator=gen)

        old = _fft_cross_correlator_lags_batched(series, max_lag=20, use_connected=True)
        new = new_cross_corr_batched(series, max_lag=20, use_connected=True)

        assert_tensor_or_nan_equal(old, new, label=f"batched_cross_corr seed={seed}")


# ---------------------------------------------------------------------------
# Basis channels parity
# ---------------------------------------------------------------------------


class TestParityBasisChannels:
    """Parity tests for GEVP basis channel lookups."""

    @pytest.mark.parametrize(
        "base",
        ["nucleon", "scalar", "pseudoscalar", "glueball", "su2", "u1", "ew_mixed"],
    )
    def test_get_basis_channels_parity(self, base: str) -> None:
        old = get_companion_gevp_basis_channels(base)
        new = new_get_basis(base)
        assert old == new, f"Basis channels differ for '{base}': {old} vs {new}"

    def test_companion_gevp_base_channels_identical(self) -> None:
        assert set(COMPANION_GEVP_BASE_CHANNELS.keys()) == set(NEW_COMPANION_GEVP.keys()), (
            "Key sets differ: "
            f"{set(COMPANION_GEVP_BASE_CHANNELS.keys()) ^ set(NEW_COMPANION_GEVP.keys())}"
        )
        for key in COMPANION_GEVP_BASE_CHANNELS:
            assert COMPANION_GEVP_BASE_CHANNELS[key] == NEW_COMPANION_GEVP[key], (
                f"Values differ for key '{key}': "
                f"{COMPANION_GEVP_BASE_CHANNELS[key]} vs {NEW_COMPANION_GEVP[key]}"
            )


# ---------------------------------------------------------------------------
# Whitener parity
# ---------------------------------------------------------------------------


class TestParityWhitener:
    """Parity tests for _build_whitener."""

    def test_build_whitener_parity(self) -> None:
        gen = torch.Generator().manual_seed(123)
        a = torch.randn(4, 4, generator=gen)
        c0 = a @ a.T + 1e-3 * torch.eye(4)

        old_w, old_idx, old_cond = _build_whitener(
            c0,
            eig_rel_cutoff=1e-2,
            cond_limit=1e4,
            shrinkage=1e-6,
        )
        new_w, new_idx, new_cond = new_whitener(
            c0,
            eig_rel_cutoff=1e-2,
            cond_limit=1e4,
            shrinkage=1e-6,
        )

        assert_tensor_or_nan_equal(old_w, new_w, label="whitener matrix")
        assert torch.equal(old_idx, new_idx), f"Kept indices differ: {old_idx} vs {new_idx}"
        assert old_cond == new_cond, f"Condition numbers differ: {old_cond} vs {new_cond}"

    def test_build_whitener_degenerate_parity(self) -> None:
        c0 = torch.zeros(4, 4)
        # Both should raise ValueError for a zero matrix (not positive definite).
        old_raised = False
        new_raised = False
        try:
            _build_whitener(c0, eig_rel_cutoff=1e-2, cond_limit=1e4, shrinkage=0.0)
        except (ValueError, RuntimeError):
            old_raised = True
        try:
            new_whitener(c0, eig_rel_cutoff=1e-2, cond_limit=1e4, shrinkage=0.0)
        except (ValueError, RuntimeError):
            new_raised = True
        assert (
            old_raised == new_raised
        ), f"Degenerate matrix error behavior differs: old_raised={old_raised}, new_raised={new_raised}"


# ---------------------------------------------------------------------------
# Mass fitting parity
# ---------------------------------------------------------------------------


class TestParityMassFitting:
    """Parity tests for mass fitting helpers."""

    def test_fit_mass_from_correlator_parity(self) -> None:
        # Synthetic exponential correlator: C(t) = A * exp(-m * t)
        t = torch.arange(0, 30, dtype=torch.float32)
        correlator = 5.0 * torch.exp(-0.3 * t)
        config = GEVPConfig(fit_mode="aic", fit_start=2, min_fit_points=2)
        new_config = NewGEVPConfig(fit_mode="aic", fit_start=2, min_fit_points=2)

        old_fit, old_wm, old_wa, old_ww, old_wr = _fit_mass_from_correlator(
            correlator,
            dt=1.0,
            config=config,
        )
        new_fit, new_wm, new_wa, new_ww, new_wr = new_fit_mass(
            correlator,
            dt=1.0,
            config=new_config,
        )

        assert_mass_fit_equal(old_fit, new_fit, label="fit_mass_from_correlator")

        # Compare optional tensor outputs
        if old_wm is not None and new_wm is not None:
            assert_tensor_or_nan_equal(old_wm, new_wm, label="window_masses")
        else:
            assert (old_wm is None) == (
                new_wm is None
            ), f"window_masses None mismatch: old={old_wm is None}, new={new_wm is None}"

        if old_wa is not None and new_wa is not None:
            assert_tensor_or_nan_equal(old_wa, new_wa, label="window_aic")
        else:
            assert (old_wa is None) == (
                new_wa is None
            ), f"window_aic None mismatch: old={old_wa is None}, new={new_wa is None}"

        assert old_ww == new_ww, f"window_widths differ: {old_ww} vs {new_ww}"

        if old_wr is not None and new_wr is not None:
            assert_tensor_or_nan_equal(old_wr, new_wr, label="window_r2")
        else:
            assert (old_wr is None) == (
                new_wr is None
            ), f"window_r2 None mismatch: old={old_wr is None}, new={new_wr is None}"

    def test_mass_from_fit_mode_parity(self) -> None:
        t = torch.arange(0, 30, dtype=torch.float32)
        correlator = 5.0 * torch.exp(-0.3 * t)
        config = GEVPConfig(fit_mode="aic", fit_start=2, min_fit_points=2)
        new_config = NewGEVPConfig(fit_mode="aic", fit_start=2, min_fit_points=2)

        old_mass = _mass_from_fit_mode(correlator, dt=1.0, config=config)
        new_mass = new_mass_fit_mode(correlator, dt=1.0, config=new_config)

        if math.isnan(old_mass) and math.isnan(new_mass):
            pass  # Both NaN is fine
        else:
            assert old_mass == new_mass, f"mass_from_fit_mode differs: {old_mass} vs {new_mass}"


# ---------------------------------------------------------------------------
# Sanitize mode parity
# ---------------------------------------------------------------------------


class TestParitySanitizeMode:
    """Parity tests for _sanitize_mode helper."""

    @pytest.mark.parametrize(
        "mode, expected",
        [
            ("aic", "aic"),
            ("linear", "linear"),
            ("AIC", "aic"),
            ("  Linear  ", "linear"),
            ("invalid_mode", "aic"),
            ("", "aic"),
            ("LINEAR_ABS", "linear_abs"),
        ],
    )
    def test_sanitize_mode_parity(self, mode: str, expected: str) -> None:
        allowed = ("aic", "linear", "linear_abs")
        fallback = "aic"
        old_result = _sanitize_mode(mode, allowed, fallback)
        new_result = new_sanitize(mode, allowed, fallback)
        assert (
            old_result == new_result
        ), f"_sanitize_mode('{mode}') differs: old={old_result!r}, new={new_result!r}"
        assert (
            old_result == expected
        ), f"_sanitize_mode('{mode}') = {old_result!r}, expected {expected!r}"
