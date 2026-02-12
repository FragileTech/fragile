"""Tests for companion glueball/baryon phase-augmented operator modes."""

from __future__ import annotations

import torch

from fragile.fractalai.qft.baryon_triplet_channels import compute_baryon_correlator_from_color
from fragile.fractalai.qft.glueball_color_channels import (
    compute_glueball_color_correlator_from_color,
)
from fragile.fractalai.qft.meson_phase_channels import compute_meson_phase_correlator_from_color


def _build_inputs(*, t_len: int = 7, n_walkers: int = 11) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(12345)
    color = torch.complex(
        torch.randn(t_len, n_walkers, 3, dtype=torch.float32),
        torch.randn(t_len, n_walkers, 3, dtype=torch.float32),
    )
    color_valid = torch.ones(t_len, n_walkers, dtype=torch.bool)
    alive = torch.ones(t_len, n_walkers, dtype=torch.bool)
    idx = torch.arange(n_walkers, dtype=torch.long)
    companions_distance = ((idx + 1) % n_walkers).view(1, n_walkers).expand(t_len, -1).clone()
    companions_clone = ((idx + 2) % n_walkers).view(1, n_walkers).expand(t_len, -1).clone()
    scores = torch.randn(t_len, n_walkers, dtype=torch.float32)
    return color, color_valid, alive, companions_distance, companions_clone, scores


def test_glueball_operator_mode_backward_compatibility() -> None:
    """Default glueball path should match explicit re_plaquette mode."""
    color, color_valid, alive, comp_d, comp_c, _ = _build_inputs()
    out_default = compute_glueball_color_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=5,
        use_connected=True,
        eps=1e-12,
        operator_mode=None,
        use_action_form=False,
    )
    out_mode = compute_glueball_color_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=5,
        use_connected=True,
        eps=1e-12,
        operator_mode="re_plaquette",
        use_action_form=False,
    )
    torch.testing.assert_close(out_default.correlator, out_mode.correlator, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(
        out_default.operator_glueball_series,
        out_mode.operator_glueball_series,
        atol=1e-7,
        rtol=1e-6,
    )


def test_glueball_phase_modes_are_finite_and_nonnegative() -> None:
    """Phase-based glueball observables should remain finite and non-negative."""
    color, color_valid, alive, comp_d, comp_c, _ = _build_inputs()
    for mode in ("phase_action", "phase_sin2"):
        out = compute_glueball_color_correlator_from_color(
            color=color,
            color_valid=color_valid,
            alive=alive,
            companions_distance=comp_d,
            companions_clone=comp_c,
            max_lag=5,
            use_connected=True,
            eps=1e-12,
            operator_mode=mode,
            use_action_form=False,
        )
        assert bool(torch.isfinite(out.correlator).all())
        assert bool(torch.isfinite(out.operator_glueball_series).all())
        assert bool((out.operator_glueball_series >= 0).all())


def test_baryon_operator_mode_backward_compatibility() -> None:
    """det_abs mode should reproduce the original baryon correlator path."""
    color, color_valid, alive, comp_d, comp_c, _ = _build_inputs()
    out_default = compute_baryon_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=5,
        use_connected=True,
        eps=1e-12,
    )
    out_mode = compute_baryon_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=5,
        use_connected=True,
        eps=1e-12,
        operator_mode="det_abs",
    )
    torch.testing.assert_close(out_default.correlator, out_mode.correlator, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(
        out_default.operator_baryon_series,
        out_mode.operator_baryon_series,
        atol=1e-7,
        rtol=1e-6,
    )


def test_baryon_flux_modes_are_finite() -> None:
    """Flux-weighted baryon modes should produce finite correlators."""
    color, color_valid, alive, comp_d, comp_c, _ = _build_inputs()
    for mode in ("flux_action", "flux_sin2", "flux_exp"):
        out = compute_baryon_correlator_from_color(
            color=color,
            color_valid=color_valid,
            alive=alive,
            companions_distance=comp_d,
            companions_clone=comp_c,
            max_lag=5,
            use_connected=True,
            eps=1e-12,
            operator_mode=mode,
            flux_exp_alpha=1.0,
        )
        assert bool(torch.isfinite(out.correlator).all())
        assert bool(torch.isfinite(out.operator_baryon_series).all())


def test_meson_score_directed_mode_matches_standard_for_equal_scores() -> None:
    """Score-directed meson mode reduces to standard mode when scores are equal."""
    color, color_valid, alive, comp_d, comp_c, _ = _build_inputs()
    scores_equal = torch.zeros_like(color_valid, dtype=torch.float32)
    out_standard = compute_meson_phase_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=5,
        use_connected=True,
        pair_selection="both",
        eps=1e-12,
        operator_mode="standard",
        scores=scores_equal,
    )
    out_directed = compute_meson_phase_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=5,
        use_connected=True,
        pair_selection="both",
        eps=1e-12,
        operator_mode="score_directed",
        scores=scores_equal,
    )
    torch.testing.assert_close(out_directed.scalar, out_standard.scalar, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(
        out_directed.pseudoscalar, out_standard.pseudoscalar, atol=1e-7, rtol=1e-6
    )


def test_meson_abs2_vacsub_mode_is_finite_and_nonnegative() -> None:
    """abs2_vacsub meson mode should produce finite scalar observables."""
    color, color_valid, alive, comp_d, comp_c, _ = _build_inputs()
    out_abs2 = compute_meson_phase_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=5,
        use_connected=True,
        pair_selection="both",
        eps=1e-12,
        operator_mode="abs2_vacsub",
    )
    assert bool(torch.isfinite(out_abs2.scalar).all())
    assert bool(torch.isfinite(out_abs2.operator_scalar_series).all())
    assert bool((out_abs2.operator_scalar_series >= 0).all())


def test_baryon_score_modes_are_finite() -> None:
    """Score-ordered baryon modes should produce finite correlators."""
    color, color_valid, alive, comp_d, comp_c, scores = _build_inputs()
    for mode in ("score_signed", "score_abs"):
        out = compute_baryon_correlator_from_color(
            color=color,
            color_valid=color_valid,
            alive=alive,
            companions_distance=comp_d,
            companions_clone=comp_c,
            max_lag=5,
            use_connected=True,
            eps=1e-12,
            operator_mode=mode,
            scores=scores,
        )
        assert bool(torch.isfinite(out.correlator).all())
        assert bool(torch.isfinite(out.operator_baryon_series).all())
        if mode == "score_abs":
            assert bool((out.operator_baryon_series >= 0).all())
