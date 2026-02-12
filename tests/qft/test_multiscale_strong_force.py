"""Tests for multiscale strong-force channel series computation."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from fragile.fractalai.qft.baryon_triplet_channels import compute_baryon_correlator_from_color
from fragile.fractalai.qft.correlator_channels import CorrelatorConfig
from fragile.fractalai.qft.glueball_color_channels import (
    compute_glueball_color_correlator_from_color,
)
from fragile.fractalai.qft.meson_phase_channels import compute_meson_phase_correlator_from_color
from fragile.fractalai.qft.multiscale_strong_force import (
    _compute_channel_series_from_kernels,
    _compute_companion_per_scale_results_preserving_original,
    _select_best_scale,
)
from fragile.fractalai.qft.vector_meson_channels import (
    compute_vector_meson_correlator_from_color_positions,
)


def _build_random_inputs(*, t_len: int, n_scales: int, n_walkers: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(1234)
    color_real = torch.randn(t_len, n_walkers, 3, dtype=torch.float32)
    color_imag = torch.randn(t_len, n_walkers, 3, dtype=torch.float32)
    color = torch.complex(color_real, color_imag)
    color_valid = torch.ones(t_len, n_walkers, dtype=torch.bool)
    positions = torch.randn(t_len, n_walkers, 3, dtype=torch.float32)
    alive = torch.ones(t_len, n_walkers, dtype=torch.bool)
    force = torch.randn(t_len, n_walkers, 3, dtype=torch.float32)

    kernels = torch.rand(t_len, n_scales, n_walkers, n_walkers, dtype=torch.float32)
    eye = torch.eye(n_walkers, dtype=torch.float32).view(1, 1, n_walkers, n_walkers)
    kernels = kernels * (1.0 - eye)
    kernels = kernels / kernels.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    dx = positions[:, :, None, :] - positions[:, None, :, :]
    pairwise_distances = torch.linalg.vector_norm(dx, dim=-1)
    finite_pos = pairwise_distances[torch.isfinite(pairwise_distances) & (pairwise_distances > 0)]
    if finite_pos.numel() > 0:
        probs = torch.linspace(0.2, 0.9, n_scales, dtype=torch.float32)
        scales = torch.quantile(finite_pos, probs).clamp(min=1e-6)
    else:
        scales = torch.linspace(1e-3, 1.0, n_scales, dtype=torch.float32)

    idx = torch.arange(n_walkers, dtype=torch.long)
    companions_distance = ((idx + 1) % n_walkers).view(1, n_walkers).expand(t_len, -1).clone()
    companions_clone = ((idx + 2) % n_walkers).view(1, n_walkers).expand(t_len, -1).clone()
    cloning_scores = torch.randn(t_len, n_walkers, dtype=torch.float32)

    return {
        "color": color,
        "color_valid": color_valid,
        "positions": positions,
        "alive": alive,
        "force": force,
        "kernels": kernels,
        "scales": scales,
        "pairwise_distances": pairwise_distances,
        "companions_distance": companions_distance,
        "companions_clone": companions_clone,
        "cloning_scores": cloning_scores,
    }


def _fake_scale_result(
    *,
    mass: float,
    r2: float,
    mass_error: float,
    n_valid_windows: int,
    aic: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        mass_fit={
            "mass": float(mass),
            "r_squared": float(r2),
            "mass_error": float(mass_error),
            "n_valid_windows": int(n_valid_windows),
            "best_window": {"aic": float(aic)},
        },
        window_masses=None,
    )


def test_select_best_scale_respects_quality_filters() -> None:
    """Best-scale selector should ignore scales that fail active quality thresholds."""
    results = [
        _fake_scale_result(mass=0.55, r2=0.92, mass_error=0.10, n_valid_windows=5, aic=3.0),
        _fake_scale_result(mass=0.50, r2=0.95, mass_error=0.08, n_valid_windows=2, aic=1.0),
        _fake_scale_result(mass=0.45, r2=0.30, mass_error=0.05, n_valid_windows=8, aic=0.5),
    ]

    best_idx = _select_best_scale(
        results,
        min_r2=0.8,
        min_windows=3,
        max_error_pct=30.0,
        remove_artifacts=False,
    )
    assert best_idx == 0


def test_select_best_scale_returns_none_when_all_filtered_out() -> None:
    """Selector should return None when no scale passes quality/artifact filters."""
    results = [
        _fake_scale_result(mass=0.60, r2=0.9, mass_error=0.0, n_valid_windows=5, aic=1.0),
        _fake_scale_result(
            mass=0.58, r2=0.91, mass_error=float("inf"), n_valid_windows=6, aic=1.2
        ),
        _fake_scale_result(mass=0.0, r2=0.95, mass_error=0.1, n_valid_windows=7, aic=0.8),
    ]

    best_idx = _select_best_scale(
        results,
        min_r2=0.0,
        min_windows=0,
        max_error_pct=30.0,
        remove_artifacts=True,
    )
    assert best_idx is None


def test_select_best_scale_respects_max_error_pct_filter() -> None:
    """Best-scale selector should skip scales above the max mass-error percentage."""
    results = [
        _fake_scale_result(mass=0.55, r2=0.95, mass_error=0.22, n_valid_windows=7, aic=2.0),
        _fake_scale_result(mass=0.52, r2=0.93, mass_error=0.12, n_valid_windows=7, aic=3.0),
        _fake_scale_result(mass=0.51, r2=0.91, mass_error=0.20, n_valid_windows=7, aic=1.0),
    ]

    best_idx = _select_best_scale(
        results,
        min_r2=0.0,
        min_windows=0,
        max_error_pct=30.0,
        remove_artifacts=False,
    )
    assert best_idx == 1


def test_companion_and_non_companion_multiscale_series_are_both_computed() -> None:
    """Companion channels should be produced alongside base multiscale channels."""
    t_len, n_scales, n_walkers = 5, 4, 8
    data = _build_random_inputs(t_len=t_len, n_scales=n_scales, n_walkers=n_walkers)
    channels = [
        "scalar",
        "pseudoscalar",
        "vector",
        "axial_vector",
        "nucleon",
        "glueball",
        "scalar_companion",
        "scalar_raw_companion",
        "scalar_abs2_vacsub_companion",
        "pseudoscalar_companion",
        "scalar_score_directed_companion",
        "pseudoscalar_score_directed_companion",
        "vector_companion",
        "axial_vector_companion",
        "tensor_companion",
        "nucleon_companion",
        "nucleon_score_signed_companion",
        "nucleon_score_abs_companion",
        "glueball_companion",
        "nucleon_flux_action_companion",
        "nucleon_flux_sin2_companion",
        "nucleon_flux_exp_companion",
        "glueball_phase_action_companion",
        "glueball_phase_sin2_companion",
    ]
    out = _compute_channel_series_from_kernels(
        color=data["color"],
        color_valid=data["color_valid"],
        positions=data["positions"],
        alive=data["alive"],
        force=data["force"],
        kernels=data["kernels"],
        scales=data["scales"],
        pairwise_distances=data["pairwise_distances"],
        companions_distance=data["companions_distance"],
        companions_clone=data["companions_clone"],
        cloning_scores=data["cloning_scores"],
        channels=channels,
    )

    assert set(channels).issubset(out.keys())
    for name in channels:
        assert out[name].shape == (n_scales, t_len)
        assert bool(torch.isfinite(out[name]).all())

    companion_names = [
        "scalar_companion",
        "scalar_raw_companion",
        "scalar_abs2_vacsub_companion",
        "pseudoscalar_companion",
        "scalar_score_directed_companion",
        "pseudoscalar_score_directed_companion",
        "vector_companion",
        "axial_vector_companion",
        "tensor_companion",
        "nucleon_companion",
        "nucleon_score_signed_companion",
        "nucleon_score_abs_companion",
        "glueball_companion",
        "nucleon_flux_action_companion",
        "nucleon_flux_sin2_companion",
        "nucleon_flux_exp_companion",
        "glueball_phase_action_companion",
        "glueball_phase_sin2_companion",
    ]
    assert any(bool(torch.any(out[name].abs() > 1e-7)) for name in companion_names)


def test_companion_channels_zero_out_when_companions_are_invalid() -> None:
    """Invalid companion indices should yield zero-valued companion series."""
    t_len, n_scales, n_walkers = 4, 3, 7
    data = _build_random_inputs(t_len=t_len, n_scales=n_scales, n_walkers=n_walkers)
    bad_comp = torch.full_like(data["companions_distance"], -1)
    channels = [
        "scalar_companion",
        "scalar_raw_companion",
        "scalar_abs2_vacsub_companion",
        "pseudoscalar_companion",
        "scalar_score_directed_companion",
        "pseudoscalar_score_directed_companion",
        "vector_companion",
        "axial_vector_companion",
        "tensor_companion",
        "nucleon_companion",
        "nucleon_score_signed_companion",
        "nucleon_score_abs_companion",
        "glueball_companion",
        "nucleon_flux_action_companion",
        "nucleon_flux_sin2_companion",
        "nucleon_flux_exp_companion",
        "glueball_phase_action_companion",
        "glueball_phase_sin2_companion",
    ]
    out = _compute_channel_series_from_kernels(
        color=data["color"],
        color_valid=data["color_valid"],
        positions=data["positions"],
        alive=data["alive"],
        force=data["force"],
        kernels=data["kernels"],
        scales=data["scales"],
        pairwise_distances=data["pairwise_distances"],
        companions_distance=bad_comp,
        companions_clone=bad_comp,
        cloning_scores=data["cloning_scores"],
        channels=channels,
    )

    for name in channels:
        assert out[name].shape == (n_scales, t_len)
        torch.testing.assert_close(out[name], torch.zeros_like(out[name]), atol=1e-8, rtol=1e-8)


def test_companion_multiscale_full_scale_matches_original_estimators() -> None:
    """At sufficiently large scale, companion multiscale should recover original correlators."""
    t_len, n_walkers = 7, 10
    data = _build_random_inputs(t_len=t_len, n_scales=3, n_walkers=n_walkers)
    color = data["color"]
    color_valid = data["color_valid"]
    positions = data["positions"]
    alive = data["alive"]
    companions_distance = data["companions_distance"]
    companions_clone = data["companions_clone"]
    cloning_scores = data["cloning_scores"]
    pairwise_distances = data["pairwise_distances"]

    d_ij = pairwise_distances.gather(2, companions_distance.clamp(min=0, max=n_walkers - 1).unsqueeze(-1)).squeeze(-1)
    d_ik = pairwise_distances.gather(2, companions_clone.clamp(min=0, max=n_walkers - 1).unsqueeze(-1)).squeeze(-1)
    flat_idx = (
        companions_distance.clamp(min=0, max=n_walkers - 1) * n_walkers
        + companions_clone.clamp(min=0, max=n_walkers - 1)
    )
    d_jk = pairwise_distances.reshape(t_len, n_walkers * n_walkers).gather(1, flat_idx)

    max_radius = float(
        torch.nan_to_num(pairwise_distances, nan=0.0, posinf=0.0, neginf=0.0).max().item() + 1.0
    )
    scales = torch.tensor([max_radius], dtype=torch.float32)
    cfg = CorrelatorConfig(max_lag=4, use_connected=True)

    override = _compute_companion_per_scale_results_preserving_original(
        color=color,
        color_valid=color_valid,
        positions=positions,
        alive=alive,
        cloning_scores=cloning_scores,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        distance_ij=d_ij,
        distance_ik=d_ik,
        distance_jk=d_jk,
        scales=scales,
        channels=[
            "scalar_companion",
            "scalar_raw_companion",
            "scalar_abs2_vacsub_companion",
            "pseudoscalar_companion",
            "scalar_score_directed_companion",
            "pseudoscalar_score_directed_companion",
            "vector_companion",
            "axial_vector_companion",
            "nucleon_companion",
            "nucleon_score_signed_companion",
            "nucleon_score_abs_companion",
            "glueball_companion",
            "nucleon_flux_action_companion",
            "nucleon_flux_sin2_companion",
            "nucleon_flux_exp_companion",
            "glueball_phase_action_companion",
            "glueball_phase_sin2_companion",
        ],
        dt=1.0,
        config=cfg,
    )

    meson = compute_meson_phase_correlator_from_color(
        color=color[..., :3],
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=4,
        use_connected=True,
        pair_selection="both",
        eps=1e-12,
    )
    meson_score_directed = compute_meson_phase_correlator_from_color(
        color=color[..., :3],
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=4,
        use_connected=True,
        pair_selection="both",
        eps=1e-12,
        operator_mode="score_directed",
        scores=cloning_scores,
    )
    meson_abs2_vacsub = compute_meson_phase_correlator_from_color(
        color=color[..., :3],
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=4,
        use_connected=True,
        pair_selection="both",
        eps=1e-12,
        operator_mode="abs2_vacsub",
    )
    vector = compute_vector_meson_correlator_from_color_positions(
        color=color[..., :3],
        color_valid=color_valid,
        positions=positions[..., :3],
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=4,
        use_connected=True,
        pair_selection="both",
        eps=1e-12,
        use_unit_displacement=False,
    )
    baryon = compute_baryon_correlator_from_color(
        color=color[..., :3],
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=4,
        use_connected=True,
        eps=1e-12,
    )
    baryon_score_signed = compute_baryon_correlator_from_color(
        color=color[..., :3],
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=4,
        use_connected=True,
        eps=1e-12,
        operator_mode="score_signed",
        scores=cloning_scores,
    )
    baryon_score_abs = compute_baryon_correlator_from_color(
        color=color[..., :3],
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=4,
        use_connected=True,
        eps=1e-12,
        operator_mode="score_abs",
        scores=cloning_scores,
    )
    baryon_flux_action = compute_baryon_correlator_from_color(
        color=color[..., :3],
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=4,
        use_connected=True,
        eps=1e-12,
        operator_mode="flux_action",
    )
    baryon_flux_sin2 = compute_baryon_correlator_from_color(
        color=color[..., :3],
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=4,
        use_connected=True,
        eps=1e-12,
        operator_mode="flux_sin2",
    )
    baryon_flux_exp = compute_baryon_correlator_from_color(
        color=color[..., :3],
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=4,
        use_connected=True,
        eps=1e-12,
        operator_mode="flux_exp",
        flux_exp_alpha=1.0,
    )
    glueball = compute_glueball_color_correlator_from_color(
        color=color[..., :3],
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=4,
        use_connected=True,
        eps=1e-12,
        use_action_form=False,
    )
    glueball_phase_action = compute_glueball_color_correlator_from_color(
        color=color[..., :3],
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=4,
        use_connected=True,
        eps=1e-12,
        operator_mode="phase_action",
        use_action_form=False,
    )
    glueball_phase_sin2 = compute_glueball_color_correlator_from_color(
        color=color[..., :3],
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=4,
        use_connected=True,
        eps=1e-12,
        operator_mode="phase_sin2",
        use_action_form=False,
    )

    torch.testing.assert_close(
        override["scalar_companion"][0].correlator,
        meson.scalar,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        override["scalar_raw_companion"][0].correlator,
        meson.scalar_raw,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        override["scalar_abs2_vacsub_companion"][0].correlator,
        meson_abs2_vacsub.scalar,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        override["pseudoscalar_companion"][0].correlator,
        meson.pseudoscalar,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        override["scalar_score_directed_companion"][0].correlator,
        meson_score_directed.scalar,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        override["pseudoscalar_score_directed_companion"][0].correlator,
        meson_score_directed.pseudoscalar,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(override["vector_companion"][0].correlator, vector.vector, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        override["axial_vector_companion"][0].correlator,
        vector.axial_vector,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        override["nucleon_companion"][0].correlator,
        baryon.correlator,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        override["nucleon_score_signed_companion"][0].correlator,
        baryon_score_signed.correlator,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        override["nucleon_score_abs_companion"][0].correlator,
        baryon_score_abs.correlator,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        override["glueball_companion"][0].correlator,
        glueball.correlator,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        override["nucleon_flux_action_companion"][0].correlator,
        baryon_flux_action.correlator,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        override["nucleon_flux_sin2_companion"][0].correlator,
        baryon_flux_sin2.correlator,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        override["nucleon_flux_exp_companion"][0].correlator,
        baryon_flux_exp.correlator,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        override["glueball_phase_action_companion"][0].correlator,
        glueball_phase_action.correlator,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        override["glueball_phase_sin2_companion"][0].correlator,
        glueball_phase_sin2.correlator,
        atol=1e-6,
        rtol=1e-5,
    )
