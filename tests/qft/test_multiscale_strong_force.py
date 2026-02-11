"""Tests for multiscale strong-force channel series computation."""

from __future__ import annotations

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
    }


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
        "pseudoscalar_companion",
        "vector_companion",
        "axial_vector_companion",
        "tensor_companion",
        "nucleon_companion",
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
        channels=channels,
    )

    assert set(channels).issubset(out.keys())
    for name in channels:
        assert out[name].shape == (n_scales, t_len)
        assert bool(torch.isfinite(out[name]).all())

    companion_names = [
        "scalar_companion",
        "pseudoscalar_companion",
        "vector_companion",
        "axial_vector_companion",
        "tensor_companion",
        "nucleon_companion",
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
        "pseudoscalar_companion",
        "vector_companion",
        "axial_vector_companion",
        "tensor_companion",
        "nucleon_companion",
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
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        distance_ij=d_ij,
        distance_ik=d_ik,
        distance_jk=d_jk,
        scales=scales,
        channels=[
            "scalar_companion",
            "pseudoscalar_companion",
            "vector_companion",
            "axial_vector_companion",
            "nucleon_companion",
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

    torch.testing.assert_close(override["scalar_companion"][0].correlator, meson.scalar, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        override["pseudoscalar_companion"][0].correlator,
        meson.pseudoscalar,
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
