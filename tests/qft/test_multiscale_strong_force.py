"""Tests for multiscale strong-force channel series computation."""

from __future__ import annotations

import torch

from fragile.fractalai.qft.multiscale_strong_force import _compute_channel_series_from_kernels


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
        "tensor_traceless_companion",
        "nucleon_companion",
        "glueball_companion",
    ]
    out = _compute_channel_series_from_kernels(
        color=data["color"],
        color_valid=data["color_valid"],
        positions=data["positions"],
        alive=data["alive"],
        force=data["force"],
        kernels=data["kernels"],
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
        "tensor_traceless_companion",
        "nucleon_companion",
        "glueball_companion",
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
        "tensor_traceless_companion",
        "nucleon_companion",
        "glueball_companion",
    ]
    out = _compute_channel_series_from_kernels(
        color=data["color"],
        color_valid=data["color_valid"],
        positions=data["positions"],
        alive=data["alive"],
        force=data["force"],
        kernels=data["kernels"],
        companions_distance=bad_comp,
        companions_clone=bad_comp,
        channels=channels,
    )

    for name in channels:
        assert out[name].shape == (n_scales, t_len)
        torch.testing.assert_close(out[name], torch.zeros_like(out[name]), atol=1e-8, rtol=1e-8)
