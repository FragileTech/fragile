"""Tests for multiscale electroweak directional and walker-type channels."""

from __future__ import annotations

import torch

import fragile.fractalai.qft.multiscale_electroweak as ms_ew


class _HistoryStub:
    def __init__(self, *, n_recorded: int = 7, n_walkers: int = 9, dim: int = 3):
        self.n_recorded = n_recorded
        self.delta_t = 0.1
        self.record_every = 1
        self.pbc = False
        self.bounds = None
        self.params = {"companion_selection_clone": {"epsilon": 0.8}}

        self.x_before_clone = torch.randn(n_recorded, n_walkers, dim, dtype=torch.float32)
        self.v_before_clone = torch.randn(n_recorded, n_walkers, dim, dtype=torch.float32)
        self.fitness = torch.rand(n_recorded - 1, n_walkers, dtype=torch.float32) + 0.1
        self.alive_mask = torch.ones(n_recorded - 1, n_walkers, dtype=torch.bool)
        self.will_clone = torch.rand(n_recorded - 1, n_walkers) > 0.5


def _normalize_kernels(kernels: torch.Tensor) -> torch.Tensor:
    n_walkers = int(kernels.shape[-1])
    eye = torch.eye(n_walkers, device=kernels.device, dtype=kernels.dtype).view(
        1, 1, n_walkers, n_walkers
    )
    kernels = kernels * (1.0 - eye)
    return kernels / kernels.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def test_compute_multiscale_electroweak_directional_and_split_channels(monkeypatch) -> None:
    history = _HistoryStub()
    scales = torch.tensor([0.4, 0.8, 1.2], dtype=torch.float32)

    def _fake_select_scales(*args, **kwargs):
        device = kwargs["device"]
        dtype = kwargs["dtype"]
        return scales.to(device=device, dtype=dtype)

    def _fake_iter_batches(*args, **kwargs):
        frame_ids = list(kwargs["frame_indices"])
        device = kwargs["device"]
        dtype = kwargs["dtype"]
        t_len = len(frame_ids)
        n_scales = int(kwargs["scales"].numel())
        n_walkers = int(history.x_before_clone.shape[1])
        kernels = torch.rand(t_len, n_scales, n_walkers, n_walkers, device=device, dtype=dtype)
        kernels = _normalize_kernels(kernels)
        yield frame_ids, None, kernels, None

    monkeypatch.setattr(ms_ew, "select_interesting_scales_from_history", _fake_select_scales)
    monkeypatch.setattr(ms_ew, "iter_smeared_kernel_batches_from_history", _fake_iter_batches)

    cfg = ms_ew.MultiscaleElectroweakConfig(
        warmup_fraction=0.1,
        end_fraction=1.0,
        h_eff=1.0,
        su2_operator_mode="score_directed",
        enable_walker_type_split=True,
        walker_type_scope="frame_global",
        n_scales=3,
        max_lag=10,
        fit_start=2,
        min_fit_points=2,
    )
    channels = [
        "su2_phase",
        "su2_phase_directed",
        "su2_phase_cloner",
        "su2_phase_resister",
        "su2_phase_persister",
    ]
    out = ms_ew.compute_multiscale_electroweak_channels(history, config=cfg, channels=channels)

    assert out.scales.shape == (3,)
    assert len(out.frame_indices) > 0
    expected_keys = [f"{name}_companion" for name in channels]
    assert set(expected_keys).issubset(out.per_scale_results.keys())
    assert set(expected_keys).issubset(out.series_by_channel.keys())
    for key in expected_keys:
        assert out.series_by_channel[key].shape[0] == 3
        assert out.series_by_channel[key].shape[1] == len(out.frame_indices)
        assert len(out.per_scale_results[key]) == 3


def test_compute_multiscale_electroweak_standard_mode_back_compat(monkeypatch) -> None:
    history = _HistoryStub()
    scales = torch.tensor([0.5, 1.0], dtype=torch.float32)

    def _fake_select_scales(*args, **kwargs):
        return scales.to(device=kwargs["device"], dtype=kwargs["dtype"])

    def _fake_iter_batches(*args, **kwargs):
        frame_ids = list(kwargs["frame_indices"])
        device = kwargs["device"]
        dtype = kwargs["dtype"]
        t_len = len(frame_ids)
        n_scales = int(kwargs["scales"].numel())
        n_walkers = int(history.x_before_clone.shape[1])
        kernels = torch.rand(t_len, n_scales, n_walkers, n_walkers, device=device, dtype=dtype)
        kernels = _normalize_kernels(kernels)
        yield frame_ids, None, kernels, None

    monkeypatch.setattr(ms_ew, "select_interesting_scales_from_history", _fake_select_scales)
    monkeypatch.setattr(ms_ew, "iter_smeared_kernel_batches_from_history", _fake_iter_batches)

    cfg = ms_ew.MultiscaleElectroweakConfig(
        su2_operator_mode="standard",
        enable_walker_type_split=False,
        n_scales=2,
        max_lag=8,
    )
    out = ms_ew.compute_multiscale_electroweak_channels(
        history,
        config=cfg,
        channels=["su2_phase", "su2_doublet"],
    )

    assert "su2_phase_companion" in out.series_by_channel
    assert "su2_doublet_companion" in out.series_by_channel
    assert out.series_by_channel["su2_phase_companion"].shape[0] == 2
    assert out.series_by_channel["su2_doublet_companion"].shape[0] == 2
