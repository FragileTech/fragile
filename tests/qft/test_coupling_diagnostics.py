"""Tests for fast vectorized coupling diagnostics."""

from __future__ import annotations

import torch

from fragile.fractalai.qft.coupling_diagnostics import (
    compute_coupling_diagnostics,
    CouplingDiagnosticsConfig,
)


class _HistoryStub:
    def __init__(self, *, n_recorded: int = 9, n_walkers: int = 12, dim: int = 3):
        self.n_recorded = n_recorded
        self.n_steps = n_recorded
        self.N = n_walkers
        self.d = dim
        self.record_every = 1
        self.recorded_steps = list(range(n_recorded))
        self.pbc = False
        self.bounds = None

        t = n_recorded - 1
        self.x_before_clone = torch.randn(n_recorded, n_walkers, dim, dtype=torch.float32)
        self.v_before_clone = torch.randn(n_recorded, n_walkers, dim, dtype=torch.float32)
        self.force_viscous = torch.randn(t, n_walkers, dim, dtype=torch.float32)
        self.fitness = torch.rand(t, n_walkers, dtype=torch.float32) + 0.05
        self.alive_mask = torch.rand(t, n_walkers) > 0.1
        self.cloning_scores = torch.randn(t, n_walkers, dtype=torch.float32)

        idx = torch.arange(n_walkers, dtype=torch.long)
        self.companions_distance = ((idx + 1) % n_walkers).view(1, n_walkers).expand(t, -1).clone()
        self.companions_clone = ((idx + 2) % n_walkers).view(1, n_walkers).expand(t, -1).clone()

        src = idx
        dst = (idx + 1) % n_walkers
        edges = torch.stack([src, dst], dim=1)
        self.neighbor_edges = [edges.clone() for _ in range(n_recorded)]

        edge_count = int(edges.shape[0])
        base = torch.linspace(0.1, 1.0, steps=edge_count, dtype=torch.float32)
        self.edge_weights = []
        for _ in range(n_recorded):
            self.edge_weights.append({
                "uniform": torch.ones(edge_count, dtype=torch.float32),
                "inverse_distance": base.clone(),
                "inverse_volume": (0.8 * base + 0.2).clone(),
                "inverse_riemannian_distance": (1.2 * base + 0.1).clone(),
                "inverse_riemannian_volume": (0.7 * base + 0.3).clone(),
                "kernel": (0.6 * base + 0.4).clone(),
                "riemannian_kernel": (0.9 * base + 0.2).clone(),
                "riemannian_kernel_volume": (1.1 * base + 0.15).clone(),
            })


def test_compute_coupling_diagnostics_runs_default() -> None:
    history = _HistoryStub()
    out = compute_coupling_diagnostics(history)

    n_frames = len(out.frame_indices)
    assert n_frames > 0
    assert out.phase_mean.shape[0] == n_frames
    assert out.phase_mean_unwrapped.shape[0] == n_frames
    assert out.phase_concentration.shape[0] == n_frames
    assert out.re_im_asymmetry.shape[0] == n_frames
    assert out.local_phase_coherence.shape[0] == n_frames
    assert out.valid_pair_counts.shape[0] == n_frames
    assert out.valid_walker_counts.shape[0] == n_frames
    assert out.summary["n_frames"] == float(n_frames)


def test_compute_coupling_diagnostics_score_abs_and_dim_projection() -> None:
    history = _HistoryStub()
    cfg = CouplingDiagnosticsConfig(
        warmup_fraction=0.2,
        end_fraction=1.0,
        companion_topology="both",
        pair_weighting="score_abs",
        color_dims=(0, 1),
        eps=1e-12,
    )

    out = compute_coupling_diagnostics(history, config=cfg)

    assert len(out.frame_indices) > 0
    assert torch.isfinite(out.phase_concentration).all()
    assert torch.isfinite(out.re_im_asymmetry).all()
    assert torch.isfinite(out.local_phase_coherence).all()
    assert out.summary["valid_pairs_mean"] >= 0.0
    assert out.summary["valid_walkers_mean"] >= 0.0


def test_compute_coupling_diagnostics_empty_range_returns_empty_output() -> None:
    history = _HistoryStub()
    cfg = CouplingDiagnosticsConfig(warmup_fraction=1.0, end_fraction=1.0)

    out = compute_coupling_diagnostics(history, config=cfg)

    assert out.frame_indices == []
    assert out.phase_mean.numel() == 0
    assert out.phase_mean_unwrapped.numel() == 0
    assert out.valid_pair_counts.numel() == 0
    assert out.summary["n_frames"] == 0.0


def test_compute_coupling_diagnostics_kernel_outputs_present() -> None:
    history = _HistoryStub()
    cfg = CouplingDiagnosticsConfig(
        n_scales=6,
        kernel_scale_frames=3,
        kernel_distance_method="floyd-warshall",
        edge_weight_mode="riemannian_kernel_volume",
    )
    out = compute_coupling_diagnostics(history, config=cfg)

    assert out.scales.numel() > 0
    assert out.scales.shape == out.coherence_by_scale.shape
    assert out.scales.shape == out.phase_spread_by_scale.shape
    assert out.scales.shape == out.screening_connected_by_scale.shape
    assert out.summary["kernel_diagnostics_available"] == 1.0


def test_compute_coupling_diagnostics_kernel_can_be_disabled() -> None:
    history = _HistoryStub()
    cfg = CouplingDiagnosticsConfig(enable_kernel_diagnostics=False)
    out = compute_coupling_diagnostics(history, config=cfg)

    assert out.scales.numel() == 0
    assert out.summary["kernel_diagnostics_available"] == 0.0
