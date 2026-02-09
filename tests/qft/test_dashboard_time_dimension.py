"""Tests for dashboard time dimension handling."""

from __future__ import annotations

import torch

from fragile.fractalai.qft.dashboard import (
    _compute_anisotropic_edge_bundle,
    _compute_channels_vectorized,
    _compute_electroweak_channels,
    AnisotropicEdgeSettings,
    ChannelSettings,
    ElectroweakSettings,
)


class MockRunHistory:
    """Minimal RunHistory stub for dashboard electroweak tests."""

    def __init__(self, *, N: int = 6, d: int = 3, n_recorded: int = 6):
        self.N = N
        self.d = d
        self.n_recorded = n_recorded
        self.record_every = 1
        self.delta_t = 0.1
        self.pbc = False
        self.bounds = None
        self.params = {}
        self.recorded_steps = list(range(n_recorded))

        T = n_recorded
        self.x_before_clone = torch.randn(T, N, d)
        self.x_final = torch.randn(T, N, d)
        self.v_before_clone = torch.randn(T, N, d)

        self.fitness = torch.randn(T - 1, N)
        self.force_viscous = torch.randn(T - 1, N, d)
        self.alive_mask = torch.ones(T - 1, N, dtype=torch.bool)
        self.companions_distance = torch.randint(0, N, (T - 1, N))
        self.companions_clone = torch.randint(0, N, (T - 1, N))
        self.neighbor_edges = None

        self.pos_squared_differences = None
        self.vel_squared_differences = None

    def get_step_index(self, step: int) -> int:
        if step not in self.recorded_steps:
            raise ValueError(f"Step {step} was not recorded")
        return self.recorded_steps.index(step)


def test_electroweak_time_dimension_t_clamps_to_available_dim():
    history = MockRunHistory(d=3, n_recorded=6)
    settings = ElectroweakSettings()
    settings.time_dimension = "t"

    results = _compute_electroweak_channels(history, settings)

    assert results
    assert "u1_phase" in results


def test_channels_mc_time_dimension_runs_end_to_end():
    history = MockRunHistory(d=4, n_recorded=10)
    settings = ChannelSettings()
    settings.time_dimension = "monte_carlo"
    settings.channel_list = "scalar,pseudoscalar,vector"
    settings.max_lag = 10
    settings.compute_bootstrap_errors = False

    results = _compute_channels_vectorized(history, settings)

    assert set(results.keys()) == {"scalar", "pseudoscalar", "vector"}
    for result in results.values():
        assert result.n_samples > 0
        assert result.correlator.shape == (settings.max_lag + 1,)


def test_anisotropic_edge_bundle_runs_with_recorded_geometry():
    history = MockRunHistory(d=3, n_recorded=10, N=8)
    edges = torch.tensor([[i, i + 1] for i in range(history.N - 1)], dtype=torch.long)
    geodesic = torch.linspace(1.0, float(history.N - 1), steps=history.N - 1)
    history.neighbor_edges = []
    history.geodesic_edge_distances = []
    history.edge_weights = []
    for _ in range(history.n_recorded):
        history.neighbor_edges.append(torch.cat([edges, edges[:, [1, 0]]], dim=0))
        history.geodesic_edge_distances.append(torch.cat([geodesic, geodesic], dim=0))
        kernel = torch.ones(2 * (history.N - 1), dtype=torch.float32) * 0.9
        history.edge_weights.append(
            {
                "riemannian_kernel_volume": kernel.clone(),
                "riemannian_kernel": kernel.clone(),
            }
        )
    history.riemannian_volume_weights = torch.ones(history.n_recorded - 1, history.N)

    settings = AnisotropicEdgeSettings()
    settings.channel_list = "scalar,nucleon,glueball"
    settings.max_lag = 10
    settings.compute_bootstrap_errors = False
    settings.component_mode = "isotropic+axes"
    settings.nucleon_triplet_mode = "direct_neighbors"

    out = _compute_anisotropic_edge_bundle(history, settings)

    assert out.n_valid_frames > 0
    assert "scalar" in out.channel_results
    assert "nucleon" in out.channel_results
    assert "scalar:axis_0" in out.channel_results


def test_anisotropic_edge_bundle_companion_nucleon_triplets():
    history = MockRunHistory(d=3, n_recorded=10, N=8)
    edges = torch.tensor([[i, i + 1] for i in range(history.N - 1)], dtype=torch.long)
    geodesic = torch.linspace(1.0, float(history.N - 1), steps=history.N - 1)
    history.neighbor_edges = []
    history.geodesic_edge_distances = []
    history.edge_weights = []
    for _ in range(history.n_recorded):
        history.neighbor_edges.append(torch.cat([edges, edges[:, [1, 0]]], dim=0))
        history.geodesic_edge_distances.append(torch.cat([geodesic, geodesic], dim=0))
        kernel = torch.ones(2 * (history.N - 1), dtype=torch.float32) * 0.9
        history.edge_weights.append(
            {
                "riemannian_kernel_volume": kernel.clone(),
                "riemannian_kernel": kernel.clone(),
            }
        )
    history.riemannian_volume_weights = torch.ones(history.n_recorded - 1, history.N)

    settings = AnisotropicEdgeSettings()
    settings.channel_list = "nucleon"
    settings.max_lag = 10
    settings.compute_bootstrap_errors = False
    settings.component_mode = "isotropic+axes"
    settings.nucleon_triplet_mode = "companions"
    settings.edge_weight_mode = "uniform"

    out = _compute_anisotropic_edge_bundle(history, settings)

    assert out.n_valid_frames > 0
    assert "nucleon" in out.channel_results
