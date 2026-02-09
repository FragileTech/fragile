"""Tests for anisotropic edge-channel correlators."""

from __future__ import annotations

import pytest
import torch

from fragile.fractalai.qft.anisotropic_edge_channels import (
    AnisotropicEdgeChannelConfig,
    compute_anisotropic_edge_channels,
)
from tests.qft.test_correlator_channels import (
    MockRunHistory,
    MockRunHistoryWithGeometry,
)


def _make_recorded_graph_bidirectional(history: MockRunHistoryWithGeometry) -> None:
    """Mirror recorded edges so all walkers can have outgoing neighbors."""
    for t, edges in enumerate(history.neighbor_edges):
        reversed_edges = edges[:, [1, 0]]
        history.neighbor_edges[t] = torch.cat([edges, reversed_edges], dim=0)

        if history.geodesic_edge_distances is not None:
            geo = history.geodesic_edge_distances[t]
            history.geodesic_edge_distances[t] = torch.cat([geo, geo], dim=0)

        if history.edge_weights is not None and t < len(history.edge_weights):
            edge_dict = history.edge_weights[t]
            for key, values in list(edge_dict.items()):
                edge_dict[key] = torch.cat([values, values], dim=0)


def test_anisotropic_edges_require_recorded_neighbors() -> None:
    """Computation should fail when recorded neighbor graph is missing."""
    history = MockRunHistory(N=24, d=3, n_recorded=18)
    config = AnisotropicEdgeChannelConfig(edge_weight_mode="uniform")

    with pytest.raises(ValueError, match="neighbor_edges"):
        compute_anisotropic_edge_channels(history, config=config, channels=["scalar"])


def test_anisotropic_edges_require_recorded_edge_weights_for_scutoid_modes() -> None:
    """Recorded scutoid edge modes should raise when edge_weights are not available."""
    history = MockRunHistoryWithGeometry(N=24, d=3, n_recorded=18)
    _make_recorded_graph_bidirectional(history)
    history.edge_weights = None
    config = AnisotropicEdgeChannelConfig(edge_weight_mode="riemannian_kernel")

    with pytest.raises(ValueError, match="edge_weights"):
        compute_anisotropic_edge_channels(history, config=config, channels=["scalar"])


def test_anisotropic_edges_mc_time_runs_with_recorded_geometry() -> None:
    """End-to-end anisotropic edge correlators should compute from recorded data."""
    history = MockRunHistoryWithGeometry(N=28, d=3, n_recorded=24)
    _make_recorded_graph_bidirectional(history)
    config = AnisotropicEdgeChannelConfig(
        edge_weight_mode="riemannian_kernel",
        use_volume_weights=True,
        component_mode="isotropic+axes",
        max_lag=8,
    )

    out = compute_anisotropic_edge_channels(history, config=config, channels=["scalar", "glueball"])

    assert out.n_valid_frames > 0
    assert out.avg_alive_walkers > 0
    assert out.avg_edges > 0
    assert "scalar" in out.channel_results
    assert "glueball" in out.channel_results
    assert "scalar:axis_0" in out.channel_results
    assert "scalar:axis_1" in out.channel_results
    assert "scalar:axis_2" in out.channel_results
    for result in out.channel_results.values():
        assert result.correlator.shape == (config.max_lag + 1,)


def test_anisotropic_edges_nucleon_direct_neighbor_triplets() -> None:
    """Nucleon channel should run from direct-neighbor triplets."""
    history = MockRunHistoryWithGeometry(N=28, d=3, n_recorded=24)
    _make_recorded_graph_bidirectional(history)
    config = AnisotropicEdgeChannelConfig(
        edge_weight_mode="riemannian_kernel",
        use_volume_weights=False,
        component_mode="isotropic+axes",
        nucleon_triplet_mode="direct_neighbors",
        max_lag=8,
    )

    out = compute_anisotropic_edge_channels(history, config=config, channels=["nucleon"])

    assert "nucleon" in out.channel_results
    assert "nucleon:axis_0" in out.channel_results
    assert out.channel_results["nucleon"].correlator.shape == (config.max_lag + 1,)
    assert out.channel_results["nucleon"].n_samples > 0


def test_anisotropic_edges_nucleon_companion_triplets() -> None:
    """Nucleon channel should run from (distance, clone) companion triplets."""
    history = MockRunHistoryWithGeometry(N=28, d=3, n_recorded=24)
    _make_recorded_graph_bidirectional(history)
    config = AnisotropicEdgeChannelConfig(
        edge_weight_mode="uniform",
        use_volume_weights=True,
        component_mode="isotropic+axes",
        nucleon_triplet_mode="companions",
        max_lag=8,
    )

    out = compute_anisotropic_edge_channels(history, config=config, channels=["nucleon"])

    assert "nucleon" in out.channel_results
    assert out.channel_results["nucleon"].correlator.shape == (config.max_lag + 1,)
    assert out.channel_results["nucleon"].n_samples > 0
