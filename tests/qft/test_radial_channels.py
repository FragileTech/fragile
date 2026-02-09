"""Tests for strict recorded-geometry radial channel computation."""

from __future__ import annotations

import pytest
import torch

from fragile.fractalai.qft.radial_channels import (
    compute_radial_channels,
    RadialChannelConfig,
)
from tests.qft.test_correlator_channels import (
    MockRunHistory,
    MockRunHistoryWithGeometry,
)


def _make_recorded_graph_bidirectional(history: MockRunHistoryWithGeometry) -> None:
    """Mirror recorded edges so every walker has outgoing neighbors."""
    for t, edges in enumerate(history.neighbor_edges):
        reversed_edges = edges[:, [1, 0]]
        history.neighbor_edges[t] = torch.cat([edges, reversed_edges], dim=0)

        geo = history.geodesic_edge_distances[t]
        history.geodesic_edge_distances[t] = torch.cat([geo, geo], dim=0)

        if history.edge_weights is not None and t < len(history.edge_weights):
            ew_dict = history.edge_weights[t]
            for key, values in list(ew_dict.items()):
                ew_dict[key] = torch.cat([values, values], dim=0)


def test_radial_channels_require_recorded_neighbors() -> None:
    """Computation should fail when recorded neighbor graph is missing."""
    history = MockRunHistory(N=30, d=4, n_recorded=20)
    config = RadialChannelConfig(
        time_axis="mc",
        neighbor_method="recorded",
        neighbor_weighting="inv_geodesic_full",
        use_volume_weights=False,
        max_lag=10,
    )

    with pytest.raises(ValueError, match="neighbor_edges"):
        compute_radial_channels(history, config=config, channels=["scalar"])


def test_radial_channels_require_recorded_geodesics() -> None:
    """Computation should fail when recorded edge geodesics are missing."""
    history = MockRunHistoryWithGeometry(N=30, d=4, n_recorded=20)
    history.geodesic_edge_distances = None
    config = RadialChannelConfig(
        time_axis="mc",
        neighbor_method="recorded",
        neighbor_weighting="inv_geodesic_full",
        use_volume_weights=False,
        max_lag=10,
    )

    with pytest.raises(ValueError, match="geodesic_edge_distances"):
        compute_radial_channels(history, config=config, channels=["scalar"])


def test_mc_time_runs_with_recorded_geometry() -> None:
    """MC-time mode should run with recorded neighbors/geodesics."""
    history = MockRunHistoryWithGeometry(N=30, d=4, n_recorded=25)
    _make_recorded_graph_bidirectional(history)
    config = RadialChannelConfig(
        time_axis="mc",
        neighbor_method="recorded",
        neighbor_weighting="inv_geodesic_full",
        use_volume_weights=True,
        max_lag=12,
    )

    bundle = compute_radial_channels(history, config=config, channels=["scalar", "glueball"])
    out = bundle.radial_4d

    assert out.distance_mode == "mc_time"
    assert "scalar" in out.channel_results
    assert "glueball" in out.channel_results
    assert out.channel_results["scalar"].n_samples > 0
    assert out.channel_results["glueball"].n_samples > 0
    assert bundle.radial_3d_avg is None


def test_radial_snapshot_runs_with_recorded_geometry() -> None:
    """Radial snapshot mode should use recorded graph and produce bins."""
    history = MockRunHistoryWithGeometry(N=30, d=4, n_recorded=25)
    _make_recorded_graph_bidirectional(history)
    config = RadialChannelConfig(
        time_axis="radial",
        mc_time_index=10,
        neighbor_method="recorded",
        neighbor_weighting="inv_geodesic_full",
        distance_mode="graph_full",
        use_volume_weights=True,
        drop_axis_average=False,
    )

    bundle = compute_radial_channels(history, config=config, channels=["scalar"])
    out = bundle.radial_4d

    assert out.distance_mode == "graph_full"
    assert out.pair_count > 0
    assert out.bin_centers.size > 0
    assert "scalar" in out.channel_results


def test_mc_time_runs_with_recorded_kernel_modes() -> None:
    """Recorded scutoid weight modes should be accepted in radial analysis."""
    history = MockRunHistoryWithGeometry(N=30, d=4, n_recorded=25)
    _make_recorded_graph_bidirectional(history)
    for ew_dict in history.edge_weights:
        if "riemannian_kernel_volume" not in ew_dict:
            ew_dict["riemannian_kernel_volume"] = ew_dict["riemannian_kernel"].clone()

    for mode in ("riemannian_kernel", "riemannian_kernel_volume"):
        config = RadialChannelConfig(
            time_axis="mc",
            neighbor_method="recorded",
            neighbor_weighting=mode,
            use_volume_weights=True,
            max_lag=12,
        )
        bundle = compute_radial_channels(history, config=config, channels=["scalar"])
        assert "scalar" in bundle.radial_4d.channel_results
        assert bundle.radial_4d.channel_results["scalar"].n_samples > 0
