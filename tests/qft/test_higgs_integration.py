"""Integration test for Higgs field observables and plotting."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fragile.fractalai.qft.higgs_observables import (
    compute_higgs_observables,
    HiggsConfig,
    HiggsObservables,
)
from fragile.fractalai.qft.higgs_plotting import build_all_higgs_plots


class MockRunHistory:
    """Minimal mock RunHistory for testing Higgs observables."""

    def __init__(
        self,
        N: int = 50,
        d: int = 4,
        n_recorded: int = 10,
        device: str = "cpu",
    ):
        self.N = N
        self.d = d
        self.n_recorded = n_recorded
        self.record_every = 10
        self.delta_t = 0.01
        self.pbc = False
        self.bounds = None

        # Generate mock trajectory data
        T = n_recorded
        self.x_before_clone = torch.randn(T, N, d, device=device)
        self.x_final = torch.randn(T, N, d, device=device)  # Last frame is [N, d] by default
        self.v_before_clone = torch.randn(T, N, d, device=device)
        self.fitness = torch.abs(torch.randn(T, N, device=device))

        # Info arrays have T-1 entries
        self.alive_mask = torch.ones(T - 1, N, dtype=torch.bool, device=device)
        self.companions_distance = torch.randint(0, N, (T - 1, N), device=device)

        # Create neighbor edges (k-nearest neighbors)
        k = min(5, N - 1)
        edges = []
        positions = self.x_final[0]  # Use first frame for neighbors
        for i in range(N):
            dists = torch.norm(positions - positions[i], dim=1)
            _, indices = torch.topk(dists, k + 1, largest=False)
            neighbors = indices[1:]  # Exclude self
            for j in neighbors:
                edges.append([i, j.item()])

        self.neighbor_edges = torch.tensor(edges, dtype=torch.long, device=device).T


def create_test_history(n_walkers: int = 50, n_dims: int = 4):
    """Create a minimal RunHistory for testing.

    Args:
        n_walkers: Number of walkers
        n_dims: Number of spatial dimensions

    Returns:
        MockRunHistory with minimal required fields
    """
    return MockRunHistory(N=n_walkers, d=n_dims)


def test_higgs_observables_basic():
    """Test basic computation of Higgs observables."""
    history = create_test_history(n_walkers=30, n_dims=4)

    config = HiggsConfig(
        mc_time_index=None,
        h_eff=1.0,
        mu_sq=1.0,
        lambda_higgs=0.5,
        alpha_gravity=0.1,
        compute_curvature=True,
        compute_action=True,
    )

    observables = compute_higgs_observables(
        history,
        config=config,
        scalar_field_source="fitness",
    )

    # Check that all fields are populated
    assert observables.n_walkers == 30
    assert observables.spatial_dims == 4
    assert observables.cell_volumes.shape == (30,)
    assert observables.metric_tensors.shape == (30, 4, 4)
    assert observables.centroid_vectors.shape == (30, 4)
    assert observables.scalar_field.shape == (30,)
    assert observables.edge_index.shape[0] == 2
    assert observables.alive.shape == (30,)

    # Check action components are finite
    assert np.isfinite(observables.kinetic_term)
    assert np.isfinite(observables.potential_term)
    assert np.isfinite(observables.total_action)

    # Check curvature is computed
    assert observables.ricci_scalars is not None
    assert observables.ricci_scalars.shape == (30,)
    assert np.isfinite(observables.volume_variance)


def test_higgs_plotting_basic():
    """Test that plotting functions work without errors."""
    history = create_test_history(n_walkers=30, n_dims=4)

    config = HiggsConfig(
        compute_curvature=True,
        compute_action=True,
    )

    observables = compute_higgs_observables(
        history,
        config=config,
        scalar_field_source="fitness",
    )

    # Get positions from the same frame that was analyzed
    mc_frame = config.mc_time_index if config.mc_time_index is not None else history.n_recorded - 1
    mc_frame = min(mc_frame, history.n_recorded - 1)
    positions = history.x_final[mc_frame].detach().cpu().numpy()

    # Build all plots
    plots = build_all_higgs_plots(
        observables,
        positions=positions,
        spatial_dims=(0, 1),
        metric_component=(0, 0),
    )

    # Check that all expected plots are present
    expected_plots = {
        "action_summary",
        "metric_tensor_heatmap",
        "centroid_vector_field",
        "scalar_field_map",
        "metric_eigenvalues_distribution",
        "geodesic_distance_scatter",
        "ricci_scalar_distribution",
        "volume_vs_curvature_scatter",
    }

    assert set(plots.keys()) == expected_plots

    # Check that plots are not None
    for plot_name, plot_obj in plots.items():
        assert plot_obj is not None, f"Plot {plot_name} is None"


def test_higgs_plotting_different_metric_components():
    """Test plotting with different metric tensor components."""
    history = create_test_history(n_walkers=30, n_dims=4)

    config = HiggsConfig(
        compute_curvature=True,
        compute_action=True,
    )

    observables = compute_higgs_observables(
        history,
        config=config,
        scalar_field_source="fitness",
    )

    # Get positions from the correct frame
    mc_frame = config.mc_time_index if config.mc_time_index is not None else history.n_recorded - 1
    mc_frame = min(mc_frame, history.n_recorded - 1)
    positions = history.x_final[mc_frame].detach().cpu().numpy()

    # Test different components
    for i in range(4):
        for j in range(4):
            plots = build_all_higgs_plots(
                observables,
                positions=positions,
                metric_component=(i, j),
            )
            assert "metric_tensor_heatmap" in plots
            assert plots["metric_tensor_heatmap"] is not None


def test_higgs_plotting_no_curvature():
    """Test plotting when curvature is not computed."""
    history = create_test_history(n_walkers=30, n_dims=4)

    config = HiggsConfig(
        compute_curvature=False,  # Disable curvature
        compute_action=True,
    )

    observables = compute_higgs_observables(
        history,
        config=config,
        scalar_field_source="fitness",
    )

    # Get positions from the correct frame
    mc_frame = config.mc_time_index if config.mc_time_index is not None else history.n_recorded - 1
    mc_frame = min(mc_frame, history.n_recorded - 1)
    positions = history.x_final[mc_frame].detach().cpu().numpy()

    plots = build_all_higgs_plots(
        observables,
        positions=positions,
    )

    # When curvature is disabled, these plots should be missing
    assert "ricci_scalar_distribution" not in plots
    assert "volume_vs_curvature_scatter" not in plots

    # Other plots should still be present
    assert "metric_tensor_heatmap" in plots
    assert "centroid_vector_field" in plots
    assert "scalar_field_map" in plots


def test_metric_tensor_properties():
    """Test that metric tensors have expected mathematical properties."""
    history = create_test_history(n_walkers=50, n_dims=3)

    config = HiggsConfig()

    observables = compute_higgs_observables(
        history,
        config=config,
        scalar_field_source="fitness",
    )

    metric_tensors = observables.metric_tensors.cpu().numpy()

    # Check that metric tensors are symmetric
    for i in range(min(10, observables.n_walkers)):  # Check first 10
        metric = metric_tensors[i]
        # Metric should be symmetric: g_ij = g_ji
        np.testing.assert_allclose(
            metric, metric.T, rtol=1e-5, err_msg=f"Metric tensor {i} is not symmetric"
        )

        # Metric should have positive eigenvalues (positive definite)
        eigenvalues = np.linalg.eigvalsh(metric)
        assert np.all(
            eigenvalues > 0
        ), f"Metric tensor {i} has non-positive eigenvalues: {eigenvalues}"


if __name__ == "__main__":
    # Run tests
    test_higgs_observables_basic()
    print("✓ test_higgs_observables_basic passed")

    test_higgs_plotting_basic()
    print("✓ test_higgs_plotting_basic passed")

    test_higgs_plotting_different_metric_components()
    print("✓ test_higgs_plotting_different_metric_components passed")

    test_higgs_plotting_no_curvature()
    print("✓ test_higgs_plotting_no_curvature passed")

    test_metric_tensor_properties()
    print("✓ test_metric_tensor_properties passed")

    print("\nAll tests passed!")
