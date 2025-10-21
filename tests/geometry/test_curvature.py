"""Tests for curvature computation methods.

This module tests the alternative curvature computation methods in
fragile.geometry.curvature, including:
    - Graph Laplacian eigenvalue computation
    - Cheeger consistency checks
    - Ricci method comparison utilities

These methods provide independent validation of the deficit angle
method implemented in scutoids.py.

References:
    - fragile.geometry.curvature for method implementations
    - curvature.md for mathematical foundations
"""

from __future__ import annotations

import numpy as np
import pytest

from fragile.geometry import (
    check_cheeger_consistency,
    compare_ricci_methods,
    compute_graph_laplacian_eigenvalues,
    create_flat_grid,
    create_hyperbolic_disk,
    create_sphere_points,
)


class TestGraphLaplacianEigenvalues:
    """Tests for graph Laplacian spectrum computation."""

    def test_simple_chain_graph(self):
        """Test eigenvalues for simple chain graph.

        Chain graph: 0 -- 1 -- 2 -- 3
        Known eigenvalues can be computed analytically.
        """
        # Build neighbor lists for chain
        neighbors = {
            0: [1],
            1: [0, 2],
            2: [1, 3],
            3: [2],
        }

        # Note: eigsh requires k < N, so we request k=3 for N=4
        eigenvals, eigenvecs = compute_graph_laplacian_eigenvalues(neighbors, k=3)

        # Check basic properties
        assert len(eigenvals) == 3
        assert eigenvals.shape == (3,)
        assert eigenvecs.shape == (4, 3)

        # λ₀ should be zero (constant eigenfunction)
        assert np.abs(eigenvals[0]) < 1e-10, "First eigenvalue should be 0"

        # Eigenvalues should be non-negative
        assert np.all(eigenvals >= -1e-10), "Eigenvalues should be non-negative"

        # Eigenvalues should be ordered
        assert np.all(np.diff(eigenvals) >= -1e-10), "Eigenvalues should be ascending"

    def test_complete_graph(self):
        """Test eigenvalues for complete graph.

        Complete graph K_n has known spectrum:
            λ₀ = 0 (multiplicity 1)
            λᵢ = n (multiplicity n-1)
        """
        N = 5
        # Build complete graph
        neighbors = {i: [j for j in range(N) if j != i] for i in range(N)}

        # Request k=4 (N-1) eigenvalues
        eigenvals, eigenvecs = compute_graph_laplacian_eigenvalues(neighbors, k=4)

        # Check basic properties
        assert len(eigenvals) == 4
        assert np.abs(eigenvals[0]) < 1e-10, "First eigenvalue should be 0"

        # For complete graph, remaining eigenvalues should all equal N
        expected_eig = float(N)
        for i in range(1, 4):
            assert np.abs(eigenvals[i] - expected_eig) < 0.1, (
                f"Complete graph eigenvalue {i} should be {N}"
            )

    def test_cycle_graph(self):
        """Test eigenvalues for cycle graph.

        Cycle graph C_n has known spectrum:
            λₖ = 2(1 - cos(2πk/n)) for k = 0, 1, ..., n-1
        """
        N = 6
        # Build cycle graph
        neighbors = {i: [(i - 1) % N, (i + 1) % N] for i in range(N)}

        # Request k=5 (N-1) eigenvalues
        eigenvals, eigenvecs = compute_graph_laplacian_eigenvalues(neighbors, k=5)

        # Check basic properties
        assert len(eigenvals) == 5
        assert np.abs(eigenvals[0]) < 1e-10, "First eigenvalue should be 0"

        # Second eigenvalue (spectral gap) for cycle
        # λ₁ = 2(1 - cos(2π/6)) = 2(1 - cos(π/3)) = 2(1 - 0.5) = 1
        expected_lambda1 = 2 * (1 - np.cos(2 * np.pi / N))
        assert np.abs(eigenvals[1] - expected_lambda1) < 0.1, (
            f"Cycle spectral gap should be {expected_lambda1:.4f}"
        )

    def test_flat_grid_spectrum(self):
        """Test graph Laplacian on 2D flat grid.

        Flat grid should have small spectral gap (related to diameter).
        """
        # Create flat grid and build neighbor graph from Delaunay
        N = 64
        positions = create_flat_grid(N, bounds=(-3, 3), jitter=0.01)

        # Build neighbor lists via simple distance threshold
        # (simplified - real test should use Delaunay)
        threshold = 1.5  # Grid spacing ~ 0.75 for 8x8 grid in [-3, 3]
        neighbors = {}
        for i in range(N):
            neighbors[i] = []
            for j in range(N):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < threshold:
                        neighbors[i].append(j)

        eigenvals, eigenvecs = compute_graph_laplacian_eigenvalues(neighbors, k=5)

        # Check basic properties
        assert np.abs(eigenvals[0]) < 1e-6, "First eigenvalue should be 0"
        assert eigenvals[1] > 0, "Spectral gap should be positive"

        # For grid, spectral gap should be moderate (not too large, not too small)
        assert 0.01 < eigenvals[1] < 5.0, f"Unexpected spectral gap: {eigenvals[1]}"

    def test_disconnected_graph(self):
        """Test handling of disconnected graph.

        Disconnected graph has multiple zero eigenvalues (one per component).
        """
        # Two separate components: {0,1} and {2,3}
        neighbors = {
            0: [1],
            1: [0],
            2: [3],
            3: [2],
        }

        eigenvals, eigenvecs = compute_graph_laplacian_eigenvalues(neighbors, k=4)

        # Should have TWO zero eigenvalues (two connected components)
        zero_eigs = np.sum(np.abs(eigenvals) < 1e-6)
        assert zero_eigs >= 2, f"Disconnected graph should have ≥2 zero eigenvalues, got {zero_eigs}"

    def test_single_node(self):
        """Test degenerate case: single isolated node."""
        neighbors = {0: []}

        eigenvals, eigenvecs = compute_graph_laplacian_eigenvalues(neighbors, k=1)

        # Single node has eigenvalue 0
        assert len(eigenvals) == 1
        assert np.abs(eigenvals[0]) < 1e-10


class TestCheegerConsistency:
    """Tests for Cheeger inequality consistency checks."""

    def test_positive_curvature_consistency(self):
        """Test that positive Ricci is consistent with positive spectral gap.

        Cheeger: Ric > κ > 0 ⟹ λ₁ ≥ C(κ)
        """
        # Simulate positive curvature (e.g., from sphere)
        ricci_scalars = np.random.uniform(0.1, 0.5, 100)

        # Simulate reasonable spectral gap
        eigenvalues = np.array([0.0, 0.05, 0.12, 0.18, 0.25])

        result = check_cheeger_consistency(ricci_scalars, eigenvalues, verbose=False)

        # Should pass consistency check
        assert result["is_consistent"], f"Positive Ricci should be consistent: {result['warning']}"
        assert result["mean_ricci"] > 0
        assert result["spectral_gap"] > 0

    def test_positive_curvature_small_gap_warning(self):
        """Test warning when positive Ricci but tiny spectral gap.

        This violates Cheeger inequality and suggests computation error.
        """
        # Positive curvature
        ricci_scalars = np.random.uniform(0.1, 0.5, 100)

        # Very small spectral gap (inconsistent!)
        eigenvalues = np.array([0.0, 0.0001, 0.002, 0.005, 0.01])

        result = check_cheeger_consistency(ricci_scalars, eigenvalues, verbose=False)

        # Should fail consistency check
        assert not result["is_consistent"], "Should detect Cheeger violation"
        assert result["warning"] is not None
        assert "Cheeger" in result["warning"]

    def test_negative_curvature_no_constraint(self):
        """Test that negative Ricci doesn't trigger warnings.

        Cheeger inequality only provides lower bound for positive curvature.
        Negative curvature has no spectral gap constraint.
        """
        # Negative curvature (e.g., hyperbolic)
        ricci_scalars = np.random.uniform(-2.0, -0.5, 100)

        # Small spectral gap (OK for negative curvature)
        eigenvalues = np.array([0.0, 0.001, 0.005, 0.01, 0.02])

        result = check_cheeger_consistency(ricci_scalars, eigenvalues, verbose=False)

        # Should pass (no constraint on negative curvature)
        assert result["is_consistent"]
        assert result["mean_ricci"] < 0

    def test_flat_space_consistency(self):
        """Test flat space (R ≈ 0) consistency."""
        # Near-zero curvature
        ricci_scalars = np.random.uniform(-0.01, 0.01, 100)

        # Moderate spectral gap
        eigenvalues = np.array([0.0, 0.02, 0.05, 0.08, 0.12])

        result = check_cheeger_consistency(ricci_scalars, eigenvalues, verbose=False)

        # Should pass (no strong constraint)
        assert result["is_consistent"]
        assert np.abs(result["mean_ricci"]) < 0.02

    def test_nan_handling(self):
        """Test graceful handling of NaN values in Ricci."""
        # Mix of valid and NaN values
        ricci_scalars = np.array([0.1, 0.2, np.nan, 0.15, np.nan, 0.18])
        eigenvalues = np.array([0.0, 0.05, 0.1, 0.15, 0.2])

        result = check_cheeger_consistency(ricci_scalars, eigenvalues, verbose=False)

        # Should filter out NaN and compute on valid values
        assert not np.isnan(result["mean_ricci"])
        assert not np.isnan(result["spectral_gap"])


class TestCompareRicciMethods:
    """Tests for Ricci method comparison utilities."""

    def test_identical_methods_perfect_correlation(self):
        """Test that identical Ricci values give perfect correlation."""
        ricci_1 = np.random.randn(100) * 0.5
        ricci_2 = ricci_1.copy()

        stats = compare_ricci_methods(ricci_1, ricci_2, method_name="identical")

        # Perfect agreement
        assert stats["correlation"] > 0.999, "Identical methods should have correlation ≈ 1"
        assert stats["rmse"] < 1e-10, "Identical methods should have RMSE ≈ 0"
        assert stats["mean_relative_error"] < 1e-10
        assert stats["max_absolute_error"] < 1e-10

    def test_highly_correlated_methods(self):
        """Test that highly correlated methods show good agreement."""
        ricci_1 = np.random.randn(100) * 0.5
        # Add small noise
        ricci_2 = ricci_1 + np.random.randn(100) * 0.05

        stats = compare_ricci_methods(ricci_1, ricci_2, method_name="noisy")

        # Good correlation
        assert stats["correlation"] > 0.9, f"Should have high correlation, got {stats['correlation']}"
        assert stats["rmse"] < 0.1, f"RMSE should be small, got {stats['rmse']}"

    def test_uncorrelated_methods(self):
        """Test that uncorrelated Ricci values show poor agreement."""
        ricci_1 = np.random.randn(100) * 0.5
        ricci_2 = np.random.randn(100) * 0.5  # Independent

        stats = compare_ricci_methods(ricci_1, ricci_2, method_name="independent")

        # Poor correlation
        assert stats["correlation"] < 0.5, "Independent methods should have low correlation"

    def test_scaled_methods(self):
        """Test methods that differ by constant scale factor."""
        ricci_1 = np.random.randn(100) * 0.5
        ricci_2 = ricci_1 * 2.0  # Scaled by 2

        stats = compare_ricci_methods(ricci_1, ricci_2, method_name="scaled")

        # Perfect correlation but large RMSE
        assert stats["correlation"] > 0.999, "Scaled methods should be perfectly correlated"
        assert stats["rmse"] > 0.1, "Scaled methods should have non-zero RMSE"

    def test_nan_filtering(self):
        """Test that NaN values are properly filtered out."""
        ricci_1 = np.array([0.1, 0.2, np.nan, 0.3, 0.4, np.nan])
        ricci_2 = np.array([0.12, 0.18, 0.25, np.nan, 0.38, 0.5])

        stats = compare_ricci_methods(ricci_1, ricci_2, method_name="with_nans")

        # Should only compare valid pairs: indices 0, 1, 4
        assert stats["n_valid"] == 3, f"Should have 3 valid pairs, got {stats['n_valid']}"
        assert not np.isnan(stats["correlation"])
        assert not np.isnan(stats["rmse"])

    def test_all_nan_input(self):
        """Test graceful handling when all values are NaN."""
        ricci_1 = np.array([np.nan, np.nan, np.nan])
        ricci_2 = np.array([np.nan, np.nan, np.nan])

        stats = compare_ricci_methods(ricci_1, ricci_2, method_name="all_nan")

        # Should return NaN for all statistics
        assert stats["n_valid"] == 0
        assert np.isnan(stats["correlation"])
        assert np.isnan(stats["rmse"])

    def test_single_valid_pair(self):
        """Test edge case: only one valid comparison pair."""
        ricci_1 = np.array([0.5, np.nan, np.nan])
        ricci_2 = np.array([0.48, np.nan, np.nan])

        stats = compare_ricci_methods(ricci_1, ricci_2, method_name="single")

        # With only 1 point, correlation is undefined
        assert stats["n_valid"] == 1
        assert np.isnan(stats["correlation"])  # Need ≥2 points for correlation
        assert stats["rmse"] > 0  # But RMSE is defined


class TestCurvatureIntegration:
    """Integration tests using analytical test surfaces."""

    def test_flat_grid_cheeger_consistency(self):
        """Test full pipeline: flat grid → Laplacian → Cheeger check."""
        # Create flat grid
        N = 64
        positions = create_flat_grid(N, bounds=(-3, 3), jitter=0.05)

        # Build neighbor lists (simplified via distance threshold)
        threshold = 1.2
        neighbors = {}
        for i in range(N):
            neighbors[i] = []
            for j in range(N):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < threshold:
                        neighbors[i].append(j)

        # Compute Laplacian eigenvalues
        eigenvals, _ = compute_graph_laplacian_eigenvalues(neighbors, k=5)

        # Simulate Ricci computation (flat space: R ≈ 0)
        ricci_scalars = np.random.uniform(-0.05, 0.05, N)

        # Check consistency
        result = check_cheeger_consistency(ricci_scalars, eigenvals, verbose=False)

        # Should be consistent (flat space, no strong constraints)
        assert result["is_consistent"]

    def test_sphere_points_graph_structure(self):
        """Test that sphere points create connected graph."""
        # Create sphere points
        N = 50
        positions = create_sphere_points(N, radius=1.0)

        # Build neighbor graph using adaptive threshold based on density
        # Estimate typical nearest neighbor distance
        from scipy.spatial.distance import pdist

        all_dists = pdist(positions)
        median_dist = np.median(all_dists)
        # Use 3x median distance to ensure connectivity
        threshold = 3.0 * median_dist

        neighbors = {}
        for i in range(N):
            neighbors[i] = []
            for j in range(N):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < threshold:
                        neighbors[i].append(j)

        # Check all nodes have neighbors (graph is connected)
        for i, nbrs in neighbors.items():
            assert len(nbrs) > 0, f"Node {i} is isolated"

        # Compute eigenvalues
        eigenvals, _ = compute_graph_laplacian_eigenvalues(neighbors, k=5)

        # Should have exactly one zero eigenvalue (connected graph)
        zero_eigs = np.sum(np.abs(eigenvals) < 1e-6)
        assert zero_eigs == 1, f"Connected graph should have 1 zero eigenvalue, got {zero_eigs}"

        # Spectral gap should be positive (positive curvature expected)
        assert eigenvals[1] > 0, "Sphere should have positive spectral gap"

    def test_hyperbolic_disk_graph_structure(self):
        """Test that hyperbolic disk creates valid graph."""
        # Create hyperbolic points
        N = 50
        positions = create_hyperbolic_disk(N, radius=0.8)

        # Build neighbor graph
        threshold = 0.5
        neighbors = {}
        for i in range(N):
            neighbors[i] = []
            for j in range(N):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < threshold:
                        neighbors[i].append(j)

        # Compute eigenvalues
        eigenvals, _ = compute_graph_laplacian_eigenvalues(neighbors, k=5)

        # Should have one zero eigenvalue
        assert np.abs(eigenvals[0]) < 1e-6

        # Spectral gap may be small (negative curvature)
        # Just check it's non-negative
        assert eigenvals[1] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
