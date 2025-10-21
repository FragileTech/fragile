"""Tests for analytical test surface generators.

This module tests the surface generation functions in fragile.geometry.test_surfaces,
which create point configurations on surfaces with known analytical curvature.

Test Coverage:
    - Flat grid generation and properties
    - Sphere point generation and projection
    - Hyperbolic disk generation
    - Analytical Ricci value functions
    - Surface type dispatcher

References:
    - fragile.geometry.test_surfaces for implementations
    - curvature.md § 2.6 "Test Cases for Validation"
"""

from __future__ import annotations

import numpy as np
import pytest

from fragile.geometry import (
    analytical_ricci_flat,
    analytical_ricci_hyperbolic,
    analytical_ricci_sphere,
    create_flat_grid,
    create_hyperbolic_disk,
    create_sphere_points,
    get_analytical_ricci,
)


class TestFlatGrid:
    """Tests for flat Euclidean grid generation."""

    def test_basic_flat_grid(self):
        """Test basic flat grid properties."""
        N = 100
        positions = create_flat_grid(N, bounds=(-1, 1), jitter=0.0)

        # Check shape
        assert positions.shape == (100, 2), f"Expected shape (100, 2), got {positions.shape}"

        # Check bounds
        assert np.all(positions >= -1.0), "Points should be within lower bound"
        assert np.all(positions <= 1.0), "Points should be within upper bound"

    def test_flat_grid_exact_count(self):
        """Test that grid produces requested number of points."""
        for N in [25, 64, 100]:
            positions = create_flat_grid(N, bounds=(-2, 2))
            assert len(positions) == N, f"Expected {N} points, got {len(positions)}"

    def test_flat_grid_custom_bounds(self):
        """Test grid with custom bounds."""
        N = 64
        bounds = (-5.0, 5.0)
        positions = create_flat_grid(N, bounds=bounds)

        # Check all points within bounds
        assert np.all(positions >= bounds[0])
        assert np.all(positions <= bounds[1])

        # Check that bounds are actually used (some points near edges)
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

        # Should span most of the domain
        assert x_max - x_min > 0.8 * (bounds[1] - bounds[0])
        assert y_max - y_min > 0.8 * (bounds[1] - bounds[0])

    def test_flat_grid_jitter(self):
        """Test that jitter adds randomness to grid."""
        N = 64
        positions_no_jitter = create_flat_grid(N, bounds=(-1, 1), jitter=0.0)

        # Set seed for reproducibility
        np.random.seed(42)
        positions_jitter = create_flat_grid(N, bounds=(-1, 1), jitter=0.1)

        # With jitter, positions should differ
        assert not np.allclose(positions_no_jitter, positions_jitter), \
            "Jittered grid should differ from perfect grid"

        # Jitter should be small compared to domain
        # (points still roughly in grid locations)
        diff = np.linalg.norm(positions_jitter - positions_no_jitter, axis=1)
        assert np.mean(diff) < 0.5, "Jitter should be moderate"

    def test_flat_grid_regularity(self):
        """Test that perfect grid (no jitter) is regular."""
        N = 64  # 8x8 grid
        positions = create_flat_grid(N, bounds=(-4, 4), jitter=0.0)

        # Check that points form regular grid
        # For 8x8 grid in [-4, 4]: spacing = 8/7 ≈ 1.14
        expected_spacing = 8.0 / 7.0

        # Compute pairwise distances
        from scipy.spatial.distance import pdist

        distances = pdist(positions)

        # Nearest neighbor distances should cluster around expected_spacing
        min_dists = []
        for i in range(N):
            dists_i = np.linalg.norm(positions - positions[i:i+1], axis=1)
            dists_i = dists_i[dists_i > 1e-10]  # Exclude self
            if len(dists_i) > 0:
                min_dists.append(np.min(dists_i))

        # Mean nearest neighbor distance should match grid spacing
        mean_nn_dist = np.mean(min_dists)
        assert np.abs(mean_nn_dist - expected_spacing) < 0.2, \
            f"Grid spacing {mean_nn_dist:.4f} differs from expected {expected_spacing:.4f}"


class TestSpherePoints:
    """Tests for sphere point generation and projection."""

    def test_sphere_basic_properties(self):
        """Test basic sphere generation properties."""
        N = 100
        radius = 1.0
        positions = create_sphere_points(N, radius=radius, projection="stereographic")

        # Check shape
        assert positions.shape == (N, 2), f"Expected shape ({N}, 2), got {positions.shape}"

        # Check no NaN or inf
        assert not np.any(np.isnan(positions)), "Should not contain NaN"
        assert not np.any(np.isinf(positions)), "Should not contain inf"

    def test_sphere_custom_radius(self):
        """Test sphere with different radii."""
        N = 100
        for radius in [0.5, 1.0, 2.0, 5.0]:
            positions = create_sphere_points(N, radius=radius)

            # Check that positions are generated successfully
            assert positions.shape == (N, 2)
            assert not np.any(np.isnan(positions))

    def test_sphere_stereographic_projection(self):
        """Test stereographic projection properties."""
        N = 100
        radius = 1.0
        positions = create_sphere_points(N, radius=radius, projection="stereographic")

        # Stereographic projection can produce points far from origin
        # (projection of points near south pole)
        # Just check they're finite
        assert np.all(np.isfinite(positions))

    def test_sphere_orthographic_projection(self):
        """Test orthographic projection properties."""
        N = 100
        radius = 1.0
        positions = create_sphere_points(N, radius=radius, projection="orthographic")

        # Orthographic projection: points should lie within disk of radius r
        distances = np.linalg.norm(positions, axis=1)
        assert np.all(distances <= radius + 0.1), \
            "Orthographic projection should stay within sphere radius"

    def test_sphere_invalid_projection(self):
        """Test that invalid projection raises error."""
        with pytest.raises(ValueError, match="Unknown projection"):
            create_sphere_points(100, radius=1.0, projection="invalid")

    def test_sphere_distribution(self):
        """Test that sphere points are reasonably distributed."""
        N = 200
        positions = create_sphere_points(N, radius=1.0, projection="orthographic")

        # For orthographic projection of upper hemisphere:
        # Points should be somewhat uniformly distributed in disk

        # Check that points cover multiple quadrants
        in_quadrant = {
            "++": np.sum((positions[:, 0] > 0) & (positions[:, 1] > 0)),
            "+-": np.sum((positions[:, 0] > 0) & (positions[:, 1] < 0)),
            "-+": np.sum((positions[:, 0] < 0) & (positions[:, 1] > 0)),
            "--": np.sum((positions[:, 0] < 0) & (positions[:, 1] < 0)),
        }

        # Each quadrant should have some points (at least 10% of total)
        for quadrant, count in in_quadrant.items():
            assert count > 0.1 * N, f"Quadrant {quadrant} has too few points: {count}"


class TestHyperbolicDisk:
    """Tests for hyperbolic disk point generation."""

    def test_hyperbolic_basic_properties(self):
        """Test basic hyperbolic disk properties."""
        N = 100
        radius = 0.95
        positions = create_hyperbolic_disk(N, radius=radius, model="poincare")

        # Check shape
        assert positions.shape == (N, 2)

        # Check all points within disk
        distances = np.linalg.norm(positions, axis=1)
        assert np.all(distances < radius), \
            f"Points should be within radius {radius}, max dist: {distances.max()}"

    def test_hyperbolic_radius_constraint(self):
        """Test that radius must be < 1.0."""
        # Valid radius
        positions = create_hyperbolic_disk(100, radius=0.9)
        assert positions.shape == (100, 2)

        # Invalid radius ≥ 1.0 should raise error
        with pytest.raises(ValueError, match="radius must be < 1.0"):
            create_hyperbolic_disk(100, radius=1.0)

        with pytest.raises(ValueError, match="radius must be < 1.0"):
            create_hyperbolic_disk(100, radius=1.5)

    def test_hyperbolic_poincare_model(self):
        """Test Poincaré disk model properties."""
        N = 100
        radius = 0.9
        positions = create_hyperbolic_disk(N, radius=radius, model="poincare")

        # All points within unit disk
        distances = np.linalg.norm(positions, axis=1)
        assert np.all(distances < 1.0)

        # Should have points at various distances (not all at origin)
        assert distances.min() < 0.2
        assert distances.max() > 0.5 * radius

    def test_hyperbolic_klein_model(self):
        """Test Klein disk model properties."""
        N = 100
        radius = 0.9
        positions = create_hyperbolic_disk(N, radius=radius, model="klein")

        # All points within unit disk (Klein model also uses unit disk)
        distances = np.linalg.norm(positions, axis=1)
        assert np.all(distances < 1.0)

    def test_hyperbolic_distribution(self):
        """Test that hyperbolic points are distributed throughout disk."""
        N = 200
        radius = 0.9
        positions = create_hyperbolic_disk(N, radius=radius)

        # Check radial distribution
        distances = np.linalg.norm(positions, axis=1)

        # Should have points in inner, middle, and outer regions
        inner = np.sum(distances < 0.3)
        middle = np.sum((distances >= 0.3) & (distances < 0.6))
        outer = np.sum(distances >= 0.6)

        assert inner > 0, "Should have points in inner region"
        assert middle > 0, "Should have points in middle region"
        assert outer > 0, "Should have points in outer region"

        # Angular distribution: check all quadrants
        angles = np.arctan2(positions[:, 1], positions[:, 0])
        in_quadrant = {
            "Q1": np.sum((angles >= 0) & (angles < np.pi / 2)),
            "Q2": np.sum((angles >= np.pi / 2) & (angles < np.pi)),
            "Q3": np.sum((angles >= -np.pi) & (angles < -np.pi / 2)),
            "Q4": np.sum((angles >= -np.pi / 2) & (angles < 0)),
        }

        for quadrant, count in in_quadrant.items():
            assert count > 0, f"Quadrant {quadrant} has no points"


class TestAnalyticalRicci:
    """Tests for analytical Ricci scalar functions."""

    def test_flat_ricci_zero(self):
        """Test that flat space has R = 0."""
        R = analytical_ricci_flat()
        assert R == 0.0, f"Flat space should have R = 0, got {R}"

    def test_sphere_ricci_formula(self):
        """Test sphere Ricci formula: R = 2/r²."""
        for radius in [0.5, 1.0, 2.0, 5.0]:
            R = analytical_ricci_sphere(radius)
            expected = 2.0 / (radius**2)
            assert np.abs(R - expected) < 1e-10, \
                f"Sphere R should be {expected}, got {R}"

    def test_sphere_ricci_positive(self):
        """Test that sphere always has positive curvature."""
        for radius in [0.1, 1.0, 10.0]:
            R = analytical_ricci_sphere(radius)
            assert R > 0, f"Sphere curvature should be positive, got {R}"

    def test_hyperbolic_ricci_formula(self):
        """Test hyperbolic Ricci formula: R = 2K."""
        for K in [-2.0, -1.0, -0.5]:
            R = analytical_ricci_hyperbolic(curvature_scale=K)
            expected = 2.0 * K
            assert np.abs(R - expected) < 1e-10, \
                f"Hyperbolic R should be {expected}, got {R}"

    def test_hyperbolic_ricci_negative(self):
        """Test that hyperbolic plane has negative curvature."""
        R = analytical_ricci_hyperbolic()  # Default K = -1
        assert R < 0, f"Hyperbolic curvature should be negative, got {R}"

        # Standard hyperbolic plane: R = -2
        assert np.abs(R - (-2.0)) < 1e-10


class TestGetAnalyticalRicci:
    """Tests for analytical Ricci dispatcher function."""

    def test_get_flat_ricci(self):
        """Test dispatcher for flat surface."""
        R = get_analytical_ricci("flat")
        assert R == 0.0

    def test_get_sphere_ricci(self):
        """Test dispatcher for sphere."""
        R = get_analytical_ricci("sphere", radius=2.0)
        expected = analytical_ricci_sphere(2.0)
        assert R == expected

        # Default radius
        R_default = get_analytical_ricci("sphere")
        assert R_default == analytical_ricci_sphere(1.0)

    def test_get_hyperbolic_ricci(self):
        """Test dispatcher for hyperbolic surface."""
        R = get_analytical_ricci("hyperbolic", curvature_scale=-2.0)
        expected = analytical_ricci_hyperbolic(-2.0)
        assert R == expected

        # Default curvature scale
        R_default = get_analytical_ricci("hyperbolic")
        assert R_default == analytical_ricci_hyperbolic(-1.0)

    def test_get_invalid_surface(self):
        """Test that invalid surface type raises error."""
        with pytest.raises(ValueError, match="Unknown surface type"):
            get_analytical_ricci("invalid_surface")

    def test_get_ricci_all_surfaces(self):
        """Test dispatcher for all surface types."""
        surfaces = {
            "flat": 0.0,
            "sphere": 2.0,  # radius=1
            "hyperbolic": -2.0,  # K=-1
        }

        for surface_type, expected in surfaces.items():
            R = get_analytical_ricci(surface_type)
            assert np.abs(R - expected) < 1e-10, \
                f"Surface {surface_type} should have R={expected}, got {R}"


class TestSurfaceIntegration:
    """Integration tests for surface generators and analytical values."""

    def test_flat_grid_matches_analytical(self):
        """Test that flat grid generator corresponds to R=0 analytical value."""
        positions = create_flat_grid(100)
        R_analytical = analytical_ricci_flat()

        assert R_analytical == 0.0
        assert positions.shape == (100, 2)

    def test_sphere_points_match_analytical(self):
        """Test that sphere generator corresponds to correct analytical R."""
        radius = 3.0
        positions = create_sphere_points(100, radius=radius)
        R_analytical = analytical_ricci_sphere(radius)

        expected = 2.0 / (radius**2)
        assert np.abs(R_analytical - expected) < 1e-10

    def test_hyperbolic_disk_matches_analytical(self):
        """Test that hyperbolic generator corresponds to correct analytical R."""
        positions = create_hyperbolic_disk(100, radius=0.8)
        R_analytical = analytical_ricci_hyperbolic()

        assert R_analytical == -2.0
        assert positions.shape == (100, 2)

    def test_all_surfaces_sign_consistency(self):
        """Test that all surfaces have expected curvature signs."""
        # Flat: R = 0
        assert get_analytical_ricci("flat") == 0.0

        # Sphere: R > 0
        assert get_analytical_ricci("sphere", radius=1.0) > 0

        # Hyperbolic: R < 0
        assert get_analytical_ricci("hyperbolic") < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
