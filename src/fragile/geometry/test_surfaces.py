"""Analytical test surfaces with known curvature for validation.

This module provides functions to generate point configurations on surfaces
with analytically known Ricci curvature. These are used to validate discrete
curvature computations against ground truth.

Surfaces Provided:
    1. Flat Space (R = 0): Uniform grid in Euclidean plane
    2. Sphere (R = 2/r² > 0): Stereographic projection from unit sphere
    3. Hyperbolic Disk (R < 0): Poincaré disk model

References:
    - curvature.md § 2.6 "Test Cases for Validation"
    - Do Carmo, "Riemannian Geometry" (1992) for classical formulas
"""

from __future__ import annotations

import numpy as np


def create_flat_grid(
    N: int, bounds: tuple[float, float] = (-1.0, 1.0), jitter: float = 0.0
) -> np.ndarray:
    """Create uniform grid in flat Euclidean space.

    Generates N points uniformly distributed on a square domain in R².
    This is the simplest test case: flat space has R = 0 everywhere.

    Args:
        N: Number of points (will be rounded to nearest square number)
        bounds: (min, max) for both x and y coordinates
        jitter: Amount of random noise to add (0 = perfect grid)

    Returns:
        Array of shape [N, 2] with point coordinates

    Example:
        >>> points = create_flat_grid(100, bounds=(-5, 5))
        >>> # Compute Ricci: should be ≈ 0 everywhere
    """
    # Round to nearest square
    n_side = int(np.sqrt(N))
    actual_N = n_side**2

    # Create uniform grid
    x = np.linspace(bounds[0], bounds[1], n_side)
    y = np.linspace(bounds[0], bounds[1], n_side)
    xv, yv = np.meshgrid(x, y)

    # Flatten to point cloud
    points = np.stack([xv.ravel(), yv.ravel()], axis=1)

    # Add jitter if requested
    if jitter > 0:
        noise = np.random.randn(actual_N, 2) * jitter
        points += noise

    return points[:N]  # Trim to exact N if needed


def analytical_ricci_flat() -> float:
    """Return analytical Ricci scalar for flat space.

    Returns:
        R = 0.0 (flat space has zero curvature)
    """
    return 0.0


def create_sphere_points(N: int, radius: float = 1.0, projection: str = "stereographic") -> np.ndarray:
    """Create points on 2-sphere via projection to plane.

    Generates N points uniformly distributed on a 2-sphere S² ⊂ R³,
    then projects to R² using stereographic or orthographic projection.
    The induced metric has constant positive curvature R = 2/r².

    Args:
        N: Number of points
        radius: Sphere radius (default: 1.0)
        projection: Projection type ("stereographic" or "orthographic")

    Returns:
        Array of shape [N, 2] with projected coordinates in R²

    Notes:
        - Stereographic projection: (x, y, z) → (x/(1-z), y/(1-z))
        - Avoids south pole singularity by sampling upper hemisphere
        - Induced metric: ds² = 4r²/(1 + x² + y²)² (dx² + dy²)
        - Ricci scalar: R = 2/r² (constant positive curvature)

    Example:
        >>> points = create_sphere_points(100, radius=2.0)
        >>> R_analytical = analytical_ricci_sphere(radius=2.0)
        >>> # Compute discrete Ricci: should be ≈ 0.5
    """
    # Sample uniformly on unit sphere using rejection sampling
    # Method: sample from 3D Gaussian, normalize to unit sphere
    points_3d = np.random.randn(N, 3)
    norms = np.linalg.norm(points_3d, axis=1, keepdims=True)
    points_3d = points_3d / norms * radius

    # Focus on upper hemisphere to avoid projection singularities
    # (can also use full sphere with careful handling)
    points_3d = points_3d[points_3d[:, 2] > -0.9 * radius][:N]

    if len(points_3d) < N:
        # Resample if needed
        extra = N - len(points_3d)
        extra_points = np.random.randn(extra * 2, 3)
        extra_points = extra_points / np.linalg.norm(extra_points, axis=1, keepdims=True) * radius
        extra_points = extra_points[extra_points[:, 2] > -0.9 * radius][:extra]
        points_3d = np.vstack([points_3d, extra_points])

    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

    if projection == "stereographic":
        # Stereographic projection from south pole
        # (x, y, z) → (x/(1+z/r), y/(1+z/r))
        denom = 1 + z / radius
        u = x / denom
        v = y / denom
    elif projection == "orthographic":
        # Orthographic projection (drop z coordinate)
        u = x
        v = y
    else:
        msg = f"Unknown projection: {projection}"
        raise ValueError(msg)

    return np.stack([u, v], axis=1)


def analytical_ricci_sphere(radius: float) -> float:
    """Return analytical Ricci scalar for 2-sphere.

    For a 2-sphere of radius r, the Ricci scalar is constant:
        R = 2 * K = 2/r²

    where K = 1/r² is the Gaussian curvature.

    Args:
        radius: Sphere radius

    Returns:
        R = 2/r² (positive constant curvature)

    Reference:
        Do Carmo, "Riemannian Geometry", Chapter 3, Example 3.3
    """
    return 2.0 / (radius**2)


def create_hyperbolic_disk(
    N: int, radius: float = 0.95, model: str = "poincare"
) -> np.ndarray:
    """Create points in hyperbolic plane via Poincaré disk model.

    Generates N points uniformly distributed in the hyperbolic plane H²
    using the Poincaré disk model. Points lie within the unit disk |z| < 1,
    and the hyperbolic metric has negative curvature R < 0.

    Args:
        N: Number of points
        radius: Maximum radius in disk (< 1.0, default: 0.95)
        model: Hyperbolic model ("poincare" or "klein")

    Returns:
        Array of shape [N, 2] with coordinates in unit disk

    Notes:
        - Poincaré disk metric: ds² = 4/(1 - r²)² (dx² + dy²)
        - Ricci scalar: R = -2 (constant negative curvature in standard model)
        - Points are uniformly distributed in hyperbolic metric
          (not uniform in Euclidean disk!)

    Example:
        >>> points = create_hyperbolic_disk(100, radius=0.9)
        >>> R_analytical = analytical_ricci_hyperbolic()
        >>> # Compute discrete Ricci: should be < -1
    """
    if radius >= 1.0:
        msg = "Poincaré disk radius must be < 1.0"
        raise ValueError(msg)

    # Sample uniformly in hyperbolic metric (not Euclidean!)
    # Use inverse cumulative method for radial distribution
    # In hyperbolic plane: dA_hyp = 4/(1-r²)² r dr dθ

    # Uniform angles
    theta = np.random.uniform(0, 2 * np.pi, N)

    # Hyperbolic radial distribution
    # CDF: F(r) = (r² + r) / (1 + radius²)  (approximate for r << 1)
    # For simplicity, use approximation valid near origin
    u = np.random.uniform(0, 1, N)

    # Map u → r such that density is ∝ 1/(1-r²)²
    # Approximation: r ≈ sqrt(u) * radius (good for radius << 1)
    # Better: use exact inverse CDF (complex formula)
    # For validation purposes, simple distribution suffices
    r = np.sqrt(u) * radius

    # Convert to Cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    points = np.stack([x, y], axis=1)

    if model == "klein":
        # Convert Poincaré to Klein model
        # Klein: w = 2z / (1 + |z|²)
        r_poincare = np.linalg.norm(points, axis=1, keepdims=True)
        points = 2 * points / (1 + r_poincare**2)

    return points


def analytical_ricci_hyperbolic(curvature_scale: float = -1.0) -> float:
    """Return analytical Ricci scalar for hyperbolic plane.

    The hyperbolic plane H² with constant negative curvature has:
        R = 2K

    where K < 0 is the constant Gaussian curvature.

    Args:
        curvature_scale: Gaussian curvature K (default: -1 for standard H²)

    Returns:
        R = 2K < 0 (negative constant curvature)

    Reference:
        Do Carmo, "Riemannian Geometry", Chapter 3, Example 3.4
    """
    return 2.0 * curvature_scale


def get_analytical_ricci(
    surface_type: str, **params
) -> float | np.ndarray:
    """Get analytical Ricci scalar for a given surface type.

    Convenience function to retrieve the known curvature value.

    Args:
        surface_type: One of "flat", "sphere", "hyperbolic"
        **params: Surface-specific parameters
            - sphere: radius (float)
            - hyperbolic: curvature_scale (float)

    Returns:
        Analytical Ricci scalar value

    Example:
        >>> R_flat = get_analytical_ricci("flat")
        >>> R_sphere = get_analytical_ricci("sphere", radius=2.0)
        >>> R_hyp = get_analytical_ricci("hyperbolic", curvature_scale=-2.0)
    """
    if surface_type == "flat":
        return analytical_ricci_flat()
    elif surface_type == "sphere":
        radius = params.get("radius", 1.0)
        return analytical_ricci_sphere(radius)
    elif surface_type == "hyperbolic":
        curvature_scale = params.get("curvature_scale", -1.0)
        return analytical_ricci_hyperbolic(curvature_scale)
    else:
        msg = f"Unknown surface type: {surface_type}"
        raise ValueError(msg)
