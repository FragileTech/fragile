"""Geometric analysis tools for the Fragile Gas framework.

This module provides utilities for computing and validating geometric properties
of walker configurations, including:

- Curvature computation via multiple methods (deficit angles, Hessian, spectral)
- Analytical test surfaces with known curvature (sphere, hyperbolic, flat)
- Cross-validation tools for verifying curvature estimates
- Dimension estimation and intrinsic geometry analysis

The geometry module complements the core scutoid tessellation framework by
providing independent implementations of curvature methods for validation and
cross-checking.

Key Components:
    - curvature: Alternative Ricci scalar computation methods
    - test_surfaces: Analytical geometries with known curvature

References:
    Mathematical foundations in old_docs/source/curvature.md
"""

from __future__ import annotations

from fragile.fractalai.geometry.curvature import (
    check_cheeger_consistency,
    compare_ricci_methods,
    compute_graph_laplacian_eigenvalues,
)
from fragile.fractalai.geometry.test_surfaces import (
    analytical_ricci_flat,
    analytical_ricci_hyperbolic,
    analytical_ricci_sphere,
    create_flat_grid,
    create_hyperbolic_disk,
    create_sphere_points,
    get_analytical_ricci,
)


__all__ = [
    "analytical_ricci_flat",
    "analytical_ricci_hyperbolic",
    "analytical_ricci_sphere",
    "check_cheeger_consistency",
    "compare_ricci_methods",
    # Curvature methods
    "compute_graph_laplacian_eigenvalues",
    # Test surfaces
    "create_flat_grid",
    "create_hyperbolic_disk",
    "create_sphere_points",
    "get_analytical_ricci",
]
