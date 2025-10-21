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

from fragile.geometry.curvature import (
    check_cheeger_consistency,
    compare_ricci_methods,
    compute_graph_laplacian_eigenvalues,
)
from fragile.geometry.test_surfaces import (
    analytical_ricci_flat,
    analytical_ricci_hyperbolic,
    analytical_ricci_sphere,
    create_flat_grid,
    create_hyperbolic_disk,
    create_sphere_points,
    get_analytical_ricci,
)

__all__ = [
    # Test surfaces
    "create_flat_grid",
    "create_sphere_points",
    "create_hyperbolic_disk",
    "analytical_ricci_flat",
    "analytical_ricci_sphere",
    "analytical_ricci_hyperbolic",
    "get_analytical_ricci",
    # Curvature methods
    "compute_graph_laplacian_eigenvalues",
    "check_cheeger_consistency",
    "compare_ricci_methods",
]
