"""Compatibility wrapper for geometry utilities."""

from fragile.fractalai.geometry import (
    analytical_ricci_flat,
    analytical_ricci_hyperbolic,
    analytical_ricci_sphere,
    check_cheeger_consistency,
    compare_ricci_methods,
    compute_graph_laplacian_eigenvalues,
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
    "compute_graph_laplacian_eigenvalues",
    "create_flat_grid",
    "create_hyperbolic_disk",
    "create_sphere_points",
    "get_analytical_ricci",
]
