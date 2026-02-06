"""Scutoid geometry-based gradient and Hessian estimation.

Provides two complementary approaches for estimating derivatives from fitness values:

1. **Finite-difference** (Primary method):
   - Direct second-order finite differences using fitness values
   - Reliable, no equilibrium assumption needed
   - O(N·d²) complexity for full Hessian

2. **Geometric** (Validation method):
   - Extract Hessian from emergent metric: H = g - ε_Σ I
   - Independent curvature measurement
   - Validates equilibrium assumption

Cross-validation between methods provides confidence in estimates.
"""

from .gradient_estimation import (
    estimate_gradient_finite_difference,
    compute_directional_derivative,
    estimate_gradient_quality_metrics,
)

from .hessian_estimation import (
    estimate_hessian_diagonal_fd,
    estimate_hessian_full_fd,
    estimate_hessian_from_metric,
    compute_emergent_metric,
)

from .validation import (
    compare_estimation_methods,
    validate_on_synthetic_function,
    plot_estimation_quality,
)

from .utils import (
    estimate_optimal_step_size,
    find_axial_neighbors,
    validate_finite_difference_inputs,
)

from .weights import (
    compute_edge_weights,
    compute_uniform_weights,
    compute_inverse_distance_weights,
    compute_gaussian_kernel_weights,
    compute_inverse_volume_weights,
    compute_inverse_riemannian_volume_weights,
    compute_inverse_riemannian_distance_weights,
    compute_riemannian_volumes,
)

from .delaunai import (
    DelaunayScutoidData,
    compute_delaunay_scutoid,
)

from .ricci import (
    compute_ricci_proxy,
    compute_ricci_proxy_full_metric,
    compute_ricci_tensor_proxy,
    compute_ricci_tensor_proxy_full_metric,
)

__all__ = [
    # Gradient estimation
    "estimate_gradient_finite_difference",
    "compute_directional_derivative",
    "estimate_gradient_quality_metrics",
    # Hessian estimation
    "estimate_hessian_diagonal_fd",
    "estimate_hessian_full_fd",
    "estimate_hessian_from_metric",
    "compute_emergent_metric",
    # Validation
    "compare_estimation_methods",
    "validate_on_synthetic_function",
    "plot_estimation_quality",
    # Utilities
    "estimate_optimal_step_size",
    "find_axial_neighbors",
    "validate_finite_difference_inputs",
    "compute_edge_weights",
    "compute_uniform_weights",
    "compute_inverse_distance_weights",
    "compute_gaussian_kernel_weights",
    "compute_inverse_volume_weights",
    "compute_inverse_riemannian_volume_weights",
    "compute_inverse_riemannian_distance_weights",
    "compute_riemannian_volumes",
    "DelaunayScutoidData",
    "compute_delaunay_scutoid",
    "compute_ricci_proxy",
    "compute_ricci_proxy_full_metric",
    "compute_ricci_tensor_proxy",
    "compute_ricci_tensor_proxy_full_metric",
]
