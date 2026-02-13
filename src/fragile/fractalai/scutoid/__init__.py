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

from .delaunai import (
    compute_delaunay_scutoid,
    DelaunayScutoidData,
)
from .gradient_estimation import (
    compute_directional_derivative,
    estimate_gradient_finite_difference,
    estimate_gradient_quality_metrics,
)
from .hessian_estimation import (
    compute_emergent_metric,
    estimate_hessian_diagonal_fd,
    estimate_hessian_from_metric,
    estimate_hessian_full_fd,
)
from .ricci import (
    compute_ricci_proxy,
    compute_ricci_proxy_full_metric,
    compute_ricci_tensor_proxy,
    compute_ricci_tensor_proxy_full_metric,
)
from .utils import (
    estimate_optimal_step_size,
    find_axial_neighbors,
    validate_finite_difference_inputs,
)
from .validation import (
    compare_estimation_methods,
    plot_estimation_quality,
    validate_on_synthetic_function,
)
from .weights import (
    compute_edge_weights,
    compute_gaussian_kernel_weights,
    compute_inverse_distance_weights,
    compute_inverse_riemannian_distance_weights,
    compute_inverse_riemannian_volume_weights,
    compute_inverse_volume_weights,
    compute_riemannian_volumes,
    compute_uniform_weights,
)


__all__ = [
    "DelaunayScutoidData",
    # Validation
    "compare_estimation_methods",
    "compute_delaunay_scutoid",
    "compute_directional_derivative",
    "compute_edge_weights",
    "compute_emergent_metric",
    "compute_gaussian_kernel_weights",
    "compute_inverse_distance_weights",
    "compute_inverse_riemannian_distance_weights",
    "compute_inverse_riemannian_volume_weights",
    "compute_inverse_volume_weights",
    "compute_ricci_proxy",
    "compute_ricci_proxy_full_metric",
    "compute_ricci_tensor_proxy",
    "compute_ricci_tensor_proxy_full_metric",
    "compute_riemannian_volumes",
    "compute_uniform_weights",
    # Gradient estimation
    "estimate_gradient_finite_difference",
    "estimate_gradient_quality_metrics",
    # Hessian estimation
    "estimate_hessian_diagonal_fd",
    "estimate_hessian_from_metric",
    "estimate_hessian_full_fd",
    # Utilities
    "estimate_optimal_step_size",
    "find_axial_neighbors",
    "plot_estimation_quality",
    "validate_finite_difference_inputs",
    "validate_on_synthetic_function",
]
