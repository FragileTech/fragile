"""Theory utilities for FractalAI experiments."""

from fragile.fractalai.theory.qsd_variance import compute_hypocoercive_variance, estimate_edge_budget
from fragile.fractalai.theory.qsd_variance_sweep import (
    create_gaussian_mixture_potential,
    run_single_parameter_experiment,
)

__all__ = [
    "compute_hypocoercive_variance",
    "create_gaussian_mixture_potential",
    "estimate_edge_budget",
    "run_single_parameter_experiment",
]
