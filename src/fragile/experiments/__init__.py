"""
Experiment utilities for Fragile Gas algorithms.

This module provides reusable computational logic for experiments,
separated from visualization code. This allows for:
- Easier debugging in terminal
- Code reuse across notebooks
- Faster iteration during development
"""

from fragile.experiments.convergence_analysis import (
    ConvergenceAnalyzer,
    ConvergenceExperiment,
    ConvergenceMetrics,
    create_multimodal_potential,
)
from fragile.experiments.interactive_euclidean_gas import (
    SwarmExplorer,
    create_dashboard,
    prepare_background,
)


__all__ = [
    "ConvergenceAnalyzer",
    "ConvergenceExperiment",
    "ConvergenceMetrics",
    "create_multimodal_potential",
    "SwarmExplorer",
    "create_dashboard",
    "prepare_background",
]
