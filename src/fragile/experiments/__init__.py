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
    ConvergenceMetrics,
    ConvergenceExperiment,
    create_multimodal_potential,
)

__all__ = [
    "ConvergenceAnalyzer",
    "ConvergenceMetrics",
    "ConvergenceExperiment",
    "create_multimodal_potential",
]
