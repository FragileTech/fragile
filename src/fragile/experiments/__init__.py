"""
Experiment utilities for Fragile Gas algorithms.

This module provides reusable computational logic for experiments,
separated from visualization code. This allows for:
- Easier debugging in terminal
- Code reuse across notebooks
- Faster iteration during development

Note:
    The `prepare_background` function is deprecated. For new code, use
    `fragile.core.benchmarks.prepare_benchmark_for_explorer` which supports
    all benchmark types (not just Mixture of Gaussians).
"""

from fragile.experiments.convergence_analysis import (
    ConvergenceAnalyzer,
    ConvergenceExperiment,
    ConvergenceMetrics,
    create_multimodal_potential,
)
from fragile.experiments.gas_config_dashboard import GasConfig
from fragile.experiments.gas_visualization_dashboard import GasVisualizer
from fragile.experiments.interactive_euclidean_gas import (
    create_dashboard,
    prepare_background,  # Deprecated - use fragile.core.benchmarks.prepare_benchmark_for_explorer
    SwarmExplorer,
)


__all__ = [
    "ConvergenceAnalyzer",
    "ConvergenceExperiment",
    "ConvergenceMetrics",
    "GasConfig",
    "GasVisualizer",
    "SwarmExplorer",
    "create_dashboard",
    "create_multimodal_potential",
    "prepare_background",
]
