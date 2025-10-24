"""
Experiment utilities for Fragile Gas algorithms.

This module provides reusable computational logic for experiments,
separated from visualization code. This allows for:
- Easier debugging in terminal
- Code reuse across notebooks
- Faster iteration during development
"""

from fragile.experiments.convergence import ConvergencePanel
from fragile.experiments.gas_config_dashboard import GasConfig
from fragile.experiments.interactive_euclidean_gas import (
    create_dashboard,
    SwarmExplorer,
)
from fragile.experiments.n_particle_swarm import GasVisualizer
from fragile.experiments.parameter_optimization import ConvergenceBoundsPanel


__all__ = [
    "ConvergenceBoundsPanel",
    "ConvergencePanel",
    "GasConfig",
    "GasVisualizer",
    "SwarmExplorer",
    "create_dashboard",
]
