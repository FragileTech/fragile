"""Compatibility wrapper for legacy benchmark imports."""

from fragile.fractalai.core.benchmarks import (
    Easom,
    easom,
    EggHolder,
    eggholder,
    holder_table,
    HolderTable,
    lennard_jones,
    LennardJones,
    Rastrigin,
    rastrigin,
    Rosenbrock,
    rosenbrock,
    Sphere,
    sphere,
    styblinski_tang,
    StyblinskiTang,
)


__all__ = [
    "Easom",
    "EggHolder",
    "HolderTable",
    "LennardJones",
    "Rastrigin",
    "Rosenbrock",
    "Sphere",
    "StyblinskiTang",
    "easom",
    "eggholder",
    "holder_table",
    "lennard_jones",
    "rastrigin",
    "rosenbrock",
    "sphere",
    "styblinski_tang",
]
