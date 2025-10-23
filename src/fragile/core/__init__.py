"""Core components of the Fragile Gas framework."""

from fragile.core.cloning import CloneOperator
from fragile.core.companion_selection import CompanionSelection
from fragile.core.euclidean_gas import EuclideanGas, SwarmState
from fragile.core.fitness import FitnessOperator
from fragile.core.fractal_set import FractalSet
from fragile.core.history import RunHistory
from fragile.core.kinetic_operator import KineticOperator
from fragile.core.scutoids import (
    BaseScutoidHistory,
    create_scutoid_history,
    Scutoid,
    ScutoidHistory2D,
    ScutoidHistory3D,
    VoronoiCell,
)


__all__ = [
    "BaseScutoidHistory",
    "CloneOperator",
    "CompanionSelection",
    "EuclideanGas",
    "FitnessOperator",
    "FractalSet",
    "KineticOperator",
    "RunHistory",
    "Scutoid",
    "ScutoidHistory2D",
    "ScutoidHistory3D",
    "SwarmState",
    "VoronoiCell",
    "create_scutoid_history",
]
