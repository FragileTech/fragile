"""Core components of the Fragile Gas framework."""

from fragile.fractalai.core.cloning import CloneOperator
from fragile.fractalai.core.companion_selection import CompanionSelection
from fragile.fractalai.core.euclidean_gas import EuclideanGas, SwarmState
from fragile.fractalai.core.fitness import FitnessOperator


try:
    from fragile.fractalai.core.fractal_set import FractalSet
except ModuleNotFoundError as exc:
    if exc.name != "networkx":
        raise

    class FractalSet:  # type: ignore[no-redef]
        """Placeholder when optional dependency is missing."""

        def __init__(self, *args, **kwargs):
            msg = "FractalSet requires the 'networkx' package. Install it to use this feature."
            raise ModuleNotFoundError(msg) from exc


from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.core.kinetic_operator import KineticOperator
from fragile.fractalai.core.scutoids import (
    BaseScutoidHistory,
    create_scutoid_history,
    Scutoid,
    ScutoidHistory2D,
    ScutoidHistory3D,
    VoronoiCell,
)
from fragile.fractalai.core.vec_history import VectorizedHistoryRecorder


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
    "VectorizedHistoryRecorder",
    "VoronoiCell",
    "create_scutoid_history",
]
