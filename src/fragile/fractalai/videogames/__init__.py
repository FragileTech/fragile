"""Fractal gas implementation for video games (Atari)."""

from fragile.fractalai.videogames.atari import AtariEnv, AtariState
from fragile.fractalai.videogames.atari_gas import AtariFractalGas, WalkerState
from fragile.fractalai.videogames.atari_history import AtariHistory
from fragile.fractalai.videogames.cloning import FractalCloningOperator
from fragile.fractalai.videogames.kinetic import RandomActionOperator

__all__ = [
    "AtariEnv",
    "AtariState",
    "AtariFractalGas",
    "WalkerState",
    "AtariHistory",
    "FractalCloningOperator",
    "RandomActionOperator",
]

# Dashboard imports are optional and only loaded when explicitly requested
# to avoid importing heavy dependencies (panel, holoviews) during normal use
def get_dashboard_components():
    """Import dashboard components on demand.

    Returns:
        tuple: (AtariGasConfigPanel, AtariGasVisualizer, create_app)

    Raises:
        ImportError: If panel or holoviews are not installed
    """
    from fragile.fractalai.videogames.dashboard import (
        AtariGasConfigPanel,
        AtariGasVisualizer,
        create_app,
    )

    return AtariGasConfigPanel, AtariGasVisualizer, create_app
