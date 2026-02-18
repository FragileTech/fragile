"""Fractal gas implementation for robotics (DM Control)."""

from fragile.fractalai.robots.robotic_gas import RoboticFractalGas
from fragile.fractalai.robots.robotic_history import RoboticHistory


__all__ = [
    "RoboticFractalGas",
    "RoboticHistory",
]


def get_dashboard_components():
    """Import dashboard components on demand.

    Returns:
        tuple: (RoboticGasConfigPanel, RoboticGasVisualizer, create_app)

    Raises:
        ImportError: If panel or holoviews are not installed
    """
    from fragile.fractalai.robots.dashboard import (
        create_app,
        RoboticGasConfigPanel,
        RoboticGasVisualizer,
    )

    return RoboticGasConfigPanel, RoboticGasVisualizer, create_app
