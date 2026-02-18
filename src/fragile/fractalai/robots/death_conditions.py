"""Custom death conditions for robotic fractal gas."""

import torch
from torch import Tensor

# Walker observation layout (dict order: orientations=14, height=1, velocity=9)
_WALKER_HEIGHT_IDX = 14


def walker_ground_death(state, height_threshold: float = 0.6) -> Tensor:
    """Return boolean mask [N] — True for walkers whose torso height < threshold."""
    return state.observations[:, _WALKER_HEIGHT_IDX] < height_threshold
