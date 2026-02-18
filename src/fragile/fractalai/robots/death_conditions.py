"""Custom death conditions for robotic fractal gas."""

from torch import Tensor

_WALKER_HEIGHT_IDX = 0


def walker_ground_death(observations: Tensor, height_threshold: float = 0.3) -> Tensor:
    """Return boolean mask [N] — True for walkers whose torso height < threshold."""
    return observations[:, _WALKER_HEIGHT_IDX] < height_threshold
