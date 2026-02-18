"""Custom death conditions for robotic fractal gas."""

import numpy as np
import torch
from torch import Tensor

_WALKER_HEIGHT_IDX = 0
_RIGHT_LEG_BODY = 3  # right knee in xpos
_LEFT_LEG_BODY = 6   # left knee in xpos


def walker_ground_death(state, height_threshold: float = 0.3, knee_threshold: float = 0.15) -> Tensor:
    """Return boolean mask [N] — True for walkers whose torso height < threshold or both knees on ground."""
    obs = state.observations
    torso_low = obs[:, _WALKER_HEIGHT_IDX] < height_threshold

    # Extract knee z-positions from DMControlState body_zpos
    body_zpos = np.array([s.body_zpos for s in state.states])  # [N, num_bodies]
    right_knee_z = torch.tensor(body_zpos[:, _RIGHT_LEG_BODY], device=obs.device)
    left_knee_z = torch.tensor(body_zpos[:, _LEFT_LEG_BODY], device=obs.device)
    both_knees_down = (right_knee_z < knee_threshold) & (left_knee_z < knee_threshold)

    return torso_low | both_knees_down
