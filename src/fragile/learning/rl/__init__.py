"""Geometric Dreamer: model-based RL on the Poincare ball."""

from .actor import GeometricActor
from .critic import GeometricCritic
from .config import DreamerConfig
from .replay_buffer import SequenceReplayBuffer
from .returns import compute_lambda_returns
from .reward_head import RewardHead
from .train_dreamer import train

__all__ = [
    "DreamerConfig",
    "GeometricActor",
    "GeometricCritic",
    "RewardHead",
    "SequenceReplayBuffer",
    "compute_lambda_returns",
    "train",
]
