"""Random action kinetic operator using plangym interface."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def gaussian_action_sampler(
    action_space,
    mean: float = 0.0,
    std: float = 0.3,
) -> callable:
    """Create an action sampler that draws from a clipped Gaussian.

    Args:
        action_space: Must expose ``.shape``, ``.minimum``, and ``.maximum``.
        mean: Mean of the Gaussian (default 0.0).
        std: Standard deviation of the Gaussian.

    Returns:
        Callable ``(N: int) -> np.ndarray[N, *action_shape]``.
    """
    low = getattr(action_space, "low", None) or action_space.minimum
    high = getattr(action_space, "high", None) or action_space.maximum
    shape = action_space.shape

    def sampler(N: int) -> np.ndarray:
        actions = np.random.normal(loc=mean, scale=std, size=(N, *shape))
        return np.clip(actions, low, high).astype(np.float64)

    return sampler


@dataclass
class RandomActionOperator:
    """Random action operator using plangym's batch stepping interface.

    Samples random actions from the environment and applies them to walker states
    using the plangym batch stepping API.

    Args:
        env: plangym environment with step_batch method
        dt_range: Range for frame skip values (min, max)
        action_sampler: Optional custom action sampling function
        seed: Random seed for reproducibility
    """

    env: Any
    dt_range: tuple[int, int] = (1, 4)
    action_sampler: Any = None
    seed: int | None = None

    def __post_init__(self):
        """Initialize tracking attributes."""
        self.last_actions = None
        self.last_dt = None
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def sample_actions(self, N: int) -> np.ndarray:
        """Sample N random actions from environment.

        Args:
            N: Number of actions to sample

        Returns:
            Array of actions [N]
        """
        if self.action_sampler is not None:
            return self.action_sampler(N)

        # Try plangym standard API first
        if hasattr(self.env, "sample_action"):
            return np.array([self.env.sample_action() for _ in range(N)])

        # Fall back to gym standard API
        if hasattr(self.env, "action_space"):
            return np.array([self.env.action_space.sample() for _ in range(N)])

        msg = "Environment must have either sample_action() or action_space.sample()"
        raise AttributeError(msg)

    def sample_dt(self, N: int) -> np.ndarray:
        """Sample N frame skip values from dt_range.

        Args:
            N: Number of dt values to sample

        Returns:
            Array of frame skip values [N]
        """
        dt_min, dt_max = self.dt_range
        return np.random.randint(dt_min, dt_max + 1, size=N)

    def apply(
        self,
        states: np.ndarray,
        actions: np.ndarray | None = None,
        dt: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """Apply kinetic step to walker states.

        Args:
            states: Environment states [N], dtype=object
            actions: Actions to apply [N], if None will sample randomly
            dt: Frame skip values [N], if None will sample from dt_range

        Returns:
            new_states: Updated environment states [N]
            observations: Observations from environment [N, ...]
            rewards: Step rewards [N]
            dones: Terminal flags [N]
            truncated: Truncation flags [N]
            infos: Info dictionaries [N]
        """
        N = len(states)

        # Sample actions if not provided
        if actions is None:
            actions = self.sample_actions(N)

        # Sample dt if not provided
        if dt is None:
            dt = self.sample_dt(N)

        # Store for tracking
        self.last_actions = actions
        self.last_dt = dt

        # Step environment batch (return_state=True is required for parallel
        # envs where VectorizedEnv passes return_state=None to workers, which
        # PlanEnv.step_batch interprets as falsy → 5-tuple, but the outer
        # unpack_transitions expects 6-tuple).
        result = self.env.step_batch(states=states, actions=actions, dt=dt, return_state=True)

        # Handle 5-tuple vs 6-tuple return (some envs don't return truncated)
        if len(result) == 5:
            new_states, observations, rewards, dones, infos = result
            truncated = np.zeros(N, dtype=bool)
        else:
            new_states, observations, rewards, dones, truncated, infos = result

        # plangym returns all values as lists; convert to numpy arrays
        # with explicit dtypes for downstream torch.tensor() and physics.set_state()
        if isinstance(observations, list):
            observations = np.array(observations, dtype=np.float64)
        if isinstance(rewards, list):
            rewards = np.array(rewards, dtype=np.float32)
        if isinstance(dones, list):
            dones = np.array(dones, dtype=bool)
        if isinstance(truncated, list):
            truncated = np.array(truncated, dtype=bool)

        return new_states, observations, rewards, dones, truncated, infos
