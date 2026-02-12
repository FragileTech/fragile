"""History tracking for DM Control Fractal Gas runs."""

from dataclasses import dataclass

import numpy as np


@dataclass
class RoboticHistory:
    """Container for robotic gas algorithm execution history.

    Stores trajectory of metrics and best walker frames for visualization.

    Attributes:
        iterations: List of iteration numbers
        rewards_mean: Mean cumulative reward per iteration
        rewards_max: Max cumulative reward per iteration
        rewards_min: Min cumulative reward per iteration
        alive_counts: Number of alive walkers per iteration
        num_cloned: Number of cloned walkers per iteration
        virtual_rewards_mean: Mean virtual reward per iteration
        virtual_rewards_max: Max virtual reward per iteration
        best_frames: RGB frames of best walker per iteration [iteration][H, W, C]
        best_rewards: Reward of best walker per iteration
        best_indices: Index of best walker per iteration
        N: Number of walkers
        max_iterations: Total iterations run
        task_name: DM Control task name (e.g. "cartpole-balance")
    """

    # Metrics per iteration
    iterations: list[int]
    rewards_mean: list[float]
    rewards_max: list[float]
    rewards_min: list[float]
    alive_counts: list[int]
    num_cloned: list[int]
    virtual_rewards_mean: list[float]
    virtual_rewards_max: list[float]

    # Best walker data per iteration
    best_frames: list[np.ndarray | None]
    best_rewards: list[float]
    best_indices: list[int]

    # Configuration metadata
    N: int
    max_iterations: int
    task_name: str

    @classmethod
    def from_run(
        cls,
        infos: list[dict],
        final_state,
        N: int,
        task_name: str,
    ) -> "RoboticHistory":
        """Construct history from run() output.

        Args:
            infos: List of info dicts from each iteration
            final_state: Final WalkerState
            N: Number of walkers
            task_name: DM Control task name

        Returns:
            RoboticHistory instance
        """
        return cls(
            iterations=list(range(len(infos))),
            rewards_mean=[info["mean_reward"] for info in infos],
            rewards_max=[info["max_reward"] for info in infos],
            rewards_min=[info.get("min_reward", 0.0) for info in infos],
            alive_counts=[info["alive_count"] for info in infos],
            num_cloned=[info["num_cloned"] for info in infos],
            virtual_rewards_mean=[info["mean_virtual_reward"] for info in infos],
            virtual_rewards_max=[info.get("max_virtual_reward", 0.0) for info in infos],
            best_frames=[info.get("best_frame") for info in infos],
            best_rewards=[info["max_reward"] for info in infos],
            best_indices=[info.get("best_walker_idx", 0) for info in infos],
            N=N,
            max_iterations=len(infos),
            task_name=task_name,
        )

    @property
    def has_frames(self) -> bool:
        """Check if frames were recorded."""
        return self.best_frames[0] is not None if self.best_frames else False
