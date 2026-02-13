"""Fractal gas cloning with uniform companion selection and distance-based fitness."""

from dataclasses import dataclass

import torch
from torch import Tensor

from fragile.fractalai.fractalai import (
    asymmetric_rescale,
    l2_norm,
    random_alive_compas,
)


@dataclass
class FractalCloningOperator:
    """Cloning operator using uniform companion selection and distance-based fitness.

    Uses uniform random companion pairing (not distance-based) to calculate fitness
    via L2 distances on observations, then decides cloning probabilistically based
    on virtual reward comparisons.

    Args:
        dist_coef: Distance coefficient for fitness (default 1.0)
        reward_coef: Reward coefficient for fitness (default 1.0)
        use_cumulative_reward: Use cumulative rewards for fitness (default False, uses step rewards)
        device: Device for tensor operations (cpu or cuda)
        eps: Small constant to avoid division by zero (default 1e-8)
    """

    dist_coef: float = 1.0
    reward_coef: float = 1.0
    use_cumulative_reward: bool = False
    device: str = "cpu"
    eps: float = 1e-8

    def __post_init__(self):
        """Initialize tracking attributes."""
        self.last_companions = None
        self.last_will_clone = None
        self.last_virtual_rewards = None
        self.last_fitness_companions = None
        self.last_clone_companions = None

    def calculate_fitness(
        self,
        observations: Tensor,
        cumulative_rewards: Tensor,
        step_rewards: Tensor,
        alive: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Calculate virtual rewards using L2 distance and rewards.

        Args:
            observations: Walker observations [N, ...], flattened for distance computation
            cumulative_rewards: Total accumulated rewards [N]
            step_rewards: Rewards from last step only [N]
            alive: Boolean mask [N], True for alive walkers

        Returns:
            virtual_rewards: Virtual rewards [N]
            companions: Companion indices [N]
        """
        N = observations.shape[0]

        # Select uniform random companions (alive only)
        companions = random_alive_compas(~alive, observations)

        # Compute L2 distance between walker and companion observations
        obs_flat = observations.reshape(N, -1)
        comp_obs_flat = obs_flat[companions]
        distances = l2_norm(obs_flat, comp_obs_flat)

        # Select reward signal based on parameter
        reward_signal = cumulative_rewards if self.use_cumulative_reward else step_rewards

        # Normalize distances and rewards
        distance_norm = asymmetric_rescale(distances)
        reward_norm = asymmetric_rescale(reward_signal)

        # Compute virtual rewards: distance^dist_coef * reward^reward_coef
        virtual_rewards = distance_norm**self.dist_coef * reward_norm**self.reward_coef

        # Store for diagnostics
        self.last_fitness_companions = companions
        self.last_virtual_rewards = virtual_rewards

        return virtual_rewards, companions

    def decide_cloning(self, virtual_rewards: Tensor, alive: Tensor) -> tuple[Tensor, Tensor]:
        """Decide which walkers should clone based on virtual rewards.

        Args:
            virtual_rewards: Virtual rewards [N]
            alive: Boolean mask [N], True for alive walkers

        Returns:
            companions: Companion indices for cloning [N]
            will_clone: Boolean mask [N], True for walkers that will clone
        """
        # Select new uniform random companions for cloning decision
        companions = random_alive_compas(~alive, virtual_rewards)

        # Compute clone probability: (companion_vr - walker_vr) / walker_vr
        vr = virtual_rewards.flatten()
        vr_comp = vr[companions]
        clone_probs = (vr_comp - vr) / torch.where(
            vr > self.eps,
            vr,
            torch.tensor(self.eps, device=vr.device, dtype=vr.dtype),
        )

        # Sample cloning decisions probabilistically
        will_clone = clone_probs > torch.rand_like(clone_probs)

        # Dead walkers always clone
        will_clone = will_clone | ~alive

        # Store for diagnostics
        self.last_clone_companions = companions
        self.last_will_clone = will_clone

        return companions, will_clone

    def apply(
        self,
        observations: Tensor,
        cumulative_rewards: Tensor,
        step_rewards: Tensor,
        alive: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Combined fitness calculation and cloning decision.

        Args:
            observations: Walker observations [N, ...]
            cumulative_rewards: Total accumulated rewards [N]
            step_rewards: Rewards from last step only [N]
            alive: Boolean mask [N], True for alive walkers

        Returns:
            virtual_rewards: Virtual rewards [N]
            companions: Companion indices for cloning [N]
            will_clone: Boolean mask [N], True for walkers that will clone
        """
        # Calculate fitness
        virtual_rewards, _fitness_companions = self.calculate_fitness(
            observations, cumulative_rewards, step_rewards, alive
        )

        # Decide cloning
        clone_companions, will_clone = self.decide_cloning(virtual_rewards, alive)

        # Store combined companions (use clone companions as the final result)
        self.last_companions = clone_companions

        return virtual_rewards, clone_companions, will_clone


def clone_walker_data(data: Tensor, companions: Tensor, will_clone: Tensor) -> Tensor:
    """Clone walker data arrays based on cloning decisions.

    Args:
        data: Walker data array [N, ...] or [N]
        companions: Companion indices [N]
        will_clone: Boolean mask [N], True for walkers that will clone

    Returns:
        Cloned data array with same shape as input
    """
    cloned_data = data.clone()
    cloned_data[will_clone] = data[companions[will_clone]]
    return cloned_data
