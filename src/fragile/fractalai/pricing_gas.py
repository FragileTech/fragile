"""
Pricing Gas: Euclidean Gas adapted for asset pricing with explicit cloning.

This module reuses the Euclidean Gas cloning machinery and companion selection while
replacing the kinetic step with asset-price SDE propagation. The cloning operator
provides variance reduction for rare-event pricing (barriers, deep OTM).
"""

from __future__ import annotations

import math

import param
import torch
from torch import Tensor

from fragile.fractalai.core.cloning import CloneOperator
from fragile.fractalai.core.companion_selection import CompanionSelection
from fragile.fractalai.core.euclidean_gas import EuclideanGas, SwarmState


class PricingSwarmState(SwarmState):
    """Swarm state for pricing with optional path statistics."""

    def __init__(
        self,
        x: Tensor,
        v: Tensor,
        weights: Tensor,
        alive: Tensor,
        running_sum: Tensor | None = None,
        step_index: int = 0,
        log_norm: float = 0.0,
    ) -> None:
        super().__init__(x, v)
        weights = weights.to(device=self.device, dtype=self.dtype)
        if weights.shape[0] != self.N:
            msg = f"Expected {self.N} weights, got {weights.shape[0]}"
            raise ValueError(msg)
        alive = alive.to(device=self.device, dtype=torch.bool)
        if alive.shape[0] != self.N:
            msg = f"Expected {self.N} alive flags, got {alive.shape[0]}"
            raise ValueError(msg)
        if running_sum is not None:
            running_sum = running_sum.to(device=self.device, dtype=self.dtype)
            if running_sum.shape != self.x.shape:
                msg = f"Expected running_sum shape {self.x.shape}, got {running_sum.shape}"
                raise ValueError(msg)
        self.weights = weights
        self.alive = alive
        self.running_sum = running_sum
        self.step_index = int(step_index)
        self.log_norm = float(log_norm)

    def clone(self) -> PricingSwarmState:
        """Deep copy the pricing state."""
        return PricingSwarmState(
            x=self.x.clone(),
            v=self.v.clone(),
            weights=self.weights.clone(),
            alive=self.alive.clone(),
            running_sum=self.running_sum.clone() if self.running_sum is not None else None,
            step_index=self.step_index,
            log_norm=self.log_norm,
        )

    def copy_from(self, other: PricingSwarmState, mask: Tensor) -> None:
        """Copy full pricing state from another swarm for masked walkers."""
        super().copy_from(other, mask)
        if not mask.any():
            return
        indices = torch.where(mask)[0]
        self.weights[indices] = other.weights[indices]
        self.alive[indices] = other.alive[indices]
        if self.running_sum is not None and other.running_sum is not None:
            self.running_sum[indices] = other.running_sum[indices]


class PricingFitness:
    """Minimal fitness operator for pricing-focused cloning."""

    def __init__(
        self,
        mode: str = "weights",
        reward_scale: float = 1.0,
        fitness_floor: float = 1e-12,
    ) -> None:
        self.mode = mode
        self.reward_scale = float(reward_scale)
        self.fitness_floor = float(fitness_floor)

    def __call__(
        self,
        positions: Tensor,
        velocities: Tensor,
        rewards: Tensor,
        alive: Tensor,
        companions: Tensor,
        bounds=None,
        pbc: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        del positions, velocities, companions, bounds, pbc
        if self.mode == "reward":
            fitness = torch.exp(self.reward_scale * rewards)
            fitness = torch.clamp(fitness, min=self.fitness_floor)
        elif self.mode == "linear":
            fitness = rewards
        else:
            fitness = torch.clamp(rewards, min=0.0)
        fitness = torch.where(alive, fitness, torch.zeros_like(fitness))
        return fitness, {"rewards": rewards}

    def compute_gradient(self, *args, **kwargs) -> Tensor:  # noqa: D401
        """Not implemented for pricing fitness."""
        raise NotImplementedError("PricingFitness does not expose gradients.")

    def compute_hessian(self, *args, **kwargs) -> Tensor:  # noqa: D401
        """Not implemented for pricing fitness."""
        raise NotImplementedError("PricingFitness does not expose Hessians.")


class PricingGas(EuclideanGas):
    """Euclidean Gas adapted for asset pricing with cloning and barrier killing."""

    d = param.Integer(default=1, bounds=(1, None), doc="Asset dimension")
    enable_kinetic = param.Boolean(default=False, doc="Disable Langevin kinetics for pricing")
    s0 = param.Number(default=100.0, bounds=(0, None), doc="Initial spot")
    r = param.Number(default=0.02, doc="Risk-free rate")
    mu = param.Number(default=None, allow_None=True, doc="Drift (defaults to r)")
    sigma = param.Number(default=0.2, bounds=(0, None), doc="Volatility")
    t = param.Number(default=1.0, bounds=(0, None), doc="Time horizon in years")
    steps_per_year = param.Integer(default=252, bounds=(1, None), doc="Time steps per year")
    strike = param.Number(default=100.0, bounds=(0, None), doc="Option strike")
    option_type = param.Selector(default="call", objects=["call", "put"])
    payoff_kind = param.Selector(default="vanilla", objects=["vanilla", "asian-arithmetic"])
    barrier = param.Number(default=None, allow_None=True, doc="Barrier level")
    barrier_type = param.Selector(default="up-and-out", objects=["up-and-out", "down-and-out"])
    reward_mode = param.Selector(default="none", objects=["payoff", "distance", "none"])
    fitness_mode = param.Selector(default="weights", objects=["weights", "reward", "linear"])
    reward_scale = param.Number(default=1.0, bounds=(0, None), doc="Fitness reward scale")
    potential_mode = param.Selector(default="none", objects=["none", "distance"])
    potential_scale = param.Number(default=1.0, bounds=(0, None), doc="Potential scale for G_t")

    def __init__(self, **params):
        super().__init__(**params)
        if self.mu is None:
            self.mu = float(self.r)
        if self.companion_selection is None:
            self.companion_selection = CompanionSelection(method="cloning", epsilon=0.1)
        if self.cloning is None:
            self.cloning = CloneOperator()
        if self.fitness_op is None:
            fitness_mode = "weights" if self.fitness_mode == "weights" else self.fitness_mode
            self.fitness_op = PricingFitness(
                mode=fitness_mode,
                reward_scale=self.reward_scale,
            )
        if self.enable_kinetic:
            msg = "PricingGas uses SDE propagation; set enable_kinetic=False."
            raise ValueError(msg)

    @property
    def n_steps(self) -> int:
        return max(1, int(self.t * self.steps_per_year))

    @property
    def dt(self) -> float:
        return float(self.t) / float(self.n_steps)

    def initialize_state(
        self, x_init: Tensor | None = None, v_init: Tensor | None = None
    ) -> PricingSwarmState:
        N, d = self.N, self.d
        if x_init is None:
            x_init = torch.full((N, d), float(self.s0), device=self.device, dtype=self.torch_dtype)
        if v_init is None:
            v_init = torch.zeros((N, d), device=self.device, dtype=self.torch_dtype)
        weights = torch.ones(N, device=self.device, dtype=self.torch_dtype)
        alive = torch.ones(N, device=self.device, dtype=torch.bool)
        running_sum = None
        if self.payoff_kind == "asian-arithmetic":
            running_sum = x_init.clone()
        return PricingSwarmState(
            x=x_init.to(device=self.device, dtype=self.torch_dtype),
            v=v_init.to(device=self.device, dtype=self.torch_dtype),
            weights=weights,
            alive=alive,
            running_sum=running_sum,
            step_index=0,
            log_norm=0.0,
        )

    def _propagate(self, state: PricingSwarmState) -> Tensor:
        dt = self.dt
        z = torch.randn_like(state.x)
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * math.sqrt(dt) * z
        return state.x * torch.exp(drift + diffusion)

    def _barrier_breached(self, x: Tensor) -> Tensor:
        if self.barrier is None:
            return torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        if self.barrier_type == "up-and-out":
            return torch.any(x >= self.barrier, dim=1)
        if self.barrier_type == "down-and-out":
            return torch.any(x <= self.barrier, dim=1)
        raise ValueError(f"Unknown barrier type: {self.barrier_type}")

    def _underlier(
        self,
        state: PricingSwarmState,
        x: Tensor,
        running_sum: Tensor | None = None,
        sample_count: int | None = None,
    ) -> Tensor:
        if self.payoff_kind == "asian-arithmetic":
            base_sum = running_sum if running_sum is not None else state.running_sum
            if base_sum is None:
                msg = "Asian payoff requested but running_sum is missing."
                raise ValueError(msg)
            count = sample_count if sample_count is not None else (state.step_index + 1)
            avg = base_sum / float(count)
            return avg.mean(dim=1)
        return x.mean(dim=1)

    def _payoff(self, underlier: Tensor) -> Tensor:
        if self.option_type == "call":
            return torch.clamp(underlier - self.strike, min=0.0)
        return torch.clamp(self.strike - underlier, min=0.0)

    def _compute_rewards(self, underlier: Tensor, alive: Tensor) -> Tensor:
        if self.reward_mode == "none":
            rewards = torch.zeros_like(underlier)
        elif self.reward_mode == "distance":
            rewards = -torch.abs(underlier - self.strike)
        else:
            rewards = self._payoff(underlier)
        return torch.where(alive, rewards, torch.zeros_like(rewards))

    def _incremental_potential(self, underlier: Tensor) -> Tensor:
        if self.potential_mode == "none":
            return torch.zeros_like(underlier)
        return self.potential_scale * torch.abs(underlier - self.strike)

    def _weight_update(self, weights: Tensor, potential: Tensor) -> tuple[Tensor, float]:
        weights = weights * torch.exp(-self.r * self.dt - potential * self.dt)
        mean_w = float(torch.mean(weights))
        if mean_w <= 0.0:
            return weights, float("-inf")
        log_norm = math.log(mean_w)
        weights = weights / mean_w
        return weights, log_norm

    def step(
        self, state: PricingSwarmState, return_info: bool = False
    ) -> tuple[PricingSwarmState, PricingSwarmState] | tuple[
        PricingSwarmState, PricingSwarmState, dict
    ]:
        x_next = self._propagate(state)
        running_sum = state.running_sum
        if running_sum is not None:
            running_sum = running_sum + x_next

        breached = self._barrier_breached(x_next)
        alive_mask = state.alive & ~breached
        if not alive_mask.any():
            state_dead = PricingSwarmState(
                x=x_next,
                v=state.v.clone(),
                weights=torch.zeros_like(state.weights),
                alive=alive_mask,
                running_sum=running_sum,
                step_index=state.step_index + 1,
                log_norm=float("-inf"),
            )
            info = {
                "fitness": torch.zeros(state.N, device=self.device),
                "rewards": torch.zeros(state.N, device=self.device),
                "weights": state_dead.weights,
                "log_norm": torch.tensor(state_dead.log_norm, device=self.device),
                "alive_mask": alive_mask,
                "cloning_scores": torch.zeros(state.N, device=self.device),
                "cloning_probs": torch.ones(state.N, device=self.device),
                "will_clone": torch.zeros(state.N, dtype=torch.bool, device=self.device),
                "num_cloned": 0,
            }
            if return_info:
                return state_dead, state_dead, info
            return state_dead, state_dead

        sample_count = state.step_index + 2
        underlier = self._underlier(state, x_next, running_sum, sample_count=sample_count)
        potential = self._incremental_potential(underlier)
        rewards = self._compute_rewards(underlier, alive_mask)

        weights = state.weights * alive_mask.to(dtype=state.weights.dtype)
        weights, log_norm = self._weight_update(weights, potential)
        log_norm = state.log_norm + log_norm

        companions = self.companion_selection(
            x=x_next,
            v=state.v,
            alive_mask=alive_mask,
            bounds=None,
            pbc=False,
        )

        fitness_input = rewards
        if self.fitness_mode == "weights":
            fitness_input = weights
        fitness, fitness_info = self.fitness_op(
            positions=x_next,
            velocities=state.v,
            rewards=fitness_input,
            alive=alive_mask,
            companions=companions,
            bounds=None,
            pbc=False,
        )

        if self.enable_cloning:
            companions_clone = self.companion_selection(
                x=x_next,
                v=state.v,
                alive_mask=alive_mask,
                bounds=None,
                pbc=False,
            )
            clone_kwargs = {"weights": weights}
            if running_sum is not None:
                clone_kwargs["running_sum"] = running_sum
            x_cloned, v_cloned, other_cloned, clone_info = self.cloning(
                positions=x_next,
                velocities=state.v,
                fitness=fitness,
                companions=companions_clone,
                alive=alive_mask,
                **clone_kwargs,
            )
            weights_pre_clone = other_cloned["weights"]
            weights = torch.ones_like(weights_pre_clone)
            if running_sum is not None:
                running_sum = other_cloned["running_sum"]
            alive_next = torch.ones_like(alive_mask)
            state_cloned = PricingSwarmState(
                x=x_cloned,
                v=v_cloned,
                weights=weights,
                alive=alive_next,
                running_sum=running_sum,
                step_index=state.step_index + 1,
                log_norm=log_norm,
            )
            clone_info = {
                **clone_info,
                "alive_mask": alive_mask,
                "weights_pre_clone": weights_pre_clone,
                **fitness_info,
            }
        else:
            state_cloned = PricingSwarmState(
                x=x_next,
                v=state.v.clone(),
                weights=weights,
                alive=alive_mask.clone(),
                running_sum=running_sum,
                step_index=state.step_index + 1,
                log_norm=log_norm,
            )
            clone_info = {
                "cloning_scores": torch.zeros(state.N, device=self.device),
                "cloning_probs": torch.ones(state.N, device=self.device),
                "will_clone": torch.zeros(state.N, dtype=torch.bool, device=self.device),
                "num_cloned": 0,
                "companions": companions,
                "alive_mask": alive_mask,
                "weights_pre_clone": weights,
                **fitness_info,
            }

        if return_info:
            info = {
                "fitness": fitness,
                "rewards": rewards,
                "weights": weights,
                "log_norm": torch.tensor(log_norm, device=self.device),
                **clone_info,
            }
            return state_cloned, state_cloned, info
        return state_cloned, state_cloned

    def estimate_price(self, state: PricingSwarmState) -> float:
        sample_count = state.step_index + 1
        underlier = self._underlier(state, state.x, sample_count=sample_count)
        payoff = self._payoff(underlier)
        price = math.exp(state.log_norm) * float(torch.mean(state.weights * payoff))
        return price

    def run_pricing(self, n_steps: int | None = None) -> tuple[float, PricingSwarmState]:
        if n_steps is None:
            n_steps = self.n_steps
        state = self.initialize_state()
        for _ in range(n_steps):
            state, _ = self.step(state)
            if not state.alive.any():
                break
        return self.estimate_price(state), state
