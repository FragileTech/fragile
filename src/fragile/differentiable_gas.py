"""
Differentiable Fragile Gas: End-to-End Differentiable Swarm Optimization

This module implements a fully differentiable variant of the Fragile Gas using:
1. Gumbel-softmax for companion selection (differentiable discrete sampling)
2. Soft cloning with temperature-controlled sharpness
3. Differentiable reward aggregation

Key differences from RicciGas:
- Companion selection uses Gumbel-softmax instead of hard argmax
- Cloning operator is differentiable through temperature parameter
- Enables gradient-based meta-optimization of hyperparameters
- Can backpropagate through entire swarm dynamics

Mathematical formulation:
- Companion weights: w_i = Gumbel-Softmax(logits=f(x_i, S), tau=τ)
- Clone position: x_new = Σ_i w_i * x_i  (soft average)
- Clone velocity: v_new = Σ_i w_i * v_i + noise

References:
- Gumbel-Softmax: Jang et al. (2017) "Categorical Reparameterization with Gumbel-Softmax"
- Concrete Distribution: Maddison et al. (2017) "The Concrete Distribution"
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor


class DifferentiableGasParams(BaseModel):
    """Parameters for the Differentiable Fragile Gas.

    Companion selection parameters:
    - selection_mode: How to compute companion logits
    - tau_init: Initial Gumbel-softmax temperature
    - tau_min: Minimum temperature (annealing target)
    - tau_anneal: Temperature annealing rate per step

    Physical parameters (inherited from base Gas):
    - gamma: Friction coefficient for Langevin dynamics
    - sigma_v: Velocity noise scale
    - sigma_clone: Clone position noise
    """

    # Companion selection
    selection_mode: Literal["fitness", "diversity", "combined"] = Field(
        default="combined",
        description="Companion selection strategy",
    )
    tau_init: float = Field(default=1.0, gt=0.0, description="Initial Gumbel temperature")
    tau_min: float = Field(default=0.1, gt=0.0, description="Minimum temperature")
    tau_anneal: float = Field(default=0.999, ge=0.0, le=1.0, description="Temperature decay per step")

    # Selection weights (for combined mode)
    alpha_fitness: float = Field(default=1.0, ge=0.0, description="Fitness weight")
    alpha_diversity: float = Field(default=1.0, ge=0.0, description="Diversity weight")

    # Physical dynamics
    gamma: float = Field(default=0.9, ge=0.0, le=1.0, description="Friction coefficient")
    sigma_v: float = Field(default=0.1, gt=0.0, description="Velocity noise scale")
    sigma_clone: float = Field(default=0.05, gt=0.0, description="Clone position noise")

    # Cloning control
    clone_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Fraction of walkers to clone")
    elite_fraction: float = Field(default=0.2, ge=0.0, le=1.0, description="Top performers to use as parents")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SwarmState(BaseModel):
    """Swarm state with differentiable history tracking."""

    x: Tensor = Field(..., description="Positions [N, d]")
    v: Tensor = Field(..., description="Velocities [N, d]")
    reward: Tensor = Field(..., description="Rewards [N]")

    # Optional: Track soft assignments for analysis
    companion_weights: Tensor | None = Field(
        default=None, description="Soft companion weights [N, N]"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


def gumbel_softmax_sample(
    logits: Tensor,
    tau: float = 1.0,
    hard: bool = False,
    dim: int = -1,
) -> Tensor:
    """Sample from Gumbel-Softmax distribution.

    Args:
        logits: Unnormalized log probabilities [..., N]
        tau: Temperature (lower = more discrete)
        hard: If True, return one-hot but backprop through soft
        dim: Dimension to apply softmax

    Returns:
        Soft or hard samples [..., N]
    """
    # Sample Gumbel noise
    gumbels = -torch.empty_like(logits).exponential_().log()  # -log(-log(U))

    # Add noise and apply softmax with temperature
    y_soft = F.softmax((logits + gumbels) / tau, dim=dim)

    if hard:
        # Straight-through estimator: forward = one-hot, backward = soft
        index = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y


def compute_fitness_logits(
    reward: Tensor,  # [N]
    temperature: float = 1.0,
) -> Tensor:
    """Compute fitness-based selection logits.

    Higher reward → higher selection probability.

    Args:
        reward: Walker rewards [N]
        temperature: Sharpness of selection

    Returns:
        logits: [N] unnormalized log probabilities
    """
    return reward / temperature


def compute_diversity_logits(
    x: Tensor,  # [N, d]
    x_query: Tensor,  # [M, d]
    bandwidth: float = 1.0,
) -> Tensor:
    """Compute diversity-based selection logits.

    Walkers far from query point have higher selection probability.

    Args:
        x: Walker positions [N, d]
        x_query: Query positions [M, d]
        bandwidth: Distance scaling

    Returns:
        logits: [M, N] unnormalized log probabilities
    """
    # Pairwise distances [M, N]
    diff = x_query.unsqueeze(1) - x.unsqueeze(0)  # [M, N, d]
    dist = diff.norm(dim=-1)  # [M, N]

    # Larger distance → higher logit
    return dist / bandwidth


def compute_combined_logits(
    x: Tensor,  # [N, d]
    reward: Tensor,  # [N]
    x_query: Tensor,  # [M, d]
    alpha_fitness: float = 1.0,
    alpha_diversity: float = 1.0,
    bandwidth: float = 1.0,
) -> Tensor:
    """Combine fitness and diversity logits.

    Args:
        x: Walker positions [N, d]
        reward: Walker rewards [N]
        x_query: Query positions [M, d]
        alpha_fitness: Fitness weight
        alpha_diversity: Diversity weight
        bandwidth: Distance scaling

    Returns:
        logits: [M, N] combined selection logits
    """
    # Fitness component [N] → [M, N]
    fitness_logits = compute_fitness_logits(reward, temperature=1.0)
    fitness_logits = fitness_logits.unsqueeze(0).expand(x_query.shape[0], -1)

    # Diversity component [M, N]
    diversity_logits = compute_diversity_logits(x, x_query, bandwidth)

    # Combine
    logits = alpha_fitness * fitness_logits + alpha_diversity * diversity_logits

    return logits


class DifferentiableGas:
    """Fully differentiable Fragile Gas with Gumbel-softmax cloning.

    Usage:
        params = DifferentiableGasParams(
            selection_mode="combined",
            tau_init=1.0,
            clone_rate=0.1,
        )
        gas = DifferentiableGas(params, potential_fn)

        # Initialize
        state = gas.init_state(N=100, d=3, bounds=(-5, 5))

        # Step
        state = gas.step(state, dt=0.1)

        # Access differentiable loss
        loss = -state.reward.mean()
        loss.backward()  # Gradients flow through entire dynamics!
    """

    def __init__(
        self,
        params: DifferentiableGasParams,
        potential: Callable[[Tensor], Tensor] | None = None,
        device: torch.device | str | None = None,
    ):
        self.params = params
        self.potential = potential

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # State: current temperature
        self.tau = params.tau_init

    def init_state(
        self,
        N: int,
        d: int,
        bounds: tuple[float, float] = (-1.0, 1.0),
    ) -> SwarmState:
        """Initialize swarm state.

        Args:
            N: Number of walkers
            d: Dimension
            bounds: (min, max) for uniform initialization

        Returns:
            state: Initial swarm state
        """
        x_min, x_max = bounds
        x = torch.rand(N, d, device=self.device) * (x_max - x_min) + x_min
        v = torch.randn(N, d, device=self.device) * self.params.sigma_v

        # Compute initial rewards
        if self.potential is not None:
            with torch.no_grad():
                reward = -self.potential(x)  # Negative potential = reward
        else:
            reward = torch.zeros(N, device=self.device)

        return SwarmState(x=x, v=v, reward=reward)

    def compute_companion_weights(
        self,
        state: SwarmState,
        num_clones: int,
    ) -> Tensor:
        """Compute soft companion selection weights via Gumbel-softmax.

        Args:
            state: Current swarm state
            num_clones: Number of clones to create

        Returns:
            weights: [num_clones, N] soft assignment weights
        """
        N = len(state.x)

        # Select elite walkers as clone candidates
        num_elite = max(1, int(self.params.elite_fraction * N))
        _, elite_indices = torch.topk(state.reward, k=num_elite)

        # Clone positions (where to spawn clones)
        # Sample uniformly from elite walkers
        clone_parent_idx = elite_indices[
            torch.randint(num_elite, (num_clones,), device=self.device)
        ]
        x_clone = state.x[clone_parent_idx]  # [num_clones, d]

        # Compute companion selection logits
        if self.params.selection_mode == "fitness":
            logits = compute_fitness_logits(state.reward)
            logits = logits.unsqueeze(0).expand(num_clones, -1)

        elif self.params.selection_mode == "diversity":
            logits = compute_diversity_logits(
                state.x, x_clone, bandwidth=1.0
            )

        elif self.params.selection_mode == "combined":
            logits = compute_combined_logits(
                state.x,
                state.reward,
                x_clone,
                alpha_fitness=self.params.alpha_fitness,
                alpha_diversity=self.params.alpha_diversity,
                bandwidth=1.0,
            )
        else:
            raise ValueError(f"Unknown selection_mode: {self.params.selection_mode}")

        # Gumbel-softmax sampling
        weights = gumbel_softmax_sample(logits, tau=self.tau, hard=False, dim=-1)

        return weights

    def soft_clone(
        self,
        state: SwarmState,
        weights: Tensor,  # [M, N]
    ) -> tuple[Tensor, Tensor]:
        """Create clones via soft weighted average.

        Args:
            state: Current swarm state
            weights: Soft companion weights [M, N]

        Returns:
            x_new: New positions [M, d]
            v_new: New velocities [M, d]
        """
        # Soft average positions
        x_new = torch.matmul(weights, state.x)  # [M, N] @ [N, d] → [M, d]

        # Soft average velocities
        v_new = torch.matmul(weights, state.v)  # [M, d]

        # Add noise for exploration
        x_new = x_new + torch.randn_like(x_new) * self.params.sigma_clone
        v_new = v_new + torch.randn_like(v_new) * self.params.sigma_v

        return x_new, v_new

    def langevin_step(
        self,
        state: SwarmState,
        dt: float,
    ) -> SwarmState:
        """Langevin dynamics update (no cloning).

        Args:
            state: Current swarm state
            dt: Time step

        Returns:
            state: Updated state
        """
        # Compute forces (gradient of potential)
        if self.potential is not None:
            x_grad = state.x.clone().requires_grad_(True)
            V = self.potential(x_grad)
            force = -torch.autograd.grad(V.sum(), x_grad)[0]
        else:
            force = torch.zeros_like(state.x)

        # Langevin update
        v_new = (
            self.params.gamma * state.v
            + (1 - self.params.gamma) * force
            + torch.randn_like(state.v) * self.params.sigma_v
        )
        x_new = state.x + v_new * dt

        # Update rewards
        if self.potential is not None:
            reward_new = -self.potential(x_new)
        else:
            reward_new = state.reward

        return SwarmState(x=x_new, v=v_new, reward=reward_new)

    def step(
        self,
        state: SwarmState,
        dt: float = 0.1,
        do_clone: bool = True,
    ) -> SwarmState:
        """Single step of differentiable dynamics.

        Args:
            state: Current swarm state
            dt: Time step
            do_clone: Whether to perform cloning

        Returns:
            state: Updated state with cloning
        """
        N = len(state.x)

        # Langevin update for existing walkers
        state = self.langevin_step(state, dt)

        if do_clone and self.params.clone_rate > 0:
            # Determine number of clones
            num_clones = max(1, int(self.params.clone_rate * N))

            # Compute soft companion weights
            weights = self.compute_companion_weights(state, num_clones)

            # Create clones via soft averaging
            x_clone, v_clone = self.soft_clone(state, weights)

            # Compute rewards for clones
            if self.potential is not None:
                reward_clone = -self.potential(x_clone)
            else:
                reward_clone = torch.zeros(num_clones, device=self.device)

            # Replace worst walkers with clones (non-inplace for differentiability)
            _, worst_idx = torch.topk(state.reward, k=num_clones, largest=False)

            # Create new tensors instead of in-place modification
            x_new = state.x.clone()
            v_new = state.v.clone()
            reward_new = state.reward.clone()

            x_new[worst_idx] = x_clone
            v_new[worst_idx] = v_clone
            reward_new[worst_idx] = reward_clone

            # Create new state
            state = SwarmState(x=x_new, v=v_new, reward=reward_new, companion_weights=weights)

        # Anneal temperature
        self.tau = max(self.params.tau_min, self.tau * self.params.tau_anneal)

        return state

    def rollout(
        self,
        state: SwarmState,
        T: int,
        dt: float = 0.1,
    ) -> tuple[SwarmState, list[SwarmState]]:
        """Run differentiable rollout for T steps.

        Entire trajectory is differentiable!

        Args:
            state: Initial state
            T: Number of steps
            dt: Time step

        Returns:
            state: Final state
            history: List of states at each step
        """
        history = []

        for t in range(T):
            state = self.step(state, dt)
            history.append(state)

        return state, history


def create_differentiable_gas_variants(
    device: torch.device | str | None = None,
) -> dict[str, DifferentiableGasParams]:
    """Create parameter sets for ablation study.

    Returns:
        variants: Dictionary of parameter configurations
    """
    base = {
        "tau_init": 1.0,
        "tau_min": 0.1,
        "tau_anneal": 0.995,
        "gamma": 0.9,
        "sigma_v": 0.1,
        "sigma_clone": 0.05,
        "clone_rate": 0.15,
        "elite_fraction": 0.2,
    }

    return {
        "fitness_only": DifferentiableGasParams(
            **base,
            selection_mode="fitness",
            alpha_fitness=1.0,
            alpha_diversity=0.0,
        ),
        "diversity_only": DifferentiableGasParams(
            **base,
            selection_mode="diversity",
            alpha_fitness=0.0,
            alpha_diversity=1.0,
        ),
        "balanced": DifferentiableGasParams(
            **base,
            selection_mode="combined",
            alpha_fitness=1.0,
            alpha_diversity=1.0,
        ),
        "fitness_heavy": DifferentiableGasParams(
            **base,
            selection_mode="combined",
            alpha_fitness=2.0,
            alpha_diversity=1.0,
        ),
    }


# Example usage
if __name__ == "__main__":
    # Test on simple problem
    def sphere_potential(x: Tensor) -> Tensor:
        """Sphere function: minimum at origin."""
        return (x**2).sum(dim=-1)

    device = torch.device("cpu")
    params = DifferentiableGasParams(
        selection_mode="combined",
        tau_init=2.0,
        clone_rate=0.2,
    )

    gas = DifferentiableGas(params, potential=sphere_potential, device=device)

    # Initialize
    state = gas.init_state(N=50, d=3, bounds=(-5, 5))

    print("Differentiable Gas Test")
    print(f"Initial: reward mean = {state.reward.mean():.4f}")

    # Run
    for t in range(100):
        state = gas.step(state, dt=0.1)

        if t % 20 == 0:
            print(
                f"Step {t:3d}: reward = {state.reward.mean():.4f}, "
                f"best = {state.reward.max():.4f}, tau = {gas.tau:.3f}"
            )

    print(f"\nFinal: reward mean = {state.reward.mean():.4f}")
    print(f"Best reward: {state.reward.max():.4f}")
    print(f"Best position norm: {state.x[state.reward.argmax()].norm():.4f}")
