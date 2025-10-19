"""
Kinetic Operator: Langevin Dynamics with BAOAB Integrator

This module implements the kinetic operator for the Euclidean Gas algorithm,
providing Langevin dynamics integration using the BAOAB scheme.

Mathematical notation:
- gamma (γ): Friction coefficient
- beta (β): Inverse temperature 1/(k_B T)
- delta_t (Δt): Time step size
"""

from __future__ import annotations

from pydantic import BaseModel, Field
import torch
from torch import Tensor


class LangevinParams(BaseModel):
    """Parameters for Langevin dynamics (kinetic operator).

    Mathematical notation:
    - gamma (γ): Friction coefficient
    - beta (β): Inverse temperature 1/(k_B T)
    - delta_t (Δt): Time step size
    """

    model_config = {"arbitrary_types_allowed": True}

    gamma: float = Field(gt=0, description="Friction coefficient (γ)")
    beta: float = Field(gt=0, description="Inverse temperature 1/(k_B T) (β)")
    delta_t: float = Field(gt=0, description="Time step size (Δt)")
    integrator: str = Field("baoab", description="Integration scheme (baoab)")

    def noise_std(self) -> float:
        """Standard deviation for BAOAB noise."""
        return (1.0 - torch.exp(torch.tensor(-2 * self.gamma * self.delta_t))).sqrt().item()


class KineticOperator:
    """Kinetic operator using BAOAB integrator for Langevin dynamics."""

    def __init__(
        self,
        params: LangevinParams,
        potential,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize kinetic operator.

        Args:
            params: Langevin parameters
            potential: Target potential (must have evaluate(x) method)
            device: PyTorch device
            dtype: PyTorch dtype
        """
        self.params = params
        self.potential = potential
        self.device = device
        self.dtype = dtype

        # Precompute BAOAB constants
        self.dt = params.delta_t
        self.gamma = params.gamma
        self.beta = params.beta

        # O-step coefficients
        self.c1 = torch.exp(torch.tensor(-self.gamma * self.dt, dtype=dtype))
        self.c2 = torch.sqrt((1.0 - self.c1**2) / self.beta)  # Noise amplitude

    def apply(self, state):
        """
        Apply BAOAB integrator for one time step.

        B: v → v - (Δt/2) * ∇U(x)
        A: x → x + (Δt/2) * v
        O: v → c1 * v + c2 * ξ, where ξ ~ N(0,I)
        A: x → x + (Δt/2) * v
        B: v → v - (Δt/2) * ∇U(x)

        Args:
            state: Current swarm state (must have .x and .v attributes)

        Returns:
            Updated state after integration
        """
        # Import SwarmState here to avoid circular dependency
        from fragile.euclidean_gas import SwarmState

        x, v = state.x.clone(), state.v.clone()
        N, d = state.N, state.d

        # First B step
        x.requires_grad_(True)
        U = self.potential.evaluate(x)  # [N]
        grad_U = torch.autograd.grad(U.sum(), x, create_graph=False)[0]  # [N, d]
        v -= 0 # self.dt / 2 * grad_U
        x.requires_grad_(False)

        # First A step
        x += self.dt / 2 * v

        # O step (Ornstein-Uhlenbeck)
        ξ = torch.randn(N, d, device=self.device, dtype=self.dtype)
        v = self.c1 * v + self.c2 * ξ

        # Second A step
        x += self.dt / 2 * v

        # Second B step
        x.requires_grad_(True)
        U = self.potential.evaluate(x)  # [N]
        grad_U = torch.autograd.grad(U.sum(), x, create_graph=False)[0]  # [N, d]
        v -= self.dt / 2 * grad_U
        x.requires_grad_(False)

        return SwarmState(x, v)
