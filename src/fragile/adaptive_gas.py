"""
Adaptive Viscous Fluid Model: Advanced Fragile Gas with Feedback Mechanisms

This module implements the Adaptive Viscous Fluid Model from clean_build/source/07_adaptative_gas.md.
It extends the Euclidean Gas with three adaptive mechanisms:

1. **Adaptive Force**: Fitness-driven guidance from mean-field potential
2. **Viscous Force**: Fluid-like velocity coupling between walkers
3. **Regularized Hessian Diffusion**: Anisotropic noise based on fitness landscape curvature

The implementation follows the "stable backbone + adaptive perturbation" philosophy,
ensuring provable stability through proper regularization.
"""

from __future__ import annotations

import warnings

from pydantic import BaseModel, Field
import torch
from torch import Tensor

from fragile.euclidean_gas import (
    EuclideanGas,
    EuclideanGasParams,
    KineticOperator,
    PotentialParams,
    SwarmState,
)


class AdaptiveParams(BaseModel):
    """Parameters for adaptive mechanisms.

    Mathematical notation from 07_adaptative_gas.md:
    - epsilon_F (ε_F): Adaptation rate for fitness-driven force
    - nu (ν): Viscosity coefficient for velocity coupling
    - epsilon_Sigma (ε_Σ): Regularization for adaptive diffusion tensor
    - A: Rescale amplitude for fitness potential
    - sigma_prime_min_patch (σ'_min,patch): Z-score regularization
    - patch_radius (ρ): Local patch radius for fitness computation
    - l_viscous: Length scale for viscous kernel
    """

    model_config = {"arbitrary_types_allowed": True}

    epsilon_F: float = Field(
        default=0.1, ge=0, description="Adaptation rate for fitness force (ε_F)"
    )
    nu: float = Field(default=0.1, ge=0, description="Viscosity coefficient (ν)")
    epsilon_Sigma: float = Field(
        gt=0, description="Diffusion regularization parameter (ε_Σ > H_max)"
    )
    A: float = Field(default=1.0, gt=0, description="Fitness potential amplitude")
    sigma_prime_min_patch: float = Field(
        default=0.1, gt=0, description="Z-score regularization (σ'_min,patch)"
    )
    patch_radius: float = Field(default=1.0, gt=0, description="Patch radius (ρ)")
    l_viscous: float = Field(default=1.0, gt=0, description="Viscous kernel length scale")
    use_adaptive_diffusion: bool = Field(
        default=True, description="Enable adaptive diffusion tensor"
    )


class AdaptiveGasParams(BaseModel):
    """Complete parameter set for Adaptive Viscous Fluid Model."""

    model_config = {"arbitrary_types_allowed": True}

    euclidean: EuclideanGasParams = Field(description="Base Euclidean Gas parameters")
    adaptive: AdaptiveParams = Field(description="Adaptive mechanism parameters")
    measurement_fn: str = Field(
        default="potential", description="Measurement function type: 'potential' or 'distance'"
    )


class MeanFieldOps:
    """Mean-field functional computations for fitness potential."""

    @staticmethod
    def compute_fitness_potential(
        state: SwarmState,
        measurement: Tensor,
        params: AdaptiveParams,
    ) -> Tensor:
        """
        Compute mean-field fitness potential V_fit[f](x) for each walker.

        From Definition 1.3 in 07_adaptative_gas.md:
        V_fit[f](x) = g_A(Z_patch[f, x])

        where Z_patch is the patched Z-score with regularization.

        Args:
            state: Current swarm state [N, d]
            measurement: Measurement values d(x_i) for each walker [N]
            params: Adaptive parameters

        Returns:
            Fitness potential values [N]
        """
        N = state.N
        device, dtype = state.device, state.dtype

        # Compute pairwise distances for patch construction
        dx = state.x.unsqueeze(1) - state.x.unsqueeze(0)  # [N, N, d]
        dist = torch.norm(dx, dim=-1)  # [N, N]

        # For each walker i, compute local patch statistics
        V_fit = torch.zeros(N, device=device, dtype=dtype)

        for i in range(N):
            # Identify walkers in patch: ||x_j - x_i|| <= ρ
            in_patch = dist[i] <= params.patch_radius  # [N]

            # Extract patch measurements
            patch_measurements = measurement[in_patch]  # [M]

            if len(patch_measurements) == 0:
                # If no neighbors in patch, use global statistics
                patch_measurements = measurement

            # Compute patch statistics with regularization
            mu_patch = torch.mean(patch_measurements)
            sigma_patch_sq = torch.var(patch_measurements, unbiased=False)
            sigma_prime_patch = torch.maximum(
                torch.sqrt(sigma_patch_sq),
                torch.tensor(params.sigma_prime_min_patch, device=device, dtype=dtype),
            )

            # Compute patched Z-score (Definition 1.3)
            Z_patch = (measurement[i] - mu_patch) / sigma_prime_patch

            # Apply rescale function g_A: sigmoid scaled to [0, A]
            # g_A(z) = A / (1 + exp(-z))
            V_fit[i] = params.A / (1.0 + torch.exp(-Z_patch))

        return V_fit

    @staticmethod
    def compute_fitness_gradient(
        state: SwarmState,
        measurement: Tensor,
        potential_evaluator: PotentialParams,
        params: AdaptiveParams,
    ) -> Tensor:
        """
        Compute gradient of fitness potential ∇_x V_fit.

        Uses finite differences for numerical stability.

        Args:
            state: Current swarm state
            measurement: Measurement values [N]
            potential_evaluator: Potential for computing measurements
            params: Adaptive parameters

        Returns:
            Fitness gradients [N, d]
        """
        N, d = state.N, state.d
        eps = 1e-5  # Finite difference step

        grad_V_fit = torch.zeros(N, d, device=state.device, dtype=state.dtype)

        # Compute V_fit at current point
        V_fit_center = MeanFieldOps.compute_fitness_potential(state, measurement, params)

        # Compute gradient via finite differences
        for j in range(d):
            # Perturb along j-th dimension
            x_plus = state.x.clone()
            x_plus[:, j] += eps

            state_plus = SwarmState(x_plus, state.v)
            measurement_plus = potential_evaluator.evaluate(x_plus)
            V_fit_plus = MeanFieldOps.compute_fitness_potential(
                state_plus, measurement_plus, params
            )

            # Compute finite difference: ∂V / ∂x_j ≈ (V(x+eps) - V(x)) / eps
            grad_V_fit[:, j] = (V_fit_plus - V_fit_center) / eps

        return grad_V_fit

    @staticmethod
    def compute_fitness_hessian(
        state: SwarmState,
        measurement: Tensor,
        potential_evaluator: PotentialParams,
        params: AdaptiveParams,
    ) -> Tensor:
        """
        Compute Hessian of fitness potential H_i = ∇²_x V_fit for each walker.

        From Definition 2.1 in 07_adaptative_gas.md:
        H_i(S) = ∇²_{x_i} V_i

        Returns approximate Hessian using finite differences for efficiency.

        Args:
            state: Current swarm state
            measurement: Measurement values [N]
            potential_evaluator: Potential for computing measurements
            params: Adaptive parameters

        Returns:
            Hessian matrices [N, d, d]
        """
        N, d = state.N, state.d
        eps = 1e-4  # Finite difference step

        # Compute gradient at current point
        grad_center = MeanFieldOps.compute_fitness_gradient(
            state, measurement, potential_evaluator, params
        )

        # Approximate Hessian via finite differences
        H = torch.zeros(N, d, d, device=state.device, dtype=state.dtype)

        for j in range(d):
            # Perturb along j-th dimension
            x_plus = state.x.clone()
            x_plus[:, j] += eps

            state_plus = SwarmState(x_plus, state.v)
            measurement_plus = potential_evaluator.evaluate(x_plus)
            grad_plus = MeanFieldOps.compute_fitness_gradient(
                state_plus, measurement_plus, potential_evaluator, params
            )

            # Compute finite difference: ∂²V / ∂x_i ∂x_j ≈ (∂V/∂x_i(x+eps) - ∂V/∂x_i(x)) / eps
            H[:, :, j] = (grad_plus - grad_center) / eps

        # Symmetrize Hessian
        return 0.5 * (H + H.transpose(1, 2))


class ViscousForce:
    """Viscous force implementation from Section 1.1 of 07_adaptative_gas.md."""

    @staticmethod
    def gaussian_kernel(r: Tensor, l: float) -> Tensor:
        """
        Gaussian viscous kernel K(r) = exp(-r²/(2l²)).

        Args:
            r: Distance [N, N]
            l: Length scale

        Returns:
            Kernel values [N, N]
        """
        return torch.exp(-(r**2) / (2 * l**2))

    @staticmethod
    def compute(state: SwarmState, nu: float, l_viscous: float) -> Tensor:
        """
        Compute viscous force F_viscous(x_i, S).

        From Definition 1.1 in 07_adaptative_gas.md:
        F_viscous(x_i, S) = ν Σ_{j≠i} K(||x_i - x_j||) (v_j - v_i)

        Args:
            state: Current swarm state
            nu: Viscosity coefficient
            l_viscous: Kernel length scale

        Returns:
            Viscous forces [N, d]
        """
        _N, _d = state.N, state.d

        # Compute pairwise distances
        dx = state.x.unsqueeze(1) - state.x.unsqueeze(0)  # [N, N, d]
        dist = torch.norm(dx, dim=-1)  # [N, N]

        # Compute kernel weights
        K = ViscousForce.gaussian_kernel(dist, l_viscous)  # [N, N]
        K.fill_diagonal_(0.0)  # Exclude self-interaction

        # Compute velocity differences
        dv = state.v.unsqueeze(1) - state.v.unsqueeze(0)  # [N, N, d]

        # Compute viscous force: F_i = ν Σ_j K_ij (v_j - v_i) = ν Σ_j K_ij (-dv_ij)
        # Note: dv[i,j] = v_i - v_j, so we want -dv[i,j] = v_j - v_i
        return -nu * torch.einsum("ij,ijk->ik", K, dv)  # [N, d]


class AdaptiveKineticOperator(KineticOperator):
    """
    Kinetic operator for Adaptive Viscous Fluid Model.

    Extends the base BAOAB integrator with:
    1. Adaptive force ε_F ∇V_fit
    2. Viscous force ν F_viscous
    3. Regularized Hessian diffusion (H + ε_Σ I)^{-1/2}
    """

    def __init__(
        self,
        euclidean_params: EuclideanGasParams,
        adaptive_params: AdaptiveParams,
    ):
        """Initialize adaptive kinetic operator."""
        super().__init__(
            euclidean_params.langevin,
            euclidean_params.potential,
            torch.device(euclidean_params.device),
            euclidean_params.torch_dtype,
        )
        self.adaptive_params = adaptive_params

    def compute_adaptive_diffusion_tensor(self, state: SwarmState, measurement: Tensor) -> Tensor:
        """
        Compute regularized adaptive diffusion tensor Σ_reg.

        From Definition 2.1 in 07_adaptative_gas.md:
        Σ_reg(x_i, S) = (H_i(S) + ε_Σ I)^{-1/2}

        Args:
            state: Current swarm state
            measurement: Measurement values [N]

        Returns:
            Diffusion tensors [N, d, d]
        """
        if not self.adaptive_params.use_adaptive_diffusion:
            # Fall back to isotropic diffusion
            N, d = state.N, state.d
            sigma = torch.sqrt(
                torch.tensor(2.0 / self.beta, device=state.device, dtype=state.dtype)
            )
            return sigma * torch.eye(d, device=state.device, dtype=state.dtype).unsqueeze(
                0
            ).expand(N, d, d)

        # Compute Hessian H_i for each walker
        H = MeanFieldOps.compute_fitness_hessian(
            state, measurement, self.potential, self.adaptive_params
        )  # [N, d, d]

        # Check for NaN or Inf in Hessian
        if torch.any(~torch.isfinite(H)):
            # Fallback to isotropic diffusion if Hessian is not finite
            warnings.warn(
                "Hessian contains NaN or Inf values. Falling back to isotropic diffusion. "
                "This may indicate walkers are too clustered or variance collapsed. "
                "Consider increasing epsilon_Sigma or sigma_prime_min_patch.",
                RuntimeWarning,
                stacklevel=2,
            )
            N, d = state.N, state.d
            sigma = torch.sqrt(
                torch.tensor(2.0 / self.beta, device=state.device, dtype=state.dtype)
            )
            return sigma * torch.eye(d, device=state.device, dtype=state.dtype).unsqueeze(
                0
            ).expand(N, d, d)

        # Regularize: H_reg = H + ε_Σ I
        eps_Sigma = self.adaptive_params.epsilon_Sigma
        I = torch.eye(state.d, device=state.device, dtype=state.dtype)
        H_reg = H + eps_Sigma * I.unsqueeze(0)  # [N, d, d]

        # Use eigendecomposition directly to avoid explicit inversion
        # This is more numerically stable: G_reg = H_reg^{-1} = V Λ^{-1} V^T
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(H_reg)  # [N, d], [N, d, d]

            # Check eigenvalues are positive (they should be with regularization)
            min_eigenvalue = torch.min(eigenvalues)
            if min_eigenvalue <= 0:
                # This should never happen with proper regularization, but failsafe
                eigenvalues = torch.maximum(
                    eigenvalues,
                    torch.tensor(eps_Sigma / 10, device=state.device, dtype=state.dtype),
                )

            # G_reg = V Λ^{-1} V^T, so G_reg^{1/2} = V Λ^{-1/2} V^T
            inv_sqrt_eigenvalues = 1.0 / torch.sqrt(eigenvalues)  # [N, d]
            Sigma_reg = (
                eigenvectors
                @ torch.diag_embed(inv_sqrt_eigenvalues)
                @ eigenvectors.transpose(1, 2)
            )

            # Final check for numerical issues
            if torch.any(~torch.isfinite(Sigma_reg)):
                msg = "Non-finite values in Sigma_reg"
                raise RuntimeError(msg)

            return Sigma_reg

        except (RuntimeError, torch._C._LinAlgError) as e:
            # If anything fails, fall back to isotropic diffusion
            warnings.warn(
                f"Error computing adaptive diffusion tensor: {e}. "
                "Falling back to isotropic diffusion. "
                "Consider increasing epsilon_Sigma for better numerical stability.",
                RuntimeWarning,
                stacklevel=2,
            )
            N, d = state.N, state.d
            sigma = torch.sqrt(
                torch.tensor(2.0 / self.beta, device=state.device, dtype=state.dtype)
            )
            return sigma * torch.eye(d, device=state.device, dtype=state.dtype).unsqueeze(
                0
            ).expand(N, d, d)

    def apply(self, state: SwarmState) -> SwarmState:
        """
        Apply hybrid BAOAB integrator with adaptive mechanisms.

        Integrates the Hybrid SDE from Definition 1.1:
        dv_i = [F_stable + F_adapt + F_viscous - γv_i] dt + Σ_reg ∘ dW_i

        Args:
            state: Current swarm state

        Returns:
            Updated state after integration
        """
        x, v = state.x.clone(), state.v.clone()
        N, d = state.N, state.d

        # Compute measurement for adaptive forces
        measurement = self.potential.evaluate(x)  # [N]

        # ============ First B step: Apply all forces ============
        x.requires_grad_(True)
        U = self.potential.evaluate(x)  # [N]
        grad_U = torch.autograd.grad(U.sum(), x, create_graph=False)[0]  # [N, d]
        x.requires_grad_(False)

        # Stable backbone force: -∇U(x)
        F_stable = -grad_U  # [N, d]

        # Adaptive force: ε_F ∇V_fit (if enabled)
        F_adapt = torch.zeros_like(F_stable)
        if self.adaptive_params.epsilon_F > 0:
            grad_V_fit = MeanFieldOps.compute_fitness_gradient(
                state, measurement, self.potential, self.adaptive_params
            )
            F_adapt = self.adaptive_params.epsilon_F * grad_V_fit

        # Viscous force: ν F_viscous (if enabled)
        F_viscous = torch.zeros_like(F_stable)
        if self.adaptive_params.nu > 0:
            F_viscous = ViscousForce.compute(
                state, self.adaptive_params.nu, self.adaptive_params.l_viscous
            )

        # Total force (excluding friction)
        F_total = F_stable + F_adapt + F_viscous

        # Apply force and friction for half timestep
        v = v + (self.dt / 2) * F_total - (self.dt / 2) * self.gamma * v

        # ============ First A step ============
        x += self.dt / 2 * v

        # ============ O step: Ornstein-Uhlenbeck with adaptive diffusion ============
        if self.adaptive_params.use_adaptive_diffusion:
            # Compute adaptive diffusion tensor
            Sigma_reg = self.compute_adaptive_diffusion_tensor(
                SwarmState(x, v), measurement
            )  # [N, d, d]

            # Generate white noise
            xi = torch.randn(N, d, device=self.device, dtype=self.dtype)  # [N, d]

            # Apply Ornstein-Uhlenbeck with state-dependent diffusion
            # v_new = c1 * v + Σ_reg @ ξ
            # Note: Scaling with sqrt((1-c1²)/β) is absorbed into Σ_reg computation
            noise_scale = torch.sqrt(2.0 / self.beta * (1.0 - self.c1**2))
            v = self.c1 * v + noise_scale * torch.einsum("nij,nj->ni", Sigma_reg, xi)
        else:
            # Standard isotropic O-step
            xi = torch.randn(N, d, device=self.device, dtype=self.dtype)
            v = self.c1 * v + self.c2 * xi

        # ============ Second A step ============
        x += self.dt / 2 * v

        # ============ Second B step ============
        x.requires_grad_(True)
        U = self.potential.evaluate(x)  # [N]
        grad_U = torch.autograd.grad(U.sum(), x, create_graph=False)[0]  # [N, d]
        x.requires_grad_(False)

        F_stable = -grad_U

        # Recompute adaptive forces at new position
        measurement = self.potential.evaluate(x)
        state_new = SwarmState(x, v)

        F_adapt = torch.zeros_like(F_stable)
        if self.adaptive_params.epsilon_F > 0:
            grad_V_fit = MeanFieldOps.compute_fitness_gradient(
                state_new, measurement, self.potential, self.adaptive_params
            )
            F_adapt = self.adaptive_params.epsilon_F * grad_V_fit

        F_viscous = torch.zeros_like(F_stable)
        if self.adaptive_params.nu > 0:
            F_viscous = ViscousForce.compute(
                state_new, self.adaptive_params.nu, self.adaptive_params.l_viscous
            )

        F_total = F_stable + F_adapt + F_viscous

        v = v + (self.dt / 2) * F_total - (self.dt / 2) * self.gamma * v

        return SwarmState(x, v)


class AdaptiveGas(EuclideanGas):
    """
    Adaptive Viscous Fluid Model implementation.

    Extends Euclidean Gas with adaptive feedback mechanisms while
    maintaining the stable backbone structure.

    Key features:
    1. Fitness-driven adaptive force
    2. Viscous velocity coupling
    3. Regularized Hessian-based diffusion
    4. Provable stability via regularization
    """

    def __init__(self, params: AdaptiveGasParams):
        """
        Initialize Adaptive Gas.

        Args:
            params: Complete adaptive gas parameters
        """
        # Initialize base Euclidean Gas (this sets up cloning operator)
        super().__init__(params.euclidean)

        # Store adaptive parameters
        self.adaptive_params = params.adaptive

        # Replace kinetic operator with adaptive version
        self.kinetic_op = AdaptiveKineticOperator(params.euclidean, params.adaptive)

    def get_fitness_potential(self, state: SwarmState) -> Tensor:
        """
        Get current fitness potential values for all walkers.

        Useful for visualization and analysis.

        Args:
            state: Current swarm state

        Returns:
            Fitness values [N]
        """
        measurement = self.kinetic_op.potential.evaluate(state.x)
        return MeanFieldOps.compute_fitness_potential(state, measurement, self.adaptive_params)
