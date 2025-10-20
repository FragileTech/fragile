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
    """Data container for Langevin dynamics parameters.

    This class exists purely for organizing parameters in higher-level configs
    like EuclideanGasParams. When instantiating KineticOperator, unpack these fields.

    Mathematical notation:
    - gamma (γ): Friction coefficient
    - beta (β): Inverse temperature 1/(k_B T)
    - delta_t (Δt): Time step size
    - epsilon_F (ε_F): Adaptation rate for fitness force
    - epsilon_Sigma (ε_Σ): Hessian regularization parameter
    """

    model_config = {"arbitrary_types_allowed": True}

    # Standard Langevin parameters
    gamma: float = Field(gt=0, description="Friction coefficient (γ)")
    beta: float = Field(gt=0, description="Inverse temperature 1/(k_B T) (β)")
    delta_t: float = Field(gt=0, description="Time step size (Δt)")
    integrator: str = Field("baoab", description="Integration scheme (baoab)")

    # Fitness-based adaptive force (Geometric Gas extension)
    epsilon_F: float = Field(
        default=0.0, ge=0, description="Adaptation rate for fitness force (ε_F)"
    )
    use_fitness_force: bool = Field(
        default=False, description="Enable fitness-based force -ε_F · ∇V_fit"
    )
    use_potential_force: bool = Field(
        default=True, description="Enable potential gradient force -∇U(x)"
    )

    # Anisotropic diffusion tensor (Hessian-based)
    epsilon_Sigma: float = Field(
        default=0.1, ge=0, description="Hessian regularization (ε_Σ) for positive definiteness"
    )
    use_anisotropic_diffusion: bool = Field(
        default=False, description="Enable Hessian-based anisotropic diffusion Σ_reg"
    )
    diagonal_diffusion: bool = Field(
        default=True, description="Use diagonal-only diffusion (faster, O(Nd) vs O(Nd²))"
    )

    def noise_std(self) -> float:
        """Standard deviation for BAOAB noise (isotropic case)."""
        return (1.0 - torch.exp(torch.tensor(-2 * self.gamma * self.delta_t))).sqrt().item()


class KineticOperator(BaseModel):
    """Kinetic operator using BAOAB integrator for Langevin dynamics.

    Supports adaptive extensions from the Geometric Viscous Fluid Model:
    - Fitness-based force: -ε_F · ∇V_fit (optional)
    - Anisotropic diffusion: Σ_reg = (∇²V_fit + ε_Σ I)^{-1/2} (optional)

    Mathematical notation:
    - gamma (γ): Friction coefficient
    - beta (β): Inverse temperature 1/(k_B T)
    - delta_t (Δt): Time step size
    - epsilon_F (ε_F): Adaptation rate for fitness force
    - epsilon_Sigma (ε_Σ): Hessian regularization parameter

    Reference: docs/source/2_geometric_gas/11_geometric_gas.md
    """

    model_config = {"arbitrary_types_allowed": True}

    # Standard Langevin parameters
    gamma: float = Field(gt=0, description="Friction coefficient (γ)")
    beta: float = Field(gt=0, description="Inverse temperature 1/(k_B T) (β)")
    delta_t: float = Field(gt=0, description="Time step size (Δt)")
    integrator: str = Field("baoab", description="Integration scheme (baoab)")

    # Fitness-based adaptive force (Geometric Gas extension)
    epsilon_F: float = Field(
        default=0.0, ge=0, description="Adaptation rate for fitness force (ε_F)"
    )
    use_fitness_force: bool = Field(
        default=False, description="Enable fitness-based force -ε_F · ∇V_fit"
    )
    use_potential_force: bool = Field(
        default=True, description="Enable potential gradient force -∇U(x)"
    )

    # Anisotropic diffusion tensor (Hessian-based)
    epsilon_Sigma: float = Field(
        default=0.1, ge=0, description="Hessian regularization (ε_Σ) for positive definiteness"
    )
    use_anisotropic_diffusion: bool = Field(
        default=False, description="Enable Hessian-based anisotropic diffusion Σ_reg"
    )
    diagonal_diffusion: bool = Field(
        default=True, description="Use diagonal-only diffusion (faster, O(Nd) vs O(Nd²))"
    )

    # Non-Pydantic fields (set in __init__)
    potential: object = Field(default=None, init=False, exclude=True)
    device: torch.device = Field(default=None, init=False, exclude=True)
    dtype: torch.dtype = Field(default=None, init=False, exclude=True)
    dt: float = Field(default=0.0, init=False, exclude=True)
    c1: torch.Tensor = Field(default=None, init=False, exclude=True)
    c2: torch.Tensor = Field(default=None, init=False, exclude=True)

    def __init__(
        self,
        gamma: float,
        beta: float,
        delta_t: float,
        integrator: str = "baoab",
        epsilon_F: float = 0.0,
        use_fitness_force: bool = False,
        use_potential_force: bool = True,
        epsilon_Sigma: float = 0.1,
        use_anisotropic_diffusion: bool = False,
        diagonal_diffusion: bool = True,
        potential=None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """
        Initialize kinetic operator.

        Args:
            gamma: Friction coefficient (γ)
            beta: Inverse temperature 1/(k_B T) (β)
            delta_t: Time step size (Δt)
            integrator: Integration scheme (default: "baoab")
            epsilon_F: Adaptation rate for fitness force (ε_F)
            use_fitness_force: Enable fitness-based force
            use_potential_force: Enable potential gradient force
            epsilon_Sigma: Hessian regularization (ε_Σ)
            use_anisotropic_diffusion: Enable Hessian-based anisotropic diffusion
            diagonal_diffusion: Use diagonal-only diffusion
            potential: Target potential (must have evaluate(x) method).
                      Required if use_potential_force=True, can be None otherwise.
            device: PyTorch device (defaults to CPU)
            dtype: PyTorch dtype (defaults to float32)

        Raises:
            ValueError: If required components are missing based on settings

        Note:
            Fitness gradients and Hessians are passed to apply() method, not stored here.
            This keeps the kinetic operator stateless with respect to fitness computations.
        """
        super().__init__(
            gamma=gamma,
            beta=beta,
            delta_t=delta_t,
            integrator=integrator,
            epsilon_F=epsilon_F,
            use_fitness_force=use_fitness_force,
            use_potential_force=use_potential_force,
            epsilon_Sigma=epsilon_Sigma,
            use_anisotropic_diffusion=use_anisotropic_diffusion,
            diagonal_diffusion=diagonal_diffusion,
        )

        self.potential = potential
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        # Validate configuration
        if self.use_potential_force and potential is None:
            msg = "potential required when use_potential_force=True"
            raise ValueError(msg)

        # Precompute BAOAB constants
        self.dt = self.delta_t

        # O-step coefficients (for isotropic case)
        self.c1 = torch.exp(torch.tensor(-self.gamma * self.dt, dtype=self.dtype))
        self.c2 = torch.sqrt((1.0 - self.c1**2) / self.beta)  # Noise amplitude

    def noise_std(self) -> float:
        """Standard deviation for BAOAB noise (isotropic case)."""
        return (1.0 - torch.exp(torch.tensor(-2 * self.gamma * self.delta_t))).sqrt().item()

    def _compute_force(
        self,
        x: Tensor,
        v: Tensor,
        grad_fitness: Tensor | None = None,
    ) -> Tensor:
        """Compute combined force from potential and/or fitness gradients.

        The total force is: F_total = F_potential + F_fitness where:
        - F_potential = -∇U(x) (if use_potential_force=True)
        - F_fitness = -ε_F · ∇V_fit (if use_fitness_force=True)

        Args:
            x: Positions [N, d]
            v: Velocities [N, d]
            grad_fitness: Precomputed fitness gradient ∇V_fit [N, d] (required if use_fitness_force=True)

        Returns:
            force: Combined force vector [N, d]

        Raises:
            ValueError: If grad_fitness is None but use_fitness_force=True

        Note:
            Both forces use negative gradients to drift toward lower potential/higher fitness.
        """
        _N, _d = x.shape
        force = torch.zeros_like(x)

        # Potential force: -∇U(x)
        if self.use_potential_force:
            x.requires_grad_(True)
            U = self.potential.evaluate(x)  # [N]
            grad_U = torch.autograd.grad(U.sum(), x, create_graph=False)[0]  # [N, d]
            force -= grad_U
            x.requires_grad_(False)

        # Fitness force: -ε_F · ∇V_fit(x)
        if self.use_fitness_force:
            if grad_fitness is None:
                msg = "grad_fitness required when use_fitness_force=True"
                raise ValueError(msg)
            force -= self.epsilon_F * grad_fitness

        return force

    def _compute_diffusion_tensor(
        self,
        x: Tensor,
        hess_fitness: Tensor | None = None,
    ) -> Tensor:
        """Compute anisotropic diffusion tensor Σ_reg = (∇²V_fit + ε_Σ I)^{-1/2}.

        The regularized Hessian ensures positive definiteness and provides
        state-dependent noise aligned with the fitness landscape geometry.

        Args:
            x: Positions [N, d]
            hess_fitness: Precomputed fitness Hessian ∇²V_fit
                         - If diagonal_diffusion=True: [N, d]
                         - If diagonal_diffusion=False: [N, d, d]
                         Required if use_anisotropic_diffusion=True

        Returns:
            If diagonal_diffusion=True: Diagonal elements [N, d]
            If diagonal_diffusion=False: Full diffusion tensor [N, d, d]

        Raises:
            ValueError: If hess_fitness is None but use_anisotropic_diffusion=True

        Note:
            For isotropic diffusion, returns constant σ = c2 (BAOAB noise amplitude).
        """
        N, d = x.shape

        if not self.use_anisotropic_diffusion:
            # Isotropic diffusion: σ I (standard BAOAB)
            if self.diagonal_diffusion:
                return self.c2 * torch.ones((N, d), device=x.device, dtype=x.dtype)
            eye = torch.eye(d, device=x.device, dtype=x.dtype)
            return self.c2 * eye.unsqueeze(0).expand(N, d, d)

        # Use precomputed Hessian
        if hess_fitness is None:
            msg = "hess_fitness required when use_anisotropic_diffusion=True"
            raise ValueError(msg)

        hess = hess_fitness

        # Regularize: H_reg = H + ε_Σ I
        # The regularization ensures positive definiteness even if Hessian has negative eigenvalues
        eps_I = self.epsilon_Sigma

        if self.diagonal_diffusion:
            # Diagonal case: simple element-wise operations
            hess_reg = hess + eps_I  # [N, d]

            # Ensure positive values (clamp to avoid NaN from negative values)
            hess_reg = torch.clamp(hess_reg, min=eps_I)

            # Σ_diag = (H_reg)^{-1/2}
            sigma = 1.0 / torch.sqrt(hess_reg)
        else:
            # Full anisotropic case: matrix inverse square root
            # Add ε_Σ I to each [d, d] block
            eye = torch.eye(d, device=x.device, dtype=x.dtype)
            hess_reg = hess + eps_I * eye.unsqueeze(0)  # [N, d, d]

            # Compute matrix inverse square root via eigendecomposition
            # For symmetric positive definite A: A^{-1/2} = Q Λ^{-1/2} Q^T
            eigenvalues, eigenvectors = torch.linalg.eigh(hess_reg)  # [N, d], [N, d, d]

            # Clamp eigenvalues to ensure positivity (handle numerical errors)
            eigenvalues = torch.clamp(eigenvalues, min=eps_I)

            eigenvalues_inv_sqrt = 1.0 / torch.sqrt(eigenvalues)  # [N, d]

            # Reconstruct: Σ = Q Λ^{-1/2} Q^T
            sigma = (
                eigenvectors
                @ torch.diag_embed(eigenvalues_inv_sqrt)
                @ eigenvectors.transpose(-2, -1)
            )

        return sigma

    def apply(
        self,
        state,
        grad_fitness: Tensor | None = None,
        hess_fitness: Tensor | None = None,
    ):
        """Apply BAOAB integrator for one time step with optional adaptive features.

        Standard BAOAB sequence:
            B: v → v + (Δt/2) · F(x, v)          [Force step]
            A: x → x + (Δt/2) · v                 [Position update]
            O: v → c1 · v + noise                 [Ornstein-Uhlenbeck]
            A: x → x + (Δt/2) · v                 [Position update]
            B: v → v + (Δt/2) · F(x, v)          [Force step]

        where F(x, v) = -∇U(x) - ε_F · ∇V_fit and noise can be:
            - Isotropic: c2 · ξ where ξ ~ N(0, I)
            - Anisotropic: Σ_reg · ξ where Σ_reg = (∇²V_fit + ε_Σ I)^{-1/2}

        Args:
            state: Current swarm state (must have .x and .v attributes)
            grad_fitness: Precomputed fitness gradient ∇V_fit [N, d]
                         (required if use_fitness_force=True)
            hess_fitness: Precomputed fitness Hessian ∇²V_fit
                         - If diagonal_diffusion=True: [N, d]
                         - If diagonal_diffusion=False: [N, d, d]
                         (required if use_anisotropic_diffusion=True)

        Returns:
            Updated state after integration

        Raises:
            ValueError: If grad_fitness/hess_fitness are None but features are enabled

        Note:
            The fitness gradient and Hessian are computed by the caller (EuclideanGas)
            using the FitnessOperator, and passed here as precomputed values.
            Reference: Geometric Viscous Fluid Model (11_geometric_gas.md)
        """
        x, v = state.x.clone(), state.v.clone()
        N, d = state.N, state.d

        # === FIRST B STEP: Apply forces ===
        force = self._compute_force(x, v, grad_fitness)
        v += (self.dt / 2) * force

        # === FIRST A STEP: Update positions ===
        x += (self.dt / 2) * v

        # === O STEP: Ornstein-Uhlenbeck with optional anisotropic noise ===
        ξ = torch.randn(N, d, device=self.device, dtype=self.dtype)

        if self.use_anisotropic_diffusion:
            # Compute state-dependent diffusion tensor Σ_reg
            sigma = self._compute_diffusion_tensor(x, hess_fitness)

            if self.diagonal_diffusion:
                # Diagonal: σ[i, j] · ξ[i, j] (element-wise)
                noise = sigma * ξ
            else:
                # Full anisotropic: Σ[i] @ ξ[i] for each walker i
                # bmm: [N, d, d] @ [N, d, 1] → [N, d, 1]
                noise = torch.bmm(sigma, ξ.unsqueeze(-1)).squeeze(-1)  # [N, d]

            v = self.c1 * v + noise
        else:
            # Isotropic: standard BAOAB with constant noise amplitude
            v = self.c1 * v + self.c2 * ξ

        # === SECOND A STEP: Update positions ===
        x += (self.dt / 2) * v

        # === SECOND B STEP: Apply forces ===
        force = self._compute_force(x, v, grad_fitness)
        v += (self.dt / 2) * force

        # Return state with same type as input
        # Create new state object using the same class as input
        return type(state)(x, v)
