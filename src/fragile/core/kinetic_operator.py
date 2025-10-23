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

from fragile.bounds import TorchBounds
from fragile.core.distance import compute_periodic_distance_matrix


def psi_v(v: Tensor, V_alg: float) -> Tensor:
    """Apply smooth velocity squashing map to ensure bounded magnitude.

    This implements the smooth radial squashing map from Lemma lem-squashing-properties-generic
    in Section 3.3 of docs/source/1_euclidean_gas/02_euclidean_gas.md:

    ψ_v(v) = V_alg * (v / (V_alg + ||v||))

    The map ensures all output velocities have magnitude strictly less than V_alg while
    preserving direction and providing smooth (C^∞) behavior away from the origin.

    Mathematical Properties (proven in framework):
    - 1-Lipschitz: ||ψ_v(v) - ψ_v(v')|| ≤ ||v - v'|| for all v, v'
    - Smooth: ψ_v ∈ C^∞(ℝ^d \\ {0})
    - Bounded: ||ψ_v(v)|| < V_alg for all v ∈ ℝ^d

    Design Rationale:
    Smooth squashing maps are chosen over hard radial projections to provide
    differentiability for both position and velocity coordinates, a prerequisite
    for one-step minorization proofs and deriving continuum limits.

    Args:
        v: Velocity vectors to squash. Shape: [N, d] or [d]
        V_alg: Algorithmic velocity bound (must be positive)

    Returns:
        Squashed velocity vectors with same shape as input.
        Guarantee: ||ψ_v(v)|| < V_alg

    Example:
        >>> import torch
        >>> v = torch.randn(100, 3)  # 100 walkers in 3D
        >>> v_squashed = psi_v(v, V_alg=1.0)
        >>> assert (v_squashed.norm(dim=-1) < 1.0).all()

    Note:
        For numerical stability at the origin, the formula naturally handles ||v|| = 0
        without special cases: ψ_v(0) = 0.

    Reference:
        - Euclidean Gas specification: docs/source/1_euclidean_gas/02_euclidean_gas.md § 3.3
        - Generic squashing lemma: lem-squashing-properties-generic
    """
    if V_alg <= 0:
        msg = f"V_alg must be positive, got {V_alg}"
        raise ValueError(msg)

    v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    return V_alg * v / (V_alg + v_norm)


class KineticOperator(BaseModel):
    """Kinetic operator using BAOAB integrator for Langevin dynamics.

    Supports adaptive extensions from the Geometric Viscous Fluid Model:
    - Fitness-based force: -ε_F · ∇V_fit (optional)
    - Anisotropic diffusion: Σ_reg = (∇²V_fit + ε_Σ I)^{-1/2} (optional)
    - Viscous coupling: ν · F_viscous for fluid-like collective behavior (optional)

    Mathematical notation:
    - gamma (γ): Friction coefficient
    - beta (β): Inverse temperature 1/(k_B T)
    - delta_t (Δt): Time step size
    - epsilon_F (ε_F): Adaptation rate for fitness force
    - epsilon_Sigma (ε_Σ): Hessian regularization parameter
    - nu (ν): Viscous coupling strength

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

    # Viscous coupling (velocity-dependent damping)
    nu: float = Field(
        default=0.0, ge=0, description="Viscous coupling strength (ν)"
    )
    use_viscous_coupling: bool = Field(
        default=False, description="Enable viscous coupling for fluid-like behavior"
    )
    viscous_length_scale: float = Field(
        default=1.0, gt=0, description="Length scale (l) for Gaussian kernel K(r) = exp(-r²/(2l²))"
    )

    # Velocity squashing map
    V_alg: float = Field(
        default=float('inf'), gt=0, description="Algorithmic velocity bound for smooth squashing map"
    )
    use_velocity_squashing: bool = Field(
        default=False, description="Enable smooth velocity squashing map ψ_v"
    )

    # Non-Pydantic fields (set in __init__)
    potential: object = Field(default=None, init=False, exclude=True)
    device: torch.device = Field(default=None, init=False, exclude=True)
    dtype: torch.dtype = Field(default=None, init=False, exclude=True)
    dt: float = Field(default=0.0, init=False, exclude=True)
    c1: torch.Tensor = Field(default=None, init=False, exclude=True)
    c2: torch.Tensor = Field(default=None, init=False, exclude=True)
    bounds: TorchBounds | None = Field(default=None, init=False, exclude=True)
    pbc: bool = Field(default=False, init=False, exclude=True)

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
        nu: float = 0.0,
        use_viscous_coupling: bool = False,
        viscous_length_scale: float = 1.0,
        V_alg: float = float('inf'),
        use_velocity_squashing: bool = False,
        potential=None,
        device: torch.device = None,
        dtype: torch.dtype = None,
        bounds: TorchBounds | None = None,
        pbc: bool = False,
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
            nu: Viscous coupling strength (ν)
            use_viscous_coupling: Enable viscous coupling
            viscous_length_scale: Length scale for Gaussian kernel
            V_alg: Algorithmic velocity bound for smooth squashing map (default: inf)
            use_velocity_squashing: Enable smooth velocity squashing map ψ_v
            potential: Target potential (must be callable: U(x) -> [N]).
                      Required if use_potential_force=True, can be None otherwise.
            device: PyTorch device (defaults to CPU)
            dtype: PyTorch dtype (defaults to float32)
            bounds: Domain bounds (required for periodic boundary conditions)
            pbc: Enable periodic boundary conditions for distance calculations

        Raises:
            ValueError: If required components are missing based on settings

        Note:
            Fitness gradients and Hessians are passed to apply() method, not stored here.
            This keeps the kinetic operator stateless with respect to fitness computations.

            When pbc=True, viscous coupling uses minimum image convention for distances,
            ensuring correct fluid behavior across periodic boundaries (torus topology).
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
            nu=nu,
            use_viscous_coupling=use_viscous_coupling,
            viscous_length_scale=viscous_length_scale,
            V_alg=V_alg,
            use_velocity_squashing=use_velocity_squashing,
        )

        self.potential = potential
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        self.bounds = bounds
        self.pbc = pbc

        # Validate configuration
        if self.use_potential_force:
            if potential is None:
                msg = "potential required when use_potential_force=True"
                raise ValueError(msg)
            if not callable(potential):
                msg = f"potential must be callable, got {type(potential)}"
                raise TypeError(msg)

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
            grad_fitness: Precomputed fitness gradient ∇V_fit [N, d]
                (required if use_fitness_force=True)

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
            x.requires_grad_(True)  # noqa: FBT003
            U = self.potential(x)  # [N]
            grad_U = torch.autograd.grad(U.sum(), x, create_graph=False)[0]  # [N, d]
            force -= grad_U
            x.requires_grad_(False)  # noqa: FBT003

        # Fitness force: -ε_F · ∇V_fit(x)
        if self.use_fitness_force:
            if grad_fitness is None:
                msg = "grad_fitness required when use_fitness_force=True"
                raise ValueError(msg)
            force -= self.epsilon_F * grad_fitness

        return force

    def _compute_viscous_force(
        self,
        x: Tensor,
        v: Tensor,
    ) -> Tensor:
        """Compute viscous coupling force using normalized graph Laplacian.

        The viscous force implements fluid-like collective behavior by coupling
        nearby walkers' velocities through a Gaussian kernel:

        F_viscous(x_i) = ν ∑_{j≠i} [K(||x_i - x_j||) / deg(i)] (v_j - v_i)

        where:
        - K(r) = exp(-r²/(2l²)) is a Gaussian kernel with length scale l
        - deg(i) = ∑_{k≠i} K(||x_i - x_k||) is the local degree (normalization)
        - ν is the viscous coupling strength

        This creates a row-normalized graph Laplacian structure that ensures
        N-uniform bounds and dissipative behavior (reduces velocity variance).

        Args:
            x: Positions [N, d]
            v: Velocities [N, d]

        Returns:
            viscous_force: Velocity-dependent damping force [N, d]

        Note:
            - When nu = 0 or use_viscous_coupling = False, returns zero
            - Dissipative property proven in docs/source/2_geometric_gas/11_geometric_gas.md § 6.3
            - N-uniform bounds from row normalization

        Reference:
            - Geometric Gas specification: docs/source/2_geometric_gas/11_geometric_gas.md § 2.1.3
            - Graph Laplacian structure: docs/source/2_geometric_gas/17_qsd_exchangeability_geometric.md
        """
        # Early return if viscous coupling is disabled or nu = 0
        if not self.use_viscous_coupling or self.nu == 0.0:
            return torch.zeros_like(v)

        N, d = x.shape

        # Compute pairwise distances
        # With PBC: use minimum image convention (wrapping)
        # Without PBC: standard Euclidean distance
        # distances[i, j] = ||x_i - x_j|| (accounting for wrapping if pbc=True)
        distances = compute_periodic_distance_matrix(
            x, y=None, bounds=self.bounds, pbc=self.pbc
        )  # [N, N]

        # Compute Gaussian kernel K(r) = exp(-r²/(2l²))
        l_sq = self.viscous_length_scale ** 2
        kernel = torch.exp(-distances ** 2 / (2 * l_sq))  # [N, N]

        # Zero out diagonal (no self-interaction)
        kernel.fill_diagonal_(0.0)

        # Compute local degree deg(i) = ∑_{j≠i} K(||x_i - x_j||)
        # Add small epsilon for numerical stability
        deg = kernel.sum(dim=1, keepdim=True)  # [N, 1]
        deg = torch.clamp(deg, min=1e-10)  # Avoid division by zero

        # Compute normalized weights w_ij = K_ij / deg_i
        weights = kernel / deg  # [N, N], broadcasting deg over columns

        # Compute velocity differences for all pairs
        # v_diff[i, j] = v_j - v_i
        v_diff = v.unsqueeze(0) - v.unsqueeze(1)  # [N, N, d]

        # Compute weighted sum: F_visc_i = ν * ∑_j w_ij * (v_j - v_i)
        # This is a batched matrix-vector product
        viscous_force = self.nu * torch.einsum('ij,ijd->id', weights, v_diff)  # [N, d]

        return viscous_force

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

        where F(x, v) = -∇U(x) - ε_F · ∇V_fit + ν · F_viscous(x, v) and noise can be:
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
        if self.use_viscous_coupling:
            force += self._compute_viscous_force(x, v)
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
        if self.use_viscous_coupling:
            force += self._compute_viscous_force(x, v)
        v += (self.dt / 2) * force

        # === VELOCITY SQUASHING: Apply smooth radial squashing map ===
        if self.use_velocity_squashing:
            v = psi_v(v, self.V_alg)

        # Return state with same type as input
        # Create new state object using the same class as input
        return type(state)(x, v)
