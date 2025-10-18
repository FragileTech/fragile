"""
Geometric Gas: An Adaptive Information-Geometric Implementation

This module implements the Geometric Viscous Fluid Model from
docs/source/2_geometric_gas/11_geometric_gas.md using PyTorch for vectorization
and Pydantic for parameter management.

The Geometric Gas extends the Euclidean Gas (euclidean_gas.py) by adding three
adaptive mechanisms:
1. Adaptive Force: fitness-driven force from �-localized fitness potential
2. Viscous Coupling: fluid-like interaction between walkers
3. Hessian Diffusion: information-geometric diffusion tensor

All tensors are vectorized with the first dimension being the number of alive walkers k.
"""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field
import torch
from torch import Tensor

from fragile.bounds import TorchBounds
from fragile.euclidean_gas import (
    CloningOperator,
    CloningParams,
    EuclideanGasParams,
    KineticOperator,
    LangevinParams,
    PotentialParams,
    SwarmState,
)


class LocalizationKernelParams(BaseModel):
    """Parameters for the �-localized measurement pipeline.

    Mathematical notation from 11_geometric_gas.md � 1.0.2:
    - rho (�): Localization scale controlling spatial extent of measurements
    - kernel_type: Type of localization kernel (gaussian, uniform)
    """

    model_config = {"arbitrary_types_allowed": True}

    rho: float = Field(gt=0, description="Localization scale (�)")
    kernel_type: str = Field("gaussian", description="Kernel type: 'gaussian' or 'uniform'")


class AdaptiveParams(BaseModel):
    """Parameters for adaptive mechanisms in the Geometric Gas.

    Mathematical notation from 11_geometric_gas.md:
    - epsilon_F (�_F): Adaptation rate controlling adaptive force strength
    - nu (�): Viscous coupling coefficient
    - epsilon_Sigma (�_�): Hessian regularization parameter
    - measurement_fn: Function d: X �  measuring local objective quality
    - rescale_amplitude (A): Amplitude of rescale function g_A
    - sigma_var_min (�'_min): Minimum variance for Z-score regularization
    """

    model_config = {"arbitrary_types_allowed": True}

    epsilon_F: float = Field(ge=0, description="Adaptation rate (�_F)")
    nu: float = Field(ge=0, description="Viscous coupling coefficient (�)")
    epsilon_Sigma: float = Field(gt=0, description="Hessian regularization (�_�)")
    rescale_amplitude: float = Field(gt=0, description="Rescale function amplitude (A)")
    sigma_var_min: float = Field(gt=0, description="Minimum variance regularization (�'_min)")
    viscous_length_scale: float = Field(gt=0, description="Length scale for viscous kernel")

    def critical_threshold(self, rho: float, kappa_backbone: float) -> float:
        """Compute critical adaptation rate threshold �_F*(�).

        From 11_geometric_gas.md � 7: Critical threshold formula
        �_F*(�) = (�_backbone - C_diff,1(�)) / (2 K_F(�))

        For now we return a heuristic based on �.
        Full implementation requires C_diff,1(�) and K_F(�) from Appendix A.
        """
        # Heuristic: smaller � requires smaller �_F
        K_F_approx = 1.0 / (1.0 + rho)  # Decreases with �
        C_diff_1_approx = 0.1 / rho if rho > 0.1 else 1.0
        return max(0.01, (kappa_backbone - C_diff_1_approx) / (2 * K_F_approx))


class GeometricGasParams(BaseModel):
    """Complete parameter set for Geometric Gas algorithm.

    Extends EuclideanGasParams with:
    - localization: �-parameterized measurement pipeline
    - adaptive: adaptive force, viscous coupling, Hessian diffusion
    """

    model_config = {"arbitrary_types_allowed": True}

    # Base Euclidean Gas parameters
    N: int = Field(gt=0, description="Number of walkers")
    d: int = Field(gt=0, description="Spatial dimension")
    potential: PotentialParams = Field(description="Target potential parameters")
    langevin: LangevinParams = Field(description="Langevin dynamics parameters")
    cloning: CloningParams = Field(
        default_factory=lambda: CloningParams(
            sigma_x=0.1,
            lambda_alg=0.0,
            epsilon_c=0.1,
            companion_selection_method="hybrid",
            alpha_restitution=0.5,
            use_inelastic_collision=True,
        ),
        description="Cloning operator parameters",
    )

    # Geometric Gas extensions
    localization: LocalizationKernelParams = Field(description="Localization kernel parameters")
    adaptive: AdaptiveParams = Field(description="Adaptive mechanism parameters")

    # Optional bounds and device settings
    bounds: TorchBounds | None = Field(None, description="Position bounds (optional)")
    device: str = Field("cpu", description="PyTorch device (cpu/cuda)")
    dtype: str = Field("float32", description="PyTorch dtype (float32/float64)")
    freeze_best: bool = Field(
        False, description="Keep highest-fitness walker untouched during updates"
    )

    @property
    def torch_dtype(self) -> torch.dtype:
        """Convert dtype string to torch dtype."""
        return torch.float64 if self.dtype == "float64" else torch.float32

    def to_euclidean_params(self) -> EuclideanGasParams:
        """Convert to EuclideanGasParams for backbone compatibility."""
        return EuclideanGasParams(
            N=self.N,
            d=self.d,
            potential=self.potential,
            langevin=self.langevin,
            cloning=self.cloning,
            bounds=self.bounds,
            device=self.device,
            dtype=self.dtype,
            freeze_best=self.freeze_best,
        )


class LocalizationKernel:
    """Implements the �-localized measurement pipeline.

    From 11_geometric_gas.md � 1.0.2:
    The localization kernel K_�(x, x') weights the contribution of walker at
    position x' when computing statistics for walker at position x.
    """

    def __init__(self, params: LocalizationKernelParams, device: torch.device, dtype: torch.dtype):
        self.params = params
        self.device = device
        self.dtype = dtype

    def compute_kernel(self, x: Tensor, x_alive: Tensor) -> Tensor:
        """Compute localization kernel K_�(x_i, x_j) for all alive walkers.

        Args:
            x: Query positions [N_query, d]
            x_alive: Alive walker positions [k, d]

        Returns:
            Kernel matrix [N_query, k]
        """
        if self.params.kernel_type == "gaussian":
            return self._gaussian_kernel(x, x_alive)
        if self.params.kernel_type == "uniform":
            return self._uniform_kernel(x, x_alive)
        msg = f"Unknown kernel type: {self.params.kernel_type}"
        raise ValueError(msg)

    def _gaussian_kernel(self, x: Tensor, x_alive: Tensor) -> Tensor:
        """Gaussian localization kernel.

        K_�(x, x') = (1/Z_�(x)) * exp(-||x - x'||� / (2��))

        Args:
            x: Query positions [N_query, d]
            x_alive: Alive walker positions [k, d]

        Returns:
            Normalized kernel weights [N_query, k]
        """
        # Compute pairwise squared distances [N_query, k]
        diff = x.unsqueeze(1) - x_alive.unsqueeze(0)  # [N_query, k, d]
        sq_dist = torch.sum(diff**2, dim=-1)  # [N_query, k]

        # Gaussian kernel (unnormalized)
        rho_sq = self.params.rho**2
        kernel = torch.exp(-sq_dist / (2 * rho_sq))  # [N_query, k]

        # Normalize: sum over alive walkers equals 1
        Z = kernel.sum(dim=1, keepdim=True)  # [N_query, 1]
        return kernel / (Z + 1e-10)  # [N_query, k]

    def _uniform_kernel(self, x: Tensor, x_alive: Tensor) -> Tensor:
        """Uniform kernel (global limit � � ).

        Args:
            x: Query positions [N_query, d]
            x_alive: Alive walker positions [k, d]

        Returns:
            Uniform weights [N_query, k] = 1/k for all entries
        """
        N_query = x.shape[0]
        k = x_alive.shape[0]
        return torch.ones(N_query, k, device=self.device, dtype=self.dtype) / k

    def compute_localized_moments(
        self, x_query: Tensor, x_alive: Tensor, measurement: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute �-localized mean and variance.

        From 11_geometric_gas.md � 1.0.3:
        �_�[f_k, d, x] = �_j w_ij(�) d(x_j)
        ò_�[f_k, d, x] = �_j w_ij(�) [d(x_j) - �_�]�

        Args:
            x_query: Query positions [N_query, d]
            x_alive: Alive walker positions [k, d]
            measurement: Measurement values at alive walkers [k]

        Returns:
            (mu_rho, sigma_sq_rho): Localized moments [N_query], [N_query]
        """
        # Compute normalized localization weights [N_query, k]
        weights = self.compute_kernel(x_query, x_alive)

        # Localized mean: �_� = �_j w_ij d_j
        mu_rho = torch.sum(weights * measurement.unsqueeze(0), dim=1)  # [N_query]

        # Localized variance: ò_� = �_j w_ij (d_j - �_�)�
        diff = measurement.unsqueeze(0) - mu_rho.unsqueeze(1)  # [N_query, k]
        sigma_sq_rho = torch.sum(weights * diff**2, dim=1)  # [N_query]

        return mu_rho, sigma_sq_rho


class FitnessPotential:
    """Implements the �-localized fitness potential V_fit[f_k, �].

    From 11_geometric_gas.md � 2.1:
    V_fit[f, �](x) = g_A(Z_�[f, d, x])

    where Z_� is the unified regularized Z-score.
    """

    def __init__(
        self,
        localization: LocalizationKernel,
        adaptive_params: AdaptiveParams,
        measurement_fn: Callable[[Tensor], Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.localization = localization
        self.params = adaptive_params
        self.measurement_fn = measurement_fn
        self.device = device
        self.dtype = dtype

    def _rescale_function(self, z: Tensor) -> Tensor:
        """Rescale function g_A:  � [0, A].

        Using sigmoid: g_A(z) = A / (1 + exp(-z))
        """
        A = self.params.rescale_amplitude
        return A / (1.0 + torch.exp(-z))

    def _rescale_derivative(self, z: Tensor) -> Tensor:
        """Derivative of rescale function g'_A(z)."""
        A = self.params.rescale_amplitude
        exp_neg_z = torch.exp(-z)
        return A * exp_neg_z / ((1.0 + exp_neg_z) ** 2)

    def compute_z_score(
        self,
        x_query: Tensor,
        x_alive: Tensor,
        alive_mask: Tensor | None = None,  # noqa: ARG002
    ) -> Tensor:
        """Compute unified regularized Z-score Z_�[f_k, d, x].

        From 11_geometric_gas.md � 1.0.4:
        Z_�[f, d, x] = (d(x) - �_�[f, d, x]) / �'_�[f, d, x]

        where �'_� = sqrt(ò_� + �'�_min) for regularization.

        Args:
            x_query: Query positions [N_query, d]
            x_alive: Alive walker positions [k, d]
            alive_mask: Boolean mask for alive walkers [N_total] (optional)

        Returns:
            Z-scores [N_query]
        """
        # Compute measurements at all positions
        measurement_alive = self.measurement_fn(x_alive)  # [k]
        measurement_query = self.measurement_fn(x_query)  # [N_query]

        # Compute localized moments
        mu_rho, sigma_sq_rho = self.localization.compute_localized_moments(
            x_query, x_alive, measurement_alive
        )

        # Regularized standard deviation
        sigma_min_sq = self.params.sigma_var_min**2
        sigma_prime = torch.sqrt(sigma_sq_rho + sigma_min_sq)  # [N_query]

        # Z-score
        return (measurement_query - mu_rho) / sigma_prime  # [N_query]

    def evaluate(
        self, x_query: Tensor, x_alive: Tensor, alive_mask: Tensor | None = None
    ) -> Tensor:
        """Evaluate fitness potential V_fit[f_k, �](x).

        Args:
            x_query: Query positions [N_query, d]
            x_alive: Alive walker positions [k, d]
            alive_mask: Boolean mask for alive walkers [N_total] (optional)

        Returns:
            Fitness values [N_query]
        """
        z_score = self.compute_z_score(x_query, x_alive, alive_mask)
        return self._rescale_function(z_score)

    def compute_gradient(
        self, x_query: Tensor, x_alive: Tensor, alive_mask: Tensor | None = None
    ) -> Tensor:
        """Compute gradient V_fit using PyTorch autodiff.

        Args:
            x_query: Query positions [N_query, d] (requires grad)
            x_alive: Alive walker positions [k, d]
            alive_mask: Boolean mask for alive walkers [N_total] (optional)

        Returns:
            Gradient [N_query, d]
        """
        x_query.requires_grad_(True)
        V_fit = self.evaluate(x_query, x_alive, alive_mask)  # [N_query]

        # Compute gradient for each query point
        return torch.autograd.grad(
            V_fit.sum(),
            x_query,
            create_graph=True,  # For second derivatives if needed
        )[0]  # [N_query, d]

    def compute_hessian(
        self, x_query: Tensor, x_alive: Tensor, alive_mask: Tensor | None = None
    ) -> Tensor:
        """Compute Hessian �V_fit for each query point.

        Args:
            x_query: Query positions [N_query, d]
            x_alive: Alive walker positions [k, d]
            alive_mask: Boolean mask for alive walkers [N_total] (optional)

        Returns:
            Hessian [N_query, d, d]
        """
        N_query, d = x_query.shape
        x_query.requires_grad_(True)

        # Compute gradient
        grad = self.compute_gradient(x_query, x_alive, alive_mask)  # [N_query, d]

        # Compute Hessian for each component
        hessian = torch.zeros(N_query, d, d, device=self.device, dtype=self.dtype)

        for i in range(d):
            # Compute second derivative w.r.t. each dimension
            grad_i = grad[:, i]  # [N_query]
            hess_i = torch.autograd.grad(
                grad_i.sum(), x_query, create_graph=False, retain_graph=(i < d - 1)
            )[0]  # [N_query, d]
            hessian[:, i, :] = hess_i

        return hessian


class AdaptiveKineticOperator(KineticOperator):
    """Adaptive kinetic operator with geometric mechanisms.

    Extends the base BAOAB integrator from EuclideanGas with:
    1. Adaptive force: �_F V_fit[f_k, �]
    2. Viscous coupling: � F_viscous
    3. Regularized Hessian diffusion: �_reg = (H + �_� I)^(-1/2)
    """

    def __init__(
        self,
        params: LangevinParams,
        potential: PotentialParams,
        fitness_potential: FitnessPotential,
        adaptive_params: AdaptiveParams,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__(params, potential, device, dtype)
        self.fitness_potential = fitness_potential
        self.adaptive_params = adaptive_params

    def _viscous_force(self, state: SwarmState, alive_mask: Tensor) -> Tensor:
        """Compute viscous coupling force.

        From 11_geometric_gas.md � 2.0.3:
        F_viscous,i = � �_j [K(||x_i - x_j||) / deg(i)] (v_j - v_i)

        Args:
            state: Current swarm state
            alive_mask: Boolean mask for alive walkers [N]

        Returns:
            Viscous force [k, d] where k = number of alive walkers
        """
        # Extract alive walkers
        x_alive = state.x[alive_mask]  # [k, d]
        v_alive = state.v[alive_mask]  # [k, d]
        k = x_alive.shape[0]

        if k < 2:
            # No interaction with less than 2 walkers
            return torch.zeros_like(v_alive)

        # Compute pairwise distances
        diff = x_alive.unsqueeze(1) - x_alive.unsqueeze(0)  # [k, k, d]
        dist = torch.norm(diff, dim=-1)  # [k, k]

        # Gaussian viscous kernel
        ell_sq = self.adaptive_params.viscous_length_scale**2
        K = torch.exp(-(dist**2) / (2 * ell_sq))  # [k, k]

        # Set diagonal to zero (no self-interaction)
        K *= 1.0 - torch.eye(k, device=self.device, dtype=self.dtype)

        # Degree normalization
        deg = K.sum(dim=1, keepdim=True)  # [k, 1]
        deg = torch.clamp(deg, min=1e-6)  # Avoid division by zero

        # Viscous force: F_i = Sum_j [K_ij / deg_i] (v_j - v_i)
        # = (1/deg_i) Sum_j K_ij v_j - v_i
        weighted_v = torch.matmul(K, v_alive) / deg  # [k, d]
        return self.adaptive_params.nu * (weighted_v - v_alive)  # [k, d]

    def _compute_diffusion_tensor(
        self, x_alive: Tensor, x_all_alive: Tensor, alive_mask: Tensor
    ) -> Tensor:
        """Compute regularized diffusion tensor �_reg = (H + �_� I)^(-1/2).

        From 11_geometric_gas.md � 2.0.4:
        The Hessian H = �V_fit is regularized and inverted.

        Args:
            x_alive: Positions to compute diffusion at [k, d]
            x_all_alive: All alive walker positions [k, d]
            alive_mask: Boolean mask for alive walkers [N]

        Returns:
            Diffusion tensor [k, d, d]
        """
        _k, d = x_alive.shape

        # Compute Hessian of fitness potential
        H = self.fitness_potential.compute_hessian(x_alive, x_all_alive, alive_mask)  # [k, d, d]

        # Regularize: H_reg = H + epsilon_Sigma * I
        eps_I = self.adaptive_params.epsilon_Sigma * torch.eye(
            d, device=self.device, dtype=self.dtype
        )
        H_reg = H + eps_I.unsqueeze(0)  # [k, d, d]

        # Compute G_reg = (H_reg)^(-1)
        G_reg = torch.linalg.inv(H_reg)  # [k, d, d]

        # Compute Sigma_reg = G_reg^(1/2) using eigendecomposition
        # G = Q Lambda Q^T => G^(1/2) = Q Lambda^(1/2) Q^T
        eigenvalues, eigenvectors = torch.linalg.eigh(G_reg)  # [k, d], [k, d, d]

        # Ensure positive eigenvalues (numerical safety)
        eigenvalues = torch.clamp(eigenvalues, min=1e-8)

        # Sigma_reg = Q diag(sqrt(Lambda)) Q^T
        sqrt_diag = torch.diag_embed(torch.sqrt(eigenvalues))  # [k, d, d]
        return torch.matmul(
            torch.matmul(eigenvectors, sqrt_diag), eigenvectors.transpose(-2, -1)
        )  # [k, d, d]

    def apply(self, state: SwarmState, alive_mask: Tensor | None = None) -> SwarmState:
        """Apply adaptive BAOAB integrator with geometric mechanisms.

        The integration follows the backbone BAOAB structure but adds:
        - Adaptive force in B steps
        - Viscous coupling in velocity updates
        - Anisotropic diffusion in O step

        Args:
            state: Current swarm state
            alive_mask: Boolean mask for alive walkers [N] (optional)

        Returns:
            Updated state after integration
        """
        N, d = state.N, state.d

        # If no alive mask provided, all walkers are alive
        if alive_mask is None:
            alive_mask = torch.ones(N, dtype=torch.bool, device=self.device)

        # Extract alive walkers
        x_alive = state.x[alive_mask].clone()  # [k, d]
        v_alive = state.v[alive_mask].clone()  # [k, d]
        k = x_alive.shape[0]

        if k == 0:
            # No alive walkers, return unchanged state
            return state

        # Compute adaptive force
        F_adapt = torch.zeros_like(v_alive)  # [k, d]
        if self.adaptive_params.epsilon_F > 0:
            grad_V_fit = self.fitness_potential.compute_gradient(
                x_alive, x_alive, alive_mask
            )  # [k, d]
            F_adapt = self.adaptive_params.epsilon_F * grad_V_fit

        # Compute viscous force
        F_viscous = self._viscous_force(
            SwarmState(x_alive, v_alive), torch.ones(k, dtype=torch.bool, device=self.device)
        )  # [k, d]

        # First B step: v � v - (�t/2) * [U(x) - F_adapt - F_viscous]
        x_alive.requires_grad_(True)
        U = self.potential.evaluate(x_alive)  # [k]
        grad_U = torch.autograd.grad(U.sum(), x_alive, create_graph=False)[0]  # [k, d]
        v_alive -= self.dt / 2 * (grad_U - F_adapt - F_viscous)

        # First A step: x -> x + (dt/2) * v
        x_alive += self.dt / 2 * v_alive

        # O step with adaptive diffusion
        if self.adaptive_params.epsilon_Sigma > 0 and self.adaptive_params.epsilon_F > 0:
            # Use regularized Hessian diffusion
            Sigma_reg = self._compute_diffusion_tensor(
                x_alive, x_alive, torch.ones(k, dtype=torch.bool, device=self.device)
            )  # [k, d, d]

            # Ornstein-Uhlenbeck with anisotropic noise
            xi = torch.randn(k, d, device=self.device, dtype=self.dtype)  # [k, d]
            # Sigma_reg @ xi for each walker
            anisotropic_noise = torch.einsum("kij,kj->ki", Sigma_reg, xi)  # [k, d]
            v_alive = self.c1 * v_alive + self.c2 * anisotropic_noise
        else:
            # Standard isotropic noise (backbone)
            xi = torch.randn(k, d, device=self.device, dtype=self.dtype)
            v_alive = self.c1 * v_alive + self.c2 * xi

        # Second A step: x � x + (�t/2) * v
        x_alive += self.dt / 2 * v_alive

        # Second B step: v � v - (�t/2) * [U(x) - F_adapt - F_viscous]
        # Recompute forces at new positions
        F_adapt = torch.zeros_like(v_alive)
        if self.adaptive_params.epsilon_F > 0:
            grad_V_fit = self.fitness_potential.compute_gradient(x_alive, x_alive, alive_mask)
            F_adapt = self.adaptive_params.epsilon_F * grad_V_fit

        F_viscous = self._viscous_force(
            SwarmState(x_alive, v_alive), torch.ones(k, dtype=torch.bool, device=self.device)
        )

        x_temp = x_alive.detach().clone().requires_grad_(True)
        U = self.potential.evaluate(x_temp)
        grad_U = torch.autograd.grad(U.sum(), x_temp, create_graph=False)[0]
        v_alive -= self.dt / 2 * (grad_U - F_adapt - F_viscous)

        # Update state with new alive walker values
        new_x = state.x.clone()
        new_v = state.v.clone()
        new_x[alive_mask] = x_alive
        new_v[alive_mask] = v_alive

        return SwarmState(new_x, new_v)


class GeometricGas:
    """Geometric Viscous Fluid Model implementation.

    Extends EuclideanGas with �-localized adaptive mechanisms:
    - Localization kernel for spatial statistics
    - Adaptive force from fitness potential
    - Viscous coupling between walkers
    - Regularized Hessian diffusion

    From docs/source/2_geometric_gas/11_geometric_gas.md
    """

    def __init__(
        self, params: GeometricGasParams, measurement_fn: Callable[[Tensor], Tensor] | None = None
    ):
        """Initialize Geometric Gas.

        Args:
            params: Complete parameter set
            measurement_fn: Measurement function d: X �  (optional, defaults to potential)
        """
        self.params = params
        self.device = torch.device(params.device)
        self.dtype = params.torch_dtype
        self.bounds = params.bounds

        # Default measurement: negative potential (higher is better)
        if measurement_fn is None:

            def default_measurement(x: Tensor) -> Tensor:
                return -params.potential.evaluate(x)

            self.measurement_fn = default_measurement
        else:
            self.measurement_fn = measurement_fn

        # Initialize localization kernel
        self.localization = LocalizationKernel(params.localization, self.device, self.dtype)

        # Initialize fitness potential
        self.fitness_potential = FitnessPotential(
            self.localization, params.adaptive, self.measurement_fn, self.device, self.dtype
        )

        # Initialize adaptive kinetic operator
        self.kinetic_op = AdaptiveKineticOperator(
            params.langevin,
            params.potential,
            self.fitness_potential,
            params.adaptive,
            self.device,
            self.dtype,
        )

        # Initialize cloning operator (with boundary enforcement)
        self.cloning_op = CloningOperator(
            params.cloning, self.device, self.dtype, bounds=self.bounds
        )

    def initialize_state(
        self, x_init: Tensor | None = None, v_init: Tensor | None = None
    ) -> SwarmState:
        """Initialize swarm state.

        Args:
            x_init: Initial positions [N, d] (optional, defaults to N(0, I))
            v_init: Initial velocities [N, d] (optional, defaults to N(0, I/β))

        Returns:
            Initial swarm state
        """
        N, d = self.params.N, self.params.d

        if x_init is None:
            x_init = torch.randn(N, d, device=self.device, dtype=self.dtype)

        if v_init is None:
            # Initialize velocities from thermal distribution
            v_std = 1.0 / torch.sqrt(torch.tensor(self.params.langevin.beta, dtype=self.dtype))
            v_init = v_std * torch.randn(N, d, device=self.device, dtype=self.dtype)

        return SwarmState(
            x_init.to(device=self.device, dtype=self.dtype),
            v_init.to(device=self.device, dtype=self.dtype),
        )

    def _freeze_mask(self, state: SwarmState) -> Tensor | None:
        """Get mask for walkers to freeze (if freeze_best enabled)."""
        if not self.params.freeze_best:
            return None

        # Use fitness as reward for selecting best walker
        if self.bounds is not None:
            alive_mask = self.bounds.contains(state.x)
        else:
            alive_mask = torch.ones(state.N, dtype=torch.bool, device=self.device)

        if not alive_mask.any():
            return None

        # Compute fitness for alive walkers
        fitness = torch.full((state.N,), float("-inf"), device=self.device, dtype=self.dtype)
        x_alive = state.x[alive_mask]
        fitness[alive_mask] = self.fitness_potential.evaluate(x_alive, x_alive, alive_mask)

        # Select best walker
        best_idx = torch.argmax(fitness)
        mask = torch.zeros(state.N, device=self.device, dtype=torch.bool)
        mask[best_idx] = True
        return mask

    def step(
        self, state: SwarmState, alive_mask: Tensor | None = None
    ) -> tuple[SwarmState, SwarmState]:
        """Perform one full step: cloning followed by adaptive kinetic.

        Args:
            state: Current swarm state
            alive_mask: Boolean mask for alive walkers [N] (optional)

        Returns:
            Tuple of (state_after_cloning, state_after_kinetic)
        """
        # Determine alive mask
        if alive_mask is None:
            if self.bounds is not None:
                alive_mask = self.bounds.contains(state.x)
            else:
                alive_mask = torch.ones(state.N, dtype=torch.bool, device=self.device)

        # Freeze best walker if requested
        freeze_mask = self._freeze_mask(state)
        reference_state = state.clone() if freeze_mask is not None else None

        # Apply cloning operator (handles boundary enforcement and resurrection)
        state_cloned = self.cloning_op.apply(state)

        if freeze_mask is not None and freeze_mask.any():
            state_cloned.copy_from(reference_state, freeze_mask)

        # Apply adaptive kinetic operator
        state_final = self.kinetic_op.apply(state_cloned, alive_mask)

        if freeze_mask is not None and freeze_mask.any():
            state_final.copy_from(reference_state, freeze_mask)

        return state_cloned, state_final

    def run(
        self,
        n_steps: int,
        x_init: Tensor | None = None,
        v_init: Tensor | None = None,
        fractal_set: FractalSet | None = None,
        record_fitness: bool = False,
    ) -> dict:
        """Run Geometric Gas for multiple steps.

        Args:
            n_steps: Number of steps to run
            x_init: Initial positions (optional)
            v_init: Initial velocities (optional)
            fractal_set: Optional FractalSet instance to record simulation data.
                        If provided, all timesteps will be recorded in the graph.
            record_fitness: If True and fractal_set is provided, compute and record
                           fitness, potential, and reward at each step.

        Returns:
            Dictionary with trajectory data (same format as EuclideanGas)
        """
        state = self.initialize_state(x_init, v_init)

        # Preallocate storage
        N, d = state.N, state.d
        x_traj = torch.zeros(n_steps + 1, N, d, device=self.device, dtype=self.dtype)
        v_traj = torch.zeros(n_steps + 1, N, d, device=self.device, dtype=self.dtype)
        fitness_traj = torch.zeros(n_steps + 1, N, device=self.device, dtype=self.dtype)

        # Store initial state
        x_traj[0] = state.x
        v_traj[0] = state.v

        # Compute initial fitness
        if self.bounds is not None:
            alive_mask = self.bounds.contains(state.x)
        else:
            alive_mask = torch.ones(N, dtype=torch.bool, device=self.device)

        if alive_mask.any():
            fitness_traj[0, alive_mask] = self.fitness_potential.evaluate(
                state.x[alive_mask], state.x[alive_mask], alive_mask
            )

        # Record initial state in FractalSet if provided
        if fractal_set is not None:
            self._record_fractal_set_timestep(
                fractal_set=fractal_set,
                state=state,
                timestep=0,
                alive_mask=alive_mask,
                record_fitness=record_fitness,
            )

        # Run steps
        for t in range(n_steps):
            # Update alive mask
            if self.bounds is not None:
                alive_mask = self.bounds.contains(state.x)
                if not alive_mask.any():
                    # All walkers dead, stop early
                    return {
                        "x": x_traj[: t + 1],
                        "v": v_traj[: t + 1],
                        "fitness": fitness_traj[: t + 1],
                        "terminated_early": True,
                        "final_step": t,
                    }

            # Perform step
            _, state = self.step(state, alive_mask)

            # Store new state
            x_traj[t + 1] = state.x
            v_traj[t + 1] = state.v

            # Compute fitness
            if alive_mask.any():
                fitness_traj[t + 1, alive_mask] = self.fitness_potential.evaluate(
                    state.x[alive_mask], state.x[alive_mask], alive_mask
                )

            # Record in FractalSet if provided
            if fractal_set is not None:
                self._record_fractal_set_timestep(
                    fractal_set=fractal_set,
                    state=state,
                    timestep=t + 1,
                    alive_mask=alive_mask,
                    record_fitness=record_fitness,
                )

        result = {
            "x": x_traj,
            "v": v_traj,
            "fitness": fitness_traj,
            "terminated_early": False,
            "final_step": n_steps,
        }

        # Include FractalSet in result if provided
        if fractal_set is not None:
            result["fractal_set"] = fractal_set

        return result

    def _record_fractal_set_timestep(
        self,
        fractal_set: FractalSet,
        state: SwarmState,
        timestep: int,
        alive_mask: Tensor,
        record_fitness: bool,
    ) -> None:
        """
        Record current timestep data into FractalSet.

        Args:
            fractal_set: FractalSet instance to record into
            state: Current swarm state
            timestep: Current timestep index
            alive_mask: Boolean mask of alive walkers [N]
            record_fitness: Whether to compute and record fitness metrics
        """
        # Compute high-error mask based on positional error
        mu_x = torch.mean(state.x, dim=0, keepdim=True)
        positional_error = torch.sqrt(torch.sum((state.x - mu_x) ** 2, dim=-1))
        threshold = torch.median(positional_error)
        high_error_mask = positional_error > threshold

        # Compute fitness-related metrics if requested
        if record_fitness:
            # Potential and reward
            potential = self.params.potential.evaluate(state.x)
            reward = -potential

            # For GeometricGas, fitness is computed via localized fitness potential
            # Compute fitness for alive walkers
            if alive_mask.any():
                fitness_vals = torch.zeros(state.N, device=self.device, dtype=self.dtype)
                fitness_vals[alive_mask] = self.fitness_potential.evaluate(
                    state.x[alive_mask], state.x[alive_mask], alive_mask
                )
                fitness = fitness_vals
            else:
                fitness = torch.zeros(state.N, device=self.device, dtype=self.dtype)

            # Note: GeometricGas doesn't use cloning in the same way as EuclideanGas
            # So we don't compute companions, distances, and cloning_probs
            companions = None
            distances = None
            cloning_probs = None

            # Compute rescaled reward (normalized by mean)
            reward_mean = reward.mean()
            reward_std = reward.std() + 1e-8
            rescaled_reward = (reward - reward_mean) / reward_std

            # GeometricGas doesn't use algorithmic distance, so rescaled_distance is None
            rescaled_distance = None

            # No cloning uniform sample for GeometricGas
            clone_uniform_sample = None
        else:
            potential = None
            reward = None
            fitness = None
            companions = None
            distances = None
            cloning_probs = None
            rescaled_reward = None
            rescaled_distance = None
            clone_uniform_sample = None

        # Record timestep in FractalSet
        fractal_set.add_timestep(
            state=state,
            timestep=timestep,
            high_error_mask=high_error_mask,
            alive_mask=alive_mask,
            fitness=fitness,
            potential=potential,
            reward=reward,
            companions=companions,
            cloning_probs=cloning_probs,
            distances=distances,
            rescaled_reward=rescaled_reward,
            rescaled_distance=rescaled_distance,
            clone_uniform_sample=clone_uniform_sample,
        )
