"""
Kinetic Operator: Langevin Dynamics with BAOAB / Boris-BAOAB Integrator

This module implements the kinetic operator for the Euclidean Gas algorithm,
providing Langevin dynamics integration using the BAOAB scheme.

Mathematical notation:
- gamma (γ): Friction coefficient
- beta (β): Inverse temperature 1/(k_B T)
- delta_t (Δt): Time step size
"""

from __future__ import annotations

import warnings

import panel as pn
import param
import torch
from torch import Tensor

from fragile.fractalai.bounds import TorchBounds
from fragile.fractalai.core.distance import compute_periodic_distance_matrix
from fragile.fractalai.core.panel_model import INPUT_WIDTH, PanelModel


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
    for one-step minorization mathster and deriving continuum limits.

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


class KineticOperator(PanelModel):
    """Kinetic operator using BAOAB integrator for Langevin dynamics.

    Supports adaptive extensions from the Geometric Viscous Fluid Model:
    - Fitness-based force: -ε_F · ∇V_fit,i (optional)
    - Anisotropic diffusion: Σ_reg = (∇²V_fit,i + ε_Σ I)^{-1/2} (optional)
    - Viscous coupling: ν · F_viscous for fluid-like collective behavior (optional)
    - Boris rotation: curl-driven velocity rotation for non-conservative reward fields (optional)

    Mathematical notation:
    - gamma (γ): Friction coefficient
    - beta (β): Inverse temperature 1/(k_B T)
    - delta_t (Δt): Time step size
    - epsilon_F (ε_F): Adaptation rate for fitness force
    - epsilon_Sigma (ε_Σ): Hessian regularization parameter
    - nu (ν): Viscous coupling strength
    - beta_curl (β_curl): Curl coupling strength for Boris rotation

    Reference: docs/source/2_geometric_gas/11_geometric_gas.md
    """

    _n_widget_columns = param.Integer(default=2, bounds=(1, None), doc="Number of widget columns")
    _max_widget_width = param.Integer(default=800, bounds=(0, None), doc="Maximum widget width")

    # Standard Langevin parameters
    gamma = param.Number(
        default=1.0,
        bounds=(0, None),
        softbounds=(0.05, 5.0),
        inclusive_bounds=(False, True),
        doc="Friction coefficient (γ)",
    )
    beta = param.Number(
        default=1.0,
        bounds=(0, None),
        softbounds=(0.01, 10.0),
        inclusive_bounds=(False, True),
        doc="Inverse temperature 1/(k_B T) (β)",
    )
    delta_t = param.Number(
        default=0.01,
        bounds=(0, None),
        softbounds=(0.01, 0.1),
        inclusive_bounds=(False, True),
        doc="Time step size (Δt)",
    )
    integrator = param.Selector(
        default="baoab",
        objects=["baoab", "boris-baoab"],
        doc="Integration scheme (baoab or boris-baoab)",
    )

    # Fitness-based adaptive force (Geometric Gas extension)
    epsilon_F = param.Number(
        default=0.0,
        bounds=(0, None),
        softbounds=(0.0, 0.5),
        doc="Adaptation rate for fitness force (ε_F)",
    )
    use_fitness_force = param.Boolean(
        default=False, doc="Enable fitness-based force -ε_F · ∇V_fit,i"
    )
    use_potential_force = param.Boolean(default=True, doc="Enable potential gradient force -∇U(x)")

    # Anisotropic diffusion tensor (Hessian-based)
    epsilon_Sigma = param.Number(
        default=0.1, bounds=(0, None), doc="Hessian regularization (ε_Σ) for positive definiteness"
    )
    use_anisotropic_diffusion = param.Boolean(
        default=False, doc="Enable Hessian-based anisotropic diffusion Σ_reg"
    )
    diagonal_diffusion = param.Boolean(
        default=True, doc="Use diagonal-only diffusion (faster, O(Nd) vs O(Nd²))"
    )

    # Viscous coupling (velocity-dependent damping)
    nu = param.Number(default=0.0, bounds=(0, None), doc="Viscous coupling strength (ν)")
    use_viscous_coupling = param.Boolean(
        default=False, doc="Enable viscous coupling for fluid-like behavior"
    )
    viscous_length_scale = param.Number(
        default=1.0,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        doc="Length scale (l) for Gaussian kernel K(r) = exp(-r²/(2l²))",
    )
    viscous_neighbor_mode = param.Selector(
        default="all",
        objects=["all", "nearest"],
        doc="Neighbor mode for viscous coupling (all or nearest)",
    )
    viscous_neighbor_threshold = param.Number(
        default=None,
        allow_None=True,
        bounds=(0.0, 1.0),
        inclusive_bounds=(False, True),
        doc="Kernel threshold for strong neighbors in viscous penalty",
    )
    viscous_neighbor_penalty = param.Number(
        default=0.0,
        bounds=(0, None),
        doc="Penalty strength for excess strong neighbors in viscous coupling",
    )
    viscous_degree_cap = param.Number(
        default=None,
        allow_None=True,
        bounds=(0, None),
        doc="Optional cap on viscous degree to saturate multi-neighbor coupling",
    )

    # Boris rotation (curl-driven velocity rotation)
    beta_curl = param.Number(
        default=0.0,
        bounds=(0, None),
        doc="Curl coupling strength (β_curl) for Boris rotation",
    )
    curl_field = param.Parameter(
        default=None,
        doc="Optional curl/2-form field for Boris rotation: callable F(x)->[N, d, d] or [N, 3]",
    )

    # Velocity squashing map
    V_alg = param.Number(
        default=float("inf"),
        bounds=(0, None),
        inclusive_bounds=(False, True),
        doc="Algorithmic velocity bound for smooth squashing map",
    )
    use_velocity_squashing = param.Boolean(
        default=False, doc="Enable smooth velocity squashing map ψ_v"
    )

    @property
    def widgets(self) -> dict[str, dict]:
        """Widget configurations for kinetic operator parameters."""
        return {
            "gamma": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "γ (friction)",
                "start": 0.05,
                "end": 5.0,
                "step": 0.05,
            },
            "beta": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "β (inverse temp)",
                "start": 0.1,
                "end": 5.0,
                "step": 0.05,
            },
            "delta_t": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "Δt (time step)",
                "start": 0.01,
                "end": 0.2,
                "step": 0.005,
            },
            "integrator": {
                "type": pn.widgets.Select,
                "width": INPUT_WIDTH,
                "name": "Integrator",
            },
            "epsilon_F": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "ε_F (fitness adapt)",
                "start": 0.0,
                "end": 0.5,
                "step": 0.01,
            },
            "use_fitness_force": {
                "type": pn.widgets.Checkbox,
                "width": INPUT_WIDTH,
                "name": "Use fitness force",
            },
            "use_potential_force": {
                "type": pn.widgets.Checkbox,
                "width": INPUT_WIDTH,
                "name": "Use potential force",
            },
            "epsilon_Sigma": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "ε_Σ (Hessian reg)",
                "start": 0.0,
                "end": 1.0,
                "step": 0.01,
            },
            "use_anisotropic_diffusion": {
                "type": pn.widgets.Checkbox,
                "width": INPUT_WIDTH,
                "name": "Anisotropic diffusion",
            },
            "diagonal_diffusion": {
                "type": pn.widgets.Checkbox,
                "width": INPUT_WIDTH,
                "name": "Diagonal diffusion",
            },
            "nu": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "ν (viscous coupling)",
                "start": 0.0,
                "end": 10.0,
                "step": 0.1,
            },
            "use_viscous_coupling": {
                "type": pn.widgets.Checkbox,
                "width": INPUT_WIDTH,
                "name": "Use viscous coupling",
            },
            "viscous_length_scale": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "l (viscous length)",
                "start": 0.1,
                "end": 5.0,
                "step": 0.1,
            },
            "viscous_neighbor_mode": {
                "type": pn.widgets.Select,
                "width": INPUT_WIDTH,
                "name": "Viscous neighbors",
            },
            "viscous_neighbor_threshold": {
                "type": pn.widgets.FloatInput,
                "width": INPUT_WIDTH,
                "name": "Viscous neighbor threshold",
                "start": 0.0,
                "end": 1.0,
                "step": 0.01,
                "value": None,
            },
            "viscous_neighbor_penalty": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "Viscous neighbor penalty",
                "start": 0.0,
                "end": 10.0,
                "step": 0.1,
            },
            "viscous_degree_cap": {
                "type": pn.widgets.FloatInput,
                "width": INPUT_WIDTH,
                "name": "Viscous degree cap",
                "start": 0.0,
                "end": 200.0,
                "step": 1.0,
                "value": None,
            },
            "beta_curl": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "β_curl (Boris strength)",
                "start": 0.0,
                "end": 10.0,
                "step": 0.1,
            },
            "V_alg": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "V_alg (velocity bound)",
                "start": 0.1,
                "end": 100.0,
                "step": 1.0,
            },
            "use_velocity_squashing": {
                "type": pn.widgets.Checkbox,
                "width": INPUT_WIDTH,
                "name": "Velocity squashing",
            },
        }

    @property
    def widget_parameters(self) -> list[str]:
        """Parameters to display in UI."""
        return [
            "gamma",
            "beta",
            "delta_t",
            "integrator",
            "epsilon_F",
            "use_fitness_force",
            "use_potential_force",
            "epsilon_Sigma",
            "use_anisotropic_diffusion",
            "diagonal_diffusion",
            "nu",
            "use_viscous_coupling",
            "viscous_length_scale",
            "viscous_neighbor_mode",
            "viscous_neighbor_threshold",
            "viscous_neighbor_penalty",
            "viscous_degree_cap",
            "beta_curl",
            "V_alg",
            "use_velocity_squashing",
        ]

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
        viscous_neighbor_mode: str = "all",
        viscous_neighbor_threshold: float | None = None,
        viscous_neighbor_penalty: float = 0.0,
        viscous_degree_cap: float | None = None,
        beta_curl: float = 0.0,
        curl_field=None,
        V_alg: float = float("inf"),
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
            viscous_neighbor_mode: Neighbor mode for viscous coupling
            viscous_neighbor_threshold: Kernel threshold for strong neighbors
            viscous_neighbor_penalty: Penalty strength for excess neighbors
            viscous_degree_cap: Degree cap for viscous coupling saturation
            beta_curl: Curl coupling strength for Boris rotation
            curl_field: Curl/2-form field; callable F(x)->[N, d, d] or [N, 3]
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
            viscous_neighbor_mode=viscous_neighbor_mode,
            viscous_neighbor_threshold=viscous_neighbor_threshold,
            viscous_neighbor_penalty=viscous_neighbor_penalty,
            viscous_degree_cap=viscous_degree_cap,
            beta_curl=beta_curl,
            curl_field=curl_field,
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

        if self.curl_field is not None and not callable(self.curl_field):
            msg = f"curl_field must be callable, got {type(self.curl_field)}"
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
        - F_fitness = -ε_F · ∇V_fit,i (if use_fitness_force=True)

        Args:
            x: Positions [N, d]
            v: Velocities [N, d]
            grad_fitness: Precomputed per-walker fitness gradient ∇V_fit,i [N, d]
                (required if use_fitness_force=True)

        Returns:
            force: Combined force vector [N, d]

        Raises:
            ValueError: If grad_fitness is None but use_fitness_force=True

        Note:
            Both forces use negative gradients to drift toward lower potential/higher fitness.
        """
        force_stable, force_adapt = self._compute_force_components(x, grad_fitness)
        return force_stable + force_adapt

    def _compute_force_components(
        self,
        x: Tensor,
        grad_fitness: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute stable and adaptive force components separately."""
        force_stable = torch.zeros_like(x)
        force_adapt = torch.zeros_like(x)

        if self.use_potential_force:
            x.requires_grad_(True)  # noqa: FBT003
            U = self.potential(x)  # [N]
            grad_U = torch.autograd.grad(U.sum(), x, create_graph=False)[0]  # [N, d]
            force_stable = -grad_U
            x.requires_grad_(False)  # noqa: FBT003

        if self.use_fitness_force:
            if grad_fitness is None:
                msg = "grad_fitness required when use_fitness_force=True"
                raise ValueError(msg)
            force_adapt = -self.epsilon_F * grad_fitness

        return force_stable, force_adapt

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

        _N, _d = x.shape

        # Compute pairwise distances
        # With PBC: use minimum image convention (wrapping)
        # Without PBC: standard Euclidean distance
        # distances[i, j] = ||x_i - x_j|| (accounting for wrapping if pbc=True)
        distances = compute_periodic_distance_matrix(
            x, y=None, bounds=self.bounds, pbc=self.pbc
        )  # [N, N]

        # Compute Gaussian kernel K(r) = exp(-r²/(2l²))
        l_sq = self.viscous_length_scale**2
        kernel = torch.exp(-(distances**2) / (2 * l_sq))  # [N, N]

        # Zero out diagonal (no self-interaction)
        kernel.fill_diagonal_(0.0)

        if self.viscous_neighbor_mode == "nearest":
            nearest_dist = distances.clone()
            nearest_dist.fill_diagonal_(float("inf"))
            nn_idx = nearest_dist.argmin(dim=1)
            mask = torch.zeros_like(kernel)
            mask.scatter_(1, nn_idx.unsqueeze(1), 1.0)
            kernel = kernel * mask

        # Compute local degree deg(i) = ∑_{j≠i} K(||x_i - x_j||)
        # Add small epsilon for numerical stability
        deg = kernel.sum(dim=1, keepdim=True)  # [N, 1]
        deg = torch.clamp(deg, min=1e-10)  # Avoid division by zero

        # Compute normalized weights w_ij = K_ij / deg_i
        weights = kernel / deg  # [N, N], broadcasting deg over columns
        if self.viscous_degree_cap is not None:
            cap = float(self.viscous_degree_cap)
            if cap <= 0:
                return torch.zeros_like(v)
            scale = torch.clamp(cap / deg, max=1.0)
            weights = weights * scale

        if self.viscous_neighbor_threshold is not None and self.viscous_neighbor_penalty > 0:
            threshold = float(self.viscous_neighbor_threshold)
            if threshold > 0:
                strong = kernel >= threshold
                strong_count = strong.sum(dim=1, keepdim=True).to(weights.dtype)
                excess = torch.clamp(strong_count - 1.0, min=0.0)
                penalty_scale = 1.0 / (1.0 + self.viscous_neighbor_penalty * excess)
                weights = weights * penalty_scale

        # Compute velocity differences for all pairs
        # v_diff[i, j] = v_j - v_i
        v_diff = v.unsqueeze(0) - v.unsqueeze(1)  # [N, N, d]

        # Compute weighted sum: F_visc_i = ν * ∑_j w_ij * (v_j - v_i)
        # This is a batched matrix-vector product
        return self.nu * torch.einsum("ij,ijd->id", weights, v_diff)  # [N, d]

    def _boris_rotate(self, v: Tensor, curl: Tensor) -> Tensor:
        """Apply Boris rotation for curl-driven velocity updates.

        Supports two curl representations:
        - Vector field [N, 3] (3D only), interpreted as a magnetic field.
        - Matrix field [N, d, d], interpreted as a skew-symmetric 2-form.
        """
        if curl.device != v.device or curl.dtype != v.dtype:
            curl = curl.to(device=v.device, dtype=v.dtype)

        if curl.dim() == 2:
            if v.shape[1] != 3:
                msg = "curl_field must return [N, d, d] for d != 3"
                raise ValueError(msg)
            if curl.shape != v.shape:
                msg = f"curl_field returned {tuple(curl.shape)}, expected {tuple(v.shape)}"
                raise ValueError(msg)

            t = 0.5 * self.beta_curl * self.dt * curl
            t_mag_sq = (t * t).sum(dim=-1, keepdim=True)
            s = 2.0 * t / (1.0 + t_mag_sq)
            v_prime = v + torch.cross(v, t, dim=-1)
            return v + torch.cross(v_prime, s, dim=-1)

        if curl.dim() == 3:
            N, d, d2 = curl.shape
            if N != v.shape[0] or d != d2 or d != v.shape[1]:
                msg = f"curl_field returned {tuple(curl.shape)}, expected [N, d, d] = {(v.shape[0], v.shape[1], v.shape[1])}"
                raise ValueError(msg)

            sym = curl + curl.transpose(-1, -2)
            max_sym = sym.abs().max().item()
            if max_sym > 1e-6:
                msg = f"curl_field must be skew-symmetric (2-form); max asymmetry {max_sym:.2e}"
                raise ValueError(msg)

            A = 0.5 * self.beta_curl * self.dt * curl
            eye = torch.eye(d, device=v.device, dtype=v.dtype).expand(N, d, d)
            rhs = torch.bmm(eye + A, v.unsqueeze(-1))
            return torch.linalg.solve(eye - A, rhs).squeeze(-1)

        msg = f"curl_field returned {curl.dim()}-D tensor; expected [N, 3] or [N, d, d]"
        raise ValueError(msg)

    def _apply_boris_kick(
        self,
        x: Tensor,
        v: Tensor,
        grad_fitness: Tensor | None,
    ) -> Tensor:
        """Apply one Boris-aware B step (half kick + rotation + half kick)."""
        force = self._compute_force(x, v, grad_fitness)

        viscous = (
            self._compute_viscous_force(x, v) if self.use_viscous_coupling else torch.zeros_like(v)
        )
        v_minus = v + (self.dt / 2) * (force + viscous)

        if self.curl_field is not None and self.beta_curl > 0:
            curl = self.curl_field(x)
            v_rot = self._boris_rotate(v_minus, curl)
        else:
            v_rot = v_minus

        viscous_rot = (
            self._compute_viscous_force(x, v_rot)
            if self.use_viscous_coupling
            else torch.zeros_like(v)
        )
        return v_rot + (self.dt / 2) * (force + viscous_rot)

    def _compute_diffusion_tensor(
        self,
        x: Tensor,
        hess_fitness: Tensor | None = None,
    ) -> Tensor:
        """Compute anisotropic diffusion tensor σ = c2 · (∇²V_fit,i + ε_Σ I)^{-1/2}.

        The regularized Hessian ensures positive definiteness and provides
        state-dependent noise aligned with the fitness landscape geometry.

        Args:
            x: Positions [N, d]
        hess_fitness: Precomputed per-walker Hessian ∇²V_fit,i
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

            # Σ_diag = c2 * (H_reg)^{-1/2}
            sigma = self.c2 / torch.sqrt(hess_reg)
        else:
            # Full anisotropic case: matrix inverse square root
            # Add ε_Σ I to each [d, d] block
            eye = torch.eye(d, device=x.device, dtype=x.dtype)
            hess_reg = hess + eps_I * eye.unsqueeze(0)  # [N, d, d]
            hess_reg = 0.5 * (hess_reg + hess_reg.transpose(-2, -1))

            # Compute matrix inverse square root via eigendecomposition
            # For symmetric positive definite A: A^{-1/2} = Q Λ^{-1/2} Q^T
            base_jitter = float(max(eps_I, 1e-6))
            eigenvalues = None
            eigenvectors = None
            for attempt in range(3):
                try:
                    if attempt == 0:
                        hess_try = hess_reg
                    else:
                        jitter = base_jitter * (10**attempt)
                        hess_try = hess_reg + jitter * eye.unsqueeze(0)
                    eigenvalues, eigenvectors = torch.linalg.eigh(hess_try)
                    break
                except RuntimeError:
                    continue

            if eigenvalues is None or eigenvectors is None:
                # Fallback: use diagonal approximation to keep simulation stable
                warnings.warn(
                    "Anisotropic diffusion: eigen-decomposition failed after retries; "
                    "falling back to diagonal diffusion for this step.",
                    RuntimeWarning,
                )
                hess_diag = torch.diagonal(hess_reg, dim1=-2, dim2=-1)
                hess_diag = torch.clamp(hess_diag, min=eps_I)
                sigma_diag = self.c2 / torch.sqrt(hess_diag)
                sigma = torch.diag_embed(sigma_diag)
            else:
                # Clamp eigenvalues to ensure positivity (handle numerical errors)
                eigenvalues = torch.clamp(eigenvalues, min=eps_I)

                eigenvalues_inv_sqrt = self.c2 / torch.sqrt(eigenvalues)  # [N, d]

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
        return_info: bool = False,
    ):
        """Apply BAOAB integrator for one time step with optional adaptive features.

        Standard BAOAB sequence:
            B: v → v + (Δt/2) · F(x, v)          [Force step]
            A: x → x + (Δt/2) · v                 [Position update]
            O: v → c1 · v + noise                 [Ornstein-Uhlenbeck]
            A: x → x + (Δt/2) · v                 [Position update]
            B: v → v + (Δt/2) · F(x, v)          [Force step]

        where F(x, v) = -∇U(x) - ε_F · ∇V_fit,i + ν · F_viscous(x, v) and noise can be:
            - Isotropic: c2 · ξ where ξ ~ N(0, I)
            - Anisotropic: c2 Σ_reg · ξ where Σ_reg = (∇²V_fit,i + ε_Σ I)^{-1/2}
            - Optional Boris rotation if curl_field is provided and beta_curl > 0

        Args:
            state: Current swarm state (must have .x and .v attributes)
            grad_fitness: Precomputed per-walker fitness gradient ∇V_fit,i [N, d]
                         (required if use_fitness_force=True)
            hess_fitness: Precomputed per-walker Hessian ∇²V_fit,i
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
        info = {}

        if return_info:
            force_stable, force_adapt = self._compute_force_components(x, grad_fitness)
            force_viscous = (
                self._compute_viscous_force(x, v)
                if self.use_viscous_coupling
                else torch.zeros_like(v)
            )
            force_friction = -self.gamma * v
            force_total = force_stable + force_adapt + force_viscous + force_friction
            info.update({
                "force_stable": force_stable,
                "force_adapt": force_adapt,
                "force_viscous": force_viscous,
                "force_friction": force_friction,
                "force_total": force_total,
            })

        # === FIRST B STEP: Apply forces + optional Boris rotation ===
        v = self._apply_boris_kick(x, v, grad_fitness)

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
            sigma = None
            noise = self.c2 * ξ

        if return_info:
            info["noise"] = noise
            if sigma is not None:
                if self.diagonal_diffusion:
                    info["sigma_reg_diag"] = sigma
                    info["sigma_reg_full"] = None
                else:
                    info["sigma_reg_diag"] = None
                    info["sigma_reg_full"] = sigma
            else:
                info["sigma_reg_diag"] = None
                info["sigma_reg_full"] = None

        # === SECOND A STEP: Update positions ===
        x += (self.dt / 2) * v

        # === SECOND B STEP: Apply forces + optional Boris rotation ===
        v = self._apply_boris_kick(x, v, grad_fitness)

        # === VELOCITY SQUASHING: Apply smooth radial squashing map ===
        if self.use_velocity_squashing:
            v = psi_v(v, self.V_alg)

        # Return state with same type as input
        # Create new state object using the same class as input
        new_state = type(state)(x, v)
        if return_info:
            return new_state, info
        return new_state
