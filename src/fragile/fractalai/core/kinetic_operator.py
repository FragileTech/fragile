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
    n_kinetic_steps = param.Integer(
        default=1,
        bounds=(1, None),
        doc="Number of kinetic substeps per algorithm iteration",
    )
    integrator = param.Selector(
        default="boris-baoab",
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

    # Anisotropic diffusion tensor (Hessian-based or fast proxy)
    epsilon_Sigma = param.Number(
        default=0.1, bounds=(0, None), doc="Hessian regularization (ε_Σ) for positive definiteness"
    )
    use_anisotropic_diffusion = param.Boolean(
        default=True, doc="Enable Hessian-based anisotropic diffusion Σ_reg"
    )
    diffusion_mode = param.Selector(
        default="hessian",
        objects=["hessian", "grad_proxy", "voronoi_proxy"],
        doc="Anisotropic diffusion mode (Hessian, gradient-proxy, or voronoi-proxy).",
    )
    diffusion_grad_scale = param.Number(
        default=1.0,
        bounds=(0, None),
        doc="Scale for gradient-proxy diffusion (multiplies |∇V|).",
    )
    diagonal_diffusion = param.Boolean(
        default=False, doc="Use diagonal-only diffusion (faster, O(Nd) vs O(Nd²))"
    )

    # Viscous coupling (velocity-dependent damping)
    nu = param.Number(default=0.1, bounds=(0, None), doc="Viscous coupling strength (ν)")
    use_viscous_coupling = param.Boolean(
        default=True, doc="Enable viscous coupling for fluid-like behavior"
    )
    viscous_length_scale = param.Number(
        default=1.0,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        doc="Length scale (l) for Gaussian kernel K(r) = exp(-r²/(2l²))",
    )
    viscous_neighbor_weighting = param.Selector(
        default="inverse_riemannian_distance",
        objects=[
            "inverse_riemannian_distance",
            "inverse_riemannian_volume",
            "riemannian_kernel",
            "riemannian_kernel_volume",
            "inverse_distance",
            "inverse_volume",
            "kernel",
            "uniform",
        ],
        doc=(
            "Weighting for viscous neighbors. Uses precomputed edge weights\n"
            "from scutoid if the mode is in neighbor_weight_modes;\n"
            "kernel/uniform/inverse_distance fall back to on-the-fly computation."
        ),
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
    viscous_volume_weighting = param.Boolean(
        default=False,
        doc="Weight viscous neighbors by Riemannian volume element. "
        "Disable when using riemannian_kernel_volume to avoid double-counting √(det g).",
    )
    compute_volume_weights = param.Boolean(
        default=False,
        doc="Compute and store Riemannian volume weights for analysis",
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
            "n_kinetic_steps": {
                "type": pn.widgets.EditableIntSlider,
                "width": INPUT_WIDTH,
                "name": "Kinetic steps per cycle",
                "start": 1,
                "end": 50,
                "step": 1,
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
            "diffusion_mode": {
                "type": pn.widgets.Select,
                "width": INPUT_WIDTH,
                "name": "Diffusion mode",
                "options": ["hessian", "grad_proxy", "voronoi_proxy"],
            },
            "diffusion_grad_scale": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "Grad-proxy scale",
                "start": 0.01,
                "end": 10.0,
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
            "viscous_neighbor_weighting": {
                "type": pn.widgets.Select,
                "width": INPUT_WIDTH,
                "name": "Viscous weighting",
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
            "viscous_volume_weighting": {
                "type": pn.widgets.Checkbox,
                "width": INPUT_WIDTH,
                "name": "Viscous volume weighting",
            },
            "compute_volume_weights": {
                "type": pn.widgets.Checkbox,
                "width": INPUT_WIDTH,
                "name": "Record volume weights",
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
            "n_kinetic_steps",
            "integrator",
            "epsilon_F",
            "use_fitness_force",
            "use_potential_force",
            "epsilon_Sigma",
            "use_anisotropic_diffusion",
            "diffusion_mode",
            "diffusion_grad_scale",
            "diagonal_diffusion",
            "nu",
            "use_viscous_coupling",
            "viscous_length_scale",
            "viscous_neighbor_weighting",
            "viscous_neighbor_threshold",
            "viscous_neighbor_penalty",
            "viscous_degree_cap",
            "viscous_volume_weighting",
            "compute_volume_weights",
            "beta_curl",
            "V_alg",
            "use_velocity_squashing",
        ]

    def __init__(
        self,
        gamma: float,
        beta: float,
        delta_t: float,
        n_kinetic_steps: int = 1,
        integrator: str = "boris-baoab",
        epsilon_F: float = 0.0,
        use_fitness_force: bool = False,
        use_potential_force: bool = True,
        epsilon_Sigma: float = 0.1,
        use_anisotropic_diffusion: bool = True,
        diffusion_mode: str = "hessian",
        diffusion_grad_scale: float = 1.0,
        diagonal_diffusion: bool = False,
        nu: float = 0.1,
        use_viscous_coupling: bool = True,
        viscous_length_scale: float = 1.0,
        viscous_neighbor_weighting: str = "inverse_riemannian_distance",
        viscous_neighbor_threshold: float | None = None,
        viscous_neighbor_penalty: float = 0.0,
        viscous_degree_cap: float | None = None,
        viscous_volume_weighting: bool = False,
        compute_volume_weights: bool = False,
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
            n_kinetic_steps: Number of kinetic substeps per algorithm iteration
            integrator: Integration scheme (default: "boris-baoab")
            epsilon_F: Adaptation rate for fitness force (ε_F)
            use_fitness_force: Enable fitness-based force
            use_potential_force: Enable potential gradient force
            epsilon_Sigma: Hessian regularization (ε_Σ)
            use_anisotropic_diffusion: Enable anisotropic diffusion
            diffusion_mode: Diffusion mode ("hessian" or "grad_proxy")
            diffusion_grad_scale: Scale factor for gradient-proxy diffusion
            diagonal_diffusion: Use diagonal-only diffusion
            nu: Viscous coupling strength (ν)
            use_viscous_coupling: Enable viscous coupling
            viscous_length_scale: Length scale for Gaussian kernel
            viscous_neighbor_weighting: Weighting mode for viscous neighbors
            viscous_neighbor_threshold: Kernel threshold for strong neighbors
            viscous_neighbor_penalty: Penalty strength for excess neighbors
            viscous_degree_cap: Degree cap for viscous coupling saturation
            viscous_volume_weighting: Weight viscous neighbors by Riemannian volume element
            compute_volume_weights: Store Riemannian volume weights in history
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
            n_kinetic_steps=n_kinetic_steps,
            integrator=integrator,
            epsilon_F=epsilon_F,
            use_fitness_force=use_fitness_force,
            use_potential_force=use_potential_force,
            epsilon_Sigma=epsilon_Sigma,
            use_anisotropic_diffusion=use_anisotropic_diffusion,
            diffusion_mode=diffusion_mode,
            diffusion_grad_scale=diffusion_grad_scale,
            diagonal_diffusion=diagonal_diffusion,
            nu=nu,
            use_viscous_coupling=use_viscous_coupling,
            viscous_length_scale=viscous_length_scale,
            viscous_neighbor_weighting=viscous_neighbor_weighting,
            viscous_neighbor_threshold=viscous_neighbor_threshold,
            viscous_neighbor_penalty=viscous_neighbor_penalty,
            viscous_degree_cap=viscous_degree_cap,
            viscous_volume_weighting=viscous_volume_weighting,
            compute_volume_weights=compute_volume_weights,
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
            x_requires_grad = x.requires_grad
            x.requires_grad_(True)  # noqa: FBT003
            U = self.potential(x)  # [N]
            if isinstance(U, torch.Tensor) and (U.requires_grad or U.grad_fn is not None):
                grad_U = torch.autograd.grad(
                    U.sum(),
                    x,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if grad_U is None:
                    grad_U = torch.zeros_like(x)
            else:
                # Constant / stop-gradient potentials contribute zero force.
                grad_U = torch.zeros_like(x)
            force_stable = -grad_U
            x.requires_grad_(x_requires_grad)  # noqa: FBT003

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
        neighbor_edges: Tensor | None = None,
        edge_weights: Tensor | None = None,
        volume_weights: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Compute viscous coupling force using neighbor graph Laplacian.

        F_viscous(x_i) = ν ∑_j w_ij (v_j - v_i)

        Weights w_ij come from scutoid (precomputed) or are computed on-the-fly
        via the ``viscous_neighbor_weighting`` setting.

        Args:
            x: Positions [N, d]
            v: Velocities [N, d]
            neighbor_edges: Directed edges [E, 2] from scutoid neighbor graph
            edge_weights: Precomputed edge weights [E] aligned with neighbor_edges
            volume_weights: Riemannian volume weights [N]
            **kwargs: Unused (grad_fitness, hess_fitness, voronoi_data, diffusion_tensors
                      kept for call-site compatibility)

        Returns:
            viscous_force: [N, d]

        """
        if not self.use_viscous_coupling or self.nu == 0.0:
            return torch.zeros_like(v)

        if neighbor_edges is None or neighbor_edges.numel() == 0:
            import warnings

            warnings.warn(
                "neighbor_edges empty for viscous coupling; returning zero viscous force. "
                "This can happen when Delaunay degenerates (e.g., clustered walkers).",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.zeros_like(v)

        # Prepare edges
        edges = neighbor_edges.to(x.device)
        i, j = edges[:, 0].long(), edges[:, 1].long()
        valid = i != j
        if not valid.any():
            return torch.zeros_like(v)
        i, j = i[valid], j[valid]

        # Compute weights via _get_viscous_weights
        weights = self._get_viscous_weights(x, i, j, edge_weights, valid, volume_weights)

        # Velocity coupling: F_visc_i = nu * sum_j w_ij (v_j - v_i)
        v_diff = v[j] - v[i]
        force = torch.zeros_like(v)
        force.index_add_(0, i, weights.unsqueeze(1) * v_diff)
        return self.nu * force

    def _get_viscous_weights(
        self,
        x: Tensor,
        i: Tensor,
        j: Tensor,
        edge_weights: Tensor | None,
        valid_mask: Tensor,
        volume_weights: Tensor | None,
    ) -> Tensor:
        """Compute per-edge viscous weights for the given (i, j) edges."""
        weighting = self.viscous_neighbor_weighting

        if edge_weights is not None:
            # Precomputed weights provided by EuclideanGas
            w = edge_weights.to(x.device, x.dtype)[valid_mask]
        elif weighting in ("kernel", "uniform", "inverse_distance"):
            # On-the-fly fallback for modes that support it
            from fragile.fractalai.scutoid.weights import compute_edge_weights

            edge_index_coo = torch.stack([i, j], dim=0)  # [2, E']
            w = compute_edge_weights(
                x,
                edge_index_coo,
                mode=weighting,
                length_scale=self.viscous_length_scale,
            )
        else:
            raise ValueError(
                f"viscous_neighbor_weighting={weighting!r} requires precomputed "
                f"edge_weights. Add {weighting!r} to neighbor_weight_modes."
            )

        # Volume weighting (multiply by sqrt(det g) of destination)
        if self.viscous_volume_weighting and volume_weights is not None:
            vw = volume_weights.to(x.device, x.dtype)
            w = w * vw[j]
            # Re-normalize after volume weighting
            deg = torch.zeros(x.shape[0], device=x.device, dtype=w.dtype)
            deg.index_add_(0, i, w)
            deg = torch.clamp(deg, min=1e-10)
            w = w / deg[i]

        # Degree cap
        if self.viscous_degree_cap is not None:
            cap = float(self.viscous_degree_cap)
            if cap <= 0:
                return torch.zeros_like(w)
            deg = torch.zeros(x.shape[0], device=x.device, dtype=w.dtype)
            deg.index_add_(0, i, w)
            scale = torch.clamp(cap / torch.clamp(deg, min=1e-10), max=1.0)
            w = w * scale[i]

        # Threshold penalty
        if self.viscous_neighbor_threshold is not None and self.viscous_neighbor_penalty > 0:
            threshold = float(self.viscous_neighbor_threshold)
            if threshold > 0:
                strong = (w >= threshold).to(w.dtype)
                strong_count = torch.zeros(x.shape[0], device=x.device, dtype=w.dtype)
                strong_count.index_add_(0, i, strong)
                excess = torch.clamp(strong_count - 1.0, min=0.0)
                penalty_scale = 1.0 / (1.0 + self.viscous_neighbor_penalty * excess)
                w = w * penalty_scale[i]

        return w

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
        neighbor_edges: Tensor | None = None,
        hess_fitness: Tensor | None = None,
        voronoi_data: dict | None = None,
        volume_weights: Tensor | None = None,
        edge_weights: Tensor | None = None,
        diffusion_tensors: Tensor | None = None,
    ) -> Tensor:
        """Apply one Boris-aware B step (half kick + rotation + half kick)."""
        force = self._compute_force(x, v, grad_fitness)

        viscous = (
            self._compute_viscous_force(
                x,
                v,
                neighbor_edges,
                grad_fitness=grad_fitness,
                hess_fitness=hess_fitness,
                voronoi_data=voronoi_data,
                volume_weights=volume_weights,
                edge_weights=edge_weights,
                diffusion_tensors=diffusion_tensors,
            )
            if self.use_viscous_coupling
            else torch.zeros_like(v)
        )
        v_minus = v + (self.dt / 2) * (force + viscous)

        if self.curl_field is not None and self.beta_curl > 0:
            curl = self.curl_field(x)
            v_rot = self._boris_rotate(v_minus, curl)
        else:
            v_rot = v_minus

        viscous_rot = (
            self._compute_viscous_force(
                x,
                v_rot,
                neighbor_edges,
                grad_fitness=grad_fitness,
                hess_fitness=hess_fitness,
                voronoi_data=voronoi_data,
                volume_weights=volume_weights,
                edge_weights=edge_weights,
                diffusion_tensors=diffusion_tensors,
            )
            if self.use_viscous_coupling
            else torch.zeros_like(v)
        )
        return v_rot + (self.dt / 2) * (force + viscous_rot)

    def _compute_diffusion_tensor(
        self,
        x: Tensor,
        grad_fitness: Tensor | None = None,
        hess_fitness: Tensor | None = None,
        voronoi_data: dict | None = None,
        diffusion_tensors: Tensor | None = None,
    ) -> Tensor:
        """Compute anisotropic diffusion tensor σ (Hessian, gradient-proxy, or voronoi-proxy).

        Three modes available:
        1. Hessian: σ = c2 · (∇²V_fit,i + ε_Σ I)^{-1/2}
        2. Gradient-proxy: σ ≈ c2 / sqrt(|∇V_fit,i| + ε_Σ) (fast, O(Nd))
        3. Voronoi-proxy: σ from cell geometry (fastest, O(N), no derivatives!)

        Args:
            x: Positions [N, d]
            grad_fitness: Precomputed per-walker fitness gradient ∇V_fit,i [N, d]
                         Required for diffusion_mode="grad_proxy"
            hess_fitness: Precomputed per-walker Hessian ∇²V_fit,i
                         - If diagonal_diffusion=True: [N, d]
                         - If diagonal_diffusion=False: [N, d, d]
                         Required for diffusion_mode="hessian"
            voronoi_data: Voronoi tessellation data from compute_voronoi_tessellation()
                         Required for diffusion_mode="voronoi_proxy"

        Returns:
            If diagonal_diffusion=True: Diagonal elements [N, d]
            If diagonal_diffusion=False: Full diffusion tensor [N, d, d]

        Raises:
            ValueError: If required data is None for the selected mode

        Note:
            For isotropic diffusion, returns constant σ = c2 (BAOAB noise amplitude).
            Voronoi-proxy mode uses cell elongation to approximate metric anisotropy.
        """
        N, d = x.shape

        if diffusion_tensors is not None:
            sigma = diffusion_tensors.to(device=x.device, dtype=x.dtype)
            sigma = torch.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0)
            return sigma * float(self.c2)

        if not self.use_anisotropic_diffusion:
            # Isotropic diffusion: σ I (standard BAOAB)
            if self.diagonal_diffusion:
                return self.c2 * torch.ones((N, d), device=x.device, dtype=x.dtype)
            eye = torch.eye(d, device=x.device, dtype=x.dtype)
            return self.c2 * eye.unsqueeze(0).expand(N, d, d)

        if self.diffusion_mode == "grad_proxy":
            if grad_fitness is None:
                msg = "grad_fitness required when diffusion_mode='grad_proxy'"
                raise ValueError(msg)
            grad_safe = torch.nan_to_num(grad_fitness, nan=0.0, posinf=0.0, neginf=0.0)
            proxy = self.diffusion_grad_scale * grad_safe.abs()
            proxy = proxy + self.epsilon_Sigma
            proxy = torch.nan_to_num(
                proxy,
                nan=self.epsilon_Sigma,
                posinf=self.epsilon_Sigma,
                neginf=self.epsilon_Sigma,
            )
            proxy = torch.clamp(proxy, min=self.epsilon_Sigma)
            sigma_diag = self.c2 / torch.sqrt(proxy)
            sigma_diag = torch.nan_to_num(
                sigma_diag,
                nan=self.c2,
                posinf=self.c2,
                neginf=self.c2,
            )
            if self.diagonal_diffusion:
                return sigma_diag
            return torch.diag_embed(sigma_diag)

        if self.diffusion_mode == "voronoi_proxy":
            # Voronoi-proxy: Use cell geometry to approximate metric anisotropy (O(N), no derivatives!)
            if voronoi_data is None:
                import warnings
                warnings.warn(
                    "voronoi_data is None but diffusion_mode='voronoi_proxy'. "
                    "Falling back to isotropic diffusion.",
                    RuntimeWarning,
                )
                # Return isotropic diffusion tensor
                if self.diagonal_diffusion:
                    return torch.ones(N, d, device=x.device, dtype=x.dtype) * self.c2
                else:
                    return torch.eye(d, device=x.device, dtype=x.dtype).unsqueeze(0).expand(N, d, d) * self.c2

            # Import here to avoid circular dependency
            from fragile.fractalai.qft.voronoi_observables import compute_voronoi_diffusion_tensor

            # Convert positions to numpy for voronoi computation
            x_np = x.detach().cpu().numpy()

            # Compute diffusion tensor from Voronoi cell geometry
            # Pass diagonal_only flag to get appropriate tensor shape
            sigma_np = compute_voronoi_diffusion_tensor(
                voronoi_data=voronoi_data,
                positions=x_np,
                epsilon_sigma=self.epsilon_Sigma,
                c2=self.c2.item() if isinstance(self.c2, torch.Tensor) else float(self.c2),
                diagonal_only=self.diagonal_diffusion,
            )

            # Convert back to torch tensor
            sigma = torch.from_numpy(sigma_np).to(device=x.device, dtype=x.dtype)

            if sigma.shape[0] != N:
                # Voronoi is computed on alive walkers only; expand to full N.
                alive_indices = voronoi_data.get("alive_indices")
                if self.diagonal_diffusion:
                    sigma = torch.nan_to_num(
                        sigma,
                        nan=self.c2,
                        posinf=self.c2,
                        neginf=self.c2,
                    )
                    sigma_full = torch.ones(N, d, device=x.device, dtype=x.dtype) * self.c2
                else:
                    sigma = torch.nan_to_num(
                        sigma,
                        nan=self.c2,
                        posinf=self.c2,
                        neginf=self.c2,
                    )
                    sigma_full = (
                        torch.eye(d, device=x.device, dtype=x.dtype)
                        .unsqueeze(0)
                        .expand(N, d, d)
                        * self.c2
                    )

                if alive_indices is None:
                    n = min(sigma.shape[0], N)
                    sigma_full[:n] = sigma[:n]
                else:
                    alive_idx = torch.as_tensor(alive_indices, device=x.device, dtype=torch.long)
                    alive_idx = alive_idx[alive_idx < N]
                    n = min(alive_idx.shape[0], sigma.shape[0])
                    if n > 0:
                        sigma_full[alive_idx[:n]] = sigma[:n]

                sigma = sigma_full

            if self.diagonal_diffusion:
                # Diagonal mode: sigma is [N, d]
                # Handle NaN/Inf values (fallback to isotropic)
                sigma = torch.nan_to_num(
                    sigma,
                    nan=self.c2,
                    posinf=self.c2,
                    neginf=self.c2,
                )
                return sigma
            else:
                # Full anisotropic mode: sigma is [N, d, d]
                # Handle NaN/Inf values (fallback to isotropic)
                sigma = torch.nan_to_num(
                    sigma,
                    nan=self.c2,
                    posinf=self.c2,
                    neginf=self.c2,
                )
                return sigma

        # Use precomputed Hessian (default)
        if hess_fitness is None:
            msg = "hess_fitness required when diffusion_mode='hessian'"
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

    def _compute_volume_weights(
        self,
        x: Tensor,
        grad_fitness: Tensor | None = None,
        hess_fitness: Tensor | None = None,
        voronoi_data: dict | None = None,
        diffusion_tensors: Tensor | None = None,
    ) -> Tensor | None:
        if voronoi_data is None:
            return None
        try:
            from fragile.fractalai.qft.voronoi_observables import (
                compute_riemannian_volume_weights,
            )
        except Exception as exc:
            warnings.warn(
                f"Riemannian volume weights unavailable: {exc}",
                RuntimeWarning,
            )
            return None

        try:
            sigma = self._compute_diffusion_tensor(
                x,
                grad_fitness=grad_fitness,
                hess_fitness=hess_fitness,
                voronoi_data=voronoi_data,
                diffusion_tensors=diffusion_tensors,
            )
        except Exception as exc:
            warnings.warn(
                f"Failed to compute diffusion tensor for volume weights: {exc}",
                RuntimeWarning,
            )
            return None

        return compute_riemannian_volume_weights(
            voronoi_data=voronoi_data,
            sigma=sigma,
            c2=self.c2,
            full_size=x.shape[0],
        )

    def apply(
        self,
        state,
        grad_fitness: Tensor | None = None,
        hess_fitness: Tensor | None = None,
        neighbor_edges: Tensor | None = None,
        voronoi_data: dict | None = None,
        edge_weights: Tensor | None = None,
        volume_weights: Tensor | None = None,
        diffusion_tensors: Tensor | None = None,
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
                         (required if use_anisotropic_diffusion=True and diffusion_mode='hessian')
            neighbor_edges: Optional directed neighbor edges [E, 2] for viscous coupling
            voronoi_data: Voronoi tessellation data from compute_voronoi_tessellation()
                         (required if use_anisotropic_diffusion=True and diffusion_mode='voronoi_proxy')
            edge_weights: Optional precomputed edge weights aligned with neighbor_edges
            volume_weights: Optional precomputed Riemannian volume weights [N]
            diffusion_tensors: Optional precomputed diffusion tensors Σ (unscaled)

        Returns:
            Updated state after integration

        Raises:
            ValueError: If grad_fitness/hess_fitness/voronoi_data are None but features are enabled

        Note:
            The fitness gradient and Hessian are computed by the caller (EuclideanGas)
            using the FitnessOperator, and passed here as precomputed values.
            Reference: Geometric Viscous Fluid Model (11_geometric_gas.md)
        """
        x, v = state.x.clone(), state.v.clone()
        N, d = state.N, state.d
        info = {}

        if volume_weights is None and voronoi_data is not None and (
            self.compute_volume_weights
            or (self.use_viscous_coupling and self.viscous_volume_weighting)
        ):
            volume_weights = self._compute_volume_weights(
                x,
                grad_fitness=grad_fitness,
                hess_fitness=hess_fitness,
                voronoi_data=voronoi_data,
                diffusion_tensors=diffusion_tensors,
            )

        if return_info:
            force_stable, force_adapt = self._compute_force_components(x, grad_fitness)
            force_viscous = (
                self._compute_viscous_force(
                    x,
                    v,
                    neighbor_edges,
                    grad_fitness=grad_fitness,
                    hess_fitness=hess_fitness,
                    voronoi_data=voronoi_data,
                    volume_weights=volume_weights,
                    edge_weights=edge_weights,
                    diffusion_tensors=diffusion_tensors,
                )
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
            if self.compute_volume_weights:
                info["riemannian_volume_weights"] = volume_weights

        # === FIRST B STEP: Apply forces + optional Boris rotation ===
        v = self._apply_boris_kick(
            x,
            v,
            grad_fitness,
            neighbor_edges,
            hess_fitness=hess_fitness,
            voronoi_data=voronoi_data,
            volume_weights=volume_weights,
            edge_weights=edge_weights,
            diffusion_tensors=diffusion_tensors,
        )

        # === FIRST A STEP: Update positions ===
        x += (self.dt / 2) * v

        # === O STEP: Ornstein-Uhlenbeck with optional anisotropic noise ===
        ξ = torch.randn(N, d, device=self.device, dtype=self.dtype)

        if self.use_anisotropic_diffusion:
            # Compute state-dependent diffusion tensor Σ_reg
            sigma = self._compute_diffusion_tensor(
                x,
                grad_fitness=grad_fitness,
                hess_fitness=hess_fitness,
                voronoi_data=voronoi_data,
                diffusion_tensors=diffusion_tensors,
            )
            if not torch.isfinite(sigma).all():
                warnings.warn(
                    "Anisotropic diffusion: non-finite sigma; falling back to isotropic noise.",
                    RuntimeWarning,
                )
                sigma = None
                noise = self.c2 * ξ
                v = self.c1 * v + noise
            else:
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
        v = self._apply_boris_kick(
            x,
            v,
            grad_fitness,
            neighbor_edges,
            hess_fitness=hess_fitness,
            voronoi_data=voronoi_data,
            volume_weights=volume_weights,
            edge_weights=edge_weights,
            diffusion_tensors=diffusion_tensors,
        )

        # === VELOCITY SQUASHING: Apply smooth radial squashing map ===
        if self.use_velocity_squashing:
            v = psi_v(v, self.V_alg)

        # Return state with same type as input
        # Create new state object using the same class as input
        new_state = type(state)(x, v)
        if return_info:
            return new_state, info
        return new_state
