"""
Kinetic Operator: Langevin Dynamics with BAOAB / Boris-BAOAB Integrator

This module implements the kinetic operator for the Euclidean Gas algorithm,
providing Langevin dynamics integration using the BAOAB scheme.

Mathematical notation:
- gamma (γ): Friction coefficient
- beta (β): Inverse temperature 1/(k_B T)
- temperature (T_eff): Effective temperature when auto_thermostat=True
- delta_t (Δt): Time step size
"""

from __future__ import annotations

import panel as pn
import param
import torch
from torch import Tensor

from fragile.physics.fractal_gas.panel_model import INPUT_WIDTH, PanelModel


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
    - temperature (T_eff): Effective temperature when auto_thermostat=True
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
        doc="Manual inverse temperature 1/(k_B T) (β), used when auto_thermostat=False",
    )
    auto_thermostat = param.Boolean(
        default=False,
        doc=("Enable temperature-based auto thermostat: " "use β_eff = 1/T instead of manual β."),
    )
    temperature = param.Number(
        default=0.5,
        bounds=(0, None),
        softbounds=(0.01, 10.0),
        inclusive_bounds=(False, True),
        doc="Effective temperature T_eff used when auto_thermostat=True. "
        "Sets noise via σ_v = √(2γT).",
    )
    delta_t = param.Number(
        default=0.01,
        bounds=(0, None),
        softbounds=(1e-5, 0.1),
        inclusive_bounds=(False, True),
        doc="Time step size (Δt)",
    )
    n_kinetic_steps = param.Integer(
        default=4,
        bounds=(1, None),
        doc="Number of kinetic substeps per algorithm iteration",
    )
    integrator = param.Selector(
        default="boris-baoab",
        objects=["baoab", "boris-baoab"],
        doc="Integration scheme (baoab or boris-baoab)",
    )

    # Viscous coupling (velocity-dependent damping)
    nu = param.Number(default=3.0, bounds=(0, None), doc="Viscous coupling strength (ν)")
    viscous_length_scale = param.Number(
        default=0.1,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        doc="Length scale (l) for Gaussian kernel K(r) = exp(-r²/(2l²))",
    )
    viscous_neighbor_weighting = param.Selector(
        default="riemannian_kernel_volume",
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
    # Boris rotation (curl-driven velocity rotation)
    beta_curl = param.Number(
        default=2.0,
        bounds=(0, None),
        doc="Curl coupling strength (β_curl) for Boris rotation",
    )
    curl_field = param.Parameter(
        default=None,
        doc="Optional curl/2-form field for Boris rotation: callable F(x)->[N, d, d] or [N, 3]",
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
            "temperature": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "T_eff (temperature)",
                "start": 0.01,
                "end": 10.0,
                "step": 0.01,
            },
            "delta_t": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "Δt (time step)",
                "start": 0.01,
                "end": 0.2,
                "step": 0.005,
            },
            "nu": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "ν (viscous coupling)",
                "start": 0.0,
                "end": 10.0,
                "step": 0.1,
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
            "beta_curl": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "β_curl (Boris strength)",
                "start": 0.0,
                "end": 10.0,
                "step": 0.1,
            },
            "n_kinetic_steps": {
                "type": pn.widgets.IntSlider,
                "width": INPUT_WIDTH,
                "name": "N kinetic steps",
                "start": 1,
                "end": 10,
                "step": 1,
            },
        }

    @property
    def widget_parameters(self) -> list[str]:
        """Parameters to display in UI."""
        return [
            "gamma",
            "beta",
            "temperature",
            "delta_t",
            "n_kinetic_steps",
            "nu",
            "viscous_length_scale",
            "viscous_neighbor_weighting",
            "beta_curl",
        ]

    def __init__(
        self,
        gamma: float,
        beta: float,
        delta_t: float,
        temperature: float = 0.5,
        nu: float = 3.0,
        use_viscous_coupling: bool = True,
        viscous_length_scale: float = 0.1,
        viscous_neighbor_weighting: str = "inverse_riemannian_distance",
        beta_curl: float = 2.0,
        curl_field=None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """
        Initialize kinetic operator.

        Args:
            gamma: Friction coefficient (γ)
            beta: Inverse temperature 1/(k_B T) (β)
            delta_t: Time step size (Δt)
            temperature: Effective temperature T_eff for auto thermostat mode
            nu: Viscous coupling strength (ν)
            use_viscous_coupling: Enable viscous coupling
            viscous_length_scale: Length scale for Gaussian kernel
            viscous_neighbor_weighting: Weighting mode for viscous neighbors
            beta_curl: Curl coupling strength for Boris rotation
            device: PyTorch device (defaults to CPU)
            dtype: PyTorch dtype (defaults to float32)

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
            temperature=temperature,
            delta_t=delta_t,
            nu=nu,
            viscous_length_scale=viscous_length_scale,
            viscous_neighbor_weighting=viscous_neighbor_weighting,
            beta_curl=beta_curl,
            curl_field=curl_field,
        )
        self.use_viscous_coupling = use_viscous_coupling
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        if self.curl_field is not None and not callable(self.curl_field):
            msg = f"curl_field must be callable, got {type(self.curl_field)}"
            raise TypeError(msg)

        self._thermostat_eps = 1e-12
        self.beta_effective = float(self.beta)

        # Precompute BAOAB constants (updated each apply in case params changed via UI).
        self._refresh_ou_coefficients()

    def effective_beta(self) -> float:
        """Return effective inverse temperature used by the OU thermostat."""
        beta_manual = max(float(self.beta), self._thermostat_eps)
        if not self.auto_thermostat:
            return beta_manual

        t_eff = max(float(self.temperature), self._thermostat_eps)
        return 1.0 / t_eff

    def effective_temperature(self) -> float:
        """Return effective temperature T_eff = 1 / β_eff used by the OU thermostat."""
        return 1.0 / self.effective_beta()

    def _refresh_ou_coefficients(self) -> None:
        """Refresh BAOAB O-step coefficients from current thermostat parameters."""
        self.dt = float(self.delta_t)
        self.beta_effective = self.effective_beta()
        self.c1 = torch.exp(
            torch.tensor(-float(self.gamma) * self.dt, dtype=self.dtype, device=self.device)
        )
        self.c2 = torch.sqrt(
            torch.tensor(1.0, dtype=self.dtype, device=self.device) - self.c1**2
        ) / torch.sqrt(torch.tensor(self.beta_effective, dtype=self.dtype, device=self.device))

    def noise_std(self) -> float:
        """Standard deviation for BAOAB noise (isotropic case)."""
        self._refresh_ou_coefficients()
        return float(self.c2.item())

    def _compute_viscous_force(
        self,
        x: Tensor,
        v: Tensor,
        neighbor_edges: Tensor | None = None,
        edge_weights: Tensor | None = None,
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

        Returns:
            viscous_force: [N, d]

        """

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
        edge_weights = edge_weights[valid]

        # Velocity coupling: F_visc_i = nu * sum_j w_ij (v_j - v_i)
        v_diff = v[j] - v[i]
        force = torch.zeros_like(v)
        force.index_add_(0, i, edge_weights.unsqueeze(1) * v_diff)
        return self.nu * force

    def _compute_viscous_curl(
        self,
        x: Tensor,
        viscous_force: Tensor,
        neighbor_edges: Tensor,
        edge_weights: Tensor,
    ) -> Tensor:
        """Compute the curl of the viscous force field via weighted least-squares Jacobian.

        For each walker i, estimates the spatial Jacobian J_i^{ab} = dF_visc^a / dx^b
        from neighbor differences using weighted least squares, then extracts the
        antisymmetric part as the emergent chromomagnetic field strength:

            F_curl_i = (J_i - J_i^T) / 2

        The viscous force plays the role of a gauge connection (coupling walkers through
        the Delaunay graph), and its curl is the corresponding field strength 2-form.

        Args:
            x: Positions [N, d]
            viscous_force: Viscous force at each walker [N, d]
            neighbor_edges: Directed edges [E, 2]
            edge_weights: Edge weights [E]

        Returns:
            curl: Skew-symmetric 2-form field [N, d, d]
        """
        N, d = x.shape
        device, dtype = x.device, x.dtype

        edges = neighbor_edges.to(device)
        i, j = edges[:, 0].long(), edges[:, 1].long()
        valid = i != j
        if not valid.any():
            return torch.zeros(N, d, d, device=device, dtype=dtype)
        i, j = i[valid], j[valid]
        w = edge_weights.to(device, dtype)[valid]

        # Displacements and force differences
        dx = x[j] - x[i]  # [E', d]
        dF = viscous_force[j] - viscous_force[i]  # [E', d]

        # Weighted outer products for least-squares: A = J @ B
        #   A^{ab}_i = sum_j w_ij * dF^a_ij * dx^b_ij
        #   B^{ab}_i = sum_j w_ij * dx^a_ij * dx^b_ij
        w_col = w.unsqueeze(-1)  # [E', 1]
        A_edges = (w_col * dF).unsqueeze(-1) * dx.unsqueeze(-2)  # [E', d, d]
        B_edges = (w_col * dx).unsqueeze(-1) * dx.unsqueeze(-2)  # [E', d, d]

        # Accumulate per walker
        A = torch.zeros(N, d * d, device=device, dtype=dtype)
        B = torch.zeros(N, d * d, device=device, dtype=dtype)
        A.index_add_(0, i, A_edges.reshape(-1, d * d))
        B.index_add_(0, i, B_edges.reshape(-1, d * d))
        A = A.reshape(N, d, d)
        B = B.reshape(N, d, d)

        # Solve A = J @ B for J via J^T = lstsq(B, A^T)  (B is symmetric)
        # lstsq handles rank-deficient B (e.g. walker with too few neighbors)
        J_T = torch.linalg.lstsq(B, A.transpose(-1, -2)).solution  # [N, d, d]

        # Extract antisymmetric part: curl = (J - J^T) / 2
        # Since J = J_T^T, this is (J_T^T - J_T) / 2
        return (J_T.transpose(-1, -2) - J_T) / 2

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
        neighbor_edges: Tensor | None = None,
        edge_weights: Tensor | None = None,
        force_viscous: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Apply one Boris-aware B step (half kick + rotation + half kick).

        Returns:
            v_new: Updated velocity [N, d]
            curl: The curl 2-form used for rotation [N, d, d], or None if no rotation.
        """
        if force_viscous is None:
            force_viscous = self._compute_viscous_force(
                x,
                v,
                neighbor_edges=neighbor_edges,
                edge_weights=edge_weights,
            )

        v_minus = v + (self.dt / 2) * force_viscous

        curl = None
        if self.curl_field is not None and self.beta_curl > 0:
            # External curl field
            curl = self.curl_field(x)
            v_rot = self._boris_rotate(v_minus, curl)
        elif self.beta_curl > 0 and neighbor_edges is not None and edge_weights is not None:
            # Emergent curl from viscous force Jacobian
            curl = self._compute_viscous_curl(x, force_viscous, neighbor_edges, edge_weights)
            v_rot = self._boris_rotate(v_minus, curl)
        else:
            v_rot = v_minus

        viscous_rot = self._compute_viscous_force(
            x,
            v_rot,
            neighbor_edges=neighbor_edges,
            edge_weights=edge_weights,
        )

        return v_rot + (self.dt / 2) * viscous_rot, curl

    def apply(
        self,
        state,
        neighbor_edges: Tensor | None = None,
        edge_weights: Tensor | None = None,
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
            neighbor_edges: Optional directed neighbor edges [E, 2] for viscous coupling
            edge_weights: Optional precomputed edge weights aligned with neighbor_edges

        Returns:
            Updated state after integration

        Raises:
            ValueError: If grad_fitness/hess_fitness/voronoi_data are None but features are enabled

        Note:
            The fitness gradient and Hessian are computed by the caller (EuclideanGas)
            using the FitnessOperator, and passed here as precomputed values.
            Reference: Geometric Viscous Fluid Model (11_geometric_gas.md)
        """
        self._refresh_ou_coefficients()
        x, v = state.x.clone(), state.v.clone()
        N, d = state.N, state.d
        force_viscous = None
        info = {}
        if return_info:
            force_viscous = self._compute_viscous_force(
                x,
                v,
                neighbor_edges,
                edge_weights=edge_weights,
            )
            force_friction = -self.gamma * v
            force_total = force_viscous + force_friction
            info.update({
                "force_stable": torch.zeros_like(v),
                "force_adapt": torch.zeros_like(v),
                "force_viscous": force_viscous,
                "force_friction": force_friction,
                "force_total": force_total,
            })

        # === FIRST B STEP: Apply forces + optional Boris rotation ===
        v, curl_1 = self._apply_boris_kick(
            x,
            v,
            neighbor_edges=neighbor_edges,
            edge_weights=edge_weights,
            force_viscous=force_viscous,
        )
        # === FIRST A STEP: Update positions ===
        x += (self.dt / 2) * v

        # === O STEP: Ornstein-Uhlenbeck with optional anisotropic noise ===
        # Isotropic: standard BAOAB with constant noise amplitude
        xi = torch.randn(N, d, device=self.device, dtype=self.dtype)
        v = self.c1 * v + self.c2 * xi
        noise = self.c2 * xi

        if return_info:
            info["noise"] = noise

        # === SECOND A STEP: Update positions ===
        x += (self.dt / 2) * v

        # === SECOND B STEP: Apply forces + optional Boris rotation ===
        v, _curl_2 = self._apply_boris_kick(
            x,
            v,
            neighbor_edges=neighbor_edges,
            edge_weights=edge_weights,
        )
        new_state = type(state)(x, v)
        if return_info:
            # Record the curl from the first B-step (representative of this timestep)
            if curl_1 is not None:
                info["curl_field"] = curl_1
            return new_state, info
        return new_state
