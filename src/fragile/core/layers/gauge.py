from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GeodesicConfig:
    """Configuration for GeodesicCrossAttention."""

    d_model: int = 256
    d_latent: int = 64
    n_heads: int = 1
    T_c: float = 0.1
    gamma_friction: float = 1.0
    dt: float = 0.01
    g_s: float = 1.0
    g_2: float = 0.5
    g_1: float = 0.3
    use_learned_thermostat: bool = False
    thermostat_residual_scale: float = 0.1


class WilsonLineApprox(nn.Module):
    """Approximate Wilson line for parallel transport."""

    def __init__(self, config: GeodesicConfig, d_k: int, d_conn: int = 8) -> None:
        super().__init__()
        self.d_k = d_k
        self.d_conn = min(d_conn, config.d_latent)

        self.delta_proj = nn.Linear(config.d_latent, self.d_conn, bias=False)

        self.basis_binding = nn.Parameter(0.01 * torch.randn(self.d_conn, d_k, d_k))
        self.basis_error = nn.Parameter(0.01 * torch.randn(self.d_conn, d_k, d_k))
        self.basis_opportunity = nn.Parameter(0.01 * torch.randn(self.d_conn, d_k, d_k))

        self.g_s = config.g_s
        self.g_2 = config.g_2
        self.g_1 = config.g_1

    def forward(self, z_query: torch.Tensor, z_key: torch.Tensor) -> torch.Tensor:
        """Compute Wilson line correction factors.

        Args:
            z_query: [B, d_latent] query positions
            z_key: [B, N, d_latent] key positions

        Returns:
            U: [B, N, d_k, d_k] approximate transport matrices
        """
        _, n, _ = z_key.shape

        # Relative displacement parameterizes the transport path in latent space.
        delta_z = z_query.unsqueeze(1) - z_key  # [B, N, d_latent]
        coeff = self.delta_proj(delta_z)  # [B, N, d_conn]

        def skew(basis: torch.Tensor) -> torch.Tensor:
            return basis - basis.transpose(-1, -2)

        # Skew-symmetric generators approximate the connection in the Lie algebra.
        h_mat = (
            self.g_s * torch.einsum("bnr,rij->bnij", coeff, skew(self.basis_binding))
            + self.g_2 * torch.einsum("bnr,rij->bnij", coeff, skew(self.basis_error))
            + self.g_1 * torch.einsum("bnr,rij->bnij", coeff, skew(self.basis_opportunity))
        )

        identity = torch.eye(self.d_k, device=z_key.device, dtype=z_key.dtype)
        identity = identity.expand(z_query.shape[0], n, self.d_k, self.d_k)
        # Linearized Wilson line: U ≈ I + ∫A, used for parallel transport.
        return identity + h_mat


class ConformalMetric(nn.Module):
    """Poincare ball/disk conformal metric utilities."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon

    def conformal_factor(self, z: torch.Tensor) -> torch.Tensor:
        """Compute conformal factor lambda(z).

        Args:
            z: [B, d] positions

        Returns:
            lambda_z: [B, 1] conformal factors
        """
        r_sq = (z**2).sum(dim=-1, keepdim=True)
        r_sq = torch.clamp(r_sq, max=1.0 - self.epsilon)
        return 2.0 / (1.0 - r_sq + self.epsilon)

    def metric(self, z: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor G_ij(z).

        Returns:
            g: [B, d, d] metric tensors
        """
        _, d = z.shape
        lambda_sq = self.conformal_factor(z) ** 2
        eye = torch.eye(d, device=z.device, dtype=z.dtype)
        return lambda_sq.unsqueeze(-1) * eye

    def metric_inv(self, z: torch.Tensor) -> torch.Tensor:
        """Compute inverse metric tensor G^{ij}(z).

        Returns:
            g_inv: [B, d, d] inverse metric tensors
        """
        _, d = z.shape
        lambda_sq_inv = 1.0 / (self.conformal_factor(z) ** 2 + self.epsilon)
        eye = torch.eye(d, device=z.device, dtype=z.dtype)
        return lambda_sq_inv.unsqueeze(-1) * eye

    def temperature(self, z: torch.Tensor, d_k: int) -> torch.Tensor:
        """Compute position-dependent attention temperature.

        Args:
            z: [B, d] positions
            d_k: key dimension

        Returns:
            tau: [B, 1] temperature values
        """
        lambda_z = self.conformal_factor(z)
        return math.sqrt(d_k) / lambda_z


class ChristoffelQuery(nn.Module):
    """Geometric query projection encoding Christoffel symbols."""

    def __init__(self, d_in: int, d_out: int, d_latent: int) -> None:
        super().__init__()
        self.W_Q = nn.Linear(d_in, d_out, bias=False)
        self.W_Qz = nn.Linear(d_latent, d_out, bias=False)
        self.W_Qv = nn.Linear(d_in, d_out, bias=False)

        self.W_Q_gamma = nn.Parameter(torch.zeros(d_out, d_latent, d_latent))
        self._init_christoffel(d_latent)
        self.W_Qzv = nn.Parameter(torch.zeros(d_out, d_latent, d_latent))

    def _init_christoffel(self, d_latent: int) -> None:
        """Initialize W_Q_gamma with a Poincare-inspired pattern."""
        with torch.no_grad():
            for k in range(min(d_latent, self.W_Q_gamma.shape[0])):
                for i in range(d_latent):
                    for j in range(d_latent):
                        if k in {i, j}:
                            self.W_Q_gamma[k, i, j] = 0.01
                        if i == j:
                            self.W_Q_gamma[k, i, j] -= 0.01

    def forward(
        self,
        x: torch.Tensor,
        z_geom: torch.Tensor,
        v_feat: torch.Tensor | None = None,
        v_geom: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute geodesic query.

        Args:
            x: [B, d_in] features
            z_geom: [B, d_latent] positions
            v_feat: [B, d_in] optional velocity features
            v_geom: [B, d_latent] optional velocity

        Returns:
            q: [B, d_out] query vectors
        """
        # Base query from features and position (local chart coordinates).
        q = self.W_Q(x) + self.W_Qz(z_geom)
        if v_feat is not None:
            q += self.W_Qv(v_feat)

        # Quadratic term encodes Christoffel-symbol corrections from curvature.
        d_latent = min(z_geom.shape[-1], self.W_Q_gamma.shape[-1])
        z_trunc = z_geom[..., :d_latent]
        q_gamma = torch.einsum(
            "aij,bi,bj->ba", self.W_Q_gamma[:, :d_latent, :d_latent], z_trunc, z_trunc
        )
        q += q_gamma

        if v_geom is not None:
            # Velocity coupling approximates connection terms along the geodesic.
            v_trunc = v_geom[..., :d_latent]
            q_zv = torch.einsum(
                "aij,bi,bj->ba", self.W_Qzv[:, :d_latent, :d_latent], z_trunc, v_trunc
            )
            q += q_zv

        return q


class ChiralProjector(nn.Module):
    """SU(2)_L chiral projector from value gradient."""

    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.grad_proj = nn.Linear(d_latent, 3, bias=False)

        self.register_buffer("identity", torch.eye(2))
        self.register_buffer("sigma_1", torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        self.register_buffer("sigma_2", torch.tensor([[0.0, -1.0], [1.0, 0.0]]))
        self.register_buffer("sigma_3", torch.tensor([[1.0, 0.0], [0.0, -1.0]]))

    def forward(self, psi_doublet: torch.Tensor, grad_V: torch.Tensor) -> torch.Tensor:
        """Apply chiral projection and compute commitment strength.

        Args:
            psi_doublet: [B, 2, d] observation-action doublet
            grad_V: [B, d_latent] value gradient

        Returns:
            psi_proj: [B, 2*d] projected doublet (flattened)
        """
        # Use value-gradient direction to define the chiral projection axis.
        n_vec = self.grad_proj(grad_V)
        n_hat = n_vec / (torch.norm(n_vec, dim=-1, keepdim=True) + 1e-8)
        n_x, n_y, n_z = n_hat.unbind(dim=-1)

        proj = 0.5 * (
            self.identity
            + n_x[:, None, None] * self.sigma_1
            + n_y[:, None, None] * self.sigma_2
            + n_z[:, None, None] * self.sigma_3
        )

        # Apply SU(2) projector, then weight by commitment strength.
        psi_proj = torch.einsum("bij,bjd->bid", proj, psi_doublet)
        commit_strength = (psi_doublet * psi_proj).sum(dim=1, keepdim=True)
        psi_proj *= commit_strength
        return psi_proj.reshape(psi_proj.shape[0], -1)


class AreaLawScreening(nn.Module):
    """Area law screening for attention weights."""

    def __init__(self, config: GeodesicConfig) -> None:
        super().__init__()
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(config.g_s**2)))

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    def string_area(
        self,
        z_query: torch.Tensor,
        z_key: torch.Tensor,
        lambda_z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute string-area proxy between positions.

        Args:
            z_query: [B, d] query positions
            z_key: [B, N, d] key positions
            lambda_z: [B, 1] conformal factor

        Returns:
            area: [B, N] string areas
        """
        delta = z_query.unsqueeze(1) - z_key
        dist_sq = (delta**2).sum(dim=-1)
        return 0.5 * (lambda_z**2) * dist_sq

    def forward(
        self,
        attention: torch.Tensor,
        z_query: torch.Tensor,
        z_key: torch.Tensor,
        lambda_z: torch.Tensor,
        level: int = 0,
    ) -> torch.Tensor:
        """Apply area law screening.

        Args:
            attention: [B, N] attention weights
            z_query: [B, d] query positions
            z_key: [B, N, d] key positions
            lambda_z: [B, 1] conformal factor
            level: hierarchy level (0=macro, L=texture)

        Returns:
            screened: [B, N] screened attention weights
        """
        area = self.string_area(z_query, z_key, lambda_z)
        l_max = 10
        sigma_eff = self.sigma * math.exp(-level / l_max)
        screening = torch.exp(-sigma_eff * area)
        return attention * screening


class CovariantAttention(nn.Module):
    """Single covariant attention head with gauge structures."""

    def __init__(
        self,
        config: GeodesicConfig,
        use_chirality: bool = False,
        use_screening: bool = False,
        head_type: str = "generic",
    ) -> None:
        super().__init__()
        self.config = config
        self.use_chirality = use_chirality
        self.use_screening = use_screening
        self.head_type = head_type

        if config.d_model % config.n_heads != 0:
            msg = "d_model must be divisible by n_heads."
            raise ValueError(msg)

        d_k = config.d_model // config.n_heads

        self.query = ChristoffelQuery(config.d_model, d_k, config.d_latent)
        self.key = nn.Linear(config.d_model, d_k, bias=False)
        self.value = nn.Linear(config.d_model, d_k, bias=False)
        self.output = nn.Linear(d_k, config.d_model, bias=False)

        self.wilson = WilsonLineApprox(config, d_k)
        self.metric = ConformalMetric()

        if use_chirality:
            self.chiral = ChiralProjector(config.d_latent)

        if use_screening:
            self.screening = AreaLawScreening(config)

        self.d_k = d_k

    def forward(
        self,
        z_query: torch.Tensor,
        z_key: torch.Tensor,
        x_query: torch.Tensor,
        x_key: torch.Tensor,
        x_value: torch.Tensor,
        v_query: torch.Tensor | None = None,
        v_query_geom: torch.Tensor | None = None,
        grad_V: torch.Tensor | None = None,
        level: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute covariant attention output.

        Args:
            z_query: [B, d_latent] query positions
            z_key: [B, N, d_latent] key positions
            x_query: [B, d_model] query features
            x_key: [B, N, d_model] key features
            x_value: [B, N, d_model] value features
            v_query: [B, d_model] optional velocity features
            v_query_geom: [B, d_latent] optional velocity
            grad_V: [B, d_latent] optional value gradient
            level: hierarchy level

        Returns:
            output: [B, d_model] attention output
            attention: [B, N] attention weights
        """
        q = self.query(x_query, z_query, v_query, v_query_geom)
        k = self.key(x_key)
        v = self.value(x_value)

        # Parallel transport keys into the query frame (Wilson line).
        u = self.wilson(z_query, z_key)
        k_transported = torch.einsum("bnij,bnj->bni", u, k)

        # Metric-aware similarity with conformal temperature scaling.
        scores = torch.einsum("bi,bni->bn", q, k_transported)
        tau = self.metric.temperature(z_query, self.d_k)
        scores /= tau + 1e-08

        attention = F.softmax(scores, dim=-1)

        if self.use_screening:
            # Area-law screening suppresses long strings in the latent geometry.
            lambda_z = self.metric.conformal_factor(z_query)
            attention = self.screening(attention, z_query, z_key, lambda_z, level)
            attention /= attention.sum(dim=-1, keepdim=True) + 1e-08

        # Aggregate transported values; optional chirality post-processes the bundle.
        output = torch.einsum("bn,bni->bi", attention, v)

        if self.use_chirality and grad_V is not None:
            if output.shape[-1] % 2 != 0:
                msg = "Chiral projection requires an even d_k."
                raise ValueError(msg)
            output_doublet = output.reshape(output.shape[0], 2, -1)
            output = self.chiral(output_doublet, grad_V)
            output = output.reshape(output.shape[0], -1)

        output = self.output(output)
        return output, attention


class GeodesicCrossAttention(nn.Module):
    """Full geodesic world model implementing BAOAB integration."""

    def __init__(self, config: GeodesicConfig) -> None:
        super().__init__()
        self.config = config

        self.dt = config.dt
        self.gamma = config.gamma_friction
        self.T_c = config.T_c
        self.c1 = math.exp(-self.gamma * self.dt)
        self.c2 = math.sqrt((1.0 - self.c1**2) * self.T_c) if self.T_c > 0 else 0.0

        self.metric = ConformalMetric()

        self.head_B1 = CovariantAttention(config, head_type="B")
        self.head_A1 = CovariantAttention(config, head_type="A")
        self.use_learned_thermostat = config.use_learned_thermostat
        self.thermostat_residual_scale = config.thermostat_residual_scale
        if self.use_learned_thermostat:
            self.head_O = CovariantAttention(config, head_type="O")
        else:
            self.head_O = None
        self.head_A2 = CovariantAttention(config, head_type="A")
        self.head_B2 = CovariantAttention(config, head_type="B")

        self.pos_encoder = nn.Linear(config.d_latent, config.d_model)
        self.grad_encoder = nn.Linear(config.d_latent, config.d_model)
        self.velocity_encoder = nn.Linear(config.d_latent, config.d_model)

        self.state_proj = nn.Linear(config.d_model, config.d_latent)

        if self.use_learned_thermostat:
            self.noise_proj = nn.Linear(config.d_latent, config.d_model)
        else:
            self.noise_proj = None

    def forward(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        context_z: torch.Tensor,
        context_x: torch.Tensor,
        context_force: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step of geodesic BAOAB integration.

        Args:
            z: [B, d_latent] current position
            p: [B, d_latent] current momentum
            context_z: [B, N, d_latent] context positions
            context_x: [B, N, d_model] context features
            context_force: [B, N, d_latent] force bank

        Returns:
            z_next: [B, d_latent] updated positions
            p_next: [B, d_latent] updated momentum
        """
        h = self.dt

        force_features = self.grad_encoder(context_force)

        # B step: momentum kick from forces (geodesic acceleration).
        delta_p1, _ = self.head_B1(
            z_query=z,
            z_key=context_z,
            x_query=self.pos_encoder(z),
            x_key=force_features,
            x_value=force_features,
        )
        delta_p1_latent = self.state_proj(delta_p1)
        p -= h / 2.0 * delta_p1_latent

        # A step: drift along metric velocity with learned correction.
        g_inv = self.metric.metric_inv(z)
        v = torch.einsum("bij,bj->bi", g_inv, p)

        delta_z1, _ = self.head_A1(
            z_query=z,
            z_key=context_z,
            x_query=self.pos_encoder(z) + self.velocity_encoder(v),
            x_key=context_x,
            x_value=context_x,
            v_query=self.velocity_encoder(v),
            v_query_geom=v,
        )
        delta_z1_latent = self.state_proj(delta_z1)
        z += h / 2.0 * (v + delta_z1_latent)
        # Keep positions inside the Poincare ball (safe harbor).
        z = self._project_to_disk(z)

        # O step: thermostat (Ornstein-Uhlenbeck) in the conformal metric.
        g_sqrt = self.metric.conformal_factor(z)
        xi = torch.randn_like(p)
        p = self.c1 * p + self.c2 * g_sqrt * xi

        if self.use_learned_thermostat:
            noise_bank = torch.randn_like(context_z)
            noise_features = self.noise_proj(noise_bank)
            delta_p_noise, _ = self.head_O(
                z_query=z,
                z_key=context_z,
                x_query=self.velocity_encoder(p),
                x_key=noise_features,
                x_value=noise_features,
            )
            delta_p_noise_latent = self.state_proj(delta_p_noise)
            p += self.thermostat_residual_scale * delta_p_noise_latent

        # A step: second drift with updated momentum.
        g_inv = self.metric.metric_inv(z)
        v = torch.einsum("bij,bj->bi", g_inv, p)

        delta_z2, _ = self.head_A2(
            z_query=z,
            z_key=context_z,
            x_query=self.pos_encoder(z) + self.velocity_encoder(v),
            x_key=context_x,
            x_value=context_x,
            v_query=self.velocity_encoder(v),
            v_query_geom=v,
        )
        delta_z2_latent = self.state_proj(delta_z2)
        z += h / 2.0 * (v + delta_z2_latent)
        # Keep positions inside the Poincare ball (safe harbor).
        z = self._project_to_disk(z)

        # B step: final momentum kick from forces.
        delta_p2, _ = self.head_B2(
            z_query=z,
            z_key=context_z,
            x_query=self.pos_encoder(z),
            x_key=force_features,
            x_value=force_features,
        )
        delta_p2_latent = self.state_proj(delta_p2)
        p -= h / 2.0 * delta_p2_latent

        return z, p

    def _project_to_disk(self, z: torch.Tensor, max_norm: float = 0.999) -> torch.Tensor:
        """Project positions to interior of the Poincare ball/disk."""
        norm = torch.norm(z, dim=-1, keepdim=True)
        return torch.where(norm > max_norm, z * max_norm / norm, z)
