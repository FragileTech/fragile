from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GeodesicConfig:
    """Configuration for gauge-covariant attention."""

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
    """Linearized Wilson line approximation."""

    def __init__(self, config: GeodesicConfig, d_k: int) -> None:
        super().__init__()
        self.config = config
        self.d_k = d_k

        self.theta_binding = nn.Parameter(torch.randn(d_k, d_k, config.d_latent) * 0.01)
        self.theta_error = nn.Parameter(torch.randn(d_k, d_k, config.d_latent) * 0.01)
        self.theta_opportunity = nn.Parameter(torch.randn(d_k, d_k, config.d_latent) * 0.01)

        eye = torch.eye(d_k)
        self.register_buffer("identity", eye)

    def forward(self, z_query: torch.Tensor, z_key: torch.Tensor) -> torch.Tensor:
        """Compute Wilson line transport matrices.

        Args:
            z_query: [B, d_latent] query positions
            z_key: [B, N, d_latent] key positions

        Returns:
            U: [B, N, d_k, d_k] transport matrices
        """
        delta = z_query.unsqueeze(1) - z_key  # [B, N, d_latent]
        theta = (
            self.config.g_s * self.theta_binding
            + self.config.g_2 * self.theta_error
            + self.config.g_1 * self.theta_opportunity
        )  # [d_k, d_k, d_latent]

        A = torch.einsum("bnd,ijd->bnij", delta, theta)  # [B, N, d_k, d_k]
        U = self.identity.unsqueeze(0).unsqueeze(0) - A  # [B, N, d_k, d_k]
        return U


class ConformalMetric(nn.Module):
    """Poincare disk conformal metric utilities."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon

    def conformal_factor(self, z: torch.Tensor) -> torch.Tensor:
        """Compute conformal factor.

        Args:
            z: [B, d] positions

        Returns:
            lambda_z: [B, 1] conformal factor
        """
        norm_sq = (z**2).sum(dim=-1, keepdim=True)  # [B, 1]
        lambda_z = 2.0 / (1.0 - norm_sq + self.epsilon)  # [B, 1]
        return lambda_z

    def metric(self, z: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor.

        Args:
            z: [B, d] positions

        Returns:
            g: [B, d, d] metric tensor
        """
        lambda_z = self.conformal_factor(z)  # [B, 1]
        eye = torch.eye(z.shape[-1], device=z.device, dtype=z.dtype)  # [d, d]
        g = (lambda_z**2).unsqueeze(-1) * eye  # [B, d, d]
        return g

    def metric_inv(self, z: torch.Tensor) -> torch.Tensor:
        """Compute inverse metric tensor.

        Args:
            z: [B, d] positions

        Returns:
            g_inv: [B, d, d] inverse metric
        """
        lambda_z = self.conformal_factor(z)  # [B, 1]
        eye = torch.eye(z.shape[-1], device=z.device, dtype=z.dtype)  # [d, d]
        g_inv = (lambda_z**-2).unsqueeze(-1) * eye  # [B, d, d]
        return g_inv

    def temperature(self, z: torch.Tensor, d_k: int) -> torch.Tensor:
        """Compute position-dependent temperature.

        Args:
            z: [B, d] positions
            d_k: key dimension

        Returns:
            tau: [B, 1] temperature
        """
        lambda_z = self.conformal_factor(z)  # [B, 1]
        tau = (d_k**0.5) / lambda_z  # [B, 1]
        return tau


class ChristoffelQuery(nn.Module):
    """Geodesic query projection with Christoffel terms."""

    def __init__(self, d_in: int, d_out: int, d_latent: int) -> None:
        super().__init__()
        self.w_x = nn.Linear(d_in, d_out)
        self.w_z = nn.Linear(d_latent, d_out)
        self.w_v = nn.Linear(d_in, d_out)

        self.w_gamma = nn.Parameter(torch.randn(d_out, d_latent, d_latent) * 0.01)
        self.w_zv = nn.Parameter(torch.randn(d_out, d_latent, d_latent) * 0.01)

    def forward(
        self,
        x: torch.Tensor,
        z_geom: torch.Tensor,
        v_feat: Optional[torch.Tensor] = None,
        v_geom: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute geodesic query.

        Args:
            x: [B, d_in] features
            z_geom: [B, d_latent] position
            v_feat: [B, d_in] optional velocity features
            v_geom: [B, d_latent] optional velocity

        Returns:
            q: [B, d_out] query vector
        """
        q_x = self.w_x(x)  # [B, d_out]
        q_z = self.w_z(z_geom)  # [B, d_out]
        q_v = self.w_v(v_feat) if v_feat is not None else torch.zeros_like(q_x)  # [B, d_out]

        q_gamma = torch.einsum("bi,oij,bj->bo", z_geom, self.w_gamma, z_geom)  # [B, d_out]
        if v_geom is not None:
            q_zv = torch.einsum("bi,oij,bj->bo", z_geom, self.w_zv, v_geom)  # [B, d_out]
        else:
            q_zv = torch.zeros_like(q_x)  # [B, d_out]

        q = q_x + q_z + q_v + q_gamma + q_zv  # [B, d_out]
        return q


class ChiralProjector(nn.Module):
    """SU(2) chiral projector using value gradients."""

    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.n_proj = nn.Linear(d_latent, 3)

        tau1 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        tau2 = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        tau3 = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
        self.register_buffer("pauli", torch.stack([tau1, tau2, tau3], dim=0))  # [3, 2, 2]

    def forward(self, psi_doublet: torch.Tensor, grad_V: torch.Tensor) -> torch.Tensor:
        """Project doublet along value gradient.

        Args:
            psi_doublet: [B, 2, d] observation-action doublet
            grad_V: [B, d_latent] value gradient

        Returns:
            psi_proj: [B, 2*d] projected doublet (flattened)
        """
        n_vec = self.n_proj(grad_V)  # [B, 3]
        n_hat = n_vec / (n_vec.norm(dim=-1, keepdim=True) + 1e-6)  # [B, 3]

        n_tau = (n_hat.view(-1, 3, 1, 1) * self.pauli.unsqueeze(0)).sum(dim=1)  # [B, 2, 2]
        eye = torch.eye(2, device=psi_doublet.device, dtype=psi_doublet.dtype)  # [2, 2]
        proj = 0.5 * (eye + n_tau)  # [B, 2, 2]

        psi_proj = torch.einsum("bij,bjd->bid", proj, psi_doublet)  # [B, 2, d]
        psi_flat = psi_proj.reshape(psi_proj.shape[0], -1)  # [B, 2*d]
        return psi_flat


class AreaLawScreening(nn.Module):
    """Area law screening for attention weights."""

    def __init__(self, config: GeodesicConfig) -> None:
        super().__init__()
        self.sigma0 = config.g_s
        self.level_scale = 1.0

    def string_area(
        self, z_query: torch.Tensor, z_key: torch.Tensor, lambda_z: torch.Tensor
    ) -> torch.Tensor:
        """Approximate string area.

        Args:
            z_query: [B, d] query positions
            z_key: [B, N, d] key positions
            lambda_z: [B, 1] conformal factor

        Returns:
            area: [B, N] string areas
        """
        delta = z_query.unsqueeze(1) - z_key  # [B, N, d]
        dist_sq = (delta**2).sum(dim=-1)  # [B, N]
        area = 0.5 * (lambda_z**2) * dist_sq  # [B, N]
        return area

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
            level: hierarchy level

        Returns:
            screened: [B, N] screened attention
        """
        level_t = torch.tensor(level, device=attention.device, dtype=attention.dtype)  # []
        sigma = self.sigma0 * torch.exp(-level_t / self.level_scale)  # []
        area = self.string_area(z_query, z_key, lambda_z)  # [B, N]
        screened = attention * torch.exp(-sigma * area)  # [B, N]
        return screened


class CovariantAttention(nn.Module):
    """Single gauge-covariant attention head."""

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

        self.query = ChristoffelQuery(config.d_model, config.d_model, config.d_latent)
        self.key_proj = nn.Linear(config.d_model, config.d_model)
        self.value_proj = nn.Linear(config.d_model, config.d_model)

        self.wilson = WilsonLineApprox(config, config.d_model)
        self.metric = ConformalMetric()
        self.chiral = ChiralProjector(config.d_latent)
        self.screening = AreaLawScreening(config)

    def forward(
        self,
        z_query: torch.Tensor,
        z_key: torch.Tensor,
        x_query: torch.Tensor,
        x_key: torch.Tensor,
        x_value: torch.Tensor,
        v_query: Optional[torch.Tensor] = None,
        v_query_geom: Optional[torch.Tensor] = None,
        grad_V: Optional[torch.Tensor] = None,
        level: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        q = self.query(x_query, z_query, v_feat=v_query, v_geom=v_query_geom)  # [B, d_model]
        k = self.key_proj(x_key)  # [B, N, d_model]
        v = self.value_proj(x_value)  # [B, N, d_model]

        U = self.wilson(z_query, z_key)  # [B, N, d_model, d_model]
        k_trans = torch.einsum("bnij,bnj->bni", U, k)  # [B, N, d_model]

        tau = self.metric.temperature(z_query, k.shape[-1])  # [B, 1]
        scores = (q.unsqueeze(1) * k_trans).sum(dim=-1)  # [B, N]
        scores = scores / (tau + 1e-6)  # [B, N]

        attention = F.softmax(scores, dim=-1)  # [B, N]

        if self.use_screening:
            lambda_z = self.metric.conformal_factor(z_query)  # [B, 1]
            attention = self.screening(  # [B, N]
                attention, z_query, z_key, lambda_z, level=level
            )
            attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-8)  # [B, N]

        output = (attention.unsqueeze(-1) * v).sum(dim=1)  # [B, d_model]

        if self.use_chirality and grad_V is not None:
            psi = torch.stack([x_query, output], dim=1)  # [B, 2, d_model]
            psi_proj = self.chiral(psi, grad_V)  # [B, 2*d_model]
            output = psi_proj.view(psi_proj.shape[0], 2, -1)[:, 1, :]  # [B, d_model]

        return output, attention


class GeodesicCrossAttention(nn.Module):
    """BAOAB-style geodesic cross-attention integrator."""

    def __init__(self, config: GeodesicConfig) -> None:
        super().__init__()
        self.config = config

        self.query_feat = nn.Linear(2 * config.d_latent, config.d_model)
        self.force_value = nn.Linear(config.d_latent, config.d_model)
        self.force_out = nn.Linear(config.d_model, config.d_latent)
        self.drift_out = nn.Linear(config.d_model, config.d_latent)

        self.head_b1 = CovariantAttention(
            config, use_chirality=False, use_screening=True, head_type="B"
        )
        self.head_a1 = CovariantAttention(
            config, use_chirality=False, use_screening=False, head_type="A"
        )
        self.head_a2 = CovariantAttention(
            config, use_chirality=False, use_screening=False, head_type="A"
        )
        self.head_b2 = CovariantAttention(
            config, use_chirality=False, use_screening=True, head_type="B"
        )

        if config.use_learned_thermostat:
            self.thermostat_residual = nn.Linear(2 * config.d_latent, config.d_latent)
        else:
            self.thermostat_residual = None

    def forward(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        context_z: torch.Tensor,
        context_x: torch.Tensor,
        context_force: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance position and momentum with covariant attention.

        Args:
            z: [B, d_latent] current positions
            p: [B, d_latent] current momentum
            context_z: [B, N, d_latent] context positions
            context_x: [B, N, d_model] context features
            context_force: [B, N, d_latent] context force bank

        Returns:
            z_next: [B, d_latent] updated positions
            p_next: [B, d_latent] updated momentum
        """
        dt = torch.tensor(self.config.dt, device=z.device, dtype=z.dtype)  # []
        gamma = torch.tensor(self.config.gamma_friction, device=z.device, dtype=z.dtype)  # []
        temp = torch.tensor(self.config.T_c, device=z.device, dtype=z.dtype)  # []

        zp = torch.cat([z, p], dim=-1)  # [B, 2*d_latent]
        q_feat = self.query_feat(zp)  # [B, d_model]

        force_value = self.force_value(context_force)  # [B, N, d_model]

        out_b1, _ = self.head_b1(z, context_z, q_feat, context_x, force_value)  # [B, d_model]
        force1 = self.force_out(out_b1)  # [B, d_latent]
        p_half = p + 0.5 * dt * force1  # [B, d_latent]

        out_a1, _ = self.head_a1(z, context_z, q_feat, context_x, context_x)  # [B, d_model]
        drift1 = self.drift_out(out_a1)  # [B, d_latent]
        z_half = z + 0.5 * dt * (p_half + drift1)  # [B, d_latent]

        c1 = torch.exp(-gamma * dt)  # []
        c2 = torch.sqrt((1.0 - c1**2) * temp)  # []
        noise = torch.randn_like(p_half)  # [B, d_latent]
        p_ou = c1 * p_half + c2 * noise  # [B, d_latent]

        if self.thermostat_residual is not None:
            thermo = self.thermostat_residual(zp)  # [B, d_latent]
            p_ou = p_ou + self.config.thermostat_residual_scale * thermo  # [B, d_latent]

        out_a2, _ = self.head_a2(z_half, context_z, q_feat, context_x, context_x)  # [B, d_model]
        drift2 = self.drift_out(out_a2)  # [B, d_latent]
        z_next = z_half + 0.5 * dt * (p_ou + drift2)  # [B, d_latent]

        out_b2, _ = self.head_b2(z_next, context_z, q_feat, context_x, force_value)  # [B, d_model]
        force2 = self.force_out(out_b2)  # [B, d_latent]
        p_next = p_ou + 0.5 * dt * force2  # [B, d_latent]

        return z_next, p_next
