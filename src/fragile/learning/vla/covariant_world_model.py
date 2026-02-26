"""Covariant geometric world model with geodesic Boris-BAOAB integration on the Poincare ball.

Replaces the MLP-based sub-modules of the original GeometricWorldModel with
covariant versions that use CovariantAttention from gauge.py, ensuring all
force computations respect the Poincare geometry.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from fragile.learning.core.layers.atlas import _project_to_ball
from fragile.learning.core.layers.gauge import (
    ConformalMetric,
    CovariantAttention,
    GeodesicConfig,
    christoffel_contraction,
    hyperbolic_distance,
    poincare_exp_map,
)
from fragile.learning.core.layers.primitives import SpectralLinear
from fragile.learning.core.layers.topology import FactorizedJumpOperator


def compute_risk_tensor(
    force: torch.Tensor,
    curl_tensor: torch.Tensor | None = None,
    lambda_inv_sq: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute risk tensor T = T_grad + T_maxwell.

    Args:
        force: [B, D] conservative force vector.
        curl_tensor: [B, D, D] antisymmetric field strength, or None.
        lambda_inv_sq: [B, 1] scalar inverse-squared conformal factor lambda^{-2}.

    Returns:
        T: [B, D, D] risk tensor (symmetric).
    """
    # Gradient stress: T_grad = f (x) f
    T = torch.einsum("bi,bj->bij", force, force)  # [B, D, D]

    if curl_tensor is not None and lambda_inv_sq is not None:
        # Maxwell stress: T_maxwell = F_ik F^k_j - 1/4 delta_ij F_kl F^kl
        # F^k_j = G^{km} F_{mj} = lambda^{-2} * F_{mj} (conformal metric is diagonal)
        F_up = lambda_inv_sq.unsqueeze(-1) * curl_tensor  # [B, D, D] scalar broadcast
        # F_ik F^k_j
        FF = torch.bmm(curl_tensor, F_up)  # [B, D, D]
        # Trace: F_kl F^kl
        trace_FF = torch.diagonal(FF, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
        D = force.shape[1]
        eye = torch.eye(D, device=force.device, dtype=force.dtype).unsqueeze(0)
        T_maxwell = FF - 0.25 * trace_FF * eye
        T = T + T_maxwell

    return T


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------


class ActionTokenizer(nn.Module):
    """Lifts a flat action [B, A] into N context tokens for cross-attention."""

    def __init__(self, action_dim: int, d_model: int, latent_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.weight = nn.Parameter(torch.randn(action_dim, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(action_dim, d_model) * 0.02)

    def forward(
        self, action: torch.Tensor, z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize action into context tokens.

        Args:
            action: [B, A] action vector.
            z: [B, D] latent position.

        Returns:
            tokens_x: [B, A, d_model] feature tokens.
            tokens_z: [B, A, D] position tokens (all at z).
        """
        tokens_x = (
            action.unsqueeze(-1) * self.weight.unsqueeze(0)
            + self.pos_embed.unsqueeze(0)
        )  # [B, A, d_model]
        tokens_z = z.unsqueeze(1).expand(-1, self.action_dim, -1).contiguous()  # [B, A, D]
        return tokens_x, tokens_z


class ChartTokenizer(nn.Module):
    """Builds chart context tokens from router weights."""

    def __init__(self, num_charts: int, d_model: int, latent_dim: int) -> None:
        super().__init__()
        self.chart_embeddings = nn.Parameter(
            torch.randn(num_charts, d_model) * 0.02,
        )
        centers = F.normalize(torch.randn(num_charts, latent_dim), dim=-1) * 0.3
        self.chart_centers = nn.Parameter(centers)

    def forward(
        self, rw: torch.Tensor, z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize chart weights into context tokens.

        Args:
            rw: [B, K] router / chart weights.
            z: [B, D] latent position (unused, kept for API consistency).

        Returns:
            tokens_x: [B, K, d_model] feature tokens.
            tokens_z: [B, K, D] chart center positions.
        """
        B = rw.shape[0]
        tokens_x = rw.unsqueeze(-1) * self.chart_embeddings.unsqueeze(0)  # [B, K, d_model]
        # Project chart centers to stay inside the Poincare ball
        safe_centers = _project_to_ball(self.chart_centers)  # [K, D]
        tokens_z = safe_centers.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, K, D]
        return tokens_x, tokens_z


# ---------------------------------------------------------------------------
# Covariant sub-modules
# ---------------------------------------------------------------------------


class CovariantPotentialNet(nn.Module):
    r"""Direct force prediction + scalar potential for diagnostics.

    Produces a cotangent force vector for the BAOAB B-step **without**
    ``torch.autograd.grad`` (no second-order graph), plus the scalar
    effective potential :math:`\Phi_{\text{eff}}` for the energy
    conservation loss.

    Force decomposition (cotangent vector):

    .. math::
        F = \alpha\,\partial U/\partial z
          + (1-\alpha)\, f_{\text{critic}}(z, K)
          + \gamma_{\text{risk}}\, f_{\text{risk}}(z, K)

    where :math:`\partial U/\partial z = -2z\,/\,(|z|\,(1-|z|^2))` is
    the *analytical* gradient of the hyperbolic drive (no learnable
    params, no autograd).  The learned force components are predicted
    directly by CovariantAttention heads.

    Scalar potential (for energy conservation loss only):

    .. math::
        \Phi_{\text{eff}} = \alpha\, U(z)
                          + (1-\alpha)\, V_{\text{critic}}(z, K)
                          + \gamma_{\text{risk}}\, \Psi_{\text{risk}}(z, K)
    """

    def __init__(
        self,
        latent_dim: int,
        num_charts: int,
        d_model: int,
        alpha: float = 0.5,
        gamma_risk: float = 0.01,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma_risk = gamma_risk
        self.latent_dim = latent_dim

        self.chart_tok = ChartTokenizer(num_charts, d_model, latent_dim)
        self.z_embed = SpectralLinear(latent_dim, d_model)

        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)

        # Critic head: shared attention → force vector + scalar V
        self.v_critic_attn = CovariantAttention(geo_cfg)
        self.v_force_out = SpectralLinear(d_model, latent_dim)
        self.v_out = SpectralLinear(d_model, 1)

        # Risk head: shared attention → force vector + scalar Ψ
        self.psi_risk_attn = CovariantAttention(geo_cfg)
        self.psi_force_out = SpectralLinear(d_model, latent_dim)
        self.psi_out = SpectralLinear(d_model, 1)

    def _analytic_U_and_grad(
        self, z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Analytic hyperbolic drive and its exact gradient.

        .. math::
            U(z) = -2\operatorname{artanh}(|z|)

            \frac{\partial U}{\partial z_i}
                = \frac{-2\, z_i}{|z|\,(1 - |z|^2)}

        Returns:
            U: [B, 1] scalar potential.
            dU_dz: [B, D] Euclidean gradient (cotangent vector).
        """
        r = z.norm(dim=-1, keepdim=True).clamp(min=1e-6, max=1.0 - 1e-6)
        U = -2.0 * torch.atanh(r)  # [B, 1]
        dU_dz = -2.0 * z / (r * (1.0 - r ** 2))  # [B, D]
        return U, dU_dz

    def forward(self, z: torch.Tensor, rw: torch.Tensor) -> torch.Tensor:
        """Compute scalar effective potential (for energy conservation loss).

        Args:
            z: [B, D] position in the Poincare ball.
            rw: [B, K] router / chart weights.

        Returns:
            phi: [B, 1] effective potential.
        """
        U, _ = self._analytic_U_and_grad(z)

        ctx_x, ctx_z = self.chart_tok(rw, z)
        x_q = self.z_embed(z)

        v_feat, _ = self.v_critic_attn(z, ctx_z, x_q, ctx_x, ctx_x)
        V = self.v_out(v_feat)  # [B, 1]

        psi_feat, _ = self.psi_risk_attn(z, ctx_z, x_q, ctx_x, ctx_x)
        Psi = self.psi_out(psi_feat)  # [B, 1]

        return self.alpha * U + (1.0 - self.alpha) * V + self.gamma_risk * Psi

    def force_and_potential(
        self, z: torch.Tensor, rw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Direct force prediction + scalar potential (no autograd).

        The force is a cotangent vector for the Hamiltonian momentum kick
        (dp/dt = -force + u_pi).  The scalar potential is for the energy
        conservation diagnostic.

        Args:
            z: [B, D] position.
            rw: [B, K] router weights.

        Returns:
            force: [B, D] conservative force (cotangent vector).
            phi: [B, 1] scalar effective potential.
        """
        U, dU_dz = self._analytic_U_and_grad(z)

        ctx_x, ctx_z = self.chart_tok(rw, z)
        x_q = self.z_embed(z)

        # Critic: force + scalar from shared attention
        v_feat, _ = self.v_critic_attn(z, ctx_z, x_q, ctx_x, ctx_x)
        f_critic = self.v_force_out(v_feat)  # [B, D]
        V = self.v_out(v_feat)  # [B, 1]

        # Risk: force + scalar from shared attention
        psi_feat, _ = self.psi_risk_attn(z, ctx_z, x_q, ctx_x, ctx_x)
        f_risk = self.psi_force_out(psi_feat)  # [B, D]
        Psi = self.psi_out(psi_feat)  # [B, 1]

        force = (
            self.alpha * dU_dz
            + (1.0 - self.alpha) * f_critic
            + self.gamma_risk * f_risk
        )
        phi = self.alpha * U + (1.0 - self.alpha) * V + self.gamma_risk * Psi

        return force, phi


class CovariantControlField(nn.Module):
    """Action-conditioned control force in tangent space using CovariantAttention."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        num_charts: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.action_tok = ActionTokenizer(action_dim, d_model, latent_dim)
        self.chart_tok = ChartTokenizer(num_charts, d_model, latent_dim)

        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)
        self.attn = CovariantAttention(geo_cfg)
        self.out = SpectralLinear(d_model, latent_dim)
        self.z_embed = SpectralLinear(latent_dim, d_model)

    def forward(
        self, z: torch.Tensor, action: torch.Tensor, rw: torch.Tensor,
    ) -> torch.Tensor:
        """Compute control force u_pi.

        Args:
            z: [B, D] position.
            action: [B, A] action vector.
            rw: [B, K] chart weights.

        Returns:
            u: [B, D] control force in tangent space.
        """
        act_x, act_z = self.action_tok(action, z)
        chart_x, chart_z = self.chart_tok(rw, z)

        # Concatenate action + chart context tokens
        ctx_x = torch.cat([act_x, chart_x], dim=1)  # [B, A+K, d_model]
        ctx_z = torch.cat([act_z, chart_z], dim=1)  # [B, A+K, D]

        x_q = self.z_embed(z)
        output, _ = self.attn(z, ctx_z, x_q, ctx_x, ctx_x)
        return self.out(output)


class CovariantValueCurl(nn.Module):
    r"""Predicts antisymmetric field strength :math:`\mathcal{F}_{ij}` for Boris rotation.

    Uses CovariantAttention over action tokens instead of an MLP.
    """

    def __init__(self, latent_dim: int, action_dim: int, d_model: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_upper = latent_dim * (latent_dim - 1) // 2

        self.action_tok = ActionTokenizer(action_dim, d_model, latent_dim)

        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)
        self.attn = CovariantAttention(geo_cfg)
        self.out = SpectralLinear(d_model, self.n_upper)
        self.z_embed = SpectralLinear(latent_dim, d_model)

        self.register_buffer(
            "_tri_rows", torch.triu_indices(latent_dim, latent_dim, offset=1)[0],
        )
        self.register_buffer(
            "_tri_cols", torch.triu_indices(latent_dim, latent_dim, offset=1)[1],
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute antisymmetric field strength tensor.

        Args:
            z: [B, D] position.
            action: [B, A] action vector.

        Returns:
            F_mat: [B, D, D] antisymmetric tensor.
        """
        B = z.shape[0]
        D = self.latent_dim

        act_x, act_z = self.action_tok(action, z)
        x_q = self.z_embed(z)
        output, _ = self.attn(z, act_z, x_q, act_x, act_x)
        upper = self.out(output)  # [B, n_upper]

        F_mat = z.new_zeros(B, D, D)
        F_mat[:, self._tri_rows, self._tri_cols] = upper
        F_mat[:, self._tri_cols, self._tri_rows] = -upper
        return F_mat


class CovariantChartTarget(nn.Module):
    """Predicts target chart logits for the jump process using CovariantAttention."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        num_charts: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts

        centers = F.normalize(torch.randn(num_charts, latent_dim), dim=-1) * 0.3
        self.chart_centers = nn.Parameter(centers)

        self.action_tok = ActionTokenizer(action_dim, d_model, latent_dim)

        self.q_gamma = nn.Parameter(
            torch.randn(num_charts, latent_dim, latent_dim) * 0.02,
        )
        self.action_proj = SpectralLinear(d_model, num_charts)
        self.metric = ConformalMetric()

    def forward(
        self, z: torch.Tensor, action: torch.Tensor, rw: torch.Tensor,
    ) -> torch.Tensor:
        """Predict chart transition logits.

        Args:
            z: [B, D] position.
            action: [B, A] action vector.
            rw: [B, K] current chart weights.

        Returns:
            logits: [B, K] chart logits.
        """
        K = self.num_charts

        # 1. Base: geodesic distance to chart centers with conformal temperature
        # Project chart centers to stay inside the Poincare ball
        safe_centers = _project_to_ball(self.chart_centers)  # [K, D]
        z_expanded = z.unsqueeze(1).expand(-1, K, -1).contiguous()  # [B, K, D]
        centers_expanded = safe_centers.unsqueeze(0).expand(
            z.shape[0], -1, -1,
        ).contiguous()  # [B, K, D]
        # Flatten for hyperbolic_distance then reshape
        dist = hyperbolic_distance(
            z_expanded.reshape(-1, z.shape[-1]),
            centers_expanded.reshape(-1, z.shape[-1]),
        ).reshape(z.shape[0], K)  # [B, K]
        tau = self.metric.temperature(z, z.shape[-1])  # [B, 1]
        base = -dist / (tau + 1e-8)  # [B, K]

        # 2. Christoffel correction: quadratic in z
        z_outer = torch.einsum("bi,bj->bij", z, z)  # [B, D, D]
        gamma_corr = torch.einsum("bij,kij->bk", z_outer, self.q_gamma)  # [B, K]

        # 3. Action contribution via tokenizer + mean pool + projection
        act_x, _act_z = self.action_tok(action, z)  # [B, A, d_model]
        act_pooled = act_x.mean(dim=1)  # [B, d_model]
        act_logits = self.action_proj(act_pooled)  # [B, K]

        return base + gamma_corr + act_logits


class CovariantJumpRate(nn.Module):
    """Predicts Poisson jump rate lambda(z, K) >= 0 using CovariantAttention."""

    def __init__(
        self, latent_dim: int, num_charts: int, d_model: int,
    ) -> None:
        super().__init__()
        self.chart_tok = ChartTokenizer(num_charts, d_model, latent_dim)

        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)
        self.attn = CovariantAttention(geo_cfg)
        self.out = SpectralLinear(d_model, 1)
        self.z_embed = SpectralLinear(latent_dim, d_model)

    def forward(self, z: torch.Tensor, rw: torch.Tensor) -> torch.Tensor:
        """Compute jump rate.

        Args:
            z: [B, D] position.
            rw: [B, K] chart weights.

        Returns:
            rate: [B, 1] non-negative jump rate.
        """
        ctx_x, ctx_z = self.chart_tok(rw, z)
        x_q = self.z_embed(z)
        output, _ = self.attn(z, ctx_z, x_q, ctx_x, ctx_x)
        return F.softplus(self.out(output))


class CovariantMomentumInit(nn.Module):
    """Initializes momentum from position with conformal metric scaling."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = SpectralLinear(latent_dim, latent_dim)
        self.metric = ConformalMetric()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Initialize momentum from position.

        Args:
            z: [B, D] position in the Poincare ball.

        Returns:
            p: [B, D] initial momentum scaled by conformal factor.
        """
        lam_sq = self.metric.conformal_factor(z) ** 2  # [B, 1]
        return lam_sq * self.net(z)


class HodgeDecomposer(nn.Module):
    """Decomposes total force into conservative, solenoidal, and harmonic parts.

    Hodge decomposition: f_total = f_conservative + f_solenoidal + f_harmonic

    - f_conservative = -nabla V (gradient of critic potential, from potential_net)
    - f_solenoidal = beta * F * G^{-1} * p (from Boris/curl field strength)
    - f_harmonic = f_total - f_conservative - f_solenoidal (residual)

    The harmonic component should be small if the model is well-structured.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(
        self,
        force_total: torch.Tensor,
        force_conservative: torch.Tensor,
        force_solenoidal: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute Hodge decomposition and diagnostics.

        Args:
            force_total: [B, D] total non-control force.
            force_conservative: [B, D] conservative part (-nabla Phi from potential_net).
            force_solenoidal: [B, D] solenoidal part (Boris curl force).

        Returns:
            Dictionary with:
                harmonic: [B, D] harmonic residual
                conservative_ratio: [B] ||f_cons|| / ||f_total||
                solenoidal_ratio: [B] ||f_sol|| / ||f_total||
                harmonic_ratio: [B] ||f_harm|| / ||f_total||
        """
        f_harmonic = force_total - force_conservative - force_solenoidal

        eps = 1e-8
        total_norm = force_total.norm(dim=-1).clamp(min=eps)  # [B]
        cons_norm = force_conservative.norm(dim=-1)  # [B]
        sol_norm = force_solenoidal.norm(dim=-1)  # [B]
        harm_norm = f_harmonic.norm(dim=-1)  # [B]

        return {
            "harmonic": f_harmonic,
            "conservative_ratio": cons_norm / total_norm,
            "solenoidal_ratio": sol_norm / total_norm,
            "harmonic_ratio": harm_norm / total_norm,
        }


# ---------------------------------------------------------------------------
# Main world model
# ---------------------------------------------------------------------------


class GeometricWorldModel(nn.Module):
    r"""Action-conditioned dynamics model using geodesic Boris-BAOAB integration.

    Drop-in replacement for the MLP-based GeometricWorldModel, using
    CovariantAttention-based sub-modules that respect Poincare geometry.

    When ``min_length > 0``, three CFL-derived bounds are enforced using
    smooth squashing maps (same :math:`\psi` from the Euclidean Gas theory):

    .. math::
        V_{\text{alg}} = \frac{\ell_{\min}}{\Delta t},\qquad
        F_{\max} = \frac{2\gamma\,\ell_{\min}}{\Delta t},\qquad
        \lambda_{\max} = \frac{2\,\ell_{\min}}{c_2\,\sqrt{D}\,\Delta t}

    Args:
        latent_dim: Dimension of the Poincare ball latent space.
        action_dim: Dimension of the action vector.
        num_charts: Number of atlas charts.
        d_model: Feature dimension for CovariantAttention modules.
        hidden_dim: Kept for API compatibility (unused).
        dt: Integration time step.
        gamma_friction: Langevin friction coefficient.
        T_c: Thermostat temperature.
        alpha_potential: Balance between analytic drive U and learned critic V.
        beta_curl: Value-curl coupling strength for Lorentz/Boris forces.
        gamma_risk: Risk-stress penalty weight in the effective potential.
        use_boris: Whether to enable Boris rotation for curl forces.
        use_jump: Whether to enable the Poisson jump process.
        jump_rate_hidden: Kept for API compatibility (unused).
        min_length: Minimum resolvable geodesic length scale.
            When > 0, derives F_max, V_alg and cf_max from CFL conditions.
            When 0, all squashing is disabled (backward compat).
    """

    def __init__(
        self,
        latent_dim: int = 16,
        action_dim: int = 14,
        num_charts: int = 8,
        d_model: int = 128,
        hidden_dim: int = 256,
        dt: float = 0.01,
        gamma_friction: float = 1.0,
        T_c: float = 0.1,
        alpha_potential: float = 0.5,
        beta_curl: float = 0.1,
        gamma_risk: float = 0.01,
        use_boris: bool = True,
        use_jump: bool = True,
        jump_rate_hidden: int = 64,
        min_length: float = 0.0,
        risk_metric_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_charts = num_charts
        self.dt = dt
        self.gamma = gamma_friction
        self.T_c = T_c
        self.beta_curl = beta_curl
        self.use_boris = use_boris
        self.use_jump = use_jump
        self.min_length = min_length
        self.risk_metric_alpha = risk_metric_alpha

        # Ornstein-Uhlenbeck coefficients
        self.c1 = math.exp(-gamma_friction * dt)
        self.c2 = math.sqrt(max(0.0, (1.0 - self.c1**2) * T_c))

        # CFL-derived bounds from minimum length scale
        if min_length > 0:
            self.V_alg = min_length / dt
            self.F_max = 2.0 * gamma_friction * min_length / dt
            # cf_max = 2 * ℓ_min / (c2 * √D * dt); floor at 2.0 (origin value)
            denom = max(self.c2, 1e-12) * math.sqrt(latent_dim) * dt
            self.cf_max = max(2.0 * min_length / denom, 2.0)
        else:
            self.V_alg = 0.0
            self.F_max = 0.0
            self.cf_max = 0.0

        # Metric
        if risk_metric_alpha > 0:
            from fragile.learning.core.layers.gauge import RiskAdaptiveConformalMetric
            self.metric = RiskAdaptiveConformalMetric(risk_coupling_alpha=risk_metric_alpha)
        else:
            self.metric = ConformalMetric()

        # Covariant sub-modules
        self.potential_net = CovariantPotentialNet(
            latent_dim, num_charts, d_model,
            alpha=alpha_potential, gamma_risk=gamma_risk,
        )
        self.control_net = CovariantControlField(
            latent_dim, action_dim, num_charts, d_model,
        )

        if use_boris:
            self.curl_net = CovariantValueCurl(latent_dim, action_dim, d_model)
        else:
            self.curl_net = None

        if use_jump:
            self.jump_rate_net = CovariantJumpRate(latent_dim, num_charts, d_model)
            self.jump_operator = FactorizedJumpOperator(num_charts, latent_dim)
        else:
            self.jump_rate_net = None
            self.jump_operator = None

        self.chart_predictor = CovariantChartTarget(
            latent_dim, action_dim, num_charts, d_model,
        )

        # Initial momentum from position
        self.momentum_init = CovariantMomentumInit(latent_dim)

        # Hodge decomposition diagnostic (no learnable parameters)
        self.hodge_decomposer = HodgeDecomposer(latent_dim)

    # -----------------------------------------------------------------
    # Boris rotation
    # -----------------------------------------------------------------

    def _boris_rotation(
        self,
        p_minus: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
        curl_F: torch.Tensor | None = None,
        lambda_inv_sq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Norm-preserving Boris rotation for the Lorentz/curl force.

        For D > 3 the cross product generalises to the antisymmetric
        matrix-vector product.  The Boris algorithm ensures
        ``|p_plus| = |p_minus|`` because F is antisymmetric.

        Args:
            p_minus: [B, D] momentum after potential half-kick.
            z: [B, D] current position.
            action: [B, A] action vector.
            curl_F: [B, D, D] pre-computed curl tensor, or None to compute.
            lambda_inv_sq: [B, 1] scalar inverse-squared conformal factor, or None to compute.

        Returns:
            p_plus: [B, D] rotated momentum (same norm).
            curl_F: [B, D, D] the curl tensor used (for reuse), or None.
        """
        if self.curl_net is None:
            return p_minus, None

        h = self.dt
        F = curl_F if curl_F is not None else self.curl_net(z, action)  # [B, D, D] antisymmetric
        if lambda_inv_sq is None:
            cf = self.metric.conformal_factor(z)  # [B, 1]
            lambda_inv_sq = 1.0 / (cf ** 2 + self.metric.epsilon)  # [B, 1]
        T = (h / 2.0) * self.beta_curl * lambda_inv_sq.unsqueeze(-1) * F  # [B, D, D]

        # Boris half-rotation
        t_vec = torch.bmm(T, p_minus.unsqueeze(-1)).squeeze(-1)  # [B, D]
        p_prime = p_minus + t_vec

        t_sq = (T ** 2).sum(dim=(-2, -1), keepdim=False)  # [B] Frobenius norm^2
        s_factor = 2.0 / (1.0 + t_sq).unsqueeze(-1)  # [B, 1]
        s_vec = s_factor * torch.bmm(T, p_prime.unsqueeze(-1)).squeeze(-1)  # [B, D]

        return p_minus + s_vec, F

    # -----------------------------------------------------------------
    # BAOAB integration step
    # -----------------------------------------------------------------

    def _baoab_step(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        action: torch.Tensor,
        rw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """One geodesic Boris-BAOAB integration step.

        Fixes five geometry bugs in the original skeleton:
        1. Uses ``poincare_exp_map`` at z (not origin-based).
        2. Includes Christoffel geodesic correction.
        3. Boris rotation for Lorentz/curl forces.
        4. Structured effective potential with Riemannian gradient.
        5. Spectral-normalised layers throughout.

        Args:
            z: [B, D] current position.
            p: [B, D] current momentum (cotangent vector).
            action: [B, A] action vector.
            rw: [B, K] chart weights.

        Returns:
            z_next: [B, D] updated position.
            p_next: [B, D] updated momentum.
            phi_eff: [B, 1] effective potential at the step.
            hodge_info: Hodge decomposition diagnostics dict (may be empty).
        """
        h = self.dt
        _use_risk = self.risk_metric_alpha > 0

        # --- B step (first half): momentum kick ---
        # Direct force prediction (cotangent vector, no autograd).
        # Hamilton's equation: dp/dt = -force + u_π  (all covectors).
        force, _ = self.potential_net.force_and_potential(z, rw)
        u_pi = self.control_net(z, action, rw)  # [B, D] cotangent force
        kick = force - u_pi

        # Pre-compute curl tensor once for both Boris rotation and risk tensor
        curl_F = None
        if self.curl_net is not None:
            curl_F = self.curl_net(z, action)

        # Compute risk tensor for metric adaptation
        risk_T = None
        lis_base = None  # lambda^{-2} at current z (base metric, no risk)
        if _use_risk:
            cf_base = self.metric.conformal_factor(z)  # [B, 1] -- base ConformalMetric
            lis_base = 1.0 / (cf_base ** 2 + self.metric.epsilon)  # [B, 1]
            risk_T = compute_risk_tensor(force, curl_F, lis_base)

        # ψ_F: smooth force squashing (1-Lipschitz, C∞, direction-preserving)
        if self.F_max > 0:
            kick = self.F_max * kick / (
                self.F_max + kick.norm(dim=-1, keepdim=True)
            )

        p_minus = p - (h / 2.0) * kick
        # pass pre-computed lambda_inv_sq (None in non-risk case, boris computes its own)
        p_plus, _ = self._boris_rotation(p_minus, z, action, curl_F=curl_F, lambda_inv_sq=lis_base)

        # Hodge decomposition: solenoidal force = Boris impulse / dt
        hodge_info: dict[str, torch.Tensor] = {}
        boris_impulse = p_plus - p_minus  # [B, D]
        f_solenoidal = boris_impulse / max(h, 1e-8)
        f_total_no_ctrl = force + f_solenoidal
        hodge_info = self.hodge_decomposer(f_total_no_ctrl, force, f_solenoidal)

        p = p_plus - (h / 2.0) * kick

        # --- A step (first half): geodesic drift ---
        if _use_risk:
            cf = self.metric.conformal_factor(z, risk_tensor=risk_T)  # [B, 1]
        else:
            cf = self.metric.conformal_factor(z)  # [B, 1]
        lambda_inv_sq = 1.0 / (cf ** 2 + self.metric.epsilon)  # [B, 1]
        v = lambda_inv_sq * p  # [B, D] contravariant velocity
        geo_corr = christoffel_contraction(z, v)
        v_corr = v - (h / 4.0) * geo_corr
        # ψ_v: smooth velocity squashing (V_alg = ℓ_min / Δt)
        if self.V_alg > 0:
            v_corr = self.V_alg * v_corr / (
                self.V_alg + v_corr.norm(dim=-1, keepdim=True)
            )
        z = poincare_exp_map(z, (h / 2.0) * v_corr)
        z = _project_to_ball(z)

        # --- O step: Ornstein-Uhlenbeck thermostat ---
        if _use_risk:
            cf = self.metric.conformal_factor(z, risk_tensor=risk_T)  # [B, 1]
        else:
            cf = self.metric.conformal_factor(z)  # [B, 1]
        if self.cf_max > 0:
            # Smooth conformal factor cap via tanh: preserves interior values,
            # smoothly saturates near boundary
            cf = self.cf_max * torch.tanh(cf / self.cf_max)
        xi = torch.randn_like(p)
        p = self.c1 * p + self.c2 * cf * xi

        # --- A step (second half): geodesic drift ---
        if _use_risk:
            cf = self.metric.conformal_factor(z, risk_tensor=risk_T)  # [B, 1]
        else:
            cf = self.metric.conformal_factor(z)  # [B, 1]
        lambda_inv_sq = 1.0 / (cf ** 2 + self.metric.epsilon)  # [B, 1]
        v = lambda_inv_sq * p  # [B, D] contravariant velocity
        geo_corr = christoffel_contraction(z, v)
        v_corr = v - (h / 4.0) * geo_corr
        if self.V_alg > 0:
            v_corr = self.V_alg * v_corr / (
                self.V_alg + v_corr.norm(dim=-1, keepdim=True)
            )
        z = poincare_exp_map(z, (h / 2.0) * v_corr)
        z = _project_to_ball(z)

        # --- B step (second half): momentum kick ---
        force2, phi_eff = self.potential_net.force_and_potential(z, rw)
        u_pi2 = self.control_net(z, action, rw)
        kick2 = force2 - u_pi2

        if self.F_max > 0:
            kick2 = self.F_max * kick2 / (
                self.F_max + kick2.norm(dim=-1, keepdim=True)
            )

        p = p - (h / 2.0) * kick2

        return z, p, phi_eff, hodge_info

    # -----------------------------------------------------------------
    # Jump process
    # -----------------------------------------------------------------

    def _jump_step(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        action: torch.Tensor,
        rw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Poisson jump process for chart transitions.

        Args:
            z: [B, D] position after BAOAB.
            p: [B, D] momentum (unused but passed for API consistency).
            action: [B, A] action vector.
            rw: [B, K] current chart weights.

        Returns:
            z: [B, D] possibly jumped position.
            rw: [B, K] updated chart weights.
            chart_logits: [B, K] chart logits.
            jump_rate: [B, 1] Poisson rate.
            jump_mask: [B] boolean jump events.
        """
        chart_logits = self.chart_predictor(z, action, rw)

        if self.jump_rate_net is not None and self.jump_operator is not None:
            jump_rate = self.jump_rate_net(z, rw)  # [B, 1]
            prob = 1.0 - torch.exp(-jump_rate * self.dt)  # [B, 1]
            uniform = torch.rand_like(prob)
            jump_mask = (uniform < prob).squeeze(-1)  # [B]

            if jump_mask.any():
                source_chart = rw.argmax(dim=-1)  # [B]
                target_chart = chart_logits.argmax(dim=-1)  # [B]

                z_jumped = self.jump_operator(z, source_chart, target_chart)
                z = torch.where(jump_mask.unsqueeze(-1), z_jumped, z)
                z = _project_to_ball(z)
        else:
            jump_rate = z.new_zeros(z.shape[0], 1)
            jump_mask = z.new_zeros(z.shape[0], dtype=torch.bool)

        rw = F.softmax(chart_logits, dim=-1)
        return z, rw, chart_logits, jump_rate, jump_mask

    # -----------------------------------------------------------------
    # Forward (multi-step rollout)
    # -----------------------------------------------------------------

    def forward(
        self,
        z_0: torch.Tensor,
        actions: torch.Tensor,
        router_weights_0: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Multi-step rollout through the world model.

        Args:
            z_0: [B, D] initial latent position.
            actions: [B, H, A] action sequence (H = prediction horizon).
            router_weights_0: [B, K] initial chart routing weights.

        Returns:
            Dictionary with:
                z_trajectory: [B, H, D] predicted latent positions.
                chart_logits: [B, H, K] predicted chart logits per step.
                momenta: [B, H, D] momenta at each step.
                jump_rates: [B, H, 1] Poisson rates (diagnostic).
                jump_masks: [B, H] jump events (diagnostic).
                phi_eff: [B, H, 1] effective potential (diagnostic).
                hodge_conservative_ratio: [B, H] ||f_cons|| / ||f_total||.
                hodge_solenoidal_ratio: [B, H] ||f_sol|| / ||f_total||.
                hodge_harmonic_ratio: [B, H] ||f_harm|| / ||f_total||.
                hodge_harmonic_forces: [B, H, D] harmonic residual forces.
        """
        B, H, _A = actions.shape
        device = z_0.device
        D = self.latent_dim
        K = self.num_charts

        # Initialize momentum from position
        p = self.momentum_init(z_0)  # [B, D]

        z = z_0
        rw = router_weights_0

        # Pre-allocate output tensors
        z_traj = torch.zeros(B, H, D, device=device)
        chart_logits_out = torch.zeros(B, H, K, device=device)
        momenta = torch.zeros(B, H, D, device=device)
        jump_rates = torch.zeros(B, H, 1, device=device)
        jump_masks = torch.zeros(B, H, dtype=torch.bool, device=device)
        phi_eff_out = torch.zeros(B, H, 1, device=device)
        hodge_conservative = torch.zeros(B, H, device=device)
        hodge_solenoidal = torch.zeros(B, H, device=device)
        hodge_harmonic = torch.zeros(B, H, device=device)
        hodge_harmonic_forces = torch.zeros(B, H, D, device=device)

        for t in range(H):
            action_t = actions[:, t, :]  # [B, A]

            # BAOAB step with geodesic corrections + Hodge diagnostics
            z, p, phi_eff, hodge_info = self._baoab_step(z, p, action_t, rw)

            # Hodge decomposition diagnostics
            if hodge_info:
                hodge_conservative[:, t] = hodge_info["conservative_ratio"]
                hodge_solenoidal[:, t] = hodge_info["solenoidal_ratio"]
                hodge_harmonic[:, t] = hodge_info["harmonic_ratio"]
                hodge_harmonic_forces[:, t] = hodge_info["harmonic"]

            # Jump process
            z, rw, cl, jr, jm = self._jump_step(z, p, action_t, rw)

            z_traj[:, t, :] = z
            chart_logits_out[:, t, :] = cl
            momenta[:, t, :] = p
            jump_rates[:, t, :] = jr
            jump_masks[:, t] = jm
            phi_eff_out[:, t, :] = phi_eff

        return {
            "z_trajectory": z_traj,
            "chart_logits": chart_logits_out,
            "momenta": momenta,
            "jump_rates": jump_rates,
            "jump_masks": jump_masks,
            "phi_eff": phi_eff_out,
            "potential_net": self.potential_net,
            "router_weights_final": rw,
            "hodge_conservative_ratio": hodge_conservative,
            "hodge_solenoidal_ratio": hodge_solenoidal,
            "hodge_harmonic_ratio": hodge_harmonic,
            "hodge_harmonic_forces": hodge_harmonic_forces,
        }
