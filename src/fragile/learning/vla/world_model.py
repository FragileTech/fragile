"""Geometric world model with geodesic Boris-BAOAB integration on the Poincaré ball.

Implements Algorithm 22.4.2 from the theory docs: a symplectic BAOAB
integrator with Boris rotation for Lorentz forces, geodesic Christoffel
corrections, and a Poisson jump process for chart transitions.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from fragile.learning.core.layers.atlas import _project_to_ball
from fragile.learning.core.layers.gauge import (
    ConformalMetric,
    christoffel_contraction,
    mobius_add,
    poincare_exp_map,
)
from fragile.learning.core.layers.primitives import NormGatedGELU, SpectralLinear
from fragile.learning.core.layers.topology import FactorizedJumpOperator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_bundle_params(dim: int) -> tuple[int, int]:
    """Resolve hidden_dim into (bundle_size, n_bundles) for NormGatedGELU.

    Tries to use a bundle_size that evenly divides dim and is reasonably
    small.  Falls back to bundle_size=1 (scalar gating).
    """
    for candidate in (16, 8, 4, 2):
        if dim % candidate == 0:
            return candidate, dim // candidate
    return 1, dim


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class EffectivePotentialNet(nn.Module):
    r"""Structured effective potential :math:`\Phi_{\text{eff}}`.

    .. math::
        \Phi_{\text{eff}} = \alpha\, U(z)
                          + (1-\alpha)\, V_{\text{critic}}(z, K)
                          + \gamma_{\text{risk}}\, \Psi_{\text{risk}}(z)

    where :math:`U(z) = -2\operatorname{artanh}(|z|)` is an analytic
    hyperbolic expansion drive (no learnable parameters).
    """

    def __init__(
        self,
        latent_dim: int,
        num_charts: int,
        hidden_dim: int,
        alpha: float = 0.5,
        gamma_risk: float = 0.01,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma_risk = gamma_risk

        # Learned value critic V(z, K)
        bs, nb = _resolve_bundle_params(hidden_dim)
        self.v_critic = nn.Sequential(
            SpectralLinear(latent_dim + num_charts, hidden_dim),
            NormGatedGELU(bundle_size=bs, n_bundles=nb),
            SpectralLinear(hidden_dim, 1),
        )

        # Learned risk penalty Ψ_risk(z)
        self.psi_risk = nn.Sequential(
            SpectralLinear(latent_dim, hidden_dim),
            NormGatedGELU(bundle_size=bs, n_bundles=nb),
            SpectralLinear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, rw: torch.Tensor) -> torch.Tensor:
        """Compute scalar effective potential.

        Args:
            z: [B, D] position in the Poincaré ball.
            rw: [B, K] router / chart weights.

        Returns:
            phi: [B, 1] effective potential.
        """
        # Analytic hyperbolic drive
        r = z.norm(dim=-1, keepdim=True).clamp(min=1e-6, max=1.0 - 1e-6)
        U = -2.0 * torch.atanh(r)  # [B, 1]

        V = self.v_critic(torch.cat([z, rw], dim=-1))  # [B, 1]
        Psi = self.psi_risk(z)  # [B, 1]

        return self.alpha * U + (1.0 - self.alpha) * V + self.gamma_risk * Psi

    def riemannian_gradient(self, z: torch.Tensor, rw: torch.Tensor) -> torch.Tensor:
        r"""Riemannian gradient :math:`\nabla_g \Phi = G^{-1} \nabla \Phi`.

        For the Poincaré ball, :math:`G^{-1} = ((1 - |z|^2)/2)^2 I`.

        Args:
            z: [B, D] position (must have requires_grad or we create graph).
            rw: [B, K] router weights.

        Returns:
            grad_phi: [B, D] Riemannian gradient of Φ_eff.
        """
        z_in = z.detach().requires_grad_(True)
        phi = self.forward(z_in, rw)  # [B, 1]
        euclidean_grad = torch.autograd.grad(
            phi.sum(), z_in, create_graph=True,
        )[0]  # [B, D]
        # G^{-1} = ((1-|z|^2)/2)^2 I  (diagonal, so element-wise multiply)
        r_sq = (z_in**2).sum(dim=-1, keepdim=True)
        g_inv_factor = ((1.0 - r_sq).clamp(min=1e-6) / 2.0) ** 2
        return g_inv_factor * euclidean_grad


class ValueCurlNet(nn.Module):
    r"""Predicts antisymmetric field strength :math:`\mathcal{F}_{ij}` for Boris rotation."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # Number of independent upper-triangle entries
        self.n_upper = latent_dim * (latent_dim - 1) // 2

        bs, nb = _resolve_bundle_params(hidden_dim)
        self.net = nn.Sequential(
            SpectralLinear(latent_dim + action_dim, hidden_dim),
            NormGatedGELU(bundle_size=bs, n_bundles=nb),
            SpectralLinear(hidden_dim, self.n_upper),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute antisymmetric field strength tensor.

        Args:
            z: [B, D] position.
            action: [B, A] action vector.

        Returns:
            F: [B, D, D] antisymmetric tensor.
        """
        B = z.shape[0]
        D = self.latent_dim
        upper = self.net(torch.cat([z, action], dim=-1))  # [B, n_upper]

        # Build antisymmetric matrix from upper triangle
        F = z.new_zeros(B, D, D)
        idx = 0
        for i in range(D):
            for j in range(i + 1, D):
                F[:, i, j] = upper[:, idx]
                F[:, j, i] = -upper[:, idx]
                idx += 1
        return F


class ControlFieldNet(nn.Module):
    """Action-conditioned control force in tangent space."""

    def __init__(self, latent_dim: int, action_dim: int, num_charts: int, hidden_dim: int) -> None:
        super().__init__()
        bs, nb = _resolve_bundle_params(hidden_dim)
        self.net = nn.Sequential(
            SpectralLinear(latent_dim + action_dim + num_charts, hidden_dim),
            NormGatedGELU(bundle_size=bs, n_bundles=nb),
            SpectralLinear(hidden_dim, latent_dim),
        )

    def forward(
        self, z: torch.Tensor, action: torch.Tensor, rw: torch.Tensor,
    ) -> torch.Tensor:
        """Compute control force u_π.

        Args:
            z: [B, D] position.
            action: [B, A] action vector.
            rw: [B, K] chart weights.

        Returns:
            u: [B, D] control force in tangent space.
        """
        return self.net(torch.cat([z, action, rw], dim=-1))


class JumpRateNet(nn.Module):
    """Predicts Poisson jump rate λ(z, K) >= 0."""

    def __init__(self, latent_dim: int, num_charts: int, hidden_dim: int) -> None:
        super().__init__()
        bs, nb = _resolve_bundle_params(hidden_dim)
        self.net = nn.Sequential(
            SpectralLinear(latent_dim + num_charts, hidden_dim),
            NormGatedGELU(bundle_size=bs, n_bundles=nb),
            SpectralLinear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, rw: torch.Tensor) -> torch.Tensor:
        """Compute jump rate.

        Args:
            z: [B, D] position.
            rw: [B, K] chart weights.

        Returns:
            rate: [B, 1] non-negative jump rate.
        """
        return F.softplus(self.net(torch.cat([z, rw], dim=-1)))


class ChartTargetPredictor(nn.Module):
    """Predicts target chart logits for the jump process."""

    def __init__(self, latent_dim: int, action_dim: int, num_charts: int, hidden_dim: int) -> None:
        super().__init__()
        bs, nb = _resolve_bundle_params(hidden_dim)
        self.net = nn.Sequential(
            SpectralLinear(latent_dim + action_dim + num_charts, hidden_dim),
            NormGatedGELU(bundle_size=bs, n_bundles=nb),
            SpectralLinear(hidden_dim, num_charts),
        )

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
        return self.net(torch.cat([z, action, rw], dim=-1))


# ---------------------------------------------------------------------------
# Main world model
# ---------------------------------------------------------------------------


class GeometricWorldModel(nn.Module):
    """Action-conditioned dynamics model using geodesic Boris-BAOAB integration.

    Given an initial latent position ``z_0`` (in the Poincaré ball), a sequence
    of actions, and initial router weights, this model predicts a trajectory of
    future latent states by simulating Langevin dynamics with:

    - **Geodesic corrections**: Christoffel contraction removes spurious
      coordinate acceleration.
    - **Boris rotation**: Norm-preserving treatment of the Lorentz/curl force
      from the value-curl antisymmetric field.
    - **Poincaré exp map at z**: Correct geodesic drift (not origin-based).
    - **Poisson jump process**: Rare chart transitions via Möbius lift-rotate-project.
    - **Spectral-normalised layers**: Lipschitz-bounded networks throughout.

    Args:
        latent_dim: Dimension of the Poincaré ball latent space.
        action_dim: Dimension of the action vector.
        num_charts: Number of atlas charts.
        hidden_dim: Width of the sub-module MLPs.
        dt: Integration time step.
        gamma_friction: Langevin friction coefficient.
        T_c: Thermostat temperature.
        alpha_potential: Balance between analytic drive U and learned critic V.
        beta_curl: Value-curl coupling strength for Lorentz/Boris forces.
        gamma_risk: Risk-stress penalty weight in the effective potential.
        use_boris: Whether to enable Boris rotation for curl forces.
        use_jump: Whether to enable the Poisson jump process.
        jump_rate_hidden: Hidden dimension for the jump rate predictor.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        action_dim: int = 14,
        num_charts: int = 8,
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

        # Ornstein-Uhlenbeck coefficients
        self.c1 = math.exp(-gamma_friction * dt)
        self.c2 = math.sqrt(max(0.0, (1.0 - self.c1**2) * T_c))

        # Metric
        self.metric = ConformalMetric()

        # Sub-modules
        self.potential_net = EffectivePotentialNet(
            latent_dim, num_charts, hidden_dim,
            alpha=alpha_potential, gamma_risk=gamma_risk,
        )
        self.control_net = ControlFieldNet(latent_dim, action_dim, num_charts, hidden_dim)

        if use_boris:
            self.curl_net = ValueCurlNet(latent_dim, action_dim, hidden_dim)
        else:
            self.curl_net = None

        if use_jump:
            self.jump_rate_net = JumpRateNet(latent_dim, num_charts, jump_rate_hidden)
            self.jump_operator = FactorizedJumpOperator(num_charts, latent_dim)
        else:
            self.jump_rate_net = None
            self.jump_operator = None

        self.chart_predictor = ChartTargetPredictor(latent_dim, action_dim, num_charts, hidden_dim)

        # Initial momentum from position
        self.momentum_init = SpectralLinear(latent_dim, latent_dim)

    # -----------------------------------------------------------------
    # Boris rotation
    # -----------------------------------------------------------------

    def _boris_rotation(
        self,
        p_minus: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Norm-preserving Boris rotation for the Lorentz/curl force.

        For D > 3 the cross product generalises to the antisymmetric
        matrix-vector product.  The Boris algorithm ensures
        ``|p_plus| = |p_minus|`` because F is antisymmetric.

        Args:
            p_minus: [B, D] momentum after potential half-kick.
            z: [B, D] current position.
            action: [B, A] action vector.

        Returns:
            p_plus: [B, D] rotated momentum (same norm).
        """
        if self.curl_net is None:
            return p_minus

        h = self.dt
        F = self.curl_net(z, action)  # [B, D, D] antisymmetric
        g_inv = self.metric.metric_inv(z)  # [B, D, D]
        T = (h / 2.0) * self.beta_curl * torch.bmm(g_inv, F)  # [B, D, D]

        # Boris half-rotation
        t_vec = torch.bmm(T, p_minus.unsqueeze(-1)).squeeze(-1)  # [B, D]
        p_prime = p_minus + t_vec

        t_sq = (T ** 2).sum(dim=(-2, -1), keepdim=False)  # [B] Frobenius norm^2
        s_factor = 2.0 / (1.0 + t_sq).unsqueeze(-1)  # [B, 1]
        s_vec = s_factor * torch.bmm(T, p_prime.unsqueeze(-1)).squeeze(-1)  # [B, D]

        return p_minus + s_vec

    # -----------------------------------------------------------------
    # BAOAB integration step
    # -----------------------------------------------------------------

    def _baoab_step(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        action: torch.Tensor,
        rw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        """
        h = self.dt

        # --- B step: momentum kick ---
        grad_phi = self.potential_net.riemannian_gradient(z, rw)  # [B, D]
        u_pi = self.control_net(z, action, rw)  # [B, D]
        kick = grad_phi - u_pi

        p_minus = p - (h / 2.0) * kick
        p_plus = self._boris_rotation(p_minus, z, action)
        p = p_plus - (h / 2.0) * kick

        # --- A step (first half): geodesic drift ---
        g_inv = self.metric.metric_inv(z)  # [B, D, D]
        v = torch.einsum("bij,bj->bi", g_inv, p)  # contravariant velocity
        geo_corr = christoffel_contraction(z, v)
        v_corr = v - (h / 4.0) * geo_corr
        z = poincare_exp_map(z, (h / 2.0) * v_corr)
        z = _project_to_ball(z)

        # --- O step: Ornstein-Uhlenbeck thermostat ---
        cf = self.metric.conformal_factor(z)  # [B, 1]
        xi = torch.randn_like(p)
        p = self.c1 * p + self.c2 * cf * xi

        # --- A step (second half): geodesic drift ---
        g_inv = self.metric.metric_inv(z)
        v = torch.einsum("bij,bj->bi", g_inv, p)
        geo_corr = christoffel_contraction(z, v)
        v_corr = v - (h / 4.0) * geo_corr
        z = poincare_exp_map(z, (h / 2.0) * v_corr)
        z = _project_to_ball(z)

        # --- B step (second half): momentum kick ---
        grad_phi2 = self.potential_net.riemannian_gradient(z, rw)
        u_pi2 = self.control_net(z, action, rw)
        kick2 = grad_phi2 - u_pi2
        p = p - (h / 2.0) * kick2

        # Effective potential for diagnostics
        with torch.no_grad():
            phi_eff = self.potential_net(z, rw)

        return z, p, phi_eff

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

        for t in range(H):
            action_t = actions[:, t, :]  # [B, A]

            # BAOAB step with geodesic corrections
            z, p, phi_eff = self._baoab_step(z, p, action_t, rw)

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
        }
