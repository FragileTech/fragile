"""Geometric Actor: chart-conditioned policy on the Poincare ball.

Reuses the same CovariantAttention + ChartTokenizer + SpectralLinear pattern
as CovariantControlField but WITHOUT ActionTokenizer (the actor *produces*
actions, it doesn't consume them).  Output is a squashed Gaussian (SAC-style).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from fragile.learning.core.layers.gauge import CovariantAttention, GeodesicConfig
from fragile.learning.core.layers.primitives import SpectralLinear
from fragile.learning.vla.covariant_world_model import ChartTokenizer


class GeometricActor(nn.Module):
    """Chart-conditioned Gaussian policy on latent state.

    Architecture mirrors ``CovariantControlField`` (same tokenizer + attention
    pattern) but omits ``ActionTokenizer`` and adds dual output heads
    (mu, log_std) for a squashed Gaussian.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        num_charts: int,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim

        self.chart_tok = ChartTokenizer(num_charts, d_model, latent_dim)
        self.z_embed = SpectralLinear(latent_dim, d_model)

        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)
        self.attn = CovariantAttention(geo_cfg)

        self.mu_head = SpectralLinear(d_model, action_dim)
        self.log_std_head = SpectralLinear(d_model, action_dim)

    def forward(
        self, z: torch.Tensor, rw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log-std of the squashed Gaussian.

        Args:
            z: [B, D] latent position on the Poincare ball.
            rw: [B, K] chart routing weights.

        Returns:
            mu: [B, A] raw mean (before tanh).
            log_std: [B, A] clamped log standard deviation.
        """
        chart_x, chart_z = self.chart_tok(rw, z)
        x_q = self.z_embed(z)
        feat, _ = self.attn(z, chart_z, x_q, chart_x, chart_x)

        mu = self.mu_head(feat)  # [B, A]
        log_std = self.log_std_head(feat)  # [B, A]
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(
        self, z: torch.Tensor, rw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action with its log-probability (reparameterised).

        Uses the tanh-squash correction from SAC:
        ``log pi(a|s) = log N(u; mu, sigma) - sum log(1 - tanh(u)^2)``

        Args:
            z: [B, D] latent position.
            rw: [B, K] chart weights.

        Returns:
            action: [B, A] in (-1, 1).
            log_prob: [B, 1] log-probability.
        """
        mu, log_std = self.forward(z, rw)
        std = log_std.exp()
        noise = torch.randn_like(mu)
        u = mu + std * noise  # pre-tanh sample

        action = torch.tanh(u)

        # Log-prob with tanh correction
        log_prob = (
            -0.5 * ((u - mu) / (std + 1e-8)).pow(2)
            - log_std
            - 0.5 * math.log(2.0 * math.pi)
        )
        # Tanh squash correction: log(1 - tanh^2(u)) = 2*(log 2 - u - softplus(-2u))
        log_prob -= 2.0 * (math.log(2.0) - u - nn.functional.softplus(-2.0 * u))
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # [B, 1]

        return action, log_prob

    def mode(
        self, z: torch.Tensor, rw: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic action (for evaluation)."""
        mu, _ = self.forward(z, rw)
        return torch.tanh(mu)
