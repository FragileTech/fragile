"""Geometric actor on the action manifold.

The actor predicts a continuous action-manifold latent from observation-space
geometry. Raw environment actions are obtained only after decoding the action
latent through the action topo-decoder.
"""

from __future__ import annotations

import torch
from torch import nn

from fragile.learning.core.layers.atlas import _project_to_ball
from fragile.learning.core.layers.gauge import CovariantAttention, GeodesicConfig
from fragile.learning.core.layers.primitives import SpectralLinear
from fragile.learning.vla.covariant_world_model import ChartTokenizer


class GeometricActor(nn.Module):
    """Chart-conditioned stochastic policy over the action manifold."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        latent_dim: int,
        action_latent_dim: int,
        num_charts: int,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.action_latent_dim = action_latent_dim

        self.chart_tok = ChartTokenizer(num_charts, d_model, latent_dim)
        self.z_embed = SpectralLinear(latent_dim, d_model)

        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)
        self.attn = CovariantAttention(geo_cfg)

        self.mu_head = SpectralLinear(d_model, action_latent_dim)
        self.log_std_head = SpectralLinear(d_model, action_latent_dim)

    def forward(
        self, z: torch.Tensor, rw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log-std of the action-latent Gaussian.

        Args:
            z: [B, D] latent position on the Poincare ball.
            rw: [B, K] chart routing weights.

        Returns:
            mu: [B, D_a] action-latent mean projected to the Poincare ball.
            log_std: [B, D_a] clamped log standard deviation.
        """
        chart_x, chart_z = self.chart_tok(rw, z)
        x_q = self.z_embed(z)
        feat, _ = self.attn(z, chart_z, x_q, chart_x, chart_x)

        mu = _project_to_ball(self.mu_head(feat))
        log_std = self.log_std_head(feat)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample_latent(
        self, z: torch.Tensor, rw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a continuous action latent and return its mean and log-std."""
        mu, log_std = self.forward(z, rw)
        std = log_std.exp()
        sample = _project_to_ball(mu + std * torch.randn_like(mu))
        return sample, mu, log_std

    def mode_latent(
        self, z: torch.Tensor, rw: torch.Tensor,
    ) -> torch.Tensor:
        """Return the deterministic action-manifold latent."""
        mu, _ = self.forward(z, rw)
        return mu
