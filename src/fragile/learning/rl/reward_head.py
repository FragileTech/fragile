"""Reward prediction head extending CovariantPotentialNet with action conditioning.

Follows the same CovariantAttention pattern, sharing the chart tokenizer
from the existing potential_net so that chart geometry is consistent
between value estimation and reward prediction.
"""

from __future__ import annotations

import torch
from torch import nn

from fragile.learning.core.layers.gauge import CovariantAttention, GeodesicConfig
from fragile.learning.core.layers.primitives import SpectralLinear
from fragile.learning.vla.covariant_world_model import (
    ActionTokenizer,
    CovariantPotentialNet,
)


class RewardHead(nn.Module):
    """Predicts instantaneous reward from (z, action, rw).

    Shares ``chart_tok`` weights with ``CovariantPotentialNet`` so that
    chart geometry is consistent.  Uses ``ActionTokenizer`` (same class
    as ``CovariantControlField``) for action conditioning.
    """

    def __init__(
        self,
        potential_net: CovariantPotentialNet,
        action_dim: int,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        latent_dim = potential_net.latent_dim

        # Share chart tokenizer weights with potential_net
        self.chart_tok = potential_net.chart_tok
        self.z_embed = potential_net.z_embed

        # Own action tokenizer and attention
        self.action_tok = ActionTokenizer(action_dim, d_model, latent_dim)
        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)
        self.reward_attn = CovariantAttention(geo_cfg)
        self.reward_out = SpectralLinear(d_model, 1)

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        rw: torch.Tensor,
    ) -> torch.Tensor:
        """Predict scalar reward.

        Args:
            z: [B, D] latent position.
            action: [B, A] action vector.
            rw: [B, K] chart weights.

        Returns:
            r_hat: [B, 1] predicted reward.
        """
        act_x, act_z = self.action_tok(action, z)
        chart_x, chart_z = self.chart_tok(rw, z)

        ctx_x = torch.cat([act_x, chart_x], dim=1)  # [B, A+K, d_model]
        ctx_z = torch.cat([act_z, chart_z], dim=1)  # [B, A+K, D]

        x_q = self.z_embed(z)
        feat, _ = self.reward_attn(z, ctx_z, x_q, ctx_x, ctx_x)
        return self.reward_out(feat)  # [B, 1]
