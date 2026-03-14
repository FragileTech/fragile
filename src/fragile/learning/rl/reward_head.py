"""Reward prediction head extending CovariantPotentialNet with boundary conditioning.

Follows the same CovariantAttention pattern, sharing the chart tokenizer
from the existing potential_net so that chart geometry is consistent
between value estimation and reward prediction.
"""

from __future__ import annotations

import torch
from torch import nn

from fragile.learning.core.layers.gauge import CovariantAttention, GeodesicConfig
from fragile.learning.core.layers.primitives import SpectralLinear
from fragile.learning.rl.boundary import lower_control
from fragile.learning.vla.covariant_world_model import (
    ActionTokenizer,
    CovariantPotentialNet,
)


class RewardHead(nn.Module):
    """Predict the conservative and non-conservative components of the reward field.

    Shares ``chart_tok`` weights with ``CovariantPotentialNet`` so that
    chart geometry is consistent. The active Dreamer path conditions on:

    - deterministic boundary action mean
    - tangent latent control field
    - chart weights
    """

    def __init__(
        self,
        potential_net: CovariantPotentialNet,
        action_dim: int,
        d_model: int = 128,
        metric: nn.Module | None = None,
    ) -> None:
        super().__init__()
        latent_dim = potential_net.latent_dim
        self.metric = metric

        # Share chart tokenizer weights with potential_net
        self.chart_tok = potential_net.chart_tok
        self.z_embed = potential_net.z_embed

        # Own action tokenizer and attention
        self.action_tok = ActionTokenizer(action_dim, d_model, latent_dim)
        self.control_tok = ActionTokenizer(latent_dim, d_model, latent_dim)
        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)
        self.reward_attn = CovariantAttention(geo_cfg)
        self.reward_cons_out = SpectralLinear(d_model, 1)
        self.reward_form_out = SpectralLinear(d_model, latent_dim)
        self.reward_curl_out = SpectralLinear(d_model, latent_dim * latent_dim)

    def decompose(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        rw: torch.Tensor,
        control: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decompose reward into conservative and non-conservative pieces."""
        latent_dim = z.shape[-1]
        act_x, act_z = self.action_tok(action, z)
        chart_x, chart_z = self.chart_tok(rw, z)
        control_x, control_z = self.control_tok(control, z)
        ctx_x = torch.cat([act_x, control_x, chart_x], dim=1)
        ctx_z = torch.cat([act_z, control_z, chart_z], dim=1)

        x_q = self.z_embed(z)
        feat, _ = self.reward_attn(z, ctx_z, x_q, ctx_x, ctx_x)

        reward_conservative = self.reward_cons_out(feat)
        reward_form_cov = self.reward_form_out(feat)
        control_cov = lower_control(z, control, metric=self.metric)
        reward_nonconservative = (reward_form_cov * control).sum(dim=-1, keepdim=True)
        curl_raw = self.reward_curl_out(feat).reshape(-1, latent_dim, latent_dim)
        reward_curl = 0.5 * (curl_raw - curl_raw.transpose(1, 2))
        reward_total = reward_conservative + reward_nonconservative
        return {
            "reward_total": reward_total,
            "reward_conservative": reward_conservative,
            "reward_nonconservative": reward_nonconservative,
            "reward_form_cov": reward_form_cov,
            "control_cov": control_cov,
            "reward_curl": reward_curl,
        }

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        rw: torch.Tensor,
        control: torch.Tensor,
    ) -> torch.Tensor:
        """Predict scalar reward.

        Args:
            z: [B, D] latent position.
            action: [B, A] deterministic boundary action.
            rw: [B, K] chart weights.
            control: [B, D] tangent latent control field.

        Returns:
            r_hat: [B, 1] predicted reward.
        """
        return self.decompose(z, action, rw, control)["reward_total"]
