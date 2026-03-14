"""Action-manifold helpers and covariant obs-action closure models."""

from __future__ import annotations

import torch
from torch import nn

from fragile.learning.core.layers.atlas import (
    _poincare_hyperbolic_score,
    _poincare_temperature,
    _poincare_weighted_mean,
    _project_to_ball,
    _routing_weights,
)
from fragile.learning.core.layers.gauge import (
    ConformalMetric,
    CovariantAttention,
    GeodesicConfig,
    mobius_add,
)
from fragile.learning.core.layers.primitives import SpectralLinear
from fragile.learning.rl.boundary import lower_control
from fragile.learning.vla.covariant_world_model import ChartTokenizer


class LatentTokenizer(nn.Module):
    """Embed a single latent point as one covariant attention token."""

    def __init__(self, latent_dim: int, d_model: int) -> None:
        super().__init__()
        self.embed = SpectralLinear(latent_dim, d_model)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embed(z).unsqueeze(1), z.unsqueeze(1)


def symbolize_latent_with_atlas(
    atlas_model: nn.Module,
    z_latent: torch.Tensor,
    *,
    hard_routing: bool = False,
    hard_routing_tau: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Attach atlas/chart/code metadata to a canonical continuous manifold latent.

    ``z_latent`` is treated as the canonical geometric latent ``z_geo``. The
    atlas is only used to derive chart weights, chart/code assignments, and the
    corresponding code latent; it does not replace the continuous latent with a
    purely symbolic reconstruction.
    """
    atlas = getattr(atlas_model, "encoder", atlas_model)
    z_latent = _project_to_ball(z_latent)
    chart_centers = _project_to_ball(atlas.chart_centers)

    cov_router = getattr(atlas, "cov_router", None)
    if cov_router is not None:
        router_weights, chart_idx = cov_router(
            z_latent,
            chart_tokens=chart_centers,
            hard_routing=hard_routing,
            hard_routing_tau=hard_routing_tau,
        )
    else:
        scores = _poincare_hyperbolic_score(
            z_latent,
            chart_centers,
            key_dim=atlas.latent_dim,
            tau_min=atlas.router_tau_min,
            tau_denom_min=atlas.router_tau_denom_min,
            eps=atlas.router_transport_eps,
        )
        latent_router = getattr(atlas, "latent_router", None)
        if latent_router is not None:
            tau = _poincare_temperature(
                z_latent,
                key_dim=atlas.latent_dim,
                tau_min=atlas.router_tau_min,
                tau_denom_min=atlas.router_tau_denom_min,
            )
            scores = scores + 0.1 * latent_router(z_latent) / tau.unsqueeze(1)
        router_weights = _routing_weights(scores, hard_routing, hard_routing_tau)
        chart_idx = router_weights.argmax(dim=-1)

    c_bar = _poincare_weighted_mean(chart_centers, router_weights)
    v_local = _project_to_ball(mobius_add(-c_bar, z_latent))
    z_q, code_idx, _indices, _vq = atlas.dynamics_vq(v_local, router_weights)
    return {
        "z_geo": z_latent,
        "z_latent": z_latent,
        "router_weights": router_weights,
        "chart_idx": chart_idx,
        "code_idx": code_idx,
        "c_bar": c_bar,
        "z_q": z_q,
        "v_local": v_local,
    }


class CovariantObsActionClosureModel(nn.Module):
    """Shared covariant closure over observation and action manifolds."""

    def __init__(
        self,
        latent_dim: int,
        num_obs_charts: int,
        num_action_charts: int,
        obs_codes_per_chart: int,
        action_codes_per_chart: int,
        d_model: int = 128,
        metric: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.metric = metric if metric is not None else ConformalMetric()
        self.obs_codes_per_chart = obs_codes_per_chart
        self.action_codes_per_chart = action_codes_per_chart

        self.obs_chart_tok = ChartTokenizer(num_obs_charts, d_model, latent_dim)
        self.action_chart_tok = ChartTokenizer(num_action_charts, d_model, latent_dim)
        self.obs_lat_tok = LatentTokenizer(latent_dim, d_model)
        self.action_lat_tok = LatentTokenizer(latent_dim, d_model)
        self.obs_code_tok = LatentTokenizer(latent_dim, d_model)
        self.action_code_tok = LatentTokenizer(latent_dim, d_model)

        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)
        self.attn = CovariantAttention(geo_cfg)
        self.obs_query = SpectralLinear(latent_dim, d_model)
        self.action_query = SpectralLinear(latent_dim, d_model)
        self.obs_state_out = SpectralLinear(
            d_model,
            num_obs_charts * obs_codes_per_chart,
        )
        self.action_state_out = SpectralLinear(
            d_model,
            num_action_charts * action_codes_per_chart,
        )
        self.control_out = SpectralLinear(d_model, latent_dim)

    def _context(
        self,
        obs_z: torch.Tensor,
        obs_rw: torch.Tensor,
        obs_code_z: torch.Tensor,
        action_z: torch.Tensor,
        action_rw: torch.Tensor,
        action_code_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_chart_x, obs_chart_z = self.obs_chart_tok(obs_rw, obs_z)
        action_chart_x, action_chart_z = self.action_chart_tok(action_rw, action_z)
        obs_lat_x, obs_lat_z = self.obs_lat_tok(obs_z)
        action_lat_x, action_lat_z = self.action_lat_tok(action_z)
        obs_code_x, obs_code_z_tok = self.obs_code_tok(obs_code_z)
        action_code_x, action_code_z_tok = self.action_code_tok(action_code_z)
        ctx_x = torch.cat(
            [
                obs_lat_x,
                obs_code_x,
                obs_chart_x,
                action_lat_x,
                action_code_x,
                action_chart_x,
            ],
            dim=1,
        )
        ctx_z = torch.cat(
            [
                obs_lat_z,
                obs_code_z_tok,
                obs_chart_z,
                action_lat_z,
                action_code_z_tok,
                action_chart_z,
            ],
            dim=1,
        )
        return ctx_x, ctx_z

    def forward(
        self,
        obs_z: torch.Tensor,
        obs_rw: torch.Tensor,
        obs_code_z: torch.Tensor,
        action_z: torch.Tensor,
        action_rw: torch.Tensor,
        action_code_z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict next symbolic states and the control induced by an action latent."""
        ctx_x, ctx_z = self._context(
            obs_z,
            obs_rw,
            obs_code_z,
            action_z,
            action_rw,
            action_code_z,
        )
        obs_feat, _ = self.attn(
            obs_z,
            ctx_z,
            self.obs_query(obs_z),
            ctx_x,
            ctx_x,
        )
        action_feat, _ = self.attn(
            action_z,
            ctx_z,
            self.action_query(action_z),
            ctx_x,
            ctx_x,
        )
        control_tan = self.control_out(obs_feat)
        control_cov = lower_control(obs_z, control_tan, metric=self.metric)
        return {
            "obs_state_logits": self.obs_state_out(obs_feat),
            "action_state_logits": self.action_state_out(action_feat),
            "control_tan": control_tan,
            "control_cov": control_cov,
        }
