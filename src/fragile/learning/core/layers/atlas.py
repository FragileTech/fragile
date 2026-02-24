from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .gauge import exp_map_zero, hyperbolic_distance, log_map_zero, mobius_add
from .primitives import IsotropicBlock, NormGatedGELU, SpectralLinear
from .topology import FactorizedJumpOperator, InvariantChartClassifier
from .ugn import SoftEquivariantLayer


def _poincare_temperature(
    z: torch.Tensor,
    key_dim: int,
    tau_min: float,
    tau_denom_min: float,
) -> torch.Tensor:
    """Compute position-dependent temperature for Poincare ball."""
    r2 = (z**2).sum(dim=-1)
    denom = (1.0 - r2).clamp(min=tau_denom_min)
    tau = math.sqrt(key_dim) * denom / 2.0
    return tau.clamp(min=tau_min)


def _poincare_hyperbolic_score(
    z: torch.Tensor,
    centers: torch.Tensor,
    key_dim: int,
    tau_min: float,
    tau_denom_min: float,
    eps: float,
) -> torch.Tensor:
    """Compute hyperbolic distance-based scores with metric temperature."""
    z_exp = z.unsqueeze(1)  # [B, 1, D]
    c_exp = centers.unsqueeze(0)  # [1, N_c, D]
    diff = z_exp - c_exp
    dist_sq = (diff**2).sum(dim=-1)  # [B, N_c]
    z_sq = (z**2).sum(dim=-1, keepdim=True)  # [B, 1]
    c_sq = (centers**2).sum(dim=-1).unsqueeze(0)  # [1, N_c]
    denom = (1.0 - z_sq) * (1.0 - c_sq)
    arg = 1.0 + 2.0 * dist_sq / (denom + eps)
    dist = torch.acosh(arg.clamp(min=1.0 + eps))  # [B, N_c]
    tau = _poincare_temperature(z, key_dim, tau_min, tau_denom_min)
    return -dist / tau.unsqueeze(1)


def _project_to_ball(
    z: torch.Tensor,
    max_norm: float = 0.99,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Project points to interior of the Poincare ball."""
    norm = z.norm(dim=-1, keepdim=True).clamp(min=eps)
    scale = (max_norm / norm).clamp(max=1.0)
    return z * scale


def _poincare_weighted_mean(
    points: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Approximate hyperbolic barycenter using log/exp maps at the origin."""
    if points.dim() == 2:
        points = points.unsqueeze(0).expand(weights.shape[0], -1, -1)
    w = weights.unsqueeze(-1)
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=eps)
    tangent = log_map_zero(points)
    mean_tan = (w * tangent).sum(dim=1) / w_sum.squeeze(1)
    return exp_map_zero(mean_tan)


def _poincare_weighted_mean_per_chart(
    points: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-chart hyperbolic barycenter for codebook soft assignment."""
    points_exp = points.unsqueeze(0).expand(weights.shape[0], -1, -1, -1)
    tangent = log_map_zero(points_exp)
    w = weights.unsqueeze(-1)
    w_sum = w.sum(dim=2, keepdim=True).clamp(min=eps)
    mean_tan = (w * tangent).sum(dim=2) / w_sum.squeeze(2)
    return exp_map_zero(mean_tan)


class AttentiveAtlasEncoder(nn.Module):
    """Attentive Atlas encoder with cross-attention routing."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_charts: int = 3,
        codes_per_chart: int = 21,
        soft_equiv_metric: bool = False,
        soft_equiv_bundle_size: int | None = None,
        soft_equiv_hidden_dim: int = 64,
        soft_equiv_use_spectral_norm: bool = True,
        soft_equiv_zero_self_mixing: bool = False,
        soft_equiv_soft_assign: bool = True,
        soft_equiv_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.latent_dim = latent_dim
        self.codes_per_chart = codes_per_chart

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.val_proj = nn.Linear(hidden_dim, latent_dim)

        # Unit-sphere init: each chart gets a distinct catchment region from
        # the first forward pass, preventing softmax winner-take-all collapse.
        self.chart_centers = nn.Parameter(
            torch.nn.functional.normalize(torch.randn(num_charts, latent_dim), dim=-1)
        )

        self.codebook = nn.Parameter(torch.randn(num_charts, codes_per_chart, latent_dim) * 0.02)

        self.soft_equiv_layers: nn.ModuleList | None = None
        if soft_equiv_metric:
            bundle_size = soft_equiv_bundle_size or latent_dim
            if bundle_size <= 0:
                msg = "soft_equiv_bundle_size must be positive."
                raise ValueError(msg)
            if latent_dim % bundle_size != 0:
                msg = "latent_dim must be divisible by soft_equiv_bundle_size."
                raise ValueError(msg)
            n_bundles = latent_dim // bundle_size
            self.soft_equiv_layers = nn.ModuleList([
                SoftEquivariantLayer(
                    n_bundles=n_bundles,
                    bundle_dim=bundle_size,
                    hidden_dim=soft_equiv_hidden_dim,
                    use_spectral_norm=soft_equiv_use_spectral_norm,
                    zero_self_mixing=soft_equiv_zero_self_mixing,
                )
                for _ in range(num_charts)
            ])
            _init_soft_equiv_layers(self.soft_equiv_layers)

        self.soft_equiv_soft_assign = soft_equiv_soft_assign
        self.soft_equiv_temperature = soft_equiv_temperature
        if soft_equiv_metric and self.soft_equiv_soft_assign and self.soft_equiv_temperature <= 0:
            msg = "soft_equiv_temperature must be positive when soft_equiv_soft_assign is enabled."
            raise ValueError(msg)
        self._last_soft_equiv_log_ratio: torch.Tensor | None = None
        self._last_soft_equiv_log_ratio: torch.Tensor | None = None

        self.structure_filter = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2 if latent_dim > 2 else latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim // 2 if latent_dim > 2 else latent_dim, latent_dim),
        )

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def _apply_soft_equiv_metric(self, diff: torch.Tensor) -> torch.Tensor:
        if self.soft_equiv_layers is None:
            self._last_soft_equiv_log_ratio = None
            return (diff**2).sum(dim=-1)
        batch_size, _num_charts, num_codes, latent_dim = diff.shape
        ratio_max = 50.0
        eps = 1e-6
        transformed = []
        log_ratio_losses = []
        for chart_idx, layer in enumerate(self.soft_equiv_layers):
            diff_chart = diff[:, chart_idx].reshape(-1, latent_dim)
            diff_chart = torch.nan_to_num(diff_chart, nan=0.0, posinf=0.0, neginf=0.0)
            diff_out = layer(diff_chart)
            diff_out = torch.nan_to_num(diff_out, nan=0.0, posinf=0.0, neginf=0.0)
            in_norm = diff_chart.norm(dim=-1, keepdim=True).clamp(min=eps)
            out_norm = diff_out.norm(dim=-1, keepdim=True)
            ratio = out_norm / in_norm
            ratio_clamped = ratio.clamp(max=ratio_max)
            scale = torch.where(ratio > 0, ratio_clamped / ratio, torch.ones_like(ratio))
            diff_out *= scale
            log_ratio = torch.log(ratio.clamp(min=eps, max=ratio_max))
            log_ratio_losses.append((log_ratio**2).mean())
            transformed.append(diff_out.view(batch_size, num_codes, latent_dim))
        diff_out = torch.stack(transformed, dim=1)
        if log_ratio_losses:
            self._last_soft_equiv_log_ratio = torch.stack(log_ratio_losses).mean()
        else:
            self._last_soft_equiv_log_ratio = torch.tensor(0.0, device=diff.device)
        return (diff_out**2).sum(dim=-1)

    def soft_equiv_l1_loss(self) -> torch.Tensor:
        if self.soft_equiv_layers is None:
            return torch.tensor(0.0, device=self.codebook.device)
        total = torch.zeros((), device=self.codebook.device)
        for layer in self.soft_equiv_layers:
            total += layer.l1_loss()
        return total / len(self.soft_equiv_layers)

    def soft_equiv_log_ratio_loss(self) -> torch.Tensor:
        if self.soft_equiv_layers is None or self._last_soft_equiv_log_ratio is None:
            return torch.tensor(0.0, device=self.codebook.device)
        return self._last_soft_equiv_log_ratio

    def forward(
        self,
        x: torch.Tensor,
        hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Forward pass through the attentive atlas.

        Returns:
            K_chart: [B] chart assignment
            K_code: [B] code index within chart
            z_n: [B, D] nuisance latent
            z_tex: [B, D] texture residual
            router_weights: [B, N_c] routing weights
            z_geo: [B, D] geometric latent
            vq_loss: [] VQ loss
            indices_stack: [B, N_c] code indices per chart
            z_n_all_charts: [B, N_c, D] per-chart nuisance
            c_bar: [B, D] chart center mixture
            v_local: [B, D] local residual (for code collapse penalty)
        """
        # Extract features via fully-connected front-end.
        features = self._encode_features(x)  # [B, H]

        # Map features into chart-space coordinates.
        v = self.val_proj(features)  # [B, D]
        scores = torch.matmul(v, self.chart_centers.t()) / math.sqrt(self.latent_dim)  # [B, N_c]
        # Chart routing distributes mass across atlas charts.
        router_weights = _routing_weights(scores, hard_routing, hard_routing_tau)  # [B, N_c]
        K_chart = torch.argmax(router_weights, dim=1)  # [B]

        # Chart-center mixture is the macro coordinate; local residual is per-chart.
        c_bar = torch.matmul(router_weights, self.chart_centers)  # [B, D]
        v_local = v - c_bar  # [B, D]

        # Per-chart codebook lookup under the (optional) soft-equivariant metric.
        codebook = self.codebook.unsqueeze(0)  # [1, N_c, K, D]
        v_exp = v_local.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
        diff = v_exp - codebook  # [B, N_c, K, D]
        dist = self._apply_soft_equiv_metric(diff)  # [B, N_c, K]
        indices = torch.argmin(dist, dim=-1)  # [B, N_c]
        indices_stack = indices  # [B, N_c]

        indices_exp = indices.unsqueeze(-1).unsqueeze(-1)  # [B, N_c, 1, 1]
        indices_exp = indices_exp.expand(-1, -1, 1, self.latent_dim)  # [B, N_c, 1, D]
        z_q_all = torch.gather(codebook.expand(v.shape[0], -1, -1, -1), 2, indices_exp)
        z_q_all = z_q_all.squeeze(2)  # [B, N_c, D]
        if self.soft_equiv_layers is not None and self.soft_equiv_soft_assign:
            temperature = max(self.soft_equiv_temperature, 1e-6)
            weights = F.softmax(-dist / temperature, dim=-1)
            z_q_soft = (weights.unsqueeze(-1) * codebook).sum(dim=2)
            # Straight-through soft assignment so gradients reach the metric network.
            z_q_all += z_q_soft - z_q_soft.detach()

        # VQ loss using geodesic distance in the Poincaré ball.
        v_bc = v_local.unsqueeze(1)  # [B, 1, D] (kept for delta computation below)
        v_proj = _project_to_ball(v_bc.expand_as(z_q_all))
        w_det = router_weights.detach()  # [B, N_c]
        d_codebook = hyperbolic_distance(z_q_all, v_proj.detach())  # [B, N_c]
        codebook_loss = (d_codebook ** 2 * w_det).mean(0).sum()
        d_commit = hyperbolic_distance(z_q_all.detach(), v_proj)  # [B, N_c]
        commitment = (d_commit ** 2 * w_det).mean(0).sum()
        vq_loss = codebook_loss + 0.25 * commitment

        # Blend codes across charts to form the macro latent per sample.
        z_q_blended = (z_q_all * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        K_code = indices_stack.gather(1, K_chart.unsqueeze(1)).squeeze(1)  # [B]

        # Structure filter extracts the gauge nuisance from per-chart residuals.
        delta = v_bc - z_q_all.detach()  # [B, N_c, D]
        z_n_all = self.structure_filter(delta.reshape(-1, self.latent_dim))  # [B*N_c, D]
        z_n_all_charts = z_n_all.view(v.shape[0], self.num_charts, self.latent_dim)  # [B, N_c, D]

        # Texture residual is what's left after nuisance subtraction.
        z_n = (z_n_all_charts * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        delta_blended = v_local - z_q_blended.detach()  # [B, D]
        z_tex = delta_blended - z_n  # [B, D]

        # Geometric latent = chart center + macro code + nuisance.
        z_q_st = v_local + (z_q_blended - v_local).detach()  # [B, D]
        z_geo = c_bar + z_q_st + z_n  # [B, D]

        return (
            K_chart,
            K_code,
            z_n,
            z_tex,
            router_weights,
            z_geo,
            vq_loss,
            indices_stack,
            z_n_all_charts,
            c_bar,
            v_local,
        )


class TopologicalDecoder(nn.Module):
    """Inverse atlas decoder with optional autonomous routing."""

    def __init__(
        self,
        latent_dim: int = 2,
        hidden_dim: int = 32,
        num_charts: int = 3,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        weight = torch.empty(num_charts, hidden_dim, latent_dim)
        nn.init.uniform_(weight, -1.0 / math.sqrt(latent_dim), 1.0 / math.sqrt(latent_dim))
        self.chart_weight = nn.Parameter(weight)
        self.chart_bias = nn.Parameter(torch.zeros(num_charts, hidden_dim))

        self.latent_router = nn.Linear(latent_dim, num_charts)
        self.tex_residual = nn.Linear(latent_dim, output_dim)
        self.tex_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.renderer = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.render_skip = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        z_geo: torch.Tensor,
        z_tex: torch.Tensor | None = None,
        chart_index: torch.Tensor | None = None,
        router_weights: torch.Tensor | None = None,
        hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode from latent geometry.

        Args:
            z_geo: [B, D] geometric latent
            z_tex: [B, D] optional texture residual (added to output)
            chart_index: [B] optional chart ids

        Returns:
            x_hat: [B, D_out] reconstruction
            router_weights: [B, N_c] routing weights
        """
        # Clamp geometry inside atlas chart range.
        z_geo = torch.tanh(z_geo)
        if router_weights is not None:
            if router_weights.ndim != 2 or router_weights.shape[1] != self.num_charts:
                msg = "router_weights must have shape [B, N_c]."
                raise ValueError(msg)
        elif chart_index is not None:
            router_weights = F.one_hot(
                chart_index, num_classes=self.num_charts
            ).float()  # [B, N_c]
        else:
            # Autonomous routing predicts chart membership from geometry.
            logits = self.latent_router(z_geo)  # [B, N_c]
            router_weights = _routing_weights(logits, hard_routing, hard_routing_tau)  # [B, N_c]

        # Chart-specific linear maps reconstruct local observations.
        h_stack = torch.einsum("bl,chl->bch", z_geo, self.chart_weight)  # [B, N_c, H]
        h_stack += self.chart_bias.unsqueeze(0)  # [B, N_c, H]
        h_global = (h_stack * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, H]

        x_hat = self.renderer(h_global) + self.render_skip(h_global)  # [B, D_out]
        if z_tex is not None:
            # Texture residual injects high-frequency details.
            z_tex = torch.tanh(z_tex)
            x_hat += self.tex_residual_scale * self.tex_residual(z_tex)
        return x_hat, router_weights


class TopoEncoder(nn.Module):
    """Attentive Atlas encoder + topological decoder."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_charts: int = 3,
        codes_per_chart: int = 21,
        soft_equiv_metric: bool = False,
        soft_equiv_bundle_size: int | None = None,
        soft_equiv_hidden_dim: int = 64,
        soft_equiv_use_spectral_norm: bool = True,
        soft_equiv_zero_self_mixing: bool = False,
        soft_equiv_soft_assign: bool = True,
        soft_equiv_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts

        self.encoder = AttentiveAtlasEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_charts=num_charts,
            codes_per_chart=codes_per_chart,
            soft_equiv_metric=soft_equiv_metric,
            soft_equiv_bundle_size=soft_equiv_bundle_size,
            soft_equiv_hidden_dim=soft_equiv_hidden_dim,
            soft_equiv_use_spectral_norm=soft_equiv_use_spectral_norm,
            soft_equiv_zero_self_mixing=soft_equiv_zero_self_mixing,
            soft_equiv_soft_assign=soft_equiv_soft_assign,
            soft_equiv_temperature=soft_equiv_temperature,
        )
        self.decoder = TopologicalDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_charts=num_charts,
            output_dim=input_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        use_hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Full forward pass.

        Args:
            x: [B, D_in] input tensor
            use_hard_routing: whether to use hard routing in decoder

        Returns:
            x_recon: [B, D_in] reconstruction
            vq_loss: [] VQ loss
            enc_router_weights: [B, N_c] encoder routing
            dec_router_weights: [B, N_c] decoder routing
            K_chart: [B] chart assignments
            z_geo: [B, D] geometric latent (macro + gauge residual)
            z_n: [B, D] nuisance latent (continuous gauge vector)
            c_bar: [B, D] chart center mixture
        """
        (
            K_chart,
            _K_code,
            z_n,
            z_tex,
            enc_router_weights,
            z_geo,
            vq_loss,
            _indices,
            _z_n_all,
            c_bar,
            _v_local,
        ) = self.encoder(
            x,
            hard_routing=use_hard_routing,
            hard_routing_tau=hard_routing_tau,
        )

        router_override = enc_router_weights if use_hard_routing else None
        x_recon, dec_router_weights = self.decoder(
            z_geo,
            z_tex,
            chart_index=None,
            router_weights=router_override,
            hard_routing=use_hard_routing,
            hard_routing_tau=hard_routing_tau,
        )

        return x_recon, vq_loss, enc_router_weights, dec_router_weights, K_chart, z_geo, z_n, c_bar

    def compute_consistency_loss(
        self, enc_weights: torch.Tensor, dec_weights: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """KL divergence between encoder and decoder routing.

        Args:
            enc_weights: [B, N_c] encoder routing weights
            dec_weights: [B, N_c] decoder routing weights

        Returns:
            loss: [] consistency loss
        """
        kl = (enc_weights * torch.log((enc_weights + eps) / (dec_weights + eps))).sum(
            dim=-1
        )  # [B]
        return kl.mean()

    def compute_perplexity(self, K_chart: torch.Tensor) -> float:
        """Chart usage perplexity.

        Args:
            K_chart: [B] chart assignments

        Returns:
            perplexity: scalar perplexity
        """
        counts = torch.bincount(K_chart, minlength=self.num_charts).float()  # [N_c]
        probs = counts / counts.sum()  # [N_c]
        probs = probs[probs > 0]  # [N_c_nonzero]
        entropy = -(probs * torch.log(probs)).sum()  # []
        return math.exp(entropy.item())


def _resolve_bundle_params(
    hidden_dim: int,
    latent_dim: int,
    bundle_size: int | None,
) -> tuple[int, int]:
    if bundle_size is None:
        if latent_dim > 0 and hidden_dim % latent_dim == 0:
            bundle_size = latent_dim
        else:
            bundle_size = 1
    if bundle_size <= 0:
        msg = "bundle_size must be positive."
        raise ValueError(msg)
    if hidden_dim % bundle_size != 0:
        msg = "hidden_dim must be divisible by bundle_size."
        raise ValueError(msg)
    return bundle_size, hidden_dim // bundle_size


def _init_soft_equiv_layers(layers: nn.ModuleList) -> None:
    """Initialize soft-equivariant layers to be purely equivariant (no mixing)."""
    with torch.no_grad():
        for layer in layers:
            if isinstance(layer.mixing_weights, torch.Tensor):
                layer.mixing_weights.zero_()
            else:
                for row in layer.mixing_weights:
                    for weight in row:
                        weight.zero_()


class CovariantChartRouter(nn.Module):
    """Gauge-covariant chart router with hyperbolic transport and metric-aware temperature.

    Uses O(n) Poincaré ball parallel transport instead of O(n³) Cayley transform.
    """

    def __init__(
        self,
        latent_dim: int,
        key_dim: int,
        num_charts: int,
        feature_dim: int | None = None,
        tensorization: str = "full",
        rank: int = 8,
        tau_min: float = 1e-2,
        tau_denom_min: float = 1e-3,
        use_transport: bool = True,
        transport_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.key_dim = key_dim
        self.num_charts = num_charts
        self.tensorization = tensorization
        self.tau_min = tau_min
        self.tau_denom_min = tau_denom_min
        self.use_transport = use_transport
        self.transport_eps = transport_eps

        if feature_dim is not None:
            self.q_feat_proj = SpectralLinear(feature_dim, key_dim, bias=True)
        else:
            self.q_feat_proj = None
        self.q_z_proj = SpectralLinear(latent_dim, key_dim, bias=True)

        if tensorization == "full":
            self.q_gamma = nn.Parameter(torch.randn(key_dim, latent_dim, latent_dim) * 0.02)
            self.q_gamma_out = None
            self.q_gamma_u = None
            self.q_gamma_v = None
        elif tensorization == "sum":
            self.q_gamma_out = nn.Parameter(torch.randn(rank, key_dim) * 0.02)
            self.q_gamma_u = nn.Parameter(torch.randn(rank, latent_dim) * 0.02)
            self.q_gamma_v = nn.Parameter(torch.randn(rank, latent_dim) * 0.02)
            self.q_gamma = None
        else:
            msg = "tensorization must be 'full' or 'sum'."
            raise ValueError(msg)

        self.chart_queries = nn.Parameter(torch.randn(num_charts, key_dim) * 0.02)
        self.chart_key_proj = SpectralLinear(latent_dim, key_dim, bias=False)

        # Note: transport_proj removed - using O(n) hyperbolic transport instead

    def _gamma_term(self, z: torch.Tensor) -> torch.Tensor:
        if self.tensorization == "full":
            # Quadratic term captures Christoffel-symbol curvature corrections.
            z_outer = z.unsqueeze(2) * z.unsqueeze(1)  # [B, D, D]
            return torch.einsum("bij,kij->bk", z_outer, self.q_gamma)
        # Low-rank quadratic term for efficiency.
        z_u = z @ self.q_gamma_u.t()  # [B, R]
        z_v = z @ self.q_gamma_v.t()  # [B, R]
        return (z_u * z_v) @ self.q_gamma_out  # [B, K]

    def _conformal_factor(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Poincaré ball conformal factor λ(z) = 2 / (1 - |z|²)."""
        r2 = (z**2).sum(dim=-1, keepdim=True)
        r2 = torch.clamp(r2, max=1.0 - self.transport_eps)
        return 2.0 / (1.0 - r2 + self.transport_eps)

    def _transport_queries(
        self, z: torch.Tensor, chart_tokens: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Transport chart queries using O(n) hyperbolic parallel transport.

        In the Poincaré ball, parallel transport from the origin scales vectors
        by the conformal factor ratio. This replaces the O(n³) Cayley transform.
        """
        batch_size = z.shape[0]
        if chart_tokens is None:
            base_queries = self.chart_queries
        else:
            if chart_tokens.ndim != 2 or chart_tokens.shape[0] != self.num_charts:
                msg = "chart_tokens must have shape [N_c, D] or [N_c, K]."
                raise ValueError(msg)
            if chart_tokens.shape[1] == self.key_dim:
                base_queries = chart_tokens
            elif chart_tokens.shape[1] == self.latent_dim:
                base_queries = self.chart_key_proj(chart_tokens)
            else:
                msg = "chart_tokens must have shape [N_c, D] or [N_c, K]."
                raise ValueError(msg)

        if not self.use_transport:
            return base_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # O(n) hyperbolic parallel transport using conformal factors
        # Transport from origin (where chart_queries live) to z
        # P_{0→z}(v) = v / λ(z) (scales by inverse conformal factor)
        lambda_z = self._conformal_factor(z)  # [B, 1]

        # Expand base_queries: [N_c, K] -> [B, N_c, K]
        queries_expanded = base_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply transport scaling: divide by conformal factor at destination
        # This preserves the hyperbolic norm of the queries
        return queries_expanded / lambda_z.unsqueeze(1)

    def _temperature(self, z: torch.Tensor) -> torch.Tensor:
        # Poincare conformal factor scales attention temperature by radius.
        r2 = (z**2).sum(dim=-1)
        denom = (1.0 - r2).clamp(min=self.tau_denom_min)
        tau = math.sqrt(self.key_dim) * denom / 2.0
        return tau.clamp(min=self.tau_min)

    def _hyperbolic_score(self, z: torch.Tensor, chart_centers: torch.Tensor) -> torch.Tensor:
        """Compute logits based on negative hyperbolic distance. O(N*D).

        Uses the Poincaré ball distance formula for efficient chart scoring
        without requiring matrix operations.

        Args:
            z: [B, D] latent positions
            chart_centers: [N_c, D] chart center positions

        Returns:
            scores: [B, N_c] negative distances (higher = closer)
        """
        # z: [B, D], chart_centers: [N_c, D]
        z_exp = z.unsqueeze(1)  # [B, 1, D]
        c_exp = chart_centers.unsqueeze(0)  # [1, N_c, D]

        # Squared Euclidean norm of difference
        diff = z_exp - c_exp
        dist_sq = (diff**2).sum(dim=-1)  # [B, N_c]

        # Boundary terms (1 - |z|²) and (1 - |c|²)
        z_sq = (z**2).sum(dim=-1, keepdim=True)  # [B, 1]
        c_sq = (chart_centers**2).sum(dim=-1).unsqueeze(0)  # [1, N_c]
        denom = (1 - z_sq) * (1 - c_sq)  # [B, N_c]

        # Poincaré distance formula: d(z, c) = acosh(1 + 2 * |z-c|² / ((1-|z|²)(1-|c|²)))
        arg = 1 + 2 * dist_sq / (denom + self.transport_eps)
        dist = torch.acosh(arg.clamp(min=1.0 + self.transport_eps))  # [B, N_c]

        # Temperature scaling
        tau = self._temperature(z)  # [B]
        return -dist / tau.unsqueeze(1)

    def forward(
        self,
        z: torch.Tensor,
        features: torch.Tensor | None = None,
        chart_tokens: torch.Tensor | None = None,
        hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route to charts using hyperbolic distance scoring.

        Args:
            z: [B, D] latent positions
            features: [B, F] optional feature vectors
            chart_tokens: [N_c, D] optional chart centers (defaults to self.chart_centers)

        Returns:
            router_weights: [B, N_c] routing weights
            K_chart: [B] argmax chart assignments
        """
        # Get chart centers for scoring
        if chart_tokens is not None:
            if chart_tokens.ndim != 2 or chart_tokens.shape[0] != self.num_charts:
                msg = "chart_tokens must have shape [N_c, D]."
                raise ValueError(msg)
            # Project to latent dim if needed
            if chart_tokens.shape[1] != self.latent_dim:
                # Use key_proj if chart_tokens are in key space
                centers = chart_tokens
            else:
                centers = chart_tokens
        else:
            # Use learned chart queries projected to latent space
            # Note: chart_queries are in key_dim, we need latent_dim for distance
            # Fall back to using q_z_proj inverse or just use chart_queries directly
            centers = self.chart_queries[:, : self.latent_dim]  # Truncate to latent_dim

        # O(n) hyperbolic distance-based scoring
        scores = self._hyperbolic_score(z, centers)

        # Optional: add feature-based corrections via gamma term
        if self.q_feat_proj is not None and features is not None:
            q = self.q_z_proj(z)
            q += self.q_feat_proj(features)
            q += self._gamma_term(z)
            # Add small correction from feature projection
            keys = self._transport_queries(z, chart_tokens=chart_tokens)
            feature_scores = (keys * q.unsqueeze(1)).sum(dim=-1)
            tau = self._temperature(z)
            scores = scores + 0.1 * feature_scores / tau.unsqueeze(1)

        router_weights = _routing_weights(scores, hard_routing, hard_routing_tau)
        K_chart = torch.argmax(router_weights, dim=1)
        return router_weights, K_chart


class PrimitiveAttentiveAtlasEncoder(nn.Module):
    """Attentive Atlas encoder using gauge-covariant primitives."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_charts: int = 3,
        codes_per_chart: int = 21,
        bundle_size: int | None = None,
        covariant_attn: bool = True,
        covariant_attn_tensorization: str = "full",
        covariant_attn_rank: int = 8,
        covariant_attn_tau_min: float = 1e-2,
        covariant_attn_denom_min: float = 1e-3,
        covariant_attn_use_transport: bool = True,
        covariant_attn_transport_eps: float = 1e-3,
        soft_equiv_metric: bool = False,
        soft_equiv_bundle_size: int | None = None,
        soft_equiv_hidden_dim: int = 64,
        soft_equiv_use_spectral_norm: bool = True,
        soft_equiv_zero_self_mixing: bool = False,
        soft_equiv_soft_assign: bool = True,
        soft_equiv_temperature: float = 1.0,
        conv_backbone: bool = False,
        img_channels: int = 1,
        img_size: int = 28,
        conv_channels: int = 0,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.latent_dim = latent_dim
        self.codes_per_chart = codes_per_chart
        self.covariant_attn = covariant_attn
        self.router_tau_min = covariant_attn_tau_min
        self.router_tau_denom_min = covariant_attn_denom_min
        self.router_transport_eps = covariant_attn_transport_eps

        bundle_size, n_bundles = _resolve_bundle_params(hidden_dim, latent_dim, bundle_size)

        if conv_backbone:
            from .vision import ConvFeatureExtractor

            self.feature_extractor = ConvFeatureExtractor(
                img_channels, hidden_dim, img_size, conv_channels,
            )
        else:
            self.feature_extractor = nn.Sequential(
                SpectralLinear(input_dim, hidden_dim, bias=True),
                NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
                SpectralLinear(hidden_dim, hidden_dim, bias=True),
                NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
            )
        if covariant_attn:
            self.cov_router = CovariantChartRouter(
                latent_dim=latent_dim,
                key_dim=hidden_dim,
                num_charts=num_charts,
                feature_dim=hidden_dim,
                tensorization=covariant_attn_tensorization,
                rank=covariant_attn_rank,
                tau_min=covariant_attn_tau_min,
                tau_denom_min=covariant_attn_denom_min,
                use_transport=covariant_attn_use_transport,
                transport_eps=covariant_attn_transport_eps,
            )
            self.key_proj = None
            self.chart_queries = None
            self.scale = None
        else:
            self.key_proj = SpectralLinear(hidden_dim, hidden_dim, bias=True)
            self.chart_queries = nn.Parameter(torch.randn(num_charts, hidden_dim) * 0.02)
            self.scale = math.sqrt(hidden_dim)

        self.val_proj = SpectralLinear(hidden_dim, latent_dim, bias=True)

        # Unit-sphere init: each chart gets a distinct catchment region from
        # the first forward pass, preventing softmax winner-take-all collapse.
        self.chart_centers = nn.Parameter(
            torch.nn.functional.normalize(torch.randn(num_charts, latent_dim), dim=-1)
        )

        self.codebook = nn.Parameter(torch.randn(num_charts, codes_per_chart, latent_dim) * 0.02)

        self.soft_equiv_layers: nn.ModuleList | None = None
        if soft_equiv_metric:
            bundle_size = soft_equiv_bundle_size or latent_dim
            if bundle_size <= 0:
                msg = "soft_equiv_bundle_size must be positive."
                raise ValueError(msg)
            if latent_dim % bundle_size != 0:
                msg = "latent_dim must be divisible by soft_equiv_bundle_size."
                raise ValueError(msg)
            n_bundles = latent_dim // bundle_size
            self.soft_equiv_layers = nn.ModuleList([
                SoftEquivariantLayer(
                    n_bundles=n_bundles,
                    bundle_dim=bundle_size,
                    hidden_dim=soft_equiv_hidden_dim,
                    use_spectral_norm=soft_equiv_use_spectral_norm,
                    zero_self_mixing=soft_equiv_zero_self_mixing,
                )
                for _ in range(num_charts)
            ])
            _init_soft_equiv_layers(self.soft_equiv_layers)

        self.soft_equiv_soft_assign = soft_equiv_soft_assign
        self.soft_equiv_temperature = soft_equiv_temperature
        if soft_equiv_metric and self.soft_equiv_soft_assign and self.soft_equiv_temperature <= 0:
            msg = "soft_equiv_temperature must be positive when soft_equiv_soft_assign is enabled."
            raise ValueError(msg)

        self.structure_filter = nn.Sequential(
            IsotropicBlock(latent_dim, latent_dim, bundle_size=latent_dim),
            SpectralLinear(latent_dim, latent_dim, bias=True),
        )

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def _apply_soft_equiv_metric(self, diff: torch.Tensor) -> torch.Tensor:
        if self.soft_equiv_layers is None:
            self._last_soft_equiv_log_ratio = None
            return (diff**2).sum(dim=-1)
        batch_size, _num_charts, num_codes, latent_dim = diff.shape
        ratio_max = 50.0
        eps = 1e-6
        transformed = []
        log_ratio_losses = []
        for chart_idx, layer in enumerate(self.soft_equiv_layers):
            diff_chart = diff[:, chart_idx].reshape(-1, latent_dim)
            diff_chart = torch.nan_to_num(diff_chart, nan=0.0, posinf=0.0, neginf=0.0)
            diff_out = layer(diff_chart)
            diff_out = torch.nan_to_num(diff_out, nan=0.0, posinf=0.0, neginf=0.0)
            in_norm = diff_chart.norm(dim=-1, keepdim=True).clamp(min=eps)
            out_norm = diff_out.norm(dim=-1, keepdim=True)
            ratio = out_norm / in_norm
            ratio_clamped = ratio.clamp(max=ratio_max)
            scale = torch.where(ratio > 0, ratio_clamped / ratio, torch.ones_like(ratio))
            diff_out *= scale
            log_ratio = torch.log(ratio.clamp(min=eps, max=ratio_max))
            log_ratio_losses.append((log_ratio**2).mean())
            transformed.append(diff_out.view(batch_size, num_codes, latent_dim))
        diff_out = torch.stack(transformed, dim=1)
        if log_ratio_losses:
            self._last_soft_equiv_log_ratio = torch.stack(log_ratio_losses).mean()
        else:
            self._last_soft_equiv_log_ratio = torch.tensor(0.0, device=diff.device)
        return (diff_out**2).sum(dim=-1)

    def soft_equiv_l1_loss(self) -> torch.Tensor:
        if self.soft_equiv_layers is None:
            return torch.tensor(0.0, device=self.codebook.device)
        total = torch.zeros((), device=self.codebook.device)
        for layer in self.soft_equiv_layers:
            total += layer.l1_loss()
        return total / len(self.soft_equiv_layers)

    def soft_equiv_log_ratio_loss(self) -> torch.Tensor:
        if self.soft_equiv_layers is None or self._last_soft_equiv_log_ratio is None:
            return torch.tensor(0.0, device=self.codebook.device)
        return self._last_soft_equiv_log_ratio

    def forward(
        self,
        x: torch.Tensor,
        hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Forward pass through the attentive atlas."""
        # Extract features and map into chart coordinates (Poincare ball).
        features = self._encode_features(x)  # [B, H]
        v = _project_to_ball(self.val_proj(features))  # [B, D]
        chart_centers = _project_to_ball(self.chart_centers)  # [N_c, D]

        if self.covariant_attn:
            # Covariant router performs gauge-aware chart assignment.
            router_weights, K_chart = self.cov_router(
                v,
                features=features,
                chart_tokens=chart_centers,
                hard_routing=hard_routing,
                hard_routing_tau=hard_routing_tau,
            )
        else:
            scores = _poincare_hyperbolic_score(
                v,
                chart_centers,
                key_dim=self.latent_dim,
                tau_min=self.router_tau_min,
                tau_denom_min=self.router_tau_denom_min,
                eps=self.router_transport_eps,
            )
            router_weights = _routing_weights(scores, hard_routing, hard_routing_tau)  # [B, N_c]
            K_chart = torch.argmax(router_weights, dim=1)  # [B]

        # Chart-center mixture defines the macro coordinate (hyperbolic barycenter).
        c_bar = _poincare_weighted_mean(chart_centers, router_weights)  # [B, D]
        v_local = _project_to_ball(mobius_add(-c_bar, v))  # [B, D]

        # Per-chart codebook lookup under optional equivariant metric.
        codebook = _project_to_ball(self.codebook)  # [N_c, K, D]
        v_exp = v_local.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
        codebook_exp = codebook.unsqueeze(0)  # [1, N_c, K, D]
        diff = mobius_add(-codebook_exp, v_exp)  # [B, N_c, K, D]
        diff_tan = log_map_zero(diff)
        dist = self._apply_soft_equiv_metric(diff_tan)  # [B, N_c, K]
        indices = torch.argmin(dist, dim=-1)  # [B, N_c]
        indices_stack = indices  # [B, N_c]

        indices_exp = indices.unsqueeze(-1).unsqueeze(-1)  # [B, N_c, 1, 1]
        indices_exp = indices_exp.expand(-1, -1, 1, self.latent_dim)  # [B, N_c, 1, D]
        z_q_all = torch.gather(codebook.expand(v.shape[0], -1, -1, -1), 2, indices_exp)
        z_q_all = z_q_all.squeeze(2)  # [B, N_c, D]
        if self.soft_equiv_layers is not None and self.soft_equiv_soft_assign:
            temperature = max(self.soft_equiv_temperature, 1e-6)
            weights = F.softmax(-dist / temperature, dim=-1)
            z_q_soft = _poincare_weighted_mean_per_chart(codebook, weights)
            # Straight-through soft assignment so gradients reach the metric network.
            z_q_all += z_q_soft - z_q_soft.detach()

        # VQ objective weighted by routing.
        w = router_weights.unsqueeze(-1).detach()  # [B, N_c, 1]
        v_bc = v_local.unsqueeze(1)  # [B, 1, D]
        delta_commit = log_map_zero(mobius_add(-z_q_all.detach(), v_bc))
        commitment = (delta_commit**2 * w).mean(dim=(0, 2)).sum()  # []
        delta_codebook = log_map_zero(mobius_add(-v_bc.detach(), z_q_all))
        codebook_loss = (delta_codebook**2 * w).mean(dim=(0, 2)).sum()  # []
        vq_loss = codebook_loss + 0.25 * commitment  # []

        # Blend chart codes to form macro latent.
        z_q_blended = _poincare_weighted_mean(z_q_all, router_weights)  # [B, D]
        K_code = indices_stack.gather(1, K_chart.unsqueeze(1)).squeeze(1)  # [B]

        # Structure filter extracts nuisance; remainder is texture.
        delta = log_map_zero(mobius_add(-z_q_all.detach(), v_bc))  # [B, N_c, D]
        z_n_all = self.structure_filter(delta.reshape(-1, self.latent_dim))  # [B*N_c, D]
        z_n_all_charts_tan = z_n_all.view(v.shape[0], self.num_charts, self.latent_dim)
        z_n_all_charts = _project_to_ball(exp_map_zero(z_n_all_charts_tan))  # [B, N_c, D]

        z_n_tan = (z_n_all_charts_tan * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        delta_blended = log_map_zero(mobius_add(-z_q_blended.detach(), v_local))  # [B, D]
        z_tex = delta_blended - z_n_tan  # [B, D]

        # Geometric latent = chart center + macro code + nuisance (Möbius sums).
        delta_to_code = log_map_zero(mobius_add(-v_local, z_q_blended))
        z_q_st = mobius_add(v_local, exp_map_zero(delta_to_code.detach()))
        z_local = mobius_add(z_q_st, exp_map_zero(z_n_tan))
        z_geo = _project_to_ball(mobius_add(c_bar, z_local))  # [B, D]

        return (
            K_chart,
            K_code,
            z_n_tan,
            z_tex,
            router_weights,
            z_geo,
            vq_loss,
            indices_stack,
            z_n_all_charts,
            c_bar,
            v_local,
        )


class PrimitiveTopologicalDecoder(nn.Module):
    """Topological decoder using gauge-covariant primitives."""

    def __init__(
        self,
        latent_dim: int = 2,
        hidden_dim: int = 32,
        num_charts: int = 3,
        output_dim: int = 2,
        bundle_size: int | None = None,
        covariant_attn: bool = True,
        covariant_attn_tensorization: str = "full",
        covariant_attn_rank: int = 8,
        covariant_attn_tau_min: float = 1e-2,
        covariant_attn_denom_min: float = 1e-3,
        covariant_attn_use_transport: bool = True,
        covariant_attn_transport_eps: float = 1e-3,
        conv_backbone: bool = False,
        img_channels: int = 1,
        img_size: int = 28,
        conv_channels: int = 0,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.covariant_attn = covariant_attn
        self.output_dim = output_dim
        self.router_tau_min = covariant_attn_tau_min
        self.router_tau_denom_min = covariant_attn_denom_min
        self.router_transport_eps = covariant_attn_transport_eps

        bundle_size, n_bundles = _resolve_bundle_params(hidden_dim, latent_dim, bundle_size)

        self.chart_projectors = nn.ModuleList([
            SpectralLinear(latent_dim, hidden_dim, bias=False) for _ in range(num_charts)
        ])
        self.chart_gate = NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles)

        # Unit-sphere init: each chart gets a distinct catchment region from
        # the first forward pass, preventing softmax winner-take-all collapse.
        self.chart_centers = nn.Parameter(
            torch.nn.functional.normalize(torch.randn(num_charts, latent_dim), dim=-1)
        )

        if covariant_attn:
            self.cov_router = CovariantChartRouter(
                latent_dim=latent_dim,
                key_dim=hidden_dim,
                num_charts=num_charts,
                feature_dim=None,
                tensorization=covariant_attn_tensorization,
                rank=covariant_attn_rank,
                tau_min=covariant_attn_tau_min,
                tau_denom_min=covariant_attn_denom_min,
                use_transport=covariant_attn_use_transport,
                transport_eps=covariant_attn_transport_eps,
            )
            self.latent_router = None
        else:
            self.latent_router = SpectralLinear(latent_dim, num_charts, bias=True)
        self.tex_residual_scale = nn.Parameter(torch.tensor(0.1))
        if conv_backbone:
            from .vision import ConvImageDecoder

            self.renderer = ConvImageDecoder(
                hidden_dim, img_channels, img_size, conv_channels,
            )
            self.render_skip = None
            # Texture residual maps to hidden_dim (added before conv decoder)
            self.tex_residual = SpectralLinear(latent_dim, hidden_dim, bias=True)
        else:
            self.renderer = nn.Sequential(
                SpectralLinear(hidden_dim, hidden_dim, bias=True),
                NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
                SpectralLinear(hidden_dim, hidden_dim, bias=True),
                NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
                SpectralLinear(hidden_dim, output_dim, bias=True),
            )
            self.render_skip = SpectralLinear(hidden_dim, output_dim, bias=True)
            self.tex_residual = SpectralLinear(latent_dim, output_dim, bias=True)

    def forward(
        self,
        z_geo: torch.Tensor,
        z_tex: torch.Tensor | None = None,
        chart_index: torch.Tensor | None = None,
        router_weights: torch.Tensor | None = None,
        hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode from latent geometry."""
        # Clamp geometry to chart range (Poincare ball).
        z_geo = _project_to_ball(z_geo)
        chart_centers = _project_to_ball(self.chart_centers)
        if router_weights is not None:
            if router_weights.ndim != 2 or router_weights.shape[1] != self.num_charts:
                msg = "router_weights must have shape [B, N_c]."
                raise ValueError(msg)
        elif chart_index is not None:
            router_weights = F.one_hot(
                chart_index, num_classes=self.num_charts
            ).float()  # [B, N_c]
        elif self.covariant_attn:
            # Covariant router predicts chart membership from geometry.
            router_weights, _ = self.cov_router(
                z_geo,
                chart_tokens=chart_centers,
                hard_routing=hard_routing,
                hard_routing_tau=hard_routing_tau,
            )
        else:
            scores = _poincare_hyperbolic_score(
                z_geo,
                chart_centers,
                key_dim=self.latent_dim,
                tau_min=self.router_tau_min,
                tau_denom_min=self.router_tau_denom_min,
                eps=self.router_transport_eps,
            )
            if self.latent_router is not None:
                tau = _poincare_temperature(
                    z_geo,
                    key_dim=self.latent_dim,
                    tau_min=self.router_tau_min,
                    tau_denom_min=self.router_tau_denom_min,
                )
                scores = scores + 0.1 * self.latent_router(z_geo) / tau.unsqueeze(1)
            router_weights = _routing_weights(scores, hard_routing, hard_routing_tau)  # [B, N_c]

        # Chart-specific projections + gauge-covariant gating.
        h_stack = torch.stack(
            [proj(z_geo) for proj in self.chart_projectors], dim=1
        )  # [B, N_c, H]
        h_stack = self.chart_gate(h_stack.view(-1, self.hidden_dim)).view(
            z_geo.shape[0], self.num_charts, self.hidden_dim
        )
        h_global = (h_stack * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, H]

        if self.render_skip is not None:
            # FC mode: two parallel paths + texture in output space
            x_hat = self.renderer(h_global) + self.render_skip(h_global)  # [B, D_out]
            if z_tex is not None:
                z_tex = torch.tanh(z_tex)
                x_hat = x_hat + self.tex_residual_scale * self.tex_residual(z_tex)
        else:
            # Conv mode: texture added in hidden space before conv decoder
            h = h_global
            if z_tex is not None:
                z_tex = torch.tanh(z_tex)
                h = h + self.tex_residual_scale * self.tex_residual(z_tex)
            x_hat = self.renderer(h)
        return x_hat, router_weights


class TopoEncoderPrimitives(nn.Module):
    """Attentive Atlas encoder + topological decoder built from primitives."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_charts: int = 3,
        codes_per_chart: int = 21,
        bundle_size: int | None = None,
        covariant_attn: bool = True,
        covariant_attn_tensorization: str = "full",
        covariant_attn_rank: int = 8,
        covariant_attn_tau_min: float = 1e-2,
        covariant_attn_denom_min: float = 1e-3,
        covariant_attn_use_transport: bool = True,
        covariant_attn_transport_eps: float = 1e-3,
        soft_equiv_metric: bool = False,
        soft_equiv_bundle_size: int | None = None,
        soft_equiv_hidden_dim: int = 64,
        soft_equiv_use_spectral_norm: bool = True,
        soft_equiv_zero_self_mixing: bool = False,
        soft_equiv_soft_assign: bool = True,
        soft_equiv_temperature: float = 1.0,
        conv_backbone: bool = False,
        img_channels: int = 1,
        img_size: int = 28,
        conv_channels: int = 0,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts

        self.encoder = PrimitiveAttentiveAtlasEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_charts=num_charts,
            codes_per_chart=codes_per_chart,
            bundle_size=bundle_size,
            covariant_attn=covariant_attn,
            covariant_attn_tensorization=covariant_attn_tensorization,
            covariant_attn_rank=covariant_attn_rank,
            covariant_attn_tau_min=covariant_attn_tau_min,
            covariant_attn_denom_min=covariant_attn_denom_min,
            covariant_attn_use_transport=covariant_attn_use_transport,
            covariant_attn_transport_eps=covariant_attn_transport_eps,
            soft_equiv_metric=soft_equiv_metric,
            soft_equiv_bundle_size=soft_equiv_bundle_size,
            soft_equiv_hidden_dim=soft_equiv_hidden_dim,
            soft_equiv_use_spectral_norm=soft_equiv_use_spectral_norm,
            soft_equiv_zero_self_mixing=soft_equiv_zero_self_mixing,
            soft_equiv_soft_assign=soft_equiv_soft_assign,
            soft_equiv_temperature=soft_equiv_temperature,
            conv_backbone=conv_backbone,
            img_channels=img_channels,
            img_size=img_size,
            conv_channels=conv_channels,
        )
        self.decoder = PrimitiveTopologicalDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_charts=num_charts,
            output_dim=input_dim,
            bundle_size=bundle_size,
            covariant_attn=covariant_attn,
            covariant_attn_tensorization=covariant_attn_tensorization,
            covariant_attn_rank=covariant_attn_rank,
            covariant_attn_tau_min=covariant_attn_tau_min,
            covariant_attn_denom_min=covariant_attn_denom_min,
            covariant_attn_use_transport=covariant_attn_use_transport,
            covariant_attn_transport_eps=covariant_attn_transport_eps,
            conv_backbone=conv_backbone,
            img_channels=img_channels,
            img_size=img_size,
            conv_channels=conv_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        use_hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            K_chart,
            _K_code,
            z_n,
            z_tex,
            enc_router_weights,
            z_geo,
            vq_loss,
            _indices,
            _z_n_all,
            c_bar,
            _v_local,
        ) = self.encoder(
            x,
            hard_routing=use_hard_routing,
            hard_routing_tau=hard_routing_tau,
        )

        router_override = enc_router_weights if use_hard_routing else None
        x_recon, dec_router_weights = self.decoder(
            z_geo,
            z_tex,
            chart_index=None,
            router_weights=router_override,
            hard_routing=use_hard_routing,
            hard_routing_tau=hard_routing_tau,
        )

        return x_recon, vq_loss, enc_router_weights, dec_router_weights, K_chart, z_geo, z_n, c_bar

    def compute_consistency_loss(
        self, enc_weights: torch.Tensor, dec_weights: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        kl = (enc_weights * torch.log((enc_weights + eps) / (dec_weights + eps))).sum(dim=-1)
        return kl.mean()

    def compute_perplexity(self, K_chart: torch.Tensor) -> float:
        counts = torch.bincount(K_chart, minlength=self.num_charts).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * torch.log(probs)).sum()
        return math.exp(entropy.item())


def _expand_list(value: int | list[int], n_levels: int, name: str) -> list[int]:
    if isinstance(value, list):
        if len(value) != n_levels:
            raise ValueError(f"{name} must have length {n_levels}.")
        return [int(v) for v in value]
    return [int(value) for _ in range(n_levels)]


def _select_chart_latent(z_by_chart: torch.Tensor, chart_idx: torch.Tensor) -> torch.Tensor:
    if z_by_chart.ndim != 3:
        msg = "z_by_chart must have shape [B, N_c, D]."
        raise ValueError(msg)
    idx = chart_idx.view(-1, 1, 1).expand(-1, 1, z_by_chart.shape[-1])
    return z_by_chart.gather(1, idx).squeeze(1)


def _routing_weights(
    scores: torch.Tensor, hard_routing: bool, hard_routing_tau: float
) -> torch.Tensor:
    if not hard_routing:
        return F.softmax(scores, dim=-1)
    tau = max(float(hard_routing_tau), 1e-6)
    return F.gumbel_softmax(scores, tau=tau, hard=True)


class _SharedFeatureExtractor(nn.Module):
    """Shared feature extractor for hierarchical atlas stacks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        bundle_size: int | None,
    ) -> None:
        super().__init__()
        bundle_size, n_bundles = _resolve_bundle_params(hidden_dim, latent_dim, bundle_size)

        self.feature_extractor = nn.Sequential(
            SpectralLinear(input_dim, hidden_dim, bias=True),
            NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
            SpectralLinear(hidden_dim, hidden_dim, bias=True),
            NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


class _AtlasEncoderLevel(nn.Module):
    """Encoder head that consumes shared features and produces charted latents."""

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        num_charts: int,
        codes_per_chart: int,
        covariant_attn: bool,
        covariant_attn_tensorization: str,
        covariant_attn_rank: int,
        covariant_attn_tau_min: float,
        covariant_attn_denom_min: float,
        covariant_attn_use_transport: bool,
        covariant_attn_transport_eps: float,
        soft_equiv_metric: bool,
        soft_equiv_bundle_size: int | None,
        soft_equiv_hidden_dim: int,
        soft_equiv_use_spectral_norm: bool,
        soft_equiv_zero_self_mixing: bool,
        soft_equiv_soft_assign: bool,
        soft_equiv_temperature: float,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.latent_dim = latent_dim
        self.codes_per_chart = codes_per_chart
        self.covariant_attn = covariant_attn

        if covariant_attn:
            self.cov_router = CovariantChartRouter(
                latent_dim=latent_dim,
                key_dim=hidden_dim,
                num_charts=num_charts,
                feature_dim=hidden_dim,
                tensorization=covariant_attn_tensorization,
                rank=covariant_attn_rank,
                tau_min=covariant_attn_tau_min,
                tau_denom_min=covariant_attn_denom_min,
                use_transport=covariant_attn_use_transport,
                transport_eps=covariant_attn_transport_eps,
            )
            self.key_proj = None
            self.chart_queries = None
            self.scale = None
        else:
            self.key_proj = SpectralLinear(hidden_dim, hidden_dim, bias=True)
            self.chart_queries = nn.Parameter(torch.randn(num_charts, hidden_dim) * 0.02)
            self.scale = math.sqrt(hidden_dim)

        self.val_proj = SpectralLinear(hidden_dim, latent_dim, bias=True)
        # Unit-sphere init: each chart gets a distinct catchment region from
        # the first forward pass, preventing softmax winner-take-all collapse.
        self.chart_centers = nn.Parameter(
            torch.nn.functional.normalize(torch.randn(num_charts, latent_dim), dim=-1)
        )
        self.codebook = nn.Parameter(torch.randn(num_charts, codes_per_chart, latent_dim) * 0.02)

        self.soft_equiv_layers: nn.ModuleList | None = None
        if soft_equiv_metric:
            bundle_size = soft_equiv_bundle_size or latent_dim
            if bundle_size <= 0:
                msg = "soft_equiv_bundle_size must be positive."
                raise ValueError(msg)
            if latent_dim % bundle_size != 0:
                msg = "latent_dim must be divisible by soft_equiv_bundle_size."
                raise ValueError(msg)
            n_bundles = latent_dim // bundle_size
            self.soft_equiv_layers = nn.ModuleList([
                SoftEquivariantLayer(
                    n_bundles=n_bundles,
                    bundle_dim=bundle_size,
                    hidden_dim=soft_equiv_hidden_dim,
                    use_spectral_norm=soft_equiv_use_spectral_norm,
                    zero_self_mixing=soft_equiv_zero_self_mixing,
                )
                for _ in range(num_charts)
            ])
            _init_soft_equiv_layers(self.soft_equiv_layers)

        self.soft_equiv_soft_assign = soft_equiv_soft_assign
        self.soft_equiv_temperature = soft_equiv_temperature
        if soft_equiv_metric and self.soft_equiv_soft_assign and self.soft_equiv_temperature <= 0:
            msg = "soft_equiv_temperature must be positive when soft_equiv_soft_assign is enabled."
            raise ValueError(msg)

        self.structure_filter = nn.Sequential(
            IsotropicBlock(latent_dim, latent_dim, bundle_size=latent_dim),
            SpectralLinear(latent_dim, latent_dim, bias=True),
        )
        self._last_soft_equiv_log_ratio: torch.Tensor | None = None

    def _apply_soft_equiv_metric(self, diff: torch.Tensor) -> torch.Tensor:
        if self.soft_equiv_layers is None:
            self._last_soft_equiv_log_ratio = None
            return (diff**2).sum(dim=-1)
        batch_size, _num_charts, num_codes, latent_dim = diff.shape
        ratio_max = 50.0
        eps = 1e-6
        transformed = []
        log_ratio_losses = []
        for chart_idx, layer in enumerate(self.soft_equiv_layers):
            diff_chart = diff[:, chart_idx].reshape(-1, latent_dim)
            diff_chart = torch.nan_to_num(diff_chart, nan=0.0, posinf=0.0, neginf=0.0)
            diff_out = layer(diff_chart)
            diff_out = torch.nan_to_num(diff_out, nan=0.0, posinf=0.0, neginf=0.0)
            in_norm = diff_chart.norm(dim=-1, keepdim=True).clamp(min=eps)
            out_norm = diff_out.norm(dim=-1, keepdim=True)
            ratio = out_norm / in_norm
            ratio_clamped = ratio.clamp(max=ratio_max)
            scale = torch.where(ratio > 0, ratio_clamped / ratio, torch.ones_like(ratio))
            diff_out *= scale
            log_ratio = torch.log(ratio.clamp(min=eps, max=ratio_max))
            log_ratio_losses.append((log_ratio**2).mean())
            transformed.append(diff_out.view(batch_size, num_codes, latent_dim))
        diff_out = torch.stack(transformed, dim=1)
        if log_ratio_losses:
            self._last_soft_equiv_log_ratio = torch.stack(log_ratio_losses).mean()
        else:
            self._last_soft_equiv_log_ratio = torch.tensor(0.0, device=diff.device)
        return (diff_out**2).sum(dim=-1)

    def soft_equiv_l1_loss(self) -> torch.Tensor:
        if self.soft_equiv_layers is None:
            return torch.tensor(0.0, device=self.codebook.device)
        total = torch.zeros((), device=self.codebook.device)
        for layer in self.soft_equiv_layers:
            total += layer.l1_loss()
        return total / len(self.soft_equiv_layers)

    def soft_equiv_log_ratio_loss(self) -> torch.Tensor:
        if self.soft_equiv_layers is None or self._last_soft_equiv_log_ratio is None:
            return torch.tensor(0.0, device=self.codebook.device)
        return self._last_soft_equiv_log_ratio

    def forward(
        self,
        features: torch.Tensor,
        hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Map shared features into chart coordinates.
        v = self.val_proj(features)  # [B, D]

        if self.covariant_attn:
            # Covariant router assigns charts with gauge-aware transport.
            router_weights, K_chart = self.cov_router(
                v,
                features=features,
                chart_tokens=self.chart_centers,
                hard_routing=hard_routing,
                hard_routing_tau=hard_routing_tau,
            )
        else:
            scores = torch.matmul(v, self.chart_centers.t()) / math.sqrt(self.latent_dim)
            router_weights = _routing_weights(scores, hard_routing, hard_routing_tau)
            K_chart = torch.argmax(router_weights, dim=1)

        # Chart-center mixture defines the macro coordinate.
        c_bar = torch.matmul(router_weights, self.chart_centers)
        v_local = v - c_bar

        # Per-chart codebook match under optional equivariant metric.
        codebook = self.codebook.unsqueeze(0)
        v_exp = v_local.unsqueeze(1).unsqueeze(2)
        diff = v_exp - codebook
        dist = self._apply_soft_equiv_metric(diff)
        indices = torch.argmin(dist, dim=-1)
        indices_stack = indices

        indices_exp = indices.unsqueeze(-1).unsqueeze(-1)
        indices_exp = indices_exp.expand(-1, -1, 1, self.latent_dim)
        z_q_all = torch.gather(codebook.expand(v.shape[0], -1, -1, -1), 2, indices_exp)
        z_q_all = z_q_all.squeeze(2)
        if self.soft_equiv_layers is not None and self.soft_equiv_soft_assign:
            temperature = max(self.soft_equiv_temperature, 1e-6)
            weights = F.softmax(-dist / temperature, dim=-1)
            z_q_soft = (weights.unsqueeze(-1) * codebook).sum(dim=2)
            z_q_all += z_q_soft - z_q_soft.detach()

        # VQ objective weighted by routing.
        w = router_weights.unsqueeze(-1).detach()
        v_bc = v_local.unsqueeze(1)
        commitment = ((v_bc - z_q_all.detach()) ** 2 * w).mean(dim=(0, 2)).sum()
        codebook_loss = ((z_q_all - v_bc.detach()) ** 2 * w).mean(dim=(0, 2)).sum()
        vq_loss = codebook_loss + 0.25 * commitment

        # Blend chart codes to form macro latent.
        z_q_blended = (z_q_all * router_weights.unsqueeze(-1)).sum(dim=1)
        K_code = indices_stack.gather(1, K_chart.unsqueeze(1)).squeeze(1)

        # Structure filter extracts nuisance; remainder is texture.
        delta = v_bc - z_q_all.detach()
        z_n_all = self.structure_filter(delta.reshape(-1, self.latent_dim))
        z_n_all_charts = z_n_all.view(v.shape[0], self.num_charts, self.latent_dim)

        z_n = (z_n_all_charts * router_weights.unsqueeze(-1)).sum(dim=1)
        delta_blended = v_local - z_q_blended.detach()
        z_tex = delta_blended - z_n

        # Geometric latent = chart center + macro code + nuisance.
        z_q_st = v_local + (z_q_blended - v_local).detach()
        z_geo = c_bar + z_q_st + z_n

        return (
            K_chart,
            K_code,
            z_n,
            z_tex,
            router_weights,
            z_geo,
            vq_loss,
            indices_stack,
            z_n_all_charts,
            c_bar,
            v_local,
        )


class HierarchicalAtlasStack(nn.Module):
    """Multi-scale TopoEncoder stack with optional shared feature extractor."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int | list[int] = 2,
        num_charts: int | list[int] = 3,
        codes_per_chart: int | list[int] = 21,
        n_levels: int = 3,
        level_update_freqs: list[int] | None = None,
        bundle_size: int | None = None,
        covariant_attn: bool = True,
        covariant_attn_tensorization: str = "full",
        covariant_attn_rank: int = 8,
        covariant_attn_tau_min: float = 1e-2,
        covariant_attn_denom_min: float = 1e-3,
        covariant_attn_use_transport: bool = True,
        covariant_attn_transport_eps: float = 1e-3,
        soft_equiv_metric: bool = False,
        soft_equiv_bundle_size: int | None = None,
        soft_equiv_hidden_dim: int = 64,
        soft_equiv_use_spectral_norm: bool = True,
        soft_equiv_zero_self_mixing: bool = False,
        soft_equiv_soft_assign: bool = True,
        soft_equiv_temperature: float = 1.0,
        share_feature_extractor: bool = True,
        enable_cross_level_jump: bool = False,
        jump_global_rank: int | None = None,
    ) -> None:
        super().__init__()
        if n_levels <= 0:
            msg = "n_levels must be positive."
            raise ValueError(msg)

        self.n_levels = n_levels
        self.share_feature_extractor = share_feature_extractor

        level_latent_dims = _expand_list(latent_dim, n_levels, "latent_dim")
        level_num_charts = _expand_list(num_charts, n_levels, "num_charts")
        level_codes = _expand_list(codes_per_chart, n_levels, "codes_per_chart")
        if level_update_freqs is None:
            level_update_freqs = [1 for _ in range(n_levels)]
        if len(level_update_freqs) != n_levels:
            msg = "level_update_freqs must match n_levels."
            raise ValueError(msg)
        self.level_update_freqs = [int(v) for v in level_update_freqs]

        if share_feature_extractor:
            self.feature_extractor = _SharedFeatureExtractor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=min(level_latent_dims),
                bundle_size=bundle_size,
            )
            self.feature_extractors = None
        else:
            self.feature_extractor = None
            self.feature_extractors = nn.ModuleList([
                _SharedFeatureExtractor(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    latent_dim=level_latent_dims[idx],
                    bundle_size=bundle_size,
                )
                for idx in range(n_levels)
            ])

        self.encoder_levels = nn.ModuleList()
        self.decoder_levels = nn.ModuleList()
        for idx in range(n_levels):
            self.encoder_levels.append(
                _AtlasEncoderLevel(
                    hidden_dim=hidden_dim,
                    latent_dim=level_latent_dims[idx],
                    num_charts=level_num_charts[idx],
                    codes_per_chart=level_codes[idx],
                    covariant_attn=covariant_attn,
                    covariant_attn_tensorization=covariant_attn_tensorization,
                    covariant_attn_rank=covariant_attn_rank,
                    covariant_attn_tau_min=covariant_attn_tau_min,
                    covariant_attn_denom_min=covariant_attn_denom_min,
                    covariant_attn_use_transport=covariant_attn_use_transport,
                    covariant_attn_transport_eps=covariant_attn_transport_eps,
                    soft_equiv_metric=soft_equiv_metric,
                    soft_equiv_bundle_size=soft_equiv_bundle_size,
                    soft_equiv_hidden_dim=soft_equiv_hidden_dim,
                    soft_equiv_use_spectral_norm=soft_equiv_use_spectral_norm,
                    soft_equiv_zero_self_mixing=soft_equiv_zero_self_mixing,
                    soft_equiv_soft_assign=soft_equiv_soft_assign,
                    soft_equiv_temperature=soft_equiv_temperature,
                )
            )
            self.decoder_levels.append(
                PrimitiveTopologicalDecoder(
                    latent_dim=level_latent_dims[idx],
                    hidden_dim=hidden_dim,
                    num_charts=level_num_charts[idx],
                    output_dim=input_dim,
                    bundle_size=bundle_size,
                    covariant_attn=covariant_attn,
                    covariant_attn_tensorization=covariant_attn_tensorization,
                    covariant_attn_rank=covariant_attn_rank,
                    covariant_attn_tau_min=covariant_attn_tau_min,
                    covariant_attn_denom_min=covariant_attn_denom_min,
                    covariant_attn_use_transport=covariant_attn_use_transport,
                    covariant_attn_transport_eps=covariant_attn_transport_eps,
                )
            )

        self.jump_operators = None
        if enable_cross_level_jump:
            if n_levels < 2:
                msg = "enable_cross_level_jump requires at least 2 levels."
                raise ValueError(msg)
            if len(set(level_latent_dims)) != 1 or len(set(level_num_charts)) != 1:
                msg = (
                    "enable_cross_level_jump requires identical latent_dim and num_charts "
                    "across levels."
                )
                raise ValueError(msg)
            if jump_global_rank is not None and jump_global_rank < 0:
                msg = "jump_global_rank must be non-negative when provided."
                raise ValueError(msg)
            rank = None if jump_global_rank in {None, 0} else int(jump_global_rank)
            self.jump_operators = nn.ModuleList([
                FactorizedJumpOperator(
                    num_charts=level_num_charts[0],
                    latent_dim=level_latent_dims[0],
                    global_rank=rank,
                )
                for _ in range(n_levels - 1)
            ])

    def forward(
        self,
        x: torch.Tensor,
        step: int | torch.Tensor | None = None,
        prev_state: list[dict[str, torch.Tensor]] | None = None,
        use_hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> list[dict[str, torch.Tensor]]:
        if prev_state is not None and len(prev_state) != self.n_levels:
            msg = "prev_state must have one entry per level."
            raise ValueError(msg)

        if step is not None:
            if isinstance(step, torch.Tensor):
                step_value = int(step.item())
            else:
                step_value = int(step)
        else:
            step_value = None

        # Shared feature extractor yields a common view for all levels.
        if self.share_feature_extractor:
            features = self.feature_extractor(x)
        else:
            features = None

        outputs: list[dict[str, torch.Tensor]] = []
        for idx in range(self.n_levels):
            if (
                step_value is not None
                and prev_state is not None
                and self.level_update_freqs[idx] > 1
                and step_value % self.level_update_freqs[idx] != 0
            ):
                outputs.append(prev_state[idx])
                continue

            # Optionally reuse cached features; otherwise compute per-level.
            if self.share_feature_extractor:
                level_features = features
            else:
                level_features = self.feature_extractors[idx](x)

            (
                K_chart,
                K_code,
                z_n,
                z_tex,
                enc_router_weights,
                z_geo,
                vq_loss,
                indices_stack,
                z_n_all_charts,
                c_bar,
                _v_local,
            ) = self.encoder_levels[idx](
                level_features,
                hard_routing=use_hard_routing,
                hard_routing_tau=hard_routing_tau,
            )

            router_override = enc_router_weights if use_hard_routing else None
            x_recon, dec_router_weights = self.decoder_levels[idx](
                z_geo,
                z_tex,
                chart_index=None,
                router_weights=router_override,
                hard_routing=use_hard_routing,
                hard_routing_tau=hard_routing_tau,
            )
            z_n_local = _select_chart_latent(z_n_all_charts, K_chart)

            outputs.append({
                "x_recon": x_recon,
                "vq_loss": vq_loss,
                "enc_router_weights": enc_router_weights,
                "dec_router_weights": dec_router_weights,
                "K_chart": K_chart,
                "K_code": K_code,
                "z_geo": z_geo,
                "z_n": z_n,
                "z_n_local": z_n_local,
                "z_tex": z_tex,
                "indices_stack": indices_stack,
                "z_n_all_charts": z_n_all_charts,
                "c_bar": c_bar,
            })

        # Optional cross-level jump operators align nuisance coordinates across scales.
        if self.jump_operators is not None:
            for idx, jump_op in enumerate(self.jump_operators):
                src = outputs[idx]
                tgt = outputs[idx + 1]
                if "z_n_local" not in src:
                    src["z_n_local"] = _select_chart_latent(src["z_n_all_charts"], src["K_chart"])
                if "z_n_local" not in tgt:
                    tgt["z_n_local"] = _select_chart_latent(tgt["z_n_all_charts"], tgt["K_chart"])
                z_jump = jump_op(src["z_n_local"], src["K_chart"], tgt["K_chart"])
                src["z_n_jump_to_next"] = z_jump
                tgt["z_n_jump_from_prev"] = z_jump

        return outputs


class TopoEncoderAttachments(nn.Module):
    """Optional modules commonly attached to TopoEncoder training."""

    def __init__(
        self,
        num_charts: int,
        latent_dim: int,
        num_classes: int | None = None,
        enable_jump: bool = True,
        enable_classifier: bool = False,
        jump_global_rank: int | None = None,
        classifier_bundle_size: int | None = None,
    ) -> None:
        super().__init__()
        if num_charts <= 0:
            msg = "num_charts must be positive."
            raise ValueError(msg)
        if latent_dim <= 0:
            msg = "latent_dim must be positive."
            raise ValueError(msg)

        self.jump_operator = None
        if enable_jump:
            self.jump_operator = FactorizedJumpOperator(
                num_charts=num_charts,
                latent_dim=latent_dim,
                global_rank=jump_global_rank,
            )

        self.classifier_head = None
        if enable_classifier:
            if num_classes is None or num_classes <= 0:
                msg = "num_classes must be positive when classifier is enabled."
                raise ValueError(msg)
            self.classifier_head = InvariantChartClassifier(
                num_charts=num_charts,
                num_classes=num_classes,
                latent_dim=latent_dim,
                bundle_size=classifier_bundle_size,
            )

    def forward(
        self,
        router_weights: torch.Tensor | None = None,
        z_geo: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        if self.classifier_head is not None:
            if router_weights is None or z_geo is None:
                msg = "router_weights and z_geo are required for classifier_head."
                raise ValueError(msg)
            outputs["classifier_logits"] = self.classifier_head(router_weights, z_geo)
        return outputs
