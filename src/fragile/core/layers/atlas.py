from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fragile.core.layers.primitives import IsotropicBlock, NormGatedGELU, SpectralLinear


class StandardVQ(nn.Module):
    """Standard Vector-Quantized VAE baseline."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_codes: int = 64,
    ) -> None:
        super().__init__()
        self.num_codes = num_codes

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.embeddings = nn.Embedding(num_codes, latent_dim)
        nn.init.uniform_(self.embeddings.weight, -1.0 / num_codes, 1.0 / num_codes)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode, quantize, and decode.

        Args:
            x: [B, D_in] input tensor

        Returns:
            x_recon: [B, D_in] reconstruction
            vq_loss: [] VQ loss
            indices: [B] code indices
        """
        z_e = self.encoder(x)  # [B, D_latent]
        embed = self.embeddings.weight  # [K, D_latent]

        z_sq = (z_e**2).sum(dim=1, keepdim=True)  # [B, 1]
        e_sq = (embed**2).sum(dim=1).unsqueeze(0)  # [1, K]
        dot = torch.matmul(z_e, embed.t())  # [B, K]
        dist = z_sq + e_sq - 2.0 * dot  # [B, K]

        indices = torch.argmin(dist, dim=1)  # [B]
        z_q = embed[indices]  # [B, D_latent]

        commitment = F.mse_loss(z_e, z_q.detach())  # []
        codebook = F.mse_loss(z_q, z_e.detach())  # []
        vq_loss = codebook + 0.25 * commitment  # []

        z_st = z_e + (z_q - z_e).detach()  # [B, D_latent]
        x_recon = self.decoder(z_st)  # [B, D_in]
        return x_recon, vq_loss, indices

    def compute_perplexity(self, indices: torch.Tensor) -> float:
        """Compute codebook perplexity.

        Args:
            indices: [B] code indices

        Returns:
            perplexity: scalar perplexity
        """
        counts = torch.bincount(indices, minlength=self.num_codes).float()  # [K]
        probs = counts / counts.sum()  # [K]
        probs = probs[probs > 0]  # [K_nonzero]
        entropy = -(probs * torch.log(probs)).sum()  # []
        return math.exp(entropy.item())


class VanillaAE(nn.Module):
    """Continuous autoencoder baseline."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, latent_dim: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and decode.

        Args:
            x: [B, D_in] input

        Returns:
            x_recon: [B, D_in] reconstruction
            z: [B, D_latent] latent
        """
        z = self.encoder(x)  # [B, D_latent]
        x_recon = self.decoder(z)  # [B, D_in]
        return x_recon, z


class AttentiveAtlasEncoder(nn.Module):
    """Attentive Atlas encoder with cross-attention routing."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_charts: int = 3,
        codes_per_chart: int = 21,
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

        self.chart_centers = nn.Parameter(torch.randn(num_charts, latent_dim) * 0.02)

        self.codebook = nn.Parameter(torch.randn(num_charts, codes_per_chart, latent_dim) * 0.02)

        self.structure_filter = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2 if latent_dim > 2 else latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim // 2 if latent_dim > 2 else latent_dim, latent_dim),
        )

    def forward(
        self, x: torch.Tensor
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
        """
        features = self.feature_extractor(x)  # [B, H]

        v = self.val_proj(features)  # [B, D]
        scores = torch.matmul(v, self.chart_centers.t()) / math.sqrt(self.latent_dim)  # [B, N_c]
        router_weights = F.softmax(scores, dim=-1)  # [B, N_c]
        K_chart = torch.argmax(router_weights, dim=1)  # [B]

        c_bar = torch.matmul(router_weights, self.chart_centers)  # [B, D]
        v_local = v - c_bar  # [B, D]

        codebook = self.codebook.unsqueeze(0)  # [1, N_c, K, D]
        v_exp = v_local.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
        diff = v_exp - codebook  # [B, N_c, K, D]
        dist = (diff**2).sum(dim=-1)  # [B, N_c, K]
        indices = torch.argmin(dist, dim=-1)  # [B, N_c]
        indices_stack = indices  # [B, N_c]

        indices_exp = indices.unsqueeze(-1).unsqueeze(-1)  # [B, N_c, 1, 1]
        indices_exp = indices_exp.expand(-1, -1, 1, self.latent_dim)  # [B, N_c, 1, D]
        z_q_all = torch.gather(codebook.expand(v.shape[0], -1, -1, -1), 2, indices_exp)
        z_q_all = z_q_all.squeeze(2)  # [B, N_c, D]

        w = router_weights.unsqueeze(-1).detach()  # [B, N_c, 1]
        v_bc = v_local.unsqueeze(1)  # [B, 1, D]
        commitment = ((v_bc - z_q_all.detach()) ** 2 * w).mean(dim=(0, 2)).sum()  # []
        codebook_loss = ((z_q_all - v_bc.detach()) ** 2 * w).mean(dim=(0, 2)).sum()  # []
        vq_loss = codebook_loss + 0.25 * commitment  # []

        z_q_blended = (z_q_all * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        K_code = indices_stack.gather(1, K_chart.unsqueeze(1)).squeeze(1)  # [B]

        delta = v_bc - z_q_all.detach()  # [B, N_c, D]
        z_n_all = self.structure_filter(delta.reshape(-1, self.latent_dim))  # [B*N_c, D]
        z_n_all_charts = z_n_all.view(v.shape[0], self.num_charts, self.latent_dim)  # [B, N_c, D]

        z_n = (z_n_all_charts * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        delta_blended = v_local - z_q_blended.detach()  # [B, D]
        z_tex = delta_blended - z_n  # [B, D]

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
        z_geo = torch.tanh(z_geo)
        if chart_index is not None:
            router_weights = F.one_hot(chart_index, num_classes=self.num_charts).float()  # [B, N_c]
        else:
            logits = self.latent_router(z_geo)  # [B, N_c]
            router_weights = F.softmax(logits, dim=-1)  # [B, N_c]

        h_stack = torch.einsum("bl,chl->bch", z_geo, self.chart_weight)  # [B, N_c, H]
        h_stack = h_stack + self.chart_bias.unsqueeze(0)  # [B, N_c, H]
        h_global = (h_stack * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, H]

        x_hat = self.renderer(h_global) + self.render_skip(h_global)  # [B, D_out]
        if z_tex is not None:
            z_tex = torch.tanh(z_tex)
            x_hat = x_hat + self.tex_residual_scale * self.tex_residual(z_tex)
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
    ) -> None:
        super().__init__()
        self.num_charts = num_charts

        self.encoder = AttentiveAtlasEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_charts=num_charts,
            codes_per_chart=codes_per_chart,
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
        ) = self.encoder(x)

        chart_index = K_chart if use_hard_routing else None
        x_recon, dec_router_weights = self.decoder(z_geo, z_tex, chart_index)

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
        kl = (enc_weights * torch.log((enc_weights + eps) / (dec_weights + eps))).sum(dim=-1)  # [B]
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
        raise ValueError("bundle_size must be positive.")
    if hidden_dim % bundle_size != 0:
        raise ValueError("hidden_dim must be divisible by bundle_size.")
    return bundle_size, hidden_dim // bundle_size


class CovariantChartRouter(nn.Module):
    """Gauge-covariant chart router with Wilson-line transport and metric-aware temperature."""

    def __init__(
        self,
        latent_dim: int,
        key_dim: int,
        num_charts: int,
        feature_dim: int | None = None,
        tensorization: str = "sum",
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
            self.q_gamma = nn.Parameter(
                torch.randn(key_dim, latent_dim, latent_dim) * 0.02
            )
            self.q_gamma_out = None
            self.q_gamma_u = None
            self.q_gamma_v = None
        elif tensorization == "sum":
            self.q_gamma_out = nn.Parameter(torch.randn(rank, key_dim) * 0.02)
            self.q_gamma_u = nn.Parameter(torch.randn(rank, latent_dim) * 0.02)
            self.q_gamma_v = nn.Parameter(torch.randn(rank, latent_dim) * 0.02)
            self.q_gamma = None
        else:
            raise ValueError("tensorization must be 'full' or 'sum'.")

        self.chart_queries = nn.Parameter(torch.randn(num_charts, key_dim) * 0.02)
        self.chart_key_proj = SpectralLinear(latent_dim, key_dim, bias=False)

        if use_transport:
            self.transport_proj = SpectralLinear(latent_dim, key_dim * key_dim, bias=False)
        else:
            self.transport_proj = None

    def _gamma_term(self, z: torch.Tensor) -> torch.Tensor:
        if self.tensorization == "full":
            z_outer = z.unsqueeze(2) * z.unsqueeze(1)  # [B, D, D]
            return torch.einsum("bij,kij->bk", z_outer, self.q_gamma)
        z_u = z @ self.q_gamma_u.t()  # [B, R]
        z_v = z @ self.q_gamma_v.t()  # [B, R]
        return (z_u * z_v) @ self.q_gamma_out  # [B, K]

    def _transport_queries(
        self, z: torch.Tensor, chart_tokens: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size = z.shape[0]
        if chart_tokens is None:
            base_queries = self.chart_queries
        else:
            if chart_tokens.ndim != 2 or chart_tokens.shape[0] != self.num_charts:
                raise ValueError("chart_tokens must have shape [N_c, D] or [N_c, K].")
            if chart_tokens.shape[1] == self.key_dim:
                base_queries = chart_tokens
            elif chart_tokens.shape[1] == self.latent_dim:
                base_queries = self.chart_key_proj(chart_tokens)
            else:
                raise ValueError("chart_tokens must have shape [N_c, D] or [N_c, K].")

        if self.transport_proj is None:
            return base_queries.unsqueeze(0).expand(batch_size, -1, -1)

        skew = self.transport_proj(z).view(batch_size, self.key_dim, self.key_dim)
        skew = 0.5 * (skew - skew.transpose(1, 2))
        eye = torch.eye(self.key_dim, device=z.device, dtype=z.dtype).expand(
            batch_size, -1, -1
        )
        eye = eye * (1.0 + self.transport_eps)
        u = torch.linalg.solve(eye + 0.5 * skew, eye - 0.5 * skew)
        return torch.einsum("bij,nj->bni", u, base_queries)

    def _temperature(self, z: torch.Tensor) -> torch.Tensor:
        r2 = (z**2).sum(dim=-1)
        denom = (1.0 - r2).clamp(min=self.tau_denom_min)
        tau = math.sqrt(self.key_dim) * denom / 2.0
        return tau.clamp(min=self.tau_min)

    def forward(
        self,
        z: torch.Tensor,
        features: torch.Tensor | None = None,
        chart_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_z_proj(z)
        if self.q_feat_proj is not None:
            if features is None:
                raise ValueError("features are required when q_feat_proj is enabled.")
            q = q + self.q_feat_proj(features)
        q = q + self._gamma_term(z)

        keys = self._transport_queries(z, chart_tokens=chart_tokens)  # [B, N_c, K]
        scores = (keys * q.unsqueeze(1)).sum(dim=-1)
        tau = self._temperature(z)
        scores = scores / tau.unsqueeze(1)
        router_weights = F.softmax(scores, dim=-1)
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
        covariant_attn_tensorization: str = "sum",
        covariant_attn_rank: int = 8,
        covariant_attn_tau_min: float = 1e-2,
        covariant_attn_denom_min: float = 1e-3,
        covariant_attn_use_transport: bool = True,
        covariant_attn_transport_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.latent_dim = latent_dim
        self.codes_per_chart = codes_per_chart
        self.covariant_attn = covariant_attn

        bundle_size, n_bundles = _resolve_bundle_params(hidden_dim, latent_dim, bundle_size)

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

        self.chart_centers = nn.Parameter(torch.randn(num_charts, latent_dim) * 0.02)

        self.codebook = nn.Parameter(torch.randn(num_charts, codes_per_chart, latent_dim) * 0.02)

        self.structure_filter = nn.Sequential(
            IsotropicBlock(latent_dim, latent_dim, bundle_size=latent_dim),
            SpectralLinear(latent_dim, latent_dim, bias=True),
        )

    def forward(
        self, x: torch.Tensor
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
    ]:
        """Forward pass through the attentive atlas."""
        features = self.feature_extractor(x)  # [B, H]
        v = self.val_proj(features)  # [B, D]

        if self.covariant_attn:
            router_weights, K_chart = self.cov_router(
                v, features=features, chart_tokens=self.chart_centers
            )
        else:
            scores = torch.matmul(v, self.chart_centers.t()) / math.sqrt(self.latent_dim)  # [B, N_c]
            router_weights = F.softmax(scores, dim=-1)  # [B, N_c]
            K_chart = torch.argmax(router_weights, dim=1)  # [B]

        c_bar = torch.matmul(router_weights, self.chart_centers)  # [B, D]
        v_local = v - c_bar  # [B, D]

        codebook = self.codebook.unsqueeze(0)  # [1, N_c, K, D]
        v_exp = v_local.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
        diff = v_exp - codebook  # [B, N_c, K, D]
        dist = (diff**2).sum(dim=-1)  # [B, N_c, K]
        indices = torch.argmin(dist, dim=-1)  # [B, N_c]
        indices_stack = indices  # [B, N_c]

        indices_exp = indices.unsqueeze(-1).unsqueeze(-1)  # [B, N_c, 1, 1]
        indices_exp = indices_exp.expand(-1, -1, 1, self.latent_dim)  # [B, N_c, 1, D]
        z_q_all = torch.gather(codebook.expand(v.shape[0], -1, -1, -1), 2, indices_exp)
        z_q_all = z_q_all.squeeze(2)  # [B, N_c, D]

        w = router_weights.unsqueeze(-1).detach()  # [B, N_c, 1]
        v_bc = v_local.unsqueeze(1)  # [B, 1, D]
        commitment = ((v_bc - z_q_all.detach()) ** 2 * w).mean(dim=(0, 2)).sum()  # []
        codebook_loss = ((z_q_all - v_bc.detach()) ** 2 * w).mean(dim=(0, 2)).sum()  # []
        vq_loss = codebook_loss + 0.25 * commitment  # []

        z_q_blended = (z_q_all * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        K_code = indices_stack.gather(1, K_chart.unsqueeze(1)).squeeze(1)  # [B]

        delta = v_bc - z_q_all.detach()  # [B, N_c, D]
        z_n_all = self.structure_filter(delta.reshape(-1, self.latent_dim))  # [B*N_c, D]
        z_n_all_charts = z_n_all.view(v.shape[0], self.num_charts, self.latent_dim)  # [B, N_c, D]

        z_n = (z_n_all_charts * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        delta_blended = v_local - z_q_blended.detach()  # [B, D]
        z_tex = delta_blended - z_n  # [B, D]

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
        covariant_attn_tensorization: str = "sum",
        covariant_attn_rank: int = 8,
        covariant_attn_tau_min: float = 1e-2,
        covariant_attn_denom_min: float = 1e-3,
        covariant_attn_use_transport: bool = True,
        covariant_attn_transport_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.hidden_dim = hidden_dim
        self.covariant_attn = covariant_attn

        bundle_size, n_bundles = _resolve_bundle_params(hidden_dim, latent_dim, bundle_size)

        self.chart_projectors = nn.ModuleList(
            [SpectralLinear(latent_dim, hidden_dim, bias=False) for _ in range(num_charts)]
        )
        self.chart_gate = NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles)

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
        self.tex_residual = SpectralLinear(latent_dim, output_dim, bias=True)
        self.tex_residual_scale = nn.Parameter(torch.tensor(0.1))

        self.renderer = nn.Sequential(
            SpectralLinear(hidden_dim, hidden_dim, bias=True),
            NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
            SpectralLinear(hidden_dim, hidden_dim, bias=True),
            NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
            SpectralLinear(hidden_dim, output_dim, bias=True),
        )
        self.render_skip = SpectralLinear(hidden_dim, output_dim, bias=True)

    def forward(
        self,
        z_geo: torch.Tensor,
        z_tex: torch.Tensor | None = None,
        chart_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode from latent geometry."""
        z_geo = torch.tanh(z_geo)
        if chart_index is not None:
            router_weights = F.one_hot(chart_index, num_classes=self.num_charts).float()  # [B, N_c]
        else:
            if self.covariant_attn:
                router_weights, _ = self.cov_router(z_geo)
            else:
                logits = self.latent_router(z_geo)  # [B, N_c]
                router_weights = F.softmax(logits, dim=-1)  # [B, N_c]

        h_stack = torch.stack([proj(z_geo) for proj in self.chart_projectors], dim=1)  # [B, N_c, H]
        h_stack = self.chart_gate(h_stack.view(-1, self.hidden_dim)).view(
            z_geo.shape[0], self.num_charts, self.hidden_dim
        )
        h_global = (h_stack * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, H]

        x_hat = self.renderer(h_global) + self.render_skip(h_global)  # [B, D_out]
        if z_tex is not None:
            z_tex = torch.tanh(z_tex)
            x_hat = x_hat + self.tex_residual_scale * self.tex_residual(z_tex)
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
        covariant_attn_tensorization: str = "sum",
        covariant_attn_rank: int = 8,
        covariant_attn_tau_min: float = 1e-2,
        covariant_attn_denom_min: float = 1e-3,
        covariant_attn_use_transport: bool = True,
        covariant_attn_transport_eps: float = 1e-3,
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
        )

    def forward(
        self,
        x: torch.Tensor,
        use_hard_routing: bool = False,
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
        ) = self.encoder(x)

        chart_index = K_chart if use_hard_routing else None
        x_recon, dec_router_weights = self.decoder(z_geo, z_tex, chart_index)

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
