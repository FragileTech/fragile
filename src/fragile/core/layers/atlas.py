from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fragile.core.layers.primitives import IsotropicBlock, NormGatedGELU, SpectralLinear
from fragile.core.layers.ugn import SoftEquivariantLayer
from fragile.core.layers.vision import CovariantRetina, StandardResNetBackbone


class TokenSelfAttentionBlock(nn.Module):
    """Tokenized self-attention block for MLP baselines."""

    def __init__(
        self,
        hidden_dim: int,
        num_tokens: int,
        attn_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive.")
        if attn_dim <= 0:
            raise ValueError("attn_dim must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if attn_dim % num_heads != 0:
            raise ValueError("attn_dim must be divisible by num_heads.")
        self.num_tokens = num_tokens
        self.attn_dim = attn_dim
        self.to_tokens = nn.Linear(hidden_dim, num_tokens * attn_dim)
        self.attn = nn.MultiheadAttention(
            attn_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(attn_dim)
        self.out_proj = nn.Linear(num_tokens * attn_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.to_tokens(x).reshape(x.shape[0], self.num_tokens, self.attn_dim)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        attn_out = self.norm(attn_out + tokens)
        flat = attn_out.reshape(x.shape[0], self.num_tokens * self.attn_dim)
        return self.out_proj(flat) + x


class _BaselineEncoder(nn.Module):
    """Baseline encoder with optional vision backbone."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        use_attention: bool,
        attn_tokens: int,
        attn_dim: int,
        attn_heads: int,
        attn_dropout: float,
        vision_preproc: bool,
        vision_in_channels: int,
        vision_height: int,
        vision_width: int,
    ) -> None:
        super().__init__()
        self.vision_shape = None
        self.vision_preproc = None
        if vision_preproc:
            if vision_in_channels <= 0 or vision_height <= 0 or vision_width <= 0:
                raise ValueError("vision_preproc requires positive vision_* dimensions.")
            self.vision_shape = (vision_in_channels, vision_height, vision_width)
            self.vision_preproc = StandardResNetBackbone(
                in_channels=vision_in_channels,
                out_dim=hidden_dim,
            )
            self.feature_extractor = None
        else:
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
            )

        encoder_layers: list[nn.Module] = []
        if use_attention:
            encoder_layers.append(
                TokenSelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_tokens=attn_tokens,
                    attn_dim=attn_dim,
                    num_heads=attn_heads,
                    dropout=attn_dropout,
                )
            )
        encoder_layers.extend(
            [
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            ]
        )
        self.head = nn.Sequential(*encoder_layers)

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.vision_preproc is None:
            return self.feature_extractor(x)
        if self.vision_shape is None:
            raise RuntimeError("vision_preproc is enabled but vision_shape is unset.")
        channels, height, width = self.vision_shape
        if x.dim() == 2:
            expected = channels * height * width
            if x.shape[1] != expected:
                raise ValueError(
                    f"Expected flattened input dim {expected}, got {x.shape[1]}."
                )
            x = x.view(x.shape[0], channels, height, width)
        elif x.dim() == 4:
            if x.shape[1] != channels or x.shape[2] != height or x.shape[3] != width:
                raise ValueError(
                    "Input tensor shape does not match vision_preproc configuration."
                )
        else:
            raise ValueError("vision_preproc expects input shape [B, D] or [B, C, H, W].")
        return self.vision_preproc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._encode_features(x)
        return self.head(features)


class StandardVQ(nn.Module):
    """Standard Vector-Quantized VAE baseline."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_codes: int = 64,
        use_attention: bool = False,
        attn_tokens: int = 4,
        attn_dim: int = 32,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
        vision_preproc: bool = False,
        vision_in_channels: int = 0,
        vision_height: int = 0,
        vision_width: int = 0,
    ) -> None:
        super().__init__()
        self.num_codes = num_codes

        self.encoder = _BaselineEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            use_attention=use_attention,
            attn_tokens=attn_tokens,
            attn_dim=attn_dim,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout,
            vision_preproc=vision_preproc,
            vision_in_channels=vision_in_channels,
            vision_height=vision_height,
            vision_width=vision_width,
        )

        self.embeddings = nn.Embedding(num_codes, latent_dim)
        nn.init.uniform_(self.embeddings.weight, -1.0 / num_codes, 1.0 / num_codes)

        decoder_layers: list[nn.Module] = [
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        ]
        if use_attention:
            decoder_layers.append(
                TokenSelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_tokens=attn_tokens,
                    attn_dim=attn_dim,
                    num_heads=attn_heads,
                    dropout=attn_dropout,
                )
            )
        decoder_layers.extend(
            [
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            ]
        )
        self.decoder = nn.Sequential(*decoder_layers)

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

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        use_attention: bool = False,
        attn_tokens: int = 4,
        attn_dim: int = 32,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
        vision_preproc: bool = False,
        vision_in_channels: int = 0,
        vision_height: int = 0,
        vision_width: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = _BaselineEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            use_attention=use_attention,
            attn_tokens=attn_tokens,
            attn_dim=attn_dim,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout,
            vision_preproc=vision_preproc,
            vision_in_channels=vision_in_channels,
            vision_height=vision_height,
            vision_width=vision_width,
        )

        decoder_layers: list[nn.Module] = [
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        ]
        if use_attention:
            decoder_layers.append(
                TokenSelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_tokens=attn_tokens,
                    attn_dim=attn_dim,
                    num_heads=attn_heads,
                    dropout=attn_dropout,
                )
            )
        decoder_layers.extend(
            [
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            ]
        )
        self.decoder = nn.Sequential(*decoder_layers)

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
        vision_preproc: bool = False,
        vision_in_channels: int = 0,
        vision_height: int = 0,
        vision_width: int = 0,
        vision_num_rotations: int = 8,
        vision_kernel_size: int = 5,
        vision_use_reflections: bool = False,
        vision_norm_nonlinearity: str = "n_sigmoid",
        vision_norm_bias: bool = True,
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

        self.vision_shape = None
        self.vision_preproc = None
        if vision_preproc:
            if vision_in_channels <= 0 or vision_height <= 0 or vision_width <= 0:
                raise ValueError("vision_preproc requires positive vision_* dimensions.")
            self.vision_shape = (vision_in_channels, vision_height, vision_width)
            self.vision_preproc = CovariantRetina(
                in_channels=vision_in_channels,
                out_dim=hidden_dim,
                num_rotations=vision_num_rotations,
                kernel_size=vision_kernel_size,
                use_reflections=vision_use_reflections,
                norm_nonlinearity=vision_norm_nonlinearity,
                norm_bias=vision_norm_bias,
            )
            self.feature_extractor = None
        else:
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )

        self.val_proj = nn.Linear(hidden_dim, latent_dim)

        self.chart_centers = nn.Parameter(torch.randn(num_charts, latent_dim) * 0.02)

        self.codebook = nn.Parameter(torch.randn(num_charts, codes_per_chart, latent_dim) * 0.02)

        self.soft_equiv_layers: nn.ModuleList | None = None
        if soft_equiv_metric:
            bundle_size = soft_equiv_bundle_size or latent_dim
            if bundle_size <= 0:
                raise ValueError("soft_equiv_bundle_size must be positive.")
            if latent_dim % bundle_size != 0:
                raise ValueError("latent_dim must be divisible by soft_equiv_bundle_size.")
            n_bundles = latent_dim // bundle_size
            self.soft_equiv_layers = nn.ModuleList(
                [
                    SoftEquivariantLayer(
                        n_bundles=n_bundles,
                        bundle_dim=bundle_size,
                        hidden_dim=soft_equiv_hidden_dim,
                        use_spectral_norm=soft_equiv_use_spectral_norm,
                        zero_self_mixing=soft_equiv_zero_self_mixing,
                    )
                    for _ in range(num_charts)
                ]
            )
            _init_soft_equiv_layers(self.soft_equiv_layers)

        self.soft_equiv_soft_assign = soft_equiv_soft_assign
        self.soft_equiv_temperature = soft_equiv_temperature
        if (
            soft_equiv_metric
            and self.soft_equiv_soft_assign
            and self.soft_equiv_temperature <= 0
        ):
            raise ValueError(
                "soft_equiv_temperature must be positive when soft_equiv_soft_assign is enabled."
            )
        self._last_soft_equiv_log_ratio: torch.Tensor | None = None
        self._last_soft_equiv_log_ratio: torch.Tensor | None = None

        self.structure_filter = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2 if latent_dim > 2 else latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim // 2 if latent_dim > 2 else latent_dim, latent_dim),
        )

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.vision_preproc is None:
            return self.feature_extractor(x)
        if self.vision_shape is None:
            raise RuntimeError("vision_preproc is enabled but vision_shape is unset.")
        channels, height, width = self.vision_shape
        if x.dim() == 2:
            expected = channels * height * width
            if x.shape[1] != expected:
                raise ValueError(
                    f"Expected flattened input dim {expected}, got {x.shape[1]}."
                )
            x = x.view(x.shape[0], channels, height, width)
        elif x.dim() == 4:
            if x.shape[1] != channels or x.shape[2] != height or x.shape[3] != width:
                raise ValueError(
                    "Input tensor shape does not match vision_preproc configuration."
                )
        else:
            raise ValueError("vision_preproc expects input shape [B, D] or [B, C, H, W].")
        return self.vision_preproc(x)

    def _apply_soft_equiv_metric(self, diff: torch.Tensor) -> torch.Tensor:
        if self.soft_equiv_layers is None:
            self._last_soft_equiv_log_ratio = None
            return (diff**2).sum(dim=-1)
        batch_size, num_charts, num_codes, latent_dim = diff.shape
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
            diff_out = diff_out * scale
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
            total = total + layer.l1_loss()
        return total / len(self.soft_equiv_layers)

    def soft_equiv_log_ratio_loss(self) -> torch.Tensor:
        if self.soft_equiv_layers is None or self._last_soft_equiv_log_ratio is None:
            return torch.tensor(0.0, device=self.codebook.device)
        return self._last_soft_equiv_log_ratio

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
        features = self._encode_features(x)  # [B, H]

        v = self.val_proj(features)  # [B, D]
        scores = torch.matmul(v, self.chart_centers.t()) / math.sqrt(self.latent_dim)  # [B, N_c]
        router_weights = F.softmax(scores, dim=-1)  # [B, N_c]
        K_chart = torch.argmax(router_weights, dim=1)  # [B]

        c_bar = torch.matmul(router_weights, self.chart_centers)  # [B, D]
        v_local = v - c_bar  # [B, D]

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
            z_q_all = z_q_all + (z_q_soft - z_q_soft.detach())

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
        vision_preproc: bool = False,
        vision_in_channels: int = 0,
        vision_height: int = 0,
        vision_width: int = 0,
        vision_num_rotations: int = 8,
        vision_kernel_size: int = 5,
        vision_use_reflections: bool = False,
        vision_norm_nonlinearity: str = "n_sigmoid",
        vision_norm_bias: bool = True,
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
            vision_preproc=vision_preproc,
            vision_in_channels=vision_in_channels,
            vision_height=vision_height,
            vision_width=vision_width,
            vision_num_rotations=vision_num_rotations,
            vision_kernel_size=vision_kernel_size,
            vision_use_reflections=vision_use_reflections,
            vision_norm_nonlinearity=vision_norm_nonlinearity,
            vision_norm_bias=vision_norm_bias,
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


def _init_soft_equiv_layers(layers: nn.ModuleList) -> None:
    """Initialize soft-equivariant layers to be purely equivariant (no mixing)."""
    with torch.no_grad():
        for layer in layers:
            for row in layer.mixing_weights:
                for weight in row:
                    weight.zero_()


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
        vision_preproc: bool = False,
        vision_in_channels: int = 0,
        vision_height: int = 0,
        vision_width: int = 0,
        vision_num_rotations: int = 8,
        vision_kernel_size: int = 5,
        vision_use_reflections: bool = False,
        vision_norm_nonlinearity: str = "n_sigmoid",
        vision_norm_bias: bool = True,
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
        self.covariant_attn = covariant_attn

        bundle_size, n_bundles = _resolve_bundle_params(hidden_dim, latent_dim, bundle_size)

        self.vision_shape = None
        self.vision_preproc = None
        if vision_preproc:
            if vision_in_channels <= 0 or vision_height <= 0 or vision_width <= 0:
                raise ValueError("vision_preproc requires positive vision_* dimensions.")
            self.vision_shape = (vision_in_channels, vision_height, vision_width)
            self.vision_preproc = CovariantRetina(
                in_channels=vision_in_channels,
                out_dim=hidden_dim,
                num_rotations=vision_num_rotations,
                kernel_size=vision_kernel_size,
                use_reflections=vision_use_reflections,
                norm_nonlinearity=vision_norm_nonlinearity,
                norm_bias=vision_norm_bias,
            )
            self.feature_extractor = None
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

        self.chart_centers = nn.Parameter(torch.randn(num_charts, latent_dim) * 0.02)

        self.codebook = nn.Parameter(torch.randn(num_charts, codes_per_chart, latent_dim) * 0.02)

        self.soft_equiv_layers: nn.ModuleList | None = None
        if soft_equiv_metric:
            bundle_size = soft_equiv_bundle_size or latent_dim
            if bundle_size <= 0:
                raise ValueError("soft_equiv_bundle_size must be positive.")
            if latent_dim % bundle_size != 0:
                raise ValueError("latent_dim must be divisible by soft_equiv_bundle_size.")
            n_bundles = latent_dim // bundle_size
            self.soft_equiv_layers = nn.ModuleList(
                [
                    SoftEquivariantLayer(
                        n_bundles=n_bundles,
                        bundle_dim=bundle_size,
                        hidden_dim=soft_equiv_hidden_dim,
                        use_spectral_norm=soft_equiv_use_spectral_norm,
                        zero_self_mixing=soft_equiv_zero_self_mixing,
                    )
                    for _ in range(num_charts)
                ]
            )
            _init_soft_equiv_layers(self.soft_equiv_layers)

        self.soft_equiv_soft_assign = soft_equiv_soft_assign
        self.soft_equiv_temperature = soft_equiv_temperature
        if (
            soft_equiv_metric
            and self.soft_equiv_soft_assign
            and self.soft_equiv_temperature <= 0
        ):
            raise ValueError(
                "soft_equiv_temperature must be positive when soft_equiv_soft_assign is enabled."
            )

        self.structure_filter = nn.Sequential(
            IsotropicBlock(latent_dim, latent_dim, bundle_size=latent_dim),
            SpectralLinear(latent_dim, latent_dim, bias=True),
        )

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.vision_preproc is None:
            return self.feature_extractor(x)
        if self.vision_shape is None:
            raise RuntimeError("vision_preproc is enabled but vision_shape is unset.")
        channels, height, width = self.vision_shape
        if x.dim() == 2:
            expected = channels * height * width
            if x.shape[1] != expected:
                raise ValueError(
                    f"Expected flattened input dim {expected}, got {x.shape[1]}."
                )
            x = x.view(x.shape[0], channels, height, width)
        elif x.dim() == 4:
            if x.shape[1] != channels or x.shape[2] != height or x.shape[3] != width:
                raise ValueError(
                    "Input tensor shape does not match vision_preproc configuration."
                )
        else:
            raise ValueError("vision_preproc expects input shape [B, D] or [B, C, H, W].")
        return self.vision_preproc(x)

    def _apply_soft_equiv_metric(self, diff: torch.Tensor) -> torch.Tensor:
        if self.soft_equiv_layers is None:
            self._last_soft_equiv_log_ratio = None
            return (diff**2).sum(dim=-1)
        batch_size, num_charts, num_codes, latent_dim = diff.shape
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
            diff_out = diff_out * scale
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
            total = total + layer.l1_loss()
        return total / len(self.soft_equiv_layers)

    def soft_equiv_log_ratio_loss(self) -> torch.Tensor:
        if self.soft_equiv_layers is None or self._last_soft_equiv_log_ratio is None:
            return torch.tensor(0.0, device=self.codebook.device)
        return self._last_soft_equiv_log_ratio

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
        features = self._encode_features(x)  # [B, H]
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
            z_q_all = z_q_all + (z_q_soft - z_q_soft.detach())

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
        vision_preproc: bool = False,
        vision_in_channels: int = 0,
        vision_height: int = 0,
        vision_width: int = 0,
        vision_num_rotations: int = 8,
        vision_kernel_size: int = 5,
        vision_use_reflections: bool = False,
        vision_norm_nonlinearity: str = "n_sigmoid",
        vision_norm_bias: bool = True,
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
            vision_preproc=vision_preproc,
            vision_in_channels=vision_in_channels,
            vision_height=vision_height,
            vision_width=vision_width,
            vision_num_rotations=vision_num_rotations,
            vision_kernel_size=vision_kernel_size,
            vision_use_reflections=vision_use_reflections,
            vision_norm_nonlinearity=vision_norm_nonlinearity,
            vision_norm_bias=vision_norm_bias,
            soft_equiv_metric=soft_equiv_metric,
            soft_equiv_bundle_size=soft_equiv_bundle_size,
            soft_equiv_hidden_dim=soft_equiv_hidden_dim,
            soft_equiv_use_spectral_norm=soft_equiv_use_spectral_norm,
            soft_equiv_zero_self_mixing=soft_equiv_zero_self_mixing,
            soft_equiv_soft_assign=soft_equiv_soft_assign,
            soft_equiv_temperature=soft_equiv_temperature,
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
