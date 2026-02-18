from __future__ import annotations

import torch
from torch import nn

from fragile.core.layers.vision import StandardResNetBackbone


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
            msg = "num_tokens must be positive."
            raise ValueError(msg)
        if attn_dim <= 0:
            msg = "attn_dim must be positive."
            raise ValueError(msg)
        if num_heads <= 0:
            msg = "num_heads must be positive."
            raise ValueError(msg)
        if attn_dim % num_heads != 0:
            msg = "attn_dim must be divisible by num_heads."
            raise ValueError(msg)
        self.num_tokens = num_tokens
        self.attn_dim = attn_dim
        self.to_tokens = nn.Linear(hidden_dim, num_tokens * attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, num_heads, dropout=dropout, batch_first=True)
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
                msg = "vision_preproc requires positive vision_* dimensions."
                raise ValueError(msg)
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
        encoder_layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        ])
        self.head = nn.Sequential(*encoder_layers)

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.vision_preproc is None:
            return self.feature_extractor(x)
        if self.vision_shape is None:
            msg = "vision_preproc is enabled but vision_shape is unset."
            raise RuntimeError(msg)
        channels, height, width = self.vision_shape
        if x.dim() == 2:
            expected = channels * height * width
            if x.shape[1] != expected:
                raise ValueError(f"Expected flattened input dim {expected}, got {x.shape[1]}.")
            x = x.view(x.shape[0], channels, height, width)
        elif x.dim() == 4:
            if x.shape[1] != channels or x.shape[2] != height or x.shape[3] != width:
                msg = "Input tensor shape does not match vision_preproc configuration."
                raise ValueError(msg)
        else:
            msg = "vision_preproc expects input shape [B, D] or [B, C, H, W]."
            raise ValueError(msg)
        return self.vision_preproc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._encode_features(x)
        return self.head(features)
