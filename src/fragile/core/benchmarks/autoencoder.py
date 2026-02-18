from __future__ import annotations

import torch
from torch import nn

from fragile.core.benchmarks.encoder import _BaselineEncoder, TokenSelfAttentionBlock
from fragile.core.layers.vision import StandardResNetDecoder


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

        self.decoder = None
        self.decoder_head = None
        self.vision_decoder = None
        if vision_preproc:
            decoder_layers = [
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
            decoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            ])
            self.decoder_head = nn.Sequential(*decoder_layers)
            self.vision_decoder = StandardResNetDecoder(
                in_dim=hidden_dim,
                out_channels=vision_in_channels,
                out_height=vision_height,
                out_width=vision_width,
            )
        else:
            decoder_layers = [
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
            decoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            ])
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
        if self.vision_decoder is not None:
            h = self.decoder_head(z)
            x_recon = self.vision_decoder(h).flatten(1)
        else:
            x_recon = self.decoder(z)
        return x_recon, z
