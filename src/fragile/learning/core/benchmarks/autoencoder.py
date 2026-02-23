from __future__ import annotations

import torch
from torch import nn

from .encoder import _BaselineEncoder, TokenSelfAttentionBlock


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
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
