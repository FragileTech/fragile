from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from fragile.core.benchmarks.encoder import _BaselineEncoder, TokenSelfAttentionBlock
from fragile.core.layers.vision import StandardResNetDecoder


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

        # Nearest-code lookup in latent space.
        z_sq = (z_e**2).sum(dim=1, keepdim=True)  # [B, 1]
        e_sq = (embed**2).sum(dim=1).unsqueeze(0)  # [1, K]
        dot = torch.matmul(z_e, embed.t())  # [B, K]
        dist = z_sq + e_sq - 2.0 * dot  # [B, K]

        indices = torch.argmin(dist, dim=1)  # [B]
        z_q = embed[indices]  # [B, D_latent]

        commitment = F.mse_loss(z_e, z_q.detach())  # []
        codebook = F.mse_loss(z_q, z_e.detach())  # []
        vq_loss = codebook + 0.25 * commitment  # []

        # Straight-through estimator to keep encoder gradients.
        z_st = z_e + (z_q - z_e).detach()  # [B, D_latent]
        if self.vision_decoder is not None:
            h = self.decoder_head(z_st)
            x_recon = self.vision_decoder(h).flatten(1)
        else:
            x_recon = self.decoder(z_st)
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
