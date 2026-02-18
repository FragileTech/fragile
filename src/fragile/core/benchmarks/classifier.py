from __future__ import annotations

import torch
from torch import nn


class BaselineClassifier(nn.Module):
    """Small MLP probe for baseline latent spaces."""

    def __init__(self, latent_dim: int, num_classes: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
