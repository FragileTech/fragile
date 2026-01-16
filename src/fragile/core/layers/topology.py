from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def class_modulated_jump_rate(
    lambda_base: torch.Tensor,
    chart_to_class: torch.Tensor,
    gamma_sep: float = 5.0,
) -> torch.Tensor:
    """Compute class-consistent jump rates.

    Args:
        lambda_base: [N_c, N_c] base jump rates
        chart_to_class: [N_c, C] chart-to-class logits
        gamma_sep: separation strength

    Returns:
        lambda_sup: [N_c, N_c] modulated rates
    """
    dominant = torch.argmax(chart_to_class, dim=1)  # [N_c]
    diff = dominant.unsqueeze(1) != dominant.unsqueeze(0)  # [N_c, N_c]
    lambda_sup = lambda_base * torch.exp(-gamma_sep * diff.float())  # [N_c, N_c]
    return lambda_sup


class SupervisedTopologyLoss(nn.Module):
    """Supervised topology loss enforcing purity, balance, and metric separation."""

    def __init__(
        self,
        num_charts: int,
        num_classes: int,
        lambda_purity: float = 0.1,
        lambda_balance: float = 0.01,
        lambda_metric: float = 0.01,
        margin: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.num_classes = num_classes
        self.lambda_purity = lambda_purity
        self.lambda_balance = lambda_balance
        self.lambda_metric = lambda_metric
        self.margin = margin
        self.temperature = temperature

        self.chart_to_class = nn.Parameter(torch.randn(num_charts, num_classes) * 0.01)

    @property
    def p_y_given_k(self) -> torch.Tensor:
        """Chart-to-class probabilities.

        Returns:
            p_y_given_k: [N_c, C] class probabilities per chart
        """
        return F.softmax(self.chart_to_class / self.temperature, dim=1)

    def forward(
        self,
        chart_assignments: torch.Tensor,
        class_labels: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute supervised topology losses.

        Args:
            chart_assignments: [B, N_c] soft routing weights
            class_labels: [B] ground-truth classes
            embeddings: [B, D] latent embeddings

        Returns:
            total_loss: [] total loss
            loss_dict: dict of individual losses
        """
        p_y_k = self.p_y_given_k  # [N_c, C]
        p_k = chart_assignments.mean(dim=0)  # [N_c]

        p_y_x = torch.matmul(chart_assignments, p_y_k)  # [B, C]
        loss_route = F.cross_entropy(torch.log(p_y_x + 1e-8), class_labels)  # []

        entropy_per_chart = -(p_y_k * torch.log(p_y_k + 1e-8)).sum(dim=1)  # [N_c]
        loss_purity = (p_k * entropy_per_chart).sum()  # []

        uniform = torch.ones_like(p_k) / self.num_charts  # [N_c]
        loss_balance = (p_k * (torch.log(p_k + 1e-8) - torch.log(uniform))).sum()  # []

        if self.lambda_metric > 0 and embeddings.shape[0] > 1:
            dists = torch.cdist(embeddings, embeddings)  # [B, B]
            match = (class_labels.unsqueeze(1) == class_labels.unsqueeze(0)).float()  # [B, B]
            diff = 1.0 - match  # [B, B]
            pos_loss = (match * dists).sum() / (match.sum() + 1e-8)  # []
            neg_loss = (diff * F.relu(self.margin - dists)).sum() / (diff.sum() + 1e-8)  # []
            loss_metric = pos_loss + neg_loss  # []
        else:
            loss_metric = torch.tensor(0.0, device=chart_assignments.device)  # []

        total_loss = (  # []
            loss_route
            + self.lambda_purity * loss_purity
            + self.lambda_balance * loss_balance
            + self.lambda_metric * loss_metric
        )

        return total_loss, {
            "loss_total": total_loss,
            "loss_route": loss_route,
            "loss_purity": loss_purity,
            "loss_balance": loss_balance,
            "loss_metric": loss_metric,
        }


class FactorizedJumpOperator(nn.Module):
    """Factorized jump operator between charts."""

    def __init__(self, num_charts: int, latent_dim: int, global_rank: int = 0) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.latent_dim = latent_dim
        self.rank = global_rank if global_rank > 0 else latent_dim

        self.B = nn.Parameter(torch.randn(num_charts, self.rank, latent_dim))
        self.c = nn.Parameter(torch.zeros(num_charts, self.rank))
        self.A = nn.Parameter(torch.randn(num_charts, latent_dim, self.rank))
        self.d = nn.Parameter(torch.zeros(num_charts, latent_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize parameters near identity."""
        eye_b = torch.eye(self.rank, self.latent_dim)  # [R, D]
        eye_a = torch.eye(self.latent_dim, self.rank)  # [D, R]
        self.B.data = eye_b.unsqueeze(0).repeat(self.num_charts, 1, 1)
        self.A.data = eye_a.unsqueeze(0).repeat(self.num_charts, 1, 1)
        self.B.data = self.B.data + torch.randn_like(self.B) * 0.01
        self.A.data = self.A.data + torch.randn_like(self.A) * 0.01

    def forward(
        self,
        z_n: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Apply chart transition.

        Args:
            z_n: [B, D] source nuisance coordinates
            source_idx: [B] source chart indices
            target_idx: [B] target chart indices

        Returns:
            z_out: [B, D] target nuisance coordinates
        """
        B_src = self.B[source_idx]  # [B, R, D]
        c_src = self.c[source_idx]  # [B, R]
        z_global = torch.bmm(B_src, z_n.unsqueeze(-1)).squeeze(-1) + c_src  # [B, R]

        A_tgt = self.A[target_idx]  # [B, D, R]
        d_tgt = self.d[target_idx]  # [B, D]
        z_out = torch.bmm(A_tgt, z_global.unsqueeze(-1)).squeeze(-1) + d_tgt  # [B, D]

        return z_out

    def get_transition_matrix(self, source: int, target: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return affine map (M, b) for chart transition.

        Returns:
            M: [D, D] linear map
            b: [D] bias
        """
        M = self.A[target] @ self.B[source]  # [D, D]
        b = self.A[target] @ self.c[source] + self.d[target]  # [D]
        return M, b
