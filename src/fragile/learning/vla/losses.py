"""VLA-specific loss functions and phase loss assemblers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fragile.learning.core.layers.gauge import ConformalMetric, hyperbolic_distance
from fragile.learning.hyperbolic_losses import (
    compute_chart_collapse_penalty,
    compute_code_collapse_penalty,
    compute_codebook_spread_loss,
    compute_diversity_loss,
    compute_hyperbolic_uniformity_loss,
    compute_radial_calibration_loss,
    compute_routing_entropy,
    compute_window_loss,
)

from .config import VLAConfig


# ---------------------------------------------------------------------------
# New dynamics losses
# ---------------------------------------------------------------------------


def compute_dynamics_geodesic_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
) -> torch.Tensor:
    """Mean geodesic distance between predicted and target latent trajectories.

    Args:
        z_pred: [B, H, D] predicted positions.
        z_target: [B, H, D] target positions.

    Returns:
        Scalar loss (mean hyperbolic distance across batch and horizon).
    """
    B, H, D = z_pred.shape
    pred_flat = z_pred.reshape(B * H, D)
    tgt_flat = z_target.reshape(B * H, D)
    return hyperbolic_distance(pred_flat, tgt_flat).mean()


def compute_dynamics_chart_loss(
    chart_logits: torch.Tensor,
    target_charts: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss for chart transition prediction.

    Args:
        chart_logits: [B, H, K] predicted chart logits.
        target_charts: [B, H] ground-truth chart indices.

    Returns:
        Scalar cross-entropy loss.
    """
    B, H, K = chart_logits.shape
    logits_flat = chart_logits.reshape(B * H, K)
    targets_flat = target_charts.reshape(B * H)
    return F.cross_entropy(logits_flat, targets_flat)


def compute_momentum_regularization(
    momenta: torch.Tensor,
    z_trajectory: torch.Tensor,
) -> torch.Tensor:
    """Metric-aware momentum penalty: mean kinetic energy 1/2 p^T G^{-1} p.

    Args:
        momenta: [B, H, D] momentum vectors.
        z_trajectory: [B, H, D] latent positions.

    Returns:
        Scalar regularization loss.
    """
    r_sq = (z_trajectory ** 2).sum(dim=-1, keepdim=True)  # [B, H, 1]
    g_inv_factor = ((1.0 - r_sq).clamp(min=1e-6) / 2.0) ** 2  # [B, H, 1]
    p_sq = (momenta ** 2).sum(dim=-1, keepdim=True)  # [B, H, 1]
    kinetic = 0.5 * g_inv_factor * p_sq
    return kinetic.mean()


def compute_energy_conservation_loss(
    phi_eff: torch.Tensor,
    momenta: torch.Tensor,
    z_trajectory: torch.Tensor,
) -> torch.Tensor:
    """Penalise Hamiltonian drift across horizon steps.

    Computes H = Φ_eff + ½ p^T G^{-1} p at each step and penalises the
    variance of H across the horizon (a perfectly symplectic integrator
    would keep H constant).

    Args:
        phi_eff: [B, H, 1] effective potential values.
        momenta: [B, H, D] momentum vectors.
        z_trajectory: [B, H, D] latent positions.

    Returns:
        Scalar loss (variance of H across horizon).
    """
    B, H, D = z_trajectory.shape
    # Kinetic energy: ½ |p|^2 * ((1 - |z|^2) / 2)^2  (diagonal inverse metric)
    r_sq = (z_trajectory ** 2).sum(dim=-1, keepdim=True)  # [B, H, 1]
    g_inv_factor = ((1.0 - r_sq).clamp(min=1e-6) / 2.0) ** 2  # [B, H, 1]
    p_sq = (momenta ** 2).sum(dim=-1, keepdim=True)  # [B, H, 1]
    kinetic = 0.5 * g_inv_factor * p_sq  # [B, H, 1]

    H_total = phi_eff + kinetic  # [B, H, 1]
    # Variance of Hamiltonian across horizon (per batch element, then mean)
    return H_total.squeeze(-1).var(dim=-1).mean()


def compute_jump_dynamics_loss(
    jump_rates: torch.Tensor,
    jump_masks: torch.Tensor,
) -> torch.Tensor:
    """L1 sparsity penalty on jump rates (jumps should be rare events).

    Args:
        jump_rates: [B, H, 1] Poisson rates.
        jump_masks: [B, H] boolean jump events (unused but available).

    Returns:
        Scalar L1 loss on jump rates.
    """
    return jump_rates.abs().mean()


# ---------------------------------------------------------------------------
# Phase loss assemblers
# ---------------------------------------------------------------------------


def compute_phase1_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    vq_loss: torch.Tensor,
    enc_router_weights: torch.Tensor,
    dec_router_weights: torch.Tensor,
    z_geo: torch.Tensor,
    encoder: torch.nn.Module,
    config: VLAConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Assemble Phase 1 (encoder-only) loss from all active terms.

    Returns:
        total_loss: Scalar loss for backward.
        metrics: Dict of individual loss components for logging.
    """
    metrics: dict[str, float] = {}

    # Feature reconstruction (MSE)
    loss_recon = F.mse_loss(x_recon, x)
    total = config.w_feature_recon * loss_recon
    metrics["recon"] = loss_recon.item()

    # VQ loss (from encoder)
    total = total + config.w_vq * vq_loss
    metrics["vq"] = vq_loss.item()

    # Routing entropy (encourage sharp routing)
    loss_entropy = compute_routing_entropy(enc_router_weights)
    total = total + config.w_entropy * loss_entropy
    metrics["entropy"] = loss_entropy.item()

    # Diversity (prevent chart collapse)
    loss_diversity = compute_diversity_loss(enc_router_weights, config.num_charts)
    total = total + config.w_diversity * loss_diversity
    metrics["diversity"] = loss_diversity.item()

    # Hyperbolic uniformity
    if config.w_uniformity > 0:
        loss_unif = compute_hyperbolic_uniformity_loss(z_geo)
        total = total + config.w_uniformity * loss_unif
        metrics["uniformity"] = loss_unif.item()

    # Radial calibration
    if config.w_radial_calibration > 0:
        loss_radcal = compute_radial_calibration_loss(
            z_geo, enc_router_weights, config.num_charts,
        )
        total = total + config.w_radial_calibration * loss_radcal
        metrics["radial_cal"] = loss_radcal.item()

    # Codebook spread
    if config.w_codebook_spread > 0 and hasattr(encoder, "encoder"):
        codebook = encoder.encoder.codebook
        loss_spread = compute_codebook_spread_loss(
            codebook, margin=config.w_codebook_spread_margin,
        )
        total = total + config.w_codebook_spread * loss_spread
        metrics["codebook_spread"] = loss_spread.item()

    # Chart collapse penalty
    if config.w_chart_collapse > 0:
        loss_chart_col = compute_chart_collapse_penalty(
            enc_router_weights, config.num_charts,
        )
        total = total + config.w_chart_collapse * loss_chart_col
        metrics["chart_collapse"] = loss_chart_col.item()

    # Code collapse penalty
    if config.w_code_collapse > 0 and hasattr(encoder, "encoder"):
        codebook = encoder.encoder.codebook
        # We need v_local from the encoder; approximate with z_geo projected
        loss_code_col = compute_code_collapse_penalty(
            z_geo, codebook, enc_router_weights,
            temperature=config.w_code_collapse_temperature,
        )
        total = total + config.w_code_collapse * loss_code_col
        metrics["code_collapse"] = loss_code_col.item()

    # Window loss
    if config.w_window > 0:
        loss_window, window_metrics = compute_window_loss(
            enc_router_weights, config.num_charts,
            eps_ground=config.w_window_eps_ground,
        )
        total = total + config.w_window * loss_window
        metrics["window"] = loss_window.item()
        metrics.update(window_metrics)

    # Encoder-decoder routing consistency
    if config.w_consistency > 0:
        eps = 1e-6
        kl = (enc_router_weights * torch.log(
            (enc_router_weights + eps) / (dec_router_weights + eps)
        )).sum(dim=-1).mean()
        total = total + config.w_consistency * kl
        metrics["consistency"] = kl.item()

    metrics["total"] = total.item()
    return total, metrics


def compute_phase2_loss(
    wm_output: dict[str, torch.Tensor],
    z_targets: torch.Tensor,
    chart_targets: torch.Tensor,
    config: VLAConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Assemble Phase 2 (world model) loss.

    Args:
        wm_output: Dict from ``GeometricWorldModel.forward``.
        z_targets: [B, H, D] ground-truth latent positions.
        chart_targets: [B, H] ground-truth chart indices.
        config: VLA configuration.

    Returns:
        total_loss, metrics dict.
    """
    metrics: dict[str, float] = {}

    loss_geo = compute_dynamics_geodesic_loss(wm_output["z_trajectory"], z_targets)
    total = config.w_geodesic * loss_geo
    metrics["geodesic"] = loss_geo.item()

    loss_chart = compute_dynamics_chart_loss(wm_output["chart_logits"], chart_targets)
    total = total + config.w_chart_transition * loss_chart
    metrics["chart_transition"] = loss_chart.item()

    loss_mom = compute_momentum_regularization(wm_output["momenta"], wm_output["z_trajectory"])
    total = total + config.w_momentum_reg * loss_mom
    metrics["momentum_reg"] = loss_mom.item()

    # Energy conservation loss (Hamiltonian drift)
    if config.w_energy_conservation > 0 and "phi_eff" in wm_output:
        loss_energy = compute_energy_conservation_loss(
            wm_output["phi_eff"], wm_output["momenta"], wm_output["z_trajectory"],
        )
        total = total + config.w_energy_conservation * loss_energy
        metrics["energy_conservation"] = loss_energy.item()

    # Jump dynamics sparsity loss
    if config.w_jump_dynamics > 0 and "jump_rates" in wm_output:
        loss_jd = compute_jump_dynamics_loss(
            wm_output["jump_rates"], wm_output["jump_masks"],
        )
        total = total + config.w_jump_dynamics * loss_jd
        metrics["jump_dynamics"] = loss_jd.item()

    metrics["total"] = total.item()
    return total, metrics


def compute_phase3_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    vq_loss: torch.Tensor,
    enc_router_weights: torch.Tensor,
    dec_router_weights: torch.Tensor,
    z_geo: torch.Tensor,
    encoder: torch.nn.Module,
    wm_output: dict[str, torch.Tensor],
    z_targets: torch.Tensor,
    chart_targets: torch.Tensor,
    config: VLAConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Assemble Phase 3 (joint fine-tuning) loss.

    Combines scaled encoder loss and dynamics loss.

    Returns:
        total_loss, metrics dict.
    """
    loss_enc, enc_metrics = compute_phase1_loss(
        x, x_recon, vq_loss, enc_router_weights, dec_router_weights,
        z_geo, encoder, config,
    )
    loss_dyn, dyn_metrics = compute_phase2_loss(
        wm_output, z_targets, chart_targets, config,
    )

    total = config.phase3_encoder_scale * loss_enc + config.phase3_dynamics_scale * loss_dyn

    metrics: dict[str, float] = {}
    for k, v in enc_metrics.items():
        metrics[f"enc/{k}"] = v
    for k, v in dyn_metrics.items():
        metrics[f"dyn/{k}"] = v
    metrics["total"] = total.item()

    return total, metrics
