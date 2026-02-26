"""VLA-specific loss functions and phase loss assemblers."""

from __future__ import annotations

from typing import Callable

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


def compute_hodge_consistency_loss(
    hodge_harmonic_forces: torch.Tensor,
) -> torch.Tensor:
    """Penalize the harmonic residual in the Hodge decomposition.

    A well-structured model should explain all forces through either
    the conservative potential or the solenoidal curl field. The harmonic
    residual should be small.

    Args:
        hodge_harmonic_forces: [B, H, D] harmonic force residuals.

    Returns:
        Scalar L2 loss on harmonic forces.
    """
    return (hodge_harmonic_forces ** 2).mean()


# ---------------------------------------------------------------------------
# Screened Poisson critic (PDE residual loss)
# ---------------------------------------------------------------------------


def hyperbolic_laplacian(
    V_func: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
) -> torch.Tensor:
    """Compute Laplace-Beltrami operator Delta_G V at positions z on the Poincare ball.

    Uses autograd to compute the Euclidean gradient and Hessian trace, then
    applies the Poincare ball conformal correction (curvature c=1):

        Delta_G f = (1/lambda^2) [Delta_E f + (D-2) lambda (z . grad f)]

    where lambda(z) = 2 / (1 - |z|^2).

    Args:
        V_func: Callable mapping z [B, D] -> V [B, 1] (must be differentiable).
        z: [B, D] positions (requires_grad will be set internally).

    Returns:
        lap_V: [B, 1] Laplace-Beltrami of V at each position.
    """
    z = z.detach().requires_grad_(True)
    V = V_func(z)  # [B, 1]

    # Gradient: nabla V  [B, D]
    (grad_V,) = torch.autograd.grad(
        V.sum(), z, create_graph=True, allow_unused=True,
    )
    if grad_V is None or not grad_V.requires_grad:
        # V is constant or linear-only w.r.t. z — Hessian trace is zero.
        # Laplacian only has the (D-2)*lambda*(z.gradV) correction for linear V.
        if grad_V is not None:
            r_sq = (z ** 2).sum(dim=-1, keepdim=True)
            one_minus_r_sq = (1.0 - r_sq).clamp(min=1e-6)
            lambda_z = 2.0 / one_minus_r_sq
            z_dot_grad = (z * grad_V).sum(dim=-1, keepdim=True)
            inv_lambda_sq = (one_minus_r_sq / 2.0) ** 2
            D = z.shape[-1]
            return inv_lambda_sq * (D - 2) * lambda_z * z_dot_grad
        return torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)

    # Euclidean Laplacian: Delta_E V = sum_i d^2 V / dz_i^2
    D = z.shape[-1]
    laplacian_E = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
    for i in range(D):
        grad2 = torch.autograd.grad(
            grad_V[:, i].sum(), z, create_graph=True, allow_unused=True,
        )[0]
        if grad2 is not None:
            laplacian_E[:, 0] = laplacian_E[:, 0] + grad2[:, i]

    # Poincare ball correction
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
    one_minus_r_sq = (1.0 - r_sq).clamp(min=1e-6)
    lambda_z = 2.0 / one_minus_r_sq  # [B, 1]

    # z . grad V
    z_dot_grad = (z * grad_V).sum(dim=-1, keepdim=True)  # [B, 1]

    # Delta_G V = (1/lambda^2) [Delta_E V + (D-2) lambda (z . grad V)]
    inv_lambda_sq = (one_minus_r_sq / 2.0) ** 2
    lap_G = inv_lambda_sq * (laplacian_E + (D - 2) * lambda_z * z_dot_grad)

    return lap_G  # [B, 1]


def compute_screened_poisson_loss(
    potential_net: torch.nn.Module,
    z_trajectory: torch.Tensor,
    z_targets: torch.Tensor,
    router_weights: torch.Tensor,
    kappa: float = 1.0,
    max_samples: int = 64,
) -> torch.Tensor:
    """PDE residual loss: ||(-Delta_G + kappa^2) V - rho_r||^2.

    Enforces that the critic V approximately solves the screened Poisson
    equation on the Poincare ball, with reward density rho_r approximated
    by the geodesic miss-distance to target.

    Args:
        potential_net: The CovariantPotentialNet module.
        z_trajectory: [B, H, D] predicted positions.
        z_targets: [B, H, D] target positions.
        router_weights: [B, K] chart routing weights.
        kappa: Screening mass (controls decay length).
        max_samples: Max z samples to evaluate (limits second-order grad cost).

    Returns:
        Scalar PDE residual loss.
    """
    B, H, D = z_trajectory.shape

    # Flatten and subsample for efficiency
    z_flat = z_trajectory.reshape(B * H, D)
    z_tgt_flat = z_targets.reshape(B * H, D)
    n = z_flat.shape[0]
    if n > max_samples:
        idx = torch.randperm(n, device=z_flat.device)[:max_samples]
        z_flat = z_flat[idx]
        z_tgt_flat = z_tgt_flat[idx]

    # Reward density proxy: geodesic distance to target
    rho_r = hyperbolic_distance(z_flat.detach(), z_tgt_flat.detach()).unsqueeze(-1)  # [N, 1]

    # Router weights for V evaluation (broadcast mean)
    rw_expanded = router_weights.mean(dim=0, keepdim=True).expand(z_flat.shape[0], -1)  # [N, K]

    # Define V as a function of z for autograd
    def V_func(z_in: torch.Tensor) -> torch.Tensor:
        n_in = z_in.shape[0]
        rw_in = rw_expanded[:n_in]
        ctx_x, ctx_z = potential_net.chart_tok(rw_in, z_in)
        x_q = potential_net.z_embed(z_in)
        v_feat, _ = potential_net.v_critic_attn(z_in, ctx_z, x_q, ctx_x, ctx_x)
        return potential_net.v_out(v_feat)  # [N, 1]

    # Compute Laplace-Beltrami of V (creates graph for backprop)
    lap_V = hyperbolic_laplacian(V_func, z_flat)  # [N, 1]

    # V at the same points (recompute for clean graph on V_val side)
    V_val = V_func(z_flat.detach().requires_grad_(False))  # [N, 1]

    # PDE residual: (-Delta_G + kappa^2) V - rho_r
    residual = -lap_V + kappa ** 2 * V_val - rho_r

    return (residual ** 2).mean()


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

    # Hodge consistency loss
    if getattr(config, "w_hodge", 0.0) > 0 and "hodge_harmonic_forces" in wm_output:
        loss_hodge = compute_hodge_consistency_loss(wm_output["hodge_harmonic_forces"])
        total = total + config.w_hodge * loss_hodge
        metrics["hodge"] = loss_hodge.item()

    # Screened Poisson critic loss
    if getattr(config, "w_screened_poisson", 0.0) > 0 and "potential_net" in wm_output:
        rw = wm_output.get(
            "router_weights_final",
            torch.softmax(wm_output["chart_logits"][:, -1, :], dim=-1),
        )
        loss_sp = compute_screened_poisson_loss(
            wm_output["potential_net"],
            wm_output["z_trajectory"],
            wm_output.get("z_targets", z_targets),
            rw,
            kappa=getattr(config, "wm_screening_kappa", 1.0),
        )
        total = total + config.w_screened_poisson * loss_sp
        metrics["screened_poisson"] = loss_sp.item()

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
