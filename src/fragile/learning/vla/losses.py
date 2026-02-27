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
    n_probes: int = 3,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute Laplace-Beltrami operator Delta_G V on the Poincare ball.

    Uses **batched finite-difference Hutchinson trace estimation** with a
    single forward pass through V_func for all perturbation points.
    The Poincare correction term ``z . grad V`` is also computed via
    finite differences (no autograd passes required).

    .. math::
        \Delta_G f = \frac{1}{\lambda^2}
            \bigl[\Delta_E f + (D-2)\,\lambda\,(z \cdot \nabla f)\bigr]

    where :math:`\lambda(z) = 2/(1-|z|^2)`.

    **Hutchinson estimator** (Rademacher probes v_i):

    .. math::
        \mathrm{Tr}(H) \approx \frac{1}{k}\sum_{i=1}^{k}
            \frac{V(z+\varepsilon v_i) - 2\,V(z) + V(z-\varepsilon v_i)}
                 {\varepsilon^2}

    **Directional derivative** (finite-difference):

    .. math::
        z \cdot \nabla V \approx \|z\| \,
            \frac{V(z+\varepsilon\hat{z}) - V(z-\varepsilon\hat{z})}
                 {2\varepsilon}

    All (2k+3)*N perturbation points are evaluated in a single batched
    V_func call. Gradients flow through the network parameters via the
    forward pass (no ``create_graph=True`` needed).

    Args:
        V_func: Callable mapping z [B, D] -> V [B, 1] (differentiable).
        z: [B, D] positions inside the Poincare ball.
        n_probes: Number of Rademacher probe vectors (default 3).
        eps: Finite-difference step size (default 1e-3).

    Returns:
        lap_G: [B, 1] Laplace-Beltrami of V at each position.
        V_center: [B, 1] V evaluated at z (reusable by caller).
    """
    N, D = z.shape
    device = z.device
    dtype = z.dtype
    z_det = z.detach()

    # --- Probe directions ---
    probes = torch.randint(0, 2, (n_probes, N, D), device=device, dtype=dtype) * 2 - 1

    # Radial unit direction: z_hat = z / ||z||  (clamped for origin safety)
    z_norm = z_det.norm(dim=-1, keepdim=True)  # [N, 1]
    z_hat = z_det / z_norm.clamp(min=1e-8)  # [N, D]

    # --- Build all perturbation points in one stack ---
    # Layout: [center, +v1, -v1, +v2, -v2, ..., +z_hat, -z_hat]
    points = [z_det]
    for i in range(n_probes):
        points.append(z_det + eps * probes[i])
        points.append(z_det - eps * probes[i])
    points.append(z_det + eps * z_hat)
    points.append(z_det - eps * z_hat)

    z_all = torch.cat(points, dim=0)  # [(2k+3)*N, D]

    # --- Single batched forward pass ---
    V_all = V_func(z_all)  # [(2k+3)*N, 1]

    # --- Split results ---
    n_groups = 2 * n_probes + 3
    V_split = V_all.reshape(n_groups, N, 1)

    V_center = V_split[0]  # [N, 1]

    # --- Hutchinson trace estimate (vectorised over probes) ---
    V_plus = V_split[1:1 + 2 * n_probes:2]   # [k, N, 1]
    V_minus = V_split[2:2 + 2 * n_probes:2]  # [k, N, 1]
    trace_terms = V_plus - 2 * V_center.unsqueeze(0) + V_minus  # [k, N, 1]
    laplacian_E = trace_terms.sum(dim=0) / (n_probes * eps ** 2)  # [N, 1]

    # --- Directional derivative: z · ∇V ≈ ||z|| * (V(z+εẑ) - V(z-εẑ)) / (2ε) ---
    V_zhat_plus = V_split[-2]   # [N, 1]
    V_zhat_minus = V_split[-1]  # [N, 1]
    z_dot_grad = z_norm * (V_zhat_plus - V_zhat_minus) / (2 * eps)  # [N, 1]

    # --- Poincare ball correction ---
    r_sq = (z_det ** 2).sum(dim=-1, keepdim=True)  # [N, 1]
    one_minus_r_sq = (1.0 - r_sq).clamp(min=1e-6)
    lambda_z = 2.0 / one_minus_r_sq  # [N, 1]
    inv_lambda_sq = (one_minus_r_sq / 2.0) ** 2  # [N, 1]

    lap_G = inv_lambda_sq * (laplacian_E + (D - 2) * lambda_z * z_dot_grad)

    return lap_G, V_center  # [N, 1], [N, 1]


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

    # Router weights: single row, broadcasts to any batch size inside V_func
    rw_row = router_weights.mean(dim=0, keepdim=True)  # [1, K]

    # Define V as a function of z (handles arbitrary batch size)
    def V_func(z_in: torch.Tensor) -> torch.Tensor:
        n_in = z_in.shape[0]
        rw_in = rw_row.expand(n_in, -1)
        ctx_x, ctx_z = potential_net.chart_tok(rw_in, z_in)
        x_q = potential_net.z_embed(z_in)
        v_feat, _ = potential_net.v_critic_attn(z_in, ctx_z, x_q, ctx_x, ctx_x)
        return potential_net.v_out(v_feat)  # [n_in, 1]

    # Single batched call inside hyperbolic_laplacian (returns V_center too)
    lap_V, V_center = hyperbolic_laplacian(V_func, z_flat)  # [N, 1], [N, 1]

    # PDE residual: (-Delta_G + kappa^2) V - rho_r
    residual = -lap_V + kappa ** 2 * V_center - rho_r

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
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Assemble Phase 1 (encoder-only) loss from all active terms.

    Returns:
        base_loss: Scalar loss for terms that do NOT flow through z_n.
        zn_reg_loss: Scalar loss for z_n regularization (uniformity, radial_cal).
        metrics: Dict of individual loss components for logging.
    """
    metrics: dict[str, float] = {}

    # Feature reconstruction (MSE)
    loss_recon = F.mse_loss(x_recon, x)
    base_loss = config.w_feature_recon * loss_recon
    metrics["recon"] = loss_recon.item()

    # VQ loss (from encoder)
    base_loss = base_loss + config.w_vq * vq_loss
    metrics["vq"] = vq_loss.item()

    # Routing entropy (encourage sharp routing)
    loss_entropy = compute_routing_entropy(enc_router_weights)
    base_loss = base_loss + config.w_entropy * loss_entropy
    metrics["entropy"] = loss_entropy.item()

    # Diversity (prevent chart collapse)
    loss_diversity = compute_diversity_loss(enc_router_weights, config.num_charts)
    base_loss = base_loss + config.w_diversity * loss_diversity
    metrics["diversity"] = loss_diversity.item()

    # z_n regularization terms (flow through z_geo -> z_n)
    zn_reg_loss = torch.zeros((), device=x.device)

    # Hyperbolic uniformity
    if config.w_uniformity > 0:
        loss_unif = compute_hyperbolic_uniformity_loss(z_geo)
        zn_reg_loss = zn_reg_loss + config.w_uniformity * loss_unif
        metrics["uniformity"] = loss_unif.item()

    # Radial calibration
    if config.w_radial_calibration > 0:
        loss_radcal = compute_radial_calibration_loss(
            z_geo, enc_router_weights, config.num_charts,
        )
        zn_reg_loss = zn_reg_loss + config.w_radial_calibration * loss_radcal
        metrics["radial_cal"] = loss_radcal.item()

    # Codebook spread
    if config.w_codebook_spread > 0 and hasattr(encoder, "encoder"):
        codebook = encoder.encoder.codebook
        loss_spread = compute_codebook_spread_loss(
            codebook, margin=config.w_codebook_spread_margin,
        )
        base_loss = base_loss + config.w_codebook_spread * loss_spread
        metrics["codebook_spread"] = loss_spread.item()

    # Chart collapse penalty
    if config.w_chart_collapse > 0:
        loss_chart_col = compute_chart_collapse_penalty(
            enc_router_weights, config.num_charts,
        )
        base_loss = base_loss + config.w_chart_collapse * loss_chart_col
        metrics["chart_collapse"] = loss_chart_col.item()

    # Code collapse penalty
    if config.w_code_collapse > 0 and hasattr(encoder, "encoder"):
        codebook = encoder.encoder.codebook
        # We need v_local from the encoder; approximate with z_geo projected
        loss_code_col = compute_code_collapse_penalty(
            z_geo, codebook, enc_router_weights,
            temperature=config.w_code_collapse_temperature,
        )
        base_loss = base_loss + config.w_code_collapse * loss_code_col
        metrics["code_collapse"] = loss_code_col.item()

    # Window loss
    if config.w_window > 0:
        loss_window, window_metrics = compute_window_loss(
            enc_router_weights, config.num_charts,
            eps_ground=config.w_window_eps_ground,
        )
        base_loss = base_loss + config.w_window * loss_window
        metrics["window"] = loss_window.item()
        metrics.update(window_metrics)

    # Encoder-decoder routing consistency
    if config.w_consistency > 0:
        eps = 1e-6
        kl = (enc_router_weights * torch.log(
            (enc_router_weights + eps) / (dec_router_weights + eps)
        )).sum(dim=-1).mean()
        base_loss = base_loss + config.w_consistency * kl
        metrics["consistency"] = kl.item()

    total = base_loss + zn_reg_loss
    metrics["total"] = total.item()
    return base_loss, zn_reg_loss, metrics


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Assemble Phase 3 (joint fine-tuning) loss.

    Combines scaled encoder loss and dynamics loss, returning split
    components for gradient surgery.

    Returns:
        base_enc_loss: Encoder base loss (no z_n gradient path).
        zn_reg_loss: z_n regularization loss (uniformity, radial_cal).
        dyn_loss: Dynamics loss from world model.
        metrics: Dict of individual loss components for logging.
    """
    base_enc, zn_reg, enc_metrics = compute_phase1_loss(
        x, x_recon, vq_loss, enc_router_weights, dec_router_weights,
        z_geo, encoder, config,
    )
    loss_dyn, dyn_metrics = compute_phase2_loss(
        wm_output, z_targets, chart_targets, config,
    )

    total = (
        config.phase3_encoder_scale * base_enc
        + config.phase3_zn_reg_scale * zn_reg
        + config.phase3_dynamics_scale * loss_dyn
    )

    metrics: dict[str, float] = {}
    for k, v in enc_metrics.items():
        metrics[f"enc/{k}"] = v
    for k, v in dyn_metrics.items():
        metrics[f"dyn/{k}"] = v
    metrics["total"] = total.item()

    return base_enc, zn_reg, loss_dyn, metrics
