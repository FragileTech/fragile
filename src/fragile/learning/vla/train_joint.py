"""Joint encoder + world model training with 3-phase structure.

Combines the detailed loss display from ``train_unsupervised.py`` with the
Boris-BAOAB world model from ``train.py``.  Each phase's epoch count is a
CLI arg (set to 0 to skip).

Phase 1: Encoder warmup (single frames, 13 losses)
Phase 2: World model warmup (encoder frozen, dynamics losses)
Phase 3: Joint fine-tuning (encoder + world model, combined losses)
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from fragile.learning.checkpoints import (
    compute_grad_norm,
    compute_param_norm,
    count_parameters,
)
from fragile.learning.core.layers import FactorizedJumpOperator, TopoEncoderPrimitives
from fragile.learning.core.layers.topology import compute_jump_consistency_loss
from fragile.learning.hyperbolic_losses import (
    compute_chart_collapse_penalty,
    compute_code_collapse_penalty,
    compute_codebook_centering_loss,
    compute_codebook_spread_loss,
    compute_diversity_loss,
    compute_hyperbolic_uniformity_loss,
    compute_radial_calibration_loss,
    compute_routing_entropy,
    compute_window_loss,
    get_jump_weight_schedule,
)
from fragile.learning.core.layers.gauge import hyperbolic_distance
from fragile.learning.vla.extract_features import VLAFeatureDataset
from fragile.learning.vla.losses import compute_phase2_loss
from fragile.learning.vla.covariant_world_model import GeometricWorldModel

# ── Tracked loss terms ────────────────────────────────────────────
ENCODER_LOSS_KEYS = [
    "recon", "vq", "entropy", "consistency",
    "diversity", "uniformity", "radial_cal",
    "codebook_spread", "codebook_center",
    "chart_collapse", "code_collapse",
    "window", "jump",
]

DYNAMICS_LOSS_KEYS = [
    "geodesic", "chart_transition", "momentum_reg",
    "energy_conservation", "jump_dynamics", "screened_poisson", "hodge",
]

INFO_KEYS = [
    "I_XK", "H_K", "H_K_given_X",
    "grad_norm", "param_norm", "update_ratio", "lr",
]


def _init_encoder_accumulators() -> dict[str, float]:
    return {k: 0.0 for k in ENCODER_LOSS_KEYS + INFO_KEYS + ["total"]}


WM_DIAG_KEYS = [
    "mean_momentum", "mean_jump_rate", "jump_fraction", "mean_phi_eff",
    "hodge_cons", "hodge_sol", "hodge_harm",
]


def _init_dynamics_accumulators() -> dict[str, float]:
    return {k: 0.0 for k in
            DYNAMICS_LOSS_KEYS + WM_DIAG_KEYS
            + ["grad_norm", "param_norm", "update_ratio", "lr", "total"]}


def _init_joint_accumulators() -> dict[str, float]:
    keys = (
        [f"enc/{k}" for k in ENCODER_LOSS_KEYS]
        + [f"dyn/{k}" for k in DYNAMICS_LOSS_KEYS]
        + [f"wm/{k}" for k in WM_DIAG_KEYS]
        + INFO_KEYS
        + ["enc_total", "dyn_total", "total"]
    )
    return {k: 0.0 for k in keys}


def _wm_diagnostics(wm_output: dict[str, torch.Tensor]) -> dict[str, float]:
    """Extract diagnostic scalars from world model output."""
    momenta = wm_output["momenta"]
    jump_rates = wm_output["jump_rates"]
    jump_masks = wm_output["jump_masks"]
    phi_eff = wm_output["phi_eff"]
    diag = {
        "mean_momentum": momenta.norm(dim=-1).mean().item(),
        "mean_jump_rate": jump_rates.mean().item(),
        "jump_fraction": jump_masks.float().mean().item(),
        "mean_phi_eff": phi_eff.mean().item(),
        "hodge_cons": 0.0,
        "hodge_sol": 0.0,
        "hodge_harm": 0.0,
    }
    if "hodge_conservative_ratio" in wm_output:
        diag["hodge_cons"] = wm_output["hodge_conservative_ratio"].mean().item()
        diag["hodge_sol"] = wm_output["hodge_solenoidal_ratio"].mean().item()
        diag["hodge_harm"] = wm_output["hodge_harmonic_ratio"].mean().item()
    return diag


def _chart_stats_from_tensor(
    K_all: torch.Tensor, num_charts: int,
) -> tuple[np.ndarray, float, int]:
    """Compute usage, perplexity, active count from chart assignments."""
    K_np = K_all.detach().cpu().reshape(-1).numpy()
    usage = np.zeros(num_charts)
    for c in K_np:
        usage[int(c)] += 1
    usage /= usage.sum() + 1e-8
    perplexity = float(np.exp(-np.sum(usage * np.log(usage + 1e-8))))
    active = int((usage > 0.01).sum())
    return usage, perplexity, active


# ── Hard-routing tau annealing ────────────────────────────────────


def _get_hard_routing_tau(args: argparse.Namespace, epoch: int, total_epochs: int) -> float:
    """Compute annealed hard-routing temperature for the current epoch.

    Linear anneal from ``args.hard_routing_tau`` to ``args.hard_routing_tau_end``
    over ``args.hard_routing_tau_anneal_epochs`` epochs (defaults to *total_epochs*).
    Returns constant ``args.hard_routing_tau`` when ``tau_end`` is None.
    """
    if args.hard_routing_tau_end is None:
        return args.hard_routing_tau
    anneal_epochs = args.hard_routing_tau_anneal_epochs or total_epochs
    if anneal_epochs <= 0:
        return args.hard_routing_tau_end
    t = min(epoch / anneal_epochs, 1.0)
    return args.hard_routing_tau + t * (args.hard_routing_tau_end - args.hard_routing_tau)


# ── Encoder loss computation (same as train_unsupervised.py) ──────


def _compute_encoder_losses(
    x: torch.Tensor,
    model: TopoEncoderPrimitives,
    jump_op: FactorizedJumpOperator,
    args: argparse.Namespace,
    epoch: int,
    hard_routing: bool = False,
    hard_routing_tau: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute all 13 encoder losses with split encoder/decoder calls.

    Returns (total_loss, metrics, z_geo, enc_w, K_chart) so callers can
    reuse encoder outputs without re-encoding.
    """
    (
        K_chart, K_code, z_n, z_tex, enc_w, z_geo,
        vq_loss, indices, z_n_all, c_bar, v_local,
    ) = model.encoder(x, hard_routing=hard_routing, hard_routing_tau=hard_routing_tau)

    # When hard routing is on, pass encoder weights to decoder so both use the
    # same one-hot assignment (matching TopoEncoderPrimitives.forward).  Without
    # this the decoder draws an independent Gumbel sample, consistency loss
    # explodes, and training diverges.
    router_override = enc_w if hard_routing else None
    x_recon, dec_w, aux_losses = model.decoder(
        z_geo, z_tex, chart_index=None,
        router_weights=router_override,
        hard_routing=hard_routing, hard_routing_tau=hard_routing_tau,
    )

    recon_loss = F.mse_loss(x_recon, x)
    K = args.num_charts
    entropy_loss = math.log(K) - compute_routing_entropy(enc_w)
    consistency_loss = model.compute_consistency_loss(enc_w, dec_w)
    diversity_loss = compute_diversity_loss(enc_w, K)
    uniformity_loss = compute_hyperbolic_uniformity_loss(z_geo)
    radial_cal_loss = compute_radial_calibration_loss(z_geo, enc_w, K)

    codebook = model.encoder.codebook
    cb_spread_loss = compute_codebook_spread_loss(codebook)
    cb_center_loss = compute_codebook_centering_loss(codebook)

    chart_collapse_loss = compute_chart_collapse_penalty(enc_w, K)
    code_collapse_loss = compute_code_collapse_penalty(v_local, codebook, enc_w)

    window_loss, window_info = compute_window_loss(enc_w, K)

    current_jump_weight = get_jump_weight_schedule(
        epoch,
        warmup_end=args.w_jump_warmup,
        ramp_end=args.w_jump_ramp_end,
        final_weight=args.w_jump,
    )
    jump_loss, _jump_info = compute_jump_consistency_loss(z_n_all, enc_w, jump_op)

    total = (
        args.w_recon * recon_loss
        + args.w_vq * vq_loss
        + args.w_entropy * entropy_loss
        + args.w_consistency * consistency_loss
        + args.w_diversity * diversity_loss
        + args.w_uniformity * uniformity_loss
        + args.w_radial_cal * radial_cal_loss
        + args.w_codebook_spread * cb_spread_loss
        + args.w_codebook_center * cb_center_loss
        + args.w_chart_collapse * chart_collapse_loss
        + args.w_code_collapse * code_collapse_loss
        + args.w_window * window_loss
        + current_jump_weight * jump_loss
    )

    metrics = {
        "recon": recon_loss.item(),
        "vq": vq_loss.item(),
        "entropy": entropy_loss.item(),
        "consistency": consistency_loss.item(),
        "diversity": diversity_loss.item(),
        "uniformity": uniformity_loss.item(),
        "radial_cal": radial_cal_loss.item(),
        "codebook_spread": cb_spread_loss.item(),
        "codebook_center": cb_center_loss.item(),
        "chart_collapse": chart_collapse_loss.item(),
        "code_collapse": code_collapse_loss.item(),
        "window": window_loss.item(),
        "jump": jump_loss.item(),
        "I_XK": float(window_info.get("I_XK", 0.0)),
        "H_K": float(window_info.get("H_K", 0.0)),
        "H_K_given_X": float(window_info.get("H_K_given_X", 0.0)),
        "jump_weight": current_jump_weight,
        "total": total.item(),
    }
    return total, metrics, z_geo, enc_w, K_chart


# ── Eval pass (chart usage, perplexity, mean_r) ──────────────────


def _eval_pass(
    model: TopoEncoderPrimitives,
    loader: DataLoader,
    K: int,
    device: torch.device,
) -> tuple[np.ndarray, float, float]:
    """Compute chart usage, perplexity, and mean radius."""
    model.eval()
    all_charts: list[torch.Tensor] = []
    all_radii: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["feature"].to(device)
            K_ch, *_, z_g, _, _, _, _, _ = model.encoder(x)
            all_charts.append(K_ch.cpu())
            all_radii.append(z_g.cpu().norm(dim=-1))
    charts_np = torch.cat(all_charts).numpy()
    radii_np = torch.cat(all_radii).numpy()

    usage = np.zeros(K)
    for c in charts_np:
        usage[int(c)] += 1
    usage /= usage.sum() + 1e-8

    perplexity = float(np.exp(-np.sum(usage * np.log(usage + 1e-8))))
    mean_r = float(radii_np.mean())
    return usage, perplexity, mean_r


# ── Minimum length scale measurement ─────────────────────────────


@torch.no_grad()
def _measure_min_length(
    model: TopoEncoderPrimitives,
    seq_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    """Measure average consecutive geodesic distance d_hyp(z_t, z_{t+1}).

    This gives the *action resolution* — the typical geodesic distance
    the latent state moves in one time step.  Used as the minimum
    resolvable length scale ℓ_min for CFL-derived force / velocity /
    noise bounds.
    """
    model.eval()
    dists: list[torch.Tensor] = []
    for i, batch in enumerate(seq_loader):
        if i >= max_batches:
            break
        features = batch["features"].to(device)  # [B, H, D_feat]
        B, H, D_feat = features.shape
        flat = features.reshape(B * H, D_feat)
        _, _, _, _, _, z_flat, *_ = model.encoder(flat)
        z_seq = z_flat.reshape(B, H, -1)  # [B, H, D]
        # Consecutive geodesic distances
        for t in range(H - 1):
            d = hyperbolic_distance(z_seq[:, t, :], z_seq[:, t + 1, :])  # [B]
            dists.append(d)
    all_dists = torch.cat(dists)
    median_dist = float(all_dists.median().item())
    mean_dist = float(all_dists.mean().item())
    print(f"  ℓ_min measurement: median={median_dist:.4f}  mean={mean_dist:.4f}  "
          f"(from {all_dists.numel()} pairs)")
    return median_dist


def _update_world_model_min_length(
    world_model: GeometricWorldModel,
    min_length: float,
) -> None:
    """Update world model CFL bounds from a measured minimum length scale."""
    import math as _math

    world_model.min_length = min_length
    dt = world_model.dt
    gamma = world_model.gamma
    c2 = world_model.c2
    D = world_model.latent_dim

    world_model.V_alg = min_length / dt
    world_model.F_max = 2.0 * gamma * min_length / dt
    denom = max(c2, 1e-12) * _math.sqrt(D) * dt
    world_model.cf_max = max(2.0 * min_length / denom, 2.0)

    print(f"  CFL bounds from ℓ_min={min_length:.4f}:")
    print(f"    F_max  = {world_model.F_max:.2f}  (force squashing)")
    print(f"    V_alg  = {world_model.V_alg:.2f}  (velocity squashing)")
    print(f"    cf_max = {world_model.cf_max:.2f}  (thermostat noise cap)")


# ── Phase 1: Encoder warmup ──────────────────────────────────────


def _run_phase1(
    model: TopoEncoderPrimitives,
    jump_op: FactorizedJumpOperator,
    single_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    global_epoch_offset: int = 0,
    total_epochs_all_phases: int = 1,
) -> dict[str, float]:
    """Phase 1: Train encoder + jump_op on single frames (13 losses)."""
    print("\n" + "=" * 60)
    print("Phase 1: Encoder warmup")
    print("=" * 60)

    all_params = list(model.parameters()) + list(jump_op.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr)
    scheduler = None
    if args.use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.phase1_epochs, eta_min=1e-6)

    K = args.num_charts
    last_metrics: dict[str, float] = {}
    for epoch in tqdm(range(args.phase1_epochs), desc="Phase 1"):
        model.train()
        jump_op.train()
        acc = _init_encoder_accumulators()
        n_batches = 0
        current_tau = _get_hard_routing_tau(
            args, global_epoch_offset + epoch, total_epochs_all_phases,
        )

        for batch in single_loader:
            x = batch["feature"].to(device)

            total, metrics, _, _, _ = _compute_encoder_losses(
                x, model, jump_op, args, epoch,
                hard_routing=args.hard_routing,
                hard_routing_tau=current_tau,
            )

            optimizer.zero_grad()
            total.backward()
            grad_norm = compute_grad_norm(all_params)
            param_norm = compute_param_norm(all_params)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
            optimizer.step()

            current_lr = optimizer.param_groups[0]["lr"]
            update_ratio = (
                current_lr * grad_norm / (param_norm + 1e-12)
                if param_norm > 0 else 0.0
            )

            acc["total"] += metrics["total"]
            for k in ENCODER_LOSS_KEYS:
                acc[k] += metrics[k]
            acc["I_XK"] += metrics["I_XK"]
            acc["H_K"] += metrics["H_K"]
            acc["H_K_given_X"] += metrics["H_K_given_X"]
            acc["grad_norm"] += grad_norm
            acc["param_norm"] += param_norm
            acc["update_ratio"] += update_ratio
            acc["lr"] += current_lr
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        for k in acc:
            acc[k] /= max(n_batches, 1)

        usage, perplexity, mean_r = _eval_pass(model, single_loader, K, device)
        active = int((usage > 0.01).sum())

        should_log = (epoch % args.log_every == 0) or (epoch == args.phase1_epochs - 1)
        if should_log:
            print(
                f"P1 E{epoch:5d} | Loss: {acc['total']:.4f} "
                f"| LR: {acc['lr']:.2e}"
            )
            print(f"  Usage: {np.array2string(usage, precision=2, separator=', ')}")
            print(
                f"  Core: recon={acc['recon']:.3f} vq={acc['vq']:.3f} "
                f"entropy={acc['entropy']:.3f} consist={acc['consistency']:.3f} "
                f"div={acc['diversity']:.3f}"
            )
            print(
                f"  Geo: unif={acc['uniformity']:.3f} "
                f"rad_cal={acc['radial_cal']:.3f} "
                f"cb_spread={acc['codebook_spread']:.3f} "
                f"cb_center={acc['codebook_center']:.3f}"
            )
            print(
                f"  Collapse: chart={acc['chart_collapse']:.4f} "
                f"code={acc['code_collapse']:.4f}"
            )
            print(
                f"  Window: {acc['window']:.3f} "
                f"(I_XK={acc['I_XK']:.3f} H_K={acc['H_K']:.3f} "
                f"H_K|X={acc['H_K_given_X']:.3f})"
            )
            print(
                f"  Jump: {acc['jump']:.3f} "
                f"(lambda={metrics.get('jump_weight', 0.0):.3f})"
            )
            print(
                f"  Train: grad={acc['grad_norm']:.2e} "
                f"upd_ratio={acc['update_ratio']:.2e} lr={acc['lr']:.2e}"
            )
            print(
                f"  Metrics: perplexity={perplexity:.2f}/{K} "
                f"active={active}/{K} mean_r={mean_r:.3f}"
            )
            print("-" * 60)

        should_save = (
            (epoch > 0 and epoch % args.save_every == 0)
            or epoch == args.phase1_epochs - 1
        )
        if should_save:
            _save_checkpoint(args, model, jump_op, None, optimizer, scheduler, epoch, 1, acc)

        last_metrics = acc

    return last_metrics


# ── Phase 2: World model warmup ──────────────────────────────────


def _run_phase2(
    model: TopoEncoderPrimitives,
    jump_op: FactorizedJumpOperator,
    world_model: GeometricWorldModel,
    seq_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    global_epoch_offset: int = 0,
    total_epochs_all_phases: int = 1,
) -> dict[str, float]:
    """Phase 2: Train world model with frozen encoder."""
    print("\n" + "=" * 60)
    print("Phase 2: World model warmup (encoder frozen)")
    print("=" * 60)

    # Freeze encoder
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in jump_op.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(world_model.parameters(), lr=args.lr_wm)
    scheduler = None
    if args.use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.phase2_epochs, eta_min=1e-6)

    # Build config namespace for compute_phase2_loss
    config_ns = SimpleNamespace(
        w_geodesic=args.w_geodesic,
        w_chart_transition=args.w_chart_transition,
        w_momentum_reg=args.w_momentum_reg,
        w_energy_conservation=args.w_energy_conservation,
        w_jump_dynamics=args.w_jump_dynamics,
        w_screened_poisson=args.w_screened_poisson,
        wm_screening_kappa=args.wm_screening_kappa,
        w_hodge=args.w_hodge,
    )

    K = args.num_charts
    last_metrics: dict[str, float] = {}

    for epoch in tqdm(range(args.phase2_epochs), desc="Phase 2"):
        world_model.train()
        acc = _init_dynamics_accumulators()
        n_batches = 0
        current_tau = _get_hard_routing_tau(
            args, global_epoch_offset + epoch, total_epochs_all_phases,
        )

        for batch in seq_loader:
            features = batch["features"].to(device)  # [B, H, D_feat]
            actions = batch["actions"].to(device)     # [B, H, A]
            B, H, D_feat = features.shape

            # Encode all frames in one batched pass (frozen encoder)
            with torch.no_grad():
                flat = features.reshape(B * H, D_feat)
                K_flat, _, _, _, rw_flat, z_flat, *_ = model.encoder(
                    flat,
                    hard_routing=args.hard_routing,
                    hard_routing_tau=current_tau,
                )
                z_all = z_flat.reshape(B, H, -1)
                K_all = K_flat.reshape(B, H)
                rw_0 = rw_flat[:B]  # router weights for first frame

            z_0 = z_all[:, 0, :]
            pred_actions = actions[:, :-1, :]
            z_targets = z_all[:, 1:, :]
            chart_targets = K_all[:, 1:]

            wm_output = world_model(z_0, pred_actions, rw_0)
            loss, metrics = compute_phase2_loss(wm_output, z_targets, chart_targets, config_ns)
            wm_diag = _wm_diagnostics(wm_output)

            optimizer.zero_grad()
            loss.backward()
            wm_params = list(world_model.parameters())
            grad_norm = compute_grad_norm(wm_params)
            param_norm = compute_param_norm(wm_params)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(wm_params, args.grad_clip)
            optimizer.step()

            current_lr = optimizer.param_groups[0]["lr"]
            update_ratio = (
                current_lr * grad_norm / (param_norm + 1e-12)
                if param_norm > 0 else 0.0
            )

            acc["total"] += metrics["total"]
            for k in DYNAMICS_LOSS_KEYS:
                if k in metrics:
                    acc[k] += metrics[k]
            for k in WM_DIAG_KEYS:
                acc[k] += wm_diag[k]
            acc["grad_norm"] += grad_norm
            acc["param_norm"] += param_norm
            acc["update_ratio"] += update_ratio
            acc["lr"] += current_lr
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        for k in acc:
            acc[k] /= max(n_batches, 1)

        # Chart usage from encoded frames (already computed, just aggregate last batch)
        usage, perplexity, active = _chart_stats_from_tensor(K_all, K)
        mean_r = z_all.norm(dim=-1).mean().item()

        should_log = (epoch % args.log_every == 0) or (epoch == args.phase2_epochs - 1)
        if should_log:
            print(
                f"P2 E{epoch:5d} | Loss: {acc['total']:.4f} "
                f"| LR: {acc['lr']:.2e}"
            )
            print(f"  Usage: {np.array2string(usage, precision=2, separator=', ')}")
            print(
                f"  Dynamics: geo={acc['geodesic']:.4f} "
                f"chart={acc['chart_transition']:.4f} "
                f"mom_reg={acc['momentum_reg']:.4f}"
            )
            print(
                f"  Energy: conservation={acc['energy_conservation']:.4f} "
                f"jump_dyn={acc['jump_dynamics']:.4f}"
            )
            print(
                f"  WM diag: |p|={acc['mean_momentum']:.4f} "
                f"lambda={acc['mean_jump_rate']:.4f} "
                f"jump%={acc['jump_fraction']:.4f} "
                f"phi_eff={acc['mean_phi_eff']:.4f}"
            )
            print(
                f"  Hodge: cons={acc['hodge_cons']:.4f} "
                f"sol={acc['hodge_sol']:.4f} "
                f"harm={acc['hodge_harm']:.4f}"
            )
            print(
                f"  Train: grad={acc['grad_norm']:.2e} "
                f"upd_ratio={acc['update_ratio']:.2e} "
                f"param={acc['param_norm']:.2e}"
            )
            print(
                f"  Metrics: perplexity={perplexity:.2f}/{K} "
                f"active={active}/{K} mean_r={mean_r:.3f}"
            )
            print("-" * 60)

        should_save = (
            (epoch > 0 and epoch % args.save_every == 0)
            or epoch == args.phase2_epochs - 1
        )
        if should_save:
            _save_checkpoint(
                args, model, jump_op, world_model, optimizer, scheduler, epoch, 2, acc,
            )

        last_metrics = acc

    # Unfreeze encoder for Phase 3
    for p in model.parameters():
        p.requires_grad_(True)
    for p in jump_op.parameters():
        p.requires_grad_(True)

    return last_metrics


# ── Phase 3: Joint fine-tuning ────────────────────────────────────


def _run_phase3(
    model: TopoEncoderPrimitives,
    jump_op: FactorizedJumpOperator,
    world_model: GeometricWorldModel,
    seq_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    global_epoch_offset: int = 0,
    total_epochs_all_phases: int = 1,
) -> dict[str, float]:
    """Phase 3: Joint fine-tuning of encoder + world model."""
    print("\n" + "=" * 60)
    print("Phase 3: Joint fine-tuning")
    print("=" * 60)

    # Two param groups with different LRs
    optimizer = torch.optim.Adam([
        {
            "params": list(model.parameters()) + list(jump_op.parameters()),
            "lr": args.lr_joint_encoder,
        },
        {
            "params": list(world_model.parameters()),
            "lr": args.lr_joint_wm,
        },
    ])
    scheduler = None
    if args.use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.phase3_epochs, eta_min=1e-6)

    # Config namespace for dynamics losses
    config_ns = SimpleNamespace(
        w_geodesic=args.w_geodesic,
        w_chart_transition=args.w_chart_transition,
        w_momentum_reg=args.w_momentum_reg,
        w_energy_conservation=args.w_energy_conservation,
        w_jump_dynamics=args.w_jump_dynamics,
        w_screened_poisson=args.w_screened_poisson,
        wm_screening_kappa=args.wm_screening_kappa,
        w_hodge=args.w_hodge,
    )

    K = args.num_charts
    last_metrics: dict[str, float] = {}

    for epoch in tqdm(range(args.phase3_epochs), desc="Phase 3"):
        model.train()
        jump_op.train()
        world_model.train()
        acc = _init_joint_accumulators()
        n_batches = 0
        current_tau = _get_hard_routing_tau(
            args, global_epoch_offset + epoch, total_epochs_all_phases,
        )

        for batch in seq_loader:
            features = batch["features"].to(device)  # [B, H, D_feat]
            actions = batch["actions"].to(device)     # [B, H, A]
            B, H, D_feat = features.shape

            # Frame 0: full encoder losses + reuse outputs (1 encoder call)
            enc_loss, enc_metrics_0, z_geo_0, enc_w_0, K_ch_0 = \
                _compute_encoder_losses(
                    features[:, 0, :], model, jump_op, args, epoch,
                    hard_routing=args.hard_routing,
                    hard_routing_tau=current_tau,
                )

            # Frames 1..H-1: batched encoding (1 encoder call)
            if H > 1:
                rest = features[:, 1:, :].reshape(B * (H - 1), D_feat)
                K_rest, _, _, _, rw_rest, z_rest, *_ = model.encoder(
                    rest,
                    hard_routing=args.hard_routing,
                    hard_routing_tau=current_tau,
                )
                z_geo_rest = z_rest.reshape(B, H - 1, -1)
                K_rest = K_rest.reshape(B, H - 1)
                z_all = torch.cat([z_geo_0.unsqueeze(1), z_geo_rest], dim=1)
                K_all = torch.cat([K_ch_0.unsqueeze(1), K_rest], dim=1)
            else:
                z_all = z_geo_0.unsqueeze(1)
                K_all = K_ch_0.unsqueeze(1)

            # World model rollout
            z_0 = z_all[:, 0, :]
            rw_0 = enc_w_0
            pred_actions = actions[:, :-1, :]
            z_targets = z_all[:, 1:, :].detach()
            chart_targets = K_all[:, 1:].detach()

            wm_output = world_model(z_0, pred_actions, rw_0)
            dyn_loss, dyn_metrics = compute_phase2_loss(
                wm_output, z_targets, chart_targets, config_ns,
            )
            wm_diag = _wm_diagnostics(wm_output)

            # Combined loss
            total = (
                args.phase3_encoder_scale * enc_loss
                + args.phase3_dynamics_scale * dyn_loss
            )

            optimizer.zero_grad()
            total.backward()
            all_params = (
                list(model.parameters())
                + list(jump_op.parameters())
                + list(world_model.parameters())
            )
            grad_norm = compute_grad_norm(all_params)
            param_norm = compute_param_norm(all_params)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
            optimizer.step()

            current_lr = optimizer.param_groups[0]["lr"]
            update_ratio = (
                current_lr * grad_norm / (param_norm + 1e-12)
                if param_norm > 0 else 0.0
            )

            acc["total"] += total.item()
            acc["enc_total"] += enc_metrics_0["total"]
            acc["dyn_total"] += dyn_metrics["total"]
            for k in ENCODER_LOSS_KEYS:
                acc[f"enc/{k}"] += enc_metrics_0[k]
            for k in DYNAMICS_LOSS_KEYS:
                if k in dyn_metrics:
                    acc[f"dyn/{k}"] += dyn_metrics[k]
            for k in WM_DIAG_KEYS:
                acc[f"wm/{k}"] += wm_diag[k]
            acc["I_XK"] += enc_metrics_0.get("I_XK", 0.0)
            acc["H_K"] += enc_metrics_0.get("H_K", 0.0)
            acc["H_K_given_X"] += enc_metrics_0.get("H_K_given_X", 0.0)
            acc["grad_norm"] += grad_norm
            acc["param_norm"] += param_norm
            acc["update_ratio"] += update_ratio
            acc["lr"] += current_lr
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        for k in acc:
            acc[k] /= max(n_batches, 1)

        # Chart stats from training-batch encodings (last batch)
        usage, perplexity, active = _chart_stats_from_tensor(K_all, K)
        mean_r = z_all.detach().norm(dim=-1).mean().item()

        should_log = (epoch % args.log_every == 0) or (epoch == args.phase3_epochs - 1)
        if should_log:
            print(
                f"P3 E{epoch:5d} | Loss: {acc['total']:.4f} "
                f"(enc={acc['enc_total']:.4f} dyn={acc['dyn_total']:.4f}) "
                f"| LR: {acc['lr']:.2e}"
            )
            print(f"  Usage: {np.array2string(usage, precision=2, separator=', ')}")
            print(
                f"  Core: recon={acc['enc/recon']:.3f} "
                f"vq={acc['enc/vq']:.3f} "
                f"entropy={acc['enc/entropy']:.3f} "
                f"consist={acc['enc/consistency']:.3f} "
                f"div={acc['enc/diversity']:.3f}"
            )
            print(
                f"  Geo: unif={acc['enc/uniformity']:.3f} "
                f"rad_cal={acc['enc/radial_cal']:.3f} "
                f"cb_spread={acc['enc/codebook_spread']:.3f} "
                f"cb_center={acc['enc/codebook_center']:.3f}"
            )
            print(
                f"  Collapse: chart={acc['enc/chart_collapse']:.4f} "
                f"code={acc['enc/code_collapse']:.4f}"
            )
            print(
                f"  Window: {acc['enc/window']:.3f} "
                f"(I_XK={acc['I_XK']:.3f} H_K={acc['H_K']:.3f} "
                f"H_K|X={acc['H_K_given_X']:.3f})"
            )
            print(
                f"  Jump: {acc['enc/jump']:.3f} "
                f"(lambda={enc_metrics_0.get('jump_weight', 0.0):.3f})"
            )
            print(
                f"  Dynamics: geo={acc['dyn/geodesic']:.4f} "
                f"chart={acc['dyn/chart_transition']:.4f} "
                f"mom_reg={acc['dyn/momentum_reg']:.4f}"
            )
            print(
                f"  Energy: conservation={acc['dyn/energy_conservation']:.4f} "
                f"jump_dyn={acc['dyn/jump_dynamics']:.4f}"
            )
            print(
                f"  WM diag: |p|={acc['wm/mean_momentum']:.4f} "
                f"lambda={acc['wm/mean_jump_rate']:.4f} "
                f"jump%={acc['wm/jump_fraction']:.4f} "
                f"phi_eff={acc['wm/mean_phi_eff']:.4f}"
            )
            print(
                f"  Hodge: cons={acc['wm/hodge_cons']:.4f} "
                f"sol={acc['wm/hodge_sol']:.4f} "
                f"harm={acc['wm/hodge_harm']:.4f}"
            )
            print(
                f"  Train: grad={acc['grad_norm']:.2e} "
                f"upd_ratio={acc['update_ratio']:.2e} lr={acc['lr']:.2e}"
            )
            print(
                f"  Metrics: perplexity={perplexity:.2f}/{K} "
                f"active={active}/{K} mean_r={mean_r:.3f}"
            )
            print("-" * 60)

        should_save = (
            (epoch > 0 and epoch % args.save_every == 0)
            or epoch == args.phase3_epochs - 1
        )
        if should_save:
            _save_checkpoint(
                args, model, jump_op, world_model, optimizer, scheduler, epoch, 3, acc,
            )

        last_metrics = acc

    return last_metrics


# ── Checkpoint helpers ────────────────────────────────────────────


def _save_checkpoint(
    args: argparse.Namespace,
    model: TopoEncoderPrimitives,
    jump_op: FactorizedJumpOperator,
    world_model: GeometricWorldModel | None,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR | None,
    epoch: int,
    phase: int,
    metrics: dict[str, float],
) -> None:
    ckpt_path = os.path.join(args.output_dir, f"p{phase}_epoch_{epoch:05d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "phase": phase,
            "model": model.state_dict(),
            "jump_op": jump_op.state_dict(),
            "world_model": world_model.state_dict() if world_model is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "args": vars(args),
            "metrics": metrics,
        },
        ckpt_path,
    )
    print(f"  Saved checkpoint: {ckpt_path}")


# ── Final diagnostics ────────────────────────────────────────────


def _run_diagnostics(
    model: TopoEncoderPrimitives,
    single_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    had_world_model: bool,
    final_dyn_metrics: dict[str, float] | None,
) -> None:
    """Run end-of-training diagnostics."""
    print("\n=== Final Diagnostics ===")
    model.eval()

    K = args.num_charts
    all_features: list[torch.Tensor] = []
    all_ep_ids: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in single_loader:
            all_features.append(batch["feature"])
            all_ep_ids.append(batch["episode_id"])
    all_features_t = torch.cat(all_features)
    all_ep_ids_t = torch.cat(all_ep_ids)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    from fragile.learning.vla.visualize import full_diagnostic, plot_chart_transitions

    config_ns = SimpleNamespace(num_charts=K)
    full_diagnostic(model, all_features_t, config_ns, save_dir=args.output_dir)

    with torch.no_grad():
        K_chart_all, *_ = model.encoder(all_features_t.to(device))

    fig = plot_chart_transitions(K_chart_all.cpu(), all_ep_ids_t)
    fig.savefig(
        os.path.join(args.output_dir, "chart_transitions.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    # World model dynamics summary
    if had_world_model and final_dyn_metrics is not None:
        print("\n--- World Model Summary ---")
        for k in DYNAMICS_LOSS_KEYS:
            # Try both prefixed and unprefixed keys
            val = final_dyn_metrics.get(f"dyn/{k}", final_dyn_metrics.get(k, 0.0))
            print(f"  {k}: {val:.4f}")

    print(f"\nAll outputs saved to {args.output_dir}")


# ── Main training function ────────────────────────────────────────


def train_joint(args: argparse.Namespace) -> None:  # noqa: C901
    """Run the 3-phase joint encoder + world model training."""
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Data loaders ──────────────────────────────────────────
    single_loader = None
    seq_loader = None

    if args.phase1_epochs > 0:
        single_ds = VLAFeatureDataset(args.feature_cache_dir, sequence_length=1)
        single_loader = DataLoader(
            single_ds, batch_size=args.batch_size, shuffle=True,
            drop_last=False, num_workers=0,
        )
        input_dim = single_ds[0]["feature"].shape[0]
        print(f"Single-frame dataset: {len(single_ds)} frames, {len(single_loader)} batches")
    else:
        # Need input_dim from a probe
        probe_ds = VLAFeatureDataset(args.feature_cache_dir, sequence_length=1)
        input_dim = probe_ds[0]["feature"].shape[0]
        # Still build single_loader for diagnostics
        single_loader = DataLoader(
            probe_ds, batch_size=args.batch_size, shuffle=False,
            drop_last=False, num_workers=0,
        )
        print(f"Skipping Phase 1 (single-frame loader built for diagnostics only)")

    if args.phase2_epochs > 0 or args.phase3_epochs > 0:
        seq_ds = VLAFeatureDataset(args.feature_cache_dir, sequence_length=args.sequence_length)
        seq_loader = DataLoader(
            seq_ds, batch_size=args.batch_size, shuffle=True,
            drop_last=True, num_workers=0,
        )
        print(f"Sequence dataset: {len(seq_ds)} windows (seq_len={args.sequence_length}), "
              f"{len(seq_loader)} batches")

    print(f"Feature dim: {input_dim}")

    # ── Models ────────────────────────────────────────────────
    K = args.num_charts
    model = TopoEncoderPrimitives(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_charts=K,
        codes_per_chart=args.codes_per_chart,
        covariant_attn=True,
        conv_backbone=False,
        film_conditioning=True,
    ).to(device)

    jump_op = FactorizedJumpOperator(
        num_charts=K,
        latent_dim=args.latent_dim,
    ).to(device)

    world_model = None
    if args.phase2_epochs > 0 or args.phase3_epochs > 0:
        _risk_alpha = getattr(args, "wm_risk_metric_alpha", None)
        if _risk_alpha is None:
            _risk_alpha = 0.0
        world_model = GeometricWorldModel(
            latent_dim=args.latent_dim,
            action_dim=args.action_dim,
            num_charts=K,
            d_model=args.wm_d_model,
            hidden_dim=args.wm_hidden_dim,
            dt=args.wm_dt,
            gamma_friction=args.wm_gamma_friction,
            T_c=args.wm_T_c,
            alpha_potential=args.wm_alpha_potential,
            beta_curl=args.wm_beta_curl,
            gamma_risk=args.wm_gamma_risk,
            use_boris=args.wm_use_boris,
            use_jump=args.wm_use_jump,
            jump_rate_hidden=args.wm_jump_rate_hidden,
            min_length=max(args.wm_min_length, 0.0),  # -1 (auto) deferred until after P1
            risk_metric_alpha=_risk_alpha,
        ).to(device)

    # ── Parameter breakdown ──────────────────────────────────
    n_enc = count_parameters(model.encoder)
    n_dec = count_parameters(model.decoder)
    n_jump = count_parameters(jump_op)
    n_total = n_enc + n_dec + n_jump
    print(f"  Encoder:  {n_enc:>10,} params")
    print(f"  Decoder:  {n_dec:>10,} params")
    print(f"  Jump op:  {n_jump:>10,} params")
    if world_model is not None:
        n_wm = count_parameters(world_model)
        n_total += n_wm
        print(f"  World model breakdown:")
        for name, child in world_model.named_children():
            nc = count_parameters(child)
            if nc > 0:
                print(f"    {name:25s} {nc:>8,} params")
        print(f"  World model total: {n_wm:>7,} params")
        if world_model.min_length > 0:
            print(f"  CFL bounds (ℓ_min={world_model.min_length:.4f}):")
            print(f"    F_max={world_model.F_max:.2f}  V_alg={world_model.V_alg:.2f}  "
                  f"cf_max={world_model.cf_max:.2f}")
        elif args.wm_min_length < 0:
            print(f"  CFL bounds: auto-measure after Phase 1")
        else:
            print(f"  CFL bounds: disabled (no squashing)")
    print(f"  TOTAL: {n_total:>13,} params")

    # ── Resume ────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        jump_op.load_state_dict(ckpt["jump_op"])
        if ckpt.get("world_model") is not None and world_model is not None:
            world_model.load_state_dict(ckpt["world_model"])
        print(f"Resumed from {args.resume} (phase {ckpt.get('phase', '?')}, "
              f"epoch {ckpt.get('epoch', '?')})")

    # ── Output dir ────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Run phases ────────────────────────────────────────────
    last_dyn_metrics = None
    had_world_model = False
    total_epochs_all_phases = args.phase1_epochs + args.phase2_epochs + args.phase3_epochs

    if args.phase1_epochs > 0:
        _run_phase1(
            model, jump_op, single_loader, args, device,
            global_epoch_offset=0,
            total_epochs_all_phases=total_epochs_all_phases,
        )

    # Auto-measure ℓ_min from encoder if requested (--wm-min-length -1)
    if args.wm_min_length < 0 and world_model is not None and seq_loader is not None:
        print("\n  Auto-measuring minimum length scale from encoder...")
        ell_min = _measure_min_length(model, seq_loader, device)
        _update_world_model_min_length(world_model, ell_min)

    if args.phase2_epochs > 0:
        assert world_model is not None
        assert seq_loader is not None
        last_dyn_metrics = _run_phase2(
            model, jump_op, world_model, seq_loader, args, device,
            global_epoch_offset=args.phase1_epochs,
            total_epochs_all_phases=total_epochs_all_phases,
        )
        had_world_model = True

    if args.phase3_epochs > 0:
        assert world_model is not None
        assert seq_loader is not None
        last_dyn_metrics = _run_phase3(
            model, jump_op, world_model, seq_loader, args, device,
            global_epoch_offset=args.phase1_epochs + args.phase2_epochs,
            total_epochs_all_phases=total_epochs_all_phases,
        )
        had_world_model = True

    # ── Final checkpoint ──────────────────────────────────────
    final_phase = 3 if args.phase3_epochs > 0 else (2 if args.phase2_epochs > 0 else 1)
    final_path = os.path.join(args.output_dir, "checkpoint_final.pt")
    torch.save(
        {
            "epoch": -1,
            "phase": final_phase,
            "model": model.state_dict(),
            "jump_op": jump_op.state_dict(),
            "world_model": world_model.state_dict() if world_model is not None else None,
            "optimizer": None,
            "scheduler": None,
            "args": vars(args),
            "metrics": last_dyn_metrics,
        },
        final_path,
    )
    print(f"\nFinal checkpoint saved to {final_path}")

    # ── Diagnostics ───────────────────────────────────────────
    _run_diagnostics(
        model, single_loader, args, device, had_world_model, last_dyn_metrics,
    )


# ── CLI ───────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for joint encoder + world model training."""
    p = argparse.ArgumentParser(
        description="Joint encoder + world model training (3-phase)",
    )

    # Data / output
    p.add_argument("--feature-cache-dir", default="outputs/vla/features")
    p.add_argument("--output-dir", default="outputs/vla/joint")

    # Architecture
    p.add_argument("--latent-dim", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-charts", type=int, default=16)
    p.add_argument("--codes-per-chart", type=int, default=64)
    p.add_argument("--action-dim", type=int, default=6)
    p.add_argument("--sequence-length", type=int, default=8)
    p.add_argument("--hard-routing", action="store_true", default=False,
                   help="Use Gumbel-softmax hard routing (one-hot forward, ST gradients)")
    p.add_argument("--hard-routing-tau", type=float, default=1.0,
                   help="Starting temperature for Gumbel-softmax hard routing")
    p.add_argument("--hard-routing-tau-end", type=float, default=None,
                   help="Final tau after annealing (None = no annealing, use constant tau)")
    p.add_argument("--hard-routing-tau-anneal-epochs", type=int, default=None,
                   help="Anneal tau linearly over this many epochs (None = total epochs)")

    # Phase epochs (0 = skip)
    p.add_argument("--phase1-epochs", type=int, default=100)
    p.add_argument("--phase2-epochs", type=int, default=50)
    p.add_argument("--phase3-epochs", type=int, default=50)

    # Learning rates
    p.add_argument("--lr", type=float, default=1e-3, help="Phase 1 encoder LR")
    p.add_argument("--lr-wm", type=float, default=1e-3, help="Phase 2 world model LR")
    p.add_argument("--lr-joint-encoder", type=float, default=1e-4, help="Phase 3 encoder LR")
    p.add_argument("--lr-joint-wm", type=float, default=1e-3, help="Phase 3 world model LR")

    # Training
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--use-scheduler", action="store_true")

    # World model params
    p.add_argument("--wm-hidden-dim", type=int, default=None,
                   help="World model MLP width (defaults to --hidden-dim)")
    p.add_argument("--wm-dt", type=float, default=0.01)
    p.add_argument("--wm-gamma-friction", type=float, default=1.0)
    p.add_argument("--wm-T-c", type=float, default=0.1)
    p.add_argument("--wm-alpha-potential", type=float, default=0.5)
    p.add_argument("--wm-beta-curl", type=float, default=0.1)
    p.add_argument("--wm-gamma-risk", type=float, default=0.01)
    p.add_argument("--wm-use-boris", action="store_true", default=True)
    p.add_argument("--wm-no-boris", dest="wm_use_boris", action="store_false")
    p.add_argument("--wm-use-jump", action="store_true", default=True)
    p.add_argument("--wm-no-jump", dest="wm_use_jump", action="store_false")
    p.add_argument("--wm-jump-rate-hidden", type=int, default=64)
    p.add_argument("--wm-min-length", type=float, default=0.03,
                   help="Minimum geodesic length scale (derives F_max, V_alg, cf_max; "
                        "0=off, -1=auto-measure from encoder after Phase 1)")
    p.add_argument("--wm-d-model", type=int, default=128,
                   help="CovariantAttention width for world model")
    p.add_argument("--wm-risk-metric-alpha", type=float, default=None,
                   help="Risk-metric coupling alpha; None uses config default (0=off)")

    # Phase 3 scaling
    p.add_argument("--phase3-encoder-scale", type=float, default=0.1)
    p.add_argument("--phase3-dynamics-scale", type=float, default=1.0)

    # Encoder loss weights (same defaults as train_unsupervised)
    p.add_argument("--w-recon", type=float, default=1.0)
    p.add_argument("--w-vq", type=float, default=1.0)
    p.add_argument("--w-entropy", type=float, default=0.1)
    p.add_argument("--w-consistency", type=float, default=0.1)
    p.add_argument("--w-diversity", type=float, default=0.1)
    p.add_argument("--w-uniformity", type=float, default=0.1)
    p.add_argument("--w-radial-cal", type=float, default=0.1)
    p.add_argument("--w-codebook-spread", type=float, default=0.05)
    p.add_argument("--w-codebook-center", type=float, default=0.01)
    p.add_argument("--w-chart-collapse", type=float, default=1.0)
    p.add_argument("--w-code-collapse", type=float, default=0.5)
    p.add_argument("--w-window", type=float, default=0.5)
    p.add_argument("--w-jump", type=float, default=0.1)
    p.add_argument("--w-jump-warmup", type=int, default=20)
    p.add_argument("--w-jump-ramp-end", type=int, default=50)

    # Dynamics loss weights
    p.add_argument("--w-geodesic", type=float, default=1.0)
    p.add_argument("--w-chart-transition", type=float, default=0.5)
    p.add_argument("--w-momentum-reg", type=float, default=0.01)
    p.add_argument("--w-energy-conservation", type=float, default=0.01)
    p.add_argument("--w-jump-dynamics", type=float, default=0.1)
    p.add_argument("--w-screened-poisson", type=float, default=0.0,
                   help="Screened Poisson PDE residual weight; 0 = disabled")
    p.add_argument("--wm-screening-kappa", type=float, default=1.0,
                   help="Screening mass kappa for screened Poisson loss")
    p.add_argument("--w-hodge", type=float, default=0.0,
                   help="Hodge consistency loss weight; 0 = disabled")

    # Logging / saving / resume
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume", default="", help="Checkpoint path to resume from")
    p.add_argument("--device", default="auto")

    args = p.parse_args()
    if args.wm_hidden_dim is None:
        args.wm_hidden_dim = args.hidden_dim
    train_joint(args)


if __name__ == "__main__":
    main()
