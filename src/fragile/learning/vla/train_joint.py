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
from fragile.learning.vla.extract_features import VLAFeatureDataset
from fragile.learning.vla.losses import compute_phase2_loss
from fragile.learning.vla.world_model import GeometricWorldModel

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
    "energy_conservation", "jump_dynamics",
]

INFO_KEYS = [
    "I_XK", "H_K", "H_K_given_X",
    "grad_norm", "param_norm", "update_ratio", "lr",
]


def _init_encoder_accumulators() -> dict[str, float]:
    return {k: 0.0 for k in ENCODER_LOSS_KEYS + INFO_KEYS + ["total"]}


def _init_dynamics_accumulators() -> dict[str, float]:
    return {k: 0.0 for k in DYNAMICS_LOSS_KEYS + ["grad_norm", "param_norm", "lr", "total"]}


def _init_joint_accumulators() -> dict[str, float]:
    keys = (
        [f"enc/{k}" for k in ENCODER_LOSS_KEYS]
        + [f"dyn/{k}" for k in DYNAMICS_LOSS_KEYS]
        + INFO_KEYS
        + ["enc_total", "dyn_total", "total"]
    )
    return {k: 0.0 for k in keys}


# ── Encoder loss computation (same as train_unsupervised.py) ──────


def _compute_encoder_losses(
    x: torch.Tensor,
    model: TopoEncoderPrimitives,
    jump_op: FactorizedJumpOperator,
    args: argparse.Namespace,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute all 13 encoder losses with split encoder/decoder calls."""
    (
        K_chart, K_code, z_n, z_tex, enc_w, z_geo,
        vq_loss, indices, z_n_all, c_bar, v_local,
    ) = model.encoder(x)

    x_recon, dec_w, aux_losses = model.decoder(
        z_geo, z_tex, chart_index=None,
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
    return total, metrics


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


# ── Phase 1: Encoder warmup ──────────────────────────────────────


def _run_phase1(
    model: TopoEncoderPrimitives,
    jump_op: FactorizedJumpOperator,
    single_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
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

        for batch in single_loader:
            x = batch["feature"].to(device)

            total, metrics = _compute_encoder_losses(x, model, jump_op, args, epoch)

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

        should_log = (epoch % args.log_every == 0) or (epoch == args.phase1_epochs - 1)
        if should_log:
            print(
                f"P1 E{epoch:5d} | Loss: {acc['total']:.4f} "
                f"| LR: {acc['lr']:.2e}"
            )
            print(f"  Usage: {np.array2string(usage, precision=2, separator=', ')}")
            print(
                f"  Core: recon={acc['recon']:.3f} vq={acc['vq']:.3f} "
                f"entropy={acc['entropy']:.3f} consist={acc['consistency']:.3f}"
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
            print(f"  Info: div={acc['diversity']:.3f}")
            print(
                f"  Train: grad={acc['grad_norm']:.2e} "
                f"upd_ratio={acc['update_ratio']:.2e} lr={acc['lr']:.2e}"
            )
            print(
                f"  Metrics: perplexity={perplexity:.2f}/{K} "
                f"mean_r={mean_r:.3f}"
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
    )

    last_metrics: dict[str, float] = {}

    for epoch in tqdm(range(args.phase2_epochs), desc="Phase 2"):
        world_model.train()
        acc = _init_dynamics_accumulators()
        n_batches = 0

        for batch in seq_loader:
            features = batch["features"].to(device)  # [B, H, D_feat]
            actions = batch["actions"].to(device)     # [B, H, A]
            B, H, D_feat = features.shape

            # Encode all frames with frozen encoder
            with torch.no_grad():
                z_list, rw_list, K_list = [], [], []
                for t in range(H):
                    K_ch, _, z_n, _, enc_w, z_geo, *_ = model.encoder(features[:, t, :])
                    z_list.append(z_geo)
                    rw_list.append(enc_w)
                    K_list.append(K_ch)
                z_all = torch.stack(z_list, dim=1)
                K_all = torch.stack(K_list, dim=1)

            z_0 = z_all[:, 0, :]
            rw_0 = rw_list[0]
            pred_actions = actions[:, :-1, :]
            z_targets = z_all[:, 1:, :]
            chart_targets = K_all[:, 1:]

            wm_output = world_model(z_0, pred_actions, rw_0)
            loss, metrics = compute_phase2_loss(wm_output, z_targets, chart_targets, config_ns)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = compute_grad_norm(list(world_model.parameters()))
            param_norm = compute_param_norm(list(world_model.parameters()))
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(world_model.parameters(), args.grad_clip)
            optimizer.step()

            acc["total"] += metrics["total"]
            for k in DYNAMICS_LOSS_KEYS:
                if k in metrics:
                    acc[k] += metrics[k]
            acc["grad_norm"] += grad_norm
            acc["param_norm"] += param_norm
            acc["lr"] += optimizer.param_groups[0]["lr"]
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        for k in acc:
            acc[k] /= max(n_batches, 1)

        should_log = (epoch % args.log_every == 0) or (epoch == args.phase2_epochs - 1)
        if should_log:
            print(
                f"P2 E{epoch:5d} | Loss: {acc['total']:.4f} "
                f"| LR: {acc['lr']:.2e}"
            )
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
                f"  Train: grad={acc['grad_norm']:.2e} "
                f"param={acc['param_norm']:.2e}"
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
    )

    K = args.num_charts
    last_metrics: dict[str, float] = {}

    for epoch in tqdm(range(args.phase3_epochs), desc="Phase 3"):
        model.train()
        jump_op.train()
        world_model.train()
        acc = _init_joint_accumulators()
        n_batches = 0

        for batch in seq_loader:
            features = batch["features"].to(device)  # [B, H, D_feat]
            actions = batch["actions"].to(device)     # [B, H, A]
            B, H, D_feat = features.shape

            # Encode all H frames with gradients
            z_list, rw_list, K_list = [], [], []
            enc_metrics_0 = None

            for t in range(H):
                if t == 0:
                    # Full encoder losses on frame 0
                    enc_loss, enc_m = _compute_encoder_losses(
                        features[:, 0, :], model, jump_op, args, epoch,
                    )
                    enc_metrics_0 = enc_m
                    # Also get z_geo and enc_w for rollout
                    with torch.no_grad():
                        K_ch, _, _, _, enc_w, z_geo, *_ = model.encoder(features[:, 0, :])
                    # Re-run with grad for z_geo used in rollout
                    K_ch, _, z_n, _, enc_w, z_geo, _, _, z_n_all, _, _ = model.encoder(
                        features[:, 0, :],
                    )
                    z_list.append(z_geo)
                    rw_list.append(enc_w)
                    K_list.append(K_ch)
                else:
                    K_ch, _, z_n, _, enc_w, z_geo, *_ = model.encoder(features[:, t, :])
                    z_list.append(z_geo)
                    rw_list.append(enc_w)
                    K_list.append(K_ch)

            z_all = torch.stack(z_list, dim=1)
            K_all = torch.stack(K_list, dim=1)

            # World model rollout
            z_0 = z_all[:, 0, :]
            rw_0 = rw_list[0]
            pred_actions = actions[:, :-1, :]
            z_targets = z_all[:, 1:, :].detach()
            chart_targets = K_all[:, 1:].detach()

            wm_output = world_model(z_0, pred_actions, rw_0)
            dyn_loss, dyn_metrics = compute_phase2_loss(
                wm_output, z_targets, chart_targets, config_ns,
            )

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

        should_log = (epoch % args.log_every == 0) or (epoch == args.phase3_epochs - 1)
        if should_log:
            print(
                f"P3 E{epoch:5d} | Loss: {acc['total']:.4f} "
                f"(enc={acc['enc_total']:.4f} dyn={acc['dyn_total']:.4f}) "
                f"| LR: {acc['lr']:.2e}"
            )
            print(
                f"  Encoder: recon={acc['enc/recon']:.3f} "
                f"vq={acc['enc/vq']:.3f} "
                f"entropy={acc['enc/entropy']:.3f} "
                f"consist={acc['enc/consistency']:.3f}"
            )
            print(
                f"  Geo: unif={acc['enc/uniformity']:.3f} "
                f"rad_cal={acc['enc/radial_cal']:.3f} "
                f"collapse: chart={acc['enc/chart_collapse']:.4f} "
                f"code={acc['enc/code_collapse']:.4f}"
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
                f"  Train: grad={acc['grad_norm']:.2e} "
                f"upd_ratio={acc['update_ratio']:.2e}"
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
        world_model = GeometricWorldModel(
            latent_dim=args.latent_dim,
            action_dim=args.action_dim,
            num_charts=K,
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
        ).to(device)

    print(f"Encoder: {count_parameters(model):,} params")
    print(f"Jump op: {count_parameters(jump_op):,} params")
    if world_model is not None:
        print(f"World model: {count_parameters(world_model):,} params")

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

    if args.phase1_epochs > 0:
        _run_phase1(model, jump_op, single_loader, args, device)

    if args.phase2_epochs > 0:
        assert world_model is not None
        assert seq_loader is not None
        last_dyn_metrics = _run_phase2(
            model, jump_op, world_model, seq_loader, args, device,
        )
        had_world_model = True

    if args.phase3_epochs > 0:
        assert world_model is not None
        assert seq_loader is not None
        last_dyn_metrics = _run_phase3(
            model, jump_op, world_model, seq_loader, args, device,
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
