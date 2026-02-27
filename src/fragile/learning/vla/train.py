"""3-phase training loop for the TopoEncoder x SmolVLA experiment."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, fields

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from fragile.learning.core.layers import TopoEncoderPrimitives
from fragile.learning.core.layers.topology import FactorizedJumpOperator
from fragile.learning.hyperbolic_losses import (
    compute_jump_consistency_loss as compute_jump_consistency_loss_hyp,
    get_jump_weight_schedule,
)

from .config import VLAConfig
from .extract_features import VLAFeatureDataset
from .losses import compute_phase1_loss, compute_phase2_loss, compute_phase3_loss
from .covariant_world_model import GeometricWorldModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_checkpoint(
    path: str,
    encoder: nn.Module,
    jump_op: nn.Module,
    world_model: nn.Module | None,
    optimizers: dict,
    epoch: int,
    phase: int,
    config: VLAConfig,
    metrics: dict | None = None,
) -> None:
    """Save a VLA training checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "epoch": epoch,
        "phase": phase,
        "config": asdict(config),
        "encoder": {k: v.cpu() for k, v in encoder.state_dict().items()},
        "jump_op": {k: v.cpu() for k, v in jump_op.state_dict().items()},
        "world_model": (
            {k: v.cpu() for k, v in world_model.state_dict().items()}
            if world_model is not None
            else None
        ),
        "optimizers": {
            name: opt.state_dict() for name, opt in optimizers.items()
        },
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def _log_metrics(
    metrics: dict[str, float],
    epoch: int,
    phase: int,
    mlflow_enabled: bool,
) -> None:
    """Print and optionally MLflow-log metrics."""
    parts = [f"P{phase} E{epoch:03d}"]
    for k, v in metrics.items():
        if k == "total":
            parts.insert(1, f"loss={v:.4f}")
        elif not k.startswith("H_") and not k.startswith("I_"):
            parts.append(f"{k}={v:.4f}")
    print(" | ".join(parts))

    if mlflow_enabled:
        try:
            from fragile.learning.mlflow_logging import log_mlflow_metrics

            prefixed = {f"phase{phase}/{k}": v for k, v in metrics.items()}
            log_mlflow_metrics(prefixed, step=epoch, enabled=True)
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------


def _run_phase1(
    encoder: nn.Module,
    jump_op: nn.Module,
    train_loader: DataLoader,
    config: VLAConfig,
    mlflow_enabled: bool = False,
) -> dict[str, float]:
    """Phase 1: Train encoder on single-frame features."""
    device = torch.device(config.device)
    encoder.to(device)
    jump_op.to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(jump_op.parameters()),
        lr=config.lr_encoder,
    )

    last_metrics: dict[str, float] = {}

    for epoch in range(1, config.phase1_epochs + 1):
        encoder.train()
        jump_op.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["feature"].to(device)

            # Forward through encoder
            (
                x_recon, vq_loss, enc_router_weights, dec_router_weights,
                K_chart, z_geo, z_n, c_bar, aux_losses,
            ) = encoder(x)

            # Phase 1 loss
            base_loss, zn_reg_loss, metrics = compute_phase1_loss(
                x, x_recon, vq_loss, enc_router_weights, dec_router_weights,
                z_geo, encoder, config,
            )
            loss = base_loss + zn_reg_loss

            # Jump consistency loss (with warmup schedule)
            jump_w = get_jump_weight_schedule(
                epoch, warmup_end=config.w_jump_warmup,
                ramp_end=config.w_jump_ramp_end, final_weight=config.w_jump,
            )
            if jump_w > 0 and hasattr(encoder, "encoder"):
                # Get z_n_all_charts from a re-run of just the encoder.encoder
                enc_out = encoder.encoder(x)
                z_n_all_charts = enc_out[8]  # z_n_all_charts
                loss_jump = compute_jump_consistency_loss_hyp(
                    jump_op, z_n_all_charts, enc_router_weights,
                )
                loss = loss + jump_w * loss_jump
                metrics["jump"] = loss_jump.item()

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(jump_op.parameters()),
                    config.grad_clip,
                )
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        last_metrics = metrics
        last_metrics["epoch_loss"] = epoch_loss / max(n_batches, 1)

        if epoch % config.log_every == 0 or epoch == config.phase1_epochs:
            _log_metrics(last_metrics, epoch, phase=1, mlflow_enabled=mlflow_enabled)

        if config.save_every > 0 and epoch % config.save_every == 0:
            ckpt_path = os.path.join(
                config.output_dir, f"checkpoint_p1_e{epoch:04d}.pt",
            )
            _save_checkpoint(
                ckpt_path, encoder, jump_op, None,
                {"encoder_jump": optimizer}, epoch, 1, config, last_metrics,
            )

    return last_metrics


def _run_phase2(
    encoder: nn.Module,
    world_model: nn.Module,
    seq_loader: DataLoader,
    config: VLAConfig,
    mlflow_enabled: bool = False,
) -> dict[str, float]:
    """Phase 2: Train world model with frozen encoder."""
    device = torch.device(config.device)
    encoder.to(device)
    world_model.to(device)

    # Freeze encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    optimizer = optim.Adam(world_model.parameters(), lr=config.lr_wm)
    last_metrics: dict[str, float] = {}

    for epoch in range(1, config.phase2_epochs + 1):
        world_model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in seq_loader:
            features = batch["features"].to(device)  # [B, H, D_feat]
            actions = batch["actions"].to(device)  # [B, H, A]

            B, H, D_feat = features.shape

            # Encode all frames
            with torch.no_grad():
                z_list = []
                rw_list = []
                K_list = []
                for t in range(H):
                    (
                        x_recon, vq_loss, enc_rw, dec_rw,
                        K_chart, z_geo, z_n, c_bar, aux,
                    ) = encoder(features[:, t, :])
                    z_list.append(z_geo)
                    rw_list.append(enc_rw)
                    K_list.append(K_chart)

                z_all = torch.stack(z_list, dim=1)  # [B, H, D_lat]
                K_all = torch.stack(K_list, dim=1)  # [B, H]

            # World model rollout from z_0
            z_0 = z_all[:, 0, :]
            rw_0 = rw_list[0]
            # Actions for prediction: use actions[0:H-1] to predict z[1:H]
            pred_actions = actions[:, :-1, :]  # [B, H-1, A]
            z_targets = z_all[:, 1:, :]  # [B, H-1, D]
            chart_targets = K_all[:, 1:]  # [B, H-1]

            wm_output = world_model(z_0, pred_actions, rw_0)

            loss, metrics = compute_phase2_loss(
                wm_output, z_targets, chart_targets, config,
            )

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(world_model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        last_metrics = metrics
        last_metrics["epoch_loss"] = epoch_loss / max(n_batches, 1)

        if epoch % config.log_every == 0 or epoch == config.phase2_epochs:
            _log_metrics(last_metrics, epoch, phase=2, mlflow_enabled=mlflow_enabled)

        if config.save_every > 0 and epoch % config.save_every == 0:
            ckpt_path = os.path.join(
                config.output_dir, f"checkpoint_p2_e{epoch:04d}.pt",
            )
            _save_checkpoint(
                ckpt_path, encoder, FactorizedJumpOperator(
                    config.num_charts, config.latent_dim,
                ), world_model,
                {"wm": optimizer}, epoch, 2, config, last_metrics,
            )

    # Unfreeze encoder for Phase 3
    for p in encoder.parameters():
        p.requires_grad_(True)

    return last_metrics


def _run_phase3(
    encoder: nn.Module,
    jump_op: nn.Module,
    world_model: nn.Module,
    seq_loader: DataLoader,
    config: VLAConfig,
    mlflow_enabled: bool = False,
) -> dict[str, float]:
    """Phase 3: Joint fine-tuning of encoder + world model."""
    device = torch.device(config.device)
    encoder.to(device)
    jump_op.to(device)
    world_model.to(device)

    # Two param groups with different LRs
    optimizer = optim.Adam([
        {"params": list(encoder.parameters()) + list(jump_op.parameters()),
         "lr": config.lr_joint_encoder},
        {"params": world_model.parameters(), "lr": config.lr_joint_wm},
    ])

    last_metrics: dict[str, float] = {}

    for epoch in range(1, config.phase3_epochs + 1):
        encoder.train()
        jump_op.train()
        world_model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in seq_loader:
            features = batch["features"].to(device)  # [B, H, D_feat]
            actions = batch["actions"].to(device)  # [B, H, A]

            B, H, D_feat = features.shape

            # Encode all frames (with gradients)
            z_list = []
            rw_list = []
            K_list = []
            x_recon_0 = None
            vq_loss_0 = None
            enc_rw_0 = None
            dec_rw_0 = None
            z_geo_0 = None

            for t in range(H):
                (
                    x_recon, vq_loss, enc_rw, dec_rw,
                    K_chart, z_geo, z_n, c_bar, aux,
                ) = encoder(features[:, t, :])
                z_list.append(z_geo)
                rw_list.append(enc_rw)
                K_list.append(K_chart)

                if t == 0:
                    x_recon_0 = x_recon
                    vq_loss_0 = vq_loss
                    enc_rw_0 = enc_rw
                    dec_rw_0 = dec_rw
                    z_geo_0 = z_geo

            z_all = torch.stack(z_list, dim=1)
            K_all = torch.stack(K_list, dim=1)

            # World model rollout
            z_0 = z_all[:, 0, :]
            rw_0 = rw_list[0].detach()
            pred_actions = actions[:, :-1, :]
            z_targets = z_all[:, 1:, :].detach()  # Detach targets for stability
            chart_targets = K_all[:, 1:].detach()

            wm_output = world_model(z_0, pred_actions, rw_0)

            base_enc, zn_reg, dyn_loss, metrics = compute_phase3_loss(
                features[:, 0, :], x_recon_0, vq_loss_0,
                enc_rw_0, dec_rw_0, z_geo_0, encoder,
                wm_output, z_targets, chart_targets, config,
            )

            # Three-pass gradient surgery: isolate z_n gradients
            optimizer.zero_grad()

            sf_params = list(encoder.encoder.structure_filter.parameters())

            protected_params = (
                list(encoder.encoder.feature_extractor.parameters())
                + list(encoder.encoder.val_proj.parameters())
                + list(encoder.encoder.cov_router.parameters())
                + [encoder.encoder.chart_centers]
            )
            if encoder.encoder.soft_equiv_layers is not None:
                for layer in encoder.encoder.soft_equiv_layers:
                    protected_params += list(layer.parameters())

            # Pass 1: base encoder loss (recon + non-z_n terms)
            (config.phase3_encoder_scale * base_enc).backward(retain_graph=True)
            # Zero structure_filter grads → blocks reconstruction from z_n
            for p in sf_params:
                if p.grad is not None:
                    p.grad.zero_()

            # Pass 2: z_n regularization (separate scale)
            if zn_reg.grad_fn is not None:
                (config.phase3_zn_reg_scale * zn_reg).backward(retain_graph=True)

            # Save protected params (they have encoder grads: base + zn_reg)
            saved_grads = [
                p.grad.clone() if p.grad is not None else None
                for p in protected_params
            ]

            # Pass 3: dynamics loss (accumulates onto all params)
            (config.phase3_dynamics_scale * dyn_loss).backward()

            # Restore protected params (removes dynamics contribution)
            for p, saved in zip(protected_params, saved_grads):
                if saved is not None:
                    p.grad = saved
                elif p.grad is not None:
                    p.grad.zero_()

            all_params = (
                list(encoder.parameters()) + list(jump_op.parameters())
                + list(world_model.parameters())
            )
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(all_params, config.grad_clip)
            optimizer.step()

            total = (
                config.phase3_encoder_scale * base_enc
                + config.phase3_zn_reg_scale * zn_reg
                + config.phase3_dynamics_scale * dyn_loss
            )
            epoch_loss += total.item()
            n_batches += 1

        last_metrics = metrics
        last_metrics["epoch_loss"] = epoch_loss / max(n_batches, 1)

        if epoch % config.log_every == 0 or epoch == config.phase3_epochs:
            _log_metrics(last_metrics, epoch, phase=3, mlflow_enabled=mlflow_enabled)

        if config.save_every > 0 and epoch % config.save_every == 0:
            ckpt_path = os.path.join(
                config.output_dir, f"checkpoint_p3_e{epoch:04d}.pt",
            )
            _save_checkpoint(
                ckpt_path, encoder, jump_op, world_model,
                {"joint": optimizer}, epoch, 3, config, last_metrics,
            )

    return last_metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def train_vla(config: VLAConfig) -> dict:
    """Run the full 3-phase VLA training pipeline.

    Returns:
        Dict with final metrics from each phase.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    # MLflow setup
    mlflow_enabled = False
    if config.mlflow:
        try:
            import mlflow

            if config.mlflow_tracking_uri:
                mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            if config.mlflow_experiment:
                mlflow.set_experiment(config.mlflow_experiment)
            run_name = config.mlflow_run_name or "vla_train"
            mlflow.start_run(run_name=run_name)
            safe_params = {}
            for k, v in asdict(config).items():
                safe_params[k] = str(v) if not isinstance(v, (int, float, str, bool)) else v
            mlflow.log_params(safe_params)
            mlflow_enabled = True
        except ImportError:
            print("MLflow not available, skipping.")

    # --- Instantiate models ---
    print("Building encoder …")
    encoder = TopoEncoderPrimitives(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_charts=config.num_charts,
        codes_per_chart=config.codes_per_chart,
        covariant_attn=config.covariant_attn,
        covariant_attn_tensorization=config.covariant_attn_tensorization,
        covariant_attn_rank=config.covariant_attn_rank,
        covariant_attn_tau_min=config.covariant_attn_tau_min,
        covariant_attn_denom_min=config.covariant_attn_denom_min,
        covariant_attn_use_transport=config.covariant_attn_use_transport,
        covariant_attn_transport_eps=config.covariant_attn_transport_eps,
        soft_equiv_metric=config.soft_equiv_metric,
        soft_equiv_bundle_size=config.soft_equiv_bundle_size or None,
        soft_equiv_hidden_dim=config.soft_equiv_hidden_dim,
        soft_equiv_use_spectral_norm=config.soft_equiv_use_spectral_norm,
        soft_equiv_zero_self_mixing=config.soft_equiv_zero_self_mixing,
        soft_equiv_soft_assign=config.soft_equiv_soft_assign,
        soft_equiv_temperature=config.soft_equiv_temperature,
        conv_backbone=config.conv_backbone,
    )

    print("Building jump operator …")
    jump_op = FactorizedJumpOperator(
        num_charts=config.num_charts,
        latent_dim=config.latent_dim,
    )

    print("Building world model …")
    world_model = GeometricWorldModel(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        num_charts=config.num_charts,
        hidden_dim=config.wm_hidden_dim,
        dt=config.wm_dt,
        gamma_friction=config.wm_gamma_friction,
        T_c=config.wm_T_c,
        alpha_potential=config.wm_alpha_potential,
        beta_curl=config.wm_beta_curl,
        gamma_risk=config.wm_gamma_risk,
        use_boris=config.wm_use_boris,
        use_jump=config.wm_use_jump,
        jump_rate_hidden=config.wm_jump_rate_hidden,
    )

    # Count parameters
    n_enc = sum(p.numel() for p in encoder.parameters())
    n_jump = sum(p.numel() for p in jump_op.parameters())
    n_wm = sum(p.numel() for p in world_model.parameters())
    print(f"Parameters: encoder={n_enc:,}  jump={n_jump:,}  wm={n_wm:,}  total={n_enc+n_jump+n_wm:,}")

    # --- Data loaders ---
    print(f"Loading features from {config.feature_cache_dir} …")

    single_ds = VLAFeatureDataset(config.feature_cache_dir, sequence_length=1)
    single_loader = DataLoader(
        single_ds, batch_size=config.batch_size, shuffle=True, drop_last=True,
    )

    seq_ds = VLAFeatureDataset(config.feature_cache_dir, sequence_length=config.sequence_length)
    seq_loader = DataLoader(
        seq_ds, batch_size=config.batch_size, shuffle=True, drop_last=True,
    )

    results: dict[str, dict] = {}

    # --- Phase 1: Encoder training ---
    print("\n" + "=" * 60)
    print("Phase 1: Encoder training")
    print("=" * 60)
    results["phase1"] = _run_phase1(
        encoder, jump_op, single_loader, config, mlflow_enabled,
    )

    # --- Phase 2: World model training (encoder frozen) ---
    print("\n" + "=" * 60)
    print("Phase 2: World model training (encoder frozen)")
    print("=" * 60)
    results["phase2"] = _run_phase2(
        encoder, world_model, seq_loader, config, mlflow_enabled,
    )

    # --- Phase 3: Joint fine-tuning ---
    print("\n" + "=" * 60)
    print("Phase 3: Joint fine-tuning")
    print("=" * 60)
    results["phase3"] = _run_phase3(
        encoder, jump_op, world_model, seq_loader, config, mlflow_enabled,
    )

    # Save final checkpoint
    final_path = os.path.join(config.output_dir, "checkpoint_final.pt")
    _save_checkpoint(
        final_path, encoder, jump_op, world_model,
        {}, 0, 3, config, results.get("phase3"),
    )
    print(f"\nFinal checkpoint saved to {final_path}")

    if mlflow_enabled:
        try:
            import mlflow
            mlflow.end_run()
        except ImportError:
            pass

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: parse args into VLAConfig and run training."""
    parser = argparse.ArgumentParser(description="Train TopoEncoder x SmolVLA")

    # Add all VLAConfig fields as CLI arguments
    import dataclasses

    _MISSING = dataclasses.MISSING
    for f in fields(VLAConfig):
        default = f.default if f.default is not _MISSING else None
        if f.type is bool:
            parser.add_argument(
                f"--{f.name}",
                type=lambda x: x.lower() in ("true", "1", "yes"),
                default=default,
                help=f"{f.name} (default: {default})",
            )
        elif f.type is int:
            parser.add_argument(f"--{f.name}", type=int, default=default)
        elif f.type is float:
            parser.add_argument(f"--{f.name}", type=float, default=default)
        elif f.type is str:
            parser.add_argument(f"--{f.name}", type=str, default=default)

    args = parser.parse_args()
    config = VLAConfig(**{k: v for k, v in vars(args).items() if v is not None})
    train_vla(config)


if __name__ == "__main__":
    main()
