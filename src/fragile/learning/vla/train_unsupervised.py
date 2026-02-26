"""Unsupervised TopoEncoder training on cached VLA features.

Follows the same loss-logging pattern as ``topoencoder_mnist.py``: split
encoder/decoder calls for full internal access, per-epoch detailed loss
display with tiers, info metrics, and training dynamics.
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

# ── Tracked loss terms (13) ────────────────────────────────────────
LOSS_KEYS = [
    "recon", "vq", "entropy", "consistency",
    "diversity", "uniformity", "radial_cal",
    "codebook_spread", "codebook_center",
    "chart_collapse", "code_collapse",
    "window", "jump",
]

# ── Info metrics (7) ───────────────────────────────────────────────
INFO_KEYS = [
    "I_XK", "H_K", "H_K_given_X",
    "grad_norm", "param_norm", "update_ratio", "lr",
]


def _init_accumulators() -> dict[str, float]:
    return {k: 0.0 for k in LOSS_KEYS + INFO_KEYS + ["total"]}


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


# ── Main training function ─────────────────────────────────────────


def train_unsupervised(args: argparse.Namespace) -> None:  # noqa: C901
    """Train a TopoEncoder unsupervised on cached VLA features."""
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Dataset ────────────────────────────────────────────────
    dataset = VLAFeatureDataset(args.feature_cache_dir, sequence_length=1)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    input_dim = dataset[0]["feature"].shape[0]
    print(f"Dataset: {len(dataset)} frames, {len(loader)} batches (bs={args.batch_size})")
    print(f"Feature dim: {input_dim}")

    # ── Model ──────────────────────────────────────────────────
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

    all_params = list(model.parameters()) + list(jump_op.parameters())
    print(f"Model: {count_parameters(model):,} params | Jump: {count_parameters(jump_op):,} params")

    # ── Optimizer & scheduler ──────────────────────────────────
    optimizer = torch.optim.Adam(all_params, lr=args.lr)
    scheduler = None
    if args.use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Resume ─────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        jump_op.load_state_dict(ckpt["jump_op"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        if scheduler and "scheduler" in ckpt and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    # ── Output dir ─────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Training loop ──────────────────────────────────────────
    for epoch in tqdm(range(start_epoch, args.epochs), desc="Training"):
        model.train()
        jump_op.train()
        acc = _init_accumulators()
        n_batches = 0
        current_tau = _get_hard_routing_tau(args, epoch, args.epochs)

        for batch in loader:
            x = batch["feature"].to(device)

            # ── Encoder forward (split call) ───────────────
            (
                K_chart, K_code, z_n, z_tex, enc_w, z_geo,
                vq_loss, indices, z_n_all, c_bar, v_local,
            ) = model.encoder(
                x,
                hard_routing=args.hard_routing,
                hard_routing_tau=current_tau,
            )

            # ── Decoder forward (dreaming mode) ────────────
            # When hard routing is on, pass encoder weights to decoder so both
            # use the same one-hot assignment (avoids consistency loss explosion).
            router_override = enc_w if args.hard_routing else None
            x_recon, dec_w, aux_losses = model.decoder(
                z_geo, z_tex, chart_index=None,
                router_weights=router_override,
                hard_routing=args.hard_routing,
                hard_routing_tau=current_tau,
            )

            # ── Loss computation (13 terms) ────────────────
            recon_loss = F.mse_loss(x_recon, x)
            entropy_loss = math.log(K) - compute_routing_entropy(enc_w)
            consistency_loss = model.compute_consistency_loss(enc_w, dec_w)
            diversity_loss = compute_diversity_loss(enc_w, K)
            uniformity_loss = compute_hyperbolic_uniformity_loss(z_geo)
            radial_cal_loss = compute_radial_calibration_loss(z_geo, enc_w, K)

            codebook = model.encoder.codebook  # [N_c, codes_per_chart, D]
            cb_spread_loss = compute_codebook_spread_loss(codebook)
            cb_center_loss = compute_codebook_centering_loss(codebook)

            chart_collapse_loss = compute_chart_collapse_penalty(enc_w, K)
            code_collapse_loss = compute_code_collapse_penalty(
                v_local, codebook, enc_w,
            )

            window_loss, window_info = compute_window_loss(enc_w, K)

            # Jump (scheduled weight)
            current_jump_weight = get_jump_weight_schedule(
                epoch,
                warmup_end=args.w_jump_warmup,
                ramp_end=args.w_jump_ramp_end,
                final_weight=args.w_jump,
            )
            jump_loss, _jump_info = compute_jump_consistency_loss(
                z_n_all, enc_w, jump_op,
            )

            # ── Total loss ─────────────────────────────────
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

            # ── Gradient step ──────────────────────────────
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
                if param_norm > 0
                else 0.0
            )

            # ── Accumulate ─────────────────────────────────
            acc["total"] += total.item()
            acc["recon"] += recon_loss.item()
            acc["vq"] += vq_loss.item()
            acc["entropy"] += entropy_loss.item()
            acc["consistency"] += consistency_loss.item()
            acc["diversity"] += diversity_loss.item()
            acc["uniformity"] += uniformity_loss.item()
            acc["radial_cal"] += radial_cal_loss.item()
            acc["codebook_spread"] += cb_spread_loss.item()
            acc["codebook_center"] += cb_center_loss.item()
            acc["chart_collapse"] += chart_collapse_loss.item()
            acc["code_collapse"] += code_collapse_loss.item()
            acc["window"] += window_loss.item()
            acc["jump"] += jump_loss.item()
            acc["I_XK"] += float(window_info.get("I_XK", 0.0))
            acc["H_K"] += float(window_info.get("H_K", 0.0))
            acc["H_K_given_X"] += float(window_info.get("H_K_given_X", 0.0))
            acc["grad_norm"] += grad_norm
            acc["param_norm"] += param_norm
            acc["update_ratio"] += update_ratio
            acc["lr"] += current_lr
            n_batches += 1

        # ── Scheduler step ─────────────────────────────────
        if scheduler is not None:
            scheduler.step()

        # ── Average over batches ───────────────────────────
        for k in acc:
            acc[k] /= max(n_batches, 1)

        # ── Usage / perplexity (eval pass) ─────────────────
        model.eval()
        all_charts: list[torch.Tensor] = []
        all_radii: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                x_eval = batch["feature"].to(device)
                K_ch, _, _, _, _, z_g, *_ = model.encoder(x_eval)
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

        # ── Logging ────────────────────────────────────────
        should_log = (epoch % args.log_every == 0) or (epoch == args.epochs - 1)
        if should_log:
            print(
                f"Epoch {epoch:5d} | Loss: {acc['total']:.4f} "
                f"| LR: {acc['lr']:.2e}"
            )
            print(
                f"  Usage: {np.array2string(usage, precision=2, separator=', ')}"
            )
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
                f"(I_XK={acc['I_XK']:.3f} "
                f"H_K={acc['H_K']:.3f} "
                f"H_K|X={acc['H_K_given_X']:.3f})"
            )
            print(
                f"  Jump: {acc['jump']:.3f} "
                f"(lambda={current_jump_weight:.3f})"
            )
            print(f"  Info: div={acc['diversity']:.3f}")
            print(
                f"  Train: grad={acc['grad_norm']:.2e} "
                f"upd_ratio={acc['update_ratio']:.2e} "
                f"lr={acc['lr']:.2e}"
            )
            print(
                f"  Metrics: perplexity={perplexity:.2f}/{K} "
                f"mean_r={mean_r:.3f}"
            )
            print("-" * 60)

        # ── Checkpoint ─────────────────────────────────────
        should_save = (
            (epoch > 0 and epoch % args.save_every == 0)
            or epoch == args.epochs - 1
        )
        if should_save:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch:05d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "jump_op": jump_op.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": (
                        scheduler.state_dict() if scheduler else None
                    ),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

    # ── Final diagnostics ──────────────────────────────────
    print("\n=== Final Diagnostics ===")
    model.eval()
    all_features: list[torch.Tensor] = []
    all_ep_ids: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
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

    # Chart transition matrix
    with torch.no_grad():
        K_chart_all, *_ = model.encoder(all_features_t.to(device))

    fig = plot_chart_transitions(K_chart_all.cpu(), all_ep_ids_t)
    fig.savefig(
        os.path.join(args.output_dir, "chart_transitions.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"\nAll outputs saved to {args.output_dir}")


# ── CLI ────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for unsupervised VLA training."""
    p = argparse.ArgumentParser(
        description="Unsupervised TopoEncoder training on VLA features",
    )

    # Data / output
    p.add_argument(
        "--feature-cache-dir",
        default="outputs/vla/features",
        help="Path to cached VLA features",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/vla/unsupervised",
        help="Output directory for checkpoints and plots",
    )

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--use-scheduler", action="store_true")

    # Architecture
    p.add_argument("--latent-dim", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-charts", type=int, default=16)
    p.add_argument("--codes-per-chart", type=int, default=64)

    # Logging / saving
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=50)

    # Resume / device
    p.add_argument("--resume", default="", help="Checkpoint path to resume from")
    p.add_argument("--device", default="auto")
    p.add_argument("--hard-routing", action="store_true", default=False,
                   help="Use Gumbel-softmax hard routing (one-hot forward, ST gradients)")
    p.add_argument("--hard-routing-tau", type=float, default=1.0,
                   help="Starting temperature for Gumbel-softmax hard routing")
    p.add_argument("--hard-routing-tau-end", type=float, default=None,
                   help="Final tau after annealing (None = no annealing, use constant tau)")
    p.add_argument("--hard-routing-tau-anneal-epochs", type=int, default=None,
                   help="Anneal tau linearly over this many epochs (None = total epochs)")

    # ── Loss weights ───────────────────────────────────────
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

    args = p.parse_args()
    train_unsupervised(args)


if __name__ == "__main__":
    main()
