"""
TopoEncoder Benchmark: CIFAR-10 Image Classification

This script benchmarks the Attentive Atlas architecture on CIFAR-10 images,
comparing against a standard VQ-VAE and Vanilla AE baseline.

CIFAR-10 contains 10 classes:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Key differences from topoencoder_2d.py:
- Input: Flattened 32x32x3 images (3072 dimensions)
- Charts: 10 (one per class)
- Visualization: Image grids instead of 3D scatter plots
- Latent coloring: By class label (tab10) instead of continuous rainbow

Usage:
    python topoencoder_cifar10.py [--epochs 500] [--n_samples 10000]

Reference: fragile-index.md Sections 7.8, 7.10
"""

import argparse
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from tqdm import tqdm

from dataviz import visualize_latent_images, visualize_results_images
from fragile.datasets import CIFAR10_CLASSES, get_cifar10_data
from fragile.core.layers import FactorizedJumpOperator, StandardVQ, TopoEncoder, VanillaAE
from fragile.core.losses import (
    compute_code_entropy_loss,
    compute_disentangle_loss,
    compute_diversity_loss,
    compute_jump_consistency_loss,
    compute_kl_prior_loss,
    compute_orthogonality_loss,
    compute_orbit_loss,
    compute_per_chart_code_entropy_loss,
    compute_routing_entropy,
    compute_separation_loss,
    compute_variance_loss,
    compute_vicreg_invariance_loss,
    compute_window_loss,
    get_jump_weight_schedule,
    SupervisedTopologyLoss,
)


# ==========================================
# 1. CONFIGURATION
# ==========================================
@dataclass
class TopoEncoderCIFAR10Config:
    """Configuration for the TopoEncoder CIFAR-10 benchmark."""

    # Data (CIFAR-10: 32x32x3 = 3072 dimensions)
    n_samples: int = 50000  # Full CIFAR-10 training set
    input_dim: int = 3072  # 32*32*3 flattened
    image_shape: tuple = (32, 32, 3)  # For visualization

    # Model architecture (larger for images)
    hidden_dim: int = 256  # Increased from 32
    latent_dim: int = 2  # For 2D visualization
    num_charts: int = 10  # One per CIFAR-10 class
    codes_per_chart: int = 64  # More codes for image complexity
    num_codes_standard: int = 256

    # Training
    epochs: int = 500
    batch_size: int = 256
    lr: float = 1e-3
    vq_commitment_cost: float = 0.25
    entropy_weight: float = 0.1
    consistency_weight: float = 0.1

    # Tier 1 losses (low overhead)
    variance_weight: float = 0.1
    diversity_weight: float = 0.1
    separation_weight: float = 0.1
    separation_margin: float = 2.0

    # Tier 2 losses (medium overhead)
    window_weight: float = 0.5
    window_eps_ground: float = 0.1
    disentangle_weight: float = 0.1

    # Tier 3 losses (geometry/codebook health)
    orthogonality_weight: float = 0.01
    code_entropy_weight: float = 0.0
    per_chart_code_entropy_weight: float = 0.1

    # Tier 4 losses (invariance)
    kl_prior_weight: float = 0.01
    orbit_weight: float = 0.0
    vicreg_inv_weight: float = 0.0
    augment_noise_std: float = 0.1

    # Tier 5: Jump Operator
    jump_weight: float = 0.1
    jump_warmup: int = 50
    jump_ramp_end: int = 100
    jump_global_rank: int = 0

    # Supervised topology loss
    enable_supervised: bool = True
    num_classes: int = 10
    sup_weight: float = 1.0
    sup_purity_weight: float = 0.1
    sup_balance_weight: float = 0.01
    sup_metric_weight: float = 0.01
    sup_metric_margin: float = 1.0
    sup_temperature: float = 1.0

    # Learning rate scheduling
    use_scheduler: bool = True
    min_lr: float = 1e-5

    # Gradient clipping
    grad_clip: float = 1.0

    # Benchmark control
    disable_ae: bool = False
    disable_vq: bool = False

    # Train/test split
    test_split: float = 0.2

    # Logging and output
    log_every: int = 50
    save_every: int = 50
    output_dir: str = "outputs/topoencoder_cifar10"

    # Device
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def compute_matching_hidden_dim(
    target_params: int,
    input_dim: int = 3072,
    latent_dim: int = 2,
    num_codes: int = 256,
) -> int:
    """Compute hidden_dim for StandardVQ to match target parameter count."""
    offset = 5 + num_codes * latent_dim
    coef_h = 2 * input_dim + 8
    discriminant = coef_h**2 + 8 * (target_params - offset)
    if discriminant < 0:
        return 128
    h = (-coef_h + math.sqrt(discriminant)) / 4
    return max(64, int(h))


# ==========================================
# 2. METRICS
# ==========================================
def compute_ami(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute Adjusted Mutual Information score."""
    return float(adjusted_mutual_info_score(labels_true, labels_pred))


# ==========================================
# 3. AUGMENTATION
# ==========================================
def augment_cifar10(
    x: torch.Tensor,
    noise_std: float = 0.1,
) -> torch.Tensor:
    """Apply noise augmentation to flattened CIFAR-10 images.

    Args:
        x: Input tensor [B, 3072] (flattened images)
        noise_std: Standard deviation of additive noise

    Returns:
        Augmented tensor [B, 3072]
    """
    return x + torch.randn_like(x) * noise_std


# ==========================================
# 4. TRAINING
# ==========================================
def train_benchmark(config: TopoEncoderCIFAR10Config) -> dict:
    """Train models and return results."""
    # Create output directory
    if config.save_every > 0:
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"Saving training progress to: {config.output_dir}/")

    # Load CIFAR-10 data
    X, labels, colors = get_cifar10_data(config.n_samples)
    labels = labels.astype(np.int64)
    print(f"Loaded {len(X)} CIFAR-10 images (shape: {X.shape})")
    print(f"Classes: {CIFAR10_CLASSES}")

    if not (0.0 <= config.test_split < 1.0):
        raise ValueError("test_split must be in [0.0, 1.0).")

    n_total = X.shape[0]
    test_size = max(1, int(n_total * config.test_split)) if config.test_split > 0 else 0
    if test_size >= n_total:
        test_size = max(1, n_total - 1)
    train_size = n_total - test_size
    perm = torch.randperm(n_total)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]
    train_idx_np = train_idx.numpy()
    test_idx_np = test_idx.numpy()

    X_train = X[train_idx]
    X_test = X[test_idx] if test_size > 0 else X
    labels_train = labels[train_idx_np]
    labels_test = labels[test_idx_np] if test_size > 0 else labels
    colors_train = colors[train_idx_np]
    colors_test = colors[test_idx_np] if test_size > 0 else colors

    print(
        f"Train/test split: {len(X_train)}/{len(X_test)} "
        f"(test={config.test_split:.2f})"
    )

    # Create TopoEncoder
    model_atlas = TopoEncoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_charts=config.num_charts,
        codes_per_chart=config.codes_per_chart,
    )
    topo_params = count_parameters(model_atlas)

    # Create StandardVQ with matching parameter count
    model_std = None
    opt_std = None
    std_params = 0
    std_hidden_dim = 0
    if not config.disable_vq:
        std_hidden_dim = compute_matching_hidden_dim(
            target_params=topo_params,
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            num_codes=config.num_codes_standard,
        )
        model_std = StandardVQ(
            input_dim=config.input_dim,
            hidden_dim=std_hidden_dim,
            latent_dim=config.latent_dim,
            num_codes=config.num_codes_standard,
        )
        std_params = count_parameters(model_std)

    # Create VanillaAE
    model_ae = None
    opt_ae = None
    ae_params = 0
    ae_hidden_dim = 0
    if not config.disable_ae:
        ae_hidden_dim = compute_matching_hidden_dim(
            target_params=topo_params,
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            num_codes=0,
        )
        model_ae = VanillaAE(
            input_dim=config.input_dim,
            hidden_dim=ae_hidden_dim,
            latent_dim=config.latent_dim,
        )
        ae_params = count_parameters(model_ae)

    print(f"\nModel Parameters (fair comparison):")
    print(f"  TopoEncoder: {topo_params:,} params (hidden_dim={config.hidden_dim})")
    if not config.disable_vq:
        print(f"  StandardVQ:  {std_params:,} params (hidden_dim={std_hidden_dim})")
    else:
        print(f"  StandardVQ:  DISABLED")
    if not config.disable_ae:
        print(f"  VanillaAE:   {ae_params:,} params (hidden_dim={ae_hidden_dim})")
    else:
        print(f"  VanillaAE:   DISABLED")

    # Move to device
    device = torch.device(config.device)
    model_atlas = model_atlas.to(device)
    if model_std is not None:
        model_std = model_std.to(device)
    if model_ae is not None:
        model_ae = model_ae.to(device)
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    print(f"  Device: {device}")

    # Initialize Jump Operator
    jump_op = FactorizedJumpOperator(
        num_charts=config.num_charts,
        latent_dim=config.latent_dim,
        global_rank=config.jump_global_rank,
    ).to(device)
    print(f"  Jump Operator: {count_parameters(jump_op):,} params")

    # Supervised topology loss
    supervised_loss = None
    num_classes = int(labels.max()) + 1 if labels.size else config.num_classes
    if config.enable_supervised:
        supervised_loss = SupervisedTopologyLoss(
            num_charts=config.num_charts,
            num_classes=num_classes,
            lambda_purity=config.sup_purity_weight,
            lambda_balance=config.sup_balance_weight,
            lambda_metric=config.sup_metric_weight,
            margin=config.sup_metric_margin,
            temperature=config.sup_temperature,
        ).to(device)
        print(
            f"  Supervised Topology: "
            f"classes={num_classes}, "
            f"lambda_purity={config.sup_purity_weight}, "
            f"lambda_balance={config.sup_balance_weight}, "
            f"lambda_metric={config.sup_metric_weight}"
        )

    # Optimizers
    if model_std is not None:
        opt_std = optim.Adam(model_std.parameters(), lr=config.lr)
    atlas_params = list(model_atlas.parameters()) + list(jump_op.parameters())
    if supervised_loss is not None:
        atlas_params.extend(list(supervised_loss.parameters()))
    opt_atlas = optim.Adam(atlas_params, lr=config.lr)
    if model_ae is not None:
        opt_ae = optim.Adam(model_ae.parameters(), lr=config.lr)

    # Learning rate scheduler
    scheduler = None
    if config.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt_atlas, T_max=config.epochs, eta_min=config.min_lr
        )

    # Create data loader
    from torch.utils.data import DataLoader, TensorDataset

    labels_train_t = torch.from_numpy(labels_train).long().to(device)
    labels_test_t = torch.from_numpy(labels_test).long().to(device)
    dataset = TensorDataset(X_train, labels_train_t)
    batch_size = config.batch_size if config.batch_size > 0 else len(X_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training history
    std_losses = []
    atlas_losses = []
    ae_losses = []
    loss_components: dict[str, list[float]] = {
        "recon": [],
        "vq": [],
        "entropy": [],
        "consistency": [],
        "variance": [],
        "diversity": [],
        "separation": [],
        "window": [],
        "disentangle": [],
        "orthogonality": [],
        "code_entropy": [],
        "per_chart_code_entropy": [],
        "kl_prior": [],
        "orbit": [],
        "vicreg_inv": [],
        "jump": [],
        "sup_total": [],
        "sup_route": [],
        "sup_purity": [],
        "sup_balance": [],
        "sup_metric": [],
        "sup_acc": [],
    }
    info_metrics: dict[str, list[float]] = {
        "I_XK": [],
        "H_K": [],
    }

    print("=" * 60)
    print("Training TopoEncoder on CIFAR-10")
    print(f"  Epochs: {config.epochs}, LR: {config.lr}, Batch size: {batch_size}")
    print(f"  Charts: {config.num_charts}, Codes/chart: {config.codes_per_chart}")
    print(f"  lambda: entropy={config.entropy_weight}, consistency={config.consistency_weight}")
    print("=" * 60)

    for epoch in tqdm(range(config.epochs + 1), desc="Training", unit="epoch"):
        epoch_std_loss = 0.0
        epoch_atlas_loss = 0.0
        epoch_ae_loss = 0.0
        epoch_losses = {k: 0.0 for k in loss_components.keys()}
        epoch_info = {"I_XK": 0.0, "H_K": 0.0}
        n_batches = 0

        for batch_X, batch_labels in dataloader:
            n_batches += 1

            # --- Standard VQ Step ---
            loss_s = torch.tensor(0.0, device=device)
            if model_std is not None:
                recon_s, vq_loss_s, _ = model_std(batch_X)
                loss_s = F.mse_loss(recon_s, batch_X) + vq_loss_s
                opt_std.zero_grad()
                loss_s.backward()
                opt_std.step()

            # --- Vanilla AE Step ---
            loss_ae = torch.tensor(0.0, device=device)
            if model_ae is not None:
                recon_ae, _ = model_ae(batch_X)
                loss_ae = F.mse_loss(recon_ae, batch_X)
                opt_ae.zero_grad()
                loss_ae.backward()
                opt_ae.step()

            # --- Atlas Step ---
            K_chart, _, z_n, z_tex, enc_w, z_geo, vq_loss_a, indices_stack, z_n_all_charts = model_atlas.encoder(batch_X)
            recon_a, dec_w = model_atlas.decoder(z_geo, z_tex, chart_index=None)

            # Core losses
            recon_loss_a = F.mse_loss(recon_a, batch_X)
            entropy = compute_routing_entropy(enc_w)
            consistency = model_atlas.compute_consistency_loss(enc_w, dec_w)

            # Tier 1 losses
            var_loss = compute_variance_loss(z_geo)
            div_loss = compute_diversity_loss(enc_w, config.num_charts)
            sep_loss = compute_separation_loss(
                z_geo, enc_w, config.num_charts, config.separation_margin
            )

            # Tier 2 losses
            window_loss, window_info = compute_window_loss(
                enc_w, config.num_charts, config.window_eps_ground
            )
            dis_loss = compute_disentangle_loss(z_geo, enc_w)

            # Tier 3 losses
            orth_loss = compute_orthogonality_loss(model_atlas)
            code_ent_loss = compute_code_entropy_loss(indices_stack, config.codes_per_chart)
            per_chart_code_ent_loss = compute_per_chart_code_entropy_loss(
                indices_stack, K_chart, config.num_charts, config.codes_per_chart
            )

            # Tier 4 losses
            if config.kl_prior_weight > 0:
                kl_loss = compute_kl_prior_loss(z_n, z_tex)
            else:
                kl_loss = torch.tensor(0.0, device=device)

            orbit_loss = torch.tensor(0.0, device=device)
            vicreg_loss = torch.tensor(0.0, device=device)

            if config.orbit_weight > 0 or config.vicreg_inv_weight > 0:
                x_aug = augment_cifar10(batch_X, config.augment_noise_std)
                _, _, _, _, enc_w_aug, z_geo_aug, _, _, _ = model_atlas.encoder(x_aug)

                if config.orbit_weight > 0:
                    orbit_loss = compute_orbit_loss(enc_w, enc_w_aug)
                if config.vicreg_inv_weight > 0:
                    vicreg_loss = compute_vicreg_invariance_loss(z_geo, z_geo_aug)

            # Tier 5: Jump Operator
            current_jump_weight = get_jump_weight_schedule(
                epoch, config.jump_warmup, config.jump_ramp_end, config.jump_weight
            )
            if current_jump_weight > 0:
                jump_loss = compute_jump_consistency_loss(jump_op, z_n_all_charts, enc_w)
            else:
                jump_loss = torch.tensor(0.0, device=device)

            # Supervised topology losses
            sup_total = torch.tensor(0.0, device=device)
            sup_route = torch.tensor(0.0, device=device)
            sup_purity = torch.tensor(0.0, device=device)
            sup_balance = torch.tensor(0.0, device=device)
            sup_metric = torch.tensor(0.0, device=device)
            sup_acc = torch.tensor(0.0, device=device)

            if supervised_loss is not None:
                sup_out = supervised_loss(enc_w, batch_labels, z_geo)
                sup_total = sup_out["loss_total"]
                sup_route = sup_out["loss_route"]
                sup_purity = sup_out["loss_purity"]
                sup_balance = sup_out["loss_balance"]
                sup_metric = sup_out["loss_metric"]

                p_y_x = torch.matmul(enc_w, supervised_loss.p_y_given_k)
                sup_acc = (p_y_x.argmax(dim=1) == batch_labels).float().mean()

            # Total loss
            loss_a = (
                recon_loss_a
                + vq_loss_a
                + config.entropy_weight * entropy
                + config.consistency_weight * consistency
                + config.variance_weight * var_loss
                + config.diversity_weight * div_loss
                + config.separation_weight * sep_loss
                + config.window_weight * window_loss
                + config.disentangle_weight * dis_loss
                + config.orthogonality_weight * orth_loss
                + config.code_entropy_weight * code_ent_loss
                + config.per_chart_code_entropy_weight * per_chart_code_ent_loss
                + config.kl_prior_weight * kl_loss
                + config.orbit_weight * orbit_loss
                + config.vicreg_inv_weight * vicreg_loss
                + current_jump_weight * jump_loss
                + config.sup_weight * sup_total
            )

            opt_atlas.zero_grad()
            loss_a.backward()
            if config.grad_clip > 0:
                all_params = list(model_atlas.parameters()) + list(jump_op.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, config.grad_clip)
            opt_atlas.step()

            # Accumulate batch losses
            epoch_std_loss += loss_s.item()
            epoch_atlas_loss += loss_a.item()
            epoch_ae_loss += loss_ae.item()
            epoch_losses["recon"] += recon_loss_a.item()
            epoch_losses["vq"] += vq_loss_a.item()
            epoch_losses["entropy"] += entropy
            epoch_losses["consistency"] += consistency.item()
            epoch_losses["variance"] += var_loss.item()
            epoch_losses["diversity"] += div_loss.item()
            epoch_losses["separation"] += sep_loss.item()
            epoch_losses["window"] += window_loss.item()
            epoch_losses["disentangle"] += dis_loss.item()
            epoch_losses["orthogonality"] += orth_loss.item()
            epoch_losses["code_entropy"] += code_ent_loss.item()
            epoch_losses["per_chart_code_entropy"] += per_chart_code_ent_loss.item()
            epoch_losses["kl_prior"] += kl_loss.item()
            epoch_losses["orbit"] += orbit_loss.item()
            epoch_losses["vicreg_inv"] += vicreg_loss.item()
            epoch_losses["jump"] += jump_loss.item()
            epoch_losses["sup_total"] += sup_total.item()
            epoch_losses["sup_route"] += sup_route.item()
            epoch_losses["sup_purity"] += sup_purity.item()
            epoch_losses["sup_balance"] += sup_balance.item()
            epoch_losses["sup_metric"] += sup_metric.item()
            epoch_losses["sup_acc"] += sup_acc.item()
            epoch_info["I_XK"] += window_info["I_XK"]
            epoch_info["H_K"] += window_info["H_K"]

        # Average over batches
        std_losses.append(epoch_std_loss / n_batches)
        atlas_losses.append(epoch_atlas_loss / n_batches)
        ae_losses.append(epoch_ae_loss / n_batches)
        for k in loss_components.keys():
            loss_components[k].append(epoch_losses[k] / n_batches)
        info_metrics["I_XK"].append(epoch_info["I_XK"] / n_batches)
        info_metrics["H_K"].append(epoch_info["H_K"] / n_batches)

        # Step LR scheduler
        if scheduler is not None:
            scheduler.step()

        # Logging and visualization
        should_log = epoch % config.log_every == 0 or epoch == config.epochs
        should_save = config.save_every > 0 and (
            epoch % config.save_every == 0 or epoch == config.epochs
        )
        if should_log or should_save:
            with torch.no_grad():
                K_chart_full, _, _, _, enc_w_full, _, _, _, _ = model_atlas.encoder(X_test)
                usage = enc_w_full.mean(dim=0).cpu().numpy()
                chart_assignments = K_chart_full.cpu().numpy()
                ami = compute_ami(labels_test, chart_assignments)
                perplexity = model_atlas.compute_perplexity(K_chart_full)

            avg_loss = atlas_losses[-1]
            avg_recon = loss_components["recon"][-1]
            avg_vq = loss_components["vq"][-1]
            avg_entropy = loss_components["entropy"][-1]
            avg_consistency = loss_components["consistency"][-1]
            avg_sup_acc = loss_components["sup_acc"][-1]
            avg_sup_route = loss_components["sup_route"][-1]
            avg_ixk = info_metrics["I_XK"][-1]
            avg_hk = info_metrics["H_K"][-1]

            log_jump_weight = get_jump_weight_schedule(
                epoch, config.jump_warmup, config.jump_ramp_end, config.jump_weight
            )

            current_lr = scheduler.get_last_lr()[0] if scheduler else config.lr
            print(f"Epoch {epoch:5d} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
            print(f"  Usage: {np.array2string(usage, precision=2, separator=', ')}")
            print(
                f"  Core: recon={avg_recon:.4f} "
                f"vq={avg_vq:.4f} "
                f"entropy={avg_entropy:.4f} "
                f"consistency={avg_consistency:.4f}"
            )
            if supervised_loss is not None:
                with torch.no_grad():
                    _, _, _, _, enc_w_test, z_geo_test, _, _, _ = model_atlas.encoder(X_test)
                    sup_test = supervised_loss(enc_w_test, labels_test_t, z_geo_test)
                    p_y_x_test = torch.matmul(enc_w_test, supervised_loss.p_y_given_k)
                    test_sup_acc = (
                        p_y_x_test.argmax(dim=1) == labels_test_t
                    ).float().mean().item()
                    test_sup_route = sup_test["loss_route"].item()

                print(
                    f"  Sup: train_acc={avg_sup_acc:.4f} "
                    f"test_acc={test_sup_acc:.4f} "
                    f"route={avg_sup_route:.4f}"
                )
            print(
                f"  Info: I(X;K)={avg_ixk:.3f} H(K)={avg_hk:.3f} "
                f"jump_w={log_jump_weight:.3f}"
            )
            print(f"  Metrics: AMI={ami:.4f} perplexity={perplexity:.2f}/{config.num_charts}")
            print("-" * 60)

            # Save visualization
            if should_save:
                save_path = f"{config.output_dir}/cifar10_epoch_{epoch:05d}.png"
                visualize_latent_images(
                    model_atlas,
                    X_test,
                    labels_test,
                    CIFAR10_CLASSES,
                    save_path,
                    epoch,
                    jump_op=jump_op,
                    image_shape=config.image_shape,
                )

    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)

    with torch.no_grad():
        # VanillaAE metrics
        mse_ae = 0.0
        ami_ae = 0.0
        recon_ae_final = None
        if model_ae is not None:
            recon_ae_final, z_ae = model_ae(X_test)
            mse_ae = F.mse_loss(recon_ae_final, X_test).item()
            z_ae_np = z_ae.cpu().numpy()
            kmeans = KMeans(n_clusters=config.num_charts, random_state=42, n_init=10)
            ae_clusters = kmeans.fit_predict(z_ae_np)
            ami_ae = compute_ami(labels_test, ae_clusters)

        # Standard VQ metrics
        mse_std = 0.0
        ami_std = 0.0
        std_perplexity = 0.0
        recon_std_final = None
        if model_std is not None:
            recon_std_final, _, indices_s = model_std(X_test)
            std_perplexity = model_std.compute_perplexity(indices_s)
            mse_std = F.mse_loss(recon_std_final, X_test).item()
            vq_clusters = indices_s.cpu().numpy() % config.num_charts
            ami_std = compute_ami(labels_test, vq_clusters)

        # Atlas metrics
        recon_atlas_final, _, enc_w, dec_w, K_chart = model_atlas(
            X_test, use_hard_routing=False
        )
        chart_assignments = K_chart.cpu().numpy()
        atlas_perplexity = model_atlas.compute_perplexity(K_chart)
        ami_atlas = compute_ami(labels_test, chart_assignments)
        mse_atlas = F.mse_loss(recon_atlas_final, X_test).item()
        final_consistency = model_atlas.compute_consistency_loss(enc_w, dec_w).item()
        sup_acc = 0.0
        if supervised_loss is not None:
            p_y_x = torch.matmul(enc_w, supervised_loss.p_y_given_k)
            sup_acc = (p_y_x.argmax(dim=1) == labels_test_t).float().mean().item()

    # Results table
    print("\n" + "-" * 70)
    print(f"{'Model':<20} {'MSE':>10} {'AMI':>10} {'Perplexity':>15}")
    print("-" * 70)
    if model_ae is not None:
        print(f"{'Vanilla AE':<20} {mse_ae:>10.5f} {ami_ae:>10.4f} {'N/A (K-Means)':<15}")
    if model_std is not None:
        print(f"{'Standard VQ':<20} {mse_std:>10.5f} {ami_std:>10.4f} {std_perplexity:>6.1f}/{config.num_codes_standard:<8}")
    print(f"{'TopoEncoder':<20} {mse_atlas:>10.5f} {ami_atlas:>10.4f} {atlas_perplexity:>6.1f}/{config.num_charts:<8}")
    print("-" * 70)

    print(f"\nRouting Consistency (KL): {final_consistency:.4f}")
    if supervised_loss is not None:
        print(f"Supervised Accuracy: {sup_acc:.4f}")

    # Save final visualization
    if config.save_every > 0:
        final_path = f"{config.output_dir}/cifar10_final.png"
        visualize_latent_images(
            model_atlas,
            X_test,
            labels_test,
            CIFAR10_CLASSES,
            final_path,
            epoch=None,
            jump_op=jump_op,
            image_shape=config.image_shape,
        )
        print(f"\nFinal visualization saved to: {final_path}")

    return {
        "std_losses": std_losses,
        "atlas_losses": atlas_losses,
        "ae_losses": ae_losses,
        "loss_components": loss_components,
        "ami_ae": ami_ae,
        "ami_std": ami_std,
        "ami_atlas": ami_atlas,
        "mse_ae": mse_ae,
        "mse_std": mse_std,
        "mse_atlas": mse_atlas,
        "std_perplexity": std_perplexity,
        "atlas_perplexity": atlas_perplexity,
        "sup_acc": sup_acc,
        "X": X_test,
        "labels": labels_test,
        "colors": colors_test,
        "X_train": X_train,
        "X_test": X_test,
        "labels_train": labels_train,
        "labels_test": labels_test,
        "colors_train": colors_train,
        "colors_test": colors_test,
        "chart_assignments": chart_assignments,
        "recon_ae": recon_ae_final,
        "recon_std": recon_std_final,
        "recon_atlas": recon_atlas_final,
        "model_ae": model_ae,
        "model_std": model_std,
        "model_atlas": model_atlas,
        "config": config,
    }


# ==========================================
# 5. MAIN
# ==========================================
def main():
    """Main entry point for CIFAR-10 benchmark."""
    parser = argparse.ArgumentParser(
        description="TopoEncoder Benchmark: CIFAR-10 Image Classification"
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size"
    )
    parser.add_argument(
        "--n_samples", type=int, default=50000, help="Number of samples"
    )
    parser.add_argument(
        "--test_split", type=float, default=0.2, help="Test split fraction"
    )
    parser.add_argument(
        "--num_charts", type=int, default=10, help="Number of atlas charts"
    )
    parser.add_argument(
        "--codes_per_chart", type=int, default=64, help="VQ codes per chart"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension"
    )
    parser.add_argument(
        "--log_every", type=int, default=50, help="Log every N epochs"
    )
    parser.add_argument(
        "--save_every", type=int, default=50, help="Save visualization every N epochs"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/topoencoder_cifar10", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device"
    )

    # Supervised topology
    parser.add_argument(
        "--disable_supervised", action="store_true", help="Disable supervised losses"
    )
    parser.add_argument(
        "--sup_weight", type=float, default=1.0, help="Supervised loss weight"
    )

    # Benchmark control
    parser.add_argument(
        "--disable_ae", action="store_true", help="Disable VanillaAE baseline"
    )
    parser.add_argument(
        "--disable_vq", action="store_true", help="Disable StandardVQ baseline"
    )

    # Training dynamics
    parser.add_argument(
        "--use_scheduler", type=lambda x: x.lower() == "true", default=True,
        help="Use cosine annealing LR scheduler"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping max norm"
    )

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create config
    config = TopoEncoderCIFAR10Config(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        test_split=args.test_split,
        num_charts=args.num_charts,
        codes_per_chart=args.codes_per_chart,
        hidden_dim=args.hidden_dim,
        log_every=args.log_every,
        save_every=args.save_every,
        output_dir=args.output_dir,
        device=args.device,
        enable_supervised=not args.disable_supervised,
        sup_weight=args.sup_weight,
        disable_ae=args.disable_ae,
        disable_vq=args.disable_vq,
        use_scheduler=args.use_scheduler,
        grad_clip=args.grad_clip,
    )

    print("=" * 50)
    print("TopoEncoder CIFAR-10 Benchmark")
    print("=" * 50)
    print(f"\nConfiguration:")
    print(f"  Epochs: {config.epochs}, Batch size: {config.batch_size}")
    print(f"  Total samples: {config.n_samples}")
    print(f"  Test split: {config.test_split}")
    print(f"  Num charts: {config.num_charts} (one per class)")
    print(f"  Codes per chart: {config.codes_per_chart}")
    print(f"  Total atlas codes: {config.num_charts * config.codes_per_chart}")
    print(f"  Input dim: {config.input_dim} (32x32x3 flattened)")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Output dir: {config.output_dir}")

    # Run benchmark
    results = train_benchmark(config)

    # Save final comparison visualization
    os.makedirs(config.output_dir, exist_ok=True)
    final_path = f"{config.output_dir}/benchmark_result.png"
    visualize_results_images(results, CIFAR10_CLASSES, save_path=final_path)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    ami_atlas = results["ami_atlas"]
    sup_acc = results["sup_acc"]
    if ami_atlas > 0.5:
        print(f"TopoEncoder AMI = {ami_atlas:.4f} - Good chart-class alignment!")
    else:
        print(f"TopoEncoder AMI = {ami_atlas:.4f} - Charts don't align well with classes.")
    if sup_acc > 0.7:
        print(f"Supervised Accuracy = {sup_acc:.4f} - Good classification performance!")
    else:
        print(f"Supervised Accuracy = {sup_acc:.4f} - Classification needs improvement.")
    print(f"\nOutput saved to: {config.output_dir}/")


if __name__ == "__main__":
    main()
