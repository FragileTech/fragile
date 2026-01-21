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
from fragile.core.layers import (
    FactorizedJumpOperator,
    InvariantChartClassifier,
    StandardVQ,
    TopoEncoder,
    VanillaAE,
)
from fragile.core.losses import (
    compute_chart_center_separation_loss,
    compute_codebook_centering_loss,
    compute_code_entropy_loss,
    compute_disentangle_loss,
    compute_diversity_loss,
    compute_jump_consistency_loss,
    compute_kl_prior_loss,
    compute_orthogonality_loss,
    compute_orbit_loss,
    compute_per_chart_code_entropy_loss,
    compute_residual_scale_loss,
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
    soft_equiv_metric: bool = False
    soft_equiv_bundle_size: int | None = None
    soft_equiv_hidden_dim: int = 64
    soft_equiv_use_spectral_norm: bool = True
    soft_equiv_zero_self_mixing: bool = False
    soft_equiv_soft_assign: bool = True
    soft_equiv_temperature: float = 1.0
    soft_equiv_l1_weight: float = 0.0
    soft_equiv_log_ratio_weight: float = 0.0
    vision_preproc: bool = False
    vision_in_channels: int = 3
    vision_height: int = 32
    vision_width: int = 32
    vision_num_rotations: int = 8
    vision_kernel_size: int = 5
    vision_use_reflections: bool = False
    vision_norm_nonlinearity: str = "n_sigmoid"
    vision_norm_bias: bool = True

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
    codebook_center_weight: float = 0.05
    chart_center_sep_weight: float = 0.05
    chart_center_sep_margin: float = 2.0
    residual_scale_weight: float = 0.01

    # Tier 2 losses (medium overhead)
    window_weight: float = 0.5
    window_eps_ground: float = 0.1
    disentangle_weight: float = 0.1

    # Tier 3 losses (geometry/codebook health)
    orthogonality_weight: float = 0.0  # Singular-value spread penalty (SVD; disabled by default)
    code_entropy_weight: float = 0.0
    per_chart_code_entropy_weight: float = 0.1

    # Tier 4 losses (invariance)
    kl_prior_weight: float = 0.01  # Radial energy prior
    orbit_weight: float = 0.0
    vicreg_inv_weight: float = 0.0  # Gram invariance (O(B^2), disabled by default)
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

    # Classifier readout (detached, invariant)
    enable_classifier_head: bool = True
    classifier_lr: float = 0.0  # 0 = use main lr
    classifier_bundle_size: int = 0  # 0 = global norm

    # Learning rate scheduling
    use_scheduler: bool = True
    min_lr: float = 1e-5

    # Gradient clipping
    grad_clip: float = 1.0

    # Benchmark control
    disable_ae: bool = False
    disable_vq: bool = False
    baseline_vision_preproc: bool = False
    baseline_attn: bool = False
    baseline_attn_tokens: int = 4
    baseline_attn_dim: int = 32
    baseline_attn_heads: int = 4
    baseline_attn_dropout: float = 0.0

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
    use_attention: bool = False,
    attn_tokens: int = 4,
    attn_dim: int = 32,
    attn_heads: int = 4,
    attn_dropout: float = 0.0,
    vision_preproc: bool = False,
    vision_in_channels: int = 0,
    vision_height: int = 0,
    vision_width: int = 0,
) -> int:
    """Compute hidden_dim for StandardVQ to match target parameter count."""
    if not use_attention and not vision_preproc:
        offset = 5 + num_codes * latent_dim
        coef_h = 2 * input_dim + 8
        discriminant = coef_h**2 + 8 * (target_params - offset)
        if discriminant < 0:
            return 128
        h = (-coef_h + math.sqrt(discriminant)) / 4
        return max(64, int(h))

    def params_for_hidden(hidden_dim: int) -> int:
        if num_codes > 0:
            model = StandardVQ(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                num_codes=num_codes,
                use_attention=use_attention,
                attn_tokens=attn_tokens,
                attn_dim=attn_dim,
                attn_heads=attn_heads,
                attn_dropout=attn_dropout,
                vision_preproc=vision_preproc,
                vision_in_channels=vision_in_channels,
                vision_height=vision_height,
                vision_width=vision_width,
            )
        else:
            model = VanillaAE(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                use_attention=use_attention,
                attn_tokens=attn_tokens,
                attn_dim=attn_dim,
                attn_heads=attn_heads,
                attn_dropout=attn_dropout,
                vision_preproc=vision_preproc,
                vision_in_channels=vision_in_channels,
                vision_height=vision_height,
                vision_width=vision_width,
            )
        return count_parameters(model)

    min_hidden = 64
    max_hidden = 4096
    low = min_hidden
    low_params = params_for_hidden(low)
    if low_params >= target_params:
        return low
    high = min_hidden * 2
    while high < max_hidden and params_for_hidden(high) < target_params:
        low = high
        high = min(high * 2, max_hidden)

    best_hidden = low
    best_diff = abs(params_for_hidden(low) - target_params)
    while low <= high:
        mid = (low + high) // 2
        params = params_for_hidden(mid)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_hidden = mid
            best_diff = diff
        if params < target_params:
            low = mid + 1
        else:
            high = mid - 1
    return max(min_hidden, int(best_hidden))


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
        vision_preproc=config.vision_preproc,
        vision_in_channels=config.vision_in_channels,
        vision_height=config.vision_height,
        vision_width=config.vision_width,
        vision_num_rotations=config.vision_num_rotations,
        vision_kernel_size=config.vision_kernel_size,
        vision_use_reflections=config.vision_use_reflections,
        vision_norm_nonlinearity=config.vision_norm_nonlinearity,
        vision_norm_bias=config.vision_norm_bias,
        soft_equiv_metric=config.soft_equiv_metric,
        soft_equiv_bundle_size=config.soft_equiv_bundle_size,
        soft_equiv_hidden_dim=config.soft_equiv_hidden_dim,
        soft_equiv_use_spectral_norm=config.soft_equiv_use_spectral_norm,
        soft_equiv_zero_self_mixing=config.soft_equiv_zero_self_mixing,
        soft_equiv_soft_assign=config.soft_equiv_soft_assign,
        soft_equiv_temperature=config.soft_equiv_temperature,
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
            use_attention=config.baseline_attn,
            attn_tokens=config.baseline_attn_tokens,
            attn_dim=config.baseline_attn_dim,
            attn_heads=config.baseline_attn_heads,
            attn_dropout=config.baseline_attn_dropout,
            vision_preproc=config.baseline_vision_preproc,
            vision_in_channels=config.vision_in_channels,
            vision_height=config.vision_height,
            vision_width=config.vision_width,
        )
        model_std = StandardVQ(
            input_dim=config.input_dim,
            hidden_dim=std_hidden_dim,
            latent_dim=config.latent_dim,
            num_codes=config.num_codes_standard,
            use_attention=config.baseline_attn,
            attn_tokens=config.baseline_attn_tokens,
            attn_dim=config.baseline_attn_dim,
            attn_heads=config.baseline_attn_heads,
            attn_dropout=config.baseline_attn_dropout,
            vision_preproc=config.baseline_vision_preproc,
            vision_in_channels=config.vision_in_channels,
            vision_height=config.vision_height,
            vision_width=config.vision_width,
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
            use_attention=config.baseline_attn,
            attn_tokens=config.baseline_attn_tokens,
            attn_dim=config.baseline_attn_dim,
            attn_heads=config.baseline_attn_heads,
            attn_dropout=config.baseline_attn_dropout,
            vision_preproc=config.baseline_vision_preproc,
            vision_in_channels=config.vision_in_channels,
            vision_height=config.vision_height,
            vision_width=config.vision_width,
        )
        model_ae = VanillaAE(
            input_dim=config.input_dim,
            hidden_dim=ae_hidden_dim,
            latent_dim=config.latent_dim,
            use_attention=config.baseline_attn,
            attn_tokens=config.baseline_attn_tokens,
            attn_dim=config.baseline_attn_dim,
            attn_heads=config.baseline_attn_heads,
            attn_dropout=config.baseline_attn_dropout,
            vision_preproc=config.baseline_vision_preproc,
            vision_in_channels=config.vision_in_channels,
            vision_height=config.vision_height,
            vision_width=config.vision_width,
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

    # Invariant classifier head (detached readout)
    classifier_head = None
    classifier_bundle_size = config.classifier_bundle_size or None
    classifier_lr = config.classifier_lr if config.classifier_lr > 0 else config.lr
    if config.enable_classifier_head:
        classifier_head = InvariantChartClassifier(
            num_charts=config.num_charts,
            num_classes=num_classes,
            latent_dim=config.latent_dim,
            bundle_size=classifier_bundle_size,
        ).to(device)
        print(
            "  Classifier Readout: "
            f"classes={num_classes}, "
            f"bundle_size={classifier_bundle_size or 'global'}, "
            f"lr={classifier_lr}"
        )

    # Optimizers
    if model_std is not None:
        opt_std = optim.Adam(model_std.parameters(), lr=config.lr)
    atlas_params = list(model_atlas.parameters()) + list(jump_op.parameters())
    if supervised_loss is not None:
        atlas_params.extend(list(supervised_loss.parameters()))
    opt_atlas = optim.Adam(atlas_params, lr=config.lr)
    opt_classifier = None
    if classifier_head is not None:
        opt_classifier = optim.Adam(classifier_head.parameters(), lr=classifier_lr)
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
        "codebook_center": [],
        "chart_center_sep": [],
        "residual_scale": [],
        "soft_equiv_l1": [],
        "soft_equiv_log_ratio": [],
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
        "cls_loss": [],
        "cls_acc": [],
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
            K_chart, _, z_n, z_tex, enc_w, z_geo, vq_loss_a, indices_stack, z_n_all_charts, _c_bar = model_atlas.encoder(batch_X)
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
            codebook_center_loss = torch.tensor(0.0, device=device)
            if config.codebook_center_weight > 0:
                codebook_center_loss = compute_codebook_centering_loss(
                    model_atlas.encoder.codebook
                )
            chart_center_sep_loss = torch.tensor(0.0, device=device)
            if config.chart_center_sep_weight > 0:
                chart_center_sep_loss = compute_chart_center_separation_loss(
                    model_atlas.encoder.chart_centers, config.chart_center_sep_margin
                )
            residual_scale_loss = torch.tensor(0.0, device=device)
            if config.residual_scale_weight > 0:
                residual_scale_loss = compute_residual_scale_loss(z_n)
            soft_equiv_l1 = torch.tensor(0.0, device=device)
            if config.soft_equiv_metric:
                soft_equiv_l1 = model_atlas.encoder.soft_equiv_l1_loss()
            soft_equiv_log_ratio = torch.tensor(0.0, device=device)
            if config.soft_equiv_metric and config.soft_equiv_log_ratio_weight > 0:
                soft_equiv_log_ratio = model_atlas.encoder.soft_equiv_log_ratio_loss()

            # Tier 2 losses
            window_loss, window_info = compute_window_loss(
                enc_w, config.num_charts, config.window_eps_ground
            )
            dis_loss = compute_disentangle_loss(z_geo, enc_w)

            # Tier 3 losses
            orth_loss = torch.tensor(0.0, device=device)
            if config.orthogonality_weight > 0:
                orth_loss = compute_orthogonality_loss(model_atlas)

            code_ent_loss = torch.tensor(0.0, device=device)
            if config.code_entropy_weight > 0:
                code_ent_loss = compute_code_entropy_loss(indices_stack, config.codes_per_chart)

            per_chart_code_ent_loss = torch.tensor(0.0, device=device)
            if config.per_chart_code_entropy_weight > 0:
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
                _, _, _, _, enc_w_aug, z_geo_aug, _, _, _, _c_bar_aug = model_atlas.encoder(x_aug)
                del x_aug  # Free memory immediately

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
                + config.codebook_center_weight * codebook_center_loss
                + config.chart_center_sep_weight * chart_center_sep_loss
                + config.residual_scale_weight * residual_scale_loss
                + config.soft_equiv_l1_weight * soft_equiv_l1
                + config.soft_equiv_log_ratio_weight * soft_equiv_log_ratio
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

            # --- Classifier Readout Step (detached) ---
            cls_loss = torch.tensor(0.0, device=device)
            cls_acc = torch.tensor(0.0, device=device)
            if classifier_head is not None and opt_classifier is not None:
                logits = classifier_head(enc_w.detach(), z_geo.detach())
                cls_loss = F.cross_entropy(logits, batch_labels)
                opt_classifier.zero_grad()
                cls_loss.backward()
                opt_classifier.step()
                cls_acc = (
                    logits.detach().argmax(dim=1) == batch_labels
                ).float().mean()

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
            epoch_losses["codebook_center"] += codebook_center_loss.item()
            epoch_losses["chart_center_sep"] += chart_center_sep_loss.item()
            epoch_losses["residual_scale"] += residual_scale_loss.item()
            epoch_losses["soft_equiv_l1"] += soft_equiv_l1.item()
            epoch_losses["soft_equiv_log_ratio"] += soft_equiv_log_ratio.item()
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
            epoch_losses["cls_loss"] += cls_loss.item()
            epoch_losses["cls_acc"] += cls_acc.item()
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
            was_training = model_atlas.training
            model_atlas.eval()
            with torch.no_grad():
                K_chart_full, _, _, _, enc_w_full, _, _, _, _, _c_bar_full = model_atlas.encoder(X_test)
                usage = enc_w_full.mean(dim=0).cpu().numpy()
                chart_assignments = K_chart_full.cpu().numpy()
                ami = compute_ami(labels_test, chart_assignments)
                perplexity = model_atlas.compute_perplexity(K_chart_full)
            if was_training:
                model_atlas.train()

            # Clear GPU cache after heavy test inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            avg_loss = atlas_losses[-1]
            avg_recon = loss_components["recon"][-1]
            avg_vq = loss_components["vq"][-1]
            avg_entropy = loss_components["entropy"][-1]
            avg_consistency = loss_components["consistency"][-1]
            avg_sup_acc = loss_components["sup_acc"][-1]
            avg_sup_route = loss_components["sup_route"][-1]
            avg_cls_loss = loss_components["cls_loss"][-1]
            avg_cls_acc = loss_components["cls_acc"][-1]
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
            enc_w_test = None
            z_geo_test = None
            if supervised_loss is not None or classifier_head is not None:
                was_training = model_atlas.training
                model_atlas.eval()
                with torch.no_grad():
                    _, _, _, _, enc_w_test, z_geo_test, _, _, _, _c_bar_test = model_atlas.encoder(X_test)
                if was_training:
                    model_atlas.train()

            if supervised_loss is not None and enc_w_test is not None:
                with torch.no_grad():
                    sup_test = supervised_loss(enc_w_test, labels_test_t, z_geo_test)
                    p_y_x_test = torch.matmul(enc_w_test, supervised_loss.p_y_given_k)
                    test_sup_acc = (
                        p_y_x_test.argmax(dim=1) == labels_test_t
                    ).float().mean().item()
                    test_sup_route = sup_test["loss_route"].item()

                # Clear GPU cache after supervised test inference
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(
                    f"  Sup: train_acc={avg_sup_acc:.4f} "
                    f"test_acc={test_sup_acc:.4f} "
                    f"route={avg_sup_route:.4f}"
                )

            if classifier_head is not None and enc_w_test is not None:
                with torch.no_grad():
                    cls_logits_test = classifier_head(enc_w_test, z_geo_test)
                    test_cls_acc = (
                        cls_logits_test.argmax(dim=1) == labels_test_t
                    ).float().mean().item()

                print(
                    f"  Readout: train_loss={avg_cls_loss:.4f} "
                    f"train_acc={avg_cls_acc:.4f} "
                    f"test_acc={test_cls_acc:.4f}"
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

    if model_ae is not None:
        model_ae.eval()
    if model_std is not None:
        model_std.eval()
    model_atlas.eval()

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
        recon_atlas_final, _, enc_w, dec_w, K_chart, _z_geo, _z_n, _c_bar = model_atlas(
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
        cls_acc = 0.0
        if classifier_head is not None:
            _, _, _, _, enc_w_cls, z_geo_cls, _, _, _, _c_bar_cls = model_atlas.encoder(X_test)
            cls_logits = classifier_head(enc_w_cls, z_geo_cls)
            cls_acc = (cls_logits.argmax(dim=1) == labels_test_t).float().mean().item()

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
    if classifier_head is not None:
        print(f"Readout Accuracy: {cls_acc:.4f}")

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
        "cls_acc": cls_acc,
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
        "--vision_preproc",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use covariant vision preprocessor (default: False)",
    )
    parser.add_argument(
        "--vision_num_rotations",
        type=int,
        default=8,
        help="Vision preproc rotation count (default: 8)",
    )
    parser.add_argument(
        "--vision_kernel_size",
        type=int,
        default=5,
        help="Vision preproc kernel size (default: 5)",
    )
    parser.add_argument(
        "--vision_use_reflections",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use reflections in vision preproc (default: False)",
    )
    parser.add_argument(
        "--vision_norm_nonlinearity",
        type=str,
        default="n_sigmoid",
        help="Vision preproc norm nonlinearity (default: n_sigmoid)",
    )
    parser.add_argument(
        "--vision_norm_bias",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Vision preproc norm bias (default: True)",
    )
    parser.add_argument(
        "--soft_equiv_metric",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable soft equivariant metric for VQ distances (default: False)",
    )
    parser.add_argument(
        "--soft_equiv_bundle_size",
        type=int,
        default=0,
        help="Bundle size for soft equivariant metric (0 = latent_dim)",
    )
    parser.add_argument(
        "--soft_equiv_hidden_dim",
        type=int,
        default=64,
        help="Hidden dim for soft equivariant metric (default: 64)",
    )
    parser.add_argument(
        "--soft_equiv_use_spectral_norm",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use spectral norm in soft equivariant layer (default: True)",
    )
    parser.add_argument(
        "--soft_equiv_zero_self_mixing",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Zero self-mixing in soft equivariant layer (default: False)",
    )
    parser.add_argument(
        "--soft_equiv_soft_assign",
        type=lambda x: x.lower() == "true",
        default=True,
        help=(
            "Use straight-through soft assignment for soft equivariant metric "
            "(default: True)"
        ),
    )
    parser.add_argument(
        "--soft_equiv_temperature",
        type=float,
        default=1.0,
        help="Soft assignment temperature for soft equivariant metric (default: 1.0)",
    )
    parser.add_argument(
        "--soft_equiv_l1_weight",
        type=float,
        default=0.0,
        help="L1 weight for soft equivariant mixing (default: 0.0)",
    )
    parser.add_argument(
        "--soft_equiv_log_ratio_weight",
        type=float,
        default=0.0,
        help="Log-ratio weight for soft equivariant metric (default: 0.0)",
    )
    parser.add_argument(
        "--baseline_vision_preproc",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable standard vision backbone in baselines (default: False)",
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

    # Tier 1 losses (residual hierarchy)
    parser.add_argument(
        "--codebook_center_weight", type=float, default=0.05,
        help="Codebook centering weight (default: 0.05)"
    )
    parser.add_argument(
        "--chart_center_sep_weight", type=float, default=0.05,
        help="Chart center separation weight (default: 0.05)"
    )
    parser.add_argument(
        "--chart_center_sep_margin", type=float, default=2.0,
        help="Chart center separation margin (default: 2.0)"
    )
    parser.add_argument(
        "--residual_scale_weight", type=float, default=0.01,
        help="Residual scale penalty weight (default: 0.01)"
    )

    # Tier 3 losses (codebook health)
    parser.add_argument(
        "--per_chart_code_entropy_weight", type=float, default=0.1,
        help="Per-chart code entropy weight (default: 0.1)"
    )
    parser.add_argument(
        "--code_entropy_weight", type=float, default=0.0,
        help="Global code entropy weight (default: 0.0)"
    )

    # Tier 4 losses (invariance)
    parser.add_argument(
        "--kl_prior_weight", type=float, default=0.01,
        help="KL prior weight on z_n, z_tex (default: 0.01)"
    )
    parser.add_argument(
        "--orbit_weight", type=float, default=0.0,
        help="Orbit invariance weight (default: 0.0)"
    )
    parser.add_argument(
        "--vicreg_inv_weight", type=float, default=0.0,
        help="VICReg invariance weight (default: 0.0)"
    )
    parser.add_argument(
        "--augment_noise_std", type=float, default=0.1,
        help="Augmentation noise std (default: 0.1)"
    )

    # Tier 5: Jump Operator
    parser.add_argument(
        "--jump_weight", type=float, default=0.1,
        help="Jump consistency weight after warmup (default: 0.1)"
    )
    parser.add_argument(
        "--jump_warmup", type=int, default=50,
        help="Epochs before jump loss starts (default: 50)"
    )
    parser.add_argument(
        "--jump_ramp_end", type=int, default=100,
        help="Epoch when jump weight reaches final value (default: 100)"
    )
    parser.add_argument(
        "--jump_global_rank", type=int, default=0,
        help="Rank of global tangent space (0 = use latent_dim, default: 0)"
    )

    # Supervised topology loss
    parser.add_argument(
        "--disable_supervised", action="store_true", help="Disable supervised losses"
    )
    parser.add_argument(
        "--sup_weight", type=float, default=1.0, help="Supervised loss weight"
    )
    parser.add_argument(
        "--sup_purity_weight", type=float, default=0.1,
        help="Supervised purity loss weight (default: 0.1)"
    )
    parser.add_argument(
        "--sup_balance_weight", type=float, default=0.01,
        help="Supervised balance loss weight (default: 0.01)"
    )
    parser.add_argument(
        "--sup_metric_weight", type=float, default=0.01,
        help="Supervised metric loss weight (default: 0.01)"
    )
    parser.add_argument(
        "--sup_metric_margin", type=float, default=1.0,
        help="Supervised metric loss margin (default: 1.0)"
    )
    parser.add_argument(
        "--sup_temperature", type=float, default=1.0,
        help="Temperature for chart-to-class mapping (default: 1.0)"
    )

    # Classifier readout (detached)
    parser.add_argument(
        "--disable_classifier_head",
        action="store_true",
        help="Disable invariant classifier readout head",
    )
    parser.add_argument(
        "--classifier_lr",
        type=float,
        default=0.0,
        help="Classifier head learning rate (0 = use main lr)",
    )
    parser.add_argument(
        "--classifier_bundle_size",
        type=int,
        default=0,
        help="Bundle size for radial readout (0 = global norm)",
    )

    # Benchmark control
    parser.add_argument(
        "--disable_ae", action="store_true", help="Disable VanillaAE baseline"
    )
    parser.add_argument(
        "--disable_vq", action="store_true", help="Disable StandardVQ baseline"
    )
    parser.add_argument(
        "--baseline_attn",
        action="store_true",
        help="Enable attention blocks in StandardVQ/VanillaAE baselines",
    )
    parser.add_argument(
        "--baseline_attn_tokens",
        type=int,
        default=4,
        help="Number of tokens for baseline attention blocks",
    )
    parser.add_argument(
        "--baseline_attn_dim",
        type=int,
        default=32,
        help="Per-token attention dimension for baselines",
    )
    parser.add_argument(
        "--baseline_attn_heads",
        type=int,
        default=4,
        help="Number of attention heads for baselines",
    )
    parser.add_argument(
        "--baseline_attn_dropout",
        type=float,
        default=0.0,
        help="Attention dropout for baselines",
    )

    # Training dynamics
    parser.add_argument(
        "--use_scheduler", type=lambda x: x.lower() == "true", default=True,
        help="Use cosine annealing LR scheduler"
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-5,
        help="Minimum LR for scheduler (default: 1e-5)"
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
        vision_preproc=args.vision_preproc,
        vision_num_rotations=args.vision_num_rotations,
        vision_kernel_size=args.vision_kernel_size,
        vision_use_reflections=args.vision_use_reflections,
        vision_norm_nonlinearity=args.vision_norm_nonlinearity,
        vision_norm_bias=args.vision_norm_bias,
        soft_equiv_metric=args.soft_equiv_metric,
        soft_equiv_bundle_size=args.soft_equiv_bundle_size or None,
        soft_equiv_hidden_dim=args.soft_equiv_hidden_dim,
        soft_equiv_use_spectral_norm=args.soft_equiv_use_spectral_norm,
        soft_equiv_zero_self_mixing=args.soft_equiv_zero_self_mixing,
        soft_equiv_soft_assign=args.soft_equiv_soft_assign,
        soft_equiv_temperature=args.soft_equiv_temperature,
        soft_equiv_l1_weight=args.soft_equiv_l1_weight,
        soft_equiv_log_ratio_weight=args.soft_equiv_log_ratio_weight,
        baseline_vision_preproc=args.baseline_vision_preproc,
        log_every=args.log_every,
        save_every=args.save_every,
        output_dir=args.output_dir,
        device=args.device,
        # Tier 1 losses
        codebook_center_weight=args.codebook_center_weight,
        chart_center_sep_weight=args.chart_center_sep_weight,
        chart_center_sep_margin=args.chart_center_sep_margin,
        residual_scale_weight=args.residual_scale_weight,
        # Tier 3 losses
        per_chart_code_entropy_weight=args.per_chart_code_entropy_weight,
        code_entropy_weight=args.code_entropy_weight,
        # Tier 4 losses
        kl_prior_weight=args.kl_prior_weight,
        orbit_weight=args.orbit_weight,
        vicreg_inv_weight=args.vicreg_inv_weight,
        augment_noise_std=args.augment_noise_std,
        # Tier 5: Jump Operator
        jump_weight=args.jump_weight,
        jump_warmup=args.jump_warmup,
        jump_ramp_end=args.jump_ramp_end,
        jump_global_rank=args.jump_global_rank,
        # Supervised topology loss
        enable_supervised=not args.disable_supervised,
        sup_weight=args.sup_weight,
        sup_purity_weight=args.sup_purity_weight,
        sup_balance_weight=args.sup_balance_weight,
        sup_metric_weight=args.sup_metric_weight,
        sup_metric_margin=args.sup_metric_margin,
        sup_temperature=args.sup_temperature,
        enable_classifier_head=not args.disable_classifier_head,
        classifier_lr=args.classifier_lr,
        classifier_bundle_size=args.classifier_bundle_size,
        # Benchmark control
        disable_ae=args.disable_ae,
        disable_vq=args.disable_vq,
        baseline_attn=args.baseline_attn,
        baseline_attn_tokens=args.baseline_attn_tokens,
        baseline_attn_dim=args.baseline_attn_dim,
        baseline_attn_heads=args.baseline_attn_heads,
        baseline_attn_dropout=args.baseline_attn_dropout,
        # Training dynamics
        use_scheduler=args.use_scheduler,
        min_lr=args.min_lr,
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
