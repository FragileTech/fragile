"""
TopoEncoder Benchmark: Attentive Atlas vs Standard VQ-VAE

This script benchmarks the Attentive Atlas architecture (from fragile-index.md Section 7.8)
against a standard VQ-VAE on the "Manifold Mixture" problem.

The Manifold Mixture consists of three distinct geometric shapes:
1. Swiss Roll (flat curvature, rolled up)
2. Circles (topological loop)
3. Moons (discontinuous clusters)

Key architectural components (from mermaid diagram):
- Cross-attention router with learnable chart query bank
- Local VQ codebooks per chart
- Recursive decomposition: delta_total → (z_n, z_tex)
- TopologicalDecoder (inverse atlas) from Section 7.10

Metrics reported:
- Convergence speed (MSE vs epoch)
- Topological accuracy (AMI between ground truth and learned charts)
- Codebook usage (perplexity)

Usage:
    python topoencoder.py [--epochs 1000] [--n_samples 3000]

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

from dataviz import visualize_latent, visualize_results
from fragile.datasets import (
    compute_chart_colors,
    find_boundary_pairs,
    get_nightmare_data,
)
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
class TopoEncoderConfig:
    """Configuration for the TopoEncoder benchmark."""

    # Data (using 3D nightmare dataset: Swiss Roll + Sphere + Moons)
    n_samples: int = 3000  # Total samples (divided by 3 per manifold)
    input_dim: int = 3  # 3D input for nightmare dataset

    # Model architecture
    hidden_dim: int = 32
    latent_dim: int = 2  # For 2D visualization
    num_charts: int = 3  # Match number of manifolds
    codes_per_chart: int = 32  # Better coverage (was 21)
    num_codes_standard: int = 64

    # Training
    epochs: int = 1000
    batch_size: int = 256  # Batch size for training (0 = full batch)
    lr: float = 1e-3
    vq_commitment_cost: float = 0.25
    entropy_weight: float = 0.1  # Encourage sharp routing (was 0.01)
    consistency_weight: float = 0.1  # Align encoder/decoder routing

    # Tier 1 losses (low overhead ~5%)
    variance_weight: float = 0.1  # Prevent latent collapse
    diversity_weight: float = 0.1  # Prevent chart collapse (was 1.0)
    separation_weight: float = 0.1  # Force chart centers apart (was 0.5)
    separation_margin: float = 2.0  # Minimum distance between chart centers

    # Tier 2 losses (medium overhead ~5%)
    window_weight: float = 0.5  # Information-stability (Theorem 15.1.3)
    window_eps_ground: float = 0.1  # Minimum I(X;K) threshold
    disentangle_weight: float = 0.1  # Gauge coherence (K ⊥ z_n)

    # Tier 3 losses (geometry/codebook health)
    orthogonality_weight: float = 0.01  # Metric isometry (W^T W ≈ I)
    code_entropy_weight: float = 0.0  # Global code entropy (disabled by default)
    per_chart_code_entropy_weight: float = 0.1  # Per-chart code diversity (enabled)

    # Tier 4 losses (invariance - expensive when enabled, disabled by default)
    kl_prior_weight: float = 0.01  # Residual KL prior on z_n, z_tex
    orbit_weight: float = 0.0  # Chart invariance under augmentation (2x slowdown)
    vicreg_inv_weight: float = 0.0  # Latent invariance (shares augmentation pass)
    augment_noise_std: float = 0.1  # Augmentation noise level
    augment_rotation_max: float = 0.3  # Max rotation in radians

    # Tier 5: Jump Operator (chart gluing - learns transition functions between charts)
    jump_weight: float = 0.1  # Final jump consistency weight after warmup
    jump_warmup: int = 50  # Epochs before jump loss starts (let atlas form first)
    jump_ramp_end: int = 100  # Epoch when jump weight reaches final value
    jump_global_rank: int = 0  # Rank of global tangent space (0 = use latent_dim)

    # Supervised topology loss (Section 7.12)
    enable_supervised: bool = True
    num_classes: int = 3
    sup_weight: float = 1.0
    sup_purity_weight: float = 0.1
    sup_balance_weight: float = 0.01
    sup_metric_weight: float = 0.01
    sup_metric_margin: float = 1.0
    sup_temperature: float = 1.0

    # Learning rate scheduling
    use_scheduler: bool = True  # Use cosine annealing LR scheduler
    min_lr: float = 1e-5  # Minimum LR at end of schedule

    # Gradient clipping
    grad_clip: float = 1.0  # Max gradient norm (0 to disable)

    # Benchmark control
    disable_ae: bool = False  # Skip VanillaAE baseline
    disable_vq: bool = False  # Skip StandardVQ baseline

    # Train/test split
    test_split: float = 0.2

    # Logging and output
    log_every: int = 100
    save_every: int = 100  # Save visualization every N epochs (0 to disable)
    output_dir: str = "outputs/topoencoder"

    # Device (CUDA if available, else CPU)
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def compute_matching_hidden_dim(
    target_params: int,
    input_dim: int = 3,
    latent_dim: int = 2,
    num_codes: int = 64,
) -> int:
    """Compute hidden_dim for StandardVQ to match target parameter count.

    StandardVQ params = 2h² + (4 + 2 + 3 + 3)h + (2 + num_codes*latent_dim + 3)
                      = 2h² + 12h + (5 + num_codes*latent_dim)

    Using quadratic formula: h = (-12 + sqrt(144 + 8*(target - offset))) / 4
    """
    offset = 5 + num_codes * latent_dim
    # Adjust for input_dim: encoder.0 has input_dim*h, decoder.4 has h*input_dim
    # Full formula: 2h² + (input_dim + 2 + 2 + input_dim + 2 + 2)h + ...
    #             = 2h² + (2*input_dim + 8)h + offset
    coef_h = 2 * input_dim + 8
    # 2h² + coef_h*h + offset = target
    # h = (-coef_h + sqrt(coef_h² + 8*(target - offset))) / 4
    discriminant = coef_h**2 + 8 * (target_params - offset)
    if discriminant < 0:
        return 32  # fallback
    h = (-coef_h + math.sqrt(discriminant)) / 4
    return max(16, int(h))


# ==========================================
# 7. METRICS
# ==========================================
def compute_ami(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute Adjusted Mutual Information score."""
    return float(adjusted_mutual_info_score(labels_true, labels_pred))


# ==========================================
# 8. AUGMENTATION (for invariance losses)
# ==========================================
def augment_nightmare(
    x: torch.Tensor,
    noise_std: float = 0.1,
    rotation_max: float = 0.3,
) -> torch.Tensor:
    """Apply random rotation + noise to 3D nightmare data.

    Augmentations preserve manifold identity but change local position.
    Used for orbit invariance and VICReg invariance losses.

    Args:
        x: Input tensor [B, 3] (3D points)
        noise_std: Standard deviation of additive noise
        rotation_max: Maximum rotation angle in radians (±)

    Returns:
        Augmented tensor [B, 3]
    """
    B = x.shape[0]
    device = x.device

    # Random rotation around Z-axis (preserves manifold structure)
    theta = torch.rand(B, device=device) * 2 * rotation_max - rotation_max
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # Apply rotation (Z-axis): only affects X and Y
    x_rot = x.clone()
    x_rot[:, 0] = cos_t * x[:, 0] - sin_t * x[:, 1]
    x_rot[:, 1] = sin_t * x[:, 0] + cos_t * x[:, 1]
    # x_rot[:, 2] unchanged (Z-axis)

    # Add small noise
    x_aug = x_rot + torch.randn_like(x) * noise_std

    return x_aug


# ==========================================
# 9. TRAINING
# ==========================================
def train_benchmark(config: TopoEncoderConfig) -> dict:
    """Train both models and return results.

    Returns dictionary with:
        - std_losses: List of StandardVQ losses per epoch
        - atlas_losses: List of TopoEncoder losses per epoch
        - ami_score: Final AMI for TopoEncoder
        - std_perplexity: Final perplexity for StandardVQ
        - atlas_perplexity: Final perplexity for TopoEncoder
        - X: Evaluation data
        - labels: Ground truth labels for eval
        - chart_assignments: Learned chart assignments
    """
    # Create output directory
    if config.save_every > 0:
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"Saving training progress to: {config.output_dir}/")

    # Generate data (3D nightmare dataset with rainbow colors)
    X, labels, colors = get_nightmare_data(config.n_samples)
    dataset_ids = {
        "swiss_roll": 0,
        "sphere": 1,
        "moons": 2,
    }
    labels = labels.astype(np.int64)
    print(f"Generated {len(X)} points from 3 manifolds (Swiss Roll, Sphere, Moons)")
    print(
        "Dataset IDs: swiss_roll=0, sphere=1, moons=2"
    )
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

    # Create TopoEncoder first to get its parameter count
    model_atlas = TopoEncoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_charts=config.num_charts,
        codes_per_chart=config.codes_per_chart,
    )
    topo_params = count_parameters(model_atlas)

    # Create StandardVQ with matching parameter count (fair comparison)
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

    # Create VanillaAE with similar parameter count (reconstruction baseline)
    model_ae = None
    opt_ae = None
    ae_params = 0
    ae_hidden_dim = 0
    if not config.disable_ae:
        ae_hidden_dim = compute_matching_hidden_dim(
            target_params=topo_params,
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            num_codes=0,  # No codebook in AE
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

    # Move models and data to device
    device = torch.device(config.device)
    model_atlas = model_atlas.to(device)
    if model_std is not None:
        model_std = model_std.to(device)
    if model_ae is not None:
        model_ae = model_ae.to(device)
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    print(f"  Device: {device}")

    # Initialize Jump Operator for chart gluing
    jump_op = FactorizedJumpOperator(
        num_charts=config.num_charts,
        latent_dim=config.latent_dim,
        global_rank=config.jump_global_rank,
    ).to(device)
    print(f"  Jump Operator: {count_parameters(jump_op):,} params")

    # Supervised topology loss (chart-to-class mapping)
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
            "  Supervised Topology: "
            f"classes={num_classes}, "
            f"λ_purity={config.sup_purity_weight}, "
            f"λ_balance={config.sup_balance_weight}, "
            f"λ_metric={config.sup_metric_weight}"
        )

    # Optimizers (joint training of atlas model and jump operator)
    if model_std is not None:
        opt_std = optim.Adam(model_std.parameters(), lr=config.lr)
    atlas_params = list(model_atlas.parameters()) + list(jump_op.parameters())
    if supervised_loss is not None:
        atlas_params.extend(list(supervised_loss.parameters()))
    opt_atlas = optim.Adam(atlas_params, lr=config.lr)
    if model_ae is not None:
        opt_ae = optim.Adam(model_ae.parameters(), lr=config.lr)

    # Learning rate scheduler (cosine annealing)
    scheduler = None
    if config.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt_atlas, T_max=config.epochs, eta_min=config.min_lr
        )

    # Create data loader for minibatching (data already on device)
    from torch.utils.data import DataLoader, TensorDataset
    labels_train_t = torch.from_numpy(labels_train).long().to(device)
    labels_test_t = torch.from_numpy(labels_test).long().to(device)
    colors_train_t = torch.from_numpy(colors_train).float().to(device)
    colors_test_t = torch.from_numpy(colors_test).float().to(device)
    dataset = TensorDataset(X_train, labels_train_t, colors_train_t)
    batch_size = config.batch_size if config.batch_size > 0 else len(X_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training history
    std_losses = []
    atlas_losses = []
    ae_losses = []  # VanillaAE baseline
    loss_components: dict[str, list[float]] = {
        "recon": [],
        "vq": [],
        "entropy": [],
        "consistency": [],
        # Tier 1 losses
        "variance": [],
        "diversity": [],
        "separation": [],
        # Tier 2 losses
        "window": [],
        "disentangle": [],
        # Tier 3 losses
        "orthogonality": [],
        "code_entropy": [],
        "per_chart_code_entropy": [],
        # Tier 4 losses (conditional)
        "kl_prior": [],
        "orbit": [],
        "vicreg_inv": [],
        # Tier 5: Jump Operator
        "jump": [],
        # Supervised topology
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
    print("Training TopoEncoder (Attentive Atlas)")
    print(f"  Epochs: {config.epochs}, LR: {config.lr}, Batch size: {batch_size}")
    print(f"  Charts: {config.num_charts}, Codes/chart: {config.codes_per_chart}")
    print(f"  λ: entropy={config.entropy_weight}, consistency={config.consistency_weight}")
    print("=" * 60)

    for epoch in tqdm(range(config.epochs + 1), desc="Training", unit="epoch"):
        # Accumulate batch losses for epoch average
        epoch_std_loss = 0.0
        epoch_atlas_loss = 0.0
        epoch_ae_loss = 0.0
        epoch_losses = {k: 0.0 for k in loss_components.keys()}
        epoch_info = {"I_XK": 0.0, "H_K": 0.0}
        n_batches = 0

        for batch_X, batch_labels, _batch_colors in dataloader:
            n_batches += 1

            # --- Standard VQ Step ---
            loss_s = torch.tensor(0.0, device=device)
            if model_std is not None:
                recon_s, vq_loss_s, _ = model_std(batch_X)
                loss_s = F.mse_loss(recon_s, batch_X) + vq_loss_s
                opt_std.zero_grad()
                loss_s.backward()
                opt_std.step()

            # --- Vanilla AE Step (reconstruction baseline) ---
            loss_ae = torch.tensor(0.0, device=device)
            if model_ae is not None:
                recon_ae, _ = model_ae(batch_X)
                loss_ae = F.mse_loss(recon_ae, batch_X)
                opt_ae.zero_grad()
                loss_ae.backward()
                opt_ae.step()

            # --- Atlas Step (dreaming mode: decoder infers routing from z_geo) ---
            # Get encoder outputs (need z_geo for regularization losses, z_n_all_charts for jump)
            K_chart, _, z_n, z_tex, enc_w, z_geo, vq_loss_a, indices_stack, z_n_all_charts = model_atlas.encoder(batch_X)

            # Decoder forward (dreaming mode - infers routing from z_geo)
            recon_a, dec_w = model_atlas.decoder(z_geo, z_tex, chart_index=None)

            # Core losses
            recon_loss_a = F.mse_loss(recon_a, batch_X)
            entropy = compute_routing_entropy(enc_w)
            consistency = model_atlas.compute_consistency_loss(enc_w, dec_w)

            # Tier 1 losses (low overhead)
            var_loss = compute_variance_loss(z_geo)
            div_loss = compute_diversity_loss(enc_w, config.num_charts)
            sep_loss = compute_separation_loss(
                z_geo, enc_w, config.num_charts, config.separation_margin
            )

            # Tier 2 losses (medium overhead)
            window_loss, window_info = compute_window_loss(
                enc_w, config.num_charts, config.window_eps_ground
            )
            dis_loss = compute_disentangle_loss(z_geo, enc_w)

            # Tier 3 losses (geometry/codebook health)
            orth_loss = compute_orthogonality_loss(model_atlas)
            code_ent_loss = compute_code_entropy_loss(indices_stack, config.codes_per_chart)
            per_chart_code_ent_loss = compute_per_chart_code_entropy_loss(
                indices_stack, K_chart, config.num_charts, config.codes_per_chart
            )

            # Tier 4 losses (invariance - expensive, conditional computation)
            # KL prior (cheap, compute if enabled)
            if config.kl_prior_weight > 0:
                kl_loss = compute_kl_prior_loss(z_n, z_tex)
            else:
                kl_loss = torch.tensor(0.0, device=device)

            # Orbit and VICReg invariance (expensive - share augmented forward pass)
            orbit_loss = torch.tensor(0.0, device=device)
            vicreg_loss = torch.tensor(0.0, device=device)

            if config.orbit_weight > 0 or config.vicreg_inv_weight > 0:
                # Single augmented forward pass (shared between both losses)
                x_aug = augment_nightmare(
                    batch_X, config.augment_noise_std, config.augment_rotation_max
                )
                _, _, _, _, enc_w_aug, z_geo_aug, _, _, _ = model_atlas.encoder(x_aug)

                if config.orbit_weight > 0:
                    orbit_loss = compute_orbit_loss(enc_w, enc_w_aug)
                if config.vicreg_inv_weight > 0:
                    vicreg_loss = compute_vicreg_invariance_loss(z_geo, z_geo_aug)

            # Tier 5: Jump Operator (scheduled warmup - let atlas form before learning transitions)
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
                # Tier 1
                + config.variance_weight * var_loss
                + config.diversity_weight * div_loss
                + config.separation_weight * sep_loss
                # Tier 2
                + config.window_weight * window_loss
                + config.disentangle_weight * dis_loss
                # Tier 3
                + config.orthogonality_weight * orth_loss
                + config.code_entropy_weight * code_ent_loss
                + config.per_chart_code_entropy_weight * per_chart_code_ent_loss
                # Tier 4 (conditional - 0 if disabled)
                + config.kl_prior_weight * kl_loss
                + config.orbit_weight * orbit_loss
                + config.vicreg_inv_weight * vicreg_loss
                # Tier 5: Jump Operator (scheduled)
                + current_jump_weight * jump_loss
                # Supervised topology
                + config.sup_weight * sup_total
            )

            opt_atlas.zero_grad()
            loss_a.backward()
            # Gradient clipping (prevents instability from competing losses)
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

        # Step LR scheduler at end of each epoch
        if scheduler is not None:
            scheduler.step()

        # Logging and visualization (matching embed_fragile.py style)
        should_log = epoch % config.log_every == 0 or epoch == config.epochs
        should_save = config.save_every > 0 and (
            epoch % config.save_every == 0 or epoch == config.epochs
        )
        if should_log or should_save:
            # Compute metrics on full dataset
            with torch.no_grad():
                K_chart_full, _, _, _, enc_w_full, _, _, _, _ = model_atlas.encoder(X_test)
                usage = enc_w_full.mean(dim=0).cpu().numpy()
                chart_assignments = K_chart_full.cpu().numpy()
                ami = compute_ami(labels_test, chart_assignments)
                perplexity = model_atlas.compute_perplexity(K_chart_full)

            # Get epoch-averaged losses
            avg_loss = atlas_losses[-1]
            avg_recon = loss_components["recon"][-1]
            avg_vq = loss_components["vq"][-1]
            avg_entropy = loss_components["entropy"][-1]
            avg_consistency = loss_components["consistency"][-1]
            avg_var = loss_components["variance"][-1]
            avg_div = loss_components["diversity"][-1]
            avg_sep = loss_components["separation"][-1]
            avg_window = loss_components["window"][-1]
            avg_disent = loss_components["disentangle"][-1]
            avg_orth = loss_components["orthogonality"][-1]
            avg_code_ent = loss_components["code_entropy"][-1]
            avg_pc_code_ent = loss_components["per_chart_code_entropy"][-1]
            avg_kl = loss_components["kl_prior"][-1]
            avg_orbit = loss_components["orbit"][-1]
            avg_vicreg = loss_components["vicreg_inv"][-1]
            avg_jump = loss_components["jump"][-1]
            avg_sup_total = loss_components["sup_total"][-1]
            avg_sup_route = loss_components["sup_route"][-1]
            avg_sup_purity = loss_components["sup_purity"][-1]
            avg_sup_balance = loss_components["sup_balance"][-1]
            avg_sup_metric = loss_components["sup_metric"][-1]
            avg_sup_acc = loss_components["sup_acc"][-1]
            avg_ixk = info_metrics["I_XK"][-1]
            avg_hk = info_metrics["H_K"][-1]

            # Get current jump weight for logging
            log_jump_weight = get_jump_weight_schedule(
                epoch, config.jump_warmup, config.jump_ramp_end, config.jump_weight
            )

            # Print in embed_fragile.py style
            current_lr = scheduler.get_last_lr()[0] if scheduler else config.lr
            print(f"Epoch {epoch:5d} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
            print(f"  Usage: {np.array2string(usage, precision=2, separator=', ')}")
            print(
                f"  Core: recon={avg_recon:.3f} "
                f"vq={avg_vq:.3f} "
                f"entropy={avg_entropy:.3f} "
                f"consistency={avg_consistency:.3f}"
            )
            print(
                f"  Tier1: var={avg_var:.3f} "
                f"div={avg_div:.3f} "
                f"sep={avg_sep:.3f}"
            )
            print(
                f"  Tier2: window={avg_window:.3f} "
                f"disent={avg_disent:.3f}"
            )
            print(
                f"  Tier3: orth={avg_orth:.3f} "
                f"code_ent={avg_code_ent:.3f} "
                f"pc_code_ent={avg_pc_code_ent:.3f}"
            )
            print(
                f"  Tier4: kl={avg_kl:.3f} "
                f"orbit={avg_orbit:.3f} "
                f"vicreg={avg_vicreg:.3f}"
            )
            print(
                f"  Tier5: jump={avg_jump:.3f} "
                f"(λ={log_jump_weight:.3f})"
            )
            if supervised_loss is not None:
                with torch.no_grad():
                    _, _, _, _, enc_w_test, z_geo_test, _, _, _ = model_atlas.encoder(
                        X_test
                    )
                    sup_test = supervised_loss(enc_w_test, labels_test_t, z_geo_test)
                    p_y_x_test = torch.matmul(enc_w_test, supervised_loss.p_y_given_k)
                    test_sup_acc = (
                        p_y_x_test.argmax(dim=1) == labels_test_t
                    ).float().mean().item()
                    test_sup_route = sup_test["loss_route"].item()

                print(
                    f"  Sup: train_acc={avg_sup_acc:.3f} "
                    f"train_route={avg_sup_route:.3f} "
                    f"test_acc={test_sup_acc:.3f} "
                    f"test_route={test_sup_route:.3f}"
                )
                print(
                    f"  Sup: total={avg_sup_total:.3f} "
                    f"route={avg_sup_route:.3f} "
                    f"purity={avg_sup_purity:.3f} "
                    f"balance={avg_sup_balance:.3f} "
                    f"metric={avg_sup_metric:.3f} "
                    f"acc={avg_sup_acc:.3f}"
                )
            print(
                f"  Info: I(X;K)={avg_ixk:.3f} "
                f"H(K)={avg_hk:.3f}"
            )
            print(f"  Metrics: AMI={ami:.4f} perplexity={perplexity:.2f}/{config.num_charts}")
            print("-" * 60)

            # Save visualization
            if should_save:
                save_path = f"{config.output_dir}/topo_epoch_{epoch:05d}.png"
                visualize_latent(
                    model_atlas,
                    X_test,
                    colors_test,
                    labels_test,
                    save_path,
                    epoch,
                    jump_op=jump_op,
                )

    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)

    with torch.no_grad():
        # VanillaAE metrics (reconstruction baseline)
        mse_ae = 0.0
        ami_ae = 0.0
        recon_ae_final = None
        if model_ae is not None:
            recon_ae_final, z_ae = model_ae(X_test)
            mse_ae = F.mse_loss(recon_ae_final, X_test).item()
            # Use K-Means on latent space for clustering (K=num_charts)
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
            # Cluster VQ codes to get comparable AMI
            vq_clusters = indices_s.cpu().numpy() % config.num_charts  # Simple modulo clustering
            ami_std = compute_ami(labels_test, vq_clusters)

        # Atlas metrics (use dreaming mode to test autonomous routing)
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

    # Interpretation (only if baselines enabled)
    if model_ae is not None and model_std is not None:
        print("\nInterpretation:")
        if mse_ae < mse_atlas < mse_std:
            print("  AE has best reconstruction (expected - no bottleneck)")
            print("  TopoEncoder beats VQ on reconstruction (atlas routing helps)")
        if ami_atlas > ami_ae and ami_atlas > ami_std:
            print("  TopoEncoder has best topology discovery (charts match manifolds)")
        if ami_ae < ami_atlas:
            print("  AE fails at topology despite good reconstruction (entangled latent)")

    print(f"\nRouting Consistency (KL): {final_consistency:.4f}")
    if supervised_loss is not None:
        print(f"Supervised Accuracy: {sup_acc:.4f}")

    # Save final visualization
    if config.save_every > 0:
        final_path = f"{config.output_dir}/topo_final.png"
        visualize_latent(
            model_atlas,
            X_test,
            colors_test,
            labels_test,
            final_path,
            epoch=None,
            jump_op=jump_op,
        )
        print(f"\nFinal visualization saved to: {final_path}")

    # Results dict uses already-computed reconstructions from final evaluation
    return {
        "std_losses": std_losses,
        "atlas_losses": atlas_losses,
        "ae_losses": ae_losses,
        "loss_components": loss_components,
        # AMI scores
        "ami_ae": ami_ae,
        "ami_std": ami_std,
        "ami_atlas": ami_atlas,
        # MSE scores
        "mse_ae": mse_ae,
        "mse_std": mse_std,
        "mse_atlas": mse_atlas,
        # Perplexity
        "std_perplexity": std_perplexity,
        "atlas_perplexity": atlas_perplexity,
        "sup_acc": sup_acc,
        # Data
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
        "dataset_ids": dataset_ids,
        # For reconstruction comparison
        "recon_ae": recon_ae_final,
        "recon_std": recon_std_final,
        "recon_atlas": recon_atlas_final,
        # Models (for further analysis)
        "model_ae": model_ae,
        "model_std": model_std,
        "model_atlas": model_atlas,
        "config": config,
    }


# ==========================================
# 10. MAIN
# ==========================================
def main():
    """Main entry point for the benchmark."""
    parser = argparse.ArgumentParser(
        description="TopoEncoder Benchmark: Attentive Atlas vs Standard VQ-VAE"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training (0 = full batch)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3000,
        help="Total samples (divided by 3 per manifold)",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Fraction of samples for test split (0.0-0.99)",
    )
    parser.add_argument(
        "--num_charts",
        type=int,
        default=3,
        help="Number of atlas charts",
    )
    parser.add_argument(
        "--codes_per_chart",
        type=int,
        default=21,
        help="VQ codes per chart",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension for TopoEncoder",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Log training metrics every N epochs",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Save visualization every N epochs (0 to disable)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/topoencoder",
        help="Output directory for visualizations",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    # Tier 3 losses (codebook health)
    parser.add_argument(
        "--per_chart_code_entropy_weight",
        type=float,
        default=0.1,
        help="Per-chart code entropy weight (forces each chart to use all codes, default: 0.1)",
    )
    parser.add_argument(
        "--code_entropy_weight",
        type=float,
        default=0.0,
        help="Global code entropy weight (all codes used uniformly, default: 0.0)",
    )

    # Tier 4 losses (invariance)
    parser.add_argument(
        "--kl_prior_weight",
        type=float,
        default=0.01,
        help="KL prior weight on z_n, z_tex (default: 0.01)",
    )
    parser.add_argument(
        "--orbit_weight",
        type=float,
        default=0.0,
        help="Orbit invariance weight (default: 0.0, enables 2x slowdown)",
    )
    parser.add_argument(
        "--vicreg_inv_weight",
        type=float,
        default=0.0,
        help="VICReg invariance weight (default: 0.0, shares augmentation pass)",
    )
    parser.add_argument(
        "--augment_noise_std",
        type=float,
        default=0.1,
        help="Augmentation noise std (default: 0.1)",
    )
    parser.add_argument(
        "--augment_rotation_max",
        type=float,
        default=0.3,
        help="Max rotation in radians for augmentation (default: 0.3)",
    )

    # Training dynamics
    parser.add_argument(
        "--use_scheduler",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use cosine annealing LR scheduler (default: True)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="Minimum LR for scheduler (default: 1e-5)",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (0 to disable, default: 1.0)",
    )

    # Tier 5: Jump Operator (chart gluing)
    parser.add_argument(
        "--jump_weight",
        type=float,
        default=0.1,
        help="Final jump consistency weight after warmup (default: 0.1)",
    )
    parser.add_argument(
        "--jump_warmup",
        type=int,
        default=50,
        help="Epochs before jump loss starts (default: 50)",
    )
    parser.add_argument(
        "--jump_ramp_end",
        type=int,
        default=100,
        help="Epoch when jump weight reaches final value (default: 100)",
    )
    parser.add_argument(
        "--jump_global_rank",
        type=int,
        default=0,
        help="Rank of global tangent space (0 = use latent_dim, default: 0)",
    )

    # Supervised topology loss
    parser.add_argument(
        "--disable_supervised",
        action="store_true",
        help="Disable supervised topology losses",
    )
    parser.add_argument(
        "--sup_weight",
        type=float,
        default=1.0,
        help="Global weight for supervised topology loss",
    )
    parser.add_argument(
        "--sup_purity_weight",
        type=float,
        default=0.1,
        help="Supervised purity loss weight",
    )
    parser.add_argument(
        "--sup_balance_weight",
        type=float,
        default=0.01,
        help="Supervised balance loss weight",
    )
    parser.add_argument(
        "--sup_metric_weight",
        type=float,
        default=0.01,
        help="Supervised metric loss weight",
    )
    parser.add_argument(
        "--sup_metric_margin",
        type=float,
        default=1.0,
        help="Supervised metric loss margin",
    )
    parser.add_argument(
        "--sup_temperature",
        type=float,
        default=1.0,
        help="Temperature for chart-to-class mapping",
    )

    # Benchmark control
    parser.add_argument(
        "--disable_ae",
        action="store_true",
        help="Disable VanillaAE baseline (faster training)",
    )
    parser.add_argument(
        "--disable_vq",
        action="store_true",
        help="Disable StandardVQ baseline (faster training)",
    )

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create config
    config = TopoEncoderConfig(
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
        # Tier 3 losses
        per_chart_code_entropy_weight=args.per_chart_code_entropy_weight,
        code_entropy_weight=args.code_entropy_weight,
        # Tier 4 losses
        kl_prior_weight=args.kl_prior_weight,
        orbit_weight=args.orbit_weight,
        vicreg_inv_weight=args.vicreg_inv_weight,
        augment_noise_std=args.augment_noise_std,
        augment_rotation_max=args.augment_rotation_max,
        # Training dynamics
        use_scheduler=args.use_scheduler,
        min_lr=args.min_lr,
        grad_clip=args.grad_clip,
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
        # Benchmark control
        disable_ae=args.disable_ae,
        disable_vq=args.disable_vq,
    )

    print("=" * 50)
    print("TopoEncoder Benchmark")
    print("Attentive Atlas vs Standard VQ-VAE")
    print("=" * 50)
    print(f"\nConfiguration:")
    print(f"  Epochs: {config.epochs}, Batch size: {config.batch_size}")
    print(f"  Total samples: {config.n_samples}")
    print(f"  Test split: {config.test_split}")
    print(f"  Num charts: {config.num_charts}")
    print(f"  Codes per chart: {config.codes_per_chart}")
    print(f"  Total atlas codes: {config.num_charts * config.codes_per_chart}")
    print(f"  Standard VQ codes: {config.num_codes_standard}")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Save every: {config.save_every} epochs")

    # Run benchmark
    results = train_benchmark(config)

    # Save final comparison visualization
    os.makedirs(config.output_dir, exist_ok=True)
    final_path = f"{config.output_dir}/benchmark_result.png"
    visualize_results(results, save_path=final_path)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    ami_atlas = results["ami_atlas"]
    ami_ae = results["ami_ae"]
    if ami_atlas > 0.8:
        print(f"TopoEncoder AMI = {ami_atlas:.4f} - Excellent! Atlas discovered the topology.")
    elif ami_atlas > 0.5:
        print(f"TopoEncoder AMI = {ami_atlas:.4f} - Good. Atlas partially learned the topology.")
    else:
        print(f"TopoEncoder AMI = {ami_atlas:.4f} - Poor. Atlas did not learn the topology well.")

    if ami_atlas > ami_ae:
        print(f"TopoEncoder beats VanillaAE ({ami_atlas:.3f} > {ami_ae:.3f}) - better topology!")
    else:
        print(f"VanillaAE beats TopoEncoder ({ami_ae:.3f} > {ami_atlas:.3f}) - K-Means works well here")
    print(f"\nOutput saved to: {config.output_dir}/")


if __name__ == "__main__":
    main()
