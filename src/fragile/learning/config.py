"""TopoEncoder configuration dataclass."""

from dataclasses import dataclass, field

import torch


@dataclass
class TopoEncoderConfig:
    """Configuration for the TopoEncoder benchmark."""

    # Data
    dataset: str = "mnist"
    n_samples: int = 3000  # Subsample size
    input_dim: int = 784  # MNIST / Fashion-MNIST default

    # Model architecture
    hidden_dim: int = 32
    latent_dim: int = 2  # For 2D visualization
    num_charts: int = 10  # Match number of classes
    codes_per_chart: int = 32  # Better coverage (was 21)
    num_codes_standard: int = 64
    covariant_attn: bool = True
    covariant_attn_tensorization: str = "full"
    covariant_attn_rank: int = 8
    covariant_attn_tau_min: float = 1e-2
    covariant_attn_denom_min: float = 1e-3
    covariant_attn_use_transport: bool = True
    covariant_attn_transport_eps: float = 1e-3
    soft_equiv_metric: bool = False
    soft_equiv_bundle_size: int | None = None
    soft_equiv_hidden_dim: int = 64
    soft_equiv_use_spectral_norm: bool = True
    soft_equiv_zero_self_mixing: bool = False
    soft_equiv_soft_assign: bool = True
    soft_equiv_temperature: float = 1.0
    soft_equiv_l1_weight: float = 0.0
    soft_equiv_log_ratio_weight: float = 0.0

    # Convolutional backbone (for image data)
    conv_backbone: bool = True  # Use conv layers instead of FC for feature extraction/reconstruction
    img_channels: int = 1  # Image channels (1 for grayscale MNIST/Fashion-MNIST)
    img_size: int = 28  # Image spatial dimension (28 for MNIST/Fashion-MNIST)

    # Training
    epochs: int = 1000
    batch_size: int = 256  # Batch size for training (0 = full batch)
    eval_batch_size: int = 0  # Batch size for eval/logging (0 = use batch_size)
    lr: float = 1e-3
    vq_commitment_cost: float = 0.25
    entropy_weight: float = 0.1  # Encourage high routing entropy (anti-collapse)
    consistency_weight: float = 0.1  # Align encoder/decoder routing

    # Tier 1 losses (low overhead ~5%)
    variance_weight: float = 0.0  # DROP: Prevent latent collapse (was 0.1)
    diversity_weight: float = 0.1  # Prevent chart collapse (was 1.0)
    separation_weight: float = 0.0  # DROP: Force chart centers apart (was 0.1)
    separation_margin: float = 2.0  # Minimum distance between chart centers
    codebook_center_weight: float = 0.05  # Zero-mean codebook deltas per chart
    chart_center_sep_weight: float = 0.0  # DROP: Separate chart center tokens (was 0.05)
    chart_center_sep_margin: float = 2.0  # Minimum distance between chart centers (token space)
    residual_scale_weight: float = 0.01  # Keep z_n small vs macro/meso scales

    # Tier 2 losses (medium overhead ~5%)
    window_weight: float = 0.5  # Information-stability (Theorem 15.1.3)
    window_eps_ground: float = 0.1  # Minimum I(X;K) threshold
    disentangle_weight: float = 0.0  # DROP: Gauge coherence (K ⊥ z_n) (was 0.1)

    # Tier 3 losses (geometry/codebook health)
    orthogonality_weight: float = 0.0  # Singular-value spread penalty (SVD; disabled by default)
    code_entropy_weight: float = 0.0  # Global code entropy (disabled by default)
    per_chart_code_entropy_weight: float = 0.1  # Per-chart code diversity (enabled)

    # Tier 4 losses (invariance)
    kl_prior_weight: float = 0.0  # DROP: Radial energy prior on z_n, z_tex

    # New geometric losses (Poincaré-aware)
    hyperbolic_uniformity_weight: float = 0.1
    hyperbolic_contrastive_weight: float = 0.0  # Disabled: causes chart collapse by clustering
    hyperbolic_contrastive_margin: float = 2.0
    radial_calibration_weight: float = 0.1
    codebook_spread_weight: float = 0.05
    codebook_spread_margin: float = 1.0

    # New symbol losses
    symbol_purity_weight: float = 0.05
    symbol_calibration_weight: float = 0.05

    # Anti-collapse penalties
    chart_collapse_weight: float = 1.0  # max(p_k) - 1/K penalty
    code_collapse_weight: float = 0.5  # soft code usage entropy penalty
    code_collapse_temperature: float = 1.0  # temperature for soft code assignments

    # Tier 5: Jump Operator (chart gluing - learns transition functions between charts)
    jump_weight: float = 0.1  # Final jump consistency weight after warmup
    jump_warmup: int = 50  # Epochs before jump loss starts (let atlas form first)
    jump_ramp_end: int = 100  # Epoch when jump weight reaches final value
    jump_global_rank: int = 0  # Rank of global tangent space (0 = use latent_dim)

    # Supervised topology loss (Section 7.12)
    enable_supervised: bool = True
    num_classes: int = 10
    sup_weight: float = 1.0
    sup_purity_weight: float = 0.1
    sup_balance_weight: float = 0.01
    sup_metric_weight: float = 0.0
    sup_metric_margin: float = 1.0
    sup_temperature: float = 1.0

    # Classifier readout (detached, invariant)
    enable_classifier_head: bool = True
    classifier_lr: float = 0.0  # 0 = use main lr
    classifier_bundle_size: int = 0  # 0 = global norm

    # Gradient clipping
    grad_clip: float = 1.0  # Max gradient norm (0 to disable)

    # LR Scheduler
    use_scheduler: bool = True
    lr_min: float = 1e-6  # eta_min for CosineAnnealingLR

    # Benchmark control
    disable_ae: bool = False  # Skip VanillaAE baseline
    disable_vq: bool = False  # Skip StandardVQ baseline
    baseline_attn: bool = False
    baseline_attn_tokens: int = 4
    baseline_attn_dim: int = 32
    baseline_attn_heads: int = 4
    baseline_attn_dropout: float = 0.0

    # Train/test split
    test_split: float = 0.2

    # Logging and output
    log_every: int = 100
    save_every: int = 100  # Save checkpoint every N epochs (0 to disable)
    output_dir: str = "outputs/topoencoder"
    resume_checkpoint: str = ""
    mlflow: bool = False
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = ""
    mlflow_run_name: str = ""
    mlflow_run_id: str = ""

    # Device (CUDA if available, else CPU)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
