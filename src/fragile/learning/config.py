"""TopoEncoder configuration dataclass."""

from dataclasses import dataclass, field

import torch


@dataclass
class TopoEncoderConfig:
    """Configuration for the TopoEncoder benchmark."""

    # Data
    dataset: str = "mnist"
    n_samples: int = 3000  # Subsample size
    input_dim: int = 784  # MNIST default (overridden for CIFAR-10)

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
    vision_preproc: bool = False
    vision_in_channels: int = 0
    vision_height: int = 0
    vision_width: int = 0
    vision_num_rotations: int = 8
    vision_kernel_size: int = 5
    vision_use_reflections: bool = False
    vision_norm_nonlinearity: str = "n_sigmoid"
    vision_norm_bias: bool = True
    # Vision backbone selection
    vision_backbone_type: str = "covariant_retina"  # "covariant_retina" or "covariant_cifar"
    vision_cifar_base_channels: int = 32  # base_channels for CovariantCIFARBackbone
    vision_cifar_bundle_size: int = 4  # bundle_size for NormGatedConv2d

    # Training
    epochs: int = 1000
    batch_size: int = 256  # Batch size for training (0 = full batch)
    eval_batch_size: int = 0  # Batch size for eval/logging (0 = use batch_size)
    lr: float = 1e-3
    vq_commitment_cost: float = 0.25
    entropy_weight: float = 0.1  # Encourage high routing entropy (anti-collapse)
    consistency_weight: float = 0.1  # Align encoder/decoder routing

    # Tier 1 losses (low overhead ~5%)
    variance_weight: float = 0.1  # Prevent latent collapse
    diversity_weight: float = 0.1  # Prevent chart collapse (was 1.0)
    separation_weight: float = 0.1  # Force chart centers apart (was 0.5)
    separation_margin: float = 2.0  # Minimum distance between chart centers
    codebook_center_weight: float = 0.05  # Zero-mean codebook deltas per chart
    chart_center_sep_weight: float = 0.05  # Separate chart center tokens
    chart_center_sep_margin: float = 2.0  # Minimum distance between chart centers (token space)
    residual_scale_weight: float = 0.01  # Keep z_n small vs macro/meso scales

    # Tier 2 losses (medium overhead ~5%)
    window_weight: float = 0.5  # Information-stability (Theorem 15.1.3)
    window_eps_ground: float = 0.1  # Minimum I(X;K) threshold
    disentangle_weight: float = 0.1  # Gauge coherence (K ⊥ z_n)

    # Tier 3 losses (geometry/codebook health)
    orthogonality_weight: float = 0.0  # Singular-value spread penalty (SVD; disabled by default)
    code_entropy_weight: float = 0.0  # Global code entropy (disabled by default)
    per_chart_code_entropy_weight: float = 0.1  # Per-chart code diversity (enabled)

    # Tier 4 losses (invariance - expensive when enabled, disabled by default)
    kl_prior_weight: float = 0.01  # Radial energy prior on z_n, z_tex
    orbit_weight: float = 0.0  # Chart invariance under augmentation (2x slowdown)
    vicreg_inv_weight: float = 0.0  # Gram invariance (O(B^2), disabled by default)
    augment_noise_std: float = 0.1  # Augmentation noise level
    augment_rotation_max: float = 0.3  # Max rotation in radians

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
    sup_metric_weight: float = 0.01
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
    baseline_vision_preproc: bool = False
    baseline_attn: bool = False
    baseline_attn_tokens: int = 4
    baseline_attn_dim: int = 32
    baseline_attn_heads: int = 4
    baseline_attn_dropout: float = 0.0

    # CIFAR backbone benchmark (gauge-covariant vs standard CNN)
    enable_cifar_backbone: bool = False  # Enable CIFAR backbone benchmark
    cifar_backbone_type: str = "both"  # "covariant", "standard", or "both"
    cifar_base_channels: int = 32  # Base channel width (16/32/64)
    cifar_bundle_size: int = 4  # Bundle size for NormGatedConv2d

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
