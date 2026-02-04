"""
TopoEncoder Benchmark: Attentive Atlas vs Standard VQ-VAE

This script benchmarks the Attentive Atlas architecture (from fragile-index.md Section 7.8)
against a standard VQ-VAE on MNIST or CIFAR-10 (MNIST default).

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
    python src/experiments/topoencoder_2d.py --dataset mnist --epochs 1000

Reference: fragile-index.md Sections 7.8, 7.10
"""

import argparse
from dataclasses import asdict, dataclass, field
import math
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm


try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    _MLFLOW_AVAILABLE = False

from fragile.core.layers import (
    FactorizedJumpOperator,
    InvariantChartClassifier,
    StandardVQ,
    TopoEncoderPrimitives,
    VanillaAE,
)
from fragile.core.losses import (
    compute_chart_center_separation_loss,
    compute_code_entropy_loss,
    compute_codebook_centering_loss,
    compute_disentangle_loss,
    compute_diversity_loss,
    compute_jump_consistency_loss,
    compute_kl_prior_loss,
    compute_orbit_loss,
    compute_orthogonality_loss,
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
from fragile.datasets import CIFAR10_CLASSES, get_cifar10_data, get_mnist_data


# ==========================================
# 1. CONFIGURATION
# ==========================================
@dataclass
class TopoEncoderConfig:
    """Configuration for the TopoEncoder benchmark."""

    # Data
    dataset: str = "mnist"
    n_samples: int = 40000  # Subsample size
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
    hard_routing: bool = False
    hard_routing_tau: float = 1.0
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
    vision_backbone_type: str = "covariant_retina"
    vision_cifar_base_channels: int = 32
    vision_cifar_bundle_size: int = 4
    vision_soft_equiv: bool = False
    vision_soft_equiv_bundle_size: int = 0
    vision_soft_equiv_hidden_dim: int = 64
    vision_soft_equiv_use_spectral_norm: bool = True
    vision_soft_equiv_zero_self_mixing: bool = False
    vision_soft_equiv_alpha: float = 0.1
    vision_soft_equiv_per_block: bool = False
    vision_standard_head: bool = False
    vision_standard_head_blocks: int = 2
    vision_spectral_head: bool = False

    # Training
    epochs: int = 50
    batch_size: int = 128  # Batch size for training (0 = full batch)
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
    jump_warmup: int = 2  # Epochs before jump loss starts (let atlas form first)
    jump_ramp_end: int = 50  # Epoch when jump weight reaches final value
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

    # Learning rate scheduling
    use_scheduler: bool = False  # Use cosine annealing LR scheduler
    min_lr: float = 1e-5  # Minimum LR at end of schedule

    # Gradient clipping
    grad_clip: float = 1.0  # Max gradient norm (0 to disable)

    # Adaptive learning rate (maximize LR under stability constraints)
    adaptive_lr: bool = True
    lr_min: float = 1e-6
    lr_max: float = 0.0  # 0 = use lr as max
    lr_increase_factor: float = 1.02
    lr_decrease_factor: float = 0.5
    lr_max_update_ratio: float = 1e-3
    lr_ema_decay: float = 0.98
    lr_loss_increase_tol: float = 0.05
    lr_grounding_warmup_epochs: int = 0  # Skip grounding checks for early LR ramp
    lr_unstable_patience: int = 3
    lr_stable_patience: int = 5
    lr_grounding_ema_decay: float = 0.98
    lr_recovery_factor: float = 1.1
    lr_recovery_threshold: float = 0.1
    lr_plateau_patience: int = 10
    lr_plateau_tol: float = 1e-3

    # Adaptive loss weights (primal-dual + PI controllers)
    adaptive_weights: bool = True
    adaptive_warmup_epochs: int = 10
    adaptive_dual_eta: float = 0.05
    adaptive_pi_kp: float = 0.05
    adaptive_pi_ki: float = 0.01
    adaptive_pi_kd: float = 0.0
    adaptive_lambda_min: float = 0.0
    adaptive_lambda_max: float = 10.0
    adaptive_violation_clip: float = 2.0
    adaptive_target_ratio: float = 1.0
    adaptive_target_ema_decay: float = 0.98
    adaptive_target_min: float = 1e-3
    entropy_target_ratio: float = 0.5
    hk_target_ratio: float = 0.9
    code_entropy_target_ratio: float = 0.9
    consistency_target: float = 0.05

    # Learned precisions for likelihood-like losses
    use_learned_precisions: bool = False

    # Benchmark control
    disable_ae: bool = False  # Skip VanillaAE baseline
    disable_vq: bool = False  # Skip StandardVQ baseline
    baseline_vision_preproc: bool = False
    baseline_attn: bool = False
    baseline_attn_tokens: int = 4
    baseline_attn_dim: int = 32
    baseline_attn_heads: int = 4
    baseline_attn_dropout: float = 0.0

    # Train/test split
    test_split: float = 0.2

    # Logging and output
    log_every: int = 1
    save_every: int = 5  # Save checkpoint every N epochs (0 to disable)
    output_dir: str = "outputs/topoencoder"
    resume_checkpoint: str = ""
    mlflow: bool = False
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = ""
    mlflow_run_name: str = ""

    # Device (CUDA if available, else CPU)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


class BaselineClassifier(nn.Module):
    """Small MLP probe for baseline latent spaces."""

    def __init__(self, latent_dim: int, num_classes: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def compute_matching_hidden_dim(
    target_params: int,
    input_dim: int = 3,
    latent_dim: int = 2,
    num_codes: int = 64,
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
    """Compute hidden_dim for StandardVQ to match target parameter count.

    StandardVQ params = 2h² + (4 + 2 + 3 + 3)h + (2 + num_codes*latent_dim + 3)
                      = 2h² + 12h + (5 + num_codes*latent_dim)

    Using quadratic formula: h = (-12 + sqrt(144 + 8*(target - offset))) / 4
    """
    if not use_attention and not vision_preproc:
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

    min_hidden = 16
    max_hidden = 2048
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
# 5. ADAPTIVE CONTROL (LR + LAMBDAS)
# ==========================================
@dataclass
class AdaptiveWeightState:
    value: float
    integral: float = 0.0
    prev_error: float = 0.0


@dataclass
class AdaptiveLRState:
    lr: float
    loss_ema: float = 0.0
    best_loss_ema: float = 0.0
    ixk_ema: float = 0.0
    unstable_steps: int = 0
    stable_steps: int = 0
    plateau_steps: int = 0


def _set_optimizer_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _compute_param_norm(params: list[torch.Tensor]) -> float:
    total = 0.0
    for p in params:
        total += p.detach().pow(2).sum().item()
    return math.sqrt(total)


def _compute_grad_norm(params: list[torch.Tensor]) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def _update_primal_dual(
    state: AdaptiveWeightState,
    violation: float,
    eta: float,
    min_val: float,
    max_val: float,
) -> float:
    state.value = float(min(max(state.value + eta * violation, min_val), max_val))
    return state.value


def _update_pi_controller(
    state: AdaptiveWeightState,
    error: float,
    kp: float,
    ki: float,
    kd: float,
    min_val: float,
    max_val: float,
) -> float:
    state.integral += error
    delta = kp * error + ki * state.integral + kd * (error - state.prev_error)
    state.prev_error = error
    state.value = float(min(max(state.value + delta, min_val), max_val))
    return state.value


def _apply_precision(loss: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return torch.exp(-log_var) * loss + log_var


def _clip_violation(value: float, clip: float) -> float:
    if clip <= 0:
        return float(value)
    return float(max(-clip, min(clip, value)))


def _energy_violation(z: torch.Tensor, target_std: float = 1.0) -> float:
    z_centered = z - z.mean(dim=0, keepdim=True)
    dim = z.shape[1]
    mean_energy = (z_centered**2).sum(dim=1).mean()
    target = (target_std**2) * dim
    return float(((mean_energy - target) / (target + 1e-6)).detach())


def _update_target_state(
    targets: dict[str, float],
    name: str,
    value: float,
    decay: float,
) -> None:
    value = float(value)
    if name not in targets or targets[name] <= 0.0:
        targets[name] = value
        return
    targets[name] = decay * targets[name] + (1.0 - decay) * value


def _compute_target_error(
    targets: dict[str, float],
    name: str,
    value: float,
    ratio: float,
    min_target: float,
) -> float:
    base = float(targets.get(name, 0.0))
    if base <= 0.0:
        base = float(value)
        targets[name] = base
    target = max(min_target, base * ratio)
    return (float(value) - target) / (target + 1e-6)


def _restore_weight_state(
    config: TopoEncoderConfig,
    resume_metrics: dict,
    names: list[str],
) -> dict[str, AdaptiveWeightState]:
    raw_state = resume_metrics.get("adaptive_weight_state", {})
    state: dict[str, AdaptiveWeightState] = {}
    for name in names:
        if name in raw_state:
            entry = raw_state[name]
            state[name] = AdaptiveWeightState(
                value=float(entry.get("value", getattr(config, name))),
                integral=float(entry.get("integral", 0.0)),
                prev_error=float(entry.get("prev_error", 0.0)),
            )
        else:
            state[name] = AdaptiveWeightState(value=float(getattr(config, name)))
    return state


def _serialize_weight_state(
    state: dict[str, AdaptiveWeightState],
) -> dict[str, dict[str, float]]:
    return {
        name: {
            "value": entry.value,
            "integral": entry.integral,
            "prev_error": entry.prev_error,
        }
        for name, entry in state.items()
    }


def _restore_lr_state(config: TopoEncoderConfig, resume_metrics: dict) -> AdaptiveLRState:
    raw_state = resume_metrics.get("adaptive_lr_state", {})
    lr = float(raw_state.get("lr", config.lr))
    loss_ema = float(raw_state.get("loss_ema", 0.0))
    best_loss_ema = float(raw_state.get("best_loss_ema", 0.0))
    ixk_ema = float(raw_state.get("ixk_ema", 0.0))
    unstable_steps = int(raw_state.get("unstable_steps", 0))
    stable_steps = int(raw_state.get("stable_steps", 0))
    plateau_steps = int(raw_state.get("plateau_steps", 0))
    return AdaptiveLRState(
        lr=lr,
        loss_ema=loss_ema,
        best_loss_ema=best_loss_ema,
        ixk_ema=ixk_ema,
        unstable_steps=unstable_steps,
        stable_steps=stable_steps,
        plateau_steps=plateau_steps,
    )


def _restore_target_state(
    resume_metrics: dict,
    names: list[str],
) -> dict[str, float]:
    raw_state = resume_metrics.get("adaptive_target_state", {})
    state: dict[str, float] = {}
    for name in names:
        if name in raw_state:
            state[name] = float(raw_state[name])
        else:
            state[name] = 0.0
    return state


def _serialize_target_state(state: dict[str, float]) -> dict[str, float]:
    return {name: float(value) for name, value in state.items()}


def _start_mlflow_run(
    config: TopoEncoderConfig,
    extra_params: dict[str, object] | None = None,
) -> bool:
    if not config.mlflow:
        return False
    if not _MLFLOW_AVAILABLE:
        print("MLflow logging requested but mlflow is not installed. Skipping MLflow.")
        return False
    if config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    if config.mlflow_experiment:
        mlflow.set_experiment(config.mlflow_experiment)
    run_name = config.mlflow_run_name or f"topoencoder_{config.dataset}"
    mlflow.start_run(run_name=run_name)
    params = asdict(config)
    if extra_params:
        params.update(extra_params)
    safe_params: dict[str, object] = {}
    for key, value in params.items():
        if isinstance(value, int | float | str | bool):
            safe_params[key] = value
        else:
            safe_params[key] = str(value)
    if safe_params:
        mlflow.log_params(safe_params)
    return True


def _log_mlflow_metrics(
    metrics: dict[str, float],
    step: int,
    enabled: bool,
) -> None:
    if not enabled:
        return
    safe_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            safe_metrics[key] = val
    if safe_metrics:
        mlflow.log_metrics(safe_metrics, step=step)


def _end_mlflow_run(enabled: bool) -> None:
    if enabled and _MLFLOW_AVAILABLE:
        mlflow.end_run()


# ==========================================
# 6. CHECKPOINTING
# ==========================================
def _state_dict_cpu(module: nn.Module | None) -> dict[str, torch.Tensor] | None:
    if module is None:
        return None
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def save_checkpoint(
    path: str,
    config: TopoEncoderConfig,
    model_atlas: nn.Module,
    jump_op: nn.Module,
    metrics: dict,
    data_snapshot: dict,
    epoch: int,
    model_std: nn.Module | None = None,
    model_ae: nn.Module | None = None,
    supervised_loss: nn.Module | None = None,
    classifier_head: nn.Module | None = None,
    classifier_std: nn.Module | None = None,
    classifier_ae: nn.Module | None = None,
    precision_module: nn.Module | None = None,
    optimizer_atlas: optim.Optimizer | None = None,
    optimizer_std: optim.Optimizer | None = None,
    optimizer_ae: optim.Optimizer | None = None,
    optimizer_classifier: optim.Optimizer | None = None,
    optimizer_classifier_std: optim.Optimizer | None = None,
    optimizer_classifier_ae: optim.Optimizer | None = None,
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
) -> None:
    """Save training checkpoint for later analysis/plotting."""
    checkpoint = {
        "epoch": epoch,
        "config": asdict(config),
        "state": {
            "atlas": _state_dict_cpu(model_atlas),
            "jump": _state_dict_cpu(jump_op),
            "supervised": _state_dict_cpu(supervised_loss),
            "classifier": _state_dict_cpu(classifier_head),
            "classifier_std": _state_dict_cpu(classifier_std),
            "classifier_ae": _state_dict_cpu(classifier_ae),
            "precisions": _state_dict_cpu(precision_module),
        },
        "optim": {
            "atlas": optimizer_atlas.state_dict() if optimizer_atlas is not None else None,
            "classifier": (
                optimizer_classifier.state_dict() if optimizer_classifier is not None else None
            ),
            "classifier_std": (
                optimizer_classifier_std.state_dict()
                if optimizer_classifier_std is not None
                else None
            ),
            "classifier_ae": (
                optimizer_classifier_ae.state_dict()
                if optimizer_classifier_ae is not None
                else None
            ),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        },
        "metrics": metrics,
        "data": data_snapshot,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str) -> dict:
    """Load checkpoint with unsafe deserialization allowed for trusted outputs."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_benchmarks(
    path: str,
    config: TopoEncoderConfig,
    model_std: nn.Module | None,
    model_ae: nn.Module | None,
    std_hidden_dim: int = 0,
    ae_hidden_dim: int = 0,
    optimizer_std: optim.Optimizer | None = None,
    optimizer_ae: optim.Optimizer | None = None,
    metrics: dict | None = None,
    epoch: int = 0,
) -> None:
    """Save baseline (VQ/AE) checkpoints for reuse between runs."""
    if model_std is None and model_ae is None:
        return
    payload = {
        "epoch": epoch,
        "config": {
            "dataset": config.dataset,
            "input_dim": config.input_dim,
            "latent_dim": config.latent_dim,
            "num_codes_standard": config.num_codes_standard,
            "baseline_vision_preproc": config.baseline_vision_preproc,
            "baseline_attn": config.baseline_attn,
            "baseline_attn_tokens": config.baseline_attn_tokens,
            "baseline_attn_dim": config.baseline_attn_dim,
            "baseline_attn_heads": config.baseline_attn_heads,
            "baseline_attn_dropout": config.baseline_attn_dropout,
            "vision_in_channels": config.vision_in_channels,
            "vision_height": config.vision_height,
            "vision_width": config.vision_width,
        },
        "state": {
            "std": _state_dict_cpu(model_std),
            "ae": _state_dict_cpu(model_ae),
        },
        "optim": {
            "std": optimizer_std.state_dict() if optimizer_std is not None else None,
            "ae": optimizer_ae.state_dict() if optimizer_ae is not None else None,
        },
        "metrics": metrics or {},
        "dims": {
            "std_hidden_dim": int(std_hidden_dim),
            "ae_hidden_dim": int(ae_hidden_dim),
        },
    }
    torch.save(payload, path)


def load_benchmarks(path: str) -> dict:
    """Load benchmark checkpoint with unsafe deserialization allowed for trusted outputs."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _benchmarks_compatible(bench_config: dict, config: TopoEncoderConfig) -> bool:
    if not bench_config:
        return False
    baseline_vision_preproc = bool(bench_config.get("baseline_vision_preproc"))
    if baseline_vision_preproc != bool(config.baseline_vision_preproc):
        return False
    if baseline_vision_preproc:
        if int(bench_config.get("vision_in_channels", -1)) != int(config.vision_in_channels):
            return False
        if int(bench_config.get("vision_height", -1)) != int(config.vision_height):
            return False
        if int(bench_config.get("vision_width", -1)) != int(config.vision_width):
            return False
    return (
        int(bench_config.get("input_dim", -1)) == int(config.input_dim)
        and int(bench_config.get("latent_dim", -1)) == int(config.latent_dim)
        and int(bench_config.get("num_codes_standard", -1)) == int(config.num_codes_standard)
        and bool(bench_config.get("baseline_vision_preproc"))
        == bool(config.baseline_vision_preproc)
        and bool(bench_config.get("baseline_attn")) == bool(config.baseline_attn)
        and int(bench_config.get("baseline_attn_tokens", -1)) == int(config.baseline_attn_tokens)
        and int(bench_config.get("baseline_attn_dim", -1)) == int(config.baseline_attn_dim)
        and int(bench_config.get("baseline_attn_heads", -1)) == int(config.baseline_attn_heads)
        and float(bench_config.get("baseline_attn_dropout", -1.0))
        == float(config.baseline_attn_dropout)
    )


def _move_optimizer_state(optimizer: optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


# ==========================================
# 7. METRICS
# ==========================================
def compute_ami(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute Adjusted Mutual Information score."""
    return float(adjusted_mutual_info_score(labels_true, labels_pred))


def _init_loss_components() -> dict[str, list[float]]:
    return {
        "recon": [],
        "vq": [],
        "entropy": [],
        "consistency": [],
        # Tier 1 losses
        "variance": [],
        "diversity": [],
        "separation": [],
        "codebook_center": [],
        "chart_center_sep": [],
        "residual_scale": [],
        "soft_equiv_l1": [],
        "soft_equiv_log_ratio": [],
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
        # Classifier readout (detached)
        "cls_loss": [],
        "cls_acc": [],
        # Baseline classifier readouts
        "std_cls_loss": [],
        "std_cls_acc": [],
        "ae_cls_loss": [],
        "ae_cls_acc": [],
    }


def _init_info_metrics() -> dict[str, list[float]]:
    return {
        "I_XK": [],
        "H_K": [],
        "H_K_given_X": [],
        "code_entropy": [],
        "per_chart_code_entropy": [],
        "grad_norm": [],
        "param_norm": [],
        "update_ratio": [],
        "lr": [],
    }


def _maybe_init_vision_shape(config: TopoEncoderConfig, dataset_name: str) -> None:
    if not (config.vision_preproc or config.baseline_vision_preproc):
        return
    if config.vision_in_channels <= 0 or config.vision_height <= 0 or config.vision_width <= 0:
        if config.dataset == "cifar10" or dataset_name == "CIFAR-10":
            config.vision_in_channels = 3
            config.vision_height = 32
            config.vision_width = 32
        elif config.dataset == "mnist" or dataset_name == "MNIST":
            config.vision_in_channels = 1
            config.vision_height = 28
            config.vision_width = 28
    expected = config.vision_in_channels * config.vision_height * config.vision_width
    if expected <= 0:
        msg = "vision_preproc requires valid vision_* dimensions."
        raise ValueError(msg)
    if config.input_dim != expected:
        raise ValueError(
            f"vision_preproc shape does not match input_dim ({config.input_dim} vs {expected})."
        )


# ==========================================
# 8. AUGMENTATION (for invariance losses)
# ==========================================
def augment_inputs(
    x: torch.Tensor,
    dataset: str,
    noise_std: float = 0.1,
    _rotation_max: float = 0.3,
) -> torch.Tensor:
    """Apply dataset-aware augmentation.

    Args:
        x: Input tensor [B, D]
        dataset: Dataset name
        noise_std: Standard deviation of additive noise
        _rotation_max: Maximum rotation angle in radians (unused for images)

    Returns:
        Augmented tensor [B, D]
    """
    if dataset not in {"mnist", "cifar10"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return x + torch.randn_like(x) * noise_std


# ==========================================
# 9. TRAINING
# ==========================================
def train_benchmark(config: TopoEncoderConfig) -> dict:
    """Train both models and return results.

    Returns dictionary with:
        - ami_ae / ami_std / ami_atlas: Final AMI scores
        - mse_ae / mse_std / mse_atlas: Final MSE scores
        - std_perplexity / atlas_perplexity: Final perplexities
        - sup_acc: Final supervised accuracy (if enabled)
        - checkpoint_path: Path to final checkpoint
    """
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"Saving checkpoints to: {config.output_dir}/")
    benchmarks_path = os.path.join(config.output_dir, "benchmarks.pt")
    benchmarks_candidates = [benchmarks_path]
    if config.resume_checkpoint:
        resume_dir = os.path.dirname(config.resume_checkpoint)
        if resume_dir and resume_dir not in benchmarks_candidates:
            benchmarks_candidates.append(os.path.join(resume_dir, "benchmarks.pt"))
    benchmarks_payload: dict | None = None
    benchmarks_state: dict = {}
    benchmarks_metrics: dict = {}
    benchmarks_dims: dict = {}
    benchmarks_compatible = False

    resume_state = None
    resume_metrics: dict = {}
    resume_optim: dict = {}
    start_epoch = 0

    if config.resume_checkpoint:
        checkpoint = load_checkpoint(config.resume_checkpoint)
        resume_state = checkpoint.get("state", {})
        resume_metrics = checkpoint.get("metrics", {})
        resume_optim = checkpoint.get("optim", {})
        data_snapshot = checkpoint.get("data", {})
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

        X_train = data_snapshot["X_train"]
        X_test = data_snapshot["X_test"]
        labels_train = data_snapshot["labels_train"]
        labels_test = data_snapshot["labels_test"]
        colors_train = data_snapshot["colors_train"]
        colors_test = data_snapshot["colors_test"]
        dataset_ids = data_snapshot.get("dataset_ids", {})
        dataset_name = data_snapshot.get("dataset_name", config.dataset)

        labels_full = (
            np.concatenate([labels_train, labels_test]) if len(labels_test) else labels_train
        )
        config.input_dim = X_train.shape[1]
        _maybe_init_vision_shape(config, dataset_name)

        print(f"Resuming from checkpoint: {config.resume_checkpoint}")
        print(f"Loaded {len(X_train) + len(X_test)} samples from {dataset_name}")
        print(f"Input dim: {config.input_dim}")
        if start_epoch > config.epochs:
            print(
                f"Checkpoint at epoch {start_epoch - 1} exceeds configured epochs "
                f"({config.epochs}). Nothing to do."
            )
            return {
                "ami_ae": resume_metrics.get("ami_ae", 0.0),
                "ami_std": resume_metrics.get("ami_std", 0.0),
                "ami_atlas": resume_metrics.get("ami_atlas", 0.0),
                "mse_ae": resume_metrics.get("mse_ae", 0.0),
                "mse_std": resume_metrics.get("mse_std", 0.0),
                "mse_atlas": resume_metrics.get("mse_atlas", 0.0),
                "sup_acc": resume_metrics.get("sup_acc", 0.0),
                "cls_acc": resume_metrics.get("cls_acc", 0.0),
                "atlas_perplexity": resume_metrics.get("atlas_perplexity", 0.0),
                "std_perplexity": resume_metrics.get("std_perplexity", 0.0),
                "checkpoint_path": config.resume_checkpoint,
            }
    else:
        # Generate data (MNIST or CIFAR-10)
        if config.dataset == "mnist":
            X, labels, colors = get_mnist_data(config.n_samples)
            dataset_ids = {str(i): i for i in range(10)}
            dataset_name = "MNIST"
        elif config.dataset == "cifar10":
            X, labels, colors = get_cifar10_data(config.n_samples)
            dataset_ids = {name: idx for idx, name in enumerate(CIFAR10_CLASSES)}
            dataset_name = "CIFAR-10"
        else:
            raise ValueError(f"Unsupported dataset: {config.dataset}")

        config.input_dim = X.shape[1]
        labels = labels.astype(np.int64)
        labels_full = labels
        _maybe_init_vision_shape(config, dataset_name)
        print(f"Loaded {len(X)} samples from {dataset_name}")
        print(f"Input dim: {config.input_dim}")
        if not (0.0 <= config.test_split < 1.0):
            msg = "test_split must be in [0.0, 1.0)."
            raise ValueError(msg)

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

        print(f"Train/test split: {len(X_train)}/{len(X_test)} (test={config.test_split:.2f})")

        X_train_cpu = X_train.clone()
        X_test_cpu = X_test.clone()
        data_snapshot = {
            "X_train": X_train_cpu,
            "X_test": X_test_cpu,
            "labels_train": labels_train,
            "labels_test": labels_test,
            "colors_train": colors_train,
            "colors_test": colors_test,
            "dataset_ids": dataset_ids,
            "dataset_name": dataset_name,
        }

    mlflow_active = _start_mlflow_run(
        config,
        extra_params={
            "dataset_name": dataset_name,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "start_epoch": start_epoch,
        },
    )

    for candidate in benchmarks_candidates:
        if os.path.exists(candidate):
            benchmarks_payload = load_benchmarks(candidate)
            benchmarks_path = candidate
            break

    if benchmarks_payload is not None:
        bench_config = benchmarks_payload.get("config", {})
        benchmarks_compatible = _benchmarks_compatible(bench_config, config)
        if benchmarks_compatible:
            benchmarks_state = benchmarks_payload.get("state", {})
            benchmarks_metrics = benchmarks_payload.get("metrics", {})
            benchmarks_dims = benchmarks_payload.get("dims", {})
            print(f"Loaded benchmarks from: {benchmarks_path}")
        else:
            print(
                f"Benchmarks found at {benchmarks_path} but config mismatch; "
                "training baselines from scratch."
            )
            benchmarks_payload = None

    # Create TopoEncoder first to get its parameter count
    model_atlas = TopoEncoderPrimitives(
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
        vision_preproc=config.vision_preproc,
        vision_in_channels=config.vision_in_channels,
        vision_height=config.vision_height,
        vision_width=config.vision_width,
        vision_num_rotations=config.vision_num_rotations,
        vision_kernel_size=config.vision_kernel_size,
        vision_use_reflections=config.vision_use_reflections,
        vision_norm_nonlinearity=config.vision_norm_nonlinearity,
        vision_norm_bias=config.vision_norm_bias,
        vision_backbone_type=config.vision_backbone_type,
        vision_cifar_base_channels=config.vision_cifar_base_channels,
        vision_cifar_bundle_size=config.vision_cifar_bundle_size,
        vision_soft_equiv=config.vision_soft_equiv,
        vision_soft_equiv_bundle_size=config.vision_soft_equiv_bundle_size,
        vision_soft_equiv_hidden_dim=config.vision_soft_equiv_hidden_dim,
        vision_soft_equiv_use_spectral_norm=config.vision_soft_equiv_use_spectral_norm,
        vision_soft_equiv_zero_self_mixing=config.vision_soft_equiv_zero_self_mixing,
        vision_soft_equiv_alpha=config.vision_soft_equiv_alpha,
        vision_soft_equiv_per_block=config.vision_soft_equiv_per_block,
        vision_standard_head=config.vision_standard_head,
        vision_standard_head_blocks=config.vision_standard_head_blocks,
        vision_spectral_head=config.vision_spectral_head,
        soft_equiv_metric=config.soft_equiv_metric,
        soft_equiv_bundle_size=config.soft_equiv_bundle_size,
        soft_equiv_hidden_dim=config.soft_equiv_hidden_dim,
        soft_equiv_use_spectral_norm=config.soft_equiv_use_spectral_norm,
        soft_equiv_zero_self_mixing=config.soft_equiv_zero_self_mixing,
        soft_equiv_soft_assign=config.soft_equiv_soft_assign,
        soft_equiv_temperature=config.soft_equiv_temperature,
    )
    if resume_state is not None and resume_state.get("atlas") is not None:
        model_atlas.load_state_dict(resume_state["atlas"])
    topo_params = count_parameters(model_atlas)

    # Create StandardVQ with matching parameter count (fair comparison)
    benchmarks_std_state = benchmarks_state.get("std") if benchmarks_compatible else None
    benchmarks_ae_state = benchmarks_state.get("ae") if benchmarks_compatible else None
    benchmarks_loaded_std = False
    benchmarks_loaded_ae = False
    model_std = None
    opt_std = None
    std_params = 0
    std_hidden_dim = 0
    if not config.disable_vq:
        if benchmarks_std_state is not None and benchmarks_dims.get("std_hidden_dim"):
            std_hidden_dim = int(benchmarks_dims["std_hidden_dim"])
        elif benchmarks_std_state is not None and benchmarks_metrics.get("std_hidden_dim"):
            std_hidden_dim = int(benchmarks_metrics["std_hidden_dim"])
        elif resume_metrics.get("std_hidden_dim"):
            std_hidden_dim = int(resume_metrics["std_hidden_dim"])
        else:
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
        if benchmarks_std_state is not None:
            model_std.load_state_dict(benchmarks_std_state)
            benchmarks_loaded_std = True
        elif resume_state is not None and resume_state.get("std") is not None:
            model_std.load_state_dict(resume_state["std"])
        std_params = count_parameters(model_std)

    # Create VanillaAE with similar parameter count (reconstruction baseline)
    model_ae = None
    opt_ae = None
    ae_params = 0
    ae_hidden_dim = 0
    if not config.disable_ae:
        if benchmarks_ae_state is not None and benchmarks_dims.get("ae_hidden_dim"):
            ae_hidden_dim = int(benchmarks_dims["ae_hidden_dim"])
        elif benchmarks_ae_state is not None and benchmarks_metrics.get("ae_hidden_dim"):
            ae_hidden_dim = int(benchmarks_metrics["ae_hidden_dim"])
        elif resume_metrics.get("ae_hidden_dim"):
            ae_hidden_dim = int(resume_metrics["ae_hidden_dim"])
        else:
            ae_hidden_dim = compute_matching_hidden_dim(
                target_params=topo_params,
                input_dim=config.input_dim,
                latent_dim=config.latent_dim,
                num_codes=0,  # No codebook in AE
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
        if benchmarks_ae_state is not None:
            model_ae.load_state_dict(benchmarks_ae_state)
            benchmarks_loaded_ae = True
        elif resume_state is not None and resume_state.get("ae") is not None:
            model_ae.load_state_dict(resume_state["ae"])
        ae_params = count_parameters(model_ae)

    print("\nModel Parameters (fair comparison):")
    print(f"  TopoEncoder: {topo_params:,} params (hidden_dim={config.hidden_dim})")
    if not config.disable_vq:
        print(f"  StandardVQ:  {std_params:,} params (hidden_dim={std_hidden_dim})")
    else:
        print("  StandardVQ:  DISABLED")
    if not config.disable_ae:
        print(f"  VanillaAE:   {ae_params:,} params (hidden_dim={ae_hidden_dim})")
    else:
        print("  VanillaAE:   DISABLED")

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
    train_std = model_std is not None
    train_ae = model_ae is not None
    if benchmarks_loaded_std and model_std is not None:
        train_std = False
        model_std.eval()
        print("  StandardVQ: loaded from benchmarks (frozen)")
    if benchmarks_loaded_ae and model_ae is not None:
        train_ae = False
        model_ae.eval()
        print("  VanillaAE: loaded from benchmarks (frozen)")

    if mlflow_active:
        mlflow.log_params({
            "topo_params": topo_params,
            "std_params": std_params,
            "ae_params": ae_params,
            "std_hidden_dim": std_hidden_dim,
            "ae_hidden_dim": ae_hidden_dim,
            "benchmarks_loaded_std": benchmarks_loaded_std,
            "benchmarks_loaded_ae": benchmarks_loaded_ae,
            "train_std": train_std,
            "train_ae": train_ae,
            "baseline_attn": config.baseline_attn,
            "baseline_attn_tokens": config.baseline_attn_tokens,
            "baseline_attn_dim": config.baseline_attn_dim,
            "baseline_attn_heads": config.baseline_attn_heads,
            "baseline_attn_dropout": config.baseline_attn_dropout,
        })

    # Initialize Jump Operator for chart gluing
    jump_op = FactorizedJumpOperator(
        num_charts=config.num_charts,
        latent_dim=config.latent_dim,
        global_rank=config.jump_global_rank,
    ).to(device)
    if resume_state is not None and resume_state.get("jump") is not None:
        jump_op.load_state_dict(resume_state["jump"])
    print(f"  Jump Operator: {count_parameters(jump_op):,} params")

    # Supervised topology loss (chart-to-class mapping)
    supervised_loss = None
    num_classes = int(labels_full.max()) + 1 if labels_full.size else config.num_classes
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
        if resume_state is not None and resume_state.get("supervised") is not None:
            supervised_loss.load_state_dict(resume_state["supervised"])
        print(
            "  Supervised Topology: "
            f"classes={num_classes}, "
            f"λ_purity={config.sup_purity_weight}, "
            f"λ_balance={config.sup_balance_weight}, "
            f"λ_metric={config.sup_metric_weight}"
        )

    # Learned precisions for recon/vq/supervised balance
    precision_module = None
    if config.use_learned_precisions:
        precision_params: dict[str, nn.Parameter] = {
            "recon": nn.Parameter(torch.zeros(())),
            "vq": nn.Parameter(torch.zeros(())),
        }
        if supervised_loss is not None:
            precision_params["sup"] = nn.Parameter(torch.zeros(()))
        precision_module = nn.ParameterDict(precision_params).to(device)
        if resume_state is not None and resume_state.get("precisions") is not None:
            precision_module.load_state_dict(resume_state["precisions"])
        print("  Learned Precisions: enabled")

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
        if resume_state is not None and resume_state.get("classifier") is not None:
            classifier_head.load_state_dict(resume_state["classifier"])
        print(
            "  Classifier Readout: "
            f"classes={num_classes}, "
            f"bundle_size={classifier_bundle_size or 'global'}, "
            f"lr={classifier_lr}"
        )

    std_classifier_head = None
    ae_classifier_head = None
    if config.enable_classifier_head:
        if model_std is not None:
            std_classifier_head = BaselineClassifier(
                latent_dim=config.latent_dim,
                num_classes=num_classes,
                hidden_dim=config.hidden_dim,
            ).to(device)
            if resume_state is not None and resume_state.get("classifier_std") is not None:
                std_classifier_head.load_state_dict(resume_state["classifier_std"])
        if model_ae is not None:
            ae_classifier_head = BaselineClassifier(
                latent_dim=config.latent_dim,
                num_classes=num_classes,
                hidden_dim=config.hidden_dim,
            ).to(device)
            if resume_state is not None and resume_state.get("classifier_ae") is not None:
                ae_classifier_head.load_state_dict(resume_state["classifier_ae"])
        if std_classifier_head is not None or ae_classifier_head is not None:
            print("  Benchmark Readout: enabled")

    # Optimizers (joint training of atlas model and jump operator)
    if train_std and model_std is not None:
        opt_std = optim.Adam(model_std.parameters(), lr=config.lr)
    atlas_params = list(model_atlas.parameters()) + list(jump_op.parameters())
    if supervised_loss is not None:
        atlas_params.extend(list(supervised_loss.parameters()))
    if precision_module is not None:
        atlas_params.extend(list(precision_module.parameters()))
    opt_atlas = optim.Adam(atlas_params, lr=config.lr)
    opt_classifier = None
    if classifier_head is not None:
        opt_classifier = optim.Adam(classifier_head.parameters(), lr=classifier_lr)
    opt_classifier_std = None
    if std_classifier_head is not None:
        opt_classifier_std = optim.Adam(std_classifier_head.parameters(), lr=classifier_lr)
    opt_classifier_ae = None
    if ae_classifier_head is not None:
        opt_classifier_ae = optim.Adam(ae_classifier_head.parameters(), lr=classifier_lr)
    if train_ae and model_ae is not None:
        opt_ae = optim.Adam(model_ae.parameters(), lr=config.lr)

    # Learning rate scheduler (cosine annealing)
    scheduler = None
    if config.use_scheduler and not config.adaptive_lr:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt_atlas, T_max=config.epochs, eta_min=config.min_lr
        )
    elif config.use_scheduler and config.adaptive_lr:
        print("  Scheduler disabled (adaptive_lr=True)")

    if resume_optim:
        if opt_atlas is not None and resume_optim.get("atlas") is not None:
            opt_atlas.load_state_dict(resume_optim["atlas"])
            _move_optimizer_state(opt_atlas, device)
        if opt_std is not None and resume_optim.get("std") is not None:
            opt_std.load_state_dict(resume_optim["std"])
            _move_optimizer_state(opt_std, device)
        if opt_ae is not None and resume_optim.get("ae") is not None:
            opt_ae.load_state_dict(resume_optim["ae"])
            _move_optimizer_state(opt_ae, device)
        if opt_classifier is not None and resume_optim.get("classifier") is not None:
            opt_classifier.load_state_dict(resume_optim["classifier"])
            _move_optimizer_state(opt_classifier, device)
        if opt_classifier_std is not None and resume_optim.get("classifier_std") is not None:
            opt_classifier_std.load_state_dict(resume_optim["classifier_std"])
            _move_optimizer_state(opt_classifier_std, device)
        if opt_classifier_ae is not None and resume_optim.get("classifier_ae") is not None:
            opt_classifier_ae.load_state_dict(resume_optim["classifier_ae"])
            _move_optimizer_state(opt_classifier_ae, device)
        if scheduler is not None and resume_optim.get("scheduler") is not None:
            scheduler.load_state_dict(resume_optim["scheduler"])

    # Create data loader for minibatching (data already on device)
    from torch.utils.data import DataLoader, TensorDataset

    labels_train_t = torch.from_numpy(labels_train).long().to(device)
    labels_test_t = torch.from_numpy(labels_test).long().to(device)
    colors_train_t = torch.from_numpy(colors_train).float().to(device)
    torch.from_numpy(colors_test).float().to(device)
    dataset = TensorDataset(X_train, labels_train_t, colors_train_t)
    batch_size = config.batch_size if config.batch_size > 0 else len(X_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batches_per_epoch = max(1, len(dataloader))
    adaptive_update_scale = 1.0 / batches_per_epoch
    if mlflow_active:
        mlflow.log_param("batches_per_epoch", batches_per_epoch)

    # Training history
    std_losses = list(resume_metrics.get("std_losses", []))
    atlas_losses = list(resume_metrics.get("atlas_losses", []))
    ae_losses = list(resume_metrics.get("ae_losses", []))  # VanillaAE baseline
    loss_components = _init_loss_components()
    if resume_metrics.get("loss_components"):
        for key, values in resume_metrics["loss_components"].items():
            loss_components.setdefault(key, [])
            loss_components[key] = list(values)
    info_metrics = _init_info_metrics()
    if resume_metrics.get("info_metrics"):
        for key, values in resume_metrics["info_metrics"].items():
            info_metrics.setdefault(key, [])
            info_metrics[key] = list(values)

    adaptive_weight_names = [
        "entropy_weight",
        "consistency_weight",
        "variance_weight",
        "diversity_weight",
        "separation_weight",
        "codebook_center_weight",
        "chart_center_sep_weight",
        "residual_scale_weight",
        "window_weight",
        "disentangle_weight",
        "orthogonality_weight",
        "code_entropy_weight",
        "per_chart_code_entropy_weight",
        "kl_prior_weight",
        "orbit_weight",
        "vicreg_inv_weight",
        "jump_weight",
        "sup_purity_weight",
        "sup_balance_weight",
        "sup_metric_weight",
    ]
    adaptive_weight_state = _restore_weight_state(config, resume_metrics, adaptive_weight_names)
    for name, state in adaptive_weight_state.items():
        setattr(config, name, state.value)

    adaptive_target_names = [
        "variance",
        "separation",
        "codebook_center",
        "chart_center_sep",
        "residual_scale",
        "disentangle",
        "orthogonality",
        "kl_prior",
        "orbit",
        "vicreg_inv",
        "jump",
        "sup_purity",
        "sup_balance",
        "sup_metric",
    ]
    adaptive_target_state = _restore_target_state(resume_metrics, adaptive_target_names)

    adaptive_lr_state = _restore_lr_state(config, resume_metrics)
    lr_current = adaptive_lr_state.lr
    lr_max = config.lr_max if config.lr_max > 0 else config.lr
    if config.adaptive_lr:
        _set_optimizer_lr(opt_atlas, lr_current)

    print("=" * 60)
    print("Training TopoEncoder (Attentive Atlas)")
    print(f"  Epochs: {config.epochs}, LR: {config.lr}, Batch size: {batch_size}")
    print(f"  Charts: {config.num_charts}, Codes/chart: {config.codes_per_chart}")
    print(f"  λ: entropy={config.entropy_weight}, consistency={config.consistency_weight}")
    if start_epoch > 0:
        print(f"  Resuming at epoch {start_epoch}")
    print("=" * 60)

    for epoch in tqdm(range(start_epoch, config.epochs + 1), desc="Training", unit="epoch"):
        # Accumulate batch losses for epoch average
        epoch_std_loss = 0.0
        epoch_atlas_loss = 0.0
        epoch_ae_loss = 0.0
        epoch_losses = dict.fromkeys(loss_components.keys(), 0.0)
        epoch_info = {
            "I_XK": 0.0,
            "H_K": 0.0,
            "H_K_given_X": 0.0,
            "code_entropy": 0.0,
            "per_chart_code_entropy": 0.0,
            "grad_norm": 0.0,
            "param_norm": 0.0,
            "update_ratio": 0.0,
            "lr": 0.0,
        }
        epoch_lr_loss_signal = 0.0
        epoch_lr_grad_norm = 0.0
        epoch_lr_param_norm = 0.0
        n_batches = 0

        for batch_X, batch_labels, _batch_colors in dataloader:
            n_batches += 1

            # --- Standard VQ Step ---
            loss_s = torch.tensor(0.0, device=device)
            if model_std is not None:
                if train_std and opt_std is not None:
                    recon_s, vq_loss_s, _ = model_std(batch_X)
                    loss_s = F.mse_loss(recon_s, batch_X) + vq_loss_s
                    opt_std.zero_grad()
                    loss_s.backward()
                    opt_std.step()
                else:
                    with torch.no_grad():
                        recon_s, vq_loss_s, _ = model_std(batch_X)
                        loss_s = F.mse_loss(recon_s, batch_X) + vq_loss_s

            # --- Vanilla AE Step (reconstruction baseline) ---
            loss_ae = torch.tensor(0.0, device=device)
            z_ae = None
            if model_ae is not None:
                if train_ae and opt_ae is not None:
                    recon_ae, z_ae = model_ae(batch_X)
                    loss_ae = F.mse_loss(recon_ae, batch_X)
                    opt_ae.zero_grad()
                    loss_ae.backward()
                    opt_ae.step()
                else:
                    with torch.no_grad():
                        recon_ae, z_ae = model_ae(batch_X)
                        loss_ae = F.mse_loss(recon_ae, batch_X)

            # --- Baseline classifier readouts (detached) ---
            std_cls_loss = torch.tensor(0.0, device=device)
            std_cls_acc = torch.tensor(0.0, device=device)
            if (
                std_classifier_head is not None
                and opt_classifier_std is not None
                and model_std is not None
            ):
                with torch.no_grad():
                    z_std = model_std.encoder(batch_X)
                logits_std = std_classifier_head(z_std)
                std_cls_loss = F.cross_entropy(logits_std, batch_labels)
                opt_classifier_std.zero_grad()
                std_cls_loss.backward()
                opt_classifier_std.step()
                std_cls_acc = (logits_std.detach().argmax(dim=1) == batch_labels).float().mean()

            ae_cls_loss = torch.tensor(0.0, device=device)
            ae_cls_acc = torch.tensor(0.0, device=device)
            if (
                ae_classifier_head is not None
                and opt_classifier_ae is not None
                and z_ae is not None
            ):
                logits_ae = ae_classifier_head(z_ae.detach())
                ae_cls_loss = F.cross_entropy(logits_ae, batch_labels)
                opt_classifier_ae.zero_grad()
                ae_cls_loss.backward()
                opt_classifier_ae.step()
                ae_cls_acc = (logits_ae.detach().argmax(dim=1) == batch_labels).float().mean()

            # --- Atlas Step (dreaming mode: decoder infers routing from z_geo) ---
            # Get encoder outputs (need z_geo for regularization losses, z_n_all_charts for jump)
            (
                K_chart,
                _,
                z_n,
                z_tex,
                enc_w,
                z_geo,
                vq_loss_a,
                indices_stack,
                z_n_all_charts,
                _c_bar,
            ) = model_atlas.encoder(
                batch_X,
                hard_routing=config.hard_routing,
                hard_routing_tau=config.hard_routing_tau,
            )

            # Decoder forward (optional hard routing via encoder chart index)
            router_override = enc_w if config.hard_routing else None
            recon_a, dec_w = model_atlas.decoder(
                z_geo,
                z_tex,
                chart_index=None,
                router_weights=router_override,
                hard_routing=config.hard_routing,
                hard_routing_tau=config.hard_routing_tau,
            )

            # Core losses
            recon_loss_a = F.mse_loss(recon_a, batch_X)
            entropy_value = compute_routing_entropy(enc_w)
            entropy_loss = math.log(config.num_charts) - entropy_value
            consistency = model_atlas.compute_consistency_loss(enc_w, dec_w)

            # Tier 1 losses (low overhead)
            # Note: geometry-aware losses (variance/disentangle/separation/centering/jump/vicreg)
            # are hyperbolic internally; tangent-space losses (KL, residual scale) stay Euclidean.
            var_loss = compute_variance_loss(z_geo)
            div_loss = compute_diversity_loss(enc_w, config.num_charts)
            sep_loss = compute_separation_loss(
                z_geo, enc_w, config.num_charts, config.separation_margin
            )
            codebook_center_loss = torch.tensor(0.0, device=device)
            if config.codebook_center_weight > 0 or config.adaptive_weights:
                codebook_center_loss = compute_codebook_centering_loss(
                    model_atlas.encoder.codebook
                )
            chart_center_sep_loss = torch.tensor(0.0, device=device)
            if config.chart_center_sep_weight > 0 or config.adaptive_weights:
                chart_center_sep_loss = compute_chart_center_separation_loss(
                    model_atlas.encoder.chart_centers, config.chart_center_sep_margin
                )
            residual_scale_loss = torch.tensor(0.0, device=device)
            if config.residual_scale_weight > 0 or config.adaptive_weights:
                residual_scale_loss = compute_residual_scale_loss(z_n)
            soft_equiv_l1 = torch.tensor(0.0, device=device)
            if config.soft_equiv_metric:
                soft_equiv_l1 = model_atlas.encoder.soft_equiv_l1_loss()
            soft_equiv_log_ratio = torch.tensor(0.0, device=device)
            if config.soft_equiv_metric and config.soft_equiv_log_ratio_weight > 0:
                soft_equiv_log_ratio = model_atlas.encoder.soft_equiv_log_ratio_loss()

            # Tier 2 losses (medium overhead)
            window_loss, window_info = compute_window_loss(
                enc_w, config.num_charts, config.window_eps_ground
            )
            dis_loss = compute_disentangle_loss(z_geo, enc_w)
            if config.adaptive_lr:
                ixk_val = window_info["I_XK"]
                if adaptive_lr_state.ixk_ema <= 0.0:
                    adaptive_lr_state.ixk_ema = ixk_val
                else:
                    adaptive_lr_state.ixk_ema = (
                        config.lr_grounding_ema_decay * adaptive_lr_state.ixk_ema
                        + (1.0 - config.lr_grounding_ema_decay) * ixk_val
                    )

            # Tier 3 losses (geometry/codebook health)
            orth_loss = torch.tensor(0.0, device=device)
            if config.orthogonality_weight > 0 or config.adaptive_weights:
                orth_loss = compute_orthogonality_loss(model_atlas)

            code_ent_loss = torch.tensor(0.0, device=device)
            if config.code_entropy_weight > 0 or config.adaptive_weights:
                code_ent_loss = compute_code_entropy_loss(indices_stack, config.codes_per_chart)

            per_chart_code_ent_loss = torch.tensor(0.0, device=device)
            if config.per_chart_code_entropy_weight > 0 or config.adaptive_weights:
                per_chart_code_ent_loss = compute_per_chart_code_entropy_loss(
                    indices_stack, K_chart, config.num_charts, config.codes_per_chart
                )
            max_code_entropy = math.log(config.codes_per_chart)
            code_entropy_value = max_code_entropy - code_ent_loss.item()
            per_chart_entropy_value = max_code_entropy - per_chart_code_ent_loss.item()

            # Tier 4 losses (invariance - expensive, conditional computation)
            # KL prior (cheap, compute if enabled)
            if config.kl_prior_weight > 0 or config.adaptive_weights:
                kl_loss = compute_kl_prior_loss(z_n, z_tex)
            else:
                kl_loss = torch.tensor(0.0, device=device)

            # Orbit and VICReg invariance (expensive - share augmented forward pass)
            orbit_loss = torch.tensor(0.0, device=device)
            vicreg_loss = torch.tensor(0.0, device=device)

            if config.orbit_weight > 0 or config.vicreg_inv_weight > 0:
                # Single augmented forward pass (shared between both losses)
                x_aug = augment_inputs(
                    batch_X,
                    config.dataset,
                    config.augment_noise_std,
                    config.augment_rotation_max,
                )
                _, _, _, _, enc_w_aug, z_geo_aug, _, _, _, _c_bar_aug = model_atlas.encoder(
                    x_aug,
                    hard_routing=config.hard_routing,
                    hard_routing_tau=config.hard_routing_tau,
                )

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

            recon_term = recon_loss_a
            vq_term = vq_loss_a
            sup_term = sup_total
            if precision_module is not None:
                recon_term = _apply_precision(recon_loss_a, precision_module["recon"])
                vq_term = _apply_precision(vq_loss_a, precision_module["vq"])
                if supervised_loss is not None and "sup" in precision_module:
                    sup_term = _apply_precision(sup_total, precision_module["sup"])

            # Total loss
            loss_a = (
                recon_term
                + vq_term
                + config.entropy_weight * entropy_loss
                + config.consistency_weight * consistency
                # Tier 1
                + config.variance_weight * var_loss
                + config.diversity_weight * div_loss
                + config.separation_weight * sep_loss
                + config.codebook_center_weight * codebook_center_loss
                + config.chart_center_sep_weight * chart_center_sep_loss
                + config.residual_scale_weight * residual_scale_loss
                + config.soft_equiv_l1_weight * soft_equiv_l1
                + config.soft_equiv_log_ratio_weight * soft_equiv_log_ratio
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
                + config.sup_weight * sup_term
            )

            lr_loss_signal = (
                recon_term
                + vq_term
                + config.entropy_weight * entropy_loss
                + config.consistency_weight * consistency
            )
            if supervised_loss is not None:
                lr_loss_signal += config.sup_weight * sup_term

            opt_atlas.zero_grad()
            loss_a.backward()
            # Gradient clipping (prevents instability from competing losses)
            if config.grad_clip > 0:
                all_params = list(model_atlas.parameters()) + list(jump_op.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, config.grad_clip)
            grad_norm_val = 0.0
            param_norm_val = 0.0
            update_ratio = 0.0
            if config.adaptive_lr:
                grad_norm = _compute_grad_norm(atlas_params)
                param_norm = _compute_param_norm(atlas_params)
                epoch_lr_loss_signal += lr_loss_signal.item()
                epoch_lr_grad_norm += grad_norm
                epoch_lr_param_norm += param_norm
                grad_norm_val = grad_norm
                param_norm_val = param_norm
                update_ratio = (
                    lr_current * grad_norm / (param_norm + 1e-12) if param_norm > 0.0 else 0.0
                )
            opt_atlas.step()

            if config.adaptive_weights:
                if epoch < config.adaptive_warmup_epochs:
                    target_decay = config.adaptive_target_ema_decay
                    _update_target_state(
                        adaptive_target_state, "variance", var_loss.item(), target_decay
                    )
                    _update_target_state(
                        adaptive_target_state, "separation", sep_loss.item(), target_decay
                    )
                    _update_target_state(
                        adaptive_target_state,
                        "codebook_center",
                        codebook_center_loss.item(),
                        target_decay,
                    )
                    _update_target_state(
                        adaptive_target_state,
                        "chart_center_sep",
                        chart_center_sep_loss.item(),
                        target_decay,
                    )
                    _update_target_state(
                        adaptive_target_state,
                        "residual_scale",
                        residual_scale_loss.item(),
                        target_decay,
                    )
                    _update_target_state(
                        adaptive_target_state, "disentangle", dis_loss.item(), target_decay
                    )
                    _update_target_state(
                        adaptive_target_state,
                        "orthogonality",
                        orth_loss.item(),
                        target_decay,
                    )
                    _update_target_state(
                        adaptive_target_state, "kl_prior", kl_loss.item(), target_decay
                    )
                    _update_target_state(
                        adaptive_target_state, "orbit", orbit_loss.item(), target_decay
                    )
                    _update_target_state(
                        adaptive_target_state, "vicreg_inv", vicreg_loss.item(), target_decay
                    )
                    if current_jump_weight > 0:
                        _update_target_state(
                            adaptive_target_state, "jump", jump_loss.item(), target_decay
                        )
                    if supervised_loss is not None:
                        _update_target_state(
                            adaptive_target_state,
                            "sup_purity",
                            sup_purity.item(),
                            target_decay,
                        )
                        _update_target_state(
                            adaptive_target_state,
                            "sup_balance",
                            sup_balance.item(),
                            target_decay,
                        )
                        _update_target_state(
                            adaptive_target_state,
                            "sup_metric",
                            sup_metric.item(),
                            target_decay,
                        )
                if epoch >= config.adaptive_warmup_epochs:
                    min_val = config.adaptive_lambda_min
                    max_val = config.adaptive_lambda_max
                    dual_eta = config.adaptive_dual_eta * adaptive_update_scale
                    pi_kp = config.adaptive_pi_kp * adaptive_update_scale
                    pi_ki = config.adaptive_pi_ki * adaptive_update_scale
                    pi_kd = config.adaptive_pi_kd * adaptive_update_scale
                    clip = config.adaptive_violation_clip
                    target_ratio = config.adaptive_target_ratio
                    target_min = config.adaptive_target_min
                    log_k = math.log(config.num_charts)
                    entropy_target = config.entropy_target_ratio * log_k
                    hk_target = config.hk_target_ratio * log_k

                    entropy_error = entropy_target - entropy_value.item()
                    config.entropy_weight = _update_pi_controller(
                        adaptive_weight_state["entropy_weight"],
                        entropy_error,
                        pi_kp,
                        pi_ki,
                        pi_kd,
                        min_val,
                        max_val,
                    )

                    diversity_error = hk_target - window_info["H_K"]
                    config.diversity_weight = _update_pi_controller(
                        adaptive_weight_state["diversity_weight"],
                        diversity_error,
                        pi_kp,
                        pi_ki,
                        pi_kd,
                        min_val,
                        max_val,
                    )

                    max_code_entropy = math.log(config.codes_per_chart)
                    code_entropy_value = max_code_entropy - code_ent_loss.item()
                    code_entropy_target = config.code_entropy_target_ratio * max_code_entropy
                    code_entropy_error = code_entropy_target - code_entropy_value
                    config.code_entropy_weight = _update_pi_controller(
                        adaptive_weight_state["code_entropy_weight"],
                        code_entropy_error,
                        pi_kp,
                        pi_ki,
                        pi_kd,
                        min_val,
                        max_val,
                    )

                    per_chart_entropy_value = max_code_entropy - per_chart_code_ent_loss.item()
                    per_chart_entropy_error = code_entropy_target - per_chart_entropy_value
                    config.per_chart_code_entropy_weight = _update_pi_controller(
                        adaptive_weight_state["per_chart_code_entropy_weight"],
                        per_chart_entropy_error,
                        pi_kp,
                        pi_ki,
                        pi_kd,
                        min_val,
                        max_val,
                    )

                    window_violation = config.window_eps_ground - window_info["I_XK"]
                    config.window_weight = _update_primal_dual(
                        adaptive_weight_state["window_weight"],
                        _clip_violation(window_violation, clip),
                        dual_eta,
                        min_val,
                        max_val,
                    )
                    config.consistency_weight = _update_primal_dual(
                        adaptive_weight_state["consistency_weight"],
                        _clip_violation(
                            consistency.item() - config.consistency_target,
                            clip,
                        ),
                        dual_eta,
                        min_val,
                        max_val,
                    )
                    config.variance_weight = _update_primal_dual(
                        adaptive_weight_state["variance_weight"],
                        _clip_violation(
                            _compute_target_error(
                                adaptive_target_state,
                                "variance",
                                var_loss.item(),
                                target_ratio,
                                target_min,
                            ),
                            clip,
                        ),
                        dual_eta,
                        min_val,
                        max_val,
                    )
                    config.separation_weight = _update_primal_dual(
                        adaptive_weight_state["separation_weight"],
                        _clip_violation(
                            _compute_target_error(
                                adaptive_target_state,
                                "separation",
                                sep_loss.item(),
                                target_ratio,
                                target_min,
                            ),
                            clip,
                        ),
                        dual_eta,
                        min_val,
                        max_val,
                    )
                    config.codebook_center_weight = _update_primal_dual(
                        adaptive_weight_state["codebook_center_weight"],
                        _clip_violation(
                            _compute_target_error(
                                adaptive_target_state,
                                "codebook_center",
                                codebook_center_loss.item(),
                                target_ratio,
                                target_min,
                            ),
                            clip,
                        ),
                        dual_eta,
                        min_val,
                        max_val,
                    )
                    config.chart_center_sep_weight = _update_primal_dual(
                        adaptive_weight_state["chart_center_sep_weight"],
                        _clip_violation(
                            _compute_target_error(
                                adaptive_target_state,
                                "chart_center_sep",
                                chart_center_sep_loss.item(),
                                target_ratio,
                                target_min,
                            ),
                            clip,
                        ),
                        dual_eta,
                        min_val,
                        max_val,
                    )
                    config.residual_scale_weight = _update_primal_dual(
                        adaptive_weight_state["residual_scale_weight"],
                        _clip_violation(
                            _compute_target_error(
                                adaptive_target_state,
                                "residual_scale",
                                residual_scale_loss.item(),
                                target_ratio,
                                target_min,
                            ),
                            clip,
                        ),
                        dual_eta,
                        min_val,
                        max_val,
                    )
                    config.disentangle_weight = _update_primal_dual(
                        adaptive_weight_state["disentangle_weight"],
                        _clip_violation(
                            _compute_target_error(
                                adaptive_target_state,
                                "disentangle",
                                dis_loss.item(),
                                target_ratio,
                                target_min,
                            ),
                            clip,
                        ),
                        dual_eta,
                        min_val,
                        max_val,
                    )
                    config.orthogonality_weight = _update_primal_dual(
                        adaptive_weight_state["orthogonality_weight"],
                        _clip_violation(
                            _compute_target_error(
                                adaptive_target_state,
                                "orthogonality",
                                orth_loss.item(),
                                target_ratio,
                                target_min,
                            ),
                            clip,
                        ),
                        dual_eta,
                        min_val,
                        max_val,
                    )
                    config.kl_prior_weight = _update_primal_dual(
                        adaptive_weight_state["kl_prior_weight"],
                        _clip_violation(
                            _compute_target_error(
                                adaptive_target_state,
                                "kl_prior",
                                kl_loss.item(),
                                target_ratio,
                                target_min,
                            ),
                            clip,
                        ),
                        dual_eta,
                        min_val,
                        max_val,
                    )
                    if config.orbit_weight > 0:
                        config.orbit_weight = _update_primal_dual(
                            adaptive_weight_state["orbit_weight"],
                            _clip_violation(
                                _compute_target_error(
                                    adaptive_target_state,
                                    "orbit",
                                    orbit_loss.item(),
                                    target_ratio,
                                    target_min,
                                ),
                                clip,
                            ),
                            dual_eta,
                            min_val,
                            max_val,
                        )
                    if config.vicreg_inv_weight > 0:
                        config.vicreg_inv_weight = _update_primal_dual(
                            adaptive_weight_state["vicreg_inv_weight"],
                            _clip_violation(
                                _compute_target_error(
                                    adaptive_target_state,
                                    "vicreg_inv",
                                    vicreg_loss.item(),
                                    target_ratio,
                                    target_min,
                                ),
                                clip,
                            ),
                            dual_eta,
                            min_val,
                            max_val,
                        )
                    if epoch >= config.jump_warmup:
                        config.jump_weight = _update_primal_dual(
                            adaptive_weight_state["jump_weight"],
                            _clip_violation(
                                _compute_target_error(
                                    adaptive_target_state,
                                    "jump",
                                    jump_loss.item(),
                                    target_ratio,
                                    target_min,
                                ),
                                clip,
                            ),
                            dual_eta,
                            min_val,
                            max_val,
                        )
                    if supervised_loss is not None:
                        config.sup_purity_weight = _update_primal_dual(
                            adaptive_weight_state["sup_purity_weight"],
                            _clip_violation(
                                _compute_target_error(
                                    adaptive_target_state,
                                    "sup_purity",
                                    sup_purity.item(),
                                    target_ratio,
                                    target_min,
                                ),
                                clip,
                            ),
                            dual_eta,
                            min_val,
                            max_val,
                        )
                        config.sup_balance_weight = _update_primal_dual(
                            adaptive_weight_state["sup_balance_weight"],
                            _clip_violation(
                                _compute_target_error(
                                    adaptive_target_state,
                                    "sup_balance",
                                    sup_balance.item(),
                                    target_ratio,
                                    target_min,
                                ),
                                clip,
                            ),
                            dual_eta,
                            min_val,
                            max_val,
                        )
                        config.sup_metric_weight = _update_primal_dual(
                            adaptive_weight_state["sup_metric_weight"],
                            _clip_violation(
                                _compute_target_error(
                                    adaptive_target_state,
                                    "sup_metric",
                                    sup_metric.item(),
                                    target_ratio,
                                    target_min,
                                ),
                                clip,
                            ),
                            dual_eta,
                            min_val,
                            max_val,
                        )
                        supervised_loss.lambda_purity = config.sup_purity_weight
                        supervised_loss.lambda_balance = config.sup_balance_weight
                        supervised_loss.lambda_metric = config.sup_metric_weight

            # --- Classifier Readout Step (detached) ---
            cls_loss = torch.tensor(0.0, device=device)
            cls_acc = torch.tensor(0.0, device=device)
            if classifier_head is not None and opt_classifier is not None:
                logits = classifier_head(enc_w.detach(), z_geo.detach())
                cls_loss = F.cross_entropy(logits, batch_labels)
                opt_classifier.zero_grad()
                cls_loss.backward()
                opt_classifier.step()
                cls_acc = (logits.detach().argmax(dim=1) == batch_labels).float().mean()

            # Accumulate batch losses
            epoch_std_loss += loss_s.item()
            epoch_atlas_loss += loss_a.item()
            epoch_ae_loss += loss_ae.item()
            epoch_losses["recon"] += recon_loss_a.item()
            epoch_losses["vq"] += vq_loss_a.item()
            epoch_losses["entropy"] += entropy_value.item()
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
            epoch_losses["std_cls_loss"] += std_cls_loss.item()
            epoch_losses["std_cls_acc"] += std_cls_acc.item()
            epoch_losses["ae_cls_loss"] += ae_cls_loss.item()
            epoch_losses["ae_cls_acc"] += ae_cls_acc.item()
            epoch_info["I_XK"] += window_info["I_XK"]
            epoch_info["H_K"] += window_info["H_K"]
            epoch_info["H_K_given_X"] += entropy_value.item()
            epoch_info["code_entropy"] += code_entropy_value
            epoch_info["per_chart_code_entropy"] += per_chart_entropy_value
            epoch_info["grad_norm"] += grad_norm_val
            epoch_info["param_norm"] += param_norm_val
            epoch_info["update_ratio"] += update_ratio
            epoch_info["lr"] += opt_atlas.param_groups[0]["lr"]

        # Average over batches
        std_losses.append(epoch_std_loss / n_batches)
        atlas_losses.append(epoch_atlas_loss / n_batches)
        ae_losses.append(epoch_ae_loss / n_batches)
        for k in loss_components.keys():
            loss_components[k].append(epoch_losses[k] / n_batches)
        info_metrics["I_XK"].append(epoch_info["I_XK"] / n_batches)
        info_metrics["H_K"].append(epoch_info["H_K"] / n_batches)
        info_metrics["H_K_given_X"].append(epoch_info["H_K_given_X"] / n_batches)
        info_metrics["code_entropy"].append(epoch_info["code_entropy"] / n_batches)
        info_metrics["per_chart_code_entropy"].append(
            epoch_info["per_chart_code_entropy"] / n_batches
        )
        info_metrics["grad_norm"].append(epoch_info["grad_norm"] / n_batches)
        info_metrics["param_norm"].append(epoch_info["param_norm"] / n_batches)
        info_metrics["update_ratio"].append(epoch_info["update_ratio"] / n_batches)
        info_metrics["lr"].append(epoch_info["lr"] / n_batches)

        if config.adaptive_lr and n_batches > 0:
            loss_val_raw = epoch_lr_loss_signal / n_batches
            grad_norm_avg = epoch_lr_grad_norm / n_batches
            param_norm_avg = epoch_lr_param_norm / n_batches
            loss_ema_raw = adaptive_lr_state.loss_ema or loss_val_raw
            loss_ema_next = (
                config.lr_ema_decay * loss_ema_raw + (1.0 - config.lr_ema_decay) * loss_val_raw
            )
            loss_val = math.log1p(loss_val_raw)
            loss_ema = math.log1p(loss_ema_raw)
            grounding_active = epoch >= config.lr_grounding_warmup_epochs
            if not grounding_active:
                adaptive_lr_state.unstable_steps = 0
                adaptive_lr_state.stable_steps = 0
                adaptive_lr_state.lr = lr_current
                adaptive_lr_state.loss_ema = loss_ema_next
                if adaptive_lr_state.best_loss_ema <= 0.0:
                    adaptive_lr_state.best_loss_ema = loss_ema_next
                adaptive_lr_state.plateau_steps = 0
                _set_optimizer_lr(opt_atlas, lr_current)
                ixk_val = info_metrics["I_XK"][-1]
                if adaptive_lr_state.ixk_ema <= 0.0:
                    ixk_for_check = ixk_val
                else:
                    ixk_for_check = max(ixk_val, adaptive_lr_state.ixk_ema)
                grounding_violation = ixk_for_check < config.window_eps_ground
                if grounding_violation:
                    adaptive_lr_state.unstable_steps += 1
            else:
                ixk_val = info_metrics["I_XK"][-1]
                if adaptive_lr_state.ixk_ema <= 0.0:
                    ixk_for_check = ixk_val
                else:
                    ixk_for_check = max(ixk_val, adaptive_lr_state.ixk_ema)
                grounding_violation = ixk_for_check < config.window_eps_ground
                if grounding_violation:
                    lr_current = max(lr_current, config.lr)
                    adaptive_lr_state.unstable_steps = 0
                    adaptive_lr_state.stable_steps = 0
                    adaptive_lr_state.lr = lr_current
                    adaptive_lr_state.loss_ema = loss_ema_next
                    if adaptive_lr_state.best_loss_ema <= 0.0:
                        adaptive_lr_state.best_loss_ema = loss_ema_next
                    adaptive_lr_state.plateau_steps = 0
                    _set_optimizer_lr(opt_atlas, lr_current)
                else:
                    unstable = loss_val > loss_ema * (1.0 + config.lr_loss_increase_tol)
                    if not math.isfinite(loss_val_raw) or not math.isfinite(grad_norm_avg):
                        unstable = True
                    if not grounding_active and unstable:
                        unstable = False
                    if unstable:
                        adaptive_lr_state.unstable_steps += 1
                        adaptive_lr_state.stable_steps = 0
                        adaptive_lr_state.plateau_steps = 0
                    else:
                        adaptive_lr_state.stable_steps += 1
                        adaptive_lr_state.unstable_steps = 0
                    plateau_triggered = False
                    if math.isfinite(loss_ema_next):
                        if adaptive_lr_state.best_loss_ema <= 0.0 or not math.isfinite(
                            adaptive_lr_state.best_loss_ema
                        ):
                            adaptive_lr_state.best_loss_ema = loss_ema_next
                            adaptive_lr_state.plateau_steps = 0
                        elif config.lr_plateau_patience > 0:
                            if unstable:
                                adaptive_lr_state.plateau_steps = 0
                            else:
                                denom = max(abs(adaptive_lr_state.best_loss_ema), 1e-8)
                                improvement = (
                                    adaptive_lr_state.best_loss_ema - loss_ema_next
                                ) / denom
                                if improvement > config.lr_plateau_tol:
                                    adaptive_lr_state.best_loss_ema = loss_ema_next
                                    adaptive_lr_state.plateau_steps = 0
                                else:
                                    adaptive_lr_state.plateau_steps += 1
                                    if (
                                        adaptive_lr_state.plateau_steps
                                        >= config.lr_plateau_patience
                                    ):
                                        plateau_triggered = True
                                        adaptive_lr_state.plateau_steps = 0
                    lr_cap = lr_max
                    if grad_norm_avg > 0.0 and param_norm_avg > 0.0:
                        lr_cap = min(
                            lr_max,
                            config.lr_max_update_ratio * param_norm_avg / (grad_norm_avg + 1e-12),
                        )
                    increase_factor = config.lr_increase_factor
                    if lr_current < lr_max * config.lr_recovery_threshold:
                        increase_factor = max(increase_factor, config.lr_recovery_factor)
                    min_lr_bound = max(config.lr_min, 1e-12)
                    if (
                        adaptive_lr_state.unstable_steps >= config.lr_unstable_patience
                        or plateau_triggered
                    ):
                        if grounding_active:
                            lr_current = max(min_lr_bound, lr_current * config.lr_decrease_factor)
                        adaptive_lr_state.unstable_steps = 0
                        adaptive_lr_state.stable_steps = 0
                        adaptive_lr_state.plateau_steps = 0
                    elif adaptive_lr_state.stable_steps >= config.lr_stable_patience:
                        lr_current = min(lr_cap, lr_current * increase_factor)
                        adaptive_lr_state.stable_steps = 0
                        adaptive_lr_state.plateau_steps = 0
                    lr_current = min(lr_current, lr_cap)
                    lr_current = max(lr_current, min_lr_bound)
                    adaptive_lr_state.lr = lr_current
                    adaptive_lr_state.loss_ema = loss_ema_next
                    _set_optimizer_lr(opt_atlas, lr_current)

        # Step LR scheduler at end of each epoch
        if scheduler is not None:
            scheduler.step()

        # Logging and checkpointing (matching embed_fragile.py style)
        should_log = epoch % config.log_every == 0 or epoch == config.epochs
        should_save = config.save_every > 0 and (
            epoch % config.save_every == 0 or epoch == config.epochs
        )
        if should_log or should_save:
            # Compute metrics on full dataset
            was_training = model_atlas.training
            model_atlas.eval()
            with torch.no_grad():
                K_chart_full, _, _, _, enc_w_full, _, _, _, _, _c_bar_full = model_atlas.encoder(
                    X_test,
                    hard_routing=config.hard_routing,
                    hard_routing_tau=config.hard_routing_tau,
                )
                usage = enc_w_full.mean(dim=0).cpu().numpy()
                chart_assignments = K_chart_full.cpu().numpy()
                ami = compute_ami(labels_test, chart_assignments)
                perplexity = model_atlas.compute_perplexity(K_chart_full)
            if was_training:
                model_atlas.train()

            # Get epoch-averaged losses
            avg_loss = atlas_losses[-1]
            avg_recon = loss_components["recon"][-1]
            avg_vq = loss_components["vq"][-1]
            avg_entropy = loss_components["entropy"][-1]
            avg_consistency = loss_components["consistency"][-1]
            avg_var = loss_components["variance"][-1]
            avg_div = loss_components["diversity"][-1]
            avg_sep = loss_components["separation"][-1]
            avg_codebook_center = loss_components["codebook_center"][-1]
            avg_chart_center_sep = loss_components["chart_center_sep"][-1]
            avg_residual_scale = loss_components["residual_scale"][-1]
            avg_soft_equiv_l1 = loss_components["soft_equiv_l1"][-1]
            avg_soft_equiv_log_ratio = loss_components["soft_equiv_log_ratio"][-1]
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
            avg_cls_loss = loss_components["cls_loss"][-1]
            avg_cls_acc = loss_components["cls_acc"][-1]
            avg_std_cls_loss = loss_components["std_cls_loss"][-1]
            avg_std_cls_acc = loss_components["std_cls_acc"][-1]
            avg_ae_cls_loss = loss_components["ae_cls_loss"][-1]
            avg_ae_cls_acc = loss_components["ae_cls_acc"][-1]
            avg_ixk = info_metrics["I_XK"][-1]
            avg_hk = info_metrics["H_K"][-1]
            avg_hk_given_x = info_metrics["H_K_given_X"][-1]
            avg_code_entropy = info_metrics["code_entropy"][-1]
            avg_pc_code_entropy = info_metrics["per_chart_code_entropy"][-1]
            avg_grad_norm = info_metrics["grad_norm"][-1]
            avg_param_norm = info_metrics["param_norm"][-1]
            avg_update_ratio = info_metrics["update_ratio"][-1]
            avg_lr = info_metrics["lr"][-1]

            # Get current jump weight for logging
            log_jump_weight = get_jump_weight_schedule(
                epoch, config.jump_warmup, config.jump_ramp_end, config.jump_weight
            )

            # Print in embed_fragile.py style
            if config.adaptive_lr:
                current_lr = avg_lr
            else:
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
                f"sep={avg_sep:.3f} "
                f"center={avg_codebook_center:.3f} "
                f"chart_sep={avg_chart_center_sep:.3f} "
                f"res_scale={avg_residual_scale:.3f} "
                f"soft_eq_l1={avg_soft_equiv_l1:.3f} "
                f"soft_eq_ratio={avg_soft_equiv_log_ratio:.3f}"
            )
            print(f"  Tier2: window={avg_window:.3f} disent={avg_disent:.3f}")
            print(
                f"  Tier3: orth={avg_orth:.3f} "
                f"code_ent={avg_code_ent:.3f} "
                f"pc_code_ent={avg_pc_code_ent:.3f}"
            )
            print(f"  Tier4: kl={avg_kl:.3f} orbit={avg_orbit:.3f} vicreg={avg_vicreg:.3f}")
            print(f"  Tier5: jump={avg_jump:.3f} (λ={log_jump_weight:.3f})")
            enc_w_test = None
            z_geo_test = None
            test_sup_acc = None
            test_sup_route = None
            test_cls_acc = None
            if supervised_loss is not None or classifier_head is not None:
                was_training = model_atlas.training
                model_atlas.eval()
                with torch.no_grad():
                    _, _, _, _, enc_w_test, z_geo_test, _, _, _, _c_bar_test = model_atlas.encoder(
                        X_test,
                        hard_routing=config.hard_routing,
                        hard_routing_tau=config.hard_routing_tau,
                    )
                if was_training:
                    model_atlas.train()

            if supervised_loss is not None and enc_w_test is not None:
                with torch.no_grad():
                    sup_test = supervised_loss(enc_w_test, labels_test_t, z_geo_test)
                    p_y_x_test = torch.matmul(enc_w_test, supervised_loss.p_y_given_k)
                    test_sup_acc = (
                        (p_y_x_test.argmax(dim=1) == labels_test_t).float().mean().item()
                    )
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

            if classifier_head is not None and enc_w_test is not None:
                with torch.no_grad():
                    cls_logits_test = classifier_head(enc_w_test, z_geo_test)
                    test_cls_acc = (
                        (cls_logits_test.argmax(dim=1) == labels_test_t).float().mean().item()
                    )

                print(
                    f"  Readout: train_loss={avg_cls_loss:.3f} "
                    f"train_acc={avg_cls_acc:.3f} "
                    f"test_acc={test_cls_acc:.3f}"
                )
            std_test_acc = None
            ae_test_acc = None
            if std_classifier_head is not None and model_std is not None:
                with torch.no_grad():
                    z_std_test = model_std.encoder(X_test)
                    std_logits_test = std_classifier_head(z_std_test)
                    std_test_acc = (
                        (std_logits_test.argmax(dim=1) == labels_test_t).float().mean().item()
                    )
                print(
                    f"  Std Readout: train_loss={avg_std_cls_loss:.3f} "
                    f"train_acc={avg_std_cls_acc:.3f} "
                    f"test_acc={std_test_acc:.3f}"
                )
            if ae_classifier_head is not None and model_ae is not None:
                with torch.no_grad():
                    _, z_ae_test = model_ae(X_test)
                    ae_logits_test = ae_classifier_head(z_ae_test)
                    ae_test_acc = (
                        (ae_logits_test.argmax(dim=1) == labels_test_t).float().mean().item()
                    )
                print(
                    f"  AE Readout: train_loss={avg_ae_cls_loss:.3f} "
                    f"train_acc={avg_ae_cls_acc:.3f} "
                    f"test_acc={ae_test_acc:.3f}"
                )
            print(f"  Info: I(X;K)={avg_ixk:.3f} H(K)={avg_hk:.3f} H(K|X)={avg_hk_given_x:.3f}")
            print(f"  Code: H(code)={avg_code_entropy:.3f} H(pc_code)={avg_pc_code_entropy:.3f}")
            if config.adaptive_lr:
                print(
                    f"  LR ctl: grad_norm={avg_grad_norm:.2e} "
                    f"upd_ratio={avg_update_ratio:.2e} "
                    f"lr_avg={avg_lr:.2e}"
                )
            print(f"  Metrics: AMI={ami:.4f} perplexity={perplexity:.2f}/{config.num_charts}")
            print("-" * 60)

            if mlflow_active:
                mlflow_metrics: dict[str, float] = {
                    "loss/atlas": avg_loss,
                    "loss/std": std_losses[-1],
                    "loss/ae": ae_losses[-1],
                    "loss/recon": avg_recon,
                    "loss/vq": avg_vq,
                    "loss/entropy": avg_entropy,
                    "loss/consistency": avg_consistency,
                    "loss/variance": avg_var,
                    "loss/diversity": avg_div,
                    "loss/separation": avg_sep,
                    "loss/codebook_center": avg_codebook_center,
                    "loss/chart_center_sep": avg_chart_center_sep,
                    "loss/residual_scale": avg_residual_scale,
                    "loss/soft_equiv_l1": avg_soft_equiv_l1,
                    "loss/soft_equiv_log_ratio": avg_soft_equiv_log_ratio,
                    "loss/window": avg_window,
                    "loss/disentangle": avg_disent,
                    "loss/orthogonality": avg_orth,
                    "loss/code_entropy": avg_code_ent,
                    "loss/per_chart_code_entropy": avg_pc_code_ent,
                    "loss/kl_prior": avg_kl,
                    "loss/orbit": avg_orbit,
                    "loss/vicreg_inv": avg_vicreg,
                    "loss/jump": avg_jump,
                    "loss/sup_total": avg_sup_total,
                    "loss/sup_route": avg_sup_route,
                    "loss/sup_purity": avg_sup_purity,
                    "loss/sup_balance": avg_sup_balance,
                    "loss/sup_metric": avg_sup_metric,
                    "loss/cls": avg_cls_loss,
                    "loss/std_cls": avg_std_cls_loss,
                    "loss/ae_cls": avg_ae_cls_loss,
                    "metric/ami": ami,
                    "metric/perplexity": perplexity,
                    "metric/sup_acc": avg_sup_acc,
                    "metric/cls_acc": avg_cls_acc,
                    "metric/std_cls_acc": avg_std_cls_acc,
                    "metric/ae_cls_acc": avg_ae_cls_acc,
                    "metric/I_XK": avg_ixk,
                    "metric/H_K": avg_hk,
                    "metric/H_K_given_X": avg_hk_given_x,
                    "metric/code_entropy": avg_code_entropy,
                    "metric/per_chart_code_entropy": avg_pc_code_entropy,
                    "control/grad_norm": avg_grad_norm,
                    "control/param_norm": avg_param_norm,
                    "control/update_ratio": avg_update_ratio,
                    "control/lr": avg_lr,
                    "control/lr_max": lr_max,
                    "control/lr_loss_ema": adaptive_lr_state.loss_ema,
                    "control/lr_best_loss_ema": adaptive_lr_state.best_loss_ema,
                    "control/ixk_ema": adaptive_lr_state.ixk_ema,
                    "control/unstable_steps": adaptive_lr_state.unstable_steps,
                    "control/stable_steps": adaptive_lr_state.stable_steps,
                    "control/lr_plateau_steps": adaptive_lr_state.plateau_steps,
                    "control/jump_weight": log_jump_weight,
                    "control/jump_weight_base": config.jump_weight,
                    "control/sup_weight": config.sup_weight,
                }
                if test_sup_acc is not None:
                    mlflow_metrics["metric/test_sup_acc"] = test_sup_acc
                if test_sup_route is not None:
                    mlflow_metrics["metric/test_sup_route"] = test_sup_route
                if test_cls_acc is not None:
                    mlflow_metrics["metric/test_cls_acc"] = test_cls_acc
                if std_test_acc is not None:
                    mlflow_metrics["metric/test_std_cls_acc"] = std_test_acc
                if ae_test_acc is not None:
                    mlflow_metrics["metric/test_ae_cls_acc"] = ae_test_acc
                for idx, value in enumerate(usage):
                    mlflow_metrics[f"usage/chart_{idx}"] = float(value)
                for name, state in adaptive_weight_state.items():
                    mlflow_metrics[f"control/weight/{name}"] = state.value
                    mlflow_metrics[f"control/weight_integral/{name}"] = state.integral
                    mlflow_metrics[f"control/weight_prev_error/{name}"] = state.prev_error
                for name, value in adaptive_target_state.items():
                    mlflow_metrics[f"control/target/{name}"] = value
                _log_mlflow_metrics(mlflow_metrics, step=epoch, enabled=mlflow_active)

            # Save checkpoint
            if should_save:
                save_path = f"{config.output_dir}/topo_epoch_{epoch:05d}.pt"
                checkpoint_metrics = {
                    "std_losses": std_losses,
                    "atlas_losses": atlas_losses,
                    "ae_losses": ae_losses,
                    "loss_components": loss_components,
                    "info_metrics": info_metrics,
                    "adaptive_weight_state": _serialize_weight_state(adaptive_weight_state),
                    "adaptive_target_state": _serialize_target_state(adaptive_target_state),
                    "adaptive_lr_state": {
                        "lr": adaptive_lr_state.lr,
                        "loss_ema": adaptive_lr_state.loss_ema,
                        "best_loss_ema": adaptive_lr_state.best_loss_ema,
                        "ixk_ema": adaptive_lr_state.ixk_ema,
                        "unstable_steps": adaptive_lr_state.unstable_steps,
                        "stable_steps": adaptive_lr_state.stable_steps,
                        "plateau_steps": adaptive_lr_state.plateau_steps,
                    },
                    "ami_atlas": ami,
                    "atlas_perplexity": perplexity,
                    "chart_assignments": chart_assignments,
                    "std_hidden_dim": std_hidden_dim,
                    "ae_hidden_dim": ae_hidden_dim,
                }
                save_checkpoint(
                    save_path,
                    config,
                    model_atlas,
                    jump_op,
                    checkpoint_metrics,
                    data_snapshot,
                    epoch,
                    model_std=model_std,
                    model_ae=model_ae,
                    supervised_loss=supervised_loss,
                    classifier_head=classifier_head,
                    classifier_std=std_classifier_head,
                    classifier_ae=ae_classifier_head,
                    precision_module=precision_module,
                    optimizer_atlas=opt_atlas,
                    optimizer_std=opt_std,
                    optimizer_ae=opt_ae,
                    optimizer_classifier=opt_classifier,
                    optimizer_classifier_std=opt_classifier_std,
                    optimizer_classifier_ae=opt_classifier_ae,
                    scheduler=scheduler,
                )
                print(f"Checkpoint saved: {save_path}")
                if train_std or train_ae:
                    benchmarks_metrics = {
                        "std_losses": std_losses,
                        "ae_losses": ae_losses,
                        "std_hidden_dim": std_hidden_dim,
                        "ae_hidden_dim": ae_hidden_dim,
                    }
                    save_benchmarks(
                        benchmarks_path,
                        config,
                        model_std,
                        model_ae,
                        std_hidden_dim=std_hidden_dim,
                        ae_hidden_dim=ae_hidden_dim,
                        optimizer_std=opt_std,
                        optimizer_ae=opt_ae,
                        metrics=benchmarks_metrics,
                        epoch=epoch,
                    )

    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)

    with torch.no_grad():
        # VanillaAE metrics (reconstruction baseline)
        if model_ae is not None:
            model_ae.eval()
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
            model_std.eval()
            recon_std_final, _, indices_s = model_std(X_test)
            std_perplexity = model_std.compute_perplexity(indices_s)
            mse_std = F.mse_loss(recon_std_final, X_test).item()
            # Cluster VQ codes to get comparable AMI
            vq_clusters = indices_s.cpu().numpy() % config.num_charts  # Simple modulo clustering
            ami_std = compute_ami(labels_test, vq_clusters)

        # Atlas metrics (use dreaming mode to test autonomous routing)
        model_atlas.eval()
        recon_atlas_final, _, enc_w, dec_w, K_chart, _z_geo, _z_n, _c_bar = model_atlas(
            X_test,
            use_hard_routing=config.hard_routing,
            hard_routing_tau=config.hard_routing_tau,
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
            _, _, _, _, enc_w_cls, z_geo_cls, _, _, _, _c_bar_cls = model_atlas.encoder(
                X_test,
                hard_routing=config.hard_routing,
                hard_routing_tau=config.hard_routing_tau,
            )
            cls_logits = classifier_head(enc_w_cls, z_geo_cls)
            cls_acc = (cls_logits.argmax(dim=1) == labels_test_t).float().mean().item()
        std_cls_acc = 0.0
        if std_classifier_head is not None and model_std is not None:
            z_std_final = model_std.encoder(X_test)
            std_logits = std_classifier_head(z_std_final)
            std_cls_acc = (std_logits.argmax(dim=1) == labels_test_t).float().mean().item()
        ae_cls_acc = 0.0
        if ae_classifier_head is not None and model_ae is not None:
            _, z_ae_final = model_ae(X_test)
            ae_logits = ae_classifier_head(z_ae_final)
            ae_cls_acc = (ae_logits.argmax(dim=1) == labels_test_t).float().mean().item()

    if mlflow_active:
        final_mlflow_metrics = {
            "final/mse_ae": mse_ae,
            "final/mse_std": mse_std,
            "final/mse_atlas": mse_atlas,
            "final/ami_ae": ami_ae,
            "final/ami_std": ami_std,
            "final/ami_atlas": ami_atlas,
            "final/std_perplexity": std_perplexity,
            "final/atlas_perplexity": atlas_perplexity,
            "final/consistency": final_consistency,
            "final/sup_acc": sup_acc,
            "final/cls_acc": cls_acc,
            "final/std_cls_acc": std_cls_acc,
            "final/ae_cls_acc": ae_cls_acc,
        }
        _log_mlflow_metrics(final_mlflow_metrics, step=config.epochs, enabled=mlflow_active)

    # Results table
    print("\n" + "-" * 70)
    print(f"{'Model':<20} {'MSE':>10} {'AMI':>10} {'Perplexity':>15}")
    print("-" * 70)
    if model_ae is not None:
        print(f"{'Vanilla AE':<20} {mse_ae:>10.5f} {ami_ae:>10.4f} {'N/A (K-Means)':<15}")
    if model_std is not None:
        print(
            f"{'Standard VQ':<20} {mse_std:>10.5f} {ami_std:>10.4f} {std_perplexity:>6.1f}/{config.num_codes_standard:<8}"
        )
    print(
        f"{'TopoEncoder':<20} {mse_atlas:>10.5f} {ami_atlas:>10.4f} {atlas_perplexity:>6.1f}/{config.num_charts:<8}"
    )
    print("-" * 70)

    # Interpretation (only if baselines enabled)
    if model_ae is not None and model_std is not None:
        print("\nInterpretation:")
        if mse_ae < mse_atlas < mse_std:
            print("  AE has best reconstruction (expected - no bottleneck)")
            print("  TopoEncoder beats VQ on reconstruction (atlas routing helps)")
        if ami_atlas > ami_ae and ami_atlas > ami_std:
            print("  TopoEncoder has best topology discovery (charts match labels)")
        if ami_ae < ami_atlas:
            print("  AE fails at topology despite good reconstruction (entangled latent)")

    print(f"\nRouting Consistency (KL): {final_consistency:.4f}")
    if supervised_loss is not None:
        print(f"Supervised Accuracy: {sup_acc:.4f}")
    if classifier_head is not None:
        print(f"Readout Accuracy: {cls_acc:.4f}")
    if std_classifier_head is not None and model_std is not None:
        print(f"StandardVQ Readout Accuracy: {std_cls_acc:.4f}")
    if ae_classifier_head is not None and model_ae is not None:
        print(f"VanillaAE Readout Accuracy: {ae_cls_acc:.4f}")

    final_checkpoint = f"{config.output_dir}/topo_final.pt"
    final_metrics = {
        "std_losses": std_losses,
        "atlas_losses": atlas_losses,
        "ae_losses": ae_losses,
        "loss_components": loss_components,
        "info_metrics": info_metrics,
        "adaptive_weight_state": _serialize_weight_state(adaptive_weight_state),
        "adaptive_target_state": _serialize_target_state(adaptive_target_state),
        "adaptive_lr_state": {
            "lr": adaptive_lr_state.lr,
            "loss_ema": adaptive_lr_state.loss_ema,
            "best_loss_ema": adaptive_lr_state.best_loss_ema,
            "ixk_ema": adaptive_lr_state.ixk_ema,
            "unstable_steps": adaptive_lr_state.unstable_steps,
            "stable_steps": adaptive_lr_state.stable_steps,
            "plateau_steps": adaptive_lr_state.plateau_steps,
        },
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
        "cls_acc": cls_acc,
        "std_cls_acc": std_cls_acc,
        "ae_cls_acc": ae_cls_acc,
        "chart_assignments": chart_assignments,
        "std_hidden_dim": std_hidden_dim,
        "ae_hidden_dim": ae_hidden_dim,
    }
    save_checkpoint(
        final_checkpoint,
        config,
        model_atlas,
        jump_op,
        final_metrics,
        data_snapshot,
        config.epochs,
        model_std=model_std,
        model_ae=model_ae,
        supervised_loss=supervised_loss,
        classifier_head=classifier_head,
        classifier_std=std_classifier_head,
        classifier_ae=ae_classifier_head,
        precision_module=precision_module,
        optimizer_atlas=opt_atlas,
        optimizer_std=opt_std,
        optimizer_ae=opt_ae,
        optimizer_classifier=opt_classifier,
        optimizer_classifier_std=opt_classifier_std,
        optimizer_classifier_ae=opt_classifier_ae,
        scheduler=scheduler,
    )
    print(f"\nFinal checkpoint saved to: {final_checkpoint}")
    if train_std or train_ae:
        benchmarks_metrics = {
            "std_losses": std_losses,
            "ae_losses": ae_losses,
            "std_hidden_dim": std_hidden_dim,
            "ae_hidden_dim": ae_hidden_dim,
            "mse_std": mse_std,
            "mse_ae": mse_ae,
            "ami_std": ami_std,
            "ami_ae": ami_ae,
            "std_perplexity": std_perplexity,
        }
        save_benchmarks(
            benchmarks_path,
            config,
            model_std,
            model_ae,
            std_hidden_dim=std_hidden_dim,
            ae_hidden_dim=ae_hidden_dim,
            optimizer_std=opt_std,
            optimizer_ae=opt_ae,
            metrics=benchmarks_metrics,
            epoch=config.epochs,
        )

    _end_mlflow_run(mlflow_active)

    return {
        "ami_ae": ami_ae,
        "ami_std": ami_std,
        "ami_atlas": ami_atlas,
        "mse_ae": mse_ae,
        "mse_std": mse_std,
        "mse_atlas": mse_atlas,
        "sup_acc": sup_acc,
        "cls_acc": cls_acc,
        "std_cls_acc": std_cls_acc,
        "ae_cls_acc": ae_cls_acc,
        "atlas_perplexity": atlas_perplexity,
        "std_perplexity": std_perplexity,
        "checkpoint_path": final_checkpoint,
    }


# ==========================================
# 10. MAIN
# ==========================================
def main():
    """Main entry point for the benchmark."""
    parser = argparse.ArgumentParser(
        description="TopoEncoder Benchmark: Attentive Atlas vs Standard VQ-VAE"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Dataset to use (mnist or cifar10)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training (0 = full batch)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=40000,
        help="Number of samples to use",
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
        default=10,
        help="Number of atlas charts (default: 10 for MNIST/CIFAR-10)",
    )
    parser.add_argument(
        "--codes_per_chart",
        type=int,
        default=21,
        help="VQ codes per chart",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=3,
        help="Latent dimension for TopoEncoder (default: 3)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension for TopoEncoder",
    )
    parser.add_argument(
        "--covariant_attn",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use covariant chart routing attention (default: True)",
    )
    parser.add_argument(
        "--covariant_attn_tensorization",
        type=str,
        default="full",
        choices=["sum", "full"],
        help="Christoffel tensorization: sum (low-rank) or full (default: full)",
    )
    parser.add_argument(
        "--covariant_attn_rank",
        type=int,
        default=8,
        help="Rank for tensor-sum Christoffel term (default: 8)",
    )
    parser.add_argument(
        "--covariant_attn_tau_min",
        type=float,
        default=1e-2,
        help="Minimum covariant attention temperature (default: 1e-2)",
    )
    parser.add_argument(
        "--covariant_attn_denom_min",
        type=float,
        default=1e-3,
        help="Minimum 1-||z||^2 denom for temperature (default: 1e-3)",
    )
    parser.add_argument(
        "--covariant_attn_use_transport",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use Wilson-line transport in covariant attention (default: True)",
    )
    parser.add_argument(
        "--covariant_attn_transport_eps",
        type=float,
        default=1e-3,
        help="Diagonal stabilizer for transport (default: 1e-3)",
    )
    parser.add_argument(
        "--hard_routing",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use hard chart routing in the decoder (default: False)",
    )
    parser.add_argument(
        "--hard_routing_tau",
        type=float,
        default=1.0,
        help="Gumbel-softmax temperature for hard routing (default: 1.0)",
    )
    parser.add_argument(
        "--vision_preproc",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use covariant vision preprocessor (default: False)",
    )
    parser.add_argument(
        "--vision_in_channels",
        type=int,
        default=0,
        help="Vision preproc input channels (0 = infer from dataset)",
    )
    parser.add_argument(
        "--vision_height",
        type=int,
        default=0,
        help="Vision preproc input height (0 = infer from dataset)",
    )
    parser.add_argument(
        "--vision_width",
        type=int,
        default=0,
        help="Vision preproc input width (0 = infer from dataset)",
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
        "--vision_backbone_type",
        type=str,
        default="covariant_retina",
        help=(
            "Vision backbone type: covariant_retina (SO(2)-equivariant), "
            "covariant_cifar (gauge-covariant), or standard_cifar (non-equivariant)"
        ),
    )
    parser.add_argument(
        "--vision_cifar_base_channels",
        type=int,
        default=32,
        help="Base channels for CovariantCIFARBackbone (default: 32)",
    )
    parser.add_argument(
        "--vision_cifar_bundle_size",
        type=int,
        default=4,
        help="Bundle size for NormGatedConv2d in CovariantCIFARBackbone (default: 4)",
    )
    parser.add_argument(
        "--vision_soft_equiv",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable soft equivariant mixing on vision features (default: False)",
    )
    parser.add_argument(
        "--vision_soft_equiv_bundle_size",
        type=int,
        default=0,
        help="Bundle size for vision soft equivariant mixing (0 = hidden_dim)",
    )
    parser.add_argument(
        "--vision_soft_equiv_hidden_dim",
        type=int,
        default=64,
        help="Hidden dim for vision soft equivariant mixing (default: 64)",
    )
    parser.add_argument(
        "--vision_soft_equiv_use_spectral_norm",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use spectral norm in vision soft equivariant mixing (default: True)",
    )
    parser.add_argument(
        "--vision_soft_equiv_zero_self_mixing",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Zero self-mixing in vision soft equivariant mixing (default: False)",
    )
    parser.add_argument(
        "--vision_soft_equiv_alpha",
        type=float,
        default=0.1,
        help="Blend factor for vision soft equivariant mixing (default: 0.1)",
    )
    parser.add_argument(
        "--vision_soft_equiv_per_block",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Apply vision soft equivariant mixing after each conv block (default: False)",
    )
    parser.add_argument(
        "--vision_standard_head",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Apply standard CNN head after covariant stages (default: False)",
    )
    parser.add_argument(
        "--vision_standard_head_blocks",
        type=int,
        default=2,
        help="Number of standard CNN blocks in the head (default: 2)",
    )
    parser.add_argument(
        "--vision_spectral_head",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Apply a spectral linear head on vision features (default: False)",
    )
    parser.add_argument(
        "--soft_equiv",
        action="store_true",
        help=(
            "Enable soft equivariant metric with defaults "
            "(sets --soft_equiv_metric true and --soft_equiv_l1_weight 0.01)"
        ),
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
        help=("Use straight-through soft assignment for soft equivariant metric (default: True)"),
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
        "--window_eps_ground",
        type=float,
        default=0.1,
        help="Minimum I(X;K) grounding threshold (default: 0.1)",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="Log training metrics every N epochs",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (0 to disable)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint to resume training",
    )
    parser.add_argument(
        "--mlflow",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable MLflow logging (default: False)",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="",
        help="MLflow tracking URI (default: empty)",
    )
    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default="",
        help="MLflow experiment name (default: empty)",
    )
    parser.add_argument(
        "--mlflow_run_name",
        type=str,
        default="",
        help="MLflow run name (default: empty)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    # Tier 1 losses (residual hierarchy)
    parser.add_argument(
        "--entropy_weight",
        type=float,
        default=0.1,
        help="Routing entropy weight (default: 0.1)",
    )
    parser.add_argument(
        "--diversity_weight",
        type=float,
        default=0.1,
        help="Chart diversity weight (default: 0.1)",
    )
    parser.add_argument(
        "--separation_weight",
        type=float,
        default=0.1,
        help="Chart separation weight (default: 0.1)",
    )
    parser.add_argument(
        "--codebook_center_weight",
        type=float,
        default=0.05,
        help="Codebook centering weight (default: 0.05)",
    )
    parser.add_argument(
        "--chart_center_sep_weight",
        type=float,
        default=0.05,
        help="Chart center separation weight (default: 0.05)",
    )
    parser.add_argument(
        "--chart_center_sep_margin",
        type=float,
        default=2.0,
        help="Chart center separation margin (default: 2.0)",
    )
    parser.add_argument(
        "--residual_scale_weight",
        type=float,
        default=0.01,
        help="Residual scale penalty weight (default: 0.01)",
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
        "--adaptive_lr",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use adaptive LR controller (default: True)",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=1e-6,
        help="Minimum adaptive LR (default: 1e-6)",
    )
    parser.add_argument(
        "--lr_max",
        type=float,
        default=0.0,
        help="Maximum adaptive LR (0 = use --lr, default: 0.0)",
    )
    parser.add_argument(
        "--lr_increase_factor",
        type=float,
        default=1.02,
        help="Adaptive LR increase factor (default: 1.02)",
    )
    parser.add_argument(
        "--lr_decrease_factor",
        type=float,
        default=0.5,
        help="Adaptive LR decrease factor (default: 0.5)",
    )
    parser.add_argument(
        "--lr_max_update_ratio",
        type=float,
        default=1e-3,
        help="Max update ratio ||dtheta||/||theta|| (default: 1e-3)",
    )
    parser.add_argument(
        "--lr_ema_decay",
        type=float,
        default=0.98,
        help="EMA decay for loss stability (default: 0.98)",
    )
    parser.add_argument(
        "--lr_loss_increase_tol",
        type=float,
        default=0.05,
        help="Loss spike tolerance before LR reduction (default: 0.05)",
    )
    parser.add_argument(
        "--lr_grounding_warmup_epochs",
        type=int,
        default=0,
        help=("Epochs to skip coupling-window grounding checks for adaptive LR (default: 0)"),
    )
    parser.add_argument(
        "--lr_unstable_patience",
        type=int,
        default=3,
        help="Batches before LR decrease after instability (default: 3)",
    )
    parser.add_argument(
        "--lr_stable_patience",
        type=int,
        default=5,
        help="Batches before LR increase after stability (default: 5)",
    )
    parser.add_argument(
        "--lr_plateau_patience",
        type=int,
        default=10,
        help="Batches of plateau before LR decrease (default: 10)",
    )
    parser.add_argument(
        "--lr_plateau_tol",
        type=float,
        default=1e-3,
        help="Relative loss improvement to reset plateau (default: 1e-3)",
    )
    parser.add_argument(
        "--lr_grounding_ema_decay",
        type=float,
        default=0.98,
        help="EMA decay for I(X;K) grounding check (default: 0.98)",
    )
    parser.add_argument(
        "--lr_recovery_factor",
        type=float,
        default=1.1,
        help="LR increase factor when recovering from low LR (default: 1.1)",
    )
    parser.add_argument(
        "--lr_recovery_threshold",
        type=float,
        default=0.1,
        help="Fraction of lr_max to trigger recovery factor (default: 0.1)",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (0 to disable, default: 1.0)",
    )

    parser.add_argument(
        "--adaptive_weights",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use adaptive lambda controllers (default: True)",
    )
    parser.add_argument(
        "--adaptive_warmup_epochs",
        type=int,
        default=10,
        help="Epochs before adaptive lambda updates (default: 10)",
    )
    parser.add_argument(
        "--adaptive_dual_eta",
        type=float,
        default=0.05,
        help="Dual ascent step size for constraints (default: 0.05)",
    )
    parser.add_argument(
        "--adaptive_pi_kp",
        type=float,
        default=0.05,
        help="PI controller Kp (default: 0.05)",
    )
    parser.add_argument(
        "--adaptive_pi_ki",
        type=float,
        default=0.01,
        help="PI controller Ki (default: 0.01)",
    )
    parser.add_argument(
        "--adaptive_pi_kd",
        type=float,
        default=0.0,
        help="PI controller Kd (default: 0.0)",
    )
    parser.add_argument(
        "--adaptive_lambda_min",
        type=float,
        default=0.0,
        help="Minimum adaptive lambda (default: 0.0)",
    )
    parser.add_argument(
        "--adaptive_lambda_max",
        type=float,
        default=10.0,
        help="Maximum adaptive lambda (default: 10.0)",
    )
    parser.add_argument(
        "--adaptive_violation_clip",
        type=float,
        default=2.0,
        help="Clamp adaptive constraint violations (<=0 disables, default: 2.0)",
    )
    parser.add_argument(
        "--adaptive_target_ratio",
        type=float,
        default=1.0,
        help="Target ratio relative to warmup baseline (default: 1.0)",
    )
    parser.add_argument(
        "--adaptive_target_ema_decay",
        type=float,
        default=0.98,
        help="EMA decay for warmup target estimates (default: 0.98)",
    )
    parser.add_argument(
        "--adaptive_target_min",
        type=float,
        default=1e-3,
        help="Minimum target value for signed errors (default: 1e-3)",
    )
    parser.add_argument(
        "--entropy_target_ratio",
        type=float,
        default=0.5,
        help="Target H(K|X) ratio of log K (default: 0.5)",
    )
    parser.add_argument(
        "--hk_target_ratio",
        type=float,
        default=0.9,
        help="Target H(K) ratio of log K (default: 0.9)",
    )
    parser.add_argument(
        "--code_entropy_target_ratio",
        type=float,
        default=0.9,
        help="Target code entropy ratio of log codes (default: 0.9)",
    )
    parser.add_argument(
        "--consistency_target",
        type=float,
        default=0.05,
        help="Target consistency KL (default: 0.05)",
    )
    parser.add_argument(
        "--use_learned_precisions",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use learned precisions for recon/vq/sup (default: False)",
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
        default=5,
        help="Epochs before jump loss starts (default: 50)",
    )
    parser.add_argument(
        "--jump_ramp_end",
        type=int,
        default=50,
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
        "--disable_ae",
        action="store_true",
        help="Disable VanillaAE baseline (faster training)",
    )
    parser.add_argument(
        "--disable_vq",
        action="store_true",
        help="Disable StandardVQ baseline (faster training)",
    )
    parser.add_argument(
        "--baseline_vision_preproc",
        action="store_true",
        help="Enable standard vision backbone in baselines",
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

    args = parser.parse_args()

    if args.soft_equiv:
        args.soft_equiv_metric = True
        if args.soft_equiv_l1_weight == 0.0:
            args.soft_equiv_l1_weight = 0.01

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        config = TopoEncoderConfig(**checkpoint["config"])
        config.resume_checkpoint = args.resume
        config.epochs = max(config.epochs, args.epochs)
        config.log_every = args.log_every
        config.save_every = args.save_every
        if args.output_dir is not None:
            config.output_dir = args.output_dir
        config.device = args.device
        config.vision_preproc = args.vision_preproc
        config.vision_in_channels = args.vision_in_channels
        config.vision_height = args.vision_height
        config.vision_width = args.vision_width
        config.vision_num_rotations = args.vision_num_rotations
        config.vision_kernel_size = args.vision_kernel_size
        config.vision_use_reflections = args.vision_use_reflections
        config.vision_norm_nonlinearity = args.vision_norm_nonlinearity
        config.vision_norm_bias = args.vision_norm_bias
        config.vision_backbone_type = args.vision_backbone_type
        config.vision_cifar_base_channels = args.vision_cifar_base_channels
        config.vision_cifar_bundle_size = args.vision_cifar_bundle_size
        config.vision_soft_equiv = args.vision_soft_equiv
        config.vision_soft_equiv_bundle_size = args.vision_soft_equiv_bundle_size
        config.vision_soft_equiv_hidden_dim = args.vision_soft_equiv_hidden_dim
        config.vision_soft_equiv_use_spectral_norm = args.vision_soft_equiv_use_spectral_norm
        config.vision_soft_equiv_zero_self_mixing = args.vision_soft_equiv_zero_self_mixing
        config.vision_soft_equiv_alpha = args.vision_soft_equiv_alpha
        config.vision_soft_equiv_per_block = args.vision_soft_equiv_per_block
        config.vision_standard_head = args.vision_standard_head
        config.vision_standard_head_blocks = args.vision_standard_head_blocks
        config.vision_spectral_head = args.vision_spectral_head
        config.use_scheduler = args.use_scheduler
        config.min_lr = args.min_lr
        config.adaptive_lr = args.adaptive_lr
        config.lr_min = args.lr_min
        config.lr_max = args.lr_max
        config.lr_increase_factor = args.lr_increase_factor
        config.lr_decrease_factor = args.lr_decrease_factor
        config.lr_max_update_ratio = args.lr_max_update_ratio
        config.lr_ema_decay = args.lr_ema_decay
        config.lr_loss_increase_tol = args.lr_loss_increase_tol
        config.lr_grounding_warmup_epochs = args.lr_grounding_warmup_epochs
        config.lr_unstable_patience = args.lr_unstable_patience
        config.lr_stable_patience = args.lr_stable_patience
        config.lr_plateau_patience = args.lr_plateau_patience
        config.lr_plateau_tol = args.lr_plateau_tol
        config.lr_grounding_ema_decay = args.lr_grounding_ema_decay
        config.lr_recovery_factor = args.lr_recovery_factor
        config.lr_recovery_threshold = args.lr_recovery_threshold
        config.adaptive_weights = args.adaptive_weights
        config.adaptive_warmup_epochs = args.adaptive_warmup_epochs
        config.adaptive_dual_eta = args.adaptive_dual_eta
        config.adaptive_pi_kp = args.adaptive_pi_kp
        config.adaptive_pi_ki = args.adaptive_pi_ki
        config.adaptive_pi_kd = args.adaptive_pi_kd
        config.adaptive_lambda_min = args.adaptive_lambda_min
        config.adaptive_lambda_max = args.adaptive_lambda_max
        config.adaptive_violation_clip = args.adaptive_violation_clip
        config.adaptive_target_ratio = args.adaptive_target_ratio
        config.adaptive_target_ema_decay = args.adaptive_target_ema_decay
        config.adaptive_target_min = args.adaptive_target_min
        config.entropy_target_ratio = args.entropy_target_ratio
        config.hk_target_ratio = args.hk_target_ratio
        config.code_entropy_target_ratio = args.code_entropy_target_ratio
        config.consistency_target = args.consistency_target
        config.use_learned_precisions = args.use_learned_precisions
        config.window_eps_ground = args.window_eps_ground
        config.entropy_weight = args.entropy_weight
        config.diversity_weight = args.diversity_weight
        config.separation_weight = args.separation_weight
        config.codebook_center_weight = args.codebook_center_weight
        config.chart_center_sep_weight = args.chart_center_sep_weight
        config.chart_center_sep_margin = args.chart_center_sep_margin
        config.residual_scale_weight = args.residual_scale_weight
        config.covariant_attn = args.covariant_attn
        config.covariant_attn_tensorization = args.covariant_attn_tensorization
        config.covariant_attn_rank = args.covariant_attn_rank
        config.covariant_attn_tau_min = args.covariant_attn_tau_min
        config.covariant_attn_denom_min = args.covariant_attn_denom_min
        config.covariant_attn_use_transport = args.covariant_attn_use_transport
        config.covariant_attn_transport_eps = args.covariant_attn_transport_eps
        config.hard_routing = args.hard_routing
        config.hard_routing_tau = args.hard_routing_tau
        config.soft_equiv_metric = args.soft_equiv_metric
        config.soft_equiv_bundle_size = args.soft_equiv_bundle_size or None
        config.soft_equiv_hidden_dim = args.soft_equiv_hidden_dim
        config.soft_equiv_use_spectral_norm = args.soft_equiv_use_spectral_norm
        config.soft_equiv_zero_self_mixing = args.soft_equiv_zero_self_mixing
        config.soft_equiv_soft_assign = args.soft_equiv_soft_assign
        config.soft_equiv_temperature = args.soft_equiv_temperature
        config.soft_equiv_l1_weight = args.soft_equiv_l1_weight
        config.soft_equiv_log_ratio_weight = args.soft_equiv_log_ratio_weight
        config.baseline_vision_preproc = args.baseline_vision_preproc
        config.baseline_attn = args.baseline_attn
        config.baseline_attn_tokens = args.baseline_attn_tokens
        config.baseline_attn_dim = args.baseline_attn_dim
        config.baseline_attn_heads = args.baseline_attn_heads
        config.baseline_attn_dropout = args.baseline_attn_dropout
        config.mlflow = args.mlflow
        config.mlflow_tracking_uri = args.mlflow_tracking_uri
        config.mlflow_experiment = args.mlflow_experiment
        config.mlflow_run_name = args.mlflow_run_name
    else:
        output_dir = args.output_dir or "outputs/topoencoder"
        config = TopoEncoderConfig(
            dataset=args.dataset,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            n_samples=args.n_samples,
            test_split=args.test_split,
            num_charts=args.num_charts,
            codes_per_chart=args.codes_per_chart,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            covariant_attn=args.covariant_attn,
            covariant_attn_tensorization=args.covariant_attn_tensorization,
            covariant_attn_rank=args.covariant_attn_rank,
            covariant_attn_tau_min=args.covariant_attn_tau_min,
            covariant_attn_denom_min=args.covariant_attn_denom_min,
            covariant_attn_use_transport=args.covariant_attn_use_transport,
            covariant_attn_transport_eps=args.covariant_attn_transport_eps,
            hard_routing=args.hard_routing,
            hard_routing_tau=args.hard_routing_tau,
            vision_preproc=args.vision_preproc,
            vision_in_channels=args.vision_in_channels,
            vision_height=args.vision_height,
            vision_width=args.vision_width,
            vision_num_rotations=args.vision_num_rotations,
            vision_kernel_size=args.vision_kernel_size,
            vision_use_reflections=args.vision_use_reflections,
            vision_norm_nonlinearity=args.vision_norm_nonlinearity,
            vision_norm_bias=args.vision_norm_bias,
            vision_backbone_type=args.vision_backbone_type,
            vision_cifar_base_channels=args.vision_cifar_base_channels,
            vision_cifar_bundle_size=args.vision_cifar_bundle_size,
            vision_soft_equiv=args.vision_soft_equiv,
            vision_soft_equiv_bundle_size=args.vision_soft_equiv_bundle_size,
            vision_soft_equiv_hidden_dim=args.vision_soft_equiv_hidden_dim,
            vision_soft_equiv_use_spectral_norm=args.vision_soft_equiv_use_spectral_norm,
            vision_soft_equiv_zero_self_mixing=args.vision_soft_equiv_zero_self_mixing,
            vision_soft_equiv_alpha=args.vision_soft_equiv_alpha,
            vision_soft_equiv_per_block=args.vision_soft_equiv_per_block,
            vision_standard_head=args.vision_standard_head,
            vision_standard_head_blocks=args.vision_standard_head_blocks,
            vision_spectral_head=args.vision_spectral_head,
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
            baseline_attn=args.baseline_attn,
            baseline_attn_tokens=args.baseline_attn_tokens,
            baseline_attn_dim=args.baseline_attn_dim,
            baseline_attn_heads=args.baseline_attn_heads,
            baseline_attn_dropout=args.baseline_attn_dropout,
            log_every=args.log_every,
            save_every=args.save_every,
            output_dir=output_dir,
            device=args.device,
            mlflow=args.mlflow,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment=args.mlflow_experiment,
            mlflow_run_name=args.mlflow_run_name,
            # Tier 1 losses
            entropy_weight=args.entropy_weight,
            diversity_weight=args.diversity_weight,
            separation_weight=args.separation_weight,
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
            augment_rotation_max=args.augment_rotation_max,
            # Training dynamics
            use_scheduler=args.use_scheduler,
            min_lr=args.min_lr,
            grad_clip=args.grad_clip,
            adaptive_lr=args.adaptive_lr,
            lr_min=args.lr_min,
            lr_max=args.lr_max,
            lr_increase_factor=args.lr_increase_factor,
            lr_decrease_factor=args.lr_decrease_factor,
            lr_max_update_ratio=args.lr_max_update_ratio,
            lr_ema_decay=args.lr_ema_decay,
            lr_loss_increase_tol=args.lr_loss_increase_tol,
            lr_grounding_warmup_epochs=args.lr_grounding_warmup_epochs,
            lr_unstable_patience=args.lr_unstable_patience,
            lr_stable_patience=args.lr_stable_patience,
            lr_plateau_patience=args.lr_plateau_patience,
            lr_plateau_tol=args.lr_plateau_tol,
            lr_grounding_ema_decay=args.lr_grounding_ema_decay,
            lr_recovery_factor=args.lr_recovery_factor,
            lr_recovery_threshold=args.lr_recovery_threshold,
            adaptive_weights=args.adaptive_weights,
            adaptive_warmup_epochs=args.adaptive_warmup_epochs,
            adaptive_dual_eta=args.adaptive_dual_eta,
            adaptive_pi_kp=args.adaptive_pi_kp,
            adaptive_pi_ki=args.adaptive_pi_ki,
            adaptive_pi_kd=args.adaptive_pi_kd,
            adaptive_lambda_min=args.adaptive_lambda_min,
            adaptive_lambda_max=args.adaptive_lambda_max,
            adaptive_violation_clip=args.adaptive_violation_clip,
            adaptive_target_ratio=args.adaptive_target_ratio,
            adaptive_target_ema_decay=args.adaptive_target_ema_decay,
            adaptive_target_min=args.adaptive_target_min,
            entropy_target_ratio=args.entropy_target_ratio,
            hk_target_ratio=args.hk_target_ratio,
            code_entropy_target_ratio=args.code_entropy_target_ratio,
            consistency_target=args.consistency_target,
            use_learned_precisions=args.use_learned_precisions,
            window_eps_ground=args.window_eps_ground,
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
        )

    print("=" * 50)
    print("TopoEncoder Benchmark")
    print("Attentive Atlas vs Standard VQ-VAE")
    print("=" * 50)
    print("\nConfiguration:")
    print(f"  Dataset: {config.dataset}")
    if config.resume_checkpoint:
        print(f"  Resume: {config.resume_checkpoint}")
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
        print(
            f"VanillaAE beats TopoEncoder ({ami_ae:.3f} > {ami_atlas:.3f}) - K-Means works well here"
        )
    print(f"\nFinal checkpoint saved to: {results['checkpoint_path']}")
    print(f"Output directory: {config.output_dir}/")


if __name__ == "__main__":
    main()
