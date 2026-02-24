"""
TopoEncoder Benchmark: Attentive Atlas vs Standard VQ-VAE

This script benchmarks the Attentive Atlas architecture (from fragile-index.md Section 7.8)
against a standard VQ-VAE on MNIST or Fashion-MNIST (MNIST default).

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
    python src/fragile/learning/topoencoder_mnist.py --dataset mnist --epochs 1000

Notes:
    Uses vanilla Adam optimizers with optional CosineAnnealingLR scheduling.
    One optimizer for the atlas core, plus per-module optimizers for detached
    baselines and auxiliary heads.

Reference: fragile-index.md Sections 7.8, 7.10
"""

import argparse
import math
import os

import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from fragile.learning.core.benchmarks import BaselineClassifier, StandardVQ, VanillaAE
from fragile.learning.core.layers import (
    FactorizedJumpOperator,
    InvariantChartClassifier,
    TopoEncoderPrimitives,
)
from fragile.learning.hyperbolic_losses import (
    # KEEP losses
    SupervisedTopologyLoss,
    compute_routing_entropy,
    compute_diversity_loss,
    compute_codebook_centering_loss,
    compute_residual_scale_loss,
    compute_window_loss,
    compute_code_entropy_loss,
    compute_per_chart_code_entropy_loss,
    compute_orthogonality_loss,
    compute_jump_consistency_loss,
    get_jump_weight_schedule,
    # NEW losses
    compute_hyperbolic_uniformity_loss,
    compute_hyperbolic_contrastive_loss,
    compute_radial_calibration_loss,
    compute_codebook_spread_loss,
    compute_symbol_purity_loss,
    compute_symbol_calibration_loss,
    # Anti-collapse penalties
    compute_chart_collapse_penalty,
    compute_code_collapse_penalty,
)
# DROP-flagged (still importable from old module for backwards compat)
from fragile.learning.core.losses import (
    compute_variance_loss,
    compute_separation_loss,
    compute_chart_center_separation_loss,
    compute_disentangle_loss,
    compute_kl_prior_loss,
)
from fragile.learning.checkpoints import (  # noqa: F401
    benchmarks_compatible as _benchmarks_compatible,
    compute_ami,
    compute_grad_norm as _compute_grad_norm,
    compute_matching_hidden_dim,
    compute_param_norm as _compute_param_norm,
    compute_perplexity as _compute_perplexity_from_assignments,
    count_parameters,
    load_benchmarks,
    load_checkpoint,
    load_optimizer_state as _load_optimizer_state,
    save_benchmarks,
    save_checkpoint,
    save_model_checkpoint,
)

# --- Extracted modules ---
from fragile.learning.config import TopoEncoderConfig  # noqa: F401
from fragile.learning.data import (
    create_data_snapshot,
    create_dataloaders,
    load_dataset,
    restore_dataset,
)
from fragile.learning.mlflow_logging import (  # noqa: F401
    end_mlflow_run as _end_mlflow_run,
    log_mlflow_metrics as _log_mlflow_metrics,
    log_mlflow_params as _log_mlflow_params,
    start_mlflow_run as _start_mlflow_run,
)


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
        # Tier 4 losses
        "kl_prior": [],
        # Tier 5: Jump Operator
        "jump": [],
        # New geometric losses
        "hyp_uniformity": [],
        "hyp_contrastive": [],
        "radial_cal": [],
        "codebook_spread": [],
        # New symbol losses
        "sym_purity": [],
        "sym_calibration": [],
        # Anti-collapse penalties
        "chart_collapse": [],
        "code_collapse": [],
        # Texture flow
        "flow": [],
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
    # Create output directory with subdirectories for each model type
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(f"{config.output_dir}/topoencoder", exist_ok=True)
    os.makedirs(f"{config.output_dir}/vq", exist_ok=True)
    os.makedirs(f"{config.output_dir}/ae", exist_ok=True)
    print(f"Saving checkpoints to: {config.output_dir}/")
    benchmarks_path = os.path.join(config.output_dir, "benchmarks.pt")
    benchmarks_candidates = [benchmarks_path]
    if config.resume_checkpoint:
        resume_dir = os.path.dirname(config.resume_checkpoint)
        if resume_dir:
            # Check same dir (old flat layout) and parent dir (new subdir layout)
            for candidate_dir in [resume_dir, os.path.dirname(resume_dir)]:
                candidate = os.path.join(candidate_dir, "benchmarks.pt")
                if candidate not in benchmarks_candidates:
                    benchmarks_candidates.append(candidate)
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
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

        bundle = restore_dataset(
            data_snapshot=checkpoint.get("data", {}),
            dataset_fallback=config.dataset,
        )
        data_snapshot = checkpoint.get("data", {})
        config.input_dim = bundle.input_dim

        print(f"Resuming from checkpoint: {config.resume_checkpoint}")
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
        bundle = load_dataset(
            dataset=config.dataset,
            n_samples=config.n_samples,
            test_split=config.test_split,
        )
        config.input_dim = bundle.input_dim
        data_snapshot = create_data_snapshot(bundle)

    # Unpack bundle for local use
    X_train, X_test = bundle.X_train, bundle.X_test
    _labels_train, labels_test = bundle.labels_train, bundle.labels_test
    _colors_train, _colors_test = bundle.colors_train, bundle.colors_test
    dataset_name = bundle.dataset_name
    labels_full = bundle.labels_full

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
        soft_equiv_metric=config.soft_equiv_metric,
        soft_equiv_bundle_size=config.soft_equiv_bundle_size,
        soft_equiv_hidden_dim=config.soft_equiv_hidden_dim,
        soft_equiv_use_spectral_norm=config.soft_equiv_use_spectral_norm,
        soft_equiv_zero_self_mixing=config.soft_equiv_zero_self_mixing,
        soft_equiv_soft_assign=config.soft_equiv_soft_assign,
        soft_equiv_temperature=config.soft_equiv_temperature,
        conv_backbone=config.conv_backbone,
        conv_channels=config.conv_channels,
        img_channels=config.img_channels,
        img_size=config.img_size,
        film_conditioning=config.film_conditioning,
        conformal_freq_gating=config.conformal_freq_gating,
        texture_flow=config.texture_flow,
        texture_flow_layers=config.texture_flow_layers,
        texture_flow_hidden=config.texture_flow_hidden,
        texture_flow_clamp=config.texture_flow_clamp,
    )
    _new_features_active = (
        config.film_conditioning or config.conformal_freq_gating or config.texture_flow
    )
    if resume_state is not None and resume_state.get("atlas") is not None:
        model_atlas.load_state_dict(
            resume_state["atlas"], strict=not _new_features_active,
        )
    topo_params = count_parameters(model_atlas)

    # Create StandardVQ with matching parameter count (fair comparison)
    benchmarks_std_state = benchmarks_state.get("std") if benchmarks_compatible else None
    benchmarks_ae_state = benchmarks_state.get("ae") if benchmarks_compatible else None
    benchmarks_loaded_std = False
    benchmarks_loaded_ae = False
    model_std = None
    opt_std = {}
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
        )
        if benchmarks_std_state is not None:
            model_std.load_state_dict(benchmarks_std_state)
            benchmarks_loaded_std = True
        elif resume_state is not None and resume_state.get("std") is not None:
            model_std.load_state_dict(resume_state["std"])
        std_params = count_parameters(model_std)

    # Create VanillaAE with similar parameter count (reconstruction baseline)
    model_ae = None
    opt_ae: optim.Adam | None = None
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

    _log_mlflow_params(
        {
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
        },
        enabled=mlflow_active,
    )

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

    # Optimizers (vanilla Adam + optional CosineAnnealingLR)
    atlas_params = list(model_atlas.parameters()) + list(jump_op.parameters())
    if supervised_loss is not None:
        atlas_params.extend(list(supervised_loss.parameters()))

    opt_std: optim.Adam | None = None
    if train_std and model_std is not None:
        opt_std = optim.Adam(model_std.parameters(), lr=config.lr)

    opt_atlas = optim.Adam(atlas_params, lr=config.lr) if atlas_params else None

    opt_classifier: optim.Adam | None = None
    if classifier_head is not None:
        opt_classifier = optim.Adam(classifier_head.parameters(), lr=classifier_lr)

    opt_classifier_std: optim.Adam | None = None
    if std_classifier_head is not None:
        opt_classifier_std = optim.Adam(std_classifier_head.parameters(), lr=classifier_lr)

    opt_classifier_ae: optim.Adam | None = None
    if ae_classifier_head is not None:
        opt_classifier_ae = optim.Adam(ae_classifier_head.parameters(), lr=classifier_lr)

    opt_ae: optim.Adam | None = None
    if train_ae and model_ae is not None:
        opt_ae = optim.Adam(model_ae.parameters(), lr=config.lr)

    if resume_optim:
        _load_optimizer_state(opt_atlas, resume_optim.get("atlas"), device)
        _load_optimizer_state(opt_std, resume_optim.get("std"), device)
        _load_optimizer_state(opt_ae, resume_optim.get("ae"), device)
        _load_optimizer_state(opt_classifier, resume_optim.get("classifier"), device)
        _load_optimizer_state(opt_classifier_std, resume_optim.get("classifier_std"), device)
        _load_optimizer_state(opt_classifier_ae, resume_optim.get("classifier_ae"), device)

    # CosineAnnealingLR schedulers
    schedulers: list[CosineAnnealingLR] = []
    if config.use_scheduler:
        T_max = max(1, config.epochs - start_epoch)
        for opt in [
            opt_atlas,
            opt_std,
            opt_ae,
            opt_classifier,
            opt_classifier_std,
            opt_classifier_ae,
        ]:
            if opt is not None:
                schedulers.append(CosineAnnealingLR(opt, T_max=T_max, eta_min=config.lr_min))

    # Create data loaders for minibatching
    dataloader, test_dataloader = create_dataloaders(
        bundle,
        config.batch_size,
        config.eval_batch_size,
        device,
    )
    batch_size = config.batch_size if config.batch_size > 0 else len(X_train)
    batches_per_epoch = max(1, len(dataloader))
    _log_mlflow_params({"batches_per_epoch": batches_per_epoch}, enabled=mlflow_active)

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

    print("=" * 60)
    print("Training TopoEncoder (Attentive Atlas)")
    print(f"  Epochs: {config.epochs}, LR: {config.lr}, Batch size: {batch_size}")
    print(f"  Charts: {config.num_charts}, Codes/chart: {config.codes_per_chart}")
    print(f"  λ: entropy={config.entropy_weight}")
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
        n_batches = 0

        for batch_X, batch_labels, _batch_colors in dataloader:
            n_batches += 1
            batch_X = batch_X.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

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
                v_local,
            ) = model_atlas.encoder(batch_X)

            # Decoder forward (dreaming mode - infers routing from z_geo)
            recon_a, dec_w, aux_losses = model_atlas.decoder(z_geo, z_tex, chart_index=None)

            # Core losses
            recon_loss_a = F.mse_loss(recon_a, batch_X)
            if config.conv_backbone and config.recon_grad_weight > 0:
                # Gradient penalty: penalize blurry reconstructions
                C, S = config.img_channels, config.img_size
                x_img = batch_X.view(-1, C, S, S)
                r_img = recon_a.view(-1, C, S, S)
                dx_real = x_img[:, :, :, 1:] - x_img[:, :, :, :-1]
                dx_fake = r_img[:, :, :, 1:] - r_img[:, :, :, :-1]
                dy_real = x_img[:, :, 1:, :] - x_img[:, :, :-1, :]
                dy_fake = r_img[:, :, 1:, :] - r_img[:, :, :-1, :]
                grad_loss = F.mse_loss(dx_fake, dx_real) + F.mse_loss(dy_fake, dy_real)
                recon_loss_a = recon_loss_a + config.recon_grad_weight * grad_loss
            entropy_value = compute_routing_entropy(enc_w)
            entropy_loss = math.log(config.num_charts) - entropy_value
            consistency = model_atlas.compute_consistency_loss(enc_w, dec_w)

            # Tier 1 losses (low overhead)
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

            # Soft equivariant regularization
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

            # Tier 3 losses (geometry/codebook health)
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
            max_code_entropy = math.log(config.codes_per_chart)
            code_entropy_value = max_code_entropy - code_ent_loss.item()
            per_chart_entropy_value = max_code_entropy - per_chart_code_ent_loss.item()

            # Tier 4 losses (invariance)
            kl_loss = torch.tensor(0.0, device=device)
            if config.kl_prior_weight > 0:
                kl_loss = compute_kl_prior_loss(z_n, z_tex)

            # Tier 5: Jump Operator (scheduled warmup - let atlas form before learning transitions)
            current_jump_weight = get_jump_weight_schedule(
                epoch, config.jump_warmup, config.jump_ramp_end, config.jump_weight
            )
            if current_jump_weight > 0:
                jump_loss = compute_jump_consistency_loss(jump_op, z_n_all_charts, enc_w)
            else:
                jump_loss = torch.tensor(0.0, device=device)

            # --- NEW GEOMETRIC LOSSES (always-on, no warmup) ---
            hyp_unif_loss = torch.tensor(0.0, device=device)
            if config.hyperbolic_uniformity_weight > 0:
                hyp_unif_loss = compute_hyperbolic_uniformity_loss(z_geo)

            hyp_contr_loss = torch.tensor(0.0, device=device)
            if config.hyperbolic_contrastive_weight > 0:
                hyp_contr_loss = compute_hyperbolic_contrastive_loss(
                    z_geo, batch_labels, config.hyperbolic_contrastive_margin
                )

            rad_cal_loss = torch.tensor(0.0, device=device)
            if config.radial_calibration_weight > 0:
                rad_cal_loss = compute_radial_calibration_loss(
                    z_geo, enc_w, config.num_charts
                )

            cb_spread_loss = torch.tensor(0.0, device=device)
            if config.codebook_spread_weight > 0:
                cb_spread_loss = compute_codebook_spread_loss(
                    model_atlas.encoder.codebook, config.codebook_spread_margin
                )

            sym_pur_loss = torch.tensor(0.0, device=device)
            if config.symbol_purity_weight > 0:
                sym_pur_loss = compute_symbol_purity_loss(
                    K_chart, indices_stack, batch_labels, enc_w,
                    config.num_charts, config.codes_per_chart,
                )

            sym_cal_loss = torch.tensor(0.0, device=device)
            if config.symbol_calibration_weight > 0:
                sym_cal_loss = compute_symbol_calibration_loss(
                    z_geo, K_chart, indices_stack,
                    config.num_charts, config.codes_per_chart,
                )

            # --- ANTI-COLLAPSE PENALTIES ---
            chart_collapse_loss = torch.tensor(0.0, device=device)
            if config.chart_collapse_weight > 0:
                chart_collapse_loss = compute_chart_collapse_penalty(
                    enc_w, config.num_charts
                )

            code_collapse_loss = torch.tensor(0.0, device=device)
            if config.code_collapse_weight > 0:
                code_collapse_loss = compute_code_collapse_penalty(
                    v_local, model_atlas.encoder.codebook, enc_w,
                    temperature=config.code_collapse_temperature,
                )

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

            # Texture flow loss
            flow_loss = aux_losses.get(
                "flow_loss", torch.tensor(0.0, device=device),
            )

            # Total loss
            loss_a = (
                config.recon_weight * recon_loss_a
                + vq_loss_a
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
                # Tier 4
                + config.kl_prior_weight * kl_loss
                # Tier 5: Jump Operator (scheduled)
                + current_jump_weight * jump_loss
                # New geometric losses
                + config.hyperbolic_uniformity_weight * hyp_unif_loss
                + config.hyperbolic_contrastive_weight * hyp_contr_loss
                + config.radial_calibration_weight * rad_cal_loss
                + config.codebook_spread_weight * cb_spread_loss
                # New symbol losses
                + config.symbol_purity_weight * sym_pur_loss
                + config.symbol_calibration_weight * sym_cal_loss
                # Anti-collapse penalties
                + config.chart_collapse_weight * chart_collapse_loss
                + config.code_collapse_weight * code_collapse_loss
                # Supervised topology
                + config.sup_weight * sup_total
                # Texture flow
                + config.texture_flow_weight * flow_loss
            )

            opt_atlas.zero_grad()
            loss_a.backward()
            grad_norm = _compute_grad_norm(atlas_params)
            param_norm = _compute_param_norm(atlas_params)
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(atlas_params, config.grad_clip)
            grad_norm_val = grad_norm
            param_norm_val = param_norm
            current_lr = opt_atlas.param_groups[0]["lr"]
            update_ratio = (
                current_lr * grad_norm / (param_norm + 1e-12) if param_norm > 0.0 else 0.0
            )
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
            epoch_losses["jump"] += jump_loss.item()
            epoch_losses["hyp_uniformity"] += hyp_unif_loss.item()
            epoch_losses["hyp_contrastive"] += hyp_contr_loss.item()
            epoch_losses["radial_cal"] += rad_cal_loss.item()
            epoch_losses["codebook_spread"] += cb_spread_loss.item()
            epoch_losses["sym_purity"] += sym_pur_loss.item()
            epoch_losses["sym_calibration"] += sym_cal_loss.item()
            epoch_losses["chart_collapse"] += chart_collapse_loss.item()
            epoch_losses["code_collapse"] += code_collapse_loss.item()
            epoch_losses["flow"] += flow_loss.item()
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

        # Step LR schedulers
        for scheduler in schedulers:
            scheduler.step()

        # Logging and checkpointing (matching embed_fragile.py style)
        should_log = epoch % config.log_every == 0 or epoch == config.epochs
        should_save = config.save_every > 0 and (
            epoch % config.save_every == 0 or epoch == config.epochs
        )
        if should_log or should_save:
            # Compute metrics on full dataset (batched to avoid OOM)
            was_training = model_atlas.training
            model_atlas.eval()
            usage_sum = torch.zeros(config.num_charts)
            chart_assignments: list[torch.Tensor] = []
            total = 0
            test_sup_hits = 0
            test_sup_route_sum = 0.0
            test_cls_hits = 0
            std_test_hits = 0
            ae_test_hits = 0
            with torch.no_grad():
                for batch_X, batch_labels, _batch_colors in test_dataloader:
                    batch_X = batch_X.to(device, non_blocking=True)
                    batch_labels = batch_labels.to(device, non_blocking=True)
                    (
                        K_chart_batch,
                        _,
                        _,
                        _,
                        enc_w_batch,
                        z_geo_batch,
                        _,
                        _,
                        _,
                        _c_bar_batch,
                        _v_local_batch,
                    ) = model_atlas.encoder(batch_X)
                    usage_sum += enc_w_batch.sum(dim=0).cpu()
                    chart_assignments.append(K_chart_batch.cpu())
                    total += batch_X.shape[0]
                    if supervised_loss is not None:
                        sup_out = supervised_loss(enc_w_batch, batch_labels, z_geo_batch)
                        p_y_x = torch.matmul(enc_w_batch, supervised_loss.p_y_given_k)
                        test_sup_hits += (p_y_x.argmax(dim=1) == batch_labels).sum().item()
                        test_sup_route_sum += sup_out["loss_route"].item() * batch_X.shape[0]
                    if classifier_head is not None:
                        cls_logits = classifier_head(enc_w_batch, z_geo_batch)
                        test_cls_hits += (cls_logits.argmax(dim=1) == batch_labels).sum().item()
                    if std_classifier_head is not None and model_std is not None:
                        z_std_test = model_std.encoder(batch_X)
                        std_logits_test = std_classifier_head(z_std_test)
                        std_test_hits += (
                            (std_logits_test.argmax(dim=1) == batch_labels).sum().item()
                        )
                    if ae_classifier_head is not None and model_ae is not None:
                        _, z_ae_test = model_ae(batch_X)
                        ae_logits_test = ae_classifier_head(z_ae_test)
                        ae_test_hits += (ae_logits_test.argmax(dim=1) == batch_labels).sum().item()
            if was_training:
                model_atlas.train()
            total = max(total, 1)
            usage = (usage_sum / total).numpy()
            chart_assignments_t = (
                torch.cat(chart_assignments)
                if chart_assignments
                else torch.empty(0, dtype=torch.long)
            )
            chart_assignments = chart_assignments_t.cpu().numpy()
            ami = compute_ami(labels_test, chart_assignments) if chart_assignments.size else 0.0
            perplexity = _compute_perplexity_from_assignments(
                chart_assignments_t, config.num_charts
            )
            test_sup_acc = test_sup_hits / total if supervised_loss is not None else None
            test_sup_route = test_sup_route_sum / total if supervised_loss is not None else None
            test_cls_acc = test_cls_hits / total if classifier_head is not None else None
            std_test_acc = (
                std_test_hits / total
                if std_classifier_head is not None and model_std is not None
                else None
            )
            ae_test_acc = (
                ae_test_hits / total
                if ae_classifier_head is not None and model_ae is not None
                else None
            )

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
            avg_disentangle = loss_components["disentangle"][-1]
            avg_orth = loss_components["orthogonality"][-1]
            avg_code_ent = loss_components["code_entropy"][-1]
            avg_pc_code_ent = loss_components["per_chart_code_entropy"][-1]
            avg_kl_prior = loss_components["kl_prior"][-1]
            avg_jump = loss_components["jump"][-1]
            avg_hyp_unif = loss_components["hyp_uniformity"][-1]
            avg_hyp_contr = loss_components["hyp_contrastive"][-1]
            avg_rad_cal = loss_components["radial_cal"][-1]
            avg_cb_spread = loss_components["codebook_spread"][-1]
            avg_sym_pur = loss_components["sym_purity"][-1]
            avg_sym_cal = loss_components["sym_calibration"][-1]
            avg_chart_collapse = loss_components["chart_collapse"][-1]
            avg_code_collapse = loss_components["code_collapse"][-1]
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
            print(f"Epoch {epoch:5d} | Loss: {avg_loss:.4f} | LR: {avg_lr:.2e}")
            print(f"  Usage: {np.array2string(usage, precision=2, separator=', ')}")
            print(
                f"  Core: recon={avg_recon:.3f} "
                f"vq={avg_vq:.3f} "
                f"entropy={avg_entropy:.3f} "
                f"consist={avg_consistency:.3f} "
                f"var={avg_var:.3f} "
                f"div={avg_div:.3f} "
                f"sep={avg_sep:.3f}"
            )
            print(
                f"  T1: center={avg_codebook_center:.3f} "
                f"chart_sep={avg_chart_center_sep:.3f} "
                f"res_scale={avg_residual_scale:.3f} "
                f"seq_l1={avg_soft_equiv_l1:.3f} "
                f"seq_ratio={avg_soft_equiv_log_ratio:.3f}"
            )
            print(
                f"  T2: window={avg_window:.3f} "
                f"disentangle={avg_disentangle:.3f}"
            )
            print(
                f"  T3: orth={avg_orth:.3f} "
                f"code_ent={avg_code_ent:.3f} "
                f"pc_code_ent={avg_pc_code_ent:.3f}"
            )
            print(f"  T4: kl_prior={avg_kl_prior:.3f}")
            print(f"  Jump: {avg_jump:.3f} (λ={log_jump_weight:.3f})")
            print(
                f"  Geo: unif={avg_hyp_unif:.3f} "
                f"contr={avg_hyp_contr:.3f} "
                f"rad_cal={avg_rad_cal:.3f} "
                f"cb_spread={avg_cb_spread:.3f}"
            )
            print(
                f"  Sym: purity={avg_sym_pur:.3f} "
                f"calibration={avg_sym_cal:.3f}"
            )
            print(
                f"  Collapse: chart={avg_chart_collapse:.4f} "
                f"code={avg_code_collapse:.4f}"
            )
            if supervised_loss is not None:
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

            if classifier_head is not None and test_cls_acc is not None:
                print(
                    f"  Readout: train_loss={avg_cls_loss:.3f} "
                    f"train_acc={avg_cls_acc:.3f} "
                    f"test_acc={test_cls_acc:.3f}"
                )
            if (
                std_classifier_head is not None
                and model_std is not None
                and std_test_acc is not None
            ):
                print(
                    f"  Std Readout: train_loss={avg_std_cls_loss:.3f} "
                    f"train_acc={avg_std_cls_acc:.3f} "
                    f"test_acc={std_test_acc:.3f}"
                )
            if ae_classifier_head is not None and model_ae is not None and ae_test_acc is not None:
                print(
                    f"  AE Readout: train_loss={avg_ae_cls_loss:.3f} "
                    f"train_acc={avg_ae_cls_acc:.3f} "
                    f"test_acc={ae_test_acc:.3f}"
                )
            print(f"  Info: I(X;K)={avg_ixk:.3f} H(K)={avg_hk:.3f} H(K|X)={avg_hk_given_x:.3f}")
            print(f"  Code: H(code)={avg_code_entropy:.3f} H(pc_code)={avg_pc_code_entropy:.3f}")
            print(
                f"  LR ctl: grad_norm={avg_grad_norm:.2e} "
                f"upd_ratio={avg_update_ratio:.2e} "
                f"lr={avg_lr:.2e}"
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
                    "loss/disentangle": avg_disentangle,
                    "loss/orthogonality": avg_orth,
                    "loss/code_entropy": avg_code_ent,
                    "loss/per_chart_code_entropy": avg_pc_code_ent,
                    "loss/kl_prior": avg_kl_prior,
                    "loss/jump": avg_jump,
                    "loss/hyp_uniformity": avg_hyp_unif,
                    "loss/hyp_contrastive": avg_hyp_contr,
                    "loss/radial_cal": avg_rad_cal,
                    "loss/codebook_spread": avg_cb_spread,
                    "loss/sym_purity": avg_sym_pur,
                    "loss/sym_calibration": avg_sym_cal,
                    "loss/chart_collapse": avg_chart_collapse,
                    "loss/code_collapse": avg_code_collapse,
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
                _log_mlflow_metrics(mlflow_metrics, step=epoch, enabled=mlflow_active)

            # Save checkpoint
            if should_save:
                save_path = f"{config.output_dir}/topoencoder/epoch_{epoch:05d}.pt"
                checkpoint_metrics = {
                    "std_losses": std_losses,
                    "atlas_losses": atlas_losses,
                    "ae_losses": ae_losses,
                    "loss_components": loss_components,
                    "info_metrics": info_metrics,
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
                    optimizer_atlas=opt_atlas,
                    optimizer_std=opt_std,
                    optimizer_ae=opt_ae,
                    optimizer_classifier=opt_classifier,
                    optimizer_classifier_std=opt_classifier_std,
                    optimizer_classifier_ae=opt_classifier_ae,
                )
                print(f"Checkpoint saved: {save_path}")
                # Save individual VQ and AE checkpoints
                if model_std is not None:
                    vq_path = f"{config.output_dir}/vq/epoch_{epoch:05d}.pt"
                    save_model_checkpoint(
                        vq_path,
                        model_std,
                        opt_std,
                        config,
                        epoch,
                        hidden_dim=std_hidden_dim,
                        model_type="standard_vq",
                        extra_metrics={"losses": std_losses},
                    )
                if model_ae is not None:
                    ae_path = f"{config.output_dir}/ae/epoch_{epoch:05d}.pt"
                    save_model_checkpoint(
                        ae_path,
                        model_ae,
                        opt_ae,
                        config,
                        epoch,
                        hidden_dim=ae_hidden_dim,
                        model_type="vanilla_ae",
                        extra_metrics={"losses": ae_losses},
                    )
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

    total_samples = len(X_test)
    total_elements = max(total_samples * config.input_dim, 1)

    with torch.no_grad():
        # VanillaAE metrics (reconstruction baseline)
        mse_ae = 0.0
        ami_ae = 0.0
        ae_cls_acc = 0.0
        if model_ae is not None:
            model_ae.eval()
            z_ae_parts: list[torch.Tensor] = []
            sse_ae = 0.0
            ae_cls_hits = 0
            for batch_X, batch_labels, _batch_colors in test_dataloader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                recon_ae, z_ae = model_ae(batch_X)
                sse_ae += F.mse_loss(recon_ae, batch_X, reduction="sum").item()
                z_ae_parts.append(z_ae.detach().cpu())
                if ae_classifier_head is not None:
                    ae_logits = ae_classifier_head(z_ae)
                    ae_cls_hits += (ae_logits.argmax(dim=1) == batch_labels).sum().item()
            if total_samples > 0:
                mse_ae = sse_ae / total_elements
            if z_ae_parts:
                z_ae_np = torch.cat(z_ae_parts).numpy()
                kmeans = KMeans(n_clusters=config.num_charts, random_state=42, n_init=10)
                ae_clusters = kmeans.fit_predict(z_ae_np)
                ami_ae = compute_ami(labels_test, ae_clusters)
            if ae_classifier_head is not None and total_samples > 0:
                ae_cls_acc = ae_cls_hits / total_samples

        # Standard VQ metrics
        mse_std = 0.0
        ami_std = 0.0
        std_perplexity = 0.0
        std_cls_acc = 0.0
        if model_std is not None:
            model_std.eval()
            sse_std = 0.0
            indices_parts: list[torch.Tensor] = []
            std_cls_hits = 0
            for batch_X, batch_labels, _batch_colors in test_dataloader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                recon_std, _, indices_s = model_std(batch_X)
                sse_std += F.mse_loss(recon_std, batch_X, reduction="sum").item()
                indices_parts.append(indices_s.detach().cpu())
                if std_classifier_head is not None:
                    z_std = model_std.encoder(batch_X)
                    std_logits = std_classifier_head(z_std)
                    std_cls_hits += (std_logits.argmax(dim=1) == batch_labels).sum().item()
            if total_samples > 0:
                mse_std = sse_std / total_elements
            if indices_parts:
                indices_s = torch.cat(indices_parts)
                std_perplexity = model_std.compute_perplexity(indices_s)
                vq_clusters = indices_s.numpy() % config.num_charts
                ami_std = compute_ami(labels_test, vq_clusters)
            if std_classifier_head is not None and total_samples > 0:
                std_cls_acc = std_cls_hits / total_samples

        # Atlas metrics (use dreaming mode to test autonomous routing)
        model_atlas.eval()
        sse_atlas = 0.0
        consistency_sum = 0.0
        chart_assignments_parts: list[torch.Tensor] = []
        sup_hits = 0
        cls_hits = 0
        for batch_X, batch_labels, _batch_colors in test_dataloader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            recon_atlas, _, enc_w, dec_w, K_chart, z_geo, _z_n, _c_bar, _aux = model_atlas(
                batch_X, use_hard_routing=False
            )
            sse_atlas += F.mse_loss(recon_atlas, batch_X, reduction="sum").item()
            consistency_sum += (
                model_atlas.compute_consistency_loss(enc_w, dec_w).item() * batch_X.shape[0]
            )
            chart_assignments_parts.append(K_chart.detach().cpu())
            if supervised_loss is not None:
                p_y_x = torch.matmul(enc_w, supervised_loss.p_y_given_k)
                sup_hits += (p_y_x.argmax(dim=1) == batch_labels).sum().item()
            if classifier_head is not None:
                cls_logits = classifier_head(enc_w, z_geo)
                cls_hits += (cls_logits.argmax(dim=1) == batch_labels).sum().item()

        chart_assignments_t = (
            torch.cat(chart_assignments_parts)
            if chart_assignments_parts
            else torch.empty(0, dtype=torch.long)
        )
        chart_assignments = chart_assignments_t.numpy()
        atlas_perplexity = _compute_perplexity_from_assignments(
            chart_assignments_t, config.num_charts
        )
        ami_atlas = compute_ami(labels_test, chart_assignments) if chart_assignments.size else 0.0
        mse_atlas = sse_atlas / total_elements if total_samples > 0 else 0.0
        final_consistency = consistency_sum / total_samples if total_samples > 0 else 0.0
        sup_acc = (
            sup_hits / total_samples if supervised_loss is not None and total_samples > 0 else 0.0
        )
        cls_acc = (
            cls_hits / total_samples if classifier_head is not None and total_samples > 0 else 0.0
        )

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

    final_checkpoint = f"{config.output_dir}/topoencoder/final.pt"
    final_metrics = {
        "std_losses": std_losses,
        "atlas_losses": atlas_losses,
        "ae_losses": ae_losses,
        "loss_components": loss_components,
        "info_metrics": info_metrics,
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
        optimizer_atlas=opt_atlas,
        optimizer_std=opt_std,
        optimizer_ae=opt_ae,
        optimizer_classifier=opt_classifier,
        optimizer_classifier_std=opt_classifier_std,
        optimizer_classifier_ae=opt_classifier_ae,
    )
    print(f"\nFinal checkpoint saved to: {final_checkpoint}")
    # Save final individual VQ and AE checkpoints
    if model_std is not None:
        vq_final = f"{config.output_dir}/vq/final.pt"
        save_model_checkpoint(
            vq_final,
            model_std,
            opt_std,
            config,
            config.epochs,
            hidden_dim=std_hidden_dim,
            model_type="standard_vq",
            extra_metrics={"losses": std_losses, "mse": mse_std, "ami": ami_std},
        )
        print(f"VQ final checkpoint saved to: {vq_final}")
    if model_ae is not None:
        ae_final = f"{config.output_dir}/ae/final.pt"
        save_model_checkpoint(
            ae_final,
            model_ae,
            opt_ae,
            config,
            config.epochs,
            hidden_dim=ae_hidden_dim,
            model_type="vanilla_ae",
            extra_metrics={"losses": ae_losses, "mse": mse_ae, "ami": ami_ae},
        )
        print(f"AE final checkpoint saved to: {ae_final}")
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
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="Dataset to use (mnist or fashion_mnist)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training (0 = full batch)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=0,
        help="Batch size for eval/logging (0 = use batch_size)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=60000,
        help="Number of samples to use (60000 = full MNIST train set)",
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
        help="Number of atlas charts (default: 10 for MNIST/Fashion-MNIST)",
    )
    parser.add_argument(
        "--codes_per_chart",
        type=int,
        default=16,
        help="VQ codes per chart",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=2,
        help="Latent dimension for TopoEncoder (default: 2)",
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
        default=10,
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
    parser.add_argument(
        "--mlflow_run_id",
        type=str,
        default="",
        help="MLflow run ID to resume logging into an existing run",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    # Core loss weights
    parser.add_argument(
        "--entropy_weight",
        type=float,
        default=0.1,
        help="Routing entropy weight (prevents chart collapse, default: 0.1)",
    )
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=0.1,
        help="Encoder/decoder routing consistency weight (default: 0.1)",
    )

    # Reconstruction
    parser.add_argument(
        "--recon_weight", type=float, default=1.0,
        help="Reconstruction loss weight (default: 1.0)",
    )
    parser.add_argument(
        "--recon_grad_weight", type=float, default=1.0,
        help="Gradient penalty weight for sharp reconstructions (0 = disable, default: 1.0)",
    )

    # Tier 1 losses
    parser.add_argument(
        "--variance_weight",
        type=float,
        default=0.0,
        help="Variance loss weight (DROP: disabled, default: 0.0)",
    )
    parser.add_argument(
        "--diversity_weight",
        type=float,
        default=0.1,
        help="Diversity loss weight (prevents chart collapse, default: 0.1)",
    )
    parser.add_argument(
        "--separation_weight",
        type=float,
        default=0.0,
        help="Separation loss weight (DROP: disabled, default: 0.0)",
    )
    parser.add_argument(
        "--separation_margin",
        type=float,
        default=2.0,
        help="Minimum distance between chart centers (default: 2.0)",
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
        default=0.0,
        help="Chart center token separation weight (DROP: disabled, default: 0.0)",
    )
    parser.add_argument(
        "--chart_center_sep_margin",
        type=float,
        default=2.0,
        help="Minimum distance between chart center tokens (default: 2.0)",
    )
    parser.add_argument(
        "--residual_scale_weight",
        type=float,
        default=0.01,
        help="Residual scale weight (keeps z_n small, default: 0.01)",
    )

    # Tier 2 losses
    parser.add_argument(
        "--window_weight",
        type=float,
        default=0.5,
        help="Information-stability window loss weight (default: 0.5)",
    )
    parser.add_argument(
        "--disentangle_weight",
        type=float,
        default=0.0,
        help="Gauge coherence / disentangle weight (DROP: disabled, default: 0.0)",
    )

    # Tier 3 losses
    parser.add_argument(
        "--orthogonality_weight",
        type=float,
        default=0.0,
        help="Orthogonality / SVD spread weight (disabled by default, default: 0.0)",
    )
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

    # Tier 4 losses
    parser.add_argument(
        "--kl_prior_weight",
        type=float,
        default=0.0,
        help="KL prior weight (radial energy prior on z_n, z_tex, default: 0.0)",
    )

    # New geometric losses
    parser.add_argument(
        "--hyperbolic_uniformity_weight", type=float, default=0.1,
        help="Hyperbolic uniformity repulsion weight (default: 0.1)",
    )
    parser.add_argument(
        "--hyperbolic_contrastive_weight", type=float, default=0.0,
        help="Hyperbolic contrastive loss weight (default: 0.5)",
    )
    parser.add_argument(
        "--hyperbolic_contrastive_margin", type=float, default=2.0,
        help="Margin for hyperbolic contrastive loss (default: 2.0)",
    )
    parser.add_argument(
        "--radial_calibration_weight", type=float, default=0.1,
        help="Radial calibration loss weight (default: 0.1)",
    )
    parser.add_argument(
        "--codebook_spread_weight", type=float, default=0.05,
        help="Codebook spread loss weight (default: 0.05)",
    )
    parser.add_argument(
        "--codebook_spread_margin", type=float, default=1.0,
        help="Margin for codebook spread loss (default: 1.0)",
    )
    # New symbol losses
    parser.add_argument(
        "--symbol_purity_weight", type=float, default=0.05,
        help="Symbol purity loss weight (default: 0.05)",
    )
    parser.add_argument(
        "--symbol_calibration_weight", type=float, default=0.05,
        help="Symbol calibration loss weight (default: 0.05)",
    )
    # Anti-collapse penalties
    parser.add_argument(
        "--chart_collapse_weight", type=float, default=1.0,
        help="Chart collapse penalty weight: max(p_k) - 1/K (default: 1.0)",
    )
    parser.add_argument(
        "--code_collapse_weight", type=float, default=0.5,
        help="Code collapse penalty weight: soft code usage skew (default: 0.5)",
    )
    parser.add_argument(
        "--code_collapse_temperature", type=float, default=1.0,
        help="Temperature for soft code assignment in collapse penalty (default: 1.0)",
    )
    parser.add_argument(
        "--use_scheduler",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use CosineAnnealingLR scheduler (default: True)",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=1e-6,
        help="Minimum LR for CosineAnnealingLR (default: 1e-6)",
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
        "--disable_conv",
        action="store_true",
        help="Disable conv backbone, use FC layers for feature extraction/reconstruction",
    )
    parser.add_argument(
        "--conv_channels",
        type=int,
        default=64,
        help="Conv layer channel width (0 = use hidden_dim)",
    )
    parser.add_argument(
        "--img_channels",
        type=int,
        default=1,
        help="Image channels (1 for grayscale MNIST/Fashion-MNIST)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=28,
        help="Image spatial dimension (28 for MNIST/Fashion-MNIST)",
    )
    parser.add_argument(
        "--disable_film_conditioning",
        action="store_true",
        help="Disable per-chart FiLM conditioning in decoder",
    )
    parser.add_argument(
        "--disable_conformal_freq_gating",
        action="store_true",
        help="Disable conformal frequency gating in decoder",
    )
    parser.add_argument(
        "--disable_texture_flow",
        action="store_true",
        help="Disable conditional normalizing flow for texture",
    )
    parser.add_argument(
        "--texture_flow_layers",
        type=int,
        default=4,
        help="Number of coupling layers in texture flow (default: 4)",
    )
    parser.add_argument(
        "--texture_flow_hidden",
        type=int,
        default=64,
        help="Hidden dim in texture flow coupling layers (default: 64)",
    )
    parser.add_argument(
        "--texture_flow_weight",
        type=float,
        default=1.0,
        help="Weight for texture flow loss (default: 1.0)",
    )
    parser.add_argument(
        "--texture_flow_clamp",
        type=float,
        default=5.0,
        help="Clamp value for texture flow log_s (default: 5.0)",
    )
    parser.add_argument(
        "--baseline_attn",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Enable attention blocks in StandardVQ/VanillaAE baselines (default: True)",
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
        config_data = dict(checkpoint["config"])
        # Remove fields from old checkpoints that no longer exist in config
        removed_fields = [
            "thermo_temperature_decay",
            "thermo_temperature_floor",
            "thermo_varentropy_gamma",
            "thermo_alignment_damping",
            "thermo_trust_region",
            "thermo_trust_region_eps",
            "thermo_snr_eps",
            "thermo_snr_floor",
            "thermo_thermal_conductivity",
            "thermo_history_window",
            "thermo_varentropy_min_history",
            "thermo_varentropy_eps",
            "thermo_use_loss_varentropy",
            "adaptive_lr",
            "lr_max",
            "lr_increase_factor",
            "lr_decrease_factor",
            "lr_max_update_ratio",
            "lr_ema_decay",
            "lr_loss_increase_tol",
            "lr_grounding_warmup_epochs",
            "lr_unstable_patience",
            "lr_stable_patience",
            "lr_grounding_ema_decay",
            "lr_recovery_factor",
            "lr_recovery_threshold",
            "lr_plateau_patience",
            "lr_plateau_tol",
            "adaptive_weights",
            "adaptive_warmup_epochs",
            "adaptive_dual_eta",
            "adaptive_pi_kp",
            "adaptive_pi_ki",
            "adaptive_pi_kd",
            "adaptive_lambda_min",
            "adaptive_lambda_max",
            "adaptive_violation_clip",
            "adaptive_target_ratio",
            "adaptive_target_ema_decay",
            "adaptive_target_min",
            "entropy_target_ratio",
            "hk_target_ratio",
            "code_entropy_target_ratio",
            "consistency_target",
            "use_learned_precisions",
            "loss_rescale",
            "loss_rescale_reference",
            "loss_rescale_target_ratio",
            "loss_rescale_ema_decay",
            "loss_rescale_min",
            "loss_rescale_max",
            "loss_rescale_eps",
            # Dropped losses (orbit/vicreg/augmentation)
            "orbit_weight",
            "vicreg_inv_weight",
            "augment_noise_std",
            "augment_rotation_max",
            # Removed vision/CIFAR support
            "vision_preproc",
            "vision_in_channels",
            "vision_height",
            "vision_width",
            "vision_num_rotations",
            "vision_kernel_size",
            "vision_use_reflections",
            "vision_norm_nonlinearity",
            "vision_norm_bias",
            "vision_backbone_type",
            "vision_cifar_base_channels",
            "vision_cifar_bundle_size",
            "baseline_vision_preproc",
            "enable_cifar_backbone",
            "cifar_backbone_type",
            "cifar_base_channels",
            "cifar_bundle_size",
            # Removed warmup schedules (new losses are always-on)
            "new_geo_warmup",
            "new_sym_warmup",
        ]
        for key in removed_fields:
            config_data.pop(key, None)
        config = TopoEncoderConfig(**config_data)
        config.resume_checkpoint = args.resume
        config.epochs = max(config.epochs, args.epochs)
        config.log_every = args.log_every
        config.save_every = args.save_every
        if args.output_dir is not None:
            config.output_dir = args.output_dir
        config.device = args.device
        config.eval_batch_size = args.eval_batch_size
        config.use_scheduler = args.use_scheduler
        config.lr_min = args.lr_min
        config.window_eps_ground = args.window_eps_ground
        config.consistency_weight = args.consistency_weight
        config.recon_weight = args.recon_weight
        config.recon_grad_weight = args.recon_grad_weight
        config.variance_weight = args.variance_weight
        config.diversity_weight = args.diversity_weight
        config.separation_weight = args.separation_weight
        config.separation_margin = args.separation_margin
        config.codebook_center_weight = args.codebook_center_weight
        config.chart_center_sep_weight = args.chart_center_sep_weight
        config.chart_center_sep_margin = args.chart_center_sep_margin
        config.residual_scale_weight = args.residual_scale_weight
        config.disentangle_weight = args.disentangle_weight
        config.orthogonality_weight = args.orthogonality_weight
        config.kl_prior_weight = args.kl_prior_weight
        # New geometric losses
        config.hyperbolic_uniformity_weight = args.hyperbolic_uniformity_weight
        config.hyperbolic_contrastive_weight = args.hyperbolic_contrastive_weight
        config.hyperbolic_contrastive_margin = args.hyperbolic_contrastive_margin
        config.radial_calibration_weight = args.radial_calibration_weight
        config.codebook_spread_weight = args.codebook_spread_weight
        config.codebook_spread_margin = args.codebook_spread_margin
        config.symbol_purity_weight = args.symbol_purity_weight
        config.symbol_calibration_weight = args.symbol_calibration_weight
        config.chart_collapse_weight = args.chart_collapse_weight
        config.code_collapse_weight = args.code_collapse_weight
        config.code_collapse_temperature = args.code_collapse_temperature
        config.covariant_attn = args.covariant_attn
        config.covariant_attn_tensorization = args.covariant_attn_tensorization
        config.covariant_attn_rank = args.covariant_attn_rank
        config.covariant_attn_tau_min = args.covariant_attn_tau_min
        config.covariant_attn_denom_min = args.covariant_attn_denom_min
        config.covariant_attn_use_transport = args.covariant_attn_use_transport
        config.covariant_attn_transport_eps = args.covariant_attn_transport_eps
        config.soft_equiv_metric = args.soft_equiv_metric
        config.soft_equiv_bundle_size = args.soft_equiv_bundle_size or None
        config.soft_equiv_hidden_dim = args.soft_equiv_hidden_dim
        config.soft_equiv_use_spectral_norm = args.soft_equiv_use_spectral_norm
        config.soft_equiv_zero_self_mixing = args.soft_equiv_zero_self_mixing
        config.soft_equiv_soft_assign = args.soft_equiv_soft_assign
        config.soft_equiv_temperature = args.soft_equiv_temperature
        config.soft_equiv_l1_weight = args.soft_equiv_l1_weight
        config.soft_equiv_log_ratio_weight = args.soft_equiv_log_ratio_weight
        config.conv_backbone = not args.disable_conv
        config.conv_channels = args.conv_channels
        config.img_channels = args.img_channels
        config.img_size = args.img_size
        config.film_conditioning = not args.disable_film_conditioning
        config.conformal_freq_gating = not args.disable_conformal_freq_gating
        config.texture_flow = not args.disable_texture_flow
        config.texture_flow_layers = args.texture_flow_layers
        config.texture_flow_hidden = args.texture_flow_hidden
        config.texture_flow_weight = args.texture_flow_weight
        config.texture_flow_clamp = args.texture_flow_clamp
        config.baseline_attn = args.baseline_attn
        config.baseline_attn_tokens = args.baseline_attn_tokens
        config.baseline_attn_dim = args.baseline_attn_dim
        config.baseline_attn_heads = args.baseline_attn_heads
        config.baseline_attn_dropout = args.baseline_attn_dropout
        config.mlflow = args.mlflow
        config.mlflow_tracking_uri = args.mlflow_tracking_uri
        config.mlflow_experiment = args.mlflow_experiment
        config.mlflow_run_name = args.mlflow_run_name
        config.mlflow_run_id = args.mlflow_run_id
    else:
        output_dir = args.output_dir or "outputs/topoencoder"
        config = TopoEncoderConfig(
            dataset=args.dataset,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
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
            soft_equiv_metric=args.soft_equiv_metric,
            soft_equiv_bundle_size=args.soft_equiv_bundle_size or None,
            soft_equiv_hidden_dim=args.soft_equiv_hidden_dim,
            soft_equiv_use_spectral_norm=args.soft_equiv_use_spectral_norm,
            soft_equiv_zero_self_mixing=args.soft_equiv_zero_self_mixing,
            soft_equiv_soft_assign=args.soft_equiv_soft_assign,
            soft_equiv_temperature=args.soft_equiv_temperature,
            soft_equiv_l1_weight=args.soft_equiv_l1_weight,
            soft_equiv_log_ratio_weight=args.soft_equiv_log_ratio_weight,
            conv_backbone=not args.disable_conv,
            conv_channels=args.conv_channels,
            img_channels=args.img_channels,
            img_size=args.img_size,
            film_conditioning=not args.disable_film_conditioning,
            conformal_freq_gating=not args.disable_conformal_freq_gating,
            texture_flow=not args.disable_texture_flow,
            texture_flow_layers=args.texture_flow_layers,
            texture_flow_hidden=args.texture_flow_hidden,
            texture_flow_weight=args.texture_flow_weight,
            texture_flow_clamp=args.texture_flow_clamp,
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
            mlflow_run_id=args.mlflow_run_id,
            # Loss weights
            entropy_weight=args.entropy_weight,
            consistency_weight=args.consistency_weight,
            recon_weight=args.recon_weight,
            recon_grad_weight=args.recon_grad_weight,
            variance_weight=args.variance_weight,
            diversity_weight=args.diversity_weight,
            separation_weight=args.separation_weight,
            separation_margin=args.separation_margin,
            codebook_center_weight=args.codebook_center_weight,
            chart_center_sep_weight=args.chart_center_sep_weight,
            chart_center_sep_margin=args.chart_center_sep_margin,
            residual_scale_weight=args.residual_scale_weight,
            window_weight=args.window_weight,
            disentangle_weight=args.disentangle_weight,
            orthogonality_weight=args.orthogonality_weight,
            per_chart_code_entropy_weight=args.per_chart_code_entropy_weight,
            code_entropy_weight=args.code_entropy_weight,
            kl_prior_weight=args.kl_prior_weight,
            # New geometric losses
            hyperbolic_uniformity_weight=args.hyperbolic_uniformity_weight,
            hyperbolic_contrastive_weight=args.hyperbolic_contrastive_weight,
            hyperbolic_contrastive_margin=args.hyperbolic_contrastive_margin,
            radial_calibration_weight=args.radial_calibration_weight,
            codebook_spread_weight=args.codebook_spread_weight,
            codebook_spread_margin=args.codebook_spread_margin,
            symbol_purity_weight=args.symbol_purity_weight,
            symbol_calibration_weight=args.symbol_calibration_weight,
            # Anti-collapse penalties
            chart_collapse_weight=args.chart_collapse_weight,
            code_collapse_weight=args.code_collapse_weight,
            code_collapse_temperature=args.code_collapse_temperature,
            # Training dynamics
            grad_clip=args.grad_clip,
            use_scheduler=args.use_scheduler,
            lr_min=args.lr_min,
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
    print(f"  Conv backbone: {config.conv_backbone}")
    print(f"  Conv channels: {config.conv_channels}")
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
