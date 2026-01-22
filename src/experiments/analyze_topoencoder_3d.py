"""
Analyze TopoEncoder checkpoints and generate plots with 3D latent views.

This mirrors analyze_topoencoder_2d.py but renders the latent space in 3D
for more rigorous inspection of 3D latent spaces.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
from typing import Any

import matplotlib
import numpy as np
import torch


if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from dataviz import (
    _compute_chart_code_colors,
    _create_image_grid,
    _plot_hyperbolic_tree,
    _select_class_representatives,
    _tensor_to_image,
    visualize_results_images,
)
from fragile.core.layers import (
    FactorizedJumpOperator,
    StandardVQ,
    TopoEncoderPrimitives,
    VanillaAE,
)
from fragile.datasets import CIFAR10_CLASSES


def _load_checkpoint(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_benchmarks(checkpoint_path: str) -> dict[str, Any] | None:
    bench_path = os.path.join(os.path.dirname(checkpoint_path), "benchmarks.pt")
    if not os.path.exists(bench_path):
        return None
    try:
        return torch.load(bench_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(bench_path, map_location="cpu")


def _load_model_state(model: torch.nn.Module, state: dict, name: str) -> None:
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(
            f"  Warning: {name} load_state_dict missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)}"
        )


def _prepare_models(
    config: dict,
    state: dict,
    metrics: dict,
    bench_state: dict | None,
    bench_dims: dict | None,
    device: str,
) -> dict[str, Any]:
    baseline_attn = bool(config.get("baseline_attn", False))
    baseline_attn_tokens = int(config.get("baseline_attn_tokens", 4))
    baseline_attn_dim = int(config.get("baseline_attn_dim", 32))
    baseline_attn_heads = int(config.get("baseline_attn_heads", 4))
    baseline_attn_dropout = float(config.get("baseline_attn_dropout", 0.0))
    baseline_vision_preproc = bool(config.get("baseline_vision_preproc", False))
    vision_in_channels = int(config.get("vision_in_channels", 0))
    vision_height = int(config.get("vision_height", 0))
    vision_width = int(config.get("vision_width", 0))

    bundle_size = config.get("bundle_size")
    if isinstance(bundle_size, int) and bundle_size <= 0:
        bundle_size = None
    soft_equiv_bundle_size = config.get("soft_equiv_bundle_size")
    if isinstance(soft_equiv_bundle_size, int) and soft_equiv_bundle_size <= 0:
        soft_equiv_bundle_size = None
    soft_equiv_soft_assign = config.get("soft_equiv_soft_assign")
    if soft_equiv_soft_assign is None:
        soft_equiv_soft_assign = True
    soft_equiv_temperature = config.get("soft_equiv_temperature")
    if soft_equiv_temperature is None:
        soft_equiv_temperature = 1.0
    covariant_attn = config.get("covariant_attn")
    if covariant_attn is None:
        covariant_attn = True

    model_kwargs = {
        "input_dim": config["input_dim"],
        "hidden_dim": config["hidden_dim"],
        "latent_dim": config["latent_dim"],
        "num_charts": config["num_charts"],
        "codes_per_chart": config["codes_per_chart"],
        "bundle_size": bundle_size,
        "covariant_attn": covariant_attn,
        "covariant_attn_tensorization": config.get("covariant_attn_tensorization", "sum"),
        "covariant_attn_rank": config.get("covariant_attn_rank", 8),
        "covariant_attn_tau_min": config.get("covariant_attn_tau_min", 1e-2),
        "covariant_attn_denom_min": config.get("covariant_attn_denom_min", 1e-3),
        "covariant_attn_use_transport": config.get("covariant_attn_use_transport", True),
        "covariant_attn_transport_eps": config.get("covariant_attn_transport_eps", 1e-3),
        "vision_preproc": config.get("vision_preproc", False),
        "vision_in_channels": config.get("vision_in_channels", 0),
        "vision_height": config.get("vision_height", 0),
        "vision_width": config.get("vision_width", 0),
        "vision_num_rotations": config.get("vision_num_rotations", 8),
        "vision_kernel_size": config.get("vision_kernel_size", 5),
        "vision_use_reflections": config.get("vision_use_reflections", False),
        "vision_norm_nonlinearity": config.get("vision_norm_nonlinearity", "n_sigmoid"),
        "vision_norm_bias": config.get("vision_norm_bias", True),
        "soft_equiv_metric": config.get("soft_equiv_metric", False),
        "soft_equiv_bundle_size": soft_equiv_bundle_size,
        "soft_equiv_hidden_dim": config.get("soft_equiv_hidden_dim", 64),
        "soft_equiv_use_spectral_norm": config.get("soft_equiv_use_spectral_norm", True),
        "soft_equiv_zero_self_mixing": config.get("soft_equiv_zero_self_mixing", False),
        "soft_equiv_soft_assign": soft_equiv_soft_assign,
        "soft_equiv_temperature": soft_equiv_temperature,
    }
    model_atlas = TopoEncoderPrimitives(**model_kwargs).to(device)
    _load_model_state(model_atlas, state["atlas"], "atlas")
    model_atlas.eval()

    model_std = None
    if state.get("std") is not None and not config.get("disable_vq", False):
        std_hidden_dim = metrics.get("std_hidden_dim", config["hidden_dim"])
        model_std = StandardVQ(
            input_dim=config["input_dim"],
            hidden_dim=std_hidden_dim,
            latent_dim=config["latent_dim"],
            num_codes=config["num_codes_standard"],
            use_attention=baseline_attn,
            attn_tokens=baseline_attn_tokens,
            attn_dim=baseline_attn_dim,
            attn_heads=baseline_attn_heads,
            attn_dropout=baseline_attn_dropout,
            vision_preproc=baseline_vision_preproc,
            vision_in_channels=vision_in_channels,
            vision_height=vision_height,
            vision_width=vision_width,
        ).to(device)
        _load_model_state(model_std, state["std"], "std")
        model_std.eval()
    elif (
        bench_state is not None
        and bench_state.get("std") is not None
        and not config.get("disable_vq", False)
    ):
        std_hidden_dim = (
            (bench_dims or {}).get("std_hidden_dim") or metrics.get("std_hidden_dim")
        ) or config["hidden_dim"]
        model_std = StandardVQ(
            input_dim=config["input_dim"],
            hidden_dim=int(std_hidden_dim),
            latent_dim=config["latent_dim"],
            num_codes=config["num_codes_standard"],
            use_attention=baseline_attn,
            attn_tokens=baseline_attn_tokens,
            attn_dim=baseline_attn_dim,
            attn_heads=baseline_attn_heads,
            attn_dropout=baseline_attn_dropout,
            vision_preproc=baseline_vision_preproc,
            vision_in_channels=vision_in_channels,
            vision_height=vision_height,
            vision_width=vision_width,
        ).to(device)
        _load_model_state(model_std, bench_state["std"], "std")
        model_std.eval()

    model_ae = None
    if state.get("ae") is not None and not config.get("disable_ae", False):
        ae_hidden_dim = metrics.get("ae_hidden_dim", config["hidden_dim"])
        model_ae = VanillaAE(
            input_dim=config["input_dim"],
            hidden_dim=ae_hidden_dim,
            latent_dim=config["latent_dim"],
            use_attention=baseline_attn,
            attn_tokens=baseline_attn_tokens,
            attn_dim=baseline_attn_dim,
            attn_heads=baseline_attn_heads,
            attn_dropout=baseline_attn_dropout,
            vision_preproc=baseline_vision_preproc,
            vision_in_channels=vision_in_channels,
            vision_height=vision_height,
            vision_width=vision_width,
        ).to(device)
        _load_model_state(model_ae, state["ae"], "ae")
        model_ae.eval()
    elif (
        bench_state is not None
        and bench_state.get("ae") is not None
        and not config.get("disable_ae", False)
    ):
        ae_hidden_dim = (
            (bench_dims or {}).get("ae_hidden_dim") or metrics.get("ae_hidden_dim")
        ) or config["hidden_dim"]
        model_ae = VanillaAE(
            input_dim=config["input_dim"],
            hidden_dim=int(ae_hidden_dim),
            latent_dim=config["latent_dim"],
            use_attention=baseline_attn,
            attn_tokens=baseline_attn_tokens,
            attn_dim=baseline_attn_dim,
            attn_heads=baseline_attn_heads,
            attn_dropout=baseline_attn_dropout,
            vision_preproc=baseline_vision_preproc,
            vision_in_channels=vision_in_channels,
            vision_height=vision_height,
            vision_width=vision_width,
        ).to(device)
        _load_model_state(model_ae, bench_state["ae"], "ae")
        model_ae.eval()

    jump_op = None
    if state.get("jump") is not None:
        jump_op = FactorizedJumpOperator(
            num_charts=config["num_charts"],
            latent_dim=config["latent_dim"],
            global_rank=config["jump_global_rank"],
        ).to(device)
        _load_model_state(jump_op, state["jump"], "jump")
        jump_op.eval()

    return {
        "model_atlas": model_atlas,
        "model_std": model_std,
        "model_ae": model_ae,
        "jump_op": jump_op,
    }


def _dataset_specs(dataset: str) -> tuple[list[str], tuple[int, int, int]]:
    if dataset == "mnist":
        class_names = [str(i) for i in range(10)]
        return class_names, (28, 28, 1)
    if dataset == "cifar10":
        return list(CIFAR10_CLASSES), (32, 32, 3)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _checkpoint_sort_key(path: str) -> tuple[int, int, str]:
    name = os.path.basename(path)
    if "final" in name:
        return (1, 0, name)
    match = re.search(r"(\\d+)", name)
    if match:
        return (0, int(match.group(1)), name)
    return (0, -1, name)


def _collect_checkpoints(checkpoint: str | None, checkpoint_dir: str | None) -> list[str]:
    if checkpoint:
        return [checkpoint]
    if checkpoint_dir:
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        candidates = [
            os.path.join(checkpoint_dir, fname)
            for fname in os.listdir(checkpoint_dir)
            if fname.endswith(".pt") and fname != "benchmarks.pt"
        ]
        if not candidates:
            raise FileNotFoundError(f"No .pt checkpoints found in: {checkpoint_dir}")
        return sorted(candidates, key=_checkpoint_sort_key)
    msg = "Provide --checkpoint or --checkpoint_dir."
    raise ValueError(msg)


def _visualize_latent_images_3d(
    model: TopoEncoderPrimitives,
    X: torch.Tensor,
    labels: np.ndarray,
    class_names: list[str],
    save_path: str,
    epoch: int | None = None,
    jump_op: FactorizedJumpOperator | None = None,
    image_shape: tuple = (32, 32, 3),
    precomputed: dict | None = None,
) -> None:
    plt.close("all")
    gc.collect()

    model.eval()
    num_classes = len(class_names)

    with torch.no_grad():
        if precomputed is None:
            K_chart, K_code, _, _z_tex, _enc_w, z_geo, _, indices_out, _z_n_all, _c_bar = (
                model.encoder(X)
            )
            recon, _, _, _, _, _, _, _ = model(X, use_hard_routing=False)
        else:
            K_chart = precomputed["K_chart"]
            K_code = precomputed["K_code"]
            z_geo = precomputed["z_geo"]
            indices_out = precomputed["indices_out"]
            recon = precomputed["recon"]

        z = z_geo.cpu().numpy()
        if z.shape[1] < 3:
            raise ValueError(
                f"Latent dimension is {z.shape[1]} (<3). Use analyze_topoencoder_2d.py."
            )
        z3 = z[:, :3]
        hard_assign = K_chart.cpu().numpy()
        code_assign = K_code.cpu().numpy()
        indices_np = indices_out.cpu().numpy()

    num_charts = model.encoder.num_charts
    codes_per_chart = model.encoder.codes_per_chart
    chart_cmap = plt.get_cmap("tab10")
    chart_palettes = [
        "Blues",
        "Oranges",
        "Greens",
        "Purples",
        "Reds",
        "Grays",
        "YlOrBr",
        "PuRd",
        "BuGn",
        "RdPu",
    ]

    sample_indices = _select_class_representatives(labels, num_classes)
    sample_images = [_tensor_to_image(X[idx], image_shape) for idx in sample_indices]
    recon_images = [_tensor_to_image(recon[idx], image_shape) for idx in sample_indices]

    fig = plt.figure(figsize=(20, 13))
    title_suffix = f" (Epoch {epoch})" if epoch is not None else " (Final)"

    ax1 = fig.add_subplot(2, 3, 1)
    _create_image_grid(ax1, sample_images, class_names, f"Input Samples{title_suffix}")

    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    ax2.scatter(
        z3[:, 0],
        z3[:, 1],
        z3[:, 2],
        c=labels,
        cmap="tab10",
        vmin=0,
        vmax=max(9, num_classes - 1),
        s=3,
        alpha=0.7,
    )
    cmap = plt.get_cmap("tab10")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(i / max(9, num_classes - 1)),
            markersize=8,
            label=class_names[i],
        )
        for i in range(num_classes)
    ]
    ax2.legend(handles=handles, loc="upper right", fontsize=6, ncol=2)
    ax2.set_title(f"Latent Space (3D){title_suffix}", fontsize=12)
    ax2.set_xlabel("z₁")
    ax2.set_ylabel("z₂")
    ax2.set_zlabel("z₃")

    ax3 = fig.add_subplot(2, 3, 3)
    mse = ((X - recon) ** 2).mean().item()
    _create_image_grid(ax3, recon_images, class_names, f"Reconstructions (MSE: {mse:.5f})")

    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    chart_colors = [chart_cmap(k / max(1, num_charts - 1)) for k in hard_assign]
    ax4.scatter(z3[:, 0], z3[:, 1], z3[:, 2], c=chart_colors, s=3, alpha=0.7)
    handles4 = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=chart_cmap(k / max(1, num_charts - 1)),
            markersize=8,
            label=f"Chart {k}",
        )
        for k in range(num_charts)
    ]
    ax4.legend(handles=handles4, loc="upper right", fontsize=6, ncol=2)
    ax4.set_title("Chart Assignments (3D)", fontsize=12)
    ax4.set_xlabel("z₁")
    ax4.set_ylabel("z₂")
    ax4.set_zlabel("z₃")

    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    symbol_colors = _compute_chart_code_colors(
        hard_assign, code_assign, num_charts, codes_per_chart, chart_palettes
    )
    ax5.scatter(z3[:, 0], z3[:, 1], z3[:, 2], c=symbol_colors, s=3, alpha=0.7)
    ax5.set_title("Code Usage per Chart (3D)", fontsize=12)
    ax5.set_xlabel("z₁")
    ax5.set_ylabel("z₂")
    ax5.set_zlabel("z₃")

    ax6 = fig.add_subplot(2, 3, 6, projection="3d")
    _plot_hyperbolic_tree(
        ax6,
        z[:, :2],
        hard_assign,
        code_assign,
        indices_np,
        num_charts,
        codes_per_chart,
        chart_cmap,
        chart_palettes,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plt.close("all")
    gc.collect()
    print(f"Saved: {save_path}")


def _analyze_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    device: str,
    skip_latent: bool,
    skip_results: bool,
    only_missing: bool,
    batch_size: int,
) -> None:
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    latent_path = os.path.join(output_dir, f"{stem}_latent_3d.png")
    results_path = os.path.join(output_dir, f"{stem}_results.png")

    do_latent = not skip_latent
    do_results = not skip_results
    if only_missing:
        if do_latent and os.path.exists(latent_path):
            do_latent = False
        if do_results and os.path.exists(results_path):
            do_results = False
        if not do_latent and not do_results:
            print(f"Skipping {os.path.basename(checkpoint_path)} (plots already exist)")
            return

    checkpoint = _load_checkpoint(checkpoint_path)
    if "data" not in checkpoint:
        print(f"Skipping {os.path.basename(checkpoint_path)} (no data in checkpoint)")
        return
    config = checkpoint["config"]
    data = checkpoint["data"]
    metrics = checkpoint.get("metrics", {})
    state = checkpoint["state"]

    benchmarks = _load_benchmarks(checkpoint_path)
    bench_state = benchmarks.get("state", {}) if benchmarks else None
    bench_dims = benchmarks.get("dims", {}) if benchmarks else None
    models = _prepare_models(config, state, metrics, bench_state, bench_dims, device)
    model_atlas = models["model_atlas"]
    model_std = models["model_std"]
    model_ae = models["model_ae"]
    jump_op = models["jump_op"]

    X_test = data["X_test"]
    labels_test = data["labels_test"]
    if isinstance(X_test, np.ndarray):
        X_test_tensor = torch.from_numpy(X_test).float()
    else:
        X_test_tensor = X_test.float()

    chart_assignments = None
    recon_atlas_cpu = None
    recon_std_cpu = None
    recon_ae_cpu = None
    z_geo_cpu = None
    k_chart_cpu = None
    k_code_cpu = None
    indices_out_cpu = None
    if do_results or do_latent:
        model_atlas.eval()
        if model_std is not None:
            model_std.eval()
        if model_ae is not None:
            model_ae.eval()

        if batch_size <= 0:
            batch_size = 256
        if batch_size > len(X_test_tensor):
            batch_size = len(X_test_tensor)

        test_dataset = TensorDataset(X_test_tensor)
        pin_memory = torch.device(device).type == "cuda"
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        )

        recon_atlas_parts: list[torch.Tensor] = []
        recon_std_parts: list[torch.Tensor] = []
        recon_ae_parts: list[torch.Tensor] = []
        k_chart_parts: list[torch.Tensor] = []
        k_code_parts: list[torch.Tensor] = []
        indices_out_parts: list[torch.Tensor] = []
        z_geo_parts: list[torch.Tensor] = []

        with torch.no_grad():
            for (batch_X,) in test_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                (
                    K_chart,
                    K_code,
                    _z_n,
                    z_tex,
                    _enc_w,
                    z_geo,
                    _vq_loss,
                    indices_out,
                    _z_n_all,
                    _c_bar,
                ) = model_atlas.encoder(batch_X)
                recon_atlas, _ = model_atlas.decoder(z_geo, z_tex, chart_index=None)

                recon_atlas_parts.append(recon_atlas.cpu())
                k_chart_parts.append(K_chart.cpu())
                k_code_parts.append(K_code.cpu())
                indices_out_parts.append(indices_out.cpu())
                z_geo_parts.append(z_geo.cpu())

                if model_std is not None and do_results:
                    recon_std, _, _ = model_std(batch_X)
                    recon_std_parts.append(recon_std.cpu())
                if model_ae is not None and do_results:
                    recon_ae, _ = model_ae(batch_X)
                    recon_ae_parts.append(recon_ae.cpu())

        recon_atlas_cpu = torch.cat(recon_atlas_parts) if recon_atlas_parts else None
        k_chart_cpu = torch.cat(k_chart_parts) if k_chart_parts else None
        k_code_cpu = torch.cat(k_code_parts) if k_code_parts else None
        indices_out_cpu = torch.cat(indices_out_parts) if indices_out_parts else None
        z_geo_cpu = torch.cat(z_geo_parts) if z_geo_parts else None

        if recon_std_parts:
            recon_std_cpu = torch.cat(recon_std_parts)
        if recon_ae_parts:
            recon_ae_cpu = torch.cat(recon_ae_parts)

        if k_chart_cpu is not None:
            chart_assignments = k_chart_cpu.numpy()

    dataset = config.get("dataset", "mnist")
    class_names, image_shape = _dataset_specs(dataset)

    if do_latent:
        _visualize_latent_images_3d(
            model_atlas,
            X_test_tensor,
            labels_test,
            class_names,
            save_path=latent_path,
            epoch=checkpoint.get("epoch"),
            jump_op=jump_op,
            image_shape=image_shape,
            precomputed={
                "K_chart": k_chart_cpu,
                "K_code": k_code_cpu,
                "z_geo": z_geo_cpu,
                "indices_out": indices_out_cpu,
                "recon": recon_atlas_cpu,
            },
        )

    if do_results:
        results = {
            "X": X_test_tensor.cpu(),
            "labels": labels_test,
            "chart_assignments": chart_assignments,
            "recon_ae": recon_ae_cpu,
            "recon_std": recon_std_cpu,
            "recon_atlas": recon_atlas_cpu,
            "z_geo": z_geo_cpu,
            "atlas_losses": metrics.get("atlas_losses", []),
            "std_losses": metrics.get("std_losses", []),
            "ae_losses": metrics.get("ae_losses", []),
            "ami_ae": metrics.get("ami_ae", 0.0),
            "ami_std": metrics.get("ami_std", 0.0),
            "ami_atlas": metrics.get("ami_atlas", 0.0),
            "mse_ae": metrics.get("mse_ae", 0.0),
            "mse_std": metrics.get("mse_std", 0.0),
            "mse_atlas": metrics.get("mse_atlas", 0.0),
            "std_perplexity": metrics.get("std_perplexity", 0.0),
            "atlas_perplexity": metrics.get("atlas_perplexity", 0.0),
            "sup_acc": metrics.get("sup_acc", 0.0),
            "cls_acc": metrics.get("cls_acc", None),
            "std_cls_acc": metrics.get("std_cls_acc", None),
            "ae_cls_acc": metrics.get("ae_cls_acc", None),
            "model_atlas": model_atlas,
        }
        visualize_results_images(
            results,
            class_names,
            save_path=results_path,
            image_shape=image_shape,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TopoEncoder 3D checkpoints")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to topoencoder_2d checkpoint (.pt)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="Directory containing topoencoder_2d checkpoints (.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory to save plots (defaults to checkpoint directory)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for analysis (cuda/cpu)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for analysis passes",
    )
    parser.add_argument(
        "--skip_latent",
        action="store_true",
        help="Skip latent visualization",
    )
    parser.add_argument(
        "--skip_results",
        action="store_true",
        help="Skip benchmark summary visualization",
    )
    parser.add_argument(
        "--only_missing",
        action="store_true",
        help="Only generate images that do not already exist",
    )

    args = parser.parse_args()

    checkpoints = _collect_checkpoints(
        args.checkpoint or None,
        args.checkpoint_dir or None,
    )
    output_dir = args.output_dir or os.path.dirname(checkpoints[0]) or "."
    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(checkpoints)} checkpoint(s). Writing plots to: {output_dir}")

    for checkpoint_path in checkpoints:
        _analyze_checkpoint(
            checkpoint_path,
            output_dir,
            args.device,
            args.skip_latent,
            args.skip_results,
            args.only_missing,
            args.batch_size,
        )


if __name__ == "__main__":
    main()
