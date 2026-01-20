"""
Analyze TopoEncoder checkpoints and generate plots.

This script loads saved checkpoints from topoencoder_2d.py and produces
the latent visualization and benchmark summary figures without re-running
training. It supports a single checkpoint or a directory of checkpoints.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any

import numpy as np
import torch

from dataviz import visualize_latent_images, visualize_results_images
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


def _prepare_models(
    config: dict,
    state: dict,
    metrics: dict,
    bench_state: dict | None,
    bench_dims: dict | None,
    device: str,
) -> dict[str, Any]:
    model_atlas = TopoEncoderPrimitives(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        num_charts=config["num_charts"],
        codes_per_chart=config["codes_per_chart"],
    ).to(device)
    model_atlas.load_state_dict(state["atlas"])
    model_atlas.eval()

    model_std = None
    if state.get("std") is not None and not config.get("disable_vq", False):
        std_hidden_dim = metrics.get("std_hidden_dim", config["hidden_dim"])
        model_std = StandardVQ(
            input_dim=config["input_dim"],
            hidden_dim=std_hidden_dim,
            latent_dim=config["latent_dim"],
            num_codes=config["num_codes_standard"],
        ).to(device)
        model_std.load_state_dict(state["std"])
        model_std.eval()
    elif bench_state is not None and bench_state.get("std") is not None and not config.get(
        "disable_vq", False
    ):
        std_hidden_dim = (
            (bench_dims or {}).get("std_hidden_dim") or metrics.get("std_hidden_dim")
        ) or config["hidden_dim"]
        model_std = StandardVQ(
            input_dim=config["input_dim"],
            hidden_dim=int(std_hidden_dim),
            latent_dim=config["latent_dim"],
            num_codes=config["num_codes_standard"],
        ).to(device)
        model_std.load_state_dict(bench_state["std"])
        model_std.eval()

    model_ae = None
    if state.get("ae") is not None and not config.get("disable_ae", False):
        ae_hidden_dim = metrics.get("ae_hidden_dim", config["hidden_dim"])
        model_ae = VanillaAE(
            input_dim=config["input_dim"],
            hidden_dim=ae_hidden_dim,
            latent_dim=config["latent_dim"],
        ).to(device)
        model_ae.load_state_dict(state["ae"])
        model_ae.eval()
    elif bench_state is not None and bench_state.get("ae") is not None and not config.get(
        "disable_ae", False
    ):
        ae_hidden_dim = (
            (bench_dims or {}).get("ae_hidden_dim") or metrics.get("ae_hidden_dim")
        ) or config["hidden_dim"]
        model_ae = VanillaAE(
            input_dim=config["input_dim"],
            hidden_dim=int(ae_hidden_dim),
            latent_dim=config["latent_dim"],
        ).to(device)
        model_ae.load_state_dict(bench_state["ae"])
        model_ae.eval()

    jump_op = None
    if state.get("jump") is not None:
        jump_op = FactorizedJumpOperator(
            num_charts=config["num_charts"],
            latent_dim=config["latent_dim"],
            global_rank=config["jump_global_rank"],
        ).to(device)
        jump_op.load_state_dict(state["jump"])
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
    raise ValueError("Provide --checkpoint or --checkpoint_dir.")


def _analyze_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    device: str,
    skip_latent: bool,
    skip_results: bool,
    only_missing: bool,
) -> None:
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    latent_path = os.path.join(output_dir, f"{stem}_latent.png")
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

    X_test_device = X_test_tensor.to(device)

    chart_assignments = None
    recon_atlas_cpu = None
    recon_std_cpu = None
    recon_ae_cpu = None
    if do_results:
        with torch.no_grad():
            recon_atlas, _, _, _, K_chart, _z_geo, _z_n, _c_bar = model_atlas(
                X_test_device, use_hard_routing=False
            )
            chart_assignments = K_chart.cpu().numpy()
            recon_atlas_cpu = recon_atlas.cpu()

            if model_std is not None:
                recon_std, _, _ = model_std(X_test_device)
                recon_std_cpu = recon_std.cpu()

            if model_ae is not None:
                recon_ae, _ = model_ae(X_test_device)
                recon_ae_cpu = recon_ae.cpu()

    dataset = config.get("dataset", "mnist")
    class_names, image_shape = _dataset_specs(dataset)

    if do_latent:
        visualize_latent_images(
            model_atlas,
            X_test_device,
            labels_test,
            class_names,
            save_path=latent_path,
            epoch=checkpoint.get("epoch"),
            jump_op=jump_op,
            image_shape=image_shape,
        )

    if do_results:
        results = {
            "X": X_test_tensor.cpu(),
            "labels": labels_test,
            "chart_assignments": chart_assignments,
            "recon_ae": recon_ae_cpu,
            "recon_std": recon_std_cpu,
            "recon_atlas": recon_atlas_cpu,
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
        visualize_results_images(results, class_names, save_path=results_path, image_shape=image_shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TopoEncoder 2D checkpoints")
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
        args.checkpoint if args.checkpoint else None,
        args.checkpoint_dir if args.checkpoint_dir else None,
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
        )


if __name__ == "__main__":
    main()
