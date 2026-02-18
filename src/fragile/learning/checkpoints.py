"""Checkpoint/benchmark I/O and utility functions for TopoEncoder."""

from dataclasses import asdict
import math

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import torch
from torch import nn, optim

from fragile.core.benchmarks import StandardVQ, VanillaAE
from fragile.learning.config import TopoEncoderConfig


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


def compute_perplexity(assignments: torch.Tensor, num_charts: int) -> float:
    """Compute chart usage perplexity from chart assignments."""
    if assignments.numel() == 0:
        return 0.0
    counts = torch.bincount(assignments, minlength=num_charts).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -(probs * torch.log(probs)).sum()
    return math.exp(entropy.item())


def _optimizer_state(
    optimizer: optim.Optimizer | dict[str, optim.Optimizer] | None,
) -> dict | None:
    if optimizer is None:
        return None
    if isinstance(optimizer, dict):
        return {name: opt.state_dict() for name, opt in optimizer.items()}
    return optimizer.state_dict()


def compute_param_norm(params: list[torch.Tensor]) -> float:
    total = 0.0
    for p in params:
        total += p.detach().pow(2).sum().item()
    return math.sqrt(total)


def compute_grad_norm(params: list[torch.Tensor]) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


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
    model_cifar_cov: nn.Module | None = None,
    model_cifar_std: nn.Module | None = None,
    supervised_loss: nn.Module | None = None,
    classifier_head: nn.Module | None = None,
    classifier_std: nn.Module | None = None,
    classifier_ae: nn.Module | None = None,
    optimizer_atlas: optim.Optimizer | None = None,
    optimizer_std: optim.Optimizer | None = None,
    optimizer_ae: optim.Optimizer | None = None,
    optimizer_cifar_cov: optim.Optimizer | None = None,
    optimizer_cifar_std: optim.Optimizer | None = None,
    optimizer_classifier: optim.Optimizer | None = None,
    optimizer_classifier_std: optim.Optimizer | None = None,
    optimizer_classifier_ae: optim.Optimizer | None = None,
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
            "cifar_cov": _state_dict_cpu(model_cifar_cov),
            "cifar_std": _state_dict_cpu(model_cifar_std),
        },
        "optim": {
            "atlas": _optimizer_state(optimizer_atlas),
            "std": _optimizer_state(optimizer_std),
            "ae": _optimizer_state(optimizer_ae),
            "cifar_cov": _optimizer_state(optimizer_cifar_cov),
            "cifar_std": _optimizer_state(optimizer_cifar_std),
            "classifier": _optimizer_state(optimizer_classifier),
            "classifier_std": _optimizer_state(optimizer_classifier_std),
            "classifier_ae": _optimizer_state(optimizer_classifier_ae),
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
            "std": _optimizer_state(optimizer_std),
            "ae": _optimizer_state(optimizer_ae),
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


def benchmarks_compatible(bench_config: dict, config: TopoEncoderConfig) -> bool:
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


def _move_optimizer_state(
    optimizer: optim.Optimizer | dict[str, optim.Optimizer],
    device: torch.device,
) -> None:
    if isinstance(optimizer, dict):
        for opt in optimizer.values():
            _move_optimizer_state(opt, device)
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_optimizer_state(
    optimizer: optim.Optimizer | dict[str, optim.Optimizer] | None,
    state: dict | None,
    device: torch.device,
) -> None:
    if optimizer is None or state is None:
        return
    if isinstance(optimizer, dict):
        if isinstance(state, dict) and "state" in state and "param_groups" in state:
            if len(optimizer) == 1:
                opt = next(iter(optimizer.values()))
                opt.load_state_dict(state)
                _move_optimizer_state(opt, device)
            else:
                print(
                    "  Optimizer state mismatch: single state for multiple optimizers; skipping."
                )
            return
        if isinstance(state, dict):
            for name, opt in optimizer.items():
                opt_state = state.get(name)
                if opt_state is not None:
                    opt.load_state_dict(opt_state)
                    _move_optimizer_state(opt, device)
        elif len(optimizer) == 1:
            opt = next(iter(optimizer.values()))
            opt.load_state_dict(state)
            _move_optimizer_state(opt, device)
        else:
            print("  Optimizer state mismatch: single state for multiple optimizers; skipping.")
        return
    if isinstance(state, dict):
        state = state.get("all")
        if state is None:
            print("  Optimizer state mismatch: multi-state for single optimizer; skipping.")
            return
    optimizer.load_state_dict(state)
    _move_optimizer_state(optimizer, device)


def compute_ami(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute Adjusted Mutual Information score."""
    return float(adjusted_mutual_info_score(labels_true, labels_pred))
