"""Wilson flow (gradient flow) analysis for coupling diagnostics.

Implements diffusive smoothing of color states on the companion graph,
measuring the plaquette action density E(t) at each flow step.  Scale
extraction via t0 (where t^2<E> = reference) and w0 (where d/dt[t^2<E>]
= reference) provides a meaningful physical scale.

All functions are pure computation -- no UI dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from fragile.physics.new_channels.glueball_color_channels import (
    _compute_color_plaquette_for_triplets,
    _glueball_observable_from_plaquette,
)
from fragile.physics.qft_utils import safe_gather_2d, safe_gather_3d


@dataclass
class WilsonFlowConfig:
    """Configuration for Wilson flow analysis."""

    n_steps: int = 100
    step_size: float = 0.02
    topology: str = "both"
    t0_reference: float = 0.3
    w0_reference: float = 0.3
    eps: float = 1e-12
    operator_mode: str = "action_re_plaquette"


@dataclass
class WilsonFlowOutput:
    """Wilson flow measurement results."""

    flow_times: Tensor
    action_density: Tensor
    action_density_per_frame: Tensor
    t2_action: Tensor
    dt2_action: Tensor
    dt2_action_times: Tensor
    t0: float
    w0: float
    sqrt_8t0: float
    n_valid_walkers_per_frame: Tensor
    config: WilsonFlowConfig


def _diffuse_color_step(
    color: Tensor,
    color_valid: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    step_size: float,
    topology: str,
) -> Tensor:
    """Single diffusion step on the companion graph.

    Averages neighbor colors and blends with current color via convex
    combination, then re-normalizes to unit norm.

    Args:
        color: Complex color states [T, N, 3].
        color_valid: Validity mask [T, N].
        companions_distance: Distance companion indices [T, N].
        companions_clone: Clone companion indices [T, N].
        step_size: Diffusion blending weight (0 < alpha < 1).
        topology: "distance", "clone", or "both".

    Returns:
        Diffused color states [T, N, 3].
    """
    device = color.device
    neighbor_sum = torch.zeros_like(color)
    neighbor_count = torch.zeros(color.shape[:2], device=device, dtype=torch.float32)

    if topology in ("distance", "both"):
        c_d, in_d = safe_gather_3d(color, companions_distance)
        v_d, _ = safe_gather_2d(color_valid, companions_distance)
        # Non-self check
        n = color.shape[1]
        anchor = torch.arange(n, device=device, dtype=torch.long).view(1, n)
        not_self_d = companions_distance != anchor
        mask_d = in_d & v_d & not_self_d
        neighbor_sum = neighbor_sum + torch.where(mask_d.unsqueeze(-1), c_d, torch.zeros_like(c_d))
        neighbor_count = neighbor_count + mask_d.float()

    if topology in ("clone", "both"):
        c_c, in_c = safe_gather_3d(color, companions_clone)
        v_c, _ = safe_gather_2d(color_valid, companions_clone)
        n = color.shape[1]
        anchor = torch.arange(n, device=device, dtype=torch.long).view(1, n)
        not_self_c = companions_clone != anchor
        mask_c = in_c & v_c & not_self_c
        neighbor_sum = neighbor_sum + torch.where(mask_c.unsqueeze(-1), c_c, torch.zeros_like(c_c))
        neighbor_count = neighbor_count + mask_c.float()

    has_neighbors = neighbor_count > 0
    neighbor_avg = torch.where(
        has_neighbors.unsqueeze(-1),
        neighbor_sum / neighbor_count.clamp(min=1.0).unsqueeze(-1),
        color,
    )

    alpha = float(step_size)
    c_new = (1.0 - alpha) * color + alpha * neighbor_avg
    norm = torch.linalg.vector_norm(c_new, dim=-1, keepdim=True).clamp(min=1e-12)
    return c_new / norm


def _measure_action_density(
    color: Tensor,
    color_valid: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    eps: float,
    operator_mode: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Measure plaquette action density E from color states.

    Returns:
        mean_E: Scalar mean action density.
        per_frame_E: Per-frame action density [T].
        valid_per_frame: Valid walker count per frame [T].
    """
    pi, valid = _compute_color_plaquette_for_triplets(
        color=color,
        color_valid=color_valid,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        eps=eps,
    )
    obs = _glueball_observable_from_plaquette(pi, operator_mode=operator_mode)

    t_len = color.shape[0]
    device = color.device
    valid_count = valid.sum(dim=1).float()
    has_valid = valid_count > 0

    per_frame = torch.zeros(t_len, dtype=torch.float32, device=device)
    weighted_sum = (obs * valid.float()).sum(dim=1)
    per_frame = torch.where(has_valid, weighted_sum / valid_count.clamp(min=1.0), per_frame)

    n_valid_total = has_valid.sum()
    if n_valid_total > 0:
        mean_e = per_frame[has_valid].mean()
    else:
        mean_e = torch.zeros((), dtype=torch.float32, device=device)

    return mean_e, per_frame, valid_count


def _interpolate_crossing(x: Tensor, y: Tensor, target: float) -> float:
    """Linear interpolation to find x where y crosses target.

    Returns nan if no crossing is found.
    """
    if x.numel() < 2:
        return float("nan")
    y_shifted = y - target
    sign_changes = y_shifted[:-1] * y_shifted[1:] <= 0
    indices = torch.where(sign_changes)[0]
    if indices.numel() == 0:
        return float("nan")
    idx = int(indices[0].item())
    y0 = float(y_shifted[idx].item())
    y1 = float(y_shifted[idx + 1].item())
    x0 = float(x[idx].item())
    x1 = float(x[idx + 1].item())
    denom = y1 - y0
    if abs(denom) < 1e-30:
        return 0.5 * (x0 + x1)
    frac = -y0 / denom
    return x0 + frac * (x1 - x0)


def compute_wilson_flow(
    color: Tensor,
    color_valid: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    config: WilsonFlowConfig | None = None,
) -> WilsonFlowOutput:
    """Compute Wilson flow by diffusing color states on the companion graph.

    Args:
        color: Complex color states [T, N, 3].
        color_valid: Validity mask [T, N].
        companions_distance: Distance companion indices [T, N].
        companions_clone: Clone companion indices [T, N].
        config: Flow configuration.

    Returns:
        WilsonFlowOutput with flow curves and extracted scales.
    """
    cfg = config or WilsonFlowConfig()
    n_steps = max(1, int(cfg.n_steps))
    step_size = float(cfg.step_size)
    topology = str(cfg.topology).strip().lower()
    if topology not in ("distance", "clone", "both"):
        raise ValueError(f"topology must be 'distance', 'clone', or 'both', got {topology!r}.")

    device = color.device
    t_len = color.shape[0]

    flow_times_list = [0.0]
    action_density_list: list[Tensor] = []
    action_density_per_frame_list: list[Tensor] = []

    # Clone color for diffusion (don't mutate input)
    c = color.clone()

    # Measure at step 0
    mean_e, per_frame_e, valid_per_frame = _measure_action_density(
        c, color_valid, companions_distance, companions_clone, cfg.eps, cfg.operator_mode,
    )
    action_density_list.append(mean_e.detach())
    action_density_per_frame_list.append(per_frame_e.detach())
    n_valid_walkers_per_frame = valid_per_frame.detach()

    # Flow loop
    for step in range(1, n_steps + 1):
        c = _diffuse_color_step(
            c, color_valid, companions_distance, companions_clone, step_size, topology,
        )
        flow_times_list.append(step * step_size)
        mean_e, per_frame_e, _ = _measure_action_density(
            c, color_valid, companions_distance, companions_clone, cfg.eps, cfg.operator_mode,
        )
        action_density_list.append(mean_e.detach())
        action_density_per_frame_list.append(per_frame_e.detach())

    flow_times = torch.tensor(flow_times_list, dtype=torch.float32, device=device)
    action_density = torch.stack(action_density_list)
    action_density_per_frame = torch.stack(action_density_per_frame_list)  # [n_steps+1, T]

    # t^2 * <E(t)>
    t2_action = flow_times * flow_times * action_density

    # d/dt[t^2 * E] via forward finite differences
    dt = flow_times[1:] - flow_times[:-1]
    dt2_action = (t2_action[1:] - t2_action[:-1]) / dt.clamp(min=1e-30)
    dt2_action_times = 0.5 * (flow_times[1:] + flow_times[:-1])

    # Scale extraction
    t0 = _interpolate_crossing(flow_times, t2_action, cfg.t0_reference)
    w0 = _interpolate_crossing(dt2_action_times, dt2_action, cfg.w0_reference)
    import math
    sqrt_8t0 = math.sqrt(8.0 * t0) if math.isfinite(t0) and t0 > 0 else float("nan")

    return WilsonFlowOutput(
        flow_times=flow_times,
        action_density=action_density,
        action_density_per_frame=action_density_per_frame,
        t2_action=t2_action,
        dt2_action=dt2_action,
        dt2_action_times=dt2_action_times,
        t0=t0,
        w0=w0,
        sqrt_8t0=sqrt_8t0,
        n_valid_walkers_per_frame=n_valid_walkers_per_frame,
        config=cfg,
    )
