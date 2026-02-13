"""Vectorized companion-pair spin-2 tensor correlators with momentum projection.

This module implements a traceless symmetric spin-2 operator from companion pairs:

    O_i^{ab}(t) = Re(c_i^† c_j) * Q^{ab}(dx_ij)

where Q^{ab} is represented in a 5-component real basis (d=3):
    - q_xy
    - q_xz
    - q_yz
    - q_xx_minus_yy / sqrt(2)
    - q_2zz_minus_xx_minus_yy / sqrt(6)

The local per-walker operator is projected to momentum modes along one axis:

    O_{n,alpha}(t) = sum_i O_{i,alpha}(t) exp(-i k_n x_i)

with k_n = 2π n / L. Correlators are computed in batched FFT form for:
    - each momentum n and component alpha
    - contracted spin-2 channel sum_alpha C_{n,alpha}(Δt)

Bootstrap error estimates are computed in batched form over the time axis.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.qft.aggregation import compute_color_states_batch, estimate_ell0
from fragile.fractalai.qft.baryon_triplet_channels import (
    _resolve_3d_dims,
    _resolve_frame_indices,
    _safe_gather_2d,
    _safe_gather_3d,
)
from fragile.fractalai.qft.correlator_channels import _fft_correlator_batched
from fragile.fractalai.qft.meson_phase_channels import (
    build_companion_pair_indices,
    PAIR_SELECTION_MODES,
)
from fragile.fractalai.qft.radial_channels import _apply_pbc_diff_torch, _slice_bounds


TENSOR_COMPONENT_LABELS = (
    "q_xy",
    "q_xz",
    "q_yz",
    "q_xx_minus_yy",
    "q_2zz_minus_xx_minus_yy",
)


@dataclass
class TensorMomentumCorrelatorConfig:
    """Configuration for companion-pair tensor momentum correlators."""

    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    mc_time_index: int | None = None
    max_lag: int = 80
    use_connected: bool = True
    h_eff: float = 1.0
    mass: float = 1.0
    ell0: float | None = None
    color_dims: tuple[int, int, int] | None = None
    position_dims: tuple[int, int, int] | None = None
    pair_selection: str = "both"
    eps: float = 1e-12
    momentum_axis: int = 0
    momentum_mode_max: int = 4
    projection_length: float | None = None
    compute_bootstrap_errors: bool = False
    n_bootstrap: int = 100


@dataclass
class TensorMomentumCorrelatorOutput:
    """Tensor momentum correlator output and diagnostics."""

    component_labels: tuple[str, ...]
    frame_indices: list[int]
    pair_selection: str
    component_series: Tensor
    component_counts_per_frame: Tensor
    n_valid_source_pairs: int
    momentum_modes: Tensor
    momentum_correlator: Tensor
    momentum_correlator_raw: Tensor
    momentum_correlator_connected: Tensor
    momentum_correlator_err: Tensor | None
    momentum_contracted_correlator: Tensor
    momentum_contracted_correlator_raw: Tensor
    momentum_contracted_correlator_connected: Tensor
    momentum_contracted_correlator_err: Tensor | None
    momentum_operator_cos_series: Tensor
    momentum_operator_sin_series: Tensor
    momentum_axis: int
    momentum_length_scale: float
    momentum_valid_frames: int


def _safe_gather_pairs_2d(values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Safely gather values[:, idx] for indices [T,N,P]."""
    if values.ndim != 2 or indices.ndim != 3:
        raise ValueError(
            f"_safe_gather_pairs_2d expects values [T,N] and indices [T,N,P], got "
            f"{tuple(values.shape)} and {tuple(indices.shape)}."
        )
    t, n, p = indices.shape
    idx_flat = indices.reshape(t, n * p)
    gathered_flat, in_range_flat = _safe_gather_2d(values, idx_flat)
    return gathered_flat.reshape(t, n, p), in_range_flat.reshape(t, n, p)


def _safe_gather_pairs_3d(values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Safely gather values[:, idx, :] for indices [T,N,P]."""
    if values.ndim != 3 or indices.ndim != 3:
        raise ValueError(
            f"_safe_gather_pairs_3d expects values [T,N,C] and indices [T,N,P], got "
            f"{tuple(values.shape)} and {tuple(indices.shape)}."
        )
    t, n, p = indices.shape
    c = values.shape[-1]
    idx_flat = indices.reshape(t, n * p)
    gathered_flat, in_range_flat = _safe_gather_3d(values, idx_flat)
    return (
        gathered_flat.reshape(t, n, p, c),
        in_range_flat.reshape(t, n, p),
    )


def _extract_axis_bounds(
    bounds: object | None,
    axis: int,
    *,
    device: torch.device | None = None,
) -> tuple[float | None, float | None]:
    """Extract (low, high) for one axis from TorchBounds or array-like bounds."""
    if bounds is None or axis < 0:
        return None, None

    try:
        if hasattr(bounds, "low") and hasattr(bounds, "high"):
            low_vec = torch.as_tensor(bounds.low, dtype=torch.float32, device=device).flatten()
            high_vec = torch.as_tensor(bounds.high, dtype=torch.float32, device=device).flatten()
            if axis < int(low_vec.numel()) and axis < int(high_vec.numel()):
                return float(low_vec[axis].item()), float(high_vec[axis].item())
            return None, None

        bounds_t = torch.as_tensor(bounds, dtype=torch.float32, device=device)
    except Exception:
        return None, None

    if bounds_t.ndim != 2:
        return None, None
    if bounds_t.shape[0] > axis and bounds_t.shape[1] >= 2:
        return float(bounds_t[axis, 0].item()), float(bounds_t[axis, 1].item())
    if bounds_t.shape[0] >= 2 and bounds_t.shape[1] > axis:
        return float(bounds_t[0, axis].item()), float(bounds_t[1, axis].item())
    return None, None


def _resolve_positive_length(
    *,
    positions_axis: Tensor,
    valid_walker: Tensor,
    box_length: float | None,
) -> float:
    """Resolve a positive projection length scale for Fourier modes."""
    if box_length is not None and float(box_length) > 0:
        return float(box_length)

    alive_pos = positions_axis[valid_walker]
    if alive_pos.numel() == 0:
        span = float((positions_axis.max() - positions_axis.min()).abs().item())
    else:
        span = float((alive_pos.max() - alive_pos.min()).abs().item())
    return max(span, 1.0)


def _traceless_tensor_components(dx: Tensor) -> Tensor:
    """Build the 5-component traceless symmetric tensor basis for dx[...,3]."""
    if dx.ndim < 1 or dx.shape[-1] != 3:
        raise ValueError(f"Expected dx[...,3], got {tuple(dx.shape)}.")

    x = dx[..., 0]
    y = dx[..., 1]
    z = dx[..., 2]
    inv_sqrt2 = float(1.0 / math.sqrt(2.0))
    inv_sqrt6 = float(1.0 / math.sqrt(6.0))

    return torch.stack(
        (
            x * y,
            x * z,
            y * z,
            (x * x - y * y) * inv_sqrt2,
            (2.0 * z * z - x * x - y * y) * inv_sqrt6,
        ),
        dim=-1,
    )


def _compute_local_tensor_components(
    *,
    color: Tensor,
    color_valid: Tensor,
    positions: Tensor,
    alive: Tensor,
    pair_indices: Tensor,
    structural_valid: Tensor,
    bounds: object | None,
    pbc: bool,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Compute local 5-component tensor operator per walker and frame."""
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color must have shape [T,N,3], got {tuple(color.shape)}.")
    if positions.shape != color.shape:
        raise ValueError(
            f"positions must have shape [T,N,3] aligned with color, got {tuple(positions.shape)}."
        )
    if color_valid.shape != color.shape[:2]:
        raise ValueError(f"color_valid must have shape [T,N], got {tuple(color_valid.shape)}.")
    if alive.shape != color.shape[:2]:
        raise ValueError(f"alive must have shape [T,N], got {tuple(alive.shape)}.")
    if pair_indices.shape[:2] != color.shape[:2]:
        raise ValueError(
            "pair_indices must have shape [T,N,P] aligned with color, got "
            f"{tuple(pair_indices.shape)}."
        )
    if structural_valid.shape != pair_indices.shape:
        raise ValueError(
            "structural_valid must match pair_indices shape, got "
            f"{tuple(structural_valid.shape)} vs {tuple(pair_indices.shape)}."
        )

    color_j, in_range = _safe_gather_pairs_3d(color, pair_indices)
    alive_j, _ = _safe_gather_pairs_2d(alive, pair_indices)
    valid_j, _ = _safe_gather_pairs_2d(color_valid, pair_indices)
    pos_j, _ = _safe_gather_pairs_3d(positions, pair_indices)

    color_i = color.unsqueeze(2).expand_as(color_j)
    pos_i = positions.unsqueeze(2).expand_as(pos_j)
    inner = (torch.conj(color_i) * color_j).sum(dim=-1).real.float()
    dx = (pos_j - pos_i).float()
    if pbc and bounds is not None:
        dx = _apply_pbc_diff_torch(dx, bounds)

    finite_inner = torch.isfinite(inner)
    finite_dx = torch.isfinite(dx).all(dim=-1)
    valid = (
        structural_valid
        & in_range
        & alive.unsqueeze(-1)
        & alive_j
        & color_valid.unsqueeze(-1)
        & valid_j
        & finite_inner
        & finite_dx
    )
    if eps > 0:
        valid = valid & (inner.abs() > float(eps))

    components_pair = inner.unsqueeze(-1) * _traceless_tensor_components(dx)  # [T,N,P,5]
    components_pair = torch.where(
        valid.unsqueeze(-1),
        components_pair,
        torch.zeros_like(components_pair),
    )

    pair_weights = valid.to(dtype=torch.float32)
    pair_count_walker = pair_weights.sum(dim=2)  # [T,N]
    valid_walker = pair_count_walker > 0

    local_components = torch.zeros(
        (*pair_count_walker.shape, 5),
        dtype=torch.float32,
        device=color.device,
    )
    if torch.any(valid_walker):
        sums = components_pair.sum(dim=2)  # [T,N,5]
        local_components[valid_walker] = sums[valid_walker] / pair_count_walker[
            valid_walker
        ].unsqueeze(-1).clamp(min=1.0)

    component_counts_per_frame = valid_walker.sum(dim=1).to(torch.int64)
    n_valid_source_pairs = int(valid.sum().item())
    return local_components, valid_walker, component_counts_per_frame, n_valid_source_pairs


def _compute_momentum_projected_tensor_correlators(
    *,
    local_components: Tensor,
    valid_walker: Tensor,
    positions_axis: Tensor,
    max_lag: int,
    use_connected: bool,
    mode_max: int,
    box_length: float,
    compute_bootstrap_errors: bool,
    n_bootstrap: int,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
    int,
]:
    """Compute batched momentum-projected tensor correlators."""
    if local_components.ndim != 3 or local_components.shape[-1] != 5:
        raise ValueError(
            f"local_components must have shape [T,N,5], got {tuple(local_components.shape)}."
        )
    if valid_walker.shape != local_components.shape[:2]:
        raise ValueError(
            "valid_walker must have shape [T,N] aligned with local_components, got "
            f"{tuple(valid_walker.shape)}."
        )
    if positions_axis.shape != local_components.shape[:2]:
        raise ValueError(
            "positions_axis must have shape [T,N] aligned with local_components, got "
            f"{tuple(positions_axis.shape)}."
        )
    if box_length <= 0:
        msg = "box_length must be positive."
        raise ValueError(msg)

    t_total, _, n_comp = local_components.shape
    n_modes = max(0, int(mode_max)) + 1
    device = local_components.device

    modes = torch.arange(n_modes, device=device, dtype=torch.float32)
    k_values = (2.0 * torch.pi / float(box_length)) * modes

    phase_arg = k_values[:, None, None] * positions_axis[None, :, :].float()  # [M,T,N]
    cos_phase = torch.cos(phase_arg)
    sin_phase = torch.sin(phase_arg)

    weights = valid_walker.to(dtype=torch.float32)
    counts_t = weights.sum(dim=1)
    valid_t = counts_t > 0
    n_valid_frames = int(valid_t.sum().item())

    op_cos = torch.zeros((n_modes, n_comp, t_total), dtype=torch.float32, device=device)
    op_sin = torch.zeros((n_modes, n_comp, t_total), dtype=torch.float32, device=device)
    if n_valid_frames > 0:
        weighted_local = local_components * weights.unsqueeze(-1)  # [T,N,5]
        cos_num = torch.einsum("tnc,mtn->mct", weighted_local, cos_phase)
        sin_num = torch.einsum("tnc,mtn->mct", weighted_local, sin_phase)
        denom = counts_t[valid_t].to(dtype=torch.float32).clamp(min=1.0)
        op_cos[:, :, valid_t] = cos_num[:, :, valid_t] / denom.unsqueeze(0).unsqueeze(0)
        op_sin[:, :, valid_t] = sin_num[:, :, valid_t] / denom.unsqueeze(0).unsqueeze(0)

    n_lags = int(max(0, max_lag)) + 1
    if n_valid_frames == 0:
        zeros_comp = torch.zeros((n_modes, n_comp, n_lags), dtype=torch.float32, device=device)
        zeros_contract = torch.zeros((n_modes, n_lags), dtype=torch.float32, device=device)
        return (
            modes,
            zeros_comp,
            zeros_comp.clone(),
            zeros_comp.clone(),
            None,
            zeros_contract,
            zeros_contract.clone(),
            zeros_contract.clone(),
            op_cos,
            op_sin,
            None,
            0,
        )

    series_cos = op_cos[:, :, valid_t]  # [M,5,Tv]
    series_sin = op_sin[:, :, valid_t]  # [M,5,Tv]
    t_len = int(series_cos.shape[-1])
    flat_cos = series_cos.reshape(n_modes * n_comp, t_len)
    flat_sin = series_sin.reshape(n_modes * n_comp, t_len)

    corr_raw_flat = _fft_correlator_batched(
        flat_cos,
        max_lag=int(max_lag),
        use_connected=False,
    ) + _fft_correlator_batched(
        flat_sin,
        max_lag=int(max_lag),
        use_connected=False,
    )
    corr_connected_flat = _fft_correlator_batched(
        flat_cos,
        max_lag=int(max_lag),
        use_connected=True,
    ) + _fft_correlator_batched(
        flat_sin,
        max_lag=int(max_lag),
        use_connected=True,
    )

    corr_raw = corr_raw_flat.reshape(n_modes, n_comp, -1)
    corr_connected = corr_connected_flat.reshape(n_modes, n_comp, -1)
    corr = corr_connected if use_connected else corr_raw

    corr_err: Tensor | None = None
    corr_contract_err: Tensor | None = None
    if bool(compute_bootstrap_errors):
        n_boot = max(1, int(n_bootstrap))
        idx = torch.randint(0, t_len, (n_boot, t_len), device=device)
        idx_expand = idx[:, None, None, :].expand(n_boot, n_modes, n_comp, t_len)

        boot_cos = torch.gather(
            series_cos.unsqueeze(0).expand(n_boot, n_modes, n_comp, t_len),
            dim=3,
            index=idx_expand,
        )
        boot_sin = torch.gather(
            series_sin.unsqueeze(0).expand(n_boot, n_modes, n_comp, t_len),
            dim=3,
            index=idx_expand,
        )

        boot_cos_flat = boot_cos.reshape(n_boot * n_modes * n_comp, t_len)
        boot_sin_flat = boot_sin.reshape(n_boot * n_modes * n_comp, t_len)
        boot_raw = (
            _fft_correlator_batched(boot_cos_flat, max_lag=int(max_lag), use_connected=False)
            + _fft_correlator_batched(boot_sin_flat, max_lag=int(max_lag), use_connected=False)
        ).reshape(n_boot, n_modes, n_comp, -1)
        boot_connected = (
            _fft_correlator_batched(boot_cos_flat, max_lag=int(max_lag), use_connected=True)
            + _fft_correlator_batched(boot_sin_flat, max_lag=int(max_lag), use_connected=True)
        ).reshape(n_boot, n_modes, n_comp, -1)
        boot_corr = boot_connected if use_connected else boot_raw
        corr_err = boot_corr.std(dim=0)
        corr_contract_err = boot_corr.sum(dim=2).std(dim=0)

    corr_contract = corr.sum(dim=1)
    corr_contract_raw = corr_raw.sum(dim=1)
    corr_contract_connected = corr_connected.sum(dim=1)

    return (
        modes,
        corr.float(),
        corr_raw.float(),
        corr_connected.float(),
        corr_err.float() if corr_err is not None else None,
        corr_contract.float(),
        corr_contract_raw.float(),
        corr_contract_connected.float(),
        op_cos,
        op_sin,
        corr_contract_err.float() if corr_contract_err is not None else None,
        n_valid_frames,
    )


def compute_tensor_momentum_correlator_from_color_positions(
    *,
    color: Tensor,
    color_valid: Tensor,
    positions: Tensor,
    positions_axis: Tensor,
    alive: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    max_lag: int,
    use_connected: bool,
    pair_selection: str,
    eps: float,
    momentum_mode_max: int,
    projection_length: float | None,
    bounds: object | None,
    pbc: bool,
    compute_bootstrap_errors: bool,
    n_bootstrap: int,
    frame_indices: list[int] | None = None,
    momentum_axis: int = 0,
) -> TensorMomentumCorrelatorOutput:
    """Compute momentum-projected companion-pair tensor correlators."""
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color must have shape [T,N,3], got {tuple(color.shape)}.")
    if positions.shape != color.shape:
        raise ValueError(
            f"positions must have shape [T,N,3] aligned with color, got {tuple(positions.shape)}."
        )
    if positions_axis.shape != color.shape[:2]:
        raise ValueError(
            "positions_axis must have shape [T,N] aligned with color, got "
            f"{tuple(positions_axis.shape)}."
        )
    if color_valid.shape != color.shape[:2]:
        raise ValueError(f"color_valid must have shape [T,N], got {tuple(color_valid.shape)}.")
    if alive.shape != color.shape[:2]:
        raise ValueError(f"alive must have shape [T,N], got {tuple(alive.shape)}.")
    if companions_distance.shape != color.shape[:2] or companions_clone.shape != color.shape[:2]:
        msg = "companion arrays must have shape [T,N] aligned with color."
        raise ValueError(msg)

    mode = str(pair_selection).strip().lower()
    if mode not in PAIR_SELECTION_MODES:
        raise ValueError(f"pair_selection must be one of {PAIR_SELECTION_MODES}.")

    pair_indices, structural_valid = build_companion_pair_indices(
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        pair_selection=mode,
    )
    (
        local_components,
        valid_walker,
        component_counts_per_frame,
        n_valid_source_pairs,
    ) = _compute_local_tensor_components(
        color=color,
        color_valid=color_valid,
        positions=positions,
        alive=alive,
        pair_indices=pair_indices,
        structural_valid=structural_valid,
        bounds=bounds,
        pbc=bool(pbc),
        eps=float(max(eps, 0.0)),
    )

    resolved_length = _resolve_positive_length(
        positions_axis=positions_axis,
        valid_walker=valid_walker,
        box_length=projection_length,
    )

    (
        momentum_modes,
        momentum_correlator,
        momentum_correlator_raw,
        momentum_correlator_connected,
        momentum_correlator_err,
        momentum_contracted_correlator,
        momentum_contracted_correlator_raw,
        momentum_contracted_correlator_connected,
        momentum_operator_cos_series,
        momentum_operator_sin_series,
        momentum_contracted_correlator_err,
        momentum_valid_frames,
    ) = _compute_momentum_projected_tensor_correlators(
        local_components=local_components,
        valid_walker=valid_walker,
        positions_axis=positions_axis,
        max_lag=int(max_lag),
        use_connected=bool(use_connected),
        mode_max=int(momentum_mode_max),
        box_length=float(resolved_length),
        compute_bootstrap_errors=bool(compute_bootstrap_errors),
        n_bootstrap=int(n_bootstrap),
    )

    # Per-frame averaged component diagnostics.
    component_series = torch.zeros(
        local_components.shape[0],
        local_components.shape[-1],
        dtype=torch.float32,
        device=local_components.device,
    )
    valid_t = component_counts_per_frame > 0
    if torch.any(valid_t):
        sums = (local_components * valid_walker.unsqueeze(-1).to(local_components.dtype)).sum(
            dim=1
        )
        component_series[valid_t] = sums[valid_t] / component_counts_per_frame[valid_t].to(
            local_components.dtype
        ).unsqueeze(-1).clamp(min=1.0)

    return TensorMomentumCorrelatorOutput(
        component_labels=TENSOR_COMPONENT_LABELS,
        frame_indices=list(range(color.shape[0])) if frame_indices is None else frame_indices,
        pair_selection=mode,
        component_series=component_series.float(),
        component_counts_per_frame=component_counts_per_frame,
        n_valid_source_pairs=int(n_valid_source_pairs),
        momentum_modes=momentum_modes.float(),
        momentum_correlator=momentum_correlator.float(),
        momentum_correlator_raw=momentum_correlator_raw.float(),
        momentum_correlator_connected=momentum_correlator_connected.float(),
        momentum_correlator_err=momentum_correlator_err.float()
        if momentum_correlator_err is not None
        else None,
        momentum_contracted_correlator=momentum_contracted_correlator.float(),
        momentum_contracted_correlator_raw=momentum_contracted_correlator_raw.float(),
        momentum_contracted_correlator_connected=momentum_contracted_correlator_connected.float(),
        momentum_contracted_correlator_err=(
            momentum_contracted_correlator_err.float()
            if momentum_contracted_correlator_err is not None
            else None
        ),
        momentum_operator_cos_series=momentum_operator_cos_series.float(),
        momentum_operator_sin_series=momentum_operator_sin_series.float(),
        momentum_axis=int(momentum_axis),
        momentum_length_scale=float(resolved_length),
        momentum_valid_frames=int(momentum_valid_frames),
    )


def compute_companion_tensor_momentum_correlator(
    history: RunHistory,
    config: TensorMomentumCorrelatorConfig | None = None,
) -> TensorMomentumCorrelatorOutput:
    """Compute vectorized companion-pair tensor momentum correlators from RunHistory."""
    config = config or TensorMomentumCorrelatorConfig()
    frame_indices = _resolve_frame_indices(
        history=history,
        warmup_fraction=float(config.warmup_fraction),
        end_fraction=float(config.end_fraction),
        mc_time_index=config.mc_time_index,
    )
    if not frame_indices:
        n_lags = int(max(0, config.max_lag)) + 1
        n_modes = int(max(0, config.momentum_mode_max)) + 1
        device = history.v_before_clone.device
        zeros_comp = torch.zeros((n_modes, 5, n_lags), dtype=torch.float32, device=device)
        zeros_contract = torch.zeros((n_modes, n_lags), dtype=torch.float32, device=device)
        return TensorMomentumCorrelatorOutput(
            component_labels=TENSOR_COMPONENT_LABELS,
            frame_indices=[],
            pair_selection=str(config.pair_selection).strip().lower(),
            component_series=torch.zeros((0, 5), dtype=torch.float32, device=device),
            component_counts_per_frame=torch.zeros(0, dtype=torch.int64, device=device),
            n_valid_source_pairs=0,
            momentum_modes=torch.arange(n_modes, dtype=torch.float32, device=device),
            momentum_correlator=zeros_comp.clone(),
            momentum_correlator_raw=zeros_comp.clone(),
            momentum_correlator_connected=zeros_comp.clone(),
            momentum_correlator_err=None,
            momentum_contracted_correlator=zeros_contract.clone(),
            momentum_contracted_correlator_raw=zeros_contract.clone(),
            momentum_contracted_correlator_connected=zeros_contract.clone(),
            momentum_contracted_correlator_err=None,
            momentum_operator_cos_series=torch.zeros(
                (n_modes, 5, 0), dtype=torch.float32, device=device
            ),
            momentum_operator_sin_series=torch.zeros(
                (n_modes, 5, 0), dtype=torch.float32, device=device
            ),
            momentum_axis=int(config.momentum_axis),
            momentum_length_scale=float(max(config.projection_length or 1.0, 1e-6)),
            momentum_valid_frames=0,
        )

    start_idx = frame_indices[0]
    end_idx = frame_indices[-1] + 1
    h_eff = float(max(config.h_eff, 1e-8))
    mass = float(max(config.mass, 1e-8))
    ell0 = float(config.ell0) if config.ell0 is not None else float(estimate_ell0(history))
    if ell0 <= 0:
        msg = "ell0 must be positive."
        raise ValueError(msg)

    color, color_valid = compute_color_states_batch(
        history=history,
        start_idx=start_idx,
        h_eff=h_eff,
        mass=mass,
        ell0=ell0,
        end_idx=end_idx,
    )
    color_dims = _resolve_3d_dims(color.shape[-1], config.color_dims, "color_dims")
    pos_dims = _resolve_3d_dims(history.d, config.position_dims, "position_dims")
    color = color[:, :, list(color_dims)]

    device = color.device
    positions = history.x_before_clone[start_idx:end_idx, :, list(pos_dims)].to(
        device=device,
        dtype=torch.float32,
    )
    alive = torch.as_tensor(
        history.alive_mask[start_idx - 1 : end_idx - 1],
        dtype=torch.bool,
        device=device,
    )
    companions_distance = torch.as_tensor(
        history.companions_distance[start_idx - 1 : end_idx - 1],
        dtype=torch.long,
        device=device,
    )
    companions_clone = torch.as_tensor(
        history.companions_clone[start_idx - 1 : end_idx - 1],
        dtype=torch.long,
        device=device,
    )

    momentum_axis = int(config.momentum_axis)
    if momentum_axis < 0 or momentum_axis >= int(history.d):
        raise ValueError(
            f"momentum_axis={momentum_axis} out of range for history.d={history.d}. "
            f"Expected 0..{history.d - 1}."
        )
    positions_axis = history.x_before_clone[start_idx:end_idx, :, momentum_axis].to(
        device=device,
        dtype=torch.float32,
    )

    low, high = _extract_axis_bounds(history.bounds, momentum_axis, device=device)
    projection_length = config.projection_length
    has_finite_bounds = (
        low is not None and high is not None and math.isfinite(low) and math.isfinite(high)
    )
    if projection_length is None and has_finite_bounds and high > low:
        projection_length = float(high - low)

    bounds = _slice_bounds(history.bounds, list(pos_dims))
    return compute_tensor_momentum_correlator_from_color_positions(
        color=color,
        color_valid=color_valid.to(dtype=torch.bool, device=device),
        positions=positions,
        positions_axis=positions_axis,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=int(config.max_lag),
        use_connected=bool(config.use_connected),
        pair_selection=str(config.pair_selection),
        eps=float(max(config.eps, 0.0)),
        momentum_mode_max=int(config.momentum_mode_max),
        projection_length=projection_length,
        bounds=bounds,
        pbc=bool(history.pbc),
        compute_bootstrap_errors=bool(config.compute_bootstrap_errors),
        n_bootstrap=int(config.n_bootstrap),
        frame_indices=frame_indices,
        momentum_axis=momentum_axis,
    )
