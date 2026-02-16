"""Companion-triplet glueball operators from color plaquettes.

Computes an SU(3)-inspired glueball observable from companion triplets
(i, j, k), where j=companions_distance[i], k=companions_clone[i]:

    Pi_i(t) = (c_i^dag c_j)(c_j^dag c_k)(c_k^dag c_i)

Glueball scalar operator at each source triplet:
- Re(Pi_i), or
- 1 - Re(Pi_i) (action-style form, configurable).

This module computes *operators only* -- no correlator computation or
RunHistory access.  All frame extraction and color-state preparation is
handled by :mod:`.preparation`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .baryon_operators import build_companion_triplets
from .config import GlueballOperatorConfig
from .preparation import _safe_gather_2d, _safe_gather_3d, PreparedChannelData


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_glueball_operator_mode(
    *,
    operator_mode: str | None,
    use_action_form: bool,
) -> str:
    """Resolve glueball operator mode with backward compatibility.

    If *operator_mode* is not set (None or blank), falls back to
    ``"action_re_plaquette"`` when *use_action_form* is True, else
    ``"re_plaquette"``.  The bare string ``"action"`` is normalised to
    ``"action_re_plaquette"`` for convenience.
    """
    if operator_mode is None or not str(operator_mode).strip():
        return "action_re_plaquette" if use_action_form else "re_plaquette"
    mode = str(operator_mode).strip().lower()
    if mode == "action":
        return "action_re_plaquette"
    return mode


def _glueball_observable_from_plaquette(pi: Tensor, *, operator_mode: str) -> Tensor:
    """Build scalar glueball observable from complex plaquette Pi."""
    mode = _resolve_glueball_operator_mode(operator_mode=operator_mode, use_action_form=False)
    if mode == "re_plaquette":
        return pi.real.float()
    if mode == "action_re_plaquette":
        return (1.0 - pi.real).float()
    if mode == "phase_action":
        phase = torch.angle(pi)
        return (1.0 - torch.cos(phase)).float()
    if mode == "phase_sin2":
        phase = torch.angle(pi)
        return torch.sin(phase).square().float()
    msg = (
        "Invalid glueball operator_mode. Expected one of "
        "{'re_plaquette','action_re_plaquette','phase_action','phase_sin2'}."
    )
    raise ValueError(msg)


def _compute_color_plaquette_for_triplets(
    color: Tensor,
    color_valid: Tensor,
    alive: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Compute Pi_i for companion triplets and a validity mask.

    Returns:
        pi: Complex color plaquette Pi_i [T, N].
        valid: Valid source/sink triplet mask [T, N].
    """
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color must have shape [T, N, 3], got {tuple(color.shape)}.")
    if color_valid.shape != color.shape[:2]:
        raise ValueError(f"color_valid must have shape [T, N], got {tuple(color_valid.shape)}.")
    if alive.shape != color.shape[:2]:
        raise ValueError(f"alive must have shape [T, N], got {tuple(alive.shape)}.")
    if companions_distance.shape != color.shape[:2] or companions_clone.shape != color.shape[:2]:
        msg = "companion arrays must have shape [T, N] aligned with color."
        raise ValueError(msg)

    _, companion_j, companion_k, structural_valid = build_companion_triplets(
        companions_distance=companions_distance,
        companions_clone=companions_clone,
    )

    color_j, in_j = _safe_gather_3d(color, companion_j)
    color_k, in_k = _safe_gather_3d(color, companion_k)
    alive_j, _ = _safe_gather_2d(alive, companion_j)
    alive_k, _ = _safe_gather_2d(alive, companion_k)
    valid_j, _ = _safe_gather_2d(color_valid, companion_j)
    valid_k, _ = _safe_gather_2d(color_valid, companion_k)

    z_ij = (torch.conj(color) * color_j).sum(dim=-1)
    z_jk = (torch.conj(color_j) * color_k).sum(dim=-1)
    z_ki = (torch.conj(color_k) * color).sum(dim=-1)
    pi = z_ij * z_jk * z_ki

    finite = torch.isfinite(pi.real) & torch.isfinite(pi.imag)
    valid = (
        structural_valid
        & in_j
        & in_k
        & alive
        & alive_j
        & alive_k
        & color_valid
        & valid_j
        & valid_k
        & finite
    )
    if eps > 0:
        valid = valid & (z_ij.abs() > eps) & (z_jk.abs() > eps) & (z_ki.abs() > eps)

    pi = torch.where(valid, pi, torch.zeros_like(pi))
    return pi, valid


def _resolve_positive_length(
    *,
    positions_axis: Tensor,
    alive: Tensor,
    box_length: float | None,
) -> float:
    """Resolve a positive projection length scale for Fourier modes."""
    if box_length is not None and float(box_length) > 0:
        return float(box_length)

    alive_pos = positions_axis[alive]
    if alive_pos.numel() == 0:
        span = float((positions_axis.max() - positions_axis.min()).abs().item())
    else:
        span = float((alive_pos.max() - alive_pos.min()).abs().item())
    return max(span, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_glueball_operators(
    data: PreparedChannelData,
    config: GlueballOperatorConfig,
) -> dict[str, Tensor]:
    """Compute glueball operator time-series from prepared channel data.

    Returns a dict:
        ``"glueball"`` -- per-frame averaged glueball observable [T].

    When ``config.use_momentum_projection`` is True, additional entries:
        ``"glueball_momentum_cos_{n}"`` -- cosine-projected operator [T] for mode *n*.
        ``"glueball_momentum_sin_{n}"`` -- sine-projected operator [T] for mode *n*.
    """
    color = data.color
    t_total = int(color.shape[0])
    device = data.device

    if t_total == 0:
        result: dict[str, Tensor] = {
            "glueball": torch.zeros(0, dtype=torch.float32, device=device)
        }
        return result

    # 1. Compute color plaquette for all triplets
    source_pi, source_valid = _compute_color_plaquette_for_triplets(
        color=color,
        color_valid=data.color_valid,
        alive=data.alive,
        companions_distance=data.companions_distance,
        companions_clone=data.companions_clone,
        eps=data.eps,
    )

    # 2. Apply glueball observable
    resolved_mode = _resolve_glueball_operator_mode(
        operator_mode=config.operator_mode,
        use_action_form=config.use_action_form,
    )
    source_obs = _glueball_observable_from_plaquette(source_pi, operator_mode=resolved_mode)

    # 3. Average per frame
    triplet_counts = source_valid.sum(dim=1).to(torch.float32)
    valid_t = triplet_counts > 0

    operator_series = torch.zeros(t_total, dtype=torch.float32, device=device)
    if torch.any(valid_t):
        weight = source_valid.to(dtype=torch.float32)
        sums = (source_obs * weight).sum(dim=1)
        operator_series[valid_t] = sums[valid_t] / triplet_counts[valid_t].clamp(min=1.0)

    result = {"glueball": operator_series}

    # 4. Momentum projection (operator series only, no correlators)
    if config.use_momentum_projection:
        if data.positions_axis is None:
            msg = "positions_axis is required when use_momentum_projection=True."
            raise ValueError(msg)
        positions_axis = data.positions_axis
        if positions_axis.shape != source_obs.shape:
            msg = (
                "positions_axis must match source_obs shape [T, N], got "
                f"{tuple(positions_axis.shape)} vs {tuple(source_obs.shape)}."
            )
            raise ValueError(msg)

        box_length = _resolve_positive_length(
            positions_axis=positions_axis,
            alive=data.alive,
            box_length=data.projection_length,
        )

        n_modes = max(0, int(config.momentum_mode_max)) + 1
        modes = torch.arange(n_modes, device=device, dtype=torch.float32)
        k_values = (2.0 * torch.pi / float(box_length)) * modes

        # phase_arg: [n_modes, T, N]
        phase_arg = k_values[:, None, None] * positions_axis[None, :, :].float()
        cos_phase = torch.cos(phase_arg)
        sin_phase = torch.sin(phase_arg)

        weights = source_valid.to(dtype=torch.float32)
        counts_t = weights.sum(dim=1)
        valid_frames = counts_t > 0

        op_cos = torch.zeros((n_modes, t_total), dtype=torch.float32, device=device)
        op_sin = torch.zeros((n_modes, t_total), dtype=torch.float32, device=device)
        if torch.any(valid_frames):
            weighted_obs = source_obs.float() * weights
            cos_num = (weighted_obs[None, :, :] * cos_phase).sum(dim=2)  # [M, T]
            sin_num = (weighted_obs[None, :, :] * sin_phase).sum(dim=2)  # [M, T]
            denom = counts_t[valid_frames].to(dtype=torch.float32).clamp(min=1.0)
            op_cos[:, valid_frames] = cos_num[:, valid_frames] / denom.unsqueeze(0)
            op_sin[:, valid_frames] = sin_num[:, valid_frames] / denom.unsqueeze(0)

        for m in range(n_modes):
            result[f"glueball_momentum_cos_{m}"] = op_cos[m]
            result[f"glueball_momentum_sin_{m}"] = op_sin[m]

    return result
