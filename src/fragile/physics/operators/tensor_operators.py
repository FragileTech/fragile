"""Companion-pair spin-2 tensor operators (bilinear sigma_{mu,nu} projection).

Implements the tensor channel using the same bilinear pattern as all other
companion channels:

    O(t) = Im[c_i^dag sigma_{mu,nu} c_j]  averaged over the 3 independent
    antisymmetric matrices for d=3.

sigma_{mu,nu} is purely imaginary, so Im[...] is parity-even -> tensor (2++).
This matches :class:`~fragile.physics.new_channels.correlator_channels.TensorChannel`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fragile.physics.qft_utils.companions import build_companion_pair_indices
from fragile.physics.qft_utils.helpers import (
    safe_gather_pairs_2d,
    safe_gather_pairs_3d,
)

from .config import TensorOperatorConfig
from .preparation import PreparedChannelData


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_sigma_matrices(d: int, device: torch.device) -> Tensor:
    """Build antisymmetric sigma_{mu,nu} matrices for the tensor channel.

    Returns:
        ``[n_pairs, d, d]`` complex tensor with n_pairs = d*(d-1)/2.
        For d=3: 3 matrices (xy, xz, yz).
    """
    dtype = torch.complex128
    sigma_list = []
    for mu in range(d):
        for nu in range(mu + 1, d):
            sigma = torch.zeros(d, d, device=device, dtype=dtype)
            sigma[mu, nu] = 1.0j
            sigma[nu, mu] = -1.0j
            sigma_list.append(sigma)
    if sigma_list:
        return torch.stack(sigma_list, dim=0)
    return torch.zeros(0, d, d, device=device, dtype=dtype)


def _per_frame_series(values: Tensor, valid: Tensor) -> Tensor:
    """Average pair values per frame with masking."""
    weights = valid.to(values.dtype)
    counts = valid.sum(dim=(1, 2)).to(torch.int64)
    sums = (values * weights).sum(dim=(1, 2))
    series = torch.zeros(values.shape[0], device=values.device, dtype=torch.float32)
    valid_t = counts > 0
    if torch.any(valid_t):
        series[valid_t] = (sums[valid_t] / counts[valid_t].to(values.dtype)).float()
    return series


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_tensor_operators(
    data: PreparedChannelData,
    config: TensorOperatorConfig,
) -> dict[str, Tensor]:
    """Compute spin-2 tensor operator time-series from prepared channel data.

    Uses the bilinear form ``Im[c_i^dag sigma_{mu,nu} c_j]`` averaged over
    the independent antisymmetric matrices, producing a single scalar per
    frame (no position/momentum dependence).

    Returns a dict:
        ``"tensor"`` -- per-frame averaged tensor series ``[T]``
        (single-scale) or ``[S, T]`` (multiscale).
    """
    device = data.device
    t_total = int(data.color.shape[0])
    d = data.color.shape[-1]

    if t_total == 0:
        if data.scales is not None:
            S = data.scales.shape[0]
            empty = torch.zeros(S, 0, dtype=torch.float32, device=device)
        else:
            empty = torch.zeros(0, dtype=torch.float32, device=device)
        return {"tensor": empty}

    # 1. Build companion pair indices
    pair_indices, structural_valid = build_companion_pair_indices(
        companions_distance=data.companions_distance,
        companions_clone=data.companions_clone,
        pair_selection=str(config.pair_selection),
    )

    # 2. Gather color states for pairs
    color_j, in_range = safe_gather_pairs_3d(data.color, pair_indices)
    valid_j, _ = safe_gather_pairs_2d(data.color_valid, pair_indices)

    color_i = data.color.unsqueeze(2).expand_as(color_j)  # [T, N, P, d]

    finite_i = torch.isfinite(color_i.real) & torch.isfinite(color_i.imag)
    finite_j = torch.isfinite(color_j.real) & torch.isfinite(color_j.imag)
    valid = (
        structural_valid
        & in_range
        & data.color_valid.unsqueeze(-1)
        & valid_j
        & finite_i.all(dim=-1)
        & finite_j.all(dim=-1)
    )

    # 3. Compute sigma projection: Im[c_i^dag sigma_{mu,nu} c_j] averaged
    sigma = _build_sigma_matrices(d, device).to(dtype=color_i.dtype)  # [n_pairs, d, d]
    if sigma.shape[0] == 0:
        op_tensor = torch.zeros(color_i.shape[:-1], device=device)
    else:
        # einsum: ...i, pij, ...j -> ...p  then mean over p and take .imag
        result = torch.einsum("...i,pij,...j->...p", color_i.conj(), sigma, color_j)
        op_tensor = result.mean(dim=-1).imag.float()  # [T, N, P]

    # 4. Mask invalid pairs
    op_tensor = torch.where(valid, op_tensor, torch.zeros_like(op_tensor))

    # 5. Average per frame (multiscale or single-scale)
    if data.scales is not None and data.pairwise_distances is not None:
        from .multiscale import gate_pair_validity_by_scale, per_frame_series_multiscale

        valid_ms = gate_pair_validity_by_scale(
            valid,
            pair_indices,
            data.pairwise_distances,
            data.scales,
        )
        tensor_series = (
            per_frame_series_multiscale(op_tensor.square(), valid_ms).clamp_min(0).sqrt()
        )
    else:
        tensor_series = _per_frame_series(op_tensor.square(), valid).clamp_min(0).sqrt()

    # RMS is a diagnostic envelope only; use compute_tensor_correlator for
    # the physical pair-propagated correlation, not an FFT of this envelope.
    return {"tensor": tensor_series}


def compute_tensor_correlator(data, config, max_lag: int, use_connected: bool = True) -> Tensor:
    """Contract tensor components after propagating fixed source pairs.

    This uses the same imaginary Hermitian bilinear as the original operator;
    exchange-odd pair values are multiplied before spatial averaging.
    """
    from fragile.physics.operators.multiscale import gate_pair_validity_by_scale
    from fragile.physics.qft_utils.statistics import (
        attach_statistics,
        record_lag,
        stack_correlators,
    )

    color, valid_color = data.color, data.color_valid
    T = len(color)
    pairs, structural = build_companion_pair_indices(
        data.companions_distance, data.companions_clone, config.pair_selection
    )
    sigma = _build_sigma_matrices(color.shape[-1], color.device).to(color.dtype)

    def values(c, v, indices, mask):
        cj, in_range = safe_gather_pairs_3d(c, indices)
        vj, _ = safe_gather_pairs_2d(v, indices)
        valid = mask & in_range & v.unsqueeze(-1) & vj
        op = torch.einsum("...i,pij,...j->...p", c.unsqueeze(2).conj(), sigma, cj).imag
        valid &= torch.isfinite(op).all(-1)
        return op, valid

    source, source_valid = values(color, valid_color, pairs, structural)
    scales = getattr(data, "scales", None)
    distances = getattr(data, "pairwise_distances", None)
    multiscale = scales is not None and distances is not None
    masks = (
        gate_pair_validity_by_scale(source_valid, pairs, distances, scales)
        if multiscale
        else source_valid.unsqueeze(1)
    )
    outputs = []
    for scale_idx in range(masks.shape[1]):
        mask = masks[:, scale_idx]
        mean = source[mask].mean(0) if mask.any() else source.new_zeros(len(sigma))
        if not use_connected:
            mean = torch.zeros_like(mean)
        sums = torch.zeros(T, max_lag + 1, device=color.device, dtype=torch.float64)
        counts = torch.zeros_like(sums)
        for lag in range(min(T, max_lag + 1)):
            sink, valid = values(
                color[lag:], valid_color[lag:], pairs[: T - lag], structural[: T - lag]
            )
            if multiscale:
                valid = gate_pair_validity_by_scale(
                    valid, pairs[: T - lag], distances[lag:], scales[scale_idx : scale_idx + 1]
                )[:, 0]
            products = ((source[: T - lag] - mean) * (sink - mean)).sum(-1)
            record_lag(sums, counts, lag, products, valid & mask[: T - lag])
        corr = sums.sum(0) / counts.sum(0).clamp_min(1)
        outputs.append(attach_statistics(corr, sums, counts))
    return stack_correlators(outputs) if multiscale else outputs[0]
