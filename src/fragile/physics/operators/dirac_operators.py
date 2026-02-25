"""Dirac spinor bilinear operator construction from companion pairs.

Computes per-frame Dirac bilinear operator time series by mapping complex
color states c ∈ ℂ³ to Dirac spinors ψ ∈ ℂ⁴ via the Hopf fibration,
then computing ψ̄_i Γ ψ_j for all five channels:

    dirac_scalar:       ψ̄ψ (Γ = I₄)
    dirac_pseudoscalar: ψ̄γ₅ψ
    dirac_vector:       (1/3)Σ_k ψ̄γ_k ψ
    dirac_axial_vector: (1/3)Σ_k ψ̄γ₅γ_k ψ
    dirac_tensor:       (1/6)Σ_{μ<ν} ψ̄σ_μν ψ

This provides a cross-check against the standard Re/Im color operators:
if the physics is correct, ground-state masses must agree.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fragile.physics.new_channels.dirac_spinors import (
    build_dirac_gamma_matrices,
    color_to_dirac_spinor,
    compute_dirac_bilinear,
)
from fragile.physics.qft_utils.companions import (
    build_companion_pair_indices,
)
from fragile.physics.qft_utils.helpers import (
    safe_gather_pairs_2d,
    safe_gather_pairs_3d,
)

from .config import ChannelConfigBase
from .preparation import PreparedChannelData


# ---------------------------------------------------------------------------
# Per-frame averaging
# ---------------------------------------------------------------------------


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
# Main entry point
# ---------------------------------------------------------------------------


def compute_dirac_operators(
    data: PreparedChannelData,
    config: ChannelConfigBase,
) -> dict[str, Tensor]:
    """Compute Dirac spinor bilinear operator time series.

    Args:
        data: Pre-extracted channel tensors from :func:`prepare_channel_data`.
        config: Operator configuration (uses base config fields only).

    Returns:
        Dictionary with keys ``"dirac_scalar"``, ``"dirac_pseudoscalar"``,
        ``"dirac_vector"``, ``"dirac_axial_vector"``, ``"dirac_tensor"``,
        each a ``[T]`` tensor of per-frame averaged operator values.
    """
    device = data.device
    t_total = int(data.color.shape[0])

    empty = torch.zeros(0, dtype=torch.float32, device=device)
    if t_total == 0:
        return {
            "dirac_scalar": empty,
            "dirac_pseudoscalar": empty.clone(),
            "dirac_vector": empty.clone(),
            "dirac_axial_vector": empty.clone(),
            "dirac_tensor": empty.clone(),
            "dirac_tensor_0k": empty.clone(),
        }

    # Require d=3 for Dirac spinor construction
    if data.color.shape[-1] != 3:
        return {
            "dirac_scalar": empty,
            "dirac_pseudoscalar": empty.clone(),
            "dirac_vector": empty.clone(),
            "dirac_axial_vector": empty.clone(),
            "dirac_tensor": empty.clone(),
            "dirac_tensor_0k": empty.clone(),
        }

    pair_selection = str(config.pair_selection).strip().lower()

    # 1. Build companion pair indices
    pair_indices, structural_valid = build_companion_pair_indices(
        companions_distance=data.companions_distance,
        companions_clone=data.companions_clone,
        pair_selection=pair_selection,
    )

    # 2. Gather color states for pairs
    color_j, in_range = safe_gather_pairs_3d(data.color, pair_indices)
    valid_j, _ = safe_gather_pairs_2d(data.color_valid, pair_indices)

    color_i = data.color.unsqueeze(2).expand_as(color_j)  # [T, N, P, 3]

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

    # 3. Convert color states to Dirac spinors
    # Reshape for batch processing: [T, N, P, 3] -> [T*N*P, 3]
    flat_shape = color_i.shape[:-1]  # [T, N, P]
    color_i_flat = color_i.reshape(-1, 3)
    color_j_flat = color_j.reshape(-1, 3)

    psi_i, valid_i_spinor = color_to_dirac_spinor(color_i_flat, eps=config.eps)
    psi_j, valid_j_spinor = color_to_dirac_spinor(color_j_flat, eps=config.eps)

    # Reshape back
    psi_i = psi_i.reshape(*flat_shape, 4)  # [T, N, P, 4]
    psi_j = psi_j.reshape(*flat_shape, 4)
    spinor_valid = (
        valid_i_spinor.reshape(flat_shape) & valid_j_spinor.reshape(flat_shape)
    )

    # Combined validity
    valid = valid & spinor_valid

    # 4. Build gamma matrices
    gamma = build_dirac_gamma_matrices(device=device)
    gamma0 = gamma["gamma0"]
    gamma5 = gamma["gamma5"]
    gamma_k = gamma["gamma_k"]
    gamma5_k = gamma["gamma5_k"]
    sigma_munu = gamma["sigma_munu"]
    I4 = torch.eye(4, device=device, dtype=gamma0.dtype)

    # 5. Compute bilinears
    op_scalar = compute_dirac_bilinear(psi_i, psi_j, gamma0, I4)         # [T, N, P]
    op_pseudo = compute_dirac_bilinear(psi_i, psi_j, gamma0, gamma5)     # [T, N, P]
    op_vector_k = compute_dirac_bilinear(psi_i, psi_j, gamma0, gamma_k)  # [T, N, P, 3]
    op_vector = op_vector_k.mean(dim=-1)                                  # [T, N, P]
    op_axial_k = compute_dirac_bilinear(psi_i, psi_j, gamma0, gamma5_k)  # [T, N, P, 3]
    op_axial = op_axial_k.mean(dim=-1)                                    # [T, N, P]
    op_tensor_mn = compute_dirac_bilinear(psi_i, psi_j, gamma0, sigma_munu)  # [T, N, P, 6]
    op_tensor_0k = op_tensor_mn[..., :3].mean(dim=-1)                        # [T, N, P] σ_0k: parity-odd
    op_tensor = op_tensor_mn[..., 3:].mean(dim=-1)                           # [T, N, P] σ_jk: parity-even

    # 6. Mask invalid pairs
    zero = torch.zeros_like(op_scalar)
    op_scalar = torch.where(valid, op_scalar, zero)
    op_pseudo = torch.where(valid, op_pseudo, zero)
    op_vector = torch.where(valid, op_vector, zero)
    op_axial = torch.where(valid, op_axial, zero)
    op_tensor = torch.where(valid, op_tensor, zero)
    op_tensor_0k = torch.where(valid, op_tensor_0k, zero)

    # 7. Average per frame (multiscale or single-scale)
    if data.scales is not None and data.pairwise_distances is not None:
        from .multiscale import gate_pair_validity_by_scale, per_frame_series_multiscale

        valid_ms = gate_pair_validity_by_scale(
            valid,
            pair_indices,
            data.pairwise_distances,
            data.scales,
        )
        scalar_series = per_frame_series_multiscale(op_scalar, valid_ms)
        pseudo_series = per_frame_series_multiscale(op_pseudo, valid_ms)
        vector_series = per_frame_series_multiscale(op_vector, valid_ms)
        axial_series = per_frame_series_multiscale(op_axial, valid_ms)
        tensor_series = per_frame_series_multiscale(op_tensor, valid_ms)
        tensor_0k_series = per_frame_series_multiscale(op_tensor_0k, valid_ms)
    else:
        scalar_series = _per_frame_series(op_scalar, valid)
        pseudo_series = _per_frame_series(op_pseudo, valid)
        vector_series = _per_frame_series(op_vector, valid)
        axial_series = _per_frame_series(op_axial, valid)
        tensor_series = _per_frame_series(op_tensor, valid)
        tensor_0k_series = _per_frame_series(op_tensor_0k, valid)

    return {
        "dirac_scalar": scalar_series,
        "dirac_pseudoscalar": pseudo_series,
        "dirac_vector": vector_series,
        "dirac_axial_vector": axial_series,
        "dirac_tensor": tensor_series,
        "dirac_tensor_0k": tensor_0k_series,
    }
