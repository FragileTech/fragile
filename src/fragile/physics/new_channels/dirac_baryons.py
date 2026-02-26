"""Dirac baryon operators for the strong sector.

Constructs baryon interpolating fields from three walkers (i, j, k) using:

1. **Color contraction**: ε_{abc} c_i^a c_j^b c_k^c = det(c_i, c_j, c_k)
   This guarantees a color-singlet (baryon has zero net color charge).

2. **Diquark spinor projection**: (ψ_j)^T C Γ ψ_k
   Two of the three quarks form a "diquark" with spinor indices contracted
   through the charge conjugation matrix C and a Dirac matrix Γ.
   Different Γ select different diquark quantum numbers.

3. **Parity projection on third quark**: P_± ψ_i = (1 ± γ₀)/2 ψ_i
   Projects the baryon onto definite parity. P₊ = upper components (positive
   parity), P₋ = lower components (negative parity).

The full baryon operator (scalar for autocorrelation):

    B_Γ(t) = Σ_{i,j,k} det(c_i,c_j,c_k) · (ψ_j^T CΓ ψ_k) · Tr(P_± ψ_i)

Baryon channels:

    | Diquark Γ | P  | Baryon J^P | State       | Mass (MeV) |
    |-----------|-----|-----------|-------------|------------|
    | Cγ₅      | P₊ | 1/2⁺      | Nucleon (N) | 939        |
    | Cγ_k     | P₊ | 3/2⁺      | Delta (Δ)   | 1232       |
    | C        | P₋ | 1/2⁻      | N*(1535)    | 1535       |
    | Cγ₅γ_k   | P₋ | 3/2⁻      | N*(1520)    | 1520       |

The existing det-based nucleon operator is equivalent to averaging over
ALL diquark/parity channels — it couples to the lightest baryon (nucleon)
but with suboptimal overlap. These Dirac operators should give cleaner
signals and access to excited baryon states.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .dirac_spinors import (
    build_dirac_gamma_matrices,
    color_to_dirac_spinor,
)


# ===========================================================================
# Diquark computation
# ===========================================================================


def compute_diquark(
    psi_a: Tensor,
    psi_b: Tensor,
    C: Tensor,
    Gamma: Tensor,
) -> Tensor:
    """Compute diquark scalar: (ψ_a)^T C Γ ψ_b.

    The diquark is the spinor contraction of two quarks through the
    charge conjugation matrix C and a Dirac matrix Γ.

    Result is a Lorentz scalar (all spinor indices contracted).

    Args:
        psi_a: Spinor at first diquark site [..., 4].
        psi_b: Spinor at second diquark site [..., 4].
        C: Charge conjugation matrix [4, 4].
        Gamma: Diquark Dirac matrix [4, 4] or [n, 4, 4].

    Returns:
        Diquark values [...] (complex) or [..., n] if Gamma is [n, 4, 4].
    """
    # (ψ_a)^T C Γ ψ_b = Σ_{st} (ψ_a)_s (CΓ)_{st} (ψ_b)_t
    if Gamma.ndim == 2:
        CG = C @ Gamma  # [4, 4]
        return torch.einsum("...s,st,...t->...", psi_a, CG, psi_b)
    else:
        # Gamma is [n, 4, 4]
        CG = torch.einsum("sa,nab->nsb", C, Gamma)  # [n, 4, 4]
        return torch.einsum("...s,nst,...t->...n", psi_a, CG, psi_b)


def parity_projection(
    psi: Tensor,
    gamma0: Tensor,
    positive: bool = True,
) -> Tensor:
    """Project spinor onto definite parity and sum components.

    P_± = (1 ± γ₀) / 2

    In Dirac representation, γ₀ = diag(1,1,-1,-1), so:
        P₊ = diag(1,1,0,0)  → picks upper 2 components (left-handed)
        P₋ = diag(0,0,1,1)  → picks lower 2 components (right-handed)

    Returns the sum of projected components: Σ_s (P ψ)_s
    This produces a scalar suitable for autocorrelation.

    Args:
        psi: Spinor [..., 4].
        gamma0: γ₀ matrix [4, 4].
        positive: True for P₊ (positive parity), False for P₋.

    Returns:
        Scalar projection [...] (complex).
    """
    I4 = torch.eye(4, device=gamma0.device, dtype=gamma0.dtype)
    if positive:
        P = 0.5 * (I4 + gamma0)
    else:
        P = 0.5 * (I4 - gamma0)

    # (P ψ)_s then sum over s
    projected = torch.einsum("st,...t->...s", P, psi)  # [..., 4]
    return projected.sum(dim=-1)  # [...] complex


# ===========================================================================
# Output dataclass
# ===========================================================================


@dataclass
class DiracBaryonSeries:
    """Baryon operator time series for all Dirac channels.

    Each field is [T] — the spatially averaged operator at each MC timestep.
    """

    # Positive parity baryons
    nucleon: Tensor       # J^P = 1/2⁺  (Γ = Cγ₅, P₊)  — proton/neutron
    delta: Tensor         # J^P = 3/2⁺  (Γ = Cγ_k, P₊)  — Δ(1232)

    # Negative parity baryons
    n_star_scalar: Tensor  # J^P = 1/2⁻  (Γ = C, P₋)    — N*(1535)
    n_star_axial: Tensor   # J^P = 3/2⁻  (Γ = Cγ₅γ_k, P₋) — N*(1520)

    # Cross-check: existing det-based operator (no spinor structure)
    nucleon_det: Tensor   # det(c_i, c_j, c_k) — mixes all channels

    # Diagnostics
    n_valid_triplets: Tensor      # [T] valid (i,j,k) triplets
    spinor_valid_fraction: Tensor  # [T] fraction of walkers with valid spinors


# ===========================================================================
# Main computation
# ===========================================================================


def compute_dirac_baryon_operators(
    color: Tensor,
    color_valid: Tensor,
    sample_indices: Tensor,
    neighbor_indices: Tensor,
    alive: Tensor,
    sample_edge_weights: Tensor | None = None,
) -> DiracBaryonSeries:
    """Compute all Dirac baryon operator time series.

    Requires d=3 (three color components) and at least 2 neighbors per sample
    to form triplets (i, j, k).

    Args:
        color: Complex color states [T, N, 3].
        color_valid: Color validity mask [T, N].
        sample_indices: Sample walker indices [T, S].
        neighbor_indices: Neighbor indices [T, S, k] where k >= 2.
        alive: Alive mask [T, N].
        sample_edge_weights: Optional Riemannian weights [T, S].

    Returns:
        DiracBaryonSeries with all baryon channel time series.
    """
    T, N, d = color.shape
    S = sample_indices.shape[1]
    k_neighbors = neighbor_indices.shape[2]
    device = color.device

    if d < 3:
        raise ValueError(f"Dirac baryons require d >= 3, got d={d}")
    if k_neighbors < 2:
        raise ValueError(
            f"Dirac baryons require >= 2 neighbors, got {k_neighbors}. "
            "Increase neighbor_k in config."
        )

    # --- Build gamma matrices ---
    gamma = build_dirac_gamma_matrices(device=device)
    gamma0 = gamma["gamma0"]
    gamma_k = gamma["gamma_k"]   # [3, 4, 4]
    gamma5 = gamma["gamma5"]
    gamma5_k = gamma["gamma5_k"]  # [3, 4, 4]
    C = gamma["C"]
    I4 = torch.eye(4, device=device, dtype=gamma0.dtype)

    # --- Convert color → spinor ---
    color_3 = color[..., :3]  # use first 3 components
    spinor, spinor_valid = color_to_dirac_spinor(color_3)  # [T, N, 4], [T, N]
    spinor_valid = spinor_valid & color_valid

    # --- Gather triplets (i, j, k) ---
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, S)

    # Walker i = sample, j = first neighbor, k = second neighbor
    idx_i = sample_indices                  # [T, S]
    idx_j = neighbor_indices[:, :, 0]       # [T, S]
    idx_k = neighbor_indices[:, :, 1]       # [T, S]

    # Color states for determinant
    c_i = color_3[t_idx, idx_i]  # [T, S, 3]
    c_j = color_3[t_idx, idx_j]  # [T, S, 3]
    c_k = color_3[t_idx, idx_k]  # [T, S, 3]

    # Spinor states
    psi_i = spinor[t_idx, idx_i]  # [T, S, 4]
    psi_j = spinor[t_idx, idx_j]  # [T, S, 4]
    psi_k = spinor[t_idx, idx_k]  # [T, S, 4]

    # --- Validity mask ---
    def _valid_at(idx: Tensor) -> Tensor:
        return (
            spinor_valid[t_idx, idx]
            & alive[t_idx.clamp(max=alive.shape[0] - 1), idx]
        )

    valid_i = _valid_at(idx_i)
    valid_j = _valid_at(idx_j)
    valid_k = _valid_at(idx_k)

    # All three must be valid AND distinct
    valid_mask = (
        valid_i & valid_j & valid_k
        & (idx_i != idx_j) & (idx_j != idx_k) & (idx_i != idx_k)
    )

    # --- Color determinant: det(c_i, c_j, c_k) ---
    # Stack into [T, S, 3, 3], columns = (c_i, c_j, c_k)
    color_matrix = torch.stack([c_i, c_j, c_k], dim=-1)
    det_color = torch.linalg.det(color_matrix)  # [T, S] complex

    # --- Diquark scalars ---
    # Nucleon diquark: ψ_j^T C γ₅ ψ_k  (scalar, 0⁺)
    diquark_nucleon = compute_diquark(psi_j, psi_k, C, gamma5)  # [T, S] complex

    # Delta diquark: ψ_j^T C γ_m ψ_k  averaged over m=1,2,3 (vector, 1⁺)
    diquark_delta = compute_diquark(psi_j, psi_k, C, gamma_k)  # [T, S, 3] complex
    diquark_delta = diquark_delta.mean(dim=-1)  # [T, S]

    # N*(1535) diquark: ψ_j^T C ψ_k  (scalar, 0⁻)
    diquark_nstar_s = compute_diquark(psi_j, psi_k, C, I4)  # [T, S]

    # N*(1520) diquark: ψ_j^T C γ₅γ_m ψ_k  averaged over m (vector, 1⁻)
    diquark_nstar_a = compute_diquark(psi_j, psi_k, C, gamma5_k)  # [T, S, 3]
    diquark_nstar_a = diquark_nstar_a.mean(dim=-1)  # [T, S]

    # --- Third quark parity projection ---
    # Positive parity: Σ_s (P₊ ψ_i)_s = ψ_i[0] + ψ_i[1]
    proj_plus = parity_projection(psi_i, gamma0, positive=True)   # [T, S] complex
    # Negative parity: Σ_s (P₋ ψ_i)_s = ψ_i[2] + ψ_i[3]
    proj_minus = parity_projection(psi_i, gamma0, positive=False)  # [T, S] complex

    # --- Full baryon operators ---
    # B = det_color × diquark × parity_projection
    op_nucleon = det_color * diquark_nucleon * proj_plus       # 1/2⁺
    op_delta = det_color * diquark_delta * proj_plus            # 3/2⁺
    op_nstar_s = det_color * diquark_nstar_s * proj_minus       # 1/2⁻
    op_nstar_a = det_color * diquark_nstar_a * proj_minus       # 3/2⁻

    # Take real part (baryon operators are real after spatial averaging)
    # The imaginary part averages to zero for parity eigenstates
    op_nucleon = op_nucleon.real
    op_delta = op_delta.real
    op_nstar_s = op_nstar_s.real
    op_nstar_a = op_nstar_a.real

    # Existing det-based nucleon (cross-check)
    op_det = det_color.real

    # --- Mask and average ---
    zero = torch.zeros_like(op_nucleon)

    def _mask_and_avg(op: Tensor) -> Tensor:
        op_masked = torch.where(valid_mask, op, zero)
        if sample_edge_weights is not None:
            w = sample_edge_weights.to(device=device, dtype=op.dtype)
            w = torch.where(valid_mask, w, zero)
        else:
            w = valid_mask.float()
        w_sum = w.sum(dim=1).clamp(min=1e-12)
        return (op_masked * w).sum(dim=1) / w_sum

    return DiracBaryonSeries(
        nucleon=_mask_and_avg(op_nucleon),
        delta=_mask_and_avg(op_delta),
        n_star_scalar=_mask_and_avg(op_nstar_s),
        n_star_axial=_mask_and_avg(op_nstar_a),
        nucleon_det=_mask_and_avg(op_det),
        n_valid_triplets=valid_mask.sum(dim=1),
        spinor_valid_fraction=(spinor_valid & alive).float().mean(dim=1),
    )


def compute_dirac_baryons_from_agg(
    agg_data: "AggregatedTimeSeries",
) -> DiracBaryonSeries:
    """Compute Dirac baryon operators from AggregatedTimeSeries.

    Convenience wrapper matching the aggregation.py API.

    Args:
        agg_data: AggregatedTimeSeries from aggregate_time_series().

    Returns:
        DiracBaryonSeries.
    """
    return compute_dirac_baryon_operators(
        color=agg_data.color,
        color_valid=agg_data.color_valid,
        sample_indices=agg_data.sample_indices,
        neighbor_indices=agg_data.neighbor_indices,
        alive=agg_data.alive,
        sample_edge_weights=agg_data.sample_edge_weights,
    )
