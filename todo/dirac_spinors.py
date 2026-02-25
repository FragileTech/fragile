"""Dirac spinor representation for strong-force meson operators.

Maps the complex color states c_i ∈ ℂ³ (from viscous force complexification)
to proper Dirac spinors ψ_i ∈ ℂ⁴, enabling meson operators ψ̄_i Γ ψ_j with
guaranteed quantum numbers from the Clifford algebra.

Parity transformation of color states:
    c^α → -(c^α)*     (v → -v, F_visc → -F_visc)

Therefore:
    Re(c^α) → -Re(c^α)    parity-ODD
    Im(c^α) → +Im(c^α)    parity-EVEN

Dirac spinor in standard representation (γ₀ = diag(I₂, -I₂)):
    ψ_L (upper 2 components) → +ψ_L   under parity  (even)
    ψ_R (lower 2 components) → -ψ_R   under parity  (odd)

Matching: Im(c) → ψ_L, Re(c) → ψ_R

The map ℝ³ → ℂ² uses the Hopf fibration / spinor section:

    ξ(w) = 1/√(2|w|(|w|+w₃)) · (|w|+w₃, w₁+iw₂)ᵀ

with chart switching near the south pole (w₃ ≈ -|w|).

Bilinear ψ̄Γψ = ψ†γ₀Γψ then gives:
    Γ = I₄         → scalar     (0⁺⁺)
    Γ = γ₅         → pseudoscalar (0⁻⁺)
    Γ = γ_k        → vector     (1⁻⁻)
    Γ = γ₅γ_k      → axial vector (1⁺⁻)
    Γ = σ_μν       → tensor     (2⁺⁺)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# ===========================================================================
# Gamma matrices: standard Dirac representation (4×4)
# ===========================================================================


def build_dirac_gamma_matrices(
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.complex128,
) -> dict[str, Tensor]:
    """Build standard 4×4 Dirac gamma matrices.

    Convention (Dirac/Pauli representation):
        γ⁰ = diag(I₂, -I₂)
        γᵏ = [[0, σ_k], [-σ_k, 0]]     k = 1, 2, 3
        γ⁵ = [[0, I₂], [I₂, 0]]

    Satisfies: {γ_μ, γ_ν} = 2 g_μν  with g = diag(+,-,-,-)
    and γ₅ anticommutes with all γ_μ.

    Returns:
        Dictionary with keys:
            "gamma0": [4, 4]
            "gamma_k": [3, 4, 4]  (spatial gammas, k=1,2,3)
            "gamma5": [4, 4]
            "gamma5_k": [3, 4, 4]  (γ₅γ_k)
            "sigma_munu": [6, 4, 4]  (σ_μν = (i/2)[γ_μ,γ_ν] for μ<ν)
            "C": [4, 4]  (charge conjugation matrix)
    """
    # Pauli matrices
    sigma_1 = torch.tensor([[0, 1], [1, 0]], device=device, dtype=dtype)
    sigma_2 = torch.tensor([[0, -1j], [1j, 0]], device=device, dtype=dtype)
    sigma_3 = torch.tensor([[1, 0], [0, -1]], device=device, dtype=dtype)
    pauli = [sigma_1, sigma_2, sigma_3]

    I2 = torch.eye(2, device=device, dtype=dtype)
    Z2 = torch.zeros(2, 2, device=device, dtype=dtype)

    # γ⁰ = diag(I₂, -I₂)
    gamma0 = torch.block_diag(I2, -I2)

    # γᵏ = [[0, σ_k], [-σ_k, 0]]
    gamma_k_list = []
    for k in range(3):
        top = torch.cat([Z2, pauli[k]], dim=1)       # [2, 4]
        bot = torch.cat([-pauli[k], Z2], dim=1)      # [2, 4]
        gamma_k_list.append(torch.cat([top, bot], dim=0))  # [4, 4]
    gamma_k = torch.stack(gamma_k_list, dim=0)  # [3, 4, 4]

    # γ⁵ = iγ⁰γ¹γ²γ³ = [[0, I₂], [I₂, 0]]
    gamma5 = torch.cat([
        torch.cat([Z2, I2], dim=1),
        torch.cat([I2, Z2], dim=1),
    ], dim=0)

    # γ₅γ_k (axial vector)
    gamma5_k = torch.stack([gamma5 @ gamma_k_list[k] for k in range(3)], dim=0)

    # σ_μν = (i/2)[γ_μ, γ_ν] for all μ < ν (0-indexed: 01, 02, 03, 12, 13, 23)
    all_gammas = [gamma0] + gamma_k_list  # γ₀, γ₁, γ₂, γ₃
    sigma_list = []
    for mu in range(4):
        for nu in range(mu + 1, 4):
            comm = all_gammas[mu] @ all_gammas[nu] - all_gammas[nu] @ all_gammas[mu]
            sigma_list.append(0.5j * comm)
    sigma_munu = torch.stack(sigma_list, dim=0)  # [6, 4, 4]

    # Charge conjugation matrix C = iγ²γ⁰
    C = 1j * gamma_k_list[1] @ gamma0

    return {
        "gamma0": gamma0,
        "gamma_k": gamma_k,
        "gamma5": gamma5,
        "gamma5_k": gamma5_k,
        "sigma_munu": sigma_munu,
        "C": C,
    }


def verify_clifford_algebra(gamma: dict[str, Tensor], tol: float = 1e-10) -> dict[str, bool]:
    """Verify that gamma matrices satisfy the Clifford algebra.

    Checks:
        1. {γ_μ, γ_ν} = 2 g_μν I₄
        2. γ₅ anticommutes with all γ_μ
        3. (γ₅)² = I₄
        4. (γ_μ)† = γ₀ γ_μ γ₀  (hermiticity)

    Returns:
        Dictionary of test names → pass/fail.
    """
    gamma0 = gamma["gamma0"]
    gamma_k = gamma["gamma_k"]
    gamma5 = gamma["gamma5"]
    I4 = torch.eye(4, device=gamma0.device, dtype=gamma0.dtype)

    all_gammas = [gamma0] + [gamma_k[k] for k in range(3)]
    g_metric = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0],
                                        device=gamma0.device, dtype=gamma0.dtype))

    results = {}

    # {γ_μ, γ_ν} = 2 g_μν I₄
    clifford_ok = True
    for mu in range(4):
        for nu in range(4):
            anticomm = all_gammas[mu] @ all_gammas[nu] + all_gammas[nu] @ all_gammas[mu]
            expected = 2.0 * g_metric[mu, nu] * I4
            if (anticomm - expected).abs().max() > tol:
                clifford_ok = False
    results["clifford_anticommutation"] = clifford_ok

    # γ₅ anticommutes with all γ_μ
    gamma5_ok = True
    for mu in range(4):
        anticomm = gamma5 @ all_gammas[mu] + all_gammas[mu] @ gamma5
        if anticomm.abs().max() > tol:
            gamma5_ok = False
    results["gamma5_anticommutation"] = gamma5_ok

    # (γ₅)² = I₄
    results["gamma5_squared"] = bool(((gamma5 @ gamma5) - I4).abs().max() < tol)

    # Hermiticity: γ₀† = γ₀, γ_k† = -γ_k → γ_μ† = γ₀ γ_μ γ₀
    herm_ok = True
    for mu in range(4):
        lhs = all_gammas[mu].conj().T
        rhs = gamma0 @ all_gammas[mu] @ gamma0
        if (lhs - rhs).abs().max() > tol:
            herm_ok = False
    results["hermiticity"] = herm_ok

    return results


# ===========================================================================
# Vector → Spinor map (Hopf fibration section)
# ===========================================================================


def vector_to_weyl_spinor(
    w: Tensor,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """Map a batch of 3-vectors to 2-component Weyl spinors.

    Uses the standard section of the Hopf fibration S³ → S²:

        ξ(w) = 1/√(2r(r+w₃)) · (r+w₃, w₁+iw₂)ᵀ      (north chart)
        ξ(w) = 1/√(2r(r-w₃)) · (w₁-iw₂, r-w₃)ᵀ       (south chart)

    where r = |w|. Switches charts when |r + w₃| < ε|r| to avoid the
    south pole singularity.

    The spinor is normalized: ξ†ξ = 1 (for unit vectors).
    For general w, we scale: ξ(w) = √|w| · ξ(ŵ), preserving the
    information about the magnitude.

    Args:
        w: Input vectors [..., 3].
        eps: Tolerance for chart switching and zero-vector detection.

    Returns:
        Tuple of (spinor [..., 2] complex, valid [...] bool).
        Invalid where |w| < eps.
    """
    shape = w.shape[:-1]
    w = w.float()

    w1 = w[..., 0]
    w2 = w[..., 1]
    w3 = w[..., 2]
    r = torch.linalg.vector_norm(w, dim=-1)  # [...]

    valid = r > eps

    # North chart: r + w₃ > 0 (away from south pole)
    # South chart: r - w₃ > 0 (away from north pole)
    use_south = (r + w3).abs() < eps * r.clamp(min=eps)

    # Allocate spinor
    spinor = torch.zeros(*shape, 2, device=w.device, dtype=torch.complex64)

    # --- North chart ---
    north = valid & ~use_south
    if north.any():
        r_n = r[north]
        w1_n = w1[north]
        w2_n = w2[north]
        w3_n = w3[north]

        rp = (r_n + w3_n).clamp(min=eps)
        norm = torch.sqrt(2.0 * r_n * rp).clamp(min=eps)

        spinor_0 = rp / norm
        spinor_1 = (w1_n + 1j * w2_n) / norm

        # Scale by √r to carry magnitude information
        scale = torch.sqrt(r_n.clamp(min=eps))
        spinor[north, 0] = (spinor_0 * scale).to(torch.complex64)
        spinor[north, 1] = (spinor_1 * scale).to(torch.complex64)

    # --- South chart ---
    south = valid & use_south
    if south.any():
        r_s = r[south]
        w1_s = w1[south]
        w2_s = w2[south]
        w3_s = w3[south]

        rm = (r_s - w3_s).clamp(min=eps)
        norm = torch.sqrt(2.0 * r_s * rm).clamp(min=eps)

        spinor_0 = (w1_s - 1j * w2_s) / norm
        spinor_1 = rm / norm

        scale = torch.sqrt(r_s.clamp(min=eps))
        spinor[south, 0] = (spinor_0 * scale).to(torch.complex64)
        spinor[south, 1] = (spinor_1 * scale).to(torch.complex64)

    return spinor, valid


def color_to_dirac_spinor(
    color: Tensor,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """Convert complex color states c ∈ ℂ³ to Dirac spinors ψ ∈ ℂ⁴.

    The mapping uses parity matching:
        Im(c) → ψ_L (upper 2 components, parity-even)
        Re(c) → ψ_R (lower 2 components, parity-odd)

    via the Hopf fibration section on each 3-vector.

    Args:
        color: Complex color states [..., 3] (complex64 or complex128).
        eps: Tolerance for singularity/zero handling.

    Returns:
        Tuple of (spinor [..., 4] complex, valid [...] bool).
        Invalid where either Re(c) or Im(c) has vanishing norm.
    """
    re_c = color.real.float()  # [..., 3] — parity-odd → ψ_R
    im_c = color.imag.float()  # [..., 3] — parity-even → ψ_L

    xi_L, valid_L = vector_to_weyl_spinor(im_c, eps=eps)  # [..., 2]
    xi_R, valid_R = vector_to_weyl_spinor(re_c, eps=eps)  # [..., 2]

    # Stack into Dirac spinor: ψ = (ξ_L, ξ_R)ᵀ
    spinor = torch.cat([xi_L, xi_R], dim=-1)  # [..., 4]
    valid = valid_L & valid_R

    return spinor.to(torch.complex128), valid


# ===========================================================================
# Dirac bilinear operators
# ===========================================================================


@dataclass
class DiracOperatorSeries:
    """Operator time series for all Dirac bilinear channels.

    Each field is [T] — the spatially averaged operator value at each MC timestep.
    """

    scalar: Tensor        # ψ̄ψ = ψ†γ₀ψ
    pseudoscalar: Tensor  # ψ̄γ₅ψ
    vector: Tensor        # (1/3)Σ_k ψ̄γ_k ψ  (averaged over spatial directions)
    axial_vector: Tensor  # (1/3)Σ_k ψ̄γ₅γ_k ψ
    tensor: Tensor        # (1/6)Σ_{μ<ν} ψ̄σ_μν ψ

    # Diagnostics
    n_valid_pairs: Tensor  # [T] number of valid (i,j) pairs per frame
    spinor_valid_fraction: Tensor  # [T] fraction of walkers with valid spinors


def compute_dirac_bilinear(
    psi_i: Tensor,
    psi_j: Tensor,
    gamma0: Tensor,
    Gamma: Tensor,
) -> Tensor:
    """Compute ψ̄_i Γ ψ_j = ψ_i† γ₀ Γ ψ_j for a batch of pairs.

    Args:
        psi_i: Dirac spinors at site i [..., 4].
        psi_j: Dirac spinors at site j [..., 4].
        gamma0: γ₀ matrix [4, 4].
        Gamma: Dirac matrix Γ [4, 4] or [n, 4, 4] for multiple components.

    Returns:
        Bilinear values [...] (real) if Gamma is [4,4],
        or [..., n] if Gamma is [n, 4, 4].
    """
    # ψ̄_i = ψ_i† γ₀  →  (ψ̄_i)_b = Σ_a (ψ_i^a)* (γ₀)_{ab}
    # ψ̄_i Γ ψ_j = Σ_{ab} (ψ̄_i)_a Γ_{ab} (ψ_j)_b
    #            = Σ_{abc} (ψ_i^c)* (γ₀)_{ca} Γ_{ab} (ψ_j)_b
    #            = Σ_{cb} (ψ_i^c)* M_{cb} (ψ_j)_b   where M = γ₀ Γ

    if Gamma.ndim == 2:
        # Single Gamma: M = γ₀ Γ [4, 4]
        M = gamma0 @ Gamma
        # ψ_i†  M  ψ_j  =  Σ_{ab} (ψ_i^a)* M_{ab} ψ_j^b
        return torch.einsum("...a,ab,...b->...", psi_i.conj(), M, psi_j).real
    else:
        # Multiple Gammas: Gamma is [n, 4, 4], M = γ₀ Γ_n [n, 4, 4]
        M = torch.einsum("ab,nbc->nac", gamma0, Gamma)  # [n, 4, 4]
        # ψ̄_i Γ_n ψ_j = Σ_{ab} (ψ_i^a)* M_n_{ab} ψ_j^b
        return torch.einsum("...a,nab,...b->...n", psi_i.conj(), M, psi_j).real


def compute_dirac_operators_from_spinors(
    spinor: Tensor,
    spinor_valid: Tensor,
    sample_indices: Tensor,
    neighbor_indices: Tensor,
    alive: Tensor,
    gamma: dict[str, Tensor],
    weights: Tensor | None = None,
) -> DiracOperatorSeries:
    """Compute all Dirac bilinear operator time series.

    Args:
        spinor: Dirac spinors [T, N, 4] (complex).
        spinor_valid: Validity mask [T, N] (bool).
        sample_indices: Sampled walker indices [T, S].
        neighbor_indices: Neighbor indices [T, S, k].
        alive: Alive mask [T, N] (bool).
        gamma: Gamma matrices dict from build_dirac_gamma_matrices().
        weights: Optional per-sample weights [T, S] for Riemannian measure.

    Returns:
        DiracOperatorSeries with all channel time series.
    """
    T, _N, _ = spinor.shape
    S = sample_indices.shape[1]
    device = spinor.device

    gamma0 = gamma["gamma0"].to(device)
    gamma_k = gamma["gamma_k"].to(device)
    gamma5 = gamma["gamma5"].to(device)
    gamma5_k = gamma["gamma5_k"].to(device)
    sigma_munu = gamma["sigma_munu"].to(device)

    I4 = torch.eye(4, device=device, dtype=gamma0.dtype)

    # Gather spinors for sample pairs
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, S)
    psi_i = spinor[t_idx, sample_indices]  # [T, S, 4]

    first_neighbor = neighbor_indices[:, :, 0]  # [T, S]
    psi_j = spinor[t_idx, first_neighbor]  # [T, S, 4]

    # Validity
    valid_i = (
        spinor_valid[t_idx, sample_indices]
        & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
    )
    valid_j = (
        spinor_valid[t_idx, first_neighbor]
        & alive[t_idx.clamp(max=alive.shape[0] - 1), first_neighbor]
    )
    valid_mask = valid_i & valid_j & (first_neighbor != sample_indices)

    # Compute bilinears
    # Scalar: ψ̄ψ = ψ†γ₀ψ (Γ = I₄)
    op_scalar = compute_dirac_bilinear(psi_i, psi_j, gamma0, I4)  # [T, S]

    # Pseudoscalar: ψ̄γ₅ψ
    op_pseudo = compute_dirac_bilinear(psi_i, psi_j, gamma0, gamma5)  # [T, S]

    # Vector: (1/3)Σ_k ψ̄γ_k ψ
    op_vector_k = compute_dirac_bilinear(psi_i, psi_j, gamma0, gamma_k)  # [T, S, 3]
    op_vector = op_vector_k.mean(dim=-1)  # [T, S]

    # Axial vector: (1/3)Σ_k ψ̄γ₅γ_k ψ
    op_axial_k = compute_dirac_bilinear(psi_i, psi_j, gamma0, gamma5_k)  # [T, S, 3]
    op_axial = op_axial_k.mean(dim=-1)  # [T, S]

    # Tensor: (1/6)Σ_{μ<ν} ψ̄σ_μν ψ
    op_tensor_mn = compute_dirac_bilinear(psi_i, psi_j, gamma0, sigma_munu)  # [T, S, 6]
    op_tensor = op_tensor_mn.mean(dim=-1)  # [T, S]

    # Mask invalid pairs
    zero = torch.zeros_like(op_scalar)
    op_scalar = torch.where(valid_mask, op_scalar, zero)
    op_pseudo = torch.where(valid_mask, op_pseudo, zero)
    op_vector = torch.where(valid_mask, op_vector, zero)
    op_axial = torch.where(valid_mask, op_axial, zero)
    op_tensor = torch.where(valid_mask, op_tensor, zero)

    # Weighted average over samples
    if weights is not None:
        w = weights.to(device=device, dtype=op_scalar.dtype)
        w = torch.where(valid_mask, w, zero)
    else:
        w = valid_mask.float()

    w_sum = w.sum(dim=1).clamp(min=1e-12)  # [T]
    n_valid = valid_mask.sum(dim=1)

    def _avg(op: Tensor) -> Tensor:
        return (op * w).sum(dim=1) / w_sum

    return DiracOperatorSeries(
        scalar=_avg(op_scalar),
        pseudoscalar=_avg(op_pseudo),
        vector=_avg(op_vector),
        axial_vector=_avg(op_axial),
        tensor=_avg(op_tensor),
        n_valid_pairs=n_valid,
        spinor_valid_fraction=(spinor_valid & alive).float().mean(dim=1),
    )


# ===========================================================================
# Integration with RunHistory / aggregation pipeline
# ===========================================================================


def compute_dirac_operator_series(
    color: Tensor,
    color_valid: Tensor,
    sample_indices: Tensor,
    neighbor_indices: Tensor,
    alive: Tensor,
    sample_edge_weights: Tensor | None = None,
) -> DiracOperatorSeries:
    """Compute Dirac operator series from pre-aggregated color data.

    This is the drop-in replacement for the individual compute_*_operators()
    functions in aggregation.py. It takes the same inputs (from
    AggregatedTimeSeries) and returns all channels at once.

    Args:
        color: Complex color states [T, N, d] where d=3.
        color_valid: Color validity mask [T, N].
        sample_indices: Sample walker indices [T, S].
        neighbor_indices: Neighbor indices [T, S, k].
        alive: Alive mask [T, N].
        sample_edge_weights: Optional Riemannian edge weights [T, S].

    Returns:
        DiracOperatorSeries with all meson channels.
    """
    T, N, d = color.shape
    device = color.device

    if d != 3:
        raise ValueError(
            f"Dirac spinor construction requires d=3, got d={d}. "
            "For d≠3, use the direct Re/Im operators instead."
        )

    # Build gamma matrices
    gamma = build_dirac_gamma_matrices(device=device)

    # Convert color → Dirac spinor
    spinor, spinor_valid = color_to_dirac_spinor(color)  # [T, N, 4], [T, N]

    # Combine validity
    spinor_valid = spinor_valid & color_valid

    return compute_dirac_operators_from_spinors(
        spinor=spinor,
        spinor_valid=spinor_valid,
        sample_indices=sample_indices,
        neighbor_indices=neighbor_indices,
        alive=alive,
        gamma=gamma,
        weights=sample_edge_weights,
    )


def compute_dirac_operators_from_agg(
    agg_data: "AggregatedTimeSeries",
) -> DiracOperatorSeries:
    """Compute Dirac operators from an AggregatedTimeSeries.

    Convenience wrapper matching the aggregation.py API.

    Args:
        agg_data: AggregatedTimeSeries from aggregate_time_series().

    Returns:
        DiracOperatorSeries.
    """
    return compute_dirac_operator_series(
        color=agg_data.color,
        color_valid=agg_data.color_valid,
        sample_indices=agg_data.sample_indices,
        neighbor_indices=agg_data.neighbor_indices,
        alive=agg_data.alive,
        sample_edge_weights=agg_data.sample_edge_weights,
    )
