"""Vectorized electroweak observables for Fractal Gas QFT analysis.

This module centralizes tensorized electroweak computations so channels,
Dirac-spectrum analysis, and auxiliary Higgs/Yukawa diagnostics can reuse
the same high-throughput kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


@dataclass
class PackedNeighbors:
    """Packed ragged neighbor data.

    Attributes:
        indices: Neighbor indices padded to [N, K].
        weights: Neighbor weights padded to [N, K].
        valid: Boolean mask indicating valid entries in padded tensors.
    """

    indices: Tensor
    weights: Tensor
    valid: Tensor


def _apply_pbc_diff(diff: Tensor, bounds: Any | None) -> Tensor:
    """Apply minimum-image convention to a difference tensor."""
    if bounds is None:
        return diff
    high = bounds.high.to(diff)
    low = bounds.low.to(diff)
    span = high - low
    return diff - span * torch.round(diff / span)


def pack_neighbor_lists(
    neighbor_indices: list[Tensor],
    neighbor_weights: list[Tensor],
    n_walkers: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PackedNeighbors:
    """Pack per-walker ragged neighbor lists into padded tensors.

    Args:
        neighbor_indices: Length-N list of tensors with neighbor indices.
        neighbor_weights: Length-N list of tensors with neighbor weights.
        n_walkers: Number of walkers N.
        device: Target device for packed tensors.
        dtype: Floating dtype for packed weights.

    Returns:
        PackedNeighbors with tensors of shape [N, K], where K is the
        maximum per-walker neighbor count.
    """
    if n_walkers <= 0:
        empty_idx = torch.zeros((0, 0), device=device, dtype=torch.long)
        empty_w = torch.zeros((0, 0), device=device, dtype=dtype)
        empty_valid = torch.zeros((0, 0), device=device, dtype=torch.bool)
        return PackedNeighbors(indices=empty_idx, weights=empty_w, valid=empty_valid)

    empty_idx = torch.empty(0, dtype=torch.long, device=device)
    empty_w = torch.empty(0, dtype=dtype, device=device)

    idx_seq = []
    w_seq = []
    for i in range(n_walkers):
        idx = neighbor_indices[i] if i < len(neighbor_indices) else empty_idx
        w = neighbor_weights[i] if i < len(neighbor_weights) else empty_w
        idx = idx.to(device=device, dtype=torch.long)
        w = w.to(device=device, dtype=dtype)
        if idx.numel() != w.numel():
            m = min(idx.numel(), w.numel())
            idx = idx[:m]
            w = w[:m]
        idx_seq.append(idx)
        w_seq.append(w)

    if not idx_seq:
        empty_idx2 = torch.zeros((n_walkers, 0), device=device, dtype=torch.long)
        empty_w2 = torch.zeros((n_walkers, 0), device=device, dtype=dtype)
        empty_valid2 = torch.zeros((n_walkers, 0), device=device, dtype=torch.bool)
        return PackedNeighbors(indices=empty_idx2, weights=empty_w2, valid=empty_valid2)

    padded_idx = pad_sequence(idx_seq, batch_first=True, padding_value=0)
    padded_w = pad_sequence(w_seq, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([x.numel() for x in idx_seq], device=device, dtype=torch.long)
    if padded_idx.shape[1] == 0:
        valid = torch.zeros((n_walkers, 0), device=device, dtype=torch.bool)
    else:
        cols = torch.arange(padded_idx.shape[1], device=device).unsqueeze(0)
        valid = cols < lengths.unsqueeze(1)

    return PackedNeighbors(indices=padded_idx, weights=padded_w, valid=valid)


def pack_neighbors_from_edges(
    edges: Tensor,
    edge_weights: Tensor,
    alive: Tensor,
    n_walkers: int,
    *,
    max_neighbors: int = 0,
    device: torch.device,
    dtype: torch.dtype,
) -> PackedNeighbors:
    """Pack directed edge list into padded neighbor tensors.

    Args:
        edges: Directed edge tensor [E, 2] with (source, destination).
        edge_weights: Per-edge weights [E].
        alive: Alive mask [N].
        n_walkers: Number of walkers.
        max_neighbors: Optional top-k truncation per source walker.
        device: Target device for output tensors.
        dtype: Floating dtype for weights.

    Returns:
        PackedNeighbors with shape [N, K], where K is max retained degree.
    """
    if n_walkers <= 0:
        empty_idx = torch.zeros((0, 0), device=device, dtype=torch.long)
        empty_w = torch.zeros((0, 0), device=device, dtype=dtype)
        empty_valid = torch.zeros((0, 0), device=device, dtype=torch.bool)
        return PackedNeighbors(indices=empty_idx, weights=empty_w, valid=empty_valid)

    if edges.numel() == 0 or edge_weights.numel() == 0:
        empty_idx2 = torch.zeros((n_walkers, 0), device=device, dtype=torch.long)
        empty_w2 = torch.zeros((n_walkers, 0), device=device, dtype=dtype)
        empty_valid2 = torch.zeros((n_walkers, 0), device=device, dtype=torch.bool)
        return PackedNeighbors(indices=empty_idx2, weights=empty_w2, valid=empty_valid2)

    edges_d = edges.to(device=device, dtype=torch.long)
    src = edges_d[:, 0]
    dst = edges_d[:, 1]
    w = edge_weights.to(device=device, dtype=dtype)
    alive_d = alive.to(device=device, dtype=torch.bool)

    in_bounds = (
        (src >= 0)
        & (src < n_walkers)
        & (dst >= 0)
        & (dst < n_walkers)
        & (src != dst)
    )
    if not in_bounds.any():
        empty_idx3 = torch.zeros((n_walkers, 0), device=device, dtype=torch.long)
        empty_w3 = torch.zeros((n_walkers, 0), device=device, dtype=dtype)
        empty_valid3 = torch.zeros((n_walkers, 0), device=device, dtype=torch.bool)
        return PackedNeighbors(indices=empty_idx3, weights=empty_w3, valid=empty_valid3)

    src = src[in_bounds]
    dst = dst[in_bounds]
    w = w[in_bounds]
    alive_edge = alive_d[src] & alive_d[dst]
    if not alive_edge.any():
        empty_idx3 = torch.zeros((n_walkers, 0), device=device, dtype=torch.long)
        empty_w3 = torch.zeros((n_walkers, 0), device=device, dtype=dtype)
        empty_valid3 = torch.zeros((n_walkers, 0), device=device, dtype=torch.bool)
        return PackedNeighbors(indices=empty_idx3, weights=empty_w3, valid=empty_valid3)
    src = src[alive_edge]
    dst = dst[alive_edge]
    w = w[alive_edge]

    # Sort by source to build row-wise ragged offsets in tensor form.
    order = torch.argsort(src)
    src = src[order]
    dst = dst[order]
    w = w[order]

    counts = torch.bincount(src, minlength=n_walkers)
    max_degree = int(counts.max().item())
    if max_degree == 0:
        empty_idx4 = torch.zeros((n_walkers, 0), device=device, dtype=torch.long)
        empty_w4 = torch.zeros((n_walkers, 0), device=device, dtype=dtype)
        empty_valid4 = torch.zeros((n_walkers, 0), device=device, dtype=torch.bool)
        return PackedNeighbors(indices=empty_idx4, weights=empty_w4, valid=empty_valid4)

    row_offsets = torch.cumsum(counts, dim=0) - counts
    col = torch.arange(src.numel(), device=device, dtype=torch.long) - row_offsets[src]

    padded_idx = torch.zeros((n_walkers, max_degree), device=device, dtype=torch.long)
    padded_w = torch.zeros((n_walkers, max_degree), device=device, dtype=dtype)
    valid = torch.zeros((n_walkers, max_degree), device=device, dtype=torch.bool)
    padded_idx[src, col] = dst
    padded_w[src, col] = w
    valid[src, col] = True

    if max_neighbors > 0 and padded_w.shape[1] > max_neighbors:
        k = min(int(max_neighbors), padded_w.shape[1])
        score = torch.where(valid, padded_w, torch.full_like(padded_w, -torch.inf))
        top_score, top_idx = torch.topk(score, k=k, dim=1, largest=True, sorted=True)
        top_valid = torch.isfinite(top_score)
        padded_idx = torch.gather(padded_idx, 1, top_idx)
        padded_w = torch.where(top_valid, torch.gather(padded_w, 1, top_idx), torch.zeros_like(top_score))
        valid = top_valid

    row_sum = padded_w.sum(dim=1, keepdim=True)
    padded_w = torch.where(
        valid & (row_sum > 0),
        padded_w / row_sum.clamp(min=1e-12),
        torch.zeros_like(padded_w),
    )
    return PackedNeighbors(indices=padded_idx, weights=padded_w, valid=valid)


def compute_weighted_electroweak_ops_vectorized(
    positions: Tensor,
    velocities: Tensor,
    fitness: Tensor,
    alive: Tensor,
    neighbors: PackedNeighbors,
    *,
    h_eff: float,
    epsilon_d: float,
    epsilon_c: float,
    epsilon_clone: float,
    lambda_alg: float,
    bounds: Any | None = None,
    pbc: bool = False,
) -> dict[str, Tensor]:
    """Compute all electroweak operators in vectorized form.

    Args:
        positions: Walker positions [N, d].
        velocities: Walker velocities [N, d].
        fitness: Fitness values [N].
        alive: Alive mask [N].
        neighbors: Packed neighbor indices/weights.
        h_eff: Effective Planck constant.
        epsilon_d: U(1) locality scale.
        epsilon_c: SU(2) locality scale.
        epsilon_clone: Cloning-score regularizer.
        lambda_alg: Velocity contribution to algorithmic distance.
        bounds: Optional bounds for PBC correction.
        pbc: Whether to apply periodic boundary conditions.

    Returns:
        Dict mapping channel name to per-walker complex operator [N].
    """
    N = positions.shape[0]
    device = positions.device
    complex_dtype = torch.complex128 if positions.dtype == torch.float64 else torch.complex64

    if N == 0 or neighbors.indices.numel() == 0:
        zeros = torch.zeros(N, device=device, dtype=complex_dtype)
        return {
            "u1_phase": zeros,
            "u1_dressed": zeros,
            "u1_phase_q2": zeros,
            "u1_dressed_q2": zeros,
            "su2_phase": zeros,
            "su2_component": zeros,
            "su2_doublet": zeros,
            "su2_doublet_diff": zeros,
            "ew_mixed": zeros,
        }

    nbr_idx = neighbors.indices
    valid = neighbors.valid
    neighbor_w = neighbors.weights.to(device=device, dtype=positions.dtype)
    weights = torch.where(valid, neighbor_w, torch.zeros_like(neighbor_w))
    weight_sum = weights.sum(dim=1, keepdim=True)
    weights = torch.where(weight_sum > 0, weights / weight_sum.clamp(min=1e-12), torch.zeros_like(weights))

    fitness_i = fitness.unsqueeze(1)
    fitness_j = fitness[nbr_idx]
    weights_c = weights.to(complex_dtype)

    # U(1) phase channels
    phase_u1 = -(fitness_j - fitness_i) / max(h_eff, 1e-12)
    u1_phase_exp = (weights_c * torch.exp(1j * phase_u1).to(complex_dtype)).sum(dim=1)
    u1_phase_q2_exp = (weights_c * torch.exp(1j * (2.0 * phase_u1)).to(complex_dtype)).sum(dim=1)

    # SU(2) phase channels
    denom = fitness_i + epsilon_clone
    denom = torch.where(
        denom.abs() < 1e-12,
        torch.full_like(denom, float(max(epsilon_clone, 1e-12))),
        denom,
    )
    su2_phase = ((fitness_j - fitness_i) / denom) / max(h_eff, 1e-12)
    su2_phase_exp = (weights_c * torch.exp(1j * su2_phase).to(complex_dtype)).sum(dim=1)

    # Locality amplitudes
    diff_x = positions.unsqueeze(1) - positions[nbr_idx]
    if pbc and bounds is not None:
        diff_x = _apply_pbc_diff(diff_x, bounds)
    diff_v = velocities.unsqueeze(1) - velocities[nbr_idx]
    dist_sq = (diff_x**2).sum(dim=-1) + float(lambda_alg) * (diff_v**2).sum(dim=-1)
    u1_weight = torch.exp(-dist_sq / (2.0 * max(float(epsilon_d), 1e-12) ** 2))
    su2_weight = torch.exp(-dist_sq / (2.0 * max(float(epsilon_c), 1e-12) ** 2))
    u1_amp = (weights * torch.sqrt(u1_weight)).sum(dim=1)
    su2_amp = (weights * torch.sqrt(su2_weight)).sum(dim=1)

    # Companion-composed SU(2) terms
    su2_comp_phase_exp = (weights_c * su2_phase_exp[nbr_idx]).sum(dim=1)
    su2_comp_amp = (weights * su2_amp[nbr_idx]).sum(dim=1)

    alive_c = alive.to(complex_dtype)
    u1_amp_c = u1_amp.to(complex_dtype)
    su2_amp_c = su2_amp.to(complex_dtype)
    su2_comp_amp_c = su2_comp_amp.to(complex_dtype)

    return {
        "u1_phase": u1_phase_exp * alive_c,
        "u1_dressed": (u1_amp_c * u1_phase_exp) * alive_c,
        "u1_phase_q2": u1_phase_q2_exp * alive_c,
        "u1_dressed_q2": (u1_amp_c * u1_phase_q2_exp) * alive_c,
        "su2_phase": su2_phase_exp * alive_c,
        "su2_component": (su2_amp_c * su2_phase_exp) * alive_c,
        "su2_doublet": (su2_amp_c * su2_phase_exp + su2_comp_amp_c * su2_comp_phase_exp) * alive_c,
        "su2_doublet_diff": (su2_amp_c * su2_phase_exp - su2_comp_amp_c * su2_comp_phase_exp) * alive_c,
        "ew_mixed": (u1_amp_c * su2_amp_c * u1_phase_exp * su2_phase_exp) * alive_c,
    }


def compute_higgs_vev_from_positions(
    positions: Tensor,
    alive: Tensor | None = None,
) -> dict[str, Any]:
    """Estimate Higgs-like VEV from radial order parameter statistics.

    Args:
        positions: Walker positions [N, d].
        alive: Optional alive mask [N].

    Returns:
        Dict with center, radii, vev (mean radius), and vev_std.
    """
    if positions.numel() == 0:
        center = torch.zeros(positions.shape[-1], device=positions.device, dtype=positions.dtype)
        radii = torch.zeros(0, device=positions.device, dtype=positions.dtype)
        return {"center": center, "radii": radii, "vev": 0.0, "vev_std": 0.0}

    if alive is not None and alive.any():
        pos = positions[alive]
    else:
        pos = positions
    center = pos.mean(dim=0)
    radii = torch.linalg.vector_norm(pos - center.unsqueeze(0), dim=-1)
    vev = float(radii.mean().item()) if radii.numel() else 0.0
    vev_std = float(radii.std(unbiased=False).item()) if radii.numel() else 0.0
    return {"center": center, "radii": radii, "vev": vev, "vev_std": vev_std}


def compute_fitness_gap_distribution(
    fitness: Tensor,
    alive: Tensor | None = None,
) -> dict[str, Any]:
    """Compute sorted fitness gaps and characteristic scale."""
    if fitness.numel() == 0:
        empty = torch.zeros(0, device=fitness.device, dtype=fitness.dtype)
        return {"fitness_sorted": empty, "gaps": empty, "phi0": 0.0}

    values = fitness[alive] if (alive is not None and alive.any()) else fitness
    if values.numel() == 0:
        empty = torch.zeros(0, device=fitness.device, dtype=fitness.dtype)
        return {"fitness_sorted": empty, "gaps": empty, "phi0": 0.0}

    values_sorted, _ = torch.sort(values)
    gaps = values_sorted[1:] - values_sorted[:-1] if values_sorted.numel() > 1 else values_sorted.new_zeros(0)
    phi0 = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
    return {"fitness_sorted": values_sorted, "gaps": gaps, "phi0": phi0}


def predict_yukawa_mass_from_fitness(
    v_higgs: float,
    fitness: Tensor,
    *,
    alive: Tensor | None = None,
    phi0: float | None = None,
    y0: float = 1.0,
    gap_quantile: float = 0.99,
) -> dict[str, float]:
    """Predict a Yukawa-suppressed fermion mass from fitness-gap statistics."""
    stats = compute_fitness_gap_distribution(fitness, alive=alive)
    gaps = stats["gaps"]
    if gaps.numel() == 0:
        return {
            "mass": 0.0,
            "yukawa": 0.0,
            "delta_phi": 0.0,
            "phi0": float(phi0 if phi0 is not None else stats["phi0"]),
        }

    phi0_val = float(phi0 if phi0 is not None else stats["phi0"])
    phi0_safe = max(phi0_val, 1e-12)
    q = float(min(max(gap_quantile, 0.0), 1.0))
    delta_phi = float(torch.quantile(gaps, q).item())
    yukawa = float(y0 * np.exp(-delta_phi / phi0_safe))
    mass = float(yukawa * max(v_higgs, 0.0))
    return {"mass": mass, "yukawa": yukawa, "delta_phi": delta_phi, "phi0": phi0_val}


def build_phase_space_antisymmetric_kernel(
    positions: np.ndarray | Tensor,
    velocities: np.ndarray | Tensor,
    fitness: np.ndarray | Tensor,
    alive_mask: np.ndarray | Tensor,
    *,
    epsilon_c: float,
    lambda_alg: float = 1.0,
    h_eff: float = 1.0,
    epsilon_clone: float = 1e-8,
    bounds: Any | None = None,
    pbc: bool = False,
    include_phase: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build vectorized antisymmetric cloning kernel in phase space.

    K_ij = exp(-d_alg(i,j)^2 / (2 eps_c^2)) * exp(i theta_ij)
    theta_ij = ((V_j - V_i)/(V_i + eps_clone))/h_eff (if include_phase)
    K_tilde = K - K^T

    Returns:
        K_tilde [N_alive, N_alive] complex matrix and alive indices [N_alive].
    """
    pos = positions.detach().cpu().numpy() if torch.is_tensor(positions) else np.asarray(positions)
    vel = velocities.detach().cpu().numpy() if torch.is_tensor(velocities) else np.asarray(velocities)
    fit = fitness.detach().cpu().numpy() if torch.is_tensor(fitness) else np.asarray(fitness)
    alive = alive_mask.detach().cpu().numpy() if torch.is_tensor(alive_mask) else np.asarray(alive_mask)
    alive = alive.astype(bool)

    alive_indices = np.where(alive)[0]
    if alive_indices.size == 0:
        return np.zeros((0, 0), dtype=np.complex128), alive_indices

    x = pos[alive]
    v = vel[alive]
    V = fit[alive].astype(np.float64)

    dx = x[:, None, :] - x[None, :, :]
    if pbc and bounds is not None:
        high = bounds.high.detach().cpu().numpy() if torch.is_tensor(bounds.high) else np.asarray(bounds.high)
        low = bounds.low.detach().cpu().numpy() if torch.is_tensor(bounds.low) else np.asarray(bounds.low)
        span = high - low
        dx = dx - span * np.round(dx / span)
    dv = v[:, None, :] - v[None, :, :]
    d_alg2 = np.sum(dx * dx, axis=-1) + float(lambda_alg) * np.sum(dv * dv, axis=-1)

    eps_c = max(float(epsilon_c), 1e-12)
    amp = np.exp(-d_alg2 / (2.0 * eps_c * eps_c))
    np.fill_diagonal(amp, 0.0)

    if include_phase:
        Vi = V[:, None]
        Vj = V[None, :]
        denom = np.where(np.abs(Vi + epsilon_clone) < 1e-30, 1e-30, Vi + epsilon_clone)
        score = (Vj - Vi) / denom
        theta = score / max(float(h_eff), 1e-12)
        kernel = amp * np.exp(1j * theta)
    else:
        kernel = amp.astype(np.complex128)

    k_tilde = kernel - kernel.T
    return k_tilde, alive_indices


def antisymmetric_singular_values(
    k_tilde: np.ndarray,
    *,
    top_k: int | None = None,
) -> np.ndarray:
    """Compute singular values of a skew-symmetric kernel via Hermitian eigenspectrum."""
    if k_tilde.size == 0:
        return np.array([])
    ih = 1j * k_tilde
    ih = 0.5 * (ih + ih.conj().T)  # Numerical Hermitian cleanup.
    evals = np.linalg.eigvalsh(ih)
    sigma = np.sort(np.abs(evals))[::-1]
    if top_k is not None:
        sigma = sigma[:top_k]
    return sigma
