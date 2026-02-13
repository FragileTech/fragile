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


SU2_BASE_OPERATOR_CHANNELS = (
    "su2_phase",
    "su2_component",
    "su2_doublet",
    "su2_doublet_diff",
)
SU2_DIRECTIONAL_OPERATOR_CHANNELS = tuple(
    f"{name}_directed" for name in SU2_BASE_OPERATOR_CHANNELS
)
WALKER_TYPE_LABELS = ("cloner", "resister", "persister")
SU2_WALKER_TYPE_CHANNELS = tuple(
    f"{name}_{label}" for name in SU2_BASE_OPERATOR_CHANNELS for label in WALKER_TYPE_LABELS
)
ELECTROWEAK_OPERATOR_CHANNELS = (
    "u1_phase",
    "u1_dressed",
    "u1_phase_q2",
    "u1_dressed_q2",
    *SU2_BASE_OPERATOR_CHANNELS,
    *SU2_DIRECTIONAL_OPERATOR_CHANNELS,
    *SU2_WALKER_TYPE_CHANNELS,
    "ew_mixed",
)


def _resolve_su2_operator_mode(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"standard", "score_directed"}:
        msg = "su2_operator_mode must be 'standard' or 'score_directed'."
        raise ValueError(msg)
    return mode_norm


def _resolve_walker_type_scope(scope: str) -> str:
    scope_norm = str(scope).strip().lower()
    if scope_norm != "frame_global":
        msg = "walker_type_scope must be 'frame_global'."
        raise ValueError(msg)
    return scope_norm


def classify_walker_types(
    fitness: Tensor,
    alive: Tensor,
    will_clone: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Classify alive walkers into (cloner, resister, persister) masks.

    Definitions:
    - cloner: alive and Bernoulli cloning event is True.
    - resister: alive, not cloner, and lower fitness than at least one other alive walker.
    - persister: all remaining alive walkers.
    """
    if fitness.ndim != 1 or alive.ndim != 1 or fitness.shape[0] != alive.shape[0]:
        msg = "fitness and alive must be aligned 1D tensors [N]."
        raise ValueError(msg)
    n_walkers = int(fitness.shape[0])
    if n_walkers == 0:
        empty = torch.zeros(0, device=fitness.device, dtype=torch.bool)
        return empty, empty, empty

    alive_b = alive.to(device=fitness.device, dtype=torch.bool)
    if will_clone is None:
        clone_b = torch.zeros_like(alive_b)
    else:
        clone_b = will_clone.to(device=fitness.device, dtype=torch.bool)
        if clone_b.ndim != 1 or clone_b.shape[0] != n_walkers:
            msg = "will_clone must be a 1D tensor [N] aligned with fitness/alive."
            raise ValueError(msg)

    cloner = alive_b & clone_b

    n_alive = int(alive_b.sum().item())
    if n_alive <= 1:
        resister = torch.zeros_like(alive_b)
    else:
        neg_inf = torch.full_like(fitness, float("-inf"))
        alive_fitness = torch.where(alive_b, fitness, neg_inf)
        max_alive = alive_fitness.max()
        has_fitter_peer = alive_b & torch.isfinite(max_alive) & (fitness < max_alive)
        resister = alive_b & (~clone_b) & has_fitter_peer

    persister = alive_b & (~cloner) & (~resister)
    return cloner, resister, persister


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

    in_bounds = (src >= 0) & (src < n_walkers) & (dst >= 0) & (dst < n_walkers) & (src != dst)
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
        padded_w = torch.where(
            top_valid, torch.gather(padded_w, 1, top_idx), torch.zeros_like(top_score)
        )
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
    neighbors: PackedNeighbors | None = None,
    *,
    h_eff: float,
    epsilon_d: float,
    epsilon_c: float,
    epsilon_clone: float,
    lambda_alg: float,
    bounds: Any | None = None,
    pbc: bool = False,
    will_clone: Tensor | None = None,
    su2_operator_mode: str = "standard",
    enable_walker_type_split: bool = False,
    walker_type_scope: str = "frame_global",
    neighbors_u1: PackedNeighbors | None = None,
    neighbors_su2: PackedNeighbors | None = None,
    neighbors_ew_mixed: PackedNeighbors | None = None,
) -> dict[str, Tensor]:
    """Compute all electroweak operators in vectorized form.

    Args:
        positions: Walker positions [N, d].
        velocities: Walker velocities [N, d].
        fitness: Fitness values [N].
        alive: Alive mask [N].
        neighbors: Optional shared packed neighbor indices/weights for all families.
        h_eff: Effective Planck constant.
        epsilon_d: U(1) locality scale.
        epsilon_c: SU(2) locality scale.
        epsilon_clone: Cloning-score regularizer.
        lambda_alg: Velocity contribution to algorithmic distance.
        bounds: Optional bounds for PBC correction.
        pbc: Whether to apply periodic boundary conditions.
        will_clone: Optional Bernoulli cloning outcomes [N].
        su2_operator_mode: Either ``standard`` or ``score_directed``.
        enable_walker_type_split: Whether to emit cloner/resister/persister SU(2) channels.
        walker_type_scope: Walker classification scope (currently ``frame_global`` only).
        neighbors_u1: Optional U(1)-family packed neighbors.
        neighbors_su2: Optional SU(2)-family packed neighbors.
        neighbors_ew_mixed: Optional EW-mixed-family packed neighbors.

    Returns:
        Dict mapping channel name to per-walker complex operator [N].
    """
    resolved_mode = _resolve_su2_operator_mode(su2_operator_mode)
    _resolve_walker_type_scope(walker_type_scope)

    N = positions.shape[0]
    device = positions.device
    complex_dtype = torch.complex128 if positions.dtype == torch.float64 else torch.complex64

    if N == 0:
        zeros = torch.zeros(N, device=device, dtype=complex_dtype)
        return {name: zeros.clone() for name in ELECTROWEAK_OPERATOR_CHANNELS}

    if (
        neighbors is None
        and neighbors_u1 is None
        and neighbors_su2 is None
        and neighbors_ew_mixed is None
    ):
        msg = "At least one neighbor set must be provided for electroweak operators."
        raise ValueError(msg)

    neighbors_u1 = neighbors_u1 if neighbors_u1 is not None else neighbors
    neighbors_su2 = neighbors_su2 if neighbors_su2 is not None else neighbors
    neighbors_ew_mixed = neighbors_ew_mixed if neighbors_ew_mixed is not None else neighbors
    if neighbors_u1 is None or neighbors_su2 is None or neighbors_ew_mixed is None:
        msg = "U(1), SU(2), and EW-mixed neighbor sets must be resolvable."
        raise ValueError(msg)

    for label, packed in (
        ("neighbors_u1", neighbors_u1),
        ("neighbors_su2", neighbors_su2),
        ("neighbors_ew_mixed", neighbors_ew_mixed),
    ):
        if int(packed.indices.shape[0]) != N:
            msg = (
                f"{label} walker axis mismatch: expected {N}, got {int(packed.indices.shape[0])}."
            )
            raise ValueError(msg)

    zeros_complex = torch.zeros(N, device=device, dtype=complex_dtype)
    zeros_real = torch.zeros(N, device=device, dtype=positions.dtype)

    def _compute_family_terms(packed: PackedNeighbors) -> dict[str, Tensor]:
        if packed.indices.numel() == 0:
            return {
                "has_neighbors": torch.zeros(N, device=device, dtype=torch.bool),
                "u1_phase_exp": zeros_complex.clone(),
                "u1_phase_q2_exp": zeros_complex.clone(),
                "u1_amp": zeros_real.clone(),
                "su2_phase_exp_standard": zeros_complex.clone(),
                "su2_phase_exp_directed": zeros_complex.clone(),
                "su2_amp": zeros_real.clone(),
                "su2_comp_phase_exp_standard": zeros_complex.clone(),
                "su2_comp_phase_exp_directed": zeros_complex.clone(),
                "su2_comp_amp": zeros_real.clone(),
            }

        nbr_idx = packed.indices
        valid = packed.valid.to(device=device, dtype=torch.bool)
        neighbor_w = packed.weights.to(device=device, dtype=positions.dtype)
        weights = torch.where(valid, neighbor_w, torch.zeros_like(neighbor_w))
        weight_sum = weights.sum(dim=1, keepdim=True)
        weights = torch.where(
            weight_sum > 0,
            weights / weight_sum.clamp(min=1e-12),
            torch.zeros_like(weights),
        )
        has_neighbors = valid.any(dim=1)

        fitness_i = fitness.unsqueeze(1)
        fitness_j = fitness[nbr_idx]
        weights_c = weights.to(complex_dtype)

        phase_u1 = -(fitness_j - fitness_i) / max(h_eff, 1e-12)
        u1_phase_exp = (weights_c * torch.exp(1j * phase_u1).to(complex_dtype)).sum(dim=1)
        u1_phase_q2_exp = (weights_c * torch.exp(1j * (2.0 * phase_u1)).to(complex_dtype)).sum(
            dim=1
        )

        denom = fitness_i + epsilon_clone
        denom = torch.where(
            denom.abs() < 1e-12,
            torch.full_like(denom, float(max(epsilon_clone, 1e-12))),
            denom,
        )
        su2_phase = ((fitness_j - fitness_i) / denom) / max(h_eff, 1e-12)
        su2_phase_pair = torch.exp(1j * su2_phase).to(complex_dtype)
        su2_phase_pair_directed = torch.where(
            su2_phase >= 0, su2_phase_pair, torch.conj(su2_phase_pair)
        )
        su2_phase_exp_standard = (weights_c * su2_phase_pair).sum(dim=1)
        su2_phase_exp_directed = (weights_c * su2_phase_pair_directed).sum(dim=1)

        diff_x = positions.unsqueeze(1) - positions[nbr_idx]
        if pbc and bounds is not None:
            diff_x = _apply_pbc_diff(diff_x, bounds)
        diff_v = velocities.unsqueeze(1) - velocities[nbr_idx]
        dist_sq = (diff_x**2).sum(dim=-1) + float(lambda_alg) * (diff_v**2).sum(dim=-1)
        u1_weight = torch.exp(-dist_sq / (2.0 * max(float(epsilon_d), 1e-12) ** 2))
        su2_weight = torch.exp(-dist_sq / (2.0 * max(float(epsilon_c), 1e-12) ** 2))
        u1_amp = (weights * torch.sqrt(u1_weight)).sum(dim=1)
        su2_amp = (weights * torch.sqrt(su2_weight)).sum(dim=1)

        su2_comp_phase_exp_standard = (weights_c * su2_phase_exp_standard[nbr_idx]).sum(dim=1)
        su2_comp_phase_exp_directed = (weights_c * su2_phase_exp_directed[nbr_idx]).sum(dim=1)
        su2_comp_amp = (weights * su2_amp[nbr_idx]).sum(dim=1)

        return {
            "has_neighbors": has_neighbors,
            "u1_phase_exp": u1_phase_exp,
            "u1_phase_q2_exp": u1_phase_q2_exp,
            "u1_amp": u1_amp,
            "su2_phase_exp_standard": su2_phase_exp_standard,
            "su2_phase_exp_directed": su2_phase_exp_directed,
            "su2_amp": su2_amp,
            "su2_comp_phase_exp_standard": su2_comp_phase_exp_standard,
            "su2_comp_phase_exp_directed": su2_comp_phase_exp_directed,
            "su2_comp_amp": su2_comp_amp,
        }

    u1_terms = _compute_family_terms(neighbors_u1)
    su2_terms = _compute_family_terms(neighbors_su2)
    ew_terms = _compute_family_terms(neighbors_ew_mixed)

    alive_c = alive.to(complex_dtype)
    u1_amp_c = u1_terms["u1_amp"].to(complex_dtype)
    su2_amp_c = su2_terms["su2_amp"].to(complex_dtype)
    su2_comp_amp_c = su2_terms["su2_comp_amp"].to(complex_dtype)

    su2_phase_exp_standard = su2_terms["su2_phase_exp_standard"]
    su2_phase_exp_directed = su2_terms["su2_phase_exp_directed"]
    su2_comp_phase_exp_standard = su2_terms["su2_comp_phase_exp_standard"]
    su2_comp_phase_exp_directed = su2_terms["su2_comp_phase_exp_directed"]

    su2_phase_op_standard = su2_phase_exp_standard * alive_c
    su2_component_op_standard = (su2_amp_c * su2_phase_exp_standard) * alive_c
    su2_doublet_op_standard = (
        su2_amp_c * su2_phase_exp_standard + su2_comp_amp_c * su2_comp_phase_exp_standard
    ) * alive_c
    su2_doublet_diff_op_standard = (
        su2_amp_c * su2_phase_exp_standard - su2_comp_amp_c * su2_comp_phase_exp_standard
    ) * alive_c

    su2_phase_op_directed = su2_phase_exp_directed * alive_c
    su2_component_op_directed = (su2_amp_c * su2_phase_exp_directed) * alive_c
    su2_doublet_op_directed = (
        su2_amp_c * su2_phase_exp_directed + su2_comp_amp_c * su2_comp_phase_exp_directed
    ) * alive_c
    su2_doublet_diff_op_directed = (
        su2_amp_c * su2_phase_exp_directed - su2_comp_amp_c * su2_comp_phase_exp_directed
    ) * alive_c

    if resolved_mode == "score_directed":
        su2_phase_op = su2_phase_op_directed
        su2_component_op = su2_component_op_directed
        su2_doublet_op = su2_doublet_op_directed
        su2_doublet_diff_op = su2_doublet_diff_op_directed
    else:
        su2_phase_op = su2_phase_op_standard
        su2_component_op = su2_component_op_standard
        su2_doublet_op = su2_doublet_op_standard
        su2_doublet_diff_op = su2_doublet_diff_op_standard

    ew_u1_amp_c = ew_terms["u1_amp"].to(complex_dtype)
    ew_su2_amp_c = ew_terms["su2_amp"].to(complex_dtype)
    ew_u1_phase = ew_terms["u1_phase_exp"]
    if resolved_mode == "score_directed":
        ew_su2_phase = ew_terms["su2_phase_exp_directed"]
    else:
        ew_su2_phase = ew_terms["su2_phase_exp_standard"]

    results: dict[str, Tensor] = {
        "u1_phase": u1_terms["u1_phase_exp"] * alive_c,
        "u1_dressed": (u1_amp_c * u1_terms["u1_phase_exp"]) * alive_c,
        "u1_phase_q2": u1_terms["u1_phase_q2_exp"] * alive_c,
        "u1_dressed_q2": (u1_amp_c * u1_terms["u1_phase_q2_exp"]) * alive_c,
        "su2_phase": su2_phase_op,
        "su2_component": su2_component_op,
        "su2_doublet": su2_doublet_op,
        "su2_doublet_diff": su2_doublet_diff_op,
        "su2_phase_directed": su2_phase_op_directed,
        "su2_component_directed": su2_component_op_directed,
        "su2_doublet_directed": su2_doublet_op_directed,
        "su2_doublet_diff_directed": su2_doublet_diff_op_directed,
        "ew_mixed": (ew_u1_amp_c * ew_su2_amp_c * ew_u1_phase * ew_su2_phase) * alive_c,
    }

    if bool(enable_walker_type_split):
        if will_clone is not None:
            will_clone_b = will_clone.to(device=device, dtype=torch.bool)
            if will_clone_b.ndim != 1 or will_clone_b.shape[0] != N:
                msg = (
                    "will_clone must be aligned with walker axis [N] when splitting walker types."
                )
                raise ValueError(msg)
        else:
            will_clone_b = torch.zeros(N, device=device, dtype=torch.bool)

        cloner_mask, resister_mask, persister_mask = classify_walker_types(
            fitness=fitness.to(device=device),
            alive=alive.to(device=device, dtype=torch.bool),
            will_clone=will_clone_b,
        )
    else:
        false_mask = torch.zeros(N, device=device, dtype=torch.bool)
        cloner_mask = false_mask
        resister_mask = false_mask
        persister_mask = false_mask

    cloner_c = cloner_mask.to(complex_dtype)
    resister_c = resister_mask.to(complex_dtype)
    persister_c = persister_mask.to(complex_dtype)
    results["su2_phase_cloner"] = su2_phase_op * cloner_c
    results["su2_phase_resister"] = su2_phase_op * resister_c
    results["su2_phase_persister"] = su2_phase_op * persister_c
    results["su2_component_cloner"] = su2_component_op * cloner_c
    results["su2_component_resister"] = su2_component_op * resister_c
    results["su2_component_persister"] = su2_component_op * persister_c
    results["su2_doublet_cloner"] = su2_doublet_op * cloner_c
    results["su2_doublet_resister"] = su2_doublet_op * resister_c
    results["su2_doublet_persister"] = su2_doublet_op * persister_c
    results["su2_doublet_diff_cloner"] = su2_doublet_diff_op * cloner_c
    results["su2_doublet_diff_resister"] = su2_doublet_diff_op * resister_c
    results["su2_doublet_diff_persister"] = su2_doublet_diff_op * persister_c

    zeros = torch.zeros(N, device=device, dtype=complex_dtype)
    for name in ELECTROWEAK_OPERATOR_CHANNELS:
        results.setdefault(name, zeros)

    # Keep split channels zeroed for walkers without any valid SU(2) neighbors.
    if not bool(su2_terms["has_neighbors"].any()):
        for name in SU2_WALKER_TYPE_CHANNELS:
            results[name] = zeros

    return results


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
    gaps = (
        values_sorted[1:] - values_sorted[:-1]
        if values_sorted.numel() > 1
        else values_sorted.new_zeros(0)
    )
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
    vel = (
        velocities.detach().cpu().numpy()
        if torch.is_tensor(velocities)
        else np.asarray(velocities)
    )
    fit = fitness.detach().cpu().numpy() if torch.is_tensor(fitness) else np.asarray(fitness)
    alive = (
        alive_mask.detach().cpu().numpy()
        if torch.is_tensor(alive_mask)
        else np.asarray(alive_mask)
    )
    alive = alive.astype(bool)

    alive_indices = np.where(alive)[0]
    if alive_indices.size == 0:
        return np.zeros((0, 0), dtype=np.complex128), alive_indices

    x = pos[alive]
    v = vel[alive]
    V = fit[alive].astype(np.float64)

    dx = x[:, None, :] - x[None, :, :]
    if pbc and bounds is not None:
        high = (
            bounds.high.detach().cpu().numpy()
            if torch.is_tensor(bounds.high)
            else np.asarray(bounds.high)
        )
        low = (
            bounds.low.detach().cpu().numpy()
            if torch.is_tensor(bounds.low)
            else np.asarray(bounds.low)
        )
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
