"""Companion triplet and pair index builders.

Ported from fragile.fractalai.qft.baryon_triplet_channels and
fragile.fractalai.qft.meson_phase_channels.
"""

from __future__ import annotations

import torch
from torch import Tensor


PAIR_SELECTION_MODES = ("distance", "clone", "both")


def build_companion_triplets(
    companions_distance: Tensor,
    companions_clone: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build companion triplet indices (i, j, k) and structural-valid mask.

    Args:
        companions_distance: Companion distance indices [T, N].
        companions_clone: Companion clone indices [T, N].

    Returns:
        Tuple:
            anchor_idx: [T, N] anchor indices i
            companion_j: [T, N] distance companion j
            companion_k: [T, N] clone companion k
            structural_valid: [T, N] index-range/distinctness validity
    """
    if companions_distance.shape != companions_clone.shape:
        raise ValueError(
            "companions_distance and companions_clone must have the same shape, got "
            f"{tuple(companions_distance.shape)} vs {tuple(companions_clone.shape)}."
        )
    if companions_distance.ndim != 2:
        raise ValueError(
            f"Expected companion arrays with shape [T, N], got {tuple(companions_distance.shape)}."
        )

    t, n = companions_distance.shape
    device = companions_distance.device
    anchor_idx = torch.arange(n, device=device, dtype=torch.long).view(1, n).expand(t, n)
    companion_j = companions_distance.to(torch.long)
    companion_k = companions_clone.to(torch.long)

    in_range_j = (companion_j >= 0) & (companion_j < n)
    in_range_k = (companion_k >= 0) & (companion_k < n)
    distinct = (
        (companion_j != anchor_idx) & (companion_k != anchor_idx) & (companion_j != companion_k)
    )
    structural_valid = in_range_j & in_range_k & distinct
    return anchor_idx, companion_j, companion_k, structural_valid


def build_companion_pair_indices(
    companions_distance: Tensor,
    companions_clone: Tensor,
    pair_selection: str = "both",
) -> tuple[Tensor, Tensor]:
    """Build companion pair indices [T,N,P] and structural validity mask.

    Args:
        companions_distance: Distance companion indices [T, N].
        companions_clone: Clone companion indices [T, N].
        pair_selection: One of {"distance", "clone", "both"}.

    Returns:
        pair_indices: Companion indices [T, N, P].
        structural_valid: In-range and non-self mask [T, N, P].
    """
    mode = str(pair_selection).strip().lower()
    if mode not in PAIR_SELECTION_MODES:
        raise ValueError(f"pair_selection must be one of {PAIR_SELECTION_MODES}.")
    anchor_idx, companion_j, companion_k, _ = build_companion_triplets(
        companions_distance=companions_distance,
        companions_clone=companions_clone,
    )
    n = companions_distance.shape[1]
    valid_j = (companion_j >= 0) & (companion_j < n) & (companion_j != anchor_idx)
    valid_k = (companion_k >= 0) & (companion_k < n) & (companion_k != anchor_idx)
    if mode == "distance":
        return companion_j.unsqueeze(-1), valid_j.unsqueeze(-1)
    if mode == "clone":
        return companion_k.unsqueeze(-1), valid_k.unsqueeze(-1)
    pair_indices = torch.stack([companion_j, companion_k], dim=-1)
    structural = torch.stack([valid_j, valid_k], dim=-1)
    return pair_indices, structural
