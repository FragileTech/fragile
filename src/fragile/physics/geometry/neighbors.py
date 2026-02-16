"""Neighbor utilities for COO/CSR graph conversion and queries and topology computation for Fractal Gas QFT analysis.

This module handles neighbor computation methods:
- Pre-recorded neighbor edges (fastest, O(E))
- Companion walker neighbors (fast, O(N))

Completely decoupled from aggregation pipeline. All functions accept
RunHistory as input and return neighbor topology data structures.

Main Functions:
    - compute_neighbor_topology: Dispatcher for neighbor computation
    - compute_full_neighbor_matrix: For Euclidean time (all walkers)

Usage:
    from fragile.physics.geometry.neighbor_analysis import (
        compute_neighbor_topology,
        compute_full_neighbor_matrix,
    )

    sample_idx, neighbor_idx, alive = compute_neighbor_topology(
        history,
        start_idx=1,
        neighbor_method="recorded",
        neighbor_k=2,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn.functional as F

if TYPE_CHECKING:
    from fragile.physics.fractal_gas.history import RunHistory


def build_csr_from_coo(
    edge_index: Tensor,
    n_nodes: int,
    edge_distances: Tensor | None = None,
    edge_facet_areas: Tensor | None = None,
    edge_types: Tensor | None = None,
) -> dict[str, Tensor]:
    """Convert COO edge_index to CSR format.

    Args:
        edge_index: [2, E] source-destination pairs
        n_nodes: Number of nodes (N for walkers only, N+W if including boundaries)
        edge_distances: Optional [E] distances
        edge_facet_areas: Optional [E] facet areas
        edge_types: Optional [E] edge types (0=walker, 1=boundary)

    Returns:
        Dictionary with:
        - "csr_ptr": [N+1] row pointers
        - "csr_indices": [E] column indices (sorted by source)
        - "csr_distances": [E] distances (if provided)
        - "csr_facet_areas": [E] facet areas (if provided)
        - "csr_types": [E] edge types (if provided)

    Algorithm:
        1. Sort edges by source index
        2. Compute row pointers (cumulative neighbor counts)
        3. Reorder all edge data by sorted order
    """
    device = edge_index.device
    n_edges = edge_index.shape[1]

    if n_edges == 0:
        # Empty graph
        return {
            "csr_ptr": torch.zeros(n_nodes + 1, dtype=torch.long, device=device),
            "csr_indices": torch.empty(0, dtype=torch.long, device=device),
            "csr_distances": (
                torch.empty(0, dtype=torch.float32, device=device)
                if edge_distances is not None
                else None
            ),
            "csr_facet_areas": (
                torch.empty(0, dtype=torch.float32, device=device)
                if edge_facet_areas is not None
                else None
            ),
            "csr_types": (
                torch.empty(0, dtype=torch.long, device=device) if edge_types is not None else None
            ),
        }

    # Sort edges by source index
    sources = edge_index[0]
    sorted_indices = torch.argsort(sources)

    sorted_sources = sources[sorted_indices]
    sorted_targets = edge_index[1, sorted_indices]

    # Compute row pointers (vectorized using bincount)
    # Count how many edges each source node has
    counts = torch.bincount(sorted_sources, minlength=n_nodes)
    # Prepend zero and do cumsum to get CSR pointers
    csr_ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts])
    csr_ptr = torch.cumsum(csr_ptr, dim=0)

    # Reorder edge attributes
    result = {
        "csr_ptr": csr_ptr,
        "csr_indices": sorted_targets,
    }

    if edge_distances is not None:
        result["csr_distances"] = edge_distances[sorted_indices]

    if edge_facet_areas is not None:
        result["csr_facet_areas"] = edge_facet_areas[sorted_indices]

    if edge_types is not None:
        result["csr_types"] = edge_types[sorted_indices]

    return result


def query_walker_neighbors_vectorized(
    csr_ptr: Tensor,
    csr_indices: Tensor,
    csr_types: Tensor | None = None,
    filter_type: int | None = None,
    pad_value: int = -1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Vectorized neighbor query for all walkers using CSR format.

    Args:
        csr_ptr: [N+1] CSR row pointers
        csr_indices: [E] CSR column indices
        csr_types: Optional [E] edge types (0=walker, 1=boundary)
        filter_type: If provided, return only neighbors of this type
            0 = walker neighbors only
            1 = boundary neighbors only
            None = all neighbors
        pad_value: Value used to pad neighbor rows to uniform length

    Returns:
        Tuple of:
        - neighbors: [N, max_degree] padded neighbor indices
        - mask: [N, max_degree] boolean mask for valid neighbors
        - counts: [N] number of neighbors per walker (after filtering)
    """
    device = csr_indices.device
    n_nodes = int(csr_ptr.numel() - 1)
    if n_nodes <= 0:
        empty = torch.empty(0, 0, dtype=csr_indices.dtype, device=device)
        return empty, empty.bool(), torch.empty(0, dtype=torch.long, device=device)

    counts = csr_ptr[1:] - csr_ptr[:-1]
    n_edges = int(csr_indices.numel())

    if n_edges == 0:
        neighbors = torch.empty(n_nodes, 0, dtype=csr_indices.dtype, device=device)
        mask = torch.empty(n_nodes, 0, dtype=torch.bool, device=device)
        return neighbors, mask, torch.zeros(n_nodes, dtype=torch.long, device=device)

    apply_filter = filter_type is not None and csr_types is not None
    row_indices = torch.repeat_interleave(torch.arange(n_nodes, device=device), counts)

    if apply_filter:
        edge_mask = csr_types == filter_type
        if not torch.any(edge_mask):
            neighbors = torch.empty(n_nodes, 0, dtype=csr_indices.dtype, device=device)
            mask = torch.empty(n_nodes, 0, dtype=torch.bool, device=device)
            return neighbors, mask, torch.zeros(n_nodes, dtype=torch.long, device=device)

        filtered_row_indices = row_indices[edge_mask]
        filtered_neighbors = csr_indices[edge_mask]

        edge_mask_int = edge_mask.to(torch.long)
        cum = torch.cumsum(edge_mask_int, dim=0)
        row_start = csr_ptr[:-1]
        row_base = torch.zeros(n_nodes, dtype=cum.dtype, device=device)
        if n_nodes > 1:
            start_indices = row_start[1:]
            safe_indices = torch.clamp(start_indices - 1, min=0)
            row_base[1:] = torch.where(
                start_indices > 0,
                cum[safe_indices],
                torch.zeros_like(start_indices),
            )

        pos_in_row = cum - row_base[row_indices]
        filtered_pos = pos_in_row[edge_mask] - 1

        counts_filtered = torch.bincount(filtered_row_indices, minlength=n_nodes)
        max_degree = int(counts_filtered.max().item()) if counts_filtered.numel() > 0 else 0
        if max_degree == 0:
            neighbors = torch.empty(n_nodes, 0, dtype=csr_indices.dtype, device=device)
            mask = torch.empty(n_nodes, 0, dtype=torch.bool, device=device)
            return neighbors, mask, counts_filtered

        neighbors = torch.full(
            (n_nodes, max_degree),
            pad_value,
            dtype=csr_indices.dtype,
            device=device,
        )
        mask = torch.zeros((n_nodes, max_degree), dtype=torch.bool, device=device)
        neighbors[filtered_row_indices, filtered_pos] = filtered_neighbors
        mask[filtered_row_indices, filtered_pos] = True
        return neighbors, mask, counts_filtered

    max_degree = int(counts.max().item()) if counts.numel() > 0 else 0
    if max_degree == 0:
        neighbors = torch.empty(n_nodes, 0, dtype=csr_indices.dtype, device=device)
        mask = torch.empty(n_nodes, 0, dtype=torch.bool, device=device)
        return neighbors, mask, counts

    edge_pos = torch.arange(n_edges, device=device) - csr_ptr[row_indices]
    neighbors = torch.full(
        (n_nodes, max_degree),
        pad_value,
        dtype=csr_indices.dtype,
        device=device,
    )
    mask = torch.zeros((n_nodes, max_degree), dtype=torch.bool, device=device)
    neighbors[row_indices, edge_pos] = csr_indices
    mask[row_indices, edge_pos] = True
    return neighbors, mask, counts


# =============================================================================
# Primary Neighbor Computation Methods
# =============================================================================


def compute_companion_batch(
    history: RunHistory,
    start_idx: int,
    neighbor_k: int,
    sample_size: int | None = None,
    end_idx: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Use stored companion indices as neighbors.

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        neighbor_k: Number of neighbors per sample.
        sample_size: Number of samples per timestep (None = all walkers).
        end_idx: Ending time index (exclusive). None = use all recorded frames.

    Returns:
        Tuple of (sample_indices [T, S], neighbor_indices [T, S, k], alive [T, N]).
    """
    n_recorded = end_idx if end_idx is not None else history.n_recorded
    T = n_recorded - start_idx
    N = history.N
    k = max(2, int(neighbor_k))
    sample_size = sample_size or N
    device = history.x_final.device

    alive = history.alive_mask[start_idx - 1 : n_recorded - 1]  # [T, N]
    companions_distance = history.companions_distance[start_idx - 1 : n_recorded - 1]
    companions_clone = history.companions_clone[start_idx - 1 : n_recorded - 1]

    all_sample_idx = []
    all_neighbor_idx = []

    for t in range(T):
        alive_t = alive[t]
        alive_indices = torch.where(alive_t)[0]

        if alive_indices.numel() == 0:
            all_sample_idx.append(torch.zeros(sample_size, device=device, dtype=torch.long))
            all_neighbor_idx.append(torch.zeros(sample_size, k, device=device, dtype=torch.long))
            continue

        if alive_indices.numel() <= sample_size:
            sample_idx = alive_indices
        else:
            sample_idx = alive_indices[:sample_size]

        actual_sample_size = sample_idx.numel()
        comp_d = companions_distance[t, sample_idx]
        comp_c = companions_clone[t, sample_idx]
        neighbor_idx = torch.zeros(actual_sample_size, k, device=device, dtype=torch.long)
        neighbor_idx[:, 0] = comp_d
        neighbor_idx[:, 1] = comp_c

        if actual_sample_size < sample_size:
            sample_idx = F.pad(sample_idx, (0, sample_size - actual_sample_size), value=0)
            neighbor_idx = F.pad(
                neighbor_idx, (0, 0, 0, sample_size - actual_sample_size), value=0
            )

        all_sample_idx.append(sample_idx)
        all_neighbor_idx.append(neighbor_idx)

    sample_indices = torch.stack(all_sample_idx, dim=0)  # [T, S]
    neighbor_indices = torch.stack(all_neighbor_idx, dim=0)  # [T, S, k]

    return sample_indices, neighbor_indices, alive


def _edges_to_neighbor_matrix(
    edges: Tensor,
    N: int,
    k: int,
    device: torch.device,
) -> Tensor:
    """Convert edge list to dense neighbor matrix using vectorized scatter.

    Args:
        edges: Edge tensor [E, 2] (src, dst pairs).
        N: Number of nodes.
        k: Number of neighbors to keep per node.
        device: Target device.

    Returns:
        Neighbor matrix [N, k] with -1 for missing neighbors.
    """
    edges_d = edges.to(device)
    src, dst = edges_d[:, 0], edges_d[:, 1]

    # Remove self-loops
    not_self = src != dst
    src = src[not_self]
    dst = dst[not_self]

    if src.numel() == 0:
        return torch.full((N, k), -1, dtype=torch.long, device=device)

    # Sort by src to group neighbors together
    sort_order = torch.argsort(src, stable=True)
    src_sorted = src[sort_order]
    dst_sorted = dst[sort_order]

    # Compute per-source offsets via cumulative count within each group
    degree = torch.bincount(src_sorted, minlength=N)
    max_deg = degree.max().item()

    # Build column indices: for each edge, its position within its source's group
    # Use cumsum trick: count occurrences so far for each source
    group_starts = torch.zeros(N, dtype=torch.long, device=device)
    group_starts[1:] = degree[:-1].cumsum(0)
    col_idx = torch.arange(src_sorted.numel(), device=device) - group_starts[src_sorted]

    # Scatter into dense matrix [N, max_deg]
    neighbor_full = torch.full((N, max(max_deg, k)), -1, dtype=torch.long, device=device)
    # Only write where col_idx < max_deg (always true by construction)
    neighbor_full[src_sorted, col_idx] = dst_sorted

    # Truncate/pad to exactly k columns
    return neighbor_full[:, :k]


def compute_recorded_neighbors_batch(
    history: RunHistory,
    start_idx: int,
    neighbor_k: int,
    sample_size: int | None = None,
    end_idx: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Use recorded neighbor edges from RunHistory.

    Vectorized: converts edge lists to dense neighbor matrices via scatter,
    then gathers for sample indices using advanced indexing.

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        neighbor_k: Number of neighbors per sample.
        sample_size: Number of samples per timestep (None = all walkers).
        end_idx: Ending time index (exclusive). None = use all recorded frames.

    Returns:
        Tuple of (sample_indices [T, S], neighbor_indices [T, S, k], alive [T, N]).
    """
    if history.neighbor_edges is None:
        raise RuntimeError(
            "compute_recorded_neighbors_batch requires history.neighbor_edges, "
            "but it is None. Use compute_companion_batch instead, or set "
            "neighbor_graph_record=True during simulation."
        )

    n_recorded = end_idx if end_idx is not None else history.n_recorded
    T = n_recorded - start_idx
    N = history.N
    k = max(1, int(neighbor_k))
    sample_size = sample_size or N
    device = history.x_final.device

    alive = history.alive_mask[start_idx - 1 : n_recorded - 1]  # [T, N]

    all_sample_idx = []
    all_neighbor_idx = []

    for t in range(T):
        alive_t = alive[t]
        alive_indices = torch.where(alive_t)[0]
        if alive_indices.numel() == 0:
            all_sample_idx.append(torch.zeros(sample_size, device=device, dtype=torch.long))
            all_neighbor_idx.append(torch.zeros(sample_size, k, device=device, dtype=torch.long))
            continue

        if alive_indices.numel() <= sample_size:
            sample_idx = alive_indices
        else:
            sample_idx = alive_indices[:sample_size]

        actual_sample_size = sample_idx.numel()

        record_idx = start_idx + t
        edges = history.neighbor_edges[record_idx]
        if not torch.is_tensor(edges) or edges.numel() == 0:
            # Fallback to self-padding
            neighbor_idx = sample_idx.unsqueeze(1).expand(-1, k).clone()
        else:
            # Vectorized: scatter edges into dense matrix, then gather for samples
            neighbor_matrix = _edges_to_neighbor_matrix(edges, N, k, device)
            neighbor_idx = neighbor_matrix[sample_idx]  # [S, k] advanced indexing

            # Replace -1 (missing neighbors) with self-index
            missing = neighbor_idx < 0
            if missing.any():
                neighbor_idx[missing] = sample_idx.unsqueeze(1).expand_as(neighbor_idx)[missing]

        if actual_sample_size < sample_size:
            sample_idx = F.pad(sample_idx, (0, sample_size - actual_sample_size), value=0)
            neighbor_idx = F.pad(
                neighbor_idx, (0, 0, 0, sample_size - actual_sample_size), value=0
            )

        all_sample_idx.append(sample_idx)
        all_neighbor_idx.append(neighbor_idx)

    sample_indices = torch.stack(all_sample_idx, dim=0)  # [T, S]
    neighbor_indices = torch.stack(all_neighbor_idx, dim=0)  # [T, S, k]

    return sample_indices, neighbor_indices, alive


# =============================================================================
# Dispatcher and Full Matrix Computation
# =============================================================================


def compute_neighbor_topology(
    history: RunHistory,
    start_idx: int,
    neighbor_method: str,
    neighbor_k: int,
    sample_size: int | None = None,
    end_idx: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute neighbor indices for all timesteps.

    Dispatches to companions or recorded based on method.

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        neighbor_method: Neighbor selection method:
            - "recorded": Use history.neighbor_edges
            - "companions": Use history.companions_clone
        neighbor_k: Number of neighbors per sample.
        sample_size: Number of samples per timestep (None = all walkers).
        end_idx: Ending time index (exclusive). None = use all recorded frames.

    Returns:
        Tuple of (sample_indices [T, S], neighbor_indices [T, S, k], alive [T, N]).
    """
    if neighbor_method == "companions":
        return compute_companion_batch(
            history, start_idx, neighbor_k, sample_size, end_idx=end_idx
        )
    if neighbor_method == "recorded":
        return compute_recorded_neighbors_batch(
            history, start_idx, neighbor_k, sample_size, end_idx=end_idx
        )
    raise ValueError(
        f"Unknown neighbor method: {neighbor_method}. "
        f"Valid options: 'recorded', 'companions'"
    )


def compute_full_neighbor_matrix(
    history: RunHistory,
    start_idx: int,
    neighbor_k: int,
    alive: Tensor,
    neighbor_method: str = "companions",
) -> Tensor:
    """Compute neighbor indices for ALL walkers (not just samples).

    Used for Euclidean time mode where we need per-walker operators.
    Delegates to compute_recorded_neighbors_batch or compute_companion_batch
    with sample_size=N, then reshapes to [T, N, k].

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        neighbor_k: Number of neighbors per walker.
        alive: Alive mask [T, N].
        neighbor_method: "recorded" or "companions" method to use.

    Returns:
        Neighbor indices [T, N, k] for all walkers.
    """
    T, N = alive.shape
    k = max(1, int(neighbor_k))

    # Delegate to batch functions with sample_size=N to get all walkers
    try:
        if neighbor_method == "recorded":
            _sample_idx, neighbor_idx, _ = compute_recorded_neighbors_batch(
                history=history,
                start_idx=start_idx,
                neighbor_k=k,
                sample_size=N,
            )
        elif neighbor_method == "companions":
            _sample_idx, neighbor_idx, _ = compute_companion_batch(
                history=history,
                start_idx=start_idx,
                neighbor_k=k,
                sample_size=N,
            )
        else:
            raise ValueError(f"Unknown neighbor method: {neighbor_method}")

        # neighbor_idx is [T, S, k] where S=N — already the right shape
        return neighbor_idx

    except RuntimeError:
        # No neighbor data at all — return self-loops
        device = alive.device
        arange = torch.arange(N, device=device).unsqueeze(0).unsqueeze(2)  # [1, N, 1]
        return arange.expand(T, N, k).clone()
