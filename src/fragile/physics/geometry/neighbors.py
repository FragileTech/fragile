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

        # Compute within-row positions using the cumsum trick.
        # filtered_row_indices is sorted (CSR preserves source order),
        # so bincount + cumsum gives group starts, and arange - starts
        # gives the 0-indexed position of each filtered edge within its row.
        counts_filtered = torch.bincount(filtered_row_indices, minlength=n_nodes)
        group_starts = torch.zeros(n_nodes, dtype=torch.long, device=device)
        group_starts[1:] = counts_filtered[:-1].cumsum(0)
        filtered_pos = (
            torch.arange(filtered_row_indices.numel(), device=device)
            - group_starts[filtered_row_indices]
        )
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

    Fully vectorized — builds a dense ``[T, N, k]`` neighbor matrix from the
    companion tensors, then delegates to :func:`_select_alive_samples_and_neighbors`
    for alive-walker selection and padding.  No Python loop over timesteps.

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

    # Build dense neighbor matrix [T, N, k] from companion data.
    # Columns 0 and 1 are the distance and clone companions; columns 2+ are 0.
    neighbor_matrix = torch.zeros(T, N, k, device=device, dtype=torch.long)
    neighbor_matrix[:, :, 0] = companions_distance
    neighbor_matrix[:, :, 1] = companions_clone

    sample_indices, neighbor_indices = _select_alive_samples_and_neighbors(
        alive, neighbor_matrix, sample_size, device,
    )

    return sample_indices, neighbor_indices, alive


def _edges_to_neighbor_matrix(
    edges: Tensor,
    N: int,
    k: int,
    device: torch.device,
    skip_self_loops: bool = True,
) -> Tensor:
    """Convert edge list to dense neighbor matrix using vectorized scatter.

    Sorts edges by source, computes within-group column positions via the
    cumsum trick, then scatters into a dense ``[N, k]`` matrix.  Only the
    first *k* neighbors per node are kept; the rest are discarded so the
    output is allocated at exactly ``[N, k]`` rather than ``[N, max_deg]``.

    Args:
        edges: Edge tensor ``[E, 2]`` (src, dst pairs).
        N: Number of nodes.
        k: Number of neighbors to keep per node.
        device: Target device.
        skip_self_loops: If ``True`` (default), remove ``(i, i)`` edges
            before processing.  Set to ``False`` when edges are guaranteed
            self-loop-free to save one comparison + filter pass.

    Returns:
        Neighbor matrix ``[N, k]`` with ``-1`` for missing neighbors.
    """
    edges_d = edges.to(device)
    src, dst = edges_d[:, 0], edges_d[:, 1]

    if skip_self_loops:
        not_self = src != dst
        src = src[not_self]
        dst = dst[not_self]

    if src.numel() == 0:
        return torch.full((N, k), -1, dtype=torch.long, device=device)

    # Sort by src to group neighbors together (stability not needed)
    sort_order = torch.argsort(src)
    src_sorted = src[sort_order]
    dst_sorted = dst[sort_order]

    # Column index = position of each edge within its source's group
    degree = torch.bincount(src_sorted, minlength=N)
    group_starts = torch.zeros(N, dtype=torch.long, device=device)
    group_starts[1:] = degree[:-1].cumsum(0)
    col_idx = torch.arange(src_sorted.numel(), device=device) - group_starts[src_sorted]

    # Only scatter edges that fit in k columns — avoids allocating [N, max_deg]
    fits = col_idx < k
    neighbor_matrix = torch.full((N, k), -1, dtype=torch.long, device=device)
    neighbor_matrix[src_sorted[fits], col_idx[fits]] = dst_sorted[fits]

    return neighbor_matrix


def _select_alive_samples_and_neighbors(
    alive: Tensor,
    neighbor_matrix: Tensor,
    sample_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Vectorized alive-walker selection and neighbor gathering.

    Replaces the per-timestep Python loop that was previously used in both
    ``compute_companion_batch`` and ``compute_recorded_neighbors_batch``.

    Algorithm:
        1. **argsort** the alive mask (descending, stable) so alive walkers
           appear first within each timestep while preserving their original
           index order — O(T N log N) but a single GPU kernel.
        2. Truncate to ``sample_size`` columns.
        3. Build a validity mask (positions < alive count per timestep).
        4. **gather** neighbor data for the selected walkers.
        5. Replace ``-1`` sentinels (missing neighbors) with self-index.
        6. Zero-pad positions beyond the alive count.

    Args:
        alive: Boolean alive mask ``[T, N]``.
        neighbor_matrix: Neighbor indices ``[T, N, k]``.  Use ``-1`` for
            missing neighbors (they will be replaced with the walker's own
            index).
        sample_size: Output width — number of samples per timestep.
        device: Target device.

    Returns:
        sample_indices: ``[T, sample_size]`` walker indices (alive walkers
            first in ascending order, then zero-padded).
        neighbor_indices: ``[T, sample_size, k]`` neighbor data (zero-padded
            at padding positions).
    """
    T, N = alive.shape
    k = neighbor_matrix.shape[2]

    # Sort so alive walkers come first; stable keeps original index order
    sort_order = torch.argsort(alive.long(), dim=1, descending=True, stable=True)
    sort_order = sort_order[:, :sample_size]  # [T, sample_size]

    # Validity mask — positions beyond alive count are padding
    alive_counts = alive.sum(dim=1, keepdim=True)  # [T, 1]
    positions = torch.arange(sample_size, device=device).unsqueeze(0)  # [1, S]
    is_valid = positions < alive_counts  # [T, S]

    # Sample indices: alive walker ids at valid positions, 0 at padding
    sample_indices = torch.where(is_valid, sort_order, torch.zeros_like(sort_order))

    # Gather neighbor data for selected walkers
    gather_idx = sort_order.unsqueeze(2).expand(-1, -1, k)  # [T, S, k]
    neighbor_indices = torch.gather(neighbor_matrix, 1, gather_idx)  # [T, S, k]

    # Replace -1 (missing neighbors) with self-index
    self_idx = sample_indices.unsqueeze(2).expand_as(neighbor_indices)
    neighbor_indices = torch.where(neighbor_indices < 0, self_idx, neighbor_indices)

    # Zero out padding positions
    neighbor_indices = neighbor_indices * is_valid.unsqueeze(2).long()

    return sample_indices, neighbor_indices


def compute_recorded_neighbors_batch(
    history: RunHistory,
    start_idx: int,
    neighbor_k: int,
    sample_size: int | None = None,
    end_idx: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Use recorded neighbor edges from RunHistory.

    Batched implementation: concatenates all per-timestep edge lists with
    per-timestep node offsets (node *i* in timestep *t* becomes global node
    ``t * N + i``), builds one combined neighbor matrix for all ``T * N``
    nodes in a single :func:`_edges_to_neighbor_matrix` call, then converts
    global indices back to per-timestep local indices and delegates
    alive-sample selection to :func:`_select_alive_samples_and_neighbors`.

    The lightweight Python loop that collects edge tensors touches only
    list metadata; all heavy computation (argsort, bincount, scatter,
    gather) runs as batched GPU kernels.

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

    # --- Collect per-timestep edges (lightweight Python loop) ---
    edge_chunks: list[Tensor] = []
    chunk_offsets: list[int] = []
    for t in range(T):
        record_idx = start_idx + t
        edges = history.neighbor_edges[record_idx]
        if torch.is_tensor(edges) and edges.numel() > 0:
            edge_chunks.append(edges)
            chunk_offsets.append(t * N)

    if edge_chunks:
        # Concatenate all edges and add per-timestep node offsets so that
        # node i at timestep t becomes global node (t * N + i).
        chunk_sizes = torch.tensor(
            [e.shape[0] for e in edge_chunks], device=device, dtype=torch.long,
        )
        offsets_tensor = torch.tensor(
            chunk_offsets, device=device, dtype=torch.long,
        )
        all_edges = torch.cat(
            [e.to(device) for e in edge_chunks], dim=0,
        )  # [total_E, 2]
        per_edge_offset = torch.repeat_interleave(offsets_tensor, chunk_sizes)
        all_edges = all_edges + per_edge_offset.unsqueeze(1)

        # One scatter pass for the combined T*N-node graph
        combined = _edges_to_neighbor_matrix(all_edges, T * N, k, device)
        neighbor_matrix = combined.reshape(T, N, k)

        # Convert global indices back to per-timestep local indices.
        # Valid entries are in [t*N, (t+1)*N); missing entries are -1.
        t_offsets = torch.arange(T, device=device).view(T, 1, 1) * N
        neighbor_matrix = torch.where(
            neighbor_matrix >= 0,
            neighbor_matrix - t_offsets,
            torch.full_like(neighbor_matrix, -1),
        )
    else:
        # No edges at any timestep — all -1 (will become self-loops below)
        neighbor_matrix = torch.full(
            (T, N, k), -1, dtype=torch.long, device=device,
        )

    # Vectorized alive-sample selection and neighbor gathering
    sample_indices, neighbor_indices = _select_alive_samples_and_neighbors(
        alive, neighbor_matrix, sample_size, device,
    )

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
