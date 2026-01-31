"""
Voronoi-weighted particle operators for QFT mass computation.

This module implements geometric neighbor selection and weighting using Voronoi
tessellation, addressing noise sensitivity issues with k-NN uniform sampling on
non-uniform, dynamical grids.

Key Functions:
    - compute_voronoi_tessellation(): Build Voronoi diagram from walker positions
    - compute_geometric_weights(): Extract facet areas, cell volumes
    - compute_meson_operator_voronoi(): Meson operator with geometric weighting
    - compute_baryon_operator_voronoi(): Baryon operator with geometric weighting

Weighting Modes:
    - "facet_area": w_ij = Area(facet_ij) / Σ_k Area(facet_ik)
    - "volume": w_i = V_i / mean(V)
    - "combined": w_ij = Area(facet_ij) / sqrt(V_i * V_j)

Mathematical Justification:
    In continuum QFT, spatial averages are volume integrals:
        ⟨O⟩ = ∫ d³x O(x) / ∫ d³x

    For non-uniform discrete grids, the correct discretization is:
        ⟨O⟩ = Σ_i V_i O_i / Σ_i V_i

    k-NN with uniform weights assumes equal volumes → incorrect for non-uniform grids.
    Voronoi + volume weighting is the correct finite volume discretization.

See Also:
    - QFT_MASS_COMPUTATION_GUIDE.md: "Advanced: Voronoi Tessellation"
    - particle_observables.py: k-NN baseline implementation
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import torch
from scipy.spatial import Voronoi, ConvexHull

from fragile.fractalai.core.history import RunHistory


def classify_boundary_cells(
    voronoi_data: dict[str, Any],
    positions: torch.Tensor,
    bounds: Any | None,
    pbc: bool,
    boundary_tolerance: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """
    Classify cells into boundary, boundary-adjacent, and interior tiers.

    This implements a two-tier boundary exclusion strategy to eliminate
    artifacts from cells touching the simulation box boundary.

    Args:
        voronoi_data: Output from compute_voronoi_tessellation (partial, needs neighbor_lists)
        positions: Original walker positions [N, d]
        bounds: Simulation box bounds
        pbc: Whether periodic boundaries are used
        boundary_tolerance: Distance threshold to classify as boundary

    Returns:
        Dictionary containing:
            - "tier": torch.Tensor [N_alive] with values:
                0 = Tier 1 (boundary, fully excluded)
                1 = Tier 2 (boundary-adjacent, no observables but valid neighbor)
                2 = Tier 3+ (interior, compute observables)
            - "is_boundary": torch.Tensor [N_alive] boolean (Tier 1)
            - "is_boundary_adjacent": torch.Tensor [N_alive] boolean (Tier 2)
            - "is_interior": torch.Tensor [N_alive] boolean (Tier 3+)
    """
    vor = voronoi_data.get("voronoi")
    neighbor_lists = voronoi_data.get("neighbor_lists", {})
    index_map = voronoi_data.get("index_map", {})
    n_alive = len(index_map)

    if n_alive == 0:
        return {
            "tier": torch.tensor([], dtype=torch.long),
            "is_boundary": torch.tensor([], dtype=torch.bool),
            "is_boundary_adjacent": torch.tensor([], dtype=torch.bool),
            "is_interior": torch.tensor([], dtype=torch.bool),
        }

    # For PBC, there are no real boundaries - all cells are interior
    if pbc:
        tier = torch.ones(n_alive, dtype=torch.long) * 2
        is_boundary = torch.zeros(n_alive, dtype=torch.bool)
        is_boundary_adjacent = torch.zeros(n_alive, dtype=torch.bool)
        is_interior = torch.ones(n_alive, dtype=torch.bool)
        return {
            "tier": tier,
            "is_boundary": is_boundary,
            "is_boundary_adjacent": is_boundary_adjacent,
            "is_interior": is_interior,
        }

    # Step 1: Detect Tier 1 (boundary cells)
    is_boundary = torch.zeros(n_alive, dtype=torch.bool)

    if vor is None:
        # Voronoi failed, mark all as interior to avoid errors
        tier = torch.ones(n_alive, dtype=torch.long) * 2
        is_boundary_adjacent = torch.zeros(n_alive, dtype=torch.bool)
        is_interior = torch.ones(n_alive, dtype=torch.bool)
        return {
            "tier": tier,
            "is_boundary": is_boundary,
            "is_boundary_adjacent": is_boundary_adjacent,
            "is_interior": is_interior,
        }

    for i in range(n_alive):
        # Method A: Check if Voronoi region has -1 vertices (infinite region)
        region_idx = vor.point_region[i]
        vertices_idx = vor.regions[region_idx]
        if -1 in vertices_idx:
            is_boundary[i] = True
            continue

        # Method B: Check if position is near box boundary
        orig_idx = index_map[i]
        pos = positions[orig_idx].cpu().numpy()

        if bounds is not None:
            near_boundary = False
            for dim in range(positions.shape[1]):
                low = float(bounds.low[dim].cpu().numpy() if torch.is_tensor(bounds.low) else bounds.low[dim])
                high = float(bounds.high[dim].cpu().numpy() if torch.is_tensor(bounds.high) else bounds.high[dim])

                if abs(pos[dim] - low) < boundary_tolerance:
                    near_boundary = True
                    break
                if abs(pos[dim] - high) < boundary_tolerance:
                    near_boundary = True
                    break

            if near_boundary:
                is_boundary[i] = True

    # Step 2: Detect Tier 2 (boundary-adjacent)
    is_boundary_adjacent = torch.zeros(n_alive, dtype=torch.bool)

    for i in range(n_alive):
        if is_boundary[i]:
            continue  # Already Tier 1

        neighbors = neighbor_lists.get(i, [])
        for j in neighbors:
            if j < n_alive and is_boundary[j]:
                is_boundary_adjacent[i] = True
                break

    # Step 3: Mark Tier 3+ (interior)
    is_interior = ~is_boundary & ~is_boundary_adjacent

    # Step 4: Create tier array
    tier = torch.zeros(n_alive, dtype=torch.long)
    tier[is_boundary] = 0
    tier[is_boundary_adjacent] = 1
    tier[is_interior] = 2

    return {
        "tier": tier,
        "is_boundary": is_boundary,
        "is_boundary_adjacent": is_boundary_adjacent,
        "is_interior": is_interior,
    }


def compute_voronoi_tessellation(
    positions: torch.Tensor,
    alive: torch.Tensor,
    bounds: Any | None,
    pbc: bool,
    pbc_mode: str = "mirror",
    exclude_boundary: bool = True,
    boundary_tolerance: float = 1e-6,
    compute_curvature: bool = True,
    prev_volumes: np.ndarray | None = None,
    dt: float = 1.0,
    spatial_dims: int | None = None,
) -> dict[str, Any]:
    """
    Compute Voronoi tessellation from walker positions.

    Args:
        positions: Walker positions [N, d]
        alive: Alive mask [N]
        bounds: Optional TorchBounds object for PBC
        pbc: Whether to use periodic boundary conditions
        pbc_mode: How to handle PBC - "mirror", "replicate", or "ignore"
        exclude_boundary: Whether to exclude boundary cells (Tier 1) from neighbor lists
        boundary_tolerance: Distance threshold for boundary cell detection
        compute_curvature: Whether to compute curvature proxies (default True)
        prev_volumes: Previous timestep volumes for Raychaudhuri expansion (optional)
        dt: Timestep for volume evolution rate (default 1.0)
        spatial_dims: If provided, only use first N dimensions for Voronoi (default: all dimensions)
                     Useful for Euclidean time analysis where 4th dimension is time, not space

    Returns:
        Dictionary containing:
            - "voronoi": scipy.spatial.Voronoi object
            - "neighbor_lists": dict[int, list[int]] mapping i → [j1, j2, ...]
            - "volumes": np.ndarray [N_alive] cell volumes
            - "facet_areas": dict[(int,int), float] mapping (i,j) → area
            - "alive_indices": np.ndarray original alive indices
            - "index_map": dict[int, int] mapping voronoi idx → original idx
            - "classification": dict with boundary tier classification (if exclude_boundary=True)
            - "curvature_proxies": dict with curvature proxy estimates (if compute_curvature=True)
    """
    # Filter to alive walkers
    alive_indices = torch.where(alive)[0].cpu().numpy()
    if len(alive_indices) == 0:
        return {
            "voronoi": None,
            "neighbor_lists": {},
            "volumes": np.array([]),
            "facet_areas": {},
            "alive_indices": alive_indices,
            "index_map": {},
        }

    # Filter positions to spatial dimensions only if requested
    if spatial_dims is not None:
        positions_alive = positions[alive, :spatial_dims].cpu().numpy()
        d = spatial_dims
    else:
        positions_alive = positions[alive].cpu().numpy()
        d = positions.shape[1]

    n_alive = len(alive_indices)

    # Handle periodic boundary conditions
    if pbc and bounds is not None and pbc_mode != "ignore":
        high = bounds.high.cpu().numpy()
        low = bounds.low.cpu().numpy()
        span = high - low

        if pbc_mode == "mirror":
            # Mirror positions across boundaries for PBC
            # In 2D: 9 copies (original + 8 mirrors)
            # In 3D: 27 copies (original + 26 mirrors)
            offsets = []
            for offset in itertools.product([-1, 0, 1], repeat=d):
                offsets.append(offset)

            positions_extended = []
            for offset in offsets:
                offset_arr = np.array(offset) * span
                positions_extended.append(positions_alive + offset_arr)

            positions_vor = np.vstack(positions_extended)

        elif pbc_mode == "replicate":
            # Similar to mirror but may use different tiling strategy
            # For simplicity, use same as mirror
            offsets = []
            for offset in itertools.product([-1, 0, 1], repeat=d):
                offsets.append(offset)

            positions_extended = []
            for offset in offsets:
                offset_arr = np.array(offset) * span
                positions_extended.append(positions_alive + offset_arr)

            positions_vor = np.vstack(positions_extended)
        else:
            positions_vor = positions_alive
    else:
        positions_vor = positions_alive

    # Compute Voronoi tessellation
    try:
        vor = Voronoi(positions_vor)
    except Exception as e:
        # Voronoi can fail for degenerate configurations (too few points, collinear, etc.)
        # Return empty structure with initialized neighbor lists
        return {
            "voronoi": None,
            "neighbor_lists": {i: [] for i in range(n_alive)},
            "volumes": np.ones(n_alive),  # Use unit volumes as fallback
            "facet_areas": {},
            "alive_indices": alive_indices,
            "index_map": {i: int(alive_indices[i]) for i in range(n_alive)},
            "error": str(e),
        }

    # Build neighbor lists and compute facet areas
    neighbor_lists: dict[int, list[int]] = {i: [] for i in range(n_alive)}
    facet_areas: dict[tuple[int, int], float] = {}

    # Extract ridges (shared facets between cells)
    for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
        # Map back to original indices if using PBC
        if pbc and bounds is not None and pbc_mode != "ignore":
            p1_orig = p1 % n_alive
            p2_orig = p2 % n_alive

            # Only keep connections within original domain or to immediate neighbors
            if p1_orig == p2_orig:
                continue  # Skip self-connections

            # Check if both are in original domain
            if p1 < n_alive and p2 < n_alive:
                i, j = p1, p2
            else:
                # Skip ghost-to-ghost connections
                continue
        else:
            i, j = p1, p2

        if i >= n_alive or j >= n_alive:
            continue

        # Add to neighbor lists
        if j not in neighbor_lists[i]:
            neighbor_lists[i].append(j)
        if i not in neighbor_lists[j]:
            neighbor_lists[j].append(i)

        # Compute facet area from ridge vertices
        ridge_vertices = vor.ridge_vertices[ridge_idx]
        if -1 not in ridge_vertices and len(ridge_vertices) >= d:
            try:
                vertices = vor.vertices[ridge_vertices]
                area = _compute_facet_area(vertices, d)
                facet_areas[(i, j)] = area
                facet_areas[(j, i)] = area
            except Exception:
                # Use a default small area if computation fails
                facet_areas[(i, j)] = 1.0
                facet_areas[(j, i)] = 1.0

    # Compute cell volumes
    volumes = np.zeros(n_alive)
    for i in range(n_alive):
        region_idx = vor.point_region[i]
        vertices_idx = vor.regions[region_idx]

        if -1 not in vertices_idx and len(vertices_idx) >= d + 1:
            try:
                vertices = vor.vertices[vertices_idx]
                volumes[i] = _compute_cell_volume(vertices, d)
            except Exception:
                # Use mean volume if computation fails
                volumes[i] = 1.0

    # Replace zero volumes with mean
    mean_vol = volumes[volumes > 0].mean() if (volumes > 0).any() else 1.0
    volumes[volumes == 0] = mean_vol

    # Create index map
    index_map = {i: int(alive_indices[i]) for i in range(n_alive)}

    # Apply boundary filtering if requested
    classification = None
    if exclude_boundary and not pbc:
        # Classify cells into tiers
        classification = classify_boundary_cells(
            voronoi_data={
                "voronoi": vor,
                "neighbor_lists": neighbor_lists,
                "index_map": index_map,
            },
            positions=positions,
            bounds=bounds,
            pbc=pbc,
            boundary_tolerance=boundary_tolerance,
        )

        # Filter Tier 1 cells from ALL neighbor lists
        neighbor_lists_filtered = {}
        for i in range(n_alive):
            if classification["tier"][i] == 0:
                # Tier 1 (boundary): completely excluded, empty neighbor list
                neighbor_lists_filtered[i] = []
            else:
                # Tier 2 or 3: keep only Tier 2+ neighbors (exclude Tier 1)
                neighbors = neighbor_lists.get(i, [])
                filtered = [
                    j for j in neighbors
                    if j < n_alive and classification["tier"][j] > 0
                ]
                neighbor_lists_filtered[i] = filtered

        neighbor_lists = neighbor_lists_filtered

    # Compute curvature proxies if requested
    curvature_proxies = None
    if compute_curvature and n_alive > 0:
        # Build the voronoi_data dict needed for curvature computation
        voronoi_data_for_curvature = {
            "voronoi": vor,
            "neighbor_lists": neighbor_lists,
            "volumes": volumes,
            "facet_areas": facet_areas,
            "alive_indices": alive_indices,
            "index_map": index_map,
            "classification": classification,
        }
        
        try:
            curvature_proxies = compute_curvature_proxies(
                voronoi_data=voronoi_data_for_curvature,
                positions=positions_alive,
                prev_volumes=prev_volumes,
                dt=dt,
            )
        except Exception as e:
            # If curvature computation fails, log warning but don't break tessellation
            import warnings
            warnings.warn(f"Curvature proxy computation failed: {e}", RuntimeWarning, stacklevel=2)
            curvature_proxies = None

    return {
        "voronoi": vor,
        "neighbor_lists": neighbor_lists,
        "volumes": volumes,
        "facet_areas": facet_areas,
        "alive_indices": alive_indices,
        "index_map": index_map,
        "classification": classification,
        "curvature_proxies": curvature_proxies,
    }


def _compute_facet_area(vertices: np.ndarray, d: int) -> float:
    """
    Compute area of a facet from its vertices.

    Args:
        vertices: Vertex coordinates [n_vertices, d]
        d: Dimension

    Returns:
        Facet area (length in 2D, area in 3D)
    """
    if d == 2:
        # In 2D, facet is a line segment
        if len(vertices) < 2:
            return 0.0
        # Distance between first two vertices
        return float(np.linalg.norm(vertices[1] - vertices[0]))

    elif d == 3:
        # In 3D, facet is a polygon
        if len(vertices) < 3:
            return 0.0

        # Compute area using cross products
        # Triangulate from first vertex
        v0 = vertices[0]
        total_area = 0.0
        for i in range(1, len(vertices) - 1):
            v1 = vertices[i]
            v2 = vertices[i + 1]
            # Area of triangle = 0.5 * ||cross(v1-v0, v2-v0)||
            cross = np.cross(v1 - v0, v2 - v0)
            total_area += 0.5 * np.linalg.norm(cross)

        return float(total_area)

    else:
        # For higher dimensions, use convex hull
        try:
            hull = ConvexHull(vertices)
            return float(hull.volume)
        except Exception:
            return 1.0


def _compute_cell_volume(vertices: np.ndarray, d: int) -> float:
    """
    Compute volume of a Voronoi cell from its vertices.

    Args:
        vertices: Vertex coordinates [n_vertices, d]
        d: Dimension

    Returns:
        Cell volume (area in 2D, volume in 3D)
    """
    if len(vertices) < d + 1:
        return 0.0

    try:
        hull = ConvexHull(vertices)
        return float(hull.volume)
    except Exception:
        return 1.0


def compute_dual_volumes_from_history(
    history: RunHistory,
    step: int | None = None,
    record_index: int | None = None,
) -> torch.Tensor:
    """Compute Voronoi dual volumes from stored history.voronoi_regions.

    Args:
        history: RunHistory with voronoi_regions recorded.
        step: Absolute step number to extract (mutually exclusive with record_index).
        record_index: Recorded index to extract (0-based).

    Returns:
        Tensor of dual volumes [N] (NaN for missing/unbounded cells).
    """
    if history.voronoi_regions is None:
        msg = "RunHistory.voronoi_regions is empty; run with neighbor_graph_record=True."
        raise ValueError(msg)

    if step is not None and record_index is not None:
        msg = "Provide either step or record_index, not both."
        raise ValueError(msg)

    if record_index is None:
        if step is None:
            record_index = history.n_recorded - 1
        else:
            record_index = history.get_step_index(step)

    if record_index < 0 or record_index >= len(history.voronoi_regions):
        msg = f"record_index {record_index} out of range for voronoi_regions"
        raise IndexError(msg)

    entry = history.voronoi_regions[record_index]
    device = history.x_final.device
    volumes = torch.full((history.N,), float("nan"), device=device, dtype=history.x_final.dtype)

    if entry is None:
        return volumes

    vertices = entry.get("vertices")
    regions = entry.get("regions")
    if vertices is None or regions is None:
        return volumes

    if torch.is_tensor(vertices):
        vertices_np = vertices.detach().cpu().numpy()
    else:
        vertices_np = np.asarray(vertices)

    d = history.d
    for idx, region in enumerate(regions):
        if region is None:
            continue
        if len(region) < d + 1 or -1 in region:
            continue
        cell_vertices = vertices_np[region]
        vol = _compute_cell_volume(cell_vertices, d)
        volumes[idx] = float(vol)

    return volumes


def compute_geometric_weights(
    voronoi_data: dict[str, Any],
    weight_mode: str = "facet_area",
    normalize: bool = True,
) -> dict[str, Any]:
    """
    Compute geometric weights for Voronoi neighbors.

    Args:
        voronoi_data: Output from compute_voronoi_tessellation
        weight_mode: Weighting scheme - "facet_area", "volume", or "combined"
        normalize: Whether to normalize weights

    Returns:
        Dictionary containing:
            - "node_weights": torch.Tensor [N] (for volume weighting)
            - "edge_weights": dict[(i,j), float] (for facet/combined weighting)
    """
    volumes = voronoi_data["volumes"]
    facet_areas = voronoi_data["facet_areas"]
    neighbor_lists = voronoi_data["neighbor_lists"]
    n = len(volumes)

    if n == 0:
        return {
            "node_weights": torch.tensor([]),
            "edge_weights": {},
        }

    node_weights = torch.from_numpy(volumes).float()
    edge_weights: dict[tuple[int, int], float] = {}

    if weight_mode == "facet_area":
        # Weight by facet area: w_ij = Area(facet_ij) / Σ_k Area(facet_ik)
        for i in range(n):
            neighbors = neighbor_lists.get(i, [])
            if not neighbors:
                continue

            # Sum of all facet areas for node i
            total_area = 0.0
            for j in neighbors:
                area = facet_areas.get((i, j), 1.0)
                total_area += area

            # Normalize if requested
            if normalize and total_area > 0:
                for j in neighbors:
                    area = facet_areas.get((i, j), 1.0)
                    edge_weights[(i, j)] = area / total_area
            else:
                for j in neighbors:
                    edge_weights[(i, j)] = facet_areas.get((i, j), 1.0)

    elif weight_mode == "volume":
        # Weight by cell volume: w_i = V_i / mean(V)
        if normalize and volumes.size > 0:
            mean_vol = volumes.mean()
            if mean_vol > 0:
                node_weights = torch.from_numpy(volumes / mean_vol).float()

        # For edge-based operators, use average of node volumes
        for i in range(n):
            neighbors = neighbor_lists.get(i, [])
            for j in neighbors:
                if normalize:
                    edge_weights[(i, j)] = (node_weights[i].item() + node_weights[j].item()) / 2.0
                else:
                    edge_weights[(i, j)] = (volumes[i] + volumes[j]) / 2.0

    elif weight_mode == "combined":
        # Combined: w_ij = Area(facet_ij) / sqrt(V_i * V_j)
        for i in range(n):
            neighbors = neighbor_lists.get(i, [])
            for j in neighbors:
                area = facet_areas.get((i, j), 1.0)
                vol_product = volumes[i] * volumes[j]
                if vol_product > 0:
                    edge_weights[(i, j)] = area / np.sqrt(vol_product)
                else:
                    edge_weights[(i, j)] = area

        # Normalize if requested
        if normalize:
            for i in range(n):
                neighbors = neighbor_lists.get(i, [])
                if not neighbors:
                    continue
                total_weight = sum(edge_weights.get((i, j), 0.0) for j in neighbors)
                if total_weight > 0:
                    for j in neighbors:
                        if (i, j) in edge_weights:
                            edge_weights[(i, j)] /= total_weight

    else:
        msg = f"Unknown weight_mode: {weight_mode}"
        raise ValueError(msg)

    return {
        "node_weights": node_weights,
        "edge_weights": edge_weights,
    }


def compute_meson_operator_voronoi(
    color: torch.Tensor,
    sample_indices: torch.Tensor,
    voronoi_neighbors: dict[int, list[int]],
    geometric_weights: dict[str, Any],
    alive: torch.Tensor,
    color_valid: torch.Tensor,
    index_map: dict[int, int],
    weight_mode: str = "facet_area",
    classification: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute meson operator using Voronoi neighbors with geometric weighting.

    Args:
        color: Complex color vectors [N, d]
        sample_indices: Walker indices to compute for [n_samples]
        voronoi_neighbors: Neighbor lists from Voronoi tessellation
        geometric_weights: Weights from compute_geometric_weights
        alive: Alive mask [N]
        color_valid: Valid color mask [N]
        index_map: Mapping from Voronoi index to original index
        weight_mode: Weighting mode ("facet_area", "volume", "combined")
        classification: Optional boundary tier classification (from compute_voronoi_tessellation)

    Returns:
        (meson_values, valid_mask) where:
            - meson_values: Complex meson operator values [n_samples]
            - valid_mask: Boolean mask of valid computations [n_samples]

    Note:
        If classification is provided, only Tier 3+ (interior) cells will have observables computed.
        Tier 1 (boundary) cells are excluded from neighbor lists.
        Tier 2 (boundary-adjacent) cells contribute as neighbors but have no observable.
    """
    n_samples = sample_indices.shape[0]
    device = color.device
    dtype = color.dtype

    meson = torch.zeros(n_samples, dtype=dtype, device=device)
    valid = torch.zeros(n_samples, dtype=torch.bool, device=device)

    edge_weights = geometric_weights["edge_weights"]
    node_weights = geometric_weights.get("node_weights")

    # Create reverse index map (original idx → voronoi idx)
    reverse_map = {v: k for k, v in index_map.items()}

    for idx, i_orig in enumerate(sample_indices):
        i_orig_int = int(i_orig.item())

        if not (alive[i_orig_int] and color_valid[i_orig_int]):
            continue

        # Map to Voronoi index
        if i_orig_int not in reverse_map:
            continue

        i_vor = reverse_map[i_orig_int]

        # Skip if not interior (Tier 3+)
        if classification is not None:
            if classification["tier"][i_vor] < 2:
                # Tier 0 or 1: skip observable computation
                valid[idx] = False
                continue

        neighbors_vor = voronoi_neighbors.get(i_vor, [])

        if not neighbors_vor:
            continue

        weighted_sum = 0.0 + 0.0j
        total_weight = 0.0

        for j_vor in neighbors_vor:
            j_orig = index_map[j_vor]

            if not (alive[j_orig] and color_valid[j_orig]):
                continue

            # Meson operator: ⟨i|j⟩ = color[i]† · color[j]
            meson_ij = torch.dot(color[i_orig_int].conj(), color[j_orig])

            # Get geometric weight
            if weight_mode == "volume" and node_weights is not None:
                w_ij = float(node_weights[j_vor].item())
            else:
                w_ij = edge_weights.get((i_vor, j_vor), 1.0)

            weighted_sum += w_ij * meson_ij
            total_weight += w_ij

        if total_weight > 0:
            meson[idx] = weighted_sum / total_weight
            valid[idx] = True

    return meson, valid


def compute_baryon_operator_voronoi(
    color: torch.Tensor,
    sample_indices: torch.Tensor,
    voronoi_neighbors: dict[int, list[int]],
    geometric_weights: dict[str, Any],
    alive: torch.Tensor,
    color_valid: torch.Tensor,
    index_map: dict[int, int],
    weight_mode: str = "facet_area",
    max_triplets: int | None = None,
    classification: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute baryon operator using Voronoi neighbor triplets with geometric weighting.

    Args:
        color: Complex color vectors [N, 3] (requires d=3!)
        sample_indices: Walker indices to compute for [n_samples]
        voronoi_neighbors: Neighbor lists from Voronoi tessellation
        geometric_weights: Weights from compute_geometric_weights
        alive: Alive mask [N]
        color_valid: Valid color mask [N]
        index_map: Mapping from Voronoi index to original index
        weight_mode: Weighting mode ("facet_area", "volume", "combined")
        max_triplets: Optional limit on number of triplets per walker
        classification: Optional boundary tier classification (from compute_voronoi_tessellation)

    Returns:
        (baryon_values, valid_mask) where:
            - baryon_values: Complex baryon operator values [n_samples]
            - valid_mask: Boolean mask of valid computations [n_samples]

    Note:
        If classification is provided, only Tier 3+ (interior) cells will have observables computed.
        Tier 1 (boundary) cells are excluded from neighbor lists.
        Tier 2 (boundary-adjacent) cells contribute as neighbors but have no observable.
    """
    if color.shape[1] != 3:
        msg = "Baryon operator requires d=3 (color dimension must be 3)"
        raise ValueError(msg)

    n_samples = sample_indices.shape[0]
    device = color.device
    dtype = color.dtype

    baryon = torch.zeros(n_samples, dtype=dtype, device=device)
    valid = torch.zeros(n_samples, dtype=torch.bool, device=device)

    edge_weights = geometric_weights["edge_weights"]
    node_weights = geometric_weights.get("node_weights")

    # Create reverse index map (original idx → voronoi idx)
    reverse_map = {v: k for k, v in index_map.items()}

    for idx, i_orig in enumerate(sample_indices):
        i_orig_int = int(i_orig.item())

        if not (alive[i_orig_int] and color_valid[i_orig_int]):
            continue

        # Map to Voronoi index
        if i_orig_int not in reverse_map:
            continue

        i_vor = reverse_map[i_orig_int]

        # Skip if not interior (Tier 3+)
        if classification is not None:
            if classification["tier"][i_vor] < 2:
                # Tier 0 or 1: skip observable computation
                valid[idx] = False
                continue

        neighbors_vor = voronoi_neighbors.get(i_vor, [])

        if len(neighbors_vor) < 2:
            continue

        # Generate all triplet pairs C(n_neighbors, 2)
        neighbor_pairs = list(itertools.combinations(neighbors_vor, 2))

        # Optionally limit number of triplets
        if max_triplets is not None and len(neighbor_pairs) > max_triplets:
            # Randomly sample triplets
            import random
            neighbor_pairs = random.sample(neighbor_pairs, max_triplets)

        weighted_sum = 0.0 + 0.0j
        total_weight = 0.0

        for j_vor, k_vor in neighbor_pairs:
            j_orig = index_map[j_vor]
            k_orig = index_map[k_vor]

            if not (alive[j_orig] and color_valid[j_orig]):
                continue
            if not (alive[k_orig] and color_valid[k_orig]):
                continue

            # Baryon operator: det([color[i], color[j], color[k]])
            matrix = torch.stack([color[i_orig_int], color[j_orig], color[k_orig]], dim=-1)
            det = torch.linalg.det(matrix)

            # Get geometric weight (average over edges in triplet)
            if weight_mode == "volume" and node_weights is not None:
                w_j = float(node_weights[j_vor].item())
                w_k = float(node_weights[k_vor].item())
                w_triplet = (w_j + w_k) / 2.0
            else:
                w_ij = edge_weights.get((i_vor, j_vor), 1.0)
                w_ik = edge_weights.get((i_vor, k_vor), 1.0)
                w_triplet = (w_ij + w_ik) / 2.0

            weighted_sum += w_triplet * det
            total_weight += w_triplet

        if total_weight > 0:
            baryon[idx] = weighted_sum / total_weight
            valid[idx] = True

    return baryon, valid


def compute_curvature_proxies(
    voronoi_data: dict[str, Any],
    positions: np.ndarray,
    prev_volumes: np.ndarray | None = None,
    dt: float = 1.0,
) -> dict[str, Any]:
    """
    Compute fast O(N) geometric curvature proxies from Voronoi tessellation.

    This function implements three fast curvature estimation methods that avoid
    expensive Hessian computation:
    
    1. **Volume Distortion**: Variance of normalized cell volumes measures curvature
       - Flat space → uniform volumes → low variance
       - Curved space → non-uniform volumes → high variance
    
    2. **Shape Distortion**: Inradius/circumradius ratio per cell
       - Regular cells (flat) → ratio ≈ 1
       - Distorted cells (curved) → ratio << 1
    
    3. **Raychaudhuri Expansion**: Volume evolution dV/dt ≈ -R
       - From discrete Raychaudhuri equation
       - Positive curvature → volumes shrink (θ < 0)
       - Negative curvature → volumes grow (θ > 0)

    Args:
        voronoi_data: Output from compute_voronoi_tessellation()
        positions: Walker positions [N, d]
        prev_volumes: Previous timestep volumes for Raychaudhuri (optional)
        dt: Timestep for volume evolution

    Returns:
        Dictionary with keys:
            - 'volume_variance': Variance of normalized cell volumes σ²(V_i/<V>)
            - 'volume_distortion': Per-cell normalized volumes V_i/<V> [N]
            - 'shape_distortion': Per-cell inradius/circumradius ratios [N]
            - 'raychaudhuri_expansion': Volume rate dV/dt if prev_volumes provided [N]
            - 'mean_curvature_estimate': Heuristic R ≈ -<dV/dt>/V from Raychaudhuri
            - 'cell_centroids': Centroid positions [N, d] for each cell
            - 'centroid_distances': Distance variance within each cell [N]

    Reference:
        Plan: /home/guillem/.claude/plans/cryptic-percolating-creek.md
        Theory: docs/source/3_fractal_gas/3_fitness_manifold/03_curvature_gravity.md
    """
    volumes = voronoi_data["volumes"]
    neighbor_lists = voronoi_data["neighbor_lists"]
    vor = voronoi_data.get("voronoi")
    n = len(volumes)
    d = positions.shape[1]

    if n == 0:
        return {
            "volume_variance": 0.0,
            "volume_distortion": np.array([]),
            "shape_distortion": np.array([]),
            "raychaudhuri_expansion": np.array([]),
            "mean_curvature_estimate": 0.0,
            "cell_centroids": np.array([]).reshape(0, d),
            "centroid_distances": np.array([]),
        }

    # 1. Volume Distortion
    mean_vol = volumes.mean() if len(volumes) > 0 else 1.0
    normalized_volumes = volumes / mean_vol if mean_vol > 0 else np.ones_like(volumes)
    volume_variance = float(np.var(normalized_volumes))

    # 2. Shape Distortion (inradius/circumradius ratio)
    shape_distortion = np.zeros(n)
    cell_centroids = np.zeros((n, d))
    centroid_distances = np.zeros(n)

    for i in range(n):
        if vor is not None:
            region_idx = vor.point_region[i]
            vertices_idx = vor.regions[region_idx]

            if -1 not in vertices_idx and len(vertices_idx) >= d + 1:
                try:
                    vertices = vor.vertices[vertices_idx]
                    centroid = vertices.mean(axis=0)
                    cell_centroids[i] = centroid

                    # Compute inradius (min distance to vertices) and circumradius (max distance)
                    distances = np.linalg.norm(vertices - centroid, axis=1)
                    inradius = distances.min() if len(distances) > 0 else 0.0
                    circumradius = distances.max() if len(distances) > 0 else 1.0

                    if circumradius > 1e-10:
                        shape_distortion[i] = inradius / circumradius
                    else:
                        shape_distortion[i] = 1.0

                    # Distance variance within cell (user's insight!)
                    centroid_distances[i] = float(np.var(distances)) if len(distances) > 1 else 0.0

                except Exception:
                    shape_distortion[i] = 1.0
                    cell_centroids[i] = positions[i] if i < len(positions) else 0.0
                    centroid_distances[i] = 0.0
            else:
                # Boundary cell or degenerate - use walker position as centroid
                shape_distortion[i] = 1.0
                cell_centroids[i] = positions[i] if i < len(positions) else 0.0
                centroid_distances[i] = 0.0
        else:
            # No Voronoi available
            shape_distortion[i] = 1.0
            cell_centroids[i] = positions[i] if i < len(positions) else 0.0
            centroid_distances[i] = 0.0

    # 3. Raychaudhuri Expansion: θ = (1/V) dV/dt ≈ -R
    result = {
        "volume_variance": volume_variance,
        "volume_distortion": normalized_volumes,
        "shape_distortion": shape_distortion,
        "cell_centroids": cell_centroids,
        "centroid_distances": centroid_distances,
    }

    if prev_volumes is not None and dt > 0:
        if len(prev_volumes) == n:
            # Compute volume rate: dV/dt ≈ (V_t - V_{t-dt}) / dt
            dV_dt = (volumes - prev_volumes) / dt

            # Raychaudhuri expansion: θ = (1/V) dV/dt
            # Avoid division by zero
            safe_volumes = np.where(volumes > 1e-10, volumes, 1.0)
            raychaudhuri_expansion = dV_dt / safe_volumes

            # Mean curvature estimate: R ≈ -<θ>
            # Filter out extreme values for robustness
            valid_mask = np.isfinite(raychaudhuri_expansion) & (np.abs(raychaudhuri_expansion) < 1e6)
            mean_curvature_estimate = 0.0
            if valid_mask.sum() > 0:
                mean_curvature_estimate = float(-raychaudhuri_expansion[valid_mask].mean())

            result["raychaudhuri_expansion"] = raychaudhuri_expansion
            result["mean_curvature_estimate"] = mean_curvature_estimate

    return result


def compute_voronoi_diffusion_tensor(
    voronoi_data: dict[str, Any],
    positions: np.ndarray,
    epsilon_sigma: float = 0.1,
    c2: float = 1.0,
    diagonal_only: bool = True,
) -> np.ndarray:
    """
    Approximate diffusion tensor from Voronoi cell geometry (O(N), no derivatives!).

    This function computes an anisotropic diffusion tensor by analyzing cell
    elongation in each coordinate direction. Cell shape encodes the metric
    anisotropy:
    - Volume distortion V_i/<V> → det(g) (scalar part)
    - Cell elongation → principal directions of metric
    - Aspect ratio → anisotropy strength

    Method (diagonal_only=True):
    1. For each cell, compute volume V_i and characteristic lengths in each direction
    2. Elongation in direction j: e_j = L_j / (V_i)^(1/d)
    3. Diffusion: σ_j ≈ c₂ / √(e_j + ε_Σ)
    4. Compressed direction → large e_j → small σ_j (less diffusion)
       Expanded direction → small e_j → large σ_j (more diffusion)

    Method (diagonal_only=False - full anisotropic):
    1. Compute inertia tensor (covariance) of cell vertices
    2. Eigendecompose: I = R @ diag(λ) @ R^T to find principal axes
    3. Compute diffusion along principal axes: σ_k = c₂ / √(λ_k + ε_Σ)
    4. Rotate back to coordinate frame: Σ = R @ diag(σ) @ R^T
    5. Result captures true cell geometry including rotation/tilt

    Theoretical Justification:
    From scutoid geometry theory, Voronoi cells adapt to fitness landscape curvature.
    Cell stretching in direction v indicates low fitness Hessian eigenvalue in that
    direction. This naturally gives the inverse metric g⁻¹ for diffusion without
    computing any derivatives!

    Args:
        voronoi_data: Output from compute_voronoi_tessellation()
        positions: Walker positions [N, d]
        epsilon_sigma: Regularization parameter (ε_Σ)
        c2: Diffusion scale factor
        diagonal_only: If True, return [N, d] diagonal elements only.
                      If False, return [N, d, d] full symmetric tensors.

    Returns:
        If diagonal_only=True: [N, d] diagonal diffusion tensor components
        If diagonal_only=False: [N, d, d] full symmetric diffusion tensors

    Reference:
        Plan: /home/guillem/.claude/plans/cryptic-percolating-creek.md § Bonus
        Theory: docs/source/3_fractal_gas/3_fitness_manifold/02_scutoid_spacetime.md
    """
    volumes = voronoi_data["volumes"]
    neighbor_lists = voronoi_data["neighbor_lists"]
    vor = voronoi_data.get("voronoi")
    n = len(volumes)
    d = positions.shape[1]

    if n == 0:
        if diagonal_only:
            return np.array([]).reshape(0, d)
        else:
            return np.array([]).reshape(0, d, d)

    # Branch based on mode
    if diagonal_only:
        # ===== DIAGONAL MODE (Original Implementation) =====
        # Initialize diffusion tensor (diagonal approximation)
        sigma = np.ones((n, d))

        for i in range(n):
            if vor is not None:
                region_idx = vor.point_region[i]
                vertices_idx = vor.regions[region_idx]

                if -1 not in vertices_idx and len(vertices_idx) >= d + 1 and volumes[i] > 1e-10:
                    try:
                        vertices = vor.vertices[vertices_idx]

                        # Compute characteristic length in each coordinate direction
                        # Use range (max - min) along each axis
                        for j in range(d):
                            coord_range = vertices[:, j].max() - vertices[:, j].min()

                            # Normalize by volume^(1/d) to get elongation
                            vol_scale = volumes[i] ** (1.0 / d)
                            if vol_scale > 1e-10:
                                elongation = coord_range / vol_scale
                            else:
                                elongation = 1.0

                            # Diffusion coefficient: σ_j = c₂ / √(e_j + ε_Σ)
                            sigma[i, j] = c2 / np.sqrt(elongation + epsilon_sigma)

                    except Exception:
                        # Fallback to isotropic
                        sigma[i, :] = c2 / np.sqrt(1.0 + epsilon_sigma)
                else:
                    # Boundary or degenerate cell - use isotropic diffusion
                    sigma[i, :] = c2 / np.sqrt(1.0 + epsilon_sigma)
            else:
                # No Voronoi - isotropic diffusion
                sigma[i, :] = c2 / np.sqrt(1.0 + epsilon_sigma)

        return sigma

    else:
        # ===== FULL ANISOTROPIC MODE (New Implementation) =====
        # Initialize full diffusion tensors
        sigma_full = np.zeros((n, d, d))
        isotropic_value = c2 / np.sqrt(1.0 + epsilon_sigma)

        for i in range(n):
            if vor is not None:
                region_idx = vor.point_region[i]
                vertices_idx = vor.regions[region_idx]

                # Need at least d+1 vertices for valid d-dimensional inertia tensor
                if (-1 not in vertices_idx and
                    len(vertices_idx) >= d + 1 and
                    volumes[i] > 1e-10):
                    try:
                        vertices = vor.vertices[vertices_idx]

                        # Compute centroid of cell vertices
                        centroid = vertices.mean(axis=0)  # [d]

                        # Center vertices at origin
                        v_centered = vertices - centroid  # [n_vertices, d]

                        # Compute inertia tensor (covariance matrix)
                        # I = (1/n_vertices) * ∑_k v_k v_k^T
                        n_verts = len(vertices)
                        inertia = (v_centered.T @ v_centered) / n_verts  # [d, d]

                        # Ensure symmetry (numerical stability)
                        inertia = 0.5 * (inertia + inertia.T)

                        # Eigendecompose: I = R @ diag(λ) @ R^T
                        eigenvalues, eigenvectors = np.linalg.eigh(inertia)  # [d], [d, d]

                        # Clamp eigenvalues to avoid division by zero
                        # Larger eigenvalue = more elongation in that direction
                        eigenvalues = np.maximum(eigenvalues, epsilon_sigma * 1e-2)

                        # Compute elongation along each principal axis
                        # elongation_k = sqrt(λ_k)
                        elongation_principal = np.sqrt(eigenvalues)

                        # Diffusion coefficients along principal axes
                        # Larger elongation → smaller diffusion
                        sigma_principal = c2 / np.sqrt(elongation_principal + epsilon_sigma)

                        # Rotate back to coordinate frame
                        # Σ = R @ diag(σ_principal) @ R^T
                        sigma_full[i] = eigenvectors @ np.diag(sigma_principal) @ eigenvectors.T

                        # Ensure symmetry (numerical stability)
                        sigma_full[i] = 0.5 * (sigma_full[i] + sigma_full[i].T)

                        # Validate positive-definiteness
                        evals_check = np.linalg.eigvalsh(sigma_full[i])
                        if np.any(evals_check <= 0):
                            # Fallback to isotropic if not positive-definite
                            sigma_full[i] = np.eye(d) * isotropic_value

                    except Exception:
                        # Fallback to isotropic on any error
                        sigma_full[i] = np.eye(d) * isotropic_value
                else:
                    # Boundary or degenerate cell - use isotropic diffusion
                    sigma_full[i] = np.eye(d) * isotropic_value
            else:
                # No Voronoi - isotropic diffusion
                sigma_full[i] = np.eye(d) * isotropic_value

        return sigma_full
