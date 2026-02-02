"""Neighbor tracking system with virtual boundaries.

This module provides pure functional utilities for:
- Detecting nearby boundary faces
- Projecting positions onto boundaries
- Computing virtual boundary neighbors
- Converting between COO and CSR graph formats
- Efficient neighbor queries
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from scipy.spatial import Voronoi
import numpy as np


@dataclass
class BoundaryWallData:
    """Virtual boundary neighbor data.

    Attributes:
        positions: [W, d] virtual walker positions on boundary walls
        normals: [W, d] outward unit normals
        facet_areas: [W] intersection areas of Voronoi cells with walls
        distances: [W] distances from walkers to walls
        walker_indices: [W] which walker each wall belongs to
        face_ids: [W] which face (0..2*d-1) each wall is on
    """

    positions: Tensor
    normals: Tensor
    facet_areas: Tensor
    distances: Tensor
    walker_indices: Tensor
    face_ids: Tensor


def detect_nearby_boundary_faces(
    position: Tensor,
    bounds: Any,
    tolerance: float,
) -> list[int]:
    """Identify which domain faces are near a position.

    Face IDs (for d-dimensional space):
        - 2D: 0=x_low, 1=x_high, 2=y_low, 3=y_high
        - 3D: 0=x_low, 1=x_high, 2=y_low, 3=y_high, 4=z_low, 5=z_high

    Args:
        position: [d] walker position
        bounds: TorchBounds object
        tolerance: Distance threshold for "near" boundary

    Returns:
        List of face IDs (0..2*d-1) that are within tolerance
    """
    low = bounds.low
    high = bounds.high
    d = len(position)

    nearby_faces = []

    for dim in range(d):
        # Low face (2*dim)
        if position[dim] - low[dim] < tolerance:
            nearby_faces.append(2 * dim)

        # High face (2*dim + 1)
        if high[dim] - position[dim] < tolerance:
            nearby_faces.append(2 * dim + 1)

    return nearby_faces


def project_to_boundary_face(
    position: Tensor,
    face_id: int,
    bounds: Any,
) -> tuple[Tensor, Tensor]:
    """Project walker position onto boundary face.

    Args:
        position: [d] walker position
        face_id: Face ID (0..2*d-1)
        bounds: TorchBounds object

    Returns:
        (projected_position, outward_normal)
        - projected_position: [d] position on boundary face
        - outward_normal: [d] unit outward normal vector

    Example:
        position = [0.01, 0.5]
        face_id = 0  # x_low
        → ([0.0, 0.5], [-1.0, 0.0])
    """
    d = len(position)
    dim = face_id // 2  # Which dimension (0=x, 1=y, 2=z)
    is_high = face_id % 2 == 1  # High or low face

    # Copy position and project onto face
    projected = position.clone()
    if is_high:
        projected[dim] = bounds.high[dim]
    else:
        projected[dim] = bounds.low[dim]

    # Compute outward normal
    normal = torch.zeros(d, dtype=position.dtype, device=position.device)
    if is_high:
        normal[dim] = 1.0  # Outward from high face
    else:
        normal[dim] = -1.0  # Outward from low face

    return projected, normal


def estimate_boundary_facet_area(
    walker_idx: int,
    face_id: int,
    vor: Voronoi,
    positions: Tensor,
    bounds: Any,
) -> float:
    """Estimate area of Voronoi cell intersection with boundary face.

    Algorithm:
        1. Get Voronoi vertices for the walker's cell
        2. Find vertices near the boundary face
        3. Compute convex hull area of intersection
        4. Fallback: Use average facet area from walker neighbors

    Args:
        walker_idx: Index of walker
        face_id: Boundary face ID
        vor: scipy Voronoi object
        positions: [N, d] walker positions
        bounds: TorchBounds object

    Returns:
        Estimated facet area (positive float)
    """
    try:
        # Get Voronoi region for this walker
        region_idx = vor.point_region[walker_idx]
        vertex_indices = vor.regions[region_idx]

        # Filter out infinite vertices (-1)
        vertex_indices = [v for v in vertex_indices if v != -1]

        if len(vertex_indices) < 2:  # Need at least 2 vertices for area
            raise ValueError("Insufficient vertices")

        vertices = vor.vertices[vertex_indices]
        d = vertices.shape[1]

        # Determine which dimension and bound
        dim = face_id // 2
        is_high = face_id % 2 == 1
        bound_value = bounds.high[dim].item() if is_high else bounds.low[dim].item()

        # Find vertices near the boundary (within tolerance)
        tolerance = 1e-3
        near_boundary = np.abs(vertices[:, dim] - bound_value) < tolerance

        if np.sum(near_boundary) < 2:
            raise ValueError("Too few vertices near boundary")

        boundary_vertices = vertices[near_boundary]

        # Estimate area based on dimensionality
        if d == 2:
            # In 2D, boundary intersection is a line segment
            # Area = length of segment
            if len(boundary_vertices) >= 2:
                # Find the two extreme points
                other_dim = 1 - dim
                sorted_verts = boundary_vertices[boundary_vertices[:, other_dim].argsort()]
                area = np.linalg.norm(sorted_verts[-1] - sorted_verts[0])
            else:
                raise ValueError("Need 2 vertices for 2D facet")

        elif d == 3:
            # In 3D, boundary intersection is a polygon
            # Use convex hull area
            if len(boundary_vertices) >= 3:
                from scipy.spatial import ConvexHull

                # Project to 2D (remove the boundary dimension)
                dims_to_keep = [i for i in range(3) if i != dim]
                verts_2d = boundary_vertices[:, dims_to_keep]

                # Compute convex hull area
                hull = ConvexHull(verts_2d)
                area = hull.volume  # In 2D, volume is area
            else:
                raise ValueError("Need 3 vertices for 3D facet")
        else:
            raise ValueError(f"Unsupported dimension: {d}")

        return float(area)

    except (ValueError, IndexError, Exception):
        # Fallback: estimate from average neighbor facet area
        # Get ridge information for this walker
        walker_ridges = []
        for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
            if p1 == walker_idx or p2 == walker_idx:
                walker_ridges.append(ridge_idx)

        if walker_ridges:
            # Estimate facet area from ridge lengths
            ridge_areas = []
            for ridge_idx in walker_ridges:
                ridge_verts = vor.ridge_vertices[ridge_idx]
                if -1 not in ridge_verts and len(ridge_verts) >= 2:
                    verts = vor.vertices[ridge_verts]
                    if len(verts) == 2:  # 2D ridge is a line
                        area = np.linalg.norm(verts[1] - verts[0])
                    else:  # 3D ridge is a polygon
                        try:
                            from scipy.spatial import ConvexHull

                            hull = ConvexHull(verts)
                            area = hull.volume
                        except Exception:
                            area = 0.1  # Default fallback
                    ridge_areas.append(area)

            if ridge_areas:
                return float(np.mean(ridge_areas))

        # Ultimate fallback: use average cell size
        d = positions.shape[1]
        if d == 2:
            return 0.1  # Reasonable default for unit box
        else:
            return 0.05  # Smaller for 3D


def compute_boundary_neighbors(
    positions: Tensor,
    tier: Tensor,
    bounds: Any,
    vor: Voronoi,
    boundary_tolerance: float = 0.1,
) -> BoundaryWallData:
    """Compute virtual boundary neighbors for walkers near domain walls.

    Only walkers with tier <= 1 (boundary and adjacent walkers) are considered.

    Algorithm:
        1. Filter to tier 0/1 walkers
        2. For each walker, detect nearby boundary faces
        3. Project onto each nearby face → virtual walker position
        4. Estimate facet area from Voronoi geometry
        5. Compute distance and outward normal

    Args:
        positions: [N, d] walker positions
        tier: [N] boundary classification (0=boundary, 1=adjacent, 2+=interior)
        bounds: TorchBounds object
        vor: scipy Voronoi object
        boundary_tolerance: Distance threshold for detecting nearby boundaries

    Returns:
        BoundaryWallData with W total virtual walls
    """
    device = positions.device
    dtype = positions.dtype
    d = positions.shape[1]

    # Filter to boundary and adjacent walkers (tier <= 1)
    boundary_mask = tier <= 1
    boundary_walker_indices = torch.where(boundary_mask)[0]

    # Accumulate boundary wall data
    all_positions = []
    all_normals = []
    all_facet_areas = []
    all_distances = []
    all_walker_indices = []
    all_face_ids = []

    for walker_idx in boundary_walker_indices:
        walker_idx_item = walker_idx.item()
        position = positions[walker_idx_item]

        # Detect nearby boundary faces
        nearby_faces = detect_nearby_boundary_faces(position, bounds, boundary_tolerance)

        for face_id in nearby_faces:
            # Project onto boundary face
            proj_pos, normal = project_to_boundary_face(position, face_id, bounds)

            # Compute distance to wall
            distance = torch.norm(position - proj_pos).item()

            # Estimate facet area
            facet_area = estimate_boundary_facet_area(
                walker_idx_item, face_id, vor, positions, bounds
            )

            # Store data
            all_positions.append(proj_pos)
            all_normals.append(normal)
            all_facet_areas.append(facet_area)
            all_distances.append(distance)
            all_walker_indices.append(walker_idx_item)
            all_face_ids.append(face_id)

    # Convert to tensors
    if len(all_positions) == 0:
        # No boundary neighbors found
        return BoundaryWallData(
            positions=torch.empty(0, d, dtype=dtype, device=device),
            normals=torch.empty(0, d, dtype=dtype, device=device),
            facet_areas=torch.empty(0, dtype=dtype, device=device),
            distances=torch.empty(0, dtype=dtype, device=device),
            walker_indices=torch.empty(0, dtype=torch.long, device=device),
            face_ids=torch.empty(0, dtype=torch.long, device=device),
        )

    return BoundaryWallData(
        positions=torch.stack(all_positions).to(device),
        normals=torch.stack(all_normals).to(device),
        facet_areas=torch.tensor(all_facet_areas, dtype=dtype, device=device),
        distances=torch.tensor(all_distances, dtype=dtype, device=device),
        walker_indices=torch.tensor(all_walker_indices, dtype=torch.long, device=device),
        face_ids=torch.tensor(all_face_ids, dtype=torch.long, device=device),
    )


def create_extended_edge_index(
    edge_index: Tensor,
    boundary_data: BoundaryWallData,
    n_walkers: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Extend edge_index to include boundary neighbors.

    Virtual boundary walls are assigned indices N, N+1, ..., N+W-1.

    Args:
        edge_index: [2, E] walker-walker edges (indices 0..N-1)
        boundary_data: Virtual boundary neighbors
        n_walkers: Number of walkers (N)

    Returns:
        Tuple of:
        - edge_index_extended: [2, E+W] includes boundary edges
        - edge_distances_extended: [E+W] distances for all edges
        - facet_areas_extended: [E+W] facet areas for all edges
        - edge_types: [E+W] 0=walker-walker, 1=walker-boundary
    """
    device = edge_index.device
    n_boundary_edges = len(boundary_data.positions)

    if n_boundary_edges == 0:
        # No boundary edges to add
        # Return original edge_index with zero distances/areas/types
        n_edges = edge_index.shape[1]
        return (
            edge_index,
            torch.zeros(n_edges, dtype=torch.float32, device=device),
            torch.zeros(n_edges, dtype=torch.float32, device=device),
            torch.zeros(n_edges, dtype=torch.long, device=device),
        )

    # Create boundary edges: (walker_idx, N + boundary_idx)
    boundary_sources = boundary_data.walker_indices
    boundary_targets = torch.arange(
        n_walkers, n_walkers + n_boundary_edges, dtype=torch.long, device=device
    )
    boundary_edges = torch.stack([boundary_sources, boundary_targets], dim=0)  # [2, W]

    # Concatenate with walker edges
    edge_index_extended = torch.cat([edge_index, boundary_edges], dim=1)  # [2, E+W]

    # Create edge distances (boundary distances are known)
    n_walker_edges = edge_index.shape[1]
    walker_distances = torch.zeros(n_walker_edges, dtype=torch.float32, device=device)
    boundary_distances = boundary_data.distances
    edge_distances_extended = torch.cat([walker_distances, boundary_distances])

    # Create facet areas
    walker_facet_areas = torch.zeros(n_walker_edges, dtype=torch.float32, device=device)
    boundary_facet_areas = boundary_data.facet_areas
    facet_areas_extended = torch.cat([walker_facet_areas, boundary_facet_areas])

    # Create edge types (0=walker, 1=boundary)
    walker_types = torch.zeros(n_walker_edges, dtype=torch.long, device=device)
    boundary_types = torch.ones(n_boundary_edges, dtype=torch.long, device=device)
    edge_types = torch.cat([walker_types, boundary_types])

    return edge_index_extended, edge_distances_extended, facet_areas_extended, edge_types


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

    # Compute row pointers
    csr_ptr = torch.zeros(n_nodes + 1, dtype=torch.long, device=device)
    for i in range(n_edges):
        src = sorted_sources[i].item()
        csr_ptr[src + 1] += 1

    # Cumulative sum
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


def query_walker_neighbors(
    walker_idx: int,
    csr_ptr: Tensor,
    csr_indices: Tensor,
    csr_types: Tensor | None = None,
    filter_type: int | None = None,
) -> Tensor:
    """Fast O(1) neighbor query using CSR format.

    Args:
        walker_idx: Walker index (0..N-1)
        csr_ptr: [N+1] CSR row pointers
        csr_indices: [E] CSR column indices
        csr_types: Optional [E] edge types (0=walker, 1=boundary)
        filter_type: If provided, return only neighbors of this type
            0 = walker neighbors only
            1 = boundary neighbors only
            None = all neighbors

    Returns:
        [k] neighbor indices

    Example:
        # Get all neighbors
        neighbors = query_walker_neighbors(i, ptr, indices)

        # Get walker neighbors only
        walker_neighbors = query_walker_neighbors(
            i, ptr, indices, types, filter_type=0
        )
    """
    start = csr_ptr[walker_idx].item()
    end = csr_ptr[walker_idx + 1].item()

    if start == end:
        # No neighbors
        return torch.empty(0, dtype=torch.long, device=csr_indices.device)

    neighbors = csr_indices[start:end]

    if filter_type is not None and csr_types is not None:
        types = csr_types[start:end]
        mask = types == filter_type
        neighbors = neighbors[mask]

    return neighbors
