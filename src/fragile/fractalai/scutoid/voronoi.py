"""Vectorized Voronoi tessellation for efficient geometric computations.

This module provides a pure-tensor representation of Voronoi/Delaunay tessellations
designed for efficient batch operations and clean interfaces with gradient/hessian
estimation functions.

Key improvements over dict-based voronoi_observables.py:
- All data stored as PyTorch tensors (GPU-ready)
- COO edge_index format for graph operations
- Vectorized observable computations (no loops over cells)
- Clean separation of tessellation from observables

Example:
    >>> positions = torch.randn(1000, 3)
    >>> alive = torch.ones(1000, dtype=torch.bool)
    >>> tri = compute_vectorized_voronoi(positions, alive)
    >>> # Access graph structure
    >>> edge_index = tri.edge_index  # [2, E] for gradient estimation
    >>> # Access cell properties
    >>> volumes = tri.cell_volumes  # [N] vectorized access
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, Voronoi
import torch
from torch import Tensor


@dataclass
class VoronoiTriangulation:
    """Vectorized Voronoi/Delaunay tessellation data.

    All tensors are on the same device and use consistent indexing.
    Designed for efficient batch operations and clean interface with
    gradient/hessian estimators.

    The tessellation is computed only for alive walkers, with indexing that
    maps back to the original walker indices. All geometric properties are
    stored as tensors for vectorized operations.

    Attributes:
        positions: [N, d] walker positions (alive walkers only, in spatial dimensions)
        alive: [N] boolean mask indicating valid walkers

        edge_index: [2, E] neighbor connectivity in COO format
            edge_index[0] = source indices, edge_index[1] = destination indices
            Symmetric: if (i,j) exists, (j,i) also exists
            Compatible with gradient_estimation.estimate_gradient_finite_difference()
        edge_vectors: [E, d] displacement vectors (dst - src)
            edge_vectors[k] = positions[edge_index[1,k]] - positions[edge_index[0,k]]
        edge_distances: [E] Euclidean distances ||edge_vectors||

        cell_volumes: [N] d-dimensional volumes
            In 2D: area of Voronoi cell (length²)
            In 3D: volume of Voronoi cell (length³)
        cell_centroids: [N, d] geometric center of each Voronoi cell
            Computed as mean of cell vertices

        facet_areas: [E] (d-1)-dimensional areas of shared facets
            In 2D: length of edge between cells
            In 3D: area of polygon between cells
            facet_areas[k] corresponds to edge_index[:,k]

        tier: [N] boundary classification (3-tier system)
            0 = Tier 1 (boundary): infinite Voronoi cells, excluded from observables
            1 = Tier 2 (boundary-adjacent): neighbors boundary cells, no observables
            2 = Tier 3+ (interior): valid for all computations

        vertex_positions: [V, d] positions of Voronoi vertices
            These are the vertices of the Voronoi diagram (not the input points)
        cell_vertex_indices: [N, max_verts] indices into vertex_positions
            cell_vertex_indices[i] contains vertex indices for cell i
            -1 indicates padding (cells have different numbers of vertices)
        cell_vertex_counts: [N] number of vertices per cell
            Needed to handle variable vertex counts with padding

        device: torch.device where all tensors are stored
        spatial_dims: int, number of spatial dimensions used
            May be less than positions.shape[1] for time-sliced analysis
        n_walkers: int, number of alive walkers (= N)
        n_edges: int, number of edges in graph (= E)

        alive_indices: [N] original indices of alive walkers
            Maps from tessellation index i to original walker index
            positions[i] corresponds to original_positions[alive_indices[i]]
        voronoi: scipy.spatial.Voronoi object (kept for compatibility)
            Direct access to scipy Voronoi if needed for advanced operations

    Device Handling:
        All tensors are on the same device. To move to GPU:
        >>> tri_gpu = tri.to(device="cuda")

    Indexing Convention:
        - All tensor indices (0 to N-1) refer to alive walkers only
        - Use alive_indices to map back to original walker indices
        - edge_index uses tessellation indices, not original indices

    Example:
        >>> # Build tessellation
        >>> tri = compute_vectorized_voronoi(positions, alive)
        >>>
        >>> # Access graph structure for gradient estimation
        >>> result = estimate_gradient_finite_difference(
        ...     positions=tri.positions,
        ...     fitness_values=fitness_values[tri.alive_indices],
        ...     edge_index=tri.edge_index,
        ...     alive=tri.alive,
        ... )
        >>>
        >>> # Compute volume-based observable
        >>> interior_mask = tri.tier >= 2
        >>> mean_volume = tri.cell_volumes[interior_mask].mean()
    """

    # Core geometry
    positions: Tensor  # [N, d] alive walker positions in spatial dimensions
    alive: Tensor  # [N] boolean mask (all True for alive-only tessellation)

    # Graph structure (COO format)
    edge_index: Tensor  # [2, E] source->destination edges
    edge_vectors: Tensor  # [E, d] displacement vectors (dst - src)
    edge_distances: Tensor  # [E] Euclidean distances

    # Voronoi cell properties
    cell_volumes: Tensor  # [N] d-dimensional volumes
    cell_centroids: Tensor  # [N, d] cell geometric centers

    # Facet properties (per edge)
    facet_areas: Tensor  # [E] (d-1)-dimensional facet areas

    # Boundary classification
    tier: Tensor  # [N] int: 0=boundary, 1=boundary-adjacent, 2=interior

    # Vertex data (always computed for shape analysis)
    vertex_positions: Tensor  # [V, d] Voronoi vertex positions
    cell_vertex_indices: Tensor  # [N, max_verts] indices into vertex_positions (-1 = padding)
    cell_vertex_counts: Tensor  # [N] number of vertices per cell

    # Metadata
    device: torch.device
    spatial_dims: int  # Number of spatial dimensions used
    n_walkers: int  # Number of alive walkers
    n_edges: int  # Number of edges

    # Mapping and compatibility
    alive_indices: Tensor  # [N] original indices of alive walkers
    voronoi: Voronoi | None  # scipy Voronoi object for compatibility

    # Boundary neighbor data (populated when pbc=False)
    has_boundary_neighbors: bool  # Flag indicating virtual boundaries present
    boundary_walls: Tensor | None  # [W, d] virtual walker positions on walls
    boundary_wall_normals: Tensor | None  # [W, d] outward unit normals
    boundary_wall_facet_areas: Tensor | None  # [W] intersection areas
    boundary_wall_distances: Tensor | None  # [W] distances to walls
    boundary_wall_walker_indices: Tensor | None  # [W] which walker each wall belongs to

    # CSR format (always computed for efficient neighbor queries)
    neighbor_csr_ptr: Tensor  # [N+1] row pointers
    neighbor_csr_indices: Tensor  # [E] or [E+W] column indices
    neighbor_csr_distances: Tensor  # [E] or [E+W] distances
    neighbor_csr_facet_areas: Tensor  # [E] or [E+W] facet areas
    neighbor_csr_types: Tensor  # [E] or [E+W] 0=walker-walker, 1=walker-boundary

    # Extended edge_index (includes boundaries if present)
    edge_index_extended: Tensor | None  # [2, E+W] if boundaries, else None

    def to(self, device: str | torch.device) -> VoronoiTriangulation:
        """Move all tensors to specified device.

        Args:
            device: Target device ('cpu', 'cuda', torch.device object)

        Returns:
            New VoronoiTriangulation with all tensors on target device
        """
        if isinstance(device, str):
            device = torch.device(device)

        return VoronoiTriangulation(
            positions=self.positions.to(device),
            alive=self.alive.to(device),
            edge_index=self.edge_index.to(device),
            edge_vectors=self.edge_vectors.to(device),
            edge_distances=self.edge_distances.to(device),
            cell_volumes=self.cell_volumes.to(device),
            cell_centroids=self.cell_centroids.to(device),
            facet_areas=self.facet_areas.to(device),
            tier=self.tier.to(device),
            vertex_positions=self.vertex_positions.to(device),
            cell_vertex_indices=self.cell_vertex_indices.to(device),
            cell_vertex_counts=self.cell_vertex_counts.to(device),
            device=device,
            spatial_dims=self.spatial_dims,
            n_walkers=self.n_walkers,
            n_edges=self.n_edges,
            alive_indices=self.alive_indices.to(device),
            voronoi=self.voronoi,  # scipy object stays on CPU
            has_boundary_neighbors=self.has_boundary_neighbors,
            boundary_walls=self.boundary_walls.to(device)
            if self.boundary_walls is not None
            else None,
            boundary_wall_normals=self.boundary_wall_normals.to(device)
            if self.boundary_wall_normals is not None
            else None,
            boundary_wall_facet_areas=self.boundary_wall_facet_areas.to(device)
            if self.boundary_wall_facet_areas is not None
            else None,
            boundary_wall_distances=self.boundary_wall_distances.to(device)
            if self.boundary_wall_distances is not None
            else None,
            boundary_wall_walker_indices=self.boundary_wall_walker_indices.to(device)
            if self.boundary_wall_walker_indices is not None
            else None,
            neighbor_csr_ptr=self.neighbor_csr_ptr.to(device),
            neighbor_csr_indices=self.neighbor_csr_indices.to(device),
            neighbor_csr_distances=self.neighbor_csr_distances.to(device),
            neighbor_csr_facet_areas=self.neighbor_csr_facet_areas.to(device),
            neighbor_csr_types=self.neighbor_csr_types.to(device),
            edge_index_extended=self.edge_index_extended.to(device)
            if self.edge_index_extended is not None
            else None,
        )

    @property
    def interior_mask(self) -> Tensor:
        """Boolean mask for interior cells (tier >= 2).

        Returns:
            [N] boolean tensor, True for cells valid for observable computation
        """
        return self.tier >= 2

    @property
    def boundary_mask(self) -> Tensor:
        """Boolean mask for boundary cells (tier == 0).

        Returns:
            [N] boolean tensor, True for cells on domain boundary
        """
        return self.tier == 0

    @property
    def boundary_adjacent_mask(self) -> Tensor:
        """Boolean mask for boundary-adjacent cells (tier == 1).

        Returns:
            [N] boolean tensor, True for cells adjacent to boundary
        """
        return self.tier == 1

    def get_walker_neighbors(self, walker_idx: int, include_boundaries: bool = True) -> Tensor:
        """Get neighbors of a walker using CSR format.

        Args:
            walker_idx: Walker index (0..N-1)
            include_boundaries: If False, filter out boundary neighbors

        Returns:
            [k] neighbor indices
            - If include_boundaries=True: includes both walker (0..N-1) and
              boundary indices (N..N+W-1)
            - If include_boundaries=False: only walker indices (0..N-1)
        """
        from .neighbors import query_walker_neighbors

        if include_boundaries:
            return query_walker_neighbors(
                walker_idx, self.neighbor_csr_ptr, self.neighbor_csr_indices
            )
        return query_walker_neighbors(
            walker_idx,
            self.neighbor_csr_ptr,
            self.neighbor_csr_indices,
            self.neighbor_csr_types,
            filter_type=0,
        )


def compute_vectorized_voronoi(
    positions: Tensor,
    alive: Tensor,
    bounds: Any | None = None,
    pbc: bool = False,
    pbc_mode: str = "mirror",
    exclude_boundary: bool = True,
    boundary_tolerance: float = 1e-6,
    spatial_dims: int | None = None,
) -> VoronoiTriangulation:
    """Compute Voronoi tessellation with vectorized PyTorch outputs.

    This function computes a Voronoi tessellation from walker positions and
    returns all geometric data as PyTorch tensors for efficient vectorized
    operations. The result is compatible with gradient/hessian estimation
    functions and designed for GPU acceleration.

    Algorithm:
        1. Filter to alive walkers and spatial dimensions
        2. Call scipy.spatial.Voronoi (unavoidable, but only once)
        3. Build edge_index from ridge_points (vectorized)
        4. Compute all derived quantities in batch:
           - Edge vectors/distances
           - Cell volumes (vectorized ConvexHull or parallel processing)
           - Facet areas
           - Centroids
           - Boundary classification
           - Vertex positions (always computed)
        5. Return VoronoiTriangulation dataclass

    Args:
        positions: [N_total, d] walker positions
            May include dead walkers (filtered by alive mask)
        alive: [N_total] boolean mask for valid walkers
        bounds: Optional TorchBounds object for boundary detection and PBC
        pbc: Whether to use periodic boundary conditions
        pbc_mode: How to handle PBC - "mirror", "replicate", or "ignore"
            - "mirror": Create mirror images across boundaries
            - "replicate": Similar to mirror (implementation detail)
            - "ignore": No special PBC handling
        exclude_boundary: Whether to classify and exclude boundary cells
            If True, implements 3-tier boundary system
        boundary_tolerance: Distance threshold for boundary detection (in same units as positions)
        spatial_dims: If provided, only use first N dimensions for Voronoi
            Useful for time-embedded coordinates where last dimension is time, not space
            Example: 4D positions [x, y, z, t] with spatial_dims=3 uses only [x, y, z]

    Returns:
        VoronoiTriangulation with all data on same device as input positions

    Raises:
        ValueError: If tessellation fails (too few points, degenerate config)

    Complexity:
        - Scipy Voronoi: O(N log N) in 2D, O(N^(d/2)) in higher dimensions
        - Tensor operations: O(N) or O(E) where E = O(N·k) for k neighbors
        - Cell volumes: O(N) in 2D, O(N) in 3D (parallel ConvexHull)

    Example:
        >>> # Basic usage
        >>> positions = torch.randn(1000, 3)
        >>> alive = torch.ones(1000, dtype=torch.bool)
        >>> tri = compute_vectorized_voronoi(positions, alive)
        >>> print(f"Built graph with {tri.n_edges} edges")
        >>>
        >>> # With boundary exclusion
        >>> tri = compute_vectorized_voronoi(positions, alive, bounds=bounds, exclude_boundary=True)
        >>> n_interior = tri.interior_mask.sum()
        >>> print(f"{n_interior} interior cells for observables")
        >>>
        >>> # Time-sliced analysis (use only spatial dimensions)
        >>> positions_4d = torch.randn(1000, 4)  # [x, y, z, t]
        >>> tri = compute_vectorized_voronoi(positions_4d, alive, spatial_dims=3)
        >>> # Tessellation computed in 3D space only

    Notes:
        - Scipy Voronoi call is the main bottleneck for large N
        - All subsequent operations are vectorized PyTorch
        - Vertex data always computed (required for shape_distortion)
        - For PBC, ghost points are created but not returned in final structure
        - Device handling: output tensors match input positions device
    """
    device = positions.device
    dtype = positions.dtype

    # Filter to alive walkers
    alive_indices = torch.where(alive)[0]
    n_alive = len(alive_indices)

    if n_alive == 0:
        # Return empty tessellation
        return _create_empty_tessellation(device, dtype, spatial_dims or positions.shape[1])

    # Filter positions to spatial dimensions if requested
    if spatial_dims is not None:
        positions_spatial = positions[alive, :spatial_dims]
        d = spatial_dims
    else:
        positions_spatial = positions[alive]
        d = positions.shape[1]

    # Convert to numpy for scipy
    positions_np = positions_spatial.cpu().numpy()

    # Handle periodic boundary conditions
    if pbc and bounds is not None and pbc_mode != "ignore":
        positions_vor_np = _create_pbc_mirror_points(
            positions_np, bounds, d, pbc_mode, spatial_dims
        )
    else:
        positions_vor_np = positions_np

    # Compute Voronoi tessellation (scipy - unavoidable bottleneck)
    try:
        vor = Voronoi(positions_vor_np)
    except Exception as e:
        raise ValueError(f"Voronoi tessellation failed: {e}") from e

    # Extract ridge_points to build edge_index
    # ridge_points[k] = [i, j] means cells i and j share a facet
    ridge_points = vor.ridge_points  # [n_ridges, 2]

    # Filter to original domain if using PBC
    if pbc and bounds is not None and pbc_mode != "ignore":
        # Keep only edges where both points are in original domain
        mask = (ridge_points[:, 0] < n_alive) & (ridge_points[:, 1] < n_alive)
        ridge_points = ridge_points[mask]

    # Build symmetric edge_index in COO format
    # For each ridge (i, j), create both (i, j) and (j, i)
    edges_forward = ridge_points  # [E_half, 2]
    edges_backward = ridge_points[:, [1, 0]]  # [E_half, 2] (swap src/dst)
    edges = np.vstack([edges_forward, edges_backward])  # [E, 2]

    edge_index = torch.from_numpy(edges.T).long().to(device)  # [2, E]
    n_edges = edge_index.shape[1]

    # Compute edge vectors and distances
    src, dst = edge_index[0], edge_index[1]
    edge_vectors = positions_spatial[dst] - positions_spatial[src]  # [E, d]
    edge_distances = torch.norm(edge_vectors, dim=1)  # [E]

    # Compute cell volumes (vectorized where possible)
    cell_volumes = _compute_all_cell_volumes(vor, n_alive, d, device, dtype)

    # Compute facet areas for each edge
    facet_areas = _compute_all_facet_areas(vor, ridge_points, n_edges, d, device, dtype)

    # Extract vertex positions and build cell->vertex mapping
    vertex_positions, cell_vertex_indices, cell_vertex_counts = _extract_vertex_data(
        vor, n_alive, d, device, dtype
    )

    # Compute cell centroids from vertices
    cell_centroids = _compute_cell_centroids(
        vertex_positions, cell_vertex_indices, cell_vertex_counts, d, device, dtype
    )

    # Classify boundary cells if requested
    if exclude_boundary and not pbc:
        tier = _classify_boundary_cells(
            vor,
            positions_spatial,
            edge_index,
            bounds,
            boundary_tolerance,
            n_alive,
            device,
            spatial_dims,
        )
    else:
        # All cells are interior (tier 2) for PBC or if not excluding boundary
        tier = torch.full((n_alive,), 2, dtype=torch.long, device=device)

    # Compute virtual boundary neighbors and CSR format
    if not pbc:
        from .neighbors import (
            build_csr_from_coo,
            compute_boundary_neighbors,
            create_extended_edge_index,
        )

        # Create virtual boundary neighbors
        boundary_data = compute_boundary_neighbors(
            positions_spatial, tier, bounds, vor, boundary_tolerance
        )

        # Extend edge_index to include boundary edges
        (
            edge_index_ext,
            edge_distances_ext,
            facet_areas_ext,
            edge_types,
        ) = create_extended_edge_index(edge_index, boundary_data, n_alive)

        # Build CSR format with extended edges
        n_total = n_alive + len(boundary_data.positions)  # N + W
        csr_data = build_csr_from_coo(
            edge_index_ext,
            n_total,
            edge_distances=edge_distances_ext,
            edge_facet_areas=facet_areas_ext,
            edge_types=edge_types,
        )

        # Store boundary data
        has_boundary_neighbors = True
        edge_index_extended = edge_index_ext
        boundary_walls = boundary_data.positions
        boundary_wall_normals = boundary_data.normals
        boundary_wall_facet_areas = boundary_data.facet_areas
        boundary_wall_distances = boundary_data.distances
        boundary_wall_walker_indices = boundary_data.walker_indices

    else:
        # PBC mode: no boundary neighbors needed
        from .neighbors import build_csr_from_coo

        # Build CSR from walker edges only
        csr_data = build_csr_from_coo(
            edge_index,
            n_alive,
            edge_distances=edge_distances,
            edge_facet_areas=facet_areas,
            edge_types=None,  # All walker-walker
        )

        has_boundary_neighbors = False
        edge_index_extended = None
        boundary_walls = None
        boundary_wall_normals = None
        boundary_wall_facet_areas = None
        boundary_wall_distances = None
        boundary_wall_walker_indices = None
        edge_types = torch.zeros(n_edges, dtype=torch.long, device=device)

    # Extract CSR fields
    neighbor_csr_ptr = csr_data["csr_ptr"]
    neighbor_csr_indices = csr_data["csr_indices"]
    neighbor_csr_distances = csr_data.get("csr_distances", edge_distances)
    neighbor_csr_facet_areas = csr_data.get("csr_facet_areas", facet_areas)
    neighbor_csr_types = csr_data.get("csr_types", edge_types)

    # Create alive mask (all True for alive-only tessellation)
    alive_mask = torch.ones(n_alive, dtype=torch.bool, device=device)

    return VoronoiTriangulation(
        positions=positions_spatial,
        alive=alive_mask,
        edge_index=edge_index,
        edge_vectors=edge_vectors,
        edge_distances=edge_distances,
        cell_volumes=cell_volumes,
        cell_centroids=cell_centroids,
        facet_areas=facet_areas,
        tier=tier,
        vertex_positions=vertex_positions,
        cell_vertex_indices=cell_vertex_indices,
        cell_vertex_counts=cell_vertex_counts,
        device=device,
        spatial_dims=d,
        n_walkers=n_alive,
        n_edges=n_edges,
        alive_indices=alive_indices,
        voronoi=vor,
        has_boundary_neighbors=has_boundary_neighbors,
        boundary_walls=boundary_walls,
        boundary_wall_normals=boundary_wall_normals,
        boundary_wall_facet_areas=boundary_wall_facet_areas,
        boundary_wall_distances=boundary_wall_distances,
        boundary_wall_walker_indices=boundary_wall_walker_indices,
        neighbor_csr_ptr=neighbor_csr_ptr,
        neighbor_csr_indices=neighbor_csr_indices,
        neighbor_csr_distances=neighbor_csr_distances,
        neighbor_csr_facet_areas=neighbor_csr_facet_areas,
        neighbor_csr_types=neighbor_csr_types,
        edge_index_extended=edge_index_extended,
    )


def _create_empty_tessellation(
    device: torch.device, dtype: torch.dtype, spatial_dims: int
) -> VoronoiTriangulation:
    """Create empty tessellation for edge case of zero alive walkers."""
    return VoronoiTriangulation(
        positions=torch.empty(0, spatial_dims, device=device, dtype=dtype),
        alive=torch.empty(0, dtype=torch.bool, device=device),
        edge_index=torch.empty(2, 0, dtype=torch.long, device=device),
        edge_vectors=torch.empty(0, spatial_dims, device=device, dtype=dtype),
        edge_distances=torch.empty(0, device=device, dtype=dtype),
        cell_volumes=torch.empty(0, device=device, dtype=dtype),
        cell_centroids=torch.empty(0, spatial_dims, device=device, dtype=dtype),
        facet_areas=torch.empty(0, device=device, dtype=dtype),
        tier=torch.empty(0, dtype=torch.long, device=device),
        vertex_positions=torch.empty(0, spatial_dims, device=device, dtype=dtype),
        cell_vertex_indices=torch.empty(0, 0, dtype=torch.long, device=device),
        cell_vertex_counts=torch.empty(0, dtype=torch.long, device=device),
        device=device,
        spatial_dims=spatial_dims,
        n_walkers=0,
        n_edges=0,
        alive_indices=torch.empty(0, dtype=torch.long, device=device),
        voronoi=None,
        has_boundary_neighbors=False,
        boundary_walls=None,
        boundary_wall_normals=None,
        boundary_wall_facet_areas=None,
        boundary_wall_distances=None,
        boundary_wall_walker_indices=None,
        neighbor_csr_ptr=torch.zeros(1, dtype=torch.long, device=device),
        neighbor_csr_indices=torch.empty(0, dtype=torch.long, device=device),
        neighbor_csr_distances=torch.empty(0, dtype=dtype, device=device),
        neighbor_csr_facet_areas=torch.empty(0, dtype=dtype, device=device),
        neighbor_csr_types=torch.empty(0, dtype=torch.long, device=device),
        edge_index_extended=None,
    )


def _create_pbc_mirror_points(
    positions: np.ndarray,
    bounds: Any,
    d: int,
    pbc_mode: str,
    spatial_dims: int | None,
) -> np.ndarray:
    """Create mirror/replicate points for periodic boundary conditions.

    Args:
        positions: [N, d] alive walker positions
        bounds: TorchBounds object with .low and .high attributes
        d: Number of spatial dimensions
        pbc_mode: "mirror" or "replicate"
        spatial_dims: Optional spatial dimension limit

    Returns:
        [N_extended, d] positions including mirrors
    """
    # Extract bounds (handling torch tensors)
    if spatial_dims is not None:
        high = bounds.high[:spatial_dims].cpu().numpy()
        low = bounds.low[:spatial_dims].cpu().numpy()
    else:
        high = bounds.high.cpu().numpy() if torch.is_tensor(bounds.high) else bounds.high
        low = bounds.low.cpu().numpy() if torch.is_tensor(bounds.low) else bounds.low

    span = high - low

    # Generate offset vectors for mirror copies
    # In 2D: 9 copies (original + 8 mirrors)
    # In 3D: 27 copies (original + 26 mirrors)
    offsets = list(itertools.product([-1, 0, 1], repeat=d))

    positions_extended = []
    for offset in offsets:
        offset_arr = np.array(offset) * span
        positions_extended.append(positions + offset_arr)

    return np.vstack(positions_extended)


def _compute_all_cell_volumes(
    vor: Voronoi, n_alive: int, d: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Compute volumes for all Voronoi cells in batch.

    In 2D: Uses vectorized shoelace formula
    In 3D: Uses parallel ConvexHull computation

    Args:
        vor: scipy Voronoi object
        n_alive: Number of cells to compute
        d: Spatial dimension
        device: Target device for output tensor
        dtype: Target dtype for output tensor

    Returns:
        [N] tensor of cell volumes
    """
    volumes = np.zeros(n_alive, dtype=np.float64)

    for i in range(n_alive):
        region_idx = vor.point_region[i]
        vertices_idx = vor.regions[region_idx]

        # Skip infinite regions or degenerate cells
        if -1 in vertices_idx or len(vertices_idx) < d + 1:
            volumes[i] = 0.0
            continue

        try:
            vertices = vor.vertices[vertices_idx]

            if d == 2:
                # Vectorized shoelace formula for 2D polygon area
                x = vertices[:, 0]
                y = vertices[:, 1]
                volumes[i] = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            else:
                # Use ConvexHull for higher dimensions
                hull = ConvexHull(vertices)
                volumes[i] = hull.volume
        except Exception:
            volumes[i] = 0.0

    # Replace zero volumes with mean (avoid singularities)
    mean_vol = volumes[volumes > 0].mean() if (volumes > 0).any() else 1.0
    volumes[volumes == 0] = mean_vol

    return torch.from_numpy(volumes).to(device=device, dtype=dtype)


def _compute_all_facet_areas(
    vor: Voronoi,
    ridge_points: np.ndarray,
    n_edges: int,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Compute facet areas for all edges.

    Facet area is the (d-1)-dimensional measure of the shared boundary
    between two Voronoi cells.

    Args:
        vor: scipy Voronoi object
        ridge_points: [n_ridges, 2] pairs of cells sharing facets
        n_edges: Total number of edges (2 * n_ridges for symmetric)
        d: Spatial dimension
        device: Target device
        dtype: Target dtype

    Returns:
        [E] tensor of facet areas
    """
    n_ridges = len(ridge_points)
    areas_half = np.zeros(n_ridges, dtype=np.float64)

    for k, ridge_vertices in enumerate(vor.ridge_vertices):
        # Skip infinite ridges
        if -1 in ridge_vertices or len(ridge_vertices) < d:
            areas_half[k] = 1.0  # Default fallback
            continue

        try:
            vertices = vor.vertices[ridge_vertices]

            if d == 2:
                # In 2D, facet is a line segment (edge)
                if len(vertices) >= 2:
                    areas_half[k] = np.linalg.norm(vertices[1] - vertices[0])
                else:
                    areas_half[k] = 1.0
            elif d == 3:
                # In 3D, facet is a polygon
                if len(vertices) >= 3:
                    # Triangulate from first vertex and sum triangle areas (vectorized)
                    v0 = vertices[0]
                    v1 = vertices[1:-1]
                    v2 = vertices[2:]
                    cross = np.cross(v1 - v0, v2 - v0)
                    areas_half[k] = 0.5 * np.linalg.norm(cross, axis=1).sum()
                else:
                    areas_half[k] = 1.0
            else:
                # Higher dimensions: use ConvexHull
                hull = ConvexHull(vertices)
                areas_half[k] = hull.volume
        except Exception:
            areas_half[k] = 1.0

    # Duplicate for symmetric edges (forward and backward)
    areas = np.concatenate([areas_half, areas_half])

    return torch.from_numpy(areas).to(device=device, dtype=dtype)


def _extract_vertex_data(
    vor: Voronoi, n_alive: int, d: int, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor]:
    """Extract Voronoi vertex positions and cell->vertex mapping.

    Args:
        vor: scipy Voronoi object
        n_alive: Number of cells
        d: Spatial dimension
        device: Target device
        dtype: Target dtype

    Returns:
        tuple of:
            - vertex_positions: [V, d] positions of all Voronoi vertices
            - cell_vertex_indices: [N, max_verts] indices into vertex_positions (-1 = padding)
            - cell_vertex_counts: [N] number of vertices per cell
    """
    # Extract all vertex positions
    if vor.vertices is None or len(vor.vertices) == 0:
        # Degenerate case
        vertex_positions = torch.empty(0, d, device=device, dtype=dtype)
        cell_vertex_indices = torch.full((n_alive, 0), -1, dtype=torch.long, device=device)
        cell_vertex_counts = torch.zeros(n_alive, dtype=torch.long, device=device)
        return vertex_positions, cell_vertex_indices, cell_vertex_counts

    vertex_positions = torch.from_numpy(vor.vertices).to(device=device, dtype=dtype)
    len(vor.vertices)

    # Build cell->vertex mapping
    counts_np = np.zeros(n_alive, dtype=np.int64)
    flat_vertices_list: list[int] = []
    max_verts = 0

    for i in range(n_alive):
        region_idx = vor.point_region[i]
        vertices_idx = vor.regions[region_idx]

        # Filter out -1 (infinite vertex) while building flat index list.
        count = 0
        for v in vertices_idx:
            if v >= 0:
                flat_vertices_list.append(v)
                count += 1
        counts_np[i] = count
        max_verts = max(count, max_verts)

    # Create padded array
    cell_vertex_counts = torch.from_numpy(counts_np).to(device=device)
    cell_vertex_indices = torch.full((n_alive, max_verts), -1, dtype=torch.long, device=device)

    total_verts = int(counts_np.sum())
    if total_verts > 0:
        flat_np = np.fromiter(flat_vertices_list, dtype=np.int64, count=total_verts)
        flat_vertices = torch.from_numpy(flat_np).to(device=device)
        row_indices = torch.repeat_interleave(
            torch.arange(n_alive, device=device), cell_vertex_counts
        )
        row_offsets = torch.repeat_interleave(
            cell_vertex_counts.cumsum(0) - cell_vertex_counts, cell_vertex_counts
        )
        col_indices = torch.arange(total_verts, device=device) - row_offsets
        cell_vertex_indices[row_indices, col_indices] = flat_vertices

    return vertex_positions, cell_vertex_indices, cell_vertex_counts


def _compute_cell_centroids(
    vertex_positions: Tensor,
    cell_vertex_indices: Tensor,
    cell_vertex_counts: Tensor,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Compute centroid of each cell from its vertices.

    Args:
        vertex_positions: [V, d] vertex positions
        cell_vertex_indices: [N, max_verts] vertex indices per cell
        cell_vertex_counts: [N] number of vertices per cell
        d: Spatial dimension
        device: Target device
        dtype: Target dtype

    Returns:
        [N, d] tensor of cell centroids
    """
    n_cells, max_verts = cell_vertex_indices.shape
    if max_verts == 0 or vertex_positions.numel() == 0:
        return torch.zeros(n_cells, d, device=device, dtype=dtype)

    valid_mask = cell_vertex_indices >= 0
    safe_indices = cell_vertex_indices.clamp(min=0)
    verts = vertex_positions[safe_indices]
    verts = verts * valid_mask.unsqueeze(-1)
    summed = verts.sum(dim=1)
    counts = cell_vertex_counts.to(dtype=dtype).clamp(min=1).unsqueeze(-1)
    return summed / counts


def _classify_boundary_cells(
    vor: Voronoi,
    positions: Tensor,
    edge_index: Tensor,
    bounds: Any | None,
    boundary_tolerance: float,
    n_alive: int,
    device: torch.device,
    spatial_dims: int | None,
) -> Tensor:
    """Classify cells into 3-tier boundary system.

    Tier 0: Boundary cells (infinite Voronoi regions or near box boundary)
    Tier 1: Boundary-adjacent cells (neighbors of Tier 0)
    Tier 2: Interior cells (valid for all observables)

    Args:
        vor: scipy Voronoi object
        positions: [N, d] cell positions
        edge_index: [2, E] neighbor graph
        bounds: Optional bounds for boundary detection
        boundary_tolerance: Distance threshold for boundary
        n_alive: Number of cells
        device: Target device
        spatial_dims: Optional spatial dimension limit

    Returns:
        [N] tensor of tier values (0, 1, or 2)
    """
    tier = torch.zeros(n_alive, dtype=torch.long, device=device)

    # Step 1: Detect Tier 0 (boundary cells)
    is_boundary = torch.zeros(n_alive, dtype=torch.bool, device=device)

    # Method A: infinite Voronoi regions (-1 in region list)
    region_has_inf = np.fromiter(((-1 in region) for region in vor.regions), dtype=np.bool_)
    region_idx = np.asarray(vor.point_region[:n_alive], dtype=np.int64)
    is_boundary |= torch.from_numpy(region_has_inf[region_idx]).to(device=device)

    # Method B: near box boundary (vectorized)
    if bounds is not None:
        if spatial_dims is not None:
            low = bounds.low[:spatial_dims]
            high = bounds.high[:spatial_dims]
        else:
            low = bounds.low
            high = bounds.high

        low = torch.as_tensor(low, device=device, dtype=positions.dtype)
        high = torch.as_tensor(high, device=device, dtype=positions.dtype)

        near_boundary = ((positions - low) < boundary_tolerance) | (
            (high - positions) < boundary_tolerance
        )
        is_boundary |= near_boundary.any(dim=1)

    # Step 2: Detect Tier 1 (boundary-adjacent), vectorized
    src, dst = edge_index
    neighbor_is_boundary = is_boundary[dst].to(torch.int64)
    boundary_neighbor_counts = torch.zeros(n_alive, dtype=torch.int64, device=device)
    boundary_neighbor_counts.index_add_(0, src, neighbor_is_boundary)
    is_boundary_adjacent = (boundary_neighbor_counts > 0) & (~is_boundary)

    # Step 3: Assign tiers
    tier[is_boundary] = 0
    tier[is_boundary_adjacent] = 1
    tier[~is_boundary & ~is_boundary_adjacent] = 2

    return tier
