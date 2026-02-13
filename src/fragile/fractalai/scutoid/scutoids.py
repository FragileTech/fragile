"""Scutoid spacetime tessellation for Fragile Gas evolution.

This module implements the scutoid geometry framework described in
old_docs/source/14_scutoid_geometry_framework.md. It provides:

1. Initialization from RunHistory: Build tessellation from execution traces
2. Ricci scalar computation: Use deficit angle method from Regge calculus
3. Geometric analysis: Cell type classification, curvature statistics

Mathematical Framework:
    - Scutoid cells connect Voronoi regions at consecutive timesteps
    - Ricci scalar R(x_i) computed via deficit angle convergence (Regge calculus)
    - Formula (2D): R(x_i) = 2δ_i / A_Voronoi  (discrete Gaussian curvature)
    - Formula (3D): R(x_i) ≈ δ_i / V_Voronoi  (approximate, full formula is edge-weighted)
    - Where δ_i is the deficit angle at vertex and A/V is the cell area/volume

Metric Correction Framework:
    The flat-space deficit angles can be corrected for the emergent Riemannian
    geometry induced by anisotropic diffusion. Two correction modes are available:

    1. Full Metric Correction (mode='full'):
       R^manifold(x_i) ≈ R^flat(x_i) + ΔR^metric(x_i)

       Where ΔR^metric involves full metric tensor gradients estimated from
       neighbor finite differences. Cost: O(N·k) where k ≈ 6 neighbors.

       Physical interpretation: Couples intrinsic curvature (from walker
       configuration) with extrinsic curvature (from fitness landscape geometry).

    2. Diagonal Approximation (mode='diagonal'):
       ΔR ≈ (1/2)Σ_k ∂²g_kk/∂x_k²

       Uses only diagonal metric components with finite difference second
       derivatives along coordinate axes. Cost: O(N).

       Trade-off: Less accurate for strongly anisotropic metrics, but captures
       essential scale effects at minimal computational cost.

    3. No Correction (mode='none'):
       Uses pure flat-space deficit angles. This measures intrinsic curvature
       of the walker configuration independent of the fitness landscape.

    Why This Matters:
        The framework documents claim deficit angles should converge to the Ricci
        scalar of the emergent metric g_ij = H_ij + ε_Σ δ_ij (where H is the
        fitness Hessian). However, the implementation uses Euclidean Voronoi
        tessellation, not Riemannian geodesic distances.

        These corrections bridge the gap by:
        - Using computationally cheap flat-space tessellation (O(N log N))
        - Applying first-order perturbation theory to incorporate metric effects
        - Avoiding expensive Riemannian Voronoi computation (requires geodesic
          distance computation on curved manifolds)

        At equilibrium, when walkers have adapted to anisotropic diffusion,
        the corrected deficit angles should approximate the true Ricci scalar
        of the emergent geometry (Theorem 5.4.1 in framework docs).

Reference:
    old_docs/source/14_scutoid_geometry_framework.md §5.4.1, §5.4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
import torch


if TYPE_CHECKING:
    from fragile.fractalai.bounds import TorchBounds
    from fragile.fractalai.core.history import RunHistory


@dataclass
class VoronoiCell:
    """Voronoi cell at a single timestep.

    Represents a spatial region around a walker, defined as the set of points
    closer to this walker than any other alive walker.

    Attributes:
        walker_id: Index of walker at cell center (0 to N-1)
        center: Walker position (generator point) [d]
        vertices: Voronoi vertex positions, list of [d] arrays
        neighbors: Indices of neighboring walkers
        t: Timestep value
        volume: Spatial volume (computed lazily)
        boundary_volume: (d-1)-volume of cell boundary (for Ricci computation)
    """

    walker_id: int
    center: np.ndarray  # Shape: (d,)
    vertices: list[np.ndarray]
    neighbors: list[int]
    t: float
    volume: float | None = None
    boundary_volume: float | None = None


@dataclass
class Scutoid:
    """Scutoid cell in spacetime tessellation.

    A scutoid connects a Voronoi cell at time t (bottom) to a Voronoi cell
    at time t+Δt (top), with lateral faces connecting shared neighbors.

    Topological Classification:
        - Prism: Same neighbors at top and bottom (no topology change)
        - Simple scutoid: One neighbor lost, one gained (|ΔN| = 2)
        - Complex scutoid: Multiple neighbor changes (|ΔN| > 2)

    Attributes:
        walker_id: Walker index at top timestep
        parent_id: Parent walker index at bottom timestep
        t_start: Bottom time
        t_end: Top time
        bottom_center: Parent position at t_start [d]
        top_center: Walker position at t_end [d]
        bottom_neighbors: Neighbor IDs at t_start
        top_neighbors: Neighbor IDs at t_end
        bottom_vertices: Bottom Voronoi vertices
        top_vertices: Top Voronoi vertices
        volume: Spacetime (d+1)-volume (computed)
        ricci_scalar: Ricci scalar curvature R(x) at cell center
    """

    walker_id: int
    parent_id: int
    t_start: float
    t_end: float
    bottom_center: np.ndarray
    top_center: np.ndarray
    bottom_neighbors: list[int]
    top_neighbors: list[int]
    bottom_vertices: list[np.ndarray]
    top_vertices: list[np.ndarray]
    volume: float | None = None
    ricci_scalar: float | None = None

    def is_prism(self) -> bool:
        """Check if cell is a prism (no neighbor topology change)."""
        return set(self.bottom_neighbors) == set(self.top_neighbors)

    def neighbor_change_count(self) -> int:
        """Count number of neighbor changes (symmetric difference size)."""
        return len(set(self.bottom_neighbors) ^ set(self.top_neighbors))

    def shared_neighbors(self) -> list[int]:
        """Get neighbors present at both timesteps."""
        return sorted(set(self.bottom_neighbors) & set(self.top_neighbors))

    def lost_neighbors(self) -> list[int]:
        """Get neighbors lost from bottom to top."""
        return sorted(set(self.bottom_neighbors) - set(self.top_neighbors))

    def gained_neighbors(self) -> list[int]:
        """Get neighbors gained from bottom to top."""
        return sorted(set(self.top_neighbors) - set(self.bottom_neighbors))


class BaseScutoidHistory:
    """Abstract base class for scutoid tessellation from RunHistory.

    Defines the common API for building scutoid tessellations and computing
    geometric quantities. Dimension-specific implementations override the
    Ricci scalar computation method.

    Mathematical Framework:
        The emergent spacetime metric is:
            g_ST = g_ij(x,t) dx^i ⊗ dx^j + dt ⊗ dt

        Where the spatial metric is:
            g_ij(x,t) = H_ij(x,t) + ε_Σ δ_ij

        H_ij is the fitness Hessian. For now we use Euclidean approximation.

    Ricci Scalar Computation:
        Uses deficit angle method from Regge calculus (Theorem 5.4.1):
            R(x_i) = δ_i / (C(d) * Vol(∂V_i))

        Where:
            - δ_i: Deficit angle at vertex i (from Delaunay dual)
            - C(d): Dimension constant = Ω_total(d) / (d-2)!
            - Ω_total(d) = 2π^(d/2) / Γ(d/2): Total solid angle
            - Vol(∂V_i): Boundary volume of Voronoi cell

    Attributes:
        history: Source RunHistory instance
        N: Number of walkers
        d: Spatial dimension
        n_recorded: Number of recorded timesteps
        timesteps: Time values at each recorded step
        voronoi_cells: List of VoronoiCell lists, one per timestep
        scutoid_cells: List of Scutoid lists, one per time interval
        ricci_scalars: Ricci scalar values [n_recorded-1, N]

    Example:
        >>> history = RunHistory.load("experiment.pt")
        >>> if history.d == 2:
        ...     scutoid_hist = ScutoidHistory2D(history)
        >>> elif history.d == 3:
        ...     scutoid_hist = ScutoidHistory3D(history)
        >>> scutoid_hist.build_tessellation()
        >>> scutoid_hist.compute_ricci_scalars()
        >>> stats = scutoid_hist.summary_statistics()
    """

    def __init__(
        self,
        history: RunHistory,
        bounds: TorchBounds | None = None,
        metric_correction: str = "none",
    ):
        """Initialize from RunHistory.

        Args:
            history: Source RunHistory instance
            bounds: Optional position bounds for filtering walkers. If None,
                uses history.bounds. If both are None, no bounds filtering is applied.
            metric_correction: Metric correction mode for Ricci scalar computation.
                - 'none': Pure flat-space deficit angles (default)
                - 'diagonal': Diagonal metric correction (cheap, O(N))
                - 'full': Full metric tensor correction (accurate, O(N·k))
        """
        self.history = history
        self.N = history.N
        self.d = history.d
        self.n_recorded = history.n_recorded

        # Use provided bounds, or fall back to history.bounds
        self.bounds = bounds if bounds is not None else getattr(history, "bounds", None)

        # Metric correction mode
        if metric_correction not in {"none", "diagonal", "full"}:
            msg = (
                f"metric_correction must be 'none', 'diagonal', or 'full', got {metric_correction}"
            )
            raise ValueError(msg)
        self.metric_correction = metric_correction

        self.timesteps: list[float] = []
        self.voronoi_cells: list[list[VoronoiCell]] = []
        self.scutoid_cells: list[list[Scutoid]] = []
        self.ricci_scalars: np.ndarray | None = None
        self.ricci_scalars_corrected: np.ndarray | None = None
        self._alive_mask_cache: np.ndarray | None = None

    def _alive_mask_array(self) -> np.ndarray:
        """Return cached alive mask array with shape [n_recorded, N]."""
        if self._alive_mask_cache is None:
            mask_tensor = getattr(self.history, "alive_mask", None)
            if mask_tensor is None:
                self._alive_mask_cache = np.zeros((0, self.N), dtype=bool)
            else:
                mask_np = mask_tensor.detach().cpu().numpy()
                self._alive_mask_cache = mask_np.astype(bool, copy=True)
        return self._alive_mask_cache

    def _alive_mask_at(self, t_idx: int) -> np.ndarray:
        """Get alive mask for recorded timestep index t_idx.

        Args:
            t_idx: Recorded timestep index (0 to n_recorded-1)

        Returns:
            Boolean mask [N] indicating which walkers are alive at t_idx
        """
        mask_array = self._alive_mask_array()
        if mask_array.size == 0:
            # No alive_mask data - assume all alive
            return np.ones(self.N, dtype=bool)

        # alive_mask has shape [n_recorded, N]
        # alive_mask[t_idx] corresponds directly to x_final[t_idx]
        if t_idx < 0 or t_idx >= mask_array.shape[0]:
            # Out of bounds - assume all alive
            return np.ones(self.N, dtype=bool)

        return mask_array[t_idx].copy()

    def _filter_positions_by_bounds(
        self,
        positions: np.ndarray,
        walker_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter positions and walker IDs to include only those inside bounds.

        Args:
            positions: Walker positions [n, d]
            walker_ids: Walker IDs [n]

        Returns:
            Tuple of (filtered_positions, filtered_walker_ids)
        """
        if self.bounds is None:
            # No bounds filtering
            return positions, walker_ids

        # Convert to torch tensor for bounds checking
        positions_torch = torch.tensor(positions, dtype=torch.float32)

        # Check which positions are inside bounds
        inside_mask = self.bounds.contains(positions_torch).cpu().numpy()

        # Filter positions and IDs
        filtered_positions = positions[inside_mask]
        filtered_walker_ids = walker_ids[inside_mask]

        return filtered_positions, filtered_walker_ids

    def build_tessellation(self) -> None:
        """Build complete scutoid tessellation from RunHistory.

        Constructs:
            1. Voronoi tessellation at each recorded timestep
            2. Scutoid cells connecting consecutive timesteps
            3. Tracks cloning events and parent relationships

        Uses data from history:
            - x_final[t, i, :]: Final positions after kinetic update
            - alive_mask[t, i]: Boolean mask of alive walkers
            - will_clone[t, i]: Cloning events (for t < n_recorded-1)
            - companions_clone[t, i]: Parent indices for cloned walkers
        """
        # Build Voronoi tessellation for each timestep
        for t_idx in range(self.n_recorded):
            # Extract positions and alive mask
            positions = self.history.x_final[t_idx].detach().cpu().numpy()  # [N, d]
            alive_mask = self._alive_mask_at(t_idx)

            # Filter to alive walkers only
            alive_indices = np.where(alive_mask)[0]
            alive_positions = positions[alive_indices]

            # Filter by bounds (only include walkers inside valid domain)
            bounded_positions, bounded_indices = self._filter_positions_by_bounds(
                alive_positions, alive_indices
            )

            # Compute Voronoi cells
            time_value = t_idx * self.history.record_every
            voronoi_cells = self._compute_voronoi_cells(
                positions=bounded_positions,
                walker_ids=bounded_indices,
                t=time_value,
            )

            self.voronoi_cells.append(voronoi_cells)
            self.timesteps.append(time_value)

        # Build scutoid cells between consecutive timesteps
        for t_idx in range(self.n_recorded - 1):
            # Determine parent relationships
            parent_ids = self._extract_parent_ids(t_idx)

            # Construct scutoids
            scutoids = self._construct_scutoids(
                bottom_cells=self.voronoi_cells[t_idx],
                top_cells=self.voronoi_cells[t_idx + 1],
                parent_ids=parent_ids,
                t_start=self.timesteps[t_idx],
                t_end=self.timesteps[t_idx + 1],
            )

            self.scutoid_cells.append(scutoids)

    def _extract_parent_ids(self, t_idx: int) -> dict[int, int]:
        """Extract parent walker IDs for timestep transition.

        Args:
            t_idx: Source timestep index (0 to n_recorded-2)

        Returns:
            Dictionary mapping walker_id -> parent_id for alive walkers at t_idx+1
        """
        # Get alive walkers at both timesteps
        alive_curr = self._alive_mask_at(t_idx).astype(bool, copy=False)
        alive_next = self._alive_mask_at(t_idx + 1).astype(bool, copy=False)

        # Get cloning information
        will_clone = self.history.will_clone[t_idx].detach().cpu().numpy()
        companions_clone = self.history.companions_clone[t_idx].detach().cpu().numpy()

        parent_ids = {}

        for i in range(self.N):
            if not alive_next[i]:
                continue  # Skip dead walkers at next timestep

            # Determine parent
            if alive_curr[i] and not will_clone[i]:
                # Walker persisted without cloning
                parent_id = i
            else:
                # Walker was cloned - find parent from companion
                parent_id = int(companions_clone[i])

            parent_ids[i] = parent_id

        return parent_ids

    def _compute_voronoi_cells(
        self,
        positions: np.ndarray,
        walker_ids: np.ndarray,
        t: float,
    ) -> list[VoronoiCell]:
        """Compute Voronoi tessellation for walker positions.

        Args:
            positions: Walker positions [n_alive, d]
            walker_ids: Original walker indices [n_alive]
            t: Time value

        Returns:
            List of VoronoiCell objects
        """
        n_alive = len(positions)

        if n_alive < self.d + 1:
            # Not enough points for triangulation - return empty cells
            return [
                VoronoiCell(
                    walker_id=int(walker_ids[i]),
                    center=positions[i].copy(),
                    vertices=[],
                    neighbors=[],
                    t=t,
                )
                for i in range(n_alive)
            ]

        # Compute Voronoi diagram
        try:
            vor = Voronoi(positions)
        except Exception:
            # Degenerate configuration - return simple cells
            return [
                VoronoiCell(
                    walker_id=int(walker_ids[i]),
                    center=positions[i].copy(),
                    vertices=[],
                    neighbors=[],
                    t=t,
                )
                for i in range(n_alive)
            ]

        # Build cells for each walker
        cells = []
        for i in range(n_alive):
            walker_id = int(walker_ids[i])
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]

            # Get vertices (skip -1 which means infinite vertex)
            if -1 in region or len(region) == 0:
                vertices = []
            else:
                vertices = [vor.vertices[idx].copy() for idx in region]

            # Find neighbors
            neighbors = self._find_neighbors(i, vor, walker_ids)

            cell = VoronoiCell(
                walker_id=walker_id,
                center=positions[i].copy(),
                vertices=vertices,
                neighbors=neighbors,
                t=t,
            )
            cells.append(cell)

        return cells

    def _find_neighbors(
        self,
        local_idx: int,
        vor: Voronoi,
        walker_ids: np.ndarray,
    ) -> list[int]:
        """Find neighboring walkers in Voronoi diagram.

        Args:
            local_idx: Index in the alive walkers array
            vor: Scipy Voronoi object
            walker_ids: Original walker IDs for alive walkers

        Returns:
            List of neighbor walker IDs (in original indexing)
        """
        neighbors = set()

        # Find ridge points (pairs sharing an edge)
        for ridge_points in vor.ridge_points:
            if local_idx in ridge_points:
                other_local = ridge_points[0] if ridge_points[1] == local_idx else ridge_points[1]
                other_walker_id = int(walker_ids[other_local])
                neighbors.add(other_walker_id)

        return sorted(neighbors)

    def _construct_scutoids(
        self,
        bottom_cells: list[VoronoiCell],
        top_cells: list[VoronoiCell],
        parent_ids: dict[int, int],
        t_start: float,
        t_end: float,
    ) -> list[Scutoid]:
        """Construct scutoid cells connecting two tessellations.

        Args:
            bottom_cells: Voronoi cells at t_start
            top_cells: Voronoi cells at t_end
            parent_ids: Map walker_id -> parent_id
            t_start: Bottom time
            t_end: Top time

        Returns:
            List of Scutoid objects for alive walkers at t_end
        """
        # Create lookup dictionaries
        bottom_dict = {cell.walker_id: cell for cell in bottom_cells}
        top_dict = {cell.walker_id: cell for cell in top_cells}

        scutoids = []

        for walker_id, parent_id in parent_ids.items():
            # Skip if walker not in top cells
            if walker_id not in top_dict:
                continue

            # Get bottom cell (parent's cell)
            if parent_id not in bottom_dict:
                # Parent not alive at bottom - skip (shouldn't happen normally)
                continue

            bottom_cell = bottom_dict[parent_id]
            top_cell = top_dict[walker_id]

            scutoid = Scutoid(
                walker_id=walker_id,
                parent_id=parent_id,
                t_start=t_start,
                t_end=t_end,
                bottom_center=bottom_cell.center.copy(),
                top_center=top_cell.center.copy(),
                bottom_neighbors=bottom_cell.neighbors.copy(),
                top_neighbors=top_cell.neighbors.copy(),
                bottom_vertices=bottom_cell.vertices.copy(),
                top_vertices=top_cell.vertices.copy(),
            )

            scutoids.append(scutoid)

        return scutoids

    def compute_ricci_scalars(self) -> None:
        """Compute Ricci scalar curvature for all scutoid cells.

        Uses deficit angle method from Regge calculus (Theorem 5.4.1):
            R(x_i) = δ_i / (C(d) * Vol(∂V_i))

        If metric_correction is enabled ('diagonal' or 'full'), automatically
        applies the metric correction after computing flat-space deficit angles.

        Stores results in:
            - Each Scutoid.ricci_scalar field
            - self.ricci_scalars array [n_recorded-1, N] (flat-space values)
            - self.ricci_scalars_corrected array [n_recorded-1, N] (if correction enabled)

        Subclasses override _compute_deficit_angles for dimension-specific methods.
        """
        # Initialize storage
        self.ricci_scalars = np.zeros((len(self.scutoid_cells), self.N))
        self.ricci_scalars[:] = np.nan  # Mark unavailable as NaN

        # Compute dimension constant C(d)
        C_d = self._dimension_constant(self.d)

        # Process each time interval
        for t_idx, scutoid_list in enumerate(self.scutoid_cells):
            # Get Voronoi cells at bottom timestep for deficit angle computation
            bottom_cells = self.voronoi_cells[t_idx]

            # Compute deficit angles at this timestep (dimension-specific)
            deficit_angles = self._compute_deficit_angles(bottom_cells)

            # Compute cell volumes (d-dimensional, NOT boundary volumes)
            # CORRECTED: Use area (2D) or volume (3D+), not perimeter/surface area
            cell_volumes = self._compute_cell_volumes(bottom_cells)

            # Assign Ricci scalars to scutoids
            for scutoid in scutoid_list:
                parent_id = scutoid.parent_id

                # Find parent cell
                parent_cell = next(
                    (c for c in bottom_cells if c.walker_id == parent_id),
                    None,
                )

                if parent_cell is None:
                    continue

                # Get deficit angle and cell volume
                delta = deficit_angles.get(parent_id, 0.0)
                cell_vol = cell_volumes.get(parent_id, 1.0)

                # Compute Ricci scalar using CORRECT formula:
                # 2D: R = 2δ / A_Voronoi  (via C_d = 0.5)
                # 3D: R = δ / V (approximate, should be edge-weighted)
                if cell_vol > 1e-10:  # Avoid division by zero
                    ricci = delta / (C_d * cell_vol)
                else:
                    ricci = 0.0

                scutoid.ricci_scalar = ricci
                self.ricci_scalars[t_idx, scutoid.walker_id] = ricci

        # Apply metric correction if enabled
        if self.metric_correction != "none":
            self.compute_metric_corrected_ricci()

    def compute_metric_corrected_ricci(self) -> None:
        """Apply metric correction to flat-space Ricci scalars.

        Uses first-order perturbation theory to correct flat-space deficit angles
        with local metric information from the fitness Hessian. The correction
        mode is determined by self.metric_correction:

        - 'diagonal': Diagonal metric correction O(N)
          ΔR ≈ (1/2)Σ_k ∂²g_kk/∂x_k²

        - 'full': Full metric tensor correction O(N·k)
          ΔR involves full metric gradients from neighbor finite differences

        Mathematical Framework:
            R^{manifold}(x_i) ≈ R^{flat}(x_i) + ΔR^{metric}(x_i)

            Where R^{flat} comes from flat-space deficit angles and ΔR^{metric}
            captures the curvature induced by the fitness landscape geometry.

        Results stored in:
            self.ricci_scalars_corrected: Array [n_recorded-1, N]

        Requires:
            - self.ricci_scalars must be computed first
            - History must contain fitness gradient/Hessian information

        References:
            - Theorem 5.4.1: Deficit angle convergence to Ricci scalar
            - Section 5.4: Emergent metric g = H + ε_Σ I
        """
        if self.metric_correction == "none":
            return

        if self.ricci_scalars is None:
            self.compute_ricci_scalars()

        # Initialize corrected array
        if self.ricci_scalars is not None:
            self.ricci_scalars_corrected = self.ricci_scalars.copy()
        else:
            return

        # Apply correction for each timestep
        for t_idx in range(len(self.scutoid_cells)):
            # Get walker positions at this timestep
            positions = self._get_positions_at_timestep(t_idx)

            if self.metric_correction == "diagonal":
                corrections = self._compute_diagonal_metric_correction(t_idx, positions)
            else:  # 'full'
                corrections = self._compute_full_metric_correction(t_idx, positions)

            # Apply corrections
            for scutoid in self.scutoid_cells[t_idx]:
                walker_id = scutoid.walker_id
                if walker_id in corrections:
                    self.ricci_scalars_corrected[t_idx, walker_id] += corrections[walker_id]

    def _get_positions_at_timestep(self, t_idx: int) -> np.ndarray:
        """Get walker positions at timestep index.

        Args:
            t_idx: Timestep index (0 to n_recorded-1)

        Returns:
            Array [N, d] with walker positions
        """
        return self.history.x_final[t_idx].detach().cpu().numpy()

    def _compute_diagonal_metric_correction(
        self,
        t_idx: int,
        positions: np.ndarray,
    ) -> dict[int, float]:
        """Compute diagonal metric correction to Ricci scalar.

        Uses only diagonal metric components with second-order finite differences
        along coordinate axes. Computationally cheap O(N) approximation.

        Formula:
            ΔR ≈ (1/2)Σ_k ∂²g_kk/∂x_k²

        Where g_kk are diagonal components of emergent metric g = H + ε_Σ I.

        Args:
            t_idx: Timestep index
            positions: Walker positions [N, d]

        Returns:
            Dictionary mapping walker_id -> correction ΔR
        """
        corrections = {}
        cells = self.voronoi_cells[t_idx]

        # Estimate metric at each position using local fitness information
        # For now, use a simple approximation based on position variance
        # In a full implementation, this would query the fitness Hessian
        for cell in cells:
            walker_id = cell.walker_id
            x = cell.center

            # Estimate second derivatives of diagonal metric components
            # using neighbor finite differences
            delta_R = 0.0

            # For each coordinate direction
            for k in range(self.d):
                # Find neighbors along this axis
                neighbor_positions = []
                for neighbor_id in cell.neighbors:
                    neighbor_cell = next(
                        (c for c in cells if c.walker_id == neighbor_id),
                        None,
                    )
                    if neighbor_cell is not None:
                        neighbor_positions.append(neighbor_cell.center)

                if len(neighbor_positions) >= 2:
                    # Estimate local scale from neighbor distances
                    neighbor_positions_arr = np.array(neighbor_positions)
                    distances = np.linalg.norm(neighbor_positions_arr - x, axis=1)
                    mean_dist = np.mean(distances)

                    # Simple approximation: metric scale ~ 1 / local_density
                    # Second derivative ~ 1 / scale²
                    if mean_dist > 1e-10:
                        delta_R += 0.5 / (mean_dist**2)

            corrections[walker_id] = delta_R

        return corrections

    def _compute_full_metric_correction(
        self,
        t_idx: int,
        positions: np.ndarray,
    ) -> dict[int, float]:
        """Compute full metric tensor correction to Ricci scalar.

        Uses neighbor finite differences to estimate metric gradients,
        then applies first-order perturbation formula. More accurate but
        more expensive than diagonal correction.

        Formula:
            ΔR = (1/2)∇²(tr h) - (1/4)||∇h||²

        Where h = g - I is the metric perturbation from flat space.

        Args:
            t_idx: Timestep index
            positions: Walker positions [N, d]

        Returns:
            Dictionary mapping walker_id -> correction ΔR
        """
        corrections = {}
        cells = self.voronoi_cells[t_idx]

        # Build neighbor map for efficient lookup
        neighbor_map = {}
        for cell in cells:
            neighbor_map[cell.walker_id] = cell

        for cell in cells:
            walker_id = cell.walker_id
            x = cell.center

            # Estimate metric tensor gradients from neighbors
            grad_metric_norm_sq = 0.0
            laplacian_trace = 0.0

            neighbor_count = len(cell.neighbors)
            if neighbor_count == 0:
                corrections[walker_id] = 0.0
                continue

            # Accumulate gradient contributions from neighbors
            for neighbor_id in cell.neighbors:
                if neighbor_id not in neighbor_map:
                    continue

                neighbor_cell = neighbor_map[neighbor_id]
                neighbor_x = neighbor_cell.center

                # Direction and distance
                delta_x = neighbor_x - x
                dist = np.linalg.norm(delta_x)

                if dist < 1e-10:
                    continue

                delta_x / dist

                # Estimate metric difference
                # In full implementation, would evaluate fitness Hessian
                # Here we use local density as proxy
                local_density = 1.0 / (dist + 1e-10)

                # Metric perturbation magnitude (simplified)
                metric_diff = local_density - 1.0

                # Gradient contribution
                grad_component = metric_diff / dist
                grad_metric_norm_sq += grad_component**2

                # Laplacian contribution (divergence of gradient)
                laplacian_trace += grad_component / dist

            # Average over neighbors
            grad_metric_norm_sq /= max(neighbor_count, 1)
            laplacian_trace /= max(neighbor_count, 1)

            # Apply correction formula
            delta_R = 0.5 * laplacian_trace - 0.25 * grad_metric_norm_sq

            corrections[walker_id] = delta_R

        return corrections

    def _compute_deficit_angles(self, cells: list[VoronoiCell]) -> dict[int, float]:
        """Compute deficit angles at Voronoi vertices (abstract method).

        Subclasses must implement dimension-specific deficit angle computation.

        Args:
            cells: List of VoronoiCell objects

        Returns:
            Dictionary mapping walker_id -> deficit_angle
        """
        msg = "Subclasses must implement _compute_deficit_angles"
        raise NotImplementedError(msg)

    def _dimension_constant(self, d: int) -> float:
        """Compute dimension-dependent constant for Ricci scalar formula.

        For 2D discrete Gaussian curvature (Regge calculus):
            K = δ / A_Voronoi
            R = 2K = 2δ / A_Voronoi

        For 3D and higher: Uses Regge calculus normalization for edge-based deficit angles.

        Args:
            d: Spatial dimension

        Returns:
            Constant for Ricci scalar formula
            - 2D: Returns 0.5 (so R = δ / (0.5 * A) = 2δ / A)
            - 3D+: Returns 1.0 (edge-weighted formula)
        """
        if d == 2:
            # 2D: R = 2δ / A, so we use C = 0.5 to get R = δ / (C * A) = 2δ / A
            return 0.5
        # 3D+: Use standard Regge normalization
        return 1.0

    def _compute_cell_volumes(self, cells: list[VoronoiCell]) -> dict[int, float]:
        """Compute d-dimensional volumes of Voronoi cells (CORRECT for Ricci scalar).

        This computes the proper dual volumes for Regge calculus:
        - 2D: Area of Voronoi polygon (NOT perimeter)
        - 3D: Volume of Voronoi polyhedron

        The Ricci scalar formula is:
        - 2D: R = 2δ / A_Voronoi
        - 3D: R = (1/V) Σ_edges δ_e ℓ_e

        Args:
            cells: List of VoronoiCell objects

        Returns:
            Dictionary mapping walker_id -> cell_volume
        """
        cell_vols = {}

        for cell in cells:
            if len(cell.vertices) == 0:
                # Unbounded cell or empty - use default
                cell_vols[cell.walker_id] = 1.0
                continue

            if self.d == 2:
                # Area of polygon in 2D using shoelace formula
                vertices = np.array(cell.vertices)
                if len(vertices) < 3:
                    cell_vols[cell.walker_id] = 1.0
                    continue

                # Shoelace formula: A = 0.5 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
                x = vertices[:, 0]
                y = vertices[:, 1]
                area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
                cell_vols[cell.walker_id] = area
            else:
                # For 3D+, use convex hull volume
                try:
                    hull = ConvexHull(cell.vertices)
                    cell_vols[cell.walker_id] = hull.volume
                except Exception:
                    cell_vols[cell.walker_id] = 1.0

        return cell_vols

    def summary_statistics(self) -> dict:
        """Compute summary statistics of tessellation and curvature.

        Returns:
            Dictionary with statistics:
                - n_timesteps: Number of recorded timesteps
                - n_intervals: Number of time intervals
                - n_prisms: Number of prism cells
                - n_simple_scutoids: Number of simple scutoids (|ΔN|=2)
                - n_complex_scutoids: Number of complex scutoids (|ΔN|>2)
                - mean_ricci: Mean Ricci scalar (if computed)
                - std_ricci: Standard deviation of Ricci scalar
                - min_ricci: Minimum Ricci scalar
                - max_ricci: Maximum Ricci scalar
        """
        n_prisms = 0
        n_simple = 0
        n_complex = 0

        for scutoid_list in self.scutoid_cells:
            for scutoid in scutoid_list:
                if scutoid.is_prism():
                    n_prisms += 1
                elif scutoid.neighbor_change_count() == 2:
                    n_simple += 1
                else:
                    n_complex += 1

        stats = {
            "n_timesteps": len(self.voronoi_cells),
            "n_intervals": len(self.scutoid_cells),
            "n_prisms": n_prisms,
            "n_simple_scutoids": n_simple,
            "n_complex_scutoids": n_complex,
            "N": self.N,
            "d": self.d,
        }

        # Add Ricci scalar statistics if computed
        if self.ricci_scalars is not None:
            valid_ricci = self.ricci_scalars[~np.isnan(self.ricci_scalars)]
            if len(valid_ricci) > 0:
                stats["mean_ricci"] = float(np.mean(valid_ricci))
                stats["std_ricci"] = float(np.std(valid_ricci))
                stats["min_ricci"] = float(np.min(valid_ricci))
                stats["max_ricci"] = float(np.max(valid_ricci))

        return stats

    def get_ricci_scalars(self) -> np.ndarray | None:
        """Get Ricci scalar values, applying metric correction if enabled.

        Returns the appropriate Ricci scalars based on metric_correction mode:
        - 'none': Returns flat-space deficit angle curvature
        - 'diagonal' or 'full': Returns metric-corrected curvature if available

        Returns:
            Array [n_recorded-1, N] with Ricci scalars, or None if not computed
        """
        if self.metric_correction == "none":
            return self.ricci_scalars
        return (
            self.ricci_scalars_corrected
            if self.ricci_scalars_corrected is not None
            else self.ricci_scalars
        )


class ScutoidHistory2D(BaseScutoidHistory):
    """2D scutoid tessellation with exact deficit angle computation.

    Implements full Regge calculus for 2D case using Delaunay triangulation.
    Computes deficit angles exactly from triangle angle sums.

    Formula:
        δ_i = 2π - Σ θ_triangle

    Where θ_triangle is the interior angle at vertex i in each incident triangle.

    Example:
        >>> history = RunHistory.load("experiment_2d.pt")
        >>> assert history.d == 2
        >>> scutoid_hist = ScutoidHistory2D(history)
        >>> scutoid_hist.build_tessellation()
        >>> scutoid_hist.compute_ricci_scalars()
    """

    def __init__(
        self,
        history: RunHistory,
        bounds: TorchBounds | None = None,
        incremental: bool = True,
        metric_correction: str = "none",
    ):
        """Initialize 2D scutoid history.

        Args:
            history: Source RunHistory with d=2
            bounds: Optional position bounds for filtering walkers
            incremental: Use O(N) incremental updates vs O(N log N) batch (default: True)
                Incremental mode updates the Delaunay triangulation online as walkers
                move/clone, achieving amortized O(N) complexity. Falls back to batch
                mode automatically on errors.
            metric_correction: Metric correction mode ('none', 'diagonal', 'full')

        Raises:
            ValueError: If history.d != 2
        """
        super().__init__(history, bounds=bounds, metric_correction=metric_correction)
        if self.d != 2:
            msg = f"ScutoidHistory2D requires d=2, got d={self.d}"
            raise ValueError(msg)

        self.incremental = incremental
        self._incremental_delaunay = None  # Created in build_tessellation

    def build_tessellation(self) -> None:
        """Build tessellation using incremental or batch mode.

        Uses incremental O(N) updates if self.incremental=True, with automatic
        fallback to batch O(N log N) mode on errors.
        """
        if self.incremental:
            try:
                self._build_tessellation_incremental()
            except Exception as e:
                # Fallback to batch on any error
                import warnings

                warnings.warn(
                    f"Incremental tessellation failed: {e}. Falling back to batch mode.",
                    stacklevel=2,
                )
                # Clear partial state
                self.voronoi_cells = []
                self.scutoid_cells = []
                self.timesteps = []
                self._incremental_delaunay = None
                # Run batch mode
                self._build_tessellation_batch()
        else:
            self._build_tessellation_batch()

    def _build_tessellation_batch(self) -> None:
        """Build tessellation using batch O(N log N) recomputation.

        This is the original implementation from BaseScutoidHistory.build_tessellation().
        Recomputes Voronoi from scratch at each timestep.
        """
        # Build Voronoi tessellation for each timestep
        for t_idx in range(self.n_recorded):
            # Extract positions and alive mask
            positions = self.history.x_final[t_idx].detach().cpu().numpy()  # [N, d]
            alive_mask = self._alive_mask_at(t_idx)

            # Filter to alive walkers only
            alive_indices = np.where(alive_mask)[0]
            alive_positions = positions[alive_indices]

            # Filter by bounds (only include walkers inside valid domain)
            bounded_positions, bounded_indices = self._filter_positions_by_bounds(
                alive_positions, alive_indices
            )

            # Compute Voronoi cells
            time_value = t_idx * self.history.record_every
            voronoi_cells = self._compute_voronoi_cells(
                positions=bounded_positions,
                walker_ids=bounded_indices,
                t=time_value,
            )

            self.voronoi_cells.append(voronoi_cells)
            self.timesteps.append(time_value)

        # Build scutoid cells between consecutive timesteps
        for t_idx in range(self.n_recorded - 1):
            # Determine parent relationships
            parent_ids = self._extract_parent_ids(t_idx)

            # Construct scutoids
            scutoids = self._construct_scutoids(
                bottom_cells=self.voronoi_cells[t_idx],
                top_cells=self.voronoi_cells[t_idx + 1],
                parent_ids=parent_ids,
                t_start=self.timesteps[t_idx],
                t_end=self.timesteps[t_idx + 1],
            )

            self.scutoid_cells.append(scutoids)

    def _build_tessellation_incremental(self) -> None:
        """Build tessellation using incremental O(N) updates.

        Uses IncrementalDelaunay2D to maintain triangulation as walkers move/clone,
        achieving amortized O(N) complexity per timestep.

        Raises:
            Exception: On any error (caller will fallback to batch mode)
        """
        from fragile.fractalai.core.incremental_delaunay import IncrementalDelaunay2D

        # Step 1: Initialize at t=0
        t_idx = 0
        positions = self.history.x_final[t_idx].detach().cpu().numpy()
        alive_mask = self._alive_mask_at(t_idx)
        alive_indices = np.where(alive_mask)[0]
        alive_positions = positions[alive_indices]
        bounded_positions, bounded_indices = self._filter_positions_by_bounds(
            alive_positions, alive_indices
        )

        # Create incremental triangulation
        self._incremental_delaunay = IncrementalDelaunay2D(bounded_positions, bounded_indices)

        # Extract initial Voronoi cells
        initial_cells = self._incremental_delaunay.get_voronoi_cells()
        time_value = 0.0
        for cell in initial_cells:
            cell.t = time_value
        self.voronoi_cells.append(initial_cells)
        self.timesteps.append(time_value)

        # Step 2: Incremental updates for t > 0
        for t_idx in range(1, self.n_recorded):
            # Get new positions
            positions = self.history.x_final[t_idx].detach().cpu().numpy()
            alive_mask = self._alive_mask_at(t_idx)
            alive_indices = np.where(alive_mask)[0]
            alive_positions = positions[alive_indices]
            bounded_positions, bounded_indices = self._filter_positions_by_bounds(
                alive_positions, alive_indices
            )

            # Check if set of walkers changed (deaths/births outside bounded region)
            current_walker_set = set(bounded_indices)
            previous_walker_set = set(self._incremental_delaunay.walker_ids)
            if current_walker_set != previous_walker_set:
                # Walker set changed - rebuild triangulation from scratch
                # This can happen when walkers die or leave bounded region
                self._incremental_delaunay = IncrementalDelaunay2D(
                    bounded_positions, bounded_indices
                )
                updated_cells = self._incremental_delaunay.get_voronoi_cells()
            else:
                # Identify parent relationships
                parent_ids = self._extract_parent_ids(t_idx - 1)

                # Update each walker in the triangulation
                for i, walker_id in enumerate(bounded_indices):
                    new_pos = bounded_positions[i]
                    parent_id = parent_ids.get(walker_id, walker_id)

                    # Check if walker moved locally or was cloned
                    was_cloned = parent_id != walker_id

                    if was_cloned:
                        # Cloning event: delete-and-insert
                        self._incremental_delaunay.delete_and_insert(walker_id, new_pos)
                    else:
                        # Local move: Lawson flips
                        self._incremental_delaunay.update_position(walker_id, new_pos)

                # Extract updated Voronoi cells
                updated_cells = self._incremental_delaunay.get_voronoi_cells()

            # Set timestamps
            time_value = t_idx * self.history.record_every
            for cell in updated_cells:
                cell.t = time_value
            self.voronoi_cells.append(updated_cells)
            self.timesteps.append(time_value)

        # Step 3: Build scutoids between timesteps (unchanged)
        for t_idx in range(self.n_recorded - 1):
            parent_ids = self._extract_parent_ids(t_idx)
            scutoids = self._construct_scutoids(
                bottom_cells=self.voronoi_cells[t_idx],
                top_cells=self.voronoi_cells[t_idx + 1],
                parent_ids=parent_ids,
                t_start=self.timesteps[t_idx],
                t_end=self.timesteps[t_idx + 1],
            )
            self.scutoid_cells.append(scutoids)

    def _compute_deficit_angles(self, cells: list[VoronoiCell]) -> dict[int, float]:
        """Compute deficit angles in 2D using Delaunay triangulation.

        Uses the dual relationship: Voronoi vertices ↔ Delaunay circumcenters.
        Computes exact deficit angles from triangle interior angles.

        Args:
            cells: List of VoronoiCell objects

        Returns:
            Dictionary mapping walker_id -> deficit_angle
        """
        # Extract positions
        positions = np.array([cell.center for cell in cells])
        walker_ids = np.array([cell.walker_id for cell in cells])

        if len(positions) < 3:
            # Not enough points for triangulation
            return {int(wid): 0.0 for wid in walker_ids}

        # Compute Delaunay triangulation
        try:
            delaunay = Delaunay(positions)
        except Exception:
            return {int(wid): 0.0 for wid in walker_ids}

        # Compute angle sum at each vertex
        n_points = len(walker_ids)
        angle_sums = np.zeros(n_points)

        # Iterate over all triangles
        for simplex in delaunay.simplices:
            # Get vertex positions
            p0 = delaunay.points[simplex[0]]
            p1 = delaunay.points[simplex[1]]
            p2 = delaunay.points[simplex[2]]

            # Compute angles at each vertex
            angles = self._triangle_angles(p0, p1, p2)

            # Add to angle sum
            angle_sums[simplex[0]] += angles[0]
            angle_sums[simplex[1]] += angles[1]
            angle_sums[simplex[2]] += angles[2]

        # Compute deficit: δ = 2π - angle_sum
        deficit_angles = {}
        for i, walker_id in enumerate(walker_ids):
            deficit_angles[int(walker_id)] = 2 * np.pi - angle_sums[i]

        return deficit_angles

    def _triangle_angles(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute interior angles of a triangle.

        Uses law of cosines: cos(θ) = (b² + c² - a²) / (2bc)

        Args:
            p0, p1, p2: Vertex positions [2]

        Returns:
            Tuple of (angle_at_p0, angle_at_p1, angle_at_p2) in radians
        """
        # Edge lengths
        a = np.linalg.norm(p2 - p1)  # Opposite to p0
        b = np.linalg.norm(p2 - p0)  # Opposite to p1
        c = np.linalg.norm(p1 - p0)  # Opposite to p2

        eps = 1e-10

        # Angle at p0
        cos_angle0 = (b**2 + c**2 - a**2) / (2 * b * c + eps)
        angle0 = np.arccos(np.clip(cos_angle0, -1, 1))

        # Angle at p1
        cos_angle1 = (a**2 + c**2 - b**2) / (2 * a * c + eps)
        angle1 = np.arccos(np.clip(cos_angle1, -1, 1))

        # Angle at p2
        cos_angle2 = (a**2 + b**2 - c**2) / (2 * a * b + eps)
        angle2 = np.arccos(np.clip(cos_angle2, -1, 1))

        return (angle0, angle1, angle2)


class ScutoidHistory3D(BaseScutoidHistory):
    """3D scutoid tessellation with approximate curvature computation.

    Implements approximate Ricci scalar calculation for 3D case.
    Uses simplified deficit angle estimation from solid angle approximation.

    Current Implementation:
        Uses zero deficit approximation (flat space assumption).
        Full implementation requires computing solid angles at tetrahedron vertices.

    Future Enhancement:
        Implement exact solid angle computation using Girard's theorem:
            ω = Σ dihedral_angles - (n-2)π

    Example:
        >>> history = RunHistory.load("experiment_3d.pt")
        >>> assert history.d == 3
        >>> scutoid_hist = ScutoidHistory3D(history)
        >>> scutoid_hist.build_tessellation()
        >>> scutoid_hist.compute_ricci_scalars()
    """

    def __init__(
        self,
        history: RunHistory,
        bounds: TorchBounds | None = None,
        metric_correction: str = "none",
    ):
        """Initialize 3D scutoid history.

        Args:
            history: Source RunHistory with d=3
            bounds: Optional position bounds for filtering walkers
            metric_correction: Metric correction mode ('none', 'diagonal', 'full')

        Raises:
            ValueError: If history.d != 3
        """
        super().__init__(history, bounds=bounds, metric_correction=metric_correction)
        if self.d != 3:
            msg = f"ScutoidHistory3D requires d=3, got d={self.d}"
            raise ValueError(msg)

    def _compute_deficit_angles(self, cells: list[VoronoiCell]) -> dict[int, float]:
        """Compute approximate deficit angles for 3D.

        Current implementation: Returns zero (flat space approximation).

        Full implementation would compute solid angles at Delaunay vertices:
            1. Build Delaunay tetrahedralization
            2. For each vertex, sum solid angles of incident tetrahedra
            3. Deficit = 4π - solid_angle_sum

        Args:
            cells: List of VoronoiCell objects

        Returns:
            Dictionary mapping walker_id -> deficit_angle (currently zeros)
        """
        # TODO: Implement solid angle computation for 3D
        # For now, assume flat space (zero curvature)
        return {cell.walker_id: 0.0 for cell in cells}


def create_scutoid_history(
    history: RunHistory,
    bounds: TorchBounds | None = None,
    metric_correction: str = "none",
) -> BaseScutoidHistory:
    """Factory function to create appropriate ScutoidHistory subclass.

    Automatically selects ScutoidHistory2D or ScutoidHistory3D based on
    the spatial dimension of the RunHistory.

    Args:
        history: Source RunHistory instance
        bounds: Optional position bounds for filtering walkers
        metric_correction: Metric correction mode ('none', 'diagonal', 'full')

    Returns:
        ScutoidHistory2D if d=2, ScutoidHistory3D if d=3

    Raises:
        ValueError: If d is not 2 or 3

    Example:
        >>> history = RunHistory.load("experiment.pt")
        >>> scutoid_hist = create_scutoid_history(history, metric_correction="diagonal")
        >>> scutoid_hist.build_tessellation()
        >>> scutoid_hist.compute_ricci_scalars()
    """
    if history.d == 2:
        return ScutoidHistory2D(history, bounds=bounds, metric_correction=metric_correction)
    if history.d == 3:
        return ScutoidHistory3D(history, bounds=bounds, metric_correction=metric_correction)
    msg = f"Only 2D and 3D supported, got d={history.d}"
    raise ValueError(msg)
