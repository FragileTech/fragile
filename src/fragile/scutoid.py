"""
Scutoid Tessellation for Fragile Gas

This module implements the scutoid spacetime tessellation described in:
- old_docs/source/scutoid_integration.md
- old_docs/source/14_scutoid_geometry_framework.md
- old_docs/source/15_scutoid_curvature_raychaudhuri.md

A scutoid is a (d+1)-dimensional cell in spacetime connecting Voronoi regions
at consecutive timesteps. The tessellation captures the discrete spacetime
geometry of the Fragile Gas evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import numpy as np
from scipy.spatial import Voronoi
from torch import Tensor


if TYPE_CHECKING:
    from fragile.euclidean_gas import SwarmState


@dataclass
class SpacetimePoint:
    """A point in (d+1)-dimensional spacetime.

    Attributes:
        x: Spatial coordinates [d]
        t: Temporal coordinate (scalar)
    """

    x: np.ndarray  # Shape: (d,)
    t: float

    def to_vector(self) -> np.ndarray:
        """Convert to (d+1)-dimensional vector [x1, ..., xd, t]."""
        return np.concatenate([self.x, [self.t]])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> SpacetimePoint:
        """Create from (d+1)-dimensional vector."""
        return cls(x=v[:-1], t=v[-1])

    def __repr__(self) -> str:
        return f"SpacetimePoint(x={self.x}, t={self.t:.3f})"


@dataclass
class VoronoiCell:
    """Represents a single Voronoi cell at a fixed time.

    Attributes:
        walker_id: Index of walker at center
        center: Position of walker (generator point)
        vertices: List of vertices defining the cell boundary
        neighbors: List of neighboring walker IDs
        t: Timestep at which this cell exists
        volume: Spatial volume of the cell (computed)
    """

    walker_id: int
    center: np.ndarray  # Shape: (d,)
    vertices: list[np.ndarray]  # List of vertex positions
    neighbors: list[int]
    t: float
    volume: float | None = None


@dataclass
class Scutoid:
    """Represents a scutoid cell in the spacetime tessellation.

    A scutoid connects a Voronoi cell at time t (bottom face) to a Voronoi
    cell at time t+Δt (top face), with lateral faces connecting shared neighbors.

    Attributes:
        walker_id: Walker whose trajectory this cell represents
        parent_id: Parent walker ID at bottom timestep
        t_start: Start time
        t_end: End time
        bottom_vertices: Vertices of bottom Voronoi cell
        top_vertices: Vertices of top Voronoi cell
        bottom_neighbors: Neighbor IDs at t_start
        top_neighbors: Neighbor IDs at t_end
        mid_vertices: Mid-level vertices (for neighbor topology changes)
        bottom_center: Walker position at t_start
        top_center: Walker position at t_end
        volume: Spacetime volume (computed)
    """

    walker_id: int
    parent_id: int
    t_start: float
    t_end: float

    bottom_vertices: list[np.ndarray]
    top_vertices: list[np.ndarray]

    bottom_neighbors: list[int]
    top_neighbors: list[int]

    bottom_center: np.ndarray
    top_center: np.ndarray

    mid_vertices: list[SpacetimePoint] | None = None
    volume: float | None = None

    def is_prism(self) -> bool:
        """Check if this is a prism (no neighbor topology change)."""
        return set(self.bottom_neighbors) == set(self.top_neighbors)

    def shared_neighbors(self) -> list[int]:
        """Get neighbors present at both timesteps."""
        return list(set(self.bottom_neighbors) & set(self.top_neighbors))

    def lost_neighbors(self) -> list[int]:
        """Get neighbors lost from bottom to top."""
        return list(set(self.bottom_neighbors) - set(self.top_neighbors))

    def gained_neighbors(self) -> list[int]:
        """Get neighbors gained from bottom to top."""
        return list(set(self.top_neighbors) - set(self.bottom_neighbors))

    def __repr__(self) -> str:
        return (
            f"Scutoid(walker={self.walker_id}, parent={self.parent_id}, "
            f"t=[{self.t_start:.3f}, {self.t_end:.3f}], "
            f"type={'prism' if self.is_prism() else 'scutoid'})"
        )


@dataclass
class MetricFunction:
    """Encapsulates the spacetime metric for Riemannian computations.

    The spacetime metric is:
        g_ST = g_ij(x,t) dx^i ⊗ dx^j + α² dt ⊗ dt

    where g_ij(x,t) = H_ij(x,t) + ε_Σ δ_ij is the spatial metric.

    Attributes:
        fitness_hessian: Function H(x,t) → Hessian matrix [d,d]
        epsilon_sigma: Regularization parameter
        alpha: Temporal scale factor (default=1)
    """

    fitness_hessian: Callable[[np.ndarray, float], np.ndarray]
    epsilon_sigma: float
    alpha: float = 1.0

    def spatial_metric(self, x: np.ndarray, t: float) -> np.ndarray:
        """Compute spatial metric g_ij(x,t) = H(x,t) + ε_Σ I.

        Args:
            x: Spatial position [d]
            t: Time

        Returns:
            Spatial metric matrix [d, d]
        """
        d = len(x)
        H = self.fitness_hessian(x, t)
        return H + self.epsilon_sigma * np.eye(d)

    def spacetime_metric(self, x: np.ndarray, t: float) -> np.ndarray:
        """Compute full (d+1) × (d+1) spacetime metric.

        Args:
            x: Spatial position [d]
            t: Time

        Returns:
            Spacetime metric matrix [d+1, d+1]
        """
        d = len(x)
        g_spatial = self.spatial_metric(x, t)

        # Build block diagonal matrix
        g_ST = np.zeros((d + 1, d + 1))
        g_ST[:d, :d] = g_spatial
        g_ST[d, d] = self.alpha**2

        return g_ST

    def inverse_metric(self, x: np.ndarray, t: float) -> np.ndarray:
        """Compute inverse spacetime metric.

        Args:
            x: Spatial position [d]
            t: Time

        Returns:
            Inverse spacetime metric matrix [d+1, d+1]
        """
        return np.linalg.inv(self.spacetime_metric(x, t))


class ScutoidTessellation:
    """Container for the complete scutoid tessellation of a Fragile Gas run.

    Stores Voronoi cells at each timestep and scutoid cells connecting them,
    along with methods for computing geometric quantities.

    Attributes:
        N: Number of walkers
        d: Spatial dimension
        voronoi_cells: List of VoronoiCell lists, one list per timestep
        scutoid_cells: List of Scutoid lists, one list per time interval
        timesteps: List of time values
        metric_fn: MetricFunction for Riemannian computations
    """

    def __init__(
        self,
        N: int,
        d: int,
        metric_fn: MetricFunction | None = None,
    ):
        """Initialize empty tessellation.

        Args:
            N: Number of walkers
            d: Spatial dimension
            metric_fn: Optional metric function for volume computations
        """
        self.N = N
        self.d = d
        self.metric_fn = metric_fn

        # Storage for tessellation data
        self.voronoi_cells: list[list[VoronoiCell]] = []
        self.scutoid_cells: list[list[Scutoid]] = []
        self.timesteps: list[float] = []

        # Metadata
        self.n_steps = 0

    def add_timestep(
        self,
        state: SwarmState,
        timestep: int,
        t: float,
        parent_ids: Tensor | None = None,
    ) -> None:
        """Add a timestep to the tessellation.

        Computes Voronoi tessellation for the current walker positions,
        and if this is not the first timestep, constructs scutoid cells
        connecting to the previous timestep.

        Args:
            state: Swarm state at this timestep
            timestep: Timestep index
            t: Time value
            parent_ids: Parent walker indices (for cloning tracking)
        """
        # Convert positions to numpy
        x_np = state.x.detach().cpu().numpy()

        # Compute Voronoi tessellation
        voronoi_cells = self._compute_voronoi_cells(x_np, t)
        self.voronoi_cells.append(voronoi_cells)
        self.timesteps.append(t)

        # If this is not the first timestep, construct scutoids
        if timestep > 0:
            # Get parent IDs
            if parent_ids is None:
                # No cloning, each walker is its own parent
                parents = np.arange(self.N)
            else:
                parents = parent_ids.detach().cpu().numpy()

            # Construct scutoid cells
            scutoids = self._construct_scutoids(
                bottom_cells=self.voronoi_cells[-2],
                top_cells=self.voronoi_cells[-1],
                parent_ids=parents,
                t_start=self.timesteps[-2],
                t_end=self.timesteps[-1],
            )
            self.scutoid_cells.append(scutoids)

        self.n_steps = timestep + 1

    def _compute_voronoi_cells(
        self,
        positions: np.ndarray,
        t: float,
    ) -> list[VoronoiCell]:
        """Compute Voronoi tessellation for walker positions.

        Args:
            positions: Walker positions [N, d]
            t: Time value

        Returns:
            List of VoronoiCell objects, one per walker
        """
        N, _d = positions.shape

        # Compute Voronoi diagram using scipy
        vor = Voronoi(positions)

        # Extract cells for each walker
        cells = []
        for i in range(N):
            # Get region index for this point
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]

            # Get vertices (filter out -1 which means vertex at infinity)
            if -1 in region:
                # Cell extends to infinity - use bounding box approximation
                # For now, skip unbounded cells or use a large bounding box
                vertices = []
                neighbors = []
            else:
                vertices = [vor.vertices[idx] for idx in region]

                # Find neighbors (walkers sharing an edge)
                neighbors = self._find_neighbors(i, vor)

            cell = VoronoiCell(
                walker_id=i,
                center=positions[i].copy(),
                vertices=vertices,
                neighbors=neighbors,
                t=t,
                volume=None,  # Computed later if needed
            )
            cells.append(cell)

        return cells

    def _find_neighbors(self, walker_id: int, vor: Voronoi) -> list[int]:
        """Find neighboring walkers in Voronoi diagram.

        Args:
            walker_id: Index of walker
            vor: Scipy Voronoi object

        Returns:
            List of neighbor walker IDs
        """
        neighbors = set()

        # Find all ridge points (pairs of generators that share an edge)
        for ridge_points in vor.ridge_points:
            if walker_id in ridge_points:
                # This walker shares an edge with the other point
                other = ridge_points[0] if ridge_points[1] == walker_id else ridge_points[1]
                neighbors.add(other)

        return sorted(neighbors)

    def _construct_scutoids(
        self,
        bottom_cells: list[VoronoiCell],
        top_cells: list[VoronoiCell],
        parent_ids: np.ndarray,
        t_start: float,
        t_end: float,
    ) -> list[Scutoid]:
        """Construct scutoid cells connecting two Voronoi tessellations.

        Args:
            bottom_cells: Voronoi cells at t_start
            top_cells: Voronoi cells at t_end
            parent_ids: Parent walker for each walker [N]
            t_start: Start time
            t_end: End time

        Returns:
            List of Scutoid objects
        """
        scutoids = []

        for i in range(self.N):
            parent_id = int(parent_ids[i])

            scutoid = Scutoid(
                walker_id=i,
                parent_id=parent_id,
                t_start=t_start,
                t_end=t_end,
                bottom_vertices=bottom_cells[parent_id].vertices,
                top_vertices=top_cells[i].vertices,
                bottom_neighbors=bottom_cells[parent_id].neighbors,
                top_neighbors=top_cells[i].neighbors,
                bottom_center=bottom_cells[parent_id].center,
                top_center=top_cells[i].center,
                mid_vertices=None,  # Computed later if needed
                volume=None,
            )

            scutoids.append(scutoid)

        return scutoids

    def compute_volumes(self) -> None:
        """Compute volumes for all scutoid cells using the metric function."""
        if self.metric_fn is None:
            msg = "Cannot compute volumes without metric function"
            raise ValueError(msg)

        for scutoid_list in self.scutoid_cells:
            for scutoid in scutoid_list:
                # For now, use simple approximation
                # Full implementation would decompose into simplices
                scutoid.volume = self._approximate_scutoid_volume(scutoid)

    def _approximate_scutoid_volume(self, scutoid: Scutoid) -> float:
        """Approximate scutoid volume (placeholder implementation).

        Full implementation would use simplicial decomposition from
        scutoid_integration.md Algorithm 3.2.

        Args:
            scutoid: Scutoid cell

        Returns:
            Approximate spacetime volume
        """
        # Simple approximation: (spatial volume at start + spatial volume at end) / 2 * Δt
        dt = scutoid.t_end - scutoid.t_start

        # Estimate spatial volumes from number of vertices (very rough)
        # In production, would properly compute Voronoi cell volumes
        vol_bottom = len(scutoid.bottom_vertices) * 0.1  # Placeholder
        vol_top = len(scutoid.top_vertices) * 0.1  # Placeholder

        return 0.5 * (vol_bottom + vol_top) * dt

    def get_scutoid(self, walker_id: int, timestep: int) -> Scutoid | None:
        """Get scutoid cell for a specific walker and timestep.

        Args:
            walker_id: Walker index
            timestep: Timestep index (0 to n_steps-2)

        Returns:
            Scutoid cell or None if not found
        """
        if timestep < 0 or timestep >= len(self.scutoid_cells):
            return None

        scutoids_at_t = self.scutoid_cells[timestep]
        for scutoid in scutoids_at_t:
            if scutoid.walker_id == walker_id:
                return scutoid

        return None

    def summary_statistics(self) -> dict:
        """Compute summary statistics of the tessellation.

        Returns:
            Dictionary with statistics
        """
        n_prisms = 0
        n_scutoids = 0
        total_volume = 0.0

        for scutoid_list in self.scutoid_cells:
            for scutoid in scutoid_list:
                if scutoid.is_prism():
                    n_prisms += 1
                else:
                    n_scutoids += 1

                if scutoid.volume is not None:
                    total_volume += scutoid.volume

        return {
            "n_timesteps": len(self.voronoi_cells),
            "n_intervals": len(self.scutoid_cells),
            "n_prisms": n_prisms,
            "n_scutoids": n_scutoids,
            "total_spacetime_volume": total_volume,
            "N": self.N,
            "d": self.d,
        }
