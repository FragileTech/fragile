"""Higgs field observables from emergent manifold geometry.

This module computes geometric observables that emerge from the anisotropic diffusion
process in the Fractal Gas framework, relating to the Higgs mechanism through the
emergent manifold structure.

Key concepts:
- Voronoi cell volumes encode local "spacetime density"
- Neighbor covariance defines the emergent metric tensor g_μν
- Centroid displacement (Lloyd vector) acts as gauge field/drift
- These combine to form a discretized Higgs action

The implementation reuses existing O(N) approximations from:
- voronoi_observables.compute_voronoi_tessellation() for volumes and neighbors
- voronoi_observables.compute_curvature_proxies() for Ricci scalars
- voronoi_time_slices.compute_time_sliced_voronoi() for Euclidean time analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.qft.voronoi_observables import (
    compute_voronoi_tessellation,
    compute_curvature_proxies,
)
from fragile.fractalai.qft.voronoi_time_slices import compute_time_sliced_voronoi


def _scatter_mean(src: Tensor, index: Tensor, dim: int, dim_size: int) -> Tensor:
    """Native PyTorch implementation of scatter_mean.

    Args:
        src: Source tensor to scatter [E, ...]
        index: Index tensor [E]
        dim: Dimension along which to scatter (always 0 in our usage)
        dim_size: Size of output dimension

    Returns:
        Scattered mean tensor [dim_size, ...]
    """
    # Create output tensor
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)

    # Scatter add for sum
    out.scatter_add_(dim, index.unsqueeze(1).expand_as(src), src)

    # Count occurrences
    ones = torch.ones(index.shape[0], dtype=src.dtype, device=src.device)
    count.scatter_add_(0, index, ones)

    # Avoid division by zero
    count = count.clamp(min=1)

    # Compute mean - handle different tensor dimensions
    if out.dim() == 1:
        # 1D case: out has shape [dim_size]
        out = out / count
    elif out.dim() >= 2:
        # Multi-dimensional case: out has shape [dim_size, ...]
        # Need to reshape count to broadcast correctly
        count_shape = [count.shape[0]] + [1] * (out.dim() - 1)
        out = out / count.view(*count_shape)
    else:
        # Scalar case (shouldn't happen)
        out = out / count

    return out


@dataclass
class HiggsConfig:
    """Configuration for Higgs field observable computation."""

    mc_time_index: int | None = None  # Which recorded slice to analyze
    warmup_fraction: float = 0.1
    h_eff: float = 1.0  # Effective Planck constant
    mu_sq: float = 1.0  # Higgs potential parameter -μ²φ²
    lambda_higgs: float = 0.5  # Higgs potential parameter λφ⁴
    alpha_gravity: float = 0.1  # Coupling strength for gravity term
    
    # Time slicing
    use_time_sliced_tessellation: bool = True
    euclidean_time_dim: int = 3
    euclidean_time_bins: int = 50
    
    # Computation flags
    compute_curvature: bool = True
    compute_action: bool = True
    
    # Voronoi parameters (passed through to existing functions)
    pbc_mode: str = "mirror"
    exclude_boundary: bool = True
    boundary_tolerance: float = 1e-6


@dataclass
class HiggsObservables:
    """Results from Higgs field observable computation."""

    # Geometric observables
    cell_volumes: Tensor  # [N] Voronoi cell volumes
    metric_tensors: Tensor  # [N, d, d] Emergent metric g_μν at each walker
    centroid_vectors: Tensor  # [N, d] Lloyd vectors (walker → centroid displacement)
    geodesic_distances: Tensor  # [E] Geodesic distances for each edge
    
    # Curvature proxies (reused from existing computation)
    ricci_scalars: Tensor | None  # [N] Ricci scalar proxy at each walker
    volume_variance: float  # Global curvature proxy
    curvature_proxies: dict[str, Any] | None  # Full curvature data from voronoi_observables
    
    # Higgs action components
    kinetic_term: float  # Spatial gradient energy
    potential_term: float  # V(φ) integrated over volumes
    gravity_term: float | None  # Einstein-Hilbert term ∫ R dV
    total_action: float
    
    # Field configuration
    scalar_field: Tensor  # [N] The Higgs field values (from fitness or custom)
    
    # Graph structure
    edge_index: Tensor  # [2, E] Edge connectivity
    alive: Tensor  # [N] Alive mask
    
    # Metadata
    n_walkers: int
    n_edges: int
    spatial_dims: int
    mc_frame: int
    config: HiggsConfig


def compute_emergent_metric(
    positions: Tensor,
    edge_index: Tensor,
    alive: Tensor,
) -> Tensor:
    """Compute emergent Riemannian metric from neighbor covariance.
    
    For each walker i, compute the covariance matrix of its neighbors:
    C_αβ = (1/k) Σ_j (x_j^α - x_i^α)(x_j^β - x_i^β)
    
    The metric tensor is: g_μν = (C^{-1})_μν
    
    This represents the local "stretching" of spacetime due to diffusion anisotropy.
    
    Args:
        positions: Walker positions [N, d]
        edge_index: Edge connectivity [2, E]
        alive: Alive mask [N]
        
    Returns:
        metric_tensors: [N, d, d] Symmetric positive-definite metric tensors
    """
    N, d = positions.shape
    device = positions.device
    
    # Initialize covariance matrices
    covariance = torch.zeros(N, d, d, device=device, dtype=positions.dtype)
    
    # Compute position differences for all edges
    row, col = edge_index[0], edge_index[1]
    diff = positions[col] - positions[row]  # [E, d]
    
    # Outer product: diff ⊗ diff
    outer_prod = diff.unsqueeze(2) * diff.unsqueeze(1)  # [E, d, d]
    
    # Aggregate to nodes using scatter_add
    covariance_flat = covariance.view(N, -1)
    covariance_flat.scatter_add_(0, row.unsqueeze(1).expand(-1, d * d), outer_prod.view(-1, d * d))
    covariance = covariance_flat.view(N, d, d)
    
    # Normalize by degree
    degree = torch.bincount(row, minlength=N).float().clamp(min=1)
    covariance = covariance / degree.view(-1, 1, 1)
    
    # Invert to get metric (with regularization for stability)
    epsilon = torch.eye(d, device=device, dtype=positions.dtype) * 1e-5
    metric_tensors = torch.linalg.inv(covariance + epsilon)
    
    # Ensure symmetry (due to numerical precision)
    metric_tensors = 0.5 * (metric_tensors + metric_tensors.transpose(-2, -1))
    
    return metric_tensors


def compute_centroid_displacement(
    positions: Tensor,
    edge_index: Tensor,
    alive: Tensor,
) -> Tensor:
    """Compute displacement from walker to neighbor centroid (Lloyd vector).
    
    L_i = (1/k) Σ_j x_j - x_i
    
    This acts as a gauge field/drift in the emergent geometry. In Lloyd's algorithm
    for Voronoi relaxation, this vector points toward the optimal position.
    
    Args:
        positions: Walker positions [N, d]
        edge_index: Edge connectivity [2, E]
        alive: Alive mask [N]
        
    Returns:
        displacement_vectors: [N, d] Lloyd vectors
    """
    N, d = positions.shape
    row, col = edge_index[0], edge_index[1]
    
    # Compute neighbor centroids using scatter_mean
    neighbor_centroids = _scatter_mean(positions[col], row, dim=0, dim_size=N)
    
    # Displacement = centroid - current position
    displacement = neighbor_centroids - positions
    
    # Zero out displacement for dead walkers
    displacement = displacement * alive.unsqueeze(1).float()
    
    # Validate output shape
    if displacement.dim() != 2:
        msg = (
            f"compute_centroid_displacement: expected 2D output [N, d], "
            f"got shape {displacement.shape}. This indicates a bug in _scatter_mean(). "
            f"positions shape: {positions.shape}, edge_index shape: {edge_index.shape}"
        )
        raise ValueError(msg)
    
    if displacement.shape != positions.shape:
        msg = (
            f"compute_centroid_displacement: displacement shape {displacement.shape} "
            f"doesn't match positions shape {positions.shape}"
        )
        raise ValueError(msg)
    
    return displacement


def compute_geodesic_distances(
    positions: Tensor,
    edge_index: Tensor,
    metric_tensors: Tensor,
    alive: Tensor,
) -> Tensor:
    """Compute geodesic distances using emergent metric.
    
    d_geo(i,j)² = Δx^T · g_ij · Δx
    where g_ij = (g_i + g_j) / 2 (interpolated metric on edge)
    
    Args:
        positions: Walker positions [N, d]
        edge_index: Edge connectivity [2, E]
        metric_tensors: Metric at each node [N, d, d]
        alive: Alive mask [N]
        
    Returns:
        geodesic_distances: [E] Geodesic length of each edge
    """
    row, col = edge_index[0], edge_index[1]
    
    # Position differences
    delta_x = positions[col] - positions[row]  # [E, d]
    
    # Interpolate metric to edge midpoint
    g_edge = 0.5 * (metric_tensors[row] + metric_tensors[col])  # [E, d, d]
    
    # Compute bilinear form: v^T G v
    # First: G * v
    Gv = torch.matmul(g_edge, delta_x.unsqueeze(2)).squeeze(2)  # [E, d]
    # Then: v^T * (G * v)
    dist_sq = (delta_x * Gv).sum(dim=1)  # [E]
    
    # Take square root (with epsilon for stability)
    geodesic_dist = torch.sqrt(dist_sq.clamp(min=1e-8))
    
    return geodesic_dist


def compute_ricci_scalars_from_curvature_proxies(
    curvature_proxies: dict[str, Any],
    n_walkers: int,
    device: torch.device,
) -> Tensor:
    """Extract Ricci scalar proxy from existing curvature computation.
    
    Reuses the volume_distortion and raychaudhuri_expansion from
    voronoi_observables.compute_curvature_proxies().
    
    The Raychaudhuri expansion θ ≈ -R for small curvature, so we use
    R ≈ -θ = -(dV/dt) / V as the primary proxy.
    
    Args:
        curvature_proxies: Output from compute_curvature_proxies()
        n_walkers: Number of walkers
        device: Torch device
        
    Returns:
        ricci_scalars: [N] Ricci scalar proxy at each walker
    """
    # Try Raychaudhuri expansion first (most physical)
    raychaudhuri = curvature_proxies.get("raychaudhuri_expansion")
    if raychaudhuri is not None and len(raychaudhuri) == n_walkers:
        # R ≈ -θ (expansion θ < 0 means positive curvature)
        ricci = -torch.from_numpy(raychaudhuri).to(device)
        return ricci
    
    # Fallback to volume distortion
    volume_distortion = curvature_proxies.get("volume_distortion")
    if volume_distortion is not None and len(volume_distortion) == n_walkers:
        # Use normalized volume deviation as curvature proxy
        ricci = torch.from_numpy(volume_distortion).to(device)
        return ricci
    
    # Last resort: zero curvature
    return torch.zeros(n_walkers, device=device)


def compute_higgs_action(
    scalar_field: Tensor,
    positions: Tensor,
    edge_index: Tensor,
    cell_volumes: Tensor,
    geodesic_distances: Tensor,
    ricci_scalars: Tensor | None = None,
    config: HiggsConfig | None = None,
) -> dict[str, float]:
    """Compute Higgs action components.
    
    S = S_kinetic - S_potential + α_gravity * S_gravity
    
    S_kinetic = (1/2) Σ_edges [(φ_i - φ_j) / d_geo(i,j)]² * d_geo(i,j)
    S_potential = Σ_i V(φ_i) * V_i
    S_gravity = Σ_i R_i * V_i
    
    where V(φ) = -μ²φ²/2 + λφ⁴/4 (Mexican hat potential)
    
    Args:
        scalar_field: Higgs field values [N]
        positions: Walker positions [N, d]
        edge_index: Edge connectivity [2, E]
        cell_volumes: Voronoi cell volumes [N]
        geodesic_distances: Geodesic edge lengths [E]
        ricci_scalars: Ricci scalar at each walker [N] (optional)
        config: Configuration (optional, uses defaults if None)
        
    Returns:
        dict with "kinetic", "potential", "gravity", "total"
    """
    if config is None:
        config = HiggsConfig()
    
    row, col = edge_index[0], edge_index[1]
    
    # 1. Kinetic term: (1/2) Σ (∇φ)² with proper measure
    phi_diff = scalar_field[col] - scalar_field[row]
    
    # Gradient squared: (Δφ / d)²
    grad_sq = (phi_diff / geodesic_distances.clamp(min=1e-8)) ** 2
    
    # Volume measure: use geodesic distance as "length element"
    kinetic_term = 0.5 * (grad_sq * geodesic_distances).sum().item()
    
    # 2. Potential term: ∫ V(φ) dV
    # V(φ) = -μ²φ²/2 + λφ⁴/4 (Mexican hat)
    potential_density = -0.5 * config.mu_sq * scalar_field**2 + 0.25 * config.lambda_higgs * scalar_field**4
    potential_term = (potential_density * cell_volumes).sum().item()
    
    # 3. Gravity term: ∫ R dV (Einstein-Hilbert)
    gravity_term = None
    if ricci_scalars is not None and config.compute_action:
        gravity_term = (ricci_scalars * cell_volumes).sum().item()
    
    # 4. Total action
    total = kinetic_term - potential_term
    if gravity_term is not None:
        total += config.alpha_gravity * gravity_term
    
    return {
        "kinetic": kinetic_term,
        "potential": potential_term,
        "gravity": gravity_term,
        "total": total,
    }


def _extract_scalar_field(
    history: RunHistory,
    mc_frame: int,
    scalar_field_source: str,
    custom_field: Tensor | None,
) -> Tensor:
    """Extract scalar field from history or custom source."""
    if custom_field is not None:
        return custom_field
    
    if scalar_field_source == "fitness":
        # Use fitness values (V_fit)
        if mc_frame == 0:
            # No fitness at t=0, use zeros
            return torch.zeros(history.N, device=history.x_final.device)
        idx = min(mc_frame - 1, len(history.fitness) - 1)
        return history.fitness[idx]
    
    elif scalar_field_source == "reward":
        # Use raw reward values
        if mc_frame == 0:
            return torch.zeros(history.N, device=history.x_final.device)
        idx = min(mc_frame - 1, len(history.rewards) - 1)
        return history.rewards[idx]
    
    elif scalar_field_source == "radius":
        # Use radial distance as field
        positions = history.x_final[mc_frame]
        return torch.norm(positions, dim=1)
    
    else:
        msg = f"Unknown scalar_field_source: {scalar_field_source}"
        raise ValueError(msg)


def compute_higgs_observables(
    history: RunHistory,
    config: HiggsConfig | None = None,
    scalar_field_source: str = "fitness",
    custom_field: Tensor | None = None,
) -> HiggsObservables:
    """Compute all Higgs field observables from RunHistory.
    
    This is the main entry point that:
    1. Selects the time slice (MC frame)
    2. Computes/loads Voronoi tessellation (reuses existing functions)
    3. Computes metric tensors, centroid vectors, geodesic distances
    4. Computes curvature proxies (Ricci scalar) - reuses existing implementation
    5. Computes Higgs action components
    
    Reuses existing code from:
    - voronoi_observables.compute_voronoi_tessellation()
    - voronoi_observables.compute_curvature_proxies()
    - voronoi_time_slices.compute_time_sliced_voronoi()
    
    Args:
        history: RunHistory from simulation
        config: Configuration (uses defaults if None)
        scalar_field_source: "fitness", "reward", "radius", or custom
        custom_field: Custom scalar field tensor [N] (overrides source)
        
    Returns:
        HiggsObservables dataclass with all computed observables
    """
    if config is None:
        config = HiggsConfig()
    
    # 1. Determine MC frame to analyze
    if config.mc_time_index is None:
        mc_frame = history.n_recorded - 1
    else:
        mc_frame = min(config.mc_time_index, history.n_recorded - 1)
    
    # 2. Extract positions and alive mask
    positions = history.x_final[mc_frame]
    if mc_frame == 0:
        alive = torch.ones(history.N, dtype=torch.bool, device=positions.device)
    else:
        idx = min(mc_frame - 1, len(history.alive_mask) - 1)
        alive = history.alive_mask[idx]
    
    # 3. Get or compute Voronoi tessellation
    # REUSE existing voronoi_observables.compute_voronoi_tessellation()
    voronoi_data = compute_voronoi_tessellation(
        positions=positions,
        alive=alive,
        bounds=history.bounds,
        pbc=history.pbc,
        pbc_mode=config.pbc_mode,
        exclude_boundary=config.exclude_boundary,
        boundary_tolerance=config.boundary_tolerance,
        compute_curvature=config.compute_curvature,
        prev_volumes=None,  # Could get from previous frame if available
        dt=history.delta_t,
        spatial_dims=history.d,
    )
    
    # 4. Extract Voronoi data
    volumes_np = voronoi_data["volumes"]
    neighbor_lists = voronoi_data["neighbor_lists"]
    curvature_proxies = voronoi_data.get("curvature_proxies")

    volume_weights_full = None
    if getattr(history, "riemannian_volume_weights", None) is not None and mc_frame > 0:
        info_idx = min(mc_frame - 1, len(history.riemannian_volume_weights) - 1)
        if info_idx >= 0:
            volume_weights_full = history.riemannian_volume_weights[info_idx]

    # Convert volumes to tensor (prefer Riemannian weights when available)
    if volume_weights_full is not None:
        cell_volumes = volume_weights_full.to(device=positions.device, dtype=positions.dtype)
    else:
        cell_volumes = torch.from_numpy(volumes_np).to(positions.device, dtype=positions.dtype)

    if config.compute_curvature and curvature_proxies is not None and volume_weights_full is not None:
        alive_indices = voronoi_data.get("alive_indices")
        if alive_indices is not None and len(alive_indices) > 0:
            alive_idx = torch.as_tensor(alive_indices, device=positions.device, dtype=torch.long)
            alive_idx = alive_idx[alive_idx < positions.shape[0]]
            if alive_idx.numel() > 0:
                volume_weights_alive = volume_weights_full[alive_idx].detach().cpu().numpy()
                positions_alive = positions[alive_idx].detach().cpu().numpy()
                curvature_proxies = compute_curvature_proxies(
                    voronoi_data=voronoi_data,
                    positions=positions_alive,
                    prev_volumes=None,
                    dt=history.delta_t,
                    volume_weights=volume_weights_alive,
                )
    
    # Build edge index from neighbor lists
    edges = []
    for i, neighbors in neighbor_lists.items():
        for j in neighbors:
            edges.append([i, j])
    
    if not edges:
        # No edges - degenerate case
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=positions.device)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long, device=positions.device).t()
    
    n_edges = edge_index.shape[1]
    
    # 5. Extract scalar field
    scalar_field = _extract_scalar_field(history, mc_frame, scalar_field_source, custom_field)
    
    # 6. Compute geometric observables
    metric_tensors = compute_emergent_metric(positions, edge_index, alive)
    centroid_vectors = compute_centroid_displacement(positions, edge_index, alive)
    geodesic_distances = compute_geodesic_distances(positions, edge_index, metric_tensors, alive)
    
    # 7. Compute curvature (REUSE existing curvature_proxies)
    ricci_scalars = None
    volume_variance = 0.0
    if config.compute_curvature and curvature_proxies is not None:
        ricci_scalars = compute_ricci_scalars_from_curvature_proxies(
            curvature_proxies, history.N, positions.device
        )
        volume_variance = float(curvature_proxies.get("volume_variance", 0.0))
    
    # 8. Compute Higgs action
    action_components = {"kinetic": 0.0, "potential": 0.0, "gravity": None, "total": 0.0}
    if config.compute_action and n_edges > 0:
        action_components = compute_higgs_action(
            scalar_field=scalar_field,
            positions=positions,
            edge_index=edge_index,
            cell_volumes=cell_volumes,
            geodesic_distances=geodesic_distances,
            ricci_scalars=ricci_scalars,
            config=config,
        )
    
    # 9. Return results
    return HiggsObservables(
        cell_volumes=cell_volumes,
        metric_tensors=metric_tensors,
        centroid_vectors=centroid_vectors,
        geodesic_distances=geodesic_distances,
        ricci_scalars=ricci_scalars,
        volume_variance=volume_variance,
        curvature_proxies=curvature_proxies,
        kinetic_term=action_components["kinetic"],
        potential_term=action_components["potential"],
        gravity_term=action_components["gravity"],
        total_action=action_components["total"],
        scalar_field=scalar_field,
        edge_index=edge_index,
        alive=alive,
        n_walkers=history.N,
        n_edges=n_edges,
        spatial_dims=history.d,
        mc_frame=mc_frame,
        config=config,
    )
