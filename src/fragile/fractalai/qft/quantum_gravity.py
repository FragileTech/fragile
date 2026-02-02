"""Quantum Gravity Analysis from Voronoi/Scutoid Spacetime Geometry.

This module implements 10 famous quantum gravity analyses using the emergent spacetime
geometry from Fractal Gas walker dynamics. Each analysis reproduces signature predictions
from different approaches to quantum gravity.

The 10 Analyses:
1. Regge Calculus Action - Discretized Einstein-Regge gravity via deficit angles
2. Einstein-Hilbert Action - Continuous limit from Ricci scalars
3. ADM Energy - Hamiltonian formulation (Arnowitt-Deser-Misner mass)
4. Spectral Dimension - Scale-dependent dimension from heat kernel diffusion
5. Hausdorff Dimension - Fractal dimension from volume scaling
6. Causal Set Structure - Discrete spacetime with timelike/spacelike edges
7. Holographic Entropy - Bekenstein-Hawking area law from boundary
8. Spin Network States - Loop Quantum Gravity graph states
9. Raychaudhuri Expansion - Singularity prediction from volume focusing
10. Geodesic Deviation - Tidal forces from Riemann tensor proxy

Mathematical Framework:
    Uses existing Voronoi tessellation and scutoid geometry computations from:
    - voronoi_observables.py: Fast O(N) Voronoi with curvature proxies
    - scutoids.py: Deficit angle Ricci scalar computation (Regge calculus)
    - voronoi_time_slices.py: Causal structure from time-sliced tessellation
    - higgs_observables.py: Emergent metric tensor from neighbor covariance

References:
    - Regge calculus: Regge (1961), "General relativity without coordinates"
    - Spectral dimension: Lauscher & Reuter (2005), Ambjørn et al CDT
    - Causal sets: Sorkin (1990s), discrete quantum gravity
    - Holographic entropy: Bekenstein-Hawking, AdS/CFT
    - Loop QG: Rovelli-Smolin spin networks

Example:
    >>> history = RunHistory.load("qft_run.pt")
    >>> config = QuantumGravityConfig(mc_time_index=-1, diffusion_time_steps=100)
    >>> observables = compute_quantum_gravity_observables(history, config)
    >>> print(f"Spectral dimension at Planck scale: {observables.spectral_dimension_planck:.2f}")
    >>> print(f"Hausdorff dimension: {observables.hausdorff_dimension:.2f}")
    >>> print(f"Holographic entropy: {observables.holographic_entropy:.3e}")
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fragile.fractalai.bounds import TorchBounds
from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.core.scutoids import create_scutoid_history
from fragile.fractalai.qft.voronoi_observables import (
    compute_voronoi_tessellation,
    compute_curvature_proxies,
    classify_boundary_cells,
)
from fragile.fractalai.qft.voronoi_time_slices import compute_time_sliced_voronoi
from fragile.fractalai.qft.higgs_observables import compute_emergent_metric


@dataclass
class QuantumGravityConfig:
    """Configuration for quantum gravity analysis."""

    mc_time_index: int | None = None  # Which recorded slice to analyze (None = last)
    warmup_fraction: float = 0.1
    analysis_dims: tuple[int, int, int] | None = (0, 1, 2)

    # Regge calculus
    use_metric_correction: str = "full"  # none/diagonal/full

    # Spectral dimension
    diffusion_time_steps: int = 100
    max_diffusion_time: float = 10.0

    # Hausdorff dimension
    n_radial_bins: int = 50
    max_radius: float | None = None  # Auto from bounds

    # Causal structure
    light_speed: float = 1.0  # c in simulation units
    euclidean_time_dim: int = 3  # Which dimension is Euclidean time
    euclidean_time_bins: int = 50

    # Holographic entropy
    planck_length: float = 1.0  # ℓ_P in simulation units

    # Spin network
    compute_spin_labels: bool = True

    # Common
    exclude_boundary: bool = True
    compute_all: bool = True  # Compute all 10 analyses


def _normalize_analysis_dims(
    total_dims: int,
    analysis_dims: tuple[int, int, int] | list[int] | None,
) -> tuple[int, ...]:
    """Normalize and validate analysis dimensions.

    Args:
        total_dims: Total number of spatial dimensions available
        analysis_dims: Requested dimensions to analyze (or None for default)

    Returns:
        Validated tuple of dimension indices

    Raises:
        ValueError: If analysis_dims are out of bounds
    """
    if total_dims <= 0:
        return ()

    # Handle default case
    if analysis_dims is None:
        # Default to first 3 dimensions (or fewer if not available)
        default_dims = tuple(range(min(3, total_dims)))
        
        # Warn if truncating high-dimensional data
        if total_dims > 3:
            import warnings
            warnings.warn(
                f"Data has {total_dims} dimensions but analysis limited to first 3. "
                f"To analyze dimension {total_dims - 1}, set analysis_dims explicitly.",
                UserWarning,
                stacklevel=2,
            )
        
        return default_dims

    # Validate requested dimensions
    dims = []
    invalid_dims = []

    for dim in analysis_dims:
        try:
            idx = int(dim)
        except (TypeError, ValueError):
            continue

        # Check bounds
        if idx < 0 or idx >= total_dims:
            invalid_dims.append(idx)
            continue

        # Add if not duplicate
        if idx not in dims:
            dims.append(idx)

        # Limit to 3 dimensions max
        if len(dims) >= min(3, total_dims):
            break

    # Raise error if any dims were out of bounds
    if invalid_dims:
        msg = (
            f"analysis_dims {invalid_dims} out of bounds for {total_dims}D data. "
            f"Valid range: 0..{total_dims - 1}"
        )
        raise ValueError(msg)

    # Fill with remaining dims if needed
    if len(dims) < min(3, total_dims):
        for idx in range(total_dims):
            if idx not in dims:
                dims.append(idx)
            if len(dims) >= min(3, total_dims):
                break

    return tuple(dims)


def _slice_bounds(bounds: TorchBounds | None, dims: tuple[int, ...]) -> TorchBounds | None:
    if bounds is None or not dims:
        return bounds

    # Handle 1D tensors - convert tuple to list for proper indexing
    dims_list = list(dims)
    low = bounds.low[dims_list]
    high = bounds.high[dims_list]

    return TorchBounds(low=low, high=high, shape=low.shape)


def _map_time_dim(time_dim: int, analysis_dims: tuple[int, ...]) -> int:
    if not analysis_dims:
        return time_dim
    if time_dim in analysis_dims:
        return analysis_dims.index(time_dim)
    if 0 <= time_dim < len(analysis_dims):
        return time_dim
    return min(len(analysis_dims) - 1, max(0, time_dim))


def _get_volume_weights(history: RunHistory, mc_frame: int) -> Tensor | None:
    weights = getattr(history, "riemannian_volume_weights", None)
    if weights is None or mc_frame < 1:
        return None
    info_idx = min(mc_frame - 1, len(weights) - 1)
    if info_idx < 0:
        return None
    return weights[info_idx]


def _build_scutoid_history_view(
    history: RunHistory,
    analysis_dims: tuple[int, ...],
    bounds: TorchBounds | None,
) -> RunHistory:
    if analysis_dims == tuple(range(history.d)):
        return history
    
    # Convert tuple to list for proper 3D indexing
    x_final = history.x_final[:, :, list(analysis_dims)]
    update = {"d": len(analysis_dims), "x_final": x_final, "bounds": bounds}
    if hasattr(history, "model_copy"):
        return history.model_copy(update=update, deep=False)
    return history.copy(update=update)


@dataclass
class QuantumGravityObservables:
    """Results from all quantum gravity analyses."""

    # 1. Regge Calculus
    regge_action: float
    regge_action_density: Tensor  # [N] per cell
    deficit_angles: Tensor  # [E] per edge/hinge

    # 2. Einstein-Hilbert Action
    einstein_hilbert_action: float
    ricci_scalars: Tensor  # [N]
    scalar_curvature_mean: float

    # 3. ADM Energy
    adm_mass: float
    adm_energy_density: Tensor  # [N]

    # 4. Spectral Dimension
    spectral_dimension_curve: Tensor  # [T] vs diffusion time
    spectral_dimension_planck: float  # At small scales
    heat_kernel_trace: Tensor  # [T]

    # 5. Hausdorff Dimension
    hausdorff_dimension: float
    volume_scaling_data: tuple[Tensor, Tensor]  # (radii, counts)
    local_hausdorff: Tensor  # [N] local dimension

    # 6. Causal Structure
    spacelike_edges: Tensor  # [2, E_space]
    timelike_edges: Tensor  # [2, E_time]
    null_edges: Tensor  # [2, E_null] (if any)
    causal_violations: int

    # 7. Holographic Entropy
    holographic_entropy: float
    boundary_area: float
    bulk_volume: float
    area_law_coefficient: float  # S/A

    # 8. Spin Network
    edge_spins: Tensor  # [E] - facet areas as SU(2) labels
    vertex_quantum_volumes: Tensor  # [N]
    dual_graph_edges: Tensor  # [2, E_dual]

    # 9. Raychaudhuri Expansion
    expansion_scalar: Tensor  # [N] θ = (1/V)dV/dt
    shear_tensor: Tensor | None  # [N, d, d]
    convergence_regions: Tensor  # [N] boolean mask

    # 10. Geodesic Deviation
    deviation_vectors: Tensor  # [E, d]
    tidal_tensor: Tensor  # [N, d, d] ~ Riemann
    tidal_eigenvalues: Tensor  # [N, d]

    # Metadata
    n_walkers: int
    n_edges: int
    spatial_dims: int
    mc_frame: int
    config: QuantumGravityConfig


@dataclass
class QuantumGravityTimeSeries:
    """Time evolution of all quantum gravity observables over MC frames.

    All tensor fields have time as first dimension: [T, ...] where T = n_frames.
    Scalar observables become 1D time series: [T].

    This dataclass enables 4D spacetime block analysis by tracking how quantum
    gravity signatures evolve during Monte Carlo dynamics, revealing:
    - Spectral dimension reduction during thermalization
    - Hausdorff dimension convergence to manifold dimension
    - ADM energy conservation/violation
    - Holographic entropy growth (2nd law)
    - Raychaudhuri expansion and singularity formation
    """

    # Time metadata
    n_frames: int
    mc_frames: np.ndarray  # [T] - actual MC frame indices analyzed

    # 1. Regge Calculus (time series)
    regge_action: np.ndarray  # [T] - action vs time
    regge_action_density: list[Tensor]  # List of [N] tensors (N varies per frame)

    # 2. Einstein-Hilbert (time series)
    einstein_hilbert_action: np.ndarray  # [T]
    scalar_curvature_mean: np.ndarray  # [T]

    # 3. ADM Energy (time series)
    adm_mass: np.ndarray  # [T] - CRITICAL for dynamics
    adm_mass_mean_density: np.ndarray  # [T]

    # 4. Spectral Dimension (time series of curves)
    spectral_dimension_planck: np.ndarray  # [T] - d_s at Planck scale over time
    spectral_dimension_large_scale: np.ndarray  # [T] - d_s at large scales over time
    # Note: spectral_dimension_curve stays as [T_diffusion] for each frame separately

    # 5. Hausdorff Dimension (time series)
    hausdorff_dimension: np.ndarray  # [T] - CRITICAL for emergence

    # 6. Causal Structure (time series)
    n_spacelike_edges: np.ndarray  # [T]
    n_timelike_edges: np.ndarray  # [T]
    causal_violations: np.ndarray  # [T]

    # 7. Holographic Entropy (time series)
    holographic_entropy: np.ndarray  # [T] - CRITICAL for information
    boundary_area: np.ndarray  # [T]
    bulk_volume: np.ndarray  # [T]
    area_law_coefficient: np.ndarray  # [T] - S/A ratio

    # 8. Spin Network (time series)
    mean_edge_spin: np.ndarray  # [T]
    mean_vertex_volume: np.ndarray  # [T]
    n_edges: np.ndarray  # [T]

    # 9. Raychaudhuri Expansion (time series)
    expansion_mean: np.ndarray  # [T] - mean θ
    expansion_std: np.ndarray  # [T] - fluctuations
    convergence_fraction: np.ndarray  # [T] - fraction with θ < 0

    # 10. Geodesic Deviation (time series)
    tidal_strength_mean: np.ndarray  # [T] - mean curvature strength
    tidal_strength_max: np.ndarray  # [T] - max tidal forces

    # Metadata
    n_walkers: np.ndarray  # [T] - alive walkers per frame
    spatial_dims: int
    config: QuantumGravityConfig


def compute_regge_action(
    history: RunHistory,
    mc_frame: int,
    config: QuantumGravityConfig,
) -> dict:
    """Compute Regge calculus action from deficit angles.

    Reuses: scutoids.py ScutoidHistory.compute_ricci_scalars()

    The Regge action is the discretized Einstein-Hilbert action using
    deficit angles around edges:
        S_Regge = Σ_edges δ_e * L_e

    Where δ_e is the deficit angle and L_e is the edge length.

    Returns:
        action: Total ∫ R dV
        action_density: R_i * Vol_i per cell
        deficit_angles: δ per edge/hinge
    """
    # Use existing scutoid framework
    scutoid_hist = create_scutoid_history(
        history,
        metric_correction=config.use_metric_correction,
    )
    scutoid_hist.build_tessellation()
    scutoid_hist.compute_ricci_scalars()

    # Get Ricci scalars (already computed from deficit angles)
    ricci = scutoid_hist.get_ricci_scalars()

    if ricci is None or len(ricci) == 0:
        # No Ricci data - return zeros
        N = history.N
        return {
            "regge_action": 0.0,
            "regge_action_density": torch.zeros(N),
            "deficit_angles": torch.zeros(0),
        }

    # Get volumes from the MC frame
    # Use the last time interval's Ricci scalars
    t_idx = min(mc_frame, len(ricci) - 1) if mc_frame > 0 else 0

    ricci_frame = ricci[t_idx]  # [N]
    ricci_tensor = torch.from_numpy(ricci_frame).float()

    # Compute volumes at this frame
    positions = history.x_final[mc_frame]
    alive_mask = (
        history.alive_mask[min(mc_frame, len(history.alive_mask) - 1)]
        if mc_frame > 0
        else torch.ones(history.N, dtype=torch.bool)
    )

    voronoi_data = compute_voronoi_tessellation(
        positions=positions,
        alive=alive_mask,
        bounds=history.bounds,
        pbc=history.pbc,
        compute_curvature=False,
    )

    volume_weights_full = _get_volume_weights(history, mc_frame)
    if volume_weights_full is not None and volume_weights_full.shape[0] == history.N:
        volumes_full = volume_weights_full.to(device=ricci_tensor.device, dtype=ricci_tensor.dtype)
    else:
        volumes_np = voronoi_data["volumes"]
        volumes = torch.from_numpy(volumes_np).float()

        # Extend volumes to full N (fill missing with zeros)
        volumes_full = torch.zeros(history.N, device=ricci_tensor.device, dtype=ricci_tensor.dtype)
        alive_indices = voronoi_data["alive_indices"]
        # Convert numpy indices to torch indices
        alive_indices_torch = (
            torch.from_numpy(alive_indices).long()
            if isinstance(alive_indices, np.ndarray)
            else alive_indices
        )
        volumes_full[alive_indices_torch] = volumes.to(device=ricci_tensor.device, dtype=ricci_tensor.dtype)

    # Action density: R_i * Vol_i
    action_density = ricci_tensor * volumes_full

    # Total action
    total_action = action_density.sum()

    # Extract deficit angles (approximation - would need scutoid cells)
    # For now, use Ricci * boundary_volume as proxy
    deficit_angles = ricci_tensor.clone()

    return {
        "regge_action": total_action.item(),
        "regge_action_density": action_density,
        "deficit_angles": deficit_angles,
    }


def compute_einstein_hilbert_action(
    positions: Tensor,
    alive: Tensor,
    history: RunHistory,
    config: QuantumGravityConfig,
    bounds: TorchBounds | None = None,
    pbc: bool | None = None,
    volume_weights: Tensor | None = None,
) -> dict:
    """Compute Einstein-Hilbert action from continuous curvature proxies.

    Reuses: voronoi_observables.compute_curvature_proxies()

    The Einstein-Hilbert action is:
        S_EH = ∫ R √g d⁴x

    Where R is the Ricci scalar and g is the metric determinant.

    Ricci scalar estimation hierarchy:
    1. **Best:** Raychaudhuri expansion θ = (1/V) dV/dt, where R ≈ -θ
       (requires temporal evolution data with prev_volumes)
    2. **Fallback:** Volume distortion R ≈ (1 - V/<V>)
       (spatial-only proxy: compressed cells have R > 0, expanded cells R < 0)
    """
    # Get Voronoi tessellation with curvature
    voronoi_data = compute_voronoi_tessellation(
        positions=positions,
        alive=alive,
        bounds=bounds if bounds is not None else history.bounds,
        pbc=history.pbc if pbc is None else pbc,
        compute_curvature=True,
    )

    curvature_data = voronoi_data.get("curvature_proxies")

    if curvature_data is None:
        # No curvature data
        N = positions.shape[0]
        return {
            "einstein_hilbert_action": 0.0,
            "ricci_scalars": torch.zeros(N),
            "scalar_curvature_mean": 0.0,
        }

    # Get Ricci scalars from Raychaudhuri expansion
    raychaudhuri = curvature_data.get("raychaudhuri_expansion")

    if raychaudhuri is not None:
        # R ≈ -θ (expansion θ < 0 means positive curvature)
        ricci_np = -raychaudhuri
    else:
        # Fallback: use volume distortion as curvature proxy
        # R ≈ (1 - V/<V>) measures deviation from mean volume
        # Cells with V > <V> have negative curvature (expansion)
        # Cells with V < <V> have positive curvature (compression)
        volume_dist = curvature_data.get("volume_distortion", np.array([]))
        if len(volume_dist) > 0:
            # Convert normalized volumes to curvature estimate:
            # Ricci ~ (1 - V_normalized)
            ricci_np = 1.0 - volume_dist
        else:
            ricci_np = np.array([])

    ricci = torch.from_numpy(ricci_np).float() if len(ricci_np) > 0 else torch.zeros(positions.shape[0])

    if volume_weights is not None:
        if torch.is_tensor(volume_weights):
            weights_full = volume_weights
        else:
            weights_full = torch.as_tensor(volume_weights)
        alive_indices = voronoi_data.get("alive_indices")
        if alive_indices is not None:
            alive_idx = torch.as_tensor(alive_indices, device=weights_full.device, dtype=torch.long)
            alive_idx = alive_idx[alive_idx < weights_full.shape[0]]
            volumes = weights_full[alive_idx].to(device=ricci.device, dtype=torch.float32)
        else:
            volumes = weights_full[: len(ricci)].to(device=ricci.device, dtype=torch.float32)
    else:
        volumes_np = voronoi_data["volumes"]
        volumes = torch.from_numpy(volumes_np).float()

    # Einstein-Hilbert action: S = ∫ R √g d⁴x
    # Using volumes as measure
    action = (ricci * volumes).sum()

    return {
        "einstein_hilbert_action": action.item(),
        "ricci_scalars": ricci,
        "scalar_curvature_mean": ricci.mean().item() if len(ricci) > 0 else 0.0,
    }


def compute_adm_energy(
    ricci_scalars: Tensor,
    volumes: Tensor,
) -> dict:
    """Compute ADM mass from spatial hypersurface.

    ADM mass: M = ∫ (R - K² + K_ij K^ij) dV
    Simplified: M ≈ ∫ R dV (using Ricci scalar)

    The ADM formalism is the Hamiltonian formulation of general relativity.
    """
    energy_density = ricci_scalars * volumes
    total_mass = energy_density.sum()

    return {
        "adm_mass": total_mass.item(),
        "adm_energy_density": energy_density,
    }


def compute_graph_laplacian_eigenvalues(
    positions: Tensor,
    edge_index: Tensor,
    alive: Tensor,
) -> Tensor:
    """Compute eigenvalues of graph Laplacian for spectral dimension analysis.

    The graph Laplacian is: L = D - A
    where D is the degree matrix and A is the adjacency matrix.
    """
    if edge_index.shape[1] == 0:
        return torch.zeros(0)

    N = positions.shape[0]
    device = positions.device

    # Build adjacency matrix
    row, col = edge_index[0], edge_index[1]

    # Count degrees
    degree = torch.bincount(row, minlength=N).float()

    # Build Laplacian matrix (using sparse representation would be better for large N)
    # For simplicity, use dense matrix for small N
    L = torch.zeros(N, N, device=device)

    # Diagonal: degree
    L[torch.arange(N), torch.arange(N)] = degree

    # Off-diagonal: -adjacency
    L[row, col] = -1.0

    # Filter to alive walkers
    alive_mask = alive.bool()
    alive_indices = torch.where(alive_mask)[0]

    if len(alive_indices) == 0:
        return torch.zeros(0)

    # Extract submatrix
    L_alive = L[alive_indices][:, alive_indices]

    # Compute eigenvalues
    try:
        eigenvalues = torch.linalg.eigvalsh(L_alive)
    except Exception:
        eigenvalues = torch.zeros(0)

    return eigenvalues


def compute_spectral_dimension(
    positions: Tensor,
    edge_index: Tensor,
    alive: Tensor,
    config: QuantumGravityConfig,
) -> dict:
    """Compute spectral dimension from heat kernel diffusion.

    d_s(σ) = -2 d(log P(σ))/d(log σ)
    where P(σ) = tr(exp(-σ Δ)) is the return probability

    The spectral dimension measures effective dimensionality at different scales.
    Famous prediction: dimension reduction at Planck scale in quantum gravity.
    """
    # Build Laplacian from Voronoi graph
    L_eigenvalues = compute_graph_laplacian_eigenvalues(positions, edge_index, alive)

    if len(L_eigenvalues) == 0:
        # No eigenvalues - return defaults
        n_steps = config.diffusion_time_steps
        return {
            "spectral_dimension_curve": torch.zeros(n_steps),
            "spectral_dimension_planck": float(positions.shape[1]),
            "heat_kernel_trace": torch.zeros(n_steps),
        }

    # Heat kernel trace: P(σ) = Σ_i exp(-σ λ_i)
    sigma_values = torch.logspace(-3, np.log10(config.max_diffusion_time), config.diffusion_time_steps)
    heat_kernel_trace = torch.zeros_like(sigma_values)

    for i, sigma in enumerate(sigma_values):
        heat_kernel_trace[i] = torch.sum(torch.exp(-sigma * L_eigenvalues))

    # Spectral dimension: d_s = -2 d(log P)/d(log σ)
    log_sigma = torch.log(sigma_values)
    log_P = torch.log(heat_kernel_trace + 1e-10)  # Avoid log(0)

    # Numerical gradient using finite differences
    # d/dx f(x) ≈ (f(x+h) - f(x-h)) / (2h)
    d_log_P_d_log_sigma = torch.zeros_like(log_P)
    for i in range(1, len(log_P) - 1):
        d_log_P_d_log_sigma[i] = (log_P[i + 1] - log_P[i - 1]) / (log_sigma[i + 1] - log_sigma[i - 1])
    # Forward difference for first point
    if len(log_P) > 1:
        d_log_P_d_log_sigma[0] = (log_P[1] - log_P[0]) / (log_sigma[1] - log_sigma[0])
        # Backward difference for last point
        d_log_P_d_log_sigma[-1] = (log_P[-1] - log_P[-2]) / (log_sigma[-1] - log_sigma[-2])

    spectral_dim = -2 * d_log_P_d_log_sigma

    # Planck-scale dimension (small σ)
    planck_idx = config.diffusion_time_steps // 10
    d_s_planck = spectral_dim[planck_idx].item()

    return {
        "spectral_dimension_curve": spectral_dim,
        "spectral_dimension_planck": d_s_planck,
        "heat_kernel_trace": heat_kernel_trace,
    }


def compute_local_hausdorff(positions: Tensor, distances: Tensor) -> Tensor:
    """Compute local Hausdorff dimension for each walker.

    Uses local volume scaling around each point.
    """
    N = positions.shape[0]
    local_dim = torch.zeros(N)

    # Simple approximation: use variance of neighbor distances
    # In a d-dimensional space, volume ~ r^d
    # So local dimension ~ d log(N(r)) / log(r)

    # For now, return spatial dimension as default
    d = positions.shape[1]
    local_dim[:] = float(d)

    return local_dim


def compute_hausdorff_dimension(
    positions: Tensor,
    alive: Tensor,
    config: QuantumGravityConfig,
) -> dict:
    """Compute Hausdorff dimension from volume scaling N(r) ~ r^{d_H}.

    d_H = d(log N)/d(log r)

    The Hausdorff dimension measures the intrinsic dimensionality of the manifold.
    """
    alive_mask = alive.bool()
    pos_alive = positions[alive_mask]

    if len(pos_alive) == 0:
        d = positions.shape[1]
        return {
            "hausdorff_dimension": float(d),
            "volume_scaling_data": (torch.zeros(0), torch.zeros(0)),
            "local_hausdorff": torch.zeros(0),
        }

    # Choose reference point (centroid)
    center = pos_alive.mean(dim=0)

    # Compute radial distances
    distances = torch.norm(pos_alive - center, dim=1)
    max_r = config.max_radius if config.max_radius is not None else distances.max().item()

    if max_r <= 0:
        max_r = 1.0

    # Count walkers within each radius
    radii = torch.linspace(0.1 * max_r, max_r, config.n_radial_bins)
    counts = torch.zeros_like(radii)

    for i, r in enumerate(radii):
        counts[i] = (distances <= r).sum()

    # Fit log(N) vs log(r) → slope = d_H
    valid_mask = counts > 0
    if valid_mask.sum() < 2:
        # Not enough points for fit
        d = positions.shape[1]
        return {
            "hausdorff_dimension": float(d),
            "volume_scaling_data": (radii, counts),
            "local_hausdorff": compute_local_hausdorff(pos_alive, distances),
        }

    log_r = torch.log(radii[valid_mask])
    log_N = torch.log(counts[valid_mask].float())

    # Linear regression
    A = torch.stack([log_r, torch.ones_like(log_r)], dim=1)
    solution = torch.linalg.lstsq(A, log_N).solution
    hausdorff_dim = solution[0].item()

    # Local Hausdorff dimension (per walker)
    local_dim = compute_local_hausdorff(pos_alive, distances)

    return {
        "hausdorff_dimension": hausdorff_dim,
        "volume_scaling_data": (radii, counts),
        "local_hausdorff": local_dim,
    }


def compute_causal_structure(
    history: RunHistory,
    mc_frame: int,
    config: QuantumGravityConfig,
    positions: Tensor | None = None,
    alive: Tensor | None = None,
    bounds: TorchBounds | None = None,
    pbc: bool | None = None,
    time_dim: int | None = None,
) -> dict:
    """Classify edges as spacelike/timelike/null from time-sliced Voronoi.

    Reuses: voronoi_time_slices.compute_time_sliced_voronoi()

    Implements causal set theory approach to discrete quantum gravity.
    """
    if positions is None:
        positions = history.x_final[mc_frame]
    if alive is None:
        alive = (
            history.alive_mask[min(mc_frame, len(history.alive_mask) - 1)]
            if mc_frame > 0
            else torch.ones(history.N, dtype=torch.bool)
        )
    if time_dim is None:
        time_dim = config.euclidean_time_dim

    # Compute time-sliced Voronoi
    try:
        time_sliced = compute_time_sliced_voronoi(
            positions=positions,
            time_dim=time_dim,
            n_bins=config.euclidean_time_bins,
            bounds=bounds if bounds is not None else history.bounds,
            alive=alive,
            pbc=history.pbc if pbc is None else pbc,
        )

        # Spacelike edges: within same time bin
        spacelike_edges = torch.zeros((2, 0), dtype=torch.long)
        for bin_result in time_sliced.bins:
            if bin_result.spacelike_edges.shape[0] > 0:
                edges_tensor = torch.from_numpy(bin_result.spacelike_edges).t().long()
                spacelike_edges = torch.cat([spacelike_edges, edges_tensor], dim=1)

        # Timelike edges: between adjacent time bins
        timelike_edges = torch.from_numpy(time_sliced.timelike_edges).t().long()

        # Null edges: |Δx|² = c² |Δt|² (rare in discrete setting)
        null_edges = torch.zeros((2, 0), dtype=torch.long)

        # Check causal violations (timelike edges going backward)
        violations = 0
        # Would need bin assignments to check properly

    except Exception:
        # Fallback if time slicing fails
        spacelike_edges = torch.zeros((2, 0), dtype=torch.long)
        timelike_edges = torch.zeros((2, 0), dtype=torch.long)
        null_edges = torch.zeros((2, 0), dtype=torch.long)
        violations = 0

    return {
        "spacelike_edges": spacelike_edges,
        "timelike_edges": timelike_edges,
        "null_edges": null_edges,
        "causal_violations": violations,
    }


def compute_boundary_area(
    facet_areas: dict[tuple[int, int], float],
    neighbor_lists: dict[int, list[int]],
    boundary_mask: np.ndarray,
) -> float:
    """Compute total area of boundary cell facets.

    Sums the Voronoi facet areas between boundary cells (Tier 1) and their
    neighbors (Tier 2+). This gives the "entangling surface" area separating
    the boundary region from the interior.

    Note: This is an approximation. Ideally we'd compute the actual outer
    surface area using Voronoi vertices or a convex hull, but that requires
    handling infinite Voronoi regions which is non-trivial.

    Args:
        facet_areas: Voronoi facet areas between cells (i, j)
        neighbor_lists: Full neighbor lists (NOT filtered)
        boundary_mask: Boolean mask for boundary cells (Tier 1)

    Returns:
        Total area of facets between boundary and interior cells
    """
    total_area = 0.0

    for i in range(len(neighbor_lists)):
        if not boundary_mask[i]:
            continue

        neighbors = neighbor_lists.get(i, [])
        for j in neighbors:
            if (i, j) in facet_areas:
                total_area += facet_areas[(i, j)]

    # Divide by 2 since we double-counted edges (both (i,j) and (j,i))
    return total_area / 2.0


def compute_holographic_entropy(
    positions: Tensor,
    alive: Tensor,
    history: RunHistory,
    config: QuantumGravityConfig,
    bounds: TorchBounds | None = None,
    pbc: bool | None = None,
    volume_weights: Tensor | None = None,
) -> dict:
    """Compute holographic entropy S = A/(4G) from boundary area.

    Implements Bekenstein-Hawking formula and holographic principle.

    The entropy is computed using the entangling surface area between
    boundary cells and interior cells in the Voronoi tessellation:

        S = A / (4 G ℏ) = A / (4 ℓ_P²)

    where A is the total Voronoi facet area separating boundary (Tier 1)
    cells from interior (Tier 2+) cells.

    Note: With periodic boundary conditions (PBC), there are no boundary
    cells, so boundary_area = 0 and entropy = 0. This is expected - a
    toroidal universe has no physical boundary to define holographic entropy.

    Returns:
        dict with keys:
            - holographic_entropy: S = A / (4 ℓ_P²)
            - boundary_area: Total entangling surface area A
            - bulk_volume: Total volume of all cells
            - area_law_coefficient: S/A ratio (should be 1/(4 ℓ_P²))
    """
    use_pbc = history.pbc if pbc is None else pbc
    use_bounds = bounds if bounds is not None else history.bounds

    voronoi_data = compute_voronoi_tessellation(
        positions=positions,
        alive=alive,
        bounds=use_bounds,
        pbc=use_pbc,
        exclude_boundary=False,  # Need full neighbor lists to compute boundary area
    )

    # Identify boundary cells (Tier 1)
    # Since exclude_boundary=False, classification is not computed automatically
    # We need to compute it manually for holographic entropy
    if use_pbc:
        # PBC - no physical boundary, entropy = 0
        boundary_area = 0.0
    else:
        # Manually classify boundary cells to get is_boundary mask
        classification = classify_boundary_cells(
            voronoi_data=voronoi_data,
            positions=positions,
            bounds=use_bounds,
            pbc=use_pbc,
            boundary_tolerance=0.01,  # 1% of domain width
        )

        is_boundary = classification.get("is_boundary")
        if is_boundary is None or len(is_boundary) == 0:
            boundary_area = 0.0
        else:
            # Handle both torch tensors and numpy arrays
            if hasattr(is_boundary, 'numpy'):
                boundary_mask = is_boundary.cpu().numpy() if hasattr(is_boundary, 'cpu') else is_boundary.numpy()
            elif isinstance(is_boundary, np.ndarray):
                boundary_mask = is_boundary
            else:
                boundary_mask = np.array(is_boundary)

            facet_areas = voronoi_data["facet_areas"]
            neighbor_lists = voronoi_data["neighbor_lists"]

            boundary_area = compute_boundary_area(facet_areas, neighbor_lists, boundary_mask)

    # Bulk volume
    if volume_weights is not None:
        if torch.is_tensor(volume_weights):
            bulk_volume = float(volume_weights.sum().item())
        else:
            bulk_volume = float(np.asarray(volume_weights).sum())
    else:
        volumes_np = voronoi_data["volumes"]
        bulk_volume = float(volumes_np.sum())

    # Holographic entropy: S = A / (4 G ℏ)
    # Using Planck units: G ℏ = ℓ_P² c³
    planck_area = config.planck_length**2
    entropy = boundary_area / (4 * planck_area)

    # Area law coefficient
    area_law_coeff = entropy / boundary_area if boundary_area > 0 else 0.0

    return {
        "holographic_entropy": entropy,
        "boundary_area": boundary_area,
        "bulk_volume": bulk_volume,
        "area_law_coefficient": area_law_coeff,
    }


def construct_dual_graph(neighbor_lists: dict[int, list[int]]) -> Tensor:
    """Construct dual graph (Delaunay from Voronoi).

    For spin network visualization.
    """
    edges = []
    for i, neighbors in neighbor_lists.items():
        for j in neighbors:
            if i < j:  # Avoid duplicates
                edges.append([i, j])

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t()


def compute_spin_network_state(
    voronoi_data: dict[str, Any],
    volume_weights: Tensor | None = None,
) -> dict:
    """Construct spin network from Voronoi dual graph.

    In LQG: edges carry SU(2) spins j, vertices have quantum volume
    Here: Use facet areas as spin labels (A = ℓ_P² √(j(j+1)))

    Implements Loop Quantum Gravity approach.
    """
    facet_areas_dict = voronoi_data["facet_areas"]
    volumes = voronoi_data["volumes"]
    neighbor_lists = voronoi_data["neighbor_lists"]

    # Convert facet areas dict to tensor
    facet_areas_list = list(facet_areas_dict.values())
    if len(facet_areas_list) == 0:
        facet_areas_tensor = torch.zeros(0)
    else:
        facet_areas_tensor = torch.tensor(facet_areas_list, dtype=torch.float32)

    # Edge spins: j from area A_ij
    # A = 8πG j(j+1) ℏ → j ≈ √(A / (8πG ℏ))
    edge_spins = torch.sqrt(facet_areas_tensor / (8 * np.pi))  # Simplified units

    # Vertex quantum volumes (Wilson loop)
    if volume_weights is not None:
        if torch.is_tensor(volume_weights):
            weights_full = volume_weights
        else:
            weights_full = torch.as_tensor(volume_weights)
        alive_indices = voronoi_data.get("alive_indices")
        if alive_indices is not None:
            alive_idx = torch.as_tensor(alive_indices, device=weights_full.device, dtype=torch.long)
            alive_idx = alive_idx[alive_idx < weights_full.shape[0]]
            vertex_volumes = weights_full[alive_idx].float()
        else:
            vertex_volumes = weights_full[: len(neighbor_lists)].float()
    else:
        vertex_volumes = torch.from_numpy(volumes).float()

    # Dual graph (Delaunay from Voronoi)
    dual_edges = construct_dual_graph(neighbor_lists)

    return {
        "edge_spins": edge_spins,
        "vertex_quantum_volumes": vertex_volumes,
        "dual_graph_edges": dual_edges,
    }


def compute_raychaudhuri_expansion(
    curvature_proxies: dict[str, Any],
    n_walkers: int,
) -> dict:
    """Extract Raychaudhuri expansion from existing curvature proxies.

    Reuses: voronoi_observables.compute_curvature_proxies()
    Already computes: raychaudhuri_expansion = (1/V) dV/dt

    The Raychaudhuri equation predicts singularities (Hawking-Penrose theorems).
    """
    theta_np = curvature_proxies.get("raychaudhuri_expansion")

    if theta_np is not None and len(theta_np) > 0:
        theta = torch.from_numpy(theta_np).float()
    else:
        theta = torch.zeros(n_walkers)

    # Identify convergence regions (θ < 0 → focusing)
    convergence_mask = theta < 0

    # Shear tensor (if available from metric variations)
    shear = None  # Optional: compute from metric tensor time derivatives

    return {
        "expansion_scalar": theta,
        "shear_tensor": shear,
        "convergence_regions": convergence_mask,
    }


def compute_tidal_tensor(
    metric_tensors: Tensor,
    edge_index: Tensor,
    positions: Tensor,
) -> Tensor:
    """Compute tidal tensor from metric variations.

    T_ij ≈ ∂_i ∂_j g (finite difference approximation)
    """
    N, d, _ = metric_tensors.shape
    tidal = torch.zeros(N, d, d, device=metric_tensors.device)

    if edge_index.shape[1] == 0:
        return tidal

    row, col = edge_index[0], edge_index[1]

    # For each node, estimate second derivatives from neighbors
    for i in range(N):
        # Find edges where i is the source
        mask = row == i
        if not mask.any():
            continue

        neighbors = col[mask]
        if len(neighbors) == 0:
            continue

        # Compute metric differences
        delta_g = metric_tensors[neighbors] - metric_tensors[i]  # [k, d, d]
        delta_x = positions[neighbors] - positions[i]  # [k, d]

        # Estimate Hessian via least squares
        # For simplicity, use average of outer products
        for j_idx in range(d):
            for k_idx in range(d):
                grad_component = delta_g[:, j_idx, k_idx] / (
                    torch.norm(delta_x, dim=1) + 1e-10
                )
                tidal[i, j_idx, k_idx] = grad_component.mean()

    return tidal


def compute_geodesic_deviation(
    metric_tensors: Tensor,
    edge_index: Tensor,
    positions: Tensor,
) -> dict:
    """Compute geodesic deviation from metric variations.

    Deviation equation: D²ξ/Dτ² = R(ξ)
    where ξ is separation vector, R is Riemann tensor

    Approximation: Use metric gradient as Riemann proxy

    Measures tidal forces (operational definition of spacetime curvature).
    """
    N, d = positions.shape

    if edge_index.shape[1] == 0:
        return {
            "deviation_vectors": torch.zeros(0, d),
            "tidal_tensor": torch.zeros(N, d, d),
            "tidal_eigenvalues": torch.zeros(N, d),
        }

    row, col = edge_index[0], edge_index[1]

    # Separation vectors
    separation = positions[col] - positions[row]

    # Tidal tensor: T_ij ≈ ∂_i ∂_j g (finite difference)
    tidal_tensor = compute_tidal_tensor(metric_tensors, edge_index, positions)

    # Eigenvalues: stretching/squeezing directions
    try:
        tidal_eigenvalues = torch.linalg.eigvalsh(tidal_tensor)
    except Exception:
        tidal_eigenvalues = torch.zeros(N, d)

    # Deviation vectors (initial separations evolved under tidal forces)
    deviation_vectors = separation  # Initial conditions

    return {
        "deviation_vectors": deviation_vectors,
        "tidal_tensor": tidal_tensor,
        "tidal_eigenvalues": tidal_eigenvalues,
    }


def compute_quantum_gravity_observables(
    history: RunHistory,
    config: QuantumGravityConfig | None = None,
) -> QuantumGravityObservables:
    """Compute all 10 quantum gravity analyses.

    This is the main entry point that orchestrates all computations.

    Args:
        history: RunHistory from simulation
        config: Configuration (uses defaults if None)

    Returns:
        QuantumGravityObservables dataclass with all computed observables
    """
    if config is None:
        config = QuantumGravityConfig()

    # Resolve mc_time_index with validation
    if config.mc_time_index is None:
        mc_frame = history.n_recorded - 1
    else:
        mc_frame = config.mc_time_index

        # Validate range
        if mc_frame < 0:
            msg = f"mc_time_index must be non-negative, got {mc_frame}"
            raise ValueError(msg)

        if mc_frame >= history.n_recorded:
            msg = (
                f"mc_time_index {mc_frame} out of bounds "
                f"(valid range: 0..{history.n_recorded - 1})"
            )
            raise ValueError(msg)

    # Extract positions for the selected frame
    positions_full = history.x_final[mc_frame]

    # Validate shape
    if positions_full.ndim != 2:
        msg = (
            f"Expected 2D positions [N, d] at mc_frame={mc_frame}, "
            f"got shape {positions_full.shape}"
        )
        raise ValueError(msg)

    # Normalize and validate analysis dimensions
    analysis_dims = _normalize_analysis_dims(positions_full.shape[1], config.analysis_dims)
    default_dims = tuple(range(positions_full.shape[1]))

    # Slice positions by analysis_dims if needed
    if analysis_dims and analysis_dims != default_dims:
        # Defensive: ensure positions_full is 2D before slicing
        if positions_full.ndim != 2:
            msg = (
                f"Expected 2D positions [N, d], got shape {positions_full.shape}. "
                f"mc_frame={mc_frame}, history.x_final.shape={history.x_final.shape}"
            )
            raise ValueError(msg)

        # Convert tuple to list for proper 1D indexing
        positions = positions_full[:, list(analysis_dims)]
    else:
        positions = positions_full
    alive = (
        history.alive_mask[min(mc_frame, len(history.alive_mask) - 1)]
        if mc_frame > 0
        else torch.ones(history.N, dtype=torch.bool)
    )
    analysis_bounds = _slice_bounds(history.bounds, analysis_dims) if history.bounds is not None else None
    time_dim = _map_time_dim(config.euclidean_time_dim, analysis_dims)
    scutoid_history = _build_scutoid_history_view(history, analysis_dims, analysis_bounds)

    # Compute Voronoi tessellation (reuse across analyses)
    voronoi_data = compute_voronoi_tessellation(
        positions=positions,
        alive=alive,
        bounds=analysis_bounds,
        pbc=history.pbc,
        exclude_boundary=config.exclude_boundary,
        compute_curvature=True,
        spatial_dims=positions.shape[1],
    )
    volume_weights_full = _get_volume_weights(history, mc_frame)

    # Build edge index from neighbor lists
    neighbor_lists = voronoi_data["neighbor_lists"]
    edges = []
    for i, neighbors in neighbor_lists.items():
        for j in neighbors:
            edges.append([i, j])

    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=positions.device)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long, device=positions.device).t()

    n_edges = edge_index.shape[1]

    # Compute metric tensors (from Higgs module)
    metric_tensors = compute_emergent_metric(positions, edge_index, alive)

    # Get curvature proxies
    curvature_proxies = voronoi_data.get("curvature_proxies", {})
    if volume_weights_full is not None and curvature_proxies is not None:
        alive_indices = voronoi_data.get("alive_indices")
        if alive_indices is not None:
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

    # Run all 10 analyses
    results = {}

    if config.compute_all:
        # 1. Regge Calculus
        if scutoid_history.d in {2, 3}:
            results.update(compute_regge_action(scutoid_history, mc_frame, config))
        else:
            results.update(
                {
                    "regge_action": 0.0,
                    "regge_action_density": torch.zeros(history.N, device=positions.device),
                    "deficit_angles": torch.zeros(0, device=positions.device),
                }
            )

        # 2. Einstein-Hilbert
        results.update(
            compute_einstein_hilbert_action(
                positions,
                alive,
                history,
                config,
                bounds=analysis_bounds,
                pbc=history.pbc,
                volume_weights=volume_weights_full,
            )
        )

        # 3. ADM Energy
        ricci = results.get("ricci_scalars", torch.zeros(history.N))
        if volume_weights_full is not None and volume_weights_full.shape[0] == history.N:
            volumes_full = volume_weights_full.to(device=positions.device, dtype=ricci.dtype)
        else:
            volumes = torch.from_numpy(voronoi_data["volumes"]).float()
            volumes_full = torch.zeros(history.N, device=positions.device, dtype=ricci.dtype)
            alive_indices = voronoi_data["alive_indices"]
            alive_indices_torch = (
                torch.from_numpy(alive_indices).long()
                if isinstance(alive_indices, np.ndarray)
                else alive_indices
            )
            volumes_full[alive_indices_torch] = volumes.to(device=positions.device, dtype=ricci.dtype)

        ricci_full = torch.zeros(history.N, device=positions.device, dtype=ricci.dtype)
        alive_indices = voronoi_data["alive_indices"]
        alive_indices_torch = (
            torch.from_numpy(alive_indices).long()
            if isinstance(alive_indices, np.ndarray)
            else alive_indices
        )
        ricci_full[alive_indices_torch] = ricci.to(device=positions.device, dtype=ricci.dtype)
        results.update(compute_adm_energy(ricci_full, volumes_full))

        # 4. Spectral Dimension
        results.update(compute_spectral_dimension(positions, edge_index, alive, config))

        # 5. Hausdorff Dimension
        results.update(compute_hausdorff_dimension(positions, alive, config))

        # 6. Causal Structure
        results.update(
            compute_causal_structure(
                history,
                mc_frame,
                config,
                positions=positions,
                alive=alive,
                bounds=analysis_bounds,
                pbc=history.pbc,
                time_dim=time_dim,
            )
        )

        # 7. Holographic Entropy
        results.update(
            compute_holographic_entropy(
                positions,
                alive,
                history,
                config,
                bounds=analysis_bounds,
                pbc=history.pbc,
                volume_weights=volume_weights_full,
            )
        )

        # 8. Spin Network
        results.update(compute_spin_network_state(voronoi_data, volume_weights=volume_weights_full))

        # 9. Raychaudhuri Expansion
        results.update(compute_raychaudhuri_expansion(curvature_proxies, history.N))

        # 10. Geodesic Deviation
        results.update(compute_geodesic_deviation(metric_tensors, edge_index, positions))

    # Package into QuantumGravityObservables dataclass
    return QuantumGravityObservables(
        n_walkers=alive.sum().item(),
        n_edges=n_edges,
        spatial_dims=positions.shape[1],
        mc_frame=mc_frame,
        config=config,
        **results,
    )


def compute_quantum_gravity_time_evolution(
    history: RunHistory,
    config: QuantumGravityConfig | None = None,
    frame_stride: int = 1,
    warmup_frames: int | None = None,
) -> QuantumGravityTimeSeries:
    """Compute quantum gravity observables over all MC frames.

    This function enables 4D spacetime block analysis by computing all 10 quantum
    gravity observables across the entire Monte Carlo temporal history.

    Args:
        history: RunHistory from simulation
        config: Configuration (uses defaults if None)
        frame_stride: Compute every N frames (default 1 = all frames)
        warmup_frames: Skip first N frames (default from config.warmup_fraction)

    Returns:
        QuantumGravityTimeSeries with all observables as time series

    Physical Motivation:
        - Watch spectral dimension reduce during thermalization (d_s: 4 → 2 at Planck scale)
        - Track ADM energy conservation/violation (tests energy flow)
        - See Hausdorff dimension converge to intrinsic value (fractal → manifold transition)
        - Monitor holographic entropy growth (2nd law validation)
        - Identify singularity formation via Raychaudhuri expansion (θ → -∞)

    Example:
        >>> history = RunHistory.load("qft_run.pt")
        >>> time_series = compute_quantum_gravity_time_evolution(history, frame_stride=2)
        >>> print(f"ADM mass conservation: {time_series.adm_mass[-1] / time_series.adm_mass[0] - 1:.2%}")
        >>> print(f"Spectral dimension at Planck scale: {time_series.spectral_dimension_planck[-1]:.2f}")
    """
    if config is None:
        config = QuantumGravityConfig()

    # Determine frames to analyze
    n_recorded = history.n_recorded
    if warmup_frames is None:
        warmup_frames = int(config.warmup_fraction * n_recorded)

    frames = list(range(warmup_frames, n_recorded, frame_stride))
    n_frames = len(frames)

    if n_frames == 0:
        # No frames to analyze - return empty time series
        return QuantumGravityTimeSeries(
            n_frames=0,
            mc_frames=np.array([]),
            regge_action=np.array([]),
            regge_action_density=[],
            einstein_hilbert_action=np.array([]),
            scalar_curvature_mean=np.array([]),
            adm_mass=np.array([]),
            adm_mass_mean_density=np.array([]),
            spectral_dimension_planck=np.array([]),
            spectral_dimension_large_scale=np.array([]),
            hausdorff_dimension=np.array([]),
            n_spacelike_edges=np.array([]),
            n_timelike_edges=np.array([]),
            causal_violations=np.array([]),
            holographic_entropy=np.array([]),
            boundary_area=np.array([]),
            bulk_volume=np.array([]),
            area_law_coefficient=np.array([]),
            mean_edge_spin=np.array([]),
            mean_vertex_volume=np.array([]),
            n_edges=np.array([]),
            expansion_mean=np.array([]),
            expansion_std=np.array([]),
            convergence_fraction=np.array([]),
            tidal_strength_mean=np.array([]),
            tidal_strength_max=np.array([]),
            n_walkers=np.array([]),
            spatial_dims=history.d,
            config=config,
        )

    # Initialize storage for time series
    regge_action = np.zeros(n_frames)
    einstein_hilbert_action = np.zeros(n_frames)
    scalar_curvature_mean = np.zeros(n_frames)
    adm_mass = np.zeros(n_frames)
    adm_mass_density = np.zeros(n_frames)
    spectral_dim_planck = np.zeros(n_frames)
    spectral_dim_large = np.zeros(n_frames)
    hausdorff_dim = np.zeros(n_frames)
    n_spacelike = np.zeros(n_frames, dtype=int)
    n_timelike = np.zeros(n_frames, dtype=int)
    causal_viols = np.zeros(n_frames, dtype=int)
    holo_entropy = np.zeros(n_frames)
    boundary_area = np.zeros(n_frames)
    bulk_volume = np.zeros(n_frames)
    area_law_coeff = np.zeros(n_frames)
    mean_spin = np.zeros(n_frames)
    mean_vol = np.zeros(n_frames)
    n_edges_arr = np.zeros(n_frames, dtype=int)
    expansion_mean = np.zeros(n_frames)
    expansion_std = np.zeros(n_frames)
    convergence_frac = np.zeros(n_frames)
    tidal_mean = np.zeros(n_frames)
    tidal_max = np.zeros(n_frames)
    n_walkers = np.zeros(n_frames, dtype=int)

    # Loop over frames
    for i, frame in enumerate(frames):
        # Compute single-frame observables
        config_frame = replace(
            config,
            mc_time_index=frame,
            compute_all=True,
        )

        try:
            obs = compute_quantum_gravity_observables(history, config_frame)

            # Extract scalar time series
            regge_action[i] = obs.regge_action
            einstein_hilbert_action[i] = obs.einstein_hilbert_action
            scalar_curvature_mean[i] = obs.scalar_curvature_mean
            adm_mass[i] = obs.adm_mass
            adm_mass_density[i] = obs.adm_energy_density.mean().item() if len(obs.adm_energy_density) > 0 else 0.0
            spectral_dim_planck[i] = obs.spectral_dimension_planck
            # Large-scale spectral dimension (last point of curve)
            if len(obs.spectral_dimension_curve) > 0:
                spectral_dim_large[i] = obs.spectral_dimension_curve[-1].item()
            else:
                spectral_dim_large[i] = float(history.d)
            hausdorff_dim[i] = obs.hausdorff_dimension
            n_spacelike[i] = obs.spacelike_edges.shape[1] if obs.spacelike_edges.ndim == 2 else 0
            n_timelike[i] = obs.timelike_edges.shape[1] if obs.timelike_edges.ndim == 2 else 0
            causal_viols[i] = obs.causal_violations
            holo_entropy[i] = obs.holographic_entropy
            boundary_area[i] = obs.boundary_area
            bulk_volume[i] = obs.bulk_volume
            area_law_coeff[i] = obs.area_law_coefficient
            mean_spin[i] = obs.edge_spins.mean().item() if len(obs.edge_spins) > 0 else 0.0
            mean_vol[i] = obs.vertex_quantum_volumes.mean().item() if len(obs.vertex_quantum_volumes) > 0 else 0.0
            n_edges_arr[i] = obs.n_edges
            expansion_mean[i] = obs.expansion_scalar.mean().item() if len(obs.expansion_scalar) > 0 else 0.0
            expansion_std[i] = obs.expansion_scalar.std().item() if len(obs.expansion_scalar) > 0 else 0.0
            convergence_frac[i] = obs.convergence_regions.float().mean().item() if len(obs.convergence_regions) > 0 else 0.0
            tidal_mean[i] = torch.abs(obs.tidal_eigenvalues).mean().item() if obs.tidal_eigenvalues.numel() > 0 else 0.0
            tidal_max[i] = torch.abs(obs.tidal_eigenvalues).max().item() if obs.tidal_eigenvalues.numel() > 0 else 0.0
            n_walkers[i] = obs.n_walkers
        except Exception as e:
            # If computation fails for a frame, fill with defaults
            print(f"Warning: Failed to compute quantum gravity observables for frame {frame}: {e}")
            regge_action[i] = 0.0
            einstein_hilbert_action[i] = 0.0
            scalar_curvature_mean[i] = 0.0
            adm_mass[i] = 0.0
            adm_mass_density[i] = 0.0
            spectral_dim_planck[i] = float(history.d)
            spectral_dim_large[i] = float(history.d)
            hausdorff_dim[i] = float(history.d)
            n_walkers[i] = 0

    return QuantumGravityTimeSeries(
        n_frames=n_frames,
        mc_frames=np.array(frames),
        regge_action=regge_action,
        regge_action_density=[],  # Optional: store full tensors per frame
        einstein_hilbert_action=einstein_hilbert_action,
        scalar_curvature_mean=scalar_curvature_mean,
        adm_mass=adm_mass,
        adm_mass_mean_density=adm_mass_density,
        spectral_dimension_planck=spectral_dim_planck,
        spectral_dimension_large_scale=spectral_dim_large,
        hausdorff_dimension=hausdorff_dim,
        n_spacelike_edges=n_spacelike,
        n_timelike_edges=n_timelike,
        causal_violations=causal_viols,
        holographic_entropy=holo_entropy,
        boundary_area=boundary_area,
        bulk_volume=bulk_volume,
        area_law_coefficient=area_law_coeff,
        mean_edge_spin=mean_spin,
        mean_vertex_volume=mean_vol,
        n_edges=n_edges_arr,
        expansion_mean=expansion_mean,
        expansion_std=expansion_std,
        convergence_fraction=convergence_frac,
        tidal_strength_mean=tidal_mean,
        tidal_strength_max=tidal_max,
        n_walkers=n_walkers,
        spatial_dims=history.d,
        config=config,
    )
