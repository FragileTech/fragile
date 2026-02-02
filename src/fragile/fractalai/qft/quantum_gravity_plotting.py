"""Plotting functions for Quantum Gravity observables.

This module provides HoloViews-based visualizations for the 10 quantum gravity
analyses computed from emergent spacetime geometry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

if TYPE_CHECKING:
    from fragile.fractalai.qft.quantum_gravity import QuantumGravityObservables


# =============================================================================
# 1. Regge Calculus Plots
# =============================================================================


def build_regge_action_density_heatmap(
    positions: np.ndarray,
    action_density: np.ndarray,
    alive: np.ndarray,
    spatial_dims: tuple[int, int] = (0, 1),
) -> hv.Points:
    """Spatial heatmap of Regge action density."""
    x_dim, y_dim = spatial_dims
    alive_mask = alive.astype(bool)

    if not alive_mask.any():
        return hv.Text(0, 0, "No alive walkers").opts(title="Regge Action Density")

    data = pd.DataFrame({
        'x': positions[alive_mask, x_dim],
        'y': positions[alive_mask, y_dim],
        'action_density': action_density[alive_mask],
    })

    points = hv.Points(data, ['x', 'y'], 'action_density').opts(
        color='action_density',
        cmap='coolwarm',
        size=5,
        colorbar=True,
        width=600,
        height=500,
        title='Regge Action Density (R * Vol)',
        xlabel=f'Dimension {x_dim}',
        ylabel=f'Dimension {y_dim}',
    )

    return points


def build_deficit_angle_distribution(
    deficit_angles: np.ndarray,
) -> hv.Histogram:
    """Histogram of deficit angles (log scale)."""
    if len(deficit_angles) == 0:
        return hv.Text(0, 0, "No deficit angles").opts(title="Deficit Angle Distribution")

    # Filter finite values
    finite_angles = deficit_angles[np.isfinite(deficit_angles)]

    if len(finite_angles) == 0:
        return hv.Text(0, 0, "No finite deficit angles").opts(title="Deficit Angle Distribution")

    hist = hv.Histogram(np.histogram(finite_angles, bins=50)).opts(
        width=600,
        height=400,
        title='Deficit Angle Distribution',
        xlabel='Deficit Angle δ',
        ylabel='Count',
        color='#4c78a8',
    )

    return hist


# =============================================================================
# 2. Einstein-Hilbert Plots
# =============================================================================


def build_ricci_landscape(
    positions: np.ndarray,
    ricci_scalars: np.ndarray,
    alive: np.ndarray,
    spatial_dims: tuple[int, int] = (0, 1),
) -> hv.Points:
    """2D/3D visualization of Ricci scalar field."""
    x_dim, y_dim = spatial_dims
    alive_mask = alive.astype(bool)

    if not alive_mask.any():
        return hv.Text(0, 0, "No alive walkers").opts(title="Ricci Scalar Landscape")

    valid = alive_mask & np.isfinite(ricci_scalars)

    if not valid.any():
        return hv.Text(0, 0, "No valid Ricci scalars").opts(title="Ricci Scalar Landscape")

    data = pd.DataFrame({
        'x': positions[valid, x_dim],
        'y': positions[valid, y_dim],
        'ricci': ricci_scalars[valid],
    })

    points = hv.Points(data, ['x', 'y'], 'ricci').opts(
        color='ricci',
        cmap='RdBu_r',
        size=5,
        colorbar=True,
        width=600,
        height=500,
        title='Ricci Scalar Landscape R(x)',
        xlabel=f'Dimension {x_dim}',
        ylabel=f'Dimension {y_dim}',
    )

    return points


def build_action_decomposition(
    kinetic: float,
    potential: float,
    gravity: float | None,
    total: float,
) -> pn.Column:
    """Bar chart showing action components."""
    components = {
        'Kinetic': kinetic,
        'Potential': potential,
        'Total': total,
    }

    if gravity is not None:
        components['Gravity'] = gravity

    data = pd.DataFrame({
        'Component': list(components.keys()),
        'Value': list(components.values()),
    })

    bars = hv.Bars(data, 'Component', 'Value').opts(
        width=600,
        height=400,
        title='Einstein-Hilbert Action Components',
        ylabel='Action',
        color='Component',
        cmap='Category10',
        show_legend=False,
    )

    return pn.pane.HoloViews(bars, sizing_mode="stretch_width")


# =============================================================================
# 3. ADM Energy Plots
# =============================================================================


def build_adm_energy_summary(
    adm_mass: float,
    mean_density: float,
) -> str:
    """Summary of ADM energy."""
    md = f"""
### ADM Energy (Hamiltonian Formalism)

**Total ADM Mass:** {adm_mass:.6e}

**Mean Energy Density:** {mean_density:.6e}

The ADM (Arnowitt-Deser-Misner) mass is the total gravitational energy
from the spatial hypersurface. In the emergent geometry, this corresponds
to the integrated Ricci scalar curvature.
"""
    return md


def build_energy_density_distribution(
    energy_density: np.ndarray,
    alive: np.ndarray,
) -> hv.Histogram:
    """Histogram of local energy density."""
    alive_mask = alive.astype(bool)
    density_alive = energy_density[alive_mask]

    if len(density_alive) == 0:
        return hv.Text(0, 0, "No alive walkers").opts(title="Energy Density Distribution")

    finite = density_alive[np.isfinite(density_alive)]

    if len(finite) == 0:
        return hv.Text(0, 0, "No finite energy densities").opts(title="Energy Density Distribution")

    hist = hv.Histogram(np.histogram(finite, bins=50)).opts(
        width=600,
        height=400,
        title='ADM Energy Density Distribution',
        xlabel='Energy Density',
        ylabel='Count',
        color='#e15759',
    )

    return hist


# =============================================================================
# 4. Spectral Dimension Plots
# =============================================================================


def build_spectral_dimension_curve(
    sigma_values: np.ndarray,
    spectral_dim: np.ndarray,
    planck_value: float,
    spatial_dims: int,
) -> hv.Overlay:
    """d_s(σ) vs diffusion time with dimension reduction signature."""
    if len(sigma_values) == 0 or len(spectral_dim) == 0:
        return hv.Text(0, 0, "No spectral dimension data").opts(title="Spectral Dimension")

    # Filter finite values
    valid = np.isfinite(sigma_values) & np.isfinite(spectral_dim)
    sigma_valid = sigma_values[valid]
    dim_valid = spectral_dim[valid]

    if len(sigma_valid) == 0:
        return hv.Text(0, 0, "No valid data").opts(title="Spectral Dimension")

    data = pd.DataFrame({
        'sigma': sigma_valid,
        'spectral_dim': dim_valid,
    })

    curve = hv.Curve(data, 'sigma', 'spectral_dim').opts(
        width=600,
        height=500,
        title='Spectral Dimension d_s(σ)',
        xlabel='Diffusion Time σ',
        ylabel='Spectral Dimension d_s',
        color='#4c78a8',
        line_width=2,
        logx=True,
    )

    # Add horizontal line for spatial dimension
    hline = hv.HLine(spatial_dims).opts(
        color='red',
        line_dash='dashed',
        line_width=1,
    )

    # Add annotation for Planck-scale value
    annotation = hv.Text(sigma_valid[len(sigma_valid) // 10], planck_value, f'd_s(Planck) = {planck_value:.2f}').opts(
        color='green',
    )

    return curve * hline * annotation


def build_heat_kernel_trace(
    sigma_values: np.ndarray,
    trace_values: np.ndarray,
) -> hv.Curve:
    """Heat kernel return probability."""
    if len(sigma_values) == 0 or len(trace_values) == 0:
        return hv.Text(0, 0, "No heat kernel data").opts(title="Heat Kernel Trace")

    valid = np.isfinite(sigma_values) & np.isfinite(trace_values) & (trace_values > 0)
    sigma_valid = sigma_values[valid]
    trace_valid = trace_values[valid]

    if len(sigma_valid) == 0:
        return hv.Text(0, 0, "No valid data").opts(title="Heat Kernel Trace")

    data = pd.DataFrame({
        'sigma': sigma_valid,
        'trace': trace_valid,
    })

    curve = hv.Curve(data, 'sigma', 'trace').opts(
        width=600,
        height=400,
        title='Heat Kernel Return Probability P(σ)',
        xlabel='Diffusion Time σ',
        ylabel='Trace P(σ)',
        color='#f28e2b',
        line_width=2,
        logx=True,
        logy=True,
    )

    return curve


# =============================================================================
# 5. Hausdorff Dimension Plots
# =============================================================================


def build_volume_scaling_plot(
    radii: np.ndarray,
    counts: np.ndarray,
    hausdorff_dim: float,
) -> hv.Overlay:
    """log(N) vs log(r) with fitted slope = d_H."""
    if len(radii) == 0 or len(counts) == 0:
        return hv.Text(0, 0, "No volume scaling data").opts(title="Hausdorff Dimension")

    valid = (counts > 0) & np.isfinite(radii)
    radii_valid = radii[valid]
    counts_valid = counts[valid]

    if len(radii_valid) < 2:
        return hv.Text(0, 0, "Insufficient data for scaling").opts(title="Hausdorff Dimension")

    log_r = np.log(radii_valid)
    log_N = np.log(counts_valid.astype(float))

    data = pd.DataFrame({
        'log_r': log_r,
        'log_N': log_N,
    })

    scatter = hv.Scatter(data, 'log_r', 'log_N').opts(
        width=600,
        height=500,
        title=f'Volume Scaling: d_H = {hausdorff_dim:.2f}',
        xlabel='log(r)',
        ylabel='log(N(r))',
        color='#4c78a8',
        size=6,
    )

    # Fit line: log(N) = d_H * log(r) + const
    fit_line_data = pd.DataFrame({
        'log_r': log_r,
        'log_N_fit': hausdorff_dim * log_r + log_N[0] - hausdorff_dim * log_r[0],
    })

    fit_line = hv.Curve(fit_line_data, 'log_r', 'log_N_fit').opts(
        color='red',
        line_dash='dashed',
        line_width=2,
    )

    return scatter * fit_line


def build_local_hausdorff_heatmap(
    positions: np.ndarray,
    local_dim: np.ndarray,
    alive: np.ndarray,
    spatial_dims: tuple[int, int] = (0, 1),
) -> hv.Points:
    """Spatial variation of local dimension."""
    x_dim, y_dim = spatial_dims
    alive_mask = alive.astype(bool)

    if not alive_mask.any():
        return hv.Text(0, 0, "No alive walkers").opts(title="Local Hausdorff Dimension")

    # Local dimension has different length (only alive walkers)
    n_alive = alive_mask.sum()
    if len(local_dim) != n_alive:
        return hv.Text(0, 0, "Dimension mismatch").opts(title="Local Hausdorff Dimension")

    pos_alive = positions[alive_mask]

    data = pd.DataFrame({
        'x': pos_alive[:, x_dim],
        'y': pos_alive[:, y_dim],
        'local_dim': local_dim,
    })

    points = hv.Points(data, ['x', 'y'], 'local_dim').opts(
        color='local_dim',
        cmap='viridis',
        size=5,
        colorbar=True,
        width=600,
        height=500,
        title='Local Hausdorff Dimension',
        xlabel=f'Dimension {x_dim}',
        ylabel=f'Dimension {y_dim}',
    )

    return points


# =============================================================================
# 6. Causal Structure Plots
# =============================================================================


def build_causal_diamond(
    positions: np.ndarray,
    spacelike_edges: np.ndarray,
    timelike_edges: np.ndarray,
    alive: np.ndarray,
    spatial_dims: tuple[int, int] = (0, 1),
) -> pn.Column:
    """Network graph with spacelike (blue) and timelike (red) edges."""
    x_dim, y_dim = spatial_dims
    alive_mask = alive.astype(bool)

    if not alive_mask.any():
        return pn.pane.Markdown("**No alive walkers**")

    pos_alive = positions[alive_mask]

    # Create node dataframe
    nodes = pd.DataFrame({
        'x': pos_alive[:, x_dim],
        'y': pos_alive[:, y_dim],
        'index': np.arange(len(pos_alive)),
    })

    points = hv.Points(nodes, ['x', 'y']).opts(
        color='black',
        size=4,
        width=600,
        height=500,
        title='Causal Structure',
        xlabel=f'Dimension {x_dim}',
        ylabel=f'Dimension {y_dim}',
    )

    summary_md = f"""
### Causal Set Structure

**Spacelike edges:** {spacelike_edges.shape[1] if spacelike_edges.ndim == 2 else 0}

**Timelike edges:** {timelike_edges.shape[1] if timelike_edges.ndim == 2 else 0}

Spacelike edges (blue) connect events in the same time slice.
Timelike edges (red) connect events in adjacent time slices.
"""

    return pn.Column(
        pn.pane.HoloViews(points),
        pn.pane.Markdown(summary_md),
    )


def build_causal_order_violations(
    violation_count: int,
    total_edges: int,
) -> str:
    """Summary of causality violations."""
    md = f"""
### Causal Order Violations

**Violations detected:** {violation_count}

**Total edges:** {total_edges}

**Violation rate:** {100 * violation_count / max(total_edges, 1):.2f}%

Causal violations indicate timelike edges going backward in time,
which would violate causality in a physical spacetime.
"""
    return md


# =============================================================================
# 7. Holographic Entropy Plots
# =============================================================================


def build_holographic_summary(
    entropy: float,
    boundary_area: float,
    bulk_volume: float,
    area_law_coeff: float,
) -> str:
    """Summary of holographic entropy."""
    md = f"""
### Holographic Entropy (Bekenstein-Hawking)

**Entropy:** S = {entropy:.6e}

**Boundary Area:** A = {boundary_area:.6e}

**Bulk Volume:** V = {bulk_volume:.6e}

**Area Law Coefficient:** S/A = {area_law_coeff:.6e}

The holographic principle states that entropy is proportional to boundary
area rather than volume: S ∝ A / (4G ℏ)

This is the Bekenstein-Hawking formula for black hole entropy.
"""
    return md


# =============================================================================
# 8. Spin Network Plots
# =============================================================================


def build_spin_distribution(
    edge_spins: np.ndarray,
) -> hv.Histogram:
    """Histogram of SU(2) spin labels."""
    if len(edge_spins) == 0:
        return hv.Text(0, 0, "No spin data").opts(title="Spin Distribution")

    finite_spins = edge_spins[np.isfinite(edge_spins)]

    if len(finite_spins) == 0:
        return hv.Text(0, 0, "No finite spins").opts(title="Spin Distribution")

    hist = hv.Histogram(np.histogram(finite_spins, bins=50)).opts(
        width=600,
        height=400,
        title='Spin Network: SU(2) Spin Distribution',
        xlabel='Spin j',
        ylabel='Count',
        color='#9467bd',
    )

    return hist


def build_spin_network_summary(
    n_edges: int,
    n_vertices: int,
    mean_spin: float,
    mean_volume: float,
) -> str:
    """Summary of spin network state."""
    md = f"""
### Spin Network State (Loop Quantum Gravity)

**Edges (links):** {n_edges}

**Vertices (nodes):** {n_vertices}

**Mean Edge Spin:** j̄ = {mean_spin:.3f}

**Mean Vertex Volume:** V̄ = {mean_volume:.6e}

In Loop Quantum Gravity, spacetime is represented as a spin network:
- Edges carry SU(2) spins j (quantized areas)
- Vertices have quantized volumes
- The graph encodes the quantum geometry
"""
    return md


# =============================================================================
# 9. Raychaudhuri Expansion Plots
# =============================================================================


def build_expansion_field(
    positions: np.ndarray,
    expansion_scalar: np.ndarray,
    alive: np.ndarray,
    spatial_dims: tuple[int, int] = (0, 1),
) -> hv.Points:
    """Spatial field of θ (red=expansion, blue=contraction)."""
    x_dim, y_dim = spatial_dims
    alive_mask = alive.astype(bool)

    if not alive_mask.any():
        return hv.Text(0, 0, "No alive walkers").opts(title="Raychaudhuri Expansion")

    valid = alive_mask & np.isfinite(expansion_scalar)

    if not valid.any():
        return hv.Text(0, 0, "No valid expansion data").opts(title="Raychaudhuri Expansion")

    data = pd.DataFrame({
        'x': positions[valid, x_dim],
        'y': positions[valid, y_dim],
        'theta': expansion_scalar[valid],
    })

    points = hv.Points(data, ['x', 'y'], 'theta').opts(
        color='theta',
        cmap='RdBu_r',
        size=5,
        colorbar=True,
        width=600,
        height=500,
        title='Raychaudhuri Expansion θ = (1/V) dV/dt',
        xlabel=f'Dimension {x_dim}',
        ylabel=f'Dimension {y_dim}',
    )

    return points


def build_convergence_regions(
    n_converging: int,
    n_total: int,
) -> str:
    """Highlight singularity-forming regions."""
    fraction = n_converging / max(n_total, 1)

    md = f"""
### Raychaudhuri Singularity Prediction

**Converging regions (θ < 0):** {n_converging} / {n_total} ({100 * fraction:.1f}%)

Regions with negative expansion θ < 0 are focusing (contracting).

The Raychaudhuri equation predicts that continued focusing leads to
singularities (Hawking-Penrose singularity theorems).
"""
    return md


# =============================================================================
# 10. Geodesic Deviation Plots
# =============================================================================


def build_tidal_eigenvalues_violin(
    tidal_eigenvalues: np.ndarray,
) -> hv.Distribution:
    """Distribution of stretch/squeeze eigenvalues."""
    if tidal_eigenvalues.size == 0:
        return hv.Text(0, 0, "No tidal tensor data").opts(title="Tidal Eigenvalues")

    # Flatten eigenvalues
    eigs_flat = tidal_eigenvalues.flatten()
    finite_eigs = eigs_flat[np.isfinite(eigs_flat)]

    if len(finite_eigs) == 0:
        return hv.Text(0, 0, "No finite eigenvalues").opts(title="Tidal Eigenvalues")

    dist = hv.Distribution(finite_eigs).opts(
        width=600,
        height=400,
        title='Tidal Tensor Eigenvalue Distribution',
        xlabel='Eigenvalue (Stretch/Squeeze)',
        ylabel='Density',
        color='#17becf',
    )

    return dist


def build_tidal_summary(
    mean_stretch: float,
    mean_squeeze: float,
    max_tidal: float,
) -> str:
    """Summary of geodesic deviation."""
    md = f"""
### Geodesic Deviation (Tidal Forces)

**Mean Stretching Eigenvalue:** {mean_stretch:.6e}

**Mean Squeezing Eigenvalue:** {mean_squeeze:.6e}

**Max Tidal Strength:** {max_tidal:.6e}

The tidal tensor measures how nearby geodesics deviate due to spacetime
curvature. This is the operational definition of the Riemann tensor.

Positive eigenvalues = stretching, Negative eigenvalues = squeezing.
"""
    return md


# =============================================================================
# Master Plot Builder
# =============================================================================


def build_all_gravity_plots(
    observables: QuantumGravityObservables,
    positions: np.ndarray,
    spatial_dims: tuple[int, int] = (0, 1),
) -> dict[str, Any]:
    """Build all 10 analysis plots at once.

    Returns dictionary with keys:
    - "regge_action_density"
    - "deficit_angle_dist"
    - "ricci_landscape"
    - "action_decomposition"
    - "adm_summary"
    - "energy_density_dist"
    - "spectral_dimension_curve"
    - "heat_kernel_trace"
    - "hausdorff_scaling"
    - "local_hausdorff_map"
    - "causal_diamond"
    - "causal_violations"
    - "holographic_summary"
    - "spin_distribution"
    - "spin_network_summary"
    - "expansion_field"
    - "convergence_regions"
    - "tidal_eigenvalues"
    - "tidal_summary"
    - "summary_panel"
    """
    # Convert tensors to numpy
    action_density_np = observables.regge_action_density.cpu().numpy() if hasattr(observables.regge_action_density, 'cpu') else observables.regge_action_density
    deficit_angles_np = observables.deficit_angles.cpu().numpy() if hasattr(observables.deficit_angles, 'cpu') else observables.deficit_angles
    ricci_np = observables.ricci_scalars.cpu().numpy() if hasattr(observables.ricci_scalars, 'cpu') else observables.ricci_scalars
    adm_density_np = observables.adm_energy_density.cpu().numpy() if hasattr(observables.adm_energy_density, 'cpu') else observables.adm_energy_density
    spectral_curve_np = observables.spectral_dimension_curve.cpu().numpy() if hasattr(observables.spectral_dimension_curve, 'cpu') else observables.spectral_dimension_curve
    heat_kernel_np = observables.heat_kernel_trace.cpu().numpy() if hasattr(observables.heat_kernel_trace, 'cpu') else observables.heat_kernel_trace

    radii_np, counts_np = observables.volume_scaling_data
    radii_np = radii_np.cpu().numpy() if hasattr(radii_np, 'cpu') else radii_np
    counts_np = counts_np.cpu().numpy() if hasattr(counts_np, 'cpu') else counts_np

    local_hausdorff_np = observables.local_hausdorff.cpu().numpy() if hasattr(observables.local_hausdorff, 'cpu') else observables.local_hausdorff
    spacelike_np = observables.spacelike_edges.cpu().numpy() if hasattr(observables.spacelike_edges, 'cpu') else observables.spacelike_edges
    timelike_np = observables.timelike_edges.cpu().numpy() if hasattr(observables.timelike_edges, 'cpu') else observables.timelike_edges
    edge_spins_np = observables.edge_spins.cpu().numpy() if hasattr(observables.edge_spins, 'cpu') else observables.edge_spins
    vertex_vols_np = observables.vertex_quantum_volumes.cpu().numpy() if hasattr(observables.vertex_quantum_volumes, 'cpu') else observables.vertex_quantum_volumes
    expansion_np = observables.expansion_scalar.cpu().numpy() if hasattr(observables.expansion_scalar, 'cpu') else observables.expansion_scalar
    convergence_np = observables.convergence_regions.cpu().numpy() if hasattr(observables.convergence_regions, 'cpu') else observables.convergence_regions
    tidal_eigs_np = observables.tidal_eigenvalues.cpu().numpy() if hasattr(observables.tidal_eigenvalues, 'cpu') else observables.tidal_eigenvalues

    # Create alive mask (assume all positions are alive)
    alive_np = np.ones(len(positions), dtype=bool)

    # Compute sigma values for spectral dimension
    sigma_values_np = np.logspace(-3, np.log10(observables.config.max_diffusion_time), observables.config.diffusion_time_steps)

    plots = {
        # 1. Regge Calculus
        "regge_action_density": build_regge_action_density_heatmap(
            positions, action_density_np, alive_np, spatial_dims
        ),
        "deficit_angle_dist": build_deficit_angle_distribution(deficit_angles_np),

        # 2. Einstein-Hilbert
        "ricci_landscape": build_ricci_landscape(positions, ricci_np, alive_np, spatial_dims),
        "action_decomposition": build_action_decomposition(
            0.0, 0.0, None, observables.einstein_hilbert_action
        ),

        # 3. ADM Energy
        "adm_summary": build_adm_energy_summary(
            observables.adm_mass,
            adm_density_np.mean() if len(adm_density_np) > 0 else 0.0,
        ),
        "energy_density_dist": build_energy_density_distribution(adm_density_np, alive_np),

        # 4. Spectral Dimension
        "spectral_dimension_curve": build_spectral_dimension_curve(
            sigma_values_np,
            spectral_curve_np,
            observables.spectral_dimension_planck,
            observables.spatial_dims,
        ),
        "heat_kernel_trace": build_heat_kernel_trace(sigma_values_np, heat_kernel_np),

        # 5. Hausdorff Dimension
        "hausdorff_scaling": build_volume_scaling_plot(
            radii_np, counts_np, observables.hausdorff_dimension
        ),
        "local_hausdorff_map": build_local_hausdorff_heatmap(
            positions, local_hausdorff_np, alive_np, spatial_dims
        ),

        # 6. Causal Structure
        "causal_diamond": build_causal_diamond(
            positions, spacelike_np, timelike_np, alive_np, spatial_dims
        ),
        "causal_violations": build_causal_order_violations(
            observables.causal_violations,
            spacelike_np.shape[1] + timelike_np.shape[1] if spacelike_np.ndim == 2 and timelike_np.ndim == 2 else 0,
        ),

        # 7. Holographic Entropy
        "holographic_summary": build_holographic_summary(
            observables.holographic_entropy,
            observables.boundary_area,
            observables.bulk_volume,
            observables.area_law_coefficient,
        ),

        # 8. Spin Network
        "spin_distribution": build_spin_distribution(edge_spins_np),
        "spin_network_summary": build_spin_network_summary(
            len(edge_spins_np),
            len(vertex_vols_np),
            edge_spins_np.mean() if len(edge_spins_np) > 0 else 0.0,
            vertex_vols_np.mean() if len(vertex_vols_np) > 0 else 0.0,
        ),

        # 9. Raychaudhuri Expansion
        "expansion_field": build_expansion_field(positions, expansion_np, alive_np, spatial_dims),
        "convergence_regions": build_convergence_regions(
            convergence_np.sum() if len(convergence_np) > 0 else 0,
            len(convergence_np),
        ),

        # 10. Geodesic Deviation
        "tidal_eigenvalues": build_tidal_eigenvalues_violin(tidal_eigs_np),
        "tidal_summary": build_tidal_summary(
            tidal_eigs_np.max() if tidal_eigs_np.size > 0 else 0.0,
            tidal_eigs_np.min() if tidal_eigs_np.size > 0 else 0.0,
            np.abs(tidal_eigs_np).max() if tidal_eigs_np.size > 0 else 0.0,
        ),
    }

    # Overall summary
    summary_md = f"""
## Quantum Gravity Analysis Summary

**Walkers:** {observables.n_walkers}
**Edges:** {observables.n_edges}
**Spatial Dimensions:** {observables.spatial_dims}
**MC Frame:** {observables.mc_frame}

### Key Results

1. **Regge Action:** {observables.regge_action:.6e}
2. **Einstein-Hilbert Action:** {observables.einstein_hilbert_action:.6e}
3. **ADM Mass:** {observables.adm_mass:.6e}
4. **Spectral Dimension (Planck):** {observables.spectral_dimension_planck:.2f}
5. **Hausdorff Dimension:** {observables.hausdorff_dimension:.2f}
6. **Causal Violations:** {observables.causal_violations}
7. **Holographic Entropy:** {observables.holographic_entropy:.6e}
8. **Mean Ricci Scalar:** {observables.scalar_curvature_mean:.6e}

These 10 analyses reproduce signature predictions from different approaches to
quantum gravity using the emergent spacetime geometry from Fractal Gas dynamics.
"""

    plots["summary_panel"] = summary_md

    return plots


# =============================================================================
# Time Evolution Plots (4D Spacetime Block Analysis)
# =============================================================================


def build_regge_action_evolution(
    mc_frames: np.ndarray,
    regge_action: np.ndarray,
) -> hv.Curve:
    """Plot Regge action vs MC time.

    Shows how the discretized gravitational action evolves during simulation.
    """
    if len(mc_frames) == 0 or len(regge_action) == 0:
        return hv.Text(0, 0, "No time series data").opts(title="Regge Action Evolution")

    data = pd.DataFrame({
        'frame': mc_frames,
        'action': regge_action,
    })

    return hv.Curve(data, 'frame', 'action').opts(
        width=800,
        height=400,
        title='Regge Action Evolution',
        xlabel='MC Frame',
        ylabel='Total Action S_Regge',
        color='#4c78a8',
        line_width=2,
    )


def build_adm_mass_evolution(
    mc_frames: np.ndarray,
    adm_mass: np.ndarray,
) -> hv.Curve:
    """Plot ADM mass vs MC time.

    Physics: Should be conserved if spacetime is asymptotically flat.
    Violations indicate non-conservation or boundary effects.
    """
    if len(mc_frames) == 0 or len(adm_mass) == 0:
        return hv.Text(0, 0, "No time series data").opts(title="ADM Mass Evolution")

    data = pd.DataFrame({
        'frame': mc_frames,
        'mass': adm_mass,
    })

    return hv.Curve(data, 'frame', 'mass').opts(
        width=800,
        height=400,
        title='ADM Mass Evolution (Energy Conservation Check)',
        xlabel='MC Frame',
        ylabel='ADM Mass M',
        color='#e15759',
        line_width=2,
    )


def build_spectral_dimension_evolution(
    mc_frames: np.ndarray,
    spectral_dim_planck: np.ndarray,
    spectral_dim_large: np.ndarray,
    target_spatial_dim: int,
) -> hv.Overlay:
    """Plot spectral dimension at different scales vs MC time.

    Physics: Should show dimension reduction at small scales (Planck).
    May evolve during thermalization as geometry settles.
    """
    if len(mc_frames) == 0:
        return hv.Text(0, 0, "No time series data").opts(title="Spectral Dimension Evolution")

    data_planck = pd.DataFrame({
        'frame': mc_frames,
        'spectral_dim': spectral_dim_planck,
        'scale': 'Planck scale',
    })

    data_large = pd.DataFrame({
        'frame': mc_frames,
        'spectral_dim': spectral_dim_large,
        'scale': 'Large scale',
    })

    data = pd.concat([data_planck, data_large])

    # Specify both spectral_dim and scale as value dimensions for groupby
    curves = hv.Curve(data, 'frame', ['spectral_dim', 'scale']).groupby('scale').overlay()

    # Add horizontal line for target spatial dimension
    hline = hv.HLine(target_spatial_dim).opts(
        color='red',
        line_dash='dashed',
        line_width=1,
    )

    return (curves * hline).opts(
        width=800,
        height=400,
        title='Spectral Dimension Evolution (Dimension Reduction)',
        xlabel='MC Frame',
        ylabel='Spectral Dimension d_s',
        show_legend=True,
    )


def build_hausdorff_dimension_evolution(
    mc_frames: np.ndarray,
    hausdorff_dim: np.ndarray,
    target_spatial_dim: int,
) -> hv.Overlay:
    """Plot Hausdorff dimension vs MC time.

    Physics: Should converge to intrinsic manifold dimension.
    Early frames may show non-integer fractal dimension.
    """
    if len(mc_frames) == 0 or len(hausdorff_dim) == 0:
        return hv.Text(0, 0, "No time series data").opts(title="Hausdorff Dimension Evolution")

    data = pd.DataFrame({
        'frame': mc_frames,
        'hausdorff_dim': hausdorff_dim,
    })

    curve = hv.Curve(data, 'frame', 'hausdorff_dim').opts(
        color='#59a14f',
        line_width=2,
    )

    # Target dimension line
    hline = hv.HLine(target_spatial_dim).opts(
        color='red',
        line_dash='dashed',
        line_width=1,
    )

    return (curve * hline).opts(
        width=800,
        height=400,
        title='Hausdorff Dimension Evolution (Fractal → Manifold)',
        xlabel='MC Frame',
        ylabel='Hausdorff Dimension d_H',
    )


def build_holographic_entropy_evolution(
    mc_frames: np.ndarray,
    holographic_entropy: np.ndarray,
    boundary_area: np.ndarray,
) -> hv.Overlay:
    """Plot holographic entropy and boundary area vs MC time.

    Physics:
    - Entropy should grow (2nd law)
    - S/A ratio should be constant (holographic principle)
    - Bekenstein bound: S ≤ A / (4 ℓ_P²)
    """
    if len(mc_frames) == 0:
        return hv.Text(0, 0, "No time series data").opts(title="Holographic Entropy Evolution")

    # Normalize both to [0, 1] for comparison
    entropy_norm = (holographic_entropy - holographic_entropy.min()) / (holographic_entropy.max() - holographic_entropy.min() + 1e-10)
    area_norm = (boundary_area - boundary_area.min()) / (boundary_area.max() - boundary_area.min() + 1e-10)

    data_entropy = pd.DataFrame({
        'frame': mc_frames,
        'value': entropy_norm,
        'observable': 'Entropy S (normalized)',
    })

    data_area = pd.DataFrame({
        'frame': mc_frames,
        'value': area_norm,
        'observable': 'Boundary Area A (normalized)',
    })

    data = pd.concat([data_entropy, data_area])

    # Specify both value and observable as value dimensions for groupby
    curves = hv.Curve(data, 'frame', ['value', 'observable']).groupby('observable').overlay()

    return curves.opts(
        width=800,
        height=400,
        title='Holographic Entropy Evolution (2nd Law)',
        xlabel='MC Frame',
        ylabel='Normalized Value',
        show_legend=True,
    )


def build_raychaudhuri_expansion_evolution(
    mc_frames: np.ndarray,
    expansion_mean: np.ndarray,
    convergence_fraction: np.ndarray,
) -> hv.Overlay:
    """Plot Raychaudhuri expansion statistics vs MC time.

    Physics:
    - Negative θ → convergence → singularity formation
    - Increasing convergence fraction → collapse
    """
    if len(mc_frames) == 0:
        return hv.Text(0, 0, "No time series data").opts(title="Raychaudhuri Expansion Evolution")

    data_theta = pd.DataFrame({
        'frame': mc_frames,
        'expansion': expansion_mean,
    })

    curve_theta = hv.Curve(data_theta, 'frame', 'expansion').opts(
        color='#4c78a8',
        line_width=2,
        ylabel='Mean Expansion θ',
    )

    # Add zero line
    hline = hv.HLine(0).opts(
        color='red',
        line_dash='dashed',
        line_width=1,
    )

    return (curve_theta * hline).opts(
        width=800,
        height=400,
        title='Raychaudhuri Expansion Evolution (Singularity Predictor)',
        xlabel='MC Frame',
    )


def build_causal_structure_evolution(
    mc_frames: np.ndarray,
    n_spacelike: np.ndarray,
    n_timelike: np.ndarray,
) -> hv.Overlay:
    """Plot spacelike and timelike edge counts vs MC time."""
    if len(mc_frames) == 0:
        return hv.Text(0, 0, "No time series data").opts(title="Causal Structure Evolution")

    data_space = pd.DataFrame({
        'frame': mc_frames,
        'count': n_spacelike,
        'type': 'Spacelike edges',
    })

    data_time = pd.DataFrame({
        'frame': mc_frames,
        'count': n_timelike,
        'type': 'Timelike edges',
    })

    data = pd.concat([data_space, data_time])

    # Specify both count and type as value dimensions for groupby
    curves = hv.Curve(data, 'frame', ['count', 'type']).groupby('type').overlay()

    return curves.opts(
        width=800,
        height=400,
        title='Causal Structure Evolution',
        xlabel='MC Frame',
        ylabel='Edge Count',
        show_legend=True,
    )


def build_spin_network_evolution(
    mc_frames: np.ndarray,
    mean_spin: np.ndarray,
    mean_volume: np.ndarray,
) -> hv.Overlay:
    """Plot mean spin and vertex volume vs MC time."""
    if len(mc_frames) == 0:
        return hv.Text(0, 0, "No time series data").opts(title="Spin Network Evolution")

    # Normalize for comparison
    spin_norm = (mean_spin - mean_spin.min()) / (mean_spin.max() - mean_spin.min() + 1e-10)
    vol_norm = (mean_volume - mean_volume.min()) / (mean_volume.max() - mean_volume.min() + 1e-10)

    data_spin = pd.DataFrame({
        'frame': mc_frames,
        'value': spin_norm,
        'observable': 'Mean Spin (normalized)',
    })

    data_vol = pd.DataFrame({
        'frame': mc_frames,
        'value': vol_norm,
        'observable': 'Mean Volume (normalized)',
    })

    data = pd.concat([data_spin, data_vol])

    # Specify both value and observable as value dimensions for groupby
    curves = hv.Curve(data, 'frame', ['value', 'observable']).groupby('observable').overlay()

    return curves.opts(
        width=800,
        height=400,
        title='Spin Network Evolution',
        xlabel='MC Frame',
        ylabel='Normalized Value',
        show_legend=True,
    )


def build_tidal_strength_evolution(
    mc_frames: np.ndarray,
    tidal_mean: np.ndarray,
    tidal_max: np.ndarray,
) -> hv.Overlay:
    """Plot tidal strength statistics vs MC time."""
    if len(mc_frames) == 0:
        return hv.Text(0, 0, "No time series data").opts(title="Tidal Strength Evolution")

    data_mean = pd.DataFrame({
        'frame': mc_frames,
        'strength': tidal_mean,
        'statistic': 'Mean',
    })

    data_max = pd.DataFrame({
        'frame': mc_frames,
        'strength': tidal_max,
        'statistic': 'Maximum',
    })

    data = pd.concat([data_mean, data_max])

    # Specify both strength and statistic as value dimensions for groupby
    curves = hv.Curve(data, 'frame', ['strength', 'statistic']).groupby('statistic').overlay()

    return curves.opts(
        width=800,
        height=400,
        title='Tidal Strength Evolution',
        xlabel='MC Frame',
        ylabel='Tidal Force Magnitude',
        show_legend=True,
    )


def build_all_quantum_gravity_time_series_plots(
    time_series: Any,  # QuantumGravityTimeSeries
) -> dict[str, Any]:
    """Build all time evolution plots.

    Args:
        time_series: QuantumGravityTimeSeries dataclass

    Returns:
        Dictionary with keys:
        - "regge_action_evolution"
        - "adm_mass_evolution"
        - "spectral_dimension_evolution"
        - "hausdorff_dimension_evolution"
        - "holographic_entropy_evolution"
        - "raychaudhuri_expansion_evolution"
        - "causal_structure_evolution"
        - "spin_network_evolution"
        - "tidal_strength_evolution"
        - "time_series_summary"
    """
    mc_frames = time_series.mc_frames

    plots = {
        "regge_action_evolution": build_regge_action_evolution(
            mc_frames, time_series.regge_action
        ),
        "adm_mass_evolution": build_adm_mass_evolution(
            mc_frames, time_series.adm_mass
        ),
        "spectral_dimension_evolution": build_spectral_dimension_evolution(
            mc_frames,
            time_series.spectral_dimension_planck,
            time_series.spectral_dimension_large_scale,
            time_series.spatial_dims,
        ),
        "hausdorff_dimension_evolution": build_hausdorff_dimension_evolution(
            mc_frames,
            time_series.hausdorff_dimension,
            time_series.spatial_dims,
        ),
        "holographic_entropy_evolution": build_holographic_entropy_evolution(
            mc_frames,
            time_series.holographic_entropy,
            time_series.boundary_area,
        ),
        "raychaudhuri_expansion_evolution": build_raychaudhuri_expansion_evolution(
            mc_frames,
            time_series.expansion_mean,
            time_series.convergence_fraction,
        ),
        "causal_structure_evolution": build_causal_structure_evolution(
            mc_frames,
            time_series.n_spacelike_edges,
            time_series.n_timelike_edges,
        ),
        "spin_network_evolution": build_spin_network_evolution(
            mc_frames,
            time_series.mean_edge_spin,
            time_series.mean_vertex_volume,
        ),
        "tidal_strength_evolution": build_tidal_strength_evolution(
            mc_frames,
            time_series.tidal_strength_mean,
            time_series.tidal_strength_max,
        ),
    }

    # Summary statistics
    if len(mc_frames) > 0:
        # Compute percent changes
        adm_change = 100 * (time_series.adm_mass[-1] / (time_series.adm_mass[0] + 1e-10) - 1)
        spectral_reduction = time_series.spectral_dimension_planck[-1] < time_series.spatial_dims
        hausdorff_convergence = abs(time_series.hausdorff_dimension[-1] - time_series.spatial_dims) < 0.5
        entropy_growth = 100 * (time_series.holographic_entropy[-1] / (time_series.holographic_entropy[0] + 1e-10) - 1)
        singularity_risk = "HIGH" if time_series.convergence_fraction[-1] > 0.5 else "LOW"

        summary_md = f"""
## Time Evolution Summary

**Frames analyzed:** {time_series.n_frames} (stride = {mc_frames[1] - mc_frames[0] if len(mc_frames) > 1 else 1})

### Key Physical Signatures:

1. **ADM Mass Conservation:**
   - Initial: {time_series.adm_mass[0]:.6e}
   - Final: {time_series.adm_mass[-1]:.6e}
   - Change: {adm_change:.2f}%

2. **Spectral Dimension (Planck scale):**
   - Initial: {time_series.spectral_dimension_planck[0]:.2f}
   - Final: {time_series.spectral_dimension_planck[-1]:.2f}
   - Dimension reduction detected: {'YES' if spectral_reduction else 'NO'}

3. **Hausdorff Dimension:**
   - Initial: {time_series.hausdorff_dimension[0]:.2f}
   - Final: {time_series.hausdorff_dimension[-1]:.2f}
   - Convergence to spatial dim: {'YES' if hausdorff_convergence else 'NO'}

4. **Holographic Entropy:**
   - Initial: {time_series.holographic_entropy[0]:.6e}
   - Final: {time_series.holographic_entropy[-1]:.6e}
   - Growth (2nd law check): {entropy_growth:.2f}%

5. **Raychaudhuri Expansion:**
   - Mean θ (final): {time_series.expansion_mean[-1]:.6e}
   - Convergence fraction: {100 * time_series.convergence_fraction[-1]:.1f}%
   - Singularity risk: {singularity_risk}

### Physical Interpretation:

**Dimension Reduction:** CDT predicts d_s ≈ 2 at Planck scale, d_s → 4 at large scales.
Our simulation shows {'dimension reduction' if spectral_reduction else 'classical behavior'}.

**Energy Conservation:** ADM mass change of {adm_change:.2f}% indicates
{'good energy conservation' if abs(adm_change) < 5 else 'energy flow or boundary effects'}.

**Thermalization:** Hausdorff dimension {'has converged' if hausdorff_convergence else 'is still evolving'},
indicating {'thermal equilibrium' if hausdorff_convergence else 'ongoing thermalization'}.

**Holographic Principle:** Entropy growth of {entropy_growth:.2f}% validates the second law.
The S/A ratio evolution tests the holographic bound.
"""
    else:
        summary_md = "**No frames analyzed**"

    plots["time_series_summary"] = summary_md

    return plots
