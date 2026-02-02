"""Plotting functions for Higgs field observables.

This module provides HoloViews-based visualizations for geometric observables
computed from the emergent manifold in the Fractal Gas framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

if TYPE_CHECKING:
    from fragile.fractalai.qft.higgs_observables import HiggsObservables


def build_metric_tensor_heatmap(
    positions: np.ndarray,
    metric_tensors: np.ndarray,
    alive: np.ndarray,
    component: tuple[int, int] = (0, 0),
    spatial_dims: tuple[int, int] = (0, 1),
) -> hv.Points:
    """Scatter plot of walkers colored by metric tensor component g_μν.

    Args:
        positions: Walker positions [N, d]
        metric_tensors: Metric tensor at each walker [N, d, d]
        alive: Alive mask [N]
        component: Which tensor component to plot (row, col)
        spatial_dims: Which spatial dimensions to use for x,y axes

    Returns:
        HoloViews Points plot
    """
    x_dim, y_dim = spatial_dims

    # Filter alive walkers
    alive_mask = alive.astype(bool)

    # Validate component indices
    i, j = component
    if i >= metric_tensors.shape[1] or j >= metric_tensors.shape[2]:
        return hv.Text(0, 0, f"Invalid component ({i},{j}) for shape {metric_tensors.shape}").opts(
            title="Metric Tensor Component - Error"
        )

    # Extract metric component values
    metric_values = metric_tensors[:, i, j]

    # Validate spatial dimensions
    if x_dim >= positions.shape[1] or y_dim >= positions.shape[1]:
        return hv.Text(
            0, 0,
            f"Spatial dims ({x_dim}, {y_dim}) invalid for {positions.shape[1]}D positions"
        ).opts(title="Metric Tensor Heatmap - Error")

    # Filter finite values
    valid = alive_mask & np.isfinite(metric_values)
    if not valid.any():
        return hv.Text(0, 0, "No valid data").opts(title=f"Metric Tensor g_{i}{j}")

    # Create scatter plot data
    data = pd.DataFrame({
        'x': positions[valid, x_dim],
        'y': positions[valid, y_dim],
        f'g_{i}{j}': metric_values[valid],
    })

    points = hv.Points(data, ['x', 'y'], f'g_{i}{j}').opts(
        color=f'g_{i}{j}',
        cmap='viridis',
        size=5,
        colorbar=True,
        width=600,
        height=500,
        title=f'Metric Tensor Component g_{i}{j}',
        xlabel=f'Dimension {x_dim}',
        ylabel=f'Dimension {y_dim}',
    )

    return points


def build_centroid_vector_field(
    positions: np.ndarray,
    centroid_vectors: np.ndarray,
    alive: np.ndarray,
    spatial_dims: tuple[int, int] = (0, 1),
    subsample: int = 1,
) -> hv.VectorField:
    """Vector field showing Lloyd vectors (drift).

    Args:
        positions: Walker positions [N, d]
        centroid_vectors: Lloyd displacement vectors [N, d]
        alive: Alive mask [N]
        spatial_dims: Which spatial dimensions to use for plotting
        subsample: Plot every Nth vector (for clarity)

    Returns:
        HoloViews VectorField
    """
    x_dim, y_dim = spatial_dims

    # Filter alive walkers
    alive_mask = alive.astype(bool)
    pos_alive = positions[alive_mask]
    vec_alive = centroid_vectors[alive_mask]

    if len(pos_alive) == 0:
        return hv.Text(0, 0, "No alive walkers").opts(title="Centroid Vector Field")

    # Defensive: ensure vectors have proper dimensionality
    if vec_alive.ndim == 1:
        # Edge case: 1D vectors (should have been 2D [N, d])
        # This indicates a bug upstream, but handle gracefully
        return hv.Text(
            0, 0,
            f"Invalid centroid vectors shape {vec_alive.shape} (expected 2D)"
        ).opts(title="Centroid Vector Field - Error")

    # Validate spatial dimensions exist
    x_dim, y_dim = spatial_dims
    if x_dim >= vec_alive.shape[1] or y_dim >= vec_alive.shape[1]:
        return hv.Text(
            0, 0,
            f"Spatial dims ({x_dim}, {y_dim}) invalid for {vec_alive.shape[1]}D vectors"
        ).opts(title="Centroid Vector Field - Error")

    # Subsample for clarity
    if subsample > 1:
        indices = np.arange(0, len(pos_alive), subsample)
        pos_alive = pos_alive[indices]
        vec_alive = vec_alive[indices]

    # Extract components
    x = pos_alive[:, x_dim]
    y = pos_alive[:, y_dim]
    u = vec_alive[:, x_dim]
    v = vec_alive[:, y_dim]

    # Compute magnitude for coloring
    magnitude = np.sqrt(u**2 + v**2)

    data = pd.DataFrame({
        'x': x,
        'y': y,
        'u': u,
        'v': v,
        'magnitude': magnitude,
    })

    vector_field = hv.VectorField(data, ['x', 'y'], ['u', 'v', 'magnitude']).opts(
        color='magnitude',
        cmap='fire',
        magnitude='magnitude',
        pivot='tail',
        colorbar=True,
        width=600,
        height=500,
        title='Centroid Displacement Field (Lloyd Vectors)',
        xlabel=f'Dimension {x_dim}',
        ylabel=f'Dimension {y_dim}',
    )

    return vector_field


def build_ricci_scalar_distribution(
    ricci_scalars: np.ndarray,
    alive: np.ndarray,
) -> hv.Histogram:
    """Histogram of Ricci scalar values with logarithmic x-axis.

    Args:
        ricci_scalars: Ricci scalar at each walker [N]
        alive: Alive mask [N]

    Returns:
        HoloViews Histogram
    """
    alive_mask = alive.astype(bool)
    ricci_alive = ricci_scalars[alive_mask]

    if len(ricci_alive) == 0:
        return hv.Text(0, 0, "No alive walkers").opts(title="Ricci Scalar Distribution")

    # Remove infinities and NaNs
    ricci_finite = ricci_alive[np.isfinite(ricci_alive)]

    if len(ricci_finite) == 0:
        return hv.Text(0, 0, "No finite Ricci scalars").opts(title="Ricci Scalar Distribution")

    # Handle negative and positive values separately for log scale
    # Use absolute value and create logarithmic bins
    ricci_abs = np.abs(ricci_finite)
    ricci_abs_nonzero = ricci_abs[ricci_abs > 0]

    if len(ricci_abs_nonzero) == 0:
        # All values are zero, use linear bins
        hist = hv.Histogram(np.histogram(ricci_finite, bins=50)).opts(
            width=600,
            height=400,
            title='Ricci Scalar Distribution',
            xlabel='Ricci Scalar R',
            ylabel='Count',
            color='#4c78a8',
        )
    else:
        # Create logarithmic bins for absolute values
        min_val = ricci_abs_nonzero.min()
        max_val = ricci_abs_nonzero.max()

        # Create log-spaced bins
        log_bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)

        hist = hv.Histogram(np.histogram(ricci_abs, bins=log_bins)).opts(
            width=600,
            height=400,
            title='Ricci Scalar Distribution (|R|)',
            xlabel='|Ricci Scalar R|',
            ylabel='Count',
            color='#4c78a8',
            logx=True,  # Logarithmic x-axis
        )

    return hist


def build_geodesic_distance_scatter(
    euclidean_distances: np.ndarray,
    geodesic_distances: np.ndarray,
) -> hv.Scatter:
    """Scatter plot comparing Euclidean vs geodesic distances.

    Args:
        euclidean_distances: Euclidean edge lengths [E]
        geodesic_distances: Geodesic edge lengths [E]

    Returns:
        HoloViews Scatter plot with diagonal reference line
    """
    # Remove infinities and NaNs
    valid = np.isfinite(euclidean_distances) & np.isfinite(geodesic_distances)
    euc = euclidean_distances[valid]
    geo = geodesic_distances[valid]

    if len(euc) == 0:
        return hv.Text(0, 0, "No valid distances").opts(title="Geodesic vs Euclidean Distances")

    data = pd.DataFrame({
        'euclidean': euc,
        'geodesic': geo,
    })

    scatter = hv.Scatter(data, 'euclidean', 'geodesic').opts(
        color='#4c78a8',
        size=3,
        alpha=0.5,
        width=600,
        height=500,
        title='Geodesic vs Euclidean Distances',
        xlabel='Euclidean Distance',
        ylabel='Geodesic Distance',
    )

    # Add diagonal reference line (geodesic = euclidean)
    max_dist = max(euc.max(), geo.max())
    min_dist = min(euc.min(), geo.min())
    diagonal = hv.Curve([(min_dist, min_dist), (max_dist, max_dist)]).opts(
        color='red',
        line_dash='dashed',
        line_width=2,
    )

    return scatter * diagonal


def build_higgs_action_summary(
    observables: HiggsObservables,
) -> pn.Column:
    """Summary panel showing action components and statistics.

    Args:
        observables: HiggsObservables dataclass

    Returns:
        Panel Column with formatted summary
    """
    config = observables.config

    # Format action components (convert tensors to Python floats first)
    mean_field = float(observables.scalar_field.mean())
    std_field = float(observables.scalar_field.std())
    min_field = float(observables.scalar_field.min())
    max_field = float(observables.scalar_field.max())
    mean_volume = float(observables.cell_volumes.mean())
    gravity_term_str = f"{observables.gravity_term:.6e}" if observables.gravity_term is not None else "N/A"

    action_md = (
        "### Higgs Action Summary\n\n"
        f"**Total Action:** {observables.total_action:.6e}\n\n"
        "**Components:**\n"
        f"- Kinetic term: {observables.kinetic_term:.6e}\n"
        f"- Potential term V(phi): {observables.potential_term:.6e}\n"
        f"- Gravity term: {gravity_term_str}\n\n"
        "**Field Statistics:**\n"
        f"- Mean phi: {mean_field:.6f}\n"
        f"- Std phi: {std_field:.6f}\n"
        f"- Min phi: {min_field:.6f}\n"
        f"- Max phi: {max_field:.6f}\n\n"
        "**Geometry Statistics:**\n"
        f"- Walkers: {observables.n_walkers}\n"
        f"- Edges: {observables.n_edges}\n"
        f"- Mean cell volume: {mean_volume:.6e}\n"
        f"- Volume variance (curvature proxy): {observables.volume_variance:.6e}\n\n"
        "**Configuration:**\n"
        f"- Frame: {observables.mc_frame}\n"
        f"- mu_sq: {config.mu_sq}\n"
        f"- lambda_higgs: {config.lambda_higgs}\n"
        f"- alpha_gravity: {config.alpha_gravity}\n"
        f"- h_eff: {config.h_eff}\n"
    )

    return pn.Column(
        pn.pane.Markdown(action_md),
        sizing_mode="stretch_width",
    )


def build_volume_vs_curvature_scatter(
    cell_volumes: np.ndarray,
    ricci_scalars: np.ndarray,
    alive: np.ndarray,
) -> hv.Scatter:
    """Scatter plot showing relationship between cell volume and curvature.

    Args:
        cell_volumes: Voronoi cell volumes [N]
        ricci_scalars: Ricci scalar at each walker [N]
        alive: Alive mask [N]

    Returns:
        HoloViews Scatter plot
    """
    alive_mask = alive.astype(bool)
    vols = cell_volumes[alive_mask]
    ricci = ricci_scalars[alive_mask]

    # Filter finite values
    valid = np.isfinite(vols) & np.isfinite(ricci) & (vols > 0)
    vols = vols[valid]
    ricci = ricci[valid]

    if len(vols) == 0:
        return hv.Text(0, 0, "No valid data").opts(title="Cell Volume vs Curvature")

    data = pd.DataFrame({
        'volume': vols,
        'ricci': ricci,
    })

    scatter = hv.Scatter(data, 'volume', 'ricci').opts(
        color='#e15759',
        size=4,
        alpha=0.6,
        width=600,
        height=500,
        title='Cell Volume vs Ricci Scalar',
        xlabel='Cell Volume',
        ylabel='Ricci Scalar R',
        logx=True,  # Often volumes span orders of magnitude
    )

    return scatter


def build_scalar_field_map(
    positions: np.ndarray,
    scalar_field: np.ndarray,
    alive: np.ndarray,
    spatial_dims: tuple[int, int] = (0, 1),
) -> hv.Points:
    """2D/3D visualization of the scalar field on the manifold.

    Args:
        positions: Walker positions [N, d]
        scalar_field: Scalar field values [N]
        alive: Alive mask [N]
        spatial_dims: Which spatial dimensions to plot

    Returns:
        HoloViews Points plot
    """
    x_dim, y_dim = spatial_dims

    alive_mask = alive.astype(bool)
    pos_alive = positions[alive_mask]
    field_alive = scalar_field[alive_mask]

    if len(pos_alive) == 0:
        return hv.Text(0, 0, "No alive walkers").opts(title="Scalar Field Configuration")

    data = pd.DataFrame({
        f'dim_{x_dim}': pos_alive[:, x_dim],
        f'dim_{y_dim}': pos_alive[:, y_dim],
        'phi': field_alive,
    })

    points = hv.Points(data, [f'dim_{x_dim}', f'dim_{y_dim}'], 'phi').opts(
        color='phi',
        cmap='coolwarm',
        size=5,
        colorbar=True,
        width=600,
        height=500,
        title='Higgs Field φ Configuration',
        xlabel=f'Dimension {x_dim}',
        ylabel=f'Dimension {y_dim}',
    )

    return points


def build_metric_eigenvalues_distribution(
    metric_tensors: np.ndarray,
    alive: np.ndarray,
) -> hv.Overlay:
    """Distribution of metric tensor eigenvalues (measures anisotropy).

    Args:
        metric_tensors: Metric tensors [N, d, d]
        alive: Alive mask [N]

    Returns:
        HoloViews overlay of histograms for each eigenvalue
    """
    alive_mask = alive.astype(bool)
    metrics_alive = metric_tensors[alive_mask]

    if len(metrics_alive) == 0:
        return hv.Text(0, 0, "No alive walkers").opts(title="Metric Eigenvalue Distribution")

    # Compute eigenvalues for each metric tensor
    eigenvalues = np.linalg.eigvalsh(metrics_alive)  # [N, d]

    # Create histogram for each eigenvalue dimension
    d = eigenvalues.shape[1]
    histograms = []

    for i in range(d):
        eigs = eigenvalues[:, i]
        eigs_finite = eigs[np.isfinite(eigs) & (eigs > 0)]

        if len(eigs_finite) > 0:
            hist = hv.Histogram(np.histogram(np.log10(eigs_finite), bins=50)).opts(
                alpha=0.6,
                color=hv.Cycle('Category10').values[i % 10],
            )
            histograms.append(hist)

    if not histograms:
        return hv.Text(0, 0, "No finite eigenvalues").opts(title="Metric Eigenvalue Distribution")

    overlay = hv.Overlay(histograms).opts(
        width=600,
        height=400,
        title='Metric Eigenvalue Distribution (log scale)',
        xlabel='log₁₀(Eigenvalue)',
        ylabel='Count',
        show_legend=True,
    )

    return overlay


def build_all_higgs_plots(
    observables: HiggsObservables,
    positions: np.ndarray,
    spatial_dims: tuple[int, int] = (0, 1),
    metric_component: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    """Build all Higgs plots at once.

    Args:
        observables: HiggsObservables dataclass
        positions: Walker positions [N, d]
        spatial_dims: Which spatial dimensions to use for 2D plots
        metric_component: Which metric tensor component to visualize (row, col)

    Returns:
        Dictionary of plot names to HoloViews objects
    """
    # Convert tensors to numpy
    centroid_np = observables.centroid_vectors.cpu().numpy() if hasattr(observables.centroid_vectors, 'cpu') else observables.centroid_vectors
    cell_volumes_np = observables.cell_volumes.cpu().numpy() if hasattr(observables.cell_volumes, 'cpu') else observables.cell_volumes
    scalar_field_np = observables.scalar_field.cpu().numpy() if hasattr(observables.scalar_field, 'cpu') else observables.scalar_field
    alive_np = observables.alive.cpu().numpy() if hasattr(observables.alive, 'cpu') else observables.alive
    metric_np = observables.metric_tensors.cpu().numpy() if hasattr(observables.metric_tensors, 'cpu') else observables.metric_tensors

    ricci_np = None
    if observables.ricci_scalars is not None:
        ricci_np = observables.ricci_scalars.cpu().numpy() if hasattr(observables.ricci_scalars, 'cpu') else observables.ricci_scalars

    # Compute Euclidean distances for comparison
    edge_index_np = observables.edge_index.cpu().numpy() if hasattr(observables.edge_index, 'cpu') else observables.edge_index
    geo_dist_np = observables.geodesic_distances.cpu().numpy() if hasattr(observables.geodesic_distances, 'cpu') else observables.geodesic_distances

    if edge_index_np.shape[1] > 0:
        row, col = edge_index_np[0], edge_index_np[1]
        delta = positions[col] - positions[row]
        euc_dist = np.linalg.norm(delta, axis=1)
    else:
        euc_dist = np.array([])

    plots = {
        "action_summary": build_higgs_action_summary(observables),
        "metric_tensor_heatmap": build_metric_tensor_heatmap(
            positions, metric_np, alive_np, metric_component, spatial_dims
        ),
        "centroid_vector_field": build_centroid_vector_field(positions, centroid_np, alive_np, spatial_dims),
        "scalar_field_map": build_scalar_field_map(positions, scalar_field_np, alive_np, spatial_dims),
        "metric_eigenvalues_distribution": build_metric_eigenvalues_distribution(metric_np, alive_np),
    }

    if len(euc_dist) > 0 and len(geo_dist_np) > 0:
        plots["geodesic_distance_scatter"] = build_geodesic_distance_scatter(euc_dist, geo_dist_np)

    if ricci_np is not None:
        plots["ricci_scalar_distribution"] = build_ricci_scalar_distribution(ricci_np, alive_np)
        plots["volume_vs_curvature_scatter"] = build_volume_vs_curvature_scatter(cell_volumes_np, ricci_np, alive_np)

    return plots
