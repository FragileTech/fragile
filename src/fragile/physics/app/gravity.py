"""Gravitational / holographic dashboard building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import torch

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.qft.einstein_equations import compute_einstein_test, EinsteinConfig
from fragile.fractalai.qft.einstein_equations_plotting import build_scalar_test_log_plot
from fragile.fractalai.qft.smeared_operators import compute_pairwise_distance_matrices_from_history


@dataclass
class HolographicPrincipleSection:
    """Container for the holographic dashboard section."""

    fractal_set_tab: pn.layout.base.Column
    fractal_set_status: pn.pane.Markdown
    fractal_set_run_button: pn.widgets.Button
    on_run_fractal_set: Callable[[Any], None]
    on_history_changed: Callable[[bool], None]


def _to_numpy(t: Any) -> np.ndarray:
    """Convert a tensor-like to ndarray."""
    if hasattr(t, "cpu"):
        return t.cpu().numpy()
    return np.asarray(t)


class FractalSetSettings(param.Parameterized):
    """Settings for Fractal Set IG/CST area-law measurements."""

    warmup_fraction = param.Number(
        default=0.1,
        bounds=(0.0, 0.95),
        doc="Fraction of early recorded transitions to discard before measuring.",
    )
    frame_stride = param.Integer(
        default=1,
        bounds=(1, 100),
        doc="Analyze every Nth recorded transition after warmup.",
    )
    max_frames = param.Integer(
        default=120,
        bounds=(1, None),
        doc="Maximum number of transitions to analyze (evenly subsampled).",
    )
    n_cut_samples = param.Integer(
        default=50,
        bounds=(5, 200),
        doc="Number of boundary samples for hyperplane/spherical sweeps.",
    )
    partition_family = param.ObjectSelector(
        default="all",
        objects=["all", "spatial", "graph", "random"],
        doc="Partition generator family.",
    )
    partition_axis = param.Integer(
        default=0,
        bounds=(0, 10),
        doc="Coordinate axis used for hyperplane/median cuts.",
    )
    cut_geometry = param.ObjectSelector(
        default="all",
        objects=["all", "hyperplane", "spherical", "median"],
        doc="Boundary geometry to evaluate.",
    )
    graph_cut_source = param.ObjectSelector(
        default="all",
        objects=["all", "distance", "fitness", "both"],
        doc="Companion graph used for spectral graph cuts.",
    )
    min_partition_size = param.Integer(
        default=5,
        bounds=(1, None),
        doc="Minimum walkers per side for graph/random cuts.",
    )
    random_partitions = param.Integer(
        default=50,
        bounds=(1, 500),
        doc="Number of random baseline partitions per analyzed transition.",
    )
    random_balanced = param.Boolean(
        default=True,
        doc="Use balanced random partitions (|A| ~= N/2).",
    )
    random_seed = param.Integer(
        default=12345,
        bounds=(0, None),
        doc="Seed for random baseline partition sampling.",
    )
    use_geometry_correction = param.Boolean(
        default=True,
        doc="Apply Riemannian distance and volume corrections to IG/CST measures.",
    )
    metric_display = param.ObjectSelector(
        default="both",
        objects=["raw", "geometry", "both"],
        doc="Which metric family to show in regressions and plots.",
    )
    geometry_kernel_length_scale = param.Number(
        default=None,
        bounds=(1e-6, None),
        allow_None=True,
        doc="Length scale for geometric kernel exp(-d_g^2/(2 l^2)); None uses history/default.",
    )
    geometry_min_eig = param.Number(
        default=1e-6,
        bounds=(0.0, None),
        doc="Minimum eigenvalue clamp for local metric tensors.",
    )
    geometry_use_volume = param.Boolean(
        default=True,
        doc="Multiply edge kernels by destination Riemannian volume weights.",
    )
    geometry_correct_area = param.Boolean(
        default=True,
        doc="Use volume-weighted lineage crossing area instead of pure lineage counts.",
    )


FRACTAL_SET_CUT_TYPES = ("hyperplane", "spherical", "median")
FRACTAL_SET_GRAPH_CUT_TYPES = ("spectral_distance", "spectral_fitness", "spectral_both")
FRACTAL_SET_RANDOM_CUT_TYPE = "random_baseline"
FRACTAL_SET_METRICS_RAW = {
    "s_dist": "S_dist",
    "s_fit": "S_fit",
    "s_total": "S_total",
}
FRACTAL_SET_METRICS_GEOM = {
    "s_dist_geom": "S_dist_geom",
    "s_fit_geom": "S_fit_geom",
    "s_total_geom": "S_total_geom",
}


def _select_fractal_set_frames(
    n_transitions: int,
    warmup_fraction: float,
    frame_stride: int,
    max_frames: int | None,
) -> list[int]:
    """Select transition indices after warmup with optional subsampling."""
    if n_transitions <= 0:
        return []

    warmup = float(np.clip(warmup_fraction, 0.0, 0.95))
    start_idx = int(np.floor(warmup * n_transitions))
    start_idx = int(np.clip(start_idx, 0, max(n_transitions - 1, 0)))
    stride = max(1, int(frame_stride))
    frame_ids = list(range(start_idx, n_transitions, stride))
    if not frame_ids:
        return []

    if max_frames is not None and max_frames > 0 and len(frame_ids) > max_frames:
        pick = np.linspace(0, len(frame_ids) - 1, num=int(max_frames), dtype=int)
        frame_ids = [frame_ids[int(i)] for i in pick]
    return frame_ids


def _build_lineage_by_transition(
    companions_fit: np.ndarray,
    will_clone: np.ndarray,
    n_walkers: int,
) -> np.ndarray:
    """Build lineage labels per transition index from clone parent mapping."""
    n_transitions = int(min(companions_fit.shape[0], will_clone.shape[0]))
    lineage = np.zeros((n_transitions, n_walkers), dtype=np.int64)
    current = np.arange(n_walkers, dtype=np.int64)

    for info_idx in range(n_transitions):
        lineage[info_idx] = current
        parent = np.arange(n_walkers, dtype=np.int64)
        clone_mask = np.asarray(will_clone[info_idx], dtype=bool)
        clone_src = np.asarray(companions_fit[info_idx], dtype=np.int64)
        clone_src = np.clip(clone_src, 0, n_walkers - 1)
        parent[clone_mask] = clone_src[clone_mask]
        current = current[parent]

    return lineage


def _build_region_masks(
    positions: np.ndarray,
    axis: int,
    cut_type: str,
    n_cut_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build binary masks A for a given boundary geometry."""
    n_walkers = positions.shape[0]
    if n_walkers <= 1:
        return np.zeros((0, n_walkers), dtype=bool), np.zeros(0, dtype=float)

    safe_axis = int(np.clip(axis, 0, max(positions.shape[1] - 1, 0)))
    coord = positions[:, safe_axis]
    sample_count = int(max(2, n_cut_samples))

    if cut_type == "hyperplane":
        low, high = float(coord.min()), float(coord.max())
        if np.isclose(low, high):
            cuts = np.array([low], dtype=float)
        else:
            cuts = np.linspace(low, high, num=sample_count, dtype=float)
        masks = coord[None, :] > cuts[:, None]
    elif cut_type == "spherical":
        center = positions.mean(axis=0, keepdims=True)
        radii = np.linalg.norm(positions - center, axis=1)
        max_radius = float(radii.max())
        if np.isclose(max_radius, 0.0):
            cuts = np.array([0.0], dtype=float)
        else:
            cuts = np.linspace(0.0, max_radius, num=sample_count, dtype=float)
        masks = radii[None, :] <= cuts[:, None]
    elif cut_type == "median":
        order = np.argsort(coord, kind="mergesort")
        mask = np.zeros(n_walkers, dtype=bool)
        mask[order[n_walkers // 2 :]] = True
        cuts = np.array([float(np.median(coord))], dtype=float)
        masks = mask[None, :]
    else:
        msg = f"Unsupported cut type: {cut_type}"
        raise ValueError(msg)

    valid = np.logical_and(masks.any(axis=1), (~masks).any(axis=1))
    return masks[valid], cuts[valid]


def _build_spectral_sweep_masks(
    companions: np.ndarray,
    n_cut_samples: int,
    min_partition_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build graph-theoretic cuts using a Fiedler-vector sweep."""
    companions_idx = np.asarray(companions, dtype=np.int64)
    if companions_idx.ndim == 1:
        companions_idx = companions_idx[None, :]
    if companions_idx.ndim != 2:
        return np.zeros((0, 0), dtype=bool), np.zeros(0, dtype=float)

    n_walkers = int(companions_idx.shape[1])
    if n_walkers <= 2:
        return np.zeros((0, n_walkers), dtype=bool), np.zeros(0, dtype=float)

    companions_idx = np.clip(companions_idx, 0, n_walkers - 1)
    adjacency = np.zeros((n_walkers, n_walkers), dtype=float)
    walker = np.arange(n_walkers, dtype=np.int64)
    for row in companions_idx:
        adjacency[walker, row] += 1.0
        adjacency[row, walker] += 1.0
    np.fill_diagonal(adjacency, 0.0)

    degree = adjacency.sum(axis=1)
    if np.allclose(degree, 0.0):
        return np.zeros((0, n_walkers), dtype=bool), np.zeros(0, dtype=float)

    # Symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
    inv_sqrt_degree = np.zeros_like(degree)
    positive = degree > 1e-12
    inv_sqrt_degree[positive] = 1.0 / np.sqrt(degree[positive])
    laplacian = np.eye(n_walkers, dtype=float) - (
        inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    )

    try:
        eigvals, eigvecs = np.linalg.eigh(laplacian)
        if eigvecs.shape[1] < 2:
            return np.zeros((0, n_walkers), dtype=bool), np.zeros(0, dtype=float)
        # First eigenvector is near-constant; second gives Fiedler direction.
        fiedler = eigvecs[:, 1]
        if not np.isfinite(eigvals[1]):
            return np.zeros((0, n_walkers), dtype=bool), np.zeros(0, dtype=float)
    except np.linalg.LinAlgError:
        return np.zeros((0, n_walkers), dtype=bool), np.zeros(0, dtype=float)

    order = np.argsort(fiedler, kind="mergesort")
    min_size = int(np.clip(min_partition_size, 1, max(1, n_walkers // 2)))
    k_values = np.linspace(
        min_size,
        n_walkers - min_size,
        num=max(2, int(n_cut_samples)),
        dtype=int,
    )
    k_values = np.unique(k_values)
    k_values = k_values[(k_values > 0) & (k_values < n_walkers)]
    if k_values.size == 0:
        return np.zeros((0, n_walkers), dtype=bool), np.zeros(0, dtype=float)

    masks = np.zeros((k_values.size, n_walkers), dtype=bool)
    for idx, k in enumerate(k_values.tolist()):
        masks[idx, order[: int(k)]] = True

    cuts = k_values.astype(float) / float(n_walkers)
    valid = np.logical_and(masks.any(axis=1), (~masks).any(axis=1))
    return masks[valid], cuts[valid]


def _build_random_partition_masks(
    n_walkers: int,
    n_partitions: int,
    min_partition_size: int,
    balanced: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build random partition baseline masks."""
    if n_walkers <= 2 or n_partitions <= 0:
        return np.zeros((0, n_walkers), dtype=bool), np.zeros(0, dtype=float)

    min_size = int(np.clip(min_partition_size, 1, max(1, n_walkers // 2)))
    masks = np.zeros((int(n_partitions), n_walkers), dtype=bool)
    sizes = np.zeros(int(n_partitions), dtype=int)
    for idx in range(int(n_partitions)):
        if balanced:
            k = n_walkers // 2
            k = int(np.clip(k, min_size, n_walkers - min_size))
        else:
            low = min_size
            high = n_walkers - min_size
            if low >= high:
                k = n_walkers // 2
            else:
                k = int(rng.integers(low=low, high=high + 1))
        selected = rng.choice(n_walkers, size=k, replace=False)
        masks[idx, selected] = True
        sizes[idx] = k

    valid = np.logical_and(masks.any(axis=1), (~masks).any(axis=1))
    cuts = sizes.astype(float) / float(n_walkers)
    return masks[valid], cuts[valid]


def _compute_companion_geometric_weights(
    positions: np.ndarray,
    companions: np.ndarray,
    metric_tensors: np.ndarray | None,
    volume_weights: np.ndarray | None,
    length_scale: float,
    min_eig: float,
    use_volume: bool,
    pbc: bool,
    bounds_low: np.ndarray | None,
    bounds_high: np.ndarray | None,
) -> np.ndarray:
    """Compute unnormalized Riemannian-kernel-volume edge weights for companion edges."""
    n_walkers = int(positions.shape[0])
    if n_walkers == 0:
        return np.zeros(0, dtype=float)

    comp_idx = np.asarray(companions, dtype=np.int64)
    comp_idx = np.clip(comp_idx, 0, n_walkers - 1)
    delta = positions[comp_idx] - positions
    if pbc and bounds_low is not None and bounds_high is not None:
        span = bounds_high - bounds_low
        safe_span = np.where(np.abs(span) > 1e-12, span, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = delta - span * np.round(delta / safe_span)

    d = int(positions.shape[1])
    metric_arr = None if metric_tensors is None else np.asarray(metric_tensors, dtype=float)
    if (
        metric_arr is None
        or metric_arr.ndim != 3
        or metric_arr.shape != (n_walkers, d, d)
        or not np.all(np.isfinite(metric_arr))
    ):
        g = np.broadcast_to(np.eye(d, dtype=float), (n_walkers, d, d)).copy()
    else:
        g = metric_arr

    try:
        eigvals, eigvecs = np.linalg.eigh(g)
        eigvals = np.clip(eigvals, float(min_eig), None)
        g = np.einsum("nik,nk,njk->nij", eigvecs, eigvals, eigvecs)
    except (np.linalg.LinAlgError, ValueError):
        g = np.broadcast_to(np.eye(d, dtype=float), (n_walkers, d, d)).copy()

    g_edge = 0.5 * (g + g[comp_idx])
    d_sq = np.einsum("ni,nij,nj->n", delta, g_edge, delta)
    d_sq = np.clip(d_sq, 0.0, None)
    l = float(max(length_scale, 1e-6))
    kernel = np.exp(-d_sq / (2.0 * l * l))

    if use_volume and volume_weights is not None and volume_weights.shape[0] == n_walkers:
        vol = np.asarray(volume_weights, dtype=float)
        vol = np.clip(vol, 1e-12, None)
        kernel = kernel * vol[comp_idx]
    return kernel


def _count_cross_boundary(masks: np.ndarray, companions: np.ndarray) -> np.ndarray:
    """Count cross-boundary companion edges for each mask in a sweep."""
    if masks.size == 0:
        return np.zeros(0, dtype=np.int64)
    companion_idx = np.asarray(companions, dtype=np.int64)
    companion_idx = np.clip(companion_idx, 0, masks.shape[1] - 1)
    return np.sum(masks ^ masks[:, companion_idx], axis=1, dtype=np.int64)


def _count_crossing_lineages(masks: np.ndarray, lineage_ids: np.ndarray) -> np.ndarray:
    """Count lineages with descendants on both sides of each boundary mask."""
    if masks.size == 0:
        return np.zeros(0, dtype=np.int64)

    labels = np.asarray(lineage_ids, dtype=np.int64)
    areas = np.zeros(masks.shape[0], dtype=np.int64)
    for idx, mask in enumerate(masks):
        lineages_a = np.unique(labels[mask])
        lineages_ac = np.unique(labels[~mask])
        areas[idx] = np.intersect1d(lineages_a, lineages_ac, assume_unique=True).size
    return areas


def _sum_cross_boundary_weights(
    masks: np.ndarray,
    companions: np.ndarray,
    edge_weights: np.ndarray,
) -> np.ndarray:
    """Sum weighted companion edges crossing the boundary for each mask."""
    if masks.size == 0:
        return np.zeros(0, dtype=float)
    comp_idx = np.asarray(companions, dtype=np.int64)
    comp_idx = np.clip(comp_idx, 0, masks.shape[1] - 1)
    cross = masks ^ masks[:, comp_idx]
    return np.sum(cross * edge_weights[None, :], axis=1, dtype=float)


def _count_crossing_lineages_weighted(
    masks: np.ndarray,
    lineage_ids: np.ndarray,
    volume_weights: np.ndarray | None,
    use_weighted_area: bool,
) -> np.ndarray:
    """Count/weight crossing lineages for each mask."""
    if masks.size == 0:
        return np.zeros(0, dtype=float)
    if not use_weighted_area or volume_weights is None:
        return _count_crossing_lineages(masks, lineage_ids).astype(float)

    labels = np.asarray(lineage_ids, dtype=np.int64)
    vols = np.asarray(volume_weights, dtype=float)
    if vols.shape[0] != labels.shape[0]:
        return _count_crossing_lineages(masks, lineage_ids).astype(float)
    vols = np.clip(vols, 1e-12, None)

    values = np.zeros(masks.shape[0], dtype=float)
    for idx, mask in enumerate(masks):
        lineages_a = np.unique(labels[mask])
        lineages_ac = np.unique(labels[~mask])
        crossing = np.intersect1d(lineages_a, lineages_ac, assume_unique=True)
        if crossing.size == 0:
            continue
        total = 0.0
        for lineage in crossing:
            in_a = np.logical_and(mask, labels == lineage)
            in_b = np.logical_and(~mask, labels == lineage)
            if not in_a.any() or not in_b.any():
                continue
            va = float(np.mean(vols[in_a]))
            vb = float(np.mean(vols[in_b]))
            total += np.sqrt(max(va, 1e-12) * max(vb, 1e-12))
        values[idx] = total
    return values


def _fit_linear_relation(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = alpha*x + b and return (alpha, b, R2)."""
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.allclose(x_arr, x_arr[0]):
        return float("nan"), float("nan"), float("nan")

    slope, intercept = np.polyfit(x_arr, y_arr, deg=1)
    y_hat = slope * x_arr + intercept
    ss_res = float(np.sum((y_arr - y_hat) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    return float(slope), float(intercept), r2


def _select_fractal_set_metric_map(settings: FractalSetSettings) -> dict[str, str]:
    """Select which metric family to regressed/plotted."""
    include_geom = bool(settings.use_geometry_correction)
    mode = str(settings.metric_display)

    metric_map: dict[str, str] = {}
    if mode in {"raw", "both"} or not include_geom:
        metric_map.update(FRACTAL_SET_METRICS_RAW)
    if include_geom and mode in {"geometry", "both"}:
        metric_map.update(FRACTAL_SET_METRICS_GEOM)
    if not metric_map:
        metric_map.update(FRACTAL_SET_METRICS_RAW)
    return metric_map


def _compute_fractal_set_measurements(
    history: RunHistory,
    settings: FractalSetSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute IG/CST boundary measurements from a RunHistory."""
    companions_dist = _to_numpy(history.companions_distance).astype(np.int64, copy=False)
    companions_fit = _to_numpy(history.companions_clone).astype(np.int64, copy=False)
    will_clone = _to_numpy(history.will_clone).astype(bool, copy=False)
    x_pre = _to_numpy(history.x_before_clone)
    volume_series = (
        _to_numpy(history.riemannian_volume_weights)
        if getattr(history, "riemannian_volume_weights", None) is not None
        else None
    )
    metric_series = (
        _to_numpy(history.diffusion_tensors_full)
        if getattr(history, "diffusion_tensors_full", None) is not None
        else None
    )

    n_transitions = int(
        min(
            companions_dist.shape[0],
            companions_fit.shape[0],
            will_clone.shape[0],
            max(0, x_pre.shape[0] - 1),
        )
    )
    n_walkers = int(history.N)

    if n_transitions <= 0 or n_walkers <= 1:
        empty = pd.DataFrame()
        return empty, empty, empty

    lineage_by_transition = _build_lineage_by_transition(
        companions_fit[:n_transitions],
        will_clone[:n_transitions],
        n_walkers,
    )
    frame_ids = _select_fractal_set_frames(
        n_transitions=n_transitions,
        warmup_fraction=float(settings.warmup_fraction),
        frame_stride=int(settings.frame_stride),
        max_frames=int(settings.max_frames) if settings.max_frames is not None else None,
    )
    if not frame_ids:
        empty = pd.DataFrame()
        return empty, empty, empty

    kernel_length = settings.geometry_kernel_length_scale
    if kernel_length is None:
        params = history.params if isinstance(history.params, dict) else {}
        kinetic = params.get("kinetic", {}) if isinstance(params.get("kinetic", {}), dict) else {}
        kernel_length = kinetic.get("viscous_length_scale", 1.0)
    try:
        kernel_length = float(max(float(kernel_length), 1e-6))
    except (TypeError, ValueError):
        kernel_length = 1.0

    use_geom = bool(settings.use_geometry_correction)
    pbc = False  # PBC no longer stored on RunHistory
    bounds_low = None
    bounds_high = None

    include_spatial = settings.partition_family in {"all", "spatial"}
    include_graph = settings.partition_family in {"all", "graph"}
    include_random = settings.partition_family in {"all", "random"}

    if include_spatial:
        if settings.cut_geometry == "all":
            spatial_cut_types = FRACTAL_SET_CUT_TYPES
        else:
            spatial_cut_types = (str(settings.cut_geometry),)
    else:
        spatial_cut_types = ()

    graph_specs: list[tuple[str, str]] = []
    if include_graph:
        graph_source = str(settings.graph_cut_source)
        if graph_source == "all":
            graph_specs = [
                ("spectral_distance", "distance"),
                ("spectral_fitness", "fitness"),
                ("spectral_both", "both"),
            ]
        elif graph_source == "distance":
            graph_specs = [("spectral_distance", "distance")]
        elif graph_source == "fitness":
            graph_specs = [("spectral_fitness", "fitness")]
        elif graph_source == "both":
            graph_specs = [("spectral_both", "both")]

    rng = np.random.default_rng(int(settings.random_seed))

    rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for info_idx in frame_ids:
        positions = np.asarray(x_pre[info_idx + 1], dtype=float)
        if positions.ndim != 2 or positions.shape[0] != n_walkers:
            continue
        axis = int(np.clip(settings.partition_axis, 0, max(positions.shape[1] - 1, 0)))
        labels = lineage_by_transition[info_idx]
        step = int(history.recorded_steps[min(info_idx + 1, len(history.recorded_steps) - 1)])
        vol_frame = None
        if volume_series is not None and info_idx < volume_series.shape[0]:
            vol_frame = np.asarray(volume_series[info_idx], dtype=float)
        metric_frame = None
        if metric_series is not None and info_idx < metric_series.shape[0]:
            metric_frame = np.asarray(metric_series[info_idx], dtype=float)

        if use_geom:
            w_dist = _compute_companion_geometric_weights(
                positions=positions,
                companions=companions_dist[info_idx],
                metric_tensors=metric_frame,
                volume_weights=vol_frame,
                length_scale=kernel_length,
                min_eig=float(settings.geometry_min_eig),
                use_volume=bool(settings.geometry_use_volume),
                pbc=pbc,
                bounds_low=bounds_low,
                bounds_high=bounds_high,
            )
            w_fit = _compute_companion_geometric_weights(
                positions=positions,
                companions=companions_fit[info_idx],
                metric_tensors=metric_frame,
                volume_weights=vol_frame,
                length_scale=kernel_length,
                min_eig=float(settings.geometry_min_eig),
                use_volume=bool(settings.geometry_use_volume),
                pbc=pbc,
                bounds_low=bounds_low,
                bounds_high=bounds_high,
            )
        else:
            w_dist = np.ones(n_walkers, dtype=float)
            w_fit = np.ones(n_walkers, dtype=float)

        def _append_partition_measurements(
            masks: np.ndarray,
            cut_values: np.ndarray,
            cut_type: str,
            partition_family: str,
        ) -> None:
            if masks.size == 0:
                return
            s_dist = _count_cross_boundary(masks, companions_dist[info_idx])
            s_fit = _count_cross_boundary(masks, companions_fit[info_idx])
            s_total = s_dist + s_fit
            area_cst = _count_crossing_lineages(masks, labels)
            s_dist_geom = _sum_cross_boundary_weights(
                masks,
                companions_dist[info_idx],
                w_dist,
            )
            s_fit_geom = _sum_cross_boundary_weights(masks, companions_fit[info_idx], w_fit)
            s_total_geom = s_dist_geom + s_fit_geom
            area_cst_geom = _count_crossing_lineages_weighted(
                masks=masks,
                lineage_ids=labels,
                volume_weights=vol_frame,
                use_weighted_area=bool(settings.geometry_correct_area and use_geom),
            )
            region_size = masks.sum(axis=1, dtype=np.int64)

            for local_idx, cut_value in enumerate(cut_values.tolist()):
                rows.append({
                    "info_idx": int(info_idx),
                    "recorded_step": step,
                    "partition_family": partition_family,
                    "cut_type": cut_type,
                    "cut_value": float(cut_value),
                    "region_size": int(region_size[local_idx]),
                    "area_cst": int(area_cst[local_idx]),
                    "area_cst_geom": float(area_cst_geom[local_idx]),
                    "s_dist": int(s_dist[local_idx]),
                    "s_fit": int(s_fit[local_idx]),
                    "s_total": int(s_total[local_idx]),
                    "s_dist_geom": float(s_dist_geom[local_idx]),
                    "s_fit_geom": float(s_fit_geom[local_idx]),
                    "s_total_geom": float(s_total_geom[local_idx]),
                })

            frame_rows.append({
                "info_idx": int(info_idx),
                "recorded_step": step,
                "partition_family": partition_family,
                "cut_type": cut_type,
                "n_partitions": int(masks.shape[0]),
                "mean_area_cst": float(np.mean(area_cst)),
                "mean_area_cst_geom": float(np.mean(area_cst_geom)),
                "mean_s_dist": float(np.mean(s_dist)),
                "mean_s_fit": float(np.mean(s_fit)),
                "mean_s_total": float(np.mean(s_total)),
                "mean_s_dist_geom": float(np.mean(s_dist_geom)),
                "mean_s_fit_geom": float(np.mean(s_fit_geom)),
                "mean_s_total_geom": float(np.mean(s_total_geom)),
            })

        for cut_type in spatial_cut_types:
            masks, cut_values = _build_region_masks(
                positions=positions,
                axis=axis,
                cut_type=cut_type,
                n_cut_samples=int(settings.n_cut_samples),
            )
            _append_partition_measurements(
                masks=masks,
                cut_values=cut_values,
                cut_type=cut_type,
                partition_family="spatial",
            )

        for cut_type, source in graph_specs:
            if source == "distance":
                companions_for_graph = companions_dist[info_idx]
            elif source == "fitness":
                companions_for_graph = companions_fit[info_idx]
            else:
                companions_for_graph = np.stack(
                    [companions_dist[info_idx], companions_fit[info_idx]],
                    axis=0,
                )
            masks, cut_values = _build_spectral_sweep_masks(
                companions=companions_for_graph,
                n_cut_samples=int(settings.n_cut_samples),
                min_partition_size=int(settings.min_partition_size),
            )
            _append_partition_measurements(
                masks=masks,
                cut_values=cut_values,
                cut_type=cut_type,
                partition_family="graph",
            )

        if include_random:
            masks, cut_values = _build_random_partition_masks(
                n_walkers=n_walkers,
                n_partitions=int(settings.random_partitions),
                min_partition_size=int(settings.min_partition_size),
                balanced=bool(settings.random_balanced),
                rng=rng,
            )
            _append_partition_measurements(
                masks=masks,
                cut_values=cut_values,
                cut_type=FRACTAL_SET_RANDOM_CUT_TYPE,
                partition_family="random",
            )

    points_df = pd.DataFrame(rows)
    frame_df = pd.DataFrame(frame_rows)
    if points_df.empty:
        empty = pd.DataFrame()
        return points_df, empty, frame_df

    metric_map = _select_fractal_set_metric_map(settings)
    regression_rows: list[dict[str, Any]] = []
    for metric_key, metric_label in metric_map.items():
        area_key = "area_cst_geom" if metric_key.endswith("_geom") else "area_cst"
        x_all = points_df[area_key].to_numpy(dtype=float)
        y_all = points_df[metric_key].to_numpy(dtype=float)
        slope, intercept, r2 = _fit_linear_relation(x_all, y_all)
        regression_rows.append({
            "metric": metric_key,
            "metric_label": metric_label,
            "area_key": area_key,
            "cut_type": "all",
            "partition_family": "all",
            "n_points": int(x_all.size),
            "slope_alpha": slope,
            "intercept": intercept,
            "r2": r2,
        })

        for cut_type in sorted(points_df["cut_type"].unique()):
            subset = points_df[points_df["cut_type"] == cut_type]
            x = subset[area_key].to_numpy(dtype=float)
            y = subset[metric_key].to_numpy(dtype=float)
            slope, intercept, r2 = _fit_linear_relation(x, y)
            families = subset["partition_family"].unique().tolist()
            family = str(families[0]) if len(families) == 1 else "mixed"
            regression_rows.append({
                "metric": metric_key,
                "metric_label": metric_label,
                "area_key": area_key,
                "cut_type": str(cut_type),
                "partition_family": family,
                "n_points": int(x.size),
                "slope_alpha": slope,
                "intercept": intercept,
                "r2": r2,
            })

        for family in sorted(points_df["partition_family"].unique()):
            subset = points_df[points_df["partition_family"] == family]
            x = subset[area_key].to_numpy(dtype=float)
            y = subset[metric_key].to_numpy(dtype=float)
            slope, intercept, r2 = _fit_linear_relation(x, y)
            regression_rows.append({
                "metric": metric_key,
                "metric_label": metric_label,
                "area_key": area_key,
                "cut_type": f"family:{family}",
                "partition_family": str(family),
                "n_points": int(x.size),
                "slope_alpha": slope,
                "intercept": intercept,
                "r2": r2,
            })

    regression_df = pd.DataFrame(regression_rows)
    return points_df, regression_df, frame_df


def _build_fractal_set_scatter_plot(
    points_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    metric_key: str,
    title: str,
) -> Any:
    """Build scatter + linear fits for S_IG metric versus Area_CST."""
    if points_df.empty:
        return hv.Text(0, 0, "No Fractal Set measurements")

    colors = {
        "hyperplane": "#4c78a8",
        "spherical": "#f58518",
        "median": "#54a24b",
        "spectral_distance": "#b279a2",
        "spectral_fitness": "#e45756",
        "spectral_both": "#72b7b2",
        FRACTAL_SET_RANDOM_CUT_TYPE: "#7f7f7f",
    }
    overlays: list[Any] = []

    cut_types = sorted(points_df["cut_type"].unique().tolist())
    for cut_type in cut_types:
        subset = points_df[points_df["cut_type"] == cut_type]
        if subset.empty:
            continue
        x_key = "area_cst_geom" if metric_key.endswith("_geom") else "area_cst"

        label = str(cut_type).replace("_", " ").title()
        color = colors.get(cut_type, "#4c78a8")
        overlays.append(
            hv.Scatter(
                subset,
                kdims=[x_key],
                vdims=[
                    metric_key,
                    "recorded_step",
                    "cut_value",
                    "region_size",
                    "partition_family",
                ],
                label=label,
            ).opts(
                color=color,
                alpha=0.55,
                size=5,
                tools=["hover"],
                marker="circle",
            )
        )

        fit_rows = regression_df[
            (regression_df["metric"] == metric_key) & (regression_df["cut_type"] == cut_type)
        ]
        if fit_rows.empty:
            continue
        fit = fit_rows.iloc[0]
        slope = float(fit.get("slope_alpha", float("nan")))
        intercept = float(fit.get("intercept", float("nan")))
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            continue

        x_min = float(subset[x_key].min())
        x_max = float(subset[x_key].max())
        if np.isclose(x_min, x_max):
            continue
        x_line = np.linspace(x_min, x_max, num=80)
        y_line = slope * x_line + intercept
        overlays.append(
            hv.Curve((x_line, y_line), label=f"{label} fit").opts(
                color=color,
                line_width=2,
                line_dash="dashed",
            )
        )

    if not overlays:
        return hv.Text(0, 0, "No valid Fractal Set data")

    y_label = (
        FRACTAL_SET_METRICS_RAW.get(metric_key)
        or FRACTAL_SET_METRICS_GEOM.get(metric_key)
        or metric_key
    )
    x_label = "Area_CST_geom(A)" if metric_key.endswith("_geom") else "Area_CST(A)"
    return hv.Overlay(overlays).opts(
        title=title,
        xlabel=x_label,
        ylabel=y_label,
        width=900,
        height=360,
        legend_position="top_left",
        show_grid=True,
        toolbar="above",
    )


def _format_fractal_set_summary(
    points_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    frame_df: pd.DataFrame,
) -> str:
    """Summarize Fractal Set boundary measurements and area-law fits."""
    if points_df.empty:
        return (
            "## Fractal Set Summary\n_No valid measurements. Load a RunHistory and run compute._"
        )

    lines = [
        "## Fractal Set Summary",
        f"- Transitions analyzed: {int(points_df['info_idx'].nunique())}",
        f"- Boundary samples: {len(points_df)}",
        f"- Geometries: {', '.join(sorted(points_df['cut_type'].unique()))}",
        f"- Partition families: {', '.join(sorted(points_df['partition_family'].unique()))}",
    ]

    all_rows = regression_df[regression_df["cut_type"] == "all"].copy()
    for _, fit in all_rows.sort_values("metric").iterrows():
        metric_key = str(fit.get("metric", ""))
        area_key = str(fit.get("area_key", "area_cst"))
        area_label = "Area_CST_geom" if area_key == "area_cst_geom" else "Area_CST"
        metric_label = str(
            fit.get("metric_label")
            or FRACTAL_SET_METRICS_RAW.get(metric_key)
            or FRACTAL_SET_METRICS_GEOM.get(metric_key)
            or metric_key
        )
        slope = float(fit.get("slope_alpha", float("nan")))
        r2 = float(fit.get("r2", float("nan")))
        n_points = int(fit.get("n_points", 0))
        if np.isfinite(slope):
            lines.append(
                f"- {metric_label} vs {area_label}: alpha={slope:.6f}, R2={r2:.4f}, n={n_points}"
            )
        else:
            lines.append(
                f"- {metric_label} vs {area_label}: insufficient area variation for fit (n={n_points})"
            )
        baseline = regression_df[
            (regression_df["metric"] == metric_key)
            & (regression_df["cut_type"] == FRACTAL_SET_RANDOM_CUT_TYPE)
        ]
        if not baseline.empty:
            b = baseline.iloc[0]
            b_alpha = float(b.get("slope_alpha", float("nan")))
            b_r2 = float(b.get("r2", float("nan")))
            if np.isfinite(b_alpha) and np.isfinite(b_r2):
                lines.append(f"  random baseline: alpha={b_alpha:.6f}, R2={b_r2:.4f}")

    if not frame_df.empty:
        lines.append(f"- Mean per-frame Area_CST: {float(frame_df['mean_area_cst'].mean()):.3f}")
        if "mean_area_cst_geom" in frame_df:
            lines.append(
                f"- Mean per-frame Area_CST_geom: "
                f"{float(frame_df['mean_area_cst_geom'].mean()):.3f}"
            )
        lines.append(f"- Mean per-frame S_total: {float(frame_df['mean_s_total'].mean()):.3f}")
        if "mean_s_total_geom" in frame_df:
            lines.append(
                f"- Mean per-frame S_total_geom: {float(frame_df['mean_s_total_geom'].mean()):.3f}"
            )

    lines.append("")
    lines.append(
        "N-scaling and geodesic/flat ablation are cross-run tests and should be done across "
        "multiple histories."
    )
    return "\n".join(lines)


def _build_fractal_set_baseline_comparison(regression_df: pd.DataFrame) -> pd.DataFrame:
    """Compare each cut family against the random partition baseline."""
    if regression_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    metric_keys = sorted(regression_df["metric"].dropna().unique().tolist())
    for metric_key in metric_keys:
        metric_df = regression_df[regression_df["metric"] == metric_key]
        if metric_df.empty:
            continue
        metric_label = str(
            metric_df.iloc[0].get("metric_label")
            or FRACTAL_SET_METRICS_RAW.get(metric_key)
            or FRACTAL_SET_METRICS_GEOM.get(metric_key)
            or metric_key
        )
        baseline_row = metric_df[metric_df["cut_type"] == FRACTAL_SET_RANDOM_CUT_TYPE]
        baseline_r2 = float("nan")
        baseline_alpha = float("nan")
        if not baseline_row.empty:
            baseline_r2 = float(baseline_row.iloc[0]["r2"])
            baseline_alpha = float(baseline_row.iloc[0]["slope_alpha"])

        for _, fit in metric_df.iterrows():
            cut_type = str(fit.get("cut_type", ""))
            if cut_type in {"all", FRACTAL_SET_RANDOM_CUT_TYPE}:
                continue
            if cut_type.startswith("family:"):
                continue
            r2 = float(fit.get("r2", float("nan")))
            alpha = float(fit.get("slope_alpha", float("nan")))
            rows.append({
                "metric": metric_label,
                "cut_type": cut_type,
                "partition_family": str(fit.get("partition_family", "")),
                "n_points": int(fit.get("n_points", 0)),
                "alpha": alpha,
                "r2": r2,
                "baseline_alpha_random": baseline_alpha,
                "baseline_r2_random": baseline_r2,
                "delta_r2_vs_random": (
                    r2 - baseline_r2
                    if np.isfinite(r2) and np.isfinite(baseline_r2)
                    else float("nan")
                ),
                "delta_alpha_vs_random": (
                    alpha - baseline_alpha
                    if np.isfinite(alpha) and np.isfinite(baseline_alpha)
                    else float("nan")
                ),
            })

    if not rows:
        return pd.DataFrame()
    comparison = pd.DataFrame(rows)
    return comparison.sort_values(["metric", "delta_r2_vs_random"], ascending=[True, False])


def _resolve_electroweak_geodesic_matrices(
    history: RunHistory | None,
    frame_indices: list[int] | None,
    state: dict[str, Any],
    *,
    method: str,
    edge_weight_mode: str,
    assume_all_alive: bool,
) -> dict[int, torch.Tensor] | None:
    if not isinstance(state, dict):
        return None

    requested_frames: list[int] = []
    for frame in frame_indices or []:
        try:
            frame_idx = int(frame)
        except (TypeError, ValueError):
            continue
        if frame_idx not in requested_frames:
            requested_frames.append(frame_idx)
    if not requested_frames:
        return None

    merged: dict[int, torch.Tensor] = {}
    cached = state.get("_multiscale_geodesic_distance_by_frame")
    if isinstance(cached, dict):
        for raw_key, value in cached.items():
            try:
                frame = int(raw_key)
            except (TypeError, ValueError):
                continue
            if torch.is_tensor(value):
                merged[frame] = value.detach().to(dtype=torch.float32, device="cpu")

    missing_frames = [frame for frame in requested_frames if frame not in merged]
    if not missing_frames or history is None:
        return merged or None

    try:
        frame_ids, distance_batch = compute_pairwise_distance_matrices_from_history(
            history,
            method=method,
            frame_indices=missing_frames,
            batch_size=1,
            edge_weight_mode=edge_weight_mode,
            assume_all_alive=bool(assume_all_alive),
            device=None,
            dtype=torch.float32,
        )
    except Exception:
        return merged or None

    for local_idx, frame_id in enumerate(frame_ids):
        if local_idx >= int(distance_batch.shape[0]):
            break
        matrix = distance_batch[local_idx]
        if torch.is_tensor(matrix):
            merged[int(frame_id)] = matrix.detach().to(dtype=torch.float32, device="cpu")

    if merged:
        state["_multiscale_geodesic_distance_by_frame"] = merged
        return merged
    return None


def _build_geodesic_distance_distribution_by_frame(
    pairwise_distance_by_frame: dict[int, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(pairwise_distance_by_frame, dict):
        return None

    per_frame_counts: list[int] = []
    distance_samples: list[np.ndarray] = []
    for frame_id, matrix in pairwise_distance_by_frame.items():
        try:
            int(frame_id)
        except (TypeError, ValueError):
            continue
        if not torch.is_tensor(matrix):
            continue
        arr = matrix.detach().to(dtype=torch.float32, device="cpu").numpy()
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            continue
        n = int(arr.shape[0])
        if n < 2:
            continue
        distances = arr[np.triu_indices(n, k=1)]
        finite_mask = np.isfinite(distances) & (distances > 0)
        distances = distances[finite_mask]
        if distances.size == 0:
            continue
        distance_samples.append(np.asarray(distances, dtype=float))
        per_frame_counts.append(int(distances.size))

    if not distance_samples:
        return {
            "n_geodesic_samples": 0,
            "n_frames_with_samples": 0,
        }

    all_distances = np.concatenate(distance_samples).astype(float)
    log_distances = np.log10(np.maximum(all_distances, np.finfo(float).tiny))
    hist_freq, hist_edges = np.histogram(log_distances, bins=60)

    return {
        "n_geodesic_samples": int(all_distances.size),
        "n_frames_with_samples": len(per_frame_counts),
        "log10_min": float(np.min(log_distances)),
        "log10_max": float(np.max(log_distances)),
        "log10_hist": hist_freq.astype(float).tolist(),
        "log10_edges": hist_edges.astype(float).tolist(),
        "frame_counts": per_frame_counts,
    }


def _build_geodesic_distribution_plot(
    distribution: dict[str, Any] | None,
    *,
    width: int = 900,
    height: int = 260,
) -> hv.Element:
    """Build a HoloViews histogram pane for precomputed geodesic samples."""
    if not isinstance(distribution, dict):
        return hv.Text(0, 0, "No geodesic pairwise distribution available.").opts(
            title="Geodesic Pairwise Distance Distribution"
        )
    n_samples = int(distribution.get("n_geodesic_samples", 0) or 0)
    if n_samples <= 0:
        return hv.Text(0, 0, "No finite geodesic pairs available.").opts(
            title="Geodesic Pairwise Distance Distribution"
        )

    hist = np.asarray(distribution.get("log10_hist", []), dtype=float)
    edges = np.asarray(distribution.get("log10_edges", []), dtype=float)
    if hist.size == 0 or edges.size != hist.size + 1:
        return hv.Text(0, 0, "Malformed geodesic distribution data.").opts(
            title="Geodesic Pairwise Distance Distribution"
        )

    return hv.Histogram((hist, edges), kdims=["log10_geodesic_distance"], vdims=["count"]).opts(
        width=width,
        height=height,
        xlabel="log10(geodesic distance)",
        ylabel="Pair count",
        title=(
            "Geodesic Pairwise Distance Distribution "
            f"(frames={int(distribution.get('n_frames_with_samples', 0) or 0)}, "
            f"samples={n_samples})"
        ),
        color="#4c78a8",
        line_color="white",
        show_grid=True,
    )


def build_holographic_principle_tab(
    *,
    state: dict[str, Any],
    run_tab_computation: Callable[
        [dict[str, Any], pn.pane.Markdown, str, Callable[[RunHistory], None]], None
    ],
    new_dirac_ew_settings: Any = None,
) -> HolographicPrincipleSection:
    """Build Holographic Principle tab with callbacks."""

    fractal_set_settings = FractalSetSettings()
    fractal_set_status = pn.pane.Markdown(
        "**Fractal Set:** Load a RunHistory and click Compute Fractal Set.",
        sizing_mode="stretch_width",
    )
    fractal_set_run_button = pn.widgets.Button(
        name="Compute Fractal Set",
        button_type="primary",
        min_width=240,
        sizing_mode="stretch_width",
        disabled=True,
    )
    fractal_set_settings_panel = pn.Param(
        fractal_set_settings,
        parameters=[
            "warmup_fraction",
            "frame_stride",
            "max_frames",
            "partition_family",
            "n_cut_samples",
            "partition_axis",
            "cut_geometry",
            "graph_cut_source",
            "min_partition_size",
            "random_partitions",
            "random_balanced",
            "random_seed",
            "use_geometry_correction",
            "metric_display",
            "geometry_kernel_length_scale",
            "geometry_min_eig",
            "geometry_use_volume",
            "geometry_correct_area",
        ],
        show_name=False,
        widgets={
            "partition_family": {
                "type": pn.widgets.Select,
                "name": "Partition family",
            },
            "cut_geometry": {
                "type": pn.widgets.Select,
                "name": "Boundary geometry",
            },
            "graph_cut_source": {
                "type": pn.widgets.Select,
                "name": "Graph cut source",
            },
            "partition_axis": {
                "name": "Partition axis",
            },
            "metric_display": {
                "type": pn.widgets.Select,
                "name": "Metric display",
            },
            "geometry_kernel_length_scale": {
                "name": "Geometry kernel length scale",
            },
            "geometry_min_eig": {
                "name": "Geometry min eigenvalue",
            },
            "geometry_use_volume": {
                "name": "Use volume in edge weights",
            },
            "geometry_correct_area": {
                "name": "Use volume-weighted CST area",
            },
        },
        default_layout=type("FractalSetSettingsGrid", (pn.GridBox,), {"ncols": 2}),
    )
    fractal_set_summary = pn.pane.Markdown(
        "## Fractal Set Summary\n_Compute Fractal Set to populate._",
        sizing_mode="stretch_width",
    )
    fractal_set_plot_dist = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    fractal_set_plot_fit = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    fractal_set_plot_total = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    fractal_set_plot_dist_geom = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    fractal_set_plot_fit_geom = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    fractal_set_plot_total_geom = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    fractal_set_geodesic_distribution_plot = pn.pane.HoloViews(
        sizing_mode="stretch_width",
        linked_axes=False,
    )
    einstein_scalar_log_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    fractal_set_regression_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    fractal_set_baseline_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    fractal_set_frame_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination="remote",
        page_size=25,
        show_index=False,
        sizing_mode="stretch_width",
    )
    fractal_set_points_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination="remote",
        page_size=25,
        show_index=False,
        sizing_mode="stretch_width",
    )

    def on_run_fractal_set(_: Any) -> None:
        """Compute IG/CST area-law measurements from recorded companion traces."""

        def _compute(history: RunHistory) -> None:
            points_df, regression_df, frame_df = compute_fractal_set_measurements(
                history,
                fractal_set_settings,
            )

            if points_df.empty:
                fractal_set_summary.object = (
                    "## Fractal Set Summary\n_No valid measurements for the selected settings._"
                )
                fractal_set_regression_table.value = pd.DataFrame()
                fractal_set_baseline_table.value = pd.DataFrame()
                fractal_set_frame_table.value = pd.DataFrame()
                fractal_set_points_table.value = pd.DataFrame()
                fractal_set_plot_dist.object = hv.Text(0, 0, "No data")
                fractal_set_plot_fit.object = hv.Text(0, 0, "No data")
                fractal_set_plot_total.object = hv.Text(0, 0, "No data")
                fractal_set_plot_dist_geom.object = hv.Text(0, 0, "No data")
                fractal_set_plot_fit_geom.object = hv.Text(0, 0, "No data")
                fractal_set_plot_total_geom.object = hv.Text(0, 0, "No data")
                fractal_set_status.object = (
                    "**Error:** Could not build any non-trivial boundary partitions."
                )
                return

            state["fractal_set_points"] = points_df
            state["fractal_set_regressions"] = regression_df
            state["fractal_set_frame_summary"] = frame_df

            display_regression = regression_df.copy()
            if not display_regression.empty:
                if "metric_label" in display_regression.columns:
                    display_regression["metric"] = display_regression["metric_label"]
                for column in ("slope_alpha", "intercept", "r2"):
                    display_regression[column] = pd.to_numeric(
                        display_regression[column], errors="coerce"
                    ).round(6)
            fractal_set_regression_table.value = display_regression
            fractal_set_baseline_table.value = build_fractal_set_baseline_comparison(regression_df)
            fractal_set_frame_table.value = frame_df.sort_values([
                "recorded_step",
                "partition_family",
                "cut_type",
            ]).reset_index(drop=True)
            fractal_set_points_table.value = points_df.sort_values([
                "recorded_step",
                "partition_family",
                "cut_type",
                "cut_value",
            ]).reset_index(drop=True)

            fractal_set_summary.object = format_fractal_set_summary(
                points_df,
                regression_df,
                frame_df,
            )
            show_geom = bool(fractal_set_settings.use_geometry_correction) and (
                fractal_set_settings.metric_display in {"geometry", "both"}
            )
            show_raw = fractal_set_settings.metric_display in {"raw", "both"} or not bool(
                fractal_set_settings.use_geometry_correction
            )

            if show_raw:
                fractal_set_plot_dist.object = build_fractal_set_scatter_plot(
                    points_df,
                    regression_df,
                    metric_key="s_dist",
                    title="S_dist vs Area_CST",
                )
                fractal_set_plot_fit.object = build_fractal_set_scatter_plot(
                    points_df,
                    regression_df,
                    metric_key="s_fit",
                    title="S_fit vs Area_CST",
                )
                fractal_set_plot_total.object = build_fractal_set_scatter_plot(
                    points_df,
                    regression_df,
                    metric_key="s_total",
                    title="S_total vs Area_CST",
                )
            else:
                fractal_set_plot_dist.object = hv.Text(0, 0, "Raw metrics hidden")
                fractal_set_plot_fit.object = hv.Text(0, 0, "Raw metrics hidden")
                fractal_set_plot_total.object = hv.Text(0, 0, "Raw metrics hidden")

            if show_geom:
                fractal_set_plot_dist_geom.object = build_fractal_set_scatter_plot(
                    points_df,
                    regression_df,
                    metric_key="s_dist_geom",
                    title="S_dist_geom vs Area_CST_geom",
                )
                fractal_set_plot_fit_geom.object = build_fractal_set_scatter_plot(
                    points_df,
                    regression_df,
                    metric_key="s_fit_geom",
                    title="S_fit_geom vs Area_CST_geom",
                )
                fractal_set_plot_total_geom.object = build_fractal_set_scatter_plot(
                    points_df,
                    regression_df,
                    metric_key="s_total_geom",
                    title="S_total_geom vs Area_CST_geom",
                )
            else:
                fractal_set_plot_dist_geom.object = hv.Text(0, 0, "Geometry metrics hidden")
                fractal_set_plot_fit_geom.object = hv.Text(0, 0, "Geometry metrics hidden")
                fractal_set_plot_total_geom.object = hv.Text(0, 0, "Geometry metrics hidden")

            n_frames = int(points_df["info_idx"].nunique())
            n_samples = len(points_df)
            all_frames = list(range(1, int(getattr(history, "n_recorded", 0))))
            if new_dirac_ew_settings is not None:
                kernel_distance_method = str(new_dirac_ew_settings.kernel_distance_method)
                edge_weight_mode = str(new_dirac_ew_settings.edge_weight_mode)
                kernel_assume_all_alive = bool(new_dirac_ew_settings.kernel_assume_all_alive)
            else:
                kernel_distance_method = "auto"
                edge_weight_mode = "riemannian_kernel_volume"
                kernel_assume_all_alive = True
            precomputed_pairs = _resolve_electroweak_geodesic_matrices(
                history,
                all_frames,
                state,
                kernel_distance_method,
                edge_weight_mode,
                kernel_assume_all_alive,
            )
            distribution = _build_geodesic_distance_distribution_by_frame(precomputed_pairs)
            state["_multiscale_geodesic_distribution"] = distribution
            fractal_set_geodesic_distribution_plot.object = _build_geodesic_distribution_plot(
                distribution
            )
            g_newton_metric = (
                "s_total_geom" if bool(fractal_set_settings.use_geometry_correction) else "s_total"
            )
            try:
                einstein_result = compute_einstein_test(
                    history,
                    EinsteinConfig(),
                    fractal_set_regressions=regression_df,
                    g_newton_metric=g_newton_metric,
                )
                state["einstein_test_result"] = einstein_result
                einstein_scalar_log_plot.object = build_scalar_test_log_plot(einstein_result)
            except Exception as exc:
                einstein_scalar_log_plot.object = hv.Text(
                    0,
                    0,
                    f"Einstein test unavailable: {exc!s}",
                ).opts(title="Einstein Scalar Test")
            fractal_set_status.object = (
                f"**Complete:** {n_samples} boundary samples from {n_frames} recorded transitions."
            )
            if precomputed_pairs is not None:
                geodesic_samples = int(
                    state.get("_multiscale_geodesic_distribution", {}).get(
                        "n_geodesic_samples",
                        0,
                    )
                    or 0
                )
                geodesic_frames = int(
                    state.get("_multiscale_geodesic_distribution", {}).get(
                        "n_frames_with_samples",
                        0,
                    )
                    or 0
                )
                fractal_set_status.object = (
                    f"{fractal_set_status.object}"
                    f"  Geodesic pairwise matrix cache: {geodesic_samples} samples from "
                    f"{geodesic_frames} frames."
                )

        run_tab_computation(state, fractal_set_status, "fractal set", _compute)

    def on_history_changed(defer_dashboard_updates: bool) -> None:
        """Update holographic-section controls when a new history is loaded."""
        fractal_set_run_button.disabled = False
        fractal_set_status.object = "**Holographic Principle ready:** click Compute Fractal Set."

    fractal_set_note = pn.pane.Alert(
        """
**Fractal Set Protocol**
- IG measurements: cross-boundary companion counts from `companions_distance` and
  `companions_clone` (`S_dist`, `S_fit`, `S_total`).
- CST measurement: number of ancestral lineages with descendants on both sides of
  the same partition (`Area_CST`).
- Partition generators:
  spatial boundaries (hyperplane/spherical/median),
  spectral graph cuts from companion graphs, and
  random partition baseline.
- Geometry correction (optional):
  Riemannian-kernel edge lengths and volume-weighted CST area
  (`S_*_geom`, `Area_CST_geom`).
                """,
        alert_type="info",
        sizing_mode="stretch_width",
    )

    fractal_set_tab = pn.Column(
        fractal_set_status,
        fractal_set_note,
        pn.Row(fractal_set_run_button, sizing_mode="stretch_width"),
        pn.Accordion(
            ("Fractal Set Settings", fractal_set_settings_panel),
            active=[0],
            sizing_mode="stretch_width",
        ),
        pn.layout.Divider(),
        fractal_set_summary,
        pn.pane.Markdown("### Linear Fits"),
        fractal_set_regression_table,
        pn.pane.Markdown("### Compare Vs Random Baseline"),
        fractal_set_baseline_table,
        pn.layout.Divider(),
        pn.pane.Markdown("### S_dist vs Area_CST"),
        fractal_set_plot_dist,
        pn.pane.Markdown("### S_fit vs Area_CST"),
        fractal_set_plot_fit,
        pn.pane.Markdown("### S_total vs Area_CST"),
        fractal_set_plot_total,
        pn.layout.Divider(),
        pn.pane.Markdown("### S_dist_geom vs Area_CST_geom"),
        fractal_set_plot_dist_geom,
        pn.pane.Markdown("### S_fit_geom vs Area_CST_geom"),
        fractal_set_plot_fit_geom,
        pn.pane.Markdown("### S_total_geom vs Area_CST_geom"),
        fractal_set_plot_total_geom,
        pn.layout.Divider(),
        pn.pane.Markdown("### Geodesic Pairwise Distance Distribution"),
        fractal_set_geodesic_distribution_plot,
        pn.pane.Markdown("### Einstein Scalar Test: R vs log10(rho)"),
        einstein_scalar_log_plot,
        pn.layout.Divider(),
        pn.pane.Markdown("### Per-Frame Means"),
        fractal_set_frame_table,
        pn.Accordion(
            ("Raw Boundary Samples", fractal_set_points_table),
            active=[],
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_both",
    )

    return HolographicPrincipleSection(
        fractal_set_tab=fractal_set_tab,
        fractal_set_status=fractal_set_status,
        fractal_set_run_button=fractal_set_run_button,
        on_run_fractal_set=on_run_fractal_set,
        on_history_changed=on_history_changed,
    )
