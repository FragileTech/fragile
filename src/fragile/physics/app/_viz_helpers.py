"""Pure visualization helper functions for the 3D swarm viewer."""

from __future__ import annotations

from typing import Any

import numpy as np


def rgba_from_color(color: str | None, alpha: float) -> str:
    """Convert a CSS color string to rgba() with the given alpha."""
    if color is None:
        return f"rgba(0, 0, 0, {alpha})"
    color = color.strip()
    if color.startswith("rgba(") and color.endswith(")"):
        parts = color[5:-1].split(",")
        if len(parts) >= 3:
            r, g, b = (p.strip() for p in parts[:3])
            return f"rgba({r}, {g}, {b}, {alpha})"
    if color.startswith("rgb(") and color.endswith(")"):
        parts = color[4:-1].split(",")
        if len(parts) >= 3:
            r, g, b = (p.strip() for p in parts[:3])
            return f"rgba({r}, {g}, {b}, {alpha})"
    if color.startswith("#"):
        hex_value = color[1:]
        if len(hex_value) == 3:
            hex_value = "".join([c * 2 for c in hex_value])
        if len(hex_value) == 6:
            r = int(hex_value[0:2], 16)
            g = int(hex_value[2:4], 16)
            b = int(hex_value[4:6], 16)
            return f"rgba({r}, {g}, {b}, {alpha})"
    return color


def normalize_edges(edges: np.ndarray | None) -> np.ndarray:
    """Deduplicate and sort edge pairs, removing self-loops."""
    if edges is None:
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.asarray(edges)
    if edges.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    edges = edges.reshape(-1, 2)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if edges.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def axis_label(dim_spec: str, euclidean_dim_idx: int = 0) -> str:
    """Return a human-readable axis label for *dim_spec*.

    For ``"euclidean_time"`` the caller must resolve and pass *euclidean_dim_idx*
    so this function stays free of ``self`` access.
    """
    if dim_spec == "mc_time":
        return "Monte Carlo Time (frame)"
    if dim_spec == "euclidean_time":
        return f"Euclidean Time (dim_{euclidean_dim_idx})"
    if dim_spec == "riemannian_volume":
        return "Riemannian Volume"
    if dim_spec.startswith("dim_"):
        dim_idx = int(dim_spec.split("_")[1])
        labels = ["X", "Y", "Z", "T"]
        if dim_idx < len(labels):
            return f"Dimension {dim_idx} ({labels[dim_idx]})"
        return f"Dimension {dim_idx}"
    return dim_spec


def compute_color_for_metric(
    metric: str,
    frame: int,
    positions_all: np.ndarray,
    alive: np.ndarray,
    *,
    fitness: np.ndarray | None = None,
    rewards: np.ndarray | None = None,
    volume_weights: np.ndarray | None = None,
    euclidean_dim_idx: int = 0,
    fallback_nan: bool = False,
) -> tuple[np.ndarray | str, bool, dict[str, Any] | None]:
    """Compute per-walker color values for a single frame.

    When *fallback_nan* is True (used by the all-frames path), metrics that
    have no data at ``frame == 0`` produce ``NaN`` arrays instead of a constant
    color string, so they can be concatenated across frames.
    """
    n_alive = int(alive.sum())

    if metric.startswith("dim_"):
        dim_idx = int(metric.split("_")[1])
        if dim_idx < positions_all.shape[1]:
            colors = positions_all[alive, dim_idx]
        else:
            colors = np.zeros(n_alive)
        return colors, True, {"title": axis_label(metric)}

    if metric == "euclidean_time":
        if euclidean_dim_idx < positions_all.shape[1]:
            colors = positions_all[alive, euclidean_dim_idx]
        else:
            colors = np.zeros(n_alive)
        return colors, True, {"title": "Euclidean Time"}

    if metric == "mc_time":
        colors = np.full(n_alive, frame, dtype=float)
        return colors, True, {"title": "MC Time (frame)"}

    if metric == "fitness":
        if frame == 0 or fitness is None:
            if fallback_nan:
                return np.full(n_alive, np.nan), True, {"title": "Fitness"}
            return "#1f77b4", False, None
        idx = min(frame - 1, len(fitness) - 1)
        colors = fitness[idx][alive]
        return colors, True, {"title": "Fitness"}

    if metric == "reward":
        if frame == 0 or rewards is None:
            if fallback_nan:
                return np.full(n_alive, np.nan), True, {"title": "Reward"}
            return "#1f77b4", False, None
        idx = min(frame - 1, len(rewards) - 1)
        colors = rewards[idx][alive]
        return colors, True, {"title": "Reward"}

    if metric == "riemannian_volume":
        if frame == 0 or volume_weights is None:
            if fallback_nan:
                return np.full(n_alive, np.nan), True, {"title": "Riemannian Volume"}
            return "#1f77b4", False, None
        idx = min(frame - 1, len(volume_weights) - 1)
        colors = volume_weights[idx][alive]
        return colors, True, {"title": "Riemannian Volume"}

    if metric == "radius":
        positions_filtered = positions_all[alive][:, : min(3, positions_all.shape[1])]
        colors = np.linalg.norm(positions_filtered, axis=1)
        return colors, True, {"title": "Radius"}

    # "constant"
    return "#1f77b4", False, None


def build_edge_segments(
    edges: np.ndarray,
    positions_all: np.ndarray,
    alive: np.ndarray,
    positions_mapped: np.ndarray,
    *,
    edge_values: np.ndarray | None = None,
    color_metric: str = "constant",
    line_width: float = 1.2,
    line_style: str = "solid",
    line_alpha: float = 0.35,
    line_color: str = "#2b2b2b",
    line_colorscale: str = "Viridis",
    trace_name: str = "Delaunay edges",
    showlegend: bool = False,
):
    """Build a Scatter3d trace for Delaunay edge segments.

    Returns a ``plotly.graph_objects.Scatter3d`` or ``None``.
    """
    import plotly.graph_objects as go

    if edges is None or edges.size == 0 or positions_mapped.size == 0:
        return None

    alive_indices = np.where(alive)[0]
    if alive_indices.size == 0:
        return None
    alive_map = {int(idx): pos for pos, idx in enumerate(alive_indices)}

    valid_edges: list[tuple[int, int, int, int]] = []
    values: list[float] | None = [] if edge_values is not None else None
    seen: set[tuple[int, int]] = set()
    for idx, (i, j) in enumerate(edges):
        if i == j:
            continue
        i_local = alive_map.get(int(i))
        j_local = alive_map.get(int(j))
        if i_local is None or j_local is None:
            continue
        key = (int(min(i, j)), int(max(i, j)))
        if key in seen:
            continue
        seen.add(key)
        valid_edges.append((int(i), int(j), i_local, j_local))
        if values is not None:
            values.append(float(edge_values[idx]))

    if not valid_edges:
        return None

    x_edges, y_edges, z_edges = [], [], []
    metric = color_metric
    label_metric = metric
    edge_values_list: list[float] | None = None
    raw_values = None
    if metric == "distance":
        edge_values_list = []
        pairs = np.array([(i, j) for i, j, _, _ in valid_edges], dtype=np.int64)
        deltas = positions_all[pairs[:, 0]] - positions_all[pairs[:, 1]]
        raw_values = np.linalg.norm(deltas, axis=1)
    elif metric == "geodesic":
        edge_values_list = []
        if values is not None and len(values) == len(valid_edges):
            raw_values = np.asarray(values, dtype=float)
        else:
            label_metric = "distance"
            pairs = np.array([(i, j) for i, j, _, _ in valid_edges], dtype=np.int64)
            deltas = positions_all[pairs[:, 0]] - positions_all[pairs[:, 1]]
            raw_values = np.linalg.norm(deltas, axis=1)

    for idx, (i, j, i_local, j_local) in enumerate(valid_edges):
        x_edges.extend([positions_mapped[i_local, 0], positions_mapped[j_local, 0], None])
        y_edges.extend([positions_mapped[i_local, 1], positions_mapped[j_local, 1], None])
        z_edges.extend([positions_mapped[i_local, 2], positions_mapped[j_local, 2], None])
        if edge_values_list is not None:
            value = float(raw_values[idx]) if raw_values is not None else float("nan")
            edge_values_list.extend([value, value, np.nan])

    if edge_values_list is not None:
        line = {
            "color": edge_values_list,
            "width": line_width,
            "dash": line_style,
            "colorscale": line_colorscale,
            "showscale": True,
            "colorbar": {
                "title": "Geodesic distance"
                if label_metric == "geodesic"
                else "Euclidean distance"
            },
        }
        if raw_values is not None and np.asarray(raw_values).size:
            line["cmin"] = float(np.nanmin(raw_values))
            line["cmax"] = float(np.nanmax(raw_values))
        return go.Scatter3d(
            x=x_edges,
            y=y_edges,
            z=z_edges,
            mode="lines",
            line=line,
            opacity=line_alpha,
            hoverinfo="skip",
            name=trace_name,
            showlegend=showlegend,
        )

    rgba = rgba_from_color(line_color, line_alpha)
    return go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode="lines",
        line={
            "color": rgba,
            "width": line_width,
            "dash": line_style,
        },
        hoverinfo="skip",
        name=trace_name,
        showlegend=showlegend,
    )
