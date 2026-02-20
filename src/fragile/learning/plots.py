"""Pure plotting functions for the TopoEncoder learning dashboard.

Each function takes numpy arrays and returns a plot object (HoloViews or Plotly).
No state, no widgets.
"""

from __future__ import annotations

from bokeh.models import HoverTool, TapTool
import holoviews as hv
import numpy as np
import plotly.graph_objects as go


# Consistent color palette
COLORS = {
    "atlas": "#4c78a8",
    "std": "#f58518",
    "ae": "#54a24b",
}


def _to_numpy(t) -> np.ndarray:
    """Convert tensor/array-like to numpy."""
    if hasattr(t, "cpu"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def plot_loss_curves(
    epochs: np.ndarray,
    atlas_losses: np.ndarray,
    std_losses: np.ndarray | None = None,
    ae_losses: np.ndarray | None = None,
) -> hv.Overlay:
    """Overlay of training loss curves for all models."""
    hover = HoverTool(tooltips=[("Model", "$label"), ("Epoch", "@x{0}"), ("Loss", "@y{0.5g}")])
    curves = [
        hv.Curve((epochs, atlas_losses), "Epoch", "Loss", label="TopoEncoder").opts(
            color=COLORS["atlas"], line_width=2, tools=[hover]
        )
    ]
    if std_losses is not None and len(std_losses) > 0:
        curves.append(
            hv.Curve((epochs[: len(std_losses)], std_losses), "Epoch", "Loss", label="VQ").opts(
                color=COLORS["std"], line_width=2, tools=[hover]
            )
        )
    if ae_losses is not None and len(ae_losses) > 0:
        curves.append(
            hv.Curve((epochs[: len(ae_losses)], ae_losses), "Epoch", "Loss", label="AE").opts(
                color=COLORS["ae"], line_width=2, tools=[hover]
            )
        )
    return hv.Overlay(curves).opts(
        title="Training Loss", legend_position="top_right", width=700, height=350, logy=True
    )


def plot_loss_components(
    epochs: np.ndarray,
    loss_components: dict[str, np.ndarray],
    selected: list[str] | None = None,
) -> hv.Overlay:
    """Individual loss terms over training."""
    keys = selected or [k for k, v in loss_components.items() if len(v) > 0]
    palette = hv.Cycle("Category20")
    hover = HoverTool(
        tooltips=[("Component", "$label"), ("Epoch", "@x{0}"), ("Value", "@y{0.5g}")]
    )
    curves = []
    for key in keys:
        vals = loss_components.get(key)
        if vals is None or len(vals) == 0:
            continue
        arr = np.asarray(vals)
        if np.all(arr == 0):
            continue
        curves.append(
            hv.Curve((epochs[: len(arr)], arr), "Epoch", "Value", label=key).opts(
                color=palette, line_width=1.5, tools=[hover]
            )
        )
    if not curves:
        return hv.Overlay([hv.Curve([], "Epoch", "Value")]).opts(title="Loss Components")
    return hv.Overlay(curves).opts(
        title="Loss Components", legend_position="right", width=700, height=350, logy=True
    )


def plot_info_metrics(
    epochs: np.ndarray,
    info_metrics: dict[str, np.ndarray],
    selected: list[str] | None = None,
) -> hv.Overlay:
    """Information-theoretic metrics: I(X;K), H(K), code entropy, etc."""
    keys = selected or [k for k, v in info_metrics.items() if len(v) > 0]
    palette = hv.Cycle("Category10")
    hover = HoverTool(tooltips=[("Metric", "$label"), ("Epoch", "@x{0}"), ("Value", "@y{0.5g}")])
    curves = []
    for key in keys:
        vals = info_metrics.get(key)
        if vals is None or len(vals) == 0:
            continue
        arr = np.asarray(vals)
        curves.append(
            hv.Curve((epochs[: len(arr)], arr), "Epoch", "Value", label=key).opts(
                color=palette, line_width=1.5, tools=[hover]
            )
        )
    if not curves:
        return hv.Overlay([hv.Curve([], "Epoch", "Value")]).opts(title="Info Metrics")
    return hv.Overlay(curves).opts(
        title="Info Metrics", legend_position="right", width=700, height=350
    )


def plot_classifier_accuracy(
    epochs: np.ndarray,
    loss_components: dict[str, np.ndarray],
) -> hv.Overlay:
    """Classifier accuracy curves: cls_acc, std_cls_acc, ae_cls_acc, sup_acc."""
    acc_keys = {
        "cls_acc": ("Atlas classifier", COLORS["atlas"]),
        "std_cls_acc": ("VQ classifier", COLORS["std"]),
        "ae_cls_acc": ("AE classifier", COLORS["ae"]),
        "sup_acc": ("Supervised", "#b279a2"),
    }
    hover = HoverTool(tooltips=[("Model", "$label"), ("Epoch", "@x{0}"), ("Accuracy", "@y{0.4f}")])
    curves = []
    for key, (label, color) in acc_keys.items():
        vals = loss_components.get(key)
        if vals is None or len(vals) == 0:
            continue
        arr = np.asarray(vals)
        if np.all(arr == 0):
            continue
        curves.append(
            hv.Curve((epochs[: len(arr)], arr), "Epoch", "Accuracy", label=label).opts(
                color=color, line_width=2, tools=[hover]
            )
        )
    if not curves:
        return hv.Overlay([hv.Curve([], "Epoch", "Accuracy")]).opts(title="Classifier Accuracy")
    return hv.Overlay(curves).opts(
        title="Classifier Accuracy",
        legend_position="bottom_right",
        width=700,
        height=350,
        ylim=(0, 1),
    )


# ---------------------------------------------------------------------------
# Chart usage & reconstructions
# ---------------------------------------------------------------------------


def plot_chart_usage(usage_array: np.ndarray) -> hv.Bars:
    """Bar chart of routing mass per chart."""
    usage = _to_numpy(usage_array)
    hover = HoverTool(tooltips=[("Chart", "@Chart"), ("Usage", "@Usage{0.4f}")])
    bars = hv.Bars(
        [(f"C{i}", float(v)) for i, v in enumerate(usage)],
        kdims="Chart",
        vdims="Usage",
    )
    return bars.opts(
        title="Chart Usage", width=700, height=300, color=COLORS["atlas"], tools=[hover]
    )


def plot_reconstruction_grid(
    originals: np.ndarray,
    recon_topo: np.ndarray,
    recon_vq: np.ndarray | None = None,
    recon_ae: np.ndarray | None = None,
    n_samples: int = 8,
    image_shape: tuple[int, ...] = (28, 28),
) -> hv.Layout:
    """Image grid: columns=[Original, TopoEncoder, VQ, AE], rows=samples."""
    n = min(n_samples, len(originals))
    columns = [("Original", originals), ("TopoEncoder", recon_topo)]
    if recon_vq is not None:
        columns.append(("VQ", recon_vq))
    if recon_ae is not None:
        columns.append(("AE", recon_ae))

    hover = HoverTool(tooltips=[("x", "$x{0.1f}"), ("y", "$y{0.1f}"), ("pixel", "@image{0.3f}")])
    images = []
    for row_idx in range(n):
        for col_label, data in columns:
            img = data[row_idx].reshape(image_shape)
            images.append(
                hv.Image(
                    img,
                    bounds=(0, 0, image_shape[1], image_shape[0]),
                    label=f"{col_label}",
                ).opts(
                    cmap="gray",
                    xaxis=None,
                    yaxis=None,
                    width=120,
                    height=120,
                    title=f"{col_label}" if row_idx == 0 else "",
                    tools=[hover],
                )
            )
    ncols = len(columns)
    return hv.Layout(images).cols(ncols).opts(title="Reconstructions")


# ---------------------------------------------------------------------------
# Latent space
# ---------------------------------------------------------------------------


def chart_to_label_map(K_chart: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Build majority-vote mapping from chart index to most frequent label.

    Returns an array of length ``num_charts`` where ``map[c]`` is the label
    most frequently assigned to chart ``c``.
    """
    num_charts = int(K_chart.max()) + 1
    mapping = np.zeros(num_charts, dtype=int)
    for c in range(num_charts):
        mask = K_chart == c
        if mask.any():
            vals, counts = np.unique(labels[mask], return_counts=True)
            mapping[c] = vals[counts.argmax()]
    return mapping


def build_latent_scatter(
    z_geo: np.ndarray,
    labels: np.ndarray,
    K_chart: np.ndarray,
    correct: np.ndarray,
    color_by: str,
    point_size: int,
    dim_i: int,
    dim_j: int,
    indices: np.ndarray | None = None,
) -> hv.Points:
    """Build a single 2D scatter for one dimension pair.

    Parameters
    ----------
    indices
        Original dataset index for each displayed point (enables click-to-inspect).
    """
    z = _to_numpy(z_geo)
    labs = _to_numpy(labels).astype(int)
    charts = _to_numpy(K_chart).astype(int)
    corr = _to_numpy(correct).astype(int)
    correct_str = np.where(corr == 1, "yes", "no")

    if color_by == "chart":
        color_col, cmap = "chart", "Category10"
    elif color_by == "correct":
        color_col, cmap = "correct", {0: "#e45756", 1: "#54a24b"}
    else:
        color_col, cmap = "label", "Category10"

    data = {
        "x": z[:, dim_i],
        "y": z[:, dim_j],
        "label": labs,
        "chart": charts,
        "correct": corr,
        "correct_str": correct_str,
    }
    vdims = ["label", "chart", "correct", "correct_str"]
    if indices is not None:
        data["idx"] = np.asarray(indices)
        vdims.append("idx")

    hover = HoverTool(
        tooltips=[
            (f"z{dim_i}", "@x{0.3f}"),
            (f"z{dim_j}", "@y{0.3f}"),
            ("Label", "@label"),
            ("Chart", "@chart"),
            ("Correct", "@correct_str"),
        ]
    )
    return hv.Points(data, kdims=["x", "y"], vdims=vdims).opts(
        color=color_col,
        cmap=cmap,
        size=point_size,
        width=350,
        height=350,
        xlim=(None, None),
        ylim=(None, None),
        xlabel=f"z{dim_i}",
        ylabel=f"z{dim_j}",
        title=f"z{dim_i} vs z{dim_j}",
        colorbar=True,
        tools=[hover, TapTool()],
    )


def plot_latent_2d_slices(
    z_geo: np.ndarray,
    labels: np.ndarray,
    K_chart: np.ndarray | None = None,
    correct: np.ndarray | None = None,
    color_by: str = "label",
    point_size: int = 3,
    indices: np.ndarray | None = None,
) -> hv.Layout:
    """2D scatter panels for every pair among the first 3 latent dims.

    ``color_by`` selects the coloring: ``"label"``, ``"chart"``, or ``"correct"``.
    """
    z = _to_numpy(z_geo)
    labs = _to_numpy(labels).astype(int)
    charts = _to_numpy(K_chart).astype(int) if K_chart is not None else np.zeros_like(labs)
    corr = _to_numpy(correct).astype(int) if correct is not None else np.ones_like(labs)
    dim = z.shape[1]

    pairs = []
    if dim >= 2:
        pairs.append((0, 1))
    if dim >= 3:
        pairs.append((0, 2))
        pairs.append((1, 2))

    panels = []
    for i, j in pairs:
        scatter = build_latent_scatter(
            z,
            labs,
            charts,
            corr,
            color_by,
            point_size,
            i,
            j,
            indices=indices,
        )
        panels.append(scatter)

    if not panels:
        return hv.Layout([hv.Points([], kdims=["x", "y"]).opts(title="No latent dims")])
    return hv.Layout(panels).cols(min(3, len(panels)))


def plot_latent_3d(
    z_geo: np.ndarray,
    labels: np.ndarray,
    K_chart: np.ndarray | None = None,
    correct: np.ndarray | None = None,
    color_by: str = "label",
    point_size: int = 2,
) -> go.Figure:
    """3D scatter of z_geo[:,0:3] colored by label, chart, or correct/incorrect."""
    z = _to_numpy(z_geo)
    labs = _to_numpy(labels).astype(int)
    charts = _to_numpy(K_chart).astype(int) if K_chart is not None else np.zeros_like(labs)
    corr = _to_numpy(correct).astype(int) if correct is not None else np.ones_like(labs)

    ndim = z.shape[1]
    x = z[:, 0] if ndim > 0 else np.zeros(len(z))
    y = z[:, 1] if ndim > 1 else np.zeros(len(z))
    z_ax = z[:, 2] if ndim > 2 else np.zeros(len(z))

    # Coloring
    if color_by == "chart":
        color_vals = charts
        color_label = "Chart"
        colorscale = "Viridis"
    elif color_by == "correct":
        color_vals = corr
        color_label = "Correct"
        colorscale = [[0, "#e45756"], [1, "#54a24b"]]
    else:
        color_vals = labs
        color_label = "Label"
        colorscale = "Viridis"

    hover_text = [
        f"z0={x[k]:.3f}<br>z1={y[k]:.3f}<br>z2={z_ax[k]:.3f}"
        f"<br>Label={labs[k]}<br>Chart={charts[k]}"
        f"<br>Correct={'yes' if corr[k] else 'no'}"
        for k in range(len(x))
    ]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z_ax,
                mode="markers",
                marker={
                    "size": point_size,
                    "color": color_vals,
                    "colorscale": colorscale,
                    "opacity": 0.7,
                    "colorbar": {"title": color_label},
                },
                text=hover_text,
                hoverinfo="text",
            )
        ]
    )
    fig.update_layout(
        title=f"Latent Space (color={color_label})",
        scene={"xaxis_title": "z0", "yaxis_title": "z1", "zaxis_title": "z2"},
        width=700,
        height=600,
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig
