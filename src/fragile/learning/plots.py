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


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#rrggbb' to (r, g, b) integers."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


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
                    kdims=["img_x", "img_y"],
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
    return hv.Layout(images).cols(ncols).opts(title="Reconstructions", shared_axes=False)


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
    confidence: np.ndarray | None = None,
    alpha_by_confidence: bool = False,
) -> hv.Scatter:
    """Build a single 2D scatter for one dimension pair.

    Parameters
    ----------
    indices
        Original dataset index for each displayed point (enables click-to-inspect).
    confidence
        Per-point confidence (max softmax probability) in [0, 1].
    alpha_by_confidence
        When True, map per-point alpha to the confidence value.
    """
    z = _to_numpy(z_geo)
    labs = _to_numpy(labels).astype(int)
    charts = _to_numpy(K_chart).astype(int)
    corr = _to_numpy(correct).astype(int)
    correct_str = np.where(corr == 1, "yes", "no")

    # Rescale from [-1, 1] → [0, 25] to work around HoloViews axis-range bug
    def _rescale(v: np.ndarray) -> np.ndarray:
        return (v + 1.0) * 12.5

    if color_by == "confidence":
        color_col, cmap = "confidence", "Viridis"
    elif color_by == "chart":
        color_col, cmap = "chart", "Category10"
    elif color_by == "correct":
        color_col, cmap = "correct", {0: "#e45756", 1: "#54a24b"}
    else:
        color_col, cmap = "label", "Category10"

    data = {
        "x": _rescale(z[:, dim_i]),
        "y": _rescale(z[:, dim_j]),
        "label": labs,
        "chart": charts,
        "correct": corr,
        "correct_str": correct_str,
    }
    vdims = ["label", "chart", "correct", "correct_str"]

    if confidence is not None:
        data["confidence"] = confidence.astype(float)
        vdims.append("confidence")

    if indices is not None:
        data["idx"] = np.asarray(indices)
        vdims.append("idx")

    tooltips = [
        (f"z{dim_i}", "@x{0.3f}"),
        (f"z{dim_j}", "@y{0.3f}"),
        ("Label", "@label"),
        ("Chart", "@chart"),
        ("Correct", "@correct_str"),
    ]
    if confidence is not None:
        tooltips.append(("Confidence", "@confidence{0.3f}"))
    hover = HoverTool(tooltips=tooltips)

    opts_kw: dict = {
        "color": color_col,
        "cmap": cmap,
        "size": point_size,
        "width": 350,
        "height": 350,
        "xlabel": f"z{dim_i}",
        "ylabel": f"z{dim_j}",
        "title": f"z{dim_i} vs z{dim_j}",
        "colorbar": True,
        "tools": [hover, TapTool()],
    }
    if color_by == "confidence" and confidence is not None:
        opts_kw["clim"] = (0, 1)
    if alpha_by_confidence and confidence is not None:
        opts_kw["alpha"] = hv.dim("confidence")
    return hv.Scatter(data, kdims=["x", "y"], vdims=vdims).opts(**opts_kw)


def plot_latent_2d_slices(
    z_geo: np.ndarray,
    labels: np.ndarray,
    K_chart: np.ndarray | None = None,
    correct: np.ndarray | None = None,
    color_by: str = "label",
    point_size: int = 3,
    indices: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    alpha_by_confidence: bool = False,
) -> hv.Layout:
    """2D scatter panels for every pair among the first 3 latent dims.

    ``color_by`` selects the coloring: ``"label"``, ``"chart"``, ``"correct"``,
    or ``"confidence"``.
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
            confidence=confidence,
            alpha_by_confidence=alpha_by_confidence,
        )
        panels.append(scatter)

    if not panels:
        return hv.Layout([hv.Scatter([], kdims=["x", "y"]).opts(title="No latent dims")])
    return hv.Layout(panels).opts(shared_axes=False).cols(min(3, len(panels)))


_CATEGORY10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def plot_latent_3d(
    z_geo: np.ndarray,
    labels: np.ndarray,
    K_chart: np.ndarray | None = None,
    correct: np.ndarray | None = None,
    color_by: str = "label",
    point_size: int = 2,
    K_code: np.ndarray | None = None,
    show_hierarchy: bool = False,
    tree_line_color: str = "black",
    tree_line_width: float = 0.5,
    confidence: np.ndarray | None = None,
    alpha_by_confidence: bool = False,
) -> go.Figure:
    """3D scatter of z_geo[:,0:3] colored by label, chart, correct, or confidence."""
    z = _to_numpy(z_geo)
    labs = _to_numpy(labels).astype(int)
    charts = _to_numpy(K_chart).astype(int) if K_chart is not None else np.zeros_like(labs)
    corr = _to_numpy(correct).astype(int) if correct is not None else np.ones_like(labs)

    ndim = z.shape[1]
    x = z[:, 0] if ndim > 0 else np.zeros(len(z))
    y = z[:, 1] if ndim > 1 else np.zeros(len(z))
    z_ax = z[:, 2] if ndim > 2 else np.zeros(len(z))
    hierarchy_z = ndim < 3  # use z-axis for hierarchy levels when embedding is 2D

    base_opacity = 0.7
    traces = []

    if color_by == "confidence" and confidence is not None:
        # Continuous colorscale — single trace
        conf = confidence.astype(float)
        hover_text = [
            f"z0={x[k]:.3f}<br>z1={y[k]:.3f}<br>z2={z_ax[k]:.3f}"
            f"<br>Label={labs[k]}<br>Chart={charts[k]}"
            f"<br>Correct={'yes' if corr[k] else 'no'}"
            f"<br>Confidence={conf[k]:.3f}"
            for k in range(len(x))
        ]
        marker: dict = {
            "size": point_size,
            "color": conf,
            "colorscale": "Viridis",
            "cmin": 0,
            "cmax": 1,
            "colorbar": {"title": "Confidence"},
            "opacity": base_opacity,
        }
        # Plotly scatter3d.marker.opacity is scalar-only; use global opacity
        # and encode per-point alpha via the colorscale when requested.
        if alpha_by_confidence:
            marker["opacity"] = 1.0
            # Build per-point RGBA from Viridis sampled at conf value
            from matplotlib.cm import viridis as _viridis_cm
            rgba_arr = _viridis_cm(conf)  # (N, 4)
            rgba_arr[:, 3] = conf  # set alpha = confidence
            marker.pop("colorscale", None)
            marker.pop("cmin", None)
            marker.pop("cmax", None)
            marker.pop("colorbar", None)
            marker["color"] = [
                f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.3f})"
                for r, g, b, a in rgba_arr
            ]
        traces.append(
            go.Scatter3d(
                x=x, y=y, z=z_ax, mode="markers", name="confidence",
                marker=marker, text=hover_text, hoverinfo="text",
            )
        )
        color_label = "Confidence"
    else:
        # Categorical traces
        if color_by == "chart":
            color_vals = charts
            color_label = "Chart"
            palette = _CATEGORY10
        elif color_by == "correct":
            color_vals = corr
            color_label = "Correct"
            palette = ["#e45756", "#54a24b"]
        else:
            color_vals = labs
            color_label = "Label"
            palette = _CATEGORY10

        categories = sorted(np.unique(color_vals))
        cat_names = (
            {0: "no", 1: "yes"} if color_by == "correct" else None
        )

        for cat in categories:
            mask = color_vals == cat
            cat_color = palette[int(cat) % len(palette)]
            name = cat_names[cat] if cat_names else str(cat)
            hover_text = [
                f"z0={x[k]:.3f}<br>z1={y[k]:.3f}<br>z2={z_ax[k]:.3f}"
                f"<br>Label={labs[k]}<br>Chart={charts[k]}"
                f"<br>Correct={'yes' if corr[k] else 'no'}"
                for k in np.where(mask)[0]
            ]
            mk: dict = {"size": point_size}
            if alpha_by_confidence and confidence is not None:
                # Encode per-point alpha via RGBA color strings
                r, g, b = _hex_to_rgb(cat_color)
                alphas = confidence[mask].astype(float)
                mk["color"] = [
                    f"rgba({r},{g},{b},{a:.3f})" for a in alphas
                ]
                mk["opacity"] = 1.0
            else:
                mk["color"] = cat_color
                mk["opacity"] = base_opacity
            traces.append(
                go.Scatter3d(
                    x=x[mask],
                    y=y[mask],
                    z=z_ax[mask],
                    mode="markers",
                    name=name,
                    marker=mk,
                    text=hover_text,
                    hoverinfo="text",
                )
            )

    # Hierarchy tree overlay
    if show_hierarchy and K_code is not None:
        codes = _to_numpy(K_code).astype(int)
        _add_hierarchy_traces(
            traces, x, y, z_ax, charts, codes, point_size, tree_line_color, tree_line_width,
            hierarchy_z=hierarchy_z,
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Latent Space (color={color_label})",
        scene={
            "xaxis_title": "z0", "yaxis_title": "z1",
            "zaxis_title": "hierarchy" if hierarchy_z else "z2",
            "xaxis": {"range": [-1, 1]},
            "yaxis": {"range": [-1, 1]},
            "zaxis": {"range": [-0.1, 1.1]} if hierarchy_z else {"range": [-1, 1]},
        },
        width=700,
        height=600,
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def _add_hierarchy_traces(
    traces: list,
    x: np.ndarray,
    y: np.ndarray,
    z_ax: np.ndarray,
    charts: np.ndarray,
    codes: np.ndarray,
    point_size: int,
    line_color_mode: str,
    line_width: float,
    hierarchy_z: bool = False,
) -> None:
    """Append hierarchy center markers and tree-edge lines to *traces* (in-place).

    When *hierarchy_z* is True (2D embeddings), z-coordinates are overridden to
    show hierarchy levels: root=1.0, chart centers=0.66, symbol centers=0.33,
    data points=0.0.
    """
    unique_charts = np.unique(charts)

    # Compute chart centers
    chart_centers: dict[int, np.ndarray] = {}
    for c in unique_charts:
        mask = charts == c
        cz = 0.66 if hierarchy_z else z_ax[mask].mean()
        chart_centers[c] = np.array([x[mask].mean(), y[mask].mean(), cz])

    # Compute symbol (chart, code) centers
    symbol_centers: dict[tuple[int, int], np.ndarray] = {}
    for c in unique_charts:
        c_mask = charts == c
        for k in np.unique(codes[c_mask]):
            mask = c_mask & (codes == k)
            sz = 0.33 if hierarchy_z else z_ax[mask].mean()
            symbol_centers[(c, k)] = np.array([x[mask].mean(), y[mask].mean(), sz])

    # Root: centered above all chart centers
    if hierarchy_z:
        all_chart_xy = np.array([cc[:2] for cc in chart_centers.values()])
        root = np.array([all_chart_xy[:, 0].mean(), all_chart_xy[:, 1].mean(), 1.0])
    else:
        root = np.array([0.0, 0.0, 0.0])

    # --- Center markers ---
    # Root
    traces.append(go.Scatter3d(
        x=[root[0]], y=[root[1]], z=[root[2]],
        mode="markers", name="root",
        marker={"size": point_size * 4, "color": "black", "symbol": "diamond"},
        hoverinfo="name", showlegend=False,
    ))
    # Chart centers
    for c, ctr in chart_centers.items():
        col = _CATEGORY10[int(c) % len(_CATEGORY10)]
        traces.append(go.Scatter3d(
            x=[ctr[0]], y=[ctr[1]], z=[ctr[2]],
            mode="markers", name=f"chart {c} center",
            marker={"size": point_size * 3, "color": col, "symbol": "diamond"},
            hoverinfo="name", showlegend=False,
        ))
    # Symbol centers
    for (c, k), ctr in symbol_centers.items():
        col = _CATEGORY10[int(c) % len(_CATEGORY10)]
        traces.append(go.Scatter3d(
            x=[ctr[0]], y=[ctr[1]], z=[ctr[2]],
            mode="markers", name=f"chart {c} code {k}",
            marker={"size": point_size * 2, "color": col, "opacity": 0.6, "symbol": "diamond"},
            hoverinfo="name", showlegend=False,
        ))

    # --- Line traces ---
    if line_color_mode == "black":
        # Single trace with all edges
        lx, ly, lz = [], [], []
        for c in unique_charts:
            _seg(lx, ly, lz, root, chart_centers[c])
            c_mask = charts == c
            for k in np.unique(codes[c_mask]):
                sym = symbol_centers[(c, k)]
                _seg(lx, ly, lz, chart_centers[c], sym)
                for idx in np.where(c_mask & (codes == k))[0]:
                    _seg(lx, ly, lz, sym, np.array([x[idx], y[idx], z_ax[idx]]))
        traces.append(go.Scatter3d(
            x=lx, y=ly, z=lz, mode="lines", name="hierarchy",
            line={"color": "black", "width": line_width},
            hoverinfo="none", showlegend=False,
        ))
    elif line_color_mode == "chart":
        # One trace per chart
        for c in unique_charts:
            col = _CATEGORY10[int(c) % len(_CATEGORY10)]
            lx, ly, lz = [], [], []
            _seg(lx, ly, lz, root, chart_centers[c])
            c_mask = charts == c
            for k in np.unique(codes[c_mask]):
                sym = symbol_centers[(c, k)]
                _seg(lx, ly, lz, chart_centers[c], sym)
                for idx in np.where(c_mask & (codes == k))[0]:
                    _seg(lx, ly, lz, sym, np.array([x[idx], y[idx], z_ax[idx]]))
            traces.append(go.Scatter3d(
                x=lx, y=ly, z=lz, mode="lines", name=f"tree chart {c}",
                line={"color": col, "width": line_width},
                hoverinfo="none", showlegend=False,
            ))
    else:  # "symbol"
        for c in unique_charts:
            col_chart = _CATEGORY10[int(c) % len(_CATEGORY10)]
            # root → chart edge (chart color)
            lx, ly, lz = [], [], []
            _seg(lx, ly, lz, root, chart_centers[c])
            traces.append(go.Scatter3d(
                x=lx, y=ly, z=lz, mode="lines",
                line={"color": col_chart, "width": line_width},
                hoverinfo="none", showlegend=False,
            ))
            c_mask = charts == c
            for k in np.unique(codes[c_mask]):
                sym = symbol_centers[(c, k)]
                # chart → symbol (chart color)
                lx, ly, lz = [], [], []
                _seg(lx, ly, lz, chart_centers[c], sym)
                traces.append(go.Scatter3d(
                    x=lx, y=ly, z=lz, mode="lines",
                    line={"color": col_chart, "width": line_width},
                    hoverinfo="none", showlegend=False,
                ))
                # symbol → data (per-symbol color)
                sym_col = _CATEGORY10[int(k) % len(_CATEGORY10)]
                lx, ly, lz = [], [], []
                for idx in np.where(c_mask & (codes == k))[0]:
                    _seg(lx, ly, lz, sym, np.array([x[idx], y[idx], z_ax[idx]]))
                if lx:
                    traces.append(go.Scatter3d(
                        x=lx, y=ly, z=lz, mode="lines",
                        line={"color": sym_col, "width": line_width},
                        hoverinfo="none", showlegend=False,
                    ))


def plot_prob_bars(
    probs: np.ndarray,
    true_label: int,
    model_name: str,
    num_classes: int = 10,
) -> hv.Bars:
    """Per-class probability bar chart for a single sample.

    Bars are colored green for the true label, blue for the predicted class
    (when correct), and red for an incorrect prediction.
    """
    predicted = int(np.argmax(probs))
    colors = []
    for c in range(num_classes):
        if c == true_label and c == predicted:
            colors.append("#54a24b")  # green – correct prediction
        elif c == predicted:
            colors.append("#e45756")  # red – wrong prediction
        elif c == true_label:
            colors.append("#54a24b")  # green – true label
        else:
            colors.append("#4c78a8")  # blue – other
    data = [(str(c), float(probs[c]), col) for c, col in enumerate(colors)]
    bars = hv.Bars(data, kdims=["Class"], vdims=["Probability", "color"])
    return bars.redim.range(Probability=(0, 1)).opts(
        title=f"{model_name} → {predicted}",
        width=250,
        height=200,
        color="color",
        xlabel="Class",
        ylabel="P",
    )


def _seg(
    lx: list, ly: list, lz: list,
    a: np.ndarray, b: np.ndarray,
) -> None:
    """Append a line segment with None separator for Plotly."""
    lx.extend([a[0], b[0], None])
    ly.extend([a[1], b[1], None])
    lz.extend([a[2], b[2], None])
