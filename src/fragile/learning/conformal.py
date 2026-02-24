"""Conformal prediction and geometric calibration analysis for TopoEncoder.

Pure computation + HoloViews plotting functions (no Panel state/widgets).
All functions take numpy arrays and return results or plot objects.
"""

from __future__ import annotations

import math

import holoviews as hv
import numpy as np


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for binomial proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def conformal_factor_np(z_geo: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Poincare conformal factor lambda(z) = 2 / (1 - |z|^2)."""
    r_sq = np.sum(z_geo**2, axis=1)
    r_sq = np.clip(r_sq, 0.0, 1.0 - eps)
    return 2.0 / (1.0 - r_sq)


def router_entropy(router_weights: np.ndarray) -> np.ndarray:
    """Per-sample entropy H = -sum(w * log(w)) of router weight distribution."""
    w = np.clip(router_weights, 1e-12, None)
    return -np.sum(w * np.log(w), axis=1)


def compute_tunneling_rate(
    router_weights: np.ndarray,
    chart_to_class: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Cross-class tunneling rate from router mass over chart-class pairs.

    Args:
        router_weights: Routing weights ``[N, N_c]`` from atlas encoder.
        chart_to_class: Majority class per chart ``[N_c]``.
        num_classes: Number of classes.

    Returns:
        Cross-class tunneling score for each sample in ``[N]``.
    """
    if len(chart_to_class) == 0:
        return np.zeros(len(router_weights), dtype=float)

    w = np.clip(router_weights, 0.0, 1.0)
    classes = np.asarray(chart_to_class, dtype=int)
    _ = num_classes
    if w.shape[1] > len(classes):
        classes = np.pad(classes, (0, w.shape[1] - len(classes)), constant_values=0)

    cross_class = (classes[:, None] != classes[None, :]).astype(float)
    return np.einsum("bi,ij,bj->b", w, cross_class, w)


def euclidean_isolation(
    z_geo: np.ndarray,
    codebook: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Distance to nearest codebook prototype in latent Euclidean space.

    This is the minimum Euclidean distance to any codebook atom.
    """
    if codebook.size == 0:
        return np.zeros(len(z_geo), dtype=float)

    x = np.asarray(z_geo, dtype=float)
    code = np.asarray(codebook, dtype=float).reshape(-1, x.shape[1])
    diff = x[:, None, :] - code[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return np.min(dist, axis=1)


def hyperbolic_knn_density(
    z_geo: np.ndarray,
    z_geo_train: np.ndarray,
    k: int = 10,
    eps: float = 1e-7,
) -> np.ndarray:
    """k-NN density proxy from hyperbolic Poincaré distance.

    Returns the distance to the k-th nearest neighbor in the training set for each
    sample. Larger values indicate lower local density (more outlier-like).
    """
    x = np.asarray(z_geo, dtype=np.float32)
    train = np.asarray(z_geo_train, dtype=np.float32)
    if x.size == 0 or train.size == 0:
        return np.zeros(len(x), dtype=float)

    n_train = train.shape[0]
    k = int(k)
    if n_train <= 1:
        return np.zeros(len(x), dtype=float)
    k = max(1, min(k, n_train - 1))

    x_norm2 = np.clip(np.sum(x * x, axis=1), 0.0, 1.0 - eps)
    t_norm2 = np.clip(np.sum(train * train, axis=1), 0.0, 1.0 - eps)
    diff = x[:, None, :] - train[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    denom = (1.0 - x_norm2)[:, None] * (1.0 - t_norm2)[None, :]
    arg = 1.0 + 2.0 * dist_sq / np.clip(denom, eps, None)
    dist = np.arccosh(np.clip(arg, 1.0 + eps, None))
    return np.partition(dist, k, axis=1)[:, k]


def geodesic_isolation(
    z_geo: np.ndarray,
    codebook: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Backward-compatible alias for ``euclidean_isolation``.

    Historically this function used a Poincaré-geodesic distance, but for OOD scoring
    this should be radius-independent to avoid confounding with ``lambda``.
    """
    _ = eps
    return euclidean_isolation(z_geo, codebook)


def radial_bins(z_geo: np.ndarray, n_bins: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Assign samples to radial shells by |z|.

    Returns (bin_indices [N], bin_edges [n_bins+1]).
    """
    radii = np.linalg.norm(z_geo, axis=1)
    bin_edges = np.linspace(0.0, radii.max() + 1e-8, n_bins + 1)
    bin_idx = np.digitize(radii, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    return bin_idx, bin_edges


def calibration_test_split(
    n: int, cal_frac: float = 0.5, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic calibration / test index split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_cal = int(n * cal_frac)
    return perm[:n_cal], perm[n_cal:]


def forward_pass_batch(
    loaded, X_input, batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched encoder + classifier on arbitrary input tensor.

    Args:
        loaded: LoadedModels instance (passed explicitly per module-level pattern).
        X_input: Input tensor [N, input_dim].
        batch_size: Batch size for inference.

    Returns:
        (z_geo, router_weights, probs, charts, codes) as numpy arrays.
        probs is None if no classifier head.
    """
    import torch
    import torch.nn.functional as F

    from fragile.learning.plots import _to_numpy

    z_parts, router_parts, prob_parts, chart_parts, code_parts = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(X_input), batch_size):
            batch = X_input[i : i + batch_size].float()
            enc_out = loaded.model_atlas.encoder(batch)
            z_parts.append(_to_numpy(enc_out[5]))
            router_parts.append(_to_numpy(enc_out[4]))
            chart_parts.append(_to_numpy(enc_out[0]))
            code_parts.append(_to_numpy(enc_out[1]))
            if loaded.classifier_head is not None:
                logits = loaded.classifier_head(enc_out[4], enc_out[5])
                prob_parts.append(_to_numpy(F.softmax(logits, dim=1)))

    z_geo = np.concatenate(z_parts)
    router_w = np.concatenate(router_parts)
    charts = np.concatenate(chart_parts)
    codes = np.concatenate(code_parts)
    probs = np.concatenate(prob_parts) if prob_parts else None
    return z_geo, router_w, probs, charts, codes


# ---------------------------------------------------------------------------
# Analysis 1: Accuracy vs Radius
# ---------------------------------------------------------------------------


def accuracy_vs_radius(
    z_geo: np.ndarray, correct: np.ndarray, n_bins: int = 15,
) -> dict:
    """Compute accuracy in radial shells with Wilson CIs."""
    bin_idx, bin_edges = radial_bins(z_geo, n_bins)
    centers, accs, ci_lo, ci_hi, counts = [], [], [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        k = int(correct[mask].sum())
        lo, hi = wilson_ci(k, n)
        centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
        accs.append(k / n)
        ci_lo.append(lo)
        ci_hi.append(hi)
        counts.append(n)
    return {
        "bin_centers": np.array(centers),
        "accuracy": np.array(accs),
        "ci_lower": np.array(ci_lo),
        "ci_upper": np.array(ci_hi),
        "counts": np.array(counts),
    }


def plot_accuracy_vs_radius(data: dict) -> hv.Overlay:
    """Accuracy vs radius curve with confidence band."""
    c = data["bin_centers"]
    acc = data["accuracy"]
    lo = data["ci_lower"]
    hi = data["ci_upper"]

    curve = hv.Curve((c, acc), "Radius |z|", "Accuracy").opts(
        color="#4c78a8", line_width=2, tools=["hover"],
    )
    band = hv.Area((c, lo, hi), "Radius |z|", ["CI lower", "CI upper"]).opts(
        color="#4c78a8", alpha=0.2,
    )
    return (band * curve).opts(
        title="Accuracy vs Hyperbolic Radius", width=500, height=350,
    )


# ---------------------------------------------------------------------------
# Analysis 2: Marginal Coverage (3 conformal methods)
# ---------------------------------------------------------------------------


def conformal_scores_standard(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Standard conformal score: 1 - p_hat(y_true)."""
    return 1.0 - probs[np.arange(len(labels)), labels]


def conformal_scores_geodesic(
    probs: np.ndarray, labels: np.ndarray, z_geo: np.ndarray,
) -> np.ndarray:
    """Geodesic-weighted conformal score: (1 - p_hat(y)) / lambda(z)."""
    lam = conformal_factor_np(z_geo)
    return (1.0 - probs[np.arange(len(labels)), labels]) / lam


def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """Conformal quantile q_hat at ceil((1-alpha)(1+1/n))-th quantile."""
    n = len(cal_scores)
    level = math.ceil((1 - alpha) * (n + 1)) / n
    level = min(level, 1.0)
    return float(np.quantile(cal_scores, level))


def conformal_quantiles_per_chart(
    cal_scores: np.ndarray, cal_charts: np.ndarray, alpha: float,
    min_samples: int = 20,
) -> dict[int, float]:
    """Per-chart conformal quantiles with fallback to global for small charts."""
    global_q = conformal_quantile(cal_scores, alpha)
    chart_ids = np.unique(cal_charts)
    result = {}
    for c in chart_ids:
        mask = cal_charts == c
        if mask.sum() < min_samples:
            result[int(c)] = global_q
        else:
            result[int(c)] = conformal_quantile(cal_scores[mask], alpha)
    return result


def conformal_quantiles_per_chart_code(
    cal_scores: np.ndarray,
    cal_charts: np.ndarray,
    cal_codes: np.ndarray,
    alpha: float,
    min_samples: int = 20,
) -> tuple[dict[tuple[int, int], float], dict]:
    """Per-(chart, code) conformal quantiles with hierarchical fallback.

    Fallback chain: (chart, code) -> chart -> global.
    Returns (quantile_dict, stats_dict) where stats has group sizes and fallback counts.
    """
    global_q = conformal_quantile(cal_scores, alpha)
    chart_qs = conformal_quantiles_per_chart(cal_scores, cal_charts, alpha, min_samples)

    # Find unique (chart, code) pairs
    pairs = np.column_stack([cal_charts, cal_codes])
    unique_pairs = np.unique(pairs, axis=0)

    result = {}
    n_fine = 0
    n_chart_fallback = 0
    n_global_fallback = 0
    min_group_size = len(cal_scores)

    for chart, code in unique_pairs:
        chart, code = int(chart), int(code)
        mask = (cal_charts == chart) & (cal_codes == code)
        group_size = int(mask.sum())
        min_group_size = min(min_group_size, group_size)

        if group_size >= min_samples:
            result[(chart, code)] = conformal_quantile(cal_scores[mask], alpha)
            n_fine += 1
        elif chart in chart_qs:
            result[(chart, code)] = chart_qs[chart]
            n_chart_fallback += 1
        else:
            result[(chart, code)] = global_q
            n_global_fallback += 1

    stats = {
        "n_groups": len(unique_pairs),
        "n_fine": n_fine,
        "n_chart_fallback": n_chart_fallback,
        "n_global_fallback": n_global_fallback,
        "min_group_size": min_group_size,
    }
    return result, stats


def conformal_quantiles_per_radius(
    cal_scores: np.ndarray,
    cal_z_geo: np.ndarray,
    alpha: float,
    n_shells: int = 10,
    min_samples: int = 30,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-radial-shell conformal quantiles with fallback to global.

    Args:
        cal_scores: Calibration nonconformity scores.
        cal_z_geo: Calibration latent coordinates [N, D].
        alpha: Miscoverage rate.
        n_shells: Number of equal-width radial shells.
        min_samples: Minimum samples per shell; smaller shells fall back to global.

    Returns:
        (bin_edges [n_shells+1], shell_quantiles [n_shells], stats_dict).
    """
    global_q = conformal_quantile(cal_scores, alpha)
    radii = np.linalg.norm(cal_z_geo, axis=1)
    bin_edges = np.linspace(0.0, radii.max() + 1e-8, n_shells + 1)
    bin_idx = np.clip(np.digitize(radii, bin_edges) - 1, 0, n_shells - 1)

    shell_quantiles = np.full(n_shells, global_q)
    n_fine = 0
    n_fallback = 0
    min_group_size = len(cal_scores)

    for b in range(n_shells):
        mask = bin_idx == b
        group_size = int(mask.sum())
        if group_size > 0:
            min_group_size = min(min_group_size, group_size)
        if group_size >= min_samples:
            shell_quantiles[b] = conformal_quantile(cal_scores[mask], alpha)
            n_fine += 1
        else:
            n_fallback += 1

    stats = {
        "n_shells": n_shells,
        "n_fine": n_fine,
        "n_fallback": n_fallback,
        "min_group_size": min_group_size,
    }
    return bin_edges, shell_quantiles, stats


def conformal_scores_geo_beta(
    probs: np.ndarray,
    labels: np.ndarray,
    z_geo: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Composite conformal score: (1 - p_hat(y)) * (1 + beta / lambda(z)).

    At the center (low lambda~2), the penalty is ~(1 + beta/2) — large.
    At the boundary (high lambda), the penalty is ~1 — negligible.
    """
    lam = conformal_factor_np(z_geo)
    base = 1.0 - probs[np.arange(len(labels)), labels]
    return base * (1.0 + beta / lam)


def tune_conformal_beta(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_z_geo: np.ndarray,
    alpha: float,
    beta_range: tuple[float, float] = (0.0, 5.0),
    n_grid: int = 51,
) -> float:
    """Find beta that minimizes mean set size on calibration set.

    The coverage guarantee holds for any score function, so we are free to
    choose the beta that produces the tightest sets.
    """
    best_beta = 0.0
    best_mss = float("inf")
    for beta in np.linspace(*beta_range, n_grid):
        scores = conformal_scores_geo_beta(cal_probs, cal_labels, cal_z_geo, beta)
        q = conformal_quantile(scores, alpha)
        incl, sizes = prediction_sets(cal_probs, q, "geo_beta", z_geo=cal_z_geo, geo_beta=beta)
        mss = float(sizes.mean())
        if mss < best_mss:
            best_mss = mss
            best_beta = float(beta)
    return best_beta


def prediction_sets(
    probs: np.ndarray,
    q_hat: float,
    method: str,
    z_geo: np.ndarray | None = None,
    charts: np.ndarray | None = None,
    chart_quantiles: dict[int, float] | None = None,
    codes: np.ndarray | None = None,
    chart_code_quantiles: dict[tuple[int, int], float] | None = None,
    radial_edges: np.ndarray | None = None,
    radial_quantiles: np.ndarray | None = None,
    geo_beta: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build prediction sets.

    Args:
        probs: [N, C] softmax probabilities.
        q_hat: Global conformal quantile.
        method: "standard", "geodesic", "chart", "chart_code", "radial",
                or "geo_beta".
        z_geo: Required for geodesic/chart/chart_code/radial/geo_beta methods.
        charts: Required for "chart" and "chart_code" methods.
        chart_quantiles: Required for "chart" method.
        codes: Required for "chart_code" method.
        chart_code_quantiles: Required for "chart_code" method.
        radial_edges: Bin edges [n_shells+1] for "radial" method.
        radial_quantiles: Per-shell quantiles [n_shells] for "radial" method.
        geo_beta: Beta parameter for "geo_beta" composite score method.

    Returns:
        (included [N, C] bool mask, set_sizes [N]).
    """
    n, c = probs.shape
    scores_per_class = 1.0 - probs  # score for each class

    if method == "geo_beta" and z_geo is not None and geo_beta is not None:
        # Composite score: (1 - p_c) * (1 + beta / lambda)
        lam = conformal_factor_np(z_geo)
        scores_per_class = scores_per_class * (1.0 + geo_beta / lam[:, None])
        included = scores_per_class <= q_hat
    elif method in ("geodesic", "chart", "chart_code", "radial") and z_geo is not None:
        lam = conformal_factor_np(z_geo)
        scores_per_class = scores_per_class / lam[:, None]

        if method == "chart" and charts is not None and chart_quantiles is not None:
            thresholds = np.array([chart_quantiles.get(int(ch), q_hat) for ch in charts])
            included = scores_per_class <= thresholds[:, None]
        elif (
            method == "chart_code"
            and charts is not None
            and codes is not None
            and chart_code_quantiles is not None
        ):
            thresholds = np.array([
                chart_code_quantiles.get((int(ch), int(co)), q_hat)
                for ch, co in zip(charts, codes)
            ])
            included = scores_per_class <= thresholds[:, None]
        elif (
            method == "radial"
            and radial_edges is not None
            and radial_quantiles is not None
        ):
            radii = np.linalg.norm(z_geo, axis=1)
            n_shells = len(radial_quantiles)
            bin_idx = np.clip(np.digitize(radii, radial_edges) - 1, 0, n_shells - 1)
            thresholds = radial_quantiles[bin_idx]
            included = scores_per_class <= thresholds[:, None]
        else:
            included = scores_per_class <= q_hat
    else:
        included = scores_per_class <= q_hat

    set_sizes = included.sum(axis=1)
    # Ensure at least the top class is always included
    empty = set_sizes == 0
    if empty.any():
        top_class = probs[empty].argmax(axis=1)
        included[np.where(empty)[0], top_class] = True
        set_sizes = included.sum(axis=1)
    return included, set_sizes


def evaluate_coverage(
    included: np.ndarray, labels: np.ndarray,
) -> tuple[float, float]:
    """Marginal coverage and mean set size."""
    coverage = float(included[np.arange(len(labels)), labels].mean())
    mean_size = float(included.sum(axis=1).mean())
    return coverage, mean_size


def format_coverage_summary_table(results: dict, alpha: float) -> str:
    """Format coverage results as markdown table.

    Args:
        results: {method_name: (coverage, mean_set_size)}
        alpha: Nominal miscoverage rate.
    """
    lines = [
        f"**Target coverage: {1 - alpha:.0%}**\n",
        "| Method | Coverage | Mean Set Size |",
        "|--------|----------|---------------|",
    ]
    for method, (cov, mss) in results.items():
        lines.append(f"| {method} | {cov:.1%} | {mss:.2f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis 3 & 4: Conditional Coverage & Set Size by Radius
# ---------------------------------------------------------------------------


def conditional_coverage_by_radius(
    pred_sets_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    z_geo: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Per-bin coverage and set size for each method.

    Args:
        pred_sets_dict: {method: (included [N,C], set_sizes [N])}
    """
    bin_idx, bin_edges = radial_bins(z_geo, n_bins)
    centers = [(bin_edges[b] + bin_edges[b + 1]) / 2 for b in range(n_bins)]

    data = {"bin_centers": [], "methods": {}}
    valid_centers = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        valid_centers.append(centers[b])
        for method, (included, sizes) in pred_sets_dict.items():
            if method not in data["methods"]:
                data["methods"][method] = {"coverage": [], "set_size": []}
            cov = float(included[mask][np.arange(mask.sum()), labels[mask]].mean())
            ss = float(sizes[mask].mean())
            data["methods"][method]["coverage"].append(cov)
            data["methods"][method]["set_size"].append(ss)
    data["bin_centers"] = np.array(valid_centers)
    for m in data["methods"]:
        data["methods"][m]["coverage"] = np.array(data["methods"][m]["coverage"])
        data["methods"][m]["set_size"] = np.array(data["methods"][m]["set_size"])
    return data


def plot_conditional_coverage(data: dict, alpha: float) -> hv.Overlay:
    """One curve per method + target coverage line."""
    centers = data["bin_centers"]
    colors = {
        "Standard": "#e45756", "Geodesic": "#4c78a8",
        "Chart": "#54a24b", "Chart×Code": "#b07aa1",
        "Radial": "#f58518", "Geo-β": "#72b7b2",
    }
    plots = []
    for method, vals in data["methods"].items():
        c = colors.get(method, "#888888")
        plots.append(
            hv.Curve((centers, vals["coverage"]), "Radius |z|", "Coverage").opts(
                color=c, line_width=2, tools=["hover"],
            ).relabel(method)
        )
    target = hv.HLine(1 - alpha).opts(color="black", line_dash="dashed", line_width=1)
    plots.append(target)
    return hv.Overlay(plots).opts(
        title="Conditional Coverage by Radius",
        width=500, height=350, legend_position="bottom_left",
    )


def plot_set_size_vs_radius(data: dict) -> hv.Overlay:
    """Mean prediction set size vs radius for each method."""
    centers = data["bin_centers"]
    colors = {
        "Standard": "#e45756", "Geodesic": "#4c78a8",
        "Chart": "#54a24b", "Chart×Code": "#b07aa1",
        "Radial": "#f58518", "Geo-β": "#72b7b2",
    }
    plots = []
    for method, vals in data["methods"].items():
        c = colors.get(method, "#888888")
        plots.append(
            hv.Curve((centers, vals["set_size"]), "Radius |z|", "Mean Set Size").opts(
                color=c, line_width=2, tools=["hover"],
            ).relabel(method)
        )
    return hv.Overlay(plots).opts(
        title="Prediction Set Size by Radius",
        width=500, height=350, legend_position="top_left",
    )


# ---------------------------------------------------------------------------
# Analysis 5: Coverage by Class (Mondrian)
# ---------------------------------------------------------------------------


def conformal_quantiles_per_class(
    cal_scores: np.ndarray, cal_labels: np.ndarray, alpha: float,
    min_samples: int = 20,
) -> dict[int, float]:
    """Per-class conformal quantiles (Mondrian conformal)."""
    global_q = conformal_quantile(cal_scores, alpha)
    classes = np.unique(cal_labels)
    result = {}
    for cls in classes:
        mask = cal_labels == cls
        if mask.sum() < min_samples:
            result[int(cls)] = global_q
        else:
            result[int(cls)] = conformal_quantile(cal_scores[mask], alpha)
    return result


def prediction_sets_mondrian(
    probs: np.ndarray, class_quantiles: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Mondrian prediction sets using per-class quantiles."""
    n, c = probs.shape
    scores = 1.0 - probs
    thresholds = np.array([class_quantiles.get(cls, 0.5) for cls in range(c)])
    included = scores <= thresholds[None, :]
    set_sizes = included.sum(axis=1)
    # Ensure at least top class
    empty = set_sizes == 0
    if empty.any():
        top_class = probs[empty].argmax(axis=1)
        included[np.where(empty)[0], top_class] = True
        set_sizes = included.sum(axis=1)
    return included, set_sizes


def coverage_by_class(
    pred_sets_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    num_classes: int = 10,
) -> dict:
    """Per-class coverage and mean set size for each method."""
    data = {}
    for method, (included, sizes) in pred_sets_dict.items():
        class_data = []
        for cls in range(num_classes):
            mask = labels == cls
            if mask.sum() == 0:
                class_data.append({"coverage": np.nan, "set_size": np.nan, "count": 0})
                continue
            cov = float(included[mask][:, cls].mean())
            ss = float(sizes[mask].mean())
            class_data.append({"coverage": cov, "set_size": ss, "count": int(mask.sum())})
        data[method] = class_data
    return data


def format_class_coverage_table(data: dict) -> str:
    """Format per-class coverage as markdown table."""
    methods = list(data.keys())
    lines = ["| Class | " + " | ".join(f"{m} Cov" for m in methods) + " | Count |"]
    lines.append("|" + "---|" * (len(methods) + 2))
    # Use first method to get class count
    n_classes = len(next(iter(data.values())))
    for cls in range(n_classes):
        row = [f"| {cls} "]
        for m in methods:
            cov = data[m][cls]["coverage"]
            row.append(f"| {cov:.1%} " if not np.isnan(cov) else "| — ")
        count = data[methods[0]][cls]["count"]
        row.append(f"| {count} |")
        lines.append("".join(row))
    return "\n".join(lines)


def format_coverage_method_comparison(
    method_specs: dict[str, dict],
    class_coverage_data: dict[str, list[dict]],
    radius_coverage_data: dict | None = None,
) -> str:
    """Format a comparison table that highlights coverage method structure.

    Args:
        method_specs: Ordered mapping where key is the display method name and value
            has ``conditions``, ``groups`` and ``needs_labels`` fields.
            Optional ``class_coverage_key`` can map to a column key in
            ``class_coverage_data`` when it differs from the display name.
            Optional ``radius_coverage_key`` can map to a method key in
            ``radius_coverage_data`` when it differs from the display name.
        class_coverage_data: Output from ``coverage_by_class`` with per-class
            coverage dictionaries.
        radius_coverage_data: Output from ``conditional_coverage_by_radius`` with
            per-radius coverage data.
    """
    lines = [
        "| Method | Conditions on | Groups | Needs labels? | Worst-class gap | Worst-radius gap |",
        "|--------|---------------|--------|---------------|-----------------|------------------|",
    ]
    radial_methods = radius_coverage_data.get("methods", {}) if radius_coverage_data else {}

    for name, spec in method_specs.items():
        keyspec = spec.get("class_coverage_key", name)
        cov_rows = class_coverage_data.get(keyspec, [])
        vals = [
            item.get("coverage", np.nan)
            for item in cov_rows
            if not np.isnan(item.get("coverage", np.nan))
        ]
        if vals:
            gap = float(np.nanmax(vals) - np.nanmin(vals))
            class_gap_str = f"{gap:.1%}"
        else:
            class_gap_str = "—"

        radius_key = spec.get("radius_coverage_key", name)
        rad_rows = radial_methods.get(radius_key, {}).get("coverage", [])
        if len(rad_rows) >= 2:
            rad_vals = [float(v) for v in rad_rows if not np.isnan(float(v))]
            if len(rad_vals) >= 2:
                rad_gap = float(np.nanmax(rad_vals) - np.nanmin(rad_vals))
                radius_gap_str = f"{rad_gap:.1%}"
            else:
                radius_gap_str = "—"
        elif len(rad_rows) == 1:
            radius_gap_str = "—"
        else:
            radius_gap_str = "—"

        needs = spec.get("needs_labels", False)
        needs_str = "Yes" if needs else "No"
        lines.append(
            f"| {name} | {spec['conditions']} | {spec['groups']} | {needs_str} | {class_gap_str} | {radius_gap_str} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis 6: OOD Detection
# ---------------------------------------------------------------------------


def load_fashion_mnist(
    n_samples: int = 10000, root: str = "./data",
) -> tuple[np.ndarray, np.ndarray]:
    """Load Fashion-MNIST test set as flat numpy arrays [N, 784]."""
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    n = min(n_samples, len(ds))
    images = ds.data[:n].float().reshape(n, -1) / 255.0
    labels = ds.targets[:n].numpy()
    return images.numpy(), labels


def cross_class_router_overlap(
    router_weights: np.ndarray, charts: np.ndarray, num_charts: int,
) -> np.ndarray:
    """Per-sample overlap score: how much router distribution differs from chart mean."""
    # Compute per-chart mean router distribution
    chart_means = np.zeros((num_charts, router_weights.shape[1]))
    for c in range(num_charts):
        mask = charts == c
        if mask.sum() > 0:
            chart_means[c] = router_weights[mask].mean(axis=0)
    # KL divergence from sample to its chart mean
    overlap = np.zeros(len(router_weights))
    for i in range(len(router_weights)):
        p = np.clip(router_weights[i], 1e-12, None)
        q = np.clip(chart_means[charts[i]], 1e-12, None)
        overlap[i] = np.sum(p * np.log(p / q))
    return overlap


def ood_scores(
    id_signals: dict[str, np.ndarray], ood_signals: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute AUROC for each OOD signal (higher = OOD)."""
    from sklearn.metrics import roc_auc_score

    aurocs = {}
    for name in id_signals:
        if name not in ood_signals:
            continue
        y_true = np.concatenate([np.zeros(len(id_signals[name])), np.ones(len(ood_signals[name]))])
        y_score = np.concatenate([id_signals[name], ood_signals[name]])
        if len(np.unique(y_true)) < 2:
            continue
        aurocs[name] = float(roc_auc_score(y_true, y_score))
    return aurocs


def plot_ood_roc(
    id_signals: dict[str, np.ndarray], ood_signals: dict[str, np.ndarray],
) -> hv.Overlay:
    """ROC curves for OOD detection signals."""
    from sklearn.metrics import roc_curve

    colors = {
        "1 - max_prob": "#e45756",
        "1/lambda": "#4c78a8",
        "router_entropy": "#54a24b",
        "router_overlap": "#f58518",
        "tunneling_rate": "#b07aa1",
        "geodesic_isolation": "#9b59b6",
        "hyperbolic_knn_density": "#e8a838",
        "combined": "#34495e",
    }
    plots = []
    for name in id_signals:
        if name not in ood_signals:
            continue
        y_true = np.concatenate([np.zeros(len(id_signals[name])), np.ones(len(ood_signals[name]))])
        y_score = np.concatenate([id_signals[name], ood_signals[name]])
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        c = colors.get(name, "#888888")
        plots.append(
            hv.Curve((fpr, tpr), "FPR", "TPR").opts(
                color=c, line_width=2,
            ).relabel(name)
        )
    diag = hv.Curve(([0, 1], [0, 1]), "FPR", "TPR").opts(
        color="gray", line_dash="dashed", line_width=1,
    )
    plots.append(diag)
    return hv.Overlay(plots).opts(
        title="OOD Detection ROC", width=500, height=400,
        legend_position="bottom_right",
    )


def format_ood_auroc_table(aurocs: dict[str, float]) -> str:
    """Format AUROC scores as markdown table."""
    lines = [
        "| Signal | AUROC |",
        "|--------|-------|",
    ]
    for name, auc in sorted(aurocs.items(), key=lambda x: -x[1]):
        lines.append(f"| {name} | {auc:.3f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis 7: Calibration Curve (Reliability Diagram)
# ---------------------------------------------------------------------------


def reliability_diagram_data(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15,
) -> dict:
    """Compute reliability diagram data."""
    # Use max prob and whether top prediction is correct
    p_max = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, observed, counts = [], [], []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (p_max >= lo) & (p_max < hi) if b < n_bins - 1 else (p_max >= lo) & (p_max <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        centers.append((lo + hi) / 2)
        observed.append(float(correct[mask].mean()))
        counts.append(n)
    return {
        "bin_centers": np.array(centers),
        "observed_freq": np.array(observed),
        "counts": np.array(counts),
    }


def recalibrate_probs(
    probs: np.ndarray, z_geo: np.ndarray, beta: float,
) -> np.ndarray:
    """Recalibrate: p_recal = p * lambda(z)^beta, then renormalize."""
    lam = conformal_factor_np(z_geo)
    scaled = probs * (lam[:, None] ** beta)
    row_sums = scaled.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    return scaled / row_sums


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE)."""
    p_max = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(probs)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (p_max >= lo) & (p_max < hi) if b < n_bins - 1 else (p_max >= lo) & (p_max <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        avg_conf = float(p_max[mask].mean())
        avg_acc = float(correct[mask].mean())
        ece += (n / n_total) * abs(avg_conf - avg_acc)
    return ece


def tune_beta(
    cal_probs: np.ndarray, cal_labels: np.ndarray, cal_z_geo: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Find beta that minimizes ECE on calibration set."""
    best_beta = 0.0
    best_ece = expected_calibration_error(cal_probs, cal_labels, n_bins)
    for beta in np.linspace(-1.0, 2.0, 61):
        recal = recalibrate_probs(cal_probs, cal_z_geo, beta)
        ece = expected_calibration_error(recal, cal_labels, n_bins)
        if ece < best_ece:
            best_ece = ece
            best_beta = float(beta)
    return best_beta


def plot_reliability_diagram(raw_data: dict, recal_data: dict | None = None) -> hv.Overlay:
    """Reliability diagram: raw + recalibrated curves vs perfect diagonal."""
    diag = hv.Curve(([0, 1], [0, 1]), "Mean Predicted Prob", "Observed Frequency").opts(
        color="gray", line_dash="dashed", line_width=1,
    )
    raw_curve = hv.Curve(
        (raw_data["bin_centers"], raw_data["observed_freq"]),
        "Mean Predicted Prob", "Observed Frequency",
    ).opts(color="#e45756", line_width=2, tools=["hover"]).relabel("Raw")

    plots = [diag, raw_curve]
    if recal_data is not None:
        recal_curve = hv.Curve(
            (recal_data["bin_centers"], recal_data["observed_freq"]),
            "Mean Predicted Prob", "Observed Frequency",
        ).opts(color="#4c78a8", line_width=2, tools=["hover"]).relabel("Recalibrated")
        plots.append(recal_curve)
    return hv.Overlay(plots).opts(
        title="Reliability Diagram", width=500, height=350,
        legend_position="bottom_right",
    )


# ---------------------------------------------------------------------------
# Analysis 8: Ablation (Feature Importance)
# ---------------------------------------------------------------------------


def ablation_feature_importance(
    correct: np.ndarray,
    p_max: np.ndarray,
    radius: np.ndarray,
    v_h: np.ndarray,
) -> dict:
    """Logistic regression to assess feature importance for correctness prediction.

    Args:
        correct: [N] binary correctness.
        p_max: [N] max softmax probability.
        radius: [N] hyperbolic radius |z|.
        v_h: [N] router entropy.

    Returns:
        dict with coefficients, importances (abs coeff), and AUC.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    features = np.column_stack([p_max, radius, v_h])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, correct)

    feature_names = ["max_prob", "radius", "router_entropy"]
    coeffs = model.coef_[0]
    importances = np.abs(coeffs)
    proba = model.predict_proba(X_scaled)[:, 1]
    auc = float(roc_auc_score(correct, proba))

    return {
        "feature_names": feature_names,
        "coefficients": coeffs.tolist(),
        "importances": importances.tolist(),
        "auc": auc,
    }


def format_ablation_table(data: dict) -> str:
    """Format ablation results as markdown."""
    lines = [
        f"**Logistic Regression AUC: {data['auc']:.3f}**\n",
        "| Feature | Coefficient | Importance |",
        "|--------|-------------|------------|",
    ]
    rows = sorted(
        zip(data["feature_names"], data["coefficients"], data["importances"]),
        key=lambda t: abs(t[2]),
        reverse=True,
    )
    for name, coeff, imp in rows:
        lines.append(f"| {name} | {coeff:+.3f} | {imp:.3f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis 9: Corruption Robustness
# ---------------------------------------------------------------------------


def corrupt_data(
    X: np.ndarray, corruption: str, intensity: float, seed: int = 42,
) -> np.ndarray:
    """Apply corruption to flat image data [N, D].

    Supported corruptions: "gaussian_noise", "rotation", "blur".
    """
    rng = np.random.default_rng(seed)
    X_out = X.copy()

    if corruption == "gaussian_noise":
        noise = rng.normal(0, intensity, size=X.shape)
        X_out = np.clip(X + noise, 0, 1)

    elif corruption == "rotation":
        from scipy.ndimage import rotate as scipy_rotate

        side = int(np.sqrt(X.shape[1]))
        for i in range(len(X_out)):
            img = X_out[i].reshape(side, side)
            angle = rng.uniform(-intensity * 45, intensity * 45)
            img = scipy_rotate(img, angle, reshape=False, mode="constant", cval=0)
            X_out[i] = img.flatten()

    elif corruption == "blur":
        from scipy.ndimage import gaussian_filter

        side = int(np.sqrt(X.shape[1]))
        sigma = intensity * 2.0
        for i in range(len(X_out)):
            img = X_out[i].reshape(side, side)
            img = gaussian_filter(img, sigma=sigma)
            X_out[i] = img.flatten()

    return X_out.astype(np.float32)


def coverage_under_corruption(
    loaded,
    X_np: np.ndarray,
    labels: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
    n_intensities: int = 5,
    corruptions: tuple[str, ...] = ("gaussian_noise", "rotation", "blur"),
) -> dict:
    """Compute coverage for each corruption at multiple intensities.

    Returns:
        {corruption: {method: coverage_array [n_intensities]}}
    """
    import torch

    intensities = np.linspace(0.0, 1.0, n_intensities + 1)[1:]  # skip 0

    # Get clean calibration scores
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    z_geo, router_w, probs, charts, codes = forward_pass_batch(loaded, X_tensor)
    if probs is None:
        return {}

    cal_scores_std = conformal_scores_standard(probs[cal_idx], labels[cal_idx])
    cal_scores_geo = conformal_scores_geodesic(probs[cal_idx], labels[cal_idx], z_geo[cal_idx])
    q_std = conformal_quantile(cal_scores_std, alpha)
    q_geo = conformal_quantile(cal_scores_geo, alpha)

    # Chart×Code quantiles for corruption robustness
    chart_code_qs, _ = conformal_quantiles_per_chart_code(
        cal_scores_geo, charts[cal_idx], codes[cal_idx], alpha,
    )

    # Radial shell quantiles
    rad_edges, rad_qs, _ = conformal_quantiles_per_radius(
        cal_scores_geo, z_geo[cal_idx], alpha,
    )

    # Geo-β: tune beta and calibrate
    best_beta = tune_conformal_beta(probs[cal_idx], labels[cal_idx], z_geo[cal_idx], alpha)
    cal_scores_gb = conformal_scores_geo_beta(
        probs[cal_idx], labels[cal_idx], z_geo[cal_idx], best_beta,
    )
    q_gb = conformal_quantile(cal_scores_gb, alpha)

    methods = ("Standard", "Geodesic", "Chart×Code", "Radial", "Geo-β")
    results = {}
    for corr in corruptions:
        results[corr] = {m: [] for m in methods}
        results[corr]["intensities"] = intensities.tolist()
        for intensity in intensities:
            X_corr = corrupt_data(X_np, corr, intensity)
            X_corr_t = torch.tensor(X_corr, dtype=torch.float32)
            z_c, _, probs_c, charts_c, codes_c = forward_pass_batch(loaded, X_corr_t)
            if probs_c is None:
                for m in methods:
                    results[corr][m].append(np.nan)
                continue

            t_idx = test_idx  # alias for brevity

            incl_std, _ = prediction_sets(probs_c[t_idx], q_std, "standard")
            cov_std, _ = evaluate_coverage(incl_std, labels[t_idx])

            incl_geo, _ = prediction_sets(
                probs_c[t_idx], q_geo, "geodesic", z_geo=z_c[t_idx],
            )
            cov_geo, _ = evaluate_coverage(incl_geo, labels[t_idx])

            incl_cc, _ = prediction_sets(
                probs_c[t_idx], q_geo, "chart_code",
                z_geo=z_c[t_idx], charts=charts_c[t_idx],
                codes=codes_c[t_idx], chart_code_quantiles=chart_code_qs,
            )
            cov_cc, _ = evaluate_coverage(incl_cc, labels[t_idx])

            incl_rad, _ = prediction_sets(
                probs_c[t_idx], q_geo, "radial",
                z_geo=z_c[t_idx], radial_edges=rad_edges,
                radial_quantiles=rad_qs,
            )
            cov_rad, _ = evaluate_coverage(incl_rad, labels[t_idx])

            incl_gb, _ = prediction_sets(
                probs_c[t_idx], q_gb, "geo_beta",
                z_geo=z_c[t_idx], geo_beta=best_beta,
            )
            cov_gb, _ = evaluate_coverage(incl_gb, labels[t_idx])

            results[corr]["Standard"].append(cov_std)
            results[corr]["Geodesic"].append(cov_geo)
            results[corr]["Chart×Code"].append(cov_cc)
            results[corr]["Radial"].append(cov_rad)
            results[corr]["Geo-β"].append(cov_gb)

        for m in methods:
            results[corr][m] = np.array(results[corr][m])
    return results


def plot_corruption_coverage(
    data: dict, alpha: float,
) -> hv.Layout:
    """One subplot per corruption type showing coverage vs intensity."""
    panels = []
    for corr, vals in data.items():
        intensities = vals.get("intensities", [])
        if not intensities:
            continue
        x = np.array(intensities)
        plots = []
        for method, color in [
            ("Standard", "#e45756"), ("Geodesic", "#4c78a8"), ("Chart×Code", "#b07aa1"),
            ("Radial", "#f58518"), ("Geo-β", "#72b7b2"),
        ]:
            if method not in vals:
                continue
            y = vals[method]
            plots.append(
                hv.Curve((x, y), "Intensity", "Coverage").opts(
                    color=color, line_width=2, tools=["hover"],
                ).relabel(method)
            )
        target = hv.HLine(1 - alpha).opts(color="black", line_dash="dashed", line_width=1)
        plots.append(target)
        panel = hv.Overlay(plots).opts(
            title=corr.replace("_", " ").title(),
            width=350, height=280, legend_position="bottom_left",
        )
        panels.append(panel)
    if not panels:
        return hv.Div("No corruption data.")
    return hv.Layout(panels).cols(min(3, len(panels)))
