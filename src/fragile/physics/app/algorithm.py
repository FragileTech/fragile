"""Algorithm diagnostics tab for the QFT dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

from fragile.physics.app.qft.plotting import build_lyapunov_plot
from fragile.physics.fractal_gas.history import RunHistory


def _to_numpy(t: Any) -> np.ndarray:
    """Convert a tensor-like to ndarray."""
    if hasattr(t, "cpu"):
        return t.cpu().numpy()
    return np.asarray(t)


def _algorithm_placeholder_plot(message: str) -> hv.Text:
    return hv.Text(0, 0, message).opts(
        width=960,
        height=280,
        toolbar=None,
    )


def _history_transition_steps(history: RunHistory, n_steps: int) -> np.ndarray:
    """Return step axis for transition-indexed arrays [n_recorded-1, ...]."""
    recorded = np.asarray(history.recorded_steps, dtype=float)
    if recorded.size >= n_steps + 1:
        return recorded[1 : n_steps + 1]
    record_every = float(max(1, int(history.record_every)))
    return np.arange(1, n_steps + 1, dtype=float) * record_every


def _compute_masked_mean_p95(
    values: np.ndarray, alive_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-step mean and p95 over alive walkers (vectorized)."""
    arr = np.asarray(values, dtype=float)
    mask = np.asarray(alive_mask, dtype=bool)
    if arr.ndim != 2 or mask.ndim != 2:
        msg = "Expected 2D [time, walkers] arrays for masked statistics."
        raise ValueError(msg)

    n_steps = min(arr.shape[0], mask.shape[0])
    arr = arr[:n_steps]
    mask = mask[:n_steps]

    masked = np.where(mask, arr, np.nan)
    counts = mask.sum(axis=1)
    sums = np.nansum(masked, axis=1)
    mean = np.divide(
        sums,
        counts,
        out=np.full(n_steps, np.nan, dtype=float),
        where=counts > 0,
    )
    p95 = np.full(n_steps, np.nan, dtype=float)
    valid = counts > 0
    if np.any(valid):
        p95[valid] = np.nanpercentile(masked[valid], 95, axis=1)
    return mean, p95


def _compute_vectorized_lyapunov(history: RunHistory) -> dict[str, np.ndarray]:
    """Compute Lyapunov trajectory from recorded states using vectorized numpy ops."""
    x = _to_numpy(history.x_final).astype(float, copy=False)
    v = _to_numpy(history.v_final).astype(float, copy=False)
    if x.ndim != 3 or v.ndim != 3:
        msg = "Expected x_final/v_final with shape [n_recorded, N, d]."
        raise ValueError(msg)

    n_steps = min(int(history.n_recorded), x.shape[0], v.shape[0])
    if n_steps <= 0:
        msg = "No recorded states available for Lyapunov diagnostics."
        raise ValueError(msg)

    x = x[:n_steps]
    v = v[:n_steps]
    n_walkers = int(x.shape[1])

    alive = np.ones((n_steps, n_walkers), dtype=bool)
    if getattr(history, "alive_mask", None) is not None:
        alive_info = _to_numpy(history.alive_mask).astype(bool, copy=False)
        info_len = min(alive_info.shape[0], max(n_steps - 1, 0))
        if info_len > 0:
            alive[1 : 1 + info_len] = alive_info[:info_len]

    alive_3d = alive[..., None]
    counts = alive.sum(axis=1).astype(float)
    safe_counts = np.clip(counts, a_min=1.0, a_max=None)

    x_mean = np.where(alive_3d, x, 0.0).sum(axis=1) / safe_counts[:, None]
    v_mean = np.where(alive_3d, v, 0.0).sum(axis=1) / safe_counts[:, None]

    x_sq = np.sum((x - x_mean[:, None, :]) ** 2, axis=-1)
    v_sq = np.sum((v - v_mean[:, None, :]) ** 2, axis=-1)

    var_x = np.where(alive, x_sq, 0.0).sum(axis=1) / safe_counts
    var_v = np.where(alive, v_sq, 0.0).sum(axis=1) / safe_counts
    v_total = var_x + var_v

    recorded = np.asarray(history.recorded_steps, dtype=float)
    if recorded.size >= n_steps:
        time = recorded[:n_steps]
    else:
        record_every = float(max(1, int(history.record_every)))
        time = np.arange(n_steps, dtype=float) * record_every

    return {
        "time": time,
        "V_total": v_total,
        "V_var_x": var_x,
        "V_var_v": var_v,
    }


def _build_timeseries_mean_p95_plot(
    *,
    step: np.ndarray,
    mean: np.ndarray,
    p95: np.ndarray,
    title: str,
    ylabel: str,
    color: str,
) -> hv.Overlay | hv.Text:
    err95 = np.clip(np.asarray(p95, dtype=float) - np.asarray(mean, dtype=float), 0.0, None)
    frame = pd.DataFrame({"step": step, "mean": mean, "err95": err95}).replace(
        [np.inf, -np.inf], np.nan
    )
    frame = frame.dropna()
    if frame.empty:
        return _algorithm_placeholder_plot("No data available")

    curve = hv.Curve(frame, "step", "mean").opts(color=color, line_width=2, tools=["hover"])
    errorbars = hv.ErrorBars(frame, "step", ["mean", "err95"]).opts(
        color=color,
        alpha=0.45,
        line_width=1,
    )
    return (errorbars * curve).opts(
        title=title,
        xlabel="Recorded step",
        ylabel=ylabel,
        width=960,
        height=320,
        show_grid=True,
    )


def _build_timeseries_mean_error_plot(
    *,
    step: np.ndarray,
    mean: np.ndarray,
    error: np.ndarray,
    title: str,
    ylabel: str,
    color: str,
) -> hv.Overlay | hv.Text:
    frame = pd.DataFrame({"step": step, "mean": mean, "error": error}).replace(
        [np.inf, -np.inf], np.nan
    )
    frame = frame.dropna()
    if frame.empty:
        return _algorithm_placeholder_plot("No data available")

    curve = hv.Curve(frame, "step", "mean").opts(color=color, line_width=2, tools=["hover"])
    errorbars = hv.ErrorBars(frame, "step", ["mean", "error"]).opts(
        color=color,
        alpha=0.45,
        line_width=1,
    )
    return (errorbars * curve).opts(
        title=title,
        xlabel="Recorded step",
        ylabel=ylabel,
        width=960,
        height=320,
        show_grid=True,
    )


def _compute_interwalker_distance_stats(
    positions: np.ndarray,
    alive_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of pairwise inter-walker distances per step (vectorized)."""
    x = np.asarray(positions, dtype=float)
    alive = np.asarray(alive_mask, dtype=bool)
    if x.ndim != 3 or alive.ndim != 2:
        msg = "Expected positions [time, walkers, dim] and alive_mask [time, walkers]."
        raise ValueError(msg)

    n_steps = min(x.shape[0], alive.shape[0])
    x = x[:n_steps]
    alive = alive[:n_steps]
    n_walkers = int(x.shape[1])
    if n_walkers < 2:
        return np.full(n_steps, np.nan), np.full(n_steps, np.nan)

    sq_norm = np.sum(x * x, axis=-1)
    gram = np.einsum("tid,tjd->tij", x, x)
    dist_sq = sq_norm[:, :, None] + sq_norm[:, None, :] - 2.0 * gram
    np.maximum(dist_sq, 0.0, out=dist_sq)
    distances = np.sqrt(dist_sq)

    valid = alive[:, :, None] & alive[:, None, :]
    upper = np.triu(np.ones((n_walkers, n_walkers), dtype=bool), k=1)
    valid &= upper[None, :, :]

    counts = valid.sum(axis=(1, 2)).astype(float)
    sums = np.where(valid, distances, 0.0).sum(axis=(1, 2))
    means = np.divide(
        sums,
        counts,
        out=np.full(n_steps, np.nan, dtype=float),
        where=counts > 0,
    )

    sq_sums = np.where(valid, distances * distances, 0.0).sum(axis=(1, 2))
    second_moment = np.divide(
        sq_sums,
        counts,
        out=np.full(n_steps, np.nan, dtype=float),
        where=counts > 0,
    )
    variances = np.maximum(second_moment - means * means, 0.0)
    stds = np.sqrt(variances)
    return means, stds


def _build_companion_distance_plot(
    *,
    step: np.ndarray,
    clone_mean: np.ndarray,
    clone_p95: np.ndarray,
    random_mean: np.ndarray,
    random_p95: np.ndarray,
) -> hv.Overlay | hv.Text:
    clone_err = np.clip(np.asarray(clone_p95) - np.asarray(clone_mean), 0.0, None)
    random_err = np.clip(np.asarray(random_p95) - np.asarray(random_mean), 0.0, None)

    clone_df = pd.DataFrame({"step": step, "mean": clone_mean, "err95": clone_err}).replace(
        [np.inf, -np.inf], np.nan
    )
    random_df = pd.DataFrame({"step": step, "mean": random_mean, "err95": random_err}).replace(
        [np.inf, -np.inf], np.nan
    )
    clone_df = clone_df.dropna()
    random_df = random_df.dropna()

    overlays: list[Any] = []
    if not clone_df.empty:
        overlays.append(
            hv.ErrorBars(clone_df, "step", ["mean", "err95"])
            .relabel("Clone p95")
            .opts(color="#e45756", alpha=0.4, line_width=1)
        )
        overlays.append(
            hv.Curve(clone_df, "step", "mean")
            .relabel("Clone mean")
            .opts(color="#e45756", line_width=2, tools=["hover"])
        )
    if not random_df.empty:
        overlays.append(
            hv.ErrorBars(random_df, "step", ["mean", "err95"])
            .relabel("Random p95")
            .opts(color="#4c78a8", alpha=0.4, line_width=1)
        )
        overlays.append(
            hv.Curve(random_df, "step", "mean")
            .relabel("Random mean")
            .opts(color="#4c78a8", line_width=2, tools=["hover"])
        )

    if not overlays:
        return _algorithm_placeholder_plot("No companion-distance data available")

    plot = overlays[0]
    for overlay in overlays[1:]:
        plot = plot * overlay
    return plot.opts(
        title="Companion Distances Over Time (mean with p95 error bars)",
        xlabel="Recorded step",
        ylabel="Distance",
        width=960,
        height=340,
        legend_position="top_left",
        show_grid=True,
    )


def build_algorithm_diagnostics(history: RunHistory) -> dict[str, Any]:
    """Build vectorized algorithm diagnostics from collected run traces."""
    alive = _to_numpy(history.alive_mask).astype(bool, copy=False)
    will_clone = _to_numpy(history.will_clone).astype(bool, copy=False)
    fitness = _to_numpy(history.fitness).astype(float, copy=False)
    rewards = _to_numpy(history.rewards).astype(float, copy=False)
    x_pre = _to_numpy(history.x_before_clone).astype(float, copy=False)
    v_pre = _to_numpy(history.v_before_clone).astype(float, copy=False)
    companions_clone = _to_numpy(history.companions_clone).astype(np.int64, copy=False)
    companions_random = _to_numpy(history.companions_distance).astype(np.int64, copy=False)

    n_steps = min(
        alive.shape[0],
        will_clone.shape[0],
        fitness.shape[0],
        rewards.shape[0],
        x_pre.shape[0],
        v_pre.shape[0],
        companions_clone.shape[0],
        companions_random.shape[0],
    )
    if n_steps <= 0:
        msg = "No transition frames found in RunHistory."
        raise ValueError(msg)

    alive = alive[:n_steps]
    will_clone = will_clone[:n_steps]
    fitness = fitness[:n_steps]
    rewards = rewards[:n_steps]
    x_pre = x_pre[:n_steps]
    v_pre = v_pre[:n_steps]
    companions_clone = companions_clone[:n_steps]
    companions_random = companions_random[:n_steps]

    if x_pre.ndim != 3:
        msg = "Expected x_before_clone with shape [time, walkers, dim]."
        raise ValueError(msg)

    step = _history_transition_steps(history, n_steps)
    n_walkers = int(x_pre.shape[1])
    companions_clone = np.clip(companions_clone, 0, n_walkers - 1)
    companions_random = np.clip(companions_random, 0, n_walkers - 1)
    time_index = np.arange(n_steps, dtype=np.int64)[:, None]

    clone_dist = np.linalg.norm(
        x_pre - x_pre[time_index, companions_clone],
        axis=-1,
    )
    random_dist = np.linalg.norm(
        x_pre - x_pre[time_index, companions_random],
        axis=-1,
    )

    alive_counts = alive.sum(axis=1).astype(float)
    clone_counts = np.logical_and(will_clone, alive).sum(axis=1).astype(float)
    clone_pct = np.divide(
        100.0 * clone_counts,
        alive_counts,
        out=np.zeros_like(alive_counts),
        where=alive_counts > 0,
    )

    fit_mean, fit_p95 = _compute_masked_mean_p95(fitness, alive)
    rew_mean, rew_p95 = _compute_masked_mean_p95(rewards, alive)
    clone_dist_mean, clone_dist_p95 = _compute_masked_mean_p95(clone_dist, alive)
    random_dist_mean, random_dist_p95 = _compute_masked_mean_p95(random_dist, alive)

    clone_frame = pd.DataFrame({"step": step, "pct_clone": clone_pct}).replace(
        [np.inf, -np.inf], np.nan
    )
    clone_frame = clone_frame.dropna()
    if clone_frame.empty:
        clone_plot: hv.Curve | hv.Text = _algorithm_placeholder_plot("No clone data available")
    else:
        clone_plot = hv.Curve(clone_frame, "step", "pct_clone").opts(
            title="Percentage of Clones Over Time",
            xlabel="Recorded step",
            ylabel="% cloned (alive)",
            width=960,
            height=300,
            color="#f58518",
            line_width=2,
            ylim=(0, 100),
            show_grid=True,
            tools=["hover"],
        )

    fitness_plot = _build_timeseries_mean_p95_plot(
        step=step,
        mean=fit_mean,
        p95=fit_p95,
        title="Mean Fitness Over Time (p95 error bars)",
        ylabel="Fitness",
        color="#4c78a8",
    )
    reward_plot = _build_timeseries_mean_p95_plot(
        step=step,
        mean=rew_mean,
        p95=rew_p95,
        title="Mean Reward Over Time (p95 error bars)",
        ylabel="Reward",
        color="#72b7b2",
    )
    companion_plot = _build_companion_distance_plot(
        step=step,
        clone_mean=clone_dist_mean,
        clone_p95=clone_dist_p95,
        random_mean=random_dist_mean,
        random_p95=random_dist_p95,
    )
    inter_mean, inter_std = _compute_interwalker_distance_stats(x_pre, alive)
    interwalker_plot = _build_timeseries_mean_error_plot(
        step=step,
        mean=inter_mean,
        error=inter_std,
        title="Average Inter-Walker Distance Over Time (mean ± 1σ)",
        ylabel="Pairwise distance",
        color="#54a24b",
    )

    # Average walker speed (||v||) over time with p95 error bars.
    speed = np.linalg.norm(v_pre, axis=-1)  # [n_steps, N]
    speed_mean, speed_p95 = _compute_masked_mean_p95(speed, alive)
    velocity_plot = _build_timeseries_mean_p95_plot(
        step=step,
        mean=speed_mean,
        p95=speed_p95,
        title="Mean Walker Speed Over Time (p95 error bars)",
        ylabel="Speed (||v||)",
        color="#e45756",
    )

    # Geodesic edge distances and riemannian_kernel_volume weights over time.
    geodesic_plot = _algorithm_placeholder_plot("No geodesic edge distance data available.")
    rkv_plot = _algorithm_placeholder_plot("No riemannian_kernel_volume data available.")
    geo_list = getattr(history, "geodesic_edge_distances", None)
    ew_list = getattr(history, "edge_weights", None)
    recorded_steps = np.asarray(history.recorded_steps, dtype=float)
    if geo_list is not None and len(geo_list) > 0:
        geo_means = np.full(len(geo_list), np.nan)
        geo_stds = np.full(len(geo_list), np.nan)
        for t, geo_t in enumerate(geo_list):
            vals = _to_numpy(geo_t).astype(float).ravel()
            if vals.size > 0:
                geo_means[t] = float(np.mean(vals))
                geo_stds[t] = float(np.std(vals))
        geo_step = (
            recorded_steps[: len(geo_list)]
            if recorded_steps.size >= len(geo_list)
            else np.arange(len(geo_list), dtype=float)
        )
        geodesic_plot = _build_timeseries_mean_error_plot(
            step=geo_step,
            mean=geo_means,
            error=geo_stds,
            title="Mean Geodesic Edge Distance Over Time (mean ± 1σ)",
            ylabel="Geodesic distance",
            color="#9d755d",
        )
    if ew_list is not None and len(ew_list) > 0:
        rkv_means = np.full(len(ew_list), np.nan)
        rkv_stds = np.full(len(ew_list), np.nan)
        for t, ew_t in enumerate(ew_list):
            if isinstance(ew_t, dict) and "riemannian_kernel_volume" in ew_t:
                vals = _to_numpy(ew_t["riemannian_kernel_volume"]).astype(float).ravel()
                if vals.size > 0:
                    rkv_means[t] = float(np.mean(vals))
                    rkv_stds[t] = float(np.std(vals))
        rkv_step = (
            recorded_steps[: len(ew_list)]
            if recorded_steps.size >= len(ew_list)
            else np.arange(len(ew_list), dtype=float)
        )
        rkv_plot = _build_timeseries_mean_error_plot(
            step=rkv_step,
            mean=rkv_means,
            error=rkv_stds,
            title="Riemannian Kernel Volume Weights Over Time (mean ± 1σ)",
            ylabel="Weight",
            color="#b279a2",
        )

    lyapunov = _compute_vectorized_lyapunov(history)
    lyapunov_plot = build_lyapunov_plot(
        lyapunov["time"],
        lyapunov["V_total"],
        lyapunov["V_var_x"],
        lyapunov["V_var_v"],
    )

    return {
        "clone_plot": clone_plot,
        "fitness_plot": fitness_plot,
        "reward_plot": reward_plot,
        "companion_plot": companion_plot,
        "interwalker_plot": interwalker_plot,
        "velocity_plot": velocity_plot,
        "geodesic_plot": geodesic_plot,
        "rkv_plot": rkv_plot,
        "lyapunov_plot": lyapunov_plot,
        "n_transition_steps": int(n_steps),
        "n_lyapunov_steps": len(lyapunov["time"]),
    }


# ---------------------------------------------------------------------------
# Dashboard integration
# ---------------------------------------------------------------------------

_PLOT_KEYS = (
    "clone_plot",
    "fitness_plot",
    "reward_plot",
    "companion_plot",
    "interwalker_plot",
    "velocity_plot",
    "geodesic_plot",
    "rkv_plot",
    "lyapunov_plot",
)

_PLOT_LABELS = {
    "clone_plot": "clone percentage",
    "fitness_plot": "fitness trend",
    "reward_plot": "reward trend",
    "companion_plot": "companion distances",
    "interwalker_plot": "inter-walker distances",
    "velocity_plot": "walker speed",
    "geodesic_plot": "geodesic edge distances",
    "rkv_plot": "riemannian kernel volume weights",
    "lyapunov_plot": "Lyapunov convergence",
}

_PLOT_SECTION_TITLES = {
    "clone_plot": "### Percentage of Clones Over Time",
    "fitness_plot": "### Mean Fitness Over Time (p95 error bars)",
    "reward_plot": "### Mean Rewards Over Time (p95 error bars)",
    "companion_plot": "### Companion Distances (Clone vs Random, p95 error bars)",
    "interwalker_plot": "### Average Inter-Walker Distance Over Time (mean ± 1σ)",
    "velocity_plot": "### Mean Walker Speed Over Time (p95 error bars)",
    "geodesic_plot": "### Mean Geodesic Edge Distance Over Time (mean ± 1σ)",
    "rkv_plot": "### Riemannian Kernel Volume Weights Over Time (mean ± 1σ)",
    "lyapunov_plot": "### Lyapunov Convergence Over Time",
}


@dataclass
class AlgorithmDiagnosticsSection:
    """Container for the algorithm diagnostics dashboard section."""

    tab: pn.Column
    status: pn.pane.Markdown
    run_button: pn.widgets.Button
    on_run: Callable[[Any], None]
    on_history_ready: Callable[[], None]
    reset_plots: Callable[[], None]


def build_algorithm_diagnostics_tab(
    state: dict[str, Any],
) -> AlgorithmDiagnosticsSection:
    """Create the Algorithm Diagnostics tab and its callbacks.

    Parameters
    ----------
    state:
        Shared dashboard state dict (must contain ``"history"`` key when
        analysis is triggered).

    Returns
    -------
    AlgorithmDiagnosticsSection
        Dataclass exposing the assembled tab, widgets, and lifecycle hooks
        that the main dashboard wires into ``set_history`` / event handlers.
    """
    status = pn.pane.Markdown(
        "**Algorithm:** run a simulation or load a RunHistory, then click Run Algorithm Analysis.",
        sizing_mode="stretch_width",
    )
    run_button = pn.widgets.Button(
        name="Run Algorithm Analysis",
        button_type="primary",
        min_width=240,
        sizing_mode="stretch_width",
        disabled=True,
    )

    plots: dict[str, pn.pane.HoloViews] = {}
    for key in _PLOT_KEYS:
        plots[key] = pn.pane.HoloViews(
            _algorithm_placeholder_plot(f"Load RunHistory to show {_PLOT_LABELS[key]}."),
            linked_axes=False,
            sizing_mode="stretch_width",
        )

    # -- lifecycle callbacks ---------------------------------------------------

    def on_history_ready() -> None:
        """Enable the run button after a history is loaded."""
        status.object = "**Algorithm ready:** click Run Algorithm Analysis."
        run_button.disabled = False

    def reset_plots_fn() -> None:
        """Reset all plot panes to 'Click Run...' placeholders."""
        for key in _PLOT_KEYS:
            plots[key].object = _algorithm_placeholder_plot(
                f"Click Run Algorithm Analysis to show {_PLOT_LABELS[key]}."
            )

    def _update_outputs(history: RunHistory) -> None:
        try:
            diagnostics = build_algorithm_diagnostics(history)
        except Exception as exc:
            status.object = f"**Algorithm error:** {exc!s}"
            fallback = _algorithm_placeholder_plot("Algorithm diagnostics unavailable.")
            for key in _PLOT_KEYS:
                plots[key].object = fallback
            return

        for key in _PLOT_KEYS:
            plots[key].object = diagnostics[key]
        status.object = (
            "**Algorithm diagnostics updated:** "
            f"{diagnostics['n_transition_steps']} transition frames, "
            f"{diagnostics['n_lyapunov_steps']} Lyapunov frames."
        )

    def on_run(_: Any) -> None:
        history = state.get("history")
        if history is None:
            status.object = "**Error:** load or run a simulation first."
            return
        status.object = "**Running algorithm diagnostics...**"
        _update_outputs(history)

    run_button.on_click(on_run)

    # -- assemble tab ----------------------------------------------------------

    tab_children: list[Any] = [
        status,
        pn.Row(run_button, sizing_mode="stretch_width"),
        pn.pane.Markdown(
            "_Vectorized diagnostics computed directly from collected RunHistory traces._"
        ),
        pn.layout.Divider(),
    ]
    for key in _PLOT_KEYS:
        tab_children.append(pn.pane.Markdown(_PLOT_SECTION_TITLES[key]))
        tab_children.append(plots[key])

    tab = pn.Column(*tab_children, sizing_mode="stretch_both")

    return AlgorithmDiagnosticsSection(
        tab=tab,
        status=status,
        run_button=run_button,
        on_run=on_run,
        on_history_ready=on_history_ready,
        reset_plots=reset_plots_fn,
    )
