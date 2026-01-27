"""QFT dashboard with simulation and analysis tabs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.experiments.gas_config_panel import GasConfigPanel
from fragile.fractalai.qft import analysis as qft_analysis
from fragile.fractalai.qft.plotting import (
    build_correlation_decay_plot,
    build_lyapunov_plot,
    build_wilson_histogram_plot,
    build_wilson_timeseries_plot,
)


# Prevent Plotly from probing the system browser during import.
os.environ.setdefault("PLOTLY_RENDERER", "json")
hv.extension("bokeh")

__all__ = ["create_app"]


class SwarmConvergence3D(param.Parameterized):
    """3D swarm convergence viewer using Plotly backend."""

    point_size = param.Number(default=4, bounds=(1, 20), doc="Walker point size")
    point_alpha = param.Number(default=0.85, bounds=(0.05, 1.0), doc="Marker opacity")
    color_metric = param.ObjectSelector(
        default="constant",
        objects=["constant", "fitness", "reward", "radius"],
        doc="Color encoding for walkers",
    )
    fix_axes = param.Boolean(default=True, doc="Fix axis ranges to bounds extent")

    def __init__(self, history: RunHistory | None, bounds_extent: float = 10.0, **params):
        super().__init__(**params)
        self.history = history
        self.bounds_extent = float(bounds_extent)

        self._x = None
        self._fitness = None
        self._rewards = None
        self._alive = None

        self.time_player = pn.widgets.Player(
            name="frame",
            start=0,
            end=0,
            value=0,
            step=1,
            interval=150,
            loop_policy="loop",
        )
        self.time_player.disabled = True
        self.time_player.sizing_mode = "stretch_width"
        self.time_player.param.watch(self._sync_frame, "value")

        self.status_pane = pn.pane.Markdown(
            "**Status:** Load or run a simulation to view 3D convergence.",
            sizing_mode="stretch_width",
        )

        self.plot_pane = pn.pane.Plotly(
            self._make_figure(0),
            sizing_mode="stretch_width",
            height=720,
        )

        self.param.watch(self._refresh_frame, ["point_size", "point_alpha", "color_metric"])

        if history is not None:
            self.set_history(history)

    def set_history(self, history: RunHistory):
        """Load RunHistory data into the viewer."""
        self.history = history
        self._x = history.x_final.detach().cpu().numpy()
        self._fitness = history.fitness.detach().cpu().numpy()
        self._rewards = history.rewards.detach().cpu().numpy()
        self._alive = history.alive_mask.detach().cpu().numpy().astype(bool)

        self.time_player.end = max(0, history.n_recorded - 1)
        self.time_player.value = 0
        self.time_player.disabled = False

        self.status_pane.object = (
            f"**RunHistory loaded:** N={history.N}, "
            f"steps={history.n_steps}, "
            f"recorded={history.n_recorded}"
        )
        self._refresh_frame()

    def _sync_frame(self, event):
        if self.history is None:
            return
        self._update_plot(int(np.clip(event.new, 0, self.history.n_recorded - 1)))

    def _refresh_frame(self, *_):
        if self.history is None:
            return
        self._update_plot(int(np.clip(self.time_player.value, 0, self.history.n_recorded - 1)))

    def _get_alive_mask(self, frame: int) -> np.ndarray:
        if self._alive is None:
            return np.ones(self._x.shape[1], dtype=bool)
        if frame == 0:
            return np.ones(self._x.shape[1], dtype=bool)
        idx = min(frame - 1, self._alive.shape[0] - 1)
        return self._alive[idx]

    def _frame_title(self, frame: int) -> str:
        if self.history is None:
            return "QFT Swarm Convergence"
        step = self.history.recorded_steps[frame]
        return f"QFT Swarm Convergence (frame {frame}, step {step})"

    def _make_figure(self, frame: int):
        import plotly.graph_objects as go

        if self.history is None or self._x is None:
            fig = go.Figure()
            fig.update_layout(
                title="QFT Swarm Convergence",
                height=720,
                margin={"l": 0, "r": 0, "t": 40, "b": 0},
            )
            return fig

        positions = self._x[frame]
        if positions.shape[1] < 3:
            msg = "Need at least 3 dimensions for 3D convergence visualization."
            raise ValueError(msg)

        alive = self._get_alive_mask(frame)
        positions = positions[alive][:, :3]

        if positions.size == 0:
            fig = go.Figure()
            fig.update_layout(title=self._frame_title(frame), height=720)
            return fig

        colors = "#1f77b4"
        colorbar = None
        showscale = False

        if self.color_metric != "constant" and frame > 0:
            if self.color_metric == "fitness":
                colors = self._fitness[frame - 1][alive]
                colorbar = {"title": "fitness"}
                showscale = True
            elif self.color_metric == "reward":
                colors = self._rewards[frame - 1][alive]
                colorbar = {"title": "reward"}
                showscale = True
            elif self.color_metric == "radius":
                colors = np.linalg.norm(positions, axis=1)
                colorbar = {"title": "radius"}
                showscale = True

        scatter = go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            marker={
                "size": self.point_size,
                "color": colors,
                "colorscale": "Viridis" if showscale else None,
                "opacity": self.point_alpha,
                "showscale": showscale,
                "colorbar": colorbar,
            },
        )

        fig = go.Figure(data=[scatter])
        fig.update_layout(
            title=self._frame_title(frame),
            height=720,
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
        )

        if self.fix_axes:
            extent = float(self.bounds_extent)
            fig.update_layout(
                scene={
                    "xaxis": {"range": [-extent, extent]},
                    "yaxis": {"range": [-extent, extent]},
                    "zaxis": {"range": [-extent, extent]},
                }
            )

        return fig

    def _update_plot(self, frame: int):
        self.plot_pane.object = self._make_figure(frame)

    def panel(self) -> pn.Column:
        """Return the Panel layout for the 3D convergence viewer."""
        return pn.Column(
            pn.pane.Markdown("## 3D Swarm Convergence (Plotly)"),
            self.time_player,
            pn.Spacer(height=10),
            pn.layout.Divider(),
            self.plot_pane,
            self.status_pane,
            sizing_mode="stretch_width",
        )


class AnalysisSettings(param.Parameterized):
    analysis_time_index = param.Integer(default=None, bounds=(0, None), allow_None=True)
    analysis_step = param.Integer(default=None, bounds=(0, None), allow_None=True)
    warmup_fraction = param.Number(default=0.1, bounds=(0.0, 1.0))
    h_eff = param.Number(default=1.0, bounds=(1e-6, None))
    correlation_r_max = param.Number(default=0.5, bounds=(1e-6, None))
    correlation_bins = param.Integer(default=50, bounds=(1, None))
    gradient_neighbors = param.Integer(default=5, bounds=(1, None))
    build_fractal_set = param.Boolean(default=False)
    fractal_set_stride = param.Integer(default=10, bounds=(1, None))

    use_local_fields = param.Boolean(default=False)
    use_connected = param.Boolean(default=False)
    density_sigma = param.Number(default=0.5, bounds=(1e-6, None))

    compute_particles = param.Boolean(default=False)
    particle_operators = param.String(default="baryon,meson,glueball")
    particle_max_lag = param.Integer(default=80, bounds=(1, None), allow_None=True)
    particle_fit_start = param.Integer(default=7, bounds=(0, None))
    particle_fit_stop = param.Integer(default=16, bounds=(0, None))
    particle_fit_mode = param.ObjectSelector(
        default="window",
        objects=["window", "plateau", "auto"],
    )
    particle_plateau_min_points = param.Integer(default=3, bounds=(1, None))
    particle_plateau_max_points = param.Integer(default=None, bounds=(1, None), allow_None=True)
    particle_plateau_max_cv = param.Number(default=0.2, bounds=(1e-6, None), allow_None=True)
    particle_mass = param.Number(default=1.0, bounds=(1e-6, None))
    particle_ell0 = param.Number(default=None, bounds=(1e-6, None), allow_None=True)
    particle_use_connected = param.Boolean(default=True)
    particle_neighbor_method = param.ObjectSelector(default="knn", objects=["companion", "knn"])
    particle_knn_k = param.Integer(default=4, bounds=(1, None))
    particle_knn_sample = param.Integer(default=512, bounds=(1, None), allow_None=True)
    particle_meson_reduce = param.ObjectSelector(default="mean", objects=["mean", "first"])
    particle_baryon_pairs = param.Integer(default=None, bounds=(1, None), allow_None=True)

    compute_string_tension = param.Boolean(default=False)
    string_tension_max_triangles = param.Integer(default=20000, bounds=(1, None))
    string_tension_bins = param.Integer(default=20, bounds=(2, None))

    def to_cli_args(self, history_path: Path, output_dir: Path, analysis_id: str) -> list[str]:
        args = [
            "analyze_fractal_gas_qft",
            "--history-path",
            str(history_path),
            "--output-dir",
            str(output_dir),
            "--analysis-id",
            analysis_id,
            "--warmup-fraction",
            str(self.warmup_fraction),
            "--h-eff",
            str(self.h_eff),
            "--correlation-r-max",
            str(self.correlation_r_max),
            "--correlation-bins",
            str(self.correlation_bins),
            "--gradient-neighbors",
            str(self.gradient_neighbors),
            "--fractal-set-stride",
            str(self.fractal_set_stride),
            "--density-sigma",
            str(self.density_sigma),
            "--particle-operators",
            self.particle_operators,
            "--particle-fit-start",
            str(self.particle_fit_start),
            "--particle-fit-stop",
            str(self.particle_fit_stop),
            "--particle-fit-mode",
            str(self.particle_fit_mode),
            "--particle-plateau-min-points",
            str(self.particle_plateau_min_points),
            "--particle-mass",
            str(self.particle_mass),
            "--particle-neighbor-method",
            str(self.particle_neighbor_method),
            "--particle-knn-k",
            str(self.particle_knn_k),
            "--particle-meson-reduce",
            str(self.particle_meson_reduce),
            "--string-tension-max-triangles",
            str(self.string_tension_max_triangles),
            "--string-tension-bins",
            str(self.string_tension_bins),
        ]

        if self.analysis_time_index is not None:
            args.extend(["--analysis-time-index", str(self.analysis_time_index)])
        if self.analysis_step is not None:
            args.extend(["--analysis-step", str(self.analysis_step)])
        if self.build_fractal_set:
            args.append("--build-fractal-set")
        if self.use_local_fields:
            args.append("--use-local-fields")
        if self.use_connected:
            args.append("--use-connected")
        if self.compute_particles:
            args.append("--compute-particles")
        if self.particle_max_lag is not None:
            args.extend(["--particle-max-lag", str(self.particle_max_lag)])
        if self.particle_plateau_max_points is not None:
            args.extend(["--particle-plateau-max-points", str(self.particle_plateau_max_points)])
        if self.particle_plateau_max_cv is not None:
            args.extend(["--particle-plateau-max-cv", str(self.particle_plateau_max_cv)])
        if self.particle_ell0 is not None:
            args.extend(["--particle-ell0", str(self.particle_ell0)])
        if self.particle_use_connected:
            args.append("--particle-use-connected")
        else:
            args.append("--no-particle-use-connected")
        if self.particle_knn_sample is not None:
            args.extend(["--particle-knn-sample", str(self.particle_knn_sample)])
        if self.particle_baryon_pairs is not None:
            args.extend(["--particle-baryon-pairs", str(self.particle_baryon_pairs)])
        if self.compute_string_tension:
            args.append("--compute-string-tension")

        return args


@contextmanager
def _temporary_argv(args: list[str]):
    original = sys.argv
    sys.argv = args
    try:
        yield
    finally:
        sys.argv = original


def _format_analysis_summary(metrics: dict[str, Any]) -> str:
    observables = metrics.get("observables", {})
    d_fit = observables.get("d_prime_correlation", {})
    r_fit = observables.get("r_prime_correlation", {})
    ew = metrics.get("electroweak_proxy", {})
    qsd = metrics.get("qsd_variance", {})

    lines = [
        "## Analysis Summary",
        f"- d_prime ξ: {d_fit.get('xi', 0.0):.4f} (R² {d_fit.get('r_squared', 0.0):.3f})",
        f"- r_prime ξ: {r_fit.get('xi', 0.0):.4f} (R² {r_fit.get('r_squared', 0.0):.3f})",
        f"- sin²θw proxy: {ew.get('sin2_theta_w_proxy', 0.0):.4f}",
        f"- QSD scaling exponent: {qsd.get('scaling_exponent', 0.0):.4f}",
    ]

    local_fields = metrics.get("local_fields")
    if local_fields:
        lines.append("\n### Local Field Fits")
        for name, info in local_fields.items():
            fit = info.get("fit", {})
            lines.append(
                f"- {name}: ξ={fit.get('xi', 0.0):.4f}, R²={fit.get('r_squared', 0.0):.3f}"
            )

    particle = metrics.get("particle_observables", {}) or {}
    operators = particle.get("operators") if isinstance(particle, dict) else None
    if operators:
        lines.append("\n### Particle Mass Estimates")
        for name, data in operators.items():
            fit = data.get("fit", {})
            lines.append(
                f"- {name}: m={fit.get('mass', 0.0):.4f} (R² {fit.get('r_squared', 0.0):.3f})"
            )

    return "\n".join(lines)


def _build_analysis_plots(metrics: dict[str, Any], arrays: dict[str, Any]) -> list[Any]:
    plots: list[Any] = []

    observables = metrics.get("observables", {})
    d_fit = observables.get("d_prime_correlation", {})
    r_fit = observables.get("r_prime_correlation", {})

    if "d_prime_bins" in arrays:
        plot = build_correlation_decay_plot(
            arrays["d_prime_bins"],
            arrays["d_prime_correlation"],
            arrays["d_prime_counts"],
            d_fit,
            "Diversity Correlation Decay",
        )
        if plot is not None:
            plots.append(plot)

    if "r_prime_bins" in arrays:
        plot = build_correlation_decay_plot(
            arrays["r_prime_bins"],
            arrays["r_prime_correlation"],
            arrays["r_prime_counts"],
            r_fit,
            "Reward Correlation Decay",
        )
        if plot is not None:
            plots.append(plot)

    if "lyapunov_time" in arrays:
        plot = build_lyapunov_plot(
            arrays["lyapunov_time"],
            arrays["lyapunov_total"],
            arrays["lyapunov_var_x"],
            arrays["lyapunov_var_v"],
        )
        plots.append(plot)

    if "wilson_time_index" in arrays:
        plot = build_wilson_timeseries_plot(
            arrays["wilson_time_index"],
            arrays["wilson_action_mean"],
        )
        if plot is not None:
            plots.append(plot)

    wilson = metrics.get("wilson_loops")
    if wilson and "wilson_values" in wilson:
        plot = build_wilson_histogram_plot(
            np.asarray(wilson["wilson_values"], dtype=float),
            "Wilson Loop Distribution",
        )
        if plot is not None:
            plots.append(plot)

    local_fields = metrics.get("local_fields") or {}
    for field_name, info in local_fields.items():
        bins_key = f"{field_name}_bins"
        corr_key = f"{field_name}_correlation"
        counts_key = f"{field_name}_counts"
        if bins_key not in arrays:
            continue
        plot = build_correlation_decay_plot(
            arrays[bins_key],
            arrays[corr_key],
            arrays[counts_key],
            info.get("fit", {}),
            f"{field_name} Correlation",
        )
        if plot is not None:
            plots.append(plot)

    return plots


BARYON_REFS = {
    "proton": 0.938272,
    "neutron": 0.939565,
    "delta": 1.232,
    "lambda": 1.115683,
    "sigma0": 1.192642,
    "xi0": 1.31486,
    "omega-": 1.67245,
}

MESON_REFS = {
    "pion": 0.13957,
    "kaon": 0.493677,
    "eta": 0.547862,
    "rho": 0.77526,
    "omega": 0.78265,
    "phi": 1.01946,
    "jpsi": 3.0969,
    "upsilon": 9.4603,
}


def _closest_reference(value: float, refs: dict[str, float]) -> tuple[str, float, float]:
    name, ref = min(refs.items(), key=lambda kv: abs(value - kv[1]))
    err = (value - ref) / ref * 100.0
    return name, ref, err


def _best_fit_scale(
    masses: dict[str, float], anchors: list[tuple[str, float, str]]
) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for _label, mass_phys, family in anchors:
        alg_mass = masses.get(family)
        if alg_mass is None or alg_mass <= 0:
            continue
        numerator += alg_mass * mass_phys
        denominator += alg_mass**2
    if denominator <= 0:
        return None
    return numerator / denominator


def _format_closest(value: float | None, refs: dict[str, float]) -> str:
    if value is None or value <= 0:
        return "n/a"
    name, ref, err = _closest_reference(value, refs)
    return f"{name} {ref:.3f} ({err:+.1f}%)"


def _extract_particle_masses(metrics: dict[str, Any]) -> dict[str, float]:
    particle = metrics.get("particle_observables") or {}
    operators = particle.get("operators") or {}
    masses: dict[str, float] = {}
    for name in ("baryon", "meson", "glueball"):
        fit = operators.get(name, {}).get("fit")
        if fit and isinstance(fit.get("mass"), int | float):
            masses[name] = float(fit["mass"])

    string_tension = metrics.get("string_tension")
    if isinstance(string_tension, dict):
        sigma = string_tension.get("sigma")
        if isinstance(sigma, int | float) and sigma > 0:
            masses["sqrt_sigma"] = float(sigma) ** 0.5

    return masses


def _build_algorithmic_mass_rows(masses: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for name in ("baryon", "meson", "glueball", "sqrt_sigma"):
        if name in masses:
            rows.append({"operator": name, "mass_alg": masses[name]})
    return rows


def _build_best_fit_rows(masses: dict[str, float]) -> list[dict[str, Any]]:
    anchors: list[tuple[str, float, str]] = []
    anchors.extend((f"baryon->{name}", mass, "baryon") for name, mass in BARYON_REFS.items())
    anchors.extend((f"meson->{name}", mass, "meson") for name, mass in MESON_REFS.items())
    baryon_anchors = [a for a in anchors if a[2] == "baryon"]
    meson_anchors = [a for a in anchors if a[2] == "meson"]
    combined_anchors = baryon_anchors + meson_anchors

    rows: list[dict[str, Any]] = []
    for label, anchor_list in (
        ("baryon refs", baryon_anchors),
        ("meson refs", meson_anchors),
        ("baryon+meson refs", combined_anchors),
    ):
        scale = _best_fit_scale(masses, anchor_list)
        if scale is None:
            rows.append({"fit_mode": label, "scale_GeV_per_alg": None})
            continue
        pred_b = masses.get("baryon", 0.0) * scale
        pred_m = masses.get("meson", 0.0) * scale
        rows.append({
            "fit_mode": label,
            "scale_GeV_per_alg": scale,
            "baryon_pred_GeV": pred_b,
            "closest_baryon": _format_closest(pred_b, BARYON_REFS),
            "meson_pred_GeV": pred_m,
            "closest_meson": _format_closest(pred_m, MESON_REFS),
        })
    return rows


def _build_anchor_rows(
    masses: dict[str, float],
    glueball_ref: tuple[str, float] | None,
    sqrt_sigma_ref: float | None,
) -> list[dict[str, Any]]:
    anchors: list[tuple[str, float, str]] = []
    anchors.extend((f"baryon->{name}", mass, "baryon") for name, mass in BARYON_REFS.items())
    anchors.extend((f"meson->{name}", mass, "meson") for name, mass in MESON_REFS.items())
    if glueball_ref is not None and masses.get("glueball", 0.0) > 0:
        label, mass = glueball_ref
        anchors.append((f"glueball->{label}", mass, "glueball"))
    if sqrt_sigma_ref is not None and masses.get("sqrt_sigma", 0.0) > 0:
        anchors.append((f"sqrt_sigma->{sqrt_sigma_ref:.3f}", sqrt_sigma_ref, "sqrt_sigma"))

    glueball_refs: dict[str, float] = {}
    if glueball_ref is not None:
        label, mass = glueball_ref
        glueball_refs[label] = mass

    rows: list[dict[str, Any]] = []
    for label, mass_phys, family in anchors:
        alg_mass = masses.get(family)
        if alg_mass is None or alg_mass <= 0:
            rows.append({"anchor": label})
            continue
        scale = mass_phys / alg_mass
        pred_b = masses.get("baryon", 0.0) * scale
        pred_m = masses.get("meson", 0.0) * scale
        row = {
            "anchor": label,
            "scale_GeV_per_alg": scale,
            "baryon_pred_GeV": pred_b,
            "closest_baryon": _format_closest(pred_b, BARYON_REFS),
            "meson_pred_GeV": pred_m,
            "closest_meson": _format_closest(pred_m, MESON_REFS),
        }
        if masses.get("glueball") and glueball_ref is not None:
            pred_g = masses.get("glueball", 0.0) * scale
            row["glueball_pred_GeV"] = pred_g
            row["closest_glueball"] = _format_closest(pred_g, glueball_refs)
        if masses.get("sqrt_sigma") and sqrt_sigma_ref is not None:
            pred_s = masses.get("sqrt_sigma", 0.0) * scale
            row["sqrt_sigma_pred_GeV"] = pred_s
            row["closest_sqrt_sigma"] = f"{sqrt_sigma_ref:.3f}"
        rows.append(row)
    return rows


SWEEP_PARAM_DEFS: dict[str, dict[str, Any]] = {
    "density_sigma": {"label": "density_sigma", "type": float},
    "correlation_r_max": {"label": "correlation_r_max", "type": float},
    "correlation_bins": {"label": "correlation_bins", "type": int},
    "gradient_neighbors": {"label": "gradient_neighbors", "type": int},
    "warmup_fraction": {"label": "warmup_fraction", "type": float},
    "fractal_set_stride": {"label": "fractal_set_stride", "type": int},
    "particle_fit_start": {"label": "particle_fit_start", "type": int},
    "particle_fit_stop": {"label": "particle_fit_stop", "type": int},
    "particle_max_lag": {"label": "particle_max_lag", "type": int},
    "particle_knn_k": {"label": "particle_knn_k", "type": int},
    "particle_knn_sample": {"label": "particle_knn_sample", "type": int},
    "particle_mass": {"label": "particle_mass", "type": float},
    "particle_ell0": {"label": "particle_ell0", "type": float},
}

SWEEP_METRICS: dict[str, str] = {
    "baryon mass": "particle_baryon_mass",
    "baryon R²": "particle_baryon_r2",
    "meson mass": "particle_meson_mass",
    "meson R²": "particle_meson_r2",
    "glueball mass": "particle_glueball_mass",
    "glueball R²": "particle_glueball_r2",
    "d_prime ξ": "d_prime_xi",
    "d_prime R²": "d_prime_r2",
    "r_prime ξ": "r_prime_xi",
    "r_prime R²": "r_prime_r2",
    "string tension σ": "string_tension_sigma",
}


def _coerce_sweep_value(param_name: str, value: float) -> Any:
    param_def = SWEEP_PARAM_DEFS.get(param_name, {})
    cast = param_def.get("type", float)
    return cast(value)


def _build_sweep_values(param_name: str, min_val: float, max_val: float, steps: int) -> list[Any]:
    steps = max(1, int(steps))
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    if steps == 1:
        values = [min_val]
    else:
        values = np.linspace(float(min_val), float(max_val), steps)
    param_type = SWEEP_PARAM_DEFS.get(param_name, {}).get("type", float)
    if param_type is int:
        cast_values = [int(round(v)) for v in values]
        return sorted(set(cast_values))
    return [float(v) for v in values]


def _resolve_fit_window(
    analysis_settings: AnalysisSettings,
    x_param: str,
    x_val: Any,
    y_param: str | None,
    y_val: Any | None,
) -> tuple[int | None, int | None]:
    fit_start = analysis_settings.particle_fit_start
    fit_stop = analysis_settings.particle_fit_stop
    if x_param == "particle_fit_start":
        fit_start = int(x_val)
    elif x_param == "particle_fit_stop":
        fit_stop = int(x_val)
    if y_param == "particle_fit_start" and y_val is not None:
        fit_start = int(y_val)
    elif y_param == "particle_fit_stop" and y_val is not None:
        fit_stop = int(y_val)
    return fit_start, fit_stop


def _metric_operator(metric_key: str) -> str | None:
    if metric_key.startswith("particle_baryon"):
        return "baryon"
    if metric_key.startswith("particle_meson"):
        return "meson"
    if metric_key.startswith("particle_glueball"):
        return "glueball"
    return None


def _extract_fit_metadata(
    metrics: dict[str, Any], operator: str
) -> tuple[int | None, int | None, str | None]:
    fit = (
        metrics.get("particle_observables", {})
        .get("operators", {})
        .get(operator, {})
        .get("fit", {})
    )
    fit_start = fit.get("fit_start")
    fit_stop = fit.get("fit_stop")
    fit_mode = fit.get("fit_mode")
    return fit_start, fit_stop, fit_mode


def _extract_metric(metrics: dict[str, Any], key: str) -> float:
    if key == "d_prime_xi":
        return float(metrics.get("observables", {}).get("d_prime_correlation", {}).get("xi", np.nan))
    if key == "d_prime_r2":
        return float(
            metrics.get("observables", {}).get("d_prime_correlation", {}).get("r_squared", np.nan)
        )
    if key == "r_prime_xi":
        return float(metrics.get("observables", {}).get("r_prime_correlation", {}).get("xi", np.nan))
    if key == "r_prime_r2":
        return float(
            metrics.get("observables", {}).get("r_prime_correlation", {}).get("r_squared", np.nan)
        )
    if key == "string_tension_sigma":
        return float(metrics.get("string_tension", {}).get("sigma", np.nan))

    particle = metrics.get("particle_observables", {}) or {}
    operators = particle.get("operators", {}) or {}
    if key == "particle_baryon_mass":
        return float(operators.get("baryon", {}).get("fit", {}).get("mass", np.nan))
    if key == "particle_baryon_r2":
        return float(operators.get("baryon", {}).get("fit", {}).get("r_squared", np.nan))
    if key == "particle_meson_mass":
        return float(operators.get("meson", {}).get("fit", {}).get("mass", np.nan))
    if key == "particle_meson_r2":
        return float(operators.get("meson", {}).get("fit", {}).get("r_squared", np.nan))
    if key == "particle_glueball_mass":
        return float(operators.get("glueball", {}).get("fit", {}).get("mass", np.nan))
    if key == "particle_glueball_r2":
        return float(operators.get("glueball", {}).get("fit", {}).get("r_squared", np.nan))

    return float(np.nan)


def _extract_particle_fit_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "baryon_mass": _extract_metric(metrics, "particle_baryon_mass"),
        "baryon_r2": _extract_metric(metrics, "particle_baryon_r2"),
        "meson_mass": _extract_metric(metrics, "particle_meson_mass"),
        "meson_r2": _extract_metric(metrics, "particle_meson_r2"),
        "glueball_mass": _extract_metric(metrics, "particle_glueball_mass"),
        "glueball_r2": _extract_metric(metrics, "particle_glueball_r2"),
    }


def _build_sweep_plot(
    dataframe: pd.DataFrame, x_param: str, metric_key: str, y_param: str | None = None
) -> hv.Element | None:
    if dataframe.empty:
        return None
    if y_param is None:
        plot = hv.Curve(dataframe, kdims=[x_param], vdims=[metric_key]).opts(
            xlabel=x_param,
            ylabel=metric_key,
            title=f"{metric_key} vs {x_param}",
            width=700,
            height=400,
        )
        return plot
    plot = hv.HeatMap(dataframe, kdims=[x_param, y_param], vdims=[metric_key]).opts(
        xlabel=x_param,
        ylabel=y_param,
        title=f"{metric_key} sweep",
        width=700,
        height=450,
        colorbar=True,
    )
    return plot


def create_app() -> pn.template.FastListTemplate:
    """Create the QFT convergence + analysis dashboard."""
    debug = os.environ.get("QFT_DASH_DEBUG", "").lower() in {"1", "true", "yes"}
    skip_sidebar = os.environ.get("QFT_DASH_SKIP_SIDEBAR", "").lower() in {"1", "true", "yes"}
    skip_visual = os.environ.get("QFT_DASH_SKIP_VIS", "").lower() in {"1", "true", "yes"}

    def _debug(msg: str):
        if debug:
            print(f"[qft-dashboard] {msg}", flush=True)

    sidebar = pn.Column(
        pn.pane.Markdown("## QFT Dashboard"),
        pn.pane.Markdown("Starting dashboard..."),
        sizing_mode="stretch_width",
    )
    main = pn.Column(
        pn.pane.Markdown("Loading visualization..."),
        sizing_mode="stretch_both",
    )

    template = pn.template.FastListTemplate(
        title="QFT Swarm Convergence Dashboard",
        sidebar=[sidebar],
        main=[main],
        sidebar_width=435,
        main_max_width="100%",
    )

    def _build_ui():
        start_total = time.time()
        _debug("initializing extensions")
        pn.extension("plotly", "tabulator")

        _debug("building config + visualizer")
        gas_config = GasConfigPanel.create_qft_config(dims=3, bounds_extent=10.0)
        visualizer = SwarmConvergence3D(history=None, bounds_extent=gas_config.bounds_extent)
        analysis_settings = AnalysisSettings()

        state: dict[str, Any] = {
            "history": None,
            "history_path": None,
            "analysis_metrics": None,
            "analysis_arrays": None,
        }

        _debug("setting up history controls")
        repo_root = Path(__file__).resolve().parents[4]
        qft_run_id = "qft_penalty_thr0p75_pen0p9_m354_ed2p80_nu1p10_N200_long"
        qft_history_path = repo_root / "outputs" / "qft_calibrated" / f"{qft_run_id}_history.pt"
        qft_history_dir = qft_history_path.parent
        qft_history_dir.mkdir(parents=True, exist_ok=True)
        history_dir = qft_history_dir
        history_path_input = pn.widgets.TextInput(
            name="QFT RunHistory path",
            value=str(qft_history_path),
            width=335,
            sizing_mode="stretch_width",
        )
        browse_button = pn.widgets.Button(
            name="Browse files...",
            button_type="default",
            width=335,
            sizing_mode="stretch_width",
        )
        file_selector_container = pn.Column(sizing_mode="stretch_width")

        load_button = pn.widgets.Button(
            name="Load RunHistory",
            button_type="primary",
            width=335,
            sizing_mode="stretch_width",
        )
        load_status = pn.pane.Markdown(
            "**Load a history**: paste a *_history.pt path or browse and click Load.",
            sizing_mode="stretch_width",
        )

        analysis_status_sidebar = pn.pane.Markdown(
            "**Analysis:** run a simulation or load a RunHistory.",
            sizing_mode="stretch_width",
        )
        analysis_status_main = pn.pane.Markdown(
            "**Analysis:** run a simulation or load a RunHistory.",
            sizing_mode="stretch_width",
        )
        analysis_summary = pn.pane.Markdown(
            "## Analysis Summary\n_Run analysis to populate._",
            sizing_mode="stretch_width",
        )
        analysis_json = pn.pane.JSON({}, depth=2, sizing_mode="stretch_width")
        analysis_plots = pn.Column(sizing_mode="stretch_width")

        analysis_output_dir = pn.widgets.TextInput(
            name="Analysis output dir",
            value="outputs/qft_dashboard_analysis",
            width=335,
            sizing_mode="stretch_width",
        )
        analysis_id_input = pn.widgets.TextInput(
            name="Analysis id",
            value="",
            width=335,
            sizing_mode="stretch_width",
            placeholder="Optional (defaults to timestamp)",
        )
        run_analysis_button = pn.widgets.Button(
            name="Run Analysis",
            button_type="primary",
            width=335,
            sizing_mode="stretch_width",
            disabled=True,
        )
        run_analysis_button_main = pn.widgets.Button(
            name="Run Analysis",
            button_type="primary",
            width=240,
            sizing_mode="fixed",
            disabled=True,
        )

        particle_status = pn.pane.Markdown(
            "**Particles:** run particle analysis to populate tables.",
            sizing_mode="stretch_width",
        )
        particle_run_button = pn.widgets.Button(
            name="Compute Particles",
            button_type="primary",
            width=240,
            sizing_mode="fixed",
            disabled=True,
        )
        glueball_label_input = pn.widgets.TextInput(
            name="Glueball label",
            value="glueball",
            width=200,
            sizing_mode="fixed",
        )
        glueball_ref_input = pn.widgets.FloatInput(
            name="Glueball ref (GeV)",
            value=None,
            step=0.01,
            width=200,
            sizing_mode="fixed",
        )
        sqrt_sigma_ref_input = pn.widgets.FloatInput(
            name="sqrt(sigma) ref (GeV)",
            value=None,
            step=0.01,
            width=200,
            sizing_mode="fixed",
        )

        particle_mass_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        particle_fit_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        particle_anchor_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
        )

        sweep_enable_2d = pn.widgets.Checkbox(name="2D sweep", value=False)
        sweep_param_x = pn.widgets.Select(
            name="Sweep param (X)",
            options={v["label"]: k for k, v in SWEEP_PARAM_DEFS.items()},
            value="density_sigma",
        )
        sweep_param_y = pn.widgets.Select(
            name="Sweep param (Y)",
            options={v["label"]: k for k, v in SWEEP_PARAM_DEFS.items()},
            value="particle_knn_k",
        )
        sweep_metric = pn.widgets.Select(
            name="Metric",
            options={label: key for label, key in SWEEP_METRICS.items()},
            value="particle_baryon_mass",
        )
        sweep_min_x = pn.widgets.FloatInput(name="X min", value=0.1, step=0.1, width=120)
        sweep_max_x = pn.widgets.FloatInput(name="X max", value=1.0, step=0.1, width=120)
        sweep_steps_x = pn.widgets.IntInput(name="X steps", value=5, step=1, width=120)
        sweep_min_y = pn.widgets.FloatInput(name="Y min", value=1.0, step=1.0, width=120)
        sweep_max_y = pn.widgets.FloatInput(name="Y max", value=5.0, step=1.0, width=120)
        sweep_steps_y = pn.widgets.IntInput(name="Y steps", value=4, step=1, width=120)
        sweep_run_button = pn.widgets.Button(
            name="Run Sweep",
            button_type="primary",
            width=240,
            sizing_mode="fixed",
            disabled=True,
        )
        sweep_status = pn.pane.Markdown(
            "**Sweep:** configure parameters and run to see results.",
            sizing_mode="stretch_width",
        )
        sweep_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
        )
        sweep_plot = pn.pane.HoloViews(sizing_mode="stretch_width")

        def _set_analysis_status(message: str) -> None:
            analysis_status_sidebar.object = message
            analysis_status_main.object = message

        def _set_particle_status(message: str) -> None:
            particle_status.object = message

        def _update_particle_tables(metrics: dict[str, Any]) -> None:
            masses = _extract_particle_masses(metrics)
            if "baryon" not in masses or "meson" not in masses:
                particle_mass_table.value = pd.DataFrame()
                particle_fit_table.value = pd.DataFrame()
                particle_anchor_table.value = pd.DataFrame()
                _set_particle_status(
                    "**Particles:** missing baryon/meson masses. Enable compute_particles."
                )
                return

            glueball_ref = None
            if glueball_ref_input.value is not None:
                glueball_ref = (glueball_label_input.value, float(glueball_ref_input.value))

            sqrt_sigma_ref = None
            if sqrt_sigma_ref_input.value is not None:
                sqrt_sigma_ref = float(sqrt_sigma_ref_input.value)

            particle_mass_table.value = pd.DataFrame(_build_algorithmic_mass_rows(masses))
            particle_fit_table.value = pd.DataFrame(_build_best_fit_rows(masses))
            particle_anchor_table.value = pd.DataFrame(
                _build_anchor_rows(masses, glueball_ref, sqrt_sigma_ref)
            )
            _set_particle_status("**Particles:** tables updated.")

        def _default_sweep_range(param_name: str) -> tuple[float, float, int]:
            current = getattr(analysis_settings, param_name, 1.0)
            if current is None:
                current = 1.0
            param_type = SWEEP_PARAM_DEFS.get(param_name, {}).get("type", float)
            if param_type is int:
                base = int(current)
                min_v = max(1, base - 5)
                max_v = max(min_v + 1, base + 5)
            else:
                base = float(current)
                if base == 0:
                    base = 1.0
                min_v = max(1e-6, base * 0.5)
                max_v = max(min_v * 1.1, base * 1.5)
            return float(min_v), float(max_v), 5

        def _sync_sweep_bounds(param_name: str, min_w, max_w, steps_w) -> None:
            min_v, max_v, steps = _default_sweep_range(param_name)
            min_w.value = min_v
            max_w.value = max_v
            steps_w.value = steps

        def _toggle_sweep_controls(event) -> None:
            enabled = bool(event.new)
            sweep_param_y.visible = enabled
            sweep_min_y.visible = enabled
            sweep_max_y.visible = enabled
            sweep_steps_y.visible = enabled

        def _on_sweep_param_x(event) -> None:
            _sync_sweep_bounds(event.new, sweep_min_x, sweep_max_x, sweep_steps_x)

        def _on_sweep_param_y(event) -> None:
            _sync_sweep_bounds(event.new, sweep_min_y, sweep_max_y, sweep_steps_y)

        _sync_sweep_bounds(sweep_param_x.value, sweep_min_x, sweep_max_x, sweep_steps_x)
        _sync_sweep_bounds(sweep_param_y.value, sweep_min_y, sweep_max_y, sweep_steps_y)
        for widget in (sweep_param_y, sweep_min_y, sweep_max_y, sweep_steps_y):
            widget.visible = sweep_enable_2d.value

        def set_history(history: RunHistory, history_path: Path | None = None) -> None:
            state["history"] = history
            state["history_path"] = history_path
            visualizer.bounds_extent = float(gas_config.bounds_extent)
            visualizer.set_history(history)
            _set_analysis_status("**Analysis ready:** click Run Analysis.")
            run_analysis_button.disabled = False
            run_analysis_button_main.disabled = False
            particle_run_button.disabled = False
            _set_particle_status("**Particles ready:** click Compute Particles.")
            sweep_run_button.disabled = False

        def on_simulation_complete(history: RunHistory):
            set_history(history)

        def _infer_bounds_extent(history: RunHistory) -> float | None:
            if history.bounds is None:
                return None
            high = history.bounds.high.detach().cpu().abs().max().item()
            low = history.bounds.low.detach().cpu().abs().max().item()
            return float(max(high, low))

        def _sync_history_path(value):
            if value:
                history_path_input.value = str(value[0])

        def _ensure_file_selector() -> pn.widgets.FileSelector:
            if file_selector_container.objects:
                return file_selector_container.objects[0]
            selector = pn.widgets.FileSelector(
                name="Select RunHistory",
                directory=str(history_dir),
                file_pattern="*_history.pt",
                only_files=True,
            )
            if qft_history_path.exists():
                selector.value = [str(qft_history_path)]
            selector.param.watch(lambda e: _sync_history_path(e.new), "value")
            file_selector_container.objects = [selector]
            return selector

        def _on_browse_clicked(_):
            _ensure_file_selector()

        def on_load_clicked(_):
            history_path = Path(history_path_input.value).expanduser()
            if not history_path.exists():
                load_status.object = "**Error:** History path does not exist."
                return
            try:
                history = RunHistory.load(str(history_path))
                inferred_extent = _infer_bounds_extent(history)
                if inferred_extent is not None:
                    visualizer.bounds_extent = inferred_extent
                    gas_config.bounds_extent = inferred_extent
                set_history(history, history_path)
                load_status.object = f"**Loaded:** `{history_path}`"
            except Exception as exc:
                load_status.object = f"**Error loading history:** {exc!s}"

        def on_bounds_change(event):
            visualizer.bounds_extent = float(event.new)
            visualizer._refresh_frame()

        def _run_analysis(force_particles: bool) -> tuple[dict[str, Any], dict[str, Any]] | None:
            history = state.get("history")
            if history is None:
                _set_analysis_status("**Error:** load or run a simulation first.")
                _set_particle_status("**Error:** load or run a simulation first.")
                return None

            if force_particles:
                analysis_settings.compute_particles = True
                if glueball_ref_input.value is not None:
                    analysis_settings.build_fractal_set = True
                if sqrt_sigma_ref_input.value is not None:
                    analysis_settings.compute_string_tension = True
                    analysis_settings.build_fractal_set = True

            output_dir = Path(analysis_output_dir.value)
            output_dir.mkdir(parents=True, exist_ok=True)
            analysis_id = analysis_id_input.value.strip() or datetime.utcnow().strftime(
                "%Y%m%d_%H%M%S"
            )
            analysis_id_input.value = analysis_id

            history_path = state.get("history_path")
            if history_path is None or not history_path.exists():
                history_path = output_dir / f"{analysis_id}_history.pt"
                history.save(str(history_path))
                state["history_path"] = history_path

            args = analysis_settings.to_cli_args(history_path, output_dir, analysis_id)
            _set_analysis_status("**Running analysis...**")
            _set_particle_status("**Running analysis...**")

            try:
                with _temporary_argv(args):
                    qft_analysis.main()
            except Exception as exc:
                _set_analysis_status(f"**Error:** {exc!s}")
                _set_particle_status(f"**Error:** {exc!s}")
                return None

            metrics_path = output_dir / f"{analysis_id}_metrics.json"
            arrays_path = output_dir / f"{analysis_id}_arrays.npz"
            if not metrics_path.exists():
                _set_analysis_status("**Error:** analysis metrics file missing.")
                _set_particle_status("**Error:** analysis metrics file missing.")
                return None

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if arrays_path.exists():
                with np.load(arrays_path) as data:
                    arrays = {key: data[key] for key in data.files}
            else:
                arrays = {}

            state["analysis_metrics"] = metrics
            state["analysis_arrays"] = arrays
            return metrics, arrays

        def _update_analysis_outputs(metrics: dict[str, Any], arrays: dict[str, Any]) -> None:
            analysis_summary.object = _format_analysis_summary(metrics)
            analysis_json.object = metrics
            analysis_plots.objects = [
                pn.pane.HoloViews(plot, sizing_mode="stretch_width")
                for plot in _build_analysis_plots(metrics, arrays)
            ]

        def on_run_analysis(_):
            result = _run_analysis(force_particles=False)
            if result is None:
                return
            metrics, arrays = result
            _update_analysis_outputs(metrics, arrays)
            _set_analysis_status("**Analysis complete.**")
            _update_particle_tables(metrics)

        def on_run_particles(_):
            result = _run_analysis(force_particles=True)
            if result is None:
                return
            metrics, arrays = result
            _update_analysis_outputs(metrics, arrays)
            _set_analysis_status("**Analysis complete.**")
            _update_particle_tables(metrics)

        def on_run_sweep(_):
            history = state.get("history")
            if history is None:
                _set_analysis_status("**Error:** load or run a simulation first.")
                _set_particle_status("**Error:** load or run a simulation first.")
                return

            x_param = sweep_param_x.value
            y_param = sweep_param_y.value if sweep_enable_2d.value else None
            metric_key = sweep_metric.value

            steps_x = max(1, int(sweep_steps_x.value))
            x_values = _build_sweep_values(
                x_param,
                float(sweep_min_x.value),
                float(sweep_max_x.value),
                steps_x,
            )
            if y_param:
                steps_y = max(1, int(sweep_steps_y.value))
                y_values = _build_sweep_values(
                    y_param,
                    float(sweep_min_y.value),
                    float(sweep_max_y.value),
                    steps_y,
                )
            else:
                y_values = [None]

            original_values = {
                x_param: getattr(analysis_settings, x_param),
                "compute_particles": analysis_settings.compute_particles,
                "build_fractal_set": analysis_settings.build_fractal_set,
                "compute_string_tension": analysis_settings.compute_string_tension,
            }
            if y_param:
                original_values[y_param] = getattr(analysis_settings, y_param)

            original_analysis_id = analysis_id_input.value
            base_id = original_analysis_id.strip() or datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            results: list[dict[str, Any]] = []
            skipped = 0
            attempted = 0
            sweep_status.object = "**Sweep:** running..."
            include_fit_meta = {
                "particle_fit_start",
                "particle_fit_stop",
            }.intersection({x_param, y_param or ""})
            fit_operator = _metric_operator(metric_key)

            try:
                for x_val in x_values:
                    x_val_cast = _coerce_sweep_value(x_param, x_val)
                    setattr(analysis_settings, x_param, x_val_cast)
                    for y_val in y_values:
                        suffix = f"{x_param}_{x_val_cast}"
                        if y_param and y_val is not None:
                            y_val_cast = _coerce_sweep_value(y_param, y_val)
                            setattr(analysis_settings, y_param, y_val_cast)
                            suffix = f"{suffix}__{y_param}_{y_val_cast}"
                        else:
                            y_val_cast = None

                        fit_start, fit_stop = _resolve_fit_window(
                            analysis_settings, x_param, x_val_cast, y_param, y_val_cast
                        )
                        if (
                            fit_start is not None
                            and fit_stop is not None
                            and fit_stop < fit_start
                        ):
                            skipped += 1
                            row = {
                                x_param: x_val_cast,
                                metric_key: np.nan,
                                "status": "invalid_fit_window",
                                "fit_start": fit_start,
                                "fit_stop": fit_stop,
                            }
                            if y_param:
                                row[y_param] = y_val_cast
                            row.update(
                                {
                                    "baryon_mass": np.nan,
                                    "baryon_r2": np.nan,
                                    "meson_mass": np.nan,
                                    "meson_r2": np.nan,
                                    "glueball_mass": np.nan,
                                    "glueball_r2": np.nan,
                                }
                            )
                            results.append(row)
                            continue

                        analysis_id_input.value = f"{base_id}_sweep_{suffix}"
                        attempted += 1
                        result = _run_analysis(force_particles=True)
                        if result is None:
                            skipped += 1
                            row = {
                                x_param: x_val_cast,
                                metric_key: np.nan,
                                "status": "analysis_error",
                                "fit_start": fit_start,
                                "fit_stop": fit_stop,
                            }
                            if y_param:
                                row[y_param] = y_val_cast
                            row.update(
                                {
                                    "baryon_mass": np.nan,
                                    "baryon_r2": np.nan,
                                    "meson_mass": np.nan,
                                    "meson_r2": np.nan,
                                    "glueball_mass": np.nan,
                                    "glueball_r2": np.nan,
                                }
                            )
                            results.append(row)
                            continue
                        metrics, _arrays = result
                        value = _extract_metric(metrics, metric_key)
                        row = {x_param: x_val_cast, metric_key: value}
                        if y_param:
                            row[y_param] = y_val_cast
                        row.update(_extract_particle_fit_metrics(metrics))
                        if fit_start is not None:
                            row["fit_start"] = fit_start
                        if fit_stop is not None:
                            row["fit_stop"] = fit_stop
                        if include_fit_meta and fit_operator is not None:
                            fit_start_used, fit_stop_used, fit_mode_used = _extract_fit_metadata(
                                metrics, fit_operator
                            )
                            row["fit_start_used"] = fit_start_used
                            row["fit_stop_used"] = fit_stop_used
                            row["fit_mode_used"] = fit_mode_used
                        results.append(row)
            finally:
                for name, value in original_values.items():
                    setattr(analysis_settings, name, value)
                analysis_id_input.value = original_analysis_id

            if not results:
                sweep_status.object = "**Sweep:** no results."
                sweep_table.value = pd.DataFrame()
                sweep_plot.object = None
                return

            df = pd.DataFrame(results)
            sweep_table.value = df
            plot = _build_sweep_plot(df, x_param, metric_key, y_param=y_param)
            sweep_plot.object = plot
            if skipped:
                sweep_status.object = (
                    f"**Sweep:** complete ({attempted} runs, {skipped} skipped)."
                )
            else:
                sweep_status.object = f"**Sweep:** complete ({attempted} runs)."

        browse_button.on_click(_on_browse_clicked)
        load_button.on_click(on_load_clicked)
        gas_config.add_completion_callback(on_simulation_complete)
        gas_config.param.watch(on_bounds_change, "bounds_extent")
        run_analysis_button.on_click(on_run_analysis)
        run_analysis_button_main.on_click(on_run_analysis)
        particle_run_button.on_click(on_run_particles)
        sweep_run_button.on_click(on_run_sweep)
        sweep_enable_2d.param.watch(_toggle_sweep_controls, "value")
        sweep_param_x.param.watch(_on_sweep_param_x, "value")
        sweep_param_y.param.watch(_on_sweep_param_y, "value")

        visualization_controls = pn.Param(
            visualizer,
            parameters=["point_size", "point_alpha", "color_metric", "fix_axes"],
            show_name=False,
        )

        analysis_core = pn.Param(
            analysis_settings,
            parameters=[
                "analysis_time_index",
                "analysis_step",
                "warmup_fraction",
                "h_eff",
                "correlation_r_max",
                "correlation_bins",
                "gradient_neighbors",
                "build_fractal_set",
                "fractal_set_stride",
            ],
            show_name=False,
        )
        analysis_local = pn.Param(
            analysis_settings,
            parameters=["use_local_fields", "use_connected", "density_sigma"],
            show_name=False,
        )
        analysis_particles = pn.Param(
            analysis_settings,
            parameters=[
                "compute_particles",
                "particle_operators",
                "particle_max_lag",
                "particle_fit_start",
                "particle_fit_stop",
                "particle_fit_mode",
                "particle_plateau_min_points",
                "particle_plateau_max_points",
                "particle_plateau_max_cv",
                "particle_mass",
                "particle_ell0",
                "particle_use_connected",
                "particle_neighbor_method",
                "particle_knn_k",
                "particle_knn_sample",
                "particle_meson_reduce",
                "particle_baryon_pairs",
            ],
            show_name=False,
        )
        analysis_string = pn.Param(
            analysis_settings,
            parameters=[
                "compute_string_tension",
                "string_tension_max_triangles",
                "string_tension_bins",
            ],
            show_name=False,
        )

        analysis_output = pn.Column(
            analysis_output_dir,
            analysis_id_input,
            run_analysis_button,
            analysis_status_sidebar,
            sizing_mode="stretch_width",
        )
        particle_anchor_controls = pn.Column(
            glueball_label_input,
            glueball_ref_input,
            sqrt_sigma_ref_input,
            sizing_mode="stretch_width",
        )
        sweep_controls = pn.Column(
            pn.pane.Markdown("### Sweep Controls"),
            pn.Row(sweep_enable_2d, sweep_metric, sizing_mode="stretch_width"),
            pn.Row(
                sweep_param_x,
                sweep_min_x,
                sweep_max_x,
                sweep_steps_x,
                sizing_mode="stretch_width",
            ),
            pn.Row(
                sweep_param_y,
                sweep_min_y,
                sweep_max_y,
                sweep_steps_y,
                sizing_mode="stretch_width",
            ),
            pn.Row(sweep_run_button, sizing_mode="stretch_width"),
            sweep_status,
            pn.layout.Divider(),
            sweep_plot,
            sweep_table,
            sizing_mode="stretch_width",
        )

        if skip_sidebar:
            sidebar.objects = [
                pn.pane.Markdown(
                    "## QFT Dashboard\n" "Sidebar disabled via QFT_DASH_SKIP_SIDEBAR=1."
                ),
                pn.pane.Markdown("### Load QFT RunHistory"),
                history_path_input,
                browse_button,
                file_selector_container,
                load_button,
                load_status,
            ]
        else:
            sidebar.objects = [
                pn.pane.Markdown(
                    "## QFT Dashboard\n"
                    "Run a simulation or load a RunHistory, then analyze results."
                ),
                pn.Accordion(
                    (
                        "RunHistory",
                        pn.Column(
                            history_path_input,
                            browse_button,
                            file_selector_container,
                            load_button,
                            load_status,
                            sizing_mode="stretch_width",
                        ),
                    ),
                    ("Simulation", gas_config.panel()),
                    ("Visualization", visualization_controls),
                    ("Analysis: Core", analysis_core),
                    ("Analysis: Local Fields", analysis_local),
                    ("Analysis: Particles", analysis_particles),
                    ("Analysis: String Tension", analysis_string),
                    ("Analysis: Output", analysis_output),
                    ("Particle Anchors", particle_anchor_controls),
                    sizing_mode="stretch_width",
                ),
            ]

        if skip_visual:
            main.objects = [
                pn.pane.Markdown(
                    "Visualization disabled via QFT_DASH_SKIP_VIS=1.",
                    sizing_mode="stretch_both",
                )
            ]
        else:
            simulation_tab = pn.Column(visualizer.panel(), sizing_mode="stretch_both")
            analysis_tab = pn.Column(
                analysis_status_main,
                pn.Row(run_analysis_button_main, sizing_mode="stretch_width"),
                analysis_summary,
                pn.layout.Divider(),
                analysis_plots,
                pn.layout.Divider(),
                pn.pane.Markdown("## Raw Metrics"),
                analysis_json,
                sizing_mode="stretch_both",
            )
            particle_tab = pn.Column(
                particle_status,
                pn.Row(particle_run_button, sizing_mode="stretch_width"),
                sweep_controls,
                pn.pane.Markdown("### Algorithmic Masses"),
                particle_mass_table,
                pn.pane.Markdown("### Best-Fit Scales"),
                particle_fit_table,
                pn.pane.Markdown("### Anchored Mass Table"),
                particle_anchor_table,
                sizing_mode="stretch_both",
            )

            main.objects = [
                pn.Tabs(
                    ("Simulation", simulation_tab),
                    ("Analysis", analysis_tab),
                    ("Particles", particle_tab),
                )
            ]

        _debug(f"ui ready ({time.time() - start_total:.2f}s)")

    pn.state.onload(_build_ui)
    return template


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5007)
    parser.add_argument("--open", action="store_true", help="Open browser on launch")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print("Starting QFT Swarm Convergence Dashboard...", flush=True)
    app = create_app()
    print(
        f"QFT Swarm Convergence Dashboard running at http://localhost:{args.port} "
        f"(use --open to launch a browser)",
        flush=True,
    )
    app.show(port=args.port, open=args.open)
