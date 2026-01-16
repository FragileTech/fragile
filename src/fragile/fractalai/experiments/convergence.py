"""Convergence analysis dashboard for Gas algorithms.

This module provides the ConvergencePanel class for analyzing convergence
behavior from RunHistory, including KL divergence, Lyapunov decay, and
exponential fit estimation.

Can be used standalone or integrated into visualization dashboards.
"""

from __future__ import annotations

import holoviews as hv
from holoviews import opts
import numpy as np
import pandas as pd
import panel as pn
import param
from scipy.stats import gaussian_kde
import torch

from fragile.fractalai.bounds import TorchBounds
from fragile.fractalai.core.history import RunHistory


__all__ = ["ConvergencePanel"]


class ConvergencePanel(param.Parameterized):
    """Panel for convergence analysis visualization.

    Computes and displays convergence metrics from RunHistory:
    - KL divergence: KL(empirical || target) works with any potential
    - Lyapunov function: V_total = Var[x] + Var[v]
    - Distance to optimum: if benchmark has best_state property
    - Exponential decay fits and convergence rates
    """

    # Parameters
    kl_n_samples = param.Integer(
        default=1000,
        bounds=(100, 5000),
        doc="Number of samples for KL divergence estimation (via KDE)",
    )
    fit_start_time = param.Integer(
        default=50,
        bounds=(0, 500),
        doc="Start exponential fit after this time step (skip transient)",
    )

    def __init__(
        self,
        history: RunHistory | None,
        potential: object,
        benchmark: object,
        bounds_extent: float,
        **params,
    ):
        """Initialize convergence analysis panel.

        Args:
            history: RunHistory object (can be None initially)
            potential: Potential function for KL computation
            benchmark: Benchmark object (may have best_state property)
            bounds_extent: Spatial bounds half-width
            **params: Override default parameters
        """
        super().__init__(**params)
        self.history = history
        self.potential = potential
        self.benchmark = benchmark
        self.bounds_extent = bounds_extent

        # Computed metrics (populated by compute_metrics())
        self.metrics = None
        self.computing = False

        # UI elements
        self.compute_button = pn.widgets.Button(
            name="Compute Convergence Metrics",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        self.compute_button.on_click(self._on_compute_click)

        self.status_pane = pn.pane.Markdown(
            "**Status:** Load a RunHistory and click 'Compute' to analyze convergence.",
            sizing_mode="stretch_width",
        )

        # Plot container (populated after computation)
        # Use a Column instead of Markdown pane so it can hold any content type
        self.plot_pane = pn.Column(
            pn.pane.Markdown(
                "**Click 'Compute Convergence Metrics' to generate plots**",
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
            min_height=400,
        )

    def set_history(self, history: RunHistory):
        """Update history for analysis."""
        self.history = history
        self.metrics = None  # Reset metrics
        self.status_pane.object = (
            f"**Status:** RunHistory loaded (N={history.N}, "
            f"steps={history.n_recorded}). Click 'Compute' to analyze."
        )
        # Update plot container with instruction message
        self.plot_pane.objects = [
            pn.pane.Markdown(
                "**Click 'Compute Convergence Metrics' to generate plots**",
                sizing_mode="stretch_width",
            )
        ]

    def update_benchmark(self, potential: object):
        """Update benchmark/potential function."""
        self.potential = potential
        self.benchmark = potential
        self.metrics = None  # Reset metrics when benchmark changes

    def _on_compute_click(self, event):
        """Handle compute button click."""
        if self.history is None:
            self.status_pane.object = "**Error:** No RunHistory loaded. Run a simulation first."
            return

        if self.computing:
            return

        self.computing = True
        self.status_pane.object = "**Status:** Computing convergence metrics..."
        self.compute_button.disabled = True

        try:
            self.compute_metrics()
            self._update_plots()
            self.status_pane.object = "**Status:** ✅ Convergence metrics computed successfully."
        except Exception as e:
            self.status_pane.object = f"**Error:** Failed to compute metrics: {e}"
        finally:
            self.computing = False
            self.compute_button.disabled = False

    def compute_metrics(self):
        """Compute all convergence metrics from RunHistory."""
        if self.history is None:
            return

        history = self.history
        n_recorded = history.n_recorded

        # Initialize storage
        times = []
        kl_divergences = []
        lyapunov_values = []
        var_x_values = []
        var_v_values = []
        distance_to_opt_values = []

        # Check if benchmark has best_state
        has_optimum = (
            hasattr(self.benchmark, "best_state") and self.benchmark.best_state is not None
        )

        # Extract target optimum if available
        if has_optimum:
            try:
                best_state = self.benchmark.best_state
                if isinstance(best_state, torch.Tensor):
                    target_opt = best_state.cpu().numpy()
                else:
                    target_opt = np.array(best_state)
            except Exception:
                has_optimum = False

        # Compute metrics for each recorded time step
        for t_idx in range(n_recorded):
            time = t_idx
            times.append(time)

            # Extract positions
            x_t = history.x_final[t_idx].detach().cpu()

            # Check if any walkers alive
            low = torch.full((history.d,), -self.bounds_extent, dtype=torch.float32)
            high = torch.full((history.d,), self.bounds_extent, dtype=torch.float32)
            bounds = TorchBounds(low=low, high=high)
            alive_mask = bounds.contains(x_t)

            if alive_mask.sum() == 0:
                # No alive walkers
                kl_divergences.append(float("inf"))
                lyapunov_values.append(float("nan"))
                var_x_values.append(float("nan"))
                var_v_values.append(float("nan"))
                if has_optimum:
                    distance_to_opt_values.append(float("nan"))
                continue

            # Extract alive positions
            x_alive = x_t[alive_mask].numpy()

            # Compute KL divergence
            kl = self._compute_kl_divergence(x_alive)
            kl_divergences.append(kl)

            # Compute variances
            var_x = torch.var(x_t, dim=0).sum().item()
            var_v = torch.var(history.v_final[t_idx], dim=0).sum().item()
            lyapunov = var_x + var_v

            var_x_values.append(var_x)
            var_v_values.append(var_v)
            lyapunov_values.append(lyapunov)

            # Compute distance to optimum if available
            if has_optimum:
                mean_pos = x_alive.mean(axis=0)
                dist = np.linalg.norm(mean_pos - target_opt[: history.d])
                distance_to_opt_values.append(dist)

        # Store metrics
        self.metrics = {
            "times": np.array(times),
            "kl_divergence": np.array(kl_divergences),
            "lyapunov": np.array(lyapunov_values),
            "var_x": np.array(var_x_values),
            "var_v": np.array(var_v_values),
            "distance_to_opt": np.array(distance_to_opt_values) if has_optimum else None,
            "has_optimum": has_optimum,
        }

    def _compute_kl_divergence(self, samples: np.ndarray) -> float:
        """Compute KL(empirical || target) for arbitrary potential.

        Uses KDE for empirical distribution and evaluates target as p(x) ∝ exp(-U(x)).

        Args:
            samples: Alive walker positions [N_alive, d]

        Returns:
            KL divergence (non-negative) or inf if computation fails
        """
        if len(samples) < 10:
            return float("inf")

        try:
            # Create KDE from samples
            kde = gaussian_kde(samples.T, bw_method="scott")

            # Sample from empirical distribution
            grid_samples = kde.resample(self.kl_n_samples).T
            grid_samples_torch = torch.tensor(grid_samples, dtype=torch.float32)

            # Evaluate empirical density
            p_emp = kde(grid_samples.T)

            # Evaluate target density (unnormalized)
            with torch.no_grad():
                U_vals = self.potential(grid_samples_torch).cpu().numpy()
            p_target_unnorm = np.exp(-U_vals)

            # Estimate partition function (Monte Carlo)
            Z_estimate = np.mean(p_target_unnorm)
            if Z_estimate <= 0:
                return float("inf")

            # Normalized target density
            p_target = p_target_unnorm / (Z_estimate * self.kl_n_samples)

            # Compute KL divergence with numerical stability
            mask = (p_emp > 1e-10) & (p_target > 1e-10)
            if not mask.any():
                return float("inf")

            kl = np.sum(p_emp[mask] * np.log(p_emp[mask] / p_target[mask]))

            return max(0.0, kl)

        except Exception:
            return float("inf")

    def _fit_exponential_decay(
        self, times: np.ndarray, values: np.ndarray
    ) -> tuple[float, float, float] | None:
        """Fit exponential decay y = C·exp(-κ·t).

        Args:
            times: Time array
            values: Metric values

        Returns:
            (kappa, C, r_squared) or None if fit fails
        """
        # Filter to fit region
        mask = (times >= self.fit_start_time) & np.isfinite(values) & (values > 0)
        if mask.sum() < 10:
            return None

        t_fit = times[mask]
        y_fit = values[mask]

        try:
            # Log-linear fit: log(y) = log(C) - κ·t
            log_y = np.log(y_fit)
            coeffs = np.polyfit(t_fit, log_y, 1)

            kappa = -coeffs[0]  # Convergence rate
            C = np.exp(coeffs[1])  # Initial value

            # R² goodness of fit
            y_pred = C * np.exp(-kappa * t_fit)
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return kappa, C, r_squared

        except Exception:
            return None

    def _update_plots(self):
        """Update visualization plots with computed metrics."""
        if self.metrics is None:
            return

        times = self.metrics["times"]
        kl_div = self.metrics["kl_divergence"]
        lyapunov = self.metrics["lyapunov"]
        var_x = self.metrics["var_x"]
        var_v = self.metrics["var_v"]
        dist_opt = self.metrics["distance_to_opt"]
        has_optimum = self.metrics["has_optimum"]

        # Create plots
        plots = []

        # 1. KL Divergence plot
        kl_plot = self._plot_kl_convergence(times, kl_div)
        if kl_plot is not None:
            plots.append(kl_plot)

        # 2. Lyapunov plot
        lyap_plot = self._plot_lyapunov_decay(times, lyapunov)
        if lyap_plot is not None:
            plots.append(lyap_plot)

        # 3. Variances plot
        var_plot = self._plot_variances(times, var_x, var_v)
        if var_plot is not None:
            plots.append(var_plot)

        # 4. Distance to optimum (if available)
        if has_optimum and dist_opt is not None:
            dist_plot = self._plot_distance_to_optimum(times, dist_opt)
            if dist_plot is not None:
                plots.append(dist_plot)

        # Create summary statistics table
        summary_table = self._summary_statistics_table()
        if summary_table is not None:
            plots.append(summary_table)

        # Layout plots in grid and update container
        if plots:
            layout = hv.Layout(plots).opts(opts.Layout(shared_axes=False)).cols(2)
            # Wrap HoloViews layout in pn.panel() for proper rendering
            self.plot_pane.objects = [pn.panel(layout, sizing_mode="stretch_width")]
        else:
            # Show message when no valid data
            self.plot_pane.objects = [
                pn.pane.Markdown(
                    "**No valid convergence data to plot**", sizing_mode="stretch_width"
                )
            ]

    def _plot_kl_convergence(self, times: np.ndarray, kl: np.ndarray):
        """Plot KL divergence vs time with exponential fit."""
        # Filter valid data
        valid_mask = np.isfinite(kl) & (kl > 0) & (kl < 1e6)
        if not valid_mask.any():
            return None

        t_valid = times[valid_mask]
        kl_valid = kl[valid_mask]

        # Scatter plot
        scatter = hv.Scatter((t_valid, kl_valid), kdims=["time"], vdims=["KL"]).opts(
            size=5, color="blue", alpha=0.6
        )

        # Try exponential fit
        fit_result = self._fit_exponential_decay(times, kl)
        if fit_result is not None:
            kappa, C, r_sq = fit_result
            t_fit = np.linspace(self.fit_start_time, t_valid.max(), 100)
            kl_fit = C * np.exp(-kappa * t_fit)

            fit_curve = hv.Curve((t_fit, kl_fit), kdims=["time"], vdims=["KL"]).opts(
                color="red", line_width=2, line_dash="dashed"
            )

            half_life = np.log(2) / kappa if kappa > 0 else float("inf")
            title = f"KL Divergence (κ={kappa:.4f}, t₁/₂={half_life:.1f}, R²={r_sq:.3f})"
        else:
            fit_curve = hv.Curve([])
            title = "KL Divergence (no exponential fit)"

        return (scatter * fit_curve).opts(
            width=400,
            height=350,
            title=title,
            xlabel="Time",
            ylabel="KL(empirical || target)",
            logy=True,
            show_grid=True,
            framewise=True,
        )

    def _plot_lyapunov_decay(self, times: np.ndarray, lyapunov: np.ndarray):
        """Plot Lyapunov function vs time with exponential fit."""
        valid_mask = np.isfinite(lyapunov) & (lyapunov > 0)
        if not valid_mask.any():
            return None

        t_valid = times[valid_mask]
        lyap_valid = lyapunov[valid_mask]

        scatter = hv.Scatter((t_valid, lyap_valid), kdims=["time"], vdims=["V_total"]).opts(
            size=5, color="green", alpha=0.6
        )

        fit_result = self._fit_exponential_decay(times, lyapunov)
        if fit_result is not None:
            kappa, C, r_sq = fit_result
            t_fit = np.linspace(self.fit_start_time, t_valid.max(), 100)
            lyap_fit = C * np.exp(-kappa * t_fit)

            fit_curve = hv.Curve((t_fit, lyap_fit), kdims=["time"], vdims=["V_total"]).opts(
                color="red", line_width=2, line_dash="dashed"
            )

            title = f"Lyapunov V_total (κ={kappa:.4f}, R²={r_sq:.3f})"
        else:
            fit_curve = hv.Curve([])
            title = "Lyapunov V_total (no exponential fit)"

        return (scatter * fit_curve).opts(
            width=400,
            height=350,
            title=title,
            xlabel="Time",
            ylabel="V_total = Var[x] + Var[v]",
            logy=True,
            show_grid=True,
            framewise=True,
        )

    def _plot_variances(self, times: np.ndarray, var_x: np.ndarray, var_v: np.ndarray):
        """Plot position and velocity variances vs time."""
        valid_mask_x = np.isfinite(var_x) & (var_x > 0)
        valid_mask_v = np.isfinite(var_v) & (var_v > 0)

        if not (valid_mask_x.any() or valid_mask_v.any()):
            return None

        curves = []
        if valid_mask_x.any():
            curve_x = hv.Curve(
                (times[valid_mask_x], var_x[valid_mask_x]),
                kdims=["time"],
                vdims=["Variance"],
                label="Var[x]",
            ).opts(color="blue", line_width=2)
            curves.append(curve_x)

        if valid_mask_v.any():
            curve_v = hv.Curve(
                (times[valid_mask_v], var_v[valid_mask_v]),
                kdims=["time"],
                vdims=["Variance"],
                label="Var[v]",
            ).opts(color="orange", line_width=2)
            curves.append(curve_v)

        if not curves:
            return None

        return hv.Overlay(curves).opts(
            width=400,
            height=350,
            title="Position & Velocity Variances",
            xlabel="Time",
            ylabel="Variance",
            logy=True,
            show_grid=True,
            legend_position="right",
            framewise=True,
        )

    def _plot_distance_to_optimum(self, times: np.ndarray, distances: np.ndarray):
        """Plot distance to known optimum vs time."""
        valid_mask = np.isfinite(distances) & (distances >= 0)
        if not valid_mask.any():
            return None

        t_valid = times[valid_mask]
        dist_valid = distances[valid_mask]

        curve = hv.Curve((t_valid, dist_valid), kdims=["time"], vdims=["Distance"]).opts(
            color="purple", line_width=2
        )

        return curve.opts(
            width=400,
            height=350,
            title="Distance to Known Optimum",
            xlabel="Time",
            ylabel="||mean(x) - x*||",
            show_grid=True,
            framewise=True,
        )

    def _summary_statistics_table(self):
        """Create summary statistics table."""
        if self.metrics is None:
            return None

        times = self.metrics["times"]
        kl = self.metrics["kl_divergence"]
        lyap = self.metrics["lyapunov"]
        var_x = self.metrics["var_x"]
        var_v = self.metrics["var_v"]

        # Final values
        stats = []

        # KL statistics
        kl_fit = self._fit_exponential_decay(times, kl)
        if kl_fit is not None:
            kappa_kl, _C_kl, r_sq_kl = kl_fit
            half_life_kl = np.log(2) / kappa_kl if kappa_kl > 0 else float("inf")
            stats.extend((
                ("KL Convergence Rate κ", f"{kappa_kl:.5f}"),
                ("KL Half-Life t₁/₂", f"{half_life_kl:.2f}"),
                ("KL Fit R²", f"{r_sq_kl:.4f}"),
            ))

        final_kl = kl[np.isfinite(kl)][-1] if len(kl[np.isfinite(kl)]) > 0 else float("nan")
        stats.append(("Final KL Divergence", f"{final_kl:.4f}"))

        # Lyapunov statistics
        lyap_fit = self._fit_exponential_decay(times, lyap)
        if lyap_fit is not None:
            kappa_lyap, _C_lyap, r_sq_lyap = lyap_fit
            stats.extend((
                ("Lyapunov Decay Rate κ", f"{kappa_lyap:.5f}"),
                ("Lyapunov Fit R²", f"{r_sq_lyap:.4f}"),
            ))

        final_lyap = (
            lyap[np.isfinite(lyap)][-1] if len(lyap[np.isfinite(lyap)]) > 0 else float("nan")
        )
        stats.append(("Final V_total", f"{final_lyap:.4f}"))

        # Variance statistics
        final_var_x = (
            var_x[np.isfinite(var_x)][-1] if len(var_x[np.isfinite(var_x)]) > 0 else float("nan")
        )
        final_var_v = (
            var_v[np.isfinite(var_v)][-1] if len(var_v[np.isfinite(var_v)]) > 0 else float("nan")
        )
        stats.extend((
            ("Final Var[x]", f"{final_var_x:.4f}"),
            ("Final Var[v]", f"{final_var_v:.4f}"),
        ))

        # Distance to optimum
        if self.metrics["has_optimum"] and self.metrics["distance_to_opt"] is not None:
            dist = self.metrics["distance_to_opt"]
            final_dist = (
                dist[np.isfinite(dist)][-1] if len(dist[np.isfinite(dist)]) > 0 else float("nan")
            )
            stats.append(("Final Distance to Optimum", f"{final_dist:.4f}"))

        # Create table
        df = pd.DataFrame(stats, columns=["Metric", "Value"])
        return hv.Table(df).opts(
            width=800,
            height=400,
            title="Convergence Summary Statistics",
        )

    def panel(self) -> pn.Column:
        """Create Panel layout for convergence analysis.

        Returns:
            Panel Column with compute button, status, and plots
        """
        return pn.Column(
            pn.pane.Markdown("### Convergence Metrics"),
            pn.pane.Markdown(
                "Analyze convergence behavior by computing KL divergence, "
                "Lyapunov decay, and other metrics from the simulation history."
            ),
            self.compute_button,
            self.status_pane,
            pn.Param(
                self.param,
                parameters=["kl_n_samples", "fit_start_time"],
                show_name=False,
            ),
            pn.pane.Markdown("---"),
            self.plot_pane,
            sizing_mode="stretch_width",
        )
