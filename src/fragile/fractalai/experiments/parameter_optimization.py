"""Parameter optimization and analysis dashboard for Gas algorithms.

This module provides the ConvergenceBoundsPanel class for analyzing theoretical
convergence bounds, validating parameter regimes, and optimizing parameters
based on empirical data from simulations.

Can be used standalone or integrated into visualization dashboards.
"""

from __future__ import annotations

import panel as pn
import param

import fragile.fractalai.convergence_bounds as cb
from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.experiments.gas_config_dashboard import GasConfig


__all__ = ["ConvergenceBoundsPanel"]


class ConvergenceBoundsPanel(param.Parameterized):
    """Panel for theoretical convergence bounds and diagnostics.

    Computes theoretical convergence bounds from the mathematical framework,
    validates parameter regimes, performs sensitivity analysis, and compares
    theoretical predictions with empirical convergence rates.
    """

    def __init__(
        self,
        gas_config: GasConfig,
        potential: object,
        bounds_extent: float,
        history: RunHistory | None = None,
        **params,
    ):
        """Initialize convergence bounds diagnostics panel.

        Args:
            gas_config: GasConfig object with simulation parameters
            potential: Potential function object
            bounds_extent: Spatial bounds half-width
            history: Optional RunHistory for empirical analysis
            **params: Override default parameters
        """
        super().__init__(**params)
        self.gas_config = gas_config
        self.potential = potential
        self.bounds_extent = bounds_extent
        self.history = history

        # Computed diagnostics (populated by compute_diagnostics())
        self.diagnostics = None
        self.computing = False

        # UI elements
        self.compute_button = pn.widgets.Button(
            name="Compute Theoretical Bounds",
            button_type="success",
            sizing_mode="stretch_width",
        )
        self.compute_button.on_click(self._on_compute_click)

        self.status_pane = pn.pane.Markdown(
            "**Status:** Click 'Compute' to analyze theoretical convergence bounds.",
            sizing_mode="stretch_width",
        )

        # Result panes (populated after computation)
        self.environment_pane = pn.pane.Markdown(
            "**Environment Characterization** — Click 'Compute' to analyze",
            sizing_mode="stretch_width",
        )
        self.bounds_pane = pn.pane.Markdown(
            "**Theoretical Bounds** — Click 'Compute' to analyze",
            sizing_mode="stretch_width",
        )
        self.validation_pane = pn.pane.Markdown(
            "**Validation** — Click 'Compute' to analyze",
            sizing_mode="stretch_width",
        )
        self.sensitivity_pane = pn.pane.Markdown(
            "**Sensitivity Analysis** — Click 'Compute' to analyze",
            sizing_mode="stretch_width",
        )
        self.optimization_pane = pn.pane.Markdown(
            "**Optimization** — Click 'Compute' to analyze",
            sizing_mode="stretch_width",
        )

        # Optimization widgets (for parameter optimization UI)
        self.strategy_selector = pn.widgets.Select(
            name="Optimization Strategy",
            options={
                "Balanced (eliminate bottlenecks)": "balanced",
                "Empirical (fit to trajectory)": "empirical",
                "Conservative (stable, slower)": "conservative",
                "Aggressive (fast, risky)": "aggressive",
            },
            value="balanced",
            sizing_mode="stretch_width",
        )

        self.suggest_button = pn.widgets.Button(
            name="Suggest Optimal Parameters",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        self.suggest_button.on_click(self._on_suggest_click)

        self.apply_button = pn.widgets.Button(
            name="Apply to Config",
            button_type="warning",
            disabled=True,  # Enable after suggestion
            sizing_mode="stretch_width",
        )
        self.apply_button.on_click(self._on_apply_click)

        self.optimization_result_pane = pn.pane.Markdown(
            "Click 'Suggest Optimal Parameters' to compute optimized settings.",
            sizing_mode="stretch_width",
        )

        # Store suggested parameters (populated by optimization)
        self.suggested_params = None
        self.optimization_diagnostics = None

    def update_benchmark(self, potential: object):
        """Update benchmark/potential function."""
        self.potential = potential
        self.diagnostics = None  # Reset diagnostics

    def set_history(self, history: RunHistory):
        """Update history data for empirical analysis.

        Args:
            history: RunHistory from simulation
        """
        self.history = history
        self.diagnostics = None  # Reset diagnostics to recompute with new data

    def _on_compute_click(self, event):
        """Handle compute button click."""
        if self.computing:
            return

        self.computing = True
        self.status_pane.object = "**Status:** Computing theoretical bounds..."
        self.compute_button.disabled = True

        try:
            self.compute_diagnostics()
            self._update_displays()
            self.status_pane.object = "**Status:** ✅ Theoretical bounds computed successfully."
        except Exception as e:
            self.status_pane.object = f"**Error:** Failed to compute bounds: {e}"
            import traceback

            traceback.print_exc()
        finally:
            self.computing = False
            self.compute_button.disabled = False

    def compute_diagnostics(self):
        """Compute all theoretical convergence bounds and diagnostics."""
        # Extract parameters from GasConfig
        # Note: sigma_v (Langevin noise) = sqrt(2 / (gamma * beta))
        import numpy as np

        from fragile.fractalai.gas_parameters import (
            estimate_landscape_from_history,
            estimate_rates_from_trajectory,
            extract_trajectory_data_from_history,
        )

        gamma_val = float(self.gas_config.gamma)
        beta_val = float(self.gas_config.beta)
        sigma_v_val = np.sqrt(2.0 / (gamma_val * beta_val))

        params = {
            "gamma": gamma_val,
            "beta": beta_val,
            "lambda_alg": float(self.gas_config.lambda_alg),
            "sigma_v": sigma_v_val,  # Computed from gamma and beta
            "tau": float(self.gas_config.delta_t),  # Time step
            "N": int(self.gas_config.N),
            "d": int(self.gas_config.dims),
            "epsilon_F": float(self.gas_config.epsilon_F),
            "use_fitness_force": bool(self.gas_config.use_fitness_force),
            "use_potential_force": bool(self.gas_config.use_potential_force),
        }

        # 1. Environment Characterization
        # Try to estimate from history if available, otherwise use defaults
        if self.history is not None:
            try:
                # Estimate landscape from empirical data
                use_bounds_analysis = not bool(self.gas_config.pbc)
                landscape_empirical = estimate_landscape_from_history(
                    self.history, use_bounds_analysis=use_bounds_analysis
                )
                lambda_min = landscape_empirical.lambda_min
                lambda_max = landscape_empirical.lambda_max
                delta_f_boundary = landscape_empirical.Delta_f_boundary
                has_empirical_landscape = True
            except Exception as e:
                # Fall back to defaults if estimation fails
                print(f"Warning: Landscape estimation failed: {e}")
                lambda_min = 1.0
                lambda_max = 10.0
                delta_f_boundary = 0.5
                has_empirical_landscape = False
        else:
            # No history: use default values
            lambda_min = 1.0
            lambda_max = 10.0
            delta_f_boundary = 0.5
            has_empirical_landscape = False

        # 2. Component Convergence Rates (Euclidean Gas)
        kappa_x = cb.kappa_x(params["lambda_alg"], params["tau"])
        kappa_v = cb.kappa_v(params["gamma"], params["tau"])
        kappa_W = cb.kappa_W(params["gamma"], lambda_min, c_hypo=0.1)
        kappa_b = cb.kappa_b(params["lambda_alg"], delta_f_boundary)

        # Total convergence rate
        kappa_total = cb.kappa_total(kappa_x, kappa_v, kappa_W, kappa_b)

        # Bottleneck analysis
        bottlenecks = cb.convergence_timescale_ratio(kappa_x, kappa_v, kappa_W, kappa_b)

        # 3. Mixing Time
        epsilon = 0.01  # Target accuracy (1%)
        V_init = 10.0  # Typical initial Lyapunov value
        C_total = 1.0  # Equilibrium constant
        T_mix = cb.T_mix(epsilon, kappa_total, V_init, C_total)

        # 4. Equilibrium Variances
        var_x_eq = cb.equilibrium_variance_x(
            params["sigma_v"], params["tau"], params["gamma"], params["lambda_alg"]
        )
        var_v_eq = cb.equilibrium_variance_v(params["d"], params["sigma_v"], params["gamma"])

        # 5. Adaptive Gas Analysis (comprehensive diagnostics)
        from fragile.fractalai.experiments.adaptive_gas import (
            create_adaptive_gas_diagnostics,
            print_adaptive_gas_report,
        )

        # Prepare parameters for adaptive gas validation
        adaptive_params = {
            "epsilon_Sigma": params.get("epsilon_Sigma", 0.1),
            "epsilon_F": params.get("epsilon_F", 0.0),
            "nu": params.get("nu", 0.0),
            "use_anisotropic_diffusion": params.get("use_anisotropic_diffusion", False),
            "use_fitness_force": params.get("use_fitness_force", False),
            "use_viscous_coupling": params.get("use_viscous_coupling", False),
            "viscous_length_scale": params.get("viscous_length_scale", 1.0),
            "H_max": lambda_max,  # Use estimated landscape curvature
            "F_adapt_max": 10.0,  # Estimated max adaptive force
            "rho": 1.0,  # Density parameter
            "gamma": params["gamma"],
        }

        # Create comprehensive adaptive gas diagnostics
        adaptive_diagnostics = create_adaptive_gas_diagnostics(
            history=self.history,
            params=adaptive_params,
            estimate_H_max=False,  # Use lambda_max as H_max estimate
        )

        # Print detailed report to console
        print_adaptive_gas_report(adaptive_diagnostics)

        # Extract key values for backward compatibility
        if adaptive_diagnostics.ellipticity is not None:
            is_elliptic = adaptive_diagnostics.ellipticity.is_valid
            c_min_val = adaptive_diagnostics.ellipticity.c_min_value
            c_max_val = adaptive_diagnostics.ellipticity.c_max_value
            if c_max_val is not None:
                kappa_geom = cb.condition_number_geometry(c_min_val, c_max_val)
            else:
                kappa_geom = None
        else:
            is_elliptic = None
            c_min_val = None
            c_max_val = None
            kappa_geom = None

        if adaptive_diagnostics.hypocoercivity is not None:
            epsilon_F_star = adaptive_diagnostics.hypocoercivity.epsilon_F_star_value
            is_hypocoercive = adaptive_diagnostics.hypocoercivity.is_valid
        else:
            epsilon_F_star = None
            is_hypocoercive = None

        # LSI constant (only if geometric gas enabled)
        if c_min_val is not None and c_max_val is not None:
            kappa_conf = lambda_min
            C_LSI = cb.C_LSI_geometric(
                1.0, c_min_val, c_max_val, params["gamma"], kappa_conf, kappa_W
            )
        else:
            C_LSI = None

        # 6. Sensitivity Analysis
        param_dict = {
            "gamma": params["gamma"],
            "lambda_alg": params["lambda_alg"],
            "sigma_v": params["sigma_v"],
            "tau": params["tau"],
            "lambda_min": lambda_min,
            "delta_f_boundary": delta_f_boundary,
        }
        M_kappa = cb.rate_sensitivity_matrix(param_dict)
        kappa_params = cb.condition_number_parameters(M_kappa)
        modes = cb.principal_coupling_modes(M_kappa, k=3)

        # 7. Optimal Parameters
        V_target = 1.0
        optimal_params = cb.balanced_parameters_closed_form(
            lambda_min, lambda_max, params["d"], V_target
        )

        # 8. Pareto Frontier
        kappa_range = (0.1, 1.0)
        C_range = (0.5, 5.0)
        pareto = cb.pareto_frontier_rate_variance(kappa_range, C_range, n_points=15)

        # 9. Empirical Rate Estimation (if history available)
        if self.history is not None:
            try:
                # Extract trajectory data with improved Wasserstein proxy
                trajectory_data = extract_trajectory_data_from_history(
                    self.history,
                    stage="final",
                    use_improved_wasserstein=True,  # Use better Wasserstein proxy
                )

                # Fit empirical convergence rates with verbose diagnostics
                print("\n" + "=" * 60)
                print("EMPIRICAL CONVERGENCE RATE ESTIMATION")
                print("=" * 60)
                empirical_rates = estimate_rates_from_trajectory(
                    trajectory_data,
                    tau=params["tau"],
                    verbose=True,  # Enable diagnostic output
                )
                print("=" * 60 + "\n")

                # Extract fit diagnostics
                fit_diagnostics = trajectory_data.get("_diagnostics", {}).get("fits", {})

                # Compute comparison metrics with fit quality
                rate_comparison = {
                    "kappa_x": {
                        "theoretical": kappa_x,
                        "empirical": empirical_rates.kappa_x,
                        "error": abs(kappa_x - empirical_rates.kappa_x) / (kappa_x + 1e-10),
                        "fit_r2": fit_diagnostics.get("kappa_x", None).r_squared
                        if fit_diagnostics.get("kappa_x")
                        else 0.0,
                    },
                    "kappa_v": {
                        "theoretical": kappa_v,
                        "empirical": empirical_rates.kappa_v,
                        "error": abs(kappa_v - empirical_rates.kappa_v) / (kappa_v + 1e-10),
                        "fit_r2": fit_diagnostics.get("kappa_v", None).r_squared
                        if fit_diagnostics.get("kappa_v")
                        else 0.0,
                    },
                    "kappa_W": {
                        "theoretical": kappa_W,
                        "empirical": empirical_rates.kappa_W,
                        "error": abs(kappa_W - empirical_rates.kappa_W) / (kappa_W + 1e-10),
                        "fit_r2": fit_diagnostics.get("kappa_W", None).r_squared
                        if fit_diagnostics.get("kappa_W")
                        else 0.0,
                    },
                    "kappa_b": {
                        "theoretical": kappa_b,
                        "empirical": empirical_rates.kappa_b,
                        "error": abs(kappa_b - empirical_rates.kappa_b) / (kappa_b + 1e-10),
                        "fit_r2": fit_diagnostics.get("kappa_b", None).r_squared
                        if fit_diagnostics.get("kappa_b")
                        else 0.0,
                    },
                    "kappa_total": {
                        "theoretical": kappa_total,
                        "empirical": empirical_rates.kappa_total,
                        "error": abs(kappa_total - empirical_rates.kappa_total)
                        / (kappa_total + 1e-10),
                        "fit_r2": 0.0,  # Total doesn't have its own fit
                    },
                }

                has_empirical_rates = True
            except Exception as e:
                print(f"Warning: Empirical rate estimation failed: {e}")
                import traceback

                traceback.print_exc()
                empirical_rates = None
                rate_comparison = None
                has_empirical_rates = False
        else:
            empirical_rates = None
            rate_comparison = None
            has_empirical_rates = False

        # Store all diagnostics
        self.diagnostics = {
            "params": params,
            "environment": {
                "lambda_min": lambda_min,
                "lambda_max": lambda_max,
                "delta_f_boundary": delta_f_boundary,
                "has_empirical": has_empirical_landscape,
            },
            "rates": {
                "kappa_x": kappa_x,
                "kappa_v": kappa_v,
                "kappa_W": kappa_W,
                "kappa_b": kappa_b,
                "kappa_total": kappa_total,
            },
            "bottlenecks": bottlenecks,
            "mixing": {
                "T_mix": T_mix,
                "epsilon": epsilon,
                "V_init": V_init,
                "C_total": C_total,
            },
            "equilibrium": {
                "var_x": var_x_eq,
                "var_v": var_v_eq,
            },
            "geometric": {
                "is_elliptic": is_elliptic,
                "c_min": c_min_val,
                "c_max": c_max_val,
                "kappa_geom": kappa_geom,
                "epsilon_F_star": epsilon_F_star,
                "is_hypocoercive": is_hypocoercive,
                "C_LSI": C_LSI,
            },
            "adaptive_gas": adaptive_diagnostics,  # Full comprehensive diagnostics
            "sensitivity": {
                "M_kappa": M_kappa,
                "kappa_params": kappa_params,
                "modes": modes,
            },
            "optimization": {
                "optimal_params": optimal_params,
                "pareto": pareto,
            },
            "empirical": {
                "has_data": has_empirical_rates,
                "rates": empirical_rates,
                "comparison": rate_comparison,
            },
        }

    def _update_displays(self):
        """Update all display panes with computed diagnostics."""
        if self.diagnostics is None:
            return

        # 1. Environment Characterization
        self.environment_pane.object = self._format_environment()

        # 2. Theoretical Bounds
        self.bounds_pane.object = self._format_bounds()

        # 3. Validation
        self.validation_pane.object = self._format_validation()

        # 4. Sensitivity Analysis
        self.sensitivity_pane.object = self._format_sensitivity()

        # 5. Optimization
        self.optimization_pane.object = self._format_optimization()

    def _format_environment(self) -> str:
        """Format environment characterization section."""
        env = self.diagnostics["environment"]
        params = self.diagnostics["params"]

        md = "## Environment Characterization\n\n"
        md += "### Simulation Parameters\n\n"
        md += f"- **N** (walkers): {params['N']}\n"
        md += f"- **d** (dimensions): {params['d']}\n"
        md += f"- **γ** (friction): {params['gamma']:.4f}\n"
        md += f"- **β** (inverse temp): {params['beta']:.4f}\n"
        md += f"- **λ_alg** (cloning rate): {params['lambda_alg']:.4f}\n"
        md += f"- **σ_v** (Langevin noise): {params['sigma_v']:.4f} (computed from γ, β)\n"
        md += f"- **τ** (time step): {params['tau']:.4f}\n"
        md += f"- **ε_F** (fitness force): {params['epsilon_F']:.4f}\n"
        md += f"- **Use fitness force**: {params['use_fitness_force']}\n"
        md += f"- **Use potential force**: {params['use_potential_force']}\n\n"

        md += "### Landscape Properties (Estimated)\n\n"
        md += f"- **λ_min** (min curvature): {env['lambda_min']:.4f}\n"
        md += f"- **λ_max** (max curvature): {env['lambda_max']:.4f}\n"
        md += f"- **Δf_boundary** (fitness drop): {env['delta_f_boundary']:.4f}\n"

        return md

    def _format_bounds(self) -> str:
        """Format theoretical bounds section."""
        rates = self.diagnostics["rates"]
        bottlenecks = self.diagnostics["bottlenecks"]
        mixing = self.diagnostics["mixing"]
        eq = self.diagnostics["equilibrium"]

        md = "## Theoretical Convergence Bounds\n\n"
        md += "### Component Contraction Rates\n\n"
        md += f"- **κ_x** (position): {rates['kappa_x']:.6f}\n"
        md += f"- **κ_v** (velocity): {rates['kappa_v']:.6f}\n"
        md += f"- **κ_W** (Wasserstein): {rates['kappa_W']:.6f}\n"
        md += f"- **κ_b** (boundary): {rates['kappa_b']:.6f}\n\n"

        md += "### Total Convergence Rate\n\n"
        md += f"- **κ_total**: {rates['kappa_total']:.6f}\n\n"

        md += "### Bottleneck Analysis\n\n"
        md += f"- **Slowest component**: {bottlenecks['bottleneck']}\n"
        md += "- **Timescale ratios** (relative to fastest):\n"
        for comp, ratio in bottlenecks.items():
            if comp != "bottleneck":
                md += f"  - {comp}: {ratio:.2f}x slower\n"
        md += "\n"

        md += "### Mixing Time\n\n"
        epsilon_pct = mixing["epsilon"] * 100
        md += f"- **T_mix** (to {epsilon_pct:.1f}% accuracy): {mixing['T_mix']:.1f} steps\n"
        md += f"- **Initial V**: {mixing['V_init']:.2f}\n"
        md += f"- **Equilibrium C**: {mixing['C_total']:.2f}\n\n"

        md += "### Equilibrium Variances\n\n"
        md += f"- **Var[x]_eq**: {eq['var_x']:.6f}\n"
        md += f"- **Var[v]_eq**: {eq['var_v']:.6f}\n\n"

        # Add empirical comparison if available
        empirical = self.diagnostics.get("empirical", {})
        if empirical.get("has_data", False) and empirical.get("comparison") is not None:
            md += "---\n\n"
            md += "## Empirical vs. Theoretical Comparison\n\n"
            md += "Convergence rates measured from simulation trajectory:\n\n"
            md += "| Component | Theoretical | Empirical | Fit Quality (R²) | Error |\n"
            md += "|-----------|-------------|-----------|------------------|-------|\n"

            comparison = empirical["comparison"]
            for comp_name, comp_label in [
                ("kappa_x", "κ_x (position)"),
                ("kappa_v", "κ_v (velocity)"),
                ("kappa_W", "κ_W (Wasserstein)"),
                ("kappa_b", "κ_b (boundary)"),
                ("kappa_total", "κ_total (total)"),
            ]:
                if comp_name in comparison:
                    comp_data = comparison[comp_name]
                    theoretical = comp_data["theoretical"]
                    empirical_val = comp_data["empirical"]
                    error_pct = comp_data["error"] * 100
                    fit_r2 = comp_data.get("fit_r2", 0.0)

                    # Color code based on fit quality and error
                    # Fit quality indicators
                    if comp_name != "kappa_total":
                        if fit_r2 > 0.9:
                            fit_status = "✅"
                        elif fit_r2 > 0.7:
                            fit_status = "⚠️"
                        else:
                            fit_status = "❌"
                        fit_str = f"{fit_status} {fit_r2:.3f}"
                    else:
                        fit_str = "—"

                    # Error indicators (only trust if fit quality is good)
                    if fit_r2 < 0.5 and comp_name != "kappa_total":
                        # Poor fit - don't trust error metric
                        error_str = f"❓ {error_pct:.1f}% (unreliable)"
                    else:
                        if error_pct < 10:
                            status = "✅"
                        elif error_pct < 30:
                            status = "⚠️"
                        else:
                            status = "❌"
                        error_str = f"{status} {error_pct:.1f}%"

                    md += f"| {comp_label} | {theoretical:.6f} | {empirical_val:.6f} | {fit_str} | {error_str} |\n"

            md += "\n**Interpretation:**\n"
            md += "- **Fit Quality (R²)**: Measures how well exponential fit matches trajectory\n"
            md += "  - ✅ Excellent (R² > 0.9) — trust empirical rate\n"
            md += "  - ⚠️ Moderate (0.7 < R² < 0.9) — interpret cautiously\n"
            md += "  - ❌ Poor (R² < 0.7) — empirical rate unreliable\n"
            md += "- **Error**: Comparison between theoretical and empirical\n"
            md += "  - ✅ Good match (<10%)\n"
            md += "  - ⚠️ Moderate discrepancy (10-30%)\n"
            md += "  - ❌ Significant mismatch (>30%) - model assumptions may be violated\n"
            md += "  - ❓ Unreliable - poor fit quality, ignore error percentage\n\n"
            md += "**Common causes of large errors:**\n"
            md += "- Simulation too short (<100 steps) — equilibrium not reached\n"
            md += "- Theoretical constants (c_fit, c_hypo) don't match actual dynamics\n"
            md += "- Parameter mapping approximations (e.g., λ_clone from α_fit)\n"
            md += "- Wasserstein/boundary proxies are crude approximations\n"

        return md

    def _format_validation(self) -> str:
        """Format validation section with comprehensive adaptive gas diagnostics."""
        adaptive_diag = self.diagnostics.get("adaptive_gas")

        if adaptive_diag is None:
            return "## Parameter Regime Validation\n\n_(No adaptive gas diagnostics available)_\n"

        md = "## Parameter Regime Validation\n\n"

        # Overall summary
        if adaptive_diag.overall_status == "valid":
            md += f"**Status:** {adaptive_diag.summary}\n\n"
        elif adaptive_diag.overall_status == "warning":
            md += f"**Status:** {adaptive_diag.summary}\n\n"
        else:  # invalid
            md += f"**Status:** {adaptive_diag.summary}\n\n"

        if adaptive_diag.is_euclidean_gas:
            md += "### Euclidean Gas Mode\n\n"
            md += "- No adaptive features enabled (pure Euclidean Gas)\n"
            md += "- Using standard isotropic diffusion and Langevin dynamics\n"
            md += "- No additional validation required\n\n"
            return md

        # Ellipticity section
        if adaptive_diag.ellipticity is not None:
            md += "### Uniform Ellipticity\n\n"
            ellip = adaptive_diag.ellipticity

            if ellip.is_valid:
                md += "✅ **CONDITION SATISFIED**\n\n"
            else:
                md += "❌ **CONDITION VIOLATED**\n\n"

            md += f"- **ε_Σ** (regularization): {ellip.epsilon_Sigma:.6f}\n"
            md += f"- **H_max** (Hessian bound): {ellip.H_max:.6f} ({ellip.H_max_source})\n"
            md += f"- **Margin** (ε_Σ - H_max): {ellip.margin:.6f}\n"
            md += f"- **c_min**: {ellip.c_min_value:.6f}\n"

            if ellip.c_max_value is not None:
                md += f"- **c_max**: {ellip.c_max_value:.6f}\n"
                md += f"- **Condition number**: {ellip.condition_number:.2f}\n"
            else:
                md += "- **c_max**: undefined (ellipticity violated)\n"

            if not ellip.is_valid:
                md += f"\n⚠️ **Recommended ε_Σ**: {ellip.recommended_epsilon_Sigma:.6f}\n"

            md += "\n"

        # Hypocoercivity section
        if adaptive_diag.hypocoercivity is not None:
            md += "### Hypocoercivity Regime\n\n"
            hypo = adaptive_diag.hypocoercivity

            if hypo.is_valid:
                md += "✅ **REGIME SATISFIED**\n\n"
            else:
                md += "⚠️ **REGIME NOT SATISFIED**\n\n"

            md += f"- **ε_F** (fitness force): {hypo.epsilon_F:.6f}\n"
            md += f"- **ε_F*** (threshold): {hypo.epsilon_F_star_value:.6f}\n"
            md += f"- **Margin** (ε_F* - ε_F): {hypo.margin:.6f}\n"
            md += f"- **ν** (viscous coupling): {hypo.nu:.6f}\n"
            md += f"- **Fitness force enabled**: {hypo.use_fitness_force}\n"
            md += f"- **Viscous coupling enabled**: {hypo.use_viscous_coupling}\n\n"

        # Viscous coupling section
        if (
            adaptive_diag.viscous_coupling is not None
            and adaptive_diag.viscous_coupling.is_enabled
        ):
            md += "### Viscous Coupling\n\n"
            visc = adaptive_diag.viscous_coupling

            md += f"- **ν** (strength): {visc.nu:.6f}\n"
            md += f"- **l** (length scale): {visc.length_scale:.6f}\n"
            md += f"- **Mean force magnitude**: {visc.mean_force_magnitude:.6f}\n"
            md += f"- **Max force magnitude**: {visc.max_force_magnitude:.6f}\n\n"

        # Diffusion metrics section
        if adaptive_diag.diffusion_metrics is not None:
            md += "### Diffusion Tensor Metrics\n\n"
            diff = adaptive_diag.diffusion_metrics

            md += f"- **Mean λ_min**: {diff.mean_min_eigenvalue:.6f}\n"
            md += f"- **Mean λ_max**: {diff.mean_max_eigenvalue:.6f}\n"
            md += f"- **Mean condition number**: {diff.mean_condition_number:.2f}\n"
            md += f"- **Worst condition number**: {diff.worst_condition_number:.2f}\n"
            md += f"- **Anisotropy ratio**: {diff.anisotropy_ratio:.2f}\n\n"

        # LSI constant (backward compatibility)
        geom = self.diagnostics["geometric"]
        if geom["C_LSI"] is not None:
            md += "### LSI Convergence\n\n"
            md += f"- **C_LSI** (N-uniform constant): {geom['C_LSI']:.4f}\n"
            md += f"- **KL convergence rate**: {1.0 / geom['C_LSI']:.6f}\n\n"

        # Add detailed guidance link
        md += "---\n\n"
        md += "_For detailed guidance on violations, see terminal output or call `print_adaptive_gas_report()`_\n"

        return md

    def _format_sensitivity(self) -> str:
        """Format sensitivity analysis section."""
        sens = self.diagnostics["sensitivity"]

        md = "## Sensitivity Analysis\n\n"
        md += "### Rate Sensitivity Matrix\n\n"
        n_rates = sens["M_kappa"].shape[0]
        n_params = sens["M_kappa"].shape[1]
        md += f"- **Shape**: {n_rates} rates × {n_params} parameters\n"
        md += f"- **Condition number κ_params**: {sens['kappa_params']:.2f}\n"
        if sens["kappa_params"] < 10:
            md += "- ✅ **Well-conditioned** (robust to parameter errors)\n"
        elif sens["kappa_params"] < 100:
            md += "- ⚠️ **Moderately conditioned** (some sensitivity)\n"
        else:
            md += "- ❌ **Ill-conditioned** (high sensitivity to parameter errors)\n"
        md += "\n"

        md += "### Principal Coupling Modes\n\n"
        md += "Top 3 singular values (strength of parameter coupling):\n"
        for i, sv in enumerate(sens["modes"]["singular_values"], 1):
            md += f"{i}. {sv:.4f}\n"

        return md

    def _format_optimization(self) -> str:
        """Format optimization section."""
        opt = self.diagnostics["optimization"]

        md = "## Optimal Parameter Selection\n\n"
        md += "### Balanced Parameters (Closed Form)\n\n"
        md += "Parameters that eliminate convergence bottlenecks:\n\n"
        for param_name, value in opt["optimal_params"].items():
            md += f"- **{param_name}**: {value:.6f}\n"
        md += "\n"

        md += "### Rate-Variance Pareto Frontier\n\n"
        md += "Trade-off between convergence rate and exploration variance:\n\n"
        md += "| κ_total | Var_eq |\n"
        md += "|---------|--------|\n"
        for kappa, var in opt["pareto"][:5]:  # Show first 5 points
            md += f"| {kappa:.4f} | {var:.4f} |\n"
        md += "\n"
        md += "**Note**: Higher rate → faster convergence, Lower variance → focused exploration\n"

        return md

    def _extract_gas_params_from_config(self):
        """Extract GasParams from current GasConfig for optimization."""
        from fragile.fractalai.gas_parameters import gas_params_from_config

        return gas_params_from_config(self.gas_config)

    def _get_default_landscape(self):
        """Get default landscape parameters (fallback when no history available)."""
        from fragile.fractalai.gas_parameters import LandscapeParams

        return LandscapeParams(
            lambda_min=1.0,
            lambda_max=10.0,
            d=self.gas_config.dims,
            f_typical=1.0,
            Delta_f_boundary=0.5,
        )

    def _on_suggest_click(self, event):
        """Handle 'Suggest Optimal Parameters' button click."""
        from fragile.fractalai.gas_parameters import (
            estimate_landscape_from_history,
            extract_trajectory_data_from_history,
            optimize_parameters_multi_strategy,
        )

        if self.diagnostics is None:
            self.status_pane.object = "⚠️ **Status:** Compute diagnostics first"
            return

        strategy = self.strategy_selector.value

        # Build current GasParams from gas_config
        current_params = self._extract_gas_params_from_config()

        # Use empirical landscape if available, else defaults
        if self.history is not None:
            try:
                use_bounds_analysis = not bool(self.gas_config.pbc)
                landscape = estimate_landscape_from_history(
                    self.history, use_bounds_analysis=use_bounds_analysis
                )
            except Exception as e:
                print(f"Warning: Landscape estimation failed, using defaults: {e}")
                landscape = self._get_default_landscape()
        else:
            landscape = self._get_default_landscape()

        # Extract trajectory data if needed for empirical strategy
        trajectory_data = None
        if strategy == "empirical":
            if self.history is None:
                self.status_pane.object = "⚠️ **Status:** Empirical strategy requires simulation data. Run a simulation first."
                return
            try:
                trajectory_data = extract_trajectory_data_from_history(self.history)
            except Exception as e:
                self.status_pane.object = f"❌ **Status:** Failed to extract trajectory data: {e}"
                return

        # Optimize parameters
        try:
            self.status_pane.object = (
                f"⏳ **Status:** Computing optimal parameters ({strategy} strategy)..."
            )

            optimal_params, diagnostics = optimize_parameters_multi_strategy(
                strategy=strategy,
                landscape=landscape,
                current_params=current_params,
                trajectory_data=trajectory_data,
                V_target=0.1,
            )

            # Store for apply action
            self.suggested_params = optimal_params
            self.optimization_diagnostics = diagnostics

            # Format and display results
            self._format_optimization_suggestion()

            # Enable apply button
            self.apply_button.disabled = False

            self.status_pane.object = f"✅ **Status:** Optimal parameters computed successfully using {strategy} strategy"

        except Exception as e:
            self.status_pane.object = f"❌ **Status:** Optimization failed: {e}"
            import traceback

            traceback.print_exc()

    def _on_apply_click(self, event):
        """Handle 'Apply to Config' button click."""
        from fragile.fractalai.gas_parameters import apply_gas_params_to_config

        if not hasattr(self, "suggested_params") or self.suggested_params is None:
            return

        try:
            # Apply optimized parameters to GasConfig
            apply_gas_params_to_config(
                self.suggested_params,
                self.gas_config,
                preserve_adaptive=True,  # Keep current adaptive settings
            )

            self.status_pane.object = "✅ **Status:** Parameters applied to configuration. Click 'Run Simulation' in the sidebar to test."
            self.apply_button.disabled = True  # Prevent double-apply

        except Exception as e:
            self.status_pane.object = f"❌ **Status:** Apply failed: {e}"
            import traceback

            traceback.print_exc()

    def _format_optimization_suggestion(self):
        """Format suggested parameters comparison table."""
        if self.suggested_params is None or self.optimization_diagnostics is None:
            self.optimization_result_pane.object = "No optimization results available."
            return

        current_params = self._extract_gas_params_from_config()
        suggested = self.suggested_params
        diag = self.optimization_diagnostics

        md = "## Suggested Optimal Parameters\n\n"
        md += f"**Strategy:** {diag['strategy']}\n\n"

        # Comparison table
        md += "### Current vs. Suggested\n\n"
        md += "| Parameter | Current | Suggested | Change |\n"
        md += "|-----------|---------|-----------|--------|\n"

        params_to_compare = [
            ("tau", "τ (time step)"),
            ("gamma", "γ (friction)"),
            ("sigma_v", "σ_v (noise)"),
            ("lambda_clone", "λ (cloning rate)"),
            ("N", "N (walkers)"),
            ("sigma_x", "σ_x (jitter)"),
            ("lambda_alg", "λ_alg (velocity weight)"),
            ("alpha_rest", "α (restitution)"),
        ]

        for param_name, param_label in params_to_compare:
            current_val = getattr(current_params, param_name)
            suggested_val = getattr(suggested, param_name)

            if isinstance(current_val, float):
                if current_val > 0:
                    change_pct = ((suggested_val - current_val) / current_val) * 100
                    md += f"| {param_label} | {current_val:.6f} | {suggested_val:.6f} | {change_pct:+.1f}% |\n"
                else:
                    md += f"| {param_label} | {current_val:.6f} | {suggested_val:.6f} | — |\n"
            else:  # Integer (N)
                change_abs = suggested_val - current_val
                md += f"| {param_label} | {current_val} | {suggested_val} | {change_abs:+d} |\n"

        # Expected improvement
        md += "\n### Expected Improvement\n\n"
        improvement_pct = diag["improvement_ratio"] * 100
        if improvement_pct > 0:
            md += f"- **Convergence rate increase**: +{improvement_pct:.1f}%\n"
        else:
            md += f"- **Convergence rate change**: {improvement_pct:.1f}%\n"

        md += f"- **κ_total**: {diag['kappa_before']:.6f} → {diag['kappa_after']:.6f}\n"
        md += f"- **Mixing time**: {diag['expected_T_mix']:.1f} time units ({diag['expected_T_mix_steps']:.0f} steps)\n"
        md += f"- **Bottleneck**: {diag['bottleneck_before']} → {diag['bottleneck_after']}\n"

        self.optimization_result_pane.object = md

    def panel(self) -> pn.Column:
        """Create Panel layout for convergence bounds diagnostics.

        Returns:
            Panel Column with compute button, status, and result displays
        """
        return pn.Column(
            pn.pane.Markdown("# Convergence Bounds Diagnostics"),
            pn.pane.Markdown(
                "Compute theoretical convergence bounds from the mathematical framework. "
                "This analysis characterizes the environment, validates parameter regimes, "
                "and compares theoretical predictions with simulation parameters."
            ),
            self.compute_button,
            self.status_pane,
            pn.pane.Markdown("---"),
            pn.Accordion(
                ("Environment Characterization", self.environment_pane),
                ("Theoretical Bounds", self.bounds_pane),
                ("Validation", self.validation_pane),
                ("Sensitivity Analysis", self.sensitivity_pane),
                (
                    "Optimization",
                    pn.Column(
                        self.optimization_pane,  # Static analysis (Pareto frontier, etc.)
                        pn.layout.Divider(),
                        pn.pane.Markdown("### Interactive Parameter Optimization"),
                        self.strategy_selector,
                        self.suggest_button,
                        pn.layout.Divider(),
                        self.optimization_result_pane,
                        pn.layout.Divider(),
                        self.apply_button,
                        sizing_mode="stretch_width",
                    ),
                ),
                active=[0, 1],  # Open first two sections by default
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )
