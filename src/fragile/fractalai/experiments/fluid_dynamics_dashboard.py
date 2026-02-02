"""Fluid Dynamics Dashboard: Navier-Stokes Validation with Viscous-Coupled Swarms.

This module demonstrates that the viscous-coupled Euclidean Gas can simulate
fluid behavior consistent with the Navier-Stokes equations using three classic
computational fluid dynamics benchmarks:

1. Taylor-Green Vortex: Decaying 2D vortex with analytical solution
2. Lid-Driven Cavity: Flow in square cavity with moving top wall
3. Kelvin-Helmholtz Instability: Shear layer instability with vortex roll-up

The dashboard provides interactive visualization of velocity fields, vorticity,
density, and quantitative validation against theoretical predictions.

Mathematical Foundation:
-----------------------
The viscous coupling F_viscous = ν Σ_j [K(||xi-xj||)/deg(i)] (vj-vi) creates
fluid-like behavior. In the mean-field limit (N→∞), this converges to a
McKean-Vlasov PDE with velocity-dependent drift, analogous to the momentum
equation in Navier-Stokes.

Reference: docs/source/2_geometric_gas/11_geometric_gas.md § 2.1.3
"""

from __future__ import annotations

import holoviews as hv
from holoviews import dim
import numpy as np
import panel as pn
import param
import torch

from fragile.fractalai.core.benchmarks import (
    KelvinHelmholtzInstability,
    LidDrivenCavity,
    TaylorGreenVortex,
)
from fragile.fractalai.core.cloning import CloneOperator
from fragile.fractalai.core.companion_selection import CompanionSelection
from fragile.fractalai.core.euclidean_gas import EuclideanGas
from fragile.fractalai.core.fitness import FitnessOperator
from fragile.fractalai.core.kinetic_operator import KineticOperator
from fragile.fractalai.experiments.fluid_utils import (
    FLUID_CONFIGS,
    FluidFieldComputer,
)
from fragile.fractalai.experiments.gas_config_dashboard import GasConfig


__all__ = [
    "FluidDynamicsExplorer",
    "create_fluid_dashboard",
]

# Main Fluid Dynamics Explorer Dashboard
# ============================================================================


class FluidDynamicsExplorer(param.Parameterized):
    """Interactive dashboard for fluid dynamics validation with viscous-coupled swarms.

    This dashboard demonstrates that the Euclidean Gas with viscous coupling
    can simulate incompressible fluid behavior consistent with Navier-Stokes
    equations. Three classic CFD benchmarks are provided with quantitative
    validation metrics.

    Example:
        >>> explorer = FluidDynamicsExplorer()
        >>> dashboard = explorer.panel()
        >>> dashboard.show()  # Launch interactive dashboard
    """

    # Benchmark selection (fluid-specific)
    benchmark_name = param.ObjectSelector(
        default="Taylor-Green Vortex",
        objects=[
            "Taylor-Green Vortex",
            "Lid-Driven Cavity (Re=100)",
            "Kelvin-Helmholtz Instability",
        ],
        doc="Select fluid dynamics benchmark",
    )

    # Visualization toggles
    show_particles = param.Boolean(default=True, doc="Show individual particles")
    show_velocity_field = param.Boolean(default=True, doc="Show velocity field (quiver)")
    show_streamlines = param.Boolean(default=False, doc="Show streamlines")
    show_vorticity = param.Boolean(default=True, doc="Show vorticity field")
    show_density = param.Boolean(default=False, doc="Show particle density")
    show_energy_plot = param.Boolean(default=True, doc="Show energy evolution")

    # Field computation parameters
    grid_resolution = param.Integer(default=50, bounds=(20, 100), doc="Grid resolution")
    kernel_bandwidth = param.Number(default=0.3, bounds=(0.1, 1.0), doc="Kernel bandwidth")
    quiver_stride = param.Integer(default=3, bounds=(1, 10), doc="Quiver plot stride")

    # Animation control
    frame_index = param.Integer(default=0, bounds=(0, 0), doc="Current frame")
    auto_play = param.Boolean(default=False, doc="Auto-play animation")

    def __init__(self, **params):
        """Initialize FluidDynamicsExplorer.

        Args:
            **params: Override default parameter values (fluid-specific params
                     are set on self, gas params are forwarded to gas_config)
        """
        # Extract gas-related parameters to forward to GasConfig
        gas_param_names = {
            "N",
            "n_steps",
            "enable_cloning",
            "enable_kinetic",
            "pbc",
            "gamma",
            "beta",
            "delta_t",
            "epsilon_F",
            "use_fitness_force",
            "use_potential_force",
            "epsilon_Sigma",
            "use_anisotropic_diffusion",
            "diagonal_diffusion",
            "nu",
            "use_viscous_coupling",
            "viscous_length_scale",
            "V_alg",
            "use_velocity_squashing",
            "sigma_x",
            "alpha_restitution",
            "p_max",
            "epsilon_clone",
            "companion_method",
            "alpha_fit",
            "beta_fit",
            "eta",
            "lambda_alg",
            "sigma_min",
            "A",
        }

        gas_params = {k: v for k, v in params.items() if k in gas_param_names}
        fluid_params = {k: v for k, v in params.items() if k not in gas_param_names}

        # Initialize fluid-specific parameters
        super().__init__(**fluid_params)

        # Create GasConfig instance with forwarded parameters
        # This handles all gas algorithm parameters (N, gamma, beta, etc.)
        self.gas_config = GasConfig(**gas_params)

        # Initialize benchmark
        self.benchmark = None
        self.history = None
        self.field_computer = FluidFieldComputer()

        # Create benchmarks dictionary
        self.benchmarks = {
            "Taylor-Green Vortex": TaylorGreenVortex(),
            "Lid-Driven Cavity (Re=100)": LidDrivenCavity(reynolds_number=100),
            "Kelvin-Helmholtz Instability": KelvinHelmholtzInstability(),
        }

        # Set initial benchmark
        self._update_benchmark()

        # UI components
        self.run_button = pn.widgets.Button(
            name="Run Simulation", button_type="primary", sizing_mode="stretch_width"
        )
        self.run_button.on_click(self._on_run_clicked)

        self.load_params_button = pn.widgets.Button(
            name="Load Recommended Parameters", button_type="default", sizing_mode="stretch_width"
        )
        self.load_params_button.on_click(lambda _: self._load_recommended_params())

        self.status_pane = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.validation_pane = pn.pane.Markdown(
            "## Validation Metrics\n\n*Run simulation to see metrics*", sizing_mode="stretch_width"
        )

        # Animation controls
        self.play_button = pn.widgets.Button(name="▶ Play", button_type="success", width=80)
        self.play_button.on_click(self._toggle_play)

        # Widget overrides for fluid-specific parameters
        self._widget_overrides = {
            "kernel_bandwidth": pn.widgets.FloatSlider(
                name="kernel_bandwidth", start=0.1, end=1.0, step=0.05
            ),
        }

        # Watch for benchmark changes
        self.param.watch(self._on_benchmark_change, "benchmark_name")

    def _update_benchmark(self):
        """Update benchmark and store recommended parameters (without applying them)."""
        self.benchmark = self.benchmarks[self.benchmark_name]

        # Get recommended parameters from FLUID_CONFIGS
        params = FLUID_CONFIGS[self.benchmark_name]

        # Store for manual loading via "Load Recommended Parameters" button
        self._recommended_params = params

    def _load_recommended_params(self):
        """Load recommended parameters from FLUID_CONFIGS into gas_config.

        This is called when user clicks "Load Recommended Parameters" button,
        allowing manual control over when parameters are updated.
        """
        # Get parameters from FLUID_CONFIGS
        params = FLUID_CONFIGS[self.benchmark_name]

        # Update gas_config parameters using param.update()
        # This will automatically trigger UI updates
        self.gas_config.param.update(**params)

        self.status_pane.object = (
            f"**Parameters loaded** from {self.benchmark_name} recommendations"
        )

    def _on_benchmark_change(self, *_):
        """Handle benchmark selection change."""
        self._update_benchmark()
        self.status_pane.object = (
            f"**Benchmark updated:** {self.benchmark_name}\n\n"
            "*Click 'Load Recommended Parameters' to apply benchmark defaults, "
            "or keep your current settings.*"
        )
        self.validation_pane.object = "## Validation Metrics\n\n*Run simulation to see metrics*"
        self.history = None  # Clear previous results

    def _get_filtered_gas_config_panel(self) -> pn.Accordion:
        """Get gas config panel with benchmark and initialization sections hidden.

        Returns:
            Panel Accordion with filtered sections from GasConfig (no benchmark/init)
        """
        # Directly build parameter panels using gas_config's _build_param_panel method
        # This avoids having to parse and filter the full panel structure

        # Swarm parameters
        swarm_params = pn.Param(
            self.gas_config.param,
            parameters=["N", "n_steps", "enable_cloning", "enable_kinetic", "pbc"],
            widgets=self.gas_config._widget_overrides,
            show_name=False,
            sizing_mode="stretch_width",
        )

        # Langevin dynamics parameters
        langevin_params = pn.Param(
            self.gas_config.param,
            parameters=[
                "gamma",
                "beta",
                "delta_t",
                "nu",
                "use_viscous_coupling",
                "viscous_length_scale",
                "epsilon_F",
                "use_fitness_force",
                "use_potential_force",
                "epsilon_Sigma",
                "use_anisotropic_diffusion",
                "diagonal_diffusion",
                "V_alg",
                "use_velocity_squashing",
            ],
            widgets=self.gas_config._widget_overrides,
            show_name=False,
            sizing_mode="stretch_width",
        )

        # Cloning parameters
        cloning_params = pn.Param(
            self.gas_config.param,
            parameters=["sigma_x", "alpha_restitution", "p_max", "epsilon_clone"],
            widgets=self.gas_config._widget_overrides,
            show_name=False,
            sizing_mode="stretch_width",
        )

        # Companion selection
        companion_params = pn.Param(
            self.gas_config.param,
            parameters=["companion_method"],
            show_name=False,
            sizing_mode="stretch_width",
        )

        # Fitness operator
        fitness_params = pn.Param(
            self.gas_config.param,
            parameters=["alpha_fit", "beta_fit", "eta", "lambda_alg", "sigma_min", "A"],
            widgets=self.gas_config._widget_overrides,
            show_name=False,
            sizing_mode="stretch_width",
        )

        # Create accordion with filtered sections (no benchmark or init)
        filtered_accordion = pn.Accordion(
            ("Swarm Parameters", swarm_params),
            ("Langevin Dynamics & Viscous Coupling", langevin_params),
            ("Cloning", cloning_params),
            ("Companion Selection", companion_params),
            ("Fitness Operator", fitness_params),
            sizing_mode="stretch_width",
        )
        filtered_accordion.active = [0, 1]  # Open first two sections by default

        return filtered_accordion

    def _on_run_clicked(self, *_):
        """Handle Run Simulation button click."""
        self.status_pane.object = "**Running simulation...**"
        self.run_button.disabled = True

        try:
            self._run_simulation()
            self.status_pane.object = (
                f"**Simulation complete!** {self.history.n_steps} steps, "
                f"{self.history.n_recorded} recorded timesteps"
            )

            # Update frame bounds
            self.param.frame_index.bounds = (0, self.history.n_recorded - 1)
            self.frame_index = 0

        except Exception as e:
            self.status_pane.object = f"**Error:** {e!s}"
            import traceback

            traceback.print_exc()
        finally:
            self.run_button.disabled = False

    def _run_simulation(self):
        """Run EuclideanGas simulation with fluid dynamics parameters."""
        # Get bounds from benchmark
        bounds = self.benchmark.bounds

        # Create kinetic operator with parameters from gas_config
        # Benchmark itself is the potential (OptimBenchmark is callable)
        kinetic_op = KineticOperator(
            gamma=float(self.gas_config.gamma),
            beta=float(self.gas_config.beta),
            delta_t=float(self.gas_config.delta_t),
            integrator="boris-baoab",
            epsilon_F=float(self.gas_config.epsilon_F),
            use_fitness_force=bool(self.gas_config.use_fitness_force),
            use_potential_force=bool(self.gas_config.use_potential_force),
            epsilon_Sigma=float(self.gas_config.epsilon_Sigma),
            use_anisotropic_diffusion=bool(self.gas_config.use_anisotropic_diffusion),
            diagonal_diffusion=bool(self.gas_config.diagonal_diffusion),
            nu=float(self.gas_config.nu),
            use_viscous_coupling=bool(self.gas_config.use_viscous_coupling),
            viscous_length_scale=float(self.gas_config.viscous_length_scale),
            V_alg=float(self.gas_config.V_alg),
            use_velocity_squashing=bool(self.gas_config.use_velocity_squashing),
            potential=self.benchmark,  # Benchmark is the potential
            device=torch.device("cpu"),
            dtype=torch.float32,
            bounds=bounds,  # Pass bounds for periodic distance calculations
            pbc=bool(self.gas_config.pbc),  # Enable periodic boundary conditions
        )

        # Create companion selection and clone operator with parameters from gas_config
        companion_selection = CompanionSelection(method=self.gas_config.companion_method)
        clone_op = CloneOperator(
            sigma_x=float(self.gas_config.sigma_x),
            alpha_restitution=float(self.gas_config.alpha_restitution),
            p_max=float(self.gas_config.p_max),
            epsilon_clone=float(self.gas_config.epsilon_clone),
        )
        # Use fitness operator with parameters from gas_config
        fitness_op = FitnessOperator(
            alpha=float(self.gas_config.alpha_fit),
            beta=float(self.gas_config.beta_fit),
            eta=float(self.gas_config.eta),
            lambda_alg=float(self.gas_config.lambda_alg),
            sigma_min=float(self.gas_config.sigma_min),
            A=float(self.gas_config.A),
        )

        # Create EuclideanGas
        gas = EuclideanGas(
            N=int(self.gas_config.N),
            d=2,
            companion_selection=companion_selection,
            potential=self.benchmark,  # Benchmark is the potential
            kinetic_op=kinetic_op,
            cloning=clone_op,
            fitness_op=fitness_op,
            bounds=bounds,
            device=torch.device("cpu"),
            dtype="float32",
            enable_cloning=bool(self.gas_config.enable_cloning),
            enable_kinetic=bool(self.gas_config.enable_kinetic),
            pbc=bool(self.gas_config.pbc),
        )

        # Get initial conditions from benchmark
        x_init, v_init = self.benchmark.get_initial_conditions(
            self.gas_config.N, torch.device("cpu"), torch.float32
        )

        # Run simulation
        self.history = gas.run(self.gas_config.n_steps, x_init=x_init, v_init=v_init)

    def _render_frame(self, frame_idx: int):
        """Render visualization for given frame index.

        Args:
            frame_idx: Frame index to render

        Returns:
            HoloViews overlay with all requested visualizations
        """
        if self.history is None:
            return hv.Text(0, 0, "Run simulation first").opts(
                width=600, height=600, xlim=(-1, 1), ylim=(-1, 1)
            )

        # Clamp frame index
        frame_idx = max(0, min(frame_idx, self.history.n_recorded - 1))

        # Get data for this frame
        positions = self.history.x_final[frame_idx]  # [N, 2]
        velocities = self.history.v_final[frame_idx]  # [N, 2]

        # Determine bounds from benchmark
        bounds = (self.benchmark.bounds.low[0].item(), self.benchmark.bounds.high[0].item())

        # Start building overlay
        elements = []

        # 1. Velocity field (quiver plot)
        if self.show_velocity_field:
            X, Y, U, V = self.field_computer.compute_velocity_field(
                positions, velocities, self.grid_resolution, self.kernel_bandwidth, bounds
            )

            # Downsample for quiver plot
            stride = self.quiver_stride
            X_sub = X[::stride, ::stride]
            Y_sub = Y[::stride, ::stride]
            U_sub = U[::stride, ::stride]
            V_sub = V[::stride, ::stride]

            # Normalize arrows to unit length for visualization
            mag = np.sqrt(U_sub**2 + V_sub**2) + 1e-10
            U_norm = U_sub / mag
            V_norm = V_sub / mag

            # Create vector field
            vectorfield = hv.VectorField((X_sub, Y_sub, U_norm, V_norm)).opts(
                magnitude="Magnitude",
                color="blue",
                alpha=0.5,
                pivot="mid",
                scale=0.5,
                width=600,
                height=600,
            )
            elements.append(vectorfield)

        # 2. Streamlines
        if self.show_streamlines:
            X, Y, U, V = self.field_computer.compute_velocity_field(
                positions, velocities, self.grid_resolution, self.kernel_bandwidth, bounds
            )
            # TODO: Implement streamlines using hv.Streamlines
            # For now, skip as it requires special data format

        # 3. Vorticity field
        if self.show_vorticity:
            X, Y, U, V = self.field_computer.compute_velocity_field(
                positions, velocities, self.grid_resolution, self.kernel_bandwidth, bounds
            )

            dx = (bounds[1] - bounds[0]) / self.grid_resolution
            dy = (bounds[1] - bounds[0]) / self.grid_resolution
            vorticity = self.field_computer.compute_vorticity(U, V, dx, dy)

            # Create vorticity image
            vorticity_img = hv.Image(
                (X[0, :], Y[:, 0], vorticity), kdims=["x", "y"], vdims=["vorticity"]
            ).opts(
                cmap="RdBu_r",
                alpha=0.6,
                colorbar=True,
                clim=(-2, 2),
                clabel="Vorticity ω",
                width=600,
                height=600,
            )
            elements.append(vorticity_img)

        # 4. Density field
        if self.show_density:
            X, Y, density = self.field_computer.compute_density_field(
                positions, self.grid_resolution, bounds, smoothing=2.0
            )

            density_img = hv.Image(
                (X[0, :], Y[:, 0], density), kdims=["x", "y"], vdims=["density"]
            ).opts(
                cmap="viridis",
                alpha=0.3,
                colorbar=True,
                clabel="Density ρ",
                width=600,
                height=600,
            )
            elements.append(density_img)

        # 5. Particles
        if self.show_particles:
            pos_np = positions.cpu().numpy()
            vel_mag = torch.linalg.vector_norm(velocities, dim=1).cpu().numpy()

            # Combine positions and velocity magnitude [N, 3]
            data = np.column_stack([pos_np, vel_mag])

            points = hv.Points(data, vdims=["velocity"]).opts(
                color=dim("velocity"),
                cmap="plasma",
                size=3,
                alpha=0.8,
                colorbar=True,
                clabel="Velocity Magnitude",
                width=600,
                height=600,
            )
            elements.append(points)

        # Combine all elements
        if elements:
            plot = hv.Overlay(elements).opts(
                xlim=(bounds[0], bounds[1]),
                ylim=(bounds[0], bounds[1]),
                aspect="equal",
                title=f"{self.benchmark_name} - Frame {frame_idx}/{self.history.n_recorded - 1}",
                xlabel="x",
                ylabel="y",
            )
        else:
            plot = hv.Text(0, 0, "Enable at least one visualization option").opts(
                width=600, height=600, xlim=bounds, ylim=bounds
            )

        return plot

    def _render_energy_plot(self):
        """Render energy evolution plot with validation."""
        if self.history is None or not self.show_energy_plot:
            return pn.pane.Markdown("*Run simulation first*")

        # Compute energy at each timestep
        velocities_all = self.history.v_final  # [T, N, 2]
        energies = []
        times = []

        for t_idx in range(self.history.n_recorded):
            v = velocities_all[t_idx]
            E = torch.mean(torch.sum(v**2, dim=1)).item()
            energies.append(E)
            times.append(t_idx * self.gas_config.delta_t)

        # Create energy plot
        energy_curve = hv.Curve((times, energies), kdims=["Time"], vdims=["Energy"]).opts(
            width=600,
            height=300,
            xlabel="Time t",
            ylabel="Kinetic Energy E(t)",
            title="Energy Evolution",
            color="blue",
            line_width=2,
        )

        # Add theoretical decay for Taylor-Green
        if isinstance(self.benchmark, TaylorGreenVortex):
            E0 = energies[0]
            nu_eff = self.gas_config.nu * self.gas_config.viscous_length_scale**2
            E_theory = [E0 * np.exp(-2 * nu_eff * t) for t in times]

            theory_curve = hv.Curve((times, E_theory), label="Theory E₀·exp(-2νt)").opts(
                color="red",
                line_dash="dashed",
                line_width=2,
            )

            plot = (energy_curve * theory_curve).opts(legend_position="top_right")
        else:
            plot = energy_curve

        return pn.pane.HoloViews(plot)

    def _update_validation_panel(self):
        """Update validation metrics panel."""
        if self.history is None:
            return

        # Get current frame index
        frame_idx = self.frame_index

        # Compute validation metrics with current parameters from gas_config
        params_dict = {
            "delta_t": self.gas_config.delta_t,
            "nu": self.gas_config.nu,
            "viscous_length_scale": self.gas_config.viscous_length_scale,
        }
        metrics = self.benchmark.compute_validation_metrics(self.history, frame_idx, params_dict)

        # Format metrics as markdown
        lines = [
            "## Validation Metrics",
            f"\n**Frame {frame_idx}/{self.history.n_recorded - 1}**\n",
        ]

        for metric in metrics:
            symbol = "✓" if metric.passed else "✗"

            lines.extend((
                f"### {symbol} {metric.metric_name}",
                f"- **Measured**: {metric.measured_value:.4f}",
            ))
            if metric.theoretical_value is not None:
                lines.append(f"- **Theory**: {metric.theoretical_value:.4f}")
                error = abs(metric.measured_value - metric.theoretical_value) / (
                    abs(metric.theoretical_value) + 1e-10
                )
                lines.append(f"- **Error**: {error:.2%}")
            lines.extend((f"- {metric.description}", ""))

        self.validation_pane.object = "\n".join(lines)

    def _toggle_play(self, *_):
        """Toggle auto-play animation."""
        self.auto_play = not self.auto_play
        self.play_button.name = "⏸ Pause" if self.auto_play else "▶ Play"

    @param.depends("frame_index")
    def _view_frame(self):
        """Dynamic view of current frame."""
        plot = self._render_frame(self.frame_index)
        self._update_validation_panel()
        return plot

    def _build_param_panel(self, names: tuple[str, ...]) -> pn.Param:
        """Build parameter panel with custom widgets.

        Args:
            names: Tuple of parameter names to include

        Returns:
            Panel Param widget with specified parameters
        """
        widgets = {
            name: self._widget_overrides[name] for name in names if name in self._widget_overrides
        }
        return pn.Param(
            self.param,
            parameters=list(names),
            widgets=widgets,
            show_name=False,
            sizing_mode="stretch_width",
        )

    def _create_sidebar_panel(self) -> pn.Column:
        """Create sidebar panel with all control parameters.

        Returns:
            Panel Column with accordion sections for parameters
        """
        # Benchmark selection (fluid-specific)
        benchmark_pane = pn.Param(
            self.param,
            parameters=["benchmark_name"],
            widgets={"benchmark_name": pn.widgets.RadioButtonGroup},
            show_name=False,
            sizing_mode="stretch_width",
        )

        # Create accordion with benchmark selection + filtered gas config panel
        benchmark_accordion = pn.Accordion(
            ("Benchmark Selection", benchmark_pane),
            sizing_mode="stretch_width",
        )
        benchmark_accordion.active = [0]  # Keep benchmark selection open

        # Get filtered gas config panel (hides benchmark and initialization sections)
        gas_config_panel = self._get_filtered_gas_config_panel()

        return pn.Column(
            pn.pane.Markdown("## Fluid Dynamics Validation"),
            benchmark_accordion,
            pn.pane.Markdown("### Gas Algorithm Parameters"),
            gas_config_panel,  # This is now an Accordion directly
            self.load_params_button,
            self.run_button,
            self.status_pane,
            sizing_mode="stretch_width",
            min_width=400,
        )

    def _create_main_panel(self) -> pn.Column:
        """Create main visualization panel.

        Returns:
            Panel Column with visualization controls and plots
        """
        # Visualization toggle parameters
        viz_params = (
            "show_particles",
            "show_velocity_field",
            "show_vorticity",
            "show_density",
            "show_energy_plot",
            "grid_resolution",
            "kernel_bandwidth",
            "quiver_stride",
        )

        viz_controls = pn.Column(
            pn.pane.Markdown("### Visualization Options"),
            self._build_param_panel(viz_params),
            sizing_mode="stretch_width",
        )

        # Animation controls
        frame_slider = pn.widgets.IntSlider.from_param(
            self.param.frame_index,
            name="Frame",
            sizing_mode="stretch_width",
        )

        animation_controls = pn.Row(
            self.play_button,
            frame_slider,
            sizing_mode="stretch_width",
        )

        return pn.Column(
            pn.pane.Markdown("## Swarm Evolution as Fluid"),
            viz_controls,
            pn.pane.Markdown("### Main Visualization"),
            self._view_frame,
            animation_controls,
            pn.pane.Markdown("### Energy Evolution"),
            self._render_energy_plot,
            self.validation_pane,
            sizing_mode="stretch_width",
        )

    def panel(self) -> pn.template.FastListTemplate:
        """Create Panel dashboard layout with sidebar.

        Returns:
            FastListTemplate with sidebar and main panels
        """
        return pn.template.FastListTemplate(
            title="Fluid Dynamics Validation Dashboard",
            sidebar=[self._create_sidebar_panel()],
            main=[self._create_main_panel()],
            sidebar_width=420,
            main_max_width="100%",
        )


# ============================================================================
# Convenience Function
# ============================================================================


def create_fluid_dashboard(**kwargs) -> pn.Column:
    """Create and return fluid dynamics dashboard.

    Args:
        **kwargs: Override default parameters

    Returns:
        Panel dashboard ready to show()

    Example:
        >>> dashboard = create_fluid_dashboard()
        >>> dashboard.show(port=5006)
    """
    explorer = FluidDynamicsExplorer(**kwargs)
    return explorer.panel()


# ============================================================================
# Standalone Execution
# ============================================================================


if __name__ == "__main__":
    # Initialize HoloViews with Bokeh backend
    hv.extension("bokeh")

    # Create and serve dashboard
    dashboard = create_fluid_dashboard()
    dashboard.show(port=5007, open=False)
    print("Dashboard running at http://localhost:5007")
