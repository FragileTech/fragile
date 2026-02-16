"""Modern parameter configuration dashboard using operator PanelModel interfaces.

This module provides a Panel-based dashboard that leverages the __panel__() methods
of EuclideanGas and its nested operators, replacing the manual GasConfig approach.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import panel as pn
import panel.widgets as pnw
import param
import torch

from fragile.fractalai.core.history import RunHistory
from fragile.physics.fractal_gas.cloning import CloneOperator
from fragile.physics.fractal_gas.euclidean_gas import EuclideanGas
from fragile.physics.fractal_gas.fitness import FitnessOperator
from fragile.physics.fractal_gas.kinetic_operator import KineticOperator


__all__ = ["GasConfigPanel"]


class GasConfigPanel(param.Parameterized):
    """Modern configuration dashboard using operator PanelModel interfaces.

    This class provides a Panel-based UI that uses the __panel__() methods from
    EuclideanGas and its nested operators (KineticOperator, CloneOperator, etc.)
    to create an organized accordion-based parameter dashboard.

    The physics EuclideanGas always runs Einstein-Hilbert actions on Delaunay
    tessellation with d=3 -- no configurable benchmark, no companion selection,
    no PBC, no toggleable operators.

    Example:
        >>> config = GasConfigPanel()
        >>> dashboard = config.panel()
        >>> dashboard.show()  # Interactive parameter selection
        >>> history = config.history  # Access result after running
    """

    # Simulation controls
    n_steps = param.Integer(
        default=240, bounds=(10, 10000), softbounds=(50, 1000), doc="Simulation steps"
    )
    record_every = param.Integer(
        default=1,
        bounds=(1, 1000),
        softbounds=(1, 200),
        doc="Record every k-th step (1=all steps)",
    )
    chunk_size = param.Integer(
        default=None,
        allow_None=True,
        bounds=(10, None),
        doc="Chunk size for history recording (None=no chunking). "
        "Reduces peak memory by flushing to disk every chunk_size steps.",
    )
    neighbor_graph_update_every = param.Integer(
        default=1,
        bounds=(1, 1000),
        softbounds=(1, 200),
        doc="Recompute neighbor graph every k steps",
    )
    neighbor_weight_modes = param.ListSelector(
        default=["inverse_riemannian_distance", "kernel", "riemannian_kernel_volume"],
        objects=[
            "uniform",
            "inverse_distance",
            "inverse_volume",
            "inverse_riemannian_volume",
            "inverse_riemannian_distance",
            "kernel",
            "riemannian_kernel",
            "riemannian_kernel_volume",
        ],
        doc="Edge weight modes to pre-compute during Voronoi tessellation",
    )

    # Initialization controls
    init_offset = param.Number(default=0.0, bounds=(-6.0, 6.0), doc="Initial position offset")
    init_spread = param.Number(default=10.0, bounds=(0.0, 50.0), doc="Initial position spread")
    init_velocity_scale = param.Number(
        default=0.0,
        bounds=(0.0, None),
        softbounds=(0.0, 2.0),
        doc="Initial velocity scale",
    )
    bounds_extent = param.Number(
        default=3.0,
        bounds=(1e-6, 1000.0),
        doc="Spatial bounds half-width",
    )

    def __init__(self, **params):
        """Initialize GasConfigPanel."""
        super().__init__(**params)
        self.dims = 3  # Always 3D (matches physics EuclideanGas default)
        self.history: RunHistory | None = None

        # Create default operators with sensible defaults
        self._create_default_operators()

        # Create UI components
        self.run_button = pn.widgets.Button(name="Run Simulation", button_type="primary")
        self.run_button.sizing_mode = "stretch_width"
        self.run_button.on_click(self._on_run_clicked)

        self.status_pane = pn.pane.Markdown("", sizing_mode="stretch_width")

        # Widget overrides for special cases
        self._widget_overrides: dict[str, pn.widgets.Widget] = {
            "n_steps": pnw.EditableIntSlider(
                name="n_steps", start=10, end=10000, value=self.n_steps, step=1
            ),
            "record_every": pnw.EditableIntSlider(
                name="record_every", start=1, end=1000, value=self.record_every, step=1
            ),
            "init_offset": pnw.FloatInput(
                name="init_offset",
                start=-6.0,
                end=6.0,
                value=float(self.init_offset),
                step=1e-6,
            ),
            "init_spread": pnw.FloatInput(
                name="init_spread",
                start=0.0,
                end=50.0,
                value=float(self.init_spread),
                step=1e-6,
            ),
            "init_velocity_scale": pnw.FloatInput(
                name="init_velocity_scale",
                start=0.0,
                end=50.0,
                value=float(self.init_velocity_scale),
                step=1e-6,
            ),
            "bounds_extent": pnw.FloatInput(
                name="bounds_extent",
                start=1e-6,
                end=1000.0,
                value=float(self.bounds_extent),
                step=1e-6,
            ),
        }
        self._widget_links: set[str] = set()

        self.progress_label = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.progress_bar = pn.indicators.Progress(
            name="Simulation progress",
            value=0,
            max=max(1, int(self.n_steps)),
            bar_color="primary",
            sizing_mode="stretch_width",
        )

        self._progress_update_interval = 0.2
        self._progress_last_emit = 0.0
        self._simulation_thread: threading.Thread | None = None

        self.param.watch(self._on_n_steps_change, "n_steps")
        n_steps_widget = self._widget_overrides.get("n_steps")
        if n_steps_widget is not None:
            n_steps_widget.param.watch(
                lambda e: self._sync_progress_total(e.new),
                "value",
            )
        self._sync_progress_total(self.n_steps)

        # Callbacks for external listeners
        self._on_simulation_complete: list[Callable[[RunHistory], None]] = []

    @staticmethod
    def create_qft_config(bounds_extent: float = 3.0) -> GasConfigPanel:
        """Create GasConfigPanel with QFT calibration defaults.

        These parameters match the calibrated simulation from
        08_qft_calibration_notebook.ipynb.

        Args:
            bounds_extent: Half-width of spatial domain (default: 3.0)

        Returns:
            GasConfigPanel configured with QFT parameters
        """
        config = GasConfigPanel()
        config.bounds_extent = float(bounds_extent)
        config.n_steps = 5000
        config.gas_params["N"] = 200
        config.gas_params["dtype"] = "float32"
        config.neighbor_weight_modes = [
            "inverse_riemannian_distance",
            "kernel",
            "riemannian_kernel_volume",
        ]
        config.init_offset = 0.0
        config.init_spread = 1.0
        config.init_velocity_scale = 0.0
        config.kinetic_op.gamma = 1.0
        config.kinetic_op.beta = 1.0
        config.kinetic_op.delta_t = 0.1005
        config.kinetic_op.nu = 0.125
        config.kinetic_op.use_viscous_coupling = True
        config.kinetic_op.viscous_length_scale = 0.251372
        config.kinetic_op.viscous_neighbor_weighting = "riemannian_kernel_volume"
        config.kinetic_op.beta_curl = 0.0
        config.cloning.p_max = 1.0
        config.cloning.epsilon_clone = 1e-6
        config.cloning.sigma_x = 0.01
        config.cloning.alpha_restitution = 0.5
        config.fitness_op.alpha = 1.0
        config.fitness_op.beta = 1.0
        config.fitness_op.eta = 0.1
        config.fitness_op.sigma_min = 1e-8
        config.fitness_op.A = 2.0
        return config

    # Backward compatibility properties
    @property
    def gamma(self):
        """Backward compatibility: delegate to kinetic_op.gamma"""
        return self.kinetic_op.gamma

    @property
    def beta(self):
        """Backward compatibility: delegate to kinetic effective beta."""
        if hasattr(self.kinetic_op, "effective_beta"):
            return float(self.kinetic_op.effective_beta())
        return self.kinetic_op.beta

    @property
    def delta_t(self):
        """Backward compatibility: delegate to kinetic_op.delta_t"""
        return self.kinetic_op.delta_t

    @property
    def N(self):
        """Backward compatibility: delegate to gas_params['N']"""
        return self.gas_params["N"]

    def _create_default_operators(self):
        """Create default operator instances with sensible defaults."""
        self.kinetic_op = KineticOperator(
            gamma=1.0,
            beta=1.0,
            delta_t=0.01,
            nu=0.1,
            use_viscous_coupling=True,
            viscous_length_scale=1.0,
            viscous_neighbor_weighting="riemannian_kernel_volume",
            beta_curl=0.0,
        )
        self.cloning = CloneOperator(
            sigma_x=1e-6,
            alpha_restitution=0.6,
            p_max=1.0,
            epsilon_clone=1e-6,
        )
        self.fitness_op = FitnessOperator(
            alpha=0.4,
            beta=2.5,
            eta=0.003,
            sigma_min=1e-8,
            A=3.5,
        )
        self.gas_params = {
            "N": 160,
            "d": self.dims,
            "dtype": "float32",
            "clone_every": 1,
        }

    def add_completion_callback(self, callback: Callable[[RunHistory], None]):
        """Register a callback to be called when simulation completes.

        Args:
            callback: Function that takes RunHistory as argument
        """
        self._on_simulation_complete.append(callback)

    def _format_eta(self, seconds: float | None) -> str:
        if seconds is None:
            return "n/a"
        seconds = max(0.0, seconds)
        total_seconds = round(seconds)
        minutes, secs = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:d}:{minutes:02d}:{secs:02d}"

    def _progress_text(self, step: int, total_steps: int, eta_seconds: float | None) -> str:
        total_steps = max(1, int(total_steps))
        step = max(0, min(int(step), total_steps))
        percent = (step / total_steps) * 100.0 if total_steps else 0.0
        eta_str = self._format_eta(eta_seconds)
        return (
            f"**Progress:** {step}/{total_steps} ({percent:.1f}%) " f"| **Remaining:** {eta_str}"
        )

    def _sync_progress_total(self, total_steps: int) -> None:
        total_steps = max(1, int(total_steps))
        current = min(int(self.progress_bar.value), total_steps)
        self.progress_bar.max = total_steps
        self.progress_bar.value = current
        self.progress_label.object = self._progress_text(current, total_steps, None)

    def _on_n_steps_change(self, event) -> None:
        self._sync_progress_total(event.new)

    def _schedule_ui_update(self, callback: Callable[[], None]) -> None:
        doc = pn.state.curdoc
        if doc is None:
            callback()
        else:
            doc.add_next_tick_callback(callback)

    def _update_progress_display(
        self, step: int, total_steps: int, eta_seconds: float | None
    ) -> None:
        total_steps = max(1, int(total_steps))
        step = max(0, min(int(step), total_steps))
        self.progress_bar.max = total_steps
        self.progress_bar.value = step
        self.progress_label.object = self._progress_text(step, total_steps, eta_seconds)

    def _progress_callback(self, step: int, total_steps: int, elapsed: float) -> None:
        now = time.perf_counter()
        if (
            step < total_steps
            and (now - self._progress_last_emit) < self._progress_update_interval
        ):
            return
        self._progress_last_emit = now
        eta = None
        if step > 0 and total_steps > step:
            eta = (elapsed / step) * (total_steps - step)
        self._schedule_ui_update(lambda: self._update_progress_display(step, total_steps, eta))

    def _current_steps_value(self) -> int:
        widget = self._widget_overrides.get("n_steps")
        if widget is not None and hasattr(widget, "value"):
            return int(widget.value)
        return int(self.n_steps)

    def _run_simulation_worker(self) -> None:
        history: RunHistory | None = None
        error: Exception | None = None
        try:
            history = self.run_simulation(progress_callback=self._progress_callback)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            error = exc

        def _finalize() -> None:
            self.run_button.disabled = False
            if error is not None:
                self.status_pane.object = f"**Error:** {error!s}"
                self.progress_bar.bar_color = "danger"
                return

            if history is None:
                self.status_pane.object = "**Error:** simulation failed without history."
                self.progress_bar.bar_color = "danger"
                return

            if history.terminated_early:
                self.status_pane.object = (
                    f"**Terminated early** at step {history.final_step}/{history.n_steps} "
                    f"({history.n_recorded} recorded timesteps)"
                )
                self.progress_bar.bar_color = "danger"
            else:
                self.status_pane.object = (
                    f"**Simulation complete!** {history.n_steps} steps, "
                    f"{history.n_recorded} recorded timesteps"
                )
                self.progress_bar.bar_color = "success"
            self._update_progress_display(history.final_step, self.progress_bar.max, 0.0)

        self._schedule_ui_update(_finalize)

    def _on_run_clicked(self, *_):
        """Handle Run button click."""
        if self._simulation_thread is not None and self._simulation_thread.is_alive():
            return

        total_steps = self._current_steps_value()
        self._sync_progress_total(total_steps)
        self._update_progress_display(0, self.progress_bar.max, None)
        self.progress_bar.bar_color = "primary"
        self.status_pane.object = "**Running simulation...**"
        self.run_button.disabled = True
        self._progress_last_emit = 0.0

        self._simulation_thread = threading.Thread(
            target=self._run_simulation_worker,
            name="gas-simulation-thread",
            daemon=True,
        )
        self._simulation_thread.start()

    def run_simulation(
        self,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> RunHistory:
        """Run EuclideanGas simulation with current parameters.

        Returns:
            RunHistory object containing complete execution trace

        Raises:
            ValueError: If parameters are invalid
        """
        for name in ("n_steps", "record_every", "chunk_size"):
            widget = self._widget_overrides.get(name)
            if widget is not None and hasattr(widget, "value"):
                setattr(self, name, widget.value)

        # Create EuclideanGas using current operator instances
        gas = EuclideanGas(
            N=int(self.gas_params["N"]),
            d=self.dims,
            kinetic_op=self.kinetic_op,
            cloning=self.cloning,
            fitness_op=self.fitness_op,
            device=torch.device("cpu"),
            dtype=self.gas_params["dtype"],
            clone_every=int(self.gas_params.get("clone_every", 1)),
            neighbor_graph_update_every=int(self.neighbor_graph_update_every),
            neighbor_weight_modes=list(self.neighbor_weight_modes),
        )

        # Initialize state
        x_init = (
            torch.randn(self.gas_params["N"], self.dims) * float(self.init_spread)
            + float(self.init_offset)
        )
        v_init = torch.randn(self.gas_params["N"], self.dims) * float(self.init_velocity_scale)

        # Run simulation
        history = gas.run(
            self.n_steps,
            x_init=x_init,
            v_init=v_init,
            record_every=int(self.record_every),
            progress_callback=progress_callback,
            chunk_size=self.chunk_size,
        )

        # Store history and notify listeners
        self.history = history
        for callback in self._on_simulation_complete:
            self._schedule_ui_update(lambda cb=callback, hist=self.history: cb(hist))

        return history

    def _build_param_panel(self, names: list[str]) -> pn.Param:
        """Build parameter panel with custom widgets."""
        widgets = {
            name: self._widget_overrides[name] for name in names if name in self._widget_overrides
        }

        def _coerce_widget_value(widget_ref: pn.widgets.Widget, value):
            if isinstance(widget_ref, pnw.EditableIntSlider):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return value
            return value

        for name, widget in widgets.items():
            if hasattr(widget, "value"):
                widget.value = _coerce_widget_value(widget, getattr(self, name))
                if name not in self._widget_links:
                    widget.param.watch(
                        lambda e, param_name=name: setattr(self, param_name, e.new),
                        "value",
                    )
                    self.param.watch(
                        lambda e, widget_ref=widget: (
                            None
                            if getattr(widget_ref, "value", None) == e.new
                            else setattr(
                                widget_ref, "value", _coerce_widget_value(widget_ref, e.new)
                            )
                        ),
                        name,
                    )
                    self._widget_links.add(name)
        return pn.Param(
            self.param,
            parameters=names,
            widgets=widgets,
            show_name=False,
            sizing_mode="stretch_width",
        )

    def _thermostat_status_markdown(
        self,
        gamma: float,
        beta_manual: float,
        auto_thermostat: bool,
        sigma_v: float,
    ) -> str:
        """Build a live thermostat status readout for the Langevin panel."""
        del gamma, beta_manual, auto_thermostat, sigma_v  # values used via reactive binding

        gamma_val = float(self.kinetic_op.gamma)
        beta_val = float(self.kinetic_op.beta)
        auto_mode = bool(getattr(self.kinetic_op, "auto_thermostat", False))
        sigma_val = float(getattr(self.kinetic_op, "sigma_v", float("nan")))
        beta_eff = (
            float(self.kinetic_op.effective_beta())
            if hasattr(self.kinetic_op, "effective_beta")
            else beta_val
        )
        t_eff = float("inf") if beta_eff <= 0 else 1.0 / beta_eff
        mode_label = "Auto thermostat (FDT)" if auto_mode else "Manual beta thermostat"

        return (
            "#### Thermostat Readout\n"
            f"- mode: `{mode_label}`\n"
            f"- friction `gamma`: `{gamma_val:.6g}`\n"
            f"- manual `beta`: `{beta_val:.6g}`\n"
            f"- noise `sigma_v`: `{sigma_val:.6g}`\n"
            f"- active `beta_effective`: `{beta_eff:.6g}`\n"
            f"- active `T_effective = 1 / beta_effective`: `{t_eff:.6g}`"
        )

    def panel(self) -> pn.Column:
        """Create Panel dashboard using operator __panel__() methods.

        Returns:
            Panel Column with organized parameter sections and Run button
        """
        # === General Panel ===
        n_slider = pn.widgets.EditableIntSlider(
            name="N (walkers)",
            value=self.gas_params["N"],
            start=2,
            end=10000,
            step=1,
        )
        n_slider.param.watch(lambda e: self.gas_params.update({"N": e.new}), "value")

        neighbor_params = [
            "neighbor_graph_update_every",
            "neighbor_weight_modes",
        ]
        general_panel = pn.Column(
            n_slider,
            self._build_param_panel(["n_steps", "record_every", "chunk_size"]),
            self._build_param_panel(neighbor_params),
            sizing_mode="stretch_width",
        )

        # === Langevin Panel ===
        langevin_params = list(self.kinetic_op.widget_parameters)
        kinetic_widgets = {
            name: (dict(cfg) if isinstance(cfg, dict) else cfg)
            for name, cfg in self.kinetic_op.widgets.items()
        }

        thermostat_readout = pn.pane.Markdown(
            pn.bind(
                self._thermostat_status_markdown,
                self.kinetic_op.param.gamma,
                self.kinetic_op.param.beta,
                self.kinetic_op.param.auto_thermostat,
                self.kinetic_op.param.sigma_v,
            ),
            sizing_mode="stretch_width",
        )
        langevin_controls = pn.Param(
            self.kinetic_op,
            show_name=False,
            parameters=langevin_params,
            widgets=self.kinetic_op.process_widgets(kinetic_widgets),
            default_layout=self.kinetic_op.default_layout,
        )
        langevin_panel = pn.Column(
            thermostat_readout,
            pn.layout.Divider(),
            langevin_controls,
            sizing_mode="stretch_width",
        )

        # === Cloning & Fitness Panel ===
        clone_every_input = pn.widgets.IntInput(
            name="Clone every (steps)",
            value=int(self.gas_params.get("clone_every", 1)),
            start=1,
            end=10000,
            step=1,
        )

        def _set_clone_every(event):
            try:
                value = int(event.new)
            except (TypeError, ValueError):
                value = 1
            value = max(1, value)
            self.gas_params["clone_every"] = value
            if getattr(clone_every_input, "value", None) != value:
                clone_every_input.value = value

        clone_every_input.param.watch(_set_clone_every, "value")

        fitness_params = list(self.fitness_op.widget_parameters)

        def _build_operator_panel(operator, parameters: list[str]) -> pn.Param:
            widgets = {
                name: (dict(cfg) if isinstance(cfg, dict) else cfg)
                for name, cfg in operator.widgets.items()
                if name in parameters
            }
            return pn.Param(
                operator,
                show_name=False,
                parameters=parameters,
                widgets=operator.process_widgets(widgets),
                default_layout=operator.default_layout,
            )

        cloning_panel_combined = pn.Column(
            pn.pane.Markdown("#### Cloning Operator"),
            clone_every_input,
            self.cloning.__panel__(),
            pn.pane.Markdown("#### Fitness Potential"),
            _build_operator_panel(self.fitness_op, fitness_params),
            sizing_mode="stretch_width",
        )

        # === Initialization Panel ===
        init_panel = self._build_param_panel([
            "init_offset",
            "init_spread",
            "init_velocity_scale",
            "bounds_extent",
        ])

        # === Accordion Organization ===
        accordion = pn.Accordion(
            ("General", general_panel),
            ("Langevin Dynamics", langevin_panel),
            ("Cloning & Fitness", cloning_panel_combined),
            ("Initialization", init_panel),
            sizing_mode="stretch_width",
        )
        # Open general section by default
        accordion.active = [0, 1, 2, 3]

        return pn.Column(
            pn.pane.Markdown("## Simulation Parameters"),
            accordion,
            self.progress_label,
            self.progress_bar,
            self.run_button,
            self.status_pane,
            sizing_mode="stretch_width",
            min_width=380,
        )
