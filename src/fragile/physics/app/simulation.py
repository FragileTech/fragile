"""Simulation tab: RunHistory I/O controls and gas config panel."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

import panel as pn

from fragile.physics.app.gas_config_panel import GasConfigPanel
from fragile.physics.app.swarm_viewer import SwarmConvergence3D
from fragile.physics.fractal_gas.history import RunHistory


class SimulationTab:
    """Encapsulates the Simulation tab: gas config, 3D visualizer, and RunHistory I/O."""

    def __init__(self, bounds_extent: float = 30.0, debug_fn: Callable | None = None):
        self._debug = debug_fn or (lambda *_args, **_kwargs: None)
        self._history_changed_callbacks: list[Callable] = []

        self._debug("building config + visualizer")
        self._gas_config = GasConfigPanel.create_qft_config(bounds_extent=bounds_extent)
        self._apply_default_gas_config()

        self._visualizer = SwarmConvergence3D(
            history=None, bounds_extent=self._gas_config.bounds_extent
        )

        # Internal state
        self._history: RunHistory | None = None
        self._history_path: Path | None = None

        # Default history path
        repo_root = Path(__file__).resolve().parents[4]
        qft_run_id = "qft_penalty_thr0p75_pen0p9_m354_ed2p80_nu1p10_N200_long"
        self._qft_history_path = (
            repo_root / "outputs" / "qft_calibrated" / f"{qft_run_id}_history.pt"
        )
        self._history_dir = self._qft_history_path.parent
        self._history_dir.mkdir(parents=True, exist_ok=True)

        # Build widgets
        self._debug("setting up history controls")
        self._history_path_input = pn.widgets.TextInput(
            name="QFT RunHistory path",
            value=str(self._qft_history_path),
            min_width=335,
            sizing_mode="stretch_width",
        )
        self._browse_button = pn.widgets.Button(
            name="Browse files...",
            button_type="default",
            min_width=335,
            sizing_mode="stretch_width",
        )
        self._file_selector_container = pn.Column(sizing_mode="stretch_width")
        self._load_button = pn.widgets.Button(
            name="Load RunHistory",
            button_type="primary",
            min_width=335,
            sizing_mode="stretch_width",
        )
        self._save_button = pn.widgets.Button(
            name="Save RunHistory",
            button_type="primary",
            min_width=335,
            sizing_mode="stretch_width",
            disabled=True,
        )
        self._load_status = pn.pane.Markdown(
            "**Load a history**: paste a *_history.pt path or browse and click Load.",
            sizing_mode="stretch_width",
        )
        self._save_status = pn.pane.Markdown(
            "**Save a history**: run a simulation or load a RunHistory first.",
            sizing_mode="stretch_width",
        )
        self._simulation_compute_status = pn.pane.Markdown(
            "**Simulation:** run or load a RunHistory, then click Compute Simulation.",
            sizing_mode="stretch_width",
        )
        self._simulation_compute_button = pn.widgets.Button(
            name="Compute Simulation",
            button_type="primary",
            min_width=240,
            sizing_mode="stretch_width",
            disabled=True,
        )

        # Wire internal events
        self._browse_button.on_click(self._on_browse_clicked)
        self._load_button.on_click(self._on_load_clicked)
        self._save_button.on_click(self._on_save_clicked)
        self._simulation_compute_button.on_click(self._on_compute_simulation_clicked)
        self._gas_config.add_completion_callback(self._on_simulation_complete)
        self._gas_config.param.watch(self._on_bounds_change, "bounds_extent")

    def _apply_default_gas_config(self) -> None:
        """Override with the best stable calibration settings found in QFT tuning."""
        self._gas_config.n_steps = 750
        self._gas_config.gas_params["N"] = 500
        self._gas_config.gas_params["dtype"] = "float32"
        self._gas_config.gas_params["clone_every"] = 20
        self._gas_config.neighbor_weight_modes = [
            "inverse_riemannian_distance",
            "kernel",
            "riemannian_kernel_volume",
        ]
        self._gas_config.init_offset = 0.0
        self._gas_config.init_spread = 0.0
        self._gas_config.init_velocity_scale = 0.0

        # Kinetic operator (Langevin + viscous coupling).
        self._gas_config.kinetic_op.gamma = 1.0
        self._gas_config.kinetic_op.beta = 1.0
        self._gas_config.kinetic_op.auto_thermostat = True
        self._gas_config.kinetic_op.delta_t = 0.002
        self._gas_config.kinetic_op.temperature = 0.33
        self._gas_config.kinetic_op.n_kinetic_steps = 1
        self._gas_config.kinetic_op.nu = 3.0
        self._gas_config.kinetic_op.beta_curl = 1.0
        self._gas_config.kinetic_op.use_viscous_coupling = True
        self._gas_config.kinetic_op.viscous_neighbor_weighting = "riemannian_kernel_volume"
        self._gas_config.kinetic_op.viscous_length_scale = 1.0

        # Cloning operator.
        self._gas_config.cloning.p_max = 1.0
        self._gas_config.cloning.epsilon_clone = 0.0
        self._gas_config.cloning.sigma_x = 0.0
        self._gas_config.cloning.alpha_restitution = 1.0

        # Fitness operator.
        self._gas_config.fitness_op.alpha = 1.0
        self._gas_config.fitness_op.beta = 1.0
        self._gas_config.fitness_op.eta = 0.0
        self._gas_config.fitness_op.sigma_min = 0.0
        self._gas_config.fitness_op.A = 2.0

    # -- Public properties --

    @property
    def gas_config(self) -> GasConfigPanel:
        return self._gas_config

    @property
    def visualizer(self) -> SwarmConvergence3D:
        return self._visualizer

    @property
    def history(self) -> RunHistory | None:
        return self._history

    @property
    def history_path(self) -> Path | None:
        return self._history_path

    # -- Public methods --

    def set_history(
        self,
        history: RunHistory,
        history_path: Path | None = None,
        defer_dashboard_updates: bool = False,
    ) -> None:
        """Update internal state and simulation widgets, then notify listeners."""
        self._history = history
        self._history_path = history_path

        if not defer_dashboard_updates:
            self._visualizer.bounds_extent = float(self._gas_config.bounds_extent)
            self._visualizer.set_history(history)

        self._simulation_compute_button.disabled = False
        self._simulation_compute_status.object = (
            "**Simulation:** click Compute Simulation to visualize this RunHistory."
        )

        if defer_dashboard_updates:
            self._visualizer.status_pane.object = (
                "**Simulation complete:** history captured; click a Compute button to "
                "run post-processing."
            )
            self._save_button.disabled = False
            self._save_status.object = "**Save a history**: choose a path and click Save."
        else:
            self._save_button.disabled = False
            self._save_status.object = "**Save a history**: choose a path and click Save."

        # Notify registered listeners
        for cb in self._history_changed_callbacks:
            cb(history, history_path, defer_dashboard_updates)

    def on_history_changed(self, callback: Callable) -> None:
        """Register callback(history: RunHistory, path: Path|None, defer: bool)."""
        self._history_changed_callbacks.append(callback)

    def build_run_history_panel(self) -> pn.Column:
        """Return the RunHistory sidebar section (path input, browse, load, save)."""
        return pn.Column(
            self._history_path_input,
            self._browse_button,
            self._file_selector_container,
            self._load_button,
            self._load_status,
            self._save_button,
            self._save_status,
            sizing_mode="stretch_width",
        )

    def build_simulation_sidebar_panel(self) -> pn.Column:
        """Return gas_config.panel() for the sidebar Simulation accordion."""
        return self._gas_config.panel()

    def build_visualization_controls(self) -> pn.Param:
        """Return visualization controls Param panel."""
        return pn.Param(
            self._visualizer,
            parameters=["point_size", "point_alpha", "color_metric", "fix_axes"],
            show_name=False,
        )

    def build_tab(self) -> pn.Column:
        """Return the main Simulation tab panel."""
        return pn.Column(
            self._simulation_compute_status,
            pn.Row(self._simulation_compute_button, sizing_mode="stretch_width"),
            self._visualizer.panel(),
            sizing_mode="stretch_both",
        )

    # -- Internal callbacks --

    def _on_simulation_complete(self, history: RunHistory) -> None:
        self.set_history(history, defer_dashboard_updates=True)

    def _sync_history_path(self, value):
        if value:
            self._history_path_input.value = str(value[0])

    def _ensure_file_selector(self) -> pn.widgets.FileSelector:
        if self._file_selector_container.objects:
            return self._file_selector_container.objects[0]
        selector = pn.widgets.FileSelector(
            name="Select RunHistory",
            directory=str(self._history_dir),
            file_pattern="*_history.pt",
            only_files=True,
        )
        if self._qft_history_path.exists():
            selector.value = [str(self._qft_history_path)]
        selector.param.watch(lambda e: self._sync_history_path(e.new), "value")
        self._file_selector_container.objects = [selector]
        return selector

    def _on_browse_clicked(self, _):
        self._ensure_file_selector()

    def _on_load_clicked(self, _):
        history_path = Path(self._history_path_input.value).expanduser()
        if not history_path.exists():
            self._load_status.object = "**Error:** History path does not exist."
            return
        try:
            history = RunHistory.load(str(history_path))
            self.set_history(history, history_path, defer_dashboard_updates=True)
            self._load_status.object = f"**Loaded:** `{history_path}`"
        except Exception as exc:
            self._load_status.object = f"**Error loading history:** {exc!s}"

    def _on_compute_simulation_clicked(self, _):
        if self._gas_config.run_button.disabled:
            self._simulation_compute_status.object = (
                "**Simulation:** simulation is currently running.\n\n"
                "Wait for completion before recomputing visualization."
            )
            return

        history = self._history
        if history is None:
            self._simulation_compute_status.object = (
                "**Error:** run a simulation or load a RunHistory first."
            )
            return

        self._simulation_compute_status.object = "**Computing Simulation...**"
        try:
            self._visualizer.set_history(history)
            self._simulation_compute_status.object = (
                f"**Simulation ready:** {history.n_steps} steps / "
                f"{history.n_recorded} recorded frames."
            )
        except Exception as exc:
            self._simulation_compute_status.object = f"**Error:** {exc!s}"

    def _on_save_clicked(self, _):
        history = self._history
        if history is None:
            self._save_status.object = "**Error:** run a simulation or load a RunHistory first."
            return
        raw_path = self._history_path_input.value.strip()
        if raw_path:
            history_path = Path(raw_path).expanduser()
        else:
            stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            history_path = self._history_dir / f"qft_{stamp}_history.pt"
        if history_path.exists() and history_path.is_dir():
            stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            history_path = history_path / f"qft_{stamp}_history.pt"
        elif history_path.suffix != ".pt":
            history_path = history_path.with_suffix(".pt")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            history.save(str(history_path))
            self._history_path = history_path
            self._history_path_input.value = str(history_path)
            self._save_status.object = f"**Saved:** `{history_path}`"
        except Exception as exc:
            self._save_status.object = f"**Error saving history:** {exc!s}"

    def _on_bounds_change(self, event):
        self._visualizer.bounds_extent = float(event.new)
        self._visualizer._refresh_frame()
