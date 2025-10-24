"""N-particle swarm visualization for Gas algorithms.

This module provides the GasVisualizer class for interactive visualization
and animation of N-particle swarm dynamics from EuclideanGas simulations.

Features:
- Real-time playback of RunHistory trajectories
- Customizable color/size encodings (velocity, fitness, reward, distance)
- Velocity and force vector visualization
- Continuous vector field overlays (fluid-like visualization)
- Distribution histograms (fitness, reward, distance, Hessian, forces, velocity)
- Support for periodic boundary conditions

Can be used standalone or integrated into visualization dashboards.
"""

from __future__ import annotations

from typing import Sequence

import holoviews as hv
from holoviews import dim
import numpy as np
import pandas as pd
import panel as pn
import param
import torch

from fragile.bounds import TorchBounds
from fragile.core.companion_selection import CompanionSelection
from fragile.core.history import RunHistory
from fragile.experiments.fluid_utils import FluidFieldComputer
from fragile.shaolin.stream_plots import Histogram


__all__ = ["GasVisualizer"]


class GasVisualizer(param.Parameterized):
    """Visualization dashboard for RunHistory exploration.

    This class provides interactive visualization and animation of RunHistory
    data from EuclideanGas simulations, with support for velocity/force vectors,
    customizable coloring, and temporal playback.

    Example:
        >>> history = gas.run(n_steps=100)
        >>> viz = GasVisualizer(history, potential, background, mode_points)
        >>> dashboard = viz.panel()
        >>> dashboard.show()
    """

    # Display parameters
    measure_stride = param.Integer(default=1, bounds=(1, 20), doc="Downsample stride")
    color_metric = param.ObjectSelector(
        default="constant",
        objects=("constant", "velocity", "fitness", "reward", "distance"),
        doc="Walker color encoding",
    )
    size_metric = param.ObjectSelector(
        default="constant",
        objects=("constant", "velocity", "fitness", "reward", "distance"),
        doc="Walker size encoding",
    )

    # Vector visualization
    show_velocity_vectors = param.Boolean(
        default=False,
        doc="Display velocity vectors showing trajectory from previous to current position",
    )
    color_vectors_by_cloning = param.Boolean(
        default=False,
        doc=(
            "Color velocity vectors yellow if walker was created by cloning "
            "(requires show_velocity_vectors)"
        ),
    )
    show_force_vectors = param.Boolean(
        default=False, doc="Display force vectors F = -∇U - ε_F·∇V_fit at current positions"
    )
    force_arrow_length = param.Number(
        default=0.5, bounds=(0.1, 2.0), doc="Length scale for normalized force arrows"
    )
    enabled_histograms = param.ListSelector(
        default=["fitness", "distance", "reward", "hessian", "forces", "velocity"],
        objects=["fitness", "distance", "reward", "hessian", "forces", "velocity"],
        doc="Select which histogram metrics to compute and display (disable to skip computation)",
    )

    # Vector field overlays
    show_velocity_field = param.Boolean(
        default=False,
        doc="Overlay continuous velocity vector field computed on grid (fluid-like visualization)",
    )
    show_force_field = param.Boolean(
        default=False,
        doc="Overlay continuous force vector field computed on grid",
    )
    field_grid_resolution = param.Integer(
        default=15,
        bounds=(8, 30),
        doc="Grid resolution for vector field computation (NxN grid)",
    )
    field_kernel_bandwidth = param.Number(
        default=0.5,
        bounds=(0.1, 2.0),
        doc="Kernel bandwidth for field interpolation (smaller = more local)",
    )
    field_scale = param.Number(
        default=1.0,
        bounds=(0.1, 3.0),
        doc="Arrow length scale for vector fields",
    )

    def __init__(
        self,
        history: RunHistory | None,
        potential: object,
        background: hv.Image,
        mode_points: hv.Points,
        companion_selection: CompanionSelection | None = None,
        fitness_op: object | None = None,
        bounds_extent: float = 6.0,
        epsilon_F: float = 0.0,
        use_fitness_force: bool = False,
        use_potential_force: bool = False,
        pbc: bool = False,
        **params,
    ):
        """Initialize GasVisualizer with RunHistory and display settings.

        Args:
            history: RunHistory object to visualize (can be None initially)
            potential: Potential function object
            background: HoloViews Image for background visualization
            mode_points: HoloViews Points for target modes
            companion_selection: CompanionSelection for recomputing fitness (optional)
            fitness_op: FitnessOperator for fitness computation (optional)
            bounds_extent: Spatial bounds half-width
            epsilon_F: Fitness force strength (for force vector display)
            use_fitness_force: Whether fitness force is enabled
            use_potential_force: Whether potential force is enabled
            pbc: Enable periodic boundary conditions (torus topology)
            **params: Override default display parameters
        """
        super().__init__(**params)
        self.potential = potential
        self.background = background
        self.mode_points = mode_points
        self.companion_selection = companion_selection
        self.fitness_op = fitness_op
        self.bounds_extent = bounds_extent
        self.epsilon_F = epsilon_F
        self.use_fitness_force = use_fitness_force
        self.use_potential_force = use_potential_force
        self.pbc = pbc

        # Create TorchBounds object for periodic distance calculations
        # Will be initialized when we know dimensionality from history
        self.bounds: TorchBounds | None = None

        self.history: RunHistory | None = None
        self.result: dict | None = None

        # Create playback controls
        self.time_player = pn.widgets.Player(
            name="time",
            start=0,
            end=0,
            value=0,
            step=1,
            interval=150,
            loop_policy="loop",
        )
        self.time_player.disabled = True
        self.time_player.sizing_mode = "stretch_width"
        self.time_player.param.watch(self._sync_stream, "value")

        # Create dynamic maps
        self.frame_stream = hv.streams.Stream.define("Frame", frame=0)()
        self.dmap_main = hv.DynamicMap(self._render_main_plot, streams=[self.frame_stream])

        # Create 6 Shaolin Histogram streaming plots
        self.histogram_fitness = Histogram(data=None, n_bins=30)
        self.histogram_fitness.opts(
            width=220,
            height=220,
            title="Fitness Distribution",
            xlabel="Fitness",
            ylabel="density",
            color="#1f77b4",
            line_color="#1f77b4",
            alpha=0.6,
        )

        self.histogram_distance = Histogram(data=None, n_bins=30)
        self.histogram_distance.opts(
            width=220,
            height=220,
            title="Distance Distribution",
            xlabel="Distance",
            ylabel="density",
            color="#2ca02c",
            line_color="#2ca02c",
            alpha=0.6,
        )

        self.histogram_reward = Histogram(data=None, n_bins=30)
        self.histogram_reward.opts(
            width=220,
            height=220,
            title="Reward Distribution",
            xlabel="Reward",
            ylabel="density",
            color="#d62728",
            line_color="#d62728",
            alpha=0.6,
        )

        self.histogram_hessian = Histogram(data=None, n_bins=30)
        self.histogram_hessian.opts(
            width=220,
            height=220,
            title="Hessian Distribution",
            xlabel="Hessian",
            ylabel="density",
            color="#8c564b",
            line_color="#8c564b",
            alpha=0.6,
        )

        self.histogram_forces = Histogram(data=None, n_bins=30)
        self.histogram_forces.opts(
            width=220,
            height=220,
            title="Forces Distribution",
            xlabel="Forces",
            ylabel="density",
            color="#ff7f0e",
            line_color="#ff7f0e",
            alpha=0.6,
        )

        self.histogram_velocity = Histogram(data=None, n_bins=30)
        self.histogram_velocity.opts(
            width=220,
            height=220,
            title="Velocity Distribution",
            xlabel="Velocity",
            ylabel="density",
            color="#9467bd",
            line_color="#9467bd",
            alpha=0.6,
        )

        # Wrap Shaolin histogram plots in Panel panes
        self.histogram_panes = {
            "fitness": pn.panel(self.histogram_fitness.plot, sizing_mode="stretch_width"),
            "distance": pn.panel(self.histogram_distance.plot, sizing_mode="stretch_width"),
            "reward": pn.panel(self.histogram_reward.plot, sizing_mode="stretch_width"),
            "hessian": pn.panel(self.histogram_hessian.plot, sizing_mode="stretch_width"),
            "forces": pn.panel(self.histogram_forces.plot, sizing_mode="stretch_width"),
            "velocity": pn.panel(self.histogram_velocity.plot, sizing_mode="stretch_width"),
        }

        # Initialize visibility based on enabled_histograms
        for metric, pane in self.histogram_panes.items():
            pane.visible = metric in self.enabled_histograms

        # Status display
        self.status_pane = pn.pane.Markdown("", sizing_mode="stretch_width")

        # Track previous enabled histograms for change detection
        self._prev_enabled_histograms = set(self.enabled_histograms)

        # Watch display parameters for frame refresh
        self.param.watch(
            self._refresh_frame,
            [
                "color_metric",
                "size_metric",
                "show_velocity_vectors",
                "color_vectors_by_cloning",
                "show_force_vectors",
                "force_arrow_length",
                "show_velocity_field",
                "show_force_field",
                "field_grid_resolution",
                "field_kernel_bandwidth",
                "field_scale",
            ],
        )

        # Separate watcher for enabled_histograms to handle reprocessing
        self.param.watch(self._on_histograms_changed, "enabled_histograms")

        # Load initial history if provided
        if history is not None:
            self.set_history(history)
        else:
            # No history loaded - hide all histogram panes
            self._update_histogram_streams(0)

    def set_history(self, history: RunHistory):
        """Load a new RunHistory for visualization.

        Args:
            history: RunHistory object to visualize
        """
        self.history = history

        # Initialize TorchBounds object now that we know dimensionality
        if self.pbc and self.bounds is None and history is not None:
            low = torch.full((history.d,), -self.bounds_extent, dtype=torch.float32)
            high = torch.full((history.d,), self.bounds_extent, dtype=torch.float32)
            self.bounds = TorchBounds(low=low, high=high)

        self._process_history()
        self._refresh_frame()

    def update_benchmark(self, potential: object, background: hv.Image, mode_points: hv.Points):
        """Update the benchmark function, background, and mode points.

        Args:
            potential: New potential function object
            background: New HoloViews Image for background
            mode_points: New HoloViews Points for target modes
        """
        self.potential = potential
        self.background = background
        self.mode_points = mode_points
        self._refresh_frame()

    def _sync_stream(self, event):
        """Sync time player value to frame stream."""
        if not self.result:
            return
        max_frame = len(self.result["times"]) - 1
        frame = int(np.clip(event.new, 0, max_frame)) if max_frame >= 0 else 0
        self.frame_stream.event(frame=frame)
        self._update_histogram_streams(frame)

    def _refresh_frame(self, *_):
        """Refresh current frame display."""
        if not self.result:
            return
        frame = self.time_player.value
        self.frame_stream.event(frame=frame)
        self._update_histogram_streams(frame)

    def _on_histograms_changed(self, event):
        """Handle changes to enabled_histograms parameter.

        Reprocesses history if new metrics were enabled that require computation.
        Updates visibility and streams for all histograms.
        """
        new_enabled = set(event.new)
        prev_enabled = self._prev_enabled_histograms

        # Check if any new metrics were enabled
        newly_enabled = new_enabled - prev_enabled

        if newly_enabled and self.history is not None:
            # New metrics require reprocessing history
            self._process_history()

        # Update visibility and streams for current frame
        if self.result:
            self._update_histogram_streams(self.time_player.value)

        # Update tracking
        self._prev_enabled_histograms = new_enabled

    def _process_history(self):
        """Process RunHistory into display-ready format."""
        if self.history is None:
            self.result = None
            return

        stride = max(1, int(self.measure_stride))
        history = self.history

        x_traj = history.x_final.detach().cpu().numpy()
        v_traj = history.v_final.detach().cpu().numpy()
        n_alive = history.n_alive.detach().cpu().numpy()
        will_clone_traj = history.will_clone.detach().cpu().numpy()

        # Check if Hessians and gradients are already computed in history (from kinetic operator)
        use_precomputed_hessians = history.fitness_hessians_diag is not None
        use_precomputed_gradients = history.fitness_gradients is not None

        # Compute variances
        var_x = torch.var(history.x_final, dim=1).sum(dim=-1).detach().cpu().numpy()
        var_v = torch.var(history.v_final, dim=1).sum(dim=-1).detach().cpu().numpy()

        indices = np.arange(0, x_traj.shape[0], stride)
        if indices[-1] != x_traj.shape[0] - 1:
            indices = np.append(indices, x_traj.shape[0] - 1)

        positions = x_traj[indices]
        V_total = (var_x + var_v)[indices]
        times = indices.astype(int)
        alive = n_alive[indices]

        # Prepare per-frame data
        velocity_series: list[np.ndarray] = []
        fitness_series: list[np.ndarray] = []
        distance_series: list[np.ndarray] = []
        reward_series: list[np.ndarray] = []
        hessian_series: list[np.ndarray] = []
        alive_masks: list[np.ndarray] = []
        previous_positions: list[np.ndarray | None] = []
        will_clone_series: list[np.ndarray] = []
        force_vectors_series: list[np.ndarray] = []
        force_magnitudes_series: list[np.ndarray] = []

        # Create bounds for alive check
        low = torch.full((history.d,), -self.bounds_extent, dtype=torch.float32)
        high = torch.full((history.d,), self.bounds_extent, dtype=torch.float32)
        bounds = TorchBounds(low=low, high=high)

        # Determine what metrics need computation based on display settings and usage
        needs_fitness = (
            any(m in self.enabled_histograms for m in ("fitness", "distance", "reward"))
            or self.color_metric in ("fitness", "distance", "reward")
            or self.size_metric in ("fitness", "distance", "reward")
        )
        needs_hessian = "hessian" in self.enabled_histograms
        needs_forces = "forces" in self.enabled_histograms or self.show_force_vectors

        for step_idx in indices:
            x_t = torch.from_numpy(x_traj[step_idx]).to(dtype=torch.float32)
            v_t = torch.from_numpy(v_traj[step_idx]).to(dtype=torch.float32)

            # Store previous position
            if step_idx == 0:
                previous_positions.append(None)
            else:
                prev_idx = max(0, step_idx - 1)
                previous_positions.append(x_traj[prev_idx])

            # Store cloning flags
            if step_idx == 0:
                will_clone_series.append(np.zeros(x_t.shape[0], dtype=bool))
            else:
                will_clone_idx = step_idx - 1
                if will_clone_idx < will_clone_traj.shape[0]:
                    will_clone_series.append(will_clone_traj[will_clone_idx])
                else:
                    will_clone_series.append(np.zeros(x_t.shape[0], dtype=bool))

            with torch.no_grad():
                alive_mask = bounds.contains(x_t)

            alive_np = alive_mask.cpu().numpy().astype(bool)
            alive_masks.append(alive_np.copy())

            vel_mag = torch.linalg.norm(v_t, dim=1).cpu().numpy()

            # Compute fitness and distances if needed
            if (
                needs_fitness
                and alive_np.any()
                and self.companion_selection is not None
                and self.fitness_op is not None
            ):
                with torch.no_grad():
                    rewards = -self.potential(x_t)
                    companions = self.companion_selection(
                        x=x_t,
                        v=v_t,
                        alive_mask=alive_mask,
                        bounds=self.bounds,
                        pbc=self.pbc,
                    )

                    # Use FitnessOperator to compute fitness
                    fitness_vals, info = self.fitness_op(
                        positions=x_t,
                        velocities=v_t,
                        rewards=rewards,
                        alive=alive_mask,
                        companions=companions,
                        bounds=self.bounds,
                        pbc=self.pbc,
                    )
                    distances = info["distances"]

                    rewards_np = rewards.detach().cpu().numpy()
                    fitness_np = fitness_vals.detach().cpu().numpy()
                    distances_np = distances.detach().cpu().numpy()
            else:
                rewards_np = np.zeros(x_t.shape[0], dtype=np.float32)
                fitness_np = np.zeros(x_t.shape[0], dtype=np.float32)
                distances_np = np.zeros(x_t.shape[0], dtype=np.float32)
                # These variables may be needed for hessian computation
                rewards = None
                companions = None

            # Get Hessian diagonal: reuse from history if available, otherwise compute
            if needs_hessian:
                if use_precomputed_hessians:
                    # Reuse Hessians computed during simulation (no gradient computation needed)
                    # Note: history stores [n_recorded, N, d]
                    # We need to map step_idx to history index (downsampled indices array)
                    hist_idx = np.where(indices == step_idx)[0][0]
                    if hist_idx < history.fitness_hessians_diag.shape[0]:
                        hessian_diag_t = history.fitness_hessians_diag[hist_idx]
                        hessian_mag = torch.linalg.norm(hessian_diag_t, dim=1)
                        hessian_np = hessian_mag.detach().cpu().numpy()
                    else:
                        hessian_np = np.zeros(x_t.shape[0], dtype=np.float32)
                elif (
                    self.fitness_op is not None and rewards is not None and companions is not None
                ):
                    # Compute Hessian diagonal on-the-fly (requires gradients and fitness data)
                    hessian_diag = self.fitness_op.compute_hessian(
                        positions=x_t,
                        velocities=v_t,
                        rewards=rewards,
                        alive=alive_mask,
                        companions=companions,
                        diagonal_only=True,
                    )
                    # Compute magnitude of Hessian diagonal per walker
                    hessian_mag = torch.linalg.norm(hessian_diag, dim=1)
                    hessian_np = hessian_mag.detach().cpu().numpy()
                else:
                    # Hessian requested but can't compute
                    # (no precomputed data or fitness not computed)
                    hessian_np = np.zeros(x_t.shape[0], dtype=np.float32)
            else:
                hessian_np = np.zeros(x_t.shape[0], dtype=np.float32)

            velocity_series.append(vel_mag[alive_np])
            fitness_series.append(fitness_np[alive_np])
            distance_series.append(distances_np[alive_np])
            reward_series.append(rewards_np[alive_np])
            hessian_series.append(hessian_np[alive_np])

            # Compute force vectors if needed
            force_vectors_np = np.zeros((x_t.shape[0], x_t.shape[1]), dtype=np.float32)
            force_mag_np = np.zeros(x_t.shape[0], dtype=np.float32)

            if (
                needs_forces
                and alive_np.any()
                and (self.use_potential_force or self.use_fitness_force)
            ):
                force_total = torch.zeros_like(x_t)

                # Potential force
                if self.use_potential_force:
                    x_t_grad = x_t.clone().requires_grad_(True)  # noqa: FBT003
                    U = self.potential(x_t_grad)
                    grad_U = torch.autograd.grad(U.sum(), x_t_grad, create_graph=False)[0]
                    force_total -= grad_U
                    x_t_grad.requires_grad_(False)  # noqa: FBT003

                # Fitness force (if enabled)
                if self.use_fitness_force:
                    if use_precomputed_gradients:
                        # Reuse fitness gradients from history (no gradient computation needed)
                        hist_idx = np.where(indices == step_idx)[0][0]
                        if hist_idx < history.fitness_gradients.shape[0]:
                            fitness_grad = history.fitness_gradients[hist_idx]
                            force_total -= self.epsilon_F * fitness_grad
                    elif self.fitness_op is not None and self.companion_selection is not None:
                        # Compute fitness gradient on-the-fly (requires gradients)
                        # This path is used when fitness force is enabled
                        # but not computed during simulation
                        pass  # Skip for now - would require full gradient computation

                force_vectors_np = force_total.detach().cpu().numpy()
                force_mag_np = np.linalg.norm(force_vectors_np, axis=1)

            force_vectors_series.append(force_vectors_np[alive_np])
            force_magnitudes_series.append(force_mag_np[alive_np])

        self.result = {
            "positions": positions,
            "V_total": V_total,
            "n_alive": alive,
            "times": times,
            "terminated": bool(history.terminated_early),
            "final_step": int(history.final_step),
            "velocity_series": velocity_series,
            "fitness_series": fitness_series,
            "distance_series": distance_series,
            "reward_series": reward_series,
            "hessian_series": hessian_series,
            "alive_masks": alive_masks,
            "previous_positions": previous_positions,
            "will_clone_series": will_clone_series,
            "force_vectors_series": force_vectors_series,
            "force_magnitudes_series": force_magnitudes_series,
        }

        # Update player
        frame_count = len(times)
        self.time_player.start = 0
        self.time_player.end = max(frame_count - 1, 0)
        self.time_player.value = 0
        self.time_player.disabled = frame_count <= 1
        self.time_player.name = f"time (stride {stride})"

        # Update status
        if frame_count:
            summary = (
                f"**Frames:** {frame_count} | "
                f"final V_total = {V_total[-1]:.4f} | alive = {int(alive[-1])}"
            )
        else:
            summary = "No frames available"
        if self.result["terminated"]:
            summary += " — terminated early"
        self.status_pane.object = summary

        self.frame_stream.event(frame=0)
        self._update_histogram_streams(0)

    def _get_frame_data(self, frame: int):
        """Get processed frame data for rendering."""
        if not self.result or not len(self.result["times"]):
            return None

        data = self.result
        max_frame = len(data["times"]) - 1
        frame = int(np.clip(frame, 0, max_frame))

        alive_mask = np.asarray(data["alive_masks"][frame], dtype=bool)
        positions_full = data["positions"][frame]
        prev_positions_full = data["previous_positions"][frame]
        was_cloned_full = data["will_clone_series"][frame]

        if alive_mask.any():
            positions = positions_full[alive_mask]
            if prev_positions_full is not None:
                prev_positions = prev_positions_full[alive_mask]
            else:
                prev_positions = None
            was_cloned = was_cloned_full[alive_mask]
            velocity_vals = np.asarray(data["velocity_series"][frame], dtype=float)
            fitness_vals = np.asarray(data["fitness_series"][frame], dtype=float)
            distance_vals = np.asarray(data["distance_series"][frame], dtype=float)
            reward_vals = np.asarray(data["reward_series"][frame], dtype=float)
            hessian_vals = np.asarray(data["hessian_series"][frame], dtype=float)
            force_vectors = np.asarray(data["force_vectors_series"][frame], dtype=float)
            force_magnitudes = np.asarray(data["force_magnitudes_series"][frame], dtype=float)
        else:
            positions = np.empty((0, positions_full.shape[1]))
            prev_positions = None
            was_cloned = np.asarray([], dtype=bool)
            velocity_vals = np.asarray([], dtype=float)
            fitness_vals = np.asarray([], dtype=float)
            distance_vals = np.asarray([], dtype=float)
            reward_vals = np.asarray([], dtype=float)
            hessian_vals = np.asarray([], dtype=float)
            force_vectors = np.empty((0, positions_full.shape[1]), dtype=float)
            force_magnitudes = np.asarray([], dtype=float)

        return {
            "frame": frame,
            "max_frame": max_frame,
            "positions": positions,
            "prev_positions": prev_positions,
            "was_cloned": was_cloned,
            "velocity_vals": velocity_vals,
            "fitness_vals": fitness_vals,
            "distance_vals": distance_vals,
            "reward_vals": reward_vals,
            "hessian_vals": hessian_vals,
            "force_vectors": force_vectors,
            "force_magnitudes": force_magnitudes,
            "data": data,
        }

    def _update_histogram_streams(self, frame: int):
        """Update histogram data streams and visibility for the given frame.

        Only streams data to visible histograms, avoiding computation overhead.

        Args:
            frame: Frame index to extract data from
        """
        frame_data = self._get_frame_data(frame)

        if frame_data is None:
            # No data available, hide all histograms
            for pane in self.histogram_panes.values():
                pane.visible = False
            return

        # Extract metric data
        fitness_vals = frame_data["fitness_vals"]
        distance_vals = frame_data["distance_vals"]
        reward_vals = frame_data["reward_vals"]
        hessian_vals = frame_data["hessian_vals"]
        force_vals = frame_data["force_magnitudes"]
        velocity_vals = frame_data["velocity_vals"]

        # Update each histogram: visibility + streaming using Shaolin .send()
        if "fitness" in self.enabled_histograms:
            self.histogram_panes["fitness"].visible = True
            self.histogram_fitness.send(fitness_vals)
        else:
            self.histogram_panes["fitness"].visible = False

        if "distance" in self.enabled_histograms:
            self.histogram_panes["distance"].visible = True
            self.histogram_distance.send(distance_vals)
        else:
            self.histogram_panes["distance"].visible = False

        if "reward" in self.enabled_histograms:
            self.histogram_panes["reward"].visible = True
            self.histogram_reward.send(reward_vals)
        else:
            self.histogram_panes["reward"].visible = False

        if "hessian" in self.enabled_histograms:
            self.histogram_panes["hessian"].visible = True
            self.histogram_hessian.send(hessian_vals)
        else:
            self.histogram_panes["hessian"].visible = False

        if "forces" in self.enabled_histograms:
            self.histogram_panes["forces"].visible = True
            self.histogram_forces.send(force_vals)
        else:
            self.histogram_panes["forces"].visible = False

        if "velocity" in self.enabled_histograms:
            self.histogram_panes["velocity"].visible = True
            self.histogram_velocity.send(velocity_vals)
        else:
            self.histogram_panes["velocity"].visible = False

    def _compute_vector_field(
        self,
        positions: np.ndarray,
        vectors: np.ndarray,
        grid_resolution: int,
        kernel_bandwidth: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute continuous vector field from particle data using kernel interpolation.

        Args:
            positions: Particle positions [N, 2]
            vectors: Vector values at each particle [N, 2]
            grid_resolution: Grid size (NxN)
            kernel_bandwidth: Gaussian kernel bandwidth

        Returns:
            X, Y, U, V: Grid coordinates and vector components
        """
        # Convert to torch tensors
        pos_tensor = torch.from_numpy(positions).to(dtype=torch.float32)
        vec_tensor = torch.from_numpy(vectors).to(dtype=torch.float32)

        # Use FluidFieldComputer
        X, Y, U, V = FluidFieldComputer.compute_velocity_field(
            positions=pos_tensor,
            velocities=vec_tensor,
            grid_resolution=grid_resolution,
            kernel_bandwidth=kernel_bandwidth,
            bounds=(-self.bounds_extent, self.bounds_extent),
        )

        return X, Y, U, V

    def _render_main_plot(self, frame: int):
        """Render the main scatter plot."""
        frame_data = self._get_frame_data(frame)

        if frame_data is None:
            return (self.background * self.mode_points).opts(
                title="Load a RunHistory to visualize the swarm",
                width=720,
                height=620,
            )

        positions = frame_data["positions"]
        prev_positions = frame_data["prev_positions"]
        was_cloned = frame_data["was_cloned"]
        velocity_vals = frame_data["velocity_vals"]
        fitness_vals = frame_data["fitness_vals"]
        distance_vals = frame_data["distance_vals"]
        reward_vals = frame_data["reward_vals"]
        force_vectors = frame_data["force_vectors"]
        force_magnitudes = frame_data["force_magnitudes"]
        data = frame_data["data"]
        frame_idx = frame_data["frame"]
        max_frame = frame_data["max_frame"]

        df = pd.DataFrame({
            "x₁": positions[:, 0] if positions.size else np.asarray([], dtype=float),
            "x₂": positions[:, 1] if positions.size else np.asarray([], dtype=float),
            "velocity": velocity_vals,
            "fitness": fitness_vals,
            "distance": distance_vals,
            "reward": reward_vals,
        })
        df["__size__"] = 8.0

        if self.size_metric != "constant" and not df.empty:
            size_values = df[self.size_metric].to_numpy(dtype=float)
            finite = np.isfinite(size_values)
            scaled = np.full_like(size_values, 8.0, dtype=float)
            if finite.any():
                vmin = size_values[finite].min()
                vmax = size_values[finite].max()
                if np.isclose(vmin, vmax):
                    scaled[finite] = 14.0
                else:
                    scaled[finite] = 6.0 + 24.0 * (size_values[finite] - vmin) / (vmax - vmin)
            df["__size__"] = scaled

        vdims = ["velocity", "fitness", "distance", "reward", "__size__"]
        points = hv.Points(df, kdims=["x₁", "x₂"], vdims=vdims).opts(
            size=dim("__size__"),
            marker="circle",
            alpha=0.75,
            line_color="white",
            line_width=0.5,
        )
        if self.color_metric != "constant" and not df.empty:
            points = points.opts(color=dim(self.color_metric), cmap="Viridis", colorbar=True)
        else:
            points = points.opts(color="navy", colorbar=False)

        # Build overlay
        overlay = self.background

        # Add velocity vectors
        if self.show_velocity_vectors and prev_positions is not None and len(positions) > 0:
            if self.color_vectors_by_cloning and len(was_cloned) > 0:
                diffusion_paths = []
                cloned_paths = []

                for i in range(len(positions)):
                    x1, y1 = positions[i]
                    x0, y0 = prev_positions[i]
                    path = [(x0, y0), (x1, y1)]

                    if was_cloned[i]:
                        cloned_paths.append(path)
                    else:
                        diffusion_paths.append(path)

                if len(diffusion_paths) > 0:
                    diffusion_arrows = hv.Path(diffusion_paths, kdims=["x₁", "x₂"]).opts(
                        color="cyan",
                        line_width=1.5,
                        alpha=0.7,
                    )
                    overlay *= diffusion_arrows

                if len(cloned_paths) > 0:
                    cloned_arrows = hv.Path(cloned_paths, kdims=["x₁", "x₂"]).opts(
                        color="#FFD700",
                        line_width=1.5,
                        alpha=0.7,
                    )
                    overlay *= cloned_arrows
            else:
                arrow_paths = []
                for i in range(len(positions)):
                    x1, y1 = positions[i]
                    x0, y0 = prev_positions[i]
                    arrow_paths.append([(x0, y0), (x1, y1)])

                if len(arrow_paths) > 0:
                    arrows = hv.Path(arrow_paths, kdims=["x₁", "x₂"]).opts(
                        color="cyan",
                        line_width=1.5,
                        alpha=0.7,
                    )
                    overlay *= arrows

        # Add force vectors
        if self.show_force_vectors and len(positions) > 0 and len(force_vectors) > 0:
            force_paths = []
            force_mags_for_color = []

            for i in range(len(positions)):
                force_mag = force_magnitudes[i]
                if force_mag > 1e-10:
                    force_norm = force_vectors[i] / force_mag
                else:
                    force_norm = np.zeros_like(force_vectors[i])

                arrow_end = positions[i] + force_norm * float(self.force_arrow_length)

                x0, y0 = positions[i]
                x1, y1 = arrow_end
                force_paths.append([(x0, y0), (x1, y1)])
                force_mags_for_color.append(force_mag)

            if len(force_paths) > 0:
                force_mags_array = np.array(force_mags_for_color)
                if force_mags_array.max() > 1e-10:
                    p5 = np.percentile(force_mags_array, 5)
                    p95 = np.percentile(force_mags_array, 95)
                    if p95 > p5:
                        force_intensity = np.clip((force_mags_array - p5) / (p95 - p5), 0, 1)
                    else:
                        force_intensity = np.ones_like(force_mags_array)
                else:
                    force_intensity = np.zeros_like(force_mags_array)

                colors = []
                for intensity in force_intensity:
                    lightness = 0.8 - 0.6 * intensity
                    if lightness > 0.5:
                        green_val = int(255 * (1 - (1 - lightness) * 2))
                    else:
                        green_val = 255
                    red_blue_val = (
                        int(255 * lightness * 2)
                        if lightness < 0.5
                        else int(255 * (1 - (lightness - 0.5) * 2))
                    )
                    color_hex = f"#{red_blue_val:02x}{green_val:02x}{red_blue_val:02x}"
                    colors.append(color_hex)

                for path, color in zip(force_paths, colors):
                    force_arrow = hv.Path([path], kdims=["x₁", "x₂"]).opts(
                        color=color,
                        line_width=2.0,
                        alpha=0.8,
                    )
                    overlay *= force_arrow

        # Add velocity field overlay (continuous field on grid)
        if self.show_velocity_field and len(positions) > 0:
            # Reconstruct velocity vectors from current and previous positions
            if prev_positions is not None:
                velocity_vectors = positions - prev_positions
            else:
                # For first frame or when prev unavailable, use zero velocities
                velocity_vectors = np.zeros_like(positions)

            try:
                X, Y, U, V = self._compute_vector_field(
                    positions=positions,
                    vectors=velocity_vectors,
                    grid_resolution=int(self.field_grid_resolution),
                    kernel_bandwidth=float(self.field_kernel_bandwidth),
                )

                # Create vector field data (x, y, u, v format)
                # Downsample for cleaner visualization
                stride = max(1, len(X) // 15)
                field_data = {
                    "x": X[::stride, ::stride].flatten(),
                    "y": Y[::stride, ::stride].flatten(),
                    "u": U[::stride, ::stride].flatten() * float(self.field_scale),
                    "v": V[::stride, ::stride].flatten() * float(self.field_scale),
                }

                velocity_field = hv.VectorField(
                    field_data, kdims=["x", "y"], vdims=["u", "v"]
                ).opts(
                    color="cyan",
                    alpha=0.6,
                    magnitude="Magnitude",
                    pivot="mid",
                    arrow_heads=True,
                    line_width=1.5,
                )
                overlay *= velocity_field
            except Exception:
                # Silently skip if field computation fails
                pass

        # Add force field overlay (continuous field on grid)
        if self.show_force_field and len(positions) > 0 and len(force_vectors) > 0:
            try:
                X, Y, U, V = self._compute_vector_field(
                    positions=positions,
                    vectors=force_vectors,
                    grid_resolution=int(self.field_grid_resolution),
                    kernel_bandwidth=float(self.field_kernel_bandwidth),
                )

                # Create vector field data
                stride = max(1, len(X) // 15)
                field_data = {
                    "x": X[::stride, ::stride].flatten(),
                    "y": Y[::stride, ::stride].flatten(),
                    "u": U[::stride, ::stride].flatten() * float(self.field_scale),
                    "v": V[::stride, ::stride].flatten() * float(self.field_scale),
                }

                force_field = hv.VectorField(field_data, kdims=["x", "y"], vdims=["u", "v"]).opts(
                    color="orange",
                    alpha=0.5,
                    magnitude="Magnitude",
                    pivot="mid",
                    arrow_heads=True,
                    line_width=1.8,
                )
                overlay *= force_field
            except Exception:
                # Silently skip if field computation fails
                pass

        overlay = overlay * points * self.mode_points

        text_lines = [
            f"t = {int(data['times'][frame_idx])}",
            f"V_total = {data['V_total'][frame_idx]:.4f}",
            f"Alive = {int(data['n_alive'][frame_idx])}",
        ]
        if data["terminated"] and frame_idx == max_frame:
            text_lines.append("⛔ terminated early")

        metrics_text = hv.Text(
            -self.bounds_extent + 0.3,
            self.bounds_extent - 0.4,
            "\n".join(text_lines),
        ).opts(text_font_size="12pt", text_align="left")

        return (overlay * metrics_text).opts(
            framewise=True,
            xlim=(-self.bounds_extent, self.bounds_extent),
            ylim=(-self.bounds_extent, self.bounds_extent),
            width=720,
            height=620,
            title="Euclidean Gas Swarm Evolution",
            show_grid=True,
            shared_axes=False,
        )

    def panel(self) -> pn.Column:
        """Create Panel dashboard for visualization.

        Returns:
            Panel Column with visualization and playback controls
        """
        # Create custom multitoggle widget for histograms
        histogram_toggle = pn.widgets.CheckButtonGroup(
            name="Enabled Histograms",
            options=["fitness", "distance", "reward", "hessian", "forces", "velocity"],
            value=list(self.enabled_histograms),
            button_type="primary",
            button_style="outline",
            sizing_mode="stretch_width",
        )

        # Link widget to parameter
        def update_histograms(event):
            self.enabled_histograms = event.new

        histogram_toggle.param.watch(update_histograms, "value")

        display_controls = pn.Column(
            pn.pane.Markdown("### Display Options"),
            pn.Param(
                self.param,
                parameters=[
                    "color_metric",
                    "size_metric",
                    "show_velocity_vectors",
                    "color_vectors_by_cloning",
                    "show_force_vectors",
                    "force_arrow_length",
                    "show_velocity_field",
                    "show_force_field",
                    "field_grid_resolution",
                    "field_kernel_bandwidth",
                    "field_scale",
                    "measure_stride",
                ],
                show_name=False,
                sizing_mode="stretch_width",
            ),
            histogram_toggle,
            pn.pane.Markdown("### Playback"),
            self.time_player,
            self.status_pane,
            sizing_mode="stretch_width",
            min_width=300,
        )

        # Create histogram grid with all 6 panes (3x2 layout)
        hist_grid = pn.GridSpec(sizing_mode="stretch_width", height=500)
        hist_grid[0, 0] = self.histogram_panes["fitness"]
        hist_grid[0, 1] = self.histogram_panes["distance"]
        hist_grid[0, 2] = self.histogram_panes["reward"]
        hist_grid[1, 0] = self.histogram_panes["hessian"]
        hist_grid[1, 1] = self.histogram_panes["forces"]
        hist_grid[1, 2] = self.histogram_panes["velocity"]

        viz_column = pn.Column(
            pn.panel(self.dmap_main.opts(framewise=True)),
            hist_grid,
            sizing_mode="stretch_width",
        )

        return pn.Row(
            display_controls,
            viz_column,
            sizing_mode="stretch_width",
        )
