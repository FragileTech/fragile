"""Visualization dashboard for exploring RunHistory from EuclideanGas simulations.

This module provides a Panel-based dashboard for visualizing and animating
RunHistory objects from EuclideanGas runs. It can be used with any RunHistory
source (live simulation, loaded from disk, etc.).

Can also be run as a standalone script to spawn an interactive dashboard:
    python -m fragile.experiments.gas_visualization_dashboard
"""

from __future__ import annotations

from typing import Sequence

import holoviews as hv
from holoviews import dim, opts
import numpy as np
import pandas as pd
import panel as pn
import param
import torch

from fragile.bounds import TorchBounds
from fragile.core.companion_selection import CompanionSelection
from fragile.core.benchmarks import prepare_benchmark_for_explorer
from fragile.core.history import RunHistory
from fragile.experiments.gas_config_dashboard import GasConfig


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
        self.dmap_hists = hv.DynamicMap(self._render_histograms, streams=[self.frame_stream])

        # Status display
        self.status_pane = pn.pane.Markdown("", sizing_mode="stretch_width")

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
            ],
        )

        # Load initial history if provided
        if history is not None:
            self.set_history(history)

    def set_history(self, history: RunHistory):
        """Load a new RunHistory for visualization.

        Args:
            history: RunHistory object to visualize
        """
        self.history = history
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

    def _refresh_frame(self, *_):
        """Refresh current frame display."""
        if not self.result:
            return
        self.frame_stream.event(frame=self.time_player.value)

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

            # Compute fitness and distances if we have companion selection
            if (
                alive_np.any()
                and self.companion_selection is not None
                and self.fitness_op is not None
            ):
                with torch.no_grad():
                    rewards = -self.potential(x_t)
                    companions = self.companion_selection(x=x_t, v=v_t, alive_mask=alive_mask)

                    # Use FitnessOperator to compute fitness
                    fitness_vals, info = self.fitness_op(
                        positions=x_t,
                        velocities=v_t,
                        rewards=rewards,
                        alive=alive_mask,
                        companions=companions,
                    )
                    distances = info["distances"]

                    rewards_np = rewards.detach().cpu().numpy()
                    fitness_np = fitness_vals.detach().cpu().numpy()
                    distances_np = distances.detach().cpu().numpy()

                # Get Hessian diagonal: reuse from history if available, otherwise compute
                if use_precomputed_hessians:
                    # Reuse Hessians computed during simulation (no gradient computation needed)
                    # Note: history stores [n_recorded, N, d], we need to map step_idx to history index
                    # The step_idx is an index into the downsampled indices array
                    hist_idx = np.where(indices == step_idx)[0][0]
                    if hist_idx < history.fitness_hessians_diag.shape[0]:
                        hessian_diag_t = history.fitness_hessians_diag[hist_idx]
                        hessian_mag = torch.linalg.norm(hessian_diag_t, dim=1)
                        hessian_np = hessian_mag.detach().cpu().numpy()
                    else:
                        hessian_np = np.zeros(x_t.shape[0], dtype=np.float32)
                elif self.fitness_op is not None:
                    # Compute Hessian diagonal on-the-fly (requires gradients)
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
                    hessian_np = np.zeros(x_t.shape[0], dtype=np.float32)
            else:
                rewards_np = np.zeros(x_t.shape[0], dtype=np.float32)
                fitness_np = np.zeros(x_t.shape[0], dtype=np.float32)
                distances_np = np.zeros(x_t.shape[0], dtype=np.float32)
                hessian_np = np.zeros(x_t.shape[0], dtype=np.float32)

            velocity_series.append(vel_mag[alive_np])
            fitness_series.append(fitness_np[alive_np])
            distance_series.append(distances_np[alive_np])
            reward_series.append(rewards_np[alive_np])
            hessian_series.append(hessian_np[alive_np])

            # Compute force vectors
            force_vectors_np = np.zeros((x_t.shape[0], x_t.shape[1]), dtype=np.float32)
            force_mag_np = np.zeros(x_t.shape[0], dtype=np.float32)

            if alive_np.any() and (self.use_potential_force or self.use_fitness_force):
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
                        # This path is used when fitness force is enabled but not computed during simulation
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

    def _make_histogram(self, values: Sequence[float], label: str, color: str) -> hv.Histogram:
        """Create histogram visualization."""
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        if array.size == 0:
            return hv.Histogram([]).opts(
                width=220,
                height=220,
                title=f"{label} Distribution",
                xlabel=label,
                ylabel="density",
                show_grid=True,
                shared_axes=False,
                framewise=True,
            )

        counts, edges = np.histogram(array, bins=30, density=True)
        return hv.Histogram((edges, np.nan_to_num(counts)), label=label).opts(
            width=220,
            height=220,
            title=f"{label} Distribution",
            xlabel=label,
            ylabel="density",
            show_grid=True,
            color=color,
            line_color=color,
            alpha=0.6,
            shared_axes=False,
            framewise=True,
        )

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

    def _render_histograms(self, frame: int):
        """Render 6 histograms in 2×3 grid layout.

        Row 1: Fitness, Distance, Reward
        Row 2: Hessian, Forces, Velocities
        """
        frame_data = self._get_frame_data(frame)

        if frame_data is None:
            # Create empty histograms for all 6 metrics
            fitness_hist = self._make_histogram([], "Fitness", "#1f77b4")
            distance_hist = self._make_histogram([], "Distance", "#2ca02c")
            reward_hist = self._make_histogram([], "Reward", "#d62728")
            hessian_hist = self._make_histogram([], "Hessian", "#8c564b")
            force_hist = self._make_histogram([], "Forces", "#ff7f0e")
            velocity_hist = self._make_histogram([], "Velocity", "#9467bd")
        else:
            fitness_vals = frame_data["fitness_vals"]
            distance_vals = frame_data["distance_vals"]
            reward_vals = frame_data["reward_vals"]
            hessian_vals = frame_data["hessian_vals"]
            force_vals = frame_data["force_magnitudes"]
            velocity_vals = frame_data["velocity_vals"]

            fitness_hist = self._make_histogram(fitness_vals, "Fitness", "#1f77b4")
            distance_hist = self._make_histogram(distance_vals, "Distance", "#2ca02c")
            reward_hist = self._make_histogram(reward_vals, "Reward", "#d62728")
            hessian_hist = self._make_histogram(hessian_vals, "Hessian", "#8c564b")
            force_hist = self._make_histogram(force_vals, "Forces", "#ff7f0e")
            velocity_hist = self._make_histogram(velocity_vals, "Velocity", "#9467bd")

        # Create 2×3 grid layout
        row1 = fitness_hist + distance_hist + reward_hist
        row2 = hessian_hist + force_hist + velocity_hist

        return (row1 + row2).opts(opts.Layout(shared_axes=False)).cols(3)

    def panel(self) -> pn.Column:
        """Create Panel dashboard for visualization.

        Returns:
            Panel Column with visualization and playback controls
        """
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
                    "measure_stride",
                ],
                show_name=False,
                sizing_mode="stretch_width",
            ),
            pn.pane.Markdown("### Playback"),
            self.time_player,
            self.status_pane,
            sizing_mode="stretch_width",
            min_width=300,
        )

        viz_column = pn.Column(
            pn.panel(self.dmap_main.opts(framewise=True)),
            pn.panel(self.dmap_hists),
            sizing_mode="stretch_width",
        )

        return pn.Row(
            display_controls,
            viz_column,
            sizing_mode="stretch_width",
        )


def create_app(dims: int = 2, n_gaussians: int = 3, bounds_extent: float = 6.0):
    """Create the Panel app for standalone usage.

    Args:
        dims: Spatial dimension
        n_gaussians: Number of Gaussian modes in potential
        bounds_extent: Spatial bounds half-width

    Returns:
        Panel template ready to serve
    """
    # Initialize extensions
    hv.extension("bokeh")
    pn.extension()

    # Prepare potential and background
    potential, background, mode_points = prepare_benchmark_for_explorer(
        benchmark_name="Mixture of Gaussians",
        dims=dims,
        bounds_range=(-bounds_extent, bounds_extent),
        resolution=100,
        n_gaussians=n_gaussians,
    )

    # Create gas configuration dashboard
    gas_config = GasConfig(potential=potential, dims=dims)

    # Create visualizer (initially with no history)
    visualizer = GasVisualizer(
        history=None,
        potential=potential,
        background=background,
        mode_points=mode_points,
        bounds_extent=bounds_extent,
    )

    # Connect simulation completion to visualizer
    def on_simulation_complete(history):
        """Update visualizer when simulation completes."""
        # Extract parameters from gas_config for force visualization
        visualizer.epsilon_F = float(gas_config.epsilon_F)
        visualizer.use_fitness_force = bool(gas_config.use_fitness_force)
        visualizer.use_potential_force = bool(gas_config.use_potential_force)

        # Set companion selection and fitness params from gas_config
        visualizer.companion_selection = gas_config.companion_selection
        visualizer.fitness_op = gas_config.fitness_op

        # Load the history in a separate thread to avoid blocking UI
        # This is critical: _process_history() contains expensive computations
        # that would freeze the UI if run synchronously
        def _update_history():
            visualizer.set_history(history)

        import threading

        thread = threading.Thread(target=_update_history, daemon=True)
        thread.start()

    gas_config.add_completion_callback(on_simulation_complete)

    # Create layout using FastListTemplate
    return pn.template.FastListTemplate(
        title="Gas Visualization Dashboard",
        sidebar=[
            pn.pane.Markdown("## Simulation Control"),
            gas_config.panel(),
        ],
        main=[
            pn.pane.Markdown("## Swarm Evolution Visualization"),
            visualizer.panel(),
        ],
        sidebar_width=400,
        main_max_width="100%",
    )


if __name__ == "__main__":
    # Create and serve the app without opening browser
    app = create_app()
    app.show(port=5007, open=False)
