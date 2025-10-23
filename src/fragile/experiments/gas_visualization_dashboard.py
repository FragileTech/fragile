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
from scipy.stats import gaussian_kde
import torch

from fragile.bounds import TorchBounds
from fragile.core.benchmarks import prepare_benchmark_for_explorer
from fragile.core.companion_selection import CompanionSelection
from fragile.core.history import RunHistory
from fragile.experiments.fluid_utils import FluidFieldComputer
from fragile.experiments.gas_config_dashboard import GasConfig


__all__ = ["GasVisualizer", "ConvergencePanel"]


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
                "enabled_histograms",
                "show_velocity_field",
                "show_force_field",
                "field_grid_resolution",
                "field_kernel_bandwidth",
                "field_scale",
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
                    # Note: history stores [n_recorded, N, d], we need to map step_idx to history index
                    # The step_idx is an index into the downsampled indices array
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
                    # Hessian requested but can't compute (no precomputed data or fitness not computed)
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

    def _render_histograms(self, frame: int):
        """Render enabled histograms in dynamic grid layout.

        Only computes and displays histograms selected in enabled_histograms.
        Layout automatically reflows based on enabled histograms.
        """
        frame_data = self._get_frame_data(frame)

        if frame_data is None:
            return hv.Text(0, 0, "No data available", fontsize=14).opts(width=720, height=200)

        # Extract all metric values
        fitness_vals = frame_data["fitness_vals"]
        distance_vals = frame_data["distance_vals"]
        reward_vals = frame_data["reward_vals"]
        hessian_vals = frame_data["hessian_vals"]
        force_vals = frame_data["force_magnitudes"]
        velocity_vals = frame_data["velocity_vals"]

        # Build list of enabled histograms
        histogram_list = []

        if "fitness" in self.enabled_histograms:
            histogram_list.append(self._make_histogram(fitness_vals, "Fitness", "#1f77b4"))
        if "distance" in self.enabled_histograms:
            histogram_list.append(self._make_histogram(distance_vals, "Distance", "#2ca02c"))
        if "reward" in self.enabled_histograms:
            histogram_list.append(self._make_histogram(reward_vals, "Reward", "#d62728"))
        if "hessian" in self.enabled_histograms:
            histogram_list.append(self._make_histogram(hessian_vals, "Hessian", "#8c564b"))
        if "forces" in self.enabled_histograms:
            histogram_list.append(self._make_histogram(force_vals, "Forces", "#ff7f0e"))
        if "velocity" in self.enabled_histograms:
            histogram_list.append(self._make_histogram(velocity_vals, "Velocity", "#9467bd"))

        if not histogram_list:
            return hv.Text(
                0, 0, "No histograms enabled. Select metrics to display.", fontsize=14
            ).opts(width=720, height=200)

        # Create dynamic layout with 3 columns (automatically reflows)
        return hv.Layout(histogram_list).opts(opts.Layout(shared_axes=False)).cols(3)

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

        # Plot panes (populated after computation)
        self.plot_pane = pn.pane.Markdown(
            "**Click 'Compute Convergence Metrics' to generate plots**",
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
        self.plot_pane.object = "**Click 'Compute Convergence Metrics' to generate plots**"

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
        has_optimum = hasattr(self.benchmark, "best_state") and self.benchmark.best_state is not None

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
                dist = np.linalg.norm(mean_pos - target_opt[:history.d])
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

        # Layout plots in grid
        if plots:
            layout = hv.Layout(plots).opts(opts.Layout(shared_axes=False)).cols(2)
            self.plot_pane.object = layout
        else:
            self.plot_pane.object = "**No valid convergence data to plot**"

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

        overlay = (scatter * fit_curve).opts(
            width=400,
            height=350,
            title=title,
            xlabel="Time",
            ylabel="KL(empirical || target)",
            logy=True,
            show_grid=True,
            framewise=True,
        )

        return overlay

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

        overlay = (scatter * fit_curve).opts(
            width=400,
            height=350,
            title=title,
            xlabel="Time",
            ylabel="V_total = Var[x] + Var[v]",
            logy=True,
            show_grid=True,
            framewise=True,
        )

        return overlay

    def _plot_variances(self, times: np.ndarray, var_x: np.ndarray, var_v: np.ndarray):
        """Plot position and velocity variances vs time."""
        valid_mask_x = np.isfinite(var_x) & (var_x > 0)
        valid_mask_v = np.isfinite(var_v) & (var_v > 0)

        if not (valid_mask_x.any() or valid_mask_v.any()):
            return None

        curves = []
        if valid_mask_x.any():
            curve_x = hv.Curve(
                (times[valid_mask_x], var_x[valid_mask_x]), kdims=["time"], vdims=["Variance"], label="Var[x]"
            ).opts(color="blue", line_width=2)
            curves.append(curve_x)

        if valid_mask_v.any():
            curve_v = hv.Curve(
                (times[valid_mask_v], var_v[valid_mask_v]), kdims=["time"], vdims=["Variance"], label="Var[v]"
            ).opts(color="orange", line_width=2)
            curves.append(curve_v)

        if not curves:
            return None

        overlay = hv.Overlay(curves).opts(
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

        return overlay

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

        plot = curve.opts(
            width=400,
            height=350,
            title="Distance to Known Optimum",
            xlabel="Time",
            ylabel="||mean(x) - x*||",
            show_grid=True,
            framewise=True,
        )

        return plot

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
            kappa_kl, C_kl, r_sq_kl = kl_fit
            half_life_kl = np.log(2) / kappa_kl if kappa_kl > 0 else float("inf")
            stats.append(("KL Convergence Rate κ", f"{kappa_kl:.5f}"))
            stats.append(("KL Half-Life t₁/₂", f"{half_life_kl:.2f}"))
            stats.append(("KL Fit R²", f"{r_sq_kl:.4f}"))

        final_kl = kl[np.isfinite(kl)][-1] if len(kl[np.isfinite(kl)]) > 0 else float("nan")
        stats.append(("Final KL Divergence", f"{final_kl:.4f}"))

        # Lyapunov statistics
        lyap_fit = self._fit_exponential_decay(times, lyap)
        if lyap_fit is not None:
            kappa_lyap, C_lyap, r_sq_lyap = lyap_fit
            stats.append(("Lyapunov Decay Rate κ", f"{kappa_lyap:.5f}"))
            stats.append(("Lyapunov Fit R²", f"{r_sq_lyap:.4f}"))

        final_lyap = lyap[np.isfinite(lyap)][-1] if len(lyap[np.isfinite(lyap)]) > 0 else float("nan")
        stats.append(("Final V_total", f"{final_lyap:.4f}"))

        # Variance statistics
        final_var_x = var_x[np.isfinite(var_x)][-1] if len(var_x[np.isfinite(var_x)]) > 0 else float("nan")
        final_var_v = var_v[np.isfinite(var_v)][-1] if len(var_v[np.isfinite(var_v)]) > 0 else float("nan")
        stats.append(("Final Var[x]", f"{final_var_x:.4f}"))
        stats.append(("Final Var[v]", f"{final_var_v:.4f}"))

        # Distance to optimum
        if self.metrics["has_optimum"] and self.metrics["distance_to_opt"] is not None:
            dist = self.metrics["distance_to_opt"]
            final_dist = dist[np.isfinite(dist)][-1] if len(dist[np.isfinite(dist)]) > 0 else float("nan")
            stats.append(("Final Distance to Optimum", f"{final_dist:.4f}"))

        # Create table
        df = pd.DataFrame(stats, columns=["Metric", "Value"])
        table = hv.Table(df).opts(
            width=800,
            height=400,
            title="Convergence Summary Statistics",
        )

        return table

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
        pbc=gas_config.pbc,
    )

    # Create convergence panel (initially with no history)
    convergence_panel = ConvergencePanel(
        history=None,
        potential=potential,
        benchmark=potential,  # Pass benchmark for best_state access
        bounds_extent=bounds_extent,
    )

    # Connect simulation completion to visualizer and convergence panel
    def on_simulation_complete(history):
        """Update visualizer and convergence panel when simulation completes."""
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
            convergence_panel.set_history(history)  # Also update convergence panel

        import threading

        thread = threading.Thread(target=_update_history, daemon=True)
        thread.start()

    gas_config.add_completion_callback(on_simulation_complete)

    def on_benchmark_change(potential, background, mode_points):
        """Update visualizer and convergence panel when benchmark changes."""
        visualizer.update_benchmark(potential, background, mode_points)
        convergence_panel.update_benchmark(potential)  # Update convergence benchmark

    gas_config.add_benchmark_callback(on_benchmark_change)

    # Create tabbed layout
    tabs = pn.Tabs(
        (
            "Evolution",
            pn.Column(
                pn.pane.Markdown("## Swarm Evolution Visualization"),
                visualizer.panel(),
                sizing_mode="stretch_width",
            ),
        ),
        (
            "Convergence",
            pn.Column(
                pn.pane.Markdown("## Convergence Analysis"),
                convergence_panel.panel(),
                sizing_mode="stretch_width",
            ),
        ),
        dynamic=True,
        sizing_mode="stretch_width",
    )

    # Create layout using FastListTemplate
    return pn.template.FastListTemplate(
        title="Gas Visualization Dashboard",
        sidebar=[
            pn.pane.Markdown("## Simulation Control"),
            gas_config.panel(),
        ],
        main=[tabs],
        sidebar_width=400,
        main_max_width="100%",
    )


if __name__ == "__main__":
    # Create and serve the app without opening browser
    app = create_app()
    app.show(port=5007, open=False)
    print("Gas Visualization Dashboard running at http://localhost:5007")
