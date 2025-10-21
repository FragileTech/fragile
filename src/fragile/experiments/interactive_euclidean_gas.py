from __future__ import annotations

from typing import Iterable, Sequence

import holoviews as hv
from holoviews import dim, opts
import numpy as np
import pandas as pd
import panel as pn
import panel.widgets as pnw
import param
import torch

from fragile.bounds import TorchBounds
from fragile.core.companion_selection import CompanionSelection
from fragile.core.euclidean_gas import (
    CloningParams,
    EuclideanGas,
    SimpleQuadraticPotential,
    SwarmState,
)
from fragile.core.fitness import FitnessOperator, FitnessParams, compute_fitness
from fragile.core.history import RunHistory
from fragile.core.kinetic_operator import KineticOperator, LangevinParams
from fragile.experiments.convergence_analysis import create_multimodal_potential


__all__ = [
    "SwarmExplorer",
    "create_dashboard",
    "prepare_background",
]


def prepare_background(
    dims: int = 2,
    n_gaussians: int = 3,
    bounds_range: tuple[float, float] = (-6.0, 6.0),
    seed: int = 42,
    resolution: int = 200,
) -> tuple[object, hv.Image, hv.Points]:
    """Pre-compute potential, density backdrop, and mode markers for the explorer."""
    potential, target_mixture = create_multimodal_potential(
        dims=dims,
        n_gaussians=n_gaussians,
        bounds_range=bounds_range,
        seed=seed,
    )

    grid_axis = np.linspace(bounds_range[0], bounds_range[1], resolution)
    X, Y = np.meshgrid(grid_axis, grid_axis)
    grid_points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)

    with torch.no_grad():
        U_grid = potential.evaluate(grid_points).cpu().numpy().reshape(X.shape)

    beta_bg = 1.0
    density = np.exp(-beta_bg * U_grid)
    density /= np.max(density)

    background = hv.Image(
        (grid_axis, grid_axis, density),
        kdims=["x₁", "x₂"],
        vdims="density",
    ).opts(
        cmap="Greys",
        alpha=0.35,
        colorbar=False,
        width=720,
        height=620,
    )

    mode_df = pd.DataFrame({
        "x₁": target_mixture.centers[:, 0].cpu().numpy(),
        "x₂": target_mixture.centers[:, 1].cpu().numpy(),
        "size": 50 * target_mixture.weights.cpu().numpy(),
    })

    mode_points = hv.Points(
        mode_df,
        kdims=["x₁", "x₂"],
        vdims="size",
        label="Target Modes",
    ).opts(
        size="size",
        color="red",
        marker="star",
        line_color="white",
        line_width=2,
        alpha=0.8,
    )

    return potential, background, mode_points


class SwarmExplorer(param.Parameterized):
    """Interactive Euclidean Gas explorer with Panel/HoloViews widgets."""

    # Simulation controls
    N = param.Integer(default=160, bounds=(10, 1000), doc="Number of walkers")
    n_steps = param.Integer(default=240, bounds=(50, 1000), doc="Simulation steps")
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

    # Langevin parameters
    gamma = param.Number(default=1.0, bounds=(0.05, 5.0), doc="Friction γ")
    beta = param.Number(default=1.0, bounds=(0.01, 10.0), doc="Inverse temperature β")
    delta_t = param.Number(default=0.05, bounds=(0.01, 0.2), doc="Time step Δt")
    epsilon_F = param.Number(default=0.0, bounds=(0.0, 0.5), doc="Fitness force rate ε_F")
    use_fitness_force = param.Boolean(default=False, doc="Enable fitness-driven force")
    use_potential_force = param.Boolean(default=False, doc="Enable potential force")
    epsilon_Sigma = param.Number(default=0.1, bounds=(0.0, 1.0), doc="Hessian regularisation ε_Σ")
    use_anisotropic_diffusion = param.Boolean(default=False, doc="Enable anisotropic diffusion")
    diagonal_diffusion = param.Boolean(default=True, doc="Use diagonal diffusion tensor")

    # Cloning parameters
    sigma_x = param.Number(default=0.15, bounds=(0.01, 1.0), doc="Cloning jitter σ_x")
    lambda_alg = param.Number(
        default=0.5, bounds=(0.0, 3.0), doc="Algorithmic distance weight λ_alg"
    )
    alpha_restitution = param.Number(default=0.6, bounds=(0.0, 1.0), doc="Restitution α_rest")
    alpha_fit = param.Number(default=1.0, bounds=(0.1, 3.0), doc="Reward exponent α")
    beta_fit = param.Number(default=1.0, bounds=(0.1, 3.0), doc="Diversity exponent β")
    eta = param.Number(default=0.1, bounds=(0.01, 0.5), doc="Positivity floor η")
    A = param.Number(default=2.0, bounds=(0.5, 5.0), doc="Logistic rescale amplitude A")
    sigma_min = param.Number(default=1e-8, bounds=(1e-9, 1e-3), doc="Standardisation σ_min")
    p_max = param.Number(default=1.0, bounds=(0.2, 10.0), doc="Maximum cloning probability p_max")
    epsilon_clone = param.Number(default=0.005, bounds=(1e-4, 0.05), doc="Cloning score ε_clone")
    companion_method = param.ObjectSelector(
        default="uniform",
        objects=("uniform", "softmax", "cloning", "random_pairing"),
        doc="Companion selection method",
    )
    companion_epsilon = param.Number(default=0.5, bounds=(0.01, 5.0), doc="Companion ε")
    integrator = param.ObjectSelector(default="baoab", objects=("baoab",), doc="Integrator")

    # Algorithm control
    enable_cloning = param.Boolean(default=True, doc="Enable cloning operator")
    enable_kinetic = param.Boolean(default=True, doc="Enable kinetic (Langevin) operator")

    # Initialisation controls
    init_offset = param.Number(default=4.5, bounds=(-6.0, 6.0), doc="Initial position offset")
    init_spread = param.Number(default=0.5, bounds=(0.1, 3.0), doc="Initial position spread")
    init_velocity_scale = param.Number(
        default=0.1, bounds=(0.01, 0.8), doc="Initial velocity scale"
    )
    bounds_extent = param.Number(default=6.0, bounds=(1, 12), doc="Spatial bounds half-width")

    auto_update = param.Boolean(default=False, doc="Auto recompute on parameter change")
    show_velocity_vectors = param.Boolean(
        default=False, doc="Display velocity vectors showing trajectory from previous to current position"
    )
    color_vectors_by_cloning = param.Boolean(
        default=False, doc="Color velocity vectors yellow if walker was created by cloning (requires show_velocity_vectors)"
    )
    show_force_vectors = param.Boolean(
        default=False, doc="Display force vectors F = -∇U - ε_F·∇V_fit at current positions"
    )
    force_arrow_length = param.Number(
        default=0.5, bounds=(0.1, 2.0), doc="Length scale for normalized force arrows"
    )

    def __init__(
        self,
        potential: object,
        background: hv.Image,
        mode_points: hv.Points,
        dims: int = 2,
        **params,
    ):
        super().__init__(**params)
        self.potential = potential
        self.background = background
        self.mode_points = mode_points
        self.dims = dims

        self.result: dict | None = None

        self.run_button = pn.widgets.Button(name="Recompute Simulation", button_type="primary")
        self.run_button.sizing_mode = "stretch_width"
        self.run_button.on_click(self._compute_simulation)

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

        self.frame_stream = hv.streams.Stream.define("Frame", frame=0)()
        self.dmap_main = hv.DynamicMap(self._render_main_plot, streams=[self.frame_stream])
        self.dmap_hists = hv.DynamicMap(self._render_histograms, streams=[self.frame_stream])

        self._control_params = [
            "N",
            "n_steps",
            "measure_stride",
            "gamma",
            "beta",
            "delta_t",
            "epsilon_F",
            "use_fitness_force",
            "use_potential_force",
            "epsilon_Sigma",
            "use_anisotropic_diffusion",
            "diagonal_diffusion",
            "sigma_x",
            "lambda_alg",
            "alpha_restitution",
            "alpha_fit",
            "beta_fit",
            "eta",
            "A",
            "sigma_min",
            "p_max",
            "epsilon_clone",
            "companion_method",
            "companion_epsilon",
            "integrator",
            "enable_cloning",
            "enable_kinetic",
            "init_offset",
            "init_spread",
            "init_velocity_scale",
            "bounds_extent",
            "auto_update",
            "show_velocity_vectors",
            "color_vectors_by_cloning",
            "show_force_vectors",
            "force_arrow_length",
            "color_metric",
            "size_metric",
        ]

        self._widget_overrides: dict[str, pn.widgets.Widget] = {
            "sigma_min": pnw.FloatSlider(
                name="sigma_min", start=1e-9, end=1e-3, value=self.sigma_min, step=1e-9
            ),
            "epsilon_clone": pnw.FloatSlider(
                name="epsilon_clone", start=1e-4, end=0.05, value=self.epsilon_clone, step=1e-4
            ),
            "gamma": pnw.FloatSlider(name="gamma", start=0.05, end=5.0, step=0.05),
            "beta": pnw.FloatSlider(name="beta", start=0.1, end=5.0, step=0.05),
            "delta_t": pnw.FloatSlider(name="delta_t", start=0.01, end=0.2, step=0.005),
            "lambda_alg": pnw.FloatSlider(name="lambda_alg", start=0.0, end=3.0, step=0.1),
        }

        self.status_pane = pn.pane.Markdown("", sizing_mode="stretch_width")

        self.param.watch(self._refresh_frame, ["color_metric", "size_metric", "show_velocity_vectors", "color_vectors_by_cloning", "show_force_vectors", "force_arrow_length"])

        self._compute_simulation()

    @param.depends(
        "N",
        "n_steps",
        "measure_stride",
        "gamma",
        "beta",
        "delta_t",
        "epsilon_F",
        "use_fitness_force",
        "use_potential_force",
        "epsilon_Sigma",
        "use_anisotropic_diffusion",
        "diagonal_diffusion",
        "sigma_x",
        "lambda_alg",
        "alpha_restitution",
        "alpha_fit",
        "beta_fit",
        "eta",
        "A",
        "sigma_min",
        "p_max",
        "epsilon_clone",
        "companion_method",
        "companion_epsilon",
        "integrator",
        "enable_cloning",
        "enable_kinetic",
        "init_offset",
        "init_spread",
        "init_velocity_scale",
        "bounds_extent",
        "auto_update",
        #watch=True,
    )
    def _auto_recompute(self, *_):
        if self.auto_update:
            self._compute_simulation()

    def _sync_stream(self, event):
        if not self.result:
            return
        max_frame = len(self.result["times"]) - 1
        frame = int(np.clip(event.new, 0, max_frame)) if max_frame >= 0 else 0
        self.frame_stream.event(frame=frame)

    def _refresh_frame(self, *_):
        if not self.result:
            return
        self.frame_stream.event(frame=self.time_player.value)

    def _compute_simulation(self, *_):
        dims = self.dims
        stride = max(1, int(self.measure_stride))

        bounds_extent = float(self.bounds_extent)
        low = torch.full((dims,), -bounds_extent, dtype=torch.float32)
        high = torch.full((dims,), bounds_extent, dtype=torch.float32)
        bounds = TorchBounds(low=low, high=high)

        companion_selection = CompanionSelection(
            method=self.companion_method,
            epsilon=float(self.companion_epsilon),
            lambda_alg=float(self.lambda_alg),
        )

        langevin_params = LangevinParams(
            gamma=float(self.gamma),
            beta=float(self.beta),
            delta_t=float(self.delta_t),
            integrator=self.integrator,
            epsilon_F=float(self.epsilon_F),
            use_fitness_force=bool(self.use_fitness_force),
            use_potential_force=bool(self.use_potential_force),
            epsilon_Sigma=float(self.epsilon_Sigma),
            use_anisotropic_diffusion=bool(self.use_anisotropic_diffusion),
            diagonal_diffusion=bool(self.diagonal_diffusion),
        )

        cloning_params = CloningParams(
            sigma_x=float(self.sigma_x),
            lambda_alg=float(self.lambda_alg),
            alpha_restitution=float(self.alpha_restitution),
            alpha=float(self.alpha_fit),
            beta=float(self.beta_fit),
            eta=float(self.eta),
            A=float(self.A),
            sigma_min=float(self.sigma_min),
            p_max=float(self.p_max),
            epsilon_clone=float(self.epsilon_clone),
            companion_selection=companion_selection,
        )

        # Create KineticOperator
        kinetic_op = KineticOperator(
            gamma=langevin_params.gamma,
            beta=langevin_params.beta,
            delta_t=langevin_params.delta_t,
            integrator=langevin_params.integrator,
            epsilon_F=langevin_params.epsilon_F,
            use_fitness_force=langevin_params.use_fitness_force,
            use_potential_force=langevin_params.use_potential_force,
            epsilon_Sigma=langevin_params.epsilon_Sigma,
            use_anisotropic_diffusion=langevin_params.use_anisotropic_diffusion,
            diagonal_diffusion=langevin_params.diagonal_diffusion,
            potential=self.potential,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # Create FitnessOperator (always required for EuclideanGas)
        fitness_params = FitnessParams(
            alpha=float(self.alpha_fit),
            beta=float(self.beta_fit),
            eta=float(self.eta),
            lambda_alg=float(self.lambda_alg),
            sigma_min=float(self.sigma_min),
            A=float(self.A),
        )
        fitness_op = FitnessOperator(
            params=fitness_params,
            companion_selection=companion_selection,
        )

        # Create EuclideanGas with direct initialization
        gas = EuclideanGas(
            N=int(self.N),
            d=dims,
            companion_selection=companion_selection,
            potential=self.potential,
            kinetic_op=kinetic_op,
            cloning=cloning_params,
            fitness_op=fitness_op,
            bounds=bounds,
            device=torch.device("cpu"),
            dtype="float32",
            enable_cloning=bool(self.enable_cloning),
            enable_kinetic=bool(self.enable_kinetic),
        )

        offset = torch.full((dims,), float(self.init_offset), dtype=torch.float32)
        x_init = torch.randn(self.N, dims) * float(self.init_spread) + offset
        x_init = torch.clamp(x_init, min=low, max=high)
        v_init = torch.randn(self.N, dims) * float(self.init_velocity_scale)

        history = gas.run(self.n_steps, x_init=x_init, v_init=v_init)

        x_traj = history.x_final.detach().cpu().numpy()
        v_traj = history.v_final.detach().cpu().numpy()
        n_alive = history.n_alive.detach().cpu().numpy()
        will_clone_traj = history.will_clone.detach().cpu().numpy()  # [n_recorded-1, N]

        # Compute variances (total variance across walkers and dimensions)
        var_x = torch.var(history.x_final, dim=1).sum(dim=-1).detach().cpu().numpy()
        var_v = torch.var(history.v_final, dim=1).sum(dim=-1).detach().cpu().numpy()

        indices = np.arange(0, x_traj.shape[0], stride)
        if indices[-1] != x_traj.shape[0] - 1:
            indices = np.append(indices, x_traj.shape[0] - 1)

        positions = x_traj[indices]
        V_total = (var_x + var_v)[indices]
        times = indices.astype(int)
        alive = n_alive[indices]

        velocity_series: list[np.ndarray] = []
        fitness_series: list[np.ndarray] = []
        distance_series: list[np.ndarray] = []
        reward_series: list[np.ndarray] = []
        alive_masks: list[np.ndarray] = []
        previous_positions: list[np.ndarray | None] = []
        will_clone_series: list[np.ndarray] = []
        force_vectors_series: list[np.ndarray] = []
        force_magnitudes_series: list[np.ndarray] = []

        for idx, step_idx in enumerate(indices):
            x_t = torch.from_numpy(x_traj[step_idx]).to(dtype=torch.float32)
            v_t = torch.from_numpy(v_traj[step_idx]).to(dtype=torch.float32)

            # Store previous position for velocity vector visualization
            if step_idx == 0:
                previous_positions.append(None)
            else:
                prev_idx = max(0, step_idx - 1)
                previous_positions.append(x_traj[prev_idx])

            # Store cloning flags for current step
            # will_clone_traj[i] corresponds to step i+1 (no data at step 0)
            # At step_idx, we want will_clone from step_idx-1 (if it exists)
            if step_idx == 0:
                # No cloning data at initial step
                will_clone_series.append(np.zeros(x_t.shape[0], dtype=bool))
            else:
                # Get cloning data from the step that produced current positions
                # step_idx-1 in the full trajectory corresponds to step_idx-1 in will_clone_traj
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

            if alive_np.any():
                with torch.no_grad():
                    rewards = -self.potential.evaluate(x_t)

                companions = companion_selection(x=x_t, v=v_t, alive_mask=alive_mask)

                with torch.no_grad():
                    fitness_vals, info = compute_fitness(
                        positions=x_t,
                        velocities=v_t,
                        rewards=rewards,
                        alive=alive_mask,
                        companions=companions,
                        alpha=cloning_params.alpha,
                        beta=cloning_params.beta,
                        eta=cloning_params.eta,
                        lambda_alg=cloning_params.lambda_alg,
                        sigma_min=cloning_params.sigma_min,
                        A=cloning_params.A,
                    )
                    distances = info["distances"]

                rewards_np = rewards.detach().cpu().numpy()
                fitness_np = fitness_vals.detach().cpu().numpy()
                distances_np = distances.detach().cpu().numpy()
            else:
                rewards_np = np.zeros(x_t.shape[0], dtype=np.float32)
                fitness_np = np.zeros(x_t.shape[0], dtype=np.float32)
                distances_np = np.zeros(x_t.shape[0], dtype=np.float32)

            velocity_series.append(vel_mag[alive_np])
            fitness_series.append(fitness_np[alive_np])
            distance_series.append(distances_np[alive_np])
            reward_series.append(rewards_np[alive_np])

            # Compute force vectors F = -∇U(x) - ε_F·∇V_fit at current positions
            force_vectors_np = np.zeros((x_t.shape[0], x_t.shape[1]), dtype=np.float32)
            force_mag_np = np.zeros(x_t.shape[0], dtype=np.float32)

            if alive_np.any():
                force_total = torch.zeros_like(x_t)

                # Potential force: -∇U(x)
                if self.use_potential_force:
                    x_t_grad = x_t.clone().requires_grad_(True)
                    U = self.potential.evaluate(x_t_grad)
                    grad_U = torch.autograd.grad(U.sum(), x_t_grad, create_graph=False)[0]
                    force_total -= grad_U
                    x_t_grad.requires_grad_(False)

                # Fitness force: -ε_F·∇V_fit (if fitness force enabled)
                if self.use_fitness_force and alive_np.any():
                    # Compute fitness gradient
                    grad_fitness = fitness_op.compute_gradient(
                        positions=x_t,
                        velocities=v_t,
                        rewards=rewards,
                        alive=alive_mask,
                        companions=companions,
                    )
                    force_total -= float(self.epsilon_F) * grad_fitness

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
            "alive_masks": alive_masks,
            "previous_positions": previous_positions,
            "will_clone_series": will_clone_series,
            "force_vectors_series": force_vectors_series,
            "force_magnitudes_series": force_magnitudes_series,
        }

        frame_count = len(times)
        self.time_player.start = 0
        self.time_player.end = max(frame_count - 1, 0)
        self.time_player.value = 0
        self.time_player.disabled = frame_count <= 1
        self.time_player.name = f"time (stride {stride})"

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
        was_cloned_full = data["will_clone_series"][frame]  # Cloning flags from previous step

        if alive_mask.any():
            positions = positions_full[alive_mask]
            # Extract previous positions for alive walkers
            if prev_positions_full is not None:
                prev_positions = prev_positions_full[alive_mask]
            else:
                prev_positions = None
            # Extract cloning flags for alive walkers
            was_cloned = was_cloned_full[alive_mask]
            velocity_vals = np.asarray(data["velocity_series"][frame], dtype=float)
            fitness_vals = np.asarray(data["fitness_series"][frame], dtype=float)
            distance_vals = np.asarray(data["distance_series"][frame], dtype=float)
            reward_vals = np.asarray(data["reward_series"][frame], dtype=float)
            # Extract force vectors and magnitudes
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
            "force_vectors": force_vectors,
            "force_magnitudes": force_magnitudes,
            "data": data,
        }

    def _render_main_plot(self, frame: int):
        """Render the main scatter plot."""
        frame_data = self._get_frame_data(frame)

        if frame_data is None:
            return (self.background * self.mode_points).opts(
                title="Run the simulation to visualise the swarm",
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

        # Build visualization overlay layer by layer
        overlay = self.background

        # Add velocity vectors if enabled
        if self.show_velocity_vectors and prev_positions is not None and len(positions) > 0:
            # Separate arrows by cloning status if feature is enabled
            if self.color_vectors_by_cloning and len(was_cloned) > 0:
                # Split into two groups: diffusion (cyan) and cloned (yellow)
                diffusion_paths = []
                cloned_paths = []

                for i in range(len(positions)):
                    x1, y1 = positions[i]  # Current position
                    x0, y0 = prev_positions[i]  # Previous position
                    path = [(x0, y0), (x1, y1)]

                    if was_cloned[i]:
                        cloned_paths.append(path)
                    else:
                        diffusion_paths.append(path)

                # Add diffusion arrows (cyan)
                if len(diffusion_paths) > 0:
                    diffusion_arrows = hv.Path(diffusion_paths, kdims=["x₁", "x₂"]).opts(
                        color="cyan",
                        line_width=1.5,
                        alpha=0.7,
                    )
                    overlay = overlay * diffusion_arrows

                # Add cloned arrows (yellow)
                if len(cloned_paths) > 0:
                    cloned_arrows = hv.Path(cloned_paths, kdims=["x₁", "x₂"]).opts(
                        color="#FFD700",  # Gold/yellow for cloned walkers
                        line_width=1.5,
                        alpha=0.7,
                    )
                    overlay = overlay * cloned_arrows
            else:
                # Original behavior: all arrows in cyan
                arrow_paths = []
                for i in range(len(positions)):
                    x1, y1 = positions[i]  # Current position
                    x0, y0 = prev_positions[i]  # Previous position
                    arrow_paths.append([(x0, y0), (x1, y1)])

                if len(arrow_paths) > 0:
                    arrows = hv.Path(arrow_paths, kdims=["x₁", "x₂"]).opts(
                        color="cyan",
                        line_width=1.5,
                        alpha=0.7,
                    )
                    overlay = overlay * arrows

        # Add force vectors if enabled
        if self.show_force_vectors and len(positions) > 0 and len(force_vectors) > 0:
            force_paths = []
            force_mags_for_color = []

            for i in range(len(positions)):
                # Normalize force vector
                force_mag = force_magnitudes[i]
                if force_mag > 1e-10:  # Avoid division by zero
                    force_norm = force_vectors[i] / force_mag
                else:
                    force_norm = np.zeros_like(force_vectors[i])

                # Scale by arrow length parameter
                arrow_end = positions[i] + force_norm * float(self.force_arrow_length)

                # Create path from position to arrow end
                x0, y0 = positions[i]
                x1, y1 = arrow_end
                force_paths.append([(x0, y0), (x1, y1)])
                force_mags_for_color.append(force_mag)

            if len(force_paths) > 0:
                # Normalize force magnitudes to [0, 1] for color mapping
                force_mags_array = np.array(force_mags_for_color)
                if force_mags_array.max() > 1e-10:
                    # Use percentile normalization to avoid outliers dominating
                    p5 = np.percentile(force_mags_array, 5)
                    p95 = np.percentile(force_mags_array, 95)
                    if p95 > p5:
                        force_intensity = np.clip((force_mags_array - p5) / (p95 - p5), 0, 1)
                    else:
                        force_intensity = np.ones_like(force_mags_array)
                else:
                    force_intensity = np.zeros_like(force_mags_array)

                # Create green gradient: light green (low force) to dark green (high force)
                # Use HSL color space: Hue=120 (green), vary Lightness
                colors = []
                for intensity in force_intensity:
                    # Map intensity to lightness: 0.8 (light) to 0.2 (dark)
                    lightness = 0.8 - 0.6 * intensity
                    # Convert HSL to RGB (approximation for green)
                    if lightness > 0.5:
                        green_val = int(255 * (1 - (1 - lightness) * 2))
                    else:
                        green_val = 255
                    red_blue_val = int(255 * lightness * 2) if lightness < 0.5 else int(255 * (1 - (lightness - 0.5) * 2))
                    color_hex = f"#{red_blue_val:02x}{green_val:02x}{red_blue_val:02x}"
                    colors.append(color_hex)

                # Create separate path for each force arrow with its color
                for path, color in zip(force_paths, colors):
                    force_arrow = hv.Path([path], kdims=["x₁", "x₂"]).opts(
                        color=color,
                        line_width=2.0,
                        alpha=0.8,
                    )
                    overlay = overlay * force_arrow

        # Add walker points and mode markers on top
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
        """Render the histogram row."""
        frame_data = self._get_frame_data(frame)

        if frame_data is None:
            # Return empty histograms
            return (
                self._make_histogram([], "Fitness", "#1f77b4")
                + self._make_histogram([], "Distance", "#2ca02c")
                + self._make_histogram([], "Reward", "#d62728")
            )

        fitness_vals = frame_data["fitness_vals"]
        distance_vals = frame_data["distance_vals"]
        reward_vals = frame_data["reward_vals"]

        fitness_hist = self._make_histogram(fitness_vals, "Fitness", "#1f77b4")
        distance_hist = self._make_histogram(distance_vals, "Distance", "#2ca02c")
        reward_hist = self._make_histogram(reward_vals, "Reward", "#d62728")

        return (fitness_hist + distance_hist + reward_hist).opts(opts.Layout(shared_axes=False))

    def _build_param_panel(self, names: Iterable[str]) -> pn.Param:
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

    def panel(self) -> pn.Row:
        general_params = (
            "N",
            "n_steps",
            "measure_stride",
            "enable_cloning",
            "enable_kinetic",
            "auto_update",
            "show_velocity_vectors",
            "color_vectors_by_cloning",
            "show_force_vectors",
            "force_arrow_length",
            "color_metric",
            "size_metric",
        )
        langevin_params = (
            "gamma",
            "beta",
            "delta_t",
            "epsilon_F",
            "use_fitness_force",
            "use_potential_force",
            "epsilon_Sigma",
            "use_anisotropic_diffusion",
            "diagonal_diffusion",
        )
        cloning_params = (
            "sigma_x",
            "lambda_alg",
            "alpha_restitution",
            "alpha_fit",
            "beta_fit",
            "eta",
            "A",
            "sigma_min",
            "p_max",
            "epsilon_clone",
            "companion_method",
            "companion_epsilon",
        )
        init_params = ("init_offset", "init_spread", "init_velocity_scale", "bounds_extent")

        accordion = pn.Accordion(
            ("General", self._build_param_panel(general_params)),
            ("Langevin Dynamics", self._build_param_panel(langevin_params)),
            ("Cloning & Selection", self._build_param_panel(cloning_params)),
            ("Initialization", self._build_param_panel(init_params)),
            sizing_mode="stretch_width",
        )
        accordion.active = [0]

        controls = pn.Column(
            pn.pane.Markdown("### Simulation Controls"),
            accordion,
            self.run_button,
            self.status_pane,
            pn.pane.Markdown("### Playback"),
            self.time_player,
            sizing_mode="stretch_width",
            min_width=380,
        )

        # Create vertical layout of main plot and histograms
        viz_column = pn.Column(
            pn.panel(self.dmap_main.opts(framewise=True)),
            pn.panel(self.dmap_hists),
            sizing_mode="stretch_width",
        )

        return pn.Row(
            controls,
            viz_column,
            sizing_mode="stretch_width",
        )


def create_dashboard(
    potential: object | None = None,
    background: hv.Image | None = None,
    mode_points: hv.Points | None = None,
    *,
    dims: int = 2,
    explorer_params: dict | None = None,
) -> tuple[SwarmExplorer, pn.Row]:
    """Factory returning (explorer, panel) for the Euclidean Gas dashboard."""
    if potential is None or background is None or mode_points is None:
        potential, background, mode_points = prepare_background(dims=dims)

    explorer_params = explorer_params or {}
    explorer = SwarmExplorer(
        potential=potential,
        background=background,
        mode_points=mode_points,
        dims=dims,
        **explorer_params,
    )
    return explorer, explorer.panel()
