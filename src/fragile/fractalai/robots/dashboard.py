"""Dashboard for DM Control Fractal Gas visualization."""

import atexit
import io
import os
import threading
import time
from typing import Callable

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param

from fragile.fractalai.dashboard_plots import build_line_plot, build_minmax_error_plot
from fragile.fractalai.planning_gas import PlanningFractalGas, PlanningHistory, PlanningTrajectory
from fragile.fractalai.robots.dm_control_env import DMControlEnv, VectorizedDMControlEnv
from fragile.fractalai.robots.robotic_gas import RoboticFractalGas
from fragile.fractalai.robots.robotic_history import RoboticHistory


# Import PIL for image conversion
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def _configure_mujoco_offscreen():
    """Set MUJOCO_GL=osmesa for offscreen rendering if not already set.

    MuJoCo requires an OpenGL context for rendering. On headless systems
    (servers, WSL, CI), the osmesa backend provides software rendering
    without a display server.
    """
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "osmesa"
        print("Offscreen mode: set MUJOCO_GL=osmesa", flush=True)


def _format_duration(seconds: float) -> str:
    """Format seconds as ``H:MM:SS`` or ``M:SS``."""
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class RoboticGasConfigPanel(param.Parameterized):
    """Configuration panel for DM Control Gas algorithm.

    Provides parameter controls, simulation execution, and callback system.
    """

    # Mode selector
    algorithm_mode = param.ObjectSelector(
        default="Single Loop",
        objects=["Single Loop", "Planning"],
        doc="Single Loop: standard swarm exploration. Planning: two-level planner using inner swarm.",
    )

    # Environment parameters
    task_name = param.ObjectSelector(
        default="walker-walk",
        objects=[
            "cartpole-balance",
            "cartpole-swingup",
            "reacher-easy",
            "reacher-hard",
            "cheetah-run",
            "walker-walk",
            "walker-stand",
            "humanoid-walk",
            "hopper-stand",
            "finger-spin",
            "acrobot-swingup",
        ],
        doc="DM Control task environment",
    )

    render_width = param.Integer(default=480, bounds=(120, 1920), doc="Render width in pixels")

    render_height = param.Integer(default=480, bounds=(120, 1920), doc="Render height in pixels")

    # Algorithm parameters
    N = param.Integer(default=100, bounds=(5, 5000), doc="Number of walkers")

    dist_coef = param.Number(
        default=1.0, bounds=(0.0, 5.0), doc="Distance coefficient in fitness calculation"
    )

    reward_coef = param.Number(
        default=1.0, bounds=(0.0, 5.0), doc="Reward coefficient in fitness calculation"
    )

    n_elite = param.Integer(
        default=5, bounds=(0, 50), doc="Number of elite walkers to preserve (0=disabled)"
    )

    action_mode = param.ObjectSelector(
        default="Uniform",
        objects=["Uniform", "Gaussian"],
        doc="Action sampling: Uniform random or Gaussian (clipped to bounds)",
    )

    gaussian_mean = param.Number(
        default=0.0,
        bounds=(-1.0, 1.0),
        doc="Mean of Gaussian action distribution",
    )

    gaussian_std = param.Number(
        default=0.3,
        bounds=(0.01, 2.0),
        doc="Std dev of Gaussian action distribution",
    )

    dt_range_min = param.Integer(default=1, bounds=(1, 10), doc="Min action repeat")
    dt_range_max = param.Integer(default=1, bounds=(1, 10), doc="Max action repeat")

    # Simulation controls
    max_iterations = param.Integer(
        default=100, bounds=(10, 1000), doc="Maximum simulation iterations"
    )

    seed = param.Integer(default=42, doc="Random seed for reproducibility")

    n_workers = param.Integer(default=1, bounds=(1, 16), doc="Parallel env workers (1=serial)")

    # Planning mode parameters
    tau_inner = param.Integer(
        default=5, bounds=(1, 50), doc="Inner planning horizon (iterations per outer step)"
    )

    outer_dt = param.Integer(
        default=1, bounds=(1, 10), doc="Action repeat for outer environment step"
    )

    def __init__(self, **params):
        super().__init__(**params)

        # State
        self.gas: RoboticFractalGas | None = None
        self.history: RoboticHistory | None = None
        self.planning_history: PlanningHistory | None = None
        self.tree_history = None  # AtariTreeHistory always enabled
        self.prune_tree_history = True
        self._simulation_thread: threading.Thread | None = None
        self._stop_requested = False
        self._env = None
        atexit.register(self._close_env)

        # UI components
        self.run_button = pn.widgets.Button(
            name="Run Simulation",
            button_type="primary",
            width=200,
        )
        self.run_button.on_click(self._on_run_clicked)

        self.stop_button = pn.widgets.Button(
            name="Stop",
            button_type="danger",
            width=200,
            disabled=True,
        )
        self.stop_button.on_click(self._on_stop_clicked)

        self.progress_bar = pn.indicators.Progress(
            name="Progress",
            value=0,
            max=100,
            bar_color="primary",
            width=200,
        )

        self.progress_text = pn.pane.Markdown("", styles={"font-family": "monospace"})

        self.status_pane = pn.pane.Markdown(
            "Ready to run simulation. "
            "**Note:** Uses dm_control MuJoCo environment backend (offscreen compatible)."
        )

        # Callbacks
        self._on_simulation_complete: list[Callable[[RoboticHistory], None]] = []

        # Mode-specific parameter sections
        self._planning_params_section = pn.Column(
            pn.pane.Markdown("### Planning"),
            pn.Param(
                self.param,
                parameters=["tau_inner", "outer_dt"],
                widgets={
                    "tau_inner": pn.widgets.EditableIntSlider,
                    "outer_dt": pn.widgets.EditableIntSlider,
                },
            ),
            visible=(self.algorithm_mode == "Planning"),
        )
        self.param.watch(self._on_mode_changed, "algorithm_mode")

        # Gaussian action params (visible only when action_mode == "Gaussian")
        self._gaussian_params_section = pn.Column(
            pn.Param(
                self.param,
                parameters=["gaussian_mean", "gaussian_std"],
                widgets={
                    "gaussian_mean": pn.widgets.EditableFloatSlider,
                    "gaussian_std": pn.widgets.EditableFloatSlider,
                },
            ),
            visible=(self.action_mode == "Gaussian"),
        )
        self.param.watch(self._on_action_mode_changed, "action_mode")

    def _on_action_mode_changed(self, event):
        """Toggle visibility of Gaussian parameter section."""
        self._gaussian_params_section.visible = event.new == "Gaussian"

    def _on_mode_changed(self, event):
        """Toggle visibility of mode-specific parameter sections."""
        is_planning = event.new == "Planning"
        self._planning_params_section.visible = is_planning

    def add_completion_callback(self, callback: Callable[[RoboticHistory], None]):
        """Register callback for simulation completion."""
        self._on_simulation_complete.append(callback)

    def _close_env(self):
        """Close tracked environment and release resources."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def _on_run_clicked(self, event):
        """Handle run button click.

        Environment creation and simulation run in a worker thread to avoid
        blocking the Tornado event loop.
        """
        if self._simulation_thread and self._simulation_thread.is_alive():
            return

        # Capture the Bokeh document on the main thread so the worker
        # thread can schedule UI updates even though pn.state.curdoc is
        # thread-local and would be None on a new thread.
        self._curdoc = pn.state.curdoc

        self._stop_requested = False
        self.run_button.disabled = True
        self.stop_button.disabled = False
        self.progress_bar.value = 0
        self.status_pane.object = "Creating environment..."

        self._simulation_thread = threading.Thread(
            target=self._run_simulation_worker,
            daemon=True,
        )
        self._simulation_thread.start()

    def _on_stop_clicked(self, event):
        """Handle stop button click."""
        self._stop_requested = True
        self.status_pane.object = "Stopping simulation..."

    def _create_environment(self):
        """Create DM Control environment.

        Returns:
            DMControlEnv compatible with RoboticFractalGas

        Raises:
            Exception: If environment creation fails
        """
        _configure_mujoco_offscreen()
        env_kwargs = {
            "name": self.task_name,
            "render_width": self.render_width,
            "render_height": self.render_height,
            "include_rgb": False,
        }
        if self.n_workers > 1:
            print(
                f"[worker] creating VectorizedDMControlEnv(name={self.task_name!r}, "
                f"n_workers={self.n_workers})...",
                flush=True,
            )
            env = VectorizedDMControlEnv(n_workers=self.n_workers, **env_kwargs)
        else:
            print(f"[worker] creating DMControlEnv(name={self.task_name!r})...", flush=True)
            env = DMControlEnv(**env_kwargs)
        print(f"[worker] Environment created: {self.task_name}", flush=True)
        self._schedule_ui_update(
            lambda: setattr(self.status_pane, "object", f"Using dm_control: {self.task_name}")
        )
        return env

    def _run_simulation_worker(self):
        """Background thread for environment creation and simulation execution."""
        try:
            self._close_env()  # close any previous env
            print("[worker] starting environment creation...", flush=True)
            self._schedule_ui_update(
                lambda: setattr(self.status_pane, "object", "Creating environment...")
            )
            env = self._create_environment()
            self._env = env  # track for cleanup

            if self.algorithm_mode == "Planning":
                self._run_planning_worker(env)
            else:
                self._run_single_loop_worker(env)

        except Exception as e:
            error_details = str(e)
            self._schedule_ui_update(
                lambda: setattr(self.status_pane, "object", f"**Error:** {error_details}")
            )
        finally:
            self._close_env()
            self._schedule_ui_update(self._cleanup_simulation)

    def _build_action_sampler(self, env):
        """Build the action sampler based on the current action_mode setting."""
        if self.action_mode == "Gaussian":
            from fragile.fractalai.videogames.kinetic import gaussian_action_sampler

            return gaussian_action_sampler(
                env.action_space,
                mean=self.gaussian_mean,
                std=self.gaussian_std,
            )
        return None

    def _run_single_loop_worker(self, env):
        """Run standard single-loop fractal gas."""
        self.planning_history = None
        print("[worker] creating RoboticFractalGas...", flush=True)
        self._schedule_ui_update(
            lambda: setattr(self.status_pane, "object", "Initializing simulation...")
        )

        action_sampler = self._build_action_sampler(env)
        self.gas = RoboticFractalGas(
            env=env,
            N=self.N,
            dist_coef=self.dist_coef,
            reward_coef=self.reward_coef,
            use_cumulative_reward=True,
            dt_range=(self.dt_range_min, self.dt_range_max),
            seed=self.seed,
            record_frames=False,
            n_elite=self.n_elite,
            action_sampler=action_sampler,
        )
        print("[worker] RoboticFractalGas created, calling reset()...", flush=True)

        state = self.gas.reset()
        print("[worker] reset() done, starting iteration loop...", flush=True)
        infos = []
        t_start = time.monotonic()

        # Optional tree history recording
        from fragile.fractalai.videogames.atari_tree_history import AtariTreeHistory

        tree = AtariTreeHistory(
            N=self.N,
            game_name=self.task_name,
            max_iterations=self.max_iterations,
        )
        tree.record_initial_atari_state(state)

        for i in range(self.max_iterations):
            if self._stop_requested:
                self._schedule_ui_update(
                    lambda: setattr(self.status_pane, "object", "**Stopped by user**")
                )
                break

            prev_state = state
            state, info = self.gas.step(prev_state)
            infos.append(info)

            tree.record_atari_step(
                state_before=prev_state,
                state_after_clone=info["_state_after_clone"],
                state_final=state,
                info=info,
                clone_companions=info["clone_companions"],
                will_clone=info["will_clone"],
                best_frame=info.get("best_frame"),
            )
            if self.prune_tree_history:
                tree.prune_dead_branches()

            if i == 0:
                print("[worker] first step() completed", flush=True)

            self._update_progress(i, t_start)

        # Build history
        self.tree_history = tree
        if self.gas is not None:
            tree.render_missing_path_frames(self.gas._render_walker_frame)
            tree.render_elite_path_frames(self.gas._render_walker_frame)
        self.history = tree.to_robotic_history()

        self._finish_simulation()

    def _run_planning_worker(self, env):
        """Run two-level planning fractal gas."""
        print("[worker] creating PlanningFractalGas (Planning mode)...", flush=True)
        self._schedule_ui_update(
            lambda: setattr(self.status_pane, "object", "Initializing planning simulation...")
        )

        action_sampler = self._build_action_sampler(env)
        pg = PlanningFractalGas(
            env=env,
            N=self.N,
            tau_inner=self.tau_inner,
            inner_gas_cls=RoboticFractalGas,
            dist_coef=self.dist_coef,
            reward_coef=self.reward_coef,
            use_cumulative_reward=True,
            dt_range=(self.dt_range_min, self.dt_range_max),
            seed=self.seed,
            record_frames=False,
            n_elite=self.n_elite,
            outer_dt=self.outer_dt,
            action_sampler=action_sampler,
        )
        print("[worker] PlanningFractalGas created, calling reset()...", flush=True)

        state, obs, info = pg.reset()
        print("[worker] reset() done, starting planning loop...", flush=True)
        t_start = time.monotonic()

        traj = PlanningTrajectory(frames=[])
        traj.states.append(state)
        traj.observations.append(obs)
        cum_reward = 0.0

        for i in range(self.max_iterations):
            if self._stop_requested:
                self._schedule_ui_update(
                    lambda: setattr(self.status_pane, "object", "**Stopped by user**")
                )
                break

            new_state, new_obs, reward, done, truncated, new_info, step_info = pg.step(
                state, obs, info
            )

            cum_reward += float(reward)
            traj.actions.append(step_info["action"])
            traj.rewards.append(float(reward))
            traj.cumulative_rewards.append(cum_reward)
            traj.dones.append(bool(done))
            traj.planning_infos.append(step_info["plan_info"])
            traj.states.append(new_state)
            traj.observations.append(new_obs)

            try:
                frame = pg.inner_gas._render_walker_frame(new_state)
                traj.frames.append(frame)
            except Exception:
                traj.frames.append(None)

            if i == 0:
                print("[worker] first planning step() completed", flush=True)

            self._update_progress(i, t_start)

            state, obs, info = new_state, new_obs, new_info
            if done or truncated:
                break

        # Build PlanningHistory and convert to RoboticHistory for the visualizer
        self.tree_history = None
        self.planning_history = PlanningHistory.from_trajectory(
            traj,
            self.N,
            self.task_name,
        )
        self.history = self.planning_history.to_robotic_history()

        self._finish_simulation()

    def _update_progress(self, i: int, t_start: float):
        """Update the progress bar and text for iteration *i*."""
        done_count = i + 1
        progress = int(done_count / self.max_iterations * 100)
        elapsed = time.monotonic() - t_start
        remaining = (
            (elapsed / done_count) * (self.max_iterations - done_count) if done_count > 0 else 0.0
        )
        remaining_str = _format_duration(remaining)
        pct = done_count / self.max_iterations * 100
        txt = (
            f"`Progress: {done_count}/{self.max_iterations} "
            f"({pct:.1f}%) | Remaining: {remaining_str}`"
        )
        self._schedule_ui_update(
            lambda p=progress, t=txt: (
                setattr(self.progress_bar, "value", p),
                setattr(self.progress_text, "object", t),
            )
        )

    def _finish_simulation(self):
        """Notify UI and callbacks after a successful simulation."""
        if not self._stop_requested:
            self._schedule_ui_update(self._on_simulation_finished)
            for callback in self._on_simulation_complete:
                self._schedule_ui_update(lambda cb=callback: cb(self.history))

    def _on_simulation_finished(self):
        """UI update when simulation completes."""
        self.status_pane.object = (
            f"**Completed:** {self.history.max_iterations} iterations, "
            f"Max reward: {max(self.history.rewards_max):.1f}"
        )

    def _cleanup_simulation(self):
        """Reset UI state after simulation."""
        self.run_button.disabled = False
        self.stop_button.disabled = True

    def _schedule_ui_update(self, func):
        """Schedule UI update on the Bokeh document thread.

        Uses the document reference captured in ``_on_run_clicked`` so that
        worker threads can schedule callbacks even though
        ``pn.state.curdoc`` is thread-local and may be ``None`` on non-main
        threads.
        """
        doc = getattr(self, "_curdoc", None) or pn.state.curdoc
        if doc:
            doc.add_next_tick_callback(func)
        else:
            func()

    def panel(self) -> pn.Column:
        """Create parameter panel layout."""
        return pn.Column(
            pn.pane.Markdown("### Mode"),
            pn.Param(
                self.param,
                parameters=["algorithm_mode"],
                widgets={"algorithm_mode": pn.widgets.RadioButtonGroup},
            ),
            pn.pane.Markdown("### Environment"),
            pn.Param(
                self.param,
                parameters=["task_name", "render_width", "render_height"],
                widgets={
                    "task_name": pn.widgets.Select,
                    "render_width": pn.widgets.EditableIntSlider,
                    "render_height": pn.widgets.EditableIntSlider,
                },
            ),
            pn.pane.Markdown("### Algorithm"),
            pn.Param(
                self.param,
                parameters=["N", "dist_coef", "reward_coef", "n_elite"],
                widgets={
                    "N": pn.widgets.EditableIntSlider,
                    "dist_coef": pn.widgets.EditableFloatSlider,
                    "reward_coef": pn.widgets.EditableFloatSlider,
                    "n_elite": pn.widgets.EditableIntSlider,
                },
            ),
            pn.Param(
                self.param,
                parameters=["action_mode"],
                widgets={"action_mode": pn.widgets.RadioButtonGroup},
            ),
            self._gaussian_params_section,
            self._planning_params_section,
            pn.pane.Markdown("### Simulation"),
            pn.Param(
                self.param,
                parameters=[
                    "max_iterations",
                    "dt_range_min",
                    "dt_range_max",
                    "seed",
                    "n_workers",
                ],
                widgets={
                    "max_iterations": pn.widgets.EditableIntSlider,
                    "dt_range_min": pn.widgets.EditableIntSlider,
                    "dt_range_max": pn.widgets.EditableIntSlider,
                    "n_workers": pn.widgets.EditableIntSlider,
                },
            ),
            pn.pane.Markdown("### Controls"),
            self.run_button,
            self.stop_button,
            self.progress_bar,
            self.progress_text,
            self.status_pane,
            sizing_mode="stretch_width",
        )


class RoboticGasVisualizer(param.Parameterized):
    """Visualization panel for DM Control Gas results.

    Features:
    - Best walker frame display with playback
    - Static full-run reward plot with min/max error bars
    - Algorithm-style error bar time-series plots (fitness, dt)
    - Clone % and alive count line plots
    """

    def __init__(self, history: RoboticHistory | None = None, **params):
        super().__init__(**params)
        self.history = history
        self.planning_history: PlanningHistory | None = None
        self.tree_history = None
        self._current_frames: list[np.ndarray | None] = []

        # Elite walker dropdown (same as Atari dashboard)
        self.elite_select = pn.widgets.Select(
            name="Elite Walker",
            options={"Best (global)": -1},
            value=-1,
            width=400,
        )
        self.elite_select.param.watch(self._on_elite_select_change, "value")

        # Game replay player (controls frame display)
        self.game_player = pn.widgets.Player(
            name="Game Frame",
            start=0,
            end=0,
            value=0,
            step=1,
            interval=200,
            loop_policy="loop",
            width=400,
        )
        self.game_player.disabled = True
        self.game_player.param.watch(self._on_game_frame_change, "value")

        # Frame display (480x480 for MuJoCo rendering)
        self.frame_pane = pn.pane.PNG(
            object=self._create_blank_frame(),
            width=480,
            height=480,
        )

        # Static plot panes
        self.reward_plot_pane = pn.pane.HoloViews(sizing_mode="stretch_width", min_height=320)
        self.fitness_plot_pane = pn.pane.HoloViews(sizing_mode="stretch_width", min_height=320)
        self.clone_pct_plot_pane = pn.pane.HoloViews(sizing_mode="stretch_width", min_height=320)
        self.alive_plot_pane = pn.pane.HoloViews(sizing_mode="stretch_width", min_height=320)
        self.dt_plot_pane = pn.pane.HoloViews(sizing_mode="stretch_width", min_height=320)

        # Info display
        self.info_pane = pn.pane.Markdown("No data loaded.")

        # Planning-specific panes
        self.step_reward_plot_pane = pn.pane.HoloViews(sizing_mode="stretch_width")
        self.inner_quality_plot_pane = pn.pane.HoloViews(sizing_mode="stretch_width")

    def set_history(self, history: RoboticHistory):
        """Load new history and build all static plots."""
        self.history = history
        self._current_frames = list(history.best_frames)
        last_idx = len(self._current_frames) - 1 if history.has_frames else len(history.iterations) - 1

        # Reset game player
        self.game_player.end = last_idx
        self.game_player.value = min(1, last_idx)
        self.game_player.value = 0
        self.game_player.disabled = False

        # Reset elite dropdown
        self.elite_select.options = {"Best (global)": -1}
        self.elite_select.value = -1

        # Update info
        self.info_pane.object = (
            f"**Loaded:** {history.max_iterations} iterations | "
            f"Task: {history.task_name} | "
            f"N={history.N} walkers | "
            f"Max reward: {max(history.rewards_max):.1f}"
        )

        # Force-update frame pane with the first frame
        if history.has_frames and self._current_frames[0] is not None:
            self.frame_pane.object = self._array_to_png(self._current_frames[0])

        # Build all static plots
        self._build_plots(history)

    def set_planning_history(self, planning_history: PlanningHistory | None):
        """Load planning history for planning-specific visualizations."""
        self.planning_history = planning_history

    def set_tree_history(self, tree_history):
        """Load tree history and populate the elite walker dropdown."""
        self.tree_history = tree_history
        elite_infos = tree_history.get_elite_branches_info()
        if not elite_infos:
            return
        options: dict[str, int] = {}
        for info in elite_infos:
            options[info["label"]] = info["node_id"]
        self.elite_select.options = options

    def _on_elite_select_change(self, event):
        """Switch displayed replay to the selected elite branch."""
        node_id = event.new
        if self.tree_history is None:
            return
        frames = self.tree_history.get_path_frames_for_node(node_id)
        if not frames:
            return
        self._current_frames = frames
        last_idx = len(frames) - 1
        self.game_player.end = last_idx
        self.game_player.value = 0
        if frames[0] is not None:
            self.frame_pane.object = self._array_to_png(frames[0])

    def _build_plots(self, h: RoboticHistory):
        """Build all static plots from the full history data."""
        steps = h.iterations

        # Reward plot: mean line + elite overlays
        reward_overlay = build_line_plot(
            step=steps,
            values=h.rewards_mean,
            title="Cumulative Reward",
            ylabel="Cumulative Reward",
            color="red",
        )
        # Overlay elite walker reward curves
        if self.tree_history is not None:
            elite_curves = self.tree_history.get_elite_reward_curves()
            for curve_data in elite_curves:
                df = pd.DataFrame({
                    "step": curve_data["steps"],
                    "reward": curve_data["rewards"],
                })
                elite_curve = hv.Curve(
                    df, "step", "reward", label=curve_data["label"],
                ).opts(color="blue", line_width=1, alpha=0.6)
                reward_overlay = reward_overlay * elite_curve
            if elite_curves:
                reward_overlay = reward_overlay.opts(legend_position="top_left")
        self.reward_plot_pane.object = reward_overlay

        # Virtual reward / fitness: mean with min/max error bars
        self.fitness_plot_pane.object = build_minmax_error_plot(
            step=steps,
            mean=h.virtual_rewards_mean,
            vmin=h.virtual_rewards_min,
            vmax=h.virtual_rewards_max,
            title="Virtual Reward (mean / min-max)",
            ylabel="Virtual Reward",
            color="blue",
        )

        # Clone %
        clone_pct = [100.0 * nc / h.N for nc in h.num_cloned]
        self.clone_pct_plot_pane.object = build_line_plot(
            step=steps,
            values=clone_pct,
            title="Clone %",
            ylabel="% walkers cloned",
            color="orange",
        )

        # Alive count
        self.alive_plot_pane.object = build_line_plot(
            step=steps,
            values=h.alive_counts,
            title="Alive Walkers",
            ylabel="Count",
            color="green",
        )

        # dt: mean with min/max error bars
        self.dt_plot_pane.object = build_minmax_error_plot(
            step=steps,
            mean=h.dt_mean,
            vmin=h.dt_min,
            vmax=h.dt_max,
            title="Action Repeat dt (mean / min-max)",
            ylabel="dt",
            color="teal",
        )

        # Planning-specific plots
        if self.planning_history is not None:
            ph = self.planning_history
            step_data = pd.DataFrame({
                "step": ph.iterations,
                "step_reward": ph.step_rewards,
            })
            step_curve = hv.Curve(
                step_data, kdims=["step"], vdims=["step_reward"], label="Step Reward"
            ).opts(
                responsive=True,
                height=300,
                title="Per-Step Rewards",
                xlabel="Step",
                ylabel="Reward",
                color="teal",
                line_width=2,
                show_grid=True,
            )
            self.step_reward_plot_pane.object = step_curve

            quality_data = pd.DataFrame({
                "step": ph.iterations,
                "inner_max_reward": ph.inner_max_rewards,
                "step_reward": ph.step_rewards,
            })
            inner_curve = hv.Curve(
                quality_data,
                kdims=["step"],
                vdims=["inner_max_reward"],
                label="Inner Max Reward",
            ).opts(
                responsive=True,
                height=300,
                title="Inner Planning Quality",
                xlabel="Step",
                ylabel="Reward",
                color="orange",
                line_width=2,
                show_grid=True,
            )
            step_overlay = hv.Curve(
                quality_data,
                kdims=["step"],
                vdims=["step_reward"],
                label="Actual Step Reward",
            ).opts(color="teal", line_width=2, alpha=0.7)
            self.inner_quality_plot_pane.object = (inner_curve * step_overlay).opts(
                legend_position="top_left"
            )

    def _on_game_frame_change(self, event):
        """Update frame display only."""
        if not self._current_frames:
            return
        idx = int(self.game_player.value)
        if idx < len(self._current_frames):
            frame = self._current_frames[idx]
            if frame is not None:
                self.frame_pane.object = self._array_to_png(frame)

    def _create_blank_frame(self) -> bytes:
        """Create blank frame as placeholder."""
        if not PIL_AVAILABLE:
            return b""

        img = Image.new("RGB", (480, 480), color="black")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _array_to_png(self, array: np.ndarray) -> bytes:
        """Convert numpy array to PNG bytes."""
        if not PIL_AVAILABLE:
            return b""

        img = Image.fromarray(array.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def panel(self) -> pn.Column:
        """Create visualization layout."""
        self._planning_stats_section = pn.Column(
            pn.pane.Markdown("### Planning Stats"),
            pn.Row(
                self.step_reward_plot_pane,
                self.inner_quality_plot_pane,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
            visible=False,
        )

        return pn.Column(
            pn.pane.Markdown("## DM Control Gas Visualization"),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("### Best Walker Frame"),
                    self.elite_select,
                    self.game_player,
                    self.frame_pane,
                ),
                pn.Column(
                    pn.pane.Markdown("### Reward"),
                    self.reward_plot_pane,
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            ),
            pn.pane.Markdown("### Algorithm Metrics"),
            pn.Row(
                self.fitness_plot_pane,
                self.dt_plot_pane,
                sizing_mode="stretch_width",
            ),
            pn.Row(
                self.clone_pct_plot_pane,
                self.alive_plot_pane,
                sizing_mode="stretch_width",
            ),
            self._planning_stats_section,
            self.info_pane,
            sizing_mode="stretch_width",
        )


def create_app() -> pn.template.FastListTemplate:
    """Create DM Control Gas Dashboard application."""
    hv.extension("bokeh")
    pn.extension()

    # Create components
    config_panel = RoboticGasConfigPanel()
    visualizer = RoboticGasVisualizer()

    # Connect callback
    def on_simulation_complete(history: RoboticHistory):
        """Update visualizer when simulation completes."""
        visualizer.set_planning_history(config_panel.planning_history)
        visualizer._planning_stats_section.visible = config_panel.planning_history is not None
        # Set tree_history before set_history so _build_plots can overlay elite curves
        visualizer.tree_history = config_panel.tree_history
        visualizer.set_history(history)
        if config_panel.tree_history is not None:
            visualizer.set_tree_history(config_panel.tree_history)

    config_panel.add_completion_callback(on_simulation_complete)

    # Create layout
    return pn.template.FastListTemplate(
        title="DM Control Fractal Gas Dashboard",
        sidebar=[config_panel.panel()],
        main=[visualizer.panel()],
        sidebar_width=400,
        main_max_width="100%",
    )


def _parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="DM Control Fractal Gas Dashboard")
    parser.add_argument("--port", type=int, default=5007, help="Port to run server on")
    parser.add_argument("--open", action="store_true", help="Open browser on launch")
    parser.add_argument("--threaded", action="store_true", help="Use multi-threaded Tornado")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _configure_mujoco_offscreen()
    print("Starting DM Control Fractal Gas Dashboard...", flush=True)
    app = create_app()
    if not args.threaded:
        print("Running in single-threaded mode (use --threaded for multi-threaded)", flush=True)
    print(
        f"Open http://localhost:{args.port} in your browser (Ctrl+C to stop)",
        flush=True,
    )
    pn.serve({"/": app}, port=args.port, show=args.open, threaded=args.threaded)
