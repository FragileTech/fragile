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

from fragile.fractalai.planning_gas import PlanningFractalGas, PlanningTrajectory
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
        default="cartpole-balance",
        objects=[
            "cartpole-balance",
            "cartpole-swingup",
            "reacher-easy",
            "reacher-hard",
            "cheetah-run",
            "walker-walk",
            "humanoid-walk",
            "hopper-stand",
            "finger-spin",
            "acrobot-swingup",
        ],
        doc="DM Control task environment",
    )

    render_width = param.Integer(
        default=480, bounds=(120, 1920), doc="Render width in pixels"
    )

    render_height = param.Integer(
        default=480, bounds=(120, 1920), doc="Render height in pixels"
    )

    # Algorithm parameters
    N = param.Integer(default=30, bounds=(5, 200), doc="Number of walkers")

    dist_coef = param.Number(
        default=1.0, bounds=(0.0, 5.0), doc="Distance coefficient in fitness calculation"
    )

    reward_coef = param.Number(
        default=1.0, bounds=(0.0, 5.0), doc="Reward coefficient in fitness calculation"
    )

    use_cumulative_reward = param.Boolean(
        default=False,
        doc="Use cumulative rewards for fitness (default: step rewards only)",
    )

    n_elite = param.Integer(
        default=0, bounds=(0, 50), doc="Number of elite walkers to preserve (0=disabled)"
    )

    dt_range_min = param.Integer(default=1, bounds=(1, 10), doc="Min action repeat")
    dt_range_max = param.Integer(default=1, bounds=(1, 10), doc="Max action repeat")

    # Simulation controls
    max_iterations = param.Integer(
        default=100, bounds=(10, 1000), doc="Maximum simulation iterations"
    )

    record_frames = param.Boolean(
        default=True,
        doc="Record best walker frames (required for visualization, uses more memory)",
    )

    device = param.ObjectSelector(
        default="cpu", objects=["cpu", "cuda"], doc="Computation device"
    )

    seed = param.Integer(default=42, doc="Random seed for reproducibility")

    n_workers = param.Integer(
        default=1, bounds=(1, 16), doc="Parallel env workers (1=serial)"
    )

    # Planning mode parameters
    tau_inner = param.Integer(
        default=5, bounds=(1, 50), doc="Inner planning horizon (iterations per outer step)"
    )

    outer_dt = param.Integer(
        default=1, bounds=(1, 10), doc="Action repeat for outer environment step"
    )

    use_tree_history = param.Boolean(
        default=False,
        doc="Use graph-backed TreeHistory for cloning lineage tracking",
    )

    prune_history = param.Boolean(
        default=False,
        doc="Dynamically prune dead branches from TreeHistory each step (requires tree history)",
    )

    def __init__(self, **params):
        super().__init__(**params)

        # State
        self.gas: RoboticFractalGas | None = None
        self.history: RoboticHistory | None = None
        self.tree_history = None  # AtariTreeHistory when use_tree_history=True
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
        env_kwargs = dict(
            name=self.task_name,
            render_width=self.render_width,
            render_height=self.render_height,
        )
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

    def _run_single_loop_worker(self, env):
        """Run standard single-loop fractal gas."""
        print("[worker] creating RoboticFractalGas...", flush=True)
        self._schedule_ui_update(
            lambda: setattr(self.status_pane, "object", "Initializing simulation...")
        )

        self.gas = RoboticFractalGas(
            env=env,
            N=self.N,
            dist_coef=self.dist_coef,
            reward_coef=self.reward_coef,
            use_cumulative_reward=self.use_cumulative_reward,
            dt_range=(self.dt_range_min, self.dt_range_max),
            device=self.device,
            seed=self.seed,
            record_frames=self.record_frames,
            n_elite=self.n_elite,
        )
        print("[worker] RoboticFractalGas created, calling reset()...", flush=True)

        state = self.gas.reset()
        print("[worker] reset() done, starting iteration loop...", flush=True)
        infos = []
        t_start = time.monotonic()

        # Optional tree history recording
        tree = None
        if self.use_tree_history:
            from fragile.fractalai.videogames.atari_tree_history import AtariTreeHistory

            tree = AtariTreeHistory(
                N=self.N, game_name=self.task_name, max_iterations=self.max_iterations,
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

            if tree is not None:
                tree.record_atari_step(
                    state_before=prev_state,
                    state_after_clone=info["_state_after_clone"],
                    state_final=state,
                    info=info,
                    clone_companions=info["clone_companions"],
                    will_clone=info["will_clone"],
                    best_frame=info.get("best_frame"),
                )
                if self.prune_history:
                    tree.prune_dead_branches()

            if i == 0:
                print("[worker] first step() completed", flush=True)

            self._update_progress(i, t_start)

        # Build history
        if tree is not None:
            self.tree_history = tree
            self.history = tree.to_robotic_history()
        else:
            self.tree_history = None
            self.history = RoboticHistory.from_run(infos, state, self.N, self.task_name)

        self._finish_simulation()

    def _run_planning_worker(self, env):
        """Run two-level planning fractal gas."""
        print("[worker] creating PlanningFractalGas (Planning mode)...", flush=True)
        self._schedule_ui_update(
            lambda: setattr(self.status_pane, "object", "Initializing planning simulation...")
        )

        pg = PlanningFractalGas(
            env=env,
            N=self.N,
            tau_inner=self.tau_inner,
            inner_gas_cls=RoboticFractalGas,
            dist_coef=self.dist_coef,
            reward_coef=self.reward_coef,
            use_cumulative_reward=self.use_cumulative_reward,
            dt_range=(self.dt_range_min, self.dt_range_max),
            device=self.device,
            seed=self.seed,
            record_frames=False,
            n_elite=self.n_elite,
            outer_dt=self.outer_dt,
        )
        print("[worker] PlanningFractalGas created, calling reset()...", flush=True)

        state, obs, info = pg.reset()
        print("[worker] reset() done, starting planning loop...", flush=True)
        t_start = time.monotonic()

        traj = PlanningTrajectory(frames=[] if self.record_frames else None)
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

            if self.record_frames:
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

        # Convert PlanningTrajectory to RoboticHistory for the visualizer
        n = traj.num_steps
        self.tree_history = None
        self.history = RoboticHistory(
            iterations=list(range(n)),
            rewards_mean=list(traj.cumulative_rewards),
            rewards_max=list(traj.cumulative_rewards),
            rewards_min=list(traj.rewards),
            alive_counts=[pi.get("alive_count", self.N) for pi in traj.planning_infos],
            num_cloned=[0] * n,
            virtual_rewards_mean=[
                pi.get("inner_mean_reward", 0.0) for pi in traj.planning_infos
            ],
            virtual_rewards_max=[
                pi.get("inner_max_reward", 0.0) for pi in traj.planning_infos
            ],
            best_frames=traj.frames if traj.frames else [None] * n,
            best_rewards=list(traj.cumulative_rewards),
            best_indices=[0] * n,
            N=self.N,
            max_iterations=n,
            task_name=self.task_name,
        )

        self._finish_simulation()

    def _update_progress(self, i: int, t_start: float):
        """Update the progress bar and text for iteration *i*."""
        done_count = i + 1
        progress = int(done_count / self.max_iterations * 100)
        elapsed = time.monotonic() - t_start
        remaining = (
            (elapsed / done_count) * (self.max_iterations - done_count)
            if done_count > 0
            else 0.0
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
                callback(self.history)

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
        """Schedule UI update on main thread."""
        if pn.state.curdoc:
            pn.state.curdoc.add_next_tick_callback(func)
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
                },
            ),
            pn.pane.Markdown("### Algorithm"),
            pn.Param(
                self.param,
                parameters=["N", "dist_coef", "reward_coef", "use_cumulative_reward", "n_elite"],
            ),
            pn.pane.Markdown("### Planning"),
            pn.Param(
                self.param,
                parameters=["tau_inner", "outer_dt"],
            ),
            pn.pane.Markdown("### Simulation"),
            pn.Param(
                self.param,
                parameters=[
                    "max_iterations",
                    "record_frames",
                    "use_tree_history",
                    "prune_history",
                    "dt_range_min",
                    "dt_range_max",
                    "device",
                    "seed",
                    "n_workers",
                ],
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
    - Cumulative reward progression curve
    - Real-time metric histograms
    """

    show_histograms = param.Boolean(default=True, doc="Show metric histograms")

    def __init__(self, history: RoboticHistory | None = None, **params):
        super().__init__(**params)
        self.history = history

        # Time player for frame-by-frame navigation
        self.time_player = pn.widgets.Player(
            name="Iteration",
            start=0,
            end=0,
            value=0,
            step=1,
            interval=200,  # 200ms = 5 FPS
            loop_policy="loop",
            width=600,
        )
        self.time_player.disabled = True
        self.time_player.param.watch(self._on_frame_change, "value")

        # Frame display (480x480 for MuJoCo rendering)
        self.frame_pane = pn.pane.PNG(
            object=self._create_blank_frame(),
            width=480,
            height=480,
        )

        # Reward progression plot
        self.reward_plot_pane = pn.pane.HoloViews(
            sizing_mode="stretch_width", min_height=300
        )

        # Info display
        self.info_pane = pn.pane.Markdown("No data loaded.")

        # Histogram panes
        self.histogram_alive_pane = pn.pane.HoloViews(sizing_mode="stretch_width")
        self.histogram_cloning_pane = pn.pane.HoloViews(sizing_mode="stretch_width")
        self.histogram_virtual_reward_pane = pn.pane.HoloViews(sizing_mode="stretch_width")

    def set_history(self, history: RoboticHistory):
        """Load new history for visualization."""
        self.history = history

        # Update time player
        self.time_player.end = len(history.iterations) - 1
        self.time_player.value = 0
        self.time_player.disabled = False

        # Update info
        self.info_pane.object = (
            f"**Loaded:** {history.max_iterations} iterations | "
            f"Task: {history.task_name} | "
            f"N={history.N} walkers | "
            f"Max reward: {max(history.rewards_max):.1f}"
        )

        # Trigger initial display
        self._on_frame_change(None)

    def _on_frame_change(self, event):
        """Update visualizations for current frame."""
        if self.history is None:
            return

        idx = int(self.time_player.value)

        # Update frame display
        if self.history.has_frames and self.history.best_frames[idx] is not None:
            frame = self.history.best_frames[idx]
            self.frame_pane.object = self._array_to_png(frame)

        # Update reward curve (show data up to current frame)
        reward_data = pd.DataFrame(
            {
                "iteration": self.history.iterations[: idx + 1],
                "max_reward": self.history.rewards_max[: idx + 1],
                "mean_reward": self.history.rewards_mean[: idx + 1],
            }
        )

        max_curve = hv.Curve(
            reward_data, kdims=["iteration"], vdims=["max_reward"], label="Max Reward"
        ).opts(
            responsive=True,
            height=350,
            title="Cumulative Reward Progression",
            xlabel="Iteration",
            ylabel="Cumulative Reward",
            color="red",
            line_width=2,
        )

        mean_curve = hv.Curve(
            reward_data, kdims=["iteration"], vdims=["mean_reward"], label="Mean Reward"
        ).opts(color="blue", line_width=2, alpha=0.6)

        self.reward_plot_pane.object = (max_curve * mean_curve).opts(legend_position="top_left")

        # Update histograms (show distribution up to current frame)
        if self.show_histograms:
            # Alive walkers histogram
            alive_data = np.array(self.history.alive_counts[: idx + 1])
            alive_hist = hv.Histogram(np.histogram(alive_data, bins=20)).opts(
                responsive=True, height=280, title="Alive Walkers", xlabel="Count", color="green"
            )
            self.histogram_alive_pane.object = alive_hist

            # Cloning events histogram
            cloning_data = np.array(self.history.num_cloned[: idx + 1])
            cloning_hist = hv.Histogram(np.histogram(cloning_data, bins=20)).opts(
                responsive=True, height=280, title="Cloning Events", xlabel="Count", color="orange"
            )
            self.histogram_cloning_pane.object = cloning_hist

            # Virtual rewards histogram
            vr_data = np.array(self.history.virtual_rewards_mean[: idx + 1])
            vr_hist = hv.Histogram(np.histogram(vr_data, bins=30)).opts(
                responsive=True, height=280, title="Virtual Rewards", xlabel="Value", color="purple"
            )
            self.histogram_virtual_reward_pane.object = vr_hist

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
        histogram_section = (
            pn.Column(
                pn.pane.Markdown("### Metrics Distribution"),
                pn.Row(
                    self.histogram_alive_pane,
                    self.histogram_cloning_pane,
                    self.histogram_virtual_reward_pane,
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            )
            if self.show_histograms
            else None
        )

        return pn.Column(
            pn.pane.Markdown("## DM Control Gas Visualization"),
            self.time_player,
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("### Best Walker Frame"),
                    self.frame_pane,
                ),
                pn.Column(
                    pn.pane.Markdown("### Reward Progression"),
                    self.reward_plot_pane,
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            ),
            histogram_section,
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
        visualizer.set_history(history)

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
