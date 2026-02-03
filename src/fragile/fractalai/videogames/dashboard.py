"""Dashboard for Atari Fractal Gas visualization."""

import io
import threading
from typing import Any, Callable

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param

from fragile.fractalai.videogames.atari_gas import AtariFractalGas
from fragile.fractalai.videogames.atari_history import AtariHistory

# Don't import plangym at module level - it will be imported lazily when needed
# This avoids OpenGL/display issues on headless systems

# Import PIL for image conversion
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class AtariGasConfigPanel(param.Parameterized):
    """Configuration panel for Atari Gas algorithm.

    Provides parameter controls, simulation execution, and callback system.
    """

    # Environment parameters
    game_name = param.ObjectSelector(
        default="PongNoFrameskip-v4",
        objects=[
            "PongNoFrameskip-v4",
            "BreakoutNoFrameskip-v4",
            "MsPacmanNoFrameskip-v4",
            "SpaceInvadersNoFrameskip-v4",
        ],
        doc="Atari game environment",
    )

    obs_type = param.ObjectSelector(
        default="ram",
        objects=["ram", "rgb", "grayscale"],
        doc="Observation type (ram is fastest, rgb needed for frame display)",
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

    dt_range_min = param.Integer(default=1, bounds=(1, 10), doc="Min frame skip")
    dt_range_max = param.Integer(default=4, bounds=(1, 10), doc="Max frame skip")

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

    def __init__(self, **params):
        super().__init__(**params)

        # State
        self.gas: AtariFractalGas | None = None
        self.history: AtariHistory | None = None
        self._simulation_thread: threading.Thread | None = None
        self._stop_requested = False

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

        self.status_pane = pn.pane.Markdown(
            "Ready to run simulation. "
            "**Note:** Running simulations requires plangym and a display (may not work on headless WSL)."
        )

        # Callbacks
        self._on_simulation_complete: list[Callable[[AtariHistory], None]] = []

    def add_completion_callback(self, callback: Callable[[AtariHistory], None]):
        """Register callback for simulation completion."""
        self._on_simulation_complete.append(callback)

    def _on_run_clicked(self, event):
        """Handle run button click.

        CRITICAL: Environment creation must happen in the MAIN THREAD to avoid
        XCB threading errors. X11/OpenGL is not thread-safe, so we create the
        environment here and pass it to the worker thread.
        """
        if self._simulation_thread and self._simulation_thread.is_alive():
            return

        self._stop_requested = False
        self.run_button.disabled = True
        self.stop_button.disabled = False
        self.progress_bar.value = 0
        self.status_pane.object = "Starting simulation..."

        # Create environment in MAIN THREAD (X11-safe)
        try:
            env = self._create_environment()
        except Exception as e:
            self.status_pane.object = f"**Error creating environment:** {e}\n\nSee terminal for details."
            self.run_button.disabled = False
            self.stop_button.disabled = True
            import traceback
            traceback.print_exc()
            return

        # Pass pre-created environment to worker thread
        self._simulation_thread = threading.Thread(
            target=self._run_simulation_worker,
            args=(env,),  # Pass environment to worker
            daemon=True,
        )
        self._simulation_thread.start()

    def _on_stop_clicked(self, event):
        """Handle stop button click."""
        self._stop_requested = True
        self.status_pane.object = "Stopping simulation..."

    def _create_environment(self):
        """Create Atari environment in main thread (X11-safe).

        CRITICAL: This method MUST be called from the main thread to avoid
        XCB threading errors. X11/OpenGL initialization is not thread-safe.

        Returns:
            Environment wrapper compatible with AtariFractalGas

        Raises:
            Exception: If environment creation fails
        """
        # Check environment before importing plangym/gymnasium
        if not self._check_display_available():
            raise RuntimeError(
                "OpenGL/Display not available. On WSL:\n"
                "1. Use: bash scripts/run_dashboard_wsl.sh\n"
                "2. Or set up xvfb manually (see scripts/run_dashboard_wsl.sh)"
            )

        # Try gymnasium first (better headless support), fall back to plangym
        env = None

        # Try gymnasium with headless rendering
        try:
            import gymnasium as gym
            from gymnasium.wrappers import FrameStack

            # Create gym environment with rgb_array rendering (headless)
            env_name = self.game_name.replace("NoFrameskip-v4", "-v4")
            base_env = gym.make(env_name, render_mode="rgb_array")

            # Wrap to match plangym interface
            class GymEnvWrapper:
                def __init__(self, env):
                    self.env = env
                    self.obs_type = "rgb"  # gymnasium uses rgb arrays

                def reset(self):
                    obs, info = self.env.reset()
                    return obs

                def step(self, action):
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    return obs, reward, terminated or truncated, info

                def render(self):
                    return self.env.render()

                def close(self):
                    self.env.close()

            env = GymEnvWrapper(base_env)
            self.status_pane.object = "Using gymnasium (headless-compatible)"

        except Exception:
            # Fall back to plangym
            try:
                from plangym import AtariEnvironment
                env = AtariEnvironment(
                    name=self.game_name,
                    obs_type=self.obs_type,
                )
                self.status_pane.object = "Using plangym"
            except ImportError:
                raise ImportError(
                    "Neither gymnasium nor plangym available.\n"
                    "Install with: pip install gymnasium[atari] gymnasium[accept-rom-license]\n"
                    "Or: pip install plangym\n\n"
                    "For WSL: Use gymnasium for better headless support"
                )

        if env is None:
            raise RuntimeError("Failed to create Atari environment")

        return env

    def _run_simulation_worker(self, env):
        """Background thread for simulation execution.

        Args:
            env: Pre-created environment (created in main thread to avoid XCB errors)

        CRITICAL: Environment must be created in main thread before calling this.
        Worker thread only uses the environment (reset/step), not create it.
        """
        try:
            # Environment already created in main thread - just use it
            self._schedule_ui_update(
                lambda: setattr(self.status_pane, "object", "Initializing simulation...")
            )

            # Create gas algorithm with pre-created environment
            self.gas = AtariFractalGas(
                env=env,
                N=self.N,
                dist_coef=self.dist_coef,
                reward_coef=self.reward_coef,
                use_cumulative_reward=self.use_cumulative_reward,
                dt_range=(self.dt_range_min, self.dt_range_max),
                device=self.device,
                seed=self.seed,
                record_frames=self.record_frames,
            )

            # Run simulation with progress updates
            state = self.gas.reset()
            infos = []

            for i in range(self.max_iterations):
                if self._stop_requested:
                    self._schedule_ui_update(
                        lambda: setattr(self.status_pane, "object", "**Stopped by user**")
                    )
                    break

                state, info = self.gas.step(state)
                infos.append(info)

                # Update progress
                progress = int((i + 1) / self.max_iterations * 100)
                self._schedule_ui_update(
                    lambda p=progress: setattr(self.progress_bar, "value", p)
                )

            # Build history
            self.history = AtariHistory.from_run(infos, state, self.N, self.game_name)

            # Update UI
            if not self._stop_requested:
                self._schedule_ui_update(self._on_simulation_finished)

                # Notify callbacks
                for callback in self._on_simulation_complete:
                    callback(self.history)

            env.close()

        except Exception as e:
            error_details = str(e)
            if "xcb" in error_details.lower():
                error_details += (
                    "\n\n**XCB threading issue detected.** "
                    "Use the WSL launcher script:\n"
                    "```bash\n"
                    "bash scripts/run_dashboard_wsl.sh\n"
                    "```"
                )

            self._schedule_ui_update(
                lambda: setattr(self.status_pane, "object", f"**Error:** {error_details}")
            )
        finally:
            self._schedule_ui_update(self._cleanup_simulation)

    def _on_simulation_finished(self):
        """UI update when simulation completes."""
        self.status_pane.object = (
            f"**Completed:** {self.history.max_iterations} iterations, "
            f"Max reward: {max(self.history.rewards_max):.1f}"
        )

    def _check_display_available(self) -> bool:
        """Check if OpenGL/display is available."""
        import os

        # Check for DISPLAY environment variable
        if not os.environ.get('DISPLAY'):
            return False

        # Try to import pyglet (will fail if OpenGL not available)
        try:
            import pyglet
            # Try to create a headless context
            pyglet.options['headless'] = True
            return True
        except Exception:
            return False

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
            pn.pane.Markdown("### Environment"),
            pn.Param(
                self.param,
                parameters=["game_name", "obs_type"],
                widgets={
                    "game_name": pn.widgets.Select,
                    "obs_type": pn.widgets.RadioButtonGroup,
                },
            ),
            pn.pane.Markdown("### Algorithm"),
            pn.Param(
                self.param,
                parameters=["N", "dist_coef", "reward_coef", "use_cumulative_reward"],
            ),
            pn.pane.Markdown("### Simulation"),
            pn.Param(
                self.param,
                parameters=[
                    "max_iterations",
                    "record_frames",
                    "dt_range_min",
                    "dt_range_max",
                    "device",
                    "seed",
                ],
            ),
            pn.pane.Markdown("### Controls"),
            self.run_button,
            self.stop_button,
            self.progress_bar,
            self.status_pane,
            sizing_mode="stretch_width",
        )


class AtariGasVisualizer(param.Parameterized):
    """Visualization panel for Atari Gas results.

    Features:
    - Best walker frame display with playback
    - Cumulative reward progression curve
    - Real-time metric histograms
    """

    show_histograms = param.Boolean(default=True, doc="Show metric histograms")

    def __init__(self, history: AtariHistory | None = None, **params):
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

        # Frame display
        self.frame_pane = pn.pane.PNG(
            object=self._create_blank_frame(),
            width=640,
            height=840,
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

    def set_history(self, history: AtariHistory):
        """Load new history for visualization."""
        self.history = history

        # Update time player
        self.time_player.end = len(history.iterations) - 1
        self.time_player.value = 0
        self.time_player.disabled = False

        # Update info
        self.info_pane.object = (
            f"**Loaded:** {history.max_iterations} iterations | "
            f"Game: {history.game_name} | "
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
            width=600,
            height=300,
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
                width=250, height=200, title="Alive Walkers", xlabel="Count", color="green"
            )
            self.histogram_alive_pane.object = alive_hist

            # Cloning events histogram
            cloning_data = np.array(self.history.num_cloned[: idx + 1])
            cloning_hist = hv.Histogram(np.histogram(cloning_data, bins=20)).opts(
                width=250, height=200, title="Cloning Events", xlabel="Count", color="orange"
            )
            self.histogram_cloning_pane.object = cloning_hist

            # Virtual rewards histogram
            vr_data = np.array(self.history.virtual_rewards_mean[: idx + 1])
            vr_hist = hv.Histogram(np.histogram(vr_data, bins=30)).opts(
                width=250, height=200, title="Virtual Rewards", xlabel="Value", color="purple"
            )
            self.histogram_virtual_reward_pane.object = vr_hist

    def _create_blank_frame(self) -> bytes:
        """Create blank frame as placeholder."""
        if not PIL_AVAILABLE:
            return b""

        img = Image.new("RGB", (160, 210), color="black")
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
        histogram_row = (
            pn.Row(
                self.histogram_alive_pane,
                self.histogram_cloning_pane,
                self.histogram_virtual_reward_pane,
            )
            if self.show_histograms
            else None
        )

        return pn.Column(
            pn.pane.Markdown("## Atari Gas Visualization"),
            self.time_player,
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("### Best Walker Frame"),
                    self.frame_pane,
                ),
                pn.Column(
                    pn.pane.Markdown("### Reward Progression"),
                    self.reward_plot_pane,
                    pn.pane.Markdown("### Metrics Distribution"),
                    histogram_row,
                ),
            ),
            self.info_pane,
            sizing_mode="stretch_width",
        )


def create_app() -> pn.template.FastListTemplate:
    """Create Atari Gas Dashboard application."""
    hv.extension("bokeh")
    pn.extension()

    # Create components
    config_panel = AtariGasConfigPanel()
    visualizer = AtariGasVisualizer()

    # Connect callback
    def on_simulation_complete(history: AtariHistory):
        """Update visualizer when simulation completes."""
        visualizer.set_history(history)

    config_panel.add_completion_callback(on_simulation_complete)

    # Create layout
    return pn.template.FastListTemplate(
        title="Atari Fractal Gas Dashboard",
        sidebar=[config_panel.panel()],
        main=[visualizer.panel()],
        sidebar_width=400,
        main_max_width="100%",
    )


def _parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Atari Fractal Gas Dashboard")
    parser.add_argument("--port", type=int, default=5006, help="Port to run server on")
    parser.add_argument("--open", action="store_true", help="Open browser on launch")
    parser.add_argument("--threaded", action="store_true", help="Use multi-threaded Tornado (default: single-threaded for WSL compatibility)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print("Starting Atari Fractal Gas Dashboard...", flush=True)
    app = create_app()
    print(
        f"Atari Fractal Gas Dashboard running at http://localhost:{args.port}",
        flush=True,
    )
    if not args.threaded:
        print("Running in single-threaded mode (use --threaded for multi-threaded)", flush=True)
    app.show(port=args.port, open=args.open, threaded=args.threaded)
