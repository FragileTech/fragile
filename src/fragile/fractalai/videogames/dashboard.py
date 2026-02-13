"""Dashboard for Atari Fractal Gas visualization."""

import atexit
import io
import threading
import time
from typing import Callable

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


def _configure_headless_wsl(force: bool = False) -> bool:
    """Auto-detect WSL and set headless env vars for Atari/OpenGL.

    Sets PYGLET_HEADLESS, LIBGL_ALWAYS_SOFTWARE, SDL_VIDEODRIVER, MPLBACKEND
    so that pyglet/OpenGL/SDL work without a display server.

    Args:
        force: If True, set vars regardless of WSL detection.

    Returns:
        True if headless vars were configured.
    """
    import os

    if not force:
        try:
            with open("/proc/version") as f:
                proc_version = f.read()
        except OSError:
            return False
        if "microsoft" not in proc_version.lower():
            return False

    headless_vars = {
        "PYGLET_HEADLESS": "1",
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "SDL_VIDEODRIVER": "dummy",
        "MPLBACKEND": "Agg",
    }
    configured = False
    for key, value in headless_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            configured = True
    # Unset DISPLAY to prevent any X11/XCB connections (WSLg sets it but
    # the X server may be broken or trigger threading errors in ALE/SDL).
    if os.environ.pop("DISPLAY", None):
        configured = True
    if configured:
        print(f"Headless mode: auto-configured env vars for {'WSL' if not force else 'headless'}", flush=True)
    return True


def _parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a semantic version string into a comparable tuple."""
    parts = version.split(".")
    parsed: list[int] = []
    for part in parts[:3]:
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        parsed.append(int(digits) if digits else 0)
    while len(parsed) < 3:
        parsed.append(0)
    return tuple(parsed)


def _format_duration(seconds: float) -> str:
    """Format seconds as ``H:MM:SS`` or ``M:SS``."""
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class AtariGasConfigPanel(param.Parameterized):
    """Configuration panel for Atari Gas algorithm.

    Provides parameter controls, simulation execution, and callback system.
    """

    # Environment parameters
    game_name = param.ObjectSelector(
        default="ALE/Pong-v5",
        objects=[
            "ALE/Pong-v5",
            "ALE/Breakout-v5",
            "ALE/MsPacman-v5",
            "ALE/SpaceInvaders-v5",
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

    n_elite = param.Integer(
        default=0, bounds=(0, 50), doc="Number of elite walkers to preserve (0=disabled)"
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

    n_workers = param.Integer(default=1, bounds=(1, 16), doc="Parallel env workers (1=serial)")

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
        self.gas: AtariFractalGas | None = None
        self.history: AtariHistory | None = None
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
            "**Note:** Uses plangym Atari environment backend (headless compatible)."
        )

        # Callbacks
        self._on_simulation_complete: list[Callable[[AtariHistory], None]] = []

    def add_completion_callback(self, callback: Callable[[AtariHistory], None]):
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
        blocking the Tornado event loop. Headless env vars (auto-set on WSL)
        eliminate X11/XCB threading concerns.
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
        """Create Atari environment.

        Returns:
            plangym Atari environment compatible with AtariFractalGas

        Raises:
            Exception: If environment creation fails
        """
        # Ensure headless env vars are set before plangym spawns worker processes.
        # When n_workers > 1, ExternalProcess forks inherit these vars.
        _configure_headless_wsl()
        print("[worker] importing plangym...", flush=True)
        try:
            import plangym
        except ImportError as e:
            msg = (
                "plangym is required for Atari dashboard.\n"
                "Install with: uv add \"plangym[atari]==0.1.32\""
            )
            raise ImportError(msg) from e

        version = getattr(plangym, "__version__", "0.0.0")
        print(f"[worker] plangym {version} imported", flush=True)
        if _parse_semver(version) < (0, 1, 32):
            raise RuntimeError(
                f"plangym>=0.1.32 is required (found {version}). "
                "Run: uv add \"plangym[atari]==0.1.32\""
            )

        # Explicit frameskip=5 is required: VectorizedEnv defaults to
        # frameskip=1, overriding AtariEnv's default of 5. Without this,
        # parallel workers step 5x fewer ALE frames per dt, producing
        # near-zero rewards and suspiciously fast iterations.
        kwargs = dict(
            name=self.game_name,
            obs_type=self.obs_type,
            frameskip=5,
            render_mode=None,
            autoreset=False,
            remove_time_limit=True,
            episodic_life=True,
        )
        if self.n_workers > 1:
            kwargs["n_workers"] = self.n_workers
        print(f"[worker] calling plangym.make({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})...", flush=True)
        env = plangym.make(**kwargs)
        print(f"[worker] plangym.make() returned: {type(env).__name__}", flush=True)
        self._schedule_ui_update(
            lambda v=version: setattr(self.status_pane, "object", f"Using plangym {v}")
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

            print("[worker] creating AtariFractalGas...", flush=True)
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
                n_elite=self.n_elite,
            )
            print("[worker] AtariFractalGas created, calling reset()...", flush=True)

            # Run simulation with progress updates
            state = self.gas.reset()
            print("[worker] reset() done, starting iteration loop...", flush=True)
            infos = []
            t_start = time.monotonic()

            # Optional tree history recording
            tree = None
            if self.use_tree_history:
                from fragile.fractalai.videogames.atari_tree_history import AtariTreeHistory

                tree = AtariTreeHistory(
                    N=self.N, game_name=self.game_name, max_iterations=self.max_iterations,
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

                # Update progress bar and text
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

            # Build history
            if tree is not None:
                self.tree_history = tree
                self.history = tree.to_atari_history()
            else:
                self.tree_history = None
                self.history = AtariHistory.from_run(infos, state, self.N, self.game_name)

            # Update UI
            if not self._stop_requested:
                self._schedule_ui_update(self._on_simulation_finished)

                # Notify callbacks
                for callback in self._on_simulation_complete:
                    callback(self.history)

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
            self._close_env()
            self._schedule_ui_update(self._cleanup_simulation)

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
                parameters=["N", "dist_coef", "reward_coef", "use_cumulative_reward", "n_elite"],
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
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            ),
            histogram_section,
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
    parser.add_argument("--headless", action="store_true", help="Force headless env vars (auto-detected on WSL)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _configure_headless_wsl(force=args.headless)
    print("Starting Atari Fractal Gas Dashboard...", flush=True)
    app = create_app()
    if not args.threaded:
        print("Running in single-threaded mode (use --threaded for multi-threaded)", flush=True)
    print(
        f"Open http://localhost:{args.port} in your browser (Ctrl+C to stop)",
        flush=True,
    )
    pn.serve({"/": app}, port=args.port, show=args.open, threaded=args.threaded)
