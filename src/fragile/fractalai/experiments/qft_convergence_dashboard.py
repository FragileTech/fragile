"""QFT-focused dashboard for 3D swarm convergence visualization.

This dashboard is inspired by the parameter-selection sidebar used in the
gas visualization dashboard, but it focuses on QFT analysis with a 3D
Plotly-based swarm view.

Run:
    python -m fragile.fractalai.experiments.qft_convergence_dashboard
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import panel as pn
import param

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.experiments.gas_config_panel import GasConfigPanel

# Prevent Plotly from probing the system browser during import.
os.environ.setdefault("PLOTLY_RENDERER", "json")


__all__ = ["create_app"]


class SwarmConvergence3D(param.Parameterized):
    """3D swarm convergence viewer using Plotly backend."""

    point_size = param.Number(default=4, bounds=(1, 20), doc="Walker point size")
    point_alpha = param.Number(default=0.85, bounds=(0.05, 1.0), doc="Marker opacity")
    color_metric = param.ObjectSelector(
        default="constant",
        objects=["constant", "fitness", "reward", "radius"],
        doc="Color encoding for walkers",
    )
    fix_axes = param.Boolean(default=True, doc="Fix axis ranges to bounds extent")

    def __init__(self, history: RunHistory | None, bounds_extent: float = 10.0, **params):
        super().__init__(**params)
        self.history = history
        self.bounds_extent = float(bounds_extent)

        self._x = None
        self._fitness = None
        self._rewards = None
        self._alive = None

        self.time_player = pn.widgets.Player(
            name="frame",
            start=0,
            end=0,
            value=0,
            step=1,
            interval=150,
            loop_policy="loop",
        )
        self.time_player.disabled = True
        self.time_player.sizing_mode = "stretch_width"
        self.time_player.param.watch(self._sync_frame, "value")

        self.status_pane = pn.pane.Markdown(
            "**Status:** Load or run a simulation to view 3D convergence.",
            sizing_mode="stretch_width",
        )

        self.plot_pane = pn.pane.Plotly(
            self._make_figure(0),
            sizing_mode="stretch_width",
            height=720,
        )

        self.param.watch(self._refresh_frame, ["point_size", "point_alpha", "color_metric"])

        if history is not None:
            self.set_history(history)

    def set_history(self, history: RunHistory):
        """Load RunHistory data into the viewer."""
        self.history = history
        self._x = history.x_final.detach().cpu().numpy()
        self._fitness = history.fitness.detach().cpu().numpy()
        self._rewards = history.rewards.detach().cpu().numpy()
        self._alive = history.alive_mask.detach().cpu().numpy().astype(bool)

        self.time_player.end = max(0, history.n_recorded - 1)
        self.time_player.value = 0
        self.time_player.disabled = False

        self.status_pane.object = (
            f"**RunHistory loaded:** N={history.N}, "
            f"steps={history.n_steps}, "
            f"recorded={history.n_recorded}"
        )
        self._refresh_frame()

    def _sync_frame(self, event):
        if self.history is None:
            return
        self._update_plot(int(np.clip(event.new, 0, self.history.n_recorded - 1)))

    def _refresh_frame(self, *_):
        if self.history is None:
            return
        self._update_plot(int(np.clip(self.time_player.value, 0, self.history.n_recorded - 1)))

    def _get_alive_mask(self, frame: int) -> np.ndarray:
        if self._alive is None:
            return np.ones(self._x.shape[1], dtype=bool)
        if frame == 0:
            return np.ones(self._x.shape[1], dtype=bool)
        idx = min(frame - 1, self._alive.shape[0] - 1)
        return self._alive[idx]

    def _frame_title(self, frame: int) -> str:
        if self.history is None:
            return "QFT Swarm Convergence"
        step = self.history.recorded_steps[frame]
        return f"QFT Swarm Convergence (frame {frame}, step {step})"

    def _make_figure(self, frame: int):
        import plotly.graph_objects as go

        if self.history is None or self._x is None:
            fig = go.Figure()
            fig.update_layout(
                title="QFT Swarm Convergence",
                height=720,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            return fig

        positions = self._x[frame]
        if positions.shape[1] < 3:
            msg = "Need at least 3 dimensions for 3D convergence visualization."
            raise ValueError(msg)

        alive = self._get_alive_mask(frame)
        positions = positions[alive][:, :3]

        if positions.size == 0:
            fig = go.Figure()
            fig.update_layout(title=self._frame_title(frame), height=720)
            return fig

        colors = "#1f77b4"
        colorbar = None
        showscale = False

        if self.color_metric != "constant" and frame > 0:
            if self.color_metric == "fitness":
                colors = self._fitness[frame - 1][alive]
                colorbar = dict(title="fitness")
                showscale = True
            elif self.color_metric == "reward":
                colors = self._rewards[frame - 1][alive]
                colorbar = dict(title="reward")
                showscale = True
            elif self.color_metric == "radius":
                colors = np.linalg.norm(positions, axis=1)
                colorbar = dict(title="radius")
                showscale = True

        scatter = go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            marker=dict(
                size=self.point_size,
                color=colors,
                colorscale="Viridis" if showscale else None,
                opacity=self.point_alpha,
                showscale=showscale,
                colorbar=colorbar,
            ),
        )

        fig = go.Figure(data=[scatter])
        fig.update_layout(
            title=self._frame_title(frame),
            height=720,
            margin=dict(l=0, r=0, t=40, b=0),
        )

        if self.fix_axes:
            extent = float(self.bounds_extent)
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[-extent, extent]),
                    yaxis=dict(range=[-extent, extent]),
                    zaxis=dict(range=[-extent, extent]),
                )
            )

        return fig

    def _update_plot(self, frame: int):
        self.plot_pane.object = self._make_figure(frame)

    def panel(self) -> pn.Column:
        """Return the Panel layout for the 3D convergence viewer."""
        return pn.Column(
            pn.pane.Markdown("## 3D Swarm Convergence (Plotly)"),
            self.time_player,
            pn.Spacer(height=10),
            pn.layout.Divider(),
            self.plot_pane,
            self.status_pane,
            sizing_mode="stretch_width",
        )


def create_app() -> pn.template.FastListTemplate:
    """Create the QFT convergence dashboard."""
    debug = os.environ.get("QFT_DASH_DEBUG", "").lower() in {"1", "true", "yes"}
    skip_sidebar = os.environ.get("QFT_DASH_SKIP_SIDEBAR", "").lower() in {"1", "true", "yes"}
    skip_visual = os.environ.get("QFT_DASH_SKIP_VIS", "").lower() in {"1", "true", "yes"}

    def _debug(msg: str):
        if debug:
            print(f"[qft-dashboard] {msg}", flush=True)

    sidebar = pn.Column(
        pn.pane.Markdown("## QFT Analysis"),
        pn.pane.Markdown("Starting dashboard..."),
        sizing_mode="stretch_width",
    )
    main = pn.Column(
        pn.pane.Markdown("Loading visualization..."),
        sizing_mode="stretch_both",
    )

    template = pn.template.FastListTemplate(
        title="QFT Swarm Convergence Dashboard",
        sidebar=[sidebar],
        main=[main],
        sidebar_width=435,
        main_max_width="100%",
    )

    def _build_ui():
        start_total = time.time()
        _debug("initializing extensions")
        pn.extension("plotly")

        _debug("building config + visualizer")
        start = time.time()
        gas_config = GasConfigPanel.create_qft_config(dims=3, bounds_extent=10.0)
        visualizer = SwarmConvergence3D(history=None, bounds_extent=gas_config.bounds_extent)
        _debug(f"config+visualizer ready ({time.time() - start:.2f}s)")

        _debug("setting up history controls")
        repo_root = Path(__file__).resolve().parents[4]
        qft_run_id = "qft_penalty_thr0p75_pen0p9_m354_ed2p80_nu1p10_N200_long"
        qft_history_path = (
            repo_root / "outputs" / "qft_calibrated" / f"{qft_run_id}_history.pt"
        )
        qft_history_dir = qft_history_path.parent
        qft_history_dir.mkdir(parents=True, exist_ok=True)
        history_dir = qft_history_dir
        history_path_input = pn.widgets.TextInput(
            name="QFT RunHistory path",
            value=str(qft_history_path),
            width=335,
            sizing_mode="stretch_width",
        )
        browse_button = pn.widgets.Button(
            name="Browse files...",
            button_type="default",
            width=335,
            sizing_mode="stretch_width",
        )
        file_selector_container = pn.Column(sizing_mode="stretch_width")

        load_button = pn.widgets.Button(
            name="Load RunHistory",
            button_type="primary",
            width=335,
            sizing_mode="stretch_width",
        )
        load_status = pn.pane.Markdown(
            "**Load a history**: paste a *_history.pt path or browse and click Load.",
            sizing_mode="stretch_width",
        )

        def on_simulation_complete(history: RunHistory):
            visualizer.bounds_extent = float(gas_config.bounds_extent)
            visualizer.set_history(history)

        def _infer_bounds_extent(history: RunHistory) -> float | None:
            if history.bounds is None:
                return None
            high = history.bounds.high.detach().cpu().abs().max().item()
            low = history.bounds.low.detach().cpu().abs().max().item()
            return float(max(high, low))

        def _sync_history_path(value):
            if value:
                history_path_input.value = str(value[0])

        def _ensure_file_selector() -> pn.widgets.FileSelector:
            if file_selector_container.objects:
                return file_selector_container.objects[0]
            selector = pn.widgets.FileSelector(
                name="Select RunHistory",
                directory=str(history_dir),
                file_pattern="*_history.pt",
                only_files=True,
            )
            if qft_history_path.exists():
                selector.value = [str(qft_history_path)]
            selector.param.watch(lambda e: _sync_history_path(e.new), "value")
            file_selector_container.objects = [selector]
            return selector

        def _on_browse_clicked(_):
            _ensure_file_selector()

        def on_load_clicked(_):
            history_path = Path(history_path_input.value).expanduser()
            if not history_path.exists():
                load_status.object = "**Error:** History path does not exist."
                return
            try:
                history = RunHistory.load(str(history_path))
                inferred_extent = _infer_bounds_extent(history)
                if inferred_extent is not None:
                    visualizer.bounds_extent = inferred_extent
                    gas_config.bounds_extent = inferred_extent
                visualizer.set_history(history)
                load_status.object = f"**Loaded:** `{history_path}`"
            except Exception as exc:
                load_status.object = f"**Error loading history:** {exc!s}"

        browse_button.on_click(_on_browse_clicked)
        load_button.on_click(on_load_clicked)

        gas_config.add_completion_callback(on_simulation_complete)

        def on_bounds_change(event):
            visualizer.bounds_extent = float(event.new)
            visualizer._refresh_frame()

        gas_config.param.watch(on_bounds_change, "bounds_extent")

        _debug("building sidebar + main panels")
        if skip_sidebar:
            sidebar.objects = [
                pn.pane.Markdown(
                    "## QFT Analysis\n"
                    "Sidebar disabled via QFT_DASH_SKIP_SIDEBAR=1."
                ),
                pn.pane.Markdown("### Load QFT RunHistory"),
                history_path_input,
                browse_button,
                file_selector_container,
                load_button,
                load_status,
            ]
        else:
            start = time.time()
            sidebar.objects = [
                pn.pane.Markdown(
                    "## QFT Analysis\n"
                    "Use the calibrated QFT parameters in the sidebar to run a simulation, "
                    "then inspect swarm convergence in 3D."
                ),
                pn.pane.Markdown("### Load QFT RunHistory"),
                history_path_input,
                browse_button,
                file_selector_container,
                load_button,
                load_status,
                gas_config.panel(),
            ]
            _debug(f"sidebar ready ({time.time() - start:.2f}s)")

        if skip_visual:
            main.objects = [
                pn.pane.Markdown(
                    "Visualization disabled via QFT_DASH_SKIP_VIS=1.",
                    sizing_mode="stretch_both",
                )
            ]
        else:
            start = time.time()
            main.objects = [visualizer.panel()]
            _debug(f"main panel ready ({time.time() - start:.2f}s)")

        _debug(f"ui ready ({time.time() - start_total:.2f}s)")

    pn.state.onload(_build_ui)
    return template


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5007)
    parser.add_argument("--open", action="store_true", help="Open browser on launch")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print("Starting QFT Swarm Convergence Dashboard...", flush=True)
    app = create_app()
    print(
        f"QFT Swarm Convergence Dashboard running at http://localhost:{args.port} "
        f"(use --open to launch a browser)",
        flush=True,
    )
    app.show(port=args.port, open=args.open)
