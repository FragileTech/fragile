"""QFT-focused dashboard for 3D swarm convergence visualization.

This dashboard is inspired by the parameter-selection sidebar used in the
gas visualization dashboard, but it focuses on QFT analysis with a 3D
Plotly-based swarm view.

Run:
    python -m fragile.fractalai.experiments.qft_convergence_dashboard
"""

from __future__ import annotations

import argparse
from pathlib import Path

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.experiments.gas_config_panel import GasConfigPanel


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

        self.frame_stream = hv.streams.Stream.define("Frame", frame=0)()
        self.dmap = hv.DynamicMap(self._render_frame, streams=[self.frame_stream])
        self.plot_pane = pn.panel(self.dmap, sizing_mode="stretch_both", height=720)

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
        frame = int(np.clip(event.new, 0, self.history.n_recorded - 1))
        self.frame_stream.event(frame=frame)

    def _refresh_frame(self, *_):
        if self.history is None:
            return
        frame = int(np.clip(self.time_player.value, 0, self.history.n_recorded - 1))
        self.frame_stream.event(frame=frame)

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

    def _render_frame(self, frame: int):
        if self.history is None or self._x is None:
            return hv.Scatter3D([], kdims=["x", "y", "z"], vdims=[])

        positions = self._x[frame]
        if positions.shape[1] < 3:
            msg = "Need at least 3 dimensions for 3D convergence visualization."
            raise ValueError(msg)

        alive = self._get_alive_mask(frame)
        positions = positions[alive][:, :3]

        if positions.size == 0:
            return hv.Scatter3D([], kdims=["x", "y", "z"], vdims=[])

        data = {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
        }

        vdims = []
        if self.color_metric != "constant" and frame > 0:
            if self.color_metric == "fitness":
                metric_values = self._fitness[frame - 1][alive]
                data["fitness"] = metric_values
                vdims = ["fitness"]
            elif self.color_metric == "reward":
                metric_values = self._rewards[frame - 1][alive]
                data["reward"] = metric_values
                vdims = ["reward"]
            elif self.color_metric == "radius":
                radii = np.linalg.norm(positions, axis=1)
                data["radius"] = radii
                vdims = ["radius"]

        df = pd.DataFrame(data)
        scatter = hv.Scatter3D(df, kdims=["x", "y", "z"], vdims=vdims)

        opts = {
            "size": self.point_size,
            "alpha": self.point_alpha,
            "title": self._frame_title(frame),
            "width": 900,
            "height": 720,
        }
        if vdims:
            opts.update({"color": vdims[0], "cmap": "Viridis", "colorbar": True})
        else:
            opts.update({"color": "#1f77b4"})

        if self.fix_axes:
            extent = float(self.bounds_extent)
            opts.update(
                {
                    "xlim": (-extent, extent),
                    "ylim": (-extent, extent),
                    "zlim": (-extent, extent),
                }
            )

        return scatter.opts(**opts)

    def panel(self) -> pn.Column:
        """Return the Panel layout for the 3D convergence viewer."""
        return pn.Column(
            pn.pane.Markdown("## 3D Swarm Convergence (Plotly)"),
            self.time_player,
            self.plot_pane,
            self.status_pane,
            sizing_mode="stretch_both",
        )


def create_app() -> pn.template.FastListTemplate:
    """Create the QFT convergence dashboard."""
    hv.extension("plotly")
    pn.extension("plotly")

    gas_config = GasConfigPanel.create_qft_config(dims=3, bounds_extent=10.0)
    visualizer = SwarmConvergence3D(history=None, bounds_extent=gas_config.bounds_extent)

    repo_root = Path(__file__).resolve().parents[4]
    qft_run_id = "qft_penalty_thr0p75_pen0p9_m354_ed2p80_nu1p10_N200_long"
    qft_history_path = (
        repo_root / "outputs" / "qft_calibrated" / f"{qft_run_id}_history.pt"
    )
    history_dir_candidates = [
        qft_history_path.parent,
        repo_root / "outputs",
        repo_root,
    ]
    history_dir = next((p for p in history_dir_candidates if p.exists()), Path("."))
    history_selector = pn.widgets.FileSelector(
        name="QFT RunHistory",
        directory=str(history_dir),
        file_pattern="*_history.pt",
        only_files=True,
    )
    if qft_history_path.exists():
        history_selector.value = [str(qft_history_path)]

    load_button = pn.widgets.Button(
        name="Load RunHistory",
        button_type="primary",
        sizing_mode="stretch_width",
    )
    load_status = pn.pane.Markdown(
        "**Load a history**: select a *_history.pt file and click Load.",
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

    def on_load_clicked(_):
        if not history_selector.value:
            load_status.object = "**Error:** Please select a RunHistory file."
            return
        history_path = Path(history_selector.value[0])
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

    load_button.on_click(on_load_clicked)

    gas_config.add_completion_callback(on_simulation_complete)

    def on_bounds_change(event):
        visualizer.bounds_extent = float(event.new)
        visualizer._refresh_frame()

    gas_config.param.watch(on_bounds_change, "bounds_extent")

    return pn.template.FastListTemplate(
        title="QFT Swarm Convergence Dashboard",
        sidebar=[
            pn.pane.Markdown(
                "## QFT Analysis\n"
                "Use the calibrated QFT parameters in the sidebar to run a simulation, "
                "then inspect swarm convergence in 3D."
            ),
            pn.pane.Markdown("### Load QFT RunHistory"),
            history_selector,
            load_button,
            load_status,
            gas_config.panel(),
        ],
        main=[visualizer.panel()],
        sidebar_width=400,
        main_max_width="100%",
    )


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
