"""
Interactive visualization panel for the Atari Gas algorithm.

This module adapts the Montezuma visualization tooling to the Atari Gas setting.
Instead of rendering the full pyramid layout, it focuses on the walker with the
highest cumulative reward at each step and streams its screen together with basic
telemetry.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import torch

from fragile.atari_gas import AtariGas, AtariSwarmState
from fragile.shaolin.stream_plots import Curve, RGB


hv.extension("bokeh")
pn.extension("tabulator")


class AtariGasDisplay:
    """Streamed visualization of the best-performing Atari walker.

    The display shows:
    - Best walker's screen (RGB frame)
    - Cumulative reward over time
    - Summary table with step info including:
        - dt: number of times the action was applied consecutively
    """

    SUMMARY_COLUMNS = [
        "step",
        "walker",
        "step_reward",
        "cumulative_reward",
        "action",
        "dt",  # Number of times action was applied consecutively
        "done",
        "truncated",
    ]

    def __init__(
        self,
        frame_shape: tuple[int, int, int] | None = (210, 160, 3),
        reward_history: int = 5000,
        frame_opts: dict[str, Any] | None = None,
        curve_opts: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            frame_shape: Initial guess for image shape used to clear the display.
            reward_history: Maximum number of points kept in the reward history plot.
            frame_opts: Extra options passed to the RGB plot.
            curve_opts: Extra options passed to the reward curve plot.
        """
        self.frame_shape = frame_shape
        frame_opts = frame_opts or {}
        curve_opts = curve_opts or {}

        height, width = self._default_frame_size()
        self.best_frame = RGB(
            bokeh_opts={"height": height, "width": width, "toolbar": None, **frame_opts},
        )
        self.reward_curve = Curve(
            data=pd.DataFrame({"step": [], "reward": []}),
            buffer_length=reward_history,
            data_names=("step", "reward"),
            bokeh_opts={
                "height": 220,
                "width": 420,
                "line_width": 3,
                "color": "#ffbf00",
                "xlabel": "Step",
                "ylabel": "Best cumulative reward",
                "tools": ["hover"],
                **curve_opts,
            },
        )
        self.summary_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=self.SUMMARY_COLUMNS),
            height=140,
            selectable=False,
            disabled=True,
        )
        self.layout = pn.Column(
            pn.pane.Markdown("### Best Walker Screen"),
            self.best_frame.plot,
            pn.pane.Markdown("### Best Reward History"),
            self.reward_curve.plot,
            pn.pane.Markdown("### Latest Snapshot"),
            self.summary_table,
        )

    def reset(self, frame: np.ndarray | None = None) -> None:
        """Clear plots and optionally seed with a starting frame."""
        self.reward_curve.data_stream.clear()
        self.summary_table.value = pd.DataFrame(columns=self.SUMMARY_COLUMNS)
        if frame is None and self.frame_shape is not None:
            frame = np.zeros(self.frame_shape, dtype=np.uint8)
        if frame is not None:
            self._update_frame(frame)

    def update(
        self,
        state: AtariSwarmState,
        cumulative_rewards: torch.Tensor,
        step: int,
        frame_override: np.ndarray | None = None,
    ) -> None:
        """Stream latest data for the best walker."""
        rewards_cpu = cumulative_rewards.detach().cpu().numpy()
        best_ix = int(rewards_cpu.argmax()) if rewards_cpu.size else 0

        if frame_override is not None:
            frame = frame_override
        else:
            frame = self._extract_frame(state, best_ix)

        if frame is not None:
            self._update_frame(frame)

        best_reward = float(rewards_cpu[best_ix]) if rewards_cpu.size else 0.0
        self.reward_curve.send(pd.DataFrame({"step": [step], "reward": [best_reward]}))

        info_row = self._build_summary_row(state, best_ix, best_reward, step)
        self.summary_table.value = info_row

    def _default_frame_size(self) -> tuple[int, int]:
        if self.frame_shape is None:
            return 210, 160
        height, width = self.frame_shape[:2]
        return int(height), int(width)

    def _update_frame(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        self.frame_shape = frame.shape
        self.best_frame.send(frame)

    def _extract_frame(self, state: AtariSwarmState, best_ix: int) -> np.ndarray | None:
        info = state.infos[best_ix] if best_ix < len(state.infos) else None
        if isinstance(info, dict) and "rgb" in info:
            return np.asarray(info["rgb"])
        observation = state.observations[best_ix].detach().cpu().numpy()
        if observation.ndim == 1:
            side = int(np.sqrt(observation.size))
            if side * side == observation.size:
                observation = observation.reshape(side, side)
            else:
                return None
        if observation.ndim == 2:
            observation = np.stack([observation] * 3, axis=-1)
        if observation.ndim == 3 and observation.shape[0] in {1, 3}:
            observation = np.transpose(observation, (1, 2, 0))
        if observation.ndim != 3 or observation.shape[-1] not in {1, 3}:
            return None
        if observation.shape[-1] == 1:
            observation = np.repeat(observation, 3, axis=-1)
        observation = observation.astype(np.float32)
        observation = np.clip(observation, 0, 255)
        return observation.astype(np.uint8)

    def _build_summary_row(
        self,
        state: AtariSwarmState,
        best_ix: int,
        best_reward: float,
        step: int,
    ) -> pd.DataFrame:
        actions = state.actions.detach().cpu().numpy()
        dts = state.dts.detach().cpu().numpy()
        dones = state.dones.detach().cpu().numpy()
        truncated = state.truncated.detach().cpu().numpy()
        step_summary = pd.DataFrame({
            "step": [step],
            "walker": [best_ix],
            "step_reward": [float(state.step_rewards[best_ix].detach().cpu().item())],
            "cumulative_reward": [best_reward],
            "action": [int(actions[best_ix]) if actions.size else -1],
            "dt": [int(dts[best_ix]) if dts.size else 1],
            "done": [bool(dones[best_ix]) if dones.size else False],
            "truncated": [bool(truncated[best_ix]) if truncated.size else False],
        })
        return step_summary[self.SUMMARY_COLUMNS]


class AtariGasRunner(param.Parameterized):
    """Panel controller that advances an Atari Gas simulation and updates plots."""

    is_running = param.Boolean(default=False)

    def __init__(
        self,
        gas: AtariGas,
        n_steps: int = 10_000,
        display: AtariGasDisplay | None = None,
        table_history: int = 200,
    ) -> None:
        super().__init__()
        self.gas = gas
        self.n_steps = n_steps
        self.display = display or AtariGasDisplay()
        self.table_history = table_history

        self.sleep_input = pn.widgets.FloatInput(name="Sleep (s)", value=0.0, width=90)
        self.reset_btn = pn.widgets.Button(name="Reset", button_type="primary")
        self.play_btn = pn.widgets.Button(name="Play", button_type="success")
        self.pause_btn = pn.widgets.Button(name="Pause", button_type="warning", disabled=True)
        self.step_btn = pn.widgets.Button(name="Step", button_type="primary")
        self.progress = pn.indicators.Progress(
            name="Progress",
            value=0,
            max=n_steps,
            width=420,
            bar_color="primary",
        )

        self.summary_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=AtariGasDisplay.SUMMARY_COLUMNS),
            height=220,
            selectable=False,
            disabled=True,
        )

        self._thread: threading.Thread | None = None
        self._history: list[dict[str, Any]] = []
        self.state: AtariSwarmState | None = None
        self.cumulative_reward: torch.Tensor | None = None
        self.iteration = 0

        self.reset_btn.on_click(self._on_reset)
        self.play_btn.on_click(self._on_play)
        self.pause_btn.on_click(self._on_pause)
        self.step_btn.on_click(self._on_step)

        self._initialize_state()

    # ------------------------------------------------------------------
    # Panel integration
    # ------------------------------------------------------------------
    def __panel__(self) -> pn.Column:
        controls = pn.Row(
            self.play_btn,
            self.pause_btn,
            self.reset_btn,
            self.step_btn,
            self.sleep_input,
        )
        return pn.Column(
            controls,
            self.progress,
            pn.Row(self.display.layout, self.summary_table),
        )

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------
    def _on_reset(self, _event) -> None:
        self._stop_thread()
        self._initialize_state()

    def _on_play(self, _event) -> None:
        if self.is_running or self.iteration >= self.n_steps:
            return
        self.is_running = True
        self.play_btn.disabled = True
        self.pause_btn.disabled = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _on_pause(self, _event) -> None:
        self.is_running = False
        self.play_btn.disabled = False
        self.pause_btn.disabled = True

    def _on_step(self, _event) -> None:
        self._advance_simulation()

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------
    def _initialize_state(self) -> None:
        self.iteration = 0
        self.progress.value = 0
        self.progress.bar_color = "primary"

        self.state = self.gas.initialize_state()
        self.cumulative_reward = self.state.rewards.clone()

        self._history.clear()
        start_frame = None
        if hasattr(self.gas.env, "get_image"):
            try:
                start_frame = np.asarray(self.gas.env.get_image())
            except Exception:  # pragma: no cover - defensive
                start_frame = None
        self.display.reset(frame=start_frame)
        self.summary_table.value = pd.DataFrame(columns=AtariGasDisplay.SUMMARY_COLUMNS)
        self.is_running = False
        self.play_btn.disabled = False
        self.pause_btn.disabled = True

    def _advance_simulation(self) -> bool:
        if self.state is None or self.cumulative_reward is None:
            return False
        if self.iteration >= self.n_steps:
            self.progress.bar_color = "success"
            return False

        _, next_state, _companions = self.gas.step(self.state)
        self.state = next_state
        self.cumulative_reward = self.state.rewards.clone()

        self.iteration += 1
        self.progress.value = self.iteration
        if self.iteration >= self.n_steps:
            self.progress.bar_color = "success"

        self.display.update(self.state, self.cumulative_reward, self.iteration)
        self._record_snapshot()
        return True

    def _record_snapshot(self) -> None:
        if self.state is None or self.cumulative_reward is None:
            return
        rewards_cpu = self.cumulative_reward.detach().cpu().numpy()
        best_ix = int(rewards_cpu.argmax()) if rewards_cpu.size else 0
        record = {
            "step": self.iteration,
            "walker": best_ix,
            "step_reward": float(self.state.step_rewards[best_ix].detach().cpu().item())
            if self.state.step_rewards.numel()
            else 0.0,
            "cumulative_reward": float(rewards_cpu[best_ix]) if rewards_cpu.size else 0.0,
            "action": int(self.state.actions[best_ix].detach().cpu().item())
            if self.state.actions.numel()
            else -1,
            "dt": int(self.state.dts[best_ix].detach().cpu().item())
            if self.state.dts.numel()
            else 1,
            "done": bool(self.state.dones[best_ix].detach().cpu().item())
            if self.state.dones.numel()
            else False,
            "truncated": bool(self.state.truncated[best_ix].detach().cpu().item())
            if self.state.truncated.numel()
            else False,
        }
        self._history.append(record)
        self._history = self._history[-self.table_history :]
        self.summary_table.value = pd.DataFrame(self._history)[AtariGasDisplay.SUMMARY_COLUMNS]

    def _run_loop(self) -> None:
        try:
            while self.is_running and self.iteration < self.n_steps:
                if not self._advance_simulation():
                    break
                sleep_time = max(float(self.sleep_input.value), 0.0)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self.is_running = False
            self.play_btn.disabled = False
            self.pause_btn.disabled = True

    def _stop_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            self.is_running = False
            self._thread.join(timeout=0.5)
        self._thread = None
