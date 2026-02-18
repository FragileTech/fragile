"""Comprehensive tests for DM Control (Robotics) Gas dashboard features."""

import io
import os
from unittest.mock import MagicMock, patch

import holoviews as hv
import numpy as np
import panel as pn
import pytest


hv.extension("bokeh")
pn.extension()

from fragile.fractalai.planning_gas import PlanningHistory, PlanningTrajectory
from fragile.fractalai.robots.dashboard import (
    _configure_mujoco_offscreen,
    _format_duration,
    create_app,
    RoboticGasConfigPanel,
    RoboticGasVisualizer,
)
from fragile.fractalai.robots.robotic_history import RoboticHistory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_history(n_iters: int = 10, with_frames: bool = False, task: str = "cartpole-balance"):
    """Build a minimal RoboticHistory for testing."""
    frames = (
        [np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8) for _ in range(n_iters)]
        if with_frames
        else [None] * n_iters
    )
    return RoboticHistory(
        iterations=list(range(n_iters)),
        rewards_mean=[float(i) * 0.1 for i in range(n_iters)],
        rewards_max=[float(i) * 0.2 for i in range(n_iters)],
        rewards_min=[0.0] * n_iters,
        alive_counts=[30] * n_iters,
        num_cloned=[3] * n_iters,
        virtual_rewards_mean=[0.5 + 0.05 * i for i in range(n_iters)],
        virtual_rewards_max=[1.0 + 0.05 * i for i in range(n_iters)],
        virtual_rewards_min=[0.2 + 0.05 * i for i in range(n_iters)],
        dt_mean=[1.0] * n_iters,
        dt_min=[1] * n_iters,
        dt_max=[1] * n_iters,
        best_frames=frames,
        best_rewards=[float(i) * 0.2 for i in range(n_iters)],
        best_indices=[0] * n_iters,
        N=30,
        max_iterations=n_iters,
        task_name=task,
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestFormatDuration:
    def test_seconds_only(self):
        assert _format_duration(45) == "0:45"

    def test_minutes_and_seconds(self):
        assert _format_duration(125) == "2:05"

    def test_hours(self):
        assert _format_duration(3661) == "1:01:01"

    def test_zero(self):
        assert _format_duration(0) == "0:00"


class TestConfigureMujocoOffscreen:
    def test_sets_mujoco_gl(self):
        saved = os.environ.pop("MUJOCO_GL", None)
        try:
            _configure_mujoco_offscreen()
            assert os.environ["MUJOCO_GL"] == "osmesa"
        finally:
            if saved is not None:
                os.environ["MUJOCO_GL"] = saved
            else:
                os.environ.pop("MUJOCO_GL", None)

    def test_does_not_overwrite_existing(self):
        saved = os.environ.get("MUJOCO_GL")
        os.environ["MUJOCO_GL"] = "egl"
        try:
            _configure_mujoco_offscreen()
            assert os.environ["MUJOCO_GL"] == "egl"
        finally:
            if saved is not None:
                os.environ["MUJOCO_GL"] = saved
            else:
                os.environ.pop("MUJOCO_GL", None)


# ---------------------------------------------------------------------------
# RoboticGasConfigPanel
# ---------------------------------------------------------------------------


class TestRoboticGasConfigPanel:
    def test_default_parameters(self):
        panel = RoboticGasConfigPanel()
        assert panel.algorithm_mode == "Single Loop"
        assert panel.task_name == "walker-walk"
        assert panel.render_width == 480
        assert panel.render_height == 480
        assert panel.N == 100
        assert panel.dist_coef == 1.0
        assert panel.reward_coef == 1.0
        assert panel.n_elite == 5
        assert panel.dt_range_min == 1
        assert panel.dt_range_max == 1
        assert panel.max_iterations == 100
        assert panel.seed == 42
        assert panel.n_workers == 1
        assert panel.tau_inner == 5
        assert panel.outer_dt == 1
        assert panel.action_mode == "Uniform"
        assert panel.gaussian_mean == 0.0
        assert panel.gaussian_std == 0.3

    def test_custom_parameters(self):
        panel = RoboticGasConfigPanel(
            N=64,
            task_name="cheetah-run",
            max_iterations=500,
            seed=0,
            render_width=320,
            render_height=320,
        )
        assert panel.N == 64
        assert panel.task_name == "cheetah-run"
        assert panel.max_iterations == 500
        assert panel.seed == 0
        assert panel.render_width == 320
        assert panel.render_height == 320

    def test_task_name_options(self):
        panel = RoboticGasConfigPanel()
        expected_tasks = [
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
        ]
        assert panel.param.task_name.objects == expected_tasks

    def test_algorithm_mode_options(self):
        panel = RoboticGasConfigPanel()
        assert panel.param.algorithm_mode.objects == ["Single Loop", "Planning"]

    def test_action_mode_options(self):
        panel = RoboticGasConfigPanel()
        assert panel.param.action_mode.objects == ["Uniform", "Gaussian"]

    def test_gaussian_params_visibility(self):
        panel = RoboticGasConfigPanel()
        assert panel._gaussian_params_section.visible is False
        panel.action_mode = "Gaussian"
        assert panel._gaussian_params_section.visible is True
        panel.action_mode = "Uniform"
        assert panel._gaussian_params_section.visible is False

    def test_ui_widgets_created(self):
        panel = RoboticGasConfigPanel()
        assert isinstance(panel.run_button, pn.widgets.Button)
        assert isinstance(panel.stop_button, pn.widgets.Button)
        assert isinstance(panel.progress_bar, pn.indicators.Progress)
        assert isinstance(panel.progress_text, pn.pane.Markdown)
        assert isinstance(panel.status_pane, pn.pane.Markdown)

    def test_initial_button_state(self):
        panel = RoboticGasConfigPanel()
        assert panel.run_button.disabled is False
        assert panel.stop_button.disabled is True

    def test_run_button_properties(self):
        panel = RoboticGasConfigPanel()
        assert panel.run_button.name == "Run Simulation"
        assert panel.run_button.button_type == "primary"

    def test_stop_button_properties(self):
        panel = RoboticGasConfigPanel()
        assert panel.stop_button.name == "Stop"
        assert panel.stop_button.button_type == "danger"

    def test_initial_state(self):
        panel = RoboticGasConfigPanel()
        assert panel.gas is None
        assert panel.history is None
        assert panel.tree_history is None
        assert panel._stop_requested is False
        assert panel._env is None

    def test_add_completion_callback(self):
        panel = RoboticGasConfigPanel()
        cb = MagicMock()
        panel.add_completion_callback(cb)
        assert cb in panel._on_simulation_complete

    def test_multiple_callbacks(self):
        panel = RoboticGasConfigPanel()
        cb1 = MagicMock()
        cb2 = MagicMock()
        panel.add_completion_callback(cb1)
        panel.add_completion_callback(cb2)
        assert len(panel._on_simulation_complete) == 2

    def test_on_stop_clicked(self):
        panel = RoboticGasConfigPanel()
        panel._on_stop_clicked(None)
        assert panel._stop_requested is True
        assert "Stopping" in panel.status_pane.object

    def test_on_run_clicked_blocks_double_start(self):
        panel = RoboticGasConfigPanel()
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        panel._simulation_thread = mock_thread
        panel._on_run_clicked(None)
        assert panel.run_button.disabled is False

    def test_on_run_clicked_starts_thread(self):
        """Run button click should start a worker thread."""
        panel = RoboticGasConfigPanel()
        with patch.object(panel, "_run_simulation_worker"):
            panel._on_run_clicked(None)
            assert panel.run_button.disabled is True
            assert panel.stop_button.disabled is False
            assert panel.progress_bar.value == 0
            assert panel._simulation_thread is not None
            # Wait for thread to complete (it's mocked, so instant)
            panel._simulation_thread.join(timeout=2)

    def test_cleanup_simulation(self):
        panel = RoboticGasConfigPanel()
        panel.run_button.disabled = True
        panel.stop_button.disabled = False
        panel._cleanup_simulation()
        assert panel.run_button.disabled is False
        assert panel.stop_button.disabled is True

    def test_close_env_no_env(self):
        panel = RoboticGasConfigPanel()
        panel._close_env()  # Should not raise

    def test_close_env_with_env(self):
        panel = RoboticGasConfigPanel()
        mock_env = MagicMock()
        panel._env = mock_env
        panel._close_env()
        mock_env.close.assert_called_once()
        assert panel._env is None

    def test_close_env_exception_swallowed(self):
        panel = RoboticGasConfigPanel()
        mock_env = MagicMock()
        mock_env.close.side_effect = RuntimeError("close failed")
        panel._env = mock_env
        panel._close_env()  # Should not raise
        assert panel._env is None

    def test_schedule_ui_update_no_curdoc(self):
        panel = RoboticGasConfigPanel()
        called = []
        panel._schedule_ui_update(lambda: called.append(True))
        assert called == [True]

    def test_update_progress(self):
        panel = RoboticGasConfigPanel(max_iterations=20)
        import time

        t_start = time.monotonic()
        panel._update_progress(9, t_start)
        assert panel.progress_bar.value == 50  # (10/20) * 100
        assert "10/20" in panel.progress_text.object

    def test_update_progress_first_iteration(self):
        panel = RoboticGasConfigPanel(max_iterations=100)
        import time

        t_start = time.monotonic()
        panel._update_progress(0, t_start)
        assert panel.progress_bar.value == 1
        assert "1/100" in panel.progress_text.object

    def test_panel_returns_column(self):
        panel = RoboticGasConfigPanel()
        layout = panel.panel()
        assert isinstance(layout, pn.Column)

    def test_finish_simulation_calls_callbacks(self):
        panel = RoboticGasConfigPanel()
        history = _make_history()
        panel.history = history
        cb = MagicMock()
        panel.add_completion_callback(cb)
        panel._stop_requested = False
        panel._finish_simulation()
        cb.assert_called_once_with(history)

    def test_finish_simulation_skips_callbacks_when_stopped(self):
        panel = RoboticGasConfigPanel()
        panel.history = _make_history()
        cb = MagicMock()
        panel.add_completion_callback(cb)
        panel._stop_requested = True
        panel._finish_simulation()
        cb.assert_not_called()

    def test_on_simulation_finished_updates_status(self):
        panel = RoboticGasConfigPanel()
        panel.history = _make_history(n_iters=5)
        panel._on_simulation_finished()
        assert "Completed" in panel.status_pane.object
        assert "5 iterations" in panel.status_pane.object

    def test_status_pane_initial_text(self):
        panel = RoboticGasConfigPanel()
        assert "Ready to run" in panel.status_pane.object
        assert "dm_control" in panel.status_pane.object


# ---------------------------------------------------------------------------
# RoboticGasVisualizer
# ---------------------------------------------------------------------------


class TestRoboticGasVisualizer:
    def test_default_initialization(self):
        viz = RoboticGasVisualizer()
        assert viz.history is None

    def test_widgets_created(self):
        viz = RoboticGasVisualizer()
        assert isinstance(viz.game_player, pn.widgets.Player)
        assert isinstance(viz.frame_pane, pn.pane.PNG)
        assert isinstance(viz.reward_plot_pane, pn.pane.HoloViews)
        assert isinstance(viz.fitness_plot_pane, pn.pane.HoloViews)
        assert isinstance(viz.clone_pct_plot_pane, pn.pane.HoloViews)
        assert isinstance(viz.alive_plot_pane, pn.pane.HoloViews)
        assert isinstance(viz.dt_plot_pane, pn.pane.HoloViews)
        assert isinstance(viz.info_pane, pn.pane.Markdown)

    def test_frame_pane_is_png(self):
        """Robotics dashboard uses PNG (single-frame)."""
        viz = RoboticGasVisualizer()
        assert isinstance(viz.frame_pane, pn.pane.PNG)

    def test_game_player_initial_state(self):
        viz = RoboticGasVisualizer()
        assert viz.game_player.start == 0
        assert viz.game_player.end == 0
        assert viz.game_player.value == 0
        assert viz.game_player.disabled is True
        assert viz.game_player.interval == 200

    def test_create_blank_frame(self):
        viz = RoboticGasVisualizer()
        frame = viz._create_blank_frame()
        assert isinstance(frame, bytes)
        assert len(frame) > 0

    def test_array_to_png(self):
        viz = RoboticGasVisualizer()
        arr = np.zeros((480, 480, 3), dtype=np.uint8)
        png = viz._array_to_png(arr)
        assert isinstance(png, bytes)
        assert len(png) > 0
        assert png[:4] == b"\x89PNG"

    def test_array_to_png_different_sizes(self):
        viz = RoboticGasVisualizer()
        for size in [(120, 120, 3), (480, 480, 3), (1920, 1920, 3)]:
            arr = np.zeros(size, dtype=np.uint8)
            png = viz._array_to_png(arr)
            assert isinstance(png, bytes)
            assert len(png) > 0

    def test_set_history(self):
        viz = RoboticGasVisualizer()
        history = _make_history(n_iters=15, task="walker-walk")
        viz.set_history(history)

        assert viz.history is history
        assert viz.game_player.end == 14
        assert viz.game_player.value == 0
        assert viz.game_player.disabled is False
        assert "15 iterations" in viz.info_pane.object
        assert "walker-walk" in viz.info_pane.object
        assert "N=30" in viz.info_pane.object

    def test_set_history_builds_static_plots(self):
        viz = RoboticGasVisualizer()
        history = _make_history(n_iters=10)
        viz.set_history(history)
        # All static plots should be populated
        assert viz.reward_plot_pane.object is not None
        assert viz.fitness_plot_pane.object is not None
        assert viz.clone_pct_plot_pane.object is not None
        assert viz.alive_plot_pane.object is not None
        assert viz.dt_plot_pane.object is not None

    def test_set_history_updates_max_reward_info(self):
        viz = RoboticGasVisualizer()
        history = _make_history(n_iters=10)
        viz.set_history(history)
        # rewards_max = [0.0, 0.2, 0.4, ..., 1.8], max = 1.8
        assert "1.8" in viz.info_pane.object

    def test_set_history_with_frames_updates_frame_pane(self):
        viz = RoboticGasVisualizer()
        history = _make_history(n_iters=5, with_frames=True)
        viz.set_history(history)
        # Frame pane should have been updated (frame at idx 0)
        assert viz.frame_pane.object is not None
        assert isinstance(viz.frame_pane.object, bytes)

    def test_on_game_frame_change_no_history(self):
        viz = RoboticGasVisualizer()
        viz._on_game_frame_change(None)  # Should not raise

    def test_on_game_frame_change_with_frames(self):
        viz = RoboticGasVisualizer()
        history = _make_history(n_iters=5, with_frames=True)
        viz.set_history(history)
        viz.game_player.value = 2
        viz._on_game_frame_change(None)
        # Frame pane should contain PNG data for frame at index 2
        assert isinstance(viz.frame_pane.object, bytes)
        assert len(viz.frame_pane.object) > 0

    def test_game_player_does_not_change_plots(self):
        """Changing game_player should only update the frame, not plots."""
        viz = RoboticGasVisualizer()
        history = _make_history(n_iters=10, with_frames=True)
        viz.set_history(history)

        # Record plot state after initial set_history
        initial_reward_plot = viz.reward_plot_pane.object

        # Change game_player — only frame should update
        viz.game_player.value = 5
        # Reward plot should still be the same object (not re-rendered by game callback)
        assert viz.reward_plot_pane.object is initial_reward_plot

    def test_panel_returns_column(self):
        viz = RoboticGasVisualizer()
        layout = viz.panel()
        assert isinstance(layout, pn.Column)

    def test_panel_includes_algorithm_metrics(self):
        viz = RoboticGasVisualizer()
        layout = viz.panel()
        # Should have the "Algorithm Metrics" section
        md_texts = [str(c.object) for c in layout.objects if isinstance(c, pn.pane.Markdown)]
        assert any("Algorithm Metrics" in t for t in md_texts)


# ---------------------------------------------------------------------------
# RoboticHistory dataclass
# ---------------------------------------------------------------------------


class TestRoboticHistoryFromRun:
    def test_from_run_basic(self):
        infos = [
            {
                "mean_reward": 0.5 + i * 0.1,
                "max_reward": 0.8 + i * 0.1,
                "min_reward": 0.1,
                "alive_count": 30,
                "num_cloned": 3,
                "mean_virtual_reward": 0.5,
                "max_virtual_reward": 1.0,
                "best_frame": None,
            }
            for i in range(10)
        ]
        history = RoboticHistory.from_run(infos, None, N=30, task_name="cartpole-balance")
        assert len(history.iterations) == 10
        assert history.N == 30
        assert history.task_name == "cartpole-balance"
        assert history.max_iterations == 10
        assert not history.has_frames

    def test_from_run_with_frames(self):
        frames = [np.zeros((480, 480, 3), dtype=np.uint8) for _ in range(5)]
        infos = [
            {
                "mean_reward": 0.5,
                "max_reward": 0.8,
                "alive_count": 30,
                "num_cloned": 3,
                "mean_virtual_reward": 0.5,
                "best_frame": frames[i],
                "best_walker_idx": i,
            }
            for i in range(5)
        ]
        history = RoboticHistory.from_run(infos, None, N=30, task_name="cheetah-run")
        assert history.has_frames
        assert len(history.best_frames) == 5
        assert all(isinstance(f, np.ndarray) for f in history.best_frames)

    def test_from_run_optional_fields_default(self):
        infos = [
            {
                "mean_reward": 1.0,
                "max_reward": 2.0,
                "alive_count": 10,
                "num_cloned": 1,
                "mean_virtual_reward": 0.5,
            }
        ]
        history = RoboticHistory.from_run(infos, None, N=10, task_name="test")
        assert history.rewards_min == [0.0]
        assert history.virtual_rewards_max == [0.0]
        assert history.best_frames == [None]
        assert history.best_indices == [0]

    def test_has_frames_empty_list(self):
        history = _make_history(n_iters=1, with_frames=False)
        assert not history.has_frames

    def test_from_run_preserves_all_metrics(self):
        infos = [
            {
                "mean_reward": float(i),
                "max_reward": float(i * 2),
                "min_reward": float(i * 0.5),
                "alive_count": 30 - i,
                "num_cloned": i,
                "mean_virtual_reward": 1.0 + i * 0.1,
                "max_virtual_reward": 2.0 + i * 0.1,
                "best_frame": None,
                "best_walker_idx": i % 5,
            }
            for i in range(8)
        ]
        history = RoboticHistory.from_run(infos, None, N=30, task_name="walker-walk")
        assert history.rewards_mean == [float(i) for i in range(8)]
        assert history.rewards_max == [float(i * 2) for i in range(8)]
        assert history.rewards_min == [float(i * 0.5) for i in range(8)]
        assert history.alive_counts == [30 - i for i in range(8)]
        assert history.num_cloned == list(range(8))
        assert history.best_indices == [i % 5 for i in range(8)]


# ---------------------------------------------------------------------------
# create_app integration
# ---------------------------------------------------------------------------


class TestCreateApp:
    def test_create_app_returns_template(self):
        app = create_app()
        assert isinstance(app, pn.template.FastListTemplate)

    def test_create_app_has_sidebar_and_main(self):
        app = create_app()
        assert len(app.sidebar) > 0
        assert len(app.main) > 0

    def test_create_app_title(self):
        app = create_app()
        assert "DM Control" in app.title

    def test_callback_wiring(self):
        """Config panel should forward history to visualizer on completion."""
        config_panel = RoboticGasConfigPanel()
        visualizer = RoboticGasVisualizer()

        def on_complete(history):
            visualizer.set_history(history)

        config_panel.add_completion_callback(on_complete)
        assert len(config_panel._on_simulation_complete) == 1

        history = _make_history(n_iters=5, task="reacher-easy")
        config_panel._on_simulation_complete[0](history)
        assert visualizer.history is history
        assert "reacher-easy" in visualizer.info_pane.object


# ---------------------------------------------------------------------------
# Planning visualization helpers
# ---------------------------------------------------------------------------


def _make_planning_history(n_steps: int = 5, with_frames: bool = False):
    """Build a minimal PlanningHistory for testing."""
    traj = PlanningTrajectory(frames=[] if with_frames else None)
    traj.states.append("init")
    traj.observations.append(np.zeros(4))
    cum = 0.0
    for i in range(n_steps):
        reward = float(i + 1)
        cum += reward
        traj.actions.append(i % 4)
        traj.rewards.append(reward)
        traj.cumulative_rewards.append(cum)
        traj.dones.append(i == n_steps - 1)
        traj.planning_infos.append({
            "alive_count": 20 - i,
            "inner_mean_reward": 0.5 + i * 0.1,
            "inner_max_reward": 1.0 + i * 0.2,
            "inner_iterations": 5,
        })
        traj.states.append(f"state_{i}")
        traj.observations.append(np.ones(4) * i)
        if with_frames:
            traj.frames.append(np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8))
    return PlanningHistory.from_trajectory(traj, N=30, env_name="cartpole-balance")


# ---------------------------------------------------------------------------
# Planning visualization tests
# ---------------------------------------------------------------------------


class TestPlanningVisualization:
    def test_set_planning_history(self):
        viz = RoboticGasVisualizer()
        ph = _make_planning_history()
        viz.set_planning_history(ph)
        assert viz.planning_history is ph

    def test_set_planning_history_none(self):
        viz = RoboticGasVisualizer()
        viz.set_planning_history(None)
        assert viz.planning_history is None

    def test_planning_plots_created(self):
        viz = RoboticGasVisualizer()
        ph = _make_planning_history(n_steps=5)
        history = _make_history(n_iters=5)
        viz.set_planning_history(ph)
        viz.set_history(history)
        assert viz.step_reward_plot_pane.object is not None
        assert viz.inner_quality_plot_pane.object is not None

    def test_planning_plots_empty_without_history(self):
        viz = RoboticGasVisualizer()
        history = _make_history(n_iters=5)
        viz.set_history(history)
        assert viz.step_reward_plot_pane.object is None
        assert viz.inner_quality_plot_pane.object is None

    def test_panel_includes_planning_section(self):
        viz = RoboticGasVisualizer()
        layout = viz.panel()
        planning_sections = [
            child
            for child in layout.objects
            if isinstance(child, pn.Column)
            and any(
                isinstance(c, pn.pane.Markdown) and "Planning Stats" in str(c.object)
                for c in getattr(child, "objects", [])
            )
        ]
        assert len(planning_sections) == 1

    def test_planning_section_initially_hidden(self):
        viz = RoboticGasVisualizer()
        viz.panel()
        assert viz._planning_stats_section.visible is False

    def test_planning_panes_exist(self):
        viz = RoboticGasVisualizer()
        assert isinstance(viz.step_reward_plot_pane, pn.pane.HoloViews)
        assert isinstance(viz.inner_quality_plot_pane, pn.pane.HoloViews)


class TestConfigPanelPlanningHistory:
    def test_planning_history_initial_none(self):
        panel = RoboticGasConfigPanel()
        assert panel.planning_history is None

    def test_planning_history_forwarded(self):
        """Callback wiring delivers planning history to visualizer."""
        config_panel = RoboticGasConfigPanel()
        visualizer = RoboticGasVisualizer()
        visualizer.panel()  # must call to create _planning_stats_section

        def on_complete(history):
            visualizer.set_planning_history(config_panel.planning_history)
            visualizer._planning_stats_section.visible = config_panel.planning_history is not None
            visualizer.set_history(history)

        config_panel.add_completion_callback(on_complete)

        # Simulate a planning run having set planning_history
        ph = _make_planning_history(n_steps=5)
        config_panel.planning_history = ph
        config_panel.history = ph.to_robotic_history()
        config_panel._stop_requested = False
        config_panel._finish_simulation()

        assert visualizer.planning_history is ph
        assert visualizer._planning_stats_section.visible is True


class TestModeSwitch:
    def test_planning_params_hidden_in_single_loop(self):
        panel = RoboticGasConfigPanel(algorithm_mode="Single Loop")
        assert panel._planning_params_section.visible is False

    def test_planning_params_shown_in_planning(self):
        panel = RoboticGasConfigPanel(algorithm_mode="Planning")
        assert panel._planning_params_section.visible is True

    def test_switching_to_planning_shows_params(self):
        panel = RoboticGasConfigPanel(algorithm_mode="Single Loop")
        assert panel._planning_params_section.visible is False
        panel.algorithm_mode = "Planning"
        assert panel._planning_params_section.visible is True

    def test_switching_to_single_loop_hides_planning_params(self):
        panel = RoboticGasConfigPanel(algorithm_mode="Planning")
        panel.algorithm_mode = "Single Loop"
        assert panel._planning_params_section.visible is False

    def test_planning_stats_hidden_after_single_loop_run(self):
        """Planning stats section should be hidden when running single loop."""
        config_panel = RoboticGasConfigPanel()
        visualizer = RoboticGasVisualizer()
        visualizer.panel()

        def on_complete(history):
            visualizer.set_planning_history(config_panel.planning_history)
            visualizer._planning_stats_section.visible = config_panel.planning_history is not None
            visualizer.set_history(history)

        config_panel.add_completion_callback(on_complete)

        # First do a "planning" run
        ph = _make_planning_history(n_steps=5)
        config_panel.planning_history = ph
        config_panel.history = ph.to_robotic_history()
        config_panel._stop_requested = False
        config_panel._finish_simulation()
        assert visualizer._planning_stats_section.visible is True

        # Now do a "single loop" run (planning_history cleared)
        config_panel.planning_history = None
        config_panel.history = _make_history(n_iters=5)
        config_panel._finish_simulation()
        assert visualizer._planning_stats_section.visible is False
