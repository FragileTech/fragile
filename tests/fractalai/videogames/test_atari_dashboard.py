"""Comprehensive tests for Atari Gas dashboard features."""

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
from fragile.fractalai.videogames.atari_history import AtariHistory
from fragile.fractalai.videogames.dashboard import (
    _configure_headless_wsl,
    _format_duration,
    _parse_semver,
    AtariGasConfigPanel,
    AtariGasVisualizer,
    create_app,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_history(n_iters: int = 10, with_frames: bool = False, game: str = "ALE/Pong-v5"):
    """Build a minimal AtariHistory for testing."""
    frames = (
        [np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8) for _ in range(n_iters)]
        if with_frames
        else [None] * n_iters
    )
    return AtariHistory(
        iterations=list(range(n_iters)),
        rewards_mean=[float(i) for i in range(n_iters)],
        rewards_max=[float(i * 2) for i in range(n_iters)],
        rewards_min=[0.0] * n_iters,
        alive_counts=[30] * n_iters,
        num_cloned=[5] * n_iters,
        virtual_rewards_mean=[1.5 + 0.1 * i for i in range(n_iters)],
        virtual_rewards_max=[2.0 + 0.1 * i for i in range(n_iters)],
        best_frames=frames,
        best_rewards=[float(i * 2) for i in range(n_iters)],
        best_indices=[0] * n_iters,
        N=30,
        max_iterations=n_iters,
        game_name=game,
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestParseSemver:
    def test_basic(self):
        assert _parse_semver("1.2.3") == (1, 2, 3)

    def test_two_parts(self):
        assert _parse_semver("0.1") == (0, 1, 0)

    def test_one_part(self):
        assert _parse_semver("3") == (3, 0, 0)

    def test_prerelease_suffix(self):
        assert _parse_semver("1.2.3rc1") == (1, 2, 3)

    def test_comparison(self):
        assert _parse_semver("0.1.32") >= (0, 1, 32)
        assert _parse_semver("0.1.31") < (0, 1, 32)
        assert _parse_semver("1.0.0") > (0, 99, 99)


class TestFormatDuration:
    def test_seconds_only(self):
        assert _format_duration(45) == "0:45"

    def test_minutes_and_seconds(self):
        assert _format_duration(125) == "2:05"

    def test_hours(self):
        assert _format_duration(3661) == "1:01:01"

    def test_zero(self):
        assert _format_duration(0) == "0:00"

    def test_exactly_one_hour(self):
        assert _format_duration(3600) == "1:00:00"


class TestConfigureHeadlessWsl:
    def test_force_sets_env_vars(self):
        """force=True should set headless env vars regardless of WSL detection."""
        env_keys = ["PYGLET_HEADLESS", "LIBGL_ALWAYS_SOFTWARE", "SDL_VIDEODRIVER", "MPLBACKEND"]
        saved = {k: os.environ.pop(k, None) for k in env_keys}
        try:
            result = _configure_headless_wsl(force=True)
            assert result is True
            assert os.environ["PYGLET_HEADLESS"] == "1"
            assert os.environ["LIBGL_ALWAYS_SOFTWARE"] == "1"
            assert os.environ["SDL_VIDEODRIVER"] == "dummy"
            assert os.environ["MPLBACKEND"] == "Agg"
        finally:
            for k in env_keys:
                os.environ.pop(k, None)
                if saved[k] is not None:
                    os.environ[k] = saved[k]

    def test_does_not_overwrite_existing(self):
        """Should not overwrite vars that are already set."""
        saved = os.environ.get("PYGLET_HEADLESS")
        os.environ["PYGLET_HEADLESS"] = "custom_value"
        try:
            _configure_headless_wsl(force=True)
            assert os.environ["PYGLET_HEADLESS"] == "custom_value"
        finally:
            if saved is not None:
                os.environ["PYGLET_HEADLESS"] = saved
            else:
                os.environ.pop("PYGLET_HEADLESS", None)

    def test_no_wsl_returns_false(self):
        """On non-WSL Linux, force=False should return False (or True on WSL)."""
        # Can't guarantee which machine this runs on, but the function shouldn't crash
        result = _configure_headless_wsl(force=False)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# AtariGasConfigPanel
# ---------------------------------------------------------------------------


class TestAtariGasConfigPanel:
    def test_default_parameters(self):
        panel = AtariGasConfigPanel()
        assert panel.algorithm_mode == "Single Loop"
        assert panel.game_name == "ALE/Pong-v5"
        assert panel.obs_type == "ram"
        assert panel.N == 30
        assert panel.dist_coef == 1.0
        assert panel.reward_coef == 1.0
        assert panel.use_cumulative_reward is False
        assert panel.n_elite == 0
        assert panel.dt_range_min == 1
        assert panel.dt_range_max == 4
        assert panel.max_iterations == 100
        assert panel.record_frames is True
        assert panel.device == "cpu"
        assert panel.seed == 42
        assert panel.n_workers == 1
        assert panel.tau_inner == 5
        assert panel.outer_dt == 1
        assert panel.use_tree_history is True
        assert panel.prune_history is True

    def test_custom_parameters(self):
        panel = AtariGasConfigPanel(
            N=50,
            game_name="ALE/Breakout-v5",
            max_iterations=200,
            seed=123,
        )
        assert panel.N == 50
        assert panel.game_name == "ALE/Breakout-v5"
        assert panel.max_iterations == 200
        assert panel.seed == 123

    def test_game_name_options(self):
        panel = AtariGasConfigPanel()
        expected_games = [
            "ALE/Pong-v5",
            "ALE/Breakout-v5",
            "ALE/MsPacman-v5",
            "ALE/SpaceInvaders-v5",
        ]
        assert panel.param.game_name.objects == expected_games

    def test_obs_type_options(self):
        panel = AtariGasConfigPanel()
        assert panel.param.obs_type.objects == ["ram", "rgb", "grayscale"]

    def test_algorithm_mode_options(self):
        panel = AtariGasConfigPanel()
        assert panel.param.algorithm_mode.objects == ["Single Loop", "Planning"]

    def test_device_options(self):
        panel = AtariGasConfigPanel()
        assert panel.param.device.objects == ["cpu", "cuda"]

    def test_ui_widgets_created(self):
        panel = AtariGasConfigPanel()
        assert isinstance(panel.run_button, pn.widgets.Button)
        assert isinstance(panel.stop_button, pn.widgets.Button)
        assert isinstance(panel.progress_bar, pn.indicators.Progress)
        assert isinstance(panel.progress_text, pn.pane.Markdown)
        assert isinstance(panel.status_pane, pn.pane.Markdown)

    def test_initial_button_state(self):
        panel = AtariGasConfigPanel()
        assert panel.run_button.disabled is False
        assert panel.stop_button.disabled is True

    def test_initial_state(self):
        panel = AtariGasConfigPanel()
        assert panel.gas is None
        assert panel.history is None
        assert panel.tree_history is None
        assert panel._stop_requested is False

    def test_add_completion_callback(self):
        panel = AtariGasConfigPanel()
        cb = MagicMock()
        panel.add_completion_callback(cb)
        assert cb in panel._on_simulation_complete

    def test_multiple_callbacks(self):
        panel = AtariGasConfigPanel()
        cb1 = MagicMock()
        cb2 = MagicMock()
        panel.add_completion_callback(cb1)
        panel.add_completion_callback(cb2)
        assert len(panel._on_simulation_complete) == 2

    def test_on_stop_clicked(self):
        panel = AtariGasConfigPanel()
        panel._on_stop_clicked(None)
        assert panel._stop_requested is True
        assert "Stopping" in panel.status_pane.object

    def test_on_run_clicked_blocks_double_start(self):
        """Running simulation should not start a second thread."""
        panel = AtariGasConfigPanel()
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        panel._simulation_thread = mock_thread
        # Should return immediately without modifying state
        panel._on_run_clicked(None)
        # run_button should NOT have been disabled (early return)
        assert panel.run_button.disabled is False

    def test_cleanup_simulation(self):
        panel = AtariGasConfigPanel()
        panel.run_button.disabled = True
        panel.stop_button.disabled = False
        panel._cleanup_simulation()
        assert panel.run_button.disabled is False
        assert panel.stop_button.disabled is True

    def test_close_env_no_env(self):
        """_close_env should be safe when no env exists."""
        panel = AtariGasConfigPanel()
        panel._close_env()  # Should not raise

    def test_close_env_with_env(self):
        panel = AtariGasConfigPanel()
        mock_env = MagicMock()
        panel._env = mock_env
        panel._close_env()
        mock_env.close.assert_called_once()
        assert panel._env is None

    def test_close_env_exception_swallowed(self):
        panel = AtariGasConfigPanel()
        mock_env = MagicMock()
        mock_env.close.side_effect = RuntimeError("close failed")
        panel._env = mock_env
        panel._close_env()  # Should not raise
        assert panel._env is None

    def test_schedule_ui_update_no_curdoc(self):
        """Without a Bokeh document, updates run directly."""
        panel = AtariGasConfigPanel()
        called = []
        panel._schedule_ui_update(lambda: called.append(True))
        assert called == [True]

    def test_update_progress(self):
        panel = AtariGasConfigPanel(max_iterations=10)
        import time

        t_start = time.monotonic()
        panel._update_progress(4, t_start)
        assert panel.progress_bar.value == 50  # (5/10) * 100
        assert "5/10" in panel.progress_text.object

    def test_panel_returns_column(self):
        panel = AtariGasConfigPanel()
        layout = panel.panel()
        assert isinstance(layout, pn.Column)

    def test_finish_simulation_calls_callbacks(self):
        panel = AtariGasConfigPanel()
        history = _make_history()
        panel.history = history
        cb = MagicMock()
        panel.add_completion_callback(cb)
        panel._stop_requested = False
        panel._finish_simulation()
        cb.assert_called_once_with(history)

    def test_finish_simulation_skips_callbacks_when_stopped(self):
        panel = AtariGasConfigPanel()
        panel.history = _make_history()
        cb = MagicMock()
        panel.add_completion_callback(cb)
        panel._stop_requested = True
        panel._finish_simulation()
        cb.assert_not_called()

    def test_on_simulation_finished_updates_status(self):
        panel = AtariGasConfigPanel()
        panel.history = _make_history(n_iters=5)
        panel._on_simulation_finished()
        assert "Completed" in panel.status_pane.object
        assert "5 iterations" in panel.status_pane.object


# ---------------------------------------------------------------------------
# AtariGasVisualizer
# ---------------------------------------------------------------------------


class TestAtariGasVisualizer:
    def test_default_initialization(self):
        viz = AtariGasVisualizer()
        assert viz.history is None
        assert viz.show_histograms is True

    def test_initialization_with_show_histograms_false(self):
        viz = AtariGasVisualizer(show_histograms=False)
        assert viz.show_histograms is False

    def test_widgets_created(self):
        viz = AtariGasVisualizer()
        assert isinstance(viz.game_player, pn.widgets.Player)
        assert isinstance(viz.plot_player, pn.widgets.Player)
        assert isinstance(viz.frame_pane, pn.pane.PNG)
        assert isinstance(viz.reward_plot_pane, pn.pane.HoloViews)
        assert isinstance(viz.info_pane, pn.pane.Markdown)
        assert isinstance(viz.histogram_alive_pane, pn.pane.HoloViews)
        assert isinstance(viz.histogram_cloning_pane, pn.pane.HoloViews)
        assert isinstance(viz.histogram_virtual_reward_pane, pn.pane.HoloViews)

    def test_players_initial_state(self):
        viz = AtariGasVisualizer()
        for player in (viz.game_player, viz.plot_player):
            assert player.start == 0
            assert player.end == 0
            assert player.value == 0
            assert player.disabled is True

    def test_create_blank_frame(self):
        viz = AtariGasVisualizer()
        frame = viz._create_blank_frame()
        assert isinstance(frame, bytes)
        assert len(frame) > 0

    def test_array_to_png(self):
        viz = AtariGasVisualizer()
        arr = np.zeros((210, 160, 3), dtype=np.uint8)
        png = viz._array_to_png(arr)
        assert isinstance(png, bytes)
        assert len(png) > 0
        assert png[:4] == b"\x89PNG"

    def test_set_history(self):
        viz = AtariGasVisualizer()
        history = _make_history(n_iters=20)
        viz.set_history(history)

        assert viz.history is history
        assert viz.game_player.end == 19
        assert viz.game_player.value == 0
        assert viz.game_player.disabled is False
        assert viz.plot_player.end == 19
        assert viz.plot_player.value == 0
        assert viz.plot_player.disabled is False
        assert "20 iterations" in viz.info_pane.object
        assert "ALE/Pong-v5" in viz.info_pane.object
        assert "N=30" in viz.info_pane.object

    def test_set_history_updates_info_with_max_reward(self):
        viz = AtariGasVisualizer()
        history = _make_history(n_iters=5)
        viz.set_history(history)
        # rewards_max = [0, 2, 4, 6, 8], max = 8
        assert "8.0" in viz.info_pane.object

    def test_on_game_frame_change_no_history(self):
        """_on_game_frame_change should be a no-op when history is None."""
        viz = AtariGasVisualizer()
        viz._on_game_frame_change(None)  # Should not raise

    def test_on_plot_frame_change_no_history(self):
        """_on_plot_frame_change should be a no-op when history is None."""
        viz = AtariGasVisualizer()
        viz._on_plot_frame_change(None)  # Should not raise

    def test_on_plot_frame_change_updates_reward_curve(self):
        viz = AtariGasVisualizer()
        history = _make_history(n_iters=10)
        viz.set_history(history)
        viz.plot_player.value = 5
        viz._on_plot_frame_change(None)
        # Reward plot should be updated
        assert viz.reward_plot_pane.object is not None

    def test_on_plot_frame_change_updates_histograms(self):
        viz = AtariGasVisualizer(show_histograms=True)
        history = _make_history(n_iters=10)
        viz.set_history(history)
        viz.plot_player.value = 5
        viz._on_plot_frame_change(None)
        assert viz.histogram_alive_pane.object is not None
        assert viz.histogram_cloning_pane.object is not None
        assert viz.histogram_virtual_reward_pane.object is not None

    def test_on_plot_frame_change_skips_histograms_when_disabled(self):
        viz = AtariGasVisualizer(show_histograms=False)
        history = _make_history(n_iters=10)
        viz.set_history(history)
        viz.plot_player.value = 3
        viz._on_plot_frame_change(None)
        # Histograms should not be populated
        assert viz.histogram_alive_pane.object is None

    def test_on_game_frame_change_updates_frame(self):
        viz = AtariGasVisualizer()
        history = _make_history(n_iters=5, with_frames=True)
        viz.set_history(history)
        viz.game_player.value = 2
        viz._on_game_frame_change(None)
        assert isinstance(viz.frame_pane.object, bytes)
        assert len(viz.frame_pane.object) > 0

    def test_game_player_independent_of_plot_player(self):
        """Changing game_player should only update the frame, not plots."""
        viz = AtariGasVisualizer(show_histograms=True)
        history = _make_history(n_iters=10, with_frames=True)
        viz.set_history(history)

        # Record plot state after initial set_history
        initial_reward_plot = viz.reward_plot_pane.object

        # Change game_player — only frame should update
        viz.game_player.value = 5
        # Reward plot should still be the same object (not re-rendered by game callback)
        assert viz.reward_plot_pane.object is initial_reward_plot

    def test_plot_player_independent_of_game_player(self):
        """Changing plot_player should only update plots, not the frame."""
        viz = AtariGasVisualizer()
        history = _make_history(n_iters=10, with_frames=True)
        viz.set_history(history)

        # Record frame state after initial set_history (game at idx 0)
        initial_frame = viz.frame_pane.object

        # Change plot_player — frame should NOT change
        viz.plot_player.value = 7
        assert viz.frame_pane.object is initial_frame

    def test_panel_returns_column(self):
        viz = AtariGasVisualizer()
        layout = viz.panel()
        assert isinstance(layout, pn.Column)

    def test_panel_includes_histogram_section(self):
        viz = AtariGasVisualizer(show_histograms=True)
        layout = viz.panel()
        # Should have the histogram section as the last non-None item
        assert any(isinstance(child, pn.Column) for child in layout.objects if child is not None)

    def test_panel_no_histogram_section_when_disabled(self):
        viz = AtariGasVisualizer(show_histograms=False)
        layout = viz.panel()
        # When show_histograms=False, histogram_section=None is passed to Column.
        # Panel wraps None as pn.pane.Str(None). Verify no histogram Column is present.
        histogram_columns = [
            child
            for child in layout.objects
            if isinstance(child, pn.Column)
            and any(
                isinstance(c, pn.pane.Markdown) and "Metrics" in str(c.object)
                for c in getattr(child, "objects", [])
            )
        ]
        assert len(histogram_columns) == 0


# ---------------------------------------------------------------------------
# AtariHistory dataclass
# ---------------------------------------------------------------------------


class TestAtariHistoryFromRun:
    def test_from_run_basic(self):
        infos = [
            {
                "mean_reward": 10.0 + i,
                "max_reward": 20.0 + i,
                "min_reward": 5.0,
                "alive_count": 30,
                "num_cloned": 5,
                "mean_virtual_reward": 1.5,
                "max_virtual_reward": 2.0,
                "best_frame": None,
            }
            for i in range(10)
        ]
        history = AtariHistory.from_run(infos, None, N=30, game_name="Pong")
        assert len(history.iterations) == 10
        assert history.N == 30
        assert history.game_name == "Pong"
        assert history.max_iterations == 10
        assert not history.has_frames

    def test_from_run_with_frames(self):
        frames = [np.zeros((210, 160, 3), dtype=np.uint8) for _ in range(5)]
        infos = [
            {
                "mean_reward": 10.0,
                "max_reward": 20.0,
                "alive_count": 30,
                "num_cloned": 5,
                "mean_virtual_reward": 1.5,
                "best_frame": frames[i],
                "best_walker_idx": i,
            }
            for i in range(5)
        ]
        history = AtariHistory.from_run(infos, None, N=30, game_name="Breakout")
        assert history.has_frames
        assert len(history.best_frames) == 5
        assert all(isinstance(f, np.ndarray) for f in history.best_frames)

    def test_from_run_optional_fields_default(self):
        """Optional info keys should fall back to defaults."""
        infos = [
            {
                "mean_reward": 1.0,
                "max_reward": 2.0,
                "alive_count": 10,
                "num_cloned": 1,
                "mean_virtual_reward": 0.5,
            }
        ]
        history = AtariHistory.from_run(infos, None, N=10, game_name="Test")
        assert history.rewards_min == [0.0]
        assert history.virtual_rewards_max == [0.0]
        assert history.best_frames == [None]
        assert history.best_indices == [0]

    def test_has_frames_empty_list(self):
        history = _make_history(n_iters=1, with_frames=False)
        assert not history.has_frames


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
        assert "Atari" in app.title

    def test_callback_wiring(self):
        """Config panel should have at least one completion callback after create_app."""
        # We inspect internals by recreating the setup
        config_panel = AtariGasConfigPanel()
        visualizer = AtariGasVisualizer()

        def on_complete(history):
            visualizer.set_history(history)

        config_panel.add_completion_callback(on_complete)
        assert len(config_panel._on_simulation_complete) == 1

        # Simulate completion
        history = _make_history(n_iters=5)
        config_panel._on_simulation_complete[0](history)
        assert visualizer.history is history


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
            traj.frames.append(np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8))
    return PlanningHistory.from_trajectory(traj, N=30, env_name="ALE/Pong-v5")


# ---------------------------------------------------------------------------
# Planning visualization tests
# ---------------------------------------------------------------------------


class TestPlanningVisualization:
    def test_set_planning_history(self):
        viz = AtariGasVisualizer()
        ph = _make_planning_history()
        viz.set_planning_history(ph)
        assert viz.planning_history is ph

    def test_set_planning_history_none(self):
        viz = AtariGasVisualizer()
        viz.set_planning_history(None)
        assert viz.planning_history is None

    def test_planning_plots_created(self):
        viz = AtariGasVisualizer()
        ph = _make_planning_history(n_steps=5)
        history = _make_history(n_iters=5)
        viz.set_planning_history(ph)
        viz.set_history(history)
        # Planning plots should be populated after _on_plot_frame_change
        assert viz.step_reward_plot_pane.object is not None
        assert viz.inner_quality_plot_pane.object is not None

    def test_planning_plots_empty_without_history(self):
        viz = AtariGasVisualizer()
        history = _make_history(n_iters=5)
        viz.set_history(history)
        # No planning history -> planning panes should remain empty
        assert viz.step_reward_plot_pane.object is None
        assert viz.inner_quality_plot_pane.object is None

    def test_frame_change_updates_planning_plots(self):
        viz = AtariGasVisualizer()
        ph = _make_planning_history(n_steps=10)
        history = _make_history(n_iters=10)
        viz.set_planning_history(ph)
        viz.set_history(history)

        # Scrub to a different frame using plot_player
        viz.plot_player.value = 5
        viz._on_plot_frame_change(None)
        assert viz.step_reward_plot_pane.object is not None
        assert viz.inner_quality_plot_pane.object is not None

    def test_panel_includes_planning_section(self):
        viz = AtariGasVisualizer()
        layout = viz.panel()
        # Find the planning stats section
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
        viz = AtariGasVisualizer()
        viz.panel()
        assert viz._planning_stats_section.visible is False

    def test_planning_panes_exist(self):
        viz = AtariGasVisualizer()
        assert isinstance(viz.step_reward_plot_pane, pn.pane.HoloViews)
        assert isinstance(viz.inner_quality_plot_pane, pn.pane.HoloViews)


class TestConfigPanelPlanningHistory:
    def test_planning_history_initial_none(self):
        panel = AtariGasConfigPanel()
        assert panel.planning_history is None

    def test_planning_history_forwarded(self):
        """Callback wiring delivers planning history to visualizer."""
        config_panel = AtariGasConfigPanel()
        visualizer = AtariGasVisualizer()
        visualizer.panel()  # must call to create _planning_stats_section

        def on_complete(history):
            visualizer.set_planning_history(config_panel.planning_history)
            visualizer._planning_stats_section.visible = config_panel.planning_history is not None
            visualizer.set_history(history)

        config_panel.add_completion_callback(on_complete)

        # Simulate a planning run having set planning_history
        ph = _make_planning_history(n_steps=5)
        config_panel.planning_history = ph
        config_panel.history = ph.to_atari_history()
        config_panel._stop_requested = False
        config_panel._finish_simulation()

        assert visualizer.planning_history is ph
        assert visualizer._planning_stats_section.visible is True


class TestModeSwitch:
    def test_planning_params_hidden_in_single_loop(self):
        panel = AtariGasConfigPanel(algorithm_mode="Single Loop")
        assert panel._planning_params_section.visible is False
        assert panel._tree_history_params_section.visible is True

    def test_planning_params_shown_in_planning(self):
        panel = AtariGasConfigPanel(algorithm_mode="Planning")
        assert panel._planning_params_section.visible is True
        assert panel._tree_history_params_section.visible is False

    def test_switching_to_planning_shows_params(self):
        panel = AtariGasConfigPanel(algorithm_mode="Single Loop")
        assert panel._planning_params_section.visible is False
        panel.algorithm_mode = "Planning"
        assert panel._planning_params_section.visible is True
        assert panel._tree_history_params_section.visible is False

    def test_switching_to_single_loop_shows_tree_params(self):
        panel = AtariGasConfigPanel(algorithm_mode="Planning")
        assert panel._tree_history_params_section.visible is False
        panel.algorithm_mode = "Single Loop"
        assert panel._tree_history_params_section.visible is True
        assert panel._planning_params_section.visible is False

    def test_planning_stats_hidden_after_single_loop_run(self):
        """Planning stats section should be hidden when running single loop."""
        config_panel = AtariGasConfigPanel()
        visualizer = AtariGasVisualizer()
        visualizer.panel()

        def on_complete(history):
            visualizer.set_planning_history(config_panel.planning_history)
            visualizer._planning_stats_section.visible = config_panel.planning_history is not None
            visualizer.set_history(history)

        config_panel.add_completion_callback(on_complete)

        # First do a "planning" run
        ph = _make_planning_history(n_steps=5)
        config_panel.planning_history = ph
        config_panel.history = ph.to_atari_history()
        config_panel._stop_requested = False
        config_panel._finish_simulation()
        assert visualizer._planning_stats_section.visible is True

        # Now do a "single loop" run (planning_history cleared)
        config_panel.planning_history = None
        config_panel.history = _make_history(n_iters=5)
        config_panel._finish_simulation()
        assert visualizer._planning_stats_section.visible is False
