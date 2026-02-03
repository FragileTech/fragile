"""Tests for Atari Gas dashboard implementation."""

import numpy as np
import pytest
import torch

from fragile.fractalai.videogames.atari_history import AtariHistory
from fragile.fractalai.videogames.cloning import FractalCloningOperator


class TestCumulativeRewardOption:
    """Test use_cumulative_reward parameter in cloning operator."""

    def test_cumulative_reward_option(self):
        """Test use_cumulative_reward parameter in cloning operator."""
        clone_op = FractalCloningOperator(use_cumulative_reward=True, device="cpu")

        N = 10
        obs = torch.randn(N, 128)
        cumulative_rewards = torch.tensor([100.0] * N)
        step_rewards = torch.tensor([1.0] * N)
        alive = torch.ones(N, dtype=torch.bool)

        vr1, _ = clone_op.calculate_fitness(obs, cumulative_rewards, step_rewards, alive)

        # Switch to step rewards
        clone_op.use_cumulative_reward = False
        vr2, _ = clone_op.calculate_fitness(obs, cumulative_rewards, step_rewards, alive)

        # Virtual rewards should be different
        assert not torch.allclose(vr1, vr2), "Virtual rewards should differ based on reward signal"

    def test_cumulative_vs_step_rewards(self):
        """Test that cumulative vs step rewards produce different fitness values."""
        N = 10
        obs = torch.randn(N, 128)

        # Create different cumulative and step rewards
        cumulative_rewards = torch.linspace(0, 100, N)
        step_rewards = torch.ones(N)
        alive = torch.ones(N, dtype=torch.bool)

        # Test with cumulative
        clone_op_cumulative = FractalCloningOperator(use_cumulative_reward=True, device="cpu")
        vr_cumulative, _ = clone_op_cumulative.calculate_fitness(
            obs, cumulative_rewards, step_rewards, alive
        )

        # Test with step
        clone_op_step = FractalCloningOperator(use_cumulative_reward=False, device="cpu")
        vr_step, _ = clone_op_step.calculate_fitness(obs, cumulative_rewards, step_rewards, alive)

        # Should produce different virtual rewards
        assert not torch.allclose(vr_cumulative, vr_step)

    def test_default_uses_step_rewards(self):
        """Test that default behavior uses step rewards."""
        clone_op = FractalCloningOperator(device="cpu")
        assert not clone_op.use_cumulative_reward, "Default should use step rewards"


class TestAtariHistory:
    """Test AtariHistory dataclass."""

    def test_history_construction(self):
        """Test AtariHistory construction from run info."""
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

        final_state = None  # Mock final state
        history = AtariHistory.from_run(infos, final_state, N=30, game_name="Pong")

        assert len(history.iterations) == 10
        assert history.N == 30
        assert history.game_name == "Pong"
        assert not history.has_frames
        assert history.max_iterations == 10

    def test_history_with_frames(self):
        """Test AtariHistory with recorded frames."""
        # Create mock frames
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
        assert all(f is not None for f in history.best_frames)
        assert all(isinstance(f, np.ndarray) for f in history.best_frames)

    def test_history_metrics(self):
        """Test that all metrics are properly stored."""
        infos = [
            {
                "mean_reward": float(i),
                "max_reward": float(i * 2),
                "min_reward": 0.0,
                "alive_count": 30 - i,
                "num_cloned": i,
                "mean_virtual_reward": 1.5 + i * 0.1,
                "max_virtual_reward": 2.0 + i * 0.1,
                "best_frame": None,
            }
            for i in range(10)
        ]

        history = AtariHistory.from_run(infos, None, N=30, game_name="Pong")

        # Check all metrics are present
        assert len(history.rewards_mean) == 10
        assert len(history.rewards_max) == 10
        assert len(history.alive_counts) == 10
        assert len(history.num_cloned) == 10
        assert len(history.virtual_rewards_mean) == 10

        # Check values match
        assert history.rewards_mean == list(range(10))
        assert history.rewards_max == [float(i * 2) for i in range(10)]


class TestFrameRecording:
    """Test frame recording in AtariFractalGas."""

    @pytest.mark.skipif(
        True, reason="Requires plangym environment, tested manually"
    )
    def test_frame_recording_integration(self):
        """Integration test for frame recording (requires plangym)."""
        from fragile.fractalai.videogames.atari_gas import AtariFractalGas

        # This would require a real environment
        # Tested manually with actual plangym environment
        pass

    def test_frame_rendering_fallback(self):
        """Test that frame rendering has proper fallback."""
        from fragile.fractalai.videogames.atari_gas import AtariFractalGas

        class MockEnv:
            """Mock environment without render capability."""
            pass

        gas = AtariFractalGas(MockEnv(), N=10, record_frames=False, device="cpu")

        # Test render fallback
        frame = gas._render_walker_frame(None)

        assert isinstance(frame, np.ndarray)
        assert frame.shape == (210, 160, 3)
        assert frame.dtype == np.uint8


class TestDashboard:
    """Test dashboard components."""

    @pytest.mark.skipif(
        True, reason="Requires panel/holoviews environment, tested manually"
    )
    def test_dashboard_creation(self):
        """Test dashboard can be instantiated."""
        from fragile.fractalai.videogames.dashboard import create_app

        app = create_app()
        assert app is not None

    def test_visualizer_initialization(self):
        """Test visualizer can be created."""
        from fragile.fractalai.videogames.dashboard import AtariGasVisualizer

        visualizer = AtariGasVisualizer()
        assert visualizer.history is None

    def test_config_panel_initialization(self):
        """Test config panel can be created."""
        from fragile.fractalai.videogames.dashboard import AtariGasConfigPanel

        config = AtariGasConfigPanel()
        assert config.N == 30
        assert config.max_iterations == 100
        assert not config.use_cumulative_reward


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
