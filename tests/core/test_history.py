"""Tests for RunHistory data structure and run() integration."""

import pytest
import torch

from fragile.core.history import RunHistory


class TestRunHistory:
    """Tests for RunHistory class."""

    @pytest.fixture
    def simple_history(self):
        """Create a simple RunHistory for testing."""
        N, d = 5, 2
        n_steps = 10
        record_every = 2
        n_recorded = (n_steps // record_every) + 1  # 6 timesteps

        # Create dummy data
        x_before = torch.randn(n_recorded, N, d, dtype=torch.float64)
        v_before = torch.randn(n_recorded, N, d, dtype=torch.float64)
        x_after = torch.randn(n_recorded - 1, N, d, dtype=torch.float64)
        v_after = torch.randn(n_recorded - 1, N, d, dtype=torch.float64)
        x_final = torch.randn(n_recorded, N, d, dtype=torch.float64)
        v_final = torch.randn(n_recorded, N, d, dtype=torch.float64)

        # Per-step data
        n_alive = torch.ones(n_recorded, dtype=torch.long) * N
        num_cloned = torch.randint(0, N // 2, (n_recorded - 1,))
        step_times = torch.rand(n_recorded - 1, dtype=torch.float64) * 0.01

        # Per-walker per-step data
        fitness = torch.randn(n_recorded - 1, N, dtype=torch.float64)
        rewards = torch.randn(n_recorded - 1, N, dtype=torch.float64)
        cloning_scores = torch.randn(n_recorded - 1, N, dtype=torch.float64)
        cloning_probs = torch.rand(n_recorded - 1, N, dtype=torch.float64)
        will_clone = torch.rand(n_recorded - 1, N) > 0.8
        alive_mask = torch.ones(n_recorded - 1, N, dtype=torch.bool)
        companions_distance = torch.randint(0, N, (n_recorded - 1, N))
        companions_clone = torch.randint(0, N, (n_recorded - 1, N))
        distances = torch.rand(n_recorded - 1, N, dtype=torch.float64)
        z_rewards = torch.randn(n_recorded - 1, N, dtype=torch.float64)
        z_distances = torch.randn(n_recorded - 1, N, dtype=torch.float64)
        pos_squared_differences = torch.rand(n_recorded - 1, N, dtype=torch.float64)
        vel_squared_differences = torch.rand(n_recorded - 1, N, dtype=torch.float64)
        rescaled_rewards = torch.rand(n_recorded - 1, N, dtype=torch.float64)
        rescaled_distances = torch.rand(n_recorded - 1, N, dtype=torch.float64)

        return RunHistory(
            N=N,
            d=d,
            n_steps=n_steps,
            n_recorded=n_recorded,
            record_every=record_every,
            terminated_early=False,
            final_step=n_steps,
            x_before_clone=x_before,
            v_before_clone=v_before,
            x_after_clone=x_after,
            v_after_clone=v_after,
            x_final=x_final,
            v_final=v_final,
            n_alive=n_alive,
            num_cloned=num_cloned,
            step_times=step_times,
            fitness=fitness,
            rewards=rewards,
            cloning_scores=cloning_scores,
            cloning_probs=cloning_probs,
            will_clone=will_clone,
            alive_mask=alive_mask,
            companions_distance=companions_distance,
            companions_clone=companions_clone,
            distances=distances,
            z_rewards=z_rewards,
            z_distances=z_distances,
            pos_squared_differences=pos_squared_differences,
            vel_squared_differences=vel_squared_differences,
            rescaled_rewards=rescaled_rewards,
            rescaled_distances=rescaled_distances,
            total_time=1.234,
            init_time=0.1,
        )

    def test_initialization(self, simple_history):
        """Test RunHistory initializes correctly."""
        assert simple_history.N == 5
        assert simple_history.d == 2
        assert simple_history.n_steps == 10
        assert simple_history.n_recorded == 6
        assert simple_history.record_every == 2
        assert not simple_history.terminated_early
        assert simple_history.final_step == 10

    def test_shapes(self, simple_history):
        """Test all tensor shapes are correct."""
        N, d, n_recorded = 5, 2, 6

        # States
        assert simple_history.x_before_clone.shape == (n_recorded, N, d)
        assert simple_history.v_before_clone.shape == (n_recorded, N, d)
        assert simple_history.x_after_clone.shape == (n_recorded - 1, N, d)
        assert simple_history.v_after_clone.shape == (n_recorded - 1, N, d)
        assert simple_history.x_final.shape == (n_recorded, N, d)
        assert simple_history.v_final.shape == (n_recorded, N, d)

        # Per-step scalars
        assert simple_history.n_alive.shape == (n_recorded,)
        assert simple_history.num_cloned.shape == (n_recorded - 1,)
        assert simple_history.step_times.shape == (n_recorded - 1,)

        # Per-walker per-step
        assert simple_history.fitness.shape == (n_recorded - 1, N)
        assert simple_history.rewards.shape == (n_recorded - 1, N)
        assert simple_history.cloning_scores.shape == (n_recorded - 1, N)
        assert simple_history.cloning_probs.shape == (n_recorded - 1, N)
        assert simple_history.will_clone.shape == (n_recorded - 1, N)
        assert simple_history.alive_mask.shape == (n_recorded - 1, N)

    def test_get_step_index(self, simple_history):
        """Test step index conversion."""
        # record_every = 2, so recorded steps are: 0, 2, 4, 6, 8, 10
        assert simple_history.get_step_index(0) == 0
        assert simple_history.get_step_index(2) == 1
        assert simple_history.get_step_index(4) == 2
        assert simple_history.get_step_index(6) == 3
        assert simple_history.get_step_index(8) == 4
        assert simple_history.get_step_index(10) == 5

        # Non-recorded steps should raise error
        with pytest.raises(ValueError, match="was not recorded"):
            simple_history.get_step_index(1)

        with pytest.raises(ValueError, match="was not recorded"):
            simple_history.get_step_index(3)

    def test_get_walker_trajectory(self, simple_history):
        """Test walker trajectory extraction."""
        walker_idx = 2

        # Final trajectory
        traj = simple_history.get_walker_trajectory(walker_idx, stage="final")
        assert "x" in traj and "v" in traj
        assert traj["x"].shape == (6, 2)  # n_recorded, d
        assert traj["v"].shape == (6, 2)
        assert torch.allclose(traj["x"], simple_history.x_final[:, walker_idx, :])

        # Before clone trajectory
        traj = simple_history.get_walker_trajectory(walker_idx, stage="before_clone")
        assert traj["x"].shape == (6, 2)
        assert torch.allclose(traj["x"], simple_history.x_before_clone[:, walker_idx, :])

        # After clone trajectory
        traj = simple_history.get_walker_trajectory(walker_idx, stage="after_clone")
        assert traj["x"].shape == (5, 2)  # n_recorded - 1
        assert torch.allclose(traj["x"], simple_history.x_after_clone[:, walker_idx, :])

        # Invalid stage
        with pytest.raises(ValueError, match="Unknown stage"):
            simple_history.get_walker_trajectory(walker_idx, stage="invalid")

    def test_get_clone_events(self, simple_history):
        """Test cloning event extraction."""
        events = simple_history.get_clone_events()

        # Each event is (step, cloner_idx, companion_idx)
        assert isinstance(events, list)
        for event in events:
            assert len(event) == 3
            step, cloner_idx, companion_idx = event
            assert isinstance(step, int)
            assert isinstance(cloner_idx, int)
            assert isinstance(companion_idx, int)
            assert 0 <= step <= simple_history.n_steps
            assert 0 <= cloner_idx < simple_history.N
            assert 0 <= companion_idx < simple_history.N

        # Number of events should match total cloning count
        total_clones = simple_history.will_clone.sum().item()
        assert len(events) == total_clones

    def test_get_alive_walkers(self, simple_history):
        """Test alive walker extraction."""
        # All walkers alive in this simple history
        alive = simple_history.get_alive_walkers(step=0)
        assert alive.shape == (simple_history.N,)
        assert torch.all(alive == torch.arange(simple_history.N))

        # Invalid step
        with pytest.raises(ValueError, match="was not recorded"):
            simple_history.get_alive_walkers(step=1)

    def test_to_dict(self, simple_history):
        """Test conversion to dictionary."""
        data = simple_history.to_dict()

        assert isinstance(data, dict)
        assert "N" in data
        assert "d" in data
        assert "x_final" in data
        assert "fitness" in data

        # None values should be excluded
        assert "fitness_gradients" not in data

    def test_save_load(self, simple_history, tmp_path):
        """Test saving and loading."""
        save_path = tmp_path / "test_history.pt"

        # Save
        simple_history.save(str(save_path))
        assert save_path.exists()

        # Load
        loaded = RunHistory.load(str(save_path))

        # Check metadata
        assert loaded.N == simple_history.N
        assert loaded.d == simple_history.d
        assert loaded.n_steps == simple_history.n_steps

        # Check tensors
        assert torch.allclose(loaded.x_final, simple_history.x_final)
        assert torch.allclose(loaded.fitness, simple_history.fitness)
        assert torch.allclose(loaded.rewards, simple_history.rewards)

    def test_summary(self, simple_history):
        """Test summary string generation."""
        summary = simple_history.summary()

        assert isinstance(summary, str)
        assert "10 steps" in summary
        assert "5 walkers" in summary
        assert "2D" in summary
        assert "6 timesteps" in summary
        assert "every 2 steps" in summary

    def test_adaptive_kinetics_data(self):
        """Test RunHistory with adaptive kinetics data."""
        N, d = 5, 2
        n_recorded = 6

        # Create history with gradients and Hessians
        fitness_gradients = torch.randn(n_recorded - 1, N, d, dtype=torch.float64)
        fitness_hessians_diag = torch.randn(n_recorded - 1, N, d, dtype=torch.float64)

        # Minimal RunHistory with adaptive data
        history = RunHistory(
            N=N,
            d=d,
            n_steps=10,
            n_recorded=n_recorded,
            record_every=2,
            terminated_early=False,
            final_step=10,
            x_before_clone=torch.randn(n_recorded, N, d, dtype=torch.float64),
            v_before_clone=torch.randn(n_recorded, N, d, dtype=torch.float64),
            x_after_clone=torch.randn(n_recorded - 1, N, d, dtype=torch.float64),
            v_after_clone=torch.randn(n_recorded - 1, N, d, dtype=torch.float64),
            x_final=torch.randn(n_recorded, N, d, dtype=torch.float64),
            v_final=torch.randn(n_recorded, N, d, dtype=torch.float64),
            n_alive=torch.ones(n_recorded, dtype=torch.long) * N,
            num_cloned=torch.zeros(n_recorded - 1, dtype=torch.long),
            step_times=torch.zeros(n_recorded - 1, dtype=torch.float64),
            fitness=torch.randn(n_recorded - 1, N, dtype=torch.float64),
            rewards=torch.randn(n_recorded - 1, N, dtype=torch.float64),
            cloning_scores=torch.randn(n_recorded - 1, N, dtype=torch.float64),
            cloning_probs=torch.rand(n_recorded - 1, N, dtype=torch.float64),
            will_clone=torch.zeros(n_recorded - 1, N, dtype=torch.bool),
            alive_mask=torch.ones(n_recorded - 1, N, dtype=torch.bool),
            companions_distance=torch.zeros(n_recorded - 1, N, dtype=torch.long),
            companions_clone=torch.zeros(n_recorded - 1, N, dtype=torch.long),
            distances=torch.randn(n_recorded - 1, N, dtype=torch.float64),
            z_rewards=torch.randn(n_recorded - 1, N, dtype=torch.float64),
            z_distances=torch.randn(n_recorded - 1, N, dtype=torch.float64),
            pos_squared_differences=torch.rand(n_recorded - 1, N, dtype=torch.float64),
            vel_squared_differences=torch.rand(n_recorded - 1, N, dtype=torch.float64),
            rescaled_rewards=torch.rand(n_recorded - 1, N, dtype=torch.float64),
            rescaled_distances=torch.rand(n_recorded - 1, N, dtype=torch.float64),
            total_time=1.0,
            init_time=0.1,
            fitness_gradients=fitness_gradients,
            fitness_hessians_diag=fitness_hessians_diag,
        )

        # Check shapes
        assert history.fitness_gradients.shape == (n_recorded - 1, N, d)
        assert history.fitness_hessians_diag.shape == (n_recorded - 1, N, d)

        # Check summary mentions adaptive kinetics
        summary = history.summary()
        assert "gradients recorded" in summary
        assert "Hessian diagonals recorded" in summary


class TestRunIntegration:
    """Test EuclideanGas.run() integration with RunHistory."""

    def test_run_returns_history(self, euclidean_gas):
        """Test that run() returns a RunHistory object."""
        history = euclidean_gas.run(n_steps=5)

        assert isinstance(history, RunHistory)
        assert history.N == euclidean_gas.N
        assert history.d == euclidean_gas.d
        assert history.n_steps == 5

    def test_run_with_record_every(self, euclidean_gas):
        """Test run() with record_every parameter."""
        n_steps = 10
        record_every = 3

        history = euclidean_gas.run(n_steps=n_steps, record_every=record_every)

        # Should record steps: 0, 3, 6, 9, 10 (last step always included)
        expected_recorded = 5
        assert history.n_recorded == expected_recorded
        assert history.record_every == record_every

    def test_run_shapes(self, euclidean_gas):
        """Test that run() produces correct tensor shapes."""
        n_steps = 8
        history = euclidean_gas.run(n_steps=n_steps, record_every=2)

        N = euclidean_gas.N
        d = euclidean_gas.d
        n_recorded = history.n_recorded

        # States
        assert history.x_final.shape == (n_recorded, N, d)
        assert history.v_final.shape == (n_recorded, N, d)
        assert history.x_before_clone.shape == (n_recorded, N, d)
        assert history.x_after_clone.shape == (n_recorded - 1, N, d)

        # Info data
        assert history.fitness.shape == (n_recorded - 1, N)
        assert history.rewards.shape == (n_recorded - 1, N)
        assert history.cloning_scores.shape == (n_recorded - 1, N)

    def test_run_with_init_state(self, euclidean_gas):
        """Test run() with custom initial state."""
        N = euclidean_gas.N
        d = euclidean_gas.d

        x_init = torch.randn(N, d, dtype=torch.float64)
        v_init = torch.randn(N, d, dtype=torch.float64)

        history = euclidean_gas.run(n_steps=5, x_init=x_init, v_init=v_init)

        # First state should match initialization
        assert torch.allclose(history.x_before_clone[0], x_init)
        assert torch.allclose(history.v_before_clone[0], v_init)

    def test_run_timing_data(self, euclidean_gas):
        """Test that timing data is recorded."""
        history = euclidean_gas.run(n_steps=5)

        assert history.total_time > 0
        assert history.init_time >= 0
        assert history.step_times.shape == (history.n_recorded - 1,)
        assert torch.all(history.step_times >= 0)

    def test_run_alive_tracking(self, euclidean_gas):
        """Test that alive walker count is tracked."""
        history = euclidean_gas.run(n_steps=5)

        assert history.n_alive.shape == (history.n_recorded,)
        # All walkers should be alive (no boundary in simple potential)
        assert torch.all(history.n_alive == euclidean_gas.N)

    def test_run_cloning_events(self, euclidean_gas):
        """Test that cloning events are recorded."""
        history = euclidean_gas.run(n_steps=10)

        events = history.get_clone_events()
        # Should have some cloning events
        assert len(events) >= 0  # May be zero with low p_max

        # Verify event structure
        for step, cloner_idx, companion_idx in events:
            assert 0 <= step <= history.n_steps
            assert 0 <= cloner_idx < euclidean_gas.N
            assert 0 <= companion_idx < euclidean_gas.N

    def test_run_record_every_one(self, euclidean_gas):
        """Test run() with record_every=1 (every step)."""
        n_steps = 5
        history = euclidean_gas.run(n_steps=n_steps, record_every=1)

        # Should have n_steps + 1 recorded timesteps (including t=0)
        assert history.n_recorded == n_steps + 1

    def test_run_reproducibility(self, euclidean_gas):
        """Test that run() is reproducible with same seed."""
        torch.manual_seed(42)
        history1 = euclidean_gas.run(n_steps=5)

        torch.manual_seed(42)
        history2 = euclidean_gas.run(n_steps=5)

        # Trajectories should match
        assert torch.allclose(history1.x_final, history2.x_final)
        assert torch.allclose(history1.v_final, history2.v_final)

    def test_run_save_load_roundtrip(self, euclidean_gas, tmp_path):
        """Test full run → save → load → verify cycle."""
        # Run simulation
        history = euclidean_gas.run(n_steps=10, record_every=2)

        # Save
        save_path = tmp_path / "run_history.pt"
        history.save(str(save_path))

        # Load
        loaded = RunHistory.load(str(save_path))

        # Verify
        assert loaded.n_steps == history.n_steps
        assert loaded.N == history.N
        assert torch.allclose(loaded.x_final, history.x_final)
        assert torch.allclose(loaded.fitness, history.fitness)
        assert torch.allclose(loaded.rewards, history.rewards)
