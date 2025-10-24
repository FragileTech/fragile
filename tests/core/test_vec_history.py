"""Tests for VectorizedHistoryRecorder."""

import pytest
import torch

from fragile.core.euclidean_gas import SwarmState
from fragile.core.history import RunHistory
from fragile.core.vec_history import VectorizedHistoryRecorder


class TestVectorizedHistoryRecorder:
    """Tests for VectorizedHistoryRecorder class."""

    @pytest.fixture
    def recorder(self):
        """Create a basic recorder for testing."""
        return VectorizedHistoryRecorder(
            N=5,
            d=2,
            n_recorded=6,
            device=torch.device("cpu"),
            dtype=torch.float64,
            record_gradients=False,
            record_hessians_diag=False,
            record_hessians_full=False,
        )

    @pytest.fixture
    def recorder_with_gradients(self):
        """Create a recorder with gradient recording enabled."""
        return VectorizedHistoryRecorder(
            N=5,
            d=2,
            n_recorded=6,
            device=torch.device("cpu"),
            dtype=torch.float64,
            record_gradients=True,
            record_hessians_diag=True,
            record_hessians_full=False,
        )

    @pytest.fixture
    def dummy_state(self):
        """Create a dummy swarm state."""
        x = torch.randn(5, 2, dtype=torch.float64)
        v = torch.randn(5, 2, dtype=torch.float64)
        return SwarmState(x, v)

    @pytest.fixture
    def dummy_info(self):
        """Create dummy info dict from step()."""
        N = 5
        return {
            "fitness": torch.randn(N, dtype=torch.float64),
            "rewards": torch.randn(N, dtype=torch.float64),
            "cloning_scores": torch.randn(N, dtype=torch.float64),
            "cloning_probs": torch.rand(N, dtype=torch.float64),
            "will_clone": torch.rand(N) > 0.8,
            "alive_mask": torch.ones(N, dtype=torch.bool),
            "companions_distance": torch.randint(0, N, (N,)),
            "companions_clone": torch.randint(0, N, (N,)),
            "distances": torch.rand(N, dtype=torch.float64),
            "z_rewards": torch.randn(N, dtype=torch.float64),
            "z_distances": torch.randn(N, dtype=torch.float64),
            "pos_squared_differences": torch.rand(N, dtype=torch.float64),
            "vel_squared_differences": torch.rand(N, dtype=torch.float64),
            "rescaled_rewards": torch.rand(N, dtype=torch.float64),
            "rescaled_distances": torch.rand(N, dtype=torch.float64),
            "mu_rewards": 0.5,
            "sigma_rewards": 0.1,
            "mu_distances": 0.3,
            "sigma_distances": 0.05,
            "num_cloned": 2,
        }

    def test_initialization(self, recorder):
        """Test recorder initializes correctly."""
        assert recorder.N == 5
        assert recorder.d == 2
        assert recorder.n_recorded == 6
        assert recorder.recorded_idx == 1

        # Check array shapes
        assert recorder.x_before_clone.shape == (6, 5, 2)
        assert recorder.v_before_clone.shape == (6, 5, 2)
        assert recorder.x_after_clone.shape == (5, 5, 2)  # n_recorded - 1
        assert recorder.x_final.shape == (6, 5, 2)

        # Check per-walker arrays
        assert recorder.fitness.shape == (5, 5)  # n_recorded - 1, N
        assert recorder.rewards.shape == (5, 5)

        # Check per-step arrays
        assert recorder.n_alive.shape == (6,)
        assert recorder.num_cloned.shape == (5,)
        assert recorder.step_times.shape == (5,)

        # Check adaptive kinetics are None by default
        assert recorder.fitness_gradients is None
        assert recorder.fitness_hessians_diag is None
        assert recorder.fitness_hessians_full is None

    def test_initialization_with_gradients(self, recorder_with_gradients):
        """Test recorder with gradient recording."""
        assert recorder_with_gradients.fitness_gradients is not None
        assert recorder_with_gradients.fitness_hessians_diag is not None
        assert recorder_with_gradients.fitness_hessians_full is None

        assert recorder_with_gradients.fitness_gradients.shape == (5, 5, 2)  # n_recorded-1, N, d
        assert recorder_with_gradients.fitness_hessians_diag.shape == (5, 5, 2)

    def test_record_initial_state(self, recorder, dummy_state):
        """Test recording initial state."""
        recorder.record_initial_state(dummy_state, n_alive=5)

        assert torch.allclose(recorder.x_before_clone[0], dummy_state.x)
        assert torch.allclose(recorder.v_before_clone[0], dummy_state.v)
        assert torch.allclose(recorder.x_final[0], dummy_state.x)
        assert torch.allclose(recorder.v_final[0], dummy_state.v)
        assert recorder.n_alive[0] == 5
        assert recorder.recorded_idx == 1  # Still at 1, not incremented

    def test_record_step(self, recorder, dummy_state, dummy_info):
        """Test recording a single step."""
        # Record initial state first
        recorder.record_initial_state(dummy_state, n_alive=5)

        # Create states for the step
        state_before = dummy_state
        state_cloned = SwarmState(
            torch.randn(5, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
        )
        state_final = SwarmState(
            torch.randn(5, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
        )

        # Record step
        recorder.record_step(
            state_before=state_before,
            state_cloned=state_cloned,
            state_final=state_final,
            info=dummy_info,
            step_time=0.01,
            grad_fitness=None,
            hess_fitness=None,
            is_diagonal_hessian=False,
        )

        # Check states were recorded at index 1
        assert torch.allclose(recorder.x_before_clone[1], state_before.x)
        assert torch.allclose(recorder.v_before_clone[1], state_before.v)
        assert torch.allclose(recorder.x_after_clone[0], state_cloned.x)
        assert torch.allclose(recorder.v_after_clone[0], state_cloned.v)
        assert torch.allclose(recorder.x_final[1], state_final.x)
        assert torch.allclose(recorder.v_final[1], state_final.v)

        # Check info was recorded at index 0 (idx_minus_1)
        assert torch.allclose(recorder.fitness[0], dummy_info["fitness"])
        assert torch.allclose(recorder.rewards[0], dummy_info["rewards"])
        assert recorder.n_alive[1] == dummy_info["alive_mask"].sum().item()
        assert recorder.num_cloned[0] == dummy_info["num_cloned"]
        assert recorder.step_times[0] == 0.01

        # Check recorded_idx was incremented
        assert recorder.recorded_idx == 2

    def test_record_step_with_gradients(self, recorder_with_gradients, dummy_state, dummy_info):
        """Test recording step with adaptive kinetics data."""
        recorder_with_gradients.record_initial_state(dummy_state, n_alive=5)

        state_cloned = SwarmState(
            torch.randn(5, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
        )
        state_final = SwarmState(
            torch.randn(5, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
        )

        grad_fitness = torch.randn(5, 2, dtype=torch.float64)
        hess_fitness = torch.randn(5, 2, dtype=torch.float64)  # Diagonal

        recorder_with_gradients.record_step(
            state_before=dummy_state,
            state_cloned=state_cloned,
            state_final=state_final,
            info=dummy_info,
            step_time=0.01,
            grad_fitness=grad_fitness,
            hess_fitness=hess_fitness,
            is_diagonal_hessian=True,
        )

        # Check gradients were recorded
        assert torch.allclose(recorder_with_gradients.fitness_gradients[0], grad_fitness)
        assert torch.allclose(recorder_with_gradients.fitness_hessians_diag[0], hess_fitness)

    def test_build_basic(self, recorder, dummy_state, dummy_info):
        """Test building RunHistory from recorder."""
        # Record initial state
        recorder.record_initial_state(dummy_state, n_alive=5)

        # Record 2 steps
        for _ in range(2):
            state_cloned = SwarmState(
                torch.randn(5, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
            )
            state_final = SwarmState(
                torch.randn(5, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
            )
            recorder.record_step(
                state_before=dummy_state,
                state_cloned=state_cloned,
                state_final=state_final,
                info=dummy_info,
                step_time=0.01,
            )

        # Build history
        history = recorder.build(
            n_steps=10,
            record_every=5,
            terminated_early=False,
            final_step=10,
            total_time=0.5,
            init_time=0.1,
            bounds=None,
        )

        # Check metadata
        assert isinstance(history, RunHistory)
        assert history.N == 5
        assert history.d == 2
        assert history.n_steps == 10
        assert history.n_recorded == 3  # Initial + 2 steps
        assert history.record_every == 5
        assert not history.terminated_early
        assert history.total_time == 0.5
        assert history.init_time == 0.1

        # Check array shapes were trimmed correctly
        assert history.x_before_clone.shape == (3, 5, 2)
        assert history.x_after_clone.shape == (2, 5, 2)  # n_recorded - 1
        assert history.x_final.shape == (3, 5, 2)
        assert history.fitness.shape == (2, 5)
        assert history.n_alive.shape == (3,)
        assert history.num_cloned.shape == (2,)

    def test_build_early_termination(self, recorder, dummy_state):
        """Test building history with early termination (only initial state)."""
        # Record only initial state
        recorder.record_initial_state(dummy_state, n_alive=0)

        # Build with early termination
        history = recorder.build(
            n_steps=0,
            record_every=1,
            terminated_early=True,
            final_step=0,
            total_time=0.0,
            init_time=0.05,
            bounds=None,
        )

        # Check it's valid
        assert history.n_recorded == 1  # Only initial state
        assert history.n_steps == 0
        assert history.terminated_early
        assert history.x_before_clone.shape == (1, 5, 2)
        assert history.x_after_clone.shape == (0, 5, 2)  # Empty
        assert history.fitness.shape == (0, 5)  # Empty

    def test_build_partial_recording(self, recorder, dummy_state, dummy_info):
        """Test building with fewer recorded steps than allocated."""
        # Record initial + 3 steps (out of allocated 6)
        recorder.record_initial_state(dummy_state, n_alive=5)

        for _ in range(3):
            state_cloned = SwarmState(
                torch.randn(5, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
            )
            state_final = SwarmState(
                torch.randn(5, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
            )
            recorder.record_step(
                state_before=dummy_state,
                state_cloned=state_cloned,
                state_final=state_final,
                info=dummy_info,
                step_time=0.01,
            )

        # Build history
        history = recorder.build(
            n_steps=15,
            record_every=5,
            terminated_early=True,
            final_step=15,
            total_time=0.3,
            init_time=0.05,
            bounds=None,
        )

        # Should have trimmed to actual_recorded = 4 (initial + 3 steps)
        assert history.n_recorded == 4
        assert history.x_before_clone.shape == (4, 5, 2)
        assert history.x_after_clone.shape == (3, 5, 2)
        assert history.fitness.shape == (3, 5)

    def test_multiple_steps_recorded(self, recorder, dummy_state, dummy_info):
        """Test recording multiple steps sequentially."""
        recorder.record_initial_state(dummy_state, n_alive=5)

        # Record 5 steps (to fill n_recorded=6 with initial)
        for i in range(5):
            state_cloned = SwarmState(
                torch.randn(5, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
            )
            state_final = SwarmState(
                torch.randn(5, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
            )
            recorder.record_step(
                state_before=dummy_state,
                state_cloned=state_cloned,
                state_final=state_final,
                info=dummy_info,
                step_time=0.01 * (i + 1),
            )

        # Check recorded_idx is at 6 now
        assert recorder.recorded_idx == 6

        # Check step times were recorded correctly
        for i in range(5):
            assert recorder.step_times[i] == 0.01 * (i + 1)

        # Build and verify
        history = recorder.build(
            n_steps=25,
            record_every=5,
            terminated_early=False,
            final_step=25,
            total_time=1.0,
            init_time=0.1,
            bounds=None,
        )

        assert history.n_recorded == 6
        assert history.x_final.shape == (6, 5, 2)


class TestIntegrationWithEuclideanGas:
    """Integration tests verifying recorder works with EuclideanGas."""

    def test_run_uses_recorder(self, euclidean_gas):
        """Test that EuclideanGas.run() uses recorder correctly."""
        history = euclidean_gas.run(n_steps=5, record_every=2)

        # Should work exactly as before
        assert isinstance(history, RunHistory)
        assert history.N == euclidean_gas.N
        assert history.d == euclidean_gas.d
        assert history.n_steps == 5

        # Check data shapes
        assert history.x_final.shape[0] == history.n_recorded
        assert history.fitness.shape[0] == history.n_recorded - 1

    def test_run_with_early_termination(self, euclidean_gas):
        """Test recorder handles early termination correctly."""
        # Initialize all walkers out of bounds
        x_init = torch.ones(euclidean_gas.N, euclidean_gas.d) * 1000.0
        v_init = torch.zeros(euclidean_gas.N, euclidean_gas.d)

        history = euclidean_gas.run(n_steps=10, x_init=x_init, v_init=v_init)

        # Should terminate early with only initial state
        assert history.n_recorded == 1
        assert history.terminated_early
        assert history.n_steps == 0

    def test_run_with_record_every(self, euclidean_gas):
        """Test recorder handles record_every correctly."""
        n_steps = 10
        record_every = 3

        history = euclidean_gas.run(n_steps=n_steps, record_every=record_every)

        # Should record: 0, 3, 6, 9, 10 = 5 timesteps
        assert history.n_recorded == 5
        assert history.record_every == record_every

    def test_recorder_preserves_data(self, euclidean_gas):
        """Test that recorder preserves all data correctly."""
        torch.manual_seed(42)
        history1 = euclidean_gas.run(n_steps=5)

        torch.manual_seed(42)
        history2 = euclidean_gas.run(n_steps=5)

        # Should get identical results (recorder doesn't affect computation)
        assert torch.allclose(history1.x_final, history2.x_final)
        assert torch.allclose(history1.fitness, history2.fitness)
        assert torch.allclose(history1.rewards, history2.rewards)
