"""Tests for the main EuclideanGas class."""

import pytest
import torch

from fragile.bounds import TorchBounds
from fragile.core.euclidean_gas import (
    CloningParams,
    EuclideanGas,
    SimpleQuadraticPotential,
    SwarmState,
)
from fragile.core.history import RunHistory


class TestEuclideanGas:
    """Tests for EuclideanGas class."""

    def test_initialization(self, euclidean_gas):
        """Test EuclideanGas initialization."""
        gas = euclidean_gas

        assert gas.N == 10
        assert gas.d == 2
        assert gas.device == torch.device("cpu")
        assert gas.dtype == "float64"
        assert gas.kinetic_op is not None
        assert gas.fitness_op is not None

    def test_initialize_state_default(self, euclidean_gas):
        """Test default state initialization."""
        gas = euclidean_gas

        torch.manual_seed(42)
        state = gas.initialize_state()

        assert state.N == gas.N
        assert state.d == gas.d
        assert state.x.device == torch.device("cpu")
        assert state.x.dtype == torch.float64

    def test_initialize_state_custom_positions(self, euclidean_gas):
        """Test state initialization with custom positions."""
        gas = euclidean_gas

        N, d = gas.N, gas.d
        x_init = torch.ones(N, d)

        state = gas.initialize_state(x_init=x_init)

        assert torch.allclose(state.x, x_init.to(dtype=torch.float64))

    def test_initialize_state_custom_velocities(self, euclidean_gas):
        """Test state initialization with custom velocities."""
        gas = euclidean_gas

        N, d = gas.N, gas.d
        v_init = torch.zeros(N, d)

        state = gas.initialize_state(v_init=v_init)

        assert torch.allclose(state.v, v_init.to(dtype=torch.float64))

    def test_initialize_state_custom_both(self, euclidean_gas):
        """Test state initialization with custom positions and velocities."""
        gas = euclidean_gas

        N, d = gas.N, gas.d
        x_init = torch.ones(N, d)
        v_init = torch.zeros(N, d)

        state = gas.initialize_state(x_init=x_init, v_init=v_init)

        assert torch.allclose(state.x, x_init.to(dtype=torch.float64))
        assert torch.allclose(state.v, v_init.to(dtype=torch.float64))

    def test_step_returns_two_states(self, euclidean_gas):
        """Test that step returns cloned and final states."""
        gas = euclidean_gas
        state = gas.initialize_state()

        torch.manual_seed(42)
        state_cloned, state_final = gas.step(state)

        assert isinstance(state_cloned, SwarmState)
        assert isinstance(state_final, SwarmState)
        assert state_cloned.N == state.N
        assert state_final.N == state.N

    def test_step_changes_state(self, euclidean_gas):
        """Test that step changes the state."""
        gas = euclidean_gas

        torch.manual_seed(42)
        state = gas.initialize_state()

        torch.manual_seed(43)
        _state_cloned, state_final = gas.step(state)

        # State should change after step
        assert not torch.allclose(state_final.x, state.x)
        assert not torch.allclose(state_final.v, state.v)

    def test_step_cloned_differs_from_original(self, euclidean_gas):
        """Test that cloned state differs from original."""
        gas = euclidean_gas

        torch.manual_seed(42)
        state = gas.initialize_state()

        torch.manual_seed(43)
        state_cloned, _ = gas.step(state)

        # Cloned state should differ from original
        assert not torch.allclose(state_cloned.x, state.x)

    def test_step_final_differs_from_cloned(self, euclidean_gas):
        """Test that final state differs from cloned state."""
        gas = euclidean_gas
        state = gas.initialize_state()

        torch.manual_seed(42)
        state_cloned, state_final = gas.step(state)

        # Final should differ from cloned (kinetic evolution)
        assert not torch.allclose(state_final.x, state_cloned.x)

    def test_step_with_return_info(self, euclidean_gas):
        """Test that step with return_info returns full info dict."""
        gas = euclidean_gas
        state = gas.initialize_state()

        torch.manual_seed(42)
        state_cloned, state_final, info = gas.step(state, return_info=True)

        # Check info dict structure
        assert "fitness" in info
        assert "rewards" in info
        assert "companions_distance" in info
        assert "companions_clone" in info
        assert "alive_mask" in info
        assert "cloning_scores" in info
        assert "cloning_probs" in info
        assert "will_clone" in info
        assert "num_cloned" in info

        # Check shapes
        N = gas.N
        assert info["fitness"].shape == (N,)
        assert info["rewards"].shape == (N,)
        assert info["companions_distance"].shape == (N,)
        assert info["companions_clone"].shape == (N,)
        assert info["alive_mask"].shape == (N,)

    def test_run_basic(self, small_swarm_gas):
        """Test basic run functionality."""
        gas = small_swarm_gas

        n_steps = 10
        torch.manual_seed(42)
        history = gas.run(n_steps)

        assert isinstance(history, RunHistory)
        assert history.n_steps == n_steps
        assert history.N == gas.N
        assert history.d == gas.d
        assert not history.terminated_early
        assert history.final_step == n_steps

    def test_run_trajectory_shapes(self, small_swarm_gas):
        """Test that run returns correct trajectory shapes."""
        gas = small_swarm_gas

        n_steps = 10
        N, d = gas.N, gas.d

        torch.manual_seed(42)
        history = gas.run(n_steps)

        # With record_every=1 (default), n_recorded = n_steps + 1
        assert history.x_final.shape == (n_steps + 1, N, d)
        assert history.v_final.shape == (n_steps + 1, N, d)
        assert history.n_alive.shape == (n_steps + 1,)

    def test_run_stores_initial_state(self, small_swarm_gas):
        """Test that run stores the initial state."""
        gas = small_swarm_gas

        N, d = gas.N, gas.d
        x_init = torch.ones(N, d)
        v_init = torch.zeros(N, d)

        torch.manual_seed(42)
        history = gas.run(n_steps=5, x_init=x_init, v_init=v_init)

        # Check initial state is stored
        assert torch.allclose(history.x_before_clone[0], x_init.to(dtype=torch.float64))
        assert torch.allclose(history.v_before_clone[0], v_init.to(dtype=torch.float64))

    def test_run_zero_steps(self, small_swarm_gas):
        """Test run with zero steps returns only initial state."""
        gas = small_swarm_gas

        torch.manual_seed(42)
        history = gas.run(n_steps=0)

        assert history.x_final.shape[0] == 1
        assert history.v_final.shape[0] == 1
        assert history.n_alive.shape[0] == 1
        assert history.n_steps == 0
        assert history.n_recorded == 1

    def test_run_multiple_times_different_results(self, small_swarm_gas):
        """Test that multiple runs without fixed seed give different results."""
        gas = small_swarm_gas

        history1 = gas.run(n_steps=10)
        history2 = gas.run(n_steps=10)

        # Should be different (stochastic)
        assert not torch.allclose(history1.x_final, history2.x_final)

    def test_run_reproducible_with_seed(self, small_swarm_gas):
        """Test that runs are reproducible with same seed."""
        gas = small_swarm_gas

        torch.manual_seed(42)
        history1 = gas.run(n_steps=10)

        torch.manual_seed(42)
        history2 = gas.run(n_steps=10)

        assert torch.allclose(history1.x_final, history2.x_final)
        assert torch.allclose(history1.v_final, history2.v_final)

    @pytest.mark.parametrize("n_steps", [1, 5, 20])
    def test_run_different_lengths(self, small_swarm_gas, n_steps):
        """Test runs of different lengths."""
        gas = small_swarm_gas

        torch.manual_seed(42)
        history = gas.run(n_steps)

        assert history.x_final.shape[0] == n_steps + 1
        assert history.v_final.shape[0] == n_steps + 1
        assert history.n_steps == n_steps

    def test_device_consistency(self, euclidean_gas):
        """Test that all tensors stay on correct device."""
        gas = euclidean_gas

        torch.manual_seed(42)
        history = gas.run(n_steps=5)

        assert history.x_final.device == torch.device("cpu")
        assert history.v_final.device == torch.device("cpu")
        assert history.n_alive.device == torch.device("cpu")

    def test_dtype_consistency(self, euclidean_gas):
        """Test that all tensors have correct dtype."""
        gas = euclidean_gas

        torch.manual_seed(42)
        history = gas.run(n_steps=5)

        assert history.x_final.dtype == torch.float64
        assert history.v_final.dtype == torch.float64

    def test_small_swarm(self, small_swarm_gas):
        """Test with small swarm."""
        gas = small_swarm_gas

        torch.manual_seed(42)
        history = gas.run(n_steps=10)

        assert history.x_final.shape[1] == gas.N

    def test_large_swarm(self, large_swarm_gas):
        """Test with large swarm."""
        gas = large_swarm_gas

        torch.manual_seed(42)
        history = gas.run(n_steps=5)  # Fewer steps for speed

        assert history.x_final.shape[1] == gas.N
        assert history.x_final.shape[2] == gas.d

    def test_initial_velocity_distribution(self, euclidean_gas):
        """Test that default initial velocities follow thermal distribution."""
        gas = euclidean_gas

        # Generate many initial states
        torch.manual_seed(42)
        states = [gas.initialize_state() for _ in range(100)]

        # Collect all velocities
        all_v = torch.cat([s.v for s in states], dim=0)

        # Mean should be near zero
        mean_v = torch.mean(all_v, dim=0)
        assert torch.allclose(mean_v, torch.zeros_like(mean_v), atol=0.5)

        # Variance should be approximately 1/beta per dimension
        beta = gas.kinetic_op.beta
        expected_var = torch.tensor(1.0 / beta, dtype=torch.float64)
        actual_var = torch.var(all_v[:, 0])
        assert torch.allclose(actual_var, expected_var, rtol=0.3)

    def test_trajectory_continuity(self, small_swarm_gas):
        """Test that trajectory is continuous (no jumps)."""
        gas = small_swarm_gas

        torch.manual_seed(42)
        history = gas.run(n_steps=10)

        # Check that consecutive positions don't jump too much
        # Note: cloning can cause larger jumps, so we use a generous bound
        for t in range(10):
            dx = history.x_final[t + 1] - history.x_final[t]
            max_displacement = torch.max(torch.norm(dx, dim=-1))

            # With dt=0.01 and cloning, max displacement can be larger
            assert max_displacement < 5.0

    def test_enable_cloning_false(self, euclidean_gas):
        """Test that enable_cloning=False skips cloning."""
        # Create gas with cloning disabled
        gas_no_clone = EuclideanGas(
            N=euclidean_gas.N,
            d=euclidean_gas.d,
            companion_selection=euclidean_gas.companion_selection,
            potential=euclidean_gas.potential,
            kinetic_op=euclidean_gas.kinetic_op,
            cloning=euclidean_gas.cloning,
            fitness_op=euclidean_gas.fitness_op,
            device=euclidean_gas.device,
            dtype=euclidean_gas.dtype,
            enable_cloning=False,
        )

        state = gas_no_clone.initialize_state()
        torch.manual_seed(42)
        state_cloned, state_final, info = gas_no_clone.step(state, return_info=True)

        # When cloning disabled, cloned state should be same as original
        assert torch.allclose(state_cloned.x, state.x)
        assert torch.allclose(state_cloned.v, state.v)

        # But kinetic should still apply
        assert not torch.allclose(state_final.x, state.x)

        # Check info shows no cloning
        assert info["num_cloned"] == 0

    def test_enable_kinetic_false(self, euclidean_gas):
        """Test that enable_kinetic=False skips kinetic evolution."""
        # Create gas with kinetic disabled
        gas_no_kinetic = EuclideanGas(
            N=euclidean_gas.N,
            d=euclidean_gas.d,
            companion_selection=euclidean_gas.companion_selection,
            potential=euclidean_gas.potential,
            kinetic_op=euclidean_gas.kinetic_op,
            cloning=euclidean_gas.cloning,
            fitness_op=euclidean_gas.fitness_op,
            device=euclidean_gas.device,
            dtype=euclidean_gas.dtype,
            enable_kinetic=False,
        )

        state = gas_no_kinetic.initialize_state()
        torch.manual_seed(42)
        state_cloned, state_final = gas_no_kinetic.step(state)

        # Cloning should still happen
        assert not torch.allclose(state_cloned.x, state.x)

        # But final state should be same as cloned (no kinetic)
        assert torch.allclose(state_final.x, state_cloned.x)
        assert torch.allclose(state_final.v, state_cloned.v)

    def test_both_operators_disabled(self, euclidean_gas):
        """Test that both operators can be disabled (diagnostic mode)."""
        # Create gas with both disabled
        gas_diagnostic = EuclideanGas(
            N=euclidean_gas.N,
            d=euclidean_gas.d,
            companion_selection=euclidean_gas.companion_selection,
            potential=euclidean_gas.potential,
            kinetic_op=euclidean_gas.kinetic_op,
            cloning=euclidean_gas.cloning,
            fitness_op=euclidean_gas.fitness_op,
            device=euclidean_gas.device,
            dtype=euclidean_gas.dtype,
            enable_cloning=False,
            enable_kinetic=False,
        )

        state = gas_diagnostic.initialize_state()
        torch.manual_seed(42)
        state_cloned, state_final = gas_diagnostic.step(state)

        # Both should be same as original
        assert torch.allclose(state_cloned.x, state.x)
        assert torch.allclose(state_cloned.v, state.v)
        assert torch.allclose(state_final.x, state.x)
        assert torch.allclose(state_final.v, state.v)
