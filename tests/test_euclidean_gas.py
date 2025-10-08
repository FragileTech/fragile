"""Tests for the main EuclideanGas class."""

import pytest
import torch

from fragile.euclidean_gas import (
    EuclideanGas,
    EuclideanGasParams,
    SwarmState,
)


class TestEuclideanGas:
    """Tests for EuclideanGas class."""

    def test_initialization(self, euclidean_gas_params):
        """Test EuclideanGas initialization."""
        gas = EuclideanGas(euclidean_gas_params)

        assert gas.params == euclidean_gas_params
        assert gas.device == torch.device("cpu")
        assert gas.dtype == torch.float64
        assert gas.cloning_op is not None
        assert gas.kinetic_op is not None

    def test_initialize_state_default(self, euclidean_gas_params):
        """Test default state initialization."""
        gas = EuclideanGas(euclidean_gas_params)

        torch.manual_seed(42)
        state = gas.initialize_state()

        assert state.N == euclidean_gas_params.N
        assert state.d == euclidean_gas_params.d
        assert state.x.device == torch.device("cpu")
        assert state.x.dtype == torch.float64

    def test_initialize_state_custom_positions(self, euclidean_gas_params):
        """Test state initialization with custom positions."""
        gas = EuclideanGas(euclidean_gas_params)

        N, d = euclidean_gas_params.N, euclidean_gas_params.d
        x_init = torch.ones(N, d)

        state = gas.initialize_state(x_init=x_init)

        assert torch.allclose(state.x, x_init.to(dtype=torch.float64))

    def test_initialize_state_custom_velocities(self, euclidean_gas_params):
        """Test state initialization with custom velocities."""
        gas = EuclideanGas(euclidean_gas_params)

        N, d = euclidean_gas_params.N, euclidean_gas_params.d
        v_init = torch.zeros(N, d)

        state = gas.initialize_state(v_init=v_init)

        assert torch.allclose(state.v, v_init.to(dtype=torch.float64))

    def test_initialize_state_custom_both(self, euclidean_gas_params):
        """Test state initialization with custom positions and velocities."""
        gas = EuclideanGas(euclidean_gas_params)

        N, d = euclidean_gas_params.N, euclidean_gas_params.d
        x_init = torch.ones(N, d)
        v_init = torch.zeros(N, d)

        state = gas.initialize_state(x_init=x_init, v_init=v_init)

        assert torch.allclose(state.x, x_init.to(dtype=torch.float64))
        assert torch.allclose(state.v, v_init.to(dtype=torch.float64))

    def test_step_returns_two_states(self, euclidean_gas_params):
        """Test that step returns cloned and final states."""
        gas = EuclideanGas(euclidean_gas_params)
        state = gas.initialize_state()

        torch.manual_seed(42)
        state_cloned, state_final = gas.step(state)

        assert isinstance(state_cloned, SwarmState)
        assert isinstance(state_final, SwarmState)
        assert state_cloned.N == state.N
        assert state_final.N == state.N

    def test_step_changes_state(self, euclidean_gas_params):
        """Test that step changes the state."""
        gas = EuclideanGas(euclidean_gas_params)

        torch.manual_seed(42)
        state = gas.initialize_state()

        torch.manual_seed(43)
        state_cloned, state_final = gas.step(state)

        # State should change after step
        assert not torch.allclose(state_final.x, state.x)
        assert not torch.allclose(state_final.v, state.v)

    def test_step_cloned_differs_from_original(self, euclidean_gas_params):
        """Test that cloned state differs from original."""
        gas = EuclideanGas(euclidean_gas_params)

        torch.manual_seed(42)
        state = gas.initialize_state()

        torch.manual_seed(43)
        state_cloned, _ = gas.step(state)

        # Cloned state should differ from original
        assert not torch.allclose(state_cloned.x, state.x)

    def test_step_final_differs_from_cloned(self, euclidean_gas_params):
        """Test that final state differs from cloned state."""
        gas = EuclideanGas(euclidean_gas_params)
        state = gas.initialize_state()

        torch.manual_seed(42)
        state_cloned, state_final = gas.step(state)

        # Final should differ from cloned (kinetic evolution)
        assert not torch.allclose(state_final.x, state_cloned.x)

    def test_run_basic(self, small_swarm_params):
        """Test basic run functionality."""
        gas = EuclideanGas(small_swarm_params)

        n_steps = 10
        torch.manual_seed(42)
        results = gas.run(n_steps)

        assert "x" in results
        assert "v" in results
        assert "var_x" in results
        assert "var_v" in results

    def test_run_trajectory_shapes(self, small_swarm_params):
        """Test that run returns correct trajectory shapes."""
        gas = EuclideanGas(small_swarm_params)

        n_steps = 10
        N, d = small_swarm_params.N, small_swarm_params.d

        torch.manual_seed(42)
        results = gas.run(n_steps)

        assert results["x"].shape == (n_steps + 1, N, d)
        assert results["v"].shape == (n_steps + 1, N, d)
        assert results["var_x"].shape == (n_steps + 1,)
        assert results["var_v"].shape == (n_steps + 1,)

    def test_run_stores_initial_state(self, small_swarm_params):
        """Test that run stores the initial state."""
        gas = EuclideanGas(small_swarm_params)

        N, d = small_swarm_params.N, small_swarm_params.d
        x_init = torch.ones(N, d)
        v_init = torch.zeros(N, d)

        torch.manual_seed(42)
        results = gas.run(n_steps=5, x_init=x_init, v_init=v_init)

        # Check initial state is stored
        assert torch.allclose(results["x"][0], x_init.to(dtype=torch.float64))
        assert torch.allclose(results["v"][0], v_init.to(dtype=torch.float64))

    def test_run_zero_steps(self, small_swarm_params):
        """Test run with zero steps returns only initial state."""
        gas = EuclideanGas(small_swarm_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=0)

        assert results["x"].shape[0] == 1
        assert results["v"].shape[0] == 1
        assert results["var_x"].shape[0] == 1
        assert results["var_v"].shape[0] == 1

    def test_run_multiple_times_different_results(self, small_swarm_params):
        """Test that multiple runs without fixed seed give different results."""
        gas = EuclideanGas(small_swarm_params)

        results1 = gas.run(n_steps=10)
        results2 = gas.run(n_steps=10)

        # Should be different (stochastic)
        assert not torch.allclose(results1["x"], results2["x"])

    def test_run_reproducible_with_seed(self, small_swarm_params):
        """Test that runs are reproducible with same seed."""
        gas = EuclideanGas(small_swarm_params)

        torch.manual_seed(42)
        results1 = gas.run(n_steps=10)

        torch.manual_seed(42)
        results2 = gas.run(n_steps=10)

        assert torch.allclose(results1["x"], results2["x"])
        assert torch.allclose(results1["v"], results2["v"])

    def test_variance_tracking(self, small_swarm_params):
        """Test that variances are tracked correctly."""
        gas = EuclideanGas(small_swarm_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=10)

        # All variances should be non-negative
        assert torch.all(results["var_x"] >= 0)
        assert torch.all(results["var_v"] >= 0)

    @pytest.mark.parametrize("n_steps", [1, 5, 20])
    def test_run_different_lengths(self, small_swarm_params, n_steps):
        """Test runs of different lengths."""
        gas = EuclideanGas(small_swarm_params)

        torch.manual_seed(42)
        results = gas.run(n_steps)

        assert results["x"].shape[0] == n_steps + 1
        assert results["v"].shape[0] == n_steps + 1

    def test_device_consistency(self, euclidean_gas_params):
        """Test that all tensors stay on correct device."""
        gas = EuclideanGas(euclidean_gas_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=5)

        assert results["x"].device == torch.device("cpu")
        assert results["v"].device == torch.device("cpu")
        assert results["var_x"].device == torch.device("cpu")
        assert results["var_v"].device == torch.device("cpu")

    def test_dtype_consistency(self, euclidean_gas_params):
        """Test that all tensors have correct dtype."""
        gas = EuclideanGas(euclidean_gas_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=5)

        assert results["x"].dtype == torch.float64
        assert results["v"].dtype == torch.float64
        assert results["var_x"].dtype == torch.float64
        assert results["var_v"].dtype == torch.float64

    def test_small_swarm(self, small_swarm_params):
        """Test with small swarm."""
        gas = EuclideanGas(small_swarm_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=10)

        assert results["x"].shape[1] == small_swarm_params.N

    def test_large_swarm(self, large_swarm_params):
        """Test with large swarm."""
        gas = EuclideanGas(large_swarm_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=5)  # Fewer steps for speed

        assert results["x"].shape[1] == large_swarm_params.N
        assert results["x"].shape[2] == large_swarm_params.d

    def test_initial_velocity_distribution(self, euclidean_gas_params):
        """Test that default initial velocities follow thermal distribution."""
        gas = EuclideanGas(euclidean_gas_params)

        # Generate many initial states
        torch.manual_seed(42)
        states = [gas.initialize_state() for _ in range(100)]

        # Collect all velocities
        all_v = torch.cat([s.v for s in states], dim=0)

        # Mean should be near zero
        mean_v = torch.mean(all_v, dim=0)
        assert torch.allclose(mean_v, torch.zeros_like(mean_v), atol=0.5)

        # Variance should be approximately 1/beta per dimension
        beta = euclidean_gas_params.langevin.beta
        expected_var = torch.tensor(1.0 / beta, dtype=torch.float64)
        actual_var = torch.var(all_v[:, 0])
        assert torch.allclose(actual_var, expected_var, rtol=0.3)

    def test_trajectory_continuity(self, small_swarm_params):
        """Test that trajectory is continuous (no jumps)."""
        gas = EuclideanGas(small_swarm_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=10)

        # Check that consecutive positions don't jump too much
        # Note: cloning can cause larger jumps, so we use a generous bound
        for t in range(10):
            dx = results["x"][t+1] - results["x"][t]
            max_displacement = torch.max(torch.norm(dx, dim=-1))

            # With dt=0.01 and cloning, max displacement can be larger
            assert max_displacement < 5.0
