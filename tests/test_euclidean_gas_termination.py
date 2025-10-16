"""Tests for early termination when all walkers die."""

import pytest
import torch

from fragile.bounds import TorchBounds
from fragile.euclidean_gas import EuclideanGas


class TestEarlyTermination:
    """Test that EuclideanGas stops when all walkers die."""

    def test_run_stops_when_all_walkers_die(self, euclidean_gas_params):
        """Run should stop early if all walkers go out of bounds."""
        # Setup: Extremely small bounds with large starting positions to ensure all escape
        N, d = euclidean_gas_params.N, euclidean_gas_params.d

        # Tiny bounds: [-0.01, 0.01]^d (essentially a point)
        bounds = TorchBounds(
            low=torch.tensor([-0.01] * d),
            high=torch.tensor([0.01] * d),
        )

        # Update params with bounds
        euclidean_gas_params.bounds = bounds
        gas = EuclideanGas(euclidean_gas_params)

        # Initialize far outside bounds so all walkers start dead
        # Even cloning can't save them if they all start out of bounds
        x_init = torch.full((N, d), 5.0)  # Far outside [-0.01, 0.01]

        # Run for many steps - should terminate immediately
        n_steps = 100
        result = gas.run(n_steps=n_steps, x_init=x_init)

        # Check that run terminated immediately (all start dead)
        assert result["terminated_early"], "Should have terminated early when all walkers died"
        assert result["final_step"] == 0, "Should terminate immediately if all walkers start dead"

        # Check that initial state has 0 alive walkers
        assert result["n_alive"][0].item() == 0, "Initial step should have 0 alive walkers"

        # Check trajectory lengths match final_step + 1 (should be 1 - just initial state)
        final_step = result["final_step"]
        assert result["x"].shape[0] == final_step + 1
        assert result["v"].shape[0] == final_step + 1
        assert result["var_x"].shape[0] == final_step + 1
        assert result["var_v"].shape[0] == final_step + 1
        assert result["n_alive"].shape[0] == final_step + 1

    def test_run_completes_normally_when_walkers_survive(self, euclidean_gas_params):
        """Run should complete all steps if walkers stay alive."""
        d = euclidean_gas_params.d

        # Large bounds: [-10, 10]^d
        bounds = TorchBounds(
            low=torch.tensor([-10.0] * d),
            high=torch.tensor([10.0] * d),
        )

        # Update params with bounds
        euclidean_gas_params.bounds = bounds
        gas = EuclideanGas(euclidean_gas_params)

        # Run for a few steps
        n_steps = 10
        result = gas.run(n_steps=n_steps)

        # Check that run completed normally
        assert not result["terminated_early"], "Should not terminate early with large bounds"
        assert result["final_step"] == n_steps, f"Should complete all {n_steps} steps"

        # Check that walkers are alive
        assert result["n_alive"][-1].item() > 0, "Should have alive walkers at end"

        # Check trajectory lengths
        assert result["x"].shape[0] == n_steps + 1
        assert result["v"].shape[0] == n_steps + 1

    def test_run_with_no_bounds(self, euclidean_gas_params):
        """Run without bounds should complete normally (all walkers always alive)."""
        N = euclidean_gas_params.N
        # Ensure no bounds set
        euclidean_gas_params.bounds = None
        gas = EuclideanGas(euclidean_gas_params)

        n_steps = 20
        result = gas.run(n_steps=n_steps)

        # Should complete all steps
        assert not result["terminated_early"]
        assert result["final_step"] == n_steps

        # All walkers should be alive (no bounds to violate)
        assert (result["n_alive"] == N).all(), "All walkers should be alive without bounds"

    def test_initially_all_dead(self, euclidean_gas_params):
        """If all walkers start dead, should return immediately."""
        N, d = euclidean_gas_params.N, euclidean_gas_params.d

        # Small bounds
        bounds = TorchBounds(
            low=torch.tensor([0.0] * d),
            high=torch.tensor([1.0] * d),
        )

        # Update params with bounds
        euclidean_gas_params.bounds = bounds
        gas = EuclideanGas(euclidean_gas_params)

        # Initialize outside bounds (all dead)
        x_init = torch.full((N, d), -10.0)

        result = gas.run(n_steps=10, x_init=x_init)

        # Should terminate immediately
        assert result["terminated_early"]
        assert result["final_step"] == 0
        assert result["n_alive"][0].item() == 0

        # Should only have initial state
        assert result["x"].shape[0] == 1
        assert result["v"].shape[0] == 1

    def test_n_alive_trajectory(self, euclidean_gas_params):
        """n_alive should be tracked over time."""
        d = euclidean_gas_params.d

        # Medium-sized bounds
        bounds = TorchBounds(
            low=torch.tensor([-2.0] * d),
            high=torch.tensor([2.0] * d),
        )

        # Update params with bounds
        euclidean_gas_params.bounds = bounds
        gas = EuclideanGas(euclidean_gas_params)

        result = gas.run(n_steps=20)

        n_alive_traj = result["n_alive"]

        # n_alive should be non-negative
        assert (n_alive_traj >= 0).all()

        # Initial n_alive should be > 0 (initialized within bounds)
        assert n_alive_traj[0].item() > 0

    def test_n_alive_in_result_keys(self, euclidean_gas_params):
        """Result dictionary should include n_alive trajectory."""
        gas = EuclideanGas(euclidean_gas_params)

        result = gas.run(n_steps=5)

        # Check keys
        assert "n_alive" in result
        assert "terminated_early" in result
        assert "final_step" in result

        # Check types
        assert isinstance(result["terminated_early"], bool)
        assert isinstance(result["final_step"], int)
        assert isinstance(result["n_alive"], torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
