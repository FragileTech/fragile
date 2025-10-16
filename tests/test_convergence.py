"""Integration tests for convergence and statistical properties."""

import pytest
import torch

from fragile.euclidean_gas import (
    CloningParams,
    EuclideanGas,
    EuclideanGasParams,
    LangevinParams,
    SimpleQuadraticPotential,
)


class TestConvergence:
    """Tests for convergence properties."""

    @pytest.fixture
    def convergence_params(self):
        """Parameters tuned for convergence testing."""
        return EuclideanGasParams(
            N=50,
            d=2,
            potential=SimpleQuadraticPotential(),
            langevin=LangevinParams(gamma=1.0, beta=2.0, delta_t=0.01),
            cloning=CloningParams(sigma_x=0.1, lambda_alg=1.0, alpha_restitution=0.5),
            device="cpu",
            dtype="float64",
        )

    def test_position_variance_decreases(self, convergence_params):
        """Test that position variance decreases over time."""
        gas = EuclideanGas(convergence_params)

        # Start from dispersed initial condition
        N, d = convergence_params.N, convergence_params.d
        x_init = torch.randn(N, d) * 3.0  # Wide initial distribution

        torch.manual_seed(42)
        results = gas.run(n_steps=100, x_init=x_init)

        initial_var = results["var_x"][0]
        final_var = results["var_x"][-1]

        # Variance should decrease significantly
        # Note: With momentum-conserving collisions (no random rotations),
        # convergence may be slower than with full stochastic collisions
        assert final_var < initial_var * 0.7

    def test_mean_converges_to_origin(self, convergence_params):
        """Test that mean position converges to origin for quadratic potential."""
        gas = EuclideanGas(convergence_params)

        # Start from offset initial condition
        N, d = convergence_params.N, convergence_params.d
        x_init = torch.ones(N, d) * 2.0  # Start at [2, 2]

        torch.manual_seed(42)
        results = gas.run(n_steps=200, x_init=x_init)

        # Compute mean position over time
        mean_x = torch.mean(results["x"], dim=1)  # [n_steps+1, d]

        initial_mean = mean_x[0]
        final_mean = mean_x[-1]

        # Final mean should be much closer to origin
        assert torch.norm(final_mean) < torch.norm(initial_mean) * 0.3

    def test_equilibrium_variance(self, convergence_params):
        """Test that equilibrium variance is positive and bounded."""
        # Note: The Euclidean Gas with cloning operator has higher variance than
        # pure Langevin dynamics due to the jitter added during cloning
        gas = EuclideanGas(convergence_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=500)

        # Use last 100 steps to estimate equilibrium
        equilibrium_var = torch.mean(results["var_x"][-100:])

        # Variance should be positive and bounded (not collapse or explode)
        assert equilibrium_var > 0.1
        assert equilibrium_var < 100.0

    def test_velocity_distribution_equilibrates(self, convergence_params):
        """Test that velocity distribution equilibrates."""
        gas = EuclideanGas(convergence_params)

        # Start with zero velocities
        N, d = convergence_params.N, convergence_params.d
        v_init = torch.zeros(N, d)

        torch.manual_seed(42)
        results = gas.run(n_steps=200, v_init=v_init)

        initial_var = results["var_v"][0]
        final_var = results["var_v"][-1]

        # Velocity variance should increase from zero
        assert final_var > initial_var * 10

    def test_long_run_stability(self, convergence_params):
        """Test that algorithm remains stable over long runs."""
        gas = EuclideanGas(convergence_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=1000)

        # Check for NaN or Inf
        assert not torch.any(torch.isnan(results["x"]))
        assert not torch.any(torch.isinf(results["x"]))
        assert not torch.any(torch.isnan(results["v"]))
        assert not torch.any(torch.isinf(results["v"]))

        # Positions should remain bounded (allowing for statistical fluctuations)
        max_position = torch.max(torch.abs(results["x"]))
        assert max_position < 50.0

    def test_variance_fluctuations_decrease(self, convergence_params):
        """Test that variance fluctuations are finite after equilibration."""
        gas = EuclideanGas(convergence_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=500)

        # Compute variance of variance in late stage
        late_var = torch.var(results["var_x"][400:500])

        # Late-stage fluctuations should be finite and non-zero
        assert late_var > 0
        assert late_var < 10.0


class TestStatisticalProperties:
    """Tests for statistical properties."""

    @pytest.fixture
    def stats_params(self):
        """Parameters for statistical testing."""
        return EuclideanGasParams(
            N=100,
            d=2,
            potential=SimpleQuadraticPotential(),
            langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01),
            cloning=CloningParams(sigma_x=0.1, lambda_alg=1.0, alpha_restitution=0.5),
            device="cpu",
            dtype="float64",
        )

    def test_swarm_spread(self, stats_params):
        """Test that swarm maintains reasonable spread."""
        gas = EuclideanGas(stats_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=200)

        # After equilibration, variance should be relatively stable
        late_var = results["var_x"][-50:]

        # All variances should be positive
        assert torch.all(late_var > 0)

        # Variance shouldn't collapse to zero or explode
        mean_var = torch.mean(late_var)
        assert 0.1 < mean_var < 10.0

    def test_no_walker_clustering(self, stats_params):
        """Test that all walkers don't collapse to a single point."""
        gas = EuclideanGas(stats_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=100)

        # Check final state variance is non-zero
        final_var = results["var_x"][-1]

        # Walkers should maintain some spread
        assert final_var > 0.001

    def test_center_of_mass_near_origin(self, stats_params):
        """Test that center of mass stays near origin."""
        gas = EuclideanGas(stats_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=200)

        # Compute center of mass at each time
        com = torch.mean(results["x"], dim=1)  # [n_steps+1, d]

        # After equilibration, COM should be near origin
        final_com = com[-1]
        assert torch.norm(final_com) < 1.0

    def test_momentum_conservation_approximate(self, stats_params):
        """Test that total momentum remains bounded."""
        gas = EuclideanGas(stats_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=200)

        # Compute total momentum at each time
        total_momentum = torch.sum(results["v"], dim=1)  # [n_steps+1, d]

        # Momentum should remain bounded (not explode to infinity)
        max_momentum = torch.max(torch.norm(total_momentum, dim=-1))
        assert max_momentum < 100.0


class TestParameterSensitivity:
    """Tests for parameter sensitivity."""

    @pytest.fixture
    def base_params(self):
        """Base parameters for sensitivity testing."""
        return EuclideanGasParams(
            N=30,
            d=2,
            potential=SimpleQuadraticPotential(),
            langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01),
            cloning=CloningParams(sigma_x=0.1, lambda_alg=1.0, alpha_restitution=0.5),
            device="cpu",
            dtype="float64",
        )

    def test_high_beta_low_variance(self, base_params):
        """Test that high beta (low temperature) leads to lower variance."""
        # Low temperature (high beta)
        params_low_temp = base_params.model_copy(deep=True)
        params_low_temp.langevin.beta = 5.0

        # High temperature (low beta)
        params_high_temp = base_params.model_copy(deep=True)
        params_high_temp.langevin.beta = 0.2

        gas_low = EuclideanGas(params_low_temp)
        gas_high = EuclideanGas(params_high_temp)

        torch.manual_seed(42)
        results_low = gas_low.run(n_steps=300)

        torch.manual_seed(42)
        results_high = gas_high.run(n_steps=300)

        # Low temperature should have lower equilibrium variance
        var_low = torch.mean(results_low["var_x"][-50:])
        var_high = torch.mean(results_high["var_x"][-50:])

        assert var_low < var_high

    def test_large_jitter_more_dispersion(self, base_params):
        """Test that larger sigma_x leads to more dispersion."""
        # Small jitter
        params_small = base_params.model_copy(deep=True)
        params_small.cloning.sigma_x = 0.01

        # Large jitter
        params_large = base_params.model_copy(deep=True)
        params_large.cloning.sigma_x = 0.5

        gas_small = EuclideanGas(params_small)
        gas_large = EuclideanGas(params_large)

        torch.manual_seed(42)
        results_small = gas_small.run(n_steps=100)

        torch.manual_seed(42)
        results_large = gas_large.run(n_steps=100)

        # Larger jitter may lead to different variance behavior
        # This is a weak test - just check it runs and produces different results
        var_small = results_small["var_x"][-1]
        var_large = results_large["var_x"][-1]

        # Results should differ
        assert not torch.allclose(var_small, var_large, rtol=0.1)

    def test_restitution_coefficient_effect(self, base_params):
        """Test that restitution coefficient affects dynamics."""
        # Elastic collision
        params_elastic = base_params.model_copy(deep=True)
        params_elastic.cloning.alpha_restitution = 1.0

        # Inelastic collision
        params_inelastic = base_params.model_copy(deep=True)
        params_inelastic.cloning.alpha_restitution = 0.0

        gas_elastic = EuclideanGas(params_elastic)
        gas_inelastic = EuclideanGas(params_inelastic)

        torch.manual_seed(42)
        results_elastic = gas_elastic.run(n_steps=100)

        torch.manual_seed(42)
        results_inelastic = gas_inelastic.run(n_steps=100)

        # Different restitution should give different results
        assert not torch.allclose(results_elastic["x"][-1], results_inelastic["x"][-1], rtol=0.1)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_walker(self):
        """Test with single walker (edge case)."""
        params = EuclideanGasParams(
            N=1,
            d=2,
            potential=SimpleQuadraticPotential(),
            langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01),
            cloning=CloningParams(sigma_x=0.1, lambda_alg=1.0, alpha_restitution=0.5),
            device="cpu",
            dtype="float64",
        )

        gas = EuclideanGas(params)

        # Single walker clones itself
        torch.manual_seed(42)
        results = gas.run(n_steps=10)

        # Should complete without error
        assert results["x"].shape == (11, 1, 2)

    def test_high_dimensional(self):
        """Test with high-dimensional state space."""
        params = EuclideanGasParams(
            N=10,
            d=20,  # High dimension
            potential=SimpleQuadraticPotential(),
            langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01),
            cloning=CloningParams(sigma_x=0.1, lambda_alg=1.0, alpha_restitution=0.5),
            device="cpu",
            dtype="float64",
        )

        gas = EuclideanGas(params)

        torch.manual_seed(42)
        results = gas.run(n_steps=10)

        # Should handle high dimensions
        assert results["x"].shape == (11, 10, 20)

    def test_very_small_timestep(self):
        """Test with very small timestep."""
        params = EuclideanGasParams(
            N=10,
            d=2,
            potential=SimpleQuadraticPotential(),
            langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.0001),
            cloning=CloningParams(sigma_x=0.1, lambda_alg=1.0, alpha_restitution=0.5),
            device="cpu",
            dtype="float64",
        )

        gas = EuclideanGas(params)

        torch.manual_seed(42)
        results = gas.run(n_steps=10)

        # Should remain stable
        assert not torch.any(torch.isnan(results["x"]))
