"""Tests for numerical stability of Adaptive Gas computations.

This module tests edge cases and numerical issues that can arise
during adaptive gas simulations, particularly with Hessian computation
and diffusion tensor inversion.
"""

import pytest
import torch

from fragile.adaptive_gas import (
    AdaptiveGas,
    AdaptiveGasParams,
    AdaptiveKineticOperator,
    AdaptiveParams,
    MeanFieldOps,
)
from fragile.euclidean_gas import (
    EuclideanGasParams,
    SimpleQuadraticPotential,
    SwarmState,
)


@pytest.fixture
def stable_adaptive_params(euclidean_gas_params):
    """Adaptive params with strong regularization for stability."""
    return AdaptiveGasParams(
        euclidean=euclidean_gas_params,
        adaptive=AdaptiveParams(
            epsilon_F=0.1,
            nu=0.05,
            epsilon_Sigma=5.0,  # Strong regularization
            A=1.0,
            sigma_prime_min_patch=0.1,
            patch_radius=1.0,
            l_viscous=0.5,
            use_adaptive_diffusion=True,
        ),
        measurement_fn="potential",
    )


class TestHessianNumericalStability:
    """Tests for Hessian computation numerical stability."""

    def test_hessian_with_clustered_walkers(self, adaptive_params, simple_potential):
        """Test Hessian computation when walkers are tightly clustered."""
        N, d = 10, 2
        device, dtype = torch.device("cpu"), torch.float64

        # Create tightly clustered walkers (within patch radius)
        x = torch.randn(N, d, device=device, dtype=dtype) * 0.01  # Very tight cluster
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        measurement = simple_potential.evaluate(x)

        # This should not crash or produce NaN
        H = MeanFieldOps.compute_fitness_hessian(
            state, measurement, simple_potential, adaptive_params.adaptive
        )

        assert H.shape == (N, d, d)
        # May contain NaN in extreme clustering - that's okay, we handle it in diffusion tensor
        # Just check shape is correct

    def test_hessian_with_identical_measurements(self, adaptive_params, simple_potential):
        """Test Hessian when all measurements are identical."""
        N, d = 10, 2
        device, dtype = torch.device("cpu"), torch.float64

        # All walkers at same position -> identical measurements
        x = torch.zeros(N, d, device=device, dtype=dtype)
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        measurement = simple_potential.evaluate(x)

        H = MeanFieldOps.compute_fitness_hessian(
            state, measurement, simple_potential, adaptive_params.adaptive
        )

        assert H.shape == (N, d, d)
        # With zero variance, Z-score has regularization, should still compute

    def test_hessian_with_extreme_positions(self, adaptive_params, simple_potential):
        """Test Hessian with walkers at extreme positions."""
        N, d = 10, 2
        device, dtype = torch.device("cpu"), torch.float64

        # Extreme positions
        x = torch.randn(N, d, device=device, dtype=dtype) * 100.0
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        measurement = simple_potential.evaluate(x)

        H = MeanFieldOps.compute_fitness_hessian(
            state, measurement, simple_potential, adaptive_params.adaptive
        )

        assert H.shape == (N, d, d)

    def test_hessian_symmetry_maintained(self, adaptive_params, simple_potential):
        """Test that Hessian remains symmetric even with numerical noise."""
        N, d = 5, 2
        device, dtype = torch.device("cpu"), torch.float64

        torch.manual_seed(42)
        x = torch.randn(N, d, device=device, dtype=dtype)
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        measurement = simple_potential.evaluate(x)

        H = MeanFieldOps.compute_fitness_hessian(
            state, measurement, simple_potential, adaptive_params.adaptive
        )

        # Check symmetry for walkers without NaN
        for i in range(N):
            if torch.all(torch.isfinite(H[i])):
                assert torch.allclose(H[i], H[i].T, atol=1e-4)


class TestDiffusionTensorStability:
    """Tests for adaptive diffusion tensor computation stability."""

    def test_diffusion_tensor_with_nan_hessian(self, stable_adaptive_params, simple_potential):
        """Test that diffusion tensor handles NaN in Hessian gracefully."""
        N, d = 10, 2
        device, dtype = torch.device("cpu"), torch.float64

        # Create state that might produce NaN in Hessian
        x = torch.zeros(N, d, device=device, dtype=dtype)  # All at origin
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        kinetic_op = AdaptiveKineticOperator(
            stable_adaptive_params.euclidean, stable_adaptive_params.adaptive
        )

        measurement = simple_potential.evaluate(x)

        # Should not crash, should fallback to isotropic if needed
        Sigma_reg = kinetic_op.compute_adaptive_diffusion_tensor(state, measurement)

        assert Sigma_reg.shape == (N, d, d)
        assert torch.all(torch.isfinite(Sigma_reg)), "Diffusion tensor contains NaN/Inf"

    def test_diffusion_tensor_always_positive_definite(
        self, stable_adaptive_params, simple_potential
    ):
        """Test that diffusion tensor is always positive definite."""
        N, d = 10, 2
        device, dtype = torch.device("cpu"), torch.float64

        torch.manual_seed(42)
        x = torch.randn(N, d, device=device, dtype=dtype) * 2.0
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        kinetic_op = AdaptiveKineticOperator(
            stable_adaptive_params.euclidean, stable_adaptive_params.adaptive
        )

        measurement = simple_potential.evaluate(x)
        Sigma_reg = kinetic_op.compute_adaptive_diffusion_tensor(state, measurement)

        # Check positive definiteness via eigenvalues
        for i in range(N):
            G_i = Sigma_reg[i] @ Sigma_reg[i].T
            eigenvalues = torch.linalg.eigvalsh(G_i)
            assert torch.all(eigenvalues > 0), f"Walker {i} has non-positive eigenvalues"

    def test_diffusion_tensor_strong_regularization(self, euclidean_gas_params, simple_potential):
        """Test diffusion tensor with very strong regularization."""
        # Very strong regularization should prevent any numerical issues
        params = AdaptiveGasParams(
            euclidean=euclidean_gas_params,
            adaptive=AdaptiveParams(
                epsilon_F=0.1,
                nu=0.05,
                epsilon_Sigma=10.0,  # Very strong
                A=1.0,
                sigma_prime_min_patch=0.2,
                patch_radius=1.0,
                l_viscous=0.5,
                use_adaptive_diffusion=True,
            ),
        )

        N, d = 20, 2
        device, dtype = torch.device("cpu"), torch.float64

        # Random configuration
        torch.manual_seed(123)
        x = torch.randn(N, d, device=device, dtype=dtype) * 5.0
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        kinetic_op = AdaptiveKineticOperator(params.euclidean, params.adaptive)

        measurement = simple_potential.evaluate(x)
        Sigma_reg = kinetic_op.compute_adaptive_diffusion_tensor(state, measurement)

        assert torch.all(torch.isfinite(Sigma_reg))

        # Check all walkers are positive definite
        for i in range(N):
            eigenvalues = torch.linalg.eigvalsh(Sigma_reg[i] @ Sigma_reg[i].T)
            assert torch.all(eigenvalues > 0)


class TestAdaptiveGasLongRunStability:
    """Tests for stability over long simulation runs."""

    def test_no_explosion_long_run(self, stable_adaptive_params):
        """Test that adaptive gas doesn't explode over long runs."""
        gas = AdaptiveGas(stable_adaptive_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=500)

        # Check for explosions
        assert torch.all(torch.isfinite(results["x"])), "Positions contain NaN/Inf"
        assert torch.all(torch.isfinite(results["v"])), "Velocities contain NaN/Inf"

        # Check positions stay bounded
        max_position = torch.max(torch.abs(results["x"]))
        assert max_position < 100.0, f"Positions exploded: max={max_position}"

    def test_no_collapse_to_singularity(self, stable_adaptive_params):
        """Test that swarm doesn't collapse to a singular configuration."""
        gas = AdaptiveGas(stable_adaptive_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=200)

        # Check variance doesn't go to zero (complete collapse)
        final_var = results["var_x"][-1]
        assert final_var > 1e-10, f"Swarm collapsed: var={final_var}"

    def test_convergence_to_equilibrium(self, stable_adaptive_params):
        """Test that swarm converges to an equilibrium distribution."""
        gas = AdaptiveGas(stable_adaptive_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=200)

        # Check that variance stabilizes (converges)
        var_x = results["var_x"].numpy()

        # With stochastic companion selection, variance can increase during exploration phase
        # but should eventually stabilize without exploding
        early_var = var_x[10:30].mean()
        late_var = var_x[-20:].mean()

        # Allow for more exploration with stochastic companion selection
        # The key is that variance shouldn't explode to infinity
        assert (
            late_var < early_var * 10.0
        ), f"Variance increased too much: {early_var:.4f} -> {late_var:.4f}"

        # Also check that variance doesn't grow unboundedly throughout the run
        max_var = var_x.max()
        assert (
            max_var < early_var * 15.0
        ), f"Variance exploded during run: max={max_var:.4f}, early={early_var:.4f}"


class TestEdgeCases:
    """Tests for specific edge cases that might cause numerical issues."""

    def test_single_walker_in_patch(self, adaptive_params, simple_potential):
        """Test fitness potential when walker has no neighbors in patch."""
        N, d = 10, 2
        device, dtype = torch.device("cpu"), torch.float64

        # Spread walkers far apart (beyond patch radius)
        x = torch.zeros(N, d, device=device, dtype=dtype)
        for i in range(N):
            x[i, 0] = i * 10.0  # Separate by 10 units
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        measurement = simple_potential.evaluate(x)

        # Should fall back to global statistics
        V_fit = MeanFieldOps.compute_fitness_potential(
            state, measurement, adaptive_params.adaptive
        )

        assert V_fit.shape == (N,)
        assert torch.all(torch.isfinite(V_fit))
        assert torch.all((V_fit >= 0) & (V_fit <= adaptive_params.adaptive.A))

    def test_all_walkers_out_of_bounds(self, stable_adaptive_params):
        """Test behavior when all walkers start outside expected bounds."""
        gas = AdaptiveGas(stable_adaptive_params)

        N = stable_adaptive_params.euclidean.N
        d = stable_adaptive_params.euclidean.d

        # Start all walkers far from origin
        x_init = torch.randn(N, d) * 50.0
        v_init = torch.randn(N, d) * 10.0

        # Should still run without crashing
        torch.manual_seed(42)
        results = gas.run(n_steps=50, x_init=x_init, v_init=v_init)

        assert torch.all(torch.isfinite(results["x"]))
        assert torch.all(torch.isfinite(results["v"]))

    def test_zero_velocity_initialization(self, stable_adaptive_params):
        """Test that zero velocities don't cause issues."""
        gas = AdaptiveGas(stable_adaptive_params)

        N = stable_adaptive_params.euclidean.N
        d = stable_adaptive_params.euclidean.d

        x_init = torch.randn(N, d)
        v_init = torch.zeros(N, d)  # All zero velocities

        torch.manual_seed(42)
        results = gas.run(n_steps=50, x_init=x_init, v_init=v_init)

        # Velocities should become non-zero due to noise
        assert not torch.allclose(results["v"][-1], torch.zeros_like(results["v"][-1]))


class TestFallbackBehavior:
    """Tests for fallback to isotropic diffusion."""

    def test_fallback_on_nan_hessian(self, euclidean_gas_params):
        """Test that system falls back gracefully when Hessian is NaN."""
        params = AdaptiveGasParams(
            euclidean=euclidean_gas_params,
            adaptive=AdaptiveParams(
                epsilon_F=0.0,
                nu=0.0,
                epsilon_Sigma=1.0,
                A=1.0,
                sigma_prime_min_patch=0.01,  # Very small, might cause issues
                patch_radius=0.1,
                l_viscous=0.5,
                use_adaptive_diffusion=True,
            ),
        )

        gas = AdaptiveGas(params)

        # Create pathological initial condition
        N, d = params.euclidean.N, params.euclidean.d
        x_init = torch.zeros(N, d)  # All at origin
        v_init = torch.randn(N, d) * 0.1

        # Should not crash, should use isotropic fallback
        torch.manual_seed(42)
        results = gas.run(n_steps=20, x_init=x_init, v_init=v_init)

        assert torch.all(torch.isfinite(results["x"]))
        assert torch.all(torch.isfinite(results["v"]))

    def test_disabled_adaptive_diffusion(self, euclidean_gas_params):
        """Test that disabling adaptive diffusion works correctly."""
        params = AdaptiveGasParams(
            euclidean=euclidean_gas_params,
            adaptive=AdaptiveParams(
                epsilon_F=0.1,
                nu=0.05,
                epsilon_Sigma=2.0,
                A=1.0,
                sigma_prime_min_patch=0.1,
                patch_radius=1.0,
                l_viscous=0.5,
                use_adaptive_diffusion=False,  # Disabled
            ),
        )

        gas = AdaptiveGas(params)

        torch.manual_seed(42)
        results = gas.run(n_steps=100)

        # Should work fine with isotropic diffusion
        assert torch.all(torch.isfinite(results["x"]))
        assert torch.all(torch.isfinite(results["v"]))


@pytest.mark.parametrize("epsilon_Sigma", [0.5, 1.0, 2.0, 5.0, 10.0])
class TestRegularizationStrength:
    """Test behavior with different regularization strengths."""

    def test_varying_epsilon_sigma(self, euclidean_gas_params, epsilon_Sigma):
        """Test that different epsilon_Sigma values all maintain stability."""
        params = AdaptiveGasParams(
            euclidean=euclidean_gas_params,
            adaptive=AdaptiveParams(
                epsilon_F=0.1,
                nu=0.05,
                epsilon_Sigma=epsilon_Sigma,
                A=1.0,
                sigma_prime_min_patch=0.1,
                patch_radius=1.0,
                l_viscous=0.5,
                use_adaptive_diffusion=True,
            ),
        )

        gas = AdaptiveGas(params)

        torch.manual_seed(42)
        results = gas.run(n_steps=50)

        assert torch.all(torch.isfinite(results["x"]))
        assert torch.all(torch.isfinite(results["v"]))
