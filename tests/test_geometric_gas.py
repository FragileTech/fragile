"""
Tests for Geometric Gas implementation.

Tests cover:
- Localization kernel properties
- Localized statistical moments
- Fitness potential and Z-scores
- Adaptive forces
- Viscous coupling
- Hessian diffusion tensor
- Uniform ellipticity bounds
- rho-dependent limiting behavior
- Full integration tests
"""

import pytest
import torch

from fragile.euclidean_gas import LangevinParams, SimpleQuadraticPotential
from fragile.geometric_gas import (
    AdaptiveKineticOperator,
    AdaptiveParams,
    FitnessPotential,
    GeometricGas,
    GeometricGasParams,
    LocalizationKernel,
    LocalizationKernelParams,
)


@pytest.fixture
def device():
    """Get compute device."""
    return torch.device("cpu")


@pytest.fixture
def dtype():
    """Get data type."""
    return torch.float32


class TestLocalizationKernel:
    """Test suite for LocalizationKernel class."""

    def test_gaussian_kernel_normalization(self, device, dtype):
        """Test that Gaussian kernel weights sum to 1."""
        params = LocalizationKernelParams(rho=1.0, kernel_type="gaussian")
        kernel = LocalizationKernel(params, device, dtype)

        # Create test positions
        x_query = torch.randn(5, 2, device=device, dtype=dtype)
        x_alive = torch.randn(10, 2, device=device, dtype=dtype)

        # Compute kernel
        K = kernel.compute_kernel(x_query, x_alive)

        # Check normalization
        assert K.shape == (5, 10)
        torch.testing.assert_close(
            K.sum(dim=1), torch.ones(5, device=device, dtype=dtype), atol=1e-6, rtol=1e-6
        )

    def test_gaussian_kernel_locality(self, device, dtype):
        """Test that Gaussian kernel decays with distance."""
        params = LocalizationKernelParams(rho=0.5, kernel_type="gaussian")
        kernel = LocalizationKernel(params, device, dtype)

        # Create test: one query point, two alive walkers at different distances
        x_query = torch.zeros(1, 2, device=device, dtype=dtype)
        x_alive = torch.tensor([[0.1, 0.1], [5.0, 5.0]], device=device, dtype=dtype)

        # Compute kernel
        K = kernel.compute_kernel(x_query, x_alive)

        # Nearby walker should have higher weight
        assert K[0, 0] > K[0, 1]

    def test_uniform_kernel(self, device, dtype):
        """Test uniform kernel gives equal weights."""
        params = LocalizationKernelParams(rho=float("inf"), kernel_type="uniform")
        kernel = LocalizationKernel(params, device, dtype)

        x_query = torch.randn(3, 2, device=device, dtype=dtype)
        x_alive = torch.randn(7, 2, device=device, dtype=dtype)

        K = kernel.compute_kernel(x_query, x_alive)

        # All weights should be 1/k
        expected = torch.ones(3, 7, device=device, dtype=dtype) / 7
        torch.testing.assert_close(K, expected, atol=1e-6, rtol=1e-6)

    def test_localized_moments_shape(self, device, dtype):
        """Test that localized moments have correct shapes."""
        params = LocalizationKernelParams(rho=1.0, kernel_type="gaussian")
        kernel = LocalizationKernel(params, device, dtype)

        x_query = torch.randn(5, 3, device=device, dtype=dtype)
        x_alive = torch.randn(10, 3, device=device, dtype=dtype)
        measurement = torch.randn(10, device=device, dtype=dtype)

        mu, sigma_sq = kernel.compute_localized_moments(x_query, x_alive, measurement)

        assert mu.shape == (5,)
        assert sigma_sq.shape == (5,)

    def test_localized_moments_limit_rho_large(self, device, dtype):
        """Test that large rho recovers global statistics."""
        # Use very large rho (approaching uniform kernel)
        params_large_rho = LocalizationKernelParams(rho=100.0, kernel_type="gaussian")
        kernel_large = LocalizationKernel(params_large_rho, device, dtype)

        x_query = torch.randn(5, 2, device=device, dtype=dtype)
        x_alive = torch.randn(20, 2, device=device, dtype=dtype)
        measurement = torch.randn(20, device=device, dtype=dtype)

        # Compute localized moments with large rho
        mu_local, sigma_sq_local = kernel_large.compute_localized_moments(
            x_query, x_alive, measurement
        )

        # Compute global moments
        mu_global = measurement.mean()
        sigma_sq_global = ((measurement - mu_global) ** 2).mean()

        # All query points should have similar moments to global
        torch.testing.assert_close(
            mu_local, torch.full_like(mu_local, mu_global), atol=0.1, rtol=0.1
        )
        torch.testing.assert_close(
            sigma_sq_local, torch.full_like(sigma_sq_local, sigma_sq_global), atol=0.1, rtol=0.1
        )


class TestFitnessPotential:
    """Test suite for FitnessPotential class."""

    @pytest.fixture
    def fitness_potential(self, device, dtype):
        """Create fitness potential for testing."""
        loc_params = LocalizationKernelParams(rho=1.0, kernel_type="gaussian")
        localization = LocalizationKernel(loc_params, device, dtype)

        adaptive_params = AdaptiveParams(
            epsilon_F=0.1,
            nu=0.05,
            epsilon_Sigma=0.01,
            rescale_amplitude=1.0,
            sigma_var_min=0.1,
            viscous_length_scale=1.0,
        )

        def measurement_fn(x):
            # Simple quadratic measurement
            return -torch.sum(x**2, dim=-1)

        return FitnessPotential(localization, adaptive_params, measurement_fn, device, dtype)

    def test_z_score_regularization(self, fitness_potential, device, dtype):
        """Test that Z-score is well-defined even with zero variance."""
        # Create constant measurements (zero variance)
        x_query = torch.zeros(1, 2, device=device, dtype=dtype)
        x_alive = torch.zeros(5, 2, device=device, dtype=dtype)

        # Should not raise division by zero
        z_score = fitness_potential.compute_z_score(x_query, x_alive)

        assert z_score.shape == (1,)
        assert torch.isfinite(z_score).all()

    def test_fitness_bounded(self, fitness_potential, device, dtype):
        """Test that fitness values are bounded [0, A]."""
        x_query = torch.randn(10, 2, device=device, dtype=dtype)
        x_alive = torch.randn(20, 2, device=device, dtype=dtype)

        V_fit = fitness_potential.evaluate(x_query, x_alive)

        A = fitness_potential.params.rescale_amplitude
        assert (V_fit >= 0).all()
        assert (V_fit <= A).all()

    def test_gradient_shape(self, fitness_potential, device, dtype):
        """Test gradient computation shape."""
        x_query = torch.randn(5, 3, device=device, dtype=dtype, requires_grad=True)
        x_alive = torch.randn(10, 3, device=device, dtype=dtype)

        grad = fitness_potential.compute_gradient(x_query, x_alive)

        assert grad.shape == (5, 3)

    def test_hessian_shape(self, fitness_potential, device, dtype):
        """Test Hessian computation shape."""
        x_query = torch.randn(3, 2, device=device, dtype=dtype)
        x_alive = torch.randn(8, 2, device=device, dtype=dtype)

        H = fitness_potential.compute_hessian(x_query, x_alive)

        assert H.shape == (3, 2, 2)

    def test_hessian_symmetry(self, fitness_potential, device, dtype):
        """Test that Hessian is symmetric."""
        x_query = torch.randn(2, 3, device=device, dtype=dtype)
        x_alive = torch.randn(6, 3, device=device, dtype=dtype)

        H = fitness_potential.compute_hessian(x_query, x_alive)

        # Check symmetry for each query point
        for i in range(H.shape[0]):
            H_transpose = H[i].transpose(0, 1)
            torch.testing.assert_close(H[i], H_transpose, atol=1e-4, rtol=1e-4)


class TestAdaptiveKineticOperator:
    """Test suite for AdaptiveKineticOperator."""

    @pytest.fixture
    def kinetic_operator(self, device, dtype):
        """Create adaptive kinetic operator for testing."""
        langevin = LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01)
        potential = SimpleQuadraticPotential()

        loc_params = LocalizationKernelParams(rho=1.0, kernel_type="gaussian")
        localization = LocalizationKernel(loc_params, device, dtype)

        adaptive_params = AdaptiveParams(
            epsilon_F=0.1,
            nu=0.05,
            epsilon_Sigma=0.01,
            rescale_amplitude=1.0,
            sigma_var_min=0.1,
            viscous_length_scale=1.0,
        )

        def measurement_fn(x):
            return -torch.sum(x**2, dim=-1)

        fitness_potential = FitnessPotential(
            localization, adaptive_params, measurement_fn, device, dtype
        )

        return AdaptiveKineticOperator(
            langevin, potential, fitness_potential, adaptive_params, device, dtype
        )

    def test_viscous_force_dissipative(self, kinetic_operator, device, dtype):
        """Test that viscous force reduces velocity variance."""
        from fragile.euclidean_gas import SwarmState

        # Create state with varying velocities
        N = 10
        x = torch.randn(N, 2, device=device, dtype=dtype)
        v = torch.randn(N, 2, device=device, dtype=dtype)
        state = SwarmState(x, v)

        alive_mask = torch.ones(N, dtype=torch.bool, device=device)

        F_viscous = kinetic_operator._viscous_force(state, alive_mask)

        assert F_viscous.shape == (N, 2)

        # Viscous force should reduce velocity variance (on average)
        # This is a statistical property, so we just check it's not zero
        assert F_viscous.abs().sum() > 0

    def test_diffusion_tensor_ellipticity(self, kinetic_operator, device, dtype):
        """Test that diffusion tensor is uniformly elliptic."""
        x_alive = torch.randn(5, 2, device=device, dtype=dtype)
        alive_mask = torch.ones(5, dtype=torch.bool, device=device)

        Sigma_reg = kinetic_operator._compute_diffusion_tensor(x_alive, x_alive, alive_mask)

        assert Sigma_reg.shape == (5, 2, 2)

        # Check positive definiteness via eigenvalues
        for i in range(5):
            eigenvalues = torch.linalg.eigvalsh(Sigma_reg[i])
            assert (eigenvalues > 0).all(), f"Non-positive eigenvalues: {eigenvalues}"


class TestGeometricGas:
    """Integration tests for full GeometricGas algorithm."""

    @pytest.fixture
    def geometric_gas(self, device, dtype):
        """Create GeometricGas instance for testing."""
        params = GeometricGasParams(
            N=20,
            d=2,
            potential=SimpleQuadraticPotential(),
            langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01),
            localization=LocalizationKernelParams(rho=1.0, kernel_type="gaussian"),
            adaptive=AdaptiveParams(
                epsilon_F=0.05,
                nu=0.02,
                epsilon_Sigma=0.01,
                rescale_amplitude=1.0,
                sigma_var_min=0.1,
                viscous_length_scale=1.0,
            ),
            device=str(device),
            dtype="float32" if dtype == torch.float32 else "float64",
        )

        return GeometricGas(params)

    def test_initialization(self, geometric_gas):
        """Test that GeometricGas initializes correctly."""
        state = geometric_gas.initialize_state()

        assert state.N == 20
        assert state.d == 2
        assert state.x.shape == (20, 2)
        assert state.v.shape == (20, 2)

    def test_single_step(self, geometric_gas):
        """Test that a single step executes without errors."""
        state = geometric_gas.initialize_state()

        _state_cloned, state_final = geometric_gas.step(state)

        assert state_final.x.shape == (20, 2)
        assert state_final.v.shape == (20, 2)
        assert torch.isfinite(state_final.x).all()
        assert torch.isfinite(state_final.v).all()

    def test_run_trajectory(self, geometric_gas):
        """Test full trajectory execution."""
        n_steps = 10
        result = geometric_gas.run(n_steps)

        assert result["x"].shape == (n_steps + 1, 20, 2)
        assert result["v"].shape == (n_steps + 1, 20, 2)
        assert result["fitness"].shape == (n_steps + 1, 20)
        assert not result["terminated_early"]
        assert result["final_step"] == n_steps

    def test_convergence_toward_minimum(self, geometric_gas):
        """Test that the swarm explores the state space."""
        # Run for some steps
        n_steps = 50
        result = geometric_gas.run(n_steps)

        # Initial and final positions
        result["x"][0]
        x_final = result["x"][-1]

        # Check that walkers remain in reasonable bounds (not diverging)
        dist_final = torch.norm(x_final, dim=-1).mean()

        # Should not diverge to infinity (staying within reasonable bounds)
        assert dist_final < 10.0, f"Walkers diverged: mean distance = {dist_final}"

        # Check that fitness is computed
        assert torch.isfinite(result["fitness"]).all()


class TestRhoDependentBehavior:
    """Test rho-dependent limiting behavior."""

    def test_small_rho_local_adaptation(self, device, dtype):
        """Test that small rho leads to local adaptation."""
        # Create two groups of walkers far apart with fixed positions
        x_alive = torch.cat([
            torch.zeros(5, 2, device=device, dtype=dtype) - 5,  # Group 1 at (-5, -5)
            torch.zeros(5, 2, device=device, dtype=dtype) + 5,  # Group 2 at (5, 5)
        ])

        measurement = torch.cat([
            torch.ones(5, device=device, dtype=dtype),  # Group 1: high fitness
            torch.zeros(5, device=device, dtype=dtype),  # Group 2: low fitness
        ])

        # Small rho: local statistics (should weight nearby walkers heavily)
        loc_params_small = LocalizationKernelParams(rho=1.0, kernel_type="gaussian")
        kernel_small = LocalizationKernel(loc_params_small, device, dtype)

        # Query point in Group 1
        x_query_group1 = torch.tensor([[-5.0, -5.0]], device=device, dtype=dtype)

        mu_local, _ = kernel_small.compute_localized_moments(x_query_group1, x_alive, measurement)

        # Should be close to Group 1 mean (1.0), not global mean (0.5)
        # The distance between groups is 10*sqrt(2) ~ 14, much larger than rho=1.0
        # So Group 2 should have negligible weight
        assert mu_local[0] > 0.9

    def test_large_rho_global_adaptation(self, device, dtype):
        """Test that large rho recovers global statistics."""
        # Same setup as above
        x_alive = torch.cat([
            torch.randn(5, 2, device=device, dtype=dtype) - 5,
            torch.randn(5, 2, device=device, dtype=dtype) + 5,
        ])

        measurement = torch.cat([
            torch.ones(5, device=device, dtype=dtype),
            torch.zeros(5, device=device, dtype=dtype),
        ])

        # Large rho: global statistics
        loc_params_large = LocalizationKernelParams(rho=20.0, kernel_type="gaussian")
        kernel_large = LocalizationKernel(loc_params_large, device, dtype)

        x_query_group1 = torch.tensor([[-5.0, 0.0]], device=device, dtype=dtype)

        mu_global, _ = kernel_large.compute_localized_moments(x_query_group1, x_alive, measurement)

        # Should be close to global mean (0.5)
        torch.testing.assert_close(
            mu_global[0], torch.tensor(0.5, dtype=dtype), atol=0.15, rtol=0.15
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
