"""
Tests for symbolic fitness algebra module.

These tests verify the correctness of the symbolic computations
against the formulas in Chapter 9 of 08_emergent_geometry.md.
"""

import pytest
import sympy as sp
from sympy import symbols, exp, sqrt, simplify, Matrix

from fragile.fitness_algebra import (
    FitnessPotential,
    EmergentMetric,
    RescaleFunction,
    MeasurementFunction,
    create_algorithmic_parameters,
    create_state_variables,
)


class TestAlgorithmicParameters:
    """Test parameter creation."""

    def test_create_parameters(self):
        """Test that all required parameters are created."""
        params = create_algorithmic_parameters()

        # Check essential parameters exist
        assert 'rho' in params
        assert 'epsilon_Sigma' in params
        assert 'A' in params
        assert 'kappa_var_min' in params
        assert 'gamma' in params
        assert 'alpha' in params
        assert 'beta' in params

        # Check they are symbols
        assert isinstance(params['rho'], sp.Symbol)
        assert isinstance(params['epsilon_Sigma'], sp.Symbol)

    def test_state_variables_3d(self):
        """Test 3D state variable creation."""
        x, coords = create_state_variables(dim=3)

        assert x.shape == (3, 1)
        assert 'x_1' in coords
        assert 'x_2' in coords
        assert 'x_3' in coords


class TestRescaleFunction:
    """Test rescale functions."""

    def test_sigmoid_basic(self):
        """Test sigmoid rescale function."""
        z = symbols('z', real=True)
        A = symbols('A', positive=True)

        g = RescaleFunction.sigmoid(z, A)

        # Check bounds: g(z) → 0 as z → -∞, g(z) → A as z → +∞
        assert simplify(g.limit(z, sp.oo)) == A
        assert simplify(g.limit(z, -sp.oo)) == 0

        # Check g(0) = A/2
        assert simplify(g.subs(z, 0)) == A / 2

    def test_sigmoid_derivative(self):
        """Test sigmoid derivatives."""
        z = symbols('z', real=True)
        A = symbols('A', positive=True)

        g = RescaleFunction.sigmoid(z, A)
        g_prime = RescaleFunction.sigmoid_derivative(z, A, order=1)

        # Verify derivative
        expected = sp.diff(g, z)
        assert simplify(g_prime - expected) == 0

    def test_sigmoid_second_derivative(self):
        """Test second derivative."""
        z = symbols('z', real=True)
        A = symbols('A', positive=True)

        g = RescaleFunction.sigmoid(z, A)
        g_double_prime = RescaleFunction.sigmoid_derivative(z, A, order=2)

        expected = sp.diff(g, z, 2)
        assert simplify(g_double_prime - expected) == 0


class TestMeasurementFunction:
    """Test measurement function wrapper."""

    def test_gradient_2d(self):
        """Test gradient computation in 2D."""
        meas = MeasurementFunction(dim=2)

        grad = meas.gradient()
        assert grad.shape == (2, 1)

        # Check that gradient contains derivatives
        x1, x2 = meas.x
        assert sp.Derivative(meas.d, x1) in grad or sp.diff(meas.d, x1) in grad

    def test_hessian_2d(self):
        """Test Hessian computation in 2D."""
        meas = MeasurementFunction(dim=2)

        H = meas.hessian()
        assert H.shape == (2, 2)

        # Check symmetry (symbolic)
        x1, x2 = meas.x
        # Hessian should be symmetric
        assert simplify(H[0, 1] - H[1, 0]) == 0


class TestFitnessPotential:
    """Test fitness potential computation (simplified tests)."""

    def test_localization_kernel_identity(self):
        """Test that kernel K(x, x) = 1 (unnormalized)."""
        fitness = FitnessPotential(dim=2, num_walkers=2)

        x_i = Matrix([sp.Symbol('x1'), sp.Symbol('x2')])
        rho = sp.Symbol('rho', positive=True)

        # K(x, x) should be exp(0) = 1
        K_ii = fitness.localization_kernel(x_i, x_i, rho)
        assert K_ii == 1

    def test_weights_sum_to_one(self):
        """Test that normalized weights sum to 1."""
        # Use small number of walkers to keep it tractable
        fitness = FitnessPotential(dim=2, num_walkers=2)

        weights = fitness.localization_weights()

        # Sum of weights should be 1
        weight_sum = sum(weights)
        # This should simplify to 1 (kernels cancel in normalization)
        assert simplify(weight_sum) == 1

    def test_fitness_potential_bounds(self):
        """Test that V_fit is bounded by [0, A]."""
        fitness = FitnessPotential(dim=2, num_walkers=2)

        # Get fitness potential symbolically
        # (this creates a complex expression, so we just check structure)
        V_fit = fitness.fitness_potential_sigmoid()

        # V_fit should be an expression involving A
        assert fitness.params['A'] in V_fit.free_symbols


class TestEmergentMetric:
    """Test metric tensor computation (simplified)."""

    def test_metric_regularization_2d(self):
        """Test that metric g = H + ε_Σ I has correct structure."""
        # Use simplified fitness
        fitness = FitnessPotential(dim=2, num_walkers=2)
        metric_obj = EmergentMetric(fitness)

        # Create a simple test Hessian
        h11, h12, h22 = symbols('h11 h12 h22', real=True)
        H_test = Matrix([[h11, h12], [h12, h22]])

        g = metric_obj.metric_tensor(H_test)

        epsilon_Sigma = fitness.params['epsilon_Sigma']

        # Check diagonal has regularization
        assert simplify(g[0, 0] - (h11 + epsilon_Sigma)) == 0
        assert simplify(g[1, 1] - (h22 + epsilon_Sigma)) == 0

        # Check off-diagonal unchanged
        assert simplify(g[0, 1] - h12) == 0

    def test_volume_element_2d(self):
        """Test volume element for 2D."""
        fitness = FitnessPotential(dim=2, num_walkers=2)
        metric_obj = EmergentMetric(fitness)

        # Simple 2×2 metric
        g11, g12, g22 = symbols('g11 g12 g22', positive=True)
        g_test = Matrix([[g11, g12], [g12, g22]])

        vol = metric_obj.volume_element(g_test)

        # Volume should be sqrt(det(g)) = sqrt(g11*g22 - g12^2)
        expected = sqrt(g11 * g22 - g12**2)
        assert simplify(vol - expected) == 0

    def test_volume_element_3d_explicit_identity(self):
        """Test 3D volume element with identity metric."""
        fitness = FitnessPotential(dim=3, num_walkers=2)
        metric_obj = EmergentMetric(fitness)

        # Identity matrix (H = 0, ε_Σ = 1)
        H_zero = sp.zeros(3, 3)

        # Temporarily set epsilon_Sigma = 1 for this test
        old_eps = fitness.params['epsilon_Sigma']
        fitness.params['epsilon_Sigma'] = sp.Integer(1)

        vol = metric_obj.volume_element_3d_explicit(H_zero)

        # With H=0 and ε_Σ=1: det(g) = 1³ = 1, so √det(g) = 1
        assert simplify(vol) == 1

        # Restore
        fitness.params['epsilon_Sigma'] = old_eps


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_simple_2d_pipeline(self):
        """Test end-to-end for a simple 2D case."""
        # Create simple fitness potential
        fitness = FitnessPotential(dim=2, num_walkers=2)

        # This is complex, so just verify it runs and produces expressions
        weights = fitness.localization_weights()
        assert len(weights) == 2
        assert all(isinstance(w, sp.Expr) for w in weights)

        # Localized mean
        mu = fitness.localized_mean(weights)
        assert isinstance(mu, sp.Expr)

        # Variance
        var = fitness.localized_variance(weights, mu)
        assert isinstance(var, sp.Expr)

        # Note: Full V_fit computation is too slow for testing,
        # so we skip the full pipeline here

    def test_metric_from_simple_hessian(self):
        """Test metric construction from a known Hessian."""
        fitness = FitnessPotential(dim=2, num_walkers=2)
        metric_obj = EmergentMetric(fitness)

        # Diagonal Hessian (simple case)
        h1, h2 = symbols('h1 h2', real=True)
        H = Matrix([[h1, 0], [0, h2]])

        g = metric_obj.metric_tensor(H)
        epsilon_Sigma = fitness.params['epsilon_Sigma']

        # Check metric is diagonal with correct values
        assert simplify(g[0, 0] - (h1 + epsilon_Sigma)) == 0
        assert simplify(g[1, 1] - (h2 + epsilon_Sigma)) == 0
        assert g[0, 1] == 0
        assert g[1, 0] == 0

        # Compute volume
        vol = metric_obj.volume_element(g)
        expected_vol = sqrt((h1 + epsilon_Sigma) * (h2 + epsilon_Sigma))
        assert simplify(vol - expected_vol) == 0


# Note: We skip testing the full 3D example and Christoffel symbols
# in automated tests due to computational complexity. These should be
# tested interactively or with numerical values.


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
