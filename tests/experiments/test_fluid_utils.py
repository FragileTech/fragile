"""Test suite for fluid dynamics utilities.

Tests all components of fluid_utils.py:
- SPH kernel interpolation
- Periodic boundary conditions
- Conservation law validation
- Incompressibility checks
- Taylor-Green vortex validation
- Flow analysis tools
"""

import numpy as np
import pytest
import torch

from fragile.experiments.fluid_utils import (
    ConservationValidator,
    FlowAnalyzer,
    FluidFieldComputer,
    TaylorGreenValidator,
    ValidationMetrics,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def taylor_green_particles():
    """Generate particles with Taylor-Green vortex velocity field."""
    N = 500
    bounds = (-np.pi, np.pi)

    # Random positions
    x = np.random.uniform(bounds[0], bounds[1], N)
    y = np.random.uniform(bounds[0], bounds[1], N)
    positions = torch.tensor(np.column_stack([x, y]), dtype=torch.float32)

    # Taylor-Green velocity field at t=0
    U0 = 1.0
    k = 1.0
    u = -U0 * np.cos(k * x) * np.sin(k * y)
    v = U0 * np.sin(k * x) * np.cos(k * y)
    velocities = torch.tensor(np.column_stack([u, v]), dtype=torch.float32)

    return positions, velocities


@pytest.fixture
def uniform_particles():
    """Generate uniformly distributed particles with zero velocity."""
    N = 300
    bounds = (-np.pi, np.pi)

    # Uniform grid with small noise
    n_side = int(np.sqrt(N))
    x = np.linspace(bounds[0], bounds[1], n_side)
    y = np.linspace(bounds[0], bounds[1], n_side)
    X, Y = np.meshgrid(x, y)
    positions = torch.tensor(np.column_stack([X.ravel()[:N], Y.ravel()[:N]]), dtype=torch.float32)

    velocities = torch.zeros((N, 2), dtype=torch.float32)

    return positions, velocities


# ============================================================================
# Test FluidFieldComputer
# ============================================================================


class TestFluidFieldComputer:
    """Test fluid field computation methods."""

    def test_sph_kernel_normalization(self):
        """Test that SPH kernel is properly normalized."""
        h = 0.3

        # Create radial grid
        r = np.linspace(0, 2 * h, 100)
        kernel = FluidFieldComputer.sph_kernel(r, h)

        # Check compact support
        assert np.all(kernel[r >= 2 * h] == 0), "Kernel should be zero beyond 2h"

        # Check positivity within support
        assert np.all(kernel[r < 2 * h] >= 0), "Kernel should be non-negative"

        # Check normalization (integrate in 2D)
        # ∫W(r,h) 2πr dr = 1
        r_grid = np.linspace(0, 2 * h, 1000)
        kernel_vals = FluidFieldComputer.sph_kernel(r_grid, h)
        integral = np.trapz(kernel_vals * 2 * np.pi * r_grid, r_grid)

        assert np.abs(integral - 1.0) < 0.01, f"Kernel should be normalized, got {integral}"

    def test_periodic_distance(self):
        """Test periodic boundary condition handling."""
        domain_size = 2 * np.pi

        # Test case: distances that need wrapping
        # dx=4.0 > π should wrap to -(2π - 4.0)
        dx = np.array([4.0, -4.0, 0.5])
        dy = np.array([4.0, -4.0, 0.5])

        dx_periodic, dy_periodic = FluidFieldComputer._apply_periodic_distance(dx, dy, domain_size)

        # Check minimum image convention
        assert np.all(np.abs(dx_periodic) <= domain_size / 2)
        assert np.all(np.abs(dy_periodic) <= domain_size / 2)

        # Specific test: dx=4.0 > π should become -(2π - 4.0)
        expected = 4.0 - domain_size
        assert np.abs(dx_periodic[0] - expected) < 1e-6

        # dx=-4.0 < -π should become 2π - 4.0
        expected_neg = -4.0 + domain_size
        assert np.abs(dx_periodic[1] - expected_neg) < 1e-6

    def test_velocity_field_computation(self, taylor_green_particles):
        """Test velocity field computation with SPH interpolation."""
        positions, velocities = taylor_green_particles

        X, Y, U, V = FluidFieldComputer.compute_velocity_field(
            positions, velocities, grid_resolution=30
        )

        # Check output shapes
        assert X.shape == (30, 30)
        assert Y.shape == (30, 30)
        assert U.shape == (30, 30)
        assert V.shape == (30, 30)

        # Check that velocity field is reasonable (not all zeros, not all nans)
        assert not np.all(U == 0) and not np.all(V == 0)
        assert not np.any(np.isnan(U)) and not np.any(np.isnan(V))

        # Check that velocity magnitude is bounded
        vel_mag = np.sqrt(U**2 + V**2)
        assert np.max(vel_mag) < 3.0, "Velocity magnitude should be bounded"

    def test_vorticity_computation(self, taylor_green_particles):
        """Test vorticity computation."""
        positions, velocities = taylor_green_particles

        _X, _Y, U, V = FluidFieldComputer.compute_velocity_field(
            positions, velocities, grid_resolution=30
        )

        dx = 2 * np.pi / 30
        dy = 2 * np.pi / 30

        omega = FluidFieldComputer.compute_vorticity(U, V, dx, dy, periodic=True)

        # Check output shape
        assert omega.shape == (30, 30)

        # Check no nans
        assert not np.any(np.isnan(omega))

        # For Taylor-Green vortex, vorticity should be non-zero
        assert np.abs(np.mean(omega**2)) > 1e-6

    def test_divergence_computation(self, taylor_green_particles):
        """Test divergence computation (should be ~0 for incompressible flow)."""
        positions, velocities = taylor_green_particles

        _X, _Y, U, V = FluidFieldComputer.compute_velocity_field(
            positions, velocities, grid_resolution=30
        )

        dx = 2 * np.pi / 30
        dy = 2 * np.pi / 30

        div = FluidFieldComputer.compute_divergence(U, V, dx, dy, periodic=True)

        # Check output shape
        assert div.shape == (30, 30)

        # Check no nans
        assert not np.any(np.isnan(div))

        # Taylor-Green is incompressible, divergence should be small
        rms_div = np.sqrt(np.mean(div**2))
        assert rms_div < 0.5, f"Divergence too large: {rms_div}"

    def test_stream_function_integration(self, taylor_green_particles):
        """Test stream function computation via integration."""
        positions, velocities = taylor_green_particles

        _X, _Y, U, V = FluidFieldComputer.compute_velocity_field(
            positions, velocities, grid_resolution=30
        )

        dx = 2 * np.pi / 30
        dy = 2 * np.pi / 30

        psi = FluidFieldComputer.compute_stream_function(U, V, dx, dy, method="integration")

        # Check output shape
        assert psi.shape == (30, 30)

        # Check no nans
        assert not np.any(np.isnan(psi))

        # Check that stream function has zero mean (arbitrary constant removed)
        assert np.abs(np.mean(psi)) < 1e-6

    def test_stream_function_poisson(self, taylor_green_particles):
        """Test stream function computation via Poisson solver."""
        positions, velocities = taylor_green_particles

        _X, _Y, U, V = FluidFieldComputer.compute_velocity_field(
            positions, velocities, grid_resolution=30
        )

        dx = 2 * np.pi / 30
        dy = 2 * np.pi / 30

        psi = FluidFieldComputer.compute_stream_function(U, V, dx, dy, method="poisson")

        # Check output shape
        assert psi.shape == (30, 30)

        # Check no nans
        assert not np.any(np.isnan(psi))

    def test_density_field_normalization(self, uniform_particles):
        """Test that density field is properly normalized."""
        positions, _velocities = uniform_particles
        N = len(positions)

        X, Y, density = FluidFieldComputer.compute_density_field(
            positions, grid_resolution=30, normalize=True
        )

        # Check output shapes
        assert X.shape == (30, 30)
        assert Y.shape == (30, 30)
        assert density.shape == (30, 30)

        # Check normalization: ∫ρ dA should equal N
        dx = 2 * np.pi / 30
        dy = 2 * np.pi / 30
        integral = np.sum(density) * dx * dy

        assert np.abs(integral - N) / N < 0.1, f"Density not normalized: ∫ρ = {integral}, N = {N}"


# ============================================================================
# Test ConservationValidator
# ============================================================================


class TestConservationValidator:
    """Test conservation law validators."""

    def test_mass_conservation_perfect(self, uniform_particles):
        """Test mass conservation with constant particle count."""
        positions, _velocities = uniform_particles
        len(positions)

        # Create history with constant N
        positions_history = [positions for _ in range(10)]

        metric = ConservationValidator.check_mass_conservation(positions_history, tolerance=0.01)

        assert metric.passed, "Should pass with constant particle count"
        assert metric.measured_value < 1e-10, "Variation should be zero"

    def test_mass_conservation_violation(self, uniform_particles):
        """Test mass conservation with varying particle count."""
        positions, _velocities = uniform_particles

        # Create history with varying N
        positions_history = [
            positions[: int(len(positions) * (0.9 + 0.1 * i / 10))] for i in range(10)
        ]

        metric = ConservationValidator.check_mass_conservation(positions_history, tolerance=0.05)

        assert not metric.passed, "Should fail with varying particle count"
        assert metric.measured_value > 0.05

    def test_momentum_conservation_zero_velocity(self, uniform_particles):
        """Test momentum conservation with zero velocities."""
        positions, velocities = uniform_particles

        # Create history
        positions_history = [positions for _ in range(10)]
        velocities_history = [velocities for _ in range(10)]

        metric = ConservationValidator.check_momentum_conservation(
            positions_history, velocities_history, tolerance=0.05
        )

        assert metric.passed, "Should pass with zero momentum"

    def test_energy_budget_constant_energy(self, uniform_particles):
        """Test energy budget with constant kinetic energy."""
        _positions, velocities = uniform_particles

        # Set constant non-zero velocity
        velocities[:] = 1.0
        velocities_history = [velocities.clone() for _ in range(20)]

        metric = ConservationValidator.check_energy_budget(
            velocities_history, dt=0.01, tolerance=0.05
        )

        assert metric.passed, "Should pass with constant energy"

    def test_incompressibility_taylor_green(self, taylor_green_particles):
        """Test incompressibility check on Taylor-Green vortex."""
        positions, velocities = taylor_green_particles

        metric = ConservationValidator.check_incompressibility(
            positions, velocities, grid_resolution=30, tolerance=0.2
        )

        # Taylor-Green is exactly incompressible
        assert metric.measured_value < 0.5, "Taylor-Green should be nearly incompressible"


# ============================================================================
# Test FlowAnalyzer
# ============================================================================


class TestFlowAnalyzer:
    """Test flow analysis tools."""

    def test_reynolds_number_computation(self, taylor_green_particles):
        """Test Reynolds number computation."""
        positions, velocities = taylor_green_particles

        viscosity = 1.0
        Re = FlowAnalyzer.compute_reynolds_number(positions, velocities, viscosity)

        # Check reasonable value
        assert Re > 0 and Re < 1000, f"Reynolds number unreasonable: {Re}"

    def test_enstrophy_computation(self, taylor_green_particles):
        """Test enstrophy computation."""
        positions, velocities = taylor_green_particles

        Z = FlowAnalyzer.compute_enstrophy(positions, velocities, grid_resolution=30)

        # Check positive value
        assert Z > 0, "Enstrophy should be positive"
        assert not np.isnan(Z), "Enstrophy should not be NaN"

    def test_enstrophy_zero_for_uniform_flow(self, uniform_particles):
        """Test that enstrophy is zero for uniform flow."""
        positions, velocities = uniform_particles

        # Set uniform velocity
        velocities[:] = torch.tensor([1.0, 0.0])

        Z = FlowAnalyzer.compute_enstrophy(positions, velocities, grid_resolution=30)

        # Should be very small (numerical errors only)
        assert Z < 0.1, f"Enstrophy should be near zero for uniform flow, got {Z}"

    def test_vorticity_statistics(self, taylor_green_particles):
        """Test vorticity statistics computation."""
        positions, velocities = taylor_green_particles

        stats = FlowAnalyzer.compute_vorticity_statistics(
            positions, velocities, grid_resolution=30
        )

        # Check all fields present
        assert "mean" in stats
        assert "std" in stats
        assert "max" in stats
        assert "min" in stats
        assert "enstrophy" in stats

        # Check reasonable values
        assert stats["std"] > 0, "Vorticity should have variation"
        assert stats["enstrophy"] > 0, "Enstrophy should be positive"
        assert not np.isnan(stats["mean"]), "Statistics should not be NaN"


# ============================================================================
# Test TaylorGreenValidator
# ============================================================================


class TestTaylorGreenValidator:
    """Test Taylor-Green vortex analytical validation."""

    def test_analytical_velocity_at_t0(self):
        """Test analytical velocity field matches initial condition."""
        validator = TaylorGreenValidator(U0=1.0, viscosity=1.0)

        # Grid
        x = np.linspace(-np.pi, np.pi, 30)
        y = np.linspace(-np.pi, np.pi, 30)
        X, Y = np.meshgrid(x, y)

        # At t=0, no decay
        u, v = validator.analytical_velocity(X, Y, t=0.0)

        # Check expected form
        k = validator.k
        u_expected = -validator.U0 * np.cos(k * X) * np.sin(k * Y)
        v_expected = validator.U0 * np.sin(k * X) * np.cos(k * Y)

        np.testing.assert_allclose(u, u_expected, rtol=1e-10)
        np.testing.assert_allclose(v, v_expected, rtol=1e-10)

    def test_analytical_velocity_decay(self):
        """Test that velocity decays exponentially in time."""
        validator = TaylorGreenValidator(U0=1.0, viscosity=1.0)

        # Grid
        X = np.array([[0.0]])
        Y = np.array([[0.0]])

        # Check decay at different times
        t1 = 0.0
        t2 = 1.0

        u1, v1 = validator.analytical_velocity(X, Y, t1)
        u2, v2 = validator.analytical_velocity(X, Y, t2)

        # Magnitude should decay
        mag1 = np.sqrt(u1**2 + v1**2)
        mag2 = np.sqrt(u2**2 + v2**2)

        # Expected decay factor
        k = validator.k
        nu = validator.nu
        expected_ratio = np.exp(-2 * nu * k**2 * (t2 - t1))

        if mag1 > 1e-6:
            actual_ratio = mag2 / mag1
            assert np.abs(actual_ratio - expected_ratio) < 1e-6

    def test_validate_velocity_field_perfect(self):
        """Test validation with perfect Taylor-Green data."""
        validator = TaylorGreenValidator(U0=1.0, viscosity=1.0)

        # Generate perfect Taylor-Green particles
        N = 500
        x = np.random.uniform(-np.pi, np.pi, N)
        y = np.random.uniform(-np.pi, np.pi, N)
        positions = torch.tensor(np.column_stack([x, y]), dtype=torch.float32)

        k = validator.k
        t = 0.5
        decay = np.exp(-2 * validator.nu * k**2 * t)
        u = -validator.U0 * np.cos(k * x) * np.sin(k * y) * decay
        v = validator.U0 * np.sin(k * x) * np.cos(k * y) * decay
        velocities = torch.tensor(np.column_stack([u, v]), dtype=torch.float32)

        # Validate
        metric = validator.validate_velocity_field(
            positions, velocities, t, grid_resolution=30, tolerance=0.3
        )

        # Should pass with small error (due to interpolation)
        assert metric.measured_value < 0.3, f"Error too large: {metric.measured_value}"

    def test_validate_vorticity_field(self, taylor_green_particles):
        """Test vorticity field validation."""
        validator = TaylorGreenValidator(U0=1.0, viscosity=1.0)
        positions, velocities = taylor_green_particles

        metric = validator.validate_vorticity_field(
            positions, velocities, t=0.0, grid_resolution=30, tolerance=0.3
        )

        # Check metric is reasonable
        assert metric.measured_value < 0.5, "Vorticity error too large"
        assert not np.isnan(metric.measured_value)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_taylor_green_workflow(self):
        """Test complete workflow: generate data, compute fields, validate."""
        # Setup
        validator = TaylorGreenValidator(U0=1.0, viscosity=0.5)

        # Generate particles at t=0
        N = 800
        x = np.random.uniform(-np.pi, np.pi, N)
        y = np.random.uniform(-np.pi, np.pi, N)
        positions = torch.tensor(np.column_stack([x, y]), dtype=torch.float32)

        k = 1.0
        u = -np.cos(k * x) * np.sin(k * y)
        v = np.sin(k * x) * np.cos(k * y)
        velocities = torch.tensor(np.column_stack([u, v]), dtype=torch.float32)

        # Compute fields
        _X, _Y, U, V = FluidFieldComputer.compute_velocity_field(
            positions, velocities, grid_resolution=40
        )

        dx = 2 * np.pi / 40
        dy = 2 * np.pi / 40

        omega = FluidFieldComputer.compute_vorticity(U, V, dx, dy)
        div = FluidFieldComputer.compute_divergence(U, V, dx, dy)
        FluidFieldComputer.compute_stream_function(U, V, dx, dy)

        # Flow analysis
        Re = FlowAnalyzer.compute_reynolds_number(positions, velocities, 0.5)
        Z = FlowAnalyzer.compute_enstrophy(positions, velocities)
        stats = FlowAnalyzer.compute_vorticity_statistics(positions, velocities)

        # Conservation checks
        metric_incomp = ConservationValidator.check_incompressibility(positions, velocities)

        # Validation
        metric_vel = validator.validate_velocity_field(positions, velocities, t=0.0)
        metric_vort = validator.validate_vorticity_field(positions, velocities, t=0.0)

        # Assertions
        assert not np.any(np.isnan(omega)), "Vorticity has NaNs"
        assert np.sqrt(np.mean(div**2)) < 0.5, "Divergence too large"
        assert Re > 0, "Reynolds number invalid"
        assert Z > 0, "Enstrophy invalid"
        assert metric_vel.measured_value < 0.5, "Velocity validation failed"

        print("\n" + "=" * 60)
        print("Full Taylor-Green Workflow Results:")
        print("=" * 60)
        print(f"Reynolds number: {Re:.2f}")
        print(f"Enstrophy: {Z:.4f}")
        print(f"RMS divergence: {np.sqrt(np.mean(div**2)):.6f}")
        print(f"Vorticity stats: {stats}")
        print(f"Velocity validation: {metric_vel.description}")
        print(f"Vorticity validation: {metric_vort.description}")
        print(f"Incompressibility: {metric_incomp.description}")
        print("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
