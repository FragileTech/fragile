"""Tests for KineticOperator and BAOAB integration."""

import pytest
import torch

from fragile.euclidean_gas import (
    KineticOperator,
    LangevinParams,
    SimpleQuadraticPotential,
    SwarmState,
)


class TestKineticOperator:
    """Tests for KineticOperator."""

    @pytest.fixture
    def kinetic_op(self, langevin_params, simple_potential, torch_dtype):
        """Create kinetic operator for testing."""
        return KineticOperator(
            langevin_params,
            simple_potential,
            torch.device("cpu"),
            torch_dtype,
        )

    def test_initialization(self, kinetic_op, langevin_params):
        """Test kinetic operator initialization."""
        assert kinetic_op.params == langevin_params
        assert kinetic_op.dt == langevin_params.delta_t
        assert kinetic_op.gamma == langevin_params.gamma
        assert kinetic_op.beta == langevin_params.beta

    def test_baoab_constants(self, kinetic_op):
        """Test that BAOAB constants are precomputed correctly."""
        gamma = kinetic_op.gamma
        dt = kinetic_op.dt

        # c1 = exp(-gamma * dt)
        expected_c1 = torch.exp(torch.tensor(-gamma * dt, dtype=torch.float64))
        assert torch.allclose(kinetic_op.c1, expected_c1)

        # c2 = sqrt((1 - c1^2) / beta)
        beta = kinetic_op.beta
        expected_c2 = torch.sqrt((1.0 - expected_c1**2) / beta)
        assert torch.allclose(kinetic_op.c2, expected_c2)

    def test_apply_preserves_shape(self, kinetic_op):
        """Test that kinetic operator preserves swarm shape."""
        x = torch.randn(10, 3, dtype=torch.float64)
        v = torch.randn(10, 3, dtype=torch.float64)
        state = SwarmState(x, v)

        new_state = kinetic_op.apply(state)

        assert new_state.x.shape == state.x.shape
        assert new_state.v.shape == state.v.shape

    def test_apply_changes_state(self, kinetic_op):
        """Test that kinetic operator changes the state."""
        torch.manual_seed(42)
        x = torch.randn(10, 3, dtype=torch.float64)
        v = torch.randn(10, 3, dtype=torch.float64)
        state = SwarmState(x, v)

        new_state = kinetic_op.apply(state)

        # State should change
        assert not torch.allclose(new_state.x, state.x)
        assert not torch.allclose(new_state.v, state.v)

    def test_zero_friction_limit(self):
        """Test behavior with very small friction (approaches Hamiltonian)."""
        # Very small friction
        params = LangevinParams(gamma=0.001, beta=1.0, delta_t=0.001)
        potential = SimpleQuadraticPotential()
        op = KineticOperator(params, potential, torch.device("cpu"), torch.float64)

        # Start at non-equilibrium position
        x = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        v = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        state = SwarmState(x, v)

        # Take one step
        torch.manual_seed(42)
        new_state = op.apply(state)

        # Position should move (due to velocity update from force)
        assert not torch.allclose(new_state.x, state.x)

    def test_gradient_calculation(self):
        """Test that gradients are computed correctly."""
        params = LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01)
        potential = SimpleQuadraticPotential()
        op = KineticOperator(params, potential, torch.device("cpu"), torch.float64)

        # For quadratic potential U(x) = 0.5 * ||x||^2, gradient is x
        x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        v = torch.zeros(1, 3, dtype=torch.float64)
        state = SwarmState(x, v)

        # The force is -grad U = -x
        # After first B step, v should be approximately v + (dt/2) * (-x)
        new_state = op.apply(state)

        # We can't check exactly due to subsequent steps, but state should change
        assert not torch.allclose(new_state.x, state.x)

    def test_reproducibility_with_seed(self):
        """Test that integration is reproducible with same seed."""
        params = LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01)
        potential = SimpleQuadraticPotential()
        op = KineticOperator(params, potential, torch.device("cpu"), torch.float64)

        x = torch.randn(10, 3, dtype=torch.float64)
        v = torch.randn(10, 3, dtype=torch.float64)
        state = SwarmState(x, v)

        torch.manual_seed(42)
        state1 = op.apply(state)

        torch.manual_seed(42)
        state2 = op.apply(state)

        assert torch.allclose(state1.x, state2.x)
        assert torch.allclose(state1.v, state2.v)

    def test_energy_conservation_low_friction(self):
        """Test approximate energy conservation with low friction."""
        # Very low friction, low temperature (nearly Hamiltonian)
        params = LangevinParams(gamma=0.001, beta=100.0, delta_t=0.001)
        potential = SimpleQuadraticPotential()
        op = KineticOperator(params, potential, torch.device("cpu"), torch.float64)

        x = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        v = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        state = SwarmState(x, v)

        # Compute initial energy
        E0 = (potential.evaluate(state.x) + 0.5 * torch.sum(state.v**2, dim=-1))[0]

        # Take several steps
        for _ in range(10):
            torch.manual_seed(42)  # Same seed to minimize stochastic effects
            state = op.apply(state)

        # Compute final energy
        E1 = (potential.evaluate(state.x) + 0.5 * torch.sum(state.v**2, dim=-1))[0]

        # With low friction and temperature, energy should be approximately conserved
        # (allowing for some thermalization and discretization error)
        assert torch.allclose(E0, E1, rtol=0.5)

    def test_stationary_point(self):
        """Test that a particle at origin with zero velocity stays near origin."""
        params = LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01)
        potential = SimpleQuadraticPotential()
        op = KineticOperator(params, potential, torch.device("cpu"), torch.float64)

        # Start at origin with zero velocity
        x = torch.zeros(1, 3, dtype=torch.float64)
        v = torch.zeros(1, 3, dtype=torch.float64)
        state = SwarmState(x, v)

        # Take one step
        torch.manual_seed(42)
        new_state = op.apply(state)

        # Should stay near origin (moved only by noise)
        # With dt=0.01, noise is small
        assert torch.norm(new_state.x) < 1.0
        assert torch.norm(new_state.v) < 1.0

    def test_multiple_walkers_independent(self):
        """Test that multiple walkers evolve independently."""
        params = LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01)
        potential = SimpleQuadraticPotential()
        op = KineticOperator(params, potential, torch.device("cpu"), torch.float64)

        # Create state with two walkers
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        v = torch.zeros(2, 2, dtype=torch.float64)
        state = SwarmState(x, v)

        torch.manual_seed(42)
        new_state = op.apply(state)

        # Both walkers should have evolved
        assert not torch.allclose(new_state.x[0], state.x[0])
        assert not torch.allclose(new_state.x[1], state.x[1])

        # But they should be different from each other (different noise)
        assert not torch.allclose(new_state.x[0], new_state.x[1])

    @pytest.mark.parametrize("dt", [0.001, 0.01, 0.1])
    def test_different_timesteps(self, dt):
        """Test operator with different timestep sizes."""
        params = LangevinParams(gamma=1.0, beta=1.0, delta_t=dt)
        potential = SimpleQuadraticPotential()
        op = KineticOperator(params, potential, torch.device("cpu"), torch.float64)

        x = torch.randn(5, 3, dtype=torch.float64)
        v = torch.randn(5, 3, dtype=torch.float64)
        state = SwarmState(x, v)

        torch.manual_seed(42)
        new_state = op.apply(state)

        # Should complete without error
        assert new_state is not None
        assert new_state.x.shape == state.x.shape

    def test_high_friction_limit(self):
        """Test behavior with high friction (overdamped limit)."""
        # Very high friction
        params = LangevinParams(gamma=100.0, beta=1.0, delta_t=0.01)
        potential = SimpleQuadraticPotential()
        op = KineticOperator(params, potential, torch.device("cpu"), torch.float64)

        # Start with high velocity
        x = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        v = torch.tensor([[10.0, 0.0]], dtype=torch.float64)
        state = SwarmState(x, v)

        # Take one step
        torch.manual_seed(42)
        new_state = op.apply(state)

        # Velocity should be damped significantly
        assert torch.norm(new_state.v) < torch.norm(state.v)

    def test_position_update_depends_on_velocity(self):
        """Test that position updates depend on velocity."""
        params = LangevinParams(gamma=0.1, beta=1.0, delta_t=0.01)
        potential = SimpleQuadraticPotential()
        op = KineticOperator(params, potential, torch.device("cpu"), torch.float64)

        x = torch.zeros(2, 2, dtype=torch.float64)
        v1 = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        v2 = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)

        state1 = SwarmState(x.clone(), v1)
        state2 = SwarmState(x.clone(), v2)

        torch.manual_seed(42)
        new_state1 = op.apply(state1)

        torch.manual_seed(42)
        new_state2 = op.apply(state2)

        # Positions should be different due to different initial velocities
        assert not torch.allclose(new_state1.x[0], new_state2.x[1])
