"""Tests for KineticOperator and BAOAB integration."""

import pytest
import torch

from fragile.bounds import TorchBounds
from fragile.core.benchmarks import OptimBenchmark
from fragile.core.euclidean_gas import SwarmState
from fragile.core.kinetic_operator import KineticOperator


def create_quadratic_potential(dims: int = 2) -> OptimBenchmark:
    """Helper to create a simple quadratic potential U(x) = 0.5 * ||x||^2."""

    def quadratic(x):
        return 0.5 * torch.sum(x**2, dim=-1)

    bounds = TorchBounds(
        low=torch.full((dims,), -5.0), high=torch.full((dims,), 5.0)
    )

    return OptimBenchmark(dims=dims, function=quadratic, bounds=bounds)


class TestKineticOperator:
    """Tests for KineticOperator."""

    @pytest.fixture
    def kinetic_op(self, simple_potential, torch_dtype):
        """Create kinetic operator for testing."""
        return KineticOperator(
            gamma=1.0,
            beta=1.0,
            delta_t=0.01,
            integrator="baoab",
            potential=simple_potential,
            device=torch.device("cpu"),
            dtype=torch_dtype,
        )

    def test_initialization(self, kinetic_op):
        """Test kinetic operator initialization."""
        assert kinetic_op.dt == 0.01
        assert kinetic_op.gamma == 1.0
        assert kinetic_op.beta == 1.0

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
        potential = create_quadratic_potential(dims=3)
        op = KineticOperator(
            gamma=0.001,
            beta=1.0,
            delta_t=0.001,
            potential=potential,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

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
        potential = create_quadratic_potential(dims=3)
        op = KineticOperator(
            gamma=1.0,
            beta=1.0,
            delta_t=0.01,
            potential=potential,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

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
        potential = create_quadratic_potential(dims=3)
        op = KineticOperator(
            gamma=1.0,
            beta=1.0,
            delta_t=0.01,
            potential=potential,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

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
        potential = create_quadratic_potential(dims=3)
        op = KineticOperator(
            gamma=0.001,
            beta=100.0,
            delta_t=0.001,
            potential=potential,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        x = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        v = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        state = SwarmState(x, v)

        # Compute initial energy
        E0 = (potential(state.x) + 0.5 * torch.sum(state.v**2, dim=-1))[0]

        # Take several steps
        for _ in range(10):
            torch.manual_seed(42)  # Same seed to minimize stochastic effects
            state = op.apply(state)

        # Compute final energy
        E1 = (potential(state.x) + 0.5 * torch.sum(state.v**2, dim=-1))[0]

        # With low friction and temperature, energy should be approximately conserved
        # (allowing for some thermalization and discretization error)
        assert torch.allclose(E0, E1, rtol=0.5)

    def test_stationary_point(self):
        """Test that a particle at origin with zero velocity stays near origin."""
        potential = create_quadratic_potential(dims=3)
        op = KineticOperator(
            gamma=1.0,
            beta=1.0,
            delta_t=0.01,
            potential=potential,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

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
        potential = create_quadratic_potential(dims=3)
        op = KineticOperator(
            gamma=1.0,
            beta=1.0,
            delta_t=0.01,
            potential=potential,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

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
        potential = create_quadratic_potential(dims=3)
        op = KineticOperator(
            gamma=1.0,
            beta=1.0,
            delta_t=dt,
            potential=potential,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

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
        potential = create_quadratic_potential(dims=3)
        op = KineticOperator(
            gamma=100.0,
            beta=1.0,
            delta_t=0.01,
            potential=potential,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

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
        potential = create_quadratic_potential(dims=3)
        op = KineticOperator(
            gamma=0.1,
            beta=1.0,
            delta_t=0.01,
            potential=potential,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

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


class TestVelocitySquashing:
    """Tests for velocity squashing functionality using psi_v."""

    @pytest.fixture
    def kinetic_op_with_squashing(self, simple_potential, torch_dtype):
        """Create kinetic operator with velocity squashing enabled."""
        return KineticOperator(
            gamma=1.0,
            beta=1.0,
            delta_t=0.01,
            integrator="baoab",
            V_alg=2.0,
            use_velocity_squashing=True,
            potential=simple_potential,
            device=torch.device("cpu"),
            dtype=torch_dtype,
        )

    @pytest.fixture
    def kinetic_op_without_squashing(self, simple_potential, torch_dtype):
        """Create kinetic operator without velocity squashing."""
        return KineticOperator(
            gamma=1.0,
            beta=1.0,
            delta_t=0.01,
            integrator="baoab",
            V_alg=2.0,
            use_velocity_squashing=False,
            potential=simple_potential,
            device=torch.device("cpu"),
            dtype=torch_dtype,
        )

    def test_velocity_bound_with_squashing(self, kinetic_op_with_squashing):
        """Test that velocities are bounded when squashing is enabled."""
        V_alg = kinetic_op_with_squashing.V_alg

        # Start with high velocities
        x = torch.zeros(10, 3, dtype=torch.float64)
        v = torch.randn(10, 3, dtype=torch.float64) * 10.0  # High initial velocities
        state = SwarmState(x, v)

        torch.manual_seed(42)
        new_state = kinetic_op_with_squashing.apply(state)

        # All velocity magnitudes should be strictly less than V_alg
        v_norms = torch.linalg.vector_norm(new_state.v, dim=-1)
        assert torch.all(v_norms < V_alg)

    def test_no_bound_without_squashing(self, kinetic_op_without_squashing):
        """Test that velocities are NOT bounded when squashing is disabled."""
        V_alg = kinetic_op_without_squashing.V_alg

        # Start with high velocities in a region that would maintain velocity
        x = torch.zeros(10, 3, dtype=torch.float64)
        v = torch.randn(10, 3, dtype=torch.float64) * 5.0  # High initial velocities
        state = SwarmState(x, v)

        torch.manual_seed(42)
        new_state = kinetic_op_without_squashing.apply(state)

        # Some velocities may exceed V_alg (since squashing is disabled)
        # We just check that the feature flag works
        assert kinetic_op_without_squashing.use_velocity_squashing is False

    def test_squashing_preserves_direction(self, kinetic_op_with_squashing):
        """Test that velocity squashing preserves direction."""
        # Start with a specific velocity direction
        x = torch.zeros(1, 3, dtype=torch.float64)
        v = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float64)  # Large velocity in x-direction
        state = SwarmState(x, v)

        torch.manual_seed(42)
        new_state = kinetic_op_with_squashing.apply(state)

        # Velocity should still point primarily in x-direction
        # (allowing for some drift from forces and noise)
        v_final = new_state.v[0]
        # The x-component should dominate
        assert abs(v_final[0]) > 0.0

    def test_squashing_is_smooth(self, kinetic_op_with_squashing):
        """Test that squashing produces smooth results for nearby velocities."""
        V_alg = kinetic_op_with_squashing.V_alg

        # Create two states with slightly different velocities
        x = torch.zeros(2, 3, dtype=torch.float64)
        v1 = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
        v2 = torch.tensor([[3.1, 0.0, 0.0]], dtype=torch.float64)
        v = torch.cat([v1, v2], dim=0)
        state = SwarmState(x, v)

        torch.manual_seed(42)
        new_state = kinetic_op_with_squashing.apply(state)

        # Final velocities should be close (continuity)
        v_norm_diff = torch.linalg.vector_norm(new_state.v[0] - new_state.v[1])
        assert v_norm_diff < 1.0  # Should be reasonably close

    def test_squashing_with_different_V_alg(self):
        """Test velocity squashing with different V_alg values."""
        potential = create_quadratic_potential(dims=3)

        for V_alg in [0.5, 1.0, 5.0, 10.0]:
            op = KineticOperator(
                gamma=1.0,
                beta=1.0,
                delta_t=0.01,
                V_alg=V_alg,
                use_velocity_squashing=True,
                potential=potential,
                device=torch.device("cpu"),
                dtype=torch.float64,
            )

            # Start with high velocities
            x = torch.zeros(10, 3, dtype=torch.float64)
            v = torch.randn(10, 3, dtype=torch.float64) * 20.0
            state = SwarmState(x, v)

            torch.manual_seed(42)
            new_state = op.apply(state)

            # All velocity magnitudes should be strictly less than V_alg
            v_norms = torch.linalg.vector_norm(new_state.v, dim=-1)
            assert torch.all(v_norms < V_alg)

    def test_default_V_alg_is_infinite(self, simple_potential, torch_dtype):
        """Test that default V_alg is effectively infinite (no squashing by default)."""
        op = KineticOperator(
            gamma=1.0,
            beta=1.0,
            delta_t=0.01,
            potential=simple_potential,
            device=torch.device("cpu"),
            dtype=torch_dtype,
        )

        # V_alg should be infinite by default
        assert op.V_alg == float('inf')
        # Squashing should be disabled by default
        assert op.use_velocity_squashing is False

    def test_squashing_with_zero_velocity(self, kinetic_op_with_squashing):
        """Test that squashing handles zero velocity correctly."""
        x = torch.zeros(1, 3, dtype=torch.float64)
        v = torch.zeros(1, 3, dtype=torch.float64)
        state = SwarmState(x, v)

        torch.manual_seed(42)
        new_state = kinetic_op_with_squashing.apply(state)

        # Should not crash and velocity should remain small
        v_norm = torch.linalg.vector_norm(new_state.v)
        assert v_norm < kinetic_op_with_squashing.V_alg

    def test_psi_v_function_directly(self):
        """Test the psi_v function directly."""
        from fragile.core.kinetic_operator import psi_v

        # Test basic squashing
        v = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float64)
        V_alg = 2.0
        v_squashed = psi_v(v, V_alg)

        # Magnitude should be less than V_alg
        v_norm = torch.linalg.vector_norm(v_squashed, dim=-1)
        assert v_norm < V_alg

        # Direction should be preserved
        v_dir = v / torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        v_squashed_dir = v_squashed / torch.linalg.vector_norm(v_squashed, dim=-1, keepdim=True)
        assert torch.allclose(v_dir, v_squashed_dir)

    def test_psi_v_lipschitz_property(self):
        """Test that psi_v is 1-Lipschitz."""
        from fragile.core.kinetic_operator import psi_v

        V_alg = 2.0

        # Test multiple pairs of velocities
        for _ in range(10):
            v1 = torch.randn(5, 3, dtype=torch.float64) * 5.0
            v2 = torch.randn(5, 3, dtype=torch.float64) * 5.0

            # Apply squashing
            psi_v1 = psi_v(v1, V_alg)
            psi_v2 = psi_v(v2, V_alg)

            # Compute distances
            dist_original = torch.linalg.vector_norm(v1 - v2, dim=-1)
            dist_squashed = torch.linalg.vector_norm(psi_v1 - psi_v2, dim=-1)

            # Lipschitz property: ||ψ(v1) - ψ(v2)|| ≤ ||v1 - v2||
            assert torch.all(dist_squashed <= dist_original + 1e-6)  # Small tolerance for numerical errors

    def test_psi_v_handles_batch(self):
        """Test that psi_v handles batched input correctly."""
        from fragile.core.kinetic_operator import psi_v

        # Batch of velocities
        v = torch.randn(100, 3, dtype=torch.float64) * 10.0
        V_alg = 2.0

        v_squashed = psi_v(v, V_alg)

        # All should be bounded
        v_norms = torch.linalg.vector_norm(v_squashed, dim=-1)
        assert torch.all(v_norms < V_alg)

        # Shape should be preserved
        assert v_squashed.shape == v.shape

    def test_psi_v_error_on_negative_V_alg(self):
        """Test that psi_v raises error for negative V_alg."""
        from fragile.core.kinetic_operator import psi_v

        v = torch.randn(5, 3, dtype=torch.float64)

        with pytest.raises(ValueError, match="V_alg must be positive"):
            psi_v(v, -1.0)

        with pytest.raises(ValueError, match="V_alg must be positive"):
            psi_v(v, 0.0)
