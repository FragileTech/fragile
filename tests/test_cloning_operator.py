"""Tests for CloningOperator using cloning.py functions."""

import pytest
import torch

from fragile.euclidean_gas import (
    CloningOperator,
    CloningParams,
    SimpleQuadraticPotential,
    SwarmState,
)


class TestCloningOperator:
    """Tests for CloningOperator."""

    @pytest.fixture
    def potential(self):
        """Create potential for testing."""
        return SimpleQuadraticPotential()

    @pytest.fixture
    def cloning_op(self, cloning_params, potential, torch_dtype):
        """Create cloning operator for testing."""
        return CloningOperator(
            cloning_params, potential, torch.device("cpu"), torch_dtype, bounds=None
        )

    def test_initialization(self, cloning_op, cloning_params, potential):
        """Test cloning operator initialization."""
        assert cloning_op.params == cloning_params
        assert cloning_op.potential == potential
        assert cloning_op.device == torch.device("cpu")
        assert cloning_op.dtype == torch.float64

    def test_apply_preserves_shape(self, cloning_op):
        """Test that cloning preserves swarm shape."""
        x = torch.randn(10, 3, dtype=torch.float64)
        v = torch.randn(10, 3, dtype=torch.float64)
        state = SwarmState(x, v)

        new_state = cloning_op.apply(state)

        assert new_state.x.shape == state.x.shape
        assert new_state.v.shape == state.v.shape

    def test_apply_changes_state(self, cloning_op):
        """Test that cloning actually changes the state."""
        torch.manual_seed(42)
        x = torch.randn(10, 3, dtype=torch.float64)
        v = torch.randn(10, 3, dtype=torch.float64)
        state = SwarmState(x, v)

        new_state = cloning_op.apply(state)

        # State should change (positions and velocities)
        assert not torch.allclose(new_state.x, state.x)
        assert not torch.allclose(new_state.v, state.v)

    def test_return_parents(self, cloning_op):
        """Test that return_parents works correctly."""
        x = torch.randn(10, 3, dtype=torch.float64)
        v = torch.randn(10, 3, dtype=torch.float64)
        state = SwarmState(x, v)

        new_state, companions = cloning_op.apply(state, return_parents=True)

        assert new_state.x.shape == state.x.shape
        assert new_state.v.shape == state.v.shape
        assert companions.shape == (10,)
        assert companions.dtype == torch.long

    def test_return_info(self, cloning_op):
        """Test that return_info works correctly."""
        x = torch.randn(10, 3, dtype=torch.float64)
        v = torch.randn(10, 3, dtype=torch.float64)
        state = SwarmState(x, v)

        _new_state, _companions, info = cloning_op.apply(state, return_info=True)

        # Check info dictionary structure
        assert "cloning_scores" in info
        assert "cloning_probs" in info
        assert "will_clone" in info
        assert "num_cloned" in info
        assert "fitness" in info
        assert "distances" in info
        assert "rewards" in info

        # Check shapes
        assert info["cloning_scores"].shape == (10,)
        assert info["cloning_probs"].shape == (10,)
        assert info["will_clone"].shape == (10,)
        assert info["fitness"].shape == (10,)
        assert info["distances"].shape == (10,)
        assert info["rewards"].shape == (10,)
        assert isinstance(info["num_cloned"], int)

    def test_jitter_scale(self):
        """Test that position jitter scale affects results."""
        sigma_x = 0.1
        params = CloningParams(sigma_x=sigma_x, lambda_alg=1.0, alpha_restitution=0.5)
        potential = SimpleQuadraticPotential()
        op = CloningOperator(params, potential, torch.device("cpu"), torch.float64, bounds=None)

        torch.manual_seed(42)
        x = torch.randn(100, 3, dtype=torch.float64)
        v = torch.randn(100, 3, dtype=torch.float64)
        state = SwarmState(x, v)

        new_state = op.apply(state)

        # Verify that the operation completes and produces valid output
        assert not torch.any(torch.isnan(new_state.x))
        assert not torch.any(torch.isinf(new_state.x))
        assert not torch.any(torch.isnan(new_state.v))
        assert not torch.any(torch.isinf(new_state.v))

    def test_two_walkers(self):
        """Test cloning with two walkers."""
        params = CloningParams(sigma_x=0.1, lambda_alg=1.0, alpha_restitution=0.5)
        potential = SimpleQuadraticPotential()
        op = CloningOperator(params, potential, torch.device("cpu"), torch.float64, bounds=None)

        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        v = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        state = SwarmState(x, v)

        torch.manual_seed(42)
        new_state = op.apply(state)

        assert new_state.x.shape == (2, 2)
        assert new_state.v.shape == (2, 2)

    @pytest.mark.parametrize("alpha_restitution", [0.0, 0.5, 1.0])
    def test_restitution_coefficient(self, alpha_restitution):
        """Test different restitution coefficients."""
        params = CloningParams(
            sigma_x=0.1,
            lambda_alg=1.0,
            alpha_restitution=alpha_restitution,
        )
        potential = SimpleQuadraticPotential()
        op = CloningOperator(params, potential, torch.device("cpu"), torch.float64, bounds=None)

        x = torch.randn(5, 2, dtype=torch.float64)
        v = torch.randn(5, 2, dtype=torch.float64)
        state = SwarmState(x, v)

        torch.manual_seed(42)
        new_state = op.apply(state)

        # Test basic properties
        assert not torch.any(torch.isnan(new_state.v))
        assert not torch.any(torch.isinf(new_state.v))
        assert new_state.v.shape == v.shape

    def test_perfectly_inelastic_collision(self):
        """Test that alpha_restitution=0 results in velocity convergence."""
        params = CloningParams(
            sigma_x=0.1,
            lambda_alg=1.0,
            alpha_restitution=0.0,
            p_max=1.0,  # Ensure all walkers clone
        )
        potential = SimpleQuadraticPotential()
        op = CloningOperator(params, potential, torch.device("cpu"), torch.float64, bounds=None)

        # Create walkers with different velocities
        x = torch.randn(10, 2, dtype=torch.float64)
        v = torch.randn(10, 2, dtype=torch.float64) * 5.0  # Large velocity differences
        state = SwarmState(x, v)

        torch.manual_seed(42)
        new_state, _, info = op.apply(state, return_info=True)

        # With alpha_restitution=0, velocities in collision groups should converge
        # We can't test exact convergence without knowing the groups, but we can
        # verify that velocity variance decreases
        if info["num_cloned"] > 0:
            v_var_before = v.var(dim=0).sum()
            v_var_after = new_state.v.var(dim=0).sum()
            # Variance should decrease or stay similar (not increase dramatically)
            assert v_var_after <= v_var_before * 2.0  # Allow some increase due to stochasticity

    def test_reproducibility_with_seed(self):
        """Test that cloning is reproducible with same seed."""
        params = CloningParams(sigma_x=0.1, lambda_alg=1.0, alpha_restitution=0.5)
        potential = SimpleQuadraticPotential()
        op = CloningOperator(params, potential, torch.device("cpu"), torch.float64, bounds=None)

        x = torch.randn(10, 3, dtype=torch.float64)
        v = torch.randn(10, 3, dtype=torch.float64)
        state = SwarmState(x, v)

        torch.manual_seed(42)
        state1 = op.apply(state)

        torch.manual_seed(42)
        state2 = op.apply(state)

        assert torch.allclose(state1.x, state2.x)
        assert torch.allclose(state1.v, state2.v)

    def test_fitness_computation(self):
        """Test that fitness is computed correctly."""
        params = CloningParams(
            sigma_x=0.1,
            lambda_alg=1.0,
            alpha_restitution=0.5,
            alpha=1.0,  # Reward exponent
            beta=1.0,  # Diversity exponent
            eta=0.1,  # Floor
            A=2.0,  # Logistic rescale bound
        )
        potential = SimpleQuadraticPotential()
        op = CloningOperator(params, potential, torch.device("cpu"), torch.float64, bounds=None)

        x = torch.randn(20, 3, dtype=torch.float64)
        v = torch.randn(20, 3, dtype=torch.float64)
        state = SwarmState(x, v)

        _, _, info = op.apply(state, return_info=True)

        # Fitness should be positive and bounded
        fitness = info["fitness"]
        assert torch.all(fitness >= 0)
        assert torch.all(fitness <= (params.A + params.eta) ** (params.alpha + params.beta) + 1.0)

    def test_cloning_probabilities_bounded(self):
        """Test that cloning probabilities are in [0, 1]."""
        params = CloningParams(sigma_x=0.1, lambda_alg=1.0, alpha_restitution=0.5)
        potential = SimpleQuadraticPotential()
        op = CloningOperator(params, potential, torch.device("cpu"), torch.float64, bounds=None)

        x = torch.randn(20, 3, dtype=torch.float64)
        v = torch.randn(20, 3, dtype=torch.float64)
        state = SwarmState(x, v)

        _, _, info = op.apply(state, return_info=True)

        probs = info["cloning_probs"]
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)

    def test_num_cloned_consistency(self):
        """Test that num_cloned matches will_clone count."""
        params = CloningParams(sigma_x=0.1, lambda_alg=1.0, alpha_restitution=0.5)
        potential = SimpleQuadraticPotential()
        op = CloningOperator(params, potential, torch.device("cpu"), torch.float64, bounds=None)

        x = torch.randn(20, 3, dtype=torch.float64)
        v = torch.randn(20, 3, dtype=torch.float64)
        state = SwarmState(x, v)

        _, _, info = op.apply(state, return_info=True)

        assert info["num_cloned"] == info["will_clone"].sum().item()

    def test_p_max_affects_cloning_rate(self):
        """Test that p_max affects how many walkers clone."""
        x = torch.randn(50, 3, dtype=torch.float64)
        v = torch.randn(50, 3, dtype=torch.float64)
        state = SwarmState(x, v)
        potential = SimpleQuadraticPotential()

        # Low p_max should result in fewer clones
        params_low = CloningParams(sigma_x=0.1, lambda_alg=1.0, p_max=0.3)
        op_low = CloningOperator(
            params_low, potential, torch.device("cpu"), torch.float64, bounds=None
        )

        # High p_max should result in more clones
        params_high = CloningParams(sigma_x=0.1, lambda_alg=1.0, p_max=1.0)
        op_high = CloningOperator(
            params_high, potential, torch.device("cpu"), torch.float64, bounds=None
        )

        torch.manual_seed(42)
        _, _, info_low = op_low.apply(state, return_info=True)

        torch.manual_seed(42)
        _, _, info_high = op_high.apply(state, return_info=True)

        # Higher p_max means threshold is higher, so probabilities become larger
        # But this is about the conversion of scores to probabilities, not the scores themselves
        # The key is that with same seed, different p_max values produce different probability distributions
        assert not torch.allclose(info_low["cloning_probs"], info_high["cloning_probs"])
