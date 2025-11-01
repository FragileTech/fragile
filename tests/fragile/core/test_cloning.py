"""Tests for CloneOperator and cloning functions from core.cloning."""

import pytest
import torch

from fragile.core.cloning import (
    clone_walkers,
    CloneOperator,
    compute_cloning_probability,
    compute_cloning_score,
    inelastic_collision_velocity,
)


class TestCloningFunctions:
    """Test individual cloning functions."""

    def test_compute_cloning_score(self):
        """Test cloning score computation."""
        fitness = torch.tensor([1.0, 2.0, 3.0])
        companion_fitness = torch.tensor([2.0, 1.0, 3.0])

        scores = compute_cloning_score(fitness, companion_fitness, epsilon_clone=0.01)

        # First walker: (2-1) / (1+0.01) ≈ 0.99 (should clone)
        # Second walker: (1-2) / (2+0.01) ≈ -0.50 (should NOT clone)
        # Third walker: (3-3) / (3+0.01) ≈ 0 (neutral)

        assert scores[0] > 0  # First should clone
        assert scores[1] < 0  # Second should not clone
        assert abs(scores[2]) < 0.01  # Third is neutral

    def test_compute_cloning_probability(self):
        """Test conversion of scores to probabilities."""
        scores = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
        p_max = 1.0

        probs = compute_cloning_probability(scores, p_max=p_max)

        # Negative scores → 0 probability
        assert probs[0] == 0.0
        # Zero score → 0 probability
        assert probs[1] == 0.0
        # 0.5 score with p_max=1.0 → 0.5 probability
        assert probs[2] == 0.5
        # 1.0 score with p_max=1.0 → 1.0 probability (clipped)
        assert probs[3] == 1.0
        # 2.0 score → 1.0 probability (clipped)
        assert probs[4] == 1.0

    def test_inelastic_collision_velocity_fully_inelastic(self):
        """Test fully inelastic collision (alpha=0)."""
        # Two walkers with different velocities
        velocities = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
        companions = torch.tensor([1, 0])  # Paired with each other
        will_clone = torch.tensor([True, True])

        new_velocities = inelastic_collision_velocity(
            velocities, companions, will_clone, alpha_restitution=0.0
        )

        # With alpha=0, both should get center of mass velocity: (1-1)/2 = 0
        expected = torch.zeros_like(velocities)
        assert torch.allclose(new_velocities, expected, atol=1e-6)

    def test_inelastic_collision_velocity_elastic(self):
        """Test elastic collision (alpha=1)."""
        # Two walkers with different velocities
        velocities = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
        companions = torch.tensor([1, 0])  # Paired with each other
        will_clone = torch.tensor([True, True])

        new_velocities = inelastic_collision_velocity(
            velocities, companions, will_clone, alpha_restitution=1.0
        )

        # With alpha=1, elastic collision preserves the relative velocities
        # For symmetric collision (same mass, opposite velocities),
        # the velocities remain the same (momentum and KE conserved)
        expected = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
        assert torch.allclose(new_velocities, expected, atol=1e-6)


class TestCloneOperator:
    """Tests for CloneOperator class."""

    @pytest.fixture
    def clone_op(self):
        """Create clone operator for testing."""
        return CloneOperator(
            p_max=1.0,
            epsilon_clone=0.01,
            sigma_x=0.1,
            alpha_restitution=0.5,
        )

    def test_initialization(self, clone_op):
        """Test clone operator initialization."""
        assert clone_op.p_max == 1.0
        assert clone_op.epsilon_clone == 0.01
        assert clone_op.sigma_x == 0.1
        assert clone_op.alpha_restitution == 0.5

    def test_call_preserves_shape(self, clone_op):
        """Test that cloning preserves tensor shapes."""
        N, d = 10, 3
        positions = torch.randn(N, d, dtype=torch.float64)
        velocities = torch.randn(N, d, dtype=torch.float64)
        fitness = torch.randn(N, dtype=torch.float64)
        companions = torch.randint(0, N, (N,))
        alive = torch.ones(N, dtype=torch.bool)

        pos_new, vel_new, _cloned_dict, _info = clone_op(
            positions, velocities, fitness, companions, alive
        )

        assert pos_new.shape == positions.shape
        assert vel_new.shape == velocities.shape

    def test_call_changes_state(self, clone_op):
        """Test that cloning actually changes positions/velocities."""
        torch.manual_seed(42)
        N, d = 10, 3
        positions = torch.randn(N, d, dtype=torch.float64)
        velocities = torch.randn(N, d, dtype=torch.float64)
        # Create fitness gradient to ensure some cloning
        fitness = torch.linspace(0.0, 2.0, N, dtype=torch.float64)
        companions = torch.randint(0, N, (N,))
        alive = torch.ones(N, dtype=torch.bool)

        pos_new, _vel_new, _cloned_dict, info = clone_op(
            positions, velocities, fitness, companions, alive
        )

        # At least some walkers should have cloned
        assert info["num_cloned"] > 0
        # Positions should change
        assert not torch.allclose(pos_new, positions)

    def test_return_info_structure(self, clone_op):
        """Test that info dict has expected structure."""
        N, d = 10, 3
        positions = torch.randn(N, d, dtype=torch.float64)
        velocities = torch.randn(N, d, dtype=torch.float64)
        fitness = torch.randn(N, dtype=torch.float64)
        companions = torch.randint(0, N, (N,))
        alive = torch.ones(N, dtype=torch.bool)

        _, _, _, info = clone_op(positions, velocities, fitness, companions, alive)

        # Check info dictionary structure
        assert "cloning_scores" in info
        assert "cloning_probs" in info
        assert "will_clone" in info
        assert "num_cloned" in info
        assert "companions" in info

        # Check shapes
        assert info["cloning_scores"].shape == (N,)
        assert info["cloning_probs"].shape == (N,)
        assert info["will_clone"].shape == (N,)
        assert info["companions"].shape == (N,)
        assert isinstance(info["num_cloned"], int)

    def test_p_max_affects_cloning_rate(self):
        """Test that p_max affects how many walkers clone."""
        N, d = 50, 3
        positions = torch.randn(N, d, dtype=torch.float64)
        velocities = torch.randn(N, d, dtype=torch.float64)
        # Create fitness gradient
        fitness = torch.linspace(0.0, 2.0, N, dtype=torch.float64)
        companions = torch.randint(0, N, (N,))
        alive = torch.ones(N, dtype=torch.bool)

        # Low p_max should result in fewer clones
        op_low = CloneOperator(p_max=0.3, sigma_x=0.1)
        torch.manual_seed(42)
        _, _, _, info_low = op_low(positions, velocities, fitness, companions, alive)

        # High p_max should result in more clones
        op_high = CloneOperator(p_max=1.0, sigma_x=0.1)
        torch.manual_seed(42)
        _, _, _, info_high = op_high(positions, velocities, fitness, companions, alive)

        # Higher p_max should lead to more cloning
        # (probabilities are higher for same scores)
        assert not torch.allclose(info_low["cloning_probs"], info_high["cloning_probs"])

    def test_reproducibility_with_seed(self, clone_op):
        """Test that cloning is reproducible with same seed."""
        N, d = 10, 3
        positions = torch.randn(N, d, dtype=torch.float64)
        velocities = torch.randn(N, d, dtype=torch.float64)
        fitness = torch.randn(N, dtype=torch.float64)
        companions = torch.randint(0, N, (N,))
        alive = torch.ones(N, dtype=torch.bool)

        torch.manual_seed(42)
        pos1, vel1, _, _ = clone_op(positions, velocities, fitness, companions, alive)

        torch.manual_seed(42)
        pos2, vel2, _, _ = clone_op(positions, velocities, fitness, companions, alive)

        assert torch.allclose(pos1, pos2)
        assert torch.allclose(vel1, vel2)

    def test_dead_walkers_revived(self, clone_op):
        """Test that dead walkers are revived through cloning."""
        N, d = 10, 3
        positions = torch.randn(N, d, dtype=torch.float64)
        velocities = torch.randn(N, d, dtype=torch.float64)
        fitness = torch.randn(N, dtype=torch.float64)
        companions = torch.randint(0, N, (N,))

        # Mark half as dead
        alive = torch.ones(N, dtype=torch.bool)
        alive[N // 2 :] = False

        # Set fitness to 0 for dead walkers (as compute_fitness would do)
        fitness[~alive] = 0.0

        torch.manual_seed(42)
        _pos_new, _vel_new, _, info = clone_op(positions, velocities, fitness, companions, alive)

        # Dead walkers should have cloned (they get infinite cloning score)
        # At minimum, the dead walkers should be among those that cloned
        dead_indices = torch.where(~alive)[0]
        assert all(info["will_clone"][i] for i in dead_indices)

    @pytest.mark.parametrize("alpha_restitution", [0.0, 0.5, 1.0])
    def test_restitution_coefficient(self, alpha_restitution):
        """Test different restitution coefficients."""
        op = CloneOperator(
            p_max=1.0,
            epsilon_clone=0.01,
            sigma_x=0.1,
            alpha_restitution=alpha_restitution,
        )

        N, d = 5, 2
        positions = torch.randn(N, d, dtype=torch.float64)
        velocities = torch.randn(N, d, dtype=torch.float64)
        fitness = torch.randn(N, dtype=torch.float64)
        companions = torch.randint(0, N, (N,))
        alive = torch.ones(N, dtype=torch.bool)

        torch.manual_seed(42)
        _, vel_new, _, _info = op(positions, velocities, fitness, companions, alive)

        # Test basic properties
        assert not torch.any(torch.isnan(vel_new))
        assert not torch.any(torch.isinf(vel_new))
        assert vel_new.shape == velocities.shape

    def test_cloning_probabilities_bounded(self, clone_op):
        """Test that cloning probabilities are in [0, 1]."""
        N, d = 20, 3
        positions = torch.randn(N, d, dtype=torch.float64)
        velocities = torch.randn(N, d, dtype=torch.float64)
        fitness = torch.randn(N, dtype=torch.float64)
        companions = torch.randint(0, N, (N,))
        alive = torch.ones(N, dtype=torch.bool)

        _, _, _, info = clone_op(positions, velocities, fitness, companions, alive)

        probs = info["cloning_probs"]
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)

    def test_num_cloned_consistency(self, clone_op):
        """Test that num_cloned matches will_clone count."""
        N, d = 20, 3
        positions = torch.randn(N, d, dtype=torch.float64)
        velocities = torch.randn(N, d, dtype=torch.float64)
        fitness = torch.randn(N, dtype=torch.float64)
        companions = torch.randint(0, N, (N,))
        alive = torch.ones(N, dtype=torch.bool)

        _, _, _, info = clone_op(positions, velocities, fitness, companions, alive)

        assert info["num_cloned"] == info["will_clone"].sum().item()

    def test_clone_additional_tensors(self, clone_op):
        """Test that additional tensors are cloned correctly."""
        N, d = 10, 3
        positions = torch.randn(N, d, dtype=torch.float64)
        velocities = torch.randn(N, d, dtype=torch.float64)
        fitness = torch.linspace(0.0, 2.0, N, dtype=torch.float64)
        companions = torch.randint(0, N, (N,))
        alive = torch.ones(N, dtype=torch.bool)

        # Additional tensor to clone
        custom_field = torch.randn(N, dtype=torch.float64)

        torch.manual_seed(42)
        _pos_new, _vel_new, cloned_dict, info = clone_op(
            positions, velocities, fitness, companions, alive, custom_field=custom_field
        )

        # Check that custom_field was cloned
        assert "custom_field" in cloned_dict
        assert cloned_dict["custom_field"].shape == custom_field.shape

        # For walkers that cloned, custom_field should match their companion's value
        cloners = torch.where(info["will_clone"])[0]
        for i in cloners:
            comp_idx = companions[i]
            assert cloned_dict["custom_field"][i] == custom_field[comp_idx]
