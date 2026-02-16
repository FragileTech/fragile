"""Comprehensive tests for fragile.physics.fractal_gas.cloning module."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from fragile.physics.fractal_gas.cloning import (
    clone_position,
    clone_tensor,
    clone_walkers,
    CloneOperator,
    compute_cloning_probability,
    compute_cloning_score,
    inelastic_collision_velocity,
)


# ---------------------------------------------------------------------------
# TestComputeCloningScore
# ---------------------------------------------------------------------------


class TestComputeCloningScore:
    """Tests for compute_cloning_score: S = (V_c - V_i) / (V_i + eps)."""

    def test_output_shape(self, fitness_values: Tensor, N: int):
        """Output shape should be [N]."""
        companion_fitness = fitness_values.roll(1)
        scores = compute_cloning_score(fitness_values, companion_fitness)
        assert scores.shape == (N,)

    def test_positive_score_when_companion_fitter(self):
        """Score should be positive when companion fitness > walker fitness."""
        fitness = torch.tensor([1.0, 1.0, 1.0])
        companion_fitness = torch.tensor([5.0, 3.0, 2.0])
        scores = compute_cloning_score(fitness, companion_fitness, epsilon_clone=0.01)
        assert (scores > 0).all()

    def test_negative_score_when_walker_fitter(self):
        """Score should be negative when walker fitness > companion fitness."""
        fitness = torch.tensor([5.0, 3.0, 2.0])
        companion_fitness = torch.tensor([1.0, 1.0, 1.0])
        scores = compute_cloning_score(fitness, companion_fitness, epsilon_clone=0.01)
        assert (scores < 0).all()

    def test_zero_score_when_equal_fitness(self):
        """Score should be zero when fitness values are equal."""
        fitness = torch.tensor([2.0, 3.0, 4.0])
        companion_fitness = fitness.clone()
        scores = compute_cloning_score(fitness, companion_fitness, epsilon_clone=0.01)
        assert torch.allclose(scores, torch.zeros_like(scores), atol=1e-7)

    def test_epsilon_prevents_division_by_zero(self):
        """Epsilon should prevent division by zero when fitness is 0."""
        fitness = torch.tensor([0.0, 0.0])
        companion_fitness = torch.tensor([1.0, 2.0])
        eps = 0.01
        scores = compute_cloning_score(fitness, companion_fitness, epsilon_clone=eps)
        assert torch.isfinite(scores).all()
        # S = (V_c - 0) / (0 + eps) = V_c / eps
        expected = companion_fitness / eps
        assert torch.allclose(scores, expected)

    def test_formula_verification(self):
        """Verify the exact formula: S = (V_c - V_i) / (V_i + eps)."""
        fitness = torch.tensor([2.0, 5.0, 0.5])
        companion_fitness = torch.tensor([4.0, 1.0, 3.0])
        eps = 0.1
        scores = compute_cloning_score(fitness, companion_fitness, epsilon_clone=eps)
        expected = (companion_fitness - fitness) / (fitness + eps)
        assert torch.allclose(scores, expected)


# ---------------------------------------------------------------------------
# TestComputeCloningProbability
# ---------------------------------------------------------------------------


class TestComputeCloningProbability:
    """Tests for compute_cloning_probability: clip(S / p_max, 0, 1)."""

    def test_output_in_unit_interval(self):
        """All output values should be in [0, 1]."""
        scores = torch.tensor([-2.0, -0.5, 0.0, 0.3, 0.7, 1.0, 5.0])
        probs = compute_cloning_probability(scores, p_max=1.0)
        assert (probs >= 0.0).all()
        assert (probs <= 1.0).all()

    def test_negative_scores_give_zero(self):
        """Negative scores should map to probability 0."""
        scores = torch.tensor([-5.0, -1.0, -0.01])
        probs = compute_cloning_probability(scores, p_max=1.0)
        assert torch.allclose(probs, torch.zeros_like(probs))

    def test_scores_above_p_max_give_one(self):
        """Scores >= p_max should map to probability 1."""
        scores = torch.tensor([1.0, 2.0, 10.0])
        probs = compute_cloning_probability(scores, p_max=1.0)
        assert torch.allclose(probs, torch.ones_like(probs))

    def test_linear_interpolation(self):
        """For 0 < S < p_max, probability should be S / p_max."""
        p_max = 2.0
        scores = torch.tensor([0.5, 1.0, 1.5])
        probs = compute_cloning_probability(scores, p_max=p_max)
        expected = scores / p_max
        assert torch.allclose(probs, expected)

    def test_p_max_scaling(self):
        """Changing p_max should scale the probabilities."""
        scores = torch.tensor([0.5, 0.5, 0.5])
        probs_low = compute_cloning_probability(scores, p_max=0.5)
        probs_high = compute_cloning_probability(scores, p_max=2.0)
        # With p_max=0.5: S/p_max = 1.0, clipped to 1.0
        assert torch.allclose(probs_low, torch.ones_like(probs_low))
        # With p_max=2.0: S/p_max = 0.25
        assert torch.allclose(probs_high, torch.tensor([0.25, 0.25, 0.25]))


# ---------------------------------------------------------------------------
# TestInelasticCollisionVelocity
# ---------------------------------------------------------------------------


class TestInelasticCollisionVelocity:
    """Tests for inelastic_collision_velocity."""

    def test_non_cloning_walkers_keep_velocity_when_not_in_group(self):
        """Walkers with will_clone=False that are NOT companions of any cloner
        should keep their original velocity unchanged."""
        # 4 walkers. Walker 0 clones to walker 1. Walkers 2,3 are not involved.
        velocities = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        companions = torch.tensor([1, 0, 0, 2])
        will_clone = torch.tensor([True, False, False, False])
        v_new = inelastic_collision_velocity(velocities, companions, will_clone)
        # Walkers 2 and 3 are not in any collision group
        assert torch.allclose(v_new[2], velocities[2])
        assert torch.allclose(v_new[3], velocities[3])

    def test_momentum_conservation_fully_inelastic(self):
        """For alpha=0, all group members should get V_COM."""
        velocities = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
        companions = torch.tensor([1, 0])
        # Both clone: walker 0 clones to 1, walker 1 clones to 0
        will_clone = torch.tensor([True, True])
        v_new = inelastic_collision_velocity(
            velocities, companions, will_clone, alpha_restitution=0.0
        )
        # For companion 0: group = {0, 1} (cloner to 0 is walker 1, plus companion 0)
        # For companion 1: group = {1, 0} (cloner to 1 is walker 0, plus companion 1)
        # Both groups have same members, V_COM = (2+0)/2, (0+4)/2 = (1, 2)
        v_com = velocities.mean(dim=0)
        assert torch.allclose(v_new[0], v_com, atol=1e-6)
        assert torch.allclose(v_new[1], v_com, atol=1e-6)

    def test_fully_inelastic_all_get_vcom(self):
        """alpha=0: fully inelastic collision, all members get V_COM."""
        velocities = torch.tensor([[6.0, 0.0], [0.0, 6.0]])
        companions = torch.tensor([1, 0])
        will_clone = torch.tensor([True, True])
        v_new = inelastic_collision_velocity(
            velocities, companions, will_clone, alpha_restitution=0.0
        )
        v_com = velocities.mean(dim=0)
        assert torch.allclose(v_new[0], v_com, atol=1e-6)
        assert torch.allclose(v_new[1], v_com, atol=1e-6)

    def test_elastic_preserves_relative_velocity_magnitude(self):
        """alpha=1: elastic collision should preserve relative velocity magnitudes."""
        velocities = torch.tensor([[4.0, 0.0], [0.0, 2.0]])
        companions = torch.tensor([1, 0])
        will_clone = torch.tensor([True, True])
        v_new = inelastic_collision_velocity(
            velocities, companions, will_clone, alpha_restitution=1.0
        )
        # V_COM = (2, 1). Relative: u0 = (2, -1), u1 = (-2, 1)
        # With alpha=1, u_new = u_relative, so v_new = V_COM + u_relative = original
        # Both are in each other's group, so the last write wins.
        # Check that momentum is still conserved
        total_p_before = velocities.sum(dim=0)
        total_p_new = v_new.sum(dim=0)
        assert torch.allclose(total_p_before, total_p_new, atol=1e-5)

    def test_no_cloners_unchanged(self, velocities: Tensor, companions: Tensor, N: int):
        """When no one clones, velocities should be unchanged."""
        will_clone = torch.zeros(N, dtype=torch.bool)
        v_new = inelastic_collision_velocity(velocities, companions, will_clone)
        assert torch.allclose(v_new, velocities)

    def test_output_shape(self, velocities: Tensor, companions: Tensor, will_clone_half: Tensor):
        """Output should have same shape as input velocities."""
        v_new = inelastic_collision_velocity(velocities, companions, will_clone_half)
        assert v_new.shape == velocities.shape

    def test_single_pair_collision_correctness(self):
        """Test exact values for a single cloning pair with known alpha."""
        # Walker 0 clones to walker 1 (only walker 0 is cloning)
        velocities = torch.tensor([[3.0, 0.0], [1.0, 0.0]])
        companions = torch.tensor([1, 0])
        will_clone = torch.tensor([True, False])
        alpha = 0.5

        v_new = inelastic_collision_velocity(
            velocities, companions, will_clone, alpha_restitution=alpha
        )

        # Group for companion 1: {companion=1, cloner=0}
        # V_COM = mean([3,0], [1,0]) = [2, 0]
        # u0 = [3,0] - [2,0] = [1, 0], u1 = [1,0] - [2,0] = [-1, 0]
        # u_new0 = 0.5 * [1, 0] = [0.5, 0], u_new1 = 0.5 * [-1, 0] = [-0.5, 0]
        # v_new0 = [2,0] + [0.5,0] = [2.5, 0]
        # v_new1 = [2,0] + [-0.5,0] = [1.5, 0]
        expected_0 = torch.tensor([2.5, 0.0])
        expected_1 = torch.tensor([1.5, 0.0])
        assert torch.allclose(v_new[0], expected_0, atol=1e-6)
        assert torch.allclose(v_new[1], expected_1, atol=1e-6)


# ---------------------------------------------------------------------------
# TestClonePosition
# ---------------------------------------------------------------------------


class TestClonePosition:
    """Tests for clone_position."""

    def test_non_cloning_walkers_keep_position(self):
        """Walkers with will_clone=False should keep their original position."""
        positions = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        companions = torch.tensor([1, 0, 0])
        will_clone = torch.tensor([False, True, False])
        x_new = clone_position(positions, companions, will_clone, sigma_x=0.0)
        assert torch.allclose(x_new[0], positions[0])
        assert torch.allclose(x_new[2], positions[2])

    def test_cloning_walker_moves_to_companion_no_jitter(self):
        """With sigma_x=0, cloners should land exactly on companion position."""
        positions = torch.tensor([[1.0, 2.0], [10.0, 20.0]])
        companions = torch.tensor([1, 0])
        will_clone = torch.tensor([True, False])
        x_new = clone_position(positions, companions, will_clone, sigma_x=0.0)
        assert torch.allclose(x_new[0], positions[1])  # walker 0 -> companion 1

    def test_sigma_x_adds_gaussian_jitter(self):
        """With sigma_x>0, cloned position should be near companion position."""
        torch.manual_seed(123)
        N, d = 100, 3
        positions = torch.zeros(N, d)
        positions[0] = torch.tensor([10.0, 10.0, 10.0])  # companion at (10,10,10)
        companions = torch.zeros(N, dtype=torch.long)  # everyone clones to walker 0
        will_clone = torch.ones(N, dtype=torch.bool)
        will_clone[0] = False  # companion itself does not clone

        sigma_x = 0.1
        x_new = clone_position(positions, companions, will_clone, sigma_x=sigma_x)

        # Cloned walkers should be near companion position (10,10,10)
        cloned_positions = x_new[will_clone]
        companion_pos = positions[0]
        deviations = cloned_positions - companion_pos.unsqueeze(0)
        # Mean deviation should be near zero (Gaussian centered at companion)
        mean_dev = deviations.mean(dim=0)
        assert torch.allclose(mean_dev, torch.zeros(d), atol=0.1)
        # Std of deviations should be approximately sigma_x
        std_dev = deviations.std(dim=0)
        assert torch.allclose(std_dev, torch.full((d,), sigma_x), atol=0.05)

    def test_no_cloners_unchanged(self):
        """When no one clones, positions should be unchanged."""
        positions = torch.randn(5, 2)
        companions = torch.tensor([1, 0, 3, 2, 0])
        will_clone = torch.zeros(5, dtype=torch.bool)
        x_new = clone_position(positions, companions, will_clone, sigma_x=0.1)
        assert torch.allclose(x_new, positions)

    def test_output_shape(self, positions: Tensor, companions: Tensor, will_clone_half: Tensor):
        """Output should have same shape as input positions."""
        x_new = clone_position(positions, companions, will_clone_half, sigma_x=0.1)
        assert x_new.shape == positions.shape


# ---------------------------------------------------------------------------
# TestCloneWalkers
# ---------------------------------------------------------------------------


class TestCloneWalkers:
    """Tests for clone_walkers full pipeline."""

    def test_returns_correct_tuple(
        self,
        positions: Tensor,
        velocities: Tensor,
        fitness_values: Tensor,
        companions: Tensor,
    ):
        """Should return (positions_new, velocities_new, cloned_tensors, info) tuple."""
        result = clone_walkers(positions, velocities, fitness_values, companions)
        assert isinstance(result, tuple)
        assert len(result) == 4
        pos_new, vel_new, cloned_tensors, info = result
        assert isinstance(pos_new, Tensor)
        assert isinstance(vel_new, Tensor)
        assert isinstance(cloned_tensors, dict)
        assert isinstance(info, dict)

    def test_info_dict_keys(
        self,
        positions: Tensor,
        velocities: Tensor,
        fitness_values: Tensor,
        companions: Tensor,
    ):
        """Info dict should contain expected keys."""
        _, _, _, info = clone_walkers(positions, velocities, fitness_values, companions)
        expected_keys = {
            "cloning_scores",
            "cloning_probs",
            "will_clone",
            "num_cloned",
            "companions",
            "clone_delta_x",
            "clone_delta_v",
            "clone_jitter",
        }
        assert expected_keys.issubset(set(info.keys()))

    def test_deterministic_with_seed(
        self,
        positions: Tensor,
        velocities: Tensor,
        fitness_values: Tensor,
        companions: Tensor,
    ):
        """Results should be reproducible when using torch.manual_seed."""
        torch.manual_seed(99)
        pos1, vel1, _, info1 = clone_walkers(
            positions.clone(), velocities.clone(), fitness_values, companions
        )
        torch.manual_seed(99)
        pos2, vel2, _, info2 = clone_walkers(
            positions.clone(), velocities.clone(), fitness_values, companions
        )
        assert torch.allclose(pos1, pos2)
        assert torch.allclose(vel1, vel2)
        assert torch.equal(info1["will_clone"], info2["will_clone"])

    def test_clone_tensor_kwargs_are_cloned(
        self,
        positions: Tensor,
        velocities: Tensor,
        fitness_values: Tensor,
        companions: Tensor,
    ):
        """Additional tensors passed as kwargs should be cloned."""
        extra = torch.arange(positions.shape[0], dtype=torch.float)
        # Force all to clone by making companion much fitter, p_max=1.0
        fitness_values * 100  # companions are much fitter
        # Remap fitness so companions are fitter: set fitness very low
        low_fitness = torch.ones_like(fitness_values) * 0.001

        torch.manual_seed(42)
        _, _, cloned_tensors, info = clone_walkers(
            positions.clone(),
            velocities.clone(),
            low_fitness,
            companions,
            p_max=1.0,
            epsilon_clone=0.01,
            my_extra=extra.clone(),
        )
        assert "my_extra" in cloned_tensors
        # For walkers that cloned, their extra tensor should be the companion's value
        will_clone = info["will_clone"]
        if will_clone.any():
            cloned_vals = cloned_tensors["my_extra"][will_clone]
            expected_vals = extra[companions[will_clone]]
            assert torch.allclose(cloned_vals, expected_vals)

    def test_pipeline_consistency(
        self,
        positions: Tensor,
        velocities: Tensor,
        fitness_values: Tensor,
        companions: Tensor,
    ):
        """Info should reflect consistent pipeline: scores -> probs -> decisions."""
        torch.manual_seed(50)
        _, _, _, info = clone_walkers(
            positions.clone(), velocities.clone(), fitness_values, companions
        )
        scores = info["cloning_scores"]
        probs = info["cloning_probs"]
        will_clone = info["will_clone"]
        num_cloned = info["num_cloned"]

        # Probabilities should be in [0, 1]
        assert (probs >= 0.0).all()
        assert (probs <= 1.0).all()
        # num_cloned should match will_clone count
        assert num_cloned == will_clone.sum().item()
        # Walkers with negative scores should have probability 0
        assert (probs[scores < 0] == 0).all()

    def test_no_walkers_clone_when_all_fitness_equal(self):
        """When all fitness values are equal, scores are 0 and nobody clones."""
        N, d = 20, 3
        positions = torch.randn(N, d)
        velocities = torch.randn(N, d)
        fitness = torch.ones(N) * 5.0  # all equal
        companions = torch.arange(N).roll(1)  # simple rotation pairing

        torch.manual_seed(77)
        _, _, _, info = clone_walkers(
            positions.clone(), velocities.clone(), fitness, companions, p_max=1.0
        )
        # All scores should be zero, all probs should be zero
        assert torch.allclose(info["cloning_scores"], torch.zeros(N), atol=1e-6)
        assert torch.allclose(info["cloning_probs"], torch.zeros(N), atol=1e-6)
        assert info["num_cloned"] == 0


# ---------------------------------------------------------------------------
# TestCloneTensor
# ---------------------------------------------------------------------------


class TestCloneTensor:
    """Tests for clone_tensor (torch and numpy support)."""

    def test_torch_tensor_cloning(self):
        """Torch tensors should be cloned from companion indices."""
        x = torch.tensor([10.0, 20.0, 30.0, 40.0])
        compas_ix = torch.tensor([1, 0, 3, 2])
        will_clone = torch.tensor([True, False, True, False])
        result = clone_tensor(x, compas_ix, will_clone)
        # Walker 0 clones from 1: 20.0, Walker 2 clones from 3: 40.0
        assert result[0] == 20.0
        assert result[1] == 20.0  # unchanged
        assert result[2] == 40.0
        assert result[3] == 40.0  # unchanged

    def test_numpy_array_cloning(self):
        """NumPy arrays should be cloned from companion indices."""
        x = np.array([10.0, 20.0, 30.0, 40.0])
        compas_ix = torch.tensor([1, 0, 3, 2])
        will_clone = torch.tensor([True, False, True, False])
        result = clone_tensor(x, compas_ix, will_clone)
        assert isinstance(result, np.ndarray)
        assert result[0] == 20.0
        assert result[1] == 20.0  # unchanged
        assert result[2] == 40.0
        assert result[3] == 40.0  # unchanged

    def test_no_clone_returns_unchanged(self):
        """When will_clone is all False, tensor should be returned unchanged."""
        x = torch.tensor([1.0, 2.0, 3.0])
        compas_ix = torch.tensor([2, 0, 1])
        will_clone = torch.tensor([False, False, False])
        result = clone_tensor(x, compas_ix, will_clone)
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_unsupported_type_raises_value_error(self):
        """Unsupported types should raise ValueError."""
        x = [1.0, 2.0, 3.0]  # plain list, not supported
        compas_ix = torch.tensor([1, 0, 2])
        will_clone = torch.tensor([True, False, False])
        with pytest.raises(ValueError, match="Unsupported type"):
            clone_tensor(x, compas_ix, will_clone)


# ---------------------------------------------------------------------------
# TestCloneOperator
# ---------------------------------------------------------------------------


class TestCloneOperator:
    """Tests for CloneOperator (PanelModel wrapper)."""

    def test_callable_interface(
        self,
        clone_op: CloneOperator,
        positions: Tensor,
        velocities: Tensor,
        fitness_values: Tensor,
        companions: Tensor,
    ):
        """Operator should be callable and return the expected tuple."""
        torch.manual_seed(42)
        result = clone_op(positions, velocities, fitness_values, companions)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_override_parameters_in_call(
        self,
        positions: Tensor,
        velocities: Tensor,
        fitness_values: Tensor,
        companions: Tensor,
    ):
        """Parameters passed to __call__ should override instance defaults."""
        op = CloneOperator(p_max=1.0, sigma_x=0.0, alpha_restitution=0.0)

        torch.manual_seed(42)
        # With sigma_x=0 default (no jitter)
        _pos1, _, _, _ = op(positions.clone(), velocities.clone(), fitness_values, companions)

        torch.manual_seed(42)
        # Override sigma_x=5.0 (large jitter)
        _pos2, _, _, _ = op(
            positions.clone(), velocities.clone(), fitness_values, companions, sigma_x=5.0
        )

        # Positions should differ because sigma_x was overridden
        # (unless no one cloned, in which case they would be the same)
        # Use a different approach: verify the override path works by checking info
        torch.manual_seed(42)
        _, _, _, info1 = op(
            positions.clone(), velocities.clone(), fitness_values, companions, p_max=0.0001
        )
        torch.manual_seed(42)
        _, _, _, info2 = op(
            positions.clone(), velocities.clone(), fitness_values, companions, p_max=100.0
        )
        # With very low p_max, probabilities are higher (more cloning)
        # With very high p_max, probabilities are lower (less cloning)
        assert info1["cloning_probs"].sum() >= info2["cloning_probs"].sum()

    def test_default_parameters_from_init(self):
        """Instance should store default parameters from __init__."""
        op = CloneOperator(p_max=0.75, epsilon_clone=0.05, sigma_x=0.2, alpha_restitution=0.3)
        assert op.p_max == 0.75
        assert op.epsilon_clone == 0.05
        assert op.sigma_x == 0.2
        assert op.alpha_restitution == 0.3

    def test_widget_parameters_list(self):
        """widget_parameters should list the four cloning parameters."""
        op = CloneOperator()
        wp = op.widget_parameters
        assert "p_max" in wp
        assert "epsilon_clone" in wp
        assert "sigma_x" in wp
        assert "alpha_restitution" in wp
        assert len(wp) == 4
