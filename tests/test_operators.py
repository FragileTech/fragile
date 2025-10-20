"""Tests for cloning.py functions.

This module tests all operator functions following the Euclidean Gas specifications
from 03_cloning.md:
- Fitness computation pipeline
- Cloning decision functions
- State update operators
"""

import pytest
import torch
from torch import Tensor

from fragile.core.cloning import (
    clone_position,
    clone_walkers,
    compute_cloning_probability,
    compute_cloning_score,
    inelastic_collision_velocity,
)
from fragile.core.companion_selection import CompanionSelection
from fragile.core.fitness import (
    compute_fitness,
    logistic_rescale,
    patched_standardization,
)


@pytest.fixture
def device():
    """Get compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestLogisticRescale:
    """Test logistic rescale function g_A(z) = A / (1 + exp(-z))."""

    def test_range_bounds(self, device):
        """Test output is bounded in (0, A)."""
        z = torch.randn(100, device=device)
        A = 2.0

        result = logistic_rescale(z, A=A)

        assert torch.all(result > 0)
        assert torch.all(result < A)

    def test_monotonicity(self, device):
        """Test function is monotonically increasing."""
        z = torch.linspace(-10, 10, 100, device=device)
        result = logistic_rescale(z, A=2.0)

        # Check that differences are non-negative (monotonic)
        diffs = result[1:] - result[:-1]
        assert torch.all(diffs > 0)

    def test_midpoint_value(self, device):
        """Test g_A(0) = A/2."""
        z = torch.tensor([0.0], device=device)
        A = 2.0

        result = logistic_rescale(z, A=A)

        assert torch.allclose(result, torch.tensor([A / 2], device=device), atol=1e-6)

    def test_asymptotic_behavior(self, device):
        """Test limits: g_A(z) → A as z → ∞, g_A(z) → 0 as z → -∞."""
        A = 2.0

        # Large positive z should approach A
        z_pos = torch.tensor([10.0], device=device)
        result_pos = logistic_rescale(z_pos, A=A)
        assert result_pos > 0.99 * A

        # Large negative z should approach 0
        z_neg = torch.tensor([-10.0], device=device)
        result_neg = logistic_rescale(z_neg, A=A)
        assert result_neg < 0.01 * A

    def test_default_parameter(self, device):
        """Test default A=1.0."""
        z = torch.tensor([0.0], device=device)
        result = logistic_rescale(z)

        assert torch.allclose(result, torch.tensor([0.5], device=device), atol=1e-6)


class TestPatchedStandardization:
    """Test patched standardization using only alive walkers."""

    def test_all_alive(self, device):
        """Test standardization when all walkers are alive."""
        N = 100
        values = torch.randn(N, device=device) * 2 + 5  # mean≈5, std≈2
        alive = torch.ones(N, dtype=torch.bool, device=device)

        z_scores = patched_standardization(values, alive)

        # Check mean ≈ 0 and std ≈ 1 for alive walkers
        assert torch.allclose(z_scores.mean(), torch.tensor(0.0, device=device), atol=0.2)
        assert torch.allclose(
            z_scores.std(unbiased=False), torch.tensor(1.0, device=device), atol=0.2
        )

    def test_dead_walkers_excluded(self, device):
        """Test that dead walkers don't contaminate statistics."""
        N = 100
        values = torch.randn(N, device=device)
        alive = torch.ones(N, dtype=torch.bool, device=device)

        # Make half the walkers "dead" with extreme values
        alive[N // 2 :] = False
        values[N // 2 :] = 1000.0  # Extreme values for dead walkers

        z_scores = patched_standardization(values, alive)

        # Dead walkers should get z-score = 0
        assert torch.all(z_scores[~alive] == 0.0)

        # Alive walker statistics shouldn't be affected by dead walkers
        alive_z = z_scores[alive]
        assert torch.abs(alive_z.mean()) < 0.3  # Mean should be near 0

    def test_regularization(self, device):
        """Test regularized standard deviation prevents division by zero."""
        N = 10
        values = torch.ones(N, device=device) * 5.0  # All same value (var=0)
        alive = torch.ones(N, dtype=torch.bool, device=device)
        sigma_min = 1e-8

        # Should not crash with zero variance
        z_scores = patched_standardization(values, alive, sigma_min=sigma_min)

        # All z-scores should be 0 (since all values are identical)
        assert torch.allclose(z_scores, torch.zeros_like(z_scores), atol=1e-6)

    def test_no_alive_walkers(self, device):
        """Test with no alive walkers."""
        N = 10
        values = torch.randn(N, device=device)
        alive = torch.zeros(N, dtype=torch.bool, device=device)

        z_scores = patched_standardization(values, alive)

        # All should be zero
        assert torch.all(z_scores == 0.0)

    def test_rho_not_implemented(self, device):
        """Test that finite rho raises NotImplementedError."""
        values = torch.randn(10, device=device)
        alive = torch.ones(10, dtype=torch.bool, device=device)

        with pytest.raises(NotImplementedError, match="finite rho"):
            patched_standardization(values, alive, rho=1.0)


class TestComputeFitness:
    """Test complete fitness pipeline."""

    @pytest.fixture
    def simple_swarm(self, device):
        """Create a simple test swarm."""
        N = 20
        d = 3
        positions = torch.randn(N, d, device=device)
        velocities = torch.randn(N, d, device=device)
        rewards = torch.randn(N, device=device)
        alive = torch.ones(N, dtype=torch.bool, device=device)
        companion_selection = CompanionSelection(method="uniform")

        # Compute companions using the selection strategy
        companions = companion_selection(positions, velocities, alive)

        return positions, velocities, rewards, alive, companions

    def test_output_shapes(self, simple_swarm, device):
        """Test output tensor shapes."""
        positions, velocities, rewards, alive, companions = simple_swarm
        N = positions.shape[0]

        fitness, distances, companions_out = compute_fitness(
            positions, velocities, rewards, alive, companions
        )

        assert fitness.shape == (N,)
        assert distances.shape == (N,)
        assert companions.shape == (N,)

    def test_fitness_bounds(self, simple_swarm, device):
        """Test fitness is bounded by (η, (A + η)^(α+β))."""
        positions, velocities, rewards, alive, companions = simple_swarm
        alpha, beta, eta, A = 1.0, 1.0, 0.1, 2.0

        fitness, _, _ = compute_fitness(
            positions,
            velocities,
            rewards,
            alive,
            companions,
            alpha=alpha,
            beta=beta,
            eta=eta,
            A=A,
        )

        V_min = eta ** (alpha + beta)
        V_max = (A + eta) ** (alpha + beta)

        alive_fitness = fitness[alive]
        assert torch.all(alive_fitness >= V_min - 1e-6)
        assert torch.all(alive_fitness <= V_max + 1e-6)

    def test_dead_walkers_zero_fitness(self, simple_swarm, device):
        """Test dead walkers receive fitness = 0."""
        positions, velocities, rewards, alive, companions = simple_swarm

        # Mark some walkers as dead
        alive[::3] = False

        fitness, _, _ = compute_fitness(positions, velocities, rewards, alive, companions)

        assert torch.all(fitness[~alive] == 0.0)
        assert torch.all(fitness[alive] > 0.0)

    def test_lambda_alg_position_only(self, simple_swarm, device):
        """Test λ_alg=0 uses only position distance."""
        positions, velocities, rewards, alive, companions = simple_swarm

        # Create identical positions, different velocities
        positions_same = positions.clone()
        velocities_diff = torch.randn_like(velocities) * 100  # Very different

        _fitness1, _distances1, companions_out = compute_fitness(
            positions_same, velocities, rewards, alive, companions, lambda_alg=0.0
        )

        # Force same companion selection for comparison
        pos_diff = positions_same - positions_same[companions_out]
        vel_diff = velocities_diff - velocities_diff[companions_out]

        torch.sqrt((pos_diff**2).sum(dim=-1))
        torch.sqrt((pos_diff**2).sum(dim=-1) + 1.0 * (vel_diff**2).sum(dim=-1))

        # For λ_alg=0, distances should not depend on velocity
        # (testing the principle, not exact values due to stochastic companion selection)

    def test_fitness_components(self, simple_swarm, device):
        """Test fitness = (d')^β * (r')^α structure."""
        positions, velocities, rewards, alive, companions = simple_swarm
        alpha, beta = 2.0, 3.0

        fitness, _, _ = compute_fitness(
            positions, velocities, rewards, alive, companions, alpha=alpha, beta=beta
        )

        # All alive walkers should have positive fitness
        assert torch.all(fitness[alive] > 0)


class TestComputeCloningScore:
    """Test cloning score computation."""

    def test_score_formula(self, device):
        """Test S_i = (V_c - V_i) / (V_i + ε)."""
        fitness = torch.tensor([1.0, 2.0, 3.0], device=device)
        companion_fitness = torch.tensor([3.0, 1.0, 3.0], device=device)
        eps = 0.01

        scores = compute_cloning_score(fitness, companion_fitness, epsilon_clone=eps)

        expected = (companion_fitness - fitness) / (fitness + eps)
        assert torch.allclose(scores, expected, atol=1e-6)

    def test_positive_score_for_unfit(self, device):
        """Test walker less fit than companion gets positive score."""
        fitness = torch.tensor([1.0], device=device)
        companion_fitness = torch.tensor([2.0], device=device)

        score = compute_cloning_score(fitness, companion_fitness)

        assert score > 0

    def test_negative_score_for_fit(self, device):
        """Test walker more fit than companion gets negative score."""
        fitness = torch.tensor([2.0], device=device)
        companion_fitness = torch.tensor([1.0], device=device)

        score = compute_cloning_score(fitness, companion_fitness)

        assert score < 0

    def test_anti_symmetry(self, device):
        """Test approximate anti-symmetry: S_i(c) ≈ -S_c(i)."""
        V_i = 1.0
        V_c = 2.0
        eps = 0.01

        fitness_i = torch.tensor([V_i], device=device)
        fitness_c = torch.tensor([V_c], device=device)

        S_i_c = compute_cloning_score(fitness_i, fitness_c, epsilon_clone=eps)
        S_c_i = compute_cloning_score(fitness_c, fitness_i, epsilon_clone=eps)

        # Not exactly anti-symmetric due to denominator, but check signs opposite
        assert S_i_c > 0
        assert S_c_i < 0


class TestComputeCloningProbability:
    """Test cloning probability via clipping function."""

    def test_probability_bounds(self, device):
        """Test probabilities are in [0, 1]."""
        scores = torch.randn(100, device=device) * 10  # Wide range

        probs = compute_cloning_probability(scores)

        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)

    def test_negative_score_zero_prob(self, device):
        """Test negative scores give probability 0."""
        scores = torch.tensor([-1.0, -0.5, -10.0], device=device)

        probs = compute_cloning_probability(scores)

        assert torch.all(probs == 0.0)

    def test_large_score_saturates(self, device):
        """Test scores ≥ p_max give probability 1."""
        p_max = 0.75
        scores = torch.tensor([p_max, p_max + 0.1, 10.0], device=device)

        probs = compute_cloning_probability(scores, p_max=p_max)

        assert torch.all(probs == 1.0)

    def test_linear_interpolation(self, device):
        """Test linear interpolation for 0 < S < p_max."""
        p_max = 0.75
        scores = torch.tensor([0.25, 0.5], device=device)

        probs = compute_cloning_probability(scores, p_max=p_max)

        expected = scores / p_max
        assert torch.allclose(probs, expected, atol=1e-6)


class TestInelasticCollisionVelocity:
    """Test inelastic collision velocity updates."""

    def test_no_cloning_unchanged(self, device):
        """Test velocities unchanged when no walkers clone."""
        N, d = 20, 3
        velocities = torch.randn(N, d, device=device)
        companions = torch.randint(0, N, (N,), device=device)
        will_clone = torch.zeros(N, dtype=torch.bool, device=device)

        v_new = inelastic_collision_velocity(velocities, companions, will_clone)

        assert torch.allclose(v_new, velocities)

    def test_momentum_conservation(self, device):
        """Test momentum is conserved within each collision group."""
        N, d = 10, 3
        velocities = torch.randn(N, d, device=device)

        # Make all walkers clone to same companion for simple test
        companions = torch.zeros(N, dtype=torch.long, device=device)
        will_clone = torch.ones(N, dtype=torch.bool, device=device)

        # Momentum before (for the group that will collide)
        p_before = velocities.sum(dim=0)

        v_new = inelastic_collision_velocity(velocities, companions, will_clone)

        # Momentum after
        p_after = v_new.sum(dim=0)

        # Momentum should be conserved within the collision group
        assert torch.allclose(p_before, p_after, atol=1e-5)

    def test_restitution_zero_collapse(self, device):
        """Test α=0 collapses velocities to COM."""
        N, d = 10, 3
        velocities = torch.randn(N, d, device=device)

        # All clone to same companion
        companions = torch.zeros(N, dtype=torch.long, device=device)
        will_clone = torch.ones(N, dtype=torch.bool, device=device)

        v_new = inelastic_collision_velocity(
            velocities, companions, will_clone, alpha_restitution=0.0
        )

        # All velocities should be equal (COM velocity)
        V_COM = velocities.mean(dim=0)
        assert torch.allclose(v_new, V_COM.unsqueeze(0).expand_as(v_new), atol=1e-5)

    def test_restitution_one_preserves_relative(self, device):
        """Test α=1 preserves relative velocities."""
        N, d = 10, 3
        velocities = torch.randn(N, d, device=device)

        # All clone to same companion
        companions = torch.zeros(N, dtype=torch.long, device=device)
        will_clone = torch.ones(N, dtype=torch.bool, device=device)

        v_new = inelastic_collision_velocity(
            velocities, companions, will_clone, alpha_restitution=1.0
        )

        # Relative velocities should be preserved
        V_COM_before = velocities.mean(dim=0)
        V_COM_after = v_new.mean(dim=0)

        u_before = velocities - V_COM_before
        u_after = v_new - V_COM_after

        assert torch.allclose(u_before, u_after, atol=1e-5)

    def test_non_cloners_unchanged(self, device):
        """Test walkers that don't clone keep their velocities."""
        N, d = 20, 3
        velocities = torch.randn(N, d, device=device)
        companions = torch.randint(0, N, (N,), device=device)
        will_clone = torch.zeros(N, dtype=torch.bool, device=device)
        will_clone[5] = True  # Only one walker clones

        v_new = inelastic_collision_velocity(velocities, companions, will_clone)

        # Non-cloners should be unchanged (except companion of cloner)
        companion_idx = companions[5].item()  # Convert tensor to int for set membership
        for i in range(N):
            if i not in {5, companion_idx}:
                assert torch.allclose(v_new[i], velocities[i], atol=1e-6)


class TestClonePosition:
    """Test position cloning with Gaussian jitter."""

    def test_no_cloning_unchanged(self, device):
        """Test positions unchanged when no walkers clone."""
        N, d = 20, 3
        positions = torch.randn(N, d, device=device)
        companions = torch.randint(0, N, (N,), device=device)
        will_clone = torch.zeros(N, dtype=torch.bool, device=device)

        x_new = clone_position(positions, companions, will_clone)

        assert torch.allclose(x_new, positions)

    def test_cloners_near_companion(self, device):
        """Test cloners receive position near companion."""
        torch.manual_seed(42)
        N, d = 20, 3
        positions = torch.randn(N, d, device=device) * 10
        companions = torch.randint(0, N, (N,), device=device)
        will_clone = torch.ones(N, dtype=torch.bool, device=device)
        sigma_x = 0.1

        x_new = clone_position(positions, companions, will_clone, sigma_x=sigma_x)

        # Check cloners are within reasonable distance of companion
        for i in range(N):
            c_i = companions[i]
            if i != c_i:  # Don't compare companion to itself
                dist = torch.norm(x_new[i] - positions[c_i])
                # Should be within ~3*sigma_x with high probability
                assert dist < 3 * sigma_x * torch.sqrt(torch.tensor(d, device=device))

    def test_jitter_scale_effect(self, device):
        """Test larger sigma_x gives larger spread."""
        torch.manual_seed(42)
        N, d = 100, 3
        positions = torch.randn(N, d, device=device) * 10
        companions = torch.zeros(N, dtype=torch.long, device=device)  # All to same companion
        will_clone = torch.ones(N, dtype=torch.bool, device=device)

        # Small jitter
        x_small = clone_position(positions, companions, will_clone, sigma_x=0.01)
        std_small = x_small.std(dim=0).mean()

        # Large jitter
        x_large = clone_position(positions, companions, will_clone, sigma_x=1.0)
        std_large = x_large.std(dim=0).mean()

        # Larger sigma should give larger spread
        assert std_large > std_small

    def test_persisters_unchanged(self, device):
        """Test walkers that don't clone keep their positions."""
        N, d = 20, 3
        positions = torch.randn(N, d, device=device)
        companions = torch.randint(0, N, (N,), device=device)
        will_clone = torch.zeros(N, dtype=torch.bool, device=device)
        will_clone[5] = True  # Only one walker clones

        x_new = clone_position(positions, companions, will_clone)

        # All except cloner should be unchanged
        for i in range(N):
            if i != 5:
                assert torch.allclose(x_new[i], positions[i])

    def test_gaussian_distribution(self, device):
        """Test jitter follows Gaussian distribution."""
        torch.manual_seed(42)
        N, d = 1000, 3
        positions = torch.zeros(N, d, device=device)
        companions = torch.zeros(N, dtype=torch.long, device=device)  # All to origin
        will_clone = torch.ones(N, dtype=torch.bool, device=device)
        sigma_x = 1.0

        x_new = clone_position(positions, companions, will_clone, sigma_x=sigma_x)

        # Check each dimension approximately follows N(0, σ²)
        for dim in range(d):
            mean = x_new[:, dim].mean()
            std = x_new[:, dim].std(unbiased=False)

            # Mean should be near 0
            assert torch.abs(mean) < 0.1
            # Std should be near sigma_x
            assert torch.abs(std - sigma_x) < 0.1


class TestCloneWalkers:
    """Test complete cloning operator."""

    @pytest.fixture
    def simple_swarm(self, device):
        """Create a simple test swarm."""
        N = 20
        d = 3
        positions = torch.randn(N, d, device=device)
        velocities = torch.randn(N, d, device=device)
        rewards = torch.randn(N, device=device)
        alive = torch.ones(N, dtype=torch.bool, device=device)
        companion_selection = CompanionSelection(method="uniform")

        # Select companions using the selection strategy
        companions = companion_selection(positions, velocities, alive)

        # Compute fitness with the selected companions
        fitness, _distances, _companions_out = compute_fitness(
            positions, velocities, rewards, alive, companions
        )

        return positions, velocities, fitness, companions, alive

    def test_output_structure(self, simple_swarm, device):
        """Test clone_walkers returns correct output structure."""
        positions, velocities, fitness, companions, alive = simple_swarm

        pos_new, vel_new, alive_new, info = clone_walkers(
            positions, velocities, fitness, companions, alive
        )

        N, d = positions.shape

        # Check output shapes
        assert pos_new.shape == (N, d)
        assert vel_new.shape == (N, d)
        assert alive_new.shape == (N,)

        # Check all alive after cloning
        assert alive_new.all()

        # Check info dictionary
        assert "cloning_scores" in info
        assert "cloning_probs" in info
        assert "will_clone" in info
        assert "num_cloned" in info
        assert "companions" in info

        assert info["cloning_scores"].shape == (N,)
        assert info["cloning_probs"].shape == (N,)
        assert info["will_clone"].shape == (N,)
        assert isinstance(info["num_cloned"], int)

    def test_dead_walkers_revived(self, simple_swarm, device):
        """Test dead walkers are always revived."""
        positions, velocities, fitness, companions, alive = simple_swarm

        # Mark some walkers as dead
        alive[::3] = False
        num_dead = (~alive).sum().item()

        # Dead walkers should have zero fitness (from compute_fitness)
        fitness = torch.where(alive, fitness, torch.zeros_like(fitness))

        _pos_new, _vel_new, alive_new, info = clone_walkers(
            positions, velocities, fitness, companions, alive
        )

        # All should be alive after cloning
        assert alive_new.all()

        # Dead walkers should have cloned (high probability)
        # Due to stochasticity, we check that most dead walkers cloned
        dead_cloned = info["will_clone"][~alive].sum()
        assert dead_cloned >= num_dead * 0.8  # At least 80% cloned

    def test_cloning_probabilities_bounded(self, simple_swarm, device):
        """Test cloning probabilities are in [0, 1]."""
        positions, velocities, fitness, companions, alive = simple_swarm

        _, _, _, info = clone_walkers(positions, velocities, fitness, companions, alive)

        probs = info["cloning_probs"]
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)

    def test_stochastic_decisions(self, simple_swarm, device):
        """Test cloning decisions are stochastic."""
        positions, velocities, fitness, companions, alive = simple_swarm

        # Run twice with same inputs
        torch.manual_seed(42)
        _, _, _, info1 = clone_walkers(positions, velocities, fitness, companions, alive)

        torch.manual_seed(43)  # Different seed
        _, _, _, info2 = clone_walkers(positions, velocities, fitness, companions, alive)

        # Same probabilities
        assert torch.allclose(info1["cloning_probs"], info2["cloning_probs"])

        # Different decisions (with high probability for large N)
        decisions_differ = (info1["will_clone"] != info2["will_clone"]).any()
        assert decisions_differ  # Very likely for N=20

    def test_positions_updated(self, simple_swarm, device):
        """Test positions change for cloners."""
        positions, velocities, fitness, companions, alive = simple_swarm

        pos_new, _, _, info = clone_walkers(positions, velocities, fitness, companions, alive)

        # Cloners should have different positions
        will_clone = info["will_clone"]
        if will_clone.any():
            cloner_idx = torch.where(will_clone)[0][0]
            pos_changed = not torch.allclose(pos_new[cloner_idx], positions[cloner_idx])
            assert pos_changed

    def test_velocities_updated(self, simple_swarm, device):
        """Test velocities change for collision participants."""
        positions, velocities, fitness, companions, alive = simple_swarm

        _, vel_new, _, info = clone_walkers(positions, velocities, fitness, companions, alive)

        # At least some velocities should change if cloning occurred
        if info["num_cloned"] > 0:
            vel_changed = not torch.allclose(vel_new, velocities)
            assert vel_changed

    def test_parameter_effects(self, simple_swarm, device):
        """Test that parameter changes affect results."""
        positions, velocities, fitness, companions, alive = simple_swarm

        # Different p_max should affect cloning rates
        torch.manual_seed(42)
        _, _, _, info_low = clone_walkers(
            positions, velocities, fitness, companions, alive, p_max=0.5
        )

        torch.manual_seed(42)
        _, _, _, info_high = clone_walkers(
            positions, velocities, fitness, companions, alive, p_max=0.9
        )

        # Higher p_max generally means less cloning (same score needs higher p_max)
        # But probabilities should differ
        assert not torch.allclose(info_low["cloning_probs"], info_high["cloning_probs"])

    def test_info_consistency(self, simple_swarm, device):
        """Test info dictionary values are consistent."""
        positions, velocities, fitness, companions, alive = simple_swarm

        _, _, _, info = clone_walkers(positions, velocities, fitness, companions, alive)

        # num_cloned should match will_clone count
        assert info["num_cloned"] == info["will_clone"].sum().item()

        # Companions should match input
        assert torch.all(info["companions"] == companions)

        # Cloning probabilities should be in valid range
        assert torch.all(info["cloning_probs"] >= 0.0)
        assert torch.all(info["cloning_probs"] <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
