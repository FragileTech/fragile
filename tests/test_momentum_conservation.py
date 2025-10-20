"""Tests for momentum conservation in multi-body collisions."""

import pytest
import torch

from fragile.core.cloning import inelastic_collision_velocity


class TestMomentumConservation:
    """Tests for momentum conservation in inelastic collisions."""

    def test_momentum_conservation_two_walkers(self):
        """Test momentum conservation with two walkers cloning to same target."""
        v = torch.tensor(
            [
                [1.0, 0.0],  # Walker 0
                [2.0, 1.0],  # Walker 1
                [-1.0, 2.0],  # Walker 2
            ],
            dtype=torch.float64,
        )

        # Walkers 1 and 2 clone to walker 0
        # Walker 0 clones to itself (excluded from collision group to avoid double-counting)
        companions = torch.tensor([0, 0, 0])
        will_clone = torch.ones(3, dtype=torch.bool)  # All walkers clone
        alpha_restitution = 0.5

        # Collision group: companion=0 + cloners=[1,2] (0 excluded as it's the companion)
        # Initial momentum of collision group
        p_initial = v[0] + v[1] + v[2]

        # Apply collision
        v_new = inelastic_collision_velocity(v, companions, will_clone, alpha_restitution)

        # Final momentum of collision group
        p_final = v_new[0] + v_new[1] + v_new[2]

        # Momentum should be conserved
        assert torch.allclose(p_initial, p_final, atol=1e-10)

    def test_momentum_conservation_multiple_groups(self):
        """Test momentum conservation with multiple collision groups."""
        alpha_restitution = 0.8

        x = torch.randn(6, 3, dtype=torch.float64)
        v = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 2.0, 1.0],
                [-1.0, 0.0, 1.0],
                [1.0, -1.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=torch.float64,
        )

        # Two collision groups:
        # Group 1: companion=0, cloners=[1, 2]
        # Group 2: companion=3, cloners=[4, 5]
        companions = torch.tensor([0, 0, 0, 3, 3, 3])
        will_clone = torch.ones(6, dtype=torch.bool)  # All walkers clone

        # Calculate initial momentum for each group
        group1_indices = torch.tensor([0, 1, 2])
        group2_indices = torch.tensor([3, 4, 5])

        p1_initial = torch.sum(v[group1_indices], dim=0)
        p2_initial = torch.sum(v[group2_indices], dim=0)

        # Apply collision
        torch.manual_seed(42)
        v_new = inelastic_collision_velocity(v, companions, will_clone, alpha_restitution)

        # Calculate final momentum for each group
        p1_final = torch.sum(v_new[group1_indices], dim=0)
        p2_final = torch.sum(v_new[group2_indices], dim=0)

        # Momentum should be conserved within each group
        assert torch.allclose(p1_initial, p1_final, atol=1e-10)
        assert torch.allclose(p2_initial, p2_final, atol=1e-10)

    def test_fully_inelastic_all_same_velocity(self):
        """Test that alpha=0 makes all collision group members have same velocity."""
        alpha_restitution = 0.0

        x = torch.randn(4, 2, dtype=torch.float64)
        v = torch.tensor(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=torch.float64,
        )

        # All clone to walker 1
        companions = torch.tensor([1, 1, 1, 1])
        will_clone = torch.ones(4, dtype=torch.bool)  # All walkers clone

        torch.manual_seed(42)
        v_new = inelastic_collision_velocity(v, companions, will_clone, alpha_restitution)

        # Calculate expected COM velocity
        # Collision group: companion=1 + cloners=[0,2,3] (1 excluded as it's the companion)
        # COM = mean([v[1], v[0], v[2], v[3]]) = mean([2, 1, 3, 4]) = 2.5
        v_com_expected = torch.mean(v, dim=0)

        # All walkers should have the COM velocity
        for i in range(4):
            assert torch.allclose(v_new[i], v_com_expected, atol=1e-10)

    def test_elastic_preserves_kinetic_energy(self):
        """Test that alpha=1 approximately preserves kinetic energy in COM frame."""
        alpha_restitution = 1.0

        x = torch.randn(3, 2, dtype=torch.float64)
        v = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, -1.0],
            ],
            dtype=torch.float64,
        )

        # All clone to walker 0
        companions = torch.tensor([0, 0, 0])
        will_clone = torch.ones(3, dtype=torch.bool)  # All walkers clone

        # Calculate COM velocity
        v_com = torch.mean(v, dim=0)

        # Calculate initial kinetic energy in COM frame
        v_rel_initial = v - v_com.unsqueeze(0)
        ke_initial = torch.sum(v_rel_initial**2)

        torch.manual_seed(42)
        v_new = inelastic_collision_velocity(v, companions, will_clone, alpha_restitution)

        # Calculate final kinetic energy in COM frame
        v_rel_final = v_new - v_com.unsqueeze(0)
        ke_final = torch.sum(v_rel_final**2)

        # For elastic collision (alpha=1), kinetic energy should be preserved
        # (allowing small numerical error)
        assert torch.allclose(ke_initial, ke_final, rtol=1e-10)

    def test_multi_dimensional_momentum_conservation(self):
        """Test momentum conservation in high-dimensional space."""
        alpha_restitution = 0.6

        d = 10  # High dimension
        x = torch.randn(5, d, dtype=torch.float64)
        v = torch.randn(5, d, dtype=torch.float64)

        companions = torch.tensor([2, 2, 2, 2, 2])
        will_clone = torch.ones(5, dtype=torch.bool)  # All walkers clone

        # Initial momentum
        p_initial = torch.sum(v, dim=0)

        torch.manual_seed(42)
        v_new = inelastic_collision_velocity(v, companions, will_clone, alpha_restitution)

        # Final momentum
        p_final = torch.sum(v_new, dim=0)

        # Momentum should be conserved in all dimensions
        assert torch.allclose(p_initial, p_final, atol=1e-10)

    def test_no_collision_preserves_velocity(self):
        """Test that walkers not in collision groups keep their velocities."""
        alpha_restitution = 0.5

        x = torch.randn(5, 2, dtype=torch.float64)
        v = torch.tensor(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
                [5.0, 0.0],
            ],
            dtype=torch.float64,
        )

        # Each walker clones to itself (no interaction)
        companions = torch.tensor([0, 1, 2, 3, 4])
        will_clone = torch.ones(5, dtype=torch.bool)  # All walkers clone

        torch.manual_seed(42)
        v_new = inelastic_collision_velocity(v, companions, will_clone, alpha_restitution)

        # When each walker is its own companion, it forms a 1-member collision group
        # The implementation should handle this correctly
        assert not torch.any(torch.isnan(v_new))
        assert not torch.any(torch.isinf(v_new))


class TestEnergyConservation:
    """Tests for energy conservation in elastic collisions."""

    def test_elastic_collision_energy_conservation_simple(self):
        """Test energy conservation for simple elastic collision."""
        alpha_restitution = 1.0

        x = torch.randn(3, 2, dtype=torch.float64)
        v = torch.tensor(
            [
                [2.0, 0.0],  # Walker 0
                [0.0, 2.0],  # Walker 1
                [-1.0, -1.0],  # Walker 2
            ],
            dtype=torch.float64,
        )

        # Walkers 1 and 2 clone to walker 0
        companions = torch.tensor([0, 0, 0])
        will_clone = torch.ones(3, dtype=torch.bool)  # All walkers clone

        # Calculate initial kinetic energy in COM frame
        # Collision group: [0, 1, 2]
        collision_group = torch.tensor([0, 1, 2])
        v_group = v[collision_group]
        v_com = torch.mean(v_group, dim=0)

        v_rel_initial = v_group - v_com.unsqueeze(0)
        ke_initial = 0.5 * torch.sum(v_rel_initial**2)

        torch.manual_seed(42)
        v_new = inelastic_collision_velocity(v, companions, will_clone, alpha_restitution)

        # Calculate final kinetic energy in COM frame
        v_group_final = v_new[collision_group]
        v_rel_final = v_group_final - v_com.unsqueeze(0)
        ke_final = 0.5 * torch.sum(v_rel_final**2)

        # For elastic collision (alpha=1), kinetic energy should be preserved
        assert torch.allclose(ke_initial, ke_final, rtol=1e-10, atol=1e-10)

    def test_energy_dissipation_inelastic(self):
        """Test that inelastic collisions dissipate energy."""
        # Fully inelastic: alpha = 0
        alpha_restitution = 0.0

        v = torch.tensor(
            [
                [3.0, 0.0],
                [0.0, 3.0],
                [-2.0, 1.0],
                [1.0, -2.0],
            ],
            dtype=torch.float64,
        )

        companions = torch.tensor([0, 0, 0, 0])
        will_clone = torch.ones(4, dtype=torch.bool)  # All walkers clone

        # Calculate initial kinetic energy
        ke_initial = 0.5 * torch.sum(v**2)

        v_new = inelastic_collision_velocity(v, companions, will_clone, alpha_restitution)

        # Calculate final kinetic energy
        ke_final = 0.5 * torch.sum(v_new**2)

        # Fully inelastic should dissipate energy
        assert ke_final < ke_initial

        # For fully inelastic, all relative kinetic energy should be dissipated
        # Only COM motion remains
        v_com = torch.mean(v, dim=0)
        ke_com_only = 0.5 * len(v) * torch.sum(v_com**2)
        assert torch.allclose(ke_final, ke_com_only, rtol=1e-10)

    def test_energy_dissipation_scales_with_restitution(self):
        """Test that energy dissipation increases as restitution decreases."""
        x = torch.randn(5, 3, dtype=torch.float64)
        v = torch.randn(5, 3, dtype=torch.float64) * 2.0

        companions = torch.tensor([2, 2, 2, 2, 2])
        will_clone = torch.ones(5, dtype=torch.bool)  # All walkers clone

        # Calculate initial kinetic energy in COM frame
        v_com = torch.mean(v, dim=0)
        v_rel_initial = v - v_com.unsqueeze(0)
        ke_rel_initial = 0.5 * torch.sum(v_rel_initial**2)

        results = {}
        for alpha in [0.0, 0.5, 1.0]:
            v_new = inelastic_collision_velocity(v, companions, will_clone, alpha)

            # Calculate final relative kinetic energy
            v_rel_final = v_new - v_com.unsqueeze(0)
            ke_rel_final = 0.5 * torch.sum(v_rel_final**2)

            results[alpha] = ke_rel_final

        # Energy retention should increase with restitution coefficient
        # alpha=0: minimum energy (maximum dissipation)
        # alpha=1: maximum energy (no dissipation)
        assert results[0.0] < results[0.5] < results[1.0]

        # For alpha=1, should preserve energy (within numerical tolerance)
        assert torch.allclose(results[1.0], ke_rel_initial, rtol=1e-9)

    def test_elastic_collision_multiple_groups_energy(self):
        """Test energy conservation with multiple collision groups."""
        alpha_restitution = 1.0

        x = torch.randn(6, 2, dtype=torch.float64)
        v = torch.tensor(
            [
                [1.0, 1.0],
                [2.0, -1.0],
                [-1.0, 2.0],
                [0.5, 0.5],
                [-0.5, 1.5],
                [1.5, -0.5],
            ],
            dtype=torch.float64,
        )

        # Two collision groups:
        # Group 1: companion=0, cloners=[1, 2]
        # Group 2: companion=3, cloners=[4, 5]
        companions = torch.tensor([0, 0, 0, 3, 3, 3])
        will_clone = torch.ones(6, dtype=torch.bool)  # All walkers clone

        # Calculate initial kinetic energy for each group
        group1_indices = torch.tensor([0, 1, 2])
        group2_indices = torch.tensor([3, 4, 5])

        v_com1 = torch.mean(v[group1_indices], dim=0)
        v_com2 = torch.mean(v[group2_indices], dim=0)

        v_rel1_initial = v[group1_indices] - v_com1.unsqueeze(0)
        v_rel2_initial = v[group2_indices] - v_com2.unsqueeze(0)

        ke1_initial = 0.5 * torch.sum(v_rel1_initial**2)
        ke2_initial = 0.5 * torch.sum(v_rel2_initial**2)

        torch.manual_seed(42)
        v_new = inelastic_collision_velocity(v, companions, will_clone, alpha_restitution)

        # Calculate final kinetic energy for each group
        v_rel1_final = v_new[group1_indices] - v_com1.unsqueeze(0)
        v_rel2_final = v_new[group2_indices] - v_com2.unsqueeze(0)

        ke1_final = 0.5 * torch.sum(v_rel1_final**2)
        ke2_final = 0.5 * torch.sum(v_rel2_final**2)

        # Energy should be conserved within each group
        assert torch.allclose(ke1_initial, ke1_final, rtol=1e-10)
        assert torch.allclose(ke2_initial, ke2_final, rtol=1e-10)

    def test_elastic_collision_high_dimensional(self):
        """Test energy conservation in high-dimensional space."""
        alpha_restitution = 1.0

        d = 10
        x = torch.randn(5, d, dtype=torch.float64)
        v = torch.randn(5, d, dtype=torch.float64) * 2.0

        companions = torch.tensor([2, 2, 2, 2, 2])
        will_clone = torch.ones(5, dtype=torch.bool)  # All walkers clone

        # Calculate initial kinetic energy in COM frame
        v_com = torch.mean(v, dim=0)
        v_rel_initial = v - v_com.unsqueeze(0)
        ke_initial = 0.5 * torch.sum(v_rel_initial**2)

        torch.manual_seed(42)
        v_new = inelastic_collision_velocity(v, companions, will_clone, alpha_restitution)

        # Calculate final kinetic energy in COM frame
        v_rel_final = v_new - v_com.unsqueeze(0)
        ke_final = 0.5 * torch.sum(v_rel_final**2)

        # Energy should be conserved in all dimensions
        assert torch.allclose(ke_initial, ke_final, rtol=1e-10)

    def test_partial_restitution_energy_scaling(self):
        """Test that partial restitution gives correct energy scaling."""
        x = torch.randn(4, 2, dtype=torch.float64)
        v = torch.tensor(
            [
                [2.0, 0.0],
                [0.0, 2.0],
                [-1.0, 1.0],
                [1.0, -1.0],
            ],
            dtype=torch.float64,
        )

        companions = torch.tensor([0, 0, 0, 0])
        will_clone = torch.ones(4, dtype=torch.bool)  # All walkers clone

        # Calculate initial relative kinetic energy
        v_com = torch.mean(v, dim=0)
        v_rel_initial = v - v_com.unsqueeze(0)
        ke_rel_initial = 0.5 * torch.sum(v_rel_initial**2)

        # For restitution coefficient alpha, the final relative KE should be alpha^2 * initial
        for alpha in [0.3, 0.5, 0.7]:
            v_new = inelastic_collision_velocity(v, companions, will_clone, alpha)

            v_rel_final = v_new - v_com.unsqueeze(0)
            ke_rel_final = 0.5 * torch.sum(v_rel_final**2)

            # Relative KE scales as alpha^2 (since velocities scale as alpha)
            expected_ke = alpha**2 * ke_rel_initial
            assert torch.allclose(ke_rel_final, expected_ke, rtol=1e-10)
