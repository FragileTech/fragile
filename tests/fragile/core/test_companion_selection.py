"""Comprehensive tests for companion selection mechanisms."""

import numpy as np
import pytest
import torch

from fragile.core.companion_selection import (
    compute_algorithmic_distance_matrix,
    random_pairing_fisher_yates,
    select_companions_for_cloning,
    select_companions_softmax,
    select_companions_uniform,
    sequential_greedy_pairing,
)


class TestAlgorithmicDistance:
    """Test algorithmic distance matrix computation."""

    def test_distance_matrix_shape(self, test_device):
        """Distance matrix should be [N, N]."""
        N, d = 10, 3
        x = torch.randn(N, d, device=test_device)
        v = torch.randn(N, d, device=test_device)

        dist_sq = compute_algorithmic_distance_matrix(x, v, lambda_alg=0.0)

        assert dist_sq.shape == (N, N)
        assert dist_sq.device.type == test_device

    def test_distance_matrix_symmetric(self):
        """Distance matrix should be symmetric."""
        N, d = 10, 3
        x = torch.randn(N, d)
        v = torch.randn(N, d)

        dist_sq = compute_algorithmic_distance_matrix(x, v, lambda_alg=1.0)

        assert torch.allclose(dist_sq, dist_sq.T, atol=1e-6)

    def test_distance_diagonal_zero(self):
        """Diagonal entries should be zero (distance from walker to itself)."""
        N, d = 10, 3
        x = torch.randn(N, d)
        v = torch.randn(N, d)

        dist_sq = compute_algorithmic_distance_matrix(x, v, lambda_alg=1.0)

        # Allow for numerical precision errors
        assert torch.allclose(dist_sq.diag(), torch.zeros(N), atol=1e-5)

    def test_position_only_mode(self):
        """With lambda_alg=0, only position distance should matter."""
        N, d = 5, 2
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
        v = torch.randn(N, d) * 100  # Large velocities should be ignored

        dist_sq = compute_algorithmic_distance_matrix(x, v, lambda_alg=0.0)

        # Distance from (0,0) to (1,0) should be 1.0
        assert torch.allclose(dist_sq[0, 1], torch.tensor(1.0), atol=1e-6)
        # Distance from (0,0) to (0,1) should be 1.0
        assert torch.allclose(dist_sq[0, 2], torch.tensor(1.0), atol=1e-6)
        # Distance from (0,0) to (1,1) should be 2.0
        assert torch.allclose(dist_sq[0, 3], torch.tensor(2.0), atol=1e-6)

    def test_phase_space_mode(self):
        """With lambda_alg>0, both position and velocity contribute."""
        _N, _d = 2, 1
        # Same position, different velocities
        x = torch.tensor([[0.0], [0.0]])
        v = torch.tensor([[0.0], [1.0]])

        dist_sq_pos_only = compute_algorithmic_distance_matrix(x, v, lambda_alg=0.0)
        dist_sq_phase = compute_algorithmic_distance_matrix(x, v, lambda_alg=1.0)

        # Position-only: distance should be 0
        assert torch.allclose(dist_sq_pos_only[0, 1], torch.tensor(0.0), atol=1e-6)
        # Phase-space: distance should be 1.0 (from velocity)
        assert torch.allclose(dist_sq_phase[0, 1], torch.tensor(1.0), atol=1e-6)


class TestCompanionSelectionProperties:
    """Test critical properties of companion selection."""

    def test_alive_walkers_only_selected_as_companions(self, test_device):
        """Property: Only alive walkers can be chosen as companions."""
        torch.manual_seed(42)
        N, d = 20, 2
        x = torch.randn(N, d, device=test_device)
        v = torch.randn(N, d, device=test_device)
        alive_mask = torch.tensor(
            [True] * 15 + [False] * 5, device=test_device
        )  # 15 alive, 5 dead

        # Test all selection functions
        companions_softmax = select_companions_softmax(
            x, v, alive_mask, epsilon=1.0, exclude_self=True
        )
        companions_uniform = select_companions_uniform(alive_mask)
        companions_cloning = select_companions_for_cloning(x, v, alive_mask, epsilon_c=0.5)

        # Get alive indices
        alive_indices = torch.where(alive_mask)[0]

        # Check softmax: alive walkers should only select alive companions
        for i in range(N):
            if alive_mask[i] and companions_softmax[i] != -1:
                assert (
                    companions_softmax[i] in alive_indices
                ), f"Alive walker {i} selected dead companion {companions_softmax[i].item()}"

        # Check uniform: all companions should be alive
        for i in range(N):
            assert (
                companions_uniform[i] in alive_indices
            ), f"Walker {i} selected dead companion {companions_uniform[i].item()}"

        # Check cloning: all companions should be alive
        for i in range(N):
            assert (
                companions_cloning[i] in alive_indices
            ), f"Walker {i} selected dead companion {companions_cloning[i].item()}"

    def test_dead_walkers_uniform_selection(self):
        """Property: Dead walkers choose companions uniformly."""
        torch.manual_seed(123)
        N, d = 50, 2
        x = torch.randn(N, d)
        v = torch.randn(N, d)

        # Create clustered positions for alive walkers to make softmax highly non-uniform
        alive_mask = torch.tensor([True] * 40 + [False] * 10)
        x[:40, :] = torch.randn(40, d) * 0.1  # Alive walkers clustered
        x[40:, :] = torch.randn(10, d) * 10  # Dead walkers far away

        # Run many trials to collect statistics
        n_trials = 1000
        dead_companion_counts = torch.zeros(40)  # Count selections for each alive walker

        for _ in range(n_trials):
            companions = select_companions_for_cloning(
                x, v, alive_mask, epsilon_c=0.1, lambda_alg=0.0
            )
            # Count companions selected by dead walkers
            dead_companions = companions[~alive_mask]
            for c in dead_companions:
                dead_companion_counts[c.item()] += 1

        # Test uniformity using chi-square test
        # Expected: each alive walker selected ~10*1000/40 = 250 times
        expected_count = (n_trials * 10) / 40
        dead_companion_counts / (n_trials * 10)

        # With uniform distribution, frequencies should be close to 1/40 = 0.025
        # Check that no walker is selected too rarely or too often (tolerance: ±50%)
        min_expected = expected_count * 0.5
        max_expected = expected_count * 1.5

        assert (dead_companion_counts >= min_expected).all(), (
            f"Some alive walkers selected too rarely by dead walkers. "
            f"Min count: {dead_companion_counts.min():.0f}, expected: {expected_count:.0f}"
        )
        assert (dead_companion_counts <= max_expected).all(), (
            f"Some alive walkers selected too often by dead walkers. "
            f"Max count: {dead_companion_counts.max():.0f}, expected: {expected_count:.0f}"
        )

    def test_alive_walkers_distance_dependent_selection(self):
        """Property: Alive walkers preferentially select nearby companions."""
        torch.manual_seed(456)
        d = 2

        # Create specific geometry: one walker nearby, one far
        x = torch.tensor([
            [0.0, 0.0],  # Walker 0
            [0.5, 0.0],  # Walker 1: close to 0 (distance 0.5)
            [5.0, 0.0],  # Walker 2: far from 0 (distance 5.0)
        ])
        v = torch.zeros(3, d)
        alive_mask = torch.tensor([True] * 3)

        # Run many trials
        n_trials = 1000
        walker0_companions = []

        for _ in range(n_trials):
            companions = select_companions_softmax(
                x, v, alive_mask, epsilon=1.0, lambda_alg=0.0, exclude_self=True
            )
            walker0_companions.append(companions[0].item())

        walker0_companions = torch.tensor(walker0_companions)

        # Walker 0 should prefer walker 1 (distance 0.5) over walker 2 (distance 5.0)
        # Count selections
        unique, counts = torch.unique(walker0_companions, return_counts=True)
        companion_counts = {int(u): int(c) for u, c in zip(unique, counts)}

        # Walker 1 should be selected much more frequently than walker 2
        # Softmax weights ratio: exp(-0.5²/(2*1²)) / exp(-5²/(2*1²))
        # ≈ exp(-0.125) / exp(-12.5) ≈ exp(12.375) ≈ 240000:1
        assert companion_counts.get(1, 0) > companion_counts.get(2, 0), (
            f"Walker 0 should strongly prefer nearby walker 1 over distant walker 2. "
            f"Counts: {companion_counts}"
        )

        # Walker 1 should be selected at least 99% of the time
        assert companion_counts.get(1, 0) > 0.99 * n_trials, (
            f"Walker 0 should select nearby walker 1 almost always. "
            f"Got {companion_counts.get(1, 0)} / {n_trials}, counts: {companion_counts}"
        )


class TestPairingProperties:
    """Test properties of mutual pairing algorithms."""

    def test_pairing_is_mutual(self):
        """Property: Pairing should be mutual (if c(i)=j then c(j)=i)."""
        torch.manual_seed(789)
        N, d = 20, 2
        x = torch.randn(N, d)
        v = torch.randn(N, d)
        alive_mask = torch.tensor([True] * 18 + [False] * 2)

        # Test both pairing methods
        companion_map_greedy = sequential_greedy_pairing(
            x, v, alive_mask, epsilon_d=1.0, lambda_alg=0.0
        )
        companion_map_random = random_pairing_fisher_yates(alive_mask)

        # Check mutuality for alive walkers
        alive_indices = torch.where(alive_mask)[0]
        for idx in alive_indices:
            i = idx.item()
            j = companion_map_greedy[i].item()
            if i != j:  # Not self-paired
                assert (
                    companion_map_greedy[j].item() == i
                ), f"Greedy pairing not mutual: c({i})={j} but c({j})={companion_map_greedy[j].item()}"

        for idx in alive_indices:
            i = idx.item()
            j = companion_map_random[i].item()
            if i != j:  # Not self-paired
                assert (
                    companion_map_random[j].item() == i
                ), f"Random pairing not mutual: c({i})={j} but c({j})={companion_map_random[j].item()}"

    def test_pairing_covers_alive_walkers(self):
        """Property: All alive walkers should be paired (or singleton if odd)."""
        torch.manual_seed(101)
        N, d = 21, 2  # Odd number of alive walkers
        x = torch.randn(N, d)
        v = torch.randn(N, d)
        alive_mask = torch.tensor([True] * 19 + [False] * 2)  # 19 alive (odd)

        companion_map = sequential_greedy_pairing(x, v, alive_mask, epsilon_d=1.0, lambda_alg=0.0)

        alive_indices = torch.where(alive_mask)[0]

        # Count how many walkers are paired vs self-paired
        self_paired = 0
        paired = 0

        for idx in alive_indices:
            i = idx.item()
            if companion_map[i].item() == i:
                self_paired += 1
            else:
                paired += 1

        # With 19 alive walkers: 18 should be paired (9 pairs), 1 should be self-paired
        assert self_paired == 1, f"Expected 1 self-paired walker, got {self_paired}"
        assert paired == 18, f"Expected 18 paired walkers, got {paired}"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_alive_walkers_error(self):
        """Should raise error when no walkers are alive."""
        N, d = 5, 2
        x = torch.randn(N, d)
        v = torch.randn(N, d)
        alive_mask = torch.tensor([False] * N)  # No alive walkers

        with pytest.raises(ValueError, match="No alive walkers"):
            select_companions_uniform(alive_mask)

        with pytest.raises(ValueError, match="No alive walkers"):
            select_companions_for_cloning(x, v, alive_mask, epsilon_c=1.0)

    def test_single_alive_walker(self):
        """With only one alive walker, functions should handle gracefully."""
        N, d = 5, 2
        torch.randn(N, d)
        torch.randn(N, d)
        alive_mask = torch.tensor([True, False, False, False, False])

        # Uniform selection: all should map to walker 0
        companions = select_companions_uniform(alive_mask)
        assert (companions == 0).all()

        # Pairing: walker 0 should map to itself
        companion_map = random_pairing_fisher_yates(alive_mask)
        assert companion_map[0].item() == 0

    def test_all_alive_walkers(self):
        """All walkers alive is a common case."""
        N, d = 10, 2
        x = torch.randn(N, d)
        v = torch.randn(N, d)
        alive_mask = torch.tensor([True] * N)

        companions = select_companions_for_cloning(x, v, alive_mask, epsilon_c=1.0)

        # All companions should be valid (in range [0, N))
        assert (companions >= 0).all()
        assert (companions < N).all()

        # All should be alive (which is all walkers in this case)
        assert alive_mask[companions].all()

    def test_exclude_self_property(self):
        """When exclude_self=True, walker should not select itself."""
        torch.manual_seed(999)
        N, d = 10, 2
        x = torch.randn(N, d)
        v = torch.randn(N, d)
        alive_mask = torch.tensor([True] * N)

        # Run many trials
        for _ in range(100):
            companions = select_companions_softmax(
                x, v, alive_mask, epsilon=1.0, lambda_alg=0.0, exclude_self=True
            )

            # No alive walker should select itself
            for i in range(N):
                if alive_mask[i]:
                    assert (
                        companions[i].item() != i
                    ), f"Walker {i} selected itself despite exclude_self=True"


class TestDeterminism:
    """Test reproducibility with fixed seed."""

    def test_softmax_reproducible(self):
        """Same seed should give same results."""
        N, d = 10, 2
        x = torch.randn(N, d)
        v = torch.randn(N, d)
        alive_mask = torch.tensor([True] * 8 + [False] * 2)

        torch.manual_seed(42)
        companions1 = select_companions_softmax(x, v, alive_mask, epsilon=1.0)

        torch.manual_seed(42)
        companions2 = select_companions_softmax(x, v, alive_mask, epsilon=1.0)

        assert torch.equal(companions1, companions2)

    def test_pairing_reproducible(self):
        """Same seed should give same pairing."""
        N, d = 10, 2
        torch.randn(N, d)
        torch.randn(N, d)
        alive_mask = torch.tensor([True] * N)

        torch.manual_seed(123)
        pairing1 = random_pairing_fisher_yates(alive_mask)

        torch.manual_seed(123)
        pairing2 = random_pairing_fisher_yates(alive_mask)

        assert torch.equal(pairing1, pairing2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
