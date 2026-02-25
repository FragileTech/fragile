"""Tests for fragile.physics.electroweak.chirality module.

Covers walker classification, chirality labels, FFT autocorrelation,
and the high-level compute_chirality_autocorrelation / compute_lr_coupling
pipelines.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.electroweak.chirality import (
    ChiralityCorrelatorOutput,
    WalkerClassification,
    _fft_autocorrelation,
    classify_walkers,
    classify_walkers_vectorized,
    compute_chirality_autocorrelation,
    compute_lr_coupling,
)


# ---------------------------------------------------------------------------
# Mock history for pipeline tests
# ---------------------------------------------------------------------------


class MockChiralityHistory:
    """Minimal mock with the four attributes chirality functions require."""

    def __init__(self, T: int = 50, N: int = 10, seed: int = 42):
        gen = torch.Generator().manual_seed(seed)
        self.will_clone = torch.randint(0, 2, (T, N), generator=gen, dtype=torch.bool)
        self.companions_clone = torch.randint(0, N, (T, N), generator=gen, dtype=torch.long)
        self.fitness = torch.rand(T, N, generator=gen)
        self.alive_mask = torch.ones(T, N, dtype=torch.bool)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_inputs(T: int, N: int, seed: int = 0):
    """Return (will_clone, companions_clone, fitness, alive) tensors."""
    gen = torch.Generator().manual_seed(seed)
    will_clone = torch.randint(0, 2, (T, N), generator=gen, dtype=torch.bool)
    companions_clone = torch.randint(0, N, (T, N), generator=gen, dtype=torch.long)
    fitness = torch.rand(T, N, generator=gen)
    alive = torch.ones(T, N, dtype=torch.bool)
    return will_clone, companions_clone, fitness, alive


def _all_pairs_disjoint(cls: WalkerClassification, alive: Tensor) -> None:
    """Assert all four masks are pairwise disjoint for alive walkers."""
    masks = [cls.delta, cls.strong_resister, cls.weak_resister, cls.persister]
    names = ["delta", "strong_resister", "weak_resister", "persister"]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            overlap = masks[i] & masks[j]
            assert not overlap.any(), (
                f"{names[i]} and {names[j]} overlap at "
                f"{overlap.nonzero(as_tuple=False).tolist()}"
            )


# ===========================================================================
# 1. Classification exhaustiveness
# ===========================================================================


class TestClassificationExhaustiveness:
    """Every alive walker belongs to exactly one of the four classes."""

    @pytest.mark.parametrize("seed", [0, 7, 42, 123])
    def test_random_inputs_loop(self, seed: int) -> None:
        will_clone, companions_clone, fitness, alive = _random_inputs(10, 20, seed)
        cls = classify_walkers(will_clone, companions_clone, fitness, alive)

        total = cls.delta | cls.strong_resister | cls.weak_resister | cls.persister
        assert (total == alive).all(), "Not all alive walkers classified"
        _all_pairs_disjoint(cls, alive)

    @pytest.mark.parametrize("seed", [0, 7, 42, 123])
    def test_random_inputs_vectorized(self, seed: int) -> None:
        will_clone, companions_clone, fitness, alive = _random_inputs(10, 20, seed)
        cls = classify_walkers_vectorized(will_clone, companions_clone, fitness, alive)

        total = cls.delta | cls.strong_resister | cls.weak_resister | cls.persister
        assert (total == alive).all(), "Not all alive walkers classified"
        _all_pairs_disjoint(cls, alive)

    def test_dead_walkers_excluded(self) -> None:
        """Dead walkers should not appear in any mask."""
        T, N = 5, 8
        will_clone, companions_clone, fitness, _ = _random_inputs(T, N, seed=99)
        alive = torch.ones(T, N, dtype=torch.bool)
        alive[:, 0] = False  # walker 0 is dead in all frames

        cls = classify_walkers(will_clone, companions_clone, fitness, alive)
        for mask in [cls.delta, cls.strong_resister, cls.weak_resister, cls.persister]:
            assert not mask[:, 0].any(), "Dead walker appears in a classification mask"


# ===========================================================================
# 2. Classification correctness with known input
# ===========================================================================


class TestClassificationCorrectness:
    """Verify classification on a hand-crafted example."""

    def _build_known_input(self):
        T, N = 1, 4
        # Walker 0: will_clone=True -> delta
        will_clone = torch.tensor([[True, False, False, False]])
        # Walker 0 targets walker 2 -> walker 2 is strong_resister
        companions_clone = torch.tensor([[2, 0, 1, 3]], dtype=torch.long)
        # Fitness: walker 0 = 0.1, walker 1 = 0.5, walker 2 = 0.3, walker 3 = 0.2
        fitness = torch.tensor([[0.1, 0.5, 0.3, 0.2]])
        alive = torch.ones(T, N, dtype=torch.bool)
        return will_clone, companions_clone, fitness, alive

    def test_delta(self) -> None:
        will_clone, comp, fit, alive = self._build_known_input()
        cls = classify_walkers(will_clone, comp, fit, alive)
        assert cls.delta[0, 0].item() is True

    def test_strong_resister(self) -> None:
        will_clone, comp, fit, alive = self._build_known_input()
        cls = classify_walkers(will_clone, comp, fit, alive)
        # Walker 2 is the target of walker 0 (delta)
        assert cls.strong_resister[0, 2].item() is True

    def test_walker1_classification(self) -> None:
        """Walker 1: not cloning, not targeted, companion is walker 0 (fitness 0.1 < 0.5) -> no fitter peer -> persister."""
        will_clone, comp, fit, alive = self._build_known_input()
        cls = classify_walkers(will_clone, comp, fit, alive)
        # Walker 1's companion is walker 0 with fitness 0.1 < walker 1's fitness 0.5
        # So companion is NOT fitter -> persister
        assert cls.persister[0, 1].item() is True

    def test_walker3_classification(self) -> None:
        """Walker 3: not cloning, not targeted, companion is self (fitness equal) -> persister."""
        will_clone, comp, fit, alive = self._build_known_input()
        cls = classify_walkers(will_clone, comp, fit, alive)
        # Walker 3's companion is walker 3 (self) -> same fitness -> not fitter -> persister
        assert cls.persister[0, 3].item() is True


# ===========================================================================
# 3. Chirality labels
# ===========================================================================


class TestChiralityLabels:
    """chi = +1 for L (D+SR), -1 for R (WR+P), 0 for dead."""

    def test_left_handed_positive(self) -> None:
        will_clone, comp, fit, alive = _random_inputs(8, 12, seed=77)
        cls = classify_walkers(will_clone, comp, fit, alive)
        left = cls.delta | cls.strong_resister
        assert (cls.chi[left] == 1.0).all()

    def test_right_handed_negative(self) -> None:
        will_clone, comp, fit, alive = _random_inputs(8, 12, seed=77)
        cls = classify_walkers(will_clone, comp, fit, alive)
        right = cls.weak_resister | cls.persister
        assert (cls.chi[right] == -1.0).all()

    def test_dead_walkers_zero(self) -> None:
        T, N = 5, 6
        will_clone, comp, fit, _ = _random_inputs(T, N, seed=55)
        alive = torch.ones(T, N, dtype=torch.bool)
        alive[:, 0] = False
        cls = classify_walkers(will_clone, comp, fit, alive)
        assert (cls.chi[:, 0] == 0.0).all()


# ===========================================================================
# 4. Loop vs vectorized equivalence
# ===========================================================================


class TestLoopVsVectorized:
    """classify_walkers and classify_walkers_vectorized produce identical results."""

    @pytest.mark.parametrize("seed", [0, 13, 42, 100])
    def test_equivalence(self, seed: int) -> None:
        will_clone, comp, fit, alive = _random_inputs(15, 25, seed)
        cls_loop = classify_walkers(will_clone, comp, fit, alive)
        cls_vec = classify_walkers_vectorized(will_clone, comp, fit, alive)

        assert torch.equal(cls_loop.delta, cls_vec.delta)
        assert torch.equal(cls_loop.strong_resister, cls_vec.strong_resister)
        assert torch.equal(cls_loop.weak_resister, cls_vec.weak_resister)
        assert torch.equal(cls_loop.persister, cls_vec.persister)
        assert torch.allclose(cls_loop.chi, cls_vec.chi)

    def test_equivalence_with_dead_walkers(self) -> None:
        T, N = 10, 15
        will_clone, comp, fit, _ = _random_inputs(T, N, seed=88)
        alive = torch.ones(T, N, dtype=torch.bool)
        alive[:, ::3] = False  # kill every 3rd walker

        cls_loop = classify_walkers(will_clone, comp, fit, alive)
        cls_vec = classify_walkers_vectorized(will_clone, comp, fit, alive)

        assert torch.equal(cls_loop.delta, cls_vec.delta)
        assert torch.equal(cls_loop.strong_resister, cls_vec.strong_resister)
        assert torch.equal(cls_loop.weak_resister, cls_vec.weak_resister)
        assert torch.equal(cls_loop.persister, cls_vec.persister)
        assert torch.allclose(cls_loop.chi, cls_vec.chi)


# ===========================================================================
# 5. FFT autocorrelation
# ===========================================================================


class TestFFTAutocorrelation:
    """Test _fft_autocorrelation with known signals."""

    def test_constant_signal(self) -> None:
        """Autocorrelation of a constant is constant (normalized -> all 1s)."""
        series = torch.ones(100)
        result = _fft_autocorrelation(series, max_lag=20, normalize=True)
        assert result.shape == (21,)
        assert torch.allclose(result, torch.ones(21), atol=1e-5)

    def test_autocorrelation_shape(self) -> None:
        series = torch.randn(200)
        result = _fft_autocorrelation(series, max_lag=50, normalize=True)
        assert result.shape == (51,)

    def test_normalized_starts_at_one(self) -> None:
        series = torch.randn(100)
        result = _fft_autocorrelation(series, max_lag=30, normalize=True)
        assert abs(result[0].item() - 1.0) < 1e-6

    def test_sinusoid_has_periodic_autocorrelation(self) -> None:
        """A sinusoid's autocorrelation should also be periodic."""
        T = 200
        period = 20
        t = torch.arange(T, dtype=torch.float32)
        series = torch.sin(2 * 3.14159265 * t / period)
        result = _fft_autocorrelation(series, max_lag=60, normalize=True)

        # At lag = period, autocorrelation should be close to 1
        assert result[period].item() > 0.9

        # At lag = period/2, autocorrelation should be close to -1
        assert result[period // 2].item() < -0.8

    def test_max_lag_exceeds_length(self) -> None:
        """When max_lag > T-1, result is zero-padded."""
        series = torch.randn(10)
        result = _fft_autocorrelation(series, max_lag=20, normalize=True)
        assert result.shape == (21,)
        # Tail should be zero-padded
        assert (result[10:] == 0.0).all()

    def test_unnormalized(self) -> None:
        """Without normalization, C(0) equals the mean squared value."""
        gen = torch.Generator().manual_seed(42)
        series = torch.randn(100, generator=gen)
        result = _fft_autocorrelation(series, max_lag=10, normalize=False)
        # C(0) = (1/T) sum(x_i^2)
        expected_c0 = (series**2).mean()
        assert abs(result[0].item() - expected_c0.item()) < 1e-4


# ===========================================================================
# 6. Chirality autocorrelation pipeline
# ===========================================================================


class TestChiralityAutocorrelation:
    """Test compute_chirality_autocorrelation with mock history."""

    def test_returns_valid_output(self) -> None:
        history = MockChiralityHistory(T=50, N=10, seed=42)
        out = compute_chirality_autocorrelation(history, max_lag=20, warmup_fraction=0.0)

        assert isinstance(out, ChiralityCorrelatorOutput)
        assert out.c_chi.shape == (21,)
        assert out.c_chi_connected.shape == (21,)
        assert isinstance(out.fermion_mass, float)
        assert isinstance(out.fermion_mass_err, float)
        assert isinstance(out.classification, WalkerClassification)
        assert out.n_cloning_frames > 0

    def test_per_walker_option(self) -> None:
        history = MockChiralityHistory(T=50, N=10, seed=42)
        out = compute_chirality_autocorrelation(
            history, max_lag=20, warmup_fraction=0.0, per_walker=True,
        )
        assert out.c_chi_per_walker is not None
        assert out.c_chi_per_walker.shape[0] == 10  # N walkers
        assert out.c_chi_per_walker.shape[1] == 21  # max_lag + 1

    def test_no_cloning_returns_empty(self) -> None:
        """When no cloning occurs, pipeline returns zeros gracefully."""
        history = MockChiralityHistory(T=20, N=5, seed=42)
        history.will_clone = torch.zeros(20, 5, dtype=torch.bool)
        out = compute_chirality_autocorrelation(
            history, max_lag=10, warmup_fraction=0.0,
        )
        assert out.n_cloning_frames == 0
        assert (out.c_chi == 0).all()

    def test_warmup_fraction(self) -> None:
        """With warmup_fraction=0.5, half the frames are discarded."""
        history = MockChiralityHistory(T=100, N=10, seed=42)
        out_full = compute_chirality_autocorrelation(
            history, max_lag=20, warmup_fraction=0.0,
        )
        out_half = compute_chirality_autocorrelation(
            history, max_lag=20, warmup_fraction=0.5,
        )
        # Different warmup should give different n_cloning_frames
        assert out_half.n_cloning_frames <= out_full.n_cloning_frames

    def test_cloning_frames_only_false(self) -> None:
        history = MockChiralityHistory(T=50, N=10, seed=42)
        out = compute_chirality_autocorrelation(
            history, max_lag=20, warmup_fraction=0.0, cloning_frames_only=False,
        )
        assert isinstance(out, ChiralityCorrelatorOutput)
        # frame_mask should be all True
        assert out.frame_mask.all()


# ===========================================================================
# 7. L-R coupling
# ===========================================================================


class TestLRCoupling:
    """Test compute_lr_coupling with mock history."""

    def test_returns_expected_keys(self) -> None:
        history = MockChiralityHistory(T=50, N=10, seed=42)
        result = compute_lr_coupling(history, warmup_fraction=0.0)

        assert "lr_coupling_magnitude" in result
        assert "lr_coupling_phase" in result
        assert "lr_coupling_complex" in result
        assert "lr_fraction" in result

    def test_output_shapes(self) -> None:
        history = MockChiralityHistory(T=50, N=10, seed=42)
        result = compute_lr_coupling(history, warmup_fraction=0.0)

        T_eff = 50  # warmup_fraction=0.0 -> all frames
        assert result["lr_coupling_magnitude"].shape == (T_eff,)
        assert result["lr_coupling_phase"].shape == (T_eff,)
        assert result["lr_coupling_complex"].shape == (T_eff,)
        assert result["lr_fraction"].shape == (T_eff,)

    def test_magnitude_non_negative(self) -> None:
        history = MockChiralityHistory(T=50, N=10, seed=42)
        result = compute_lr_coupling(history, warmup_fraction=0.0)
        assert (result["lr_coupling_magnitude"] >= 0).all()

    def test_lr_fraction_bounded(self) -> None:
        history = MockChiralityHistory(T=50, N=10, seed=42)
        result = compute_lr_coupling(history, warmup_fraction=0.0)
        assert (result["lr_fraction"] >= 0).all()
        assert (result["lr_fraction"] <= 1.0 + 1e-6).all()


# ===========================================================================
# 8. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases: no cloning, all cloning, single walker, etc."""

    def test_no_cloning_all_persisters(self) -> None:
        """When will_clone is all False, all alive walkers are persisters with chi=-1."""
        T, N = 5, 8
        will_clone = torch.zeros(T, N, dtype=torch.bool)
        companions_clone = torch.zeros(T, N, dtype=torch.long)
        fitness = torch.ones(T, N)
        alive = torch.ones(T, N, dtype=torch.bool)

        cls = classify_walkers(will_clone, companions_clone, fitness, alive)
        assert cls.delta.sum() == 0
        assert cls.strong_resister.sum() == 0
        assert cls.weak_resister.sum() == 0
        assert (cls.persister == alive).all()
        assert (cls.chi == -1.0).all()

    def test_all_cloning_all_deltas(self) -> None:
        """When will_clone is all True, all alive walkers are deltas with chi=+1."""
        T, N = 5, 8
        will_clone = torch.ones(T, N, dtype=torch.bool)
        companions_clone = torch.zeros(T, N, dtype=torch.long)
        fitness = torch.ones(T, N)
        alive = torch.ones(T, N, dtype=torch.bool)

        cls = classify_walkers(will_clone, companions_clone, fitness, alive)
        # All walkers are deltas, but they also target walker 0, making walker 0
        # a target. However, walker 0 is also cloning, so it's still a delta.
        # Delta = will_clone & alive, strong_resister = ~will_clone & alive & is_target
        # Since all will_clone=True, strong_resister is empty.
        assert (cls.delta == alive).all()
        assert cls.strong_resister.sum() == 0
        assert cls.weak_resister.sum() == 0
        assert cls.persister.sum() == 0
        assert (cls.chi == 1.0).all()

    def test_single_walker(self) -> None:
        """Single walker should be classified correctly."""
        T, N = 5, 1
        will_clone = torch.zeros(T, N, dtype=torch.bool)
        companions_clone = torch.zeros(T, N, dtype=torch.long)
        fitness = torch.ones(T, N)
        alive = torch.ones(T, N, dtype=torch.bool)

        cls = classify_walkers(will_clone, companions_clone, fitness, alive)
        assert (cls.persister == alive).all()

    def test_all_dead(self) -> None:
        """All dead walkers: no classification, chi = 0 everywhere."""
        T, N = 5, 4
        will_clone = torch.ones(T, N, dtype=torch.bool)
        companions_clone = torch.zeros(T, N, dtype=torch.long)
        fitness = torch.ones(T, N)
        alive = torch.zeros(T, N, dtype=torch.bool)

        cls = classify_walkers(will_clone, companions_clone, fitness, alive)
        assert cls.delta.sum() == 0
        assert cls.strong_resister.sum() == 0
        assert cls.weak_resister.sum() == 0
        assert cls.persister.sum() == 0
        assert (cls.chi == 0.0).all()

    def test_alive_none_defaults_to_all_alive(self) -> None:
        """When alive=None, all walkers should be treated as alive."""
        T, N = 5, 6
        gen = torch.Generator().manual_seed(33)
        will_clone = torch.randint(0, 2, (T, N), generator=gen, dtype=torch.bool)
        companions_clone = torch.randint(0, N, (T, N), generator=gen, dtype=torch.long)
        fitness = torch.rand(T, N, generator=gen)

        cls_none = classify_walkers(will_clone, companions_clone, fitness, alive=None)
        cls_ones = classify_walkers(
            will_clone, companions_clone, fitness,
            alive=torch.ones(T, N, dtype=torch.bool),
        )
        assert torch.equal(cls_none.chi, cls_ones.chi)

    def test_counts_match_masks(self) -> None:
        """Per-frame counts should match mask sums."""
        will_clone, comp, fit, alive = _random_inputs(10, 20, seed=77)
        cls = classify_walkers(will_clone, comp, fit, alive)

        assert torch.equal(cls.n_delta, cls.delta.sum(dim=1))
        assert torch.equal(cls.n_strong_resister, cls.strong_resister.sum(dim=1))
        assert torch.equal(cls.n_weak_resister, cls.weak_resister.sum(dim=1))
        assert torch.equal(cls.n_persister, cls.persister.sum(dim=1))

    def test_left_right_handed_properties(self) -> None:
        """WalkerClassification.left_handed and .right_handed are correct."""
        will_clone, comp, fit, alive = _random_inputs(8, 12, seed=55)
        cls = classify_walkers(will_clone, comp, fit, alive)

        assert torch.equal(cls.left_handed, cls.delta | cls.strong_resister)
        assert torch.equal(cls.right_handed, cls.weak_resister | cls.persister)

    def test_n_frames_and_n_walkers(self) -> None:
        T, N = 7, 13
        will_clone, comp, fit, alive = _random_inputs(T, N, seed=11)
        cls = classify_walkers(will_clone, comp, fit, alive)
        assert cls.n_frames == T
        assert cls.n_walkers == N
