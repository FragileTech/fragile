"""AIC parity tests for electroweak channel resolvers and helpers.

Verifies that the AIC copies (fragile.physics.aic.electroweak_channels and
fragile.physics.aic.multiscale_electroweak) produce identical outputs to the
originals (fragile.fractalai.qft.electroweak_channels and
fragile.fractalai.qft.multiscale_electroweak) when alive is all-True,
bounds=None, and pbc=False.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Old (canonical) imports — these still accept alive/bounds/pbc
# ---------------------------------------------------------------------------
from fragile.fractalai.qft.electroweak_channels import (
    _compute_d_alg_sq_for_companions,
    _resolve_companion_topology,
    _resolve_neighbor_method_strict,
    _resolve_su2_operator_mode,
    _resolve_walker_type_scope,
)
from fragile.fractalai.qft.multiscale_electroweak import (
    _resolve_su2_operator_mode as old_mew_resolve_su2,
    _resolve_walker_type_scope as old_mew_resolve_walker_scope,
    _select_best_scale as old_select_best_scale,
)

# ---------------------------------------------------------------------------
# New (AIC) imports — alive/bounds/pbc removed from signatures
# ---------------------------------------------------------------------------
from fragile.physics.aic.electroweak_channels import (
    _compute_d_alg_sq_for_companions as new_compute_d_alg_sq,
    _resolve_companion_topology as new_resolve_topology,
    _resolve_neighbor_method_strict as new_resolve_neighbor,
    _resolve_su2_operator_mode as new_resolve_su2,
    _resolve_walker_type_scope as new_resolve_walker_scope,
)
from fragile.physics.aic.multiscale_electroweak import (
    _resolve_su2_operator_mode as new_mew_resolve_su2,
    _resolve_walker_type_scope as new_mew_resolve_walker_scope,
    _select_best_scale as new_select_best_scale,
)
from tests.physics.aic.conftest import assert_tensor_or_nan_equal


# ===========================================================================
# Electroweak channel resolver tests
# ===========================================================================


class TestParityResolveNeighborMethod:
    """Parity tests for _resolve_neighbor_method_strict."""

    @pytest.mark.parametrize("mode", ["auto", "recorded", "companions", "uniform"])
    def test_valid_modes(self, mode: str) -> None:
        old = _resolve_neighbor_method_strict(mode)
        new = new_resolve_neighbor(mode)
        assert old == new, f"mode={mode!r}: old={old!r} vs new={new!r}"
        assert old == "companions"

    def test_voronoi_raises_both(self) -> None:
        with pytest.raises(ValueError, match="voronoi"):
            _resolve_neighbor_method_strict("voronoi")
        with pytest.raises(ValueError, match="voronoi"):
            new_resolve_neighbor("voronoi")

    def test_unknown_raises_both(self) -> None:
        with pytest.raises(ValueError):
            _resolve_neighbor_method_strict("unknown_method")
        with pytest.raises(ValueError):
            new_resolve_neighbor("unknown_method")


class TestParityResolveCompanionTopology:
    """Parity tests for _resolve_companion_topology."""

    @pytest.mark.parametrize("mode", ["distance", "clone", "both"])
    def test_valid_modes(self, mode: str) -> None:
        old = _resolve_companion_topology(mode)
        new = new_resolve_topology(mode)
        assert old == new, f"mode={mode!r}: old={old!r} vs new={new!r}"

    def test_invalid_raises_both(self) -> None:
        with pytest.raises(ValueError):
            _resolve_companion_topology("invalid")
        with pytest.raises(ValueError):
            new_resolve_topology("invalid")


class TestParityResolveSU2OperatorMode:
    """Parity tests for _resolve_su2_operator_mode (electroweak_channels)."""

    @pytest.mark.parametrize("mode", ["standard", "score_directed"])
    def test_valid_modes(self, mode: str) -> None:
        old = _resolve_su2_operator_mode(mode)
        new = new_resolve_su2(mode)
        assert old == new, f"mode={mode!r}: old={old!r} vs new={new!r}"

    def test_invalid_raises_both(self) -> None:
        with pytest.raises(ValueError):
            _resolve_su2_operator_mode("invalid")
        with pytest.raises(ValueError):
            new_resolve_su2("invalid")


class TestParityResolveWalkerTypeScope:
    """Parity tests for _resolve_walker_type_scope (electroweak_channels)."""

    def test_frame_global(self) -> None:
        old = _resolve_walker_type_scope("frame_global")
        new = new_resolve_walker_scope("frame_global")
        assert old == new
        assert old == "frame_global"

    def test_other_raises_both(self) -> None:
        with pytest.raises(ValueError):
            _resolve_walker_type_scope("other")
        with pytest.raises(ValueError):
            new_resolve_walker_scope("other")

    def test_empty_raises_both(self) -> None:
        with pytest.raises(ValueError):
            _resolve_walker_type_scope("per_walker")
        with pytest.raises(ValueError):
            new_resolve_walker_scope("per_walker")


# ===========================================================================
# Companion distance computation
# ===========================================================================


class TestParityCompanionDistances:
    """Parity tests for _compute_d_alg_sq_for_companions."""

    def test_d_alg_sq_parity(self) -> None:
        gen = torch.Generator().manual_seed(77)
        T, N, D = 5, 10, 3
        positions = torch.randn(T, N, D, generator=gen)
        velocities = torch.randn(T, N, D, generator=gen)
        alive = torch.ones(T, N, dtype=torch.bool)
        companions = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).clone()

        old_d_alg_sq, old_valid = _compute_d_alg_sq_for_companions(
            positions=positions,
            velocities=velocities,
            alive=alive,
            companions=companions,
            lambda_alg=0.0,
            bounds=None,
            pbc=False,
        )
        new_d_alg_sq, new_valid = new_compute_d_alg_sq(
            positions=positions,
            velocities=velocities,
            companions=companions,
            lambda_alg=0.0,
        )

        assert torch.equal(old_d_alg_sq, new_d_alg_sq), (
            f"d_alg_sq differs: max abs diff = "
            f"{(old_d_alg_sq - new_d_alg_sq).abs().max().item():.2e}"
        )
        assert torch.equal(old_valid, new_valid), "valid mask differs"

    def test_d_alg_sq_with_lambda(self) -> None:
        gen = torch.Generator().manual_seed(88)
        T, N, D = 5, 10, 3
        positions = torch.randn(T, N, D, generator=gen)
        velocities = torch.randn(T, N, D, generator=gen)
        alive = torch.ones(T, N, dtype=torch.bool)
        companions = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).clone()

        old_d_alg_sq, old_valid = _compute_d_alg_sq_for_companions(
            positions=positions,
            velocities=velocities,
            alive=alive,
            companions=companions,
            lambda_alg=0.5,
            bounds=None,
            pbc=False,
        )
        new_d_alg_sq, new_valid = new_compute_d_alg_sq(
            positions=positions,
            velocities=velocities,
            companions=companions,
            lambda_alg=0.5,
        )

        assert torch.equal(old_d_alg_sq, new_d_alg_sq), (
            f"d_alg_sq (lambda=0.5) differs: max abs diff = "
            f"{(old_d_alg_sq - new_d_alg_sq).abs().max().item():.2e}"
        )
        assert torch.equal(old_valid, new_valid), "valid mask (lambda=0.5) differs"

    def test_d_alg_sq_all_alive(self) -> None:
        """With all walkers alive, old (alive=all-True, bounds=None, pbc=False)
        must match new (no alive/bounds/pbc params)."""
        gen = torch.Generator().manual_seed(99)
        T, N, D = 5, 10, 3
        positions = torch.randn(T, N, D, generator=gen)
        velocities = torch.randn(T, N, D, generator=gen)
        alive = torch.ones(T, N, dtype=torch.bool)
        companions = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).clone()

        old_d_alg_sq, old_valid = _compute_d_alg_sq_for_companions(
            positions=positions,
            velocities=velocities,
            alive=alive,
            companions=companions,
            lambda_alg=0.0,
            bounds=None,
            pbc=False,
        )
        new_d_alg_sq, new_valid = new_compute_d_alg_sq(
            positions=positions,
            velocities=velocities,
            companions=companions,
            lambda_alg=0.0,
        )

        assert torch.equal(old_d_alg_sq, new_d_alg_sq), "d_alg_sq (all alive) differs"
        assert torch.equal(old_valid, new_valid), "valid mask (all alive) differs"


# ===========================================================================
# Multiscale electroweak resolver tests
# ===========================================================================


class TestParityMultiscaleEWResolveSU2:
    """Parity tests for _resolve_su2_operator_mode from multiscale_electroweak."""

    @pytest.mark.parametrize("mode", ["standard", "score_directed"])
    def test_valid_modes(self, mode: str) -> None:
        old = old_mew_resolve_su2(mode)
        new = new_mew_resolve_su2(mode)
        assert old == new, f"mode={mode!r}: old={old!r} vs new={new!r}"

    def test_invalid_raises_both(self) -> None:
        with pytest.raises(ValueError):
            old_mew_resolve_su2("invalid")
        with pytest.raises(ValueError):
            new_mew_resolve_su2("invalid")


class TestParityMultiscaleEWResolveWalkerScope:
    """Parity tests for _resolve_walker_type_scope from multiscale_electroweak."""

    def test_frame_global(self) -> None:
        old = old_mew_resolve_walker_scope("frame_global")
        new = new_mew_resolve_walker_scope("frame_global")
        assert old == new
        assert old == "frame_global"

    def test_other_raises_both(self) -> None:
        with pytest.raises(ValueError):
            old_mew_resolve_walker_scope("other")
        with pytest.raises(ValueError):
            new_mew_resolve_walker_scope("other")


# ===========================================================================
# Multiscale electroweak _select_best_scale tests
# ===========================================================================


class TestParityMultiscaleEWSelectBest:
    """Parity tests for _select_best_scale from multiscale_electroweak."""

    @staticmethod
    def _make_synthetic_results(
        n_scales: int = 6,
        seed: int = 42,
    ) -> list:
        """Build a list of mock ChannelCorrelatorResult objects with synthetic mass_fit data."""
        from fragile.fractalai.qft.correlator_channels import ChannelCorrelatorResult

        gen = torch.Generator().manual_seed(seed)
        results = []
        for s_idx in range(n_scales):
            mass = 0.1 + 0.05 * s_idx + 0.01 * torch.rand(1, generator=gen).item()
            mass_error = 0.005 + 0.002 * torch.rand(1, generator=gen).item()
            r_squared = 0.6 + 0.3 * torch.rand(1, generator=gen).item()
            aic_val = 10.0 - 2.0 * r_squared + torch.randn(1, generator=gen).item()
            n_valid_windows = 15 + int(5 * torch.rand(1, generator=gen).item())

            mass_fit = {
                "mass": float(mass),
                "mass_error": float(mass_error),
                "r_squared": float(r_squared),
                "n_valid_windows": n_valid_windows,
                "best_window": {"aic": float(aic_val), "width": 10},
                "scale": float(s_idx),
                "scale_index": s_idx,
            }

            series = torch.randn(50, generator=gen)
            correlator = torch.randn(40, generator=gen).abs()
            effective_mass = torch.randn(39, generator=gen).abs()

            result = ChannelCorrelatorResult(
                channel_name=f"test_ch_s{s_idx}",
                correlator=correlator,
                correlator_err=None,
                effective_mass=effective_mass,
                mass_fit=mass_fit,
                series=series,
                n_samples=50,
                dt=1.0,
            )
            results.append(result)
        return results

    def test_select_best_scale_parity(self) -> None:
        results = self._make_synthetic_results(n_scales=8, seed=42)
        old_idx = old_select_best_scale(
            results,
            min_r2=0.5,
            min_windows=10,
            max_error_pct=30.0,
            remove_artifacts=True,
        )
        new_idx = new_select_best_scale(
            results,
            min_r2=0.5,
            min_windows=10,
            max_error_pct=30.0,
            remove_artifacts=True,
        )
        assert old_idx == new_idx, f"best scale index: old={old_idx} vs new={new_idx}"

    def test_select_best_scale_no_valid(self) -> None:
        results = self._make_synthetic_results(n_scales=4, seed=99)
        # Use very strict filters that reject all candidates
        old_idx = old_select_best_scale(
            results,
            min_r2=0.999,
            min_windows=1000,
            max_error_pct=0.001,
            remove_artifacts=True,
        )
        new_idx = new_select_best_scale(
            results,
            min_r2=0.999,
            min_windows=1000,
            max_error_pct=0.001,
            remove_artifacts=True,
        )
        assert old_idx is None
        assert new_idx is None

    def test_select_best_scale_relaxed_filters(self) -> None:
        results = self._make_synthetic_results(n_scales=6, seed=55)
        old_idx = old_select_best_scale(
            results,
            min_r2=0.0,
            min_windows=0,
            max_error_pct=100.0,
            remove_artifacts=False,
        )
        new_idx = new_select_best_scale(
            results,
            min_r2=0.0,
            min_windows=0,
            max_error_pct=100.0,
            remove_artifacts=False,
        )
        assert old_idx == new_idx, f"relaxed filters: old={old_idx} vs new={new_idx}"

    def test_select_best_scale_empty(self) -> None:
        old_idx = old_select_best_scale([])
        new_idx = new_select_best_scale([])
        assert old_idx is None
        assert new_idx is None
