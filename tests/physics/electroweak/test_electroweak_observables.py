"""Parity tests: fragile.fractalai.qft.electroweak_observables vs fragile.physics.electroweak.electroweak_observables.

Each test imports from BOTH locations and asserts exact equality.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fragile.fractalai.qft.electroweak_observables import (
    antisymmetric_singular_values as old_antisymmetric_singular_values,
    build_phase_space_antisymmetric_kernel as old_build_phase_space_antisymmetric_kernel,
    classify_walker_types as old_classify_walker_types,
    compute_fitness_gap_distribution as old_compute_fitness_gap_distribution,
    compute_higgs_vev_from_positions as old_compute_higgs_vev_from_positions,
    compute_weighted_electroweak_ops_vectorized as old_compute_ops,
    ELECTROWEAK_OPERATOR_CHANNELS as old_ELECTROWEAK_OPERATOR_CHANNELS,
    pack_neighbor_lists as old_pack_neighbor_lists,
    pack_neighbors_from_edges as old_pack_neighbors_from_edges,
    PARITY_VELOCITY_CHANNELS as old_PARITY_VELOCITY_CHANNELS,
    predict_yukawa_mass_from_fitness as old_predict_yukawa_mass_from_fitness,
    SU2_BASE_OPERATOR_CHANNELS as old_SU2_BASE_OPERATOR_CHANNELS,
    SU2_DIRECTIONAL_OPERATOR_CHANNELS as old_SU2_DIRECTIONAL_OPERATOR_CHANNELS,
    SU2_WALKER_TYPE_CHANNELS as old_SU2_WALKER_TYPE_CHANNELS,
    SYMMETRY_BREAKING_CHANNELS as old_SYMMETRY_BREAKING_CHANNELS,
    WALKER_TYPE_LABELS as old_WALKER_TYPE_LABELS,
)
from fragile.physics.electroweak.electroweak_observables import (
    antisymmetric_singular_values as new_antisymmetric_singular_values,
    build_phase_space_antisymmetric_kernel as new_build_phase_space_antisymmetric_kernel,
    classify_walker_types as new_classify_walker_types,
    compute_fitness_gap_distribution as new_compute_fitness_gap_distribution,
    compute_higgs_vev_from_positions as new_compute_higgs_vev_from_positions,
    compute_weighted_electroweak_ops_vectorized as new_compute_ops,
    ELECTROWEAK_OPERATOR_CHANNELS as new_ELECTROWEAK_OPERATOR_CHANNELS,
    pack_neighbor_lists as new_pack_neighbor_lists,
    pack_neighbors_from_edges as new_pack_neighbors_from_edges,
    PackedNeighbors,
    PARITY_VELOCITY_CHANNELS as new_PARITY_VELOCITY_CHANNELS,
    predict_yukawa_mass_from_fitness as new_predict_yukawa_mass_from_fitness,
    SU2_BASE_OPERATOR_CHANNELS as new_SU2_BASE_OPERATOR_CHANNELS,
    SU2_DIRECTIONAL_OPERATOR_CHANNELS as new_SU2_DIRECTIONAL_OPERATOR_CHANNELS,
    SU2_WALKER_TYPE_CHANNELS as new_SU2_WALKER_TYPE_CHANNELS,
    SYMMETRY_BREAKING_CHANNELS as new_SYMMETRY_BREAKING_CHANNELS,
    WALKER_TYPE_LABELS as new_WALKER_TYPE_LABELS,
)
from tests.physics.electroweak.conftest import (
    assert_dict_floats_equal,
    assert_dict_tensors_equal,
)


# ===================================================================
# Constants parity
# ===================================================================


class TestParityConstants:
    def test_electroweak_operator_channels(self):
        assert old_ELECTROWEAK_OPERATOR_CHANNELS == new_ELECTROWEAK_OPERATOR_CHANNELS

    def test_su2_base_operator_channels(self):
        assert old_SU2_BASE_OPERATOR_CHANNELS == new_SU2_BASE_OPERATOR_CHANNELS

    def test_su2_directional_operator_channels(self):
        assert old_SU2_DIRECTIONAL_OPERATOR_CHANNELS == new_SU2_DIRECTIONAL_OPERATOR_CHANNELS

    def test_walker_type_labels(self):
        assert old_WALKER_TYPE_LABELS == new_WALKER_TYPE_LABELS

    def test_su2_walker_type_channels(self):
        assert old_SU2_WALKER_TYPE_CHANNELS == new_SU2_WALKER_TYPE_CHANNELS

    def test_symmetry_breaking_channels(self):
        assert old_SYMMETRY_BREAKING_CHANNELS == new_SYMMETRY_BREAKING_CHANNELS

    def test_parity_velocity_channels(self):
        assert old_PARITY_VELOCITY_CHANNELS == new_PARITY_VELOCITY_CHANNELS


# ===================================================================
# classify_walker_types parity
# ===================================================================


class TestParityClassifyWalkerTypes:
    def test_basic(self, tiny_fitness, tiny_alive):
        old = old_classify_walker_types(tiny_fitness, tiny_alive)
        new = new_classify_walker_types(tiny_fitness, tiny_alive)
        for o, n in zip(old, new):
            assert torch.equal(o, n)

    def test_with_will_clone(self, tiny_fitness, tiny_alive, tiny_will_clone):
        old = old_classify_walker_types(tiny_fitness, tiny_alive, tiny_will_clone)
        new = new_classify_walker_types(tiny_fitness, tiny_alive, tiny_will_clone)
        for o, n in zip(old, new):
            assert torch.equal(o, n)

    def test_partial_alive(self, tiny_fitness):
        alive = torch.tensor([True, False, True])
        old = old_classify_walker_types(tiny_fitness, alive)
        new = new_classify_walker_types(tiny_fitness, alive)
        for o, n in zip(old, new):
            assert torch.equal(o, n)


# ===================================================================
# pack_neighbor_lists parity
# ===================================================================


class TestParityPackNeighborLists:
    def test_basic(self):
        idx = [torch.tensor([1, 2]), torch.tensor([0]), torch.tensor([0, 1, 2])]
        weights = [torch.tensor([0.5, 0.5]), torch.tensor([1.0]), torch.tensor([0.3, 0.3, 0.4])]
        kwargs = {"n_walkers": 3, "device": torch.device("cpu"), "dtype": torch.float32}
        old = old_pack_neighbor_lists(idx, weights, **kwargs)
        new = new_pack_neighbor_lists(idx, weights, **kwargs)
        assert torch.equal(old.indices, new.indices)
        assert torch.equal(old.weights, new.weights)
        assert torch.equal(old.valid, new.valid)

    def test_empty(self):
        kwargs = {"n_walkers": 0, "device": torch.device("cpu"), "dtype": torch.float32}
        old = old_pack_neighbor_lists([], [], **kwargs)
        new = new_pack_neighbor_lists([], [], **kwargs)
        assert torch.equal(old.indices, new.indices)
        assert torch.equal(old.weights, new.weights)
        assert torch.equal(old.valid, new.valid)


# ===================================================================
# pack_neighbors_from_edges parity
# ===================================================================


class TestParityPackNeighborsFromEdges:
    def _make_edges(self, seed=42):
        gen = torch.Generator().manual_seed(seed)
        N = 6
        E = 12
        src = torch.randint(0, N, (E,), generator=gen)
        dst = torch.randint(0, N, (E,), generator=gen)
        edges = torch.stack([src, dst], dim=1)
        weights = torch.rand(E, generator=gen)
        alive = torch.ones(N, dtype=torch.bool)
        return edges, weights, alive, N

    def test_basic(self):
        edges, weights, alive, N = self._make_edges()
        kwargs = {
            "edges": edges,
            "edge_weights": weights,
            "alive": alive,
            "n_walkers": N,
            "device": torch.device("cpu"),
            "dtype": torch.float32,
        }
        old = old_pack_neighbors_from_edges(**kwargs)
        new = new_pack_neighbors_from_edges(**kwargs)
        assert torch.equal(old.indices, new.indices)
        assert torch.equal(old.weights, new.weights)
        assert torch.equal(old.valid, new.valid)

    def test_max_neighbors(self):
        edges, weights, alive, N = self._make_edges()
        kwargs = {
            "edges": edges,
            "edge_weights": weights,
            "alive": alive,
            "n_walkers": N,
            "max_neighbors": 2,
            "device": torch.device("cpu"),
            "dtype": torch.float32,
        }
        old = old_pack_neighbors_from_edges(**kwargs)
        new = new_pack_neighbors_from_edges(**kwargs)
        assert torch.equal(old.indices, new.indices)
        assert torch.equal(old.weights, new.weights)
        assert torch.equal(old.valid, new.valid)

    def test_empty_edges(self):
        kwargs = {
            "edges": torch.zeros((0, 2), dtype=torch.long),
            "edge_weights": torch.zeros(0),
            "alive": torch.ones(5, dtype=torch.bool),
            "n_walkers": 5,
            "device": torch.device("cpu"),
            "dtype": torch.float32,
        }
        old = old_pack_neighbors_from_edges(**kwargs)
        new = new_pack_neighbors_from_edges(**kwargs)
        assert torch.equal(old.indices, new.indices)
        assert torch.equal(old.weights, new.weights)
        assert torch.equal(old.valid, new.valid)


# ===================================================================
# compute_weighted_electroweak_ops_vectorized parity
# ===================================================================


def _make_packed_neighbors(N, seed=50):
    """Build simple cyclic PackedNeighbors for N walkers."""
    src = torch.arange(N, dtype=torch.long)
    dst = src.roll(-1)
    edges = torch.stack([src, dst], dim=1)
    weights = torch.ones(N, dtype=torch.float32)
    alive = torch.ones(N, dtype=torch.bool)
    from fragile.physics.electroweak.electroweak_observables import (
        pack_neighbors_from_edges as _pnfe,
    )

    return _pnfe(
        edges=edges,
        edge_weights=weights,
        alive=alive,
        n_walkers=N,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _base_ops_kwargs(N=10, d=3, seed=77):
    gen = torch.Generator().manual_seed(seed)
    positions = torch.randn(N, d, generator=gen)
    velocities = torch.randn(N, d, generator=gen)
    fitness = torch.rand(N, generator=gen).clamp(min=1e-6)
    alive = torch.ones(N, dtype=torch.bool)
    neighbors = _make_packed_neighbors(N)
    return {
        "positions": positions,
        "velocities": velocities,
        "fitness": fitness,
        "alive": alive,
        "neighbors": neighbors,
        "h_eff": 1.0,
        "epsilon_d": 1.0,
        "epsilon_c": 1.0,
        "epsilon_clone": 1e-8,
        "lambda_alg": 0.0,
    }


class TestParityComputeOps:
    def test_standard_mode(self):
        kwargs = _base_ops_kwargs()
        old = old_compute_ops(**kwargs, su2_operator_mode="standard")
        new = new_compute_ops(**kwargs, su2_operator_mode="standard")
        assert_dict_tensors_equal(old, new)

    def test_score_directed_mode(self):
        kwargs = _base_ops_kwargs()
        old = old_compute_ops(**kwargs, su2_operator_mode="score_directed")
        new = new_compute_ops(**kwargs, su2_operator_mode="score_directed")
        assert_dict_tensors_equal(old, new)

    def test_walker_type_split_disabled(self):
        kwargs = _base_ops_kwargs()
        old = old_compute_ops(**kwargs, enable_walker_type_split=False)
        new = new_compute_ops(**kwargs, enable_walker_type_split=False)
        assert_dict_tensors_equal(old, new)

    def test_walker_type_split_enabled(self):
        kwargs = _base_ops_kwargs()
        gen = torch.Generator().manual_seed(88)
        N = kwargs["positions"].shape[0]
        will_clone = torch.randint(0, 2, (N,), generator=gen, dtype=torch.bool)
        old = old_compute_ops(**kwargs, will_clone=will_clone, enable_walker_type_split=True)
        new = new_compute_ops(**kwargs, will_clone=will_clone, enable_walker_type_split=True)
        assert_dict_tensors_equal(old, new)

    def test_separate_neighbor_families(self):
        N = 10
        kwargs = _base_ops_kwargs(N=N)
        kwargs.pop("neighbors")
        nbr_u1 = _make_packed_neighbors(N, seed=51)
        nbr_su2 = _make_packed_neighbors(N, seed=52)
        nbr_ew = _make_packed_neighbors(N, seed=53)
        old = old_compute_ops(
            **kwargs,
            neighbors_u1=nbr_u1,
            neighbors_su2=nbr_su2,
            neighbors_ew_mixed=nbr_ew,
        )
        new = new_compute_ops(
            **kwargs,
            neighbors_u1=nbr_u1,
            neighbors_su2=nbr_su2,
            neighbors_ew_mixed=nbr_ew,
        )
        assert_dict_tensors_equal(old, new)


# ===================================================================
# compute_higgs_vev_from_positions parity
# ===================================================================


class TestParityHiggsVev:
    def test_basic(self, tiny_positions):
        pos = tiny_positions[0]
        old = old_compute_higgs_vev_from_positions(pos)
        new = new_compute_higgs_vev_from_positions(pos)
        assert torch.equal(old["center"], new["center"])
        assert torch.equal(old["radii"], new["radii"])
        assert old["vev"] == new["vev"]
        assert old["vev_std"] == new["vev_std"]

    def test_with_alive_mask(self, tiny_positions, tiny_alive):
        pos = tiny_positions[0]
        old = old_compute_higgs_vev_from_positions(pos, alive=tiny_alive)
        new = new_compute_higgs_vev_from_positions(pos, alive=tiny_alive)
        assert torch.equal(old["center"], new["center"])
        assert torch.equal(old["radii"], new["radii"])
        assert old["vev"] == new["vev"]
        assert old["vev_std"] == new["vev_std"]


# ===================================================================
# compute_fitness_gap_distribution parity
# ===================================================================


class TestParityFitnessGapDistribution:
    def test_basic(self, tiny_fitness):
        old = old_compute_fitness_gap_distribution(tiny_fitness)
        new = new_compute_fitness_gap_distribution(tiny_fitness)
        assert torch.equal(old["fitness_sorted"], new["fitness_sorted"])
        assert torch.equal(old["gaps"], new["gaps"])
        assert old["phi0"] == new["phi0"]

    def test_with_alive_mask(self, tiny_fitness, tiny_alive):
        old = old_compute_fitness_gap_distribution(tiny_fitness, alive=tiny_alive)
        new = new_compute_fitness_gap_distribution(tiny_fitness, alive=tiny_alive)
        assert torch.equal(old["fitness_sorted"], new["fitness_sorted"])
        assert torch.equal(old["gaps"], new["gaps"])
        assert old["phi0"] == new["phi0"]


# ===================================================================
# predict_yukawa_mass_from_fitness parity
# ===================================================================


class TestParityYukawaMass:
    def test_basic(self, tiny_fitness):
        old = old_predict_yukawa_mass_from_fitness(v_higgs=246.0, fitness=tiny_fitness)
        new = new_predict_yukawa_mass_from_fitness(v_higgs=246.0, fitness=tiny_fitness)
        assert_dict_floats_equal(old, new)

    def test_custom_phi0(self, tiny_fitness):
        old = old_predict_yukawa_mass_from_fitness(
            v_higgs=246.0,
            fitness=tiny_fitness,
            phi0=0.5,
            y0=2.0,
        )
        new = new_predict_yukawa_mass_from_fitness(
            v_higgs=246.0,
            fitness=tiny_fitness,
            phi0=0.5,
            y0=2.0,
        )
        assert_dict_floats_equal(old, new)


# ===================================================================
# build_phase_space_antisymmetric_kernel parity
# ===================================================================


class TestParityAntisymmetricKernel:
    def _base_kwargs(self, N=8, d=3, seed=33):
        gen = torch.Generator().manual_seed(seed)
        return {
            "positions": torch.randn(N, d, generator=gen),
            "velocities": torch.randn(N, d, generator=gen),
            "fitness": torch.rand(N, generator=gen).clamp(min=1e-6),
            "alive_mask": torch.ones(N, dtype=torch.bool),
            "epsilon_c": 1.0,
        }

    def test_basic(self):
        kwargs = self._base_kwargs()
        ok, oi = old_build_phase_space_antisymmetric_kernel(**kwargs)
        nk, ni = new_build_phase_space_antisymmetric_kernel(**kwargs)
        np.testing.assert_array_equal(ok, nk)
        np.testing.assert_array_equal(oi, ni)

    def test_no_phase(self):
        kwargs = self._base_kwargs()
        ok, oi = old_build_phase_space_antisymmetric_kernel(**kwargs, include_phase=False)
        nk, ni = new_build_phase_space_antisymmetric_kernel(**kwargs, include_phase=False)
        np.testing.assert_array_equal(ok, nk)
        np.testing.assert_array_equal(oi, ni)

    def test_numpy_inputs(self):
        gen = torch.Generator().manual_seed(34)
        N, d = 6, 3
        kwargs = {
            "positions": torch.randn(N, d, generator=gen).numpy(),
            "velocities": torch.randn(N, d, generator=gen).numpy(),
            "fitness": torch.rand(N, generator=gen).clamp(min=1e-6).numpy(),
            "alive_mask": np.ones(N, dtype=bool),
            "epsilon_c": 1.0,
        }
        ok, oi = old_build_phase_space_antisymmetric_kernel(**kwargs)
        nk, ni = new_build_phase_space_antisymmetric_kernel(**kwargs)
        np.testing.assert_array_equal(ok, nk)
        np.testing.assert_array_equal(oi, ni)


# ===================================================================
# antisymmetric_singular_values parity
# ===================================================================


class TestParityAntisymmetricSingularValues:
    def _kernel(self, N=8, seed=35):
        gen = torch.Generator().manual_seed(seed)
        kwargs = {
            "positions": torch.randn(N, 3, generator=gen),
            "velocities": torch.randn(N, 3, generator=gen),
            "fitness": torch.rand(N, generator=gen).clamp(min=1e-6),
            "alive_mask": torch.ones(N, dtype=torch.bool),
            "epsilon_c": 1.0,
        }
        k, _ = old_build_phase_space_antisymmetric_kernel(**kwargs)
        return k

    def test_basic(self):
        k = self._kernel()
        old = old_antisymmetric_singular_values(k)
        new = new_antisymmetric_singular_values(k)
        np.testing.assert_array_equal(old, new)

    def test_top_k(self):
        k = self._kernel()
        old = old_antisymmetric_singular_values(k, top_k=3)
        new = new_antisymmetric_singular_values(k, top_k=3)
        np.testing.assert_array_equal(old, new)
