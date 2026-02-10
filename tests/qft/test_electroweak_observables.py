"""Tests for vectorized electroweak observables and Dirac kernel modes."""

from __future__ import annotations

import numpy as np
import torch

from fragile.fractalai.qft.dirac_spectrum import (
    compute_dirac_spectrum,
    dedup_skew_sv,
    DiracSpectrumConfig,
)
from fragile.fractalai.qft.electroweak_channels import ELECTROWEAK_CHANNELS
from fragile.fractalai.qft.electroweak_observables import (
    antisymmetric_singular_values,
    build_phase_space_antisymmetric_kernel,
    compute_weighted_electroweak_ops_vectorized,
    pack_neighbor_lists,
    pack_neighbors_from_edges,
)


def _reference_weighted_ops(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    fitness: torch.Tensor,
    alive: torch.Tensor,
    neighbor_indices: list[torch.Tensor],
    neighbor_weights: list[torch.Tensor],
    *,
    h_eff: float,
    epsilon_d: float,
    epsilon_c: float,
    epsilon_clone: float,
    lambda_alg: float,
) -> dict[str, torch.Tensor]:
    n_walkers = positions.shape[0]
    device = positions.device
    complex_dtype = torch.complex128 if positions.dtype == torch.float64 else torch.complex64

    u1_phase_exp = torch.zeros(n_walkers, device=device, dtype=complex_dtype)
    u1_phase_q2_exp = torch.zeros_like(u1_phase_exp)
    su2_phase_exp = torch.zeros_like(u1_phase_exp)
    u1_amp = torch.zeros(n_walkers, device=device, dtype=positions.dtype)
    su2_amp = torch.zeros_like(u1_amp)

    alive_idx = torch.where(alive)[0].tolist()
    for idx in alive_idx:
        neighbors = neighbor_indices[idx]
        weights = neighbor_weights[idx]
        if neighbors.numel() == 0:
            continue
        w = weights / weights.sum()
        fitness_i = fitness[idx]
        fitness_j = fitness[neighbors]

        phase = -(fitness_j - fitness_i) / h_eff
        phase_exp = torch.exp(1j * phase).to(complex_dtype)
        phase_q2_exp = torch.exp(1j * (2.0 * phase)).to(complex_dtype)
        u1_phase_exp[idx] = (w * phase_exp).sum()
        u1_phase_q2_exp[idx] = (w * phase_q2_exp).sum()

        denom = fitness_i + epsilon_clone
        denom = torch.where(denom.abs() < 1e-12, torch.full_like(denom, epsilon_clone), denom)
        scores = (fitness_j - fitness_i) / denom
        su2_phase = scores / h_eff
        su2_phase_exp[idx] = (w * torch.exp(1j * su2_phase).to(complex_dtype)).sum()

        diff_x = positions[idx] - positions[neighbors]
        diff_v = velocities[idx] - velocities[neighbors]
        dist_sq = (diff_x**2).sum(dim=-1) + lambda_alg * (diff_v**2).sum(dim=-1)
        u1_weight = torch.exp(-dist_sq / (2.0 * epsilon_d**2))
        su2_weight = torch.exp(-dist_sq / (2.0 * epsilon_c**2))
        u1_amp[idx] = (w * torch.sqrt(u1_weight)).sum()
        su2_amp[idx] = (w * torch.sqrt(su2_weight)).sum()

    su2_comp_phase_exp = torch.zeros_like(su2_phase_exp)
    su2_comp_amp = torch.zeros_like(su2_amp)
    for idx in alive_idx:
        neighbors = neighbor_indices[idx]
        weights = neighbor_weights[idx]
        if neighbors.numel() == 0:
            continue
        w = weights / weights.sum()
        su2_comp_phase_exp[idx] = (w * su2_phase_exp[neighbors]).sum()
        su2_comp_amp[idx] = (w * su2_amp[neighbors]).sum()

    mask = alive.to(u1_phase_exp.dtype)
    u1_amp_c = u1_amp.to(u1_phase_exp.dtype)
    su2_amp_c = su2_amp.to(u1_phase_exp.dtype)
    su2_comp_amp_c = su2_comp_amp.to(u1_phase_exp.dtype)

    return {
        "u1_phase": u1_phase_exp * mask,
        "u1_dressed": u1_amp_c * u1_phase_exp * mask,
        "u1_phase_q2": u1_phase_q2_exp * mask,
        "u1_dressed_q2": u1_amp_c * u1_phase_q2_exp * mask,
        "su2_phase": su2_phase_exp * mask,
        "su2_component": su2_amp_c * su2_phase_exp * mask,
        "su2_doublet": (su2_amp_c * su2_phase_exp + su2_comp_amp_c * su2_comp_phase_exp) * mask,
        "su2_doublet_diff": (su2_amp_c * su2_phase_exp - su2_comp_amp_c * su2_comp_phase_exp) * mask,
        "ew_mixed": (u1_amp_c * su2_amp_c) * u1_phase_exp * su2_phase_exp * mask,
    }


def test_pack_neighbors_from_edges_respects_alive_and_topk():
    edges = torch.tensor(
        [
            [0, 1],
            [0, 2],  # dropped (dead dst)
            [1, 0],
            [1, 3],
            [1, 2],  # dropped (dead dst)
            [3, 0],
        ],
        dtype=torch.long,
    )
    edge_weights = torch.tensor([0.2, 0.8, 0.4, 0.6, 0.9, 0.5], dtype=torch.float32)
    alive = torch.tensor([True, True, False, True], dtype=torch.bool)

    packed = pack_neighbors_from_edges(
        edges=edges,
        edge_weights=edge_weights,
        alive=alive,
        n_walkers=4,
        max_neighbors=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert packed.indices.shape[0] == 4
    assert packed.weights.shape == packed.indices.shape
    assert packed.valid.shape == packed.indices.shape

    # Walker 0 only keeps neighbor 1.
    assert packed.valid[0, 0].item()
    assert packed.indices[0, 0].item() == 1
    assert np.isclose(float(packed.weights[0, 0].item()), 1.0)

    # Walker 1 keeps neighbors 3 and 0, sorted by descending edge weight.
    valid_1 = packed.valid[1]
    idx_1 = packed.indices[1][valid_1].tolist()
    w_1 = packed.weights[1][valid_1].tolist()
    by_neighbor = dict(zip(idx_1, w_1, strict=False))
    assert set(idx_1) == {0, 3}
    np.testing.assert_allclose(
        [by_neighbor[3], by_neighbor[0]],
        [0.6, 0.4],
        atol=1e-6,
        rtol=0.0,
    )


def test_vectorized_weighted_electroweak_ops_match_reference():
    torch.manual_seed(7)
    n_walkers = 9
    dim = 3
    positions = torch.randn(n_walkers, dim, dtype=torch.float64)
    velocities = torch.randn(n_walkers, dim, dtype=torch.float64)
    fitness = torch.randn(n_walkers, dtype=torch.float64).abs() + 0.2
    alive = torch.tensor([True, True, True, True, False, True, False, True, True], dtype=torch.bool)

    neighbor_indices: list[torch.Tensor] = []
    neighbor_weights: list[torch.Tensor] = []
    for i in range(n_walkers):
        if not alive[i]:
            neighbor_indices.append(torch.zeros(0, dtype=torch.long))
            neighbor_weights.append(torch.zeros(0, dtype=torch.float64))
            continue
        candidates = torch.tensor([j for j in range(n_walkers) if j != i], dtype=torch.long)
        choice = candidates[torch.randperm(len(candidates))[:4]]
        weights = torch.rand(choice.shape[0], dtype=torch.float64) + 0.05
        neighbor_indices.append(choice)
        neighbor_weights.append(weights)

    params = {
        "h_eff": 0.75,
        "epsilon_d": 1.2,
        "epsilon_c": 0.9,
        "epsilon_clone": 0.03,
        "lambda_alg": 0.5,
    }
    packed = pack_neighbor_lists(
        neighbor_indices,
        neighbor_weights,
        n_walkers,
        device=positions.device,
        dtype=positions.dtype,
    )
    vec = compute_weighted_electroweak_ops_vectorized(
        positions=positions,
        velocities=velocities,
        fitness=fitness,
        alive=alive,
        neighbors=packed,
        bounds=None,
        pbc=False,
        **params,
    )
    ref = _reference_weighted_ops(
        positions=positions,
        velocities=velocities,
        fitness=fitness,
        alive=alive,
        neighbor_indices=neighbor_indices,
        neighbor_weights=neighbor_weights,
        **params,
    )

    for channel in ELECTROWEAK_CHANNELS:
        torch.testing.assert_close(vec[channel], ref[channel], rtol=1e-8, atol=1e-10)


class _DiracHistoryStub:
    def __init__(self, n_recorded: int = 6, n_walkers: int = 10, dim: int = 3):
        self.n_recorded = n_recorded
        self.d = dim
        self.pbc = False
        self.bounds = None
        self.params = {"companion_selection_clone": {"epsilon": 0.85}}

        # Recorded arrays are indexed at [0, n_recorded-2] in dirac_spectrum.py.
        time_points = n_recorded - 1
        self.alive_mask = torch.ones(time_points, n_walkers, dtype=torch.bool)
        self.fitness = torch.rand(time_points, n_walkers, dtype=torch.float64) + 0.1
        self.will_clone = torch.rand(time_points, n_walkers) > 0.5
        self.force_viscous = torch.randn(time_points, n_walkers, dim, dtype=torch.float64)
        self.x_before_clone = torch.randn(n_recorded, n_walkers, dim, dtype=torch.float64)
        self.v_before_clone = torch.randn(n_recorded, n_walkers, dim, dtype=torch.float64)


def test_dirac_spectrum_phase_space_matches_direct_kernel():
    history = _DiracHistoryStub()
    cfg = DiracSpectrumConfig(
        kernel_mode="phase_space",
        epsilon_c=0.7,
        epsilon_clone=0.02,
        lambda_alg=0.4,
        h_eff=0.9,
        include_phase=True,
        min_sector_size=1,
    )
    result = compute_dirac_spectrum(history, cfg)

    t = history.n_recorded - 2
    fitness_t = history.fitness[t].cpu().numpy()
    alive_t = history.alive_mask[t].cpu().numpy()
    x_t = history.x_before_clone[t].cpu().numpy()
    v_t = history.v_before_clone[t].cpu().numpy()
    k_tilde, alive_indices = build_phase_space_antisymmetric_kernel(
        positions=x_t,
        velocities=v_t,
        fitness=fitness_t,
        alive_mask=alive_t,
        epsilon_c=cfg.epsilon_c if cfg.epsilon_c is not None else 1.0,
        lambda_alg=cfg.lambda_alg,
        h_eff=cfg.h_eff,
        epsilon_clone=cfg.epsilon_clone,
        bounds=None,
        pbc=False,
        include_phase=cfg.include_phase,
    )
    expected_full = dedup_skew_sv(antisymmetric_singular_values(k_tilde))

    np.testing.assert_allclose(result.full_singular_values, expected_full, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(result.alive_indices, alive_indices)
    assert result.n_alive == int(alive_t.sum())


def test_dirac_spectrum_fitness_ratio_mode_runs():
    history = _DiracHistoryStub()
    cfg = DiracSpectrumConfig(
        kernel_mode="fitness_ratio",
        epsilon_clone=0.02,
        min_sector_size=1,
    )
    result = compute_dirac_spectrum(history, cfg)
    assert result.full_singular_values.size > 0
