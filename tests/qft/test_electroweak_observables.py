"""Tests for vectorized electroweak observables and Dirac kernel modes."""

from __future__ import annotations

import numpy as np
import torch

from fragile.fractalai.qft.dirac_spectrum import (
    compute_dirac_spectrum,
    dedup_skew_sv,
    DiracSpectrumConfig,
)
from fragile.fractalai.qft.electroweak_channels import (
    compute_electroweak_coupling_constants,
    compute_electroweak_snapshot_operators,
    compute_emergent_electroweak_scales,
    ELECTROWEAK_CHANNELS,
    ElectroweakChannelConfig,
)
from fragile.fractalai.qft.electroweak_observables import (
    antisymmetric_singular_values,
    build_phase_space_antisymmetric_kernel,
    classify_walker_types,
    compute_weighted_electroweak_ops_vectorized,
    pack_neighbor_lists,
    pack_neighbors_from_edges,
    SU2_BASE_OPERATOR_CHANNELS,
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
    su2_phase_exp_directed = torch.zeros_like(u1_phase_exp)
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
        su2_pair = torch.exp(1j * su2_phase).to(complex_dtype)
        su2_phase_exp[idx] = (w * su2_pair).sum()
        su2_oriented = torch.where(scores >= 0, su2_pair, torch.conj(su2_pair))
        su2_phase_exp_directed[idx] = (w * su2_oriented).sum()

        diff_x = positions[idx] - positions[neighbors]
        diff_v = velocities[idx] - velocities[neighbors]
        dist_sq = (diff_x**2).sum(dim=-1) + lambda_alg * (diff_v**2).sum(dim=-1)
        u1_weight = torch.exp(-dist_sq / (2.0 * epsilon_d**2))
        su2_weight = torch.exp(-dist_sq / (2.0 * epsilon_c**2))
        u1_amp[idx] = (w * torch.sqrt(u1_weight)).sum()
        su2_amp[idx] = (w * torch.sqrt(su2_weight)).sum()

    su2_comp_phase_exp = torch.zeros_like(su2_phase_exp)
    su2_comp_phase_exp_directed = torch.zeros_like(su2_phase_exp)
    su2_comp_amp = torch.zeros_like(su2_amp)
    for idx in alive_idx:
        neighbors = neighbor_indices[idx]
        weights = neighbor_weights[idx]
        if neighbors.numel() == 0:
            continue
        w = weights / weights.sum()
        su2_comp_phase_exp[idx] = (w * su2_phase_exp[neighbors]).sum()
        su2_comp_phase_exp_directed[idx] = (w * su2_phase_exp_directed[neighbors]).sum()
        su2_comp_amp[idx] = (w * su2_amp[neighbors]).sum()

    mask = alive.to(u1_phase_exp.dtype)
    u1_amp_c = u1_amp.to(u1_phase_exp.dtype)
    su2_amp_c = su2_amp.to(u1_phase_exp.dtype)
    su2_comp_amp_c = su2_comp_amp.to(u1_phase_exp.dtype)

    su2_standard = {
        "u1_phase": u1_phase_exp * mask,
        "u1_dressed": u1_amp_c * u1_phase_exp * mask,
        "u1_phase_q2": u1_phase_q2_exp * mask,
        "u1_dressed_q2": u1_amp_c * u1_phase_q2_exp * mask,
        "su2_phase": su2_phase_exp * mask,
        "su2_component": su2_amp_c * su2_phase_exp * mask,
        "su2_doublet": (su2_amp_c * su2_phase_exp + su2_comp_amp_c * su2_comp_phase_exp) * mask,
        "su2_doublet_diff": (su2_amp_c * su2_phase_exp - su2_comp_amp_c * su2_comp_phase_exp)
        * mask,
    }
    su2_directional = {
        "su2_phase_directed": su2_phase_exp_directed * mask,
        "su2_component_directed": su2_amp_c * su2_phase_exp_directed * mask,
        "su2_doublet_directed": (
            su2_amp_c * su2_phase_exp_directed + su2_comp_amp_c * su2_comp_phase_exp_directed
        )
        * mask,
        "su2_doublet_diff_directed": (
            su2_amp_c * su2_phase_exp_directed - su2_comp_amp_c * su2_comp_phase_exp_directed
        )
        * mask,
    }
    out = {**su2_standard, **su2_directional}
    for name in SU2_BASE_OPERATOR_CHANNELS:
        out[f"{name}_cloner"] = torch.zeros_like(mask)
        out[f"{name}_resister"] = torch.zeros_like(mask)
        out[f"{name}_persister"] = torch.zeros_like(mask)
    out["ew_mixed"] = (u1_amp_c * su2_amp_c) * u1_phase_exp * su2_phase_exp * mask
    # Symmetry-breaking channels
    out["fitness_phase"] = (-fitness).to(complex_dtype) * mask
    out["clone_indicator"] = torch.zeros(n_walkers, device=device, dtype=complex_dtype)
    # Parity velocity channels (all zero when walker type split is disabled)
    out["velocity_norm_cloner"] = torch.zeros(n_walkers, device=device, dtype=complex_dtype)
    out["velocity_norm_resister"] = torch.zeros(n_walkers, device=device, dtype=complex_dtype)
    out["velocity_norm_persister"] = torch.zeros(n_walkers, device=device, dtype=complex_dtype)
    return out


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
    alive = torch.tensor(
        [True, True, True, True, False, True, False, True, True], dtype=torch.bool
    )

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


def test_vectorized_walker_type_split_matches_masks_in_score_directed_mode():
    torch.manual_seed(13)
    n_walkers = 8
    dim = 3
    positions = torch.randn(n_walkers, dim, dtype=torch.float64)
    velocities = torch.randn(n_walkers, dim, dtype=torch.float64)
    fitness = torch.tensor([0.2, 0.8, 0.5, 1.1, 0.7, 0.4, 1.3, 0.6], dtype=torch.float64)
    alive = torch.tensor([True, True, True, True, True, True, True, False], dtype=torch.bool)
    will_clone = torch.tensor(
        [False, True, False, False, True, False, False, False], dtype=torch.bool
    )

    neighbor_indices = []
    neighbor_weights = []
    for i in range(n_walkers):
        if not alive[i]:
            neighbor_indices.append(torch.zeros(0, dtype=torch.long))
            neighbor_weights.append(torch.zeros(0, dtype=torch.float64))
            continue
        neighbors = torch.tensor(
            [j for j in range(n_walkers) if j != i and alive[j]], dtype=torch.long
        )
        weights = torch.ones(neighbors.shape[0], dtype=torch.float64)
        neighbor_indices.append(neighbors)
        neighbor_weights.append(weights)

    packed = pack_neighbor_lists(
        neighbor_indices,
        neighbor_weights,
        n_walkers,
        device=positions.device,
        dtype=positions.dtype,
    )
    out = compute_weighted_electroweak_ops_vectorized(
        positions=positions,
        velocities=velocities,
        fitness=fitness,
        alive=alive,
        neighbors=packed,
        h_eff=1.0,
        epsilon_d=1.1,
        epsilon_c=0.9,
        epsilon_clone=0.05,
        lambda_alg=0.4,
        pbc=False,
        will_clone=will_clone,
        su2_operator_mode="score_directed",
        enable_walker_type_split=True,
        walker_type_scope="frame_global",
    )

    cloner, resister, persister = classify_walker_types(
        fitness=fitness,
        alive=alive,
        will_clone=will_clone,
    )
    assert int(cloner.sum()) > 0
    assert int(resister.sum()) > 0
    assert int(persister.sum()) > 0

    cloner_c = cloner.to(out["su2_phase"].dtype)
    resister_c = resister.to(out["su2_phase"].dtype)
    persister_c = persister.to(out["su2_phase"].dtype)
    for base_name in SU2_BASE_OPERATOR_CHANNELS:
        torch.testing.assert_close(
            out[base_name], out[f"{base_name}_directed"], atol=1e-12, rtol=0.0
        )
        torch.testing.assert_close(
            out[f"{base_name}_cloner"], out[base_name] * cloner_c, atol=1e-12, rtol=0.0
        )
        torch.testing.assert_close(
            out[f"{base_name}_resister"], out[base_name] * resister_c, atol=1e-12, rtol=0.0
        )
        torch.testing.assert_close(
            out[f"{base_name}_persister"], out[base_name] * persister_c, atol=1e-12, rtol=0.0
        )


def test_vectorized_weighted_ops_family_neighbors_match_shared_when_identical():
    torch.manual_seed(41)
    n_walkers = 7
    dim = 3
    positions = torch.randn(n_walkers, dim, dtype=torch.float64)
    velocities = torch.randn(n_walkers, dim, dtype=torch.float64)
    fitness = torch.rand(n_walkers, dtype=torch.float64) + 0.2
    alive = torch.tensor([True, True, True, False, True, True, True], dtype=torch.bool)

    neighbor_indices: list[torch.Tensor] = []
    neighbor_weights: list[torch.Tensor] = []
    for i in range(n_walkers):
        if not alive[i]:
            neighbor_indices.append(torch.zeros(0, dtype=torch.long))
            neighbor_weights.append(torch.zeros(0, dtype=torch.float64))
            continue
        choices = torch.tensor([j for j in range(n_walkers) if j != i], dtype=torch.long)
        perm = torch.randperm(len(choices))[:3]
        picked = choices[perm]
        weights = torch.rand(picked.shape[0], dtype=torch.float64) + 0.1
        neighbor_indices.append(picked)
        neighbor_weights.append(weights)

    packed = pack_neighbor_lists(
        neighbor_indices,
        neighbor_weights,
        n_walkers,
        device=positions.device,
        dtype=positions.dtype,
    )
    kwargs = {
        "h_eff": 0.9,
        "epsilon_d": 1.1,
        "epsilon_c": 0.8,
        "epsilon_clone": 0.04,
        "lambda_alg": 0.25,
        "bounds": None,
        "pbc": False,
    }
    legacy = compute_weighted_electroweak_ops_vectorized(
        positions=positions,
        velocities=velocities,
        fitness=fitness,
        alive=alive,
        neighbors=packed,
        **kwargs,
    )
    routed = compute_weighted_electroweak_ops_vectorized(
        positions=positions,
        velocities=velocities,
        fitness=fitness,
        alive=alive,
        neighbors_u1=packed,
        neighbors_su2=packed,
        neighbors_ew_mixed=packed,
        **kwargs,
    )
    for channel in ELECTROWEAK_CHANNELS:
        torch.testing.assert_close(legacy[channel], routed[channel], rtol=1e-10, atol=1e-12)


def test_vectorized_weighted_ops_family_neighbor_routing_isolated_by_operator_family():
    dtype = torch.float64
    positions = torch.tensor(
        [[0.0, 0.0], [1.2, -0.2], [-0.3, 1.5]],
        dtype=dtype,
    )
    velocities = torch.tensor(
        [[0.0, 0.0], [0.4, -0.1], [-0.2, 0.3]],
        dtype=dtype,
    )
    fitness = torch.tensor([0.3, 1.1, 2.4], dtype=dtype)
    alive = torch.tensor([True, True, True], dtype=torch.bool)

    def _pack(indices: list[list[int]]) -> torch.Tensor:
        return pack_neighbor_lists(
            [torch.tensor(row, dtype=torch.long) for row in indices],
            [torch.ones(len(row), dtype=dtype) for row in indices],
            n_walkers=3,
            device=positions.device,
            dtype=dtype,
        )

    neighbors_u1 = _pack([[1], [0], [1]])
    neighbors_su2 = _pack([[2], [2], [0]])
    neighbors_su2_alt = _pack([[1], [0], [1]])
    neighbors_mixed = _pack([[1, 2], [0, 2], [0, 1]])
    neighbors_mixed_alt = _pack([[1], [2], [1]])

    kwargs = {
        "h_eff": 1.0,
        "epsilon_d": 1.2,
        "epsilon_c": 0.9,
        "epsilon_clone": 0.05,
        "lambda_alg": 0.3,
        "bounds": None,
        "pbc": False,
    }
    base = compute_weighted_electroweak_ops_vectorized(
        positions=positions,
        velocities=velocities,
        fitness=fitness,
        alive=alive,
        neighbors_u1=neighbors_u1,
        neighbors_su2=neighbors_su2,
        neighbors_ew_mixed=neighbors_mixed,
        **kwargs,
    )

    su2_changed = compute_weighted_electroweak_ops_vectorized(
        positions=positions,
        velocities=velocities,
        fitness=fitness,
        alive=alive,
        neighbors_u1=neighbors_u1,
        neighbors_su2=neighbors_su2_alt,
        neighbors_ew_mixed=neighbors_mixed,
        **kwargs,
    )
    for channel in ("u1_phase", "u1_dressed", "u1_phase_q2", "u1_dressed_q2"):
        torch.testing.assert_close(base[channel], su2_changed[channel], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(base["ew_mixed"], su2_changed["ew_mixed"], rtol=0.0, atol=1e-12)
    assert any(
        (base[channel] - su2_changed[channel]).abs().max().item() > 1e-9
        for channel in (
            "su2_phase",
            "su2_component",
            "su2_doublet",
            "su2_doublet_diff",
            "su2_phase_directed",
            "su2_component_directed",
            "su2_doublet_directed",
            "su2_doublet_diff_directed",
        )
    )

    mixed_changed = compute_weighted_electroweak_ops_vectorized(
        positions=positions,
        velocities=velocities,
        fitness=fitness,
        alive=alive,
        neighbors_u1=neighbors_u1,
        neighbors_su2=neighbors_su2,
        neighbors_ew_mixed=neighbors_mixed_alt,
        **kwargs,
    )
    for channel in ("u1_phase", "u1_dressed", "u1_phase_q2", "u1_dressed_q2"):
        torch.testing.assert_close(base[channel], mixed_changed[channel], rtol=0.0, atol=1e-12)
    for channel in (
        "su2_phase",
        "su2_component",
        "su2_doublet",
        "su2_doublet_diff",
        "su2_phase_directed",
        "su2_component_directed",
        "su2_doublet_directed",
        "su2_doublet_diff_directed",
    ):
        torch.testing.assert_close(base[channel], mixed_changed[channel], rtol=0.0, atol=1e-12)
    assert (base["ew_mixed"] - mixed_changed["ew_mixed"]).abs().max().item() > 1e-9


class _EmergentScaleHistoryStub:
    def __init__(self):
        self.n_recorded = 2
        self.d = 2
        self.pbc = False
        self.bounds = None
        self.params = {"fitness": {"lambda_alg": 0.0}, "kinetic": {"nu": 1.0}}
        self.x_before_clone = torch.tensor(
            [
                [[0.0, 0.0], [3.0, 4.0], [0.0, 5.0]],
                [[0.0, 0.0], [3.0, 4.0], [0.0, 5.0]],
            ],
            dtype=torch.float64,
        )
        self.v_before_clone = torch.zeros(2, 3, 2, dtype=torch.float64)
        self.alive_mask = torch.ones(1, 3, dtype=torch.bool)
        self.fitness = torch.tensor([[0.3, 1.0, 1.8]], dtype=torch.float64)
        self.companions_distance = torch.tensor([[1, 0, 0]], dtype=torch.long)
        self.companions_clone = torch.tensor([[2, 2, 1]], dtype=torch.long)
        self.distances = None
        self.force_viscous = torch.zeros(1, 3, 2, dtype=torch.float64)


def test_compute_emergent_electroweak_scales_from_companions():
    history = _EmergentScaleHistoryStub()
    scales = compute_emergent_electroweak_scales(
        history,
        frame_indices=[1],
        lambda_alg=0.0,
    )
    assert np.isclose(float(scales["eps_distance_emergent"]), 5.0, rtol=0.0, atol=1e-10)
    assert np.isclose(float(scales["eps_clone_emergent"]), np.sqrt(15.0), rtol=0.0, atol=1e-10)
    assert int(scales["n_distance_samples"]) == 3
    assert int(scales["n_clone_samples"]) == 3


def test_compute_electroweak_coupling_constants_uses_emergent_scales():
    history = _EmergentScaleHistoryStub()
    couplings = compute_electroweak_coupling_constants(
        history,
        h_eff=1.0,
        frame_indices=[1],
        lambda_alg=0.0,
    )
    assert np.isclose(float(couplings["g1_est"]), 0.2, rtol=0.0, atol=1e-10)
    assert np.isclose(float(couplings["g2_est"]), np.sqrt(2.0 / 15.0), rtol=0.0, atol=1e-10)
    assert np.isclose(float(couplings["sin2_theta_w_emergent"]), 3.0 / 13.0, rtol=0.0, atol=1e-10)
    assert np.isclose(float(couplings["tan_theta_w_emergent"]), np.sqrt(0.3), rtol=0.0, atol=1e-10)


def test_snapshot_operator_routing_uses_family_companion_topologies():
    history = _EmergentScaleHistoryStub()
    cfg = ElectroweakChannelConfig(
        neighbor_method="companions",
        companion_topology="distance",
        companion_topology_u1="distance",
        companion_topology_su2="clone",
        companion_topology_ew_mixed="both",
        h_eff=1.0,
        epsilon_d=1.2,
        epsilon_c=0.9,
        epsilon_clone=0.05,
        lambda_alg=0.0,
    )
    base = compute_electroweak_snapshot_operators(
        history,
        config=cfg,
        channels=list(ELECTROWEAK_CHANNELS),
        frame_idx=1,
    )

    history.companions_clone = torch.tensor([[2, 1, 0]], dtype=torch.long)
    clone_changed = compute_electroweak_snapshot_operators(
        history,
        config=cfg,
        channels=list(ELECTROWEAK_CHANNELS),
        frame_idx=1,
    )

    for channel in ("u1_phase", "u1_dressed", "u1_phase_q2", "u1_dressed_q2"):
        torch.testing.assert_close(base[channel], clone_changed[channel], rtol=0.0, atol=1e-12)
    assert any(
        (base[channel] - clone_changed[channel]).abs().max().item() > 1e-9
        for channel in (
            "su2_phase",
            "su2_component",
            "su2_doublet",
            "su2_doublet_diff",
            "su2_phase_directed",
            "su2_component_directed",
            "su2_doublet_directed",
            "su2_doublet_diff_directed",
        )
    )
    assert (base["ew_mixed"] - clone_changed["ew_mixed"]).abs().max().item() > 1e-9


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
