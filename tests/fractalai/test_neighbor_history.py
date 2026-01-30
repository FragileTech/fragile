import torch

from fragile.fractalai.core.euclidean_gas import SwarmState
from fragile.fractalai.core.vec_history import VectorizedHistoryRecorder


def _make_info(N: int, d: int, device: torch.device, dtype: torch.dtype) -> dict:
    zeros_n = torch.zeros(N, device=device, dtype=dtype)
    zeros_nd = torch.zeros(N, d, device=device, dtype=dtype)
    zeros_n_long = torch.zeros(N, device=device, dtype=torch.long)
    alive = torch.ones(N, dtype=torch.bool, device=device)
    return {
        "alive_mask": alive,
        "companions_distance": zeros_n_long,
        "companions_clone": zeros_n_long,
        "rewards": zeros_n,
        "fitness": zeros_n,
        "cloning_scores": zeros_n,
        "cloning_probs": zeros_n,
        "will_clone": torch.zeros(N, dtype=torch.bool, device=device),
        "num_cloned": 0,
        "clone_jitter": zeros_nd,
        "clone_delta_x": zeros_nd,
        "clone_delta_v": zeros_nd,
        "distances": zeros_n,
        "z_rewards": zeros_n,
        "z_distances": zeros_n,
        "pos_squared_differences": zeros_n,
        "vel_squared_differences": zeros_n,
        "rescaled_rewards": zeros_n,
        "rescaled_distances": zeros_n,
        "mu_rewards": zeros_n,
        "sigma_rewards": zeros_n,
        "mu_distances": zeros_n,
        "sigma_distances": zeros_n,
        "U_before": zeros_n,
        "U_after_clone": zeros_n,
        "U_final": zeros_n,
    }


def test_neighbor_history_records_empty_entries() -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    N, d = 3, 3

    recorder = VectorizedHistoryRecorder(
        N=N,
        d=d,
        n_recorded=2,
        device=device,
        dtype=dtype,
        record_gradients=False,
        record_hessians_diag=False,
        record_hessians_full=False,
        record_sigma_reg_diag=False,
        record_sigma_reg_full=False,
        record_neighbors=True,
        record_voronoi=True,
    )

    state = SwarmState(torch.zeros(N, d, device=device), torch.zeros(N, d, device=device))
    recorder.record_initial_state(state, n_alive=N)

    info = _make_info(N, d, device, dtype)
    recorder.record_step(
        state_before=state,
        state_cloned=state,
        state_final=state,
        info=info,
        step_time=0.01,
        grad_fitness=None,
        hess_fitness=None,
        is_diagonal_hessian=False,
        kinetic_info=None,
    )

    history = recorder.build(
        n_steps=1,
        record_every=1,
        terminated_early=False,
        final_step=1,
        total_time=0.02,
        init_time=0.0,
        bounds=None,
        recorded_steps=[0, 1],
        delta_t=0.1,
        pbc=False,
        params={},
        rng_seed=None,
        rng_state=None,
    )

    assert history.neighbor_edges is not None
    assert len(history.neighbor_edges) == history.n_recorded
    assert all(torch.is_tensor(edges) for edges in history.neighbor_edges)

    assert history.voronoi_regions is not None
    assert len(history.voronoi_regions) == history.n_recorded
    assert all(isinstance(entry, dict) for entry in history.voronoi_regions)
