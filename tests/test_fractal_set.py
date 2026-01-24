import pytest
import torch


try:
    import networkx  # noqa: F401

    _HAS_NETWORKX = True
except ModuleNotFoundError:
    _HAS_NETWORKX = False

if _HAS_NETWORKX:
    from fragile.fractalai.core.fractal_set import FractalSet
else:
    FractalSet = None

pytestmark = pytest.mark.skipif(not _HAS_NETWORKX, reason="networkx not installed")

from fragile.fractalai.bounds import TorchBounds
from fragile.fractalai.core.cloning import CloneOperator
from fragile.fractalai.core.companion_selection import CompanionSelection
from fragile.fractalai.core.euclidean_gas import EuclideanGas
from fragile.fractalai.core.fitness import FitnessOperator
from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.core.kinetic_operator import KineticOperator


def _make_simple_gas(bounds: TorchBounds) -> EuclideanGas:
    def potential(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x**2, dim=-1)

    companion_selection = CompanionSelection(method="cloning", epsilon=0.1, lambda_alg=0.0)
    cloning = CloneOperator(p_max=1.0, sigma_x=0.0, alpha_restitution=0.0)
    kinetic = KineticOperator(
        gamma=0.1,
        beta=1.0,
        delta_t=0.1,
        use_potential_force=False,
        potential=potential,
    )
    fitness = FitnessOperator()
    return EuclideanGas(
        N=2,
        d=1,
        companion_selection=companion_selection,
        potential=potential,
        kinetic_op=kinetic,
        cloning=cloning,
        fitness_op=fitness,
        bounds=bounds,
        enable_kinetic=False,
        pbc=False,
    )


def test_history_get_alive_walkers_indexing():
    bounds = TorchBounds(low=torch.tensor([-1.0]), high=torch.tensor([1.0]))
    gas = _make_simple_gas(bounds)
    x_init = torch.tensor([[0.0], [2.0]])
    v_init = torch.zeros_like(x_init)

    history = gas.run(n_steps=2, record_every=1, x_init=x_init, v_init=v_init, seed=123)

    expected_step0 = torch.where(bounds.contains(history.x_before_clone[0]))[0]
    expected_step1 = torch.where(history.alive_mask[0])[0]
    expected_step2 = torch.where(history.alive_mask[1])[0]

    assert torch.equal(history.get_alive_walkers(0), expected_step0)
    assert torch.equal(history.get_alive_walkers(1), expected_step1)
    assert torch.equal(history.get_alive_walkers(2), expected_step2)


def test_fractal_set_structure_counts():
    bounds = TorchBounds(low=torch.tensor([-1.0]), high=torch.tensor([1.0]))
    gas = _make_simple_gas(bounds)
    x_init = torch.tensor([[0.0], [2.0]])
    v_init = torch.zeros_like(x_init)

    history = gas.run(n_steps=2, record_every=1, x_init=x_init, v_init=v_init, seed=123)
    fs = FractalSet(history)

    n_recorded = history.n_recorded
    expected_nodes = history.N * (2 * n_recorded + max(n_recorded - 1, 0))
    assert fs.total_nodes == expected_nodes
    assert fs.num_clone_edges == history.N * max(n_recorded - 1, 0)

    expected_cst = int(history.alive_mask.sum().item())
    assert fs.num_cst_edges == expected_cst

    expected_ig = 0
    for t_idx in range(n_recorded - 1):
        alive = history.alive_mask[t_idx]
        comp_dist = history.companions_distance[t_idx]
        comp_clone = history.companions_clone[t_idx]
        alive_ids = {int(i) for i in torch.where(alive)[0].tolist()}
        for walker_id in alive_ids:
            dist_id = int(comp_dist[walker_id].item())
            clone_id = int(comp_clone[walker_id].item())
            if dist_id == clone_id:
                if dist_id in alive_ids and dist_id != walker_id:
                    expected_ig += 1
            else:
                if dist_id in alive_ids and dist_id != walker_id:
                    expected_ig += 1
                if clone_id in alive_ids and clone_id != walker_id:
                    expected_ig += 1

    assert fs.num_ig_edges == expected_ig
    assert fs.num_ia_edges == expected_ig
    assert fs.num_triangles == expected_ig

    if fs.num_clone_edges:
        assert torch.allclose(fs.edges["clone"]["delta_x"][0], history.clone_delta_x[0, 0])
        assert torch.allclose(fs.edges["clone"]["delta_x"][1], history.clone_delta_x[0, 1])


def test_fractal_set_pbc_delta_x():
    bounds = TorchBounds(low=torch.tensor([0.0]), high=torch.tensor([1.0]))
    dtype = torch.float32

    x_before = torch.tensor([[[0.9]], [[0.9]]], dtype=dtype)
    x_after = torch.tensor([[[0.9]]], dtype=dtype)
    x_final = torch.tensor([[[0.9]], [[0.1]]], dtype=dtype)
    v_before = torch.zeros_like(x_before)
    v_after = torch.zeros_like(x_after)
    v_final = torch.zeros_like(x_final)

    history = RunHistory(
        N=1,
        d=1,
        n_steps=1,
        n_recorded=2,
        record_every=1,
        terminated_early=False,
        final_step=1,
        recorded_steps=[0, 1],
        delta_t=1.0,
        pbc=True,
        params={},
        rng_seed=None,
        rng_state=None,
        bounds=bounds,
        x_before_clone=x_before,
        v_before_clone=v_before,
        U_before=torch.zeros(2, 1, dtype=dtype),
        x_after_clone=x_after,
        v_after_clone=v_after,
        U_after_clone=torch.zeros(1, 1, dtype=dtype),
        x_final=x_final,
        v_final=v_final,
        U_final=torch.zeros(2, 1, dtype=dtype),
        n_alive=torch.ones(2, dtype=torch.long),
        num_cloned=torch.zeros(1, dtype=torch.long),
        step_times=torch.zeros(1, dtype=torch.float32),
        fitness=torch.zeros(1, 1, dtype=dtype),
        rewards=torch.zeros(1, 1, dtype=dtype),
        cloning_scores=torch.zeros(1, 1, dtype=dtype),
        cloning_probs=torch.zeros(1, 1, dtype=dtype),
        will_clone=torch.zeros(1, 1, dtype=torch.bool),
        alive_mask=torch.ones(1, 1, dtype=torch.bool),
        companions_distance=torch.zeros(1, 1, dtype=torch.long),
        companions_clone=torch.zeros(1, 1, dtype=torch.long),
        clone_jitter=torch.zeros(1, 1, 1, dtype=dtype),
        clone_delta_x=torch.zeros(1, 1, 1, dtype=dtype),
        clone_delta_v=torch.zeros(1, 1, 1, dtype=dtype),
        distances=torch.zeros(1, 1, dtype=dtype),
        z_rewards=torch.zeros(1, 1, dtype=dtype),
        z_distances=torch.zeros(1, 1, dtype=dtype),
        pos_squared_differences=torch.zeros(1, 1, dtype=dtype),
        vel_squared_differences=torch.zeros(1, 1, dtype=dtype),
        rescaled_rewards=torch.zeros(1, 1, dtype=dtype),
        rescaled_distances=torch.zeros(1, 1, dtype=dtype),
        mu_rewards=torch.zeros(1, dtype=dtype),
        sigma_rewards=torch.zeros(1, dtype=dtype),
        mu_distances=torch.zeros(1, dtype=dtype),
        sigma_distances=torch.zeros(1, dtype=dtype),
        fitness_gradients=None,
        fitness_hessians_diag=None,
        fitness_hessians_full=None,
        force_stable=torch.zeros(1, 1, 1, dtype=dtype),
        force_adapt=torch.zeros(1, 1, 1, dtype=dtype),
        force_viscous=torch.zeros(1, 1, 1, dtype=dtype),
        force_friction=torch.zeros(1, 1, 1, dtype=dtype),
        force_total=torch.zeros(1, 1, 1, dtype=dtype),
        noise=torch.zeros(1, 1, 1, dtype=dtype),
        sigma_reg_diag=None,
        sigma_reg_full=None,
        total_time=0.0,
        init_time=0.0,
    )

    fs = FractalSet(history)
    delta_x = fs.edges["cst"]["delta_x"][0]
    assert torch.allclose(delta_x, torch.tensor([0.2], dtype=dtype), atol=1e-6)
