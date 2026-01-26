import torch

from fragile.fractalai.bounds import TorchBounds
from fragile.fractalai.core.cloning import CloneOperator
from fragile.fractalai.core.companion_selection import CompanionSelection
from fragile.fractalai.core.euclidean_gas import EuclideanGas
from fragile.fractalai.core.fitness import FitnessOperator, patched_standardization
from fragile.fractalai.core.kinetic_operator import KineticOperator


class QuadraticPotential:
    def __init__(self, dims: int, extent: float = 5.0) -> None:
        self.bounds = TorchBounds.from_tuples([(-extent, extent)] * dims)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x**2).sum(dim=-1)


def _run_single_step_cloning(rho: float | None, seed: int = 123) -> int:
    torch.manual_seed(seed)
    potential = QuadraticPotential(dims=2)
    companion = CompanionSelection(
        method="uniform",
        epsilon=1.0,
        lambda_alg=0.0,
        exclude_self=True,
    )
    fitness_op = FitnessOperator(rho=rho)
    kinetic = KineticOperator(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        use_potential_force=False,
        use_fitness_force=False,
    )
    cloning = CloneOperator(p_max=0.2)
    gas = EuclideanGas(
        N=32,
        d=2,
        potential=potential,
        companion_selection=companion,
        companion_selection_clone=companion,
        kinetic_op=kinetic,
        cloning=cloning,
        fitness_op=fitness_op,
        bounds=potential.bounds,
        device=torch.device("cpu"),
        dtype="float32",
        enable_cloning=True,
        enable_kinetic=False,
        pbc=False,
    )
    state = gas.initialize_state()
    _state_cloned, _state_final, info = gas.step(state, return_info=True)
    return int(info["num_cloned"])


def test_patched_standardization_mean_field_stats():
    values = torch.tensor([1.0, 2.0, 3.0])
    alive = torch.tensor([True, True, False])

    z_scores, mu, sigma = patched_standardization(values, alive, return_statistics=True)

    assert torch.isclose(mu, torch.tensor(1.5))
    assert torch.isclose(sigma, torch.tensor(0.5))
    assert torch.allclose(z_scores, torch.tensor([-1.0, 1.0, 0.0]))


def test_patched_standardization_local_fallback_to_global():
    values = torch.tensor([1.0, 2.0, -1.0])
    alive = torch.tensor([True, True, True])
    positions = torch.tensor([[0.0], [1.0], [2.0]])
    velocities = torch.zeros_like(positions)

    z_global, mu_global, sigma_global = patched_standardization(
        values, alive, return_statistics=True
    )
    z_local, mu_local, sigma_local = patched_standardization(
        values,
        alive,
        positions=positions,
        velocities=velocities,
        rho=1e-3,
        return_statistics=True,
    )

    assert torch.allclose(mu_local, mu_global.expand_as(mu_local))
    assert torch.allclose(sigma_local, sigma_global.expand_as(sigma_local))
    assert torch.any(torch.abs(z_local) > 1e-3)


def test_cloning_events_with_local_rho():
    assert _run_single_step_cloning(rho=0.01) > 0


def test_cloning_events_mean_field():
    assert _run_single_step_cloning(rho=None) > 0
