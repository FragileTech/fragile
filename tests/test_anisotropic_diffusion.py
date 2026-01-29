from collections.abc import Callable

import pytest
import torch

from fragile.fractalai.bounds import TorchBounds
from fragile.fractalai.core.cloning import CloneOperator
from fragile.fractalai.core.companion_selection import CompanionSelection
from fragile.fractalai.core.euclidean_gas import EuclideanGas
from fragile.fractalai.core.fitness import FitnessOperator
from fragile.fractalai.core.kinetic_operator import KineticOperator


def _quadratic_potential(x: torch.Tensor) -> torch.Tensor:
    return (x**2).sum(dim=-1)


def _constant_potential(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)


def _build_gas(
    diagonal_diffusion: bool,
    potential_fn: Callable[[torch.Tensor], torch.Tensor] = _quadratic_potential,
) -> EuclideanGas:
    n_walkers = 6
    dims = 2
    bounds_extent = 3.0
    low = torch.full((dims,), -bounds_extent, dtype=torch.float32)
    high = torch.full((dims,), bounds_extent, dtype=torch.float32)
    bounds = TorchBounds(low=low, high=high)

    kinetic = KineticOperator(
        gamma=0.2,
        beta=1.0,
        delta_t=0.05,
        use_potential_force=True,
        use_anisotropic_diffusion=True,
        diagonal_diffusion=diagonal_diffusion,
        epsilon_Sigma=1e-4,
        epsilon_F=0.0,
        use_fitness_force=False,
    )
    kinetic.potential = potential_fn
    kinetic.bounds = bounds
    kinetic.pbc = False

    companion_selection = CompanionSelection(method="softmax", epsilon=1.0, lambda_alg=0.2)
    cloning = CloneOperator(
        sigma_x=0.1,
        alpha_restitution=0.5,
        p_max=1.0,
        epsilon_clone=0.01,
    )
    fitness = FitnessOperator(
        alpha=1.0,
        beta=1.0,
        eta=1e-3,
        lambda_alg=0.0,
        sigma_min=1e-6,
        A=2.0,
    )

    return EuclideanGas(
        N=n_walkers,
        d=dims,
        companion_selection=companion_selection,
        potential=potential_fn,
        kinetic_op=kinetic,
        cloning=cloning,
        fitness_op=fitness,
        bounds=bounds,
        device=torch.device("cpu"),
        dtype="float32",
        freeze_best=False,
        enable_cloning=False,
        enable_kinetic=True,
        pbc=False,
    )


@pytest.mark.parametrize("diagonal_diffusion", [True, False])
def test_anisotropic_diffusion_no_nan(diagonal_diffusion: bool) -> None:
    torch.manual_seed(0)
    gas = _build_gas(diagonal_diffusion)
    history = gas.run(n_steps=4, record_every=1, seed=0)

    assert torch.isfinite(history.noise).all()
    assert torch.isfinite(history.v_final).all()

    if diagonal_diffusion:
        assert history.sigma_reg_diag is not None
        assert history.fitness_hessians_diag is not None
        assert torch.isfinite(history.sigma_reg_diag).all()
        assert torch.isfinite(history.fitness_hessians_diag).all()
        assert history.sigma_reg_full is None
    else:
        assert history.sigma_reg_full is not None
        assert history.fitness_hessians_full is not None
        assert torch.isfinite(history.sigma_reg_full).all()
        assert torch.isfinite(history.fitness_hessians_full).all()
        assert history.sigma_reg_diag is None


def test_constant_potential_runs_without_grad_error() -> None:
    torch.manual_seed(0)
    gas = _build_gas(diagonal_diffusion=True, potential_fn=_constant_potential)
    history = gas.run(n_steps=3, record_every=1, seed=0)
    assert torch.isfinite(history.force_stable).all()
    assert torch.allclose(history.force_stable, torch.zeros_like(history.force_stable))
