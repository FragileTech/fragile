"""Cloning donors must be alive for every companion selection scheme."""

from unittest.mock import Mock

import pytest
import torch

from fragile.fractalai.bounds import TorchBounds
from fragile.fractalai.core.cloning import clone_walkers, CloneOperator
from fragile.fractalai.core.companion_selection import CompanionSelection
from fragile.fractalai.core.euclidean_gas import EuclideanGas, SwarmState
from fragile.fractalai.core.fitness import FitnessOperator


METHODS = ["softmax", "uniform", "random_pairing", "cloning", "greedy_pairing"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "mask",
    [[True] * 5, [False, True, False, True, True], [False, False, True, False, False], [True]],
)
def test_companions_are_alive(method, mask):
    torch.manual_seed(17)
    alive = torch.tensor(mask)
    x = torch.arange(len(mask), dtype=torch.float32).unsqueeze(1)
    companions = CompanionSelection(method=method)(x, torch.zeros_like(x), alive)
    assert companions.shape == alive.shape
    assert alive[companions].all()
    if alive.sum() == 1:
        assert (companions == torch.where(alive)[0][0]).all()
    if method in {"random_pairing", "greedy_pairing"}:
        indices = torch.where(alive)[0]
        assert torch.equal(companions[companions[indices]], indices)


@pytest.mark.parametrize("method", METHODS)
def test_no_survivors_raises(method):
    x = torch.zeros(3, 1)
    with pytest.raises(ValueError, match="No alive walkers"):
        CompanionSelection(method=method)(x, x, torch.zeros(3, dtype=torch.bool))


def make_gas(method="uniform", clone_selector=None, cloning=None):
    return EuclideanGas(
        N=3,
        d=1,
        potential=lambda x: x.square().sum(dim=1),
        bounds=TorchBounds.from_tuples([(-1.0, 1.0)]),
        companion_selection=CompanionSelection(method=method),
        companion_selection_clone=clone_selector,
        cloning=cloning if cloning is not None else CloneOperator(sigma_x=0.0),
        fitness_op=FitnessOperator(),
        enable_kinetic=False,
        device=torch.device("cpu"),
        dtype="float32",
        pbc=False,
    )


@pytest.mark.parametrize("method", METHODS)
def test_single_survivor_revives_dead_walkers(method):
    state = SwarmState(torch.tensor([[2.0], [0.25], [-2.0]]), torch.zeros(3, 1))
    cloned, _, info = make_gas(method).step(state, return_info=True)
    assert torch.equal(info["companions_clone"], torch.ones(3, dtype=torch.long))
    assert info["will_clone"][[0, 2]].all()
    assert torch.equal(cloned.x, torch.full((3, 1), 0.25))


@pytest.mark.parametrize("donor", [0, -1, 3])
def test_custom_invalid_selector_raises_before_operator(donor):
    selector = Mock(return_value=torch.tensor([donor, 1, 1]))
    operator = Mock()
    gas = make_gas(clone_selector=selector, cloning=operator)
    state = SwarmState(torch.tensor([[2.0], [0.25], [-2.0]]), torch.zeros(3, 1))
    before = state.x.clone()
    with pytest.raises(ValueError, match="Cloning companion"):
        gas.step(state)
    operator.assert_not_called()
    assert torch.equal(state.x, before)


@pytest.mark.parametrize("donor", [0, -1, 3])
def test_direct_cloning_rejects_invalid_donors(donor):
    x = torch.tensor([[2.0], [0.25], [-2.0]])
    before = x.clone()
    with pytest.raises(ValueError, match="Cloning companion"):
        clone_walkers(
            x,
            torch.zeros_like(x),
            torch.ones(3),
            torch.tensor([donor, 1, 1]),
            torch.tensor([False, True, False]),
        )
    assert torch.equal(x, before)


def test_direct_cloning_rejects_extinction():
    with pytest.raises(ValueError, match="No alive walkers"):
        clone_walkers(
            torch.zeros(3, 1),
            torch.zeros(3, 1),
            torch.zeros(3),
            torch.arange(3),
            torch.zeros(3, dtype=torch.bool),
        )


@pytest.mark.parametrize("method", METHODS)
def test_dead_state_never_enters_selection_calculations(method, monkeypatch):
    from fragile.fractalai.core import companion_selection

    alive = torch.tensor([False, True, False, True, True])
    x = torch.tensor([[float("nan")], [0.0], [float("inf")], [0.5], [0.75]])
    v = x.clone()
    distance = Mock(wraps=companion_selection.compute_algorithmic_distance_matrix)
    monkeypatch.setattr(companion_selection, "compute_algorithmic_distance_matrix", distance)
    torch.manual_seed(19)
    selector = CompanionSelection(method=method, lambda_alg=1.0)
    companions = selector(x, v, alive)
    assert companions.shape == (5,)
    assert alive[companions].all()
    for call in distance.call_args_list:
        assert torch.equal(call.args[0], x[alive])
        assert torch.equal(call.args[1], v[alive])
    if method in {"softmax", "cloning", "greedy_pairing"}:
        distance.assert_called_once()
    else:
        distance.assert_not_called()
    # Dead coordinates cannot change either the survivor law or revival donors.
    x[~alive] = -123.0
    v[~alive] = 456.0
    torch.manual_seed(19)
    assert torch.equal(selector(x, v, alive), companions)
