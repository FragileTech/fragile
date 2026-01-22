import pytest
import torch

from fragile.core.optimizers import ThermodynamicAdam


def test_thermodynamic_adam_decreases_quadratic() -> None:
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = ThermodynamicAdam(
        [param],
        lr=0.05,
        governor_sensitivity=0.0,
        oscillation_brake=1.0,
        cosine_anneal=False,
    )

    def loss_fn() -> torch.Tensor:
        return 0.5 * (param**2).sum()

    loss_start = loss_fn().item()
    for _ in range(25):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
    loss_end = loss_fn().item()

    assert loss_end < loss_start


def test_thermodynamic_adam_brake_applies() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = ThermodynamicAdam(
        [param],
        lr=0.1,
        governor_sensitivity=0.0,
        oscillation_brake=0.2,
        cosine_anneal=False,
    )
    param.grad = torch.tensor([1.0])
    state = optimizer.state[param]
    state["exp_avg"] = torch.tensor([-1.0])
    state["exp_avg_sq"] = torch.tensor([1.0])
    state["step"] = 1

    optimizer.step()

    assert optimizer.last_lr_scale == pytest.approx(0.2, rel=1e-6)


def test_thermodynamic_adam_varentropy_scales_lr() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = ThermodynamicAdam(
        [param],
        lr=0.1,
        governor_sensitivity=5.0,
        history_window=3,
        varentropy_min_history=2,
        oscillation_brake=1.0,
        max_lr_scale=10.0,
        cosine_anneal=False,
    )

    param.grad = torch.tensor([1.0])
    optimizer.step()
    param.grad = torch.tensor([3.0])
    optimizer.step()

    assert optimizer.last_lr_scale > 1.0
    assert optimizer.last_lr == pytest.approx(
        optimizer.param_groups[0]["lr"] * optimizer.last_lr_scale,
        rel=1e-6,
    )


def test_thermodynamic_adam_loss_varentropy_path() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = ThermodynamicAdam(
        [param],
        lr=0.1,
        governor_sensitivity=5.0,
        history_window=3,
        varentropy_min_history=2,
        oscillation_brake=1.0,
        use_loss_varentropy=True,
        cosine_anneal=False,
    )

    param.grad = torch.tensor([1.0])
    optimizer.step(loss=torch.tensor(1.0))
    param.grad = torch.tensor([1.0])
    optimizer.step(loss=torch.tensor(3.0))

    assert optimizer.last_lr_scale > 1.0
