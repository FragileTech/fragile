import torch

from fragile.fractalai.core.kinetic_operator import KineticOperator


def test_viscous_neighbor_penalty_reduces_force():
    x = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.0, 0.2],
        ],
        dtype=torch.float64,
    )
    v = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    base = dict(
        gamma=1.0,
        beta=1.0,
        delta_t=0.1,
        nu=1.0,
        use_viscous_coupling=True,
        use_potential_force=False,
        viscous_length_scale=1.0,
        viscous_neighbor_threshold=0.5,
    )

    kin_no = KineticOperator(**base, viscous_neighbor_penalty=0.0)
    kin_pen = KineticOperator(**base, viscous_neighbor_penalty=1.0)

    force_no = kin_no._compute_viscous_force(x, v)
    force_pen = kin_pen._compute_viscous_force(x, v)

    assert torch.norm(force_pen) < torch.norm(force_no)
