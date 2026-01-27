import json

import torch

from fragile.fractalai.qft import simulation as qft_simulation


def test_qft_simulation_module_runs_and_saves(tmp_path):
    potential_cfg = qft_simulation.PotentialWellConfig(dims=2, alpha=0.2, bounds_extent=2.0)
    operator_cfg = qft_simulation.OperatorConfig()
    run_cfg = qft_simulation.RunConfig(
        N=8,
        n_steps=3,
        record_every=1,
        seed=123,
        device="cpu",
        dtype="float32",
        record_rng_state=False,
    )

    history, potential = qft_simulation.run_simulation(
        potential_cfg,
        operator_cfg,
        run_cfg,
        show_progress=False,
    )

    assert history.N == run_cfg.N
    assert history.d == potential_cfg.dims
    assert history.n_steps == run_cfg.n_steps
    assert history.n_recorded >= 2

    zeros = torch.zeros((run_cfg.N, potential_cfg.dims))
    assert torch.allclose(potential(zeros), torch.zeros(run_cfg.N))

    paths = qft_simulation.save_outputs(
        history,
        tmp_path,
        "test_run",
        potential_cfg,
        operator_cfg,
        run_cfg,
    )

    assert paths["history"].exists()
    assert paths["summary"].exists()
    assert paths["metadata"].exists()

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["run"]["N"] == run_cfg.N
    assert metadata["potential"]["dims"] == potential_cfg.dims
