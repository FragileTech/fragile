import json
import sys

import numpy as np

from fragile.fractalai.qft import analysis as qft_analysis, simulation as qft_simulation


def test_qft_analysis_module_pipeline(tmp_path, monkeypatch):
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

    history, _ = qft_simulation.run_simulation(
        potential_cfg,
        operator_cfg,
        run_cfg,
        show_progress=False,
    )

    paths = qft_simulation.save_outputs(
        history,
        tmp_path,
        "analysis_seed",
        potential_cfg,
        operator_cfg,
        run_cfg,
    )

    output_dir = tmp_path / "analysis"
    analysis_id = "unit_test"
    argv = [
        "analyze_fractal_gas_qft",
        "--history-path",
        str(paths["history"]),
        "--output-dir",
        str(output_dir),
        "--analysis-id",
        analysis_id,
        "--analysis-time-index",
        "1",
        "--correlation-bins",
        "10",
        "--correlation-r-max",
        "1.0",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    qft_analysis.main()

    metrics_path = output_dir / f"{analysis_id}_metrics.json"
    arrays_path = output_dir / f"{analysis_id}_arrays.npz"

    assert metrics_path.exists()
    assert arrays_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["history_path"] == str(paths["history"])
    assert metrics["analysis_time_index"] == 1
    assert "observables" in metrics
    assert "d_prime_correlation" in metrics["observables"]

    arrays = np.load(arrays_path)
    assert "d_prime_bins" in arrays
    assert "r_prime_bins" in arrays
