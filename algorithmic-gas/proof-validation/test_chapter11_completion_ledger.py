"""Regressions for chapter11 quantization, transport and trajectory uncertainty."""

from pathlib import Path

from chapter11_completion_ledger import diagnostic_curves, h2, metrics, w2
import numpy as np
import pytest


def test_exact_transport_is_normalized_and_permutation_invariant():
    x = np.array([-1.0, 0.0, 2.0])
    p = np.array([0.25, 0.5, 0.25])
    y = np.array([0.0, 1.0])
    q = np.array([0.5, 0.5])
    assert w2(x, p, y, q) == pytest.approx(0.75)
    assert w2(x[::-1], p[::-1], y[::-1], q[::-1]) == pytest.approx(0.75)


def test_zero_mass_keeps_mass_metric_and_does_not_invent_shape():
    p = np.zeros(12)
    q = np.array([0.1, 0.2, 0.7] + [0.0] * 9)
    row = metrics(p, q)
    assert row["grid_H_squared"] == pytest.approx(1.0)
    assert row["root_mass_squared"] == pytest.approx(1.0)
    assert "grid_shape_H_squared" not in row
    assert "grid_W2_squared" not in row
    assert h2(p, p) == pytest.approx(0.0)


def test_common_stream_bootstrap_resamples_entire_native_pairs(tmp_path):
    def make(seed, initial):
        common = [0.5, 0.5] + [0.0] * 10
        return {
            "seed": seed,
            "h": 0.1,
            "N": 8,
            "d": 1,
            "initial": {"bins": initial, "projected_coordinates": [0.0, 1.0]},
            "observations": [
                {
                    "physical_time": 0.1,
                    "observation": {"bins": common, "projected_coordinates": [0.0, 1.0]},
                },
                {
                    "physical_time": 0.2,
                    "observation": {"bins": common, "projected_coordinates": [0.0, 1.0]},
                },
            ],
            "terminal": {"mass": 1.0},
        }

    groups = {
        "example/left": [make(i, [0.2, 0.8] + [0.0] * 10) for i in range(4)],
        "example/right": [make(i, [0.8, 0.2] + [0.0] * 10) for i in range(4)],
    }
    curves, manifest = diagnostic_curves(groups, tmp_path, 20)
    assert len(curves) == len(manifest) == 1
    assert curves[0]["common_intrinsic_seed_stream"]
    for row in curves[0]["metrics"]:
        assert row["values"][-1] == pytest.approx(0.0)
        assert row["physical_time_log_linear_decay_rate"] is None
        assert row["pointwise_bootstrap_95_intervals"][-1] == [0.0, 0.0]


def test_actual_survival_calibration_uses_whole_trajectory_variance(tmp_path):
    from chapter11_completion_ledger import survival_calibration

    conditional = []
    runs = []
    for seed in range(4):
        alive = 0.25 if seed % 2 else 0.75
        runs.append({
            "group": "survival",
            "seed": seed,
            "N": 4,
            "d": 2,
            "derived_execution_root": "test",
            "declared_planned_horizon": 1,
            "observations": [{}],
            "first_killing_transition_step": None,
        })
        conditional.append({
            "group": "survival",
            "seed": seed,
            "frames": [
                {
                    "step": 1,
                    "N": 4,
                    "d": 2,
                    "conditional_mean": 0.5,
                    "conditional_variance": 0.0625,
                    "actual_alive_fraction": alive,
                    "centered_mass_innovation": alive - 0.5,
                    "conditioning": "B1_input",
                }
            ],
        })
    rows, manifest = survival_calibration(
        conditional, {"survival": runs}, tmp_path, 30, Path("test"), 1
    )
    assert len(rows) == len(manifest) == 1
    assert rows[0]["actual_mean_innovation_sum"] == pytest.approx(0.0)
    assert rows[0]["actual_to_predicted_second_moment_ratio"] == pytest.approx(1.0)
    assert rows[0]["conditional_Hoeffding_passed"]
    assert rows[0]["variance_sensitive_mean_passed"]


def test_vectorized_pooled_transport_matches_exact_weighted_monotone_coupling():
    from chapter11_exact_marginal_transport import uniform_transport

    x = np.array([-1.0, 0.0, 2.0])
    y = np.array([0.0, 1.0])
    reference = w2(x, np.full(3, 1 / 3), y, np.full(2, 1 / 2))
    assert uniform_transport(x, y) == pytest.approx(reference)
    assert uniform_transport(x[::-1], y[::-1]) == pytest.approx(reference)
    assert uniform_transport(np.repeat(x, 2), np.repeat(y, 3)) == pytest.approx(reference)
    assert uniform_transport([], y) is None
