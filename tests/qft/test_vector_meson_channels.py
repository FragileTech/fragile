"""Tests for vectorized companion-pair vector meson channels."""

from __future__ import annotations

import numpy as np
import torch

from fragile.fractalai.qft.vector_meson_channels import (
    compute_companion_vector_meson_correlator,
    compute_vector_meson_correlator_from_color_positions,
    VectorMesonCorrelatorConfig,
)
from tests.qft.test_correlator_channels import MockRunHistory


def _naive_vector_correlators(
    color: torch.Tensor,
    color_valid: torch.Tensor,
    positions: torch.Tensor,
    alive: torch.Tensor,
    companions_distance: torch.Tensor,
    companions_clone: torch.Tensor,
    max_lag: int,
    pair_selection: str,
    eps: float = 1e-12,
    use_unit_displacement: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    color_np = color.detach().cpu().numpy()
    valid_np = color_valid.detach().cpu().numpy().astype(bool)
    pos_np = positions.detach().cpu().numpy()
    alive_np = alive.detach().cpu().numpy().astype(bool)
    comp_d = companions_distance.detach().cpu().numpy()
    comp_c = companions_clone.detach().cpu().numpy()

    t_total, n, _ = color_np.shape
    lag_max = min(max_lag, max(0, t_total - 1))
    n_lags = max_lag + 1

    vector_source = np.zeros((t_total, n, 2, 3), dtype=np.float64)
    axial_source = np.zeros((t_total, n, 2, 3), dtype=np.float64)
    source_valid = np.zeros((t_total, n, 2), dtype=bool)

    use_dist = pair_selection in {"distance", "both"}
    use_clone = pair_selection in {"clone", "both"}

    for t in range(t_total):
        for i in range(n):
            if use_dist:
                j = int(comp_d[t, i])
                if 0 <= j < n and j != i and alive_np[t, i] and alive_np[t, j]:
                    if valid_np[t, i] and valid_np[t, j]:
                        inner = np.vdot(color_np[t, i], color_np[t, j])
                        dx = pos_np[t, j] - pos_np[t, i]
                        if np.isfinite(inner.real) and np.isfinite(inner.imag) and np.all(np.isfinite(dx)):
                            if use_unit_displacement:
                                norm_dx = np.linalg.norm(dx)
                                if norm_dx <= eps:
                                    pass
                                else:
                                    dx = dx / norm_dx
                                    if abs(inner) > eps:
                                        vector_source[t, i, 0] = float(inner.real) * dx
                                        axial_source[t, i, 0] = float(inner.imag) * dx
                                        source_valid[t, i, 0] = True
                            elif abs(inner) > eps:
                                vector_source[t, i, 0] = float(inner.real) * dx
                                axial_source[t, i, 0] = float(inner.imag) * dx
                                source_valid[t, i, 0] = True
            if use_clone:
                j = int(comp_c[t, i])
                slot = 0 if pair_selection == "clone" else 1
                if 0 <= j < n and j != i and alive_np[t, i] and alive_np[t, j]:
                    if valid_np[t, i] and valid_np[t, j]:
                        inner = np.vdot(color_np[t, i], color_np[t, j])
                        dx = pos_np[t, j] - pos_np[t, i]
                        if np.isfinite(inner.real) and np.isfinite(inner.imag) and np.all(np.isfinite(dx)):
                            if use_unit_displacement:
                                norm_dx = np.linalg.norm(dx)
                                if norm_dx <= eps:
                                    pass
                                else:
                                    dx = dx / norm_dx
                                    if abs(inner) > eps:
                                        vector_source[t, i, slot] = float(inner.real) * dx
                                        axial_source[t, i, slot] = float(inner.imag) * dx
                                        source_valid[t, i, slot] = True
                            elif abs(inner) > eps:
                                vector_source[t, i, slot] = float(inner.real) * dx
                                axial_source[t, i, slot] = float(inner.imag) * dx
                                source_valid[t, i, slot] = True

    if source_valid.any():
        mean_vector = vector_source[source_valid].reshape(-1, 3).mean(axis=0)
        mean_axial = axial_source[source_valid].reshape(-1, 3).mean(axis=0)
    else:
        mean_vector = np.zeros(3, dtype=np.float64)
        mean_axial = np.zeros(3, dtype=np.float64)

    vector_raw = np.zeros(n_lags, dtype=np.float64)
    vector_conn = np.zeros(n_lags, dtype=np.float64)
    axial_raw = np.zeros(n_lags, dtype=np.float64)
    axial_conn = np.zeros(n_lags, dtype=np.float64)
    counts = np.zeros(n_lags, dtype=np.int64)

    for lag in range(lag_max + 1):
        source_len = t_total - lag
        raw_vec_vals: list[float] = []
        conn_vec_vals: list[float] = []
        raw_ax_vals: list[float] = []
        conn_ax_vals: list[float] = []

        for t in range(source_len):
            for i in range(n):
                companions: list[tuple[int, int]] = []
                if use_dist:
                    companions.append((int(comp_d[t, i]), 0))
                if use_clone:
                    slot = 0 if pair_selection == "clone" else 1
                    companions.append((int(comp_c[t, i]), slot))

                for j, slot in companions:
                    if j < 0 or j >= n or j == i:
                        continue
                    if not source_valid[t, i, slot]:
                        continue

                    tt = t + lag
                    if not (alive_np[tt, i] and alive_np[tt, j] and valid_np[tt, i] and valid_np[tt, j]):
                        continue

                    inner_sink = np.vdot(color_np[tt, i], color_np[tt, j])
                    dx_sink = pos_np[tt, j] - pos_np[tt, i]
                    if not np.isfinite(inner_sink.real) or not np.isfinite(inner_sink.imag):
                        continue
                    if not np.all(np.isfinite(dx_sink)):
                        continue
                    if use_unit_displacement:
                        norm_sink = np.linalg.norm(dx_sink)
                        if norm_sink <= eps:
                            continue
                        dx_sink = dx_sink / norm_sink
                    if abs(inner_sink) <= eps:
                        continue

                    vec_sink = float(inner_sink.real) * dx_sink
                    ax_sink = float(inner_sink.imag) * dx_sink
                    vec_src = vector_source[t, i, slot]
                    ax_src = axial_source[t, i, slot]

                    raw_vec_vals.append(float(np.dot(vec_src, vec_sink)))
                    conn_vec_vals.append(float(np.dot(vec_src - mean_vector, vec_sink - mean_vector)))
                    raw_ax_vals.append(float(np.dot(ax_src, ax_sink)))
                    conn_ax_vals.append(float(np.dot(ax_src - mean_axial, ax_sink - mean_axial)))

        counts[lag] = len(raw_vec_vals)
        if raw_vec_vals:
            vector_raw[lag] = float(np.mean(raw_vec_vals))
            vector_conn[lag] = float(np.mean(conn_vec_vals))
            axial_raw[lag] = float(np.mean(raw_ax_vals))
            axial_conn[lag] = float(np.mean(conn_ax_vals))

    return vector_raw, vector_conn, axial_raw, axial_conn, counts


def test_vector_meson_constant_signal_connected_subtracts_mean() -> None:
    t_total = 6
    n = 2
    color = torch.zeros(t_total, n, 3, dtype=torch.complex64)
    color[:, 0, 0] = 1.0 + 0j
    color[:, 1, 0] = 1.0 + 0j
    color_valid = torch.ones(t_total, n, dtype=torch.bool)
    alive = torch.ones(t_total, n, dtype=torch.bool)
    positions = torch.zeros(t_total, n, 3, dtype=torch.float32)
    positions[:, 1, 0] = 1.0
    comp_d = torch.tensor([[1, 1]] * t_total, dtype=torch.long)
    comp_c = torch.tensor([[1, 1]] * t_total, dtype=torch.long)

    out_raw = compute_vector_meson_correlator_from_color_positions(
        color=color,
        color_valid=color_valid,
        positions=positions,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=4,
        use_connected=False,
        pair_selection="distance",
    )
    out_conn = compute_vector_meson_correlator_from_color_positions(
        color=color,
        color_valid=color_valid,
        positions=positions,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=4,
        use_connected=True,
        pair_selection="distance",
    )

    torch.testing.assert_close(out_raw.vector[:5], torch.ones(5), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_conn.vector[:5], torch.zeros(5), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_raw.axial_vector[:5], torch.zeros(5), atol=1e-6, rtol=1e-6)
    assert out_raw.counts[0].item() > 0


def test_vector_meson_correlator_matches_naive_reference() -> None:
    torch.manual_seed(17)
    t_total = 7
    n = 9
    color = torch.randn(t_total, n, 3) + 1j * torch.randn(t_total, n, 3)
    color = color.to(torch.complex64)
    positions = torch.randn(t_total, n, 3)
    color_valid = torch.rand(t_total, n) > 0.1
    alive = torch.rand(t_total, n) > 0.1
    comp_d = torch.randint(-1, n + 1, (t_total, n), dtype=torch.long)
    comp_c = torch.randint(-1, n + 1, (t_total, n), dtype=torch.long)

    max_lag = 5
    out = compute_vector_meson_correlator_from_color_positions(
        color=color,
        color_valid=color_valid,
        positions=positions,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=max_lag,
        use_connected=True,
        pair_selection="both",
    )
    vec_raw_ref, vec_conn_ref, ax_raw_ref, ax_conn_ref, counts_ref = _naive_vector_correlators(
        color=color,
        color_valid=color_valid,
        positions=positions,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=max_lag,
        pair_selection="both",
    )

    np.testing.assert_allclose(out.vector_raw.cpu().numpy(), vec_raw_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out.vector_connected.cpu().numpy(), vec_conn_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out.axial_vector_raw.cpu().numpy(), ax_raw_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        out.axial_vector_connected.cpu().numpy(), ax_conn_ref, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(out.counts.cpu().numpy(), counts_ref)


def test_history_wrapper_runs_end_to_end() -> None:
    history = MockRunHistory(N=28, d=4, n_recorded=24)
    cfg = VectorMesonCorrelatorConfig(
        max_lag=8,
        use_connected=True,
        pair_selection="both",
        color_dims=(0, 1, 2),
        position_dims=(0, 1, 2),
    )
    out = compute_companion_vector_meson_correlator(history, cfg)
    assert out.vector.shape == (9,)
    assert out.axial_vector.shape == (9,)
    assert out.counts.shape == (9,)
    assert out.pair_counts_per_frame.ndim == 1
    assert out.operator_vector_series.ndim == 2
    assert out.operator_vector_series.shape[-1] == 3
    assert len(out.frame_indices) == out.pair_counts_per_frame.shape[0]


def test_vector_meson_score_directed_matches_standard_for_equal_scores() -> None:
    torch.manual_seed(31)
    t_total = 8
    n = 10
    color = (torch.randn(t_total, n, 3) + 1j * torch.randn(t_total, n, 3)).to(torch.complex64)
    positions = torch.randn(t_total, n, 3)
    color_valid = torch.rand(t_total, n) > 0.1
    alive = torch.rand(t_total, n) > 0.1
    comp_d = torch.randint(-1, n + 1, (t_total, n), dtype=torch.long)
    comp_c = torch.randint(-1, n + 1, (t_total, n), dtype=torch.long)
    scores_equal = torch.zeros(t_total, n, dtype=torch.float32)

    out_standard = compute_vector_meson_correlator_from_color_positions(
        color=color,
        color_valid=color_valid,
        positions=positions,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=5,
        use_connected=True,
        pair_selection="both",
        operator_mode="standard",
        projection_mode="full",
    )
    out_directed = compute_vector_meson_correlator_from_color_positions(
        color=color,
        color_valid=color_valid,
        positions=positions,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=5,
        use_connected=True,
        pair_selection="both",
        operator_mode="score_directed",
        projection_mode="full",
        scores=scores_equal,
    )

    torch.testing.assert_close(out_directed.vector_raw, out_standard.vector_raw, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        out_directed.vector_connected, out_standard.vector_connected, atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        out_directed.axial_vector_raw, out_standard.axial_vector_raw, atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        out_directed.axial_vector_connected,
        out_standard.axial_vector_connected,
        atol=1e-6,
        rtol=1e-5,
    )


def test_vector_meson_score_directed_projection_modes_are_finite() -> None:
    torch.manual_seed(32)
    t_total = 8
    n = 10
    color = (torch.randn(t_total, n, 3) + 1j * torch.randn(t_total, n, 3)).to(torch.complex64)
    positions = torch.randn(t_total, n, 3)
    color_valid = torch.rand(t_total, n) > 0.1
    alive = torch.rand(t_total, n) > 0.1
    comp_d = torch.randint(-1, n + 1, (t_total, n), dtype=torch.long)
    comp_c = torch.randint(-1, n + 1, (t_total, n), dtype=torch.long)
    scores = torch.randn(t_total, n, dtype=torch.float32)

    for projection_mode in ("longitudinal", "transverse"):
        out = compute_vector_meson_correlator_from_color_positions(
            color=color,
            color_valid=color_valid,
            positions=positions,
            alive=alive,
            companions_distance=comp_d,
            companions_clone=comp_c,
            max_lag=5,
            use_connected=True,
            pair_selection="both",
            operator_mode="score_directed",
            projection_mode=projection_mode,
            scores=scores,
        )
        assert out.vector.shape == (6,)
        assert out.axial_vector.shape == (6,)
        assert bool(torch.isfinite(out.vector).all())
        assert bool(torch.isfinite(out.axial_vector).all())
