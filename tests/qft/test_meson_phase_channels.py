"""Tests for vectorized companion-pair meson phase channels."""

from __future__ import annotations

import numpy as np
import torch

from fragile.fractalai.qft.meson_phase_channels import (
    build_companion_pair_indices,
    compute_companion_meson_phase_correlator,
    compute_meson_phase_correlator_from_color,
    MesonPhaseCorrelatorConfig,
)
from tests.qft.test_correlator_channels import MockRunHistory


def _naive_meson_correlators(
    color: torch.Tensor,
    color_valid: torch.Tensor,
    alive: torch.Tensor,
    companions_distance: torch.Tensor,
    companions_clone: torch.Tensor,
    max_lag: int,
    pair_selection: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    color_np = color.detach().cpu().numpy()
    valid_np = color_valid.detach().cpu().numpy().astype(bool)
    alive_np = alive.detach().cpu().numpy().astype(bool)
    comp_d = companions_distance.detach().cpu().numpy()
    comp_c = companions_clone.detach().cpu().numpy()

    t_total, n, _ = color_np.shape
    lag_max = min(max_lag, max(0, t_total - 1))
    n_lags = max_lag + 1

    ps_source = np.zeros((t_total, n, 2), dtype=np.float64)
    s_source = np.zeros((t_total, n, 2), dtype=np.float64)
    source_valid = np.zeros((t_total, n, 2), dtype=bool)

    use_dist = pair_selection in {"distance", "both"}
    use_clone = pair_selection in {"clone", "both"}
    for t in range(t_total):
        for i in range(n):
            if use_dist:
                j = int(comp_d[t, i])
                if 0 <= j < n and j != i:
                    if alive_np[t, i] and alive_np[t, j] and valid_np[t, i] and valid_np[t, j]:
                        inner = np.vdot(color_np[t, i], color_np[t, j])
                        if np.isfinite(inner.real) and np.isfinite(inner.imag):
                            ps_source[t, i, 0] = float(inner.imag)
                            s_source[t, i, 0] = float(inner.real)
                            source_valid[t, i, 0] = abs(inner) > 1e-12
            if use_clone:
                k = int(comp_c[t, i])
                slot = 0 if pair_selection == "clone" else 1
                if 0 <= k < n and k != i:
                    if alive_np[t, i] and alive_np[t, k] and valid_np[t, i] and valid_np[t, k]:
                        inner = np.vdot(color_np[t, i], color_np[t, k])
                        if np.isfinite(inner.real) and np.isfinite(inner.imag):
                            ps_source[t, i, slot] = float(inner.imag)
                            s_source[t, i, slot] = float(inner.real)
                            source_valid[t, i, slot] = abs(inner) > 1e-12

    mean_ps = ps_source[source_valid].mean() if source_valid.any() else 0.0
    mean_s = s_source[source_valid].mean() if source_valid.any() else 0.0

    ps_raw = np.zeros(n_lags, dtype=np.float64)
    ps_conn = np.zeros(n_lags, dtype=np.float64)
    s_raw = np.zeros(n_lags, dtype=np.float64)
    s_conn = np.zeros(n_lags, dtype=np.float64)
    counts = np.zeros(n_lags, dtype=np.int64)

    for lag in range(lag_max + 1):
        src_len = t_total - lag
        raw_ps_vals: list[float] = []
        conn_ps_vals: list[float] = []
        raw_s_vals: list[float] = []
        conn_s_vals: list[float] = []

        for t in range(src_len):
            for i in range(n):
                companions = []
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
                    if not (
                        alive_np[tt, i] and alive_np[tt, j] and valid_np[tt, i] and valid_np[tt, j]
                    ):
                        continue
                    inner_sink = np.vdot(color_np[tt, i], color_np[tt, j])
                    if not np.isfinite(inner_sink.real) or not np.isfinite(inner_sink.imag):
                        continue
                    if abs(inner_sink) <= 1e-12:
                        continue
                    ps_sink = float(inner_sink.imag)
                    s_sink = float(inner_sink.real)
                    ps_src = ps_source[t, i, slot]
                    s_src = s_source[t, i, slot]
                    raw_ps_vals.append(ps_src * ps_sink)
                    conn_ps_vals.append((ps_src - mean_ps) * (ps_sink - mean_ps))
                    raw_s_vals.append(s_src * s_sink)
                    conn_s_vals.append((s_src - mean_s) * (s_sink - mean_s))

        counts[lag] = len(raw_ps_vals)
        if raw_ps_vals:
            ps_raw[lag] = float(np.mean(raw_ps_vals))
            ps_conn[lag] = float(np.mean(conn_ps_vals))
            s_raw[lag] = float(np.mean(raw_s_vals))
            s_conn[lag] = float(np.mean(conn_s_vals))

    return ps_raw, ps_conn, s_raw, s_conn, counts


def test_build_companion_pair_indices_shapes() -> None:
    t_total = 5
    n = 7
    comp_d = torch.randint(0, n, (t_total, n), dtype=torch.long)
    comp_c = torch.randint(0, n, (t_total, n), dtype=torch.long)

    pair_idx_both, valid_both = build_companion_pair_indices(comp_d, comp_c, "both")
    assert pair_idx_both.shape == (t_total, n, 2)
    assert valid_both.shape == (t_total, n, 2)

    pair_idx_dist, valid_dist = build_companion_pair_indices(comp_d, comp_c, "distance")
    assert pair_idx_dist.shape == (t_total, n, 1)
    assert valid_dist.shape == (t_total, n, 1)

    pair_idx_clone, valid_clone = build_companion_pair_indices(comp_d, comp_c, "clone")
    assert pair_idx_clone.shape == (t_total, n, 1)
    assert valid_clone.shape == (t_total, n, 1)


def test_meson_correlator_constant_signal_connected_subtracts_mean() -> None:
    t_total = 6
    n = 4
    color = torch.zeros(t_total, n, 3, dtype=torch.complex64)
    color[:, 0, 0] = 1.0 + 0j
    color[:, 1, 0] = 1.0 + 0j
    color[:, 2, 0] = 1.0 + 0j
    color[:, 3, 0] = 1.0 + 0j
    color_valid = torch.ones(t_total, n, dtype=torch.bool)
    alive = torch.ones(t_total, n, dtype=torch.bool)
    comp_d = torch.tensor([[1, 0, 3, 2]] * t_total, dtype=torch.long)
    comp_c = torch.tensor([[1, 0, 3, 2]] * t_total, dtype=torch.long)

    out_raw = compute_meson_phase_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=4,
        use_connected=False,
        pair_selection="distance",
    )
    out_conn = compute_meson_phase_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=4,
        use_connected=True,
        pair_selection="distance",
    )

    torch.testing.assert_close(out_raw.scalar[:5], torch.ones(5), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_conn.scalar[:5], torch.zeros(5), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_raw.pseudoscalar[:5], torch.zeros(5), atol=1e-6, rtol=1e-6)
    assert out_raw.counts[0].item() > 0


def test_meson_correlator_abs2_vacsub_connected_subtracts_constant_signal() -> None:
    t_total = 6
    n = 4
    color = torch.zeros(t_total, n, 3, dtype=torch.complex64)
    color[:, 0, 0] = 1.0 + 0j
    color[:, 1, 0] = 1.0 + 0j
    color[:, 2, 0] = 1.0 + 0j
    color[:, 3, 0] = 1.0 + 0j
    color_valid = torch.ones(t_total, n, dtype=torch.bool)
    alive = torch.ones(t_total, n, dtype=torch.bool)
    comp_d = torch.tensor([[1, 0, 3, 2]] * t_total, dtype=torch.long)
    comp_c = torch.tensor([[1, 0, 3, 2]] * t_total, dtype=torch.long)

    out_raw = compute_meson_phase_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=4,
        use_connected=False,
        pair_selection="distance",
        operator_mode="abs2_vacsub",
    )
    out_conn = compute_meson_phase_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=4,
        use_connected=True,
        pair_selection="distance",
        operator_mode="abs2_vacsub",
    )

    torch.testing.assert_close(out_raw.scalar[:5], torch.ones(5), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_conn.scalar[:5], torch.zeros(5), atol=1e-6, rtol=1e-6)
    assert bool((out_raw.scalar[:5] >= 0).all())
    assert out_raw.counts[0].item() > 0


def test_meson_correlator_matches_naive_reference() -> None:
    torch.manual_seed(11)
    t_total = 7
    n = 9
    color = torch.randn(t_total, n, 3) + 1j * torch.randn(t_total, n, 3)
    color = color.to(torch.complex64)
    color_valid = torch.rand(t_total, n) > 0.1
    alive = torch.rand(t_total, n) > 0.1
    comp_d = torch.randint(-1, n + 1, (t_total, n), dtype=torch.long)
    comp_c = torch.randint(-1, n + 1, (t_total, n), dtype=torch.long)

    max_lag = 5
    out = compute_meson_phase_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=max_lag,
        use_connected=True,
        pair_selection="both",
    )
    ps_raw_ref, ps_conn_ref, s_raw_ref, s_conn_ref, counts_ref = _naive_meson_correlators(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=max_lag,
        pair_selection="both",
    )

    np.testing.assert_allclose(
        out.pseudoscalar_raw.cpu().numpy(), ps_raw_ref, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        out.pseudoscalar_connected.cpu().numpy(), ps_conn_ref, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(out.scalar_raw.cpu().numpy(), s_raw_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        out.scalar_connected.cpu().numpy(), s_conn_ref, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(out.counts.cpu().numpy(), counts_ref)


def test_history_wrapper_runs_end_to_end() -> None:
    history = MockRunHistory(N=28, d=3, n_recorded=24)
    cfg = MesonPhaseCorrelatorConfig(max_lag=8, use_connected=True, pair_selection="both")
    out = compute_companion_meson_phase_correlator(history, cfg)
    assert out.scalar.shape == (9,)
    assert out.pseudoscalar.shape == (9,)
    assert out.counts.shape == (9,)
    assert out.pair_counts_per_frame.ndim == 1
    assert len(out.frame_indices) == out.pair_counts_per_frame.shape[0]
