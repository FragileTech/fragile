"""Tests for vectorized companion-triplet color glueball channels."""

from __future__ import annotations

import numpy as np
import torch

from fragile.fractalai.qft.glueball_color_channels import (
    compute_companion_glueball_color_correlator,
    compute_glueball_color_correlator_from_color,
    GlueballColorCorrelatorConfig,
)
from tests.qft.test_correlator_channels import MockRunHistory


def _naive_glueball_correlator(
    color: torch.Tensor,
    color_valid: torch.Tensor,
    alive: torch.Tensor,
    companions_distance: torch.Tensor,
    companions_clone: torch.Tensor,
    max_lag: int,
    eps: float,
    use_action_form: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    color_np = color.detach().cpu().numpy()
    color_valid_np = color_valid.detach().cpu().numpy().astype(bool)
    alive_np = alive.detach().cpu().numpy().astype(bool)
    comp_d_np = companions_distance.detach().cpu().numpy()
    comp_c_np = companions_clone.detach().cpu().numpy()

    t_total, n, _ = color_np.shape
    lag_max = min(max_lag, max(0, t_total - 1))
    n_lags = max_lag + 1

    source_obs = np.zeros((t_total, n), dtype=np.float64)
    source_valid = np.zeros((t_total, n), dtype=bool)
    for t in range(t_total):
        for i in range(n):
            j = int(comp_d_np[t, i])
            k = int(comp_c_np[t, i])
            if j < 0 or j >= n or k < 0 or k >= n:
                continue
            if i in {j, k} or j == k:
                continue
            if not (
                alive_np[t, i]
                and alive_np[t, j]
                and alive_np[t, k]
                and color_valid_np[t, i]
                and color_valid_np[t, j]
                and color_valid_np[t, k]
            ):
                continue
            z_ij = np.vdot(color_np[t, i], color_np[t, j])
            z_jk = np.vdot(color_np[t, j], color_np[t, k])
            z_ki = np.vdot(color_np[t, k], color_np[t, i])
            finite_triplet = np.isfinite(
                np.array(
                    [z_ij.real, z_ij.imag, z_jk.real, z_jk.imag, z_ki.real, z_ki.imag],
                    dtype=np.float64,
                )
            ).all()
            if not finite_triplet:
                continue
            if abs(z_ij) <= eps or abs(z_jk) <= eps or abs(z_ki) <= eps:
                continue
            pi = z_ij * z_jk * z_ki
            obs = float(np.real(pi))
            if use_action_form:
                obs = 1.0 - obs
            source_obs[t, i] = obs
            source_valid[t, i] = True

    mean_obs = source_obs[source_valid].mean() if source_valid.any() else 0.0

    corr_raw = np.zeros(n_lags, dtype=np.float64)
    corr_conn = np.zeros(n_lags, dtype=np.float64)
    counts = np.zeros(n_lags, dtype=np.int64)
    for lag in range(lag_max + 1):
        source_len = t_total - lag
        raw_vals: list[float] = []
        conn_vals: list[float] = []
        for s in range(source_len):
            for i in range(n):
                if not source_valid[s, i]:
                    continue
                j = int(comp_d_np[s, i])
                k = int(comp_c_np[s, i])
                if j < 0 or j >= n or k < 0 or k >= n:
                    continue
                if i in {j, k} or j == k:
                    continue
                t_sink = s + lag
                if not (
                    alive_np[t_sink, i]
                    and alive_np[t_sink, j]
                    and alive_np[t_sink, k]
                    and color_valid_np[t_sink, i]
                    and color_valid_np[t_sink, j]
                    and color_valid_np[t_sink, k]
                ):
                    continue
                z_ij = np.vdot(color_np[t_sink, i], color_np[t_sink, j])
                z_jk = np.vdot(color_np[t_sink, j], color_np[t_sink, k])
                z_ki = np.vdot(color_np[t_sink, k], color_np[t_sink, i])
                finite_triplet = np.isfinite(
                    np.array(
                        [z_ij.real, z_ij.imag, z_jk.real, z_jk.imag, z_ki.real, z_ki.imag],
                        dtype=np.float64,
                    )
                ).all()
                if not finite_triplet:
                    continue
                if abs(z_ij) <= eps or abs(z_jk) <= eps or abs(z_ki) <= eps:
                    continue
                pi_sink = z_ij * z_jk * z_ki
                sink_obs = float(np.real(pi_sink))
                if use_action_form:
                    sink_obs = 1.0 - sink_obs
                src_obs = source_obs[s, i]
                raw_vals.append(src_obs * sink_obs)
                conn_vals.append((src_obs - mean_obs) * (sink_obs - mean_obs))
        counts[lag] = len(raw_vals)
        if raw_vals:
            corr_raw[lag] = float(np.mean(raw_vals))
            corr_conn[lag] = float(np.mean(conn_vals))

    return corr_raw, corr_conn, counts


def test_glueball_correlator_constant_signal_connected_subtracts_mean() -> None:
    t_total = 6
    n = 4
    color = torch.zeros(t_total, n, 3, dtype=torch.complex64)
    color[:, 0, 0] = 1.0 + 0j
    color[:, 1, 0] = 1.0 + 0j
    color[:, 2, 0] = 1.0 + 0j
    color[:, 3, 0] = 1.0 + 0j
    color_valid = torch.ones(t_total, n, dtype=torch.bool)
    alive = torch.ones(t_total, n, dtype=torch.bool)
    alive[:, 3] = False
    comp_d = torch.tensor([[1, 1, 1, 1]] * t_total, dtype=torch.long)
    comp_c = torch.tensor([[2, 2, 2, 2]] * t_total, dtype=torch.long)

    out_raw = compute_glueball_color_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=4,
        use_connected=False,
    )
    out_conn = compute_glueball_color_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        max_lag=4,
        use_connected=True,
    )

    torch.testing.assert_close(out_raw.correlator[:5], torch.ones(5), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_conn.correlator[:5], torch.zeros(5), atol=1e-6, rtol=1e-6)
    assert out_raw.counts[0].item() == t_total


def test_glueball_correlator_matches_naive_reference() -> None:
    torch.manual_seed(23)
    t_total = 7
    n = 9
    color = torch.randn(t_total, n, 3) + 1j * torch.randn(t_total, n, 3)
    color = color.to(torch.complex64)
    color_valid = torch.rand(t_total, n) > 0.15
    alive = torch.rand(t_total, n) > 0.1
    companions_distance = torch.randint(-1, n + 1, (t_total, n), dtype=torch.long)
    companions_clone = torch.randint(-1, n + 1, (t_total, n), dtype=torch.long)

    max_lag = 5
    eps = 1e-12
    out = compute_glueball_color_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=max_lag,
        use_connected=True,
        eps=eps,
        use_action_form=False,
    )
    raw_ref, conn_ref, counts_ref = _naive_glueball_correlator(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=max_lag,
        eps=eps,
        use_action_form=False,
    )

    np.testing.assert_allclose(out.correlator_raw.cpu().numpy(), raw_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        out.correlator_connected.cpu().numpy(), conn_ref, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(out.counts.cpu().numpy(), counts_ref)


def test_glueball_action_form_matches_naive_reference() -> None:
    torch.manual_seed(37)
    t_total = 6
    n = 8
    color = torch.randn(t_total, n, 3) + 1j * torch.randn(t_total, n, 3)
    color = color.to(torch.complex64)
    color_valid = torch.rand(t_total, n) > 0.1
    alive = torch.rand(t_total, n) > 0.1
    companions_distance = torch.randint(-1, n + 1, (t_total, n), dtype=torch.long)
    companions_clone = torch.randint(-1, n + 1, (t_total, n), dtype=torch.long)

    max_lag = 4
    eps = 1e-12
    out = compute_glueball_color_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=max_lag,
        use_connected=False,
        eps=eps,
        use_action_form=True,
    )
    raw_ref, _, counts_ref = _naive_glueball_correlator(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=max_lag,
        eps=eps,
        use_action_form=True,
    )
    np.testing.assert_allclose(out.correlator.cpu().numpy(), raw_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(out.counts.cpu().numpy(), counts_ref)


def test_history_wrapper_runs_end_to_end() -> None:
    history = MockRunHistory(N=32, d=3, n_recorded=26)
    cfg = GlueballColorCorrelatorConfig(max_lag=8, use_connected=True)
    out = compute_companion_glueball_color_correlator(history, cfg)
    assert out.correlator.shape == (9,)
    assert out.counts.shape == (9,)
    assert out.triplet_counts_per_frame.ndim == 1
    assert len(out.frame_indices) == out.triplet_counts_per_frame.shape[0]
