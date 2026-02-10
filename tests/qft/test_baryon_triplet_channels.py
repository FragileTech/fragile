"""Tests for vectorized companion-triplet baryon observables."""

from __future__ import annotations

import numpy as np
import torch

from fragile.fractalai.qft.baryon_triplet_channels import (
    BaryonTripletCorrelatorConfig,
    compute_baryon_correlator_from_color,
    compute_companion_baryon_correlator,
    compute_triplet_coherence,
    compute_triplet_coherence_from_velocity,
    TripletCoherenceConfig,
)
from tests.qft.test_correlator_channels import MockRunHistory


def _det3_np(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (
        a[..., 0] * (b[..., 1] * c[..., 2] - b[..., 2] * c[..., 1])
        - a[..., 1] * (b[..., 0] * c[..., 2] - b[..., 2] * c[..., 0])
        + a[..., 2] * (b[..., 0] * c[..., 1] - b[..., 1] * c[..., 0])
    )


def _naive_baryon_correlator(
    color: torch.Tensor,
    color_valid: torch.Tensor,
    alive: torch.Tensor,
    companions_distance: torch.Tensor,
    companions_clone: torch.Tensor,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    color_np = color.detach().cpu().numpy()
    color_valid_np = color_valid.detach().cpu().numpy().astype(bool)
    alive_np = alive.detach().cpu().numpy().astype(bool)
    comp_d_np = companions_distance.detach().cpu().numpy()
    comp_c_np = companions_clone.detach().cpu().numpy()

    t_total, n, _ = color_np.shape
    lag_max = min(max_lag, max(0, t_total - 1))
    n_lags = max_lag + 1

    det_src = np.zeros((t_total, n), dtype=np.complex128)
    valid_src = np.zeros((t_total, n), dtype=bool)
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
            m = np.column_stack([color_np[t, i], color_np[t, j], color_np[t, k]])
            det_val = np.linalg.det(m)
            if not np.isfinite(det_val.real) or not np.isfinite(det_val.imag):
                continue
            if abs(det_val) <= 1e-12:
                continue
            det_src[t, i] = det_val
            valid_src[t, i] = True

    mean_b = det_src[valid_src].mean() if valid_src.any() else 0.0 + 0.0j
    src_centered = det_src - mean_b

    corr_raw = np.zeros(n_lags, dtype=np.float64)
    corr_conn = np.zeros(n_lags, dtype=np.float64)
    counts = np.zeros(n_lags, dtype=np.int64)

    for lag in range(lag_max + 1):
        s_len = t_total - lag
        raw_vals: list[float] = []
        conn_vals: list[float] = []
        for s in range(s_len):
            for i in range(n):
                if not valid_src[s, i]:
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
                m_sink = np.column_stack(
                    [color_np[t_sink, i], color_np[t_sink, j], color_np[t_sink, k]]
                )
                det_sink = np.linalg.det(m_sink)
                if not np.isfinite(det_sink.real) or not np.isfinite(det_sink.imag):
                    continue
                if abs(det_sink) <= 1e-12:
                    continue
                raw_vals.append(float(np.real(np.conjugate(det_src[s, i]) * det_sink)))
                conn_vals.append(
                    float(np.real(np.conjugate(src_centered[s, i]) * (det_sink - mean_b)))
                )
        counts[lag] = len(raw_vals)
        if raw_vals:
            corr_raw[lag] = float(np.mean(raw_vals))
            corr_conn[lag] = float(np.mean(conn_vals))

    return corr_raw, corr_conn, counts


def _naive_triplet_coherence(
    velocities: torch.Tensor,
    alive: torch.Tensor,
    companions_distance: torch.Tensor,
    companions_clone: torch.Tensor,
    max_hops: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    vel = velocities.detach().cpu().numpy()
    alive_np = alive.detach().cpu().numpy().astype(bool)
    comp_d = companions_distance.detach().cpu().numpy()
    comp_c = companions_clone.detach().cpu().numpy()

    t_total, n, _ = vel.shape
    coh = np.zeros(max_hops, dtype=np.float64)
    counts = np.zeros(max_hops, dtype=np.int64)

    for t in range(t_total):
        for i in range(n):
            j = int(comp_d[t, i])
            k = int(comp_c[t, i])
            if j < 0 or j >= n or k < 0 or k >= n:
                continue
            if i in {j, k} or j == k:
                continue
            if not (alive_np[t, i] and alive_np[t, j] and alive_np[t, k]):
                continue

            det0 = abs(np.linalg.det(np.column_stack([vel[t, i], vel[t, j], vel[t, k]])))
            if not np.isfinite(det0) or det0 <= eps:
                continue

            counts[0] += 1
            coh[0] += 1.0
            p1, p2, p3 = i, j, k
            path_valid = True

            for hop in range(1, max_hops):
                if not path_valid:
                    continue
                p1n = int(comp_d[t, p1]) if 0 <= p1 < n else -1
                p2n = int(comp_d[t, p2]) if 0 <= p2 < n else -1
                p3n = int(comp_d[t, p3]) if 0 <= p3 < n else -1
                p1, p2, p3 = p1n, p2n, p3n

                in_range = all(0 <= idx < n for idx in (p1, p2, p3))
                if not in_range:
                    path_valid = False
                    continue
                if p1 in {p2, p3} or p2 == p3:
                    continue
                if not (alive_np[t, p1] and alive_np[t, p2] and alive_np[t, p3]):
                    continue

                deth = abs(np.linalg.det(np.column_stack([vel[t, p1], vel[t, p2], vel[t, p3]])))
                if not np.isfinite(deth) or deth <= eps:
                    continue
                coh[hop] += deth / det0
                counts[hop] += 1

    if counts[0] > 0:
        coh[0] /= counts[0]
    for hop in range(1, max_hops):
        if counts[hop] > 0:
            coh[hop] /= counts[hop]
    return coh, counts


def test_baryon_correlator_constant_signal_connected_subtracts_mean() -> None:
    """Connected correlator should remove pure disconnected constant signals."""
    t_total = 6
    n = 4
    color = torch.zeros(t_total, n, 3, dtype=torch.complex64)
    color[:, 0, 0] = 1.0 + 0j
    color[:, 1, 1] = 1.0 + 0j
    color[:, 2, 2] = 1.0 + 0j
    color[:, 3, :] = torch.tensor([1.0, 1.0, 1.0], dtype=torch.complex64) / np.sqrt(3.0)

    alive = torch.ones(t_total, n, dtype=torch.bool)
    alive[:, 3] = False  # keep only one structurally valid source triplet
    color_valid = torch.ones(t_total, n, dtype=torch.bool)

    companions_distance = torch.tensor([[1, 1, 1, 1]] * t_total, dtype=torch.long)
    companions_clone = torch.tensor([[2, 2, 2, 2]] * t_total, dtype=torch.long)

    out_raw = compute_baryon_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=4,
        use_connected=False,
    )
    out_conn = compute_baryon_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=4,
        use_connected=True,
    )

    torch.testing.assert_close(out_raw.correlator[:5], torch.ones(5), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_conn.correlator[:5], torch.zeros(5), atol=1e-6, rtol=1e-6)
    assert out_raw.counts[0].item() == t_total
    assert out_raw.disconnected_contribution > 0.0


def test_baryon_correlator_matches_naive_reference() -> None:
    """Vectorized baryon correlator should match a naive reference implementation."""
    torch.manual_seed(7)
    t_total = 7
    n = 9
    color = torch.randn(t_total, n, 3) + 1j * torch.randn(t_total, n, 3)
    color = color.to(torch.complex64)
    color_valid = torch.rand(t_total, n) > 0.15
    alive = torch.rand(t_total, n) > 0.1
    companions_distance = torch.randint(-1, n + 1, (t_total, n), dtype=torch.long)
    companions_clone = torch.randint(-1, n + 1, (t_total, n), dtype=torch.long)

    max_lag = 5
    out = compute_baryon_correlator_from_color(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=max_lag,
        use_connected=True,
    )
    raw_ref, conn_ref, counts_ref = _naive_baryon_correlator(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=max_lag,
    )

    np.testing.assert_allclose(out.correlator_raw.cpu().numpy(), raw_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        out.correlator_connected.cpu().numpy(), conn_ref, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(out.counts.cpu().numpy(), counts_ref)


def test_triplet_coherence_matches_naive_reference() -> None:
    """Vectorized triplet coherence diagnostic should match naive reference."""
    torch.manual_seed(13)
    t_total = 6
    n = 8
    velocities = torch.randn(t_total, n, 3)
    alive = torch.rand(t_total, n) > 0.15
    companions_distance = torch.randint(-1, n + 1, (t_total, n), dtype=torch.long)
    companions_clone = torch.randint(-1, n + 1, (t_total, n), dtype=torch.long)
    max_hops = 6
    eps = 1e-12

    out = compute_triplet_coherence_from_velocity(
        velocities=velocities,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_hops=max_hops,
        eps=eps,
    )
    coh_ref, counts_ref = _naive_triplet_coherence(
        velocities=velocities,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_hops=max_hops,
        eps=eps,
    )

    np.testing.assert_allclose(out.coherence.cpu().numpy(), coh_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(out.counts.cpu().numpy(), counts_ref)


def test_history_wrappers_run_end_to_end() -> None:
    """History-based wrappers should produce valid output shapes."""
    history = MockRunHistory(N=32, d=3, n_recorded=26)

    baryon_cfg = BaryonTripletCorrelatorConfig(max_lag=8, use_connected=True)
    baryon_out = compute_companion_baryon_correlator(history, baryon_cfg)
    assert baryon_out.correlator.shape == (9,)
    assert baryon_out.counts.shape == (9,)
    assert baryon_out.triplet_counts_per_frame.ndim == 1
    assert len(baryon_out.frame_indices) == baryon_out.triplet_counts_per_frame.shape[0]

    coherence_cfg = TripletCoherenceConfig(max_hops=7)
    coherence_out = compute_triplet_coherence(history, coherence_cfg)
    assert coherence_out.coherence.shape == (7,)
    assert coherence_out.counts.shape == (7,)
    assert len(coherence_out.frame_indices) > 0
