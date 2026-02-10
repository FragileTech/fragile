"""Einstein equation verification on the fractal gas lattice.

Computes the full Riemann -> Ricci -> Einstein tensor pipeline from
pre-computed metric tensors and neighbor graphs stored in RunHistory,
then tests G_uv + Lambda*g_uv = 8*pi*G_N * T_uv at each vertex.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


def _to_numpy(t) -> np.ndarray:
    """Convert tensor-like to numpy array."""
    return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)


# =========================================================================
# Config & result dataclasses
# =========================================================================


@dataclass
class EinsteinConfig:
    """Configuration for Einstein equation verification."""

    mc_time_index: int | None = None
    regularization: float = 1e-6
    stress_energy_mode: str = "full"
    bulk_fraction: float = 0.8
    fd_regularization: float = 1e-4
    scalar_density_mode: str = "volume"  # "volume" or "knn"
    knn_k: int = 10
    coarse_grain_bins: int = 0
    coarse_grain_min_points: int = 5
    coarse_grain_method: str = "radial"
    temporal_average_enabled: bool = False
    temporal_window_frames: int = 8
    temporal_stride: int = 1
    bootstrap_samples: int = 0
    bootstrap_confidence: float = 0.95
    bootstrap_seed: int = 12345
    bootstrap_frame_block_size: int = 1


@dataclass
class EinsteinTestResult:
    """Results of Einstein equation verification."""

    # Per-walker tensors
    positions: np.ndarray
    metrics: np.ndarray
    ricci_tensor: np.ndarray
    ricci_scalar: np.ndarray
    einstein_tensor: np.ndarray
    stress_energy_tensor: np.ndarray
    volumes: np.ndarray
    gradients: np.ndarray

    # Scalar test: R = slope * rho + intercept
    scalar_r2: float
    scalar_slope: float
    scalar_intercept: float
    g_newton_einstein: float
    lambda_measured: float
    scalar_density_mode: str
    knn_k: int | None
    scalar_r2_coarse: float | None
    scalar_slope_coarse: float | None
    scalar_intercept_coarse: float | None
    scalar_bin_count_coarse: int | None
    scalar_density: np.ndarray | None
    scalar_rho_coarse: np.ndarray | None
    scalar_R_coarse: np.ndarray | None
    scalar_counts_coarse: np.ndarray | None
    scalar_regression_density: np.ndarray | None
    scalar_regression_ricci: np.ndarray | None
    scalar_regression_valid_mask: np.ndarray | None
    temporal_average_enabled: bool
    temporal_frame_indices: np.ndarray | None
    temporal_frame_count: int
    scalar_r2_ci_bootstrap: tuple[float, float] | None
    scalar_slope_ci_bootstrap: tuple[float, float] | None
    scalar_intercept_ci_bootstrap: tuple[float, float] | None
    g_newton_ci_bootstrap: tuple[float, float] | None
    lambda_ci_bootstrap: tuple[float, float] | None
    scalar_r2_ci_jackknife: tuple[float, float] | None
    scalar_slope_ci_jackknife: tuple[float, float] | None
    scalar_intercept_ci_jackknife: tuple[float, float] | None
    g_newton_ci_jackknife: tuple[float, float] | None
    lambda_ci_jackknife: tuple[float, float] | None
    scalar_bootstrap_samples: int
    scalar_bootstrap_confidence: float | None

    # Tensor test: G_uv = slope * T_uv + intercept
    tensor_r2: float
    tensor_slope: float
    tensor_r2_per_component: np.ndarray
    component_labels: list[str]

    # Cross-check: full Ricci vs ricci_scalar_proxy
    ricci_proxy: np.ndarray | None
    proxy_vs_full_r2: float | None

    # Bulk vs boundary
    bulk_mask: np.ndarray
    bulk_scalar_r2: float
    boundary_scalar_r2: float

    # G_N reference
    g_newton_area_law: float
    g_newton_source: str
    g_newton_ratio: float

    # Metadata
    n_walkers: int
    spatial_dim: int
    mc_frame: int
    config: EinsteinConfig
    valid_mask: np.ndarray
    scalar_valid_mask: np.ndarray | None = None
    ricci_scalar_full: np.ndarray | None = None
    ricci_scalar_source: str = "full_derivative_pipeline"
    volumes_full: np.ndarray | None = None
    scalar_r2_full_volume: float | None = None
    scalar_slope_full_volume: float | None = None
    scalar_intercept_full_volume: float | None = None
    g_newton_einstein_full_volume: float | None = None
    lambda_measured_full_volume: float | None = None
    bulk_scalar_r2_full_volume: float | None = None
    boundary_scalar_r2_full_volume: float | None = None


# =========================================================================
# Pipeline functions
# =========================================================================


def _edges_to_neighbors(edge_index: np.ndarray, N: int) -> list[list[int]]:
    """Convert [E, 2] directed edge array to symmetric adjacency lists."""
    neighbors: list[set[int]] = [set() for _ in range(N)]
    for e in range(edge_index.shape[0]):
        s, d = int(edge_index[e, 0]), int(edge_index[e, 1])
        neighbors[s].add(d)
        neighbors[d].add(s)
    return [sorted(nb) for nb in neighbors]


def _compute_metric_derivatives(
    positions: np.ndarray,
    metrics: np.ndarray,
    edge_index: np.ndarray,
    reg: float,
) -> np.ndarray:
    """Compute metric derivatives dg_{bc}/dx^a via weighted LS per node.

    Adapts the `_fit_local_quadratic` pattern from scutoid/ricci.py.

    Args:
        positions: [N, d]
        metrics: [N, d, d]
        edge_index: [E, 2] directed edges (src, dst)
        reg: Tikhonov regularization

    Returns:
        dg: [N, d, d, d] where dg[i, a, b, c] = partial_a g_{bc} at node i
    """
    N, d = positions.shape
    dg = np.zeros((N, d, d, d), dtype=positions.dtype)

    if edge_index.shape[0] == 0:
        return dg

    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    delta = positions[dst] - positions[src]  # [E, d]

    # For each metric component (b, c), fit a linear model
    for b in range(d):
        for c in range(b, d):
            y = metrics[dst, b, c] - metrics[src, b, c]  # [E]

            # Accumulate X^T X and X^T y per source node
            xtx = np.zeros((N, d, d), dtype=positions.dtype)
            xty = np.zeros((N, d), dtype=positions.dtype)

            # Outer products: delta_e delta_e^T weighted, scattered to src
            outer = delta[:, :, None] * delta[:, None, :]  # [E, d, d]
            yw = y[:, None] * delta  # [E, d]

            for e_idx in range(len(src)):
                s = src[e_idx]
                xtx[s] += outer[e_idx]
                xty[s] += yw[e_idx]

            # Vectorized solve with regularization
            eye = reg * np.eye(d, dtype=positions.dtype)
            for i in range(N):
                A = xtx[i] + eye
                try:
                    grad = np.linalg.solve(A, xty[i])
                except np.linalg.LinAlgError:
                    grad = np.zeros(d, dtype=positions.dtype)
                dg[i, :, b, c] = grad
                if b != c:
                    dg[i, :, c, b] = grad

    return dg


def _compute_metric_derivatives_vectorized(
    positions: np.ndarray,
    metrics: np.ndarray,
    edge_index: np.ndarray,
    reg: float,
) -> np.ndarray:
    """Vectorized metric derivatives using scatter-add.

    Faster than per-node loops for large graphs.
    """
    N, d = positions.shape
    dg = np.zeros((N, d, d, d), dtype=positions.dtype)

    if edge_index.shape[0] == 0:
        return dg

    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    delta = positions[dst] - positions[src]  # [E, d]
    outer = delta[:, :, None] * delta[:, None, :]  # [E, d, d]

    # Accumulate X^T X per source node via np.add.at
    xtx = np.zeros((N, d, d), dtype=positions.dtype)
    np.add.at(xtx, src, outer)

    eye = reg * np.eye(d, dtype=positions.dtype)
    xtx_reg = xtx + eye[None, :, :]

    # Pre-compute inverse of (X^T X + reg*I) per node
    xtx_inv = np.linalg.inv(xtx_reg)  # [N, d, d]

    for b in range(d):
        for c in range(b, d):
            y = metrics[dst, b, c] - metrics[src, b, c]  # [E]
            yw = y[:, None] * delta  # [E, d]

            xty = np.zeros((N, d), dtype=positions.dtype)
            np.add.at(xty, src, yw)

            grad = np.einsum("nij,nj->ni", xtx_inv, xty)  # [N, d]
            dg[:, :, b, c] = grad
            if b != c:
                dg[:, :, c, b] = grad

    return dg


def _compute_christoffel(
    metrics: np.ndarray,
    dg: np.ndarray,
) -> np.ndarray:
    """Compute Christoffel symbols from metric and its derivatives.

    Gamma^a_{bc} = 0.5 * g^{ad} * (dg_{d,b,c} + dg_{d,c,b} - dg_{b,c,d})
    where dg_{a,b,c} = partial_a g_{bc}

    But dg[i, a, b, c] = partial_a g_{bc}, so:
    Gamma^a_{bc} = 0.5 * g^{ad} * (dg[b, d, c] + dg[c, d, b] - dg[d, b, c])

    Args:
        metrics: [N, d, d]
        dg: [N, d, d, d] with dg[i, a, b, c] = partial_a g_{bc}

    Returns:
        Gamma: [N, d, d, d] with Gamma[i, a, b, c] = Gamma^a_{bc}
    """
    g_inv = np.linalg.inv(metrics)  # [N, d, d]

    # Bracket: dg[b,d,c] + dg[c,d,b] - dg[d,b,c]
    # dg indices: [N, partial_idx, row, col]
    # dg[n, b, d, c] -> np.einsum index: dg_{nbdc}
    bracket = dg.transpose(0, 2, 3, 1) + dg.transpose(0, 3, 2, 1) - dg.transpose(0, 1, 2, 3)
    # bracket shape interpretation:
    # We need bracket[n, d, b, c] = dg[n,b,d,c] + dg[n,c,d,b] - dg[n,d,b,c]
    # dg[n,b,d,c]: take dg and swap axes 1<->2 partially
    # Let's do it explicitly with einsum-like indexing:
    # dg has shape [N, a, b, c] meaning partial_a g_{bc}
    # We need:
    #   term1[n, d, b, c] = dg[n, b, d, c]  (partial_b g_{dc})
    #   term2[n, d, b, c] = dg[n, c, d, b]  (partial_c g_{db})
    #   term3[n, d, b, c] = dg[n, d, b, c]  (partial_d g_{bc})
    term1 = np.swapaxes(dg, 1, 2)  # [N, b->d, a->b, c] hmm, need careful indexing

    N, d_dim = metrics.shape[:2]
    bracket = np.zeros_like(dg)  # [N, d, d, d] -> bracket[n, d_idx, b, c]
    for n in range(N):
        for d_idx in range(d_dim):
            for b in range(d_dim):
                for c in range(d_dim):
                    bracket[n, d_idx, b, c] = (
                        dg[n, b, d_idx, c]
                        + dg[n, c, d_idx, b]
                        - dg[n, d_idx, b, c]
                    )

    # Gamma^a_{bc} = 0.5 * g^{ad} * bracket[d, b, c]
    # gamma[n, a, b, c] = 0.5 * sum_d g_inv[n, a, d] * bracket[n, d, b, c]
    gamma = 0.5 * np.einsum("nad,ndbc->nabc", g_inv, bracket)
    return gamma


def _compute_christoffel_vectorized(
    metrics: np.ndarray,
    dg: np.ndarray,
) -> np.ndarray:
    """Vectorized Christoffel symbols without Python loops over indices.

    Args:
        metrics: [N, d, d]
        dg: [N, d, d, d] with dg[i, a, b, c] = partial_a g_{bc}

    Returns:
        Gamma: [N, d, d, d] with Gamma[i, a, b, c] = Gamma^a_{bc}
    """
    g_inv = np.linalg.inv(metrics)  # [N, d, d]

    # bracket[n, d_idx, b, c] = dg[n, b, d_idx, c] + dg[n, c, d_idx, b] - dg[n, d_idx, b, c]
    # dg shape: [N, partial_a, row_b, col_c]
    # dg[n, b, d_idx, c] -> transpose axes (0, 2, 1, 3)
    # dg[n, c, d_idx, b] -> transpose axes (0, 3, 1, 2)
    # dg[n, d_idx, b, c] -> already in place (axes 0, 1, 2, 3)
    bracket = (
        dg.transpose(0, 2, 1, 3)  # dg[n, b, d_idx, c]
        + dg.transpose(0, 3, 1, 2)  # dg[n, c, d_idx, b]
        - dg  # dg[n, d_idx, b, c]
    )

    gamma = 0.5 * np.einsum("nad,ndbc->nabc", g_inv, bracket)
    return gamma


def _compute_christoffel_derivatives(
    positions: np.ndarray,
    christoffels: np.ndarray,
    edge_index: np.ndarray,
    reg: float,
) -> np.ndarray:
    """Compute Christoffel derivatives via LS on neighbor stencil.

    Args:
        positions: [N, d]
        christoffels: [N, d, d, d] Gamma^a_{bc}
        edge_index: [E, 2]
        reg: regularization

    Returns:
        dGamma: [N, d, d, d, d] where dGamma[i, e, a, b, c] = partial_e Gamma^a_{bc}
    """
    N, d = positions.shape
    dGamma = np.zeros((N, d, d, d, d), dtype=positions.dtype)

    if edge_index.shape[0] == 0:
        return dGamma

    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    delta = positions[dst] - positions[src]

    # Precompute (X^T X + reg I)^{-1} per node
    outer = delta[:, :, None] * delta[:, None, :]
    xtx = np.zeros((N, d, d), dtype=positions.dtype)
    np.add.at(xtx, src, outer)
    eye = reg * np.eye(d, dtype=positions.dtype)
    xtx_inv = np.linalg.inv(xtx + eye[None, :, :])

    for a in range(d):
        for b in range(d):
            for c in range(b, d):
                y = christoffels[dst, a, b, c] - christoffels[src, a, b, c]
                yw = y[:, None] * delta

                xty = np.zeros((N, d), dtype=positions.dtype)
                np.add.at(xty, src, yw)

                grad = np.einsum("nij,nj->ni", xtx_inv, xty)
                dGamma[:, :, a, b, c] = grad
                if b != c:
                    dGamma[:, :, a, c, b] = grad

    return dGamma


def _compute_riemann(
    christoffels: np.ndarray,
    dGamma: np.ndarray,
) -> np.ndarray:
    """Compute Riemann tensor from Christoffel symbols and their derivatives.

    R^a_{bcd} = partial_c Gamma^a_{bd} - partial_d Gamma^a_{bc}
                + Gamma^a_{ce} Gamma^e_{bd} - Gamma^a_{de} Gamma^e_{bc}

    Args:
        christoffels: [N, d, d, d] Gamma^a_{bc}
        dGamma: [N, d, d, d, d] dGamma[e_idx, a, b, c] = partial_e Gamma^a_{bc}

    Returns:
        R: [N, d, d, d, d] with R[n, a, b, c, d_idx] = R^a_{bcd}
    """
    # dGamma[n, e, a, b, c] = partial_e Gamma^a_{bc}
    # term1: partial_c Gamma^a_{bd} = dGamma[n, c, a, b, d]
    # term2: partial_d Gamma^a_{bc} = dGamma[n, d, a, b, c]
    # We'll build R[n, a, b, c, d] = R^a_{bcd}

    # Derivative terms
    # R1[n, a, b, c, d] = dGamma[n, c, a, b, d]
    R1 = dGamma.transpose(0, 3, 2, 1, 4)  # hmm, let me be explicit

    N = christoffels.shape[0]
    d_dim = christoffels.shape[1]
    R = np.zeros((N, d_dim, d_dim, d_dim, d_dim), dtype=christoffels.dtype)

    # Derivative terms: dGamma[n, e, a, b, c]
    # partial_c Gamma^a_{bd} -> dGamma[n, c, a, b, d_idx]
    # partial_d Gamma^a_{bc} -> dGamma[n, d_idx, a, b, c]
    for n in range(N):
        for a in range(d_dim):
            for b in range(d_dim):
                for c in range(d_dim):
                    for d_idx in range(d_dim):
                        val = dGamma[n, c, a, b, d_idx] - dGamma[n, d_idx, a, b, c]
                        for e in range(d_dim):
                            val += (
                                christoffels[n, a, c, e] * christoffels[n, e, b, d_idx]
                                - christoffels[n, a, d_idx, e] * christoffels[n, e, b, c]
                            )
                        R[n, a, b, c, d_idx] = val
    return R


def _compute_riemann_vectorized(
    christoffels: np.ndarray,
    dGamma: np.ndarray,
) -> np.ndarray:
    """Vectorized Riemann tensor computation.

    R^a_{bcd} = partial_c Gamma^a_{bd} - partial_d Gamma^a_{bc}
                + Gamma^a_{ce} Gamma^e_{bd} - Gamma^a_{de} Gamma^e_{bc}

    Args:
        christoffels: [N, d, d, d] Gamma[n, a, b, c] = Gamma^a_{bc}
        dGamma: [N, d, d, d, d] dGamma[n, e, a, b, c] = partial_e Gamma^a_{bc}

    Returns:
        R: [N, d, d, d, d] with R[n, a, b, c, d_idx] = R^a_{bcd}
    """
    # Derivative terms
    # partial_c Gamma^a_{bd} = dGamma[n, c, a, b, d] -> need to rearrange
    # dGamma has indices [n, e, a, b, c], we want [n, a, b, c, d] = dGamma[n, c, a, b, d]

    # Build R^a_{bcd}:
    # term1[n, a, b, c, d] = dGamma[n, c, a, b, d]
    term1 = dGamma.transpose(0, 2, 3, 1, 4)

    # term2[n, a, b, c, d] = dGamma[n, d, a, b, c]
    term2 = dGamma.transpose(0, 2, 3, 4, 1)

    # Gamma*Gamma terms via einsum
    # term3[n,a,b,c,d] = sum_e Gamma^a_{ce} * Gamma^e_{bd}
    # Gamma[n,a,c,e] * Gamma[n,e,b,d] -> einsum 'nace,nebd->nabcd'
    term3 = np.einsum("nace,nebd->nabcd", christoffels, christoffels)

    # term4[n,a,b,c,d] = sum_e Gamma^a_{de} * Gamma^e_{bc}
    # Gamma[n,a,d,e] * Gamma[n,e,b,c] -> einsum 'nade,nebc->nabcd'
    # But we want the d index in position 4: [n,a,b,c,d]
    # term4[n,a,b,c,d] = Gamma[n,a,d,e] * Gamma[n,e,b,c]
    term4 = np.einsum("nade,nebc->nabdc", christoffels, christoffels)
    # nabdc -> we need nabcd, so swap last two:
    term4 = term4.transpose(0, 1, 2, 4, 3)

    R = term1 - term2 + term3 - term4
    return R


def _compute_ricci_and_einstein(
    riemann: np.ndarray,
    metrics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Ricci tensor, scalar, and Einstein tensor.

    R_{bd} = R^a_{bad} (contract first and third indices)
    R = g^{bd} R_{bd}
    G_{bd} = R_{bd} - 0.5 * R * g_{bd}

    Args:
        riemann: [N, d, d, d, d] R^a_{bcd}
        metrics: [N, d, d]

    Returns:
        (ricci, R_scalar, einstein) shapes [N,d,d], [N], [N,d,d]
    """
    # R_{bd} = R^a_{bad} = sum over a of R[n, a, b, a, d]
    ricci = np.einsum("nabad->nbd", riemann)

    g_inv = np.linalg.inv(metrics)
    R_scalar = np.einsum("nbd,nbd->n", g_inv, ricci)

    einstein = ricci - 0.5 * R_scalar[:, None, None] * metrics

    return ricci, R_scalar, einstein


def _compute_stress_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    fitness: np.ndarray,
    gradients: np.ndarray,
    metrics: np.ndarray,
    volumes: np.ndarray,
    neighbors: list[list[int]],
    mode: str = "full",
) -> np.ndarray:
    """Compute stress-energy tensor T_{uv} from simulation data.

    Three modes:
    - "fitness_only": T_{uv} = nabla_u phi * nabla_v phi - 0.5 g_{uv} (nabla phi)^2
    - "kinetic_pressure": adds kinetic term rho * u_u * u_v
    - "full": adds pressure term  P * g_{uv}

    Args:
        positions: [N, d]
        velocities: [N, d]
        fitness: [N]
        gradients: [N, d]
        metrics: [N, d, d]
        volumes: [N]
        neighbors: adjacency list
        mode: "fitness_only", "kinetic_pressure", or "full"

    Returns:
        T: [N, d, d]
    """
    N, d = positions.shape
    T = np.zeros((N, d, d), dtype=positions.dtype)
    g_inv = np.linalg.inv(metrics)

    # Scalar field (fitness) contribution: T^(phi)_{uv}
    # T^(phi)_{uv} = grad_u * grad_v - 0.5 * g_{uv} * g^{ab} * grad_a * grad_b
    grad_outer = gradients[:, :, None] * gradients[:, None, :]  # [N, d, d]
    grad_sq = np.einsum("nab,na,nb->n", g_inv, gradients, gradients)  # [N]
    T_phi = grad_outer - 0.5 * grad_sq[:, None, None] * metrics
    T += T_phi

    if mode in ("kinetic_pressure", "full"):
        # Kinetic contribution: rho * v_u * v_v
        # Use 1/volume as energy density proxy
        rho = np.where(volumes > 0, 1.0 / volumes, 0.0)
        v_outer = velocities[:, :, None] * velocities[:, None, :]
        T_kin = rho[:, None, None] * v_outer
        T += T_kin

    if mode == "full":
        # Pressure term: P * g_{uv}
        # P = (1/d) * rho * v^2
        rho = np.where(volumes > 0, 1.0 / volumes, 0.0)
        v_sq = np.einsum("nab,na,nb->n", g_inv, velocities, velocities)
        P = rho * v_sq / d
        T_pressure = P[:, None, None] * metrics
        T += T_pressure

    return T


def _extract_g_newton(
    regressions_df: Any | None,
    preference: str = "s_total_geom",
    manual_value: float = 1.0,
) -> tuple[float, str]:
    """Extract G_N = 1/(4*alpha) from fractal set regression DataFrame.

    Args:
        regressions_df: DataFrame with 'metric_key' and 'slope_alpha' columns
        preference: preferred metric key
        manual_value: fallback manual G_N

    Returns:
        (G_N, source_description)
    """
    if regressions_df is None or (hasattr(regressions_df, "empty") and regressions_df.empty):
        return (manual_value, "manual (no regressions)")

    import pandas as pd

    if not isinstance(regressions_df, pd.DataFrame):
        return (manual_value, "manual (invalid regressions)")

    # Preference order
    pref_order = ["s_total_geom", "s_total", "s_dist_geom", "s_dist"]
    if preference != "manual" and preference not in pref_order:
        pref_order.insert(0, preference)
    elif preference in pref_order:
        pref_order.remove(preference)
        pref_order.insert(0, preference)

    for key in pref_order:
        if "metric_key" in regressions_df.columns:
            row = regressions_df[regressions_df["metric_key"] == key]
        elif "metric" in regressions_df.columns:
            row = regressions_df[regressions_df["metric"] == key]
        else:
            continue

        if not row.empty and "slope_alpha" in row.columns:
            alpha = float(row.iloc[0]["slope_alpha"])
            if abs(alpha) > 1e-12:
                g_n = 1.0 / (4.0 * alpha)
                return (g_n, f"1/(4*alpha) from {key}")

    return (manual_value, "manual (fallback)")


def _compute_gradient_from_neighbors(
    positions: np.ndarray,
    fitness: np.ndarray,
    edge_index: np.ndarray,
    reg: float,
) -> np.ndarray:
    """Compute fitness gradients via LS when not available in history."""
    N, d = positions.shape
    grads = np.zeros((N, d), dtype=positions.dtype)

    if edge_index.shape[0] == 0:
        return grads

    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    delta = positions[dst] - positions[src]
    y = fitness[dst] - fitness[src]

    outer = delta[:, :, None] * delta[:, None, :]
    xtx = np.zeros((N, d, d), dtype=positions.dtype)
    np.add.at(xtx, src, outer)

    yw = y[:, None] * delta
    xty = np.zeros((N, d), dtype=positions.dtype)
    np.add.at(xty, src, yw)

    eye = reg * np.eye(d, dtype=positions.dtype)
    xtx_inv = np.linalg.inv(xtx + eye[None, :, :])
    grads = np.einsum("nij,nj->ni", xtx_inv, xty)
    return grads


def _run_scalar_test(
    ricci_scalar: np.ndarray,
    volumes: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, float]:
    """Run scalar test: linregress(1/volume, R).

    Returns: (r2, slope, intercept)
    """
    density = np.where(volumes > 0, 1.0 / volumes, np.nan)
    return _run_scalar_test_from_density(ricci_scalar, density, mask)


def _run_scalar_test_from_density(
    ricci_scalar: np.ndarray,
    density: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, float]:
    """Run scalar test: linregress(rho, R) with an explicit density field."""
    valid = mask & np.isfinite(density)
    if valid.sum() < 3:
        return (0.0, 0.0, 0.0)

    rho = density[valid]
    R = ricci_scalar[valid]

    finite = np.isfinite(rho) & np.isfinite(R)
    if finite.sum() < 3:
        return (0.0, 0.0, 0.0)

    result = stats.linregress(rho[finite], R[finite])
    return (result.rvalue**2, result.slope, result.intercept)


def _vectorized_weighted_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized weighted linear regression for batches of samples.

    Args:
        x: [B, K] independent variable
        y: [B, K] dependent variable
        w: [B, K] non-negative sample weights/masks

    Returns:
        (slope, intercept, r2, n_eff) each with shape [B]
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    w_arr = np.asarray(w, dtype=np.float64)

    if x_arr.ndim != 2 or y_arr.ndim != 2 or w_arr.ndim != 2:
        raise ValueError("x, y, and w must be rank-2 arrays [B, K].")
    if x_arr.shape != y_arr.shape or x_arr.shape != w_arr.shape:
        raise ValueError("x, y, and w must share identical shape.")

    w_sum = np.sum(w_arr, axis=1)
    safe = w_sum > 0.0

    x_mean = np.divide(
        np.sum(w_arr * x_arr, axis=1),
        w_sum,
        out=np.full_like(w_sum, np.nan),
        where=safe,
    )
    y_mean = np.divide(
        np.sum(w_arr * y_arr, axis=1),
        w_sum,
        out=np.full_like(w_sum, np.nan),
        where=safe,
    )

    exx = np.divide(
        np.sum(w_arr * x_arr * x_arr, axis=1),
        w_sum,
        out=np.full_like(w_sum, np.nan),
        where=safe,
    )
    eyy = np.divide(
        np.sum(w_arr * y_arr * y_arr, axis=1),
        w_sum,
        out=np.full_like(w_sum, np.nan),
        where=safe,
    )
    exy = np.divide(
        np.sum(w_arr * x_arr * y_arr, axis=1),
        w_sum,
        out=np.full_like(w_sum, np.nan),
        where=safe,
    )

    cov = exy - x_mean * y_mean
    var_x = np.maximum(exx - x_mean * x_mean, 0.0)
    var_y = np.maximum(eyy - y_mean * y_mean, 0.0)

    eps = 1e-30
    valid = safe & (w_sum >= 3.0) & (var_x > eps) & (var_y > eps)

    slope = np.full_like(w_sum, np.nan)
    intercept = np.full_like(w_sum, np.nan)
    r2 = np.full_like(w_sum, np.nan)

    slope[valid] = cov[valid] / var_x[valid]
    intercept[valid] = y_mean[valid] - slope[valid] * x_mean[valid]
    r2[valid] = np.clip((cov[valid] * cov[valid]) / (var_x[valid] * var_y[valid]), 0.0, 1.0)

    return slope, intercept, r2, w_sum


def _temporal_frame_indices(
    base_frame: int,
    n_recorded_fitness_frames: int,
    enabled: bool,
    window_frames: int,
    stride: int,
) -> np.ndarray:
    """Select frames used by temporal averaging."""
    if n_recorded_fitness_frames <= 0:
        return np.empty(0, dtype=np.int64)

    frame = int(np.clip(base_frame, 0, n_recorded_fitness_frames - 1))
    if not enabled:
        return np.array([frame], dtype=np.int64)

    step = int(max(1, stride))
    window = int(max(1, window_frames))
    start = max(0, frame - (window - 1) * step)
    frames = np.arange(start, frame + 1, step, dtype=np.int64)
    if frames.size == 0:
        frames = np.array([frame], dtype=np.int64)
    return frames


def _bootstrap_scalar_regression_cis(
    density_frames: np.ndarray,
    ricci_frames: np.ndarray,
    valid_frames: np.ndarray,
    n_samples: int,
    confidence: float,
    seed: int,
    frame_block_size: int,
) -> tuple[
    tuple[float, float] | None,
    tuple[float, float] | None,
    tuple[float, float] | None,
    tuple[float, float] | None,
    tuple[float, float] | None,
    int,
    float | None,
]:
    """Compute bootstrap CIs with frame-block and walker resampling."""
    b = int(max(0, n_samples))
    if b == 0:
        return (None, None, None, None, None, 0, None)

    conf = float(np.clip(confidence, 1e-6, 1.0 - 1e-6))
    x_f = np.asarray(density_frames, dtype=np.float64)
    y_f = np.asarray(ricci_frames, dtype=np.float64)
    m_f = np.asarray(valid_frames, dtype=bool)
    if x_f.ndim != 2 or y_f.ndim != 2 or m_f.ndim != 2:
        raise ValueError("density_frames, ricci_frames, and valid_frames must be [F, N].")
    if x_f.shape != y_f.shape or x_f.shape != m_f.shape:
        raise ValueError("Bootstrap frame arrays must share identical shape.")

    f_count, n_count = x_f.shape
    if f_count == 0 or n_count == 0:
        return (None, None, None, None, None, 0, None)

    rng = np.random.default_rng(int(seed))

    if f_count == 1:
        sampled_frames = np.zeros((b, 1), dtype=np.int64)
    else:
        block = int(max(1, frame_block_size))
        if block <= 1:
            sampled_frames = rng.integers(0, f_count, size=(b, f_count), dtype=np.int64)
        else:
            n_blocks = int(np.ceil(f_count / block))
            starts = rng.integers(0, f_count, size=(b, n_blocks), dtype=np.int64)
            offsets = np.arange(block, dtype=np.int64)[None, None, :]
            sampled_blocks = (starts[:, :, None] + offsets) % f_count
            sampled_frames = sampled_blocks.reshape(b, -1)[:, :f_count]

    x_sel = x_f[sampled_frames]  # [B, F', N]
    y_sel = y_f[sampled_frames]
    m_sel = m_f[sampled_frames]

    walker_idx = rng.integers(0, n_count, size=(b, sampled_frames.shape[1], n_count), dtype=np.int64)
    x_boot = np.take_along_axis(x_sel, walker_idx, axis=2)
    y_boot = np.take_along_axis(y_sel, walker_idx, axis=2)
    w_boot = np.take_along_axis(m_sel, walker_idx, axis=2).astype(np.float64)

    slope, intercept, r2, n_eff = _vectorized_weighted_linear_regression(
        x_boot.reshape(b, -1),
        y_boot.reshape(b, -1),
        w_boot.reshape(b, -1),
    )

    valid = np.isfinite(slope) & np.isfinite(intercept) & np.isfinite(r2) & (n_eff >= 3.0)
    n_valid = int(np.sum(valid))
    if n_valid < 3:
        return (None, None, None, None, None, n_valid, conf)

    q_low = (1.0 - conf) * 0.5
    q_hi = 1.0 - q_low

    def _ci(values: np.ndarray) -> tuple[float, float]:
        vals = values[valid]
        return (float(np.quantile(vals, q_low)), float(np.quantile(vals, q_hi)))

    r2_ci = _ci(r2)
    slope_ci = _ci(slope)
    intercept_ci = _ci(intercept)
    g_ci = (slope_ci[0] / (16.0 * np.pi), slope_ci[1] / (16.0 * np.pi))
    lambda_ci = (intercept_ci[0] / 6.0, intercept_ci[1] / 6.0)

    return (r2_ci, slope_ci, intercept_ci, g_ci, lambda_ci, n_valid, conf)


def _jackknife_scalar_regression_cis(
    density_frames: np.ndarray,
    ricci_frames: np.ndarray,
    valid_frames: np.ndarray,
    confidence: float,
) -> tuple[
    tuple[float, float] | None,
    tuple[float, float] | None,
    tuple[float, float] | None,
    tuple[float, float] | None,
    tuple[float, float] | None,
]:
    """Compute leave-one-frame-out jackknife CIs for temporal averages."""
    x_f = np.asarray(density_frames, dtype=np.float64)
    y_f = np.asarray(ricci_frames, dtype=np.float64)
    w_f = np.asarray(valid_frames, dtype=np.float64)
    if x_f.ndim != 2 or y_f.ndim != 2 or w_f.ndim != 2:
        raise ValueError("Jackknife frame arrays must be rank-2 [F, N].")
    if x_f.shape != y_f.shape or x_f.shape != w_f.shape:
        raise ValueError("Jackknife frame arrays must share identical shape.")

    f_count = x_f.shape[0]
    if f_count < 2:
        return (None, None, None, None, None)

    sw_f = np.sum(w_f, axis=1)
    sx_f = np.sum(w_f * x_f, axis=1)
    sy_f = np.sum(w_f * y_f, axis=1)
    sxx_f = np.sum(w_f * x_f * x_f, axis=1)
    syy_f = np.sum(w_f * y_f * y_f, axis=1)
    sxy_f = np.sum(w_f * x_f * y_f, axis=1)

    sw_all = float(np.sum(sw_f))
    sx_all = float(np.sum(sx_f))
    sy_all = float(np.sum(sy_f))
    sxx_all = float(np.sum(sxx_f))
    syy_all = float(np.sum(syy_f))
    sxy_all = float(np.sum(sxy_f))

    sw = sw_all - sw_f
    sx = sx_all - sx_f
    sy = sy_all - sy_f
    sxx = sxx_all - sxx_f
    syy = syy_all - syy_f
    sxy = sxy_all - sxy_f

    safe = sw > 0.0
    x_mean = np.divide(sx, sw, out=np.full_like(sw, np.nan), where=safe)
    y_mean = np.divide(sy, sw, out=np.full_like(sw, np.nan), where=safe)
    cov = np.divide(sxy, sw, out=np.full_like(sw, np.nan), where=safe) - x_mean * y_mean
    var_x = np.divide(sxx, sw, out=np.full_like(sw, np.nan), where=safe) - x_mean * x_mean
    var_y = np.divide(syy, sw, out=np.full_like(sw, np.nan), where=safe) - y_mean * y_mean

    eps = 1e-30
    valid = safe & (sw >= 3.0) & (var_x > eps) & (var_y > eps)
    if int(np.sum(valid)) < 2:
        return (None, None, None, None, None)

    slope = np.full_like(sw, np.nan)
    intercept = np.full_like(sw, np.nan)
    r2 = np.full_like(sw, np.nan)
    slope[valid] = cov[valid] / var_x[valid]
    intercept[valid] = y_mean[valid] - slope[valid] * x_mean[valid]
    r2[valid] = np.clip((cov[valid] * cov[valid]) / (var_x[valid] * var_y[valid]), 0.0, 1.0)

    conf = float(np.clip(confidence, 1e-6, 1.0 - 1e-6))
    z = float(stats.norm.ppf(0.5 + 0.5 * conf))

    def _jk_ci(values: np.ndarray) -> tuple[float, float] | None:
        vals = values[np.isfinite(values)]
        m = vals.size
        if m < 2:
            return None
        mean = float(np.mean(vals))
        se = float(np.sqrt((m - 1.0) / m * np.sum((vals - mean) ** 2)))
        return (mean - z * se, mean + z * se)

    r2_ci = _jk_ci(r2[valid])
    slope_ci = _jk_ci(slope[valid])
    intercept_ci = _jk_ci(intercept[valid])
    g_ci = None if slope_ci is None else (slope_ci[0] / (16.0 * np.pi), slope_ci[1] / (16.0 * np.pi))
    lambda_ci = None if intercept_ci is None else (intercept_ci[0] / 6.0, intercept_ci[1] / 6.0)

    return (r2_ci, slope_ci, intercept_ci, g_ci, lambda_ci)


def _compute_knn_density(
    positions: np.ndarray,
    k: int,
) -> np.ndarray:
    """Compute k-nearest-neighbor density estimate for each point."""
    from scipy.spatial import cKDTree

    n, d = positions.shape
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if d <= 0:
        return np.full(n, np.nan, dtype=np.float64)

    k_eff = int(max(1, min(int(k), max(1, n - 1))))
    tree = cKDTree(positions)
    dists, _ = tree.query(positions, k=k_eff + 1)
    r_k = np.asarray(dists[:, -1], dtype=np.float64)
    r_k = np.clip(r_k, 1e-12, None)

    if d == 1:
        unit_ball_volume = 2.0
    elif d == 2:
        unit_ball_volume = np.pi
    elif d == 3:
        unit_ball_volume = 4.0 * np.pi / 3.0
    else:
        from scipy.special import gamma

        unit_ball_volume = (np.pi ** (0.5 * d)) / gamma(0.5 * d + 1.0)

    density = float(k_eff) / (unit_ball_volume * (r_k**d))
    return np.where(np.isfinite(density), density, np.nan)


def _run_scalar_test_coarse_grained(
    positions: np.ndarray,
    ricci_scalar: np.ndarray,
    density: np.ndarray,
    mask: np.ndarray,
    n_bins: int,
    min_points_per_bin: int,
    method: str = "radial",
    volumes: np.ndarray | None = None,
    density_mode: str = "volume",
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray] | None:
    """Run coarse-grained scalar Einstein test on spatial bins."""
    if int(n_bins) <= 0:
        return None
    if method != "radial":
        raise ValueError(f"Unsupported coarse_grain_method: {method!r}.")

    valid = (
        mask
        & np.isfinite(ricci_scalar)
        & np.isfinite(density)
    )
    if valid.sum() < 3:
        return None

    pos_v = positions[valid]
    r_v = ricci_scalar[valid]
    rho_v = density[valid]
    vol_v = None
    if volumes is not None and volumes.shape == density.shape:
        vol_v = volumes[valid]

    center = pos_v.mean(axis=0)
    radii = np.linalg.norm(pos_v - center, axis=1)
    if not np.all(np.isfinite(radii)):
        return None

    quantiles = np.linspace(0.0, 100.0, int(n_bins) + 1)
    edges = np.percentile(radii, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        return None

    bin_ids = np.digitize(radii, edges[1:-1], right=False)
    min_count = int(max(1, min_points_per_bin))

    r_bins: list[float] = []
    rho_bins: list[float] = []
    counts: list[int] = []
    for bin_idx in range(int(edges.size - 1)):
        in_bin = bin_ids == bin_idx
        count = int(np.sum(in_bin))
        if count < min_count:
            continue

        if (
            density_mode == "volume"
            and vol_v is not None
            and np.all(np.isfinite(vol_v[in_bin]))
            and np.all(vol_v[in_bin] > 0)
        ):
            weights = vol_v[in_bin]
            r_avg = float(np.average(r_v[in_bin], weights=weights))
            rho_avg = float(count / np.sum(weights))
        else:
            r_avg = float(np.mean(r_v[in_bin]))
            rho_avg = float(np.mean(rho_v[in_bin]))

        if not (np.isfinite(r_avg) and np.isfinite(rho_avg)):
            continue
        r_bins.append(r_avg)
        rho_bins.append(rho_avg)
        counts.append(count)

    if len(r_bins) < 3:
        return None

    rho_arr = np.asarray(rho_bins, dtype=np.float64)
    r_arr = np.asarray(r_bins, dtype=np.float64)
    cnt_arr = np.asarray(counts, dtype=np.int64)
    finite = np.isfinite(rho_arr) & np.isfinite(r_arr)
    if finite.sum() < 3:
        return None

    reg = stats.linregress(rho_arr[finite], r_arr[finite])
    return (
        reg.rvalue**2,
        float(reg.slope),
        float(reg.intercept),
        rho_arr[finite],
        r_arr[finite],
        cnt_arr[finite],
    )


def _run_tensor_test(
    einstein: np.ndarray,
    stress_energy: np.ndarray,
    mask: np.ndarray,
    d: int,
) -> tuple[float, float, np.ndarray, list[str]]:
    """Run tensor test: linregress(T_uv_flat, G_uv_flat) per component.

    Returns: (overall_r2, slope, per_component_r2, labels)
    """
    labels = []
    r2_components = []

    # Only use upper-triangle components (symmetric tensors)
    G_all = []
    T_all = []

    for b in range(d):
        for c in range(b, d):
            label = f"({b},{c})"
            labels.append(label)

            G_comp = einstein[mask, b, c]
            T_comp = stress_energy[mask, b, c]

            finite = np.isfinite(G_comp) & np.isfinite(T_comp)
            if finite.sum() < 3:
                r2_components.append(0.0)
                continue

            res = stats.linregress(T_comp[finite], G_comp[finite])
            r2_components.append(res.rvalue ** 2)

            G_all.append(G_comp[finite])
            T_all.append(T_comp[finite])

    # Overall regression on all flattened components
    if G_all:
        G_flat = np.concatenate(G_all)
        T_flat = np.concatenate(T_all)
        finite = np.isfinite(G_flat) & np.isfinite(T_flat)
        if finite.sum() >= 3:
            res = stats.linregress(T_flat[finite], G_flat[finite])
            overall_r2 = res.rvalue ** 2
            slope = res.slope
        else:
            overall_r2, slope = 0.0, 0.0
    else:
        overall_r2, slope = 0.0, 0.0

    return (overall_r2, slope, np.array(r2_components), labels)


def _compute_bulk_mask(
    positions: np.ndarray,
    bulk_fraction: float,
) -> np.ndarray:
    """Identify bulk walkers (those closer to center of mass).

    Returns: boolean mask [N]
    """
    center = positions.mean(axis=0)
    dists = np.linalg.norm(positions - center, axis=1)
    threshold = np.quantile(dists, bulk_fraction)
    return dists <= threshold


def _compute_full_volume_elements(
    history: Any,
    frame: int,
    metrics: np.ndarray,
) -> np.ndarray | None:
    """Compute V_full = V_euclidean * sqrt(det g) from recorded Voronoi regions."""
    if getattr(history, "voronoi_regions", None) is None:
        return None

    try:
        from fragile.fractalai.scutoid.voronoi_observables import compute_dual_volumes_from_history

        dual_volumes = compute_dual_volumes_from_history(history, record_index=frame + 1)
    except Exception:
        return None

    dual_np = _to_numpy(dual_volumes).reshape(-1)
    N = metrics.shape[0]
    if dual_np.shape[0] != N:
        return None

    sign, logdet = np.linalg.slogdet(metrics)
    sqrt_det = np.exp(0.5 * logdet)
    sqrt_det = np.where(sign > 0, sqrt_det, np.nan)

    volumes_full = dual_np * sqrt_det
    return np.where(np.isfinite(volumes_full) & (volumes_full > 0), volumes_full, np.nan)


def _compute_einstein_frame_data(
    history: Any,
    frame: int,
    config: EinsteinConfig,
) -> dict[str, Any]:
    """Compute Einstein pipeline tensors and scalar regression ingredients for one frame."""
    # x_final has shape [n_recorded, N, d], fitness has [n_recorded-1, N]
    positions = _to_numpy(history.x_final[frame + 1])  # [N, d]
    velocities = _to_numpy(history.v_final[frame + 1])  # [N, d]
    fitness_vals = _to_numpy(history.fitness[frame])  # [N]
    n_walkers, d = positions.shape

    metrics = None
    if history.diffusion_tensors_full is not None:
        metrics = _to_numpy(history.diffusion_tensors_full[frame])  # [N, d, d]
    elif history.fitness_hessians_full is not None:
        metrics = _to_numpy(history.fitness_hessians_full[frame])  # [N, d, d]

    if metrics is None:
        metrics = np.tile(np.eye(d, dtype=np.float64), (n_walkers, 1, 1))

    # Ensure positive-definite local metrics.
    for i in range(n_walkers):
        eigvals, eigvecs = np.linalg.eigh(metrics[i])
        eigvals = np.maximum(eigvals, config.regularization)
        metrics[i] = eigvecs @ np.diag(eigvals) @ eigvecs.T

    volumes = np.ones(n_walkers, dtype=np.float64)
    if history.riemannian_volume_weights is not None:
        volumes = _to_numpy(history.riemannian_volume_weights[frame])

    if history.neighbor_edges is not None and len(history.neighbor_edges) > frame:
        edges_tensor = history.neighbor_edges[frame]
        edge_index = _to_numpy(edges_tensor)  # [E, 2]
        if edge_index.ndim == 1:
            edge_index = edge_index.reshape(-1, 2)
    else:
        msg = "neighbor_edges not available in history"
        raise ValueError(msg)

    edge_index_rev = edge_index[:, ::-1]
    edge_index_full = np.concatenate([edge_index, edge_index_rev], axis=0)
    edge_set = set()
    unique_edges = []
    for e in range(edge_index_full.shape[0]):
        key = (int(edge_index_full[e, 0]), int(edge_index_full[e, 1]))
        if key not in edge_set:
            edge_set.add(key)
            unique_edges.append(key)
    edge_index_full = np.array(unique_edges, dtype=np.int64)
    neighbors = _edges_to_neighbors(edge_index_full, n_walkers)

    if history.fitness_gradients is not None:
        gradients = _to_numpy(history.fitness_gradients[frame])  # [N, d]
    else:
        gradients = _compute_gradient_from_neighbors(
            positions, fitness_vals, edge_index_full, config.fd_regularization,
        )

    ricci_proxy = None
    if history.ricci_scalar_proxy is not None:
        ricci_proxy = _to_numpy(history.ricci_scalar_proxy[frame])

    dg = _compute_metric_derivatives_vectorized(
        positions, metrics, edge_index_full, config.fd_regularization,
    )
    christoffels = _compute_christoffel_vectorized(metrics, dg)
    d_gamma = _compute_christoffel_derivatives(
        positions, christoffels, edge_index_full, config.fd_regularization,
    )
    riemann = _compute_riemann_vectorized(christoffels, d_gamma)
    ricci_tensor, ricci_scalar_full, einstein = _compute_ricci_and_einstein(riemann, metrics)

    if ricci_proxy is not None:
        ricci_scalar = ricci_proxy
        ricci_scalar_source = "riemannian_mix_proxy"
    else:
        ricci_scalar = ricci_scalar_full
        ricci_scalar_source = "full_derivative_pipeline"

    stress_energy = _compute_stress_energy(
        positions, velocities, fitness_vals, gradients,
        metrics, volumes, neighbors, config.stress_energy_mode,
    )

    valid_base = np.isfinite(ricci_scalar_full) & np.all(np.isfinite(einstein.reshape(n_walkers, -1)), axis=1)
    valid = valid_base & (volumes > 0)

    density_mode = str(config.scalar_density_mode).strip().lower()
    if density_mode == "volume":
        scalar_density = np.where(volumes > 0, 1.0 / volumes, np.nan)
        knn_k_used: int | None = None
        valid_scalar = valid
    elif density_mode == "knn":
        scalar_density = _compute_knn_density(positions, k=int(config.knn_k))
        knn_k_used = int(max(1, min(int(config.knn_k), max(1, n_walkers - 1))))
        valid_scalar = valid_base & np.isfinite(scalar_density)
    else:
        raise ValueError(
            "scalar_density_mode must be one of {'volume', 'knn'}, "
            f"got {config.scalar_density_mode!r}."
        )

    return {
        "frame": int(frame),
        "positions": positions,
        "metrics": metrics,
        "volumes": volumes,
        "gradients": gradients,
        "ricci_tensor": ricci_tensor,
        "ricci_scalar": ricci_scalar,
        "ricci_scalar_full": ricci_scalar_full,
        "ricci_scalar_source": ricci_scalar_source,
        "einstein_tensor": einstein,
        "stress_energy_tensor": stress_energy,
        "ricci_proxy": ricci_proxy,
        "valid_base": valid_base,
        "valid_mask": valid,
        "scalar_density": scalar_density,
        "scalar_valid_mask": valid_scalar,
        "knn_k_used": knn_k_used,
    }


# =========================================================================
# Orchestrator
# =========================================================================


def compute_einstein_test(
    history: Any,
    config: EinsteinConfig,
    fractal_set_regressions: Any | None = None,
    g_newton_metric: str = "s_total_geom",
    g_newton_manual: float = 1.0,
) -> EinsteinTestResult:
    """Main orchestrator: extract data, run pipeline, package results.

    Args:
        history: RunHistory object
        config: EinsteinConfig
        fractal_set_regressions: DataFrame from fractal set tab
        g_newton_metric: which area-law metric to use for G_N
        g_newton_manual: manual G_N value

    Returns:
        EinsteinTestResult
    """
    # 1. Select MC frame
    n_rec_minus1 = history.n_recorded - 1
    if n_rec_minus1 <= 0:
        raise ValueError("history has no recorded fitness frames.")

    if config.mc_time_index is not None:
        frame = int(min(config.mc_time_index, n_rec_minus1 - 1))
    else:
        frame = int(n_rec_minus1 - 1)

    frame_indices = _temporal_frame_indices(
        base_frame=frame,
        n_recorded_fitness_frames=n_rec_minus1,
        enabled=bool(config.temporal_average_enabled),
        window_frames=int(config.temporal_window_frames),
        stride=int(config.temporal_stride),
    )

    frame_payloads = [
        _compute_einstein_frame_data(history=history, frame=int(idx), config=config)
        for idx in frame_indices
    ]
    base_payload = frame_payloads[-1]

    positions = np.asarray(base_payload["positions"])
    metrics = np.asarray(base_payload["metrics"])
    ricci = np.asarray(base_payload["ricci_tensor"])
    einstein = np.asarray(base_payload["einstein_tensor"])
    T = np.asarray(base_payload["stress_energy_tensor"])
    volumes = np.asarray(base_payload["volumes"])
    gradients = np.asarray(base_payload["gradients"])
    ricci_proxy = base_payload["ricci_proxy"]
    valid_base = np.asarray(base_payload["valid_base"], dtype=bool)
    valid = np.asarray(base_payload["valid_mask"], dtype=bool)
    ricci_scalar_source = str(base_payload["ricci_scalar_source"])
    R_scalar_full = np.asarray(base_payload["ricci_scalar_full"])
    N, d = positions.shape

    scalar_density_frames = np.stack(
        [np.asarray(payload["scalar_density"], dtype=np.float64) for payload in frame_payloads],
        axis=0,
    )
    scalar_ricci_frames = np.stack(
        [np.asarray(payload["ricci_scalar"], dtype=np.float64) for payload in frame_payloads],
        axis=0,
    )
    scalar_valid_frames = np.stack(
        [np.asarray(payload["scalar_valid_mask"], dtype=bool) for payload in frame_payloads],
        axis=0,
    )
    density_mode = str(config.scalar_density_mode).strip().lower()
    knn_k_used = base_payload["knn_k_used"]

    temporal_enabled = bool(config.temporal_average_enabled) and frame_indices.size > 1
    if temporal_enabled:
        weights = scalar_valid_frames.astype(np.float64)
        weights_sum = np.sum(weights, axis=0)
        scalar_density = np.divide(
            np.sum(weights * scalar_density_frames, axis=0),
            weights_sum,
            out=np.full(N, np.nan, dtype=np.float64),
            where=weights_sum > 0.0,
        )
        R_scalar_for_scalar_test = np.divide(
            np.sum(weights * scalar_ricci_frames, axis=0),
            weights_sum,
            out=np.full(N, np.nan, dtype=np.float64),
            where=weights_sum > 0.0,
        )
        valid_scalar = weights_sum > 0.0
    else:
        scalar_density = scalar_density_frames[-1]
        R_scalar_for_scalar_test = scalar_ricci_frames[-1]
        valid_scalar = scalar_valid_frames[-1]

    # 2. Extract G_N from fractal set regressions
    if g_newton_metric == "manual":
        g_n, g_n_source = g_newton_manual, "manual"
    else:
        g_n, g_n_source = _extract_g_newton(
            fractal_set_regressions, g_newton_metric, g_newton_manual,
        )

    # 3. Scalar test
    scalar_r2, scalar_slope, scalar_intercept = _run_scalar_test_from_density(
        R_scalar_for_scalar_test, scalar_density, valid_scalar
    )
    g_n_einstein = scalar_slope / (16.0 * np.pi) if abs(scalar_slope) > 1e-30 else 0.0
    lambda_measured = scalar_intercept / 6.0

    # 4. Scalar test uncertainty (vectorized bootstrap + temporal jackknife)
    (
        scalar_r2_ci_bootstrap,
        scalar_slope_ci_bootstrap,
        scalar_intercept_ci_bootstrap,
        g_newton_ci_bootstrap,
        lambda_ci_bootstrap,
        scalar_bootstrap_samples,
        scalar_bootstrap_confidence,
    ) = _bootstrap_scalar_regression_cis(
        density_frames=scalar_density_frames,
        ricci_frames=scalar_ricci_frames,
        valid_frames=scalar_valid_frames,
        n_samples=int(max(0, config.bootstrap_samples)),
        confidence=float(config.bootstrap_confidence),
        seed=int(config.bootstrap_seed),
        frame_block_size=int(max(1, config.bootstrap_frame_block_size)),
    )

    (
        scalar_r2_ci_jackknife,
        scalar_slope_ci_jackknife,
        scalar_intercept_ci_jackknife,
        g_newton_ci_jackknife,
        lambda_ci_jackknife,
    ) = _jackknife_scalar_regression_cis(
        density_frames=scalar_density_frames,
        ricci_frames=scalar_ricci_frames,
        valid_frames=scalar_valid_frames,
        confidence=float(config.bootstrap_confidence),
    ) if temporal_enabled else (None, None, None, None, None)

    # 5. Coarse-grained scalar test
    coarse_r2 = None
    coarse_slope = None
    coarse_intercept = None
    coarse_rho = None
    coarse_r = None
    coarse_counts = None
    coarse_bin_count = None
    if int(config.coarse_grain_bins) > 0:
        coarse_result = _run_scalar_test_coarse_grained(
            positions=positions,
            ricci_scalar=R_scalar_for_scalar_test,
            density=scalar_density,
            mask=valid_scalar,
            n_bins=int(config.coarse_grain_bins),
            min_points_per_bin=int(config.coarse_grain_min_points),
            method=str(config.coarse_grain_method),
            volumes=volumes,
            density_mode=density_mode,
        )
        if coarse_result is not None:
            (
                coarse_r2,
                coarse_slope,
                coarse_intercept,
                coarse_rho,
                coarse_r,
                coarse_counts,
            ) = coarse_result
            coarse_bin_count = int(coarse_rho.shape[0])

    # 6. Theory-aligned scalar test with full volume element V_eucl * sqrt(det g)
    volumes_full = _compute_full_volume_elements(history, frame, metrics)
    scalar_r2_full = None
    scalar_slope_full = None
    scalar_intercept_full = None
    g_n_einstein_full = None
    lambda_measured_full = None
    bulk_r2_full = None
    boundary_r2_full = None
    if volumes_full is not None:
        scalar_r2_full, scalar_slope_full, scalar_intercept_full = _run_scalar_test(
            R_scalar_for_scalar_test, volumes_full, valid_base,
        )
        g_n_einstein_full = (
            scalar_slope_full / (16.0 * np.pi) if abs(scalar_slope_full) > 1e-30 else 0.0
        )
        lambda_measured_full = scalar_intercept_full / 6.0

    # 7. Tensor test
    tensor_r2, tensor_slope, tensor_r2_per, comp_labels = _run_tensor_test(
        einstein, T, valid, d,
    )

    # 8. Cross-check
    proxy_r2 = None
    if ricci_proxy is not None:
        finite_both = valid & np.isfinite(ricci_proxy) & np.isfinite(R_scalar_full)
        if finite_both.sum() >= 3:
            res = stats.linregress(ricci_proxy[finite_both], R_scalar_full[finite_both])
            proxy_r2 = res.rvalue ** 2

    # 9. Bulk vs boundary
    bulk_mask = _compute_bulk_mask(positions, config.bulk_fraction)
    bulk_r2, _, _ = _run_scalar_test_from_density(
        R_scalar_for_scalar_test, scalar_density, valid_scalar & bulk_mask
    )
    boundary_r2, _, _ = _run_scalar_test_from_density(
        R_scalar_for_scalar_test, scalar_density, valid_scalar & ~bulk_mask
    )
    if volumes_full is not None:
        bulk_r2_full, _, _ = _run_scalar_test(
            R_scalar_for_scalar_test, volumes_full, valid_base & bulk_mask,
        )
        boundary_r2_full, _, _ = _run_scalar_test(
            R_scalar_for_scalar_test, volumes_full, valid_base & ~bulk_mask,
        )

    # 10. G_N ratio
    g_n_ratio = g_n_einstein / g_n if abs(g_n) > 1e-30 else 0.0

    return EinsteinTestResult(
        positions=positions,
        metrics=metrics,
        ricci_tensor=ricci,
        ricci_scalar=R_scalar_for_scalar_test,
        einstein_tensor=einstein,
        stress_energy_tensor=T,
        volumes=volumes,
        gradients=gradients,
        scalar_r2=scalar_r2,
        scalar_slope=scalar_slope,
        scalar_intercept=scalar_intercept,
        g_newton_einstein=g_n_einstein,
        lambda_measured=lambda_measured,
        scalar_density_mode=density_mode,
        knn_k=knn_k_used,
        scalar_r2_coarse=coarse_r2,
        scalar_slope_coarse=coarse_slope,
        scalar_intercept_coarse=coarse_intercept,
        scalar_bin_count_coarse=coarse_bin_count,
        scalar_density=scalar_density,
        scalar_rho_coarse=coarse_rho,
        scalar_R_coarse=coarse_r,
        scalar_counts_coarse=coarse_counts,
        scalar_regression_density=scalar_density,
        scalar_regression_ricci=R_scalar_for_scalar_test,
        scalar_regression_valid_mask=valid_scalar,
        temporal_average_enabled=bool(config.temporal_average_enabled),
        temporal_frame_indices=np.asarray(frame_indices, dtype=np.int64),
        temporal_frame_count=int(frame_indices.size),
        scalar_r2_ci_bootstrap=scalar_r2_ci_bootstrap,
        scalar_slope_ci_bootstrap=scalar_slope_ci_bootstrap,
        scalar_intercept_ci_bootstrap=scalar_intercept_ci_bootstrap,
        g_newton_ci_bootstrap=g_newton_ci_bootstrap,
        lambda_ci_bootstrap=lambda_ci_bootstrap,
        scalar_r2_ci_jackknife=scalar_r2_ci_jackknife,
        scalar_slope_ci_jackknife=scalar_slope_ci_jackknife,
        scalar_intercept_ci_jackknife=scalar_intercept_ci_jackknife,
        g_newton_ci_jackknife=g_newton_ci_jackknife,
        lambda_ci_jackknife=lambda_ci_jackknife,
        scalar_bootstrap_samples=int(scalar_bootstrap_samples),
        scalar_bootstrap_confidence=scalar_bootstrap_confidence,
        tensor_r2=tensor_r2,
        tensor_slope=tensor_slope,
        tensor_r2_per_component=tensor_r2_per,
        component_labels=comp_labels,
        ricci_proxy=ricci_proxy,
        proxy_vs_full_r2=proxy_r2,
        bulk_mask=bulk_mask,
        bulk_scalar_r2=bulk_r2,
        boundary_scalar_r2=boundary_r2,
        g_newton_area_law=g_n,
        g_newton_source=g_n_source,
        g_newton_ratio=g_n_ratio,
        n_walkers=N,
        spatial_dim=d,
        mc_frame=frame,
        config=config,
        valid_mask=valid,
        scalar_valid_mask=valid_scalar,
        ricci_scalar_full=R_scalar_full,
        ricci_scalar_source=ricci_scalar_source,
        volumes_full=volumes_full,
        scalar_r2_full_volume=scalar_r2_full,
        scalar_slope_full_volume=scalar_slope_full,
        scalar_intercept_full_volume=scalar_intercept_full,
        g_newton_einstein_full_volume=g_n_einstein_full,
        lambda_measured_full_volume=lambda_measured_full,
        bulk_scalar_r2_full_volume=bulk_r2_full,
        boundary_scalar_r2_full_volume=boundary_r2_full,
    )
