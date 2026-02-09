"""Einstein equation verification on the fractal gas lattice.

Computes the full Riemann -> Ricci -> Einstein tensor pipeline from
pre-computed metric tensors and neighbor graphs stored in RunHistory,
then tests G_uv + Lambda*g_uv = 8*pi*G_N * T_uv at each vertex.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    valid = mask & (volumes > 0)
    if valid.sum() < 3:
        return (0.0, 0.0, 0.0)

    rho = 1.0 / volumes[valid]
    R = ricci_scalar[valid]

    # Filter out non-finite values
    finite = np.isfinite(rho) & np.isfinite(R)
    if finite.sum() < 3:
        return (0.0, 0.0, 0.0)

    result = stats.linregress(rho[finite], R[finite])
    return (result.rvalue ** 2, result.slope, result.intercept)


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
    if config.mc_time_index is not None:
        frame = min(config.mc_time_index, n_rec_minus1 - 1)
    else:
        frame = n_rec_minus1 - 1

    # 2. Extract positions, velocities, fitness
    # x_final has shape [n_recorded, N, d], fitness has [n_recorded-1, N]
    positions = _to_numpy(history.x_final[frame + 1])  # [N, d] (frame+1 because x_final is full)
    velocities = _to_numpy(history.v_final[frame + 1])  # [N, d]
    fitness_vals = _to_numpy(history.fitness[frame])  # [N]
    N, d = positions.shape

    # 3. Get metrics from history
    metrics = None
    if history.diffusion_tensors_full is not None:
        metrics = _to_numpy(history.diffusion_tensors_full[frame])  # [N, d, d]
    elif history.fitness_hessians_full is not None:
        metrics = _to_numpy(history.fitness_hessians_full[frame])  # [N, d, d]

    if metrics is None:
        # Fallback: identity metric
        metrics = np.tile(np.eye(d, dtype=np.float64), (N, 1, 1))

    # Regularize metric: ensure positive definite
    for i in range(N):
        eigvals, eigvecs = np.linalg.eigh(metrics[i])
        eigvals = np.maximum(eigvals, config.regularization)
        metrics[i] = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # 4. Get volumes
    volumes = np.ones(N, dtype=np.float64)
    if history.riemannian_volume_weights is not None:
        volumes = _to_numpy(history.riemannian_volume_weights[frame])

    # 5. Get neighbor edges
    if history.neighbor_edges is not None and len(history.neighbor_edges) > frame:
        edges_tensor = history.neighbor_edges[frame]
        edge_index = _to_numpy(edges_tensor)  # [E, 2]
        if edge_index.ndim == 1:
            edge_index = edge_index.reshape(-1, 2)
    else:
        raise ValueError("neighbor_edges not available in history")

    # Make undirected: add reverse edges
    edge_index_rev = edge_index[:, ::-1]
    edge_index_full = np.concatenate([edge_index, edge_index_rev], axis=0)
    # Remove duplicates
    edge_set = set()
    unique_edges = []
    for e in range(edge_index_full.shape[0]):
        key = (int(edge_index_full[e, 0]), int(edge_index_full[e, 1]))
        if key not in edge_set:
            edge_set.add(key)
            unique_edges.append(key)
    edge_index_full = np.array(unique_edges, dtype=np.int64)

    neighbors = _edges_to_neighbors(edge_index_full, N)

    # 6. Get fitness gradients
    if history.fitness_gradients is not None:
        gradients = _to_numpy(history.fitness_gradients[frame])  # [N, d]
    else:
        gradients = _compute_gradient_from_neighbors(
            positions, fitness_vals, edge_index_full, config.fd_regularization,
        )

    # 7. Get ricci_scalar_proxy (for cross-check)
    ricci_proxy = None
    if history.ricci_scalar_proxy is not None:
        ricci_proxy = _to_numpy(history.ricci_scalar_proxy[frame])

    # 8. Compute geometric pipeline
    # Use vectorized versions for d <= 6, loop versions for safety
    use_vectorized = d <= 6

    dg = _compute_metric_derivatives_vectorized(
        positions, metrics, edge_index_full, config.fd_regularization,
    )

    christoffels = _compute_christoffel_vectorized(metrics, dg)

    dGamma = _compute_christoffel_derivatives(
        positions, christoffels, edge_index_full, config.fd_regularization,
    )

    riemann = _compute_riemann_vectorized(christoffels, dGamma)

    ricci, R_scalar, einstein = _compute_ricci_and_einstein(riemann, metrics)

    # 9. Compute stress-energy tensor
    T = _compute_stress_energy(
        positions, velocities, fitness_vals, gradients,
        metrics, volumes, neighbors, config.stress_energy_mode,
    )

    # 10. Extract G_N from fractal set regressions
    if g_newton_metric == "manual":
        g_n, g_n_source = g_newton_manual, "manual"
    else:
        g_n, g_n_source = _extract_g_newton(
            fractal_set_regressions, g_newton_metric, g_newton_manual,
        )

    # 11. Validity mask: finite Ricci scalar and positive volume
    valid = np.isfinite(R_scalar) & (volumes > 0) & np.all(np.isfinite(einstein.reshape(N, -1)), axis=1)

    # 12. Scalar test
    scalar_r2, scalar_slope, scalar_intercept = _run_scalar_test(R_scalar, volumes, valid)
    g_n_einstein = scalar_slope / (16.0 * np.pi) if abs(scalar_slope) > 1e-30 else 0.0
    lambda_measured = scalar_intercept / 6.0

    # 13. Tensor test
    tensor_r2, tensor_slope, tensor_r2_per, comp_labels = _run_tensor_test(
        einstein, T, valid, d,
    )

    # 14. Cross-check
    proxy_r2 = None
    if ricci_proxy is not None:
        finite_both = valid & np.isfinite(ricci_proxy)
        if finite_both.sum() >= 3:
            res = stats.linregress(ricci_proxy[finite_both], R_scalar[finite_both])
            proxy_r2 = res.rvalue ** 2

    # 15. Bulk vs boundary
    bulk_mask = _compute_bulk_mask(positions, config.bulk_fraction)
    bulk_r2, _, _ = _run_scalar_test(R_scalar, volumes, valid & bulk_mask)
    boundary_r2, _, _ = _run_scalar_test(R_scalar, volumes, valid & ~bulk_mask)

    # 16. G_N ratio
    g_n_ratio = g_n_einstein / g_n if abs(g_n) > 1e-30 else 0.0

    return EinsteinTestResult(
        positions=positions,
        metrics=metrics,
        ricci_tensor=ricci,
        ricci_scalar=R_scalar,
        einstein_tensor=einstein,
        stress_energy_tensor=T,
        volumes=volumes,
        gradients=gradients,
        scalar_r2=scalar_r2,
        scalar_slope=scalar_slope,
        scalar_intercept=scalar_intercept,
        g_newton_einstein=g_n_einstein,
        lambda_measured=lambda_measured,
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
    )
