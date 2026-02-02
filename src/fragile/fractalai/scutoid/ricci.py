"""Fast Ricci scalar proxies for scutoid metrics."""

from __future__ import annotations

import torch
from torch import Tensor


def _prepare_u(metric_det: Tensor, spatial_dim: int, eps: float) -> Tensor:
    det_safe = torch.clamp(metric_det, min=eps)
    return 0.5 * torch.log(det_safe) / float(spatial_dim)


def _fit_local_quadratic(
    positions: Tensor,
    u: Tensor,
    edge_index: Tensor,
    edge_weights: Tensor | None,
    reg: float,
) -> tuple[Tensor, Tensor]:
    """Fit u via weighted local quadratic model per node."""
    n_nodes, d = positions.shape
    device = positions.device
    dtype = positions.dtype

    if edge_index.numel() == 0:
        grad = torch.zeros(n_nodes, d, device=device, dtype=dtype)
        hess = torch.zeros(n_nodes, d, d, device=device, dtype=dtype)
        return grad, hess

    src, dst = edge_index
    delta = positions[dst] - positions[src]  # [E, d]
    y = u[dst] - u[src]  # [E]

    if edge_weights is None:
        edge_weights = torch.ones_like(y)

    triu_i, triu_j = torch.triu_indices(d, d, device=device)
    phi_hess = delta[:, triu_i] * delta[:, triu_j]
    diag_mask = triu_i == triu_j
    if diag_mask.any():
        phi_hess = torch.where(diag_mask, 0.5 * phi_hess, phi_hess)

    phi = torch.cat([delta, phi_hess], dim=1)  # [E, p]
    p = phi.shape[1]

    w = edge_weights.to(dtype=dtype).unsqueeze(1)
    phi_w = phi * w

    outer = phi_w[:, :, None] * phi[:, None, :]  # [E, p, p]
    outer_flat = outer.reshape(phi.shape[0], p * p)
    xtx_flat = torch.zeros(n_nodes, p * p, device=device, dtype=dtype)
    xtx_flat.index_add_(0, src, outer_flat)
    xtx = xtx_flat.view(n_nodes, p, p)

    x_ty = torch.zeros(n_nodes, p, device=device, dtype=dtype)
    x_ty.index_add_(0, src, phi_w * y.unsqueeze(1))

    eye = torch.eye(p, device=device, dtype=dtype).unsqueeze(0)
    xtx = xtx + reg * eye
    beta = torch.linalg.solve(xtx, x_ty.unsqueeze(-1)).squeeze(-1)

    grad = beta[:, :d]
    hess_params = beta[:, d:]
    hess = torch.zeros(n_nodes, d, d, device=device, dtype=dtype)
    hess[:, triu_i, triu_j] = hess_params
    hess[:, triu_j, triu_i] = hess_params
    return grad, hess


def compute_ricci_proxy(
    metric_det: Tensor,
    edge_index: Tensor,
    edge_weights: Tensor,
    spatial_dim: int,
) -> Tensor:
    """Compute a fast Ricci scalar proxy via a graph Laplacian on log det(g).

    Uses the conformal approximation:
        u = (1 / (2d)) log det(g),  R ≈ -2(d-1) Δu
    where Δ is a weighted graph Laplacian (edge_weights).
    """
    if metric_det.numel() == 0:
        return torch.empty(0, device=metric_det.device, dtype=metric_det.dtype)
    if edge_index.numel() == 0:
        return torch.zeros_like(metric_det)

    u = _prepare_u(metric_det, spatial_dim, eps=1e-12)

    src, dst = edge_index
    diff = u[dst] - u[src]
    contrib = edge_weights * diff

    lap = torch.zeros_like(u)
    lap.scatter_add_(0, src, contrib)

    return -2.0 * (float(spatial_dim) - 1.0) * lap


def compute_ricci_proxy_full_metric(
    positions: Tensor,
    metric_tensors: Tensor,
    edge_index: Tensor,
    edge_weights: Tensor | None = None,
    metric_det: Tensor | None = None,
    spatial_dim: int | None = None,
    reg: float = 1e-6,
    eps: float = 1e-12,
    include_conformal_factor: bool = True,
) -> Tensor:
    """Compute a richer Ricci proxy using the full metric tensor.

    This uses a conformal-style approximation with a graph Laplacian and
    a gradient-norm term:
        u = (1 / (2d)) log det(g)
        R ≈ -2(d-1) e^{-2u} (Δu + (d-2) |∇u|^2)

    The gradient is estimated via weighted least squares from neighbors.
    """
    if spatial_dim is None:
        spatial_dim = positions.shape[1]

    if metric_det is None:
        sign, logdet = torch.linalg.slogdet(metric_tensors)
        det = torch.exp(logdet)
        metric_det = torch.where(sign > 0, det, torch.zeros_like(det))

    if metric_det.numel() == 0:
        return torch.empty(0, device=metric_det.device, dtype=metric_det.dtype)
    if edge_index.numel() == 0:
        return torch.zeros_like(metric_det)

    u = _prepare_u(metric_det, spatial_dim, eps=eps)

    grad_u, hess_u = _fit_local_quadratic(
        positions=positions,
        u=u,
        edge_index=edge_index,
        edge_weights=edge_weights,
        reg=reg,
    )

    lap = torch.diagonal(hess_u, dim1=1, dim2=2).sum(dim=1)
    g_inv_grad = torch.linalg.solve(metric_tensors, grad_u.unsqueeze(-1)).squeeze(-1)
    grad_norm_sq = (grad_u * g_inv_grad).sum(dim=1)

    scale = torch.exp(-2.0 * u) if include_conformal_factor else 1.0
    d = float(spatial_dim)
    return -2.0 * (d - 1.0) * scale * (lap + (d - 2.0) * grad_norm_sq)


def compute_ricci_tensor_proxy(
    metric_tensors: Tensor,
    ricci_scalar: Tensor,
    spatial_dim: int | None = None,
) -> Tensor:
    """Build an isotropic Ricci tensor proxy from a scalar curvature field.

    This enforces trace consistency: g^{ij} R_ij = R by setting
    R_ij = (R / d) * g_ij.
    """
    if spatial_dim is None:
        spatial_dim = metric_tensors.shape[-1]

    scale = ricci_scalar / float(spatial_dim)
    return metric_tensors * scale[:, None, None]


def compute_ricci_tensor_proxy_full_metric(
    positions: Tensor,
    metric_tensors: Tensor,
    edge_index: Tensor,
    edge_weights: Tensor | None = None,
    metric_det: Tensor | None = None,
    spatial_dim: int | None = None,
    reg: float = 1e-6,
    eps: float = 1e-12,
    include_conformal_factor: bool = True,
) -> tuple[Tensor, Tensor]:
    """Compute an anisotropic Ricci tensor proxy from the full metric.

    Uses a conformal-style approximation in coordinate basis:
        R_ij ≈ -(d-2)(u_ij - u_i u_j) - (Δu + (d-2)|∇u|^2) δ_ij
    with u = (1 / (2d)) log det(g) and derivatives estimated from neighbors.

    Returns:
        ricci_tensor: [N, d, d] anisotropic proxy
        ricci_scalar: [N] scalar proxy consistent with contraction by g^{-1}
    """
    if spatial_dim is None:
        spatial_dim = positions.shape[1]

    if metric_det is None:
        sign, logdet = torch.linalg.slogdet(metric_tensors)
        det = torch.exp(logdet)
        metric_det = torch.where(sign > 0, det, torch.zeros_like(det))

    if metric_det.numel() == 0:
        empty = torch.empty(0, device=metric_tensors.device, dtype=metric_tensors.dtype)
        return empty, empty

    if edge_index.numel() == 0:
        zeros_scalar = torch.zeros_like(metric_det)
        zeros_tensor = torch.zeros(
            metric_det.shape[0],
            spatial_dim,
            spatial_dim,
            device=metric_det.device,
            dtype=metric_det.dtype,
        )
        return zeros_tensor, zeros_scalar

    u = _prepare_u(metric_det, spatial_dim, eps=eps)

    grad_u, hess_u = _fit_local_quadratic(
        positions=positions,
        u=u,
        edge_index=edge_index,
        edge_weights=edge_weights,
        reg=reg,
    )

    lap = torch.diagonal(hess_u, dim1=1, dim2=2).sum(dim=1)
    g_inv_grad = torch.linalg.solve(metric_tensors, grad_u.unsqueeze(-1)).squeeze(-1)
    grad_norm_sq = (grad_u * g_inv_grad).sum(dim=1)

    d = float(spatial_dim)
    eye = torch.eye(spatial_dim, device=positions.device, dtype=positions.dtype).unsqueeze(0)

    ricci_tensor = -(d - 2.0) * (hess_u - grad_u[:, :, None] * grad_u[:, None, :])
    ricci_tensor = ricci_tensor - (lap + (d - 2.0) * grad_norm_sq)[:, None, None] * eye

    scale = torch.exp(-2.0 * u) if include_conformal_factor else 1.0
    ricci_scalar = -2.0 * (d - 1.0) * scale * (lap + (d - 2.0) * grad_norm_sq)

    return ricci_tensor, ricci_scalar
