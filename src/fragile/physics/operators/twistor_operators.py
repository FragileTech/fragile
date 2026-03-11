"""Effective twistor-inspired operators on companion geometry.

This module does not claim a fundamental twistor reconstruction from walkers.
Instead it provides a deterministic, geometry-driven effective map from
companion edge data to normalized two-spinors and the associated twistor-like
invariants used by the companion correlator channels.

The construction is:

1. For an anchor walker ``i`` and companion ``j``, build an effective
   Lorentzian-style edge four-vector ``X_ij = (dt, dx)`` and a velocity
   four-vector ``V_ij = (0, dv)`` from the first three spatial dimensions.
2. Map both through the sigma-matrix map to obtain ``2x2`` bispinors.
3. Form a complex edge bispinor ``B_ij = X_ij + i * velocity_scale * V_ij``.
4. Extract a normalized spinor from the dominant column of ``B_ij`` and use the
   spinor bracket between the distance and clone companion edges as the local
   twistor-like invariant.

The resulting scalar, pseudoscalar, and ``|.|^2`` glueball-like observables are
used as operator families in companion-tracking correlators.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fragile.physics.qft_utils import build_companion_triplets, resolve_3d_dims, safe_gather_2d
from fragile.physics.qft_utils.helpers import safe_gather_3d


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return a complex dtype compatible with *dtype*."""
    if dtype in {torch.float64, torch.complex128}:
        return torch.complex128
    return torch.complex64


def _sigma_matrices(*, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return the four sigma matrices as a ``[4, 2, 2]`` tensor."""
    complex_dtype = _complex_dtype(dtype)
    zero = torch.tensor(0.0, device=device, dtype=complex_dtype)
    one = torch.tensor(1.0, device=device, dtype=complex_dtype)
    imag = torch.tensor(1.0j, device=device, dtype=complex_dtype)
    return torch.stack(
        [
            torch.stack([torch.stack([one, zero]), torch.stack([zero, one])]),
            torch.stack([torch.stack([zero, one]), torch.stack([one, zero])]),
            torch.stack([torch.stack([zero, -imag]), torch.stack([imag, zero])]),
            torch.stack([torch.stack([one, zero]), torch.stack([zero, -one])]),
        ],
        dim=0,
    )


def _select_spatial_components(
    values: Tensor,
    spatial_dims: tuple[int, int, int] | None = None,
) -> Tensor:
    """Select exactly three spatial components from ``[..., d]`` values."""
    dims = resolve_3d_dims(values.shape[-1], spatial_dims, "spatial_dims")
    return values[..., list(dims)]


def _expand_delta_t(
    *,
    delta_t: float | Tensor,
    shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Expand a scalar or per-frame ``delta_t`` to shape ``[T, N, 1]``."""
    dt = torch.as_tensor(delta_t, device=device, dtype=dtype)
    if dt.ndim == 0:
        return dt.expand(*shape, 1)
    if dt.ndim == 1 and int(dt.shape[0]) == int(shape[0]):
        return dt.view(shape[0], 1, 1).expand(*shape, 1)
    msg = (
        "delta_t must be a scalar or a length-T tensor matching the first "
        f"dimension of the input. Got shape {tuple(dt.shape)} for target {shape}."
    )
    raise ValueError(msg)


def sigma_map_four_vector(four_vector: Tensor) -> Tensor:
    """Map a real or complex ``[..., 4]`` four-vector to a ``[..., 2, 2]`` bispinor."""
    if four_vector.shape[-1] != 4:
        raise ValueError(
            f"four_vector must have trailing dimension 4, got {tuple(four_vector.shape)}."
        )
    sigma = _sigma_matrices(device=four_vector.device, dtype=four_vector.dtype)
    vec = four_vector.to(dtype=sigma.dtype)
    return torch.einsum("...m,mab->...ab", vec, sigma)


def build_edge_four_vectors(
    positions: Tensor,
    velocities: Tensor,
    companion_indices: Tensor,
    *,
    delta_t: float | Tensor,
    spatial_dims: tuple[int, int, int] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build edge displacement and velocity four-vectors for companion edges.

    Returns:
        displacement_4: ``[T, N, 4]`` with components ``(dt, dx1, dx2, dx3)``.
        velocity_4: ``[T, N, 4]`` with components ``(0, dv1, dv2, dv3)``.
        valid: finite/in-range mask ``[T, N]``.
    """
    if positions.ndim != 3 or velocities.ndim != 3:
        raise ValueError(
            f"positions and velocities must have shape [T, N, d], got "
            f"{tuple(positions.shape)} and {tuple(velocities.shape)}."
        )
    if positions.shape != velocities.shape:
        raise ValueError(
            f"positions and velocities must share the same shape, got "
            f"{tuple(positions.shape)} vs {tuple(velocities.shape)}."
        )
    if companion_indices.shape != positions.shape[:2]:
        raise ValueError(
            "companion_indices must have shape [T, N] aligned with positions, got "
            f"{tuple(companion_indices.shape)} vs {tuple(positions.shape[:2])}."
        )

    pos3 = _select_spatial_components(positions, spatial_dims).float()
    vel3 = _select_spatial_components(velocities, spatial_dims).float()
    pos_j, in_range_pos = safe_gather_3d(pos3, companion_indices)
    vel_j, in_range_vel = safe_gather_3d(vel3, companion_indices)

    dx = pos_j - pos3
    dv = vel_j - vel3

    dt = _expand_delta_t(
        delta_t=delta_t,
        shape=positions.shape[:2],
        device=positions.device,
        dtype=pos3.dtype,
    )
    zeros = torch.zeros_like(dt)

    displacement_4 = torch.cat([dt, dx], dim=-1)
    velocity_4 = torch.cat([zeros, dv], dim=-1)

    finite = torch.isfinite(displacement_4).all(dim=-1) & torch.isfinite(velocity_4).all(dim=-1)
    valid = in_range_pos & in_range_vel & finite
    return displacement_4, velocity_4, valid


def build_effective_edge_bispinor(
    displacement_4: Tensor,
    velocity_4: Tensor,
    *,
    velocity_scale: float = 1.0,
) -> Tensor:
    """Build the complex edge bispinor ``B = X + i * velocity_scale * V``."""
    x_matrix = sigma_map_four_vector(displacement_4)
    v_matrix = sigma_map_four_vector(velocity_4)
    scale = torch.as_tensor(
        complex(0.0, float(velocity_scale)),
        device=x_matrix.device,
        dtype=x_matrix.dtype,
    )
    return x_matrix + scale * v_matrix


def build_effective_edge_twistor(
    bispinor: Tensor,
    displacement_4: Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """Extract a normalized effective twistor ``(mu, lambda)`` from a bispinor."""
    if bispinor.shape[-2:] != (2, 2):
        raise ValueError(f"bispinor must have trailing shape [2, 2], got {tuple(bispinor.shape)}.")
    if displacement_4.shape[:-1] != bispinor.shape[:-2]:
        raise ValueError(
            "displacement_4 must align with bispinor batch dimensions, got "
            f"{tuple(displacement_4.shape)} vs {tuple(bispinor.shape)}."
        )

    col0 = bispinor[..., :, 0]
    col1 = bispinor[..., :, 1]
    norm0 = col0.abs().square().sum(dim=-1)
    norm1 = col1.abs().square().sum(dim=-1)
    use_col1 = norm1 > norm0
    lam = torch.where(use_col1.unsqueeze(-1), col1, col0)

    lam_norm = lam.abs().square().sum(dim=-1).sqrt()
    lam_valid = (
        torch.isfinite(lam.real).all(dim=-1)
        & torch.isfinite(lam.imag).all(dim=-1)
        & (lam_norm > float(eps))
    )
    lam_safe = torch.where(
        lam_valid.unsqueeze(-1),
        lam / lam_norm.clamp(min=float(eps)).unsqueeze(-1),
        torch.zeros_like(lam),
    )

    x_matrix = sigma_map_four_vector(displacement_4).to(dtype=lam_safe.dtype)
    mu = torch.einsum("...ab,...b->...a", x_matrix, lam_safe)
    mu = torch.where(lam_valid.unsqueeze(-1), mu, torch.zeros_like(mu))

    twistor = torch.cat([mu, lam_safe], dim=-1)
    return twistor, lam_valid


def infinity_twistor_bracket(twistor_a: Tensor, twistor_b: Tensor) -> Tensor:
    """Return the chiral twistor bracket from the ``lambda`` components."""
    if twistor_a.shape[-1] != 4 or twistor_b.shape[-1] != 4:
        raise ValueError(
            f"twistor inputs must have trailing dimension 4, got "
            f"{tuple(twistor_a.shape)} and {tuple(twistor_b.shape)}."
        )
    lam_a = twistor_a[..., 2:]
    lam_b = twistor_b[..., 2:]
    return lam_a[..., 0] * lam_b[..., 1] - lam_a[..., 1] * lam_b[..., 0]


def twistor_vector_bilinear(twistor_a: Tensor, twistor_b: Tensor) -> Tensor:
    """Return the complex Pauli-vector bilinear from the twistor ``lambda`` components."""
    if twistor_a.shape[-1] != 4 or twistor_b.shape[-1] != 4:
        raise ValueError(
            f"twistor inputs must have trailing dimension 4, got "
            f"{tuple(twistor_a.shape)} and {tuple(twistor_b.shape)}."
        )
    lam_a = twistor_a[..., 2:]
    lam_b = twistor_b[..., 2:]
    sigma = _sigma_matrices(device=twistor_a.device, dtype=twistor_a.dtype)[1:]
    return torch.einsum("...a,iab,...b->...i", lam_a.conj(), sigma, lam_b)


def twistor_tensor_components(vector_bilinear: Tensor) -> Tensor:
    """Return five real spin-2-like components from a complex spatial bilinear."""
    if vector_bilinear.shape[-1] != 3:
        raise ValueError(
            f"vector_bilinear must have trailing dimension 3, got {tuple(vector_bilinear.shape)}."
        )

    x = vector_bilinear[..., 0]
    y = vector_bilinear[..., 1]
    z = vector_bilinear[..., 2]

    inv_sqrt2 = float(1.0 / torch.sqrt(torch.tensor(2.0)).item())
    inv_sqrt6 = float(1.0 / torch.sqrt(torch.tensor(6.0)).item())

    return torch.stack(
        [
            (x * y).real.float(),
            (x * z).real.float(),
            (y * z).real.float(),
            ((x * x - y * y).real * inv_sqrt2).float(),
            ((2.0 * z * z - x * x - y * y).real * inv_sqrt6).float(),
        ],
        dim=-1,
    )


def _build_twistor_triplet_state(
    positions: Tensor,
    velocities: Tensor,
    alive_mask: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    *,
    delta_t: float | Tensor,
    spatial_dims: tuple[int, int, int] | None = None,
    velocity_scale: float = 1.0,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build distance/clone edge twistors and a common validity mask."""
    if positions.ndim != 3 or velocities.ndim != 3:
        raise ValueError(
            f"positions and velocities must have shape [T, N, d], got "
            f"{tuple(positions.shape)} and {tuple(velocities.shape)}."
        )
    if positions.shape != velocities.shape:
        raise ValueError(
            f"positions and velocities must share shape, got "
            f"{tuple(positions.shape)} vs {tuple(velocities.shape)}."
        )
    if alive_mask.shape != positions.shape[:2]:
        raise ValueError(
            f"alive_mask must have shape [T, N], got {tuple(alive_mask.shape)}."
        )
    if companions_distance.shape != positions.shape[:2] or companions_clone.shape != positions.shape[:2]:
        raise ValueError(
            "companion arrays must have shape [T, N] aligned with positions, got "
            f"{tuple(companions_distance.shape)} and {tuple(companions_clone.shape)}."
        )

    anchor_idx, companion_j, companion_k, structural_valid = build_companion_triplets(
        companions_distance=companions_distance,
        companions_clone=companions_clone,
    )

    displacement_j, velocity_j, edge_valid_j = build_edge_four_vectors(
        positions,
        velocities,
        companion_j,
        delta_t=delta_t,
        spatial_dims=spatial_dims,
    )
    displacement_k, velocity_k, edge_valid_k = build_edge_four_vectors(
        positions,
        velocities,
        companion_k,
        delta_t=delta_t,
        spatial_dims=spatial_dims,
    )

    bispinor_j = build_effective_edge_bispinor(
        displacement_j,
        velocity_j,
        velocity_scale=velocity_scale,
    )
    bispinor_k = build_effective_edge_bispinor(
        displacement_k,
        velocity_k,
        velocity_scale=velocity_scale,
    )

    twistor_j, twistor_valid_j = build_effective_edge_twistor(
        bispinor_j,
        displacement_j,
        eps=eps,
    )
    twistor_k, twistor_valid_k = build_effective_edge_twistor(
        bispinor_k,
        displacement_k,
        eps=eps,
    )

    alive = alive_mask.to(dtype=torch.bool, device=positions.device)
    alive_j, in_range_j = safe_gather_2d(alive, companion_j)
    alive_k, in_range_k = safe_gather_2d(alive, companion_k)

    finite_anchor = (
        torch.isfinite(_select_spatial_components(positions, spatial_dims)).all(dim=-1)
        & torch.isfinite(_select_spatial_components(velocities, spatial_dims)).all(dim=-1)
    )
    valid = (
        structural_valid
        & alive
        & alive_j
        & alive_k
        & in_range_j
        & in_range_k
        & edge_valid_j
        & edge_valid_k
        & twistor_valid_j
        & twistor_valid_k
        & finite_anchor
        & (companion_j != anchor_idx)
        & (companion_k != anchor_idx)
    )
    return twistor_j, twistor_k, valid


def compute_twistor_triplet_fields(
    positions: Tensor,
    velocities: Tensor,
    alive_mask: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    *,
    delta_t: float | Tensor,
    spatial_dims: tuple[int, int, int] | None = None,
    velocity_scale: float = 1.0,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute the full effective twistor companion operator family.

    Returns:
        scalar: ``Re(<lambda_ij, lambda_ik>)``.
        pseudoscalar: ``Im(<lambda_ij, lambda_ik>)``.
        glueball: ``|<lambda_ij, lambda_ik>|^2``.
        vector: ``Re(lambda_ij^dagger sigma lambda_ik)`` as ``[..., 3]``.
        axial_vector: ``Im(lambda_ij^dagger sigma lambda_ik)`` as ``[..., 3]``.
        tensor: Mean of the five real spin-2-like components built from the
            complex vector bilinear.
        valid: Structural/finite/alive mask.
    """
    twistor_j, twistor_k, valid = _build_twistor_triplet_state(
        positions,
        velocities,
        alive_mask,
        companions_distance,
        companions_clone,
        delta_t=delta_t,
        spatial_dims=spatial_dims,
        velocity_scale=velocity_scale,
        eps=eps,
    )

    pairing = infinity_twistor_bracket(twistor_j, twistor_k)
    pairing = torch.where(valid, pairing, torch.zeros_like(pairing))

    vector_bilinear = twistor_vector_bilinear(twistor_j, twistor_k)
    vector_bilinear = torch.where(
        valid.unsqueeze(-1),
        vector_bilinear,
        torch.zeros_like(vector_bilinear),
    )
    tensor_components = twistor_tensor_components(vector_bilinear)
    tensor = tensor_components.mean(dim=-1)

    scalar = pairing.real.float()
    pseudoscalar = pairing.imag.float()
    glueball = pairing.abs().square().float()
    vector = vector_bilinear.real.float()
    axial_vector = vector_bilinear.imag.float()
    return scalar, pseudoscalar, glueball, vector, axial_vector, tensor, valid


def compute_twistor_triplet_observables(
    positions: Tensor,
    velocities: Tensor,
    alive_mask: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    *,
    delta_t: float | Tensor,
    spatial_dims: tuple[int, int, int] | None = None,
    velocity_scale: float = 1.0,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute scalar, pseudoscalar, and glueball-like triplet observables.

    The anchor walker ``i`` is paired with its source-frame distance companion
    ``j`` and clone companion ``k``. Two effective twistors are built from the
    edges ``(i, j)`` and ``(i, k)``, then the local invariant

        ``tau_i = <lambda_ij, lambda_ik>``

    is decomposed into:
    - scalar: ``Re(tau_i)``
    - pseudoscalar: ``Im(tau_i)``
    - glueball-like scalar: ``|tau_i|^2``
    """
    scalar, pseudoscalar, glueball, _vector, _axial, _tensor, valid = (
        compute_twistor_triplet_fields(
            positions,
            velocities,
            alive_mask,
            companions_distance,
            companions_clone,
            delta_t=delta_t,
            spatial_dims=spatial_dims,
            velocity_scale=velocity_scale,
            eps=eps,
        )
    )
    return scalar, pseudoscalar, glueball, valid
