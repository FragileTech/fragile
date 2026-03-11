"""Tests for effective twistor-inspired operator helpers."""

from __future__ import annotations

import torch

from fragile.physics.operators.twistor_operators import (
    build_edge_four_vectors,
    build_effective_edge_bispinor,
    build_effective_edge_twistor,
    compute_twistor_triplet_fields,
    compute_twistor_triplet_observables,
    sigma_map_four_vector,
)


def _sample_geometry(T: int = 4, N: int = 5, d: int = 4) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(123)
    positions = torch.randn(T, N, d, generator=gen)
    velocities = torch.randn(T, N, d, generator=gen)
    alive = torch.ones(T, N, dtype=torch.bool)
    comp_d = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).clone()
    comp_c = torch.arange(N).roll(-2).unsqueeze(0).expand(T, -1).clone()
    return positions, velocities, alive, comp_d, comp_c


def test_sigma_map_four_vector_shape() -> None:
    four_vec = torch.tensor([[1.0, 0.25, -0.5, 0.75]], dtype=torch.float32)
    mapped = sigma_map_four_vector(four_vec)
    assert mapped.shape == (1, 2, 2)
    assert mapped.dtype.is_complex


def test_edge_bispinor_and_twistor_are_finite() -> None:
    positions, velocities, _alive, comp_d, _comp_c = _sample_geometry(T=3, N=4)
    dx4, dv4, valid = build_edge_four_vectors(
        positions,
        velocities,
        comp_d,
        delta_t=1.0,
    )
    bispinor = build_effective_edge_bispinor(dx4, dv4, velocity_scale=0.5)
    twistor, twistor_valid = build_effective_edge_twistor(bispinor, dx4)
    assert bispinor.shape == (3, 4, 2, 2)
    assert twistor.shape == (3, 4, 4)
    assert bool(torch.isfinite(bispinor.real).all())
    assert bool(torch.isfinite(bispinor.imag).all())
    assert bool(torch.isfinite(twistor.real).all())
    assert bool(torch.isfinite(twistor.imag).all())
    assert torch.equal(twistor_valid, valid)


def test_triplet_observables_are_finite_and_consistent() -> None:
    positions, velocities, alive, comp_d, comp_c = _sample_geometry()
    scalar, pseudoscalar, glueball, valid = compute_twistor_triplet_observables(
        positions=positions,
        velocities=velocities,
        alive_mask=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        delta_t=1.0,
        velocity_scale=0.75,
    )
    assert scalar.shape == positions.shape[:2]
    assert pseudoscalar.shape == positions.shape[:2]
    assert glueball.shape == positions.shape[:2]
    assert valid.shape == positions.shape[:2]
    assert bool(torch.isfinite(scalar).all())
    assert bool(torch.isfinite(pseudoscalar).all())
    assert bool(torch.isfinite(glueball).all())
    assert bool((glueball >= 0).all())
    if torch.any(valid):
        lhs = glueball[valid]
        rhs = scalar[valid].square() + pseudoscalar[valid].square()
        assert torch.allclose(lhs, rhs, atol=1e-5, rtol=1e-5)


def test_triplet_fields_include_vector_axial_and_tensor() -> None:
    positions, velocities, alive, comp_d, comp_c = _sample_geometry()
    scalar, pseudoscalar, glueball, vector, axial, tensor, valid = compute_twistor_triplet_fields(
        positions=positions,
        velocities=velocities,
        alive_mask=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        delta_t=1.0,
        velocity_scale=0.6,
    )
    assert scalar.shape == positions.shape[:2]
    assert pseudoscalar.shape == positions.shape[:2]
    assert glueball.shape == positions.shape[:2]
    assert vector.shape == (*positions.shape[:2], 3)
    assert axial.shape == (*positions.shape[:2], 3)
    assert tensor.shape == positions.shape[:2]
    assert valid.shape == positions.shape[:2]
    assert bool(torch.isfinite(vector).all())
    assert bool(torch.isfinite(axial).all())
    assert bool(torch.isfinite(tensor).all())


def test_triplet_observables_mask_invalid_companions() -> None:
    positions, velocities, alive, comp_d, comp_c = _sample_geometry(T=2, N=4)
    comp_d[0, 0] = -1
    comp_c[1, 1] = 99
    scalar, pseudoscalar, glueball, valid = compute_twistor_triplet_observables(
        positions=positions,
        velocities=velocities,
        alive_mask=alive,
        companions_distance=comp_d,
        companions_clone=comp_c,
        delta_t=1.0,
    )
    assert not bool(valid[0, 0].item())
    assert not bool(valid[1, 1].item())
    assert scalar[0, 0].item() == 0.0
    assert pseudoscalar[1, 1].item() == 0.0
    assert glueball[0, 0].item() == 0.0
