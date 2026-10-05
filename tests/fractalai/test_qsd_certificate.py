"""Independent checks of the new primitive QSD certificate and its stage maps."""

import math

import numpy as np
import pytest
from scipy.special import log_ndtr
import torch

from fragile.fractalai.theory.coupled_gas_diagnostics import gaussian_viscous_force
from fragile.fractalai.theory.qsd_certificate import viscous_qsd_primitive_certificate


def reference_parameters(**changes):
    parameters = {
        "n_walkers": 200,
        "dimension": 3,
        "timestep": 0.04,
        "viscosity": 0.3,
        "bandwidth": 1.0,
        "force_lipschitz": 1.0,
        "force_at_origin": 0.0,
        "terminal_half_width": 2.0,
        "clone_jitter": 0.1,
        "restitution": 0.5,
        "velocity_cap": 2.0,
        "ou_coefficient": math.exp(-0.04),
        "ou_amplitude": math.sqrt(-math.expm1(-0.08) / 2),
        "position_amplitude": 0.02,
    }
    return parameters | changes


@pytest.mark.parametrize("row_normalized", [False, True])
def test_unchanged_reference_closes_in_logarithms(row_normalized):
    result = viscous_qsd_primitive_certificate(
        **reference_parameters(row_normalized=row_normalized)
    )
    assert result.certified
    assert not result.limitations
    assert result.coercivity_margin > 0
    assert result.drift_margin > 0
    assert math.isfinite(result.log_eigenfunction_ratio_upper_bound)
    assert result.log_eigenfunction_ratio_upper_bound > 0
    assert result.log_doob_minorization_lower_bound <= result.log_minorization_lower_bound < 0
    assert result.log_tail_probability_upper_bound <= 2 * result.log_survival_lower_bound
    # The mathematical bound is positive even though converting it to float gives zero.
    assert math.exp(result.log_eigenfunction_min_lower_bound) == 0


@pytest.mark.parametrize("row_normalized", [False, True])
@pytest.mark.parametrize("force_profile", [(0.0, 0.0), (2.0, 0.0), (2.5, 0.3)])
def test_flat_anisotropic_and_nonconvex_profiles_are_not_excluded(row_normalized, force_profile):
    lipschitz, at_origin = force_profile
    result = viscous_qsd_primitive_certificate(
        **reference_parameters(
            n_walkers=7,
            force_lipschitz=lipschitz,
            force_at_origin=at_origin,
            row_normalized=row_normalized,
        )
    )
    assert result.certified


def test_absent_jitter_retains_qualitative_bound_but_not_multislot_ratio():
    result = viscous_qsd_primitive_certificate(**reference_parameters(clone_jitter=0))
    assert not result.certified
    assert result.log_minorization_lower_bound is not None
    assert any("positive clone jitter" in reason for reason in result.limitations)
    singleton = viscous_qsd_primitive_certificate(
        **reference_parameters(n_walkers=1, clone_jitter=0, viscosity=1e6, row_normalized=True)
    )
    assert singleton.certified  # The configured singleton force is zero.


def test_pair_row_weights_do_not_require_a_global_degree_floor():
    result = viscous_qsd_primitive_certificate(
        **reference_parameters(n_walkers=2, row_normalized=True, bandwidth=1e-6)
    )
    assert result.certified
    assert result.drift_margin == pytest.approx(1 - 0.02**2)


def test_failed_sufficient_margins_are_reported_without_changing_parameters():
    excessive_force = viscous_qsd_primitive_certificate(
        **reference_parameters(force_lipschitz=3000)
    )
    assert not excessive_force.certified
    assert excessive_force.coercivity_margin < 0
    assert excessive_force.log_minorization_lower_bound is None
    narrow_row = viscous_qsd_primitive_certificate(
        **reference_parameters(row_normalized=True, bandwidth=0.01)
    )
    assert not narrow_row.certified
    assert narrow_row.coercivity_margin > 0
    assert narrow_row.log_minorization_lower_bound is not None
    assert any("first-drift" in reason for reason in narrow_row.limitations)
    insufficient_tail = viscous_qsd_primitive_certificate(
        **reference_parameters(jitter_event_radius=0.01, noise_event_radius=1)
    )
    assert not insufficient_tail.certified
    assert any("discarded-event" in reason for reason in insufficient_tail.limitations)


@pytest.mark.parametrize(
    "changes",
    [
        {"ou_amplitude": 0},
        {"clone_jitter": -1},
        {"force_lipschitz": math.inf},
        {"n_walkers": True},
        {"dimension": 1.5},
        {"ou_coefficient": 1.1},
        {"noise_event_radius": math.nan},
    ],
)
def test_invalid_parameters_are_rejected(changes):
    with pytest.raises(ValueError):
        viscous_qsd_primitive_certificate(**reference_parameters(**changes))


@pytest.mark.parametrize("row_normalized", [False, True])
def test_actual_nonconvex_drift_and_fixed_position_second_kick_have_bounded_determinants(
    row_normalized,
):
    """Differentiate the real maps for F=-x+2*tanh(x), a two-valley force."""
    n, d = 4, 2
    t, viscosity = 0.02, 0.3
    result = viscous_qsd_primitive_certificate(
        **reference_parameters(
            n_walkers=n, dimension=d, force_lipschitz=3, row_normalized=row_normalized
        )
    )
    generator = torch.Generator().manual_seed(918)
    x = torch.randn((n, d), generator=generator, dtype=torch.float64)
    v = torch.randn((n, d), generator=generator, dtype=torch.float64)
    v *= 2 / (2 + v.norm(dim=1, keepdim=True))

    def drift(flat_x):
        positions = flat_x.reshape(n, d)
        force = -positions + 2 * positions.tanh()
        viscosity_force = gaussian_viscous_force(
            positions, v, viscosity=viscosity, bandwidth=1, row_normalized=row_normalized
        )
        return (positions + t * v + t * t * (force + viscosity_force)).reshape(-1)

    derivative = torch.autograd.functional.jacobian(drift, x.reshape(-1))
    assert float(derivative.det()) >= result.drift_margin ** (n * d)

    def second_kick(flat_v):
        velocities = flat_v.reshape(n, d)
        viscosity_force = gaussian_viscous_force(
            x, velocities, viscosity=viscosity, bandwidth=1, row_normalized=row_normalized
        )
        return (velocities + t * (-x + 2 * x.tanh() + viscosity_force)).reshape(-1)

    derivative = torch.autograd.functional.jacobian(second_kick, v.reshape(-1))
    factor = 2 if row_normalized else 1
    assert float(derivative.det()) >= (1 - factor * t * viscosity) ** (n * d)


@pytest.mark.parametrize("row_normalized", [False, True])
def test_survival_floor_against_exact_boundary_landing_probabilities(row_normalized):
    """Use extreme box positions and bounded velocities, integrating the actual law."""
    parameters = reference_parameters(n_walkers=4, row_normalized=row_normalized)
    result = viscous_qsd_primitive_certificate(**parameters)
    x = torch.tensor([[2, 2, 2], [-2, -2, -2], [2, -2, 2], [-2, 2, -2]], dtype=torch.float64)
    v = torch.tensor([[4, 0, 0], [-4, 0, 0], [0, 4, 0], [0, -4, 0]], dtype=torch.float64)
    force = gaussian_viscous_force(x, v, viscosity=0.3, bandwidth=1, row_normalized=row_normalized)
    v1 = v + 0.02 * (-x + force)
    means = (x + 0.02 * (1 + math.exp(-0.04)) * v1).numpy()
    sd = math.hypot(0.02 * parameters["ou_amplitude"], 0.02)
    for mean in means:
        # Use symmetry to keep both CDF arguments on the negative half-line.
        upper = log_ndtr((2 - np.abs(mean)) / sd)
        lower = log_ndtr((-2 - np.abs(mean)) / sd)
        log_probability = np.sum(upper + np.log(-np.expm1(lower - upper)))
        assert log_probability >= result.log_survival_lower_bound


def test_density_comparison_bound_against_direct_finite_perron_eigenproblems():
    """Check the comparison argument on unrelated killed kernels with exact densities."""
    generator = np.random.default_rng(78)
    for size in (2, 5, 9):
        kernel = generator.uniform(0.01, 1, size=(size, size))
        kernel /= 1.5 * kernel.sum(axis=1, keepdims=True)
        kernel *= generator.uniform(0.2, 1, size=(size, 1))
        energies, vectors = np.linalg.eig(kernel)
        index = np.argmax(energies.real)
        eigenfunction = vectors[:, index].real
        eigenfunction /= eigenfunction.max()
        if eigenfunction.min() < 0:
            eigenfunction = vectors[:, index].real / vectors[:, index].real.min()
        eigenfunction /= eigenfunction.max()
        survival_floor = kernel.sum(axis=1).min()
        density_lower = kernel.min()
        density_upper = (kernel @ kernel).max()
        primitive_minimum = density_lower * survival_floor**2 / (2 * density_upper)
        assert eigenfunction.min() >= primitive_minimum
