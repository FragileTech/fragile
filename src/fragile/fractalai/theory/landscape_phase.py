"""Landscape observations for the existing regional Keystone certificates.

These maps observe the unchanged kernel. They retain phase occupation and
Gaussian crossings; they do not require distinct wells to contract together.
Root values are floating diagnostics of the analytic brackets in Chapter 18a.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from scipy.optimize import brentq
from scipy.special import ndtr
import torch
from torch import Tensor

from fragile.fractalai.theory.coupled_gas_diagnostics import gaussian_viscous_force
from fragile.fractalai.theory.keystone_uniform import _arrays


def rastrigin_haar_potential_mean(
    position_mean: Tensor,
    component_radii: Tensor,
    *,
    orbital_scale: float,
    gaussian_variance: float,
) -> Tensor:
    """Exact d=3 potential after component Haar rotations, KULN.79.

    Rows are m_i + b*sum_C O_C*B_iC + sqrt(variance)*Z_i. Supply
    the actual m_i=g(X_i)+b*M_i and |B_iC| from the frozen source,
    first-viscosity and component-mean/deviation account. One O_C is
    shared by every row in component C; this marginal mean does not
    assert independent row outputs. Both original position noises are
    integrated in their combined variance, without a tail cutoff.
    """
    _arrays(position_mean, component_radii)
    if (
        position_mean.ndim != 2
        or position_mean.shape[1] != 3
        or component_radii.ndim != 2
        or component_radii.shape[0] != position_mean.shape[0]
    ):
        msg = "position means must be [N,3] and component radii [N,C]"
        raise ValueError(msg)
    if (
        (component_radii < 0).any()
        or not math.isfinite(orbital_scale)
        or not math.isfinite(gaussian_variance)
        or gaussian_variance < 0
    ):
        msg = "component radii and Gaussian variance must be nonnegative and finite"
        raise ValueError(msg)
    squared_position = (
        position_mean.square().sum(dim=1)
        + orbital_scale**2 * component_radii.square().sum(dim=1)
        + 3 * gaussian_variance
    )
    characteristic = torch.sinc(2 * orbital_scale * component_radii).prod(dim=1)
    characteristic *= math.exp(-2 * math.pi**2 * gaussian_variance)
    return (
        squared_position
        + 30
        - 10 * characteristic * torch.cos(2 * math.pi * position_mean).sum(dim=1)
    )


def radial_cap_second_moment_bound(mean_square: Tensor, *, velocity_cap: float) -> Tensor:
    """Jensen bound for the original radial map V*w/(V+|w|).

    ``mean_square`` is the integrated uncapped squared norm. The same
    bound applies to a normalized population average, without independent
    rows and without dropping any Gaussian tails.
    """
    _arrays(mean_square)
    if (mean_square < 0).any() or not math.isfinite(velocity_cap) or velocity_cap <= 0:
        msg = "a nonnegative moment and a positive finite cap are required"
        raise ValueError(msg)
    radius = torch.sqrt(mean_square)
    return (velocity_cap * radius / (velocity_cap + radius)) ** 2


@dataclass(frozen=True)
class RastriginGaussianEnergy:
    """Full-Gaussian BAOAB energy before the actual B2 graph and radial cap.

    The OU-position covariance is included. The graph work and cap loss
    must be added as in the joint-energy identity in Chapter 18a.
    """

    position_mean: Tensor
    ou_velocity_mean: Tensor
    potential_mean: Tensor
    potential_variance: Tensor
    force_mean: Tensor
    force_second_moment: Tensor
    force_derivative_mean: Tensor
    pre_graph_kinetic_mean: Tensor


def rastrigin_gaussian_energy(
    prepared_position: Tensor,
    first_viscous_velocity: Tensor,
    *,
    half_step: float,
    ou_retention: float,
    ou_amplitude: float,
    position_amplitude: float,
) -> RastriginGaussianEnergy:
    """Integrate the native force and both original Gaussian innovations.

    Inputs are the actual prepared positions and B1 viscous velocities,
    after source selection, component Haar and recipient jitter. This
    conditional identity is valid for either viscous normalization.
    Values refer to all rows, before any survivor reweighting.
    """
    _arrays(prepared_position, first_viscous_velocity)
    x, u = prepared_position, first_viscous_velocity
    if x.ndim != 2 or u.shape != x.shape or min(x.shape) <= 0:
        msg = "prepared positions and velocities must have equal shape [N, d]"
        raise ValueError(msg)
    if (
        not math.isfinite(half_step)
        or half_step <= 0
        or not math.isfinite(ou_retention)
        or not 0 <= ou_retention <= 1
        or any(not math.isfinite(v) or v < 0 for v in (ou_amplitude, position_amplitude))
    ):
        msg = "invalid BAOAB Gaussian coefficient"
        raise ValueError(msg)
    t, c, q, s = half_step, ou_retention, ou_amplitude, position_amplitude
    omega, amplitude = 2 * math.pi, 20 * math.pi
    force_x = -2 * x - amplitude * torch.sin(omega * x)
    b = t * (1 + c)
    mean = x + b * (u + t * force_x)
    velocity_mean = c * (u + t * force_x)
    ou_position_variance = t**2 * q**2
    final_variance = ou_position_variance + s**2

    def potential_moments(variance: float) -> tuple[Tensor, Tensor]:
        attenuation = math.exp(-(omega**2) * variance / 2)
        cosine, sine = torch.cos(omega * mean), torch.sin(omega * mean)
        expected = mean**2 + variance + 10 - 10 * attenuation * cosine
        cosine_variance = (
            1 + math.exp(-2 * omega**2 * variance) * torch.cos(2 * omega * mean)
        ) / 2 - attenuation**2 * cosine**2
        square_cosine_covariance = attenuation * (
            -(omega**2) * variance**2 * cosine - 2 * mean * omega * variance * sine
        )
        variance_u = (
            2 * variance**2
            + 4 * mean**2 * variance
            + 100 * cosine_variance
            - 20 * square_cosine_covariance
        )
        return expected.sum(dim=1), variance_u.sum(dim=1)

    potential_mean, potential_variance = potential_moments(final_variance)
    attenuation = math.exp(-(omega**2) * ou_position_variance / 2)
    cosine, sine = torch.cos(omega * mean), torch.sin(omega * mean)
    force_mean = -2 * mean - amplitude * attenuation * sine
    mixed = attenuation * (mean * sine + omega * ou_position_variance * cosine)
    force_second = (
        4 * (mean**2 + ou_position_variance)
        + 4 * amplitude * mixed
        + amplitude**2
        / 2
        * (1 - math.exp(-2 * omega**2 * ou_position_variance) * torch.cos(2 * omega * mean))
    )
    derivative_mean = -2 - amplitude * omega * attenuation * cosine
    kinetic = (
        velocity_mean**2
        + q**2
        + 2 * t * (velocity_mean * force_mean + t * q**2 * derivative_mean)
        + t**2 * force_second
    ).sum(dim=1) / 2
    return RastriginGaussianEnergy(
        mean,
        velocity_mean,
        potential_mean,
        potential_variance,
        force_mean,
        force_second,
        derivative_mean,
        kinetic,
    )


@dataclass(frozen=True)
class RastriginCentralVelocityBound:
    """KUER.24 reference count-mode kinetic comparison with matched sources."""

    persisting_secant_second_moment: float
    jitter_secant_second_moment: float
    ou_velocity_moment: float
    graph_defect: float
    velocity_coefficient: float


def rastrigin_reference_central_velocity_bound() -> RastriginCentralVelocityBound:
    """Evaluate the unchanged d=3 reference, |source_r| <= .01, in count mode.

    This includes the complete Gaussian tail. It compares one raw kinetic
    step, and does not assert contraction of a conditioned return kernel.
    """
    d, epsilon, sigma, t, a = 3, 0.01, 0.1, 0.02, 0.006
    c = math.exp(-0.04)
    q2 = -math.expm1(-0.08) / 2
    b = t * (1 + c)
    eta, velocity_shift = b * t, b * 4
    linear, periodic = c - 2 * eta, 40 * math.pi**2 * eta
    sinc_min = math.sin(2 * math.pi * velocity_shift) / (2 * math.pi * velocity_shift)
    ou_attenuation = math.exp(-2 * math.pi**2 * t**2 * q2)
    ou_second = math.exp(-8 * math.pi**2 * t**2 * q2)

    def envelope(r: float) -> float:
        g = (1 - 2 * eta) * r - 20 * math.pi * eta * math.sin(2 * math.pi * r)
        cosine = math.cos(2 * math.pi * min(g + velocity_shift, 0.5))
        return max(
            linear**2
            - 2 * linear * periodic * sinc * ou_attenuation * cosine
            + periodic**2 * sinc**2 / 2 * (1 - ou_second + 2 * ou_second * cosine**2)
            for sinc in (sinc_min, 1)
        )

    def radial_mass(r: float) -> float:
        return float(ndtr((r - epsilon) / sigma) - ndtr((-r - epsilon) / sigma))

    tail = float(ndtr(-(0.5 - epsilon) / sigma) + ndtr(-(0.5 + epsilon) / sigma))
    jitter_bound = (
        math.fsum(
            envelope(j / 100) * (radial_mass(j / 100) - radial_mass((j - 1) / 100))
            for j in range(1, 51)
        )
        + (linear + periodic) ** 2 * tail
    )
    persisting = envelope(epsilon)
    x2 = math.sqrt(d * (epsilon**2 + sigma**2))
    u2 = 4 + t * (2 * x2 + 20 * math.pi * math.sqrt(d))
    w2 = math.sqrt(c**2 * u2**2 + d * q2)
    defect = 4 * a * math.exp(-0.5) * b * (1 + a) * w2
    coefficient = (
        math.sqrt(max(persisting, jitter_bound)) + a * (linear + periodic) + a * c + defect
    )
    return RastriginCentralVelocityBound(persisting, jitter_bound, w2, defect, coefficient)


@dataclass(frozen=True)
class RastriginRegionalProfile:
    """Native force, core curvature and actual one-dimensional barriers."""

    integer_centers: tuple[int, ...]
    stable_roots: tuple[float, ...]
    barriers: tuple[float, ...]
    core_curvature_lower: float
    global_curvature_upper: float


@dataclass(frozen=True)
class RastriginJitterProfile:
    """Exact native periodic-force moments with unrestricted Gaussian jitter."""

    mean: Tensor
    variance: Tensor


@dataclass(frozen=True)
class RastriginRegionalBound:
    """KUR.1 regional multipliers; no population parameter is required."""

    mean_map_squared: float
    accepted_coordinate_variance: float
    gate_refresh_bound: float
    positional_coefficient: float
    conservative_floor: float


def rastrigin_regional_bound(
    *,
    dimension: int,
    timestep: float,
    friction: float,
    clone_jitter: float,
    ou_amplitude: float,
    position_amplitude: float,
    velocity_cap: float,
    restitution: float,
    viscosity: float,
    core_radius: float = 1 / 16,
    largest_well: int = 2,
) -> RastriginRegionalBound:
    """Evaluate the unchanged native force on the declared root cores.

    Both positive dense first-kick matrices obey the same bound. Sources
    must satisfy the stated core condition; their unrestricted output noise
    and crossings remain in the phase ledger. This floor does not establish
    long residence or global inter-phase mixing.
    """
    positive = (timestep, velocity_cap, core_radius)
    nonnegative = (friction, clone_jitter, ou_amplitude, position_amplitude, viscosity)
    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension <= 0
        or any(not math.isfinite(v) or v <= 0 for v in positive)
        or any(not math.isfinite(v) or v < 0 for v in nonnegative)
    ):
        msg = "invalid dimension or physical coefficient"
        raise ValueError(msg)
    if (
        not isinstance(largest_well, int)
        or isinstance(largest_well, bool)
        or not 0 <= largest_well <= 21
        or not math.isfinite(restitution)
        or not 0 <= timestep * viscosity / 2 <= 1
    ):
        msg = "invalid root range, restitution or nonconvex first viscous kick"
        raise ValueError(msg)
    omega = 2 * math.pi
    m = 2 + 20 * math.sqrt(2) * math.pi**2
    radius = core_radius + 2 * largest_well / m
    if radius >= 1 / 8:
        msg = "declared root core is outside the proved regional curvature interval"
        raise ValueError(msg)
    t = timestep / 2
    c = math.exp(-friction * timestep)
    b = t * (1 + c)
    eta = t * b
    ell = 1 - 2 * eta
    amplitude = 20 * math.pi * eta
    a = math.exp(-(omega**2) * clone_jitter**2 / 2)
    cos_min = math.cos(omega * radius)
    rho = max(abs(ell - amplitude * omega * a), abs(ell - amplitude * omega * a * cos_min))
    if not 0 < rho < 1 or ell < 0:
        msg = "the displayed regional attraction criterion does not pass"
        raise ValueError(msg)
    kj = (
        ell**2 * clone_jitter**2
        - 2 * ell * amplitude * omega * clone_jitter**2 * a * cos_min
        + amplitude**2
        / 2
        * (1 - math.exp(-2 * omega**2 * clone_jitter**2) * math.cos(2 * omega * radius))
    )
    refresh = amplitude * (1 - a) * math.sin(omega * radius)
    coefficient = (1 + rho**2) / 2
    epsilon = math.sqrt(coefficient / rho**2) - 1
    vc = (1 + 2 * abs(restitution)) * velocity_cap
    floor = (
        (1 + epsilon) * ((1 + 1 / epsilon) * dimension * refresh**2 + dimension * kj)
        + (1 + 1 / epsilon) * b**2 * vc**2
        + dimension * (t**2 * ou_amplitude**2 + position_amplitude**2)
    )
    return RastriginRegionalBound(rho**2, kj, refresh, coefficient, floor)


def rastrigin_jitter_profile(
    sources: Tensor, *, drift_coefficient: float, clone_jitter: float
) -> RastriginJitterProfile:
    """Integrate g(source+sigma*Z), g(x)=x+eta*F_Rastrigin(x).

    Variance is per coordinate; summing it gives the row variance trace.
    These moments include every Gaussian excursion, including other wells.
    """
    _arrays(sources)
    if (
        not math.isfinite(drift_coefficient)
        or drift_coefficient < 0
        or not math.isfinite(clone_jitter)
        or clone_jitter < 0
    ):
        msg = "finite nonnegative drift coefficient and jitter required"
        raise ValueError(msg)
    omega = 2 * math.pi
    ell = 1 - 2 * drift_coefficient
    amplitude = 20 * math.pi * drift_coefficient
    a = math.exp(-(omega**2) * clone_jitter**2 / 2)
    b = math.exp(-2 * omega**2 * clone_jitter**2)
    sine = torch.sin(omega * sources)
    mean = ell * sources - amplitude * a * sine
    variance = (
        ell**2 * clone_jitter**2
        - 2 * ell * amplitude * omega * clone_jitter**2 * a * torch.cos(omega * sources)
        + amplitude**2 * ((1 - b * torch.cos(2 * omega * sources)) / 2 - a**2 * sine.square())
    )
    return RastriginJitterProfile(mean, variance)


def count_viscous_ou_cross(
    prepared_positions: Tensor,
    first_kick_velocity: Tensor,
    *,
    half_step: float,
    drift_duration: float,
    ou_coefficient: float,
    ou_amplitude: float,
    viscosity: float,
    bandwidth: float,
) -> Tensor:
    """Exact E <xi, F_visc(Y,c*u+q*xi)>_N at the second count kick.

    Y=X+drift_duration*u+half_step*q*xi. Pair integration retains the
    position/velocity noise correlation in the actual dense B2 force.
    """
    _arrays(prepared_positions, first_kick_velocity)
    pars = (half_step, drift_duration, ou_coefficient, ou_amplitude, viscosity, bandwidth)
    valid_parameters = (
        all(math.isfinite(v) and v >= 0 for v in pars) and bandwidth > 0 and ou_coefficient <= 1
    )
    if (
        prepared_positions.ndim != 2
        or not prepared_positions.numel()
        or first_kick_velocity.shape != prepared_positions.shape
        or not valid_parameters
    ):
        msg = "matching [N,d] arrays and valid physical coefficients required"
        raise ValueError(msg)
    n, d = prepared_positions.shape
    means = prepared_positions + drift_duration * first_kick_velocity
    delta = means[:, None] - means[None]
    delta_u = first_kick_velocity[:, None] - first_kick_velocity[None]
    variance = bandwidth**2 + 2 * (half_step * ou_amplitude) ** 2
    radius2 = delta.square().sum(2)
    weight = (bandwidth**2 / variance) ** (d / 2) * torch.exp(-radius2 / (2 * variance))
    integrand = weight * (
        -2 * ou_coefficient * half_step * ou_amplitude / variance * (delta * delta_u).sum(2)
        + ou_amplitude
        * (
            2 * d * bandwidth**2 / variance
            + 4 * (half_step * ou_amplitude) ** 2 * radius2 / variance**2
        )
    )
    integrand.fill_diagonal_(0)
    return -viscosity * integrand.sum() / (2 * n**2)


def rastrigin_regional_profile(first_well: int, last_well: int) -> RastriginRegionalProfile:
    """Evaluate roots inside proved integer/half-integer brackets.

    Native U=sum[x^2+10(1-cos(2*pi*x))]. The brackets used here
    certify wells -21,...,21; the declared domain determines which to retain.
    """
    integer_indices = (
        isinstance(first_well, int)
        and isinstance(last_well, int)
        and not isinstance(first_well, bool)
        and not isinstance(last_well, bool)
    )
    if not integer_indices or first_well > last_well or first_well < -21 or last_well > 21:
        msg = "integer well indices must lie in the proved bracket range [-21,21]"
        raise ValueError(msg)

    def gradient(x: float) -> float:
        return 2 * x + 20 * math.pi * math.sin(2 * math.pi * x)

    wells = tuple(range(first_well, last_well + 1))
    roots = tuple(brentq(gradient, k - 0.125, k + 0.125) for k in wells)
    barriers = tuple(brentq(gradient, k + 0.375, k + 0.625) for k in wells[:-1])
    return RastriginRegionalProfile(
        wells, roots, barriers, 2 + 20 * math.sqrt(2) * math.pi**2, 2 + 40 * math.pi**2
    )


@dataclass(frozen=True)
class GaussianPhaseMoments:
    """Unnormalized box moments, with an explicit exterior final column."""

    probability: Tensor
    first_moment: Tensor
    centered_second_moment: Tensor


def gaussian_phase_moments(
    means: Tensor, lower: Tensor, upper: Tensor, centers: Tensor, *, standard_deviation: float
) -> GaussianPhaseMoments:
    """Integrate independent Gaussian coordinates through disjoint phase boxes.

    Inputs are means [N,d] and box/center arrays [K,d]. Returns [N,K+1]
    probabilities and second moments, and [N,K+1,d] first moments. The final
    phase is the entire exterior, centered at zero, including all noise tails.
    At zero noise boxes are half-open except at their outermost upper faces.
    """
    _arrays(means, lower, upper, centers)
    if means.ndim != 2 or lower.ndim != 2 or not means.numel() or not lower.numel():
        msg = "nonempty [N,d] means and [K,d] boxes required"
        raise ValueError(msg)
    matching_boxes = lower.shape == upper.shape == centers.shape
    if (
        not matching_boxes
        or means.shape[1] != lower.shape[1]
        or (lower >= upper).any()
        or not math.isfinite(standard_deviation)
        or standard_deviation < 0
    ):
        msg = "nonempty [N,d] means, valid [K,d] boxes/centers and nonnegative sigma required"
        raise ValueError(msg)
    overlap = torch.minimum(upper[:, None], upper[None]) > torch.maximum(
        lower[:, None], lower[None]
    )
    overlap = overlap.all(2)
    overlap.fill_diagonal_(False)
    if overlap.any():
        msg = "phase boxes must have disjoint interiors"
        raise ValueError(msg)
    m = means[:, None, :]
    sigma = standard_deviation
    if sigma == 0:
        inside = (m >= lower) & ((m < upper) | ((m == upper) & (upper == upper.max(0).values)))
        probability = inside.all(2).to(means.dtype)
        first = probability[:, :, None] * m
        second = probability * (m - centers).square().sum(2)
    else:
        a, b = (lower - m) / sigma, (upper - m) / sigma
        interval = torch.where(
            a > 0,
            torch.special.ndtr(-a) - torch.special.ndtr(-b),
            torch.special.ndtr(b) - torch.special.ndtr(a),
        ).clamp_min(0)
        probability = interval.prod(2)
        phi_a = torch.exp(-a.square() / 2) / math.sqrt(2 * math.pi)
        phi_b = torch.exp(-b.square() / 2) / math.sqrt(2 * math.pi)
        # Product of other coordinate masses, without division by a tiny mass.
        others = torch.stack(
            [
                torch.cat((interval[:, :, :j], interval[:, :, j + 1 :]), 2).prod(2)
                for j in range(means.shape[1])
            ],
            2,
        )
        first = (m * interval + sigma * (phi_a - phi_b)) * others
        relative = m - centers
        second = (
            (
                (relative.square() + sigma**2) * interval
                + 2 * sigma * relative * (phi_a - phi_b)
                + sigma**2
                * (
                    torch.where(torch.isfinite(a), a, 0) * phi_a
                    - torch.where(torch.isfinite(b), b, 0) * phi_b
                )
            )
            * others
        ).sum(2)
    exterior_probability = (1 - probability.sum(1)).clamp_min(0)
    exterior_first = means - first.sum(1)
    raw_inside_second = (
        second + 2 * (centers * first).sum(2) - centers.square().sum(1) * probability
    )
    exterior_second = means.square().sum(1) + means.shape[1] * sigma**2 - raw_inside_second.sum(1)
    return GaussianPhaseMoments(
        torch.cat((probability, exterior_probability[:, None]), 1),
        torch.cat((first, exterior_first[:, None]), 1),
        torch.cat((second, exterior_second[:, None]), 1),
    )


@dataclass(frozen=True)
class ClonePhaseFlux:
    """Conditional full-source flux and fixed-center error, with exterior."""

    recipient_to_source: Tensor
    source_to_landing: Tensor
    output_fraction: Tensor
    centered_second_moment: Tensor
    row_probability: Tensor


def cloning_phase_flux(
    positions: Tensor,
    tokens: Tensor,
    lower: Tensor,
    upper: Tensor,
    centers: Tensor,
    *,
    clone_jitter: float,
) -> ClonePhaseFlux:
    """Integrate actual [N,2,N] tokens after freezing all fitness measurements.

    The first matrix records recipient-to-donor transport; the second records
    donor-to-jittered-phase transport. Every matrix is normalized by output N.
    Dead recipients and outside landings are retained explicitly. The moment
    is an expected error about declared centers, not a ratio of expected
    phase counts mislabeled as expected empirical within-phase variance.
    """
    _arrays(positions, tokens)
    if positions.ndim != 2 or not positions.numel():
        msg = "nonempty [N,d] positions required"
        raise ValueError(msg)
    n = positions.shape[0]
    if (
        tokens.shape != (n, 2, n)
        or (tokens < 0).any()
        or not torch.allclose(
            tokens.sum((1, 2)), torch.ones(n, dtype=tokens.dtype, device=tokens.device)
        )
    ):
        msg = "positions [N,d] and nonnegative stochastic tokens [N,2,N] required"
        raise ValueError(msg)
    persisted = gaussian_phase_moments(positions, lower, upper, centers, standard_deviation=0)
    accepted = gaussian_phase_moments(
        positions, lower, upper, centers, standard_deviation=clone_jitter
    )
    source_law = tokens.sum(1)
    recipient_source = persisted.probability.T @ source_law @ persisted.probability / n
    incoming_persist = tokens[:, 0].sum(0)
    incoming_accept = tokens[:, 1].sum(0)
    landing_flux = (
        persisted.probability.T
        @ (
            incoming_persist[:, None] * persisted.probability
            + incoming_accept[:, None] * accepted.probability
        )
        / n
    )
    output_fraction = landing_flux.sum(0)
    center_error = (
        incoming_persist[:, None] * persisted.centered_second_moment
        + incoming_accept[:, None] * accepted.centered_second_moment
    ).sum(0) / n
    row_probability = tokens[:, 0] @ persisted.probability + tokens[:, 1] @ accepted.probability
    return ClonePhaseFlux(
        recipient_source, landing_flux, output_fraction, center_error, row_probability
    )


def baoab_landing_phase_moments(
    positions: Tensor,
    velocities: Tensor,
    external_force: Tensor,
    lower: Tensor,
    upper: Tensor,
    centers: Tensor,
    *,
    timestep: float,
    viscosity: float,
    bandwidth: float,
    row_normalized: bool,
    ou_coefficient: float,
    ou_amplitude: float,
    position_amplitude: float,
) -> GaussianPhaseMoments:
    """Exact landing-phase integrals conditional on the entire preparation.

    The external force is its actual first evaluation, including the configured
    landscape. Both noise stages are integrated without a cutoff. B2 and the
    cap act after this landing position; joint position/velocity phase classes
    must retain their full innovation integral instead of using these marginals.
    """
    _arrays(positions, velocities, external_force)
    if velocities.shape != positions.shape or external_force.shape != positions.shape:
        msg = "prepared positions, retained collision velocities and force must all be [N,d]"
        raise ValueError(msg)
    parameters = (timestep, viscosity, bandwidth, ou_coefficient, ou_amplitude, position_amplitude)
    if (
        any(not math.isfinite(v) for v in parameters)
        or timestep <= 0
        or bandwidth <= 0
        or min(parameters[1], *parameters[3:]) < 0
        or ou_coefficient > 1
    ):
        msg = "invalid physical step, amplitudes, OU coefficient or viscosity"
        raise ValueError(msg)
    t = timestep / 2
    first_acceleration = external_force + gaussian_viscous_force(
        positions,
        velocities,
        viscosity=viscosity,
        bandwidth=bandwidth,
        row_normalized=row_normalized,
    )
    mean = positions + t * (1 + ou_coefficient) * (velocities + t * first_acceleration)
    sigma = math.hypot(t * ou_amplitude, position_amplitude)
    return gaussian_phase_moments(mean, lower, upper, centers, standard_deviation=sigma)


def conditional_phase_count_covariance(probability: Tensor) -> Tensor:
    """Exact normalized-count covariance for independent conditional rows.

    A random preparation adds Cov(E[counts | preparation]) to this matrix;
    that retained term must not be removed by assuming independent swarms.
    """
    _arrays(probability)
    if (
        probability.ndim != 2
        or not probability.numel()
        or (probability < 0).any()
        or not torch.allclose(probability.sum(1), torch.ones_like(probability[:, 0]))
    ):
        msg = "probability must be nonnegative row-stochastic [N,K]"
        raise ValueError(msg)
    n = probability.shape[0]
    return (torch.diag(probability.sum(0)) - probability.T @ probability) / n**2


@dataclass(frozen=True)
class SurvivingPhaseCounts:
    """Moments of N-normalized phase counts after removing all-dead output."""

    mean: Tensor
    covariance: Tensor
    survival_probability: Tensor


def surviving_phase_count_moments(probability: Tensor) -> SurvivingPhaseCounts:
    """Condition independent prepared rows on a nonempty terminal alive set.

    Alive phase boxes must partition the actual terminal domain D. The last
    column is terminal death, rather than an arbitrary exterior of some cores.
    Counts are divided by N, not by the random number of survivors. Averaging
    random preparations requires their survival tilt and between-preparation
    covariance, as stated in KLQ.5d.
    """
    covariance = conditional_phase_count_covariance(probability)
    mean = probability.mean(0)
    extinction = probability[:, -1].prod()
    survival = 1 - extinction
    if survival <= 0:
        msg = "survivor-conditioned moments require positive survival probability"
        raise ValueError(msg)
    dead_vector = torch.zeros_like(mean)
    dead_vector[-1] = 1
    surviving_mean = (mean - extinction * dead_vector) / survival
    second = (
        covariance + torch.outer(mean, mean) - extinction * torch.outer(dead_vector, dead_vector)
    ) / survival
    return SurvivingPhaseCounts(
        surviving_mean, second - torch.outer(surviving_mean, surviving_mean), survival
    )


@dataclass(frozen=True)
class PhaseVariance:
    """Exact within/between decomposition of an actual alive empirical array."""

    weights: Tensor
    means: Tensor
    within: Tensor
    between: Tensor
    total: Tensor


def alive_phase_variance(positions: Tensor, labels: Tensor, alive: Tensor) -> PhaseVariance:
    """Use the current alive normalization, retaining between-phase variance."""
    _arrays(positions)
    if positions.ndim != 2 or not positions.numel():
        msg = "nonempty positions [N,d] required"
        raise ValueError(msg)
    valid_labels = labels.shape == (positions.shape[0],) and labels.dtype == torch.int64
    valid_alive = alive.shape == labels.shape and alive.dtype == torch.bool
    if not valid_labels or not valid_alive or not alive.any() or (labels[alive] < 0).any():
        msg = "positions [N,d], nonnegative alive int64 labels and nonempty bool alive required"
        raise ValueError(msg)
    x, phase = positions[alive], labels[alive]
    count = torch.bincount(phase)
    sums = torch.zeros(count.numel(), x.shape[1], dtype=x.dtype, device=x.device)
    sums.index_add_(0, phase, x)
    means = sums / count.clamp_min(1)[:, None]
    weights = count.to(x.dtype) / x.shape[0]
    within = (x - means[phase]).square().sum(1).mean()
    between = (weights[:, None] * (means - x.mean(0)).square()).sum()
    total = (x - x.mean(0)).square().sum(1).mean()
    return PhaseVariance(weights, means, within, between, total)
