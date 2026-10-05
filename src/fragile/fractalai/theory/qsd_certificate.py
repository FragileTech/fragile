"""Logarithmic primitive-parameter bounds for the real-coordinate killed viscous gas.

The evaluator implements ``thm-cgd-primitive-eigenfunction``. It does not solve
the kernel eigenproblem or change a simulation. The configured force must have
the analytic, global derivative profile required by that theorem; the evaluator
checks the resulting scalar inequalities, not that functional hypothesis.
Floating-point evaluations are diagnostics, not interval-arithmetic enclosures.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from scipy.special import gammainc, log_ndtr


@dataclass(frozen=True)
class PrimitiveQSDCertificate:
    """Bounds in natural logarithms, preserving positive bounds below float range.

    A missing bound means this certificate did not close, rather than that the
    configured gas has no QSD. The force profile and complete canonical kernel
    hypotheses still have to be checked outside this numerical evaluator.
    """

    coercivity_margin: float
    drift_margin: float | None
    log_survival_lower_bound: float
    log_minorization_lower_bound: float | None
    log_two_step_density_upper_bound: float | None
    log_output_density_lower_bound: float | None
    log_tail_probability_upper_bound: float | None
    log_eigenfunction_min_lower_bound: float | None
    log_doob_minorization_lower_bound: float | None
    jitter_event_radius: float
    noise_event_radius: float
    output_position_radius: float | None
    precap_velocity_radius: float | None
    limitations: tuple[str, ...]

    @property
    def certified(self) -> bool:
        """Whether the evaluated scalar certificate supplies an eigenfunction bound."""
        return self.log_eigenfunction_min_lower_bound is not None

    @property
    def log_eigenfunction_ratio_upper_bound(self) -> float | None:
        """Bound on log(max e / min e), independent of eigenfunction normalization."""
        value = self.log_eigenfunction_min_lower_bound
        return -value if value is not None else None

    @property
    def log_tv_prefactor_upper_bound(self) -> float | None:
        """Logarithm of the conditioned-TV prefactor 4 / m**2."""
        value = self.log_eigenfunction_min_lower_bound
        return math.log(4) - 2 * value if value is not None else None


def _log_ball_probability(dimension: int, radius: float) -> float:
    probability = float(gammainc(dimension / 2, radius**2 / 2))
    if probability > 0:
        return math.log(probability)
    # Integrate the minimum Gaussian density over the ball if gammainc underflows.
    return (
        dimension * math.log(radius)
        - dimension / 2 * math.log(2)
        - math.lgamma(1 + dimension / 2)
        - radius**2 / 2
    )


def _log_add(left: float, right: float) -> float:
    larger, smaller = max(left, right), min(left, right)
    return larger + math.log1p(math.exp(smaller - larger))


def viscous_qsd_primitive_certificate(
    *,
    n_walkers: int,
    dimension: int,
    timestep: float,
    viscosity: float,
    bandwidth: float,
    force_lipschitz: float,
    force_at_origin: float,
    terminal_half_width: float,
    clone_jitter: float,
    restitution: float,
    velocity_cap: float,
    ou_coefficient: float,
    ou_amplitude: float,
    position_amplitude: float,
    row_normalized: bool = False,
    survival_jitter_radius: float | None = None,
    jitter_event_radius: float | None = None,
    noise_event_radius: float | None = None,
) -> PrimitiveQSDCertificate:
    """Evaluate the two-update eigenfunction and conditioned-mixing certificate.

    ``force_lipschitz`` bounds the actual global derivative of the real-analytic
    force, and ``force_at_origin`` is its norm at zero. These are computed force
    profiles, not permission to replace the configured landscape. The noises
    are the standard deviations at the OU and final-position stages. Analysis
    radii select events in the proof; no Gaussian draw is clipped.

    Both existing complete-Gaussian viscosity normalizations are supported.
    Positive clone jitter is needed for the two-update density argument when
    N > 1. Equal fitness, zero acceptance, and shared component rotations are
    covered: no lower bound on acceptance probabilities is used.
    """
    for name, value in {"n_walkers": n_walkers, "dimension": dimension}.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            msg = f"{name} must be a positive integer"
            raise ValueError(msg)
    positive = {
        "timestep": timestep,
        "bandwidth": bandwidth,
        "terminal_half_width": terminal_half_width,
        "velocity_cap": velocity_cap,
        "ou_amplitude": ou_amplitude,
        "position_amplitude": position_amplitude,
    }
    nonnegative = {
        "viscosity": viscosity,
        "force_lipschitz": force_lipschitz,
        "force_at_origin": force_at_origin,
        "clone_jitter": clone_jitter,
        "ou_coefficient": ou_coefficient,
    }
    if any(not math.isfinite(v) or v <= 0 for v in positive.values()):
        msg = "step, bandwidth, box, cap and both kinetic noise amplitudes must be positive"
        raise ValueError(msg)
    if any(not math.isfinite(v) or v < 0 for v in nonnegative.values()):
        msg = "viscosity, force profiles, clone jitter and OU coefficient must be nonnegative"
        raise ValueError(msg)
    if ou_coefficient > 1 or not math.isfinite(restitution):
        msg = "OU coefficient must be at most one and restitution must be finite"
        raise ValueError(msg)
    for value in (survival_jitter_radius, jitter_event_radius, noise_event_radius):
        if value is not None and (not math.isfinite(value) or value <= 0):
            msg = "analysis radii must be finite and positive"
            raise ValueError(msg)

    n, d = n_walkers, dimension
    m = n * d
    t = timestep / 2
    nu = viscosity if n > 1 else 0.0
    q, s, c = ou_amplitude, position_amplitude, ou_coefficient
    Vc = (1 + 2 * abs(restitution)) * velocity_cap
    RD = math.sqrt(d) * terminal_half_width
    ell = math.exp(-0.5) / bandwidth
    J0 = survival_jitter_radius or (clone_jitter if clone_jitter > 0 else 1.0)
    log_pj = _log_ball_probability(d, J0 / clone_jitter) if clone_jitter > 0 else 0.0

    def stages(jitter_radius: float) -> tuple[float, float, float]:
        B0 = RD + jitter_radius
        B1 = (1 + 2 * t * nu) * Vc + t * (force_at_origin + force_lipschitz * B0)
        return B0, B1, B0 + t * B1

    _, B1, _ = stages(J0)
    mean_coordinate = terminal_half_width + J0 + t * (1 + c) * B1
    position_sd = math.hypot(t * q, s)
    upper = float(log_ndtr((terminal_half_width - mean_coordinate) / position_sd))
    lower = float(log_ndtr((-terminal_half_width - mean_coordinate) / position_sd))
    if not math.isfinite(upper) or lower >= upper:
        msg = "survival interval exceeds floating-point log-CDF resolution"
        raise ValueError(msg)
    log_a = log_pj + d * (upper + math.log(-math.expm1(lower - upper)))
    # Each of the three union-bound terms is <= a**2 / 6 at these radii.
    standard_radius = math.sqrt(2 * d * (math.log(12 * d * n) - 2 * log_a))
    J = jitter_event_radius or (clone_jitter * standard_radius if clone_jitter > 0 else J0)
    G = noise_event_radius or standard_radius
    viscous_factor = 2 if row_normalized else 1
    kappa = 1 - t * t * force_lipschitz - viscous_factor * t * nu

    def density_lower(position_radius: float, velocity_radius: float) -> float:
        _, first_velocity, first_position = stages(J0)
        Z = (velocity_radius + t * (force_at_origin + force_lipschitz * first_position)) / kappa
        if not row_normalized:
            Z *= math.sqrt(n)
        R2 = first_position + t * Z
        base = 1 + t * t * force_lipschitz + 2 * t * nu
        log_derivative = math.log(base)
        if nu > 0 and not (row_normalized and n == 2):
            derivative_term = (8 if row_normalized else 4) * t * t * nu * ell * Z
            log_extra = math.log(derivative_term)
            if row_normalized:
                log_extra += 2 * (R2 / bandwidth) ** 2
            log_derivative = _log_add(log_derivative, log_extra)
        log_jacobian = m * (0.5 * math.log(n) + log_derivative)
        return (
            n * log_pj
            - m * math.log(2 * math.pi * q * s)
            - n * (Z + c * first_velocity) ** 2 / (2 * q * q)
            - n * (math.sqrt(d) * position_radius + first_position + t * Z) ** 2 / (2 * s * s)
            - log_jacobian
        )

    problems = []
    log_epsilon = None
    if kappa > 0:
        target_position = terminal_half_width / 2
        target_velocity = velocity_cap / 2
        inverse_cap = velocity_cap * target_velocity / (velocity_cap - target_velocity)
        log_volume = m * math.log(2 * target_position) + n * (
            d / 2 * math.log(math.pi) + d * math.log(target_velocity) - math.lgamma(1 + d / 2)
        )
        log_epsilon = min(-math.log(2), density_lower(target_position, inverse_cap) + log_volume)
    else:
        problems.append("the global B2 coercivity margin is not positive")
    if n > 1 and clone_jitter == 0:
        problems.append("the two-update density certificate requires positive clone jitter")

    B0, B1, Bx = stages(J)
    force_derivative = force_lipschitz
    if row_normalized and n > 2:
        force_derivative += 16 * nu * Vc * B0 / bandwidth**2
    elif not row_normalized:
        force_derivative += 4 * nu * Vc * ell
    drift_margin = 1 - t * t * force_derivative
    if drift_margin <= 0:
        problems.append("the first-drift derivative margin on the jitter event is not positive")
    log_tails = [math.log(2 * d * n) - G * G / (2 * d)] * 2
    if clone_jitter > 0:
        log_tails.append(math.log(2 * d * n) - (J / clone_jitter) ** 2 / (2 * d))
    log_tail = log_tails[0]
    for term in log_tails[1:]:
        log_tail = _log_add(log_tail, term)
    if log_tail > 2 * log_a - math.log(2) + 1e-10:
        problems.append("the discarded-event bound exceeds half the squared survival floor")

    log_M = log_L = log_min = log_delta = None
    Z0 = c * B1 + q * G
    R2 = Bx + t * Z0
    H = (1 + 2 * t * nu) * Z0 + t * (force_at_origin + force_lipschitz * R2)
    R = R2 + s * G
    if not problems:
        assert log_epsilon is not None
        tau = min(s, clone_jitter) if n > 1 else s
        linear_margin = 1 - viscous_factor * t * nu
        log_patterns = n * math.log(4 * n * n)
        log_M = (
            log_patterns
            - m * math.log(2 * math.pi * tau * q)
            - m * (math.log(drift_margin) + math.log(linear_margin))
        )
        log_L = density_lower(R, H)
        log_min = min(0.0, log_L + 2 * log_a - math.log(2) - log_M)
        log_delta = log_epsilon + log_min
        if not all(math.isfinite(v) for v in (log_M, log_L, log_min, log_delta)):
            msg = "derived certificate exceeds floating-point logarithmic range"
            raise ValueError(msg)
    return PrimitiveQSDCertificate(
        coercivity_margin=kappa,
        drift_margin=drift_margin,
        log_survival_lower_bound=log_a,
        log_minorization_lower_bound=log_epsilon,
        log_two_step_density_upper_bound=log_M,
        log_output_density_lower_bound=log_L,
        log_tail_probability_upper_bound=log_tail,
        log_eigenfunction_min_lower_bound=log_min,
        log_doob_minorization_lower_bound=log_delta,
        jitter_event_radius=J,
        noise_event_radius=G,
        output_position_radius=R,
        precap_velocity_radius=H,
        limitations=tuple(problems),
    )
