//! Chapter 6 interfaces with explicit source laws and absorbing conventions.
//! Supplied source-moment hypotheses are never inferred from sampled maxima.
use algorithmic_gas::{GasError, Result};
use serde::{Deserialize, Serialize};

fn invalid(message: &str) -> GasError {
    GasError::Configuration(message.into())
}
fn nonnegative(x: f64) -> bool {
    x.is_finite() && x >= 0.
}
fn log_sum(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}
fn log_one_minus_exp(log_x: f64) -> f64 {
    if log_x < -std::f64::consts::LN_2 {
        (-log_x.exp()).ln_1p()
    } else {
        (-log_x.exp_m1()).ln()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UniformBoxHazard {
    pub dimension: usize,
    pub population: usize,
    pub half_width: f64,
    pub final_position_standard_deviation: f64,
    pub log_per_row_death_lower: f64,
    pub log_native_zero_alive_hazard_lower: f64,
    pub log_chapter_fewer_two_alive_hazard_lower: f64,
    pub log_native_expected_lifetime_upper: f64,
    pub log_chapter_expected_lifetime_upper: f64,
    pub per_row_death_lower_float: f64,
    pub native_hazard_lower_float: f64,
    pub chapter_hazard_lower_float: f64,
    pub arithmetic_scope: String,
    pub hypotheses: Vec<String>,
}

/// Uniform over every pre-final-noise mean, including random selected-source,
/// collision, graph, jitter and OU preparation. Native absorption is k=0;
/// external Chapter 6 stopping is k<2. These fixed-N hazards decay with N.
///
/// For z=L/s>0, Mills gives 2 Phi(-z) >= 2 phi(z) z/(1+z²).
/// Product box landing is maximized at mean zero. If q is this one-coordinate
/// lower bound, a d-coordinate row dies with probability >=1-(1-q)^d.
pub fn uniform_final_gaussian_box_hazard(
    dimension: usize,
    population: usize,
    half_width: f64,
    sigma: f64,
) -> Result<UniformBoxHazard> {
    if dimension == 0
        || population < 2
        || !half_width.is_finite()
        || half_width <= 0.
        || !sigma.is_finite()
        || sigma <= 0.
    {
        return Err(invalid(
            "positive dimension, L, final Gaussian sigma and N>=2 required",
        ));
    }
    let z = half_width / sigma;
    if !z.is_finite() || z == 0. || !z.powi(2).is_finite() {
        return Err(invalid("box/noise ratio outside supported finite range"));
    }
    let log_coordinate =
        std::f64::consts::LN_2 - 0.5 * (2. * std::f64::consts::PI).ln() - z.powi(2) / 2. + z.ln()
            - z.powi(2).ln_1p();
    // Lower bound from the event of exactly one coordinate failure. Together
    // with the single-coordinate lower bound this avoids false numerical zero.
    let log_single = (dimension as f64).ln()
        + log_coordinate
        + (dimension - 1) as f64 * log_one_minus_exp(log_coordinate);
    let log_row = log_coordinate.max(log_single);
    let log_native = population as f64 * log_row;
    let log_chapter = log_sum(
        log_native,
        (population as f64).ln() + log_one_minus_exp(log_row) + (population - 1) as f64 * log_row,
    );
    Ok(UniformBoxHazard {
        dimension, population, half_width,
        final_position_standard_deviation: sigma,
        log_per_row_death_lower: log_row,
        log_native_zero_alive_hazard_lower: log_native,
        log_chapter_fewer_two_alive_hazard_lower: log_chapter,
        log_native_expected_lifetime_upper: -log_native,
        log_chapter_expected_lifetime_upper: -log_chapter,
        per_row_death_lower_float: log_row.exp(),
        native_hazard_lower_float: log_native.exp(),
        chapter_hazard_lower_float: log_chapter.exp(),
        arithmetic_scope: "Real analytic Mills bounds evaluated in binary64. A lower bound may underflow to zero; its finite log remains authoritative. No positive lower floor is inserted. Not directed interval arithmetic.".into(),
        hypotheses: vec!["independent coordinatewise final position Gaussian of standard deviation sigma".into(),
            "actual absorbing box [-L,L]^d; random preparation integrates without restricting its mean".into(),
            "all N rows are subject to the final positional draw before terminal validity; extinct chains stay absorbed".into(),
            "no claim that fixed-N whole-swarm hazard is independent of N".into()],
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeTailParameters {
    pub dimension: usize,
    pub moment_order: f64,
    pub timestep: f64,
    pub friction: f64,
    pub harmonic_curvature: f64,
    /// Certified ||F(x)+omega*x|| <= residual_per_coordinate * sqrt(d).
    pub residual_per_coordinate: f64,
    pub clone_jitter_standard_deviation: f64,
    pub ou_standard_deviation: f64,
    pub final_position_standard_deviation: f64,
    /// Bound for EVERY frozen entering collision velocity, including dead slots.
    pub frozen_velocity_norm_bound: f64,
    pub restitution: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeTailInterface {
    pub parameters: NativeTailParameters,
    pub gaussian_norm_lp_upper: f64,
    pub source_lp_multiplier: f64,
    pub additive_lp_budget: f64,
    pub selected_source_moment: f64,
    pub one_step_moment_upper: f64,
    pub source_moment_coefficient: f64,
    pub drift_additive_moment: f64,
    pub hypotheses: Vec<String>,
    pub scope: String,
}

/// Actual BAOAB positional moment with count/row first stochastic viscosity.
/// The zero-jitter source moment has probability normalization 1/N. Its source
/// law includes mandatory revival and accepted copying, before fresh jitter.
/// Graph weights may depend on jitter; only pathwise stochasticity is used.
pub fn native_selected_source_tail_interface(
    p: &NativeTailParameters,
    selected_source_moment: f64,
) -> Result<NativeTailInterface> {
    if p.dimension == 0
        || !p.moment_order.is_finite()
        || !(1. ..=128.).contains(&p.moment_order)
        || !p.timestep.is_finite()
        || p.timestep <= 0.
        || ![
            p.friction,
            p.harmonic_curvature,
            p.residual_per_coordinate,
            p.clone_jitter_standard_deviation,
            p.ou_standard_deviation,
            p.final_position_standard_deviation,
            p.frozen_velocity_norm_bound,
            selected_source_moment,
        ]
        .iter()
        .all(|x| nonnegative(*x))
        || !p.restitution.is_finite()
        || !(0. ..=1.).contains(&p.restitution)
    {
        return Err(invalid(
            "valid native tail parameters, p>=1 and normalized source moment required",
        ));
    }
    let half = p.timestep / 2.;
    let drift = half * (1. + (-p.friction * p.timestep).exp());
    let eta = half * drift;
    let multiplier = (1. - eta * p.harmonic_curvature).abs();
    // For arbitrary p>=1, interpolate downward from the next even Gaussian
    // norm moment. Exact when p is an even integer; a valid Jensen bound otherwise.
    let k = (p.moment_order / 2.).ceil() as usize;
    let gaussian = (0..k)
        .map(|j| (p.dimension as f64 + 2. * j as f64).ln())
        .sum::<f64>()
        .mul_add(1. / (2. * k as f64), 0.)
        .exp();
    let sigma = (half * p.ou_standard_deviation).hypot(p.final_position_standard_deviation);
    let additive = multiplier * p.clone_jitter_standard_deviation * gaussian
        + drift * (1. + 2. * p.restitution) * p.frozen_velocity_norm_bound
        + eta * p.residual_per_coordinate * (p.dimension as f64).sqrt()
        + sigma * gaussian;
    let moment = (multiplier * selected_source_moment.powf(1. / p.moment_order) + additive)
        .powf(p.moment_order);
    // Young gives a strictly subunit source-moment multiplier when a<1.
    let (coefficient, budget) = if multiplier == 0. {
        (0., additive.powf(p.moment_order))
    } else if p.moment_order == 1. {
        (multiplier, additive)
    } else if multiplier < 1. {
        let coefficient = (1. + multiplier.powf(p.moment_order)) / 2.;
        let epsilon =
            (coefficient / multiplier.powf(p.moment_order)).powf(1. / (p.moment_order - 1.)) - 1.;
        (
            coefficient,
            (1. + 1. / epsilon).powf(p.moment_order - 1.) * additive.powf(p.moment_order),
        )
    } else {
        (
            2_f64.powf(p.moment_order - 1.) * multiplier.powf(p.moment_order),
            2_f64.powf(p.moment_order - 1.) * additive.powf(p.moment_order),
        )
    };
    if ![gaussian, additive, moment, coefficient, budget]
        .iter()
        .all(|x| x.is_finite())
    {
        return Err(invalid(
            "native tail moment exceeds supported finite arithmetic",
        ));
    }
    Ok(NativeTailInterface {
        parameters: p.clone(), gaussian_norm_lp_upper: gaussian,
        source_lp_multiplier: multiplier, additive_lp_budget: additive,
        selected_source_moment, one_step_moment_upper: moment,
        source_moment_coefficient: coefficient, drift_additive_moment: budget,
        hypotheses: vec!["F(x)=-omega*x+r(x) with a GLOBAL residual norm bound, not a sampled maximum".into(),
            "selected zero-jitter source moment includes actual donor/acceptance/revival probabilities or a stated frozen plan".into(),
            "EVERY frozen collision input obeys its norm bound, including revived recipients".into(),
            "actual first viscous matrix is stochastic; count or row normalization with 0<=h*nu/2<=1".into(),
            "original recipient jitter and independent OU/final position Gaussians retain full support".into()],
        scope: "N-normalized one-step unconditional positional moment, or killed observable dominated by it. A global-time drift needs a separately proved selected-source moment inequality; surviving moments require each law's own survival denominator. No selected-source drift is inferred from simulations.".into(),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MomentClosure {
    pub coefficient: f64,
    pub additive: f64,
    pub closes: bool,
    pub invariant_moment_upper: Option<f64>,
    pub source_hypothesis: String,
}

/// Compose ONLY with a supplied analytic expectation bound
/// E M_source,p <= A_source*M_input,p+B_source for the very same kernel.
pub fn compose_proved_source_moment(
    interface: &NativeTailInterface,
    source_coefficient: f64,
    source_additive: f64,
    source_hypothesis: &str,
) -> Result<MomentClosure> {
    if !nonnegative(source_coefficient)
        || !nonnegative(source_additive)
        || source_hypothesis.trim().is_empty()
    {
        return Err(invalid(
            "explicit nonnegative analytic source moment hypothesis required",
        ));
    }
    let coefficient = interface.source_moment_coefficient * source_coefficient;
    let additive =
        interface.source_moment_coefficient * source_additive + interface.drift_additive_moment;
    if !coefficient.is_finite() || !additive.is_finite() {
        return Err(invalid("composed source drift arithmetic overflow"));
    }
    let closes = coefficient < 1.;
    Ok(MomentClosure {
        coefficient,
        additive,
        closes,
        invariant_moment_upper: closes.then(|| additive / (1. - coefficient)),
        source_hypothesis: source_hypothesis.into(),
    })
}

/// Exact pre-OU conditional standard-native BAOAB displacement budgets.
/// Both moments are normalized by the original population N; the variance is
/// centered over the k entering alive rows. No terminal cap bounds v1.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeDisplacementBudget {
    pub population: usize,
    pub entering_alive: usize,
    pub dimension: usize,
    pub first_kick_coefficient: f64,
    pub gaussian_coordinate_variance: f64,
    pub squared_displacement: f64,
    pub centered_squared_displacement: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn native_prepared_displacement_budget(
    population: usize,
    entering_alive: usize,
    dimension: usize,
    timestep: f64,
    friction: f64,
    ou_sigma: f64,
    position_sigma: f64,
    first_kick_total: f64,
    first_kick_variance: f64,
) -> Result<NativeDisplacementBudget> {
    if population == 0
        || entering_alive > population
        || dimension == 0
        || !timestep.is_finite()
        || timestep <= 0.
        || ![
            friction,
            ou_sigma,
            position_sigma,
            first_kick_total,
            first_kick_variance,
        ]
        .into_iter()
        .all(nonnegative)
        || first_kick_variance > first_kick_total * (1. + 32. * f64::EPSILON)
        || (entering_alive == 0 && first_kick_total != 0.)
        || (entering_alive <= 1 && first_kick_variance != 0.)
    {
        return Err(invalid(
            "invalid pre-innovation N-normalized displacement operands",
        ));
    }
    let coefficient = timestep * (1. + (-friction * timestep).exp()) / 2.;
    let variance = (timestep * ou_sigma / 2.).powi(2) + position_sigma.powi(2);
    let raw = coefficient.powi(2) * first_kick_total
        + entering_alive as f64 / population as f64 * dimension as f64 * variance;
    let centered = coefficient.powi(2) * first_kick_variance
        + entering_alive.saturating_sub(1) as f64 / population as f64 * dimension as f64 * variance;
    if ![coefficient, variance, raw, centered]
        .into_iter()
        .all(nonnegative)
    {
        return Err(invalid("displacement evaluation overflow"));
    }
    Ok(NativeDisplacementBudget {
        population,
        entering_alive,
        dimension,
        first_kick_coefficient: coefficient,
        gaussian_coordinate_variance: variance,
        squared_displacement: raw,
        centered_squared_displacement: centered,
    })
}

/// Algebra conditional on two separately proved same-kernel affine estimates:
/// P_C Vx <= r_C Vx+b_C and P_C Dh² <= a_D Vx+b_D.
/// Returns the full composed coefficient, source and its contraction gate.
pub fn compose_native_position_budget(
    theta: f64,
    cloning_rate: f64,
    cloning_source: f64,
    displacement_coefficient: f64,
    displacement_source: f64,
) -> Result<(f64, f64, bool)> {
    if !theta.is_finite()
        || theta <= 0.
        || ![
            cloning_rate,
            cloning_source,
            displacement_coefficient,
            displacement_source,
        ]
        .into_iter()
        .all(nonnegative)
    {
        return Err(invalid(
            "finite nonnegative affine operands and theta>0 required",
        ));
    }
    let rate = (1. + theta) * cloning_rate + (1. + theta.recip()) * displacement_coefficient;
    let source = (1. + theta) * cloning_source + (1. + theta.recip()) * displacement_source;
    if !rate.is_finite() || !source.is_finite() {
        return Err(invalid("position composition overflow"));
    }
    Ok((rate, source, rate < 1.))
}
