//! Euclidean reference-kernel boundary moduli with explicit metric transfer.
//!
//! These diagnostics evaluate finite sets and analytic convolution formulas.
//! They do not certify perimeter or inverse-distance hypotheses for arbitrary
//! domains, nor identify the heat parameter with a native kinetic time step.
use crate::convergence_framework::BoundCheck;
use algorithmic_gas::{
    GasError, ObservationBatch, Result, TensorBatch,
    geometry::{AlgorithmicDistance, Distance},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, f64::consts::PI};

const BALL: &str = "lem-boundary-uniform-ball";
const HEAT: &str = "lem-boundary-heat-kernel";
const INPUT_SCOPE: &str = "Euclidean R^d reference kernels, sigma>0, supplied finite Per(E), and supplied physical inverse-distance constant H; rowwise metric transfer uses a minimizing empirical matching and d_N squared normalized by N";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmoothingInputs {
    pub dimensions: usize,
    pub sigma: f64,
    pub perimeter: f64,
    /// Hypothesis |x-y| <= H d_alg(x,y) on the compared source region.
    pub inverse_distance_factor: f64,
    pub walkers: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmoothingConstants {
    pub inputs: SmoothingInputs,
    /// Only zero or positive finite representable values are stored here.
    pub values: BTreeMap<String, f64>,
    /// Positive mathematical constants retain their logs even if exp overflows
    /// or underflows. A zero perimeter branch has no finite logarithm.
    pub natural_logs: BTreeMap<String, f64>,
    pub formulas: BTreeMap<String, String>,
    pub scope: String,
}

fn configuration(message: &str) -> GasError {
    GasError::Configuration(message.into())
}

fn validate_inputs(input: &SmoothingInputs) -> Result<()> {
    if !(1..=4096).contains(&input.dimensions)
        || input.walkers == 0
        || !input.sigma.is_finite()
        || input.sigma <= 0.
        || !input.perimeter.is_finite()
        || input.perimeter < 0.
        || !input.inverse_distance_factor.is_finite()
        || input.inverse_distance_factor <= 0.
    {
        return Err(configuration("invalid Euclidean smoothing inputs"));
    }
    Ok(())
}

/// Log volume of the unit Euclidean ball, using omega_d=(2pi/d)omega_(d-2).
pub fn log_unit_ball_volume(dimensions: usize) -> Result<f64> {
    if dimensions > 4096 {
        return Err(configuration("unit-ball dimension exceeds 4096"));
    }
    let even = dimensions.is_multiple_of(2);
    let mut log_volume = if even { 0. } else { 2_f64.ln() };
    for d in (if even { 2 } else { 3 }..=dimensions).step_by(2) {
        log_volume += (2. * PI).ln() - (d as f64).ln();
    }
    Ok(log_volume)
}

fn store_positive(output: &mut SmoothingConstants, name: &str, logarithm: f64, formula: &str) {
    output.natural_logs.insert(name.into(), logarithm);
    output.formulas.insert(name.into(), formula.into());
    let value = logarithm.exp();
    if value.is_finite() && value > 0. {
        output.values.insert(name.into(), value);
    }
}

/// Computes both physical branches and the explicit H sqrt(N) transfer.
pub fn smoothing_constants(input: &SmoothingInputs) -> Result<SmoothingConstants> {
    validate_inputs(input)?;
    let d = input.dimensions as f64;
    let log_sigma = input.sigma.ln();
    let log_volume = log_unit_ball_volume(input.dimensions)?;
    let log_ball_universal = d.ln() - 2_f64.ln() - log_sigma;
    let log_heat_universal = -2_f64.ln() - 0.5 * PI.ln() - log_sigma;
    let log_heat_density = -0.5 * d * ((4. * PI).ln() + 2. * log_sigma);
    let log_inverse = input.inverse_distance_factor.ln();
    let log_matching_factor = log_inverse + 0.5 * (input.walkers as f64).ln();
    let mut output = SmoothingConstants {
        inputs: input.clone(),
        values: BTreeMap::from([
            ("perimeter".into(), input.perimeter),
            ("H".into(), input.inverse_distance_factor),
            ("sqrt_N".into(), (input.walkers as f64).sqrt()),
        ]),
        natural_logs: BTreeMap::new(),
        formulas: BTreeMap::new(),
        scope: INPUT_SCOPE.into(),
    };
    store_positive(&mut output, "omega_d", log_volume, "pi^(d/2)/Gamma(1+d/2)");
    store_positive(
        &mut output,
        "uniform_ball_universal",
        log_ball_universal,
        "d/(2*sigma)",
    );
    store_positive(
        &mut output,
        "heat_universal",
        log_heat_universal,
        "1/(2*sqrt(pi)*sigma)",
    );
    store_positive(
        &mut output,
        "heat_density_max",
        log_heat_density,
        "(4*pi*sigma^2)^(-d/2)",
    );
    store_positive(
        &mut output,
        "row_metric_factor",
        log_matching_factor,
        "H*sqrt(N)",
    );
    if input.perimeter == 0. {
        for name in [
            "uniform_ball_perimeter",
            "heat_perimeter",
            "uniform_ball_pointwise",
            "heat_pointwise",
            "uniform_ball_matched_pair",
            "heat_matched_pair",
            "uniform_ball_empirical_mean",
            "heat_empirical_mean",
        ] {
            output.values.insert(name.into(), 0.);
        }
    } else {
        let ball_perimeter = input.perimeter.ln() - log_volume - d * log_sigma;
        let heat_perimeter = input.perimeter.ln() + log_heat_density;
        let ball_pointwise = log_ball_universal.min(ball_perimeter);
        let heat_pointwise = log_heat_universal.min(heat_perimeter);
        for (name, logarithm, formula) in [
            (
                "uniform_ball_perimeter",
                ball_perimeter,
                "Per(E)/(omega_d*sigma^d)",
            ),
            (
                "heat_perimeter",
                heat_perimeter,
                "Per(E)*(4*pi*sigma^2)^(-d/2)",
            ),
            (
                "uniform_ball_pointwise",
                ball_pointwise,
                "min(universal,perimeter)",
            ),
            ("heat_pointwise", heat_pointwise, "min(universal,perimeter)"),
            (
                "uniform_ball_matched_pair",
                ball_pointwise + log_matching_factor,
                "H*sqrt(N)*L_ball",
            ),
            (
                "heat_matched_pair",
                heat_pointwise + log_matching_factor,
                "H*sqrt(N)*L_heat",
            ),
            (
                "uniform_ball_empirical_mean",
                ball_pointwise + log_inverse,
                "H*L_ball",
            ),
            (
                "heat_empirical_mean",
                heat_pointwise + log_inverse,
                "H*L_heat",
            ),
        ] {
            store_positive(&mut output, name, logarithm, formula);
        }
    }
    Ok(output)
}

// Deterministic adaptive Simpson evaluation. Analytic formulas and exact
// derivatives are kept distinct from numerical evaluations in every scope.
fn integrate(f: &impl Fn(f64) -> f64, lower: f64, upper: f64) -> f64 {
    fn refine(
        f: &impl Fn(f64) -> f64,
        interval: [f64; 2],
        values: [f64; 3],
        whole: f64,
        tolerance: f64,
        depth: usize,
    ) -> f64 {
        let [a, b] = interval;
        let [fa, fm, fb] = values;
        let m = 0.5 * (a + b);
        let fl = f(0.5 * (a + m));
        let fr = f(0.5 * (m + b));
        let left = (m - a) * (fa + 4. * fl + fm) / 6.;
        let right = (b - m) * (fm + 4. * fr + fb) / 6.;
        let difference = left + right - whole;
        if depth == 0 || difference.abs() <= 15. * tolerance {
            left + right + difference / 15.
        } else {
            refine(f, [a, m], [fa, fl, fm], left, tolerance / 2., depth - 1)
                + refine(f, [m, b], [fm, fr, fb], right, tolerance / 2., depth - 1)
        }
    }
    if upper <= lower {
        return 0.;
    }
    let values = [f(lower), f(0.5 * (lower + upper)), f(upper)];
    let whole = (upper - lower) * (values[0] + 4. * values[1] + values[2]) / 6.;
    refine(f, [lower, upper], values, whole, 2e-14, 24)
}

fn positive_gaussian_integral(lower: f64, upper: f64) -> f64 {
    if lower >= 28. || upper <= lower {
        return 0.;
    }
    let width = upper.min(28.) - lower;
    let pieces = (width / 0.5).ceil().max(1.) as usize;
    let scaled = |u: f64| (-u * (2. * lower + u)).exp();
    let integral = (0..pieces)
        .map(|i| {
            integrate(
                &scaled,
                width * i as f64 / pieces as f64,
                width * (i + 1) as f64 / pieces as f64,
            )
        })
        .sum::<f64>();
    (-lower * lower).exp() * integral
}

fn gaussian_integral(lower: f64, upper: f64) -> f64 {
    if upper <= 0. {
        positive_gaussian_integral(-upper, -lower)
    } else if lower >= 0. {
        positive_gaussian_integral(lower, upper)
    } else {
        positive_gaussian_integral(0., -lower) + positive_gaussian_integral(0., upper)
    }
}

fn standardized(bound: f64, center: f64, sigma: f64) -> f64 {
    let difference = bound - center;
    if difference.is_finite() {
        0.5 * (difference / sigma)
    } else {
        0.5 * (bound / sigma - center / sigma)
    }
}

fn validate_interval(lower: f64, upper: f64, center: f64, sigma: f64) -> Result<()> {
    if !lower.is_finite()
        || !upper.is_finite()
        || upper <= lower
        || !center.is_finite()
        || !sigma.is_finite()
        || sigma <= 0.
    {
        return Err(configuration(
            "invalid smoothing interval or heat/ball scale",
        ));
    }
    Ok(())
}

/// Analytic interval probability 1-[erf((b-x)/(2sigma))-erf((a-x)/(2sigma))]/2,
/// evaluated through a stable deterministic Gaussian integral. The heat law
/// is N(x,2 sigma^2), not N(x,sigma^2).
pub fn heat_interval_death_probability(
    lower: f64,
    upper: f64,
    center: f64,
    sigma: f64,
) -> Result<f64> {
    Ok(1. - heat_interval_probability(lower, upper, center, sigma)?)
}

fn heat_interval_probability(lower: f64, upper: f64, center: f64, sigma: f64) -> Result<f64> {
    validate_interval(lower, upper, center, sigma)?;
    let inside = gaussian_integral(
        standardized(lower, center, sigma),
        standardized(upper, center, sigma),
    ) / PI.sqrt();
    Ok(inside.clamp(0., 1.))
}

/// Exact one-dimensional uniform-ball overlap law.
pub fn ball_interval_death_probability(
    lower: f64,
    upper: f64,
    center: f64,
    sigma: f64,
) -> Result<f64> {
    validate_interval(lower, upper, center, sigma)?;
    let a = ((lower - center) / sigma).clamp(-1., 1.);
    let b = ((upper - center) / sigma).clamp(-1., 1.);
    Ok((1. - (b - a) / 2.).clamp(0., 1.))
}

fn heat_density(offset: f64, sigma: f64) -> f64 {
    let z = offset / sigma;
    (-0.25 * z * z - 2_f64.ln() - 0.5 * PI.ln() - sigma.ln()).exp()
}

fn ball_marginal_cdf(d: usize, coordinate: f64) -> Result<f64> {
    if coordinate <= -1. {
        return Ok(0.);
    }
    if coordinate >= 1. {
        return Ok(1.);
    }
    let u = coordinate.abs();
    let deviation = match d {
        1 => 0.5 * u,
        2 => (u.asin() + u * (1. - u * u).sqrt()) / PI,
        3 => 0.75 * u - 0.25 * u.powi(3),
        _ => {
            let ratio = (log_unit_ball_volume(d - 1)? - log_unit_ball_volume(d)?).exp();
            ratio * integrate(&|t| (1. - t * t).powf((d - 1) as f64 / 2.), 0., u)
        }
    };
    Ok(0.5 + coordinate.signum() * deviation)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmoothingFixture {
    pub name: String,
    pub box_half_width: f64,
    pub constants: SmoothingConstants,
    pub observations: BTreeMap<String, f64>,
    pub scope: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntervalSwarmReport {
    pub walkers: usize,
    pub constants: SmoothingConstants,
    pub observations: BTreeMap<String, f64>,
    pub checks: Vec<BoundCheck>,
    pub scope: String,
}

fn modulus_check(
    id: &str,
    label: &str,
    scope: &str,
    observed: f64,
    log_constant: f64,
    displacement: f64,
) -> BoundCheck {
    if displacement == 0. {
        return BoundCheck::upper(id, &[label], scope, observed, 0.);
    }
    let log_bound = log_constant + displacement.ln();
    let bound = log_bound.exp();
    if bound.is_finite() && bound > 0. {
        BoundCheck::upper(id, &[label], scope, observed, bound)
    } else {
        let log_one_plus_bound = if log_bound > 0. {
            log_bound + (-log_bound).exp().ln_1p()
        } else {
            log_bound.exp().ln_1p()
        };
        BoundCheck::upper(
            &format!("{id}_log1p_comparison"),
            &[label],
            &format!(
                "{scope} Comparison uses ln(1+value) because the linear bound is outside floating-point representability; the positive constant retains its natural logarithm."
            ),
            observed.ln_1p(),
            log_one_plus_bound,
        )
    }
}

/// Compares unordered one-dimensional swarms. Sorting intrinsic coordinates
/// realizes optimal uniform transport for the monotone native radial squash.
/// Indices are storage addresses, and every observation is permutation invariant.
pub fn interval_swarm_diagnostic(
    left: &[f64],
    right: &[f64],
    projection_radius: f64,
    physical_bound: f64,
    sigma: f64,
) -> Result<IntervalSwarmReport> {
    if left.is_empty()
        || left.len() != right.len()
        || left.len() > 64
        || !projection_radius.is_finite()
        || projection_radius <= 0.
        || !physical_bound.is_finite()
        || physical_bound <= 0.
        || left
            .iter()
            .chain(right)
            .any(|x| !x.is_finite() || x.abs() > physical_bound)
    {
        return Err(configuration("invalid bounded interval swarm comparison"));
    }
    let inverse = (1. + physical_bound / projection_radius).powi(2);
    let constants = smoothing_constants(&SmoothingInputs {
        dimensions: 1,
        sigma,
        perimeter: 2.,
        inverse_distance_factor: inverse,
        walkers: left.len(),
    })?;
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort_by(f64::total_cmp);
    right.sort_by(f64::total_cmp);
    let observations = |x: Vec<f64>| -> Result<ObservationBatch<f64>> {
        let mut o = ObservationBatch::positions(TensorBatch::vectors(x.len(), 1, x.clone())?);
        o.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(x.len(), 1, vec![0.; x.len()])?,
        );
        Ok(o)
    };
    let a = observations(left.clone())?;
    let b = observations(right.clone())?;
    let distance = Distance::SquashedPhaseSpace {
        positions: "positions".into(),
        velocities: "velocities".into(),
        position_radius: projection_radius,
        velocity_radius: 1.,
        lambda: 0.,
    };
    let mut squared = 0.;
    let mut physical_max: f64 = 0.;
    let mut projected_max: f64 = 0.;
    for (i, (&x, &y)) in left.iter().zip(&right).enumerate() {
        let delta = distance.compare(&a, i, &b, i)?;
        squared += delta * delta;
        physical_max = physical_max.max((x - y).abs());
        projected_max = projected_max.max(delta);
    }
    let d_n = (squared / left.len() as f64).sqrt();
    let mut measured = BTreeMap::from([
        ("d_N".into(), d_n),
        ("max_physical_displacement".into(), physical_max),
        ("max_projected_displacement".into(), projected_max),
    ]);
    let scope = "All atoms alive, |x|<=B, native radial position squash radius R, H=(1+B/R)^2; sorted intrinsic positions give optimal empirical transport. Matched pair bound H sqrt(N) L d_N and empirical mean bound H L d_N concern the supplied compact physical chart.";
    let mut checks = vec![BoundCheck::upper(
        "physical_inverse_distance_transfer",
        &[BALL, HEAT],
        scope,
        physical_max,
        inverse * projected_max,
    )];
    for (name, label, kernel) in [
        (
            "uniform_ball",
            BALL,
            ball_interval_death_probability as fn(f64, f64, f64, f64) -> Result<f64>,
        ),
        (
            "heat",
            HEAT,
            heat_interval_death_probability as fn(f64, f64, f64, f64) -> Result<f64>,
        ),
    ] {
        let q_left = left
            .iter()
            .map(|&x| kernel(-0.75, 0.75, x, sigma))
            .collect::<Result<Vec<_>>>()?;
        let q_right = right
            .iter()
            .map(|&x| kernel(-0.75, 0.75, x, sigma))
            .collect::<Result<Vec<_>>>()?;
        let max_difference = q_left
            .iter()
            .zip(&q_right)
            .map(|(x, y)| (x - y).abs())
            .fold(0., f64::max);
        let mean_difference =
            ((q_left.iter().sum::<f64>() - q_right.iter().sum::<f64>()) / left.len() as f64).abs();
        measured.insert(
            format!("{name}_max_matched_probability_change"),
            max_difference,
        );
        measured.insert(
            format!("{name}_empirical_mean_probability_change"),
            mean_difference,
        );
        checks.push(modulus_check(
            &format!("{name}_matched_pair_metric_modulus"),
            label,
            scope,
            max_difference,
            constants.natural_logs[&format!("{name}_matched_pair")],
            d_n,
        ));
        checks.push(modulus_check(
            &format!("{name}_empirical_mean_metric_modulus"),
            label,
            scope,
            mean_difference,
            constants.natural_logs[&format!("{name}_empirical_mean")],
            d_n,
        ));
    }
    Ok(IntervalSwarmReport {
        walkers: left.len(),
        constants,
        observations: measured,
        checks,
        scope: scope.into(),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmoothingReport {
    pub fixtures: Vec<SmoothingFixture>,
    pub swarm_cases: Vec<IntervalSwarmReport>,
    pub checks: Vec<BoundCheck>,
    pub scope_notes: Vec<String>,
}

/// Scale families exercise both finite-perimeter branches, universal
/// directional halfspace slopes, interval/box convolution derivatives and
/// their deterministic finite differences. No hypothesis is inferred globally.
pub fn smoothing_validation_cases() -> Result<SmoothingReport> {
    let mut fixtures = Vec::new();
    let mut checks = Vec::new();
    for d in [1, 2, 3, 4, 8] {
        for sigma in [0.01_f64, 0.1, 1., 10.] {
            let halfspace_ball =
                (log_unit_ball_volume(d - 1)? - log_unit_ball_volume(d)? - sigma.ln()).exp();
            let halfspace_heat = (-2_f64.ln() - 0.5 * PI.ln() - sigma.ln()).exp();
            let universal_scope = "Exact directional halfspace derivative at the boundary for the Euclidean reference kernel; halfspaces have infinite global perimeter, so only the measurable-set universal branch is used";
            checks.push(BoundCheck::upper(
                &format!("ball_halfspace_d{d}_sigma{sigma}"),
                &[BALL],
                universal_scope,
                halfspace_ball,
                d as f64 / (2. * sigma),
            ));
            checks.push(BoundCheck::upper(
                &format!("heat_halfspace_d{d}_sigma{sigma}"),
                &[HEAT],
                universal_scope,
                halfspace_heat,
                1. / (2. * PI.sqrt() * sigma),
            ));
            for (profile, half_width) in [
                ("small_scaled", 0.01 * sigma),
                ("large_scaled", 1.25 * sigma),
                ("fixed", 1.),
            ] {
                let perimeter = 2. * d as f64 * (2. * half_width).powi((d - 1) as i32);
                let constants = smoothing_constants(&SmoothingInputs {
                    dimensions: d,
                    sigma,
                    perimeter,
                    inverse_distance_factor: 4.,
                    walkers: 16,
                })?;
                let center = half_width + 0.5 * sigma;
                let inside_axis =
                    heat_interval_probability(-half_width, half_width, center, sigma)?;
                let inside_transverse =
                    heat_interval_probability(-half_width, half_width, 0., sigma)?;
                let heat_slope = (heat_density(half_width - center, sigma)
                    - heat_density(-half_width - center, sigma))
                .abs()
                    * inside_transverse.powi((d - 1) as i32);
                let heat_inside = inside_axis * inside_transverse.powi((d - 1) as i32);
                let heat_probability = 1. - heat_inside;
                let h = 1e-4 * sigma;
                let box_inside = |x: f64| -> Result<f64> {
                    Ok(
                        heat_interval_probability(-half_width, half_width, x, sigma)?
                            * inside_transverse.powi((d - 1) as i32),
                    )
                };
                let finite_slope =
                    (box_inside(center + h)? - box_inside(center - h)?).abs() / (2. * h);
                let finite_difference_scope = "Finite-perimeter axis-aligned box E=[-a,a]^d or its complement (same perimeter). Heat N(x,2sigma² I); exact product/derivative formulas and deterministic Gaussian integral evaluation. Finite differences have a stated numerical tolerance and are diagnostics only.";
                let id = format!("d{d}_sigma{sigma}_{profile}");
                for (suffix, observed, bound) in [
                    (
                        "heat_box_universal",
                        heat_slope,
                        constants.values["heat_universal"],
                    ),
                    (
                        "heat_box_perimeter",
                        heat_slope,
                        constants.values["heat_perimeter"],
                    ),
                    (
                        "heat_box_minimum",
                        heat_slope,
                        constants.values["heat_pointwise"],
                    ),
                    (
                        "heat_box_derivative_integral_relative_error",
                        (finite_slope / heat_slope - 1.).abs(),
                        1e-6,
                    ),
                ] {
                    checks.push(BoundCheck::upper(
                        &format!("{id}_{suffix}"),
                        &[HEAT],
                        finite_difference_scope,
                        observed,
                        bound,
                    ));
                }
                let mut observed = BTreeMap::from([
                    ("heat_inside_probability".into(), heat_inside),
                    ("heat_death_probability".into(), heat_probability),
                    ("heat_exact_directional_slope".into(), heat_slope),
                    ("heat_finite_difference_slope".into(), finite_slope),
                    ("ball_halfspace_boundary_slope".into(), halfspace_ball),
                    ("heat_halfspace_boundary_slope".into(), halfspace_heat),
                ]);
                if half_width >= sigma {
                    let ball_center = half_width - 0.5 * sigma;
                    let power = (d - 1) as f64 / 2.;
                    let ball_slope = halfspace_ball * 0.75_f64.powf(power);
                    let ball_probability = 1. - ball_marginal_cdf(d, 0.5)?;
                    let finite_slope =
                        (ball_marginal_cdf(d, (half_width - (ball_center - h)) / sigma)?
                            - ball_marginal_cdf(d, (half_width - (ball_center + h)) / sigma)?)
                        .abs()
                            / (2. * h);
                    let ball_scope = "Finite-perimeter box with a>=sigma, translated on its first axis near one face; transverse faces contain every ball section and the opposite first face is outside the ball. Exact section-area derivative and 1D/2D/3D cap formulas; other cap integrals use deterministic quadrature.";
                    for (suffix, measured, bound) in [
                        (
                            "ball_box_universal",
                            ball_slope,
                            constants.values["uniform_ball_universal"],
                        ),
                        (
                            "ball_box_perimeter",
                            ball_slope,
                            constants.values["uniform_ball_perimeter"],
                        ),
                        (
                            "ball_box_minimum",
                            ball_slope,
                            constants.values["uniform_ball_pointwise"],
                        ),
                        (
                            "ball_box_derivative_integral_relative_error",
                            (finite_slope / ball_slope - 1.).abs(),
                            1e-6,
                        ),
                    ] {
                        checks.push(BoundCheck::upper(
                            &format!("{id}_{suffix}"),
                            &[BALL],
                            ball_scope,
                            measured,
                            bound,
                        ));
                    }
                    observed.insert("ball_death_probability".into(), ball_probability);
                    observed.insert("ball_exact_directional_slope".into(), ball_slope);
                    observed.insert("ball_finite_difference_slope".into(), finite_slope);
                }
                fixtures.push(SmoothingFixture {
                    name: id,
                    box_half_width: half_width,
                    constants,
                    observations: observed,
                    scope: finite_difference_scope.into(),
                });
            }
        }
    }
    let mut swarm_cases = Vec::new();
    for n in [1, 4, 16] {
        for sigma in [0.01, 0.1, 1., 10.] {
            let left: Vec<f64> = (0..n)
                .map(|i| {
                    if n == 1 {
                        0.74
                    } else {
                        -1. + 2. * i as f64 / (n - 1) as f64
                    }
                })
                .collect();
            let mut right = left.clone();
            right[n / 2] += 0.01;
            swarm_cases.push(interval_swarm_diagnostic(&left, &right, 0.25, 2., sigma)?);
        }
    }
    Ok(SmoothingReport {
        fixtures, swarm_cases, checks,
        scope_notes: vec![
            "Uniform-ball density is 1/(omega_d sigma^d); heat density is (4pi sigma²)^(-d/2) exp(-|z|²/(4sigma²)). The universal branches scale as sigma^-1. Perimeter branches use the full dimensional density normalization.".into(),
            "Finite perimeter and the physical inverse-distance inequality are supplied analytic hypotheses. A forward projection Lipschitz constant alone cannot transfer a physical event-probability bound to a projected distance.".into(),
            "Per-pair transfer along a minimizing N-atom matching is H sqrt(N) L; averaging the paired probabilities uses Cauchy-Schwarz and gives H L. Intrinsic sorting makes all reported swarm values invariant under independent storage permutations.".into(),
            "Halfspace slope formulas exercise the universal measurable-set branch only; boxes/intervals exercise finite-perimeter branches. Finite deterministic examples do not establish a theorem for arbitrary domains or source regions.".into(),
        ],
    })
}
