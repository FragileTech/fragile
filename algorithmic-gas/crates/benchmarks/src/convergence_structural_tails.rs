//! Dimension-explicit Gaussian region and weighted-tail envelopes.
//!
//! These are analytic one-particle landing/moment bounds, or consequences of a
//! supplied moment hypothesis. They do not certify a coupled discrepancy matrix,
//! a uniform QSD rate, or a global moment bound from sampled moments.
use algorithmic_gas::{GasError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

fn invalid(message: &str) -> GasError {
    GasError::Configuration(message.into())
}
fn finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.
}
fn positive(value: f64) -> bool {
    value.is_finite() && value > 0.
}
fn upper_exp(log_value: f64) -> Result<f64> {
    if log_value.is_nan() || log_value > f64::MAX.ln() {
        return Err(invalid(
            "Gaussian envelope overflows the supported f64 range",
        ));
    }
    // Every finite-radius Gaussian upper bound here is positive. Preserve the
    // log bound and avoid interpreting underflow as impossible escape.
    Ok(log_value.exp().max(f64::MIN_POSITIVE))
}
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let largest = a.max(b);
    largest + ((a - largest).exp() + (b - largest).exp()).ln()
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CoordinateInterval {
    pub lower: f64,
    pub upper: f64,
}
impl CoordinateInterval {
    fn valid(&self) -> bool {
        self.lower.is_finite() && self.upper.is_finite() && self.lower <= self.upper
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GaussianRegionBudget {
    pub dimension: usize,
    pub center_coordinate_intervals: Vec<CoordinateInterval>,
    pub center_norm_upper: f64,
    /// Square roots of certified covariance eigenvalue bounds, not sampled SDs.
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub landing_lower: f64,
    pub log_density_landing_lower: f64,
    pub escape_upper: f64,
    pub source_labels: Vec<String>,
    pub scope: String,
    pub floating_evaluation_scope: String,
}

/// Uniform over the supplied mean intervals and covariance spectral bounds.
/// The Gaussian may have correlated coordinates. The target box is an analysis
/// region, never a truncation or modification of the algorithm's Gaussian.
pub fn gaussian_region_budget(
    centers: &[CoordinateInterval],
    target: &[CoordinateInterval],
    sigma_min: f64,
    sigma_max: f64,
) -> Result<GaussianRegionBudget> {
    if centers.is_empty()
        || centers.len() != target.len()
        || centers.iter().any(|x| !x.valid())
        || target.iter().any(|x| !x.valid() || x.lower == x.upper)
        || !positive(sigma_min)
        || !positive(sigma_max)
        || sigma_min > sigma_max
    {
        return Err(invalid(
            "valid equal-dimensional boxes and positive covariance eigenvalue bounds required",
        ));
    }
    let dimension = centers.len();
    let mut log_density_landing_lower = 0.;
    let mut norm_squared = 0.;
    let mut log_union_escape = f64::NEG_INFINITY;
    let all_centers_inside = centers
        .iter()
        .zip(target)
        .all(|(m, b)| m.lower >= b.lower && m.upper <= b.upper);
    for (m, b) in centers.iter().zip(target) {
        let width = b.upper - b.lower;
        let farthest = (b.lower - m.upper).abs().max((b.upper - m.lower).abs());
        let absolute = m.lower.abs().max(m.upper.abs());
        norm_squared += absolute * absolute;
        log_density_landing_lower += width.ln()
            - ((2. * std::f64::consts::PI).sqrt() * sigma_max).ln()
            - 0.5 * (farthest / sigma_min).powi(2);
        if all_centers_inside {
            log_union_escape = log_add(
                log_union_escape,
                -0.5 * ((m.lower - b.lower) / sigma_max).powi(2),
            );
            log_union_escape = log_add(
                log_union_escape,
                -0.5 * ((b.upper - m.upper) / sigma_max).powi(2),
            );
        }
    }
    if !norm_squared.is_finite() || !log_density_landing_lower.is_finite() {
        return Err(invalid("mean intervals outside supported numerical range"));
    }
    let density_lower = log_density_landing_lower.exp().min(1.);
    let union_escape = if all_centers_inside {
        upper_exp(log_union_escape)?.min(1.)
    } else {
        1.
    };
    // Density lower bounds can underflow to zero: that remains a valid lower
    // bound. A finite target never receives Gaussian mass exactly one.
    let landing_lower = density_lower
        .max(1. - union_escape)
        .clamp(0., 1. - f64::EPSILON);
    let escape_upper = union_escape.min(1. - density_lower).max(f64::MIN_POSITIVE);
    Ok(GaussianRegionBudget {
        dimension,
        center_coordinate_intervals: centers.to_vec(),
        center_norm_upper: norm_squared.sqrt(),
        sigma_min,
        sigma_max,
        landing_lower,
        log_density_landing_lower,
        escape_upper,
        source_labels: vec!["lem-convergence-gaussian-regional-tail".into()],
        scope: "one-particle Gaussian transition, conditional on a proved mean/covariance envelope; no coupled discrepancy or QSD certificate".into(),
        floating_evaluation_scope: "exact real analytic formulas evaluated in f64; positive upper floors and log bounds preserve rare-tail information; not verified interval arithmetic".into(),
    })
}

/// log E[Y^k 1_{Y>=u}] envelope for Y~chi-square(d), integer k>=0.
/// Chernoff is optimized within the exact exponentially tilted moment formula.
fn chi_square_weighted_log_upper(d: usize, k: usize, u: f64) -> f64 {
    let log_moment: f64 = (0..k).map(|j| (d as f64 + 2. * j as f64).ln()).sum();
    let tilted_dimension = d as f64 + 2. * k as f64;
    if u <= tilted_dimension {
        log_moment
    } else if u == f64::INFINITY {
        f64::NEG_INFINITY
    } else {
        log_moment + 0.5 * (tilted_dimension - u + tilted_dimension * (u / tilted_dimension).ln())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GaussianTailBudget {
    pub dimension: usize,
    pub center_norm_upper: f64,
    pub sigma: f64,
    pub cutoff: f64,
    pub cost_order: f64,
    pub escape_upper: f64,
    pub log_escape_upper: f64,
    pub tail_cost_upper: f64,
    pub log_tail_cost_upper: f64,
    pub source_labels: Vec<String>,
    pub scope: String,
}

/// X=m+AZ, ||m||<=c, ||A||op<=sigma: upper bounds on P(||X||>R)
/// and E[||X||^r 1_{||X||>R}], with no compact-support assumption.
pub fn gaussian_tail_budget(
    dimension: usize,
    center_norm_upper: f64,
    sigma: f64,
    cutoff: f64,
    cost_order: f64,
) -> Result<GaussianTailBudget> {
    if dimension == 0
        || !finite_nonnegative(center_norm_upper)
        || !positive(sigma)
        || !positive(cutoff)
        || !finite_nonnegative(cost_order)
        || cost_order > 128.
    {
        return Err(invalid(
            "finite Gaussian tail parameters and cost order in [0,128] required",
        ));
    }
    let u = ((cutoff - center_norm_upper).max(0.) / sigma).powi(2);
    if !u.is_finite() {
        return Err(invalid("Gaussian cutoff standardized square overflows f64"));
    }
    let log_escape_upper = chi_square_weighted_log_upper(dimension, 0, u).min(0.);
    let log_tail_cost_upper = if cost_order == 0. {
        log_escape_upper
    } else {
        let q = cost_order / 2.;
        let k = q.ceil() as usize;
        let log_integer_tail = chi_square_weighted_log_upper(dimension, k, u);
        let log_fractional_tail = if q == k as f64 {
            log_integer_tail
        } else {
            (q / k as f64) * log_integer_tail + (1. - q / k as f64) * log_escape_upper
        };
        let center_term = if center_norm_upper == 0. {
            f64::NEG_INFINITY
        } else {
            cost_order * center_norm_upper.ln() + log_escape_upper
        };
        (cost_order - 1.).max(0.) * 2_f64.ln()
            + log_add(center_term, cost_order * sigma.ln() + log_fractional_tail)
    };
    Ok(GaussianTailBudget {
        dimension, center_norm_upper, sigma, cutoff, cost_order,
        escape_upper: upper_exp(log_escape_upper)?.min(1.),
        log_escape_upper,
        tail_cost_upper: upper_exp(log_tail_cost_upper)?,
        log_tail_cost_upper,
        source_labels: vec!["lem-convergence-gaussian-regional-tail".into()],
        scope: "conditional one-particle Gaussian tail; analytic envelope, no sampled probability substitution or whole-swarm mixing claim".into(),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MomentTailBudget {
    pub moment_order: f64,
    pub moment_bound: f64,
    pub cutoff: f64,
    pub cost_order: f64,
    pub mass_upper: f64,
    pub tail_cost_upper: f64,
    pub source_labels: Vec<String>,
    pub scope: String,
}

/// Caller supplies a proved moment hypothesis for the precise normalized law.
/// A finite sampled moment is not a uniform-time analytic moment certificate.
pub fn moment_tail_budget(
    moment_order: f64,
    moment_bound: f64,
    cutoff: f64,
    cost_order: f64,
) -> Result<MomentTailBudget> {
    if !positive(moment_order)
        || !finite_nonnegative(moment_bound)
        || !positive(cutoff)
        || !finite_nonnegative(cost_order)
        || moment_order <= cost_order
    {
        return Err(invalid("a proved finite p-moment and 0<=r<p are required"));
    }
    let evaluate = |order: f64| -> Result<f64> {
        if moment_bound == 0. {
            Ok(0.)
        } else {
            upper_exp(moment_bound.ln() - order * cutoff.ln())
        }
    };
    Ok(MomentTailBudget {
        moment_order, moment_bound, cutoff, cost_order,
        mass_upper: evaluate(moment_order)?.min(1.),
        tail_cost_upper: evaluate(moment_order - cost_order)?,
        source_labels: vec!["lem-convergence-moment-tail-transfer".into()],
        scope: "conditional on the supplied analytic moment bound for this law; N-normalized full-swarm moments and conditional uniform-alive moments require their respective denominators".into(),
    })
}

/// For any coupling of X,Y with p-moments Mx,My, bounds the tail contribution
/// of cost <=c0+c1(||X||^r+||Y||^r) on {max(||X||,||Y||)>R}.
/// No independence or assignment of persistent walker identities is needed.
pub fn coupled_tail_cost_upper(
    moment_order: f64,
    moment_x: f64,
    moment_y: f64,
    cutoff: f64,
    cost_order: f64,
    constant_cost: f64,
    polynomial_cost: f64,
) -> Result<f64> {
    let x = moment_tail_budget(moment_order, moment_x, cutoff, cost_order)?;
    let y = moment_tail_budget(moment_order, moment_y, cutoff, cost_order)?;
    if !finite_nonnegative(constant_cost) || !finite_nonnegative(polynomial_cost) {
        return Err(invalid("nonnegative cost-growth coefficients required"));
    }
    let union = (x.mass_upper + y.mass_upper).min(1.);
    if cost_order == 0. {
        let result = (constant_cost + 2. * polynomial_cost) * union;
        if !result.is_finite() {
            return Err(invalid("constant coupled tail envelope overflows f64"));
        }
        return Ok(
            if result == 0.
                && (constant_cost > 0. || polynomial_cost > 0.)
                && (moment_x > 0. || moment_y > 0.)
            {
                f64::MIN_POSITIVE
            } else {
                result
            },
        );
    }
    let a = cost_order / moment_order;
    let moment_log = |m: f64| if m == 0. { f64::NEG_INFINITY } else { m.ln() };
    let cutoff_log = (moment_order - cost_order) * cutoff.ln();
    let lx = moment_log(moment_x);
    let ly = moment_log(moment_y);
    let cross_log = if moment_x == 0. || moment_y == 0. {
        f64::NEG_INFINITY
    } else {
        log_add(a * lx + (1. - a) * ly, a * ly + (1. - a) * lx) - cutoff_log
    };
    let polynomial_log = log_add(log_add(lx - cutoff_log, ly - cutoff_log), cross_log);
    let constant_log = if constant_cost == 0. || union == 0. {
        f64::NEG_INFINITY
    } else {
        constant_cost.ln() + union.ln()
    };
    let weighted_polynomial_log = if polynomial_cost == 0. {
        f64::NEG_INFINITY
    } else {
        polynomial_cost.ln() + polynomial_log
    };
    let bound_log = log_add(constant_log, weighted_polynomial_log);
    if bound_log == f64::NEG_INFINITY {
        return Ok(0.);
    }
    upper_exp(bound_log)
}

/// Actual BAOAB position center for any explicitly supplied landscape gradient.
/// The final B-force and velocity cap do not modify this position.
pub fn baoab_position_center(
    positions: &[f64],
    velocities: &[f64],
    gradient: &[f64],
    h: f64,
    gamma: f64,
) -> Result<Vec<f64>> {
    if positions.is_empty()
        || positions.len() != velocities.len()
        || positions.len() != gradient.len()
        || !positive(h)
        || !positive(gamma)
        || positions
            .iter()
            .chain(velocities)
            .chain(gradient)
            .any(|x| !x.is_finite())
    {
        return Err(invalid(
            "finite same-dimensional native BAOAB inputs required",
        ));
    }
    let b = 0.5 * h * (1. + (-gamma * h).exp());
    let center: Vec<_> = positions
        .iter()
        .zip(velocities)
        .zip(gradient)
        .map(|((&x, &v), &f)| x + b * v - 0.5 * h * b * f)
        .collect();
    if center.iter().any(|x| !x.is_finite()) {
        return Err(invalid("BAOAB position center overflows f64"));
    }
    Ok(center)
}

/// Explicit nonquadratic example, not a replacement algorithm or a global QSD
/// certificate. All moment bounds here are conditional one-step Gaussian ones.
pub fn nonquadratic_gaussian_example(dimension: usize) -> Result<Value> {
    if dimension == 0 {
        return Err(invalid("positive example dimension required"));
    }
    let epsilon = 0.2_f64;
    let frequency = 2_f64;
    let h = 0.04_f64;
    let gamma = 1_f64;
    let thermostat = 1_f64;
    let sigma_position = 0.1_f64;
    let ou_variance = thermostat.powi(2) * (1. - (-2. * gamma * h).exp()) / (2. * gamma);
    let sigma = (0.25 * h.powi(2) * ou_variance + h * sigma_position.powi(2)).sqrt();
    let region_radius = 0.15_f64;
    let velocity_radius = 0.1_f64;
    let b = 0.5 * h * (1. + (-gamma * h).exp());
    let force_coefficient = 0.5 * h * b;
    let center_slope = 1. - force_coefficient;
    let sine_upper = (frequency * region_radius).sin();
    let mut regions = vec![];
    for (name, coordinate) in [("slow", 0.), ("core", std::f64::consts::PI / frequency)] {
        let x = vec![coordinate; dimension];
        let gradient: Vec<_> = x
            .iter()
            .map(|&xj| xj - epsilon * frequency * (frequency * xj).sin())
            .collect();
        let mean = baoab_position_center(&x, &vec![0.; dimension], &gradient, h, gamma)?;
        // Uniform over x_j in this regional interval and |v_j|<=.1;
        // both example intervals have |sin(2x_j)|<=sin(.3).
        let centers = vec![
            CoordinateInterval {
                lower: center_slope * (coordinate - region_radius)
                    - b * velocity_radius
                    - force_coefficient * epsilon * frequency * sine_upper,
                upper: center_slope * (coordinate + region_radius)
                    + b * velocity_radius
                    + force_coefficient * epsilon * frequency * sine_upper,
            };
            dimension
        ];
        let target = vec![
            CoordinateInterval {
                lower: coordinate - 0.3,
                upper: coordinate + 0.3
            };
            dimension
        ];
        let landing = gaussian_region_budget(&centers, &target, sigma, sigma)?;
        let c = landing.center_norm_upper;
        let moment_four = c.powi(4)
            + 2. * (dimension as f64 + 2.) * c.powi(2) * sigma.powi(2)
            + dimension as f64 * (dimension as f64 + 2.) * sigma.powi(4);
        let regional_hessian_interval = if name == "slow" {
            [
                1. - epsilon * frequency.powi(2),
                1. - epsilon * frequency.powi(2) * (frequency * region_radius).cos(),
            ]
        } else {
            [
                1. + epsilon * frequency.powi(2) * (frequency * region_radius).cos(),
                1. + epsilon * frequency.powi(2),
            ]
        };
        regions.push(json!({"region":name,"position":x,"gradient":gradient,"position_center":mean,
            "position_coordinate_interval":[coordinate-region_radius,coordinate+region_radius],
            "velocity_coordinate_interval":[-velocity_radius,velocity_radius],
            "target_coordinate_interval":[coordinate-0.3,coordinate+0.3],
            "regional_hessian_interval":regional_hessian_interval,
            "local_hessian":1.-epsilon*frequency.powi(2)*(frequency*coordinate).cos(),
            "landing":landing,"gaussian_tail":gaussian_tail_budget(dimension,c,sigma,3.*(dimension as f64).sqrt(),2.)?,
            "conditional_moment_tail":moment_tail_budget(4.,moment_four,3.*(dimension as f64).sqrt(),2.)?}));
    }
    Ok(
        json!({"dimension":dimension,"landscape":"V(x)=||x||^2/2 + epsilon sum_j cos(frequency x_j)",
        "epsilon":epsilon,"frequency":frequency,"h":h,"gamma":gamma,
        "hessian_interval":[1.-epsilon*frequency.powi(2),1.+epsilon*frequency.powi(2)],
        "force_perturbation_norm_upper":epsilon*frequency*(dimension as f64).sqrt(),
        "confinement":"V(x) >= ||x||^2/2 - epsilon*d, on all R^d",
        "position_noise_std":sigma,"regions":regions,
        "scope":"actual one-step BAOAB centers and conditional Gaussian moments; core/slow analysis regions do not truncate the landscape or certify the parent's coupled discrepancy coefficients"}),
    )
}
