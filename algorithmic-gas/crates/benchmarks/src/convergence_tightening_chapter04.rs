//! Dimension-explicit, population-uniform Chapter 4 specializations.
//! Existing native data stay immutable; derived conditional certificates are
//! saved separately. Moment bounds do not imply an unconditional law rate.
use crate::convergence_experiments::{ArchiveStore, sha256_file};
use algorithmic_gas::{GasError, Result, kinetic::ViscousForceConfig};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};

fn error(s: impl Into<String>) -> GasError {
    GasError::Configuration(s.into())
}
fn norm2(x: &[f64]) -> f64 {
    x.iter().map(|x| x * x).sum()
}
fn delta(x: &[f64], y: &[f64]) -> Vec<f64> {
    x.iter().zip(y).map(|(x, y)| x - y).collect()
}
fn mean(x: &[Vec<f64>]) -> Vec<f64> {
    (0..x[0].len())
        .map(|a| x.iter().map(|p| p[a]).sum::<f64>() / x.len() as f64)
        .collect()
}
fn points(v: &Value) -> Result<Vec<Vec<f64>>> {
    serde_json::from_value(v.clone()).map_err(|e| error(e.to_string()))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PotentialProfile {
    pub dimension: usize,
    pub quadratic_curvature: f64,
    pub cosine_amplitude: f64,
    pub frequency: f64,
}
impl PotentialProfile {
    pub fn validate(&self) -> Result<()> {
        if self.dimension == 0
            || [
                self.quadratic_curvature,
                self.cosine_amplitude,
                self.frequency,
            ]
            .iter()
            .any(|x| !x.is_finite())
            || self.quadratic_curvature < 0.
            || self.frequency < 0.
        {
            return Err(error(
                "Invalid dimension or quadratic/cosine potential parameters",
            ));
        }
        Ok(())
    }
    pub fn curvature_interval(&self) -> Result<(f64, f64)> {
        self.validate()?;
        let a = self.cosine_amplitude.abs() * self.frequency.powi(2);
        Ok((self.quadratic_curvature - a, self.quadratic_curvature + a))
    }
    pub fn force_lipschitz(&self) -> Result<f64> {
        let (lo, hi) = self.curvature_interval()?;
        Ok(lo.abs().max(hi.abs()))
    }
    pub fn value(&self, x: &[f64]) -> Result<f64> {
        self.validate()?;
        if x.len() != self.dimension || x.iter().any(|z| !z.is_finite()) {
            return Err(error("Potential point schema mismatch"));
        }
        Ok(0.5 * self.quadratic_curvature * norm2(x)
            + self.cosine_amplitude
                * x.iter()
                    .map(|z| 1. - (self.frequency * z).cos())
                    .sum::<f64>())
    }
    /// Exact stationary-point enumeration on a genuine declared scalar box.
    /// The moment reward bound below needs no box at all.
    pub fn reward_lipschitz_on_box(&self, radius: f64) -> Result<f64> {
        self.validate()?;
        if !radius.is_finite() || radius < 0. {
            return Err(error("Invalid analytic reward box radius"));
        }
        let k = self.quadratic_curvature;
        let a = self.cosine_amplitude;
        let w = self.frequency;
        let g = |x: f64| (k * x + a * w * (w * x).sin()).abs();
        let mut maximum = g(radius).max(g(-radius));
        if a != 0. && w > 0. {
            let c = -k / (a * w * w);
            if c.abs() <= 1. {
                let angle = c.acos();
                for sign in [-1., 1.] {
                    let start =
                        ((-radius * w - sign * angle) / std::f64::consts::TAU).ceil() as i64;
                    let end = ((radius * w - sign * angle) / std::f64::consts::TAU).floor() as i64;
                    if end - start > 100_000 {
                        return Err(error(
                            "Analytic frequency exceeds stationary-point enumeration budget",
                        ));
                    }
                    for j in start..=end {
                        maximum =
                            maximum.max(g((sign * angle + std::f64::consts::TAU * j as f64) / w));
                    }
                }
            }
        }
        Ok((self.dimension as f64).sqrt() * maximum)
    }
    /// |mean U(X)-mean U(Y)| <= L/√2 sqrt(M2X+M2Y) sqrt(E|X-Y|²).
    pub fn reward_mean_bound(&self, m2x: f64, m2y: f64, cost: f64) -> Result<f64> {
        if [m2x, m2y, cost].iter().any(|x| !x.is_finite() || *x < 0.) {
            return Err(error(
                "Reward certificate requires finite nonnegative normalized second moments",
            ));
        }
        Ok(self.force_lipschitz()? * ((m2x + m2y) * cost / 2.).sqrt())
    }

    /// Split confining and bounded periodic terms. Coordinate transport moments
    /// pay for only the directions displaced by an admissible probability plan.
    pub fn reward_separated_moment_bound(
        &self,
        m2x: f64,
        m2y: f64,
        coordinate_cost: &[f64],
    ) -> Result<f64> {
        self.validate()?;
        if coordinate_cost.len() != self.dimension
            || coordinate_cost
                .iter()
                .chain([&m2x, &m2y])
                .any(|v| !v.is_finite() || *v < 0.)
        {
            return Err(error("Invalid coordinate reward coupling moments"));
        }
        let total = coordinate_cost.iter().sum::<f64>();
        Ok(self.quadratic_curvature * ((m2x + m2y) * total / 2.).sqrt()
            + self.cosine_amplitude.abs()
                * coordinate_cost
                    .iter()
                    .map(|c| 2_f64.min(self.frequency * c.sqrt()))
                    .sum::<f64>())
    }
}

/// Copied-position variance inside an r-dimensional donor support of diameter D.
/// r may be a certified upper bound on affine rank. Isotropic noise still uses d.
pub fn dimension_reset(
    dimension: usize,
    rank: usize,
    diameter_squared: f64,
    jitter: f64,
    mean_acceptance: f64,
    n: usize,
) -> Result<f64> {
    if dimension == 0
        || rank > dimension
        || n == 0
        || [diameter_squared, jitter, mean_acceptance]
            .iter()
            .any(|x| !x.is_finite() || *x < 0.)
        || mean_acceptance > 1. + 1e-10
        || (rank == 0 && diameter_squared != 0.)
    {
        return Err(error("Invalid dimension-aware reset certificate"));
    }
    let geometry = rank as f64 * diameter_squared / (2. * (rank + 1) as f64);
    Ok(geometry + (1. - 1. / n as f64) * dimension as f64 * jitter * jitter * mean_acceptance)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OccupationCertificate {
    pub dimension: usize,
    pub entering_alive: usize,
    pub normalized_occupation: Vec<f64>,
    pub normalized_second_moment: f64,
    pub occupation_variance: f64,
    pub copy_barycenter_covariance: f64,
    pub mean_acceptance: f64,
    pub jitter_contribution: f64,
    pub expected_variance: f64,
    pub second_moment_upper: f64,
    pub coordinate_span_dimension: usize,
    pub diameter_squared: f64,
    pub dimension_upper: f64,
}

/// Conditional on complete sampled fitness; independent recipient source draws
/// and centered independent unit-covariance isotropic jitter. Dead coordinates
/// cannot enter this certificate because only live source columns carry mass.
pub fn occupation_certificate(
    x: &[Vec<f64>],
    alive: &[bool],
    accepted: &[Vec<f64>],
    jitter: f64,
) -> Result<OccupationCertificate> {
    let n = x.len();
    let d = x.first().map_or(0, Vec::len);
    if n == 0
        || d == 0
        || alive.len() != n
        || accepted.len() != n
        || x.iter()
            .any(|r| r.len() != d || r.iter().any(|z| !z.is_finite()))
        || accepted
            .iter()
            .any(|r| r.len() != n || r.iter().any(|z| !z.is_finite() || *z < 0.))
        || !jitter.is_finite()
        || jitter < 0.
    {
        return Err(error("Invalid conditional occupation inputs"));
    }
    let live: Vec<_> = x
        .iter()
        .zip(alive)
        .filter_map(|(p, &a)| a.then_some(p.clone()))
        .collect();
    if live.is_empty() {
        return Err(GasError::Extinction);
    }
    let p: Vec<_> = accepted.iter().map(|r| r.iter().sum::<f64>()).collect();
    if p.iter().any(|p| *p > 1. + 1e-10)
        || (0..n).any(|i| !alive[i] && (p[i] - 1.).abs() > 1e-10)
        || accepted
            .iter()
            .any(|r| r.iter().zip(alive).any(|(&q, &a)| !a && q != 0.))
    {
        return Err(error(
            "Occupation requires complete revival and live-only donor columns",
        ));
    }
    let q: Vec<_> = (0..n)
        .map(|j| {
            ((if alive[j] { 1. - p[j] } else { 0. }) + accepted.iter().map(|r| r[j]).sum::<f64>())
                / n as f64
        })
        .collect();
    let center = mean(&live);
    let mut m = vec![0.; d];
    let mut second = 0.;
    for j in 0..n {
        if q[j] > 0. {
            let dx = delta(&x[j], &center);
            second += q[j] * norm2(&dx);
            for a in 0..d {
                m[a] += q[j] * dx[a];
            }
        }
    }
    let occupation_variance = second - norm2(&m);
    let mut trace = 0.;
    for i in 0..n {
        let mut row_mean = vec![0.; d];
        for a in 0..d {
            row_mean[a] = (if alive[i] { 1. - p[i] } else { 0. })
                * if alive[i] { x[i][a] } else { 0. }
                + accepted[i]
                    .iter()
                    .enumerate()
                    .filter(|(_, q)| **q > 0.)
                    .map(|(j, q)| q * x[j][a])
                    .sum::<f64>();
        }
        if alive[i] {
            trace += (1. - p[i]) * norm2(&delta(&x[i], &row_mean));
        }
        for j in 0..n {
            if accepted[i][j] > 0. {
                trace += accepted[i][j] * norm2(&delta(&x[j], &row_mean));
            }
        }
    }
    let mean_acceptance = p.iter().sum::<f64>() / n as f64;
    let jitter_contribution = (1. - 1. / n as f64) * d as f64 * jitter * jitter * mean_acceptance;
    let covariance = trace / (n * n) as f64;
    let rank = (0..d)
        .filter(|&a| live.iter().any(|p| p[a] != live[0][a]))
        .count();
    let diameter_squared = live
        .iter()
        .flat_map(|a| live.iter().map(move |b| norm2(&delta(a, b))))
        .fold(0., f64::max);
    Ok(OccupationCertificate {
        dimension: d,
        entering_alive: live.len(),
        normalized_occupation: q,
        normalized_second_moment: second,
        occupation_variance,
        copy_barycenter_covariance: covariance,
        mean_acceptance,
        jitter_contribution,
        expected_variance: occupation_variance - covariance + jitter_contribution,
        second_moment_upper: occupation_variance + jitter_contribution,
        coordinate_span_dimension: rank,
        diameter_squared,
        dimension_upper: dimension_reset(d, rank, diameter_squared, jitter, mean_acceptance, n)?,
    })
}

/// Row-normalized max-row derivative majorant on the SAME genuine proof-event
/// radius R. The Gaussian draws themselves remain unbounded.
pub fn row_uniform_derivative(
    n: usize,
    nu: f64,
    velocity_radius: f64,
    position_radius: f64,
    rho: f64,
) -> Result<f64> {
    if n == 0
        || [nu, velocity_radius, position_radius]
            .iter()
            .any(|x| !x.is_finite() || *x < 0.)
        || !rho.is_finite()
        || rho <= 0.
    {
        return Err(error("Invalid row derivative parameters"));
    }
    Ok(if n <= 2 {
        0.
    } else {
        1.5 * 3_f64.sqrt() * nu * velocity_radius * position_radius / rho.powi(2)
    })
}

/// Symmetric count-normalized global Hilbert derivative, not maximum-row norm.
pub fn count_hilbert_derivative(n: usize, nu: f64, velocity_radius: f64, rho: f64) -> Result<f64> {
    if n == 0
        || !nu.is_finite()
        || nu < 0.
        || !velocity_radius.is_finite()
        || velocity_radius < 0.
        || !rho.is_finite()
        || rho <= 0.
    {
        return Err(error("Invalid Hilbert derivative parameters"));
    }
    Ok(if n == 1 {
        0.
    } else {
        2. * nu * velocity_radius * (-0.5_f64).exp() / rho
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PropagatedMomentBudget {
    pub dimension: usize,
    pub source_second_moment: f64,
    pub a_x: f64,
    pub b: f64,
    pub c: f64,
    pub sigma_y_squared: f64,
    pub sigma_w_squared: f64,
    pub prepared_position_second_moment: f64,
    pub prepared_velocity_second_moment: f64,
    pub ou_position_second_moment: f64,
    pub ou_velocity_second_moment: f64,
    pub final_position_second_moment: f64,
    pub final_phase_second_moment: f64,
}
/// Full-Gaussian, normalized moment propagation. A uniform input H2 must be
/// supplied by confinement/Safe Harbor; the routine never infers it from the
/// largest sampled particle or assumes unbounded sources have compact support.
#[allow(clippy::too_many_arguments)]
pub fn propagated_moment_budget(
    profile: &PotentialProfile,
    source_second_moment: f64,
    mean_acceptance: f64,
    jitter: f64,
    velocity_radius: f64,
    velocity_cap: f64,
    h: f64,
    gamma: f64,
    q_squared: f64,
    position_noise: f64,
    nu: f64,
) -> Result<PropagatedMomentBudget> {
    profile.validate()?;
    if [
        source_second_moment,
        mean_acceptance,
        jitter,
        velocity_radius,
        velocity_cap,
        h,
        gamma,
        q_squared,
        position_noise,
        nu,
    ]
    .iter()
    .any(|x| !x.is_finite() || *x < 0.)
        || h == 0.
        || mean_acceptance > 1. + 1e-10
        || h * nu / 2. > 1.
    {
        return Err(error(
            "Moment propagation needs finite budgets and a convex first row kick (0<=t nu<=1)",
        ));
    }
    let d = profile.dimension as f64;
    let t = h / 2.;
    let c = (-gamma * h).exp();
    let ax = 1. - profile.quadratic_curvature * t * t * (1. + c);
    let b = t * (1. + c);
    let periodic = profile.cosine_amplitude.abs() * profile.frequency * d.sqrt();
    let by = ax.abs() * source_second_moment.sqrt() + b * velocity_radius + b * t * periodic;
    let bw = c * t * profile.quadratic_curvature * source_second_moment.sqrt()
        + c * velocity_radius
        + c * t * periodic;
    let sy = ax * ax * jitter * jitter * mean_acceptance + t * t * q_squared;
    let sw =
        c * c * t * t * profile.quadratic_curvature.powi(2) * jitter * jitter * mean_acceptance
            + q_squared;
    let y = (by + (d * sy).sqrt()).powi(2);
    let w = (bw + (d * sw).sqrt()).powi(2);
    let final_x = y + d * position_noise * position_noise;
    Ok(PropagatedMomentBudget {
        dimension: profile.dimension,
        source_second_moment,
        a_x: ax,
        b,
        c,
        sigma_y_squared: sy,
        sigma_w_squared: sw,
        prepared_position_second_moment: source_second_moment
            + d * jitter * jitter * mean_acceptance,
        prepared_velocity_second_moment: velocity_radius.powi(2),
        ou_position_second_moment: y,
        ou_velocity_second_moment: w,
        final_position_second_moment: final_x,
        final_phase_second_moment: final_x + velocity_cap.powi(2),
    })
}

fn logsumexp(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    maximum + values.iter().map(|v| (v - maximum).exp()).sum::<f64>().ln()
}
/// Pointwise optimized Young inequality, valid even if the bounded shift
/// depends on the Gaussian. The full Gaussian tail is integrated exactly.
pub fn gaussian_log_budget(d: usize, chi: f64, bound: f64, variance: f64) -> Result<f64> {
    if d == 0
        || !chi.is_finite()
        || chi <= 0.
        || [bound, variance].iter().any(|x| !x.is_finite() || *x < 0.)
        || 2. * chi * variance >= 1.
    {
        return Err(error("Invalid full-Gaussian exponential budget"));
    }
    if variance == 0. {
        return Ok(chi * bound * bound);
    }
    let a = 2. * chi * variance;
    if bound == 0. {
        return Ok(-0.5 * d as f64 * (-a).ln_1p());
    }
    let lambda = (a + (a * a + 4. * (1. - a) * d as f64 * variance / (bound * bound)).sqrt())
        / (2. * (1. - a));
    Ok(chi * (1. + lambda) * bound * bound - 0.5 * d as f64 * (-a * (1. + 1. / lambda)).ln_1p())
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RowContinuityBudget {
    pub dimension: usize,
    pub chi: f64,
    pub old_log_m: f64,
    pub sharp_log_m: f64,
    pub old_h: f64,
    pub sharp_h: f64,
    pub old_log_k: f64,
    pub sharp_log_k: f64,
    pub gamma: f64,
    pub marked_exponent: f64,
}
fn row_log_k(chi: f64, log_m: f64, h: f64, rho: f64) -> f64 {
    let r0_squared = (std::f64::consts::LN_2 + log_m) / chi;
    let b_squared =
        (std::f64::consts::LN_2 + log_m + r0_squared / rho.powi(2) + 1. / rho.powi(2)) / chi;
    let ell = (-0.5_f64).exp() / rho;
    let kg = 64_f64.ln() + 2. * (1. + 2. * ell * h.sqrt()).ln() + 4. * r0_squared / rho.powi(2);
    let kt = 2_f64.ln()
        + b_squared.ln()
        + 0.5 * (64_f64.ln() + log_m + logsumexp(&[0., 4_f64.ln() + log_m - 2. - 2. * chi.ln()]));
    let last = 8_f64.ln() + b_squared.ln() + (1. + h).ln();
    0.5 * logsumexp(&[kg, kt, last])
}
/// Sharpened RWM register for the actual harmonic source-box population laws.
/// This is not a pathwise degree floor for a finite random empirical array and
/// does not assert the small-viscosity attraction endpoints hold at the preset.
#[allow(clippy::too_many_arguments)]
pub fn row_continuity_budget(
    d: usize,
    source_radius: f64,
    jitter: f64,
    velocity_radius: f64,
    h: f64,
    gamma: f64,
    q_squared: f64,
    rho: f64,
) -> Result<RowContinuityBudget> {
    if !source_radius.is_finite() || source_radius < 0. {
        return Err(error("Invalid population source radius"));
    }
    let profile = PotentialProfile {
        dimension: d,
        quadratic_curvature: 1.,
        cosine_amplitude: 0.,
        frequency: 0.,
    };
    let moment = propagated_moment_budget(
        &profile,
        source_radius.powi(2),
        1.,
        jitter,
        velocity_radius,
        velocity_radius,
        h,
        gamma,
        q_squared,
        0.,
        0.,
    )?;
    if jitter <= 0. || !rho.is_finite() || rho <= 0. {
        return Err(error(
            "Row register requires positive Gaussian jitter and bandwidth",
        ));
    }
    let t = h / 2.;
    let by = moment.a_x.abs() * source_radius + moment.b * velocity_radius;
    let bw = moment.c * t * source_radius + moment.c * velocity_radius;
    let chi = 1_f64
        .min(1. / (8. * jitter * jitter))
        .min(1. / (8. * moment.sigma_y_squared))
        .min(1. / (8. * moment.sigma_w_squared));
    let old_log_m = 0.5 * d as f64 * 2_f64.ln()
        + 2. * chi
            * source_radius
                .powi(2)
                .max(velocity_radius.powi(2))
                .max(by * by)
                .max(bw * bw);
    let x_log = -0.5 * d as f64 * (-2. * chi * jitter * jitter).ln_1p()
        + chi * source_radius.powi(2) / (1. - 2. * chi * jitter * jitter);
    let sharp_log_m = x_log
        .max(chi * velocity_radius.powi(2))
        .max(gaussian_log_budget(d, chi, by, moment.sigma_y_squared)?)
        .max(gaussian_log_budget(d, chi, bw, moment.sigma_w_squared)?);
    let old_h = old_log_m / chi;
    let sharp_h = moment
        .prepared_position_second_moment
        .max(moment.prepared_velocity_second_moment)
        .max(moment.ou_position_second_moment)
        .max(moment.ou_velocity_second_moment)
        .min(sharp_log_m / chi);
    let c = 4. / rho.powi(2);
    let exponent = chi / (4. * (c + chi));
    Ok(RowContinuityBudget {
        dimension: d,
        chi,
        old_log_m,
        sharp_log_m,
        old_h,
        sharp_h,
        old_log_k: row_log_k(chi, old_log_m, old_h, rho),
        sharp_log_k: row_log_k(chi, sharp_log_m, sharp_h, rho),
        gamma: exponent,
        marked_exponent: exponent.powi(2) / 32.,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ViscousCertificate {
    pub dimension: usize,
    pub force: Vec<Vec<f64>>,
    pub jacobian_action: Vec<Vec<f64>>,
    pub row_moment_derivative: Vec<f64>,
    pub max_row_moment_derivative: f64,
    pub direction_max_row_norm: f64,
    pub action_max_row_norm: f64,
    pub direction_hilbert_norm: f64,
    pub action_hilbert_norm: f64,
}
pub fn viscous_certificate(
    x: &[Vec<f64>],
    v: &[Vec<f64>],
    h: &[Vec<f64>],
    config: &ViscousForceConfig,
) -> Result<ViscousCertificate> {
    let n = x.len();
    let d = x.first().map_or(0, Vec::len);
    if n == 0
        || d == 0
        || v.len() != n
        || h.len() != n
        || x.iter()
            .chain(v)
            .chain(h)
            .any(|p| p.len() != d || p.iter().any(|z| !z.is_finite()))
        || !config.coefficient.is_finite()
        || config.coefficient < 0.
        || !config.bandwidth.is_finite()
        || config.bandwidth <= 0.
    {
        return Err(error("Invalid viscous derivative inputs"));
    }
    let mut force = vec![vec![0.; d]; n];
    let mut action = force.clone();
    let mut bounds = vec![0.; n];
    let inv = 1. / config.bandwidth.powi(2);
    for i in 0..n {
        let weights: Vec<_> = (0..n)
            .map(|j| {
                if i == j {
                    0.
                } else {
                    (-0.5 * inv * norm2(&delta(&x[i], &x[j]))).exp()
                }
            })
            .collect();
        let mass = weights.iter().sum::<f64>();
        if mass == 0. {
            if config.row_normalized && n > 1 {
                return Err(error(
                    "Gaussian row denominator underflow prevents a ratio derivative certificate",
                ));
            }
            continue;
        }
        if config.row_normalized {
            let probs: Vec<_> = weights.iter().map(|w| w / mass).collect();
            let mx: Vec<_> = (0..d)
                .map(|a| (0..n).map(|j| probs[j] * x[j][a]).sum::<f64>())
                .collect();
            let mv: Vec<_> = (0..d)
                .map(|a| (0..n).map(|j| probs[j] * v[j][a]).sum::<f64>())
                .collect();
            let vx = (0..n)
                .map(|j| probs[j] * norm2(&delta(&x[j], &mx)))
                .sum::<f64>();
            let vv = (0..n)
                .map(|j| probs[j] * norm2(&delta(&v[j], &mv)))
                .sum::<f64>();
            let rx = (0..n)
                .map(|j| probs[j] * norm2(&delta(&x[i], &x[j])))
                .sum::<f64>();
            bounds[i] = if n <= 2 {
                0.
            } else {
                config.coefficient * inv * vv.sqrt() * (vx.sqrt() + rx.sqrt())
            };
            for a in 0..d {
                force[i][a] = config.coefficient * (mv[a] - v[i][a]);
            }
            for j in 0..n {
                if probs[j] > 0. {
                    let log_derivative = -inv
                        * delta(&x[i], &x[j])
                            .iter()
                            .zip(delta(&h[i], &h[j]))
                            .map(|(x, h)| x * h)
                            .sum::<f64>();
                    for a in 0..d {
                        action[i][a] +=
                            config.coefficient * probs[j] * (v[j][a] - mv[a]) * log_derivative;
                    }
                }
            }
        } else {
            let vx = (0..n)
                .map(|j| weights[j] * norm2(&delta(&x[i], &x[j])))
                .sum::<f64>()
                / n as f64;
            let vv = (0..n)
                .map(|j| weights[j] * norm2(&delta(&v[i], &v[j])))
                .sum::<f64>()
                / n as f64;
            bounds[i] = 2. * config.coefficient * inv * (vx * vv).sqrt();
            for j in 0..n {
                if weights[j] > 0. {
                    let dw = -weights[j]
                        * inv
                        * delta(&x[i], &x[j])
                            .iter()
                            .zip(delta(&h[i], &h[j]))
                            .map(|(x, h)| x * h)
                            .sum::<f64>();
                    for a in 0..d {
                        force[i][a] +=
                            config.coefficient * weights[j] * (v[j][a] - v[i][a]) / n as f64;
                        action[i][a] += config.coefficient * dw * (v[j][a] - v[i][a]) / n as f64;
                    }
                }
            }
        }
    }
    Ok(ViscousCertificate {
        dimension: d,
        force,
        jacobian_action: action.clone(),
        max_row_moment_derivative: bounds.iter().copied().fold(0., f64::max),
        row_moment_derivative: bounds,
        direction_max_row_norm: h.iter().map(|r| norm2(r).sqrt()).fold(0., f64::max),
        action_max_row_norm: action.iter().map(|r| norm2(r).sqrt()).fold(0., f64::max),
        direction_hilbert_norm: (h.iter().map(|r| norm2(r)).sum::<f64>() / n as f64).sqrt(),
        action_hilbert_norm: (action.iter().map(|r| norm2(r)).sum::<f64>() / n as f64).sqrt(),
    })
}

/// Analytic upper certificates for def-slc-profiles. They bound the defining
/// suprema; they are not estimates from observed walkers. None depends on N.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionalLandscapeProfile {
    pub dimension: usize,
    pub region_lower: Vec<f64>,
    pub region_upper: Vec<f64>,
    pub harmonic_curvature: f64,
    pub bounded_perturbation_amplitude: f64,
    pub perturbation_lipschitz: f64,
    pub regional_perturbation_lipschitz: f64,
    pub curvature_lower: f64,
    pub curvature_upper: f64,
    pub force_lipschitz: f64,
    pub force_sup_upper: f64,
    pub scale: f64,
    pub force_modulus_upper: f64,
    pub perturbation_modulus_upper: f64,
    pub comparison_lipschitz: f64,
    pub excess_modulus_upper: f64,
    pub restoring_k: f64,
    pub radial_defect_upper: f64,
    pub pairwise_defect_upper: f64,
    pub global_radial_defect_upper: Option<f64>,
    pub reward_quadratic_growth: f64,
    pub reward_constant_growth: f64,
    pub reward_oscillation_upper: f64,
    pub certified_restoring_region: bool,
    pub certification: String,
}
fn cosine_range(lo: f64, hi: f64) -> (f64, f64) {
    let mut lower = lo.cos().min(hi.cos());
    let mut upper = lo.cos().max(hi.cos());
    if (lo / std::f64::consts::TAU).ceil() <= (hi / std::f64::consts::TAU).floor() {
        upper = 1.;
    }
    if ((lo - std::f64::consts::PI) / std::f64::consts::TAU).ceil()
        <= ((hi - std::f64::consts::PI) / std::f64::consts::TAU).floor()
    {
        lower = -1.;
    }
    (lower, upper)
}
fn quadratic_max(a: f64, b: f64, r: f64) -> f64 {
    let mut out = 0_f64.max(a * r * r + b * r);
    if a < 0. {
        let t = (-b / (2. * a)).clamp(0., r);
        out = out.max(a * t * t + b * t);
    }
    out
}
/// Convex declared boxes support segment Hessian bounds. A box is an analysis
/// region, never a replacement for the full Gaussian excursions. The global
/// radial defect retains unbounded support and is absent when not finite.
pub fn regional_landscape_profile(
    p: &PotentialProfile,
    lower: &[f64],
    upper: &[f64],
    scale: f64,
    k: f64,
    l: f64,
) -> Result<RegionalLandscapeProfile> {
    p.validate()?;
    if lower.len() != p.dimension
        || upper.len() != p.dimension
        || lower
            .iter()
            .zip(upper)
            .any(|(a, b)| !a.is_finite() || !b.is_finite() || a > b)
        || [scale, k, l].iter().any(|v| !v.is_finite() || *v < 0.)
    {
        return Err(error("Invalid regional profile box or scale"));
    }
    let amplitude = p.cosine_amplitude.abs() * p.frequency;
    let md = amplitude * (p.dimension as f64).sqrt();
    let mut curvature_lower = f64::INFINITY;
    let mut curvature_upper = f64::NEG_INFINITY;
    let mut le = 0_f64;
    let mut radius2 = 0.;
    let mut radial = 0.;
    let mut diameter2 = 0.;
    for (&lo, &hi) in lower.iter().zip(upper) {
        let (c0, c1) = cosine_range(p.frequency * lo, p.frequency * hi);
        let h0 = p.quadratic_curvature + p.cosine_amplitude * p.frequency.powi(2) * c0;
        let h1 = p.quadratic_curvature + p.cosine_amplitude * p.frequency.powi(2) * c1;
        curvature_lower = curvature_lower.min(h0.min(h1));
        curvature_upper = curvature_upper.max(h0.max(h1));
        le = le.max(p.cosine_amplitude.abs() * p.frequency.powi(2) * c0.abs().max(c1.abs()));
        let r = lo.abs().max(hi.abs());
        radius2 += r * r;
        diameter2 += (hi - lo).powi(2);
        radial += quadratic_max(k - p.quadratic_curvature, amplitude, r);
    }
    let lf = curvature_lower.abs().max(curvature_upper.abs());
    let force_sup = p.quadratic_curvature * radius2.sqrt() + md;
    let omega = (lf * scale).min(2. * force_sup);
    let omega_e = (le * scale).min(2. * md);
    let pair_hessian = (k - curvature_lower).max(0.) * scale.powi(2);
    // Maximize the bounded-perturbation quadratic over all separations <=r.
    let pair_bounded = quadratic_max(k - p.quadratic_curvature, 2. * md, scale);
    let pair_lipschitz = (k - p.quadratic_curvature + le).max(0.) * scale.powi(2);
    let global = if md == 0. && k <= p.quadratic_curvature {
        Some(0.)
    } else if k < p.quadratic_curvature {
        Some(md.powi(2) / (4. * (p.quadratic_curvature - k)))
    } else {
        None
    };
    if let Some(g) = global {
        radial = radial.min(g);
    }
    Ok(RegionalLandscapeProfile {dimension:p.dimension,region_lower:lower.to_vec(),region_upper:upper.to_vec(),harmonic_curvature:p.quadratic_curvature,
        bounded_perturbation_amplitude:md,perturbation_lipschitz:p.cosine_amplitude.abs()*p.frequency.powi(2),regional_perturbation_lipschitz:le,
        curvature_lower,curvature_upper,force_lipschitz:lf,force_sup_upper:force_sup,scale,force_modulus_upper:omega,perturbation_modulus_upper:omega_e,
        comparison_lipschitz:l,excess_modulus_upper:(omega-l*scale).max(0.),restoring_k:k,radial_defect_upper:radial,
        pairwise_defect_upper:pair_hessian.min(pair_bounded).min(pair_lipschitz),global_radial_defect_upper:global,
        reward_quadratic_growth:p.quadratic_curvature/2.,reward_constant_growth:2.*p.cosine_amplitude.abs()*p.dimension as f64,
        reward_oscillation_upper:force_sup*diameter2.sqrt(),certified_restoring_region:curvature_lower>0.,
        certification:"Analytic trigonometric interval enclosure and quadratic maxima; convex analysis box; full exterior retained separately; no population size or sampled maxima".into()})
}

fn check(id: &str, observed: f64, bound: f64, label: &str, scope: &str) -> Value {
    json!({"id":id,"observed":observed,"bound":bound,"relation":"upper","passed":observed.is_finite() && bound.is_finite() && observed<=bound+2e-9*(1.+bound.abs()),"standard_error":0.,"kind":"every_retained_input_analytic_check","source_labels":[label],"scope":scope})
}
fn retain_worst<'a>(
    map: &mut BTreeMap<&'a str, (f64, f64, usize)>,
    name: &'a str,
    observed: f64,
    bound: f64,
) {
    let entry = map.entry(name).or_insert((observed, bound, 0));
    entry.2 += 1;
    if observed - bound > entry.0 - entry.1 {
        entry.0 = observed;
        entry.1 = bound;
    }
}
fn empirical_check(id: &str, values: &[(f64, f64)], label: &str) -> Value {
    let n = values.len() as f64;
    let observed = values.iter().map(|v| v.0).sum::<f64>() / n;
    let bound = values.iter().map(|v| v.1).sum::<f64>() / n;
    let residual = observed - bound;
    let standard_error = if n > 1. {
        (values
            .iter()
            .map(|v| (v.0 - v.1 - residual).powi(2))
            .sum::<f64>()
            / (n * (n - 1.)))
            .sqrt()
    } else {
        0.
    };
    json!({"id":id,"observed":observed,"bound":bound,"relation":"upper","passed":residual<=6.*standard_error+2e-9,"standard_error":standard_error,"samples":values.len(),"kind":"conditional_independent_restore_expectation","source_labels":[label],"scope":"Conditional expectations over addressed independent innovations from fixed initial state; six standard errors are a diagnostic, not a deterministic certificate. Full Gaussian innovations and revived recipients retained."})
}
fn number(value: &Value, name: &str) -> Result<f64> {
    value[name]
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| error(format!("Missing finite {name}")))
}
fn stage_points(
    stage: &algorithmic_gas::tracking::StageSnapshot,
    name: &str,
) -> Result<Vec<Vec<f64>>> {
    let f = stage
        .fields
        .get(name)
        .ok_or_else(|| error(format!("Missing archived {name}")))?;
    let d = f.item_shape.iter().product::<usize>();
    if d == 0 {
        return Err(error("Invalid archived field shape"));
    }
    Ok(f.values.chunks(d).map(<[f64]>::to_vec).collect())
}
fn profile(name: &str, d: usize) -> Result<PotentialProfile> {
    let (k, a, w) = match name {
        "quadratic" => (1., 0., 0.),
        "sphere" => (2., 0., 0.),
        "rastrigin" => (2., 10., std::f64::consts::TAU),
        "constant" => (0., 0., 0.),
        _ => return Err(error("Unsupported native potential profile")),
    };
    Ok(PotentialProfile {
        dimension: d,
        quadratic_curvature: k,
        cosine_amplitude: a,
        frequency: w,
    })
}
/// Re-analyze immutable lossless native frames without another simulation.
/// The output must be a fresh directory: ArchiveStore refuses replacement.
pub fn analyze_dataset(dataset: &Path, output: &Path) -> Result<Value> {
    let source_path = dataset.join("chapter04-report.json");
    let source: Value =
        serde_json::from_slice(&fs::read(&source_path).map_err(|e| error(e.to_string()))?)
            .map_err(|e| error(e.to_string()))?;
    let input = ArchiveStore::open(dataset)?;
    let mut store = ArchiveStore::new(output)?;
    let mut comparisons = vec![];
    let mut cases = vec![];
    let mut references = vec![];
    let mut frames_count = 0;
    for case in source["cases"]
        .as_array()
        .ok_or_else(|| error("Missing saved cases"))?
    {
        let id = case["case"].as_u64().unwrap();
        let d = case["d"].as_u64().unwrap() as usize;
        let n = case["N"].as_u64().unwrap() as usize;
        let p = profile(case["benchmark"].as_str().unwrap(), d)?;
        let cfg = &case["config"];
        if cfg["boundary"]["kind"] != "absorbing_box"
            || cfg["boundary"]["field"] != "positions"
            || cfg["boundary"]["domain"]["lower"] != json!(vec![-2.; d])
            || cfg["boundary"]["domain"]["upper"] != json!(vec![2.; d])
            || cfg["kinetic"]["boundary_schedule"] != "end_of_step"
            || cfg["kinetic"]["integrator"]["kind"] != "baoab"
        {
            return Err(error(
                "Tightening reader requires the recorded end-of-step terminal-box BAOAB specialization",
            ));
        }
        for noise in [&cfg["clone_transform"]["jitter"], &cfg["kinetic"]["noise"]] {
            if noise["innovation"] != "gaussian"
                || noise["geometry"]["kind"] != "isotropic"
                || noise["geometry"]["scale"]["kind"] != "constant"
                || noise["geometry"]["scale"]["values"] != json!([1.])
            {
                return Err(error(
                    "Moment specialization requires full isotropic unit-covariance Gaussian innovations",
                ));
            }
        }
        let jitter = number(&cfg["clone_transform"], "jitter_amplitude")?;
        let h = number(&cfg["kinetic"]["integrator"], "dt")?;
        let gamma = number(&cfg["kinetic"]["integrator"], "friction")?;
        let cap = number(&cfg["kinetic"], "velocity_cap")?;
        let vc = (1. + 2. * number(&cfg["clone_transform"], "restitution")?.abs()) * cap;
        let q2 = if gamma == 0. {
            h
        } else {
            -(-2. * gamma * h).exp_m1() / (2. * gamma)
        };
        let s = number(&cfg["kinetic"], "position_diffusion")? * h.sqrt();
        let declared = regional_landscape_profile(
            &p,
            &vec![-2.; d],
            &vec![2.; d],
            4. * (d as f64).sqrt(),
            p.quadratic_curvature / 2.,
            p.force_lipschitz()?,
        )?;
        let core = regional_landscape_profile(
            &p,
            &vec![-0.125; d],
            &vec![0.125; d],
            0.25 * (d as f64).sqrt(),
            p.quadratic_curvature / 2.,
            p.force_lipschitz()?,
        )?;
        let barrier = regional_landscape_profile(
            &p,
            &vec![0.375; d],
            &vec![0.625; d],
            0.25 * (d as f64).sqrt(),
            p.quadratic_curvature / 2.,
            p.force_lipschitz()?,
        )?;
        let terminal_box_uniform_budget =
            propagated_moment_budget(&p, 4. * d as f64, 1., jitter, vc, cap, h, gamma, q2, s, 0.)?;
        let mut samples: BTreeMap<&str, Vec<(f64, f64)>> = BTreeMap::new();
        let mut outputs = vec![];
        let mut exact: BTreeMap<&str, (f64, f64, usize)> = BTreeMap::new();
        let mut old_reset = 0.;
        let mut sharp_reset = 0.;
        let mut sharp_occupation = 0.;
        let mut source_moment_max = 0_f64;
        for filename in case["operator_frame_archives"].as_array().unwrap() {
            let filename = filename.as_str().unwrap();
            let block = input.load_json(filename)?;
            let native = [
                input.load_archive(
                    block["archive_left"]
                        .as_str()
                        .ok_or_else(|| error("Missing native left archive"))?,
                )?,
                input.load_archive(
                    block["archive_right"]
                        .as_str()
                        .ok_or_else(|| error("Missing native right archive"))?,
                )?,
            ];
            let mut certificates = vec![];
            for (repetition, frame) in block["frames"].as_array().unwrap().iter().enumerate() {
                frames_count += 1;
                let mut pair_dimension = 0.;
                let mut pair_moment = 0.;
                let mut pair_exact = 0.;
                let mut sides = vec![];
                let mut kinetic_pairs = BTreeMap::<&str, (f64, f64)>::new();
                for side in 0..2 {
                    let x = points(&frame["before"]["positions"][side])?;
                    let alive: Vec<bool> =
                        serde_json::from_value(frame["before"]["alive"][side].clone())
                            .map_err(|e| error(e.to_string()))?;
                    if x.iter()
                        .zip(&alive)
                        .any(|(x, a)| *a && x.iter().any(|z| z.abs() > 2.))
                    {
                        return Err(error("Alive source outside actual recorded terminal box"));
                    }
                    let accepted =
                        points(&frame["conditional_moments"][side]["accepted_edge_probabilities"])?;
                    let certificate = occupation_certificate(&x, &alive, &accepted, jitter)?;
                    let native_variance = number(
                        &frame["conditional_moments"][side],
                        "expected_position_variance",
                    )?;
                    retain_worst(
                        &mut exact,
                        "moment-identity",
                        (certificate.expected_variance - native_variance).abs(),
                        2e-10,
                    );
                    retain_worst(
                        &mut exact,
                        "occupation-bound",
                        certificate.expected_variance,
                        certificate.second_moment_upper,
                    );
                    retain_worst(
                        &mut exact,
                        "dimension-bound",
                        certificate.expected_variance,
                        certificate.dimension_upper,
                    );
                    let m2 = certificate
                        .normalized_occupation
                        .iter()
                        .zip(&x)
                        .map(|(q, x)| q * norm2(x))
                        .sum::<f64>();
                    source_moment_max = source_moment_max.max(m2);
                    let budget = propagated_moment_budget(
                        &p,
                        m2,
                        certificate.mean_acceptance,
                        jitter,
                        vc,
                        cap,
                        h,
                        gamma,
                        q2,
                        s,
                        0.,
                    )?;
                    let step = native[side]
                        .steps
                        .get(repetition)
                        .ok_or_else(|| error("Native frame alignment mismatch"))?;
                    for (label, stage_name, field, bound) in [
                        (
                            "prepared-position-moment",
                            "B1_input",
                            "positions",
                            budget.prepared_position_second_moment,
                        ),
                        (
                            "ou-position-moment",
                            "A2",
                            "positions",
                            budget.ou_position_second_moment,
                        ),
                        (
                            "ou-velocity-moment",
                            "O",
                            "velocities",
                            budget.ou_velocity_second_moment,
                        ),
                    ] {
                        let stage = step
                            .stages
                            .iter()
                            .find(|s| s.stage == stage_name)
                            .ok_or_else(|| {
                                error(format!("Missing native moment stage {stage_name}"))
                            })?;
                        let pts = stage_points(stage, field)?;
                        let observed = pts.iter().map(|x| norm2(x)).sum::<f64>() / pts.len() as f64;
                        let pair = kinetic_pairs.entry(label).or_default();
                        pair.0 += observed;
                        pair.1 += bound;
                    }
                    let field = step
                        .final_population
                        .observations
                        .fields
                        .get("positions")
                        .ok_or_else(|| error("Missing native final positions"))?;
                    let observed = field.values().iter().map(|x| x * x).sum::<f64>() / n as f64;
                    let pair = kinetic_pairs.entry("final-position-moment").or_default();
                    pair.0 += observed;
                    pair.1 += budget.final_position_second_moment;
                    let mut diagnostics = vec![];
                    for (j, point) in x.iter().enumerate().filter(|(j, _)| alive[*j]) {
                        let force: Vec<_> = point
                            .iter()
                            .map(|z| {
                                -p.quadratic_curvature * z
                                    - p.cosine_amplitude * p.frequency * (p.frequency * z).sin()
                            })
                            .collect();
                        let radial = (point.iter().zip(&force).map(|(x, f)| x * f).sum::<f64>()
                            + declared.restoring_k * norm2(point))
                        .max(0.);
                        diagnostics.push(json!({"source":j,"force_norm":norm2(&force).sqrt(),"radial_defect":radial}));
                        retain_worst(
                            &mut exact,
                            "regional-radial",
                            radial,
                            declared.radial_defect_upper,
                        );
                    }
                    pair_dimension += certificate.dimension_upper;
                    pair_moment += certificate.second_moment_upper;
                    pair_exact += certificate.expected_variance;
                    sides.push(json!({"occupation":certificate,"uncentered_source_second_moment":m2,"propagated_moments":budget,"sampled_diagnostics":diagnostics}));
                }
                for (name, values) in kinetic_pairs {
                    samples.entry(name).or_default().push(values);
                }
                let live_points: Vec<Vec<Vec<f64>>> = (0..2)
                    .map(|side| {
                        let x = points(&frame["before"]["positions"][side]).unwrap();
                        x.into_iter()
                            .enumerate()
                            .filter_map(|(i, p)| {
                                frame["before"]["alive"][side][i]
                                    .as_bool()
                                    .unwrap()
                                    .then_some(p)
                            })
                            .collect()
                    })
                    .collect();
                let m2: Vec<_> = live_points
                    .iter()
                    .map(|x| x.iter().map(|x| norm2(x)).sum::<f64>() / x.len() as f64)
                    .collect();
                let bary: Vec<_> = live_points.iter().map(|x| mean(x)).collect();
                let coupling_cost = (m2[0] + m2[1]
                    - 2. * bary[0]
                        .iter()
                        .zip(&bary[1])
                        .map(|(x, y)| x * y)
                        .sum::<f64>())
                .max(0.);
                let rewards: Vec<_> = live_points
                    .iter()
                    .map(|x| x.iter().map(|x| p.value(x).unwrap()).sum::<f64>() / x.len() as f64)
                    .collect();
                let reward_difference = (rewards[0] - rewards[1]).abs();
                retain_worst(
                    &mut exact,
                    "reward-moment",
                    reward_difference,
                    p.reward_mean_bound(m2[0], m2[1], coupling_cost)?,
                );
                retain_worst(
                    &mut exact,
                    "reward-box",
                    reward_difference,
                    p.reward_lipschitz_on_box(2.)? * coupling_cost.sqrt(),
                );
                let coordinate_cost: Vec<_> = (0..d)
                    .map(|a| {
                        let m0 = live_points[0].iter().map(|x| x[a] * x[a]).sum::<f64>()
                            / live_points[0].len() as f64;
                        let m1 = live_points[1].iter().map(|x| x[a] * x[a]).sum::<f64>()
                            / live_points[1].len() as f64;
                        (m0 + m1 - 2. * bary[0][a] * bary[1][a]).max(0.)
                    })
                    .collect();
                retain_worst(
                    &mut exact,
                    "reward-separated",
                    reward_difference,
                    p.reward_separated_moment_bound(m2[0], m2[1], &coordinate_cost)?,
                );
                let proxy = number(frame, "positional_proxy")?;
                let centered = number(frame, "centered_positional_transport")?;
                samples
                    .entry("dimension-reset")
                    .or_default()
                    .push((proxy, pair_dimension));
                samples
                    .entry("occupation-reset")
                    .or_default()
                    .push((proxy, pair_moment));
                samples
                    .entry("exact-conditional-variance")
                    .or_default()
                    .push((proxy, pair_exact));
                samples
                    .entry("centered-dimension-reset")
                    .or_default()
                    .push((centered, pair_dimension));
                old_reset += number(frame, "C_reset")?;
                sharp_reset += pair_dimension;
                sharp_occupation += pair_moment;
                certificates.push(json!({"sides":sides,"native_step_in_source_chunk":repetition,"sampled_proxy":proxy,"sampled_centered_transport":centered,"dimension_bound":pair_dimension,"occupation_bound":pair_moment,"exact_conditional_variance":pair_exact}));
            }
            outputs.push(store.save_json(
                &format!("chapter04-case{id}-tightening-{}", outputs.len()),
                &json!({"source_archive":filename,"native_archives":[block["archive_left"],block["archive_right"]],"certificates":certificates}),
            )?);
        }
        for (name, values) in &samples {
            comparisons.push(empirical_check(
                &format!("case{id}:{name}"),
                values,
                if name.contains("dimension") {
                    "lem-w2-dimension-aware-cloning-reset"
                } else if name.contains("moment") {
                    "lem-w2-propagated-moment-budget"
                } else {
                    "lem-w2-normalized-occupation-reset"
                },
            ));
        }
        let reward_bounds: Value = exact
            .iter()
            .filter(|(name, _)| name.starts_with("reward-"))
            .map(|(name, (observed, bound, _))| {
                (name.to_string(), json!({"observed":observed,"bound":bound}))
            })
            .collect();
        for (name, (observed, bound, predicates)) in exact {
            let label = if name == "dimension-bound" {
                "lem-w2-dimension-aware-cloning-reset"
            } else if name.starts_with("reward-") {
                "lem-w2-curvature-reward-moment"
            } else if name == "regional-radial" {
                "def-slc-profiles"
            } else {
                "lem-w2-normalized-occupation-reset"
            };
            let mut c = check(
                &format!("case{id}:{name}"),
                observed,
                bound,
                label,
                "Largest individual signed residual retained; every raw input checked, full conditional fitness and independent copy/noise laws; exact affine coordinate-span certificate, revived outputs, dead storage excluded. Regional bounds are analytic, never inferred from sampled maxima.",
            );
            c["individual_predicates"] = json!(predicates);
            comparisons.push(c);
        }
        let count = samples["dimension-reset"].len() as f64;
        cases.push(json!({"case":id,"N":n,"dimension":d,"benchmark":case["benchmark"],"profile":case["profile"],"old_reset":old_reset/count,"sharp_dimension_reset":sharp_reset/count,"sharp_occupation_reset":sharp_occupation/count,"reward_bounds":reward_bounds,"source_second_moment_max":source_moment_max,"actual_terminal_box_uniform_budget":terminal_box_uniform_budget,"uniform_budget_hypothesis":"Actual native terminal absorbing box [-2,2]^d and complete live-source revival imply normalized source second moment <=4d; no prepared Gaussian or exterior truncation. This specialization does not replace unbounded confinement in another kernel.","regional_profiles":{"declared_domain":declared,"central_well":core,"barrier":barrier},"certificates":outputs,"scope":"Conditional N-independent normalized occupation certificates and actual terminal-box law moment envelope. For an unbounded-source kernel, a uniform source moment must instead be propagated by confinement/Safe Harbor; finite sampled maxima are diagnostics only."}));
    }
    for reference in source["references"].as_array().unwrap() {
        let row = reference["row_normalized"].as_bool().unwrap();
        let prim = &reference["primitive"];
        let cfg: ViscousForceConfig =
            serde_json::from_value(reference["native_config"]["qft"]["viscosity"].clone())
                .map_err(|e| error(e.to_string()))?;
        let d = reference["d"].as_u64().unwrap() as usize;
        let n = reference["N"].as_u64().unwrap() as usize;
        let vc = number(prim, "V_c")?;
        let cx = if row {
            row_uniform_derivative(
                n,
                cfg.coefficient,
                vc,
                number(prim, "R_D")? + number(prim, "J")?,
                cfg.bandwidth,
            )?
        } else {
            count_hilbert_derivative(n, cfg.coefficient, vc, cfg.bandwidth)?
        };
        let beta = number(prim, "t")?.powi(2) * (number(prim, "L_F")? + cx);
        let register = row_continuity_budget(
            d,
            number(prim, "R_D")?,
            number(prim, "sigma_J")?,
            vc,
            number(prim, "h")?,
            number(prim, "gamma")?,
            number(prim, "q_squared")?,
            cfg.bandwidth,
        )?;
        let mut outputs = vec![];
        let mut native_residual = 0_f64;
        let mut fd_residual = 0_f64;
        let mut max_ratio = 0_f64;
        for file in reference["raw_archives"].as_array().unwrap() {
            let filename = file["archive"].as_str().unwrap();
            let archive = input.load_archive(filename)?;
            let mut probes = vec![];
            for step in &archive.steps {
                for name in ["B1", "B2"] {
                    let stage = step
                        .stages
                        .iter()
                        .find(|s| s.stage == format!("{name}_input"))
                        .ok_or_else(|| error("Missing derivative input stage"))?;
                    if stage.validity.iter().any(|s| s.terminated || s.truncated) {
                        return Err(error(
                            "Dense derivative specialization requires all-revived force stage",
                        ));
                    }
                    let x = stage_points(stage, "positions")?;
                    let v = stage_points(stage, "velocities")?;
                    let direction: Vec<Vec<f64>> = (0..n)
                        .map(|i| {
                            (0..d)
                                .map(|a| {
                                    ((i as f64 * 2_f64.sqrt() + a as f64).sin()) / (d as f64).sqrt()
                                })
                                .collect()
                        })
                        .collect();
                    let cert = viscous_certificate(&x, &v, &direction, &cfg)?;
                    let actual = step
                        .field_evaluations
                        .iter()
                        .find(|f| f.stage == name && f.field == "viscous_force")
                        .ok_or_else(|| error("Missing archived native force"))?;
                    let force_residual = cert
                        .force
                        .concat()
                        .iter()
                        .zip(&actual.values)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0., f64::max);
                    native_residual = native_residual.max(force_residual);
                    let epsilon = 1e-6;
                    let shifted = |sign: f64| {
                        x.iter()
                            .zip(&direction)
                            .map(|(x, h)| {
                                x.iter()
                                    .zip(h)
                                    .map(|(x, h)| x + sign * epsilon * h)
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    };
                    let plus = viscous_certificate(&shifted(1.), &v, &direction, &cfg)?.force;
                    let minus = viscous_certificate(&shifted(-1.), &v, &direction, &cfg)?.force;
                    let finite_difference: Vec<Vec<f64>> = plus
                        .iter()
                        .zip(&minus)
                        .map(|(a, b)| {
                            a.iter()
                                .zip(b)
                                .map(|(a, b)| (a - b) / (2. * epsilon))
                                .collect()
                        })
                        .collect();
                    let residual = cert
                        .jacobian_action
                        .concat()
                        .iter()
                        .zip(finite_difference.concat())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0., f64::max);
                    fd_residual = fd_residual.max(residual);
                    let bound = cert.max_row_moment_derivative * cert.direction_max_row_norm;
                    max_ratio = max_ratio.max(if bound > 0. {
                        cert.action_max_row_norm / bound
                    } else {
                        0.
                    });
                    comparisons.push(check(&format!("reference-{row}:moment-action:{}:{name}",step.report.step),cert.action_max_row_norm,bound,"lem-w2-viscous-moment-derivative","Exact directional Jacobian and normalized donor covariance at retained physical force stage. Conditional, no uniform source moment inferred."));
                    comparisons.push(check(&format!("reference-{row}:finite-difference:{}:{name}",step.report.step),residual,2e-6,"lem-w2-viscous-moment-derivative","Central difference of the actual analytic native provider at epsilon=1e-6; numerical derivative diagnostic."));
                    if !row && name == "B1" {
                        let vmax = v.iter().map(|p| norm2(p).sqrt()).fold(0., f64::max);
                        comparisons.push(check(&format!("reference-count:velocity-hypothesis:{}",step.report.step),vmax,vc,"lem-w2-viscous-moment-derivative","Analytic cap plus collision velocity bound at B1; never applied to unbounded post-OU B2 velocities."));
                        comparisons.push(check(&format!("reference-count:hilbert-action:{}",step.report.step),cert.action_hilbert_norm,cx*cert.direction_hilbert_norm,"lem-w2-viscous-moment-derivative","Normalized population Hilbert norm; coefficient independent of N and d. This coefficient is not a maximum-row norm bound."));
                    }
                    probes.push(json!({"step":step.report.step,"stage":name,"direction":direction,"certificate":cert,"native_force_residual":force_residual,"finite_difference_action":finite_difference,"epsilon":epsilon,"finite_difference_residual":residual}));
                }
            }
            outputs.push(store.save_json(
                &format!("chapter04-reference-{row}-derivatives-{}", outputs.len()),
                &json!({"source_archive":filename,"probes":probes}),
            )?);
        }
        comparisons.push(check(&format!("reference-{row}:native-force"),native_residual,2e-9,"def-w2-finite-population-qsd-regime","Every archived force stage checked separately; continuous trajectory; no SEM averaging of violations."));
        references.push(json!({"row_normalized":row,"dimension":d,"N":n,"old_beta":prim["beta_F"],"sharp_beta":beta,"norm":if row{"maximum row"}else{"normalized population Hilbert"},"primary_population_uniform":!row,"auxiliary_scope":if row{"Same survival-event radius as original source; J(log N) dependence retained. Not a primary N-uniform bound."}else{"Bounded B1 collision velocities, no position/noise cutoff"},"row_continuity_register":register,"maximum_native_force_residual":native_residual,"maximum_finite_difference_residual":fd_residual,"maximum_moment_action_ratio":max_ratio,"derivative_archives":outputs}));
    }
    let failed = comparisons.iter().filter(|c| c["passed"] == false).count();
    let report = json!({"chapter":4,"title":"Dimension and structural-landscape tightening","schema_version":1,"source_dataset":dataset,"source_report_sha256":sha256_file(&source_path)?,"source_formula_coverage":"No global source-expression coverage credited by these local specializations","cases":cases,"references":references,"comparisons":comparisons,"summary":{"cases":cases.len(),"operator_frames":frames_count,"comparisons":comparisons.len(),"comparisons_failed":failed,"new_native_updates":0},"gaps":["Global N-uniform source occupation moment propagation must be supplied by a confinement/Safe Harbor certificate","Regional profiles do not certify transition/residence probabilities, donor pressure or a global QSD spectral rate","Preset row attraction small-viscosity endpoints remain unverified; sharpened RWM register is a population-law continuity bound"]});
    let saved = store.save_json("chapter04-tightening-report", &report)?;
    store.finish("complete")?;
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| error(e.to_string()))?,
    )
    .map_err(|e| error(e.to_string()))?;
    eprintln!(
        "Chapter 4: {} comparisons, {} failures; {}",
        comparisons.len(),
        failed,
        saved
    );
    Ok(report)
}
