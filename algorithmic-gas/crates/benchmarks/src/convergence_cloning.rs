//! Source-mapped diagnostics for chapter 03 of the convergence program.
//!
//! Conditional moments integrate *fresh* independent cloning companions, gates,
//! and centered isotropic jitter while retaining the actual sampled fitness.
//! A realized variance need not decrease and is never compared with an expected
//! drift as a pathwise assertion. The terminal boundary and kinetic stage are
//! excluded from the cloning observable.
use algorithmic_gas::{
    GasError, ObservationBatch, Result, RunArchive, TensorBatch,
    cloning::CloneDecision,
    donor::{DonorModule, SamplingLaw},
    fitness::{PositiveMap, Standardizer},
    geometry::{AlgorithmicDistance, InteractionKernel},
    noise::{FactorValues, NoiseGeometry},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CloningCheck {
    pub id: String,
    pub source_labels: Vec<String>,
    pub kind: String,
    pub scope: String,
    pub observed: f64,
    pub expected_or_bound: f64,
    pub residual: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CloningMomentReport {
    pub rows: usize,
    pub dimension: usize,
    pub alive: usize,
    pub acceptance_probabilities: Vec<f64>,
    pub accepted_edge_probabilities: Vec<Vec<f64>>,
    pub output_means: Vec<Vec<f64>>,
    pub covariance_traces: Vec<f64>,
    /// N-normalized variance of the entering alive rows, centered on their own mean.
    pub entering_alive_variance: f64,
    pub expected_position_variance: f64,
    pub expected_position_drift: f64,
    pub conditional_barycenter_variance: f64,
    /// Var_F(mean_i m_i), zero before measurement averaging.
    pub measurement_barycenter_variance: f64,
    pub total_barycenter_variance: f64,
    pub jitter_variance_contribution: f64,
    pub collective_flux: Option<f64>,
    pub collective_flux_identity_residual: Option<f64>,
    pub checks: Vec<CloningCheck>,
    pub scope: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CloningStepReport {
    pub step: u64,
    pub checks: Vec<CloningCheck>,
    pub moments: Option<CloningMomentReport>,
    pub realized_position_variance: f64,
    pub realized_velocity_variance: Option<f64>,
    pub fitness_lower_bound: Option<f64>,
    pub fitness_upper_bound: Option<f64>,
    pub constants: CanonicalCloningConstants,
    pub unavailable: Vec<String>,
}

/// Canonical (fixed-parameter) constants from (3.B1)-(3.B6).
/// These are reference values, not automatically applicable to a parameter sweep.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanonicalCloningConstants {
    pub d_m: f64,
    pub epsilon_s: f64,
    pub kappa_c: f64,
    pub c: f64,
    pub fitness_min: f64,
    pub fitness_max: f64,
    pub l_0: f64,
    pub l_a: f64,
    pub b_0: f64,
    pub balanced_two_cluster_chi_0: f64,
    pub scope: String,
}

/// Logarithms retain the proved Keystone rates and cover sizes when their
/// numerical magnitude is outside f64. `values` omits unrepresentable positive
/// constants; a zero is never substituted for a strictly positive proof rate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeystoneConstantsReport {
    pub dimensions: usize,
    pub error_threshold: f64,
    pub reward_lipschitz: f64,
    pub values: BTreeMap<String, f64>,
    pub natural_logs: BTreeMap<String, f64>,
    pub scope: String,
}

/// Discharge (3.CC1)-(3.CC14) for the canonical parameter configuration and a
/// *supplied analytic* joint reward Lipschitz bound on the valid alive region.
/// A maximum gradient on sampled points cannot be used as that bound.
pub fn canonical_keystone_constants(
    dimensions: usize,
    w_0: f64,
    reward_lipschitz: f64,
) -> Result<KeystoneConstantsReport> {
    if !(1..=256).contains(&dimensions)
        || !w_0.is_finite()
        || w_0 <= 0.
        || !reward_lipschitz.is_finite()
        || reward_lipschitz < 0.
        || w_0 > 64. * dimensions as f64
    {
        return Err(error(
            "invalid canonical Keystone threshold or analytic reward Lipschitz bound",
        ));
    }
    let e_max = 64. * dimensions as f64;
    let m_x = (1. + (dimensions as f64).sqrt()).powi(-2);
    let m_z = m_x;
    let d_0 = 32_f64.sqrt();
    let b_f = 2.;
    let d_m = (32_f64 + 1e-6).sqrt() - 0.001;
    let s_star = (d_m * d_m / 4. + 0.01).sqrt();
    let z_star = d_m / 0.1;
    let l_a = 0.5 * reward_lipschitz / (0.1 * m_z);
    let v_0 = m_x * m_x * w_0 / 4.;
    if v_0 >= 32. {
        return Err(error("Keystone threshold needs m_x^2 W_0/4 < D_0^2"));
    }
    let h_f = (v_0 / 2.).sqrt();
    let rho_f = (v_0 / 2.) / (32. - v_0 / 2.);
    // Rationalization keeps the raw separation gap accurate at tiny W_0.
    let delta_f =
        (3. * h_f * h_f / 4.) / ((h_f * h_f + 1e-6).sqrt() + (h_f * h_f / 4. + 1e-6).sqrt());
    let t_f = delta_f / s_star;
    if !t_f.is_finite() || t_f <= 0. {
        return Err(error(
            "Keystone finite-increment gap underflowed; choose representable W_0",
        ));
    }
    // The logistic increment is symmetric and unimodal on [-Z,Z-t], so its
    // exact minimum is at either endpoint. The additive floor cancels.
    let log_omega = 2_f64.ln() - z_star + t_f.exp_m1().ln()
        - (-z_star).exp().ln_1p()
        - (-z_star + t_f).exp().ln_1p();
    let log_omega_derivative = 2_f64.ln() - z_star + t_f.ln() - 2. * (-z_star).exp().ln_1p();
    let log_r = (h_f / 2.).ln().min(if l_a > 0. {
        0.1_f64.ln() + log_omega - (2. * 2.1 * l_a).ln()
    } else {
        f64::INFINITY
    });
    let log_gamma = 0.05_f64.ln() + log_omega;
    let log_a_0 = (log_gamma - (4.41_f64 + 1e-6).ln()).min(0.);
    let log_c_0 = -12. + rho_f.ln() + log_a_0;
    let log_side = (2. * b_f * (2. * dimensions as f64).sqrt()).ln() - log_r;
    let log_m_r = 2.
        * dimensions as f64
        * if log_side < 700. {
            log_side.exp().ceil().ln()
        } else {
            log_side
        };
    let log_chi_0 = log_c_0 + 2. * w_0.ln() - 4_f64.ln() - 2. * e_max.ln() - 2. * log_m_r;
    let log_chi_star = log_chi_0 + 2_f64.ln();
    let log_b_star = log_c_0 + 2. * e_max.ln() - w_0.ln();
    let log_n_0 = 2_f64.ln() + 2. * e_max.ln() + log_m_r - 2. * w_0.ln();
    let mut values = BTreeMap::from([
        ("B_x".into(), 2. * (dimensions as f64).sqrt()),
        ("B_v".into(), 2.),
        ("D_0".into(), d_0),
        ("B_f".into(), b_f),
        ("m_x".into(), m_x),
        ("m_z".into(), m_z),
        ("kappa_D".into(), (-4_f64).exp()),
        ("kappa_C".into(), (-4_f64).exp()),
        ("D_m_span".into(), d_m),
        ("s_star".into(), s_star),
        ("Z_star".into(), z_star),
        ("A_minus".into(), 0.1),
        ("f_plus".into(), 2.1),
        ("F_star_upper".into(), 4.41),
        ("L_H".into(), 0.5),
        ("L_A".into(), l_a),
        ("v_0".into(), v_0),
        ("h_f".into(), h_f),
        ("rho_f".into(), rho_f),
        ("Delta_f".into(), delta_f),
        ("t_f".into(), t_f),
        ("E_max".into(), e_max),
    ]);
    let natural_logs: BTreeMap<String, f64> = BTreeMap::from([
        ("omega_f".into(), log_omega),
        (
            "omega_f_derivative_lower_bound".into(),
            log_omega_derivative,
        ),
        ("r".into(), log_r),
        ("gamma_0".into(), log_gamma),
        ("a_0".into(), log_a_0),
        ("C_0".into(), log_c_0),
        ("M_r".into(), log_m_r),
        ("chi_0".into(), log_chi_0),
        ("chi_star".into(), log_chi_star),
        ("B_star".into(), log_b_star),
        ("N_0_unrounded".into(), log_n_0),
    ]);
    for (name, log_value) in &natural_logs {
        let value: f64 = (*log_value).exp();
        if value.is_finite() && value > 0. {
            values.insert(name.clone(), value);
        }
    }
    if let Some(n0) = values.get("N_0_unrounded").copied() {
        values.insert("N_0".into(), n0.ceil());
    }
    Ok(KeystoneConstantsReport { dimensions,error_threshold:w_0,reward_lipschitz,values,natural_logs,
        scope:"Fixed canonical pipeline, squashed independent Gaussian measurement/cloning companions, entering alive x in (-2,2)^d and |v|<=2. The caller supplies an analytic joint reward Lipschitz bound. Constants are independent of N; (3.CC8) has an explicit finite-population correction, (3.CC9)/(3.CC11) require the displayed population threshold. Rates are selection pressure, not a blanket positional or structural contraction.".into() })
}

pub fn canonical_cloning_constants() -> CanonicalCloningConstants {
    let d_m = (32_f64 + 1e-6).sqrt();
    let epsilon_s: f64 = 0.1;
    let kappa_c = (-4_f64).exp();
    let fitness_min: f64 = 0.01;
    let fitness_max: f64 = 4.41;
    let l_0 = 1.05 * (d_m / epsilon_s + 3. * d_m.powi(3) / (2. * epsilon_s.powi(3)));
    let l_a = (1. / (fitness_min + 1e-6)).max((fitness_max + 1e-6) / (fitness_min + 1e-6).powi(2));
    CanonicalCloningConstants {
        d_m,
        epsilon_s,
        kappa_c,
        c: 1. / kappa_c,
        fitness_min,
        fitness_max,
        l_0,
        l_a,
        b_0: 1. + 1. / kappa_c + 2. * l_a * l_0,
        balanced_two_cluster_chi_0: 4. / 9. * canonical_two_cluster_acceptance_floor(0.5),
        scope: "Canonical squashed phase space radii 2, lambda=1, Gaussian width 2, global scales .1, logistic amplitude 2/floor .1, exponents 1, clone epsilon 1e-6/saturation 1. Balanced chi additionally needs equal two-site populations, radius [.5,2), zero velocity and quadratic rewards.".into(),
    }
}

/// The A_0(a) in (3.KB1), (3.KB2), and the two-cluster noise-balance proposition.
pub fn canonical_two_cluster_acceptance_floor(a: f64) -> f64 {
    let ell = 4. * a / (2. + a);
    let delta = (ell * ell + 1e-6).sqrt() - 0.001;
    let s_max = (delta * delta / 4. + 0.01).sqrt();
    (1.1 * (delta / (2. * s_max)).tanh() / (1.21 + 1e-6)).clamp(0., 1.)
}

fn error(message: &str) -> GasError {
    GasError::Configuration(message.into())
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

fn delta(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(a, b)| a - b).collect()
}

fn center(points: &[Vec<f64>]) -> Vec<f64> {
    let mut result = vec![0.; points[0].len()];
    for point in points {
        for (value, x) in result.iter_mut().zip(point) {
            *value += x / points.len() as f64;
        }
    }
    result
}

fn variance(points: &[Vec<f64>]) -> f64 {
    let mean = center(points);
    points.iter().map(|x| norm2(&delta(x, &mean))).sum::<f64>() / points.len() as f64
}

fn equality(id: &str, labels: &[&str], observed: f64, expected: f64) -> CloningCheck {
    let residual = observed - expected;
    CloningCheck {
        id: id.into(),
        source_labels: labels.iter().map(|x| (*x).into()).collect(),
        kind: "exact_identity".into(),
        scope: "Actual frozen cloning proposal before kinetics and terminal boundary".into(),
        observed,
        expected_or_bound: expected,
        residual,
        passed: residual.abs() <= 1e-9 * (1. + observed.abs() + expected.abs()),
    }
}

fn upper(id: &str, labels: &[&str], observed: f64, bound: f64) -> CloningCheck {
    let mut check = equality(id, labels, observed, bound);
    check.kind = "exact_upper_bound".into();
    check.passed = observed <= bound + 1e-9 * (1. + bound.abs());
    check
}

/// Exact first-two-moment integration of independent current-frame donors.
/// `donor_probabilities` includes revival donors, and rows must sum to one.
/// Living singleton rows may use a self donor; acceptance then equals zero.
#[allow(clippy::too_many_arguments)]
pub fn exact_cloning_moments(
    positions: &[Vec<f64>],
    fitness: &[f64],
    donor_probabilities: &[Vec<f64>],
    alive: &[bool],
    decision: &CloneDecision,
    step: u64,
    jitter: f64,
) -> Result<CloningMomentReport> {
    decision.validate()?;
    let n = positions.len();
    let d = positions.first().map_or(0, Vec::len);
    if n == 0
        || d == 0
        || fitness.len() != n
        || alive.len() != n
        || donor_probabilities.len() != n
        || !jitter.is_finite()
        || jitter < 0.
        || positions
            .iter()
            .any(|x| x.len() != d || x.iter().any(|v| !v.is_finite()))
        || donor_probabilities.iter().any(|row| {
            row.len() != n
                || row.iter().any(|p| !p.is_finite() || *p < 0.)
                || (row.iter().sum::<f64>() - 1.).abs() > 1e-10
        })
        || alive
            .iter()
            .enumerate()
            .any(|(i, a)| *a && (!fitness[i].is_finite() || fitness[i] <= 0.))
    {
        return Err(error("invalid independent cloning moment inputs"));
    }
    let k = alive.iter().filter(|&&a| a).count();
    if k == 0 {
        return Err(GasError::Extinction);
    }
    if donor_probabilities
        .iter()
        .any(|row| row.iter().zip(alive).any(|(p, a)| !a && *p > 0.))
    {
        return Err(error("cloning moment donors must be entering alive slots"));
    }
    let mut accepted = vec![vec![0.; n]; n];
    let mut p = vec![0.; n];
    let mut means = vec![vec![0.; d]; n];
    let mut traces = vec![0.; n];
    let mut copy_traces = vec![0.; n];
    // Use donor-centered moments for dead rows; their discarded coordinates may
    // be enormous and must not introduce subtractive cancellation.
    for i in 0..n {
        for j in 0..n {
            accepted[i][j] = donor_probabilities[i][j]
                * if alive[i] && alive[j] {
                    decision.acceptance_probability(step, fitness[i], fitness[j])
                } else if !alive[i] && alive[j] {
                    1.
                } else {
                    0.
                };
            p[i] += accepted[i][j];
        }
        for a in 0..d {
            means[i][a] = (1. - p[i]) * if alive[i] { positions[i][a] } else { 0. }
                + (0..n)
                    .map(|j| accepted[i][j] * positions[j][a])
                    .sum::<f64>();
        }
        copy_traces[i] = (1. - p[i])
            * if alive[i] {
                norm2(&delta(&positions[i], &means[i]))
            } else {
                0.
            }
            + (0..n)
                .map(|j| accepted[i][j] * norm2(&delta(&positions[j], &means[i])))
                .sum::<f64>();
        traces[i] = copy_traces[i] + d as f64 * jitter * jitter * p[i];
    }
    let trace_sum = traces.iter().sum::<f64>();
    let expected = variance(&means) + (1. - 1. / n as f64) * trace_sum / n as f64;
    let eligible: Vec<_> = positions
        .iter()
        .zip(alive)
        .filter_map(|(x, a)| a.then_some(x.clone()))
        .collect();
    let entering = variance(&eligible) * k as f64 / n as f64;
    let diameter2 = eligible
        .iter()
        .flat_map(|a| eligible.iter().map(move |b| norm2(&delta(a, b))))
        .fold(0_f64, f64::max);
    let mean_p = p.iter().sum::<f64>() / n as f64;
    let barycenter_variance = trace_sum / (n * n) as f64;
    let jitter_contribution = (1. - 1. / n as f64) * d as f64 * jitter * jitter * mean_p;
    let mut checks = vec![
        upper(
            "conditional_barycenter_variance_bound",
            &[
                "lem-cloning-individual-centered-displacement",
                "thm-cloning-canonical-barycenter-concentration",
            ],
            barycenter_variance,
            (diameter2 + d as f64 * jitter * jitter) / n as f64,
        ),
        upper(
            "positional_reset_bound",
            &["thm-positional-variance-contraction"],
            expected,
            diameter2 / 2. + (1. - 1. / n as f64) * d as f64 * jitter * jitter,
        ),
        upper(
            "nonnegative_conditional_covariance",
            &["def-cloning-frozen-positional-moments"],
            -traces.iter().copied().fold(f64::INFINITY, f64::min),
            0.,
        ),
        upper(
            "cloning_probability_at_most_one",
            &["def-cloning-probability"],
            p.iter().copied().fold(0_f64, f64::max),
            1.,
        ),
    ];
    let (flux, flux_residual) = if k == n {
        let mean = center(positions);
        let r2: Vec<_> = positions.iter().map(|x| norm2(&delta(x, &mean))).collect();
        let flux = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| accepted[i][j] * (r2[j] - r2[i]))
            .sum::<f64>()
            / n as f64;
        let mean_shift = delta(&center(&means), &mean);
        let drift_flux =
            flux - norm2(&mean_shift) - copy_traces.iter().sum::<f64>() / (n * n) as f64
                + jitter_contribution;
        let residual = expected - entering - drift_flux;
        checks.push(equality(
            "collective_variance_flux_identity",
            &["lem-keystone-contraction-alive"],
            expected - entering,
            drift_flux,
        ));
        for i in 0..n {
            let ti = delta(&means[i], &positions[i]);
            let centered_shift = delta(&ti, &mean_shift);
            let di = delta(&positions[i], &mean);
            let drift = 2.
                * di.iter()
                    .zip(&centered_shift)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                + norm2(&centered_shift)
                + (1. - 2. / n as f64) * traces[i]
                + trace_sum / (n * n) as f64;
            let actual = norm2(&delta(&means[i], &center(&means)))
                + (1. - 2. / n as f64) * traces[i]
                + trace_sum / (n * n) as f64
                - r2[i];
            checks.push(equality(
                &format!("moving_barycenter_row_{i}"),
                &["lem-cloning-individual-centered-displacement"],
                actual,
                drift,
            ));
        }
        (Some(flux), Some(residual))
    } else {
        (None, None)
    };
    Ok(CloningMomentReport {
        rows: n, dimension: d, alive: k, acceptance_probabilities: p,
        accepted_edge_probabilities: accepted, output_means: means, covariance_traces: traces,
        entering_alive_variance: entering, expected_position_variance: expected,
        expected_position_drift: expected - entering,
        conditional_barycenter_variance: barycenter_variance, measurement_barycenter_variance: 0.,
        total_barycenter_variance: barycenter_variance, jitter_variance_contribution: jitter_contribution,
        collective_flux: flux, collective_flux_identity_residual: flux_residual, checks,
        scope: "Conditional on the complete entering state and retained sampled fitness; independent eligible current donors, gates, centered isotropic unit-covariance jitter. All-slot proposal output; alive N-normalized input. Measurement averaging and terminal survival conditioning are separate.".into(),
    })
}

fn current_donor_probabilities(
    module: &DonorModule,
    observations: &ObservationBatch<f64>,
    alive: &[bool],
) -> Result<Vec<Vec<f64>>> {
    if module.law != SamplingLaw::Independent || module.history_window != 0 || module.count != 1 {
        return Err(error(
            "exact moments require independent count-one current donors",
        ));
    }
    let n = alive.len();
    let kind = <algorithmic_gas::geometry::Distance as AlgorithmicDistance<f64>>::comparison_kind(
        &module.distance,
    );
    let mut probabilities = vec![vec![0.; n]; n];
    for i in 0..n {
        let eligible: Vec<_> = (0..n)
            .filter(|&j| alive[j] && (module.allow_self || i != j))
            .collect();
        let eligible = if eligible.is_empty() && alive[i] {
            vec![i]
        } else {
            eligible
        };
        if eligible.is_empty() {
            return Err(GasError::Extinction);
        }
        let logs: Vec<_> = eligible
            .iter()
            .map(|&j| {
                module.kernel.log_weight(
                    module.distance.compare(observations, i, observations, j)?,
                    kind,
                )
            })
            .collect::<Result<_>>()?;
        let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !maximum.is_finite() {
            return Err(error("nonfinite donor log weights"));
        }
        let denominator = logs.iter().map(|x| (x - maximum).exp()).sum::<f64>();
        for (&j, log) in eligible.iter().zip(logs) {
            probabilities[i][j] = (log - maximum).exp() / denominator;
        }
    }
    Ok(probabilities)
}

fn map_bounds(map: &PositiveMap, exponent: f64) -> Option<(f64, f64)> {
    if exponent == 0. {
        return Some((1., 1.));
    }
    match map {
        PositiveMap::Logistic { amplitude, floor } => {
            Some((floor.powf(exponent), (floor + amplitude).powf(exponent)))
        }
        _ => None,
    }
}

/// Diagnose actual recorded fitness and clone decisions, and integrate the
/// conditional position law where its independent-current-donor hypotheses hold.
pub fn analyze_cloning_step(
    archive: &RunArchive<f64>,
    step_index: usize,
) -> Result<CloningStepReport> {
    let step = archive
        .steps
        .get(step_index)
        .ok_or_else(|| error("missing recorded cloning step"))?;
    let report = &step.report;
    let config = &archive.gas_config;
    let alive = &report.pre_clone_eligible;
    let fitness = &report.pre_clone_fitness;
    let mut observations = step.before.observations.clone();
    if let Some(stage) = step.stages.iter().find(|s| s.stage == "pre_clone") {
        for (name, field) in &stage.fields {
            observations.fields.insert(
                name.clone(),
                TensorBatch::new(alive.len(), field.item_shape.clone(), field.values.clone())?,
            );
        }
    }
    let x = observations.field("positions")?;
    let points: Vec<_> = x.values().chunks(x.width()).map(<[f64]>::to_vec).collect();
    let mut checks = vec![];
    let mut unavailable = vec![];
    for (channel, values, recorded_z, recorded_stats, standardizer) in [
        (
            "reward",
            &fitness.oriented_reward,
            &fitness.reward_z,
            &fitness.reward_stats,
            &config.fitness.reward_standardizer,
        ),
        (
            "diversity",
            &fitness.separation,
            &fitness.diversity_z,
            &fitness.diversity_stats,
            &config.fitness.diversity_standardizer,
        ),
    ] {
        let (z, stats) = standardizer.apply(values, alive, &observations)?;
        let max_residual = z
            .iter()
            .zip(recorded_z)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f64, f64::max);
        checks.push(equality(
            &format!("{channel}_standardization"),
            &["def-standardization-operator"],
            max_residual,
            0.,
        ));
        checks.push(equality(
            &format!("{channel}_regularized_scale"),
            &["def-patched-std-dev-function"],
            stats
                .scale
                .iter()
                .zip(&recorded_stats.scale)
                .map(|(a, b)| (a - b).abs())
                .fold(0_f64, f64::max),
            0.,
        ));
        if let Standardizer::Global { sigma_min } = standardizer {
            let active: Vec<_> = values
                .iter()
                .zip(alive)
                .filter_map(|(v, a)| a.then_some(*v))
                .collect();
            let range = active.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - active.iter().copied().fold(f64::INFINITY, f64::min);
            checks.push(upper(
                &format!("{channel}_z_score_support"),
                &["lem-compact-support-z-scores"],
                z.iter().map(|z| z.abs()).fold(0_f64, f64::max),
                range / sigma_min,
            ));
            checks.push(upper(
                &format!("{channel}_scale_lower_bound"),
                &["lem-patching-properties"],
                *sigma_min,
                stats.scale[0],
            ));
            checks.push(upper(
                &format!("{channel}_scale_upper_bound"),
                &["def-max-patched-std"],
                stats.scale[0],
                (range * range / 4. + sigma_min * sigma_min).sqrt(),
            ));
        } else {
            unavailable.push(format!("{channel}: global compact-support/Popoviciu proof branch not applicable to this standardizer"));
        }
    }
    let recomputed = config
        .fitness
        .combine(&fitness.reward_z, &fitness.diversity_z, alive)?;
    checks.push(equality(
        "fitness_product",
        &["def-fitness-potential-operator"],
        recomputed
            .iter()
            .zip(&fitness.fitness)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f64, f64::max),
        0.,
    ));
    let bounds = map_bounds(&config.fitness.reward_map, config.fitness.reward_exponent)
        .zip(map_bounds(
            &config.fitness.diversity_map,
            config.fitness.diversity_exponent,
        ))
        .map(|((a, b), (c, d))| (a * c, b * d));
    if let Some((lower, upper_bound)) = bounds {
        let active: Vec<_> = fitness
            .fitness
            .iter()
            .zip(alive)
            .filter_map(|(v, a)| a.then_some(*v))
            .collect();
        checks.push(upper(
            "fitness_lower_bound",
            &["lem-potential-bounds"],
            lower,
            active.iter().copied().fold(f64::INFINITY, f64::min),
        ));
        checks.push(upper(
            "fitness_upper_bound",
            &["lem-potential-bounds"],
            active.iter().copied().fold(0_f64, f64::max),
            upper_bound,
        ));
    } else {
        unavailable.push("Global bounded-fitness theorem requires bounded logistic maps; legacy asymmetric map does not supply its upper constant".into());
    }
    for (i, choice) in report.clone_plan.choices.iter().enumerate() {
        let expected = if !alive[i] {
            1.
        } else if let Some(donor) = choice.donors.first() {
            config.clone_decision.acceptance_probability(
                report.step,
                fitness.fitness[i],
                step.donor_fitness[donor.pool_index as usize],
            )
        } else {
            0.
        };
        if let Some(probability) = choice.probability {
            let labels = if alive[i] {
                vec!["def-cloning-score", "def-cloning-probability"]
            } else {
                vec!["def-cloning-probability", "lem-dead-walker-clone-prob"]
            };
            checks.push(equality(
                &format!("acceptance_probability_row_{i}"),
                &labels,
                probability,
                expected,
            ));
        } else {
            unavailable.push(format!(
                "row {i}: custom decision did not record its actual probability"
            ));
        }
        if !alive[i] {
            let has_live_current_donor = choice.donors.first().is_some_and(|donor| {
                report
                    .clone_plan
                    .sources
                    .get(donor.pool_index as usize)
                    .is_some_and(|source| {
                        source.frame == report.step - 1
                            && alive.get(source.slot as usize) == Some(&true)
                    })
            });
            checks.push(equality(
                &format!("mandatory_revival_row_{i}"),
                &["lem-dead-walker-clone-prob", "lem-eg-scheduled-revival"],
                if choice.accepted && choice.revival && has_live_current_donor {
                    1.
                } else {
                    0.
                },
                1.,
            ));
        }
    }
    let jitter = match &config.clone_transform.jitter {
        None => Some(0.),
        Some(noise) => match &noise.geometry {
            NoiseGeometry::Isotropic {
                scale: FactorValues::Constant { values },
            } if values.len() == 1 => {
                Some(config.clone_transform.jitter_amplitude * values[0].abs())
            }
            _ => None,
        },
    };
    let moments = if let Some(jitter) = jitter {
        match current_donor_probabilities(&config.cloning_donors, &observations, alive) {
            Ok(mut donors) => {
                if !config.clone_decision.revival_from_companion {
                    let k = alive.iter().filter(|&&a| a).count();
                    for (i, row) in donors.iter_mut().enumerate() {
                        if !alive[i] {
                            for (p, a) in row.iter_mut().zip(alive) {
                                *p = if *a { 1. / k as f64 } else { 0. };
                            }
                        }
                    }
                }
                let moments = exact_cloning_moments(
                    &points,
                    &fitness.fitness,
                    &donors,
                    alive,
                    &config.clone_decision,
                    report.step,
                    jitter,
                )?;
                checks.extend(moments.checks.clone());
                Some(moments)
            }
            Err(e) => {
                unavailable.push(format!("Conditional independent position moments: {e}"));
                None
            }
        }
    } else {
        unavailable.push("Position moments require constant isotropic jitter; no isotropic surrogate substituted for configured covariance".into());
        None
    };
    let output = step
        .stages
        .iter()
        .find(|s| s.stage == "post_transform")
        .ok_or_else(|| error("cloning analysis requires post_transform stage recording"))?;
    checks.push(equality(
        "post_cloning_all_slots_alive",
        &["lem-eg-scheduled-revival", "def-eg-component-collision"],
        output
            .validity
            .iter()
            .filter(|mark| mark.eligible(false))
            .count() as f64,
        alive.len() as f64,
    ));
    let ox = output
        .fields
        .get("positions")
        .ok_or_else(|| error("missing post-transform positions"))?;
    let output_points: Vec<_> = ox.values.chunks(x.width()).map(<[f64]>::to_vec).collect();
    let realized_position_variance = variance(&output_points);
    let mut realized_velocity_variance = None;
    if let (Some(input), Some(output_v)) = (
        observations.fields.get("velocities"),
        output.fields.get("velocities"),
    ) {
        let before: Vec<_> = input
            .values()
            .chunks(input.width())
            .map(<[f64]>::to_vec)
            .collect();
        let after: Vec<_> = output_v
            .values
            .chunks(input.width())
            .map(<[f64]>::to_vec)
            .collect();
        realized_velocity_variance = Some(variance(&after));
        if let Some(alpha) = config.clone_transform.restitution {
            checks.push(equality(
                "full_slot_velocity_momentum",
                &[
                    "prop-cloning-component-conservation",
                    "thm-eg-component-balances",
                    "thm-cloning-canonical-barycenter-concentration",
                ],
                norm2(&delta(&center(&before), &center(&after))),
                0.,
            ));
            checks.push(upper(
                "full_slot_velocity_energy_nonincrease",
                &[
                    "prop-cloning-component-conservation",
                    "prop-bounded-velocity-expansion",
                ],
                after.iter().map(|v| norm2(v)).sum(),
                before.iter().map(|v| norm2(v)).sum(),
            ));
            for energy in step
                .field_evaluations
                .iter()
                .filter(|f| f.field == "collision_relative_energy_before")
            {
                if let Some(after_energy) = step.field_evaluations.iter().find(|f| {
                    f.field == "collision_relative_energy_after" && f.stage == energy.stage
                }) {
                    let residual = energy
                        .values
                        .iter()
                        .zip(&after_energy.values)
                        .zip(&energy.available)
                        .filter(|(_, a)| **a)
                        .map(|((a, b), _)| (b - alpha * alpha * a).abs())
                        .fold(0_f64, f64::max);
                    checks.push(equality(
                        "component_energy_restitution",
                        &[
                            "prop-cloning-component-conservation",
                            "thm-eg-component-balances",
                        ],
                        residual,
                        0.,
                    ));
                }
            }
        }
    }
    Ok(CloningStepReport {
        step: report.step,
        checks,
        moments,
        realized_position_variance,
        realized_velocity_variance,
        fitness_lower_bound: bounds.map(|x| x.0),
        fitness_upper_bound: bounds.map(|x| x.1),
        constants: canonical_cloning_constants(),
        unavailable,
    })
}

/// Full measurement-law integration of the chapter's four-walker spreading
/// example. Uses the real standardizer, fitness map, geometry and gate law.
/// There are only eight distinct distance patterns, including all tie outcomes.
pub fn exact_four_walker_fixture(jitter: f64) -> Result<CloningMomentReport> {
    let config = algorithmic_gas::GasConfig::euclidean(1, 0.04)?;
    let positions = vec![vec![0.], vec![0.], vec![0.], vec![0.1]];
    let mut observations =
        ObservationBatch::positions(TensorBatch::vectors(4, 1, vec![0., 0., 0., 0.1])?);
    observations.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(4, 1, vec![0.; 4])?,
    );
    let alive = vec![true; 4];
    let donors = current_donor_probabilities(&config.cloning_donors, &observations, &alive)?;
    let measurement = current_donor_probabilities(&config.distance_donors, &observations, &alive)?;
    let distance = config
        .distance_donors
        .distance
        .compare(&observations, 0, &observations, 3)?;
    let rewards = algorithmic_gas::RewardBatch::new(vec![0., 0., 0., 0.005], Default::default());
    let mut aggregate: Option<CloningMomentReport> = None;
    let mut center_second = 0.;
    let mut center_mean = 0.;
    for mask in 0..8 {
        let mut distances = vec![0.; 4];
        let mut mass = 1.;
        for (i, value) in distances.iter_mut().enumerate().take(3) {
            if mask & (1 << i) == 0 {
                mass *= 1. - measurement[i][3];
            } else {
                *value = distance;
                mass *= measurement[i][3];
            }
        }
        distances[3] = distance;
        let fitness = config
            .fitness
            .evaluate(&rewards, &distances, &alive, &observations, 0)?;
        let result = exact_cloning_moments(
            &positions,
            &fitness.fitness,
            &donors,
            &alive,
            &config.clone_decision,
            1,
            jitter,
        )?;
        let barycenter = center(&result.output_means)[0];
        center_mean += mass * barycenter;
        center_second += mass * barycenter * barycenter;
        if let Some(total) = &mut aggregate {
            total.expected_position_variance += mass * result.expected_position_variance;
            total.expected_position_drift += mass * result.expected_position_drift;
            total.conditional_barycenter_variance += mass * result.conditional_barycenter_variance;
            total.jitter_variance_contribution += mass * result.jitter_variance_contribution;
            for i in 0..4 {
                total.acceptance_probabilities[i] += mass * result.acceptance_probabilities[i];
                total.covariance_traces[i] += mass * result.covariance_traces[i];
                total.output_means[i][0] += mass * result.output_means[i][0];
                for j in 0..4 {
                    total.accepted_edge_probabilities[i][j] +=
                        mass * result.accepted_edge_probabilities[i][j];
                }
            }
        } else {
            let mut total = result;
            total.expected_position_variance *= mass;
            total.expected_position_drift *= mass;
            total.conditional_barycenter_variance *= mass;
            total.jitter_variance_contribution *= mass;
            for i in 0..4 {
                total.acceptance_probabilities[i] *= mass;
                total.covariance_traces[i] *= mass;
                total.output_means[i][0] *= mass;
                for j in 0..4 {
                    total.accepted_edge_probabilities[i][j] *= mass;
                }
            }
            // Conditional identities are checked per pattern, not on an
            // averaged fitness vector that has a different decision law.
            total.checks.clear();
            total.collective_flux = None;
            total.collective_flux_identity_residual = None;
            aggregate = Some(total);
        }
    }
    let mut total = aggregate.unwrap();
    total.measurement_barycenter_variance = (center_second - center_mean * center_mean).max(0.);
    total.total_barycenter_variance =
        total.conditional_barycenter_variance + total.measurement_barycenter_variance;
    total.scope="Full measurement-law expectation for canonical all-alive x=[0,0,0,.1], zero velocities, quadratic rewards; all eight distinct measurement patterns, independent donor/gate/jitter integration, before kinetics/boundary. Output row means/edge probabilities/covariance traces are measurement averages; rows are not unconditionally independent.".into();
    total.checks.push(upper(
        "four_walker_canonical_barycenter_concentration",
        &["thm-cloning-canonical-barycenter-concentration"],
        total.total_barycenter_variance,
        (0.01 + jitter * jitter + 0.01 * canonical_cloning_constants().b_0.powi(2) / 2.) / 4.,
    ));
    Ok(total)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CloningEnsembleEstimate {
    pub source_labels: Vec<String>,
    pub samples: usize,
    pub sample_mean: f64,
    pub standard_error: f64,
    pub exact_expectation: Option<f64>,
    pub proved_lower_bound: Option<f64>,
    pub confidence_multiplier: f64,
    /// Compatibility within sampling error is distinct from proving the bound.
    pub consistent_with_theory: bool,
    pub lower_confidence_endpoint: f64,
    pub upper_confidence_endpoint: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CloningEnsembleCase {
    pub name: String,
    pub walkers: usize,
    pub jitter: f64,
    pub input_variance: f64,
    pub positional_variance: CloningEnsembleEstimate,
    pub barycenter_mean: CloningEnsembleEstimate,
    pub barycenter_variance: CloningEnsembleEstimate,
    pub acceptance_pressure: CloningEnsembleEstimate,
    pub drift_lower_bound: Option<f64>,
    pub all_consistent: bool,
    pub scope: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndependentCloningReport {
    pub samples_per_case: usize,
    pub seed: u64,
    pub cases: Vec<CloningEnsembleCase>,
    pub all_consistent: bool,
    pub interpretation: String,
}

/// Complete measurement averaging for canonical balanced two-site swarms.
/// The finite 2^N distance patterns are exactly enumerable for N<=12.
pub fn exact_balanced_two_site_fixture(
    walkers: usize,
    radius: f64,
    jitter: f64,
) -> Result<CloningMomentReport> {
    if !(4..=12).contains(&walkers)
        || !walkers.is_multiple_of(2)
        || !radius.is_finite()
        || radius <= 0.
        || radius >= 2.
    {
        return Err(error(
            "balanced fixture requires even N=4..12 and radius in (0,2)",
        ));
    }
    let config = algorithmic_gas::GasConfig::euclidean(1, 0.04)?;
    let positions: Vec<_> = (0..walkers)
        .map(|i| vec![if i < walkers / 2 { radius } else { -radius }])
        .collect();
    let mut observations = ObservationBatch::positions(TensorBatch::vectors(
        walkers,
        1,
        positions.iter().map(|x| x[0]).collect(),
    )?);
    observations.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(walkers, 1, vec![0.; walkers])?,
    );
    let alive = vec![true; walkers];
    let donors = current_donor_probabilities(&config.cloning_donors, &observations, &alive)?;
    let measurement = current_donor_probabilities(&config.distance_donors, &observations, &alive)?;
    let distance =
        config
            .distance_donors
            .distance
            .compare(&observations, 0, &observations, walkers / 2)?;
    let q = measurement[0][walkers / 2..].iter().sum::<f64>();
    let rewards =
        algorithmic_gas::RewardBatch::new(vec![radius * radius / 2.; walkers], Default::default());
    let mut aggregate: Option<CloningMomentReport> = None;
    let mut center_second = 0.;
    for mask in 0_usize..1 << walkers {
        let high = mask.count_ones() as i32;
        let mass = q.powi(high) * (1. - q).powi(walkers as i32 - high);
        let distances: Vec<_> = (0..walkers)
            .map(|i| if mask & (1 << i) == 0 { 0. } else { distance })
            .collect();
        let fitness = config
            .fitness
            .evaluate(&rewards, &distances, &alive, &observations, 0)?;
        let result = exact_cloning_moments(
            &positions,
            &fitness.fitness,
            &donors,
            &alive,
            &config.clone_decision,
            1,
            jitter,
        )?;
        center_second += mass * norm2(&center(&result.output_means));
        if let Some(total) = &mut aggregate {
            add_moment_average(total, &result, mass);
        } else {
            let mut total = result.clone();
            total.expected_position_variance = 0.;
            total.expected_position_drift = 0.;
            total.conditional_barycenter_variance = 0.;
            total.jitter_variance_contribution = 0.;
            total.acceptance_probabilities.fill(0.);
            total.covariance_traces.fill(0.);
            for row in &mut total.output_means {
                row.fill(0.);
            }
            for row in &mut total.accepted_edge_probabilities {
                row.fill(0.);
            }
            total.checks.clear();
            total.collective_flux = None;
            total.collective_flux_identity_residual = None;
            add_moment_average(&mut total, &result, mass);
            aggregate = Some(total);
        }
    }
    let mut total = aggregate.unwrap();
    total.measurement_barycenter_variance =
        (center_second - norm2(&center(&total.output_means))).max(0.);
    total.total_barycenter_variance =
        total.conditional_barycenter_variance + total.measurement_barycenter_variance;
    let pressure = total.acceptance_probabilities.iter().sum::<f64>() / walkers as f64;
    let lower = canonical_two_cluster_acceptance_floor(radius) * q * (1. - q);
    let pressure_labels = if radius >= 0.5 {
        vec![
            "prop-cloning-two-cluster-noise-balance",
            "cor-keystone-canonical-balanced-structural",
        ]
    } else {
        vec!["prop-cloning-two-cluster-noise-balance"]
    };
    total.checks.push(upper(
        "balanced_two_site_acceptance_pressure",
        &pressure_labels,
        lower,
        pressure,
    ));
    let lower_drift = jitter * jitter * (1. - 1. / walkers as f64) * lower
        - 4. * radius * radius / walkers as f64 * q * q * (1. - q * q);
    total.checks.push(upper(
        "balanced_two_site_noise_drift",
        &["prop-cloning-two-cluster-noise-balance"],
        lower_drift,
        total.expected_position_drift,
    ));
    total.scope="Complete independent measurement-law expectation for equal two-site populations at +/-radius, zero velocities, canonical quadratic rewards and parameters; all 2^N high/low distance patterns, including all ties; proposal before kinetics and terminal killing. The specialized pressure rate is distinct from a contraction rate.".into();
    Ok(total)
}

fn add_moment_average(total: &mut CloningMomentReport, result: &CloningMomentReport, mass: f64) {
    total.expected_position_variance += mass * result.expected_position_variance;
    total.expected_position_drift += mass * result.expected_position_drift;
    total.conditional_barycenter_variance += mass * result.conditional_barycenter_variance;
    total.jitter_variance_contribution += mass * result.jitter_variance_contribution;
    for i in 0..total.rows {
        total.acceptance_probabilities[i] += mass * result.acceptance_probabilities[i];
        total.covariance_traces[i] += mass * result.covariance_traces[i];
        for a in 0..total.dimension {
            total.output_means[i][a] += mass * result.output_means[i][a];
        }
        for j in 0..total.rows {
            total.accepted_edge_probabilities[i][j] +=
                mass * result.accepted_edge_probabilities[i][j];
        }
    }
}

fn ensemble_estimate(
    values: &[f64],
    expectation: Option<f64>,
    lower_bound: Option<f64>,
    labels: &[&str],
) -> CloningEnsembleEstimate {
    let samples = values.len();
    let sample_mean = values.iter().sum::<f64>() / samples as f64;
    let variance = values
        .iter()
        .map(|x| (x - sample_mean).powi(2))
        .sum::<f64>()
        / (samples - 1) as f64;
    let standard_error = (variance / samples as f64).sqrt();
    let confidence_multiplier = 6.;
    let tolerance = 1e-12 * (1. + sample_mean.abs());
    let lower_confidence_endpoint = sample_mean - confidence_multiplier * standard_error;
    let upper_confidence_endpoint = sample_mean + confidence_multiplier * standard_error;
    let consistent_with_theory = expectation.is_none_or(|expected| {
        (sample_mean - expected).abs() <= confidence_multiplier * standard_error + tolerance
    }) && lower_bound
        .is_none_or(|bound| upper_confidence_endpoint + tolerance >= bound);
    CloningEnsembleEstimate {
        source_labels: labels.iter().map(|x| (*x).into()).collect(),
        samples,
        sample_mean,
        standard_error,
        exact_expectation: expectation,
        proved_lower_bound: lower_bound,
        confidence_multiplier,
        consistent_with_theory,
        lower_confidence_endpoint,
        upper_confidence_endpoint,
    }
}

/// Independently seeded actual-engine tests of full measurement-averaged laws.
/// Barycenter variances are estimated around their exactly integrated means,
/// avoiding reuse of a fitted mean in an otherwise unbiased moment comparison.
pub async fn validate_independent_cloning(
    samples: usize,
    seed: u64,
) -> Result<IndependentCloningReport> {
    if !(64..=100_000).contains(&samples) {
        return Err(error(
            "cloning ensembles require 64..100000 independent seeds per case",
        ));
    }
    let mut scenarios = vec![
        (
            "four_walker_jitter_0".to_string(),
            vec![0., 0., 0., 0.1],
            0.,
            None,
        ),
        (
            "four_walker_jitter_0.1".to_string(),
            vec![0., 0., 0., 0.1],
            0.1,
            None,
        ),
    ];
    for walkers in [4, 8] {
        for radius in [0.5, 1., 1.9] {
            let positions = (0..walkers)
                .map(|i| if i < walkers / 2 { radius } else { -radius })
                .collect();
            scenarios.push((
                format!("balanced_N{walkers}_radius_{radius}"),
                positions,
                0.1,
                Some(radius),
            ));
        }
    }
    let mut cases = vec![];
    for (case_index, (name, positions, jitter, radius)) in scenarios.into_iter().enumerate() {
        let walkers = positions.len();
        let exact = if let Some(radius) = radius {
            exact_balanced_two_site_fixture(walkers, radius, jitter)?
        } else {
            exact_four_walker_fixture(jitter)?
        };
        let mean = center(&exact.output_means)[0];
        let exact_pressure = exact.acceptance_probabilities.iter().sum::<f64>() / walkers as f64;
        let pressure_lower = radius.map(|a| {
            let ell = 4. * a / (2. + a);
            let w = (-ell * ell / 8.).exp();
            let m = (walkers / 2) as f64;
            let q = m * w / (m - 1. + m * w);
            canonical_two_cluster_acceptance_floor(a) * q * (1. - q)
        });
        let drift_lower = radius.map(|a| {
            let ell = 4. * a / (2. + a);
            let w = (-ell * ell / 8.).exp();
            let m = (walkers / 2) as f64;
            let q = m * w / (m - 1. + m * w);
            jitter * jitter * (1. - 1. / walkers as f64) * pressure_lower.unwrap()
                - 4. * a * a / walkers as f64 * q * q * (1. - q * q)
        });
        let mut variances = Vec::with_capacity(samples);
        let mut centers = Vec::with_capacity(samples);
        let mut center_squares = Vec::with_capacity(samples);
        let mut pressure = Vec::with_capacity(samples);
        for replicate in 0..samples {
            let mut config = algorithmic_gas::GasConfig::euclidean(1, 0.04)?;
            config.seed = seed
                .wrapping_add(case_index as u64 * 1_000_000)
                .wrapping_add(replicate as u64);
            config.clone_transform.jitter_amplitude = jitter;
            let mut observations =
                ObservationBatch::positions(TensorBatch::vectors(walkers, 1, positions.clone())?);
            observations.fields.insert(
                "velocities".into(),
                TensorBatch::vectors(walkers, 1, vec![0.; walkers])?,
            );
            let population = algorithmic_gas::Population::new(observations)?;
            let model = crate::BenchmarkModel {
                benchmark: crate::Benchmark::Quadratic,
                field: "positions".into(),
                direction: config.fitness.direction,
            };
            let mut gas = algorithmic_gas::GasBuilder::new(population, model.clone())
                .gradient(model)
                .config(config)
                .build()
                .await?;
            gas.start_recording(Default::default())?;
            gas.step().await?;
            let step = &gas.recording().unwrap().steps[0];
            let output = step
                .stages
                .iter()
                .find(|s| s.stage == "post_transform")
                .ok_or_else(|| error("missing actual cloning proposal stage"))?;
            let values = &output.fields["positions"].values;
            let center = values.iter().sum::<f64>() / walkers as f64;
            centers.push(center);
            center_squares.push((center - mean).powi(2));
            variances
                .push(values.iter().map(|x| (x - center).powi(2)).sum::<f64>() / walkers as f64);
            pressure.push(
                step.report
                    .clone_plan
                    .choices
                    .iter()
                    .map(|c| c.probability.unwrap_or(0.))
                    .sum::<f64>()
                    / walkers as f64,
            );
        }
        let positional_labels = if radius.is_some() {
            vec![
                "lem-variance-change-decomposition",
                "prop-cloning-two-cluster-noise-balance",
            ]
        } else {
            vec![
                "lem-variance-change-decomposition",
                "ex-cloning-position-spreading",
            ]
        };
        let positional_variance = ensemble_estimate(
            &variances,
            Some(exact.expected_position_variance),
            drift_lower.map(|bound| bound + exact.entering_alive_variance),
            &positional_labels,
        );
        let barycenter_mean = ensemble_estimate(
            &centers,
            Some(mean),
            None,
            &["def-cloning-frozen-positional-moments"],
        );
        let barycenter_variance = ensemble_estimate(
            &center_squares,
            Some(exact.total_barycenter_variance),
            None,
            &[
                "thm-cloning-canonical-barycenter-concentration",
                "lem-cloning-individual-centered-displacement",
            ],
        );
        let pressure_labels = if radius.is_some() {
            vec![
                "def-cloning-probability",
                "cor-keystone-canonical-balanced-structural",
            ]
        } else {
            vec!["def-cloning-probability"]
        };
        let acceptance_pressure = ensemble_estimate(
            &pressure,
            Some(exact_pressure),
            pressure_lower,
            &pressure_labels,
        );
        let all_consistent = positional_variance.consistent_with_theory
            && barycenter_mean.consistent_with_theory
            && barycenter_variance.consistent_with_theory
            && acceptance_pressure.consistent_with_theory;
        cases.push(CloningEnsembleCase {
            name,
            walkers,
            jitter,
            input_variance: exact.entering_alive_variance,
            positional_variance,
            barycenter_mean,
            barycenter_variance,
            acceptance_pressure,
            drift_lower_bound: drift_lower,
            all_consistent,
            scope: exact.scope,
        });
    }
    let all_consistent = cases.iter().all(|case| case.all_consistent);
    Ok(IndependentCloningReport { samples_per_case:samples,seed,cases,all_consistent,
        interpretation:"Independent actual Rust engine proposals compared with complete finite measurement/donor/gate/jitter integration. Six-standard-error bands quantify Monte Carlo compatibility, not global theorem certificates. Positive expected positional drift is permitted and is observed by the canonical spreading fixture. Pressure lower bounds retain balanced two-site hypotheses and do not assert blanket contraction.".into() })
}
