//! Source coefficient algebra and native scalar-pipeline diagnostics for chapters 1--2.
//! Supplied analytic moduli are explicit inputs; their validity for a landscape is
//! not inferred from the algebraic checks or fitted to trajectories.
use crate::{
    convergence_continuity::optimal_swarm_displacement, convergence_framework::BoundCheck,
};
use algorithmic_gas::{
    GasError, ObservationBatch, Result, TensorBatch,
    cloning::CloneDecision,
    fitness::{PositiveMap, PositiveMapping, Standardizer},
    geometry::Distance,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComputedConstant {
    pub name: String,
    pub value: f64,
    pub formula: String,
    pub source_labels: Vec<String>,
    pub scope: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoefficientReport {
    pub computed: Vec<ComputedConstant>,
    pub checks: Vec<BoundCheck>,
    pub conditional_inputs: Vec<CompositeInputs>,
    pub unavailable: BTreeMap<String, String>,
    pub scope: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StandardizationCoefficients {
    pub mean_value_lipschitz: f64,
    pub second_value_lipschitz: f64,
    pub scale_value_lipschitz: f64,
    pub mean_structural_lipschitz: f64,
    pub second_structural_lipschitz: f64,
    pub scale_structural_lipschitz: f64,
    pub value_direct: f64,
    pub value_mean: f64,
    pub value_scale: f64,
    pub value_total: f64,
    pub structural_direct: f64,
    pub structural_indirect: f64,
    pub structural_indirect_split: f64,
    pub mean_structural_lipschitz_sasaki: f64,
    pub second_structural_lipschitz_sasaki: f64,
    pub scale_structural_lipschitz_sasaki: f64,
    /// Chapter 2 uses its 3V/k_min moduli and k2 in the denominator-shift
    /// norm bound. The chapter 1 k1 coefficient above is kept separately.
    pub structural_indirect_split_sasaki: f64,
}

fn require(value: bool, message: &str) -> Result<()> {
    if value {
        Ok(())
    } else {
        Err(GasError::Configuration(message.into()))
    }
}
fn finite(values: &[f64]) -> bool {
    values.iter().all(|x| x.is_finite() && *x >= 0.)
}

/// Empirical-aggregator specializations of both structural coefficient definitions.
/// `floor` is sqrt(kappa_var_min+epsilon_std²), or the canonical sigma_min.
pub fn standardization_coefficients(
    k1: usize,
    k2: usize,
    stable: usize,
    maximum: f64,
    floor: f64,
) -> Result<StandardizationCoefficients> {
    require(
        k1 > 0 && k2 > 0 && stable <= k1.min(k2),
        "nonempty alive sets and valid stable count required",
    )?;
    require(
        maximum.is_finite() && maximum >= 0. && floor.is_finite() && floor > 0.,
        "finite nonnegative value bound and positive floor required",
    )?;
    let k = k1 as f64;
    let mu_m = 1. / k.sqrt();
    let m2_m = 2. * maximum / k.sqrt();
    let scale_m = (m2_m + 2. * maximum * mu_m) / (2. * floor);
    let mu_s = 2. * maximum / k2 as f64;
    let m2_s = 2. * maximum.powi(2) / k2 as f64;
    let scale_s = (m2_s + 2. * maximum * mu_s) / (2. * floor);
    let direct = 1. / floor.powi(2);
    let mean = k * mu_m.powi(2) / floor.powi(2);
    let scale = k * (2. * maximum / floor).powi(2) * (scale_m / floor).powi(2);
    let structural_direct = (2. * maximum / floor).powi(2);
    let structural_indirect =
        stable as f64 * (mu_s / floor + 2. * maximum * scale_s / floor.powi(2)).powi(2);
    let structural_indirect_split = 2. * stable as f64 * mu_s.powi(2) / floor.powi(2)
        + 2. * k * (2. * maximum / floor).powi(2) * scale_s.powi(2) / floor.powi(2);
    let k_min = k1.min(k2) as f64;
    let mu_s_sasaki = 3. * maximum / k_min;
    let m2_s_sasaki = 3. * maximum.powi(2) / k_min;
    let scale_s_sasaki = (m2_s_sasaki + 2. * maximum * mu_s_sasaki) / (2. * floor);
    let structural_indirect_split_sasaki = 2. * stable as f64 * mu_s_sasaki.powi(2) / floor.powi(2)
        + 2. * k2 as f64 * (2. * maximum / floor).powi(2) * scale_s_sasaki.powi(2) / floor.powi(2);
    require(
        finite(&[
            mu_m,
            m2_m,
            scale_m,
            mu_s,
            m2_s,
            scale_s,
            direct,
            mean,
            scale,
            structural_direct,
            structural_indirect,
            structural_indirect_split,
            mu_s_sasaki,
            m2_s_sasaki,
            scale_s_sasaki,
            structural_indirect_split_sasaki,
        ]),
        "coefficient overflow; choose representable counts, bounds and floors",
    )?;
    let total = 3. * (direct + mean + scale);
    require(total.is_finite(), "total value coefficient overflow")?;
    Ok(StandardizationCoefficients {
        mean_value_lipschitz: mu_m,
        second_value_lipschitz: m2_m,
        scale_value_lipschitz: scale_m,
        mean_structural_lipschitz: mu_s,
        second_structural_lipschitz: m2_s,
        scale_structural_lipschitz: scale_s,
        value_direct: direct,
        value_mean: mean,
        value_scale: scale,
        value_total: total,
        structural_direct,
        structural_indirect,
        structural_indirect_split,
        mean_structural_lipschitz_sasaki: mu_s_sasaki,
        second_structural_lipschitz_sasaki: m2_s_sasaki,
        scale_structural_lipschitz_sasaki: scale_s_sasaki,
        structural_indirect_split_sasaki,
    })
}

/// Analytic inputs needed for source cloning and final-stage coefficient assembly.
/// A1..A4 bound F_pot by A1 Delta_pos²+A2 n_c+A3 n_c²+A4.
/// These are supplied hypotheses, not constants estimated by this module.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompositeInputs {
    pub walkers: usize,
    pub alive1: usize,
    pub diameter: f64,
    pub status_penalty: f64,
    pub death_lipschitz: f64,
    pub boundary_exponent: f64,
    pub perturbation_moment_squared: f64,
    pub failure_probability: f64,
    pub potential_error_coefficients: [f64; 4],
    pub clone_value_lipschitz: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompositeCoefficients {
    pub potential_quadratic: [f64; 3],
    pub probability_linear: f64,
    pub probability_half: f64,
    pub probability_offset: f64,
    pub clone_linear: f64,
    pub clone_half: f64,
    pub clone_offset: f64,
    pub status_holder: f64,
    pub status_variance: f64,
    pub perturbation_mean: f64,
    pub perturbation_fluctuation: f64,
    pub perturbation_offset: f64,
    pub composite_a: f64,
    pub composite_k0: f64,
    pub final_linear: f64,
    pub final_holder: f64,
    pub final_offset: f64,
    pub small_exponent: f64,
    pub large_exponent: f64,
}
impl CompositeInputs {
    pub fn validate(&self) -> Result<()> {
        require(
            self.walkers >= 2 && self.alive1 >= 2 && self.alive1 <= self.walkers,
            "composite cloning requires N>=k1>=2",
        )?;
        require(
            finite(&[
                self.diameter,
                self.status_penalty,
                self.death_lipschitz,
                self.boundary_exponent,
                self.perturbation_moment_squared,
                self.clone_value_lipschitz,
            ]) && finite(&self.potential_error_coefficients)
                && self.diameter > 0.
                && self.status_penalty > 0.
                && self.boundary_exponent > 0.
                && self.boundary_exponent <= 1.
                && self.failure_probability.is_finite()
                && self.failure_probability > 0.
                && self.failure_probability < 1.,
            "invalid/nonfinite composite moduli, exponent or failure probability",
        )
    }
}
pub fn composite_coefficients(input: &CompositeInputs) -> Result<CompositeCoefficients> {
    input.validate()?;
    let n = input.walkers as f64;
    let alpha = input.boundary_exponent;
    let [a1, a2, a3, a4] = input.potential_error_coefficients;
    let b1 = n * a1 + n / input.status_penalty * a2;
    let b2 = (n / input.status_penalty).powi(2) * a3;
    let b0 = a4;
    let c_struct = 2. / (input.alive1 - 1) as f64;
    let cp = n * n * c_struct / input.status_penalty
        + n * input.clone_value_lipschitz * (2. * n * b2).sqrt();
    let hp = n * input.clone_value_lipschitz * (2. * n * b1).sqrt();
    let kp = 2. * n + n * input.clone_value_lipschitz * (2. * n * b0).sqrt();
    let clone_l = 3. + 3. * input.diameter.powi(2) * cp / n;
    let clone_h = 3. * input.diameter.powi(2) * hp / n;
    let clone_k = 3. * input.diameter.powi(2) * kp / n;
    let c_status = input.death_lipschitz.powi(2) * n.powf(1. - alpha);
    let k_status = n / 2.;
    let mean = n * input.perturbation_moment_squared;
    let fluctuation =
        input.diameter.powi(2) * (n / 2. * (2. / input.failure_probability).ln()).sqrt();
    let k_pert =
        6. * mean + 6. * fluctuation + input.failure_probability * n * input.diameter.powi(2);
    let a = input.status_penalty * n.powf(alpha - 1.) * c_status;
    let k0 = (k_pert + input.status_penalty * k_status) / n;
    let result = CompositeCoefficients {
        potential_quadratic: [b2, b1, b0],
        probability_linear: cp,
        probability_half: hp,
        probability_offset: kp,
        clone_linear: clone_l,
        clone_half: clone_h,
        clone_offset: clone_k,
        status_holder: c_status,
        status_variance: k_status,
        perturbation_mean: mean,
        perturbation_fluctuation: fluctuation,
        perturbation_offset: k_pert,
        composite_a: a,
        composite_k0: k0,
        final_linear: 3. * clone_l,
        final_holder: 3. * clone_h + a * clone_l.powf(alpha) + a * clone_h.powf(alpha),
        final_offset: 3. * clone_k + a * clone_k.powf(alpha) + k0,
        small_exponent: alpha / 2.,
        large_exponent: 0.5_f64.max(alpha),
    };
    require(
        finite(&[
            b2,
            b1,
            b0,
            cp,
            hp,
            kp,
            clone_l,
            clone_h,
            clone_k,
            c_status,
            k_status,
            mean,
            fluctuation,
            k_pert,
            a,
            k0,
            result.final_linear,
            result.final_holder,
            result.final_offset,
        ]),
        "composite coefficient overflow",
    )?;
    Ok(result)
}

fn norm_squared(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}
fn difference_squared(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum()
}
fn masked_second_moment(values: &[f64], alive: &[bool]) -> f64 {
    values
        .iter()
        .zip(alive)
        .filter(|(_, alive)| **alive)
        .map(|(value, _)| value * value)
        .sum::<f64>()
        / alive.iter().filter(|a| **a).count() as f64
}
fn record(
    report: &mut CoefficientReport,
    name: &str,
    value: f64,
    formula: &str,
    labels: &[&str],
    scope: &str,
) {
    report.computed.push(ComputedConstant {
        name: name.into(),
        value,
        formula: formula.into(),
        source_labels: labels.iter().map(|s| (*s).into()).collect(),
        scope: scope.into(),
    });
}
fn check(
    report: &mut CoefficientReport,
    id: &str,
    labels: &[&str],
    scope: &str,
    observed: f64,
    bound: f64,
) {
    report
        .checks
        .push(BoundCheck::upper(id, labels, scope, observed, bound));
}

fn standardization_cases(report: &mut CoefficientReport) -> Result<()> {
    for n in [2usize, 3, 8, 16] {
        let obs = ObservationBatch::positions(TensorBatch::vectors(n, 1, vec![0.; n])?);
        let mut masks = vec![vec![true; n]];
        for length in [1, n / 2, n - 1] {
            masks.push((0..n).map(|i| i < length).collect());
        }
        masks.push((0..n).map(|i| i % 2 == 0).collect());
        for floor in [0.01_f64, 0.1, 1.] {
            let native = Standardizer::Global { sigma_min: floor };
            for pattern in 0..4 {
                let raw: Vec<f64> = (0..n)
                    .map(|i| match pattern {
                        0 => 0.25,
                        1 => {
                            if i % 2 == 0 {
                                -1.
                            } else {
                                1.
                            }
                        }
                        2 => 1e-10 * (i as f64).sin(),
                        _ => (i as f64 * 1.7).sin(),
                    })
                    .collect();
                let changed: Vec<f64> = raw
                    .iter()
                    .enumerate()
                    .map(|(i, x)| (x + 0.07 * (i as f64 + 0.3).cos()).clamp(-1., 1.))
                    .collect();
                for mask1 in &masks {
                    for mask2 in &masks {
                        let k1 = mask1.iter().filter(|x| **x).count();
                        let k2 = mask2.iter().filter(|x| **x).count();
                        let stable = mask1.iter().zip(mask2).filter(|(a, b)| **a && **b).count();
                        let nc = mask1.iter().zip(mask2).filter(|(a, b)| a != b).count() as f64;
                        let coeff = standardization_coefficients(k1, k2, stable, 1., floor)?;
                        let (z1, s1) = native.apply(&raw, mask1, &obs)?;
                        let (inter, si) = native.apply(&changed, mask1, &obs)?;
                        let (z2, s2) = native.apply(&changed, mask2, &obs)?;
                        let value_delta = raw
                            .iter()
                            .zip(&changed)
                            .zip(mask1)
                            .filter(|(_, a)| **a)
                            .map(|((a, b), _)| (a - b).powi(2))
                            .sum::<f64>();
                        let ev = difference_squared(&z1, &inter);
                        let es = difference_squared(&inter, &z2);
                        let all = difference_squared(&z1, &z2);
                        let scope = format!(
                            "native global quadratic floor={floor}; N={n}, k1={k1},k2={k2},stable={stable},n_c={nc}; bounded frozen scalar arrays pattern={pattern}"
                        );
                        check(
                            report,
                            "value_standardization",
                            &[
                                "def-value-error-coefficients",
                                "def-lipschitz-value-error-coefficients",
                                "thm-lipschitz-value-error-bound",
                                "thm-standardization-value-error-mean-square",
                                "thm-sasaki-standardization-value-sq",
                            ],
                            &format!(
                                "{scope}; scalar-value inequality only; the spatial reward bound is exercised separately by the compact linear-reward Sasaki fixture"
                            ),
                            ev,
                            coeff.value_total * value_delta,
                        );
                        check(
                            report,
                            "structural_combined_coefficient",
                            &[
                                "def-structural-error-coefficients",
                                "thm-standardization-structural-error-mean-square",
                            ],
                            &scope,
                            es,
                            coeff.structural_direct * nc + coeff.structural_indirect * nc * nc,
                        );
                        check(
                            report,
                            "structural_split_coefficient",
                            &[
                                "def-lipschitz-structural-error-coefficients",
                                "thm-lipschitz-structural-error-bound",
                            ],
                            &scope,
                            es,
                            coeff.structural_direct * nc
                                + coeff.structural_indirect_split * nc * nc,
                        );
                        let direct_structural = (0..n)
                            .filter(|&i| mask1[i] != mask2[i])
                            .map(|i| (inter[i] - z2[i]).powi(2))
                            .sum::<f64>();
                        let indirect_structural = (0..n)
                            .filter(|&i| mask1[i] && mask2[i])
                            .map(|i| (inter[i] - z2[i]).powi(2))
                            .sum::<f64>();
                        check(
                            report,
                            "sasaki_structural_orthogonal_decomposition",
                            &["lem-sasaki-structural-error-decomposition"],
                            &scope,
                            (es - direct_structural - indirect_structural).abs(),
                            0.,
                        );
                        check(
                            report,
                            "sasaki_direct_structural_error_bound",
                            &[
                                "lem-sasaki-direct-structural-error-sq",
                                "def-sasaki-structural-coeffs-sq",
                            ],
                            &scope,
                            direct_structural,
                            coeff.structural_direct * nc,
                        );
                        check(
                            report,
                            "sasaki_indirect_structural_error_bound",
                            &[
                                "lem-sasaki-indirect-structural-error-sq",
                                "def-sasaki-structural-coeffs-sq",
                            ],
                            &scope,
                            indirect_structural,
                            coeff.structural_indirect_split_sasaki * nc * nc,
                        );
                        check(
                            report,
                            "sasaki_structural_total_bound",
                            &["thm-sasaki-standardization-structural-sq"],
                            &scope,
                            es,
                            coeff.structural_direct * nc
                                + coeff.structural_indirect_split_sasaki * nc * nc,
                        );
                        check(
                            report,
                            "global_standardization",
                            &["thm-global-continuity-patched-standardization"],
                            &scope,
                            all,
                            2. * coeff.value_total * value_delta
                                + 2. * coeff.structural_direct * nc
                                + 2. * coeff.structural_indirect_split * nc * nc,
                        );
                        check(
                            report,
                            "value_structural_decomposition",
                            &[
                                "thm-deterministic-error-decomposition",
                                "thm-standardization-operator-unified-mean-square-continuity",
                            ],
                            &scope,
                            all,
                            2. * ev + 2. * es,
                        );
                        check(
                            report,
                            "structural_mean_modulus",
                            &[
                                "lem-stats-structural-continuity",
                                "lem-sasaki-aggregator-structural",
                            ],
                            &scope,
                            (si.mean[0] - s2.mean[0]).abs(),
                            coeff.mean_structural_lipschitz * nc,
                        );
                        check(
                            report,
                            "sasaki_structural_second_moment_modulus",
                            &["lem-sasaki-aggregator-structural"],
                            &scope,
                            (masked_second_moment(&changed, mask1)
                                - masked_second_moment(&changed, mask2))
                            .abs(),
                            coeff.second_structural_lipschitz_sasaki * nc,
                        );
                        check(
                            report,
                            "structural_scale_modulus",
                            &["lem-stats-structural-continuity"],
                            &scope,
                            (si.scale[0] - s2.scale[0]).abs(),
                            coeff.scale_structural_lipschitz * nc,
                        );
                        let direct: Vec<f64> = (0..n)
                            .map(|i| {
                                if mask1[i] {
                                    (raw[i] - changed[i]) / s1.scale[i]
                                } else {
                                    0.
                                }
                            })
                            .collect();
                        let mean: Vec<f64> = (0..n)
                            .map(|i| {
                                if mask1[i] {
                                    (si.mean[i] - s1.mean[i]) / s1.scale[i]
                                } else {
                                    0.
                                }
                            })
                            .collect();
                        let denominator: Vec<f64> = (0..n)
                            .map(|i| inter[i] * (si.scale[i] - s1.scale[i]) / s1.scale[i])
                            .collect();
                        let residual: Vec<f64> = (0..n)
                            .map(|i| z1[i] - inter[i] - direct[i] - mean[i] - denominator[i])
                            .collect();
                        check(
                            report,
                            "value_error_algebra",
                            &[
                                "lem-sub-value-error-decomposition",
                                "lem-algebraic-value-error-decomposition",
                                "lem-sasaki-value-error-decomposition",
                            ],
                            &scope,
                            norm_squared(&residual),
                            1e-24,
                        );
                        check(
                            report,
                            "direct_shift_bound",
                            &[
                                "lem-direct-value-shift-bound",
                                "lem-sasaki-direct-shift-bound-sq",
                            ],
                            &scope,
                            norm_squared(&direct),
                            coeff.value_direct * value_delta,
                        );
                        check(
                            report,
                            "mean_shift_bound",
                            &["lem-sub-mean-shift-bound", "lem-sasaki-mean-shift-bound-sq"],
                            &scope,
                            norm_squared(&mean),
                            coeff.value_mean * value_delta,
                        );
                        check(
                            report,
                            "scale_shift_bound",
                            &[
                                "lem-sub-statistical-fluctuation-bound",
                                "lem-sasaki-denom-shift-bound-sq",
                            ],
                            &scope,
                            norm_squared(&denominator),
                            coeff.value_scale * value_delta,
                        );
                        check(
                            report,
                            "value_three_component_norm_bound",
                            &["lem-sasaki-value-error-decomposition"],
                            &scope,
                            ev,
                            3. * (norm_squared(&direct)
                                + norm_squared(&mean)
                                + norm_squared(&denominator)),
                        );
                        if pattern == 0 && mask1 == &masks[0] && mask2 == &masks[1] {
                            for (name, value, formula, labels) in [
                                (
                                    "L_mu_M",
                                    coeff.mean_value_lipschitz,
                                    "1/sqrt(k1)",
                                    vec!["lem-empirical-moments-lipschitz"],
                                ),
                                (
                                    "L_m2_M",
                                    coeff.second_value_lipschitz,
                                    "2Vmax/sqrt(k1)",
                                    vec!["lem-empirical-moments-lipschitz"],
                                ),
                                (
                                    "L_sigma_M",
                                    coeff.scale_value_lipschitz,
                                    "(L_m2_M+2Vmax L_mu_M)/(2m)",
                                    vec!["lem-stats-value-continuity"],
                                ),
                                (
                                    "L_mu_S",
                                    coeff.mean_structural_lipschitz,
                                    "2Vmax/k2",
                                    vec!["lem-empirical-aggregator-properties"],
                                ),
                                (
                                    "L_m2_S",
                                    coeff.second_structural_lipschitz,
                                    "2Vmax²/k2",
                                    vec!["lem-empirical-aggregator-properties"],
                                ),
                                (
                                    "L_sigma_S",
                                    coeff.scale_structural_lipschitz,
                                    "(L_m2_S+2Vmax L_mu_S)/(2m)",
                                    vec!["lem-stats-structural-continuity"],
                                ),
                                (
                                    "C_V_direct",
                                    coeff.value_direct,
                                    "1/m²",
                                    vec![
                                        "def-value-error-coefficients",
                                        "def-lipschitz-value-error-coefficients",
                                    ],
                                ),
                                (
                                    "C_V_mu",
                                    coeff.value_mean,
                                    "k L_mu,M²/m²",
                                    vec!["def-value-error-coefficients"],
                                ),
                                (
                                    "C_V_sigma",
                                    coeff.value_scale,
                                    "k(2Vmax/m)²(L_sigma,M/m)²",
                                    vec!["def-value-error-coefficients"],
                                ),
                                (
                                    "C_V_total",
                                    coeff.value_total,
                                    "3(C_V_direct+C_V_mu+C_V_sigma)",
                                    vec!["def-value-error-coefficients"],
                                ),
                                (
                                    "C_S_direct",
                                    coeff.structural_direct,
                                    "4Vmax²/m²",
                                    vec![
                                        "def-structural-error-coefficients",
                                        "def-lipschitz-structural-error-coefficients",
                                    ],
                                ),
                                (
                                    "C_S_indirect_combined",
                                    coeff.structural_indirect,
                                    "stable(L_mu,S/m+2Vmax L_sigma,S/m²)²",
                                    vec!["def-structural-error-coefficients"],
                                ),
                                (
                                    "C_S_indirect_split",
                                    coeff.structural_indirect_split,
                                    "2stable L_mu,S²/m²+2k1(2Vmax/m)² L_sigma,S²/m²",
                                    vec!["def-lipschitz-structural-error-coefficients"],
                                ),
                                (
                                    "L_mu_S_sasaki",
                                    coeff.mean_structural_lipschitz_sasaki,
                                    "3Vmax/k_min",
                                    vec!["lem-sasaki-aggregator-structural"],
                                ),
                                (
                                    "L_m2_S_sasaki",
                                    coeff.second_structural_lipschitz_sasaki,
                                    "3Vmax²/k_min",
                                    vec!["lem-sasaki-aggregator-structural"],
                                ),
                                (
                                    "L_sigma_S_sasaki",
                                    coeff.scale_structural_lipschitz_sasaki,
                                    "(L_m2_S_sasaki+2Vmax L_mu_S_sasaki)/(2m)",
                                    vec!["lem-sasaki-indirect-structural-error-sq"],
                                ),
                                (
                                    "structural_indirect_split_sasaki",
                                    coeff.structural_indirect_split_sasaki,
                                    "2stable L_mu_S_sasaki²/m²+2k2(2Vmax/m)² L_sigma_S_sasaki²/m²",
                                    vec!["def-sasaki-structural-coeffs-sq"],
                                ),
                            ] {
                                record(report, name, value, formula, &labels, &scope);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// The reward is the actual coordinate R(x)=x on the compact interval |x|<=1.
/// Radius-2 squashing has physical inverse Lipschitz constant (1+1/2)^2 there.
/// Each comparison uses representatives from the optimal marked input plan;
/// storage order is irrelevant and standardized output costs are coupling upper bounds.
fn sasaki_composite_cases(report: &mut CoefficientReport) -> Result<()> {
    let metric = Distance::SquashedPhaseSpace {
        positions: "positions".into(),
        velocities: "velocities".into(),
        position_radius: 2.,
        velocity_radius: 2.,
        lambda: 1.,
    };
    let observe = |values: &[f64]| -> Result<ObservationBatch<f64>> {
        let mut obs =
            ObservationBatch::positions(TensorBatch::vectors(values.len(), 1, values.to_vec())?);
        obs.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(values.len(), 1, vec![0.; values.len()])?,
        );
        Ok(obs)
    };
    for n in [2usize, 3, 8, 16] {
        let masks = [
            vec![true; n],
            (0..n).map(|i| i == 0).collect(),
            (0..n).map(|i| i < n / 2).collect(),
            (0..n).map(|i| i % 2 == 0).collect(),
        ];
        for pattern in 0..3 {
            let x1 = (0..n)
                .map(|i| match pattern {
                    0 => 0.25,
                    1 => {
                        if i % 2 == 0 {
                            -1.
                        } else {
                            1.
                        }
                    }
                    _ => (i as f64 * 1.7).sin(),
                })
                .collect::<Vec<f64>>();
            let x2 = x1
                .iter()
                .enumerate()
                .map(|(i, x)| (x + 0.07 * (i as f64 + 0.3).cos()).clamp(-1., 1.))
                .collect::<Vec<_>>();
            let obs1 = observe(&x1)?;
            let obs2 = observe(&x2)?;
            for first_mask in &masks {
                for second_mask in &masks {
                    let transport = optimal_swarm_displacement(
                        &metric,
                        &obs1,
                        &obs2,
                        first_mask,
                        second_mask,
                        1.,
                    )?;
                    let permutation = transport
                        .transport_plan
                        .iter()
                        .map(|row| row.iter().position(|mass| *mass > 0.5 / n as f64))
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| {
                            GasError::Numerical(
                                "uniform Sasaki transport did not return a permutation optimizer"
                                    .into(),
                            )
                        })?;
                    let aligned_mask = permutation
                        .iter()
                        .map(|&j| second_mask[j])
                        .collect::<Vec<_>>();
                    let r1 = x1
                        .iter()
                        .zip(first_mask)
                        .map(|(x, alive)| if *alive { *x } else { 0. })
                        .collect::<Vec<_>>();
                    let r2 = permutation
                        .iter()
                        .map(|&j| if second_mask[j] { x2[j] } else { 0. })
                        .collect::<Vec<_>>();
                    let k1 = first_mask.iter().filter(|a| **a).count();
                    let k2 = aligned_mask.iter().filter(|a| **a).count();
                    let stable = first_mask
                        .iter()
                        .zip(&aligned_mask)
                        .filter(|(a, b)| **a && **b)
                        .count();
                    let nc = transport.status_mismatch_count;
                    for floor in [0.01_f64, 0.1, 1.] {
                        let native = Standardizer::Global { sigma_min: floor };
                        let coeff = standardization_coefficients(k1, k2, stable, 1., floor)?;
                        let (z1, _) = native.apply(&r1, first_mask, &obs1)?;
                        let (z2, _) = native.apply(&r2, &aligned_mask, &obs1)?;
                        let lr = 2.25_f64;
                        let d2 = transport.metric_squared;
                        let linear = 2. * coeff.value_total * lr.powi(2) * n as f64
                            + n as f64 * (2. * coeff.value_total + 2. * coeff.structural_direct);
                        let higher =
                            2. * coeff.structural_indirect_split_sasaki * (n as f64).powi(2);
                        let scope = format!(
                            "native global floor={floor}; compact linear reward R(x)=x on |x|<=1, Vmax=1, v=0, radius-2 squashed Sasaki metric, L_R=2.25 by compact inverse projection; N={n},k1={k1},k2={k2},stable={stable},n_c={nc},pattern={pattern}; optimal all-slot marked input transport selects comparison representatives, output squared L2 is the cost of a supplied admissible standardized-output coupling and bounds the optimal output cost"
                        );
                        let reward_labels: &[&str] = if nc == 0. {
                            &[
                                "thm-sasaki-standardization-value-sq",
                                "thm-sasaki-standardization-composite-sq",
                            ]
                        } else {
                            &["thm-sasaki-standardization-composite-sq"]
                        };
                        check(
                            report,
                            "sasaki_compact_linear_reward_modulus",
                            reward_labels,
                            &scope,
                            difference_squared(&r1, &r2),
                            lr.powi(2) * transport.positional_sum + nc,
                        );
                        check(
                            report,
                            "sasaki_composite_standardization_bound",
                            &[
                                "thm-sasaki-standardization-composite-sq",
                                "lem-sasaki-standardization-lipschitz",
                            ],
                            &scope,
                            difference_squared(&z1, &z2),
                            linear * d2 + higher * d2 * d2,
                        );
                        if pattern == 0 && first_mask == &masks[0] && second_mask == &masks[1] {
                            record(
                                report,
                                "L_z_L_squared_sasaki",
                                linear,
                                "2C_V_total L_R²N+(N/lambda_status)(2C_V_total Vmax²+2C_S_direct)",
                                &["thm-sasaki-standardization-composite-sq"],
                                &scope,
                            );
                            record(
                                report,
                                "L_z_H_squared_sasaki",
                                higher,
                                "2C_S_indirect_split_sasaki(N/lambda_status)²",
                                &["thm-sasaki-standardization-composite-sq"],
                                &scope,
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn potential_cases(report: &mut CoefficientReport) -> Result<()> {
    let map = PositiveMap::Logistic {
        amplitude: 2.,
        floor: 0.1,
    };
    let own_gate = CloneDecision {
        epsilon: 0.01,
        saturation: 0.7,
        ..Default::default()
    };
    let eta = 0.1_f64;
    let upper = 2.1_f64;
    for alpha in [0.0_f64, 0.25, 1., 2.] {
        for beta in [0.0_f64, 0.25, 1., 2.] {
            if alpha + beta == 0. {
                continue;
            }
            let v_min = eta.powf(alpha + beta);
            let v_max = upper.powf(alpha + beta);
            let l_r = alpha
                * if alpha >= 1. {
                    upper.powf(alpha - 1.)
                } else {
                    eta.powf(alpha - 1.)
                }
                * upper.powf(beta)
                * 0.5;
            let l_d =
                beta * if beta >= 1. {
                    upper.powf(beta - 1.)
                } else {
                    eta.powf(beta - 1.)
                } * upper.powf(alpha)
                    * 0.5;
            let l_pc = 1. / (own_gate.saturation * own_gate.epsilon);
            let l_pi =
                (v_max + own_gate.epsilon) / (own_gate.saturation * own_gate.epsilon.powi(2));
            let scope = format!(
                "native logistic amplitude=2,eta=0.1; exponents alpha={alpha},beta={beta}; no global swarm attraction asserted"
            );
            for (name, value, formula, labels) in [
                (
                    "V_pot_min",
                    v_min,
                    "eta^(alpha+beta)",
                    vec!["lem-potential-boundedness"],
                ),
                (
                    "V_pot_max",
                    v_max,
                    "(g_max+eta)^(alpha+beta), logistic g_max=2",
                    vec!["lem-potential-boundedness"],
                ),
                (
                    "L_F_reward",
                    l_r,
                    "alpha max(eta^(alpha-1),upper^(alpha-1)) upper^beta L_g",
                    vec!["lem-component-potential-lipschitz"],
                ),
                (
                    "L_F_diversity",
                    l_d,
                    "beta max(eta^(beta-1),upper^(beta-1)) upper^alpha L_g",
                    vec!["lem-component-potential-lipschitz"],
                ),
                (
                    "L_pi_companion",
                    l_pc,
                    "1/(p_max epsilon_clone)",
                    vec!["lem-cloning-probability-lipschitz"],
                ),
                (
                    "L_pi_own",
                    l_pi,
                    "(V_pot_max+epsilon_clone)/(p_max epsilon_clone²)",
                    vec!["lem-cloning-probability-lipschitz"],
                ),
            ] {
                record(report, name, value, formula, &labels, &scope);
            }
            for j in -80..80 {
                let r = j as f64 / 8.;
                let d = (j as f64 * 0.63).sin() * 8.;
                let r2 = r + 0.13;
                let d2 = d - 0.09;
                let f = map.map(r)?.powf(alpha) * map.map(d)?.powf(beta);
                let f2 = map.map(r2)?.powf(alpha) * map.map(d2)?.powf(beta);
                check(
                    report,
                    "potential_lower",
                    &["lem-potential-boundedness"],
                    &scope,
                    v_min,
                    f,
                );
                check(
                    report,
                    "potential_upper",
                    &["lem-potential-boundedness"],
                    &scope,
                    f,
                    v_max,
                );
                check(
                    report,
                    "potential_component_modulus",
                    &["lem-component-potential-lipschitz"],
                    &scope,
                    (f - f2).abs(),
                    l_r * (r - r2).abs() + l_d * (d - d2).abs(),
                );
                let other = (f * 0.87).max(v_min);
                let other2 = (f2 * 0.91).max(v_min);
                check(
                    report,
                    "clone_gate_modulus",
                    &["lem-cloning-probability-lipschitz"],
                    &scope,
                    (own_gate.acceptance_probability(0, other, f)
                        - own_gate.acceptance_probability(0, other2, f2))
                    .abs(),
                    l_pc * (f - f2).abs() + l_pi * (other - other2).abs(),
                );
            }
            // Both status and raw-value contributions in the actual native pipeline.
            let n = 8usize;
            let obs = ObservationBatch::positions(TensorBatch::vectors(n, 1, vec![0.; n])?);
            let mask1 = vec![true; n];
            let mask2: Vec<bool> = (0..n).map(|i| i < 5).collect();
            let nc = 3.;
            let raw1: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
            let raw2: Vec<f64> = (0..n).map(|i| 0.8 * (i as f64 + 0.1).sin()).collect();
            let distance1: Vec<f64> = raw1.iter().map(|v| (v + 1.) / 2.).collect();
            let distance2: Vec<f64> = raw2.iter().map(|v| (v + 1.) / 2.).collect();
            let native = Standardizer::Global { sigma_min: 0.1 };
            let (r1, _) = native.apply(&raw1, &mask1, &obs)?;
            let (r2, _) = native.apply(&raw2, &mask2, &obs)?;
            let (d1, _) = native.apply(&distance1, &mask1, &obs)?;
            let (d2, _) = native.apply(&distance2, &mask2, &obs)?;
            let coeff = standardization_coefficients(8, 5, 5, 1., 0.1)?;
            let global = |delta: f64| {
                2. * coeff.value_total * delta
                    + 2. * coeff.structural_direct * nc
                    + 2. * coeff.structural_indirect_split * nc * nc
            };
            let potential1 = (0..n)
                .map(|i| Ok(map.map(r1[i])?.powf(alpha) * map.map(d1[i])?.powf(beta)))
                .collect::<Result<Vec<_>>>()?;
            let potential2 = (0..n)
                .map(|i| {
                    if mask2[i] {
                        Ok(map.map(r2[i])?.powf(alpha) * map.map(d2[i])?.powf(beta))
                    } else {
                        Ok(0.)
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let c_struct = 2. / (n - 1) as f64;
            let c_value = l_pc.max(l_pi);
            record(
                report,
                "C_struct_pi",
                c_struct,
                "2/max(1,k1-1)",
                &["thm-expected-cloning-action-continuity"],
                "auxiliary uniform current-frame donor law; N=k1=8",
            );
            record(
                report,
                "C_val_pi",
                c_value,
                "max(L_pi_companion,L_pi_own)",
                &["thm-expected-cloning-action-continuity"],
                &scope,
            );
            for i in 0..5 {
                let expectation = |values: &[f64], mask: &[bool]| {
                    let donors: Vec<usize> = (0..n).filter(|j| *j != i && mask[*j]).collect();
                    donors
                        .iter()
                        .map(|j| own_gate.acceptance_probability(0, values[i], values[*j]))
                        .sum::<f64>()
                        / donors.len() as f64
                };
                let p1 = expectation(&potential1, &mask1);
                let p_intermediate = expectation(&potential1, &mask2);
                let p2 = expectation(&potential2, &mask2);
                check(
                    report,
                    "conditional_clone_structural_modulus",
                    &["lem-total-clone-prob-structural-error"],
                    "exact uniform donor expectations; stable recipient, fixed potentials, changed support",
                    (p1 - p_intermediate).abs(),
                    c_struct * nc,
                );
                let donor_error = (0..n)
                    .filter(|j| *j != i)
                    .map(|j| (potential1[j] - potential2[j]).abs())
                    .sum::<f64>()
                    / (n - 1) as f64;
                check(
                    report,
                    "conditional_clone_action_modulus",
                    &["thm-expected-cloning-action-continuity"],
                    "exact uniform donor expectations; stable recipient and frozen bounded fitness",
                    (p1 - p2).abs(),
                    c_struct * nc + c_value * (donor_error + (potential1[i] - potential2[i]).abs()),
                );
            }
            check(
                report,
                "deterministic_potential_pipeline",
                &[
                    "thm-deterministic-potential-continuity",
                    "thm-potential-operator-is-mean-square-continuous",
                    "lem-sub-potential-unstable-error-mean-square",
                    "lem-sub-potential-stable-error-mean-square",
                ],
                &scope,
                difference_squared(&potential1, &potential2),
                v_max.powi(2) * nc
                    + 2. * l_r.powi(2) * global(difference_squared(&raw1, &raw2))
                    + 2. * l_d.powi(2) * global(difference_squared(&distance1, &distance2)),
            );
        }
    }
    Ok(())
}

fn inequality_and_composite_cases(report: &mut CoefficientReport) -> Result<()> {
    for floor in [0.01_f64, 0.1, 1.] {
        for order in 1..=8 {
            let mut falling = 1.;
            for j in 0..order {
                falling *= (0.5 - j as f64).abs();
            }
            let bound = falling * floor.powf(1. - 2. * order as f64);
            record(
                report,
                &format!("sigma_reg_derivative_{order}"),
                bound,
                "(2n-3)!!/(2^n m^(2n-1))",
                &["lem-sigma-reg-derivative-bounds"],
                &format!("derivative order={order},floor={floor}; nonnegative variance"),
            );
            for var in [0.0_f64, 1e-12, 0.01, 1., 100.] {
                let actual = falling * (var + floor.powi(2)).powf(0.5 - order as f64);
                check(
                    report,
                    "sigma_derivative_bound",
                    if order == 1 {
                        &[
                            "lem-sigma-reg-derivative-bounds",
                            "lem-sigma-patch-derivative-bound",
                        ]
                    } else {
                        &["lem-sigma-reg-derivative-bounds"]
                    },
                    "analytic derivatives of the configured quadratic scale",
                    actual,
                    bound,
                );
            }
        }
    }
    for alpha in [0.1_f64, 0.25, 0.5, 0.9, 1.] {
        for a in [0.0_f64, 1e-12, 0.1, 1., 10., 1e6] {
            for b in [0.0_f64, 0.01, 1., 100.] {
                check(
                    report,
                    "concave_jensen",
                    &["lem-inequality-toolbox"],
                    "nonnegative x, normalized weights 0.2,0.3,0.5",
                    0.3 * a.powf(alpha) + 0.5 * b.powf(alpha),
                    (0.3 * a + 0.5 * b).powf(alpha),
                );
                check(
                    report,
                    "squared_mean",
                    &["lem-inequality-toolbox"],
                    "finite signed X with normalized weights",
                    (-0.3 * a + 0.5 * b).powi(2),
                    0.3 * a * a + 0.5 * b * b,
                );
                check(
                    report,
                    "sqrt_subadditivity",
                    &["lem-inequality-toolbox"],
                    "nonnegative inputs",
                    (a + b).sqrt(),
                    a.sqrt() + b.sqrt(),
                );
                check(
                    report,
                    "power_subadditivity",
                    &["lem-subadditivity-power"],
                    "0<alpha<=1; nonnegative inputs",
                    (a + b).powf(alpha),
                    a.powf(alpha) + b.powf(alpha),
                );
            }
        }
        let input = CompositeInputs {
            walkers: 8,
            alive1: 5,
            diameter: 4.,
            status_penalty: 1.3,
            death_lipschitz: 0.7,
            boundary_exponent: alpha,
            perturbation_moment_squared: 0.02,
            failure_probability: 0.05,
            potential_error_coefficients: [0.4, 0.2, 0.1, 0.03],
            clone_value_lipschitz: 0.6,
        };
        let c = composite_coefficients(&input)?;
        report.conditional_inputs.push(input.clone());
        let scope = format!(
            "conditional source algebra with explicitly supplied moduli A1..A4,L_death,C_val; alpha_B={alpha}; not a fitted or certified swarm rate"
        );
        for (name, value, formula, labels) in [
            (
                "C_P",
                c.probability_linear,
                "N² C_struct/lambda_status+N C_val sqrt(2N B2)",
                vec!["lem-sub-bound-sum-total-cloning-probs"],
            ),
            (
                "H_P",
                c.probability_half,
                "N C_val sqrt(2N B1)",
                vec!["lem-sub-bound-sum-total-cloning-probs"],
            ),
            (
                "K_P",
                c.probability_offset,
                "2N+N C_val sqrt(2N B0)",
                vec!["lem-sub-bound-sum-total-cloning-probs"],
            ),
            (
                "C_clone_L",
                c.clone_linear,
                "3+3D² C_P/N",
                vec!["def-cloning-operator-continuity-coeffs-recorrected"],
            ),
            (
                "C_clone_H",
                c.clone_half,
                "3D² H_P/N",
                vec!["def-cloning-operator-continuity-coeffs-recorrected"],
            ),
            (
                "K_clone",
                c.clone_offset,
                "3D² K_P/N",
                vec!["def-cloning-operator-continuity-coeffs-recorrected"],
            ),
            (
                "C_status_H",
                c.status_holder,
                "L_death² N^(1-alpha_B)",
                vec!["def-final-status-change-coeffs"],
            ),
            (
                "K_status_var",
                c.status_variance,
                "N/2",
                vec!["def-final-status-change-coeffs"],
            ),
            (
                "B_M",
                c.perturbation_mean,
                "N M_pert²",
                vec!["def-perturbation-fluctuation-bounds-reproof"],
            ),
            (
                "B_S",
                c.perturbation_fluctuation,
                "D² sqrt(N/2 ln(2/delta))",
                vec!["def-perturbation-fluctuation-bounds-reproof"],
            ),
            (
                "K_pert",
                c.perturbation_offset,
                "6 B_M+6 B_S+delta N D²",
                vec!["lem-final-positional-displacement-bound"],
            ),
            (
                "C_Psi_L",
                c.final_linear,
                "3 C_clone_L",
                vec!["def-composite-continuity-coeffs-recorrected"],
            ),
            (
                "C_Psi_H",
                c.final_holder,
                "3b+A a^alpha+A b^alpha",
                vec!["def-composite-continuity-coeffs-recorrected"],
            ),
            (
                "K_Psi",
                c.final_offset,
                "3c+A c^alpha+K0",
                vec!["def-composite-continuity-coeffs-recorrected"],
            ),
            (
                "p_minus",
                c.small_exponent,
                "alpha_B/2",
                vec!["def-composite-continuity-coeffs-recorrected"],
            ),
            (
                "p_plus",
                c.large_exponent,
                "max(1/2,alpha_B)",
                vec!["def-composite-continuity-coeffs-recorrected"],
            ),
        ] {
            record(report, name, value, formula, &labels, &scope);
        }
        for v in [0.0_f64, 1e-12, 1e-6, 0.1, 1., 2., 100., 1e6] {
            let [b2, b1, b0] = c.potential_quadratic;
            let direct_prob = (8. * 8. * 2. / 4. / input.status_penalty) * v
                + 8. * input.clone_value_lipschitz * (16. * (b2 * v * v + b1 * v + b0)).sqrt()
                + 16.;
            let separated_prob =
                c.probability_linear * v + c.probability_half * v.sqrt() + c.probability_offset;
            check(
                report,
                "clone_probability_expansion",
                &["lem-sub-bound-sum-total-cloning-probs"],
                &scope,
                direct_prob,
                separated_prob,
            );
            let bv = c.clone_linear * v + c.clone_half * v.sqrt() + c.clone_offset;
            let stage = 3. * bv + c.composite_a * bv.powf(alpha) + c.composite_k0;
            let expanded = 3. * c.clone_linear * v
                + 3. * c.clone_half * v.sqrt()
                + c.composite_a * c.clone_linear.powf(alpha) * v.powf(alpha)
                + c.composite_a * c.clone_half.powf(alpha) * v.powf(alpha / 2.)
                + c.final_offset;
            let simplified = c.final_linear * v
                + c.final_holder
                    * v.powf(if v <= 1. {
                        c.small_exponent
                    } else {
                        c.large_exponent
                    })
                + c.final_offset;
            check(
                report,
                "composite_fractional_expansion",
                &[
                    "thm-swarm-update-operator-continuity-recorrected",
                    "lem-subadditivity-power",
                ],
                &scope,
                stage,
                expanded,
            );
            check(
                report,
                "composite_small_large_unification",
                &[
                    "def-composite-continuity-coeffs-recorrected",
                    "lem-sub-unify-holder-terms",
                ],
                &scope,
                expanded,
                simplified,
            );
            let a_sum = 3. * c.clone_half
                + c.composite_a * c.clone_linear.powf(alpha)
                + c.composite_a * c.clone_half.powf(alpha);
            let fractional = 3. * c.clone_half * v.powf(0.5)
                + c.composite_a * c.clone_linear.powf(alpha) * v.powf(alpha)
                + c.composite_a * c.clone_half.powf(alpha) * v.powf(alpha / 2.);
            check(
                report,
                "global_power_unification",
                &["lem-sub-unify-holder-terms"],
                &scope,
                fractional,
                a_sum
                    * (if v <= 1. {
                        1.
                    } else {
                        v.powf(c.large_exponent)
                    }),
            );
        }
    }
    Ok(())
}

pub fn coefficient_validation_cases() -> Result<CoefficientReport> {
    let mut report=CoefficientReport{computed:Vec::new(),checks:Vec::new(),conditional_inputs:Vec::new(),unavailable:BTreeMap::new(),scope:"Native Rust standardization, positive maps and clone gates on bounded scalar fixtures; all explicit source coefficient families and inequality algebra are evaluated. Conditional composite inputs are supplied hypotheses, not certificates of their validity for a landscape or of geometric ergodicity.".into()};
    report.unavailable.insert("unconditional_A1_A2_A3_A4".into(),"The generic stochastic potential bound names state-dependent A1..A4 without assigning universal numerical values. Composite tests explicitly supply them and do not certify their hypotheses for an interacting trajectory.".into());
    report.unavailable.insert("full_swarm_convergence_rate".into(),"The first three chapters' one-step coefficient bounds do not establish a complete long-time convergence rate; OU gamma is only its thermostat substep damping.".into());
    standardization_cases(&mut report)?;
    sasaki_composite_cases(&mut report)?;
    potential_cases(&mut report)?;
    inequality_and_composite_cases(&mut report)?;
    require(
        report.computed.iter().all(|c| c.value.is_finite()),
        "nonfinite recorded coefficient",
    )?;
    Ok(report)
}
