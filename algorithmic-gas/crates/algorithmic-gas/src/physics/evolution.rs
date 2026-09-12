//! Conditional replica estimates for the exact discrete process. Sample keys
//! are recorded and checked across calibration and independent validation sets.
use crate::{Result, error::require};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplicaSample {
    pub key: u64,
    pub increment: Vec<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricEvolutionReport {
    pub calibration_keys: Vec<u64>,
    pub validation_keys: Vec<u64>,
    pub conditional_increment_mean: Vec<f64>,
    pub increment_covariance: Vec<f64>,
    pub mean_standard_error: Vec<f64>,
    pub validation_increment_mean: Vec<f64>,
    pub validation_residual: Vec<f64>,
    pub validation_standard_error: Vec<f64>,
    pub interpretation: String,
}
fn moments(samples: &[ReplicaSample]) -> Result<(Vec<f64>, Vec<f64>)> {
    require(
        samples.len() >= 2 && !samples[0].increment.is_empty() && samples[0].increment.len() <= 256,
        "replica sample count/dimension",
    )?;
    let d = samples[0].increment.len();
    let mut mean = vec![0.; d];
    let mut m2 = vec![0.; d * d];
    for (n, sample) in samples.iter().enumerate() {
        require(
            sample.increment.len() == d && sample.increment.iter().all(|x| x.is_finite()),
            "replica increment shape/finiteness",
        )?;
        let delta: Vec<_> = sample
            .increment
            .iter()
            .zip(&mean)
            .map(|(x, m)| x - m)
            .collect();
        for i in 0..d {
            mean[i] += delta[i] / (n + 1) as f64;
        }
        for i in 0..d {
            for j in 0..d {
                m2[i * d + j] += delta[i] * (sample.increment[j] - mean[j]);
            }
        }
    }
    for v in &mut m2 {
        *v /= (samples.len() - 1) as f64;
    }
    Ok((mean, m2))
}
pub fn metric_evolution(
    calibration: &[ReplicaSample],
    validation: &[ReplicaSample],
) -> Result<MetricEvolutionReport> {
    let mut keys = BTreeSet::new();
    require(
        calibration
            .iter()
            .chain(validation)
            .all(|s| keys.insert(s.key)),
        "replica keys must be unique and validation independent of calibration",
    )?;
    let (mean, covariance) = moments(calibration)?;
    let (validation_mean, validation_cov) = moments(validation)?;
    let d = mean.len();
    require(
        validation_mean.len() == d,
        "validation observable dimension",
    )?;
    let stderr: Vec<_> = (0..d)
        .map(|i| (covariance[i * d + i].max(0.) / calibration.len() as f64).sqrt())
        .collect();
    Ok(MetricEvolutionReport{calibration_keys:calibration.iter().map(|s|s.key).collect(),validation_keys:validation.iter().map(|s|s.key).collect(),validation_residual:validation_mean.iter().zip(&mean).map(|(v,m)|v-m).collect(),validation_standard_error:(0..d).map(|i|(stderr[i]*stderr[i]+validation_cov[i*d+i].max(0.)/validation.len() as f64).sqrt()).collect(),mean_standard_error:stderr,conditional_increment_mean:mean,increment_covariance:covariance,validation_increment_mean:validation_mean,interpretation:"conditional mean and covariance of one complete discrete update from disjoint replica pools".into()})
}
