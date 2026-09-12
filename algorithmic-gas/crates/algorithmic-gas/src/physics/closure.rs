//! Held-out tests of a specified linear constitutive candidate. Fits do not
//! constitute a derivation; keys prevent the same observations testing themselves.
use super::thermodynamics::solve;
use crate::{Result, error::require};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClosureSample {
    pub key: u64,
    pub predictors: Vec<f64>,
    pub observed: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClosureReport {
    pub coefficients: Vec<f64>,
    pub training_count: usize,
    pub validation_count: usize,
    pub validation_rmse: f64,
    pub constant_baseline_rmse: f64,
    pub validation_bias: f64,
    pub validation_residuals: Vec<f64>,
    pub status: String,
}
/// Predictors must explicitly include any desired intercept. Standard errors
/// need an independently justified sampling model; this report only gives errors.
pub fn test_linear_closure(
    training: &[ClosureSample],
    validation: &[ClosureSample],
    ridge: f64,
) -> Result<ClosureReport> {
    require(
        !training.is_empty() && !validation.is_empty() && ridge.is_finite() && ridge >= 0.,
        "closure samples/ridge",
    )?;
    let p = training[0].predictors.len();
    require(
        p > 0 && p <= 256 && training.len() > p,
        "closure needs more training samples than coefficients",
    )?;
    let mut keys = BTreeSet::new();
    for s in training.iter().chain(validation) {
        require(
            keys.insert(s.key)
                && s.predictors.len() == p
                && s.observed.is_finite()
                && s.predictors.iter().all(|v| v.is_finite()),
            "closure samples must be finite, shape-consistent and disjoint",
        )?;
    }
    let mut gram = vec![0.; p * p];
    let mut rhs = vec![0.; p];
    for s in training {
        for i in 0..p {
            rhs[i] += s.predictors[i] * s.observed;
            for j in 0..p {
                gram[i * p + j] += s.predictors[i] * s.predictors[j];
            }
        }
    }
    for i in 0..p {
        gram[i * p + i] += ridge;
    }
    let coefficients = solve(gram, rhs, p)?;
    let baseline = training.iter().map(|s| s.observed).sum::<f64>() / training.len() as f64;
    let residuals: Vec<_> = validation
        .iter()
        .map(|s| {
            s.observed
                - s.predictors
                    .iter()
                    .zip(&coefficients)
                    .map(|(x, b)| x * b)
                    .sum::<f64>()
        })
        .collect();
    let n = validation.len() as f64;
    let rmse = (residuals.iter().map(|r| r * r).sum::<f64>() / n).sqrt();
    let constant = (validation
        .iter()
        .map(|s| (s.observed - baseline).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    Ok(ClosureReport {
        coefficients,
        training_count: training.len(),
        validation_count: validation.len(),
        validation_rmse: rmse,
        constant_baseline_rmse: constant,
        validation_bias: residuals.iter().sum::<f64>() / n,
        validation_residuals: residuals,
        status: "held-out constitutive evaluation".into(),
    })
}
