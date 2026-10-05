//! Probability-normalized, permutation-invariant finite-population interfaces.
use algorithmic_gas::{GasError, Result};
use serde::{Deserialize, Serialize};
pub const SOURCE: &str = include_str!(
    "../../../../docs/source/2_fractal_gas/convergence_program/12_qsd_exchangeability_theory.md"
);
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Comparison {
    pub label: String,
    pub expression: String,
    pub lhs: f64,
    pub rhs: f64,
    pub equality: bool,
    pub allowance: f64,
    pub passed: bool,
    pub operands: serde_json::Value,
    pub scope: String,
}
impl Comparison {
    pub fn new(
        label: &str,
        expression: &str,
        lhs: f64,
        rhs: f64,
        equality: bool,
        operands: serde_json::Value,
        scope: &str,
    ) -> Self {
        let allowance = 2e-10 * (1. + lhs.abs() + rhs.abs());
        Self {
            label: label.into(),
            expression: expression.into(),
            lhs,
            rhs,
            equality,
            allowance,
            passed: lhs.is_finite()
                && rhs.is_finite()
                && if equality {
                    (lhs - rhs).abs() <= allowance
                } else {
                    lhs <= rhs + allowance
                },
            operands,
            scope: scope.into(),
        }
    }
}
pub fn collision_probability(n: usize, k: usize) -> Result<f64> {
    if n == 0 || k == 0 || k > n {
        return Err(GasError::Configuration("1 <= k <= N required".into()));
    }
    Ok(-(0..k)
        .map(|i| (-(i as f64) / (n as f64)).ln_1p())
        .sum::<f64>()
        .exp_m1())
}
pub fn union_bound(n: usize, k: usize) -> Result<f64> {
    collision_probability(n, k)?;
    Ok((k as f64 * (k - 1) as f64 / (2. * n as f64)).min(1.))
}
pub fn entropy_mse_bound(n: usize, b: f64, h: f64) -> Result<f64> {
    if n == 0 || !b.is_finite() || b < 0. || !h.is_finite() || h < 0. {
        return Err(GasError::Configuration(
            "finite nonnegative entropy/bound and positive N required".into(),
        ));
    }
    Ok(4. * b * b / n as f64 * (h + 0.5 * 2_f64.ln()))
}
pub fn distinct_covariance(n: usize, empirical: f64, diagonal: f64) -> Result<f64> {
    if n < 2 {
        return Err(GasError::Configuration("N >= 2 required".into()));
    }
    Ok((n as f64 * empirical - diagonal) / (n - 1) as f64)
}
pub fn gaussian_lsi_constant(theta: f64, curvature: f64) -> Result<f64> {
    if !theta.is_finite() || theta <= 0. || !curvature.is_finite() || curvature <= 0. {
        return Err(GasError::Configuration(
            "positive temperature and curvature required".into(),
        ));
    }
    Ok(theta.max(theta / curvature))
}
/// Exact exchangeable bounded joint tilt of independent Bernoulli(q) coordinates.
/// Count masses include the binomial multiplicity; total joint KL stays <= beta.
pub fn tilted_bernoulli(n: usize, q: f64, beta: f64) -> Result<serde_json::Value> {
    if n == 0 || n > 1024 || !(0. < q && q < 1.) || !beta.is_finite() || beta < 0. {
        return Err(GasError::Configuration(
            "valid finite Bernoulli parameters required".into(),
        ));
    }
    let mut logbin = 0.;
    let mut logs = vec![];
    for s in 0..=n {
        if s > 0 {
            logbin += ((n + 1 - s) as f64 / (s as f64)).ln();
        }
        logs.push(logbin + s as f64 * q.ln() + (n - s) as f64 * (-q).ln_1p());
    }
    let raw = logs
        .iter()
        .enumerate()
        .map(|(s, l)| (l + beta * (s as f64 / n as f64 - q).powi(2)).exp())
        .collect::<Vec<_>>();
    let z = raw.iter().sum::<f64>();
    let p = raw.iter().map(|x| x / z).collect::<Vec<_>>();
    let mean = p
        .iter()
        .enumerate()
        .map(|(s, p)| p * s as f64 / n as f64)
        .sum::<f64>();
    let second = p
        .iter()
        .enumerate()
        .map(|(s, p)| p * (s as f64 / n as f64).powi(2))
        .sum::<f64>();
    let distinct = p
        .iter()
        .enumerate()
        .map(|(s, p)| p * s as f64 * (s.saturating_sub(1)) as f64 / (n * (n - 1).max(1)) as f64)
        .sum::<f64>()
        - mean * mean;
    let entropy = p
        .iter()
        .enumerate()
        .map(|(s, p)| p * (beta * (s as f64 / n as f64 - q).powi(2) - z.ln()))
        .sum::<f64>();
    let mgf = p
        .iter()
        .enumerate()
        .map(|(s, _)| (logs[s] + n as f64 * (s as f64 / n as f64 - q).powi(2) / 4.).exp())
        .sum::<f64>();
    Ok(
        serde_json::json!({"N":n,"q":q,"beta":beta,"count_probabilities":p,"log_product_count_masses":logs,"normalizer":z,"mean":mean,"empirical_variance":second-mean*mean,"distinct_covariance":distinct,"diagonal_covariance":mean*(1.-mean),"total_relative_entropy":entropy.max(0.),"mse":second-2.*q*mean+q*q,"product_square_exponential_moment":mgf}),
    )
}
