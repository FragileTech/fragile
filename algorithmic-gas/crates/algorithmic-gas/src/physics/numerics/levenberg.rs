//! Levenberg–Marquardt on whitened residuals with Gaussian priors.
use super::{Prior, Whitener};
use crate::{GasError, Result};

/// A vector-valued model `f(p)` compared with the data it is fitted to.
pub trait Model {
    fn parameters(&self) -> usize;
    /// Length of `out` in `evaluate`.
    fn outputs(&self) -> usize;
    fn evaluate(&self, p: &[f64], out: &mut [f64]);
    /// `out` is `[outputs, parameters]`. Central finite differences by default.
    fn jacobian(&self, p: &[f64], out: &mut [f64]) {
        let (m, k) = (self.outputs(), self.parameters());
        let mut q = p.to_vec();
        let (mut hi, mut lo) = (vec![0.; m], vec![0.; m]);
        for j in 0..k {
            let h = 1e-6 * p[j].abs().max(1e-6);
            q[j] = p[j] + h;
            self.evaluate(&q, &mut hi);
            q[j] = p[j] - h;
            self.evaluate(&q, &mut lo);
            q[j] = p[j];
            for i in 0..m {
                out[i * k + j] = (hi[i] - lo[i]) / (2. * h);
            }
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevenbergConfig {
    pub max_iterations: usize,
    /// Relative decrease of the augmented χ² that ends the iteration.
    pub tolerance: f64,
    pub initial_damping: f64,
}
impl Default for LevenbergConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-10,
            initial_damping: 1e-3,
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct NonlinearFit {
    pub p: Vec<f64>,
    /// `[k, k]`, inverse of the Gauss–Newton Hessian of the augmented χ².
    pub covariance: Vec<f64>,
    /// Augmented χ²: the data part plus `chi2_prior`.
    pub chi2: f64,
    pub chi2_prior: f64,
    /// Data points minus parameters that carry no Gaussian prior.
    pub dof: usize,
    /// χ² probability of the augmented χ² at `dof`; `None` when `dof = 0`.
    pub q: Option<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// Minimize `|W (data − f(p))|² + Σ ((p_j − μ_j)/σ_j)²` from `start`.
/// `priors` has one entry per parameter. A model that produces nonfinite
/// values at `start` is `GasError::Numerical`; failure to converge is reported
/// in `converged`, not as an error.
pub fn minimize(
    model: &dyn Model,
    data: &[f64],
    whitener: &Whitener,
    priors: &[Prior],
    start: &[f64],
    config: &LevenbergConfig,
) -> Result<NonlinearFit> {
    let _ = (model, data, whitener, priors, start, config);
    Err(GasError::Capability("pending: numerics::levenberg".into()))
}
