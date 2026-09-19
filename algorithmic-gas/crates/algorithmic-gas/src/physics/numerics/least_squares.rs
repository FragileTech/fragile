//! Correlated linear least squares with the full parameter covariance.
use super::Whitener;
use crate::{GasError, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct LinearFit {
    pub beta: Vec<f64>,
    /// `[p, p]`, `(Xᵀ C⁻¹ X)⁻¹`.
    pub covariance: Vec<f64>,
    pub chi2: f64,
    pub dof: usize,
}

/// Minimize `|W (y − X β)|²` for a `[rows, parameters]` design `X`, solving the
/// normal equations with `tessellation::linalg::lu_solve_multi`. A singular
/// system is `GasError::Numerical`.
pub fn linear_fit(
    design: &[f64],
    y: &[f64],
    whitener: &Whitener,
    parameters: usize,
) -> Result<LinearFit> {
    let _ = (design, y, whitener, parameters);
    Err(GasError::Capability(
        "pending: numerics::least_squares".into(),
    ))
}
