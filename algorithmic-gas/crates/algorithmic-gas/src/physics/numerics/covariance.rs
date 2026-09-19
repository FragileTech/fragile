//! Conditioned inverse square root of a covariance and the correlated χ².
use crate::{GasError, Result};

/// `W` with `Wᵀ W = C̃⁻¹`, where `C̃` is `C` with the eigenvalues of its
/// correlation matrix `R_ij = C_ij / sqrt(C_ii C_jj)` raised to the floor
/// `max(svd_cut, 1e-12) · λ_max`. Flooring only enlarges variances, and the
/// unit diagonal keeps `qft::math::symmetric_eigen` inside its absolute
/// threshold whatever the scale of `C`.
#[derive(Clone, Debug, PartialEq)]
pub struct Whitener {
    n: usize,
    /// `[n, n]`, applied as `W r`.
    transform: Vec<f64>,
    /// Ascending eigenvalues of the correlation matrix, before flooring.
    eigenvalues: Vec<f64>,
    floor: f64,
    rank: usize,
}
impl Whitener {
    /// `covariance` is `[n, n]`, symmetric with a positive diagonal
    /// (`GasError::Numerical` otherwise); `svd_cut` in `[0, 1)`.
    pub fn new(covariance: &[f64], n: usize, svd_cut: f64) -> Result<Self> {
        let _ = (covariance, n, svd_cut);
        Err(GasError::Capability("pending: numerics::covariance".into()))
    }
    /// Uncorrelated errors: `W = diag(1/σ)`.
    pub fn diagonal(errors: &[f64]) -> Result<Self> {
        let _ = errors;
        Err(GasError::Capability("pending: numerics::covariance".into()))
    }
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }
    pub fn floor(&self) -> f64 {
        self.floor
    }
    /// Eigenvalues at or above the floor. With `rank < n` the χ² is deflated
    /// and its probability is no longer calibrated.
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn apply(&self, residual: &[f64]) -> Vec<f64> {
        let _ = (residual, &self.transform);
        vec![0.; self.n]
    }
    /// Whiten the rows of a `[n, columns]` design matrix.
    pub fn apply_columns(&self, matrix: &[f64], columns: usize) -> Vec<f64> {
        let _ = matrix;
        vec![0.; self.n * columns]
    }
    pub fn chi2(&self, residual: &[f64]) -> f64 {
        self.apply(residual).iter().map(|x| x * x).sum()
    }
    /// `λ_max / max(λ_min, floor)`; `None` for an empty spectrum.
    pub fn condition(&self) -> Option<f64> {
        None
    }
}

/// Rows and columns `indices` of a flat `[n, n]` matrix.
pub fn submatrix(matrix: &[f64], n: usize, indices: &[usize]) -> Vec<f64> {
    indices
        .iter()
        .flat_map(|&i| indices.iter().map(move |&j| matrix[i * n + j]))
        .collect()
}
