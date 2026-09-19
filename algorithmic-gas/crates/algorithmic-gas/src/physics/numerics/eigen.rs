//! Generalized symmetric eigenproblem `A v = λ B v`.
use crate::{GasError, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct GeneralizedEigen {
    /// Descending, `[rank]`.
    pub values: Vec<f64>,
    /// `[rank, n]`, row `k` is the eigenvector of `values[k]`, `vᵀ B v = 1`.
    pub vectors: Vec<f64>,
    /// Directions of `B` kept: eigenvalues above `cut · λ_max(B)`.
    pub rank: usize,
}

/// `a`, `b` are `[n, n]` symmetric, `b` positive semidefinite. `b` is reduced
/// through its own eigen-decomposition; directions below `cut · λ_max(b)` are
/// discarded. An indefinite `b` beyond roundoff is `GasError::Numerical`.
pub fn generalized_symmetric(a: &[f64], b: &[f64], n: usize, cut: f64) -> Result<GeneralizedEigen> {
    let _ = (a, b, n, cut);
    Err(GasError::Capability("pending: numerics::eigen".into()))
}
