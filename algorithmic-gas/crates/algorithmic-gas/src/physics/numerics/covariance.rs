//! Conditioned inverse square root of a covariance and the correlated χ².
use crate::{GasError, Result, error::require, physics::qft::math::symmetric_eigen};

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
        require(
            (1..=4096).contains(&n)
                && covariance.len() == n * n
                && svd_cut.is_finite()
                && (0. ..1.).contains(&svd_cut),
            "whitener needs a [n, n] covariance with n in 1..=4096 and svd_cut in [0, 1)",
        )?;
        let numerical = |message: &str| Err(GasError::Numerical(message.into()));
        let d: Vec<f64> = (0..n).map(|i| covariance[i * n + i].sqrt()).collect();
        if !d.iter().all(|d| d.is_finite() && *d > 0.) {
            return numerical("covariance diagonal must be finite and positive");
        }
        let mut r = vec![0.; n * n];
        for i in 0..n {
            for j in 0..n {
                let (x, y) = (covariance[i * n + j], covariance[j * n + i]);
                let tol = 1e-9 * d[i] * d[j];
                if !(x.is_finite() && (x - y).abs() <= tol) {
                    return numerical("covariance must be finite and symmetric");
                }
                r[i * n + j] = if i == j {
                    1.
                } else {
                    0.5 * (x + y) / (d[i] * d[j])
                };
            }
        }
        let (values, vectors) = symmetric_eigen(&r, n);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| values[i].total_cmp(&values[j]));
        let residual = order
            .iter()
            .flat_map(|&k| (0..n).map(move |i| (i, k)))
            .map(|(i, k)| {
                let rv: f64 = (0..n).map(|j| r[i * n + j] * vectors[j * n + k]).sum();
                (rv - values[k] * vectors[i * n + k]).abs()
            })
            .fold(0., f64::max);
        if residual > 1e-10 * n as f64 {
            return numerical("covariance eigendecomposition did not converge");
        }
        let eigenvalues: Vec<f64> = order.iter().map(|&k| values[k]).collect();
        let floor = svd_cut.max(1e-12) * eigenvalues[n - 1];
        let mut transform = vec![0.; n * n];
        for (row, &k) in order.iter().enumerate() {
            let scale = values[k].max(floor).sqrt();
            for j in 0..n {
                transform[row * n + j] = vectors[j * n + k] / (scale * d[j]);
            }
        }
        if !transform.iter().all(|w| w.is_finite()) {
            return numerical("whitening transform overflow");
        }
        Ok(Self {
            n,
            transform,
            rank: eigenvalues.iter().filter(|v| **v >= floor).count(),
            eigenvalues,
            floor,
        })
    }
    /// Uncorrelated errors: `W = diag(1/σ)`.
    pub fn diagonal(errors: &[f64]) -> Result<Self> {
        let n = errors.len();
        require(
            (1..=4096).contains(&n),
            "whitener needs 1..=4096 uncorrelated errors",
        )?;
        if !errors.iter().all(|e| e.is_finite() && *e > 0.) {
            return Err(GasError::Numerical(
                "uncorrelated errors must be finite and positive".into(),
            ));
        }
        let mut transform = vec![0.; n * n];
        for (i, e) in errors.iter().enumerate() {
            transform[i * n + i] = 1. / e;
        }
        if !transform.iter().all(|w| w.is_finite()) {
            return Err(GasError::Numerical("whitening transform overflow".into()));
        }
        Ok(Self {
            n,
            transform,
            eigenvalues: vec![1.; n],
            floor: 0.,
            rank: n,
        })
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
    /// `W r` for a residual `[n]`.
    pub fn apply(&self, residual: &[f64]) -> Vec<f64> {
        self.apply_columns(residual, 1)
    }
    /// Whiten the rows of a `[n, columns]` design matrix.
    pub fn apply_columns(&self, matrix: &[f64], columns: usize) -> Vec<f64> {
        let n = self.n;
        assert!(
            matrix.len() == n * columns,
            "whitened matrix must be [n, columns]"
        );
        let mut out = vec![0.; n * columns];
        for (row, w) in self.transform.chunks_exact(n).enumerate() {
            for (j, w) in w.iter().enumerate() {
                for c in 0..columns {
                    out[row * columns + c] += w * matrix[j * columns + c];
                }
            }
        }
        out
    }
    pub fn chi2(&self, residual: &[f64]) -> f64 {
        self.apply(residual).iter().map(|x| x * x).sum()
    }
    /// `λ_max / max(λ_min, floor)`; `None` for an empty spectrum.
    pub fn condition(&self) -> Option<f64> {
        let (min, max) = (self.eigenvalues.first()?, self.eigenvalues.last()?);
        Some(max / min.max(self.floor))
    }
}

/// Rows and columns `indices` of a flat `[n, n]` matrix.
pub fn submatrix(matrix: &[f64], n: usize, indices: &[usize]) -> Vec<f64> {
    indices
        .iter()
        .flat_map(|&i| indices.iter().map(move |&j| matrix[i * n + j]))
        .collect()
}
