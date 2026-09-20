//! Generalized symmetric eigenproblem `A v = λ B v`.
//! The metric is reduced on its unit-diagonal scaling, so the rank cut and
//! the eigenvalues do not depend on the normalization of the basis. With a
//! deficient rank the pencil is solved on the retained directions of `B` only.
use crate::{
    GasError, Result,
    error::require,
    physics::qft::math::{ridentity, symmetric_eigen},
    tessellation::linalg::bilinear,
};

#[derive(Clone, Debug, PartialEq)]
pub struct GeneralizedEigen {
    /// Descending, `[rank]`.
    pub values: Vec<f64>,
    /// `[rank, n]`, row `k` is the eigenvector of `values[k]`, `vᵀ B v = 1`.
    /// The component of largest magnitude is positive.
    pub vectors: Vec<f64>,
    /// Directions of `B` kept: eigenvalues of `B_ij / sqrt(B_ii B_jj)` above
    /// `max(cut, 1e-12) · λ_max`.
    pub rank: usize,
}
impl GeneralizedEigen {
    /// `vₙᵀ a vₙ` of every retained eigenvector against a symmetric `[n, n]`
    /// matrix, `[rank]`. Since `vᵀ B v = 1`, this is the eigenvalue the
    /// pencil `(a, B)` carries along that one direction: held across a family
    /// of `a`, it follows one state instead of whatever sits at a place in
    /// the ordering of each member.
    pub fn project(&self, a: &[f64], n: usize) -> Vec<f64> {
        self.vectors
            .chunks_exact(n)
            .map(|v| bilinear(a, v, v))
            .collect()
    }
}

/// Eigenvalues and column eigenvectors of a symmetric `[n, n]` matrix whose
/// entries are of order one, the domain of the absolute rotation threshold of
/// `symmetric_eigen`. A decomposition that does not reproduce the matrix to
/// `1e-10` is `GasError::Numerical`.
fn spectrum(m: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let (values, q) = symmetric_eigen(m, n);
    let worst = (0..n * n)
        .map(|c| {
            let (i, k) = (c / n, c % n);
            let mq: f64 = (0..n).map(|j| m[i * n + j] * q[j * n + k]).sum();
            (mq - q[i * n + k] * values[k]).abs()
        })
        .fold(0., f64::max);
    if worst <= 1e-10 && m.iter().chain(&values).all(|x| x.is_finite()) {
        Ok((values, q))
    } else {
        Err(GasError::Numerical(
            "symmetric eigenproblem did not converge".into(),
        ))
    }
}
fn symmetric(m: &[f64], n: usize) -> bool {
    let tol = 2e-13 * m.iter().fold(0., |s: f64, x| s.max(x.abs()));
    (0..n).all(|i| (0..i).all(|j| (m[i * n + j] - m[j * n + i]).abs() <= tol))
}

/// `a`, `b` are `[n, n]` symmetric, `b` positive semidefinite with a positive
/// diagonal. `b` is reduced through the eigen-decomposition of
/// `b_ij / sqrt(b_ii b_jj)`; directions below `max(cut, 1e-12) · λ_max` of that
/// matrix are discarded. A nonpositive diagonal entry of `b` or an indefinite
/// `b` beyond roundoff is `GasError::Numerical`.
pub fn generalized_symmetric(a: &[f64], b: &[f64], n: usize, cut: f64) -> Result<GeneralizedEigen> {
    require(
        (1..=64).contains(&n)
            && a.len() == n * n
            && b.len() == n * n
            && a.iter().chain(b).all(|x| x.is_finite())
            && cut.is_finite()
            && (0. ..1.).contains(&cut)
            && symmetric(a, n)
            && symmetric(b, n),
        "generalized eigenproblem needs two finite symmetric matrices of 1..=64 rows \
         and a cut in [0, 1)",
    )?;
    let d: Vec<f64> = (0..n).map(|i| 1. / b[i * n + i].sqrt()).collect();
    if !d.iter().all(|x| x.is_finite() && *x > 0.) {
        return Err(GasError::Numerical(
            "metric of the generalized eigenproblem has a nonpositive diagonal".into(),
        ));
    }
    let metric: Vec<f64> = (0..n * n)
        .map(|c| (c / n, c % n))
        .map(|(i, j)| {
            if i == j {
                1.
            } else {
                0.5 * (b[i * n + j] + b[j * n + i]) * d[i] * d[j]
            }
        })
        .collect();
    let (beta, u) = spectrum(&metric, n)?;
    let top = beta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if beta.iter().any(|x| *x < -1e-8 * top) {
        return Err(GasError::Numerical(
            "indefinite metric of the generalized eigenproblem".into(),
        ));
    }
    let floor = cut.max(1e-12) * top;
    let keep: Vec<usize> = (0..n).filter(|&k| beta[k] > floor).collect();
    let rank = keep.len();
    // w is [n, rank] with wᵀ b w = 1.
    let w: Vec<f64> = (0..n)
        .flat_map(|i| keep.iter().map(move |&k| (i, k)))
        .map(|(i, k)| d[i] * u[i * n + k] / beta[k].sqrt())
        .collect();
    let aw: Vec<f64> = (0..n * rank)
        .map(|c| (c / rank, c % rank))
        .map(|(i, k)| {
            (0..n)
                .map(|j| 0.5 * (a[i * n + j] + a[j * n + i]) * w[j * rank + k])
                .sum()
        })
        .collect();
    let reduced: Vec<f64> = (0..rank * rank)
        .map(|c| (c / rank, c % rank))
        .map(|(k, l)| {
            (0..n)
                .map(|i| {
                    0.5 * (w[i * rank + k] * aw[i * rank + l] + w[i * rank + l] * aw[i * rank + k])
                })
                .sum()
        })
        .collect();
    // The reduced matrix decays with the eigenvalues; at unit scale the
    // absolute rotation threshold keeps the relative accuracy of small ones.
    let scale = reduced.iter().fold(0., |s: f64, x| s.max(x.abs()));
    let (mu, y) = if scale > 0. {
        let unit: Vec<f64> = reduced.iter().map(|x| x / scale).collect();
        spectrum(&unit, rank)?
    } else {
        (vec![0.; rank], ridentity(rank))
    };
    let mut order: Vec<usize> = (0..rank).collect();
    order.sort_by(|&x, &y| mu[y].total_cmp(&mu[x]).then(x.cmp(&y)));
    let values: Vec<f64> = order.iter().map(|&k| mu[k] * scale).collect();
    let mut vectors = Vec::with_capacity(rank * n);
    for &k in &order {
        let v: Vec<f64> = (0..n)
            .map(|i| (0..rank).map(|l| w[i * rank + l] * y[l * rank + k]).sum())
            .collect();
        let lead = v
            .iter()
            .fold(0., |s: f64, x| if x.abs() > s.abs() { *x } else { s });
        vectors.extend(v.iter().map(|x| if lead < 0. { -x } else { *x }));
    }
    if !values.iter().chain(&vectors).all(|x| x.is_finite()) {
        return Err(GasError::Numerical(
            "generalized eigenproblem overflow".into(),
        ));
    }
    Ok(GeneralizedEigen {
        values,
        vectors,
        rank,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn spectrum_reconstructs_a_unit_scale_matrix_and_refuses_a_nonfinite_one() {
        let m = [1., 0.5, 0.2, 0.5, 1., -0.3, 0.2, -0.3, 1.];
        let (values, q) = spectrum(&m, 3).unwrap();
        assert!((values.iter().sum::<f64>() - 3.).abs() < 1e-13);
        for (c, entry) in m.iter().enumerate() {
            let (i, j) = (c / 3, c % 3);
            let back: f64 = (0..3)
                .map(|k| q[i * 3 + k] * values[k] * q[j * 3 + k])
                .sum();
            assert!((back - entry).abs() < 1e-13);
        }
        assert!(matches!(
            spectrum(&[1., f64::NAN, f64::NAN, 1.], 2),
            Err(GasError::Numerical(_))
        ));
    }
    #[test]
    fn symmetry_is_judged_relative_to_the_largest_entry() {
        assert!(symmetric(&[1e9, 2., 2. + 1e-5, 3.], 2));
        assert!(!symmetric(&[1., 2., 2. + 1e-5, 3.], 2));
        assert!(symmetric(&[0.; 4], 2));
    }
}
