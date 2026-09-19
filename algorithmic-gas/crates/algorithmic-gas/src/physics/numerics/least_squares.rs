//! Correlated linear least squares with the full parameter covariance.
//! Normal equations are solved on their unit-diagonal scaling, so a fit does
//! not depend on the units of the design columns. A collinear design is an
//! error; no pseudo-inverse is taken.
use super::Whitener;
use crate::{
    GasError, Result, error::require, physics::qft::math::ridentity,
    tessellation::linalg::lu_solve_multi,
};

#[derive(Clone, Debug, PartialEq)]
pub struct LinearFit {
    pub beta: Vec<f64>,
    /// `[p, p]`, `(Xᵀ C⁻¹ X)⁻¹`.
    pub covariance: Vec<f64>,
    pub chi2: f64,
    pub dof: usize,
}

/// Symmetric inverse of a symmetric positive definite `[p, p]` normal matrix.
/// The system solved is `d a d` with `d_j = a_jj^{-1/2}`, whose inverse has
/// the variance inflation factors on its diagonal: each is at least 1 and
/// grows without bound as a column becomes a combination of the others. A
/// nonpositive diagonal, a singular pivot or an inflation above `1e13` is
/// `GasError::Numerical`.
pub(super) fn normal_inverse(a: &[f64], p: usize) -> Result<Vec<f64>> {
    debug_assert!(a.len() == p * p);
    let singular = || GasError::Numerical("singular least-squares normal matrix".into());
    let d: Vec<f64> = (0..p).map(|j| 1. / a[j * p + j].sqrt()).collect();
    if !d.iter().all(|x| x.is_finite() && *x > 0.) {
        return Err(singular());
    }
    let mut scaled: Vec<f64> = (0..p * p)
        .map(|k| 0.5 * (a[k] + a[k % p * p + k / p]) * d[k / p] * d[k % p])
        .collect();
    let mut z = ridentity(p);
    lu_solve_multi(&mut scaled, &mut z, p, p).map_err(|_| singular())?;
    if !(0..p).all(|j| z[j * p + j].is_finite() && (0.5..=1e13).contains(&z[j * p + j])) {
        return Err(singular());
    }
    Ok((0..p * p)
        .map(|k| 0.5 * (z[k] + z[k % p * p + k / p]) * d[k / p] * d[k % p])
        .collect())
}

/// Minimize `|W (y − X β)|²` for a `[rows, parameters]` design `X`, solving the
/// normal equations with `tessellation::linalg::lu_solve_multi`. A singular
/// system is `GasError::Numerical`. `dof = rows − parameters`; a whitener of
/// deficient rank deflates `chi2` without changing `dof`.
pub fn linear_fit(
    design: &[f64],
    y: &[f64],
    whitener: &Whitener,
    parameters: usize,
) -> Result<LinearFit> {
    let (n, p) = (whitener.n(), parameters);
    require(
        (1..=64).contains(&p)
            && (p..=4096).contains(&n)
            && y.len() == n
            && design.len() == n * p
            && design.iter().chain(y).all(|x| x.is_finite()),
        "linear fit needs 1..=64 parameters, at most 4096 rows and at least one per parameter, \
         and a finite design and data of the whitener's size",
    )?;
    let x = whitener.apply_columns(design, p);
    let t = whitener.apply(y);
    let mut normal = vec![0.; p * p];
    let mut moment = vec![0.; p];
    for (row, &target) in x.chunks_exact(p).zip(&t) {
        for a in 0..p {
            moment[a] += row[a] * target;
            for b in a..p {
                normal[a * p + b] += row[a] * row[b];
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            normal[a * p + b] = normal[b * p + a];
        }
    }
    let covariance = normal_inverse(&normal, p)?;
    let beta: Vec<f64> = covariance
        .chunks_exact(p)
        .map(|row| row.iter().zip(&moment).map(|(c, m)| c * m).sum())
        .collect();
    let chi2 = x
        .chunks_exact(p)
        .zip(&t)
        .map(|(row, &target)| {
            let r = target - row.iter().zip(&beta).map(|(x, b)| x * b).sum::<f64>();
            r * r
        })
        .sum::<f64>();
    if !beta.iter().chain(&covariance).all(|v| v.is_finite()) || !chi2.is_finite() {
        return Err(GasError::Numerical("linear fit overflow".into()));
    }
    Ok(LinearFit {
        beta,
        covariance,
        chi2,
        dof: n - p,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normal_inverse_matches_the_closed_form_in_any_column_units() {
        let a = [4., 2., 2., 3.];
        let z = normal_inverse(&a, 2).unwrap();
        for (got, want) in z.iter().zip([0.375, -0.25, -0.25, 0.5]) {
            assert!((got - want).abs() < 1e-15);
        }
        let s = [1e9, 1e-7];
        let scaled: Vec<f64> = (0..4).map(|k| a[k] * s[k / 2] * s[k % 2]).collect();
        let w = normal_inverse(&scaled, 2).unwrap();
        for k in 0..4 {
            assert!((w[k] * s[k / 2] * s[k % 2] - z[k]).abs() < 1e-14);
        }
        assert_eq!(w[1].to_bits(), w[2].to_bits());
    }
    #[test]
    fn collinear_and_nonpositive_normal_matrices_are_explicit_numerical_failures() {
        let numerical = |a: &[f64], p| matches!(normal_inverse(a, p), Err(GasError::Numerical(_)));
        assert!(numerical(&[1., 2., 2., 4.], 2));
        assert!(numerical(&[1., 3., 3., 9. + 1e-14], 2));
        assert!(numerical(&[0., 0., 0., 1.], 2));
        assert!(numerical(&[-1.], 1));
        assert!(numerical(&[f64::NAN], 1));
        assert!(numerical(&[1., 0., 0., f64::INFINITY], 2));
    }
}
