//! Small dense kernels for d <= 4 geometry and p <= 14 local fits. Row-major
//! slices, generic over the run dtype, no allocation-free tricks: clarity and
//! one deterministic arithmetic path matter more than the last nanosecond.
use crate::{GasError, Real, Result, physics::geometry::symmetric_eigen};

/// Q diag(l) Q^T, assembled once per symmetric entry.
pub fn compose<T: Real>(q: &[T], l: &[T], d: usize) -> Vec<T> {
    let mut out = vec![T::ZERO; d * d];
    for i in 0..d {
        for j in i..d {
            let value = (0..d).fold(T::ZERO, |s, k| s + q[i * d + k] * l[k] * q[j * d + k]);
            out[i * d + j] = value;
            out[j * d + i] = value;
        }
    }
    out
}
/// x^T A y.
pub fn bilinear<T: Real>(a: &[T], x: &[T], y: &[T]) -> T {
    let d = x.len();
    let mut s = T::ZERO;
    for i in 0..d {
        let mut row = T::ZERO;
        for j in 0..d {
            row = row + a[i * d + j] * y[j];
        }
        s = s + x[i] * row;
    }
    s
}
pub fn mat_vec<T: Real>(a: &[T], x: &[T]) -> Vec<T> {
    let d = x.len();
    (0..d)
        .map(|i| (0..d).fold(T::ZERO, |s, j| s + a[i * d + j] * x[j]))
        .collect()
}

/// Spectral data of a symmetric positive metric, with the eigenvalues a clamp
/// or a sign repair had to move: a spectrum that was already a positive
/// definite metric inside its bounds must be distinguishable from one that was
/// made into one.
#[derive(Clone, Debug, PartialEq)]
pub struct SpdSpectrum<T: Real> {
    pub eigenvalues: Vec<T>,
    pub eigenvectors: Vec<T>,
    /// Per eigenvalue: the raw value was nonpositive, or a bound moved it.
    pub clipped: Vec<bool>,
}
impl<T: Real> SpdSpectrum<T> {
    pub fn dimension(&self) -> usize {
        self.eigenvalues.len()
    }
    /// True when the raw spectrum was not a positive definite metric inside
    /// the clamp, so the returned one is a repair and not a measurement.
    pub fn repaired(&self) -> bool {
        self.clipped.iter().any(|&moved| moved)
    }
    pub fn matrix(&self) -> Vec<T> {
        compose(&self.eigenvectors, &self.eigenvalues, self.dimension())
    }
    pub fn power(&self, p: f64, floor: T) -> Vec<T> {
        let l: Vec<T> = self
            .eigenvalues
            .iter()
            .map(|&v| v.max(floor).powf(T::from_f64(p)))
            .collect();
        compose(&self.eigenvectors, &l, self.dimension())
    }
    pub fn determinant(&self) -> T {
        self.eigenvalues.iter().fold(T::ONE, |s, &v| s * v)
    }
    pub fn log_determinant(&self, floor: T) -> T {
        self.eigenvalues
            .iter()
            .fold(T::ZERO, |s, &v| s + v.max(floor).ln())
    }
}
/// Eigen-clamp of a symmetric matrix: eigenvalues limited to [min, max]. An
/// eigenvalue the clamp moved, and every nonpositive one, is reported through
/// `clipped`: an indefinite matrix leaves here positive, and the caller, not
/// this kernel, decides whether that repair is admissible.
pub fn clamp_spectrum<T: Real>(
    a: &[T],
    d: usize,
    min_eig: Option<T>,
    max_eig: Option<T>,
) -> Result<SpdSpectrum<T>> {
    let (mut l, q) = symmetric_eigen(a, d)?;
    let mut clipped = vec![false; d];
    for (v, moved) in l.iter_mut().zip(&mut clipped) {
        let raw = *v;
        if let Some(lo) = min_eig {
            *v = v.max(lo);
        }
        if let Some(hi) = max_eig {
            *v = v.min(hi);
        }
        *moved = *v != raw || raw <= T::ZERO;
    }
    Ok(SpdSpectrum {
        eigenvalues: l,
        eigenvectors: q,
        clipped,
    })
}
/// Moore-Penrose inverse of a symmetric PSD matrix followed by an eigenvalue
/// clamp, from one eigensolve. Eigenvalues at or below `rcond * lambda_max`
/// invert to zero, the pseudo-inverse convention for a rank-deficient input.
/// A negative eigenvalue inverts to a negative one that the lower bound then
/// turns positive, which would hide the sign: `clipped` reports it, together
/// with every null direction and every eigenvalue a bound moved.
pub fn pinv_clamped<T: Real>(
    c: &[T],
    d: usize,
    rcond: T,
    min_eig: Option<T>,
    max_eig: Option<T>,
) -> Result<SpdSpectrum<T>> {
    let (l, q) = symmetric_eigen(c, d)?;
    let top = l.iter().fold(T::ZERO, |s, v| s.max(v.abs()));
    let cutoff = rcond * top;
    let mut clipped = vec![false; d];
    let eigenvalues = l
        .into_iter()
        .zip(&mut clipped)
        .map(|(v, moved)| {
            let null = v.abs() <= cutoff;
            let raw = if null { T::ZERO } else { T::ONE / v };
            let mut inv = raw;
            if let Some(lo) = min_eig {
                inv = inv.max(lo);
            }
            if let Some(hi) = max_eig {
                inv = inv.min(hi);
            }
            *moved = null || v < T::ZERO || inv != raw;
            inv
        })
        .collect();
    Ok(SpdSpectrum {
        eigenvalues,
        eigenvectors: q,
        clipped,
    })
}

/// In-place LU with partial pivoting: solves A X = B for `nrhs` right-hand
/// sides stored as the columns of the row-major n x nrhs matrix `b`.
pub fn lu_solve_multi<T: Real>(a: &mut [T], b: &mut [T], n: usize, nrhs: usize) -> Result<()> {
    debug_assert!(a.len() == n * n && b.len() == n * nrhs);
    for k in 0..n {
        let mut pivot = k;
        let mut best = a[k * n + k].abs();
        for r in k + 1..n {
            let v = a[r * n + k].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best == T::ZERO || !best.is_finite() {
            return Err(GasError::Numerical("singular local linear system".into()));
        }
        if pivot != k {
            for c in 0..n {
                a.swap(k * n + c, pivot * n + c);
            }
            for c in 0..nrhs {
                b.swap(k * nrhs + c, pivot * nrhs + c);
            }
        }
        let diagonal = a[k * n + k];
        for r in k + 1..n {
            let factor = a[r * n + k] / diagonal;
            if factor == T::ZERO {
                continue;
            }
            a[r * n + k] = T::ZERO;
            for c in k + 1..n {
                a[r * n + c] = a[r * n + c] - factor * a[k * n + c];
            }
            for c in 0..nrhs {
                b[r * nrhs + c] = b[r * nrhs + c] - factor * b[k * nrhs + c];
            }
        }
    }
    for k in (0..n).rev() {
        for c in 0..nrhs {
            let mut s = b[k * nrhs + c];
            for j in k + 1..n {
                s = s - a[k * n + j] * b[j * nrhs + c];
            }
            b[k * nrhs + c] = s / a[k * n + k];
        }
    }
    Ok(())
}
/// Cayley transform of a skew matrix applied to v: (I - A)^{-1} (I + A) v.
/// Orthogonal for every skew A, so it rotates without changing |v|.
pub fn cayley_apply<T: Real>(a: &[T], v: &[T]) -> Result<Vec<T>> {
    let d = v.len();
    let mut lhs = vec![T::ZERO; d * d];
    let mut rhs = vec![T::ZERO; d];
    for i in 0..d {
        let mut s = v[i];
        for j in 0..d {
            lhs[i * d + j] = if i == j { T::ONE } else { T::ZERO } - a[i * d + j];
            s = s + a[i * d + j] * v[j];
        }
        rhs[i] = s;
    }
    lu_solve_multi(&mut lhs, &mut rhs, d, 1)?;
    Ok(rhs)
}
/// Frobenius norm / sqrt(2): the rotation rate of a skew matrix in 2D and 3D.
pub fn skew_rate<T: Real>(a: &[T]) -> T {
    (a.iter().fold(T::ZERO, |s, &x| s + x * x) / T::from_f64(2.)).sqrt()
}

/// One-sided Jacobi singular values (descending) and right singular vectors
/// (rows of the returned cols x cols matrix) of a rows x cols matrix. Resolves
/// small singular values to relative precision, unlike an eigensolve of A^T A.
pub fn hestenes_svd(a: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut u = a.to_vec();
    let mut v = vec![0.; cols * cols];
    for k in 0..cols {
        v[k * cols + k] = 1.;
    }
    for _ in 0..60 {
        let mut rotated = false;
        for p in 0..cols {
            for q in p + 1..cols {
                let (mut alpha, mut beta, mut gamma) = (0., 0., 0.);
                for r in 0..rows {
                    let (x, y) = (u[r * cols + p], u[r * cols + q]);
                    alpha += x * x;
                    beta += y * y;
                    gamma += x * y;
                }
                if gamma == 0. || gamma.abs() <= 1e-15 * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2. * gamma);
                let t = zeta.signum() / (zeta.abs() + (1. + zeta * zeta).sqrt());
                let t = if zeta == 0. { 1. } else { t };
                let c = 1. / (1. + t * t).sqrt();
                let s = c * t;
                for r in 0..rows {
                    let (x, y) = (u[r * cols + p], u[r * cols + q]);
                    u[r * cols + p] = c * x - s * y;
                    u[r * cols + q] = s * x + c * y;
                }
                for r in 0..cols {
                    let (x, y) = (v[r * cols + p], v[r * cols + q]);
                    v[r * cols + p] = c * x - s * y;
                    v[r * cols + q] = s * x + c * y;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let mut order: Vec<(f64, usize)> = (0..cols)
        .map(|k| {
            (
                (0..rows)
                    .map(|r| u[r * cols + k] * u[r * cols + k])
                    .sum::<f64>()
                    .sqrt(),
                k,
            )
        })
        .collect();
    order.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let sigma = order.iter().map(|x| x.0).collect();
    let mut basis = vec![0.; cols * cols];
    for (row, &(_, k)) in order.iter().enumerate() {
        for c in 0..cols {
            basis[row * cols + c] = v[c * cols + k];
        }
    }
    (sigma, basis)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spd() -> Vec<f64> {
        vec![4., 1., 0.5, 1., 3., 0.25, 0.5, 0.25, 2.]
    }
    #[test]
    fn pinv_of_full_rank_matrix_is_its_inverse() {
        let a = spd();
        let s = pinv_clamped(&a, 3, 1e-15, None, None).unwrap();
        assert!(!s.repaired() && s.clipped == [false; 3]);
        let g = s.matrix();
        for i in 0..3 {
            for j in 0..3 {
                let v: f64 = (0..3).map(|k| a[i * 3 + k] * g[k * 3 + j]).sum();
                assert!((v - if i == j { 1. } else { 0. }).abs() < 1e-13);
            }
        }
    }
    #[test]
    fn pinv_zeroes_null_directions_then_clamps() {
        let a = vec![2., 0., 0., 0.];
        let s = pinv_clamped(&a, 2, 1e-12, Some(1e-6), None).unwrap();
        let mut l = s.eigenvalues.clone();
        l.sort_by(f64::total_cmp);
        assert_eq!(l, vec![1e-6, 0.5]);
        assert_eq!(s.clipped, [false, true]);
        assert!((s.log_determinant(1e-300) - (0.5e-6f64).ln()).abs() < 1e-12);
    }
    #[test]
    fn an_indefinite_input_is_reported_and_never_silently_positive() {
        // 1/v for v < 0 is negative; the lower bound makes it positive.
        let s = pinv_clamped(&[2., 0., 0., -4.], 2, 1e-12, Some(1e-6), None).unwrap();
        assert_eq!(s.eigenvalues, [0.5, 1e-6]);
        assert_eq!(s.clipped, [false, true]);
        assert!(s.repaired());
        // The same for a metric given directly, and a clamp that binds above.
        let s = clamp_spectrum(&[2., 0., 0., -3.], 2, Some(1e-6), None).unwrap();
        assert_eq!(
            (s.eigenvalues, s.clipped),
            (vec![2., 1e-6], vec![false, true])
        );
        let s = clamp_spectrum(&spd(), 3, None, Some(1.)).unwrap();
        assert!(s.repaired() && s.eigenvalues.iter().all(|&v| v <= 1.));
    }
    #[test]
    fn lu_solves_multiple_right_hand_sides() {
        let mut a = vec![0., 2., 1., 1., 1., 0., 3., 0., 1.];
        let original = a.clone();
        let x = [1., -2., 0.5, 3., 0.25, -1.];
        let mut b = vec![0.; 6];
        for i in 0..3 {
            for c in 0..2 {
                b[i * 2 + c] = (0..3).map(|j| original[i * 3 + j] * x[j * 2 + c]).sum();
            }
        }
        lu_solve_multi(&mut a, &mut b, 3, 2).unwrap();
        assert!(b.iter().zip(&x).all(|(u, v)| (u - v).abs() < 1e-14));
        assert!(lu_solve_multi(&mut [1., 2., 2., 4.], &mut [1., 1.], 2, 1).is_err());
    }
    #[test]
    fn cayley_preserves_the_norm() {
        let a = [0., 0.3, -0.2, -0.3, 0., 0.7, 0.2, -0.7, 0.];
        let v = [1., -2., 0.5];
        let w = cayley_apply(&a, &v).unwrap();
        let n = |x: &[f64]| x.iter().map(|t| t * t).sum::<f64>();
        assert!((n(&w) - n(&v)).abs() < 1e-14);
        assert!((skew_rate(&a) - (0.09f64 + 0.04 + 0.49).sqrt()).abs() < 1e-15);
    }
    #[test]
    fn svd_resolves_rank_and_small_singular_values() {
        // Points on a line in 3D plus a 1e-9 wobble orthogonal to the line parameter.
        let wobble = [1e-9, -1e-9, 0., 0., -1e-9, 1e-9];
        let mut a = Vec::new();
        for (i, w) in wobble.into_iter().enumerate() {
            let t = i as f64 - 2.5;
            a.extend([t, 2. * t, w]);
        }
        let (s, basis) = hestenes_svd(&a, 6, 3);
        // The exact third singular value is zero; roundoff leaves O(eps * sigma_0).
        assert!(s[0] > 1. && (s[1] - 2e-9).abs() < 1e-13 && s[2] < 1e-14);
        let axis = [basis[0], basis[1], basis[2]];
        assert!((axis[1] / axis[0] - 2.).abs() < 1e-12 && axis[2].abs() < 1e-9);
    }
}
