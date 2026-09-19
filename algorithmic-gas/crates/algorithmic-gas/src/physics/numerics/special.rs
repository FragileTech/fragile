//! Special functions for fit quality.

/// `ln Γ(x)` (Lanczos); `None` unless `x` is finite and positive.
pub fn ln_gamma(x: f64) -> Option<f64> {
    let _ = x;
    None
}

/// Regularized upper incomplete gamma function `Q(a, x)` (series below
/// `a + 1`, continued fraction above); `None` outside `a > 0`, `x ≥ 0`.
pub fn gamma_q(a: f64, x: f64) -> Option<f64> {
    let _ = (a, x);
    None
}

/// Probability that a χ² variable with `dof` degrees of freedom exceeds
/// `chi2`; `None` for `dof = 0` or a nonfinite or negative χ².
pub fn chi2_q(chi2: f64, dof: usize) -> Option<f64> {
    (dof > 0)
        .then(|| gamma_q(dof as f64 / 2., chi2 / 2.))
        .flatten()
}
