//! Special functions for fit quality.
use std::f64::consts::{PI, TAU};

/// `ln Γ(x)` (Lanczos); `None` unless `x` is finite and positive, and where
/// the value overflows.
pub fn ln_gamma(x: f64) -> Option<f64> {
    (x.is_finite() && x > 0.)
        .then(|| lanczos(x))
        .filter(|value| value.is_finite())
}

/// Lanczos approximation with `g = 7` and nine coefficients, reflected below
/// one half. Caller contract: `x` is finite and positive.
fn lanczos(x: f64) -> f64 {
    if x < 0.5 {
        return (PI / (PI * x).sin()).ln() - lanczos(1. - x);
    }
    let x = x - 1.;
    let s = [
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7,
    ]
    .iter()
    .enumerate()
    .fold(0.9999999999998099, |s, (i, c)| s + c / (x + (i + 1) as f64));
    let t = x + 7.5;
    0.5 * TAU.ln() + (x + 0.5) * t.ln() - t + s.ln()
}

/// Regularized upper incomplete gamma function `Q(a, x)` (series below
/// `a + 1`, continued fraction above); `None` outside `a > 0`, `x ≥ 0`, where
/// `ln Γ(a)` overflows, and when an expansion does not converge.
pub fn gamma_q(a: f64, x: f64) -> Option<f64> {
    if !(a.is_finite() && a > 0. && x.is_finite() && x >= 0.) {
        return None;
    }
    if x == 0. {
        return Some(1.);
    }
    let scale = (-x + a * x.ln() - ln_gamma(a)?).exp();
    let q = if x < a + 1. {
        let (mut term, mut sum) = (1. / a, 1. / a);
        let mut converged = false;
        for n in 1..100_000 {
            term *= x / (a + n as f64);
            sum += term;
            if term < 1e-16 * sum {
                converged = true;
                break;
            }
        }
        converged.then_some(1. - scale * sum)
    } else {
        // Modified Lentz evaluation; `floor` replaces a vanishing denominator.
        let floor = 1e-300;
        let mut b = x + 1. - a;
        let (mut c, mut d) = (1. / floor, 1. / b);
        let mut h = d;
        let mut converged = false;
        for i in 1..100_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.;
            d = an * d + b;
            if d.abs() < floor {
                d = floor;
            }
            c = b + an / c;
            if c.abs() < floor {
                c = floor;
            }
            d = 1. / d;
            h *= d * c;
            if (d * c - 1.).abs() <= f64::EPSILON {
                converged = true;
                break;
            }
        }
        converged.then_some(scale * h)
    };
    q.filter(|q| q.is_finite()).map(|q| q.clamp(0., 1.))
}

/// Probability that a χ² variable with `dof` degrees of freedom exceeds
/// `chi2`; `None` for `dof = 0` or a nonfinite or negative χ².
pub fn chi2_q(chi2: f64, dof: usize) -> Option<f64> {
    (dof > 0)
        .then(|| gamma_q(dof as f64 / 2., chi2 / 2.))
        .flatten()
}
