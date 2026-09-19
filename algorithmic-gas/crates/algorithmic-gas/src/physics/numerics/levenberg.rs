//! Levenberg–Marquardt on whitened residuals with Gaussian priors.
//! A prior is one more residual `(p_j − μ_j)/σ_j`, so the reported covariance
//! and χ² are those of the augmented problem. The start is the caller's and
//! the iteration is deterministic; a local minimum is not excluded.
use super::{Prior, Whitener, chi2_q, least_squares::normal_inverse};
use crate::{GasError, Result, error::require, tessellation::linalg::lu_solve_multi};

/// A vector-valued model `f(p)` compared with the data it is fitted to.
pub trait Model {
    fn parameters(&self) -> usize;
    /// Length of `out` in `evaluate`.
    fn outputs(&self) -> usize;
    fn evaluate(&self, p: &[f64], out: &mut [f64]);
    /// `out` is `[outputs, parameters]`. Central finite differences by default,
    /// with the step `1e-6 · max(|p_j|, 1)`: parameters are expected at unit
    /// scale, as logarithms are.
    fn jacobian(&self, p: &[f64], out: &mut [f64]) {
        let (m, k) = (self.outputs(), self.parameters());
        let mut q = p.to_vec();
        let (mut hi, mut lo) = (vec![0.; m], vec![0.; m]);
        for j in 0..k {
            let h = 1e-6 * p[j].abs().max(1.);
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
impl LevenbergConfig {
    pub fn validate(&self) -> Result<()> {
        require(
            (1..=10_000).contains(&self.max_iterations)
                && self.tolerance.is_finite()
                && self.tolerance >= 0.
                && self.initial_damping.is_finite()
                && self.initial_damping > 0.
                && self.initial_damping <= 1e15,
            "Levenberg-Marquardt needs 1..=10000 iterations, a finite nonnegative tolerance \
             and a damping in (0, 1e15]",
        )
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
    /// Damped iterations started; the closing undamped refinement is not counted.
    pub iterations: usize,
    pub converged: bool,
}

/// Residuals of the augmented problem at one parameter point.
struct Point {
    p: Vec<f64>,
    /// `W (data − f(p))`.
    whitened: Vec<f64>,
    chi2_data: f64,
    chi2_prior: f64,
}
impl Point {
    fn chi2(&self) -> f64 {
        self.chi2_data + self.chi2_prior
    }
}
struct Problem<'a> {
    model: &'a dyn Model,
    data: &'a [f64],
    whitener: &'a Whitener,
    priors: &'a [Prior],
}
impl Problem<'_> {
    fn point(&self, p: Vec<f64>) -> Point {
        let mut r = vec![0.; self.data.len()];
        self.model.evaluate(&p, &mut r);
        for (r, y) in r.iter_mut().zip(self.data) {
            *r = y - *r;
        }
        let whitened = self.whitener.apply(&r);
        let chi2_prior = p
            .iter()
            .zip(self.priors)
            .map(|(x, prior)| match prior {
                Prior::Flat => 0.,
                Prior::Gaussian { mean, sigma } => ((x - mean) / sigma).powi(2),
            })
            .sum();
        Point {
            chi2_data: whitened.iter().map(|x| x * x).sum(),
            chi2_prior,
            whitened,
            p,
        }
    }
    /// Gauss–Newton Hessian `JᵀJ` `[k, k]` and gradient `Jᵀr` `[k]` of half
    /// the augmented χ², with `J = [−W ∂f/∂p; diag(1/σ)]`.
    fn normal(&self, at: &Point) -> (Vec<f64>, Vec<f64>) {
        let k = at.p.len();
        let mut jacobian = vec![0.; self.data.len() * k];
        self.model.jacobian(&at.p, &mut jacobian);
        let w = self.whitener.apply_columns(&jacobian, k);
        let (mut a, mut g) = (vec![0.; k * k], vec![0.; k]);
        for (row, r) in w.chunks_exact(k).zip(&at.whitened) {
            for i in 0..k {
                g[i] -= row[i] * r;
                for j in i..k {
                    a[i * k + j] += row[i] * row[j];
                }
            }
        }
        for i in 0..k {
            for j in 0..i {
                a[i * k + j] = a[j * k + i];
            }
            if let Prior::Gaussian { mean, sigma } = self.priors[i] {
                a[i * k + i] += 1. / (sigma * sigma);
                g[i] += (at.p[i] - mean) / (sigma * sigma);
            }
        }
        (a, g)
    }
    /// The point `p + δ` with `(A + λ diag A) δ = −g`, solved on the scaling
    /// `d A d` with `d` from `scales`; `None` when the system is singular or
    /// the augmented χ² would grow by more than the relative `slack`. A
    /// parameter of scale zero does not move.
    fn step(
        &self,
        at: &Point,
        a: &[f64],
        g: &[f64],
        d: &[f64],
        damping: f64,
        slack: f64,
    ) -> Option<Point> {
        let k = d.len();
        let mut lhs: Vec<f64> = (0..k * k).map(|c| a[c] * d[c / k] * d[c % k]).collect();
        let mut delta: Vec<f64> = (0..k).map(|j| -g[j] * d[j]).collect();
        for j in 0..k {
            lhs[j * k + j] += damping;
        }
        lu_solve_multi(&mut lhs, &mut delta, k, 1).ok()?;
        let next = self.point((0..k).map(|j| at.p[j] + d[j] * delta[j]).collect());
        (next.chi2().is_finite() && next.chi2() <= at.chi2() * (1. + slack)).then_some(next)
    }
}
/// `1/sqrt(A_jj)`, zero where the χ² does not depend on the parameter.
fn scales(a: &[f64], k: usize) -> Vec<f64> {
    (0..k)
        .map(|j| a[j * k + j])
        .map(|x| if x > 0. { 1. / x.sqrt() } else { 0. })
        .collect()
}
/// Largest `|g_j| / sqrt(A_jj)`: the residual norm times the largest cosine
/// between the residual and a column of the Jacobian.
fn cosine(a: &[f64], g: &[f64]) -> f64 {
    let d = scales(a, g.len());
    g.iter()
        .zip(&d)
        .map(|(g, d)| g.abs() * d)
        .fold(0., f64::max)
}

/// Minimize `|W (data − f(p))|² + Σ ((p_j − μ_j)/σ_j)²` from `start`.
/// `priors` has one entry per parameter. A model that produces nonfinite
/// values at `start`, or a nonfinite Jacobian at an accepted point, is
/// `GasError::Numerical`; failure to converge is reported in `converged`, not
/// as an error.
///
/// Steps solve `(A + λ diag A) δ = −g`; a step is kept when the augmented χ²
/// does not increase, then `λ` drops tenfold, otherwise it grows tenfold up to
/// `1e15`. The iteration converges when every column of the Jacobian is
/// orthogonal to the residual to `1e-8` in cosine, when the augmented χ² is
/// below `1e-24 |W data|²` (an exact fit, whose residual is roundoff and has
/// no direction), or when the relative decrease falls below `tolerance` with
/// `λ ≤ 1e-6`. A converged fit is refined by at most 20 undamped steps, each
/// kept only if it lowers the gradient without raising the χ² beyond its
/// roundoff. A Hessian that is singular at
/// the returned point (a parameter without data sensitivity and without
/// prior) is `GasError::Numerical`.
pub fn minimize(
    model: &dyn Model,
    data: &[f64],
    whitener: &Whitener,
    priors: &[Prior],
    start: &[f64],
    config: &LevenbergConfig,
) -> Result<NonlinearFit> {
    config.validate()?;
    let (m, k) = (model.outputs(), model.parameters());
    require(
        (1..=64).contains(&k)
            && (1..=4096).contains(&m)
            && whitener.n() == m
            && data.len() == m
            && priors.len() == k
            && start.len() == k
            && data.iter().chain(start).all(|x| x.is_finite()),
        "nonlinear fit needs 1..=64 parameters with one prior and one finite start each, \
         and 1..=4096 finite data points of the whitener's size",
    )?;
    for prior in priors {
        prior.validate()?;
    }
    let problem = Problem {
        model,
        data,
        whitener,
        priors,
    };
    let mut at = problem.point(start.to_vec());
    if !at.chi2().is_finite() {
        return Err(GasError::Numerical(
            "nonfinite model value at the start of the fit".into(),
        ));
    }
    let exact = 1e-24 * whitener.chi2(data);
    let mut damping = config.initial_damping;
    let (mut iterations, mut converged) = (0, false);
    while iterations < config.max_iterations {
        iterations += 1;
        let (a, g) = problem.normal(&at);
        if !a.iter().chain(&g).all(|x| x.is_finite()) {
            return Err(GasError::Numerical(
                "nonfinite Jacobian of the fitted model".into(),
            ));
        }
        let d = scales(&a, k);
        if cosine(&a, &g) <= 1e-8 * at.chi2().sqrt() || at.chi2() <= exact {
            converged = true;
            break;
        }
        let mut accepted = None;
        while accepted.is_none() && damping <= 1e15 {
            accepted = problem.step(&at, &a, &g, &d, damping, 0.);
            if accepted.is_none() {
                damping *= 10.;
            }
        }
        let Some(next) = accepted else { break };
        let decrease = (at.chi2() - next.chi2()) / at.chi2();
        at = next;
        damping = (damping / 10.).max(1e-15);
        if decrease < config.tolerance && damping <= 1e-6 {
            converged = true;
            break;
        }
    }
    let (mut a, mut g) = problem.normal(&at);
    let mut refinements = 0;
    while converged && refinements < 20 {
        refinements += 1;
        // Near the minimum a Gauss-Newton step changes the χ² by less than
        // its roundoff, so a decrease cannot be demanded of it; the gradient
        // decides, and it grows where the undamped iteration is unstable.
        let Some(next) = problem.step(&at, &a, &g, &scales(&a, k), 0., 1e-10) else {
            break;
        };
        let (a_next, g_next) = problem.normal(&next);
        let finite = a_next.iter().chain(&g_next).all(|x| x.is_finite());
        if !finite || cosine(&a_next, &g_next) >= cosine(&a, &g) {
            break;
        }
        let settled = next
            .p
            .iter()
            .zip(&at.p)
            .all(|(x, y)| (x - y).abs() <= 1e-14 * y.abs().max(1.));
        (at, a, g) = (next, a_next, g_next);
        if settled {
            break;
        }
    }
    let covariance = normal_inverse(&a, k)
        .map_err(|_| GasError::Numerical("singular Hessian of the fitted parameters".into()))?;
    let flat = priors.iter().filter(|x| matches!(x, Prior::Flat)).count();
    let dof = m.saturating_sub(flat);
    let chi2 = at.chi2();
    if !at.p.iter().chain(&covariance).all(|x| x.is_finite()) {
        return Err(GasError::Numerical("nonlinear fit overflow".into()));
    }
    Ok(NonlinearFit {
        p: at.p,
        covariance,
        chi2,
        chi2_prior: at.chi2_prior,
        dof,
        q: chi2_q(chi2, dof),
        iterations,
        converged,
    })
}
