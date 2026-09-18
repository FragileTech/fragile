//! Scalar curvature of the conformal part of the metric.
//!
//! Writing det g = e^{2 d u}, the conformally flat metric e^{2u} delta has
//! R = -2 (d-1) e^{-2u} [Lap u + (d-2)/2 |grad u|^2]. The Laplacian estimator
//! replaces Lap u by a weighted graph Laplacian; the quadratic-fit estimator
//! recovers grad u and Hess u from a weighted local least-squares fit.
use super::{
    frame::GeometryFrame,
    linalg::lu_solve_multi,
    par::{Parallelism, map_indexed, try_map_indexed},
};
use crate::{Real, Result};

/// u = log(max(det g, floor)) / (2 d).
pub fn conformal_potential<T: Real>(determinant: &[T], dimension: usize, floor: f64) -> Vec<T> {
    let scale = T::from_f64(2. * dimension as f64);
    determinant
        .iter()
        .map(|&v| v.max(T::from_f64(floor)).ln() / scale)
        .collect()
}
/// R_i = -2 (d-1) sum_j w_ij (u_j - u_i).
pub fn conformal_laplacian<T: Real>(
    frame: &GeometryFrame<'_, T>,
    u: &[T],
    weights: &[T],
    par: Parallelism,
) -> Vec<T> {
    let factor = T::from_f64(-2. * (frame.dimension as f64 - 1.));
    map_indexed(frame.walkers(), par, |i| {
        let mut lap = T::ZERO;
        for e in frame.graph.range(i) {
            let j = frame.graph.neighbors()[e] as usize;
            lap = lap + weights[e] * (u[j] - u[i]);
        }
        factor * lap
    })
}
/// Weighted local quadratic fit of u around every walker: gradient `[N, d]`
/// and symmetric Hessian `[N, d, d]`, Tikhonov-regularized by `reg`.
pub fn fit_local_quadratic<T: Real>(
    frame: &GeometryFrame<'_, T>,
    u: &[T],
    weights: Option<&[T]>,
    reg: f64,
    par: Parallelism,
) -> Result<(Vec<T>, Vec<T>)> {
    let d = frame.dimension;
    let pairs: Vec<(usize, usize)> = (0..d).flat_map(|a| (a..d).map(move |b| (a, b))).collect();
    let p = d + pairs.len();
    let half = T::from_f64(0.5);
    let fits = try_map_indexed(frame.walkers(), par, |i| {
        let mut xtx = vec![T::ZERO; p * p];
        let mut xty = vec![T::ZERO; p];
        let mut phi = vec![T::ZERO; p];
        for e in frame.graph.range(i) {
            let j = frame.graph.neighbors()[e] as usize;
            let dx = frame.delta(i, j);
            phi[..d].copy_from_slice(&dx);
            for (k, &(a, b)) in pairs.iter().enumerate() {
                phi[d + k] = dx[a] * dx[b] * if a == b { half } else { T::ONE };
            }
            let w = weights.map_or(T::ONE, |w| w[e]);
            let y = u[j] - u[i];
            for a in 0..p {
                let wa = phi[a] * w;
                xty[a] = xty[a] + wa * y;
                for b in 0..p {
                    xtx[a * p + b] = xtx[a * p + b] + wa * phi[b];
                }
            }
        }
        for a in 0..p {
            xtx[a * p + a] = xtx[a * p + a] + T::from_f64(reg);
        }
        lu_solve_multi(&mut xtx, &mut xty, p, 1)?;
        Ok(xty)
    })?;
    let n = frame.walkers();
    let mut gradient = Vec::with_capacity(n * d);
    let mut hessian = vec![T::ZERO; n * d * d];
    for (i, beta) in fits.into_iter().enumerate() {
        gradient.extend_from_slice(&beta[..d]);
        for (k, &(a, b)) in pairs.iter().enumerate() {
            hessian[i * d * d + a * d + b] = beta[d + k];
            hessian[i * d * d + b * d + a] = beta[d + k];
        }
    }
    Ok((gradient, hessian))
}
/// Scalar curvature from the local quadratic fit and, on request, the
/// coordinate-basis Ricci tensor
/// R_ab = -(d-2)(u_ab - u_a u_b) - (Lap u + (d-2)|grad u|^2) delta_ab.
pub fn conformal_quadratic<T: Real>(
    frame: &GeometryFrame<'_, T>,
    u: &[T],
    weights: Option<&[T]>,
    reg: f64,
    conformal_factor: bool,
    tensor: bool,
    par: Parallelism,
) -> Result<(Vec<T>, Option<Vec<T>>)> {
    let d = frame.dimension;
    let df = d as f64;
    let (gradient, hessian) = fit_local_quadratic(frame, u, weights, reg, par)?;
    let n = frame.walkers();
    let mut scalar = Vec::with_capacity(n);
    let mut ricci = tensor.then(|| vec![T::ZERO; n * d * d]);
    for i in 0..n {
        let g = &gradient[i * d..(i + 1) * d];
        let h = &hessian[i * d * d..(i + 1) * d * d];
        let lap = (0..d).fold(T::ZERO, |s, a| s + h[a * d + a]);
        let norm = g.iter().fold(T::ZERO, |s, &v| s + v * v);
        let scale = if conformal_factor {
            (T::from_f64(-2.) * u[i]).exp()
        } else {
            T::ONE
        };
        scalar.push(
            T::from_f64(-2. * (df - 1.)) * scale * (lap + T::from_f64(0.5 * (df - 2.)) * norm),
        );
        if let Some(r) = &mut ricci {
            let trace = lap + T::from_f64(df - 2.) * norm;
            for a in 0..d {
                for b in 0..d {
                    let v = T::from_f64(-(df - 2.)) * (h[a * d + b] - g[a] * g[b]);
                    r[i * d * d + a * d + b] = if a == b { v - trace } else { v };
                }
            }
        }
    }
    Ok((scalar, ricci))
}
