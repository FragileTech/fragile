//! Heuristic curvature proxies read off the Voronoi cells. They are cheap
//! indicators rather than convergent estimators: positive curvature focuses
//! walkers, which shrinks, distorts and contracts their cells.
use super::{
    par::{Parallelism, map_indexed},
    voronoi::VoronoiCells,
};
use crate::{Real, Result, physics::geometry::symmetric_eigen, tessellation::linalg::compose};

pub struct ProxyField<T: Real> {
    pub scalar: Vec<T>,
    pub valid: Vec<bool>,
    /// Mean of the scalar over the valid cells.
    pub mean: T,
}
fn finish<T: Real>(scalar: Vec<T>, valid: Vec<bool>) -> ProxyField<T> {
    let mut sum = T::ZERO;
    let mut count = 0usize;
    for (v, _) in scalar.iter().zip(&valid).filter(|p| *p.1) {
        sum = sum + *v;
        count += 1;
    }
    ProxyField {
        mean: if count > 0 {
            sum / T::from_f64(count as f64)
        } else {
            T::ZERO
        },
        scalar,
        valid,
    }
}
fn usable<T: Real>(cells: &VoronoiCells<T>, i: usize) -> bool {
    cells.bounded[i] && cells.volume[i] > T::ZERO && cells.volume[i].is_finite()
}
/// 1 - V_i / <V>: positive where cells are smaller than average.
pub fn volume_distortion<T: Real>(cells: &VoronoiCells<T>) -> ProxyField<T> {
    let n = cells.walkers();
    let valid: Vec<bool> = (0..n).map(|i| usable(cells, i)).collect();
    let mut sum = T::ZERO;
    let mut count = 0usize;
    for i in (0..n).filter(|&i| valid[i]) {
        sum = sum + cells.volume[i];
        count += 1;
    }
    let mean = if count > 0 {
        sum / T::from_f64(count as f64)
    } else {
        T::ONE
    };
    let scalar = (0..n)
        .map(|i| {
            if valid[i] && mean > T::ZERO {
                T::ONE - cells.volume[i] / mean
            } else {
                T::ZERO
            }
        })
        .collect();
    finish(scalar, valid)
}
/// 1 - r_in / r_circ about the vertex centroid: zero for a round cell.
pub fn shape_distortion<T: Real>(cells: &VoronoiCells<T>, par: Parallelism) -> ProxyField<T> {
    let d = cells.dimension;
    let n = cells.walkers();
    let rows = map_indexed(n, par, |i| {
        let v = cells.cell_vertices(i);
        let count = v.len() / d.max(1);
        if !usable(cells, i) || count < d + 1 {
            return (T::ZERO, false);
        }
        let mut centroid = vec![T::ZERO; d];
        for p in v.chunks_exact(d) {
            for (c, &x) in centroid.iter_mut().zip(p) {
                *c = *c + x;
            }
        }
        for c in &mut centroid {
            *c = *c / T::from_f64(count as f64);
        }
        let mut inner = T::from_f64(f64::INFINITY);
        let mut outer = T::ZERO;
        for p in v.chunks_exact(d) {
            let r = p
                .iter()
                .zip(&centroid)
                .fold(T::ZERO, |s, (&a, &b)| s + (a - b) * (a - b))
                .sqrt();
            inner = inner.min(r);
            outer = outer.max(r);
        }
        if outer > T::from_f64(1e-10) {
            (T::ONE - inner / outer, true)
        } else {
            (T::ZERO, true)
        }
    });
    let (scalar, valid) = rows.into_iter().unzip();
    finish(scalar, valid)
}
/// Raychaudhuri proxy R_i = -theta_i with theta = (V - V_prev) / (dt V).
pub fn raychaudhuri<T: Real>(
    cells: &VoronoiCells<T>,
    previous: Option<&[T]>,
    dt: f64,
) -> ProxyField<T> {
    let n = cells.walkers();
    let Some(previous) = previous.filter(|p| p.len() == n) else {
        return finish(vec![T::ZERO; n], vec![false; n]);
    };
    let dt = T::from_f64(dt);
    let limit = T::from_f64(1e6);
    let mut valid = vec![false; n];
    let scalar = (0..n)
        .map(|i| {
            if !usable(cells, i) || !previous[i].is_finite() || previous[i] <= T::ZERO {
                return T::ZERO;
            }
            let theta = (cells.volume[i] - previous[i]) / (dt * cells.volume[i]);
            if theta.is_finite() && theta.abs() < limit {
                valid[i] = true;
                -theta
            } else {
                T::ZERO
            }
        })
        .collect();
    finish(scalar, valid)
}
/// Diffusion factors from the cell shape, `[walkers, d, d]`: along an axis of
/// elongation lambda the factor is c2 / sqrt(lambda + epsilon_sigma), so noise
/// is reduced along stretched directions. `diagonal` uses the axis-aligned
/// extents over V^{1/d}; otherwise the vertex covariance is diagonalized.
/// Open or degenerate cells get the isotropic value c2 / sqrt(1 + epsilon_sigma).
pub fn voronoi_diffusion<T: Real>(
    cells: &VoronoiCells<T>,
    epsilon_sigma: f64,
    c2: f64,
    diagonal: bool,
    par: Parallelism,
) -> Result<Vec<T>> {
    let d = cells.dimension;
    let eps = T::from_f64(epsilon_sigma);
    let c2 = T::from_f64(c2);
    let isotropic = c2 / (T::ONE + eps).sqrt();
    let rows = super::par::try_map_indexed(cells.walkers(), par, |i| {
        let mut out = vec![T::ZERO; d * d];
        let v = cells.cell_vertices(i);
        let count = v.len() / d.max(1);
        if !usable(cells, i) || count < d + 1 {
            for a in 0..d {
                out[a * d + a] = isotropic;
            }
            return Ok(out);
        }
        if diagonal {
            let scale = cells.volume[i].powf(T::from_f64(1. / d as f64));
            for a in 0..d {
                let (mut lo, mut hi) = (v[a], v[a]);
                for p in v.chunks_exact(d) {
                    lo = lo.min(p[a]);
                    hi = hi.max(p[a]);
                }
                out[a * d + a] = c2 / ((hi - lo) / scale + eps).sqrt();
            }
            return Ok(out);
        }
        let mut mean = vec![T::ZERO; d];
        for p in v.chunks_exact(d) {
            for (m, &x) in mean.iter_mut().zip(p) {
                *m = *m + x;
            }
        }
        for m in &mut mean {
            *m = *m / T::from_f64(count as f64);
        }
        let mut cov = vec![T::ZERO; d * d];
        for p in v.chunks_exact(d) {
            for a in 0..d {
                for b in a..d {
                    cov[a * d + b] = cov[a * d + b] + (p[a] - mean[a]) * (p[b] - mean[b]);
                }
            }
        }
        for a in 0..d {
            for b in a..d {
                let value = cov[a * d + b] / T::from_f64(count as f64);
                cov[a * d + b] = value;
                cov[b * d + a] = value;
            }
        }
        let (l, q) = symmetric_eigen(&cov, d)?;
        let sigma: Vec<T> = l
            .iter()
            .map(|&v| c2 / (v.max(T::ZERO) + eps).sqrt())
            .collect();
        Ok(compose(&q, &sigma, d))
    })?;
    Ok(rows.into_iter().flatten().collect())
}
