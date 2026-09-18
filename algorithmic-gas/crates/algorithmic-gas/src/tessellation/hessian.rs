//! Hessian of a scalar walker field estimated on the neighbor graph.
//!
//! The diagonal estimator is a second-order central difference along every
//! coordinate axis, using the closest neighbor inside a 30 degree cone on each
//! side. The full estimator is the quadratic part of a local least-squares fit.
//! Axes without a neighbor on both sides contribute zero.
use super::{
    frame::GeometryFrame,
    par::{Parallelism, map_indexed},
    ricci::fit_local_quadratic,
};
use crate::{Real, Result};

const MAX_ANGLE_DEGREES: f64 = 30.;

/// `[walkers, d, d]` Hessian estimates of `values`.
pub fn estimate<T: Real>(
    frame: &GeometryFrame<'_, T>,
    values: &[T],
    full: bool,
    par: Parallelism,
) -> Result<Vec<T>> {
    let d = frame.dimension;
    if full {
        return Ok(fit_local_quadratic(frame, values, None, 1e-6, par)?.1);
    }
    let cos_threshold = T::from_f64(MAX_ANGLE_DEGREES.to_radians().cos());
    let tiny = T::from_f64(1e-10);
    let rows = map_indexed(frame.walkers(), par, |i| {
        let mut h = vec![T::ZERO; d * d];
        // Closest aligned neighbor on the positive and negative side per axis.
        let mut side: Vec<[Option<(T, usize)>; 2]> = vec![[None, None]; d];
        for &j in frame.graph.row(i) {
            let dx = frame.delta(i, j as usize);
            let norm = dx.iter().fold(T::ZERO, |s, &v| s + v * v).sqrt() + tiny;
            for a in 0..d {
                if (dx[a] / norm).abs() < cos_threshold {
                    continue;
                }
                let slot = &mut side[a][usize::from(dx[a] < T::ZERO)];
                if slot.is_none_or(|(best, _)| dx[a].abs() < best) {
                    *slot = Some((dx[a].abs(), j as usize));
                }
            }
        }
        for a in 0..d {
            if let [Some((hp, jp)), Some((hn, jn))] = side[a] {
                let step = T::from_f64(0.5) * (hp + hn);
                h[a * d + a] =
                    (values[jp] + values[jn] - T::from_f64(2.) * values[i]) / (step * step + tiny);
            }
        }
        h
    });
    Ok(rows.into_iter().flatten().collect())
}
